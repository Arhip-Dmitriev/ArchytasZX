# Copyright 2026 Arkhip A. Dmitriev
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normal-form driver: reduce a diagram to the fixpoint of the normal-form rule set.

:func:`normal_form` runs :func:`~archytaszx.rewrite.engine.apply_until_fixpoint` over
:func:`~archytaszx.rewrite.engine.normal_form_rules` (or the rules given), records every
:class:`~archytaszx.rewrite.engine.RewriteResult`, and keys the result by the
:func:`~archytaszx.diagram.compare.canonical_key` of its :func:`comparison_view`.

:attr:`NormalForm.diagram` is the exact strategy output, so the recorded results replay onto
it id for id. :func:`comparison_view` sorts each Z and X spider's legs per direction, clears
the parameter environment and replaces the scalar by its
:meth:`~archytaszx.algebra.scalar.Scalar.simplify` form; two normal forms are the same when
their views share a key and are :func:`~archytaszx.diagram.compare.isomorphic`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from archytaszx.algebra.scalar import Scalar, ScalarError
from archytaszx.diagram.compare import canonical_key, isomorphic
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.cache import RewriteCache
from archytaszx.rewrite.engine import (
    DEFAULT_GUARD,
    RewriteResult,
    StopReason,
    StrategyOutcome,
    TerminationGuard,
    apply_until_fixpoint,
    normal_form_rules,
)
from archytaszx.rewrite.rule import (
    ConstraintOutcome,
    DimensionConstraint,
    RewriteGrammarError,
    Rule,
)


@dataclass(frozen=True, slots=True, eq=False)
class NormalForm:
    """A diagram's reduction: the source, the reduced diagram, its provenance and its key."""

    source: Diagram
    diagram: Diagram
    outcome: StrategyOutcome
    results: tuple[RewriteResult, ...]
    key: str

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        if not isinstance(self.source, Diagram):
            raise RewriteGrammarError(
                f"NormalForm.source must be a Diagram, got {type(self.source).__name__}"
            )
        if not isinstance(self.diagram, Diagram):
            raise RewriteGrammarError(
                f"NormalForm.diagram must be a Diagram, got {type(self.diagram).__name__}"
            )
        if not isinstance(self.outcome, StrategyOutcome):
            raise RewriteGrammarError(
                f"NormalForm.outcome must be a StrategyOutcome, got {type(self.outcome).__name__}"
            )
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, RewriteResult) for result in self.results
        ):
            raise RewriteGrammarError("NormalForm.results must be a tuple of RewriteResult")
        if not isinstance(self.key, str):
            raise RewriteGrammarError(
                f"NormalForm.key must be a str, got {type(self.key).__name__}"
            )

    @property
    def reached_fixpoint(self) -> bool:
        """True when the strategy stopped at a fixpoint rather than a guard."""
        return self.outcome.stop_reason is StopReason.FIXPOINT

    @property
    def assumed_constraints(self) -> tuple[DimensionConstraint, ...]:
        """Every ``DEFERRED`` dimension constraint across the steps, in step order, deduplicated."""
        seen: set[DimensionConstraint] = set()
        collected: list[DimensionConstraint] = []
        for step in self.outcome.steps:
            for constraint in step.dimension_constraints:
                if constraint.outcome is not ConstraintOutcome.DEFERRED or constraint in seen:
                    continue
                seen.add(constraint)
                collected.append(constraint)
        return tuple(collected)


def _require_diagram(what: str, value: object) -> Diagram:
    """``value`` itself when it is a Diagram, else a RewriteGrammarError naming ``what``."""
    if not isinstance(value, Diagram):
        raise RewriteGrammarError(f"{what} must be a Diagram, got {type(value).__name__}")
    return value


def _simplified(scalar: Scalar) -> Scalar:
    """``scalar.simplify()``, or ``scalar`` unchanged when simplification raises."""
    try:
        return scalar.simplify()
    except ScalarError:
        return scalar


def _leg_key(diagram: Diagram, ref: PortRef, partner: PortRef | None) -> tuple[object, ...]:
    """A node-id-free sort key for one spider leg: boundary position, else its partner's shape."""
    dim = str(diagram.nodes[ref.node_id].legs(ref.direction)[ref.index].dim)
    for side, refs in ((0, diagram.boundary_inputs), (1, diagram.boundary_outputs)):
        if ref in refs:
            return (0, side, refs.index(ref), dim)
    if partner is None:
        return (2, dim, ref.index)
    node = diagram.nodes[partner.node_id]
    spider = node.generator_type in (Z_SPIDER, X_SPIDER)
    return (
        1,
        dim,
        node.generator_type.name,
        partner.direction.value,
        -1 if spider else partner.index,
        node.num_inputs,
        node.num_outputs,
        repr(node.phase),
        ref.index,
    )


def _canonical_legs(diagram: Diagram) -> Diagram:
    """A renumbered copy of ``diagram`` with each Z and X spider's legs sorted per direction by
    :func:`_leg_key`; ``diagram`` itself when it carries a bang box."""
    if diagram.bang_boxes:
        return diagram
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a] = wire.b
        partner[wire.b] = wire.a
    order: dict[tuple[NodeId, Direction], list[PortRef]] = {}
    new_index: dict[PortRef, int] = {}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        for direction in (Direction.INPUT, Direction.OUTPUT):
            refs = [PortRef(node_id, direction, i) for i in range(len(node.legs(direction)))]
            if node.generator_type in (Z_SPIDER, X_SPIDER):
                refs.sort(key=lambda ref: _leg_key(diagram, ref, partner.get(ref)))
            order[(node_id, direction)] = refs
            for position, ref in enumerate(refs):
                new_index[ref] = position
    view = Diagram()
    new_id: dict[NodeId, NodeId] = {}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        new_id[node_id] = view.add_node(
            node.generator_type,
            [node.inputs[ref.index].dim for ref in order[(node_id, Direction.INPUT)]],
            [node.outputs[ref.index].dim for ref in order[(node_id, Direction.OUTPUT)]],
            phase=node.phase,
        )

    def moved(ref: PortRef) -> PortRef:
        return PortRef(new_id[ref.node_id], ref.direction, new_index[ref])

    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        view.add_wire(moved(wire.a), moved(wire.b))
    view.set_boundary_inputs([moved(ref) for ref in diagram.boundary_inputs])
    view.set_boundary_outputs([moved(ref) for ref in diagram.boundary_outputs])
    view.set_scalar(diagram.scalar)
    view.set_parameters(diagram.parameters)
    return view


def comparison_view(diagram: Diagram) -> Diagram:
    """A copy of ``diagram`` with Z and X spider legs in canonical order, an empty parameter
    environment and a simplified scalar."""
    _require_diagram("comparison_view: diagram", diagram)
    view = _canonical_legs(diagram).copy()
    view.set_parameters({})
    view.set_scalar(_simplified(view.scalar))
    return view


def _graph_view(diagram: Diagram) -> Diagram:
    """A copy of ``diagram`` with an empty parameter environment and scalar one."""
    view = diagram.copy()
    view.set_parameters({})
    view.set_scalar(Scalar.one())
    return view


def normal_form(
    diagram: Diagram,
    *,
    rules: Sequence[Rule] | None = None,
    guard: TerminationGuard = DEFAULT_GUARD,
    cache: RewriteCache | None = None,
) -> NormalForm:
    """Reduce ``diagram`` under ``rules`` (default :func:`normal_form_rules`) to a fixpoint."""
    _require_diagram("normal_form: diagram", diagram)
    if rules is None:
        rule_set: tuple[Rule, ...] = normal_form_rules()
    else:
        if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
            raise RewriteGrammarError(
                f"normal_form: rules must be a Sequence of Rule, got {type(rules).__name__}"
            )
        if not all(isinstance(rule, Rule) for rule in rules):
            raise RewriteGrammarError("normal_form: every element of rules must be a Rule")
        rule_set = tuple(rules)
    source = diagram.copy()
    results: list[RewriteResult] = []
    outcome = apply_until_fixpoint(
        source.copy(), rule_set, guard=guard, cache=cache, on_result=results.append
    )
    return NormalForm(
        source=source,
        diagram=outcome.diagram,
        outcome=outcome,
        results=tuple(results),
        key=canonical_key(comparison_view(outcome.diagram)),
    )


def same_normal_form(a: NormalForm, b: NormalForm) -> bool:
    """Whether ``a`` and ``b`` share a key and their comparison views are isomorphic."""
    if not isinstance(a, NormalForm) or not isinstance(b, NormalForm):
        raise RewriteGrammarError(
            f"same_normal_form requires two NormalForm instances, got {type(a).__name__} "
            f"and {type(b).__name__}"
        )
    if a.key != b.key:
        return False
    return isomorphic(comparison_view(a.diagram), comparison_view(b.diagram))


def isomorphic_up_to_scalar(a: Diagram, b: Diagram) -> bool:
    """Whether ``a`` and ``b`` are isomorphic once scalar and parameters are ignored."""
    _require_diagram("isomorphic_up_to_scalar: a", a)
    _require_diagram("isomorphic_up_to_scalar: b", b)
    return isomorphic(_graph_view(a), _graph_view(b))
