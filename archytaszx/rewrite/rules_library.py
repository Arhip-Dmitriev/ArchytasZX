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

"""Concrete rewrite rules, starting with spider fusion, each recording its exact scalar.

Phase 5 registers one rule, :data:`SPIDER_FUSION`. :data:`RULES` and :func:`lookup_rule`
resolve a :class:`~qufzx.rewrite.engine.RewriteStep`'s ``rule_name`` back to its
:class:`~qufzx.rewrite.rule.Rule`, keeping :mod:`qufzx.rewrite.engine` generic.

Scalar. Same-color fusion across one wire introduces no factor in either wire shape
condition 4 (``consumed_wire_direction_permitted_for_color``) permits, so both land on
:meth:`~qufzx.algebra.scalar.Scalar.one`:

* Alternating output-to-input, either color. Z: both spiders are diagonal with entry
  ``e^{i*angle(k)}`` at the all-axes-``k`` position, so contracting an output leg against
  an input leg identifies their ``k``. X: ``X_{m->n} = F^{ox n} . Z_{m->n} .
  (conj(F))^{ox m}``, and the wire contracts an ``F`` against a ``conj(F)`` on the shared
  axis, which cancel to the identity, ``F`` being unitary and symmetric.
* Same-direction, Z only. ``_z_tensor`` is diagonal in every axis and
  :mod:`qufzx.semantics.contract_numeric` applies no conjugation at contraction time, so
  the same index ``k`` is identified. A same-direction X wire contracts ``F`` against ``F``,
  giving a permutation matrix, and is not this rule.

Neither derivation depends on the consumed wire being the pair's only wire: a further wire
between the same nodes is never contracted by this rule.

Merged leg ordering, a choice rather than a derivation: the merged node's inputs are A's
surviving inputs in original index order, then B's; outputs likewise. "A" is the lower
:class:`~qufzx.diagram.graph.NodeId`.

Merged dimension. Every surviving port is built at
:attr:`~qufzx.rewrite.match.FusionMatch.shared_dim`, never its own original ``Dim``.
Condition 6 unifies every surviving leg against the resolved ``shared_dim`` before a match
is returned, and this builder calls the same
:func:`~qufzx.rewrite.match.resolve_fusion_match` fresh against the diagram it was handed.

Fusion may fire on a ``DEFERRED`` dimension pair, though FULL_PLAN.md's Phase 5 states the
pattern as spiders "sharing a dimension". A ``d``/``d*e`` leg pair is already legal,
non-hard-error input under ``ALL_LEGS_EQUAL``
(:class:`~qufzx.diagram.validate.IssueKind.DIMENSION_DEFERRED`), and the assumption is
recorded either way. Two consequences: a neighbouring wire that was an exact match before
the fusion may be merely deferred after, which :mod:`qufzx.rewrite.engine`'s step-8
relative postcondition permits; and a surviving leg on a boundary is rebuilt at
``shared_dim``, so the finished diagram's interface holds only under the same recorded
assumption.

Recorded is not satisfiable. A surviving leg of ``d**2`` forced onto ``shared_dim = d`` is
a legal ``DEFERRED`` unify recording ``d**2 == d``, which holds over the positive integers
only at ``d = 1``. Discharging such a constraint is Phase 10's job.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseDomainError, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import FOURIER_BOX, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef
from qufzx.rewrite.match import (
    CAP_SIDE_CONDITIONS,
    FOURIER_SIDE_CONDITIONS,
    FUSION_SIDE_CONDITIONS,
    CapMatch,
    CapPattern,
    FourierCancellationPattern,
    FourierMatch,
    FusionMatch,
    FusionPattern,
    find_cap_matches,
    find_fourier_matches,
    innermost_node_scope_box,
    reattach_phase,
    resolve_fusion_match,
)
from qufzx.rewrite.rule import (
    BuildResult,
    Match,
    Quantifiers,
    RewriteDomainError,
    RewriteGrammarError,
    Rule,
    check_side_condition_coverage,
)


def _surviving_legs(
    node_id: NodeId, node: Node, direction: Direction, consumed_ref: PortRef
) -> list[tuple[PortRef, Port]]:
    """Every ``(PortRef, Port)`` of ``node`` on ``direction`` except ``consumed_ref``, in order."""
    legs = node.legs(direction)
    surviving = []
    for index, port in enumerate(legs):
        ref = PortRef(node_id, direction, index)
        if ref == consumed_ref:
            continue
        surviving.append((ref, port))
    return surviving


def _over_shared_dim(
    phase: PhaseVector | None, shared_dim: Dim, bindings: Mapping[str, Dim]
) -> tuple[PhaseVector, Mapping[str, Dim]]:
    """``phase``'s entries, with ``bindings`` substituted in, reattached to ``shared_dim``
    -- or an all-zero vector over ``shared_dim`` if ``phase`` is absent.

    Delegates to :func:`qufzx.rewrite.match.reattach_phase`, returning its vector and the
    subset of ``bindings`` it substituted into an entry's value. Raises
    :class:`RewriteDomainError`, not :class:`~qufzx.algebra.phase.PhaseDomainError`, if an
    entry index falls outside ``shared_dim``'s range.
    """
    if phase is None:
        return PhaseVector(shared_dim, {}), MappingProxyType({})
    try:
        return reattach_phase(phase, shared_dim, bindings)
    except PhaseDomainError as exc:
        raise RewriteDomainError(
            f"spider_fusion cannot reattach a phase vector to shared dimension {shared_dim}: {exc}"
        ) from exc


def _merged_phase(
    node_a: Node,
    node_b: Node,
    a_id: NodeId,
    b_id: NodeId,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
    *,
    any_legs_survive: bool,
) -> tuple[PhaseVector | None, Mapping[NodeId, Mapping[str, Dim]]]:
    """The merged node's phase: componentwise sum, both operands read over ``shared_dim``.

    A `None` phase on both sides stays `None`, except when ``any_legs_survive`` is `False`: a
    merged node with no surviving legs has no port to carry ``shared_dim``, so an all-zero
    ``PhaseVector(shared_dim, {})`` is returned instead. Otherwise both operands are read via
    :func:`_over_shared_dim` before adding, since
    :meth:`~qufzx.algebra.phase.PhaseVector.__add__` demands exactly equal ``Dim``\\ s.

    The second return value is every node whose phase had a binding substituted into an
    entry, keyed by node id.
    """
    if node_a.phase is None and node_b.phase is None:
        if not any_legs_survive:
            return PhaseVector(shared_dim, {}), {}
        return None, {}
    phase_a, applied_a = _over_shared_dim(node_a.phase, shared_dim, bindings)
    phase_b, applied_b = _over_shared_dim(node_b.phase, shared_dim, bindings)
    substitutions: dict[NodeId, Mapping[str, Dim]] = {}
    if applied_a:
        substitutions[a_id] = applied_a
    if applied_b:
        substitutions[b_id] = applied_b
    return phase_a + phase_b, substitutions


def spider_fusion_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`SPIDER_FUSION`: merge the two matched spiders.

    Mutates ``diagram`` in place by adding the merged node (module docstring: leg ordering,
    scalar, merged dimension) and returns the :class:`BuildResult`
    :mod:`qufzx.rewrite.engine` needs to splice it in; never removes the matched nodes or
    touches any wire or boundary entry.

    Trusts nothing about ``match`` for graph surgery until it has been re-derived. In order:

    1. ``isinstance(match, FusionMatch)``.
    2. :func:`~qufzx.rewrite.rule.check_side_condition_coverage` against the module-level
       :data:`FUSION_SIDE_CONDITIONS`; this builder is reachable directly.
    3. :func:`~qufzx.rewrite.match.resolve_fusion_match`, called fresh against ``diagram``.
       Everything downstream builds from ``resolution``'s fields, never ``match``'s.
    4. ``match.shared_dim``, ``bindings``, ``dimension_constraints`` and
       ``side_condition_outcomes``, each checked for exact agreement with ``resolution``'s.
       ``apply`` records the match's own copies, so this equality is what makes them the
       certificate's ground truth.

    Raises :class:`~qufzx.rewrite.rule.RewriteDomainError` on any disagreement at step 3 or
    4, and :class:`~qufzx.rewrite.rule.RewriteGrammarError` for a foreign match type or a
    structurally malformed one (equal node ids, a node absent from ``diagram``, a wire not
    incident on both or not in ``diagram.wires``).
    """
    if not isinstance(match, FusionMatch):
        raise RewriteGrammarError(
            f"spider_fusion requires a FusionMatch, got {type(match).__name__}"
        )
    # The module-level constant, not spider_fusion_builder.side_conditions, which would be a
    # self-reference to this function's own global name. That attribute exists solely for
    # Rule.__post_init__.
    check_side_condition_coverage(match, FUSION_SIDE_CONDITIONS, "spider_fusion")

    resolution = resolve_fusion_match(diagram, match.a_id, match.b_id, match.wire)
    if not resolution.passed:
        failed = [outcome.name for outcome in resolution.outcomes if not outcome.passed]
        raise RewriteDomainError(
            f"spider_fusion: match at ({match.a_id!r}, {match.b_id!r}) over wire "
            f"{match.wire!r} fails side condition(s) {failed} when re-verified fresh "
            "against the diagram it is being applied to; match.side_condition_outcomes "
            "claimed every condition passed, but a match's own outcomes are never taken "
            "on faith for graph surgery -- see resolve_fusion_match"
        )
    assert resolution.shared_dim is not None  # invariant: passed implies shared_dim is set
    if match.shared_dim != resolution.shared_dim:
        raise RewriteDomainError(
            f"spider_fusion: match.shared_dim {match.shared_dim!r} disagrees with the "
            f"shared dimension {resolution.shared_dim!r} resolve_fusion_match derives "
            "fresh from the diagram for this wire; a match's own shared_dim is never "
            "trusted for graph surgery without this agreement"
        )
    if dict(match.bindings) != dict(resolution.bindings):
        raise RewriteDomainError(
            f"spider_fusion: match.bindings {dict(match.bindings)!r} disagrees with the "
            f"bindings {dict(resolution.bindings)!r} resolve_fusion_match derives fresh "
            "from the diagram for this wire"
        )
    # apply records these two fields verbatim onto RewriteStep, so a fabricated pair must be
    # rejected here or the certificate records a claim the rewrite never made.
    if match.dimension_constraints != resolution.dimension_constraints:
        raise RewriteDomainError(
            f"spider_fusion: match.dimension_constraints {match.dimension_constraints!r} "
            f"disagrees with {resolution.dimension_constraints!r}, which "
            "resolve_fusion_match derives fresh from the diagram for this wire -- a "
            "match's own dimension_constraints is never trusted for the certificate "
            "without this agreement"
        )
    if match.side_condition_outcomes != resolution.outcomes:
        raise RewriteDomainError(
            "spider_fusion: match.side_condition_outcomes disagrees with the outcomes "
            "resolve_fusion_match derives fresh from the diagram for this wire -- a "
            "match's own side_condition_outcomes is never trusted for the certificate "
            "without this agreement"
        )

    node_a = diagram.nodes[match.a_id]
    node_b = diagram.nodes[match.b_id]
    wire = match.wire
    consumed_ref_a = wire.a if wire.a.node_id == match.a_id else wire.b
    consumed_ref_b = wire.b if wire.a.node_id == match.a_id else wire.a

    surviving_inputs_a = _surviving_legs(match.a_id, node_a, Direction.INPUT, consumed_ref_a)
    surviving_outputs_a = _surviving_legs(match.a_id, node_a, Direction.OUTPUT, consumed_ref_a)
    surviving_inputs_b = _surviving_legs(match.b_id, node_b, Direction.INPUT, consumed_ref_b)
    surviving_outputs_b = _surviving_legs(match.b_id, node_b, Direction.OUTPUT, consumed_ref_b)

    merged_inputs = surviving_inputs_a + surviving_inputs_b
    merged_outputs = surviving_outputs_a + surviving_outputs_b

    any_legs_survive = bool(merged_inputs or merged_outputs)
    merged_phase, phase_substitutions = _merged_phase(
        node_a,
        node_b,
        match.a_id,
        match.b_id,
        resolution.shared_dim,
        resolution.bindings,
        any_legs_survive=any_legs_survive,
    )

    # node_a.generator_type alone: resolution.passed confirmed it equals node_b's.
    new_node_id = diagram.add_node(
        node_a.generator_type,
        input_dims=[resolution.shared_dim] * len(merged_inputs),
        output_dims=[resolution.shared_dim] * len(merged_outputs),
        phase=merged_phase,
    )

    port_mapping: dict[PortRef, PortRef] = {}
    for new_index, (old_ref, _) in enumerate(merged_inputs):
        port_mapping[old_ref] = PortRef(new_node_id, Direction.INPUT, new_index)
    for new_index, (old_ref, _) in enumerate(merged_outputs):
        port_mapping[old_ref] = PortRef(new_node_id, Direction.OUTPUT, new_index)

    # Bang box "left intact" (Phase 7): condition 6 already required both matched nodes
    # to share one innermost node-scope box, or neither to have one. If they do, that
    # box's scope drops the two consumed nodes and gains the merged one, in place --
    # multiplicity, and every other field, untouched, since this fusion is a single
    # symbolic rewrite standing for one fusion per future instantiated copy.
    enclosing_box = innermost_node_scope_box(diagram, match.a_id)
    if enclosing_box is not None:
        box = diagram.bang_boxes[enclosing_box]
        new_scope = (box.node_scope - {match.a_id, match.b_id}) | {new_node_id}
        diagram.set_bang_box_node_scope(enclosing_box, frozenset(new_scope))

    # A port-scope box names a leg of one of the merged nodes; that leg survives on the
    # merged node, so the box follows it through the same port_mapping the boundary does.
    # A consumed port is never boxed: a port-scope box's port must be a boundary slot, and
    # condition 5 already required neither consumed port to be one.
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if not any(ref in port_mapping for ref in box.port_scope):
            continue
        diagram.set_bang_box_port_scope(
            box_id, frozenset(port_mapping.get(ref, ref) for ref in box.port_scope)
        )

    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_node_id,),
        consumed_node_ids=(match.a_id, match.b_id),
        consumed_wires=(wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
        phase_substitutions=MappingProxyType(phase_substitutions),
    )


spider_fusion_builder.side_conditions = FUSION_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The single declared side-condition tuple this builder is meant to be paired with.

Read only by :class:`~qufzx.rewrite.rule.Rule`'s constructor-time consistency check, which
makes two contradicting tuples for one builder impossible to construct. The builder's own
body reads the module-level constant.
"""


SPIDER_FUSION = Rule(
    name="spider_fusion",
    pattern=FusionPattern(),
    builder=spider_fusion_builder,
    side_conditions=FUSION_SIDE_CONDITIONS,
    quantifiers=Quantifiers(
        leg_counts=("m_a", "n_a", "m_b", "n_b"),
        dimensions=("d",),
    ),
    scalar_introduced=Scalar.one(),
)
"""Same-color spider fusion across one wire -- any direction for Z, output-to-input only for X.

Any further wire joining the same pair is not consumed: it survives as a self-loop on the
merged spider (condition 3 in :mod:`qufzx.rewrite.match`).
"""


def lookup_rule(name: str) -> Rule:
    """Resolve a rule name back to its :class:`Rule`.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` if ``name`` is not in
    :data:`RULES`.
    """
    try:
        return RULES[name]
    except KeyError:
        raise RewriteGrammarError(f"no such rule: {name!r}") from None


def fourier_cancellation_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`FOURIER_CANCELLATION`: one identity spider for four F boxes.

    Adds a phaseless one-in-one-out Z spider (the identity on a wire) and maps the chain's
    free input and output onto its legs; never removes the matched nodes or touches a wire.

    Trusts nothing about ``match`` for graph surgery until it has been re-derived: the match
    must be among those :func:`~qufzx.rewrite.match.find_fourier_matches` finds afresh in
    ``diagram``, which settles ``node_ids``, ``wires``, ``shared_dim``,
    ``dimension_constraints`` and ``side_condition_outcomes`` in one equality. This builder
    is reachable directly, so a fabricated match must not reach surgery.
    """
    if not isinstance(match, FourierMatch):
        raise RewriteGrammarError(
            f"fourier_cancellation_builder requires a FourierMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, FOURIER_SIDE_CONDITIONS, "fourier_cancellation_builder")
    if match not in find_fourier_matches(diagram):
        raise RewriteDomainError(
            "fourier_cancellation_builder: the match is not among those rediscovered in this "
            "diagram, so it is not evidence of an F^4 chain"
        )
    for node_id in match.node_ids:
        node = diagram.nodes.get(node_id)
        if node is None or node.generator_type.name != FOURIER_BOX.name:
            raise RewriteGrammarError(
                f"fourier_cancellation_builder: node {node_id!r} is not an F box in this diagram"
            )
    dim = match.shared_dim
    enclosing_boxes = {innermost_node_scope_box(diagram, node_id) for node_id in match.node_ids}
    if len(enclosing_boxes) != 1:
        found = sorted(box for box in enclosing_boxes if box is not None)
        raise RewriteDomainError(
            "fourier_cancellation_builder: the four F boxes do not share one innermost "
            f"node-scope bang box (found {found!r}); the chain must be wholly inside one "
            "box or wholly outside every box"
        )
    new_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    first, last = match.node_ids[0], match.node_ids[-1]
    port_mapping = {
        PortRef(first, Direction.INPUT, 0): PortRef(new_id, Direction.INPUT, 0),
        PortRef(last, Direction.OUTPUT, 0): PortRef(new_id, Direction.OUTPUT, 0),
    }

    # Bang box "left intact" (Phase 7), as in spider_fusion_builder: the enclosing box drops
    # the four consumed nodes and gains the identity spider, every other field untouched.
    (enclosing_box,) = enclosing_boxes
    if enclosing_box is not None:
        box = diagram.bang_boxes[enclosing_box]
        new_scope = (box.node_scope - set(match.node_ids)) | {new_id}
        diagram.set_bang_box_node_scope(enclosing_box, frozenset(new_scope))

    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_id,),
        consumed_node_ids=tuple(match.node_ids),
        consumed_wires=tuple(match.wires),
        port_mapping=port_mapping,
        scalar_introduced=FOURIER_CANCELLATION_SCALAR,
    )


FOURIER_CANCELLATION_SCALAR = Scalar.one()
"""The exact scalar F^4 cancellation introduces: one.

The character sum contributes a factor of ``d`` twice and the four boxes contribute
``d^{-1/2}`` each, so the product is exactly one. An implementation that drops either
factor lands on ``d``, ``d^{-1}`` or ``d^{-2}`` instead, which is a wrong global factor.
"""


FOURIER_CANCELLATION = Rule(
    name="fourier_cancellation",
    pattern=FourierCancellationPattern(),
    builder=fourier_cancellation_builder,
    side_conditions=FOURIER_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=FOURIER_CANCELLATION_SCALAR,
)
"""Four Fourier boxes in series collapse to the identity wire, introducing exactly one."""


def zx_cap_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`ZX_CAP`: nothing, with ``d ** (1/2)`` recorded.

    Removes both nodes and the wire joining them, leaving whatever else the diagram holds.

    Trusts nothing about ``match`` for graph surgery until it has been re-derived: the match
    must be among those :func:`~qufzx.rewrite.match.find_cap_matches` finds afresh in
    ``diagram``.
    """
    if not isinstance(match, CapMatch):
        raise RewriteGrammarError(f"zx_cap_builder requires a CapMatch, got {type(match).__name__}")
    check_side_condition_coverage(match, CAP_SIDE_CONDITIONS, "zx_cap_builder")
    if match not in find_cap_matches(diagram):
        raise RewriteDomainError(
            "zx_cap_builder: the match is not among those rediscovered in this diagram, so "
            "it is not evidence of a Z state wired into an X effect"
        )
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=(match.state_id, match.effect_id),
        consumed_wires=(match.wire,),
        port_mapping={},
        scalar_introduced=zx_cap_scalar(match.shared_dim),
    )


def zx_cap_scalar(dim: Dim) -> Scalar:
    """The exact scalar the cap introduces at ``dim``: ``dim ** (1/2)``."""
    return Scalar.dim_power(dim, 1, 2)


ZX_CAP = Rule(
    name="zx_cap",
    pattern=CapPattern(),
    builder=zx_cap_builder,
    side_conditions=CAP_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=zx_cap_scalar(Dim.symbol("d")),
    scalar_in_dim=zx_cap_scalar,
)
"""A phaseless Z state capped by a phaseless X effect is the empty diagram times ``d ** (1/2)``.

The X effect reads ``sum_j conj(omega_d^{j k}) / sqrt(d)``, which the character sum closes to
``sqrt(d) * [k == 0 mod d]``; summing that against the Z state's all-ones vector leaves exactly
``sqrt(d)``. Unlike the other two rules, the scalar this introduces is not one, so a dropped
factor changes the answer.
"""


RULES: Mapping[str, Rule] = MappingProxyType(
    {
        SPIDER_FUSION.name: SPIDER_FUSION,
        FOURIER_CANCELLATION.name: FOURIER_CANCELLATION,
        ZX_CAP.name: ZX_CAP,
    }
)
"""Every rule this module registers, keyed by :attr:`~qufzx.rewrite.rule.Rule.name`.

A ``MappingProxyType``, so a caller cannot mutate the registry through it.
"""
