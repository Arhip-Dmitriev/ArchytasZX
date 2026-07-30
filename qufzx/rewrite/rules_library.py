"""Concrete rewrite rules, starting with spider fusion, each recording its exact scalar.

Phase 5 registers exactly one rule, :data:`SPIDER_FUSION`, built from
:class:`~qufzx.rewrite.match.FusionPattern` and :func:`spider_fusion_builder` below.

Rule registry. A :class:`~qufzx.rewrite.engine.RewriteStep` records only a rule's
``name`` (a plain string), not the :class:`~qufzx.rewrite.rule.Rule` object itself, so
Phase 6's certificate replay needs a way to resolve that name back to the actual rule it
names before it can re-apply anything. :data:`RULES` and :func:`lookup_rule` are that
resolution path: every rule this module defines is registered there, keyed by
:attr:`~qufzx.rewrite.rule.Rule.name`, so :mod:`qufzx.rewrite.engine` (which must stay
generic over any future rule, per its own module docstring) never needs to import a
specific rule module itself.

Scalar derivation (not assertion). Same-color, single-wire, output-to-input fusion
introduces no scalar factor at all -- the merged node's denotation, read directly off
:mod:`qufzx.semantics.denote`'s formulas, already equals the pre-fusion diagram's
contraction with no leftover coefficient:

* Z spider. Both ``Z_a`` and ``Z_b`` are diagonal in the computational basis: entry
  ``e^{i*angle_a(k)}`` (resp. ``e^{i*angle_b(k)}``) sits at the all-axes-equal-``k``
  position, zero elsewhere. Contracting an output leg of one against an input leg of the
  other identifies the two spiders' ``k`` in the sum: the contraction is
  ``sum_k e^{i*angle_a(k)} e^{i*angle_b(k)} |k>^{ox (surviving outputs)} <k|^{ox (surviving inputs)}``,
  i.e. exactly the merged spider's tensor with phase vector ``alpha + beta`` (componentwise
  addition, per :meth:`~qufzx.algebra.phase.PhaseVector.__add__`) and no extra factor:
  ``e^{i*angle_a(k)} * e^{i*angle_b(k)} = e^{i*(angle_a(k) + angle_b(k))}`` is an exact
  identity of complex exponentials, not an approximation.
* X spider. ``X_{m->n} = F^{ox n} . Z_{m->n} . (conj(F))^{ox m}``. An output-to-input wire
  contracts an ``F`` (applied to the OUTPUT-side spider's leg) against a ``conj(F)``
  (applied to the INPUT-side spider's leg) on the shared axis -- this is exactly why the
  ``wire_direction_output_to_input`` side condition in :mod:`qufzx.rewrite.match` is
  checked uniformly for both colors. Since ``F`` is unitary, ``F^dagger . F = conj(F)^T . F
  = I`` (``F`` is symmetric, so ``F^dagger = conj(F)``), so the two Fourier factors on the
  contracted leg cancel to the identity, leaving exactly the Z-spider fusion argument above
  sandwiched between the surviving ``F``/``conj(F)`` factors on every other leg -- which is
  precisely the merged X spider's own denotation. Again no leftover factor.

Both derivations independently land on the scalar :meth:`~qufzx.algebra.scalar.Scalar.one`,
which is what :data:`SPIDER_FUSION` declares and what :func:`spider_fusion_builder` returns
per application -- this was derived from the denotation formulas above, not asserted by
fiat, and it agrees with the Phase 4 oracle in ``tests/test_phase5_oracle.py``. Had the two
disagreed, this docstring would say so explicitly rather than quietly matching one to the
other; they do not.

Merged leg-ordering convention. A choice, stated once here, not a derivation (matching how
:mod:`qufzx.semantics.denote` states its outputs-before-inputs axis convention as a choice):
the merged node's inputs are A's surviving inputs, in their original index order, followed
by B's surviving inputs, in their original index order; its outputs are A's surviving
outputs followed by B's surviving outputs, under the same rule. "A" and "B" here are fixed
by :class:`~qufzx.rewrite.match.FusionMatch`'s own convention -- A is always the lower
:class:`~qufzx.diagram.graph.NodeId` of the matched pair. "Surviving" means every leg
except the single one consumed by the matched wire.

Corner case: no legs survive at all. When A is ``0->1`` and B is ``1->0``, wired
output-to-input, fusion consumes every leg of both spiders and the merged node ends up
with zero inputs and zero outputs. Because this codebase stores dimension per port and
nowhere else (see ``claude.md``), such a node would otherwise carry no record of
``shared_dim`` whatsoever -- not lost precision, but lost information, since the
legless spider denotes the scalar ``sum_k 1 = d``. :func:`_merged_phase` handles this by
returning an explicit ``PhaseVector(shared_dim, {})`` instead of `None` in that one case,
giving the dimension a port-shaped home even though the phase itself is zero either way.
Every other phaseless fusion (any node with at least one surviving leg) keeps returning
`None`, unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef
from qufzx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionMatch, FusionPattern
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


def _merged_phase(
    node_a: Node, node_b: Node, shared_dim: Dim, *, any_legs_survive: bool
) -> PhaseVector | None:
    """The merged node's phase: the componentwise sum, treating an absent phase as zero.

    A `None` phase on both sides stays `None` (rather than becoming an explicit
    all-zero :class:`~qufzx.algebra.phase.PhaseVector`), matching the phase-free
    convention every other diagram-building helper in this codebase uses -- *except*
    when ``any_legs_survive`` is `False`. Since dimension is stored per port and nowhere
    else (see ``claude.md``), a merged node with no surviving legs at all has no port to
    carry ``shared_dim``; leaving its phase `None` would silently drop the dimension,
    and :func:`qufzx.semantics.denote.resolve_dimension` would then have no way to
    recover it. So in that one case this returns an explicit
    ``PhaseVector(shared_dim, {})`` -- an all-zero phase, mathematically equivalent to no
    phase at all -- purely to give the dimension a place to live on the legless node.

    Defensive check, not a normal path: :mod:`qufzx.rewrite.match`'s
    ``phase_dimension_agreement`` side condition already guarantees every phase present
    carries a ``Dim`` exactly equal to ``shared_dim`` before a match is ever returned, so
    ``phase_a.dim == phase_b.dim`` should always hold here. If it is somehow violated
    anyway (a hand-built match bypassing the matcher), raise :class:`RewriteDomainError`
    rather than let :meth:`~qufzx.algebra.phase.PhaseVector.__add__`'s
    :class:`~qufzx.algebra.phase.PhaseDomainError` -- a different module's exception
    hierarchy entirely -- escape through this builder.
    """
    if node_a.phase is None and node_b.phase is None:
        if not any_legs_survive:
            return PhaseVector(shared_dim, {})
        return None
    phase_a = node_a.phase if node_a.phase is not None else PhaseVector(shared_dim, {})
    phase_b = node_b.phase if node_b.phase is not None else PhaseVector(shared_dim, {})
    if phase_a.dim != phase_b.dim:
        raise RewriteDomainError(
            f"spider_fusion cannot merge phases over mismatched dimensions "
            f"{phase_a.dim} and {phase_b.dim}; the matcher should never hand this "
            "builder such a match"
        )
    return phase_a + phase_b


def spider_fusion_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`SPIDER_FUSION`: merge the two matched spiders.

    Mutates ``diagram`` in place by adding the merged node (see the module docstring for
    the leg-ordering and scalar conventions) and returns the :class:`BuildResult`
    :mod:`qufzx.rewrite.engine` needs to splice it in; never removes the matched nodes or
    touches any wire or boundary entry itself -- see :class:`~qufzx.rewrite.rule.BuildResult`.
    Verifies ``match``'s ``side_condition_outcomes`` exactly covers
    :data:`~qufzx.rewrite.match.FUSION_SIDE_CONDITIONS`, with every outcome passed (via
    :func:`~qufzx.rewrite.rule.check_side_condition_coverage`), before doing any graph
    surgery -- this builder is reachable directly, not only through
    :func:`qufzx.rewrite.engine.apply`, so it cannot rely on that function having already
    checked this.
    """
    if not isinstance(match, FusionMatch):
        raise RewriteGrammarError(
            f"spider_fusion requires a FusionMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, FUSION_SIDE_CONDITIONS, "spider_fusion")

    node_a = diagram.nodes.get(match.a_id)
    node_b = diagram.nodes.get(match.b_id)
    if node_a is None or node_b is None:
        raise RewriteGrammarError(
            f"matched node(s) {match.a_id!r}, {match.b_id!r} not found in the given diagram"
        )

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
    merged_phase = _merged_phase(
        node_a, node_b, match.shared_dim, any_legs_survive=any_legs_survive
    )

    new_node_id = diagram.add_node(
        node_a.generator_type,
        input_dims=[port.dim for _, port in merged_inputs],
        output_dims=[port.dim for _, port in merged_outputs],
        phase=merged_phase,
    )

    port_mapping: dict[PortRef, PortRef] = {}
    for new_index, (old_ref, _) in enumerate(merged_inputs):
        port_mapping[old_ref] = PortRef(new_node_id, Direction.INPUT, new_index)
    for new_index, (old_ref, _) in enumerate(merged_outputs):
        port_mapping[old_ref] = PortRef(new_node_id, Direction.OUTPUT, new_index)

    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_node_id,),
        consumed_node_ids=(match.a_id, match.b_id),
        consumed_wires=(wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


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
"""Same-color, single-wire spider fusion. See the module docstring for the scalar derivation."""


RULES: Mapping[str, Rule] = MappingProxyType({SPIDER_FUSION.name: SPIDER_FUSION})
"""Every rule this module registers, keyed by :attr:`~qufzx.rewrite.rule.Rule.name`.

See the module docstring's "Rule registry" paragraph for why this exists. A
``MappingProxyType`` so a caller cannot mutate the registry through the reference
:func:`lookup_rule` or this constant hands out.
"""


def lookup_rule(name: str) -> Rule:
    """Resolve a rule name (e.g. ``"spider_fusion"``) back to its :class:`Rule` object.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` if ``name`` is not registered
    in :data:`RULES` -- an unknown rule name is a malformed replay request, not a
    mathematical domain violation.
    """
    try:
        return RULES[name]
    except KeyError:
        raise RewriteGrammarError(f"no such rule: {name!r}") from None
