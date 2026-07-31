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
introduces no scalar factor -- read directly off :mod:`qufzx.semantics.denote`'s formulas,
the merged node's denotation already equals the pre-fusion diagram's contraction with no
leftover coefficient:

* Z spider. Both spiders are diagonal, entry ``e^{i*angle(k)}`` at the all-axes-``k``
  position. Contracting an output leg of one against an input leg of the other identifies
  their ``k`` in the sum, giving exactly the merged spider's tensor with phase vector
  ``alpha + beta`` (:meth:`~qufzx.algebra.phase.PhaseVector.__add__`) and no extra factor.
* X spider. ``X_{m->n} = F^{ox n} . Z_{m->n} . (conj(F))^{ox m}``; an output-to-input wire
  contracts an ``F`` against a ``conj(F)`` on the shared axis -- why
  ``wire_direction_output_to_input`` in :mod:`qufzx.rewrite.match` is checked uniformly for
  both colors -- and since ``F`` is unitary and symmetric these cancel to the identity,
  leaving exactly the Z-spider argument above sandwiched between the surviving Fourier
  factors on every other leg.

Both derivations land on :meth:`~qufzx.algebra.scalar.Scalar.one`, which is what
:data:`SPIDER_FUSION` declares and what :func:`spider_fusion_builder` returns per
application, agreeing with the Phase 4 oracle in ``tests/test_phase5_oracle.py``.

Merged leg-ordering convention. A choice, stated once here, not a derivation: the merged
node's inputs are A's surviving inputs, original index order, then B's; its outputs follow
the same rule. "A" and "B" are :class:`~qufzx.rewrite.match.FusionMatch`'s own convention
(A is always the lower :class:`~qufzx.diagram.graph.NodeId`); "surviving" means every leg
except the one consumed by the matched wire.

Dimension of the merged node. Every surviving port -- A's and B's alike -- is built at
exactly :attr:`~qufzx.rewrite.match.FusionMatch.shared_dim`, never each leg's own original
``Dim``; Z and X are ``ALL_LEGS_EQUAL`` (:mod:`qufzx.diagram.generators`), so collapsing
every surviving leg onto one canonical dimension is sound under the generator's own
policy. When the matched wire's ``dimension_agreement`` only deferred, ``shared_dim`` is
A's own raw, unbound dim, and forcing it onto B's survivors carries that assumed equality
into the diagram -- a neighbouring wire that was an exact match before this fusion may
become merely deferred after it. That is expected, not a defect. See :func:`_merged_phase`
for the legless corner case, where dimension can only survive via the phase slot.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseDomainError, PhaseVector
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


def _over_shared_dim(phase: PhaseVector | None, shared_dim: Dim) -> PhaseVector:
    """``phase``'s entries reattached to ``shared_dim`` unchanged, or an all-zero vector if absent.

    Only the container's ``Dim`` is replaced -- entries carry over as-is, mirroring exactly
    what :mod:`qufzx.rewrite.match`'s ``phase_dimension_agreement`` condition itself
    verifies, never a deeper substitution into the entries. Raises
    :class:`RewriteDomainError`, not :class:`~qufzx.algebra.phase.PhaseDomainError`, if an
    entry index falls outside ``shared_dim``'s valid range -- this builder is reachable
    directly, not only through :func:`qufzx.rewrite.engine.apply`, so a foreign or
    hand-built match must not leak a different module's exception hierarchy through it.
    ``phase_dimension_agreement`` now performs this exact same construction as part of
    matching (see that condition's entry in :mod:`qufzx.rewrite.match`'s module docstring),
    so for any match ``find_matches`` actually returned, this call cannot raise -- the
    ``except`` branch below is unreachable from such a match and exists only to keep this
    function safe against a foreign or hand-built one.
    """
    if phase is None:
        return PhaseVector(shared_dim, {})
    try:
        return PhaseVector(shared_dim, phase.entries())
    except PhaseDomainError as exc:
        raise RewriteDomainError(
            f"spider_fusion cannot reattach a phase vector to shared dimension "
            f"{shared_dim}: {exc}"
        ) from exc


def _merged_phase(
    node_a: Node, node_b: Node, shared_dim: Dim, *, any_legs_survive: bool
) -> PhaseVector | None:
    """The merged node's phase: componentwise sum, both operands read over ``shared_dim``.

    A `None` phase on both sides stays `None` -- *except* when ``any_legs_survive`` is
    `False`: a merged node with no surviving legs has no port left to carry ``shared_dim``
    (dimension lives per port and nowhere else, see ``claude.md``), so this returns an
    explicit all-zero ``PhaseVector(shared_dim, {})`` purely to give it a place to live.
    Otherwise, both operands are read via :func:`_over_shared_dim` before adding, since
    :meth:`~qufzx.algebra.phase.PhaseVector.__add__` demands its two operands' ``Dim``\\ s
    be exactly equal.
    """
    if node_a.phase is None and node_b.phase is None:
        if not any_legs_survive:
            return PhaseVector(shared_dim, {})
        return None
    return _over_shared_dim(node_a.phase, shared_dim) + _over_shared_dim(node_b.phase, shared_dim)


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
        input_dims=[match.shared_dim] * len(merged_inputs),
        output_dims=[match.shared_dim] * len(merged_outputs),
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
