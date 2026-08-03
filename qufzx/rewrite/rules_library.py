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

Scalar derivation (not assertion). Same-color, single-wire fusion introduces no scalar
factor in either shape the rule matches -- read directly off
:mod:`qufzx.semantics.denote`'s formulas, the merged node's denotation already equals the
pre-fusion diagram's contraction with no leftover coefficient. Two distinct wire shapes are
possible, gated by :mod:`qufzx.rewrite.match`'s condition 4
(``wire_direction_output_to_input`` -- see that module's docstring for the full account of
exactly which direction combinations each color permits, and why): an alternating
output-to-input wire, valid fusion for both colors, and a same-direction (output-output or
input-input) wire, valid fusion for Z only.

* Alternating output-to-input, either color. Z: both spiders are diagonal, entry
  ``e^{i*angle(k)}`` at the all-axes-``k`` position; contracting an output leg of one
  against an input leg of the other identifies their ``k`` in the sum, giving exactly the
  merged spider's tensor with phase vector ``alpha + beta``
  (:meth:`~qufzx.algebra.phase.PhaseVector.__add__`) and no extra factor. X:
  ``X_{m->n} = F^{ox n} . Z_{m->n} . (conj(F))^{ox m}``; an output-to-input wire contracts
  an ``F`` against a ``conj(F)`` on the shared axis, and since ``F`` is unitary and
  symmetric these cancel to the identity, leaving exactly the Z-spider argument above
  sandwiched between the surviving Fourier factors on every other leg.
* Same-direction, Z only. ``_z_tensor`` is diagonal in every axis regardless of direction,
  and :mod:`qufzx.semantics.contract_numeric` contracts a wire by assigning its two
  endpoints the same einsum axis label unconditionally, applying no conjugation at
  contraction time at all (for X, conjugation is applied only at ``denote()`` time, when an
  input axis's Fourier factor is built as ``conj(F)`` rather than ``F`` -- it plays no part
  in contraction itself). A same-direction Z-Z wire therefore identifies the same basis
  index ``k`` on both endpoints exactly as the alternating case does, yielding the merged
  spider's tensor with phase ``alpha + beta`` and no leftover coefficient -- the identical
  derivation to the Z case above, merely without requiring one endpoint to be an input and
  the other an output. This argument does not carry over to X: X's own tensor is not
  diagonal (its diagonal Z core sits sandwiched between Fourier factors), so a
  same-direction X-X wire would contract ``F`` against ``F`` (or ``conj(F)`` against
  ``conj(F)``) on the shared axis rather than ``F`` against ``conj(F)`` -- giving
  ``F^T F`` (or its conjugate), a nontrivial permutation matrix, not the identity. That is a
  different (and, for Phase 5, unimplemented) rule, which is exactly why condition 4
  restricts same-direction fusion to Z.

Both shapes land on :meth:`~qufzx.algebra.scalar.Scalar.one`, which is what
:data:`SPIDER_FUSION` declares and what :func:`spider_fusion_builder` returns per
application, agreeing with the Phase 4 oracle in ``tests/test_phase5_oracle.py`` and with
the fuzz-tested oracle comparisons in ``tests/test_fusion_properties.py``.

Merged leg-ordering convention. A choice, stated once here, not a derivation: the merged
node's inputs are A's surviving inputs, original index order, then B's; its outputs follow
the same rule. "A" and "B" are :class:`~qufzx.rewrite.match.FusionMatch`'s own convention
(A is always the lower :class:`~qufzx.diagram.graph.NodeId`); "surviving" means every leg
except the one consumed by the matched wire.

Dimension of the merged node. Every surviving port -- A's and B's alike -- is built at
exactly :attr:`~qufzx.rewrite.match.FusionMatch.shared_dim`, never each leg's own original
``Dim``. This is *not* sound merely because Z and X are ``ALL_LEGS_EQUAL``
(:mod:`qufzx.diagram.generators`): ``ALL_LEGS_EQUAL`` is enforced by
:mod:`qufzx.diagram.validate` via :meth:`~qufzx.algebra.dimension.Dim.unify`, and
``validate`` reports no issue at all -- not even a deferred one -- for two leg dims that
unify by binding a free symbol (e.g. a leg stated over ``Dim(2)`` sitting beside a leg
stated over the still-free symbol ``Dim("d")``); forcing every surviving leg onto
``shared_dim`` without first checking it against each leg's own dim would silently
overwrite such a leg -- or, worse, a leg whose dim plainly does not unify with
``shared_dim`` at all -- with no record and no rejected match. What actually makes this
construction sound is that :mod:`qufzx.rewrite.match`'s ``dimension_agreement`` condition
(condition 5 in that module's docstring) itself unifies every surviving leg of both nodes
against the resolved ``shared_dim`` before a :class:`~qufzx.rewrite.match.FusionMatch` is
ever returned: a leg that fails to unify makes the candidate a non-match, and a leg that
only unifies by deferring or binding a symbol is recorded in
:attr:`~qufzx.rewrite.match.FusionMatch.dimension_constraints`, with ``shared_dim`` itself
possibly refined further by that leg's binding. By the time this builder runs, every
surviving port this loop assigns ``shared_dim`` to has therefore already been verified (or
assumed, with the assumption on record) to agree with it -- match-approval and
build-applicability are the same predicate by construction (see
:mod:`qufzx.rewrite.match`'s module docstring), not a policy this builder re-derives on
its own. When the matched wire's ``dimension_agreement`` only deferred (on the connecting
pair or on some surviving leg), the affected leg's assumed equality with ``shared_dim`` is
carried into the diagram -- a neighbouring wire that was an exact match before this fusion
may become merely deferred after it. That is expected, not a defect. See
:func:`_merged_phase` for the legless corner case, where dimension can only survive via
the phase slot.

Phase 5 judgement call (deliberate, not a defect): fusion is permitted to fire on a
``DEFERRED`` dimension pair at all, even though FULL_PLAN.md's Phase 5 item (ii) states the
pattern as "two same-color spiders joined by a wire and sharing a dimension" -- a
``DEFERRED`` unify means it is not actually known that the two legs share one, only that
:meth:`~qufzx.algebra.dimension.Dim.unify`'s deliberately weak placeholder could not decide
either way. Concretely, a node with legs ``[d, d*e]`` fused against a node with leg ``d``
produces a merged node whose surviving port is ``Dim(d)``; the ``d*e`` label is gone from
the diagram entirely, surviving only as a ``dimension_constraints`` entry on the
:class:`~qufzx.rewrite.engine.RewriteStep` certificate. This is consistent with
:mod:`qufzx.diagram.validate`'s own deferral posture (a ``DEFERRED`` unify is not a hard
error there either), and is fully argued in both this module's account above and
:mod:`qufzx.rewrite.match`'s module docstring ("Dimension constraints"). But it is worth
naming plainly: the assumption a diagram-level unifier must eventually justify is recorded
only on the certificate, not in the diagram itself, which is in tension with ``CLAUDE.md``'s
"the diagram is the single source of truth" -- a diagram inspected on its own, without its
certificate history, cannot recover that a surviving leg's dimension was ever anything other
than what it now says. This is flagged here as a known, accepted Phase 5 limitation; closing
it properly needs Phase 10's real unifier (which can resolve a ``DEFERRED`` pair outright
instead of assuming it), not a change to this builder.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseDomainError, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef
from qufzx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionMatch, FusionPattern, _reattach_phase
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


def _over_shared_dim(phase: PhaseVector | None, shared_dim: Dim, bindings: Mapping[str, Dim]) -> PhaseVector:
    """``phase``'s entries, with ``bindings`` substituted in, reattached to ``shared_dim``.

    Or an all-zero vector if ``phase`` is absent. Delegates to
    :func:`qufzx.rewrite.match._reattach_phase` -- the same function
    ``phase_dimension_agreement`` uses as its own trial construction while matching -- so
    that match-approval and build-applicability are the same predicate by construction, not
    two similar-looking computations kept in sync by hand. See that function's docstring for
    the full account of why substituting ``bindings`` into the entries (rather than
    reattaching them unchanged, the pre-fix behavior) is the correct resolution of the
    ``_over_shared_dim`` defect family: an entry stated in terms of a dimension symbol that
    ``shared_dim`` resolution has since bound (e.g. ``1/d turns`` once ``d := 2``) denotes a
    different, and wrong, angle if left unsubstituted once the container ``Dim`` becomes the
    concrete ``2``.

    Raises :class:`RewriteDomainError`, not :class:`~qufzx.algebra.phase.PhaseDomainError`,
    if an entry index falls outside ``shared_dim``'s valid range -- this builder is
    reachable directly, not only through :func:`qufzx.rewrite.engine.apply`, so a foreign or
    hand-built match must not leak a different module's exception hierarchy through it.
    ``phase_dimension_agreement`` performs this exact same construction (with the same
    ``bindings``) as part of matching, so for any match ``find_matches`` actually returned,
    this call cannot raise -- the ``except`` branch below is unreachable from such a match
    and exists only to keep this function safe against a foreign or hand-built one.
    """
    if phase is None:
        return PhaseVector(shared_dim, {})
    try:
        return _reattach_phase(phase, shared_dim, bindings)
    except PhaseDomainError as exc:
        raise RewriteDomainError(
            f"spider_fusion cannot reattach a phase vector to shared dimension "
            f"{shared_dim}: {exc}"
        ) from exc


def _merged_phase(
    node_a: Node,
    node_b: Node,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
    *,
    any_legs_survive: bool,
) -> PhaseVector | None:
    """The merged node's phase: componentwise sum, both operands read over ``shared_dim``.

    A `None` phase on both sides stays `None` -- *except* when ``any_legs_survive`` is
    `False`: a merged node with no surviving legs has no port left to carry ``shared_dim``
    (dimension lives per port and nowhere else, see ``claude.md``), so this returns an
    explicit all-zero ``PhaseVector(shared_dim, {})`` purely to give it a place to live.
    Otherwise, both operands are read via :func:`_over_shared_dim` (which substitutes
    ``bindings`` into each operand's entries before reattaching to ``shared_dim`` -- see
    that function's docstring) before adding, since
    :meth:`~qufzx.algebra.phase.PhaseVector.__add__` demands its two operands' ``Dim``\\ s
    be exactly equal.
    """
    if node_a.phase is None and node_b.phase is None:
        if not any_legs_survive:
            return PhaseVector(shared_dim, {})
        return None
    return _over_shared_dim(node_a.phase, shared_dim, bindings) + _over_shared_dim(
        node_b.phase, shared_dim, bindings
    )


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
    checked this. Also verifies ``match.wire`` actually joins ``match.a_id`` and
    ``match.b_id`` (raising :class:`~qufzx.rewrite.rule.RewriteGrammarError` naming the
    wire and both node ids if not), for the same reason: any match ``find_matches`` itself
    produced satisfies this by construction, but a hand-built or foreign ``FusionMatch``
    whose wire names some other pair would otherwise make the consumed-ref selection just
    below silently pick the wrong port(s) on ``node_a``/``node_b``, and fail much later
    with a step-8 relative-postcondition error that names the wrong defect entirely.
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
    wire_node_ids = {wire.a.node_id, wire.b.node_id}
    if wire_node_ids != {match.a_id, match.b_id}:
        raise RewriteGrammarError(
            f"spider_fusion: match's wire {wire!r} is not incident on both matched nodes "
            f"{match.a_id!r} and {match.b_id!r} (wire connects {sorted(wire_node_ids)!r}); "
            "a hand-built or foreign FusionMatch whose wire does not actually join a_id "
            "and b_id would otherwise have its consumed-ref selection below silently pick "
            "the wrong port(s), failing much later with an unrelated error"
        )
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
        node_a, node_b, match.shared_dim, match.bindings, any_legs_survive=any_legs_survive
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
"""Same-color, single-wire spider fusion -- any direction for Z, output-to-input only for X.

See the module docstring for the scalar derivation (both wire shapes) and condition 4 in
:mod:`qufzx.rewrite.match` for exactly which direction combinations each color permits.
"""


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
