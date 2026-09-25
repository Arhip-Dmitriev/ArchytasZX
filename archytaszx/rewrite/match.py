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

"""The fusion matcher: locates occurrences of same-color spider fusion.

Phase 5 implements one :class:`~archytaszx.rewrite.rule.Pattern`: two spiders of the same
generator type joined by a wire whose connected legs agree on dimension. A pair joined by k
wires yields up to one match per wire, each decided on its own; a match fuses across its own
wire and leaves the rest as self-loops on the merged node.

Side conditions, in the order applied (see ``FUSION_SIDE_CONDITIONS``):

1. ``distinct_nodes`` -- the endpoints are different nodes. Always True.
2. ``same_generator_type`` -- both nodes carry the identical registered
   :class:`~archytaszx.diagram.generators.GeneratorType`, and it is fusable (Z or X).
3. ``parallel_wires_become_self_loops`` -- always True, carrying the count of other wires
   joining the pair; each survives as a self-loop on the merged spider.
4. ``consumed_wire_direction_permitted_for_color`` -- for X the consumed wire runs
   OUTPUT-to-INPUT; Z permits any direction combination.
5. ``consumed_ports_singly_claimed`` -- neither consumed port is claimed by a second wire
   or listed on a boundary.
6. ``bang_box_scope_agreement`` (Phase 7) -- both nodes' innermost enclosing node-scope
   bang box, if any, must be identical: both unboxed, or the same box.
7. ``dimension_agreement`` -- the connected legs' :class:`~archytaszx.algebra.dimension.Dim`
   unify. A ``FAILURE`` is a non-match; a ``DEFERRED`` or binding-only ``SUCCESS`` is
   recorded as a dimension constraint. Every surviving leg of both nodes is then unified
   against the running ``shared_dim``, each refinement carrying forward.
8. ``phase_dimension_agreement`` -- every phase vector present must unify with
   ``shared_dim``; unlike condition 7 a ``DEFERRED`` is rejected. Conditions 7 and 8 form
   one bounded fixpoint; see :func:`resolve_fusion_match`.
9. ``dimension_guards_satisfied`` (Phase 10) -- every
   :class:`~archytaszx.rewrite.rule.DimensionGuard` the rule declares holds at the settled
   ``shared_dim``. An ``UNDECIDED`` guard blocks, with ``deferred`` False.

Conditions 1 and 3 are structural facts recorded for the certificate, not decisions. The
numbering above is machine-checked against ``FUSION_SIDE_CONDITIONS`` by
``tests/test_engine.py::TestConditionNumberingMatchesDeclaredOrder``.

One verification predicate. :func:`resolve_fusion_match` decides every condition above.
:func:`find_matches` calls it to decide whether a candidate is a match, and
:func:`~archytaszx.rewrite.rules_library.spider_fusion_builder` calls it again, fresh, against
the diagram it was handed, building only from its result.

Malformed references. :func:`find_matches` validates both endpoints of every wire and every
boundary entry in a pre-pass, raising :class:`~archytaszx.rewrite.rule.RewriteGrammarError`.

Match-implies-applicable. Every match returned here applies under
:func:`~archytaszx.rewrite.engine.apply` without raising anything except the step-8
relative-postcondition :class:`~archytaszx.rewrite.rule.RewriteDomainError`.

Dimension constraints. ``dimension_constraints`` records every equality accepted without a
syntactic identity -- a ``DEFERRED`` unify or a binding-only ``SUCCESS`` -- as
:class:`~archytaszx.rewrite.rule.DimensionConstraint`, at most one per
:class:`~archytaszx.rewrite.rule.ConstraintSource`; a symbolic binding stays an assumption.

Determinism. :func:`find_matches` sorts its result by node ids, then by the consumed wire's
(direction, index) on each side; every order-observable set iteration is sorted by a
hash-independent key.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from archytaszx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from archytaszx.algebra.phase import (
    PhaseDomainError,
    PhaseSubstituteValue,
    PhaseSymbolKey,
    PhaseVector,
)
from archytaszx.diagram.generators import (
    FOURIER_BOX,
    REGISTRY,
    TRIANGLE,
    TRIANGLE_INVERSE,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, Node, NodeId, PortRef, Wire
from archytaszx.rewrite.rule import (
    ConstraintOutcome,
    ConstraintSource,
    ConstraintSourceKind,
    DimensionConstraint,
    DimensionGuard,
    GuardOutcome,
    Match,
    Pattern,
    RewriteGrammarError,
    SideCondition,
    SideConditionOutcome,
)

FUSION_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "distinct_nodes",
        "recorded fact, never a decision: the two matched nodes are not the same node",
    ),
    SideCondition("same_generator_type", "both nodes are the same registered spider color"),
    SideCondition(
        "parallel_wires_become_self_loops",
        "recorded fact, never a decision: every other wire joining the two nodes survives as "
        "a self-loop on the merged spider",
    ),
    SideCondition(
        "consumed_wire_direction_permitted_for_color",
        "for X, the consumed wire runs OUTPUT to INPUT; for Z, any direction combination "
        "is valid fusion",
    ),
    SideCondition(
        "consumed_ports_singly_claimed",
        "neither consumed port is claimed by a second wire, and neither is listed on "
        "either boundary list",
    ),
    SideCondition(
        "bang_box_scope_agreement",
        "both matched nodes' innermost enclosing node-scope bang box, if any, are "
        "identical (both unboxed, or the same box) -- Phase 7",
    ),
    SideCondition(
        "dimension_agreement",
        "the connecting pair and every surviving leg of both nodes unify, in a bounded "
        "fixpoint, against the shared dimension -- equal outright, or unify defers or "
        "binds a symbol",
    ),
    SideCondition(
        "phase_dimension_agreement",
        "every phase vector present unifies with the resolved shared leg dimension -- equal "
        "outright, or unify binds a symbol to a concrete value (never merely defers, and "
        "never binds to another still-symbolic Dim -- see the module docstring's "
        "'Dimension constraints' note)",
    ),
    SideCondition(
        "dimension_guards_satisfied",
        "every dimension guard the rule declares holds at the matched shared dimension",
    ),
)
"""The declared side-condition specs for :class:`FusionPattern`, in the module docstring's
numbered order. Entries 1 and 3 are structural facts a candidate cannot fail; the rest are
decisions."""


@dataclass(frozen=True, slots=True)
class FusionMatch:
    """One located fusion occurrence: the two spiders, the consumed wire, and the shared dim.

    ``a_id`` is always the lower :class:`~archytaszx.diagram.graph.NodeId` of the pair,
    ``b_id`` the higher -- the convention :mod:`archytaszx.rewrite.rules_library` reuses as its
    merged-leg ordering ("A's surviving legs, then B's"), and the one that seeds
    ``shared_dim`` from the A-side consumed leg's ``Dim``. A connecting pair that unifies
    resolves the seed away; one that only defers leaves it standing as ``shared_dim``, so a
    ``d``/``d*e`` pair fuses onto whichever the lower-id node carried.

    ``bindings`` is the whole-candidate accumulator of every concrete symbol binding
    conditions 7 and 8 produced. The builder substitutes it into a present phase's entries,
    via :func:`reattach_phase`, before reattaching them to ``shared_dim``.
    """

    a_id: NodeId
    b_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()
    bindings: Mapping[str, Dim] = MappingProxyType({})

    def __hash__(self) -> int:
        """Hash every field, with ``bindings`` hashed as an order-independent frozenset.

        Explicit: the generated ``__hash__`` would hash a
        :class:`~types.MappingProxyType`, which is unhashable. The frozenset matches the
        generated ``__eq__``'s mapping equality, so ``a == b`` implies ``hash(a) == hash(b)``.

        Within-process only: ``Wire`` and ``DimensionConstraint`` reach ``enum.Enum``
        members transitively, whose hashes are ``PYTHONHASHSEED``-dependent.
        """
        return hash(
            (
                self.a_id,
                self.b_id,
                self.wire,
                self.shared_dim,
                self.side_condition_outcomes,
                self.dimension_constraints,
                frozenset(self.bindings.items()),
            )
        )

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed.

        See :class:`archytaszx.rewrite.rule.Match`.
        """
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The two fused node ids, ascending."""
        return _support_ids(self.a_id, self.b_id, self.wire.a.node_id, self.wire.b.node_id)


_FUSABLE_GENERATOR_NAMES = frozenset((Z_SPIDER.name, X_SPIDER.name))
_SAME_DIRECTION_FUSABLE_GENERATOR_NAMES = frozenset((Z_SPIDER.name,))
"""Generator names for which a same-direction connecting wire is still valid fusion.

Z only; a same-direction X wire is a different, unimplemented rule (module docstring,
condition 4)."""

_MAX_FIXPOINT_PASSES = 32
"""Iteration budget for :func:`resolve_fusion_match`'s joint leg/phase fixpoint.

Module-level so a test can patch it low and exercise the exhaustion path. Unreachable in
practice: ``bindings`` is monotone and drawn from the finite free-symbol set of both nodes'
legs, phases, and the connecting pair, so a non-stabilising pass adds at least one fresh
key. Kept as a structural guard on :meth:`~archytaszx.algebra.dimension.Dim.unify`'s
contract."""


def _resolve_with_bindings(dim: Dim, bindings: Mapping[str, Dim]) -> Dim:
    """Substitute the concrete entries of ``bindings`` into ``dim``.

    Non-concrete bindings (``d := e``) are dropped rather than substituted, since
    :meth:`Dim.substitute` accepts only concrete replacements; ``dim`` stays unchanged for
    that symbol.
    """
    concrete_bindings = {name: value for name, value in bindings.items() if value.is_concrete}
    if not concrete_bindings:
        return dim
    return dim.substitute(cast(Mapping[DimSymbolKey, DimSubstituteValue], concrete_bindings))


class _FailureReason(enum.Enum):
    """Why one of the fixpoint's unify helpers failed this pass.

    ``PHASE_DEFERRED`` and ``PHASE_NON_CONCRETE_BINDING`` are
    :func:`_unify_phase_dims`-only.
    """

    UNIFY_FAILURE = "unify_failure"
    """:meth:`~archytaszx.algebra.dimension.Dim.unify` returned ``FAILURE`` outright."""

    CONTRADICTORY_REBIND = "contradictory_rebind"
    """The unify succeeded, but :func:`_merge_bindings` rejected its binding as a
    contradictory rebind of a symbol already bound to a different concrete value."""

    PHASE_DEFERRED = "phase_deferred"
    """A phase's own dimension unify ``DEFERRED`` against the shared leg dimension --
    tolerated for a leg or the connecting pair, not for a phase (condition 8,
    ``phase_dimension_agreement``)."""

    PHASE_NON_CONCRETE_BINDING = "phase_non_concrete_binding"
    """A phase's own dimension unify succeeded only by binding to a non-concrete ``Dim`` --
    tolerated for a leg, not for a phase, whose entries can reference its ``dim``'s free
    symbols directly."""


@dataclass(frozen=True, slots=True)
class _ResolutionFailure:
    """One unify helper's failure this pass: why, and the two operands involved.

    ``assumed``/``equal_to`` are the operands as checked this pass, already resolved through
    the running ``bindings`` accumulator. Call sites render their detail strings from these
    fields and ``reason``.
    """

    reason: _FailureReason
    assumed: Dim
    equal_to: Dim


def _merge_bindings(bindings: dict[str, Dim], new_bindings: Mapping[str, Dim]) -> bool:
    """Merge the concrete entries of ``new_bindings`` into ``bindings``, in place.

    Only concrete-valued bindings are stored. Returns ``False``, leaving ``bindings``
    unmodified, iff a name already bound to one concrete ``Dim`` would be rebound to a
    different one; ``True`` on a clean merge. A ``False`` return makes the candidate a
    non-match, like a ``FAILURE``.

    The contradiction branch does not fire on any current call site: every operand is first
    passed through :func:`_resolve_with_bindings`, so an already-bound symbol is never free
    in what reaches ``Dim.unify``. Kept as a structural guard on that invariant.
    """
    concrete = {name: value for name, value in new_bindings.items() if value.is_concrete}
    for name, value in concrete.items():
        existing = bindings.get(name)
        if existing is not None and existing != value:
            return False
    bindings.update(concrete)
    return True


class _ConstraintRecord:
    """The source-keyed, insertion-ordered record of one candidate's dimension assumptions.

    One entry per :class:`~archytaszx.rewrite.rule.ConstraintSource`, never one per check: the
    leg/phase fixpoint re-checks the same source once per pass, and each re-check
    :meth:`record`\\ s over the previous entry in place, so the finished sequence is in
    first-derivation order.

    The invariant is adequacy: the finished ``dimension_constraints`` alone implies every
    ``(assumed, equal_to)`` pair any check ever asserted, including one a later pass replaced
    or dropped. Checked by
    ``tests/test_phase5_certificate_sweep.py::TestConstraintRecordAdequacy``; every cell of
    the table below is pinned by ``tests/test_match.py::TestConstraintRecordPolicyTable``.

    The policy over (previous entry, this check's outcome):

    * (none, ``DEFERRED`` / ``BOUND``): record it.
    * (none, bare identity via :meth:`record_identity`): no-op.
    * (``DEFERRED``, anything recorded): overwrite, at its currently-resolved operands.
    * (``DEFERRED``, bare identity): drop -- nothing is assumed any more.
    * (``BOUND``, anything recorded): overwrite; the record holds each source's most-resolved
      current statement, not a history.
    * (``BOUND``, bare identity): **keep** the ``BOUND`` entry, the identity holding only
      under that binding. The one cell where :meth:`record_identity` does not mirror
      :meth:`record`.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[ConstraintSource, DimensionConstraint] = {}

    def record(
        self,
        source: ConstraintSource,
        assumed: Dim,
        equal_to: Dim,
        outcome: ConstraintOutcome,
        bound_here: Mapping[str, Dim] | None = None,
    ) -> None:
        """Record (or re-record, in place) ``source``'s assumed equality.

        ``bound_here`` is the raw ``UnifyResult.bindings`` this check produced -- non-empty
        when ``outcome`` is ``BOUND``, omitted otherwise.
        """
        self._entries[source] = DimensionConstraint(
            assumed=assumed,
            equal_to=equal_to,
            source=source,
            outcome=outcome,
            bound_here=tuple(sorted((bound_here or {}).items())),
        )

    def record_identity(self, source: ConstraintSource) -> None:
        """Note that ``source`` re-checked as a bare identity. See the class docstring."""
        existing = self._entries.get(source)
        if existing is not None and existing.outcome is ConstraintOutcome.DEFERRED:
            del self._entries[source]

    def entries(self) -> tuple[DimensionConstraint, ...]:
        """Every recorded constraint, in first-derivation order."""
        return tuple(self._entries.values())

    def entry_for(self, source: ConstraintSource) -> DimensionConstraint | None:
        """The current entry for ``source``, or ``None`` if never recorded, or since discharged."""
        return self._entries.get(source)

    def leg_count(self) -> int:
        """How many entries came from a surviving leg (not the connecting pair or a phase)."""
        return sum(
            1
            for entry in self._entries.values()
            if entry.source.kind is ConstraintSourceKind.SURVIVING_LEG
        )

    def any_leg_deferred(self) -> bool:
        """True iff a connecting-pair or surviving-leg entry is, finally, a ``DEFERRED`` one.

        Computed from the finished record, never a flag accumulated across passes.
        """
        return any(
            entry.deferred
            and entry.source.kind
            in (ConstraintSourceKind.CONNECTING_PAIR, ConstraintSourceKind.SURVIVING_LEG)
            for entry in self._entries.values()
        )


def _unify_surviving_legs(
    node: Node,
    node_id: NodeId,
    consumed_ref: PortRef,
    shared_dim: Dim,
    bindings: dict[str, Dim],
    record: _ConstraintRecord,
) -> Dim | _ResolutionFailure:
    """Unify every surviving leg of ``node`` (both directions) against ``shared_dim`` in turn.

    "Surviving" means every leg except ``consumed_ref``, checked in input-then-output,
    original-index order. Each leg's ``Dim`` is resolved through the running ``bindings``
    accumulator before being unified; each new concrete binding is merged back in and used to
    refine ``shared_dim``.

    Returns the (possibly refined) shared dimension, or a :class:`_ResolutionFailure` if a
    leg's resolved dim does not unify or unifies only via a contradictory binding -- either
    makes the candidate a non-match. Every leg's outcome is written into ``record`` under its
    own :meth:`~archytaszx.rewrite.rule.ConstraintSource.surviving_leg` key.
    """
    for direction in (Direction.INPUT, Direction.OUTPUT):
        for index, port in enumerate(node.legs(direction)):
            ref = PortRef(node_id, direction, index)
            if ref == consumed_ref:
                continue
            source = ConstraintSource.surviving_leg(ref)
            leg_dim = _resolve_with_bindings(port.dim, bindings)
            result = leg_dim.unify(shared_dim)
            if result.is_failure:
                return _ResolutionFailure(_FailureReason.UNIFY_FAILURE, leg_dim, shared_dim)
            bound_this_pass = result.is_success and bool(result.bindings)
            if result.is_deferred:
                record.record(source, leg_dim, shared_dim, ConstraintOutcome.DEFERRED)
            elif bound_this_pass:
                record.record(
                    source,
                    leg_dim,
                    shared_dim,
                    ConstraintOutcome.BOUND,
                    bound_here=result.bindings,
                )
            else:
                record.record_identity(source)
            if bound_this_pass:
                if not _merge_bindings(bindings, result.bindings):
                    return _ResolutionFailure(
                        _FailureReason.CONTRADICTORY_REBIND, leg_dim, shared_dim
                    )
                shared_dim = _resolve_with_bindings(shared_dim, result.bindings)
    return shared_dim


def _unify_phase_dims(
    node_a: Node,
    node_b: Node,
    a_id: NodeId,
    b_id: NodeId,
    shared_dim: Dim,
    bindings: dict[str, Dim],
    record: _ConstraintRecord,
) -> Dim | _ResolutionFailure:
    """Unify every phase vector actually present (A's, then B's) against ``shared_dim``.

    Same accumulator discipline as :func:`_unify_surviving_legs`. Unlike a leg, a
    ``DEFERRED`` result, or one whose binding is not concrete, is never accepted (module
    docstring, condition 8). Returns a :class:`_ResolutionFailure` on any of those or on a
    contradictory rebind; its ``equal_to`` is the ``shared_dim`` actually checked against the
    failing phase. On success returns the refined ``shared_dim``, having written each phase's
    binding into ``record`` under its
    :meth:`~archytaszx.rewrite.rule.ConstraintSource.node_phase` key.
    """
    for node_id, phase in ((a_id, node_a.phase), (b_id, node_b.phase)):
        if phase is None:
            continue
        source = ConstraintSource.node_phase(node_id)
        phase_dim = _resolve_with_bindings(phase.dim, bindings)
        phase_unify = phase_dim.unify(shared_dim)
        if phase_unify.is_failure:
            return _ResolutionFailure(_FailureReason.UNIFY_FAILURE, phase_dim, shared_dim)
        if phase_unify.is_deferred:
            return _ResolutionFailure(_FailureReason.PHASE_DEFERRED, phase_dim, shared_dim)
        if not phase_unify.bindings:
            record.record_identity(source)
            continue
        if not all(value.is_concrete for value in phase_unify.bindings.values()):
            return _ResolutionFailure(
                _FailureReason.PHASE_NON_CONCRETE_BINDING, phase_dim, shared_dim
            )
        record.record(
            source,
            phase_dim,
            shared_dim,
            ConstraintOutcome.BOUND,
            bound_here=phase_unify.bindings,
        )
        if not _merge_bindings(bindings, phase_unify.bindings):
            return _ResolutionFailure(_FailureReason.CONTRADICTORY_REBIND, phase_dim, shared_dim)
        shared_dim = _resolve_with_bindings(shared_dim, phase_unify.bindings)
    return shared_dim


def _unify_connecting_pair(
    port_a_dim: Dim,
    port_b_dim: Dim,
    shared_dim: Dim,
    bindings: dict[str, Dim],
    record: _ConstraintRecord,
) -> Dim | _ResolutionFailure:
    """Re-derive the connecting pair's own equality, at its most-resolved form, this pass.

    The connecting pair relates its own two legs to each other rather than to ``shared_dim``,
    and seeds ``shared_dim`` on the first pass. Called once per pass, so a later pass sees
    whatever a leg or phase check has since bound. Both legs are resolved through the running
    ``bindings``, then unified against each other. Returns the (possibly refined) shared
    dimension, or a :class:`_ResolutionFailure` on ``FAILURE`` or a contradictory rebind.
    """
    resolved_a = _resolve_with_bindings(port_a_dim, bindings)
    resolved_b = _resolve_with_bindings(port_b_dim, bindings)
    result = resolved_a.unify(resolved_b)
    if result.is_failure:
        return _ResolutionFailure(_FailureReason.UNIFY_FAILURE, resolved_a, resolved_b)
    source = ConstraintSource.connecting_pair()
    bound_this_pass = result.is_success and bool(result.bindings)
    if result.is_deferred:
        record.record(source, resolved_a, resolved_b, ConstraintOutcome.DEFERRED)
    elif bound_this_pass:
        record.record(
            source,
            resolved_a,
            resolved_b,
            ConstraintOutcome.BOUND,
            bound_here=result.bindings,
        )
    else:
        record.record_identity(source)
    if bound_this_pass:
        if not _merge_bindings(bindings, result.bindings):
            return _ResolutionFailure(_FailureReason.CONTRADICTORY_REBIND, resolved_a, resolved_b)
        shared_dim = _resolve_with_bindings(shared_dim, result.bindings)
    return shared_dim


def _verify_fixpoint_closure(
    node_a: Node,
    node_b: Node,
    a_id: NodeId,
    b_id: NodeId,
    ref_a: PortRef,
    ref_b: PortRef,
    port_a_dim: Dim,
    port_b_dim: Dim,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
) -> bool:
    """Re-verify, from scratch, that the finished fixpoint's claim holds.

    The connecting pair's two legs, every surviving leg of both nodes, and every present
    phase -- each resolved under the final ``bindings`` -- must unify with the final
    ``shared_dim`` without ``FAILURE``.

    Called only on :func:`resolve_fusion_match`'s stabilised-convergence path, where a
    ``False`` return is unreachable: the loop's last pass changed neither ``shared_dim`` nor
    ``bindings``, so it already re-checked every one of these against this same state. A
    structural guard, pinned by direct call in
    ``tests/test_match.py::TestStructuralGuardsThatTheFixpointNeverReaches``.
    """
    for dim in (port_a_dim, port_b_dim):
        if _resolve_with_bindings(dim, bindings).unify(shared_dim).is_failure:
            return False
    for node, node_id, consumed_ref in ((node_a, a_id, ref_a), (node_b, b_id, ref_b)):
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index, port in enumerate(node.legs(direction)):
                ref = PortRef(node_id, direction, index)
                if ref == consumed_ref:
                    continue
                if _resolve_with_bindings(port.dim, bindings).unify(shared_dim).is_failure:
                    return False
    for phase in (node_a.phase, node_b.phase):
        if phase is None:
            continue
        if _resolve_with_bindings(phase.dim, bindings).unify(shared_dim).is_failure:
            return False
    return True


def _connecting_pair_detail(
    port_a_dim: Dim, port_b_dim: Dim, bindings: Mapping[str, Dim], record: _ConstraintRecord
) -> str:
    """Human-readable summary of the connecting pair's finished record entry.

    Every operand and binding is read off ``entry``, never recomputed against the final
    ``port_a_dim``/``port_b_dim``/``bindings``, which can have moved on since the pair's own
    check ran. Those three are used only when no entry was recorded at all -- the pair was a
    bare identity on every pass.
    """
    entry = record.entry_for(ConstraintSource.connecting_pair())
    if entry is None:
        resolved_a = _resolve_with_bindings(port_a_dim, bindings)
        resolved_b = _resolve_with_bindings(port_b_dim, bindings)
        return f"{resolved_a} == {resolved_b}"
    if entry.outcome is ConstraintOutcome.DEFERRED:
        return f"{entry.assumed} == {entry.equal_to} (deferred, assumed)"
    # DimensionConstraint.__post_init__ guarantees a BOUND entry carries a non-empty
    # bound_here, so the fall-through is reachable only for a genuine non-concrete binding.
    if entry.bound_here and all(value.is_concrete for _, value in entry.bound_here):
        binding_desc = ", ".join(f"{name} := {value}" for name, value in entry.bound_here)
        return f"{entry.assumed} == {entry.equal_to} (bound: {binding_desc})"
    return (
        f"{entry.assumed} == {entry.equal_to} (bound to a non-concrete Dim; left unused for "
        "shared-dimension resolution, see the module docstring's 'Non-concrete bindings')"
    )


def _dimension_agreement_outcome(
    port_a_dim: Dim,
    port_b_dim: Dim,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
    record: _ConstraintRecord,
) -> SideConditionOutcome:
    """Build condition 7's (``dimension_agreement``) passing outcome from a leg-sweep state.

    Shared by :func:`resolve_fusion_match`'s stabilised-success and phase-failure paths, both
    reporting from the leg sweep's own ``shared_dim``/``bindings``, never from a state a
    later phase check has advanced past.
    """
    leg_detail = _connecting_pair_detail(port_a_dim, port_b_dim, bindings, record)
    leg_constraint_count = record.leg_count()
    return SideConditionOutcome(
        "dimension_agreement",
        True,
        leg_detail
        + (
            ""
            if not leg_constraint_count
            else (
                f"; surviving leg(s) resolved to shared_dim={shared_dim} with "
                f"{leg_constraint_count} additional assumed dimension equality/ies"
            )
        ),
        deferred=record.any_leg_deferred(),
    )


def reattach_phase(
    phase: PhaseVector, shared_dim: Dim, bindings: Mapping[str, Dim]
) -> tuple[PhaseVector, Mapping[str, Dim]]:
    """Substitute ``bindings`` into ``phase``'s entries, then reattach to ``shared_dim``.

    Returns the reattached vector together with the subset of ``bindings`` actually
    substituted into an entry's value. Public: :func:`resolve_fusion_match` calls it as a
    trial construction and :mod:`archytaszx.rewrite.rules_library`'s builder to build the merged
    phase, so both decide reattachability the same way.

    Raises :class:`~archytaszx.algebra.phase.PhaseDomainError` if, after substitution, an entry's
    index falls outside ``shared_dim``'s range.
    """
    concrete_bindings = {name: dim.to_int() for name, dim in bindings.items() if dim.is_concrete}
    substituted = (
        phase.substitute(cast(Mapping[PhaseSymbolKey, PhaseSubstituteValue], concrete_bindings))
        if concrete_bindings
        else phase
    )
    entry_symbols: set[str] = set()
    for entry in phase.entries().values():
        entry_symbols |= entry.free_symbols
    applied = {name: bindings[name] for name in concrete_bindings if name in entry_symbols}
    return PhaseVector(shared_dim, substituted.entries()), MappingProxyType(applied)


def _validate_wire_endpoint(
    diagram: Diagram, wire_or_boundary_ref: Wire | PortRef, ref: PortRef
) -> None:
    """Raise ``RewriteGrammarError`` if ``ref`` names an unknown node or out-of-range index.

    Called for both endpoints of every wire and for every boundary entry.
    ``wire_or_boundary_ref`` only phrases the message: the enclosing ``Wire`` at a
    wire-endpoint call site, the bare ``PortRef`` at a boundary one.
    """
    node = diagram.nodes.get(ref.node_id)
    if node is None:
        if isinstance(wire_or_boundary_ref, Wire):
            context = f"wire {wire_or_boundary_ref!r}"
            explanation = "a live wire can never legitimately name a removed node"
        else:
            context = f"boundary entry {wire_or_boundary_ref!r}"
            explanation = "a live boundary entry can never legitimately name a removed node"
        raise RewriteGrammarError(
            f"{context} references node id {ref.node_id!r} absent from the diagram; "
            f"Diagram.remove_node cascades, so {explanation}"
        )
    legs = node.legs(ref.direction)
    if ref.index >= len(legs):
        kind = "wire endpoint" if isinstance(wire_or_boundary_ref, Wire) else "boundary entry"
        raise RewriteGrammarError(
            f"{kind} {ref!r} is out of range for node {ref.node_id!r}: it has only "
            f"{len(legs)} {ref.direction.value} leg(s)"
        )


@dataclass(frozen=True, slots=True)
class FusionResolution:
    """The result of the one verification predicate behind :data:`FUSION_SIDE_CONDITIONS`.

    Computed fresh from ``(diagram, a_id, b_id, wire)`` alone, never from a pre-existing
    :class:`FusionMatch`'s fields. ``outcomes`` always covers exactly the
    :data:`FUSION_SIDE_CONDITIONS` names in declared order -- a condition an earlier failure
    stopped it reaching is recorded as a failing outcome whose detail says so.

    When ``passed``, ``shared_dim``, ``bindings`` and ``dimension_constraints`` are the ground
    truth to build a merged node from. When not, ``shared_dim`` is ``None`` and the other two
    are empty.
    """

    passed: bool
    shared_dim: Dim | None
    bindings: Mapping[str, Dim]
    dimension_constraints: tuple[DimensionConstraint, ...]
    outcomes: tuple[SideConditionOutcome, ...]


def _connecting_pair_failure_detail(failure: _ResolutionFailure) -> str:
    """Render a connecting-pair :class:`_ResolutionFailure`, distinguishing its two causes."""
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"{failure.assumed} == {failure.equal_to} unifies, but only by binding a "
            "symbol to a value that contradicts an earlier binding accumulated this "
            "resolution (see _merge_bindings)"
        )
    return f"{failure.assumed} != {failure.equal_to}: the connecting pair does not unify"


def _leg_failure_detail(side: str, failure: _ResolutionFailure) -> str:
    """Render a surviving-leg :class:`_ResolutionFailure` for node ``side`` ('A' or 'B')."""
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"a surviving leg of the {side}-side node ({failure.assumed}) unifies with "
            f"shared_dim ({failure.equal_to}) only by binding a symbol to a value that "
            "contradicts an earlier binding accumulated this resolution"
        )
    return (
        f"a surviving leg of the {side}-side node ({failure.assumed}) does not unify with "
        f"shared_dim ({failure.equal_to})"
    )


def _phase_failure_detail(failure: _ResolutionFailure) -> str:
    """Render an in-loop phase-dimension :class:`_ResolutionFailure`, by cause.

    Four causes (module docstring, condition 8): a ``FAILURE``, a ``DEFERRED`` unify, a
    binding to a non-concrete ``Dim``, and a contradictory rebind. The post-loop
    ``reattach_phase`` failure is rendered at its own call site in
    :func:`resolve_fusion_match`.
    """
    if failure.reason is _FailureReason.PHASE_DEFERRED:
        return (
            f"a present phase dimension ({failure.assumed}) unifies only as DEFERRED "
            f"against the resolved shared leg dimension {failure.equal_to} -- a DEFERRED "
            "unify is not accepted for a phase, see the module docstring, condition 8"
        )
    if failure.reason is _FailureReason.PHASE_NON_CONCRETE_BINDING:
        return (
            f"a present phase dimension ({failure.assumed}) unifies with the resolved "
            f"shared leg dimension {failure.equal_to} only by binding to a non-concrete "
            "Dim -- not accepted for a phase, see the module docstring's 'Non-concrete "
            "bindings' section"
        )
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"a present phase dimension ({failure.assumed}) unifies with the resolved "
            f"shared leg dimension {failure.equal_to}, but only by binding a symbol to a "
            "value that contradicts an earlier binding accumulated this resolution"
        )
    return (
        f"a present phase dimension ({failure.assumed}) does not unify with the resolved "
        f"shared leg dimension {failure.equal_to}"
    )


def _consumed_port_claim_conflict(
    diagram: Diagram, ref: PortRef, consuming_wire: Wire
) -> str | None:
    """``None`` if ``ref`` is claimed only by ``consuming_wire`` and is on no boundary list.

    Otherwise a description of what else claims it: a second wire, a boundary entry, or both.
    Recomputed from ``diagram`` alone on every call.
    """
    other_wire_claims = sum(
        1 for wire in diagram.wires if wire != consuming_wire and ref in (wire.a, wire.b)
    )
    on_boundary = ref in diagram.boundary_inputs or ref in diagram.boundary_outputs
    if not other_wire_claims and not on_boundary:
        return None
    parts = []
    if other_wire_claims:
        parts.append(f"claimed by {other_wire_claims} other wire(s)")
    if on_boundary:
        parts.append("listed on a boundary list")
    return f"{ref} is " + " and ".join(parts)


def innermost_node_scope_box(diagram: Diagram, node_id: NodeId) -> BangBoxId | None:
    """The most specific node-scope bang box containing ``node_id``, or ``None`` (Phase 7).

    "Most specific" means smallest ``node_scope``, tie-broken by id -- Phase 7 never
    needs node-scope-inside-node-scope nesting, so this is a simple, deterministic
    fallback rather than a real depth computation.
    """
    candidates = [
        box
        for box in diagram.bang_boxes.values()
        if box.is_node_scope and node_id in box.node_scope
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda b: (len(b.node_scope), b.id)).id


_GUARD_CONDITION_NAME = "dimension_guards_satisfied"
"""The name of the Phase 10 side condition :func:`_dimension_guards_outcome` reports."""


def _dimension_guards_outcome(
    guards: tuple[DimensionGuard, ...], shared_dim: Dim
) -> SideConditionOutcome:
    """Evaluate every guard at ``shared_dim``. ``UNDECIDED`` blocks, with ``deferred`` False."""
    if not guards:
        return SideConditionOutcome(
            _GUARD_CONDITION_NAME, True, "no dimension guards declared", deferred=False
        )
    verdicts = [(guard, guard.evaluate(shared_dim)) for guard in guards]
    failed = [guard for guard, verdict in verdicts if verdict is GuardOutcome.FAILED]
    if failed:
        detail = (
            f"dimension guard(s) {', '.join(str(guard) for guard in failed)} do not hold at "
            f"the matched shared dimension {shared_dim}"
        )
        return SideConditionOutcome(_GUARD_CONDITION_NAME, False, detail, deferred=False)
    undecided = [guard for guard, verdict in verdicts if verdict is GuardOutcome.UNDECIDED]
    if undecided:
        detail = (
            f"dimension guard(s) {', '.join(str(guard) for guard in undecided)} cannot be "
            f"decided at the non-concrete matched shared dimension {shared_dim}, so the "
            "rewrite is blocked rather than assumed"
        )
        return SideConditionOutcome(_GUARD_CONDITION_NAME, False, detail, deferred=False)
    detail = (
        f"dimension guard(s) {', '.join(str(guard) for guard in guards)} hold at the matched "
        f"shared dimension {shared_dim}"
    )
    return SideConditionOutcome(_GUARD_CONDITION_NAME, True, detail, deferred=False)


def resolve_fusion_match(
    diagram: Diagram,
    a_id: NodeId,
    b_id: NodeId,
    wire: Wire,
    *,
    dimension_guards: tuple[DimensionGuard, ...] = (),
) -> FusionResolution:
    """Decide, from ``diagram`` alone, whether ``wire`` is a legal fusion of ``a_id``/``b_id``.

    The single shared predicate behind every condition in the module docstring.
    :func:`find_matches` calls it once per candidate wire;
    :func:`~archytaszx.rewrite.rules_library.spider_fusion_builder` calls it again, fresh, and
    trusts only its return value for graph surgery.

    Raises :class:`~archytaszx.rewrite.rule.RewriteGrammarError` for a request that cannot be
    evaluated at all: ``a_id == b_id``, either node id absent from ``diagram``, ``wire`` not
    incident on both, ``wire`` not an element of ``diagram.wires``, or either endpoint naming
    an unknown node id or out-of-range port index.

    Never mutates ``diagram``.
    """
    if a_id == b_id:
        raise RewriteGrammarError(
            f"resolve_fusion_match: a_id and b_id must be distinct, both were {a_id!r}"
        )
    node_a = diagram.nodes.get(a_id)
    node_b = diagram.nodes.get(b_id)
    if node_a is None or node_b is None:
        raise RewriteGrammarError(
            f"resolve_fusion_match: node id(s) {a_id!r}, {b_id!r} not both present in the diagram"
        )
    wire_node_ids = {wire.a.node_id, wire.b.node_id}
    if wire_node_ids != {a_id, b_id}:
        raise RewriteGrammarError(
            f"resolve_fusion_match: wire {wire!r} does not connect {a_id!r} and {b_id!r} "
            f"(it connects {sorted(wire_node_ids)!r})"
        )
    _validate_wire_endpoint(diagram, wire, wire.a)
    _validate_wire_endpoint(diagram, wire, wire.b)

    # A freestanding Wire built against two real, correctly-incident ports would otherwise
    # reach graph surgery. Snapshotted once: Diagram.wires rebuilds on every access.
    all_wires = diagram.wires
    if wire not in all_wires:
        raise RewriteGrammarError(
            f"resolve_fusion_match: wire {wire!r} is not an element of diagram.wires -- "
            "a wire naming two correctly-incident ports is not itself proof that the "
            "diagram actually contains it"
        )

    ref_a = wire.a if wire.a.node_id == a_id else wire.b
    ref_b = wire.b if wire.a.node_id == a_id else wire.a

    other_wire_count = sum(
        1
        for other in all_wires
        if other != wire and {other.a.node_id, other.b.node_id} == {a_id, b_id}
    )

    outcomes: list[SideConditionOutcome] = [
        SideConditionOutcome("distinct_nodes", True, f"{a_id!r} != {b_id!r}"),
    ]

    # Identical generator type and that type being fusable are both decided under
    # same_generator_type.
    generator_types_match = node_a.generator_type == node_b.generator_type
    is_fusable_type = (
        generator_types_match
        and REGISTRY.is_registered(node_a.generator_type)
        and node_a.generator_type.name in _FUSABLE_GENERATOR_NAMES
    )
    if not generator_types_match:
        same_type_detail = (
            f"{a_id!r} is {node_a.generator_type.name!r} but {b_id!r} is "
            f"{node_b.generator_type.name!r}"
        )
    elif not is_fusable_type:
        same_type_detail = (
            f"both nodes are {node_a.generator_type.name!r}, but that is not a registered "
            "fusable generator type (the name is unregistered, or names a registered type "
            "whose fields differ from this one's)"
        )
    else:
        same_type_detail = f"both nodes are {node_a.generator_type.name!r}"
    same_type = is_fusable_type
    outcomes.append(SideConditionOutcome("same_generator_type", same_type, same_type_detail))

    outcomes.append(
        SideConditionOutcome(
            "parallel_wires_become_self_loops",
            True,
            f"{other_wire_count} other wire(s) join the two nodes, surviving as "
            "self-loop(s) on the merged spider",
        )
    )

    def _failed(remaining_names: tuple[str, ...], reason: str = "") -> FusionResolution:
        for name in remaining_names:
            outcomes.append(SideConditionOutcome(name, False, reason))
        if not any(outcome.name == _GUARD_CONDITION_NAME for outcome in outcomes):
            outcomes.append(SideConditionOutcome(_GUARD_CONDITION_NAME, False, reason))
        return FusionResolution(
            passed=False,
            # None, not a placeholder Dim: a failed resolution has no shared dimension a
            # caller may read. See FusionResolution's docstring.
            shared_dim=None,
            bindings=MappingProxyType({}),
            dimension_constraints=(),
            outcomes=tuple(outcomes),
        )

    if not same_type:
        return _failed(
            (
                "consumed_wire_direction_permitted_for_color",
                "consumed_ports_singly_claimed",
                "bang_box_scope_agreement",
                "dimension_agreement",
                "phase_dimension_agreement",
            ),
            "not evaluated: same_generator_type failed first",
        )

    # same_type guarantees the pair is registered fusable and same-typed.
    same_direction = ref_a.direction == ref_b.direction
    direction_ok = (
        not same_direction or node_a.generator_type.name in _SAME_DIRECTION_FUSABLE_GENERATOR_NAMES
    )

    direction_detail = (
        f"{ref_a} (direction={ref_a.direction.value}) -> "
        f"{ref_b} (direction={ref_b.direction.value})"
    )
    if same_direction:
        direction_detail += (
            f" (same-direction {ref_a.direction.value}-{ref_b.direction.value} wire, "
            f"permitted for {node_a.generator_type.name!r} only -- see the module "
            "docstring, condition 4)"
            if direction_ok
            else (
                f" (same-direction {ref_a.direction.value}-{ref_b.direction.value} wire "
                f"is not permitted for {node_a.generator_type.name!r} -- see the module "
                "docstring, condition 4)"
            )
        )
    outcomes.append(
        SideConditionOutcome(
            "consumed_wire_direction_permitted_for_color", direction_ok, direction_detail
        )
    )

    if not direction_ok:
        return _failed(
            (
                "consumed_ports_singly_claimed",
                "bang_box_scope_agreement",
                "dimension_agreement",
                "phase_dimension_agreement",
            ),
            "not evaluated: consumed_wire_direction_permitted_for_color failed first",
        )

    # Condition 5. A candidate that fails it has no legal port_mapping regardless of what
    # the fixpoint would find, so it is checked before paying for one.
    claim_conflict_a = _consumed_port_claim_conflict(diagram, ref_a, wire)
    claim_conflict_b = _consumed_port_claim_conflict(diagram, ref_b, wire)
    claims_ok = claim_conflict_a is None and claim_conflict_b is None
    claim_detail = (
        f"neither {ref_a} nor {ref_b} is claimed by another wire or listed on a boundary"
        if claims_ok
        else "; ".join(d for d in (claim_conflict_a, claim_conflict_b) if d is not None)
    )
    outcomes.append(SideConditionOutcome("consumed_ports_singly_claimed", claims_ok, claim_detail))
    if not claims_ok:
        return _failed(
            ("bang_box_scope_agreement", "dimension_agreement", "phase_dimension_agreement"),
            "not evaluated: consumed_ports_singly_claimed failed first",
        )

    # Condition 6 (Phase 7). Purely structural -- decidable from diagram.bang_boxes alone,
    # no dimension-unification work -- so it runs before the fixpoint below for the same
    # reason condition 5 runs before it: a candidate failing it has no legal fusion
    # regardless of what dimension unification would find.
    box_a = innermost_node_scope_box(diagram, a_id)
    box_b = innermost_node_scope_box(diagram, b_id)
    scope_ok = box_a == box_b
    if scope_ok:
        scope_detail = (
            f"both nodes are inside bang box {box_a!r}"
            if box_a is not None
            else "neither node is inside a bang box"
        )
    else:
        scope_detail = (
            f"{a_id!r} is inside bang box {box_a!r} but {b_id!r} is inside {box_b!r} -- "
            "fusion cannot span two different bang-box scopes (or a scope boundary), "
            "since that would fix what should be a per-copy rewrite to a one-off merge"
        )
    outcomes.append(SideConditionOutcome("bang_box_scope_agreement", scope_ok, scope_detail))
    if not scope_ok:
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            "not evaluated: bang_box_scope_agreement failed first",
        )

    legs_a = node_a.legs(ref_a.direction)
    legs_b = node_b.legs(ref_b.direction)
    port_a = legs_a[ref_a.index]
    port_b = legs_b[ref_b.index]

    record = _ConstraintRecord()
    # Seeded from the A-side consumed leg, A being the lower NodeId. Observable only when
    # the connecting pair merely DEFERS; see FusionMatch and
    # tests/test_match.py::TestSharedDimSeedComesFromTheLowerIdNode.
    shared_dim = port_a.dim
    bindings: dict[str, Dim] = {}

    # An equality the diagram's own parameter environment refutes is decided, not deferred:
    # the two spiders do not share a dimension, so this is a non-match.
    resolved_a = diagram.resolve_dim(port_a.dim)
    resolved_b = diagram.resolve_dim(port_b.dim)
    if (resolved_a, resolved_b) != (port_a.dim, port_b.dim) and resolved_a.unify(
        resolved_b
    ).is_failure:
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            f"the parameter environment resolves {port_a.dim} and {port_b.dim} to "
            f"{resolved_a} and {resolved_b}, which do not unify",
        )

    # Conditions 6 and 7 run as one bounded fixpoint: each pass re-derives the connecting
    # pair, then every surviving leg of both nodes, then every present phase's Dim -- each
    # against shared_dim as of the point reached so far in that pass, refining `bindings` and
    # shared_dim in place on any concrete binding.
    #
    # The exit condition is a full pass that adds nothing to *either* shared_dim or bindings.
    # Stopping on shared_dim alone is unsound: bindings can grow on a pass whose new binding
    # touches no symbol in shared_dim, leaving what was checked earlier in that pass
    # unre-checked against it -- which is how an unsatisfiable set like e*f == 2, e == 2,
    # f == 2 escapes. Pinned by
    # tests/test_match.py::TestFixpointExitRequiresBothToStabilise.
    fixpoint_budget_exhausted = False

    for _pass_index in range(_MAX_FIXPOINT_PASSES):
        pass_start_dim = shared_dim
        pass_start_bindings = dict(bindings)

        next_dim = _unify_connecting_pair(port_a.dim, port_b.dim, shared_dim, bindings, record)
        if isinstance(next_dim, _ResolutionFailure):
            return _failed(
                ("dimension_agreement", "phase_dimension_agreement"),
                _connecting_pair_failure_detail(next_dim),
            )
        shared_dim = next_dim

        next_dim = _unify_surviving_legs(node_a, a_id, ref_a, shared_dim, bindings, record)
        if isinstance(next_dim, _ResolutionFailure):
            return _failed(
                ("dimension_agreement", "phase_dimension_agreement"),
                _leg_failure_detail("A", next_dim),
            )
        shared_dim = next_dim

        next_dim = _unify_surviving_legs(node_b, b_id, ref_b, shared_dim, bindings, record)
        if isinstance(next_dim, _ResolutionFailure):
            return _failed(
                ("dimension_agreement", "phase_dimension_agreement"),
                _leg_failure_detail("B", next_dim),
            )
        shared_dim = next_dim

        # Snapshotted before _unify_phase_dims, which can bind phase A's symbol -- refining
        # both in place -- before failing on phase B in the same call. Condition 6 must be
        # reported against the state its own leg sweep was verified at.
        leg_verified_shared_dim = shared_dim
        leg_verified_bindings = dict(bindings)

        phase_result = _unify_phase_dims(node_a, node_b, a_id, b_id, shared_dim, bindings, record)
        if isinstance(phase_result, _ResolutionFailure):
            # Reported directly rather than falling through to _verify_fixpoint_closure,
            # whose unreachability argument holds only on the convergence path.
            # dimension_agreement is True from the leg-verified snapshot above -- a leg-sweep
            # FAILURE already returned via _failed.
            outcomes.append(
                _dimension_agreement_outcome(
                    port_a.dim, port_b.dim, leg_verified_shared_dim, leg_verified_bindings, record
                )
            )
            phase_detail = _phase_failure_detail(phase_result)
            outcomes.append(
                SideConditionOutcome(
                    "phase_dimension_agreement", False, phase_detail, deferred=False
                )
            )
            # Same failure convention as _failed(), but not routed through it: _failed marks
            # every remaining name False and cannot express this path's mix. The guard
            # outcome is appended here for the same coverage reason _failed appends it.
            outcomes.append(
                SideConditionOutcome(_GUARD_CONDITION_NAME, False, phase_detail, deferred=False)
            )
            return FusionResolution(
                passed=False,
                shared_dim=None,
                bindings=MappingProxyType({}),
                dimension_constraints=(),
                outcomes=tuple(outcomes),
            )
        shared_dim = phase_result

        if shared_dim == pass_start_dim and bindings == pass_start_bindings:
            break
    else:
        # The fixpoint decides conditions 7 and 8 jointly, so when it does not terminate
        # neither was decided, and both report the same budget-exhaustion detail.
        fixpoint_budget_exhausted = True

    if fixpoint_budget_exhausted:
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            f"the bounded leg/phase resolution fixpoint did not stabilise within "
            f"{_MAX_FIXPOINT_PASSES} passes (_MAX_FIXPOINT_PASSES): a resolver iteration "
            "budget, not a dimension or phase-dimension disagreement -- neither condition "
            "was decided",
        )

    # Reached only via the loop's own convergence break: a phase failure and budget
    # exhaustion both return directly above.
    if not _verify_fixpoint_closure(
        node_a, node_b, a_id, b_id, ref_a, ref_b, port_a.dim, port_b.dim, shared_dim, bindings
    ):
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            "post-loop closure check failed: a resolved leg, phase, or the connecting pair "
            "does not unify with the final shared_dim under the final bindings (see "
            "_verify_fixpoint_closure) -- a structural guard, unreachable on the convergence "
            "path this call site is reached from",
        )

    outcomes.append(
        _dimension_agreement_outcome(port_a.dim, port_b.dim, shared_dim, bindings, record)
    )

    phase_a_dim = node_a.phase.dim if node_a.phase is not None else None
    phase_b_dim = node_b.phase.dim if node_b.phase is not None else None
    phase_dims_present = tuple(d for d in (phase_a_dim, phase_b_dim) if d is not None)

    # Against the final, post-fixpoint shared_dim, so an entry falling out of range only
    # once the later bindings resolve is still caught. A different failure from the in-loop
    # phase-dim ones: a *unifying* Dim whose entries fall out of range once substituted.
    for phase in (node_a.phase, node_b.phase):
        if phase is None:
            continue
        try:
            reattach_phase(phase, shared_dim, bindings)
        except PhaseDomainError as exc:
            phase_detail = (
                f"a present phase dimension unifies with the resolved shared leg dimension "
                f"{shared_dim}, but at least one of its own entries falls out of range once "
                f"every binding this fixpoint accumulated is substituted in ({exc})"
            )
            outcomes.append(
                SideConditionOutcome(
                    "phase_dimension_agreement", False, phase_detail, deferred=False
                )
            )
            return _failed(())

    # Read off record.entries(), in first-derivation order; a name a later pass re-bound
    # keeps its one dict entry, at that later value.
    phase_bound_values: dict[str, Dim] = {}
    for phase_entry in record.entries():
        if (
            phase_entry.source.kind is ConstraintSourceKind.NODE_PHASE
            and phase_entry.outcome is ConstraintOutcome.BOUND
        ):
            phase_bound_values.update(phase_entry.bound_here)
    unique_bound_names = list(phase_bound_values)
    phase_detail = (
        "no phase present on either node"
        if not phase_dims_present
        else (
            "present phase dimension(s) unify with the resolved shared leg dimension "
            f"{shared_dim}"
            + (
                "; assuming "
                + ", ".join(f"{name} := {phase_bound_values[name]}" for name in unique_bound_names)
                if unique_bound_names
                else ""
            )
        )
    )
    outcomes.append(
        # Always deferred=False: a DEFERRED phase-dim unify is rejected outright, so a
        # passing outcome rests at most on a binding, which dimension_constraints records.
        SideConditionOutcome("phase_dimension_agreement", True, phase_detail, deferred=False)
    )

    guard_outcome = _dimension_guards_outcome(dimension_guards, shared_dim)
    outcomes.append(guard_outcome)
    if not guard_outcome.passed:
        return FusionResolution(
            passed=False,
            shared_dim=None,
            bindings=MappingProxyType({}),
            dimension_constraints=(),
            outcomes=tuple(outcomes),
        )
    return FusionResolution(
        passed=True,
        shared_dim=shared_dim,
        bindings=MappingProxyType(dict(bindings)),
        dimension_constraints=record.entries(),
        outcomes=tuple(outcomes),
    )


def _support_ids(*node_ids: NodeId) -> tuple[NodeId, ...]:
    """The given node ids, deduplicated and ascending."""
    return tuple(sorted(set(node_ids)))


def _neighbour_ids(diagram: Diagram) -> dict[NodeId, set[NodeId]]:
    """Each node id mapped to the node ids one wire hop away."""
    neighbours: dict[NodeId, set[NodeId]] = {node_id: set() for node_id in diagram.nodes}
    for wire in diagram.wires:
        neighbours.setdefault(wire.a.node_id, set()).add(wire.b.node_id)
        neighbours.setdefault(wire.b.node_id, set()).add(wire.a.node_id)
    return neighbours


def _within_hops(
    neighbours: Mapping[NodeId, set[NodeId]], seeds: Iterable[NodeId], hops: int
) -> set[NodeId]:
    """Every node id reachable from ``seeds`` in at most ``hops`` wire hops."""
    reached = set(seeds)
    frontier = set(reached)
    for _ in range(hops):
        nxt = {other for node_id in frontier for other in neighbours.get(node_id, ())} - reached
        if not nxt:
            break
        reached |= nxt
        frontier = nxt
    return reached


def _seed_meets_anchors(
    anchors: frozenset[NodeId] | None,
    neighbours: Mapping[NodeId, set[NodeId]] | None,
    seeds: tuple[NodeId, ...],
    hops: int,
) -> bool:
    """True when ``anchors`` is None, or meets the ``hops``-hop closure of ``seeds``."""
    if anchors is None:
        return True
    if hops == 0 or neighbours is None:
        return not anchors.isdisjoint(seeds)
    return not anchors.isdisjoint(_within_hops(neighbours, seeds, hops))


def find_matches(
    diagram: Diagram,
    *,
    dimension_guards: tuple[DimensionGuard, ...] = (),
    anchors: frozenset[NodeId] | None = None,
) -> tuple[FusionMatch, ...]:
    """Find every same-color spider fusion occurrence in ``diagram``. See the module docstring.

    Never mutates ``diagram``, and does not require it to be well-formed --
    :func:`~archytaszx.diagram.validate.validate` is never called here. Returns matches sorted by
    ``(a_id, b_id)``, tiebroken by the consumed wire's own per-side (direction, index).
    """
    # Malformed-wire detection is independent of every other property of the wire or its
    # candidate pair, so both endpoints of every wire are checked before any grouping.
    #
    # Sorted, not the raw frozenset: Wire's hash folds in Direction's member-name hash, which
    # is PYTHONHASHSEED-dependent, and the pass below raises on the first offending wire.
    # Snapshotted once, since Diagram.wires rebuilds on every access.
    wires = tuple(sorted(diagram.wires, key=lambda w: w.sort_key()))

    for wire in wires:
        _validate_wire_endpoint(diagram, wire, wire.a)
        _validate_wire_endpoint(diagram, wire, wire.b)

    # A boundary entry is held to the same standard as a wire endpoint: engine's
    # _remap_endpoint treats both identically once a match reaches apply. Both lists are
    # walked in declared order, boundary_inputs first.
    for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs):
        _validate_wire_endpoint(diagram, ref, ref)

    # No filter for a multiply-claimed consumed port: that is condition 5
    # (consumed_ports_singly_claimed), decided by resolve_fusion_match.
    candidates_by_pair: dict[frozenset[NodeId], list[Wire]] = {}
    for wire in wires:
        if wire.a.node_id == wire.b.node_id:
            continue
        key = frozenset((wire.a.node_id, wire.b.node_id))
        candidates_by_pair.setdefault(key, []).append(wire)

    # Condition 3's other-wire count is recomputed inside resolve_fusion_match from diagram
    # alone, not threaded through here.
    wire_candidates = [
        wire for connecting_wires in candidates_by_pair.values() for wire in connecting_wires
    ]

    matches: list[FusionMatch] = []
    for wire in wire_candidates:
        a_id, b_id = _ordered_pair(wire)
        if not _seed_meets_anchors(anchors, None, (a_id, b_id), 0):
            continue

        # Conditions 2 and 4-7 are decided by exactly this call -- the same function
        # spider_fusion_builder calls again to re-verify the match.
        resolution = resolve_fusion_match(
            diagram, a_id, b_id, wire, dimension_guards=dimension_guards
        )
        if not resolution.passed:
            continue
        assert resolution.shared_dim is not None  # invariant: passed implies shared_dim is set

        matches.append(
            FusionMatch(
                a_id=a_id,
                b_id=b_id,
                wire=wire,
                shared_dim=resolution.shared_dim,
                side_condition_outcomes=resolution.outcomes,
                dimension_constraints=resolution.dimension_constraints,
                bindings=MappingProxyType(dict(resolution.bindings)),
            )
        )

    matches.sort(
        key=lambda m: (
            int(m.a_id),
            int(m.b_id),
            (m.wire.a if m.wire.a.node_id == m.a_id else m.wire.b).direction.value,
            (m.wire.a if m.wire.a.node_id == m.a_id else m.wire.b).index,
            (m.wire.b if m.wire.a.node_id == m.a_id else m.wire.a).direction.value,
            (m.wire.b if m.wire.a.node_id == m.a_id else m.wire.a).index,
        )
    )
    return tuple(matches)


def _ordered_pair(wire: Wire) -> tuple[NodeId, NodeId]:
    """The wire's two node ids as ``(lower, higher)``. See :class:`FusionMatch`."""
    if wire.a.node_id <= wire.b.node_id:
        return wire.a.node_id, wire.b.node_id
    return wire.b.node_id, wire.a.node_id


@dataclass(frozen=True, slots=True)
class FusionPattern(Pattern):
    """Same-color spider fusion, optionally restricted by :class:`DimensionGuard`\\ s.

    Locality radius 1.
    """

    dimension_guards: tuple[DimensionGuard, ...] = ()

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_matches`. See the module docstring."""
        return find_matches(diagram, dimension_guards=self.dimension_guards)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_matches` with the candidate wires restricted to ``anchors``."""
        return find_matches(diagram, dimension_guards=self.dimension_guards, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The pair's node ids, tiebroken by the consumed wire's per-side direction and index."""
        fusion = cast(FusionMatch, match)
        near = fusion.wire.a if fusion.wire.a.node_id == fusion.a_id else fusion.wire.b
        far = fusion.wire.b if fusion.wire.a.node_id == fusion.a_id else fusion.wire.a
        return (
            int(fusion.a_id),
            int(fusion.b_id),
            near.direction.value,
            near.index,
            far.direction.value,
            far.index,
        )


FOURIER_CHAIN_LENGTH = 4
"""How many Fourier boxes in series the cancellation pattern consumes: F^4 is the identity."""


FOURIER_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "chain_is_series",
        "four F boxes joined output-to-input in series, each internal node used only there",
    ),
    SideCondition("same_dimension", "every leg of the four boxes carries one dimension"),
)


@dataclass(frozen=True, slots=True)
class FourierMatch:
    """One located F^4 chain: its four node ids in series order, its three internal wires."""

    node_ids: tuple[NodeId, ...]
    wires: tuple[Wire, ...]
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The chain's four node ids, ascending."""
        return _support_ids(*self.node_ids)


def _wire_by_port(diagram: Diagram) -> dict[PortRef, Wire]:
    """Map every wired port to the wire carrying it."""
    mapping: dict[PortRef, Wire] = {}
    for wire in diagram.wires:
        mapping[wire.a] = wire
        mapping[wire.b] = wire
    return mapping


def _fourier_successor(
    diagram: Diagram, node_id: NodeId, by_port: Mapping[PortRef, Wire]
) -> tuple[NodeId, Wire] | None:
    """The F box wired to this one's single output, and the wire joining them."""
    wire = by_port.get(PortRef(node_id, Direction.OUTPUT, 0))
    if wire is None:
        return None
    other = wire.b if wire.a.node_id == node_id else wire.a
    if other.node_id == node_id or other.direction is not Direction.INPUT or other.index != 0:
        return None
    successor = diagram.nodes[other.node_id]
    if successor.generator_type.name != FOURIER_BOX.name:
        return None
    return other.node_id, wire


def find_fourier_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[FourierMatch, ...]:
    """Every chain of four Fourier boxes in series, ordered by the chain's first node id."""
    by_port = _wire_by_port(diagram)
    matches: list[FourierMatch] = []
    for start in sorted(diagram.nodes):
        node = diagram.nodes[start]
        if not REGISTRY.is_registered(node.generator_type):
            continue
        if node.generator_type.name != FOURIER_BOX.name:
            continue
        chain = [start]
        wires: list[Wire] = []
        while len(chain) < FOURIER_CHAIN_LENGTH:
            step = _fourier_successor(diagram, chain[-1], by_port)
            if step is None:
                break
            next_id, wire = step
            if next_id in chain:
                break
            chain.append(next_id)
            wires.append(wire)
        if len(chain) != FOURIER_CHAIN_LENGTH:
            continue
        dims = [
            diagram.nodes[node_id].legs(direction)[0].dim
            for node_id in chain
            for direction in (Direction.OUTPUT, Direction.INPUT)
        ]
        shared = dims[0]
        constraints: list[DimensionConstraint] = []
        failed = False
        for other in dims[1:]:
            if other == shared:
                continue
            result = shared.unify(other)
            if result.is_failure:
                failed = True
                break
            constraints.append(
                DimensionConstraint(
                    assumed=shared,
                    equal_to=other,
                    source=ConstraintSource.connecting_pair(),
                    outcome=ConstraintOutcome.DEFERRED,
                )
            )
        if failed:
            continue
        # Applied here, downstream of every step that can raise on a malformed diagram, so
        # an anchored scan raises exactly what the full scan raises.
        if anchors is not None and anchors.isdisjoint(chain):
            continue
        outcomes = (
            SideConditionOutcome(
                "chain_is_series", True, f"{FOURIER_CHAIN_LENGTH} F boxes in series at {chain[0]}"
            ),
            SideConditionOutcome(
                "same_dimension",
                True,
                f"every leg carries {shared}",
                deferred=bool(constraints),
            ),
        )
        matches.append(
            FourierMatch(
                node_ids=tuple(chain),
                wires=tuple(wires),
                shared_dim=shared,
                side_condition_outcomes=outcomes,
                dimension_constraints=tuple(constraints),
            )
        )
    return tuple(matches)


class FourierCancellationPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for F^4 cancellation.

    Locality radius 3.
    """

    locality_radius = 3

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_fourier_matches`."""
        return find_fourier_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_fourier_matches` with the resolved chains restricted."""
        return find_fourier_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The chain's first node id."""
        return (int(cast(FourierMatch, match).node_ids[0]),)


CAP_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("state_is_a_z_spider", "a Z spider with no input and one output"),
    SideCondition("effect_is_an_x_spider", "an X spider with one input and no output"),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("phaseless", "neither node carries a phase vector"),
    SideCondition("same_dimension", "both legs carry one dimension"),
)


@dataclass(frozen=True, slots=True)
class CapMatch:
    """One located Z-state-into-X-effect cap: its two node ids and the wire joining them."""

    state_id: NodeId
    effect_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The state's and the effect's node ids, ascending."""
        return _support_ids(self.state_id, self.effect_id)


def find_cap_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[CapMatch, ...]:
    """Every phaseless Z state wired into a phaseless X effect, ordered by the state's node id."""
    claimed: dict[PortRef, int] = {}
    for wire in diagram.wires:
        claimed[wire.a] = claimed.get(wire.a, 0) + 1
        claimed[wire.b] = claimed.get(wire.b, 0) + 1
    boundary = set(diagram.boundary_inputs) | set(diagram.boundary_outputs)

    matches: list[CapMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for state_ref, effect_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if state_ref.direction is not Direction.OUTPUT:
                continue
            state = diagram.nodes.get(state_ref.node_id)
            effect = diagram.nodes.get(effect_ref.node_id)
            if state is None or effect is None or state is effect:
                continue
            if not REGISTRY.is_registered(state.generator_type):
                continue
            if not REGISTRY.is_registered(effect.generator_type):
                continue
            if state.generator_type.name != Z_SPIDER.name:
                continue
            if effect.generator_type.name != X_SPIDER.name:
                continue
            if (state.num_inputs, state.num_outputs) != (0, 1):
                continue
            if (effect.num_inputs, effect.num_outputs) != (1, 0):
                continue
            if state.phase is not None or effect.phase is not None:
                continue
            if claimed.get(state_ref, 0) != 1 or claimed.get(effect_ref, 0) != 1:
                continue
            if state_ref in boundary or effect_ref in boundary:
                continue
            state_dim = state.outputs[0].dim
            effect_dim = effect.inputs[0].dim
            if state_dim != effect_dim:
                continue
            matches.append(
                CapMatch(
                    state_id=state_ref.node_id,
                    effect_id=effect_ref.node_id,
                    wire=wire,
                    shared_dim=state_dim,
                    side_condition_outcomes=tuple(
                        SideConditionOutcome(condition.name, True, "re-derived from the diagram")
                        for condition in CAP_SIDE_CONDITIONS
                    ),
                )
            )
    matches.sort(key=lambda m: (m.state_id, m.effect_id))
    return tuple(matches)


class CapPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for the Z-state/X-effect cap.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_cap_matches`."""
        return find_cap_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_cap_matches` with the candidate wires restricted."""
        return find_cap_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The state's then the effect's node id."""
        cap = cast(CapMatch, match)
        return (int(cap.state_id), int(cap.effect_id))


def _port_claims(diagram: Diagram) -> tuple[dict[PortRef, int], frozenset[PortRef]]:
    """How many wires claim each port, and every port listed on either boundary list."""
    claims: dict[PortRef, int] = {}
    for wire in diagram.wires:
        claims[wire.a] = claims.get(wire.a, 0) + 1
        claims[wire.b] = claims.get(wire.b, 0) + 1
    return claims, frozenset(diagram.boundary_inputs) | frozenset(diagram.boundary_outputs)


def _claimed_at_most_once(
    ref: PortRef, claims: Mapping[PortRef, int], boundary: frozenset[PortRef]
) -> bool:
    """True iff ``ref`` carries at most one wire and, together with a boundary slot, one claim."""
    return claims.get(ref, 0) + (1 if ref in boundary else 0) <= 1


def _claimed_exactly_once_by_a_wire(
    ref: PortRef, claims: Mapping[PortRef, int], boundary: frozenset[PortRef]
) -> bool:
    """True iff exactly one wire claims ``ref`` and no boundary list names it."""
    return claims.get(ref, 0) == 1 and ref not in boundary


def _far_end(wire: Wire, near: PortRef) -> PortRef:
    """The endpoint of ``wire`` that is not ``near``."""
    return wire.b if wire.a == near else wire.a


def _all_passed(conditions: tuple[SideCondition, ...]) -> tuple[SideConditionOutcome, ...]:
    """One passing outcome per declared condition, in declared order."""
    return tuple(
        SideConditionOutcome(condition.name, True, "re-derived from the diagram")
        for condition in conditions
    )


def _is_spider(node: Node) -> bool:
    """True iff ``node`` is a registered Z or X spider."""
    return REGISTRY.is_registered(node.generator_type) and node.generator_type.name in (
        Z_SPIDER.name,
        X_SPIDER.name,
    )


def _exhausts_a_node_scope_box(diagram: Diagram, node_ids: tuple[NodeId, ...]) -> bool:
    """True iff removing ``node_ids`` would leave some node-scope bang box with no nodes."""
    return any(
        box.is_node_scope and box.node_scope and box.node_scope <= frozenset(node_ids)
        for box in diagram.bang_boxes.values()
    )


def _in_any_node_scope_box(diagram: Diagram, node_ids: tuple[NodeId, ...]) -> bool:
    """True iff any of ``node_ids`` sits in a node-scope bang box."""
    return any(
        box.is_node_scope and not box.node_scope.isdisjoint(node_ids)
        for box in diagram.bang_boxes.values()
    )


def _is_phaseless(node: Node) -> bool:
    """True iff ``node`` carries no phase vector, or an all-zero one."""
    return node.phase is None or node.phase.is_zero


def _uniform_leg_dim(node: Node) -> Dim | None:
    """``node``'s single leg dimension, or None if it has no legs or they disagree."""
    dims = {port.dim for port in (*node.inputs, *node.outputs)}
    if len(dims) != 1:
        return None
    return dims.pop()


IDENTITY_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("node_is_a_phaseless_spider", "a registered Z or X spider with no phase"),
    SideCondition("one_in_one_out", "exactly one input leg and one output leg"),
    SideCondition("not_a_self_loop", "no wire joins the node's two legs"),
    SideCondition(
        "legs_singly_claimed",
        "neither leg carries a second wire or a boundary slot alongside one, and at least "
        "one leg carries a wire to splice through",
    ),
    SideCondition(
        "splice_keeps_two_ports",
        "the spliced wire's far port is not the surviving leg's own neighbour",
    ),
    SideCondition("same_dimension", "both legs carry one dimension"),
    SideCondition(
        "leaves_every_bang_box_populated",
        "no node-scope bang box holds the node and nothing else",
    ),
)


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    """One located identity spider: the node, the wire spliced out, and the surviving leg."""

    node_id: NodeId
    wire: Wire
    surviving_ref: PortRef
    far_ref: PortRef
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The removed node's id and the spliced wire's far node id, ascending."""
        return _support_ids(
            self.node_id, self.wire.a.node_id, self.wire.b.node_id, self.far_ref.node_id
        )


def find_identity_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[IdentityMatch, ...]:
    """Every phaseless one-in-one-out spider, ordered by node id.

    The output leg's wire is the spliced one whenever it has one; otherwise the input
    leg's.
    """
    claims, boundary = _port_claims(diagram)
    by_port = _wire_by_port(diagram)
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    matches: list[IdentityMatch] = []
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        if not _is_spider(node) or not _is_phaseless(node):
            continue
        if not _seed_meets_anchors(anchors, neighbours, (node_id,), 1):
            continue
        if (node.num_inputs, node.num_outputs) != (1, 1):
            continue
        shared_dim = _uniform_leg_dim(node)
        if shared_dim is None:
            continue
        port_in = PortRef(node_id, Direction.INPUT, 0)
        port_out = PortRef(node_id, Direction.OUTPUT, 0)
        if not _claimed_at_most_once(port_in, claims, boundary):
            continue
        if not _claimed_at_most_once(port_out, claims, boundary):
            continue
        wire_in = by_port.get(port_in)
        wire_out = by_port.get(port_out)
        if wire_out is not None and wire_out == wire_in:
            continue
        if wire_out is not None:
            wire, consumed_ref, surviving_ref = wire_out, port_out, port_in
        elif wire_in is not None:
            wire, consumed_ref, surviving_ref = wire_in, port_in, port_out
        else:
            continue
        far_ref = _far_end(wire, consumed_ref)
        if far_ref.node_id == node_id:
            continue
        if anchors is not None and anchors.isdisjoint((node_id, far_ref.node_id)):
            continue
        neighbour = by_port.get(surviving_ref)
        if neighbour is not None and _far_end(neighbour, surviving_ref) == far_ref:
            continue
        if _exhausts_a_node_scope_box(diagram, (node_id,)):
            continue
        matches.append(
            IdentityMatch(
                node_id=node_id,
                wire=wire,
                surviving_ref=surviving_ref,
                far_ref=far_ref,
                shared_dim=shared_dim,
                side_condition_outcomes=_all_passed(IDENTITY_SIDE_CONDITIONS),
            )
        )
    return tuple(matches)


class IdentityRemovalPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for identity removal.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_identity_matches`."""
        return find_identity_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_identity_matches` with the candidate nodes restricted."""
        return find_identity_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The removed node's id."""
        return (int(cast(IdentityMatch, match).node_id),)


TRIANGLE_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("inverse_pair_in_series", "a T and a Ti joined output-to-input, either order"),
    SideCondition("joining_wire_unclaimed", "neither joined port carries a second claim"),
    SideCondition(
        "outer_legs_singly_claimed",
        "neither outer leg carries a second claim, and at least one carries a wire to "
        "splice through",
    ),
    SideCondition(
        "splice_keeps_two_ports",
        "the spliced wire's far port is not the surviving outer leg's own neighbour",
    ),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition(
        "bang_box_scope_agreement",
        "both nodes' innermost enclosing node-scope bang box, if any, are identical",
    ),
    SideCondition(
        "leaves_every_bang_box_populated",
        "no node-scope bang box holds the pair and nothing else",
    ),
)


@dataclass(frozen=True, slots=True)
class TriangleMatch:
    """One located T/Ti pair: the two nodes, the wire between them, and the wire spliced out."""

    first_id: NodeId
    second_id: NodeId
    wire: Wire
    spliced_wire: Wire
    surviving_ref: PortRef
    far_ref: PortRef
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The pair's node ids and the spliced wire's far node id, ascending."""
        return _support_ids(
            self.first_id,
            self.second_id,
            self.wire.a.node_id,
            self.wire.b.node_id,
            self.spliced_wire.a.node_id,
            self.spliced_wire.b.node_id,
            self.far_ref.node_id,
        )


_TRIANGLE_PAIR_NAMES = frozenset((TRIANGLE.name, TRIANGLE_INVERSE.name))
"""The two generator names a :class:`TriangleMatch` joins, one of each."""


def find_triangle_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[TriangleMatch, ...]:
    """Every T/Ti pair in series, ordered by the first node's id.

    The second node's output wire is the spliced one whenever it has one; otherwise the
    first node's input wire.
    """
    claims, boundary = _port_claims(diagram)
    by_port = _wire_by_port(diagram)
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    matches: list[TriangleMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, neighbours, (wire.a.node_id, wire.b.node_id), 1):
            continue
        for out_ref, in_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if out_ref.direction is not Direction.OUTPUT or in_ref.direction is not Direction.INPUT:
                continue
            if out_ref.index != 0 or in_ref.index != 0:
                continue
            first = diagram.nodes.get(out_ref.node_id)
            second = diagram.nodes.get(in_ref.node_id)
            if first is None or second is None or first.id == second.id:
                continue
            names = {first.generator_type.name, second.generator_type.name}
            if names != _TRIANGLE_PAIR_NAMES:
                continue
            if not REGISTRY.is_registered(first.generator_type):
                continue
            if not REGISTRY.is_registered(second.generator_type):
                continue
            if (first.num_inputs, first.num_outputs) != (1, 1):
                continue
            if (second.num_inputs, second.num_outputs) != (1, 1):
                continue
            if not _claimed_exactly_once_by_a_wire(out_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(in_ref, claims, boundary):
                continue
            dims = {
                port.dim
                for port in (*first.inputs, *first.outputs, *second.inputs, *second.outputs)
            }
            if len(dims) != 1:
                continue
            outer_in = PortRef(first.id, Direction.INPUT, 0)
            outer_out = PortRef(second.id, Direction.OUTPUT, 0)
            if not _claimed_at_most_once(outer_in, claims, boundary):
                continue
            if not _claimed_at_most_once(outer_out, claims, boundary):
                continue
            wire_head = by_port.get(outer_in)
            wire_tail = by_port.get(outer_out)
            if wire_tail is not None:
                spliced, consumed_ref, surviving_ref = wire_tail, outer_out, outer_in
            elif wire_head is not None:
                spliced, consumed_ref, surviving_ref = wire_head, outer_in, outer_out
            else:
                continue
            far_ref = _far_end(spliced, consumed_ref)
            if far_ref.node_id in (first.id, second.id):
                continue
            if anchors is not None and anchors.isdisjoint((first.id, second.id, far_ref.node_id)):
                continue
            neighbour = by_port.get(surviving_ref)
            if neighbour is not None and _far_end(neighbour, surviving_ref) == far_ref:
                continue
            enclosing = {
                innermost_node_scope_box(diagram, first.id),
                innermost_node_scope_box(diagram, second.id),
            }
            if len(enclosing) != 1:
                continue
            if _exhausts_a_node_scope_box(diagram, (first.id, second.id)):
                continue
            matches.append(
                TriangleMatch(
                    first_id=first.id,
                    second_id=second.id,
                    wire=wire,
                    spliced_wire=spliced,
                    surviving_ref=surviving_ref,
                    far_ref=far_ref,
                    shared_dim=next(iter(dims)),
                    side_condition_outcomes=_all_passed(TRIANGLE_SIDE_CONDITIONS),
                )
            )
    matches.sort(key=lambda m: (int(m.first_id), int(m.second_id)))
    return tuple(matches)


class TriangleInverseCancellationPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for T/Ti cancellation.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_triangle_matches`."""
        return find_triangle_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_triangle_matches` with the candidate wires restricted."""
        return find_triangle_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The first then the second node id."""
        triangle = cast(TriangleMatch, match)
        return (int(triangle.first_id), int(triangle.second_id))


STATE_COPY_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "state_is_a_phaseless_x_spider", "an X spider with no input, one output, and no phase"
    ),
    SideCondition("spider_is_a_phaseless_z_spider", "a Z spider with one input and no phase"),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition(
        "outside_every_bang_box",
        "neither node lies in any node-scope bang box",
    ),
)


@dataclass(frozen=True, slots=True)
class StateCopyMatch:
    """One located X state feeding a Z spider: both node ids, the wire, and the Z's output count."""

    state_id: NodeId
    spider_id: NodeId
    wire: Wire
    output_count: int
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The state's and the spider's node ids, ascending."""
        return _support_ids(self.state_id, self.spider_id)


def find_state_copy_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[StateCopyMatch, ...]:
    """Every phaseless X state wired into a phaseless Z spider's only input, by state id."""
    claims, boundary = _port_claims(diagram)
    matches: list[StateCopyMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for state_ref, spider_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if state_ref.direction is not Direction.OUTPUT:
                continue
            if spider_ref.direction is not Direction.INPUT or spider_ref.index != 0:
                continue
            state = diagram.nodes.get(state_ref.node_id)
            spider = diagram.nodes.get(spider_ref.node_id)
            if state is None or spider is None or state.id == spider.id:
                continue
            if not REGISTRY.is_registered(state.generator_type):
                continue
            if not REGISTRY.is_registered(spider.generator_type):
                continue
            if state.generator_type.name != X_SPIDER.name:
                continue
            if spider.generator_type.name != Z_SPIDER.name:
                continue
            if (state.num_inputs, state.num_outputs) != (0, 1) or spider.num_inputs != 1:
                continue
            if not _is_phaseless(state) or not _is_phaseless(spider):
                continue
            if not _claimed_exactly_once_by_a_wire(state_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(spider_ref, claims, boundary):
                continue
            dims = {port.dim for port in (*state.outputs, *spider.inputs, *spider.outputs)}
            if len(dims) != 1:
                continue
            if _in_any_node_scope_box(diagram, (state.id, spider.id)):
                continue
            matches.append(
                StateCopyMatch(
                    state_id=state.id,
                    spider_id=spider.id,
                    wire=wire,
                    output_count=spider.num_outputs,
                    shared_dim=next(iter(dims)),
                    side_condition_outcomes=_all_passed(STATE_COPY_SIDE_CONDITIONS),
                )
            )
    matches.sort(key=lambda m: (int(m.state_id), int(m.spider_id)))
    return tuple(matches)


class StateCopyPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for state copy.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_state_copy_matches`."""
        return find_state_copy_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_state_copy_matches` with the candidate wires restricted."""
        return find_state_copy_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The state's then the spider's node id."""
        copy = cast(StateCopyMatch, match)
        return (int(copy.state_id), int(copy.spider_id))


HOPF_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("z_and_x_spiders", "a registered Z spider and a registered X spider"),
    SideCondition(
        "two_paths",
        "one Z output runs straight into an X input, a second Z output runs into that same "
        "X through two Fourier boxes in series",
    ),
    SideCondition(
        "path_ports_unclaimed", "every port along the two paths carries exactly one wire"
    ),
    SideCondition(
        "same_dimension", "every leg along the two paths and both spiders share one dimension"
    ),
    SideCondition(
        "phase_dimension_agreement",
        "each spider's phase vector, if present, is over that dimension",
    ),
    SideCondition(
        "outside_every_bang_box",
        "none of the four nodes lies in any node-scope bang box",
    ),
)


@dataclass(frozen=True, slots=True)
class HopfMatch:
    """One located Hopf pair: the two spiders, the two F boxes, and the four consumed wires."""

    z_id: NodeId
    x_id: NodeId
    fourier_ids: tuple[NodeId, NodeId]
    wires: tuple[Wire, ...]
    z_leg_indices: tuple[int, int]
    x_leg_indices: tuple[int, int]
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The two spiders' and the two F boxes' node ids, ascending."""
        return _support_ids(self.z_id, self.x_id, *self.fourier_ids)


def _fourier_pair_path(
    diagram: Diagram,
    start: PortRef,
    by_port: Mapping[PortRef, Wire],
    claims: Mapping[PortRef, int],
    boundary: frozenset[PortRef],
) -> tuple[tuple[NodeId, NodeId], tuple[Wire, Wire, Wire], PortRef] | None:
    """The two F boxes in series leaving ``start``, their three wires, and the far port."""
    boxes: list[NodeId] = []
    wires: list[Wire] = []
    current = start
    for _ in range(2):
        wire = by_port.get(current)
        if wire is None:
            return None
        entry = _far_end(wire, current)
        node = diagram.nodes.get(entry.node_id)
        if node is None or not REGISTRY.is_registered(node.generator_type):
            return None
        if node.generator_type.name != FOURIER_BOX.name:
            return None
        if entry.direction is not Direction.INPUT or entry.index != 0:
            return None
        if (node.num_inputs, node.num_outputs) != (1, 1):
            return None
        if node.id in boxes:
            return None
        exit_ref = PortRef(node.id, Direction.OUTPUT, 0)
        if not _claimed_exactly_once_by_a_wire(entry, claims, boundary):
            return None
        if not _claimed_exactly_once_by_a_wire(exit_ref, claims, boundary):
            return None
        boxes.append(node.id)
        wires.append(wire)
        current = exit_ref
    final = by_port.get(current)
    if final is None:
        return None
    wires.append(final)
    return (boxes[0], boxes[1]), (wires[0], wires[1], wires[2]), _far_end(final, current)


def find_hopf_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[HopfMatch, ...]:
    """Every Z/X pair joined by one plain wire and one wire through two F boxes, by node id."""
    claims, boundary = _port_claims(diagram)
    by_port = _wire_by_port(diagram)
    neighbours = _neighbour_ids(diagram) if anchors is not None else None
    matches: list[HopfMatch] = []
    for z_id in sorted(diagram.nodes):
        z_node = diagram.nodes[z_id]
        if not REGISTRY.is_registered(z_node.generator_type):
            continue
        if z_node.generator_type.name != Z_SPIDER.name:
            continue
        if not _seed_meets_anchors(anchors, neighbours, (z_id,), 3):
            continue
        z_dim = _uniform_leg_dim(z_node)
        if z_dim is None:
            continue
        for direct_index in range(z_node.num_outputs):
            direct_ref = PortRef(z_id, Direction.OUTPUT, direct_index)
            direct_wire = by_port.get(direct_ref)
            if direct_wire is None:
                continue
            if not _claimed_exactly_once_by_a_wire(direct_ref, claims, boundary):
                continue
            x_direct = _far_end(direct_wire, direct_ref)
            x_node = diagram.nodes.get(x_direct.node_id)
            if x_node is None or x_node.id == z_id:
                continue
            if not REGISTRY.is_registered(x_node.generator_type):
                continue
            if x_node.generator_type.name != X_SPIDER.name:
                continue
            if x_direct.direction is not Direction.INPUT:
                continue
            if not _claimed_exactly_once_by_a_wire(x_direct, claims, boundary):
                continue
            if _uniform_leg_dim(x_node) != z_dim:
                continue
            for fourier_index in range(z_node.num_outputs):
                if fourier_index == direct_index:
                    continue
                start = PortRef(z_id, Direction.OUTPUT, fourier_index)
                if not _claimed_exactly_once_by_a_wire(start, claims, boundary):
                    continue
                path = _fourier_pair_path(diagram, start, by_port, claims, boundary)
                if path is None:
                    continue
                fourier_ids, fourier_wires, x_fourier = path
                if anchors is not None and anchors.isdisjoint((z_id, x_node.id, *fourier_ids)):
                    continue
                if x_fourier.node_id != x_node.id or x_fourier.direction is not Direction.INPUT:
                    continue
                if x_fourier == x_direct:
                    continue
                if not _claimed_exactly_once_by_a_wire(x_fourier, claims, boundary):
                    continue
                if any(_uniform_leg_dim(diagram.nodes[box_id]) != z_dim for box_id in fourier_ids):
                    continue
                if any(
                    node.phase is not None and node.phase.dim != z_dim for node in (z_node, x_node)
                ):
                    continue
                if _in_any_node_scope_box(diagram, (z_id, x_node.id, *fourier_ids)):
                    continue
                matches.append(
                    HopfMatch(
                        z_id=z_id,
                        x_id=x_node.id,
                        fourier_ids=fourier_ids,
                        wires=(direct_wire, *fourier_wires),
                        z_leg_indices=(direct_index, fourier_index),
                        x_leg_indices=(x_direct.index, x_fourier.index),
                        shared_dim=z_dim,
                        side_condition_outcomes=_all_passed(HOPF_SIDE_CONDITIONS),
                    )
                )
    matches.sort(key=lambda m: (int(m.z_id), int(m.x_id), m.z_leg_indices, m.x_leg_indices))
    return tuple(matches)


class HopfPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for the Hopf law.

    Locality radius 3.
    """

    locality_radius = 3

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_hopf_matches`."""
        return find_hopf_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_hopf_matches` with the Z seed nodes restricted."""
        return find_hopf_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """Both spiders' node ids, then the Z-side and X-side leg indices."""
        hopf = cast(HopfMatch, match)
        return (int(hopf.z_id), int(hopf.x_id), hopf.z_leg_indices, hopf.x_leg_indices)


BIALGEBRA_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition(
        "x_is_a_phaseless_two_to_one", "an X spider with two inputs, one output, and no phase"
    ),
    SideCondition(
        "z_is_a_phaseless_one_to_two", "a Z spider with one input, two outputs, and no phase"
    ),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition(
        "outside_every_bang_box",
        "neither node lies in any node-scope bang box",
    ),
)


@dataclass(frozen=True, slots=True)
class BialgebraMatch:
    """One located X_{2->1} into Z_{1->2}: both node ids and the wire joining them."""

    x_id: NodeId
    z_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The X's and the Z's node ids, ascending."""
        return _support_ids(self.x_id, self.z_id)


def find_bialgebra_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[BialgebraMatch, ...]:
    """Every phaseless X_{2->1} whose output feeds a phaseless Z_{1->2}, by X node id."""
    claims, boundary = _port_claims(diagram)
    matches: list[BialgebraMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for x_ref, z_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            if x_ref.direction is not Direction.OUTPUT or x_ref.index != 0:
                continue
            if z_ref.direction is not Direction.INPUT or z_ref.index != 0:
                continue
            x_node = diagram.nodes.get(x_ref.node_id)
            z_node = diagram.nodes.get(z_ref.node_id)
            if x_node is None or z_node is None or x_node.id == z_node.id:
                continue
            if not REGISTRY.is_registered(x_node.generator_type):
                continue
            if not REGISTRY.is_registered(z_node.generator_type):
                continue
            if x_node.generator_type.name != X_SPIDER.name:
                continue
            if z_node.generator_type.name != Z_SPIDER.name:
                continue
            if (x_node.num_inputs, x_node.num_outputs) != (2, 1):
                continue
            if (z_node.num_inputs, z_node.num_outputs) != (1, 2):
                continue
            if not _is_phaseless(x_node) or not _is_phaseless(z_node):
                continue
            if not _claimed_exactly_once_by_a_wire(x_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(z_ref, claims, boundary):
                continue
            x_dim = _uniform_leg_dim(x_node)
            if x_dim is None or _uniform_leg_dim(z_node) != x_dim:
                continue
            if _in_any_node_scope_box(diagram, (x_node.id, z_node.id)):
                continue
            matches.append(
                BialgebraMatch(
                    x_id=x_node.id,
                    z_id=z_node.id,
                    wire=wire,
                    shared_dim=x_dim,
                    side_condition_outcomes=_all_passed(BIALGEBRA_SIDE_CONDITIONS),
                )
            )
    matches.sort(key=lambda m: (int(m.x_id), int(m.z_id)))
    return tuple(matches)


class BialgebraPattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for the bialgebra law.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_bialgebra_matches`."""
        return find_bialgebra_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_bialgebra_matches` with the candidate wires restricted."""
        return find_bialgebra_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The X's then the Z's node id."""
        bialgebra = cast(BialgebraMatch, match)
        return (int(bialgebra.x_id), int(bialgebra.z_id))


FOURIER_STATE_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("spider_is_a_phaseless_z_state", "a Z spider with no phase and a single leg"),
    SideCondition("fourier_box_on_that_leg", "a registered F box wired to it in series"),
    SideCondition("joined_and_unclaimed", "one wire joins them and neither port is otherwise used"),
    SideCondition("free_leg_singly_claimed", "the F box's other leg carries at most one claim"),
    SideCondition("same_dimension", "every leg of the pair carries one dimension"),
    SideCondition(
        "bang_box_scope_agreement",
        "both nodes' innermost enclosing node-scope bang box, if any, are identical",
    ),
)


@dataclass(frozen=True, slots=True)
class FourierStateMatch:
    """One located F box on a phaseless Z state's output, or on a Z effect's input."""

    spider_id: NodeId
    fourier_id: NodeId
    wire: Wire
    free_ref: PortRef
    is_state: bool
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        """The spider's and the F box's node ids, ascending."""
        return _support_ids(self.spider_id, self.fourier_id, self.free_ref.node_id)


def find_fourier_state_matches(
    diagram: Diagram, *, anchors: frozenset[NodeId] | None = None
) -> tuple[FourierStateMatch, ...]:
    """Every F box in series with a phaseless Z state or Z effect, by spider node id."""
    claims, boundary = _port_claims(diagram)
    matches: list[FourierStateMatch] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if not _seed_meets_anchors(anchors, None, (wire.a.node_id, wire.b.node_id), 0):
            continue
        for spider_ref, fourier_ref in ((wire.a, wire.b), (wire.b, wire.a)):
            spider = diagram.nodes.get(spider_ref.node_id)
            fourier = diagram.nodes.get(fourier_ref.node_id)
            if spider is None or fourier is None or spider.id == fourier.id:
                continue
            if not REGISTRY.is_registered(spider.generator_type):
                continue
            if not REGISTRY.is_registered(fourier.generator_type):
                continue
            if spider.generator_type.name != Z_SPIDER.name:
                continue
            if fourier.generator_type.name != FOURIER_BOX.name:
                continue
            if (fourier.num_inputs, fourier.num_outputs) != (1, 1):
                continue
            if not _is_phaseless(spider):
                continue
            is_state = spider_ref.direction is Direction.OUTPUT
            if is_state:
                if (spider.num_inputs, spider.num_outputs) != (0, 1):
                    continue
                if fourier_ref != PortRef(fourier.id, Direction.INPUT, 0):
                    continue
                free_ref = PortRef(fourier.id, Direction.OUTPUT, 0)
            else:
                if (spider.num_inputs, spider.num_outputs) != (1, 0):
                    continue
                if fourier_ref != PortRef(fourier.id, Direction.OUTPUT, 0):
                    continue
                free_ref = PortRef(fourier.id, Direction.INPUT, 0)
            if spider_ref.index != 0:
                continue
            if not _claimed_exactly_once_by_a_wire(spider_ref, claims, boundary):
                continue
            if not _claimed_exactly_once_by_a_wire(fourier_ref, claims, boundary):
                continue
            if not _claimed_at_most_once(free_ref, claims, boundary):
                continue
            dims = {
                port.dim
                for port in (*spider.inputs, *spider.outputs, *fourier.inputs, *fourier.outputs)
            }
            if len(dims) != 1:
                continue
            enclosing = {
                innermost_node_scope_box(diagram, spider.id),
                innermost_node_scope_box(diagram, fourier.id),
            }
            if len(enclosing) != 1:
                continue
            matches.append(
                FourierStateMatch(
                    spider_id=spider.id,
                    fourier_id=fourier.id,
                    wire=wire,
                    free_ref=free_ref,
                    is_state=is_state,
                    shared_dim=next(iter(dims)),
                    side_condition_outcomes=_all_passed(FOURIER_STATE_SIDE_CONDITIONS),
                )
            )
    matches.sort(key=lambda m: (int(m.spider_id), int(m.fourier_id)))
    return tuple(matches)


class FourierStateColorChangePattern(Pattern):
    """The :class:`~archytaszx.rewrite.rule.Pattern` implementation for the F-on-a-state rule.

    Locality radius 1.
    """

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_fourier_state_matches`."""
        return find_fourier_state_matches(diagram)

    def find_matches_anchored(
        self, diagram: Diagram, anchors: frozenset[NodeId]
    ) -> tuple[Match, ...]:
        """Delegate to :func:`find_fourier_state_matches` with the candidate wires restricted."""
        return find_fourier_state_matches(diagram, anchors=anchors)

    def order_key(self, match: Match) -> tuple[object, ...]:
        """The spider's then the F box's node id."""
        fourier_state = cast(FourierStateMatch, match)
        return (int(fourier_state.spider_id), int(fourier_state.fourier_id))
