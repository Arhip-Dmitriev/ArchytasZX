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

Phase 5 implements exactly one :class:`~qufzx.rewrite.rule.Pattern`: two spiders of the
same generator type joined by a wire whose connected legs agree on dimension. A pair joined
by k wires yields up to one match per wire, each decided on its own; a match fuses across
its own wire and leaves the rest as self-loops on the merged node.

Side conditions, in the order applied (see ``FUSION_SIDE_CONDITIONS``):

1. ``distinct_nodes`` -- the endpoints are different nodes. Reported True always: a
   self-loop wire is dropped before candidate grouping, and equal ids raise from
   :func:`resolve_fusion_match` rather than failing here.
2. ``same_generator_type`` -- both nodes carry the identical registered
   :class:`~qufzx.diagram.generators.GeneratorType`, and that type is fusable
   (``Z_SPIDER``/``X_SPIDER``).
3. ``parallel_wires_become_self_loops`` -- reported True always, carrying the count of other
   wires joining the pair; each survives as a self-loop on the merged spider.
4. ``consumed_wire_direction_permitted_for_color`` -- for X, the consumed wire must run
   OUTPUT-to-INPUT, so that the contraction is ``F^dagger F = I`` and fusion is scalar-free.
   Z is diagonal in every axis, so any direction is valid.
5. ``consumed_ports_singly_claimed`` -- neither consumed port is claimed by a second wire
   or listed on a boundary.
6. ``dimension_agreement`` -- the connected legs' :class:`~qufzx.algebra.dimension.Dim`
   unify. A ``FAILURE`` is a non-match; a ``DEFERRED`` or binding-only ``SUCCESS`` is
   recorded as a dimension constraint. Every surviving leg of both nodes is then unified
   against the running ``shared_dim`` in turn, each refinement carrying forward.
7. ``phase_dimension_agreement`` -- every phase vector present must unify with
   ``shared_dim``. Unlike condition 6, a ``DEFERRED`` is rejected rather than recorded.
   Conditions 6 and 7 form one bounded fixpoint; see :func:`resolve_fusion_match`.

Conditions 1 and 3 are structural facts recorded for the certificate, not decisions: no
candidate can fail either. The numbering above is authoritative and machine-checked against
``FUSION_SIDE_CONDITIONS`` by
``tests/test_engine.py::TestConditionNumberingMatchesDeclaredOrder``.

One verification predicate. :func:`resolve_fusion_match` decides every condition above.
:func:`find_matches` calls it to decide whether a candidate is a match, and
:func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls it again, fresh, against
the diagram it was handed, building only from its result.

Malformed references. :mod:`qufzx.diagram.graph` is deliberately permissive about what a
:class:`~qufzx.diagram.graph.Wire` or boundary entry may name. :func:`find_matches` checks
both endpoints of every wire and every boundary entry via :func:`_validate_wire_endpoint`,
in a pre-pass that runs before grouping, raising
:class:`~qufzx.rewrite.rule.RewriteGrammarError`. Detection does not depend on any other
property of the wire or its candidate pair.

Match-implies-applicable. Every match returned here can be applied by
:func:`~qufzx.rewrite.engine.apply` without raising anything except the step-8
relative-postcondition :class:`~qufzx.rewrite.rule.RewriteDomainError`.

Dimension constraints. ``dimension_constraints`` records every dimension equality accepted
without a syntactic identity: both a ``DEFERRED`` unify and a ``SUCCESS`` holding only
under a binding. Entries are :class:`~qufzx.rewrite.rule.DimensionConstraint`, keyed by
:class:`~qufzx.rewrite.rule.ConstraintSource`.

Non-concrete bindings. :meth:`Dim.unify` can bind a symbol to another symbolic ``Dim``
(``d := e``), but :meth:`Dim.substitute` and ``PhaseVector.substitute`` accept only
concrete replacements, so such a binding is carried as an assumption rather than resolved
through.

Determinism. :func:`find_matches` sorts its result by node ids, then by the consumed wire's
(direction, index) on each side. Every set iteration whose order could reach a returned
value, a certificate field, or an exception message is sorted by a hash-independent key.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from qufzx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from qufzx.algebra.phase import PhaseDomainError, PhaseSubstituteValue, PhaseSymbolKey, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef, Wire
from qufzx.rewrite.rule import (
    ConstraintOutcome,
    ConstraintSource,
    ConstraintSourceKind,
    DimensionConstraint,
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
        "'Non-concrete bindings' note)",
    ),
)
"""The declared side-condition specs for :class:`FusionPattern`, in the module docstring's
numbered order. Entries 1 and 3 are structural facts a candidate cannot fail; the rest are
decisions."""


@dataclass(frozen=True, slots=True)
class FusionMatch:
    """One located fusion occurrence: the two spiders, the consumed wire, and the shared dim.

    ``a_id`` is always the lower :class:`~qufzx.diagram.graph.NodeId` of the pair,
    ``b_id`` the higher -- the convention :mod:`qufzx.rewrite.rules_library` reuses as its
    merged-leg ordering ("A's surviving legs, then B's"), and the one that seeds
    ``shared_dim`` from the A-side consumed leg's ``Dim``. A connecting pair that unifies
    resolves the seed away; one that only defers leaves it standing as ``shared_dim``, so a
    ``d``/``d*e`` pair fuses onto whichever the lower-id node carried.

    ``bindings`` is the whole-candidate accumulator of every concrete symbol binding
    conditions 6 and 7 produced. The builder substitutes it into a present phase's entries,
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
        """True iff every recorded side condition passed. See :class:`qufzx.rewrite.rule.Match`."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)


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
key. Kept as a guard against :meth:`~qufzx.algebra.dimension.Dim.unify`'s placeholder
contract, which Phase 10 replaces."""


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
    """:meth:`~qufzx.algebra.dimension.Dim.unify` returned ``FAILURE`` outright."""

    CONTRADICTORY_REBIND = "contradictory_rebind"
    """The unify succeeded, but :func:`_merge_bindings` rejected its binding as a
    contradictory rebind of a symbol already bound to a different concrete value."""

    PHASE_DEFERRED = "phase_deferred"
    """A phase's own dimension unify ``DEFERRED`` against the shared leg dimension --
    tolerated for a leg or the connecting pair, not for a phase (condition 7,
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
    in what reaches ``Dim.unify``. Kept as a structural guard against
    :meth:`Dim.unify`'s placeholder contract, which Phase 10 replaces.
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

    One entry per :class:`~qufzx.rewrite.rule.ConstraintSource`, never one per check: the
    leg/phase fixpoint re-checks the same source once per pass, and each re-check
    :meth:`record`\\ s over the previous entry in place, so the finished sequence is in
    first-derivation order.

    The invariant is adequacy: the finished ``dimension_constraints`` alone implies every
    ``(assumed, equal_to)`` pair any check ever asserted, including one a later pass replaced
    or dropped. ``bindings`` is not part of it -- ``RewriteStep`` carries only the
    constraints. Checked by
    ``tests/test_phase5_certificate_sweep.py::TestConstraintRecordAdequacy``; every cell of
    the table below is pinned by
    ``tests/test_match.py::TestConstraintRecordPolicyTable``.

    The policy over (previous entry, this check's outcome):

    * (none, ``DEFERRED`` / ``BOUND``): record it.
    * (none, bare identity via :meth:`record_identity`): no-op.
    * (``DEFERRED``, ``DEFERRED``): overwrite, restated at its currently-resolved operands.
    * (``DEFERRED``, ``BOUND``): overwrite.
    * (``DEFERRED``, bare identity): drop -- nothing is assumed any more.
    * (``BOUND``, ``DEFERRED``): overwrite; the record holds each source's most-resolved
      current statement, not a history.
    * (``BOUND``, ``BOUND``): overwrite.
    * (``BOUND``, bare identity): **keep** the ``BOUND`` entry -- the identity holds only
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
    own :meth:`~qufzx.rewrite.rule.ConstraintSource.surviving_leg` key.
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
    docstring, condition 7). Returns a :class:`_ResolutionFailure` on any of those or on a
    contradictory rebind; its ``equal_to`` is the ``shared_dim`` actually checked against the
    failing phase. On success returns the refined ``shared_dim``, having written each phase's
    binding into ``record`` under its
    :meth:`~qufzx.rewrite.rule.ConstraintSource.node_phase` key.
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
    """Build condition 6's (``dimension_agreement``) passing outcome from a leg-sweep state.

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
    trial construction and :mod:`qufzx.rewrite.rules_library`'s builder to build the merged
    phase, so both decide reattachability the same way.

    Raises :class:`~qufzx.algebra.phase.PhaseDomainError` if, after substitution, an entry's
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

    Four causes (module docstring, condition 7): a ``FAILURE``, a ``DEFERRED`` unify, a
    binding to a non-concrete ``Dim``, and a contradictory rebind. The post-loop
    ``reattach_phase`` failure is rendered at its own call site in
    :func:`resolve_fusion_match`.
    """
    if failure.reason is _FailureReason.PHASE_DEFERRED:
        return (
            f"a present phase dimension ({failure.assumed}) unifies only as DEFERRED "
            f"against the resolved shared leg dimension {failure.equal_to} -- a DEFERRED "
            "unify is not accepted for a phase, see the module docstring, condition 7"
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


def resolve_fusion_match(
    diagram: Diagram, a_id: NodeId, b_id: NodeId, wire: Wire
) -> FusionResolution:
    """Decide, from ``diagram`` alone, whether ``wire`` is a legal fusion of ``a_id``/``b_id``.

    The single shared predicate behind every condition in the module docstring.
    :func:`find_matches` calls it once per candidate wire;
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls it again, fresh, and
    trusts only its return value for graph surgery.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` for a request that cannot be
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
        generator_types_match and node_a.generator_type.name in _FUSABLE_GENERATOR_NAMES
    )
    if not generator_types_match:
        same_type_detail = (
            f"{a_id!r} is {node_a.generator_type.name!r} but {b_id!r} is "
            f"{node_b.generator_type.name!r}"
        )
    elif not is_fusable_type:
        same_type_detail = (
            f"both nodes are {node_a.generator_type.name!r}, but that is not a registered "
            "fusable generator type"
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
            ("consumed_ports_singly_claimed", "dimension_agreement", "phase_dimension_agreement"),
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
            ("dimension_agreement", "phase_dimension_agreement"),
            "not evaluated: consumed_ports_singly_claimed failed first",
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
            # every remaining name False and cannot express this path's mix.
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
        # The fixpoint decides conditions 6 and 7 jointly, so when it does not terminate
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

    # Read off record.entries(), in first-derivation order, de-duplicated with dict.fromkeys
    # to drop a name a later pass re-bound.
    phase_bound_values: dict[str, Dim] = {}
    for phase_entry in record.entries():
        if (
            phase_entry.source.kind is ConstraintSourceKind.NODE_PHASE
            and phase_entry.outcome is ConstraintOutcome.BOUND
        ):
            phase_bound_values.update(phase_entry.bound_here)
    unique_bound_names = list(dict.fromkeys(phase_bound_values))
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
    return FusionResolution(
        passed=True,
        shared_dim=shared_dim,
        bindings=MappingProxyType(dict(bindings)),
        dimension_constraints=record.entries(),
        outcomes=tuple(outcomes),
    )


def find_matches(diagram: Diagram) -> tuple[FusionMatch, ...]:
    """Find every same-color spider fusion occurrence in ``diagram``. See the module docstring.

    Never mutates ``diagram``, and does not require it to be well-formed --
    :func:`~qufzx.diagram.validate.validate` is never called here. Returns matches sorted by
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

        # Conditions 2 and 4-7 are decided by exactly this call -- the same function
        # spider_fusion_builder calls again to re-verify the match.
        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
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


class FusionPattern(Pattern):
    """The :class:`~qufzx.rewrite.rule.Pattern` implementation for same-color spider fusion."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_matches`. See the module docstring."""
        return find_matches(diagram)
