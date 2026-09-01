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

"""The fusion matcher: finds occurrences of same-color spider fusion, and only that pattern.

Phase 5 implements exactly one :class:`~qufzx.rewrite.rule.Pattern`: two spiders of the
same generator type joined by a wire whose connected legs agree on dimension. A pair
joined by k wires yields one candidate per wire; each fuses across its own wire and
leaves the rest as self-loops on the merged node. Every other pattern shape (bialgebra,
Hopf, copy, identity removal) is out of scope until Phase 11.

Side conditions, in the order applied (see ``FUSION_SIDE_CONDITIONS`` for their declared
names, and :class:`FusionMatch` for where per-candidate outcomes are recorded):

1. ``distinct_nodes`` -- the endpoints are different nodes. Enforced structurally: a
   self-loop wire is dropped before candidate grouping.
2. ``same_generator_type`` -- both nodes carry the identical registered
   :class:`~qufzx.diagram.generators.GeneratorType`, and that type is in this pattern's
   fusable set (``Z_SPIDER``/``X_SPIDER``). Both facts are reported as this one condition.
3. ``parallel_wires_become_self_loops`` -- k joining wires yield up to k candidates. Both
   endpoints of a leftover wire land in the builder's ``port_mapping``, so ``apply``'s
   step-5 remap turns it into a self-loop. Every other condition still applies per
   candidate, so a pair may yield fewer candidates than wires.
4. ``consumed_wire_direction_permitted_for_color`` -- for X, the consumed wire must run
   OUTPUT-to-INPUT. Per :mod:`qufzx.semantics.denote`'s axis convention X applies ``F`` to
   output axes and ``conj(F)`` to input axes, so an output-to-input contraction gives
   ``F^dagger F = I`` and fusion is scalar-free; a same-direction one gives ``F^T F``, a
   different rule. The Z tensor is diagonal in every axis and contraction applies no
   conjugation, so any direction is valid for Z.
5. ``consumed_ports_singly_claimed`` -- neither consumed port is claimed by a second wire
   or listed on a boundary. Decided from ``diagram`` alone, before the dimension fixpoint;
   see "Match-implies-applicable" below.
6. ``dimension_agreement`` -- the connected legs' :class:`~qufzx.algebra.dimension.Dim`
   unify. A ``FAILURE`` is a non-match; a ``DEFERRED`` or binding-only ``SUCCESS`` is
   recorded as a dimension constraint. Any binding is substituted into the A-side leg's
   ``Dim`` to give a provisional ``shared_dim``. Every *surviving* leg of both nodes is
   then unified against the running ``shared_dim`` in turn, each refinement carrying
   forward, so leg dims ``d``, ``d``, ``2`` resolve to ``2``.
7. ``phase_dimension_agreement`` -- every phase vector present must unify with
   ``shared_dim``. Unlike condition 6 (``dimension_agreement``), a ``DEFERRED`` here is
   rejected rather than
   recorded: a phase's entries can reference its own ``dim``'s free symbols, so
   reattaching them to a more-resolved ``shared_dim`` is correct only under an actual
   binding to substitute through them. A phase's binding also refines ``shared_dim``
   (``phase_schema`` ``TIED_TO_LEG_DIM`` ties phase dim to leg dim), so conditions 6 and 7
   form one bounded fixpoint -- see :func:`resolve_fusion_match` for its termination
   argument. This condition also verifies that :func:`reattach_phase` succeeds against the
   final ``shared_dim``, catching e.g. an entry index that falls out of range once the
   phase's own symbol binds. Since the builder calls the same :func:`reattach_phase`,
   match-approval and build-applicability are one predicate by construction.

One verification predicate, not two kept in sync by hand. :func:`resolve_fusion_match` is
the single function that decides every condition above. :func:`find_matches` calls it to
decide whether a candidate is a match at all, and
:func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls the same function again,
fresh, against the diagram it was handed, building only from its result. Conditions 2 and
4-7 are therefore re-derived at build time rather than trusted from the match's own claimed
fields, which a foreign or hand-built match could have fabricated.

Malformed references. :mod:`qufzx.diagram.graph` is deliberately permissive about what a
:class:`~qufzx.diagram.graph.Wire` or boundary entry may name, so an un-validated diagram
can hold a reference to an unknown node id or an out-of-range port index.
:func:`find_matches` checks both endpoints of every wire and every entry of both boundary
lists via :func:`_validate_wire_endpoint`, in a pre-pass that runs before the self-loop
skip, before grouping, and before any other candidate property, raising
:class:`~qufzx.rewrite.rule.RewriteGrammarError`. The ordering is load-bearing: detection
must not depend on any other property of the wire or its candidate pair.

This does not conflict with :mod:`qufzx.rewrite.engine`'s step 8, which holds that a
diagram already carrying a hard-error *validation issue* is legitimately rewritable. The
categories differ, along the line :mod:`qufzx.rewrite.rule` draws for the package:
``RewriteDomainError`` for a value outside the domain a rewrite requires,
``RewriteGrammarError`` for a malformed request.

Match-implies-applicable and multiply-claimed ports. A port claimed by more than one wire
(``PORT_WIRED_TWICE``) or both wired and on a boundary (``PORT_WIRED_AND_BOUNDARY``) is
not a fusion occurrence even when it is the port a candidate would consume: ``apply``
requires every consumed port to appear in the builder's ``port_mapping``, but a builder
maps only *surviving* ports. The invariant is that every match returned here can be
applied by :func:`~qufzx.rewrite.engine.apply` without raising anything except the step-8
relative-postcondition :class:`~qufzx.rewrite.rule.RewriteDomainError`.

Dimension constraints. ``dimension_constraints`` records every dimension equality accepted
without a syntactic identity: both a ``DEFERRED`` unify and a ``SUCCESS`` holding only
under a binding. Entries are :class:`~qufzx.rewrite.rule.DimensionConstraint`, keyed by
:class:`~qufzx.rewrite.rule.ConstraintSource` -- ``CONNECTING_PAIR``, ``SURVIVING_LEG``
(by ``(NodeId, Direction, index)``), or ``NODE_PHASE`` (by node id). All are assumed
equalities a diagram-level unifier (Phase 10) must eventually justify, so all belong in
the certificate; a bare identity assumes nothing and is left out.

Non-concrete bindings. :meth:`Dim.unify` can bind a symbol to another symbolic ``Dim``
(``d := e``), but :meth:`Dim.substitute` and ``PhaseVector.substitute`` accept only
concrete replacements by contract, so such a binding is not expressible through the
Phase 1/2 substitution APIs. Solving it is :meth:`Dim.unify`'s Phase 10 carve-out; every
site here that consumes a binding carries it as an assumption instead.

Determinism. :func:`find_matches` sorts its result by node ids, then by the consumed
wire's (direction, index) on each side, never by set or dict iteration order. This is a
whole-module discipline: every set iteration whose order could reach a returned value, a
certificate field, or an exception message is sorted by the same hash-independent key
(:meth:`~qufzx.diagram.graph.Wire.sort_key`,
:meth:`~qufzx.diagram.graph.PortRef.sort_key`).

No :class:`FusionMatch` is ever constructed for a failing candidate.
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
    SideCondition("distinct_nodes", "the two matched nodes are not the same node"),
    SideCondition("same_generator_type", "both nodes are the same registered spider color"),
    SideCondition(
        "parallel_wires_become_self_loops",
        "every other wire joining the two nodes survives as a self-loop on the merged spider",
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
"""The declared side-condition specs for :class:`FusionPattern`. See the module docstring."""


@dataclass(frozen=True, slots=True)
class FusionMatch:
    """One located fusion occurrence: the two spiders, the consumed wire, and the shared dim.

    ``a_id`` is always the lower :class:`~qufzx.diagram.graph.NodeId` of the pair and
    ``b_id`` the higher -- a fixed, deterministic convention (not a claim about which node
    was "created first") that :mod:`qufzx.rewrite.rules_library` reuses as its merged-leg
    ordering convention ("A's surviving legs, then B's").

    ``bindings`` is the whole-candidate accumulator of every concrete symbol binding
    conditions 6 and 7 (``dimension_agreement``, ``phase_dimension_agreement``) produced
    while resolving ``shared_dim`` -- the connecting pair's own, every surviving leg's on
    either node, and every present phase's own, across every pass of their shared fixpoint
    (see :func:`find_matches`'s local ``bindings`` dict, which this field is built from
    verbatim). :mod:`qufzx.rewrite.rules_library`'s builder substitutes it into a present
    phase's entries, via :func:`reattach_phase`, before reattaching them to ``shared_dim``
    -- see that function's docstring for why this, not a stricter matcher, is the
    resolution of the ``_over_shared_dim`` defect family.
    """

    a_id: NodeId
    b_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()
    bindings: Mapping[str, Dim] = MappingProxyType({})

    def __hash__(self) -> int:
        """Explicit, mirroring :meth:`~qufzx.rewrite.engine.RewriteStep.__hash__` exactly.

        ``@dataclass(frozen=True)`` with the default ``eq=True`` would otherwise generate a
        ``__hash__`` that hashes every field verbatim, including ``bindings`` -- a
        :class:`~types.MappingProxyType`, which is unhashable (its backing ``dict`` is
        mutable even though the proxy itself is read-only). Defining ``__hash__`` here
        explicitly, in the class body, makes ``dataclass`` leave it alone rather than
        overwrite it with the broken auto-generated one. Every other field is hashed as-is
        (``Dim``, ``Wire``, ``NodeId``, and the generated-``__hash__`` ``SideConditionOutcome``
        and its tuple are all already hashable); ``bindings`` is hashed as
        ``frozenset(bindings.items())`` -- order-independent, matching the dataclass-generated
        ``__eq__``, which compares ``bindings`` via plain mapping equality (also
        order-independent) -- so ``a == b`` still implies ``hash(a) == hash(b)``, the same
        contract :class:`~qufzx.rewrite.engine.RewriteStep` needs for Phase 12's cache, which
        embeds a ``FusionMatch`` in its own ``match`` field and therefore needs this to hold
        transitively.

        Same cross-process disclaimer as :meth:`RewriteStep.__hash__` (Phase 5 post-closing
        audit round 18, Defect 1): this contract is a within-process one. ``Wire`` (via
        ``PortRef`` -> ``Direction``) and ``DimensionConstraint`` (via ``ConstraintSourceKind``
        / ``ConstraintOutcome``) are reached transitively here, and every one of those is an
        ``enum.Enum`` hashed by member name -- ``PYTHONHASHSEED``-dependent, so
        ``hash(match)`` legitimately differs across processes for two values this module's own
        ``find_matches`` would otherwise report identically. What round 18 actually fixed is
        the *value and order* of every field this hash is computed over (in particular
        ``dimension_constraints`` and ``side_condition_outcomes``, whose upstream inputs used
        to depend on ``diagram.wires``' own hash-seed-dependent iteration order -- see
        :mod:`qufzx.diagram.validate`'s docstring), not ``hash()`` stability itself, which was
        never a promise this method could make and does not make now.
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
"""Generator names for which a same-direction (OUTPUT-OUTPUT or INPUT-INPUT) connecting
wire is still valid fusion -- see condition 4
(``consumed_wire_direction_permitted_for_color``)
(``consumed_wire_direction_permitted_for_color``) in the module docstring. Z only: X's
Fourier-conjugate structure makes a same-direction wire a
different, unimplemented rule, not fusion."""

_MAX_FIXPOINT_PASSES = 32
"""Iteration budget for :func:`resolve_fusion_match`'s joint condition-5/6 fixpoint.

Module-level, not a function local, so a test can patch it low and actually exercise the
exhaustion path (``tests/test_match.py::TestFixpointBudgetExhaustion``).

Unreachability argument (D1's fix restates this in terms of ``bindings``, not
``shared_dim``'s own concretization -- the pre-fix version of this docstring argued from
the latter, which is exactly the false converse D1 was): ``bindings`` is monotone -- a key
is only ever added, never rebound to a different value. Round 23 correction: this is not
because :func:`_merge_bindings` rejects a contradictory rebind (instrumented across 15,000
seeds, that guard has 0 hits -- see its own docstring); it is because every operand this
fixpoint unifies is first resolved through :func:`_resolve_with_bindings`, which substitutes
every symbol ``bindings`` already names, so an already-bound symbol is never free in what
reaches ``Dim.unify`` and therefore never reappears as a fresh binding key. ``_merge_bindings``
is kept as a structural guard for that same invariant, not as the mechanism that produces it
-- and every key ``bindings`` can ever hold is drawn from the finite set of free symbols
appearing in
node_a's legs and phase, node_b's legs and phase, and the connecting pair (equivalently,
``shared_dim``'s own lineage, which starts at ``port_a.dim`` and is only ever refined by
substituting members of that same finite set). A pass that does not stabilise therefore adds
at least one fresh key to ``bindings``, so the number of non-stabilising passes is bounded by
that finite symbol count -- unreachable for any diagram with fewer distinct dimension symbols
in play than the cap. The guard is kept anyway, conservatively refusing the candidate rather
than looping forever, because that bound rests on :meth:`~qufzx.algebra.dimension.Dim.unify`'s
current placeholder contract, which Phase 10 replaces."""


def _resolve_with_bindings(dim: Dim, bindings: Mapping[str, Dim]) -> Dim:
    """Substitute ``bindings`` into ``dim``, or return it unchanged if none apply.

    ``bindings`` is empty both when :meth:`Dim.unify` deferred and when it succeeded via a
    bare syntactic identity with nothing bound -- in both cases this is the identity
    function, which is exactly the "keep the raw Dim unchanged" behavior condition 6 in
    the module docstring calls for. A bare symbol can also unify by binding to *another*
    still-symbolic ``Dim`` (e.g. ``d`` against ``e`` binds ``d := e``); :meth:`Dim.substitute`
    only ever accepts a concrete replacement value, so such a binding is dropped here rather
    than resolving through it -- ``dim`` stays raw and unchanged for that symbol, the same
    treatment a deferred pair gets, rather than crashing on a substitution ``Dim`` was never
    built to perform.
    """
    concrete_bindings = {name: value for name, value in bindings.items() if value.is_concrete}
    if not concrete_bindings:
        return dim
    return dim.substitute(cast(Mapping[DimSymbolKey, DimSubstituteValue], concrete_bindings))


class _FailureReason(enum.Enum):
    """Why one of the fixpoint's unify helpers (:func:`_unify_surviving_legs`,
    :func:`_unify_phase_dims`, :func:`_unify_connecting_pair`) failed this pass.

    Phase 5 post-closing audit round 23, Task 3: introduced because a failure's detail
    string used to be derived from *which call site* returned ``None``/``(None, ...)``, not
    from *what actually went wrong* -- a leg/pair that never unifies at all (``UNIFY_FAILURE``)
    and one that unifies fine but whose binding contradicts an earlier one
    (``CONTRADICTORY_REBIND``) were reported with the identical generic wording, even though
    they are different failures with different remedies. ``PHASE_DEFERRED`` and
    ``PHASE_NON_CONCRETE_BINDING`` are :func:`_unify_phase_dims`-only: a leg or the connecting
    pair tolerates a deferral or a non-concrete binding (see the module docstring's
    "Non-concrete bindings" section), a phase does not.
    """

    UNIFY_FAILURE = "unify_failure"
    """:meth:`~qufzx.algebra.dimension.Dim.unify` returned ``FAILURE`` outright."""

    CONTRADICTORY_REBIND = "contradictory_rebind"
    """The unify succeeded, but :func:`_merge_bindings` rejected its binding as a
    contradictory rebind of a symbol already bound to a different concrete value."""

    PHASE_DEFERRED = "phase_deferred"
    """A phase's own dimension unify ``DEFERRED`` against the shared leg dimension -- tolerated
    for a leg or the connecting pair, but not for a phase (condition 7,
    ``phase_dimension_agreement``)."""

    PHASE_NON_CONCRETE_BINDING = "phase_non_concrete_binding"
    """A phase's own dimension unify succeeded only by binding to a non-concrete ``Dim`` --
    tolerated for a leg (left unused for shared-dimension resolution) but not for a phase,
    since a phase's own entries can reference its ``dim``'s free symbols directly (module
    docstring, "Non-concrete bindings")."""


@dataclass(frozen=True, slots=True)
class _ResolutionFailure:
    """One unify helper's failure this pass: why, and the two operands actually involved.

    ``assumed``/``equal_to`` are the operands *as checked this pass* (already resolved
    through the running ``bindings`` accumulator where applicable) -- the same pair a
    successful check would have recorded as a :class:`~qufzx.rewrite.rule.DimensionConstraint`
    had it not failed. A call site renders its detail string from these fields and ``reason``
    directly, never by re-deriving "what failed" from which branch of its own code returned
    this value (the same discipline :func:`_connecting_pair_detail` already applies to a
    *passing* check's own detail -- see that function's docstring).
    """

    reason: _FailureReason
    assumed: Dim
    equal_to: Dim


def _merge_bindings(
    bindings: dict[str, Dim], new_bindings: Mapping[str, Dim]
) -> tuple[str, Dim, Dim] | None:
    """Merge the concrete entries of ``new_bindings`` into ``bindings``, in place.

    Only concrete-valued bindings are stored; see :func:`_resolve_with_bindings` for why a
    binding to a non-concrete ``Dim`` is dropped rather than resolved through. Returns
    ``(name, existing_value, new_value)``, leaving ``bindings`` unmodified, iff a name
    already bound to one concrete ``Dim`` would be rebound to a different one -- a
    contradictory assumption, e.g. two surviving legs forcing the same symbol to two
    values -- and ``None`` on a clean merge. The conflict tuple lets a call site's failure
    detail name the symbol and both values, rather than folding this into the generic
    message a plain ``Dim.unify`` ``FAILURE`` gets: the two are different failures (a pair
    that does not unify at all, versus one that unifies but conflicts with an earlier
    binding). A non-``None`` return makes the candidate a non-match, exactly like a
    ``FAILURE``.

    Two things this guard is not. It is not what catches an unsatisfiable constraint set
    such as ``e*f == 2, e == 2, f == 2`` from legs ``[e*f, e, f]`` -- that is caught by
    :func:`resolve_fusion_match`'s fixpoint requiring a full pass to add nothing before
    exiting, which forces ``e*f`` to be re-resolved through the now-bound ``e``/``f`` and
    surfaces the contradiction as an ordinary ``Dim.unify`` ``FAILURE``. Nor is it what
    makes ``bindings`` monotone: that falls out of :func:`_resolve_with_bindings`
    substituting every known binding into an operand before it reaches ``Dim.unify``, so an
    already-bound symbol is never free in what gets unified and cannot reappear as a fresh
    binding key here.

    For that structural reason the contradiction branch does not currently fire on any call
    site. It is kept because it is the correct guard for a caller that updates ``bindings``
    with something not first passed through ``_resolve_with_bindings`` -- not an invariant
    this function can enforce itself -- and because :meth:`Dim.unify` is a placeholder Phase
    10 replaces, at which point the pre-resolution argument may no longer hold.
    ``tests/test_fusion_properties.py::TestSpiderFusionProperties::test_random_diagrams_fuse_soundly``
    wraps this function to assert a clean merge on every call, so a Phase 10 change that
    makes it reachable is announced by a failing assertion.
    """
    concrete = {name: value for name, value in new_bindings.items() if value.is_concrete}
    for name, value in concrete.items():
        existing = bindings.get(name)
        if existing is not None and existing != value:
            return (name, existing, value)
    bindings.update(concrete)
    return None


class _ConstraintRecord:
    """The source-keyed, insertion-ordered record of one candidate's dimension assumptions.

    One entry per :class:`~qufzx.rewrite.rule.ConstraintSource`, never one per check: the
    leg/phase fixpoint re-checks the same source once per pass, and each re-check
    :meth:`record`\\ s over the previous entry in place (a ``dict`` assignment to an
    existing key keeps its position), so the finished sequence is in first-derivation order.

    :meth:`record` overwrites unconditionally, for every outcome, so a previously ``BOUND``
    entry can be displaced by a later pass's ``DEFERRED``. The invariant that must hold is
    not "a ``BOUND`` entry is never displaced" but:

        **Adequacy.** The conjunction of the finished ``dimension_constraints`` and the
        finished ``bindings`` must imply every ``(assumed, equal_to)`` pair any check in the
        fixpoint ever asserted, including one a later pass replaced or dropped.

    This holds -- checked mechanically by
    ``tests/test_phase5_certificate_sweep.py::TestConstraintRecordAdequacy`` -- because only
    a *concrete* binding enters the shared ``bindings`` accumulator (:func:`_merge_bindings`
    drops non-concrete ones). No source's operands are ever built by resolving through
    another source's non-concrete binding, only through concrete ones, which are never later
    contradicted and are always separately recorded under their own source. A displaced entry
    can therefore only become implied through a concrete binding still on the finished record.
    The same reasoning rules out transitive breakage: a source stated in terms of another can
    only ever have been stated in terms of a concrete value that other contributed.

    The policy over (previous entry, this check's outcome):

    * (none, ``DEFERRED`` / ``BOUND``): record it -- the source's first assumption.
    * (none, bare identity via :meth:`record_identity`): no-op -- nothing was assumed.
    * (``DEFERRED``, ``DEFERRED``): overwrite -- the same open assumption, restated at its
      currently-resolved operands.
    * (``DEFERRED``, ``BOUND``): overwrite -- the deferral is discharged into a decided
      fact, strictly more informative.
    * (``DEFERRED``, bare identity): drop -- discharged into a syntactic identity, so
      nothing is assumed any more.
    * (``BOUND``, ``DEFERRED``): overwrite. Adequacy holds transitively; the record's job is
      each source's most-resolved current statement, not a history.
    * (``BOUND``, ``BOUND``): overwrite -- same fact restated, or a refined ``bound_here``.
    * (``BOUND``, bare identity): **keep** the ``BOUND`` entry. The binding is the
      assumption: the later identity holds only because that binding was made, so dropping
      the entry would erase what makes it true. The one cell where
      :meth:`record_identity` does not mirror :meth:`record`.
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

        ``bound_here`` is the raw ``UnifyResult.bindings`` this specific check produced --
        required (non-empty) when ``outcome`` is ``BOUND``, omitted (empty) otherwise. See
        :attr:`~qufzx.rewrite.rule.DimensionConstraint.bound_here`.
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

        Computed from the finished record rather than from a flag accumulated across passes,
        so a leg that deferred on one pass and bound (or resolved to an identity) on a later
        one does not leave a stale ``deferred=True`` on ``dimension_agreement``.
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

    "Surviving" means every leg of ``node`` except ``consumed_ref``, checked in
    input-then-output, original-index order for this one node -- the same per-node order
    :mod:`qufzx.rewrite.rules_library`'s ``_surviving_legs`` uses, though the two callers
    (this function for A, then again for B) do not combine into the same global order the
    builder assembles the merged node's ports in; see the module docstring, condition 6.

    ``bindings`` is the running, whole-candidate accumulator of every concrete symbol
    binding seen so far (starting with the connecting pair's own) -- this function updates
    it in place with each surviving leg's own concrete binding, on top of using it (via
    :func:`_resolve_with_bindings`) to refine ``shared_dim`` as it goes. Each leg's own
    ``Dim`` is resolved through that accumulator *before* being unified against
    ``shared_dim``, never in its raw form, exactly as :func:`_unify_phase_dims` does for a
    phase.

    Returns the (possibly refined) shared dimension, or a :class:`_ResolutionFailure` if any
    surviving leg's resolved dim is non-unifiable with it, or unifies but only via a binding
    that contradicts an earlier one -- either makes the whole candidate a non-match, exactly
    like a ``FAILURE`` on the connecting-leg pair itself, since forcing that leg onto
    ``shared_dim`` anyway would destroy a real dimension conflict rather than report it.
    Every leg's outcome is written into ``record`` under its own
    :meth:`~qufzx.rewrite.rule.ConstraintSource.surviving_leg` key -- deferred, bound, or
    (via :meth:`_ConstraintRecord.record_identity`) a bare identity.
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
                    source, leg_dim, shared_dim, ConstraintOutcome.BOUND,
                    bound_here=result.bindings,
                )
            else:
                record.record_identity(source)
            if bound_this_pass:
                conflict = _merge_bindings(bindings, result.bindings)
                if conflict is not None:
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
    """Unify every phase vector actually present (A's, then B's) against ``shared_dim``, in turn.

    Mirrors :func:`_unify_surviving_legs`'s accumulator discipline exactly, extended to
    phases (module docstring, condition 7): each phase's own ``Dim`` is first resolved
    through the running ``bindings`` accumulator (a phase whose symbol a *later* leg or an
    earlier phase in this same call already bound concretely must be checked against its
    resolved value, never its stale raw one), then unified against the *current*
    ``shared_dim``; a concrete binding refines both ``bindings`` and ``shared_dim`` in place
    before the next phase (if any) is examined. Two present phases therefore cannot each
    bind the same symbol against the same stale ``shared_dim``, and a binding a phase alone
    produces is folded into ``shared_dim`` immediately rather than held back as though a
    phase's binding could never matter to what a leg shares.

    Unlike a leg, a genuinely ``DEFERRED`` result, or a result whose binding is not
    concrete, is never accepted here -- see the module docstring, condition 7, for why a
    phase's own entries make reattaching under an unproven or non-concrete assumption
    unsafe. Returns a :class:`_ResolutionFailure` on any of those (or a contradictory
    rebind), making the whole candidate a non-match -- its ``equal_to`` is the ``shared_dim``
    value actually checked against the failing phase, not necessarily the caller's pre-call
    ``shared_dim``. Returns the (possibly refined) ``shared_dim`` on success, having written
    each present phase's binding-only success into ``record`` under its own
    :meth:`~qufzx.rewrite.rule.ConstraintSource.node_phase` key.

    Round 20, Task 7: previously returned ``(resolved_shared_dim, bound_names)``, threading a
    second, parallel accumulator of bound symbol names out to the caller alongside ``record``
    itself. ``resolve_fusion_match`` then read the *values* for those names back out of
    ``record.entries()`` while reading the *names* from this separately-threaded list --
    two different collections that happened to agree only because ``Dim.unify`` (Phase 5's
    placeholder) binds at most one symbol per call and a phase binding is always concrete, so
    ``phase_bound_names`` could never name anything ``record`` had not *also* just bound. See
    the module docstring's "Round 20" section for why that agreement is not something a
    future ``Dim.unify`` (Phase 10's real unifier) can be trusted to preserve, and why
    ``record`` is now the *only* source the caller reads for both.
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
            source, phase_dim, shared_dim, ConstraintOutcome.BOUND,
            bound_here=phase_unify.bindings,
        )
        new_concrete = dict(phase_unify.bindings)
        conflict = _merge_bindings(bindings, new_concrete)
        if conflict is not None:
            return _ResolutionFailure(
                _FailureReason.CONTRADICTORY_REBIND, phase_dim, shared_dim
            )
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

    Unlike every ``SURVIVING_LEG`` and ``NODE_PHASE`` check, the connecting pair relates its
    own two legs to *each other*, not to ``shared_dim`` (see
    :attr:`~qufzx.rewrite.rule.ConstraintSourceKind.CONNECTING_PAIR`): it is what seeds
    ``shared_dim`` in the first place, on the fixpoint's first pass, when both ports are
    still resolved through an empty ``bindings`` accumulator (so ``resolved_a`` there is
    ``port_a_dim`` itself, unify-ing trivially against ``shared_dim``, which
    :func:`resolve_fusion_match` also seeds at ``port_a_dim`` -- and ``resolved_b`` reduces
    to exactly the pre-fixpoint ``port_a_dim.unify(port_b_dim)`` this replaces).

    D2's fix: called once per pass, not once before the loop, so a later pass sees whatever
    a leg or phase check elsewhere in this same fixpoint has since bound -- the same
    treatment every other source already had. Both legs are resolved through the running
    ``bindings`` first, then unified against each other; a ``FAILURE`` here is a non-match,
    exactly like a leg's own. Returns the (possibly refined) shared dimension, or a
    :class:`_ResolutionFailure` on ``FAILURE`` or a contradictory rebind
    (:func:`_merge_bindings`).

    Unlike :func:`_unify_surviving_legs` and :func:`_unify_phase_dims`, *both* of this
    function's unify operands (``resolved_a``, ``resolved_b``) are pre-resolved through
    ``bindings`` before ``Dim.unify`` ever sees them -- there is no second, raw operand
    (those two functions each unify their own pre-resolved value against the caller's
    ``shared_dim`` parameter directly, unresolved). A ``CONTRADICTORY_REBIND`` here is
    therefore not merely unreached in practice (like the other two, and like
    :func:`_merge_bindings` itself) but unconstructible by any input to this function at
    all: any symbol free in ``resolved_a``/``resolved_b`` is, by definition, not already a
    key in ``bindings`` -- if it were, resolution would have already replaced it with its
    bound value, leaving nothing free for ``Dim.unify`` to (re)bind.
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
            source, resolved_a, resolved_b, ConstraintOutcome.BOUND,
            bound_here=result.bindings,
        )
    else:
        record.record_identity(source)
    if bound_this_pass:
        conflict = _merge_bindings(bindings, result.bindings)
        if conflict is not None:
            return _ResolutionFailure(
                _FailureReason.CONTRADICTORY_REBIND, resolved_a, resolved_b
            )
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
    """Re-verify, from scratch, that the finished fixpoint's own claim actually holds.

    The fixpoint's own definition made explicit, as a structural guarantee rather than a
    property only tests assert (D1's required post-loop check): the connecting pair's two
    legs, every surviving leg of both nodes, and every present phase -- each resolved under
    the *final* ``bindings`` -- must unify with the *final* ``shared_dim`` without
    ``FAILURE``. Given the fixpoint only exits *via its own convergence break* (a full pass
    that adds nothing to either ``shared_dim`` or ``bindings`` -- D1's fixed termination
    signal) and every ``bindings`` update is contradiction-checked (:func:`_merge_bindings`),
    this is unreachable on that path: the loop's last pass already re-checked every one of
    these against the state this function re-checks them against again. It is called anyway,
    unconditionally on that path, rather than left to be unreachable in principle only.

    Where this now runs (Phase 5 post-closing audit round 18, Defect 3 -- corrected from a
    prior version of this docstring, which claimed unreachability unconditionally):
    :func:`resolve_fusion_match` calls this function *only* on the stabilised-convergence
    path just described. A phase-dim ``FAILURE`` inside the fixpoint loop, and the
    ``_MAX_FIXPOINT_PASSES`` budget-exhaustion path, both ``return`` directly from within
    the loop -- neither one reaches this call at all any more. Before this fix, every exit
    from the loop (a genuine convergence break, *and* a phase-dim ``FAILURE`` that ``break``
    out of the loop with ``phase_dims_agree = False``) fell through to this same
    unconditional call; for the ``FAILURE`` case, that re-verified phases under the exact
    unify contract that had just failed them, so this function necessarily returned
    ``False`` -- which was not evidence of some new problem, but this function faithfully
    reporting the failure that was already known and had already been given its own
    dedicated report. Reaching it from that path was the defect (see
    :func:`resolve_fusion_match`'s inline commentary at its phase-failure return for the
    full account), not anything wrong with this function itself, whose own re-verification
    logic is unchanged. Its unreachability claim is now proven, not merely asserted, by
    ``tests/test_phase5_exhaustive_oracle.py`` and
    ``tests/test_fusion_properties.py::TestSpiderFusionProperties
    ::test_random_diagrams_fuse_soundly``, both of which wrap this function to assert it
    returns ``True`` on every call across their respective sweeps.
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
    """Human-readable summary of the connecting pair's *finished* record entry.

    Built from the finished record, not the fixpoint's first pass alone (D2's knock-on fix):
    since the connecting pair is now re-derived every pass, whether it ended up deferred,
    bound, or (once a later pass's binding discharges it) a bare identity is only known once
    the fixpoint itself has finished -- reading it off the first pass alone, as the pre-fix
    code did, could contradict the finished ``dimension_constraints`` it sits beside in the
    certificate.

    Every operand and binding rendered below is read directly off ``entry`` -- the same
    :class:`~qufzx.rewrite.rule.DimensionConstraint` ``dimension_constraints`` itself is
    built from -- never recomputed against the final ``port_a_dim``/``port_b_dim``/
    ``bindings`` state (Phase 5 post-closing audit round 19, Defect 4 continued). Recomputing
    was the round-18 fix's own residual bug: this pass's ``resolved_a == resolved_b`` need
    not equal ``entry.assumed == entry.equal_to`` (a surviving leg elsewhere can concretize
    both sides *after* the connecting pair's own check ran), and a symbol-occurrence
    intersection against ``bindings`` can attribute a binding some other check made to this
    entry, or drop a real one that never reached ``bindings`` because it was non-concrete.
    ``port_a_dim``/``port_b_dim``/``bindings`` are used for exactly one case: no entry was
    ever recorded at all, which only happens when the pair was a bare identity on every pass
    -- there ``resolved_a`` and ``resolved_b`` are trivially equal, so there is no second
    source of truth to diverge from.
    """
    entry = record.entry_for(ConstraintSource.connecting_pair())
    if entry is None:
        resolved_a = _resolve_with_bindings(port_a_dim, bindings)
        resolved_b = _resolve_with_bindings(port_b_dim, bindings)
        return f"{resolved_a} == {resolved_b}"
    if entry.outcome is ConstraintOutcome.DEFERRED:
        return f"{entry.assumed} == {entry.equal_to} (deferred, assumed)"
    # entry.outcome is BOUND: render exactly what this check's own unify bound
    # (entry.bound_here), not a value looked up by symbol coincidence.
    #
    # Round 20, Task 6: entry.bound_here being non-empty here is no longer merely assumed --
    # DimensionConstraint.__post_init__ now enforces, structurally, that a BOUND outcome
    # always carries a non-empty bound_here (see that class's docstring). This is exactly the
    # class round 18's Defect 3 named: a guard whose correctness argument used to rest on
    # every caller happening to get it right is now backed by a constructor that rejects the
    # violating case outright. The `entry.bound_here and` half of the condition below is
    # therefore dead in the sense that it can no longer be False for a BOUND entry -- kept
    # anyway as a direct, readable mirror of the invariant it relies on, rather than deleted
    # and replaced with an assert (which would just relocate the same trust one line up
    # without making it any more visible). The fall-through branch's claim -- "bound to a
    # non-concrete Dim" -- is therefore now provably reachable only for a genuine
    # non-concrete binding, not for an unrelated bug that happened to leave bound_here empty.
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

    Shared by :func:`resolve_fusion_match`'s stabilised-success path and its phase-failure
    path (Defect 3, Phase 5 post-closing audit round 18): both must report condition 6 from
    the *leg* sweep's own state -- ``shared_dim``/``bindings`` exactly as the connecting
    pair and every surviving leg were actually checked against -- never from a state a later
    phase check has since advanced past what the legs saw. See ``resolve_fusion_match``'s
    own inline commentary at its call sites for why the two states can differ within one
    fixpoint pass.
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
    """Substitute ``bindings`` into ``phase``'s entries, then reattach the result to ``shared_dim``.

    Returns the reattached vector together with the subset of ``bindings`` actually
    substituted into an entry's *value* -- distinct from ``shared_dim``, which every caller
    reattaches to whether or not any entry mentioned it.

    Public, not ``_reattach_phase``: both :func:`resolve_fusion_match` here and
    :mod:`qufzx.rewrite.rules_library`'s :func:`~qufzx.rewrite.rules_library._over_shared_dim`
    treat it as the shared match-approval / build-applicability contract, which makes it
    public API in all but name.

    It is the one shared resolution of the ``_over_shared_dim`` defect family (see
    ``rules_library``'s module docstring, "Dimension of the merged node"), used identically
    as a trial construction here and to build the merged phase there. A phase legally stated
    over a symbolic dimension -- ``PhaseVector(d, {1: Phase.root_of_unity(1, d)})`` -- whose
    ``shared_dim`` is resolved past by a binding ``d := 2`` must not be reattached to the
    concrete dimension with its entries verbatim: an entry ``1/d turns`` on a container
    dimension of ``2`` denotes a different angle, and silently keeps citing a symbol its own
    container has resolved past.

    The alternative was to have ``dimension_agreement`` refuse any candidate whose phase
    entries reference a bound dimension symbol. That satisfies the spec's "phases are
    first-class symbolic objects" only in the thinnest sense -- never touching them -- while
    rejecting a fusion that is well-defined at exactly the binding
    :attr:`FusionMatch.bindings` already records. Substituting is what
    :meth:`~qufzx.algebra.phase.PhaseVector.substitute` exists to do, and it keeps the
    resulting phase the actual angle implied by the binding rather than an approximation or
    a refusal. It also keeps match-approval and build-applicability one predicate: this
    function's return value, or the error it raises, is identical from either caller.

    Raises :class:`~qufzx.algebra.phase.PhaseDomainError` if, after substitution, an entry's
    index falls outside ``shared_dim``'s range. Substitution changes only an entry's value,
    never its index, so this is the same index-bound check either caller would hit
    reattaching the unsubstituted entries.
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

    Called for both endpoints of every wire in the diagram, and (Phase 5 post-closing audit
    round 18, Defect 2) for every ``boundary_inputs``/``boundary_outputs`` entry -- see
    "Malformed wire references" in the module docstring for why this must not depend on any
    other property of the wire (or, now, the boundary list) or the candidate pair it might
    otherwise sit on. ``wire_or_boundary_ref`` is only used to phrase the raised message: a
    ``Wire`` for the wire-endpoint call sites (``ref`` is one of its own two endpoints, so
    the message can name the whole wire for context), or the bare ``PortRef`` itself for a
    boundary entry (there is no enclosing ``Wire`` to name -- the ref *is* the reference,
    passed as both parameters at the boundary call site).
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
    """The one verification predicate behind every :data:`FUSION_SIDE_CONDITIONS` entry.

    Returned by :func:`resolve_fusion_match`, computed fresh from ``(diagram, a_id, b_id,
    wire)`` alone -- never from a pre-existing :class:`FusionMatch`'s own fields. See the
    module docstring's "One verification predicate" paragraph for why this exists and who
    calls it.

    ``outcomes`` covers exactly the seven :data:`FUSION_SIDE_CONDITIONS` names, in that
    order, each independently derived. ``passed`` is ``True`` iff every one of them passed.
    When ``True``, ``shared_dim``, ``bindings``, and ``dimension_constraints`` are the ground
    truth to build a merged node from -- the only values
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` may use for graph surgery,
    never a pre-existing match's own same-named fields, which a hand-built or foreign
    ``FusionMatch`` could have fabricated. When ``False``, ``bindings`` and
    ``dimension_constraints`` are best-effort placeholders (computed only as far as
    resolution got before the first failing condition) and must not be used for anything but
    diagnostics -- a condition that was never reached because an earlier one already failed
    is still recorded, as a failing outcome whose detail says so, so ``outcomes`` always has
    exactly seven entries regardless of where resolution stopped.

    ``shared_dim`` is ``Dim | None``, not a placeholder ``Dim`` (Phase 5 post-closing audit):
    a failed resolution has no shared dimension to report -- the two legs may not even agree
    on one, which is exactly why resolution failed -- so unlike ``bindings``/
    ``dimension_constraints`` (which have a natural, harmless "nothing accumulated yet"
    empty value), there is no ``Dim`` that means "no shared dimension" without also being a
    plausible-looking real one. ``None`` makes "resolution failed, don't read this" a type
    error at every call site rather than a caller-side discipline of checking ``passed``
    first and hoping every caller remembers to. ``shared_dim`` is only ever non-``None`` when
    ``passed`` is ``True``, and every reader of this field (``find_matches``,
    ``spider_fusion_builder``) is downstream of its own ``if not resolution.passed`` guard.
    """

    passed: bool
    shared_dim: Dim | None
    bindings: Mapping[str, Dim]
    dimension_constraints: tuple[DimensionConstraint, ...]
    outcomes: tuple[SideConditionOutcome, ...]


def _connecting_pair_failure_detail(failure: _ResolutionFailure) -> str:
    """Render a connecting-pair :class:`_ResolutionFailure`, distinguishing its two causes.

    Phase 5 post-closing audit round 23, Task 3: a non-unifying pair and a pair that unifies
    fine but whose binding contradicts an earlier one are different failures -- the former
    means the two legs are provably incompatible, the latter means they are compatible with
    each other but not with an assumption a different check already made -- so they get
    genuinely different wording, derived from ``failure.reason`` rather than folded into one
    generic "does not unify" string regardless of which one actually happened.
    """
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"{failure.assumed} == {failure.equal_to} unifies, but only by binding a "
            "symbol to a value that contradicts an earlier binding accumulated this "
            "resolution (see _merge_bindings)"
        )
    return f"{failure.assumed} != {failure.equal_to}: the connecting pair does not unify"


def _leg_failure_detail(side: str, failure: _ResolutionFailure) -> str:
    """Render a surviving-leg :class:`_ResolutionFailure` for node ``side`` ('A' or 'B').

    Same distinction as :func:`_connecting_pair_failure_detail`, for the leg-sweep call
    sites.
    """
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
    """Render a phase-dimension :class:`_ResolutionFailure`, distinguishing all four causes.

    A phase has more ways to fail than a leg or the connecting pair (module docstring,
    condition 7): a genuine ``FAILURE``, a ``DEFERRED`` unify (tolerated for a leg, not for a
    phase), a binding to a non-concrete ``Dim`` (also tolerated for a leg, not for a phase),
    and a contradictory rebind. Each gets its own wording -- this is the *in-loop* phase
    failure detail; the distinct post-loop ``reattach_phase`` detail (an entry falling out of
    range once every binding is substituted in, rather than a non-unifying ``Dim``) is built
    separately at its own call site in :func:`resolve_fusion_match`, deliberately, since it
    is a genuinely different failure this function never sees.
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

    Otherwise, a human-readable description of what else claims it: a second wire, or a
    boundary entry (or both). Recomputed from ``diagram`` alone on every call -- consistent
    with every other condition :func:`resolve_fusion_match` decides -- rather than threaded
    in from a precomputed whole-diagram accumulator the way :func:`find_matches` used to
    build one (see the module docstring, "Match-implies-applicable and multiply-claimed
    ports", Phase 5 post-closing audit round 23, Task 2).
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

    The single shared predicate behind all seven conditions in the module docstring -- see "One
    verification predicate" there for the full account of why this function exists and the
    Phase 5 round-12 audit defect (A1/A2/A4) it closes. :func:`find_matches` calls this once
    per candidate wire to decide whether to report a match at all, and to populate the
    :class:`FusionMatch` it returns. :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`
    calls it again, fresh, against the diagram it was actually handed, and trusts only this
    function's return value for graph surgery -- never ``match.shared_dim``,
    ``match.bindings``, or an unverified ``node_a.generator_type`` -- so a foreign or
    hand-built match cannot smuggle a fabricated value past the builder by simply asserting
    it in a passing-looking ``side_condition_outcomes`` tuple.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` for a structurally malformed
    request: ``a_id == b_id``, either node id absent from ``diagram``, ``wire`` not actually
    incident on both ``a_id`` and ``b_id``, or either of ``wire``'s own endpoints naming an
    unknown node id or an out-of-range port index (via :func:`_validate_wire_endpoint`).
    These are requests that cannot even be evaluated, not candidates that evaluate to
    "no" -- the same domain/grammar split :mod:`qufzx.rewrite.rule`'s module docstring
    states for this package as a whole.

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

    # Defect 1 (Phase 5 post-closing audit): everything above validates that ``wire``
    # *looks* like it could join a_id and b_id -- its own endpoints name real nodes and
    # ports, and its node ids are exactly {a_id, b_id} -- but none of that establishes
    # that ``wire`` is actually an element of ``diagram.wires`` rather than a freestanding
    # ``Wire`` object a caller merely constructed to look like one. A wire ghost-written
    # against two real, correctly-incident ports is otherwise accepted by every check
    # above and passed on to graph surgery, which then consumes a wire the diagram never
    # had. This is a malformed request, the same category as non-incidence above, so it
    # is checked here, before any side condition is evaluated, and raises the same
    # RewriteGrammarError. ``all_wires`` is snapshotted once (``Diagram.wires`` rebuilds a
    # fresh ``frozenset`` on every access) and reused below for ``other_wire_count``.
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

    # Two distinct facts, reported under the condition that actually decides them (Phase 5
    # post-closing audit): "do the two nodes carry the identical generator type" and "is
    # that generator type one this pattern is registered to fuse at all" used to be folded
    # into one -- an unregistered same-typed pair reported same_generator_type=True (the
    # types genuinely are equal) with the real reason ("not a registered fusable generator
    # type") buried inside consumed_wire_direction_permitted_for_color's detail instead,
    # a condition whose own declared description is about wire direction, not fusability.
    # same_generator_type's own declared description ("both nodes are the same *registered*
    # spider color") already promised this; the implementation now actually checks it.
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
            # None, not a placeholder concrete Dim: a failed resolution has no shared
            # dimension a caller may read (see FusionResolution's own docstring) -- making
            # this unrepresentable at the type level, rather than a caller-side "check
            # passed first" discipline, closes the Phase 5 post-closing audit's
            # "_failed() returns a meaningless dimension a caller ignoring passed could
            # silently read" defect.
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

    # By construction, same_type above already guarantees node_a.generator_type.name is in
    # _FUSABLE_GENERATOR_NAMES -- resolution returns early via _failed() otherwise, so this
    # condition is only ever reached for a registered fusable, same-typed pair. No separate
    # "not fusable" branch is needed here any more (Phase 5 post-closing audit).
    same_direction = ref_a.direction == ref_b.direction
    direction_ok = (
        not same_direction
        or node_a.generator_type.name in _SAME_DIRECTION_FUSABLE_GENERATOR_NAMES
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

    # Condition 5: decided from (diagram, a_id, b_id, wire) alone, before the dimension
    # fixpoint below, exactly as every other condition is -- see the module docstring,
    # "Match-implies-applicable and multiply-claimed ports" (Phase 5 post-closing audit
    # round 23, Task 2). A candidate that fails this has no legal port_mapping regardless of
    # what the fixpoint would find, so it is checked first rather than after paying for a
    # fixpoint whose result would be discarded either way.
    claim_conflict_a = _consumed_port_claim_conflict(diagram, ref_a, wire)
    claim_conflict_b = _consumed_port_claim_conflict(diagram, ref_b, wire)
    claims_ok = claim_conflict_a is None and claim_conflict_b is None
    claim_detail = (
        f"neither {ref_a} nor {ref_b} is claimed by another wire or listed on a boundary"
        if claims_ok
        else "; ".join(d for d in (claim_conflict_a, claim_conflict_b) if d is not None)
    )
    outcomes.append(
        SideConditionOutcome("consumed_ports_singly_claimed", claims_ok, claim_detail)
    )
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
    shared_dim = port_a.dim
    bindings: dict[str, Dim] = {}

    # Conditions 6 and 7 run as one bounded fixpoint: each pass re-derives the connecting
    # pair's own equality, then re-unifies every surviving leg of both nodes, then every
    # present phase's own Dim -- each against shared_dim (or, for the connecting pair,
    # against the other connected leg) as of the point reached *so far in that same pass*,
    # refining `bindings` and shared_dim in place on any concrete binding. The pass repeats
    # until a full pass adds nothing: both shared_dim and bindings are unchanged from the
    # pass's own start (D1's fix -- see below for why shared_dim stopping alone is not a
    # sound termination signal). `bindings` is not pass-scoped: it is the single
    # whole-candidate accumulator every pass reads from and writes into, so each successive
    # pass is strictly more informed than the last. It is also monotone -- a key is only
    # ever added, never rebound to a different value -- but not because _merge_bindings
    # rejects a contradictory rebind (round 23 correction: it never actually gets the
    # chance to. Every operand reaching a unify call in this fixpoint has already been
    # passed through _resolve_with_bindings first, which substitutes every symbol `bindings`
    # already names with its bound value; a symbol that is already bound is therefore no
    # longer *free* in what gets unified, so Dim.unify's own UnifyResult.bindings can never
    # name it again. Monotonicity is a consequence of that pre-resolution discipline, not of
    # _merge_bindings' own contradiction check -- see that function's docstring for what its
    # guard is actually for. Duplicate assumptions are ruled out by the *record*, not by
    # this loop: `record` is keyed by ConstraintSource, so a source re-derived on a later
    # pass replaces its own entry rather than appending a second one (see _ConstraintRecord).
    #
    # D1's root cause: "shared_dim stopped changing" is not the same fact as "bindings
    # stopped changing". bindings can grow on a pass whose new binding does not touch any
    # symbol appearing in shared_dim itself (e.g. binding a symbol that occurs only in a
    # surviving leg's own dim, or rebinding a symbol to the value it already holds) --
    # exiting on shared_dim alone then leaves whatever was checked earlier in that very pass
    # unre-checked against the newly accumulated bindings, which is exactly how an
    # unsatisfiable constraint set (e.g. e*f == 2 and e == 2 and f == 2, from legs
    # [e*f, e, f]) went undetected. Checking both signals -- i.e. requiring an entire pass to
    # add nothing to either shared_dim or bindings before exiting -- is what actually catches
    # it: the pass that binds e := 2 and f := 2 does not itself stabilise (bindings grew), so
    # a further pass runs and re-resolves e*f through those bindings before re-unifying it,
    # surfacing the contradiction as an ordinary Dim.unify FAILURE on that later pass -- not
    # by _merge_bindings. _verify_fixpoint_closure is the same check restated as a
    # structural post-loop guard on the convergence path, rather than something left to pass
    # repetition alone to have caught.
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

        # Defect 3 (Phase 5 post-closing audit round 18): snapshotted *before* calling
        # _unify_phase_dims, not read back out of `shared_dim`/`bindings` after it returns.
        # This pass's leg sweep (connecting pair, A's surviving legs, B's surviving legs,
        # all three just above) has, as of this point, been fully verified against exactly
        # this shared_dim/bindings state -- that is what condition 6 (dimension_agreement)
        # must be reported against if a phase now fails. _unify_phase_dims can bind phase
        # A's own symbol -- refining both `bindings` and `shared_dim` in place -- before
        # failing on phase B in that same call (see its own docstring); reporting condition
        # 6 against the state *after* that call would claim the legs were verified against
        # a shared_dim they were never actually checked against.
        leg_verified_shared_dim = shared_dim
        leg_verified_bindings = dict(bindings)

        phase_result = _unify_phase_dims(node_a, node_b, a_id, b_id, shared_dim, bindings, record)
        if isinstance(phase_result, _ResolutionFailure):
            # Root cause (Defect 3): a phase-dim FAILURE is decided -- condition 7 does not
            # hold -- and is reported as exactly that, directly, rather than falling through
            # to _verify_fixpoint_closure. A prior version of this function let every break
            # out of this loop (phase failure or genuine convergence alike) fall through to
            # that post-loop closure check; closure re-verifies phases from scratch under
            # the same unify contract that just failed, so it necessarily failed too,
            # reporting BOTH dimension_agreement and phase_dimension_agreement as failed
            # with the closure guard's own "this is unreachable" message -- which was false
            # exactly here: the guard's unreachability argument (every check the closure
            # re-verifies was already verified this same pass, against this same state) only
            # holds when the loop reaches its termination condition below, never when it
            # exits via a phase failure. See _verify_fixpoint_closure's own docstring, fixed
            # to describe only the path it now actually runs on. dimension_agreement is
            # reported True (from the leg-verified snapshot above, via
            # _dimension_agreement_outcome -- a leg-sweep FAILURE already returned above,
            # via _failed, so reaching here means every leg genuinely did unify) with a
            # leg-accurate detail; phase_dimension_agreement is reported False with the
            # dedicated per-phase detail below -- never the closure-guard string, which
            # therefore appears in neither outcome for this path.
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
            # Same placeholder convention as _failed() (Phase 5 post-closing audit,
            # pre-round-18 Defect 3): shared_dim=None, bindings/dimension_constraints empty
            # on any failure -- see FusionResolution's own docstring. Not routed through
            # _failed() itself since that helper always marks its ``remaining_names`` False
            # and appends nothing for dimension_agreement; this path needs the opposite mix
            # (dimension_agreement True, phase_dimension_agreement False), which _failed()
            # cannot express.
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
        # The cap is a resolver *iteration budget*, not a dimension disagreement, and it is
        # reported as exactly that. bindings is monotone (never shrinks, never rebinds --
        # _merge_bindings) and is drawn from the finite free-symbol set of node_a's legs and
        # phase, node_b's legs and phase, and shared_dim's own lineage from port_a.dim; a
        # pass that does not stabilise therefore strictly grows bindings by at least one
        # fresh key, which bounds the number of non-stabilising passes by that finite symbol
        # count. The guard is kept anyway, conservatively refusing the candidate rather than
        # looping forever, because that bound rests on Dim.unify's current contract and
        # Phase 10 replaces its body. Both conditions 6 and 7 are reported failed, with the
        # same detail: the fixpoint decides them jointly, so when it does not terminate
        # neither one was decided, and blaming either alone would send a reader to the wrong
        # place.
        fixpoint_budget_exhausted = True

    if fixpoint_budget_exhausted:
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            f"the bounded leg/phase resolution fixpoint did not stabilise within "
            f"{_MAX_FIXPOINT_PASSES} passes (_MAX_FIXPOINT_PASSES): this is a resolver "
            "iteration budget, not a dimension or phase-dimension disagreement -- neither "
            "condition was decided",
        )

    # Reached only via the loop's own convergence break above (a phase failure returns
    # directly, above; budget exhaustion returns directly, just above) -- see
    # _verify_fixpoint_closure's own docstring for why its "unreachable" claim is actually
    # true on exactly this path, and only this path (Defect 3, Phase 5 post-closing audit
    # round 18).
    if not _verify_fixpoint_closure(
        node_a, node_b, a_id, b_id, ref_a, ref_b, port_a.dim, port_b.dim, shared_dim, bindings
    ):
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            "post-loop closure check failed: a resolved leg, phase, or the connecting pair "
            "does not unify with the final shared_dim under the final bindings -- see "
            "_verify_fixpoint_closure; this is unreachable on the convergence path this "
            "call site is reached from (a phase failure or budget exhaustion both return "
            "before reaching here) given the fixpoint's own termination guarantee, and is "
            "checked anyway as a structural guard, not a property left to tests alone",
        )

    outcomes.append(
        _dimension_agreement_outcome(port_a.dim, port_b.dim, shared_dim, bindings, record)
    )

    phase_a_dim = node_a.phase.dim if node_a.phase is not None else None
    phase_b_dim = node_b.phase.dim if node_b.phase is not None else None
    phase_dims_present = tuple(d for d in (phase_a_dim, phase_b_dim) if d is not None)

    # reattach_phase's own index-bound check must run against the FINAL shared_dim
    # (post-fixpoint), not an intermediate one from an earlier pass: substitution only
    # changes an entry's *value*, never its index, so an entry that falls out of range only
    # once the fixpoint's own later bindings resolve is still caught here. This is a
    # genuinely different failure mode than a phase-dim FAILURE inside the loop above (see
    # _phase_failure_detail: that is a non-unifying Dim, a DEFERRED unify, or a non-concrete
    # binding) -- this one is a *unifying* Dim whose own entries fall out of range once
    # substituted, discoverable only after the fixpoint has fully converged and
    # reattach_phase can be tried for real. Round 23 correction: a prior version of this
    # detail string listed every possible cause verbatim (including the in-loop ones this
    # path cannot actually be reached by, since a phase-dim FAILURE/DEFERRED/non-concrete
    # binding already returns above), making the two phase_dimension_agreement failure
    # details near-identical copies of each other -- see the module docstring's Task 3 note.
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

    # Round 20, Task 7: both the rendered *names* and their *values* are now read off the
    # single source this detail claims to describe -- ``record.entries()`` -- rather than
    # names from one accumulator (the now-deleted ``phase_bound_names``, threaded out of
    # ``_unify_phase_dims`` across passes) and values from another (``record`` itself). The
    # two used to agree only because ``Dim.unify``'s current placeholder body binds at most
    # one symbol per call and a phase binding is only ever recorded once fully concrete --
    # true today, not a property this rendering step should depend on (see the module
    # docstring's "Round 20" section, completing round 18's Defect 4 / round 19's Task 1
    # class: a rendered detail must read every operand it prints off the same record it
    # claims to describe, including the keys it iterates, not only the values it looks up).
    # Walked in ``record.entries()``'s own first-derivation order (see that method's
    # docstring) and de-duplicated with ``dict.fromkeys`` to preserve that order while
    # dropping a name a later pass re-bound under the same key (D6's original concern,
    # preserved here even though the accumulator it was patching over is gone).
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
                + ", ".join(
                    f"{name} := {phase_bound_values[name]}" for name in unique_bound_names
                )
                if unique_bound_names
                else ""
            )
        )
    )
    outcomes.append(
        # Always deferred=False: unlike condition 6, a genuinely DEFERRED phase-dim
        # unify is rejected outright (see _unify_phase_dims) rather than
        # accepted-and-flagged, so a *passing* phase_dimension_agreement outcome is
        # never itself resting on an undecided unify -- only, at most, on a binding
        # (which dimension_constraints records, but which this flag -- following
        # condition 6's own convention -- does not count as "deferred").
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

    Never mutates ``diagram``, and does not require ``diagram`` to be well-formed --
    :func:`~qufzx.diagram.validate.validate` is never called here. Returns matches sorted
    by ``(a_id, b_id)``, tiebroken by the consumed wire's own per-side (direction, index).
    """
    # Malformed-wire detection (an unknown node id or an out-of-range port index) must be
    # independent of every other property of the wire or the candidate pair it happens to
    # sit on -- generator color, fusable-color-ness, wire direction, single-vs-parallel
    # wiring, and self-loop-ness alike -- so every wire's both endpoints are checked here,
    # before any grouping or filtering. Checking this only after grouping (as an earlier
    # version did) let a malformed wire escape undetected as a bare non-match whenever it
    # happened to be dropped first by the self-loop skip or the parallel-wire-pair filter
    # below, masking the same structural defect differently depending on unrelated shape.
    # Snapshotted once, not re-read from ``diagram.wires`` on every pass below: the two
    # passes over the wire set (malformed-endpoint check, pair grouping) used to each
    # re-materialise ``diagram.wires`` independently, doubling the cost of building whatever
    # collection backs that property for no reason -- neither pass needs a live view, and
    # neither mutates ``diagram``.
    #
    # Sorted, not the raw frozenset (Phase 5 post-closing audit round 18, Defect 1):
    # ``diagram.wires`` is a frozenset, and ``Wire``'s hash folds in ``Direction``'s
    # member-name hash, which is PYTHONHASHSEED-dependent. The malformed-endpoint pass below
    # raises on the *first* offending wire it finds -- with more than one malformed wire in
    # the diagram, an unsorted iteration would report a different one (a different exception
    # message) across processes. The later pass (pair grouping) would still produce the same
    # *matches* even unsorted, since ``find_matches`` sorts its returned tuple explicitly
    # below regardless -- but sorting once, up front, keeps both passes uniformly
    # deterministic rather than leaving readers to work out which one needs it and which
    # merely happens not to.
    wires = tuple(sorted(diagram.wires, key=lambda w: w.sort_key()))

    for wire in wires:
        _validate_wire_endpoint(diagram, wire, wire.a)
        _validate_wire_endpoint(diagram, wire, wire.b)

    # Defect 2 (Phase 5 post-closing audit round 18): a boundary entry naming an unknown
    # node id or an out-of-range port index is held to the identical standard as a wire
    # endpoint above -- both are references ``_remap_endpoint`` (in
    # :mod:`qufzx.rewrite.engine`) treats identically once a match reaches ``apply``, so a
    # malformed one of either kind must be caught here, before any candidate is even
    # grouped, rather than only on the wire side. Before this fix, a boundary_inputs/
    # boundary_outputs entry naming no real port at all (e.g. an out-of-range index on a
    # node with fewer legs, or an unknown node id -- both legitimately constructible on an
    # un-validated diagram, since :mod:`qufzx.diagram.graph` is deliberately permissive, see
    # that module's docstring) reached ``apply`` unexamined as long as it did not happen to
    # sit on a consumed port; when it did, ``apply``'s step 5 (via ``_remap_endpoint``)
    # raised ``RewriteDomainError`` -- a *different* error, from a *different* step, than
    # this module's own "malformed wire reference" contract promises, and one this module's
    # own module docstring's "Match-implies-applicable" section did not cover at all,
    # despite ``find_matches`` returning a match whose ``apply`` was not, in fact,
    # guaranteed to succeed cleanly.
    #
    # Sorted for the same reason the wire pass above is (Phase 5 post-closing audit round
    # 18, Defect 1): ``PortRef``'s hash is PYTHONHASHSEED-dependent (via ``Direction``), and
    # ``boundary_inputs``/``boundary_outputs`` are already ordered tuples -- but a malformed
    # entry on one list should not have its reported order interleaved arbitrarily with the
    # other, so both lists are walked in their own declared order, boundary_inputs first
    # (the same order ``Diagram.boundary_inputs``/``boundary_outputs`` themselves are always
    # read in elsewhere in this codebase), each entry checked via the same
    # :func:`_validate_wire_endpoint` machinery the wire pass uses -- reusing that function
    # (renamed in spirit only; it validates *any* ``PortRef``, not just a wire's own) rather
    # than a second, hand-duplicated existence/range check that could drift out of sync with
    # it. ``wire`` is only used by :func:`_validate_wire_endpoint` to name the offending
    # object in its raised message; passing the ref itself there (rather than a real
    # ``Wire``) is what the ``wire_or_boundary_ref`` parameter name below documents.
    for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs):
        _validate_wire_endpoint(diagram, ref, ref)

    # Defect 2 (match-implies-applicable), Phase 5 post-closing audit round 23, Task 2: a
    # port claimed by more than one wire, or both wired and listed on a boundary, is not a
    # legitimate fusion occurrence -- fusing across it would ask the builder to remap a
    # consumed port that a *third* reference (another wire, or a boundary entry) also still
    # names, which qufzx.rewrite.engine.apply's port_mapping coverage check (step 5)
    # correctly refuses to do silently. This used to be enforced by a bare filter here,
    # before any FusionMatch was even constructed -- it is now condition 5
    # (``consumed_ports_singly_claimed``), decided by :func:`resolve_fusion_match` itself
    # from ``diagram`` alone, the same single decision point every other condition already
    # goes through (see the module docstring, "Match-implies-applicable and multiply-claimed
    # ports"). No separate filter is needed here any more: a candidate that fails it is
    # simply a resolution whose ``passed`` is ``False``, dropped by the ordinary
    # ``if not resolution.passed: continue`` below.
    candidates_by_pair: dict[frozenset[NodeId], list[Wire]] = {}
    for wire in wires:
        if wire.a.node_id == wire.b.node_id:
            continue
        key = frozenset((wire.a.node_id, wire.b.node_id))
        candidates_by_pair.setdefault(key, []).append(wire)

    # Flattened once so the loop below stays single-level: every wire in every candidate
    # pair. How many *other* wires join that same pair (condition 3,
    # ``parallel_wires_become_self_loops``) is recomputed, from ``diagram`` alone, inside
    # :func:`resolve_fusion_match` below -- not threaded through here -- so that function
    # stays the single source of truth for every one of the seven side conditions, not six
    # of them plus one still assembled by this loop.
    wire_candidates = [
        wire for connecting_wires in candidates_by_pair.values() for wire in connecting_wires
    ]

    matches: list[FusionMatch] = []
    for wire in wire_candidates:
        a_id, b_id = _ordered_pair(wire)

        # See the module docstring, "One verification predicate": conditions 2 and 4-7 are
        # decided by exactly this call, the same function
        # :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls again to
        # re-verify the match before trusting any of its fields -- not a second,
        # independently-maintained copy of this logic.
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
