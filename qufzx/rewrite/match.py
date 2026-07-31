"""The fusion matcher: finds occurrences of same-color spider fusion, and only that pattern.

Phase 5 implements exactly one :class:`~qufzx.rewrite.rule.Pattern`: two spiders of the
same generator type (both Z or both X), joined by exactly one wire, whose connected legs
agree on dimension. Every other pattern shape (bialgebra, Hopf, copy, identity removal,
...) is out of scope until Phase 11.

Side conditions checked, in the order applied below (see ``FUSION_SIDE_CONDITIONS`` for
their declared names and one-line descriptions, and :class:`FusionMatch` for where their
per-candidate outcomes are recorded):

1. ``distinct_nodes`` -- the two nodes are not the same node. Enforced structurally: a
   wire whose two endpoints share a node id (a self-loop on one spider) is excluded from
   candidate grouping entirely, before any other condition is even reached, since a
   self-loop is not a joining wire between two nodes at all.
2. ``same_generator_type`` -- both nodes carry the identical, registered
   :class:`~qufzx.diagram.generators.GeneratorType` (Z/Z or X/X; Z/X never matches).
3. ``single_connecting_wire`` -- exactly one wire joins the candidate pair. Two or more
   wires between the same pair is refused outright (not matched at all): fusing across one
   of them would leave the others as self-loops on the merged spider, which is Hopf/copy
   territory (Phase 11), with a different scalar than plain fusion.
4. ``wire_direction_output_to_input`` -- the consumed wire runs from an OUTPUT port of one
   node to an INPUT port of the other. This is load-bearing, not cosmetic, and is checked
   uniformly for both Z and X even though it only bites for X. The graph model in
   :mod:`qufzx.diagram.graph` permits an OUTPUT-OUTPUT (or INPUT-INPUT) wire; per
   :mod:`qufzx.semantics.denote`'s axis convention, X applies the Fourier matrix ``F`` to
   output axes and ``conj(F)`` to input axes. An output-to-input contraction pairs one
   ``F`` with one ``conj(F)`` on the shared leg, giving ``F^dagger F = I`` -- fusion is
   scalar-free. An output-to-output (or input-to-input) contraction instead pairs ``F``
   with ``F``, giving ``F^T F``, which is a nontrivial permutation matrix, not the
   identity -- that is a different (and, for Phase 5, unimplemented) rule, not fusion. For
   Z the tensor is diagonal so both wirings coincide numerically, but this condition is
   applied identically regardless of color, so the matcher is correct for X without a
   color-specific carve-out.
5. ``dimension_agreement`` -- the two connected legs' :class:`~qufzx.algebra.dimension.Dim`
   are equal, or :meth:`~qufzx.algebra.dimension.Dim.unify` defers or succeeds only by
   binding a symbol (both recorded as a :class:`FusionMatch` dimension constraint -- see
   below -- not silently accepted). A ``FAILURE`` from ``unify`` is a non-match. The
   match's ``shared_dim`` is *not* simply the A-side leg's raw ``Dim``: when ``unify``
   succeeds by binding a symbol (e.g. leg dims ``d`` and ``3`` bind ``d := 3``), the
   binding is substituted into the A-side leg's ``Dim`` (see
   :meth:`~qufzx.algebra.dimension.Dim.substitute`) to produce ``shared_dim``, so it is
   ``3`` in that example, not the still-unbound ``d``. When ``unify`` only defers, or
   succeeds with no binding at all (a bare syntactic identity), ``shared_dim`` is the
   A-side leg's raw ``Dim``, unchanged.
6. ``phase_dimension_agreement`` -- every phase vector actually present (on either node,
   or both), after substituting condition 5's binding (if any) into its own
   :class:`~qufzx.algebra.dimension.Dim`, must equal the resolved ``shared_dim`` from
   condition 5, checked with plain ``Dim`` equality, never a fresh call to
   :meth:`~qufzx.algebra.dimension.Dim.unify`. If a phase is present on *both* nodes,
   their two raw (unsubstituted) ``Dim``\\ s must *additionally* be exactly equal to each
   other -- not merely each equal to ``shared_dim`` after substitution. :func:`spider_fusion_builder`
   in :mod:`qufzx.rewrite.rules_library` reattaches each present phase vector's entries to
   ``shared_dim`` unchanged, without substituting into the entries themselves, so this
   second check keeps two *actual* phase vectors from combining unless they already agreed
   before any resolution; a phase absent on one or both sides needs no such check, since
   the builder synthesizes a zero-entry vector directly at ``shared_dim`` and an all-zero
   vector has no entries that could reference a stale symbol. Agreement on the container
   ``Dim`` alone is not sufficient: condition 5 can resolve ``shared_dim`` to something more
   concrete than a present phase's own (unsubstituted) ``Dim`` by binding a symbol (e.g. a
   phase legally stated over symbolic ``d`` with an entry at index 5 has ``Dim`` equal to
   ``d``, which matches a ``shared_dim`` of ``2`` under a binding ``d := 2`` -- yet index 5
   is out of range once ``d`` is actually ``2``). This condition therefore also verifies,
   for every phase actually present, that reattaching its unchanged entries to
   ``shared_dim`` is itself legal -- literally by attempting the same construction
   :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`'s ``_over_shared_dim`` performs
   (``PhaseVector(shared_dim, phase.entries())``) and treating a
   :class:`~qufzx.algebra.phase.PhaseDomainError` as a failed condition (non-match) rather
   than letting it escape from the builder later. This makes match-approval and
   build-applicability the same predicate by construction, not two predicates kept in sync
   by hand: the invariant is that every match this function returns can be applied by
   :func:`~qufzx.rewrite.engine.apply` without raising anything except the step-8 relative-
   postcondition :class:`~qufzx.rewrite.rule.RewriteDomainError`.

Malformed wire references. :mod:`qufzx.diagram.graph` is deliberately permissive about
what a :class:`~qufzx.diagram.graph.Wire` may name (see that module's docstring on
validation ownership), so an un-validated diagram can hold a wire endpoint naming a node
id absent from the diagram, or one present but with an out-of-range port index for that
side. :func:`find_matches` checks both endpoints of every candidate wire for both faults
before checking any other candidate property -- generator color, fusable-color-ness, and
wire direction alike -- and raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` naming
the offending :class:`~qufzx.diagram.graph.PortRef` (and, for the index case, the node's
actual leg count) -- the same treatment :mod:`qufzx.diagram.validate` gives both as hard
errors (``UNKNOWN_NODE``, ``PORT_INDEX_OUT_OF_RANGE``), rather than letting either escape
this module's declared error hierarchy as a bare ``KeyError``/``IndexError`` or, for the
node case, passing silently as a non-match. This ordering is deliberate and load-bearing:
detection of a malformed wire must not depend on unrelated properties of the candidate
pair it happens to sit on, so a wire with an out-of-range port index is rejected
identically whether it joins two Z spiders, a Z and an X, or a pair with matching wire
directions -- not only for the shapes that happen to survive far enough through the other
side conditions to reach the check.

Dimension constraints. A :class:`FusionMatch`'s ``dimension_constraints`` records every
leg-dimension pair that :func:`find_matches` did not verify as a syntactic identity but
still accepted: both a ``DEFERRED`` outcome from :meth:`Dim.unify` (truly undecided, per
that method's contract) and a ``SUCCESS`` outcome that only holds because ``unify`` bound
a free symbol (decided, but only under that binding -- e.g. leg dims ``d`` and ``3``
unify by binding ``d := 3``, and the fusion is valid only at that value). Both provenances
are assumed equalities a diagram-level unifier must eventually justify, so both belong in
the certificate Phase 6 will read from this field; only a unify outcome that is a bare
syntactic identity (no binding, nothing deferred) is left out, since nothing was assumed.
The corresponding ``SideConditionOutcome.deferred`` flag is ``True`` only for the
``DEFERRED`` case -- it continues to mean "``Dim.unify`` could not decide this at all",
not "some assumption was recorded" -- so a binding-based ``SUCCESS`` is reported with
``deferred=False`` even though it, too, appends to ``dimension_constraints``.

A candidate that fails condition 1, 2, or 3 is dropped before any :class:`FusionMatch` is
constructed at all -- there is no "failed match" object for those, since they gate whether
a pair is a fusion candidate in the first place, not a property of one. Conditions 4-6 are
checked per surviving candidate and, when they fail, the candidate is likewise dropped
(never included in the returned tuple) rather than reported as a match with a False
outcome; every :class:`FusionMatch` this module returns therefore has
``all_side_conditions_passed`` True by construction. This mirrors
:mod:`qufzx.diagram.validate`'s existing deferred/hard-failure split for dimension issues.

Determinism. :func:`find_matches` sorts its result by node ids (the two matched nodes
uniquely determine at most one fusion candidate under condition 3, so no further tiebreak
is mathematically necessary here) -- never by set or dict iteration order, since
certificates and Phase 12's cache tests will compare match lists directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from qufzx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from qufzx.algebra.phase import PhaseDomainError, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, NodeId, Wire
from qufzx.rewrite.rule import (
    Match,
    Pattern,
    RewriteGrammarError,
    SideCondition,
    SideConditionOutcome,
)

FUSION_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("distinct_nodes", "the two matched nodes are not the same node"),
    SideCondition("same_generator_type", "both nodes are the same registered spider color"),
    SideCondition("single_connecting_wire", "exactly one wire joins the two nodes"),
    SideCondition(
        "wire_direction_output_to_input",
        "the consumed wire runs from an OUTPUT port of one node to an INPUT port of the other",
    ),
    SideCondition(
        "dimension_agreement",
        "the two connected legs' dimensions are equal, or unify defers or binds a symbol",
    ),
    SideCondition(
        "phase_dimension_agreement",
        "every phase vector present has a dimension exactly equal to the shared leg dimension",
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
    """

    a_id: NodeId
    b_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[tuple[Dim, Dim], ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed. See :class:`qufzx.rewrite.rule.Match`."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)


_FUSABLE_GENERATOR_NAMES = frozenset((Z_SPIDER.name, X_SPIDER.name))


def _resolve_with_bindings(dim: Dim, bindings: Mapping[str, Dim]) -> Dim:
    """Substitute ``bindings`` into ``dim``, or return it unchanged if none apply.

    ``bindings`` is empty both when :meth:`Dim.unify` deferred and when it succeeded via a
    bare syntactic identity with nothing bound -- in both cases this is the identity
    function, which is exactly the "keep the raw Dim unchanged" behavior condition 5 in
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


def find_matches(diagram: Diagram) -> tuple[FusionMatch, ...]:
    """Find every same-color spider fusion occurrence in ``diagram``. See the module docstring.

    Never mutates ``diagram``. Returns matches sorted by ``(a_id, b_id)``.
    """
    candidates_by_pair: dict[frozenset[NodeId], list[Wire]] = {}
    for wire in diagram.wires:
        if wire.a.node_id == wire.b.node_id:
            continue
        key = frozenset((wire.a.node_id, wire.b.node_id))
        candidates_by_pair.setdefault(key, []).append(wire)

    matches: list[FusionMatch] = []
    for connecting_wires in candidates_by_pair.values():
        if len(connecting_wires) != 1:
            continue
        wire = connecting_wires[0]
        a_id, b_id = _ordered_pair(wire)

        node_a = diagram.nodes.get(a_id)
        node_b = diagram.nodes.get(b_id)
        if node_a is None or node_b is None:
            missing = [nid for nid, node in ((a_id, node_a), (b_id, node_b)) if node is None]
            raise RewriteGrammarError(
                f"wire {wire!r} references node id(s) {missing!r} absent from the diagram; "
                "Diagram.remove_node cascades, so a live wire can never legitimately name a "
                "removed node"
            )

        ref_a = wire.a if wire.a.node_id == a_id else wire.b
        ref_b = wire.b if wire.a.node_id == a_id else wire.a

        # Malformed-wire detection (an out-of-range port index) must be independent of
        # every other candidate property -- generator color, fusable-color-ness, and
        # wire direction alike -- so it is checked here, before any of those can drop
        # the candidate via `continue`. Checking it only after those side conditions
        # (as an earlier version did) made a malformed wire raise when it happened to
        # join two same-color, opposite-direction spiders, but silently pass through
        # as a non-match for a Z/X pair, a same-direction pair, or any other candidate
        # shape that gets dropped first -- masking the same structural defect
        # differently depending on unrelated properties of the pair.
        legs_a = node_a.legs(ref_a.direction)
        if ref_a.index >= len(legs_a):
            raise RewriteGrammarError(
                f"wire endpoint {ref_a!r} is out of range for node {a_id!r}: it has only "
                f"{len(legs_a)} {ref_a.direction.value} leg(s)"
            )
        legs_b = node_b.legs(ref_b.direction)
        if ref_b.index >= len(legs_b):
            raise RewriteGrammarError(
                f"wire endpoint {ref_b!r} is out of range for node {b_id!r}: it has only "
                f"{len(legs_b)} {ref_b.direction.value} leg(s)"
            )

        if node_a.generator_type != node_b.generator_type:
            continue
        if node_a.generator_type.name not in _FUSABLE_GENERATOR_NAMES:
            continue
        if ref_a.direction == ref_b.direction:
            continue

        port_a = legs_a[ref_a.index]
        port_b = legs_b[ref_b.index]
        leg_unify = port_a.dim.unify(port_b.dim)
        if leg_unify.is_failure:
            continue

        dimension_constraints: list[tuple[Dim, Dim]] = []
        leg_deferred = leg_unify.is_deferred
        leg_bound = leg_unify.is_success and bool(leg_unify.bindings)
        if leg_deferred or leg_bound:
            dimension_constraints.append((port_a.dim, port_b.dim))
        if leg_deferred:
            leg_detail = f"{port_a.dim} == {port_b.dim} (deferred, assumed)"
        elif leg_bound:
            binding_desc = ", ".join(
                f"{name} := {value}" for name, value in sorted(leg_unify.bindings.items())
            )
            leg_detail = f"{port_a.dim} == {port_b.dim} (bound: {binding_desc})"
        else:
            leg_detail = f"{port_a.dim} == {port_b.dim}"

        shared_dim = _resolve_with_bindings(port_a.dim, leg_unify.bindings)

        phase_a_dim = node_a.phase.dim if node_a.phase is not None else None
        phase_b_dim = node_b.phase.dim if node_b.phase is not None else None
        phase_dims_present = tuple(d for d in (phase_a_dim, phase_b_dim) if d is not None)
        phase_dims_agree = all(
            _resolve_with_bindings(phase_dim, leg_unify.bindings) == shared_dim
            for phase_dim in phase_dims_present
        )
        if phase_dims_agree and phase_a_dim is not None and phase_b_dim is not None:
            phase_dims_agree = phase_a_dim == phase_b_dim
        if phase_dims_agree:
            for phase in (node_a.phase, node_b.phase):
                if phase is None:
                    continue
                try:
                    PhaseVector(shared_dim, phase.entries())
                except PhaseDomainError:
                    phase_dims_agree = False
                    break
        if not phase_dims_agree:
            continue

        outcomes = (
            SideConditionOutcome("distinct_nodes", True, f"{a_id!r} != {b_id!r}"),
            SideConditionOutcome(
                "same_generator_type",
                True,
                f"both nodes are {node_a.generator_type.name!r}",
            ),
            SideConditionOutcome(
                "single_connecting_wire", True, "exactly one wire joins the two nodes"
            ),
            SideConditionOutcome(
                "wire_direction_output_to_input",
                True,
                f"{ref_a} (direction={ref_a.direction.value}) -> "
                f"{ref_b} (direction={ref_b.direction.value})",
            ),
            SideConditionOutcome(
                "dimension_agreement",
                True,
                leg_detail,
                deferred=leg_deferred,
            ),
            SideConditionOutcome(
                "phase_dimension_agreement",
                True,
                (
                    "no phase present on either node"
                    if not phase_dims_present
                    else (
                        "present phase dimension(s), after any leg-unify binding, equal "
                        f"the resolved shared leg dimension {shared_dim}"
                    )
                ),
            ),
        )

        matches.append(
            FusionMatch(
                a_id=a_id,
                b_id=b_id,
                wire=wire,
                shared_dim=shared_dim,
                side_condition_outcomes=outcomes,
                dimension_constraints=tuple(dimension_constraints),
            )
        )

    matches.sort(key=lambda m: (int(m.a_id), int(m.b_id)))
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
