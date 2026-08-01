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
   :meth:`~qufzx.algebra.dimension.Dim.substitute`) to produce a provisional ``shared_dim``,
   so it is ``3`` in that example, not the still-unbound ``d``. When ``unify`` only defers,
   or succeeds with no binding at all (a bare syntactic identity), the provisional
   ``shared_dim`` is the A-side leg's raw ``Dim``, unchanged. This condition does not stop
   at the connecting pair: every *surviving* leg of both nodes (every leg other than the
   one consumed by the matched wire) is then unified against the provisional
   ``shared_dim``, in turn, in the same per-node order (inputs, then outputs, original
   index order) :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` uses to build
   the merged node's ports -- a later leg is checked against whatever ``shared_dim`` an
   earlier leg's binding refined it to, so a chain such as leg dims ``d``, ``d``, ``2``
   resolves to a final ``shared_dim`` of ``2`` and every step along the way is recorded. A
   ``FAILURE`` on any surviving leg makes the whole candidate a non-match, exactly like a
   ``FAILURE`` on the connecting pair itself: forcing a genuinely non-unifiable surviving
   leg onto ``shared_dim`` anyway would silently rewrite that leg's dimension (or erase a
   pre-existing hard dimension conflict on it) rather than report the conflict. A
   ``DEFERRED`` or binding-only ``SUCCESS`` on a surviving leg is recorded as a dimension
   constraint exactly like the connecting pair's own (see "Dimension constraints" below),
   and, if it binds, the newly-refined ``shared_dim`` carries forward to every leg checked
   after it. Only once every surviving leg has been checked is ``shared_dim`` final.
6. ``phase_dimension_agreement`` -- every phase vector actually present (on either node,
   or both), after substituting condition 5's bindings (if any) into its own
   :class:`~qufzx.algebra.dimension.Dim`, must equal the resolved ``shared_dim`` from
   condition 5, checked with plain ``Dim`` equality, never a fresh call to
   :meth:`~qufzx.algebra.dimension.Dim.unify`. "Condition 5's bindings" is every concrete
   symbol binding produced anywhere in condition 5 -- the connecting pair's own
   (``leg_unify.bindings``) *and* every surviving leg's, on either node, accumulated as
   condition 5 proceeds (see :func:`_unify_surviving_legs`) -- not only the connecting
   pair's; a phase's ``Dim`` can reference a symbol that only a *surviving* leg happens to
   bind (e.g. a phase stated over symbolic ``d`` on a node whose connecting leg is also
   ``d`` but whose surviving leg is a concrete ``2``, binding ``d := 2`` only once the
   surviving leg is unified against ``shared_dim``), and condition 5's own ``shared_dim``
   is already refined by exactly that binding by the time this condition runs -- resolving
   the phase's ``Dim`` against a narrower set would compare it, unsubstituted, against a
   ``shared_dim`` it was in fact refined to agree with. If a phase is present on *both* nodes,
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
side. :func:`find_matches` checks both endpoints of *every* wire in the diagram for both
faults -- via :func:`_validate_wire_endpoint` -- in a first pass that runs before any
candidate grouping or filtering: before the self-loop skip, before wires are grouped by
node pair, before the parallel-wire-pair (``len(connecting_wires) != 1``) filter, and
before any other candidate property (generator color, fusable-color-ness, wire direction)
is even reached. It raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` naming the
offending :class:`~qufzx.diagram.graph.PortRef` (and, for the index case, the node's
actual leg count) -- the same treatment :mod:`qufzx.diagram.validate` gives both as hard
errors (``UNKNOWN_NODE``, ``PORT_INDEX_OUT_OF_RANGE``), rather than letting either escape
this module's declared error hierarchy as a bare ``KeyError``/``IndexError`` or passing
silently as a non-match. This ordering is deliberate and load-bearing: detection of a
malformed wire must not depend on any other property of the wire or the candidate pair it
happens to sit on, so a wire with an out-of-range port index (or an unknown node id) is
rejected identically whether it joins two Z spiders, a Z and an X, a pair with matching
wire directions, a pair also joined by a second parallel wire, or even both of its own
endpoints on the very same node (a self-loop) -- every one of those shapes used to make
the malformed-wire check unreachable for that wire specifically, since each is dropped
(via `continue`, without ever raising) by a filter that used to run first.

Match-implies-applicable and multiply-claimed ports. A port that is claimed by more than
one wire (:class:`~qufzx.diagram.validate.IssueKind.PORT_WIRED_TWICE`), or that is both
wired and listed on a boundary
(:class:`~qufzx.diagram.validate.IssueKind.PORT_WIRED_AND_BOUNDARY`), is not treated as a
fusion occurrence even when it happens to be the port a candidate wire would consume.
:mod:`qufzx.rewrite.engine`'s ``apply`` requires every consumed port to appear in the
builder's ``port_mapping``, but a builder only ever maps *surviving* ports -- the consumed
port itself is deliberately absent, since it no longer exists on the merged node. If that
same port is also named by a second wire (to a third node) or a boundary entry, ``apply``
would need to remap that second reference too, and cannot: the port is gone, and nothing
in the match or the builder says what it should become. :func:`find_matches` rejects any
candidate whose consumed port (on either side) is claimed by more than one wire in the
whole diagram, or is on either boundary list, before constructing a
:class:`FusionMatch` at all -- the same structural, no-``FusionMatch``-object treatment
given self-loops and parallel-wire pairs above. This is what makes "every match this
function returns can be applied by :func:`~qufzx.rewrite.engine.apply` without raising
anything except the step-8 relative-postcondition" (see condition 6 below) true without
qualification: :mod:`qufzx.rewrite.engine`'s own docstring states the matching half of
this resolution on its side.

Dimension constraints. A :class:`FusionMatch`'s ``dimension_constraints`` records every
leg-dimension pair that :func:`find_matches` did not verify as a syntactic identity but
still accepted: both a ``DEFERRED`` outcome from :meth:`Dim.unify` (truly undecided, per
that method's contract) and a ``SUCCESS`` outcome that only holds because ``unify`` bound
a free symbol (decided, but only under that binding -- e.g. leg dims ``d`` and ``3``
unify by binding ``d := 3``, and the fusion is valid only at that value). This applies
uniformly to the connecting pair (recorded as ``(port_a.dim, port_b.dim)``) and to every
surviving leg condition 5 checks against the running ``shared_dim`` (recorded as
``(leg.dim, shared_dim)`` at the point that leg was checked): both are assumed equalities
a diagram-level unifier must eventually justify, so both belong in the certificate Phase 6
will read from this field; only a unify outcome that is a bare syntactic identity (no
binding, nothing deferred) is left out, since nothing was assumed. The corresponding
``SideConditionOutcome.deferred`` flag on ``dimension_agreement`` is ``True`` if *any* of
these checks -- the connecting pair's or any surviving leg's -- returned ``DEFERRED``; it
continues to mean "``Dim.unify`` could not decide this at all" for at least one of them,
not "some assumption was recorded" -- so a check that only ever bound symbols, with none
deferred, is reported with ``deferred=False`` even though it, too, appends to
``dimension_constraints``.

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
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef, Wire
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


def _unify_surviving_legs(
    node: Node,
    node_id: NodeId,
    consumed_ref: PortRef,
    shared_dim: Dim,
    bindings: dict[str, Dim],
) -> tuple[Dim, list[tuple[Dim, Dim]], bool] | None:
    """Unify every surviving leg of ``node`` (both directions) against ``shared_dim`` in turn.

    "Surviving" means every leg of ``node`` except ``consumed_ref``, checked in the same
    per-node order (inputs, then outputs, original index order) that
    :mod:`qufzx.rewrite.rules_library`'s ``_surviving_legs`` uses to build the merged
    node's ports, so the constraint list this returns lines up with that ordering.

    ``bindings`` is the running, whole-candidate accumulator of every concrete symbol
    binding seen so far (starting with the connecting pair's own, from ``leg_unify``) --
    this function updates it in place with each surviving leg's own concrete binding, on
    top of using it (via :func:`_resolve_with_bindings`) to refine ``shared_dim`` as it
    goes. Condition 6 (``phase_dimension_agreement``) resolves each present phase's ``Dim``
    against this same accumulator, not only against the connecting pair's bindings -- see
    the module docstring's account of why a phase's ``Dim`` can depend on a symbol a
    *surviving* leg bound, not just the one the connecting pair bound.

    Returns ``(resolved_shared_dim, new_dimension_constraints, any_deferred)`` on success.
    Returns ``None`` if any surviving leg's dim is non-unifiable with the (possibly
    already-refined-by-an-earlier-leg) ``shared_dim`` -- a ``FAILURE`` here makes the whole
    candidate a non-match, exactly like a ``FAILURE`` on the connecting-leg pair itself,
    since forcing that leg onto ``shared_dim`` anyway (as the pre-fix builder silently did)
    would destroy a real dimension conflict rather than report it. Each surviving leg whose
    unify only deferred, or only succeeded by binding a symbol, is recorded as a dimension
    constraint pair ``(leg.dim, shared_dim)`` -- the same provenance and shape the
    connecting-leg pair itself is recorded under (see the module docstring) -- and, if that
    binding makes ``shared_dim`` more concrete, every leg checked afterward (including on
    the other node) is checked against the refined value.
    """
    constraints: list[tuple[Dim, Dim]] = []
    any_deferred = False
    for direction in (Direction.INPUT, Direction.OUTPUT):
        for index, port in enumerate(node.legs(direction)):
            ref = PortRef(node_id, direction, index)
            if ref == consumed_ref:
                continue
            result = port.dim.unify(shared_dim)
            if result.is_failure:
                return None
            deferred_here = result.is_deferred
            bound_here = result.is_success and bool(result.bindings)
            if deferred_here or bound_here:
                constraints.append((port.dim, shared_dim))
            if deferred_here:
                any_deferred = True
            if bound_here:
                bindings.update(
                    {name: value for name, value in result.bindings.items() if value.is_concrete}
                )
                shared_dim = _resolve_with_bindings(shared_dim, result.bindings)
    return shared_dim, constraints, any_deferred


def _validate_wire_endpoint(diagram: Diagram, wire: Wire, ref: PortRef) -> None:
    """Raise ``RewriteGrammarError`` if ``ref`` names an unknown node or out-of-range index.

    Called for both endpoints of every wire in the diagram, unconditionally -- see
    "Malformed wire references" in the module docstring for why this must not depend on
    any other property of the wire or the candidate pair it might otherwise sit on.
    """
    node = diagram.nodes.get(ref.node_id)
    if node is None:
        raise RewriteGrammarError(
            f"wire {wire!r} references node id {ref.node_id!r} absent from the diagram; "
            "Diagram.remove_node cascades, so a live wire can never legitimately name a "
            "removed node"
        )
    legs = node.legs(ref.direction)
    if ref.index >= len(legs):
        raise RewriteGrammarError(
            f"wire endpoint {ref!r} is out of range for node {ref.node_id!r}: it has only "
            f"{len(legs)} {ref.direction.value} leg(s)"
        )


def find_matches(diagram: Diagram) -> tuple[FusionMatch, ...]:
    """Find every same-color spider fusion occurrence in ``diagram``. See the module docstring.

    Never mutates ``diagram``, and does not require ``diagram`` to be well-formed --
    :func:`~qufzx.diagram.validate.validate` is never called here. Returns matches sorted
    by ``(a_id, b_id)``.
    """
    # Malformed-wire detection (an unknown node id or an out-of-range port index) must be
    # independent of every other property of the wire or the candidate pair it happens to
    # sit on -- generator color, fusable-color-ness, wire direction, single-vs-parallel
    # wiring, and self-loop-ness alike -- so every wire's both endpoints are checked here,
    # before any grouping or filtering. Checking this only after grouping (as an earlier
    # version did) let a malformed wire escape undetected as a bare non-match whenever it
    # happened to be dropped first by the self-loop skip or the parallel-wire-pair filter
    # below, masking the same structural defect differently depending on unrelated shape.
    # Snapshotted once, not re-read from ``diagram.wires`` on every pass below: the three
    # passes over the wire set (malformed-endpoint check, wired-ref counting, pair grouping)
    # used to each re-materialise ``diagram.wires`` independently, tripling the cost of
    # building whatever collection backs that property for no reason -- none of the three
    # passes needs a live view, and none of them mutates ``diagram``.
    wires = diagram.wires

    for wire in wires:
        _validate_wire_endpoint(diagram, wire, wire.a)
        _validate_wire_endpoint(diagram, wire, wire.b)

    # Defect 2 (match-implies-applicable): a port that is claimed by more than one wire, or
    # that is both wired and listed on a boundary, is not a legitimate fusion occurrence at
    # all -- fusing across it would ask the builder to remap a consumed port that a *third*
    # reference (another wire, or a boundary entry) also still names, which
    # qufzx.rewrite.engine.apply's port_mapping coverage check (step 5) correctly refuses to
    # do silently. Rejecting the candidate here, structurally, keeps every match this
    # function returns applicable by construction, the same invariant condition 5 already
    # gives dimension_agreement -- see the module docstring's account of this resolution and
    # qufzx.rewrite.engine's docstring for the matching statement on its side.
    wired_ref_counts: dict[PortRef, int] = {}
    for wire in wires:
        wired_ref_counts[wire.a] = wired_ref_counts.get(wire.a, 0) + 1
        wired_ref_counts[wire.b] = wired_ref_counts.get(wire.b, 0) + 1
    boundary_ref_set = frozenset(diagram.boundary_inputs) | frozenset(diagram.boundary_outputs)

    candidates_by_pair: dict[frozenset[NodeId], list[Wire]] = {}
    for wire in wires:
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

        node_a = diagram.nodes[a_id]
        node_b = diagram.nodes[b_id]

        ref_a = wire.a if wire.a.node_id == a_id else wire.b
        ref_b = wire.b if wire.a.node_id == a_id else wire.a

        if wired_ref_counts[ref_a] > 1 or wired_ref_counts[ref_b] > 1:
            continue
        if ref_a in boundary_ref_set or ref_b in boundary_ref_set:
            continue

        legs_a = node_a.legs(ref_a.direction)
        legs_b = node_b.legs(ref_b.direction)

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

        # Defect 2 (Phase 5 round-7 audit): the whole-candidate accumulator of every
        # concrete symbol binding seen so far, starting with the connecting pair's own.
        # ``_unify_surviving_legs`` below updates this in place with each surviving leg's
        # own binding, so a present phase's Dim (resolved against this same accumulator
        # further down) sees a binding a *surviving* leg produced, not only one the
        # connecting pair produced -- see the module docstring, condition 6.
        bindings: dict[str, Dim] = {
            name: value for name, value in leg_unify.bindings.items() if value.is_concrete
        }

        survive_a = _unify_surviving_legs(node_a, a_id, ref_a, shared_dim, bindings)
        if survive_a is None:
            continue
        shared_dim, extra_constraints_a, deferred_a = survive_a

        survive_b = _unify_surviving_legs(node_b, b_id, ref_b, shared_dim, bindings)
        if survive_b is None:
            continue
        shared_dim, extra_constraints_b, deferred_b = survive_b

        dimension_constraints.extend(extra_constraints_a)
        dimension_constraints.extend(extra_constraints_b)
        leg_deferred = leg_deferred or deferred_a or deferred_b

        phase_a_dim = node_a.phase.dim if node_a.phase is not None else None
        phase_b_dim = node_b.phase.dim if node_b.phase is not None else None
        phase_dims_present = tuple(d for d in (phase_a_dim, phase_b_dim) if d is not None)
        phase_dims_agree = all(
            _resolve_with_bindings(phase_dim, bindings) == shared_dim
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
                leg_detail
                + (
                    ""
                    if not (extra_constraints_a or extra_constraints_b)
                    else (
                        f"; surviving leg(s) resolved to shared_dim={shared_dim} with "
                        f"{len(extra_constraints_a) + len(extra_constraints_b)} additional "
                        "assumed dimension equality/ies"
                    )
                ),
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
