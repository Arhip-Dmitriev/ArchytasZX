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
   below -- not silently accepted). A ``FAILURE`` from ``unify`` is a non-match.
6. ``phase_dimension_agreement`` -- every phase vector actually present (on either node,
   or both) must carry a :class:`~qufzx.algebra.dimension.Dim` *exactly* equal to the
   match's ``shared_dim`` (the connected legs' dimension), checked with plain ``Dim``
   equality, never :meth:`~qufzx.algebra.dimension.Dim.unify`. This mirrors
   :meth:`~qufzx.algebra.phase.PhaseVector.__add__`'s own requirement (exact ``Dim``
   equality, not a unify success-with-binding) -- :func:`spider_fusion_builder` in
   :mod:`qufzx.rewrite.rules_library` calls that ``__add__`` directly once a match is
   built, and has no unifier of its own to bind a symbol first. A unify ``DEFERRED`` or a
   unify ``SUCCESS`` that only holds via a binding is therefore *not* good enough here,
   unlike condition 5: there is no way to honor a deferred phase-dimension equality until
   a real unifier (Phase 10) exists to bind the symbols before the vectors are added, so
   any such candidate is dropped as a non-match rather than reported with a constraint. A
   `None` phase on either or both sides trivially satisfies this condition --
   :mod:`qufzx.rewrite.rules_library` treats a missing phase as the zero vector over the
   shared dimension when it builds the merged node.

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

from dataclasses import dataclass

from qufzx.algebra.dimension import Dim
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, NodeId, Wire
from qufzx.rewrite.rule import Match, Pattern, SideCondition, SideConditionOutcome

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
            continue

        if node_a.generator_type != node_b.generator_type:
            continue
        if node_a.generator_type.name not in _FUSABLE_GENERATOR_NAMES:
            continue

        ref_a = wire.a if wire.a.node_id == a_id else wire.b
        ref_b = wire.b if wire.a.node_id == a_id else wire.a
        if ref_a.direction == ref_b.direction:
            continue

        port_a = node_a.legs(ref_a.direction)[ref_a.index]
        port_b = node_b.legs(ref_b.direction)[ref_b.index]
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

        phase_dims_present = tuple(
            p.dim for p in (node_a.phase, node_b.phase) if p is not None
        )
        if any(phase_dim != port_a.dim for phase_dim in phase_dims_present):
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
                    else f"present phase dimension(s) equal the shared leg dimension {port_a.dim}"
                ),
            ),
        )

        matches.append(
            FusionMatch(
                a_id=a_id,
                b_id=b_id,
                wire=wire,
                shared_dim=port_a.dim,
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
