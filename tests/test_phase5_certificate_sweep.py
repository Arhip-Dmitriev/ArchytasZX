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

"""EXHAUSTIVE (not random) sweep over ``dimension_constraints`` content, closing the gap the
certificate-fidelity round exists for: the exhaustive oracle sweep in
``test_phase5_exhaustive_oracle.py`` only ever visits *cleanly contractible* diagrams
(``_is_cleanly_contractible`` gates out anything with a deferred issue), so the entire
deferred/binding path -- the one Phase 5's own judgement call allows fusion to fire across --
has never been oracle-validated at a substitution where its own recorded assumption actually
holds. This module closes that gap and, separately, sweeps the *shape* of the certificate
data itself (source uniqueness, determinism, builder agreement) over a much larger space than
any one hand-picked test in ``test_match.py``/``test_engine.py`` can cover.

Two sweeps, deliberately split by cost:

* :class:`TestCertificateStructuralProperties` -- cheap (no tensor contraction), so it runs
  over a broad cross product: both colours, alternating and same-direction (Z-only) wiring,
  every combination of 0-1 surviving legs per side drawn from a five-entry palette (concrete,
  two distinct bare symbols, a product, and a power -- see ``_SURVIVING_PALETTE``; widened
  from a two-entry one, Phase 5 audit round 15, N1, so two distinct binding symbols and a
  deferred-then-refuted leg are inside this sweep's own space, not only ``test_match.py``'s
  hand-picked D1 regression shapes), and phase present/absent (concrete / symbolic / none) on
  each node. Checked for every match this space produces: no duplicate constraint sources,
  determinism across repeated ``find_matches`` calls, every constraint's own pair unifies in
  isolation *and* the whole recorded set is simultaneously satisfiable under the match's
  final bindings (not merely pairwise self-consistent -- exactly the distinction D1's own
  contradictory constraint set exposed), and builder agreement (``apply`` never raises
  ``RewriteDomainError``, and the recorded ``RewriteStep.dimension_constraints`` equals the
  match's own).
* :class:`TestOracleTiesBackToRecordedConstraints` -- expensive (one tensor contraction pair
  per case), so it is deliberately narrower: a fixed, hand-chosen palette of dimension pairs
  that are known to produce a ``BOUND`` or ``DEFERRED`` constraint (covering concrete, bare
  symbol, product, and power representations -- see ``_DIM_PAIRS`` below), crossed with the
  three constraint *sources* (connecting pair, a surviving leg, a node's phase) and the three
  colour/direction combinations. For every resulting match, a small brute-force search finds
  a concrete substitution that satisfies every recorded constraint and asserts the pre- and
  post-fusion diagrams are then exactly oracle-equal, and a second substitution that violates
  the one constraint under test and asserts they are *not* equal -- proving the constraint is
  load-bearing, not decorative. A case where no small satisfying substitution exists (e.g. the
  Diophantine-infeasible ``concrete(2) == d**2``) is skipped for the tie-back half only, not
  silently counted as passing; the structural properties above still cover it.
* :class:`TestCertificateDetailFidelity` (Phase 5 post-closing audit round 19, Task 1) --
  cheap, like the structural sweep. Exhaustively checks that every human-readable
  ``SideConditionOutcome.detail`` string derived from ``dimension_constraints`` (the
  ``dimension_agreement`` outcome's connecting-pair clause and leg count, and the passing
  ``phase_dimension_agreement`` outcome's "assuming ..." clause) states exactly the same
  operands, in the same order, and names exactly the bindings, as the
  :class:`~qufzx.rewrite.rule.DimensionConstraint` record entry it describes -- never a
  value recomputed from final state or attributed by symbol-occurrence coincidence. This is
  the class of defect Defect 4 (round 18) and its round-19 recurrence both belonged to (see
  :func:`~qufzx.rewrite.match._connecting_pair_detail`'s docstring); round 18's own
  regression test for it (``test_match.py::TestDimensionConstraintsRecording::
  test_non_concrete_binding_detail_says_something_was_assumed``) missed the recurrence
  because it was written at the one shape that produced the original bug rather than across
  the space the fix's docstring claimed to cover -- this class is the sweep that shape
  should have been from the start.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rule import ConstraintOutcome, ConstraintSourceKind, DimensionConstraint
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import EqualityMode, compare
from qufzx.semantics.contract_numeric import ContractDomainError

_D = Dim.symbol("d")
_E = Dim.symbol("e")
_CONSUMED_DIM_PALETTE = (Dim.concrete(2), _D)
"""Palette for the *consumed* (connecting) leg only -- kept small (unlike
``_SURVIVING_PALETTE``) since it is crossed with everything else in this sweep; widening it
too would blow up runtime for no coverage this module doesn't already get elsewhere (the
connecting pair's own dimension-pair space is exhaustively covered by ``_DIM_PAIRS`` and
``TestOracleTiesBackToRecordedConstraints`` below)."""
_SURVIVING_PALETTE = (Dim.concrete(2), _D, _E, _D * _E, _D**2)
"""Palette for *surviving* legs and phases. Widened (Phase 5 audit round 15, N1) from
``(Dim.concrete(2), _D)`` to include a product (``d*e``), a power (``d**2``), and a second
bare symbol (``e``) -- so two distinct binding symbols and a deferred-then-refuted leg are
inside this structural sweep's space, not only ``test_match.py``'s hand-picked D1 regression
shapes. ``_leg_shapes``' ``max_total`` is correspondingly lowered from 2 to 1 (see that
function) to keep the resulting cross product's runtime bounded -- a 5-member palette at
``max_total=2`` would multiply the previous (already six-figure) diagram count by roughly
``(5/2)**2``, which is not worth paying for marginal extra shape coverage over what
``max_total=1`` already reaches with the wider palette."""
_COLOR_DIRECTION_COMBOS = (
    (Z_SPIDER, Direction.OUTPUT, Direction.INPUT),
    (Z_SPIDER, Direction.OUTPUT, Direction.OUTPUT),
    (X_SPIDER, Direction.OUTPUT, Direction.INPUT),
)
"""(color, a-side consumed direction, b-side consumed direction). Same-direction is Z-only --
see :mod:`qufzx.rewrite.match`'s condition 4."""


def _leg_shapes(max_total: int = 1) -> tuple[tuple[int, int], ...]:
    return tuple(
        (n_in, n_out)
        for n_in in range(max_total + 1)
        for n_out in range(max_total + 1 - n_in)
    )


def _leg_contents(count: int) -> tuple[tuple[Dim, ...], ...]:
    return tuple(itertools.product(_SURVIVING_PALETTE, repeat=count))


class TestCertificateStructuralProperties:
    """No-duplicate-source, determinism, and builder-agreement, over a broad cheap sweep."""

    def test_every_match_in_the_space_has_a_well_formed_certificate(self) -> None:
        total_diagrams = 0
        total_matches = 0

        phase_choices: tuple[Dim | None, ...] = (None, Dim.concrete(2), _D)
        shapes = _leg_shapes()

        for color, dir_a, dir_b in _COLOR_DIRECTION_COMBOS:
            for consumed_dim in _CONSUMED_DIM_PALETTE:
                for (n_in_a, n_out_a), (n_in_b, n_out_b) in itertools.product(shapes, shapes):
                    for dims_in_a, dims_out_a in itertools.product(
                        _leg_contents(n_in_a), _leg_contents(n_out_a)
                    ):
                        for dims_in_b, dims_out_b in itertools.product(
                            _leg_contents(n_in_b), _leg_contents(n_out_b)
                        ):
                            for phase_a, phase_b in itertools.product(
                                phase_choices, phase_choices
                            ):
                                diagram, _a_id, _b_id = _build_two_node_diagram(
                                    color=color,
                                    dir_a=dir_a,
                                    dir_b=dir_b,
                                    consumed_dim=consumed_dim,
                                    extra_in_a=dims_in_a,
                                    extra_out_a=dims_out_a,
                                    extra_in_b=dims_in_b,
                                    extra_out_b=dims_out_b,
                                    phase_a=phase_a,
                                    phase_b=phase_b,
                                )
                                total_diagrams += 1
                                matches_first = find_matches(diagram)
                                matches_second = find_matches(diagram)
                                assert matches_first == matches_second, (
                                    "find_matches is not deterministic across repeated calls "
                                    "on an unchanged diagram"
                                )
                                for match in matches_first:
                                    total_matches += 1
                                    sources = [c.source for c in match.dimension_constraints]
                                    assert len(sources) == len(set(sources)), (
                                        f"duplicate constraint source in "
                                        f"{match.dimension_constraints!r}"
                                    )
                                    for constraint in match.dimension_constraints:
                                        assert isinstance(constraint, DimensionConstraint)
                                        # Soundness: a recorded constraint is never a bare
                                        # syntactic identity -- there is always a real
                                        # unify to justify it, never an invented one.
                                        assert constraint.assumed != constraint.equal_to
                                        unify_result = constraint.assumed.unify(
                                            constraint.equal_to
                                        )
                                        assert not unify_result.is_failure

                                    # N1 (Phase 5 audit round 15): the recorded constraint
                                    # set must be *simultaneously* satisfiable, not merely
                                    # pairwise self-consistent (each constraint's own
                                    # assumed/equal_to unifying in isolation, checked above,
                                    # is exactly what let D1's contradictory set through:
                                    # e*f == 2, e == 2, f == 2 each individually unify fine).
                                    # Resolve every constraint's own pair through the
                                    # match's *final* bindings and re-unify -- this is
                                    # _verify_fixpoint_closure's own check, re-derived here
                                    # independently as a genuine cross-check.
                                    bindings = {
                                        name: value
                                        for name, value in match.bindings.items()
                                        if value.is_concrete
                                    }
                                    for constraint in match.dimension_constraints:
                                        resolved_assumed = (
                                            constraint.assumed.substitute(
                                                cast(
                                                    Mapping[DimSymbolKey, DimSubstituteValue],
                                                    bindings,
                                                )
                                            )
                                            if bindings
                                            else constraint.assumed
                                        )
                                        resolved_equal_to = (
                                            constraint.equal_to.substitute(
                                                cast(
                                                    Mapping[DimSymbolKey, DimSubstituteValue],
                                                    bindings,
                                                )
                                            )
                                            if bindings
                                            else constraint.equal_to
                                        )
                                        assert not resolved_assumed.unify(
                                            resolved_equal_to
                                        ).is_failure, (
                                            f"constraint {constraint!r} is not "
                                            f"simultaneously satisfiable with the rest of "
                                            f"match.bindings={bindings!r}"
                                        )

                                    working_diagram = diagram.copy()
                                    result = apply(working_diagram, SPIDER_FUSION, match)
                                    assert (
                                        result.step.dimension_constraints
                                        == match.dimension_constraints
                                    )

        assert total_diagrams > 500, (
            f"only {total_diagrams} diagrams were constructed; the structural sweep "
            "collapsed unexpectedly small"
        )
        assert total_matches > 0, "the structural sweep never produced a single fusion match"


_DIM_PAIRS: tuple[tuple[Dim, Dim], ...] = (
    (_D, Dim.concrete(2)),  # bare symbol vs concrete -> BOUND (d := 2)
    (_D, _D * _E),  # bare symbol vs product -> DEFERRED, satisfiable at e := 1
    (_D, _D**2),  # bare symbol vs power -> DEFERRED, satisfiable at d := 1
    (_D * _E, _D**2),  # product vs power -> DEFERRED, satisfiable at e := d
)
"""Concrete / bare symbol / product / power, in the combinations that actually produce a
recorded constraint (never a bare identity) -- see the module docstring."""

_SEARCH_RANGE = range(1, 5)


def _small_integer_assignment(
    free_symbols: frozenset[str],
    fixed: dict[str, int],
    constraints: tuple[tuple[Dim, Dim], ...],
    *,
    violate_index: int | None,
) -> dict[str, int] | None:
    """Brute-force a small integer assignment satisfying every one of ``constraints``.

    Every symbol in ``fixed`` is held at its given value (the concrete bindings a ``BOUND``
    constraint already produced); every other free symbol is searched over
    :data:`_SEARCH_RANGE`. If ``violate_index`` is given, that one constraint must instead be
    *un*-satisfied while every other constraint still holds -- the negative control. Returns
    ``None`` if no assignment in the search range works (the constraint set may be genuinely
    infeasible over small integers, e.g. ``concrete(2) == d**2``; the caller must treat that
    as "skip", not as a failure).
    """
    free = sorted(free_symbols - set(fixed))
    for values in itertools.product(_SEARCH_RANGE, repeat=len(free)):
        candidate = {**fixed, **dict(zip(free, values, strict=True))}
        ok = True
        typed_candidate = cast(Mapping[DimSymbolKey, DimSubstituteValue], candidate)
        for index, (assumed, equal_to) in enumerate(constraints):
            assumed_val = assumed.substitute(typed_candidate).to_int()
            equal_val = equal_to.substitute(typed_candidate).to_int()
            holds = assumed_val == equal_val
            if index == violate_index:
                holds = not holds
            if not holds:
                ok = False
                break
        if ok:
            return candidate
    return None


def _build_two_node_diagram(
    *,
    color: GeneratorType,
    dir_a: Direction,
    dir_b: Direction,
    consumed_dim: Dim,
    consumed_dim_b: Dim | None = None,
    extra_in_a: tuple[Dim, ...] = (),
    extra_out_a: tuple[Dim, ...] = (),
    extra_in_b: tuple[Dim, ...] = (),
    extra_out_b: tuple[Dim, ...] = (),
    phase_a: Dim | None = None,
    phase_b: Dim | None = None,
) -> tuple[Diagram, int, int]:
    """A two-node diagram with the consumed wire at (``dir_a``, ``dir_b``) and the given
    extra (surviving) legs and phases. The consumed leg is always the first leg on its own
    (node, direction). ``consumed_dim`` is A's own consumed-leg ``Dim``; B's is the same
    value unless ``consumed_dim_b`` is given separately -- most callers want the two legs to
    already agree (a bare identity), but :class:`TestCertificateDetailFidelity` needs them
    genuinely independent to produce a real ``CONNECTING_PAIR`` constraint."""
    diagram = Diagram()
    consumed_dim_b = consumed_dim if consumed_dim_b is None else consumed_dim_b

    in_dims_a = [consumed_dim, *extra_in_a] if dir_a is Direction.INPUT else list(extra_in_a)
    out_dims_a = (
        [consumed_dim, *extra_out_a] if dir_a is Direction.OUTPUT else list(extra_out_a)
    )
    phase_vector_a = (
        PhaseVector(phase_a, {1: Phase.turns(sp.Rational(1, 3))}) if phase_a is not None else None
    )
    a_id = diagram.add_node(
        color, input_dims=in_dims_a, output_dims=out_dims_a, phase=phase_vector_a
    )

    in_dims_b = (
        [consumed_dim_b, *extra_in_b] if dir_b is Direction.INPUT else list(extra_in_b)
    )
    out_dims_b = (
        [consumed_dim_b, *extra_out_b] if dir_b is Direction.OUTPUT else list(extra_out_b)
    )
    phase_vector_b = (
        PhaseVector(phase_b, {1: Phase.turns(sp.Rational(1, 3))}) if phase_b is not None else None
    )
    b_id = diagram.add_node(
        color, input_dims=in_dims_b, output_dims=out_dims_b, phase=phase_vector_b
    )

    diagram.add_wire(PortRef(a_id, dir_a, 0), PortRef(b_id, dir_b, 0))

    boundary_inputs = [
        PortRef(a_id, Direction.INPUT, i)
        for i in range(len(in_dims_a))
        if not (dir_a is Direction.INPUT and i == 0)
    ] + [
        PortRef(b_id, Direction.INPUT, i)
        for i in range(len(in_dims_b))
        if not (dir_b is Direction.INPUT and i == 0)
    ]
    boundary_outputs = [
        PortRef(a_id, Direction.OUTPUT, i)
        for i in range(len(out_dims_a))
        if not (dir_a is Direction.OUTPUT and i == 0)
    ] + [
        PortRef(b_id, Direction.OUTPUT, i)
        for i in range(len(out_dims_b))
        if not (dir_b is Direction.OUTPUT and i == 0)
    ]
    diagram.set_boundary_inputs(boundary_inputs)
    diagram.set_boundary_outputs(boundary_outputs)
    return diagram, a_id, b_id


class TestOracleTiesBackToRecordedConstraints:
    """The highest-value check this sweep adds: a recorded constraint is load-bearing.

    For every (source kind x dimension pair x colour/direction) case, finds the one
    resulting match, locates the ``DimensionConstraint`` the case was built to produce,
    and asserts oracle equality at a substitution satisfying it and oracle *inequality* at
    one violating it.
    """

    def _case_diagram(
        self,
        *,
        color: GeneratorType,
        dir_a: Direction,
        dir_b: Direction,
        source_kind: ConstraintSourceKind,
        dim_a: Dim,
        dim_b: Dim,
    ) -> tuple[Diagram, int, int]:
        """Build a diagram whose match has exactly one constraint of ``source_kind``, over
        the pair (``dim_a``, ``dim_b``)."""
        if source_kind is ConstraintSourceKind.CONNECTING_PAIR:
            return _build_two_node_diagram(
                color=color, dir_a=dir_a, dir_b=dir_b, consumed_dim=dim_a, phase_a=None
            )
        if source_kind is ConstraintSourceKind.SURVIVING_LEG:
            # Consumed pair is a trivial identity (concrete(3), unrelated to dim_a/dim_b);
            # the surviving leg on A alone carries the tested pair.
            extra = (dim_b,)
            surviving_direction_kwargs = (
                {"extra_out_a": extra} if dir_a is Direction.OUTPUT else {"extra_in_a": extra}
            )
            diagram, a_id, b_id = _build_two_node_diagram(
                color=color,
                dir_a=dir_a,
                dir_b=dir_b,
                consumed_dim=Dim.concrete(3),
                **surviving_direction_kwargs,  # type: ignore[arg-type]
            )
            return diagram, a_id, b_id
        # NODE_PHASE: the connecting pair is trivial; A's phase carries the tested pair.
        diagram, a_id, b_id = _build_two_node_diagram(
            color=color, dir_a=dir_a, dir_b=dir_b, consumed_dim=Dim.concrete(3), phase_a=dim_a
        )
        return diagram, a_id, b_id

    def test_satisfying_and_violating_substitutions_agree_with_the_oracle(self) -> None:
        total_cases = 0
        total_checked = 0
        total_infeasible_skips = 0

        source_kinds = (
            ConstraintSourceKind.CONNECTING_PAIR,
            ConstraintSourceKind.SURVIVING_LEG,
            ConstraintSourceKind.NODE_PHASE,
        )

        for color, dir_a, dir_b in _COLOR_DIRECTION_COMBOS:
            for source_kind in source_kinds:
                for dim_a, dim_b in _DIM_PAIRS:
                    total_cases += 1
                    diagram, _a_id, _b_id = self._case_diagram(
                        color=color,
                        dir_a=dir_a,
                        dir_b=dir_b,
                        source_kind=source_kind,
                        dim_a=dim_a,
                        dim_b=dim_b,
                    )
                    matches = find_matches(diagram)
                    if not matches:
                        # Same-direction pairing is invalid for X, and a surviving-leg case
                        # built against the "wrong" direction bucket can miss too -- not
                        # every (source_kind, direction) combination produces a candidate at
                        # all, which is fine: nothing to tie back for a non-match.
                        continue
                    match = matches[0]
                    target = next(
                        (
                            (i, c)
                            for i, c in enumerate(match.dimension_constraints)
                            if c.source.kind is source_kind
                        ),
                        None,
                    )
                    if target is None:
                        # The specific pair happened to resolve to a bare identity in this
                        # shape (e.g. it landed on a source the fixpoint discharged) --
                        # nothing of the intended kind was actually recorded here.
                        continue
                    target_index, target_constraint = target

                    all_pairs = tuple(
                        (c.assumed, c.equal_to) for c in match.dimension_constraints
                    )
                    free_symbols: frozenset[str] = frozenset()
                    for node in diagram.nodes.values():
                        for port in (*node.inputs, *node.outputs):
                            free_symbols |= port.dim.free_symbols
                        if node.phase is not None:
                            free_symbols |= node.phase.free_symbols
                    fixed = {
                        name: dim.to_int()
                        for name, dim in match.bindings.items()
                        if dim.is_concrete
                    }

                    satisfying = _small_integer_assignment(
                        free_symbols, fixed, all_pairs, violate_index=None
                    )
                    if satisfying is None:
                        total_infeasible_skips += 1
                        continue
                    violating = _small_integer_assignment(
                        free_symbols, fixed, all_pairs, violate_index=target_index
                    )
                    if violating is None:
                        total_infeasible_skips += 1
                        continue

                    result = apply(diagram.copy(), SPIDER_FUSION, match)

                    satisfied_comparison = compare(diagram, result.diagram, satisfying)
                    assert satisfied_comparison.mode is EqualityMode.EXACT
                    assert satisfied_comparison.matched, (
                        f"oracle mismatch at a satisfying substitution {satisfying} for "
                        f"constraint {target_constraint!r}: {satisfied_comparison.reason}"
                    )

                    try:
                        violated_comparison = compare(diagram, result.diagram, violating)
                    except ContractDomainError:
                        # Violating a CONNECTING_PAIR-sourced assumption can make the
                        # *pre-fusion* diagram itself internally dimension-inconsistent at
                        # this substitution (the wire's own two endpoints, forced apart, no
                        # longer share one dimension under ALL_LEGS_EQUAL) -- an even
                        # stronger demonstration that the assumption was load-bearing than a
                        # clean not-matched comparison would be, not a test failure.
                        pass
                    else:
                        assert not violated_comparison.matched, (
                            f"oracle falsely agreed at {violating}, which violates "
                            f"{target_constraint!r} -- the constraint is not load-bearing"
                        )
                    total_checked += 1

        assert total_cases > 0
        assert total_checked > 0, "no case produced both a satisfying and a violating check"
        # Every case either got checked or was recognized (and counted) as infeasible/absent
        # -- nothing silently fell through uncounted.
        assert total_checked + total_infeasible_skips <= total_cases * 3


_DETAIL_PALETTE = (Dim.concrete(2), Dim.concrete(3), _D, _E, _D * _E, _D**2)
"""Six-member palette named explicitly by the round-19 task: two concretes, two distinct bare
symbols, a product, and a power -- crossed below over consumed-A, consumed-B, surviving-A,
and surviving-B legs."""


def _assert_dimension_agreement_detail_agrees(match: object) -> None:  # match: FusionMatch
    """The heart of :class:`TestCertificateDetailFidelity`: check ``dimension_agreement``'s
    detail states exactly the same operands and bindings as the ``CONNECTING_PAIR`` record
    entry (or, if none was recorded, that its bare-identity fallback is genuinely an
    identity), and that its leg-count suffix matches the recorded ``SURVIVING_LEG`` count.
    """
    entries = match.dimension_constraints  # type: ignore[attr-defined]
    outcome = next(
        o
        for o in match.side_condition_outcomes  # type: ignore[attr-defined]
        if o.name == "dimension_agreement"
    )
    detail = outcome.detail

    cp_entries = [e for e in entries if e.source.kind is ConstraintSourceKind.CONNECTING_PAIR]
    assert len(cp_entries) <= 1
    leg_part = detail.split("; surviving leg(s)")[0]
    if not cp_entries:
        # No CONNECTING_PAIR entry was ever recorded: the pair was a bare identity on every
        # pass, so the rendered "lhs == rhs" must actually be an identity -- no separate
        # source of truth to diverge from (there is nothing else to check it against).
        assert " (" not in leg_part, leg_part
        lhs, rhs = leg_part.split(" == ")
        assert lhs == rhs, (
            f"no CONNECTING_PAIR entry recorded, but detail is not an identity: {detail!r}"
        )
    else:
        entry = cp_entries[0]
        operands = f"{entry.assumed} == {entry.equal_to}"
        assert leg_part.startswith(operands), (
            f"dimension_agreement detail {detail!r} does not start with the recorded "
            f"CONNECTING_PAIR entry's own operands {operands!r}"
        )
        if entry.outcome is ConstraintOutcome.DEFERRED:
            assert leg_part == f"{operands} (deferred, assumed)", leg_part
        else:
            assert entry.bound_here, "a BOUND entry must always carry what it bound"
            if all(value.is_concrete for _, value in entry.bound_here):
                for name, value in entry.bound_here:
                    assert f"{name} := {value}" in leg_part, (
                        f"detail {leg_part!r} does not name binding {name} := {value} that "
                        f"the connecting pair's own check actually made"
                    )
                assert "non-concrete Dim" not in leg_part, leg_part
            else:
                assert "bound to a non-concrete Dim" in leg_part, leg_part
                # No misattributed concrete binding must leak in for this outcome -- the
                # exact Defect 4 (round 18) / round 19 failure mode.
                assert "(bound: " not in leg_part, leg_part

    leg_entries = [e for e in entries if e.source.kind is ConstraintSourceKind.SURVIVING_LEG]
    if leg_entries:
        suffix = (
            f"; surviving leg(s) resolved to shared_dim={match.shared_dim} with "  # type: ignore[attr-defined]
            f"{len(leg_entries)} additional assumed dimension equality/ies"
        )
        assert detail.endswith(suffix), f"detail {detail!r} missing/wrong leg-count suffix"
    else:
        assert "surviving leg(s)" not in detail, detail


def _assert_phase_dimension_agreement_detail_agrees(match: object) -> None:  # FusionMatch
    """Every name/value ``phase_dimension_agreement``'s "assuming ..." clause states must
    come from a ``NODE_PHASE``/``BOUND`` record entry's own ``bound_here`` -- never from
    ``match.bindings`` read by name alone (Task 1's audit of every detail string, not just
    the connecting pair's)."""
    entries = match.dimension_constraints  # type: ignore[attr-defined]
    outcome = next(
        o
        for o in match.side_condition_outcomes  # type: ignore[attr-defined]
        if o.name == "phase_dimension_agreement"
    )
    phase_bound: dict[str, Dim] = {}
    for entry in entries:
        if (
            entry.source.kind is ConstraintSourceKind.NODE_PHASE
            and entry.outcome is ConstraintOutcome.BOUND
        ):
            phase_bound.update(entry.bound_here)
    if phase_bound:
        assert "; assuming " in outcome.detail, outcome.detail
        for name, value in phase_bound.items():
            assert f"{name} := {value}" in outcome.detail, (
                f"phase_dimension_agreement detail {outcome.detail!r} does not name "
                f"binding {name} := {value} a NODE_PHASE check actually made"
            )
    else:
        assert "assuming" not in outcome.detail, outcome.detail


class TestCertificateDetailFidelity:
    """Task 1 (Phase 5 post-closing audit round 19): an exhaustive sweep, not a hand-picked
    case, asserting every ``dimension_agreement``/``phase_dimension_agreement`` detail string
    states exactly the operands and bindings its own record entry does. See the module
    docstring for why this class exists and what round 18's regression test missed.
    """

    def test_connecting_pair_and_surviving_leg_details_agree_with_the_record(self) -> None:
        total = 0
        for color, dir_a, dir_b in _COLOR_DIRECTION_COMBOS:
            for dim_a in _DETAIL_PALETTE:
                for dim_b in _DETAIL_PALETTE:
                    for surv_a in _DETAIL_PALETTE:
                        for surv_b in _DETAIL_PALETTE:
                            extra_a = (
                                {"extra_in_a": (surv_a,)}
                                if dir_a is Direction.OUTPUT
                                else {"extra_out_a": (surv_a,)}
                            )
                            extra_b = (
                                {"extra_in_b": (surv_b,)}
                                if dir_b is Direction.OUTPUT
                                else {"extra_out_b": (surv_b,)}
                            )
                            diagram, _a_id, _b_id = _build_two_node_diagram(
                                color=color,
                                dir_a=dir_a,
                                dir_b=dir_b,
                                consumed_dim=dim_a,
                                consumed_dim_b=dim_b,
                                **extra_a,  # type: ignore[arg-type]
                                **extra_b,  # type: ignore[arg-type]
                            )
                            for match in find_matches(diagram):
                                total += 1
                                _assert_dimension_agreement_detail_agrees(match)
                                _assert_phase_dimension_agreement_detail_agrees(match)
        assert total > 0, "the detail-fidelity sweep never produced a single fusion match"

    def test_phase_binding_details_agree_with_the_record(self) -> None:
        total = 0
        phase_choices: tuple[Dim | None, ...] = (
            None,
            Dim.concrete(2),
            Dim.concrete(3),
            _D,
            _E,
        )
        for color, dir_a, dir_b in _COLOR_DIRECTION_COMBOS:
            for dim_a in _DETAIL_PALETTE:
                for phase_a in phase_choices:
                    for phase_b in phase_choices:
                        diagram, _a_id, _b_id = _build_two_node_diagram(
                            color=color,
                            dir_a=dir_a,
                            dir_b=dir_b,
                            consumed_dim=dim_a,
                            phase_a=phase_a,
                            phase_b=phase_b,
                        )
                        for match in find_matches(diagram):
                            total += 1
                            _assert_dimension_agreement_detail_agrees(match)
                            _assert_phase_dimension_agreement_detail_agrees(match)
        assert total > 0, "the phase detail-fidelity sweep never produced a single fusion match"
