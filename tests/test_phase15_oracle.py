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

"""Phase 15 done-when: the prover returns checkable derivations for a catalogue of identities,
refutes unequal pairs, reports every failure cleanly, and proves seeded random rewrites."""

from __future__ import annotations

import functools
import random
from collections.abc import Callable
from dataclasses import replace

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, abstract_port_count
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram
from archytaszx.rewrite.egraph import APPLICATION_ERRORS, saturation_rules, simplify
from archytaszx.rewrite.engine import apply, apply_until_fixpoint
from archytaszx.rewrite.rules_library import SPIDER_FUSION, lookup_rule
from archytaszx.rewrite.tactics import SearchLimits, SearchStatus
from archytaszx.semantics.certificate import Certificate, replay, verify
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import sample_grid
from archytaszx.semantics.prove import ProofOutcome, ProofStatus, check_proof, prove

from . import test_decide as TD
from . import test_phase8_oracle as T8
from .test_normal_form import _chain, _single_spider
from .test_phase14_oracle import (
    bialgebra_4,
    bialgebra_5,
    bialgebra_6,
    seeded_diagram,
    state_copy_grows,
)
from .test_rules_library_phase11 import (
    bialgebra_diagram,
    bialgebra_right_hand_side,
    fourier_state_diagram,
    hopf_diagram,
    identity_chain,
    inp,
    out,
    state_copy_diagram,
    triangle_chain,
)

D = Dim("d")
E = Dim("e")
VERIFY_SAMPLES = 3
"""How many grid assignments each certificate half is verified at, at most."""

Pair = tuple[Diagram, Diagram]


# -- builders ---------------------------------------------------------------------------


def rewritten(diagram: Diagram, rule_name: str) -> Diagram:
    """``diagram`` after one application of the named rule at its first passing match."""
    rule = lookup_rule(rule_name)
    match = next(m for m in rule.pattern.find_matches(diagram) if m.all_side_conditions_passed)
    return apply(diagram, rule, match).diagram


def fused_chain(dim: Dim) -> Diagram:
    """``_chain`` in the order ``(2, 1, 0)`` fused to its spider-fusion fixpoint."""
    return apply_until_fixpoint(_chain((2, 1, 0), dim), [SPIDER_FUSION]).diagram


def mixed_dimensions(*, fused: bool) -> Diagram:
    """A Z cup over ``d`` beside an X cup over ``e``; unfused, each cup's second leg passes
    through a one-in-one-out spider of its colour."""
    diagram = Diagram()
    legs = []
    for generator, dim in ((Z_SPIDER, D), (X_SPIDER, E)):
        cup = diagram.add_node(generator, [], [dim, dim])
        if fused:
            legs += [out(cup, 0), out(cup, 1)]
            continue
        middle = diagram.add_node(generator, [dim], [dim])
        diagram.add_wire(out(cup, 1), inp(middle))
        legs += [out(cup, 0), out(middle)]
    diagram.set_boundary_outputs(legs)
    return diagram


def unfused_ghz(dim: Dim) -> Diagram:
    """A Z cup whose second leg feeds a Z copy: three boundary outputs over ``dim``."""
    diagram = Diagram()
    cup = diagram.add_node(Z_SPIDER, [], [dim, dim])
    copy = diagram.add_node(Z_SPIDER, [dim], [dim, dim])
    diagram.add_wire(out(cup, 1), inp(copy))
    diagram.set_boundary_outputs([out(cup, 0), out(copy, 0), out(copy, 1)])
    return diagram


def boxed_state_copy() -> Pair:
    """An X state into a Z spider with ``n`` port-boxed outputs, and ``n`` node-boxed X states
    scaled by ``d ** ((1 - n) / 2)``."""
    left = Diagram()
    state = left.add_node(X_SPIDER, [], [D])
    spider = left.add_node(Z_SPIDER, [D], [D])
    left.add_wire(out(state), inp(spider))
    left.set_boundary_outputs([out(spider)])
    left, _box, n = abstract_port_count(left, out(spider), 1, stem="n")
    right = Diagram()
    copy = right.add_node(X_SPIDER, [], [D])
    right.set_boundary_outputs([out(copy)])
    right.multiply_scalar(Scalar(sp.Pow(D.to_sympy(), (1 - n.to_sympy()) / 2)))
    right.add_bang_box(Mult("n"), node_scope=frozenset({copy}))
    return left, right


# -- catalogues -------------------------------------------------------------------------

PROVED: dict[str, Callable[[], Pair]] = {
    "fusion_chain_d2": lambda: (_chain((0, 1, 2), Dim.concrete(2)), fused_chain(Dim.concrete(2))),
    "fusion_chain_d3": lambda: (_chain((0, 1, 2), Dim.concrete(3)), fused_chain(Dim.concrete(3))),
    "fusion_chain_symbolic": lambda: (_chain((0, 1, 2), D), fused_chain(D)),
    "ghz": lambda: TD.ghz_pair(D),
    "identity_z_vs_x": lambda: (identity_chain(3), identity_chain(3, X_SPIDER)),
    "identity_removal": lambda: (
        identity_chain(3),
        rewritten(identity_chain(3), "identity_removal"),
    ),
    "hopf": lambda: (hopf_diagram(3, 1, 1, 1, 1), rewritten(hopf_diagram(3, 1, 1, 1, 1), "hopf")),
    "hopf_phased": lambda: (
        hopf_diagram(3, 2, 1, 1, 2, phases=True),
        rewritten(hopf_diagram(3, 2, 1, 1, 2, phases=True), "hopf"),
    ),
    "state_copy": lambda: (
        state_copy_diagram(3, 3),
        rewritten(state_copy_diagram(3, 3), "state_copy"),
    ),
    "fourier_colour_change": lambda: (
        fourier_state_diagram(3, is_state=True),
        rewritten(fourier_state_diagram(3, is_state=True), "fourier_state_color_change"),
    ),
    "triangle_orders": lambda: (triangle_chain(3), triangle_chain(3, inverse_first=True)),
    "bialgebra_4": lambda: (bialgebra_4(), simplify(bialgebra_4())[0].diagram),
    "bialgebra_5": lambda: (bialgebra_5(), simplify(bialgebra_5())[0].diagram),
    "bialgebra_6": lambda: (bialgebra_6(), simplify(bialgebra_6())[0].diagram),
    "state_copy_grows": lambda: (
        state_copy_grows(),
        rewritten(state_copy_grows(), "state_copy"),
    ),
    "bialgebra_once": lambda: (
        bialgebra_diagram(2),
        rewritten(bialgebra_diagram(2), "bialgebra"),
    ),
    "boxed_fusion": T8._build_boxed_fusion_family,
    "two_index_fusion": T8._build_two_index_fusion_family,
    "deferred_fusion": lambda: (
        TD.deferred_fusion_diagram(),
        rewritten(TD.deferred_fusion_diagram(), "spider_fusion"),
    ),
    "mixed_dimensions": lambda: (
        mixed_dimensions(fused=False),
        mixed_dimensions(fused=True),
    ),
    "goal_side_only": lambda: (_single_spider(D, 3), unfused_ghz(D)),
}

BIALGEBRA_FIRST = ("bialgebra_4", "bialgebra_5", "bialgebra_6")

NOT_FOUND: dict[str, Callable[[], Pair]] = {
    "boxed_state_copy": boxed_state_copy,
}

UNEQUAL_ORACLE: dict[str, Callable[[], Pair]] = {
    **TD.UNEQUAL_ORACLE,
    "false_near_identity_t8": T8._build_false_near_identity,
    "bialgebra_without_scalar": lambda: (bialgebra_diagram(2), bialgebra_right_hand_side(2)),
}


@functools.cache
def proved(name: str) -> tuple[Diagram, Diagram, ProofOutcome]:
    """The ``PROVED`` catalogue pair ``name`` and its default :func:`prove` outcome."""
    start, goal = PROVED[name]()
    return start, goal, prove(start, goal)


def is_symbolic(start: Diagram, goal: Diagram) -> bool:
    """Whether the families of ``start`` and ``goal`` carry a free symbol."""
    return bool(sample_grid(TD.family(start), TD.family(goal))[0])


def assert_verifies(certificate: Certificate) -> None:
    """Fail unless ``certificate`` verifies at every one of its first VERIFY_SAMPLES grid
    assignments the oracle accepts, and at least one."""
    verified = 0
    for assignment in sample_grid(certificate.initial, certificate.initial):
        if verified >= VERIFY_SAMPLES:
            break
        try:
            report = verify(certificate, assignment)
        except TD.ORACLE_REFUSALS:
            continue
        assert report.verified, (dict(assignment), report.reason)
        verified += 1
    assert verified, "no assignment verified the certificate"


def assert_not_found(outcome: ProofOutcome, status: SearchStatus) -> None:
    """Fail unless ``outcome`` is ``NOT_FOUND`` with a search ending in ``status``."""
    assert outcome.status is ProofStatus.NOT_FOUND, outcome.reason
    assert outcome.search is not None
    assert outcome.search.status is status
    assert outcome.certificate is None and outcome.check is None
    assert outcome.reason.startswith(f"search {status.value} after ")


# -- tests ------------------------------------------------------------------------------


class TestIdentityCatalogue:
    """Every catalogue identity proves, and its certificate checks, replays and verifies."""

    @pytest.mark.parametrize("name", sorted(PROVED))
    def test_proves_and_checks(self, name: str) -> None:
        start, goal, outcome = proved(name)
        assert outcome.status is ProofStatus.PROVED, outcome.reason
        assert outcome.certificate is not None and outcome.check is not None
        assert outcome.check.verified, outcome.check.reason
        check = check_proof(outcome.certificate)
        assert check.verified, check.reason
        assert check.meet_isomorphic
        if is_symbolic(start, goal):
            assert check.samples_checked >= 2, check.reason
        else:
            assert check.samples_checked == 1, check.reason

    @pytest.mark.parametrize("name", sorted(PROVED))
    def test_halves_replay_and_verify(self, name: str) -> None:
        _, _, outcome = proved(name)
        certificate = outcome.certificate
        assert certificate is not None
        for half in (certificate.forward, certificate.backward):
            replayed = replay(half)
            assert replayed.reproduced, replayed.reason
            assert_verifies(half)

    @pytest.mark.parametrize("name", BIALGEBRA_FIRST)
    def test_bialgebra_first_is_proved_by_a_climb(self, name: str) -> None:
        _, _, outcome = proved(name)
        assert outcome.certificate is not None
        assert outcome.certificate.moves[0] == "rule(bialgebra)"
        assert outcome.search is not None and outcome.search.depth == 1

    def test_goal_side_only_proves_backward(self) -> None:
        _, _, outcome = proved("goal_side_only")
        assert outcome.certificate is not None
        assert outcome.certificate.forward.steps == ()
        assert outcome.certificate.backward.steps

    def test_goal_side_only_needs_the_goal_side(self) -> None:
        start, goal = PROVED["goal_side_only"]()
        outcome = prove(start, goal, bidirectional=False)
        assert outcome.status is ProofStatus.NOT_FOUND, outcome.reason

    def test_deferred_fusion_is_conditional(self) -> None:
        _, _, outcome = proved("deferred_fusion")
        assert len(outcome.assumptions) == 1
        assert "conditional" in outcome.reason


class TestNotFoundCatalogue:
    """Identities beyond default limits end ``NOT_FOUND`` with populated search counters."""

    @pytest.mark.parametrize("name", sorted(NOT_FOUND))
    def test_ends_not_found(self, name: str) -> None:
        start, goal = NOT_FOUND[name]()
        outcome = prove(start, goal, induction_depth=0)
        assert outcome.status is ProofStatus.NOT_FOUND, outcome.reason
        result = outcome.search
        assert result is not None and not result.found
        assert result.states >= 2 and result.expanded >= 1
        assert result.depth >= 1
        assert outcome.counterexample is None

    def test_boxed_state_copy_is_oracle_equal(self) -> None:
        start, goal = boxed_state_copy()
        check = TD.oracle_samples(start, goal)
        assert len(check) >= 2
        assert all(matched for _, matched in check), check


class TestInduction:
    """A symbolic-n identity search cannot reach is proved by induction on its multiplicity."""

    def test_the_boxed_state_copy_is_proved_by_induction(self) -> None:
        outcome = prove(*boxed_state_copy())
        assert outcome.status is ProofStatus.PROVED, outcome.reason
        assert outcome.certificate is None and outcome.induction is not None
        assert outcome.induction.index == "n"
        assert outcome.check is not None and outcome.check.verified
        assert outcome.check.samples_checked >= 2

    def test_the_step_exposes_the_hypothesis_by_rewriting(self) -> None:
        induction = prove(*boxed_state_copy()).induction
        assert induction is not None
        assert [step.rule_name for step in induction.expose.steps] == [
            "port_box_unfusion",
            "state_copy",
        ]

    def test_the_proof_rechecks(self) -> None:
        induction = prove(*boxed_state_copy()).induction
        assert induction is not None
        check = check_proof(induction)
        assert check.verified, check.reason

    def test_a_moved_region_fails_the_check(self) -> None:
        induction = prove(*boxed_state_copy()).induction
        assert induction is not None
        check = check_proof(replace(induction, region=frozenset()))
        assert not check.verified

    def test_a_base_case_from_the_wrong_sides_fails_the_check(self) -> None:
        induction = prove(*boxed_state_copy()).induction
        assert induction is not None
        check = check_proof(replace(induction, base=induction.step))
        assert not check.verified
        assert "base case" in check.reason

    def test_a_false_family_is_not_proved_without_the_oracle(self) -> None:
        start, goal = T8._build_false_near_identity()
        outcome = prove(start, goal, refute=False)
        assert outcome.status is not ProofStatus.PROVED, outcome.reason


class TestUnequalCatalogue:
    """Unequal pairs are refuted by interface or by an oracle counterexample."""

    @pytest.mark.parametrize("name", sorted(TD.UNEQUAL_INTERFACE))
    def test_interface_refutes(self, name: str) -> None:
        start, goal = TD.UNEQUAL_INTERFACE[name]()
        outcome = prove(start, goal)
        assert outcome.status is ProofStatus.REFUTED, outcome.reason
        assert outcome.search is None and outcome.counterexample is None
        assert outcome.certificate is None

    @pytest.mark.parametrize("name", sorted(UNEQUAL_ORACLE))
    def test_oracle_refutes(self, name: str) -> None:
        start, goal = UNEQUAL_ORACLE[name]()
        outcome = prove(start, goal)
        assert outcome.status is ProofStatus.REFUTED, outcome.reason
        assert outcome.search is None and outcome.certificate is None
        assert outcome.counterexample is not None
        result = compare(TD.family(start), TD.family(goal), outcome.counterexample)
        assert not result.matched, result.reason

    def test_unrefuted_unequal_pair_is_not_proved(self) -> None:
        start, goal = UNEQUAL_ORACLE["different_phases"]()
        outcome = prove(start, goal, refute=False)
        assert outcome.status is ProofStatus.NOT_FOUND, outcome.reason
        assert outcome.search is not None
        assert outcome.search.status is SearchStatus.EXHAUSTED


class TestFailuresReportCleanly:
    """Each search limit, an empty move set and a tampered certificate fail without raising."""

    @staticmethod
    def pair() -> Pair:
        """The ``bialgebra_4`` catalogue pair."""
        return PROVED["bialgebra_4"]()

    def test_no_moves_exhausts(self) -> None:
        outcome = prove(*self.pair(), moves=())
        assert_not_found(outcome, SearchStatus.EXHAUSTED)
        assert outcome.search is not None and outcome.search.applications == 0

    def test_zero_depth(self) -> None:
        outcome = prove(*self.pair(), limits=SearchLimits(max_depth=0))
        assert_not_found(outcome, SearchStatus.DEPTH_LIMIT)
        assert outcome.search is not None and outcome.search.expanded == 0

    @pytest.mark.parametrize("states", [1, 2])
    def test_state_limit(self, states: int) -> None:
        outcome = prove(*self.pair(), limits=SearchLimits(max_states=states))
        assert_not_found(outcome, SearchStatus.STATE_LIMIT)
        assert outcome.search is not None and outcome.search.states == 2

    @pytest.mark.parametrize("applications", [0, 1])
    def test_application_limit(self, applications: int) -> None:
        outcome = prove(*self.pair(), limits=SearchLimits(max_applications=applications))
        assert_not_found(outcome, SearchStatus.APPLICATION_LIMIT)
        assert outcome.search is not None
        assert outcome.search.applications == applications

    def test_zero_node_margin_prunes(self) -> None:
        outcome = prove(*self.pair(), limits=SearchLimits(node_margin=0))
        assert outcome.status is ProofStatus.NOT_FOUND, outcome.reason
        assert outcome.search is not None and outcome.search.pruned > 0
        assert outcome.reason.startswith(f"search {outcome.search.status.value} after ")

    @pytest.mark.parametrize("name", ["bialgebra_4", "goal_side_only"])
    def test_swapped_halves_fail_the_check(self, name: str) -> None:
        _, _, outcome = proved(name)
        certificate = outcome.certificate
        assert certificate is not None
        swapped = replace(certificate, forward=certificate.backward, backward=certificate.forward)
        check = check_proof(swapped)
        assert not check.verified
        assert "does not begin at start" in check.reason


# -- seeded random sweep ----------------------------------------------------------------

SEEDS = 400
MIN_PROVED_FRACTION = 0.9


def derived_goal(seed: int) -> tuple[Diagram, Diagram, tuple[str, ...]] | None:
    """``seeded_diagram(seed)``, the diagram after 1 to 3 seeded random passing
    :func:`saturation_rules` applications, and their rule names; ``None`` when no diagram or
    no application exists."""
    start = seeded_diagram(seed)
    if start is None:
        return None
    rng = random.Random(seed)
    rules = saturation_rules()
    goal = start
    names: list[str] = []
    for _ in range(rng.randint(1, 3)):
        options = [
            (rule, match)
            for rule in rules
            for match in rule.pattern.find_matches(goal)
            if match.all_side_conditions_passed
        ]
        rng.shuffle(options)
        for rule, match in options:
            try:
                goal = apply(goal, rule, match).diagram
            except APPLICATION_ERRORS:
                continue
            names.append(rule.name)
            break
        else:
            break
    if not names:
        return None
    return start, goal, tuple(names)


@pytest.mark.slow
class TestRandomSweep:
    """Seeded random diagrams against seeded random rewrites of themselves."""

    @pytest.mark.parametrize("seed", range(SEEDS))
    def test_random_rewrite_is_proved(self, seed: int) -> None:
        derived = derived_goal(seed)
        if derived is None:
            pytest.skip("no diagram or no passing rule application")
        start, goal, names = derived
        outcome = prove(start, goal)
        assert outcome.status in (ProofStatus.PROVED, ProofStatus.NOT_FOUND), (
            names,
            outcome.reason,
        )
        if outcome.proved:
            assert outcome.certificate is not None
            check = check_proof(outcome.certificate)
            assert check.verified, (names, check.reason)

    def test_most_random_rewrites_are_proved(self) -> None:
        attempted = 0
        found = 0
        for seed in range(SEEDS):
            derived = derived_goal(seed)
            if derived is None:
                continue
            attempted += 1
            found += prove(derived[0], derived[1]).proved
        assert attempted >= SEEDS // 2
        assert found >= MIN_PROVED_FRACTION * attempted, (found, attempted)
