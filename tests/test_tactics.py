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

"""Phase 15 tactic suite: primitives, combinators, laziness, the chaining invariant, input
immutability, operators, budgets, failures, argument validation, and proof search."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.compare import compare_structure, isomorphic
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.cache import RewriteCache
from archytaszx.rewrite.egraph import APPLICATION_ERRORS, saturation_rules
from archytaszx.rewrite.engine import (
    TerminationGuard,
    apply,
    apply_until_fixpoint,
    normal_form_rules,
)
from archytaszx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionPattern
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import (
    Match,
    RewriteError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
)
from archytaszx.rewrite.rules_library import RULES, SPIDER_FUSION, lookup_rule
from archytaszx.rewrite.tactics import (
    DEFAULT_SEARCH_LIMITS,
    BudgetExhausted,
    ProofPath,
    ProofStep,
    SearchLimits,
    SearchResult,
    SearchStatus,
    Side,
    Tactic,
    TacticContext,
    TacticFailure,
    TacticOutcome,
    attempt,
    choice,
    default_moves,
    fail,
    first,
    fixpoint,
    identity,
    normalize,
    once,
    repeat,
    rewrite_tactics,
    rule,
    search,
    seq,
)
from archytaszx.semantics.certificate import certify, replay
from archytaszx.semantics.check import compare

from .test_egraph import BOOM
from .test_normal_form import _chain, _exact_rewrite, _single_spider
from .test_rules_library_phase11 import bialgebra_diagram, inp, out

D = Dim("d")
FUSE = "rule(spider_fusion)"


def _chain_d() -> Diagram:
    """Three phased Z spiders in a line over ``d``; spider fusion matches twice."""
    return _chain((0, 1, 2), D)


def _two_spiders() -> Diagram:
    """Two phaseless Z spiders, one wire between them, three boundary outputs."""
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, [], [D, D])
    b = diagram.add_node(Z_SPIDER, [D], [D, D])
    diagram.add_wire(out(a, 1), inp(b))
    diagram.set_boundary_outputs([out(a), out(b), out(b, 1)])
    return diagram


def _rule_for(name: str) -> Rule:
    """The registered rule, or a test rule, named ``name``."""
    return {"boom": BOOM, "first_fails": FIRST_FAILS}.get(name) or lookup_rule(name)


def _assert_chained(source: Diagram, outcome: TacticOutcome) -> None:
    """Every result re-applies to the previous diagram id for id, ending on ``outcome.diagram``."""
    previous = source
    for result in outcome.results:
        again = apply(previous, _rule_for(result.step.rule_name), result.step.match)
        comparison = compare_structure(again.diagram, result.diagram)
        assert comparison.identical, comparison.reason
        previous = result.diagram
    assert outcome.diagram is previous


def _assert_replays(source: Diagram, outcome: TacticOutcome) -> None:
    """``outcome``'s results certify from ``source`` and replay onto ``outcome.diagram``."""
    replayed = replay(certify(source, outcome.results), rediscover=False)
    assert replayed.reproduced, replayed.reason
    assert compare_structure(replayed.diagram, outcome.diagram).identical


def _fingerprints(outcomes: tuple[TacticOutcome, ...]) -> list[tuple[tuple[str, str], ...]]:
    """Each outcome's (rule name, match repr) sequence."""
    return [tuple((s.rule_name, repr(s.match)) for s in o.steps) for o in outcomes]


class _FirstFailsPattern(FusionPattern):
    """Spider fusion with the first match's side conditions forced to fail."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        matches = super().find_matches(diagram)
        if not matches:
            return matches
        failed = (SideConditionOutcome("forced", False, "forced to fail"),)
        return (replace(matches[0], side_condition_outcomes=failed), *matches[1:])  # type: ignore[type-var]


FIRST_FAILS = Rule(
    name="first_fails",
    pattern=_FirstFailsPattern(),
    builder=SPIDER_FUSION.builder,
    side_conditions=FUSION_SIDE_CONDITIONS,
    quantifiers=SPIDER_FUSION.quantifiers,
    scalar_introduced=SPIDER_FUSION.scalar_introduced,
)


class _LatePattern(FusionPattern):
    """Spider fusion matching only in diagrams of fewer than three nodes."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        return tuple(super().find_matches(diagram)) if len(diagram.nodes) < 3 else ()


LATE_BOOM = replace(BOOM, name="late_boom", pattern=_LatePattern())


class _Counting(Tactic):
    """Wraps ``child`` and counts how many of its outcomes were pulled."""

    def __init__(self, child: Tactic) -> None:
        self.child = child
        self.pulled = 0

    @property
    def name(self) -> str:
        return f"counting({self.child.name})"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        for outcome in self.child._outcomes(diagram, context):
            self.pulled += 1
            yield outcome


class _Echo(Tactic):
    """Claims one result whose diagram is a copy of the input: a synthetic loop."""

    @property
    def name(self) -> str:
        return "echo"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        source = _chain_d()
        result = apply(source, SPIDER_FUSION, SPIDER_FUSION.pattern.find_matches(source)[0])
        echoed = replace(result, diagram=diagram.copy())
        yield TacticOutcome(echoed.diagram, (echoed,))


TACTICS: dict[str, Callable[[], Tactic]] = {
    "rule": lambda: rule(SPIDER_FUSION),
    "rule_by_name": lambda: rule("spider_fusion"),
    "rule_focus": lambda: rule("spider_fusion", focus=[NodeId(2)]),
    "identity": identity,
    "fail": fail,
    "seq": lambda: seq(rule("spider_fusion"), rule("spider_fusion")),
    "seq_identity": lambda: seq(identity(), rule("spider_fusion"), identity()),
    "first": lambda: first(fail(), rule("spider_fusion")),
    "choice": lambda: choice(rule("spider_fusion"), identity(), normalize()),
    "attempt": lambda: attempt(rule("hopf")),
    "once": lambda: once(rule("spider_fusion")),
    "repeat": lambda: repeat(rule("spider_fusion")),
    "fixpoint": lambda: fixpoint(["spider_fusion"]),
    "normalize": normalize,
    "nested": lambda: attempt(once(rule("spider_fusion")) >> repeat(choice(fail(), normalize()))),
}


class TestPrimitives:
    def test_rule_yields_one_outcome_per_match_in_match_order(self) -> None:
        diagram = _chain_d()
        outcomes = rule(SPIDER_FUSION).all(diagram)
        matches = SPIDER_FUSION.pattern.find_matches(diagram)
        assert len(outcomes) == len(matches) == 2
        assert [o.results[0].step.match for o in outcomes] == list(matches)
        assert all(len(o.results) == 1 for o in outcomes)

    def test_rule_name_resolves_and_unknown_name_is_rejected(self) -> None:
        assert rule("spider_fusion").name == FUSE
        with pytest.raises(RewriteGrammarError, match="no such rule"):
            rule("no_such_rule")

    def test_rule_skips_matches_whose_side_conditions_fail(self) -> None:
        outcomes = rule(FIRST_FAILS).all(_chain_d())
        assert len(outcomes) == 1
        assert outcomes[0].steps[0].match == SPIDER_FUSION.pattern.find_matches(_chain_d())[1]

    def test_rule_focus_keeps_matches_meeting_the_focus(self) -> None:
        diagram = _chain_d()
        assert len(rule("spider_fusion", focus=[NodeId(0)]).all(diagram)) == 1
        assert len(rule("spider_fusion", focus=[NodeId(1)]).all(diagram)) == 2
        assert len(rule("spider_fusion", focus=[NodeId(2)]).all(diagram)) == 1
        assert rule("spider_fusion", focus=[NodeId(99)]).all(diagram) == ()
        assert (
            rule("spider_fusion", focus=[NodeId(2), NodeId(0)]).name
            == "rule(spider_fusion, focus=[0, 2])"
        )

    def test_rule_through_a_cache_matches_the_uncached_outcomes(self) -> None:
        cache = RewriteCache()
        context = TacticContext(cache=cache)
        cached = rule("spider_fusion").all(_chain_d(), context)
        again = rule("spider_fusion").all(_chain_d(), context)
        plain = rule("spider_fusion").all(_chain_d())
        assert _fingerprints(cached) == _fingerprints(plain) == _fingerprints(again)
        assert cache.stats.hits >= 1

    def test_identity_returns_the_input_object(self) -> None:
        diagram = _chain_d()
        (outcome,) = identity().all(diagram)
        assert outcome.diagram is diagram
        assert outcome.results == () and outcome.steps == ()
        assert identity().name == "identity"

    def test_fail_has_no_outcomes(self) -> None:
        assert fail().all(_chain_d()) == ()
        assert fail().run(_chain_d()) is None
        assert fail().name == "fail"

    def test_fixpoint_matches_apply_until_fixpoint(self) -> None:
        diagram = _chain_d()
        (outcome,) = fixpoint(["spider_fusion"]).all(diagram)
        strategy = apply_until_fixpoint(diagram, [SPIDER_FUSION])
        assert outcome.steps == strategy.steps
        assert compare_structure(outcome.diagram, strategy.diagram).identical
        assert fixpoint([SPIDER_FUSION, "hopf"]).name == "fixpoint(spider_fusion, hopf)"

    def test_fixpoint_charges_each_result(self) -> None:
        context = TacticContext()
        (outcome,) = fixpoint(["spider_fusion"]).all(_chain_d(), context)
        assert context.applications == len(outcome.results) == 2

    def test_empty_fixpoint_is_identity_like(self) -> None:
        diagram = _chain_d()
        (outcome,) = fixpoint([]).all(diagram)
        assert outcome.diagram is diagram and outcome.results == ()
        assert fixpoint([]).name == "fixpoint()"

    def test_fixpoint_respects_its_guard(self) -> None:
        guard = TerminationGuard(max_steps=1)
        (outcome,) = fixpoint(["spider_fusion"], guard=guard).all(_chain_d())
        assert len(outcome.results) == 1

    def test_normalize_is_the_normal_form_fixpoint(self) -> None:
        diagram = _chain_d()
        (outcome,) = normalize().all(diagram)
        assert outcome.steps == apply_until_fixpoint(diagram, normal_form_rules()).steps
        assert normalize().name == "normalize"

    def test_rewrite_tactics_and_default_moves(self) -> None:
        names = tuple(f"rule({r.name})" for r in saturation_rules())
        assert tuple(t.name for t in rewrite_tactics()) == names
        assert tuple(t.name for t in rewrite_tactics([SPIDER_FUSION])) == (FUSE,)
        assert rewrite_tactics([]) == ()
        assert tuple(t.name for t in default_moves()) == ("normalize", *names)


class TestCombinators:
    def test_seq_binds_depth_first(self) -> None:
        diagram = _chain_d()
        outcomes = seq(rule("spider_fusion"), rule("spider_fusion")).all(diagram)
        expected = []
        for head in rule("spider_fusion").all(diagram):
            for tail in rule("spider_fusion").all(head.diagram):
                expected.append(head.steps + tail.steps)
        assert [o.steps for o in outcomes] == expected
        assert len(outcomes) == 2 and all(len(o.results) == 2 for o in outcomes)

    def test_seq_with_a_failing_tactic_has_no_outcomes(self) -> None:
        assert seq(rule("spider_fusion"), fail()).all(_chain_d()) == ()
        assert seq(fail(), rule("spider_fusion")).all(_chain_d()) == ()

    def test_seq_of_identities_keeps_the_input_object(self) -> None:
        diagram = _chain_d()
        (outcome,) = seq(identity(), identity()).all(diagram)
        assert outcome.diagram is diagram

    def test_first_takes_the_first_tactic_with_outcomes(self) -> None:
        diagram = _chain_d()
        fused = _fingerprints(rule("spider_fusion").all(diagram))
        assert _fingerprints(first(fail(), rule("spider_fusion")).all(diagram)) == fused
        assert _fingerprints(first(rule("spider_fusion"), identity()).all(diagram)) == fused
        assert first(fail(), fail()).all(diagram) == ()
        (outcome,) = first(rule("hopf"), identity()).all(diagram)
        assert outcome.diagram is diagram

    def test_choice_concatenates_in_order(self) -> None:
        diagram = _chain_d()
        outcomes = choice(identity(), rule("spider_fusion"), fail(), identity()).all(diagram)
        assert [len(o.results) for o in outcomes] == [0, 1, 1, 0]
        assert outcomes[0].diagram is diagram and outcomes[3].diagram is diagram

    def test_attempt(self) -> None:
        diagram = _chain_d()
        (kept,) = attempt(rule("hopf")).all(diagram)
        assert kept.diagram is diagram
        assert len(attempt(rule("spider_fusion")).all(diagram)) == 2
        assert attempt(rule("hopf")).name == "first(rule(hopf), identity)"

    def test_once(self) -> None:
        diagram = _chain_d()
        (outcome,) = once(rule("spider_fusion")).all(diagram)
        assert _fingerprints((outcome,)) == _fingerprints(rule("spider_fusion").all(diagram))[:1]
        assert once(fail()).all(diagram) == ()
        assert once(fail()).name == "once(fail)"

    def test_repeat_runs_to_a_stall(self) -> None:
        diagram = _chain_d()
        (outcome,) = repeat(rule("spider_fusion")).all(diagram)
        assert len(outcome.results) == 2
        assert len(outcome.diagram.nodes) == 1
        assert repeat(rule("spider_fusion")).name == "repeat(rule(spider_fusion))"

    def test_repeat_respects_max_times(self) -> None:
        diagram = _chain_d()
        (one,) = repeat(rule("spider_fusion"), max_times=1).all(diagram)
        assert len(one.results) == 1
        (zero,) = repeat(rule("spider_fusion"), max_times=0).all(diagram)
        assert zero.diagram is diagram and zero.results == ()
        assert repeat(fail(), max_times=3).name == "repeat(fail, max_times=3)"

    def test_repeat_stops_on_no_outcome_and_on_no_results(self) -> None:
        diagram = _chain_d()
        for tactic in (repeat(fail()), repeat(identity())):
            (outcome,) = tactic.all(diagram)
            assert outcome.diagram is diagram and outcome.results == ()

    def test_repeat_drops_the_looping_outcome(self) -> None:
        diagram = _chain_d()
        (outcome,) = repeat(_Echo()).all(diagram)
        assert outcome.diagram is diagram and outcome.results == ()

    def test_operators(self) -> None:
        a, b, c = rule("spider_fusion"), identity(), fail()
        assert (a >> b).name == f"seq({FUSE}, identity)"
        assert (a >> b >> c).name == f"seq({FUSE}, identity, fail)"
        assert (c | a).name == f"first(fail, {FUSE})"
        assert (a | b | c).name == f"first({FUSE}, identity, fail)"
        assert repr(a >> (b | c)) == f"seq({FUSE}, first(identity, fail))"
        assert choice(choice(a, b), c).name == f"choice({FUSE}, identity, fail)"
        assert _fingerprints((a >> b).all(_chain_d())) == _fingerprints(a.all(_chain_d()))

    def test_operators_reject_non_tactics(self) -> None:
        with pytest.raises(RewriteGrammarError):
            identity() >> "fail"  # type: ignore[operator]
        with pytest.raises(RewriteGrammarError):
            identity() | 3  # type: ignore[operator]


class TestLaziness:
    def test_run_pulls_one_outcome(self) -> None:
        counting = _Counting(rule("spider_fusion"))
        assert counting.run(_chain_d()) is not None
        assert counting.pulled == 1

    def test_first_pulls_only_what_is_consumed(self) -> None:
        counting = _Counting(rule("spider_fusion"))
        later = _Counting(identity())
        assert first(counting, later).run(_chain_d()) is not None
        assert (counting.pulled, later.pulled) == (1, 0)

    def test_once_pulls_one(self) -> None:
        counting = _Counting(rule("spider_fusion"))
        assert len(once(counting).all(_chain_d())) == 1
        assert counting.pulled == 1

    def test_all_with_limit_pulls_at_most_limit(self) -> None:
        counting = _Counting(rule("spider_fusion"))
        assert len(counting.all(_chain_d(), limit=1)) == 1
        assert counting.pulled == 1
        assert counting.all(_chain_d(), limit=0) == ()

    def test_choice_pulls_only_what_is_consumed(self) -> None:
        counting, later = _Counting(rule("spider_fusion")), _Counting(identity())
        assert len(choice(counting, later).all(_chain_d(), limit=1)) == 1
        assert (counting.pulled, later.pulled) == (1, 0)

    def test_seq_streams_its_first_tactic(self) -> None:
        counting = _Counting(rule("spider_fusion"))
        assert seq(counting, identity()).run(_chain_d()) is not None
        assert counting.pulled == 1

    def test_rule_applies_only_what_is_pulled(self) -> None:
        context = TacticContext()
        assert rule("spider_fusion").run(_chain_d(), context) is not None
        assert context.applications == 1

    def test_outcomes_is_lazy_until_iterated(self) -> None:
        context = TacticContext()
        stream = rule("spider_fusion").outcomes(_chain_d(), context)
        assert context.applications == 0
        next(stream)
        assert context.applications == 1


SOURCES: dict[str, Callable[[], Diagram]] = {
    "chain": _chain_d,
    "bialgebra": lambda: bialgebra_diagram(2),
}


class TestChainingInvariant:
    @pytest.mark.parametrize("source", SOURCES.values(), ids=SOURCES.keys())
    @pytest.mark.parametrize("make", TACTICS.values(), ids=TACTICS.keys())
    def test_every_outcome_chains(
        self, make: Callable[[], Tactic], source: Callable[[], Diagram]
    ) -> None:
        diagram = source()
        for outcome in make().all(diagram):
            _assert_chained(diagram, outcome)
            if not outcome.results:
                assert outcome.diagram is diagram

    @pytest.mark.parametrize("source", SOURCES.values(), ids=SOURCES.keys())
    @pytest.mark.parametrize("make", TACTICS.values(), ids=TACTICS.keys())
    def test_every_outcome_replays_as_a_certificate(
        self, make: Callable[[], Tactic], source: Callable[[], Diagram]
    ) -> None:
        diagram = source()
        for outcome in make().all(diagram):
            _assert_replays(diagram, outcome)

    @pytest.mark.parametrize("source", SOURCES.values(), ids=SOURCES.keys())
    @pytest.mark.parametrize("make", TACTICS.values(), ids=TACTICS.keys())
    def test_inputs_are_never_mutated(
        self, make: Callable[[], Tactic], source: Callable[[], Diagram]
    ) -> None:
        diagram = source()
        snapshot = diagram.copy()
        outcomes = make().all(diagram)
        assert compare_structure(diagram, snapshot).identical
        for outcome in outcomes:
            for result in outcome.results:
                assert result.diagram is not diagram

    def test_outcomes_are_oracle_equal_to_the_input(self) -> None:
        diagram = _chain_d()
        for make in TACTICS.values():
            for outcome in make().all(diagram):
                assert compare(diagram, outcome.diagram, {"d": 2}).matched


class TestBudgetAndFailures:
    def test_charge_counts_and_raises_at_the_ceiling(self) -> None:
        context = TacticContext(max_applications=2)
        context.charge()
        context.charge()
        with pytest.raises(BudgetExhausted) as info:
            context.charge()
        assert info.value.applications == 2 == context.applications
        assert isinstance(info.value, RewriteError)

    def test_unbounded_context(self) -> None:
        context = TacticContext()
        for _ in range(5):
            context.charge()
        assert context.applications == 5 and context.max_applications is None
        assert context.cache is None and context.failures == []

    def test_rule_raises_when_the_budget_runs_out(self) -> None:
        context = TacticContext(max_applications=1)
        with pytest.raises(BudgetExhausted):
            rule("spider_fusion").all(_chain_d(), context)
        assert context.applications == 1

    def test_fixpoint_raises_when_the_budget_runs_out(self) -> None:
        context = TacticContext(max_applications=1)
        with pytest.raises(BudgetExhausted):
            normalize().all(_chain_d(), context)
        assert context.applications == 1

    def test_zero_budget(self) -> None:
        context = TacticContext(max_applications=0)
        assert identity().all(_chain_d(), context) != ()
        with pytest.raises(BudgetExhausted):
            rule("spider_fusion").run(_chain_d(), context)

    def test_raising_rule_records_failures_and_yields_nothing(self) -> None:
        context = TacticContext()
        assert rule(BOOM).all(_chain_d(), context) == ()
        assert context.applications == 2
        assert [f.rule_name for f in context.failures] == ["boom", "boom"]
        assert all(f.message == "RewriteError: deliberate failure" for f in context.failures)

    def test_raising_fixpoint_records_a_failure_and_keeps_its_prefix(self) -> None:
        context = TacticContext()
        (outcome,) = fixpoint([BOOM]).all(_chain_d(), context)
        assert outcome.results == ()
        assert [f.rule_name for f in context.failures] == ["fixpoint(boom)"]

    def test_raising_fixpoint_keeps_a_non_empty_prefix(self) -> None:
        diagram, context = _chain_d(), TacticContext()
        (outcome,) = fixpoint([LATE_BOOM, "spider_fusion"]).all(diagram, context)
        assert len(outcome.results) == 1 == context.applications
        assert outcome.steps[0].rule_name == "spider_fusion"
        assert [f.rule_name for f in context.failures] == ["fixpoint(late_boom, spider_fusion)"]
        _assert_chained(diagram, outcome)

    @pytest.mark.parametrize(
        "make",
        [
            lambda t: seq(identity(), t),
            lambda t: first(fail(), t),
            lambda t: choice(identity(), t),
            attempt,
            once,
            repeat,
            lambda t: t,
        ],
    )
    @pytest.mark.parametrize("inner", [lambda: rule("spider_fusion"), normalize])
    def test_budget_exhaustion_propagates_through_every_combinator(
        self, make: Callable[[Tactic], Tactic], inner: Callable[[], Tactic]
    ) -> None:
        context = TacticContext(max_applications=0)
        with pytest.raises(BudgetExhausted):
            make(inner()).all(_chain_d(), context)
        assert context.failures == [] and context.applications == 0

    def test_budget_exhausted_pickles(self) -> None:
        again = pickle.loads(pickle.dumps(BudgetExhausted(3)))
        assert isinstance(again, BudgetExhausted) and again.applications == 3
        assert str(again) == str(BudgetExhausted(3))

    def test_failure_errors_are_the_egraph_set(self) -> None:
        assert RewriteError in APPLICATION_ERRORS
        assert issubclass(BudgetExhausted, APPLICATION_ERRORS)


class TestValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: rule(3),  # type: ignore[arg-type]
            lambda: rule("spider_fusion", focus="ab"),  # type: ignore[arg-type]
            lambda: rule("spider_fusion", focus=[True]),  # type: ignore[list-item]
            lambda: rule("spider_fusion", focus=5),  # type: ignore[arg-type]
            lambda: seq(),
            lambda: first(),
            lambda: choice(),
            lambda: seq(identity(), "fail"),  # type: ignore[arg-type]
            lambda: choice(None),  # type: ignore[arg-type]
            lambda: attempt("x"),  # type: ignore[arg-type]
            lambda: once(None),  # type: ignore[arg-type]
            lambda: repeat(None),  # type: ignore[arg-type]
            lambda: repeat(identity(), max_times=-1),
            lambda: repeat(identity(), max_times=True),
            lambda: repeat(identity(), max_times=1.0),  # type: ignore[arg-type]
            lambda: fixpoint("spider_fusion"),
            lambda: fixpoint([3]),  # type: ignore[list-item]
            lambda: fixpoint(["no_such_rule"]),
            lambda: fixpoint([], guard=None),  # type: ignore[arg-type]
            lambda: normalize(guard=3),  # type: ignore[arg-type]
            lambda: rewrite_tactics("spider_fusion"),  # type: ignore[arg-type]
            lambda: rewrite_tactics(["spider_fusion"]),  # type: ignore[list-item]
            lambda: TacticContext(cache=3),  # type: ignore[arg-type]
            lambda: TacticContext(max_applications=-1),
            lambda: TacticContext(max_applications=False),
            lambda: TacticFailure(3, "x"),  # type: ignore[arg-type]
            lambda: TacticFailure("x", None),  # type: ignore[arg-type]
            lambda: TacticOutcome("x", ()),  # type: ignore[arg-type]
            lambda: TacticOutcome(_chain_d(), [1]),  # type: ignore[arg-type]
            lambda: SearchLimits(max_depth=-1),
            lambda: SearchLimits(max_states=True),
            lambda: SearchLimits(max_applications=1.5),  # type: ignore[arg-type]
            lambda: SearchLimits(node_margin=None),  # type: ignore[arg-type]
            lambda: ProofStep(3, ()),  # type: ignore[arg-type]
            lambda: ProofStep("x", ()),
            lambda: ProofPath(_chain_d(), "g", (), ()),  # type: ignore[arg-type]
            lambda: ProofPath(_chain_d(), _chain_d(), [], ()),  # type: ignore[arg-type]
        ],
    )
    def test_bad_arguments_are_rejected(self, call: Callable[[], object]) -> None:
        with pytest.raises(RewriteGrammarError):
            call()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: identity().all("diagram"),  # type: ignore[arg-type]
            lambda: identity().run(_chain_d(), context=3),  # type: ignore[arg-type]
            lambda: identity().outcomes(None),  # type: ignore[arg-type]
            lambda: identity().all(_chain_d(), limit=-1),
            lambda: identity().all(_chain_d(), limit=True),
        ],
    )
    def test_bad_run_arguments_are_rejected_eagerly(self, call: Callable[[], object]) -> None:
        with pytest.raises(RewriteGrammarError):
            call()

    def test_outcome_diagram_must_be_the_last_result_diagram(self) -> None:
        diagram = _chain_d()
        (outcome,) = once(rule("spider_fusion")).all(diagram)
        with pytest.raises(RewriteGrammarError, match="last result"):
            TacticOutcome(outcome.diagram.copy(), outcome.results)

    def test_search_result_path_must_match_status(self) -> None:
        with pytest.raises(RewriteGrammarError):
            SearchResult(SearchStatus.FOUND, None, 1, 0, 0, 0, 0, ())
        path = ProofPath(_chain_d(), _chain_d(), (), ())
        with pytest.raises(RewriteGrammarError):
            SearchResult(SearchStatus.EXHAUSTED, path, 1, 0, 0, 0, 0, ())
        with pytest.raises(RewriteGrammarError):
            SearchResult(SearchStatus.EXHAUSTED, None, True, 0, 0, 0, 0, ())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"moves": "rule"},
            {"moves": [identity(), "x"]},
            {"limits": None},
            {"bidirectional": 1},
            {"cache": 3},
        ],
    )
    def test_search_rejects_bad_arguments(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(RewriteGrammarError):
            search(_chain_d(), _chain_d(), **kwargs)  # type: ignore[arg-type]

    def test_search_rejects_non_diagrams(self) -> None:
        with pytest.raises(RewriteGrammarError):
            search("a", _chain_d())  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            search(_chain_d(), None)  # type: ignore[arg-type]


def _assert_sound_path(result: SearchResult, start: Diagram, goal: Diagram) -> ProofPath:
    """``result`` found a path that replays from both ends and meets in the middle."""
    assert result.found and result.status is SearchStatus.FOUND
    path = result.path
    assert path is not None
    assert path.start is not start and path.goal is not goal
    assert compare_structure(path.start, start).identical
    assert compare_structure(path.goal, goal).identical
    for initial, results, meet in (
        (path.start, path.forward_results, path.forward_meet),
        (path.goal, path.backward_results, path.backward_meet),
    ):
        replayed = replay(certify(initial, results), rediscover=False)
        assert replayed.reproduced, replayed.reason
        assert compare_structure(replayed.diagram, meet).identical
    assert isomorphic(comparison_view(path.forward_meet), comparison_view(path.backward_meet))
    assert path.length == len(path.forward_results) + len(path.backward_results)
    assert path.moves == tuple(step.tactic for step in path.forward)
    assert path.backward_moves == tuple(step.tactic for step in path.backward)
    return path


def _summary(result: SearchResult) -> tuple[object, ...]:
    """Everything observable about ``result`` as reprs."""
    path = result.path
    steps = () if path is None else tuple(repr(r.step) for r in path.forward_results)
    back = () if path is None else tuple(repr(r.step) for r in path.backward_results)
    counters = (result.states, result.expanded, result.applications, result.pruned, result.depth)
    return (result.status, counters, steps, back, repr(result.failures))


class TestSearch:
    def test_isomorphic_inputs_need_no_steps(self) -> None:
        start, goal = _chain((0, 1, 2), D), _chain((2, 0, 1), D)
        result = search(start, goal)
        path = _assert_sound_path(result, start, goal)
        assert path.forward == path.backward == ()
        assert path.length == 0
        assert path.forward_meet is path.start and path.backward_meet is path.goal
        assert (result.states, result.expanded, result.applications, result.depth) == (1, 0, 0, 0)

    def test_one_directional_forward_path(self) -> None:
        start = _chain_d()
        goal = _exact_rewrite(start, "spider_fusion")
        result = search(start, goal, moves=(rule("spider_fusion"),), bidirectional=False)
        path = _assert_sound_path(result, start, goal)
        assert path.moves == (FUSE,) and path.backward == ()
        assert result.depth == 0
        assert compare(start, goal, {"d": 2}).matched

    def test_default_moves_find_a_two_fusion_path(self) -> None:
        start, goal = _chain_d(), apply_until_fixpoint(_chain_d(), [SPIDER_FUSION]).diagram
        result = search(start, goal)
        path = _assert_sound_path(result, start, goal)
        assert path.length >= 1

    def test_meet_needs_the_backward_side(self) -> None:
        start, goal = _single_spider(D, 3), _two_spiders()
        one_way = search(start, goal, bidirectional=False)
        assert one_way.status is SearchStatus.EXHAUSTED and one_way.path is None
        both = search(start, goal)
        path = _assert_sound_path(both, start, goal)
        assert path.forward == () and len(path.backward) == 1
        assert compare(start, goal, {"d": 3}).matched

    def test_goal_side_meet_with_both_halves_non_empty(self) -> None:
        start, goal = _two_spiders(), Diagram()
        a = goal.add_node(Z_SPIDER, [], [D, D, D])
        b = goal.add_node(Z_SPIDER, [D], [D])
        goal.add_wire(out(a, 2), inp(b))
        goal.set_boundary_outputs([out(a), out(a, 1), out(b)])
        assert not isomorphic(comparison_view(start), comparison_view(goal))
        moves = (rule("spider_fusion"),)
        result = search(start, goal, moves=moves)
        path = _assert_sound_path(result, start, goal)
        assert (path.moves, path.backward_moves) == ((FUSE,), (FUSE,))
        assert (result.states, result.expanded, result.depth) == (3, 2, 0)
        assert search(start, goal, moves=moves, bidirectional=False).status is (
            SearchStatus.EXHAUSTED
        )
        assert compare(start, goal, {"d": 2}).matched

    def test_backward_steps_replay_from_the_goal(self) -> None:
        start, goal = _single_spider(D, 3), _two_spiders()
        path = _assert_sound_path(search(start, goal, moves=(rule("spider_fusion"),)), start, goal)
        assert path.backward_moves == (FUSE,)
        assert path.forward_meet is path.start

    def test_exhausted(self) -> None:
        result = search(_chain_d(), _single_spider(D, 4))
        assert result.status is SearchStatus.EXHAUSTED and not result.found
        assert result.path is None and result.depth >= 1

    def test_no_moves_exhausts_at_once(self) -> None:
        result = search(_chain_d(), _single_spider(D, 4), moves=())
        assert result.status is SearchStatus.EXHAUSTED
        assert (result.states, result.expanded, result.depth) == (2, 2, 1)

    def test_depth_limit(self) -> None:
        limits = SearchLimits(max_depth=1)
        result = search(_chain_d(), _single_spider(D, 4), limits=limits)
        assert result.status is SearchStatus.DEPTH_LIMIT and result.depth == 1
        zero = search(_chain_d(), _single_spider(D, 4), limits=SearchLimits(max_depth=0))
        assert zero.status is SearchStatus.DEPTH_LIMIT
        assert (zero.depth, zero.expanded, zero.states) == (0, 0, 2)

    def test_state_limit(self) -> None:
        limits = SearchLimits(max_states=2)
        result = search(_chain_d(), _single_spider(D, 4), limits=limits)
        assert result.status is SearchStatus.STATE_LIMIT
        assert result.states == 2

    def test_application_limit(self) -> None:
        limits = SearchLimits(max_applications=1)
        result = search(_chain_d(), _single_spider(D, 4), limits=limits)
        assert result.status is SearchStatus.APPLICATION_LIMIT
        assert result.applications == 1 and result.path is None

    def test_pruning(self) -> None:
        start = bialgebra_diagram(2)
        moves = (rule("bialgebra"),)
        tight = search(start, _single_spider(D, 4), moves=moves, limits=SearchLimits(node_margin=0))
        assert tight.pruned == 1 and tight.status is SearchStatus.EXHAUSTED
        loose = search(start, _single_spider(D, 4), moves=moves)
        assert loose.pruned == 0 and loose.states > tight.states

    def test_failures_are_reported(self) -> None:
        result = search(_chain_d(), _single_spider(D, 4), moves=(rule(BOOM),))
        assert result.status is SearchStatus.EXHAUSTED
        assert result.failures and all(f.rule_name == "boom" for f in result.failures)

    def test_determinism(self) -> None:
        start, goal = _single_spider(D, 3), _two_spiders()
        assert _summary(search(start, goal)) == _summary(search(start, goal))
        start, goal = _chain_d(), apply_until_fixpoint(_chain_d(), [SPIDER_FUSION]).diagram
        assert _summary(search(start, goal)) == _summary(search(start, goal))
        miss = (_chain_d(), _single_spider(D, 4))
        assert _summary(search(*miss)) == _summary(search(*miss))

    def test_custom_moves_built_from_combinators(self) -> None:
        start = _chain_d()
        goal = apply_until_fixpoint(start, [SPIDER_FUSION]).diagram
        move = repeat(rule("spider_fusion")) >> attempt(rule("hopf"))
        result = search(start, goal, moves=(move,), bidirectional=False)
        path = _assert_sound_path(result, start, goal)
        assert path.moves == ("seq(repeat(rule(spider_fusion)), first(rule(hopf), identity))",)
        assert len(path.forward[0].results) == 2

    def test_cache_does_not_change_the_search(self) -> None:
        start, goal = _single_spider(D, 3), _two_spiders()
        assert _summary(search(start, goal, cache=RewriteCache())) == _summary(search(start, goal))

    def test_inputs_are_not_mutated_and_paths_are_detached(self) -> None:
        start, goal = _chain_d(), apply_until_fixpoint(_chain_d(), [SPIDER_FUSION]).diagram
        snapshots = (start.copy(), goal.copy())
        path = _assert_sound_path(search(start, goal), start, goal)
        assert compare_structure(start, snapshots[0]).identical
        assert compare_structure(goal, snapshots[1]).identical
        results = path.forward_results + path.backward_results
        assert all(r.diagram is not start and r.diagram is not goal for r in results)

    def test_public_constants(self) -> None:
        assert DEFAULT_SEARCH_LIMITS == SearchLimits(4, 2048, 20_000, 4)
        assert [s.value for s in Side] == ["start", "goal"]
        assert set(RULES) >= {"spider_fusion", "bialgebra"}
