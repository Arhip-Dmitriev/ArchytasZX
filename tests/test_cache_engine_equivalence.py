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

"""Establishes Phase 12's done-when, "no change in results": a cached strategy run returns the
outcome, the diagram, the certificate and the oracle verdict an uncached one returns, under every
cache shape and under adversarial cache reuse."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable, Sequence
from typing import ClassVar

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.compare import compare_structure
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.cache import (
    CacheGrammarError,
    IncrementalMatcher,
    MatchCache,
    RewriteCache,
)
from archytaszx.rewrite.engine import (
    StopReason,
    StrategyOutcome,
    apply,
    apply_until_fixpoint,
    normal_form_rules,
    toward_normal_form,
)
from archytaszx.rewrite.rule import (
    BuildResult,
    DimensionConstraint,
    Match,
    Pattern,
    Quantifiers,
    RewriteError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
)
from archytaszx.rewrite.rules_library import (
    BIALGEBRA,
    FOURIER_CANCELLATION,
    IDENTITY_REMOVAL,
    SPIDER_FUSION,
    STATE_COPY,
    ZX_CAP,
)
from archytaszx.semantics.certificate import Certificate, certify, verify
from archytaszx.semantics.check import compare

from .test_incremental_match import (
    CORPUS_NAMES,
    cap_pairs,
    corpus,
    fourier_chain,
    fusion_chain,
)
from .test_rules_library_phase11 import identity_chain, inp, out

ORACLE_ASSIGNMENT = {"d": 3}
"""The concrete dimension the oracle checks every cached rewrite at."""

FAST_SWEEP_SEEDS = tuple(range(6))
"""The randomized seeds the default tier sweeps."""

SLOW_SWEEP_SEEDS = tuple(range(600))
"""The randomized seeds the ``slow`` tier sweeps."""

CacheFactory = Callable[[Sequence[Rule]], RewriteCache | None]
"""How each cache shape is built for one rule set."""


def _no_cache(rules: Sequence[Rule]) -> RewriteCache | None:
    """No cache at all."""
    return None


def _memo_cache(rules: Sequence[Rule]) -> RewriteCache | None:
    """A plain match memo."""
    return RewriteCache()


def _incremental_cache(rules: Sequence[Rule]) -> RewriteCache | None:
    """A match memo plus an ``IncrementalMatcher`` over the rule set's own patterns."""
    return RewriteCache(incremental=IncrementalMatcher([rule.pattern for rule in rules]))


def _small_memo_cache(rules: Sequence[Rule]) -> RewriteCache | None:
    """A match memo holding one entry, so every call evicts."""
    return RewriteCache(match_cache=MatchCache(max_entries=1))


def _shared_incremental_cache(rules: Sequence[Rule]) -> RewriteCache | None:
    """An ``IncrementalMatcher`` sharing one ``MatchCache`` with the engine's memo."""
    memo = MatchCache()
    patterns = [rule.pattern for rule in rules]
    return RewriteCache(match_cache=memo, incremental=IncrementalMatcher(patterns, cache=memo))


CACHE_SHAPES: dict[str, CacheFactory] = {
    "no_cache": _no_cache,
    "memo": _memo_cache,
    "incremental": _incremental_cache,
    "one_entry_memo": _small_memo_cache,
    "shared_memo": _shared_incremental_cache,
}
"""Every cache shape a strategy run must be invariant under."""

CACHED_SHAPES: tuple[str, ...] = tuple(name for name in CACHE_SHAPES if name != "no_cache")
"""The cache shapes to compare against an uncached run."""


@dataclasses.dataclass(frozen=True, slots=True)
class _Run:
    """One strategy run's result: an outcome, or the exception type and message it raised."""

    outcome: StrategyOutcome | None
    failure: tuple[str, str] | None


def run_fixpoint(diagram: Diagram, rules: Sequence[Rule], cache: RewriteCache | None) -> _Run:
    """``apply_until_fixpoint`` recorded as a :class:`_Run`."""
    try:
        return _Run(apply_until_fixpoint(diagram, rules, cache=cache), None)
    except (RewriteError, KeyError, IndexError) as exc:
        return _Run(None, (type(exc).__name__, str(exc)))


def run_normal_form(diagram: Diagram, cache: RewriteCache | None) -> _Run:
    """``toward_normal_form`` recorded as a :class:`_Run`."""
    try:
        return _Run(toward_normal_form(diagram, cache=cache), None)
    except (RewriteError, KeyError, IndexError) as exc:
        return _Run(None, (type(exc).__name__, str(exc)))


def assert_runs_agree(expected: _Run, found: _Run, label: str) -> None:
    """Pin two runs equal field for field, diagram id for id, or identically failed."""
    assert (expected.failure is None) == (found.failure is None), label
    if expected.failure is not None:
        assert expected.failure == found.failure, label
        return
    first, second = expected.outcome, found.outcome
    assert first is not None
    assert second is not None
    assert repr(first.steps) == repr(second.steps), label
    assert first.scalar_accumulated == second.scalar_accumulated, label
    assert first.stop_reason is second.stop_reason, label
    assert first.steps_attempted == second.steps_attempted, label
    structural = compare_structure(first.diagram, second.diagram)
    assert structural.identical, f"{label}: {structural.reason}"


def certificate_for(
    diagram: Diagram, rules: Sequence[Rule], cache: RewriteCache | None, limit: int = 8
) -> Certificate:
    """A certificate over the same first-match-first walk the strategy loop takes."""
    current = diagram
    results = []
    for _ in range(limit):
        chosen = None
        for rule in rules:
            matches = (
                rule.pattern.find_matches(current)
                if cache is None
                else cache.matches(rule.pattern, current)
            )
            if matches:
                chosen = (rule, matches[0])
                break
        if chosen is None:
            break
        result = apply(current, chosen[0], chosen[1])
        results.append(result)
        current = result.diagram
    return certify(diagram, results)


def random_diagram(seed: int) -> Diagram:
    """A small deterministic pseudo-random diagram of Z and X spiders."""
    rng = random.Random(seed)
    dim = Dim.concrete(2)
    diagram = Diagram()
    node_ids: list[NodeId] = []
    for _ in range(rng.randint(2, 5)):
        generator = rng.choice((Z_SPIDER, X_SPIDER))
        inputs = rng.randint(0, 2)
        outputs = rng.randint(1, 2)
        node_ids.append(diagram.add_node(generator, [dim] * inputs, [dim] * outputs))
    free_outputs = [
        out(node_id, index)
        for node_id in node_ids
        for index in range(diagram.nodes[node_id].num_outputs)
    ]
    free_inputs = [
        inp(node_id, index)
        for node_id in node_ids
        for index in range(diagram.nodes[node_id].num_inputs)
    ]
    rng.shuffle(free_outputs)
    rng.shuffle(free_inputs)
    for source, target in zip(free_outputs, free_inputs):
        if source.node_id != target.node_id:
            diagram.add_wire(source, target)
    claimed = {ref for wire in diagram.wires for ref in (wire.a, wire.b)}
    diagram.set_boundary_inputs([ref for ref in free_inputs if ref not in claimed])
    diagram.set_boundary_outputs([ref for ref in free_outputs if ref not in claimed])
    return diagram


@dataclasses.dataclass(frozen=True, slots=True)
class _ScriptedMatch:
    """A hand-built ``Match`` the stateful pattern below reports."""

    side_condition_outcomes: tuple[SideConditionOutcome, ...] = ()
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        return True

    @property
    def support_node_ids(self) -> tuple[NodeId, ...]:
        return ()


class _StatefulPattern(Pattern):
    """A hand-written ``Pattern`` holding mutable class state behind an identity ``repr``."""

    calls: ClassVar[list[int]] = []

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        self.calls.append(len(diagram.nodes))
        return (_ScriptedMatch(),) if len(self.calls) <= 2 else ()


def _no_op_builder(working: Diagram, match: Match) -> BuildResult:
    """A builder that consumes nothing and creates nothing."""
    return BuildResult(
        diagram=working,
        new_node_ids=(),
        consumed_node_ids=(),
        consumed_wires=(),
        port_mapping={},
        scalar_introduced=Scalar.one(),
    )


def stateful_rule() -> Rule:
    """A rule whose pattern is a fresh stateful class the cache cannot key."""

    class _Fresh(_StatefulPattern):
        calls: ClassVar[list[int]] = []

    return Rule(
        name="stateful_no_op",
        pattern=_Fresh(),
        builder=_no_op_builder,
        side_conditions=(),
        quantifiers=Quantifiers(),
        scalar_introduced=Scalar.one(),
    )


PRIORITY_RULE_SETS: dict[str, tuple[Rule, ...]] = {
    "fusion_first": (SPIDER_FUSION, IDENTITY_REMOVAL, ZX_CAP),
    "identity_first": (IDENTITY_REMOVAL, SPIDER_FUSION, ZX_CAP),
    "cap_first": (ZX_CAP, IDENTITY_REMOVAL, SPIDER_FUSION),
    "fourier_first": (FOURIER_CANCELLATION, SPIDER_FUSION, IDENTITY_REMOVAL),
}
"""Rule orders whose first applicable rule differs, to pin priority under every cache shape."""

ALTERNATIVE_RULE_SETS: dict[str, tuple[Rule, ...]] = {
    "normal_form": normal_form_rules(),
    "fusion_only": (SPIDER_FUSION,),
    "with_bialgebra": (BIALGEBRA, SPIDER_FUSION, IDENTITY_REMOVAL),
    "with_state_copy": (STATE_COPY, ZX_CAP, SPIDER_FUSION),
}
"""The rule sets one cache is reused across."""


class TestApplyUntilFixpointIsCacheInvariant:
    @pytest.mark.parametrize("shape", CACHED_SHAPES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_a_cached_run_matches_an_uncached_one(self, name: str, shape: str) -> None:
        rules = normal_form_rules()
        expected = run_fixpoint(corpus()[name], rules, None)
        found = run_fixpoint(corpus()[name], rules, CACHE_SHAPES[shape](rules))
        assert_runs_agree(expected, found, f"{name}/{shape}")

    @pytest.mark.parametrize("shape", CACHED_SHAPES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_toward_normal_form_matches_an_uncached_one(self, name: str, shape: str) -> None:
        rules = normal_form_rules()
        expected = run_normal_form(corpus()[name], None)
        found = run_normal_form(corpus()[name], CACHE_SHAPES[shape](rules))
        assert_runs_agree(expected, found, f"{name}/{shape}")

    def test_the_corpus_really_rewrites_under_the_normal_form_rules(self) -> None:
        rules = normal_form_rules()
        steps = 0
        for diagram in corpus().values():
            run = run_fixpoint(diagram, rules, None)
            assert run.outcome is not None
            steps += len(run.outcome.steps)
        assert steps >= 10

    @pytest.mark.parametrize("ruleset", tuple(ALTERNATIVE_RULE_SETS))
    def test_every_rule_set_is_cache_invariant_over_the_corpus(self, ruleset: str) -> None:
        rules = ALTERNATIVE_RULE_SETS[ruleset]
        for name in CORPUS_NAMES:
            expected = run_fixpoint(corpus()[name], rules, None)
            for shape in CACHED_SHAPES:
                found = run_fixpoint(corpus()[name], rules, CACHE_SHAPES[shape](rules))
                assert_runs_agree(expected, found, f"{ruleset}/{name}/{shape}")

    def test_a_guard_trip_is_reported_identically(self) -> None:
        rules = (stateful_rule(),)
        expected = run_fixpoint(cap_pairs(2, 1), rules, None)
        found = run_fixpoint(cap_pairs(2, 1), rules, RewriteCache())
        assert expected.outcome is not None
        assert expected.outcome.stop_reason is not StopReason.FIXPOINT
        assert_runs_agree(expected, found, "guard trip")


class TestCertificatesAreIdentical:
    @pytest.mark.parametrize(
        "name", ("identity_chain", "fourier_chain_four", "cap", "fusion_chain", "ghz_with_copy")
    )
    def test_a_cached_walk_certifies_identically(self, name: str) -> None:
        rules = normal_form_rules()
        uncached = certificate_for(corpus()[name], rules, None)
        cached = certificate_for(corpus()[name], rules, RewriteCache())
        assert uncached.steps
        assert cached == uncached

    @pytest.mark.parametrize("name", ("identity_chain", "fourier_chain_four", "cap"))
    def test_both_certificates_verify_identically(self, name: str) -> None:
        rules = normal_form_rules()
        uncached = certificate_for(corpus()[name], rules, None)
        cached = certificate_for(corpus()[name], rules, RewriteCache())
        first = verify(uncached, {"d": 2})
        second = verify(cached, {"d": 2})
        assert first.verified
        assert second.verified == first.verified
        assert second.reason == first.reason

    def test_a_shared_cache_certifies_every_diagram_identically(self) -> None:
        rules = normal_form_rules()
        cache = RewriteCache()
        certified = 0
        for diagram in corpus().values():
            cached = certificate_for(diagram, rules, cache)
            assert cached == certificate_for(diagram, rules, None)
            certified += len(cached.steps)
        assert certified > 0


class TestOracleSoundnessOfCachedRewrites:
    @pytest.mark.parametrize("shape", CACHED_SHAPES)
    @pytest.mark.parametrize(
        "name", ("identity_chain", "fourier_chain_four", "cap", "fusion_chain", "state_copy")
    )
    def test_a_cached_rewrite_preserves_the_denoted_map(self, name: str, shape: str) -> None:
        rules = normal_form_rules()
        before = corpus()[name]
        run = run_fixpoint(before, rules, CACHE_SHAPES[shape](rules))
        assert run.outcome is not None
        assert run.outcome.steps
        result = compare(before, run.outcome.diagram, ORACLE_ASSIGNMENT)
        assert result.matched, result.reason

    def test_the_uncached_baseline_is_sound_too(self) -> None:
        rules = normal_form_rules()
        checked = 0
        for name in ("identity_chain", "cap", "fusion_chain"):
            before = corpus()[name]
            run = run_fixpoint(before, rules, None)
            assert run.outcome is not None
            assert compare(before, run.outcome.diagram, ORACLE_ASSIGNMENT).matched
            checked += 1
        assert checked == 3


class TestAdversarialCacheReuse:
    def test_one_cache_across_many_different_diagrams(self) -> None:
        rules = normal_form_rules()
        cache = RewriteCache()
        for name in CORPUS_NAMES:
            expected = run_fixpoint(corpus()[name], rules, None)
            assert_runs_agree(expected, run_fixpoint(corpus()[name], rules, cache), name)

    def test_one_cache_across_different_rule_sets(self) -> None:
        cache = RewriteCache()
        diagram = fusion_chain(2, 4)
        for ruleset, rules in ALTERNATIVE_RULE_SETS.items():
            expected = run_fixpoint(diagram, rules, None)
            assert_runs_agree(expected, run_fixpoint(diagram, rules, cache), ruleset)

    def test_a_one_entry_memo_evicting_on_every_call(self) -> None:
        memo = MatchCache(max_entries=1)
        cache = RewriteCache(match_cache=memo)
        rules = normal_form_rules()
        for name in ("identity_chain", "fourier_chain_four", "cap", "fusion_chain"):
            expected = run_fixpoint(corpus()[name], rules, None)
            assert_runs_agree(expected, run_fixpoint(corpus()[name], rules, cache), name)
        assert memo.stats.evictions > 0

    def test_an_incremental_matcher_seeded_on_an_unrelated_diagram(self) -> None:
        rules = normal_form_rules()
        matcher = IncrementalMatcher([rule.pattern for rule in rules])
        matcher.seed(cap_pairs(3, 3))
        cache = RewriteCache(incremental=matcher)
        for name in ("fourier_chain_four", "fusion_chain", "identity_chain"):
            expected = run_fixpoint(corpus()[name], rules, None)
            assert_runs_agree(expected, run_fixpoint(corpus()[name], rules, cache), name)

    def test_two_interleaved_runs_sharing_one_cache(self) -> None:
        rules = normal_form_rules()
        first, second = fourier_chain(2, 8), fusion_chain(2, 5)
        expected_first = run_fixpoint(first, rules, None)
        expected_second = run_fixpoint(second, rules, None)
        cache = RewriteCache(incremental=IncrementalMatcher([r.pattern for r in rules]))
        for _ in range(3):
            assert_runs_agree(expected_first, run_fixpoint(first, rules, cache), "first")
            assert_runs_agree(expected_second, run_fixpoint(second, rules, cache), "second")

    def test_a_diagram_mutated_in_place_between_two_cached_calls(self) -> None:
        rules = normal_form_rules()
        cache = RewriteCache(incremental=IncrementalMatcher([r.pattern for r in rules]))
        diagram = fusion_chain(2, 4)
        assert_runs_agree(
            run_fixpoint(diagram, rules, None), run_fixpoint(diagram, rules, cache), "before"
        )
        dim = Dim.concrete(2)
        extra = diagram.add_node(Z_SPIDER, [dim], [dim])
        diagram.add_wire(out(extra), diagram.boundary_inputs[0])
        assert_runs_agree(
            run_fixpoint(diagram, rules, None), run_fixpoint(diagram, rules, cache), "after"
        )
        diagram.remove_node(extra)
        assert_runs_agree(
            run_fixpoint(diagram, rules, None), run_fixpoint(diagram, rules, cache), "restored"
        )

    def test_a_stateful_pattern_takes_the_uncached_fallback(self) -> None:
        rule = stateful_rule()
        cache = RewriteCache()
        expected = run_fixpoint(cap_pairs(2, 1), (rule,), None)
        found = run_fixpoint(cap_pairs(2, 1), (stateful_rule(),), cache)
        assert_runs_agree(expected, found, "stateful")
        assert cache.stats.unkeyable > 0

    def test_a_stateful_pattern_is_refused_by_an_incremental_matcher(self) -> None:
        with pytest.raises(CacheGrammarError):
            IncrementalMatcher([stateful_rule().pattern])

    def test_a_stateful_pattern_beside_a_tracked_one_still_agrees(self) -> None:
        rules = (SPIDER_FUSION, stateful_rule())
        matcher = IncrementalMatcher([SPIDER_FUSION.pattern])
        cache = RewriteCache(incremental=matcher)
        expected = run_fixpoint(fusion_chain(2, 4), rules, None)
        found = run_fixpoint(fusion_chain(2, 4), (SPIDER_FUSION, stateful_rule()), cache)
        assert_runs_agree(expected, found, "mixed")
        assert cache.stats.unkeyable > 0


class TestRulePriorityIsPreserved:
    @pytest.mark.parametrize("shape", tuple(CACHE_SHAPES))
    @pytest.mark.parametrize("ruleset", tuple(PRIORITY_RULE_SETS))
    def test_the_first_step_names_the_earliest_matching_rule(
        self, ruleset: str, shape: str
    ) -> None:
        rules = PRIORITY_RULE_SETS[ruleset]
        diagram = identity_chain(2)
        expected = None
        for rule in rules:
            if rule.pattern.find_matches(diagram):
                expected = rule.name
                break
        assert expected is not None
        run = run_fixpoint(diagram, rules, CACHE_SHAPES[shape](rules))
        assert run.outcome is not None
        assert run.outcome.steps
        assert run.outcome.steps[0].rule_name == expected

    @pytest.mark.parametrize("shape", tuple(CACHE_SHAPES))
    def test_the_whole_step_sequence_is_order_faithful(self, shape: str) -> None:
        rules = PRIORITY_RULE_SETS["cap_first"]
        expected = run_fixpoint(fourier_chain(2, 4), rules, None)
        found = run_fixpoint(fourier_chain(2, 4), rules, CACHE_SHAPES[shape](rules))
        assert expected.outcome is not None
        assert_runs_agree(expected, found, shape)

    def test_two_rule_orders_really_choose_differently(self) -> None:
        diagram = identity_chain(2)
        first = run_fixpoint(diagram, PRIORITY_RULE_SETS["identity_first"], None)
        second = run_fixpoint(diagram, PRIORITY_RULE_SETS["fusion_first"], None)
        assert first.outcome is not None
        assert second.outcome is not None
        assert first.outcome.steps[0].rule_name != second.outcome.steps[0].rule_name


class TestTheCacheArgumentIsChecked:
    @pytest.mark.parametrize(
        "argument",
        [MatchCache(), IncrementalMatcher([SPIDER_FUSION.pattern]), 3, "cache", object()],
    )
    def test_a_non_rewrite_cache_is_rejected(self, argument: object) -> None:
        with pytest.raises(RewriteGrammarError):
            apply_until_fixpoint(cap_pairs(2, 1), normal_form_rules(), cache=argument)  # type: ignore[arg-type]

    @pytest.mark.parametrize("argument", [MatchCache(), 3, "cache"])
    def test_toward_normal_form_rejects_it_too(self, argument: object) -> None:
        with pytest.raises(RewriteGrammarError):
            toward_normal_form(cap_pairs(2, 1), cache=argument)  # type: ignore[arg-type]


class TestRandomizedSweep:
    @pytest.mark.parametrize("seed", FAST_SWEEP_SEEDS)
    def test_a_random_diagram_is_cache_invariant(self, seed: int) -> None:
        rules = normal_form_rules()
        expected = run_fixpoint(random_diagram(seed), rules, None)
        for shape in CACHED_SHAPES:
            found = run_fixpoint(random_diagram(seed), rules, CACHE_SHAPES[shape](rules))
            assert_runs_agree(expected, found, f"seed {seed}/{shape}")

    def test_the_fast_sweep_rewrites_something(self) -> None:
        rules = normal_form_rules()
        steps = 0
        for seed in FAST_SWEEP_SEEDS:
            run = run_fixpoint(random_diagram(seed), rules, None)
            if run.outcome is not None:
                steps += len(run.outcome.steps)
        assert steps > 0

    @pytest.mark.slow
    def test_the_large_sweep_is_cache_invariant(self) -> None:
        rules = normal_form_rules()
        cache = RewriteCache(incremental=IncrementalMatcher([r.pattern for r in rules]))
        steps = 0
        for seed in SLOW_SWEEP_SEEDS:
            expected = run_fixpoint(random_diagram(seed), rules, None)
            assert_runs_agree(expected, run_fixpoint(random_diagram(seed), rules, cache), str(seed))
            for shape in CACHED_SHAPES:
                found = run_fixpoint(random_diagram(seed), rules, CACHE_SHAPES[shape](rules))
                assert_runs_agree(expected, found, f"seed {seed}/{shape}")
            if expected.outcome is not None:
                steps += len(expected.outcome.steps)
        assert steps > 0
