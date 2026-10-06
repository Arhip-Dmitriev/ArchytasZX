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

"""Phase 13 normal-form driver suite: reduction, keys, comparison views, certificates, guards,
caching, the ``on_result`` strategy hook, and input validation."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.compare import canonical_key, isomorphic
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram
from archytaszx.rewrite.cache import IncrementalMatcher, RewriteCache
from archytaszx.rewrite.engine import (
    RewriteResult,
    StopReason,
    TerminationGuard,
    apply,
    apply_until_fixpoint,
    normal_form_rules,
)
from archytaszx.rewrite.normal_form import (
    NormalForm,
    comparison_view,
    isomorphic_up_to_scalar,
    normal_form,
    same_normal_form,
)
from archytaszx.rewrite.rule import ConstraintOutcome, RewriteGrammarError
from archytaszx.rewrite.rules_library import SPIDER_FUSION, STATE_COPY
from archytaszx.semantics.certificate import certify, replay, verify
from archytaszx.semantics.check import compare

from .helpers import build_ghz_with_copy
from .test_rules_library_phase11 import (
    fourier_state_diagram,
    hopf_diagram,
    identity_chain,
    inp,
    out,
    state_copy_diagram,
    triangle_chain,
)


def _phase(dim: Dim, turns: sp.Rational) -> PhaseVector:
    """A phase vector over ``dim`` with entry 1 at ``turns``."""
    return PhaseVector(dim, {1: Phase.turns(turns)})


def _chain(order: tuple[int, int, int], dim: Dim) -> Diagram:
    """Three phased Z spiders in a line, each with one boundary output, added in ``order``."""
    turns = (sp.Rational(1, 3), sp.Rational(1, 4), sp.Rational(1, 6))
    specs = ((0, 2), (1, 2), (1, 1))
    diagram = Diagram()
    ids = {}
    for position in order:
        n_in, n_out = specs[position]
        ids[position] = diagram.add_node(
            Z_SPIDER, [dim] * n_in, [dim] * n_out, phase=_phase(dim, turns[position])
        )
    diagram.add_wire(out(ids[0], 1), inp(ids[1]))
    diagram.add_wire(out(ids[1], 1), inp(ids[2]))
    diagram.set_boundary_outputs([out(ids[0]), out(ids[1]), out(ids[2])])
    return diagram


def _single_spider(dim: Dim, legs: int, phase: PhaseVector | None = None) -> Diagram:
    """One Z spider with ``legs`` outputs, all on the boundary."""
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, [], [dim] * legs, phase=phase)
    diagram.set_boundary_outputs([out(node, i) for i in range(legs)])
    return diagram


def _exact_rewrite(diagram: Diagram, rule_name: str) -> Diagram:
    """``diagram`` after one application of the named rule at its first match."""
    rule = {"state_copy": STATE_COPY, "spider_fusion": SPIDER_FUSION}[rule_name]
    return apply(diagram, rule, rule.pattern.find_matches(diagram)[0]).diagram


def _deferred_fusion_diagram() -> Diagram:
    """A Z spider with legs ``d**p`` and ``d**q`` fused into an effect on the ``d**p`` leg."""
    d, p, q = Dim.symbol("d"), Dim.symbol("p"), Dim.symbol("q")
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, [], [d**p, d**q])
    b = diagram.add_node(Z_SPIDER, [d**p], [])
    diagram.add_wire(out(a), inp(b))
    diagram.set_boundary_outputs([out(a, 1)])
    return diagram


EQUAL_PAIRS: dict[str, Callable[[], tuple[Diagram, Diagram]]] = {
    "fusion_chain_orders": lambda: (_chain((0, 1, 2), Dim("d")), _chain((2, 0, 1), Dim("d"))),
    "fusion_chain_concrete": lambda: (
        _chain((1, 2, 0), Dim.concrete(3)),
        _chain((2, 1, 0), Dim.concrete(3)),
    ),
    "ghz_vs_single_spider": lambda: (
        build_ghz_with_copy(Dim("d"))[0],
        _single_spider(Dim("d"), 3),
    ),
    "identity_removal_colours": lambda: (identity_chain(3), identity_chain(3, X_SPIDER)),
    "triangle_inverse_orders": lambda: (
        triangle_chain(3),
        triangle_chain(3, inverse_first=True),
    ),
    "hopf": lambda: (hopf_diagram(3, 1, 1, 1, 1), hopf_diagram(3, 1, 1, 1, 1)),
    "fourier_state": lambda: (
        fourier_state_diagram(3, is_state=True),
        fourier_state_diagram(3, is_state=True),
    ),
    "state_copy": lambda: (
        state_copy_diagram(3, 3),
        _exact_rewrite(state_copy_diagram(3, 3), "state_copy"),
    ),
}

UNEQUAL_PAIRS: dict[str, Callable[[], tuple[Diagram, Diagram]]] = {
    "different_phase": lambda: (
        _single_spider(Dim("d"), 2, _phase(Dim("d"), sp.Rational(1, 3))),
        _single_spider(Dim("d"), 2, _phase(Dim("d"), sp.Rational(1, 4))),
    ),
    "different_leg_count": lambda: (_single_spider(Dim("d"), 2), _single_spider(Dim("d"), 3)),
    "extra_scalar": lambda: (
        _single_spider(Dim("d"), 2),
        _scaled(_single_spider(Dim("d"), 2), Scalar.rational(2)),
    ),
    "z_cup_vs_x_cup": lambda: (_single_spider(Dim("d"), 2), _x_cup(Dim("d"))),
}


def _scaled(diagram: Diagram, factor: Scalar) -> Diagram:
    """``diagram`` with its scalar multiplied by ``factor``."""
    diagram.multiply_scalar(factor)
    return diagram


def _x_cup(dim: Dim) -> Diagram:
    """One X spider with two outputs on the boundary."""
    diagram = Diagram()
    node = diagram.add_node(X_SPIDER, [], [dim, dim])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    return diagram


def _concrete_assignments(diagram: Diagram) -> list[dict[str, int]]:
    """Small dimension assignments over ``diagram``'s free port symbols."""
    names = sorted(
        {
            name
            for node in diagram.nodes.values()
            for port in (*node.inputs, *node.outputs)
            for name in port.dim.free_symbols
        }
    )
    return [{name: value for name in names} for value in (2, 3)]


class TestKnownEqualPairsShareANormalForm:
    @pytest.mark.parametrize("name", sorted(EQUAL_PAIRS))
    def test_identical_normal_forms(self, name: str) -> None:
        left, right = EQUAL_PAIRS[name]()
        nf_left, nf_right = normal_form(left), normal_form(right)
        assert nf_left.reached_fixpoint and nf_right.reached_fixpoint
        assert nf_left.key == nf_right.key
        assert same_normal_form(nf_left, nf_right)

    @pytest.mark.parametrize("name", sorted(EQUAL_PAIRS))
    def test_the_pair_is_oracle_equal(self, name: str) -> None:
        left, right = EQUAL_PAIRS[name]()
        for assignment in _concrete_assignments(left):
            assert compare(left, right, assignment).matched

    @pytest.mark.parametrize("name", sorted(EQUAL_PAIRS))
    def test_each_normal_form_is_oracle_equal_to_its_source(self, name: str) -> None:
        for diagram in EQUAL_PAIRS[name]():
            nf = normal_form(diagram)
            for assignment in _concrete_assignments(diagram):
                assert compare(diagram, nf.diagram, assignment).matched


class TestKnownUnequalPairsHaveDistinctNormalForms:
    @pytest.mark.parametrize("name", sorted(UNEQUAL_PAIRS))
    def test_distinct_normal_forms(self, name: str) -> None:
        left, right = UNEQUAL_PAIRS[name]()
        assert not same_normal_form(normal_form(left), normal_form(right))

    @pytest.mark.parametrize("name", sorted(UNEQUAL_PAIRS))
    def test_the_pair_is_oracle_unequal_somewhere(self, name: str) -> None:
        left, right = UNEQUAL_PAIRS[name]()
        results = [compare(left, right, {"d": value}).matched for value in (2, 3)]
        assert not all(results)


class TestNormalFormShape:
    def test_source_is_a_copy_and_input_is_not_mutated(self) -> None:
        diagram = build_ghz_with_copy(Dim("d"))[0]
        before = canonical_key(diagram), len(diagram.nodes)
        snapshot = diagram.copy()
        nf = normal_form(diagram)
        assert nf.source is not diagram
        assert isomorphic(nf.source, snapshot)
        assert (canonical_key(diagram), len(diagram.nodes)) == before
        assert isomorphic(diagram, snapshot)

    def test_results_match_the_outcome_steps(self) -> None:
        nf = normal_form(_chain((0, 1, 2), Dim("d")))
        assert tuple(result.step for result in nf.results) == nf.outcome.steps
        assert nf.results[-1].diagram is nf.diagram
        assert nf.diagram is nf.outcome.diagram

    def test_key_is_the_canonical_key_of_the_comparison_view(self) -> None:
        nf = normal_form(_chain((0, 1, 2), Dim("d")))
        assert nf.key == canonical_key(comparison_view(nf.diagram))

    def test_an_irreducible_diagram_is_its_own_normal_form(self) -> None:
        diagram = _single_spider(Dim("d"), 2)
        nf = normal_form(diagram)
        assert nf.results == ()
        assert nf.reached_fixpoint
        assert isomorphic(nf.diagram, diagram)

    def test_custom_rules_are_used(self) -> None:
        diagram = _chain((0, 1, 2), Dim("d"))
        assert normal_form(diagram, rules=()).results == ()
        assert len(normal_form(diagram, rules=(SPIDER_FUSION,)).results) == 2

    def test_assumed_constraints_collect_deferred_ones(self) -> None:
        nf = normal_form(_deferred_fusion_diagram())
        assert nf.assumed_constraints
        assert all(c.outcome is ConstraintOutcome.DEFERRED for c in nf.assumed_constraints)
        assert len(set(nf.assumed_constraints)) == len(nf.assumed_constraints)

    def test_no_constraints_when_nothing_deferred(self) -> None:
        assert normal_form(_chain((0, 1, 2), Dim("d"))).assumed_constraints == ()

    def test_determinism(self) -> None:
        first = normal_form(_chain((0, 1, 2), Dim("d")))
        second = normal_form(_chain((0, 1, 2), Dim("d")))
        assert first.key == second.key
        assert first.outcome.steps == second.outcome.steps


class TestComparisonView:
    def test_parameters_are_cleared_and_the_input_kept(self) -> None:
        diagram = _single_spider(Dim("d"), 2)
        diagram.bind_parameter("d", 3)
        view = comparison_view(diagram)
        assert dict(view.parameters) == {}
        assert dict(diagram.parameters) == {"d": 3}

    def test_parameter_environment_does_not_change_the_key(self) -> None:
        plain = _single_spider(Dim("d"), 2)
        bound = _single_spider(Dim("d"), 2)
        bound.bind_parameter("d", 5)
        assert same_normal_form(normal_form(plain), normal_form(bound))

    def test_scalar_is_simplified(self) -> None:
        diagram = _single_spider(Dim("d"), 1)
        diagram.multiply_scalar(Scalar.index_sum(Dim("d"), lambda _k: Scalar.one()))
        view = comparison_view(diagram)
        assert view.scalar == diagram.scalar.simplify()
        assert view.scalar != diagram.scalar
        other = _single_spider(Dim("d"), 1)
        other.multiply_scalar(Scalar.from_dim(Dim("d")))
        assert same_normal_form(normal_form(diagram), normal_form(other))

    def test_spider_legs_are_put_in_canonical_order(self) -> None:
        dim = Dim("d")
        a = _single_spider(dim, 3, _phase(dim, sp.Rational(1, 5)))
        b = _single_spider(dim, 3, _phase(dim, sp.Rational(1, 5)))
        node = next(iter(b.nodes))
        b.set_boundary_outputs([out(node, 2), out(node, 0), out(node, 1)])
        assert not isomorphic(a, b)
        assert isomorphic(comparison_view(a), comparison_view(b))
        assert compare(a, b, {"d": 3}).matched

    def test_zero_scalar_survives(self) -> None:
        diagram = _single_spider(Dim("d"), 1)
        diagram.multiply_scalar(Scalar.zero())
        assert comparison_view(diagram).scalar.is_zero


class TestIsomorphicUpToScalar:
    def test_scalar_and_parameters_are_ignored(self) -> None:
        a = _single_spider(Dim("d"), 2)
        b = _scaled(_single_spider(Dim("d"), 2), Scalar.rational(7))
        b.bind_parameter("d", 2)
        assert isomorphic_up_to_scalar(a, b)
        assert not isomorphic(a, b)

    def test_graph_difference_is_seen(self) -> None:
        assert not isomorphic_up_to_scalar(_single_spider(Dim("d"), 2), _x_cup(Dim("d")))


class TestCertificates:
    @pytest.mark.parametrize("name", ["fusion_chain_concrete", "state_copy", "hopf"])
    def test_the_derivation_replays_and_verifies(self, name: str) -> None:
        for diagram in EQUAL_PAIRS[name]():
            nf = normal_form(diagram)
            certificate = certify(nf.source, nf.results, label="nf")
            replayed = replay(certificate)
            assert replayed.reproduced, replayed.reason
            assert isomorphic(replayed.diagram, nf.diagram)
            assert verify(certificate, {}).verified

    def test_symbolic_derivation_verifies_at_a_concrete_tuple(self) -> None:
        nf = normal_form(_chain((2, 1, 0), Dim("d")))
        certificate = certify(nf.source, nf.results)
        for value in (2, 3):
            report = verify(certificate, {"d": value})
            assert report.verified, report.reason


class TestGuardAndCache:
    def test_guard_trip_reports_no_fixpoint(self) -> None:
        nf = normal_form(_chain((0, 1, 2), Dim("d")), guard=TerminationGuard(max_steps=1))
        assert nf.outcome.stop_reason is StopReason.STEP_LIMIT
        assert not nf.reached_fixpoint
        assert len(nf.results) == 1

    @pytest.mark.parametrize(
        "make_cache",
        [
            lambda: RewriteCache(),
            lambda: RewriteCache(
                incremental=IncrementalMatcher([rule.pattern for rule in normal_form_rules()])
            ),
        ],
    )
    def test_cached_and_uncached_agree(self, make_cache: Callable[[], RewriteCache]) -> None:
        for name in sorted(EQUAL_PAIRS):
            for diagram in EQUAL_PAIRS[name]():
                plain = normal_form(diagram)
                cached = normal_form(diagram, cache=make_cache())
                assert plain.key == cached.key
                assert plain.outcome.steps == cached.outcome.steps
                assert isomorphic(plain.diagram, cached.diagram)


class TestOnResultHook:
    def test_the_hook_sees_every_result_and_changes_nothing(self) -> None:
        diagram = _chain((0, 1, 2), Dim("d"))
        seen: list[RewriteResult] = []
        hooked = apply_until_fixpoint(diagram, normal_form_rules(), on_result=seen.append)
        plain = apply_until_fixpoint(diagram, normal_form_rules())
        assert tuple(result.step for result in seen) == hooked.steps == plain.steps
        assert hooked.stop_reason is plain.stop_reason
        assert isomorphic(hooked.diagram, plain.diagram)

    def test_a_non_callable_hook_is_refused(self) -> None:
        with pytest.raises(RewriteGrammarError):
            apply_until_fixpoint(
                _single_spider(Dim("d"), 1),
                normal_form_rules(),
                on_result=3,  # type: ignore[arg-type]
            )


class TestGrammar:
    @pytest.mark.parametrize("bad", [None, 3, "diagram"])
    def test_normal_form_rejects_a_non_diagram(self, bad: object) -> None:
        with pytest.raises(RewriteGrammarError):
            normal_form(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["rules", 3, (SPIDER_FUSION, 3)])
    def test_normal_form_rejects_bad_rules(self, bad: object) -> None:
        with pytest.raises(RewriteGrammarError):
            normal_form(_single_spider(Dim("d"), 1), rules=bad)  # type: ignore[arg-type]

    def test_normal_form_rejects_a_bad_guard(self) -> None:
        with pytest.raises(RewriteGrammarError):
            normal_form(_single_spider(Dim("d"), 1), guard=5)  # type: ignore[arg-type]

    def test_other_entry_points_reject_bad_types(self) -> None:
        nf = normal_form(_single_spider(Dim("d"), 1))
        with pytest.raises(RewriteGrammarError):
            comparison_view(nf)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            same_normal_form(nf, nf.diagram)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            isomorphic_up_to_scalar(nf.diagram, nf)  # type: ignore[arg-type]

    def test_normal_form_fields_are_validated(self) -> None:
        nf = normal_form(_single_spider(Dim("d"), 1))
        with pytest.raises(RewriteGrammarError):
            NormalForm(nf.source, nf.diagram, nf.outcome, nf.results, key=3)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            NormalForm(nf.source, nf.diagram, nf.outcome, [], nf.key)  # type: ignore[arg-type]
