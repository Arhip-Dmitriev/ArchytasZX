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

"""Phase 14 e-graph suite: hash-consing, union-find merges, determinism, stop reasons, limits,
failed applications, pruning, root paths, explanations, extraction, caching, continuation,
oracle soundness of every class member, and input validation."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.compare import canonical_key, isomorphic
from archytaszx.diagram.graph import Diagram
from archytaszx.rewrite.cache import RewriteCache
from archytaszx.rewrite.egraph import (
    DEFAULT_LIMITS,
    EClassId,
    EdgeKind,
    EGraph,
    ENodeId,
    Extraction,
    FailedApplication,
    SaturationLimits,
    SaturationReport,
    SaturationStop,
    default_cost,
    node_count_cost,
    saturation_rules,
    simplify,
)
from archytaszx.rewrite.match import FUSION_SIDE_CONDITIONS, FusionPattern
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import (
    BuildResult,
    Match,
    Quantifiers,
    RewriteError,
    RewriteGrammarError,
    Rule,
)
from archytaszx.rewrite.rules_library import RULES, SPIDER_FUSION
from archytaszx.semantics.certificate import certify, replay
from archytaszx.semantics.check import compare

from .helpers import build_ghz_with_copy
from .test_normal_form import _chain, _exact_rewrite, _single_spider
from .test_rules_library_phase11 import bialgebra_diagram, identity_chain, state_copy_diagram

SMALL = SaturationLimits(max_iterations=4, max_enodes=64, max_applications=500)


def _chain_d() -> Diagram:
    """Three phased Z spiders in a line over the symbolic dimension ``d``."""
    return _chain((0, 1, 2), Dim("d"))


def _saturated(*diagrams: Diagram, rules: tuple[Rule, ...] | None = None) -> EGraph:
    """A fresh e-graph holding ``diagrams`` as roots, saturated under ``SMALL``."""
    graph = EGraph()
    for diagram in diagrams:
        graph.add(diagram)
    assert graph.saturate(rules, limits=SMALL).saturated
    return graph


def _one_fusion() -> Diagram:
    """The chain after one spider fusion at its first match."""
    return _exact_rewrite(_chain_d(), "spider_fusion")


def _merged() -> tuple[EGraph, ENodeId, ENodeId]:
    """The chain and its one-fusion rewrite as two roots, saturated into one class."""
    graph = EGraph()
    first = graph.add(_chain_d())
    second = graph.add(_one_fusion())
    assert graph.saturate(limits=SMALL).saturated
    return graph, first, second


def _keys(graph: EGraph) -> tuple[str, ...]:
    """Every e-node's comparison-view key, by id."""
    return tuple(
        canonical_key(comparison_view(graph.diagram(ENodeId(i)))) for i in range(len(graph))
    )


def _signature(graph: EGraph) -> tuple[object, ...]:
    """Every edge's (parent, child, rule, kind), every e-node's view key, and every class."""
    edges = tuple((e.parent, e.child, e.rule_name, e.kind) for e in graph.edges)
    return edges, _keys(graph), graph.classes(), graph.roots


def _builder_that_raises(diagram: Diagram, match: Match) -> BuildResult:
    """A builder that always raises a RewriteError."""
    raise RewriteError("deliberate failure")


BOOM = Rule(
    name="boom",
    pattern=FusionPattern(),
    builder=_builder_that_raises,
    side_conditions=FUSION_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("m_a", "n_a", "m_b", "n_b"), dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)


class TestHashConsing:
    def test_isomorphic_diagrams_share_an_enode(self) -> None:
        graph = EGraph()
        first = graph.add(_chain((0, 1, 2), Dim("d")))
        second = graph.add(_chain((2, 0, 1), Dim("d")))
        assert first == second == 0
        assert len(graph) == 1
        assert graph.roots == (first,)

    def test_parameters_do_not_change_identity(self) -> None:
        graph = EGraph()
        plain = graph.add(_single_spider(Dim("d"), 2))
        bound = _single_spider(Dim("d"), 2)
        bound.bind_parameter("d", 3)
        assert graph.add(bound) == plain

    def test_distinct_diagrams_get_distinct_enodes_and_classes(self) -> None:
        graph = EGraph()
        a = graph.add(_single_spider(Dim("d"), 2))
        b = graph.add(_single_spider(Dim("d"), 3))
        assert (a, b) == (0, 1)
        assert graph.classes() == (0, 1)
        assert not graph.equivalent(a, b)

    def test_add_copies_its_input(self) -> None:
        graph = EGraph()
        diagram = _single_spider(Dim("d"), 2)
        enode = graph.add(diagram)
        diagram.multiply_scalar(Scalar.rational(5))
        assert graph.lookup(diagram) is None
        held = graph.diagram(enode)
        assert held is not graph.diagram(enode)
        assert isomorphic(held, _single_spider(Dim("d"), 2))

    def test_lookup(self) -> None:
        graph = _saturated(_chain_d())
        assert graph.lookup(_chain((1, 2, 0), Dim("d"))) == 0
        assert graph.lookup(_single_spider(Dim("d"), 7)) is None
        for enode in range(len(graph)):
            assert graph.lookup(graph.diagram(ENodeId(enode))) == enode


class TestUnionFind:
    def test_two_roots_merge_through_a_merge_edge(self) -> None:
        graph, first, second = _merged()
        assert graph.roots == (first, second)
        assert graph.equivalent(first, second)
        assert graph.classes() == (0,)
        assert graph.find(second) == graph.class_of(second) == 0
        merges = [edge for edge in graph.edges if edge.kind is EdgeKind.MERGE]
        assert merges
        for edge in merges:
            assert isomorphic(
                comparison_view(edge.result.diagram), comparison_view(graph.diagram(edge.child))
            )

    def test_merge_report_counts_the_union(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        graph.add(_one_fusion())
        report = graph.saturate(limits=SMALL)
        assert report.merges >= 1
        assert report.merges == sum(edge.kind is EdgeKind.MERGE for edge in graph.edges)

    def test_tree_edge_child_is_the_result_diagram(self) -> None:
        graph = _saturated(_chain_d())
        tree = [edge for edge in graph._edges if edge.kind is EdgeKind.TREE]
        assert len(tree) == len(graph) - 1
        for edge in tree:
            # identity of the stored representative; the public accessors return copies
            assert edge.result.diagram is graph._reps[edge.child]
            assert graph.equivalent(edge.parent, edge.child)

    def test_members_are_ascending_and_cover_the_class(self) -> None:
        graph, _, _ = _merged()
        assert graph.members(EClassId(0)) == tuple(ENodeId(i) for i in range(len(graph)))
        assert graph.members(EClassId(len(graph) - 1)) == graph.members(EClassId(0))


class TestDeterminism:
    @pytest.mark.parametrize("make", [lambda: (_chain_d(),), lambda: (_chain_d(), _one_fusion())])
    def test_two_fresh_runs_agree(self, make: Callable[[], tuple[Diagram, ...]]) -> None:
        assert _signature(_saturated(*make())) == _signature(_saturated(*make()))

    def test_reports_agree(self) -> None:
        reports = []
        for _ in range(2):
            graph = EGraph()
            graph.add(_chain_d())
            reports.append(graph.saturate(limits=SMALL))
        assert reports[0] == reports[1]


class TestStopReasons:
    def test_saturated(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        report = graph.saturate(limits=SMALL)
        assert report.stop_reason is SaturationStop.SATURATED
        assert report.saturated
        assert report.enodes_added == len(graph) - 1 == 3

    def test_iteration_limit(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        report = graph.saturate(limits=SaturationLimits(max_iterations=1))
        assert report.stop_reason is SaturationStop.ITERATION_LIMIT
        assert not report.saturated
        assert report.iterations == 1

    def test_zero_iterations(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        report = graph.saturate(limits=SaturationLimits(max_iterations=0))
        assert report.stop_reason is SaturationStop.ITERATION_LIMIT
        assert (report.iterations, report.applications, len(graph)) == (0, 0, 1)

    def test_enode_limit(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        report = graph.saturate(limits=SaturationLimits(max_enodes=2))
        assert report.stop_reason is SaturationStop.ENODE_LIMIT
        assert len(graph) == 2
        assert report.enodes_added == 1

    def test_application_limit(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        report = graph.saturate(limits=SaturationLimits(max_applications=1))
        assert report.stop_reason is SaturationStop.APPLICATION_LIMIT
        assert report.applications == 1

    @pytest.mark.parametrize(
        "limits",
        [
            SaturationLimits(max_iterations=1),
            SaturationLimits(max_enodes=2),
            SaturationLimits(max_applications=1),
        ],
    )
    def test_saturate_again_continues_to_the_same_closure(self, limits: SaturationLimits) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        assert not graph.saturate(limits=limits).saturated
        assert graph.saturate(limits=SMALL).saturated
        reference = _saturated(_chain_d())
        assert len(graph) == len(reference)
        assert graph.classes() == reference.classes()
        assert set(_keys(graph)) == set(_keys(reference))

    def test_resaturation_does_nothing(self) -> None:
        graph = _saturated(_chain_d())
        before = _signature(graph)
        report = graph.saturate(limits=SMALL)
        assert report.saturated
        assert (report.iterations, report.applications, report.enodes_added) == (0, 0, 0)
        assert _signature(graph) == before

    def test_a_rule_sharing_a_name_still_re_expands(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        assert graph.saturate((SPIDER_FUSION,), limits=SMALL).saturated
        impostor = Rule(
            name=SPIDER_FUSION.name,
            pattern=FusionPattern(),
            builder=_builder_that_raises,
            side_conditions=FUSION_SIDE_CONDITIONS,
            quantifiers=BOOM.quantifiers,
            scalar_introduced=Scalar.one(),
        )
        report = graph.saturate((impostor,), limits=SMALL)
        assert report.applications > 0
        assert report.failed

    def test_a_different_rule_tuple_re_expands(self) -> None:
        graph = _saturated(_chain_d())
        report = graph.saturate((SPIDER_FUSION,), limits=SMALL)
        assert report.saturated
        assert report.applications > 0
        assert report.enodes_added == 0


class TestLimitsValidation:
    def test_defaults(self) -> None:
        assert DEFAULT_LIMITS == SaturationLimits(6, 512, 20_000, 4)

    @pytest.mark.parametrize(
        "name", ["max_iterations", "max_enodes", "max_applications", "node_margin"]
    )
    @pytest.mark.parametrize("bad", [True, False, -1, 1.0, "3", None])
    def test_bad_field_is_refused(self, name: str, bad: object) -> None:
        with pytest.raises(RewriteGrammarError, match=name):
            SaturationLimits(**{name: bad})  # type: ignore[arg-type]

    def test_zero_is_allowed(self) -> None:
        assert SaturationLimits(0, 0, 0, 0).node_margin == 0


class TestFailedApplicationsAndPruning:
    def test_a_raising_builder_is_recorded_not_raised(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        report = graph.saturate((BOOM, SPIDER_FUSION), limits=SMALL)
        assert report.saturated
        assert report.failed
        assert all(isinstance(item, FailedApplication) for item in report.failed)
        assert all(item.rule_name == "boom" for item in report.failed)
        assert all("deliberate failure" in item.message for item in report.failed)
        assert {item.enode for item in report.failed} <= set(range(len(graph)))
        assert len(graph) == len(_saturated(_chain_d())) == 4
        fusions = report.applications - len(report.failed)
        assert fusions == report.enodes_added + report.merges + report.redundant + report.pruned

    def test_bialgebra_growth_is_pruned_at_zero_margin(self) -> None:
        graph = EGraph()
        graph.add(bialgebra_diagram(2))
        report = graph.saturate(limits=SaturationLimits(node_margin=0))
        assert report.saturated
        assert report.pruned >= 1
        assert len(graph) == 1
        assert graph.edges == ()

    def test_a_larger_margin_re_expands_a_pruned_enode(self) -> None:
        graph = EGraph()
        graph.add(bialgebra_diagram(2))
        assert graph.saturate(limits=SaturationLimits(node_margin=0)).pruned >= 1
        assert len(graph) == 1
        report = graph.saturate(limits=SaturationLimits(max_iterations=1, node_margin=2))
        assert report.applications > 0
        assert len(graph) == 2

    def test_bialgebra_growth_is_kept_with_a_margin(self) -> None:
        graph = EGraph()
        graph.add(bialgebra_diagram(2))
        report = graph.saturate(limits=SaturationLimits(max_iterations=1, node_margin=2))
        assert report.pruned == 0
        assert len(graph) == 2


class TestCertifiedPaths:
    @pytest.mark.parametrize(
        "make", [_chain_d, lambda: identity_chain(3), lambda: state_copy_diagram(3, 2)]
    )
    def test_path_from_root_replays(self, make: Callable[[], Diagram]) -> None:
        graph = _saturated(make())
        for enode in range(len(graph)):
            root, results = graph.path_from_root(ENodeId(enode))
            assert root in graph.roots
            certificate = certify(graph.diagram(root), results, label=f"enode {enode}")
            replayed = replay(certificate, rediscover=False)
            assert replayed.reproduced, replayed.reason
            assert isomorphic(
                comparison_view(replayed.diagram), comparison_view(graph.diagram(ENodeId(enode)))
            )

    def test_a_root_has_an_empty_path(self) -> None:
        graph, first, second = _merged()
        assert graph.path_from_root(second) == (second, ())
        assert graph.path_from_root(first) == (first, ())

    def test_explain_same_node_and_unrelated_nodes(self) -> None:
        graph = EGraph()
        a = graph.add(_single_spider(Dim("d"), 2))
        b = graph.add(_single_spider(Dim("d"), 3))
        assert graph.explain(a, a) == ()
        assert graph.explain(a, b) is None

    def test_explain_edges_certify_and_land_on_the_child(self) -> None:
        graph, first, second = _merged()
        for a in range(len(graph)):
            for b in range(len(graph)):
                path = graph.explain(ENodeId(a), ENodeId(b))
                assert path is not None
                assert (len(path) == 0) == (a == b)
                for edge in path:
                    parent = graph.diagram(edge.parent)
                    certificate = certify(parent, [edge.result], label=edge.rule_name)
                    replayed = replay(certificate, rediscover=False)
                    assert replayed.reproduced, replayed.reason
                    assert isomorphic(
                        comparison_view(certificate.final),
                        comparison_view(graph.diagram(edge.child)),
                    )
        path = graph.explain(first, second)
        again = graph.explain(first, second)
        assert path is not None and again is not None and len(path) >= 1
        assert [(e.parent, e.child, e.rule_name, e.kind) for e in path] == [
            (e.parent, e.child, e.rule_name, e.kind) for e in again
        ]

    def test_explain_path_is_connected(self) -> None:
        graph, first, second = _merged()
        path = graph.explain(first, second)
        assert path is not None
        current = first
        for edge in path:
            assert current in (edge.parent, edge.child)
            current = edge.child if current == edge.parent else edge.parent
        assert current == second


class TestExtraction:
    def test_default_cost_picks_the_fully_fused_spider(self) -> None:
        graph = _saturated(_chain_d())
        extraction = graph.extract(EClassId(0))
        assert isinstance(extraction, Extraction)
        assert len(extraction.diagram.nodes) == 1
        assert extraction.cost == default_cost(extraction.diagram)
        costs = [default_cost(graph.diagram(m)) for m in graph.members(EClassId(0))]
        assert extraction.cost == min(costs)
        assert extraction.enode == costs.index(min(costs))
        assert extraction.eclass == 0

    def test_extraction_results_replay(self) -> None:
        graph = _saturated(_chain_d())
        extraction = graph.extract(EClassId(0))
        certificate = certify(graph.diagram(extraction.root), extraction.results)
        replayed = replay(certificate, rediscover=False)
        assert replayed.reproduced, replayed.reason
        assert isomorphic(comparison_view(replayed.diagram), comparison_view(extraction.diagram))

    def test_ties_go_to_the_lowest_enode_id(self) -> None:
        graph = _saturated(_chain_d())
        assert graph.extract(EClassId(0), lambda _d: 0).enode == 0
        members = graph.members(EClassId(0))
        two_nodes = [m for m in members if node_count_cost(graph.diagram(m)) == 2]
        assert len(two_nodes) >= 2
        extraction = graph.extract(EClassId(0), lambda d: abs(node_count_cost(d) - 2))
        assert extraction.enode == min(two_nodes)

    def test_custom_costs(self) -> None:
        graph = _saturated(_chain_d())
        assert graph.extract(EClassId(0), node_count_cost).cost == 1
        largest = graph.extract(EClassId(0), lambda d: -node_count_cost(d))
        assert largest.cost == -3
        assert largest.enode == 0
        assert graph.extract(EClassId(0), lambda d: (float(len(d.wires)), 1)).cost == (0.0, 1)

    @pytest.mark.parametrize(
        "cost",
        [
            lambda _d: "cheap",
            lambda _d: True,
            lambda _d: None,
            lambda _d: (1, "a"),
            lambda _d: ((1,),),
            lambda _d: (1, False),
            lambda d: 1 if len(d.nodes) == 3 else (1,),
            lambda _d: float("nan"),
            lambda d: float("nan") if len(d.nodes) == 3 else 1.0,
            lambda d: (1, float("nan")) if len(d.nodes) == 1 else (1, 0.0),
        ],
    )
    def test_invalid_cost_values_raise(self, cost: Callable[[Diagram], object]) -> None:
        graph = _saturated(_chain_d())
        with pytest.raises(RewriteGrammarError):
            graph.extract(EClassId(0), cost)  # type: ignore[arg-type]

    def test_a_non_callable_cost_is_refused(self) -> None:
        graph = _saturated(_chain_d())
        with pytest.raises(RewriteGrammarError):
            graph.extract(EClassId(0), 3)  # type: ignore[arg-type]

    def test_extract_on_a_stale_class_id(self) -> None:
        graph = EGraph()
        graph.add(_chain_d())
        second = graph.add(_one_fusion())
        stale = graph.class_of(second)
        assert stale == 1 and second == 1
        assert graph.saturate(limits=SMALL).saturated
        assert graph.find(stale) == 0
        assert graph.extract(stale).eclass == 0
        assert graph.extract(stale).enode == graph.extract(EClassId(0)).enode

    def test_cost_receives_a_copy(self) -> None:
        graph = _saturated(_chain_d())

        def mutating(diagram: Diagram) -> int:
            diagram.multiply_scalar(Scalar.rational(3))
            return len(diagram.nodes)

        before = _signature(graph)
        graph.extract(EClassId(0), mutating)
        assert _signature(graph) == before


class TestNoAliasing:
    def test_mutating_handed_out_diagrams_leaves_the_graph_intact(self) -> None:
        graph = _saturated(_chain_d())
        before = _signature(graph)
        scalars = [graph.diagram(ENodeId(i)).scalar for i in range(len(graph))]
        extraction = graph.extract(EClassId(0))
        handed_out = [extraction.diagram, *(result.diagram for result in extraction.results)]
        handed_out += [edge.result.diagram for edge in graph.edges]
        path = graph.explain(ENodeId(0), ENodeId(len(graph) - 1))
        assert path
        handed_out += [edge.result.diagram for edge in path]
        handed_out += [r.diagram for r in graph.path_from_root(ENodeId(len(graph) - 1))[1]]
        for diagram in handed_out:
            diagram.multiply_scalar(Scalar.rational(7))
        assert _signature(graph) == before
        assert [graph.diagram(ENodeId(i)).scalar for i in range(len(graph))] == scalars
        assert all(edge.result.diagram is graph._reps[edge.child] for edge in graph._edges)
        replayed = replay(
            certify(graph.diagram(ENodeId(0)), graph.extract(EClassId(0)).results),
            rediscover=False,
        )
        assert replayed.reproduced, replayed.reason


class TestCache:
    @pytest.mark.parametrize("make", [lambda: (_chain_d(),), lambda: (_chain_d(), _one_fusion())])
    def test_cached_and_uncached_agree(self, make: Callable[[], tuple[Diagram, ...]]) -> None:
        plain, cached = EGraph(), EGraph()
        for diagram in make():
            plain.add(diagram)
        for diagram in make():
            cached.add(diagram)
        plain_report = plain.saturate(limits=SMALL)
        cache = RewriteCache()
        cached_report = cached.saturate(limits=SMALL, cache=cache)
        assert plain_report == cached_report
        assert _signature(plain) == _signature(cached)
        assert cached.saturate((SPIDER_FUSION,), limits=SMALL, cache=cache).saturated


class TestSimplify:
    def test_simplify_extracts_the_cheapest_member(self) -> None:
        extraction, report = simplify(_chain_d(), limits=SMALL)
        assert report.saturated
        assert isinstance(report, SaturationReport)
        assert len(extraction.diagram.nodes) == 1
        assert extraction.root == 0
        assert extraction.eclass == 0

    def test_simplify_with_custom_rules_and_cost(self) -> None:
        extraction, report = simplify(_chain_d(), (), cost=node_count_cost, limits=SMALL)
        assert report.applications == 0
        assert extraction.cost == 3
        assert extraction.results == ()

    def test_simplify_matches_a_manual_run(self) -> None:
        extraction, _ = simplify(build_ghz_with_copy(Dim("d"))[0], limits=SMALL)
        graph = _saturated(build_ghz_with_copy(Dim("d"))[0])
        manual = graph.extract(EClassId(0))
        assert extraction.enode == manual.enode
        assert extraction.cost == manual.cost

    def test_saturation_rules_are_the_registry(self) -> None:
        assert saturation_rules() == tuple(RULES.values())


class TestClassMembersAreOracleEqual:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: (_chain_d(),),
            lambda: (_chain_d(), _one_fusion()),
            lambda: (build_ghz_with_copy(Dim("d"))[0],),
        ],
    )
    @pytest.mark.parametrize("value", [2, 3])
    def test_every_member_matches_the_root(
        self, make: Callable[[], tuple[Diagram, ...]], value: int
    ) -> None:
        graph = _saturated(*make())
        root = graph.diagram(graph.roots[0])
        for eclass in graph.classes():
            for member in graph.members(eclass):
                assert compare(root, graph.diagram(member), {"d": value}).matched


class TestGrammar:
    @pytest.mark.parametrize("bad", [None, 3, "diagram", comparison_view])
    def test_add_and_lookup_reject_a_non_diagram(self, bad: object) -> None:
        graph = EGraph()
        with pytest.raises(RewriteGrammarError):
            graph.add(bad)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            graph.lookup(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [True, False, -1, 99, 1.0, "0", None])
    def test_id_accessors_reject_bad_ids(self, bad: object) -> None:
        graph = EGraph()
        graph.add(_single_spider(Dim("d"), 2))
        accessors: list[Callable[[object], object]] = [
            graph.find,  # type: ignore[list-item]
            graph.class_of,  # type: ignore[list-item]
            graph.members,  # type: ignore[list-item]
            graph.diagram,  # type: ignore[list-item]
            graph.path_from_root,  # type: ignore[list-item]
            graph.extract,  # type: ignore[list-item]
            lambda x: graph.equivalent(x, ENodeId(0)),  # type: ignore[arg-type]
            lambda x: graph.equivalent(ENodeId(0), x),  # type: ignore[arg-type]
            lambda x: graph.explain(x, ENodeId(0)),  # type: ignore[arg-type]
            lambda x: graph.explain(ENodeId(0), x),  # type: ignore[arg-type]
        ]
        for accessor in accessors:
            with pytest.raises(RewriteGrammarError):
                accessor(bad)

    def test_unknown_id_message_names_the_id(self) -> None:
        graph = EGraph()
        with pytest.raises(RewriteGrammarError, match="42"):
            graph.find(42)

    @pytest.mark.parametrize("bad", ["rules", b"rules", 3, (SPIDER_FUSION, 3), {SPIDER_FUSION}])
    def test_saturate_rejects_bad_rules(self, bad: object) -> None:
        graph = EGraph()
        graph.add(_single_spider(Dim("d"), 2))
        with pytest.raises(RewriteGrammarError):
            graph.saturate(bad)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            simplify(_single_spider(Dim("d"), 2), bad)  # type: ignore[arg-type]

    def test_saturate_rejects_bad_limits_and_cache(self) -> None:
        graph = EGraph()
        graph.add(_single_spider(Dim("d"), 2))
        with pytest.raises(RewriteGrammarError):
            graph.saturate(limits=6)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            graph.saturate(cache={})  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [None, 3, "diagram"])
    def test_simplify_rejects_a_non_diagram(self, bad: object) -> None:
        with pytest.raises(RewriteGrammarError):
            simplify(bad)  # type: ignore[arg-type]

    def test_simplify_rejects_a_non_callable_cost(self) -> None:
        with pytest.raises(RewriteGrammarError):
            simplify(_single_spider(Dim("d"), 2), cost=3)  # type: ignore[arg-type]

    def test_costs_reject_a_non_diagram(self) -> None:
        with pytest.raises(RewriteGrammarError):
            default_cost(3)  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            node_count_cost("x")  # type: ignore[arg-type]

    def test_value_objects_validate_their_fields(self) -> None:
        with pytest.raises(RewriteGrammarError):
            FailedApplication(True, "r", "m")  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            FailedApplication(ENodeId(0), 3, "m")  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            SaturationReport("saturated", 0, 0, 0, 0, 0, 0, ())  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            SaturationReport(SaturationStop.SATURATED, 0, 0, 0, 0, 0, 0, [])  # type: ignore[arg-type]
        with pytest.raises(RewriteGrammarError):
            SaturationReport(SaturationStop.SATURATED, True, 0, 0, 0, 0, 0, ())
