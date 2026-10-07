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

"""Phase 14 done-when: equality saturation reaches diagrams greedy rewriting cannot, its
extractions certify and replay, and every saturated class member is oracle-equal to the input."""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.rewrite.egraph import EGraph, default_cost, saturation_rules, simplify
from archytaszx.rewrite.engine import (
    apply,
    apply_until_fixpoint,
    normal_form_rules,
    toward_normal_form,
)
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import Rule
from archytaszx.rewrite.rules_library import RULES
from archytaszx.semantics.certificate import certify, replay
from archytaszx.semantics.check import EqualityMode, compare, compare_symbolic
from archytaszx.semantics.contract_symbolic import contract_symbolic

D = Dim.symbol("d")


def out(node: NodeId, index: int = 0) -> PortRef:
    """Output port ``index`` of ``node``."""
    return PortRef(node, Direction.OUTPUT, index)


def inp(node: NodeId, index: int = 0) -> PortRef:
    """Input port ``index`` of ``node``."""
    return PortRef(node, Direction.INPUT, index)


def bialgebra_4() -> Diagram:
    """A Z state and a boundary leg into X_{2->1}, then Z_{1->2}, both outputs into one X
    effect."""
    g = Diagram()
    s = g.add_node(Z_SPIDER, input_dims=[], output_dims=[D])
    x = g.add_node(X_SPIDER, input_dims=[D, D], output_dims=[D])
    z = g.add_node(Z_SPIDER, input_dims=[D], output_dims=[D, D])
    e = g.add_node(X_SPIDER, input_dims=[D, D], output_dims=[])
    g.add_wire(out(s), inp(x, 1))
    g.add_wire(out(x), inp(z))
    g.add_wire(out(z, 0), inp(e, 0))
    g.add_wire(out(z, 1), inp(e, 1))
    g.set_boundary_inputs([inp(x, 0)])
    g.set_boundary_outputs([])
    return g


def bialgebra_5() -> Diagram:
    """Two Z_{1->2} into X_{2->1}, then Z_{1->2}, one of whose outputs feeds an X_{2->1}."""
    g = Diagram()
    za = g.add_node(Z_SPIDER, [D], [D, D])
    zb = g.add_node(Z_SPIDER, [D], [D, D])
    x = g.add_node(X_SPIDER, [D, D], [D])
    z = g.add_node(Z_SPIDER, [D], [D, D])
    xc = g.add_node(X_SPIDER, [D, D], [D])
    g.add_wire(out(za, 1), inp(x, 0))
    g.add_wire(out(zb, 1), inp(x, 1))
    g.add_wire(out(x), inp(z))
    g.add_wire(out(z, 0), inp(xc, 1))
    g.set_boundary_inputs([inp(za), inp(zb), inp(xc, 0)])
    g.set_boundary_outputs([out(za, 0), out(zb, 0), out(xc), out(z, 1)])
    return g


def bialgebra_6() -> Diagram:
    """Two Z_{1->2} into X_{2->1}, then Z_{1->2} into two X_{2->1}; outer spiders keep two
    boundary legs each."""
    g = Diagram()
    za = g.add_node(Z_SPIDER, [D], [D, D])
    zb = g.add_node(Z_SPIDER, [D], [D, D])
    x = g.add_node(X_SPIDER, [D, D], [D])
    z = g.add_node(Z_SPIDER, [D], [D, D])
    xc = g.add_node(X_SPIDER, [D, D], [D])
    xd = g.add_node(X_SPIDER, [D, D], [D])
    g.add_wire(out(za, 1), inp(x, 0))
    g.add_wire(out(zb, 1), inp(x, 1))
    g.add_wire(out(x), inp(z))
    g.add_wire(out(z, 0), inp(xc, 1))
    g.add_wire(out(z, 1), inp(xd, 1))
    g.set_boundary_inputs([inp(za), inp(zb), inp(xc, 0), inp(xd, 0)])
    g.set_boundary_outputs([out(za, 0), out(zb, 0), out(xc), out(xd)])
    return g


def state_copy_grows() -> Diagram:
    """A phaseless X state into Z_{1->3} with three boundary outputs."""
    g = Diagram()
    s = g.add_node(X_SPIDER, input_dims=[], output_dims=[D])
    z = g.add_node(Z_SPIDER, input_dims=[D], output_dims=[D, D, D])
    g.add_wire(out(s), inp(z))
    g.set_boundary_outputs([out(z, i) for i in range(3)])
    return g


REACHED: dict[str, tuple[Callable[[], Diagram], int, int]] = {
    "bialgebra_4": (bialgebra_4, 4, 3),
    "bialgebra_5": (bialgebra_5, 5, 4),
    "bialgebra_6": (bialgebra_6, 6, 4),
}
"""Name -> (builder, greedy node count, extracted node count); extraction leaves the input."""

EXAMPLES: dict[str, tuple[Callable[[], Diagram], int, int]] = {
    **REACHED,
    "state_copy_grows": (state_copy_grows, 3, 2),
}
"""Name -> (builder, greedy node count, best node count)."""

DIMENSIONS = (2, 3, 4, 5)


def assert_oracle_equal(a: Diagram, b: Diagram, dims: tuple[int, ...]) -> None:
    """``a`` and ``b`` agree exactly, scalar included, at every ``d`` in ``dims``."""
    for k in dims:
        result = compare(a, b, {"d": k}, mode=EqualityMode.EXACT)
        assert result.matched, f"d={k}: {result.reason}"


def saturated_class(diagram: Diagram) -> tuple[EGraph, tuple[Diagram, ...]]:
    """A saturated e-graph holding ``diagram`` and every member of its class."""
    graph = EGraph()
    root = graph.add(diagram)
    graph.saturate()
    return graph, tuple(graph.diagram(m) for m in graph.members(graph.class_of(root)))


def descend(diagram: Diagram, rules: tuple[Rule, ...]) -> Diagram:
    """Greedy descent: apply the first application that strictly lowers ``default_cost`` until
    none does."""
    current = diagram
    while True:
        cost = default_cost(current)
        for rule in rules:
            for match in rule.pattern.find_matches(current):
                if not match.all_side_conditions_passed:
                    continue
                result = apply(current, rule, match).diagram
                if default_cost(result) < cost:
                    current = result
                    break
            else:
                continue
            break
        else:
            return current


def neighbour_costs(diagram: Diagram, rules: tuple[Rule, ...]) -> list[tuple[int, int, int]]:
    """``default_cost`` of every one-step rewrite of ``diagram``."""
    return [
        default_cost(apply(diagram, rule, match).diagram)
        for rule in rules
        for match in rule.pattern.find_matches(diagram)
        if match.all_side_conditions_passed
    ]


def same_view(a: Diagram, b: Diagram) -> bool:
    """Whether ``a`` and ``b`` share a comparison view up to isomorphism."""
    return isomorphic(comparison_view(a), comparison_view(b))


class TestDoneWhen:
    """Saturation beats greedy, extractions replay, class members are oracle-equal."""

    @pytest.mark.parametrize("name", sorted(EXAMPLES))
    def test_extraction_beats_greedy(self, name: str) -> None:
        builder, greedy_count, best_count = EXAMPLES[name]
        diagram = builder()
        greedy = toward_normal_form(diagram)
        assert len(greedy.diagram.nodes) == greedy_count
        extraction, report = simplify(diagram)
        assert report.saturated
        assert not report.failed
        assert len(extraction.diagram.nodes) < len(greedy.diagram.nodes)
        assert len(extraction.diagram.nodes) == best_count

    @pytest.mark.parametrize("name", sorted(REACHED))
    def test_greedy_is_stuck_where_saturation_moves(self, name: str) -> None:
        builder, greedy_count, _ = REACHED[name]
        diagram = builder()
        greedy = toward_normal_form(diagram)
        assert greedy.steps == ()
        assert same_view(greedy.diagram, diagram)
        rules = saturation_rules()
        stuck = descend(diagram, rules)
        assert same_view(stuck, diagram)
        costs = neighbour_costs(stuck, rules)
        assert costs
        assert all(cost > default_cost(stuck) for cost in costs)
        extraction, _ = simplify(diagram)
        assert not same_view(extraction.diagram, diagram)
        assert extraction.cost < default_cost(stuck)  # type: ignore[operator]
        assert len(extraction.diagram.nodes) < greedy_count

    @pytest.mark.parametrize("name", sorted(REACHED))
    def test_extraction_path_climbs_before_it_descends(self, name: str) -> None:
        diagram = REACHED[name][0]()
        extraction, _ = simplify(diagram)
        names = [result.step.rule_name for result in extraction.results]
        assert "bialgebra" in names
        costs = [default_cost(result.diagram) for result in extraction.results]
        assert max(costs) > default_cost(diagram)
        assert costs[-1] == extraction.cost

    @pytest.mark.parametrize("name", sorted(EXAMPLES))
    def test_extraction_path_certifies_and_replays(self, name: str) -> None:
        diagram = EXAMPLES[name][0]()
        extraction, _ = simplify(diagram)
        certificate = certify(diagram, extraction.results)
        replayed = replay(certificate, rediscover=False)
        assert replayed.reproduced, replayed.reason
        assert isomorphic(comparison_view(replayed.diagram), comparison_view(extraction.diagram))

    @pytest.mark.parametrize("name", sorted(EXAMPLES))
    def test_extraction_is_oracle_equal(self, name: str) -> None:
        diagram = EXAMPLES[name][0]()
        extraction, _ = simplify(diagram)
        assert_oracle_equal(diagram, extraction.diagram, DIMENSIONS)

    @pytest.mark.parametrize("name", sorted(REACHED))
    def test_oracle_check_sees_the_scalar(self, name: str) -> None:
        diagram = REACHED[name][0]()
        extraction, _ = simplify(diagram)
        assert extraction.diagram.scalar != diagram.scalar
        unscaled = extraction.diagram.copy()
        unscaled.multiply_scalar(Scalar.dim_power(D, -1, 2))
        for k in DIMENSIONS:
            assert not compare(diagram, unscaled, {"d": k}).matched, f"d={k}"

    @pytest.mark.parametrize("name", sorted(EXAMPLES))
    def test_every_class_member_is_oracle_equal(self, name: str) -> None:
        diagram = EXAMPLES[name][0]()
        _, members = saturated_class(diagram)
        assert len(members) > 1
        assert sum(not same_view(member, diagram) for member in members) == len(members) - 1
        for member in members:
            assert_oracle_equal(diagram, member, DIMENSIONS)

    def test_saturation_beats_greedy_over_every_rule_but_bialgebra(self) -> None:
        without_bialgebra = tuple(rule for rule in RULES.values() if rule.name != "bialgebra")
        assert {rule.name for rule in normal_form_rules()} <= {r.name for r in without_bialgebra}
        for name, (builder, _, _) in sorted(REACHED.items()):
            diagram = builder()
            extracted = len(simplify(diagram)[0].diagram.nodes)
            for rules in (normal_form_rules(), without_bialgebra):
                greedy = apply_until_fixpoint(diagram, rules)
                assert extracted < len(greedy.diagram.nodes), (name, len(rules))

    def test_state_copy_grows_the_normal_form(self) -> None:
        diagram = state_copy_grows()
        greedy = toward_normal_form(diagram)
        assert [step.rule_name for step in greedy.steps] == ["state_copy"]
        assert len(greedy.diagram.nodes) > len(diagram.nodes)
        extraction, _ = simplify(diagram)
        assert extraction.results == ()
        assert same_view(extraction.diagram, diagram)


TURNS = (sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4))


def random_diagram(rng: random.Random, size: int) -> Diagram | None:
    """A random connected diagram of ``size`` Z, X and Fourier nodes over ``d``, a third of the
    spiders phased, or ``None``."""
    g = Diagram()
    ids: list[NodeId] = []
    for _ in range(size):
        kind = rng.choices(["Z", "X", "F"], weights=[4, 4, 1])[0]
        if kind == "F":
            ids.append(g.add_node(FOURIER_BOX, [D], [D]))
            continue
        n_in, n_out = rng.randint(0, 2), rng.randint(0, 2)
        if n_in + n_out == 0:
            n_out = 1
        phase = (
            PhaseVector(D, {1: Phase.turns(rng.choice(TURNS))}) if rng.random() < 1 / 3 else None
        )
        generator = Z_SPIDER if kind == "Z" else X_SPIDER
        ids.append(g.add_node(generator, [D] * n_in, [D] * n_out, phase=phase))
    outs = [out(i, k) for i in ids for k in range(g.nodes[i].num_outputs)]
    ins = [inp(i, k) for i in ids for k in range(g.nodes[i].num_inputs)]
    rng.shuffle(outs)
    rng.shuffle(ins)
    ceiling = min(len(outs), len(ins))
    wires = rng.randint(size - 1, ceiling) if ceiling >= size - 1 else size - 1
    used_out: list[PortRef] = []
    used_in: list[PortRef] = []
    for o, i in zip(outs, ins):
        if len(used_out) >= wires:
            break
        if o.node_id == i.node_id:
            continue
        g.add_wire(o, i)
        used_out.append(o)
        used_in.append(i)
    adjacent: dict[NodeId, set[NodeId]] = {i: set() for i in ids}
    for wire in g.wires:
        adjacent[wire.a.node_id].add(wire.b.node_id)
        adjacent[wire.b.node_id].add(wire.a.node_id)
    seen = {ids[0]}
    stack = [ids[0]]
    while stack:
        for neighbour in adjacent[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if len(seen) != len(ids):
        return None
    g.set_boundary_inputs([i for i in ins if i not in used_in])
    g.set_boundary_outputs([o for o in outs if o not in used_out])
    return g


ATTEMPTS = 20
SEEDS = 150


def seeded_diagram(seed: int) -> Diagram | None:
    """The first connected valid :func:`random_diagram` of seed ``seed``, or ``None``."""
    rng = random.Random(seed)
    for _ in range(ATTEMPTS):
        diagram = random_diagram(rng, rng.randint(3, 5))
        if diagram is not None and validate(diagram).is_valid:
            return diagram
    return None


@pytest.mark.slow
class TestRandomSweep:
    """Seeded random small diagrams: every class member is oracle-equal, extraction is the
    cheapest member."""

    @pytest.mark.parametrize("seed", range(SEEDS))
    def test_saturated_class_is_oracle_equal(self, seed: int) -> None:
        diagram = seeded_diagram(seed)
        if diagram is None:
            pytest.skip(f"no connected valid diagram in {ATTEMPTS} attempts")
        graph, members = saturated_class(diagram)
        for member in members:
            assert_oracle_equal(diagram, member, (2, 3, 4))
        extraction, _ = simplify(diagram)
        assert extraction.cost == min(default_cost(member) for member in members)
        found = graph.lookup(extraction.diagram)
        assert found is not None
        assert graph.equivalent(found, graph.roots[0])

    def test_sweep_exercises_non_trivial_classes(self) -> None:
        sizes = []
        improved = 0
        phased = 0
        for seed in range(SEEDS):
            diagram = seeded_diagram(seed)
            assert diagram is not None, seed
            phased += any(node.phase is not None for node in diagram.nodes.values())
            sizes.append(len(saturated_class(diagram)[1]))
            extraction, _ = simplify(diagram)
            improved += extraction.cost < default_cost(diagram)  # type: ignore[operator]
        assert sum(size > 1 for size in sizes) >= SEEDS // 2
        assert max(sizes) >= 10
        assert improved >= SEEDS // 2
        assert phased >= SEEDS // 3

    @pytest.mark.parametrize("name", sorted(EXAMPLES))
    def test_no_class_member_differs_with_d_formal(self, name: str) -> None:
        diagram = EXAMPLES[name][0]()
        _, members = saturated_class(diagram)
        reference = contract_symbolic(diagram)
        for member in members:
            result = compare_symbolic(reference, contract_symbolic(member))
            assert result.matched or result.reason.startswith("indeterminate"), result.reason
