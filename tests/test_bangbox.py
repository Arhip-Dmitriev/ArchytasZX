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

"""Unit tests for archytaszx.diagram.bangbox: Mult, BangBox construction, and copy/kill/merge.

Instantiate itself (the GHZ-family and nested-two-index mechanics) is exercised end to
end, against the numeric oracle, by tests/test_phase7_oracle.py; this module covers the
pieces that oracle suite does not: the Mult expression type, BangBox's own construction
invariants, and copy/merge, which FULL_PLAN.md's Phase 7 requires but no oracle scenario
exercises.
"""

from __future__ import annotations

import numpy as np
import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.bangbox import (
    BangBox,
    BangBoxDomainError,
    BangBoxGrammarError,
    Mult,
    abstract_subgraph_count,
    free_mult_symbols,
    instantiate_symbol,
    kill,
    merge,
    peel_one,
)
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import IssueKind, validate
from archytaszx.semantics.check import score


class TestMult:
    def test_concrete_and_symbol_construction(self) -> None:
        assert Mult.concrete(3).to_int() == 3
        assert Mult.symbol("n").is_bare_symbol
        assert Mult.symbol("n").bare_symbol_name() == "n"

    def test_negative_concrete_is_rejected(self) -> None:
        with pytest.raises(BangBoxDomainError):
            Mult.concrete(-1)

    def test_sum_and_product_normalize_commutatively(self) -> None:
        n1 = Mult.symbol("n1")
        n2 = Mult.symbol("n2")
        assert n1 + n2 == n2 + n1
        assert n1 * n2 == n2 * n1
        assert not (n1 + n2).is_bare_symbol
        assert (n1 + n2).free_symbols == frozenset({"n1", "n2"})

    def test_substitute_is_partial_and_returns_a_new_mult(self) -> None:
        combined = Mult.symbol("n1") + Mult.symbol("n2")
        partial = combined.substitute({"n1": 3})
        assert not partial.is_concrete
        assert partial.free_symbols == frozenset({"n2"})
        full = partial.substitute({"n2": 4})
        assert full.is_concrete
        assert full.to_int() == 7

    def test_substitute_rejects_negative(self) -> None:
        with pytest.raises(BangBoxDomainError):
            Mult.symbol("n").substitute({"n": -1})

    def test_abstract_then_substitute_round_trips(self) -> None:
        symbol, binding = Mult.concrete(5).abstract(stem="n")
        assert dict(binding) == {"n": 5}
        assert symbol.substitute(binding).to_int() == 5

    def test_abstract_avoids_taken_names(self) -> None:
        symbol, binding = Mult.concrete(2).abstract(avoid={"n", "n1"}, stem="n")
        assert symbol.bare_symbol_name() == "n2"
        assert dict(binding) == {"n2": 2}

    def test_is_concrete_and_zero_is_legal(self) -> None:
        assert Mult.concrete(0).is_concrete
        assert Mult.concrete(0).to_int() == 0


class TestBangBoxConstruction:
    def _diagram_with_two_nodes(self) -> tuple[Diagram, NodeId, NodeId]:
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        b = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        return diagram, a, b

    def test_empty_scope_is_rejected_directly(self) -> None:
        with pytest.raises(BangBoxGrammarError):
            BangBox(id=0, multiplicity=Mult.concrete(1))  # type: ignore[arg-type]

    def test_both_scopes_non_empty_is_rejected(self) -> None:
        _diagram, a, _b = self._diagram_with_two_nodes()
        ref = PortRef(a, Direction.OUTPUT, 0)
        with pytest.raises(BangBoxGrammarError):
            BangBox(
                id=0,  # type: ignore[arg-type]
                multiplicity=Mult.concrete(1),
                node_scope=frozenset({a}),
                port_scope=frozenset({ref}),
            )

    def test_add_bang_box_via_diagram_round_trips(self) -> None:
        diagram, a, _b = self._diagram_with_two_nodes()
        box_id = diagram.add_bang_box(Mult.symbol("k"), node_scope=frozenset({a}))
        box = diagram.bang_boxes[box_id]
        assert box.node_scope == frozenset({a})
        assert box.is_node_scope


class TestCopyBox:
    def test_copy_produces_an_independent_fresh_symbol(self) -> None:
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 0)])
        diagram, box_id, symbol = abstract_subgraph_count(diagram, frozenset({a}), 1, stem="m")

        from archytaszx.diagram.bangbox import copy_box

        copied_diagram, new_box_id = copy_box(diagram, box_id)
        original_box = copied_diagram.bang_boxes[box_id]
        new_box = copied_diagram.bang_boxes[new_box_id]

        assert new_box_id != box_id
        assert new_box.multiplicity != original_box.multiplicity
        assert new_box.multiplicity.is_bare_symbol
        assert len(new_box.node_scope) == 1
        assert next(iter(new_box.node_scope)) != a
        # The original box and its node are untouched.
        assert original_box.node_scope == frozenset({a})
        assert a in copied_diagram.nodes
        assert symbol == original_box.multiplicity

    def test_copy_of_port_scope_box_is_refused(self) -> None:
        from archytaszx.diagram.bangbox import abstract_port_count, copy_box

        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        ref = PortRef(a, Direction.OUTPUT, 0)
        diagram.set_boundary_outputs([ref])
        diagram, box_id, _n = abstract_port_count(diagram, ref, 1, stem="n")
        with pytest.raises(BangBoxGrammarError):
            copy_box(diagram, box_id)


class TestMerge:
    def _two_sibling_boxes(self, shared_symbol: bool) -> tuple[Diagram, BangBoxId, BangBoxId]:
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        b = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs(
            [PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.OUTPUT, 0)]
        )
        symbol = Mult.symbol("m") if shared_symbol else None
        box_a = diagram.add_bang_box(symbol or Mult.symbol("m1"), node_scope=frozenset({a}))
        box_b = diagram.add_bang_box(symbol or Mult.symbol("m2"), node_scope=frozenset({b}))
        return diagram, box_a, box_b

    def test_merge_with_identical_multiplicity_succeeds(self) -> None:
        diagram, box_a, box_b = self._two_sibling_boxes(shared_symbol=True)
        merged = merge(diagram, box_a, box_b)
        assert set(merged.bang_boxes) == {2}
        (box,) = merged.bang_boxes.values()
        assert box.node_scope == frozenset({0, 1})
        assert box.multiplicity == Mult.symbol("m")

    def test_merge_with_different_multiplicity_requires_assume_equal(self) -> None:
        diagram, box_a, box_b = self._two_sibling_boxes(shared_symbol=False)
        with pytest.raises(BangBoxGrammarError):
            merge(diagram, box_a, box_b)
        merged = merge(diagram, box_a, box_b, assume_equal=True)
        assert len(merged.bang_boxes) == 1

    def test_merge_of_overlapping_scopes_is_rejected(self) -> None:
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 0)])
        box_a = diagram.add_bang_box(Mult.symbol("m1"), node_scope=frozenset({a}))
        box_b = diagram.add_bang_box(Mult.symbol("m2"), node_scope=frozenset({a}))
        with pytest.raises(BangBoxGrammarError):
            merge(diagram, box_a, box_b, assume_equal=True)

    def test_merge_of_non_siblings_is_rejected(self) -> None:
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        b = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs(
            [PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.OUTPUT, 0)]
        )
        parent_id = diagram.add_bang_box(Mult.symbol("p"), node_scope=frozenset({a, b}))
        box_a = diagram.add_bang_box(Mult.symbol("m1"), node_scope=frozenset({a}), parent=parent_id)
        box_b = diagram.add_bang_box(Mult.symbol("m2"), node_scope=frozenset({b}))
        with pytest.raises(BangBoxGrammarError):
            merge(diagram, box_a, box_b, assume_equal=True)


class TestKill:
    def test_kill_by_id_matches_instantiate_at_zero(self) -> None:
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 0)])
        diagram, box_id, _m = abstract_subgraph_count(diagram, frozenset({a}), 1, stem="m")
        killed = kill(diagram, box_id)
        assert killed.bang_boxes == {}
        assert killed.nodes == {}

    def test_kill_of_unknown_box_raises(self) -> None:
        diagram = Diagram()
        with pytest.raises(BangBoxGrammarError):
            kill(diagram, BangBoxId(999))


class TestCompoundMultiplicityInstantiation:
    """Phase 7's multiplicity arithmetic reaches instantiation, not just Mult."""

    @staticmethod
    def _family(multiplicity: Mult) -> Diagram:
        d = Dim(2)
        diagram = Diagram()
        node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs([PortRef(node, Direction.OUTPUT, 0)])
        diagram.add_bang_box(multiplicity, node_scope=frozenset({node}))
        return diagram

    @pytest.mark.parametrize(
        "multiplicity,assignment,copies",
        [
            (Mult.symbol("k") * 2, {"k": 3}, 6),
            (Mult.symbol("k") + 1, {"k": 2}, 3),
            (Mult.symbol("k") * 2, {"k": 0}, 0),
            (Mult.symbol("k") + 1, {"k": 0}, 1),
            (Mult.symbol("k1") + Mult.symbol("k2"), {"k1": 2, "k2": 3}, 5),
            (Mult.symbol("k1") * Mult.symbol("k2"), {"k1": 2, "k2": 3}, 6),
        ],
    )
    def test_compound_multiplicity_expands_to_the_arithmetic_value(
        self, multiplicity: Mult, assignment: dict[str, int], copies: int
    ) -> None:
        tensor = score(self._family(multiplicity), assignment).tensor
        assert np.allclose(tensor, np.ones((2,) * copies))

    def test_a_partially_supplied_compound_keeps_its_box(self) -> None:
        diagram = self._family(Mult.symbol("k1") + Mult.symbol("k2"))
        partial = instantiate_symbol(diagram, "k1", 2)
        assert free_mult_symbols(partial) == frozenset({"k2"})
        assert validate(partial).is_valid
        assert np.allclose(score(partial, {"k2": 1}).tensor, np.ones((2,) * 3))

    def test_an_unmentioned_symbol_is_still_refused(self) -> None:
        with pytest.raises(BangBoxGrammarError):
            instantiate_symbol(self._family(Mult.symbol("k") * 2), "nope", 1)


class TestBoundaryOrderIsContinuousAtOne:
    """Instantiating at 1 splices the boundary the way every larger count does."""

    @staticmethod
    def _non_contiguous_crossings() -> Diagram:
        """A boxed two-leg spider whose boundary is split by an unboxed spider's leg."""
        d = Dim(2)
        diagram = Diagram()
        boxed = diagram.add_node(
            Z_SPIDER, input_dims=[], output_dims=[d, d], phase=PhaseVector(d, {})
        )
        other = diagram.add_node(X_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs(
            [
                PortRef(boxed, Direction.OUTPUT, 0),
                PortRef(other, Direction.OUTPUT, 0),
                PortRef(boxed, Direction.OUTPUT, 1),
            ]
        )
        diagram, _box, _m = abstract_subgraph_count(diagram, frozenset({boxed}), 1, stem="m")
        return diagram

    @staticmethod
    def _generator_order(diagram: Diagram) -> list[str]:
        return [diagram.nodes[ref.node_id].generator_type.name for ref in diagram.boundary_outputs]

    def test_the_boxed_block_is_contiguous_at_every_count(self) -> None:
        for k in (1, 2, 3):
            order = self._generator_order(
                instantiate_symbol(self._non_contiguous_crossings(), "m", k)
            )
            boxed_positions = [i for i, name in enumerate(order) if name == "Z"]
            assert boxed_positions == list(range(len(boxed_positions))), (k, order)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_peel_then_instantiate_matches_instantiate_at_one_more(self, k: int) -> None:
        from archytaszx.semantics.induction import successor_diagram

        family = self._non_contiguous_crossings()
        direct = score(instantiate_symbol(family, "m", k + 1), {}).tensor
        successor = successor_diagram(self._non_contiguous_crossings(), "m")
        step = min(free_mult_symbols(successor))
        peeled = peel_one(successor, min(successor.bang_boxes)).diagram
        assert np.allclose(direct, score(instantiate_symbol(peeled, step, k), {}).tensor)


def _leaf(diagram: Diagram, outputs: int = 1, generator: GeneratorType = Z_SPIDER) -> NodeId:
    d = Dim(2)
    return diagram.add_node(
        generator, input_dims=[], output_dims=[d] * outputs, phase=PhaseVector(d)
    )


def _out(node: NodeId, index: int) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def _generators(diagram: Diagram) -> list[str]:
    return [diagram.nodes[ref.node_id].generator_type.name for ref in diagram.boundary_outputs]


class TestInstantiationEdgeCases:
    def test_killing_purges_compound_and_deep_descendant_bindings(self) -> None:
        diagram = Diagram()
        a, b, c = _leaf(diagram), _leaf(diagram), _leaf(diagram)
        diagram.set_boundary_outputs([_out(a, 0), _out(b, 0), _out(c, 0)])
        outer = diagram.add_bang_box(Mult("a") + Mult("b"), node_scope=frozenset({a, b, c}))
        middle = diagram.add_bang_box(Mult("k") * 2, node_scope=frozenset({a, b}), parent=outer)
        diagram.add_bang_box(Mult("j") + 2, node_scope=frozenset({a}), parent=middle)
        diagram.bind_parameter("j", 1)
        diagram.bind_parameter("k", 1)
        killed = instantiate_symbol(instantiate_symbol(diagram, "a", 0), "b", 0)
        assert dict(killed.parameters) == {}
        assert not killed.nodes
        assert dict(kill(diagram, outer).parameters) == {}
        assert score(killed, {}).shape == ()

    def test_a_port_scope_box_over_two_legs_of_one_node_grows_both(self) -> None:
        diagram = Diagram()
        z = _leaf(diagram, 3)
        diagram.set_boundary_outputs([_out(z, 0), _out(z, 1), _out(z, 2)])
        diagram.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0), _out(z, 2)}))
        for n, legs in ((0, 1), (2, 5), (3, 7)):
            grown = instantiate_symbol(diagram, "n", n)
            (node,) = grown.nodes.values()
            assert len(node.outputs) == legs
            assert grown.boundary_outputs == tuple(_out(node.id, i) for i in range(legs))
            assert validate(grown).is_valid

    def test_killing_every_child_of_a_box_at_zero_empties_it_cleanly(self) -> None:
        diagram = Diagram()
        a, b, c = _leaf(diagram), _leaf(diagram, generator=X_SPIDER), _leaf(diagram)
        diagram.set_boundary_outputs([_out(a, 0), _out(c, 0), _out(b, 0)])
        outer = diagram.add_bang_box(Mult("k") * 2, node_scope=frozenset({a, b}))
        diagram.add_bang_box(Mult("k"), node_scope=frozenset({a}), parent=outer)
        diagram.add_bang_box(Mult("j"), node_scope=frozenset({b}), parent=outer)
        diagram.bind_parameter("j", 2)
        result = instantiate_symbol(diagram, "k", 0)
        assert list(result.nodes) == [c] and not result.bang_boxes
        assert result.boundary_outputs == (_out(c, 0),)
        assert dict(result.parameters) == {}

    def test_a_wire_looping_back_onto_a_grown_node_survives(self) -> None:
        d = Dim(2)
        diagram = Diagram()
        z = diagram.add_node(Z_SPIDER, [d], [d, d], phase=PhaseVector(d))
        diagram.add_wire(_out(z, 1), PortRef(z, Direction.INPUT, 0))
        diagram.set_boundary_outputs([_out(z, 0)])
        diagram.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0)}))
        grown = instantiate_symbol(diagram, "n", 2)
        assert len(grown.wires) == 1
        assert validate(grown).is_valid
        assert np.allclose(score(grown, {}).tensor, score(diagram, {"n": 2}).tensor)

    def test_a_nested_port_scope_box_moves_up_or_dies_with_its_parent(self) -> None:
        diagram = Diagram()
        z = _leaf(diagram, 2)
        diagram.set_boundary_outputs([_out(z, 0), _out(z, 1)])
        outer = diagram.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0), _out(z, 1)}))
        diagram.add_bang_box(Mult("m"), port_scope=frozenset({_out(z, 0)}), parent=outer)
        grown = instantiate_symbol(diagram, "n", 2)
        (child,) = grown.bang_boxes.values()
        assert child.parent is None and len(child.port_scope) == 2
        assert validate(grown).is_valid
        killed = instantiate_symbol(diagram, "n", 0)
        assert not killed.bang_boxes and validate(killed).is_valid
        (node,) = killed.nodes.values()
        assert node.outputs == ()

    def test_validate_rejects_a_node_scope_box_under_a_port_scope_box(self) -> None:
        diagram = Diagram()
        a, b = _leaf(diagram), _leaf(diagram)
        diagram.set_boundary_outputs([_out(a, 0), _out(b, 0)])
        legs = diagram.add_bang_box(Mult("n"), port_scope=frozenset({_out(a, 0), _out(b, 0)}))
        child = diagram.add_bang_box(Mult("k"), node_scope=frozenset({a}), parent=legs)
        (issue,) = validate(diagram).errors
        assert issue.kind is IssueKind.BANGBOX_NESTING_MISMATCH and issue.bang_box_id == child

    def test_killing_an_inner_box_first_keeps_the_outer_block_in_place(self) -> None:
        diagram = Diagram()
        a, x, b = _leaf(diagram), _leaf(diagram, generator=X_SPIDER), _leaf(diagram)
        diagram.set_boundary_outputs([_out(b, 0), _out(x, 0), _out(a, 0)])
        outer = diagram.add_bang_box(Mult("k"), node_scope=frozenset({a, b}))
        diagram.add_bang_box(Mult("j"), node_scope=frozenset({b}), parent=outer)
        inner_first = instantiate_symbol(instantiate_symbol(diagram, "j", 0), "k", 1)
        outer_first = instantiate_symbol(instantiate_symbol(diagram, "k", 1), "j", 0)
        assert _generators(inner_first) == _generators(outer_first) == ["Z", "X"]


def _hub_family(outside_generator: GeneratorType = Z_SPIDER) -> tuple[Diagram, NodeId, NodeId]:
    """A hub wired to one boxed X satellite, both with a boundary leg."""
    d = Dim(2)
    diagram = Diagram()
    hub = diagram.add_node(
        outside_generator, input_dims=[], output_dims=[d, d], phase=PhaseVector(d)
    )
    satellite = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[d], phase=PhaseVector(d))
    diagram.add_wire(_out(hub, 1), PortRef(satellite, Direction.INPUT, 0))
    diagram.set_boundary_outputs([_out(hub, 0), _out(satellite, 0)])
    diagram.add_bang_box(Mult("n"), node_scope=frozenset({satellite}))
    return diagram, hub, satellite


def _ghz(legs: int) -> Diagram:
    d = Dim(2)
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d] * legs, phase=PhaseVector(d))
    diagram.set_boundary_outputs([_out(node, i) for i in range(legs)])
    return diagram


class TestOutsideCrossing:
    """A node-scope box wired to a node outside it fans that node's port out per copy."""

    def test_validate_and_instantiate_agree(self) -> None:
        diagram, _hub, _satellite = _hub_family()
        assert validate(diagram).is_valid
        for k in range(5):
            expanded = instantiate_symbol(diagram, "n", k)
            assert validate(expanded).is_valid
            assert np.allclose(score(expanded, {}).tensor, score(_ghz(k + 1), {}).tensor)

    def test_hub_legs_are_copy_ordered(self) -> None:
        diagram, _hub, _satellite = _hub_family()
        expanded = instantiate_symbol(diagram, "n", 3)
        (hub,) = (n for n in expanded.nodes.values() if n.generator_type.name == "Z")
        assert len(hub.outputs) == 4
        satellites = [
            w.b.node_id if w.a.node_id == hub.id else w.a.node_id
            for i in range(1, 4)
            for w in expanded.wires
            if _out(hub.id, i) in (w.a, w.b)
        ]
        assert satellites == [ref.node_id for ref in expanded.boundary_outputs[1:]]

    def test_kill_removes_the_hub_leg(self) -> None:
        diagram, _hub, _satellite = _hub_family()
        killed = kill(diagram, min(diagram.bang_boxes))
        (hub,) = killed.nodes.values()
        assert len(hub.outputs) == 1 and not killed.wires

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_peel_then_instantiate_matches_instantiate_at_one_more(self, k: int) -> None:
        diagram, _hub, _satellite = _hub_family()
        diagram.set_bang_box_multiplicity(min(diagram.bang_boxes), Mult("n") + 1)
        peeled = peel_one(diagram, min(diagram.bang_boxes)).diagram
        assert validate(peeled).is_valid
        assert np.allclose(
            score(instantiate_symbol(peeled, "n", k), {}).tensor,
            score(instantiate_symbol(diagram, "n", k), {}).tensor,
        )

    def test_nested_family_is_order_independent(self) -> None:
        diagram, hub, satellite = _hub_family()
        diagram.set_boundary_outputs([_out(hub, 0), _out(satellite, 0)])
        (inner,) = diagram.bang_boxes
        diagram.remove_bang_box(inner)
        outer = diagram.add_bang_box(Mult("m"), node_scope=frozenset({hub, satellite}))
        diagram.add_bang_box(Mult("n"), node_scope=frozenset({satellite}), parent=outer)
        assert validate(diagram).is_valid
        for m, n in [(1, 0), (1, 2), (2, 2), (2, 3)]:
            inner_first = instantiate_symbol(instantiate_symbol(diagram, "n", n), "m", m)
            outer_first = instantiate_symbol(instantiate_symbol(diagram, "m", m), "n", n)
            assert np.allclose(score(inner_first, {}).tensor, score(outer_first, {}).tensor)
            assert [len(node.outputs) for node in inner_first.nodes.values()].count(n + 1) == m

    def test_two_wired_sibling_boxes_expand_to_a_complete_bipartite_graph(self) -> None:
        diagram = Diagram()
        d = Dim(2)
        left = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d))
        right = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[], phase=PhaseVector(d))
        diagram.add_wire(_out(left, 0), PortRef(right, Direction.INPUT, 0))
        diagram.add_bang_box(Mult("a"), node_scope=frozenset({left}))
        diagram.add_bang_box(Mult("b"), node_scope=frozenset({right}))
        assert validate(diagram).is_valid
        a_first = instantiate_symbol(instantiate_symbol(diagram, "a", 2), "b", 3)
        b_first = instantiate_symbol(instantiate_symbol(diagram, "b", 3), "a", 2)
        for expanded in (a_first, b_first):
            assert validate(expanded).is_valid
            assert len(expanded.nodes) == 5 and len(expanded.wires) == 6

    def test_copy_box_fans_the_hub_out_too(self) -> None:
        from archytaszx.diagram.bangbox import copy_box

        diagram, _hub, _satellite = _hub_family()
        copied, _box = copy_box(diagram, min(diagram.bang_boxes))
        assert validate(copied).is_valid
        (hub,) = (n for n in copied.nodes.values() if n.generator_type.name == "Z")
        assert len(hub.outputs) == 3

    def test_a_fixed_arity_outside_node_is_refused_by_validate(self) -> None:
        from archytaszx.diagram.generators import FOURIER_BOX
        from archytaszx.diagram.validate import IssueKind

        d = Dim(2)
        diagram = Diagram()
        box = diagram.add_node(FOURIER_BOX, input_dims=[d], output_dims=[d])
        satellite = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[], phase=PhaseVector(d))
        diagram.add_wire(_out(box, 0), PortRef(satellite, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(box, Direction.INPUT, 0)])
        diagram.add_bang_box(Mult("n"), node_scope=frozenset({satellite}))
        kinds = {issue.kind for issue in validate(diagram).errors}
        assert IssueKind.BANGBOX_CROSSING_FIXED_ARITY in kinds
