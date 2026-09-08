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

"""Unit tests for qufzx.diagram.bangbox: Mult, BangBox construction, and copy/kill/merge.

Instantiate itself (the GHZ-family and nested-two-index mechanics) is exercised end to
end, against the numeric oracle, by tests/test_phase7_oracle.py; this module covers the
pieces that oracle suite does not: the Mult expression type, BangBox's own construction
invariants, and copy/merge, which FULL_PLAN.md's Phase 7 requires but no oracle scenario
exercises.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.diagram.bangbox import (
    BangBox,
    BangBoxDomainError,
    BangBoxGrammarError,
    Mult,
    abstract_subgraph_count,
    free_mult_symbols,
    instantiate_symbol,
    kill,
    merge,
)
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import validate
from qufzx.semantics.check import score


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

        from qufzx.diagram.bangbox import copy_box

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
        from qufzx.diagram.bangbox import abstract_port_count, copy_box

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
