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

"""Phase 7 numeric-oracle check: free n end to end, including nesting and multiple indices.

Per FULL_PLAN.md, Phase 7 is done when "free n works end to end, including nesting and
multiple indices", tested by: the bang-boxed GHZ family instantiating correctly at k =
0, 1, 2, 3; a nested two-index family instantiating correctly on both indices; and
fusion under a bang box preserving exact oracle equality across several tuples of counts
and d. All three are exercised here, plus the negative controls the plan calls for.
"""

from __future__ import annotations

import numpy as np
import pytest

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.diagram.bangbox import (
    BangBoxDomainError,
    BangBoxGrammarError,
    abstract_port_count,
    abstract_subgraph_count,
    free_mult_symbols,
    instantiate_symbol,
)
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import CheckGrammarError, EqualityMode, compare, score

from .helpers import build_ghz_with_copy


def _instantiate_if_present(diagram: Diagram, name: str, value: int) -> Diagram:
    """``instantiate_symbol`` if ``name`` is still free, otherwise a no-op.

    Mirrors what :func:`~qufzx.semantics.check.score`/``compare`` do: a symbol a prior
    instantiation has already eliminated (Group B's k1=0 case) is simply not asked for.
    """
    if name in free_mult_symbols(diagram):
        return instantiate_symbol(diagram, name, value)
    return diagram


def _build_ghz_family(dim_value: int) -> Diagram:
    """One Z spider, output leg port-scope boxed under symbol ``n`` -- the minimal
    bang-boxed GHZ family (Group A)."""
    d = Dim(dim_value)
    diagram = Diagram()
    s = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(s, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    diagram, _box_id, _n = abstract_port_count(diagram, ref, 1, stem="n")
    return diagram


class TestGroupA_BangBoxedGHZInstantiation:
    """The bang-boxed GHZ family at k = 0, 1, 2, 3, at d = 2 and d = 3."""

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_k_zero_is_the_scalar_d(self, d_value: int) -> None:
        diagram = instantiate_symbol(_build_ghz_family(d_value), "n", 0)
        result = score(diagram, {})
        assert result.tensor.shape == ()
        assert np.allclose(result.tensor, d_value)

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_k_one_is_the_all_ones_vector(self, d_value: int) -> None:
        diagram = instantiate_symbol(_build_ghz_family(d_value), "n", 1)
        result = score(diagram, {})
        assert result.tensor.shape == (d_value,)
        assert np.allclose(result.tensor, np.ones(d_value))

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_k_two_is_the_diagonal_sum_jj(self, d_value: int) -> None:
        diagram = instantiate_symbol(_build_ghz_family(d_value), "n", 2)
        result = score(diagram, {})
        expected = np.zeros((d_value, d_value), dtype=complex)
        for j in range(d_value):
            expected[j, j] = 1
        assert np.allclose(result.tensor, expected)

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_k_three_is_the_ghz_state(self, d_value: int) -> None:
        diagram = instantiate_symbol(_build_ghz_family(d_value), "n", 3)
        result = score(diagram, {})
        expected = np.zeros((d_value, d_value, d_value), dtype=complex)
        for j in range(d_value):
            expected[j, j, j] = 1
        assert np.allclose(result.tensor, expected)

    @pytest.mark.parametrize("d_value", [2, 3])
    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_score_and_direct_instantiate_agree(self, d_value: int, k: int) -> None:
        """``score`` (driving expansion through the ordinary oracle entry point) and a
        direct :func:`instantiate_symbol` call must contract to the same tensor."""
        via_score = score(_build_ghz_family(d_value), {"n": k}).tensor
        via_direct = score(instantiate_symbol(_build_ghz_family(d_value), "n", k), {}).tensor
        assert np.allclose(via_score, via_direct)


def _build_nested_two_index_family(dim_value: int) -> Diagram:
    """One Z spider whose output leg is port-scope boxed under ``k2``, the whole gadget
    node-scope boxed under ``k1`` (Group B)."""
    d = Dim(dim_value)
    diagram = Diagram()
    s = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(s, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    diagram, inner_id, _k2 = abstract_port_count(diagram, ref, 1, stem="k2")
    diagram, outer_id, _k1 = abstract_subgraph_count(diagram, frozenset({s}), 1, stem="k1")
    inner = diagram.bang_boxes[inner_id]
    diagram.remove_bang_box(inner_id)
    diagram.add_bang_box(inner.multiplicity, port_scope=inner.port_scope, parent=outer_id)
    return diagram


class TestGroupB_NestedTwoIndexFamily:
    """A nested two-index family instantiates correctly on both indices, independent of
    which index is driven first."""

    _CASES: tuple[tuple[int, int, tuple[int, ...]], ...] = (
        (0, 0, ()),  # scalar 1
        (0, 2, ()),  # scalar 1, k2's value irrelevant
        (2, 0, ()),  # scalar d**2
        (1, 1, (2,)),  # a single d-vector of ones
        (2, 3, (2, 2, 2, 2, 2, 2)),  # two independent 3-qudit GHZ blocks
    )

    @pytest.mark.parametrize("k1,k2,expected_shape", _CASES)
    def test_closed_form_at_d_equals_2(
        self, k1: int, k2: int, expected_shape: tuple[int, ...]
    ) -> None:
        d_value = 2
        diagram = _build_nested_two_index_family(d_value)
        diagram = _instantiate_if_present(diagram, "k1", k1)
        diagram = _instantiate_if_present(diagram, "k2", k2)
        result = score(diagram, {})
        assert result.tensor.shape == expected_shape

        if k1 == 0:
            assert np.allclose(result.tensor, 1.0)
        elif (k1, k2) == (2, 0):
            assert np.allclose(result.tensor, float(d_value**2))
        elif (k1, k2) == (1, 1):
            assert np.allclose(result.tensor, np.ones(d_value))
        elif (k1, k2) == (2, 3):
            block = np.zeros((d_value, d_value, d_value), dtype=complex)
            for j in range(d_value):
                block[j, j, j] = 1
            expected = np.tensordot(block, block, axes=0)
            assert np.allclose(result.tensor, expected)

    @pytest.mark.parametrize("k1,k2,_shape", _CASES)
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_order_independence(
        self, k1: int, k2: int, _shape: tuple[int, ...], d_value: int
    ) -> None:
        outer_first = _build_nested_two_index_family(d_value)
        outer_first = _instantiate_if_present(outer_first, "k1", k1)
        outer_first = _instantiate_if_present(outer_first, "k2", k2)

        inner_first = _build_nested_two_index_family(d_value)
        inner_first = _instantiate_if_present(inner_first, "k2", k2)
        inner_first = _instantiate_if_present(inner_first, "k1", k1)

        result = compare(outer_first, inner_first, {}, mode=EqualityMode.EXACT)
        assert result.matched, result.reason


def _fuse_once(diagram: Diagram) -> Diagram:
    matches = find_matches(diagram)
    assert len(matches) == 1, "expected exactly one fusion match on the boxed A-into-B example"
    return apply(diagram, SPIDER_FUSION, matches[0]).diagram


class TestGroupC_FusionUnderABangBox:
    """Fusion firing inside a bang box preserves exact oracle equality, box left intact."""

    # (5, 4) is dropped: 3 legs/copy * 5 copies at d=4 contracts to 4**15 (~1.07e9)
    # elements, well beyond what an exact numeric contraction can hold in memory --
    # a genuine size limit of the rung-3 oracle (qufzx.semantics.check's own module
    # docstring: "a small-instance path"), not a defect in the mechanism under test.
    _MATRIX: tuple[tuple[int, int], ...] = tuple(
        (m, d) for m in (0, 1, 2, 3, 5) for d in (2, 3, 4) if (m, d) != (5, 4)
    )

    def test_box_survives_fusion_with_updated_scope(self) -> None:
        d = Dim("d")
        pre, a_id, b_id = build_ghz_with_copy(d)
        pre, box_id, _m = abstract_subgraph_count(pre, frozenset({a_id, b_id}), 1, stem="m")
        post = _fuse_once(pre)

        assert set(post.bang_boxes) == {box_id}
        box = post.bang_boxes[box_id]
        assert box.multiplicity == pre.bang_boxes[box_id].multiplicity
        (merged_id,) = post.nodes
        assert box.node_scope == frozenset({merged_id})

    @pytest.mark.parametrize("m_value,d_value", _MATRIX)
    def test_exact_oracle_equality_across_counts_and_d(self, m_value: int, d_value: int) -> None:
        d = Dim("d")
        pre, a_id, b_id = build_ghz_with_copy(d)
        pre, _box_id, _m = abstract_subgraph_count(pre, frozenset({a_id, b_id}), 1, stem="m")
        post = _fuse_once(pre)

        pre_instantiated = instantiate_symbol(pre, "m", m_value)
        post_instantiated = instantiate_symbol(post, "m", m_value)
        result = compare(
            pre_instantiated,
            post_instantiated,
            {"d": d_value},
            mode=EqualityMode.EXACT,
            max_elements=20_000_000,
        )
        assert result.matched, result.reason


class TestNegativeAndRegressionControls:
    def test_unsupplied_count_symbol_raises(self) -> None:
        # abstract_port_count always binds the parameter environment too (mirroring
        # Dim.abstract), so a genuinely *unsupplied* symbol needs a hand-built box with
        # no such binding -- not the ordinary abstraction path, which never leaves one.
        from qufzx.diagram.bangbox import Mult

        d = Dim(2)
        diagram = Diagram()
        s = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        ref = PortRef(s, Direction.OUTPUT, 0)
        diagram.set_boundary_outputs([ref])
        diagram.add_bang_box(Mult.symbol("n"), port_scope=frozenset({ref}))
        with pytest.raises(CheckGrammarError):
            score(diagram, {})

    def test_fusion_across_a_bang_box_boundary_is_refused(self) -> None:
        d = Dim("d")
        pre, a_id, _b_id = build_ghz_with_copy(d)
        # Box only A: the fusable pair now straddles a box boundary (A inside, B outside).
        pre, _box_id, _m = abstract_subgraph_count(pre, frozenset({a_id}), 1, stem="m")
        assert find_matches(pre) == ()

    def test_wire_crossing_to_another_live_node_is_refused_at_instantiation(self) -> None:
        """A node-scope box's crossing wire must land on the diagram boundary (module
        docstring's Phase 7 scope restriction); a crossing to another live node is
        refused with a clear error rather than mishandled."""
        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        b = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[], phase=PhaseVector(d, {}))
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        diagram, _box_id, _m = abstract_subgraph_count(diagram, frozenset({a}), 1, stem="m")
        with pytest.raises(BangBoxGrammarError):
            instantiate_symbol(diagram, "m", 2)

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_two_boxes_sharing_one_count_symbol_expand_together(self, k: int) -> None:
        """One symbol means one multiplicity: every box owning it expands to the same k.

        This is the shape instantiating an enclosing box produces -- it re-parents the
        duplicated children to siblings, all still owning the inner name.
        """
        from qufzx.diagram.bangbox import Mult

        d = Dim(2)
        diagram = Diagram()
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        b = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
        diagram.set_boundary_outputs(
            [PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.OUTPUT, 0)]
        )
        diagram.add_bang_box(Mult.symbol("k"), node_scope=frozenset({a}))
        diagram.add_bang_box(Mult.symbol("k"), node_scope=frozenset({b}))
        assert validate(diagram).is_valid

        expanded = instantiate_symbol(diagram, "k", k)
        assert validate(expanded).is_valid
        tensor = score(expanded, {}).tensor
        assert np.allclose(tensor, np.ones((2,) * (2 * k)))

    @pytest.mark.parametrize("k1", [0, 1, 2, 3])
    def test_instantiation_preserves_validity(self, k1: int) -> None:
        """Instantiating one index of a nested family leaves a diagram that still validates."""
        diagram = _build_nested_two_index_family(2)
        assert validate(diagram).is_valid
        assert validate(instantiate_symbol(diagram, "k1", k1)).is_valid

    def test_vacuous_inner_symbol_after_kill_needs_no_assignment(self) -> None:
        diagram = _build_nested_two_index_family(2)
        diagram = instantiate_symbol(diagram, "k1", 0)
        assert "k2" not in free_mult_symbols(diagram)
        result = score(diagram, {})  # must not raise for a missing k2
        assert np.allclose(result.tensor, 1.0)

    @pytest.mark.parametrize("assignment", [{"k1": 0}, {"k1": 0, "k2": 3}])
    def test_kill_driven_through_score_matches_pre_instantiation(
        self, assignment: dict[str, int]
    ) -> None:
        """Letting score instantiate the killing count agrees with killing it first."""
        pre = score(instantiate_symbol(_build_nested_two_index_family(2), "k1", 0), {}).tensor
        via_score = score(_build_nested_two_index_family(2), assignment).tensor
        assert np.allclose(via_score, pre)

    def test_a_dead_count_symbol_still_rejects_an_out_of_domain_value(self) -> None:
        with pytest.raises(BangBoxDomainError):
            score(_build_nested_two_index_family(2), {"k1": 0, "k2": -1})


class TestFusionUnderAPortScopeBox:
    """A port-scope box follows its leg onto the merged spider (Phase 7 iv)."""

    @staticmethod
    def _boxed_gadget(d_value: int) -> Diagram:
        diagram, _a_id, _b_id = build_ghz_with_copy(Dim(d_value))
        diagram, _box, _k = abstract_port_count(diagram, diagram.boundary_outputs[-1], 1, stem="k")
        return diagram

    def test_fusion_fires_and_leaves_the_port_box_on_a_live_port(self) -> None:
        diagram = self._boxed_gadget(2)
        matches = find_matches(diagram)
        assert len(matches) == 1
        fused = apply(diagram, SPIDER_FUSION, matches[0]).diagram
        report = validate(fused)
        assert report.is_valid, [issue.kind.value for issue in report.errors]
        for box in fused.bang_boxes.values():
            for ref in box.port_scope:
                assert ref.node_id in fused.nodes
                assert ref in fused.boundary_inputs or ref in fused.boundary_outputs

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    @pytest.mark.parametrize("d_value", [2, 3])
    def test_fusion_under_a_port_box_preserves_the_oracle(self, k: int, d_value: int) -> None:
        diagram = self._boxed_gadget(d_value)
        fused = apply(diagram, SPIDER_FUSION, find_matches(diagram)[0]).diagram
        result = compare(diagram, fused, {"k": k}, mode=EqualityMode.EXACT)
        assert result.matched, result.reason
