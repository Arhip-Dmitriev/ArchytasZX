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

"""Symbolic contraction of families whose boundary grows with a multiplicity."""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, free_mult_symbols
from archytaszx.diagram.generators import FOURIER_BOX, W_NODE, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.repl.parser import parse_dirac_source
from archytaszx.semantics.check import compare_symbolic, score
from archytaszx.semantics.contract_symbolic import (
    ReplicatedBlock,
    SymbolicContractionDomainError,
    SymbolicContractionUnsupportedError,
    contract_symbolic,
)

D = Dim("d")
DENSE_CAP = 2000


def out(node: int, index: int) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def inp(node: int, index: int) -> PortRef:
    return PortRef(node, Direction.INPUT, index)


def third() -> PhaseVector:
    return PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))})


def z_with_input() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, [D], [D, D], phase=third())
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 0)}))
    return diagram


def x_two_ports() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(X_SPIDER, [D], [D, D], phase=third())
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 0), inp(node, 0)}))
    return diagram


def x_wired() -> Diagram:
    diagram = Diagram()
    x = diagram.add_node(X_SPIDER, [], [D, D])
    z = diagram.add_node(Z_SPIDER, [D], [D, D], phase=third())
    diagram.add_wire(out(x, 0), inp(z, 0))
    diagram.set_boundary_outputs([out(z, 0), out(x, 1), out(z, 1)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(x, 1)}))
    return diagram


def z_alone() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, [], [D], phase=third())
    diagram.set_boundary_outputs([out(node, 0)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 0)}))
    return diagram


def two_symbols() -> Diagram:
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, [D], [D, D])
    x = diagram.add_node(X_SPIDER, [D], [D])
    diagram.add_wire(out(z, 1), inp(x, 0))
    diagram.set_boundary_inputs([inp(z, 0)])
    diagram.set_boundary_outputs([out(x, 0), out(z, 0)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(z, 0)}))
    diagram.add_bang_box(Mult("m"), port_scope=frozenset({out(x, 0)}))
    return diagram


def node_scope_identity() -> Diagram:
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, [D], [D], phase=third())
    b = diagram.add_node(Z_SPIDER, [], [D, D])
    diagram.set_boundary_inputs([inp(a, 0)])
    diagram.set_boundary_outputs([out(b, 0), out(a, 0), out(b, 1)])
    diagram.add_bang_box(Mult("n"), node_scope=frozenset({a}))
    return diagram


def node_scope_with_fourier() -> Diagram:
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, [D], [D, D])
    f = diagram.add_node(FOURIER_BOX, [D], [D])
    z = diagram.add_node(Z_SPIDER, [], [D])
    diagram.add_wire(out(a, 1), inp(f, 0))
    diagram.set_boundary_inputs([inp(a, 0)])
    diagram.set_boundary_outputs([out(f, 0), out(z, 0), out(a, 0)])
    diagram.add_bang_box(Mult("n"), node_scope=frozenset({a, f}))
    return diagram


def node_scope_with_closed_child() -> Diagram:
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, [D], [D])
    x = diagram.add_node(X_SPIDER, [], [D])
    z = diagram.add_node(Z_SPIDER, [D], [])
    diagram.add_wire(out(x, 0), inp(z, 0))
    diagram.set_boundary_inputs([inp(a, 0)])
    diagram.set_boundary_outputs([out(a, 0)])
    outer = diagram.add_bang_box(Mult("n"), node_scope=frozenset({a, x, z}))
    diagram.add_bang_box(Mult("m"), node_scope=frozenset({x, z}), parent=outer)
    return diagram


def shifted_count() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, [], [D, D])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    diagram.add_bang_box(Mult("n") + 1, port_scope=frozenset({out(node, 1)}))
    return diagram


def w_boxed() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(W_NODE, [D], [D, D])
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 1)}))
    return diagram


def w_only_boxed() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(W_NODE, [D], [D])
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 0)}))
    return diagram


def w_wired_with_z() -> Diagram:
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, [], [D], phase=third())
    node = diagram.add_node(W_NODE, [D], [D, D])
    z = diagram.add_node(Z_SPIDER, [], [D, D])
    diagram.add_wire(out(state, 0), inp(node, 0))
    diagram.set_boundary_outputs([out(node, 0), out(z, 0), out(node, 1), out(z, 1)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 1), out(z, 1)}))
    return diagram


def w_two_boxes() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(W_NODE, [D], [D, D])
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(node, 0)}))
    diagram.add_bang_box(Mult("m"), port_scope=frozenset({out(node, 1)}))
    return diagram


def nested_ghz() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, [], [D, D], phase=third())
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    outer = diagram.add_bang_box(Mult("n"), node_scope=frozenset({node}))
    diagram.add_bang_box(Mult("m"), port_scope=frozenset({out(node, 1)}), parent=outer)
    return diagram


def nested_with_input() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(X_SPIDER, [D], [D, D])
    z = diagram.add_node(Z_SPIDER, [], [D])
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0), out(z, 0), out(node, 1)])
    outer = diagram.add_bang_box(Mult("n"), node_scope=frozenset({node}))
    diagram.add_bang_box(Mult("m"), port_scope=frozenset({out(node, 0)}), parent=outer)
    return diagram


def nested_w() -> Diagram:
    diagram = Diagram()
    node = diagram.add_node(W_NODE, [D], [D, D])
    diagram.set_boundary_inputs([inp(node, 0)])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    outer = diagram.add_bang_box(Mult("n"), node_scope=frozenset({node}))
    diagram.add_bang_box(Mult("m"), port_scope=frozenset({out(node, 1)}), parent=outer)
    return diagram


FAMILIES: dict[str, Callable[[], Diagram]] = {
    "z_with_input": z_with_input,
    "x_two_ports": x_two_ports,
    "x_wired": x_wired,
    "z_alone": z_alone,
    "two_symbols": two_symbols,
    "node_scope_identity": node_scope_identity,
    "node_scope_with_fourier": node_scope_with_fourier,
    "node_scope_with_closed_child": node_scope_with_closed_child,
    "shifted_count": shifted_count,
    "w_boxed": w_boxed,
    "w_only_boxed": w_only_boxed,
    "w_wired_with_z": w_wired_with_z,
    "w_two_boxes": w_two_boxes,
    "nested_ghz": nested_ghz,
    "nested_with_input": nested_with_input,
    "nested_w": nested_w,
}


class TestAgainstTheOracle:
    """The family, substituted, equals the numeric oracle's tensor of the instance."""

    @pytest.mark.parametrize("name", sorted(FAMILIES))
    @pytest.mark.parametrize("d", [2, 3])
    def test_every_count_up_to_three(self, name: str, d: int) -> None:
        diagram = FAMILIES[name]()
        family = contract_symbolic(diagram)
        assert family.is_variable_rank
        names = sorted(free_mult_symbols(diagram))
        for values in itertools.product(range(4), repeat=len(names)):
            assignment = {"d": d, **dict(zip(names, values, strict=True))}
            instance = family.substitute(assignment)
            assert not instance.is_variable_rank
            if d**instance.rank > DENSE_CAP:
                continue
            expected = score(diagram, assignment).tensor
            dense = instance.to_dense()
            assert isinstance(dense, np.ndarray)
            assert dense.shape == expected.shape, assignment
            assert np.allclose(dense, expected), assignment


class TestTheFlagshipInstance:
    """The user's own instance comes out of the closed form, never out of a dense tensor."""

    def test_ghz_thirty_at_three(self) -> None:
        diagram = parse_dirac_source("sum_{k=0}^{3-1} |k>^{30}")
        family = contract_symbolic(diagram)
        assert family.is_variable_rank and family.rank == 0
        instance = family.substitute(dict(diagram.parameters))
        assert instance.rank == 30 and instance.dims() == (Dim.concrete(3),) * 30
        assert instance.value_at((2,) * 30) == Scalar.one()
        assert instance.value_at((2,) * 29 + (1,)) == Scalar.zero()

    def test_ghz_thirty_with_a_copy(self) -> None:
        diagram = parse_dirac_source("sum_{k=0}^{3-1} |k>^{30}; copy")
        family = contract_symbolic(diagram)
        assert family.shared == ()
        instance = family.substitute(dict(diagram.parameters))
        assert instance.rank == 31
        assert instance.value_at((1,) * 31) == Scalar.one()
        assert instance.value_at((1,) * 30 + (0,)) == Scalar.zero()

    def test_the_layout_places_the_block_where_the_leg_was(self) -> None:
        family = contract_symbolic(parse_dirac_source("sum_{k=0}^{3-1} |k>^{30}; copy"))
        assert isinstance(family.slots()[0], ReplicatedBlock)
        assert [type(slot).__name__ for slot in family.slots()[1:]] == ["SymbolicAxis"] * 2


class TestRefusals:
    def test_a_variable_rank_tensor_has_no_entries_until_substituted(self) -> None:
        family = contract_symbolic(z_alone())
        with pytest.raises(SymbolicContractionDomainError, match="substitute"):
            family.value_at(())
        with pytest.raises(SymbolicContractionDomainError, match="substitute"):
            family.to_dense()
        with pytest.raises(SymbolicContractionDomainError, match="symbolic"):
            family.flatten()


class TestComparison:
    def test_equal_families_match(self) -> None:
        result = compare_symbolic(
            contract_symbolic(z_with_input()), contract_symbolic(z_with_input())
        )
        assert result.matched, result.reason

    def test_different_counts_are_indeterminate_not_unequal(self) -> None:
        result = compare_symbolic(contract_symbolic(shifted_count()), contract_symbolic(z_alone()))
        assert not result.matched and result.reason.startswith("indeterminate")

    def test_a_fixed_rank_tensor_never_matches_a_variable_rank_one(self) -> None:
        fixed = parse_dirac_source("sum_{k=0}^{d-1} |k,k>")
        result = compare_symbolic(contract_symbolic(fixed), contract_symbolic(z_alone()))
        assert not result.matched

    def test_equal_nested_families_match(self) -> None:
        result = compare_symbolic(contract_symbolic(nested_ghz()), contract_symbolic(nested_ghz()))
        assert result.matched

    def test_w_and_nested_families_differ_from_their_neighbours(self) -> None:
        result = compare_symbolic(contract_symbolic(w_boxed()), contract_symbolic(z_with_input()))
        assert not result.matched


def test_a_box_over_two_w_nodes_is_refused() -> None:
    diagram = Diagram()
    first = diagram.add_node(W_NODE, [D], [D])
    second = diagram.add_node(W_NODE, [D], [D])
    diagram.set_boundary_inputs([inp(first, 0), inp(second, 0)])
    diagram.set_boundary_outputs([out(first, 0), out(second, 0)])
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(first, 0), out(second, 0)}))
    with pytest.raises(SymbolicContractionUnsupportedError):
        contract_symbolic(diagram)
