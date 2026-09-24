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


"""Covers :mod:`archytaszx.diagram.compare` directly: the id-for-id comparison and the
renaming-invariant canonical key."""

from __future__ import annotations

import os
import subprocess
import sys

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.compare import canonical_key, compare_structure, isomorphic
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef

DIM = Dim.concrete(2)


def _pair(padding: int) -> Diagram:
    """A Z state into an X effect, with ``padding`` node ids burnt first."""
    diagram = Diagram()
    for _ in range(padding):
        spare = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
        diagram.remove_node(spare)
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
    effect = diagram.add_node(X_SPIDER, input_dims=[DIM], output_dims=[])
    diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    return diagram


class TestCompareStructure:
    def test_a_diagram_equals_its_own_copy(self) -> None:
        diagram = _pair(0)
        outcome = compare_structure(diagram, diagram.copy())
        assert outcome.identical
        assert outcome.reason == "identical"

    def test_renumbering_is_not_identical(self) -> None:
        outcome = compare_structure(_pair(0), _pair(3))
        assert not outcome.identical
        assert outcome.reason.startswith("node ids differ")

    def test_a_generator_difference_is_reported_by_node(self) -> None:
        a = Diagram()
        a.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
        b = Diagram()
        b.add_node(X_SPIDER, input_dims=[], output_dims=[DIM])
        outcome = compare_structure(a, b)
        assert not outcome.identical
        assert "generator_type differs" in outcome.reason

    def test_a_phase_difference_is_reported(self) -> None:
        a = _pair(0)
        b = a.copy()
        b.set_phase(next(iter(b.nodes)), PhaseVector(DIM, {1: Phase.turns(sp.Rational(1, 2))}))
        outcome = compare_structure(a, b)
        assert not outcome.identical
        assert "phase differs" in outcome.reason

    def test_a_scalar_difference_is_reported(self) -> None:
        a = _pair(0)
        b = a.copy()
        b.multiply_scalar(Scalar.rational(2))
        outcome = compare_structure(a, b)
        assert not outcome.identical
        assert outcome.reason.startswith("scalar differs")


class TestCanonicalKey:
    def test_a_diagram_matches_its_own_copy(self) -> None:
        diagram = _pair(0)
        assert canonical_key(diagram) == canonical_key(diagram.copy())
        assert isomorphic(diagram, diagram.copy())

    def test_renumbering_does_not_move_the_key(self) -> None:
        a, b = _pair(0), _pair(4)
        assert compare_structure(a, b).identical is False
        assert canonical_key(a) == canonical_key(b)
        assert isomorphic(a, b)

    def test_a_different_diagram_gets_a_different_key(self) -> None:
        a = _pair(0)
        b = Diagram()
        b.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
        assert canonical_key(a) != canonical_key(b)
        assert not isomorphic(a, b)

    def test_a_colour_swap_is_not_isomorphic(self) -> None:
        a = _pair(0)
        b = Diagram()
        state = b.add_node(X_SPIDER, input_dims=[], output_dims=[DIM])
        effect = b.add_node(Z_SPIDER, input_dims=[DIM], output_dims=[])
        b.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
        assert canonical_key(a) != canonical_key(b)
        assert not isomorphic(a, b)

    def test_the_scalar_is_part_of_the_key(self) -> None:
        a = _pair(0)
        b = a.copy()
        b.multiply_scalar(Scalar.rational(2))
        assert canonical_key(a) != canonical_key(b)
        assert not isomorphic(a, b)

    def test_the_boundary_order_is_part_of_the_key(self) -> None:
        a = Diagram()
        first = a.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
        second = a.add_node(X_SPIDER, input_dims=[], output_dims=[DIM])
        b = a.copy()
        a.set_boundary_outputs(
            [PortRef(first, Direction.OUTPUT, 0), PortRef(second, Direction.OUTPUT, 0)]
        )
        b.set_boundary_outputs(
            [PortRef(second, Direction.OUTPUT, 0), PortRef(first, Direction.OUTPUT, 0)]
        )
        assert canonical_key(a) != canonical_key(b)
        assert not isomorphic(a, b)

    def test_a_bang_box_multiplicity_is_part_of_the_key(self) -> None:
        a = _pair(0)
        a.add_bang_box(Mult.concrete(1), node_scope=frozenset([next(iter(a.nodes))]))
        b = _pair(0)
        b.add_bang_box(Mult.concrete(2), node_scope=frozenset([next(iter(b.nodes))]))
        assert canonical_key(a) != canonical_key(b)
        assert not isomorphic(a, b)

    def test_a_renumbered_bang_box_keeps_the_key(self) -> None:
        a = _pair(0)
        a.add_bang_box(Mult.concrete(1), node_scope=frozenset([next(iter(a.nodes))]))
        b = _pair(5)
        b.add_bang_box(Mult.concrete(1), node_scope=frozenset([next(iter(b.nodes))]))
        assert canonical_key(a) == canonical_key(b)
        assert isomorphic(a, b)

    def test_a_non_diagram_is_rejected(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            canonical_key(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            isomorphic(_pair(0), object())  # type: ignore[arg-type]


class TestCanonicalKeyCrossProcessDeterminism:
    """The key must not vary by ``PYTHONHASHSEED``."""

    SCRIPT = """
import sys

import sympy as sp
sys.path.insert(0, %r)
from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.compare import canonical_key
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef

dim = Dim.concrete(2)
diagram = Diagram()
state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[dim])
diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
diagram.set_boundary_outputs(
    [PortRef(state, Direction.OUTPUT, 1), PortRef(effect, Direction.OUTPUT, 0)]
)
print(canonical_key(diagram))
"""

    def _run_with_seed(self, seed: str) -> str:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT % root],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_the_key_is_byte_identical_across_hash_seeds(self) -> None:
        first = self._run_with_seed("0")
        second = self._run_with_seed("2147483647")
        assert first.strip()
        assert second == first
