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

"""W nodes in the network export: w_state instances against the W node fed |1>, and the
[Quarter] reduction of its CRy variables at n = 2, and at n = 3 [Turn] on S reading path variables,
[Average] on a context and atoms split on the one-hot support."""

from __future__ import annotations

import os
from fractions import Fraction

import pytest

from archytaszx.algebra.pathsum import (
    Cyclo,
    PathSum,
    PathSumUnsupported,
    PhasePoly,
    canonical,
    reduce,
)
from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.decide import _instantiate_index
from archytaszx.semantics.lean_pathsum import export_net

from .test_tree_w_fusion import CIRCUIT, SPEC


def _pair(n: int) -> tuple[Diagram, Diagram]:
    return (
        _instantiate_index(parse_dsl(CIRCUIT), "n", n),
        _instantiate_index(parse_dsl(SPEC), "n", n),
    )


def test_the_w_node_is_a_fragment_and_its_sum_constraint_a_pin() -> None:
    """At n = 1 the spec's W node is wnode_frag and its input digit is pinned to the outputs'."""
    left, right = _pair(1)
    source = export_net(left, right)
    assert "wnode_frag E σ ownR labsR digR hown hσ _ rfl rfl" in source
    assert ".pin [[0], [1]]" in source


def test_a_w_node_on_dimension_4_is_refused() -> None:
    four = """a = X(0 -> 1, literal 4)
w = W(1 -> 2, literal 4)
wire a.out0 -- w.in0
outputs w.out0, w.out1"""
    with pytest.raises(PathSumUnsupported):
        export_net(parse_dsl(four), parse_dsl(four))


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


def test_quarter_sums_a_variable_with_boundary_quarter_coefficients() -> None:
    """Sum_y exp(2 pi i y (x0 + 3 x1 + 2 x0 x1) / 4) is 2, 1 + i, 1 - i, 0 at the four (x0, x1)."""
    phase = PhasePoly(
        {
            frozenset({0, 2}): Fraction(1, 4),
            frozenset({1, 2}): Fraction(3, 4),
            frozenset({0, 1, 2}): Fraction(1, 2),
        }
    )
    red = reduce(PathSum((0, 1), {2}, phase, Cyclo.rational(1)))
    assert [st.rule for st in red.trace] == ["quarter"]
    canon = canonical(red)
    assert canon is not None
    i = Cyclo.root(Fraction(1, 4))
    one = Cyclo.rational(1)
    expected = {(0, 0): one + one, (1, 0): one + i, (0, 1): one - i, (1, 1): Cyclo.rational(0)}
    for (x0, x1), value in expected.items():
        assert canon.value({0: x0, 1: x1}) == value


QUARTER_Y = ".quarter [[0], [1]] [[1]] 0"
"""The [Quarter] step of the first CRy variable at n = 2: b0 = x0 + x1, b1 = x1."""


def test_w_state_at_two_sums_its_cry_variables_by_quarter() -> None:
    """At n = 2 both leftover circuit variables take [Quarter] and the atoms cancel by acomp."""
    source = export_net(*_pair(2))
    assert QUARTER_Y in source
    assert ".quarter [[], [0], [1]] [[3]] 0" in source
    assert ".acomp 0" in source
    assert "theorem qreqsAL" in source


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize("n", [0, 1, 2])
def test_lean_checks_w_state_instances(n: int, tmp_path: object) -> None:
    left, right = _pair(n)
    ok, output = L.check(export_net(left, right), f"{tmp_path}/w{n}.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_rejects_a_tampered_quarter_step(tmp_path: object) -> None:
    """b1 of the first [Quarter] step set to x0 instead of x1 fails quarterOK."""
    source = export_net(*_pair(2))
    tampered = source.replace(QUARTER_Y, ".quarter [[0], [1]] [[0]] 0")
    assert tampered != source
    ok, _output = L.check(tampered, f"{tmp_path}/w2bad.lean")
    assert not ok


AVCTX = ".avctx 17 [[[4], [5]], [[5]]]"
"""The first [Average] on a context at n = 3: y17 on the context x4 + x5, x5."""


def test_w_state_at_three_reduces_by_held_turns_and_context_averages() -> None:
    """At n = 3 [Turn] with S on path variables and [Average] on a context leave no path
    variable, and the export splits atoms on the one-hot support."""
    from archytaszx.semantics.pathsum_diagram import from_diagram

    raw = from_diagram(_pair(3)[0], w_nodes=True)
    assert isinstance(raw, PathSum)
    red = reduce(raw)
    assert not red.path
    assert {"turn_held", "average_ctx"} <= {st.rule for st in red.trace}
    source = export_net(*_pair(3))
    assert AVCTX in source
    assert ".asplit" in source


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_checks_w_state_at_three(tmp_path: object) -> None:
    ok, output = L.check(export_net(*_pair(3)), f"{tmp_path}/w3.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_rejects_a_tampered_context_average(tmp_path: object) -> None:
    """The context x4 + x5, x5 of the first [Average] on a context changed to x4 + x5, x4 fails
    avctxOK."""
    source = export_net(*_pair(3))
    tampered = source.replace(AVCTX, ".avctx 17 [[[4], [5]], [[4]]]", 1)
    assert tampered != source
    ok, _output = L.check(tampered, f"{tmp_path}/w3bad.lean")
    assert not ok
