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

"""W nodes in the network export: w_state instances against the W node fed |1>."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
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


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize("n", [0, 1])
def test_lean_checks_w_state_instances(n: int, tmp_path: object) -> None:
    left, right = _pair(n)
    ok, output = L.check(export_net(left, right), f"{tmp_path}/w{n}.lean")
    assert ok, output
