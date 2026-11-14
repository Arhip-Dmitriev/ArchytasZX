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

"""Kernel regroup layout facts exported through ``regroup_layout`` (REMAINING_BUGS_SPEC 79.16b):
the box groups of a row and of the cluster_state_2d proof as slot blocks, a pair differing beyond
box groups refused, the semantic export with no ``lay`` axiom, and Lean accepting the layout proof
and rejecting a block that is no boundary slot of its chain."""

from __future__ import annotations

import os

import pytest

from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel, lean_regroup
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_net import check, lean_project
from tests.test_columns import ROW
from tests.test_nested_chain import _grid_pair


def _grouped() -> tuple[Diagram, Diagram]:
    d = parse_dsl(ROW)
    grouped = d.copy()
    v = next(n for n, node in d.nodes.items() if len(node.outputs) == 2)
    a = next(b for b, box in d.bang_boxes.items() if v in box.node_scope)
    w = next(n for n in d.bang_boxes[a].node_scope if n != v)
    grouped.set_bang_box_extension(a, groups=(frozenset({v}), frozenset({w})))
    return d, grouped


def test_a_row_regroup_exports_through_regroup_layout() -> None:
    d, grouped = _grouped()
    source = lean_regroup.export_regroup(grouped, d)
    assert source is not None and "regroup_layout L G G'" in source
    assert "def G : List Blk := [([0], [0, 2]), ([0], [1]), ([0], [3, 4])]" in source
    other = d.copy()
    other.remove_node(next(iter(other.nodes)))
    assert lean_regroup.export_regroup(other, d) is None
    step = kernel.regroup(kernel.refl(d), grouped, grouped)
    lean = kernel.lean(step, "main")
    assert "namespace ArchytasZX.Export.R" in lean and "axiom lay" not in lean
    assert "axiom lay" in kernel.lean(step, "main", semantic=False)


def test_the_cluster_state_2d_regroup_exports_through_regroup_layout() -> None:
    left, right = _grid_pair(1)
    theorem = kernel.from_decision(decide_equal(left, right, samples=[]), left, right)
    source = kernel.lean(theorem, "main")
    assert "axiom slots" in source and "axiom lay" not in source
    assert source.count("regroup_layout L G G'") == 2


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_layout_and_refuses_a_stray_block(tmp_path: object) -> None:
    d, grouped = _grouped()
    source = lean_regroup.export_regroup(grouped, d)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/regroup.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("def G' : List Blk := [", "def G' : List Blk := [([], [0]), ")
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
