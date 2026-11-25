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

"""kernel.align_merge: aligned boxes merged into one box, instantiation-checked at k = 0..4,
tampered merges refused; fca_boxed equal to its spec by the radix window; merges exported over
evalP_merge_flat and Lean-checked."""

from __future__ import annotations

import os

import pytest

from archytaszx.diagram.bangbox import Mult, align_merged, instantiate_symbol
from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import check, kernel, lean_merge
from archytaszx.semantics.lean_net import check as lean_check
from archytaszx.semantics.lean_net import lean_project

ALIGNED = """
count n
x = Z(1 -> 2, literal 2)
y = Z(1 -> 1, literal 2) phase{1: (j+1)/8}
z = Z(2 -> 1, literal 2)
wire x.out1 -- y.in0, y.out0 -- z.in1, x.out0 -- z.in0
repeat n as A index j: x
repeat n as B index j: y
repeat n as C index j: z
align B with A
align C with A
inputs x.in0
outputs z.out0
"""

TWO_SLOTS = """
count n
x = Z(1 -> 2, literal 2)
y = Z(1 -> 1, literal 2) phase{1: (j+1)/8}
wire x.out1 -- y.in0
repeat n as A index j: x
repeat n as B index j: y
align B with A
inputs x.in0
outputs x.out0, y.out0
"""

FCA_SPEC = """dim d = 2
count n
h = Z(0 -> 1, literal 1)
b = B(d^j, d)
s = X(1 -> 1, d^n) phase{k: -2*k/d^n}
t = S(d^j, d)
e = Z(1 -> 0, literal 1)
wire h.out0 -- b.in0, b.out0 -- s.in0, s.out0 -- t.in0, t.out0 -- e.in0
repeat n as V index j: b
chain V: b.out0 -> b.in0
repeat n as U index j: t
chain U: t.in0 -> t.out0
inputs b.in1
outputs t.out1
"""


def _root(d: Diagram) -> object:
    return next(b.align for b in d.bang_boxes.values() if b.align is not None)


@pytest.mark.parametrize("text", [ALIGNED, TWO_SLOTS])
def test_align_merge_commutes_with_instantiation(text: str) -> None:
    d = parse_dsl(text)
    merged = align_merged(d, _root(d))  # type: ignore[arg-type]
    theorem = kernel.align_merge(d, merged)
    assert theorem.inference.kind == "align_merge"
    assert len(merged.bang_boxes) == 1
    for k in range(5):
        a, b = (instantiate_symbol(x, "n", k) for x in (d, merged))
        assert check.compare(a, b, {}).matched, k


def test_tampered_merges_are_refused() -> None:
    d = parse_dsl(ALIGNED)
    root = _root(d)
    merged = align_merged(d, root)  # type: ignore[arg-type]
    y = next(n for n, node in merged.nodes.items() if node.phase is not None)
    phased = merged.copy()
    phased.set_phase(y, None)
    with pytest.raises(kernel.KernelError):
        kernel.align_merge(d, phased)
    member = next(b for b, box in d.bang_boxes.items() if box.align is not None)
    for change in ("count", "index", "overlap"):
        tampered = d.copy()
        if change == "count":
            tampered.set_bang_box_multiplicity(member, Mult("n") + 1)
        elif change == "index":
            tampered.set_bang_box_extension(member, index="i")
        else:
            tampered.add_bang_box(Mult(2), node_scope=tampered.bang_boxes[member].node_scope)
        with pytest.raises(kernel.KernelError):
            kernel.align_merge(tampered, merged)
    two = parse_dsl(TWO_SLOTS)
    grouped = align_merged(two, _root(two))  # type: ignore[arg-type]
    ungrouped = grouped.copy()
    (box_id,) = ungrouped.bang_boxes
    ungrouped.set_bang_box_extension(box_id, groups=())
    with pytest.raises(kernel.KernelError):
        kernel.align_merge(two, ungrouped)


def test_fca_boxed_equals_its_spec_by_the_radix_window() -> None:
    from archytaszx.semantics.decide import decide_equal
    from tests.test_first_peel import _FCA

    left, right = parse_dsl(_FCA), parse_dsl(FCA_SPEC)
    decision = decide_equal(left, right, samples=[])
    assert decision.verdict.name == "EQUAL"
    theorem = kernel.from_decision(decision, left, right)
    assert theorem.inference.kind == "induction" or theorem.inference.premises
    assert "FcaEng.engFam" in kernel.lean(theorem, "main", semantic=True)
    source = kernel.lean(theorem, "main", semantic=True, guards={"n": 0})
    assert "\naxiom align_merge" not in source and "evalP_merge_flat ok ss" in source


@pytest.mark.parametrize("text", [ALIGNED, TWO_SLOTS])
def test_align_merges_export_through_evalP_merge_flat(text: str) -> None:
    d = parse_dsl(text)
    merged = align_merged(d, _root(d))  # type: ignore[arg-type]
    source = kernel.lean(kernel.align_merge(d, merged), "main", semantic=True)
    assert "\naxiom align_merge" not in source and "evalP_merge_flat ok ss" in source
    assert lean_merge.export_merge(merged, d) is None


def test_a_merge_of_another_diagram_is_not_exported() -> None:
    d = parse_dsl(ALIGNED)
    merged = align_merged(d, _root(d))  # type: ignore[arg-type]
    y = next(n for n, node in merged.nodes.items() if node.phase is not None)
    phased = merged.copy()
    phased.set_phase(y, None)
    assert lean_merge.export_merge(d, phased) is None


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_an_align_merge_and_refuses_a_moved_copy_reading(tmp_path: object) -> None:
    d = parse_dsl(ALIGNED)
    source = lean_merge.export_merge(d, align_merged(d, _root(d)))  # type: ignore[arg-type]
    assert source is not None
    ok, output = lean_check(source, f"{tmp_path}/merge.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    head, tail = source.split("def L' ", 1)
    bad = head + "def L' " + tail.replace("cph := [(1, 0, 0, 0)]", "cph := [(1, 0, 1, 0)]", 1)
    assert bad != source
    ok, output = lean_check(bad, f"{tmp_path}/bad.lean")
    assert not ok


def test_align_merge_names_the_group_refusal() -> None:
    d = parse_dsl(ALIGNED)
    merged = align_merged(d, _root(d))  # type: ignore[arg-type]
    member = next(b for b, box in d.bang_boxes.items() if box.align is not None)
    tampered = d.copy()
    tampered.set_bang_box_multiplicity(member, Mult("n") + 1)
    with pytest.raises(kernel.KernelError, match="differ in kind, count or index"):
        kernel.align_merge(tampered, merged)
