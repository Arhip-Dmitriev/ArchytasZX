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

"""The free column count rung (REMAINING_BUGS_SPEC 79.3): boxed cluster_state_2d EQUAL with a
kernel proof and no oracle sample, its CZ^-1 variant not; the kernel's regroup and identity
slide refuse what they do not license; aligned boxes in a zipped chain instantiate."""

from __future__ import annotations

import pytest

from archytaszx.diagram.bangbox import instantiate_symbol
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.check import EqualityMode, compare
from archytaszx.semantics.decide import EqualityVerdict, decide_equal
from tests.test_nested_chain import _grid_pair


def test_cluster_state_2d_is_equal_with_a_kernel_proof() -> None:
    left, right = _grid_pair(1)
    decision = decide_equal(left, right, samples=[])
    assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
    theorem = kernel.from_decision(decision, left, right)
    assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)
    assert [c.text() for c in theorem.domain] == ["s = -1 + n"]


def test_the_inverse_cz_grid_is_not_proved() -> None:
    left, right = _grid_pair(-1)
    assert decide_equal(left, right, samples=[]).verdict is not EqualityVerdict.EQUAL


ROW = """dim d
count m
v = Z(1 -> 2, d)
u = Z(1 -> 1, d)
w = Z(1 -> 1, d)
wire v.out1 -- u.in0
repeat m as A: v, w
repeat m as B: u
outputs v.out0, u.out0, w.out0
inputs v.in0, w.in0"""
"""Copies of a copy spider whose second leg runs through an identity, beside an idle wire."""


def test_regroup_refuses_a_change_on_one_side_only() -> None:
    d = parse_dsl(ROW)
    grouped = d.copy()
    v = next(n for n, node in d.nodes.items() if len(node.outputs) == 2)
    a = next(b for b, box in d.bang_boxes.items() if v in box.node_scope)
    w = next(n for n in d.bang_boxes[a].node_scope if n != v)
    grouped.set_bang_box_extension(a, groups=(frozenset({v}), frozenset({w})))
    with pytest.raises(kernel.KernelError, match="lay out their slots differently"):
        kernel.regroup(kernel.refl(d), grouped, d)
    assert kernel._same(kernel.regroup(kernel.refl(d), grouped, grouped).lhs, grouped)


def test_identity_slide_refuses_a_diagram_that_is_no_slide() -> None:
    d = parse_dsl(ROW)
    with pytest.raises(kernel.KernelError, match="no identity slide"):
        kernel.identity_slide(d, d, frozenset())


ALIGNED = """dim d
count m, s
a = X(0 -> 1, d)
b = Z(1 -> 1, d)
c = Z(1 -> 2, d)
e = Z(1 -> 1, d)
r = X(0 -> 1, d)
q = Z(1 -> 0, d)
wire a.out0 -- b.in0, b.out0 -- c.in0, c.out1 -- e.in0, r.out0 -- q.in0
repeat m as H: a, c
repeat s as R: b, r, q
repeat m as Ri in R: b
chain R: b.out0 -> b.in0
repeat m as X: e
align X with H
outputs c.out0, e.out0"""
"""Columns of an X state passed through s identity rows (each with a closed scalar pair) into a
copy spider whose second leg runs through an aligned box of identities."""


@pytest.mark.parametrize(("s", "m"), [(1, 2), (2, 2), (2, 3)])
def test_aligned_boxes_in_a_zipped_chain_instantiate_in_either_order(s: int, m: int) -> None:
    d = parse_dsl(ALIGNED)
    first = instantiate_symbol(instantiate_symbol(d, "s", s), "m", m)
    second = instantiate_symbol(instantiate_symbol(d, "m", m), "s", s)
    assert compare(first, second, {"d": 2}).matched
    text = ALIGNED.replace("count m, s", "count m").replace("repeat s as R: b, r, q\n", "")
    text = text.replace("repeat m as Ri in R: b\nchain R: b.out0 -> b.in0", "repeat m as Ri: b")
    plain = instantiate_symbol(parse_dsl(f"{text}\nalign Ri with H"), "m", m)
    assert compare(first, plain, {"d": 2}, mode=EqualityMode.UP_TO_POSITIVE_SCALAR).matched
