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

"""Joint peel of an alignment group and the split of a power box (qram, 77.2)."""

from __future__ import annotations

import pytest

from archytaszx.diagram.bangbox import (
    BangBoxGrammarError,
    index_owners,
    peel_aligned,
    peel_owners,
    split_power,
)
from archytaszx.diagram.graph import BangBoxId, Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.check import compare
from archytaszx.semantics.prove import induction_renamed

QRAM = """
count n
g0 = Z(1 -> 2, literal 2)
g1 = Z(1 -> 2, literal 2)
g2 = X(0 -> 1, literal 2^n) phase{k: -v*k/2^n}
g3 = S(literal 2, literal 2^j)
g4 = Z(1 -> 0, literal 1)
g5 = X(2 -> 1, literal 2)
g6 = X(1 -> 1, literal 2) phase{1: 1/2}
g7 = T(literal 2)
g8 = Z(2 -> 1, literal 2)
g9 = T(literal 2)
g10 = Ti(literal 2)
g11 = X(2 -> 1, literal 2)
wire g2.out0 -- g3.in0, g3.out1 -- g4.in0, g0.out1 -- g5.in0, g3.out0 -- g5.in1
wire g5.out0 -- g6.in0, g6.out0 -- g7.in0, g7.out0 -- g8.in1, g1.out1 -- g9.in0
wire g9.out0 -- g8.in0, g8.out0 -- g10.in0, g10.out0 -- g11.in1
repeat n as C0: g0
repeat 2^n as C1 index v: g1, g2, g3, g4, g5, g6, g7, g8, g9, g10
repeat n as C2 index j in C1: g3, g5, g6, g7
chain C2: g3.in0 -> g3.out1
align C2 with C0
inputs g0.in0, g1.in0, g11.in0
outputs g0.out0, g1.out0, g11.out0
"""


C1 = BangBoxId(1)


def _source() -> tuple[Diagram, Diagram]:
    circuit = parse_dsl(QRAM)
    return circuit, induction_renamed(circuit, "n", "k")


@pytest.mark.parametrize("last", [False, True])
def test_an_alignment_group_peels_one_copy_of_each_box(last: bool) -> None:
    circuit, source = _source()
    owners = frozenset(index_owners(source, "k")) if last else frozenset()
    peeled = peel_owners(source, "k", last=owners, partial=True).diagram
    assert peeled.bang_boxes[BangBoxId(2)].align == BangBoxId(0)
    for value in (0, 1):
        assert compare(peeled, circuit, {"k": value, "n": value + 1}).matched


def test_a_box_aligned_to_no_other_box_has_no_aligned_peel() -> None:
    _circuit, source = _source()
    with pytest.raises(BangBoxGrammarError):
        peel_aligned(source, C1)


def test_a_power_box_splits_into_two_halves() -> None:
    circuit, source = _source()
    peeled = peel_owners(source, "k", partial=True).diagram
    split, boxes = split_power(peeled, C1)
    assert [str(split.bang_boxes[b].multiplicity) for b in boxes] == ["2**k", "2**k"]
    assert {split.bang_boxes[b].align for b in split.bang_boxes if b not in (0, *boxes)} == {
        BangBoxId(0)
    }
    for value in (0, 1):
        assert compare(split, circuit, {"k": value, "n": value + 1}).matched
    theorem = kernel.power_split(peeled, split)
    assert kernel._same(theorem.rhs, split)
    with pytest.raises(kernel.KernelError):
        kernel.power_split(peeled, peeled)


def test_a_box_whose_exponent_may_be_zero_does_not_split() -> None:
    circuit, _source_k = _source()
    with pytest.raises(BangBoxGrammarError):
        split_power(circuit, C1)
