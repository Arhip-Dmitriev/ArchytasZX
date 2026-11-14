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

"""The boxed incrementer by induction through the lemma window (77.5)."""

from __future__ import annotations

import pytest

from archytaszx.diagram.bangbox import index_owners, peel_owners, peels_first
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics import lemmas as L
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import EqualityVerdict, decide_equal
from archytaszx.semantics.prove import induction_renamed

SPEC = """
count n
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
s = X(1 -> 1, literal 2^n) phase{k: -k/2^n}
t = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
wire h.out0 -- b.in1, b.out0 -- s.in0, s.out0 -- t.in0, t.out1 -- e.in0
repeat n as V index m: b
chain V: b.out0 -> b.in1
repeat n as U index m: t
chain U: t.in0 -> t.out1
inputs b.in0
outputs t.out0
"""

CIRCUIT = """
count n
g0 = Z(1 -> 2, literal 2)
g1 = X(2 -> 1, literal 2)
g2 = Z(1 -> 1, literal 2)
g3 = Ti(literal 2)
g4 = T(literal 2)
wire g0.out0 -- g1.in0, g2.out0 -- g3.in0, g3.out0 -- g1.in1, g0.out1 -- g4.in0, g4.out0 -- g2.in0
repeat n as C0 index j: g0, g1, g2, g3
pairs P0 of C0 index m: g4
left P0: g0.out1
right P0: g2.in0
inputs g0.in0
outputs g1.out0
scalar 2^(n/2)
"""


def test_the_incrementer_is_equal_by_induction_with_a_kernel_theorem() -> None:
    left, right = parse_dsl(CIRCUIT), parse_dsl(SPEC)
    decision = decide_equal(left, right, samples=[])
    assert decision.verdict is EqualityVerdict.EQUAL
    theorem = kernel.from_decision(decision, left, right)
    assert isinstance(theorem, kernel.Theorem)


@pytest.mark.parametrize(
    ("old", "new"),
    [("phase{k: -k/2^n}", "phase{k: k/2^n}"), ("g3 = Ti(", "g3 = T(")],
)
def test_a_broken_incrementer_is_not_proved(old: str, new: str) -> None:
    left, right = parse_dsl(CIRCUIT.replace(old, new)), parse_dsl(SPEC.replace(old, new))
    assert decide_equal(left, right, samples=[]).verdict is not EqualityVerdict.EQUAL


@pytest.mark.parametrize("name", ["zero_register", "and_gated", "and_split"])
def test_a_coined_walk_lemma_is_proved_with_a_kernel_theorem(name: str) -> None:
    entry = L.lemma(name)
    left, right = L.sides(entry)
    for value in (0, 1, 2):
        assert compare(left, right, {"p": value}).matched
    assert isinstance(L.proved(entry), kernel.Theorem)


@pytest.mark.parametrize(
    ("text", "old", "new"),
    [
        ("_ZERO_RIGHT", "zr = X(0 -> 1,", "zr = Z(0 -> 1,"),
        ("_SPLIT_RIGHT", "tr = T(", "tr = Ti("),
        ("_GATED_RIGHT", "scalar -1/sqrt(2)", "scalar 1/sqrt(2)"),
    ],
)
def test_a_broken_coined_walk_lemma_is_not_proved(text: str, old: str, new: str) -> None:
    left_text = {"_ZERO_RIGHT": L._ZERO_LEFT, "_SPLIT_RIGHT": L._SPLIT_LEFT}.get(
        text, L._GATED_LEFT
    )
    right = parse_dsl(getattr(L, text).replace(old, new))
    verdict = decide_equal(parse_dsl(left_text), right, samples=[]).verdict
    assert verdict is not EqualityVerdict.EQUAL


def test_the_spec_peeled_at_either_end_of_its_splitter_chain_is_one_diagram() -> None:
    spec = parse_dsl(SPEC)
    source = induction_renamed(spec, "n", "k")
    firsts = frozenset(b for b in index_owners(source, "k") if peels_first(source.bang_boxes[b]))
    assert firsts
    theorem = kernel.repeel(source, "k", frozenset(), firsts)
    assert kernel._same(theorem.lhs, peel_owners(source, "k").diagram)
    for value in (0, 1, 2):
        assert compare(theorem.lhs, theorem.rhs, {"k": value}).matched
        assert compare(theorem.rhs, spec, {"k": value, "n": value + 1}).matched
