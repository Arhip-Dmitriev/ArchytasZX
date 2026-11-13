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

"""The stated-lemma library: the carry shift lemma, its instances and its use as a rewrite."""

from __future__ import annotations

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics import lemmas as L
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

_CONTEXT = """
z = Z(1 -> 1, literal 2) phase{1: 1/4}
b = B(literal 2, literal 2^k)
x = X(1 -> 1, literal 2^(k + 1)) phase{j: -j/2^(k + 1)}
s = S(literal 2, literal 2^k)
w = X(1 -> 1, literal 2^k) phase{j: j/2^k}
wire z.out0 -- b.in0, b.out0 -- x.in0, x.out0 -- s.in0, s.out1 -- w.in0
inputs z.in0, b.in1
outputs s.out0, w.out0
"""


def test_the_carry_shift_lemma_is_proved_with_its_symbol_formal() -> None:
    theorem = L.proved(L.CARRY_SHIFT)
    assert isinstance(theorem, kernel.Theorem)
    assert theorem.inference.kind == "contraction"


@pytest.mark.parametrize("value", ["k", "k + 1"])
def test_an_instance_is_a_kernel_theorem(value: str) -> None:
    k = Dim.symbol("k").to_sympy()
    theorem = L.instance(L.CARRY_SHIFT, k if value == "k" else k + 1)
    assert isinstance(theorem, kernel.Theorem)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.set_scalar(Scalar.one()),
        lambda d: d.set_boundary_outputs(list(reversed(d.boundary_outputs))),
    ],
)
def test_a_broken_right_side_is_not_proved(edit: object) -> None:
    left, right = L.sides(L.CARRY_SHIFT)
    edit(right)  # type: ignore[operator]
    assert decide_equal(left, right, samples=[]).verdict is not EqualityVerdict.EQUAL


def test_the_lemma_rewrites_its_left_side_inside_a_diagram() -> None:
    context = parse_dsl(_CONTEXT)
    theorem = L.rewrite(context, L.CARRY_SHIFT)
    assert isinstance(theorem, kernel.Theorem)
    assert theorem.inference.kind == "congr"
    assert len(theorem.rhs.nodes) == 12
    for k in (1, 2):
        assert compare(theorem.lhs, theorem.rhs, {"k": k}).matched


def test_a_shift_by_minus_one_is_not_rewritten() -> None:
    broken = parse_dsl(_CONTEXT.replace("phase{j: -j/2^(k + 1)}", "phase{j: j/2^(k + 1)}"))
    assert L.rewrite(broken, L.CARRY_SHIFT) is None


def test_the_and_register_lemma_is_proved_by_induction() -> None:
    theorem = L.proved(L.AND_REGISTER)
    assert isinstance(theorem, kernel.Theorem)
    assert theorem.inference.kind == "trans"
    left, right = L.sides(L.AND_REGISTER)
    for p in (1, 2, 3):
        assert compare(left, right, {"p": p}).matched


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("phase{k: k/2^(p + 1)}", "phase{k: -k/2^(p + 1)}"),
        ("tr = T(", "tr = Ti("),
        ("scalar -1/sqrt(2)", "scalar 1/sqrt(2)"),
        ("weight{k: -1}", "weight{k: I}"),
    ],
)
def test_a_broken_and_register_right_side_is_not_proved(old: str, new: str) -> None:
    left = parse_dsl(L._AND_LEFT)
    right = parse_dsl(L._AND_RIGHT.replace(old, new))
    assert decide_equal(left, right, samples=[]).verdict is not EqualityVerdict.EQUAL
