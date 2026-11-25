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

"""The radix chain lemmas: pair rows through B and S chains, the top digit split off a chain
(PROOFS.txt 4.4)."""

from __future__ import annotations

from typing import cast

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra import scalar
from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.bangbox import Mult
from archytaszx.semantics import kernel, lemmas

NAMES = ["row_bind", "row_split", "top_bind", "top_split", "radix_add", "row_split_up"]
REWRITTEN = [name for name in NAMES if name != "radix_add"]


@pytest.mark.parametrize("name", NAMES)
def test_proved_for_every_p_with_a_kernel_theorem(name: str) -> None:
    entry = lemmas.lemma(name)
    assert entry in lemmas.LIBRARY
    theorem = cast(kernel.Theorem, lemmas.proved(entry))
    left, right = lemmas.sides(entry)
    assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)


@pytest.mark.parametrize("name", REWRITTEN)
def test_rewrite_finds_the_left_side(name: str) -> None:
    entry = lemmas.lemma(name)
    n = Mult.symbol("n").to_sympy()
    left, right = lemmas.sides(entry, n)
    theorem = cast(kernel.Theorem, lemmas.rewrite(left, entry))
    kernel.iso(theorem.rhs, right)


@pytest.mark.parametrize("shift", [0, 1])
def test_radix_add_instance_at_symbolic_p(shift: int) -> None:
    p = Mult.symbol("n").to_sympy() + shift
    theorem = cast(kernel.Theorem, lemmas.instance(lemmas.RADIX_ADD, p))
    left, right = lemmas.sides(lemmas.RADIX_ADD, p)
    assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)


def test_radix_moduli_take_symbolic_sub_monomials() -> None:
    d = Dim.symbol("d").to_sympy()
    n = Mult.symbol("n").to_sympy()
    assert d * d**n in scalar._radix_moduli(d**2 * d**n)


def test_sign_flipped_row_split_up_is_refused() -> None:
    entry = lemmas.ROW_SPLIT_UP
    flipped = lemmas.Lemma(
        name="row_split_up_flipped",
        symbol="p",
        statement="ROW_SPLIT_UP with the right pair phase sign flipped",
        build=lambda p: lemmas._dsl_sides(
            lemmas._ROW_SPLIT_LEFT.replace("-x*y", "x*y"), lemmas._ROW_SPLIT_RIGHT, p
        ),
        values=entry.values,
    )
    with pytest.raises((lemmas.LemmaError, kernel.KernelError)):
        lemmas.proved(flipped)


def test_bounds_take_symbolic_coefficients_of_known_sign() -> None:
    d = Dim.symbol("d").to_sympy()
    k = Mult.symbol("k").to_sympy()
    i2, i3 = (sp.Symbol(name, integer=True, nonnegative=True) for name in ("_i2", "_i3"))
    assert scalar._bounds(i2 + i3 * d, {i2: d, i3: d**k}) == (0, sp.expand(d * d**k - 1))


if __name__ == "__main__":
    for name in NAMES:
        test_proved_for_every_p_with_a_kernel_theorem(name)
        if name in REWRITTEN:
            test_rewrite_finds_the_left_side(name)
        print(name, "ok")
    test_radix_add_instance_at_symbolic_p(0)
    test_radix_add_instance_at_symbolic_p(1)
    test_radix_moduli_take_symbolic_sub_monomials()
    test_sign_flipped_row_split_up_is_refused()
    test_bounds_take_symbolic_coefficients_of_known_sign()
    print("instance, flipped, bounds ok")
