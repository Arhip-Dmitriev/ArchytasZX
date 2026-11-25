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

"""The Cooley-Tukey lemma: the shift by one over s*t between B(s, t) and S(s, t) in digit-wise
form (docs/manual, kernel chapter)."""

from __future__ import annotations

from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.bangbox import Mult
from archytaszx.semantics import kernel, lemmas


def test_proved_with_both_dims_formal() -> None:
    theorem = cast(kernel.Theorem, lemmas.proved(lemmas.COOLEY_TUKEY))
    left, right = lemmas.sides(lemmas.COOLEY_TUKEY)
    assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)


def test_instance_at_symbolic_sizes() -> None:
    d = Dim.symbol("d").to_sympy()
    n = Mult.symbol("n").to_sympy()
    for value in (sp.Tuple(2, 2**n), sp.Tuple(d, d**n), sp.Tuple(d**n, d)):
        theorem = cast(kernel.Theorem, lemmas.instance(lemmas.COOLEY_TUKEY, value))
        left, right = lemmas.sides(lemmas.COOLEY_TUKEY, value)
        assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)


def test_rewrite_finds_the_binder() -> None:
    d = Dim.symbol("d").to_sympy()
    left, right = lemmas.sides(lemmas.COOLEY_TUKEY, sp.Tuple(3, d))
    theorem = cast(kernel.Theorem, lemmas.rewrite(left, lemmas.COOLEY_TUKEY))
    kernel.iso(theorem.rhs, right)


def test_shift_by_two_proved_and_instances() -> None:
    d = Dim.symbol("d").to_sympy()
    n = Mult.symbol("n").to_sympy()
    for value in (None, sp.Tuple(2, 2**n), sp.Tuple(d, d**n)):
        theorem = cast(
            kernel.Theorem,
            lemmas.proved(lemmas.COOLEY_TUKEY_2)
            if value is None
            else lemmas.instance(lemmas.COOLEY_TUKEY_2, value),
        )
        left, right = lemmas.sides(lemmas.COOLEY_TUKEY_2, value)
        assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)


def test_shift_by_two_rewrite_finds_the_binder() -> None:
    d = Dim.symbol("d").to_sympy()
    n = Mult.symbol("n").to_sympy()
    left, right = lemmas.sides(lemmas.COOLEY_TUKEY_2, sp.Tuple(d, d**n))
    theorem = cast(kernel.Theorem, lemmas.rewrite(left, lemmas.COOLEY_TUKEY_2))
    kernel.iso(theorem.rhs, right)
    assert lemmas.rewrite(left, lemmas.COOLEY_TUKEY) is None


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
