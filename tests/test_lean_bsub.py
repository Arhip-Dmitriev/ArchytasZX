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

"""[HH] on the top digit of a dyadic group with a boundary-only low bit, then [BSub]
(Step.bsub, bsub_sound): the addition sum out = a + b mod 4 and the kernel path_sum export of
carry_shift at p = 3."""

from __future__ import annotations

import os
from fractions import Fraction
from itertools import product

import pytest
import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra import pathsum as P
from archytaszx.semantics import kernel
from archytaszx.semantics import lean_net as L
from archytaszx.semantics import lemmas as LM


def _adder() -> P.PathSum:
    """Sum_Y exp(2 pi i Y S / 4), Y = y6 + 2 y7, S = o0 + 2 o1 - a0 - 2 a1 - b0 - 2 b1 on the
    boundary o0, o1, a0, a1, b0, b1 = 0..5."""
    phase = P.PhasePoly()
    for v, w in ((0, 1), (1, 2), (2, -1), (3, -2), (4, -1), (5, -2)):
        phase.add({6, v}, Fraction(w, 4))
        phase.add({7, v}, Fraction(w, 2))
    return P.PathSum((0, 1, 2, 3, 4, 5), {6, 7}, phase, P.Cyclo.rational(1))


def test_the_adder_sum_takes_bsub_steps_and_no_dyadic_step() -> None:
    red = P.reduce(_adder())
    assert [s.rule for s in red.trace] == ["hh_constraint", "bsub", "hh_constraint"]
    assert red.trace[1].var == 0
    assert red.trace[1].value == frozenset({frozenset({2}), frozenset({4})})


def test_the_adder_canonical_form_is_addition_mod_4() -> None:
    canon = P.canonical(P.reduce(_adder()))
    assert canon is not None
    for a0, a1, b0, b1 in product((0, 1), repeat=4):
        s = a0 + 2 * a1 + b0 + 2 * b1
        point = canon.complete({2: a0, 3: a1, 4: b0, 5: b1})
        assert point[0] + 2 * point[1] == s % 4
        assert canon.value(point) == P.Cyclo.rational(4)


def _source() -> str:
    theorem = LM.instance(LM.CARRY_SHIFT, sp.Integer(3))
    assert isinstance(theorem, kernel.Theorem)
    return kernel.lean(theorem, "p3")


def test_the_carry_shift_export_takes_bsub_steps() -> None:
    source = _source()
    assert ".hhCons, .bsub 1, " in source
    assert ".dyadicB" not in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_checks_the_carry_shift_bsub_export(tmp_path: object) -> None:
    ok, output = L.check(_source(), f"{tmp_path}/p3.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_rejects_a_tampered_bsub_step(tmp_path: object) -> None:
    """The first substitution of boundary variable 1 moved to variable 2 fails runOK."""
    source = _source()
    tampered = source.replace(".hhCons, .bsub 1, ", ".hhCons, .bsub 2, ", 1)
    assert tampered != source
    ok, _output = L.check(tampered, f"{tmp_path}/p3bad.lean")
    assert not ok
