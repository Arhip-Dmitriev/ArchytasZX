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

"""[Turn]: a path variable with phase terms y S / 2^k (k >= 3), S off the path, summed into one
atom per value of S, in Python and replayed through AStep.turn in Lean."""

from __future__ import annotations

import cmath
import os
from fractions import Fraction

import pytest

from archytaszx.algebra.pathsum import (
    Cyclo,
    PathSum,
    PhasePoly,
    canonical,
    reduce,
    turn_digits,
)
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.lean_pathsum import export_proof

X8 = """a = X(1 -> 1, literal 2) phase{1: 1/8}
inputs a.in0
outputs a.out0"""
"""An X spider with phase pi / 4."""

X8_SPLIT = """a = X(1 -> 1, literal 2) phase{1: 7/8}
b = X(1 -> 1, literal 2) phase{1: 1/4}
wire a.out0 -- b.in0
inputs a.in0
outputs b.out0"""
"""X spiders with phases 7 pi / 4 and pi / 2 in sequence."""

HZH = """h = F(literal 2)
z = Z(1 -> 1, literal 2) phase{1: 1/8}
g = F(literal 2)
wire h.out0 -- z.in0, z.out0 -- g.in0
inputs h.in0
outputs g.out0"""
"""H, a Z spider with phase pi / 4 and H."""

TURN_X8 = ".turn [[[]], [], [[0], [1]]] [(1, 0), (2, 1), (3, 2), (5, 3), (6, 4), (7, 5)]"
"""The [Turn] step of X8: S = 1 + 4 (x0 + x1)."""


def test_turn_digits_spell_s() -> None:
    s = {frozenset(): 5, frozenset({0}): 3, frozenset({0, 1}): 6}
    bs = turn_digits(s, 3)
    for x0 in (0, 1):
        for x1 in (0, 1):
            want = (5 + 3 * x0 + 6 * x0 * x1) % 8
            got = sum(
                (sum(all((x0, x1)[v] for v in m) for m in b) % 2) << i for i, b in enumerate(bs)
            )
            assert got == want


@pytest.mark.parametrize("k", [3, 4])
def test_turn_sums_a_variable_with_eighth_coefficients(k: int) -> None:
    """Sum_y exp(2 pi i y (1 + 3 x0 + 2 x1) / 2^k) at the four (x0, x1)."""
    den = 1 << k
    phase = PhasePoly(
        {
            frozenset({2}): Fraction(1, den),
            frozenset({0, 2}): Fraction(3, den),
            frozenset({1, 2}): Fraction(2, den),
        }
    )
    red = reduce(PathSum((0, 1), {2}, phase, Cyclo.rational(1)))
    assert [st.rule for st in red.trace] == ["turn"]
    canon = canonical(red)
    assert canon is not None
    for x0 in (0, 1):
        for x1 in (0, 1):
            want = 1 + cmath.exp(2j * cmath.pi * (1 + 3 * x0 + 2 * x1) / den)
            assert abs(canon.value({0: x0, 1: x1}).to_complex() - want) < 1e-9


def test_turn_exports_an_astep() -> None:
    source = export_proof(parse_dsl(X8), parse_dsl(X8_SPLIT), "x8")
    assert TURN_X8 in source
    assert "theorem qreqsL : ∀ p ∈ aqreqs 8 stepsL, 2 * A p.2 = 1 + om 8 ^ p.1" in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize("right", [X8_SPLIT, HZH])
def test_lean_checks_a_turn_pair(right: str, tmp_path: object) -> None:
    ok, output = L.check(export_proof(parse_dsl(X8), parse_dsl(right), "x8"), f"{tmp_path}/x8.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_rejects_a_tampered_turn_step(tmp_path: object) -> None:
    """The low digit of the [Turn] step set to 0 instead of 1 fails turnOK."""
    source = export_proof(parse_dsl(X8), parse_dsl(X8_SPLIT), "x8")
    tampered = source.replace(TURN_X8, TURN_X8.replace(".turn [[[]], ", ".turn [[], ", 1))
    assert tampered != source
    ok, _output = L.check(tampered, f"{tmp_path}/x8bad.lean")
    assert not ok
