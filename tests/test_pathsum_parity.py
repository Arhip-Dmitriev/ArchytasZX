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

"""Canonical forms with unlifted parity terms (an XOR lift past XOR_LIFT_TERMS): values,
the phase difference and the witness of _compare_parity."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from archytaszx.algebra import pathsum as P
from archytaszx.semantics import pathsum_diagram as PD

_ROW = frozenset(
    {
        frozenset({0}),
        frozenset({1, 2}),
        frozenset({3, 4}),
        frozenset({5, 6}),
        frozenset({7, 8}),
    }
)


def _sum(c: Fraction) -> P.PathSum:
    """[x0 = x1 x2 + x3 x4 + x5 x6 + x7 x8] exp(2 pi i (c x0 + x1 / 4)) on the boundary 0..8."""
    phase = P.PhasePoly()
    if c:
        phase.add({0}, c)
    phase.add({1}, Fraction(1, 4))
    return P.PathSum(tuple(range(9)), set(), phase, P.Cyclo.rational(1), constraints=[_ROW])


def test_substitute_past_the_budget_leaves_the_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    phase = _sum(Fraction(1, 512)).phase
    before = dict(phase.terms)
    monkeypatch.setattr(P, "XOR_LIFT_TERMS", 4)
    with pytest.raises(P.PathSumUnsupported):
        phase.substitute(0, _ROW - {frozenset({0})})
    assert phase.terms == before


def test_parity_terms_keep_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    full = P.canonical(_sum(Fraction(1, 512)))
    assert full is not None and not full.parity
    monkeypatch.setattr(P, "XOR_LIFT_TERMS", 4)
    with pytest.raises(P.PathSumUnsupported):
        P.canonical(_sum(Fraction(1, 512)))
    held = P.canonical(_sum(Fraction(1, 512)), parity=True)
    assert held is not None
    assert held.parity == ((((0,),), Fraction(1, 512)),)
    fast = P.canon_evaluator(held)
    for bits in product((0, 1), repeat=8):
        point = held.complete(dict(zip(range(1, 9), bits, strict=True)))
        assert held.value(point) == full.value(point) == fast(point)
    monkeypatch.setattr(P, "XOR_LIFT_TERMS", 1 << 20)
    diff = P.parity_difference(held, full)
    assert diff is not None and not diff.terms


def test_compare_parity_finds_the_exact_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "XOR_LIFT_TERMS", 4)
    ps_l, ps_r = _sum(Fraction(1, 512)), _sum(Fraction(0))
    canon_l, canon_r = P.canonical(ps_l, parity=True), P.canonical(ps_r, parity=True)
    assert canon_l is not None and canon_r is not None
    assert P.parity_difference(canon_l, canon_r) is None
    found = PD._compare_parity(canon_l, canon_r, ps_l, ps_r, [1] * 9)
    assert found.status == "unequal"
    assert found.left is not None and found.right is not None
    assert found.left == found.right * P.Cyclo.root(Fraction(1, 512))
    same = PD._compare_parity(canon_l, canon_l, ps_l, ps_l, [1] * 9)
    assert same.status == "equal"
