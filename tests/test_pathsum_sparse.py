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

"""Sparse phase-vector reads of pathsum_diagram against the dense table."""

from __future__ import annotations

import random
import types
from fractions import Fraction

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.pathsum import PhasePoly
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.semantics import pathsum_diagram as P


def _node(k: int, entries: dict[int, Phase]) -> object:
    return types.SimpleNamespace(phase=PhaseVector(Dim(1 << k), entries))


def test_sparse_read_is_the_dense_interpolation() -> None:
    rng = random.Random(1)
    for _ in range(400):
        k = rng.randint(1, 6)
        m = rng.choice([1, 1, 2, 3, rng.randint(0, (1 << k) - 1)])
        idx = rng.sample(range(1, 1 << k), min(m, (1 << k) - 1))
        ent = {
            v: Phase.turns(sp.Rational(rng.randint(-20, 20), rng.choice([1, 2, 3, 8, 7, 1024])))
            for v in idx
        }
        node = _node(k, ent)
        digits = rng.sample(range(100), k)
        a, b = PhasePoly(), PhasePoly()
        for poly in (a, b):
            poly.add((digits[0],), Fraction(1, 8))
        P._phase_terms(a, digits, "n", node)  # type: ignore[arg-type]
        P._interpolate(b, digits, P._turns_table("n", node, 1 << k))  # type: ignore[arg-type]
        assert list(a.terms.items()) == list(b.terms.items())


def test_one_entry_on_many_digits_is_one_monomial() -> None:
    k = 31
    node = _node(k, {(1 << k) - 1: Phase.turns(sp.Rational(1, 2))})
    poly = PhasePoly()
    P._phase_terms(poly, list(range(k)), "n", node)  # type: ignore[arg-type]
    assert poly.terms == {frozenset(range(k)): Fraction(1, 2)}


def test_sparse_cyclotomic_reduction_is_the_dense_one() -> None:
    from archytaszx.algebra.pathsum import Cyclo

    rng = random.Random(3)
    ns = [*range(1, 61), 64, 81, 125, 210, 128]
    for _ in range(300):
        n = rng.choice(ns)
        p = next(q for q in range(2, n + 1) if n % q == 0) if n > 1 else 1
        x = Cyclo(n, {rng.randrange(n): Fraction(rng.randint(-3, 3)) for _ in range(3)})
        if p > 1 and rng.random() < 0.5:
            k, c = rng.randrange(n), Fraction(rng.randint(1, 4))
            x = x + Cyclo(n, {k + j * (n // p): c for j in range(p)})
        assert x.is_zero == (not any(x.reduced()))
        y = x + Cyclo(n, {0: Fraction(rng.randint(0, 1))})
        assert (x == y) == (x.reduced() == y.reduced())


def test_a_root_of_huge_order_is_compared_without_its_cyclotomic() -> None:
    from archytaszx.algebra.pathsum import Cyclo

    n = 1400**3
    one = Cyclo(n, {0: Fraction(1)})
    assert (one + Cyclo(1400, {700: Fraction(1)})).is_zero
    assert Cyclo.root(Fraction(1, n)) != Cyclo.root(Fraction(2, n))
    assert Cyclo.root(Fraction(3, n)) == Cyclo.root(Fraction(3, n) + 1)


def test_a_root_of_large_squarefree_order_is_compared_by_prime_coordinates() -> None:
    """At n = 4 (2^16 - 1) (radical 2 3 5 17 257) equality reads per-prime coordinates
    instead of reducing modulo Phi_131070 (dqc1/main! missing_gate n = 16)."""
    from archytaszx.algebra.pathsum import Cyclo

    n = 4 * 65535
    x = Cyclo(n, {5: Fraction(1), 17: Fraction(-2), 1000: Fraction(1, 3)})
    assert x == x + Cyclo(257, {j: Fraction(1) for j in range(257)})
    assert x != x + Cyclo(257, {j: Fraction(1) for j in range(256)})
    assert Cyclo(n, {k * (n // 3): Fraction(1) for k in range(3)}).is_zero


def _substitute_by_adds(p: PhasePoly, var: int, value: frozenset[frozenset[int]]) -> None:
    from archytaszx.algebra.pathsum import _mono_key, _xor_lift

    hit = [(m, c) for m, c in p.terms.items() if var in m]
    for m, _c in hit:
        del p.terms[m]
    if not hit:
        return
    lift = _xor_lift(sorted(value, key=_mono_key), 840)
    for m, c in hit:
        for mono, k in lift.items():
            if c.numerator * k % c.denominator:
                p.add((m - {var}) | mono, Fraction(c.numerator * k % c.denominator, c.denominator))


def test_substitute_matches_term_by_term_adds_in_value_and_order() -> None:
    rng = random.Random(7)
    dens = (2, 4, 8, 3, 6)
    for _ in range(300):
        p = PhasePoly()
        for _t in range(rng.randint(1, 25)):
            mono = frozenset(rng.sample(range(6), rng.randint(0, 3)))
            p.add(mono, Fraction(rng.randint(1, 7), rng.choice(dens)))
        var = rng.randrange(6)
        value = frozenset(
            frozenset(rng.sample([v for v in range(6) if v != var], rng.randint(0, 2)))
            for _m in range(rng.randint(1, 4))
        )
        want = p.copy()
        _substitute_by_adds(want, var, value)
        p.substitute(var, value)
        assert list(p.terms.items()) == list(want.terms.items())


def test_xor_lift_is_the_frozenset_fold_in_order() -> None:
    from archytaszx.algebra.pathsum import _xor_lift

    rng = random.Random(11)
    for _ in range(200):
        items = [frozenset(rng.sample(range(40, 48), rng.randint(0, 3))) for _m in range(5)]
        modulus = rng.choice((2, 4, 8, 6, 512))
        want: dict[frozenset[int], int] = {}
        for m in items:
            prev = [(k, 2 * a % modulus) for k, a in want.items() if 2 * a % modulus]
            want[m] = (want.get(m, 0) + 1) % modulus
            for k, b in prev:
                want[k | m] = (want.get(k | m, 0) - b) % modulus
        got = _xor_lift(items, modulus)
        assert list(got.items()) == [(k, a) for k, a in want.items() if a]


def test_xor_lift_array_fold_is_the_dict_fold_in_order() -> None:
    from archytaszx.algebra.pathsum import LIFT_ARRAY_ITEMS, _mono_key, _xor_lift_masks

    rng = random.Random(19)
    for _ in range(60):
        nv = rng.randint(4, 12)
        items = [
            frozenset(rng.sample(range(nv), rng.randint(0, min(nv, 4))))
            for _m in range(rng.randint(LIFT_ARRAY_ITEMS, 2 * LIFT_ARRAY_ITEMS))
        ]
        if rng.random() < 0.7:
            items = sorted(set(items), key=_mono_key)
        modulus = rng.choice((2, 4, 8, 16, 6, 12, 840))
        names = sorted({v for m in items for v in m})
        want: dict[int, int] = {}
        for m in items:
            mask = sum(1 << names.index(v) for v in m)
            prev = [(k, 2 * a % modulus) for k, a in want.items() if 2 * a % modulus]
            want[mask] = (want.get(mask, 0) + 1) % modulus
            for k, b in prev:
                want[k | mask] = (want.get(k | mask, 0) - b) % modulus
        got_names, got = _xor_lift_masks(items, modulus)
        assert got_names == names
        assert list(got.items()) == list(want.items())


def test_table_decode_is_the_bit_decode_in_order() -> None:
    from archytaszx.algebra.pathsum import _decode, _decode_all

    rng = random.Random(37)
    for _ in range(12):
        count = rng.randint(1, 62)
        names = sorted(rng.sample(range(500), count))
        raw = {rng.getrandbits(count): rng.randint(0, 3) for _k in range(rng.choice((10, 2000)))}
        want = [(tuple(_decode(k, names)), k, a) for k, a in raw.items() if a]
        assert [(tuple(m), k, a) for m, k, a in _decode_all(raw, names)] == want


def test_substitute_array_path_is_the_dict_loop_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from archytaszx.algebra import pathsum

    rng = random.Random(29)
    for _ in range(400):
        nv = rng.randint(2, 9)
        p = PhasePoly()
        for _t in range(rng.randint(1, 60)):
            mono = frozenset(rng.sample(range(nv), rng.randint(0, min(nv, 4))))
            p.add(mono, Fraction(rng.randint(1, 15), rng.choice((2, 4, 8, 3, 6, 16))))
        var = rng.randrange(nv)
        value = frozenset(
            frozenset(rng.sample([v for v in range(nv + 3) if v != var], rng.randint(0, 3)))
            for _m in range(rng.randint(1, 8))
        )
        want, got = p.copy(), p.copy()
        monkeypatch.setattr(pathsum, "SUBSTITUTE_ARRAY_PAIRS", 1 << 40)
        want.substitute(var, value)
        monkeypatch.setattr(pathsum, "SUBSTITUTE_ARRAY_PAIRS", 0)
        monkeypatch.setattr(pathsum, "SUBSTITUTE_CHUNK", rng.choice((1, 3, 7, 1 << 21)))
        got.substitute(var, value)
        assert list(got.terms.items()) == list(want.terms.items())


def test_occurrences_on_the_path_are_the_full_ones() -> None:
    from archytaszx.algebra.pathsum import _occurrences

    rng = random.Random(13)
    p = PhasePoly()
    for _t in range(60):
        p.add(frozenset(rng.sample(range(10), rng.randint(1, 3))), Fraction(rng.randint(1, 7), 8))
    path = {1, 4, 7}
    full, some = _occurrences(p), _occurrences(p, path)
    assert some == {v: full[v] for v in full if v in path}
