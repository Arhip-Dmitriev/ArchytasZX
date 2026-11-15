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
