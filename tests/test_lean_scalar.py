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

"""Ideal membership with cofactors in lean_scalar._lift."""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.semantics.lean_scalar import _lift


def _cyclic(n: int) -> tuple[list[sp.Symbol], list[sp.Expr]]:
    """The cyclic-n polynomial system."""
    xs = list(sp.symbols(f"x0:{n}"))
    rels = [
        sp.Add(*(sp.Mul(*(xs[(i + j) % n] for j in range(k))) for i in range(n)))
        for k in range(1, n)
    ]
    return xs, [*rels, sp.Mul(*xs) - 1]


def _top_basis_element(xs: list[sp.Symbol], rels: list[sp.Expr]) -> sp.Expr:
    """The highest-degree element of the reduced grevlex basis."""
    basis = sp.groebner(rels, *xs, order="grevlex").exprs
    return max(basis, key=lambda e: sp.Poly(e, *xs).total_degree())


def test_cyclic5_basis_element_lifts_to_cofactors() -> None:
    xs, rels = _cyclic(5)
    target = _top_basis_element(xs, rels)
    cofactors = _lift(target, rels, xs)
    assert cofactors is not None
    assert sp.expand(sum(c * r for c, r in zip(cofactors, rels, strict=True)) - target) == 0


def test_cyclic4_non_member_is_refused() -> None:
    xs, rels = _cyclic(4)
    assert _lift(xs[0], rels, xs) is None
    assert _lift(sp.Integer(1), rels, xs) is None
