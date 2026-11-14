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

"""Pivot rows of F_2 constraints (contract_symbolic._solved_rows)."""

from __future__ import annotations

import itertools
import random

from archytaszx.algebra.pathsum import _f2_eval
from archytaszx.semantics.contract_symbolic import _solved_rows

Poly = frozenset[frozenset[int]]


def _poly(*monos: tuple[int, ...]) -> Poly:
    return frozenset(frozenset(m) for m in monos)


def test_two_forms_of_one_adder_support_give_one_row_set() -> None:
    """The 2-digit adder constraints of the circuit and of the specification (draper_adder n = 2)
    differ by a multiple of another constraint and solve to the same rows."""
    common = [_poly((0,), (4,)), _poly((1,), (5,)), _poly((1,), (3,), (7,))]
    circuit = [*common, _poly((0,), (1, 3), (1, 7), (2,), (3,), (3, 7), (6,))]
    spec = [*common, _poly((0,), (1,), (1, 3), (1, 7), (2,), (3, 7), (6,), (7,))]
    order = tuple(range(8))
    rows_c, rows_s = _solved_rows(circuit, order), _solved_rows(spec, order)
    assert rows_c is not None
    assert rows_c == rows_s
    assert sorted(rows_c) == [0, 1, 2, 3]


def test_rows_keep_the_support() -> None:
    random.seed(7)
    solved = 0
    for _ in range(300):
        nv = random.randint(2, 6)
        cons = [
            frozenset(
                frozenset(random.sample(range(nv), random.randint(0, 2)))
                for _ in range(random.randint(1, 4))
            )
            for _ in range(random.randint(1, 4))
        ]
        rows = _solved_rows(cons, tuple(range(nv)))
        if rows is None:
            continue
        solved += 1
        assert not any(v in rows for r in rows.values() for m in r for v in m)
        for p in itertools.product((0, 1), repeat=nv):
            val = dict(enumerate(p))
            inside = all(_f2_eval(c, val) == 0 for c in cons)
            assert inside == all(val[q] == _f2_eval(r, val) for q, r in rows.items())
    assert solved > 50


def test_constant_one_constraint_has_no_rows() -> None:
    assert _solved_rows([_poly((0,), (1,)), _poly((0,), (1,), ())], (0, 1)) is None
