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

"""Mermin-Peres magic square game, quantum strategy on two Bell pairs (a1, b1), (a2, b2) with the
square rows (IZ, ZI, ZZ), (XI, IX, XX), (-XZ, -ZX, YY): Alice measures the two generators of her
row r, Bob those of his column c (row 3 and column 3 by a Bell-basis measurement), each derives
the third entry from the row product +1 and column product -1; the xor of the two values of the
shared cell is computed into a fresh bit, which is 0."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H

_ALICE = {
    1: ([], [(["a2"], 0), (["a1"], 0), (["a1", "a2"], 0)]),
    2: ([H("a1"), H("a2")], [(["a1"], 0), (["a2"], 0), (["a1", "a2"], 0)]),
    3: ([CZ("a1", "a2"), H("a1"), H("a2")], [(["a1"], 1), (["a2"], 1), (["a1", "a2"], 0)]),
}
_BOB = {
    1: ([H("b1")], [(["b2"], 0), (["b1"], 0), (["b1", "b2"], 1)]),
    2: ([H("b2")], [(["b1"], 0), (["b2"], 0), (["b1", "b2"], 1)]),
    3: ([CX("b1", "b2"), H("b1")], [(["b2"], 0), (["b1"], 0), (["b1", "b2"], 1)]),
}


def _game(r, c):
    def build(n, d):
        """Bell pairs, Alice's row r and Bob's column c measurements, the agreement bit w."""
        q = ["a1", "a2", "b1", "b2"]
        gates = [Init(x) for x in q]
        gates += [F("a1"), CX("a1", "b1"), F("a2"), CX("a2", "b2")]
        ga, ea = _ALICE[r]
        gb, eb = _BOB[c]
        gates += [*ga, *gb, *(Measure(x) for x in q), Init("w")]
        (wa, ka), (wb, kb) = ea[c - 1], eb[r - 1]
        gates += [CX(x, "w") for x in (*wa, *wb)]
        if ka ^ kb:
            gates.append(X("w"))
        gates += [Measure("w"), *(Discard(x) for x in q)]
        return circuit([*q, "w"], gates)

    return build


def _agree(c):
    """The bit w = 0."""
    return f"w = X(0 -> 1, {c.D})\nscalar 2^(-1/2)"


SUBJECT = Subject(
    id="magic_square_game",
    name="Mermin-Peres magic square game",
    category="protocol",
    source="mermin1990simple",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=tuple(
        Equation(f"r{r}c{c}", _game(r, c), _agree, form="fixed")
        for r, c in ((1, 3), (3, 1), (3, 3))
    ),
)
