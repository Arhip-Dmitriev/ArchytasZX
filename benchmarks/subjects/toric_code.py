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

"""Kitaev toric code on the 2 x 2 torus, [[8,2,2]]: qubits on the eight edges, three independent
vertex stars X and three plaquettes Z; the standard-form encoder against the CSS normal form."""

from benchmarks.subjects._stab import code_subject, css_generators, support_rows

L = 2


def h(r, c):
    """The horizontal edge leaving vertex (r, c)."""
    return 2 * ((r % L) * L + c % L)


def v(r, c):
    """The vertical edge leaving vertex (r, c)."""
    return 2 * ((r % L) * L + c % L) + 1


STARS = [[h(r, c), h(r, c - 1), v(r, c), v(r - 1, c)] for r in range(L) for c in range(L)]
PLAQUETTES = [[h(r, c), h(r + 1, c), v(r, c), v(r, c + 1)] for r in range(L) for c in range(L)]

SUBJECT = code_subject(
    id="toric_code",
    name="Toric code, 2 x 2 torus",
    source="kitaev2003fault",
    generators=css_generators(support_rows(8, STARS[:-1]), support_rows(8, PLAQUETTES[:-1])),
)
