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

"""Haah's cubic code on the 2 x 2 x 2 torus, two qubits per site: per cube an X term with
polynomials (1 + xy + yz + zx, 1 + x + y + z) and a Z term with the conjugate polynomials
(1 + x^-1 + y^-1 + z^-1, 1 + x^-1 y^-1 + y^-1 z^-1 + z^-1 x^-1), [[16,6,2]]."""

import itertools

from benchmarks.subjects._stab import code_subject, independent, pauli_string

_L = 2
_N = 2 * _L**3
_P1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
_P2 = [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]


def _q(p, m, i):
    """Qubit i at site p + m."""
    x, y, z = ((a + b) % _L for a, b in zip(p, m))
    return 2 * ((x * _L + y) * _L + z) + i


def _neg(ms):
    return [tuple(-c for c in m) for m in ms]


_SITES = list(itertools.product(range(_L), repeat=3))
_GX = [
    pauli_string(_N, [(_q(p, m, 0), "X") for m in _P2] + [(_q(p, m, 1), "X") for m in _P1])
    for p in _SITES
]
_GZ = [
    pauli_string(
        _N, [(_q(p, m, 0), "Z") for m in _neg(_P1)] + [(_q(p, m, 1), "Z") for m in _neg(_P2)]
    )
    for p in _SITES
]

SUBJECT = code_subject(
    id="haah_code",
    name="Haah's cubic code",
    source="haah2011local",
    generators=independent(_GX + _GZ),
)
