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

"""Honeycomb Floquet code of Hastings and Haah on the 3 x 3 honeycomb torus, as the instantaneous
stabilizer group after a round of colour-0 checks: every Kitaev plaquette (the product of the six
edge checks around a hexagon, sign dropped) and every colour-0 edge check, [[18,2]]."""

from benchmarks.subjects._lattice import Honeycomb
from benchmarks.subjects._stab import code_subject, independent, pauli_string

_H = Honeycomb(3)
_PLAQUETTES = [
    pauli_string(_H.n, [(q, t) for u, v, t, _ in _H.edges if u in h and v in h for q in (u, v)])
    for h in _H.hexes.values()
]
_CHECKS = [pauli_string(_H.n, [(u, t), (v, t)]) for u, v, t, c in _H.edges if c == 0]

SUBJECT = code_subject(
    id="honeycomb_code",
    name="Honeycomb Floquet code (instantaneous stabilizer group)",
    source="hastings2021dynamically",
    generators=independent(_PLAQUETTES + _CHECKS),
)
