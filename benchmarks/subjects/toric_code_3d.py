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

"""Three-dimensional toric code on the 2 x 2 x 2 cubic torus, qubits on edges: an X star of six
edges per site, a Z plaquette per face, [[24,3,2]]."""

from benchmarks.subjects._lattice import Cubic
from benchmarks.subjects._stab import code_subject, independent, pauli_string

_C = Cubic(2)
_STARS = [
    pauli_string(_C.n, [(_C.edge(q, d), "X") for d in range(3) for q in (p, _C.step(p, d, -1))])
    for p in _C.sites
]
_FACES = [
    pauli_string(
        _C.n,
        [
            (_C.edge(p, a), "Z"),
            (_C.edge(_C.step(p, a), b), "Z"),
            (_C.edge(_C.step(p, b), a), "Z"),
            (_C.edge(p, b), "Z"),
        ],
    )
    for p in _C.sites
    for a, b in ((0, 1), (1, 2), (0, 2))
]

SUBJECT = code_subject(
    id="toric_code_3d",
    name="3D toric code",
    source="castelnovo2008topological",
    generators=independent(_STARS + _FACES),
)
