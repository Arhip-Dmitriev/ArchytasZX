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

"""X-cube fracton model on the 2 x 2 x 2 cubic torus, qubits on edges: X on the twelve edges of
every cube, Z on the four edges at a site in each coordinate plane, [[24,9,2]]."""

import itertools

from benchmarks.subjects._lattice import Cubic
from benchmarks.subjects._stab import code_subject, independent, pauli_string

_C = Cubic(2)


def _cube(p):
    """The twelve edges of the cube at corner p."""
    out = []
    for d in range(3):
        others = [k for k in range(3) if k != d]
        for s in itertools.product((0, 1), repeat=2):
            q = list(p)
            for k, v in zip(others, s):
                q[k] += v
            out.append(_C.edge(tuple(q), d))
    return out


_CUBES = [pauli_string(_C.n, [(e, "X") for e in _cube(p)]) for p in _C.sites]
_CROSSES = [
    pauli_string(
        _C.n,
        [(_C.edge(q, d), "Z") for d in range(3) if d != mu for q in (p, _C.step(p, d, -1))],
    )
    for p in _C.sites
    for mu in range(3)
]

SUBJECT = code_subject(
    id="x_cube_code",
    name="X-cube fracton code",
    source="vijay2016fracton",
    generators=independent(_CUBES + _CROSSES),
)
