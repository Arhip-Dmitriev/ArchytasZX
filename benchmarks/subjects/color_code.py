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

"""Two-dimensional colour code, the 6.6.6 lattice on the 3 x 3 honeycomb torus: X and Z on every
hexagon, [[18,4,4]]."""

from benchmarks.subjects._lattice import Honeycomb
from benchmarks.subjects._stab import code_subject, independent, pauli_string

_H = Honeycomb(3)
_GENS = [pauli_string(_H.n, [(q, p) for q in h]) for p in "XZ" for h in _H.hexes.values()]

SUBJECT = code_subject(
    id="color_code",
    name="Colour code (6.6.6, torus)",
    source="bombin2006topological",
    generators=independent(_GENS),
)
