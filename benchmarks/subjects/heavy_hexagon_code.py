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

"""Heavy-hexagon [[9,1,3]] subsystem code: Z stabilizers the surface-code plaquettes and
vertical boundary pairs, X stabilizers the Bacon-Shor column pairs; the encoder takes the
logical and two gauge qubits as inputs; the standard-form encoder against the CSS normal form."""

from benchmarks.subjects._stab import code_subject, css_generators, support_rows

_X = [[3 * r + c for c in (a, a + 1) for r in range(3)] for a in range(2)]
_Z = [[0, 1, 3, 4], [4, 5, 7, 8], [3, 6], [2, 5]]

SUBJECT = code_subject(
    id="heavy_hexagon_code",
    name="Heavy-hexagon subsystem code",
    source="chamberland2020topological",
    generators=css_generators(support_rows(9, _X), support_rows(9, _Z)),
)
