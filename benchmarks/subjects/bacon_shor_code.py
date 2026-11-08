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

"""Bacon-Shor [[9,1,3]] subsystem code on a 3 x 3 grid: stabilizers X on adjacent row pairs and
Z on adjacent column pairs; the encoder takes the logical qubit and the four gauge qubits as
inputs; the standard-form encoder against the CSS normal form."""

from benchmarks.subjects._stab import code_subject, css_generators, support_rows

_ROWS = [[3 * r + c for r in (a, a + 1) for c in range(3)] for a in range(2)]
_COLS = [[3 * r + c for c in (a, a + 1) for r in range(3)] for a in range(2)]

SUBJECT = code_subject(
    id="bacon_shor_code",
    name="Bacon-Shor subsystem code",
    source="bacon2006operator",
    generators=css_generators(support_rows(9, _ROWS), support_rows(9, _COLS)),
)
