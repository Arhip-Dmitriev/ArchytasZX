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

"""Quantum Hamming [[15,7,3]] code, the CSS code of the [15,11,3] Hamming code with itself: X
and Z checks the four rows of the Hamming parity-check matrix; the standard-form encoder against
the CSS normal form."""

from benchmarks.subjects._stab import code_subject, css_generators

_H = [[1 if (j + 1) >> i & 1 else 0 for j in range(15)] for i in range(4)]

SUBJECT = code_subject(
    id="quantum_hamming_code",
    name="Quantum Hamming [[15,7,3]] code",
    source="steane1996simple",
    generators=css_generators(_H, _H),
)
