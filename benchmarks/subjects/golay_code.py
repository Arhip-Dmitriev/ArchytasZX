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

"""Quantum Golay [[23,1,7]] code, the CSS code of the [23,12,7] Golay code: X and Z checks the
cyclic shifts of g(x)(1 + x), g(x) = 1 + x^2 + x^4 + x^5 + x^6 + x^10 + x^11; the standard-form
encoder against the CSS normal form."""

from benchmarks.subjects._stab import code_subject, css_generators

_G = [0, 2, 4, 5, 6, 10, 11]
_EVEN = sorted(set(_G) ^ {e + 1 for e in _G})
_ROWS = [[1 if (j - s) in _EVEN else 0 for j in range(23)] for s in range(11)]

SUBJECT = code_subject(
    id="golay_code",
    name="Quantum Golay [[23,1,7]] code",
    source="steane2003overhead",
    generators=css_generators(_ROWS, _ROWS),
)
