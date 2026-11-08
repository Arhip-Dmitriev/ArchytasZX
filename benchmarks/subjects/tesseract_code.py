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

"""Tesseract [[16,6,4]] code, the 4D hypercube color code: X and Z checks the all-ones row and
the four coordinate rows of RM(1,4); the standard-form encoder against the CSS normal form."""

from benchmarks.subjects._stab import code_subject, css_generators

_RM = [[1] * 16] + [[v >> i & 1 for v in range(16)] for i in range(4)]

SUBJECT = code_subject(
    id="tesseract_code",
    name="Tesseract [[16,6,4]] code",
    source="reichardt2024demonstration",
    generators=css_generators(_RM, _RM),
)
