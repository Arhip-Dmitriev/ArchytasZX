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

"""Rotated surface code of distance 3, [[9,1,3]] on a 3 x 3 grid: weight-4 plaquettes in a
checkerboard and weight-2 boundary checks; the standard-form encoder against the CSS normal
form, and one Y error corrected by syndrome lookup."""

from benchmarks.subjects._stab import code_subject, css_generators, support_rows

X_CHECKS = [[0, 1], [1, 2, 4, 5], [3, 4, 6, 7], [7, 8]]
Z_CHECKS = [[0, 1, 3, 4], [4, 5, 7, 8], [3, 6], [2, 5]]

SUBJECT = code_subject(
    id="rotated_surface_code",
    name="Rotated surface code, distance 3",
    source="horsman2012surface",
    generators=css_generators(support_rows(9, X_CHECKS), support_rows(9, Z_CHECKS)),
    correct=(4, "Y"),
)
