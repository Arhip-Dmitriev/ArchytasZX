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

"""XY surface code of distance 3: the rotated surface code with every Z replaced by Y; the
checks are X- and Y-type; the standard-form encoder against the projector form."""

from benchmarks.subjects._stab import code_subject, css_generators, support_rows
from benchmarks.subjects.rotated_surface_code import X_CHECKS, Z_CHECKS

_GENS = [
    g.replace("Z", "Y")
    for g in css_generators(support_rows(9, X_CHECKS), support_rows(9, Z_CHECKS))
]

SUBJECT = code_subject(
    id="xy_surface_code",
    name="XY surface code, distance 3",
    source="tuckett2018ultrahigh",
    generators=_GENS,
)
