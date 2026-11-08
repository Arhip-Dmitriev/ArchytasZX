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

"""XZZX surface code of distance 3: the rotated surface code with X and Z exchanged on the odd
sites of the checkerboard, every plaquette reading XZZX; the standard-form encoder against the
projector form."""

from benchmarks.subjects._stab import code_subject, css_generators, hadamard_conjugate, support_rows
from benchmarks.subjects.rotated_surface_code import X_CHECKS, Z_CHECKS

_ODD = [j for j in range(9) if (j // 3 + j % 3) % 2]

SUBJECT = code_subject(
    id="xzzx_surface_code",
    name="XZZX surface code, distance 3",
    source="bonillaataides2021xzzx",
    generators=hadamard_conjugate(
        css_generators(support_rows(9, X_CHECKS), support_rows(9, Z_CHECKS)), _ODD
    ),
)
