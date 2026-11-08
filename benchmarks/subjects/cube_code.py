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

"""The [[8,3,2]] cube code, the smallest 3D color code: X on all eight vertices of a cube, Z on
its faces; the standard-form encoder against the CSS normal form; T on the even vertices and
T^dagger on the odd ones after the encoder against the logical diagonal before it: in the
standard-form logical basis the sign -1 on |001> only, a CCZ with both controls negated."""

from fractions import Fraction

from benchmarks.subjects._gates import P
from benchmarks.subjects._stab import (
    Logical,
    code_subject,
    css_generators,
    logical_phase,
    pre_diagonal,
)

_TURNS = [Fraction((-1) ** v.bit_count(), 8) for v in range(8)]

_ALL = [[1] * 8]
_FACES = [
    [1 if (v >> b & 1) == s else 0 for v in range(8)] for b, s in ((0, 0), (1, 0), (2, 0), (0, 1))
]

SUBJECT = code_subject(
    id="cube_code",
    name="[[8,3,2]] cube code",
    source="campbell2016smallest",
    generators=css_generators(_ALL, _FACES),
    logical=[
        Logical(
            "transversal_t",
            lambda code: [P(f"q{v}", _TURNS[v]) for v in range(8)],
            lambda code: pre_diagonal(logical_phase(code, _TURNS), 3),
        )
    ],
)
