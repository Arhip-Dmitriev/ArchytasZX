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

"""Quantum Reed-Muller [[15,1,3]] code (the tetrahedral 3D color code): X checks the four
weight-8 rows of the punctured RM(1,4), Z checks those and their ten weight-4 pairwise products;
the standard-form encoder against the CSS normal form; transversal T after the encoder against
the logical T^dagger before it."""

from fractions import Fraction
from itertools import combinations

from benchmarks.subjects._gates import T
from benchmarks.subjects._stab import Logical, code_subject, css_generators, pre_diagonal, wires

_BITS = [[1 if (j + 1) >> i & 1 else 0 for j in range(15)] for i in range(4)]
_PAIRS = [[a & b for a, b in zip(_BITS[i], _BITS[k])] for i, k in combinations(range(4), 2)]

SUBJECT = code_subject(
    id="reed_muller_code",
    name="Quantum Reed-Muller [[15,1,3]] code",
    source="knill1996threshold",
    generators=css_generators(_BITS, _BITS + _PAIRS),
    logical=[
        Logical(
            "transversal_t",
            lambda code: [T(q) for q in wires(code.n)],
            lambda code: pre_diagonal(lambda x: Fraction(-x[0], 8), 1),
        )
    ],
)
