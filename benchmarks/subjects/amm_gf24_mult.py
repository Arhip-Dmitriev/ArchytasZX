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

"""Multiplication in GF(2^4) = GF(2)[x]/(x^4 + x + 1) (Amy-Maslov-Mosca benchmark gf2^4_mult):
c ^= a b on coefficient registers a, b, c (index = power of x), c starting in |0>; the
specification is one Toffoli a_i b_j -> c_k per k in the reduction of x^(i+j)."""

from benchmarks.ir import MCX
from benchmarks.subjects._qc import amm_subject


def _powers(k):
    """The coefficients of x^k reduced by x^4 = x + 1."""
    p = [0] * 7
    p[k] = 1
    for j in (6, 5, 4):
        if p[j]:
            p[j] = 0
            p[j - 4] ^= 1
            p[j - 3] ^= 1
    return [i for i in range(4) if p[i]]


_GATES = [
    MCX((f"a{i}", f"b{j}"), f"c{k}") for i in range(4) for j in range(4) for k in _powers(i + j)
]

SUBJECT = amm_subject("gf2^4_mult", "GF(2^4) multiplier", None, "", gates=_GATES)
