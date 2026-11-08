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

"""Quantum Hamming code family [[2^r - 1, 2^r - 1 - 2r, 3]]: X and Z checks the r rows of the
Hamming parity-check matrix (columns all nonzero r-bit words); the standard-form encoder against
the CSS normal form."""

from benchmarks.subject import Domain
from benchmarks.subjects._stab import code_family, css_generators


def generators(r):
    """The Hamming rows as X and Z checks."""
    n = 2**r - 1
    rows = [[(j + 1) >> i & 1 for j in range(n)] for i in range(r)]
    return tuple(css_generators(rows, rows))


SUBJECT = code_family(
    id="quantum_hamming_family",
    name="Quantum Hamming code family",
    source="steane1996simple",
    make=generators,
    size_parameter="check rows",
    domain=Domain(n_min=3, d_min=2, d_max=2),
)
