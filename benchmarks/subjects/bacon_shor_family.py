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

"""Bacon-Shor subsystem code family on an L x L grid, [[L^2, 1, L]]: X stabilizers on adjacent
row pairs, Z stabilizers on adjacent column pairs; the encoder takes the logical and the gauge
qubits as inputs; the standard-form encoder against the CSS normal form."""

from benchmarks.subject import Domain
from benchmarks.subjects._stab import code_family, css_generators, support_rows


def generators(L):
    """Row-pair X and column-pair Z stabilizers."""
    rows = [[L * r + c for r in (a, a + 1) for c in range(L)] for a in range(L - 1)]
    cols = [[L * r + c for c in (a, a + 1) for r in range(L)] for a in range(L - 1)]
    return tuple(css_generators(support_rows(L * L, rows), support_rows(L * L, cols)))


SUBJECT = code_family(
    id="bacon_shor_family",
    name="Bacon-Shor code family",
    source="bacon2006operator",
    make=generators,
    size_parameter="grid side",
    domain=Domain(n_min=2, d_min=2, d_max=2),
)
