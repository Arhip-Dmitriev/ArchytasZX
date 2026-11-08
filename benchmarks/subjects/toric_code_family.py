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

"""Toric code family on the L x L torus, [[2L^2, 2, L]]: qubits on edges, vertex stars X and
plaquettes Z, one of each dropped as dependent; the standard-form encoder against the CSS normal
form."""

from benchmarks.subject import Domain
from benchmarks.subjects._stab import code_family, css_generators, support_rows


def generators(L):
    """Stars and plaquettes of the L x L torus."""
    h = lambda r, c: 2 * ((r % L) * L + c % L)
    v = lambda r, c: 2 * ((r % L) * L + c % L) + 1
    stars = [[h(r, c), h(r, c - 1), v(r, c), v(r - 1, c)] for r in range(L) for c in range(L)]
    plaq = [[h(r, c), h(r + 1, c), v(r, c), v(r, c + 1)] for r in range(L) for c in range(L)]
    n = 2 * L * L
    return tuple(css_generators(support_rows(n, stars[:-1]), support_rows(n, plaq[:-1])))


SUBJECT = code_family(
    id="toric_code_family",
    name="Toric code family",
    source="kitaev2003fault",
    make=generators,
    size_parameter="torus side",
    domain=Domain(n_min=2, d_min=2, d_max=2),
)
