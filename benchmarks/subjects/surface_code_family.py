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

"""Rotated surface code family of odd distance L on an L x L grid, [[L^2, 1, L]]: weight-4
plaquettes alternating X and Z, weight-2 X checks on the top and bottom edges and Z checks on
the left and right; the standard-form encoder against the CSS normal form."""

from benchmarks.subject import Domain
from benchmarks.subjects._stab import code_family, css_generators, support_rows


def generators(L):
    """The rotated surface code's checks at distance L."""
    q = lambda r, c: r * L + c
    xs, zs = [], []
    for r in range(L - 1):
        for c in range(L - 1):
            block = [q(r, c), q(r, c + 1), q(r + 1, c), q(r + 1, c + 1)]
            (xs if (r + c) % 2 else zs).append(block)
    for c in range(L - 1):
        if c % 2 == 0:
            xs.append([q(0, c), q(0, c + 1)])
        if (L - 1 + c) % 2:
            xs.append([q(L - 1, c), q(L - 1, c + 1)])
    for r in range(L - 1):
        if r % 2:
            zs.append([q(r, 0), q(r + 1, 0)])
        if (r + L - 1) % 2 == 0:
            zs.append([q(r, L - 1), q(r + 1, L - 1)])
    return tuple(css_generators(support_rows(L * L, xs), support_rows(L * L, zs)))


SUBJECT = code_family(
    id="surface_code_family",
    name="Rotated surface code family",
    source="horsman2012surface",
    make=generators,
    size_parameter="code distance",
    domain=Domain(n_min=3, n_step=2, d_min=2, d_max=2),
)
