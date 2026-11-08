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

"""Qudit toric code family on the L x L torus, [[2L^2, 2, L]]_d: oriented stars X^(+-1), logical
X along the two non-contractible dual loops; the carrier-fan encoder against the integer CSS
normal form."""

from benchmarks.ir import circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._qudit_codes import css_encoder, css_spec


def _data(L):
    """Stars, logical rows and carriers of the L x L torus."""
    n = 2 * L * L
    h = lambda r, c: 2 * ((r % L) * L + c % L)
    v = lambda r, c: 2 * ((r % L) * L + c % L) + 1
    stars = []
    for r in range(L):
        for c in range(L):
            row = [0] * n
            row[h(r, c)] += 1
            row[h(r, c - 1)] -= 1
            row[v(r, c)] += 1
            row[v(r - 1, c)] -= 1
            stars.append(row)
    logs = [
        [1 if j in {v(0, c) for c in range(L)} else 0 for j in range(n)],
        [1 if j in {h(r, 0) for r in range(L)} else 0 for j in range(n)],
    ]
    return n, stars[:-1], logs, [v(0, 0), h(0, 0)]


def _circuit(L, d):
    """The carrier-fan encoder."""
    n, stabs, logs, carriers = _data(L)
    return circuit([f"q{j}" for j in range(n)], css_encoder(n, stabs, logs, carriers))


def _spec(c):
    """The integer CSS normal form."""
    n, stabs, logs, carriers = _data(c.nv)
    return css_spec(c, n, stabs, logs, carriers)


SUBJECT = Subject(
    id="qudit_toric_family",
    name="Qudit toric code family",
    category="code",
    source="bullock2007qudit",
    size_parameter="torus side",
    dimension="general",
    domain=Domain(n_min=2, d_min=1),
    equations=(Equation("encoding", _circuit, _spec, form="unrolled"),),
    clifford=True,
)
