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

"""Qudit toric code on the 2 x 2 torus, [[8,2,2]]_d: oriented stars X^(+-1), plaquettes Z^(+-1);
logical X along the two non-contractible dual loops; the carrier-fan encoder against the integer
CSS normal form."""

from benchmarks.ir import circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._qudit_codes import css_encoder, css_spec
from benchmarks.subjects.toric_code import L, h, v


def _star(r, c):
    row = [0] * 8
    row[h(r, c)] += 1
    row[h(r, c - 1)] -= 1
    row[v(r, c)] += 1
    row[v(r - 1, c)] -= 1
    return row


_STABS = [_star(r, c) for r in range(L) for c in range(L)][:-1]
_LOG = [
    [1 if j in (v(0, 0), v(0, 1)) else 0 for j in range(8)],
    [1 if j in (h(0, 0), h(1, 0)) else 0 for j in range(8)],
]
_CARRIERS = [v(0, 0), h(0, 0)]


def _circuit(n, d):
    """The carrier-fan encoder."""
    return circuit([f"q{j}" for j in range(8)], css_encoder(8, _STABS, _LOG, _CARRIERS))


def _spec(c):
    """The integer CSS normal form."""
    return css_spec(c, 8, _STABS, _LOG, _CARRIERS)


SUBJECT = Subject(
    id="qudit_toric_code",
    name="Qudit toric code, 2 x 2 torus",
    category="code",
    source="bullock2007qudit",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("encoding", _circuit, _spec, form="fixed"),),
    clifford=True,
)
