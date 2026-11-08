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

"""Phase-polynomial (CNOT + phase) synthesis of a uniformly controlled phase: the diagonal e^(2
pi i theta_c t) with theta = (0, 1/8, 1/4, 3/8) over the controls c, expanded in parities, each
parity computed onto its last qubit by CX, phased, uncomputed."""

from fractions import Fraction
from itertools import product

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import diagonal

_W = ["c0", "c1", "t"]
_THETA = [Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(3, 8)]


def _f(x):
    return _THETA[2 * x[0] + x[1]] * x[2]


def _terms():
    """The parity expansion f = a_0 + sum_S a_S [xor of S]."""
    pts = list(product((0, 1), repeat=3))
    out, total = {}, Fraction(0)
    for S in product((0, 1), repeat=3):
        if not any(S):
            continue
        b = sum(_f(x) * (-1) ** sum(s & v for s, v in zip(S, x)) for x in pts) / 8
        out[S] = -2 * b
        total += b
    a0 = sum(_f(x) for x in pts) / 8 + total
    return a0, out


def _circuit(n, d):
    """One CX-phase-CX block per parity."""
    a0, terms = _terms()
    gates = []
    for S, a in terms.items():
        if not a:
            continue
        qs = [w for w, s in zip(_W, S) if s]
        comp = [CX(q, qs[-1]) for q in qs[:-1]]
        gates += comp + [ZPhase(qs[-1], {1: a})] + list(reversed(comp))
    return circuit(_W, gates, scalar=f"exp(2*pi*I*({a0}))")


def _spec(c):
    """The diagonal."""
    return diagonal(c, _W, _f)


SUBJECT = Subject(
    id="phase_polynomial_synthesis",
    name="Phase-polynomial synthesis",
    category="circuit",
    source="amy2014polynomial",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
