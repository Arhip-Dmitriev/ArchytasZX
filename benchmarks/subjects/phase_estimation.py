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

"""Quantum phase estimation with t clock qudits of an eigenphase m / d^t, m = 1: F on the clock,
clock qudit j controls U^(d^(t-1-j)) on the eigenstate |1> of U = diag(e^(2 pi i k m / d^t)),
then the inverse QFT; the clock reads |m> exactly."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, qft


def _circuit(n, d):
    """Clock in F|0>, controlled powers as phase tables, inverse QFT."""
    c = unrolled("c", n)
    D = d**n
    gates = [g for q in c for g in (Init(q), F(q))] + [Init("u", 1)]
    for j, q in enumerate(c):
        p = d ** (n - 1 - j)
        gates.append(PhaseFn((q, "u"), lambda x, p=p: Fraction(x[0] * x[1] * p, D)))
    gates += inverse(qft(c, d))
    return circuit([*c, "u"], gates)


def _spec(c):
    """|0 ... 0 1> on the clock beside |1>."""
    n = c.nv
    digits = [0] * (n - 1) + [1]
    lines = [
        f"c{i} = X(0 -> 1, {c.D})" + (" phase{k: -k/" + c.d + "}" if v else "")
        for i, v in enumerate(digits)
    ]
    lines.append(f"u = X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}")
    lines.append(f"scalar {c.d}^(-{n + 1}/2)")
    return "\n".join(lines)


SUBJECT = Subject(
    id="phase_estimation",
    name="Quantum phase estimation",
    category="algorithm",
    source="kitaev1995quantum",
    size_parameter="clock qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", literal=True),),
)
