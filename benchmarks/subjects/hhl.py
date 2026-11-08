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

"""HHL on A = [[3/2, 1/2], [1/2, 3/2]] (eigenvalues 2 on |+>, 1 on |->) and b = |0>: phase
estimation of U = e^(2 pi i A / 4) on two clock qubits, ancilla rotations Ry(pi) and Ry(pi/3)
for 1/lambda with C = 1, uncomputation, ancilla and clock post-selected; the output is A^(-1)
|b> = (3/4)|0> - (1/4)|1>."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, S, Sdg, inverse, qft


def _ry(q, t):
    """e^(i pi t) Ry(2 pi t)."""
    return [Sdg(q), H(q), P(q, t), H(q), S(q)]


def _cry(c, q, t):
    """Controlled Ry(2 pi t): Ry(-pi t), CX, Ry(pi t), CX in time order reversed."""
    return [CX(c, q), *_ry(q, -Fraction(t) / 2), CX(c, q), *_ry(q, Fraction(t) / 2)]


def _qpe():
    """Clock in |+>, controlled U from c1, controlled U^2 = X from c0, inverse QFT."""
    out = [F("c0"), F("c1")]
    out += [H("b"), PhaseFn(("c1", "b"), lambda x: Fraction(x[0] * (2 - x[1]), 4)), H("b")]
    out += [CX("c0", "b")]
    out += inverse(qft(["c0", "c1"], 2))
    return out


def _circuit(n, d):
    """QPE, rotations conditioned on the clock, inverse QPE, post-selection."""
    qpe = _qpe()
    gates = [Init("c0"), Init("c1"), Init("a"), *qpe]
    gates += _cry("c1", "a", Fraction(1, 2)) + _cry("c0", "a", Fraction(1, 6))
    gates += inverse(qpe)
    gates += [Post("a", 1), Post("c0"), Post("c1")]
    return circuit(["c0", "c1", "a", "b"], gates)


def _spec(c):
    """A^(-1) from its eigenbasis: F, the weights (1/2, 1), F."""
    return f"F({c.D}) -> Z(1 -> 1, {c.D}) weight{{1: 2}} -> F({c.D})\nscalar 1/2"


SUBJECT = Subject(
    id="hhl",
    name="HHL linear-system solver, 2 x 2",
    category="algorithm",
    source="harrow2009quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
