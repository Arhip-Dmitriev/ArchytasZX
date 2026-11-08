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

"""Draper's QFT adder on two n-qudit registers: the QFT on b, a phase w^(a y) / d^n per digit
pair of a and the Fourier digits y of b, the inverse QFT; equals |a, b> -> |a, a + b mod d^n>."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, qft
from benchmarks.subjects._spec import register_adder


def _circuit(n, d):
    """QFT(b), the digit-pair phases, QFT^-1(b)."""
    a, b = unrolled("a", n), unrolled("b", n)
    D = d**n
    gates = qft(b, d)
    for i in range(n):
        for j in range(n):
            e = (n - 1 - i) + (n - 1 - j)
            if e < n:
                gates.append(PhaseFn((a[i], b[j]), lambda x, e=e: Fraction(x[0] * x[1] * d**e, D)))
    gates += inverse(qft(b, d))
    return circuit([*a, *b], gates)


def _spec(c):
    """Addition on the bound registers."""
    return register_adder(c, c.nv, c.dv).text()


SUBJECT = Subject(
    id="draper_adder",
    name="Draper QFT adder",
    category="circuit",
    source="draper2000addition",
    size_parameter="digits per register",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", literal=True),),
)
