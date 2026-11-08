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

"""Grover-Rudolph state preparation of the distribution (1/4, 1/4, 1/2, 0) on two qubits:
Ry(pi/2) on the first qubit for P(0) = 1/2, then on the second a rotation Ry(pi/2) conditioned
on the first being 0; the state is (1/2, 1/2, 1/sqrt(2), 0)."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, S, Sdg


def _ry(q, t):
    return [Sdg(q), H(q), P(q, t), H(q), S(q)]


def _circuit(n, d):
    """Ry(pi/2) on a; X, controlled Ry(pi/2) from a onto b, X."""
    gates = [
        Init("a"),
        Init("b"),
        *_ry("a", Fraction(1, 4)),
        X("a"),
        CX("a", "b"),
        *_ry("b", Fraction(-1, 8)),
        CX("a", "b"),
        *_ry("b", Fraction(1, 8)),
        X("a"),
    ]
    return circuit(["a", "b"], gates, scalar="exp(-I*pi/4)")


def _spec(c):
    """The amplitudes on one four-level leg, split into two qubits."""
    return (
        "z = Z(0 -> 1, literal 4) weight{2: sqrt(2), 3: 0}\ns = S(literal 2, literal 2)\n"
        "wire z.out0 -- s.in0\noutputs s.out0, s.out1\nscalar 1/2"
    )


SUBJECT = Subject(
    id="grover_rudolph",
    name="Grover-Rudolph state preparation",
    category="algorithm",
    source="grover2002creating",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
