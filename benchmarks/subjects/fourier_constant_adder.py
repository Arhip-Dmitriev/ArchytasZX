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

"""Fourier constant adder on n qudits: the QFT, the phase w^(c y) / d^n on the Fourier digits y
for the constant c = 2 as single-qudit phases, the inverse QFT; equals |x> -> |x + 2 mod d^n>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, qft
from benchmarks.subjects._spec import Dsl, bind, split

_C = 2


def _circuit(n, d):
    """QFT, digit phases, inverse QFT."""
    x = unrolled("x", n)
    D = d**n
    gates = qft(x, d)
    for j in range(n):
        gates.append(ZPhase(x[j], rule=f"{_C}*k*{d ** (n - 1 - j)}/{D}"))
    return circuit(x, gates + inverse(qft(x, d)))


def _spec(c):
    """The shift by c on the bound register."""
    n, d = c.nv, c.dv
    D = d**n
    b = Dsl()
    ins, port = bind(b, n, d)
    s = b.node(f"X(1 -> 1, literal {D}) phase{{k: -{_C}*k/{D}}}")
    b.wire(port, f"{s}.in0")
    b.ins += ins
    b.outs += split(b, f"{s}.out0", n, d)
    return b.text()


SUBJECT = Subject(
    id="fourier_constant_adder",
    name="Fourier constant adder",
    category="circuit",
    source="draper2000addition",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled", literal=True),),
)
