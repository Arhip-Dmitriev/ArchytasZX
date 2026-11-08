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

"""Hamiltonian simulation by a first-order product formula: one Trotter step of the
transverse-field Ising chain on n qubits, e^(-i 2 pi t sum_j Z_j Z_(j+1)) then
e^(-i 2 pi t sum_j X_j), t = 1/16; each ZZ term as CX, P(2t), CX and each X term as H P(2t) H,
the phase e^(-2 pi i t) per term. The specification is the product of the Pauli gadgets of
the terms in the same order."""

import itertools
from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import Dsl, pauli_gadget

_T = Fraction(1, 16)


def _circuit(n, d):
    """ZZ terms along the chain, then X terms."""
    q = unrolled("q", n)
    gates = []
    for a, b in itertools.pairwise(q):
        gates += [CX(a, b), ZPhase(b, {1: 2 * _T}), CX(a, b)]
    gates += [XPhase(x, {1: 2 * _T}) for x in q]
    terms = (n - 1) + n
    return circuit(q, gates, scalar=f"exp(-2*pi*I*({_T})*{terms})")


def _spec(c):
    """One ZZ gadget per bond, one X gadget per site."""
    n = c.nv
    q = unrolled("q", n)
    b = Dsl()
    ports = {}
    for x in q:
        z = b.node(f"Z(1 -> 1, {c.D})")
        b.ins.append(f"{z}.in0")
        ports[x] = f"{z}.out0"
    for a, bb in itertools.pairwise(q):
        pauli_gadget(b, ports, {a: "Z", bb: "Z"}, str(_T), c.D)
    for x in q:
        pauli_gadget(b, ports, {x: "X"}, str(_T), c.D)
    b.outs = [ports[x] for x in q]
    return b.text()


SUBJECT = Subject(
    id="trotter_ising",
    name="Trotter step, transverse-field Ising chain",
    category="algorithm",
    source="lloyd1996universal",
    size_parameter="qubits",
    dimension=2,
    domain=Domain(n_min=2, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
