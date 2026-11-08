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

"""Instantaneous quantum polynomial-time (IQP) circuit on the n-path: H on every qubit, a diagonal
of CZ on each edge and T on each qubit, H on every qubit, from |0>^n. The specification is the
X-basis programme: CZ_(ab) = e^(i pi/4) e^(-i pi/4 X_a) e^(-i pi/4 X_b) e^(i pi/4 X_a X_b) and
T = e^(i pi/8) e^(-i pi/8 X) under the Hadamard conjugation, as Pauli gadgets on |0>^n."""

import itertools
from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import T
from benchmarks.subjects._spec import Dsl, pauli_gadget


def _circuit(n, d):
    """|0>^n, H, CZ per edge, T per qubit, H."""
    q = unrolled("q", n)
    gates = [Init(x) for x in q] + [F(x) for x in q]
    gates += [CZ(a, b) for a, b in itertools.pairwise(q)] + [T(x) for x in q] + [F(x) for x in q]
    return circuit(q, gates)


def _spec(c):
    """X and XX gadgets on |0>^n, the phase e^(i pi/4) per edge and e^(i pi/8) per qubit."""
    n = c.nv
    q = unrolled("q", n)
    b = Dsl()
    ports = {}
    for x in q:
        z = b.node(f"X(0 -> 1, {c.D})")
        ports[x] = f"{z}.out0"
    b.scalar.append(f"2^(-{n}/2)")
    for a, bb in itertools.pairwise(q):
        pauli_gadget(b, ports, {a: "X"}, str(Fraction(1, 8)), c.D)
        pauli_gadget(b, ports, {bb: "X"}, str(Fraction(1, 8)), c.D)
        pauli_gadget(b, ports, {a: "X", bb: "X"}, str(Fraction(-1, 8)), c.D)
    for x in q:
        pauli_gadget(b, ports, {x: "X"}, str(Fraction(1, 16)), c.D)
    b.scalar.append(f"exp(I*pi/4*{n - 1}) * exp(I*pi/8*{n})")
    b.outs = [ports[x] for x in q]
    return b.text()


SUBJECT = Subject(
    id="iqp_circuit",
    name="IQP circuit",
    category="algorithm",
    source="shepherd2009temporally",
    size_parameter="qubits",
    dimension=2,
    domain=Domain(n_min=2, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
