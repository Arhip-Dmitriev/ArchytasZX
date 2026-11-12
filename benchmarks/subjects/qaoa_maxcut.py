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

"""QAOA, one layer, for MaxCut on the n-cycle: |+>^n, the cost unitary e^(-i gamma C) with gamma
= pi/4 as CX, P(-1/8), CX per edge, the mixer e^(-i beta X) per qubit; the specification writes
the cost unitary as one two-wire phase gadget per edge."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import Dsl

_BETA = Fraction(1, 8)


def _edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def _start(q):
    return [g for x in q for g in (Init(x), F(x))]


def _mixer(q):
    return [XPhase(x, {1: _BETA}) for x in q]


def _circuit(n, d):
    """|+>^n, CX P CX per edge, X phases."""
    q = unrolled("q", n)
    gates = _start(q)
    for a, b in _edges(n):
        gates += [CX(q[a], q[b]), ZPhase(q[b], {1: Fraction(-1, 8)}), CX(q[a], q[b])]
    return circuit(q, gates + _mixer(q))


def _spec(c):
    """|+>^n, per edge the gadget Z legs, X hub, phase -1/8 on odd parity, X phases."""
    n = c.nv
    q = unrolled("q", n)
    b = Dsl()
    ports = {x: f"{b.node(f'Z(0 -> 1, {c.D})')}.out0" for x in q}
    for i, j in _edges(n):
        hub = b.node(f"X(2 -> 1, {c.D})")
        for k, x in enumerate((q[i], q[j])):
            leg = b.node(f"Z(1 -> 2, {c.D})")
            b.wire(ports[x], f"{leg}.in0")
            b.wire(f"{leg}.out1", f"{hub}.in{k}")
            ports[x] = f"{leg}.out0"
        ph = b.node(f"Z(1 -> 0, {c.D}) phase{{1: -1/8}}")
        b.wire(f"{hub}.out0", f"{ph}.in0")
    for x in q:
        mix = b.node(f"X(1 -> 1, {c.D}) phase{{1: {_BETA}}}")
        b.wire(ports[x], f"{mix}.in0")
        ports[x] = f"{mix}.out0"
    b.outs += [ports[x] for x in q]
    b.scalar.append(f"2^(-{n}/2) * 2^({len(_edges(n))}/2)")
    return b.text()


SUBJECT = Subject(
    id="qaoa_maxcut",
    name="QAOA MaxCut on a cycle",
    category="algorithm",
    source="farhi2014quantum",
    size_parameter="qubits",
    dimension=2,
    domain=Domain(n_min=3, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
