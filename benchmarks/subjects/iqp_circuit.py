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

"""Instantaneous quantum polynomial-time (IQP) circuit on the path of n + 1 qubits: H on every
qubit, a diagonal of CZ on each edge and T on each qubit, H on every qubit, from |0>^(n+1). The
specification is the X-basis programme: CZ_(ab) = e^(i pi/4) e^(-i pi/4 X_a) e^(-i pi/4 X_b)
e^(i pi/4 X_a X_b) and T = e^(i pi/8) e^(-i pi/8 X) under the Hadamard conjugation, as Pauli
gadgets on |0>^(n+1), one chain copy per edge."""

import itertools
from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import T


def _unrolled(n, d):
    """|0>^(n+1), H, CZ per edge, T per qubit, H."""
    q = unrolled("q", n + 1)
    gates = [Init(x) for x in q] + [F(x) for x in q]
    gates += [CZ(a, b) for a, b in itertools.pairwise(q)] + [T(x) for x in q] + [F(x) for x in q]
    return circuit(q, gates)


def _circuit(n, d):
    """|+> on q, then per copy |+> on r and CZ(q, r), q carried on to r; T and H on every
    qubit after the chain."""
    body = (Init("r"), F("r"), CZ("q", "r"))
    gates = [Init("q"), F("q"), Chain(n, body, carries=(("q", "r"),))]
    gates += [T("r"), T("q"), F("r"), F("q")]
    return circuit(["r", "q"], gates, registers={"r": n})


def _x_gadget(lines, wires, name, ports, turn, D):
    """The X gadget e^(-i 2 pi turn X...X) on the chained ports ``ports`` (each a pair of the
    port it starts from and a list collecting its new end), its nodes named ``name``*."""
    legs = []
    for i, (start, end) in enumerate(ports):
        a, z, b = f"{name}a{i}", f"{name}z{i}", f"{name}b{i}"
        lines += [f"{a} = F({D})", f"{z} = Z(1 -> 2, {D})", f"{b} = F({D})"]
        if start is not None:
            wires.append(f"{start} -- {a}.in0")
        wires += [f"{a}.out0 -- {z}.in0", f"{z}.out0 -- {b}.in0"]
        legs.append(f"{z}.out1")
        end.append((f"{a}.in0", f"{b}.out0"))
    hub, ph = f"{name}h", f"{name}p"
    lines += [f"{hub} = X({len(legs)} -> 1, {D})", f"{ph} = Z(1 -> 0, {D}) phase{{1: 2*({turn})}}"]
    wires += [f"{leg} -- {hub}.in{i}" for i, leg in enumerate(legs)]
    wires.append(f"{hub}.out0 -- {ph}.in0")


def _boxed_spec(c):
    """The X gadgets per edge of the path as a chain: copy m holds qubit m + 1's state, the
    gadgets of edge (m, m + 1) and qubit m + 1's T gadget; qubit 0's T gadget before it."""
    D = c.D
    lines, wires = ["h = X(0 -> 1, " + D + ")"], []
    end = []
    _x_gadget(lines, wires, "g", [("h.out0", end)], Fraction(1, 16), D)
    head = end[-1][1]
    lines.append(f"s = X(0 -> 1, {D})")
    ce, se = [], []
    _x_gadget(lines, wires, "ea", [(None, ce)], Fraction(1, 8), D)
    q_port = ce[0][0]
    _x_gadget(lines, wires, "eb", [("s.out0", se)], Fraction(1, 8), D)
    _x_gadget(lines, wires, "ec", [(ce[-1][1], ce), (se[-1][1], se)], Fraction(-1, 8), D)
    _x_gadget(lines, wires, "ed", [(se[-1][1], se)], Fraction(1, 16), D)
    names = []
    for line in lines[1:]:
        name = line.split(" = ")[0]
        if not name.startswith("g"):
            names.append(name)
    wires.append(f"{head} -- {q_port}")
    out = [
        *lines,
        "wire " + ", ".join(wires),
        f"repeat {c.N} as V: " + ", ".join(names),
        f"chain V: {se[-1][1]} -> {q_port}",
        f"outputs {ce[-1][1]}, {se[-1][1]}",
        "scalar 2^(-1/2)",
    ]
    return "\n".join(out)


SUBJECT = Subject(
    id="iqp_circuit",
    name="IQP circuit",
    category="algorithm",
    source="shepherd2009temporally",
    size_parameter="qubits after the first",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _boxed_spec, form="boxed"),),
)
