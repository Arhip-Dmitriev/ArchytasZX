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

"""QAOA, one layer, for MaxCut on the cycle of n + 2 qubits: |+>^(n+2), the cost unitary
e^(-i gamma C) with gamma = pi/4 as CX, P(-1/8), CX per edge, the mixer e^(-i beta X) per qubit;
the specification writes the cost unitary as one two-wire phase gadget per edge."""

from fractions import Fraction

from benchmarks.subjects._common import *

_BETA = Fraction(1, 8)


def _edge(a, b):
    """CX(a, b), P(-1/8) on b, CX(a, b)."""
    return [CX(a, b), ZPhase(b, {1: Fraction(-1, 8)}), CX(a, b)]


def _unrolled(n, d):
    """|+>^(n+2), CX P CX per edge (i, i + 1 mod n + 2), X phases."""
    m = n + 2
    q = unrolled("q", m)
    gates = [g for x in q for g in (Init(x), F(x))]
    for i in range(m):
        gates += _edge(q[i], q[(i + 1) % m])
    return circuit(q, gates + [XPhase(x, {1: _BETA}) for x in q])


def _circuit(n, d):
    """|+> on a (qubit 0) and q, the edge (a, q); per copy |+> on r and the edge (q, r), q
    carried on to r; the edge (q, a) and the X phases after the chain."""
    body = (Init("r"), F("r"), *_edge("q", "r"))
    gates = [Init("a"), F("a"), Init("q"), F("q"), *_edge("a", "q")]
    gates += [Chain(n, body, carries=(("q", "r"),)), *_edge("q", "a")]
    gates += [XPhase(x, {1: _BETA}) for x in ("a", "r", "q")]
    return circuit(["a", "r", "q"], gates, registers={"r": n})


def _gadget(name, ins, D):
    """Nodes and wires of the two-leg Z gadget ``name`` with phase -1/8 on the ports ``ins``;
    its legs' through ports are ``name``l0.out0 and ``name``l1.out0."""
    nodes = [f"{name}l{i} = Z(1 -> 2, {D})" for i in range(2)]
    nodes += [f"{name}h = X(2 -> 1, {D})", f"{name}p = Z(1 -> 0, {D}) phase{{1: -1/8}}"]
    wires = [f"{p} -- {name}l{i}.in0" for i, p in enumerate(ins) if p is not None]
    wires += [f"{name}l{i}.out1 -- {name}h.in{i}" for i in range(2)]
    wires.append(f"{name}h.out0 -- {name}p.in0")
    return nodes, wires


def _spec(c):
    """|+> per qubit, per edge the gadget Z legs, X hub, phase -1/8 on odd parity, X phases:
    the edge (0, 1) before a chain whose copy m holds qubit m + 2 and the edge (m + 1, m + 2),
    the closing edge after its tail."""
    D = c.D
    first, w0 = _gadget("e", ["a.out0", "h.out0"], D)
    copy, w1 = _gadget("c", [None, "s.out0"], D)
    last, w2 = _gadget("t", [None, "el0.out0"], D)
    mix = f"X(1 -> 1, {D}) phase{{1: {_BETA}}}"
    names = [line.split(" = ")[0] for line in copy]
    wires = [*w0, *w1, *w2, "el1.out0 -- cl0.in0", "cl1.out0 -- tl0.in0"]
    wires += ["cl0.out0 -- mc.in0", "tl1.out0 -- ma.in0", "tl0.out0 -- mt.in0"]
    return "\n".join(
        [
            f"a = Z(0 -> 1, {D})",
            f"h = Z(0 -> 1, {D})",
            *first,
            f"s = Z(0 -> 1, {D})",
            *copy,
            f"mc = {mix}",
            *last,
            f"ma = {mix}",
            f"mt = {mix}",
            "wire " + ", ".join(wires),
            f"repeat {c.N} as V: s, " + ", ".join(names) + ", mc",
            "chain V: cl1.out0 -> cl0.in0",
            "outputs ma.out0, mc.out0, mt.out0",
        ]
    )


SUBJECT = Subject(
    id="qaoa_maxcut",
    name="QAOA MaxCut on a cycle",
    category="algorithm",
    source="farhi2014quantum",
    size_parameter="qubits after the first two",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
