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
transverse-field Ising chain on n + 1 qubits, e^(-i 2 pi t sum_j Z_j Z_(j+1)) then
e^(-i 2 pi t sum_j X_j), t = 1/16; each ZZ term as CX, P(2t), CX and each X term as H P(2t) H,
the phase e^(-2 pi i t) per term. The specification is the product of the Pauli gadgets of
the terms in the same order."""

import itertools
from fractions import Fraction

from benchmarks.subjects._common import *

_T = Fraction(1, 16)


def _zz(a, b):
    """CX(a, b), P(2t) on b, CX(a, b)."""
    return [CX(a, b), ZPhase(b, {1: 2 * _T}), CX(a, b)]


def _scalar(n):
    """e^(-2 pi i t) per term: n ZZ terms and n + 1 X terms."""
    return f"exp(-2*pi*I*({_T})*(2*({n}) + 1))"


def _unrolled(n, d):
    """ZZ terms along the chain, then X terms."""
    q = unrolled("q", n + 1)
    gates = [g for a, b in itertools.pairwise(q) for g in _zz(a, b)]
    gates += [XPhase(x, {1: 2 * _T}) for x in q]
    return circuit(q, gates, scalar=_scalar(n))


def _circuit(n, d):
    """Per copy the ZZ term on (q, r), q carried on to r; the X terms after the chain. Inputs q
    (qubit 0) then r, outputs r then q."""
    gates = [Chain(n, tuple(_zz("q", "r")), carries=(("q", "r"),))]
    gates += [XPhase("r", {1: 2 * _T}), XPhase("q", {1: 2 * _T})]
    return circuit(["q", "r"], gates, registers={"r": n}, scalar=_scalar(n), outputs=["r", "q"])


def _x_gadget(name, D):
    """The X gadget e^(-i 2 pi t X) ``name``: F, Z leg, F, X hub, phase 2t; through ports
    ``name``a.in0 and ``name``b.out0."""
    nodes = [
        f"{name}a = F({D})",
        f"{name}l = Z(1 -> 2, {D})",
        f"{name}b = F({D})",
        f"{name}h = X(1 -> 1, {D})",
        f"{name}p = Z(1 -> 0, {D}) phase{{1: {2 * _T}}}",
    ]
    wires = [
        f"{name}a.out0 -- {name}l.in0",
        f"{name}l.out0 -- {name}b.in0",
        f"{name}l.out1 -- {name}h.in0",
        f"{name}h.out0 -- {name}p.in0",
    ]
    return nodes, wires


def _spec(c):
    """One ZZ gadget per bond in the chain's copies (copy m: bond (m, m + 1), qubit m + 1's input
    s), the X gadgets on each copy's output and on the tail."""
    D = c.D
    xc, wc = _x_gadget("x", D)
    xt, wt = _x_gadget("y", D)
    copy = [
        f"s = Z(1 -> 1, {D})",
        f"l0 = Z(1 -> 2, {D})",
        f"l1 = Z(1 -> 2, {D})",
        f"g = X(2 -> 1, {D})",
        f"p = Z(1 -> 0, {D}) phase{{1: {2 * _T}}}",
        *xc,
    ]
    wires = [
        "h.out0 -- l0.in0",
        "s.out0 -- l1.in0",
        "l0.out1 -- g.in0",
        "l1.out1 -- g.in1",
        "g.out0 -- p.in0",
        "l0.out0 -- xa.in0",
        "l1.out0 -- ya.in0",
        *wc,
        *wt,
    ]
    names = [line.split(" = ")[0] for line in copy]
    return "\n".join(
        [
            f"h = Z(1 -> 1, {D})",
            *copy,
            *xt,
            "wire " + ", ".join(wires),
            f"repeat {c.N} as V: " + ", ".join(names),
            "chain V: l1.out0 -> l0.in0",
            f"copy scalar V: 2^(1/2) * exp(-4*pi*I*({_T}))",
            "inputs h.in0, s.in0",
            "outputs xb.out0, yb.out0",
            f"scalar exp(-2*pi*I*({_T}))",
        ]
    )


SUBJECT = Subject(
    id="trotter_ising",
    name="Trotter step, transverse-field Ising chain",
    category="algorithm",
    source="lloyd1996universal",
    size_parameter="qubits after the first",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
