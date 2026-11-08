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

"""Pauli exponential e^(-i pi/8 X (x) Y (x) Z) synthesised by basis changes, a CX ladder and one
Z rotation; the specification is the phase gadget: legs in the X, Y, Z bases joined by an X hub
to a Z phase."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, S, Sdg


def _circuit(n, d):
    """Basis change, ladder, e^(-i pi/8 Z) = e^(-i pi/8) P(1/8), ladder, basis change."""
    gates = [
        H("a"),
        Sdg("b"),
        H("b"),
        CX("a", "c"),
        CX("b", "c"),
        P("c", Fraction(1, 8)),
        CX("b", "c"),
        CX("a", "c"),
        H("a"),
        H("b"),
        S("b"),
    ]
    return circuit(["a", "b", "c"], gates, scalar="exp(-I*pi/8)")


def _spec(c):
    """The XYZ phase gadget with phase 1/8 on the odd parity."""
    D = c.D
    return "\n".join(
        [
            f"ha = F({D})",
            f"la = Z(1 -> 2, {D})",
            f"ha2 = F({D})",
            f"sb = Z(1 -> 1, {D}) phase{{1: 3/4}}",
            f"hb = F({D})",
            f"lb = Z(1 -> 2, {D})",
            f"hb2 = F({D})",
            f"sb2 = Z(1 -> 1, {D}) phase{{1: 1/4}}",
            f"lc = Z(1 -> 2, {D})",
            f"hub = X(3 -> 1, {D})",
            f"ph = Z(1 -> 0, {D}) phase{{1: 1/8}}",
            (
                "wire ha.out0 -- la.in0, la.out0 -- ha2.in0, sb.out0 -- hb.in0, hb.out0 -- lb.in0, "
                "lb.out0 -- hb2.in0, hb2.out0 -- sb2.in0, la.out1 -- hub.in0, lb.out1 -- hub.in1, "
                "lc.out1 -- hub.in2, hub.out0 -- ph.in0"
            ),
            "inputs ha.in0, sb.in0, lc.in0",
            "outputs ha2.out0, sb2.out0, lc.out0",
            "scalar 2 * exp(-I*pi/8)",
        ]
    )


SUBJECT = Subject(
    id="pauli_gadget",
    name="Pauli exponential synthesis",
    category="circuit",
    source="cowtan2020phase",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
