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

"""Toffoli from Hadamard, CX and T gates; equals |a, b, c> -> |a, b, c + ab>."""

from benchmarks.subjects._common import *


def _ccz_gates(a, b, c):
    """The 7-T CCZ: phase polynomial over a, b, c (Nielsen and Chuang, figure 4.9)."""
    t = lambda q: ZPhase(q, {1: "1/8"})
    td = lambda q: ZPhase(q, {1: "-1/8"})
    return [
        CX(b, c),
        td(c),
        CX(a, c),
        t(c),
        CX(b, c),
        td(c),
        CX(a, c),
        t(b),
        t(c),
        CX(a, b),
        t(a),
        td(b),
        CX(a, b),
    ]


def _ccz_spec(c, h=False):
    """Binders to dimension 8, phase 1/2 at |111>, splitters; with h, F on the target around it."""
    lines = [
        "b1 = B(literal 2, literal 2)",
        "b2 = B(literal 4, literal 2)",
        "wire b1.out0 -- b2.in0",
        "z = Z(1 -> 1, literal 8) phase{7: 1/2}",
        "wire b2.out0 -- z.in0",
        "s2 = S(literal 4, literal 2)",
        "s1 = S(literal 2, literal 2)",
        "wire z.out0 -- s2.in0, s2.out0 -- s1.in0",
    ]
    if h:
        lines += [
            "h1 = F(literal 2)",
            "h2 = F(literal 2)",
            "wire h1.out0 -- b2.in1, s2.out1 -- h2.in0",
        ]
        lines += ["inputs b1.in0, b1.in1, h1.in0", "outputs s1.out0, s1.out1, h2.out0"]
    else:
        lines += ["inputs b1.in0, b1.in1, b2.in1", "outputs s1.out0, s1.out1, s2.out1"]
    return "\n".join(lines)


def _circuit(n, d):
    """H on the target around the phase-polynomial CCZ."""
    return circuit(["a", "b", "c"], [F("c"), *_ccz_gates("a", "b", "c"), F("c")])


def _spec(c):
    """CCZ conjugated by F on the target."""
    return _ccz_spec(c, h=True)


def _reference(n, d):
    """The Toffoli gate."""
    return circuit(["a", "b", "c"], [MCX(("a", "b"), "c")])


SUBJECT = Subject(
    id="toffoli_decomposition",
    name="Toffoli Clifford+T decomposition",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
