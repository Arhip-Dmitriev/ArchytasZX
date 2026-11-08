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

"""Szegedy walk of the two-state Markov chain P = [[3/4, 1/4], [1/4, 3/4]]: W = SWAP (2 Pi - I),
Pi = sum_x |x><x| (x) |phi_x><phi_x|, |phi_x> = sum_y sqrt(P_xy) |y>. The circuit is
SWAP U (I (x) (2|0><0| - I)) U^dag with U |x, 0> = |x, phi_x> the controlled Ry(2 theta_x),
theta_0 = pi/6, theta_1 = pi/3, as Ry(-pi/6) between CX's then Ry(pi/2). The specification is
the closed-form matrix of W on one four-level leg per side."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, ry

_U = [CX("x", "y"), *ry("y", Fraction(-1, 12)), CX("x", "y"), *ry("y", Fraction(1, 4))]

_R = {0: [["1", "sqrt(3)"], ["sqrt(3)", "-1"]], 1: [["-1", "sqrt(3)"], ["sqrt(3)", "1"]]}


def _circuit(n, d):
    """U^dag, Z on y, U, then the register swap."""
    return circuit(["x", "y"], [*inverse(_U), Z("y"), *_U, Swap("x", "y")])


def _entry(a, b, x, y):
    """2 <a, b| W |x, y> = 2 <b, a| (2 Pi - I) |x, y>."""
    return _R[x][a][y] if b == x else "0"


def _spec(c):
    """sum_ij W_ij |i><j| as a weighted 16-level state, split into output and input legs, the
    input leg capped against the fused inputs."""
    w = {}
    for i in range(4):
        for j in range(4):
            w[4 * i + j] = _entry(i >> 1, i & 1, j >> 1, j & 1)
    table = ", ".join(f"{k}: {v}" for k, v in w.items())
    return (
        f"m = Z(0 -> 1, literal 16) weight{{{table}}}\n"
        "s = S(literal 4, literal 4)\n"
        "o = S(literal 2, literal 2)\n"
        "b = B(literal 2, literal 2)\n"
        "cap = Z(2 -> 0, literal 4)\n"
        "wire m.out0 -- s.in0\n"
        "wire s.out0 -- o.in0\n"
        "wire s.out1 -- cap.in0\n"
        "wire b.out0 -- cap.in1\n"
        "inputs b.in0, b.in1\n"
        "outputs o.out0, o.out1\n"
        "scalar 1/2"
    )


SUBJECT = Subject(
    id="szegedy_walk",
    name="Szegedy walk, two-state Markov chain",
    category="algorithm",
    source="szegedy2004quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
