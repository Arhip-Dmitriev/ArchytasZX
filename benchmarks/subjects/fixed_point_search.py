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

"""Grover's pi/3 fixed-point search: U = Ry(pi/2) prepares failure probability 1/2; U R_0
U^dagger R_t U with the selective phase shifts R = diag(e^(i pi/3) on the marked state) leaves
failure probability (1/2)^3 = 1/8."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ry


def _circuit(n, d):
    """U, R_t = P(1/6), U^dagger, R_0 = diag(e^(i pi/3), 1), U, measure."""
    u = Fraction(1, 4)
    gates = [
        Init("q"),
        *ry("q", u),
        ZPhase("q", {1: Fraction(1, 6)}),
        *ry("q", -u),
        X("q"),
        ZPhase("q", {1: Fraction(1, 6)}),
        X("q"),
        *ry("q", u),
        Measure("q"),
    ]
    return circuit(["q"], gates)


def _spec(c):
    """The bit with distribution (1/8, 7/8)."""
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: sqrt(7)}}\nm = Z(1 -> 2, {c.D})\n"
        f"g = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\noutputs m.out0\n"
        "scalar 2^(-3/2)"
    )


SUBJECT = Subject(
    id="fixed_point_search",
    name="Fixed-point quantum search",
    category="algorithm",
    source="grover2005fixed",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
