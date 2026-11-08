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

"""CHSH game, optimal quantum strategy: uniform inputs x, y; a Bell pair; Alice measures Z or X,
Bob at angle -pi/4 + y pi/2; the win bit a + b + xy is 0 with probability cos^2(pi/8)."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch, cry, ry


def _bit(q):
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Inputs, Bell pair, rotated measurements, the win bit."""
    gates = [
        *_bit("x"),
        *_bit("y"),
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        *ch("x", "a"),
        *ry("b", Fraction(-1, 8)),
        *cry("y", "b", Fraction(1, 4)),
        Measure("a"),
        Measure("b"),
        Init("w"),
        CX("a", "w"),
        CX("b", "w"),
        MCX(("x", "y"), "w"),
        Measure("w"),
        Discard("x"),
        Discard("y"),
        Discard("a"),
        Discard("b"),
    ]
    return circuit(["x", "y", "a", "b", "w"], gates)


def _spec(c):
    """The bit with distribution (cos^2(pi/8), sin^2(pi/8))."""
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: sqrt(2) - 1}}\nm = Z(1 -> 2, {c.D})\n"
        f"g = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\noutputs m.out0\n"
        "scalar sqrt(2 + sqrt(2))/2"
    )


SUBJECT = Subject(
    id="chsh_game",
    name="CHSH game",
    category="protocol",
    source="clauser1969proposed",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
