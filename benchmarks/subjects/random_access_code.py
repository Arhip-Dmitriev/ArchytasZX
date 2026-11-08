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

"""Quantum random access code 2 -> 1: two uniform bits encoded as X^a0 Z^a1 Ry(pi/4)|0>, the
first bit decoded by a Z measurement; the error bit (outcome + a0) is 0 with probability
cos^2(pi/8)."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ry


def _bit(q):
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Encode, measure Z, compare with a0."""
    gates = [
        *_bit("a0"),
        *_bit("a1"),
        Init("q"),
        *ry("q", Fraction(1, 8)),
        CZ("a1", "q"),
        CX("a0", "q"),
        Measure("q"),
        CX("a0", "q"),
        Discard("a0"),
        Discard("a1"),
    ]
    return circuit(["a0", "a1", "q"], gates)


def _spec(c):
    """The bit with distribution (cos^2(pi/8), sin^2(pi/8))."""
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: sqrt(2) - 1}}\nm = Z(1 -> 2, {c.D})\n"
        f"g = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\noutputs m.out0\n"
        "scalar sqrt(2 + sqrt(2))/2"
    )


SUBJECT = Subject(
    id="random_access_code",
    name="Quantum random access code",
    category="protocol",
    source="ambainis2002dense",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
