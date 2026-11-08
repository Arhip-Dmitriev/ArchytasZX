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

"""Hadamard test of U = S on |+>: ancilla H, controlled S (Clifford+T), H, measure; the ancilla
reads 0 with probability (1 + Re <+|S|+>)/2 = 3/4."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cphase


def _circuit(n, d):
    """The Hadamard test; the register discarded."""
    gates = [
        Init("a"),
        F("a"),
        Init("q"),
        F("q"),
        *cphase("a", "q", "1/4"),
        F("a"),
        Measure("a"),
        Discard("q"),
    ]
    return circuit(["a", "q"], gates)


def _spec(c):
    """The bit with distribution (3/4, 1/4)."""
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: 1/sqrt(3)}}\nm = Z(1 -> 2, {c.D})\n"
        f"g = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\noutputs m.out0\n"
        "scalar sqrt(3)/2"
    )


SUBJECT = Subject(
    id="hadamard_test",
    name="Hadamard test",
    category="algorithm",
    source="aharonov2009polynomial",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
