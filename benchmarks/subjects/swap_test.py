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

"""Swap test on |0> and |+>: ancilla in |+>, controlled swap (Clifford+T Fredkin), H, measure;
the ancilla reads 0 with probability (1 + |<0|+>|^2)/2 = 3/4."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap


def _circuit(n, d):
    """The swap test circuit; both registers discarded."""
    gates = [
        Init("c"),
        F("c"),
        Init("a"),
        Init("b"),
        F("b"),
        *cswap("c", "a", "b"),
        F("c"),
        Measure("c"),
        Discard("a"),
        Discard("b"),
    ]
    return circuit(["c", "a", "b"], gates)


def _spec(c):
    """The bit with distribution (3/4, 1/4): sqrt(3/4)(|0> + |1>/sqrt(3)) dephased."""
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: 1/sqrt(3)}}\nm = Z(1 -> 2, {c.D})\n"
        f"g = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\noutputs m.out0\n"
        "scalar sqrt(3)/2"
    )


SUBJECT = Subject(
    id="swap_test",
    name="Swap test",
    category="algorithm",
    source="buhrman2001quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
