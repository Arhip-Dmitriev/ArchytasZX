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

"""Swap test of the n-qubit states |0>^n and |+>^n: one control qubit in |+>, a Clifford+T
controlled swap per qubit pair, H, measure; the control reads 0 with probability (1 + 2^-n)/2."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap


def _circuit(n, d):
    """The swap test over registers a, b of n qubits."""
    a, b = unrolled("a", n), unrolled("b", n)
    gates = [Init("c"), F("c")] + [Init(x) for x in a] + [g for x in b for g in (Init(x), F(x))]
    for x, y in zip(a, b):
        gates += cswap("c", x, y)
    gates += [F("c"), Measure("c")] + [Discard(x) for x in (*a, *b)]
    return circuit(["c", *a, *b], gates)


def _spec(c):
    """The bit with distribution ((1 + 2^-n)/2, (1 - 2^-n)/2)."""
    n = c.nv
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: sqrt((1 - 2^(-{n}))/(1 + 2^(-{n})))}}\n"
        f"m = Z(1 -> 2, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\n"
        f"outputs m.out0\nscalar sqrt((1 + 2^(-{n}))/2)"
    )


SUBJECT = Subject(
    id="swap_test_register",
    name="Swap test on registers",
    category="algorithm",
    source="buhrman2001quantum",
    size_parameter="qubits per register",
    dimension=2,
    domain=Domain(n_min=0, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
