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

"""Grover search on three qubits with two marked items, 011 and 101: sin^2(theta) = 1/4; one
iteration (the oracle phase table, the diffusion H X CCZ X H) reaches the uniform superposition
of the marked items exactly."""

from fractions import Fraction

from benchmarks.subjects._common import *

_Q = ["a", "b", "c"]
_MARKED = {(0, 1, 1), (1, 0, 1)}


def _circuit(n, d):
    """|+>^3, the oracle, the diffusion."""
    gates = [g for q in _Q for g in (Init(q), F(q))]
    gates.append(PhaseFn(tuple(_Q), lambda x: Fraction(1, 2) if x in _MARKED else 0))
    gates += [F(q) for q in _Q] + [X(q) for q in _Q] + [MCZ(("a", "b"), "c")]
    gates += [X(q) for q in _Q] + [F(q) for q in _Q]
    return circuit(_Q, gates, scalar="-1")


def _spec(c):
    """(|01> + |10>) |1> / sqrt(2)."""
    return (
        f"z = Z(0 -> 2, {c.D})\nn = X(1 -> 1, {c.D}) phase{{1: 1/2}}\n"
        f"one = X(0 -> 1, {c.D}) phase{{1: 1/2}}\nwire z.out1 -- n.in0\n"
        "outputs z.out0, n.out0, one.out0\nscalar 1/2"
    )


SUBJECT = Subject(
    id="grover_two_marked",
    name="Grover search, two marked items",
    category="algorithm",
    source="boyer1998tight",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
