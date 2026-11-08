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

"""Reversible full adder: two Clifford+T Toffolis and three CX on (a, b, c) and a clean carry
qubit; equals |a, b, c, 0> -> |a, b, a + b + c mod 2, maj(a, b, c)>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import toffoli
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """Toffoli(a, b, z), CX(a, b), Toffoli(b, c, z), CX(b, c), CX(a, b)."""
    gates = [
        Init("z"),
        *toffoli("a", "b", "z"),
        CX("a", "b"),
        *toffoli("b", "c", "z"),
        CX("b", "c"),
        CX("a", "b"),
    ]
    return circuit(["a", "b", "c", "z"], gates)


def _p(x):
    a, b, c, z = x >> 3 & 1, x >> 2 & 1, x >> 1 & 1, x & 1
    maj = (a & b) | (a & c) | (b & c)
    return 8 * a + 4 * b + 2 * (a ^ b ^ c) + (z ^ maj)


def _spec(c):
    """|0> on the carry, then the full-adder permutation."""
    return defining(c, ["a", "b", "c", "z"], [Init("z"), Perm(("a", "b", "c", "z"), _p)])


SUBJECT = Subject(
    id="full_adder",
    name="Reversible full adder",
    category="circuit",
    source="vedral1996quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
