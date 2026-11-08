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

"""Multiplication by 7 modulo 15 on four qubits: a rotation of the bits right by one
(multiplication by 8) by three swaps, then NOT on every bit (negation); equals y -> 7y mod 15 on
1..14, exchanging 0 and 15."""

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import permutation

_W = ["w0", "w1", "w2", "w3"]


def _circuit(n, d):
    """Swaps (w2, w3), (w1, w2), (w0, w1) as three CX each, then X on all."""
    gates = []
    for a, b in (("w2", "w3"), ("w1", "w2"), ("w0", "w1")):
        gates += [CX(a, b), CX(b, a), CX(a, b)]
    return circuit(_W, gates + [X(q) for q in _W])


def _p(y):
    return 15 - y if y in (0, 15) else 7 * y % 15


def _spec(c):
    """The permutation."""
    return permutation(c, _W, _p)


SUBJECT = Subject(
    id="modular_multiplication",
    name="Modular multiplication by 7 mod 15",
    category="circuit",
    source="markov2012constant",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
