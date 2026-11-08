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

"""Fredkin (controlled-SWAP) gate from CX, a Clifford+T Toffoli and CX; equals |c, a, b> -> |c,
a, b> for c = 0 and |c, b, a> for c = 1."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap
from benchmarks.subjects._spec import permutation


def _circuit(n, d):
    """CX(b, a), Toffoli(c, a, b), CX(b, a)."""
    return circuit(["c", "a", "b"], cswap("c", "a", "b"))


def _spec(c):
    """The controlled swap as a permutation."""
    return permutation(c, ["c", "a", "b"], lambda x: x if x < 4 else 4 + 2 * (x & 1) + (x >> 1 & 1))


def _reference(n, d):
    """CX, MCX, CX."""
    return circuit(["c", "a", "b"], [CX("b", "a"), MCX(("c", "a"), "b"), CX("b", "a")])


SUBJECT = Subject(
    id="fredkin_decomposition",
    name="Fredkin gate decomposition",
    category="circuit",
    source="fredkin1982conservative",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
