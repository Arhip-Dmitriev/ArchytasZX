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

"""Peres gate: a Clifford+T Toffoli followed by CX(a, b); equals |a, b, c> -> |a, a + b, c +
ab>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import toffoli
from benchmarks.subjects._spec import permutation


def _circuit(n, d):
    """Toffoli(a, b, c), CX(a, b)."""
    return circuit(["a", "b", "c"], [*toffoli("a", "b", "c"), CX("a", "b")])


def _p(x):
    a, b, c = x >> 2 & 1, x >> 1 & 1, x & 1
    return 4 * a + 2 * (a ^ b) + (c ^ (a & b))


def _spec(c):
    """The Peres permutation."""
    return permutation(c, ["a", "b", "c"], _p)


def _reference(n, d):
    """MCX then CX."""
    return circuit(["a", "b", "c"], [MCX(("a", "b"), "c"), CX("a", "b")])


SUBJECT = Subject(
    id="peres_gate",
    name="Peres gate",
    category="circuit",
    source="peres1985reversible",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
