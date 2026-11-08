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

"""Three-qubit hypergraph state: |+>^3 and the hyperedge CCZ by the 7-T phase polynomial; equals
the phase table of the hyperedge applied to |+>^3."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ccz
from benchmarks.subjects._spec import defining

_Q = ["a", "b", "c"]


def _plus():
    return [g for q in _Q for g in (Init(q), F(q))]


def _circuit(n, d):
    """|+>^3 and the Clifford+T CCZ."""
    return circuit(_Q, [*_plus(), *ccz("a", "b", "c")])


def _spec(c):
    """|+>^3 and the CCZ phase table."""
    return defining(
        c, _Q, [*_plus(), PhaseFn(tuple(_Q), lambda x: Fraction(x[0] * x[1] * x[2], 2))]
    )


SUBJECT = Subject(
    id="hypergraph_state",
    name="Hypergraph state",
    category="circuit",
    source="rossi2013quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
