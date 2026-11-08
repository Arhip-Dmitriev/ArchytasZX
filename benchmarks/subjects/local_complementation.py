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

"""Local complementation of a graph state at vertex 1 of the path 0 - 1 - 2: the path state
followed by e^(-i pi/4 X_1) and e^(i pi/4 Z_u) on the neighbours equals the triangle's graph
state."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, S, Sdg
from benchmarks.subjects._qudit_codes import graph_spec


def _circuit(n, d):
    """Path graph state, then H S H on vertex 1 and S^dagger on 0 and 2."""
    q = ["q0", "q1", "q2"]
    gates = [g for x in q for g in (Init(x), F(x))] + [CZ("q0", "q1"), CZ("q1", "q2")]
    gates += [H("q1"), S("q1"), H("q1"), Sdg("q0"), Sdg("q2")]
    return circuit(q, gates)


def _spec(c):
    """The triangle graph state."""
    return graph_spec(c, 3, [(0, 1), (1, 2), (0, 2)], [], [])


SUBJECT = Subject(
    id="local_complementation",
    name="Local complementation",
    category="circuit",
    source="vandennest2004graphical",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
