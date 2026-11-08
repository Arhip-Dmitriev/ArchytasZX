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

"""Five-qudit [[5,1,3]]_d code in graph form: the ring C5 with the logical set all five
vertices, |x> -> Z^x (x) ... (x) Z^x |C5>; CZ from the carrier, F on it and the ring's CZ edges
against the graph-form diagram."""

from benchmarks.ir import circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._qudit_codes import graph_encoder, graph_spec

_EDGES = [(j, (j + 1) % 5) for j in range(5)]
_N = [[0, 1, 2, 3, 4]]


def _circuit(n, d):
    """The graph-code encoder, carrier q0."""
    return circuit([f"q{j}" for j in range(5)], graph_encoder(5, _EDGES, _N, [0]))


def _spec(c):
    """The graph-form encoder."""
    return graph_spec(c, 5, _EDGES, _N, [0])


SUBJECT = Subject(
    id="five_qudit_code",
    name="Five-qudit code, graph form",
    category="code",
    source="schlingemann2001quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("encoding", _circuit, _spec, form="fixed"),),
    clifford=True,
)
