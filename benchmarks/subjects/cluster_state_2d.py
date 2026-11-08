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

"""Two-dimensional cluster state on an n x n grid: F|0> per vertex, CZ on every grid edge; the
specification is the graph-state diagram."""

from benchmarks.subjects._common import *
from benchmarks.subjects._qudit_codes import graph_spec


def _edges(n):
    out = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                out.append((r * n + c, r * n + c + 1))
            if r + 1 < n:
                out.append((r * n + c, (r + 1) * n + c))
    return out


def _circuit(n, d):
    """|+> per vertex, the CZ edges."""
    q = unrolled("q", n * n)
    gates = [g for x in q for g in (Init(x), F(x))]
    gates += [CZ(q[a], q[b]) for a, b in _edges(n)]
    return circuit(q, gates)


def _spec(c):
    """The grid graph state."""
    return graph_spec(c, c.nv * c.nv, _edges(c.nv), [], [])


SUBJECT = Subject(
    id="cluster_state_2d",
    name="Two-dimensional cluster state",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="grid side",
    dimension="general",
    domain=Domain(n_min=1, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
