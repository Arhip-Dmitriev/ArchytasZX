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

"""Linear cluster state on n + 1 qudits: |+> each, CZ between neighbours; the graph state of the
path, Z spiders joined by Fourier edges."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """|+>^(n+1), then CZ(q_i, q_(i+1))."""
    q = unrolled("q", n + 1)
    gates = [g for x in q for g in (Init(x), F(x))] + [CZ(q[i], q[i + 1]) for i in range(n)]
    return circuit(q, gates)


def _spec(c):
    """Path graph: Z spider per vertex, F on each edge, d^(-(n+1)/2) d^(n/2)."""
    n = c.nv
    lines = []
    for i in range(n + 1):
        legs = 1 + (i > 0) + (i < n)
        lines.append(f"v{i} = Z(0 -> {legs}, {c.D})")
    for i in range(n):
        lines.append(f"f{i} = F({c.D})")
        lines.append(f"wire v{i}.out{1 if i == 0 else 2} -- f{i}.in0, f{i}.out0 -- v{i + 1}.out1")
    lines.append("outputs " + ", ".join(f"v{i}.out0" for i in range(n + 1)))
    lines.append(f"scalar {c.d}^(-1/2)")
    return "\n".join(lines)


SUBJECT = Subject(
    id="cluster_state",
    name="Linear cluster state",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
