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

"""Ring cluster state on n qudits (n >= 3): |+> everywhere, CZ around the cycle; the cycle graph's
Z spiders joined by Fourier edges."""

from benchmarks.subjects._common import *


def _graph_spec(c, edges, m):
    """Graph state: a Z spider per vertex (one output), F on each edge, normalised."""
    deg = [0] * m
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    lines = [f"v{i} = Z(0 -> {1 + deg[i]}, {c.D})" for i in range(m)]
    used = [1] * m
    for k, (a, b) in enumerate(edges):
        lines.append(f"f{k} = F({c.D})")
        lines.append(f"wire v{a}.out{used[a]} -- f{k}.in0, f{k}.out0 -- v{b}.out{used[b]}")
        used[a] += 1
        used[b] += 1
    lines.append("outputs " + ", ".join(f"v{i}.out0" for i in range(m)))
    lines.append(f"scalar {c.d}^(({len(edges)} - {m})/2)")
    return "\n".join(lines)


def _graph_circuit(edges, m):
    """|+> on every vertex, CZ on every edge."""
    q = unrolled("q", m)
    return circuit(q, [g for x in q for g in (Init(x), F(x))] + [CZ(q[a], q[b]) for a, b in edges])


def _edges(m):
    return [(i, (i + 1) % m) for i in range(m)]


def _circuit(n, d):
    """CZ around the ring."""
    return _graph_circuit(_edges(n), n)


def _spec(c):
    """The cycle graph."""
    return _graph_spec(c, _edges(c.nv), c.nv)


SUBJECT = Subject(
    id="ring_cluster_state",
    name="Ring cluster state",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=3, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
