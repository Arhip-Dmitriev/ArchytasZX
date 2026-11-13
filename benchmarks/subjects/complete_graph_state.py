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

"""Complete graph state on n + 1 qudits: |+> everywhere, CZ between every pair; Z spiders pairwise
joined by Fourier edges."""

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
    return [(a, b) for a in range(m) for b in range(a + 1, m)]


def _unrolled(n, d):
    """CZ on every pair."""
    return _graph_circuit(_edges(n + 1), n + 1)


def _circuit(n, d):
    """|+> on each of n + 1 copies, CZ on every pair of copies."""
    gates = [Init("q"), F("q"), Pairs("q", (CZ("left", "right"),))]
    return circuit(["q"], gates, registers={"q": f"{n} + 1"})


def _spec(c):
    """The complete graph: a Z spider per copy, an F edge per pair of copies."""
    return "\n".join(
        [
            f"repeat {c.N} + 1 as V index j: v = Z(0 -> 3, {c.D})",
            f"pairs P of V index m: f = F({c.D})",
            "left P: v.out1",
            "right P: v.out2",
            "wire v.out1 -- f.in0, f.out0 -- v.out2",
            "outputs v.out0",
            f"copy scalar P: {c.d}^(1/2)",
            f"copy scalar V: {c.d}^(-1/2)",
        ]
    )


SUBJECT = Subject(
    id="complete_graph_state",
    name="Complete graph state",
    category="circuit",
    source="hein2004multiparty",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
