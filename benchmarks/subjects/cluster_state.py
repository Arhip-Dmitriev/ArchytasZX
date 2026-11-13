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


def _unrolled(n, d):
    """|+>^(n+1), then CZ(q_i, q_(i+1))."""
    q = unrolled("q", n + 1)
    gates = [g for x in q for g in (Init(x), F(x))] + [CZ(q[i], q[i + 1]) for i in range(n)]
    return circuit(q, gates)


def _circuit(n, d):
    """|+> on q, then per copy |+> on r and CZ(q, r), q carried on to r."""
    body = (Init("r"), F("r"), CZ("q", "r"))
    gates = [Init("q"), F("q"), Chain(n, body, carries=(("q", "r"),))]
    return circuit(["r", "q"], gates, registers={"r": n})


def _spec(c):
    """Path graph: Z spiders joined by F edges, vertex m's spider split into one ending copy
    m - 1 (or the head) and one starting copy m, the last vertex carried to the end."""
    return "\n".join(
        [
            f"h = Z(0 -> 1, {c.D})",
            f"repeat {c.N} as V: v = Z(1 -> 2, {c.D}), f = F({c.D}), w = Z(1 -> 1, {c.D})",
            "wire h.out0 -- v.in0, v.out1 -- f.in0, f.out0 -- w.in0",
            "chain V: w.out0 -> v.in0",
            "outputs v.out0, w.out0",
            f"scalar {c.d}^(-1/2)",
        ]
    )


SUBJECT = Subject(
    id="cluster_state",
    name="Linear cluster state",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="qudits after the first",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
