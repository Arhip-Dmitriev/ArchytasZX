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

"""Ring cluster state on n + 2 qudits (n >= 1): |+> everywhere, CZ around the cycle; the cycle
graph's Z spiders joined by Fourier edges."""

from benchmarks.subjects._common import *


def _unrolled(n, d):
    """|+> on every vertex, CZ(q_i, q_(i+1)) around the ring of n + 2."""
    m = n + 2
    q = unrolled("q", m)
    gates = [g for x in q for g in (Init(x), F(x))]
    return circuit(q, gates + [CZ(q[i], q[(i + 1) % m]) for i in range(m)])


def _circuit(n, d):
    """|+> on a (vertex 0) and q, CZ(a, q); per copy |+> on r and CZ(q, r), q carried on to r;
    CZ(q, a) after the chain."""
    body = (Init("r"), F("r"), CZ("q", "r"))
    gates = [Init("a"), F("a"), Init("q"), F("q"), CZ("a", "q")]
    gates += [Chain(n, body, carries=(("q", "r"),)), CZ("q", "a")]
    return circuit(["a", "r", "q"], gates, registers={"r": n})


def _spec(c):
    """The cycle graph: vertex 0's spider a joined by F edges to the head h and to the spider t
    after the path chain's tail; vertex m's spider split as in cluster_state."""
    D = c.D
    return "\n".join(
        [
            f"a = Z(0 -> 3, {D})",
            f"fa = F({D})",
            f"h = Z(1 -> 1, {D})",
            f"repeat {c.N} as V: v = Z(1 -> 2, {D}), f = F({D}), w = Z(1 -> 1, {D})",
            f"t = Z(1 -> 2, {D})",
            f"fb = F({D})",
            (
                "wire a.out1 -- fa.in0, fa.out0 -- h.in0, h.out0 -- v.in0, v.out1 -- f.in0, "
                "f.out0 -- w.in0, w.out0 -- t.in0, t.out1 -- fb.in0, fb.out0 -- a.out2"
            ),
            "chain V: w.out0 -> v.in0",
            "outputs a.out0, v.out0, t.out0",
        ]
    )


SUBJECT = Subject(
    id="ring_cluster_state",
    name="Ring cluster state",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="qudits after the first two",
    dimension="general",
    domain=Domain(n_min=1, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
