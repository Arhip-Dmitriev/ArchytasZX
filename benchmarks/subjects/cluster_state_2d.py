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


def _edges(n):
    out = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                out.append((r * n + c, r * n + c + 1))
            if r + 1 < n:
                out.append((r * n + c, (r + 1) * n + c))
    return out


def _reference(n, d):
    """|+> per vertex, the CZ edges."""
    q = unrolled("q", n * n)
    gates = [g for x in q for g in (Init(x), F(x))]
    gates += [CZ(q[a], q[b]) for a, b in _edges(n)]
    return circuit(q, gates)


def _circuit(n, d):
    """Row 0 as |+> on the register c with CZ between neighbours, then per row |+> on the row
    x of the two-count register, CZ between neighbours and CZ from c, c carried on to x."""
    rows = n - 1 if isinstance(n, int) else f"{n} - 1"
    body = (Init("x"), F("x"), Neighbours("x"), CZ("c", "x"))
    gates = [Init("c"), F("c"), Neighbours("c"), Chain(rows, body, carries=(("c", "x"),))]
    return circuit(["x", "c"], gates, registers={"c": n, "x": (rows, n)})


def _spec(c):
    """The grid graph state: a row chain R through the column chain C, vertex v's out1 linked
    to the next column's F edge and out2 to the next row's; the ends are X(0 -> 1) states
    before an F edge (the unit) and Z(1 -> 0) effects, the vertical ones in boxes PH and PT."""
    D = c.D
    return "\n".join(
        [
            f"v = Z(2 -> 3, {D})",
            f"hv = F({D})",
            f"hh = F({D})",
            f"hs = X(0 -> 1, {D})",
            f"he = Z(1 -> 0, {D})",
            f"vs = X(0 -> 1, {D})",
            f"ve = Z(1 -> 0, {D})",
            "wire hv.out0 -- v.in0, hh.out0 -- v.in1, hs.out0 -- hh.in0, v.out1 -- he.in0",
            "wire vs.out0 -- hv.in0, v.out2 -- ve.in0",
            f"repeat {c.N} as R: v, hv, hh, hs, he",
            f"repeat {c.N} as C in R: v, hv, hh",
            "chain C: v.out1 -> hh.in0",
            "chain R: v.out2 -> hv.in0",
            f"repeat {c.N} as PH: vs",
            f"repeat {c.N} as PT: ve",
            "outputs v.out0",
            f"scalar {c.d}^((({c.n})^2 - 2*({c.n}))/2)",
        ]
    )


SUBJECT = Subject(
    id="cluster_state_2d",
    name="Two-dimensional cluster state",
    category="circuit",
    source="raussendorf2001one",
    size_parameter="grid side",
    dimension="general",
    domain=Domain(n_min=1, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
