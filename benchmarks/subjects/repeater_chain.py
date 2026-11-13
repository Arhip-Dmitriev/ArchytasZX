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

"""Entanglement swapping along a chain of n + 1 Bell pairs; the two end qudits share a Bell pair."""

from benchmarks.subjects._common import *


def _unrolled(n, d):
    """Bell pairs (a_i, b_i); a Bell measurement on (b_i, a_(i+1)), corrections on b_(i+1)."""
    a, b = unrolled("a", n + 1), unrolled("b", n + 1)
    gates = []
    for i in range(n + 1):
        gates += [Init(a[i]), F(a[i]), Init(b[i]), CX(a[i], b[i])]
    for i in range(n):
        x, y, t = b[i], a[i + 1], b[i + 1]
        gates += [
            CX(x, y, -1),
            F(x),
            Measure(x),
            Measure(y),
            CX(y, t, -1),
            CZ(x, t, -1),
            Discard(x),
            Discard(y),
        ]
    return circuit([w for i in range(n + 1) for w in (a[i], b[i])], gates)


def _circuit(n, d):
    """The Bell pair (a, b), then per copy a Bell pair (x, y) and a Bell measurement on the
    carried b and x, corrected on y, which goes on."""
    body = (
        Init("x"),
        F("x"),
        Init("y"),
        CX("x", "y"),
        CX("b", "x", -1),
        F("b"),
        Measure("b"),
        Measure("x"),
        CX("x", "y", -1),
        CZ("b", "y", -1),
        Discard("b"),
        Discard("x"),
    )
    gates = [Init("a"), F("a"), Init("b"), CX("a", "b"), Chain(n, body, carries=(("b", "y"),))]
    return circuit(["a", "b", "x", "y"], gates, registers={"x": n, "y": n})


def _spec(c):
    """sum_k |k, k> / sqrt(d) on the end qudits, the second leg through one identity per
    copy."""
    return "\n".join(
        [
            f"e = Z(0 -> 2, {c.D})",
            f"repeat {c.N} as C: i = Z(1 -> 1, {c.D})",
            "wire e.out1 -- i.in0",
            "chain C: i.out0 -> i.in0",
            "outputs e.out0, i.out0",
            f"scalar {c.d}^(-1/2)",
        ]
    )


SUBJECT = Subject(
    id="repeater_chain",
    name="Quantum repeater chain",
    category="protocol",
    source="briegel1998quantum",
    size_parameter="swaps",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
