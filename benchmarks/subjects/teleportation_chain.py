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

"""Teleportation relayed over n + 1 hops, each with its own Bell pair; the channel from the first
qudit to the last is the identity."""

from benchmarks.subjects._common import *


def _unrolled(n, d):
    """Teleport q_i to q_(i+1) through the pair (a_i, q_(i+1))."""
    q, a = unrolled("q", n + 2), unrolled("a", n + 1)
    gates = []
    for i in range(n + 1):
        gates += [Init(a[i]), F(a[i]), Init(q[i + 1]), CX(a[i], q[i + 1])]
    for i in range(n + 1):
        x, y, t = q[i], a[i], q[i + 1]
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
    return circuit([*q, *a], gates)


def _circuit(n, d):
    """Per copy: a Bell pair (a, t), the carried q teleported onto t, which goes on."""
    body = (
        Init("a"),
        F("a"),
        Init("t"),
        CX("a", "t"),
        CX("q", "a", -1),
        F("q"),
        Measure("q"),
        Measure("a"),
        CX("a", "t", -1),
        CZ("q", "t", -1),
        Discard("q"),
        Discard("a"),
    )
    regs = {"t": f"{n} + 1", "a": f"{n} + 1"}
    return circuit(
        ["q", "t", "a"], [Chain(f"{n} + 1", body, carries=(("q", "t"),))], registers=regs
    )


def _spec(c):
    """The identity channel, one identity per copy."""
    return f"repeat {c.N} + 1 as C: i = Z(1 -> 1, {c.D})\nchain C: i.out0 -> i.in0"


SUBJECT = Subject(
    id="teleportation_chain",
    name="Teleportation chain",
    category="protocol",
    source="bennett1993teleporting",
    size_parameter="relays",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
