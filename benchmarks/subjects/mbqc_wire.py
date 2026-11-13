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

"""Measurement-based quantum wire: the input teleported along a linear cluster of n more qudits,
each link a CZ to |+>, each measured qudit read in the Fourier basis and its byproduct X^-m
corrected on the next; the channel is F^n."""

from benchmarks.subjects._common import *


def _unrolled(n, d):
    """n one-bit teleportations in a row."""
    q = ["q", *unrolled("a", n)]
    gates = []
    for i in range(n):
        gates += [
            Init(q[i + 1]),
            F(q[i + 1]),
            CZ(q[i], q[i + 1]),
            F(q[i], 3),
            Measure(q[i]),
            CX(q[i], q[i + 1], -1),
            Discard(q[i]),
        ]
    return circuit(q, gates)


def _circuit(n, d):
    """Per copy one one-bit teleportation of the carried q onto a fresh a, which goes on."""
    body = (
        Init("a"),
        F("a"),
        CZ("q", "a"),
        F("q", 3),
        Measure("q"),
        CX("q", "a", -1),
        Discard("q"),
    )
    return circuit(["q", "a"], [Chain(n, body, carries=(("q", "a"),))], registers={"a": n})


def _spec(c):
    """F^n, one F per copy."""
    return f"repeat {c.N} as C: f = F({c.D})\nchain C: f.out0 -> f.in0"


SUBJECT = Subject(
    id="mbqc_wire",
    name="Measurement-based quantum wire",
    category="protocol",
    source="raussendorf2003measurement",
    size_parameter="cluster qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
