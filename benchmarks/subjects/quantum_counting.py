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

"""Quantum counting on N = 2 with one marked item: the Grover iterate G = X Z has eigenphases
1/4 and 3/4; phase estimation with two clock qubits (controlled G, and controlled G^2 = -I as Z
on the clock), the inverse QFT, the search register discarded; the clock's low bit is 1, the
high bit uniform, giving M = N sin^2(pi/4) = 1."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, qft


def _circuit(n, d):
    """Clock in |+>, controlled G^2 and G, inverse QFT, measurement."""
    gates = [
        Init("c0"),
        F("c0"),
        Init("c1"),
        F("c1"),
        Init("q"),
        F("q"),
        Z("c0"),
        CZ("c1", "q"),
        CX("c1", "q"),
        *inverse(qft(["c0", "c1"], 2)),
        Measure("c0"),
        Measure("c1"),
        Discard("q"),
    ]
    return circuit(["c0", "c1", "q"], gates)


def _spec(c):
    """I/2 (x) |1><1|."""
    return (
        f"m = G(0 -> 1, {c.D})\nz = X(0 -> 1, {c.D}) phase{{1: 1/2}}\noutputs m.out0, z.out0\n"
        "scalar 1/2"
    )


SUBJECT = Subject(
    id="quantum_counting",
    name="Quantum counting",
    category="algorithm",
    source="brassard1998quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
