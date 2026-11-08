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

"""Grover search on one ququart with the marked item 2: F prepares the uniform state, the oracle
phase -1 at 2, the diffusion F diag(1, -1, -1, -1) F^dagger; one iteration lands on |2> exactly."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Uniform state, oracle, diffusion."""
    gates = [
        Init("q"),
        F("q"),
        ZPhase("q", {2: "1/2"}),
        F("q", 3),
        ZPhase("q", {1: "1/2", 2: "1/2", 3: "1/2"}),
        F("q"),
    ]
    return circuit(["q"], gates)


def _spec(c):
    """|2>."""
    return f"q = X(0 -> 1, {c.D}) phase{{k: -2*k/{c.d}}}\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="grover_ququart",
    name="Grover search on a ququart",
    category="algorithm",
    source="grover1996fast",
    size_parameter=None,
    dimension=4,
    domain=Domain(d_min=4, d_max=4),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
