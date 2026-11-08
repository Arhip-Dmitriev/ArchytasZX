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

"""Continuous-time quantum walk on the n-cube for time pi/2: e^(-i t sum_j X_j) factorises into
e^(-i t X) = H e^(-i t Z) H per qubit, and transfers |0>^n perfectly to (-i)^n |1>^n."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Per qubit: H, e^(-i pi/2 Z) = -i P(1/2), H."""
    gates = [Init("q"), F("q"), ZPhase("q", {1: "1/2"}), F("q")]
    return circuit(["q"], gates, registers={"q": n}, scalar="(-I)^n")


def _spec(c):
    """(-i)^n |1>^n."""
    return f"repeat {c.N}: X(0 -> 1, {c.D}) phase{{1: 1/2}}\nscalar (-I)^{c.n} * 2^(-{c.n}/2)"


SUBJECT = Subject(
    id="quantum_walk_hypercube",
    name="Quantum walk on the hypercube, perfect state transfer",
    category="algorithm",
    source="childs2003exponential",
    size_parameter="cube dimension",
    dimension=2,
    domain=Domain(n_min=0, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
