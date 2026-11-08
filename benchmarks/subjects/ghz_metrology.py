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

"""GHZ-state phase estimation at the Heisenberg limit: a GHZ state of 1 + n qubits, the phase
P(1/8) on every qubit, the CX fan undone; the first qubit carries (|0> + e^(2 pi i (n + 1)/8)
|1>)/sqrt(2), the rest |0>^n."""

from fractions import Fraction

from benchmarks.subjects._common import *


def _circuit(n, d):
    """GHZ, phases, disentangle."""
    gates = [
        Init("a"),
        F("a"),
        Init("r"),
        CX("a", "r"),
        ZPhase("a", {1: Fraction(1, 8)}),
        ZPhase("r", {1: Fraction(1, 8)}),
        CX("a", "r"),
    ]
    return circuit(["a", "r"], gates, registers={"r": n})


def _spec(c):
    """The accumulated phase on a, |0>^n."""
    return (
        f"a = Z(0 -> 1, {c.D}) phase{{1: ({c.n} + 1)/8}}\nrepeat {c.N}: X(0 -> 1, {c.D})\n"
        f"scalar 2^(-({c.n} + 1)/2)"
    )


SUBJECT = Subject(
    id="ghz_metrology",
    name="GHZ phase estimation",
    category="protocol",
    source="bollinger1996optimal",
    size_parameter="qubits after the first",
    dimension=2,
    domain=Domain(n_min=0, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
