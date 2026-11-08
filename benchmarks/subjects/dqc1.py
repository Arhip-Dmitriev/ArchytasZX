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

"""DQC1 trace estimation (one clean qubit): a control in |+>, n maximally mixed qubits,
controlled U = Z^(x n) as CZ to each, H, measure; the control reads 0 with probability (1 + Re
tr(U)/2^n)/2."""

from fractions import Fraction

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Clean qubit, mixed register, controlled U, H, measure."""
    h = Fraction(1, 8)
    gates = [
        Init("c"),
        F("c"),
        Init("r"),
        F("r"),
        Measure("r"),
        ZPhase("r", {1: h}),
        CX("c", "r"),
        ZPhase("r", {1: -h}),
        CX("c", "r"),
        ZPhase("c", {1: "n/8"}),
        F("c"),
        Measure("c"),
        Discard("r"),
    ]
    return circuit(["c", "r"], gates, registers={"r": n})


def _spec(c):
    """The bit with distribution ((1 + r)/2, (1 - r)/2)."""
    r = f"(2^(-{c.n}/2) * (exp(I*pi*{c.n}/4) + exp(-I*pi*{c.n}/4))/2)"
    return (
        f"z = Z(0 -> 1, {c.D}) weight{{1: sqrt((1 - {r})/(1 + {r}))}}\nm = Z(1 -> 2, {c.D})\n"
        f"g = G(1 -> 0, {c.D})\nwire z.out0 -- m.in0, m.out1 -- g.in0\noutputs m.out0\n"
        f"scalar sqrt((1 + {r})/2)"
    )


SUBJECT = Subject(
    id="dqc1",
    name="DQC1 trace estimation",
    category="algorithm",
    source="knill1998power",
    size_parameter="mixed qubits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
