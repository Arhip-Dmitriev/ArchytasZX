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

"""Period finding over Z_(d^2) for f(x) = x mod d (period d): both digits in F|0>, the low digit
copied into a fresh qudit that is discarded, the inverse QFT; the outcome is a multiple of d,
high digit uniform and low digit 0."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import inverse, qft


def _circuit(n, d):
    """Superposition, oracle, inverse QFT, measurement."""
    gates = [
        Init("x0"),
        F("x0"),
        Init("x1"),
        F("x1"),
        Init("y"),
        CX("x1", "y"),
        Discard("y"),
        *inverse(qft(["x0", "x1"], d)),
        Measure("x0"),
        Measure("x1"),
    ]
    return circuit(["x0", "x1", "y"], gates)


def _spec(c):
    """I/d (x) |0><0|."""
    return f"m = G(0 -> 1, {c.D})\nz = X(0 -> 1, {c.D})\noutputs m.out0, z.out0\nscalar {c.d}^(-1)"


SUBJECT = Subject(
    id="period_finding",
    name="Period finding",
    category="algorithm",
    source="shor1997polynomial",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", literal=True),),
)
