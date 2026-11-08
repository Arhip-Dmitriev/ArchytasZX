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

"""Logical AND computed by a Clifford+T Toffoli into a fresh qubit and uncomputed by
measurement: H and an X-basis measurement of the target, a classically controlled CZ(a, b); the
channel is the identity on (a, b)."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import toffoli


def _circuit(n, d):
    """Compute ab, measure it out in the X basis, correct by CZ."""
    gates = [
        Init("t"),
        *toffoli("a", "b", "t"),
        F("t"),
        Measure("t"),
        MCZ(("t", "a"), "b"),
        Discard("t"),
    ]
    return circuit(["a", "b", "t"], gates)


def _spec(c):
    """The identity on two qubits."""
    return f"i = Z(1 -> 1, {c.D})\nj = Z(1 -> 1, {c.D})"


SUBJECT = Subject(
    id="measurement_uncomputation",
    name="Measurement-based uncomputation",
    category="circuit",
    source="gidney2018halving",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
