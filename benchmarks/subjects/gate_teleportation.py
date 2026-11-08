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

"""Gate teleportation (Gottesman-Chuang) of the qudit phase gate P = diag(e^(2 pi i k^2/d)): a
Bell pair with P on its second half, a Bell measurement of the input with the first half, Pauli
corrections X^-a Z^-s Z^-2a; the channel is P."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Resource, Bell measurement, corrections."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        ZPhase("b", rule="k**2/d"),
        CX("q", "a", -1),
        F("q"),
        Measure("q"),
        Measure("a"),
        CX("a", "b", -1),
        CZ("q", "b", -1),
        CZ("a", "b", -1),
        CZ("a", "b", -1),
        Discard("q"),
        Discard("a"),
    ]
    return circuit(["q", "a", "b"], gates)


def _spec(c):
    """P."""
    return f"p = Z(1 -> 1, {c.D}) phase{{k: k^2/{c.d}}}"


SUBJECT = Subject(
    id="gate_teleportation",
    name="Gate teleportation",
    category="protocol",
    source="gottesman1999demonstrating",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
