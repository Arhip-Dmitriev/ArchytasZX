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

"""Qudit teleportation: a Bell pair (a, b), CX^-1 from q onto a and F on q, both measured, the
corrections X^-m(a) and Z^-m(q) on b; the channel from q to b is the identity."""

from benchmarks.subjects._common import (
    CX,
    CZ,
    Discard,
    Domain,
    Equation,
    F,
    Init,
    Measure,
    Subject,
    circuit,
)


def _circuit(n, d):
    """Bell pair, Bell measurement, classically controlled corrections."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        CX("q", "a", -1),
        F("q"),
        Measure("q"),
        Measure("a"),
        CX("a", "b", -1),
        CZ("q", "b", -1),
        Discard("q"),
        Discard("a"),
    ]
    return circuit(["q", "a", "b"], gates)


def _spec(c):
    """The identity channel."""
    return f"i = Z(1 -> 1, {c.D})"


SUBJECT = Subject(
    id="teleportation",
    name="Quantum teleportation",
    category="protocol",
    source="bennett1993teleporting",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
