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

"""Entanglement swapping: Bell pairs (a, b) and (c, e), a Bell measurement on (b, c) with the
teleportation corrections on e; a and e end in a Bell pair."""

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
    """Two Bell pairs, the middle Bell measurement, corrections on e."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        Init("c"),
        F("c"),
        Init("e"),
        CX("c", "e"),
        CX("b", "c", -1),
        F("b"),
        Measure("b"),
        Measure("c"),
        CX("c", "e", -1),
        CZ("b", "e", -1),
        Discard("b"),
        Discard("c"),
    ]
    return circuit(["a", "b", "c", "e"], gates)


def _spec(c):
    """sum_k |k, k> / sqrt(d)."""
    return f"e = Z(0 -> 2, {c.D})\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="entanglement_swapping",
    name="Entanglement swapping",
    category="protocol",
    source="zukowski1993event",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
