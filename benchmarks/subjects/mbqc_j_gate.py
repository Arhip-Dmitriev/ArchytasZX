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

"""Measurement-based J gate: the input measured at angle phi(k) = k^2/d after CZ to |+> (the
phase, F^dagger, measurement), the byproduct X^-m corrected; the channel is F P_phi."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Phase, CZ, F^dagger and measurement, X correction."""
    gates = [
        Init("a"),
        F("a"),
        ZPhase("q", rule="k**2/d"),
        CZ("q", "a"),
        F("q", 3),
        Measure("q"),
        CX("q", "a", -1),
        Discard("q"),
    ]
    return circuit(["q", "a"], gates)


def _spec(c):
    """P_phi then F."""
    return f"Z(1 -> 1, {c.D}) phase{{k: k^2/{c.d}}} -> F({c.D})"


SUBJECT = Subject(
    id="mbqc_j_gate",
    name="Measurement-based J gate",
    category="protocol",
    source="raussendorf2001one",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
