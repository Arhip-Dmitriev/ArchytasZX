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

"""S-gate injection with the Y state S|+>: CX from the data onto it, the ancilla measured, a
classically controlled Z; the channel is S."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import S


def _circuit(n, d):
    """Injection with Z correction."""
    gates = [Init("m"), F("m"), S("m"), CX("q", "m"), Measure("m"), CZ("m", "q"), Discard("m")]
    return circuit(["q", "m"], gates)


def _spec(c):
    """S."""
    return f"s = Z(1 -> 1, {c.D}) phase{{1: 1/4}}"


SUBJECT = Subject(
    id="y_state_injection",
    name="Y-state injection",
    category="protocol",
    source="aliferis2006quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
