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

"""Remote state preparation of the phase state sum_k e^(2 pi i k^2/d) |k> / sqrt(d): a Bell
pair, Alice applies the phases and F^dagger and measures m, Bob corrects Z^m."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Bell pair, Alice's rotated measurement, Bob's correction."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        ZPhase("a", rule="k**2/d"),
        F("a", 3),
        Measure("a"),
        CZ("a", "b"),
        Discard("a"),
    ]
    return circuit(["a", "b"], gates)


def _spec(c):
    """The phase state."""
    return f"z = Z(0 -> 1, {c.D}) phase{{k: k^2/{c.d}}}\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="remote_state_preparation",
    name="Remote state preparation",
    category="protocol",
    source="bennett2001remote",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
