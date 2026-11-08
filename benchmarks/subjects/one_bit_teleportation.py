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

"""One-bit teleportation: the input and |+> joined by CZ, the input measured in the Fourier
basis, the outcome m corrected by X^-m; the channel is F."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """CZ, F^dagger and measurement, X correction."""
    gates = [
        Init("a"),
        F("a"),
        CZ("q", "a"),
        F("q", 3),
        Measure("q"),
        CX("q", "a", -1),
        Discard("q"),
    ]
    return circuit(["q", "a"], gates)


def _spec(c):
    """F."""
    return f"f = F({c.D})"


SUBJECT = Subject(
    id="one_bit_teleportation",
    name="One-bit teleportation",
    category="protocol",
    source="zhou2000methodology",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
