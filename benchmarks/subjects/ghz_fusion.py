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

"""GHZ fusion: two Bell pairs (p, q) and (r, s), CX from q onto r, r measured, X^-m and the
antipode on s; p, q, s end in a three-party GHZ state."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Fuse the two pairs at q and r."""
    gates = [
        Init("p"),
        F("p"),
        Init("q"),
        CX("p", "q"),
        Init("r"),
        F("r"),
        Init("s"),
        CX("r", "s"),
        CX("q", "r"),
        Measure("r"),
        CX("r", "s", -1),
        Discard("r"),
        F("s", 2),
    ]
    return circuit(["p", "q", "r", "s"], gates)


def _spec(c):
    """sum_k |k, k, k> / sqrt(d)."""
    return f"g = Z(0 -> 3, {c.D})\nscalar {c.d}^(-1/2)"


SUBJECT = Subject(
    id="ghz_fusion",
    name="GHZ fusion",
    category="protocol",
    source="browne2005resource",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
