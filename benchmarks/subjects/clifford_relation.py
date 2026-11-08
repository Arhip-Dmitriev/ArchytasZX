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

"""Single-qubit Clifford group relation (S H)^3 = e^(i pi/4) I."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, S


def _circuit(n, d):
    """S H three times."""
    return circuit(["q"], [g for _ in range(3) for g in (H("q"), S("q"))])


def _spec(c):
    """e^(i pi/4) times the identity."""
    return f"i = Z(1 -> 1, {c.D})\nscalar exp(I*pi/4)"


SUBJECT = Subject(
    id="clifford_relation",
    name="Clifford relation (SH)^3",
    category="circuit",
    source="gottesman1998heisenberg",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
