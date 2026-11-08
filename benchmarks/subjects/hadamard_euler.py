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

"""Euler decomposition of the Hadamard gate: S, the X-phase pi/2, S equal e^(i pi/4) H."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import XP, S


def _circuit(n, d):
    """S, X phase 1/4, S, the global phase removed."""
    return circuit(["q"], [S("q"), XP("q", Fraction(1, 4)), S("q")], scalar="exp(-I*pi/4)")


def _spec(c):
    """H."""
    return f"h = F({c.D})"


SUBJECT = Subject(
    id="hadamard_euler",
    name="Euler decomposition of H",
    category="circuit",
    source="coecke2008interacting",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
