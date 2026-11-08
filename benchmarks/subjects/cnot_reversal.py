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

"""Reversal of a controlled add by Fourier conjugation: F on both qudits, CX(a, b), F^dagger on
both; equals CX^-1(b, a), the qubit identity H H CX H H = CX reversed."""

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """F F, CX(a, b), F^3 F^3."""
    return circuit(["a", "b"], [F("a"), F("b"), CX("a", "b"), F("a", 3), F("b", 3)])


def _spec(c):
    """CX^-1 from b onto a."""
    return defining(c, ["a", "b"], [CX("b", "a", -1)])


SUBJECT = Subject(
    id="cnot_reversal",
    name="Controlled-add reversal",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
