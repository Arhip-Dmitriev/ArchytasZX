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

"""Magic state injection: the magic state T|+>, CX from the data onto it, the ancilla measured
and a classically controlled S applied; the channel is the T gate."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import T


def _circuit(n, d):
    """Injection with S correction."""
    gates = [
        Init("m"),
        F("m"),
        T("m"),
        CX("q", "m"),
        Measure("m"),
        PhaseFn(("m", "q"), lambda x: Fraction(x[0] * x[1], 4)),
        Discard("m"),
    ]
    return circuit(["q", "m"], gates)


def _spec(c):
    """T."""
    return f"t = Z(1 -> 1, {c.D}) phase{{1: 1/8}}"


SUBJECT = Subject(
    id="magic_state_injection",
    name="Magic state injection",
    category="protocol",
    source="bravyi2005universal",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
