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

"""Controlled phase CP(t) from P(t/2) on both qubits and P(-t/2) between two CX; the controlled
S (t = 1/4) and the controlled T (t = 1/8)."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cphase
from benchmarks.subjects._spec import diagonal


def _eq(key, t):
    def build(n, d):
        """P(t/2), P(t/2), CX, P(-t/2), CX."""
        return circuit(["a", "b"], cphase("a", "b", t))

    def spec(c):
        """The diagonal with phase t at |11>."""
        return diagonal(c, ["a", "b"], lambda x: t * x[0] * x[1])

    return Equation(key, build, spec, form="fixed")


SUBJECT = Subject(
    id="controlled_phase_decomposition",
    name="Controlled phase decomposition",
    category="circuit",
    source="barenco1995elementary",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(_eq("cs", Fraction(1, 4)), _eq("ct", Fraction(1, 8))),
)
