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

"""Linear combination of unitaries: PREPARE = F on a d-level ancilla, SELECT = sum_j |j><j| (x)
X^j (one CX), PREPARE^dagger, ancilla post-selected on |0>; the block encodes (1/d) sum_j X^j =
|+><+|."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F, CX, F^dagger on the ancilla, then <0|."""
    gates = [Init("a"), F("a"), CX("a", "q"), F("a", 3), Post("a")]
    return circuit(["a", "q"], gates)


def _spec(c):
    """|+><+| = Z(1 -> 0) Z(0 -> 1) / d."""
    return (
        f"e = Z(1 -> 0, {c.D})\ns = Z(0 -> 1, {c.D})\ninputs e.in0\noutputs s.out0\n"
        f"scalar {c.d}^(-1)"
    )


SUBJECT = Subject(
    id="lcu_shift_average",
    name="Linear combination of unitaries, shift average",
    category="algorithm",
    source="childs2012hamiltonian",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
