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

"""Block encoding of H = (X + Z)/2 by a linear combination of unitaries: PREPARE = H on the
ancilla, SELECT = X when the ancilla is 0 and Z when it is 1, PREPARE^dagger, the ancilla post-
selected on |0>; the block is (X + Z)/2 = H/sqrt(2)."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """PREPARE, SELECT, PREPARE^dagger, <0|."""
    gates = [Init("a"), F("a"), X("a"), CX("a", "q"), X("a"), CZ("a", "q"), F("a"), Post("a")]
    return circuit(["a", "q"], gates)


def _spec(c):
    """H / sqrt(2)."""
    return f"h = F({c.D})\nscalar 2^(-1/2)"


SUBJECT = Subject(
    id="block_encoding",
    name="Block encoding by LCU",
    category="algorithm",
    source="gilyen2019quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
