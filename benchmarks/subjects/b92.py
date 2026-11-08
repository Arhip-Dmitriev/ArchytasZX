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

"""B92 key distribution: Alice sends |0> for bit 0 and |+> for bit 1; Bob measures in a uniform
basis b and keeps the conclusive outcome 1; conclusive rounds have b = 1 - a, each with weight
1/8."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _bit(q):
    """A uniform classical bit."""
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Encoding by a controlled H, Bob's basis, the conclusive outcome post-selected."""
    gates = [*_bit("a"), *_bit("b"), Init("q"), *ch("a", "q"), *ch("b", "q"), Post("q", 1)]
    return circuit(["a", "b", "q"], gates)


def _spec(c):
    """sum_a (1/8) |a, 1 - a><a, 1 - a|."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nn = X(1 -> 1, {c.D}) phase{{1: 1/2}}\n"
        "wire z.out2 -- g.in0, z.out1 -- n.in0\noutputs z.out0, n.out0\nscalar 2^(-3/2)"
    )


SUBJECT = Subject(
    id="b92",
    name="B92 key distribution",
    category="protocol",
    source="bennett1992quantumb92",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
