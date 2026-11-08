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

"""BB84 under intercept-resend: Eve measures each qubit in a uniform basis and resends it; with
matching Alice-Bob bases Bob's bit differs from Alice's with probability 1/4."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _bit(q):
    """A uniform classical bit."""
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Encoding, Eve's measurement in basis e, Bob's decoding."""
    gates = [
        *_bit("a"),
        *_bit("b"),
        *_bit("e"),
        Init("q"),
        CX("a", "q"),
        *ch("b", "q"),
        *ch("e", "q"),
        Measure("q"),
        *ch("e", "q"),
        *ch("b", "q"),
        Measure("q"),
        Discard("b"),
        Discard("e"),
    ]
    return circuit(["a", "b", "e", "q"], gates)


def _spec(c):
    """sum_a (1/2) |a><a| (x) ((3/4)|a><a| + (1/4)|1-a><1-a|), purified and traced."""
    D = c.D
    return "\n".join(
        [
            f"z = Z(0 -> 3, {D})",
            f"e = Z(0 -> 2, {D}) weight{{1: 1/sqrt(3)}}",
            f"x = X(2 -> 1, {D})",
            f"g1 = G(1 -> 0, {D})",
            f"g2 = G(1 -> 0, {D})",
            "wire z.out1 -- x.in0, e.out0 -- x.in1, z.out2 -- g1.in0, e.out1 -- g2.in0",
            "outputs z.out0, x.out0",
            "scalar sqrt(3)/2",
        ]
    )


SUBJECT = Subject(
    id="bb84_intercept_resend",
    name="BB84 under intercept-resend",
    category="protocol",
    source="bennett1984quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
