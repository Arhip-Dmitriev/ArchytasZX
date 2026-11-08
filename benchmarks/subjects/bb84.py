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

"""BB84 key distribution, matching bases: Alice's uniform key bit a sets X^a, her uniform basis
bit b a controlled H; Bob undoes the basis with the same b and measures; the key pair is
perfectly correlated and uniform."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _bit(q):
    """A uniform classical bit."""
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Key and basis bits, encoding, decoding in the same basis, measurement."""
    gates = [
        *_bit("a"),
        *_bit("b"),
        Init("q"),
        CX("a", "q"),
        *ch("b", "q"),
        *ch("b", "q"),
        Measure("q"),
        Discard("b"),
    ]
    return circuit(["a", "b", "q"], gates)


def _spec(c):
    """sum_a (1/2) |a, a><a, a|: Z(0 -> 3) with one leg discarded."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out2 -- g.in0\n"
        "outputs z.out0, z.out1\nscalar 2^(-1/2)"
    )


SUBJECT = Subject(
    id="bb84",
    name="BB84 key distribution",
    category="protocol",
    source="bennett1984quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
