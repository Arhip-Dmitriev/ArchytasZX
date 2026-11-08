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

"""BBM92 entanglement-based key distribution: a Bell pair, both halves measured in the same
uniform basis (controlled H); the outcomes are perfectly correlated and uniform."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _bit(q):
    """A uniform classical bit."""
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Bell pair, shared basis bit, both measured."""
    gates = [
        *_bit("x"),
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        *ch("x", "a"),
        *ch("x", "b"),
        Measure("a"),
        Measure("b"),
        Discard("x"),
    ]
    return circuit(["x", "a", "b"], gates)


def _spec(c):
    """sum_a (1/2) |a, a><a, a|: Z(0 -> 3) with one leg discarded."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out2 -- g.in0\n"
        "outputs z.out0, z.out1\nscalar 2^(-1/2)"
    )


SUBJECT = Subject(
    id="bbm92",
    name="BBM92 key distribution",
    category="protocol",
    source="bennett1992quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
