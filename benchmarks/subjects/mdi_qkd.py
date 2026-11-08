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

"""Measurement-device-independent key distribution, matching bases: Alice and Bob send BB84 states
of their uniform key bits a, k in the shared uniform basis bit b to Charlie, whose Bell
measurement (CX, H, both measured) announces the parity a + k on the Z outcome of qb when b = 0 and
of qa when b = 1; Bob adds that outcome to k; the key pair is perfectly correlated and uniform."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _bit(q):
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Keys and basis, BB84 states, Bell measurement, Bob's flip."""
    gates = [*_bit("a"), *_bit("k"), *_bit("b"), Init("qa"), Init("qb")]
    gates += [CX("a", "qa"), CX("k", "qb"), *ch("b", "qa"), *ch("b", "qb")]
    gates += [CX("qa", "qb"), F("qa"), Measure("qa"), Measure("qb")]
    gates += [MCX(("b", "qb"), "k", values=(0, 1)), MCX(("b", "qa"), "k", values=(1, 1))]
    gates += [Discard("b"), Discard("qa"), Discard("qb")]
    return circuit(["a", "k", "b", "qa", "qb"], gates)


def _spec(c):
    """sum_a (1/2) |a, a><a, a|: Z(0 -> 3) with one leg discarded."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out2 -- g.in0\n"
        "outputs z.out0, z.out1\nscalar 2^(-1/2)"
    )


SUBJECT = Subject(
    id="mdi_qkd",
    name="Measurement-device-independent key distribution",
    category="protocol",
    source="lo2012measurement",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
