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

"""Shor's discrete-logarithm algorithm in the cyclic group Z_d, generator g = 1 and h = 2
(log_g h = 2): f(a, b) = a g - b h into a fresh qudit, discarded; after F on a and b the
pair is uniform on {(u, -2u)}, dephased."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F on a and b, f(a, b) = a - 2b into y, y discarded, F on a and b, both measured."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        F("b"),
        Init("y"),
        CX("a", "y"),
        CX("b", "y", -1),
        CX("b", "y", -1),
        Discard("y"),
        F("a"),
        F("b"),
        Measure("a"),
        Measure("b"),
    ]
    return circuit(["a", "b", "y"], gates)


def _spec(c):
    """sum_u |u, -2u>, both legs dephased."""
    D = c.D
    return "\n".join(
        [
            f"z = Z(0 -> 3, {D})",
            f"x = X(2 -> 1, {D})",
            f"f1 = F({D})",
            f"f2 = F({D})",
            f"ma = Z(1 -> 2, {D})",
            f"mb = Z(1 -> 2, {D})",
            f"ga = G(1 -> 0, {D})",
            f"gb = G(1 -> 0, {D})",
            (
                "wire z.out0 -- ma.in0, z.out1 -- x.in0, z.out2 -- x.in1, x.out0 -- f1.in0, "
                "f1.out0 -- f2.in0, f2.out0 -- mb.in0, ma.out1 -- ga.in0, mb.out1 -- gb.in0"
            ),
            "outputs ma.out0, mb.out0",
        ]
    )


SUBJECT = Subject(
    id="discrete_log",
    name="Shor's discrete logarithm, cyclic group",
    category="algorithm",
    source="shor1997polynomial",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
