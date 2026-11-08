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

"""Superdense coding: classical dits a, b applied to Alice's half of a Bell pair as X^a and
Z^b, decoded by CX^-1, F^3 and the antipode F^2; the channel from (a, b) is the dephasing of
both dits."""

from benchmarks.subjects._common import (
    CX,
    CZ,
    Discard,
    Domain,
    Equation,
    F,
    Init,
    Measure,
    Subject,
    Swap,
    circuit,
)


def _circuit(n, d):
    """Encode two dits into one qudit of a Bell pair, then decode on both halves."""
    gates = [
        Measure("a"),
        Measure("b"),
        Init("p"),
        F("p"),
        Init("q"),
        CX("p", "q"),
        CX("a", "p"),
        CZ("b", "p"),
        Discard("a"),
        Discard("b"),
        CX("p", "q", -1),
        F("p", 3),
        F("q", 2),
        Swap("p", "q"),
    ]
    return circuit(["a", "b", "p", "q"], gates)


def _spec(c):
    """Dephasing of each dit, a onto the first output and b onto the second."""
    return (
        f"za = Z(1 -> 2, {c.D}); ga = G(1 -> 0, {c.D}); wire za.out1 -- ga.in0\n"
        f"zb = Z(1 -> 2, {c.D}); gb = G(1 -> 0, {c.D}); wire zb.out1 -- gb.in0\n"
        "inputs za.in0, zb.in0\noutputs za.out0, zb.out0"
    )


SUBJECT = Subject(
    id="superdense_coding",
    name="Superdense coding",
    category="protocol",
    source="bennett1992communication",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
