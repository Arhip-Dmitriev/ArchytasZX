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

"""Nonlocal controlled add (Eisert et al.): one Bell pair, two classical dits and local operations
implement CX from Alice's c to Bob's t."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Bell pair (a, b); CX^-1(c, a), measure a, X^-1 on b; CX(b, t), F on b, measure b,
    Z^-1 on c."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        CX("c", "a", -1),
        Measure("a"),
        CX("a", "b", -1),
        Discard("a"),
        CX("b", "t"),
        F("b"),
        Measure("b"),
        CZ("b", "c", -1),
        Discard("b"),
    ]
    return circuit(["c", "a", "b", "t"], gates)


def _spec(c):
    """CX(c, t) as a channel."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"inputs z.in0, x.in0\noutputs z.out0, x.out0\nscalar {c.d}^(1/2)"
    )


SUBJECT = Subject(
    id="nonlocal_cnot",
    name="Nonlocal CNOT",
    category="protocol",
    source="eisert2000optimal",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
