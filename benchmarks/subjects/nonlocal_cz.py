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

"""Nonlocal controlled phase: the nonlocal controlled add between F^3 and F on Bob's qudit
implements CZ from Alice's c to Bob's t."""

from benchmarks.subjects._common import *
from benchmarks.subjects.nonlocal_cnot import _circuit as _cnot


def _circuit(n, d):
    """F^3(t), the nonlocal CNOT protocol, F(t)."""
    c = _cnot(n, d)
    return circuit(c.wires, [F("t", 3), *c.gates, F("t")])


def _spec(c):
    """CZ(c, t) as a channel."""
    return (
        f"x = Z(1 -> 2, {c.D})\ny = Z(1 -> 2, {c.D})\nf = F({c.D})\n"
        "wire x.out1 -- f.in0, f.out0 -- y.out1\ninputs x.in0, y.in0\noutputs x.out0, y.out0\n"
        f"scalar {c.d}^(1/2)"
    )


SUBJECT = Subject(
    id="nonlocal_cz",
    name="Nonlocal CZ",
    category="protocol",
    source="eisert2000optimal",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
