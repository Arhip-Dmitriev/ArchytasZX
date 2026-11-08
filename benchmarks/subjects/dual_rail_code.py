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

"""Dual-rail encoding of a qubit into two: |x> -> |x, 1 - x> by CX onto a fresh qubit and X; the
codewords have weight one."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """CX, then X on the second rail."""
    return circuit(["a", "b"], [Init("b"), CX("a", "b"), X("b")])


def _spec(c):
    """The copy spider with NOT on the second leg."""
    return (
        f"z = Z(1 -> 2, {c.D})\nn = X(1 -> 1, {c.D}) phase{{1: 1/2}}\nwire z.out1 -- n.in0\n"
        "outputs z.out0, n.out0"
    )


SUBJECT = Subject(
    id="dual_rail_code",
    name="Dual-rail code",
    category="code",
    source="knill2001scheme",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
