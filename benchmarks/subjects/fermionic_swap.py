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

"""Fermionic swap: three CX and a CZ; equals |x, y> -> (-1)^(xy) |y, x>."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """CX(a, b), CX(b, a), CX(a, b), CZ(a, b)."""
    return circuit(["a", "b"], [CX("a", "b"), CX("b", "a"), CX("a", "b"), CZ("a", "b")])


def _spec(c):
    """SWAP and the sign at |11>."""
    return defining(
        c, ["a", "b"], [Swap("a", "b"), PhaseFn(("a", "b"), lambda x: Fraction(x[0] * x[1], 2))]
    )


SUBJECT = Subject(
    id="fermionic_swap",
    name="Fermionic swap",
    category="circuit",
    source="bravyi2002fermionic",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
