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

"""Margolus relative-phase Toffoli from three CX and four Ry(+-pi/4); equals the Toffoli gate
with the sign of |101> flipped."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ry
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """Ry, CX(b, c), Ry, CX(a, c), Ry^-1, CX(b, c), Ry^-1 on the target."""
    t = Fraction(1, 8)
    gates = [
        *ry("c", t),
        CX("b", "c"),
        *ry("c", t),
        CX("a", "c"),
        *ry("c", -t),
        CX("b", "c"),
        *ry("c", -t),
    ]
    return circuit(["a", "b", "c"], gates)


def _defining():
    return [
        MCX(("a", "b"), "c"),
        PhaseFn(("a", "b", "c"), lambda x: Fraction(1, 2) if x == (1, 0, 1) else 0),
    ]


def _spec(c):
    """Toffoli and the phase -1 at |101>."""
    return defining(c, ["a", "b", "c"], _defining())


def _reference(n, d):
    """MCX and the phase table."""
    return circuit(["a", "b", "c"], _defining())


SUBJECT = Subject(
    id="margolus_gate",
    name="Margolus gate",
    category="circuit",
    source="barenco1995elementary",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
