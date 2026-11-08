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

"""Toffoli gate from five two-qubit gates (Sleator-Weinfurter, Barenco et al.): controlled V
from b, CX(a, b), controlled V^dagger from b, CX(a, b), controlled V from a, with V = sqrt(X) =
H S H realised as H, controlled S, H."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, cphase
from benchmarks.subjects._spec import defining


def _cv(c, t, sign):
    """Controlled sqrt(X)^sign: H, controlled P(sign/4), H."""
    return [H(t), *cphase(c, t, Fraction(sign, 4)), H(t)]


def _circuit(n, d):
    """CV(b, c), CX(a, b), CV^dagger(b, c), CX(a, b), CV(a, c)."""
    gates = [*_cv("b", "c", 1), CX("a", "b"), *_cv("b", "c", -1), CX("a", "b"), *_cv("a", "c", 1)]
    return circuit(["a", "b", "c"], gates)


def _spec(c):
    """The Toffoli gate."""
    return defining(c, ["a", "b", "c"], [MCX(("a", "b"), "c")])


SUBJECT = Subject(
    id="sleator_weinfurter_toffoli",
    name="Sleator-Weinfurter Toffoli",
    category="circuit",
    source="barenco1995elementary",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
