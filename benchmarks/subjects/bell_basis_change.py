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

"""Bell measurement unitary on a Bell pair: CX^-1 and F^3 map sum_k |k, k> / sqrt(d) to |0, 0>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Prepare the Bell pair, undo it."""
    return circuit(
        ["a", "b"], [Init("a"), F("a"), Init("b"), CX("a", "b"), CX("a", "b", -1), F("a", 3)]
    )


def _spec(c):
    """|0, 0>."""
    return f"a = X(0 -> 1, {c.D})\nb = X(0 -> 1, {c.D})\nscalar {c.d}^(-1)"


def _reference(n, d):
    """|0, 0>."""
    return circuit(["a", "b"], [Init("a"), Init("b")])


SUBJECT = Subject(
    id="bell_basis_change",
    name="Bell basis change",
    category="circuit",
    source="nielsen2010quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
