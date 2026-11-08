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

"""Qudit SWAP from three controlled adds and the antipode: CX(a,b), CX^-1(b,a), CX(a,b), F^2
on a."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """(a, b) -> (a, a+b) -> (-b, a+b) -> (-b, a) -> (b, a)."""
    return circuit(["a", "b"], [CX("a", "b"), CX("b", "a", -1), CX("a", "b"), F("a", 2)])


def _swap_spec(c):
    """Two crossed identity wires."""
    return (
        f"i = Z(1 -> 1, {c.D})\nj = Z(1 -> 1, {c.D})\ninputs i.in0, j.in0\noutputs j.out0, i.out0"
    )


def _reference(n, d):
    """The SWAP gate."""
    return circuit(["a", "b"], [Swap("a", "b")])


SUBJECT = Subject(
    id="qudit_swap",
    name="SWAP from controlled adds",
    category="circuit",
    source="gottesman1999fault",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _swap_spec, form="fixed", reference=_reference),),
)
