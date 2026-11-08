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

"""Toffoli from a temporary logical AND: the 4-T AND into a fresh qubit, CX from it onto the
target, the AND uncomputed by an X-basis measurement and a classically controlled CZ; the
channel is the Toffoli gate."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import AND_GATES, H
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """Compute, copy, measure out."""
    gates = [
        *AND_GATES("a", "b", "t"),
        CX("t", "c"),
        H("t"),
        Measure("t"),
        MCZ(("t", "a"), "b"),
        Discard("t"),
    ]
    return circuit(["a", "b", "c", "t"], gates)


def _spec(c):
    """The Toffoli gate."""
    return defining(c, ["a", "b", "c"], [MCX(("a", "b"), "c")])


SUBJECT = Subject(
    id="toffoli_temporary_and",
    name="Toffoli via temporary AND",
    category="circuit",
    source="jones2013low",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
