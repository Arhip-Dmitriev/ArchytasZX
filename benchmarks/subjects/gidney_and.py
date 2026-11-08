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

"""Gidney's temporary logical AND: the target prepared in T H|0>, four CX, T^dagger on both
controls and T on the target, two CX, H and S; equals |a, b, 0> -> |a, b, ab> with four T gates."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import AND_GATES
from benchmarks.subjects._spec import defining


def _circuit(n, d):
    """The 4-T AND computation."""
    return circuit(["a", "b", "t"], AND_GATES("a", "b", "t"))


def _spec(c):
    """|0> on t, then the Toffoli gate."""
    return defining(c, ["a", "b", "t"], [Init("t"), MCX(("a", "b"), "t")])


SUBJECT = Subject(
    id="gidney_and",
    name="Temporary logical AND",
    category="circuit",
    source="gidney2018halving",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
