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

"""Teleportation of n qudits in parallel, one Bell pair per qudit; the channel is the identity on
the register."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """The teleportation circuit on registers q, a, b."""
    gates = [
        Init("a"),
        F("a"),
        Init("b"),
        CX("a", "b"),
        CX("q", "a", -1),
        F("q"),
        Measure("q"),
        Measure("a"),
        CX("a", "b", -1),
        CZ("q", "b", -1),
        Discard("q"),
        Discard("a"),
    ]
    return circuit(["q", "a", "b"], gates, registers={"q": n, "a": n, "b": n})


def _spec(c):
    """The identity on n qudits."""
    return f"repeat {c.N}: Z(1 -> 1, {c.D})"


SUBJECT = Subject(
    id="teleportation_register",
    name="Register teleportation",
    category="protocol",
    source="bennett1993teleporting",
    size_parameter="qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
