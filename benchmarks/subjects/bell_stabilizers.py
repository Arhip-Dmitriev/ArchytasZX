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

"""Stabilizers of the Bell state sum_k |k, k>: X (x) X and Z (x) Z^-1 leave it unchanged."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Bell pair, then X X and Z Z^-1."""
    gates = [Init("a"), F("a"), Init("b"), CX("a", "b"), X("a"), X("b"), Z("a"), Z("b", -1)]
    return circuit(["a", "b"], gates)


def _spec(c):
    """sum_k |k, k> / sqrt(d)."""
    return f"e = Z(0 -> 2, {c.D})\nscalar {c.d}^(-1/2)"


def _reference(n, d):
    """The Bell pair."""
    return circuit(["a", "b"], [Init("a"), F("a"), Init("b"), CX("a", "b")])


SUBJECT = Subject(
    id="bell_stabilizers",
    name="Bell state stabilizers",
    category="circuit",
    source="gottesman1999fault",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
