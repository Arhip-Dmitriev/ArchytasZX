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

"""Grover search on two qubits with the marked item |11>: one iteration (oracle CZ, diffusion
H X CZ X H) maps |00> to the marked item exactly; the circuit carries the diffusion's global
phase -1."""

from benchmarks.subjects._common import CZ, Domain, Equation, F, Init, Subject, X, circuit


def _circuit(n, d):
    """H on both, CZ oracle, diffusion."""
    gates = [Init("a"), Init("b"), F("a"), F("b"), CZ("a", "b")]
    gates += [F("a"), F("b"), X("a"), X("b"), CZ("a", "b"), X("a"), X("b"), F("a"), F("b")]
    return circuit(["a", "b"], gates, scalar="-1")


def _spec(c):
    """|11>."""
    return f"a = X(0 -> 1, {c.D}) phase{{1: 1/2}}\nb = X(0 -> 1, {c.D}) phase{{1: 1/2}}\nscalar 1/2"


def _reference(n, d):
    """|11> by X on both."""
    return circuit(["a", "b"], [Init("a"), Init("b"), X("a"), X("b")])


SUBJECT = Subject(
    id="grover",
    name="Grover search, two qubits",
    category="algorithm",
    source="grover1996fast",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed", reference=_reference),),
)
