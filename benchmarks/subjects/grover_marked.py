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

"""Grover search on two qubits, one iteration, for each of the four marked items: the oracle is CZ
conjugated by X on the zero bits of the mark; |00> goes to the mark (the diffusion's global
phase -1 carried by the circuit)."""

from benchmarks.subjects._common import *


def _make(mark):
    """The equation for one marked bit pair."""

    def _circuit(n, d):
        flips = [X(q) for q, b in zip("ab", mark) if b == 0]
        gates = [Init("a"), Init("b"), F("a"), F("b"), *flips, CZ("a", "b"), *flips]
        gates += [F("a"), F("b"), X("a"), X("b"), CZ("a", "b"), X("a"), X("b"), F("a"), F("b")]
        return circuit(["a", "b"], gates, scalar="-1")

    def _spec(c):
        ph = [" phase{1: 1/2}" if b else "" for b in mark]
        return f"a = X(0 -> 1, {c.D}){ph[0]}\nb = X(0 -> 1, {c.D}){ph[1]}\nscalar 1/2"

    def _reference(n, d):
        return circuit(["a", "b"], [Init("a"), Init("b"), *(X(q) for q, b in zip("ab", mark) if b)])

    return Equation(
        "m" + "".join(map(str, mark)), _circuit, _spec, form="fixed", reference=_reference
    )


SUBJECT = Subject(
    id="grover_marked",
    name="Grover search, every marked item",
    category="algorithm",
    source="grover1996fast",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(*(_make(m) for m in ((0, 0), (0, 1), (1, 0), (1, 1))),),
)
