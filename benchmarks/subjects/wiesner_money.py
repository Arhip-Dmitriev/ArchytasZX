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

"""Wiesner quantum money on n qubits: per qubit a secret key bit a and basis bit b, the state
H^b X^a |0>; the bank's verification undoes the basis, XORs the key and measures; every check
bit reads 0."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import ch


def _bit(q):
    """A uniform classical bit."""
    return [Init(q), F(q), Measure(q)]


def _circuit(n, d):
    """Minting and verification per qubit, the keys discarded."""
    gates = [
        *_bit("a"),
        *_bit("b"),
        Init("q"),
        CX("a", "q"),
        *ch("b", "q"),
        *ch("b", "q"),
        CX("a", "q"),
        Measure("q"),
        Discard("a"),
        Discard("b"),
    ]
    return circuit(["a", "b", "q"], gates, registers={"a": n, "b": n, "q": n})


def _spec(c):
    """|0>^n."""
    return f"repeat {c.N}: X(0 -> 1, {c.D})\nscalar 2^(-{c.n}/2)"


SUBJECT = Subject(
    id="wiesner_money",
    name="Wiesner quantum money",
    category="protocol",
    source="wiesner1983conjugate",
    size_parameter="qubits",
    dimension=2,
    domain=Domain(n_min=0, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
