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

"""Deutsch-Jozsa over Z_d: constant f = 1 (the oracle is X on the ancilla) and balanced
f(x) = x_a (one CX from the first qudit); the register after F is |0>^n when f is constant
and |1> (x) |0>^n when balanced, the ancilla F|1> picking up the kicked-back phase."""

from benchmarks.subjects._common import CX, Domain, Equation, F, Init, Subject, X, circuit


def _constant(n, d):
    """Register in F|0>, oracle X^1 on the ancilla F|1>, F^3 on the register."""
    gates = [Init("x"), F("x"), Init("y", 1), F("y"), X("y", 1), F("x", 3)]
    return circuit(["x", "y"], gates, registers={"x": n})


def _constant_spec(c):
    """w^(-1) |0>^n (x) F|1>."""
    return (
        f"repeat {c.N}: X(0 -> 1, {c.D})\n"
        f"y = Z(0 -> 1, {c.D}) phase{{k: k/{c.d}}}\n"
        f"scalar exp(-2*pi*I/{c.d}) * {c.d}^(-({c.n}+1)/2)"
    )


def _balanced(n, d):
    """f(x) = x_a: one CX from a onto the ancilla; F on a, F^3 on the rest."""
    gates = [
        Init("a"),
        F("a"),
        Init("x"),
        F("x"),
        Init("y", 1),
        F("y"),
        CX("a", "y"),
        F("a"),
        F("x", 3),
    ]
    return circuit(["a", "x", "y"], gates, registers={"x": n})


def _balanced_spec(c):
    """|1> (x) |0>^n (x) F|1>."""
    return (
        f"a = X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}\n"
        f"repeat {c.N}: X(0 -> 1, {c.D})\n"
        f"y = Z(0 -> 1, {c.D}) phase{{k: k/{c.d}}}\n"
        f"scalar {c.d}^(-({c.n}+2)/2)"
    )


SUBJECT = Subject(
    id="deutsch_jozsa",
    name="Deutsch-Jozsa",
    category="algorithm",
    source="deutsch1992rapid",
    size_parameter="register qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(
        Equation("constant", _constant, _constant_spec, form="boxed"),
        Equation("balanced", _balanced, _balanced_spec, form="boxed"),
    ),
)
