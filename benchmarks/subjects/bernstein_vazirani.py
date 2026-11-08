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

"""Bernstein-Vazirani over Z_d^n with secret s = (1, ..., 1): the oracle |x, y> -> |x, y + s.x>
as one CX per qudit onto the ancilla F|1>, then F on the register; it returns |s>."""

from benchmarks.subjects._common import CX, Domain, Equation, F, Init, Subject, X, circuit, unrolled


def _circuit(n, d):
    """Register x in F|0>, ancilla y in F|1>, the CX fan, F on x."""
    gates = [Init("x"), F("x"), Init("y", 1), F("y"), CX("x", "y"), F("x")]
    return circuit(["x", "y"], gates, registers={"x": n})


def _spec(c):
    """|1>^n (x) F|1>."""
    return (
        f"repeat {c.N}: X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}\n"
        f"y = Z(0 -> 1, {c.D}) phase{{k: k/{c.d}}}\n"
        f"scalar {c.d}^(-({c.n}+1)/2)"
    )


def _reference(n, d):
    """|1>^n beside F|1>."""
    x = unrolled("x", n)
    return circuit([*x, "y"], [*(Init(q) for q in x), *(X(q) for q in x), Init("y", 1), F("y")])


SUBJECT = Subject(
    id="bernstein_vazirani",
    name="Bernstein-Vazirani",
    category="algorithm",
    source="bernstein1997complexity",
    size_parameter="register qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
    restricted=(("n >= 1",), ("d >= 2",)),
)
