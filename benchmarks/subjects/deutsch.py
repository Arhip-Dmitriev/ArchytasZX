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

"""Deutsch's algorithm over Z_d for f(x) = a x + b with a, b in {0, 1}: the register after F, the
oracle |x, y> -> |x, y + f(x)> and F is |a> beside the ancilla F|1> with phase w^-b."""

from benchmarks.subjects._common import *


def _make(a, b):
    """The equation for f(x) = a x + b."""

    def _circuit(n, d):
        oracle = ([CX("x", "y")] if a else []) + ([X("y")] if b else [])
        gates = [Init("x"), F("x"), Init("y", 1), F("y"), *oracle, F("x")]
        return circuit(["x", "y"], gates)

    def _spec(c):
        ph = f" phase{{k: -{a}*k/{c.d}}}" if a else ""
        sc = f" * exp(-2*pi*I/{c.d})" if b else ""
        return (
            f"x = X(0 -> 1, {c.D}){ph}\ny = Z(0 -> 1, {c.D}) phase{{k: k/{c.d}}}\n"
            f"scalar {c.d}^(-1){sc}"
        )

    return Equation(f"f{a}{b}", _circuit, _spec, form="fixed")


SUBJECT = Subject(
    id="deutsch",
    name="Deutsch's algorithm",
    category="algorithm",
    source="deutsch1985quantum",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(*(_make(a, b) for a in (0, 1) for b in (0, 1)),),
)
