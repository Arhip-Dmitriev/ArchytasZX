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

"""Hidden shift for the bent function f(x, y) = x.y over Z_d^(2n), shift s = (1, ..., 1): F, the
shifted oracle, F, the dual oracle, F; returns |s>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Per pair (x, y): F F, X X, CZ, X^-1 X^-1, F F, CZ, F F."""
    gates = [
        Init("x"),
        Init("y"),
        F("x"),
        F("y"),
        X("x"),
        X("y"),
        CZ("x", "y"),
        X("x", -1),
        X("y", -1),
        F("x"),
        F("y"),
        CZ("x", "y"),
        F("x"),
        F("y"),
    ]
    return circuit(["x", "y"], gates, registers={"x": n, "y": n})


def _spec(c):
    """|1>^(2n), copy-major."""
    return (
        f"repeat {c.N}: X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}, "
        f"X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}\n"
        f"scalar {c.d}^(-{c.n})"
    )


def _reference(n, d):
    """|1>^(2n) by X, copy-major."""
    w = [x for i in range(n) for x in (f"x{i}", f"y{i}")]
    return circuit(w, [*(Init(q) for q in w), *(X(q) for q in w)])


SUBJECT = Subject(
    id="hidden_shift",
    name="Hidden shift, inner-product bent function",
    category="algorithm",
    source="vandam2003quantum",
    size_parameter="qudit pairs",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
