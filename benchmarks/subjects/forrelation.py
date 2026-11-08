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

"""Forrelation of the inner-product bent function with its dual (Phi = 1) on n qudit pairs: F,
the phase oracle CZ, F, the dual oracle CZ, F; the all-zero amplitude is 1 and the state is
|0>^(2n)."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Per pair: F F, CZ, F F, CZ, F F."""
    gates = [
        Init("x"),
        Init("y"),
        F("x"),
        F("y"),
        CZ("x", "y"),
        F("x"),
        F("y"),
        CZ("x", "y"),
        F("x"),
        F("y"),
    ]
    return circuit(["x", "y"], gates, registers={"x": n, "y": n})


def _spec(c):
    """|0>^(2n)."""
    return f"repeat {c.N}: X(0 -> 1, {c.D}), X(0 -> 1, {c.D})\nscalar {c.d}^(-{c.n})"


SUBJECT = Subject(
    id="forrelation",
    name="Forrelation",
    category="algorithm",
    source="aaronson2015forrelation",
    size_parameter="qudit pairs",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
