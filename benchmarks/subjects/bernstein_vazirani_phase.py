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

"""Bernstein-Vazirani with the phase oracle |x> -> w^(s.x) |x>, s = (1, ..., 1), applied as Z on
each qudit between F and F^3; returns |s>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F, the phase oracle Z, F^3."""
    return circuit(["x"], [Init("x"), F("x"), Z("x"), F("x", 3)], registers={"x": n})


def _spec(c):
    """|1>^n."""
    return f"repeat {c.N}: X(0 -> 1, {c.D}) phase{{k: -k/{c.d}}}\nscalar {c.d}^(-{c.n}/2)"


def _reference(n, d):
    """|1>^n by X."""
    x = unrolled("x", n)
    return circuit(x, [*(Init(q) for q in x), *(X(q) for q in x)])


SUBJECT = Subject(
    id="bernstein_vazirani_phase",
    name="Bernstein-Vazirani, phase oracle",
    category="algorithm",
    source="bernstein1997complexity",
    size_parameter="register qudits",
    dimension="general",
    domain=Domain(n_min=0, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
