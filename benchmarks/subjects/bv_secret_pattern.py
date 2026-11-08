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

"""Bernstein-Vazirani over Z_d^n with the secret s_i = i mod d, the phase oracle Z^(s_i) on qudit i
between F and F^3; returns |s>."""

from benchmarks.subjects._common import *


def _circuit(n, d):
    """F, Z^(i) on qudit i, F^3."""
    x = unrolled("x", n)
    gates = [g for i, q in enumerate(x) for g in (Init(q), F(q), Z(q, i), F(q, 3))]
    return circuit(x, gates)


def _spec(c):
    """|s_0, ..., s_(n-1)>."""
    lines = [f"s{i} = X(0 -> 1, {c.D}) phase{{k: -{i}*k/{c.d}}}" for i in range(c.nv)]
    return "\n".join([*lines, f"scalar {c.d}^(-{c.nv}/2)"])


def _reference(n, d):
    """X^(i) on qudit i."""
    x = unrolled("x", n)
    return circuit(x, [g for i, q in enumerate(x) for g in (Init(q), X(q, i))])


SUBJECT = Subject(
    id="bv_secret_pattern",
    name="Bernstein-Vazirani, patterned secret",
    category="algorithm",
    source="bernstein1997complexity",
    size_parameter="register qudits",
    dimension="general",
    domain=Domain(n_min=1, d_min=1),
    equations=(Equation("main", _circuit, _spec, form="unrolled", reference=_reference),),
)
