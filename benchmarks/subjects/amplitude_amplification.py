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

"""Exact amplitude amplification: A = Ry(pi/3) prepares amplitude 1/2 on the good state |1>; one
round -A S_0 A^dagger S_chi triples the angle to pi/2 and yields |1>."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, S, Sdg


def _ry(q, t):
    """e^(i pi t) Ry(2 pi t) as S^dagger, H, P(t), H, S."""
    return [Sdg(q), H(q), P(q, t), H(q), S(q)]


def _circuit(n, d):
    """A|0>, then S_chi = Z, A^dagger, S_0 = -Z, A."""
    gates = [Init("q"), *_ry("q", "1/6"), Z("q"), *_ry("q", "-1/6"), Z("q"), *_ry("q", "1/6")]
    return circuit(["q"], gates, scalar="exp(-I*pi/6)")


def _spec(c):
    """|1>."""
    return f"q = X(0 -> 1, {c.D}) phase{{1: 1/2}}\nscalar 2^(-1/2)"


SUBJECT = Subject(
    id="amplitude_amplification",
    name="Amplitude amplification, exact single round",
    category="algorithm",
    source="brassard2002quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
