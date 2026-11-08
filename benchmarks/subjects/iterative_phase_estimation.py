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

"""Kitaev's iterative phase estimation of the eigenphase 3/4 = 0.11 (binary) of P(3/4) on |1>:
the first round kicks back U^2 and reads the low bit, the second kicks back U, removes the low
bit by a classically controlled P(-1/4), and reads the high bit; both bits are 1."""

from fractions import Fraction

from benchmarks.subjects._common import *


def _circuit(n, d):
    """Two Hadamard-test rounds with feed-forward."""
    gates = [
        Init("u", 1),
        Init("a"),
        F("a"),
        PhaseFn(("a", "u"), lambda x: Fraction(3 * 2 * x[0] * x[1], 4)),
        F("a"),
        Measure("a"),
        Init("b"),
        F("b"),
        PhaseFn(("b", "u"), lambda x: Fraction(3 * x[0] * x[1], 4)),
        PhaseFn(("a", "b"), lambda x: Fraction(-x[0] * x[1], 4)),
        F("b"),
        Measure("b"),
    ]
    return circuit(["a", "b", "u"], gates)


def _spec(c):
    """|1>|1>|1>."""
    return "\n".join(f"{q} = X(0 -> 1, {c.D}) phase{{1: 1/2}}" for q in "abu") + "\nscalar 2^(-3/2)"


SUBJECT = Subject(
    id="iterative_phase_estimation",
    name="Iterative phase estimation",
    category="algorithm",
    source="kitaev1995quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
