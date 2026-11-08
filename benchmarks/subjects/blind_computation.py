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

"""Universal blind quantum computation, one step: the client's qubit P(theta)|+> with theta a
uniform multiple of pi/4 (three key bits), the server's CZ to |+> and measurement at delta = phi
+ theta + r pi (phi = pi/4, r a uniform bit), the client's correction X^(s + r); the output is H
P(-phi)|+> whatever the keys."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H


def _bit(q):
    return [Init(q), F(q), Measure(q)]


def _theta(a, sign):
    """P(sign theta) controlled by the key bits t0, t1, t2."""
    return [
        PhaseFn(("t0", a), lambda x: Fraction(sign * x[0] * x[1], 8)),
        PhaseFn(("t1", a), lambda x: Fraction(sign * x[0] * x[1], 4)),
        CZ("t2", a),
    ]


def _circuit(n, d):
    """Client preparation, server CZ and measurement, client decoding."""
    gates = [
        *_bit("t0"),
        *_bit("t1"),
        *_bit("t2"),
        *_bit("r"),
        Init("a"),
        H("a"),
        *_theta("a", 1),
        Init("b"),
        H("b"),
        CZ("a", "b"),
        ZPhase("a", {1: Fraction(-1, 8)}),
        *_theta("a", -1),
        CZ("r", "a"),
        H("a"),
        Measure("a"),
        CX("a", "b"),
        CX("r", "b"),
    ]
    gates += [Discard(q) for q in ("t0", "t1", "t2", "r", "a")]
    return circuit(["t0", "t1", "t2", "r", "a", "b"], gates)


def _spec(c):
    """H P(-pi/4) |+>."""
    return f"Z(0 -> 1, {c.D}) phase{{1: -1/8}} -> F({c.D})\nscalar 2^(-1/2)"


SUBJECT = Subject(
    id="blind_computation",
    name="Universal blind quantum computation",
    category="protocol",
    source="broadbent2009universal",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
