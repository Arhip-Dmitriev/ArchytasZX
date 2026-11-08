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

"""E91 entanglement-based key distribution, key rounds: a Bell pair, both halves measured along the
shared uniform setting s (a qutrit) among the analyser angles 0, pi/4, pi/2, each a controlled
Ry(2 phi) from the flags [s = 1], [s = 2] (computed and uncomputed by oracles); (Ry (x) Ry) fixes
the Bell pair, so the outcomes are perfectly correlated and uniform. The CHSH rounds are the
chsh_game subject."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cry


def _flags():
    return [
        Oracle(("s",), "f1", lambda x: int(x[0] == 1)),
        Oracle(("s",), "f2", lambda x: int(x[0] == 2)),
    ]


def _rotate(q):
    """Ry(pi/2) on flag 1, Ry(pi) on flag 2."""
    return [*cry("f1", q, Fraction(1, 4)), *cry("f2", q, Fraction(1, 2))]


def _circuit(n, d):
    """Setting trit, flags, Bell pair, rotations, measurements."""
    gates = [Init("s"), F("s"), Measure("s"), Init("f1"), Init("f2"), *_flags()]
    gates += [Init("a"), F("a"), Init("b"), CX("a", "b"), *_rotate("a"), *_rotate("b")]
    gates += [*_flags(), Measure("a"), Measure("b"), Discard("s"), Discard("f1"), Discard("f2")]
    return circuit(["s", "f1", "f2", "a", "b"], gates, dims={"s": 3})


def _spec(c):
    """sum_a (1/2) |a, a><a, a|: Z(0 -> 3) with one leg discarded."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\nwire z.out2 -- g.in0\n"
        "outputs z.out0, z.out1\nscalar 2^(-1/2)"
    )


SUBJECT = Subject(
    id="e91",
    name="E91 key distribution",
    category="protocol",
    source="ekert1991quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
