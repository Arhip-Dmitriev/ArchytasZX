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

"""Amplitude estimation of a = sin^2(pi/8) prepared by A = Ry(pi/4): the Grover operator Q = A Z
A^dagger Z = Ry(pi/2) has eigenphases +-1/8; phase estimation on three clock qubits (controlled
Q, Q^2 = Q Q, Q^4 = -I as Z on the clock), the inverse QFT, the register discarded; the clock is
|001> or |111> with probability 1/2 each."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cry, inverse, qft, ry

_T = Fraction(1, 8)


def _cq(c, q):
    """Controlled Q: CZ, controlled A^dagger, CZ, controlled A."""
    return [CZ(c, q), *cry(c, q, -_T), CZ(c, q), *cry(c, q, _T)]


def _circuit(n, d):
    """A|0>, clock in |+>, controlled powers of Q, inverse QFT, measurement."""
    clock = ["c0", "c1", "c2"]
    gates = [g for x in clock for g in (Init(x), F(x))] + [Init("q"), *ry("q", _T)]
    gates += [Z("c0"), *_cq("c1", "q"), *_cq("c1", "q"), *_cq("c2", "q")]
    gates += inverse(qft(clock, 2)) + [Measure(x) for x in clock] + [Discard("q")]
    return circuit([*clock, "q"], gates)


def _spec(c):
    """(|001><001| + |111><111|)/2."""
    return (
        f"z = Z(0 -> 3, {c.D})\ng = G(1 -> 0, {c.D})\none = X(0 -> 1, {c.D}) phase{{1: 1/2}}\n"
        "wire z.out2 -- g.in0\noutputs z.out0, z.out1, one.out0\nscalar 1/2"
    )


SUBJECT = Subject(
    id="amplitude_estimation",
    name="Amplitude estimation",
    category="algorithm",
    source="brassard2002quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
