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

"""DEJMPS entanglement purification of two Bell pairs, each phase-flipped with probability 1/4:
Rx(pi/2) on Alice's halves and Rx(-pi/2) on Bob's turn phase flips into bit flips, bilateral CX,
the target pair's parity post-selected even; the kept pair is (9/16) Phi+ + (1/16) Psi+."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import H, P, ry


def _rx(q, t):
    return [H(q), P(q, t), H(q)]


def _noisy_pair(a, b, e):
    return [
        Init(a),
        F(a),
        Init(b),
        CX(a, b),
        Init(e),
        *ry(e, Fraction(1, 6)),
        Measure(e),
        CZ(e, b),
        Discard(e),
    ]


def _circuit(n, d):
    """Two phase-flipped pairs, the rotations, bilateral CX, parity post-selected."""
    gates = [*_noisy_pair("a1", "b1", "e1"), *_noisy_pair("a2", "b2", "e2")]
    for a, b in (("a1", "b1"), ("a2", "b2")):
        gates += [*_rx(a, Fraction(1, 4)), *_rx(b, Fraction(-1, 4))]
    gates += [CX("a1", "a2"), CX("b1", "b2"), CX("a2", "b2"), Post("b2"), Discard("a2")]
    return circuit(["a1", "b1", "e1", "a2", "b2", "e2"], gates)


def _spec(c):
    """A Bell pair with a bit flip of relative weight 1/9, purified and traced."""
    D = c.D
    return "\n".join(
        [
            f"z = Z(0 -> 2, {D})",
            f"e = Z(0 -> 2, {D}) weight{{1: 1/3}}",
            f"x = X(2 -> 1, {D})",
            f"g = G(1 -> 0, {D})",
            "wire z.out1 -- x.in0, e.out0 -- x.in1, e.out1 -- g.in0",
            "outputs z.out0, x.out0",
            "scalar 3/4",
        ]
    )


SUBJECT = Subject(
    id="dejmps_purification",
    name="DEJMPS entanglement purification",
    category="protocol",
    source="deutsch1996quantum",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
