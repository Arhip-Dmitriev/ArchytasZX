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

"""Shor's order finding for N = 15, a = 7, three clock qubits and a four-qubit work register in
|1>: clock qubit 2 controls multiplication by 7 (controlled rotation of the work bits, then
NOT), clock qubit 1 multiplication by 4 (two controlled swaps), clock qubit 0 the identity; the
inverse QFT, the work register discarded. The order is 4 and the clock is uniform on {0, 2, 4,
6}: its low bit |0>, the others maximally mixed."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap, inverse, qft

_W = ["w0", "w1", "w2", "w3"]


def times7(c, w):
    """Controlled multiplication by 7 mod 15 on w (w0 most significant): rotate right, NOT."""
    out = cswap(c, w[2], w[3]) + cswap(c, w[1], w[2]) + cswap(c, w[0], w[1])
    return out + [CX(c, q) for q in w]


def times4(c, w):
    """Controlled multiplication by 4 mod 15: rotate left by two."""
    return cswap(c, w[0], w[2]) + cswap(c, w[1], w[3])


def _circuit(n, d):
    """Clock in |+>, controlled multiplications, inverse QFT, work discarded, clock measured."""
    clock = ["c0", "c1", "c2"]
    gates = [g for q in clock for g in (Init(q), F(q))]
    gates += [Init(q) for q in _W] + [X("w3")]
    gates += times7("c2", _W) + times4("c1", _W)
    gates += inverse(qft(clock, 2))
    gates += [Discard(q) for q in _W] + [Measure(q) for q in clock]
    return circuit([*clock, *_W], gates)


def _spec(c):
    """I/2 (x) I/2 (x) |0><0|."""
    return (
        f"m0 = G(0 -> 1, {c.D})\nm1 = G(0 -> 1, {c.D})\nz = X(0 -> 1, {c.D})\n"
        "outputs m0.out0, m1.out0, z.out0\nscalar 2^(-3/2)"
    )


SUBJECT = Subject(
    id="shor_factoring",
    name="Shor's algorithm, N = 15",
    category="algorithm",
    source="shor1997polynomial",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="fixed"),),
)
