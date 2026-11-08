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

"""Qubitization: U = H_a (|0><0| (x) X + |1><1| (x) Z) H_a is a Hermitian block encoding of
A = (X + Z)/2; the walk (R U)^n with R = 2|0><0| - I = Z_a on the ancilla block-encodes the
Chebyshev polynomial T_n(A). With A^2 = I/2 the recurrence T_(k+1) = 2 A T_k - T_(k-1) gives
T_n(A) = b_n A for odd n; the specification is b_n 2^(-1/2) H."""

from benchmarks.subjects._common import *


def _walk(a, s):
    """U then R."""
    return [F(a), X(a), CX(a, s), X(a), CZ(a, s), F(a), Z(a)]


def _circuit(n, d):
    """Ancilla |0>, n walk steps, ancilla post-selected on <0|."""
    gates = [Init("a")]
    for _ in range(n):
        gates += _walk("a", "s")
    return circuit(["a", "s"], [*gates, Post("a")])


def _coefficient(n):
    """b_n with T_n(A) = a_n I + b_n A, from T_0 = I, T_1 = A and A^2 = I/2."""
    prev, cur = (1, 0), (0, 1)
    for _ in range(n - 1):
        prev, cur = cur, (cur[1] - prev[0], 2 * cur[0] - prev[1])
    return cur[1]


def _spec(c):
    """b_n 2^(-1/2) H."""
    return f"f = F({c.D})\nscalar ({_coefficient(c.nv)}) * 2^(-1/2)"


SUBJECT = Subject(
    id="qubitization_chebyshev",
    name="Qubitization, Chebyshev block encoding",
    category="algorithm",
    source="low2019hamiltonian",
    size_parameter="walk steps",
    dimension=2,
    domain=Domain(n_min=1, n_step=2, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
