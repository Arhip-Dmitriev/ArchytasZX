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
T_n(A) = b_n A for odd n = 2k + 1, b_n = 2^(1/2) cos(n pi/4); the specification is
cos((2k + 1) pi/4) H, k the size parameter."""

from benchmarks.subjects._common import *


def _walk(a, s):
    """U then R."""
    return [F(a), X(a), CX(a, s), X(a), CZ(a, s), F(a), Z(a)]


def _unrolled(k, d):
    """Ancilla |0>, 2k + 1 walk steps, ancilla post-selected on <0|."""
    gates = [Init("a")]
    for _ in range(2 * k + 1):
        gates += _walk("a", "s")
    return circuit(["a", "s"], [*gates, Post("a")])


def _circuit(k, d):
    """Ancilla |0>, a chain of k copies of two walk steps on a and s, one more walk step,
    ancilla post-selected on <0|."""
    body = (*_walk("a", "s"), *_walk("a", "s"))
    gates = [Init("a"), Chain(k, body, through=("a", "s")), *_walk("a", "s"), Post("a")]
    return circuit(["a", "s"], gates)


def _spec(c):
    """cos((2k + 1) pi/4) H."""
    turn = f"({c.N}/4 + 1/8)"
    return f"f = F({c.D})\nscalar (exp(2*pi*I*{turn}) + exp(-2*pi*I*{turn}))/2"


SUBJECT = Subject(
    id="qubitization_chebyshev",
    name="Qubitization, Chebyshev block encoding",
    category="algorithm",
    source="low2019hamiltonian",
    size_parameter="walk step pairs after the first step",
    dimension=2,
    domain=Domain(n_min=0, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
