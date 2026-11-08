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

"""W state on n = 2^m qubits by a binary tree: |1> on qubit 0, then per level each active qubit
splits its excitation with its partner by a controlled Ry(pi/2) and a CX back; the specification
is the W node fed |1>."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cry


def _circuit(m, d):
    """The splitting tree over m levels."""
    n = 2**m
    q = unrolled("q", n)
    gates = [Init(x) for x in q] + [X(q[0])]
    for level in range(m):
        for i in range(2**level):
            j = i + 2**level
            gates += cry(q[i], q[j], Fraction(1, 4)) + [CX(q[j], q[i])]
    return circuit(q, gates)


def _spec(c):
    """W(1 -> n) on |1>, normalised."""
    m = c.nv
    return (
        f"x = X(0 -> 1, {c.D}) phase{{1: 1/2}}\nw = W(1 -> {2**m}, {c.D})\nwire x.out0 -- w.in0\n"
        f"scalar 2^(-{m + 1}/2)"
    )


SUBJECT = Subject(
    id="w_state",
    name="W state preparation",
    category="circuit",
    source="dur2000three",
    size_parameter="tree levels (2^m qubits)",
    dimension=2,
    domain=Domain(n_min=0, n_max=3, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
