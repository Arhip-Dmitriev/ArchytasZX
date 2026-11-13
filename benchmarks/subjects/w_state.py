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


def _split(a, b):
    """Controlled Ry(pi/2) from a onto b, then CX(b, a)."""
    return [*cry(a, b, Fraction(1, 4)), CX(b, a)]


def _unrolled(m, d):
    """The splitting tree over m levels, qubits in leaf-major order (each holder followed by its
    partner)."""
    n = 2**m
    q = unrolled("q", n)
    gates = [Init(x) for x in q] + [X(q[0])]
    leaves = [0]
    for level in range(m):
        for i in range(2**level):
            gates += _split(q[i], q[i + 2**level])
        leaves = [x for i in leaves for x in (i, i + 2**level)]
    return circuit([q[i] for i in leaves], gates)


def _circuit(m, d):
    """|1> on q, then a binary tree of depth m whose copy splits its excitation with a fresh r."""
    body = (Init("r"), *_split("q", "r"))
    return circuit(["q"], [Init("q"), X("q"), Tree(m, body, "q", ("q", "r"))])


def _spec(c):
    """W(1 -> 2^m) on |1>, normalised."""
    return (
        f"x = X(0 -> 1, {c.D}) phase{{1: 1/2}}\nw = W(1 -> 1, {c.D})\nwire x.out0 -- w.in0\n"
        f"repeat 2^{c.N}: w.out0\nscalar 2^(-({c.N} + 1)/2)"
    )


SUBJECT = Subject(
    id="w_state",
    name="W state preparation",
    category="circuit",
    source="dur2000three",
    size_parameter="tree levels (2^m qubits)",
    dimension=2,
    domain=Domain(n_min=0, n_max=3, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed"),),
)
