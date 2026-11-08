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

"""QAOA, one layer, for MaxCut on the n-cycle: |+>^n, the cost unitary e^(-i gamma C) with gamma
= pi/4 as CX, P(-1/8), CX per edge, the mixer e^(-i beta X) per qubit; the specification writes
the cost unitary as the diagonal of the cut function."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._spec import defining

_BETA = Fraction(1, 8)


def _edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def _start(q):
    return [g for x in q for g in (Init(x), F(x))]


def _mixer(q):
    return [XPhase(x, {1: _BETA}) for x in q]


def _circuit(n, d):
    """|+>^n, CX P CX per edge, X phases."""
    q = unrolled("q", n)
    gates = _start(q)
    for a, b in _edges(n):
        gates += [CX(q[a], q[b]), ZPhase(q[b], {1: Fraction(-1, 8)}), CX(q[a], q[b])]
    return circuit(q, gates + _mixer(q))


def _spec(c):
    """|+>^n, the cut diagonal, X phases."""
    n = c.nv
    q = unrolled("q", n)

    def cut(x):
        return Fraction(-sum(x[a] ^ x[b] for a, b in _edges(n)), 8)

    return defining(c, q, [*_start(q), PhaseFn(tuple(q), cut), *_mixer(q)])


SUBJECT = Subject(
    id="qaoa_maxcut",
    name="QAOA MaxCut on a cycle",
    category="algorithm",
    source="farhi2014quantum",
    size_parameter="qubits",
    dimension=2,
    domain=Domain(n_min=3, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
