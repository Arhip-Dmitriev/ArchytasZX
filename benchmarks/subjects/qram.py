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

"""Quantum RAM lookup with n address bits and 2^n memory bits by swap routing: per address bit
(least significant first) controlled swaps pair the memory cells and bring the addressed cell to
cell 0, CX onto the output, the routing undone; equals one MCX per address value v, controlled on
a = v and m_v, onto t."""

from fractions import Fraction

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap
from benchmarks.subjects._spec import defining


def _wires(n):
    return unrolled("a", n), unrolled("m", 2**n), "t"


def _circuit(n, d):
    """Route, copy, unroute."""
    a, m, t = _wires(n)
    route = []
    for j in range(n):
        for i in range(0, 2**n, 2 ** (j + 1)):
            route += cswap(a[j], m[i], m[i + 2**j])
    unroute = []
    for j in reversed(range(n)):
        for i in reversed(range(0, 2**n, 2 ** (j + 1))):
            unroute += cswap(a[j], m[i], m[i + 2**j])
    return circuit([*a, *m, t], [*route, CX(m[0], t), *unroute])


def _flip(x):
    return Fraction(x[-1], 2)


def _mcx(cs, t, values):
    """MCX of ``cs`` = ``values`` onto t: F, the phase table with its one nonzero entry, F^3."""
    return [F(t), PhaseFn((*cs, t), _flip, support=((*values, 1),)), F(t, 3)]


def _spec(c):
    """Per address value v, the MCX on a = v and m_v onto t."""
    n = c.nv
    a, m, t = _wires(n)
    gates = [
        g for v in range(2**n) for g in _mcx((*a, m[v]), t, (*((v >> j) & 1 for j in range(n)), 1))
    ]
    return defining(c, [*a, *m, t], gates)


SUBJECT = Subject(
    id="qram",
    name="Quantum RAM lookup",
    category="circuit",
    source="giovannetti2008quantum",
    size_parameter="address bits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="unrolled"),),
)
