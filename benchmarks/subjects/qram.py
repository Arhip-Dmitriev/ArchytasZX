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

"""Quantum RAM lookup with n address bits and 2^n memory bits: for each address value v, one
multi-controlled X onto t, controlled on the address bits at the bits of v (copy j of the address
register a the bit of value 2^j) and on memory cell m_v; equals, on the address bound into one
2^n register, a phase gadget per value v testing a - v = 0, m_v = 1 and t = 1 in the Fourier
basis of t. The swap-routing circuit (per address bit, least significant first, controlled swaps
bring the addressed cell to cell 0, CX onto the output, the routing undone) is the reference."""

from benchmarks.subjects._common import *
from benchmarks.subjects._gates import cswap


def _reference(n, d):
    """Route, copy, unroute."""
    a, m, t = unrolled("a", n), unrolled("m", 2**n), "t"
    route = []
    for j in range(n):
        for i in range(0, 2**n, 2 ** (j + 1)):
            route += cswap(a[j], m[i], m[i + 2**j])
    unroute = []
    for j in reversed(range(n)):
        for i in reversed(range(0, 2**n, 2 ** (j + 1))):
            unroute += cswap(a[j], m[i], m[i + 2**j])
    return circuit([*a, *m, t], [*route, CX(m[0], t), *unroute])


def _circuit(n, d):
    """One MCX per address value v on a = v and m_v onto t."""
    return circuit(
        ["a", "m", "t"],
        [IndexedMCX("a", "m", "t")],
        registers={"a": n, "m": f"2**{n}"},
        dims={"a": 2, "m": 2, "t": 2},
    )


def _spec(c):
    """The address bound copy by copy (binder chain V) into one 2^n wire, copied into each value
    copy v (box W), shifted by -v and bound with m_v and t to a Z effect with phase 1/2 at
    (0, 1, 1); t between F and F^3; the address split back (splitter chain U)."""
    N = c.N
    return "\n".join(
        [
            "h = Z(0 -> 1, literal 1)",
            "b = B(literal 2, literal 2^m)",
            f"z = Z(1 -> 2, literal 2^{N})",
            f"x = X(1 -> 1, literal 2^{N}) phase{{k: v*k/2^{N}}}",
            "c = Z(1 -> 2, literal 2)",
            f"p = B(literal 2^{N}, literal 2)",
            f"q = B(literal 2^({N} + 1), literal 2)",
            f"g = Z(1 -> 0, literal 2^({N} + 2)) phase{{3: 1/2}}",
            "f0 = F(literal 2)",
            "y = Z(1 -> 2, literal 2)",
            "f1 = F(literal 2)",
            "f2 = F(literal 2)",
            "f3 = F(literal 2)",
            "s = S(literal 2, literal 2^m)",
            "e = Z(1 -> 0, literal 1)",
            "wire h.out0 -- b.in1, b.out0 -- z.in0, z.out0 -- s.in0, s.out1 -- e.in0",
            "wire z.out1 -- x.in0, x.out0 -- p.in0, c.out1 -- p.in1, p.out0 -- q.in0",
            "wire y.out1 -- q.in1, q.out0 -- g.in0",
            "wire f0.out0 -- y.in0, y.out0 -- f1.in0, f1.out0 -- f2.in0, f2.out0 -- f3.in0",
            f"repeat {N} as V index m: b",
            "chain V: b.out0 -> b.in1",
            f"repeat 2^{N} as W index v: x, c, p, q, g",
            f"repeat {N} as U index m: s",
            "chain U: s.in0 -> s.out1",
            "inputs b.in0, c.in0, f0.in0",
            "outputs s.out0, c.out0, f3.out0",
        ]
    )


SUBJECT = Subject(
    id="qram",
    name="Quantum RAM lookup",
    category="circuit",
    source="giovannetti2008quantum",
    size_parameter="address bits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
