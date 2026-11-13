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

"""Coined quantum walk on the cycle of 2^n sites, one step: the Hadamard coin, then the shift x
+ 1 when the coin is 1 and x - 1 when it is 0, each a multi-controlled incrementer; equals F on
the coin followed by the conditional shift x -> x + 2 coin - 1 on the bound position register.
The position register is listed low bit first."""

from benchmarks.subjects._common import *


def _inc(x, coin, value):
    """Increment of x (x[0] the top bit) controlled on coin = value: MCX from the top bit down."""
    n = len(x)
    return [MCX((coin, *x[i + 1 :]), x[i], (value,) + (1,) * (n - 1 - i)) for i in range(n)]


def _reference(n, d):
    """Coin, controlled increment, controlled decrement, the position listed low bit first."""
    x = unrolled("x", n)
    gates = [F("coin"), *_inc(x, "coin", 1), *reversed(_inc(x, "coin", 0))]
    return circuit(["coin", *reversed(x)], gates)


def _circuit(n, d):
    """Coin, then on the register x (copy m the bit of value 2^m) the increment controlled on
    coin = 1 (top copy down) and the decrement controlled on coin = 0 (bottom copy up)."""
    gates = [
        F("coin"),
        LowerMCX("x", ("coin",), (1,)),
        LowerMCX("x", ("coin",), (0,), reverse=True),
    ]
    return circuit(["coin", "x"], gates, registers={"x": n}, dims={"coin": 2, "x": 2})


def _spec(c):
    """F on the coin; x bound copy by copy (binder chain V), shifted by -1, the coin added twice
    as |c> in a 2^n register (c, then n - 1 zero bits by binder chains R and Q), split back
    (splitter chain U)."""
    N = c.N
    return "\n".join(
        [
            "h = Z(0 -> 1, literal 1)",
            "b = B(literal 2, literal 2^m)",
            f"f = F({c.D})",
            f"dec = X(1 -> 1, literal 2^{N}) phase{{k: k/2^{N}}}",
            f"z = Z(1 -> 3, {c.D})",
            "r = B(literal 2, literal 2^(m + 1))",
            f"zr = X(0 -> 1, {c.D})",
            "s = B(literal 2, literal 2^(m + 1))",
            f"zs = X(0 -> 1, {c.D})",
            f"a1 = X(2 -> 1, literal 2^{N})",
            f"a2 = X(2 -> 1, literal 2^{N})",
            "t = S(literal 2, literal 2^m)",
            "e = Z(1 -> 0, literal 1)",
            "wire h.out0 -- b.in1, b.out0 -- dec.in0, dec.out0 -- a1.in0, a1.out0 -- a2.in0",
            "wire a2.out0 -- t.in0, t.out1 -- e.in0, f.out0 -- z.in0, z.out1 -- r.in1",
            "wire zr.out0 -- r.in0, r.out0 -- a1.in1, z.out2 -- s.in1, zs.out0 -- s.in0",
            "wire s.out0 -- a2.in1",
            f"repeat {N} as V index m: b",
            "chain V: b.out0 -> b.in1",
            f"repeat {N} - 1 as R index m: r, zr",
            "chain R: r.out0 -> r.in1",
            f"repeat {N} - 1 as Q index m: s, zs",
            "chain Q: s.out0 -> s.in1",
            f"repeat {N} as U index m: t",
            "chain U: t.in0 -> t.out1",
            "inputs f.in0, b.in0",
            "outputs z.out0, t.out0",
            "scalar 2",
        ]
    )


SUBJECT = Subject(
    id="coined_quantum_walk",
    name="Coined quantum walk on a cycle",
    category="algorithm",
    source="aharonov1993quantum",
    size_parameter="position bits",
    dimension=2,
    domain=Domain(n_min=1, d_min=2, d_max=2),
    equations=(Equation("main", _circuit, _spec, form="boxed", reference=_reference),),
)
