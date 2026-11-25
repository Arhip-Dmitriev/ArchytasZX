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

"""The telescoping carry lemma U decided for every n and d by induction at the chain's last
copy (docs/manual, kernel chapter)."""

from __future__ import annotations

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

U_LHS = """dim d
count n
ex = X(0 -> 1, literal 3); bx = B(literal 3, d)
ec = X(0 -> 1, d); bc = B(d, literal 3)
ad = X(2 -> 1, 3*d); sp = S(literal 3, d)
zi = Z(2 -> 1, d); zo = Z(2 -> 1, d)
pa = Zp(1 -> 1, d) phase{x, y: x*y/d^(j+2)}
po = Zp(1 -> 1, d) phase{x, y: -x*y/d^(j+2)}
wire zi.out0 -- bx.in1, sp.out1 -- zo.in0
wire ex.out0 -- bx.in0, ec.out0 -- bc.in0, bx.out0 -- ad.in0, bc.out0 -- ad.in1, ad.out0 -- sp.in0
zy = Z(1 -> 3, d)
wire zy.out1 -- pa.in0, pa.out0 -- zi.in1, zy.out2 -- po.in0, po.out0 -- zo.in1
repeat n as K index j: ex, bx, ec, bc, ad, sp, zi, zo, pa, po
chain K: bc.in1 -> sp.out0
top = Z(1 -> 0, literal 3)
wire sp.out0 -- top.in0
inputs zy.in0, zi.in0, bc.in1
outputs zy.out0, zo.out0
copy scalar K: (3*d)^(1/2)*3^(-1/2)*d^(-1/2)"""
"""Copies j = 0 .. n - 1 of a carry adder, y fanned to pair phases on x_j and s_j."""

U_RHS = """dim d
count n
ex = X(0 -> 1, literal 3); bx = B(literal 3, d)
ec = X(0 -> 1, d); bc = B(d, literal 3)
ad = X(2 -> 1, 3*d); sp = S(literal 3, d)
wire ex.out0 -- bx.in0, ec.out0 -- bc.in0, bx.out0 -- ad.in0, bc.out0 -- ad.in1, ad.out0 -- sp.in0
repeat n as K index j: ex, bx, ec, bc, ad, sp
chain K: bc.in1 -> sp.out0
top = Zp(in: [literal 3, d], out: []) phase{x, y: x*y/d}
zy = Z(1 -> 3, d)
zc = Z(1 -> 2, literal 3)
ez = Zp(in: [literal 3, d], out: []) phase{x, y: -x*y/d^(n+1)}
wire zc.out0 -- bc.in1, zc.out1 -- ez.in0, zy.out2 -- ez.in1, sp.out0 -- top.in0, zy.out1 -- top.in1
inputs zy.in0, bx.in1, zc.in0
outputs zy.out0, sp.out1
copy scalar K: (3*d)^(1/2)*3^(-1/2)*d^(-1/2)"""
"""The adder chain with the end terms Zp(c_out, y) and Zp(c_in, y)."""


def test_carry_lemma_every_n_and_d() -> None:
    """U at symbolic n and d: equal by induction with a kernel proof."""
    decision = decide_equal(parse_dsl(U_LHS), parse_dsl(U_RHS))
    assert decision.verdict is EqualityVerdict.EQUAL
    assert "chain_window" in decision.reason
    assert "kernel proof built" in decision.reason
