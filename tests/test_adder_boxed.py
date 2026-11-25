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

"""fca_boxed and draper_boxed equal to their specs by induction with a kernel proof, at d = 2 and
symbolic d; sign-flipped pair phases not proved EQUAL."""

from __future__ import annotations

import pytest

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.decide import decide_equal

FCA = """
count n
g0 = Z(2 -> 1, d)
g1 = F(d)
g2 = Z(1 -> 2, d)
g3 = Zp(1 -> 1, d) phase{x, y: d^(j - m - 1)*x*y}
g4 = Z(1 -> 1, d) phase{k: 2*d^(j - n)*k}
g5 = Z(1 -> 2, d)
g6 = F(d)
g7 = F(d)
g8 = F(d)
g9 = Z(2 -> 1, d)
g10 = Zp(1 -> 1, d) phase{x, y: -d^(j - m - 1)*x*y}
wire g0.out0 -- g1.in0, g1.out0 -- g2.in0, g2.out1 -- g3.in0, g3.out0 -- g0.in1
wire g2.out0 -- g4.in0, g4.out0 -- g5.in0, g5.out0 -- g6.in0, g6.out0 -- g7.in0
wire g7.out0 -- g8.in0, g8.out0 -- g9.in0, g5.out1 -- g10.in0, g10.out0 -- g9.in1
repeat n as C0 index j: g0, g1, g2, g4, g5, g6, g7, g8, g9
pairs P0 of C0 index m: g3
left P0: g2.out1
right P0: g0.in1
pairs P3 of C0 index m: g10
left P3: g5.out1
right P3: g9.in1
inputs g0.in0
outputs g9.out0
"""

FCA_SPEC = """
count n
h = Z(0 -> 1, literal 1)
b = B(d^j, d)
s = X(1 -> 1, d^n) phase{k: -2*k/d^n}
t = S(d^j, d)
e = Z(1 -> 0, literal 1)
wire h.out0 -- b.in0, b.out0 -- s.in0, s.out0 -- t.in0, t.out0 -- e.in0
repeat n as V index j: b
chain V: b.out0 -> b.in0
repeat n as U index j: t
chain U: t.in0 -> t.out0
inputs b.in1
outputs t.out1
"""

DRAPER = """
count n
g0 = Z(2 -> 1, d)
g1 = F(d)
g2 = Z(1 -> 2, d)
g3 = Zp(1 -> 1, d) phase{x, y: d^(j - m - 1)*x*y}
g4 = Z(1 -> 2, d)
g5 = Z(2 -> 1, d)
g6 = F(d)
g7 = Z(2 -> 1, d)
g8 = Z(1 -> 2, d)
g9 = Zp(1 -> 1, d) phase{x, y: d^(j - m - 1)*x*y}
g10 = Z(1 -> 2, d)
g11 = F(d)
g12 = F(d)
g13 = F(d)
g14 = Z(2 -> 1, d)
g15 = Zp(1 -> 1, d) phase{x, y: -d^(j - m - 1)*x*y}
wire g0.out0 -- g1.in0, g1.out0 -- g2.in0, g2.out1 -- g3.in0, g3.out0 -- g0.in1
wire g2.out0 -- g5.in0, g4.out1 -- g6.in0, g6.out0 -- g5.in1, g4.out0 -- g7.in0
wire g5.out0 -- g8.in0, g8.out1 -- g9.in0, g9.out0 -- g7.in1, g8.out0 -- g10.in0
wire g10.out0 -- g11.in0, g11.out0 -- g12.in0, g12.out0 -- g13.in0, g13.out0 -- g14.in0
wire g10.out1 -- g15.in0, g15.out0 -- g14.in1
repeat n as C0 index j: g0, g1, g2, g4, g5, g6, g7, g8, g10, g11, g12, g13, g14
pairs P0 of C0 index m: g3
left P0: g2.out1
right P0: g0.in1
pairs P3 of C0 index m: g9
left P3: g8.out1
right P3: g7.in1
pairs P6 of C0 index m: g15
left P6: g10.out1
right P6: g14.in1
inputs g4.in0, g0.in0
outputs g7.out0, g14.out0
scalar d^(n/2)
"""

DRAPER_SPEC = """
count n
ha = Z(0 -> 1, literal 1)
ba = B(d^j, d)
ta = S(d^j, d)
ea = Z(1 -> 0, literal 1)
wire ha.out0 -- ba.in0, ta.out0 -- ea.in0
hb = Z(0 -> 1, literal 1)
bb = B(d^j, d)
tb = S(d^j, d)
eb = Z(1 -> 0, literal 1)
wire hb.out0 -- bb.in0, tb.out0 -- eb.in0
z = Z(1 -> 2, d^n)
x = X(2 -> 1, d^n)
wire ba.out0 -- z.in0, bb.out0 -- x.in0, z.out1 -- x.in1
wire z.out0 -- ta.in0, x.out0 -- tb.in0
repeat n as V index j: ba, bb
chain V: ba.out0 -> ba.in0, bb.out0 -> bb.in0
repeat n as U index j: ta, tb
chain U: ta.in0 -> ta.out0, tb.in0 -> tb.out0
inputs ba.in1, bb.in1
outputs ta.out1, tb.out1
scalar d^(n/2)
"""

DIMS = ["dim d = 2", "dim d"]


def _pair(boxed: str, spec: str, dim: str) -> tuple[str, str]:
    return dim + boxed, dim + spec


def _flipped(boxed: str) -> str:
    flipped = boxed.replace("phase{x, y: d^(j - m - 1)*x*y}", "phase{x, y: -d^(j - m - 1)*x*y}", 1)
    assert flipped != boxed
    return flipped


@pytest.mark.slow
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("family", ["fca", "draper"])
def test_boxed_equals_spec_by_induction_with_a_kernel_proof(family: str, dim: str) -> None:
    boxed, spec = (FCA, FCA_SPEC) if family == "fca" else (DRAPER, DRAPER_SPEC)
    left, right = (parse_dsl(t) for t in _pair(boxed, spec, dim))
    decision = decide_equal(left, right, samples=[])
    assert decision.verdict.name == "EQUAL", decision.reason
    assert decision.method.name == "INDUCTION" and decision.induction_proof is not None
    theorem = kernel.from_decision(decision, left, right)
    assert theorem.assumptions == ()
    assert kernel._same(theorem.lhs, left) and kernel._same(theorem.rhs, right)


@pytest.mark.slow
@pytest.mark.parametrize("family", ["fca", "draper"])
def test_sign_flipped_pair_phase_is_not_equal(family: str) -> None:
    boxed, spec = (FCA, FCA_SPEC) if family == "fca" else (DRAPER, DRAPER_SPEC)
    left, right = (parse_dsl(t) for t in _pair(_flipped(boxed), spec, DIMS[0]))
    decision = decide_equal(left, right, samples=[])
    assert decision.verdict.name != "EQUAL"
    assert decision.induction_proof is None


if __name__ == "__main__":
    import time

    for family in ("fca", "draper"):
        for dim in DIMS:
            t0 = time.time()
            test_boxed_equals_spec_by_induction_with_a_kernel_proof(family, dim)
            print(family, dim, "ok", f"{time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        test_sign_flipped_pair_phase_is_not_equal(family)
        print(family, "flipped ok", f"{time.time() - t0:.1f}s", flush=True)
