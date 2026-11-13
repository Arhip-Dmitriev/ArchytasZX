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

"""Tree window with W fusion through the fed box: the boxed W-state splitting tree against the
W node fed |1>."""

from __future__ import annotations

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

CIRCUIT = """count n
g0 = X(0 -> 1, literal 2)
g1 = X(1 -> 1, literal 2) phase{k: -k/2}
g2 = X(0 -> 1, literal 2)
g3 = Z(1 -> 2, literal 2)
g4 = X(2 -> 1, literal 2)
g5 = Z(1 -> 1, literal 2) phase{1: 3/4}
g6 = F(literal 2)
g7 = Z(1 -> 1, literal 2) phase{1: 7/8}
g8 = F(literal 2)
g9 = Z(1 -> 1, literal 2) phase{1: 1/4}
g10 = Z(1 -> 2, literal 2)
g11 = X(2 -> 1, literal 2)
g12 = Z(1 -> 1, literal 2) phase{1: 3/4}
g13 = F(literal 2)
g14 = Z(1 -> 1, literal 2) phase{1: 1/8}
g15 = F(literal 2)
g16 = Z(1 -> 1, literal 2) phase{1: 1/4}
g17 = Z(1 -> 2, literal 2)
g18 = X(2 -> 1, literal 2)
wire g0.out0 -- g1.in0, g1.out0 -- g3.in0, g2.out0 -- g4.in0, g3.out1 -- g4.in1
wire g4.out0 -- g5.in0, g5.out0 -- g6.in0, g6.out0 -- g7.in0, g7.out0 -- g8.in0
wire g8.out0 -- g9.in0, g3.out0 -- g10.in0, g9.out0 -- g11.in0, g10.out1 -- g11.in1
wire g11.out0 -- g12.in0, g12.out0 -- g13.in0, g13.out0 -- g14.in0, g14.out0 -- g15.in0
wire g15.out0 -- g16.in0, g16.out0 -- g17.in0, g10.out0 -- g18.in0, g17.out1 -- g18.in1
repeat n as C0: g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18
tree C0: g3.in0 <- g18.out0, g17.out0
copy scalar C0: 2
outputs g18.out0, g17.out0
scalar sqrt(2)/2"""
"""w_state's translated circuit: |1>, then a depth-n tree whose copy splits by CRy(pi/2), CX."""

SPEC = """count n
x = X(0 -> 1, literal 2) phase{1: 1/2}
w = W(1 -> 1, literal 2)
wire x.out0 -- w.in0
repeat 2^n: w.out0
scalar 2^(-(n + 1)/2)"""
"""w_state's spec: W(1 -> 2^n) on |1>, normalised."""


def _kinds(t: kernel.Theorem) -> set[str]:
    """Every inference kind in ``t``'s proof tree."""
    out = {t.inference.kind}
    for premise in t.inference.premises:
        out |= _kinds(premise)
    return out


def test_leaf_copies_fuse_into_the_w_leg() -> None:
    """Each leaf copy is W(1 -> 2) after a weight on |1>, fused into the W node's boxed leg;
    a phase in the copy or a changed copy scalar is not proved."""
    decision = decide_equal(parse_dsl(CIRCUIT), parse_dsl(SPEC))
    assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
    assert "step case proved by tree_window" in decision.reason, decision.reason
    assert isinstance(decision.induction_proof, kernel.Theorem)
    assert _kinds(decision.induction_proof) >= {"induction", "w_fusion"}
    broken = (
        CIRCUIT.replace(
            "g14 = Z(1 -> 1, literal 2) phase{1: 1/8}", "g14 = Z(1 -> 1, literal 2) phase{1: 3/8}"
        ),
        CIRCUIT.replace("copy scalar C0: 2", "copy scalar C0: 3"),
    )
    for circuit in broken:
        verdict = decide_equal(parse_dsl(circuit), parse_dsl(SPEC), max_samples=0).verdict
        assert verdict not in (EqualityVerdict.EQUAL, EqualityVerdict.EQUAL_UNCERTIFIED)
