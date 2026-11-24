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

"""Parametric induction: the hypothesis taken at a substituted phase parameter, and a phase
constant generalised to a parameter (PROOFS.txt 4.4)."""

from __future__ import annotations

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

CHAIN = """dim d = 2
count n
a0 = Z(1 -> 1, d)
g = Z(1 -> 1, d) phase{{k: {c}*2^(j - n)*k}}
wire a0.out0 -- g.in0
repeat n as C index j: g
chain C: g.out0 -> g.in0
inputs a0.in0
outputs g.out0"""

SUM = """dim d = 2
count n
g = Z(1 -> 1, d) phase{{k: {c}*(1 - 2^(-n))*k}}
inputs g.in0
outputs g.out0"""


def _kinds(theorem: kernel.Theorem) -> set[str]:
    out = {theorem.inference.kind}
    for premise in theorem.inference.premises:
        out |= _kinds(premise)
    return out


def test_parameter_hypothesis_at_half() -> None:
    decision = decide_equal(parse_dsl(CHAIN.format(c="a")), parse_dsl(SUM.format(c="a")))
    assert decision.verdict is EqualityVerdict.EQUAL
    assert decision.induction_proof is not None
    assert "induction_param" in _kinds(decision.induction_proof)


def test_constant_generalised() -> None:
    decision = decide_equal(parse_dsl(CHAIN.format(c="3")), parse_dsl(SUM.format(c="3")))
    assert decision.verdict is EqualityVerdict.EQUAL
    assert decision.induction_proof is not None
    assert {"induction_param", "phase_instance"} <= _kinds(decision.induction_proof)


def test_wrong_parameter_sum_refuted() -> None:
    wrong = SUM.format(c="2*a")
    decision = decide_equal(parse_dsl(CHAIN.format(c="a")), parse_dsl(wrong))
    assert decision.verdict is EqualityVerdict.UNEQUAL


def test_phase_instance_refuses_copy_index() -> None:
    decision = decide_equal(parse_dsl(CHAIN.format(c="a")), parse_dsl(SUM.format(c="a")))
    assert decision.induction_proof is not None
    try:
        kernel.phase_instance(decision.induction_proof, "a", "j")
    except kernel.KernelError:
        return
    raise AssertionError("a copy index was taken as a parameter value")
