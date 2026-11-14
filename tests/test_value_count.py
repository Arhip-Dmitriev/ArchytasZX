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

"""Boxed qram over a free value count N (79.5), aligned boxes in a cut (79.8), the spec
step exposed by inserted identities (79.10), the per-copy lemma (79.7) and qram EQUAL over
the value count."""

from __future__ import annotations

import pytest

from archytaszx.diagram.bangbox import instantiate_symbol
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import induction, kernel, lemmas
from archytaszx.semantics.decide import EqualityVerdict, decide_equal
from archytaszx.semantics.valuecount import generalise, specialised

NESTED = """
count n
a = Z(1 -> 2, literal 2)
x = X(1 -> 1, literal 2) phase{1: 1/2}
g = X(1 -> 0, literal 2)
wire a.out1 -- x.in0, x.out0 -- g.in0
repeat n as A: a
repeat N as P index v: x, g
repeat n as C index j in P: x
align C with A
inputs a.in0
outputs a.out0
"""


def test_a_parent_box_at_zero_keeps_the_partner_of_its_aligned_child() -> None:
    zero = instantiate_symbol(parse_dsl(NESTED), "N", 0)
    assert [str(box.multiplicity) for box in zero.bang_boxes.values()] == ["n"]
    (spider,) = zero.nodes.values()
    assert spider.generator_type is Z_SPIDER
    assert (len(spider.inputs), len(spider.outputs)) == (1, 1)


FANNED = """
count n
a = Z(1 -> 2, literal 2) phase{1: 1/4}
e = Z(1 -> 0, literal 2)
wire a.out1 -- e.in0
repeat 2^n as W: e
inputs a.in0
outputs a.out0
"""

PLAIN = """
count n
a = Z(1 -> 1, literal 2) phase{1: 1/4}
inputs a.in0
outputs a.out0
"""


def test_a_theorem_over_the_value_count_specialises_to_the_power() -> None:
    left, right = parse_dsl(FANNED), parse_dsl(PLAIN)
    found = generalise(left, right)
    assert found is not None and (found.symbol, str(found.power)) == ("N", "2**n")
    family = kernel.contraction(found.left, found.right)
    theorem = specialised(family, found, left, right)
    assert theorem.lhs is left and theorem.rhs is right and theorem.variables == ("n",)
    assert "(fam" in kernel.lean(theorem, semantic=False)
    with pytest.raises(kernel.KernelError):
        kernel.instance(family, "n", found.power)


def test_the_hypothesis_is_cut_out_through_an_aligned_partner_and_put_back_aligned() -> None:
    nested = parse_dsl(NESTED)
    ob = induction.build_obligation(nested, nested, index="N", base=0, base_proved=True)
    successor, hypothesis = ob.left_at_successor, ob.left_at_k
    peeled = induction._peel_step_index(successor, ob.step_symbol)
    assert peeled is not None
    candidates = induction.copy_candidates(peeled[0], frozenset(successor.nodes), peeled[1])
    found = induction.expose_cut(peeled[0], candidates, hypothesis)
    assert found is not None
    exposed, cut, steps = found
    assert [step.kind for step in steps] == ["split"]
    (leaf,) = steps[0].copy_after - steps[0].copy_before
    (home,) = [box for box in exposed.bang_boxes.values() if leaf in box.node_scope]
    assert home.parent is None and home.align is not None and leaf not in cut.region
    replaced = induction.replace_cut_region(exposed, cut, hypothesis, hypothesis)
    moved = replaced.bang_boxes[home.id]
    assert moved.align is not None and moved.align not in exposed.bang_boxes
    assert isomorphic(replaced, exposed)


QRAM_SPEC = """
count n
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
z = Z(1 -> 2, literal 2^n)
x = X(1 -> 1, literal 2^n) phase{k: v*k/2^n}
c = Z(1 -> 2, literal 2)
p = B(literal 2^n, literal 2)
q = B(literal 2^(n + 1), literal 2)
g = Z(1 -> 0, literal 2^(n + 2)) phase{3: 1/2}
f0 = F(literal 2)
y = Z(1 -> 2, literal 2)
f1 = F(literal 2)
f2 = F(literal 2)
f3 = F(literal 2)
s = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
wire h.out0 -- b.in1, b.out0 -- z.in0, z.out0 -- s.in0, s.out1 -- e.in0
wire z.out1 -- x.in0, x.out0 -- p.in0, c.out1 -- p.in1, p.out0 -- q.in0
wire y.out1 -- q.in1, q.out0 -- g.in0
wire f0.out0 -- y.in0, y.out0 -- f1.in0, f1.out0 -- f2.in0, f2.out0 -- f3.in0
repeat n as V index m: b
chain V: b.out0 -> b.in1
repeat N as W index v: x, c, p, q, g
repeat n as U index m: s
chain U: s.in0 -> s.out1
inputs b.in0, c.in0, f0.in0
outputs s.out0, c.out0, f3.out0
"""


def test_the_qram_spec_step_over_the_value_count_is_exposed_by_inserted_identities() -> None:
    spec = parse_dsl(QRAM_SPEC)
    ob = induction.build_obligation(spec, spec, index="N", base=0, base_proved=True)
    successor, hypothesis = ob.right_at_successor, ob.right_at_k
    peeled = induction._peel_step_index(successor, ob.step_symbol)
    assert peeled is not None
    candidates = induction.copy_candidates(peeled[0], frozenset(successor.nodes), peeled[1])
    found = lemmas.expose_by_insertion(peeled[0], candidates, hypothesis)
    assert found is not None
    theorem, cut, _copy = found
    assert theorem.lhs is peeled[0] and not theorem.domain
    extracted = induction._cut_extract(theorem.rhs, cut)
    assert isomorphic(induction._bare(extracted), induction._bare(hypothesis))


def test_the_qram_per_copy_lemma_is_a_kernel_induction_over_its_bit_count() -> None:
    theorem = lemmas.proved(lemmas.QRAM_COPY)
    assert isinstance(theorem, kernel.Theorem)
    assert theorem.inference.kind == "induction" and not theorem.domain
    assert theorem.statement() == "for all p"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("phase{3: 1/2}", "phase{2: 1/2}"),
        ("g6 = X(1 -> 1, literal 2) phase{1: 1/2}", "g6 = X(1 -> 1, literal 2)"),
    ],
)
def test_a_broken_per_copy_lemma_is_not_proved(old: str, new: str) -> None:
    left, right = (t.replace(old, new) for t in (lemmas._COPY_LEFT, lemmas._COPY_RIGHT))
    broken = lemmas.Lemma(
        "broken_copy",
        "p",
        "broken",
        lambda p: lemmas._dsl_sides(left, right, p),
        prove=lemmas.induct_by_insertion,
    )
    with pytest.raises((lemmas.LemmaError, kernel.KernelError)):
        lemmas.proved(broken)


QRAM_CIRCUIT = """
count n
g0 = Z(1 -> 2, literal 2)
g1 = Z(1 -> 2, literal 2)
g2 = X(0 -> 1, literal 2^n) phase{k: -v*k/2^n}
g3 = S(literal 2, literal 2^j)
g4 = Z(1 -> 0, literal 1)
g5 = X(2 -> 1, literal 2)
g6 = X(1 -> 1, literal 2) phase{1: 1/2}
g7 = T(literal 2)
g8 = Z(2 -> 1, literal 2)
g9 = T(literal 2)
g10 = Ti(literal 2)
g11 = X(2 -> 1, literal 2)
wire g2.out0 -- g3.in0, g3.out1 -- g4.in0, g0.out1 -- g5.in0, g3.out0 -- g5.in1
wire g5.out0 -- g6.in0, g6.out0 -- g7.in0, g7.out0 -- g8.in1, g1.out1 -- g9.in0
wire g9.out0 -- g8.in0, g8.out0 -- g10.in0, g10.out0 -- g11.in1
repeat n as C0: g0
repeat 2^n as C1 index v: g1, g2, g3, g4, g5, g6, g7, g8, g9, g10
repeat n as C2 index j in C1: g3, g5, g6, g7
chain C2: g3.in0 -> g3.out1
align C2 with C0
inputs g0.in0, g1.in0, g11.in0
outputs g0.out0, g1.out0, g11.out0
scalar 2^(2^n*(n + 1)/2)*(2^(-n/2))^(2^n)
"""


def test_qram_is_equal_over_its_value_count_with_a_kernel_proof() -> None:
    circuit = parse_dsl(QRAM_CIRCUIT)
    spec = parse_dsl(QRAM_SPEC.replace("repeat N as W", "repeat 2^n as W"))
    decision = decide_equal(circuit, spec, samples=[])
    assert decision.verdict is EqualityVerdict.EQUAL
    theorem = kernel.from_decision(decision, circuit, spec)
    assert theorem.statement() == "for all n" and not theorem.domain


def test_a_shifted_qram_spec_is_not_proved() -> None:
    circuit = parse_dsl(QRAM_CIRCUIT)
    shifted = QRAM_SPEC.replace("repeat N as W", "repeat 2^n as W").replace(
        "phase{k: v*k/2^n}", "phase{k: (v + 1)*k/2^n}"
    )
    decision = decide_equal(circuit, parse_dsl(shifted), samples=[])
    assert decision.verdict is not EqualityVerdict.EQUAL
