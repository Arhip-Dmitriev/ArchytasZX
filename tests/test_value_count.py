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

"""Boxed qram over a free value count N (79.5) and aligned boxes in a cut (79.8)."""

from __future__ import annotations

import pytest

from archytaszx.diagram.bangbox import instantiate_symbol
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import induction, kernel
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
