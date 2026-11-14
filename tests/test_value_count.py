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

"""Boxed qram over a free value count N (79.5)."""

from __future__ import annotations

from archytaszx.diagram.bangbox import instantiate_symbol
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.repl.parser import parse_dsl

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
