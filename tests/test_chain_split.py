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

"""kernel.chain_split: a chain box over independent chains split into aligned chain boxes,
instantiation-checked at k = 0..4, tampered splits refused."""

from __future__ import annotations

import pytest

from archytaszx.diagram.bangbox import Mult, chain_splitted, instantiate_symbol
from archytaszx.diagram.graph import BangBoxId, Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import check, kernel

ENDS = """
ha = Z(0 -> 1, literal 2)
ea = Z(1 -> 0, literal 2)
hb = X(0 -> 1, literal 2)
eb = X(1 -> 0, literal 2)
wire ha.out0 -- a.in0, a.out0 -- ea.in0, hb.out0 -- b.in0, b.out0 -- eb.in0
repeat n as V index j: a, b
chain V: a.out0 -> a.in0, b.out0 -> b.in0
"""

TWO_CHAINS = (
    """count n
a = Z(2 -> 1, literal 2) phase{1: (j+1)/8}
b = X(1 -> 2, literal 2)
"""
    + ENDS
    + """inputs a.in1
outputs b.out1
"""
)

COPYWISE = (
    """count n
a = Z(2 -> 2, literal 2) phase{1: (j+1)/8}
b = X(2 -> 2, literal 2)
wire a.out1 -- b.in1
"""
    + ENDS
    + """inputs a.in1
outputs b.out1
"""
)

INTERLEAVED = (
    """count n
a = Z(2 -> 1, literal 2) phase{1: (j+1)/8}
b = X(2 -> 1, literal 2)
"""
    + ENDS
    + """inputs a.in1, b.in1
"""
)


def _split(d: Diagram) -> tuple[BangBoxId, Diagram]:
    (box_id,) = d.bang_boxes
    scope = d.bang_boxes[box_id].node_scope
    parts = [frozenset(n for n in scope if d.nodes[n].generator_type.name == kind) for kind in "ZX"]
    return box_id, chain_splitted(d, box_id, parts)


@pytest.mark.parametrize("text", [TWO_CHAINS, COPYWISE])
def test_chain_split_commutes_with_instantiation(text: str) -> None:
    d = parse_dsl(text)
    _, split = _split(d)
    theorem = kernel.chain_split(d, split)
    assert theorem.inference.kind == "chain_split"
    assert len(split.bang_boxes) == 2
    for k in range(5):
        a, b = (instantiate_symbol(x, "n", k) for x in (d, split))
        assert check.compare(a, b, {}).matched, k


def test_tampered_splits_are_refused() -> None:
    d = parse_dsl(COPYWISE)
    box_id, split = _split(d)
    member = next(b for b, box in split.bang_boxes.items() if box.align is not None)
    for change in ("phase", "count", "index", "unlinked", "link_moved", "overlap"):
        tampered = split.copy()
        if change == "phase":
            a = next(n for n, node in tampered.nodes.items() if node.phase is not None)
            tampered.set_phase(a, None)
        elif change == "count":
            tampered.set_bang_box_multiplicity(member, Mult("n") + 1)
        elif change == "index":
            tampered.set_bang_box_extension(member, index="i")
        elif change == "unlinked":
            tampered.set_bang_box_extension(member, links=())
        elif change == "link_moved":
            links = tampered.bang_boxes[member].links
            tampered.set_bang_box_extension(member, links=())
            tampered.set_bang_box_extension(box_id, links=tampered.bang_boxes[box_id].links + links)
        else:
            tampered.add_bang_box(Mult(2), node_scope=tampered.bang_boxes[member].node_scope)
        with pytest.raises(kernel.KernelError):
            kernel.chain_split(d, tampered)
    interleaved = parse_dsl(INTERLEAVED)
    with pytest.raises(kernel.KernelError):
        kernel.chain_split(interleaved, _split(interleaved)[1])
