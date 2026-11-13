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

"""Copy-aligned box-to-box wiring (align): instantiation in both orders against unrolled
diagrams, refusals."""

from __future__ import annotations

from fractions import Fraction

import pytest

from archytaszx.diagram.bangbox import BangBoxGrammarError, instantiate_symbol, peel_one
from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import ParseDomainError, parse_dsl
from archytaszx.semantics import check

BOXED = """
count n, l
x = Z(1 -> 2, literal 2)
y = Z(1 -> 1, literal 2) phase{1: (v+1)*(j+1)/16}
w = Z(0 -> 0, literal 2) phase{1: 1/4}
wire x.out1 -- y.in0
repeat n as A: x
repeat l as V index v: w, y
repeat n as J index j in V: y
ALIGN
inputs x.in0
outputs x.out0, y.out0
"""


def _unrolled(n: int, l: int) -> Diagram:
    """Copy j of x wired to copy j of y in every copy v."""
    lines = [f"x{j} = Z(1 -> {1 + l}, literal 2)" for j in range(n)]
    for v in range(l):
        lines.append(f"w{v} = Z(0 -> 0, literal 2) phase{{1: 1/4}}")
        for j in range(n):
            phase = Fraction((v + 1) * (j + 1), 16)
            lines.append(f"y{v}_{j} = Z(1 -> 1, literal 2) phase{{1: {phase}}}")
            lines.append(f"wire x{j}.out{1 + v} -- y{v}_{j}.in0")
    lines.append("inputs " + ", ".join(f"x{j}.in0" for j in range(n)))
    outs = [f"x{j}.out0" for j in range(n)] + [f"y{v}_{j}.out0" for v in range(l) for j in range(n)]
    lines.append("outputs " + ", ".join(outs))
    return parse_dsl("\n".join(lines))


def _at(d: Diagram, first: str, n: int, l: int) -> Diagram:
    values = {"n": n, "l": l}
    second = "l" if first == "n" else "n"
    return instantiate_symbol(instantiate_symbol(d, first, values[first]), second, values[second])


@pytest.mark.parametrize("first", ["n", "l"])
@pytest.mark.parametrize(("n", "l"), [(1, 2), (2, 1), (2, 2), (3, 2)])
def test_aligned_equals_unrolled(first: str, n: int, l: int) -> None:
    d = parse_dsl(BOXED.replace("ALIGN", "align J with A"))
    assert check.compare(_at(d, first, n, l), _unrolled(n, l), {}).matched


def test_unaligned_differs() -> None:
    d = parse_dsl(BOXED.replace("ALIGN", ""))
    assert not check.compare(_at(d, "n", 2, 2), _unrolled(2, 2), {}).matched


def test_alignment_refusals() -> None:
    d = parse_dsl(BOXED.replace("ALIGN", "align J with A"))
    with pytest.raises(BangBoxGrammarError):
        peel_one(d, next(b for b, box in d.bang_boxes.items() if box.align is not None))
    other = BOXED.replace("ALIGN", "align J with A").replace("repeat n as A", "repeat l as A")
    with pytest.raises(ParseDomainError):
        parse_dsl(other)
