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

"""Boundary order of a chain box with inputs: the copies' carry ends listed before the chain's
tail give the one-place shift of the unrolled ladder."""

from __future__ import annotations

import pytest

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

_SWAPS = (
    "dim d\ncount n\ng0 = Z(1 -> 1, d)\ng1 = Z(1 -> 1, d)\nrepeat n as C0: g0, g1\n"
    "chain C0: g1.out0 -> g1.in0\ninputs g1.in0, g0.in0\noutputs {outs}"
)
"""A SWAP chain: the head carried through every copy, each copy's register wire straight."""

SHIFT = _SWAPS.format(outs="g0.out0, g1.out0")
"""Copies first, tail last: input i + 1 to output i, input 0 to the last output."""

CARRY_FIRST = _SWAPS.format(outs="g1.out0, g0.out0")
"""Tail first, copies after: every input to its own output."""


def _wires(outs: str) -> str:
    return (
        "dim d\ncount n\nh = Z(1 -> 1, d)\nrepeat n as V: w = Z(1 -> 1, d)\n"
        f"inputs h.in0, w.in0\noutputs {outs}"
    )


CYCLIC = _wires("w.out0, h.out0")
"""The cyclic shift down by one place, boxed."""

IDENTITY = _wires("h.out0, w.out0")
"""The identity on n + 1 wires, boxed."""


def _unrolled_shift(n: int) -> str:
    lines = ["dim d"] + [f"w{i} = Z(1 -> 1, d)" for i in range(n + 1)]
    lines.append("inputs " + ", ".join(f"w{i}.in0" for i in range(n + 1)))
    lines.append("outputs " + ", ".join(f"w{(i + 1) % (n + 1)}.out0" for i in range(n + 1)))
    return "\n".join(lines)


def test_copies_first_is_the_cyclic_shift() -> None:
    assert decide_equal(parse_dsl(SHIFT), parse_dsl(CYCLIC)).verdict is EqualityVerdict.EQUAL
    assert decide_equal(parse_dsl(CARRY_FIRST), parse_dsl(IDENTITY)).verdict in (
        EqualityVerdict.EQUAL,
        EqualityVerdict.EQUAL_UNCERTIFIED,
    )
    assert decide_equal(parse_dsl(SHIFT), parse_dsl(IDENTITY)).verdict is EqualityVerdict.UNEQUAL


@pytest.mark.parametrize("n", [1, 2, 3])
def test_copies_first_matches_the_unrolled_shift(n: int) -> None:
    unrolled = parse_dsl(_unrolled_shift(n))
    assert compare(parse_dsl(SHIFT), unrolled, {"n": n, "d": 2}).matched
    assert not compare(parse_dsl(CARRY_FIRST), unrolled, {"n": n, "d": 2}).matched
