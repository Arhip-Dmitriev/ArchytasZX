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

"""Chains whose link dimensions read the copy index: a binder chain B(2, 2^m) and a splitter
chain S(2, 2^m) around the shift by one on 2^n values."""

from __future__ import annotations

import pytest

from archytaszx.repl.parser import ParseDomainError, parse_dsl
from archytaszx.semantics.check import instantiate
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

SHIFT = """count n
h = Z(0 -> 1, literal 1)
b = B(literal 2, literal 2^m)
s = X(1 -> 1, literal 2^n) phase{k: -k/2^n}
t = S(literal 2, literal 2^m)
e = Z(1 -> 0, literal 1)
wire h.out0 -- b.in1, b.out0 -- s.in0, s.out0 -- t.in0, t.out1 -- e.in0
repeat n as V index m: b
chain V: b.out0 -> b.in1
repeat n as U index m: t
chain U: t.in0 -> t.out1
inputs b.in0
outputs t.out0"""
"""The shift by one on n qubits, copy m the bit of value 2^m."""

UNROLLED = """b = B(literal 2, literal 2)
s = X(1 -> 1, literal 4) phase{k: -k/4}
t = S(literal 2, literal 2)
wire b.out0 -- s.in0, s.out0 -- t.in0
inputs b.in1, b.in0
outputs t.out1, t.out0"""
"""The shift by one on 2 qubits, low bit first."""


def test_a_binder_chain_reads_its_copy_index_in_dimensions() -> None:
    """At n = 2 the chains instantiate to the two-qubit shift; a link whose next copy does not
    match is refused."""
    at_two = instantiate(parse_dsl(SHIFT), {"n": 2})
    assert decide_equal(at_two, parse_dsl(UNROLLED)).verdict is EqualityVerdict.EQUAL
    with pytest.raises(ParseDomainError):
        parse_dsl(SHIFT.replace("b = B(literal 2, literal 2^m)", "b = B(literal 2, literal 3^m)"))
