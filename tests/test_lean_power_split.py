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

"""Kernel power splits exported over linked diagrams (REMAINING_BUGS_SPEC 79.6, 79.6d, 79.6g): an
indexed box with a phase reading its index and a nested chain box split through
``evalL_split``, the reverse refused, top-level spiders with legs bundled over the box split into
two legs, base 3 as two chained splits, qram's box C1 outside the form, the semantic export with
no ``power_split`` axiom, and Lean accepting the proofs and rejecting a wrong copy offset or a
wrong tail leg."""

from __future__ import annotations

import os

import pytest

from archytaszx.diagram.bangbox import peel_owners, split_power
from archytaszx.diagram.graph import BangBoxId, Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel, lean_split
from archytaszx.semantics.check import compare
from archytaszx.semantics.lean_net import check, lean_project
from tests.test_power_split import C1, _source

ROW = """
count n
a = Z(1 -> 1, literal 2) phase{1: v/2^n}
b = X(1 -> 1, literal 2)
c = Z(1 -> 1, literal 2)
e = X(1 -> 1, literal 2) phase{1: 1/4}
wire a.out0 -- b.in0, b.out0 -- c.in0, c.out0 -- e.in0
repeat 2^(n+1) as P index v: a, b, c, e
repeat n as Q in P: c
chain Q: c.in0 -> c.out0
inputs a.in0
outputs e.out0
"""
"""A box of 2^(n+1) copies whose first node reads the copy index, a chain box inside."""

TAIL = """
count n
a = Z(1 -> 2, literal 2)
b = X(1 -> 1, literal 2) phase{1: v/2^n}
c = X(2 -> 1, literal 2)
wire a.out1 -- b.in0, b.out0 -- c.in1
repeat 2^(n+1) as P index v: b
inputs a.in0, c.in0
outputs a.out0, c.out0
"""
"""A box of 2^(n+1) copies wired to a top-level Z and a top-level X by bundled legs."""


def _split(source: str = ROW, base: int = 2) -> tuple[Diagram, Diagram]:
    d = parse_dsl(source.replace("2^(n+1)", f"{base}^(n+1)").replace("/2^n", f"/{base}^n"))
    return d, split_power(d, BangBoxId(0))[0]


def test_a_power_split_exports_through_evalL_split() -> None:
    d, s = _split()
    for value in (0, 1):
        assert compare(s, d, {"n": value}).matched
    source = lean_split.export_split(d, s)
    assert source is not None and "evalL_split ok1" in source
    assert "cph := [(0, 0, 0, 2), (4, 3, 1, 2)]" in source
    assert lean_split.export_split(s, d) is None
    lean = kernel.lean(kernel.power_split(d, s), "main")
    assert "namespace ArchytasZX.Export.Q" in lean and "axiom power_split" not in lean


def test_bundled_legs_split_into_two_legs() -> None:
    d, s = _split(TAIL)
    for value in (0, 1):
        assert compare(s, d, {"n": value}).matched
    source = lean_split.export_split(d, s)
    assert source is not None
    assert "def T1 : List (Port × Port) := [(⟨0, true, 1⟩, ⟨0, true, 2⟩), " in source
    assert "(⟨2, false, 1⟩, ⟨2, false, 2⟩)]" in source
    lean = kernel.lean(kernel.power_split(d, s), "main")
    assert "axiom power_split" not in lean


@pytest.mark.parametrize("text", [ROW, TAIL])
def test_a_base_3_split_is_two_chained_splits(text: str) -> None:
    d, s = _split(text, 3)
    for value in (0, 1):
        assert compare(s, d, {"n": value}).matched
    source = lean_split.export_split(d, s)
    assert source is not None
    assert "theorem ok2 :" in source and "(evalL_split ok2 E ks h2 _).trans" in source
    assert "def M1 : LDiagram" in source and "theorem ok3" not in source


def test_the_qram_split_is_outside_the_linked_form() -> None:
    _circuit, source = _source()
    peeled = peel_owners(source, "k", partial=True).diagram
    split, _boxes = split_power(peeled, C1)
    assert lean_split.export_split(peeled, split) is None
    assert "axiom power_split" in kernel.lean(kernel.power_split(peeled, split), "main")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_split_and_refuses_a_wrong_copy_offset(tmp_path: object) -> None:
    d, s = _split()
    source = lean_split.export_split(d, s)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/split.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("(4, 3, 1, 2)", "(4, 3, 2, 2)")
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_tails_and_base_3_and_refuses_a_wrong_tail(tmp_path: object) -> None:
    d, s = _split(TAIL, 3)
    source = lean_split.export_split(d, s)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/tail3.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("(⟨0, true, 1⟩, ⟨0, true, 2⟩)", "(⟨0, true, 1⟩, ⟨0, true, 0⟩)", 1)
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
