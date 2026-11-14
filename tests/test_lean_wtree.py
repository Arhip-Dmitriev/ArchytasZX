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

"""The w_state tree families: the circuit copy against the weighted W fan copy through the atom
form (psup, aphase, aflip), exported by export_tree_theorem with one coefficient vector A."""

from __future__ import annotations

import os

import pytest

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_chain import export_tree_theorem

from .test_tree_w_fusion import CIRCUIT, SPEC

COPY_WEIGHT = "(((1 : ℂ) / 2) * (rt 2)) * ex (Φ 9 v)"
"""The fan copy's weighted Z coefficient sqrt(2) / 2 on |1>."""


def _source() -> str:
    decision = decide_equal(parse_dsl(CIRCUIT), parse_dsl(SPEC))
    return export_tree_theorem(decision.induction_proof, checks=())


def test_the_w_state_copy_pair_takes_the_atom_form() -> None:
    source = _source()
    assert "import ArchytasZX.PathSumNetAmp" in source
    for step in (".psup [1] [2]", ".aphase [2] 7 2", ".aflip 0 4 3"):
        assert step in source
    assert COPY_WEIGHT in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_checks_the_w_state_tree_families(tmp_path: object) -> None:
    ok, output = L.check(_source(), f"{tmp_path}/wtree.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_rejects_a_tampered_copy_weight(tmp_path: object) -> None:
    """The fan copy's weight sqrt(2) / 2 changed to 1 / 2 fails the copy pair's constants."""
    source = _source()
    tampered = source.replace(COPY_WEIGHT, "((1 : ℂ) / 2) * ex (Φ 9 v)")
    assert tampered != source
    ok, _output = L.check(tampered, f"{tmp_path}/wtree_bad.lean")
    assert not ok
