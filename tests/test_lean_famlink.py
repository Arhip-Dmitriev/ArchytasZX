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

"""Linked family inductions (81.60-d): ghz_metrology, whose goal's phase reads the box
multiplicity, exported with the induction as ``Lnk<i>.P`` at ``F n``, no axiom; its point
claims as that theorem with its corollary, of one size at every point; Lean accepting the
family theorem and rejecting a goal phase changed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archytaszx.semantics.lean_famlink import linked_indices
from archytaszx.semantics.lean_net import check, lean_project
from benchmarks import stages
from benchmarks import subject as S


def _export(tmp: Path, nv: int | None) -> str:
    """The Lean export of ghz_metrology/main at ``n = nv`` (None: n free)."""
    sub = next(s for s in S.catalogue() if s.id == "ghz_metrology")
    sc = S.script(S.Instance(sub, sub.equations[0]), nv, None)
    run_ = stages.new_run()
    assert stages.decide_stage(run_, sc.text)["verdict"] == "equal"
    path = tmp / f"gm_{nv}.lean"
    stages.lean_export_stage(run_, path)
    return path.read_text(encoding="utf-8")


def test_the_family_theorem_links_its_induction(tmp_path: Path) -> None:
    sub = next(s for s in S.catalogue() if s.id == "ghz_metrology")
    sc = S.script(S.Instance(sub, sub.equations[0]), None, None)
    run_ = stages.new_run()
    stages.decide_stage(run_, sc.text)
    assert linked_indices(stages.execute(run_, "certificate kernel").detail) == {"n"}
    source = _export(tmp_path, None)
    assert "\naxiom " not in source
    assert "theorem main (n : ℕ) :\n    Equal ((F n).env (ksOf [n]))" in source
    assert "Equal.of_phRows" in source


def test_point_claims_are_the_corollary_at_one_size(tmp_path: Path) -> None:
    sizes = {}
    for nv in (61, 216):
        source = _export(tmp_path, nv)
        assert "\naxiom " not in source
        assert f"main_diagrams ({nv} : ℕ)" in source
        sizes[nv] = len(source.encode())
    assert abs(sizes[61] - sizes[216]) < 64


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_family_and_refuses_a_changed_phase(tmp_path: Path) -> None:
    source = _export(tmp_path, None)
    ok, output = check(source, f"{tmp_path}/fam.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    row = "if k = 1 then (((1 : ℝ) / 8) + (((1 : ℝ) / 8) * (table_n : ℝ)))"
    assert row in source
    bad = source.replace(row, row.replace("/ 8) +", "/ 4) +", 1))
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
