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

"""A chain box against a node box (82.AQ-e): cyclic_shift exported through
``ChainNode.cyclic_every``, no axiom, its point claims the corollary of one size at every point;
Lean accepting the family theorem and rejecting a changed copy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archytaszx.semantics.lean_net import check, lean_project
from benchmarks import stages
from benchmarks import subject as S


def _export(tmp: Path, nv: int | None) -> str:
    """The Lean export of cyclic_shift/main at ``n = nv`` (None: n free)."""
    sub = next(s for s in S.catalogue() if s.id == "cyclic_shift")
    sc = S.script(S.Instance(sub, sub.equations[0]), nv, None)
    run_ = stages.new_run()
    assert stages.decide_stage(run_, sc.text)["verdict"] == "equal"
    path = tmp / f"cs_{nv}.lean"
    stages.lean_export_stage(run_, path)
    return path.read_text(encoding="utf-8")


def test_the_family_goes_through_cyclic_every(tmp_path: Path) -> None:
    source = _export(tmp_path, None)
    assert "\naxiom " not in source
    assert "ChainNode.cyclic_every" in source


def test_point_claims_are_the_corollary_at_one_size(tmp_path: Path) -> None:
    sizes = {}
    for nv in (10**3, 10**6):
        source = _export(tmp_path, nv)
        assert f"main_diagrams d ({nv} : ℕ)" in source
        sizes[nv] = len(source.encode())
    assert abs(sizes[10**3] - sizes[10**6]) < 64


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_family_and_refuses_a_changed_copy(tmp_path: Path) -> None:
    source = _export(tmp_path, 10**6)
    ok, output = check(source, f"{tmp_path}/fam.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    head = "def D0 : Diagram :=\n  { nodes := [⟨0, .Z,"
    assert head in source
    ok, output = check(source.replace(head, head.replace(".Z", ".X")), f"{tmp_path}/bad.lean")
    assert not ok
