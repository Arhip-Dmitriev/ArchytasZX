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

"""The copy-index phase export: the point instances of bv_secret_pattern as the family theorem
of the gadget pair, restated over the linked diagrams, and its corollary at n."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from archytaszx.semantics import lean_net as L
from benchmarks import stages, subject


def _export(n: int, tmp_path: object) -> str:
    sub = [s for s in subject.catalogue() if s.id == "bv_secret_pattern"]
    inst = next(i for i in subject.instances(sub))
    run = stages.new_run()
    stages.decide_stage(run, subject.script(inst, n, 490).text)
    assert stages.certificate_stage(run)["kernel"] == "symbolic"
    path = Path(f"{tmp_path}/bvs_{n}.lean")
    stages.lean_export_stage(run, path)
    return path.read_text(encoding="utf-8")


def test_the_point_export_is_the_gadget_family_and_its_corollary(tmp_path: object) -> None:
    small, large = _export(1000, tmp_path), _export(1000000, tmp_path)
    for source in (small, large):
        assert "theorem main_diagrams_original :" in source
        assert "LDiagram.evalL_eq_of_gad" in source
        assert "cph := [(" in source
    assert "main_diagrams_original (490 : ℕ) (by norm_num) (1000000 : ℕ) σ" in large
    assert re.sub(r"1000000|1000", "N", small) == re.sub(r"1000000|1000", "N", large)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_copy_phase_family_and_rejects_a_tampered_copy(tmp_path: object) -> None:
    source = _export(1000000, tmp_path)
    ok, output = L.check(source, f"{tmp_path}/bvs.lean")
    assert ok, output
    bad = source.replace("((Nat.unpair p).2 : ℝ)", "((Nat.unpair p).2 + 1 : ℝ)", 1)
    assert bad != source
    ok, _ = L.check(bad, f"{tmp_path}/bvs_bad.lean")
    assert not ok
