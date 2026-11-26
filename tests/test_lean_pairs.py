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

"""The pair box family export: complete_graph_state's point instance ends in ``DEqual`` of the
two pair box expansions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archytaszx.semantics import kernel as K
from archytaszx.semantics import lean_net as L
from benchmarks import stages, subject


def _export(n: int, tmp_path: object) -> str:
    sid = "complete_graph_state"
    inst = next(i for i in subject.instances([s for s in subject.catalogue() if s.id == sid]))
    run = stages.new_run()
    stages.decide_stage(run, subject.script(inst, n, 2).text)
    path = Path(f"{tmp_path}/{sid}_{n}.lean")
    stages.lean_export_stage(run, path)
    return path.read_text(encoding="utf-8")


def test_the_pair_export_ends_in_dequal_of_the_expansions(tmp_path: object) -> None:
    source = _export(2, tmp_path)
    assert "DEqual_of_pairNet okA okB" in source
    assert "theorem main_diagrams (d : ℕ) (h_d : 1 ≤ d) (n : ℕ)" in source
    assert K.diagram_level(source)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_pair_diagrams_and_rejects_a_tampered_copy(tmp_path: object) -> None:
    source = _export(2, tmp_path)
    ok, output = L.check(source, f"{tmp_path}/cgs.lean")
    assert ok, output
    bad = source.replace("⟨1, .F, 0, 0, [0], [0]⟩", "⟨1, .X, 0, 0, [0], [0]⟩", 1)
    assert bad != source
    ok, _ = L.check(bad, f"{tmp_path}/cgs_bad.lean")
    assert not ok
