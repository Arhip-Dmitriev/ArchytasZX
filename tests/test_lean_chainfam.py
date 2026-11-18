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

"""The closed chain family export: the point instances of teleportation_chain and
repeater_chain as the family theorem and its corollary at n."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from archytaszx.semantics import lean_net as L
from benchmarks import stages, subject


def _export(sid: str, n: int, tmp_path: object) -> str:
    inst = next(i for i in subject.instances([s for s in subject.catalogue() if s.id == sid]))
    run = stages.new_run()
    stages.decide_stage(run, subject.script(inst, n, None).text)
    assert stages.certificate_stage(run)["kernel"] == "symbolic"
    path = Path(f"{tmp_path}/{sid}_{n}.lean")
    stages.lean_export_stage(run, path)
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("sid", ["teleportation_chain", "repeater_chain"])
def test_the_point_export_is_the_family_and_its_corollary(sid: str, tmp_path: object) -> None:
    small, large = _export(sid, 1000, tmp_path), _export(sid, 1000000, tmp_path)
    for source in (small, large):
        assert "chain_pair okA okB" in source
        assert "theorem family (d : ℕ) (h_d : 1 ≤ d) (k : ℕ)" in source
    assert "main d h_d (1000000 : ℕ)" in large
    assert re.sub(r"1000000|1000", "N", small) == re.sub(r"1000000|1000", "N", large)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_teleportation_family_and_rejects_a_tampered_copy(
    tmp_path: object,
) -> None:
    source = _export("teleportation_chain", 1000000, tmp_path)
    ok, output = L.check(source, f"{tmp_path}/tc.lean")
    assert ok, output
    bad = source.replace("⟨1, .F, 0, 0, [0], [0]⟩", "⟨1, .X, 0, 0, [0], [0]⟩", 1)
    assert bad != source
    ok, _ = L.check(bad, f"{tmp_path}/tc_bad.lean")
    assert not ok
