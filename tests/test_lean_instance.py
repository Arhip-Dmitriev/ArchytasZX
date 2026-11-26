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

"""A restricted family export whose family theorem is encoding level keeps it and ends in the
``DEqual`` of the instantiated diagrams (namespace ``ArchytasZX.Export.Inst``)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from archytaszx.semantics import kernel as K
from archytaszx.semantics import lean_net as L
from benchmarks import stages, subject


def _export(sid: str, tmp_path: object) -> str:
    inst = next(i for i in subject.instances([s for s in subject.catalogue() if s.id == sid]))
    run = stages.new_run()
    stages.decide_stage(run, subject.script(inst, 2, 2).text)
    path = Path(f"{tmp_path}/{sid}.lean")
    stages.lean_export_stage(run, path)
    return path.read_text(encoding="utf-8")


def test_the_family_and_the_instance_are_both_in_the_file(tmp_path: object) -> None:
    source = _export("cyclic_shift", tmp_path)
    assert "ChainNode.cyclic_every" in source
    assert "namespace ArchytasZX.Export.Inst" in source
    assert source.rstrip().endswith("end ArchytasZX.Export.Inst")
    assert K.diagram_level(source)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_instance_and_rejects_a_tampered_one(tmp_path: object) -> None:
    source = _export("cyclic_shift", tmp_path)
    ok, output = L.check(source, f"{tmp_path}/cs.lean")
    assert ok, output
    head, inst = source.split("namespace ArchytasZX.Export.Inst", 1)
    m = re.compile(r"\.Z,").search(inst, inst.index("def D1 : Diagram"))
    assert m is not None
    bad = f"{head}namespace ArchytasZX.Export.Inst{inst[: m.start()]}.X,{inst[m.end() :]}"
    ok, _ = L.check(bad, f"{tmp_path}/cs_bad.lean")
    assert not ok
