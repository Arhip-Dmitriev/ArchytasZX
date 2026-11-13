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

"""The kernel's w_fusion step in the network export: a W leg fanned into n copies of a phased
Z spider, fused into the weighted Z spider before the W node (wFuseZ, WFuse.lean)."""

from __future__ import annotations

import os

import pytest

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics import lean_net as L

FAN = """count n
x = X(0 -> 1, literal 2) phase{1: 1/2}
w = W(1 -> 1, literal 2)
z = Z(1 -> 1, literal 2) phase{1: 1/8}
wire x.out0 -- w.in0, w.out0 -- z.in0
repeat n as C: z
outputs z.out0"""
"""A W node on |1> whose leg feeds n copies of Z(pi/4)."""


def _fused() -> kernel.Theorem:
    return kernel.w_fusion(parse_dsl(FAN))


def test_w_fusion_exports_without_axioms() -> None:
    source = L.export(_fused())
    assert "wFuseZ" in source
    assert "\naxiom " not in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_checks_w_fusion_and_rejects_a_changed_weight(tmp_path: object) -> None:
    source = L.export(_fused())
    ok, output = L.check(source, f"{tmp_path}/wfuse.lean")
    assert ok, output
    weight = "| 3 => fun k => if k = 1 then (om 8) else 1"
    assert weight in source
    tampered = source.replace(weight, "| 3 => fun k => if k = 1 then (om 4) else 1")
    ok, _ = L.check(tampered, f"{tmp_path}/wfuse_tampered.lean")
    assert not ok
