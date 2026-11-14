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

"""Kernel peel steps exported over linked diagrams (REMAINING_BUGS_SPEC 79.16a): the
cluster_state_2d peel proved by ``evalL_peel``, its reverse refused, the semantic export linking it
by a ``net`` axiom, and Lean accepting the proof and rejecting a wrong copy count."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from archytaszx.semantics import kernel, lean_linked
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_net import check, lean_project
from tests.test_nested_chain import _grid_pair


def _peels(t: kernel.Theorem, seen: set[int] | None = None) -> Iterator[kernel.Theorem]:
    seen = set() if seen is None else seen
    if id(t) in seen:
        return
    seen.add(id(t))
    if t.inference.kind == "peel":
        yield t
    for p in t.inference.premises:
        yield from _peels(p, seen)


def _cluster() -> kernel.Theorem:
    left, right = _grid_pair(1)
    return kernel.from_decision(decide_equal(left, right, samples=[]), left, right)


def test_the_cluster_state_2d_peel_exports_through_evalL_peel() -> None:
    theorem = _cluster()
    peels = list(_peels(theorem))
    assert peels
    for t in peels:
        source = lean_linked.export_peel(t.lhs, t.rhs)
        assert source is not None and "evalL_peel ok" in source
        assert lean_linked.export_peel(t.rhs, t.lhs) is None
    source = kernel.lean(theorem, "main", semantic=True)
    assert "namespace ArchytasZX.Export.P" in source
    assert "a joint peel of one box family" not in source


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_peel_and_refuses_a_wrong_copy_count(tmp_path: object) -> None:
    (t, *_) = _peels(_cluster())
    source = lean_linked.export_peel(t.lhs, t.rhs)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/peel.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("+ 1 →", "+ 2 →")
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
