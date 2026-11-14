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
by a ``net`` axiom, and Lean accepting the proof and rejecting a wrong copy count; link ports on
the boundary behind an identity spider (79.16e); identity slides through
``evalL_slide`` (79.16d)."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import cast

import pytest

from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics import kernel, lean_linked
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.induction import rename_multiplicity
from archytaszx.semantics.lean_net import check, lean_project
from archytaszx.semantics.prove import InductionSide, _with_scalar_index
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


def _succ_peels(t: kernel.Theorem) -> Iterator[tuple[Diagram, Diagram]]:
    """The successor peels over ``m`` of the induction steps under ``t``: ``d`` at
    ``m = k + 1`` and its peel."""
    seen: set[int] = set()
    stack = [t]
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        stack.extend(s.inference.premises)
        data = s.inference.data
        if s.inference.kind != "induction" or str(data["index"]) != "m":
            continue
        own, other = (s.lhs, s.rhs) if data["side"] is InductionSide.START else (s.rhs, s.lhs)
        value = Mult(str(data["step"])) + 1
        for d, peeled in ((own, data["peeled"]), (other, data["other_peeled"])):
            before = rename_multiplicity(d, "m", value)
            yield _with_scalar_index(before, "m", value), cast(Diagram, peeled)


def test_successor_peels_with_boundary_link_ports_export() -> None:
    theorem = _cluster()
    pairs = list(_succ_peels(theorem))
    assert len(pairs) == 4
    for before, peeled in pairs:
        source = lean_linked.export_peel(before, peeled)
        assert source is not None and "evalL_peel ok" in source
    assert "\naxiom peel" not in kernel.lean(theorem, "main", semantic=True)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_a_boundary_link_peel_and_refuses_a_moved_end(tmp_path: object) -> None:
    before, peeled = next(_succ_peels(_cluster()))
    source = lean_linked.export_peel(before, peeled)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/peel.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    found = re.search(r"links := \[⟨0, (⟨[^⟩]*⟩, ){2}(⟨[^⟩]*⟩)", source.split("def L' ")[1])
    assert found is not None
    bad = source.replace(found[0], found[0].replace(found[2], "⟨0, false, 0⟩"))
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


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


def _slides(t: kernel.Theorem) -> list[kernel.Theorem]:
    """The identity slide steps under ``t``."""
    seen: set[int] = set()
    out, stack = [], [t]
    while stack:
        s = stack.pop()
        if id(s) not in seen:
            seen.add(id(s))
            stack.extend(s.inference.premises)
            if s.inference.kind == "identity_slide":
                out.append(s)
    return out


def test_the_cluster_state_2d_identity_slide_exports_through_evalL_slide() -> None:
    theorem = _cluster()
    slides = _slides(theorem)
    assert slides
    for t in slides:
        source = lean_linked.export_slide(t.lhs, t.rhs)
        assert source is not None and "evalL_slide ok ok' same" in source
    source = kernel.lean(theorem, "main", semantic=True)
    assert "namespace ArchytasZX.Export.S" in source
    assert "a boxed identity spider moved" not in source


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_slide_and_refuses_a_phased_identity(tmp_path: object) -> None:
    (t, *_) = _slides(_cluster())
    source = lean_linked.export_slide(t.lhs, t.rhs)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/slide.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = re.sub(r"E\.phase (\d+) k = 0", lambda m: f"E.phase {int(m[1]) + 1} k = 0", source)
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
