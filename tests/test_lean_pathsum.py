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

"""The qudit path-sum Lean export over Z_p: step lists, Gauss sums in Legendre form, Lean."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.lean_pathsum import export_qudit_proof

_D = "literal 3"

_CUP5 = "b = Z(1 -> 0, literal 5); wire a.out0 -- b.in0"

PAIRS: dict[str, tuple[str, str]] = {
    "gauss": (
        "; ".join(
            [
                f"a = F({_D}); b = Z(1 -> 1, {_D}) phase {{1: 1/3, 2: 1/3}}; c = F({_D})",
                "wire a.out0 -- b.in0; wire b.out0 -- c.in0",
            ]
        ),
        "; ".join(
            [
                f"a = Z(1 -> 1, {_D}) phase {{1: 2/3, 2: 2/3}}; b = F({_D})",
                f"c = Z(1 -> 1, {_D}) phase {{1: 2/3, 2: 2/3}}",
                "wire a.out0 -- b.in0; wire b.out0 -- c.in0; scalar I",
            ]
        ),
    ),
    "f4": (
        "; ".join(
            [
                f"a = F({_D}); b = F({_D}); c = F({_D}); e = F({_D})",
                "wire a.out0 -- b.in0; wire b.out0 -- c.in0; wire c.out0 -- e.in0",
            ]
        ),
        f"a = Z(1 -> 1, {_D})",
    ),
    "xx": (
        f"a = X(1 -> 2, {_D}); b = X(2 -> 1, {_D}); wire a.out0 -- b.in0; wire a.out1 -- b.in1",
        f"a = Z(1 -> 1, {_D})",
    ),
    "g5a2": (
        f"a = Z(0 -> 1, literal 5) phase {{1: 2/5, 2: 3/5, 3: 3/5, 4: 2/5}}; {_CUP5}",
        f"a = Z(0 -> 1, literal 5); {_CUP5}; scalar -sqrt(5)/5",
    ),
}


def _export(name: str) -> str:
    left, right = PAIRS[name]
    return export_qudit_proof(parse_dsl(left), parse_dsl(right), name)


def test_a_gauss_step_is_replayed_with_its_legendre_form() -> None:
    source = _export("gauss")
    assert "def stepsL : List QSem.Step := [.perm [2], .gauss 1, .norm]" in source
    assert "rw [qg_legendre (by norm_num) 1 (by decide)]" in source
    assert "theorem lg2 : lg 3 2 = -1 := by decide" in source


def test_pivots_and_constraints_become_steps() -> None:
    source = _export("f4")
    assert ".hhPivot" in source and ".hhCons" in source and ".canon 0" in source


def test_a_non_residue_gauss_sum_takes_its_sign() -> None:
    source = _export("g5a2")
    assert ".gauss 2" in source and "theorem lg2 : lg 5 2 = -1 := by decide" in source


def test_unequal_qudit_sums_are_refused() -> None:
    left, _ = PAIRS["gauss"]
    with pytest.raises(PathSumUnsupported):
        export_qudit_proof(parse_dsl(left), parse_dsl(f"a = Z(1 -> 1, {_D})"))


def test_qubit_sums_are_refused() -> None:
    wire = parse_dsl("a = Z(1 -> 1, literal 2)")
    with pytest.raises(PathSumUnsupported):
        export_qudit_proof(wire, wire)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


needs_lean = pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")


@needs_lean
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_lean_proves_a_qudit_pair(name: str, tmp_path: object) -> None:
    ok, output = L.check(_export(name), f"{tmp_path}/{name}.lean")
    assert ok, output


@needs_lean
def test_lean_rejects_a_tampered_qudit_export(tmp_path: object) -> None:
    source = _export("gauss")
    marker = "([(0, 2)], 2)"
    assert marker in source
    ok, _ = L.check(source.replace(marker, "([(0, 2)], 1)"), f"{tmp_path}/tampered.lean")
    assert not ok
