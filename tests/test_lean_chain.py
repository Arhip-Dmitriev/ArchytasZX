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

"""The chain-induction Lean export: chainNet families from a chain_window induction; tree
families in heap order."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_chain import export_chain, export_tree, export_tree_theorem

CIRCUIT = "\n".join(
    [
        "dim d = 2",
        "count n",
        "g0 = X(0 -> 1, d)",
        "g1 = F(d)",
        "g2 = X(0 -> 1, d)",
        "g3 = F(d)",
        "g4 = Z(1 -> 2, d)",
        "g5 = Z(2 -> 1, d)",
        "g6 = F(d)",
        "wire g0.out0 -- g1.in0, g2.out0 -- g3.in0, g1.out0 -- g4.in0, g3.out0 -- g5.in0, "
        + "g4.out1 -- g6.in0, g6.out0 -- g5.in1",
        "repeat n as C0: g2, g3, g4, g5, g6",
        "chain C0: g5.out0 -> g4.in0",
        "outputs g4.out0, g5.out0",
        "scalar 1/sqrt(d)",
    ]
)
"""The cluster_state circuit: |+> on q, then per copy |+> on r and CZ(q, r), q carried to r."""

_SPEC_LINES = (
    "dim d = 2",
    "count n",
    "h = Z(0 -> 1, d)",
    "repeat n as V: v = Z(1 -> 2, d), f = F(d), w = Z(1 -> 1, d)",
    "wire h.out0 -- v.in0, v.out1 -- f.in0, f.out0 -- w.in0",
    "chain V: w.out0 -> v.in0",
    "outputs v.out0, w.out0",
    "scalar d^(-1/2)",
)
SPEC = "\n".join(_SPEC_LINES)
"""The path graph state as a chain of Z spiders joined by F edges."""


INDEXED = (
    CIRCUIT.replace("g6 = F(d)", "g6 = F(d)\ng7 = Z(1 -> 1, d) phase {1: i/4}")
    .replace("g6.out0 -- g5.in1", "g6.out0 -- g7.in0, g7.out0 -- g5.in1")
    .replace(
        "repeat n as C0: g2, g3, g4, g5, g6", "box C0 x n nodes g2, g3, g4, g5, g6, g7 index i"
    ),
    SPEC.replace(
        "repeat n as V: v = Z(1 -> 2, d), f = F(d), w = Z(1 -> 1, d)",
        "v = Z(1 -> 2, d)\nf = F(d)\nw = Z(1 -> 1, d) phase {1: i/4}\n"
        "box V x n nodes v, f, w index i",
    ),
)
"""The cluster chains with a phase i/4 on copy i of both sides."""

SCALED = (CIRCUIT + "\ncopy scalar C0: 2", SPEC + "\ncopy scalar V: 2")
"""The cluster chains with the copy scalar 2 on both sides."""


def _theorem(pair: tuple[str, str] = (CIRCUIT, SPEC)) -> object:
    decision = decide_equal(parse_dsl(pair[0]), parse_dsl(pair[1]))
    assert decision.induction_proof is not None
    return decision.induction_proof


def test_the_cluster_chain_exports_as_two_families() -> None:
    source = export_chain(_theorem(), {"d": 2}, "cluster")
    assert "theorem main : ∀ k, Equal E" in source
    assert "def cut : List ℕ := [0]" in source
    assert "chain_equal E W cut" in source


def test_the_instance_at_a_checked_k_is_linked() -> None:
    source = export_chain(_theorem(SCALED), {"d": 2}, "scaled", checks=(1, 2))
    assert "namespace At1" in source and "namespace At2" in source
    assert "theorem inst : Equal E (((2 : ℂ) * (rt 2))) DL (((2 : ℂ) * (rt 2))) DR :=" in source


def test_the_qudit_cluster_chain_takes_the_qudit_networks() -> None:
    source = export_chain(_theorem(), {"d": 3}, "cluster3")
    assert "import ArchytasZX.PathSumQuditNet" in source
    assert "def Δ (k : ℕ) : ℕ := 3" in source


def test_an_indexed_chain_keeps_one_period_of_copies() -> None:
    source = export_chain(_theorem(INDEXED), {"d": 2}, "indexed")
    assert "namespace Copy3" in source and "namespace Copy4" not in source
    assert "generalize i % 4 = j at h ⊢" in source


def test_a_copy_scalar_scales_each_copy() -> None:
    source = export_chain(_theorem(SCALED), {"d": 2}, "scaled")
    assert "noncomputable def aOt : List ℂ := [((2 : ℂ))]" in source


def test_a_theorem_without_a_chain_window_is_refused() -> None:
    from archytaszx.semantics import kernel as K

    wire = parse_dsl("a = Z(1 -> 1, literal 2)")
    with pytest.raises(PathSumUnsupported):
        export_chain(K.refl(wire))


def _tree(d: int) -> str:
    base = parse_dsl(f"r = Z(0 -> 1, literal {d})")
    cop_o = parse_dsl(f"g = Z(1 -> 2, literal {d})")
    cop_x = parse_dsl(
        f"a = Z(1 -> 1, literal {d}); g = Z(1 -> 2, literal {d}); wire a.out0 -- g.in0"
    )
    return export_tree((base, base), [(cop_o, cop_x)], 0, 2, [0, 1], f"tree{d}")


def test_a_binary_tree_plugs_its_copies_in_heap_order() -> None:
    source = _tree(3)
    assert "def p : ℕ → ℕ := heapPlug W leaf0 kids" in source
    assert "tree_equal E W r p" in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_cluster_chain(tmp_path: object) -> None:
    ok, output = L.check(export_chain(_theorem(), {"d": 2}, "cluster"), f"{tmp_path}/chain.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_tree_families(tmp_path: object) -> None:
    for d in (2, 3):
        ok, output = L.check(_tree(d), f"{tmp_path}/tree{d}.lean")
        assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize(
    ("pair", "d"), [((CIRCUIT, SPEC), 3), ((CIRCUIT, SPEC), 4), (INDEXED, 2), (SCALED, 2)]
)
def test_lean_proves_the_lifted_chains(pair: tuple[str, str], d: int, tmp_path: object) -> None:
    ok, output = L.check(export_chain(_theorem(pair), {"d": d}, "lifted"), f"{tmp_path}/c.lean")
    assert ok, output


def _reversed_pair(scalar: str | None = None) -> tuple[str, str]:
    from tests.test_chain_order import _reversed_indexed

    circuit, spec = _reversed_indexed()
    if scalar is None:
        return circuit, spec
    return circuit + f"\ncopy scalar C0: {scalar}", spec + f"\ncopy scalar V: {scalar}"


def test_an_offset_induction_exports_over_every_offset() -> None:
    from archytaszx.semantics.kernel import Theorem

    theorem = _theorem(_reversed_pair())
    assert isinstance(theorem, Theorem) and theorem.inference.kind == "induction_offset"
    source = export_chain(theorem, {"d": 2}, "reversed", checks=(1, 2))
    assert "theorem main : ∀ s k, Equal E" in source
    assert "(chainNet W cut Base.NL (fun j => cO (s + k - 1 - j)) k)" in source
    assert "    (fun _ => hD _ _) (main 0 2) (by decide) (by decide)" in source
    growing = export_chain(_theorem(_reversed_pair("2^i")), {"d": 2}, "growing", checks=(1, 2))
    assert "noncomputable def aO (i : ℕ) : ℂ := ((16 : ℂ)) ^ (i / 4) * perIdx aOt 1 i" in growing
    assert "  · exact copyS _ copyEq3" in growing


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize("scalar", [None, "2", "2^i"])
def test_lean_proves_the_offset_chains(scalar: str | None, tmp_path: object) -> None:
    source = export_chain(_theorem(_reversed_pair(scalar)), {"d": 2}, "offset", checks=(1, 2))
    ok, output = L.check(source, f"{tmp_path}/offset.lean")
    assert ok, output


MIXED = (
    CIRCUIT.replace("g6 = F(d)", "g6 = F(d)\ng8 = Z(1 -> 1, literal 4)")
    .replace("repeat n as C0: g2, g3, g4, g5, g6", "repeat n as C0: g2, g3, g4, g5, g6, g8")
    .replace("outputs g4.out0, g5.out0", "outputs g4.out0, g8.out0, g5.out0\ninputs g8.in0"),
    SPEC.replace("w = Z(1 -> 1, d)", "w = Z(1 -> 1, d), u = Z(1 -> 1, literal 4)").replace(
        "outputs v.out0, w.out0", "outputs v.out0, u.out0, w.out0\ninputs u.in0"
    ),
)
"""The cluster chains with a dimension-4 wire through each copy."""


def test_a_mixed_dimension_chain_exports_over_level_labels() -> None:
    source = export_chain(_theorem(MIXED), {"d": 2}, "mixed")
    assert "import ArchytasZX.ChainMixed" in source
    assert "theorem main : ∀ k, Equal (lvEnv E W C B k)" in source
    assert "chain_equal_lv E W cut bO cO bX cX C B hC hB hcut" in source
    assert "noncomputable def EK : Env := { E with labs := instLabs ++ lvls W C 2 ++ B }" in source


MIXED_GROWING = (
    MIXED[0].replace(
        "repeat n as C0: g2, g3, g4, g5, g6, g8",
        "box C0 x n nodes g2, g3, g4, g5, g6, g8 index i",
    )
    + "\ncopy scalar C0: 2^i",
    MIXED[1].replace(
        "repeat n as V: v = Z(1 -> 2, d), f = F(d), w = Z(1 -> 1, d), u = Z(1 -> 1, literal 4)",
        "v = Z(1 -> 2, d)\nf = F(d)\nw = Z(1 -> 1, d)\nu = Z(1 -> 1, literal 4)\n"
        "box V x n nodes v, f, w, u index i",
    )
    + "\ncopy scalar V: 2^i",
)
"""The mixed-dimension cluster chains with copy scalar 2^i on both sides."""


def test_a_growing_copy_scalar_exports_over_mixed_dimensions() -> None:
    source = export_chain(_theorem(MIXED_GROWING), {"d": 2}, "mixedgrow", checks=(1, 2))
    assert "noncomputable def aO (i : ℕ) : ℂ := ((2 : ℂ)) ^ (i / 1) * perIdx aOt 1 i" in source
    assert "  · exact copyS _ copyEq0" in source
    assert "theorem main : ∀ k, Equal (lvEnv E W C B k)" in source


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize("pair", [MIXED, MIXED_GROWING])
def test_lean_proves_the_mixed_dimension_chain(pair: tuple[str, str], tmp_path: object) -> None:
    source = export_chain(_theorem(pair), {"d": 2}, "mixed")
    ok, output = L.check(source, f"{tmp_path}/mixed.lean")
    assert ok, output


def _tree_window() -> object:
    from archytaszx.semantics import kernel as K
    from archytaszx.semantics.decide import decide_equal
    from tests.test_tree_window import SPEC as TSPEC
    from tests.test_tree_window import TREE

    proof = decide_equal(parse_dsl(TREE), parse_dsl(TSPEC), use_symbolic=False).proof
    assert isinstance(proof, K.Theorem)
    stack = [proof]
    while stack:
        t = stack.pop()
        if t.inference.kind == "induction":
            return t
        stack += t.inference.premises
    raise AssertionError("no induction")


def test_the_tree_window_theorem_exports_as_tree_families() -> None:
    source = export_tree_theorem(_tree_window(), {"d": 2}, "ghz")
    assert "tree_equal E W r p" in source
    assert "def kids : List ℕ := [1572864, 1572865]" in source


def test_the_tree_families_are_linked_to_the_instances_by_fusion() -> None:
    source = export_tree_theorem(_tree_window(), {"d": 2}, "ghz", checks=(1, 2))
    assert "namespace At1" in source and "namespace At2" in source
    assert "theorem linkR : ((treeNet W r p Base.NR cX 3).mapL (ren πR)).nodes.Perm" in source
    assert "  (instK.congr_scalar scalL scalR).trans Fuse.fuseEq" in source


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_tree_window_theorem(tmp_path: object) -> None:
    for d in (2, 3):
        source = export_tree_theorem(_tree_window(), {"d": d}, f"ghz{d}")
        ok, output = L.check(source, f"{tmp_path}/ghz{d}.lean")
        assert ok, output
