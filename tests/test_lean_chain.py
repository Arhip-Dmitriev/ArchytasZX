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

"""The chain-induction Lean export: chainNet families from a chain_window induction."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_chain import export_chain

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


def _theorem() -> object:
    decision = decide_equal(parse_dsl(CIRCUIT), parse_dsl(SPEC))
    assert decision.induction_proof is not None
    return decision.induction_proof


def test_the_cluster_chain_exports_as_two_families() -> None:
    source = export_chain(_theorem(), {"d": 2}, "cluster")
    assert "theorem main : ∀ k, Equal E" in source
    assert "def cut : List ℕ := [0]" in source
    assert "chain_equal E W cut" in source


def test_a_theorem_without_a_chain_window_is_refused() -> None:
    from archytaszx.semantics import kernel as K

    wire = parse_dsl("a = Z(1 -> 1, literal 2)")
    with pytest.raises(PathSumUnsupported):
        export_chain(K.refl(wire))


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_cluster_chain(tmp_path: object) -> None:
    ok, output = L.check(export_chain(_theorem(), {"d": 2}, "cluster"), f"{tmp_path}/chain.lean")
    assert ok, output
