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

"""Qudit network proofs over digit variables (lean_qdigits)."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.lean_qdigits import export_qudit_digits_net

_D = "literal 3"

DRAPER = (
    "; ".join(
        [
            f"g0 = F({_D}); g1 = B({_D}, {_D})",
            "g2 = Z(1 -> 1, literal 9) phase {4: 1/3, 5: 2/3, 7: 2/3, 8: 1/3}",
            f"g3 = S({_D}, {_D}); g4 = F({_D}); g5 = F({_D}); g6 = F({_D})",
            "wire g0.out0 -- g1.in1; wire g1.out0 -- g2.in0; wire g2.out0 -- g3.in0",
            "wire g3.out1 -- g4.in0; wire g4.out0 -- g5.in0; wire g5.out0 -- g6.in0",
            "inputs g1.in0, g0.in0; outputs g3.out0, g6.out0",
        ]
    ),
    "; ".join(
        [
            f"v0 = Z(1 -> 1, {_D}); v1 = Z(1 -> 1, {_D}); v2 = Z(1 -> 2, {_D})",
            f"v3 = X(2 -> 1, {_D})",
            "wire v0.out0 -- v2.in0; wire v1.out0 -- v3.in0; wire v2.out1 -- v3.in1",
            "inputs v0.in0, v1.in0; outputs v2.out0, v3.out0; scalar sqrt(3)",
        ]
    ),
)
"""The Draper adder on one qutrit and its controlled-shift form."""


def _export() -> str:
    return export_qudit_digits_net(parse_dsl(DRAPER[0]), parse_dsl(DRAPER[1]))


def test_a_binder_and_a_wide_spider_take_digit_fragments() -> None:
    source = _export()
    assert "theorem main : Equal E" in source
    assert "b_frag_q" in source and "s_frag_q" in source and "z_frag_qd" in source
    assert "lift_q" in source and ".pin " in source


def test_single_digit_networks_are_left_to_the_qudit_export() -> None:
    wire = parse_dsl(f"a = Z(1 -> 1, {_D})")
    with pytest.raises(PathSumUnsupported):
        export_qudit_digits_net(wire, wire)


def test_unequal_digit_networks_are_refused() -> None:
    wrong = DRAPER[0].replace("4: 1/3", "4: 2/3")
    with pytest.raises(PathSumUnsupported):
        export_qudit_digits_net(parse_dsl(wrong), parse_dsl(DRAPER[1]))


def test_a_digit_contraction_exports_at_the_diagram_level() -> None:
    from archytaszx.repl.commands import Session, run
    from archytaszx.semantics import kernel

    s = Session()
    s.names.update(l=parse_dsl(DRAPER[0]), r=parse_dsl(DRAPER[1]))
    run(s, "load @l")
    run(s, "decide against @r")
    run(s, "certificate kernel")
    assert isinstance(s.certificate, kernel.Theorem)
    assert s.certificate.inference.kind == "contraction"
    source = L.export(s.certificate)
    assert kernel.diagram_level(source) and "theorem main_diagrams" in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


_FUSED = (
    "a = Z(1 -> 1, literal 9); b = Z(1 -> 1, literal 9); wire a.out0 -- b.in0",
    "a = Z(1 -> 1, literal 9)",
)
"""Two fused spiders on 9: boundary labels 0 and 2, relabelled onto the slots."""


def _fused() -> str:
    return export_qudit_digits_net(parse_dsl(_FUSED[0]), parse_dsl(_FUSED[1]), diagrams=True)


def test_a_digit_network_export_ends_in_dequal_of_the_diagrams() -> None:
    from archytaszx.semantics import kernel

    source = _fused()
    assert "def πB : List (ℕ × ℕ) := [(0, 0), (1, 2), (2, 1)]" in source
    assert "theorem main_diagrams : DEqual { E with labs := labsD }" in source
    assert kernel.diagram_level(source) and "Equal.relabel" in source
    plain = export_qudit_digits_net(parse_dsl(_FUSED[0]), parse_dsl(_FUSED[1]))
    assert not kernel.diagram_level(plain)


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_refuses_a_tampered_digit_diagram_literal(tmp_path: object) -> None:
    source = _fused()
    tampered = source.replace("⟨1, .Z, 2, 0, [2], [2]⟩", "⟨1, .X, 2, 0, [2], [2]⟩", 1)
    assert tampered != source
    ok, _ = L.check(tampered, f"{tmp_path}/qdigits_tampered.lean")
    assert not ok


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_digit_network_pair(tmp_path: object) -> None:
    ok, output = L.check(_export(), f"{tmp_path}/qdigits.lean")
    assert ok, output
