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

"""lean_adder: the fca pair's exports at 0 … 5 are FcaEng.engL / engR, tampered pairs refused,
the fca decision exported as FcaEng.engFam, five other units' exports byte-identical with and
without the family export."""

from __future__ import annotations

import os

import pytest

from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel, lean_adder
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_net import check, lean_project
from tests.test_adder_boxed import DRAPER, DRAPER_SPEC, FCA, FCA_SPEC, _flipped


def _fca(dim: str = "dim d = 2", boxed: str = FCA) -> tuple[Diagram, Diagram]:
    return parse_dsl(dim + boxed), parse_dsl(dim + FCA_SPEC)


def test_the_fca_pair_matches_the_lean_networks() -> None:
    for dim in ("dim d = 2", "dim d"):
        left, right = _fca(dim)
        source = lean_adder.export_family(left, right, ("n",), "fca")
        assert source is not None
        assert "FcaEng.engFam d hd" in source and source.count("by decide +kernel") == 12


def test_tampered_pairs_are_refused() -> None:
    left, right = _fca()
    tampers = [
        _flipped(FCA),
        FCA.replace("phase{k: 2*d^(j - n)*k}", "phase{k: 3*d^(j - n)*k}", 1),
        FCA.replace("g6.out0 -- g7.in0", "g6.out0 -- g8.in0", 1).replace(
            "g7.out0 -- g8.in0", "g7.out0 -- g7.in0", 1
        ),
    ]
    for text in tampers:
        assert text != FCA
        bad, _ = _fca(boxed=text)
        assert lean_adder.export_family(bad, right, ("n",), "fca") is None
    assert lean_adder.export_family(right, left, ("n",), "fca") is None
    assert lean_adder.export_family(left, right, (), "fca") is None
    assert lean_adder.export_family(left, right, ("n", "m"), "fca") is None
    draper = parse_dsl("dim d = 2" + DRAPER), parse_dsl("dim d = 2" + DRAPER_SPEC)
    assert lean_adder.export_family(*draper, ("n",), "draper") is None


def _decided(left: Diagram, right: Diagram) -> kernel.Theorem:
    decision = decide_equal(left, right, samples=[])
    assert decision.verdict.name == "EQUAL", decision.reason
    return kernel.from_decision(decision, left, right)


@pytest.mark.slow
def test_the_fca_decision_exports_the_family_theorem() -> None:
    source = kernel.lean(_decided(*_fca()), "fca")
    assert "\naxiom " not in source
    assert "theorem fca : ∀ (d : ℕ), 0 < d → ∀ (n : ℕ), Equal (FcaEng.engE d n)" in source


def _units() -> list[kernel.Theorem]:
    from tests.test_first_peel import _FCA, _PAIRS, _pair_repeel
    from tests.test_nested_chain import _link_identity_pair

    base, identity, _ = _link_identity_pair(4)
    return [
        _pair_repeel(_PAIRS),
        _pair_repeel(_FCA),
        _decided(parse_dsl(base), parse_dsl(identity)),
        _decided(parse_dsl("dim d = 2" + DRAPER), parse_dsl("dim d = 2" + DRAPER_SPEC)),
        _decided(*reversed(_fca())),
    ]


@pytest.mark.slow
def test_other_units_export_byte_identically() -> None:
    units = _units()
    with_family = [kernel.lean(t, "main") for t in units]
    saved = lean_adder.export_family
    lean_adder.export_family = lambda *_a, **_k: None
    try:
        without = [kernel.lean(t, "main") for t in units]
    finally:
        lean_adder.export_family = saved
    assert with_family == without
    assert all("FcaEng.engFam" not in s for s in with_family)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_family_export(tmp_path: object) -> None:
    left, right = _fca()
    source = lean_adder.export_family(left, right, ("n",), "fca")
    assert source is not None
    ok, output = check(source + "\n#print axioms ArchytasZX.Export.Fca.fca\n", f"{tmp_path}/f.lean")
    assert ok and ": error" not in output and "sorry" not in output, output
    assert "[propext, Classical.choice, Quot.sound]" in output, output
    bad = source.replace(
        "def litL3 : Net := ⟨([⟨.F, [7], [6]", "def litL3 : Net := ⟨([⟨.F, [6], [7]", 1
    )
    assert bad != source
    ok, _ = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
