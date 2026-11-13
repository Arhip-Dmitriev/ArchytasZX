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

"""The proof kernel: theorems from certificates, refusals, and the Lean 4 export."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import replace

import pytest

from archytaszx.diagram.generators import X_SPIDER
from archytaszx.repl.commands import Session, run
from archytaszx.semantics import kernel as K
from archytaszx.semantics.certificate import Certificate, Derivation, DerivationKind
from archytaszx.semantics.lean_net import check, lean_project
from archytaszx.semantics.prove import prove

from .test_phase15_oracle import boxed_state_copy
from .test_prove import _certificate, _two_sided
from .test_rules_library_phase11 import identity_chain


def test_theorem_needs_the_kernel() -> None:
    d = _certificate().start
    with pytest.raises(K.KernelError):
        K.Theorem(object(), d, d, K.Inference("refl"))


def test_step_sequence_and_two_sided_proofs() -> None:
    one = K.check(_certificate())
    assert one.axioms() == ["spider_fusion", "spider_fusion"]
    two = K.check(_two_sided())
    assert K.check(_two_sided()).size() == two.size()
    assert "identity_removal" in two.axioms()


def test_induction_proof() -> None:
    induction = prove(*boxed_state_copy()).induction
    assert induction is not None
    theorem = K.check(induction)
    assert theorem.inference.kind == "induction" and theorem.variables == ("n",)
    with pytest.raises(K.KernelError):
        K.check(replace(induction, region=frozenset()))
    with pytest.raises(K.KernelError):
        K.check(replace(induction, base=induction.step))


def test_trans_needs_equal_middles() -> None:
    a = K.check(_certificate())
    with pytest.raises(K.KernelError, match="middle"):
        K.trans(a, a)
    assert K.trans(a, K.symm(a)).rhs is a.lhs


def test_induction_certificates_are_refused() -> None:
    d = _certificate().start
    cert = Certificate(Derivation(DerivationKind.STEP_SEQUENCE, d, d))
    assert K.check(cert).inference.kind == "refl"
    with pytest.raises(K.KernelError):
        K.check("proof")


def test_contraction_checks_the_symbolic_contractions() -> None:
    """Equal sides give a contraction theorem with an opaque axiom; unequal ones are refused."""
    chained = K.check(_certificate())
    theorem = K.contraction(chained.lhs, chained.rhs)
    assert theorem.inference.kind == "contraction"
    assert "axiom contraction1 :" in K.lean(theorem, semantic=False)
    with pytest.raises(K.KernelError, match="contractions differ"):
        K.contraction(chained.lhs, K.check(_two_sided()).lhs)


def test_a_doubled_decision_gives_the_doubled_theorem() -> None:
    """A DOUBLED decision proves its doubled pair, from the originals or the pair itself."""
    import archytaszx.diagram.mixed as M
    from archytaszx.semantics.decide import DecisionMethod, decide_equal

    left, right = M.compose(M.dephase(2), M.dephase(2)), M.dephase(2)
    decision = decide_equal(left, right)
    assert decision.method is DecisionMethod.DOUBLED and decision.doubled is not None
    for pair in ((left, right), decision.doubled):
        theorem = K.from_decision(decision, *pair)
        assert theorem.lhs is decision.doubled[0] and theorem.rhs is decision.doubled[1]
    with pytest.raises(K.KernelError, match="not the doubling"):
        K.from_decision(decision, right, left)


def _declared(source: str) -> set[str]:
    return set(re.findall(r"^(?:axiom|theorem) (\w+)", source, re.MULTILINE))


@pytest.mark.parametrize("make", [_certificate, _two_sided])
def test_lean_chain_opaque(make: object) -> None:
    source = K.lean(K.check(make()), semantic=False)  # type: ignore[operator]
    assert source.startswith("/-") and "namespace ArchytasZX" in source
    assert "axiom Diagram : Type" in source and "theorem main :" in source
    used = set(re.findall(r"\b((?:ax|iso|d)\d+)\b", source))
    assert used <= _declared(source)


@pytest.mark.parametrize("make", [_certificate, _two_sided])
def test_lean_chain_semantic(make: object) -> None:
    source = K.lean(K.check(make()))  # type: ignore[operator]
    assert "import ArchytasZX" in source and "\naxiom " not in source
    assert re.search(r"^theorem main .*:\n    Equal ", source, re.MULTILINE)


def test_lean_induction() -> None:
    induction = prove(*boxed_state_copy()).induction
    assert induction is not None
    source = K.lean(K.check(induction), name="copy", semantic=False)
    assert "theorem copy : ∀ (n : Nat)," in source
    assert "induction n with" in source and "| succ k ih =>" in source
    assert "congrArg" in source
    used = set(re.findall(r"\b((?:ax|iso|d|inst|peel|plug|hole)\d+)\b", source))
    assert used <= _declared(source)


def test_repl_kernel_and_lean(tmp_path: object) -> None:
    left, right = boxed_state_copy()
    s = Session()
    s.names.update(l=left, r=right)
    assert run(s, "load @l").ok and run(s, "prove against @r").ok
    checked = run(s, "certificate kernel")
    assert checked.ok and isinstance(checked.detail, K.Theorem)
    lines = [b.text for b in checked.blocks]  # type: ignore[union-attr]
    assert any(line.startswith("kernel: theorem") and "for all n" in line for line in lines)
    target = f"{tmp_path}/proof.lean"
    wrote = run(s, f"certificate lean {target}")
    assert wrote.ok
    with open(target, encoding="utf-8") as handle:
        assert "induction n with" in handle.read()


def _lean() -> str | None:
    found = shutil.which("lean")
    home = os.path.expanduser("~/.elan/bin/lean")
    return found or (home if os.path.exists(home) else None)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(_lean() is None, reason="Lean 4 is not installed")
def test_lean_accepts_the_opaque_induction(tmp_path: object) -> None:
    source = K.lean(K.check(prove(*boxed_state_copy()).induction), semantic=False)
    assert "axiom Diagram : Type" in source
    path = f"{tmp_path}/induction.lean"
    lean = _lean()
    assert lean is not None
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    done = subprocess.run([lean, path], capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    bad = source.replace("ih).trans", "ih.symm).trans")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(bad)
    assert (
        subprocess.run([lean, path], capture_output=True, timeout=300, check=False).returncode != 0
    )


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
@pytest.mark.parametrize("which", ["chain", "two_sided"])
def test_lean_proves_the_semantic_export(which: str, tmp_path: object) -> None:
    artefact = _certificate() if which == "chain" else _two_sided()
    source = K.lean(K.check(artefact)).replace(
        "end ArchytasZX.Export", "#print axioms main\nend ArchytasZX.Export"
    )
    ok, output = check(source, f"{tmp_path}/{which}.lean")
    assert ok, output
    assert "depends on axioms: [propext, Classical.choice, Quot.sound]" in " ".join(output.split())


def test_repl_reports_the_semantic_export() -> None:
    s = Session()
    s.names.update(l=identity_chain(3), r=identity_chain(3, X_SPIDER))
    assert run(s, "load @l").ok and run(s, "prove against @r").ok
    lines = [b.text for b in run(s, "certificate lean").blocks]  # type: ignore[union-attr]
    assert "lean: tensor-network semantics, every step proved" in lines


def test_a_kept_induction_pair_is_not_decided_again() -> None:
    from archytaszx.repl.parser import parse_dsl

    x = parse_dsl("a = Z(1 -> 1, literal 2)")
    y = parse_dsl("a = Z(1 -> 1, literal 2) phase {1: 1/2}")
    K._remember_induction(None, x, y)
    assert K._recalled_induction(x, y) == (True, None)
    assert K._induction_rung(x, y) is None
