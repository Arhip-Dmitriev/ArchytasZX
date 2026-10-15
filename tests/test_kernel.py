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

import re
from dataclasses import replace

import pytest

from archytaszx.repl.commands import Session, run
from archytaszx.semantics import kernel as K
from archytaszx.semantics.certificate import Certificate, Derivation, DerivationKind
from archytaszx.semantics.prove import prove

from .test_phase15_oracle import boxed_state_copy
from .test_prove import _certificate, _two_sided


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


def _declared(source: str) -> set[str]:
    return set(re.findall(r"^(?:axiom|theorem) (\w+)", source, re.MULTILINE))


@pytest.mark.parametrize("make", [_certificate, _two_sided])
def test_lean_chain(make: object) -> None:
    source = K.lean(K.check(make()))  # type: ignore[operator]
    assert source.startswith("/-") and "namespace ArchytasZX" in source
    assert "axiom Diagram : Type" in source and "theorem main :" in source
    used = set(re.findall(r"\b((?:ax|iso|d)\d+)\b", source))
    assert used <= _declared(source)


def test_lean_induction() -> None:
    induction = prove(*boxed_state_copy()).induction
    assert induction is not None
    source = K.lean(K.check(induction), name="copy")
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
