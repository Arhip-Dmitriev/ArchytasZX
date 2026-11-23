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

"""Claims with finitely many values: the case split over the instance proofs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

_L = "dim d = 2\nrepeat n as V: e = Z(1 -> 1, d)\ninputs e.in0\noutputs e.out0\n"
_R = "dim d = 2\nrepeat n as V: e = X(1 -> 1, d)\ninputs e.in0\noutputs e.out0\n"


def _finite() -> kernel.FiniteProof:
    decision = decide_equal(parse_dsl(_L), parse_dsl(_R), restrict="1 <= n <= 3")
    assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
    assert isinstance(decision.proof, kernel.FiniteProof)
    return decision.proof


def test_a_bounded_claim_is_the_case_split_of_its_instances() -> None:
    source = kernel.finite_symbolic(_finite()).source
    assert "theorem main_finite : ∀ n : ℕ, 1 ≤ n → n ≤ 3 → " in source
    assert "  rcases h with rfl | rfl | rfl\n" in source
    assert "namespace ArchytasZX.Export.I3" in source


def test_a_tampered_case_split_is_rejected() -> None:
    proof = _finite()
    dropped = replace(proof, instances=proof.instances[:2])
    first, second, third = proof.instances
    swapped = replace(proof, instances=(first, (second[0], third[1]), third))
    for bad in (dropped, swapped):
        with pytest.raises(kernel.KernelError):
            kernel.finite_symbolic(bad)
