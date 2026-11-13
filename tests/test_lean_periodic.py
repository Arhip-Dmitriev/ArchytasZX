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

"""The periodic Lean export: a chain side as a famNet family, the other side fixed."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel as K
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.lean_periodic import export_periodic

S_CHAIN = """count n
a = Z(0 -> 1, literal 2)
repeat n as V: x = Z(1 -> 1, literal 2) phase{1: 1/4}
b = X(1 -> 0, literal 2) phase{1: 1/2}
wire a.out0 -- x.in0, x.out0 -- b.in0
chain V: x.out0 -> x.in0"""
"""<1| S^n |+> by a chain of n S spiders."""

PHASED = """count n
a = Z(0 -> 1, literal 2)
b = X(1 -> 0, literal 2) phase{1: 1/2}
wire a.out0 -- b.in0
scalar I^n"""
"""i^n <1|+>."""

FLIPS = """count n
a = X(0 -> 1, literal 2)
repeat n as V: x = X(1 -> 1, literal 2) phase{1: 1/2}
b = X(1 -> 0, literal 2)
wire a.out0 -- x.in0, x.out0 -- b.in0
chain V: x.out0 -> x.in0"""
"""<0| X^n |0> by a chain of n X(pi) spiders."""

PARITY = """count n
a = X(0 -> 1, literal 2)
b = X(1 -> 0, literal 2)
wire a.out0 -- b.in0
scalar (1 + (-1)^n)/2"""
"""<0|0> times the parity of n."""


def _theorem(left: str, right: str) -> K.Theorem:
    decision = decide_equal(parse_dsl(left), parse_dsl(right), max_samples=0)
    proof = decision.induction_proof
    assert proof is not None and proof.inference.kind == "periodic", decision.reason
    return proof


def test_the_s_chain_export_states_the_family_at_every_index() -> None:
    """S^4 is the identity: period 4, the instances at 2 and 3 linked."""
    source = export_periodic(_theorem(S_CHAIN, PHASED), name="s_chain", checks=(2, 3))
    assert "theorem main : ∀ k, Equal E (perS sT A 4 k) (famNet W u tl cp hd bd k)" in source
    assert "theorem h0 : Equal E 1 (blkNet W u cp hd 4) c (headNet u hd)" in source
    assert "namespace At3" in source


def test_a_pair_zero_at_one_index_is_exported() -> None:
    """The flips pair at n = 1: the chain side reaches a constraint 1 = 0, the other side has
    scalar zero."""
    source = export_periodic(_theorem(FLIPS, PARITY), name="flips", checks=(2, 3))
    assert "theorem baseEq : Equal E (1) NL ((0 : ℂ)) NR := by" in source
    assert "PSem.val_of_zeroOK 4 _ (run 4 stepsL" in source


def test_a_theorem_of_another_kind_is_refused() -> None:
    wire = parse_dsl("a = Z(1 -> 1, literal 2)\ninputs a.in0\noutputs a.out0")
    with pytest.raises(PathSumUnsupported):
        export_periodic(K.refl(wire))


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_s_chain_family(tmp_path: object) -> None:
    source = export_periodic(_theorem(S_CHAIN, PHASED), name="s_chain", checks=(2, 3))
    ok, output = L.check(source, f"{tmp_path}/s_chain.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_proves_the_flips_family(tmp_path: object) -> None:
    source = export_periodic(_theorem(FLIPS, PARITY), name="flips", checks=(2, 3))
    ok, output = L.check(source, f"{tmp_path}/flips.lean")
    assert ok, output
