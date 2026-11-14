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

"""[Dyadic] with given binary digits (Step.dyadicB, bitsOK over the truncated lift): the
kernel path_sum export of carry_shift at p = 3."""

from __future__ import annotations

import os

import pytest
import sympy as sp  # type: ignore[import-untyped]

from archytaszx.semantics import kernel
from archytaszx.semantics import lean_net as L
from archytaszx.semantics import lemmas as LM

LOW_BITS = ".dyadicB [8, 9, 10, 11] "
"""The 4-digit [Dyadic] step of the carry_shift circuit side."""


def _source() -> str:
    theorem = LM.instance(LM.CARRY_SHIFT, sp.Integer(3))
    assert isinstance(theorem, kernel.Theorem)
    return kernel.lean(theorem, "p3")


def test_the_dyadic_step_carries_its_bits() -> None:
    source = _source()
    assert LOW_BITS + "[([], 15)" in source
    assert "[12, 13, 14, 15] [[[], [1], [5]], " in source


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_checks_the_carry_shift_path_sum_export(tmp_path: object) -> None:
    ok, output = L.check(_source(), f"{tmp_path}/p3.lean")
    assert ok, output


@pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")
def test_lean_rejects_a_tampered_digit(tmp_path: object) -> None:
    """The low binary digit 1 + x1 + x5 changed to 1 + x1 fails bitsOK."""
    source = _source()
    tampered = source.replace("[12, 13, 14, 15] [[[], [1], [5]], ", "[12, 13, 14, 15] [[[], [1]], ")
    assert tampered != source
    ok, _output = L.check(tampered, f"{tmp_path}/p3bad.lean")
    assert not ok
