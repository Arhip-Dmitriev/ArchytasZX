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

"""Lean refutations: the source states and proves that two diagrams are not equal."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from archytaszx.repl.parser import parse_diagram
from archytaszx.semantics import kernel as K
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.decide import EqualityVerdict, decide_equal
from archytaszx.semantics.lean_numeric import export_numeric_refutation
from archytaszx.semantics.lean_refute import export_refutation

CASES: dict[str, tuple[str, str]] = {
    "colour": ("a = Z(0 -> 1, d)", "a = X(0 -> 1, d)"),
    "phase": ("a = Z(0 -> 1, d) phase {1: theta}", "a = Z(0 -> 1, d)"),
    "scalar": ("a = Z(1 -> 1, d); scalar 2", "a = Z(1 -> 1, d)"),
    "sqrt": ("a = Z(1 -> 1, d); scalar sqrt(d)", "a = Z(1 -> 1, d)"),
    "w": ("w = W(1 -> 2, d)", "a = Z(1 -> 2, d)"),
    "fourier": ("a = F(d)", "a = Z(1 -> 1, d)"),
    "three": ("a = X(0 -> 1, 3)", "a = Z(0 -> 1, 3)"),
    "ports": (
        "count n; a = Z(0 -> 2, d); b = Z(1 -> 1, d); wire a.out0 -- b.in0; box p x n ports a.out1",
        "count n; a = X(0 -> 2, d); b = Z(1 -> 1, d); wire a.out0 -- b.in0; box p x n ports a.out1",
    ),
    "nodes": (
        "count n; a = Z(0 -> 1, d); box p x n nodes a",
        "count n; a = X(0 -> 1, d); box p x n nodes a",
    ),
    "arity": ("a = Z(0 -> 1, d)", "a = Z(0 -> 2, d)"),
}


def _source(name: str) -> str:
    left, right = (parse_diagram(t) for t in CASES[name])
    decision = decide_equal(left, right)
    assert decision.verdict is EqualityVerdict.UNEQUAL
    return export_refutation(left, right, decision.counterexample or {})


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


@pytest.mark.parametrize("name", [n for n in CASES if n != "arity"])
def test_states_and_evaluates_the_refutation(name: str) -> None:
    source = _source(name)
    assert "theorem main :\n    ¬ " in source
    assert "eval_cyc" in source and "cyc_ne" in source
    assert "\naxiom " not in source and "sorry" not in source


def test_parameters_are_universally_quantified() -> None:
    assert "¬ ∀ (d : ℕ) (h_d : 0 < d), Equal (E d) 1 N0 1 N1" in _source("colour")
    assert "have he := h 2 (by norm_num)" in _source("colour")


def test_arity_refutation_compares_boundaries() -> None:
    source = _source("arity")
    assert "refute_bnd (by decide)" in source
    assert "eval_cyc" not in source


def test_a_point_without_values_falls_back_to_a_sample_point() -> None:
    left, right = (parse_diagram(t) for t in CASES["colour"])
    assert "refutation at d = 2" in export_refutation(left, right, {})
    assert "refutation at d = 5" in export_refutation(left, right, {"d": 5})


def test_agreeing_diagrams_have_no_refutation() -> None:
    left = parse_diagram("a = Z(1 -> 1, d)")
    with pytest.raises(L.NetUnsupported, match="agree"):
        export_refutation(left, left.copy(), {"d": 2})


def test_kernel_reads_a_counterexample() -> None:
    left, right = (parse_diagram(t) for t in CASES["colour"])
    ref = K.refutation(decide_equal(left, right), left, right)
    assert ref.point == {"d": 2}
    assert "cyc_ne" in K.lean(ref)
    with pytest.raises(K.KernelError, match="no counterexample"):
        K.refutation(decide_equal(left, left.copy()), left, left)


needs_lean = pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")


@pytest.mark.slow
@needs_lean
@pytest.mark.parametrize("name", list(CASES))
def test_lean_checks_the_refutation(name: str, tmp_path: Path) -> None:
    source = _source(name)
    passed, message = L.check(source, str(tmp_path / f"{name}.lean"))
    assert passed, message


def _chain(n: int, d: int, phase: str) -> str:
    parts = [f"a{i} = X(1 -> 1, literal {d}) phase {{1: {phase}}}" for i in range(n)]
    parts += [f"wire a{i}.out0 -- a{i + 1}.in0" for i in range(n - 1)]
    return "; ".join(parts)


NUMERIC: dict[str, tuple[str, str]] = {
    "irrational_phase": ("a = Z(0 -> 1, literal 2) phase {1: sqrt(2)}", "a = Z(0 -> 1, literal 2)"),
    "pi_scalar": ("a = Z(1 -> 1, literal 2); scalar pi", "a = Z(1 -> 1, literal 2)"),
}

REFUTATIONS: dict[str, tuple[str, str]] = {
    "contracted": (_chain(9, 3, "1/3"), "a = X(1 -> 1, literal 3) phase {2: 1/3}"),
    "dimensions": ("a = Z(0 -> 1, d)", "a = Z(0 -> 1, e)"),
    "sqrt_five": ("a = Z(1 -> 1, literal 5); scalar sqrt(5)", "a = Z(1 -> 1, literal 5)"),
    "boxed_ports": (
        "count n; a = Z(0 -> 2, d); box p x n ports a.out0",
        "count n; a = X(0 -> 2, d); box p x n ports a.out1",
    ),
}


def _numeric(name: str) -> str:
    left, right = (parse_diagram(t) for t in NUMERIC[name])
    return export_numeric_refutation(left, right, decide_equal(left, right).counterexample or {})


def _refutation(name: str) -> str:
    left, right = (parse_diagram(t) for t in REFUTATIONS[name])
    return export_refutation(left, right, decide_equal(left, right).counterexample or {})


def test_a_large_network_is_contracted() -> None:
    assert "eval_ve " in _refutation("contracted")


def test_square_roots_are_gauss_sums() -> None:
    assert "sqrt_of_sq" in _refutation("sqrt_five")


@pytest.mark.parametrize("name", list(NUMERIC))
def test_irrational_values_use_balls(name: str) -> None:
    source = _numeric(name)
    assert "ne_zero_of_near" in source and "ball arithmetic" in source


def test_pi_is_stated_not_quantified() -> None:
    assert "¬ Equal E (Real.pi : ℂ) N0 1 N0" in _numeric("pi_scalar")


def test_interfaces_compare_leg_dimensions() -> None:
    source = _refutation("dimensions")
    assert "List.replicate 1 d" in source and "List.replicate 1 e" in source


def test_families_with_different_boundaries_are_instantiated() -> None:
    assert "the families instantiated at n = " in _refutation("boxed_ports")


@pytest.mark.slow
@needs_lean
@pytest.mark.parametrize("name", list(NUMERIC))
def test_lean_checks_the_ball_refutation(name: str, tmp_path: Path) -> None:
    passed, message = L.check(_numeric(name), str(tmp_path / f"{name}.lean"))
    assert passed, message


@pytest.mark.slow
@needs_lean
@pytest.mark.parametrize("name", list(REFUTATIONS))
def test_lean_checks_the_other_refutations(name: str, tmp_path: Path) -> None:
    passed, message = L.check(_refutation(name), str(tmp_path / f"{name}.lean"))
    assert passed, message
