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

"""The qudit path-sum Lean export over Z_p: step lists, Gauss sums in Legendre form, the network
export with its pin steps, Lean."""

from __future__ import annotations

import os

import pytest

from archytaszx.algebra.pathsum import PathSumUnsupported
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import lean_net as L
from archytaszx.semantics.lean_pathsum import (
    export_net,
    export_proof,
    export_qudit_net,
    export_qudit_proof,
)
from archytaszx.semantics.lean_scalar import ScalarUnsupported

_D = "literal 3"

_CUP5 = "b = Z(1 -> 0, literal 5); wire a.out0 -- b.in0"

PAIRS: dict[str, tuple[str, str]] = {
    "gauss": (
        "; ".join(
            [
                f"a = F({_D}); b = Z(1 -> 1, {_D}) phase {{1: 1/3, 2: 1/3}}; c = F({_D})",
                "wire a.out0 -- b.in0; wire b.out0 -- c.in0",
            ]
        ),
        "; ".join(
            [
                f"a = Z(1 -> 1, {_D}) phase {{1: 2/3, 2: 2/3}}; b = F({_D})",
                f"c = Z(1 -> 1, {_D}) phase {{1: 2/3, 2: 2/3}}",
                "wire a.out0 -- b.in0; wire b.out0 -- c.in0; scalar I",
            ]
        ),
    ),
    "f4": (
        "; ".join(
            [
                f"a = F({_D}); b = F({_D}); c = F({_D}); e = F({_D})",
                "wire a.out0 -- b.in0; wire b.out0 -- c.in0; wire c.out0 -- e.in0",
            ]
        ),
        f"a = Z(1 -> 1, {_D})",
    ),
    "xx": (
        f"a = X(1 -> 2, {_D}); b = X(2 -> 1, {_D}); wire a.out0 -- b.in0; wire a.out1 -- b.in1",
        f"a = Z(1 -> 1, {_D})",
    ),
    "g5a2": (
        f"a = Z(0 -> 1, literal 5) phase {{1: 2/5, 2: 3/5, 3: 3/5, 4: 2/5}}; {_CUP5}",
        f"a = Z(0 -> 1, literal 5); {_CUP5}; scalar -sqrt(5)/5",
    ),
}


def _gauss_pair(p: int, a: int, scalar: str) -> tuple[str, str]:
    """``Σ_k ζ_p^(a k^2)`` as a quadratic phase state on a cup, against the cup times ``scalar``."""
    phases = ", ".join(f"{k}: {(a * k * k) % p}/{p}" for k in range(1, p))
    cup = f"b = Z(1 -> 0, literal {p}); wire a.out0 -- b.in0"
    return (
        f"a = Z(0 -> 1, literal {p}) phase {{{phases}}}; {cup}",
        f"a = Z(0 -> 1, literal {p}); {cup}; scalar {scalar}",
    )


PAIRS |= {
    "g7": _gauss_pair(7, 1, "I*sqrt(7)/7"),
    "g7a3": _gauss_pair(7, 3, "-I*sqrt(7)/7"),
    "g13": _gauss_pair(13, 1, "sqrt(13)/13"),
    "g13a2": _gauss_pair(13, 2, "-sqrt(13)/13"),
}


def _export(name: str) -> str:
    left, right = PAIRS[name]
    return export_qudit_proof(parse_dsl(left), parse_dsl(right), name)


QUBIT_NETS: dict[str, tuple[str, str]] = {
    "legless": (
        "a = Z(0 -> 0, literal 2) phase {1: 1/4}; b = Z(1 -> 1, literal 2)",
        "b = Z(1 -> 1, literal 2); scalar 1 + I",
    ),
    "weighted": (
        "a = Z(1 -> 1, literal 4) weight{1: I, 2: -1, 3: -I}",
        "a = Z(1 -> 1, literal 4) phase {1: 1/4, 2: 1/2, 3: 3/4}",
    ),
    "eighths": (
        (
            "a = Z(1 -> 1, literal 2) phase {1: 1/8}; b = Z(1 -> 1, literal 2) phase {1: 1/8}; "
            "wire a.out0 -- b.in0"
        ),
        "a = Z(1 -> 1, literal 2) phase {1: 1/4}",
    ),
}
"""Qubit network pairs: a legless spider, a weighted spider, phase tables finer than the sum's."""


def _qubit_net(name: str) -> str:
    left, right = QUBIT_NETS[name]
    return export_net(parse_dsl(left), parse_dsl(right), name)


def test_a_legless_spider_takes_the_legless_x_fragment() -> None:
    assert "(val_z_legless E σ _ rfl rfl rfl).trans" in _qubit_net("legless")


def test_a_weighted_spider_scales_its_fragment() -> None:
    source = _qubit_net("weighted")
    assert "w_frag E σ" in source and "coef := Ψ" in source


def test_the_phase_modulus_covers_every_table() -> None:
    assert "/ ((8 : ℕ) : ℝ)" in _qubit_net("eighths")


def _export_net(name: str) -> str:
    left, right = PAIRS[name]
    return export_qudit_net(parse_dsl(left), parse_dsl(right), name)


def test_a_qudit_network_pins_its_summed_labels() -> None:
    source = _export_net("gauss")
    assert ".pin " in source and "theorem gauss : Equal E" in source
    assert "f_frag_q (p := 3)" in source and "z_frag_q (p := 3)" in source


def test_unequal_qudit_networks_are_refused() -> None:
    left, _ = PAIRS["gauss"]
    with pytest.raises(PathSumUnsupported):
        export_qudit_net(parse_dsl(left), parse_dsl(f"a = Z(1 -> 1, {_D})"))


def test_a_gauss_step_is_replayed_with_its_legendre_form() -> None:
    source = _export("gauss")
    assert "def stepsL : List QSem.Step := [.perm [2], .gauss 1, .norm]" in source
    assert "rw [qg_legendre (by norm_num) 1 (by decide)]" in source
    assert "theorem lg2 : lg 3 2 = -1 := by decide" in source


def test_pivots_and_constraints_become_steps() -> None:
    source = _export("f4")
    assert ".hhPivot" in source and ".hhCons" in source and ".canon 0" in source


def test_a_non_residue_gauss_sum_takes_its_sign() -> None:
    source = _export("g5a2")
    assert ".gauss 2" in source and "theorem lg2 : lg 5 2 = -1 := by decide" in source


def test_a_square_root_of_p_takes_the_gauss_sign() -> None:
    assert "gauss_rt 7 " in _export_net("g7") and "gauss_rt 13 " in _export_net("g13")
    left, _ = PAIRS["g7"]
    _, wrong = _gauss_pair(7, 1, "-I*sqrt(7)/7")
    with pytest.raises((PathSumUnsupported, ScalarUnsupported)):
        export_qudit_net(parse_dsl(left), parse_dsl(wrong))


def test_unequal_qudit_sums_are_refused() -> None:
    left, _ = PAIRS["gauss"]
    with pytest.raises(PathSumUnsupported):
        export_qudit_proof(parse_dsl(left), parse_dsl(f"a = Z(1 -> 1, {_D})"))


def test_qubit_sums_are_refused() -> None:
    wire = parse_dsl("a = Z(1 -> 1, literal 2)")
    with pytest.raises(PathSumUnsupported):
        export_qudit_proof(wire, wire)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


needs_lean = pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")


@needs_lean
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_lean_proves_a_qudit_pair(name: str, tmp_path: object) -> None:
    ok, output = L.check(_export(name), f"{tmp_path}/{name}.lean")
    assert ok, output


@needs_lean
def test_lean_rejects_a_tampered_qudit_export(tmp_path: object) -> None:
    source = _export("gauss")
    marker = "([(0, 2)], 2)"
    assert marker in source
    ok, _ = L.check(source.replace(marker, "([(0, 2)], 1)"), f"{tmp_path}/tampered.lean")
    assert not ok


@needs_lean
@pytest.mark.parametrize("name", sorted(PAIRS))
def test_lean_proves_a_qudit_network_pair(name: str, tmp_path: object) -> None:
    ok, output = L.check(_export_net(name), f"{tmp_path}/net_{name}.lean")
    assert ok, output


@needs_lean
@pytest.mark.parametrize("name", sorted(QUBIT_NETS))
def test_lean_proves_a_qubit_network_pair(name: str, tmp_path: object) -> None:
    ok, output = L.check(_qubit_net(name), f"{tmp_path}/qnet_{name}.lean")
    assert ok, output


_Q = "literal 2"

AMP_PAIRS: dict[str, tuple[str, str]] = {
    "average": (
        (
            f"a = Z(0 -> 1, {_Q}) weight{{1: sqrt(2)}}; b = X(1 -> 1, {_Q}); c = Z(1 -> 0, {_Q}); "
            "wire a.out0 -- b.in0; wire b.out0 -- c.in0"
        ),
        "scalar 1 + sqrt(2)",
    ),
    "zero": (
        f"a = Z(1 -> 1, {_Q}) weight{{1: 0}}",
        f"a = X(0 -> 1, {_Q}); b = X(1 -> 0, {_Q}); scalar 1/2",
    ),
    "magnitudes": (
        (
            f"a = Z(1 -> 1, {_Q}) weight{{1: sqrt(2)}}; b = Z(1 -> 1, {_Q}) phase {{1: 1/2}}; "
            "wire a.out0 -- b.in0"
        ),
        f"a = Z(1 -> 1, {_Q}) weight{{1: -sqrt(2)}}",
    ),
    "held_parent": (
        (
            f"a = Z(0 -> 1, {_Q}); c = Z(1 -> 1, {_Q}); b = Z(1 -> 1, {_Q}) weight{{1: sqrt(2)}}; "
            f"d = Z(1 -> 0, {_Q}); wire a.out0 -- c.in0; wire c.out0 -- b.in0; wire b.out0 -- d.in0"
        ),
        "scalar 1 + sqrt(2)",
    ),
    "x_root2": (
        f"a = X(0 -> 1, {_Q}) weight{{1: sqrt(2)}}; b = X(1 -> 0, {_Q}); wire a.out0 -- b.in0",
        "scalar 1 + sqrt(2)",
    ),
    "x_zero": (
        f"a = X(0 -> 1, {_Q}) weight{{1: 0}}; b = X(1 -> 0, {_Q}); wire a.out0 -- b.in0",
        "scalar 1",
    ),
    "x_legless": (f"a = X(0 -> 0, {_Q}) weight{{1: sqrt(2)}}", "scalar 1 + sqrt(2)"),
    "z_legless": (f"a = Z(0 -> 0, {_Q}) weight{{1: sqrt(2)}}", "scalar 1 + sqrt(2)"),
    "z_legless_zero": (f"a = Z(0 -> 0, {_Q}) weight{{1: 0}}", "scalar 1"),
    "fourier": (
        (
            f"a = Z(0 -> 1, {_Q}) weight{{1: sqrt(2)}}; f = F(1 -> 1, {_Q}); "
            f"c = X(1 -> 0, {_Q}); wire a.out0 -- f.in0; wire f.out0 -- c.in0"
        ),
        "scalar 1 + sqrt(2)",
    ),
    "x_into_z": (
        f"a = X(0 -> 1, {_Q}) weight{{1: sqrt(2)}}; b = Z(1 -> 0, {_Q}); wire a.out0 -- b.in0",
        "scalar sqrt(2)",
    ),
    "x_open": (
        f"a = X(0 -> 1, {_Q}) weight{{1: sqrt(2)}}; b = Z(1 -> 1, {_Q}); wire a.out0 -- b.in0",
        f"a = X(0 -> 1, {_Q}) weight{{1: sqrt(2)}}",
    ),
    "x_square": (
        (
            f"a = X(1 -> 1, {_Q}) weight{{1: sqrt(2)}}; b = X(1 -> 1, {_Q}) weight{{1: sqrt(2)}}; "
            "wire a.out0 -- b.in0"
        ),
        (
            f"a = X(1 -> 1, {_Q}) weight{{1: -sqrt(2)}}; "
            f"b = X(1 -> 1, {_Q}) weight{{1: -sqrt(2)}}; wire a.out0 -- b.in0"
        ),
    ),
}
"""Pairs with amplitude atoms: an [Average] after an hh_pivot, a zero weight, weights 1 and
sqrt 2 with a phase, a pinned anchor whose parent is held, atoms on a weighted X spider's and a
legless weighted spider's own digits, an F node beside atoms, a weighted X state into a Z effect
(pin), a weighted X state on an open leg and two fused weighted X spiders ([Average] through a
half phase)."""


def test_weighted_x_legs_reduce_through_pin_and_half_average() -> None:
    assert ".asubB 1 []" in _amp("x_into_z", False)
    assert ".averageH " in _amp("x_open", False)
    assert ".averageH " in _amp("x_square", True)


def _amp(name: str, net: bool) -> str:
    left, right = AMP_PAIRS[name]
    return (export_net if net else export_proof)(parse_dsl(left), parse_dsl(right), name)


def test_an_average_trace_becomes_an_average_step() -> None:
    assert ".average 0 [[]] 1 [] [0]" in _amp("average", False)
    assert "NetPSDA E NL" in _amp("average", True)


def test_a_pinned_atom_digit_holds_its_parent() -> None:
    source = _amp("held_parent", True)
    assert ".hold" in source and ".asubB " in source


@pytest.mark.parametrize(
    "source",
    [
        f"a = X(1 -> 1, {_Q}) weight{{1: sqrt(2)}}",
        f"a = X(1 -> 2, {_Q}) weight{{1: -sqrt(2)/2}}",
        f"a = X(2 -> 1, {_Q}) weight{{1: 0}}",
        f"a = X(0 -> 0, {_Q}) weight{{1: sqrt(2)}}",
        "a = X(1 -> 1, literal 4) weight{1: sqrt(2), 2: 0, 3: -1}",
    ],
)
def test_a_weighted_x_path_sum_is_its_tensor(source: str) -> None:
    import itertools

    import numpy as np

    from archytaszx.algebra.pathsum import PathSum, entry_enumerated
    from archytaszx.semantics.contract_numeric import contract
    from archytaszx.semantics.pathsum_diagram import from_diagram

    diagram = parse_dsl(source)
    ps = from_diagram(diagram)
    assert isinstance(ps, PathSum)
    tensor = np.asarray(contract(diagram).tensor)
    for bits in itertools.product((0, 1), repeat=len(ps.boundary)):
        entry = entry_enumerated(ps, dict(zip(ps.boundary, bits, strict=True)), 1 << 12)
        assert entry is not None
        index, pos = [], 0
        for dim in tensor.shape:
            k = dim.bit_length() - 1
            index.append(sum(bits[pos + j] << j for j in range(k)))
            pos += k
        want = complex(tensor[tuple(index)]) if tensor.ndim else complex(tensor)
        assert abs(entry.to_complex() - want) < 1e-9


def test_unequal_atoms_are_refused() -> None:
    left, _ = AMP_PAIRS["magnitudes"]
    with pytest.raises(PathSumUnsupported):
        export_net(parse_dsl(left), parse_dsl(f"a = Z(1 -> 1, {_Q}) weight{{1: sqrt(2)}}"))


@needs_lean
@pytest.mark.parametrize("name", sorted(AMP_PAIRS))
@pytest.mark.parametrize("net", [False, True])
def test_lean_proves_a_pair_with_atoms(name: str, net: bool, tmp_path: object) -> None:
    ok, output = L.check(_amp(name, net), f"{tmp_path}/amp_{name}_{net}.lean")
    assert ok, output


@needs_lean
def test_lean_rejects_a_tampered_atom_table(tmp_path: object) -> None:
    source = _amp("held_parent", True)
    marker = "(([[], [0]] : List (List ℕ))"
    assert marker in source
    ok, _ = L.check(source.replace(marker, "(([[], []] : List (List ℕ))"), f"{tmp_path}/t.lean")
    assert not ok
