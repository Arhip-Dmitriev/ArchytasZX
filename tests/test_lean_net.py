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

"""The tensor-network Lean export: one rewrite step per rule, checked by Lean."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import sympy as sp

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.diagram.generators import (
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import RULES, lookup_rule
from archytaszx.semantics import kernel as K
from archytaszx.semantics import lean_net as L

from .test_incremental_match import fourier_chain
from .test_rules_library_phase11 import (
    bialgebra_diagram,
    fourier_state_diagram,
    hopf_diagram,
    identity_chain,
    state_copy_diagram,
    triangle_chain,
)
from .test_rules_library_swapped import cap_diagram, swap_colours
from .test_rules_zxw import (
    bind_then_split,
    connective_ends,
    split_then_bind,
    state_into,
    triangle_into_effect,
    w_after_z,
    w_into_effect,
    w_into_w,
    z_into_w,
)

CASES: dict[str, Callable[[], Diagram]] = {
    "identity_removal": lambda: identity_chain(3),
    "identity_removal/X": lambda: identity_chain(3, X_SPIDER),
    "triangle_inverse_cancellation": lambda: triangle_chain(3),
    "triangle_inverse_cancellation/Ti": lambda: triangle_chain(3, inverse_first=True),
    "state_copy": lambda: state_copy_diagram(3, 2),
    "state_copy/0": lambda: state_copy_diagram(2, 0),
    "state_copy_swapped": lambda: swap_colours(state_copy_diagram(3, 3)),
    "hopf": lambda: hopf_diagram(3, 1, 1, 1, 1, phases=True),
    "hopf_swapped": lambda: swap_colours(hopf_diagram(3, 1, 1, 1, 1, phases=True)),
    "bialgebra": lambda: bialgebra_diagram(2),
    "bialgebra_swapped": lambda: swap_colours(bialgebra_diagram(2)),
    "fourier_state_color_change": lambda: fourier_state_diagram(3, is_state=True),
    "fourier_state_color_change/effect": lambda: fourier_state_diagram(3, is_state=False),
    "fourier_state_color_change_swapped": lambda: swap_colours(
        fourier_state_diagram(3, is_state=True)
    ),
    "fourier_state_color_change_swapped/effect": lambda: swap_colours(
        fourier_state_diagram(3, is_state=False)
    ),
    "zx_cap": lambda: cap_diagram(3),
    "zx_cap_swapped": lambda: swap_colours(cap_diagram(3)),
    "fourier_cancellation": lambda: fourier_chain(3, 4),
    "w_fusion": w_into_w,
    "w_identity": w_after_z,
    "w_zero_copy": lambda: state_into(W_NODE, 2),
    "triangle_zero_state": lambda: state_into(TRIANGLE, 1),
    "triangle_zero_state/Ti": lambda: state_into(TRIANGLE_INVERSE, 1),
    "triangle_zero_effect": triangle_into_effect,
    "w_z_effect": lambda: w_into_effect(3, 1, Z_SPIDER),
    "w_zero_effect": lambda: w_into_effect(3, 1, X_SPIDER),
    "connective_states": lambda: connective_ends(Z_SPIDER, False),
    "connective_states/effects": lambda: connective_ends(Z_SPIDER, True),
    "connective_states_swapped": lambda: connective_ends(X_SPIDER, False),
    "connective_states_swapped/effects": lambda: connective_ends(X_SPIDER, True),
    "connective_inverse": split_then_bind,
    "connective_inverse_bind_first": bind_then_split,
}

UNPROVED = {"wz_bialgebra", "port_box_unfusion"}


def _one_step(name: str, diagram: Diagram) -> K.Theorem:
    rule = lookup_rule(name.split("/")[0])
    matches = rule.pattern.find_matches(diagram)
    assert matches, name
    return K.rule(diagram, apply(diagram, rule, matches[0]).step)


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


def _fusion_with_phases(dim: Dim, generator: object) -> Diagram:
    diagram = Diagram()
    concrete = dim.is_concrete
    first = PhaseVector(dim, {1: Phase.symbol("alpha")}) if concrete else None
    second = PhaseVector(dim, {1: Phase.symbol("beta"), 2: Phase.turns(sp.Rational(3, 4))})
    a = diagram.add_node(generator, [dim], [dim, dim], phase=first)  # type: ignore[arg-type]
    b = diagram.add_node(
        generator,
        [dim],
        [dim],
        phase=second if concrete else None,  # type: ignore[arg-type]
    )
    diagram.add_wire(PortRef(a, Direction.OUTPUT, 1), PortRef(b, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(a, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.OUTPUT, 0)])
    return diagram


def test_every_rule_has_a_lemma_or_is_listed() -> None:
    assert set(RULES) - set(L._RULES) == UNPROVED


@pytest.mark.parametrize("name", sorted(CASES))
def test_rule_step_exports_without_axioms(name: str) -> None:
    source = L.export(_one_step(name, CASES[name]()))
    assert "\naxiom " not in source and "import ArchytasZX" in source


def test_rule_without_a_lemma_is_an_unproved_axiom() -> None:
    source = L.export(_one_step("wz_bialgebra", z_into_w(2, 2)))
    assert "\naxiom unproved_wz_bialgebra_" in source
    assert "unproved steps: unproved_wz_bialgebra_" in source


def test_free_multiplicities_are_refused() -> None:
    from archytaszx.semantics.prove import prove

    from .test_phase15_oracle import boxed_state_copy

    induction = prove(*boxed_state_copy()).induction
    with pytest.raises(L.NetUnsupported):
        L.export(K.check(induction))
    assert "axiom Diagram : Type" in K.lean(K.check(induction))


def test_symbolic_parameters_become_binders() -> None:
    source = L.export(_one_step("spider_fusion", _fusion_with_phases(Dim.concrete(3), X_SPIDER)))
    assert "theorem main (alpha : ℝ) (beta : ℝ)" in source
    source = L.export(_one_step("spider_fusion", _fusion_with_phases(Dim.symbol("d"), Z_SPIDER)))
    assert "theorem main (d : ℕ) (h_d : 0 < d)" in source


needs_lean = pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")


@needs_lean
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_lean_proves_each_rule(name: str, tmp_path: object) -> None:
    source = L.export(_one_step(name, CASES[name]()))
    ok, output = L.check(source, f"{tmp_path}/step.lean")
    assert ok, output


@needs_lean
@pytest.mark.parametrize(
    ("dim", "generator"),
    [(Dim.concrete(3), Z_SPIDER), (Dim.concrete(3), X_SPIDER), (Dim.symbol("d"), X_SPIDER)],
)
def test_lean_proves_symbolic_fusion(dim: Dim, generator: object, tmp_path: object) -> None:
    source = L.export(_one_step("spider_fusion", _fusion_with_phases(dim, generator)))
    ok, output = L.check(source, f"{tmp_path}/fusion.lean")
    assert ok, output


@needs_lean
def test_lean_rejects_a_tampered_export(tmp_path: object) -> None:
    source = L.export(_one_step("zx_cap", cap_diagram(3)))
    marker = "Equal E 1 N0 (rt (E.dimv 0)) N1 :="
    assert marker in source
    tampered = source.replace(marker, "Equal E 1 N0 (2 * rt (E.dimv 0)) N1 :=")
    ok, _ = L.check(tampered, f"{tmp_path}/tampered.lean")
    assert not ok
