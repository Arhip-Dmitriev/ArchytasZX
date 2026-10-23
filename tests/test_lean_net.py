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
    "wz_bialgebra": lambda: z_into_w(2, 2),
    "wz_bialgebra/3x2": lambda: z_into_w(3, 2),
}

UNPROVED: set[str] = set()


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


def test_rule_without_a_lemma_is_an_unproved_axiom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(L._RULES, "wz_bialgebra")
    source = L.export(_one_step("wz_bialgebra", z_into_w(2, 2)))
    assert "\naxiom unproved_wz_bialgebra_" in source
    assert "unproved steps: unproved_wz_bialgebra_" in source


def _induction() -> K.Theorem:
    from archytaszx.semantics.prove import prove

    from .test_phase15_oracle import boxed_state_copy

    induction = prove(*boxed_state_copy()).induction
    assert induction is not None
    return K.check(induction)


def _family(name: str) -> K.Theorem:
    from archytaszx.semantics.prove import prove

    from .test_phase15_oracle import PROVED

    outcome = prove(*PROVED[name]())
    assert outcome.certificate is not None
    return K.check(outcome.certificate)


def test_an_induction_exports_without_axioms() -> None:
    source = L.export(_induction())
    assert "\naxiom " not in source
    assert "induction n with" in source and "Equal.of_split" in source
    assert "∀ n : ℕ, Equal ((F d).env (ksOf [n])) (sS d n) N0 (sG d n) N1" in source


def test_a_boxed_rewrite_is_lifted_to_every_multiplicity() -> None:
    source = L.export(_family("boxed_fusion"))
    assert "\naxiom " not in source
    assert "theorem bun1 (d : ℕ) (j : ℕ)" in source and "induction j with" in source
    assert "theorem main (d : ℕ) (m : ℕ) (h_d : 0 < d) :" in source


def test_deferred_dimensions_are_identified() -> None:
    theorem = _family("deferred_fusion")
    assert theorem.assumptions
    source = L.export(theorem)
    assert "\naxiom " not in source
    assert "dimensions assumed equal are one dimension here: d**q = d**p" in source


def test_nested_multiplicities_export_without_axioms() -> None:
    source = L.export(_family("two_index_fusion"))
    assert "\naxiom " not in source
    assert "fusionZG" in source and "ksOf [k2, m]" in source


def _port_family(generator: object, leg: int) -> K.Theorem:
    from archytaszx.diagram.bangbox import abstract_port_count
    from archytaszx.semantics.prove import prove

    from .helpers import build_ghz_with_copy
    from .test_phase8_oracle import _fuse_once

    diagram, _a, _b = build_ghz_with_copy(Dim("d"), generator_type=generator)
    diagram, _box, _n = abstract_port_count(diagram, diagram.boundary_outputs[leg], 1, stem="n")
    outcome = prove(diagram, _fuse_once(diagram))
    assert outcome.certificate is not None
    return K.check(outcome.certificate)


def _two_boxes() -> K.Theorem:
    from archytaszx.diagram.bangbox import abstract_subgraph_count
    from archytaszx.rewrite.rules_library import SPIDER_FUSION
    from archytaszx.semantics.prove import prove

    d = Dim("d")
    diagram = Diagram()
    outs, pairs = [], []
    for _ in range(2):
        a = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        outs += [PortRef(a, Direction.OUTPUT, 1), PortRef(b, Direction.OUTPUT, 0)]
        pairs.append(frozenset({a, b}))
    diagram.set_boundary_outputs(outs)
    diagram, _box, _m = abstract_subgraph_count(diagram, pairs[0], 1, stem="m")
    diagram, _box, _p = abstract_subgraph_count(diagram, pairs[1], 1, stem="p")
    goal = diagram
    for _ in range(2):
        goal = apply(goal, SPIDER_FUSION, SPIDER_FUSION.pattern.find_matches(goal)[0]).diagram
    outcome = prove(diagram, goal)
    assert outcome.certificate is not None
    return K.check(outcome.certificate)


def _boxed_rule(name: str) -> K.Theorem:
    from archytaszx.diagram.bangbox import abstract_subgraph_count

    diagram = CASES[name]()
    dim = next(iter(next(iter(diagram.nodes.values())).outputs)).dim
    extra = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    diagram.set_boundary_outputs([*diagram.boundary_outputs, PortRef(extra, Direction.OUTPUT, 0)])
    diagram, _box, _m = abstract_subgraph_count(diagram, frozenset(diagram.nodes), 1, stem="m")
    rule = lookup_rule(name.split("/")[0])
    match = next(m for m in rule.pattern.find_matches(diagram) if m.all_side_conditions_passed)
    return K.rule(diagram, apply(diagram, rule, match).step)


BOXED = [
    "fourier_cancellation",
    "fourier_state_color_change",
    "fourier_state_color_change/effect",
    "identity_removal",
    "identity_removal/X",
    "triangle_inverse_cancellation",
    "triangle_inverse_cancellation/Ti",
    "w_fusion",
    "w_identity",
]


@pytest.mark.parametrize(("generator", "leg"), [(Z_SPIDER, -1), (X_SPIDER, -1), (X_SPIDER, 0)])
def test_fusion_with_a_port_bundle_exports_without_axioms(generator: object, leg: int) -> None:
    assert "\naxiom " not in L.export(_port_family(generator, leg))


def test_two_boxes_export_without_axioms() -> None:
    source = L.export(_two_boxes())
    assert "\naxiom " not in source and "ksOf [m, p]" in source and "bunI" in source


@pytest.mark.parametrize("name", BOXED)
def test_a_rule_inside_a_box_exports_without_axioms(name: str) -> None:
    source = L.export(_boxed_rule(name))
    assert "\naxiom " not in source and "theorem bun" in source


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
    marker = "theorem main :\n    Equal E 1 N0 "
    assert marker in source
    tampered = source.replace(marker, "theorem main :\n    Equal E 2 N0 ")
    ok, _ = L.check(tampered, f"{tmp_path}/tampered.lean")
    assert not ok


@needs_lean
def test_lean_states_the_diagram_scalars(tmp_path: object) -> None:
    source = L.export(_one_step("state_copy", state_copy_diagram(3, 2)))
    assert "Equal E 1 N0 (((1 : ℂ) / 3) * (rt 3)) N1 :=" in source
    ok, output = L.check(source, f"{tmp_path}/copy.lean")
    assert ok, output


@needs_lean
def test_lean_proves_the_induction(tmp_path: object) -> None:
    source = L.export(_induction()).replace(
        "end ArchytasZX.Export", "#print axioms main\nend ArchytasZX.Export"
    )
    ok, output = L.check(source, f"{tmp_path}/induction.lean")
    assert ok, output
    assert "[propext, Classical.choice, Quot.sound]" in output


@needs_lean
@pytest.mark.slow
@pytest.mark.parametrize("name", ["boxed_fusion", "deferred_fusion"])
def test_lean_proves_a_family(name: str, tmp_path: object) -> None:
    source = L.export(_family(name))
    ok, output = L.check(source, f"{tmp_path}/{name}.lean")
    assert ok, output


@needs_lean
@pytest.mark.slow
@pytest.mark.parametrize("name", ["two_index_fusion", "port_z", "port_x", "two_boxes"])
def test_lean_proves_several_multiplicities(name: str, tmp_path: object) -> None:
    theorem = {
        "two_index_fusion": lambda: _family("two_index_fusion"),
        "port_z": lambda: _port_family(Z_SPIDER, -1),
        "port_x": lambda: _port_family(X_SPIDER, -1),
        "two_boxes": _two_boxes,
    }[name]()
    ok, output = L.check(L.export(theorem), f"{tmp_path}/{name}.lean")
    assert ok, output


@needs_lean
@pytest.mark.slow
@pytest.mark.parametrize("name", BOXED)
def test_lean_proves_a_rule_inside_a_box(name: str, tmp_path: object) -> None:
    ok, output = L.check(L.export(_boxed_rule(name)), f"{tmp_path}/box.lean")
    assert ok, output


def _nested_rule(name: str) -> K.Theorem:
    from .test_nested_box_rewrite import _FIXTURES

    rule_name, build = _FIXTURES[name]
    diagram = build(3)
    rule = RULES[rule_name]
    match = next(m for m in rule.pattern.find_matches(diagram) if m.all_side_conditions_passed)
    return K.rule(diagram, apply(diagram, rule, match).step)


NESTED = [
    "fusion",
    "fusion_three_level",
    "fusion_port_scope",
    "identity",
    "triangle",
    "fourier",
    "fourier_state",
]


@pytest.mark.parametrize("name", NESTED)
def test_a_rule_inside_nested_boxes_exports_without_axioms(name: str) -> None:
    source = L.export(_nested_rule(name))
    assert "\naxiom " not in source and "theorem bun" in source


@needs_lean
@pytest.mark.slow
@pytest.mark.parametrize("name", NESTED)
def test_lean_proves_a_rule_inside_nested_boxes(name: str, tmp_path: object) -> None:
    ok, output = L.check(L.export(_nested_rule(name)), f"{tmp_path}/nested.lean")
    assert ok, output


def _shape(name: str) -> K.Theorem:
    from archytaszx.diagram.bangbox import Mult

    d = Dim.concrete(3)

    def o(n: object, i: int = 0) -> PortRef:
        return PortRef(n, Direction.OUTPUT, i)  # type: ignore[arg-type]

    def i_(n: object, i: int = 0) -> PortRef:
        return PortRef(n, Direction.INPUT, i)  # type: ignore[arg-type]

    g = Diagram()
    if name == "crossing":
        a = g.add_node(Z_SPIDER, [], [d])
        b = g.add_node(Z_SPIDER, [d], [d])
        c = g.add_node(Z_SPIDER, [d], [d])
        g.add_wire(o(a), i_(b))
        g.add_wire(o(b), i_(c))
        g.set_boundary_outputs([o(c)])
        outer = g.add_bang_box(Mult("k"), node_scope=frozenset({a, b, c}))
        g.add_bang_box(Mult("j"), node_scope=frozenset({a}), parent=outer)
        rule = "spider_fusion"
    else:
        a = g.add_node(Z_SPIDER, [], [d] if name == "two_legs" else [d, d])
        b = g.add_node(Z_SPIDER, [d], [d, d] if name == "two_legs" else [d])
        g.add_wire(o(a, 0), i_(b))
        legs = [o(b, 0), o(b, 1)] if name == "two_legs" else [o(a, 1), o(b)]
        g.set_boundary_outputs(legs)
        g.add_bang_box(Mult("n"), port_scope=frozenset(legs))
        rule = "spider_fusion"
    r = RULES[rule]
    match = next(m for m in r.pattern.find_matches(g) if m.all_side_conditions_passed)
    return K.rule(g, apply(g, r, match).step)


def _nested_induction() -> K.Theorem:
    from archytaszx.algebra.scalar import Scalar
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics.prove import prove

    d = Dim.symbol("d")
    left = Diagram()
    s = left.add_node(X_SPIDER, [], [d])
    z = left.add_node(Z_SPIDER, [d], [d, d])
    left.add_wire(PortRef(s, Direction.OUTPUT, 0), PortRef(z, Direction.INPUT, 0))
    outs = [PortRef(z, Direction.OUTPUT, 0), PortRef(z, Direction.OUTPUT, 1)]
    left.set_boundary_outputs(outs)
    left.add_bang_box(Mult("n"), port_scope=frozenset({outs[0]}))
    left.add_bang_box(Mult("m"), port_scope=frozenset({outs[1]}))
    right = Diagram()
    c1 = right.add_node(X_SPIDER, [], [d])
    c2 = right.add_node(X_SPIDER, [], [d])
    right.set_boundary_outputs([PortRef(c1, Direction.OUTPUT, 0), PortRef(c2, Direction.OUTPUT, 0)])
    n, m = (sp.Symbol(x, integer=True, nonnegative=True, multiplicity=True) for x in "nm")
    right.multiply_scalar(Scalar(sp.Pow(d.to_sympy(), (1 - n - m) / 2)))
    right.add_bang_box(Mult("n"), node_scope=frozenset({c1}))
    right.add_bang_box(Mult("m"), node_scope=frozenset({c2}))
    outcome = prove(left, right)
    assert outcome.induction is not None
    return K.check(outcome.induction)


@pytest.mark.parametrize("name", ["crossing", "two_legs", "two_nodes"])
def test_box_shapes_export_without_axioms(name: str) -> None:
    assert "\naxiom " not in L.export(_shape(name))


def test_a_nested_induction_exports_without_axioms() -> None:
    source = L.export(_nested_induction())
    assert "\naxiom " not in source and "theorem ind1 " in source


@needs_lean
@pytest.mark.slow
@pytest.mark.parametrize("name", ["crossing", "two_legs", "two_nodes"])
def test_lean_proves_box_shapes(name: str, tmp_path: object) -> None:
    ok, output = L.check(L.export(_shape(name)), f"{tmp_path}/shape.lean")
    assert ok, output


@needs_lean
@pytest.mark.slow
def test_lean_proves_a_nested_induction(tmp_path: object) -> None:
    ok, output = L.check(L.export(_nested_induction()), f"{tmp_path}/nested_induction.lean")
    assert ok, output
