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

"""Tests for archytaszx.rewrite.sheet: rewrites performed in scalable notation."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, expand_concrete_boxes, kill, peel_one
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef, Wire
from archytaszx.diagram.scalable import (
    Bundle,
    ScalableBuilder,
    ScalableDiagram,
    ScaleId,
    ScaleKind,
    from_scalable,
    strip,
    to_scalable,
    validate_scalable,
)
from archytaszx.rewrite import sheet
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import RewriteError, RewriteGrammarError
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.rewrite.sheet import (
    SHEET_OPERATIONS,
    SheetDomainError,
    SheetError,
    SheetGrammarError,
    SheetStep,
    dissolve,
    enclose,
    fuse_sheet,
    join_scales,
    kill_scale,
    replay_sheet_step,
    split_scale,
)
from archytaszx.semantics.check import compare

from .helpers import build_ghz_with_copy
from .test_scalable import (
    BANG_BOX_FAMILIES,
    fusion_pair,
    multi_port,
    nested_copies,
    nested_port_in_copies,
    sheet_family,
)

D2 = Dim(2)
D3 = Dim(3)
S0 = ScaleId(0)
S1 = ScaleId(1)


def _in(node: int, index: int) -> PortRef:
    return PortRef(NodeId(node), Direction.INPUT, index)


def _out(node: int, index: int) -> PortRef:
    return PortRef(NodeId(node), Direction.OUTPUT, index)


def _with_mult(diagram: Diagram, box: int, mult: Mult) -> Diagram:
    working = diagram.copy()
    working.set_bang_box_multiplicity(BangBoxId(box), mult)
    return working


def _last_scale(s: ScalableDiagram) -> ScaleId:
    return max(sc.id for sc in s.scales)


def _assert_valid(step: SheetStep) -> None:
    assert validate_scalable(step.before) == ()
    assert validate_scalable(step.after) == ()
    assert [name for name, _ in step.arguments] == sorted(name for name, _ in step.arguments)
    assert replay_sheet_step(step) == step


def _oracle_equal(a: Diagram, b: Diagram, assignment: dict[str, int]) -> bool:
    return compare(a, b, assignment).matched


# boxes whose multiplicity can be made n + 1: (family, box id)
PEELABLE = [
    ("fusion_pair", 0),
    ("nested_copies", 0),
    ("nested_copies", 1),
    ("nested_port_in_copies", 0),
    ("two_components", 0),
    ("non_bundle_normal", 0),
    ("siblings_sharing_symbol", 1),
    ("qufinite_mixed_dims", 0),
    ("symbolic_phase", 0),
]


# -- split ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("family", "box"), PEELABLE)
class TestSplitAgainstPeel:
    def test_split_n_plus_one_equals_peel_one(self, family: str, box: int) -> None:
        diagram = _with_mult(BANG_BOX_FAMILIES[family](), box, Mult("n") + 1)
        step = split_scale(to_scalable(diagram), ScaleId(box), Mult("n"))
        _assert_valid(step)
        peeled = peel_one(diagram, BangBoxId(box)).diagram
        assert isomorphic(from_scalable(step.after), peeled)

    def test_split_then_join_is_the_identity(self, family: str, box: int) -> None:
        s = to_scalable(_with_mult(BANG_BOX_FAMILIES[family](), box, Mult("n") + 1))
        split = split_scale(s, ScaleId(box), Mult("n"), dissolve_unit=False)
        joined = join_scales(split.after, ScaleId(box), ScaleId(_last_scale(s) + 1))
        _assert_valid(joined)
        assert joined.after == s

    def test_enclose_then_join_undoes_a_peel(self, family: str, box: int) -> None:
        diagram = _with_mult(BANG_BOX_FAMILIES[family](), box, Mult("n") + 1)
        peeled = peel_one(diagram, BangBoxId(box))
        wrapped = enclose(to_scalable(peeled.diagram), peeled.copy_node_ids)
        _assert_valid(wrapped)
        joined = join_scales(wrapped.after, ScaleId(box), _last_scale(wrapped.after))
        assert joined.after == to_scalable(diagram)


class TestSplitSymbolic:
    @pytest.mark.parametrize(
        ("total", "first", "second"),
        [
            (Mult("a") + Mult("b"), Mult("a"), Mult("b")),
            (Mult("k") * 2, Mult("k"), Mult("k")),
            (Mult("k") + 3, 2, Mult("k") + 1),
            (Mult("a") * Mult("b") + Mult("a"), Mult("a"), Mult("a") * Mult("b")),
        ],
    )
    def test_parts_and_family_equality(self, total: Mult, first: Mult | int, second: Mult) -> None:
        s = to_scalable(_with_mult(fusion_pair(), 0, total))
        step = split_scale(s, S0, first)
        _assert_valid(step)
        assert step.after.scale(S0).multiplicity == Mult(first)
        assert step.after.scale(S1).multiplicity == second
        assert step.after.scale(S1).parent is None
        assert dict(step.arguments)["first"] == Mult(first)
        names = sorted(s.free_mult_symbols())
        for values in ((0, 0), (1, 2), (2, 1), (3, 0)):
            env = dict(zip(names, values, strict=False))
            assert isomorphic(strip(step.after, env), strip(s, env))

    def test_split_layout_puts_the_new_bundle_after_the_original(self) -> None:
        step = split_scale(to_scalable(_with_mult(fusion_pair(), 0, Mult("k") * 2)), S0, Mult("k"))
        (first, second) = step.after.outputs
        assert isinstance(first, Bundle) and isinstance(second, Bundle)
        assert (first.scale, second.scale) == (S0, S1)
        assert [n.id for n in step.after.nodes] == [0, 1, 2, 3]
        assert {n.scale for n in step.after.nodes} == {S0, S1}

    def test_split_dissolves_both_unit_parts(self) -> None:
        step = split_scale(to_scalable(_with_mult(fusion_pair(), 0, Mult(2))), S0, 1)
        assert step.after.scales == ()
        assert all(n.scale is None for n in step.after.nodes)
        assert isomorphic(
            from_scalable(step.after), expand_concrete_boxes(_with_mult(fusion_pair(), 0, Mult(2)))
        )

    def test_nested_split_copies_child_scales(self) -> None:
        s = to_scalable(_with_mult(nested_copies(), 0, Mult("k1") * 2))
        step = split_scale(s, S0, Mult("k1"))
        kinds = [(sc.id, sc.multiplicity, sc.parent) for sc in step.after.scales]
        assert kinds == [
            (0, Mult("k1"), None),
            (1, Mult("k2"), 0),
            (2, Mult("k1"), None),
            (3, Mult("k2"), 2),
        ]
        for env in ({"k1": 1, "k2": 2}, {"k1": 2, "k2": 0}):
            assert isomorphic(strip(step.after, env), strip(s, env))

    def test_split_oracle(self) -> None:
        diagram = _with_mult(fusion_pair(), 0, Mult("a") + Mult("b"))
        after = from_scalable(split_scale(to_scalable(diagram), S0, Mult("a")).after)
        for env in ({"a": 0, "b": 1}, {"a": 1, "b": 1}, {"a": 2, "b": 0}):
            assert _oracle_equal(diagram, after, env)


# -- dissolve and kill ---------------------------------------------------------------------

UNIT_BOXES = [
    ("ghz", 0),
    ("multi_port", 0),
    ("fusion_pair", 0),
    ("nested_copies", 0),
    ("nested_copies", 1),
    ("nested_port_in_copies", 0),
    ("nested_port_in_copies", 1),
    ("two_components", 0),
    ("siblings_sharing_symbol", 0),
    ("compound", 0),
    ("compound", 1),
    ("qufinite_mixed_dims", 0),
    ("qufinite_mixed_dims", 1),
    ("symbolic_phase", 0),
]


@pytest.mark.parametrize(("family", "box"), UNIT_BOXES)
class TestDissolveAndKill:
    def test_dissolve_matches_instantiation_at_one(self, family: str, box: int) -> None:
        diagram = _with_mult(BANG_BOX_FAMILIES[family](), box, Mult(1))
        step = dissolve(to_scalable(diagram), ScaleId(box))
        _assert_valid(step)
        assert all(sc.id != box for sc in step.after.scales)
        assert isomorphic(from_scalable(step.after), expand_concrete_boxes(diagram))

    def test_kill_matches_instantiation_at_zero(self, family: str, box: int) -> None:
        diagram = _with_mult(BANG_BOX_FAMILIES[family](), box, Mult(0))
        step = kill_scale(to_scalable(diagram), ScaleId(box))
        _assert_valid(step)
        assert isomorphic(from_scalable(step.after), kill(diagram, BangBoxId(box)))


class TestDissolveAndKillDetails:
    def test_dissolve_legs_under_legs_moves_fans_to_the_parent(self) -> None:
        b = ScalableBuilder()
        outer = b.add_scale(ScaleKind.LEGS, "n")
        inner = b.add_scale(ScaleKind.LEGS, 1, parent=outer)
        z = b.add_node(Z_SPIDER, [], [D2, D2], PhaseVector(D2))
        b.set_fan(_out(z, 0), inner)
        b.set_fan(_out(z, 1), outer)
        b.set_outputs([_out(z, 0), _out(z, 1)])
        step = dissolve(b.build(), inner)
        _assert_valid(step)
        assert [p.fan for p in step.after.node(z).outputs] == [outer, outer]

    def test_kill_inner_scale_keeps_the_outer_one(self) -> None:
        diagram = _with_mult(nested_copies(), 1, Mult(0))
        step = kill_scale(to_scalable(diagram), S1)
        assert [sc.id for sc in step.after.scales] == [S0]
        assert [n.id for n in step.after.nodes] == [0]
        assert step.after.outputs == (Bundle(S0, (_out(0, 0), _out(0, 1))),)

    def test_kill_legs_shifts_later_indices(self) -> None:
        s = to_scalable(_with_mult(multi_port(), 0, Mult(0)))
        step = kill_scale(s, S0)
        (z,) = step.after.nodes
        assert (len(z.inputs), len(z.outputs)) == (0, 1)
        assert step.after.outputs == (_out(z.id, 0),)

    def test_kill_drops_the_binding_of_a_vanished_child_symbol(self) -> None:
        diagram = _with_mult(nested_copies(), 0, Mult(0))
        diagram.bind_parameter("k2", 2)
        step = kill_scale(to_scalable(diagram), S0)
        _assert_valid(step)
        assert dict(step.after.parameters) == {}
        assert dict(step.after.parameters) == dict(kill(diagram, BangBoxId(0)).parameters)
        assert _oracle_equal(from_scalable(step.before), from_scalable(step.after), {})

    def test_kill_drops_the_binding_of_a_vanished_compound_symbol(self) -> None:
        b = ScalableBuilder()
        dead = b.add_scale(ScaleKind.COPIES, 0)
        legs = b.add_scale(ScaleKind.LEGS, Mult("j") + 2, parent=dead)
        x = b.add_node(X_SPIDER, [D2], [], PhaseVector(D2), scale=dead)
        b.set_fan(_in(x, 0), legs)
        b.set_inputs([Bundle(dead, (_in(x, 0),))])
        b.add_node(Z_SPIDER, [], [], PhaseVector(D2))
        b.bind_parameter("j", 0)
        step = kill_scale(b.build(), dead)
        _assert_valid(step)
        assert step.after.scales == () and dict(step.after.parameters) == {}
        assert _oracle_equal(from_scalable(step.before), from_scalable(step.after), {})

    def test_kill_keeps_a_binding_still_carried_elsewhere(self) -> None:
        b = ScalableBuilder()
        dead = b.add_scale(ScaleKind.COPIES, 0)
        inner = b.add_scale(ScaleKind.COPIES, "j", parent=dead)
        live = b.add_scale(ScaleKind.COPIES, Mult("j") + 1)
        p = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=dead)
        q = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=inner)
        r = b.add_node(X_SPIDER, [], [D2], PhaseVector(D2), scale=live)
        b.set_outputs(
            [Bundle(dead, (_out(p, 0), Bundle(inner, (_out(q, 0),)))), Bundle(live, (_out(r, 0),))]
        )
        b.bind_parameter("j", 1)
        step = kill_scale(b.build(), dead)
        _assert_valid(step)
        assert dict(step.after.parameters) == {"j": 1}
        assert _oracle_equal(from_scalable(step.before), from_scalable(step.after), {})

    def test_kill_legs_drops_the_binding_of_a_vanished_sub_fan(self) -> None:
        b = ScalableBuilder()
        outer = b.add_scale(ScaleKind.LEGS, 0)
        inner = b.add_scale(ScaleKind.LEGS, "m", parent=outer)
        z = b.add_node(Z_SPIDER, [], [D2, D2, D2], PhaseVector(D2))
        b.set_fan(_out(z, 0), inner)
        b.set_fan(_out(z, 1), outer)
        b.set_outputs([_out(z, 0), _out(z, 1), _out(z, 2)])
        b.bind_parameter("m", 2)
        step = kill_scale(b.build(), outer)
        _assert_valid(step)
        assert step.after.scales == () and dict(step.after.parameters) == {}
        assert _oracle_equal(from_scalable(step.before), from_scalable(step.after), {})

    def test_each_operation_validates_its_result(self) -> None:
        s = to_scalable(fusion_pair())
        with pytest.raises(SheetGrammarError, match="result is not valid"):
            enclose(s, [NodeId(0), NodeId(1)])


# -- fusion ---------------------------------------------------------------------------------


def _engine_fuse(diagram: Diagram) -> Diagram:
    (match, *_rest) = SPIDER_FUSION.pattern.find_matches(diagram)
    return apply(diagram, SPIDER_FUSION, match).diagram


def _boxed_pair(
    generator: str = "z", parallel: bool = False, phases: bool = False, dim: Dim = D2
) -> Diagram:
    gen = Z_SPIDER if generator == "z" else X_SPIDER
    pa = PhaseVector(dim, {1: Phase.symbol("alpha")}) if phases else None
    pb = PhaseVector(dim, {1: Phase.turns(sp.Rational(1, 3))}) if phases else None
    d = Diagram()
    a = d.add_node(gen, [dim], [dim, dim, dim], phase=pa)
    b = d.add_node(gen, [dim, dim], [dim], phase=pb)
    d.add_wire(_out(a, 0), _in(b, 0))
    if parallel:
        d.add_wire(_out(a, 2), _in(b, 1))
        d.set_boundary_inputs([_in(a, 0)])
    else:
        d.set_boundary_inputs([_in(b, 1), _in(a, 0)])
    d.set_boundary_outputs([_out(b, 0), _out(a, 1)] + ([] if parallel else [_out(a, 2)]))
    d.add_bang_box(Mult("k"), node_scope=frozenset({a, b}))
    return d


FUSION_CASES: dict[str, Callable[[], Diagram]] = {
    "fusion_pair": fusion_pair,
    "z_boxed": _boxed_pair,
    "x_boxed": lambda: _boxed_pair("x"),
    "parallel_wires": lambda: _boxed_pair(parallel=True),
    "phases": lambda: _boxed_pair(phases=True),
    "symbolic_dim": lambda: _boxed_pair(dim=Dim.symbol("d")),
    "unboxed": lambda: build_ghz_with_copy(D2)[0],
}


@pytest.mark.parametrize("name", sorted(FUSION_CASES))
class TestFuseSheet:
    def test_matches_engine_spider_fusion(self, name: str) -> None:
        diagram = FUSION_CASES[name]()
        s = to_scalable(diagram)
        (wire,) = sorted(s.wires, key=lambda w: w.sort_key())[:1]
        step = fuse_sheet(s, wire)
        _assert_valid(step)
        engine = _engine_fuse(diagram)
        assert isomorphic(from_scalable(step.after), engine)
        assert isomorphic(comparison_view(from_scalable(step.after)), comparison_view(engine))

    def test_oracle_equal_at_several_sizes(self, name: str) -> None:
        diagram = FUSION_CASES[name]()
        s = to_scalable(diagram)
        (wire,) = sorted(s.wires, key=lambda w: w.sort_key())[:1]
        after = from_scalable(fuse_sheet(s, wire).after)
        for k in (0, 1, 2):
            for d in (2, 3):
                env = {"k": k, "m": k, "d": d, "alpha": 0}
                assert _oracle_equal(diagram, after, env), env


class TestFuseSheetDetails:
    def test_leg_order_phase_and_id_policy(self) -> None:
        s = to_scalable(_boxed_pair(phases=True))
        wire = Wire(_out(0, 0), _in(1, 0))
        step = fuse_sheet(s, wire)
        (merged,) = step.after.nodes
        assert merged.id == 2 and merged.scale == S0
        assert len(merged.inputs) == 2 and len(merged.outputs) == 3
        assert merged.phase == PhaseVector(
            D2, {1: Phase.symbol("alpha") + Phase.turns(sp.Rational(1, 3))}
        )
        assert step.after.inputs == (Bundle(S0, (_in(2, 1), _in(2, 0))),)
        assert step.after.outputs == (Bundle(S0, (_out(2, 2), _out(2, 0), _out(2, 1))),)

    def test_parallel_wire_becomes_a_self_loop(self) -> None:
        s = to_scalable(_boxed_pair(parallel=True))
        step = fuse_sheet(s, Wire(_in(1, 0), _out(0, 0)))
        assert step.after.wires == frozenset({Wire(_out(2, 1), _in(2, 1))})

    def test_fans_follow_their_ports(self) -> None:
        b = ScalableBuilder()
        copies = b.add_scale(ScaleKind.COPIES, "k")
        legs = b.add_scale(ScaleKind.LEGS, "n", parent=copies)
        a = b.add_node(Z_SPIDER, [], [D2], scale=copies)
        c = b.add_node(Z_SPIDER, [D2], [D2], scale=copies)
        b.add_wire(_out(a, 0), _in(c, 0))
        b.set_fan(_out(c, 0), legs)
        b.set_outputs([Bundle(copies, (_out(c, 0),))])
        step = fuse_sheet(b.build(), Wire(_out(a, 0), _in(c, 0)))
        (merged,) = step.after.nodes
        assert merged.outputs[0].fan == legs and merged.phase is None
        engine = _engine_fuse(from_scalable(step.before))
        assert isomorphic(from_scalable(step.after), engine)

    def test_no_surviving_leg_gets_a_zero_phase(self) -> None:
        b = ScalableBuilder()
        a = b.add_node(Z_SPIDER, [], [D3])
        c = b.add_node(Z_SPIDER, [D3], [])
        b.add_wire(_out(a, 0), _in(c, 0))
        step = fuse_sheet(b.build(), Wire(_out(a, 0), _in(c, 0)))
        assert step.after.nodes[0].phase == PhaseVector(D3)

    def test_nested_scales_fuse_natively(self) -> None:
        b = ScalableBuilder()
        outer = b.add_scale(ScaleKind.COPIES, "a")
        inner = b.add_scale(ScaleKind.COPIES, "b", parent=outer)
        p = b.add_node(Z_SPIDER, [], [D2, D2], PhaseVector(D2), scale=inner)
        q = b.add_node(Z_SPIDER, [D2], [D2], PhaseVector(D2), scale=inner)
        x = b.add_node(X_SPIDER, [], [D2], PhaseVector(D2), scale=outer)
        b.add_wire(_out(p, 0), _in(q, 0))
        b.set_outputs([Bundle(outer, (_out(x, 0), Bundle(inner, (_out(p, 1), _out(q, 0)))))])
        s = b.build()
        after = from_scalable(fuse_sheet(s, Wire(_out(p, 0), _in(q, 0))).after)
        for env in ({"a": 1, "b": 2}, {"a": 2, "b": 1}, {"a": 0, "b": 1}):
            assert _oracle_equal(from_scalable(s), after, env)

    def test_rule_scalar_is_raised_to_the_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sheet, "SPIDER_FUSION", SimpleNamespace(scalar_introduced=Scalar.rational(2))
        )
        wire = Wire(_out(0, 0), _in(1, 0))
        concrete = to_scalable(_with_mult(fusion_pair(), 0, Mult(3)))
        assert fuse_sheet(concrete, wire).after.scalar == Scalar.rational(8)
        with pytest.raises(SheetDomainError, match="symbolic size"):
            fuse_sheet(to_scalable(fusion_pair()), wire)


# -- refusals -------------------------------------------------------------------------------


def _pair_family(second_phase: PhaseVector | None = None, second_dim: Dim = D2) -> ScalableDiagram:
    b = ScalableBuilder()
    one = b.add_scale(ScaleKind.COPIES, "a")
    two = b.add_scale(ScaleKind.COPIES, "b")
    x = b.add_node(Z_SPIDER, [], [D2], scale=one)
    y = b.add_node(Z_SPIDER, [], [second_dim], second_phase, scale=two)
    b.set_outputs([Bundle(one, (_out(x, 0),)), Bundle(two, (_out(y, 0),))])
    return b.build()


class TestRefusals:
    def test_errors_derive_from_rewrite_error(self) -> None:
        assert issubclass(SheetGrammarError, SheetError)
        assert issubclass(SheetDomainError, SheetError)
        assert issubclass(SheetError, RewriteError)
        assert issubclass(SheetGrammarError, RewriteGrammarError)

    @pytest.mark.parametrize("name", sorted(SHEET_OPERATIONS))
    def test_non_scalable_input(self, name: str) -> None:
        args = {"enclose": ([0],), "fuse_sheet": (None,), "join_scales": (S0, S1)}
        with pytest.raises(SheetGrammarError, match="requires a ScalableDiagram"):
            SHEET_OPERATIONS[name](
                Diagram(), *args.get(name, (S0, 1)[: 2 if name == "split_scale" else 1])
            )

    def test_invalid_input(self) -> None:
        s = to_scalable(fusion_pair())
        broken = replace(s, inputs=(_out(0, 0),))
        with pytest.raises(SheetGrammarError, match="input is not valid"):
            dissolve(broken, S0)

    @pytest.mark.parametrize(
        ("args", "error", "pattern"),
        [
            ((ScaleId(9), Mult("m")), SheetGrammarError, "no such scale"),
            ((True, Mult("m")), SheetGrammarError, "must be an int"),
            ((S0, "m"), SheetGrammarError, "Mult or int"),
            ((S0, -1), SheetDomainError, ">= 0"),
            ((S0, 0), SheetDomainError, "zero part"),
            ((S0, Mult("m")), SheetDomainError, "zero part"),
            ((S0, Mult("m") + 1), SheetDomainError, "negative"),
            ((S0, Mult("j")), SheetDomainError, "negative"),
        ],
    )
    def test_split_refusals(self, args: tuple[object, object], error: type, pattern: str) -> None:
        with pytest.raises(error, match=pattern):
            split_scale(to_scalable(fusion_pair()), *args)  # type: ignore[arg-type]

    def test_split_refuses_a_legs_scale_and_a_bad_flag(self) -> None:
        with pytest.raises(SheetGrammarError, match="not a COPIES"):
            split_scale(to_scalable(multi_port()), S0, 1)
        with pytest.raises(SheetGrammarError, match="dissolve_unit"):
            split_scale(to_scalable(fusion_pair()), S0, 1, dissolve_unit=1)  # type: ignore[arg-type]

    def test_join_refusals(self) -> None:
        good = _pair_family()
        assert join_scales(good, S0, S1).after.scale(S0).multiplicity == Mult("a") + Mult("b")
        cases: list[tuple[ScalableDiagram, ScaleId, ScaleId, str]] = [
            (good, S0, S0, "same scale"),
            (to_scalable(nested_port_in_copies()), S0, S1, "COPIES"),
            (to_scalable(nested_copies()), S0, S1, "not siblings"),
            (_pair_family(PhaseVector(D2, {1: Phase.turns(sp.Rational(1, 2))})), S0, S1, "phase"),
            (_pair_family(second_dim=D3), S0, S1, "ports"),
        ]
        for s, a, b, pattern in cases:
            with pytest.raises(SheetGrammarError, match=pattern):
                join_scales(s, a, b)

    def test_join_refuses_mismatched_structure(self) -> None:
        def family(variant: str) -> ScalableDiagram:
            b = ScalableBuilder()
            one = b.add_scale(ScaleKind.COPIES, "a")
            two = b.add_scale(ScaleKind.COPIES, "a")
            inner = b.add_scale(ScaleKind.COPIES, "c" if variant == "mult" else "b", parent=two)
            ref = b.add_scale(ScaleKind.COPIES, "b", parent=one)
            p = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=ref)
            q = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=one)
            r = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=inner)
            t = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=two)
            if variant == "order":
                b.set_outputs(
                    [
                        Bundle(one, (Bundle(ref, (_out(p, 0),)), _out(q, 0))),
                        Bundle(two, (_out(t, 0), Bundle(inner, (_out(r, 0),)))),
                    ]
                )
            elif variant == "apart":
                b.set_outputs(
                    [
                        Bundle(two, (Bundle(inner, (_out(r, 0),)), _out(t, 0))),
                        Bundle(one, (Bundle(ref, (_out(p, 0),)), _out(q, 0))),
                    ]
                )
            else:
                b.set_outputs(
                    [
                        Bundle(one, (Bundle(ref, (_out(p, 0),)), _out(q, 0))),
                        Bundle(two, (Bundle(inner, (_out(r, 0),)), _out(t, 0))),
                    ]
                )
            return b.build()

        assert join_scales(family("ok"), S0, S1).after.scale(S0).multiplicity == Mult("a") * 2
        for variant, pattern in (
            ("mult", "kind or size"),
            ("order", "do not correspond"),
            ("apart", "does not directly follow"),
        ):
            with pytest.raises(SheetGrammarError, match=pattern):
                join_scales(family(variant), S0, S1)

    def test_join_refuses_counts_nesting_wires_and_one_sided_bundles(self) -> None:
        assert join_scales(_wired_pair(second_wired=True), S0, S1).after.wires == frozenset(
            {Wire(_out(0, 0), _in(1, 0))}
        )
        with pytest.raises(SheetGrammarError, match="wires"):
            join_scales(_wired_pair(second_wired=False), S0, S1)
        b = ScalableBuilder()
        one = b.add_scale(ScaleKind.COPIES, "a")
        two = b.add_scale(ScaleKind.COPIES, "a")
        x = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=one)
        b.add_node(Z_SPIDER, [], [], PhaseVector(D2), scale=one)
        y = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=two)
        b.set_outputs([Bundle(one, (_out(x, 0),)), Bundle(two, (_out(y, 0),))])
        with pytest.raises(SheetGrammarError, match="node counts"):
            join_scales(b.build(), S0, S1)
        b = ScalableBuilder()
        one = b.add_scale(ScaleKind.COPIES, "a")
        two = b.add_scale(ScaleKind.COPIES, "a")
        legs = b.add_scale(ScaleKind.LEGS, "n", parent=one)
        x = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=one)
        y = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=two)
        b.set_fan(_out(x, 0), legs)
        b.set_outputs([Bundle(one, (_out(x, 0),)), Bundle(two, (_out(y, 0),))])
        with pytest.raises(SheetGrammarError, match="nesting"):
            join_scales(b.build(), S0, S1)
        b = ScalableBuilder()
        one = b.add_scale(ScaleKind.COPIES, "a")
        two = b.add_scale(ScaleKind.COPIES, "a")
        x = b.add_node(Z_SPIDER, [D2], [], PhaseVector(D2), scale=one)
        y = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=two)
        b.set_inputs([Bundle(one, (_in(x, 0),))])
        b.set_outputs([Bundle(two, (_out(y, 0),))])
        with pytest.raises(SheetGrammarError, match="ports"):
            join_scales(b.build(), S0, S1)

    @pytest.mark.parametrize(
        ("left", "right", "pattern"),
        [((1, 1), (1, 2), "pairs with both"), ((1, 2), (1, 1), "do not correspond")],
    )
    def test_join_refuses_a_non_bijective_scale_pairing(
        self, left: tuple[int, int], right: tuple[int, int], pattern: str
    ) -> None:
        b = ScalableBuilder()
        roots = (b.add_scale(ScaleKind.COPIES, "a"), b.add_scale(ScaleKind.COPIES, "a"))
        children = [
            [b.add_scale(ScaleKind.COPIES, "c", parent=root) for _ in range(max(side))]
            for root, side in zip(roots, (left, right), strict=True)
        ]
        for root, side, kids in zip(roots, (left, right), children, strict=True):
            b.add_node(X_SPIDER, [], [], PhaseVector(D2), scale=root)
            for choice in side:
                b.add_node(Z_SPIDER, [], [], PhaseVector(D2), scale=kids[choice - 1])
        with pytest.raises(SheetGrammarError, match=pattern):
            join_scales(b.build(), S0, S1)

    def test_enclose_refusals(self) -> None:
        s = to_scalable(BANG_BOX_FAMILIES["two_components"]())
        cases: list[tuple[ScalableDiagram, object, type, str]] = [
            (s, [], SheetGrammarError, "empty"),
            (s, "ab", SheetGrammarError, "iterable"),
            (s, [NodeId(7)], SheetGrammarError, "no such node"),
            (s, [NodeId(0), NodeId(2)], SheetGrammarError, "only partly enclosed"),
            (to_scalable(fusion_pair()), [NodeId(0)], SheetGrammarError, "joins an enclosed"),
            (
                to_scalable(BANG_BOX_FAMILIES["siblings_sharing_symbol"]()),
                [NodeId(0), NodeId(1)],
                SheetGrammarError,
                "different scales",
            ),
        ]
        for diagram, nodes, error, pattern in cases:
            with pytest.raises(error, match=pattern):
                enclose(diagram, nodes)  # type: ignore[arg-type]
        with pytest.raises(SheetDomainError, match="multiplicity must be 1"):
            enclose(s, [NodeId(0)], multiplicity=2)

    def test_enclose_refuses_partial_and_scattered_sets(self) -> None:
        b = ScalableBuilder()
        child = b.add_scale(ScaleKind.COPIES, "k")
        p = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2))
        q = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=child)
        r = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=child)
        x = b.add_node(X_SPIDER, [], [D2], PhaseVector(D2))
        b.set_outputs([_out(p, 0), _out(x, 0), Bundle(child, (_out(q, 0), _out(r, 0)))])
        s = b.build()
        with pytest.raises(SheetGrammarError, match="only partly enclosed"):
            enclose(s, [p, q])
        with pytest.raises(SheetGrammarError, match="not contiguous"):
            enclose(s, [p, q, r])
        step = enclose(s, [x, q, r])
        _assert_valid(step)
        assert step.after.outputs[1:] == (
            Bundle(ScaleId(1), (_out(x, 0), Bundle(child, (_out(q, 0), _out(r, 0))))),
        )
        assert step.after.scale(child).parent == ScaleId(1)
        assert dict(step.arguments)["nodes"] == (q, r, x)
        with pytest.raises(SheetGrammarError, match="outside scale"):
            enclose(_apart(), [NodeId(0), NodeId(1)])

    def test_enclose_moves_legs_and_refuses_a_partly_covered_one(self) -> None:
        s = to_scalable(BANG_BOX_FAMILIES["ghz"]())
        step = enclose(s, [NodeId(0)])
        assert step.after.scale(S0).parent == S1
        b = ScalableBuilder()
        legs = b.add_scale(ScaleKind.LEGS, "n")
        p = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2))
        q = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2))
        b.set_fan(_out(p, 0), legs)
        b.set_fan(_out(q, 0), legs)
        b.set_outputs([_out(p, 0), _out(q, 0)])
        with pytest.raises(SheetGrammarError, match="LEGS scale 0 is only partly"):
            enclose(b.build(), [p])

    def test_dissolve_and_kill_refusals(self) -> None:
        s = to_scalable(fusion_pair())
        with pytest.raises(SheetDomainError, match="multiplicity"):
            dissolve(s, S0)
        with pytest.raises(SheetDomainError, match="multiplicity"):
            kill_scale(s, S0)
        with pytest.raises(SheetGrammarError, match="no such scale"):
            kill_scale(s, ScaleId(4))

    def test_fuse_refusals(self) -> None:
        s = to_scalable(fusion_pair())
        b = ScalableBuilder()
        z = b.add_node(Z_SPIDER, [D2], [D2], PhaseVector(D2))
        b.add_wire(_out(z, 0), _in(z, 0))
        loop = b.build()
        cases: list[tuple[ScalableDiagram, object, type, str]] = [
            (s, "w", SheetGrammarError, "not a wire"),
            (s, Wire(_out(0, 1), _in(1, 0)), SheetGrammarError, "not a wire"),
            (loop, Wire(_out(0, 0), _in(0, 0)), SheetGrammarError, "self-loop"),
            (_mixed(Z_SPIDER, X_SPIDER), Wire(_out(0, 0), _in(1, 0)), SheetDomainError, "colour"),
            (
                _mixed(FOURIER_BOX, FOURIER_BOX),
                Wire(_out(0, 0), _in(1, 0)),
                SheetDomainError,
                "colour",
            ),
            (_x_same_direction(), Wire(_out(0, 0), _out(1, 0)), SheetDomainError, "output to"),
            (_mismatch(phase=False), Wire(_out(0, 0), _in(1, 0)), SheetDomainError, "dimension"),
            (_mismatch(phase=True), Wire(_out(0, 0), _in(1, 0)), SheetDomainError, "dimension"),
        ]
        for diagram, wire, error, pattern in cases:
            with pytest.raises(error, match=pattern):
                fuse_sheet(diagram, wire)  # type: ignore[arg-type]


def _wired_pair(*, second_wired: bool) -> ScalableDiagram:
    b = ScalableBuilder()
    one = b.add_scale(ScaleKind.COPIES, "a")
    two = b.add_scale(ScaleKind.COPIES, "a")
    p = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=one)
    q = b.add_node(Z_SPIDER, [D2], [], PhaseVector(D2), scale=one)
    r = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=two)
    t = b.add_node(Z_SPIDER, [D2], [], PhaseVector(D2), scale=two)
    b.add_wire(_out(p, 0), _in(q, 0))
    if second_wired:
        b.add_wire(_out(r, 0), _in(t, 0))
    else:
        b.set_inputs([Bundle(two, (_in(t, 0),))])
        b.set_outputs([Bundle(two, (_out(r, 0),))])
    return b.build()


def _apart() -> ScalableDiagram:
    b = ScalableBuilder()
    a = b.add_scale(ScaleKind.COPIES, "a")
    c = b.add_scale(ScaleKind.COPIES, "c")
    inner = b.add_scale(ScaleKind.COPIES, "b", parent=c)
    p = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=a)
    q = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2), scale=inner)
    b.add_node(X_SPIDER, [], [], PhaseVector(D2), scale=c)
    b.set_outputs([Bundle(a, (_out(p, 0),)), Bundle(c, (Bundle(inner, (_out(q, 0),)),))])
    return b.build()


def _mixed(first: object, second: object) -> ScalableDiagram:
    b = ScalableBuilder()
    a = b.add_node(first, [D2], [D2])  # type: ignore[arg-type]
    c = b.add_node(second, [D2], [D2])  # type: ignore[arg-type]
    b.add_wire(_out(a, 0), _in(c, 0))
    b.set_inputs([_in(a, 0)])
    b.set_outputs([_out(c, 0)])
    return b.build()


def _x_same_direction() -> ScalableDiagram:
    b = ScalableBuilder()
    a = b.add_node(X_SPIDER, [], [D2], PhaseVector(D2))
    c = b.add_node(X_SPIDER, [], [D2], PhaseVector(D2))
    b.add_wire(_out(a, 0), _out(c, 0))
    return b.build()


def _mismatch(*, phase: bool) -> ScalableDiagram:
    d, e = Dim.symbol("d"), Dim.symbol("e")
    b = ScalableBuilder()
    a = b.add_node(Z_SPIDER, [], [d, d if phase else e], PhaseVector(e) if phase else None)
    c = b.add_node(Z_SPIDER, [d], [])
    b.add_wire(_out(a, 0), _in(c, 0))
    b.set_outputs([_out(a, 1)])
    return b.build()


# -- randomized sweeps ----------------------------------------------------------------------

_SWEEP_MULTS = [Mult("k") + 1, Mult("k") * 2, Mult("a") + Mult("b"), Mult(2), Mult(1), Mult(0)]
_SWEEP_ENVS = [
    {"a": a, "b": b, "k": k, "k1": 1, "k2": b, "n": a, "m": 1, "d": 2, "alpha": 0}
    for a, b, k in ((0, 1, 1), (1, 0, 1), (1, 1, 0))
]


def _sweep_candidates(s: ScalableDiagram) -> list[tuple[str, dict[str, object]]]:
    found: list[tuple[str, dict[str, object]]] = []
    for sc in s.scales:
        found += [("dissolve", {"scale": sc.id}), ("kill_scale", {"scale": sc.id})]
        if sc.kind is ScaleKind.COPIES:
            for first in (1, Mult("k"), Mult("a"), Mult("k") + 1):
                found.append(("split_scale", {"scale": sc.id, "first": first}))
            found += [
                ("join_scales", {"first": sc.id, "second": other.id})
                for other in s.scales
                if other.id != sc.id and other.parent == sc.parent
            ]
    found += [("fuse_sheet", {"wire": w}) for w in sorted(s.wires, key=lambda w: w.sort_key())]
    found += [("enclose", {"nodes": (n.id,)}) for n in s.nodes]
    return found


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(BANG_BOX_FAMILIES))
def test_random_operation_sequences_preserve_the_oracle(family: str) -> None:
    rng = random.Random(family)
    for _trial in range(12):
        diagram = BANG_BOX_FAMILIES[family]()
        for box in sorted(diagram.bang_boxes):
            if not diagram.parameters:
                diagram.set_bang_box_multiplicity(box, rng.choice(_SWEEP_MULTS))
        s = to_scalable(diagram)
        for _round in range(6):
            name, kwargs = rng.choice(_sweep_candidates(s) or [("", {})])
            if not name:
                break
            try:
                step = SHEET_OPERATIONS[name](s, **kwargs)
            except SheetError:
                continue
            assert replay_sheet_step(step) == step
            before, after = from_scalable(step.before), from_scalable(step.after)
            for env in _SWEEP_ENVS:
                env = {k: v for k, v in env.items() if k not in s.parameters}
                assert _oracle_equal(before, after, env), (family, name, kwargs, env)
            s = step.after


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
def test_join_accepts_only_oracle_equal_subtrees(seed: int) -> None:
    rng = random.Random(seed)
    s = to_scalable(_with_mult(nested_port_in_copies(), 0, Mult("a") + Mult("b")))
    split = split_scale(s, S0, Mult("a"), dissolve_unit=False).after
    second = _last_scale(s) + 1
    variants = [
        split,
        replace(split, outputs=tuple(reversed(split.outputs))),
        replace(
            split,
            scales=tuple(
                replace(sc, multiplicity=Mult(rng.randint(0, 3))) if sc.parent == second else sc
                for sc in split.scales
            ),
        ),
        replace(
            split,
            nodes=tuple(
                replace(n, phase=PhaseVector(D2, {1: Phase.turns(sp.Rational(1, 2))}))
                if n.scale == second
                else n
                for n in split.nodes
            ),
        ),
    ]
    for variant in variants:
        try:
            step = join_scales(variant, S0, ScaleId(second))
        except SheetError:
            continue
        for env in ({"a": 1, "b": 2, "k2": 1}, {"a": 2, "b": 1, "k2": 2}):
            assert _oracle_equal(from_scalable(step.before), from_scalable(step.after), env)


# -- replay and registry ------------------------------------------------------------------------


class TestReplay:
    def test_registry_is_read_only_and_complete(self) -> None:
        assert sorted(SHEET_OPERATIONS) == [
            "dissolve",
            "enclose",
            "fuse_sheet",
            "join_scales",
            "kill_scale",
            "split_scale",
        ]
        with pytest.raises(TypeError):
            SHEET_OPERATIONS["x"] = dissolve  # type: ignore[index]

    def test_replay_detects_tampering(self) -> None:
        s = to_scalable(_with_mult(fusion_pair(), 0, Mult("k") + 1))
        step = split_scale(s, S0, Mult("k"))
        assert replay_sheet_step(step) == step
        tampered = replace(step, after=replace(step.after, scalar=Scalar.rational(2)))
        with pytest.raises(SheetGrammarError, match="does not reproduce"):
            replay_sheet_step(tampered)
        renamed = replace(step, operation="peel")
        with pytest.raises(SheetGrammarError, match="unknown operation"):
            replay_sheet_step(renamed)
        bad_args = replace(step, arguments=(("first", 1),))
        with pytest.raises(SheetGrammarError, match="bad arguments"):
            replay_sheet_step(bad_args)
        loose = replace(step, arguments=(("dissolve_unit", True), ("first", "k"), ("scale", 0)))
        with pytest.raises(SheetGrammarError, match="Mult or int"):
            replay_sheet_step(loose)
        unit = split_scale(s, S0, 1)
        raw = replace(unit, arguments=(("dissolve_unit", True), ("first", 1), ("scale", 0)))
        with pytest.raises(SheetGrammarError, match="not normalized"):
            replay_sheet_step(raw)
        with pytest.raises(SheetGrammarError, match="requires a SheetStep"):
            replay_sheet_step(step.after)  # type: ignore[arg-type]

    def test_scalable_first_family_round(self) -> None:
        s = sheet_family()
        step = split_scale(
            replace(
                s,
                scales=tuple(
                    replace(sc, multiplicity=Mult("k") + 1) if sc.id == 0 else sc for sc in s.scales
                ),
            ),
            S0,
            Mult("k"),
        )
        _assert_valid(step)


# -- determinism ----------------------------------------------------------------------------

_DETERMINISM_SCRIPT = """
from tests.test_sheet import PEELABLE, FUSION_CASES, _with_mult
from tests.test_scalable import BANG_BOX_FAMILIES
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.scalable import ScaleId, to_scalable
from archytaszx.rewrite.sheet import fuse_sheet, split_scale
def show(s):
    print(s.nodes, s.scales, s.inputs, s.outputs, sorted(w.sort_key() for w in s.wires), s.scalar)
for family, box in PEELABLE:
    s = to_scalable(_with_mult(BANG_BOX_FAMILIES[family](), box, Mult("n") + 1))
    show(split_scale(s, ScaleId(box), Mult("n")).after)
for name in sorted(FUSION_CASES):
    s = to_scalable(FUSION_CASES[name]())
    show(fuse_sheet(s, sorted(s.wires, key=lambda w: w.sort_key())[0]).after)
"""


def _run_with_seed(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    result = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_output_is_identical_across_hash_seeds() -> None:
    first = _run_with_seed("0")
    assert first
    assert _run_with_seed("2147483647") == first
