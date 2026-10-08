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

"""Tests for archytaszx.diagram.scalable: the notation, both translations, and strip.

G1: ``from_scalable(to_scalable(D))`` is family-equal to D (id for id when D is bundle-normal
with contiguous ids). G2: ``to_scalable(from_scalable(S)) == S.renumbered()``. G3:
``strip(to_scalable(D), a)`` is isomorphic to instantiating D at a with
:func:`~archytaszx.diagram.bangbox.instantiate_symbol`.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import (
    Mult,
    abstract_port_count,
    expand_concrete_boxes,
    free_mult_symbols,
    instantiate_symbol,
)
from archytaszx.diagram.compare import compare_structure, isomorphic
from archytaszx.diagram.generators import DIM_BINDER, X_SPIDER, Z_SPIDER, GeneratorType
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from archytaszx.diagram.scalable import (
    Bundle,
    ScalableBuilder,
    ScalableDiagram,
    ScalableDomainError,
    ScalableGrammarError,
    Scale,
    ScaledNode,
    ScaleId,
    ScaleKind,
    SheetPort,
    from_scalable,
    is_bundle_normal,
    strip,
    to_scalable,
    validate_scalable,
)
from archytaszx.diagram.validate import validate
from archytaszx.semantics.check import score

from .helpers import build_ghz_with_copy

D2 = Dim(2)
D3 = Dim(3)


def _in(node: NodeId, index: int) -> PortRef:
    return PortRef(node, Direction.INPUT, index)


def _out(node: NodeId, index: int) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def _spider(
    diagram: Diagram,
    ins: int,
    outs: int,
    dim: Dim = D2,
    generator: GeneratorType = Z_SPIDER,
    phase: PhaseVector | None = None,
) -> NodeId:
    return diagram.add_node(
        generator, [dim] * ins, [dim] * outs, phase=phase if phase is not None else PhaseVector(dim)
    )


# -- bang-box families ---------------------------------------------------------------------


def ghz() -> Diagram:
    d = Diagram()
    z = _spider(d, 0, 1)
    d.set_boundary_outputs([_out(z, 0)])
    d.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0)}))
    return d


def ghz_abstracted() -> Diagram:
    d = Diagram()
    z = _spider(d, 0, 1, D3)
    d.set_boundary_outputs([_out(z, 0)])
    d, _box, _n = abstract_port_count(d, _out(z, 0), 1, stem="n")
    return d


def multi_port() -> Diagram:
    d = Diagram()
    z = _spider(d, 1, 3)
    d.set_boundary_inputs([_in(z, 0)])
    d.set_boundary_outputs([_out(z, 0), _out(z, 1), _out(z, 2)])
    d.add_bang_box(Mult("n"), port_scope=frozenset({_in(z, 0), _out(z, 0), _out(z, 2)}))
    return d


def fusion_pair() -> Diagram:
    d, a, b = build_ghz_with_copy(D2)
    d.add_bang_box(Mult("m"), node_scope=frozenset({a, b}))
    return d


def nested_copies() -> Diagram:
    d = Diagram()
    a = _spider(d, 0, 2)
    b = _spider(d, 0, 1, D3, X_SPIDER)
    d.set_boundary_outputs([_out(a, 0), _out(b, 0), _out(a, 1)])
    outer = d.add_bang_box(Mult("k1"), node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("k2"), node_scope=frozenset({b}), parent=outer)
    return d


def nested_port_in_copies() -> Diagram:
    d = Diagram()
    z = _spider(d, 0, 1)
    d.set_boundary_outputs([_out(z, 0)])
    outer = d.add_bang_box(Mult("k1"), node_scope=frozenset({z}))
    d.add_bang_box(Mult("k2"), port_scope=frozenset({_out(z, 0)}), parent=outer)
    return d


def two_components() -> Diagram:
    d = Diagram()
    a = _spider(d, 1, 1)
    b = _spider(d, 0, 2, generator=X_SPIDER)
    x = _spider(d, 0, 1)
    d.set_boundary_inputs([_in(a, 0)])
    d.set_boundary_outputs([_out(a, 0), _out(b, 0), _out(b, 1), _out(x, 0)])
    d.add_bang_box(Mult("k"), node_scope=frozenset({a, b}))
    return d


def non_bundle_normal() -> Diagram:
    d = Diagram()
    a = _spider(d, 0, 1)
    x = _spider(d, 0, 1, generator=X_SPIDER)
    b = _spider(d, 0, 1)
    d.set_boundary_outputs([_out(a, 0), _out(x, 0), _out(b, 0)])
    d.add_bang_box(Mult("k"), node_scope=frozenset({a, b}))
    return d


def siblings_sharing_symbol() -> Diagram:
    d = Diagram()
    a = _spider(d, 0, 1)
    b = _spider(d, 0, 1, generator=X_SPIDER)
    d.set_boundary_outputs([_out(a, 0), _out(b, 0)])
    d.add_bang_box(Mult("k"), node_scope=frozenset({a}))
    d.add_bang_box(Mult("k"), node_scope=frozenset({b}))
    return d


def compound() -> Diagram:
    d = Diagram()
    a = _spider(d, 0, 2)
    b = _spider(d, 0, 1, generator=X_SPIDER)
    d.set_boundary_outputs([_out(a, 0), _out(a, 1), _out(b, 0)])
    d.add_bang_box(Mult("k") * 2, node_scope=frozenset({a}))
    d.add_bang_box(Mult("k") + 1, port_scope=frozenset({_out(b, 0)}))
    return d


def qufinite_mixed_dims() -> Diagram:
    d = Diagram()
    binder = d.add_node(DIM_BINDER, [D2, D3], [Dim(6)])
    sym = Dim.symbol("d")
    z = _spider(d, 0, 1, sym)
    d.set_boundary_inputs([_in(binder, 0), _in(binder, 1)])
    d.set_boundary_outputs([_out(z, 0), _out(binder, 0)])
    d.add_bang_box(Mult("k"), node_scope=frozenset({binder}))
    d.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0)}))
    return d


def symbolic_phase() -> Diagram:
    d = Diagram()
    z = _spider(d, 1, 2, D3, phase=PhaseVector(D3, {1: Phase.symbol("alpha")}))
    d.set_boundary_inputs([_in(z, 0)])
    d.set_boundary_outputs([_out(z, 0), _out(z, 1)])
    d.add_bang_box(Mult("k"), node_scope=frozenset({z}))
    d.multiply_scalar(Scalar.rational(3))
    return d


def plain() -> Diagram:
    d, _a, _b = build_ghz_with_copy(D2)
    return d


def emptied_by_children() -> Diagram:
    d = Diagram()
    a = _spider(d, 0, 1)
    b = _spider(d, 0, 1, generator=X_SPIDER)
    c = _spider(d, 0, 1)
    d.set_boundary_outputs([_out(a, 0), _out(b, 0), _out(c, 0)])
    outer = d.add_bang_box(Mult("k") * 2, node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("k"), node_scope=frozenset({a}), parent=outer)
    d.add_bang_box(Mult("j"), node_scope=frozenset({b}), parent=outer)
    return d


def inner_killed_first() -> Diagram:
    d = Diagram()
    a = _spider(d, 0, 1)
    x = _spider(d, 0, 1, generator=X_SPIDER)
    b = _spider(d, 0, 2)
    d.set_boundary_outputs([_out(b, 0), _out(x, 0), _out(a, 0), _out(b, 1)])
    outer = d.add_bang_box(Mult("k"), node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("j"), port_scope=frozenset({_out(b, 0)}), parent=outer)
    return d


def self_loop_fanned() -> Diagram:
    d = Diagram()
    z = _spider(d, 1, 2)
    d.add_wire(_out(z, 1), _in(z, 0))
    d.set_boundary_outputs([_out(z, 0)])
    d.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0)}))
    return d


def legs_sharing_a_symbol() -> Diagram:
    d = Diagram()
    z = _spider(d, 0, 3)
    d.set_boundary_outputs([_out(z, 0), _out(z, 1), _out(z, 2)])
    outer = d.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0), _out(z, 2)}))
    d.add_bang_box(Mult("n") + 1, port_scope=frozenset({_out(z, 2)}), parent=outer)
    return d


BANG_BOX_FAMILIES: dict[str, Callable[[], Diagram]] = {
    "ghz": ghz,
    "ghz_abstracted": ghz_abstracted,
    "multi_port": multi_port,
    "fusion_pair": fusion_pair,
    "nested_copies": nested_copies,
    "nested_port_in_copies": nested_port_in_copies,
    "two_components": two_components,
    "non_bundle_normal": non_bundle_normal,
    "siblings_sharing_symbol": siblings_sharing_symbol,
    "compound": compound,
    "qufinite_mixed_dims": qufinite_mixed_dims,
    "symbolic_phase": symbolic_phase,
    "plain": plain,
    "emptied_by_children": emptied_by_children,
    "inner_killed_first": inner_killed_first,
    "self_loop_fanned": self_loop_fanned,
    "legs_sharing_a_symbol": legs_sharing_a_symbol,
}

NOT_BUNDLE_NORMAL = frozenset({"non_bundle_normal", "inner_killed_first"})


# -- scalable-first families ----------------------------------------------------------------


def sheet_family() -> ScalableDiagram:
    b = ScalableBuilder()
    copies = b.add_scale(ScaleKind.COPIES, "k")
    inner_legs = b.add_scale(ScaleKind.LEGS, "m", parent=copies)
    top_legs = b.add_scale(ScaleKind.LEGS, "n")
    x = b.add_node(X_SPIDER, [D2], [D2, D2], PhaseVector(D2), scale=copies)
    z = b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2))
    b.set_fan(_in(x, 0), inner_legs)
    b.set_fan(_out(z, 0), top_legs)
    b.set_inputs([Bundle(copies, (_in(x, 0),))])
    b.set_outputs([Bundle(copies, (_out(x, 0), _out(x, 1))), _out(z, 0)])
    return b.build()


def legs_under_legs() -> ScalableDiagram:
    b = ScalableBuilder()
    outer = b.add_scale(ScaleKind.LEGS, "n")
    inner = b.add_scale(ScaleKind.LEGS, "m", parent=outer)
    z = b.add_node(Z_SPIDER, [], [D2, D2], PhaseVector(D2))
    b.set_fan(_out(z, 0), inner)
    b.set_fan(_out(z, 1), outer)
    b.set_outputs([_out(z, 0), _out(z, 1)])
    return b.build()


def nested_sheets() -> ScalableDiagram:
    b = ScalableBuilder()
    outer = b.add_scale(ScaleKind.COPIES, "a")
    inner = b.add_scale(ScaleKind.COPIES, "b", parent=outer)
    p = b.add_node(Z_SPIDER, [D2], [D2], PhaseVector(D2), scale=outer)
    q = b.add_node(Z_SPIDER, [D2], [D2], PhaseVector(D2), scale=inner)
    r = b.add_node(X_SPIDER, [D2], [], PhaseVector(D2), scale=inner)
    b.add_wire(_out(q, 0), _in(r, 0))
    b.set_inputs([Bundle(outer, (Bundle(inner, (_in(q, 0),)), _in(p, 0)))])
    b.set_outputs([Bundle(outer, (_out(p, 0),))])
    b.multiply_scalar(Scalar.rational(1, 2))
    b.bind_parameter("a", 2)
    return b.build()


def two_fans_on_one_node() -> ScalableDiagram:
    b = ScalableBuilder()
    left = b.add_scale(ScaleKind.LEGS, "n")
    right = b.add_scale(ScaleKind.LEGS, "m")
    z = b.add_node(Z_SPIDER, [], [D2, D2], PhaseVector(D2))
    b.set_fan(_out(z, 0), left)
    b.set_fan(_out(z, 1), right)
    b.set_outputs([_out(z, 0), _out(z, 1)])
    return b.build()


SCALABLE_FAMILIES: dict[str, Callable[[], ScalableDiagram]] = {
    "sheet_family": sheet_family,
    "legs_under_legs": legs_under_legs,
    "nested_sheets": nested_sheets,
    "two_fans_on_one_node": two_fans_on_one_node,
}


# -- helpers --------------------------------------------------------------------------------


def _instantiate(diagram: Diagram, assignment: Mapping[str, int]) -> Diagram:
    """Instantiate every bang box of ``diagram`` at ``assignment`` over its parameters."""
    env = {**diagram.parameters, **assignment}
    working = diagram
    for name in sorted(env):
        if name in free_mult_symbols(working):
            working = instantiate_symbol(working, name, env[name])
    return expand_concrete_boxes(working)


def _assignments(
    symbols: frozenset[str], values: tuple[int, ...] = (0, 1, 2)
) -> list[dict[str, int]]:
    names = sorted(symbols)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(values, repeat=len(names))
    ]


def _contiguous(diagram: Diagram) -> bool:
    return sorted(diagram.nodes) == list(range(len(diagram.nodes))) and sorted(
        diagram.bang_boxes
    ) == list(range(len(diagram.bang_boxes)))


# -- guarantees on bang-box families --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BANG_BOX_FAMILIES))
class TestBangBoxFamilies:
    def test_family_is_valid_and_translates_to_a_valid_scalable_diagram(self, name: str) -> None:
        diagram = BANG_BOX_FAMILIES[name]()
        assert validate(diagram).is_valid
        assert validate_scalable(to_scalable(diagram)) == ()

    def test_g1_round_trip_is_family_equal(self, name: str) -> None:
        diagram = BANG_BOX_FAMILIES[name]()
        back = from_scalable(to_scalable(diagram))
        for assignment in _assignments(free_mult_symbols(diagram)):
            assert isomorphic(_instantiate(back, assignment), _instantiate(diagram, assignment))

    def test_g1_round_trip_is_identical_when_bundle_normal(self, name: str) -> None:
        diagram = BANG_BOX_FAMILIES[name]()
        assert _contiguous(diagram)
        assert is_bundle_normal(diagram) is (name not in NOT_BUNDLE_NORMAL)
        comparison = compare_structure(from_scalable(to_scalable(diagram)), diagram)
        assert comparison.identical is (name not in NOT_BUNDLE_NORMAL), comparison.reason

    def test_g2_translation_back_and_forth_is_renumbering(self, name: str) -> None:
        s = to_scalable(BANG_BOX_FAMILIES[name]())
        assert to_scalable(from_scalable(s)) == s.renumbered()
        assert s.renumbered() == s

    def test_g3_strip_matches_instantiation(self, name: str) -> None:
        diagram = BANG_BOX_FAMILIES[name]()
        s = to_scalable(diagram)
        for assignment in _assignments(free_mult_symbols(diagram), (0, 1, 2, 3)):
            stripped = strip(s, assignment)
            assert not stripped.bang_boxes
            assert isomorphic(stripped, _instantiate(diagram, assignment)), assignment


@pytest.mark.parametrize("name", sorted(SCALABLE_FAMILIES))
class TestScalableFirstFamilies:
    def test_g2(self, name: str) -> None:
        s = SCALABLE_FAMILIES[name]()
        assert validate_scalable(s) == ()
        assert to_scalable(from_scalable(s)) == s.renumbered()

    def test_g1_on_the_bang_box_form(self, name: str) -> None:
        diagram = from_scalable(SCALABLE_FAMILIES[name]())
        assert validate(diagram).is_valid
        assert is_bundle_normal(diagram)
        back = from_scalable(to_scalable(diagram))
        assert compare_structure(back, diagram).identical

    def test_g3_strip_matches_instantiation(self, name: str) -> None:
        s = SCALABLE_FAMILIES[name]()
        diagram = from_scalable(s)
        for assignment in _assignments(s.free_mult_symbols()):
            expected = _instantiate(diagram, assignment)
            assert isomorphic(strip(s, assignment), expected), assignment
            assert isomorphic(strip(to_scalable(diagram), assignment), expected), assignment


class TestOracle:
    @pytest.mark.parametrize("name", ["ghz", "fusion_pair", "nested_port_in_copies", "compound"])
    def test_strip_and_instantiation_contract_to_the_same_tensor(self, name: str) -> None:
        diagram = BANG_BOX_FAMILIES[name]()
        s = to_scalable(diagram)
        for assignment in _assignments(free_mult_symbols(diagram)):
            via_strip = score(strip(s, assignment), {}).tensor
            via_box = score(diagram, assignment).tensor
            assert np.allclose(via_strip, via_box), assignment


# -- types and widths ------------------------------------------------------------------------


class TestTypesAndWidths:
    def test_ghz_has_output_type_n(self) -> None:
        s = to_scalable(ghz())
        assert s.output_type() == (Mult("n"),)
        assert s.input_type() == ()
        (node,) = s.nodes
        assert node.outputs[0].fan == ScaleId(0)
        assert s.scales == (Scale(ScaleId(0), ScaleKind.LEGS, Mult("n"), None),)

    def test_a_node_scope_box_crossing_two_outputs_is_one_bundle_of_width_2k(self) -> None:
        d = Diagram()
        z = _spider(d, 0, 2)
        d.set_boundary_outputs([_out(z, 0), _out(z, 1)])
        d.add_bang_box(Mult("k"), node_scope=frozenset({z}))
        s = to_scalable(d)
        assert s.outputs == (Bundle(ScaleId(0), (_out(z, 0), _out(z, 1))),)
        assert s.output_type() == (Mult("k") * 2,)

    def test_fusion_pair_widths(self) -> None:
        s = to_scalable(fusion_pair())
        assert s.output_type() == (Mult("m") * 3,)
        (wire,) = s.wires
        assert s.wire_width(wire) == Mult("m")
        assert s.size(None) == Mult(1)

    def test_nested_copies_sizes_and_bundle_tree(self) -> None:
        s = to_scalable(nested_copies())
        k1, k2 = Mult("k1"), Mult("k2")
        assert s.size(ScaleId(1)) == k1 * k2
        assert s.outputs == (
            Bundle(
                ScaleId(0),
                (_out(NodeId(0), 0), Bundle(ScaleId(1), (_out(NodeId(1), 0),)), _out(NodeId(0), 1)),
            ),
        )
        assert s.output_type() == (k1 * (k2 + 2),)

    def test_nested_port_in_copies_width(self) -> None:
        s = to_scalable(nested_port_in_copies())
        assert s.output_type() == (Mult("k1") * Mult("k2"),)
        assert s.fan_width(ScaleId(1)) == Mult("k2")

    def test_compound_multiplicities(self) -> None:
        s = to_scalable(compound())
        k = Mult("k")
        assert s.output_type() == (k * 4, k + 1)
        assert s.free_mult_symbols() == frozenset({"k"})

    def test_qufinite_input_bundle(self) -> None:
        s = to_scalable(qufinite_mixed_dims())
        assert s.input_type() == (Mult("k") * 2,)
        assert s.output_type() == (Mult("n"), Mult("k"))

    def test_scalable_first_types(self) -> None:
        k, m, n = Mult("k"), Mult("m"), Mult("n")
        s = sheet_family()
        assert s.input_type() == (k * m,)
        assert s.output_type() == (k * 2, n)
        nested = legs_under_legs()
        assert nested.output_type() == (n * m, n)
        assert nested.fan_width(ScaleId(1)) == n * m
        sheets = nested_sheets()
        a, b = Mult("a"), Mult("b")
        assert sheets.input_type() == (a * (b + 1),)
        assert sheets.output_type() == (a,)

    def test_plain_diagram_has_no_scales(self) -> None:
        s = to_scalable(plain())
        assert s.scales == ()
        assert s.output_type() == (Mult(1),) * 3
        assert all(isinstance(item, PortRef) for item in s.outputs)


# -- value semantics -------------------------------------------------------------------------


class TestValueSemantics:
    def test_equality_and_hash_are_structural(self) -> None:
        a, b = to_scalable(fusion_pair()), to_scalable(fusion_pair())
        assert a == b and hash(a) == hash(b)
        d = fusion_pair()
        d.multiply_scalar(Scalar.rational(2))
        assert to_scalable(d) != a
        assert "ScalableDiagram(nodes=2" in repr(a)

    def test_parameters_are_read_only_and_sorted(self) -> None:
        s = nested_sheets()
        assert list(s.parameters) == ["a"]
        with pytest.raises(TypeError):
            s.parameters["b"] = 1  # type: ignore[index]

    def test_renumbered_compacts_ids(self) -> None:
        d = Diagram()
        dead = _spider(d, 0, 0)
        z = _spider(d, 0, 2)
        dead_box = d.add_bang_box(Mult("j"), node_scope=frozenset({dead}))
        d.remove_bang_box(dead_box)
        d.remove_node(dead)
        d.set_boundary_outputs([_out(z, 0), _out(z, 1)])
        d.add_bang_box(Mult("k"), port_scope=frozenset({_out(z, 1)}))
        s = to_scalable(d)
        assert [n.id for n in s.nodes] == [1] and [sc.id for sc in s.scales] == [1]
        r = s.renumbered()
        assert [n.id for n in r.nodes] == [0] and [sc.id for sc in r.scales] == [0]
        assert r.nodes[0].outputs[1].fan == ScaleId(0)
        assert r.outputs == (_out(NodeId(0), 0), _out(NodeId(0), 1))
        assert r != s
        assert to_scalable(from_scalable(s)) == r
        assert isomorphic(from_scalable(s), d)

    def test_lookups_refuse_unknown_ids(self) -> None:
        s = to_scalable(ghz())
        with pytest.raises(ScalableGrammarError):
            s.node(NodeId(7))
        with pytest.raises(ScalableGrammarError):
            s.scale(ScaleId(7))

    def test_duplicate_ids_are_refused(self) -> None:
        node = ScaledNode(NodeId(0), Z_SPIDER, (), (), PhaseVector(D2))
        with pytest.raises(ScalableGrammarError):
            ScalableDiagram((node, node), frozenset(), (), (), ())


# -- strip -----------------------------------------------------------------------------------


class TestStrip:
    def test_strip_falls_back_to_parameters_and_drops_consumed_symbols(self) -> None:
        diagram = ghz_abstracted()
        assert dict(diagram.parameters) == {"n": 1}
        stripped = strip(to_scalable(diagram))
        assert dict(stripped.parameters) == {}
        assert isomorphic(stripped, instantiate_symbol(diagram, "n", 1))

    def test_an_inner_symbol_under_a_zero_count_is_not_needed(self) -> None:
        s = to_scalable(nested_port_in_copies())
        assert not strip(s, {"k1": 0}).nodes
        with pytest.raises(ScalableDomainError):
            strip(s, {"k1": 1})

    def test_unresolved_and_out_of_domain_assignments_are_refused(self) -> None:
        s = to_scalable(ghz())
        with pytest.raises(ScalableDomainError):
            strip(s)
        with pytest.raises(ScalableDomainError):
            strip(s, {"n": -1})

    def test_zero_fan_removes_the_port_and_shifts_later_indices(self) -> None:
        stripped = strip(to_scalable(multi_port()), {"n": 0})
        (node,) = stripped.nodes.values()
        assert len(node.inputs) == 0 and len(node.outputs) == 1
        assert stripped.boundary_outputs == (_out(node.id, 0),)

    def test_copy_major_layout(self) -> None:
        stripped = strip(to_scalable(two_components()), {"k": 2})
        names = [stripped.nodes[r.node_id].generator_type.name for r in stripped.boundary_outputs]
        assert names == ["Z", "X", "X", "Z", "X", "X", "Z"]

    def test_strip_of_an_invalid_diagram_is_refused(self) -> None:
        node = ScaledNode(NodeId(0), Z_SPIDER, (), (SheetPort(D2),), PhaseVector(D2))
        with pytest.raises(ScalableGrammarError):
            strip(ScalableDiagram((node,), frozenset(), (), (), ()))


# -- refusals --------------------------------------------------------------------------------


def _node(
    node_id: int,
    outs: int = 1,
    *,
    ins: int = 0,
    scale: int | None = None,
    fans: Mapping[int, int] | None = None,
) -> ScaledNode:
    fans = fans or {}
    return ScaledNode(
        NodeId(node_id),
        Z_SPIDER,
        tuple(SheetPort(D2) for _ in range(ins)),
        tuple(SheetPort(D2, None if i not in fans else ScaleId(fans[i])) for i in range(outs)),
        PhaseVector(D2),
        None if scale is None else ScaleId(scale),
    )


def _scale(scale_id: int, kind: ScaleKind, parent: int | None = None) -> Scale:
    return Scale(
        ScaleId(scale_id), kind, Mult(f"s{scale_id}"), None if parent is None else ScaleId(parent)
    )


C, L = ScaleKind.COPIES, ScaleKind.LEGS
R0 = _out(NodeId(0), 0)
R1 = _out(NodeId(1), 0)

INVALID: dict[str, tuple[ScalableDiagram, str]] = {
    "unknown_parent": (
        ScalableDiagram(
            (_node(0, scale=0),), frozenset(), (_scale(0, C, 5),), (), (Bundle(ScaleId(0), (R0,)),)
        ),
        "unknown parent",
    ),
    "cycle": (
        ScalableDiagram((_node(0),), frozenset(), (_scale(0, C, 1), _scale(1, C, 0)), (), (R0,)),
        "cyclic",
    ),
    "copies_under_legs": (
        ScalableDiagram(
            (_node(0, scale=1, fans={0: 0}),),
            frozenset(),
            (_scale(0, L), _scale(1, C, 0)),
            (),
            (Bundle(ScaleId(1), (R0,)),),
        ),
        "has LEGS parent",
    ),
    "node_in_legs_scale": (
        ScalableDiagram((_node(0, scale=0, fans={0: 0}),), frozenset(), (_scale(0, L),), (), (R0,)),
        "is not a COPIES scale",
    ),
    "fan_is_copies": (
        ScalableDiagram((_node(0, fans={0: 0}),), frozenset(), (_scale(0, C),), (), (R0,)),
        "is not a LEGS scale",
    ),
    "fan_outside_node_scale": (
        ScalableDiagram(
            (_node(0, scale=0, fans={0: 1}),),
            frozenset(),
            (_scale(0, C), _scale(1, L)),
            (),
            (Bundle(ScaleId(0), (R0,)),),
        ),
        "lies outside the scale",
    ),
    "legs_child_not_strict": (
        ScalableDiagram(
            (_node(0, fans={0: 1}),), frozenset(), (_scale(0, L), _scale(1, L, 0)), (), (R0,)
        ),
        "strictly fewer ports",
    ),
    "port_unused": (ScalableDiagram((_node(0),), frozenset(), (), (), ()), "used 0 times"),
    "port_used_twice": (
        ScalableDiagram((_node(0),), frozenset(), (), (), (R0, R0)),
        "used 2 times",
    ),
    "unresolvable_ref": (
        ScalableDiagram((_node(0),), frozenset(), (), (), (R0, R1)),
        "does not resolve",
    ),
    "fanned_port_wired": (
        ScalableDiagram(
            (_node(0, fans={0: 0}), _node(1, 0, ins=1)),
            frozenset({Wire(R0, _in(NodeId(1), 0))}),
            (_scale(0, L),),
            (),
            (),
        ),
        "fanned port",
    ),
    "wire_across_scales": (
        ScalableDiagram(
            (_node(0, scale=0), _node(1, 0, ins=1)),
            frozenset({Wire(R0, _in(NodeId(1), 0))}),
            (_scale(0, C),),
            (),
            (),
        ),
        "different scales",
    ),
    "empty_legs": (
        ScalableDiagram((_node(0),), frozenset(), (_scale(0, L),), (), (R0,)),
        "covers no port",
    ),
    "empty_copies": (
        ScalableDiagram((_node(0),), frozenset(), (_scale(0, C),), (), (R0,)),
        "contains no node",
    ),
    "ref_outside_its_bundle": (
        ScalableDiagram((_node(0, scale=0),), frozenset(), (_scale(0, C),), (), (R0,)),
        "sits in bundles",
    ),
    "unscaled_ref_in_a_bundle": (
        ScalableDiagram(
            (_node(0, scale=0), _node(1)),
            frozenset(),
            (_scale(0, C),),
            (),
            (Bundle(ScaleId(0), (R0, R1)),),
        ),
        "sits in bundles",
    ),
    "two_bundles_one_scale": (
        ScalableDiagram(
            (_node(0, 2, scale=0),),
            frozenset(),
            (_scale(0, C),),
            (),
            (Bundle(ScaleId(0), (R0,)), Bundle(ScaleId(0), (_out(NodeId(0), 1),))),
        ),
        "not at most one",
    ),
    "empty_bundle": (
        ScalableDiagram(
            (_node(0, scale=0),),
            frozenset(),
            (_scale(0, C),),
            (),
            (Bundle(ScaleId(0), (R0,)), Bundle(ScaleId(0), ())),
        ),
        "is empty",
    ),
    "bundle_of_legs_scale": (
        ScalableDiagram(
            (_node(0, fans={0: 0}),), frozenset(), (_scale(0, L),), (), (Bundle(ScaleId(0), (R0,)),)
        ),
        "bundle scale 0 is not a COPIES scale",
    ),
    "output_port_on_input_side": (
        ScalableDiagram((_node(0),), frozenset(), (), (R0,), ()),
        "bang-box form",
    ),
}


class TestRefusals:
    @pytest.mark.parametrize("name", sorted(INVALID))
    def test_validate_scalable_reports_the_problem(self, name: str) -> None:
        s, fragment = INVALID[name]
        issues = validate_scalable(s)
        assert any(fragment in issue for issue in issues), issues

    def test_build_with_check_raises_and_without_check_returns(self) -> None:
        b = ScalableBuilder()
        b.add_node(Z_SPIDER, [], [D2], PhaseVector(D2))
        with pytest.raises(ScalableGrammarError, match="used 0 times"):
            b.build()
        assert validate_scalable(b.build(check=False))

    def test_builder_input_errors(self) -> None:
        b = ScalableBuilder()
        with pytest.raises(ScalableDomainError):
            b.add_scale(ScaleKind.COPIES, -1)
        with pytest.raises(ScalableGrammarError):
            b.add_scale(ScaleKind.COPIES, "not an identifier")
        with pytest.raises(ScalableGrammarError):
            b.set_fan(R0, ScaleId(0))
        with pytest.raises(ScalableGrammarError):
            b.add_wire(R0, R0)
        with pytest.raises(ScalableGrammarError):
            b.bind_parameter("x", True)
        with pytest.raises(ScalableGrammarError):
            b.set_outputs([3])  # type: ignore[list-item]
        with pytest.raises(ScalableGrammarError):
            b.multiply_scalar(2)  # type: ignore[arg-type]

    def test_to_scalable_refuses_a_bang_box_validate_error(self) -> None:
        d = Diagram()
        z = _spider(d, 0, 1)
        d.set_boundary_outputs([_out(z, 0)])
        d.add_bang_box(Mult("a"), node_scope=frozenset({z}))
        d.add_bang_box(Mult("b"), node_scope=frozenset({z}))
        with pytest.raises(ScalableGrammarError, match="overlap"):
            to_scalable(d)

    def test_to_scalable_refuses_a_node_scope_box_under_a_port_scope_box(self) -> None:
        d = Diagram()
        a = _spider(d, 0, 1)
        b = _spider(d, 0, 1)
        d.set_boundary_outputs([_out(a, 0), _out(b, 0)])
        legs = d.add_bang_box(Mult("n"), port_scope=frozenset({_out(a, 0), _out(b, 0)}))
        d.add_bang_box(Mult("k"), node_scope=frozenset({a}), parent=legs)
        assert validate(d).is_valid
        with pytest.raises(ScalableGrammarError, match="LEGS parent"):
            to_scalable(d)

    def test_to_scalable_refuses_a_wire_crossing_a_node_scope_box(self) -> None:
        d = Diagram()
        a = _spider(d, 0, 1)
        b = _spider(d, 1, 0)
        d.add_wire(_out(a, 0), _in(b, 0))
        d.add_bang_box(Mult("k"), node_scope=frozenset({a}))
        with pytest.raises(ScalableGrammarError, match="crosses"):
            to_scalable(d)

    def test_to_scalable_refuses_an_unresolvable_ref(self) -> None:
        d = Diagram()
        d.set_boundary_outputs([R0])
        with pytest.raises(ScalableGrammarError):
            to_scalable(d)

    def test_wrong_argument_types(self) -> None:
        with pytest.raises(ScalableGrammarError):
            to_scalable(to_scalable(ghz()))  # type: ignore[arg-type]
        with pytest.raises(ScalableGrammarError):
            from_scalable(ghz())  # type: ignore[arg-type]
        with pytest.raises(ScalableGrammarError):
            Bundle(ScaleId(0), (1,))  # type: ignore[arg-type]

    def test_malformed_constructor_arguments_raise_grammar_errors(self) -> None:
        with pytest.raises(ScalableGrammarError):
            ScalableDiagram(("x",), frozenset(), (), (), ())  # type: ignore[arg-type]
        with pytest.raises(ScalableGrammarError):
            ScalableDiagram((), frozenset(), ("x",), (), ())  # type: ignore[arg-type]
        with pytest.raises(ScalableGrammarError):
            ScalableDiagram((), frozenset(), (), (), (), parameters={"b": 1, 2: 1})  # type: ignore[dict-item]
        dangling = ScalableDiagram((), frozenset(), (), (Bundle(ScaleId(3), ()),), ())
        with pytest.raises(ScalableGrammarError):
            dangling.renumbered()


# -- determinism -----------------------------------------------------------------------------

_DETERMINISM_SCRIPT = """
from tests.test_scalable import BANG_BOX_FAMILIES, INVALID
from archytaszx.diagram.scalable import strip, to_scalable, validate_scalable
for name in sorted(BANG_BOX_FAMILIES):
    s = to_scalable(BANG_BOX_FAMILIES[name]())
    print(name, s.nodes, s.scales, s.inputs, s.outputs, sorted(w.sort_key() for w in s.wires))
    a = {sym: 2 for sym in sorted(s.free_mult_symbols())}
    d = strip(s, a)
    print(d.boundary_inputs, d.boundary_outputs, sorted(w.sort_key() for w in d.wires))
for name in sorted(INVALID):
    print(name, validate_scalable(INVALID[name][0]))
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
