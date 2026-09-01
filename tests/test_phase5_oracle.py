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

"""Phase 5 numeric-oracle check: the graph-to-fuse-to-graph half of this phase's completion
condition, verified.

Per the spec, every build phase ends with a numeric oracle check, and per the build
plan Phase 5 is done when "the full path Dirac to graph to fuse to graph runs and the
oracle confirms exact equality" -- de-risking the whole project. This module exercises the
graph-to-fuse-to-graph half of that: build "A into B" via ``build_ghz_with_copy``, find the
fusion match, apply it, and confirm the oracle (:mod:`qufzx.semantics.check`) reports the
pre- and post-fusion diagrams exactly equal at several concrete ``d``, with no symbolic
phase, with a symbolic phase on A, on B, and on both, and for both the Z and X spider
colors -- the X case exists specifically to exercise the
``consumed_wire_direction_permitted_for_color`` side condition documented in
:mod:`qufzx.rewrite.match`, since only X's non-diagonal denotation can tell a correct
fusion apart from a wrongly-wired one. It also carries the negative controls and import-
boundary check the build plan calls out explicitly.

The *Dirac* half of the phrase -- parsing a Dirac-notation expression into this graph in
the first place -- is not exercised here, and cannot honestly be claimed done: there is no
Dirac representation, parser, or printer in this codebase yet.
:mod:`qufzx.repl.parser`/:mod:`qufzx.repl.printer` are license-header-only skeletons, owned
by Phases 18 and 17 respectively, not this one. ``build_ghz_with_copy`` (in
``tests/helpers.py``) builds the graph by hand, standing in for what a Dirac parser will
one day produce. This is a deliberate, stated deferral to those later phases, not an
oversight -- see ``README.md``'s "Current state" section for the same statement kept in
sync.
"""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import EqualityMode, compare

from .helpers import build_ghz_with_copy

_CONCRETE_DS = (2, 3, 5)
_CONCRETE_PHASES = (sp.Integer(0), sp.Rational(1, 3), sp.Rational(2, 5))


def _fuse(diagram: Diagram) -> Diagram:
    matches = find_matches(diagram)
    assert len(matches) == 1, "expected exactly one fusion match on the A-into-B example"
    return apply(diagram, SPIDER_FUSION, matches[0]).diagram


class TestSpiderFusionOracleZ:
    def test_fuses_to_a_single_spider_with_expected_leg_count(self) -> None:
        d = Dim.symbol("d")
        pre, _a, _b = build_ghz_with_copy(d)
        post = _fuse(pre)
        assert len(post.nodes) == 1
        (merged,) = post.nodes.values()
        assert merged.num_inputs == 0
        assert merged.num_outputs == 3

    def test_exact_equality_at_several_concrete_d(self) -> None:
        d = Dim.symbol("d")
        pre, _a, _b = build_ghz_with_copy(d)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        d = Dim.concrete(3)
        pre, _a, _b = build_ghz_with_copy(d)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred


class TestSpiderFusionOracleX:
    """Repeats the Z-color checks for X, exercising the output->input wire-direction condition."""

    def test_fuses_to_a_single_spider_with_expected_leg_count(self) -> None:
        d = Dim.symbol("d")
        pre, _a, _b = build_ghz_with_copy(d, generator_type=X_SPIDER)
        post = _fuse(pre)
        assert len(post.nodes) == 1
        (merged,) = post.nodes.values()
        assert merged.generator_type is X_SPIDER
        assert merged.num_inputs == 0
        assert merged.num_outputs == 3

    def test_exact_equality_at_several_concrete_d(self) -> None:
        d = Dim.symbol("d")
        pre, _a, _b = build_ghz_with_copy(d, generator_type=X_SPIDER)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        d = Dim.concrete(3)
        pre, _a, _b = build_ghz_with_copy(d, generator_type=X_SPIDER)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred


def _build_z_output_to_output(dim: Dim) -> Diagram:
    """Two Z spiders joined by an OUTPUT-OUTPUT wire, with one surviving boundary output.

    Regression coverage for the Phase 5 final fix round's Step 4 decision (see match.py's
    module docstring, condition 4): a Z-Z wire of any direction is valid fusion, since Z's
    tensor is diagonal in every axis and contraction never conjugates -- unlike X, where
    only an OUTPUT-to-INPUT wire is fusion (see :class:`TestSpiderFusionOracleX`).
    """
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.OUTPUT, 0))
    diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])
    return diagram


def _build_z_input_to_input(dim: Dim) -> Diagram:
    """Two single-leg Z spiders joined by an INPUT-INPUT wire; both legs are consumed.

    No leg survives, so the merged node is legless (a bare scalar diagram, per
    :func:`qufzx.semantics.denote._z_tensor`'s ``rank == 0`` reading) and there is no
    boundary at all, before or after fusion.
    """
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(a_id, Direction.INPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    return diagram


class TestSpiderFusionOracleZSameDirection:
    """Z-Z fusion across a same-direction wire -- see the ``_build_z_*`` helpers above."""

    def test_output_to_output_exact_equality_at_several_concrete_d(self) -> None:
        d = Dim.symbol("d")
        pre = _build_z_output_to_output(d)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_input_to_input_exact_equality_at_several_concrete_d(self) -> None:
        d = Dim.symbol("d")
        pre = _build_z_input_to_input(d)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        d = Dim.concrete(3)
        pre = _build_z_output_to_output(d)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred


def _build_parallel_wire_pair(
    dim: Dim, generator_type: GeneratorType, *, same_direction_leftover: bool
) -> tuple[Diagram, Wire]:
    """A: ``0->2``, B fusable to it by one consumed OUTPUT-INPUT wire plus a leftover wire.

    The leftover wire is alternating (OUTPUT-INPUT) when ``same_direction_leftover`` is
    False, or same-direction (OUTPUT-OUTPUT) when True -- condition 4 restricts only the
    *consumed* wire (always OUTPUT-INPUT here, so this shape is legal fusion for X too).
    Returns the diagram and the consumed wire, so a caller can pick out (of the two matches
    this pair now yields) the one that fuses across it.
    """
    diagram = Diagram()
    a_id = diagram.add_node(generator_type, input_dims=[], output_dims=[dim, dim])
    if same_direction_leftover:
        b_id = diagram.add_node(generator_type, input_dims=[dim], output_dims=[dim])
        leftover = (PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.OUTPUT, 0))
    else:
        b_id = diagram.add_node(generator_type, input_dims=[dim, dim], output_dims=[])
        leftover = (PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
    consumed = (PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.add_wire(*consumed)
    diagram.add_wire(*leftover)
    return diagram, Wire(*consumed)


class TestParallelWireFusionOracle:
    """Parallel-wire fusion is exactly equal pre/post at d = 2, 3, 5, for both
    colors and both leftover-wire shapes, and the merged node actually carries the
    leftover wire as a self-loop.
    """

    def _run(self, generator_type: GeneratorType, *, same_direction_leftover: bool) -> None:
        d = Dim.symbol("d")
        pre, consumed_wire = _build_parallel_wire_pair(
            d, generator_type, same_direction_leftover=same_direction_leftover
        )
        matches = find_matches(pre)
        # X's leftover wire is itself a candidate consumed wire too -- unless it is
        # same-direction, which condition 4 refuses as a *consumed* wire for X (it is
        # still fine as a mere self-loop leftover of the other candidate).
        expected = 1 if generator_type is X_SPIDER and same_direction_leftover else 2
        assert len(matches) == expected
        match = next(m for m in matches if m.wire == consumed_wire)
        post = apply(pre, SPIDER_FUSION, match).diagram
        (merged_id,) = post.nodes.keys()
        assert any(
            wire.a.node_id == merged_id and wire.b.node_id == merged_id for wire in post.wires
        ), "the leftover wire must survive as a self-loop on the merged node"
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_z_alternating_leftover(self) -> None:
        self._run(Z_SPIDER, same_direction_leftover=False)

    def test_z_same_direction_leftover(self) -> None:
        self._run(Z_SPIDER, same_direction_leftover=True)

    def test_x_alternating_leftover(self) -> None:
        self._run(X_SPIDER, same_direction_leftover=False)

    def test_x_same_direction_leftover(self) -> None:
        self._run(X_SPIDER, same_direction_leftover=True)


def _build_all_legs_consumed(dim: Dim, generator_type: GeneratorType) -> Diagram:
    """A: ``0->1``, B: ``1->0``, wired output-to-input -- fusion consumes every leg of both.

    The corner case from :mod:`qufzx.rewrite.rules_library`'s module docstring ("See
    :func:`_merged_phase` for the legless corner case, where dimension can only survive
    via the phase slot") and from :func:`~qufzx.rewrite.rules_library._merged_phase`'s own
    docstring: the merged node ends up with zero inputs and zero outputs, so its
    dimension can only survive via an explicit zero phase.
    """
    diagram = Diagram()
    a_id = diagram.add_node(generator_type, input_dims=[], output_dims=[dim])
    b_id = diagram.add_node(generator_type, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    return diagram


class TestSpiderFusionOracleAllLegsConsumed:
    """The all-legs-consumed corner case, for both colors: the fix under test."""

    def test_z_merged_node_has_no_legs(self) -> None:
        pre = _build_all_legs_consumed(Dim.symbol("d"), Z_SPIDER)
        post = _fuse(pre)
        assert len(post.nodes) == 1
        (merged,) = post.nodes.values()
        assert merged.num_inputs == 0
        assert merged.num_outputs == 0

    def test_z_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre = _build_all_legs_consumed(Dim.concrete(3), Z_SPIDER)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_z_exact_equality_at_several_concrete_d(self) -> None:
        pre = _build_all_legs_consumed(Dim.symbol("d"), Z_SPIDER)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason

    def test_x_merged_node_has_no_legs(self) -> None:
        pre = _build_all_legs_consumed(Dim.symbol("d"), X_SPIDER)
        post = _fuse(pre)
        assert len(post.nodes) == 1
        (merged,) = post.nodes.values()
        assert merged.generator_type is X_SPIDER
        assert merged.num_inputs == 0
        assert merged.num_outputs == 0

    def test_x_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre = _build_all_legs_consumed(Dim.concrete(3), X_SPIDER)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_x_exact_equality_at_several_concrete_d(self) -> None:
        pre = _build_all_legs_consumed(Dim.symbol("d"), X_SPIDER)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason


def _build_preexisting_self_loop_on_a(
    dim: Dim, generator_type: GeneratorType = Z_SPIDER
) -> tuple[Diagram, NodeId]:
    """A carries its own self-loop (two distinct ports wired to each other) plus the fusion wire.

    A: 1 input, 3 outputs. ``A_in0`` is wired to ``A_out2`` -- a self-loop entirely on A,
    pre-existing before any rewrite. ``A_out0`` is the fusion wire into B. ``A_out1`` and
    B's own output are the boundary. Both self-loop endpoints are on A, a consumed node, and
    neither is the fusion pattern's own wire, so this exercises exactly the case the engine
    module docstring calls out ("a pre-existing self-loop on one of the consumed nodes, both
    endpoints get remapped, yielding a self-loop on the merged node") without any test ever
    having built it. ``generator_type`` defaults to ``Z_SPIDER``; the X-color subclass below
    passes ``X_SPIDER`` to exercise the same shape through the
    consumed_wire_direction_permitted_for_color side condition.
    """
    diagram = Diagram()
    a_id = diagram.add_node(generator_type, input_dims=[dim], output_dims=[dim, dim, dim])
    b_id = diagram.add_node(generator_type, input_dims=[dim], output_dims=[dim])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.add_wire(PortRef(a_id, Direction.INPUT, 0), PortRef(a_id, Direction.OUTPUT, 2))
    diagram.set_boundary_outputs(
        [PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.OUTPUT, 0)]
    )
    return diagram, a_id


class _PreexistingSelfLoopCases:
    """The self-loop case the engine docstring claims works, shared across both colors."""

    generator_type: GeneratorType

    def test_self_loop_survives_as_a_self_loop_on_the_merged_node(self) -> None:
        pre, a_id = _build_preexisting_self_loop_on_a(Dim.symbol("d"), self.generator_type)
        post = _fuse(pre)
        self_loops = [wire for wire in post.wires if wire.a.node_id == wire.b.node_id]
        assert len(self_loops) == 1
        (loop,) = self_loops
        assert loop.a.node_id in post.nodes
        assert a_id not in post.nodes

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre, _a_id = _build_preexisting_self_loop_on_a(Dim.concrete(3), self.generator_type)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_exact_equality_at_several_concrete_d(self) -> None:
        pre, _a_id = _build_preexisting_self_loop_on_a(Dim.symbol("d"), self.generator_type)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason


class TestSpiderFusionPreservesAPreexistingSelfLoopZ(_PreexistingSelfLoopCases):
    generator_type = Z_SPIDER


class TestSpiderFusionPreservesAPreexistingSelfLoopX(_PreexistingSelfLoopCases):
    generator_type = X_SPIDER


def _build_boundary_input_and_output(dim: Dim) -> tuple[Diagram, NodeId, NodeId]:
    """A's only input is a boundary input; every other engine/oracle test uses output-only.

    A: 1 input, 1 output. B: 1 input, 0 outputs. ``A_out0 -> B_in0`` is the fusion wire;
    ``A_in0`` survives fusion and is the sole boundary input.
    """
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
    return diagram, a_id, b_id


class TestSpiderFusionRemapsABoundaryInput:
    """Every prior engine/oracle test used output-only boundaries; this covers an input."""

    def test_boundary_input_is_remapped_to_the_merged_node(self) -> None:
        pre, a_id, _b_id = _build_boundary_input_and_output(Dim.concrete(2))
        post = _fuse(pre)
        assert len(post.boundary_inputs) == 1
        (remapped_ref,) = post.boundary_inputs
        assert a_id not in post.nodes
        assert remapped_ref.node_id in post.nodes
        assert remapped_ref.direction is Direction.INPUT

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre, _a_id, _b_id = _build_boundary_input_and_output(Dim.concrete(3))
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_exact_equality_at_several_concrete_d(self) -> None:
        pre, _a_id, _b_id = _build_boundary_input_and_output(Dim.symbol("d"))
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason


def _build_multiple_boundary_inputs(dim: Dim) -> tuple[Diagram, NodeId, NodeId]:
    """A and B each contribute two surviving boundary inputs, interleaved in the list.

    A: 2 inputs (indices 0, 1), 1 output (the fusion wire). B: 3 inputs (index 0 is the
    fusion wire; indices 1, 2 survive), 0 outputs. The ordered boundary-input list is
    ``[A_in0, A_in1, B_in1, B_in2]`` -- A's survivors in original order, then B's, matching
    the merged leg-ordering convention documented in :mod:`qufzx.rewrite.rules_library`.
    Every prior boundary-input test in this suite used a single boundary input, so order
    preservation across several, interleaved from both matched nodes, was untested.
    """
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[dim, dim], output_dims=[dim])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[dim, dim, dim], output_dims=[])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_inputs(
        [
            PortRef(a_id, Direction.INPUT, 0),
            PortRef(a_id, Direction.INPUT, 1),
            PortRef(b_id, Direction.INPUT, 1),
            PortRef(b_id, Direction.INPUT, 2),
        ]
    )
    return diagram, a_id, b_id


class TestSpiderFusionPreservesBoundaryInputOrder:
    def test_boundary_input_order_matches_the_leg_ordering_convention(self) -> None:
        pre, _a_id, _b_id = _build_multiple_boundary_inputs(Dim.concrete(2))
        post = _fuse(pre)
        assert len(post.boundary_inputs) == 4
        (merged,) = post.nodes.values()
        assert post.boundary_inputs == (
            PortRef(merged.id, Direction.INPUT, 0),
            PortRef(merged.id, Direction.INPUT, 1),
            PortRef(merged.id, Direction.INPUT, 2),
            PortRef(merged.id, Direction.INPUT, 3),
        )

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre, _a_id, _b_id = _build_multiple_boundary_inputs(Dim.concrete(3))
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_exact_equality_at_several_concrete_d(self) -> None:
        pre, _a_id, _b_id = _build_multiple_boundary_inputs(Dim.symbol("d"))
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason


def _build_b_output_into_a_input(
    dim: Dim, generator_type: GeneratorType = Z_SPIDER
) -> tuple[Diagram, NodeId, NodeId]:
    """The fusion wire runs from the higher-id node's output into the lower-id node's input.

    Every other fixture in this suite has the lower-id node (by construction order) as the
    wire's OUTPUT side; ``FusionMatch.a_id`` is always the lower ``NodeId`` regardless of
    which side of the wire it sits on (see that class's docstring), so this is the mirror
    case: ``a_id`` (added first, lower id) sits on the wire's INPUT side, ``b_id`` (added
    second, higher id) on the OUTPUT side. A: 1 input (the fusion wire), 1 output
    (survives). B: 0 inputs, 2 outputs (index 0 is the fusion wire; index 1 survives).
    """
    diagram = Diagram()
    a_id = diagram.add_node(generator_type, input_dims=[dim], output_dims=[dim])
    b_id = diagram.add_node(generator_type, input_dims=[], output_dims=[dim, dim])
    diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(a_id, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.OUTPUT, 1)]
    )
    return diagram, a_id, b_id


class _BOutputIntoAInputCases:
    """Wire runs B.out -> A.in with a_id < b_id, shared across both colors."""

    generator_type: GeneratorType

    def test_fuses_to_a_single_spider(self) -> None:
        pre, _a_id, _b_id = _build_b_output_into_a_input(Dim.symbol("d"), self.generator_type)
        post = _fuse(pre)
        assert len(post.nodes) == 1

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre, _a_id, _b_id = _build_b_output_into_a_input(Dim.concrete(3), self.generator_type)
        post = _fuse(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_exact_equality_at_several_concrete_d(self) -> None:
        pre, _a_id, _b_id = _build_b_output_into_a_input(Dim.symbol("d"), self.generator_type)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason


class TestBOutputIntoAInputZ(_BOutputIntoAInputCases):
    generator_type = Z_SPIDER


class TestBOutputIntoAInputX(_BOutputIntoAInputCases):
    generator_type = X_SPIDER


class TestRepeatedApplicationReachesAFixpointOnAChain:
    """Fuse a three-node chain A->B->C to a fixpoint: two applications, one surviving node."""

    def _build_chain(self, dim: Dim) -> Diagram:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
        return diagram

    def _fuse_to_fixpoint(self, diagram: Diagram) -> tuple[Diagram, int]:
        working = diagram
        steps = 0
        while True:
            matches = find_matches(working)
            if not matches:
                return working, steps
            working = apply(working, SPIDER_FUSION, matches[0]).diagram
            steps += 1

    def test_two_steps_and_one_surviving_node(self) -> None:
        pre = self._build_chain(Dim.symbol("d"))
        post, steps = self._fuse_to_fixpoint(pre)
        assert steps == 2
        assert len(post.nodes) == 1

    def test_post_diagram_is_valid_with_no_deferred_issues(self) -> None:
        pre = self._build_chain(Dim.concrete(3))
        post, _steps = self._fuse_to_fixpoint(pre)
        report = validate(post)
        assert report.is_valid
        assert not report.deferred

    def test_exact_equality_at_several_concrete_d(self) -> None:
        pre = self._build_chain(Dim.symbol("d"))
        post, _steps = self._fuse_to_fixpoint(pre)
        for d_value in _CONCRETE_DS:
            result = compare(pre, post, {"d": d_value})
            assert result.mode is EqualityMode.EXACT
            assert result.matched, result.reason


def _phase_vector(dim: Dim, index: int, turns: sp.Expr) -> PhaseVector:
    return PhaseVector(dim, {index: Phase.turns(turns)})


class _SymbolicPhaseCases:
    """Shared symbolic-phase oracle checks, parameterized by generator color."""

    generator_type: GeneratorType

    def test_symbolic_phase_on_a(self) -> None:
        d = Dim.symbol("d")
        alpha = Phase.symbol("alpha")
        phase_a = PhaseVector(d, {1: alpha})
        pre, _a, _b = build_ghz_with_copy(d, phase_on_a=phase_a, generator_type=self.generator_type)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            for alpha_value in _CONCRETE_PHASES:
                result = compare(pre, post, {"d": d_value, "alpha": alpha_value})
                assert result.matched, (d_value, alpha_value, result.reason)

    def test_symbolic_phase_on_b(self) -> None:
        d = Dim.symbol("d")
        beta = Phase.symbol("beta")
        phase_b = PhaseVector(d, {1: beta})
        pre, _a, _b = build_ghz_with_copy(d, phase_on_b=phase_b, generator_type=self.generator_type)
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            for beta_value in _CONCRETE_PHASES:
                result = compare(pre, post, {"d": d_value, "beta": beta_value})
                assert result.matched, (d_value, beta_value, result.reason)

    def test_symbolic_phase_on_both(self) -> None:
        d = Dim.symbol("d")
        alpha = Phase.symbol("alpha")
        beta = Phase.symbol("beta")
        phase_a = PhaseVector(d, {1: alpha})
        phase_b = PhaseVector(d, {1: beta})
        pre, _a, _b = build_ghz_with_copy(
            d, phase_on_a=phase_a, phase_on_b=phase_b, generator_type=self.generator_type
        )
        post = _fuse(pre)
        for d_value in _CONCRETE_DS:
            for alpha_value in _CONCRETE_PHASES:
                for beta_value in _CONCRETE_PHASES:
                    result = compare(
                        pre, post, {"d": d_value, "alpha": alpha_value, "beta": beta_value}
                    )
                    assert result.matched, (d_value, alpha_value, beta_value, result.reason)


class TestSymbolicPhaseZ(_SymbolicPhaseCases):
    generator_type = Z_SPIDER


class TestSymbolicPhaseX(_SymbolicPhaseCases):
    generator_type = X_SPIDER


class TestNegativeControl:
    def test_spurious_scalar_factor_fails_exact_comparison(self) -> None:
        d = Dim.concrete(3)
        pre, _a, _b = build_ghz_with_copy(d)
        post = _fuse(pre)
        correct = compare(pre, post, {})
        assert correct.matched

        wrong = post.copy()
        wrong.multiply_scalar(Scalar.omega(Dim.concrete(4), 1))
        wrong_result = compare(pre, wrong, {})
        assert wrong_result.mode is EqualityMode.EXACT
        assert not wrong_result.matched

    def test_dropped_phase_fails_exact_comparison(self) -> None:
        d = Dim.concrete(3)
        phase_a = _phase_vector(d, 1, sp.Rational(1, 3))
        phase_b = _phase_vector(d, 1, sp.Rational(1, 5))
        pre, _a, _b = build_ghz_with_copy(d, phase_on_a=phase_a, phase_on_b=phase_b)
        correct_post = _fuse(pre)
        correct_result = compare(pre, correct_post, {})
        assert correct_result.matched

        # A wrongly-fused diagram that keeps only A's phase, dropping B's, instead of adding.
        wrong = Diagram()
        merged_id = wrong.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d, d], phase=phase_a)
        wrong.set_boundary_outputs(
            [
                PortRef(merged_id, Direction.OUTPUT, 0),
                PortRef(merged_id, Direction.OUTPUT, 1),
                PortRef(merged_id, Direction.OUTPUT, 2),
            ]
        )
        wrong_result = compare(pre, wrong, {})
        assert wrong_result.mode is EqualityMode.EXACT
        assert not wrong_result.matched


class TestFusionMatchNegativeControls:
    """Mirrors the corresponding tests in test_match.py; restated here as part of the

    Phase 5 completion condition's explicit checklist (different colors, parallel wires,
    self-loop, third-node survival, dimension mismatch).
    """

    def test_no_match_between_different_colors(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        b_id = diagram.add_node(X_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()

    def test_two_wires_between_same_pair_now_yield_two_matches(self) -> None:
        """A node pair joined by k wires now yields up to k candidates, one per wire, each
        leaving the others as self-loops -- see match.py's module docstring, condition 3.
        See :class:`TestParallelWireFusionOracle` for the oracle coverage."""
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
        assert len(find_matches(diagram)) == 2

    def test_no_match_for_a_self_loop_on_one_spider(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        node_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        diagram.add_wire(
            PortRef(node_id, Direction.OUTPUT, 0), PortRef(node_id, Direction.INPUT, 0)
        )
        assert find_matches(diagram) == ()

    def test_match_survives_a_third_node_wiring_and_remapping_holds(self) -> None:
        d = Dim.symbol("d")
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
        c_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a_id, Direction.OUTPUT, 1)])
        pre_snapshot_scalar = diagram.scalar

        result = apply(diagram, SPIDER_FUSION, find_matches(diagram)[0])
        post = result.diagram
        assert c_id in post.nodes
        for d_value in (2, 3):
            comparison = compare(diagram, post, {"d": d_value})
            assert comparison.matched, (d_value, comparison.reason)
        assert diagram.scalar == pre_snapshot_scalar  # input still untouched

    def test_dimension_mismatch_is_a_non_match(self) -> None:
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[Dim.concrete(2)])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[Dim.concrete(3)], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        assert find_matches(diagram) == ()


class TestRewriteNeverImportsSemantics:
    """Enforces the spec's "rewriting never contracts" rule at the import level.

    Parses each module's AST rather than substring-searching its source, since several
    modules' docstrings *discuss* :mod:`qufzx.semantics` (e.g. explaining why a boundary
    check matters) without importing it -- a substring check would misfire on prose.
    """

    def test_no_rewrite_module_imports_semantics(self) -> None:
        import ast
        import pathlib

        import qufzx.rewrite as rewrite_pkg

        package_dir = pathlib.Path(rewrite_pkg.__file__).parent
        for path in package_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module] if node.module else []
                else:
                    continue
                offending = [n for n in names if n is not None and n.startswith("qufzx.semantics")]
                assert not offending, (
                    f"{path.name} imports {offending} from qufzx.semantics; "
                    "rewriting never contracts"
                )
