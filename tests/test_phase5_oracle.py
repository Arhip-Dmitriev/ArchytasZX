"""Phase 5 numeric-oracle check: the completion condition for this phase.

Per ``claude.md``, every build phase ends with a numeric oracle check, and per the build
plan Phase 5 is done when "the full path Dirac to graph to fuse to graph runs and the
oracle confirms exact equality" -- de-risking the whole project. This module exercises
exactly that: build "A into B" via ``build_ghz_with_copy``, find the fusion match, apply
it, and confirm the oracle (:mod:`qufzx.semantics.check`) reports the pre- and post-fusion
diagrams exactly equal at several concrete ``d``, with no symbolic phase, with a symbolic
phase on A, on B, and on both, and for both the Z and X spider colors -- the X case exists
specifically to exercise the ``wire_direction_output_to_input`` side condition documented
in :mod:`qufzx.rewrite.match`, since only X's non-diagonal denotation can tell a correct
fusion apart from a wrongly-wired one. It also carries the negative controls and import-
boundary check the build plan calls out explicitly.
"""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, PortRef
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


def _build_all_legs_consumed(dim: Dim, generator_type: GeneratorType) -> Diagram:
    """A: ``0->1``, B: ``1->0``, wired output-to-input -- fusion consumes every leg of both.

    The corner case from the module docstring's "Corner case: no legs survive at all"
    note in :mod:`qufzx.rewrite.rules_library`: the merged node ends up with zero inputs
    and zero outputs, so its dimension can only survive via an explicit zero phase.
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

    def test_no_match_across_two_parallel_wires(self) -> None:
        d = Dim.concrete(2)
        diagram = Diagram()
        a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
        b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[])
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
        assert find_matches(diagram) == ()

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
    """Enforces ``claude.md``'s "rewriting never contracts" rule at the import level.

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
