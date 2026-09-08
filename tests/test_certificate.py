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

"""Unit coverage for :mod:`archytaszx.semantics.certificate`: structural comparison, derivation
and certificate construction, replay, and oracle-backed verification.
"""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.engine import RewriteResult, apply
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.semantics.certificate import (
    Certificate,
    CertificateGrammarError,
    CheckMethod,
    Derivation,
    DerivationKind,
    certify,
    compare_structure,
    replay,
    verify,
)
from archytaszx.semantics.check import CheckGrammarError, EqualityMode, compare

from .helpers import build_ghz_with_copy


def _fuse_once(diagram: Diagram) -> RewriteResult:
    """Apply SPIDER_FUSION at the first match found in ``diagram``."""
    match = find_matches(diagram)[0]
    return apply(diagram, SPIDER_FUSION, match)


def _three_spider_chain() -> Diagram:
    """The three-Z-spider chain admitting two successive fusions."""
    d = Dim.concrete(2)
    diagram = Diagram()
    s1 = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
    s2 = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    s3 = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    diagram.add_wire(PortRef(s1, Direction.OUTPUT, 0), PortRef(s2, Direction.INPUT, 0))
    diagram.add_wire(PortRef(s2, Direction.OUTPUT, 0), PortRef(s3, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(s1, Direction.INPUT, 0)])
    diagram.set_boundary_outputs(
        [PortRef(s1, Direction.OUTPUT, 1), PortRef(s3, Direction.OUTPUT, 0)]
    )
    return diagram


class TestCompareStructure:
    def test_a_fresh_copy_is_identical(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        comparison = compare_structure(diagram, diagram.copy())
        assert comparison.identical
        assert comparison.reason == "identical"

    def test_a_missing_node_is_reported(self) -> None:
        d = Dim.concrete(2)
        a, _a_id, b_id = build_ghz_with_copy(d)
        b = a.copy()
        b.remove_node(b_id)
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "only in a [1]" in comparison.reason
        assert "only in b []" in comparison.reason

    def test_a_changed_generator_type_is_reported(self) -> None:
        d = Dim.concrete(2)
        a, _a_id, _b_id = build_ghz_with_copy(d, generator_type=Z_SPIDER)
        b, _a_id2, _b_id2 = build_ghz_with_copy(d, generator_type=X_SPIDER)
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "node 0" in comparison.reason
        assert "generator_type" in comparison.reason

    def test_a_changed_phase_is_reported(self) -> None:
        d = Dim.concrete(2)
        a, _a_id, b_id = build_ghz_with_copy(d)
        b = a.copy()
        b.set_phase(b_id, PhaseVector(d, {1: Phase.turns(sp.Rational(1, 2))}))
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "phase" in comparison.reason

    def test_a_changed_wire_is_reported(self) -> None:
        d = Dim.concrete(2)
        a, a_id, b_id = build_ghz_with_copy(d)
        b = a.copy()
        b.remove_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "wires" in comparison.reason

    def test_boundary_order_is_significant(self) -> None:
        d = Dim.concrete(2)
        a, _a_id, _b_id = build_ghz_with_copy(d)
        b = a.copy()
        outputs = list(b.boundary_outputs)
        outputs[0], outputs[1] = outputs[1], outputs[0]
        b.set_boundary_outputs(outputs)
        assert not compare_structure(a, b).identical

    def test_a_changed_scalar_is_reported(self) -> None:
        d = Dim.concrete(2)
        a, _a_id, _b_id = build_ghz_with_copy(d)
        b = a.copy()
        b.multiply_scalar(Scalar.rational(2))
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "scalar" in comparison.reason

    def test_a_changed_parameter_environment_is_reported(self) -> None:
        a, _a_id, _b_id = build_ghz_with_copy(Dim.symbol("d"))
        b = a.copy()
        b.bind_parameter("d", 3)
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "parameter" in comparison.reason

    def test_check_order_reports_the_earliest_difference(self) -> None:
        d = Dim.concrete(2)
        a, _a_id, b_id = build_ghz_with_copy(d)
        b = a.copy()
        b.remove_node(b_id)
        b.multiply_scalar(Scalar.rational(2))
        comparison = compare_structure(a, b)
        assert not comparison.identical
        assert "node ids differ" in comparison.reason
        assert "scalar" not in comparison.reason

    def test_a_non_diagram_is_a_grammar_error(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        with pytest.raises(CertificateGrammarError):
            compare_structure(diagram, "not a diagram")  # type: ignore[arg-type]


class TestDerivationShape:
    def test_rejects_a_non_diagram_initial(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        with pytest.raises(CertificateGrammarError):
            Derivation(
                kind=DerivationKind.STEP_SEQUENCE,
                initial="not a diagram",  # type: ignore[arg-type]
                final=diagram,
            )

    def test_rejects_a_non_step_in_steps(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        with pytest.raises(CertificateGrammarError):
            Derivation(
                kind=DerivationKind.STEP_SEQUENCE,
                initial=diagram,
                final=diagram,
                steps=("not a step",),  # type: ignore[arg-type]
            )

    def test_rejects_children_on_a_step_sequence(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        child = Derivation(kind=DerivationKind.STEP_SEQUENCE, initial=diagram, final=diagram)
        with pytest.raises(CertificateGrammarError, match="step_sequence"):
            Derivation(
                kind=DerivationKind.STEP_SEQUENCE,
                initial=diagram,
                final=diagram,
                children=(child,),
            )

    def test_accepts_an_empty_step_tuple(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        derivation = Derivation(kind=DerivationKind.STEP_SEQUENCE, initial=diagram, final=diagram)
        assert derivation.steps == ()


class TestDerivationEquality:
    def test_two_certifications_of_the_same_fusion_are_equal(self) -> None:
        d = Dim.concrete(2)
        diagram_one, _a1, _b1 = build_ghz_with_copy(d)
        diagram_two, _a2, _b2 = build_ghz_with_copy(d)
        cert_one = certify(diagram_one, [_fuse_once(diagram_one)])
        cert_two = certify(diagram_two, [_fuse_once(diagram_two)])
        assert cert_one.derivation == cert_two.derivation
        assert cert_one == cert_two

    def test_a_differing_final_diagram_is_unequal(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        other = dataclasses.replace(certificate.derivation, final=diagram.copy())
        assert certificate.derivation != other

    def test_a_derivation_is_unhashable(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        derivation = Derivation(kind=DerivationKind.STEP_SEQUENCE, initial=diagram, final=diagram)
        with pytest.raises(TypeError):
            hash(derivation)

    def test_equality_with_a_non_derivation_is_false(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        derivation = Derivation(kind=DerivationKind.STEP_SEQUENCE, initial=diagram, final=diagram)
        assert derivation != object()


class TestCertify:
    def test_stores_copies_of_its_diagrams(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        before = certificate.initial.copy()
        diagram.multiply_scalar(Scalar.rational(2))
        assert compare_structure(certificate.initial, before).identical

    def test_no_results_gives_a_reflexive_certificate(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        certificate = certify(diagram, [])
        assert certificate.steps == ()
        assert compare_structure(certificate.initial, certificate.final).identical

    def test_rejects_a_non_rewrite_result(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        with pytest.raises(CertificateGrammarError):
            certify(diagram, [object()])  # type: ignore[list-item]

    def test_records_the_requested_check_method_and_label(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        certificate = certify(diagram, [], label="a label", check_method=CheckMethod.NUMERIC_ORACLE)
        assert certificate.derivation.label == "a label"
        assert certificate.check_method is CheckMethod.NUMERIC_ORACLE


class TestReplay:
    def test_reproduces_the_phase5_fusion(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        replayed = replay(certificate)
        assert replayed.reproduced
        assert len(replayed.steps) == 1
        assert replayed.reason == "reproduced"

    def test_replays_on_an_independent_fresh_copy(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        replayed = replay(certificate, source=diagram.copy())
        assert replayed.reproduced

    def test_does_not_mutate_its_source(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        source = diagram.copy()
        before = source.copy()
        replay(certificate, source=source)
        assert compare_structure(source, before).identical

    def test_an_identically_rebuilt_source_still_replays(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        second_diagram, _a2, _b2 = build_ghz_with_copy(d)
        replayed = replay(certificate, source=second_diagram)
        assert replayed.reproduced

    def test_a_foreign_source_reports_the_missing_consumed_node(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        foreign = Diagram()
        foreign.add_node(Z_SPIDER, [], [d])
        replayed = replay(certificate, source=foreign)
        assert replayed.reproduced is False
        assert "consumed node 1 is absent" in replayed.reason

    def test_an_unknown_rule_name_is_reported_not_raised(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        tampered_step = dataclasses.replace(result.step, rule_name="no_such_rule")
        tampered_result = dataclasses.replace(result, step=tampered_step)
        certificate = certify(diagram, [tampered_result])
        replayed = replay(certificate)
        assert replayed.reproduced is False
        assert "rule lookup failed" in replayed.reason

    def test_a_tampered_recorded_step_is_reported(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        tampered_step = dataclasses.replace(
            result.step, scalar_introduced=Scalar.one() + Scalar.one()
        )
        tampered_result = dataclasses.replace(result, step=tampered_step)
        certificate = certify(diagram, [tampered_result])
        replayed = replay(certificate)
        assert replayed.reproduced is False
        assert "the replayed step differs" in replayed.reason
        assert "scalar_introduced" in replayed.reason

    def test_a_tampered_final_diagram_is_reported(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        derivation = Derivation(
            kind=DerivationKind.STEP_SEQUENCE,
            initial=diagram.copy(),
            final=diagram.copy(),
            steps=(result.step,),
        )
        certificate = Certificate(derivation=derivation)
        replayed = replay(certificate)
        assert replayed.reproduced is False
        assert replayed.reason.startswith("final diagram differs")

    def test_rediscovery_catches_an_altered_diagram_before_apply(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])

        source = diagram.copy()
        source.set_phase(b_id, PhaseVector(d, {1: Phase.turns(sp.Rational(1, 2))}))

        rediscovered = replay(certificate, source=source, rediscover=True)
        assert rediscovered.reproduced is False
        assert "recorded match is not among the matches rediscovered" in rediscovered.reason

        not_rediscovered = replay(certificate, source=source, rediscover=False)
        assert not_rediscovered.reproduced is False
        assert "apply raised RewriteDomainError" in not_rediscovered.reason

    def test_stops_at_the_first_failing_step(self) -> None:
        diagram = _three_spider_chain()
        result_one = _fuse_once(diagram)
        result_two = _fuse_once(result_one.diagram)
        tampered_step_one = dataclasses.replace(result_one.step, rule_name="no_such_rule")
        derivation = Derivation(
            kind=DerivationKind.STEP_SEQUENCE,
            initial=diagram.copy(),
            final=result_two.diagram.copy(),
            steps=(tampered_step_one, result_two.step),
        )
        certificate = Certificate(derivation=derivation)
        replayed = replay(certificate)
        assert replayed.reproduced is False
        assert len(replayed.steps) == 1

    def test_a_zero_step_certificate_replays(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        certificate = certify(diagram, [])
        replayed = replay(certificate)
        assert replayed.reproduced
        assert replayed.steps == ()

    def test_steps_recorded_out_of_order_fail(self) -> None:
        diagram = _three_spider_chain()
        result_one = _fuse_once(diagram)
        result_two = _fuse_once(result_one.diagram)
        derivation = Derivation(
            kind=DerivationKind.STEP_SEQUENCE,
            initial=diagram.copy(),
            final=result_two.diagram.copy(),
            steps=(result_two.step, result_one.step),
        )
        certificate = Certificate(derivation=derivation)
        replayed = replay(certificate)
        assert replayed.reproduced is False
        assert "consumed node" in replayed.reason or "rediscovered" in replayed.reason

    def test_repeated_replays_agree(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        first = replay(certificate)
        second = replay(certificate)
        assert first.reproduced == second.reproduced
        assert first.reason == second.reason

    def test_does_not_mutate_the_certificate(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        initial_before = certificate.initial.copy()
        final_before = certificate.final.copy()
        replay(certificate)
        verify(certificate, {})
        assert compare_structure(certificate.initial, initial_before).identical
        assert compare_structure(certificate.final, final_before).identical


class TestVerify:
    def test_verifies_the_fusion_at_d_two_three_and_four(self) -> None:
        for d_value in (2, 3, 4):
            diagram, _a_id, _b_id = build_ghz_with_copy(Dim.concrete(d_value))
            result = _fuse_once(diagram)
            certificate = certify(diagram, [result])
            report = verify(certificate, {})
            assert report.verified, report.reason

    def test_reports_a_failed_replay_without_calling_the_oracle(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        tampered_step = dataclasses.replace(result.step, rule_name="no_such_rule")
        tampered_result = dataclasses.replace(result, step=tampered_step)
        certificate = certify(diagram, [tampered_result])
        report = verify(certificate, {})
        assert report.verified is False
        assert report.comparison is None
        assert report.reason.startswith("replay failed")

    def test_records_the_assignment_and_mode(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        report = verify(certificate, {"extra": 1}, mode=EqualityMode.EXACT)
        assert dict(report.assignment) == {"extra": 1}
        assert report.mode is EqualityMode.EXACT
        assert report.check_method is CheckMethod.NUMERIC_ORACLE

    def test_an_unknown_check_method_is_a_grammar_error(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        object.__setattr__(certificate, "check_method", "not_a_check_method")
        with pytest.raises(CertificateGrammarError):
            verify(certificate, {})

    def test_an_unsupplied_symbol_propagates_the_check_error(self) -> None:
        diagram, _a_id, _b_id = build_ghz_with_copy(Dim.symbol("d"))
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        with pytest.raises(CheckGrammarError):
            verify(certificate, {})

    def test_up_to_global_phase_is_opt_in(self) -> None:
        d = Dim.concrete(2)
        diagram, _a_id, _b_id = build_ghz_with_copy(d)
        result = _fuse_once(diagram)
        certificate = certify(diagram, [result])
        report = verify(certificate, {})
        assert report.verified

        shift = Scalar(sp.exp(sp.I * sp.pi / 4))
        shifted_final = report.replay.diagram.copy()
        shifted_final.multiply_scalar(shift)

        exact = compare(certificate.initial, shifted_final, {}, mode=EqualityMode.EXACT)
        assert not exact.matched

        up_to_phase = compare(
            certificate.initial, shifted_final, {}, mode=EqualityMode.UP_TO_GLOBAL_PHASE
        )
        assert up_to_phase.matched

        plumbed = verify(certificate, {}, mode=EqualityMode.UP_TO_GLOBAL_PHASE)
        assert plumbed.verified
        assert plumbed.mode is EqualityMode.UP_TO_GLOBAL_PHASE
        assert plumbed.comparison is not None
        assert plumbed.comparison.mode is EqualityMode.UP_TO_GLOBAL_PHASE


class TestStructuralComparisonSeesBangBoxes:
    """compare_structure reports a bang-box difference, not just nodes and wires."""

    @staticmethod
    def _boxed(multiplicity: Mult | None, scope_node: int = 0) -> Diagram:
        d = Dim(2)
        diagram = Diagram()
        first = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        second = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
        diagram.set_boundary_outputs(
            [PortRef(first, Direction.OUTPUT, 0), PortRef(second, Direction.OUTPUT, 0)]
        )
        if multiplicity is not None:
            scope = first if scope_node == 0 else second
            diagram.add_bang_box(multiplicity, node_scope=frozenset({scope}))
        return diagram

    def test_differing_multiplicity_is_not_identical(self) -> None:
        result = compare_structure(self._boxed(Mult.concrete(2)), self._boxed(Mult.concrete(5)))
        assert not result.identical
        assert "multiplicity" in result.reason

    def test_a_missing_box_is_not_identical(self) -> None:
        result = compare_structure(self._boxed(Mult.concrete(2)), self._boxed(None))
        assert not result.identical
        assert "bang box ids differ" in result.reason

    def test_differing_scope_is_not_identical(self) -> None:
        result = compare_structure(
            self._boxed(Mult.concrete(2), scope_node=0),
            self._boxed(Mult.concrete(2), scope_node=1),
        )
        assert not result.identical
        assert "node scope" in result.reason

    def test_concrete_against_symbolic_is_not_identical(self) -> None:
        result = compare_structure(self._boxed(Mult.concrete(2)), self._boxed(Mult.symbol("n")))
        assert not result.identical

    def test_the_same_box_is_identical(self) -> None:
        assert compare_structure(
            self._boxed(Mult.symbol("n")), self._boxed(Mult.symbol("n"))
        ).identical
