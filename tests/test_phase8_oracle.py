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

"""Phase 8 numeric-oracle check: symbolic-n equalities proved, not merely spot-checked.

Per FULL_PLAN.md:131-132, Phase 8 is done when "symbolic-n equalities can be proved, not
merely spot-checked", tested by: the fusion GHZ identity proved for all n by induction
rather than by sampling, and the induction failing cleanly on a false near-identity.
Group A covers the first, Group B the second; Groups C, D and E cover multi-index
induction, determinism, and certificate replay and verification.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.diagram.bangbox import (
    abstract_port_count,
    abstract_subgraph_count,
    free_mult_symbols,
)
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.semantics import induction
from archytaszx.semantics.certificate import (
    Certificate,
    CheckMethod,
    DerivationKind,
    certify,
    compare_structure,
    replay,
    verify,
)
from archytaszx.semantics.check import EqualityMode, compare
from archytaszx.semantics.induction import (
    InductionGrammarError,
    InductionResult,
    StepDischarge,
    Verdict,
)

from .helpers import build_ghz_with_copy


def _fuse_once(diagram: Diagram) -> Diagram:
    matches = find_matches(diagram)
    assert len(matches) == 1, "expected exactly one fusion match on the boxed A-into-B example"
    return apply(diagram, SPIDER_FUSION, matches[0]).diagram


def _build_boxed_fusion_family() -> tuple[Diagram, Diagram]:
    """The boxed "A into B" example before and after one fusion, over a symbolic ``d``.

    The returned diagrams carry ``Dim("d")`` free; every caller binds it in its ``witness``.
    """
    pre, a_id, b_id = build_ghz_with_copy(Dim("d"))
    pre, _box_id, _m = abstract_subgraph_count(pre, frozenset({a_id, b_id}), 1, stem="m")
    post = _fuse_once(pre)
    return pre, post


def _build_false_near_identity() -> tuple[Diagram, Diagram]:
    """The bang-boxed GHZ family against the boxed product family, over a symbolic ``d``.

    The left side is one Z spider whose single output leg is port-scope boxed under ``n``.
    The right side is ``n`` independent single-leg Z spiders with no wires between them.
    They agree at ``n = 1`` and diverge at ``n = 2``.
    """
    d = Dim("d")

    left = Diagram()
    left_spider = left.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    left_ref = PortRef(left_spider, Direction.OUTPUT, 0)
    left.set_boundary_outputs([left_ref])
    left, _left_box, _left_n = abstract_port_count(left, left_ref, 1, stem="n")

    right = Diagram()
    right_spider = right.add_node(
        Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {})
    )
    right.set_boundary_outputs([PortRef(right_spider, Direction.OUTPUT, 0)])
    right, _right_box, _right_n = abstract_subgraph_count(
        right, frozenset({right_spider}), 1, stem="n"
    )
    return left, right


class TestGroupA_FusionGHZProvedForAllN:
    """The fusion GHZ identity proved for all n by induction rather than by sampling."""

    def test_the_boxed_fusion_identity_is_proved_by_induction_over_m(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2})
        assert result.proved is True, result.reason
        assert result.verdict in (Verdict.PROVED_UNIFORM, Verdict.PROVED_INDUCTION), result.reason
        assert result.counterexample is None, result.reason

    def test_the_proof_is_not_merely_a_schema_check(self) -> None:
        """FULL_PLAN.md:132's done-when, asserted directly."""
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2})
        assert result.verdict is not Verdict.SCHEMA_CHECKED, result.reason

    def test_the_proof_names_the_index_it_inducted_on(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2})
        assert result.index == "m", result.reason

    def test_the_proof_reports_which_tier_settled_it(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2})
        assert result.discharge in (
            StepDischarge.UNIFORM_REWRITE,
            StepDischarge.INDUCTION_REWRITE,
        ), result.reason

    @pytest.mark.parametrize("d_value", [2, 3, 4])
    def test_the_proof_holds_at_d_two_three_and_four(self, d_value: int) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": d_value})
        assert result.proved, result.reason

    def test_the_proof_at_base_one_claims_only_multiplicities_above_zero(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2}, base=1)
        assert result.base_value == 1, result.reason
        assert result.proved, result.reason

    def test_the_proof_does_not_mutate_either_input_diagram(self) -> None:
        pre, post = _build_boxed_fusion_family()
        pre_before, post_before = pre.copy(), post.copy()
        induction.prove_by_induction(pre, post, witness={"d": 2})
        assert compare_structure(pre, pre_before).identical
        assert compare_structure(post, post_before).identical

    def test_a_proved_result_carries_no_counterexample_and_no_failing_comparison(self) -> None:
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 2})
        assert result.proved, result.reason
        assert result.counterexample is None and result.comparison is None

    def test_the_proof_holds_at_d_four_which_sampling_could_not_reach(self) -> None:
        """d = 4 proves; test_phase7_oracle.py:194-197 drops (m, d) = (5, 4) at 4**15."""
        pre, post = _build_boxed_fusion_family()
        result = induction.prove_by_induction(pre, post, witness={"d": 4})
        assert result.proved, result.reason


class TestGroupB_AFalseNearIdentityFailsCleanly:
    """The induction refuted, with its counterexample, on a family true at the base and
    false above it.

    The two families agree at ``n = 1``, and differ at both ``n = 0`` and ``n = 2``. Tests
    naming an explicit ``base=1`` exercise the window-tier refutation above the base; tests
    on the default base exercise the base-failure short circuit.
    """

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_the_false_near_identity_agrees_with_the_true_family_at_the_base(
        self, d_value: int
    ) -> None:
        left, right = _build_false_near_identity()
        at_one = compare(left, right, {"n": 1, "d": d_value}, mode=EqualityMode.EXACT)
        assert at_one.matched is True, at_one.reason
        at_zero = compare(left, right, {"n": 0, "d": d_value}, mode=EqualityMode.EXACT)
        assert at_zero.matched is False, at_zero.reason
        at_two = compare(left, right, {"n": 2, "d": d_value}, mode=EqualityMode.EXACT)
        assert at_two.matched is False, at_two.reason

    def test_the_false_near_identity_is_not_proved(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result.proved is False, result.reason

    def test_the_false_near_identity_is_refuted_not_inconclusive(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result.verdict is Verdict.REFUTED, result.reason

    def test_the_false_near_identity_is_refuted_at_the_default_base(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2})
        assert result.verdict is Verdict.REFUTED, result.reason
        assert result.counterexample == 0, result.reason
        assert result.tiers == ()

    def test_the_refutation_witnesses_the_smallest_diverging_count_above_the_base(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result.verdict is Verdict.REFUTED, result.reason
        assert result.counterexample == 2, result.reason

    def test_the_refutation_carries_the_oracle_comparison_that_refuted_it(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result.comparison is not None
        assert result.comparison.matched is False
        assert result.comparison.max_abs_deviation > 0

    def test_the_refutation_carries_the_witness_assignment_in_force(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2})
        assert "d" in result.witness

    def test_the_refutation_raises_nothing(self) -> None:
        """Completion is the assertion."""
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result is not None

    def test_the_refutation_reason_is_non_empty(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2})
        assert result.reason

    def test_the_refutation_emits_no_certificate(self) -> None:
        left, right = _build_false_near_identity()
        result = induction.prove_by_induction(left, right, witness={"d": 2}, base=1)
        assert result.derivation is None, result.reason


def _build_two_index_fusion_family() -> tuple[Diagram, Diagram]:
    """The boxed "A into B" gadget before and after one fusion, with a second index.

    One surviving boundary leg is port-scope boxed under ``k2``, nested inside the
    node-scope box over the whole gadget under ``m``. Unlike
    :func:`_build_nested_two_index_family`, the two sides genuinely differ, so proving them
    equal takes a real rewrite rather than structural identity.
    """
    pre, a_id, b_id = build_ghz_with_copy(Dim("d"))
    pre, inner_id, _k2 = abstract_port_count(pre, pre.boundary_outputs[-1], 1, stem="k2")
    pre, outer_id, _m = abstract_subgraph_count(pre, frozenset({a_id, b_id}), 1, stem="m")
    inner = pre.bang_boxes[inner_id]
    pre.remove_bang_box(inner_id)
    pre.add_bang_box(inner.multiplicity, port_scope=inner.port_scope, parent=outer_id)
    return pre, _fuse_once(pre)


def _build_nested_two_index_family() -> Diagram:
    """One Z spider whose output leg is port-scope boxed under ``k2``, the whole gadget
    node-scope boxed under ``k1``, over the symbolic dimension ``Dim("d")``."""
    d = Dim("d")
    diagram = Diagram()
    s = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(s, Direction.OUTPUT, 0)
    diagram.set_boundary_outputs([ref])
    diagram, inner_id, _k2 = abstract_port_count(diagram, ref, 1, stem="k2")
    diagram, outer_id, _k1 = abstract_subgraph_count(diagram, frozenset({s}), 1, stem="k1")
    inner = diagram.bang_boxes[inner_id]
    diagram.remove_bang_box(inner_id)
    diagram.add_bang_box(inner.multiplicity, port_scope=inner.port_scope, parent=outer_id)
    return diagram


def _cert_prove_fusion(d_value: int) -> InductionResult:
    """Prove the boxed fusion identity by induction over ``m`` at the given ``d``."""
    pre, post = _build_boxed_fusion_family()
    return induction.prove_by_induction(pre, post, witness={"d": d_value})


def _cert_certificate_for(result: InductionResult) -> Certificate:
    """Wrap a proved result's derivation into a certificate carrying the INDUCTION method."""
    assert result.derivation is not None
    return Certificate(derivation=result.derivation, check_method=CheckMethod.INDUCTION)


def _cert_phase_six_certificate() -> Certificate:
    """An ordinary Phase 6 STEP_SEQUENCE certificate: one spider fusion at concrete d = 2."""
    diagram, _a_id, _b_id = build_ghz_with_copy(Dim.concrete(2))
    match = find_matches(diagram)[0]
    result = apply(diagram, SPIDER_FUSION, match)
    return certify(diagram, [result])


class TestGroupC_MultiIndexInductsOneIndexAtATime:
    """A two-index family is proved one index at a time, the other index left symbolic."""

    @pytest.mark.parametrize("index,held", [("m", "k2"), ("k2", "m")])
    def test_a_genuinely_differing_family_is_proved_one_index_at_a_time(
        self, index: str, held: str
    ) -> None:
        """The two sides differ by a fusion, so the proof takes a real rewrite step."""
        left, right = _build_two_index_fusion_family()
        result = induction.prove_by_induction(left, right, index=index, witness={"d": 2})
        assert result.proved, result.reason
        assert held in result.held_symbolic
        settled = [tier for tier in result.tiers if tier.settled]
        assert settled, result.reason
        assert any(tier.steps for tier in settled), (
            "the family was settled without rewriting anything, so this asserts only X = X"
        )

    def test_inducting_on_the_outer_index_holds_the_inner_one_symbolic(self) -> None:
        left = _build_nested_two_index_family()
        right = _build_nested_two_index_family()
        result = induction.prove_by_induction(left, right, index="k1", witness={"d": 2})
        assert result.proved, result.reason
        assert "k2" in result.held_symbolic

    def test_inducting_on_the_inner_index_holds_the_outer_one_symbolic(self) -> None:
        left = _build_nested_two_index_family()
        right = _build_nested_two_index_family()
        result = induction.prove_by_induction(left, right, index="k2", witness={"d": 2})
        assert result.proved, result.reason
        assert "k1" in result.held_symbolic

    def test_both_orders_of_induction_reach_the_same_verdict(self) -> None:
        result_k1 = induction.prove_by_induction(
            _build_nested_two_index_family(),
            _build_nested_two_index_family(),
            index="k1",
            witness={"d": 2},
        )
        result_k2 = induction.prove_by_induction(
            _build_nested_two_index_family(),
            _build_nested_two_index_family(),
            index="k2",
            witness={"d": 2},
        )
        assert result_k1.proved == result_k2.proved

    def test_the_default_index_is_the_lexicographically_least(self) -> None:
        left = _build_nested_two_index_family()
        right = _build_nested_two_index_family()
        result = induction.prove_by_induction(left, right, witness={"d": 2})
        assert result.index == "k1"

    def test_an_unbound_second_index_is_refused_in_a_numeric_tier(self) -> None:
        """The tier itself refuses an unbound symbol; a ladder declines instead of raising."""
        left = _build_nested_two_index_family()
        right = _build_nested_two_index_family()
        obligation = induction.build_obligation(left, right, index="k1", witness={"d": 2})
        with pytest.raises(InductionGrammarError, match="uninstantiated"):
            induction.discharge_oracle_window(obligation)

        result = induction.prove_by_induction(
            left,
            right,
            index="k1",
            witness={"d": 2},
            ladder=(StepDischarge.ORACLE_WINDOW,),
        )
        assert not result.proved
        assert any("uninstantiated" in tier.reason for tier in result.tiers)

    SCRIPT = Path(__file__).parent / "_induction_determinism_script.py"

    def _run_with_seed(self, seed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, str(self.SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_two_proofs_of_the_same_family_are_identical(self) -> None:
        d_value = 2
        left_first, right_first = _build_false_near_identity()
        first = induction.prove_by_induction(left_first, right_first, witness={"d": d_value})

        left_second, right_second = _build_false_near_identity()
        second = induction.prove_by_induction(left_second, right_second, witness={"d": d_value})

        assert first.counterexample == second.counterexample
        assert first.reason == second.reason
        assert first.verdict == second.verdict

    def test_the_witness_is_stable_across_hash_seeds(self) -> None:
        # Two seeds, mirroring tests/test_engine.py:1583-1600 -- each subprocess pays sympy's
        # import cost.
        first = self._run_with_seed("0")
        second = self._run_with_seed("2147483647")
        assert first, "the driver script printed nothing"
        assert second == first


class TestGroupE_CertificateReplay:
    """A proved induction emits a certificate that replays, and whose verification touches each
    child rather than the symbolic node."""

    def test_a_proved_induction_emits_an_induction_derivation(self) -> None:
        result = _cert_prove_fusion(2)
        assert result.proved, result.reason
        assert result.derivation is not None
        assert result.derivation.kind is DerivationKind.INDUCTION
        assert len(result.derivation.children) == 2

    def test_the_emitted_certificate_replays(self) -> None:
        certificate = _cert_certificate_for(_cert_prove_fusion(2))
        assert replay(certificate).reproduced is True

    def test_the_replay_reproduces_both_children(self) -> None:
        certificate = _cert_certificate_for(_cert_prove_fusion(2))
        replayed = replay(certificate)
        assert len(replayed.children) == 2
        assert all(child.reproduced for child in replayed.children)

    @pytest.mark.parametrize("d_value", [2, 3])
    def test_verification_of_the_induction_certificate_passes_at_d_two_and_three(
        self, d_value: int
    ) -> None:
        certificate = _cert_certificate_for(_cert_prove_fusion(d_value))
        report = verify(certificate, {"d": d_value})
        assert report.verified is True, report.reason

    def test_verification_compares_each_child_and_not_the_symbolic_node(self) -> None:
        result = _cert_prove_fusion(2)
        assert result.derivation is not None
        certificate = _cert_certificate_for(result)
        report = verify(certificate, {"d": 2})

        assert report.verified is True, report.reason
        assert report.comparison is None
        assert len(report.child_comparisons) == 2
        assert all(child.matched for child in report.child_comparisons)

        assert len(free_mult_symbols(certificate.initial)) == 1
        for child in result.derivation.children:
            assert free_mult_symbols(child.initial) == frozenset()
            assert free_mult_symbols(child.final) == frozenset()

    def test_a_phase_six_certificate_still_replays_unchanged(self) -> None:
        replayed = replay(_cert_phase_six_certificate())
        assert replayed.children == ()
        assert replayed.reproduced
