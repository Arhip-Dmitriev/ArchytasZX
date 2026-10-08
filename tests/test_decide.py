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


"""The normal-form driver and the ``decide_equal`` ladder, each verdict checked by the oracle."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionError
from archytaszx.algebra.phase import Phase, PhaseError, PhaseVector
from archytaszx.algebra.scalar import Scalar, ScalarError
from archytaszx.diagram.bangbox import BangBoxError, Mult
from archytaszx.diagram.compare import canonical_key, isomorphic
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorError, GeneratorType
from archytaszx.diagram.graph import Diagram, Direction, GraphDomainError, GraphError, PortRef
from archytaszx.diagram.validate import ValidateError
from archytaszx.rewrite.egraph import EGraph, SaturationLimits, SaturationStop
from archytaszx.rewrite.engine import DEFAULT_GUARD, TerminationGuard, apply
from archytaszx.rewrite.normal_form import (
    NormalForm,
    comparison_view,
    isomorphic_up_to_scalar,
    normal_form,
    same_normal_form,
    view_key,
)
from archytaszx.rewrite.rule import ConstraintOutcome, DimensionConstraint, RewriteGrammarError
from archytaszx.rewrite.rules_library import BIALGEBRA, STATE_COPY, lookup_rule
from archytaszx.semantics import decide as decide_module
from archytaszx.semantics.certificate import (
    CertificateDomainError,
    ReplayResult,
    replay,
    verify,
)
from archytaszx.semantics.check import CheckError, compare, score
from archytaszx.semantics.contract_numeric import ContractError
from archytaszx.semantics.decide import (
    DECIDE_SATURATION_LIMITS,
    DecideError,
    DecideGrammarError,
    Decision,
    DecisionMethod,
    EqualityVerdict,
    decide_equal,
    dimension_floors,
    refute_by_oracle,
    refute_by_symbolic_witness,
    sample_grid,
)
from archytaszx.semantics.denote import DenoteError

from . import test_phase8_oracle as T8
from .helpers import build_ghz_with_copy
from .test_normal_form import _deferred_fusion_diagram as deferred_fusion_diagram
from .test_rules_library_phase11 import (
    bialgebra_diagram,
    fourier_state_diagram,
    hopf_diagram,
    identity_chain,
    inp,
    out,
    state_copy_diagram,
    state_copy_right_hand_side,
    triangle_chain,
)

D = Dim("d")
NO_FUSION = TerminationGuard(max_steps=0)
ORACLE_SAMPLES = 6
"""How many grid assignments each EQUAL cross-check evaluates at most."""

ORACLE_REFUSALS: tuple[type[Exception], ...] = (
    BangBoxError,
    CheckError,
    ContractError,
    DenoteError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    ScalarError,
    ValidateError,
)

Pair = tuple[Diagram, Diagram]


# -- builders ---------------------------------------------------------------------------


def z_state(dim: Dim, phase: PhaseVector | None = None) -> Diagram:
    """One Z spider with no inputs and one output on the boundary."""
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim], phase=phase)
    diagram.set_boundary_outputs([out(node)])
    return diagram


def cup(generator: GeneratorType, dim: Dim) -> Diagram:
    """One phaseless spider with no inputs and two boundary outputs."""
    diagram = Diagram()
    node = diagram.add_node(generator, input_dims=[], output_dims=[dim, dim])
    diagram.set_boundary_outputs([out(node, 0), out(node, 1)])
    return diagram


def ghz_spider(dim: Dim) -> Diagram:
    """One phaseless Z spider with no inputs and three boundary outputs."""
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim] * 3)
    diagram.set_boundary_outputs([out(node, i) for i in range(3)])
    return diagram


def self_loop(dim: Dim) -> Diagram:
    """A Z spider ``[dim] -> [dim, dim]`` whose second output feeds its own input."""
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(out(node, 1), inp(node, 0))
    diagram.set_boundary_outputs([out(node, 0)])
    return diagram


def fusion_chain(order: str, dim: Dim = D) -> Diagram:
    """A Z state, a phased one-in-one-out Z spider and a Z copy in series, nodes added in
    ``order`` (a permutation of ``"sme"``)."""
    phase = PhaseVector(dim, {1: Phase.turns(sp.Rational(1, 3))})
    diagram = Diagram()
    ids = {}
    for key in order:
        if key == "s":
            ids[key] = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        elif key == "m":
            ids[key] = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim], phase=phase)
        else:
            ids[key] = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(out(ids["s"]), inp(ids["m"]))
    diagram.add_wire(out(ids["m"]), inp(ids["e"]))
    diagram.set_boundary_outputs([out(ids["e"], 0), out(ids["e"], 1)])
    return diagram


def state_copy_pair(d: int, n: int) -> Pair:
    """``state_copy_diagram(d, n)`` and its exact STATE_COPY rewrite."""
    left = state_copy_diagram(d, n)
    match = STATE_COPY.pattern.find_matches(left)[0]
    return left, apply(left, STATE_COPY, match).diagram


def scaled(diagram: Diagram, factor: Scalar) -> Diagram:
    """A copy of ``diagram`` with its scalar multiplied by ``factor``."""
    result = diagram.copy()
    result.multiply_scalar(factor)
    return result


def phase_turns(dim: Dim, turns: sp.Rational) -> PhaseVector:
    """A phase vector over ``dim`` with leg 1 at ``turns``."""
    return PhaseVector(dim, {1: Phase.turns(turns)})


def ghz_pair(dim: Dim) -> Pair:
    """The GHZ-with-copy example and the single fused spider it reduces to."""
    return build_ghz_with_copy(dim)[0], ghz_spider(dim)


def concrete_false_near_identity() -> Pair:
    """``T8._build_false_near_identity`` with ``d`` substituted by 2."""
    left, right = T8._build_false_near_identity()
    return left.substitute({"d": 2}), right.substitute({"d": 2})


def zero_phase_pair(dim: Dim = D) -> Pair:
    """A Z state with an explicit all-zero phase vector, and the phaseless one."""
    return z_state(dim, PhaseVector(dim, {})), z_state(dim)


# -- oracle cross-checks ----------------------------------------------------------------


def family(diagram: Diagram) -> Diagram:
    """A copy of ``diagram`` with an empty parameter environment."""
    view = diagram.copy()
    view.set_parameters({})
    return view


def oracle_samples(left: Diagram, right: Diagram) -> list[tuple[Mapping[str, object], bool]]:
    """Up to ORACLE_SAMPLES grid assignments the oracle evaluates, each with its match flag."""
    left, right = family(left), family(right)
    evaluated: list[tuple[Mapping[str, object], bool]] = []
    for assignment in sample_grid(left, right):
        if len(evaluated) >= ORACLE_SAMPLES:
            break
        try:
            matched = compare(left, right, assignment).matched
        except ORACLE_REFUSALS:
            continue
        evaluated.append((assignment, matched))
    return evaluated


def assert_oracle_agrees(left: Diagram, right: Diagram, decision: Decision) -> None:
    """Fail unless the numeric oracle supports ``decision``'s verdict."""
    if decision.verdict is EqualityVerdict.EQUAL:
        evaluated = oracle_samples(left, right)
        assert evaluated, "the oracle evaluated no assignment"
        assert all(matched for _, matched in evaluated), evaluated
    elif decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE:
        assert decision.counterexample is not None
        result = compare(family(left), family(right), decision.counterexample)
        assert not result.matched, result.reason
    elif decision.method is DecisionMethod.SYMBOLIC_WITNESS:
        assert decision.witness is not None and decision.witness.deviation > 0
        assert decision.counterexample == decision.witness.assignment
    elif decision.method is DecisionMethod.INTERFACE:
        evaluated = oracle_samples(left, right)
        assert evaluated, "the oracle evaluated no assignment"
        assert not any(matched for _, matched in evaluated), evaluated


def assert_certificates_verify(decision: Decision) -> None:
    """Fail unless both normal-form certificates, or every saturation edge certificate, replay
    and verify at a concrete assignment."""
    assert decision.left_nf is not None and decision.right_nf is not None
    if decision.method is DecisionMethod.SATURATION:
        assert decision.saturation is not None
        for certificate in decision.certificates:
            assert certificate.derivation.label.startswith("saturation edge ")
    else:
        assert len(decision.certificates) == 2
    for certificate in decision.certificates:
        assert replay(certificate).reproduced
        verified = False
        for assignment in sample_grid(certificate.initial, certificate.initial):
            try:
                report = verify(certificate, assignment)
            except ORACLE_REFUSALS:
                continue
            assert report.verified, report.reason
            verified = True
            break
        assert verified, "no assignment verified the certificate"


def snapshot(diagram: Diagram) -> tuple[str, Diagram, dict[str, int], Scalar]:
    """Everything a mutation would change: key, a copy, parameters and scalar."""
    return canonical_key(diagram), diagram.copy(), dict(diagram.parameters), diagram.scalar


def assert_unchanged(diagram: Diagram, before: tuple[str, Diagram, dict[str, int], Scalar]) -> None:
    """Fail unless ``diagram`` still matches its ``snapshot``."""
    key, copy, parameters, scalar = before
    assert canonical_key(diagram) == key
    assert isomorphic(diagram, copy)
    assert dict(diagram.parameters) == parameters
    assert diagram.scalar == scalar


# -- the ladder's settled pairs ---------------------------------------------------------

NF_EQUAL: dict[str, Callable[[], Pair]] = {
    "hopf_self": lambda: (hopf_diagram(3, 1, 1, 1, 1), hopf_diagram(3, 1, 1, 1, 1)),
    "identity_z_vs_x": lambda: (identity_chain(3), identity_chain(3, X_SPIDER)),
    "triangle_orders": lambda: (triangle_chain(3), triangle_chain(3, inverse_first=True)),
    "fourier_rebuilt": lambda: (
        fourier_state_diagram(3, is_state=True),
        fourier_state_diagram(3, is_state=True),
    ),
    "ghz_fused": lambda: ghz_pair(D),
    "state_copy_rewrite": lambda: state_copy_pair(3, 3),
    "fusion_node_orders": lambda: (fusion_chain("sme"), fusion_chain("ems")),
}

UNEQUAL_INTERFACE: dict[str, Callable[[], Pair]] = {
    "arity": lambda: (build_ghz_with_copy(D)[0], state_copy_right_hand_side(3, 2)),
    "dimension": lambda: (z_state(Dim.concrete(2)), z_state(Dim.concrete(3))),
}

UNEQUAL_ORACLE: dict[str, Callable[[], Pair]] = {
    "false_near_identity": T8._build_false_near_identity,
    "different_phases": lambda: (
        z_state(D, phase_turns(D, sp.Rational(1, 3))),
        z_state(D, phase_turns(D, sp.Rational(1, 4))),
    ),
    "extra_scalar": lambda: (
        hopf_diagram(3, 1, 1, 1, 1),
        scaled(hopf_diagram(3, 1, 1, 1, 1), Scalar.rational(2)),
    ),
    "z_cup_vs_x_cup": lambda: (cup(Z_SPIDER, D), cup(X_SPIDER, D)),
}

SYMBOLIC_EQUAL: dict[str, Callable[[], Pair]] = {
    "zero_phase_vector": zero_phase_pair,
    "full_turn_phase": lambda: (z_state(D, phase_turns(D, sp.Rational(1, 1))), z_state(D)),
    "concrete_self_loop": lambda: (self_loop(Dim.concrete(2)), z_state(Dim.concrete(2))),
}

INDUCTION_EQUAL: dict[str, Callable[[], Pair]] = {
    "boxed_fusion": T8._build_boxed_fusion_family,
    "two_index_fusion": T8._build_two_index_fusion_family,
}

ALL_PAIRS: dict[str, tuple[Callable[[], Pair], dict[str, object]]] = {
    **{name: (builder, {}) for name, builder in NF_EQUAL.items()},
    **{name: (builder, {}) for name, builder in UNEQUAL_INTERFACE.items()},
    **{name: (builder, {}) for name, builder in UNEQUAL_ORACLE.items()},
    **{name: (builder, {}) for name, builder in SYMBOLIC_EQUAL.items()},
    **{name: (builder, {"guard": NO_FUSION}) for name, builder in INDUCTION_EQUAL.items()},
}


class TestNormalForm:
    """``normal_form`` and its comparison helpers."""

    def test_returns_a_fixpoint_with_replayable_results(self) -> None:
        left, _ = ghz_pair(D)
        nf = normal_form(left)
        assert isinstance(nf, NormalForm)
        assert nf.reached_fixpoint
        assert len(nf.results) == len(nf.outcome.steps) >= 1
        assert nf.assumed_constraints == ()
        working = nf.source.copy()
        for result in nf.results:
            rule = lookup_rule(result.step.rule_name)
            working = apply(working, rule, result.step.match).diagram
        assert isomorphic(working, nf.diagram)

    def test_source_is_a_copy_and_the_input_is_untouched(self) -> None:
        left, _ = ghz_pair(D)
        before = snapshot(left)
        nf = normal_form(left)
        assert nf.source is not left
        assert isomorphic(nf.source, left)
        assert_unchanged(left, before)

    def test_key_is_the_view_key_of_the_comparison_view(self) -> None:
        nf = normal_form(hopf_diagram(3, 1, 1, 1, 1))
        assert nf.key == view_key(comparison_view(nf.diagram))

    @pytest.mark.parametrize("order", ["sme", "mse", "esm", "ems", "sem", "mes"])
    def test_node_insertion_order_does_not_change_the_key(self, order: str) -> None:
        assert normal_form(fusion_chain(order)).key == normal_form(fusion_chain("sme")).key

    def test_same_normal_form_and_distinct_keys(self) -> None:
        a = normal_form(identity_chain(3))
        b = normal_form(identity_chain(3, X_SPIDER))
        c = normal_form(cup(Z_SPIDER, D))
        d = normal_form(cup(X_SPIDER, D))
        assert a.key == b.key and same_normal_form(a, b)
        assert c.key != d.key and not same_normal_form(c, d)

    def test_a_step_limit_stops_short_of_the_fixpoint(self) -> None:
        pre, _ = T8._build_boxed_fusion_family()
        assert not normal_form(pre, guard=NO_FUSION).reached_fixpoint
        assert normal_form(pre).reached_fixpoint

    def test_comparison_view_clears_parameters_only(self) -> None:
        diagram, _ = ghz_pair(D)
        diagram.bind_parameter("d", 3)
        view = comparison_view(diagram)
        assert dict(view.parameters) == {}
        assert dict(diagram.parameters) == {"d": 3}
        assert normal_form(diagram).key == normal_form(ghz_pair(D)[0]).key

    def test_isomorphic_up_to_scalar_ignores_the_scalar(self) -> None:
        base = hopf_diagram(3, 1, 1, 1, 1)
        assert isomorphic_up_to_scalar(base, scaled(base, Scalar.rational(2)))
        assert not isomorphic(base, scaled(base, Scalar.rational(2)))
        assert not isomorphic_up_to_scalar(cup(Z_SPIDER, D), cup(X_SPIDER, D))

    @pytest.mark.parametrize(
        "call",
        [
            lambda: normal_form("diagram"),  # type: ignore[arg-type]
            lambda: normal_form(z_state(D), rules="rules"),  # type: ignore[arg-type]
            lambda: normal_form(z_state(D), rules=[1]),  # type: ignore[list-item]
            lambda: same_normal_form(normal_form(z_state(D)), z_state(D)),  # type: ignore[arg-type]
            lambda: comparison_view(1),  # type: ignore[arg-type]
            lambda: isomorphic_up_to_scalar(z_state(D), None),  # type: ignore[arg-type]
        ],
    )
    def test_malformed_arguments_raise(self, call: Callable[[], object]) -> None:
        with pytest.raises(RewriteGrammarError):
            call()


class TestSampleGrid:
    """The default oracle assignments."""

    def test_no_free_symbol_gives_the_empty_assignment(self) -> None:
        grid = sample_grid(hopf_diagram(2, 1, 1, 1, 1), hopf_diagram(2, 1, 1, 1, 1))
        assert [dict(sample) for sample in grid] == [{}]

    def test_one_dimension_symbol(self) -> None:
        grid = sample_grid(*ghz_pair(D))
        assert [dict(sample) for sample in grid] == [{"d": 2}, {"d": 3}, {"d": 4}, {"d": 1}]

    def test_diagonals_come_first_and_the_grid_is_full(self) -> None:
        grid = [dict(sample) for sample in sample_grid(*T8._build_boxed_fusion_family())]
        assert grid[:4] == [
            {"d": 2, "m": 0},
            {"d": 3, "m": 1},
            {"d": 4, "m": 2},
            {"d": 1, "m": 3},
        ]
        assert len(grid) == 16
        assert len({tuple(sorted(sample.items())) for sample in grid}) == 16

    def test_symbols_of_either_side_are_covered(self) -> None:
        grid = sample_grid(cup(Z_SPIDER, D), cup(Z_SPIDER, Dim("e")))
        assert all(set(sample) == {"d", "e"} for sample in grid)


class TestLadder:
    """Each rung settles its pairs, and the oracle agrees with every verdict."""

    @pytest.mark.parametrize("name", sorted(NF_EQUAL))
    def test_normal_form_equal(self, name: str) -> None:
        left, right = NF_EQUAL[name]()
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.NORMAL_FORM
        assert decision.decided
        assert decision.left_nf is not None and decision.right_nf is not None
        assert decision.left_nf.key == decision.right_nf.key
        assert decision.samples_checked == 0
        assert decision.assumptions == ()
        assert decision.counterexample is None
        assert_oracle_agrees(left, right, decision)
        assert_certificates_verify(decision)

    @pytest.mark.parametrize("name", sorted(UNEQUAL_INTERFACE))
    def test_interface_unequal(self, name: str) -> None:
        left, right = UNEQUAL_INTERFACE[name]()
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.INTERFACE
        assert decision.left_nf is None and decision.right_nf is None
        assert decision.certificates == ()
        assert decision.counterexample is None
        assert_oracle_agrees(left, right, decision)

    @pytest.mark.parametrize("name", sorted(UNEQUAL_ORACLE))
    def test_oracle_counterexample_unequal(self, name: str) -> None:
        left, right = UNEQUAL_ORACLE[name]()
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert decision.counterexample is not None
        assert decision.comparison is not None and not decision.comparison.matched
        assert 1 <= decision.samples_checked <= 24
        assert decision.left_nf is not None and decision.right_nf is not None
        assert decision.left_nf.key != decision.right_nf.key
        assert_oracle_agrees(left, right, decision)
        assert_certificates_verify(decision)

    @pytest.mark.parametrize(
        ("name", "counterexample"),
        [
            ("false_near_identity", {"d": 2, "n": 0}),
            ("different_phases", {"d": 2}),
            ("extra_scalar", {}),
            ("z_cup_vs_x_cup", {"d": 3}),
        ],
    )
    def test_counterexample_is_the_first_grid_mismatch(
        self, name: str, counterexample: dict[str, int]
    ) -> None:
        decision = decide_equal(*UNEQUAL_ORACLE[name]())
        assert decision.counterexample is not None
        assert dict(decision.counterexample) == counterexample

    @pytest.mark.parametrize("name", sorted(SYMBOLIC_EQUAL))
    def test_symbolic_contraction_equal(self, name: str) -> None:
        left, right = SYMBOLIC_EQUAL[name]()
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION
        assert decision.samples_checked >= 1
        assert decision.left_nf is not None and decision.right_nf is not None
        assert decision.left_nf.key != decision.right_nf.key
        assert_oracle_agrees(left, right, decision)
        assert_certificates_verify(decision)

    @pytest.mark.parametrize("name", sorted(INDUCTION_EQUAL))
    def test_induction_equal(self, name: str) -> None:
        left, right = INDUCTION_EQUAL[name]()
        decision = decide_equal(
            left, right, guard=NO_FUSION, use_saturation=False, use_symbolic=False
        )
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.INDUCTION
        assert decision.induction is not None and decision.induction.proved
        assert not decision.induction_reversed
        assert decision.counterexample is None
        assert_oracle_agrees(left, right, decision)
        assert_certificates_verify(decision)

    @pytest.mark.parametrize("name", sorted(INDUCTION_EQUAL))
    def test_without_the_guard_induction_pairs_settle_by_normal_form(self, name: str) -> None:
        left, right = INDUCTION_EQUAL[name]()
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.NORMAL_FORM
        assert decision.induction is None
        assert_oracle_agrees(left, right, decision)

    def test_induction_depth_zero_gives_unknown(self) -> None:
        left, right = T8._build_boxed_fusion_family()
        decision = decide_equal(
            left, right, guard=NO_FUSION, max_depth=0, use_saturation=False, use_symbolic=False
        )
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.method is DecisionMethod.NONE
        assert decision.samples_checked >= 1
        assert "induction depth exhausted" in decision.reason
        assert all(matched for _, matched in oracle_samples(left, right))


class TestUnknown:
    """Pairs no enabled rung settles."""

    def test_zero_phase_with_symbolic_and_induction_disabled(self) -> None:
        left, right = zero_phase_pair()
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.method is DecisionMethod.NONE
        assert not decision.decided
        assert decision.samples_checked == 4
        assert decision.counterexample is None
        assert "symbolic contraction disabled" in decision.reason
        assert "induction disabled" in decision.reason
        assert all(matched for _, matched in oracle_samples(left, right))

    def test_empty_samples_never_yield_an_oracle_unequal(self) -> None:
        left, right = T8._build_false_near_identity()
        decision = decide_equal(left, right, samples=[], use_induction=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.samples_checked == 0
        assert decision.counterexample is None


class TestSamples:
    """The ``samples`` override, the ``max_samples`` cap and the evaluated-sample count."""

    def test_caller_samples_replace_the_grid(self) -> None:
        left, right = cup(Z_SPIDER, D), cup(X_SPIDER, D)
        decision = decide_equal(left, right, samples=[{"d": 3}])
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert dict(decision.counterexample or {}) == {"d": 3}
        assert decision.samples_checked == 1
        assert_oracle_agrees(left, right, decision)

    def test_matching_caller_samples_leave_the_pair_unrefuted(self) -> None:
        left, right = cup(Z_SPIDER, D), cup(X_SPIDER, D)
        decision = decide_equal(left, right, samples=[{"d": 2}])
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.samples_checked == 1
        assert compare(left, right, {"d": 2}).matched

    def test_max_samples_caps_the_oracle(self) -> None:
        left, right = cup(Z_SPIDER, D), cup(X_SPIDER, D)
        decision = decide_equal(left, right, max_samples=1, use_symbolic=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.samples_checked == 1
        assert not compare(left, right, {"d": 3}).matched

    @pytest.mark.parametrize("cap", [0, 1, 2, 3, 4, 10])
    def test_samples_checked_never_exceeds_the_cap(self, cap: int) -> None:
        decision = decide_equal(*zero_phase_pair(), max_samples=cap)
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION
        assert decision.samples_checked == min(cap, 4)

    def test_refused_samples_are_not_counted(self) -> None:
        left, right = zero_phase_pair()
        decision = decide_equal(left, right, samples=[{}, {"e": 2}, {"d": 2}])
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION
        assert decision.samples_checked == 1

    def test_normal_form_match_skips_the_oracle(self) -> None:
        decision = decide_equal(*ghz_pair(D), samples=[{"d": 2}, {"d": 3}])
        assert decision.method is DecisionMethod.NORMAL_FORM
        assert decision.samples_checked == 0


class TestParameters:
    """The parameter environment never changes a verdict."""

    @pytest.mark.parametrize("name", ["ghz_fused", "z_cup_vs_x_cup", "zero_phase_vector"])
    def test_binding_one_side_only(self, name: str) -> None:
        builder = {**NF_EQUAL, **UNEQUAL_ORACLE, **SYMBOLIC_EQUAL}[name]
        left, right = builder()
        unbound = decide_equal(left, right)
        left.bind_parameter("d", 3)
        bound = decide_equal(left, right)
        assert unbound.parameters_agree
        assert not bound.parameters_agree
        assert bound.verdict is unbound.verdict
        assert bound.method is unbound.method
        assert dict(bound.counterexample or {}) == dict(unbound.counterexample or {})
        assert_oracle_agrees(left, right, bound)

    def test_binding_both_sides_alike_agrees(self) -> None:
        left, right = ghz_pair(D)
        left.bind_parameter("d", 2)
        right.bind_parameter("d", 2)
        decision = decide_equal(left, right)
        assert decision.parameters_agree
        assert decision.method is DecisionMethod.NORMAL_FORM


class TestPurity:
    """Inputs are never mutated and decisions are deterministic."""

    @pytest.mark.parametrize("name", sorted(ALL_PAIRS))
    def test_inputs_are_not_mutated(self, name: str) -> None:
        builder, options = ALL_PAIRS[name]
        left, right = builder()
        left.bind_parameter("d", 2)
        before_l, before_r = snapshot(left), snapshot(right)
        decide_equal(left, right, **options)  # type: ignore[arg-type]
        assert_unchanged(left, before_l)
        assert_unchanged(right, before_r)

    @pytest.mark.parametrize("name", sorted(ALL_PAIRS))
    def test_decisions_are_deterministic(self, name: str) -> None:
        builder, options = ALL_PAIRS[name]
        first = decide_equal(*builder(), **options)  # type: ignore[arg-type]
        second = decide_equal(*builder(), **options)  # type: ignore[arg-type]
        assert first.verdict is second.verdict
        assert first.method is second.method
        assert first.reason == second.reason
        assert first.samples_checked == second.samples_checked
        assert dict(first.counterexample or {}) == dict(second.counterexample or {})
        for a, b in ((first.left_nf, second.left_nf), (first.right_nf, second.right_nf)):
            assert (a is None) == (b is None)
            if a is not None and b is not None:
                assert a.key == b.key

    @pytest.mark.parametrize("name", sorted(ALL_PAIRS))
    def test_swapping_sides_never_contradicts(self, name: str) -> None:
        builder, options = ALL_PAIRS[name]
        left, right = builder()
        forward = decide_equal(left, right, **options)  # type: ignore[arg-type]
        backward = decide_equal(right, left, **options)  # type: ignore[arg-type]
        assert {forward.verdict, backward.verdict} != {
            EqualityVerdict.EQUAL,
            EqualityVerdict.UNEQUAL,
        }
        assert forward.verdict is backward.verdict
        assert forward.method is backward.method


class TestGrammar:
    """Malformed requests raise DecideGrammarError."""

    @pytest.mark.parametrize(
        ("args", "options"),
        [
            ((1, None), {}),
            (("left", "right"), {}),
            ((None, None), {"max_samples": -1}),
            ((None, None), {"max_samples": True}),
            ((None, None), {"max_samples": 2.0}),
            ((None, None), {"samples": "d=2"}),
            ((None, None), {"samples": [1]}),
            ((None, None), {"samples": {"d": 2}}),
            ((None, None), {"guard": None}),
            ((None, None), {"guard": 10}),
            ((None, None), {"rules": "rules"}),
            ((None, None), {"rules": [1]}),
            ((None, None), {"max_depth": -1}),
            ((None, None), {"max_elements": 0}),
            ((None, None), {"use_symbolic": 1}),
            ((None, None), {"use_induction": None}),
            ((None, None), {"tolerance": -1.0}),
            ((None, None), {"tolerance": True}),
            ((None, None), {"tolerance": float("nan")}),
        ],
    )
    def test_malformed_arguments(
        self, args: tuple[object, object], options: dict[str, object]
    ) -> None:
        left, right = (z_state(D) if a is None else a for a in args)
        with pytest.raises(DecideGrammarError):
            decide_equal(left, right, **options)  # type: ignore[arg-type]

    def test_grammar_error_is_a_decide_error(self) -> None:
        assert issubclass(DecideGrammarError, DecideError)

    @pytest.mark.parametrize(
        "fields",
        [
            ("equal", DecisionMethod.NONE, ""),
            (EqualityVerdict.EQUAL, "none", ""),
            (EqualityVerdict.EQUAL, DecisionMethod.NONE, 3),
        ],
    )
    def test_decision_validates_its_fields(self, fields: tuple[object, object, object]) -> None:
        with pytest.raises(DecideGrammarError):
            Decision(*fields)  # type: ignore[arg-type]


class TestRegressions:
    """Previously undecided or mis-sampled cases, each cross-checked against the oracle."""

    def test_dimension_symbol_only_in_a_phase_vector(self) -> None:
        left = Diagram()
        left.add_node(Z_SPIDER, input_dims=[], output_dims=[], phase=PhaseVector(D, {}))
        right = Diagram()
        assert [dict(sample) for sample in sample_grid(left, right)] == [
            {"d": 2},
            {"d": 3},
            {"d": 4},
            {"d": 1},
        ]
        assert not compare(left, right, {"d": 2}).matched
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert dict(decision.counterexample or {}) == {"d": 2}
        assert_oracle_agrees(left, right, decision)

    def test_concrete_base_case_refutation_is_lifted(self) -> None:
        left, right = concrete_false_near_identity()
        assert not compare(left, right, {"n": 0}).matched
        decision = decide_equal(left, right, samples=[{}])
        assert decision.samples_checked == 0
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert dict(decision.counterexample or {}) == {"n": 0}
        assert "base case n=0 refuted" in decision.reason
        assert_oracle_agrees(left, right, decision)

    @pytest.mark.parametrize("name", sorted(INDUCTION_EQUAL))
    def test_induction_pairs_swapped(self, name: str) -> None:
        left, right = INDUCTION_EQUAL[name]()
        forward = decide_equal(
            left, right, guard=NO_FUSION, use_saturation=False, use_symbolic=False
        )
        backward = decide_equal(
            right, left, guard=NO_FUSION, use_saturation=False, use_symbolic=False
        )
        assert backward.verdict is EqualityVerdict.EQUAL, backward.reason
        assert backward.method is DecisionMethod.INDUCTION
        assert ("with sides reversed" in backward.reason) is backward.induction_reversed
        assert not forward.induction_reversed
        assert forward.induction is not None and backward.induction is not None
        assert backward.induction.proved
        assert isomorphic(
            backward.induction.obligation.left_at_base, forward.induction.obligation.left_at_base
        )
        assert_oracle_agrees(right, left, backward)

    def test_caller_samples_reach_the_base_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[object] = []
        inner = decide_module.decide_equal

        def spy(*args: object, **kwargs: object) -> Decision:
            seen.append(kwargs.get("samples"))
            return inner(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(decide_module, "decide_equal", spy)
        left, right = T8._build_false_near_identity()
        samples = [{"d": 3, "n": 1}, {"d": 3, "n": 1, "e": 5}, {"d": 2, "n": 1}]
        assert all(compare(left, right, {"d": d, "n": 1}).matched for d in (2, 3))
        decision = spy(left, right, samples=samples)
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert dict(decision.counterexample or {}) == {"d": 3, "n": 0}
        assert seen[0] == samples
        assert [dict(sample) for sample in seen[1]] == [{"d": 3}, {"d": 2}]  # type: ignore[attr-defined]
        assert_oracle_agrees(left, right, decision)

    def test_default_samples_leave_the_base_case_on_its_grid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[object] = []
        inner = decide_module.decide_equal

        def spy(*args: object, **kwargs: object) -> Decision:
            seen.append(kwargs.get("samples", "absent"))
            return inner(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(decide_module, "decide_equal", spy)
        spy(
            *T8._build_boxed_fusion_family(),
            guard=NO_FUSION,
            use_saturation=False,
            use_symbolic=False,
        )
        assert seen == ["absent", None]


# -- review regressions -----------------------------------------------------------------


def phased_state(dim: Dim, turns: sp.Expr) -> Diagram:
    """A Z state over ``dim`` whose leg-1 phase is ``turns`` turns."""
    return z_state(dim, PhaseVector(dim, {1: Phase(turns)}))


def zx_scalar_pairs(count: int, factor: int = 1) -> Pair:
    """``2 * count`` disconnected Z-into-X pairs against ``count`` Z loops scaled by
    ``factor``, all over dimension 3."""
    dim = Dim.concrete(3)
    left, right = Diagram(), Diagram()
    for _ in range(2 * count):
        a = left.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        b = left.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
        left.add_wire(out(a), inp(b))
    for _ in range(count):
        a = right.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        b = right.add_node(Z_SPIDER, input_dims=[dim], output_dims=[])
        right.add_wire(out(a), inp(b))
    right.multiply_scalar(Scalar.rational(factor))
    return left, right


def vanishing_box_pair() -> Pair:
    """A zero-valued scalar spider and an inner loop, boxed under ``a`` and ``b``, against the
    unboxed scalar spider; they differ only at ``a = 0``."""
    half = PhaseVector(Dim.concrete(2), {1: Phase.turns(sp.Rational(1, 2))})
    one = Dim.concrete(1)
    left = Diagram()
    y = left.add_node(Z_SPIDER, input_dims=[], output_dims=[], phase=half)
    a = left.add_node(Z_SPIDER, input_dims=[], output_dims=[one])
    b = left.add_node(Z_SPIDER, input_dims=[one], output_dims=[])
    left.add_wire(out(a), inp(b))
    outer = left.add_bang_box(Mult("a"), node_scope=frozenset({y, a, b}))
    left.add_bang_box(Mult("b"), node_scope=frozenset({a, b}), parent=outer)
    right = Diagram()
    right.add_node(Z_SPIDER, input_dims=[], output_dims=[], phase=half)
    return left, right


def boxed_cup_vs_identity() -> Pair:
    """A ``Mult(1)``-boxed two-output Z spider against the one-in-one-out identity spider."""
    left = Diagram()
    z = left.add_node(Z_SPIDER, input_dims=[], output_dims=[D, D])
    left.set_boundary_outputs([out(z, 0), out(z, 1)])
    left.add_bang_box(Mult(1), node_scope=frozenset({z}))
    right = Diagram()
    w = right.add_node(Z_SPIDER, input_dims=[D], output_dims=[D])
    right.set_boundary_inputs([inp(w)])
    right.set_boundary_outputs([out(w)])
    return left, right


def deferred_constraint() -> DimensionConstraint:
    """The one DEFERRED constraint the deferred-fusion diagram's normal form assumes."""
    (constraint,) = normal_form(deferred_fusion_diagram()).assumed_constraints
    return constraint


class TestReviewRegressions:
    """Sampling, tolerance, induction-base, assumption, interface and error-handling fixes,
    each cross-checked against the oracle."""

    def test_integer_symbol_is_sampled_at_integers(self) -> None:
        j = sp.Symbol("j", integer=True)
        dim = Dim.concrete(3)
        left, right = phased_state(dim, j), z_state(dim)
        assert [dict(sample) for sample in sample_grid(left, right)] == [
            {"j": 1},
            {"j": 2},
            {"j": 0},
            {"j": 3},
        ]
        assert not compare(left, right, {"j": sp.Rational(1, 3)}).matched
        decision = decide_equal(left, right)
        assert decision.verdict is not EqualityVerdict.UNEQUAL, decision.reason
        assert all(matched for _, matched in oracle_samples(left, right))
        assert_oracle_agrees(left, right, decision)

    def test_positive_symbols_skip_zero(self) -> None:
        k = sp.Symbol("k", integer=True, positive=True)
        r = sp.Symbol("r", positive=True)
        grid = sample_grid(phased_state(D, k), phased_state(D, r))
        assert [sample["k"] for sample in grid[:4]] == [1, 2, 3, 4]
        assert [sample["r"] for sample in grid[:4]] == [
            sp.Rational(1, 3),
            sp.Rational(1, 4),
            sp.Rational(1, 2),
            1,
        ]

    def test_a_sample_breaking_an_assumption_is_never_a_counterexample(self) -> None:
        j = sp.Symbol("j", integer=True)
        dim = Dim.concrete(3)
        left, right = phased_state(dim, j), z_state(dim)
        bad = {"j": sp.Rational(1, 3)}
        assert not compare(left, right, bad).matched
        decision = decide_equal(left, right, samples=[bad], use_symbolic=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN, decision.reason
        assert decision.samples_checked == 0
        assert decision.counterexample is None

    def test_large_magnitude_rounding_is_not_a_counterexample(self) -> None:
        left, right = zx_scalar_pairs(20)
        result = compare(left, right, {})
        assert not result.matched and 0 < result.max_abs_deviation < 1e-3
        magnitude = abs(complex(score(right, {}).tensor))
        assert result.max_abs_deviation <= 1e-9 * magnitude
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN, decision.reason
        assert decision.counterexample is None
        assert decision.samples_checked == 1

    def test_large_magnitude_mismatch_is_still_found(self) -> None:
        left, right = zx_scalar_pairs(20, factor=2)
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert_oracle_agrees(left, right, decision)

    def test_base_one_only_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inner = decide_module._instantiate_index

        def no_zero(diagram: Diagram, index: str, value: int) -> Diagram:
            if value == 0:
                raise DimensionError("patched: no base 0")
            return inner(diagram, index, value)

        outer = decide_module.decide_equal
        calls = [0]

        def base_by_normal_form(*args: object, **kwargs: object) -> Decision:
            calls[0] += 1
            if calls[0] == 1:
                return outer(*args, **kwargs)  # type: ignore[arg-type]
            return outer(*args, **{**kwargs, "guard": DEFAULT_GUARD})  # type: ignore[arg-type]

        monkeypatch.setattr(decide_module, "_instantiate_index", no_zero)
        monkeypatch.setattr(decide_module, "decide_equal", base_by_normal_form)
        left, right = T8._build_boxed_fusion_family()
        decision = base_by_normal_form(
            left, right, guard=NO_FUSION, use_saturation=False, use_symbolic=False
        )
        assert decision.verdict is EqualityVerdict.UNKNOWN, decision.reason
        assert decision.method is DecisionMethod.NONE
        assert "proved for m >= 1 only" in decision.reason
        assert decision.induction is not None and decision.induction.proved
        assert decision.induction.base_value == 1
        assert all(matched for _, matched in oracle_samples(left, right))

    def test_a_claim_failing_at_zero_is_never_equal(self) -> None:
        left, right = vanishing_box_pair()
        assert not compare(left, right, {"a": 0, "b": 0}).matched
        decision = decide_equal(left, right)
        assert decision.verdict is not EqualityVerdict.EQUAL, decision.reason
        assert_oracle_agrees(left, right, decision)

    def test_induction_assumptions_merge_the_base_case(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        constraint = deferred_constraint()
        inner = decide_module.decide_equal
        depth = [0]

        def spy(*args: object, **kwargs: object) -> Decision:
            depth[0] += 1
            try:
                decision = inner(*args, **kwargs)  # type: ignore[arg-type]
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                return decision
            return Decision(
                decision.verdict,
                decision.method,
                decision.reason,
                assumptions=(constraint, constraint),
            )

        monkeypatch.setattr(decide_module, "decide_equal", spy)
        left, right = T8._build_boxed_fusion_family()
        decision = spy(left, right, guard=NO_FUSION, use_saturation=False, use_symbolic=False)
        assert decision.method is DecisionMethod.INDUCTION, decision.reason
        assert decision.induction is not None
        deferred = [
            c
            for tier in decision.induction.tiers
            for step in tier.steps
            for c in step.dimension_constraints
            if c.outcome is ConstraintOutcome.DEFERRED
        ]
        expected = [constraint]
        for c in deferred:
            if c not in expected:
                expected.append(c)
        assert list(decision.assumptions) == expected
        assert_oracle_agrees(left, right, decision)

    def test_symbolic_equal_drops_normal_form_assumptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(decide_module, "same_normal_form", lambda a, b: False)
        monkeypatch.setattr(
            decide_module,
            "_symbolic_match",
            lambda a, b: decide_module._SymbolicRun(True, "patched"),
        )
        left, right = deferred_fusion_diagram(), deferred_fusion_diagram()
        decision = decide_equal(left, right, use_saturation=False)
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION, decision.reason
        assert decision.left_nf is not None and decision.left_nf.assumed_constraints
        assert decision.assumptions == ()
        assert_oracle_agrees(left, right, decision)

    @pytest.mark.parametrize("cap", [0, 24])
    def test_deferred_normal_form_match_is_conditional(self, cap: int) -> None:
        left, right = deferred_fusion_diagram(), deferred_fusion_diagram()
        decision = decide_equal(left, right, max_samples=cap)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.NORMAL_FORM
        assert decision.assumptions == (deferred_constraint(),)
        assert "conditional on" in decision.reason
        assert ("no oracle sample evaluated" in decision.reason) is (cap == 0)
        assert (decision.samples_checked == 0) is (cap == 0)
        assert_oracle_agrees(left, right, decision)

    @pytest.mark.parametrize("cap", [0, 24])
    def test_symbolic_rung_skips_a_boxed_boundary_of_another_interface(self, cap: int) -> None:
        left, right = boxed_cup_vs_identity()
        decision = decide_equal(left, right, max_samples=cap)
        assert decision.verdict is not EqualityVerdict.EQUAL, decision.reason
        if cap == 0:
            assert decision.verdict is EqualityVerdict.UNKNOWN
            assert "boundary input dimensions" in decision.reason
        else:
            assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert_oracle_agrees(left, right, decision)

    def test_symbolic_interface_reason_compares_splits(self) -> None:
        reason = decide_module._symbolic_interface_reason
        assert reason(cup(Z_SPIDER, D), cup(X_SPIDER, D)) is None
        assert reason(*boxed_cup_vs_identity()) is not None
        assert reason(z_state(D), z_state(Dim("e"))) is not None
        assert reason(cup(Z_SPIDER, D), self_loop(D)) is not None

    @pytest.mark.parametrize(
        ("target", "error"),
        [
            ("normal_form", GraphDomainError("patched")),
            ("normal_form", DimensionError("patched")),
            ("certify", ValidateError("patched")),
            ("replay", CertificateDomainError("patched")),
        ],
    )
    def test_domain_errors_in_the_normal_form_rung_fall_through(
        self, monkeypatch: pytest.MonkeyPatch, target: str, error: Exception
    ) -> None:
        def boom(*args: object, **kwargs: object) -> object:
            raise error

        monkeypatch.setattr(decide_module, target, boom)
        left, right = ghz_pair(D)
        decision = decide_equal(
            left, right, use_symbolic=False, use_induction=False, use_saturation=False
        )
        assert decision.verdict is EqualityVerdict.UNKNOWN, decision.reason
        assert f"normal form failed: {type(error).__name__}" in decision.reason
        assert decision.left_nf is None and decision.certificates == ()
        assert all(matched for _, matched in oracle_samples(left, right))

    def test_a_certificate_that_does_not_replay_blocks_normal_form_equal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(certificate: object, **kwargs: object) -> ReplayResult:
            return ReplayResult(False, "patched", Diagram(), ())

        monkeypatch.setattr(decide_module, "replay", failing)
        left, right = ghz_pair(D)
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN, decision.reason
        assert decision.method is DecisionMethod.NONE
        assert "certificate did not replay: patched" in decision.reason
        assert all(matched for _, matched in oracle_samples(left, right))

    def test_normal_form_equal_replays_both_certificates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[object] = []

        def spy(certificate: object, **kwargs: object) -> ReplayResult:
            seen.append(certificate)
            return replay(certificate, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(decide_module, "replay", spy)
        decision = decide_equal(*ghz_pair(D))
        assert decision.method is DecisionMethod.NORMAL_FORM
        assert seen == list(decision.certificates)


# -- the saturation rung ----------------------------------------------------------------


def bialgebra_pair(d: int = 2) -> Pair:
    """``bialgebra_diagram(d)`` and its BIALGEBRA rewrite."""
    left = bialgebra_diagram(d)
    return left, apply(left, BIALGEBRA, BIALGEBRA.pattern.find_matches(left)[0]).diagram


def deferred_fusion_pair() -> Pair:
    """The deferred-fusion diagram and its one SPIDER_FUSION rewrite."""
    left = deferred_fusion_diagram()
    rule = lookup_rule("spider_fusion")
    return left, apply(left, rule, rule.pattern.find_matches(left)[0]).diagram


class TestSaturationRung:
    """The SATURATION rung: settled pairs, fall-through, evidence and argument checks."""

    @pytest.mark.parametrize("d", [2, 3])
    def test_bialgebra_pair_settles_by_saturation_not_normal_form(self, d: int) -> None:
        left, right = bialgebra_pair(d)
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.SATURATION
        assert decision.left_nf is not None and decision.right_nf is not None
        assert not same_normal_form(decision.left_nf, decision.right_nf)
        assert decision.saturation is not None and decision.saturation.merges >= 1
        assert decision.assumptions == ()
        assert decision.samples_checked >= 1
        assert "1 edge(s)" in decision.reason
        assert_oracle_agrees(left, right, decision)

    def test_edge_certificates_replay_verify_and_chain(self) -> None:
        left, right = bialgebra_pair()
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert len(decision.certificates) == 1
        (certificate,) = decision.certificates
        assert certificate.derivation.label == "saturation edge 0: bialgebra"
        assert replay(certificate).reproduced
        assert isomorphic(comparison_view(certificate.initial), comparison_view(left))
        assert isomorphic(comparison_view(certificate.final), comparison_view(right))
        assert verify(certificate, {}).verified

    def test_deferred_constraints_become_assumptions(self) -> None:
        left, right = deferred_fusion_pair()
        decision = decide_equal(
            left, right, guard=NO_FUSION, use_symbolic=False, use_induction=False
        )
        assert decision.method is DecisionMethod.SATURATION, decision.reason
        assert decision.assumptions == (deferred_constraint(),)
        assert "conditional on the 1 deferred dimension constraint(s)" in decision.reason
        assert f"no oracle mismatch in {decision.samples_checked} sample(s)" in decision.reason
        assert decision.samples_checked >= 1
        assert_oracle_agrees(left, right, decision)

    def test_sides_sharing_an_e_node_need_no_edge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(decide_module, "same_normal_form", lambda a, b: False)
        left, right = ghz_pair(D)[0], ghz_pair(D)[0]
        decision = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.SATURATION
        assert decision.certificates == () and decision.assumptions == ()
        assert "0 edge(s), both sides one e-node" in decision.reason
        assert_oracle_agrees(left, right, decision)

    def test_disabled_saturation_falls_through(self) -> None:
        left, right = bialgebra_pair()
        decision = decide_equal(
            left, right, use_saturation=False, use_symbolic=False, use_induction=False
        )
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.method is DecisionMethod.NONE
        assert decision.saturation is None
        assert "saturation disabled" in decision.reason

    def test_disabled_saturation_leaves_the_symbolic_rung(self) -> None:
        left, right = bialgebra_pair()
        decision = decide_equal(left, right, use_saturation=False)
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION, decision.reason
        assert decision.saturation is None

    def test_an_unsettled_rung_still_reports(self) -> None:
        left, right = zero_phase_pair()
        decision = decide_equal(left, right)
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION, decision.reason
        assert decision.saturation is not None and decision.saturation.merges == 0
        unknown = decide_equal(left, right, use_symbolic=False, use_induction=False)
        assert unknown.verdict is EqualityVerdict.UNKNOWN
        assert unknown.saturation is not None
        assert "saturation: no merge" in unknown.reason

    def test_rungs_before_saturation_leave_no_report(self) -> None:
        assert decide_equal(*ghz_pair(D)).saturation is None
        assert decide_equal(*UNEQUAL_ORACLE["different_phases"]()).saturation is None
        assert decide_equal(*UNEQUAL_INTERFACE["arity"]()).saturation is None

    def test_limits_bound_the_rung(self) -> None:
        left, right = bialgebra_pair()
        limits = SaturationLimits(max_iterations=0)
        decision = decide_equal(
            left, right, saturation_limits=limits, use_symbolic=False, use_induction=False
        )
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.saturation is not None
        assert decision.saturation.stop_reason is SaturationStop.ITERATION_LIMIT

    def test_rules_replace_the_default_saturation_rules(self) -> None:
        left, right = bialgebra_pair()
        settled = decide_equal(
            left, right, rules=[BIALGEBRA], guard=NO_FUSION, use_symbolic=False, use_induction=False
        )
        assert settled.method is DecisionMethod.SATURATION, settled.reason
        decision = decide_equal(
            left, right, rules=[STATE_COPY], use_symbolic=False, use_induction=False
        )
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.saturation is not None and decision.saturation.applications == 0

    def test_a_saturation_error_becomes_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Failing(EGraph):
            def saturate(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
                raise GraphDomainError("patched")

        monkeypatch.setattr(decide_module, "EGraph", Failing)
        decision = decide_equal(*bialgebra_pair(), use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert decision.saturation is None
        assert "saturation: failed: GraphDomainError: patched" in decision.reason

    def test_an_edge_that_does_not_replay_blocks_equal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(certificate: object, **kwargs: object) -> ReplayResult:
            return ReplayResult(False, "patched", Diagram(), ())

        monkeypatch.setattr(decide_module, "replay", failing)
        decision = decide_equal(*bialgebra_pair(), use_symbolic=False, use_induction=False)
        assert decision.verdict is EqualityVerdict.UNKNOWN
        assert "edge 0 did not replay: patched" in decision.reason

    def test_induction_forwards_the_saturation_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[object, object]] = []
        inner = decide_module.decide_equal

        def spy(*args: object, **kwargs: object) -> Decision:
            seen.append((kwargs.get("use_saturation"), kwargs.get("saturation_limits")))
            return inner(*args, **kwargs)  # type: ignore[arg-type]

        limits = SaturationLimits(max_iterations=3)
        monkeypatch.setattr(decide_module, "decide_equal", spy)
        left, right = T8._build_boxed_fusion_family()
        decision = spy(
            left,
            right,
            guard=NO_FUSION,
            use_saturation=False,
            saturation_limits=limits,
            use_symbolic=False,
        )
        assert decision.method is DecisionMethod.INDUCTION, decision.reason
        assert seen == [(False, limits), (False, limits)]

    def test_default_limits(self) -> None:
        assert DECIDE_SATURATION_LIMITS == SaturationLimits(
            max_iterations=3, max_enodes=128, max_applications=1024, node_margin=4
        )

    @pytest.mark.parametrize("value", [1, 0, None, "yes"])
    def test_use_saturation_must_be_a_bool(self, value: object) -> None:
        with pytest.raises(DecideGrammarError, match="use_saturation"):
            decide_equal(*ghz_pair(D), use_saturation=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [None, 4, {}, (4, 256, 4096, 4)])
    def test_saturation_limits_must_be_saturation_limits(self, value: object) -> None:
        with pytest.raises(DecideGrammarError, match="saturation_limits"):
            decide_equal(*ghz_pair(D), saturation_limits=value)  # type: ignore[arg-type]


def _phased_cap(phase: PhaseVector | None) -> Diagram:
    """A Z state, phased by ``phase`` when given, capped by a phaseless X effect."""
    diagram = Diagram()
    if phase is None:
        z = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[D])
    else:
        z = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[D], phase=phase)
    x = diagram.add_node(X_SPIDER, input_dims=[D], output_dims=[])
    diagram.add_wire(PortRef(z, Direction.OUTPUT, 0), PortRef(x, Direction.INPUT, 0))
    return diagram


def _ring(size: int, phase: PhaseVector | None) -> Diagram:
    """``size`` alternating Z and X spiders in a closed ring, the first with a second output on
    the boundary and carrying ``phase`` when given, every other capped by its own colour."""
    diagram = Diagram()
    nodes = []
    for position in range(size):
        generator = Z_SPIDER if position % 2 == 0 else X_SPIDER
        if position == 0 and phase is not None:
            nodes.append(diagram.add_node(generator, [D], [D, D], phase=phase))
        else:
            nodes.append(diagram.add_node(generator, [D], [D, D]))
    for position, node in enumerate(nodes):
        following = nodes[(position + 1) % size]
        diagram.add_wire(PortRef(node, Direction.OUTPUT, 1), PortRef(following, Direction.INPUT, 0))
    diagram.set_boundary_outputs([PortRef(nodes[0], Direction.OUTPUT, 0)])
    for position, node in enumerate(nodes[1:], start=1):
        colour = Z_SPIDER if position % 2 == 0 else X_SPIDER
        cap = diagram.add_node(colour, input_dims=[D], output_dims=[])
        diagram.add_wire(PortRef(node, Direction.OUTPUT, 0), PortRef(cap, Direction.INPUT, 0))
    return diagram


def _phased_ring(size: int) -> Diagram:
    return _ring(size, PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))}))


class TestOracleAtScale:
    """The oracle evaluates diagrams past einsum's 52 labels, and says why it skips a sample."""

    def test_an_extra_phase_on_a_large_ring_is_refuted(self) -> None:
        phase = PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))})
        left, right = _ring(30, None), _ring(30, phase)
        assert len(left.wires) > 52
        decision = decide_equal(left, right)
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.ORACLE_COUNTEREXAMPLE
        assert decision.samples_checked >= 1

    def test_refused_samples_carry_their_reason(self) -> None:
        samples = [{"d": 2}, {"d": 3}]
        refutation = refute_by_oracle(
            _ring(30, None), _ring(30, None), samples=samples, max_elements=1
        )
        assert not refutation.evaluated
        assert refutation.refusals and "ContractSizeError" in refutation.refusals[0]
        decision = decide_equal(
            _ring(4, None), _phased_ring(4), samples=samples, max_elements=1, use_symbolic=False
        )
        assert decision.samples_checked == 0
        assert "no oracle sample evaluated; 2 sample(s) refused" in decision.reason


def wide_ghz(legs: int, dim: Dim, phase: PhaseVector | None = None) -> Diagram:
    """One Z spider with ``legs`` boundary outputs over ``dim``, carrying ``phase``."""
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim] * legs, phase=phase)
    diagram.set_boundary_outputs([out(node, i) for i in range(legs)])
    return diagram


def wide_ghz_phase_pair(legs: int = 24, dim: Dim = D) -> Pair:
    """A ``legs``-leg GHZ spider against itself with a quarter-turn phase at index 1."""
    return wide_ghz(legs, dim), wide_ghz(legs, dim, phase_turns(dim, sp.Rational(1, 4)))


class TestSymbolicWitness:
    """An entry of two differing symbolic contractions refutes a pair the oracle refuses."""

    @pytest.mark.parametrize("legs", [24, 40])
    @pytest.mark.parametrize("dim", [D, Dim.concrete(2)], ids=["d", "2"])
    def test_wide_ghz_with_one_phase_is_unequal(self, legs: int, dim: Dim) -> None:
        decision = decide_equal(*wide_ghz_phase_pair(legs, dim))
        assert decision.verdict is EqualityVerdict.UNEQUAL, decision.reason
        assert decision.method is DecisionMethod.SYMBOLIC_WITNESS
        assert decision.samples_checked == 0
        witness = decision.witness
        assert witness is not None and witness.index == (1,) * legs
        assert witness.left == 1 and abs(witness.right - 1j) < 1e-12
        assert decision.counterexample == witness.assignment

    def test_the_witness_matches_the_oracle_at_small_size(self) -> None:
        left, right = wide_ghz_phase_pair(3)
        decision = decide_equal(left, right, use_saturation=False, max_samples=0)
        assert decision.method is DecisionMethod.SYMBOLIC_WITNESS, decision.reason
        witness = decision.witness
        assert witness is not None
        tensor_l = score(left, witness.assignment).tensor
        tensor_r = score(right, witness.assignment).tensor
        assert abs(tensor_l[witness.index] - witness.left) < 1e-12
        assert abs(tensor_r[witness.index] - witness.right) < 1e-12

    def test_a_ring_the_oracle_refuses_is_refuted(self) -> None:
        decision = decide_equal(
            _ring(4, None), _phased_ring(4), samples=[{"d": 2}, {"d": 3}], max_elements=1
        )
        assert decision.method is DecisionMethod.SYMBOLIC_WITNESS, decision.reason
        assert decision.samples_checked == 0

    def test_equal_pairs_have_no_witness(self) -> None:
        assert refute_by_symbolic_witness(*zero_phase_pair()) is None
        assert refute_by_symbolic_witness(wide_ghz(24, D), wide_ghz(24, D)) is None

    def test_a_difference_below_the_dimension_floor_is_no_witness(self) -> None:
        phase = PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))})
        assert refute_by_symbolic_witness(_phased_cap(phase), _phased_cap(None)) is None

    def test_refute_by_symbolic_witness_takes_samples(self) -> None:
        left, right = wide_ghz_phase_pair(30)
        witness = refute_by_symbolic_witness(left, right, samples=[{"d": 5}])
        assert witness is not None and dict(witness.assignment) == {"d": 5}
        assert refute_by_symbolic_witness(left, right, samples=[{"d": 1}]) is None


class TestDimensionFloors:
    """A phase index k over a bare dimension symbol d confines the family to d > k."""

    def test_floors_come_from_phase_indices(self) -> None:
        phase = PhaseVector(
            D, {1: Phase.turns(sp.Rational(1, 3)), 3: Phase.turns(1 / sp.Integer(2))}
        )
        assert dimension_floors(_phased_cap(phase), _phased_cap(None)) == {"d": 4}
        assert dimension_floors(_phased_cap(None)) == {}

    def test_a_difference_living_only_below_the_floor_is_equal(self) -> None:
        phase = PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))})
        decision = decide_equal(_phased_cap(phase), _phased_cap(None), use_saturation=False)
        assert decision.verdict is EqualityVerdict.EQUAL, decision.reason
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION
        assert "given d >= 2" in decision.reason
