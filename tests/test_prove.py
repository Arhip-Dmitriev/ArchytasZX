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

"""Phase 15 proving suite: ``prove`` statuses, proof certificates and their checks, tampering,
the ``decide`` wrappers, and argument validation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.compare import compare_structure, isomorphic
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram
from archytaszx.rewrite.engine import apply, apply_until_fixpoint
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import ConstraintOutcome, RewriteGrammarError
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.rewrite.tactics import (
    ProofPath,
    SearchLimits,
    SearchStatus,
    Tactic,
    TacticContext,
    TacticOutcome,
    rule,
    search,
)
from archytaszx.semantics import prove as prove_module
from archytaszx.semantics.certificate import (
    Certificate,
    Derivation,
    DerivationKind,
    InductionClaim,
    certify,
    replay,
    verify,
)
from archytaszx.semantics.check import compare
from archytaszx.semantics.decide import (
    DecideGrammarError,
    OracleRefutation,
    interface_reason,
    refute_by_oracle,
    sample_grid,
)
from archytaszx.semantics.prove import (
    ProofCertificate,
    ProofOutcome,
    ProofStatus,
    ProveError,
    ProveGrammarError,
    certify_proof,
    check_proof,
    prove,
)

from . import test_decide as TD
from .test_normal_form import _chain, _single_spider
from .test_rules_library_phase11 import identity_chain, inp, out
from .test_tactics import _two_spiders

D = Dim("d")


def _chain_d() -> Diagram:
    """Three phased Z spiders in a line over ``d``."""
    return _chain((0, 1, 2), D)


def _fused_d() -> Diagram:
    """``_chain`` in another node order, fused to a fixpoint."""
    return apply_until_fixpoint(_chain((2, 1, 0), D), [SPIDER_FUSION]).diagram


def _deferred_pair() -> tuple[Diagram, Diagram]:
    """A fusion whose application records a ``DEFERRED`` constraint, and its result."""
    start = TD.deferred_fusion_diagram()
    match = SPIDER_FUSION.pattern.find_matches(start)[0]
    return start, apply(start, SPIDER_FUSION, match).diagram


def _proved(start: Diagram, goal: Diagram, **kwargs: object) -> ProofOutcome:
    """``prove(start, goal)``, asserted PROVED with every evidence field set."""
    outcome = prove(start, goal, **kwargs)  # type: ignore[arg-type]
    assert outcome.status is ProofStatus.PROVED, outcome.reason
    assert outcome.proved
    assert outcome.certificate is not None and outcome.check is not None
    assert outcome.search is not None and outcome.search.found
    assert outcome.check.verified and outcome.counterexample is None
    return outcome


def _certificate() -> ProofCertificate:
    """The certificate of the chain-to-fused proof over ``d``."""
    certificate = _proved(_chain_d(), _fused_d()).certificate
    assert certificate is not None
    return certificate


def _two_sided() -> ProofCertificate:
    """A certificate with a non-empty half on each side."""
    path = search(identity_chain(3), identity_chain(3, X_SPIDER)).path
    assert path is not None and path.forward and path.backward
    return certify_proof(path)


def _mismatched_wire() -> Diagram:
    """Two one-in-one-out Z spiders, over 2 and over 3, wired in series."""
    diagram = Diagram()
    first = diagram.add_node(Z_SPIDER, [Dim.concrete(2)], [Dim.concrete(2)])
    second = diagram.add_node(Z_SPIDER, [Dim.concrete(3)], [Dim.concrete(3)])
    diagram.add_wire(out(first), inp(second))
    diagram.set_boundary_inputs([inp(first)])
    diagram.set_boundary_outputs([out(second)])
    return diagram


def _boxed_internal_port() -> Diagram:
    """``identity_chain(3)`` with a port bang box over its state's wired output."""
    diagram = identity_chain(3)
    diagram.add_bang_box(Mult("n"), port_scope=frozenset({out(min(diagram.nodes))}))
    return diagram


class _Raising(Tactic):
    """A move whose outcomes raise RewriteGrammarError."""

    @property
    def name(self) -> str:
        return "raising"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        raise RewriteGrammarError("raising move")


class TestProved:
    def test_certificate_halves_replay_and_verify(self) -> None:
        start, goal = _chain_d(), _fused_d()
        outcome = _proved(start, goal)
        certificate = outcome.certificate
        assert certificate is not None
        assert compare_structure(certificate.start, start).identical
        assert compare_structure(certificate.goal, goal).identical
        assert certificate.forward.derivation.label == "forward"
        assert certificate.backward.derivation.label == "backward"
        for half in (certificate.forward, certificate.backward):
            assert half.derivation.kind is DerivationKind.STEP_SEQUENCE
            assert replay(half).reproduced
            for assignment in sample_grid(half.initial, half.initial)[:3]:
                report = verify(half, assignment)
                assert report.verified, report.reason
        assert certificate.length == len(certificate.forward.steps) + len(
            certificate.backward.steps
        )
        assert outcome.check is not None and outcome.check.samples_checked >= 2
        assert outcome.check.meet_isomorphic
        assert outcome.assumptions == () and "conditional" not in outcome.reason

    def test_two_sided_certificate_checks(self) -> None:
        certificate = _two_sided()
        assert certificate.moves and certificate.backward_moves
        check = check_proof(certificate)
        assert check.verified, check.reason
        assert check.forward_replay is not None and check.forward_replay.reproduced
        assert check.backward_replay is not None and check.backward_replay.reproduced
        assert isomorphic(
            comparison_view(check.forward_replay.diagram),
            comparison_view(check.backward_replay.diagram),
        )
        assert check_proof(certificate, rediscover=False).verified

    def test_certify_proof_mirrors_the_path(self) -> None:
        path = search(_chain_d(), _fused_d(), moves=(rule("spider_fusion"),)).path
        assert path is not None
        certificate = certify_proof(path)
        assert certificate.start is path.start and certificate.goal is path.goal
        assert certificate.moves == path.moves
        assert certificate.backward_moves == path.backward_moves
        assert certificate.forward.steps == tuple(r.step for r in path.forward_results)
        assert certificate.backward.steps == tuple(r.step for r in path.backward_results)

    def test_isomorphic_inputs_prove_in_zero_steps(self) -> None:
        start, goal = _chain_d(), _chain((2, 0, 1), D)
        outcome = _proved(start, goal)
        assert outcome.certificate is not None and outcome.certificate.length == 0
        assert outcome.reason.startswith("proved in 0 step(s)")

    def test_backward_only_proof(self) -> None:
        outcome = _proved(_single_spider(D, 3), _two_spiders())
        assert outcome.certificate is not None
        assert outcome.certificate.forward.steps == ()
        assert len(outcome.certificate.backward.steps) == 1

    def test_deferred_assumptions_are_carried(self) -> None:
        start, goal = _deferred_pair()
        outcome = _proved(start, goal)
        assert outcome.certificate is not None
        assert len(outcome.assumptions) == 1
        assert outcome.assumptions == outcome.certificate.assumptions
        assert all(c.outcome is ConstraintOutcome.DEFERRED for c in outcome.assumptions)
        assert "conditional on the 1 deferred dimension constraint(s)" in outcome.reason

    def test_assumptions_deduplicate_across_halves(self) -> None:
        start, goal = _deferred_pair()
        certificate = _proved(start, goal).certificate
        assert certificate is not None
        doubled = replace(certificate, backward=certificate.forward)
        assert doubled.assumptions == certificate.assumptions

    def test_inputs_are_not_mutated(self) -> None:
        start, goal = _chain_d(), _fused_d()
        snapshots = (start.copy(), goal.copy())
        outcome = _proved(start, goal)
        assert compare_structure(start, snapshots[0]).identical
        assert compare_structure(goal, snapshots[1]).identical
        assert outcome.certificate is not None
        assert outcome.certificate.start is not start and outcome.certificate.goal is not goal

    def test_determinism(self) -> None:
        def summary() -> tuple[object, ...]:
            outcome = prove(_chain_d(), _fused_d())
            certificate = outcome.certificate
            assert certificate is not None
            steps = tuple(
                repr(s) for s in (*certificate.forward.steps, *certificate.backward.steps)
            )
            return outcome.status, outcome.reason, certificate.moves, steps

        assert summary() == summary()

    def test_explicit_samples_drive_the_check(self) -> None:
        outcome = _proved(_chain_d(), _fused_d(), samples=[{"d": 2}, {"d": 3}])
        assert outcome.check is not None and outcome.check.samples_checked == 2
        none = _proved(_chain_d(), _fused_d(), max_samples=0)
        assert none.check is not None and none.check.samples_checked == 0
        assert "no oracle sample evaluated" in none.reason


class TestCheckProofTampering:
    def test_untampered_certificate_verifies(self) -> None:
        check = check_proof(_certificate())
        assert check.verified and check.meet_isomorphic and check.counterexample is None
        assert check.reason.startswith("both halves replay and meet")

    def test_wrong_start(self) -> None:
        check = check_proof(replace(_certificate(), start=_single_spider(D, 3)))
        assert not check.verified and "forward half does not begin at start" in check.reason
        assert check.forward_replay is None and check.samples_checked == 0

    def test_wrong_goal(self) -> None:
        check = check_proof(replace(_certificate(), goal=_chain_d()))
        assert not check.verified and "backward half does not begin at goal" in check.reason

    def test_swapped_halves(self) -> None:
        certificate = _two_sided()
        swapped = replace(certificate, forward=certificate.backward, backward=certificate.forward)
        check = check_proof(swapped)
        assert not check.verified and "does not begin at start" in check.reason

    def test_truncated_steps(self) -> None:
        certificate = _certificate()
        derivation = certificate.forward.derivation
        assert len(derivation.steps) >= 1
        truncated = Certificate(replace(derivation, steps=derivation.steps[:-1]))
        check = check_proof(replace(certificate, forward=truncated))
        assert not check.verified
        assert "forward half did not replay" in check.reason
        assert check.forward_replay is not None and not check.forward_replay.reproduced

    def test_halves_that_do_not_meet(self) -> None:
        certificate = _certificate()
        stalled = certify(certificate.start, [], label="forward")
        check = check_proof(replace(certificate, forward=stalled))
        assert not check.verified and not check.meet_isomorphic
        assert "different comparison views" in check.reason
        assert check.forward_replay is not None and check.backward_replay is not None

    def test_oracle_counterexample_fails_the_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        certificate = _certificate()
        honest = refute_by_oracle(certificate.start, certificate.goal)
        mismatch = compare(*TD.UNEQUAL_ORACLE["different_phases"](), {"d": 2})
        assert not mismatch.matched
        forged = OracleRefutation(honest.evaluated[:1], honest.evaluated[0], mismatch)

        def fake(*args: object, **kwargs: object) -> OracleRefutation:
            return forged

        monkeypatch.setattr(prove_module, "refute_by_oracle", fake)
        check = check_proof(certificate)
        assert not check.verified and check.meet_isomorphic
        assert check.counterexample == honest.evaluated[0]
        assert check.comparison is mismatch and check.samples_checked == 1
        assert check.reason.startswith("oracle mismatch at")

    def test_modified_final(self) -> None:
        certificate = _certificate()
        final = certificate.forward.final.copy()
        final.multiply_scalar(Scalar(2))
        forged = Certificate(replace(certificate.forward.derivation, final=final))
        check = check_proof(replace(certificate, forward=forged))
        assert not check.verified and "final diagram differs: scalar differs" in check.reason

    def test_steps_from_another_diagram(self) -> None:
        certificate = _certificate()
        foreign = _two_sided().forward.steps
        forged = Certificate(replace(certificate.forward.derivation, steps=foreign))
        check = check_proof(replace(certificate, forward=forged))
        assert not check.verified and "forward half did not replay" in check.reason

    @pytest.mark.parametrize("scaled", [False, True])
    def test_goal_replaced_by_an_unequal_diagram(self, scaled: bool) -> None:
        certificate = _certificate()
        if scaled:
            goal = certificate.goal.copy()
            goal.multiply_scalar(Scalar(2))
        else:
            goal = TD.UNEQUAL_ORACLE["different_phases"]()[0]
        forged = replace(certificate, goal=goal, backward=certify(goal, [], label="backward"))
        check = check_proof(forged)
        assert not check.verified and "different comparison views" in check.reason

    @pytest.mark.parametrize("build", [_mismatched_wire, _boxed_internal_port])
    def test_invalid_endpoints(self, build: Callable[[], Diagram]) -> None:
        diagram = build()
        half = certify(diagram, [], label="forward")
        forged = ProofCertificate(diagram, diagram.copy(), half, half, (), ())
        check = check_proof(forged)
        assert not check.verified and check.reason.startswith("start fails validation: ")
        assert check.forward_replay is None
        valid = _chain_d()
        check = check_proof(ProofCertificate(valid, diagram, certify(valid, []), half, (), ()))
        assert not check.verified and check.reason.startswith("goal fails validation: ")

    def test_replay_error_is_reported(self) -> None:
        certificate = _certificate()
        step = certificate.forward.steps[0]
        renamed = replace(certificate.forward.derivation, steps=(replace(step, rule_name="nope"),))
        check = check_proof(replace(certificate, forward=Certificate(renamed)))
        assert not check.verified and "rule lookup failed" in check.reason


class TestRefutedAndNotFound:
    @pytest.mark.parametrize("name", sorted(TD.UNEQUAL_INTERFACE))
    def test_refuted_by_interface(self, name: str) -> None:
        start, goal = TD.UNEQUAL_INTERFACE[name]()
        outcome = prove(start, goal)
        assert outcome.status is ProofStatus.REFUTED and not outcome.proved
        assert outcome.reason.startswith("interfaces never agree: boundary")
        assert outcome.search is None and outcome.certificate is None
        assert outcome.counterexample is None

    @pytest.mark.parametrize("name", ["different_phases", "extra_scalar", "z_cup_vs_x_cup"])
    def test_refuted_by_oracle(self, name: str) -> None:
        start, goal = TD.UNEQUAL_ORACLE[name]()
        outcome = prove(start, goal)
        assert outcome.status is ProofStatus.REFUTED
        assert outcome.counterexample is not None and outcome.search is None
        assert outcome.reason.startswith("oracle mismatch at")
        result = compare(TD.family(start), TD.family(goal), outcome.counterexample)
        assert not result.matched

    def test_refute_false_skips_the_oracle_and_searches(self) -> None:
        start, goal = TD.UNEQUAL_ORACLE["different_phases"]()
        outcome = prove(start, goal, refute=False)
        assert outcome.status is ProofStatus.NOT_FOUND
        assert outcome.search is not None and outcome.counterexample is None
        assert prove(start, goal, refute=False, moves=()).status is ProofStatus.NOT_FOUND

    def test_refute_false_still_checks_with_the_oracle(self) -> None:
        outcome = _proved(_chain_d(), _fused_d(), refute=False)
        assert outcome.check is not None and outcome.check.samples_checked >= 2

    @pytest.mark.parametrize(
        ("kwargs", "status"),
        [
            ({"moves": ()}, SearchStatus.EXHAUSTED),
            ({"limits": SearchLimits(max_depth=0)}, SearchStatus.DEPTH_LIMIT),
            ({"limits": SearchLimits(max_states=2)}, SearchStatus.STATE_LIMIT),
            ({"limits": SearchLimits(max_applications=0)}, SearchStatus.APPLICATION_LIMIT),
        ],
    )
    def test_not_found(self, kwargs: dict[str, object], status: SearchStatus) -> None:
        outcome = prove(_chain_d(), _single_spider(D, 3), refute=False, **kwargs)  # type: ignore[arg-type]
        assert outcome.status is ProofStatus.NOT_FOUND and not outcome.proved
        assert outcome.search is not None and outcome.search.status is status
        assert outcome.certificate is None and outcome.check is None
        assert outcome.reason.startswith(f"search {status.value} after ")
        for counter in ("state(s)", "expanded", "application(s)", "pruned", "failure(s)"):
            assert counter in outcome.reason

    def test_check_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        start, goal = _chain_d(), _fused_d()
        real = search(start, goal)

        def fake(*args: object, **kwargs: object) -> object:
            return replace(real, path=ProofPath(start.copy(), goal.copy(), (), ()))

        monkeypatch.setattr(prove_module, "search", fake)
        outcome = prove(start, goal)
        assert outcome.status is ProofStatus.CHECK_FAILED and not outcome.proved
        assert outcome.reason.startswith("proof check failed: ")
        assert outcome.certificate is not None and outcome.check is not None
        assert not outcome.check.verified and outcome.assumptions == ()

    @pytest.mark.parametrize("refute", [True, False])
    def test_the_oracle_runs_once(self, refute: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, object]] = []
        real = refute_by_oracle

        def counting(*args: object, **kwargs: object) -> OracleRefutation:
            calls.append(kwargs)
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(prove_module, "refute_by_oracle", counting)
        samples = [{"d": 2}, {"d": 4}]
        outcome = _proved(_chain_d(), _fused_d(), refute=refute, samples=samples, max_samples=1)
        assert len(calls) == 1
        assert calls[0]["samples"] is samples and calls[0]["max_samples"] == 1
        assert outcome.check is not None and outcome.check.samples_checked == 1

    def test_rewrite_side_errors_propagate(self) -> None:
        with pytest.raises(RewriteGrammarError, match="raising move"):
            prove(_chain_d(), _fused_d(), moves=(_Raising(),))


class TestDecideWrappers:
    def test_interface_reason(self) -> None:
        start, goal = TD.UNEQUAL_INTERFACE["arity"]()
        assert interface_reason(start, goal) == "boundary output count 3 != 2"
        dims = TD.UNEQUAL_INTERFACE["dimension"]()
        assert (reason := interface_reason(*dims)) is not None and "never equal" in reason
        assert interface_reason(_chain_d(), _fused_d()) is None

    def test_interface_reason_ignores_parameters(self) -> None:
        left, right = TD.z_state(D), TD.z_state(Dim.concrete(3))
        left.set_parameters({"d": 2})
        assert interface_reason(left, right) is None
        assert dict(left.parameters) == {"d": 2}

    def test_refute_by_oracle(self) -> None:
        equal = refute_by_oracle(_chain_d(), _fused_d())
        assert not equal.refuted and equal.counterexample is None and equal.comparison is None
        assert len(equal.evaluated) >= 2
        start, goal = TD.UNEQUAL_ORACLE["different_phases"]()
        unequal = refute_by_oracle(start, goal)
        assert unequal.refuted and unequal.counterexample == unequal.evaluated[-1]
        assert unequal.comparison is not None and not unequal.comparison.matched

    def test_refute_by_oracle_samples_and_cap(self) -> None:
        run = refute_by_oracle(_chain_d(), _fused_d(), samples=[{"d": 5}, {"d": 2}])
        assert [dict(s) for s in run.evaluated] == [{"d": 5}, {"d": 2}]
        assert refute_by_oracle(_chain_d(), _fused_d(), max_samples=0).evaluated == ()
        assert len(refute_by_oracle(_chain_d(), _fused_d(), max_samples=1).evaluated) == 1

    @pytest.mark.parametrize(
        "call",
        [
            lambda: interface_reason("x", _chain_d()),  # type: ignore[arg-type]
            lambda: refute_by_oracle(_chain_d(), None),  # type: ignore[arg-type]
            lambda: refute_by_oracle(_chain_d(), _chain_d(), samples="d"),  # type: ignore[arg-type]
            lambda: refute_by_oracle(_chain_d(), _chain_d(), max_samples=-1),
            lambda: refute_by_oracle(_chain_d(), _chain_d(), max_samples=True),
            lambda: refute_by_oracle(_chain_d(), _chain_d(), tolerance=-1.0),
            lambda: refute_by_oracle(_chain_d(), _chain_d(), max_elements=0),
        ],
    )
    def test_wrapper_validation(self, call: Callable[[], object]) -> None:
        with pytest.raises(DecideGrammarError):
            call()


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"moves": "rule"},
            {"moves": [rule("spider_fusion"), 3]},
            {"limits": None},
            {"bidirectional": 1},
            {"refute": None},
            {"cache": object()},
            {"samples": "d"},
            {"samples": [3]},
            {"max_samples": -1},
            {"max_samples": False},
            {"tolerance": -0.5},
            {"tolerance": True},
            {"max_elements": 0},
        ],
    )
    def test_prove_rejects_bad_arguments(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ProveGrammarError):
            prove(_chain_d(), _fused_d(), **kwargs)  # type: ignore[arg-type]

    def test_prove_rejects_non_diagrams(self) -> None:
        with pytest.raises(ProveGrammarError):
            prove("a", _chain_d())  # type: ignore[arg-type]
        with pytest.raises(ProveGrammarError):
            prove(_chain_d(), None)  # type: ignore[arg-type]
        assert issubclass(ProveGrammarError, ProveError)

    @pytest.mark.parametrize("build", [_mismatched_wire, _boxed_internal_port])
    def test_prove_rejects_invalid_diagrams(self, build: Callable[[], Diagram]) -> None:
        with pytest.raises(ProveGrammarError, match="^start fails validation: "):
            prove(build(), build())
        with pytest.raises(ProveGrammarError, match="^goal fails validation: "):
            prove(identity_chain(3), build())

    def test_bad_moves_are_rejected_before_an_interface_refutation(self) -> None:
        start, goal = TD.UNEQUAL_INTERFACE["arity"]()
        with pytest.raises(ProveGrammarError):
            prove(start, goal, moves=["x"])  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "kwargs",
        [{"rediscover": 1}, {"samples": "d"}, {"max_samples": -1}, {"tolerance": -1.0}],
    )
    def test_check_proof_rejects_bad_arguments(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ProveGrammarError):
            check_proof(_certificate(), **kwargs)  # type: ignore[arg-type]
        with pytest.raises(ProveGrammarError):
            check_proof("certificate")  # type: ignore[arg-type]

    def test_certify_proof_rejects_a_non_path(self) -> None:
        with pytest.raises(ProveGrammarError):
            certify_proof(_chain_d())  # type: ignore[arg-type]

    def test_proof_certificate_fields(self) -> None:
        certificate = _certificate()
        induction = Derivation(
            DerivationKind.INDUCTION,
            certificate.start,
            certificate.start,
            children=(certificate.forward.derivation, certificate.forward.derivation),
            induction=InductionClaim("m", 0),
        )
        bad: list[Mapping[str, object]] = [
            {"start": None},
            {"goal": 1},
            {"forward": certificate.forward.derivation},
            {"backward": Certificate(induction)},
            {"moves": ["normalize"]},
            {"backward_moves": (1,)},
        ]
        for fields in bad:
            with pytest.raises(ProveGrammarError):
                replace(certificate, **fields)

    def test_proof_outcome_fields(self) -> None:
        with pytest.raises(ProveGrammarError):
            ProofOutcome("proved", "")  # type: ignore[arg-type]
        with pytest.raises(ProveGrammarError):
            ProofOutcome(ProofStatus.PROVED, 3)  # type: ignore[arg-type]
        with pytest.raises(ProveGrammarError):
            ProofOutcome(ProofStatus.PROVED, "", check="x")  # type: ignore[arg-type]
        with pytest.raises(ProveGrammarError):
            ProofOutcome(ProofStatus.PROVED, "", counterexample=[1])  # type: ignore[arg-type]
        with pytest.raises(ProveGrammarError):
            ProofOutcome(ProofStatus.PROVED, "", assumptions=[1])  # type: ignore[arg-type]
        assert not ProofOutcome(ProofStatus.NOT_FOUND, "").proved

    def test_public_constants(self) -> None:
        assert [s.value for s in ProofStatus] == ["proved", "refuted", "not_found", "check_failed"]
