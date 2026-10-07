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

"""Goal-directed proving: search for a derivation between two diagrams, certify it, check it.

:func:`prove` settles a goal in order: an interface that can never agree
(:func:`~archytaszx.semantics.decide.interface_reason`) is ``REFUTED``; when ``refute`` is set,
an oracle counterexample (:func:`~archytaszx.semantics.decide.refute_by_oracle`) is ``REFUTED``;
a :func:`~archytaszx.rewrite.tactics.search` that finds no path is ``NOT_FOUND``; a found path
is certified by :func:`certify_proof` and checked by :func:`check_proof`, giving ``PROVED`` or
``CHECK_FAILED``.

A :class:`ProofCertificate` holds two ``STEP_SEQUENCE`` certificates: ``forward`` from
``start`` and ``backward`` from ``goal``. :func:`check_proof` checks, stopping at the first
failure: (a) ``start`` and ``goal`` pass validation and each half's initial diagram is
identical, id for id, to its end; (b) each half replays; (c) the two replayed diagrams have
isomorphic :func:`~archytaszx.rewrite.normal_form.comparison_view`; (d) the oracle finds no
counterexample between ``start`` and ``goal``. Zero evaluated samples passes (d), noted in its
reason.

:func:`prove` raises :class:`ProveGrammarError` when ``start`` or ``goal`` fails validation.
A ``PROVED`` outcome whose steps recorded ``DEFERRED`` dimension constraints is conditional on
them; they are listed in :attr:`ProofOutcome.assumptions`.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from archytaszx.algebra.dimension import DimensionError
from archytaszx.algebra.phase import PhaseError
from archytaszx.algebra.scalar import ScalarError
from archytaszx.diagram.bangbox import BangBoxError
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, GraphError
from archytaszx.diagram.validate import ValidateError, validate
from archytaszx.rewrite.cache import RewriteCache
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import ConstraintOutcome, DimensionConstraint, RewriteError
from archytaszx.rewrite.tactics import (
    DEFAULT_SEARCH_LIMITS,
    ProofPath,
    SearchLimits,
    SearchResult,
    Tactic,
    search,
)
from archytaszx.semantics.certificate import (
    Certificate,
    CertificateError,
    DerivationKind,
    ReplayResult,
    certify,
    compare_structure,
    replay,
)
from archytaszx.semantics.check import DEFAULT_TOLERANCE, CheckAssignmentValue, ComparisonResult
from archytaszx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS
from archytaszx.semantics.decide import OracleRefutation, interface_reason, refute_by_oracle

_CHECK_ERRORS: tuple[type[Exception], ...] = (
    BangBoxError,
    CertificateError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    RewriteError,
    ScalarError,
    ValidateError,
)


class ProveError(Exception):
    """Base class for all errors raised by this module."""


class ProveGrammarError(ProveError):
    """A request is malformed: a wrong argument type or an out-of-range bound."""


class ProofStatus(enum.Enum):
    """What :func:`prove` established."""

    PROVED = "proved"
    REFUTED = "refuted"
    NOT_FOUND = "not_found"
    CHECK_FAILED = "check_failed"


def _deferred(certificates: Sequence[Certificate]) -> tuple[DimensionConstraint, ...]:
    """Every ``DEFERRED`` constraint of ``certificates``' steps, first occurrence order kept."""
    kept: list[DimensionConstraint] = []
    for certificate in certificates:
        for step in certificate.steps:
            for constraint in step.dimension_constraints:
                if constraint.outcome is ConstraintOutcome.DEFERRED and constraint not in kept:
                    kept.append(constraint)
    return tuple(kept)


@dataclass(frozen=True, slots=True, eq=False)
class ProofCertificate:
    """A found derivation as two step-sequence certificates: from ``start`` and from ``goal``."""

    start: Diagram
    goal: Diagram
    forward: Certificate
    backward: Certificate
    moves: tuple[str, ...]
    backward_moves: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        for name in ("start", "goal"):
            if not isinstance(getattr(self, name), Diagram):
                raise ProveGrammarError(f"ProofCertificate.{name} must be a Diagram")
        for name in ("forward", "backward"):
            value = getattr(self, name)
            if not isinstance(value, Certificate):
                raise ProveGrammarError(f"ProofCertificate.{name} must be a Certificate")
            if value.derivation.kind is not DerivationKind.STEP_SEQUENCE:
                raise ProveGrammarError(f"ProofCertificate.{name} must be a step sequence")
        for name in ("moves", "backward_moves"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(isinstance(m, str) for m in value):
                raise ProveGrammarError(f"ProofCertificate.{name} must be a tuple of str")

    @property
    def assumptions(self) -> tuple[DimensionConstraint, ...]:
        """Every ``DEFERRED`` dimension constraint of both halves, deduplicated in order."""
        return _deferred((self.forward, self.backward))

    @property
    def length(self) -> int:
        """The total number of steps in both halves."""
        return len(self.forward.steps) + len(self.backward.steps)


def certify_proof(path: ProofPath) -> ProofCertificate:
    """Certify ``path``'s forward results from its start and backward results from its goal."""
    if not isinstance(path, ProofPath):
        raise ProveGrammarError(f"path must be a ProofPath, got {type(path).__name__}")
    return ProofCertificate(
        start=path.start,
        goal=path.goal,
        forward=certify(path.start, path.forward_results, label="forward"),
        backward=certify(path.goal, path.backward_results, label="backward"),
        moves=path.moves,
        backward_moves=path.backward_moves,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ProofCheck:
    """The outcome of :func:`check_proof`: a verdict, the first failure or a summary, and the
    evidence gathered up to it."""

    verified: bool
    reason: str
    forward_replay: ReplayResult | None
    backward_replay: ReplayResult | None
    meet_isomorphic: bool
    samples_checked: int
    counterexample: Mapping[str, CheckAssignmentValue] | None
    comparison: ComparisonResult | None


def _check_oracle_args(
    samples: object, max_samples: object, tolerance: object, max_elements: object
) -> None:
    """Raise ProveGrammarError on a malformed oracle argument."""
    if samples is not None and (
        isinstance(samples, (str, bytes))
        or not isinstance(samples, Sequence)
        or not all(isinstance(sample, Mapping) for sample in samples)
    ):
        raise ProveGrammarError("samples must be None or a Sequence of Mapping")
    for name, value, low in (("max_samples", max_samples, 0), ("max_elements", max_elements, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ProveGrammarError(f"{name} must be an int >= {low}, got {value!r}")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not tolerance >= 0:
        raise ProveGrammarError(f"tolerance must be a non-negative float, got {tolerance!r}")


def _invalid_reason(diagram: Diagram) -> str | None:
    """Why ``diagram`` fails validation, or None when it has no hard-failure issue."""
    errors = validate(diagram).errors
    if not errors:
        return None
    return f"fails validation: {'; '.join(issue.message for issue in errors)}"


def _check(
    certificate: ProofCertificate,
    rediscover: bool,
    oracle: OracleRefutation | None,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> ProofCheck:
    """:func:`check_proof` on validated arguments, reusing ``oracle`` for (d) when given."""
    replays: list[ReplayResult] = []

    def failed(reason: str, meet: bool = False) -> ProofCheck:
        forward = replays[0] if replays else None
        backward = replays[1] if len(replays) > 1 else None
        return ProofCheck(False, reason, forward, backward, meet, 0, None, None)

    for which, diagram in (("start", certificate.start), ("goal", certificate.goal)):
        invalid = _invalid_reason(diagram)
        if invalid is not None:
            return failed(f"{which} {invalid}")
    for label, half, end in (
        ("forward", certificate.forward, certificate.start),
        ("backward", certificate.backward, certificate.goal),
    ):
        structural = compare_structure(half.initial, end)
        if not structural.identical:
            which = "start" if label == "forward" else "goal"
            return failed(f"the {label} half does not begin at {which}: {structural.reason}")
    for label, half in (("forward", certificate.forward), ("backward", certificate.backward)):
        try:
            replayed = replay(half, rediscover=rediscover)
        except _CHECK_ERRORS as exc:
            return failed(f"the {label} half's replay raised {type(exc).__name__}: {exc}")
        replays.append(replayed)
        if not replayed.reproduced:
            return failed(f"the {label} half did not replay: {replayed.reason}")
    try:
        meet = isomorphic(comparison_view(replays[0].diagram), comparison_view(replays[1].diagram))
    except _CHECK_ERRORS as exc:
        return failed(f"comparing the meet raised {type(exc).__name__}: {exc}")
    if not meet:
        return failed("the two halves end on diagrams with different comparison views")
    if oracle is None:
        oracle = refute_by_oracle(
            certificate.start,
            certificate.goal,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    checked = len(oracle.evaluated)
    if oracle.counterexample is not None:
        detail = oracle.comparison.reason if oracle.comparison is not None else ""
        return ProofCheck(
            False,
            f"oracle mismatch at {dict(oracle.counterexample)!r}: {detail}",
            replays[0],
            replays[1],
            True,
            checked,
            oracle.counterexample,
            oracle.comparison,
        )
    sampled = (
        "no oracle sample evaluated"
        if checked == 0
        else f"no oracle mismatch in {checked} sample(s)"
    )
    return ProofCheck(
        True,
        f"both halves replay and meet; {sampled}",
        replays[0],
        replays[1],
        True,
        checked,
        None,
        None,
    )


def check_proof(
    certificate: ProofCertificate,
    *,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    rediscover: bool = True,
) -> ProofCheck:
    """Check ``certificate`` by the module docstring's steps (a) to (d)."""
    if not isinstance(certificate, ProofCertificate):
        raise ProveGrammarError(
            f"certificate must be a ProofCertificate, got {type(certificate).__name__}"
        )
    _check_oracle_args(samples, max_samples, tolerance, max_elements)
    if not isinstance(rediscover, bool):
        raise ProveGrammarError(f"rediscover must be a bool, got {rediscover!r}")
    return _check(certificate, rediscover, None, samples, max_samples, tolerance, max_elements)


@dataclass(frozen=True, slots=True, eq=False)
class ProofOutcome:
    """A :func:`prove` verdict, its reason, and the evidence behind it."""

    status: ProofStatus
    reason: str
    certificate: ProofCertificate | None = None
    check: ProofCheck | None = None
    search: SearchResult | None = None
    counterexample: Mapping[str, CheckAssignmentValue] | None = None
    assumptions: tuple[DimensionConstraint, ...] = ()

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        if not isinstance(self.status, ProofStatus):
            raise ProveGrammarError(f"status must be a ProofStatus, got {self.status!r}")
        if not isinstance(self.reason, str):
            raise ProveGrammarError(f"reason must be a str, got {type(self.reason).__name__}")
        for name, kind in (
            ("certificate", ProofCertificate),
            ("check", ProofCheck),
            ("search", SearchResult),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, kind):
                raise ProveGrammarError(f"{name} must be a {kind.__name__} or None")
        if self.counterexample is not None and not isinstance(self.counterexample, Mapping):
            raise ProveGrammarError("counterexample must be a Mapping or None")
        if not isinstance(self.assumptions, tuple) or not all(
            isinstance(item, DimensionConstraint) for item in self.assumptions
        ):
            raise ProveGrammarError("assumptions must be a tuple of DimensionConstraint")

    @property
    def proved(self) -> bool:
        """True when the status is PROVED."""
        return self.status is ProofStatus.PROVED


def _search_summary(result: SearchResult) -> str:
    """``result``'s status and counters as one line."""
    return (
        f"search {result.status.value} after {result.depth} layer(s): {result.states} state(s), "
        f"{result.expanded} expanded, {result.applications} application(s), "
        f"{result.pruned} pruned, {len(result.failures)} failure(s)"
    )


def prove(
    start: Diagram,
    goal: Diagram,
    *,
    moves: Sequence[Tactic] | None = None,
    limits: SearchLimits = DEFAULT_SEARCH_LIMITS,
    bidirectional: bool = True,
    cache: RewriteCache | None = None,
    refute: bool = True,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ProofOutcome:
    """Prove ``start`` equal to ``goal`` by the module docstring's order: interface, oracle,
    search, certification and check."""
    for name, value in (("start", start), ("goal", goal)):
        if not isinstance(value, Diagram):
            raise ProveGrammarError(f"{name} must be a Diagram, got {type(value).__name__}")
    if moves is not None and (
        isinstance(moves, (str, bytes))
        or not isinstance(moves, Sequence)
        or not all(isinstance(move, Tactic) for move in moves)
    ):
        raise ProveGrammarError("moves must be None or a Sequence of Tactic")
    if not isinstance(limits, SearchLimits):
        raise ProveGrammarError(f"limits must be a SearchLimits, got {type(limits).__name__}")
    for name, flag in (("bidirectional", bidirectional), ("refute", refute)):
        if not isinstance(flag, bool):
            raise ProveGrammarError(f"{name} must be a bool, got {flag!r}")
    if cache is not None and not isinstance(cache, RewriteCache):
        raise ProveGrammarError(f"cache must be a RewriteCache, got {type(cache).__name__}")
    _check_oracle_args(samples, max_samples, tolerance, max_elements)
    for name, value in (("start", start), ("goal", goal)):
        invalid = _invalid_reason(value)
        if invalid is not None:
            raise ProveGrammarError(f"{name} {invalid}")

    interface = interface_reason(start, goal)
    if interface is not None:
        return ProofOutcome(ProofStatus.REFUTED, f"interfaces never agree: {interface}")
    oracle: OracleRefutation | None = None
    if refute:
        oracle = refute_by_oracle(
            start,
            goal,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
        )
        if oracle.counterexample is not None:
            detail = oracle.comparison.reason if oracle.comparison is not None else ""
            return ProofOutcome(
                ProofStatus.REFUTED,
                f"oracle mismatch at {dict(oracle.counterexample)!r}: {detail}",
                counterexample=oracle.counterexample,
            )
    found = search(
        start, goal, moves=moves, limits=limits, bidirectional=bidirectional, cache=cache
    )
    if found.path is None:
        return ProofOutcome(ProofStatus.NOT_FOUND, _search_summary(found), search=found)
    certificate = certify_proof(found.path)
    check = _check(certificate, True, oracle, samples, max_samples, tolerance, max_elements)
    if not check.verified:
        return ProofOutcome(
            ProofStatus.CHECK_FAILED,
            f"proof check failed: {check.reason}",
            certificate=certificate,
            check=check,
            search=found,
            counterexample=check.counterexample,
        )
    assumptions = certificate.assumptions
    conditional = (
        f", conditional on the {len(assumptions)} deferred dimension constraint(s) in assumptions"
        if assumptions
        else ""
    )
    return ProofOutcome(
        ProofStatus.PROVED,
        f"proved in {certificate.length} step(s), {len(certificate.forward.steps)} forward and "
        f"{len(certificate.backward.steps)} backward{conditional}; {check.reason}",
        certificate=certificate,
        check=check,
        search=found,
        assumptions=assumptions,
    )
