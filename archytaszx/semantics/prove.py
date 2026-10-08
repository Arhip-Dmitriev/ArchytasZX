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

When search finds no path and a multiplicity is free, :func:`prove` tries an
:class:`InductionProof`: the base case at 0 is proved the same way; for the step, one side's
peeled successor at ``k + 1`` is searched until a union of its regions is that side at ``k``,
the hypothesis replaces it by the other side at ``k``, and the result is proved against the
other side's peeled successor. :func:`check_proof` rebuilds and re-checks every piece.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from archytaszx.algebra.dimension import DimensionError
from archytaszx.algebra.phase import PhaseError
from archytaszx.algebra.scalar import Scalar, ScalarError
from archytaszx.diagram.bangbox import (
    BangBoxError,
    Mult,
    free_mult_symbols,
    instantiate_symbol,
    peel_one,
)
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, GraphError, NodeId
from archytaszx.diagram.validate import ValidateError, validate
from archytaszx.rewrite.cache import RewriteCache
from archytaszx.rewrite.normal_form import comparison_view, views_isomorphic
from archytaszx.rewrite.rule import ConstraintOutcome, DimensionConstraint, RewriteError
from archytaszx.rewrite.tactics import (
    DEFAULT_SEARCH_LIMITS,
    ProofPath,
    SearchLimits,
    SearchResult,
    Tactic,
    search,
    search_for,
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
from archytaszx.semantics.decide import (
    OracleRefutation,
    interface_reason,
    oracle_summary,
    refute_by_oracle,
)
from archytaszx.semantics.induction import (
    InductionError,
    _bare,
    _extract,
    find_hypothesis_region,
    rename_multiplicity,
    replace_region,
)

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


class InductionSide(enum.Enum):
    """Which side's successor the induction hypothesis rewrote."""

    START = "start"
    GOAL = "goal"


@dataclass(frozen=True, slots=True, eq=False)
class InductionProof:
    """``start == goal`` for every value of ``index`` from 0: ``base`` at 0; ``expose`` from
    ``side``'s peeled successor to a diagram whose ``region`` is that side at ``step_symbol``;
    ``step`` from that region replaced by the other side to the other peeled successor."""

    start: Diagram
    goal: Diagram
    index: str
    step_symbol: str
    base: ProofCertificate | InductionProof
    side: InductionSide
    expose: Certificate
    region: frozenset[NodeId]
    step: ProofCertificate | InductionProof

    @property
    def assumptions(self) -> tuple[DimensionConstraint, ...]:
        """Every ``DEFERRED`` dimension constraint of every piece, deduplicated in order."""
        kept = list(self.base.assumptions)
        for constraint in (*_deferred((self.expose,)), *self.step.assumptions):
            if constraint not in kept:
                kept.append(constraint)
        return tuple(kept)

    @property
    def length(self) -> int:
        """The total number of rewrite steps in every piece."""
        return self.base.length + len(self.expose.steps) + self.step.length


Proof = ProofCertificate | InductionProof


def _with_scalar_index(diagram: Diagram, index: str, value: Mult) -> Diagram:
    """``diagram`` with ``index`` replaced by ``value`` in its scalar too."""
    if index not in diagram.scalar.free_symbols:
        return diagram
    working = diagram.copy()
    working.set_scalar(diagram.scalar.substitute({index: Scalar(value.to_sympy())}))
    return working


def induction_base(diagram: Diagram, index: str) -> Diagram:
    """``diagram`` with the multiplicity ``index`` at 0, boxes expanded and scalar substituted."""
    based = instantiate_symbol(diagram, index, 0)
    if index in based.scalar.free_symbols:
        based.set_scalar(based.scalar.substitute({index: 0}))
    return based


def induction_hypothesis(diagram: Diagram, index: str, step_symbol: str) -> Diagram:
    """``diagram`` with ``index`` renamed to ``step_symbol``, scalar included."""
    renamed = rename_multiplicity(diagram, index, Mult(step_symbol))
    return _with_scalar_index(renamed, index, Mult(step_symbol))


def induction_successor(diagram: Diagram, index: str, step_symbol: str) -> Diagram | None:
    """``diagram`` at ``step_symbol + 1`` with one copy peeled off its one box carrying the
    index, or None when no single box carries it."""
    value = Mult(step_symbol) + 1
    renamed = _with_scalar_index(rename_multiplicity(diagram, index, value), index, value)
    owners = [
        box_id
        for box_id, box in sorted(renamed.bang_boxes.items())
        if step_symbol in box.multiplicity.free_symbols
    ]
    if len(owners) != 1:
        return None
    return peel_one(renamed, owners[0]).diagram


def _step_symbol(start: Diagram, goal: Diagram, index: str) -> str:
    """A multiplicity name free in neither diagram."""
    taken = free_mult_symbols(start) | free_mult_symbols(goal)
    taken |= start.scalar.free_symbols | goal.scalar.free_symbols
    name, suffix = "k", 0
    while name in taken or name == index:
        suffix += 1
        name = f"k{suffix}"
    return name


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
        meet = views_isomorphic(
            comparison_view(replays[0].diagram), comparison_view(replays[1].diagram)
        )
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
    sampled = oracle_summary(checked, oracle.refusals)
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


def _crosses(diagram: Diagram, region: frozenset[NodeId]) -> bool:
    """Whether a wire joins ``region`` to a node outside it."""
    return any((w.a.node_id in region) != (w.b.node_id in region) for w in diagram.wires)


def _check_piece(
    proof: Proof,
    rediscover: bool,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> str | None:
    """Why ``proof`` fails its check, or None when it passes."""
    if isinstance(proof, ProofCertificate):
        check = _check(proof, rediscover, None, samples, max_samples, tolerance, max_elements)
        return None if check.verified else check.reason
    return _check_induction(proof, rediscover, samples, max_samples, tolerance, max_elements)


def _check_induction(
    proof: InductionProof,
    rediscover: bool,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> str | None:
    """Why the induction ``proof`` fails, or None; every diagram is rebuilt from its two sides."""
    args = (rediscover, samples, max_samples, tolerance, max_elements)
    for which, diagram in (("start", proof.start), ("goal", proof.goal)):
        invalid = _invalid_reason(diagram)
        if invalid is not None:
            return f"{which} {invalid}"
    index, k = proof.index, proof.step_symbol
    live = free_mult_symbols(proof.start) | free_mult_symbols(proof.goal)
    if index not in live or k in live:
        return f"{index!r} is not a free multiplicity or {k!r} is not fresh"
    try:
        base_start = induction_base(proof.start, index)
        base_goal = induction_base(proof.goal, index)
        if not (
            compare_structure(proof.base.start, base_start).identical
            and compare_structure(proof.base.goal, base_goal).identical
        ):
            return "the base case is not the two sides at multiplicity 0"
        failure = _check_piece(proof.base, *args)
        if failure is not None:
            return f"the base case fails: {failure}"
        own, other = (
            (proof.start, proof.goal)
            if proof.side is InductionSide.START
            else (proof.goal, proof.start)
        )
        peeled = induction_successor(own, index, k)
        other_peeled = induction_successor(other, index, k)
        if peeled is None or other_peeled is None:
            return "a successor has no single box to peel"
        if not compare_structure(proof.expose.initial, peeled).identical:
            return "the exposing derivation does not begin at the peeled successor"
        replayed = replay(proof.expose, rediscover=rediscover)
        if not replayed.reproduced:
            return f"the exposing derivation did not replay: {replayed.reason}"
        exposed = replayed.diagram
        hypothesis = induction_hypothesis(own, index, k)
        if not proof.region <= frozenset(exposed.nodes) or _crosses(exposed, proof.region):
            return "the hypothesis region is not a closed part of the exposed diagram"
        region_diagram = _bare(_extract(exposed, proof.region, with_boxes=True))
        if not isomorphic(region_diagram, _bare(hypothesis), symmetric_legs=True):
            return "the hypothesis region is not the hypothesis diagram"
        replaced = replace_region(
            exposed, proof.region, hypothesis, induction_hypothesis(other, index, k)
        )
        if not (
            compare_structure(proof.step.start, replaced).identical
            and compare_structure(proof.step.goal, other_peeled).identical
        ):
            return "the step does not run from the rewritten successor to the other side's"
        failure = _check_piece(proof.step, *args)
        if failure is not None:
            return f"the step fails: {failure}"
    except (*_CHECK_ERRORS, InductionError) as exc:
        return f"rebuilding the induction raised {type(exc).__name__}: {exc}"
    return None


def check_proof(
    certificate: Proof,
    *,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    rediscover: bool = True,
) -> ProofCheck:
    """Check ``certificate`` by the module docstring's steps (a) to (d); an
    :class:`InductionProof` piece by piece, then the oracle on its two sides."""
    if not isinstance(certificate, (ProofCertificate, InductionProof)):
        raise ProveGrammarError(
            f"certificate must be a ProofCertificate or InductionProof, got "
            f"{type(certificate).__name__}"
        )
    _check_oracle_args(samples, max_samples, tolerance, max_elements)
    if not isinstance(rediscover, bool):
        raise ProveGrammarError(f"rediscover must be a bool, got {rediscover!r}")
    if isinstance(certificate, InductionProof):
        return _check_whole_induction(
            certificate, rediscover, None, samples, max_samples, tolerance, max_elements
        )
    return _check(certificate, rediscover, None, samples, max_samples, tolerance, max_elements)


def _check_whole_induction(
    proof: InductionProof,
    rediscover: bool,
    oracle: OracleRefutation | None,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> ProofCheck:
    """:func:`check_proof` of an induction proof, reusing ``oracle`` when given."""
    failure = _check_induction(proof, rediscover, samples, max_samples, tolerance, max_elements)
    if failure is not None:
        return ProofCheck(False, failure, None, None, False, 0, None, None)
    if oracle is None:
        oracle = refute_by_oracle(
            proof.start,
            proof.goal,
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
            None,
            None,
            True,
            checked,
            oracle.counterexample,
            oracle.comparison,
        )
    return ProofCheck(
        True,
        f"base and step check by replay and the hypothesis; "
        f"{oracle_summary(checked, oracle.refusals)}",
        None,
        None,
        True,
        checked,
        None,
        None,
    )


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
    induction: InductionProof | None = None

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
        if self.induction is not None and not isinstance(self.induction, InductionProof):
            raise ProveGrammarError("induction must be an InductionProof or None")

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


@dataclass(frozen=True, slots=True)
class _Searcher:
    """The search settings every piece of one :func:`prove` call shares."""

    moves: Sequence[Tactic] | None
    limits: SearchLimits
    bidirectional: bool
    cache: RewriteCache | None


def _prove_piece(start: Diagram, goal: Diagram, run: _Searcher, depth: int) -> Proof | None:
    """A search proof of ``start == goal``, else an induction proof ``depth`` levels deep."""
    found = search(
        start,
        goal,
        moves=run.moves,
        limits=run.limits,
        bidirectional=run.bidirectional,
        cache=run.cache,
    )
    if found.path is not None:
        return certify_proof(found.path)
    if depth <= 0:
        return None
    return _induct(start, goal, run, depth)


def _induct(start: Diagram, goal: Diagram, run: _Searcher, depth: int) -> InductionProof | None:
    """An :class:`InductionProof` of ``start == goal`` on some free multiplicity, or None."""
    for index in sorted(free_mult_symbols(start) | free_mult_symbols(goal)):
        try:
            k = _step_symbol(start, goal, index)
            base_start, base_goal = induction_base(start, index), induction_base(goal, index)
            if _invalid_reason(base_start) or _invalid_reason(base_goal):
                continue
            base = _prove_piece(base_start, base_goal, run, depth - 1)
            if base is None:
                continue
            for side in (InductionSide.START, InductionSide.GOAL):
                own, other = (start, goal) if side is InductionSide.START else (goal, start)
                peeled = induction_successor(own, index, k)
                other_peeled = induction_successor(other, index, k)
                if peeled is None or other_peeled is None:
                    continue
                hypothesis = induction_hypothesis(own, index, k)
                _, hit = search_for(
                    peeled,
                    lambda diagram, h=hypothesis: find_hypothesis_region(diagram, h),
                    moves=run.moves,
                    limits=run.limits,
                    cache=run.cache,
                )
                if hit is None:
                    continue
                region = cast(frozenset[NodeId], hit.value)
                replaced = replace_region(
                    hit.path.goal, region, hypothesis, induction_hypothesis(other, index, k)
                )
                step = _prove_piece(replaced, other_peeled, run, depth - 1)
                if step is None:
                    continue
                return InductionProof(
                    start=start.copy(),
                    goal=goal.copy(),
                    index=index,
                    step_symbol=k,
                    base=base,
                    side=side,
                    expose=certify(peeled, hit.path.forward_results, label="expose"),
                    region=region,
                    step=step,
                )
        except (*_CHECK_ERRORS, InductionError):
            continue
    return None


DEFAULT_INDUCTION_DEPTH = 2


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
    induction_depth: int = DEFAULT_INDUCTION_DEPTH,
) -> ProofOutcome:
    """Prove ``start`` equal to ``goal`` by the module docstring's order: interface, oracle,
    search, certification and check, then induction over a free multiplicity when search
    finds no path, nesting at most ``induction_depth`` inductions (0 turns it off)."""
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
    if (
        isinstance(induction_depth, bool)
        or not isinstance(induction_depth, int)
        or induction_depth < 0
    ):
        raise ProveGrammarError(f"induction_depth must be an int >= 0, got {induction_depth!r}")
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
        run = _Searcher(moves, limits, bidirectional, cache)
        inducted = _induct(start, goal, run, induction_depth) if induction_depth else None
        if inducted is None:
            return ProofOutcome(ProofStatus.NOT_FOUND, _search_summary(found), search=found)
        check = _check_whole_induction(
            inducted, True, oracle, samples, max_samples, tolerance, max_elements
        )
        if not check.verified:
            return ProofOutcome(
                ProofStatus.CHECK_FAILED,
                f"induction proof check failed: {check.reason}",
                check=check,
                search=found,
                counterexample=check.counterexample,
                induction=inducted,
            )
        assumptions = inducted.assumptions
        return ProofOutcome(
            ProofStatus.PROVED,
            f"proved by induction on {inducted.index} from 0 in {inducted.length} step(s) after "
            f"{_search_summary(found)}; {check.reason}",
            check=check,
            search=found,
            assumptions=assumptions,
            induction=inducted,
        )
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
