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
an oracle counterexample (:func:`~archytaszx.semantics.decide.refute_by_oracle`) is ``REFUTED``
with its Lean refutation (:func:`~archytaszx.semantics.decide.lean_refutation`), else
``NOT_FOUND``;
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

Mixed operands (a discard on either side). :func:`prove` first searches with the discard rules
added (:func:`~archytaszx.rewrite.tactics.mixed_moves`) and returns a :class:`MixedCertificate`
of kind ``NATIVE`` holding the checked single-sheet proof. Otherwise, and in
:func:`prove_exhaustive`, it proves the doubled diagrams of both sides with the pure
machinery and returns a :class:`MixedCertificate` of kind ``DOUBLED`` in
:attr:`ProofOutcome.mixed`: both mixed diagrams, their doubled diagrams with the origin of
every node, and the inner pure proof, which is also
:attr:`ProofOutcome.certificate` or :attr:`ProofOutcome.induction`. :func:`check_mixed`
re-doubles both sides, requires them identical id for id to the stored doubled diagrams, and
checks the inner proof.
"""

from __future__ import annotations

import enum
import functools
import itertools
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import DimensionError
from archytaszx.algebra.phase import PhaseError
from archytaszx.algebra.scalar import Scalar, ScalarError
from archytaszx.diagram.bangbox import (
    BangBoxError,
    Mult,
    free_mult_symbols,
    instantiate_symbol,
    offset_index,
    peel_one,
    peels_first,
    substitute_index_phases,
)
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, GraphError, NodeId
from archytaszx.diagram.mixed import Origin, double_with_origin, is_pure
from archytaszx.diagram.validate import ValidateError, validate
from archytaszx.rewrite.cache import RewriteCache
from archytaszx.rewrite.exhaustive import (
    ExhaustedError,
    LayerReport,
    algebraic_search,
    exhaustive_search,
)
from archytaszx.rewrite.normal_form import algebraic_view, comparison_view, views_isomorphic
from archytaszx.rewrite.rule import DimensionConstraint, RewriteError, assumed
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
    lean_refutation,
    oracle_summary,
    refute_by_oracle,
)
from archytaszx.semantics.induction import (
    InductionError,
    _all_free_symbols,
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
_INDUCTION_ERRORS: tuple[type[Exception], ...] = (*_CHECK_ERRORS, InductionError)


if TYPE_CHECKING:
    from archytaszx.semantics.restrict import Restriction


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


def _assumed(certificates: Sequence[Certificate]) -> tuple[DimensionConstraint, ...]:
    """Every assumed dimension equation of ``certificates``' steps, by :func:`assumed`."""
    return assumed(
        c
        for certificate in certificates
        for step in certificate.steps
        for c in step.dimension_constraints
    )


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
        return _assumed((self.forward, self.backward))

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
        """Every assumed dimension equation of every piece, by :func:`assumed`."""
        return assumed((*self.base.assumptions, *_assumed((self.expose,)), *self.step.assumptions))

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
    """``diagram`` with the multiplicity ``index`` at 0 in boxes, scalar and phases."""
    based = instantiate_symbol(diagram, index, 0)
    if index in based.scalar.free_symbols:
        based.set_scalar(based.scalar.substitute({index: 0}))
    return substitute_index_phases(based, index, 0)


def induction_hypothesis(diagram: Diagram, index: str, step_symbol: str) -> Diagram:
    """``diagram`` with ``index`` renamed to ``step_symbol``, scalar included."""
    renamed = rename_multiplicity(diagram, index, Mult(step_symbol))
    return _with_scalar_index(renamed, index, Mult(step_symbol))


def induction_successor(diagram: Diagram, index: str, step_symbol: str) -> Diagram | None:
    """``diagram`` at ``step_symbol + 1`` with one copy peeled off its one box carrying the
    index, or None when no single box carries it; unpeeled when that box's multiplicity has
    no positive constant term (``b^(k + 1)``) and it is no tree box."""
    value = Mult(step_symbol) + 1
    renamed = _with_scalar_index(rename_multiplicity(diagram, index, value), index, value)
    owners = [
        box_id
        for box_id, box in sorted(renamed.bang_boxes.items())
        if step_symbol in box.multiplicity.free_symbols
    ]
    if len(owners) != 1:
        return None
    box = renamed.bang_boxes[owners[0]]
    constant = sp.expand(box.multiplicity.to_sympy()).as_coeff_Add()[0]
    if not box.is_tree and not (constant.is_Integer and constant >= 1):
        return renamed
    return peel_one(renamed, owners[0]).diagram


def peels_shifted(diagram: Diagram, index: str) -> bool:
    """Whether the one box carrying ``index`` peels its first copy and reads a copy index."""
    owners = [b for b in diagram.bang_boxes.values() if index in b.multiplicity.free_symbols]
    return len(owners) == 1 and owners[0].index is not None and peels_first(owners[0])


def induction_offset(diagram: Diagram, index: str, by: sp.Expr) -> Diagram:
    """``diagram`` with the one box carrying ``index`` reading its copy index as
    ``copy index + by``; ``diagram`` itself when that box reads none."""
    owners = [
        box_id
        for box_id, box in sorted(diagram.bang_boxes.items())
        if index in box.multiplicity.free_symbols
    ]
    if len(owners) != 1:
        raise InductionError(f"no single box carries {index!r}")
    if diagram.bang_boxes[owners[0]].index is None:
        return diagram
    return offset_index(diagram, owners[0], by)


def offset_symbol(*diagrams: Diagram) -> sp.Symbol:
    """A natural-number symbol named in none of ``diagrams``."""
    taken: set[str] = set()
    for d in diagrams:
        taken |= _all_free_symbols(d)
        for box in d.bang_boxes.values():
            if box.index is not None:
                taken.add(box.index)
    name, suffix = "s", 0
    while name in taken:
        suffix += 1
        name = f"s{suffix}"
    return sp.Symbol(name, integer=True, nonnegative=True)


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


@dataclass(frozen=True, slots=True, eq=False)
class _DeferredOracle(OracleRefutation):
    """An oracle run with its deferred large samples."""

    deferred: tuple[Mapping[str, CheckAssignmentValue], ...] = ()


def _run_deferred(
    oracle: OracleRefutation,
    left: Diagram,
    right: Diagram,
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> OracleRefutation:
    """``oracle``, or its deferred samples run when it evaluated none."""
    if isinstance(oracle, _DeferredOracle) and not oracle.evaluated and oracle.deferred:
        return refute_by_oracle(
            left,
            right,
            samples=oracle.deferred,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    return oracle


CERTIFY_SAMPLE_ELEMENTS = 2**20
"""The largest planned sample size certification's oracle evaluates before every smaller one."""


def _certify_oracle(
    left: Diagram,
    right: Diagram,
    *,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    defer: bool = False,
) -> OracleRefutation:
    """:func:`refute_by_oracle` with samples planned above :data:`CERTIFY_SAMPLE_ELEMENTS`
    deferred: they run only when no smaller sample evaluates, or, with ``defer``, not at all,
    kept on the result for :func:`_run_deferred`."""
    from archytaszx.semantics.decide import _family, _planned_size, sample_grid

    family_l, family_r = _family(left), _family(right)
    candidates = tuple(samples) if samples is not None else sample_grid(family_l, family_r)
    sizes = [_planned_size(family_l, family_r, c) for c in candidates]
    large = [c for c, n in zip(candidates, sizes, strict=True) if n > CERTIFY_SAMPLE_ELEMENTS]
    if not large:
        return refute_by_oracle(
            left,
            right,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    small = [c for c, n in zip(candidates, sizes, strict=True) if n <= CERTIFY_SAMPLE_ELEMENTS]
    first = refute_by_oracle(
        left,
        right,
        samples=small,
        max_samples=max_samples,
        tolerance=tolerance,
        max_elements=max_elements,
    )
    if first.counterexample is not None or first.evaluated or defer:
        deferred = tuple(
            f"{dict(c)!r}: deferred (planned {n:g} elements, over {CERTIFY_SAMPLE_ELEMENTS})"
            for c, n in zip(candidates, sizes, strict=True)
            if n > CERTIFY_SAMPLE_ELEMENTS
        )
        return _DeferredOracle(
            first.evaluated,
            first.counterexample,
            first.comparison,
            (*first.refusals, *deferred),
            tuple(large),
        )
    return refute_by_oracle(
        left,
        right,
        samples=large,
        max_samples=max_samples,
        tolerance=tolerance,
        max_elements=max_elements,
    )


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
        try:
            meet = views_isomorphic(
                algebraic_view(replays[0].diagram), algebraic_view(replays[1].diagram)
            )
        except (*_CHECK_ERRORS, ValueError, ArithmeticError):  # type: ignore[misc]
            meet = False
    if not meet:
        return failed("the two halves end on diagrams with different comparison views")
    if oracle is None:
        oracle = _certify_oracle(
            certificate.start,
            certificate.goal,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    oracle = _run_deferred(
        oracle, certificate.start, certificate.goal, max_samples, tolerance, max_elements
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
    except _INDUCTION_ERRORS as exc:
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
        oracle = _certify_oracle(
            proof.start,
            proof.goal,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    oracle = _run_deferred(oracle, proof.start, proof.goal, max_samples, tolerance, max_elements)
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


class MixedKind(enum.Enum):
    """How a :class:`MixedCertificate` proves its mixed equality."""

    DOUBLED = "doubled"
    NATIVE = "native"


@dataclass(frozen=True, slots=True, eq=False)
class MixedCertificate:
    """A proof that two mixed diagrams are equal.

    ``DOUBLED``: ``inner`` proves ``start_doubled`` equal to ``goal_doubled``, the doubled
    diagrams of ``start`` and ``goal``, whose nodes map back to them through the origins.
    ``NATIVE``: ``inner`` is a proof between ``start`` and ``goal`` themselves.
    """

    kind: MixedKind
    start: Diagram
    goal: Diagram
    inner: ProofCertificate | InductionProof
    start_doubled: Diagram | None = None
    goal_doubled: Diagram | None = None
    start_origin: Mapping[NodeId, Origin] | None = None
    goal_origin: Mapping[NodeId, Origin] | None = None

    def __post_init__(self) -> None:
        """Validate the kind, the diagrams and the inner proof."""
        if not isinstance(self.kind, MixedKind):
            raise ProveGrammarError(f"kind must be a MixedKind, got {self.kind!r}")
        if not isinstance(self.inner, (ProofCertificate, InductionProof)):
            raise ProveGrammarError("inner must be a ProofCertificate or InductionProof")
        doubled = (self.start_doubled, self.goal_doubled)
        if self.kind is MixedKind.DOUBLED and not all(isinstance(d, Diagram) for d in doubled):
            raise ProveGrammarError("a DOUBLED certificate needs both doubled diagrams")

    @property
    def length(self) -> int:
        """The number of steps of the inner proof."""
        return self.inner.length


def check_mixed(
    certificate: MixedCertificate,
    *,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ProofCheck:
    """Check ``certificate``: re-double both sides against the stored doubled diagrams, then
    :func:`check_proof` the inner proof."""
    if not isinstance(certificate, MixedCertificate):
        raise ProveGrammarError(
            f"certificate must be a MixedCertificate, got {type(certificate).__name__}"
        )
    if certificate.kind is MixedKind.DOUBLED:
        for side, stored in (
            (certificate.start, certificate.start_doubled),
            (certificate.goal, certificate.goal_doubled),
        ):
            assert stored is not None
            redone = double_with_origin(side).diagram
            same = compare_structure(redone, stored)
            if not same.identical:
                return ProofCheck(
                    False,
                    f"re-doubling differs from the stored diagram: {same.reason}",
                    None,
                    None,
                    False,
                    0,
                    None,
                    None,
                )
    inner = check_proof(
        certificate.inner,
        samples=samples,
        max_samples=max_samples,
        tolerance=tolerance,
        max_elements=max_elements,
    )
    if not inner.verified:
        return inner
    return replace(inner, reason=f"{certificate.kind.value}: {inner.reason}")


def _native_outcome(
    start: Diagram,
    goal: Diagram,
    moves: Sequence[Tactic] | None,
    limits: SearchLimits,
    bidirectional: bool,
    cache: RewriteCache | None,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> ProofOutcome | None:
    """A ``NATIVE`` proof of two mixed diagrams by search with the discard rules, checked;
    None when the interface differs, search finds no path or the check fails."""
    from archytaszx.rewrite.tactics import mixed_moves

    if interface_reason(start, goal) is not None:
        return None
    found = search(
        start,
        goal,
        moves=mixed_moves() if moves is None else moves,
        limits=limits,
        bidirectional=bidirectional,
        cache=cache,
    )
    if found.path is None:
        return None
    certificate = certify_proof(found.path)
    check = _check(certificate, True, None, samples, max_samples, tolerance, max_elements)
    if not check.verified:
        return None
    mixed = MixedCertificate(MixedKind.NATIVE, start, goal, certificate)
    return ProofOutcome(
        ProofStatus.PROVED,
        f"native: proved in {certificate.length} step(s), {len(certificate.forward.steps)} "
        f"forward and {len(certificate.backward.steps)} backward; {check.reason}",
        certificate=certificate,
        check=check,
        search=found,
        assumptions=certificate.assumptions,
        mixed=mixed,
    )


def _doubled_outcome(
    start: Diagram, goal: Diagram, prover: Callable[[Diagram, Diagram], ProofOutcome]
) -> ProofOutcome:
    """``prover`` on the doubled diagrams of ``start`` and ``goal``, wrapped as ``DOUBLED``."""
    left = double_with_origin(start)
    right = double_with_origin(goal)
    outcome = prover(left.diagram, right.diagram)
    inner = outcome.certificate if outcome.certificate is not None else outcome.induction
    mixed = (
        None
        if inner is None or outcome.status is not ProofStatus.PROVED
        else MixedCertificate(
            MixedKind.DOUBLED,
            start,
            goal,
            inner,
            left.diagram,
            right.diagram,
            left.origin,
            right.origin,
        )
    )
    return replace(outcome, reason=f"doubled: {outcome.reason}", mixed=mixed)


@dataclass(frozen=True, slots=True, eq=False)
class ProofOutcome:
    """A :func:`prove` verdict, its reason, and the evidence behind it; a restricted proof
    carries its ``restriction``, the unrestricted ``family`` outcome, ``family_holds`` when that
    outcome proved the claim, the ``restricted_pair`` it proved otherwise and the counterexample
    in the original symbols; ``refutation`` is the Lean refutation source of an oracle
    ``REFUTED``; ``instances`` holds, for a restriction with finitely many values, each
    instance's label and outcome (one certificate per instance)."""

    status: ProofStatus
    reason: str
    certificate: ProofCertificate | None = None
    check: ProofCheck | None = None
    search: SearchResult | None = None
    counterexample: Mapping[str, CheckAssignmentValue] | None = None
    assumptions: tuple[DimensionConstraint, ...] = ()
    induction: InductionProof | None = None
    mixed: MixedCertificate | None = None
    restriction: Restriction | None = None
    family: ProofOutcome | None = None
    family_holds: bool = False
    restricted_pair: tuple[Diagram, Diagram] | None = None
    counterexample_original: Mapping[str, CheckAssignmentValue] | None = None
    refutation: str | None = None
    instances: tuple[tuple[str, ProofOutcome], ...] = ()

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


def restricted_outcome(
    left: Diagram,
    right: Diagram,
    restrict: Restriction | str | Mapping[str, str],
    run: Callable[..., ProofOutcome],
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    family_first: bool = True,
) -> ProofOutcome:
    """The three steps of a restricted proof: ``run`` on the family, kept when it proves the
    claim or refutes it in range; otherwise ``run`` on the restricted pair, ``samples`` read
    over its symbols. ``family_first=False`` runs the restricted pair only."""
    from archytaszx.semantics.decide import DecideGrammarError
    from archytaszx.semantics.restrict import (
        in_domain,
        parse_restriction,
        restricted_samples,
        subst_pair,
        to_original,
        validate,
    )

    if not isinstance(family_first, bool):
        raise ProveGrammarError(f"family_first must be a bool, got {family_first!r}")
    try:
        restriction = validate(parse_restriction(restrict), left, right)
    except DecideGrammarError as exc:
        raise ProveGrammarError(str(exc)) from exc
    if restriction.empty:
        return run(left, right, samples=samples)
    if not family_first:
        if restriction.finite:
            return _by_instances(left, right, restriction, run, samples)
        return _pair_only(left, right, restriction, run, samples)
    family = run(left, right, samples=samples)
    if family.proved:
        return replace(family, restriction=restriction, family=family, family_holds=True)
    point = family.counterexample
    if family.status is ProofStatus.REFUTED and point is not None and in_domain(restriction, point):
        return replace(
            family,
            restriction=restriction,
            family=family,
            counterexample_original=MappingProxyType(dict(point)),
        )
    if restriction.finite:
        return replace(_by_instances(left, right, restriction, run, samples), family=family)
    try:
        pair = subst_pair(left, right, restriction)
    except DecideGrammarError as exc:
        raise ProveGrammarError(str(exc)) from exc
    inner_samples = None if samples is None else restricted_samples(restriction, samples)
    inner = run(*pair, samples=inner_samples)
    original = None
    if inner.counterexample is not None:
        original = MappingProxyType(to_original(restriction, inner.counterexample))
        inner = _inside(inner, restriction, original)
    return replace(
        inner,
        restriction=restriction,
        family=family,
        restricted_pair=pair,
        counterexample_original=original,
    )


def _by_instances(
    left: Diagram,
    right: Diagram,
    restriction: Restriction,
    run: Callable[..., ProofOutcome],
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
) -> ProofOutcome:
    """``run`` on each instance of the finite clauses: PROVED when every instance is, carrying
    each instance's outcome; the first REFUTED instance; else NOT_FOUND naming the instances
    left."""
    finite = restriction.finite
    done: list[tuple[str, ProofOutcome]] = []
    open_: list[str] = []
    for combo in itertools.product(*(c.values for c in finite)):
        at = {c.symbol: v for c, v in zip(finite, combo, strict=True)}
        label = ", ".join(f"{k} = {v}" for k, v in at.items())
        one = restricted_outcome(
            left, right, restriction.instance_at(at), run, samples, family_first=False
        )
        if one.status is ProofStatus.REFUTED:
            return replace(one, reason=f"instance {label}: {one.reason}", restriction=restriction)
        if one.proved:
            done.append((label, one))
        else:
            open_.append(label)
    if open_ or not done:
        settled = f" (proved at {'; '.join(k for k, _o in done)})" if done else ""
        return ProofOutcome(
            ProofStatus.NOT_FOUND,
            f"instances left unproved: {'; '.join(open_)}{settled}",
            restriction=restriction,
            instances=tuple(done),
        )
    return ProofOutcome(
        ProofStatus.PROVED,
        f"every instance proved ({'; '.join(k for k, _o in done)}), one certificate each",
        restriction=restriction,
        instances=tuple(done),
    )


def _pair_only(
    left: Diagram,
    right: Diagram,
    restriction: Restriction,
    run: Callable[..., ProofOutcome],
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    **extra: object,
) -> ProofOutcome:
    """``run`` on the restricted pair alone, the outcome carrying the restriction, no family."""
    from archytaszx.semantics.decide import DecideGrammarError
    from archytaszx.semantics.restrict import restricted_samples, subst_pair, to_original

    try:
        pair = subst_pair(left, right, restriction)
    except DecideGrammarError as exc:
        raise ProveGrammarError(str(exc)) from exc
    inner_samples = None if samples is None else restricted_samples(restriction, samples)
    inner = run(*pair, samples=inner_samples, **extra)
    original = None
    if inner.counterexample is not None:
        original = MappingProxyType(to_original(restriction, inner.counterexample))
        inner = _inside(inner, restriction, original)
    return replace(
        inner,
        restriction=restriction,
        family=None,
        family_holds=False,
        restricted_pair=pair,
        counterexample_original=original,
    )


def _inside(
    outcome: ProofOutcome, restriction: Restriction, original: Mapping[str, object]
) -> ProofOutcome:
    """``outcome``, a REFUTED one turned NOT_FOUND when its counterexample lies outside the
    restriction's clauses and side conditions."""
    from archytaszx.semantics.restrict import in_domain

    if outcome.status is ProofStatus.REFUTED and not in_domain(restriction, original):
        return replace(
            outcome,
            status=ProofStatus.NOT_FOUND,
            reason=f"{outcome.reason}; that point lies outside {restriction.text()}",
            counterexample=None,
        )
    return outcome


class _Stopped(BaseException):
    """Raised in a layer callback to stop a search another search has made moot."""


def _restricted_exhaustive(
    left: Diagram,
    right: Diagram,
    restrict: Restriction | str | Mapping[str, str],
    run: Callable[..., ProofOutcome],
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    on_layer: Callable[[LayerReport], None] | None,
    family_first: bool = True,
) -> ProofOutcome:
    """The restricted exhaustive proof: the family search and the restricted pair's search run
    side by side in two threads, time-sliced; the family's PROVED or in-range REFUTED wins,
    otherwise the restricted pair's outcome, the other search stopped at its next layer.
    ``family_first=False`` runs the restricted pair's search only."""
    from archytaszx.semantics.decide import DecideGrammarError
    from archytaszx.semantics.restrict import (
        in_domain,
        parse_restriction,
        restricted_samples,
        subst_pair,
        to_original,
        validate,
    )

    if not isinstance(family_first, bool):
        raise ProveGrammarError(f"family_first must be a bool, got {family_first!r}")
    try:
        restriction = validate(parse_restriction(restrict), left, right)
        pair = None if restriction.empty else subst_pair(left, right, restriction)
    except DecideGrammarError as exc:
        raise ProveGrammarError(str(exc)) from exc
    if pair is None:
        return run(left, right, samples=samples, on_layer=on_layer)
    if not family_first:
        return _pair_only(left, right, restriction, run, samples, on_layer=on_layer)
    inner_samples = None if samples is None else restricted_samples(restriction, samples)
    done = threading.Condition()
    results: dict[str, ProofOutcome | BaseException] = {}
    stop: set[str] = set()

    def layer(name: str) -> Callable[[LayerReport], None]:
        def report(item: LayerReport) -> None:
            if name in stop:
                raise _Stopped
            if on_layer is not None:
                on_layer(item)

        return report

    def work(name: str, a: Diagram, b: Diagram, pts: object) -> None:
        try:
            out: ProofOutcome | BaseException = run(a, b, samples=pts, on_layer=layer(name))
        except BaseException as exc:  # noqa: BLE001
            out = exc
        with done:
            results[name] = out
            done.notify_all()

    def decisive_family(out: ProofOutcome) -> bool:
        if out.proved:
            return True
        point = out.counterexample
        return (
            out.status is ProofStatus.REFUTED
            and point is not None
            and in_domain(restriction, point)
        )

    threads = [
        threading.Thread(target=work, args=("family", left, right, samples), daemon=True),
        threading.Thread(target=work, args=("pair", *pair, inner_samples), daemon=True),
    ]
    for thread in threads:
        thread.start()
    with done:
        while True:
            fam = results.get("family")
            inner = results.get("pair")
            if isinstance(fam, ProofOutcome) and decisive_family(fam):
                stop.add("pair")
                break
            if isinstance(inner, ProofOutcome) and (
                "family" in results or inner.proved or inner.status is ProofStatus.REFUTED
            ):
                stop.add("family")
                break
            if isinstance(inner, BaseException) and not isinstance(inner, _Stopped):
                stop.add("family")
                raise inner
            done.wait()
    fam = results.get("family")
    if isinstance(fam, ProofOutcome) and decisive_family(fam):
        if fam.proved:
            return replace(fam, restriction=restriction, family=fam, family_holds=True)
        point = fam.counterexample
        assert point is not None
        return replace(
            fam,
            restriction=restriction,
            family=fam,
            counterexample_original=MappingProxyType(dict(point)),
        )
    inner = results["pair"]
    assert isinstance(inner, ProofOutcome)
    original = None
    if inner.counterexample is not None:
        original = MappingProxyType(to_original(restriction, inner.counterexample))
        inner = _inside(inner, restriction, original)
    return replace(
        inner,
        restriction=restriction,
        family=fam if isinstance(fam, ProofOutcome) else None,
        restricted_pair=pair,
        counterexample_original=original,
    )


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
                    functools.partial(find_hypothesis_region, hypothesis=hypothesis),
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
        except _INDUCTION_ERRORS:
            continue
    return None


DEFAULT_INDUCTION_DEPTH = 2


def _oracle_refuted(start: Diagram, goal: Diagram, oracle: OracleRefutation) -> ProofOutcome:
    """``REFUTED`` at the oracle's counterexample with its Lean refutation; ``NOT_FOUND``, the
    missing refutation named in its reason, when there is none."""
    from archytaszx.semantics.lean_net import NetUnsupported

    detail = oracle.comparison.reason if oracle.comparison is not None else ""
    reason = f"oracle mismatch at {dict(oracle.counterexample or {})!r}: {detail}"
    try:
        source = lean_refutation(start, goal, oracle.counterexample)
    except NetUnsupported as exc:
        return ProofOutcome(
            ProofStatus.NOT_FOUND,
            f"{reason}; not refuted: no Lean refutation ({exc})",
            counterexample=oracle.counterexample,
        )
    return ProofOutcome(
        ProofStatus.REFUTED, reason, counterexample=oracle.counterexample, refutation=source
    )


def _residual_refuted(
    start: Diagram, goal: Diagram, outcome: ProofOutcome, max_paths: int
) -> ProofOutcome:
    """``outcome`` REFUTED when both pure diagrams' path sums, residual sums evaluated exactly
    up to ``max_paths`` path variables, differ at a boundary index; else ``outcome`` with the
    comparison's reason appended."""
    from archytaszx.algebra.pathsum import PathSumUnsupported
    from archytaszx.semantics.pathsum_diagram import compare_path_sums

    if not (is_pure(start) and is_pure(goal)):
        return outcome
    try:
        found = compare_path_sums(start, goal, max_paths)
    except PathSumUnsupported:
        return outcome
    if found.status != "unequal" or found.left is None or found.right is None:
        return replace(outcome, reason=f"{outcome.reason}; {found.reason}")
    return ProofOutcome(
        ProofStatus.REFUTED,
        f"path sums differ at boundary index {found.index}: {found.left.to_complex()} vs "
        f"{found.right.to_complex()} ({found.reason})",
        counterexample=MappingProxyType({}),
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
    induction_depth: int = DEFAULT_INDUCTION_DEPTH,
    restrict: Restriction | str | Mapping[str, str] | None = None,
    family_first: bool = True,
    enumerate_residual: bool = False,
    max_residual_paths: int = 6,
) -> ProofOutcome:
    """Prove ``start`` equal to ``goal`` by the module docstring's order: interface, oracle,
    search, certification and check, then induction over a free multiplicity when search
    finds no path, nesting at most ``induction_depth`` inductions (0 turns it off); under
    ``restrict``, by :func:`restricted_outcome`, the family skipped when ``family_first`` is
    False. With ``enumerate_residual``, an unrestricted pure pair left NOT_FOUND is REFUTED
    when the path-sum comparison with residual enumeration (at most ``max_residual_paths``
    path variables) finds a differing entry."""
    if not isinstance(family_first, bool):
        raise ProveGrammarError(f"family_first must be a bool, got {family_first!r}")
    if not isinstance(enumerate_residual, bool):
        raise ProveGrammarError(f"enumerate_residual must be a bool, got {enumerate_residual!r}")
    if (
        isinstance(max_residual_paths, bool)
        or not isinstance(max_residual_paths, int)
        or max_residual_paths < 0
    ):
        raise ProveGrammarError(
            f"max_residual_paths must be a nonnegative int, got {max_residual_paths!r}"
        )
    if enumerate_residual:
        outcome = prove(
            start,
            goal,
            moves=moves,
            limits=limits,
            bidirectional=bidirectional,
            cache=cache,
            refute=refute,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
            induction_depth=induction_depth,
            restrict=restrict,
            family_first=family_first,
        )
        if outcome.status is not ProofStatus.NOT_FOUND or restrict is not None:
            return outcome
        return _residual_refuted(start, goal, outcome, max_residual_paths)
    if restrict is not None:
        return restricted_outcome(
            start,
            goal,
            restrict,
            functools.partial(
                prove,
                moves=moves,
                limits=limits,
                bidirectional=bidirectional,
                cache=cache,
                refute=refute,
                max_samples=max_samples,
                tolerance=tolerance,
                max_elements=max_elements,
                induction_depth=induction_depth,
            ),
            samples,
            family_first,
        )
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
    if not (is_pure(start) and is_pure(goal)):
        native = _native_outcome(
            start,
            goal,
            moves,
            limits,
            bidirectional,
            cache,
            samples,
            max_samples,
            tolerance,
            max_elements,
        )
        if native is not None:
            return native
        return _doubled_outcome(
            start,
            goal,
            functools.partial(
                prove,
                moves=moves,
                limits=limits,
                bidirectional=bidirectional,
                cache=cache,
                refute=refute,
                samples=samples,
                max_samples=max_samples,
                tolerance=tolerance,
                max_elements=max_elements,
                induction_depth=induction_depth,
            ),
        )

    interface = interface_reason(start, goal)
    if interface is not None:
        return ProofOutcome(ProofStatus.REFUTED, f"interfaces never agree: {interface}")
    oracle: OracleRefutation | None = None
    if refute:
        oracle = _certify_oracle(
            start,
            goal,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
            defer=True,
        )
        if oracle.counterexample is not None:
            return _oracle_refuted(start, goal, oracle)
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


def prove_exhaustive(
    start: Diagram,
    goal: Diagram,
    *,
    refute: bool = True,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    on_layer: Callable[[LayerReport], None] | None = None,
    algebraic: bool = False,
    restrict: Restriction | str | Mapping[str, str] | None = None,
    family_first: bool = True,
) -> ProofOutcome:
    """Prove ``start`` equal to ``goal`` by :func:`~archytaszx.rewrite.exhaustive.exhaustive_search`
    (or, with ``algebraic``, :func:`~archytaszx.rewrite.exhaustive.algebraic_search`) over every
    registered rule, after the interface and oracle refutation rungs of :func:`prove`; no limit
    bounds the search; under ``restrict``, by :func:`restricted_outcome`, the family skipped
    when ``family_first`` is False."""
    if not isinstance(family_first, bool):
        raise ProveGrammarError(f"family_first must be a bool, got {family_first!r}")
    if restrict is not None:
        return _restricted_exhaustive(
            start,
            goal,
            restrict,
            functools.partial(
                prove_exhaustive,
                refute=refute,
                max_samples=max_samples,
                tolerance=tolerance,
                max_elements=max_elements,
                algebraic=algebraic,
            ),
            samples,
            on_layer,
            family_first,
        )
    for name, value in (("start", start), ("goal", goal)):
        if not isinstance(value, Diagram):
            raise ProveGrammarError(f"{name} must be a Diagram, got {type(value).__name__}")
        invalid = _invalid_reason(value)
        if invalid is not None:
            raise ProveGrammarError(f"{name} {invalid}")
    _check_oracle_args(samples, max_samples, tolerance, max_elements)
    if not (is_pure(start) and is_pure(goal)):
        return _doubled_outcome(
            start,
            goal,
            functools.partial(
                prove_exhaustive,
                refute=refute,
                samples=samples,
                max_samples=max_samples,
                tolerance=tolerance,
                max_elements=max_elements,
                on_layer=on_layer,
                algebraic=algebraic,
            ),
        )
    interface = interface_reason(start, goal)
    if interface is not None:
        return ProofOutcome(ProofStatus.REFUTED, f"interfaces never agree: {interface}")
    oracle: OracleRefutation | None = None
    if refute:
        oracle = _certify_oracle(
            start,
            goal,
            samples=samples,
            max_samples=max_samples,
            tolerance=tolerance,
            max_elements=max_elements,
            defer=True,
        )
        if oracle.counterexample is not None:
            return _oracle_refuted(start, goal, oracle)
    if algebraic:
        for name, value in (("start", start), ("goal", goal)):
            try:
                algebraic_view(value)
            except (ValueError, ArithmeticError) as exc:
                raise ProveGrammarError(
                    f"{name}: algebraic search needs concrete dimensions and algebraic "
                    f"parameters: {exc}"
                ) from exc
    try:
        search = algebraic_search if algebraic else exhaustive_search
        path = search(start, goal, on_layer=on_layer)
    except ExhaustedError as exc:
        return ProofOutcome(ProofStatus.NOT_FOUND, f"exhaustive search: {exc}")
    certificate = certify_proof(path)
    check = _check(certificate, True, oracle, samples, max_samples, tolerance, max_elements)
    if not check.verified:
        return ProofOutcome(
            ProofStatus.CHECK_FAILED,
            f"proof check failed: {check.reason}",
            certificate=certificate,
            check=check,
            counterexample=check.counterexample,
        )
    return ProofOutcome(
        ProofStatus.PROVED,
        f"proved exhaustively in {certificate.length} step(s), "
        f"{len(certificate.forward.steps)} forward and {len(certificate.backward.steps)} "
        f"backward; {check.reason}",
        certificate=certificate,
        check=check,
        assumptions=certificate.assumptions,
    )
