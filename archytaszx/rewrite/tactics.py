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

"""Tactic language and proof search that finds and certifies a derivation between two diagrams.

A :class:`Tactic` maps a diagram to a lazy, deterministically ordered stream of
:class:`TacticOutcome`. Every outcome's ``results`` chain: ``results[0]`` is
:func:`~archytaszx.rewrite.engine.apply` on the input diagram, ``results[i + 1]`` is ``apply``
on ``results[i].diagram``, and ``diagram is results[-1].diagram``; an outcome with no results
holds the input diagram object itself. No tactic mutates a diagram, so
``certify(input, outcome.results)`` replays id for id.

Primitives are :func:`rule` (one outcome per passing match), :func:`identity`, :func:`fail`,
:func:`fixpoint` and :func:`normalize`; combinators are :func:`seq` (depth-first bind, also
``a >> b``), :func:`first` (also ``a | b``), :func:`choice`, :func:`attempt`, :func:`once` and
:func:`repeat`. A :class:`TacticContext` threads a shared cache, an application budget whose
exhaustion raises :class:`BudgetExhausted`, and the :class:`TacticFailure` list recording each
application that raised one of :data:`~archytaszx.rewrite.egraph.APPLICATION_ERRORS`.

:func:`search` is a layered breadth-first search from ``start`` and, when bidirectional, from
``goal``. A state is a diagram hash-consed by its
:func:`~archytaszx.rewrite.normal_form.comparison_view`: bucketed by
:func:`~archytaszx.diagram.compare.canonical_key`, confirmed with
:func:`~archytaszx.diagram.compare.isomorphic`. Each layer expands, in ascending state id, the
start side's states at the current depth and then the goal side's, running every move in order
over each. An outcome with no results is skipped and one over the node bound (the larger input's
node count plus ``node_margin``) is pruned. An outcome meeting a state of the other side ends
the search with a :class:`ProofPath`; one meeting its own side is dropped; any other becomes a
new state one layer deeper. The path's forward results replay from ``start``, its backward
results from ``goal``, and their final diagrams have isomorphic comparison views. Every
diagram in a path is a copy. :class:`SearchLimits` stop the search deterministically, and not
finding a path is a :class:`SearchStatus`, never an exception.
"""

from __future__ import annotations

import abc
import enum
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

from archytaszx.diagram.compare import canonical_key, isomorphic
from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.cache import CacheError, RewriteCache
from archytaszx.rewrite.egraph import APPLICATION_ERRORS, saturation_rules
from archytaszx.rewrite.engine import (
    DEFAULT_GUARD,
    RewriteResult,
    RewriteStep,
    TerminationGuard,
    apply,
    apply_until_fixpoint,
    normal_form_rules,
)
from archytaszx.rewrite.normal_form import comparison_view, view_key, views_isomorphic
from archytaszx.rewrite.rule import Match, RewriteError, RewriteGrammarError, Rule
from archytaszx.rewrite.rules_library import lookup_rule


def _is_int(value: object) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_diagram(what: str, value: object) -> Diagram:
    """``value`` itself when it is a Diagram, else a RewriteGrammarError naming ``what``."""
    if not isinstance(value, Diagram):
        raise RewriteGrammarError(f"{what} must be a Diagram, got {type(value).__name__}")
    return value


def _require_count(what: str, value: object, *, optional: bool = False) -> int | None:
    """``value`` when it is a non-negative int (or ``None`` when ``optional``)."""
    if value is None and optional:
        return None
    if not _is_int(value):
        raise RewriteGrammarError(f"{what} must be an int, got {type(value).__name__}")
    assert isinstance(value, int)
    if value < 0:
        raise RewriteGrammarError(f"{what} must be >= 0, got {value}")
    return value


def _resolve_rule(what: str, value: object) -> Rule:
    """``value`` when it is a Rule, the registered rule when it is a name."""
    if isinstance(value, Rule):
        return value
    if isinstance(value, str):
        return lookup_rule(value)
    raise RewriteGrammarError(f"{what} must be a Rule or a rule name, got {type(value).__name__}")


def _first_of(stream: Iterator[TacticOutcome]) -> TacticOutcome | None:
    """The first item of ``stream`` or ``None``, closing ``stream`` afterwards."""
    try:
        return next(stream, None)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


class BudgetExhausted(RewriteError):
    """A :class:`TacticContext`'s application budget ran out."""

    def __init__(self, applications: int) -> None:
        super().__init__(f"application budget exhausted after {applications} application(s)")
        self.applications = applications

    def __reduce__(self) -> tuple[type[BudgetExhausted], tuple[int]]:
        """Rebuild from ``applications``."""
        return (type(self), (self.applications,))


@dataclass(frozen=True, slots=True)
class TacticFailure:
    """One application that raised: the rule and the error text."""

    rule_name: str
    message: str

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        for name in ("rule_name", "message"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise RewriteGrammarError(
                    f"TacticFailure.{name} must be a str, got {type(value).__name__}"
                )


class TacticContext:
    """Mutable state shared by one tactic run: cache, application budget and failures."""

    def __init__(
        self, *, cache: RewriteCache | None = None, max_applications: int | None = None
    ) -> None:
        """A fresh context with no applications charged and no failures."""
        if cache is not None and not isinstance(cache, RewriteCache):
            raise RewriteGrammarError(
                f"TacticContext: cache must be a RewriteCache, got {type(cache).__name__}"
            )
        self.cache = cache
        self.max_applications = _require_count(
            "TacticContext: max_applications", max_applications, optional=True
        )
        self.applications = 0
        self.failures: list[TacticFailure] = []

    def charge(self) -> None:
        """Count one application, raising :class:`BudgetExhausted` when none is left."""
        if self.max_applications is not None and self.applications >= self.max_applications:
            raise BudgetExhausted(self.applications)
        self.applications += 1

    def matches(self, rule: Rule, diagram: Diagram) -> tuple[Match, ...]:
        """``rule``'s matches in ``diagram``, through the cache when there is one."""
        if self.cache is None:
            return tuple(rule.pattern.find_matches(diagram))
        try:
            return self.cache.matches(rule.pattern, diagram)
        except CacheError as exc:
            raise RewriteGrammarError(
                f"rule {rule.name!r}: the cache rejected its pattern: {type(exc).__name__}: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True, eq=False)
class TacticOutcome:
    """One result of a tactic: the reached diagram and the chained results leading to it."""

    diagram: Diagram
    results: tuple[RewriteResult, ...]

    def __post_init__(self) -> None:
        """Validate the field types and that ``diagram`` is the last result's diagram."""
        _require_diagram("TacticOutcome.diagram", self.diagram)
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, RewriteResult) for result in self.results
        ):
            raise RewriteGrammarError("TacticOutcome.results must be a tuple of RewriteResult")
        if self.results and self.diagram is not self.results[-1].diagram:
            raise RewriteGrammarError("TacticOutcome.diagram must be the last result's diagram")

    @property
    def steps(self) -> tuple[RewriteStep, ...]:
        """Every result's step, in order."""
        return tuple(result.step for result in self.results)


def _require_context(what: str, context: object) -> TacticContext:
    """``context`` when it is a TacticContext, a fresh one when ``None``."""
    if context is None:
        return TacticContext()
    if not isinstance(context, TacticContext):
        raise RewriteGrammarError(
            f"{what}: context must be a TacticContext, got {type(context).__name__}"
        )
    return context


class Tactic(abc.ABC):
    """A diagram-to-outcomes map; see the module docstring for the chaining invariant."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A readable expression of the tactic."""

    @abc.abstractmethod
    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        """Every outcome on ``diagram``, lazily, without validating arguments."""

    def outcomes(
        self, diagram: Diagram, context: TacticContext | None = None
    ) -> Iterator[TacticOutcome]:
        """Every outcome on ``diagram``, lazily, under ``context`` (a fresh one when ``None``)."""
        _require_diagram(f"{self.name}: diagram", diagram)
        return self._outcomes(diagram, _require_context(self.name, context))

    def run(self, diagram: Diagram, context: TacticContext | None = None) -> TacticOutcome | None:
        """The first outcome on ``diagram``, or ``None``."""
        return _first_of(self.outcomes(diagram, context))

    def all(
        self, diagram: Diagram, context: TacticContext | None = None, *, limit: int | None = None
    ) -> tuple[TacticOutcome, ...]:
        """The outcomes on ``diagram``, at most ``limit`` of them when given."""
        count = _require_count(f"{self.name}: limit", limit, optional=True)
        stream = self.outcomes(diagram, context)
        return tuple(stream if count is None else itertools.islice(stream, count))

    def __rshift__(self, other: Tactic) -> Tactic:
        """``seq(self, other)``."""
        return seq(self, other)

    def __or__(self, other: Tactic) -> Tactic:
        """``first(self, other)``."""
        return first(self, other)

    def __repr__(self) -> str:
        """The tactic's name."""
        return self.name


def _require_tactics(what: str, tactics: Sequence[object]) -> tuple[Tactic, ...]:
    """``tactics`` as a non-empty tuple of Tactic."""
    if not tactics:
        raise RewriteGrammarError(f"{what} needs at least one tactic")
    for item in tactics:
        if not isinstance(item, Tactic):
            raise RewriteGrammarError(f"{what}: expected a Tactic, got {type(item).__name__}")
    return tuple(item for item in tactics if isinstance(item, Tactic))


class _Rule(Tactic):
    """Each passing match of one rule, applied."""

    def __init__(self, rule: Rule, focus: frozenset[NodeId] | None) -> None:
        self._rule = rule
        self._focus = focus

    @property
    def name(self) -> str:
        """``rule(<name>)``, with the sorted focus when given."""
        if self._focus is None:
            return f"rule({self._rule.name})"
        return f"rule({self._rule.name}, focus={sorted(self._focus)})"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        for match in context.matches(self._rule, diagram):
            if not match.all_side_conditions_passed:
                continue
            if self._focus is not None and self._focus.isdisjoint(match.support_node_ids):
                continue
            context.charge()
            try:
                result = apply(diagram, self._rule, match)
            except APPLICATION_ERRORS as exc:
                message = f"{type(exc).__name__}: {exc}"
                context.failures.append(TacticFailure(self._rule.name, message))
                continue
            yield TacticOutcome(result.diagram, (result,))


class _Identity(Tactic):
    """The input, unchanged."""

    @property
    def name(self) -> str:
        """``identity``."""
        return "identity"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        yield TacticOutcome(diagram, ())


class _Fail(Tactic):
    """No outcome."""

    @property
    def name(self) -> str:
        """``fail``."""
        return "fail"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        return iter(())


class _Combinator(Tactic):
    """A tactic over a tuple of children, named ``label(child, ...)``."""

    label = ""

    def __init__(self, children: tuple[Tactic, ...]) -> None:
        self._children = children

    @property
    def name(self) -> str:
        """``label(child, ...)``."""
        return f"{self.label}({', '.join(child.name for child in self._children)})"


class _Seq(_Combinator):
    """Depth-first bind of the children."""

    label = "seq"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        return self._bind(0, diagram, (), context)

    def _bind(
        self,
        index: int,
        diagram: Diagram,
        results: tuple[RewriteResult, ...],
        context: TacticContext,
    ) -> Iterator[TacticOutcome]:
        if index == len(self._children):
            yield TacticOutcome(diagram, results)
            return
        for outcome in self._children[index]._outcomes(diagram, context):
            yield from self._bind(index + 1, outcome.diagram, results + outcome.results, context)


class _First(_Combinator):
    """The outcomes of the first child yielding any."""

    label = "first"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        for child in self._children:
            stream = child._outcomes(diagram, context)
            head = next(stream, None)
            if head is not None:
                yield head
                yield from stream
                return


class _Choice(_Combinator):
    """Every outcome of every child, in child order."""

    label = "choice"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        for child in self._children:
            yield from child._outcomes(diagram, context)


class _Once(_Combinator):
    """The first outcome of the child only."""

    label = "once"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        head = _first_of(self._children[0]._outcomes(diagram, context))
        if head is not None:
            yield head


class _Repeat(Tactic):
    """The child's first outcome, taken repeatedly until it stalls, loops or hits ``max_times``."""

    def __init__(self, child: Tactic, max_times: int) -> None:
        self._child = child
        self._max_times = max_times

    @property
    def name(self) -> str:
        """``repeat(child)``, with ``max_times`` when not the default."""
        if self._max_times == _DEFAULT_REPEAT:
            return f"repeat({self._child.name})"
        return f"repeat({self._child.name}, max_times={self._max_times})"

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        current = diagram
        results: tuple[RewriteResult, ...] = ()
        visited: dict[str, list[Diagram]] = {canonical_key(diagram): [diagram]}
        for _ in range(self._max_times):
            outcome = _first_of(self._child._outcomes(current, context))
            if outcome is None or not outcome.results:
                break
            bucket = visited.setdefault(canonical_key(outcome.diagram), [])
            if any(isomorphic(outcome.diagram, seen) for seen in bucket):
                break
            bucket.append(outcome.diagram)
            results += outcome.results
            current = outcome.diagram
        yield TacticOutcome(current, results)


class _Fixpoint(Tactic):
    """One :func:`~archytaszx.rewrite.engine.apply_until_fixpoint` run."""

    def __init__(self, rules: tuple[Rule, ...], guard: TerminationGuard, name: str) -> None:
        self._rules = rules
        self._guard = guard
        self._name = name

    @property
    def name(self) -> str:
        """``fixpoint(rule, ...)`` or the name it was given."""
        return self._name

    def _outcomes(self, diagram: Diagram, context: TacticContext) -> Iterator[TacticOutcome]:
        collected: list[RewriteResult] = []

        def on_result(result: RewriteResult) -> None:
            context.charge()
            collected.append(result)

        try:
            apply_until_fixpoint(
                diagram, self._rules, guard=self._guard, cache=context.cache, on_result=on_result
            )
        except BudgetExhausted:
            raise
        except APPLICATION_ERRORS as exc:
            context.failures.append(TacticFailure(self.name, f"{type(exc).__name__}: {exc}"))
        reached = collected[-1].diagram if collected else diagram
        yield TacticOutcome(reached, tuple(collected))


_DEFAULT_REPEAT = 128


def rule(r: Rule | str, *, focus: Iterable[NodeId] | None = None) -> Tactic:
    """One outcome per match of ``r`` whose side conditions pass and, when ``focus`` is given,
    whose support meets ``focus``; a name resolves through
    :func:`~archytaszx.rewrite.rules_library.lookup_rule`."""
    resolved = _resolve_rule("rule", r)
    if focus is None:
        return _Rule(resolved, None)
    if isinstance(focus, (str, bytes)) or not isinstance(focus, Iterable):
        raise RewriteGrammarError(f"rule: focus must be an iterable of node ids, got {focus!r}")
    ids = tuple(focus)
    if not all(_is_int(node_id) for node_id in ids):
        raise RewriteGrammarError(f"rule: every focus entry must be an int node id, got {ids!r}")
    return _Rule(resolved, frozenset(ids))


def identity() -> Tactic:
    """One outcome with no results."""
    return _Identity()


def fail() -> Tactic:
    """No outcomes."""
    return _Fail()


def _flatten(kind: type[_Combinator], tactics: tuple[Tactic, ...]) -> tuple[Tactic, ...]:
    """``tactics`` with each direct child of type ``kind`` replaced by its children."""
    flat: list[Tactic] = []
    for tactic in tactics:
        if type(tactic) is kind:
            assert isinstance(tactic, _Combinator)
            flat.extend(tactic._children)
        else:
            flat.append(tactic)
    return tuple(flat)


def seq(*tactics: Tactic) -> Tactic:
    """Every outcome of the last tactic over every outcome of the ones before, depth first."""
    return _Seq(_flatten(_Seq, _require_tactics("seq", tactics)))


def first(*tactics: Tactic) -> Tactic:
    """The outcomes of the first tactic that has at least one."""
    return _First(_flatten(_First, _require_tactics("first", tactics)))


def choice(*tactics: Tactic) -> Tactic:
    """Every outcome of each tactic, in order."""
    return _Choice(_flatten(_Choice, _require_tactics("choice", tactics)))


def attempt(t: Tactic) -> Tactic:
    """``first(t, identity())``."""
    return first(*_require_tactics("attempt", (t,)), identity())


def once(t: Tactic) -> Tactic:
    """The first outcome of ``t`` only."""
    return _Once(_require_tactics("once", (t,)))


def repeat(t: Tactic, *, max_times: int = _DEFAULT_REPEAT) -> Tactic:
    """One outcome: ``t``'s first outcome taken repeatedly, at most ``max_times`` times, until it
    has none, has no results or revisits an isomorphic diagram (that step not taken)."""
    (child,) = _require_tactics("repeat", (t,))
    count = _require_count("repeat: max_times", max_times)
    assert count is not None
    return _Repeat(child, count)


def _require_rule_list(what: str, rules: object) -> tuple[Rule, ...]:
    """``rules`` as a tuple of Rule, names resolved."""
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise RewriteGrammarError(f"{what}: rules must be a Sequence, got {type(rules).__name__}")
    return tuple(_resolve_rule(what, item) for item in rules)


def fixpoint(rules: Sequence[Rule | str], *, guard: TerminationGuard = DEFAULT_GUARD) -> Tactic:
    """One outcome: :func:`~archytaszx.rewrite.engine.apply_until_fixpoint` over ``rules``, each
    result charged to the context."""
    resolved = _require_rule_list("fixpoint", rules)
    if not isinstance(guard, TerminationGuard):
        raise RewriteGrammarError(
            f"fixpoint: guard must be a TerminationGuard, got {type(guard).__name__}"
        )
    return _Fixpoint(resolved, guard, f"fixpoint({', '.join(r.name for r in resolved)})")


def normalize(*, guard: TerminationGuard = DEFAULT_GUARD) -> Tactic:
    """``fixpoint(normal_form_rules())``, named ``normalize``."""
    if not isinstance(guard, TerminationGuard):
        raise RewriteGrammarError(
            f"normalize: guard must be a TerminationGuard, got {type(guard).__name__}"
        )
    return _Fixpoint(normal_form_rules(), guard, "normalize")


def rewrite_tactics(rules: Sequence[Rule] | None = None) -> tuple[Tactic, ...]:
    """One :func:`rule` tactic per rule, default
    :func:`~archytaszx.rewrite.egraph.saturation_rules`."""
    if rules is None:
        return tuple(_Rule(r, None) for r in saturation_rules())
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise RewriteGrammarError(
            f"rewrite_tactics: rules must be a Sequence of Rule, got {type(rules).__name__}"
        )
    if not all(isinstance(r, Rule) for r in rules):
        raise RewriteGrammarError("rewrite_tactics: every element of rules must be a Rule")
    return tuple(_Rule(r, None) for r in rules)


def default_moves() -> tuple[Tactic, ...]:
    """``(normalize(), *rewrite_tactics())``."""
    return (normalize(), *rewrite_tactics())


class SearchStatus(enum.Enum):
    """How :func:`search` ended."""

    FOUND = "found"
    EXHAUSTED = "exhausted"
    DEPTH_LIMIT = "depth_limit"
    STATE_LIMIT = "state_limit"
    APPLICATION_LIMIT = "application_limit"


@dataclass(frozen=True, slots=True)
class SearchLimits:
    """Ceilings on one :func:`search`: moves per side, distinct states, applications, and the
    node margin of its bound."""

    max_depth: int = 4
    max_states: int = 2048
    max_applications: int = 20_000
    node_margin: int = 4

    def __post_init__(self) -> None:
        """Reject a field that is not a non-negative int."""
        for name in ("max_depth", "max_states", "max_applications", "node_margin"):
            _require_count(f"SearchLimits.{name}", getattr(self, name))


DEFAULT_SEARCH_LIMITS = SearchLimits()


class Side(enum.Enum):
    """Which input a search state descends from."""

    START = "start"
    GOAL = "goal"


@dataclass(frozen=True, slots=True, eq=False)
class ProofStep:
    """One search move: the tactic's name and its non-empty chained results."""

    tactic: str
    results: tuple[RewriteResult, ...]

    def __post_init__(self) -> None:
        """Validate every field's type, and that ``results`` is non-empty."""
        if not isinstance(self.tactic, str):
            raise RewriteGrammarError(
                f"ProofStep.tactic must be a str, got {type(self.tactic).__name__}"
            )
        if (
            not isinstance(self.results, tuple)
            or not self.results
            or not all(isinstance(result, RewriteResult) for result in self.results)
        ):
            raise RewriteGrammarError(
                "ProofStep.results must be a non-empty tuple of RewriteResult"
            )


def _flat_results(steps: tuple[ProofStep, ...]) -> tuple[RewriteResult, ...]:
    """Every step's results, concatenated."""
    return tuple(result for step in steps for result in step.results)


@dataclass(frozen=True, slots=True, eq=False)
class ProofPath:
    """Moves from ``start`` and from ``goal`` whose final diagrams have isomorphic views."""

    start: Diagram
    goal: Diagram
    forward: tuple[ProofStep, ...]
    backward: tuple[ProofStep, ...]

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        _require_diagram("ProofPath.start", self.start)
        _require_diagram("ProofPath.goal", self.goal)
        for name in ("forward", "backward"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                isinstance(step, ProofStep) for step in value
            ):
                raise RewriteGrammarError(f"ProofPath.{name} must be a tuple of ProofStep")

    @property
    def forward_results(self) -> tuple[RewriteResult, ...]:
        """The forward moves' results, in order."""
        return _flat_results(self.forward)

    @property
    def backward_results(self) -> tuple[RewriteResult, ...]:
        """The backward moves' results, in order."""
        return _flat_results(self.backward)

    @property
    def forward_meet(self) -> Diagram:
        """The last forward result's diagram, or ``start``."""
        results = self.forward_results
        return results[-1].diagram if results else self.start

    @property
    def backward_meet(self) -> Diagram:
        """The last backward result's diagram, or ``goal``."""
        results = self.backward_results
        return results[-1].diagram if results else self.goal

    @property
    def length(self) -> int:
        """The total number of results on both sides."""
        return len(self.forward_results) + len(self.backward_results)

    @property
    def moves(self) -> tuple[str, ...]:
        """The forward moves' tactic names."""
        return tuple(step.tactic for step in self.forward)

    @property
    def backward_moves(self) -> tuple[str, ...]:
        """The backward moves' tactic names."""
        return tuple(step.tactic for step in self.backward)


@dataclass(frozen=True, slots=True, eq=False)
class SearchResult:
    """A :func:`search`'s status, path when found, counters and recorded failures."""

    status: SearchStatus
    path: ProofPath | None
    states: int
    expanded: int
    applications: int
    pruned: int
    depth: int
    failures: tuple[TacticFailure, ...]

    def __post_init__(self) -> None:
        """Validate every field's type, and that ``path`` is set exactly when found."""
        if not isinstance(self.status, SearchStatus):
            raise RewriteGrammarError(
                f"SearchResult.status must be a SearchStatus, got {type(self.status).__name__}"
            )
        if self.path is not None and not isinstance(self.path, ProofPath):
            raise RewriteGrammarError(
                f"SearchResult.path must be a ProofPath or None, got {type(self.path).__name__}"
            )
        if (self.path is not None) != (self.status is SearchStatus.FOUND):
            raise RewriteGrammarError("SearchResult.path must be set exactly when FOUND")
        for name in ("states", "expanded", "applications", "pruned", "depth"):
            if not _is_int(getattr(self, name)):
                raise RewriteGrammarError(f"SearchResult.{name} must be an int")
        if not isinstance(self.failures, tuple) or not all(
            isinstance(item, TacticFailure) for item in self.failures
        ):
            raise RewriteGrammarError("SearchResult.failures must be a tuple of TacticFailure")

    @property
    def found(self) -> bool:
        """True when a path was found."""
        return self.status is SearchStatus.FOUND


@dataclass(frozen=True, slots=True, eq=False)
class _State:
    """One search state: its side, diagram as produced, view, parent, producing move, depth."""

    side: Side
    diagram: Diagram
    view: Diagram
    parent: int | None
    step: ProofStep | None
    depth: int


class _Found(Exception):
    """Internal signal carrying the forward and backward step chains of a meet."""

    def __init__(self, forward: tuple[ProofStep, ...], backward: tuple[ProofStep, ...]) -> None:
        super().__init__("found")
        self.forward = forward
        self.backward = backward


class _Accepted(Exception):
    """Internal signal carrying the step chain to a state the predicate accepted."""

    def __init__(self, forward: tuple[ProofStep, ...], diagram: Diagram, value: object) -> None:
        super().__init__("accepted")
        self.forward = forward
        self.diagram = diagram
        self.value = value


class _Limit(Exception):
    """Internal signal ending a search with ``status``."""

    def __init__(self, status: SearchStatus) -> None:
        super().__init__(status.value)
        self.status = status


class _Search:
    """The mutable state of one :func:`search` run."""

    def __init__(
        self,
        moves: tuple[Tactic, ...],
        limits: SearchLimits,
        bound: int,
        ctx: TacticContext,
        accept: Callable[[Diagram], object] | None = None,
    ) -> None:
        self.accept = accept
        self.moves = moves
        self.limits = limits
        self.bound = bound
        self.ctx = ctx
        self.states: list[_State] = []
        self.buckets: dict[str, list[int]] = {}
        self.expanded = 0
        self.pruned = 0

    def find(self, view: Diagram) -> tuple[str, int | None]:
        """``view``'s key and the state whose view is isomorphic to it, if any."""
        key = view_key(view)
        for index in self.buckets.get(key, ()):
            if views_isomorphic(view, self.states[index].view):
                return key, index
        return key, None

    def insert(self, state: _State, key: str) -> int:
        """Store ``state`` under ``key`` and return its id."""
        self.states.append(state)
        self.buckets.setdefault(key, []).append(len(self.states) - 1)
        return len(self.states) - 1

    def chain(self, index: int | None) -> tuple[ProofStep, ...]:
        """The steps from state ``index``'s root to it."""
        steps: list[ProofStep] = []
        while index is not None:
            state = self.states[index]
            if state.step is not None:
                steps.append(state.step)
            index = state.parent
        return tuple(reversed(steps))

    def expand(self, index: int) -> None:
        """Run every move on state ``index`` and record what its outcomes reach."""
        self.expanded += 1
        state = self.states[index]
        for move in self.moves:
            for outcome in move._outcomes(state.diagram, self.ctx):
                if not outcome.results:
                    continue
                if len(outcome.diagram.nodes) > self.bound:
                    self.pruned += 1
                    continue
                view = comparison_view(outcome.diagram)
                key, found = self.find(view)
                step = ProofStep(move.name, outcome.results)
                if found is None and self.accept is not None:
                    value = self.accept(outcome.diagram)
                    if value is not None:
                        raise _Accepted(self.chain(index) + (step,), outcome.diagram, value)
                if found is not None:
                    if self.states[found].side is state.side:
                        continue
                    here = self.chain(index) + (step,)
                    there = self.chain(found)
                    if state.side is Side.START:
                        raise _Found(here, there)
                    raise _Found(there, here)
                if len(self.states) >= self.limits.max_states:
                    raise _Limit(SearchStatus.STATE_LIMIT)
                child = _State(state.side, outcome.diagram, view, index, step, state.depth + 1)
                self.insert(child, key)


def _detached_steps(steps: tuple[ProofStep, ...]) -> tuple[ProofStep, ...]:
    """``steps`` with every result carrying a copy of its diagram."""
    return tuple(
        replace(
            step,
            results=tuple(replace(r, diagram=r.diagram.copy()) for r in step.results),
        )
        for step in steps
    )


def search(
    start: Diagram,
    goal: Diagram,
    *,
    moves: Sequence[Tactic] | None = None,
    limits: SearchLimits = DEFAULT_SEARCH_LIMITS,
    bidirectional: bool = True,
    cache: RewriteCache | None = None,
) -> SearchResult:
    """A derivation between ``start`` and ``goal`` by layered search over ``moves`` (default
    :func:`default_moves`); see the module docstring."""
    _require_diagram("search: start", start)
    _require_diagram("search: goal", goal)
    if moves is None:
        move_set = default_moves()
    elif isinstance(moves, (str, bytes)) or not isinstance(moves, Sequence):
        raise RewriteGrammarError(
            f"search: moves must be a Sequence of Tactic, got {type(moves).__name__}"
        )
    elif not all(isinstance(move, Tactic) for move in moves):
        raise RewriteGrammarError("search: every element of moves must be a Tactic")
    else:
        move_set = tuple(moves)
    if not isinstance(limits, SearchLimits):
        raise RewriteGrammarError(
            f"search: limits must be a SearchLimits, got {type(limits).__name__}"
        )
    if not isinstance(bidirectional, bool):
        raise RewriteGrammarError(
            f"search: bidirectional must be a bool, got {type(bidirectional).__name__}"
        )
    ctx = TacticContext(cache=cache, max_applications=limits.max_applications)
    bound = max(len(start.nodes), len(goal.nodes)) + limits.node_margin
    run = _Search(move_set, limits, bound, ctx)
    start_copy, goal_copy = start.copy(), goal.copy()
    start_view = comparison_view(start_copy)
    run.insert(_State(Side.START, start_copy, start_view, None, None, 0), view_key(start_view))
    sides = (Side.START, Side.GOAL) if bidirectional else (Side.START,)
    status = SearchStatus.DEPTH_LIMIT
    forward: tuple[ProofStep, ...] = ()
    backward: tuple[ProofStep, ...] = ()
    depth = 0
    goal_view = comparison_view(goal_copy)
    key, found = run.find(goal_view)
    if found is not None:
        status = SearchStatus.FOUND
    else:
        run.insert(_State(Side.GOAL, goal_copy, goal_view, None, None, 0), key)
        try:
            for depth in range(limits.max_depth):
                for side in sides:
                    layer = [
                        index
                        for index, state in enumerate(run.states)
                        if state.side is side and state.depth == depth
                    ]
                    for index in layer:
                        run.expand(index)
                if not any(s.side in sides and s.depth == depth + 1 for s in run.states):
                    status = SearchStatus.EXHAUSTED
                    depth += 1
                    break
            else:
                depth = limits.max_depth
        except _Found as meet:
            status = SearchStatus.FOUND
            forward, backward = meet.forward, meet.backward
        except _Limit as signal:
            status = signal.status
        except BudgetExhausted:
            status = SearchStatus.APPLICATION_LIMIT
    path = None
    if status is SearchStatus.FOUND:
        path = ProofPath(
            start=start.copy(),
            goal=goal.copy(),
            forward=_detached_steps(forward),
            backward=_detached_steps(backward),
        )
    return SearchResult(
        status=status,
        path=path,
        states=len(run.states),
        expanded=run.expanded,
        applications=ctx.applications,
        pruned=run.pruned,
        depth=depth,
        failures=tuple(ctx.failures),
    )


@dataclass(frozen=True, slots=True, eq=False)
class AcceptedState:
    """A :func:`search_for` hit: the path from the start, and what the predicate returned."""

    path: ProofPath
    value: object


def search_for(
    start: Diagram,
    accept: Callable[[Diagram], object],
    *,
    moves: Sequence[Tactic] | None = None,
    limits: SearchLimits = DEFAULT_SEARCH_LIMITS,
    cache: RewriteCache | None = None,
) -> tuple[SearchResult, AcceptedState | None]:
    """The first diagram reachable from ``start``, ``start`` included, on which ``accept``
    returns a value other than None, with the forward path to it as a :class:`ProofPath`."""
    _require_diagram("search_for: start", start)
    if not callable(accept):
        raise RewriteGrammarError("search_for: accept must be callable")
    move_set = default_moves() if moves is None else tuple(moves)
    if not all(isinstance(move, Tactic) for move in move_set):
        raise RewriteGrammarError("search_for: every element of moves must be a Tactic")
    if not isinstance(limits, SearchLimits):
        raise RewriteGrammarError(
            f"search_for: limits must be a SearchLimits, got {type(limits).__name__}"
        )
    ctx = TacticContext(cache=cache, max_applications=limits.max_applications)
    run = _Search(move_set, limits, len(start.nodes) + limits.node_margin, ctx, accept)
    start_copy = start.copy()
    start_view = comparison_view(start_copy)
    run.insert(_State(Side.START, start_copy, start_view, None, None, 0), view_key(start_view))
    status = SearchStatus.DEPTH_LIMIT
    hit: AcceptedState | None = None
    depth = 0
    first = accept(start_copy)
    if first is not None:
        status = SearchStatus.FOUND
        hit = AcceptedState(ProofPath(start.copy(), start.copy(), (), ()), first)
    else:
        try:
            for depth in range(limits.max_depth):
                for index in [i for i, st in enumerate(run.states) if st.depth == depth]:
                    run.expand(index)
                if not any(st.depth == depth + 1 for st in run.states):
                    status = SearchStatus.EXHAUSTED
                    depth += 1
                    break
            else:
                depth = limits.max_depth
        except _Accepted as accepted:
            status = SearchStatus.FOUND
            steps = _detached_steps(accepted.forward)
            hit = AcceptedState(
                ProofPath(start.copy(), accepted.diagram.copy(), steps, ()), accepted.value
            )
        except _Limit as signal:
            status = signal.status
        except BudgetExhausted:
            status = SearchStatus.APPLICATION_LIMIT
    result = SearchResult(
        status=status,
        path=hit.path if hit is not None else None,
        states=len(run.states),
        expanded=run.expanded,
        applications=ctx.applications,
        pruned=run.pruned,
        depth=depth,
        failures=tuple(ctx.failures),
    )
    return result, hit
