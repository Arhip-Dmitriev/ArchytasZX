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

"""Equality saturation: a non-destructive e-graph over diagrams with cost-based extraction.

An e-node is one diagram up to isomorphism of its
:func:`~archytaszx.rewrite.normal_form.comparison_view`: identity is the view's
:func:`~archytaszx.diagram.compare.canonical_key`, a key hit confirmed with
:func:`~archytaszx.diagram.compare.isomorphic`. Each e-node keeps the first-seen representative
exactly as produced. E-classes are a union-find over e-nodes, the smaller root id winning a
union; a class id is its root's e-node id.

Every stored :class:`EGraphEdge` is one :func:`~archytaszx.rewrite.engine.apply` call on its
parent's representative. A ``TREE`` edge created its child, whose representative is the
result diagram, in the parent's class. A ``MERGE`` edge's result is view-isomorphic to a child
already in another class, and unions the two classes. A result already in the parent's own
class is counted as redundant and not stored.

:meth:`EGraph.saturate` runs rounds. A round snapshots the e-nodes not yet expanded under the
given rules, in ascending id, and applies every rule at every match whose side conditions
all pass. An error from one application is recorded as a :class:`FailedApplication`; a result
over the node bound (largest root's node count plus ``node_margin``) is pruned. An e-node is
expanded again under a different rule tuple (by identity) or a larger node bound. The
:class:`SaturationLimits` stop the run deterministically: the e-node limit before an insertion,
the application limit before an apply, the iteration limit before a round.

:meth:`EGraph.extract` is the exact argmin of a :data:`CostFunction` over a class's members,
ties going to the lowest e-node id. :meth:`EGraph.path_from_root` follows ``TREE`` edges back
to a root, so its results replay from the root's representative onto the e-node's;
:meth:`EGraph.explain` is a shortest path of stored edges between two equivalent e-nodes.
:func:`simplify` adds one diagram, saturates and extracts its class. Every diagram, edge and
result the e-graph hands out carries a copy of the stored diagram.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import NewType

from archytaszx.algebra.dimension import DimensionError
from archytaszx.algebra.phase import PhaseError
from archytaszx.algebra.scalar import ScalarError
from archytaszx.diagram.bangbox import BangBoxError
from archytaszx.diagram.compare import canonical_key, isomorphic
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, GraphError
from archytaszx.diagram.validate import ValidateError
from archytaszx.rewrite.cache import CacheError, RewriteCache
from archytaszx.rewrite.engine import RewriteResult, apply
from archytaszx.rewrite.normal_form import comparison_view
from archytaszx.rewrite.rule import Match, RewriteError, RewriteGrammarError, Rule
from archytaszx.rewrite.rules_library import RULES

ENodeId = NewType("ENodeId", int)
EClassId = NewType("EClassId", int)

CostValue = int | float | tuple[int | float, ...]
CostFunction = Callable[[Diagram], CostValue]

_APPLICATION_ERRORS: tuple[type[Exception], ...] = (
    RewriteError,
    BangBoxError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    ScalarError,
    ValidateError,
)


def _is_int(value: object) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_cost(value: object) -> bool:
    """Whether ``value`` is an int, a non-NaN float, or a tuple of them, bools excluded."""
    if isinstance(value, tuple):
        return all(_is_cost(item) and not isinstance(item, tuple) for item in value)
    if isinstance(value, float):
        return not math.isnan(value)
    return isinstance(value, int) and not isinstance(value, bool)


def _require_diagram(what: str, value: object) -> Diagram:
    """``value`` itself when it is a Diagram, else a RewriteGrammarError naming ``what``."""
    if not isinstance(value, Diagram):
        raise RewriteGrammarError(f"{what} must be a Diagram, got {type(value).__name__}")
    return value


def _require_rules(what: str, rules: Sequence[Rule] | None) -> tuple[Rule, ...]:
    """``rules`` as a tuple, :func:`saturation_rules` when ``None``."""
    if rules is None:
        return saturation_rules()
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise RewriteGrammarError(
            f"{what}: rules must be a Sequence of Rule, got {type(rules).__name__}"
        )
    if not all(isinstance(rule, Rule) for rule in rules):
        raise RewriteGrammarError(f"{what}: every element of rules must be a Rule")
    return tuple(rules)


def _detached(result: RewriteResult) -> RewriteResult:
    """``result`` with a copy of its diagram."""
    return replace(result, diagram=result.diagram.copy())


def _same_rules(first: tuple[Rule, ...], second: tuple[Rule, ...]) -> bool:
    """Whether the two tuples hold the same rule objects in the same order."""
    return len(first) == len(second) and all(a is b for a, b in zip(first, second, strict=True))


def saturation_rules() -> tuple[Rule, ...]:
    """Every rule in :data:`~archytaszx.rewrite.rules_library.RULES`, in registry order."""
    return tuple(RULES.values())


class EdgeKind(enum.Enum):
    """How a stored edge relates its child to the graph."""

    TREE = "tree"
    """The edge created its child."""

    MERGE = "merge"
    """The child pre-existed in another class; the edge unioned the two classes."""


class SaturationStop(enum.Enum):
    """Why :meth:`EGraph.saturate` stopped."""

    SATURATED = "saturated"
    ITERATION_LIMIT = "iteration_limit"
    ENODE_LIMIT = "enode_limit"
    APPLICATION_LIMIT = "application_limit"


@dataclass(frozen=True, slots=True)
class SaturationLimits:
    """Ceilings on one :meth:`EGraph.saturate` call and the node margin of its bound."""

    max_iterations: int = 6
    max_enodes: int = 512
    max_applications: int = 20_000
    node_margin: int = 4

    def __post_init__(self) -> None:
        """Reject a field that is not a non-negative int."""
        for name in ("max_iterations", "max_enodes", "max_applications", "node_margin"):
            value = getattr(self, name)
            if not _is_int(value):
                raise RewriteGrammarError(
                    f"SaturationLimits.{name} must be an int, got {type(value).__name__}"
                )
            if value < 0:
                raise RewriteGrammarError(f"SaturationLimits.{name} must be >= 0, got {value}")


DEFAULT_LIMITS = SaturationLimits()


@dataclass(frozen=True, slots=True)
class FailedApplication:
    """One application that raised: the e-node, the rule and the error text."""

    enode: ENodeId
    rule_name: str
    message: str

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        if not _is_int(self.enode):
            raise RewriteGrammarError(
                f"FailedApplication.enode must be an int, got {type(self.enode).__name__}"
            )
        if not isinstance(self.rule_name, str):
            raise RewriteGrammarError(
                f"FailedApplication.rule_name must be a str, got {type(self.rule_name).__name__}"
            )
        if not isinstance(self.message, str):
            raise RewriteGrammarError(
                f"FailedApplication.message must be a str, got {type(self.message).__name__}"
            )


@dataclass(frozen=True, slots=True)
class SaturationReport:
    """The counters and stop reason of one :meth:`EGraph.saturate` call."""

    stop_reason: SaturationStop
    iterations: int
    enodes_added: int
    merges: int
    applications: int
    redundant: int
    pruned: int
    failed: tuple[FailedApplication, ...]

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        if not isinstance(self.stop_reason, SaturationStop):
            raise RewriteGrammarError(
                "SaturationReport.stop_reason must be a SaturationStop, got "
                f"{type(self.stop_reason).__name__}"
            )
        for name in ("iterations", "enodes_added", "merges", "applications", "redundant", "pruned"):
            if not _is_int(getattr(self, name)):
                raise RewriteGrammarError(f"SaturationReport.{name} must be an int")
        if not isinstance(self.failed, tuple) or not all(
            isinstance(item, FailedApplication) for item in self.failed
        ):
            raise RewriteGrammarError(
                "SaturationReport.failed must be a tuple of FailedApplication"
            )

    @property
    def saturated(self) -> bool:
        """True when the run stopped with nothing left to expand."""
        return self.stop_reason is SaturationStop.SATURATED


@dataclass(frozen=True, slots=True, eq=False)
class EGraphEdge:
    """One stored application: ``result`` is ``apply`` on ``parent``'s representative."""

    parent: ENodeId
    child: ENodeId
    rule_name: str
    result: RewriteResult
    kind: EdgeKind

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        if not _is_int(self.parent):
            raise RewriteGrammarError(
                f"EGraphEdge.parent must be an int, got {type(self.parent).__name__}"
            )
        if not _is_int(self.child):
            raise RewriteGrammarError(
                f"EGraphEdge.child must be an int, got {type(self.child).__name__}"
            )
        if not isinstance(self.rule_name, str):
            raise RewriteGrammarError(
                f"EGraphEdge.rule_name must be a str, got {type(self.rule_name).__name__}"
            )
        if not isinstance(self.result, RewriteResult):
            raise RewriteGrammarError(
                f"EGraphEdge.result must be a RewriteResult, got {type(self.result).__name__}"
            )
        if not isinstance(self.kind, EdgeKind):
            raise RewriteGrammarError(
                f"EGraphEdge.kind must be an EdgeKind, got {type(self.kind).__name__}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class Extraction:
    """A class's cheapest member: its id, a copy of its representative, its cost, and the
    root and results that replay onto it."""

    eclass: EClassId
    enode: ENodeId
    diagram: Diagram
    cost: CostValue
    root: ENodeId
    results: tuple[RewriteResult, ...]

    def __post_init__(self) -> None:
        """Validate every field's type, in declaration order."""
        for name in ("eclass", "enode"):
            if not _is_int(getattr(self, name)):
                raise RewriteGrammarError(f"Extraction.{name} must be an int")
        if not isinstance(self.diagram, Diagram):
            raise RewriteGrammarError(
                f"Extraction.diagram must be a Diagram, got {type(self.diagram).__name__}"
            )
        if not _is_cost(self.cost):
            raise RewriteGrammarError(
                "Extraction.cost must be an int, a non-NaN float or a tuple of them, got "
                f"{type(self.cost).__name__}"
            )
        if not _is_int(self.root):
            raise RewriteGrammarError(
                f"Extraction.root must be an int, got {type(self.root).__name__}"
            )
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, RewriteResult) for result in self.results
        ):
            raise RewriteGrammarError("Extraction.results must be a tuple of RewriteResult")


def node_count_cost(diagram: Diagram) -> int:
    """The number of nodes in ``diagram``."""
    return len(_require_diagram("node_count_cost: diagram", diagram).nodes)


def default_cost(diagram: Diagram) -> tuple[int, int, int]:
    """``(node count, wire count, count of nodes with a nonzero phase)``."""
    _require_diagram("default_cost: diagram", diagram)
    phased = sum(
        1 for node in diagram.nodes.values() if node.phase is not None and not node.phase.is_zero
    )
    return (len(diagram.nodes), len(diagram.wires), phased)


class _Stop(Exception):
    """Internal signal ending a saturation run with ``reason``."""

    def __init__(self, reason: SaturationStop) -> None:
        super().__init__(reason.value)
        self.reason = reason


class EGraph:
    """A hash-consed set of diagrams under isomorphism, partitioned into e-classes."""

    def __init__(self) -> None:
        """An empty e-graph."""
        self._reps: list[Diagram] = []
        self._views: list[Diagram] = []
        self._parent: list[int] = []
        self._buckets: dict[str, list[int]] = {}
        self._edges: list[EGraphEdge] = []
        self._tree_edge: list[int | None] = []
        self._roots: list[ENodeId] = []
        self._expanded: list[tuple[tuple[Rule, ...], int] | None] = []

    def __len__(self) -> int:
        """The number of e-nodes."""
        return len(self._reps)

    @property
    def edges(self) -> tuple[EGraphEdge, ...]:
        """Every stored edge, in insertion order."""
        return tuple(self._detached_edge(position) for position in range(len(self._edges)))

    @property
    def roots(self) -> tuple[ENodeId, ...]:
        """Every e-node :meth:`add` created, in creation order."""
        return tuple(self._roots)

    def _require_id(self, what: str, value: object) -> int:
        """``value`` when it names an e-node, else a RewriteGrammarError naming it."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise RewriteGrammarError(f"{what} must be an int id, got {type(value).__name__}")
        if not 0 <= value < len(self._reps):
            raise RewriteGrammarError(f"{what}: unknown id {value}")
        return value

    def _detached_edge(self, position: int) -> EGraphEdge:
        """Stored edge ``position`` with a copy of its result diagram."""
        edge = self._edges[position]
        return replace(edge, result=_detached(edge.result))

    def _pending(self, index: int, rules: tuple[Rule, ...], bound: int) -> bool:
        """Whether e-node ``index`` still needs expanding under ``rules`` and ``bound``."""
        marker = self._expanded[index]
        return marker is None or not _same_rules(marker[0], rules) or bound > marker[1]

    def _root(self, index: int) -> int:
        """The union-find root of ``index``, compressing the path."""
        root = index
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[index] != root:
            self._parent[index], index = root, self._parent[index]
        return root

    def _find_view(self, view: Diagram) -> tuple[str, int | None]:
        """``view``'s key and the e-node whose view is isomorphic to it, if any."""
        key = canonical_key(view)
        for index in self._buckets.get(key, ()):
            if isomorphic(view, self._views[index]):
                return key, index
        return key, None

    def _insert(self, diagram: Diagram, view: Diagram, key: str, eclass: int | None) -> int:
        """A new e-node for ``diagram``, in ``eclass`` or in a class of its own."""
        index = len(self._reps)
        self._reps.append(diagram)
        self._views.append(view)
        self._parent.append(index if eclass is None else eclass)
        self._buckets.setdefault(key, []).append(index)
        self._tree_edge.append(None)
        self._expanded.append(None)
        return index

    def add(self, diagram: Diagram) -> ENodeId:
        """A root e-node holding a copy of ``diagram``, or the isomorphic e-node already held."""
        _require_diagram("EGraph.add: diagram", diagram)
        view = comparison_view(diagram)
        key, found = self._find_view(view)
        if found is not None:
            return ENodeId(found)
        index = ENodeId(self._insert(diagram.copy(), view, key, None))
        self._roots.append(index)
        return index

    def lookup(self, diagram: Diagram) -> ENodeId | None:
        """The e-node isomorphic to ``diagram`` under the comparison view, or ``None``."""
        _require_diagram("EGraph.lookup: diagram", diagram)
        found = self._find_view(comparison_view(diagram))[1]
        return None if found is None else ENodeId(found)

    def find(self, eclass_or_enode: int) -> EClassId:
        """The class id of an e-node or of any member id of a class."""
        return EClassId(self._root(self._require_id("EGraph.find", eclass_or_enode)))

    def class_of(self, enode: ENodeId) -> EClassId:
        """The class id of ``enode``."""
        return EClassId(self._root(self._require_id("EGraph.class_of", enode)))

    def equivalent(self, a: ENodeId, b: ENodeId) -> bool:
        """Whether ``a`` and ``b`` share a class."""
        first = self._require_id("EGraph.equivalent: a", a)
        second = self._require_id("EGraph.equivalent: b", b)
        return self._root(first) == self._root(second)

    def members(self, eclass: EClassId) -> tuple[ENodeId, ...]:
        """Every e-node in the class ``eclass`` names, ascending."""
        root = self._root(self._require_id("EGraph.members", eclass))
        return tuple(ENodeId(i) for i in range(len(self._reps)) if self._root(i) == root)

    def classes(self) -> tuple[EClassId, ...]:
        """Every class id, ascending."""
        return tuple(EClassId(i) for i in range(len(self._reps)) if self._root(i) == i)

    def diagram(self, enode: ENodeId) -> Diagram:
        """A copy of ``enode``'s representative."""
        return self._reps[self._require_id("EGraph.diagram", enode)].copy()

    def path_from_root(self, enode: ENodeId) -> tuple[ENodeId, tuple[RewriteResult, ...]]:
        """The root ``enode`` descends from by ``TREE`` edges, and their results in order."""
        index = self._require_id("EGraph.path_from_root", enode)
        results: list[RewriteResult] = []
        edge_index = self._tree_edge[index]
        while edge_index is not None:
            edge = self._edges[edge_index]
            results.append(_detached(edge.result))
            index = edge.parent
            edge_index = self._tree_edge[index]
        return ENodeId(index), tuple(reversed(results))

    def explain(self, a: ENodeId, b: ENodeId) -> tuple[EGraphEdge, ...] | None:
        """A shortest path of stored edges between ``a`` and ``b``, read undirected; ``None``
        when they are not equivalent."""
        start = self._require_id("EGraph.explain: a", a)
        goal = self._require_id("EGraph.explain: b", b)
        if start == goal:
            return ()
        if self._root(start) != self._root(goal):
            return None
        adjacent: dict[int, list[tuple[int, int]]] = {}
        for position, edge in enumerate(self._edges):
            adjacent.setdefault(edge.parent, []).append((position, edge.child))
            adjacent.setdefault(edge.child, []).append((position, edge.parent))
        via: dict[int, tuple[int, int]] = {start: (-1, -1)}
        queue = deque([start])
        while queue and goal not in via:
            current = queue.popleft()
            for position, neighbour in adjacent.get(current, ()):
                if neighbour not in via:
                    via[neighbour] = (position, current)
                    queue.append(neighbour)
        if goal not in via:
            return None
        path: list[EGraphEdge] = []
        current = goal
        while current != start:
            position, current = via[current]
            path.append(self._detached_edge(position))
        return tuple(reversed(path))

    def _matches(self, rule: Rule, rep: Diagram, cache: RewriteCache | None) -> tuple[Match, ...]:
        """``rule``'s matches in ``rep``, through ``cache`` when given."""
        if cache is None:
            return tuple(rule.pattern.find_matches(rep))
        try:
            return cache.matches(rule.pattern, rep)
        except CacheError as exc:
            raise RewriteGrammarError(
                f"rule {rule.name!r}: the cache rejected its pattern: {type(exc).__name__}: {exc}"
            ) from exc

    def saturate(
        self,
        rules: Sequence[Rule] | None = None,
        *,
        limits: SaturationLimits = DEFAULT_LIMITS,
        cache: RewriteCache | None = None,
    ) -> SaturationReport:
        """Expand every e-node not yet expanded under ``rules`` (default
        :func:`saturation_rules`), round by round, until nothing is left or a limit stops it."""
        rule_set = _require_rules("EGraph.saturate", rules)
        if not isinstance(limits, SaturationLimits):
            raise RewriteGrammarError(
                f"EGraph.saturate: limits must be a SaturationLimits, got {type(limits).__name__}"
            )
        if cache is not None and not isinstance(cache, RewriteCache):
            raise RewriteGrammarError(
                f"EGraph.saturate: cache must be a RewriteCache, got {type(cache).__name__}"
            )
        bound = max((len(self._reps[r].nodes) for r in self._roots), default=0)
        bound += limits.node_margin
        counts = {"enodes": 0, "merges": 0, "applications": 0, "redundant": 0, "pruned": 0}
        failed: list[FailedApplication] = []
        iterations = 0
        stop = SaturationStop.SATURATED
        try:
            while True:
                pending = [i for i in range(len(self._reps)) if self._pending(i, rule_set, bound)]
                if not pending:
                    break
                if iterations >= limits.max_iterations:
                    raise _Stop(SaturationStop.ITERATION_LIMIT)
                iterations += 1
                for index in pending:
                    self._expand(index, rule_set, limits, bound, cache, counts, failed)
                    self._expanded[index] = (rule_set, bound)
        except _Stop as signal:
            stop = signal.reason
        return SaturationReport(
            stop_reason=stop,
            iterations=iterations,
            enodes_added=counts["enodes"],
            merges=counts["merges"],
            applications=counts["applications"],
            redundant=counts["redundant"],
            pruned=counts["pruned"],
            failed=tuple(failed),
        )

    def _expand(
        self,
        index: int,
        rules: tuple[Rule, ...],
        limits: SaturationLimits,
        bound: int,
        cache: RewriteCache | None,
        counts: dict[str, int],
        failed: list[FailedApplication],
    ) -> None:
        """Apply every rule at every passing match of e-node ``index`` and insert the results."""
        rep = self._reps[index]
        for rule in rules:
            for match in self._matches(rule, rep, cache):
                if not match.all_side_conditions_passed:
                    continue
                if counts["applications"] >= limits.max_applications:
                    raise _Stop(SaturationStop.APPLICATION_LIMIT)
                counts["applications"] += 1
                try:
                    result = apply(rep, rule, match)
                except _APPLICATION_ERRORS as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    failed.append(FailedApplication(ENodeId(index), rule.name, message))
                    continue
                self._record(index, rule.name, result, limits, bound, counts)

    def _record(
        self,
        index: int,
        rule_name: str,
        result: RewriteResult,
        limits: SaturationLimits,
        bound: int,
        counts: dict[str, int],
    ) -> None:
        """Insert ``result`` from e-node ``index`` as pruned, redundant, a merge or a new e-node."""
        if len(result.diagram.nodes) > bound:
            counts["pruned"] += 1
            return
        view = comparison_view(result.diagram)
        key, found = self._find_view(view)
        parent = ENodeId(index)
        if found is None:
            if len(self._reps) >= limits.max_enodes:
                raise _Stop(SaturationStop.ENODE_LIMIT)
            child = self._insert(result.diagram, view, key, self._root(index))
            self._tree_edge[child] = len(self._edges)
            self._edges.append(EGraphEdge(parent, ENodeId(child), rule_name, result, EdgeKind.TREE))
            counts["enodes"] += 1
            return
        here, there = self._root(index), self._root(found)
        if here == there:
            counts["redundant"] += 1
            return
        self._edges.append(EGraphEdge(parent, ENodeId(found), rule_name, result, EdgeKind.MERGE))
        self._parent[max(here, there)] = min(here, there)
        counts["merges"] += 1

    def extract(self, eclass: EClassId, cost: CostFunction = default_cost) -> Extraction:
        """The member of ``eclass`` minimizing ``cost``, the lowest e-node id winning a tie."""
        root = self._root(self._require_id("EGraph.extract", eclass))
        if not callable(cost):
            raise RewriteGrammarError(
                f"EGraph.extract: cost must be callable, got {type(cost).__name__}"
            )
        best: tuple[CostValue, int] | None = None
        for member in self.members(EClassId(root)):
            value = cost(self._reps[member].copy())
            if not _is_cost(value):
                raise RewriteGrammarError(
                    "EGraph.extract: cost must return an int, a non-NaN float or a tuple of "
                    f"them, got {value!r}"
                )
            try:
                better = best is None or value < best[0]  # type: ignore[operator]
            except TypeError as exc:
                raise RewriteGrammarError(
                    f"EGraph.extract: cost values are not mutually comparable: {exc}"
                ) from exc
            if better:
                best = (value, member)
        assert best is not None
        value, enode = best
        origin, results = self.path_from_root(ENodeId(enode))
        return Extraction(
            eclass=EClassId(root),
            enode=ENodeId(enode),
            diagram=self._reps[enode].copy(),
            cost=value,
            root=origin,
            results=results,
        )


def simplify(
    diagram: Diagram,
    rules: Sequence[Rule] | None = None,
    *,
    cost: CostFunction = default_cost,
    limits: SaturationLimits = DEFAULT_LIMITS,
    cache: RewriteCache | None = None,
) -> tuple[Extraction, SaturationReport]:
    """Saturate a fresh e-graph holding ``diagram`` and extract its class under ``cost``."""
    _require_diagram("simplify: diagram", diagram)
    if not callable(cost):
        raise RewriteGrammarError(f"simplify: cost must be callable, got {type(cost).__name__}")
    graph = EGraph()
    enode = graph.add(diagram)
    report = graph.saturate(rules, limits=limits, cache=cache)
    return graph.extract(graph.class_of(enode), cost), report
