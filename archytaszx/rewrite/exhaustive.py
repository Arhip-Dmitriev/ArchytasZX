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

"""Exhaustive search: a fair bidirectional breadth-first enumeration of rule applications.

From ``start`` and from ``goal``, layer by layer, every match of every registered rule
(automatic or not, converses included) is applied to every diagram of the current layer.
A diagram whose comparison view is isomorphic to one already reached on its side is pruned.
After each layer the two sides are checked for a common diagram, up to isomorphism of
comparison views; the first found gives a :class:`~archytaszx.rewrite.tactics.ProofPath`.

Layer ``k`` of the start side is expanded before layer ``k`` of the goal side, and both before
layer ``k + 1`` of either. A derivation of length ``n`` whose steps are rule applications at
matches the patterns enumerate (a free variable at its default value) is therefore found
once the start side reaches layer ``n``, and once it reaches layer ``ceil(n/2)`` and the goal
side layer ``floor(n/2)`` when the last ``floor(n/2)`` steps use rules that have converses
(every rule but the connective ones and port-box unfusion).
There is no depth cap and no time limit: on two diagrams no enumerated derivation joins, the
search does not terminate. A diagram whose construction or canonical key raises (for example a
scalar beyond Python's integer conversion limit) is skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from archytaszx.diagram.graph import Diagram
from archytaszx.rewrite.egraph import APPLICATION_ERRORS
from archytaszx.rewrite.engine import RewriteResult, apply
from archytaszx.rewrite.normal_form import comparison_view, view_key, views_isomorphic
from archytaszx.rewrite.rule import Rule
from archytaszx.rewrite.rules_library import RULES
from archytaszx.rewrite.tactics import ProofPath, ProofStep

_SKIPPED: tuple[type[Exception], ...] = (*APPLICATION_ERRORS, ValueError, ArithmeticError)
"""Errors that skip one candidate diagram."""


@dataclass(frozen=True, slots=True)
class _Reached:
    """A diagram one side reached, its comparison view, and the results leading to it."""

    diagram: Diagram
    view: Diagram
    results: tuple[RewriteResult, ...]


@dataclass(frozen=True, slots=True)
class LayerReport:
    """The size of each side after one layer."""

    side: str
    depth: int
    reached: int
    frontier: int


class _Side:
    """One side of the search: every diagram reached, keyed by view key."""

    def __init__(self, name: str, root: Diagram) -> None:
        self.name = name
        view = comparison_view(root)
        first = _Reached(root, view, ())
        self.seen: dict[str, list[_Reached]] = {view_key(view): [first]}
        self.frontier: list[_Reached] = [first]
        self.depth = 0

    def lookup(self, key: str, view: Diagram) -> _Reached | None:
        for reached in self.seen.get(key, ()):
            if views_isomorphic(reached.view, view):
                return reached
        return None

    def expand(self, rules: Sequence[Rule]) -> list[tuple[str, _Reached]]:
        """Every new diagram one application away from the frontier, in a fixed order."""
        added: list[tuple[str, _Reached]] = []
        nxt: list[_Reached] = []
        for reached in self.frontier:
            for rule in rules:
                try:
                    matches = rule.pattern.find_matches(reached.diagram)
                except _SKIPPED:
                    continue
                for match in matches:
                    try:
                        result = apply(reached.diagram, rule, match)
                        view = comparison_view(result.diagram)
                        key = view_key(view)
                    except _SKIPPED:
                        continue
                    if self.lookup(key, view) is not None:
                        continue
                    item = _Reached(result.diagram, view, (*reached.results, result))
                    self.seen.setdefault(key, []).append(item)
                    nxt.append(item)
                    added.append((key, item))
        self.frontier = nxt
        self.depth += 1
        return added


def _path(start: Diagram, goal: Diagram, left: _Reached, right: _Reached) -> ProofPath:
    forward = (ProofStep("exhaustive", left.results),) if left.results else ()
    backward = (ProofStep("exhaustive", right.results),) if right.results else ()
    return ProofPath(start, goal, forward, backward)


def exhaustive_search(
    start: Diagram,
    goal: Diagram,
    *,
    rules: Sequence[Rule] | None = None,
    on_layer: Callable[[LayerReport], None] | None = None,
) -> ProofPath:
    """A derivation between ``start`` and ``goal`` by fair bidirectional breadth-first search
    over ``rules`` (default: every rule in :data:`RULES`). Returns only on success."""
    rule_set = tuple(RULES.values()) if rules is None else tuple(rules)
    left, right = _Side("start", start), _Side("goal", goal)
    for key, items in left.seen.items():
        for item in items:
            hit = right.lookup(key, item.view)
            if hit is not None:
                return _path(start, goal, item, hit)
    while True:
        for side, other in ((left, right), (right, left)):
            for key, item in side.expand(rule_set):
                hit = other.lookup(key, item.view)
                if hit is not None:
                    found = (item, hit) if side is left else (hit, item)
                    return _path(start, goal, *found)
            if on_layer is not None:
                on_layer(
                    LayerReport(
                        side.name,
                        side.depth,
                        sum(len(v) for v in side.seen.values()),
                        len(side.frontier),
                    )
                )
            if not left.frontier and not right.frontier:
                return _exhausted()


class ExhaustedError(Exception):
    """Both sides reached every diagram they can without meeting."""


def _exhausted() -> ProofPath:
    raise ExhaustedError("both sides are closed under every rule and never meet")


__all__ = ["ExhaustedError", "LayerReport", "exhaustive_search"]
