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

"""Memoization of matches and denotations, with incremental re-matching after local edits.

Contents. :func:`node_digest` and :func:`incidence_digest` hash one node's id-independent data
and its wire ends; :func:`fingerprint` folds those into a :class:`DiagramFingerprint` carrying a
per-node entry plus exact global keys. :class:`LruMemo` is the bounded store under
:class:`MatchCache` (pattern matches and ``canonical_key``) and :class:`ValueMemo` (the generic
content-keyed memo the semantics layer instantiates with its own ``compute``).

Every digest is sha256 over ``repr``, so keys agree across processes. No ordering here comes from
set iteration: node and box ids are sorted, ports and wires use ``PortRef.sort_key`` and
``Wire.sort_key``, and enums enter only through ``.value`` or ``.name``.

This module never imports :mod:`archytaszx.semantics`; denotation memoization is generic.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar, cast

import numpy as np

from archytaszx.diagram.bangbox import BangBox
from archytaszx.diagram.compare import canonical_key
from archytaszx.diagram.graph import BangBoxId, Diagram, Node, NodeId, PortRef, Wire
from archytaszx.rewrite.rule import Match, Pattern


class CacheError(Exception):
    """Base class for every error raised in :mod:`archytaszx.rewrite.cache`."""


class CacheGrammarError(CacheError):
    """A cache argument of the wrong type or shape."""


class CacheDomainError(CacheError):
    """A well-formed cache argument naming something the diagram does not contain."""


_IDENTITY_REPR = " object at 0x"
"""The substring marking a default, identity-dependent ``repr``."""


def _digest(payload: object) -> str:
    """A hash of ``payload``'s ``repr``, stable across processes."""
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:32]


def _require_diagram(diagram: object, what: str) -> Diagram:
    """Return ``diagram`` unchanged, or raise :class:`CacheGrammarError`."""
    if not isinstance(diagram, Diagram):
        raise CacheGrammarError(f"{what} requires a Diagram, got {type(diagram).__name__}")
    return diagram


def _box_chain(diagram: Diagram, box: BangBox) -> tuple[str, ...]:
    """The multiplicity reprs of ``box``'s ancestors, outermost last, cut at the first repeat."""
    chain: list[str] = []
    seen: set[BangBoxId] = {box.id}
    parent = box.parent
    while parent is not None and parent in diagram.bang_boxes and parent not in seen:
        seen.add(parent)
        ancestor = diagram.bang_boxes[parent]
        chain.append(repr(ancestor.multiplicity))
        parent = ancestor.parent
    return tuple(chain)


def _boundary_positions(diagram: Diagram) -> dict[NodeId, list[tuple[str, str, int, int]]]:
    """Per node, the ``(side, direction, port index, boundary position)`` slots it occupies."""
    positions: dict[NodeId, list[tuple[str, str, int, int]]] = {nid: [] for nid in diagram.nodes}
    for side, refs in (("in", diagram.boundary_inputs), ("out", diagram.boundary_outputs)):
        for position, ref in enumerate(refs):
            if ref.node_id in positions:
                positions[ref.node_id].append((side, ref.direction.value, ref.index, position))
    return positions


def _membership(diagram: Diagram) -> dict[NodeId, list[tuple[str, ...]]]:
    """Per node, its bang-box membership entries with each box's multiplicity chain."""
    membership: dict[NodeId, list[tuple[str, ...]]] = {nid: [] for nid in diagram.nodes}
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        static = (repr(box.multiplicity), *_box_chain(diagram, box))
        for nid in sorted(box.node_scope):
            if nid in membership:
                membership[nid].append(("node", *static))
        for ref in sorted(box.port_scope, key=PortRef.sort_key):
            if ref.node_id in membership:
                membership[ref.node_id].append(
                    ("port", ref.direction.value, str(ref.index), *static)
                )
    return membership


def _incidence(diagram: Diagram) -> dict[NodeId, list[tuple[str, int, int, str, int]]]:
    """Per node, its wire ends as ``(own direction, own index, other node, other direction,
    other index)``, in ``Wire.sort_key`` order."""
    incidence: dict[NodeId, list[tuple[str, int, int, str, int]]] = {
        nid: [] for nid in diagram.nodes
    }
    for wire in sorted(diagram.wires, key=Wire.sort_key):
        for near, far in ((wire.a, wire.b), (wire.b, wire.a)):
            if near.node_id in incidence:
                incidence[near.node_id].append(
                    (
                        near.direction.value,
                        near.index,
                        int(far.node_id),
                        far.direction.value,
                        far.index,
                    )
                )
    return incidence


def _node_payload(
    node: Node,
    boundary: list[tuple[str, str, int, int]],
    membership: list[tuple[str, ...]],
) -> tuple[object, ...]:
    """The id-independent data of one node, as a sorted, repr-only tuple."""
    return (
        node.generator_type.name,
        tuple(repr(port.dim) for port in node.inputs),
        tuple(repr(port.dim) for port in node.outputs),
        repr(node.phase),
        tuple(sorted(boundary)),
        tuple(sorted(membership)),
    )


def node_digest(diagram: Diagram, node_id: NodeId) -> str:
    """Digest of a node's generator type, port dims, phase, boundary slots and bang-box chain."""
    diagram = _require_diagram(diagram, "node_digest")
    if node_id not in diagram.nodes:
        raise CacheDomainError(f"node_digest: no node {node_id!r} in diagram")
    return _digest(
        _node_payload(
            diagram.nodes[node_id],
            _boundary_positions(diagram)[node_id],
            _membership(diagram)[node_id],
        )
    )


def incidence_digest(diagram: Diagram, node_id: NodeId) -> str:
    """Digest of a node's sorted wire ends, each naming both its own and the far port."""
    diagram = _require_diagram(diagram, "incidence_digest")
    if node_id not in diagram.nodes:
        raise CacheDomainError(f"incidence_digest: no node {node_id!r} in diagram")
    return _digest(tuple(sorted(_incidence(diagram)[node_id])))


def _box_descriptors(diagram: Diagram) -> tuple[tuple[object, ...], ...]:
    """Every bang box as a sorted, id-exact descriptor tuple."""
    descriptors: list[tuple[object, ...]] = []
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        parent = box.parent
        descriptors.append(
            (
                int(box_id),
                repr(box.multiplicity),
                _box_chain(diagram, box),
                tuple(sorted(int(nid) for nid in box.node_scope)),
                tuple(ref.sort_key() for ref in sorted(box.port_scope, key=PortRef.sort_key)),
                int(parent) if parent is not None else None,
            )
        )
    return tuple(descriptors)


@dataclass(frozen=True, slots=True)
class DiagramFingerprint:
    """A diagram's exact, id-dependent content keys: one entry per node plus global keys."""

    nodes: Mapping[NodeId, str]
    global_key: str
    boundary_key: str
    env_key: str
    boxes_key: str
    match_key: str
    """:attr:`global_key` with the scalar left out."""
    params_key: str
    """The parameter environment alone, without the scalar."""
    dangling_boundary_key: str
    """The boundary entries naming no live node, which no node entry records."""

    def changed_nodes(self, other: DiagramFingerprint) -> frozenset[NodeId]:
        """Node ids present in exactly one fingerprint, plus those whose entry differs."""
        if not isinstance(other, DiagramFingerprint):
            raise CacheGrammarError(
                f"changed_nodes requires a DiagramFingerprint, got {type(other).__name__}"
            )
        mine = set(self.nodes)
        theirs = set(other.nodes)
        changed = mine ^ theirs
        changed |= {nid for nid in mine & theirs if self.nodes[nid] != other.nodes[nid]}
        return frozenset(changed)

    def global_change(self, other: DiagramFingerprint) -> bool:
        """True when the boundary, parameter environment or bang-box key differs."""
        if not isinstance(other, DiagramFingerprint):
            raise CacheGrammarError(
                f"global_change requires a DiagramFingerprint, got {type(other).__name__}"
            )
        return (
            self.boundary_key != other.boundary_key
            or self.env_key != other.env_key
            or self.boxes_key != other.boxes_key
        )

    def match_change(self, other: DiagramFingerprint) -> bool:
        """True when the parameter environment, a bang box or a dangling boundary entry differs."""
        if not isinstance(other, DiagramFingerprint):
            raise CacheGrammarError(
                f"match_change requires a DiagramFingerprint, got {type(other).__name__}"
            )
        return (
            self.params_key != other.params_key
            or self.boxes_key != other.boxes_key
            or self.dangling_boundary_key != other.dangling_boundary_key
        )


def _dangling_boundary(diagram: Diagram) -> tuple[tuple[str, int, tuple[int, str, int]], ...]:
    """Every boundary entry naming a node the diagram does not hold, with its side and position."""
    dangling: list[tuple[str, int, tuple[int, str, int]]] = []
    for side, refs in (("in", diagram.boundary_inputs), ("out", diagram.boundary_outputs)):
        for position, ref in enumerate(refs):
            if ref.node_id not in diagram.nodes:
                dangling.append((side, position, ref.sort_key()))
    return tuple(dangling)


def fingerprint(diagram: Diagram) -> DiagramFingerprint:
    """Fingerprint ``diagram`` in one pass over its nodes, wires and bang boxes."""
    diagram = _require_diagram(diagram, "fingerprint")
    boundary = _boundary_positions(diagram)
    membership = _membership(diagram)
    incidence = _incidence(diagram)

    entries: dict[NodeId, str] = {}
    for nid in sorted(diagram.nodes):
        entries[nid] = _digest(
            (
                _digest(_node_payload(diagram.nodes[nid], boundary[nid], membership[nid])),
                _digest(tuple(sorted(incidence[nid]))),
            )
        )

    boundary_key = _digest(
        (
            tuple(ref.sort_key() for ref in diagram.boundary_inputs),
            tuple(ref.sort_key() for ref in diagram.boundary_outputs),
        )
    )
    params_key = _digest(tuple(sorted(diagram.parameters.items())))
    env_key = _digest((repr(diagram.scalar), tuple(sorted(diagram.parameters.items()))))
    boxes_key = _digest(_box_descriptors(diagram))
    dangling_boundary_key = _digest(_dangling_boundary(diagram))
    body = (
        boundary_key,
        boxes_key,
        tuple((int(nid), entries[nid]) for nid in sorted(entries)),
        tuple(wire.sort_key() for wire in sorted(diagram.wires, key=Wire.sort_key)),
    )
    return DiagramFingerprint(
        nodes=MappingProxyType(entries),
        global_key=_digest((env_key, *body)),
        boundary_key=boundary_key,
        env_key=env_key,
        boxes_key=boxes_key,
        match_key=_digest((params_key, *body)),
        params_key=params_key,
        dangling_boundary_key=dangling_boundary_key,
    )


_fingerprint = fingerprint
"""Module-level alias, reachable where a parameter named ``fingerprint`` shadows the function."""


@dataclass(frozen=True, slots=True)
class CacheStats:
    """A cache's counters at one instant."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    unkeyable: int = 0
    """Lookups :func:`pattern_key` rejected, served by an uncached scan."""

    @property
    def total(self) -> int:
        """Lookups served, hits plus misses."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hits as a fraction of lookups; 0.0 with no lookups."""
        return self.hits / self.total if self.total else 0.0


class _Counters:
    """The mutable box behind a read-only :class:`CacheStats`."""

    __slots__ = ("evictions", "hits", "invalidations", "misses", "unkeyable")

    def __init__(self) -> None:
        """Start every counter at zero."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.unkeyable = 0

    def snapshot(self) -> CacheStats:
        """The counters as a frozen :class:`CacheStats`."""
        return CacheStats(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            invalidations=self.invalidations,
            unkeyable=self.unkeyable,
        )


def _require_max_entries(max_entries: object) -> int:
    """Return ``max_entries`` as a positive int, or raise :class:`CacheGrammarError`."""
    if isinstance(max_entries, bool) or not isinstance(max_entries, int):
        raise CacheGrammarError(f"max_entries must be an int, got {max_entries!r}")
    if max_entries < 1:
        raise CacheGrammarError(f"max_entries must be >= 1, got {max_entries}")
    return max_entries


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class LruMemo(Generic[K, V]):
    """A bounded memo over an ``OrderedDict``, evicting the least recently used entry."""

    __slots__ = ("_counters", "_entries", "_max_entries")

    def __init__(self, max_entries: int, *, counters: _Counters | None = None) -> None:
        """Build an empty memo holding at most ``max_entries`` entries."""
        self._max_entries = _require_max_entries(max_entries)
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._counters = counters if counters is not None else _Counters()

    @property
    def max_entries(self) -> int:
        """The entry ceiling."""
        return self._max_entries

    @property
    def stats(self) -> CacheStats:
        """The counters at this instant."""
        return self._counters.snapshot()

    def __len__(self) -> int:
        """How many entries are held."""
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` is held, without touching recency or counters."""
        return key in self._entries

    def keys(self) -> tuple[K, ...]:
        """Every held key, least recently used first."""
        return tuple(self._entries)

    def get_or_compute(self, key: K, compute: Callable[[], V]) -> V:
        """Return the stored value for ``key``, else store and return ``compute()``."""
        if not callable(compute):
            raise CacheGrammarError(f"compute must be callable, got {type(compute).__name__}")
        if key in self._entries:
            self._entries.move_to_end(key)
            self._counters.hits += 1
            return self._entries[key]
        self._counters.misses += 1
        value = compute()
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._counters.evictions += 1
        return value

    def invalidate(self, key: K) -> bool:
        """Drop ``key``'s entry, reporting whether one was held."""
        if key in self._entries:
            del self._entries[key]
            self._counters.invalidations += 1
            return True
        return False

    def invalidate_where(self, predicate: Callable[[K], bool]) -> int:
        """Drop every entry whose key satisfies ``predicate``, returning how many went."""
        if not callable(predicate):
            raise CacheGrammarError(f"predicate must be callable, got {type(predicate).__name__}")
        doomed = [key for key in self._entries if predicate(key)]
        for key in doomed:
            del self._entries[key]
            self._counters.invalidations += 1
        return len(doomed)

    def clear(self) -> int:
        """Drop every entry, returning how many went."""
        dropped = len(self._entries)
        self._entries.clear()
        self._counters.invalidations += dropped
        return dropped


def _slot_names(klass: type) -> tuple[str, ...]:
    """One class's declared slot names, with a bare-string ``__slots__`` read as one name."""
    declared = klass.__dict__.get("__slots__", ())
    if isinstance(declared, str):
        names: tuple[str, ...] = (declared,)
    elif isinstance(declared, Iterable):
        names = tuple(str(name) for name in declared)
    else:
        names = ()
    return tuple(name for name in names if not (name.startswith("__") and name.endswith("__")))


def _mutable_class_attributes(klass: type) -> tuple[str, ...]:
    """The names a class in ``klass``'s MRO binds to a ``list``, ``dict``, ``set`` or
    ``bytearray``, excluding the dunders."""
    found: set[str] = set()
    for ancestor in klass.__mro__:
        if ancestor in (object, Pattern):
            continue
        for name, value in vars(ancestor).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, list | dict | set | bytearray):
                found.add(name)
    return tuple(sorted(found))


def _instance_state(pattern: Pattern) -> tuple[str, ...]:
    """The names of every attribute ``pattern``'s value depends on: its ``__dict__`` keys, every
    filled slot across the MRO, and every mutable class attribute."""
    held = set(getattr(pattern, "__dict__", {}))
    for klass in type(pattern).__mro__:
        held.update(name for name in _slot_names(klass) if hasattr(pattern, name))
    held.update(_mutable_class_attributes(type(pattern)))
    return tuple(sorted(held))


def pattern_key(pattern: Pattern) -> str:
    """A pattern's cache key: its type's dotted name and its ``repr``.

    A pattern carrying per-instance state is a frozen dataclass, so its ``repr`` is
    value-stable. A stateless pattern keys on its type's dotted name alone; one that holds state
    behind an identity ``repr`` is rejected with :class:`CacheGrammarError`.
    """
    if not isinstance(pattern, Pattern):
        raise CacheGrammarError(f"pattern_key requires a Pattern, got {type(pattern).__name__}")
    text = repr(pattern)
    dotted = f"{type(pattern).__module__}.{type(pattern).__qualname__}"
    if _IDENTITY_REPR in text:
        if _instance_state(pattern):
            raise CacheGrammarError(
                f"pattern_key requires a value-stable repr, got an identity repr: {text}"
            )
        return f"{dotted}|"
    return f"{dotted}|{text}"


def keyable(pattern: Pattern) -> bool:
    """Whether :func:`pattern_key` accepts ``pattern``."""
    try:
        pattern_key(pattern)
    except CacheGrammarError:
        return False
    return True


def _require_fingerprint(
    given: DiagramFingerprint | None, diagram: Diagram, what: str
) -> DiagramFingerprint:
    """Return ``given``, or fingerprint ``diagram`` when it is ``None``."""
    if given is None:
        return _fingerprint(diagram)
    if not isinstance(given, DiagramFingerprint):
        raise CacheGrammarError(
            f"{what}: fingerprint must be a DiagramFingerprint or None, got {type(given).__name__}"
        )
    return given


def _require_anchors(anchors: object) -> frozenset[NodeId]:
    """Return ``anchors`` as a ``frozenset`` of node ids, or raise :class:`CacheGrammarError`."""
    if not isinstance(anchors, frozenset | set):
        raise CacheGrammarError(
            f"anchors must be a frozenset of NodeId, got {type(anchors).__name__}"
        )
    offenders = sorted(
        repr(nid) for nid in anchors if isinstance(nid, bool) or not isinstance(nid, int)
    )
    if offenders:
        raise CacheGrammarError(f"anchors entries must be node ids, got {', '.join(offenders)}")
    return frozenset(cast("frozenset[NodeId]", anchors))


class MatchCache:
    """A memo for pattern matches, keyed on a diagram's match key, and for ``canonical_key``,
    keyed on its global key."""

    __slots__ = ("_canonical", "_counters", "_matches")

    def __init__(self, max_entries: int = 256) -> None:
        """Build a match cache holding at most ``max_entries`` entries per memo."""
        limit = _require_max_entries(max_entries)
        self._counters = _Counters()
        self._matches: LruMemo[tuple[str, str], tuple[Match, ...]] = LruMemo(
            limit, counters=self._counters
        )
        self._canonical: LruMemo[str, str] = LruMemo(limit, counters=self._counters)

    @property
    def stats(self) -> CacheStats:
        """The counters of both memos combined."""
        return self._counters.snapshot()

    def matches(
        self,
        pattern: Pattern,
        diagram: Diagram,
        *,
        fingerprint: DiagramFingerprint | None = None,
    ) -> tuple[Match, ...]:
        """Every match of ``pattern`` in ``diagram``, memoized on the pattern and match key.

        A pattern :func:`pattern_key` cannot key is scanned uncached and counted in
        :attr:`CacheStats.unkeyable`.
        """
        diagram = _require_diagram(diagram, "MatchCache.matches")
        fp = _require_fingerprint(fingerprint, diagram, "MatchCache.matches")
        if not keyable(pattern):
            self._counters.unkeyable += 1
            return tuple(pattern.find_matches(diagram))
        key = pattern_key(pattern)
        return self._matches.get_or_compute(
            (key, fp.match_key), lambda: tuple(pattern.find_matches(diagram))
        )

    def matches_anchored(
        self,
        pattern: Pattern,
        diagram: Diagram,
        anchors: frozenset[NodeId],
        *,
        fingerprint: DiagramFingerprint | None = None,
    ) -> tuple[Match, ...]:
        """Every match of ``pattern`` whose support meets ``anchors``, computed without caching."""
        diagram = _require_diagram(diagram, "MatchCache.matches_anchored")
        if not keyable(pattern):
            self._counters.unkeyable += 1
        if fingerprint is not None and not isinstance(fingerprint, DiagramFingerprint):
            raise CacheGrammarError(
                "MatchCache.matches_anchored: fingerprint must be a DiagramFingerprint or None, "
                f"got {type(fingerprint).__name__}"
            )
        return tuple(pattern.find_matches_anchored(diagram, _require_anchors(anchors)))

    def canonical(self, diagram: Diagram, *, fingerprint: DiagramFingerprint | None = None) -> str:
        """``canonical_key(diagram)``, memoized on the diagram's global key."""
        diagram = _require_diagram(diagram, "MatchCache.canonical")
        fp = _require_fingerprint(fingerprint, diagram, "MatchCache.canonical")
        return self._canonical.get_or_compute(fp.global_key, lambda: canonical_key(diagram))

    def invalidate(self, fingerprint: DiagramFingerprint) -> int:
        """Drop every entry keyed on ``fingerprint``'s global key, returning how many went."""
        if not isinstance(fingerprint, DiagramFingerprint):
            raise CacheGrammarError(
                "MatchCache.invalidate requires a DiagramFingerprint, got "
                f"{type(fingerprint).__name__}"
            )
        match_key = fingerprint.match_key
        global_key = fingerprint.global_key
        dropped = self._matches.invalidate_where(lambda key: key[1] == match_key)
        dropped += self._canonical.invalidate_where(lambda key: key == global_key)
        return dropped

    def clear(self) -> int:
        """Drop every entry of both memos, returning how many went."""
        return self._matches.clear() + self._canonical.clear()


def node_key(node: Node) -> str:
    """A node's content key: generator type, port dim reprs and phase repr, with no ids."""
    if not isinstance(node, Node):
        raise CacheGrammarError(f"node_key requires a Node, got {type(node).__name__}")
    return _digest(
        (
            node.generator_type.name,
            tuple(repr(port.dim) for port in node.inputs),
            tuple(repr(port.dim) for port in node.outputs),
            repr(node.phase),
        )
    )


def diagram_key(diagram: Diagram) -> str:
    """A diagram's content key: ``fingerprint(diagram).global_key``."""
    return _fingerprint(_require_diagram(diagram, "diagram_key")).global_key


def _store(value: V) -> V:
    """A numpy value as a private copy, any other value unchanged."""
    if isinstance(value, np.ndarray):
        return cast("V", value.copy())
    return value


def _expose(value: V) -> V:
    """A numpy value as a read-only view, any other value unchanged."""
    if isinstance(value, np.ndarray):
        view = value.view()
        view.setflags(write=False)
        return cast("V", view)
    return value


class ValueMemo(Generic[T, V]):
    """A content-keyed memo over ``compute``, keyed by ``key``: numpy values are stored as a
    private copy and handed out as a read-only view."""

    __slots__ = ("_compute", "_key", "_memo")

    def __init__(
        self,
        compute: Callable[[T], V],
        key: Callable[[T], str],
        max_entries: int = 1024,
    ) -> None:
        """Build an empty memo over ``compute``, keyed by ``key``."""
        if not callable(compute):
            raise CacheGrammarError(f"compute must be callable, got {type(compute).__name__}")
        if not callable(key):
            raise CacheGrammarError(f"key must be callable, got {type(key).__name__}")
        self._compute = compute
        self._key = key
        self._memo: LruMemo[str, V] = LruMemo(max_entries)

    @property
    def stats(self) -> CacheStats:
        """The counters at this instant."""
        return self._memo.stats

    @property
    def max_entries(self) -> int:
        """The entry ceiling."""
        return self._memo.max_entries

    def __len__(self) -> int:
        """How many entries are held."""
        return len(self._memo)

    def key_for(self, subject: T) -> str:
        """``subject``'s memo key, as produced by ``key``."""
        computed = self._key(subject)
        if not isinstance(computed, str):
            raise CacheGrammarError(f"key must return a str, got {type(computed).__name__}")
        return computed

    def get(self, subject: T) -> V:
        """The memoized value of ``compute(subject)``."""
        stored = self._memo.get_or_compute(
            self.key_for(subject), lambda: _store(self._compute(subject))
        )
        return _expose(stored)

    def invalidate(self, subject: T) -> bool:
        """Drop ``subject``'s entry, reporting whether one was held."""
        return self._memo.invalidate(self.key_for(subject))

    def clear(self) -> int:
        """Drop every entry, returning how many went."""
        return self._memo.clear()


def _adjacency(diagram: Diagram) -> dict[NodeId, frozenset[NodeId]]:
    """Per node, the ids of the nodes a wire joins it to."""
    neighbours: dict[NodeId, set[NodeId]] = {nid: set() for nid in diagram.nodes}
    for wire in diagram.wires:
        near, far = wire.a.node_id, wire.b.node_id
        if near in neighbours:
            neighbours[near].add(far)
        if far in neighbours:
            neighbours[far].add(near)
    return {nid: frozenset(ids) for nid, ids in neighbours.items()}


def _expand(
    seed: frozenset[NodeId],
    adjacencies: tuple[Mapping[NodeId, frozenset[NodeId]], ...],
    radius: int,
) -> frozenset[NodeId]:
    """``seed`` grown by ``radius`` hops over the union of ``adjacencies``."""
    reached = set(seed)
    frontier = set(seed)
    for _ in range(max(radius, 0)):
        if not frontier:
            break
        nxt: set[NodeId] = set()
        for nid in frontier:
            for adjacency in adjacencies:
                for other in adjacency.get(nid, frozenset()):
                    if other not in reached:
                        reached.add(other)
                        nxt.add(other)
        frontier = nxt
    return frozenset(reached)


def _box_scopes(diagram: Diagram) -> dict[BangBoxId, tuple[object, ...]]:
    """Per bang box, its descriptor tuple paired with the node ids it scopes."""
    scopes: dict[BangBoxId, tuple[object, ...]] = {}
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        nodes = tuple(sorted(int(nid) for nid in box.node_scope))
        ports = tuple(ref.sort_key() for ref in sorted(box.port_scope, key=PortRef.sort_key))
        parent = box.parent
        scopes[box_id] = (
            repr(box.multiplicity),
            _box_chain(diagram, box),
            nodes,
            ports,
            int(parent) if parent is not None else None,
        )
    return scopes


def _box_members(diagram: Diagram, box_id: BangBoxId) -> frozenset[NodeId]:
    """Every node id one bang box scopes, by node scope or by port scope."""
    box = diagram.bang_boxes[box_id]
    members = set(box.node_scope)
    members.update(ref.node_id for ref in box.port_scope)
    return frozenset(members)


def _support(match: Match) -> tuple[NodeId, ...] | None:
    """A match's non-empty ``support_node_ids`` as a tuple of node ids, or ``None``."""
    support = getattr(match, "support_node_ids", None)
    if not isinstance(support, tuple) or not support:
        return None
    if any(isinstance(nid, bool) or not isinstance(nid, int) for nid in support):
        return None
    return cast("tuple[NodeId, ...]", support)


def _match_identity(match: Match) -> tuple[object, ...] | None:
    """A match's de-duplication identity, its support ids and its ``repr``, or ``None`` when it
    has no usable support or its ``repr`` is identity-dependent."""
    support = _support(match)
    if support is None:
        return None
    text = repr(match)
    if _IDENTITY_REPR in text:
        return None
    return (tuple(int(nid) for nid in support), text)


@dataclass(frozen=True, slots=True)
class IncrementalStats:
    """An :class:`IncrementalMatcher`'s counters: the last rematch's sizes, and run totals."""

    dirty_nodes: int = 0
    closure_nodes: int = 0
    retained: int = 0
    rescanned: int = 0
    full_rescans: int = 0
    rematches: int = 0


class IncrementalMatcher:
    """Re-matches a fixed pattern set after a local edit, rescanning only the affected region.

    :meth:`seed` scans in full and stores the fingerprint, the adjacency and the bang-box scopes;
    :meth:`rematch` keeps every baseline match whose support lies outside the dirty closure and
    unchanged, and re-enumerates the rest through ``Pattern.find_matches_anchored``.
    """

    __slots__ = (
        "_adjacency",
        "_baseline",
        "_boxes",
        "_cache",
        "_keys",
        "_matches",
        "_patterns",
        "_radius",
        "_stats",
    )

    def __init__(self, patterns: Sequence[Pattern], *, cache: MatchCache | None = None) -> None:
        """Build a matcher over ``patterns``, optionally serving full scans from ``cache``."""
        if isinstance(patterns, str) or not isinstance(patterns, Sequence):
            raise CacheGrammarError(
                f"patterns must be a sequence of Pattern, got {type(patterns).__name__}"
            )
        checked: list[Pattern] = []
        for pattern in patterns:
            if not isinstance(pattern, Pattern):
                raise CacheGrammarError(
                    f"patterns entries must be Pattern, got {type(pattern).__name__}"
                )
            pattern_key(pattern)
            checked.append(pattern)
        if cache is not None and not isinstance(cache, MatchCache):
            raise CacheGrammarError(
                f"cache must be a MatchCache or None, got {type(cache).__name__}"
            )
        self._patterns: tuple[Pattern, ...] = tuple(checked)
        self._keys: tuple[str, ...] = tuple(pattern_key(p) for p in checked)
        self._cache = cache
        self._radius = max((p.locality_radius for p in self._patterns), default=0)
        self._baseline: DiagramFingerprint | None = None
        self._adjacency: dict[NodeId, frozenset[NodeId]] = {}
        self._boxes: dict[BangBoxId, tuple[object, ...]] = {}
        self._matches: dict[str, tuple[Match, ...]] = {}
        self._stats = IncrementalStats()

    @property
    def patterns(self) -> tuple[Pattern, ...]:
        """The patterns this matcher tracks, in the order given."""
        return self._patterns

    @property
    def radius(self) -> int:
        """The closure radius: the largest ``locality_radius`` among the patterns."""
        return self._radius

    @property
    def stats(self) -> IncrementalStats:
        """The counters at this instant."""
        return self._stats

    def reset(self) -> None:
        """Drop the baseline, keeping the counters."""
        self._baseline = None
        self._adjacency = {}
        self._boxes = {}
        self._matches = {}

    def clear(self) -> None:
        """Drop the baseline and zero the counters."""
        self.reset()
        self._stats = IncrementalStats()

    def _full_scan(self, diagram: Diagram, fp: DiagramFingerprint) -> dict[str, tuple[Match, ...]]:
        """Every pattern scanned in full, through the match cache when there is one."""
        found: dict[str, tuple[Match, ...]] = {}
        for pattern, key in zip(self._patterns, self._keys, strict=True):
            found[key] = self._scan_one(pattern, diagram, fp)
        return found

    def _adopt(
        self,
        diagram: Diagram,
        fp: DiagramFingerprint,
        found: dict[str, tuple[Match, ...]],
        adjacency: dict[NodeId, frozenset[NodeId]] | None = None,
    ) -> Mapping[str, tuple[Match, ...]]:
        """Store ``found`` as the baseline for ``diagram`` and return it read-only."""
        self._baseline = fp
        self._adjacency = _adjacency(diagram) if adjacency is None else adjacency
        self._boxes = _box_scopes(diagram)
        self._matches = found
        return MappingProxyType(dict(found))

    def seed(
        self, diagram: Diagram, *, fingerprint: DiagramFingerprint | None = None
    ) -> Mapping[str, tuple[Match, ...]]:
        """Scan every pattern over ``diagram`` in full and make the result the baseline."""
        diagram = _require_diagram(diagram, "IncrementalMatcher.seed")
        fp = _require_fingerprint(fingerprint, diagram, "IncrementalMatcher.seed")
        return self._adopt(diagram, fp, self._full_scan(diagram, fp))

    def _dirty(self, diagram: Diagram, fp: DiagramFingerprint) -> frozenset[NodeId]:
        """The changed node ids, plus every node a changed bang box scopes."""
        baseline = self._baseline
        assert baseline is not None
        dirty = set(fp.changed_nodes(baseline))
        current = _box_scopes(diagram)
        for box_id, descriptor in current.items():
            if self._boxes.get(box_id) != descriptor:
                dirty.update(_box_members(diagram, box_id))
        for box_id, descriptor in self._boxes.items():
            if current.get(box_id) != descriptor:
                dirty.update(cast("tuple[NodeId, ...]", descriptor[2]))
        return frozenset(dirty)

    def _keeps(
        self,
        support: tuple[NodeId, ...],
        fp: DiagramFingerprint,
        closure: frozenset[NodeId],
    ) -> bool:
        """True when ``support`` lies outside ``closure`` and every node of it is unchanged."""
        baseline = self._baseline
        assert baseline is not None
        if not closure.isdisjoint(support):
            return False
        return all(nid in fp.nodes and fp.nodes[nid] == baseline.nodes.get(nid) for nid in support)

    def _scan_one(
        self, pattern: Pattern, diagram: Diagram, fp: DiagramFingerprint
    ) -> tuple[Match, ...]:
        """One pattern scanned in full, through the match cache when there is one."""
        if self._cache is not None:
            return self._cache.matches(pattern, diagram, fingerprint=fp)
        return tuple(pattern.find_matches(diagram))

    def _locally_rematched(
        self,
        pattern: Pattern,
        key: str,
        diagram: Diagram,
        fp: DiagramFingerprint,
        closure: frozenset[NodeId],
    ) -> tuple[int, tuple[Match, ...]] | None:
        """One pattern's retained count and merged matches, or ``None`` when the caller must scan
        it in full: a match with no usable ``support_node_ids`` or an identity ``repr``, or any
        foreign exception out of the pattern."""
        baseline: list[tuple[tuple[object, ...], Match]] = []
        for match in self._matches.get(key, ()):
            identity = _match_identity(match)
            if identity is None:
                return None
            baseline.append((identity, match))
        try:
            retained = [
                entry
                for entry in baseline
                if self._keeps(cast("tuple[NodeId, ...]", entry[0][0]), fp, closure)
            ]
            fresh: list[tuple[tuple[object, ...], Match]] = []
            for match in pattern.find_matches_anchored(diagram, closure):
                identity = _match_identity(match)
                if identity is None:
                    return None
                fresh.append((identity, match))
            merged: dict[tuple[object, ...], Match] = {}
            for identity, match in (*retained, *fresh):
                if identity not in merged:
                    merged[identity] = match
            ordered = tuple(
                match
                for _, match in sorted(
                    merged.items(), key=lambda pair: (*pattern.order_key(pair[1]), pair[0][1])
                )
            )
        except CacheError:
            raise
        except Exception:  # noqa: BLE001 - any foreign failure means a full scan
            return None
        return len(retained), ordered

    def rematch(
        self, diagram: Diagram, *, fingerprint: DiagramFingerprint | None = None
    ) -> Mapping[str, tuple[Match, ...]]:
        """Re-match every pattern over the region ``diagram`` changed since the baseline."""
        diagram = _require_diagram(diagram, "IncrementalMatcher.rematch")
        fp = _require_fingerprint(fingerprint, diagram, "IncrementalMatcher.rematch")
        baseline = self._baseline
        totals = self._stats
        if baseline is None or fp.match_change(baseline):
            rescan = self._full_scan(diagram, fp)
            self._stats = IncrementalStats(
                dirty_nodes=len(fp.nodes),
                closure_nodes=len(fp.nodes),
                retained=0,
                rescanned=sum(len(v) for v in rescan.values()),
                full_rescans=totals.full_rescans + 1,
                rematches=totals.rematches + 1,
            )
            return self._adopt(diagram, fp, rescan)

        dirty = self._dirty(diagram, fp)
        adjacency = _adjacency(diagram)
        closure = _expand(dirty, (self._adjacency, adjacency), self._radius)
        found: dict[str, tuple[Match, ...]] = {}
        retained_total = 0
        fresh_total = 0
        for pattern, key in zip(self._patterns, self._keys, strict=True):
            local = self._locally_rematched(pattern, key, diagram, fp, closure)
            if local is None:
                scanned = self._scan_one(pattern, diagram, fp)
                found[key] = scanned
                fresh_total += len(scanned)
                continue
            retained, merged = local
            retained_total += retained
            fresh_total += len(merged) - retained
            found[key] = merged
        self._stats = IncrementalStats(
            dirty_nodes=len(dirty),
            closure_nodes=len(closure),
            retained=retained_total,
            rescanned=fresh_total,
            full_rescans=totals.full_rescans,
            rematches=totals.rematches + 1,
        )
        return self._adopt(diagram, fp, found, adjacency)


@dataclass(frozen=True, slots=True)
class RewriteCache:
    """The one cache object :mod:`archytaszx.rewrite.engine` takes: a match memo and an optional
    :class:`IncrementalMatcher`."""

    match_cache: MatchCache = field(default_factory=MatchCache)
    incremental: IncrementalMatcher | None = None

    def __post_init__(self) -> None:
        """Reject a field of the wrong type."""
        if not isinstance(self.match_cache, MatchCache):
            raise CacheGrammarError(
                f"match_cache must be a MatchCache, got {type(self.match_cache).__name__}"
            )
        if self.incremental is not None and not isinstance(self.incremental, IncrementalMatcher):
            raise CacheGrammarError(
                "incremental must be an IncrementalMatcher or None, got "
                f"{type(self.incremental).__name__}"
            )

    @property
    def stats(self) -> CacheStats:
        """The match memo's counters."""
        return self.match_cache.stats

    def fingerprint(self, diagram: Diagram) -> DiagramFingerprint:
        """``fingerprint(diagram)``."""
        return _fingerprint(_require_diagram(diagram, "RewriteCache.fingerprint"))

    def matches(
        self,
        pattern: Pattern,
        diagram: Diagram,
        *,
        fingerprint: DiagramFingerprint | None = None,
    ) -> tuple[Match, ...]:
        """Every match of ``pattern`` in ``diagram``, through the match memo."""
        return self.match_cache.matches(pattern, diagram, fingerprint=fingerprint)

    def canonical(self, diagram: Diagram, *, fingerprint: DiagramFingerprint | None = None) -> str:
        """``canonical_key(diagram)``, through the match memo."""
        return self.match_cache.canonical(diagram, fingerprint=fingerprint)

    def clear(self) -> int:
        """Drop every memo entry and any incremental baseline, returning how many entries went."""
        if self.incremental is not None:
            self.incremental.reset()
        return self.match_cache.clear()
