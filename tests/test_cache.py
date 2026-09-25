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

"""Covers :mod:`archytaszx.rewrite.cache`'s Stage 1 and Stage 2 surface: the node and incidence
digests, :class:`DiagramFingerprint`, :class:`LruMemo`, :class:`CacheStats`, :func:`pattern_key`,
:class:`MatchCache`, :class:`ValueMemo`, the exception hierarchy, and cross-process determinism."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.compare import canonical_key
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from archytaszx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef
from archytaszx.rewrite.cache import (
    CacheDomainError,
    CacheError,
    CacheGrammarError,
    CacheStats,
    DiagramFingerprint,
    LruMemo,
    MatchCache,
    ValueMemo,
    diagram_key,
    fingerprint,
    incidence_digest,
    keyable,
    node_digest,
    node_key,
    pattern_key,
)
from archytaszx.rewrite.match import (
    BialgebraPattern,
    CapPattern,
    FourierCancellationPattern,
    FourierStateColorChangePattern,
    FusionPattern,
    HopfPattern,
    IdentityRemovalPattern,
    StateCopyPattern,
    TriangleInverseCancellationPattern,
)
from archytaszx.rewrite.rule import DimensionGuard, DimensionGuardKind, Match, Pattern
from tests.helpers import build_ghz_with_copy

DIM = Dim.concrete(2)
OTHER_DIM = Dim.concrete(3)
PHASE_HALF = PhaseVector(DIM, {1: Phase.turns(sp.Rational(1, 2))})
PHASE_THIRD = PhaseVector(OTHER_DIM, {1: Phase.turns(sp.Rational(1, 3))})
SYMBOL_MULT = Mult.symbol("n")

ALL_PATTERNS: tuple[Pattern, ...] = (
    FusionPattern(),
    FourierCancellationPattern(),
    CapPattern(),
    IdentityRemovalPattern(),
    TriangleInverseCancellationPattern(),
    StateCopyPattern(),
    HopfPattern(),
    BialgebraPattern(),
    FourierStateColorChangePattern(),
)

Build = Callable[[], tuple[Diagram, NodeId]]


def _pair(
    *,
    padding: int = 0,
    generator_type: GeneratorType = Z_SPIDER,
    in_dim: Dim = DIM,
    out_dim: Dim = DIM,
    phase: PhaseVector | None = None,
    wires: Sequence[tuple[int, int]] = ((0, 0),),
    boundary: Sequence[int] = (0, 1),
) -> tuple[Diagram, NodeId]:
    """A two-output Z state feeding node B, whose id is returned alongside the diagram.

    ``padding`` burns node ids first, ``wires`` lists ``(A output index, B input index)`` pairs,
    and ``boundary`` lists B's output indices in boundary order.
    """
    diagram = Diagram()
    for _ in range(padding):
        spare = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
        diagram.remove_node(spare)
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM, DIM])
    b_id = diagram.add_node(
        generator_type, input_dims=[in_dim, DIM], output_dims=[out_dim, DIM], phase=phase
    )
    for a_index, b_index in wires:
        diagram.add_wire(
            PortRef(a_id, Direction.OUTPUT, a_index), PortRef(b_id, Direction.INPUT, b_index)
        )
    diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, index) for index in boundary])
    return diagram, b_id


def _boxed(
    *,
    multiplicity: Mult = SYMBOL_MULT,
    port_scope: bool = False,
    nested: bool = False,
) -> tuple[Diagram, NodeId]:
    """The ``_pair`` diagram with B inside one bang box, optionally by port scope or nested."""
    diagram, b_id = _pair()
    parent: int | None = None
    if nested:
        parent = diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({b_id}))
    scope = (
        {"port_scope": frozenset({PortRef(b_id, Direction.OUTPUT, 0)})}
        if port_scope
        else {"node_scope": frozenset({b_id})}
    )
    diagram.add_bang_box(multiplicity, parent=parent, **scope)  # type: ignore[arg-type]
    return diagram, b_id


NODE_FIELD_EDITS: tuple[tuple[str, Build], ...] = (
    ("generator type", lambda: _pair(generator_type=X_SPIDER)),
    ("input port dim", lambda: _pair(in_dim=OTHER_DIM)),
    ("output port dim", lambda: _pair(out_dim=OTHER_DIM)),
    ("phase present", lambda: _pair(phase=PHASE_HALF)),
    ("boundary membership", lambda: _pair(boundary=())),
    ("boundary position", lambda: _pair(boundary=(1, 0))),
    ("bang-box node membership", lambda: _boxed()),
    ("bang-box port membership", lambda: _boxed(port_scope=True)),
    ("bang-box nesting", lambda: _boxed(nested=True)),
    ("bang-box multiplicity", lambda: _boxed(multiplicity=Mult.concrete(4))),
)

INCIDENCE_EDITS: tuple[tuple[str, Build], ...] = (
    ("no incident wire", lambda: _pair(wires=())),
    ("a second incident wire", lambda: _pair(wires=((0, 0), (1, 1)))),
    ("the wire moved to another port", lambda: _pair(wires=((0, 1),))),
    ("the wire moved to another far port", lambda: _pair(wires=((1, 0),))),
)


def _scalar_edit() -> Diagram:
    """The ``_pair`` diagram carrying a global scalar factor."""
    diagram, _ = _pair()
    diagram.multiply_scalar(Scalar.rational(3))
    return diagram


def _parameter_edit() -> Diagram:
    """The ``_pair`` diagram with one bound parameter."""
    diagram, _ = _pair()
    diagram.bind_parameter("n", 2)
    return diagram


def _extra_node_edit() -> Diagram:
    """The ``_pair`` diagram with one further, unwired node."""
    diagram, _ = _pair()
    diagram.add_node(X_SPIDER, input_dims=[DIM], output_dims=[])
    return diagram


def _removed_node_edit() -> Diagram:
    """The ``_pair`` diagram with the A spider's wire and node removed."""
    diagram, b_id = _pair()
    a_id = next(nid for nid in diagram.nodes if nid != b_id)
    diagram.remove_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.remove_node(a_id)
    return diagram


GLOBAL_KEY_EDITS: tuple[tuple[str, Callable[[], Diagram]], ...] = (
    *((label, lambda build=build: build()[0]) for label, build in NODE_FIELD_EDITS),
    *((label, lambda build=build: build()[0]) for label, build in INCIDENCE_EDITS),
    ("scalar", _scalar_edit),
    ("parameter environment", _parameter_edit),
    ("an added node", _extra_node_edit),
    ("a removed node", _removed_node_edit),
)


def _corpus() -> tuple[tuple[str, Diagram], ...]:
    """A small corpus of built diagrams: the worked example, a cap, a boxed pair and variants."""
    ghz, _, _ = build_ghz_with_copy(DIM)
    cap = Diagram()
    state = cap.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM])
    effect = cap.add_node(X_SPIDER, input_dims=[DIM], output_dims=[])
    cap.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    return (
        ("ghz_with_copy", ghz),
        ("cap", cap),
        ("pair", _pair()[0]),
        ("boxed_pair", _boxed()[0]),
        ("phased_pair", _pair(phase=PHASE_HALF)[0]),
        ("unwired_pair", _pair(wires=())[0]),
        ("empty", Diagram()),
    )


CORPUS = _corpus()


class _StatelessPattern(Pattern):
    """A pattern with an identity ``repr`` and no per-instance state."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """No matches, ever."""
        return ()


class _StatefulPattern(Pattern):
    """A pattern with an identity ``repr`` holding per-instance state in its ``__dict__``."""

    def __init__(self, threshold: int) -> None:
        """Record ``threshold`` as instance state."""
        self.threshold = threshold

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """No matches, ever."""
        return ()


class _BareStringSlotsPattern(Pattern):
    """A pattern whose ``__slots__`` is a bare string naming one state-holding slot."""

    __slots__ = "threshold"  # noqa: PLC0205 - the bare-string declaration under test

    def __init__(self, threshold: int) -> None:
        """Record ``threshold`` in the single slot."""
        self.threshold = threshold

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """No matches, ever."""
        return ()


class _MutableClassAttributePattern(Pattern):
    """A pattern sharing a mutable list on the class."""

    seen: list[int] = []  # noqa: RUF012 - the shared mutable attribute under test

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """No matches, ever."""
        return ()


def _first_node(diagram: Diagram) -> NodeId:
    """The lowest node id in ``diagram``."""
    return min(diagram.nodes)


class TestNodeDigest:
    def test_independently_built_equal_data_digests_equally(self) -> None:
        left, left_id = _pair()
        right, right_id = _pair(padding=5)
        assert left_id != right_id
        assert node_digest(left, left_id) == node_digest(right, right_id)

    @pytest.mark.parametrize(
        ("label", "build"), NODE_FIELD_EDITS, ids=[e[0] for e in NODE_FIELD_EDITS]
    )
    def test_each_contributing_field_changes_the_digest(self, label: str, build: Build) -> None:
        base, base_id = _pair()
        edited, edited_id = build()
        assert node_digest(base, base_id) != node_digest(edited, edited_id), label

    def test_two_different_phases_digest_differently(self) -> None:
        left, left_id = _pair(phase=PHASE_HALF)
        right, right_id = _pair(phase=PHASE_THIRD)
        assert node_digest(left, left_id) != node_digest(right, right_id)

    def test_a_wire_edit_leaves_the_node_digest_alone(self) -> None:
        base, base_id = _pair()
        rewired, rewired_id = _pair(wires=((0, 1),))
        assert node_digest(base, base_id) == node_digest(rewired, rewired_id)

    def test_an_unknown_node_is_a_domain_error(self) -> None:
        diagram, _ = _pair()
        with pytest.raises(CacheDomainError):
            node_digest(diagram, NodeId(9999))


class TestIncidenceDigest:
    def test_a_copy_digests_equally(self) -> None:
        diagram, b_id = _pair()
        assert incidence_digest(diagram, b_id) == incidence_digest(diagram.copy(), b_id)

    def test_the_far_node_id_is_part_of_the_digest(self) -> None:
        left, left_id = _pair()
        right, right_id = _pair(padding=5)
        assert incidence_digest(left, left_id) != incidence_digest(right, right_id)

    @pytest.mark.parametrize(
        ("label", "build"), INCIDENCE_EDITS, ids=[e[0] for e in INCIDENCE_EDITS]
    )
    def test_each_incidence_edit_changes_the_digest(self, label: str, build: Build) -> None:
        base, base_id = _pair()
        edited, edited_id = build()
        assert incidence_digest(base, base_id) != incidence_digest(edited, edited_id), label

    def test_a_wire_edit_changes_both_endpoints(self) -> None:
        base, base_id = _pair()
        edited, edited_id = _pair(wires=((0, 1),))
        base_a = _first_node(base)
        edited_a = _first_node(edited)
        assert incidence_digest(base, base_a) != incidence_digest(edited, edited_a)
        assert incidence_digest(base, base_id) != incidence_digest(edited, edited_id)

    def test_a_phase_edit_leaves_the_incidence_digest_alone(self) -> None:
        base, base_id = _pair()
        phased, phased_id = _pair(phase=PHASE_HALF)
        assert incidence_digest(base, base_id) == incidence_digest(phased, phased_id)

    def test_an_unknown_node_is_a_domain_error(self) -> None:
        diagram, _ = _pair()
        with pytest.raises(CacheDomainError):
            incidence_digest(diagram, NodeId(9999))


class TestFingerprint:
    def test_a_copy_fingerprints_identically(self) -> None:
        diagram, _ = _pair()
        fp = fingerprint(diagram)
        assert isinstance(fp, DiagramFingerprint)
        assert fp == fingerprint(diagram.copy())

    def test_a_node_entry_folds_both_digests(self) -> None:
        diagram, b_id = _pair()
        fp = fingerprint(diagram)
        assert set(fp.nodes) == set(diagram.nodes)
        assert fp.nodes[b_id] not in (node_digest(diagram, b_id), incidence_digest(diagram, b_id))

    def test_a_phase_edit_dirties_exactly_that_node(self) -> None:
        diagram, b_id = _pair()
        before = fingerprint(diagram)
        edited = diagram.copy()
        edited.set_phase(b_id, PHASE_HALF)
        assert fingerprint(edited).changed_nodes(before) == frozenset({b_id})

    def test_a_wire_edit_dirties_both_endpoints(self) -> None:
        diagram, b_id = _pair()
        before = fingerprint(diagram)
        edited = diagram.copy()
        a_id = _first_node(edited)
        edited.add_wire(PortRef(a_id, Direction.OUTPUT, 1), PortRef(b_id, Direction.INPUT, 1))
        assert fingerprint(edited).changed_nodes(before) == frozenset({a_id, b_id})

    def test_an_added_node_is_the_only_dirty_node(self) -> None:
        diagram, _ = _pair()
        before = fingerprint(diagram)
        edited = diagram.copy()
        fresh = edited.add_node(X_SPIDER, input_dims=[DIM], output_dims=[])
        assert fingerprint(edited).changed_nodes(before) == frozenset({fresh})

    def test_a_removed_node_is_dirty_together_with_its_neighbour(self) -> None:
        diagram, b_id = _pair()
        before = fingerprint(diagram)
        edited = diagram.copy()
        a_id = _first_node(edited)
        edited.remove_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
        edited.remove_node(a_id)
        assert fingerprint(edited).changed_nodes(before) == frozenset({a_id, b_id})

    def test_changed_nodes_is_empty_between_equal_diagrams(self) -> None:
        diagram, _ = _pair()
        assert fingerprint(diagram).changed_nodes(fingerprint(diagram.copy())) == frozenset()

    def test_changed_nodes_is_symmetric(self) -> None:
        diagram, b_id = _pair()
        before = fingerprint(diagram)
        edited = diagram.copy()
        edited.set_phase(b_id, PHASE_HALF)
        after = fingerprint(edited)
        assert after.changed_nodes(before) == before.changed_nodes(after)

    def test_a_non_fingerprint_comparison_is_a_grammar_error(self) -> None:
        fp = fingerprint(_pair()[0])
        with pytest.raises(CacheGrammarError):
            fp.changed_nodes(object())  # type: ignore[arg-type]
        with pytest.raises(CacheGrammarError):
            fp.global_change(object())  # type: ignore[arg-type]


class TestGlobalChange:
    def test_a_boundary_change_is_global(self) -> None:
        base = fingerprint(_pair()[0])
        assert fingerprint(_pair(boundary=(1, 0))[0]).global_change(base)
        assert fingerprint(_pair(boundary=())[0]).global_change(base)

    def test_a_scalar_change_is_global(self) -> None:
        assert fingerprint(_scalar_edit()).global_change(fingerprint(_pair()[0]))

    def test_a_parameter_change_is_global(self) -> None:
        assert fingerprint(_parameter_edit()).global_change(fingerprint(_pair()[0]))

    def test_a_bang_box_change_is_global(self) -> None:
        base = fingerprint(_pair()[0])
        assert fingerprint(_boxed()[0]).global_change(base)
        assert fingerprint(_boxed(multiplicity=Mult.concrete(4))[0]).global_change(
            fingerprint(_boxed()[0])
        )

    def test_a_phase_change_is_not_global(self) -> None:
        after = fingerprint(_pair(phase=PHASE_HALF)[0])
        base = fingerprint(_pair()[0])
        assert not after.global_change(base)
        assert after.changed_nodes(base)

    def test_a_wire_change_is_not_global(self) -> None:
        after = fingerprint(_pair(wires=((0, 0), (1, 1)))[0])
        base = fingerprint(_pair()[0])
        assert not after.global_change(base)
        assert after.global_key != base.global_key

    def test_an_added_node_is_not_global(self) -> None:
        assert not fingerprint(_extra_node_edit()).global_change(fingerprint(_pair()[0]))


class TestGlobalKey:
    def test_a_copy_keys_identically(self) -> None:
        diagram, _ = _pair()
        assert fingerprint(diagram).global_key == fingerprint(diagram.copy()).global_key

    def test_renumbering_changes_the_exact_key(self) -> None:
        assert fingerprint(_pair()[0]).global_key != fingerprint(_pair(padding=4)[0]).global_key

    @pytest.mark.parametrize(
        ("label", "build"), GLOBAL_KEY_EDITS, ids=[e[0] for e in GLOBAL_KEY_EDITS]
    )
    def test_every_single_field_edit_moves_the_global_key(
        self, label: str, build: Callable[[], Diagram]
    ) -> None:
        assert fingerprint(_pair()[0]).global_key != fingerprint(build()).global_key, label

    def test_the_component_keys_are_distinct_from_each_other(self) -> None:
        fp = fingerprint(_boxed()[0])
        keys = (fp.global_key, fp.boundary_key, fp.env_key, fp.boxes_key)
        assert len(set(keys)) == len(keys)


class TestLruMemo:
    def test_a_miss_computes_and_a_repeat_hits(self) -> None:
        memo: LruMemo[str, int] = LruMemo(4)
        calls: list[str] = []

        def compute() -> int:
            calls.append("a")
            return 1

        assert memo.get_or_compute("a", compute) == 1
        assert memo.get_or_compute("a", compute) == 1
        assert calls == ["a"]
        assert memo.stats.hits == 1
        assert memo.stats.misses == 1

    def test_the_least_recently_used_entry_is_evicted(self) -> None:
        memo: LruMemo[str, int] = LruMemo(2)
        memo.get_or_compute("a", lambda: 1)
        memo.get_or_compute("b", lambda: 2)
        assert memo.get_or_compute("a", lambda: 99) == 1
        memo.get_or_compute("c", lambda: 3)
        assert memo.keys() == ("a", "c")
        assert "b" not in memo
        assert memo.stats.evictions == 1

    def test_eviction_order_follows_insertion_without_touches(self) -> None:
        memo: LruMemo[int, int] = LruMemo(3)
        for key in range(5):
            memo.get_or_compute(key, lambda key=key: key)  # type: ignore[misc]
        assert memo.keys() == (2, 3, 4)
        assert len(memo) == 3
        assert memo.stats == CacheStats(hits=0, misses=5, evictions=2)

    def test_invalidate_reports_whether_an_entry_was_held(self) -> None:
        memo: LruMemo[str, int] = LruMemo(4)
        memo.get_or_compute("a", lambda: 1)
        assert memo.invalidate("a") is True
        assert memo.invalidate("a") is False
        assert memo.stats.invalidations == 1
        assert len(memo) == 0

    def test_invalidate_where_drops_the_matching_keys_only(self) -> None:
        memo: LruMemo[int, int] = LruMemo(8)
        for key in range(6):
            memo.get_or_compute(key, lambda key=key: key)  # type: ignore[misc]
        assert memo.invalidate_where(lambda key: key % 2 == 0) == 3
        assert memo.keys() == (1, 3, 5)
        assert memo.stats.invalidations == 3

    def test_clear_reports_how_many_entries_went(self) -> None:
        memo: LruMemo[int, int] = LruMemo(8)
        for key in range(3):
            memo.get_or_compute(key, lambda key=key: key)  # type: ignore[misc]
        assert memo.clear() == 3
        assert memo.clear() == 0
        assert memo.stats.invalidations == 3
        assert memo.keys() == ()

    def test_contains_touches_neither_recency_nor_counters(self) -> None:
        memo: LruMemo[str, int] = LruMemo(2)
        memo.get_or_compute("a", lambda: 1)
        memo.get_or_compute("b", lambda: 2)
        assert "a" in memo
        memo.get_or_compute("c", lambda: 3)
        assert memo.keys() == ("b", "c")
        assert memo.stats.hits == 0

    @pytest.mark.parametrize("bad", [0, -1, True, False, "4", 4.0, None])
    def test_a_non_positive_or_non_int_capacity_is_rejected(self, bad: object) -> None:
        with pytest.raises(CacheGrammarError):
            LruMemo(bad)  # type: ignore[arg-type]

    def test_a_non_callable_compute_is_rejected(self) -> None:
        memo: LruMemo[str, int] = LruMemo(2)
        with pytest.raises(CacheGrammarError):
            memo.get_or_compute("a", 7)  # type: ignore[arg-type]

    def test_a_non_callable_predicate_is_rejected(self) -> None:
        memo: LruMemo[str, int] = LruMemo(2)
        with pytest.raises(CacheGrammarError):
            memo.invalidate_where(7)  # type: ignore[arg-type]

    def test_max_entries_is_reported(self) -> None:
        assert LruMemo(5).max_entries == 5


class TestCacheStats:
    def test_total_is_hits_plus_misses(self) -> None:
        assert CacheStats(hits=3, misses=1, evictions=7, invalidations=9).total == 4

    def test_hit_rate_is_the_hit_fraction(self) -> None:
        assert CacheStats(hits=3, misses=1).hit_rate == 0.75

    def test_an_empty_stats_object_has_a_zero_hit_rate(self) -> None:
        empty = CacheStats()
        assert empty.total == 0
        assert empty.hit_rate == 0.0

    def test_a_miss_only_run_has_a_zero_hit_rate(self) -> None:
        memo: LruMemo[int, int] = LruMemo(4)
        memo.get_or_compute(1, lambda: 1)
        assert memo.stats.hit_rate == 0.0

    def test_counters_accumulate_across_operations(self) -> None:
        memo: LruMemo[int, int] = LruMemo(2)
        memo.get_or_compute(1, lambda: 1)
        memo.get_or_compute(1, lambda: 1)
        memo.get_or_compute(2, lambda: 2)
        memo.get_or_compute(3, lambda: 3)
        memo.invalidate(3)
        assert memo.stats == CacheStats(hits=1, misses=3, evictions=1, invalidations=1)
        assert memo.stats.total == 4

    def test_the_unkeyable_counter_rises_for_an_unkeyable_pattern(self) -> None:
        cache = MatchCache()
        diagram, _ = _pair()
        cache.matches(_StatefulPattern(2), diagram)
        assert cache.stats.unkeyable == 1
        assert cache.stats.total == 0

    def test_the_unkeyable_counter_stays_at_zero_for_a_real_pattern(self) -> None:
        cache = MatchCache()
        diagram, _ = _pair()
        cache.matches(FusionPattern(), diagram)
        assert cache.stats.unkeyable == 0


class TestPatternKey:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS, ids=[type(p).__name__ for p in ALL_PATTERNS])
    def test_every_real_pattern_keys_stably(self, pattern: Pattern) -> None:
        assert pattern_key(pattern) == pattern_key(type(pattern)())
        assert keyable(pattern)

    def test_the_real_patterns_key_distinctly(self) -> None:
        keys = [pattern_key(pattern) for pattern in ALL_PATTERNS]
        assert len(set(keys)) == len(ALL_PATTERNS)

    def test_a_key_names_the_pattern_type(self) -> None:
        assert pattern_key(CapPattern()).startswith("archytaszx.rewrite.match.CapPattern|")

    def test_a_dataclass_pattern_keys_on_its_field_values(self) -> None:
        guarded = FusionPattern(dimension_guards=(DimensionGuard(DimensionGuardKind.PRIME),))
        assert pattern_key(FusionPattern()) != pattern_key(guarded)
        assert pattern_key(guarded) == pattern_key(
            FusionPattern(dimension_guards=(DimensionGuard(DimensionGuardKind.PRIME),))
        )
        assert repr(FusionPattern()) in pattern_key(FusionPattern())

    def test_a_stateless_identity_repr_pattern_keys_on_its_dotted_name(self) -> None:
        key = pattern_key(_StatelessPattern())
        assert key == f"{__name__}._StatelessPattern|"
        assert key == pattern_key(_StatelessPattern())

    def test_a_stateful_identity_repr_pattern_is_rejected(self) -> None:
        with pytest.raises(CacheGrammarError):
            pattern_key(_StatefulPattern(2))
        assert not keyable(_StatefulPattern(2))

    def test_a_bare_string_slots_pattern_holding_state_is_rejected(self) -> None:
        with pytest.raises(CacheGrammarError):
            pattern_key(_BareStringSlotsPattern(2))
        assert not keyable(_BareStringSlotsPattern(2))

    def test_a_mutable_class_attribute_pattern_is_rejected(self) -> None:
        with pytest.raises(CacheGrammarError):
            pattern_key(_MutableClassAttributePattern())
        assert not keyable(_MutableClassAttributePattern())

    def test_a_non_pattern_is_rejected(self) -> None:
        with pytest.raises(CacheGrammarError):
            pattern_key(object())  # type: ignore[arg-type]


class TestMatchCache:
    def test_a_repeated_scan_hits(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        cache = MatchCache()
        pattern = FusionPattern()
        first = cache.matches(pattern, diagram)
        second = cache.matches(pattern, diagram)
        assert first
        assert second == first
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_an_edit_misses(self) -> None:
        diagram, a_id, _ = build_ghz_with_copy(DIM)
        cache = MatchCache()
        pattern = FusionPattern()
        cache.matches(pattern, diagram)
        edited = diagram.copy()
        edited.set_phase(a_id, PHASE_HALF)
        cache.matches(pattern, edited)
        assert cache.stats.misses == 2
        assert cache.stats.hits == 0

    def test_two_patterns_do_not_share_an_entry(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        cache = MatchCache()
        cache.matches(FusionPattern(), diagram)
        cache.matches(CapPattern(), diagram)
        assert cache.stats.misses == 2
        assert cache.stats.hits == 0

    def test_a_supplied_fingerprint_keys_the_same_entry(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        cache = MatchCache()
        pattern = FusionPattern()
        cache.matches(pattern, diagram)
        cache.matches(pattern, diagram, fingerprint=fingerprint(diagram))
        assert cache.stats.hits == 1

    def test_an_unkeyable_pattern_is_served_uncached(self) -> None:
        diagram, _ = _pair()
        cache = MatchCache()
        pattern = _StatefulPattern(2)
        assert cache.matches(pattern, diagram) == ()
        assert cache.matches(pattern, diagram) == ()
        assert cache.stats.unkeyable == 2
        assert cache.stats.total == 0

    @pytest.mark.parametrize("name, diagram", CORPUS, ids=[entry[0] for entry in CORPUS])
    def test_canonical_agrees_with_canonical_key(self, name: str, diagram: Diagram) -> None:
        cache = MatchCache()
        assert cache.canonical(diagram) == canonical_key(diagram), name
        assert cache.canonical(diagram) == canonical_key(diagram)
        assert cache.stats.hits == 1

    @pytest.mark.parametrize("name, diagram", CORPUS, ids=[entry[0] for entry in CORPUS])
    def test_matches_anchored_agrees_with_the_filtered_full_scan(
        self, name: str, diagram: Diagram
    ) -> None:
        cache = MatchCache()
        node_ids = frozenset(diagram.nodes)
        anchor_sets = [frozenset(), node_ids, *(frozenset({nid}) for nid in sorted(node_ids))]
        for pattern in ALL_PATTERNS:
            full = pattern.find_matches(diagram)
            for anchors in anchor_sets:
                expected = tuple(
                    match for match in full if not anchors.isdisjoint(match.support_node_ids)
                )
                assert cache.matches_anchored(pattern, diagram, anchors) == expected, (
                    f"{name} {type(pattern).__name__} {sorted(anchors)}"
                )

    def test_the_corpus_scan_is_not_vacuous(self) -> None:
        cache = MatchCache()
        found = sum(
            len(cache.matches(pattern, diagram))
            for _, diagram in CORPUS
            for pattern in ALL_PATTERNS
        )
        assert found > 0

    def test_anchoring_to_every_node_reproduces_the_full_scan(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        pattern = FusionPattern()
        anchored = MatchCache().matches_anchored(pattern, diagram, frozenset(diagram.nodes))
        assert anchored == pattern.find_matches(diagram)
        assert anchored

    def test_anchoring_to_nothing_finds_nothing(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        assert MatchCache().matches_anchored(FusionPattern(), diagram, frozenset()) == ()

    def test_matches_anchored_is_not_cached(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        cache = MatchCache()
        anchors = frozenset(diagram.nodes)
        cache.matches_anchored(FusionPattern(), diagram, anchors)
        cache.matches_anchored(FusionPattern(), diagram, anchors)
        assert cache.stats.total == 0

    def test_invalidate_drops_only_that_diagram_s_entries(self) -> None:
        diagram, a_id, _ = build_ghz_with_copy(DIM)
        other = diagram.copy()
        other.set_phase(a_id, PHASE_HALF)
        cache = MatchCache()
        pattern = FusionPattern()
        cache.matches(pattern, diagram)
        cache.matches(pattern, other)
        cache.canonical(diagram)
        assert cache.invalidate(fingerprint(diagram)) == 2
        assert cache.invalidate(fingerprint(diagram)) == 0
        cache.matches(pattern, other)
        assert cache.stats.hits == 1

    def test_clear_drops_both_memos(self) -> None:
        diagram, _, _ = build_ghz_with_copy(DIM)
        cache = MatchCache()
        cache.matches(FusionPattern(), diagram)
        cache.canonical(diagram)
        assert cache.clear() == 2
        assert cache.clear() == 0
        cache.matches(FusionPattern(), diagram)
        assert cache.stats.misses == 3

    def test_entries_are_bounded_by_max_entries(self) -> None:
        cache = MatchCache(max_entries=2)
        pattern = FusionPattern()
        for extra in range(4):
            diagram, _, _ = build_ghz_with_copy(DIM)
            for _ in range(extra):
                diagram.add_node(X_SPIDER, input_dims=[DIM], output_dims=[])
            cache.matches(pattern, diagram)
        assert cache.stats.evictions == 2


class TestValueMemo:
    def _node_memo(self) -> tuple[ValueMemo[Node, int], list[Node]]:
        """A memo counting a node's legs, and the calls it made."""
        calls: list[Node] = []

        def compute(node: Node) -> int:
            calls.append(node)
            return node.num_inputs + node.num_outputs

        return ValueMemo(compute=compute, key=node_key), calls

    def test_equal_node_content_hits_regardless_of_id(self) -> None:
        memo, calls = self._node_memo()
        left, left_id = _pair()
        right, right_id = _pair(padding=3)
        assert memo.get(left.nodes[left_id]) == 4
        assert memo.get(right.nodes[right_id]) == 4
        assert len(calls) == 1
        assert memo.stats.hits == 1

    def test_different_node_content_misses(self) -> None:
        memo, calls = self._node_memo()
        base, base_id = _pair()
        other, other_id = _pair(out_dim=OTHER_DIM)
        memo.get(base.nodes[base_id])
        memo.get(other.nodes[other_id])
        assert len(calls) == 2
        assert memo.stats.misses == 2

    def test_diagram_key_is_the_fingerprint_global_key(self) -> None:
        for name, diagram in CORPUS:
            assert diagram_key(diagram) == fingerprint(diagram).global_key, name

    def test_a_diagram_keyed_memo_hits_on_a_copy(self) -> None:
        diagram, _ = _pair()
        calls: list[Diagram] = []

        def compute(subject: Diagram) -> int:
            calls.append(subject)
            return len(subject.nodes)

        memo: ValueMemo[Diagram, int] = ValueMemo(compute=compute, key=diagram_key)
        assert memo.get(diagram) == 2
        assert memo.get(diagram.copy()) == 2
        assert len(calls) == 1

    def test_key_for_reports_the_content_key(self) -> None:
        memo, _ = self._node_memo()
        diagram, b_id = _pair()
        assert memo.key_for(diagram.nodes[b_id]) == node_key(diagram.nodes[b_id])

    def _array_memo(self) -> tuple[ValueMemo[Node, np.ndarray], np.ndarray]:
        """A memo handing out one fixed source array, and that array."""
        source = np.array([1.0, 2.0, 3.0])
        return ValueMemo(compute=lambda node: source, key=node_key), source

    def test_a_numpy_value_is_handed_out_read_only(self) -> None:
        memo, _ = self._array_memo()
        diagram, b_id = _pair()
        value = memo.get(diagram.nodes[b_id])
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value[0] = 99.0

    def test_mutating_the_source_array_does_not_corrupt_the_entry(self) -> None:
        memo, source = self._array_memo()
        diagram, b_id = _pair()
        first = np.array(memo.get(diagram.nodes[b_id]))
        source[0] = 99.0
        second = memo.get(diagram.nodes[b_id])
        assert np.array_equal(second, first)
        assert second[0] == 1.0

    def test_each_get_yields_a_fresh_view(self) -> None:
        memo, _ = self._array_memo()
        diagram, b_id = _pair()
        first = memo.get(diagram.nodes[b_id])
        second = memo.get(diagram.nodes[b_id])
        assert first is not second
        assert np.array_equal(first, second)

    def test_a_non_numpy_value_is_passed_through(self) -> None:
        memo, _ = self._node_memo()
        diagram, b_id = _pair()
        assert memo.get(diagram.nodes[b_id]) == 4

    def test_the_least_recently_used_entry_is_evicted(self) -> None:
        nodes = [
            _pair(out_dim=Dim.concrete(size))[0].nodes[_pair(out_dim=Dim.concrete(size))[1]]
            for size in (2, 3, 5)
        ]
        bounded: ValueMemo[Node, int] = ValueMemo(
            compute=lambda node: node.num_outputs, key=node_key, max_entries=2
        )
        for node in nodes:
            bounded.get(node)
        assert len(bounded) == 2
        assert bounded.stats.evictions == 1

    def test_invalidate_and_clear_report_what_went(self) -> None:
        memo, _ = self._node_memo()
        diagram, b_id = _pair()
        node = diagram.nodes[b_id]
        memo.get(node)
        assert memo.invalidate(node) is True
        assert memo.invalidate(node) is False
        memo.get(node)
        assert memo.clear() == 1
        assert memo.clear() == 0

    def test_max_entries_is_reported(self) -> None:
        memo: ValueMemo[Node, int] = ValueMemo(compute=lambda node: 0, key=node_key, max_entries=7)
        assert memo.max_entries == 7

    def test_a_non_callable_compute_or_key_is_rejected(self) -> None:
        with pytest.raises(CacheGrammarError):
            ValueMemo(compute=7, key=node_key)  # type: ignore[arg-type]
        with pytest.raises(CacheGrammarError):
            ValueMemo(compute=node_key, key=7)  # type: ignore[arg-type]

    def test_a_bad_capacity_is_rejected(self) -> None:
        with pytest.raises(CacheGrammarError):
            ValueMemo(compute=lambda node: 0, key=node_key, max_entries=0)

    def test_a_key_function_returning_a_non_string_is_rejected(self) -> None:
        memo: ValueMemo[Node, int] = ValueMemo(
            compute=lambda node: 0,
            key=lambda node: 7,  # type: ignore[arg-type,return-value]
        )
        diagram, b_id = _pair()
        with pytest.raises(CacheGrammarError):
            memo.get(diagram.nodes[b_id])

    def test_node_key_ignores_ids_and_wires(self) -> None:
        left, left_id = _pair()
        right, right_id = _pair(padding=3, wires=())
        assert node_key(left.nodes[left_id]) == node_key(right.nodes[right_id])

    def test_node_key_tracks_the_node_data(self) -> None:
        base, base_id = _pair()
        variants: tuple[Build, ...] = (
            lambda: _pair(generator_type=X_SPIDER),
            lambda: _pair(in_dim=OTHER_DIM),
            lambda: _pair(out_dim=OTHER_DIM),
            lambda: _pair(phase=PHASE_HALF),
        )
        for build in variants:
            other, other_id = build()
            assert node_key(base.nodes[base_id]) != node_key(other.nodes[other_id])


def _bad_input_calls() -> tuple[tuple[str, Callable[[], object]], ...]:
    """Every public entry point paired with a call that must raise a :class:`CacheError`."""
    diagram, b_id = _pair()
    fp = fingerprint(diagram)
    memo: LruMemo[str, int] = LruMemo(2)
    value_memo: ValueMemo[Node, int] = ValueMemo(compute=lambda node: 0, key=node_key)
    cache = MatchCache()
    pattern = FusionPattern()
    return (
        ("node_digest non-diagram", lambda: node_digest(object(), b_id)),  # type: ignore[arg-type]
        ("node_digest unknown node", lambda: node_digest(diagram, NodeId(999))),
        ("node_digest non-int node", lambda: node_digest(diagram, "b")),  # type: ignore[arg-type]
        ("incidence_digest non-diagram", lambda: incidence_digest(None, b_id)),  # type: ignore[arg-type]
        ("incidence_digest unknown node", lambda: incidence_digest(diagram, NodeId(999))),
        ("fingerprint non-diagram", lambda: fingerprint("diagram")),  # type: ignore[arg-type]
        ("fingerprint of a node", lambda: fingerprint(diagram.nodes[b_id])),  # type: ignore[arg-type]
        ("changed_nodes non-fingerprint", lambda: fp.changed_nodes(diagram)),  # type: ignore[arg-type]
        ("global_change non-fingerprint", lambda: fp.global_change(None)),  # type: ignore[arg-type]
        ("LruMemo zero capacity", lambda: LruMemo(0)),
        ("LruMemo bool capacity", lambda: LruMemo(True)),
        ("LruMemo float capacity", lambda: LruMemo(2.5)),  # type: ignore[arg-type]
        ("get_or_compute non-callable", lambda: memo.get_or_compute("a", None)),  # type: ignore[arg-type]
        ("invalidate_where non-callable", lambda: memo.invalidate_where("a")),  # type: ignore[arg-type]
        ("pattern_key non-pattern", lambda: pattern_key("FusionPattern")),  # type: ignore[arg-type]
        ("pattern_key stateful repr", lambda: pattern_key(_StatefulPattern(1))),
        ("pattern_key bare-string slots", lambda: pattern_key(_BareStringSlotsPattern(1))),
        (
            "pattern_key mutable class attribute",
            lambda: pattern_key(_MutableClassAttributePattern()),
        ),
        ("MatchCache bad capacity", lambda: MatchCache(0)),
        ("MatchCache.matches non-diagram", lambda: cache.matches(pattern, object())),  # type: ignore[arg-type]
        (
            "MatchCache.matches bad fingerprint",
            lambda: cache.matches(pattern, diagram, fingerprint=object()),  # type: ignore[arg-type]
        ),
        (
            "matches_anchored non-diagram",
            lambda: cache.matches_anchored(pattern, object(), frozenset()),  # type: ignore[arg-type]
        ),
        (
            "matches_anchored list anchors",
            lambda: cache.matches_anchored(pattern, diagram, [b_id]),  # type: ignore[arg-type]
        ),
        (
            "matches_anchored string anchors",
            lambda: cache.matches_anchored(pattern, diagram, frozenset({"b"})),  # type: ignore[arg-type]
        ),
        (
            "matches_anchored bool anchors",
            lambda: cache.matches_anchored(pattern, diagram, frozenset({True})),  # type: ignore[arg-type]
        ),
        (
            "matches_anchored bad fingerprint",
            lambda: cache.matches_anchored(pattern, diagram, frozenset(), fingerprint=7),  # type: ignore[arg-type]
        ),
        ("canonical non-diagram", lambda: cache.canonical(object())),  # type: ignore[arg-type]
        (
            "canonical bad fingerprint",
            lambda: cache.canonical(diagram, fingerprint="key"),  # type: ignore[arg-type]
        ),
        ("MatchCache.invalidate non-fingerprint", lambda: cache.invalidate(diagram)),  # type: ignore[arg-type]
        ("node_key non-node", lambda: node_key(diagram)),  # type: ignore[arg-type]
        ("diagram_key non-diagram", lambda: diagram_key(diagram.nodes[b_id])),  # type: ignore[arg-type]
        ("ValueMemo non-callable compute", lambda: ValueMemo(compute=1, key=node_key)),  # type: ignore[arg-type]
        ("ValueMemo non-callable key", lambda: ValueMemo(compute=node_key, key=1)),  # type: ignore[arg-type]
        (
            "ValueMemo bad capacity",
            lambda: ValueMemo(compute=node_key, key=node_key, max_entries=-3),
        ),
        ("ValueMemo.get bad subject", lambda: value_memo.get(object())),  # type: ignore[arg-type]
    )


BAD_INPUT_CALLS = _bad_input_calls()

FOREIGN = (AttributeError, KeyError, TypeError, IndexError, ValueError)


class TestErrorHierarchy:
    @pytest.mark.parametrize(
        ("label", "call"), BAD_INPUT_CALLS, ids=[entry[0] for entry in BAD_INPUT_CALLS]
    )
    def test_bad_input_raises_only_a_cache_error(
        self, label: str, call: Callable[[], object]
    ) -> None:
        with pytest.raises(CacheError) as caught:
            call()
        assert not isinstance(caught.value, FOREIGN), label

    def test_no_foreign_exception_escapes_a_public_entry_point(self) -> None:
        for label, call in BAD_INPUT_CALLS:
            try:
                call()
            except CacheError:
                continue
            except Exception as exc:  # noqa: BLE001 - the point of the assertion
                pytest.fail(f"{label} raised {type(exc).__name__}: {exc}")
            else:
                pytest.fail(f"{label} raised nothing")

    def test_the_grammar_and_domain_errors_are_cache_errors(self) -> None:
        assert issubclass(CacheGrammarError, CacheError)
        assert issubclass(CacheDomainError, CacheError)
        assert not issubclass(CacheError, FOREIGN)

    def test_a_compute_callable_s_own_error_is_not_wrapped(self) -> None:
        memo: LruMemo[str, int] = LruMemo(2)

        def boom() -> int:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            memo.get_or_compute("a", boom)


class TestCacheCrossProcessDeterminism:
    """Every cache key must be byte-identical across ``PYTHONHASHSEED`` values."""

    SCRIPT = Path(__file__).parent / "_cache_determinism_script.py"
    SEEDS = ("0", "1", "12345", "2147483647")

    def _run_with_seed(self, seed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, str(self.SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_digests_fingerprints_keys_and_matches_are_byte_identical(self) -> None:
        outputs = [self._run_with_seed(seed) for seed in self.SEEDS]
        assert outputs[0], "the driver script printed nothing"
        assert "global_key=" in outputs[0]
        assert "pattern_key=" in outputs[0]
        assert "matches=(FusionMatch(" in outputs[0]
        for seed, output in zip(self.SEEDS[1:], outputs[1:], strict=True):
            assert output == outputs[0], f"PYTHONHASHSEED={seed} disagreed"
