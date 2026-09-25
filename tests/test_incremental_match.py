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

"""Establishes the Phase 12 locality hooks and ``IncrementalMatcher``: the anchored-scan
contract, the ``order_key`` ordering and totality contract, and ``rematch(D) == seed(D)``
after every kind of local edit."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Mapping
from functools import partial
from typing import cast

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.cache import IncrementalMatcher, MatchCache, pattern_key
from archytaszx.rewrite.engine import apply, normal_form_rules
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
    find_fourier_matches,
)
from archytaszx.rewrite.rule import DimensionConstraint, Match, Pattern, SideConditionOutcome

from .helpers import build_ghz_with_copy
from .test_rules_library_phase11 import (
    bialgebra_diagram,
    fourier_state_diagram,
    hopf_diagram,
    identity_chain,
    inp,
    out,
    state_copy_diagram,
    triangle_chain,
)

PATTERNS: tuple[Pattern, ...] = (
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
"""Every concrete :class:`~archytaszx.rewrite.rule.Pattern` in ``match.py``."""

PATTERN_NAMES: tuple[str, ...] = tuple(type(pattern).__name__ for pattern in PATTERNS)
"""The nine pattern class names, used as parametrization ids."""

ABSENT_NODE_ID = NodeId(9_999)
"""A node id no corpus diagram contains."""


def phase_vector(d: int) -> PhaseVector:
    """A non-zero phase vector over ``Dim.concrete(d)``."""
    return PhaseVector(Dim.concrete(d), {k: Phase.turns(sp.Rational(k, d)) for k in range(1, d)})


def fourier_chain(d: int, length: int) -> Diagram:
    """A Z state into ``length`` F boxes in series, the last box's output on the boundary."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    boxes = [diagram.add_node(FOURIER_BOX, [dim], [dim]) for _ in range(length)]
    diagram.add_wire(out(state), inp(boxes[0]))
    for first, second in itertools.pairwise(boxes):
        diagram.add_wire(out(first), inp(second))
    diagram.set_boundary_outputs([out(boxes[-1])])
    return diagram


def cap_pairs(d: int, count: int) -> Diagram:
    """``count`` independent phaseless Z states each wired into a phaseless X effect."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    for _ in range(count):
        state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
        effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
        diagram.add_wire(out(state), inp(effect))
    return diagram


def fusion_chain(d: int, length: int) -> Diagram:
    """``length`` Z spiders in series, each with one spare output on the boundary."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    spiders = [diagram.add_node(Z_SPIDER, [dim], [dim, dim]) for _ in range(length)]
    for first, second in itertools.pairwise(spiders):
        diagram.add_wire(out(first, 0), inp(second))
    diagram.set_boundary_inputs([inp(spiders[0])])
    diagram.set_boundary_outputs([out(spiders[-1], 0), *[out(node_id, 1) for node_id in spiders]])
    return diagram


def self_loop(d: int) -> Diagram:
    """A Z spider with one output wired back into its own input, feeding a second Z spider."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    spider = diagram.add_node(Z_SPIDER, [dim], [dim, dim, dim])
    partner = diagram.add_node(Z_SPIDER, [dim], [dim])
    diagram.add_wire(out(spider, 0), inp(spider, 0))
    diagram.add_wire(out(spider, 1), inp(partner))
    diagram.set_boundary_outputs([out(spider, 2), out(partner)])
    return diagram


def parallel_wires(d: int) -> Diagram:
    """Two Z spiders joined by two parallel wires."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    first = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    second = diagram.add_node(Z_SPIDER, [dim, dim], [dim])
    diagram.add_wire(out(first, 0), inp(second, 0))
    diagram.add_wire(out(first, 1), inp(second, 1))
    diagram.set_boundary_outputs([out(second)])
    return diagram


def bang_boxed(d: int) -> Diagram:
    """A state-copy diagram beside a spare Z spider held under one symbolic bang box."""
    dim = Dim.concrete(d)
    diagram = state_copy_diagram(d, 2)
    spare = diagram.add_node(Z_SPIDER, [dim], [dim])
    diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({spare}))
    return diagram


def nested_bang_boxed(d: int) -> Diagram:
    """A state-copy diagram beside two spare Z spiders under nested bang boxes."""
    dim = Dim.concrete(d)
    diagram = state_copy_diagram(d, 2)
    outer_node = diagram.add_node(Z_SPIDER, [dim], [dim])
    inner_node = diagram.add_node(Z_SPIDER, [dim], [dim])
    outer = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({outer_node, inner_node}))
    diagram.add_bang_box(Mult.concrete(2), node_scope=frozenset({inner_node}), parent=outer)
    return diagram


def symbolic_dim() -> Diagram:
    """Two fusable Z spiders whose every leg carries the symbolic dimension ``d``."""
    dim = Dim.symbol("d")
    diagram = Diagram()
    first = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    second = diagram.add_node(Z_SPIDER, [dim], [dim])
    diagram.add_wire(out(first, 0), inp(second))
    diagram.set_boundary_outputs([out(first, 1), out(second)])
    return diagram


def corpus() -> dict[str, Diagram]:
    """A freshly built diagram per corpus name."""
    return {
        "ghz_with_copy": build_ghz_with_copy(Dim.concrete(2))[0],
        "identity_chain": identity_chain(2),
        "triangle_chain": triangle_chain(2),
        "triangle_chain_inverse_first": triangle_chain(3, inverse_first=True),
        "state_copy": state_copy_diagram(3, 2),
        "hopf": hopf_diagram(2, 1, 1, 1, 1),
        "hopf_with_phases": hopf_diagram(3, 0, 1, 1, 0, phases=True),
        "bialgebra": bialgebra_diagram(2),
        "fourier_state": fourier_state_diagram(2, is_state=True),
        "fourier_effect": fourier_state_diagram(3, is_state=False),
        "fourier_chain_four": fourier_chain(2, 4),
        "fourier_chain_seven": fourier_chain(3, 7),
        "cap": cap_pairs(2, 1),
        "three_caps": cap_pairs(3, 3),
        "fusion_chain": fusion_chain(2, 4),
        "self_loop": self_loop(2),
        "parallel_wires": parallel_wires(2),
        "bang_boxed": bang_boxed(2),
        "nested_bang_boxed": nested_bang_boxed(2),
        "symbolic_dim": symbolic_dim(),
    }


CORPUS_NAMES: tuple[str, ...] = tuple(corpus())
"""Every corpus entry's name, used as parametrization ids."""


def anchor_sets(diagram: Diagram) -> tuple[frozenset[NodeId], ...]:
    """The empty set, every node, each singleton, and one id the diagram does not contain."""
    ids = sorted(diagram.nodes)
    return (
        frozenset(),
        frozenset(ids),
        frozenset({ABSENT_NODE_ID}),
        frozenset({ABSENT_NODE_ID, ids[0]}),
        *[frozenset({node_id}) for node_id in ids],
    )


def free_port(diagram: Diagram, direction: Direction) -> PortRef | None:
    """The lowest-id port in ``direction`` that no wire claims."""
    claimed = {ref for wire in diagram.wires for ref in (wire.a, wire.b)}
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        legs = node.num_inputs if direction is Direction.INPUT else node.num_outputs
        for index in range(legs):
            ref = PortRef(node_id, direction, index)
            if ref not in claimed:
                return ref
    return None


def _edit_add_node(diagram: Diagram) -> bool:
    """Add an unwired one-in-one-out Z spider."""
    dim = Dim.concrete(2)
    diagram.add_node(Z_SPIDER, [dim], [dim])
    return True


def _edit_add_and_wire_node(diagram: Diagram) -> bool:
    """Add a Z state and wire it into a free input port."""
    target = free_port(diagram, Direction.INPUT)
    if target is None:
        return False
    dim = diagram.nodes[target.node_id].inputs[target.index].dim
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    diagram.add_wire(out(state), target)
    return True


def _edit_remove_node(diagram: Diagram) -> bool:
    """Remove the highest-id node."""
    if not diagram.nodes:
        return False
    diagram.remove_node(max(diagram.nodes))
    return True


def _edit_add_wire(diagram: Diagram) -> bool:
    """Wire one free output port into one free input port on another node."""
    source = free_port(diagram, Direction.OUTPUT)
    target = free_port(diagram, Direction.INPUT)
    if source is None or target is None or source.node_id == target.node_id:
        return False
    diagram.add_wire(source, target)
    return True


def _edit_remove_wire(diagram: Diagram) -> bool:
    """Remove the first wire in ``Wire.sort_key`` order."""
    if not diagram.wires:
        return False
    wire = min(diagram.wires, key=lambda w: w.sort_key())
    diagram.remove_wire(wire.a, wire.b)
    return True


def _edit_set_phase(diagram: Diagram) -> bool:
    """Put a non-zero phase on the lowest-id node."""
    if not diagram.nodes:
        return False
    node_id = min(diagram.nodes)
    if diagram.nodes[node_id].phase is not None:
        return False
    diagram.set_phase(node_id, phase_vector(2))
    return True


def _edit_clear_phase(diagram: Diagram) -> bool:
    """Clear the phase of the lowest-id node that carries one."""
    for node_id in sorted(diagram.nodes):
        if diagram.nodes[node_id].phase is not None:
            diagram.set_phase(node_id, None)
            return True
    return False


def _edit_reorder_boundary(diagram: Diagram) -> bool:
    """Reverse the output boundary list."""
    if len(diagram.boundary_outputs) < 2:
        return False
    diagram.set_boundary_outputs(list(reversed(diagram.boundary_outputs)))
    return True


def _edit_truncate_boundary(diagram: Diagram) -> bool:
    """Drop the last output boundary entry."""
    if not diagram.boundary_outputs:
        return False
    diagram.set_boundary_outputs(list(diagram.boundary_outputs)[:-1])
    return True


def _edit_add_bang_box(diagram: Diagram) -> bool:
    """Put the lowest-id node under a fresh symbolic bang box."""
    if not diagram.nodes:
        return False
    diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({min(diagram.nodes)}))
    return True


def _edit_change_bang_box_scope(diagram: Diagram) -> bool:
    """Reduce the lowest-id bang box's node scope to the lowest-id node."""
    if not diagram.bang_boxes or not diagram.nodes:
        return False
    box_id = min(diagram.bang_boxes)
    scope = frozenset({min(diagram.nodes)})
    if diagram.bang_boxes[box_id].node_scope == scope:
        return False
    diagram.set_bang_box_node_scope(box_id, scope)
    return True


def _edit_change_bang_box_multiplicity(diagram: Diagram) -> bool:
    """Set the lowest-id bang box's multiplicity to a concrete five."""
    if not diagram.bang_boxes:
        return False
    diagram.set_bang_box_multiplicity(min(diagram.bang_boxes), Mult.concrete(5))
    return True


def _edit_remove_bang_box(diagram: Diagram) -> bool:
    """Remove the highest-id bang box."""
    if not diagram.bang_boxes:
        return False
    diagram.remove_bang_box(max(diagram.bang_boxes))
    return True


EDITS: Mapping[str, Callable[[Diagram], bool]] = {
    "add_node": _edit_add_node,
    "add_and_wire_node": _edit_add_and_wire_node,
    "remove_node": _edit_remove_node,
    "add_wire": _edit_add_wire,
    "remove_wire": _edit_remove_wire,
    "set_phase": _edit_set_phase,
    "clear_phase": _edit_clear_phase,
    "reorder_boundary": _edit_reorder_boundary,
    "truncate_boundary": _edit_truncate_boundary,
    "add_bang_box": _edit_add_bang_box,
    "change_bang_box_scope": _edit_change_bang_box_scope,
    "change_bang_box_multiplicity": _edit_change_bang_box_multiplicity,
    "remove_bang_box": _edit_remove_bang_box,
}
"""Every local edit the ``rematch == seed`` contract is exercised over, by name."""

EDIT_NAMES: tuple[str, ...] = tuple(EDITS)
"""Every edit name, used as parametrization ids."""


def matcher() -> IncrementalMatcher:
    """A matcher over all nine patterns, with no match cache."""
    return IncrementalMatcher(PATTERNS)


def full_scan(diagram: Diagram) -> dict[str, tuple[Match, ...]]:
    """Every pattern's full scan of ``diagram``, keyed as ``IncrementalMatcher`` keys it."""
    return {pattern_key(pattern): pattern.find_matches(diagram) for pattern in PATTERNS}


def assert_rematch_equals_seed(before: Diagram, after: Diagram) -> None:
    """Pin ``rematch(after) == seed(after)`` elementwise per pattern for a matcher seeded on
    ``before``."""
    incremental = matcher()
    incremental.seed(before)
    rematched = dict(incremental.rematch(after))
    seeded = dict(matcher().seed(after))
    assert set(rematched) == set(seeded)
    for key in sorted(seeded):
        assert rematched[key] == seeded[key], key


def wire_to_absent_node(d: int) -> Diagram:
    """Four F boxes in series whose last output runs to a node id the diagram lacks."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    boxes = [diagram.add_node(FOURIER_BOX, [dim], [dim]) for _ in range(4)]
    for first, second in itertools.pairwise(boxes):
        diagram.add_wire(out(first), inp(second))
    diagram.add_wire(out(boxes[-1]), PortRef(ABSENT_NODE_ID, Direction.INPUT, 0))
    return diagram


def legless_fourier_in_chain(d: int) -> Diagram:
    """Four F boxes in series, the third of them carrying no legs at all."""
    dim = Dim.concrete(d)
    diagram = Diagram()
    first = diagram.add_node(FOURIER_BOX, [dim], [dim])
    second = diagram.add_node(FOURIER_BOX, [dim], [dim])
    legless = diagram.add_node(FOURIER_BOX, input_dims=[], output_dims=[])
    fourth = diagram.add_node(FOURIER_BOX, [dim], [dim])
    diagram.add_wire(out(first), inp(second))
    diagram.add_wire(out(second), inp(legless))
    diagram.add_wire(out(legless), inp(fourth))
    return diagram


MALFORMED: Mapping[str, Callable[[], Diagram]] = {
    "wire_to_absent_node": lambda: wire_to_absent_node(2),
    "legless_fourier_in_chain": lambda: legless_fourier_in_chain(2),
}
"""The malformed diagrams the anchored Fourier scan must fail on exactly as the full scan does."""


def raised(call: Callable[[], object]) -> tuple[str, str]:
    """``call``'s exception type name and message; fails the test when it does not raise."""
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the parity of any raise is what is under test
        return type(exc).__name__, str(exc)
    pytest.fail("expected a raise, got a result")


@dataclasses.dataclass(frozen=True, slots=True)
class _SupportlessMatch:
    """A ``Match`` carrying no ``support_node_ids`` at all."""

    node_id: int
    side_condition_outcomes: tuple[SideConditionOutcome, ...] = ()
    dimension_constraints: tuple[DimensionConstraint, ...] = ()

    @property
    def all_side_conditions_passed(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class _SupportlessPattern(Pattern):
    """A keyable pattern whose matches expose no ``support_node_ids``: one per node."""

    locality_radius = 1

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        found = tuple(_SupportlessMatch(int(node_id)) for node_id in sorted(diagram.nodes))
        return cast("tuple[Match, ...]", found)


class TestCorpusIsNonDegenerate:
    def test_every_pattern_matches_somewhere_in_the_corpus(self) -> None:
        built = corpus()
        for pattern in PATTERNS:
            total = sum(len(pattern.find_matches(d)) for d in built.values())
            assert total > 0, type(pattern).__name__

    def test_some_diagram_carries_two_matches_of_one_pattern(self) -> None:
        built = corpus()
        pairs = [
            (type(pattern).__name__, name)
            for pattern in PATTERNS
            for name, diagram in built.items()
            if len(pattern.find_matches(diagram)) >= 2
        ]
        assert pairs

    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_every_corpus_entry_carries_at_least_one_match(self, name: str) -> None:
        diagram = corpus()[name]
        assert sum(len(pattern.find_matches(diagram)) for pattern in PATTERNS) > 0


class TestAnchoredScanEqualsFilteredFullScan:
    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_anchored_equals_the_filtered_full_scan(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        full = pattern.find_matches(diagram)
        for anchors in anchor_sets(diagram):
            expected = tuple(m for m in full if set(m.support_node_ids) & anchors)
            assert pattern.find_matches_anchored(diagram, anchors) == expected, sorted(anchors)

    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_all_nodes_anchored_reproduces_the_full_scan(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        full = pattern.find_matches(diagram)
        everything = frozenset(diagram.nodes)
        assert pattern.find_matches_anchored(diagram, everything) == full

    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_an_empty_anchor_set_finds_nothing(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        assert pattern.find_matches_anchored(diagram, frozenset()) == ()

    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_an_absent_anchor_finds_nothing(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        assert pattern.find_matches_anchored(diagram, frozenset({ABSENT_NODE_ID})) == ()

    def test_the_singleton_anchors_partition_the_full_scan(self) -> None:
        built = corpus()
        covered = 0
        for pattern in PATTERNS:
            for diagram in built.values():
                full = pattern.find_matches(diagram)
                union: set[str] = set()
                for node_id in sorted(diagram.nodes):
                    for match in pattern.find_matches_anchored(diagram, frozenset({node_id})):
                        union.add(repr(match))
                assert union == {repr(match) for match in full}
                covered += len(full)
        assert covered > 0


class TestAnchoredScanRaisesWhatTheFullScanRaises:
    @pytest.mark.parametrize("name", tuple(MALFORMED))
    def test_the_module_level_fourier_finder_raises_identically(self, name: str) -> None:
        diagram = MALFORMED[name]()
        expected = raised(lambda: find_fourier_matches(diagram))
        for anchors in (frozenset(), frozenset(diagram.nodes), frozenset({NodeId(0)})):
            got = raised(partial(find_fourier_matches, diagram, anchors=anchors))
            assert got == expected

    @pytest.mark.parametrize("name", tuple(MALFORMED))
    def test_the_fourier_pattern_raises_identically(self, name: str) -> None:
        diagram = MALFORMED[name]()
        pattern = FourierCancellationPattern()
        expected = raised(lambda: pattern.find_matches(diagram))
        for anchors in (frozenset(), frozenset(diagram.nodes)):
            got = raised(partial(pattern.find_matches_anchored, diagram, anchors))
            assert got == expected

    @pytest.mark.parametrize("name", tuple(MALFORMED))
    def test_the_malformed_diagrams_really_do_raise(self, name: str) -> None:
        diagram = MALFORMED[name]()
        with pytest.raises((KeyError, IndexError)):
            find_fourier_matches(diagram)


class TestFindMatchesIsOrderedByOrderKey:
    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_the_full_scan_is_sorted_by_order_key(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        full = pattern.find_matches(diagram)
        assert full == tuple(sorted(full, key=pattern.order_key))

    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_order_key_is_total(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        full = pattern.find_matches(diagram)
        keys = [repr(pattern.order_key(match)) for match in full]
        assert len(set(keys)) == len(keys), keys

    @pytest.mark.parametrize("pattern", PATTERNS, ids=PATTERN_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_an_anchored_scan_is_sorted_by_order_key(self, name: str, pattern: Pattern) -> None:
        diagram = corpus()[name]
        for anchors in anchor_sets(diagram):
            found = pattern.find_matches_anchored(diagram, anchors)
            assert found == tuple(sorted(found, key=pattern.order_key))

    def test_support_node_ids_is_ascending_deduplicated_and_present(self) -> None:
        built = corpus()
        seen = 0
        for pattern in PATTERNS:
            for diagram in built.values():
                for match in pattern.find_matches(diagram):
                    support = match.support_node_ids
                    assert support
                    assert list(support) == sorted(set(support))
                    assert all(node_id in diagram.nodes for node_id in support)
                    seen += 1
        assert seen > 0


class TestRematchEqualsSeedAfterALocalEdit:
    @pytest.mark.parametrize("edit", EDIT_NAMES)
    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_rematch_equals_seed(self, name: str, edit: str) -> None:
        before = corpus()[name]
        after = before.copy()
        if not EDITS[edit](after):
            pytest.skip(f"{edit} does not apply to {name}")
        assert_rematch_equals_seed(before, after)

    @pytest.mark.parametrize("edit", EDIT_NAMES)
    def test_every_edit_applies_somewhere_in_the_corpus(self, edit: str) -> None:
        applied = [name for name, d in corpus().items() if EDITS[edit](d)]
        assert applied, edit

    @pytest.mark.parametrize("edit", EDIT_NAMES)
    def test_an_applied_edit_changes_the_diagram(self, edit: str) -> None:
        changed = 0
        for diagram in corpus().values():
            before = _state(diagram)
            if EDITS[edit](diagram):
                assert _state(diagram) != before
                changed += 1
        assert changed > 0

    def test_rematch_equals_seed_in_place_on_a_mutated_diagram(self) -> None:
        diagram = fusion_chain(2, 4)
        incremental = matcher()
        incremental.seed(diagram)
        for edit in EDIT_NAMES:
            if not EDITS[edit](diagram):
                continue
            rematched = dict(incremental.rematch(diagram))
            seeded = dict(matcher().seed(diagram))
            assert rematched == seeded, edit

    @pytest.mark.parametrize(
        ("name", "mapping"),
        [("symbolic_dim", {"d": 3}), ("bang_boxed", {"n": 2}), ("nested_bang_boxed", {"n": 3})],
    )
    def test_rematch_equals_seed_after_substitute(self, name: str, mapping: dict[str, int]) -> None:
        before = corpus()[name]
        after = before.substitute(mapping)
        assert_rematch_equals_seed(before, after)

    @pytest.mark.parametrize("name", CORPUS_NAMES)
    def test_rematch_equals_seed_through_chained_engine_steps(self, name: str) -> None:
        current = corpus()[name]
        incremental = matcher()
        incremental.seed(current)
        for _ in range(4):
            step = _first_step(current)
            if step is None:
                break
            current = step
            rematched = dict(incremental.rematch(current))
            seeded = dict(matcher().seed(current))
            assert rematched == seeded

    def test_the_engine_step_sweep_really_rewrites_something(self) -> None:
        rewritten = [name for name, d in corpus().items() if _first_step(d) is not None]
        assert len(rewritten) >= 10


class TestIncrementalStats:
    def test_an_unchanged_rematch_retains_everything_and_rescans_nothing(self) -> None:
        diagram = fusion_chain(2, 4)
        incremental = matcher()
        seeded = incremental.seed(diagram)
        baseline_total = sum(len(found) for found in seeded.values())
        assert baseline_total > 0
        rematched = incremental.rematch(diagram)
        assert dict(rematched) == dict(seeded)
        stats = incremental.stats
        assert stats.dirty_nodes == 0
        assert stats.rescanned == 0
        assert stats.retained == baseline_total
        assert stats.full_rescans == 0
        assert stats.rematches == 1

    def test_a_bang_box_change_forces_a_full_rescan(self) -> None:
        diagram = bang_boxed(2)
        incremental = matcher()
        seeded = incremental.seed(diagram)
        assert sum(len(found) for found in seeded.values()) > 0
        assert _edit_change_bang_box_multiplicity(diagram)
        rematched = incremental.rematch(diagram)
        assert incremental.stats.full_rescans == 1
        assert incremental.stats.retained == 0
        assert dict(rematched) == dict(matcher().seed(diagram))

    def test_a_parameter_binding_forces_a_full_rescan(self) -> None:
        diagram = fusion_chain(2, 4)
        diagram.bind_parameter("k", 2)
        incremental = matcher()
        seeded = incremental.seed(diagram)
        assert sum(len(found) for found in seeded.values()) > 0
        diagram.bind_parameter("k", 3)
        rematched = incremental.rematch(diagram)
        assert incremental.stats.full_rescans == 1
        assert dict(rematched) == dict(matcher().seed(diagram))

    def test_a_local_change_does_not_force_a_full_rescan(self) -> None:
        diagram = fusion_chain(2, 4)
        incremental = matcher()
        incremental.seed(diagram)
        assert _edit_add_node(diagram)
        incremental.rematch(diagram)
        assert incremental.stats.full_rescans == 0
        assert incremental.stats.dirty_nodes > 0

    def test_repeated_matching_on_a_stable_region_hits_the_shared_cache(self) -> None:
        cache = MatchCache()
        incremental = IncrementalMatcher(PATTERNS, cache=cache)
        diagram = fusion_chain(2, 4)
        incremental.seed(diagram)
        first = cache.stats
        assert first.misses == len(PATTERNS)
        assert first.hits == 0
        for _ in range(3):
            incremental.seed(diagram)
        later = cache.stats
        assert later.misses == first.misses
        assert later.hits == 3 * len(PATTERNS)
        assert later.hit_rate > 0.0

    def test_repeated_pattern_lookups_raise_hits_and_not_misses(self) -> None:
        cache = MatchCache()
        pattern = FusionPattern()
        diagram = fusion_chain(2, 4)
        assert cache.matches(pattern, diagram)
        assert cache.stats.hits == 0
        assert cache.stats.misses == 1
        for index in range(1, 5):
            assert cache.matches(pattern, diagram) == pattern.find_matches(diagram)
            assert cache.stats.hits == index
            assert cache.stats.misses == 1

    def test_the_counters_accumulate_across_rematches(self) -> None:
        diagram = fusion_chain(2, 4)
        incremental = matcher()
        incremental.seed(diagram)
        for expected in (1, 2, 3):
            incremental.rematch(diagram)
            assert incremental.stats.rematches == expected


class TestAMatchWithoutSupportForcesAFullScan:
    def test_a_supportless_match_is_never_retained(self) -> None:
        pattern = _SupportlessPattern()
        incremental = IncrementalMatcher([pattern])
        diagram = fusion_chain(2, 4)
        seeded = incremental.seed(diagram)
        assert len(seeded[pattern_key(pattern)]) == len(diagram.nodes)
        assert _edit_add_node(diagram)
        rematched = incremental.rematch(diagram)
        assert rematched[pattern_key(pattern)] == pattern.find_matches(diagram)
        assert incremental.stats.retained == 0
        assert incremental.stats.rescanned == len(diagram.nodes)

    def test_a_supportless_match_leaks_no_attribute_error(self) -> None:
        pattern = _SupportlessPattern()
        incremental = IncrementalMatcher([pattern])
        diagram = fusion_chain(2, 4)
        incremental.seed(diagram)
        assert _edit_add_and_wire_node(diagram)
        incremental.rematch(diagram)
        assert _edit_remove_node(diagram)
        assert incremental.rematch(diagram)[pattern_key(pattern)] == pattern.find_matches(diagram)

    def test_the_supportless_pattern_really_has_no_support(self) -> None:
        pattern = _SupportlessPattern()
        (match,) = pattern.find_matches(cap_pairs(2, 1))[:1]
        assert not hasattr(match, "support_node_ids")


def _state(diagram: Diagram) -> tuple[object, ...]:
    """A diagram's mutable content as a comparable, deterministic tuple."""
    return (
        tuple(sorted(repr(diagram.nodes[node_id]) for node_id in diagram.nodes)),
        tuple(sorted(wire.sort_key() for wire in diagram.wires)),
        tuple(ref.sort_key() for ref in diagram.boundary_inputs),
        tuple(ref.sort_key() for ref in diagram.boundary_outputs),
        tuple(sorted(repr(diagram.bang_boxes[box_id]) for box_id in diagram.bang_boxes)),
        repr(diagram.scalar),
    )


def _first_step(diagram: Diagram) -> Diagram | None:
    """The diagram one normal-form rewrite on, or ``None`` when none applies."""
    for rule in normal_form_rules():
        matches = rule.pattern.find_matches(diagram)
        if matches:
            return apply(diagram, rule, matches[0]).diagram
    return None
