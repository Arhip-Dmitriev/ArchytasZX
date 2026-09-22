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

"""Establishes DimensionGuard's grammar and three-valued evaluation, and the guarded
Z-fusion rule's firing behaviour."""

from __future__ import annotations

from typing import ClassVar

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef, Wire
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import (
    FUSION_SIDE_CONDITIONS,
    FusionMatch,
    FusionPattern,
    FusionResolution,
    find_matches,
    resolve_fusion_match,
)
from archytaszx.rewrite.rule import (
    DimensionGuard,
    DimensionGuardKind,
    GuardOutcome,
    Quantifiers,
    RewriteDomainError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
    check_side_condition_coverage,
)
from archytaszx.rewrite.rules_library import (
    PRIME_DIMENSION_GUARDS,
    RULES,
    SPIDER_FUSION,
    Z_FUSION_PRIME_D,
    lookup_rule,
    z_fusion_prime_d_builder,
)

_GUARD_CONDITION = "dimension_guards_satisfied"

_VALUED_KINDS = (
    DimensionGuardKind.AT_LEAST,
    DimensionGuardKind.EQUALS,
    DimensionGuardKind.DIVISIBLE_BY,
)
_VALUELESS_KINDS = (DimensionGuardKind.PRIME, DimensionGuardKind.COMPOSITE)


def _pair(a_dim: Dim, b_dim: Dim | None = None) -> tuple[Diagram, Wire]:
    """A two-Z-spider diagram joined output-to-input, and the joining wire."""
    b_dim = a_dim if b_dim is None else b_dim
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[a_dim], output_dims=[a_dim])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[b_dim], output_dims=[b_dim])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(a_id, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
    return diagram, next(w for w in diagram.wires if w.a.node_id != w.b.node_id)


def _resolve_guarded(diagram: Diagram, wire: Wire) -> FusionResolution:
    """``resolve_fusion_match`` over ``wire`` under :data:`PRIME_DIMENSION_GUARDS`."""
    a_id, b_id = sorted((wire.a.node_id, wire.b.node_id))
    return resolve_fusion_match(diagram, a_id, b_id, wire, dimension_guards=PRIME_DIMENSION_GUARDS)


class TestDimensionGuardGrammar:
    """``value`` is required exactly for the kinds taking one, as a positive non-bool int."""

    @pytest.mark.parametrize("kind", _VALUED_KINDS)
    def test_missing_value_is_rejected(self, kind: DimensionGuardKind) -> None:
        with pytest.raises(RewriteGrammarError, match="requires a value"):
            DimensionGuard(kind)

    @pytest.mark.parametrize("kind", _VALUELESS_KINDS)
    def test_forbidden_value_is_rejected(self, kind: DimensionGuardKind) -> None:
        with pytest.raises(RewriteGrammarError, match="takes no value"):
            DimensionGuard(kind, 2)

    @pytest.mark.parametrize("kind", _VALUED_KINDS)
    @pytest.mark.parametrize("value", [0, -1, -7])
    def test_non_positive_value_is_rejected(self, kind: DimensionGuardKind, value: int) -> None:
        with pytest.raises(RewriteGrammarError, match=">= 1"):
            DimensionGuard(kind, value)

    @pytest.mark.parametrize("kind", _VALUED_KINDS)
    @pytest.mark.parametrize("value", [True, False])
    def test_bool_value_is_rejected(self, kind: DimensionGuardKind, value: bool) -> None:
        with pytest.raises(RewriteGrammarError, match="never a bool"):
            DimensionGuard(kind, value)

    def test_non_kind_is_rejected(self) -> None:
        with pytest.raises(RewriteGrammarError, match="must be a DimensionGuardKind"):
            DimensionGuard("prime")  # type: ignore[arg-type]

    @pytest.mark.parametrize("kind", _VALUELESS_KINDS)
    def test_valueless_kind_builds(self, kind: DimensionGuardKind) -> None:
        assert DimensionGuard(kind).value is None

    @pytest.mark.parametrize("kind", _VALUED_KINDS)
    def test_valued_kind_builds_and_prints_its_value(self, kind: DimensionGuardKind) -> None:
        assert "3" in str(DimensionGuard(kind, 3))


class TestDimensionGuardEvaluate:
    """The full table over ``d = 1..12``, plus the ``UNDECIDED`` cell."""

    _RANGE = range(1, 13)
    _PRIMES: ClassVar[frozenset[int]] = frozenset({2, 3, 5, 7, 11})
    _COMPOSITES: ClassVar[frozenset[int]] = frozenset({4, 6, 8, 9, 10, 12})

    @pytest.mark.parametrize("value", _RANGE)
    def test_prime(self, value: int) -> None:
        outcome = DimensionGuard(DimensionGuardKind.PRIME).evaluate(Dim.concrete(value))
        expected = GuardOutcome.PASSED if value in self._PRIMES else GuardOutcome.FAILED
        assert outcome is expected

    @pytest.mark.parametrize("value", _RANGE)
    def test_composite(self, value: int) -> None:
        outcome = DimensionGuard(DimensionGuardKind.COMPOSITE).evaluate(Dim.concrete(value))
        expected = GuardOutcome.PASSED if value in self._COMPOSITES else GuardOutcome.FAILED
        assert outcome is expected

    def test_one_is_neither_prime_nor_composite(self) -> None:
        one = Dim.concrete(1)
        assert DimensionGuard(DimensionGuardKind.PRIME).evaluate(one) is GuardOutcome.FAILED
        assert DimensionGuard(DimensionGuardKind.COMPOSITE).evaluate(one) is GuardOutcome.FAILED

    @pytest.mark.parametrize("value", _RANGE)
    def test_at_least(self, value: int) -> None:
        outcome = DimensionGuard(DimensionGuardKind.AT_LEAST, 5).evaluate(Dim.concrete(value))
        expected = GuardOutcome.PASSED if value >= 5 else GuardOutcome.FAILED
        assert outcome is expected

    @pytest.mark.parametrize("value", _RANGE)
    def test_equals(self, value: int) -> None:
        outcome = DimensionGuard(DimensionGuardKind.EQUALS, 5).evaluate(Dim.concrete(value))
        expected = GuardOutcome.PASSED if value == 5 else GuardOutcome.FAILED
        assert outcome is expected

    @pytest.mark.parametrize("value", _RANGE)
    def test_divisible_by(self, value: int) -> None:
        outcome = DimensionGuard(DimensionGuardKind.DIVISIBLE_BY, 3).evaluate(Dim.concrete(value))
        expected = GuardOutcome.PASSED if value % 3 == 0 else GuardOutcome.FAILED
        assert outcome is expected

    @pytest.mark.parametrize(
        "guard",
        [
            DimensionGuard(DimensionGuardKind.PRIME),
            DimensionGuard(DimensionGuardKind.COMPOSITE),
            DimensionGuard(DimensionGuardKind.AT_LEAST, 2),
            DimensionGuard(DimensionGuardKind.EQUALS, 2),
            DimensionGuard(DimensionGuardKind.DIVISIBLE_BY, 2),
        ],
    )
    def test_symbolic_and_absent_dims_are_undecided(self, guard: DimensionGuard) -> None:
        assert guard.evaluate(Dim.symbol("d")) is GuardOutcome.UNDECIDED
        assert guard.evaluate(Dim.symbol("d") * Dim.concrete(2)) is GuardOutcome.UNDECIDED
        assert guard.evaluate(None) is GuardOutcome.UNDECIDED

    def test_primality_agrees_with_sympy_over_the_range(self) -> None:
        guard = DimensionGuard(DimensionGuardKind.PRIME)
        for value in self._RANGE:
            passed = guard.evaluate(Dim.concrete(value)) is GuardOutcome.PASSED
            assert passed == bool(sp.isprime(value))


class TestGuardedRuleRegistration:
    """The guarded rule is a first-class registry entry declaring its own guards."""

    def test_lookup_and_registry(self) -> None:
        assert lookup_rule("z_fusion_prime_d") is Z_FUSION_PRIME_D
        assert RULES["z_fusion_prime_d"] is Z_FUSION_PRIME_D

    def test_declares_the_prime_guard_and_the_fusion_conditions(self) -> None:
        assert Z_FUSION_PRIME_D.dimension_guards == PRIME_DIMENSION_GUARDS
        assert Z_FUSION_PRIME_D.dimension_guards == (DimensionGuard(DimensionGuardKind.PRIME),)
        assert Z_FUSION_PRIME_D.side_conditions == FUSION_SIDE_CONDITIONS
        assert Z_FUSION_PRIME_D.scalar_introduced == SPIDER_FUSION.scalar_introduced

    def test_pattern_carries_the_same_guards(self) -> None:
        pattern = Z_FUSION_PRIME_D.pattern
        assert isinstance(pattern, FusionPattern)
        assert pattern.dimension_guards == PRIME_DIMENSION_GUARDS

    def test_default_pattern_declares_no_guards(self) -> None:
        assert FusionPattern().dimension_guards == ()

    def test_rejects_a_non_guard_in_dimension_guards(self) -> None:
        with pytest.raises(RewriteGrammarError, match="dimension_guards must be a tuple"):
            Rule(
                name="bad_guards",
                pattern=FusionPattern(),
                builder=SPIDER_FUSION.builder,
                side_conditions=FUSION_SIDE_CONDITIONS,
                quantifiers=Quantifiers(),
                scalar_introduced=SPIDER_FUSION.scalar_introduced,
                dimension_guards=("prime",),  # type: ignore[arg-type]
            )


class TestGuardedRuleFiresAtPrimeDimensions:
    @pytest.mark.parametrize("value", [2, 3, 5, 7])
    def test_pattern_finds_and_apply_succeeds(self, value: int) -> None:
        diagram, _wire = _pair(Dim.concrete(value))
        matches = Z_FUSION_PRIME_D.pattern.find_matches(diagram)
        assert len(matches) == 1
        result = apply(diagram, Z_FUSION_PRIME_D, matches[0])
        assert len(result.new_node_ids) == 1

    @pytest.mark.parametrize("value", [2, 3, 5, 7])
    def test_the_condition_names_the_guard_and_the_dim(self, value: int) -> None:
        diagram, wire = _pair(Dim.concrete(value))
        resolution = _resolve_guarded(diagram, wire)
        assert resolution.passed
        outcome = next(o for o in resolution.outcomes if o.name == _GUARD_CONDITION)
        assert outcome.passed and not outcome.deferred
        assert "prime" in outcome.detail
        assert str(value) in outcome.detail


class TestGuardedRuleRefusesCompositeDimensions:
    @pytest.mark.parametrize("value", [4, 6, 9])
    def test_pattern_returns_no_candidate(self, value: int) -> None:
        diagram, _wire = _pair(Dim.concrete(value))
        assert Z_FUSION_PRIME_D.pattern.find_matches(diagram) == ()
        # The unguarded pattern still matches the very same diagram.
        assert len(FusionPattern().find_matches(diagram)) == 1

    @pytest.mark.parametrize("value", [4, 6, 9])
    def test_direct_resolution_reports_the_condition_failed(self, value: int) -> None:
        diagram, wire = _pair(Dim.concrete(value))
        resolution = _resolve_guarded(diagram, wire)
        assert not resolution.passed
        assert resolution.shared_dim is None
        outcome = next(o for o in resolution.outcomes if o.name == _GUARD_CONDITION)
        assert not outcome.passed
        assert not outcome.deferred
        assert "do not hold" in outcome.detail
        # Every earlier condition passed: only the guard rejected this candidate.
        assert [o.name for o in resolution.outcomes if not o.passed] == [_GUARD_CONDITION]

    @pytest.mark.parametrize("value", [4, 6, 9])
    def test_the_builder_refuses_a_fabricated_match(self, value: int) -> None:
        diagram, wire = _pair(Dim.concrete(value))
        match = FusionMatch(
            a_id=min(wire.a.node_id, wire.b.node_id),
            b_id=max(wire.a.node_id, wire.b.node_id),
            wire=wire,
            shared_dim=Dim.concrete(value),
            side_condition_outcomes=tuple(
                SideConditionOutcome(condition.name, True, "fabricated")
                for condition in FUSION_SIDE_CONDITIONS
            ),
        )
        with pytest.raises(RewriteDomainError, match=_GUARD_CONDITION):
            z_fusion_prime_d_builder(diagram.copy(), match)


class TestUndecidedBlocksTheRewrite:
    """A non-concrete shared dimension blocks: ``passed`` False, ``deferred`` False."""

    def test_symbolic_dim_is_not_a_candidate(self) -> None:
        diagram, _wire = _pair(Dim.symbol("d"))
        assert Z_FUSION_PRIME_D.pattern.find_matches(diagram) == ()

    def test_symbolic_dim_reports_undecided_without_deferring(self) -> None:
        diagram, wire = _pair(Dim.symbol("d"))
        resolution = _resolve_guarded(diagram, wire)
        assert not resolution.passed
        outcome = next(o for o in resolution.outcomes if o.name == _GUARD_CONDITION)
        assert not outcome.passed
        assert not outcome.deferred, (
            "an undecidable guard is not an assumed dimension equality, so it must never be "
            "reported as deferred"
        )
        assert "cannot be decided" in outcome.detail
        assert "d" in outcome.detail


class TestUnguardedFusionStillFiresEverywhere:
    @pytest.mark.parametrize("dim", [Dim.concrete(2), Dim.concrete(4), Dim.symbol("d")])
    def test_the_new_condition_passes_vacuously(self, dim: Dim) -> None:
        diagram, _wire = _pair(dim)
        matches = find_matches(diagram)
        assert len(matches) == 1
        outcome = next(o for o in matches[0].side_condition_outcomes if o.name == _GUARD_CONDITION)
        assert outcome.passed
        assert not outcome.deferred
        assert outcome.detail == "no dimension guards declared"
        apply(diagram, SPIDER_FUSION, matches[0])

    def test_the_condition_is_declared_last(self) -> None:
        assert FUSION_SIDE_CONDITIONS[-1].name == _GUARD_CONDITION

    def test_every_outcome_covers_the_declared_conditions(self) -> None:
        diagram, _wire = _pair(Dim.concrete(3))
        (match,) = find_matches(diagram)
        assert [o.name for o in match.side_condition_outcomes] == [
            condition.name for condition in FUSION_SIDE_CONDITIONS
        ]


class TestCoverageDemandsTheNewOutcome:
    """A match omitting the new outcome is a coverage failure, not a silent pass."""

    def test_omitting_the_guard_outcome_is_a_domain_error(self) -> None:
        diagram, _wire = _pair(Dim.concrete(3))
        (real,) = find_matches(diagram)
        stripped = FusionMatch(
            a_id=real.a_id,
            b_id=real.b_id,
            wire=real.wire,
            shared_dim=real.shared_dim,
            side_condition_outcomes=tuple(
                o for o in real.side_condition_outcomes if o.name != _GUARD_CONDITION
            ),
            dimension_constraints=real.dimension_constraints,
            bindings=real.bindings,
        )
        with pytest.raises(RewriteDomainError, match=_GUARD_CONDITION):
            check_side_condition_coverage(stripped, FUSION_SIDE_CONDITIONS, "z_fusion_prime_d")
        with pytest.raises(RewriteDomainError, match=_GUARD_CONDITION):
            z_fusion_prime_d_builder(diagram.copy(), stripped)
