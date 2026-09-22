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

"""Tests for archytaszx.algebra.dimension.solve."""

from __future__ import annotations

import itertools
from typing import cast

import pytest

import archytaszx.algebra.dimension as dimension_module
from archytaszx.algebra.dimension import Dim, DimSymbolKey, UnifyStatus, solve

_A = Dim.symbol("a")
_B = Dim.symbol("b")
_D = Dim.symbol("d")
_E = Dim.symbol("e")
_F = Dim.symbol("f")
_G = Dim.symbol("g")
_P = Dim.symbol("p")
_Q = Dim.symbol("q")

# A system whose fixpoint provably needs more than one pass: pass one can only bind b, and
# the 2-vs-a*b pair stays undecided until that binding lands.
_TWO_PASS_SYSTEM: tuple[tuple[Dim, Dim], ...] = (
    (_A * _B, _A),
    (Dim.concrete(2), _A * _B),
)


class TestTrivialSystems:
    def test_empty_system_succeeds(self) -> None:
        result = solve([])
        assert result.is_success
        assert dict(result.bindings) == {}
        assert result.residual_pairs == ()
        assert result.exhausted is False

    def test_reflexive_pair_succeeds_without_bindings(self) -> None:
        result = solve([(_D, _D)])
        assert result.is_success
        assert dict(result.bindings) == {}

    def test_contradictory_concretes_fail(self) -> None:
        result = solve([(Dim.concrete(2), Dim.concrete(3))])
        assert result.is_failure


class TestMultiPassSystems:
    def test_two_pass_system_converges_to_success(self) -> None:
        result = solve(_TWO_PASS_SYSTEM)
        assert result.is_success
        assert dict(result.bindings) == {"a": Dim.concrete(2), "b": Dim.concrete(1)}
        assert result.residual_pairs == ()
        assert result.exhausted is False

    def test_chained_bindings_propagate_through_the_system(self) -> None:
        result = solve([(_D, _E), (_E, _F), (_F, Dim.concrete(3))])
        assert result.is_success
        assert dict(result.bindings) == {
            "d": Dim.concrete(3),
            "e": Dim.concrete(3),
            "f": Dim.concrete(3),
        }

    def test_a_product_constraint_resolves_after_its_factors(self) -> None:
        result = solve([(_D, Dim.concrete(2)), (_E, Dim.concrete(3)), (_F, _D * _E)])
        assert result.is_success
        assert dict(result.bindings)["f"] == Dim.concrete(6)

    def test_rebinding_to_an_incompatible_value_fails(self) -> None:
        result = solve([(_D, Dim.concrete(2)), (_D, Dim.concrete(3))])
        assert result.is_failure

    def test_contradiction_reached_only_through_propagation_fails(self) -> None:
        result = solve([(_D, _E), (_E, Dim.concrete(2)), (_D, Dim.concrete(5))])
        assert result.is_failure


class TestSymbolicBindings:
    def test_symbolic_binding_is_reported(self) -> None:
        result = solve([(_D, _E * _F)])
        assert result.is_success
        assert dict(result.bindings) == {"d": _E * _F}

    def test_symbolic_bindings_are_resolved_against_each_other(self) -> None:
        result = solve([(_D, _E * _F), (_E, _G)])
        assert result.is_success
        bindings = dict(result.bindings)
        assert bindings["e"] == _G
        assert bindings["d"] == _G * _F

    def test_bindings_are_a_fixpoint_of_themselves(self) -> None:
        systems: tuple[tuple[tuple[Dim, Dim], ...], ...] = (
            ((_D, _E * _F), (_E, _G)),
            ((_E, _D), (_D, _E * _F)),
            ((_D, _E * _F), (_F, _D * _G)),
            _TWO_PASS_SYSTEM,
        )
        for system in systems:
            result = solve(system)
            mapping = cast(dict[DimSymbolKey, Dim], dict(result.bindings))
            for name, value in result.bindings.items():
                assert name not in value.free_symbols
                assert value.rewrite(mapping) == value


class TestResidualsAndDeferral:
    def test_undecidable_pair_is_reported_as_a_residual(self) -> None:
        result = solve([(2 * _D, 3 * _E)])
        assert result.status is UnifyStatus.DEFERRED
        assert result.residual_pairs == ((2 * _D, 3 * _E),)
        assert result.exhausted is False

    def test_residuals_are_reported_in_reduced_form(self) -> None:
        result = solve([(_D * _E**_P, _D * _E**_Q)])
        assert result.is_deferred
        assert result.residual_pairs == ((_E**_P, _E**_Q),)

    def test_decided_pairs_coexist_with_residual_ones(self) -> None:
        result = solve([(_D, Dim.concrete(2)), (_A**_P, _A**_Q)])
        assert result.is_deferred
        assert dict(result.bindings) == {"d": Dim.concrete(2)}
        assert result.residual_pairs == ((_A**_P, _A**_Q),)

    def test_duplicate_residuals_are_reported_once(self) -> None:
        result = solve([(2 * _D, 3 * _E), (2 * _D, 3 * _E)])
        assert result.residual_pairs == ((2 * _D, 3 * _E),)


class TestOrderIndependence:
    @pytest.mark.parametrize(
        "system",
        [
            _TWO_PASS_SYSTEM,
            ((_D, _E), (_E, _F), (_F, Dim.concrete(3))),
            ((_D, Dim.concrete(2)), (_A**_P, _A**_Q)),
            ((_D, Dim.concrete(2)), (_D, Dim.concrete(3))),
        ],
    )
    def test_every_permutation_agrees(self, system: tuple[tuple[Dim, Dim], ...]) -> None:
        results = [solve(list(perm)) for perm in itertools.permutations(system)]
        first = results[0]
        for result in results[1:]:
            assert result.status is first.status
            assert dict(result.bindings) == dict(first.bindings)
            assert result.residual_pairs == first.residual_pairs
            assert result.exhausted is first.exhausted


class TestBudgetExhaustion:
    def test_exhaustion_is_flagged_and_reports_the_final_pass_residual(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dimension_module, "_MAX_SOLVE_PASSES", 1)
        result = solve(_TWO_PASS_SYSTEM)

        assert result.is_deferred
        assert result.exhausted is True
        assert result.residual_pairs, (
            "an exhausted result must report what the final pass actually left unresolved"
        )

    def test_max_passes_argument_overrides_the_module_budget(self) -> None:
        assert solve(_TWO_PASS_SYSTEM, max_passes=1).exhausted is True
        assert solve(_TWO_PASS_SYSTEM, max_passes=32).exhausted is False

    def test_converged_deferred_is_not_flagged_exhausted(self) -> None:
        result = solve([(_A**_P, _A**_Q)])
        assert result.is_deferred
        assert result.exhausted is False
