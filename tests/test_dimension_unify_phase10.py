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

"""Tests for the Phase 10 Dim.unify decision procedure and Dim.rewrite."""

from __future__ import annotations

import itertools
import random
from typing import cast

import pytest

from archytaszx.algebra.dimension import (
    Dim,
    DimensionDomainError,
    DimensionError,
    DimensionGrammarError,
    DimSubstituteValue,
    DimSymbolKey,
    UnifyStatus,
)

_D = Dim.symbol("d")
_E = Dim.symbol("e")
_F = Dim.symbol("f")
_D1 = Dim.symbol("d1")
_D2 = Dim.symbol("d2")
_P = Dim.symbol("p")
_Q = Dim.symbol("q")

_DIM_NAMES = ("d", "e", "f")
_EXPONENT_NAMES = ("p", "q")
_DIM_RANGE = (1, 2, 3, 4)
_EXPONENT_RANGE = (0, 1, 2)


def _assignments(dim_names: list[str], exponent_names: list[str]) -> list[dict[str, int]]:
    """Every assignment of the given symbols over the small brute-force ranges."""
    names = dim_names + exponent_names
    ranges = [_DIM_RANGE] * len(dim_names) + [_EXPONENT_RANGE] * len(exponent_names)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*ranges)]


def _split_symbols(*dims: Dim) -> tuple[list[str], list[str]]:
    """The dimension-symbol names and exponent-symbol names free in the given Dims."""
    dim_names: set[str] = set()
    exponent_names: set[str] = set()
    for dim in dims:
        for symbol in dim.to_sympy().free_symbols:
            name = str(symbol.name)
            if symbol.assumptions0.get("positive"):
                dim_names.add(name)
            else:
                exponent_names.add(name)
    return sorted(dim_names), sorted(exponent_names)


def _has_solution(left: Dim, right: Dim) -> bool:
    """Brute-force search for an assignment making both sides equal."""
    dim_names, exponent_names = _split_symbols(left, right)
    for assignment in _assignments(dim_names, exponent_names):
        typed = cast(dict[DimSymbolKey, DimSubstituteValue], dict(assignment))
        if left.substitute(typed) == right.substitute(typed):
            return True
    return False


def _random_dim(rng: random.Random, depth: int = 0) -> Dim:
    """A random Dim built from the small symbol palette, concretes, products, and powers."""
    choices = ("concrete", "symbol") if depth >= 2 else ("concrete", "symbol", "product", "power")
    kind = rng.choice(choices)
    if kind == "concrete":
        return Dim.concrete(rng.choice((1, 2, 3, 4, 6, 8)))
    if kind == "symbol":
        return Dim.symbol(rng.choice(_DIM_NAMES))
    if kind == "product":
        return _random_dim(rng, depth + 1) * _random_dim(rng, depth + 1)
    base = _random_dim(rng, depth + 1)
    exponent: Dim | int = (
        Dim.symbol(rng.choice(_EXPONENT_NAMES)) if rng.random() < 0.5 else rng.choice((0, 1, 2, 3))
    )
    try:
        return base**exponent
    except DimensionError:
        # A power of a power can leave the exponent grammar; the base alone stands in.
        return base


class TestAcceptanceTable:
    def test_shared_factor_cancels(self) -> None:
        result = (_D * _E).unify(_D * _F)
        assert result.is_success
        assert dict(result.bindings) in ({"e": _F}, {"f": _E})

    def test_residual_factor_is_forced_to_one(self) -> None:
        result = _D.unify(_D * _E)
        assert result.is_success
        assert dict(result.bindings) == {"e": Dim.concrete(1)}

    def test_exact_square_root_binds(self) -> None:
        result = (_D**2).unify(Dim.concrete(4))
        assert result.is_success
        assert dict(result.bindings) == {"d": Dim.concrete(2)}

    def test_inexact_square_root_fails(self) -> None:
        assert (_D**2).unify(Dim.concrete(6)).is_failure

    def test_exact_cube_root_binds(self) -> None:
        result = (_D**3).unify(Dim.concrete(8))
        assert result.is_success
        assert dict(result.bindings) == {"d": Dim.concrete(2)}

    def test_factorization_of_a_concrete_is_deferred_not_failed(self) -> None:
        result = (_D1 * _D2).unify(Dim.concrete(6))
        assert result.status is UnifyStatus.DEFERRED
        assert result.constraints == ((_D1 * _D2, Dim.concrete(6)),)

    def test_lone_symbol_binds_to_a_symbolic_product(self) -> None:
        result = _D.unify(_D1 * _D2)
        assert result.is_success
        assert dict(result.bindings) == {"d": _D1 * _D2}

    def test_symbol_against_its_own_symbolic_power_is_deferred(self) -> None:
        result = _D.unify(_D**_P)
        assert result.is_deferred
        assert result.constraints == ((_D, _D**_P),)

    def test_two_symbolic_powers_of_one_base_are_deferred(self) -> None:
        result = (_D**_P).unify(_D**_Q)
        assert result.is_deferred

    def test_distinct_concrete_coefficients_are_deferred(self) -> None:
        result = (2 * _D).unify(3 * _E)
        assert result.is_deferred
        assert result.constraints == ((2 * _D, 3 * _E),)

    def test_product_equal_to_one_binds_every_factor(self) -> None:
        result = (_D * _E).unify(Dim.concrete(1))
        assert result.is_success
        assert dict(result.bindings) == {"d": Dim.concrete(1), "e": Dim.concrete(1)}

    def test_symbolic_power_equal_to_one_is_deferred(self) -> None:
        result = (_D**_P).unify(Dim.concrete(1))
        assert result.is_deferred

    def test_concrete_coefficient_against_one_fails(self) -> None:
        assert (2 * _D).unify(Dim.concrete(1)).is_failure

    def test_constraints_are_reported_in_reduced_form(self) -> None:
        result = (_D * _E**_P).unify(_D * _E**_Q)
        assert result.is_deferred
        assert result.constraints == ((_E**_P, _E**_Q),)

    def test_non_dim_operand_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            _D.unify(cast(Dim, 2))


class TestRewrite:
    def test_symbolic_value_is_substituted(self) -> None:
        assert _D.rewrite({"d": _D1 * _D2}) == _D1 * _D2

    def test_reaches_into_a_product(self) -> None:
        assert (_D * _E).rewrite({"d": _E}) == _E**2

    def test_partial_mapping_leaves_other_symbols_alone(self) -> None:
        assert (_D * _E).rewrite({"d": Dim.concrete(2)}) == 2 * _E

    def test_empty_mapping_returns_an_equal_dim(self) -> None:
        assert (_D * _E).rewrite({}) == _D * _E

    def test_bare_symbol_dim_key_accepted(self) -> None:
        assert _D.rewrite({_D: Dim.concrete(3)}) == Dim.concrete(3)

    def test_non_symbol_key_rejected(self) -> None:
        with pytest.raises(DimensionGrammarError):
            _D.rewrite({cast(DimSymbolKey, _D * _E): Dim.concrete(2)})

    def test_int_value_rejected(self) -> None:
        with pytest.raises(TypeError):
            _D.rewrite({"d": cast(Dim, 2)})

    def test_exponent_symbol_takes_a_concrete_value(self) -> None:
        assert (_D**_P).rewrite({"p": Dim.concrete(2)}) == _D**2

    def test_exponent_symbol_rejects_a_compound_value(self) -> None:
        with pytest.raises(DimensionGrammarError):
            (_D**_P).rewrite({"p": _D1 * _D2})

    def test_original_is_unchanged(self) -> None:
        expression = _D * _E
        expression.rewrite({"d": Dim.concrete(2)})
        assert expression == _D * _E

    def test_substitute_still_rejects_symbolic_values(self) -> None:
        with pytest.raises(DimensionDomainError):
            _D.substitute({"d": _D1 * _D2})


class TestSoundnessSweep:
    """Brute-force cross-check of both soundness directions over random equations."""

    @pytest.mark.parametrize("seed", range(200))
    def test_success_bindings_really_solve_the_equation(self, seed: int) -> None:
        rng = random.Random(seed)
        left, right = _random_dim(rng), _random_dim(rng)
        result = left.unify(right)
        if not result.is_success:
            return
        bindings = cast(dict[DimSymbolKey, Dim], dict(result.bindings))
        bound_left = left.rewrite(bindings)
        bound_right = right.rewrite(bindings)
        dim_names, exponent_names = _split_symbols(bound_left, bound_right)
        for assignment in _assignments(dim_names, exponent_names):
            typed = cast(dict[DimSymbolKey, DimSubstituteValue], dict(assignment))
            assert bound_left.substitute(typed) == bound_right.substitute(typed)

    @pytest.mark.parametrize("seed", range(200))
    def test_failure_means_no_solution_in_range(self, seed: int) -> None:
        rng = random.Random(1000 + seed)
        left, right = _random_dim(rng), _random_dim(rng)
        result = left.unify(right)
        if not result.is_failure:
            return
        assert not _has_solution(left, right)

    @pytest.mark.parametrize("seed", range(200))
    def test_every_verdict_is_one_of_the_three(self, seed: int) -> None:
        rng = random.Random(5000 + seed)
        left, right = _random_dim(rng), _random_dim(rng)
        result = left.unify(right)
        assert [result.is_success, result.is_failure, result.is_deferred].count(True) == 1
