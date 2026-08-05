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

"""Tests for qufzx.diagram.generators: the Phase 3 generator-type registry."""

import pytest

from qufzx.diagram.generators import (
    REGISTRY,
    X_SPIDER,
    Z_SPIDER,
    DimensionPolicy,
    GeneratorDomainError,
    GeneratorGrammarError,
    GeneratorRegistry,
    GeneratorType,
    LegPolicy,
    PhaseSchema,
)


class TestLegPolicy:
    def test_default_allows_any_leg_count(self) -> None:
        policy = LegPolicy()
        assert policy.allows(0, 0)
        assert policy.allows(0, 2)
        assert policy.allows(5, 5)

    def test_bounded_policy_rejects_out_of_range(self) -> None:
        policy = LegPolicy(min_inputs=1, max_inputs=1, min_outputs=0, max_outputs=2)
        assert policy.allows(1, 0)
        assert policy.allows(1, 2)
        assert not policy.allows(0, 0)
        assert not policy.allows(2, 0)
        assert not policy.allows(1, 3)

    def test_negative_min_rejected(self) -> None:
        with pytest.raises(GeneratorDomainError):
            LegPolicy(min_inputs=-1)

    def test_max_below_min_rejected(self) -> None:
        with pytest.raises(GeneratorDomainError):
            LegPolicy(min_inputs=3, max_inputs=1)


class TestZAndXRegistration:
    def test_z_and_x_are_registered(self) -> None:
        assert REGISTRY.names() == frozenset({"Z", "X"})

    def test_get_returns_registered_types(self) -> None:
        assert REGISTRY.get("Z") is Z_SPIDER
        assert REGISTRY.get("X") is X_SPIDER

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(GeneratorGrammarError):
            REGISTRY.get("W")

    def test_z_and_x_allow_any_leg_count_including_zero(self) -> None:
        for gen in (Z_SPIDER, X_SPIDER):
            assert gen.leg_policy.allows(0, 0)
            assert gen.leg_policy.allows(0, 2)
            assert gen.leg_policy.allows(1, 2)

    def test_z_and_x_use_all_legs_equal_and_tied_phase_schema(self) -> None:
        for gen in (Z_SPIDER, X_SPIDER):
            assert gen.dimension_policy is DimensionPolicy.ALL_LEGS_EQUAL
            assert gen.phase_schema is PhaseSchema.TIED_TO_LEG_DIM

    def test_all_types_view_contains_both(self) -> None:
        all_types = REGISTRY.all_types()
        assert set(all_types) == {"Z", "X"}


class TestRegistryDuplicates:
    def test_duplicate_name_rejected(self) -> None:
        registry = GeneratorRegistry()
        registry.register(
            GeneratorType(
                name="Z",
                leg_policy=LegPolicy(),
                phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
                dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
            )
        )
        with pytest.raises(GeneratorGrammarError):
            registry.register(
                GeneratorType(
                    name="Z",
                    leg_policy=LegPolicy(),
                    phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
                    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
                )
            )

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(GeneratorGrammarError):
            GeneratorType(
                name="",
                leg_policy=LegPolicy(),
                phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
                dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
            )
