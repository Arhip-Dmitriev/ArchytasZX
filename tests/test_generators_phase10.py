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

"""Tests for the Phase 10 generator registrations and the PRODUCT_OF_LEGS_EQUAL policy."""

from __future__ import annotations

import pytest

from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    REGISTRY,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    DimensionPolicy,
    GeneratorType,
    PhaseSchema,
)

_NEW = (TRIANGLE, TRIANGLE_INVERSE, W_NODE, DIM_BINDER, DIM_SPLITTER)


class TestRegistryMembership:
    def test_every_new_name_is_registered(self) -> None:
        assert {"T", "Ti", "W", "B", "S"} <= REGISTRY.names()

    def test_the_registry_holds_exactly_these_names(self) -> None:
        """Pins the whole registry, so an unreviewed tenth generator fails here."""
        assert REGISTRY.names() == frozenset({"Z", "X", "F", "T", "Ti", "W", "B", "S"})
        assert set(REGISTRY.all_types()) == {"Z", "X", "F", "T", "Ti", "W", "B", "S"}

    @pytest.mark.parametrize("gen", _NEW, ids=lambda g: g.name)
    def test_get_returns_the_module_constant(self, gen: GeneratorType) -> None:
        assert REGISTRY.get(gen.name) is gen

    @pytest.mark.parametrize("gen", _NEW, ids=lambda g: g.name)
    def test_is_registered_agrees(self, gen: GeneratorType) -> None:
        assert REGISTRY.is_registered(gen)


class TestPolicies:
    @pytest.mark.parametrize("gen", _NEW, ids=lambda g: g.name)
    def test_all_new_generators_are_phase_free(self, gen: GeneratorType) -> None:
        assert gen.phase_schema is PhaseSchema.NONE

    @pytest.mark.parametrize("gen", (TRIANGLE, TRIANGLE_INVERSE, W_NODE), ids=lambda g: g.name)
    def test_triangle_and_w_share_one_dimension(self, gen: GeneratorType) -> None:
        assert gen.dimension_policy is DimensionPolicy.ALL_LEGS_EQUAL

    @pytest.mark.parametrize("gen", (DIM_BINDER, DIM_SPLITTER), ids=lambda g: g.name)
    def test_connectives_use_the_product_policy(self, gen: GeneratorType) -> None:
        assert gen.dimension_policy is DimensionPolicy.PRODUCT_OF_LEGS_EQUAL

    def test_product_of_legs_equal_member_value(self) -> None:
        assert DimensionPolicy.PRODUCT_OF_LEGS_EQUAL.value == "product_of_legs_equal"


class TestLegPolicies:
    @pytest.mark.parametrize("gen", (TRIANGLE, TRIANGLE_INVERSE), ids=lambda g: g.name)
    def test_triangles_are_one_in_one_out(self, gen: GeneratorType) -> None:
        policy = gen.leg_policy
        assert (policy.min_inputs, policy.max_inputs) == (1, 1)
        assert (policy.min_outputs, policy.max_outputs) == (1, 1)
        assert policy.allows(1, 1)
        assert not policy.allows(1, 2)
        assert not policy.allows(2, 1)

    @pytest.mark.parametrize("gen", (W_NODE, DIM_SPLITTER), ids=lambda g: g.name)
    def test_one_in_two_out(self, gen: GeneratorType) -> None:
        policy = gen.leg_policy
        assert (policy.min_inputs, policy.max_inputs) == (1, 1)
        assert (policy.min_outputs, policy.max_outputs) == (2, 2)
        assert policy.allows(1, 2)
        assert not policy.allows(1, 1)
        assert not policy.allows(1, 3)

    def test_binder_is_two_in_one_out(self) -> None:
        policy = DIM_BINDER.leg_policy
        assert (policy.min_inputs, policy.max_inputs) == (2, 2)
        assert (policy.min_outputs, policy.max_outputs) == (1, 1)
        assert policy.allows(2, 1)
        assert not policy.allows(1, 1)
