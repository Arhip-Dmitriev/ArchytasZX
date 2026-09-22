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

"""Registry of generator types and their leg, phase, and dimension policies.

A :class:`GeneratorType` is a descriptive value object: it records, for a family of nodes
such as the Z spider, what leg counts are permitted, how a node's
:class:`~archytaszx.algebra.phase.PhaseVector` slot relates to its leg dimension, and how
dimensions are shared across a node's legs. It carries no formula, no matrix and no lambda;
a generator's denotation lives in :mod:`archytaszx.semantics.denote`. This module answers
"is this node's shape legal", never "what does this node mean".

Dimension policy is an enum the validator dispatches on, never a generator-name special
case. ``ALL_LEGS_EQUAL``: every leg shares one
:class:`~archytaszx.algebra.dimension.Dim`. ``PRODUCT_OF_LEGS_EQUAL``: the product of the input
dims equals the product of the output dims, an empty side having product ``Dim.concrete(1)``.

Leg policy is likewise a value object (:class:`LegPolicy`), so a validation report can name
*which* policy a node's leg count violates.

Registered here: ``"Z"``, ``"X"``, ``"F"``, ``"T"``, ``"Ti"``, ``"W"``, ``"B"``, ``"S"``. No
bang box is defined here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import MappingProxyType


class GeneratorError(Exception):
    """Base class for all errors raised by this module."""


class GeneratorDomainError(GeneratorError):
    """A value is outside the mathematical domain a generator policy requires.

    Raised for a negative minimum leg count, and for a maximum leg count below its
    corresponding minimum.
    """


class GeneratorGrammarError(GeneratorError):
    """A lookup or registration falls outside the grammar this module accepts.

    Raised for an unknown generator name and for registering a duplicate name.
    """


class DimensionPolicy(enum.Enum):
    """How a generator type's leg dimensions relate to one another.

    ``ALL_LEGS_EQUAL``: every input and output port carries one shared
    :class:`~archytaszx.algebra.dimension.Dim`. ``PRODUCT_OF_LEGS_EQUAL``: the product of the
    input dims equals the product of the output dims.
    """

    ALL_LEGS_EQUAL = "all_legs_equal"
    PRODUCT_OF_LEGS_EQUAL = "product_of_legs_equal"


@dataclass(frozen=True, slots=True)
class LegPolicy:
    """How many input and output legs a generator type permits.

    ``min_inputs`` and ``min_outputs`` are inclusive lower bounds; ``None`` for
    ``max_inputs`` or ``max_outputs`` means unbounded.
    """

    min_inputs: int = 0
    max_inputs: int | None = None
    min_outputs: int = 0
    max_outputs: int | None = None

    def __post_init__(self) -> None:
        """Validate that the bounds are non-negative integers with min <= max."""
        for name, value in (
            ("min_inputs", self.min_inputs),
            ("min_outputs", self.min_outputs),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise GeneratorGrammarError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise GeneratorDomainError(f"{name} must be >= 0, got {value}")
        for name, bound, value in (
            ("max_inputs", self.max_inputs, self.min_inputs),
            ("max_outputs", self.max_outputs, self.min_outputs),
        ):
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, int):
                raise GeneratorGrammarError(f"{name} must be an int or None, got {bound!r}")
            if bound < value:
                raise GeneratorDomainError(
                    f"{name} ({bound}) must be >= corresponding min ({value})"
                )

    def allows(self, num_inputs: int, num_outputs: int) -> bool:
        """True iff the given input and output leg counts satisfy this policy."""
        if num_inputs < self.min_inputs or num_outputs < self.min_outputs:
            return False
        if self.max_inputs is not None and num_inputs > self.max_inputs:
            return False
        if self.max_outputs is None:
            return True
        return num_outputs <= self.max_outputs


class PhaseSchema(enum.Enum):
    """How a generator type's phase slot relates to its leg dimension.

    ``TIED_TO_LEG_DIM``: the node's :class:`~archytaszx.algebra.phase.PhaseVector`, when
    present, is built over the same :class:`~archytaszx.algebra.dimension.Dim` every leg
    carries. ``NONE`` marks a phase-free generator.
    """

    TIED_TO_LEG_DIM = "tied_to_leg_dim"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class GeneratorType:
    """Descriptive policy metadata for one generator family, e.g. the Z spider.

    A value object holding a leg policy, a phase schema and a dimension policy. It records
    no denotation; see :mod:`archytaszx.semantics.denote`.
    """

    name: str
    leg_policy: LegPolicy
    phase_schema: PhaseSchema
    dimension_policy: DimensionPolicy

    def __post_init__(self) -> None:
        """Validate every field's type, the same way every other value object here does."""
        if not isinstance(self.name, str) or not self.name:
            raise GeneratorGrammarError(
                f"generator name must be a non-empty str, got {self.name!r}"
            )
        if not isinstance(self.leg_policy, LegPolicy):
            raise GeneratorGrammarError(
                f"generator {self.name!r}: leg_policy must be a LegPolicy, "
                f"got {type(self.leg_policy).__name__}"
            )
        if not isinstance(self.phase_schema, PhaseSchema):
            raise GeneratorGrammarError(
                f"generator {self.name!r}: phase_schema must be a PhaseSchema, "
                f"got {type(self.phase_schema).__name__}"
            )
        if not isinstance(self.dimension_policy, DimensionPolicy):
            raise GeneratorGrammarError(
                f"generator {self.name!r}: dimension_policy must be a DimensionPolicy, "
                f"got {type(self.dimension_policy).__name__}"
            )


class GeneratorRegistry:
    """A name -> :class:`GeneratorType` registry with duplicate and unknown-name checks."""

    __slots__ = ("_types",)
    _types: dict[str, GeneratorType]

    def __init__(self) -> None:
        """Build an empty registry."""
        self._types = {}

    def register(self, generator_type: GeneratorType) -> None:
        """Register a new generator type. Raises GeneratorGrammarError on a duplicate name."""
        if generator_type.name in self._types:
            raise GeneratorGrammarError(
                f"generator type {generator_type.name!r} already registered"
            )
        self._types[generator_type.name] = generator_type

    def get(self, name: str) -> GeneratorType:
        """Look up a registered generator type by name. Raises GeneratorGrammarError if unknown."""
        try:
            return self._types[name]
        except KeyError:
            raise GeneratorGrammarError(
                f"unknown generator type {name!r}; registered names: {sorted(self._types)}"
            ) from None

    def is_registered(self, generator_type: GeneratorType) -> bool:
        """Whether ``generator_type``'s name resolves here to a type equal to it.

        False for a name this registry does not carry, and for one it carries under a
        type whose other fields differ.
        """
        registered = self._types.get(generator_type.name)
        return registered is not None and registered == generator_type

    def names(self) -> frozenset[str]:
        """The set of all registered generator type names."""
        return frozenset(self._types)

    def all_types(self) -> MappingProxyType[str, GeneratorType]:
        """A read-only view of the full name -> GeneratorType registry."""
        return MappingProxyType(dict(self._types))


REGISTRY = GeneratorRegistry()
"""The module-level registry, pre-populated with every generator type this module defines."""

Z_SPIDER = GeneratorType(
    name="Z",
    leg_policy=LegPolicy(),
    phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The qudit Z spider: any number of legs, phase vector tied to the shared leg dim."""

FOURIER_BOX = GeneratorType(
    name="F",
    leg_policy=LegPolicy(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    phase_schema=PhaseSchema.NONE,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The Fourier box: the unitary DFT on one wire, F[j][k] = omega_d^{j*k} / sqrt(d)."""


X_SPIDER = GeneratorType(
    name="X",
    leg_policy=LegPolicy(),
    phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The qudit X spider: any number of legs, phase vector tied to the shared leg dim."""

TRIANGLE = GeneratorType(
    name="T",
    leg_policy=LegPolicy(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    phase_schema=PhaseSchema.NONE,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The generalised triangle: one in, one out, T[out][in] = 1 iff out == 0 or out == in."""

TRIANGLE_INVERSE = GeneratorType(
    name="Ti",
    leg_policy=LegPolicy(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    phase_schema=PhaseSchema.NONE,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The exact inverse of :data:`TRIANGLE`: one in, one out."""

W_NODE = GeneratorType(
    name="W",
    leg_policy=LegPolicy(min_inputs=1, max_inputs=1, min_outputs=2, max_outputs=2),
    phase_schema=PhaseSchema.NONE,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The W node: one in, two out, the qudit generalisation of |00><0| + (|01> + |10>)<1|."""

DIM_BINDER = GeneratorType(
    name="B",
    leg_policy=LegPolicy(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    phase_schema=PhaseSchema.NONE,
    dimension_policy=DimensionPolicy.PRODUCT_OF_LEGS_EQUAL,
)
"""The dimension binder: two inputs of dims s, t into one output of dim s*t."""

DIM_SPLITTER = GeneratorType(
    name="S",
    leg_policy=LegPolicy(min_inputs=1, max_inputs=1, min_outputs=2, max_outputs=2),
    phase_schema=PhaseSchema.NONE,
    dimension_policy=DimensionPolicy.PRODUCT_OF_LEGS_EQUAL,
)
"""The dimension splitter: one input of dim s*t into two outputs of dims s, t."""

REGISTRY.register(Z_SPIDER)
REGISTRY.register(X_SPIDER)
REGISTRY.register(FOURIER_BOX)
REGISTRY.register(TRIANGLE)
REGISTRY.register(TRIANGLE_INVERSE)
REGISTRY.register(W_NODE)
REGISTRY.register(DIM_BINDER)
REGISTRY.register(DIM_SPLITTER)
