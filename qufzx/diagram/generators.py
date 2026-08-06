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

A :class:`GeneratorType` is a purely descriptive value object: it records, for a family
of nodes such as the Z spider or the X spider, what leg counts are permitted, how a
node's :class:`~qufzx.algebra.phase.PhaseVector` slot relates to its leg dimension, and
how dimensions are shared across a node's legs. It carries no formula, no matrix, and no
lambda -- the denotation of a generator (what tensor it actually stands for) is a leaf of
the evaluator and belongs to :mod:`qufzx.semantics.denote`, which arrives in Phase 4.
This module only ever answers "is this node's shape legal", never "what does this node
mean".

Dimension policy is deliberately an enum rather than a hardcoded assumption baked into
the validator. Z and X are ``ALL_LEGS_EQUAL``: every leg of the node -- input or output --
shares one :class:`~qufzx.algebra.dimension.Dim`. Phase 10 introduces generators (the
triangle, W, and dimension-connective generators) with genuinely mixed per-leg
dimensions; those will add new :class:`DimensionPolicy` members rather than requiring
:mod:`qufzx.diagram.validate` to special-case generator names.

Leg policy is likewise a value object (:class:`LegPolicy`) rather than a bare predicate,
so that a validation report can name *which* policy a node's leg count violates.

This module registers only ``"Z"`` and ``"X"``. No other generator, no bang box, and no
dimension connective is defined here -- those are later phases.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import MappingProxyType


class GeneratorError(Exception):
    """Base class for all errors raised by this module."""


class GeneratorDomainError(GeneratorError):
    """A value is outside the mathematical domain a generator policy requires.

    Raised for a negative minimum leg count and other malformed policy values.
    """


class GeneratorGrammarError(GeneratorError):
    """A lookup or registration falls outside the grammar this module accepts.

    Raised for an unknown generator name and for registering a duplicate name.
    """


class DimensionPolicy(enum.Enum):
    """How a generator type's leg dimensions relate to one another.

    ``ALL_LEGS_EQUAL`` is the only member needed for Z and X: every input and output
    port of the node carries one shared :class:`~qufzx.algebra.dimension.Dim`. This is
    an enum, not a boolean flag, so Phase 10's triangle, W, and dimension-connective
    generators -- which have genuinely mixed per-leg dimensions -- can add new members
    here without requiring :mod:`qufzx.diagram.validate` to be rewritten; the validator
    dispatches on the policy value rather than on the generator name.
    """

    ALL_LEGS_EQUAL = "all_legs_equal"


@dataclass(frozen=True, slots=True)
class LegPolicy:
    """How many input and output legs a generator type permits.

    ``min_inputs`` and ``min_outputs`` are inclusive lower bounds; ``None`` for
    ``max_inputs`` or ``max_outputs`` means unbounded. Z and X use the unbounded
    zero-or-more policy on both sides (a spider may have zero legs on a side, as the
    "A into B" worked example's state-preparation node does).
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

    ``TIED_TO_LEG_DIM`` is the only member needed for Z and X: the node's
    :class:`~qufzx.algebra.phase.PhaseVector`, when present, must be built over the same
    :class:`~qufzx.algebra.dimension.Dim` that :class:`DimensionPolicy.ALL_LEGS_EQUAL`
    assigns to every leg. ``NONE`` marks a phase-free generator (no later phase in this
    plan currently needs it for Z or X, but the member exists so a future phase-free
    generator does not have to invent a sentinel).
    """

    TIED_TO_LEG_DIM = "tied_to_leg_dim"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class GeneratorType:
    """Descriptive policy metadata for one generator family, e.g. the Z spider.

    Purely a value object: leg policy, phase schema, and dimension policy. It records no
    denotation -- see the module docstring for why that belongs to Phase 4's
    :mod:`qufzx.semantics.denote` instead.
    """

    name: str
    leg_policy: LegPolicy
    phase_schema: PhaseSchema
    dimension_policy: DimensionPolicy

    def __post_init__(self) -> None:
        """Validate that ``name`` is a non-empty string."""
        if not isinstance(self.name, str) or not self.name:
            raise GeneratorGrammarError(
                f"generator name must be a non-empty str, got {self.name!r}"
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

    def names(self) -> frozenset[str]:
        """The set of all registered generator type names."""
        return frozenset(self._types)

    def all_types(self) -> MappingProxyType[str, GeneratorType]:
        """A read-only view of the full name -> GeneratorType registry."""
        return MappingProxyType(dict(self._types))


REGISTRY = GeneratorRegistry()
"""The module-level registry, pre-populated with Z and X."""

Z_SPIDER = GeneratorType(
    name="Z",
    leg_policy=LegPolicy(),
    phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The qudit Z spider: any number of legs, phase vector tied to the shared leg dim."""

X_SPIDER = GeneratorType(
    name="X",
    leg_policy=LegPolicy(),
    phase_schema=PhaseSchema.TIED_TO_LEG_DIM,
    dimension_policy=DimensionPolicy.ALL_LEGS_EQUAL,
)
"""The qudit X spider: any number of legs, phase vector tied to the shared leg dim."""

REGISTRY.register(Z_SPIDER)
REGISTRY.register(X_SPIDER)
