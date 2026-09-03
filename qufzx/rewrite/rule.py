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

"""Rule: a left-hand pattern, right-hand builder, side conditions, quantifiers, and exact scalar.

:class:`Rule` is a frozen value object; it performs no graph surgery. :class:`Pattern` is the
abstract seam later phases implement to add new rewrite shapes. :class:`Match` is a
``typing.Protocol``, so each pattern defines its own match type carrying whatever location
data it needs. :class:`BuildResult` is the generic engine/builder contract:
:func:`qufzx.rewrite.engine.apply` splices from its fields alone, knowing nothing
rule-specific.

:class:`Quantifiers` is declared metadata only in this phase; Phase 7 and Phase 10 make it
checkable.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, NodeId, PortRef, Wire


class RewriteError(Exception):
    """Base class for all errors raised anywhere in :mod:`qufzx.rewrite`."""


class RewriteDomainError(RewriteError):
    """A value or state is outside the mathematical domain a rewrite operation requires.

    Raised for: a match whose ``side_condition_outcomes`` do not exactly cover, or do not all
    pass, its rule's declared conditions; a builder whose ``scalar_introduced`` disagrees with
    its rule's; a match whose ``shared_dim``, ``bindings``, ``dimension_constraints`` or
    ``side_condition_outcomes`` disagree with a fresh
    :func:`~qufzx.rewrite.match.resolve_fusion_match`, or that re-resolves as a non-match; a
    phase whose entries fall outside the shared dimension once reattached; a wire or boundary
    entry naming a consumed port absent from the builder's ``port_mapping``; and a rewrite
    that introduces a hard validation issue the input did not carry.
    """


class RewriteGrammarError(RewriteError):
    """A rewrite request is malformed independent of mathematical domain.

    Raised for: a value object built outside its own contract (a :class:`ConstraintSource`
    whose kind and reference disagree, a :class:`DimensionConstraint` whose ``bound_here`` is
    ill-shaped or disagrees with its outcome, a :class:`Rule` field of the wrong type or one
    whose ``side_conditions`` disagree with its builder's); a match or wire that does not
    belong to the diagram it is applied against; a :class:`BuildResult` naming a node, port or
    wire the working diagram does not have, repeating an id, or whose ``port_mapping`` is not
    injective; and an unknown rule name at
    :func:`~qufzx.rewrite.rules_library.lookup_rule`.
    """


@dataclass(frozen=True, slots=True)
class SideCondition:
    """One named entry in a :class:`Pattern`'s declared condition list.

    Metadata about the pattern, declared once (see ``FUSION_SIDE_CONDITIONS`` in
    :mod:`qufzx.rewrite.match`); the per-candidate result lives in
    :class:`SideConditionOutcome`. Not every entry is a decision: a pattern may declare a
    condition it always reports True, as a structural fact for the certificate to carry.
    """

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class SideConditionOutcome:
    """The result of one named :class:`SideCondition` against one match candidate.

    ``deferred`` is True for an outcome that passed on an assumed rather than verified
    dimension equality, mirroring
    :class:`qufzx.diagram.validate.ValidationIssue`'s convention.
    """

    name: str
    passed: bool
    detail: str
    deferred: bool = False


class ConstraintSourceKind(enum.Enum):
    """Which kind of check produced a :class:`DimensionConstraint`."""

    CONNECTING_PAIR = "connecting_pair"
    """The two legs the consumed wire joins, related to each other. The only kind whose
    :attr:`DimensionConstraint.equal_to` is another leg's dimension rather than the running
    shared dimension."""

    SURVIVING_LEG = "surviving_leg"
    """One surviving leg, identified by :attr:`ConstraintSource.port_ref`, related to the
    shared dimension as of the check."""

    NODE_PHASE = "node_phase"
    """One node's phase vector dimension, identified by :attr:`ConstraintSource.node_id`,
    related to the shared dimension as of the check."""


@dataclass(frozen=True, slots=True)
class ConstraintSource:
    """*Which* check produced a :class:`DimensionConstraint` -- the record's identity key.

    :mod:`qufzx.rewrite.match` records at most one constraint per source, replacing an
    entry it re-derives in place. :meth:`__post_init__` enforces that ``CONNECTING_PAIR``
    carries neither reference, ``SURVIVING_LEG`` exactly ``port_ref``, and ``NODE_PHASE``
    exactly ``node_id``. Build through :meth:`connecting_pair`, :meth:`surviving_leg` or
    :meth:`node_phase`.
    """

    kind: ConstraintSourceKind
    port_ref: PortRef | None = None
    node_id: NodeId | None = None

    def __post_init__(self) -> None:
        """Reject any (kind, reference) combination the kind's own contract forbids."""
        if self.kind is ConstraintSourceKind.CONNECTING_PAIR:
            expected = self.port_ref is None and self.node_id is None
        elif self.kind is ConstraintSourceKind.SURVIVING_LEG:
            expected = self.port_ref is not None and self.node_id is None
        else:
            expected = self.port_ref is None and self.node_id is not None
        if not expected:
            raise RewriteGrammarError(
                f"ConstraintSource kind {self.kind.value!r} does not accept "
                f"port_ref={self.port_ref!r}, node_id={self.node_id!r}"
            )

    @classmethod
    def connecting_pair(cls) -> ConstraintSource:
        """The consumed wire's own two legs, related to each other."""
        return cls(ConstraintSourceKind.CONNECTING_PAIR)

    @classmethod
    def surviving_leg(cls, port_ref: PortRef) -> ConstraintSource:
        """The surviving leg at ``port_ref`` (``(NodeId, Direction, index)``)."""
        return cls(ConstraintSourceKind.SURVIVING_LEG, port_ref=port_ref)

    @classmethod
    def node_phase(cls, node_id: NodeId) -> ConstraintSource:
        """The phase vector on node ``node_id``."""
        return cls(ConstraintSourceKind.NODE_PHASE, node_id=node_id)

    def __str__(self) -> str:
        if self.kind is ConstraintSourceKind.CONNECTING_PAIR:
            return "connecting pair"
        if self.kind is ConstraintSourceKind.SURVIVING_LEG:
            ref = self.port_ref
            assert ref is not None  # invariant, enforced in __post_init__
            return f"surviving leg {ref.node_id}.{ref.direction.value}[{ref.index}]"
        return f"phase on node {self.node_id}"


class ConstraintOutcome(enum.Enum):
    """Why a :class:`DimensionConstraint` was recorded: the unify deferred, or it bound.

    A unify that was a bare syntactic identity is never recorded at all.
    """

    DEFERRED = "deferred"
    """:meth:`~qufzx.algebra.dimension.Dim.unify` could not decide the equality at all."""

    BOUND = "bound"
    """:meth:`~qufzx.algebra.dimension.Dim.unify` succeeded, but only by binding a free
    symbol -- decided, but only under that binding."""


@dataclass(frozen=True, slots=True)
class DimensionConstraint:
    """One dimension equality a rewrite assumed rather than verified as a syntactic identity.

    For ``SURVIVING_LEG`` and ``NODE_PHASE``, ``assumed`` is the leg's or phase's own ``Dim``
    with every binding accumulated so far substituted in, and ``equal_to`` is the shared
    dimension as of that check. For ``CONNECTING_PAIR``, both are raw leg dims.

    ``bound_here`` is exactly what this check's own ``Dim.unify`` bound: empty when
    ``DEFERRED``, and the name-sorted ``UnifyResult.bindings`` when ``BOUND``. It may hold a
    binding to a non-concrete ``Dim``, unlike the running ``bindings`` accumulator. A detail
    string must read what a check bound off this field, never by intersecting symbol
    occurrences in ``assumed``/``equal_to``.
    """

    assumed: Dim
    equal_to: Dim
    source: ConstraintSource
    outcome: ConstraintOutcome
    bound_here: tuple[tuple[str, Dim], ...] = ()

    def __post_init__(self) -> None:
        """Require ``bound_here`` non-empty iff ``outcome`` is ``BOUND``, and a tuple of
        ``(str, Dim)`` pairs."""
        if self.outcome is ConstraintOutcome.BOUND and not self.bound_here:
            raise RewriteGrammarError(
                f"DimensionConstraint with outcome=BOUND must carry a non-empty "
                f"bound_here, got {self.bound_here!r}"
            )
        if self.outcome is ConstraintOutcome.DEFERRED and self.bound_here:
            raise RewriteGrammarError(
                f"DimensionConstraint with outcome=DEFERRED must carry an empty "
                f"bound_here, got {self.bound_here!r}"
            )
        if not isinstance(self.bound_here, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], Dim)
            for entry in self.bound_here
        ):
            raise RewriteGrammarError(
                f"DimensionConstraint.bound_here must be a tuple of (str, Dim) pairs, "
                f"got {self.bound_here!r}"
            )

    @property
    def deferred(self) -> bool:
        """True iff ``unify`` could not decide this equality (as opposed to binding for it)."""
        return self.outcome is ConstraintOutcome.DEFERRED

    def __str__(self) -> str:
        return f"{self.assumed} == {self.equal_to} ({self.outcome.value}, {self.source})"


@dataclass(frozen=True, slots=True)
class Quantifiers:
    """Declared quantifier metadata: which leg-count and dimension names a rule ranges over.

    ``leg_counts`` names the leg-count variables the rule's equation is stated over;
    ``dimensions`` names the dimension variables. Not consulted in this phase.
    """

    leg_counts: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()


class Match(Protocol):
    """Structural protocol every concrete match type (e.g. ``FusionMatch``) must satisfy.

    Rule-specific location data lives on the concrete match type;
    :mod:`qufzx.rewrite.engine` never reads it, receiving what it needs through
    :class:`BuildResult`.
    """

    @property
    def side_condition_outcomes(self) -> tuple[SideConditionOutcome, ...]:
        """Every named side condition the pattern checked for this candidate, and its outcome."""
        ...

    @property
    def dimension_constraints(self) -> tuple[DimensionConstraint, ...]:
        """Every dimension equality this match assumed rather than verified as an identity.

        Source-keyed: at most one entry per :class:`ConstraintSource`. See
        :mod:`qufzx.rewrite.match`'s module docstring, "Dimension constraints".
        """
        ...

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every entry of :attr:`side_condition_outcomes` passed.

        Vacuously True over an empty or incomplete tuple; full coverage is
        :func:`check_side_condition_coverage`'s job.
        """
        ...


@dataclass(frozen=True, slots=True)
class BuildResult:
    """What a rule's right-hand builder hands back to :mod:`qufzx.rewrite.engine`.

    ``diagram`` is the same working diagram the builder was given, mutated in place to add
    the replacement node(s); the builder never removes matched nodes or touches wires and
    boundaries. :func:`~qufzx.rewrite.engine.apply` checks ``diagram is working`` by object
    identity. ``new_node_ids`` reports every node the builder created, in a deterministic
    order.
    """

    diagram: Diagram
    new_node_ids: tuple[NodeId, ...]
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    port_mapping: Mapping[PortRef, PortRef]
    scalar_introduced: Scalar
    verified_phase_substitutions: Mapping[NodeId, Mapping[str, Dim]] | None = None
    """Per-node bindings a builder actually substituted into a phase's entries. ``None``
    means the rule re-derived nothing; :func:`~qufzx.rewrite.engine.apply` then records an
    empty mapping. There is no match-side counterpart to compare this against.
    """


class Pattern(abc.ABC):
    """A left-hand-side pattern: locates every occurrence of some rewrite shape in a diagram."""

    @abc.abstractmethod
    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Return every occurrence of this pattern in ``diagram``, in a deterministic order.

        Must never mutate ``diagram``. Implementations must never return a candidate whose
        side conditions did not all pass -- see :attr:`Match.all_side_conditions_passed`.
        """
        raise NotImplementedError


RuleBuilder = Callable[[Diagram, Match], BuildResult]
"""The right-hand-side builder signature: consumes a working diagram and a located match.

Mutates and returns the diagram it was given, never the original passed to
:func:`qufzx.rewrite.engine.apply`.
"""


@dataclass(frozen=True, slots=True)
class Rule:
    """A frozen, named rewrite rule: pattern, builder, side conditions, quantifiers, and scalar.

    ``name`` is a stable identifier a certificate can reference. ``side_conditions`` declares
    every named entry :attr:`pattern` reports on. ``scalar_introduced`` is the exact scalar
    this rule introduces on every application; :mod:`qufzx.rewrite.engine` checks a builder's
    against it. ``__post_init__`` validates every field's type, including that
    ``scalar_introduced`` is a ``Scalar`` and never a bare ``float``.
    """

    name: str
    pattern: Pattern
    builder: RuleBuilder
    side_conditions: tuple[SideCondition, ...]
    quantifiers: Quantifiers
    scalar_introduced: Scalar

    def __post_init__(self) -> None:
        """Validate every field's type, the same way every other value object here does."""
        if not isinstance(self.name, str) or not self.name:
            raise RewriteGrammarError(f"rule name must be a non-empty str, got {self.name!r}")
        if not isinstance(self.pattern, Pattern):
            raise RewriteGrammarError(
                f"rule {self.name!r}: pattern must be a Pattern, got {type(self.pattern).__name__}"
            )
        if not callable(self.builder):
            raise RewriteGrammarError(
                f"rule {self.name!r}: builder must be callable, got {type(self.builder).__name__}"
            )
        if not isinstance(self.side_conditions, tuple) or not all(
            isinstance(condition, SideCondition) for condition in self.side_conditions
        ):
            raise RewriteGrammarError(
                f"rule {self.name!r}: side_conditions must be a tuple of SideCondition, "
                f"got {self.side_conditions!r}"
            )
        if not isinstance(self.quantifiers, Quantifiers):
            raise RewriteGrammarError(
                f"rule {self.name!r}: quantifiers must be a Quantifiers, "
                f"got {type(self.quantifiers).__name__}"
            )
        if not isinstance(self.scalar_introduced, Scalar):
            raise RewriteGrammarError(
                f"rule {self.name!r}: scalar_introduced must be a Scalar, "
                f"got {type(self.scalar_introduced).__name__}"
            )
        # A builder declares the tuple it expects by setting a `side_conditions` attribute
        # on the callable; a Rule wrapping it must agree, or the two give contradictory
        # verdicts on the same match. A builder with no such attribute is unconstrained.
        builder_side_conditions = getattr(self.builder, "side_conditions", None)
        if (
            builder_side_conditions is not None
            and tuple(builder_side_conditions) != self.side_conditions
        ):
            raise RewriteGrammarError(
                f"rule {self.name!r}: side_conditions {self.side_conditions!r} disagrees "
                f"with its builder's own declared side_conditions "
                f"{tuple(builder_side_conditions)!r} -- a builder that declares its own "
                "side_conditions (see check_side_condition_coverage) must be wrapped in a "
                "Rule that agrees with it exactly, so there is exactly one source of truth "
                "for which conditions a match must cover, not two verdicts kept in sync by "
                "hand"
            )


def check_side_condition_coverage(
    match: Match, side_conditions: tuple[SideCondition, ...], context: str
) -> None:
    """Verify ``match`` carries a complete, all-passing outcome for every declared condition.

    Requires the set of ``outcome.name`` to equal exactly the set of ``condition.name``,
    with no duplicates, and only then that every outcome passed. ``context`` (typically a
    rule name) is folded into the message. Both :func:`qufzx.rewrite.engine.apply` and each
    rule's own builder call this first, since a builder is reachable directly. Raises
    :class:`RewriteDomainError`.
    """
    outcomes = match.side_condition_outcomes
    outcome_names = [outcome.name for outcome in outcomes]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in outcome_names:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)

    declared_names = {condition.name for condition in side_conditions}
    reported_names = set(outcome_names)
    missing = declared_names - reported_names
    unexpected = reported_names - declared_names
    if duplicates or missing or unexpected:
        raise RewriteDomainError(
            f"{context}: side_condition_outcomes do not exactly cover the declared side "
            f"conditions (duplicate={sorted(duplicates)}, missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)})"
        )

    failed = [outcome.name for outcome in outcomes if not outcome.passed]
    if failed:
        raise RewriteDomainError(f"{context}: match has failing side condition(s) {failed}")
