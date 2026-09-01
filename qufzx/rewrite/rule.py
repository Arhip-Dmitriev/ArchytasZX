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

A :class:`Rule` is a frozen value object bundling everything Phase 6's certificate needs to
describe why a rewrite step was legal, without performing any graph surgery: the
:class:`Pattern` that locates candidates, a builder turning one :class:`Match` into a
:class:`BuildResult`, the named side conditions the pattern checks, declared quantifier
metadata, and the exact scalar the rule introduces.

Pattern is an abstract base class with a single method, :meth:`Pattern.find_matches`, so
Phase 7 onward adds new shapes (bang-boxed fusion, bialgebra, Hopf) by implementing that
seam rather than teaching :mod:`qufzx.rewrite.match` a family of hardcoded rule names.

Match is a ``typing.Protocol``, not a dataclass base. Every pattern produces its own match
type carrying whatever location data it needs, and dataclass inheritance would force one
consistent constructor shape across unrelated ones. Any object exposing
``side_condition_outcomes``, ``dimension_constraints``, and ``all_side_conditions_passed``
satisfies it structurally.

:class:`DimensionConstraint` is source-keyed, not a bare pair: it carries the assumed
equality plus the :class:`ConstraintSource` that produced it and the
:class:`ConstraintOutcome` it came from. That lets a pattern whose dimension resolution
iterates to a fixpoint record each source exactly once, at its most-resolved form, by
replacing the entry for a source it re-derives.

:class:`BuildResult` is the generic engine/builder contract. Since ``apply()`` stays
generic over any future rule, a builder must report not just a diagram and a scalar but
which nodes and wires the match consumed and how surviving ports were remapped, so the
engine can splice without knowing anything rule-specific. It also carries
``verified_side_condition_outcomes`` and ``verified_dimension_constraints``: the channel
through which a builder that independently re-derives its match's assumptions returns that
ground truth, rather than an unaudited claim a foreign match could have fabricated.

Quantifiers are declared metadata, not enforced machinery, in this phase. Every ZX rule is
mathematically "for all leg counts n, for all dimensions d satisfying some constraint, this
holds"; :class:`Quantifiers` records the names of those variables but is not yet consulted
to restrict matching. Phase 7 (bang-box multiplicities) and Phase 10 (a real unifier with
constraints such as "only at prime d") turn them into checked constraints -- the field
exists now so those phases extend metadata that already has a place to live.

Typed errors follow this codebase's domain/grammar split. :class:`RewriteError` is the base
for everything raised in :mod:`qufzx.rewrite`; :class:`RewriteDomainError` covers a value
or state outside the domain a rewrite requires (a match whose side conditions did not pass,
a builder whose scalar disagrees with its rule's); :class:`RewriteGrammarError` covers a
malformed request (a match or diagram not belonging to the rule being applied).
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

    Raised for a match whose side conditions were not all satisfied, and for a builder
    whose introduced scalar disagrees with its rule's declared :attr:`Rule.scalar_introduced`.
    """


class RewriteGrammarError(RewriteError):
    """A rewrite request is malformed independent of mathematical domain.

    Raised for a match that does not belong to the diagram or rule it is applied against.
    """


@dataclass(frozen=True, slots=True)
class SideCondition:
    """A named, individually-reportable predicate a :class:`Pattern` checks per candidate.

    Declared once per pattern (see ``FUSION_SIDE_CONDITIONS`` in
    :mod:`qufzx.rewrite.match`) so a certificate can say *which* condition was checked and
    what it returned for a given match, rather than reporting only "it matched". This is
    metadata about the pattern; the per-candidate result lives in
    :class:`SideConditionOutcome`.
    """

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class SideConditionOutcome:
    """The result of checking one named :class:`SideCondition` against one match candidate.

    ``deferred`` mirrors :class:`qufzx.diagram.validate.ValidationIssue`'s convention:
    True marks an outcome that passed only because a dimension equality could not yet be
    decided and was assumed (see :meth:`qufzx.algebra.dimension.Dim.unify`'s ``DEFERRED``
    status), not because it was verified outright.
    """

    name: str
    passed: bool
    detail: str
    deferred: bool = False


class ConstraintSourceKind(enum.Enum):
    """Which kind of check in a pattern's resolution produced a :class:`DimensionConstraint`.

    The discriminator that makes :attr:`Match.dimension_constraints` a uniformly-readable
    record: the connecting pair relates two legs to each other, while every other entry
    relates one leg or phase to the shared dimension.
    """

    CONNECTING_PAIR = "connecting_pair"
    """The two legs the matched (consumed) wire joins, related to each other. This is the
    only kind whose :attr:`DimensionConstraint.equal_to` is another leg's dimension rather
    than the candidate's running shared dimension: it is what *starts* the resolution, so
    there is no shared dimension yet to relate it to."""

    SURVIVING_LEG = "surviving_leg"
    """One specific surviving leg, identified by :attr:`ConstraintSource.port_ref`
    (``(NodeId, Direction, index)``), related to the shared dimension as of the check."""

    NODE_PHASE = "node_phase"
    """One specific node's phase vector dimension, identified by
    :attr:`ConstraintSource.node_id`, related to the shared dimension as of the check."""


@dataclass(frozen=True, slots=True)
class ConstraintSource:
    """*Which* check produced a :class:`DimensionConstraint` -- the record's identity key.

    A resolution that iterates to a fixpoint checks the same leg or phase several times,
    each against a more-resolved shared dimension. Keying the record by source makes "a
    source is checked many times but recorded once, at its most-resolved form" structural:
    :mod:`qufzx.rewrite.match` replaces the entry for a source it re-derives, in place,
    preserving first-derivation order.

    Enforced by :meth:`__post_init__`: ``CONNECTING_PAIR`` carries neither reference
    (there is exactly one connecting pair per candidate), ``SURVIVING_LEG`` carries exactly
    ``port_ref``, and ``NODE_PHASE`` carries exactly ``node_id``. Build one through
    :meth:`connecting_pair`, :meth:`surviving_leg`, or :meth:`node_phase` rather than by
    hand.
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

    Both are assumptions a real unifier (Phase 10) must eventually justify, and they are
    different assumptions, so the record says which. A unify that was a bare syntactic
    identity is never recorded at all -- hence no third member.
    """

    DEFERRED = "deferred"
    """:meth:`~qufzx.algebra.dimension.Dim.unify` could not decide the equality at all."""

    BOUND = "bound"
    """:meth:`~qufzx.algebra.dimension.Dim.unify` succeeded, but only by binding a free
    symbol -- decided, but only under that binding."""


@dataclass(frozen=True, slots=True)
class DimensionConstraint:
    """One dimension equality a rewrite assumed rather than verified as a syntactic identity.

    ``assumed == equal_to`` is the assumed equality, ``source`` says which check produced
    it (and is the key a fixpoint re-derivation replaces in place -- see
    :class:`ConstraintSource`), and ``outcome`` says whether the underlying
    :meth:`~qufzx.algebra.dimension.Dim.unify` deferred or bound.

    For ``SURVIVING_LEG`` and ``NODE_PHASE`` sources, ``assumed`` is the leg's or phase's
    own ``Dim`` with every concrete binding accumulated so far already substituted in, and
    ``equal_to`` is the candidate's shared dimension as of that check. For
    ``CONNECTING_PAIR``, both are raw leg dims -- see
    :attr:`ConstraintSourceKind.CONNECTING_PAIR`.

    ``bound_here`` is exactly what this check's own ``Dim.unify`` call bound: empty for a
    ``DEFERRED`` outcome, and for ``BOUND`` the raw ``UnifyResult.bindings``, name-sorted.
    It may include a binding to a non-concrete ``Dim`` (``d := e``), unlike the running
    ``bindings`` accumulator, since it records what one check bound rather than feeding
    later resolution. A detail string must read what a check bound off this field, never by
    intersecting symbol occurrences in ``assumed``/``equal_to`` against another check's
    bindings.
    """

    assumed: Dim
    equal_to: Dim
    source: ConstraintSource
    outcome: ConstraintOutcome
    bound_here: tuple[tuple[str, Dim], ...] = ()

    def __post_init__(self) -> None:
        """Require ``bound_here`` non-empty iff ``outcome`` is ``BOUND``, and well-shaped.

        The shape check (a tuple of ``(str, Dim)`` pairs) mirrors
        :meth:`ConstraintSource.__post_init__`'s treatment of ``(kind, reference)``.
        """
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

    Pure documentation in Phase 5; Phase 7 and Phase 10 make it checkable. ``leg_counts``
    names the leg-count variables the rule's equation is stated over; ``dimensions`` names
    the dimension variables (for spider fusion, the one shared leg dimension ``"d"``).
    """

    leg_counts: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()


class Match(Protocol):
    """Structural protocol every concrete match type (e.g. ``FusionMatch``) must satisfy.

    Deliberately a ``Protocol``, not a shared dataclass base -- see the module docstring.
    A match's rule-specific location data (which nodes, which wire, ...) lives entirely on
    the concrete match type; :mod:`qufzx.rewrite.engine` never reads it directly, since
    that data reaches the engine through :class:`BuildResult` instead.
    """

    @property
    def side_condition_outcomes(self) -> tuple[SideConditionOutcome, ...]:
        """Every named side condition the pattern checked for this candidate, and its outcome."""
        ...

    @property
    def dimension_constraints(self) -> tuple[DimensionConstraint, ...]:
        """Every dimension equality this match assumed rather than verified as an identity.

        Source-keyed: at most one entry per :class:`ConstraintSource`, each carrying its
        own :class:`ConstraintOutcome` (deferred, or decided only under a binding). See
        :class:`DimensionConstraint` and :mod:`qufzx.rewrite.match`'s module docstring,
        "Dimension constraints", for the recording contract and the tests that enforce it.
        """
        ...

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every entry of :attr:`side_condition_outcomes` passed.

        A pattern never returns a candidate with a failing side condition, so this is
        always True for a match a pattern produced. It is not the full invariant:
        ``all(...)`` over an empty or incomplete :attr:`side_condition_outcomes` is
        vacuously True, so it cannot catch a hand-built match missing outcomes. Full
        coverage is checked separately by :func:`check_side_condition_coverage`, which both
        :mod:`qufzx.rewrite.engine`'s ``apply`` and each rule's builder call first.
        """
        ...


@dataclass(frozen=True, slots=True)
class BuildResult:
    """What a rule's right-hand builder hands back to :mod:`qufzx.rewrite.engine`.

    See the module docstring for why this is the generic engine/builder contract.
    ``diagram`` is the same working diagram the builder was given, mutated in place to add
    the replacement node(s); the builder never removes matched nodes or touches wires and
    boundaries. Splicing the replacement into the rest of the diagram is
    :mod:`qufzx.rewrite.engine`'s job, done generically from these fields alone.
    :func:`~qufzx.rewrite.engine.apply` checks ``diagram is working`` by object identity
    right after calling the builder.

    ``new_node_ids`` reports every node the builder created, in a deterministic order --
    one for spider fusion, several for a rule such as Phase 11's bialgebra or Hopf/copy.
    """

    diagram: Diagram
    new_node_ids: tuple[NodeId, ...]
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    port_mapping: Mapping[PortRef, PortRef]
    scalar_introduced: Scalar
    verified_side_condition_outcomes: tuple[SideConditionOutcome, ...] | None = None
    verified_dimension_constraints: tuple[DimensionConstraint, ...] | None = None
    """The facts a builder independently re-derived, for the certificate to record instead
    of the match's own unverified claims.

    A builder that re-checks its match against the diagram it was handed (e.g.
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`, via
    :func:`~qufzx.rewrite.match.resolve_fusion_match`) computes a ground-truth
    ``side_condition_outcomes``/``dimension_constraints`` pair as a side effect. Without
    this channel the certificate would be built from the match's claims, which can assert a
    dimension binding the rewrite never assumed, or omit one it did.

    ``None`` means the rule re-derived nothing new -- the correct value for a rule with no
    verification step of its own. :func:`~qufzx.rewrite.engine.apply` prefers these fields
    over ``match``'s whenever they are not ``None``. A builder must populate them only
    after checking that the match's own claims agree; :func:`spider_fusion_builder` raises
    :class:`RewriteDomainError` on disagreement rather than silently preferring one value.
    """
    verified_phase_substitutions: Mapping[NodeId, Mapping[str, Dim]] | None = None
    """Per-node bindings a builder actually substituted into a phase's entries, through the
    same channel ``verified_dimension_constraints`` uses. ``None`` means the rule re-derived
    nothing new.
    """


class Pattern(abc.ABC):
    """A left-hand-side pattern: locates every occurrence of some rewrite shape in a diagram.

    See the module docstring for why this is an abstract base rather than a hardcoded
    dispatch on a rule name.
    """

    @abc.abstractmethod
    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Return every occurrence of this pattern in ``diagram``, in a deterministic order.

        Must never mutate ``diagram``. Implementations must never return a candidate whose
        side conditions did not all pass -- see :attr:`Match.all_side_conditions_passed`.
        """
        raise NotImplementedError


RuleBuilder = Callable[[Diagram, Match], BuildResult]
"""The right-hand-side builder signature: consumes a working diagram and a located match.

Mutates and returns the diagram it was given (see :class:`BuildResult`); never receives
or returns the original, unmutated diagram passed to :func:`qufzx.rewrite.engine.apply`.
"""


@dataclass(frozen=True, slots=True)
class Rule:
    """A frozen, named rewrite rule: pattern, builder, side conditions, quantifiers, and scalar.

    ``name`` is a stable identifier (e.g. ``"spider_fusion"``) a certificate can reference.
    ``side_conditions`` documents every named predicate :attr:`pattern` checks; a match's
    own ``side_condition_outcomes`` names the same conditions with their per-candidate
    result. ``scalar_introduced`` is the exact scalar this rule introduces on every
    application; :mod:`qufzx.rewrite.engine` checks a builder's
    :attr:`BuildResult.scalar_introduced` against it and raises
    :class:`RewriteDomainError` on disagreement. ``__post_init__`` validates every field's
    type, including that ``scalar_introduced`` is a ``Scalar`` and never a bare ``float``.
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
                f"rule {self.name!r}: builder must be callable, "
                f"got {type(self.builder).__name__}"
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
        # A builder that calls check_side_condition_coverage itself (it is reachable
        # directly, not only through apply) must check against exactly the tuple this Rule
        # declares, or the two give contradictory verdicts on the same match. A builder
        # declares its expectation by setting a `side_conditions` attribute on the callable;
        # one with no such attribute is unconstrained here.
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

    :attr:`Match.all_side_conditions_passed` alone cannot catch a match whose
    ``side_condition_outcomes`` is empty or incomplete, since ``all()`` over ``()`` is
    vacuously True. This function requires the set of ``outcome.name`` to equal exactly the
    set of ``condition.name`` in ``side_conditions``, with no duplicates, and only then
    checks that every outcome passed. ``context`` (typically a rule name) is folded into
    the raised message.

    Both :func:`qufzx.rewrite.engine.apply` and each rule's own builder call this before
    doing any work, since a builder is reachable directly. Raises
    :class:`RewriteDomainError`: a coverage or passedness failure is a match outside the
    domain a rewrite requires, not a malformed request.
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
