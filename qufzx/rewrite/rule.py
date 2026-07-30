"""Rule: a left-hand pattern, right-hand builder, side conditions, quantifiers, and exact scalar.

A :class:`Rule` is a frozen value object bundling everything Phase 6's certificate will
need to describe *why* a rewrite step was legal, without itself performing any graph
surgery: the pattern (:class:`Pattern`) that locates candidate occurrences, a builder
callable that turns one located :class:`Match` into a replacement fragment
(:class:`BuildResult`), the named side conditions the pattern is documented to check,
declared quantifier metadata, and the exact scalar the rule is expected to introduce.

Pattern as a real abstraction, not a hardcoded string. :class:`Pattern` is an abstract
base class with a single method, :meth:`Pattern.find_matches`, so that Phase 7 onward can
add new pattern shapes (bang-boxed fusion, bialgebra, Hopf, ...) by implementing the same
seam rather than by teaching :mod:`qufzx.rewrite.match` a family of hardcoded rule names.
Phase 5 registers exactly one concrete pattern, ``FusionPattern`` in
:mod:`qufzx.rewrite.match`.

Match as a structural protocol, not a shared dataclass base. Every concrete pattern
produces its own match type carrying whatever location data it needs (spider fusion needs
two node ids and a wire; a future bialgebra pattern will need a different shape entirely).
Rather than force every future match type through one dataclass inheritance hierarchy
(which forces awkward field-ordering compromises across unrelated shapes, since Python
dataclass inheritance requires a single consistent constructor shape), :class:`Match` here
is a ``typing.Protocol``: any match object that exposes ``side_condition_outcomes``,
``dimension_constraints``, and ``all_side_conditions_passed`` satisfies it structurally.
:mod:`qufzx.rewrite.match`'s ``FusionMatch`` is a plain dataclass that happens to expose
those three members; it does not need to subclass anything here.

BuildResult as the generic engine/builder contract. :mod:`qufzx.rewrite.engine`'s
``apply()`` must remain generic over any future rule, so a builder cannot hand back only
a Diagram and a scalar -- it must also tell the engine *which* nodes and wires the match
consumed and how surviving ports were remapped, so the engine can splice the replacement
into the working diagram without knowing anything rule-specific. :class:`BuildResult`
carries exactly that: the (mutated) working diagram, the freshly created node id(s)
introduced by the builder, the node ids and wires the match consumed (to be removed once
every reference to them has been remapped), the old-port -> new-port remapping for every
surviving port, and the exact scalar introduced. See :mod:`qufzx.rewrite.engine` for how
these fields are consumed.

Quantifiers are declared metadata, not enforced machinery, in this phase. Every ZX rewrite
rule is, mathematically, "for all leg counts n, for all dimensions d satisfying some
constraint, this equation holds." Phase 5 has no bang boxes and no real dimension
unifier, so :class:`Quantifiers` only *records* the names of the leg-count and dimension
variables a rule is stated over (for spider fusion: the two spiders' unbounded leg counts,
and the one shared leg dimension) -- it is not yet consulted anywhere to restrict matching.
Phase 7 (bang-box multiplicities) and Phase 10 (a real dimension unifier with domain
constraints such as "only at prime d") are what turn these declared names into checked
constraints; the field exists now precisely so those phases extend metadata that already
has a place to live, rather than bolting a new concept onto :class:`Rule` later.

Typed errors. Following this codebase's established domain/grammar split (see
:mod:`qufzx.diagram.graph`, :mod:`qufzx.algebra.dimension`, etc.), :class:`RewriteError` is
the base for every error raised anywhere in :mod:`qufzx.rewrite`;
:class:`RewriteDomainError` covers a value or state outside the mathematical domain a
rewrite operation requires (e.g. a match whose side conditions did not all pass, or a
builder that introduced a scalar disagreeing with its rule's declared one);
:class:`RewriteGrammarError` covers a malformed request (e.g. a match or diagram that does
not belong to the rule being applied). :mod:`qufzx.rewrite.match`,
:mod:`qufzx.rewrite.rules_library`, and :mod:`qufzx.rewrite.engine` all raise these same
three classes rather than defining their own hierarchies.
"""

from __future__ import annotations

import abc
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


@dataclass(frozen=True, slots=True)
class Quantifiers:
    """Declared quantifier metadata for a rule: which leg-count and dimension names it ranges over.

    Pure documentation in Phase 5 -- see the module docstring for why this is metadata
    Phase 7 and Phase 10 will make checkable, not dead fields. ``leg_counts`` names the
    (currently unbounded, per :class:`~qufzx.diagram.generators.LegPolicy`) leg-count
    variables the rule's equation is stated over; ``dimensions`` names the dimension
    variables (e.g. the one shared leg dimension ``"d"`` for spider fusion).
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
    def dimension_constraints(self) -> tuple[tuple[Dim, Dim], ...]:
        """Dimension pairs this match assumed rather than verified as a syntactic identity.

        Two distinct provenances land here (see :mod:`qufzx.rewrite.match`'s module
        docstring for the full account): a pair :meth:`Dim.unify` deferred on
        outright, and a pair it reported ``SUCCESS`` for only because it bound a free
        symbol (e.g. ``d`` and ``3`` unify by binding ``d := 3`` -- decided, but only
        under that binding). Both are assumptions a real unifier (Phase 10) must
        eventually justify, so a certificate needs both; only a bare syntactic identity
        (nothing bound, nothing deferred) is left out, since nothing was assumed there.
        """
        ...

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every entry of :attr:`side_condition_outcomes` passed.

        A pattern must never include a candidate with a failing side condition in its
        returned matches at all (see :mod:`qufzx.rewrite.match`'s module docstring), so
        this is expected to always be True for a match a pattern actually returned. This
        alone is *not* the full side-condition invariant, though: ``all(...)`` over an
        empty (or merely incomplete) :attr:`side_condition_outcomes` is vacuously True, so
        this property cannot by itself catch a hand-built or foreign match that is simply
        missing outcomes for some -- or all -- of a rule's declared
        :class:`SideCondition`\\ s. Verifying full coverage (that the outcome names exactly
        match the rule's declared side conditions, with no duplicates and no gaps) is a
        separate, explicit check performed by :func:`check_side_condition_coverage` below,
        which :mod:`qufzx.rewrite.engine`'s ``apply`` and each rule's own builder (e.g.
        :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`) both call before doing
        any work -- not by this property, and not implicitly.
        """
        ...


@dataclass(frozen=True, slots=True)
class BuildResult:
    """What a rule's right-hand builder hands back to :mod:`qufzx.rewrite.engine`.

    See the module docstring for why this is the generic engine/builder contract.
    ``diagram`` is the same working diagram the builder was given, mutated in place to add
    the replacement node(s) -- the builder never removes the matched nodes or touches any
    wire or boundary entry beyond that; splicing the replacement into the rest of the
    diagram (remapping wires and boundaries, then removing ``consumed_node_ids``) is
    :mod:`qufzx.rewrite.engine`'s job, done generically from these fields alone.

    ``new_node_ids`` reports every node the builder created, in a deterministic order --
    one for spider fusion, but a rule such as Phase 11's bialgebra (which creates m*n
    nodes) or Hopf/copy (which create several) reports all of them here, not just one.
    """

    diagram: Diagram
    new_node_ids: tuple[NodeId, ...]
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    port_mapping: Mapping[PortRef, PortRef]
    scalar_introduced: Scalar


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
    ``side_conditions`` documents every named predicate :attr:`pattern` is expected to
    check (see :class:`SideCondition`); a match's own ``side_condition_outcomes`` names the
    same conditions with their per-candidate result. ``scalar_introduced`` is the exact
    scalar this rule is declared to introduce on every application; :mod:`qufzx.rewrite.engine`
    checks a builder's :attr:`BuildResult.scalar_introduced` against this declared value and
    raises :class:`RewriteDomainError` on disagreement, rather than silently trusting
    whichever value the builder happened to compute.
    """

    name: str
    pattern: Pattern
    builder: RuleBuilder
    side_conditions: tuple[SideCondition, ...]
    quantifiers: Quantifiers
    scalar_introduced: Scalar

    def __post_init__(self) -> None:
        """Validate that ``name`` is a non-empty string."""
        if not isinstance(self.name, str) or not self.name:
            raise RewriteGrammarError(f"rule name must be a non-empty str, got {self.name!r}")


def check_side_condition_coverage(
    match: Match, side_conditions: tuple[SideCondition, ...], context: str
) -> None:
    """Verify ``match`` carries a complete, all-passing outcome for every declared condition.

    :attr:`Match.all_side_conditions_passed` alone cannot catch a match whose
    ``side_condition_outcomes`` is empty or merely incomplete -- ``all()`` over ``()`` is
    vacuously True (see that property's docstring). This function closes that hole: it
    requires the set of ``outcome.name`` in ``match.side_condition_outcomes`` to equal
    exactly the set of ``condition.name`` in ``side_conditions`` (no missing name, no
    unexpected name), requires no duplicate outcome names, and only then checks that every
    outcome passed. ``context`` (typically a rule name, e.g. ``"spider_fusion"``) is folded
    into the raised message so a certificate-adjacent caller can tell which rule rejected
    the match.

    Both :func:`qufzx.rewrite.engine.apply` and each rule's own builder (e.g.
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`) call this before doing any
    work, since a builder is reachable directly and not only through ``apply``. Raises
    :class:`RewriteDomainError` -- a coverage or passedness failure is a match outside the
    mathematical domain a rewrite requires, the same category as a single failed side
    condition, not a malformed request.
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
