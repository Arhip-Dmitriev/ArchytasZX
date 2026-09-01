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

"""Rewrite engine: applies rules at matches, returns new diagrams, and records step provenance.

:func:`apply` is the single entry point. It is generic over any future
:class:`~qufzx.rewrite.rule.Rule` -- it never inspects a match's rule-specific fields, only
the generic contract :class:`~qufzx.rewrite.rule.Match` and
:class:`~qufzx.rewrite.rule.BuildResult` expose. All rule-specific work (which nodes are
consumed, the replacement node's legs and phase, the scalar introduced) happens in the
rule's own builder in :mod:`qufzx.rewrite.rules_library`; this module only splices a
``BuildResult`` into the rest of a diagram.

Algorithm.

1. Verify the match's certificate.
   :func:`~qufzx.rewrite.rule.check_side_condition_coverage` rejects a match whose
   ``side_condition_outcomes`` do not name exactly ``rule.side_conditions`` -- no fewer, no
   more, no duplicates -- or whose outcomes are not all ``passed``. Checking coverage and
   not only passedness matters: ``all(...)`` over an empty tuple is vacuously True.
2. Copy. Never mutate the diagram passed in; work on ``diagram.copy()`` and return that.
3. Build. Call ``rule.builder(working, match)``, which per ``BuildResult``'s contract adds
   the replacement node(s) only -- no wire, boundary, or node removal. ``build_result
   .diagram`` must ``is``-match ``working``.
4. Verify the build result belongs to this diagram. The introduced scalar must agree with
   the rule's declared ``scalar_introduced``; every consumed wire and node id must exist in
   ``working``; every ``new_node_ids`` entry must exist and every ``port_mapping`` value
   must name a real port (step 5 and step 9 consume these with no check of their own); and
   ``consumed_node_ids`` must have no duplicate, which step 6's removal loop would
   otherwise turn into a ``GraphGrammarError`` escaping this module's error hierarchy.
   These are malformed requests, so ``RewriteGrammarError``.
5. Remap every reference, before any node is removed. For each wire in ``working``: if it
   is a consumed wire, drop it; otherwise, if either endpoint sits on a consumed node,
   re-add the wire with that endpoint replaced via ``port_mapping``. An endpoint on a
   consumed node absent from ``port_mapping`` raises ``RewriteDomainError`` rather than
   leaving a wire pointing at a node step 6 is about to remove, whose removal cascade would
   silently drop it. Both endpoints directed to a single port raises
   ``RewriteGrammarError`` rather than building a degenerate wire. One uniform rule covers
   every case: a wire to a third node, a pre-existing self-loop on a consumed node, and the
   consumed wire itself. Both ordered boundary lists are rebuilt through the same
   ``_remap_endpoint``, in place, so position is preserved and a boundary ref is held to
   the identical standard as a wire endpoint.

   ``_remap_endpoint``'s raise is unreachable for any match :func:`~qufzx.rewrite.match
   .find_matches` returned -- it rejects candidates whose consumed port is multiply claimed,
   and validates every wire endpoint and boundary entry for malformed references. It stays
   as a defensive check against a hand-built or foreign ``Match``.
6. Remove the consumed nodes, only once every reference to them has been replaced.
   :meth:`~qufzx.diagram.graph.Diagram.remove_node`'s cascade is then a no-op; removing
   before step 5 would silently drop wires to third nodes before they were remapped.
7. Multiply the scalar, after every structural change.
8. Verify the rewrite is not a relative regression. :func:`~qufzx.diagram.validate.validate`
   runs on both the input and the finished ``working``; a hard-failure issue in ``working``
   not accounted for in the input raises ``RewriteDomainError``. The comparison is a
   *multiset* over ``(kind, offending ref)`` (:func:`_issue_key`), never over bare
   ``IssueKind``\\ s -- a set comparison cannot see a second independent issue of an
   already-present kind, nor one the rewrite removed -- and never by message, since a node
   id in a message legitimately changes across a rewrite.

   Input-side keys are first mapped into ``working``'s coordinate space by
   :func:`_translate_input_issue_key` (via ``port_mapping`` for a port on a consumed node,
   ``new_node_ids`` for a node id, fail-closed when a rule's consumed-to-new cardinality is
   not one). Without this, any reference anchored on a consumed node would always read as
   "introduced", since that id and those port indices are gone.

   The check is relative, never absolute: a diagram that already carries a hard error is
   legitimately rewritable, and only a rewrite that makes things worse is blocked.

   The compare covers ``.errors`` only. ``.deferred`` issues are an assumed diagram state,
   not a regression, so step 8 never blocks on them; instead the same translation machinery
   runs over both sides' ``.deferred`` to populate
   :attr:`RewriteStep.removed_deferred_issues` and
   :attr:`RewriteStep.introduced_deferred_issues` -- a rewrite can create an assumption as
   readily as it resolves one. That compare is a multiset too: two node-anchored deferred
   issues, one on each fused spider, translate to the same key, which a dict-by-key
   comparison would drop to last-write-wins. When several issues collide on one key and
   only some have a counterpart, the one reported is arbitrary but deterministic (first in
   that side's own order), and :attr:`RewriteStep.deferred_issue_identity_ambiguous` says so.
9. Record provenance. A :class:`RewriteStep` carrying the rule name, the ``match`` verbatim
   (so Phase 6 replays from it rather than re-running the matcher), the match location as
   it was in the input diagram, every side condition and dimension constraint, the scalar
   introduced, and the full old-port -> new-port remapping. Side conditions and constraints
   come from ``build_result``'s ``verified_*`` fields when the builder supplied them,
   falling back to the match's own: a builder that independently re-derives these hands
   back ground truth, not the match's unaudited claim. Phase 6 implements replay; this
   module does not.

What this module deliberately does not do. It does not search for matches (that is
:mod:`qufzx.rewrite.match`); it does not choose which rule or match to apply, or iterate to
a fixpoint (Phase 11's strategy layer); and, per the spec, it never contracts or evaluates
a diagram numerically -- nothing here imports :mod:`qufzx.semantics`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, NodeId, PortRef, Wire
from qufzx.diagram.validate import IssueKind, ValidationIssue, validate
from qufzx.rewrite.rule import (
    DimensionConstraint,
    Match,
    RewriteDomainError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
    check_side_condition_coverage,
)


@dataclass(frozen=True, slots=True)
class RewriteStep:
    """Structured provenance for one rewrite application. See the module docstring, step 9.

    Every field is drawn either from the ``Match`` that was applied or from the
    ``BuildResult`` its rule's builder produced, never re-derived from the mutated working
    diagram, so this record describes the rewrite *as it was applied to the input*.
    ``match`` is the field Phase 6 replays from: the exact ``Match`` object ``apply`` was
    given, stored verbatim rather than re-derived, so a replayer never needs to re-run the
    matcher and re-select a candidate by node id -- it can look ``rule_name`` up (via
    :func:`~qufzx.rewrite.rules_library.lookup_rule`) and re-apply it at this stored match
    directly.
    """

    rule_name: str
    match: Match
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...]
    """Every dimension equality this rewrite assumed rather than verified as a syntactic
    identity -- see :attr:`~qufzx.rewrite.rule.Match.dimension_constraints` for the recording
    contract itself (source-keyed, at most one entry per
    :class:`~qufzx.rewrite.rule.ConstraintSource`).

    A ``DEFERRED`` entry here is a recorded *assumption*, not a claim that
    the assumption is satisfiable on anything but a single degenerate point -- e.g. a
    surviving leg of dimension ``d**2`` forced onto a shared dimension ``d`` records ``d**2
    == d``, true only at ``d = 1``. The rewrite is sound relative to what it recorded (the
    oracle can confirm it at any substitution that actually satisfies the assumption), but
    Phase 5's placeholder :meth:`~qufzx.algebra.dimension.Dim.unify` cannot itself distinguish
    a constraint satisfiable broadly from one satisfiable only degenerately -- discharging
    that distinction is explicitly Phase 10's job (see ``Dim.unify``'s own docstring), not
    something this field or this phase's matcher fabricates a guess at. See
    :mod:`qufzx.rewrite.rules_library`'s module docstring, "What that does not
    claim", for the fuller
    account and the worked example.
    """
    scalar_introduced: Scalar
    port_mapping: Mapping[PortRef, PortRef]
    new_node_ids: tuple[NodeId, ...]
    removed_deferred_issues: tuple[ValidationIssue, ...] = ()
    """Every :attr:`~qufzx.diagram.validate.ValidationReport.deferred` issue ``diagram``
    carried that a *multiset* compare (translated key, via
    :func:`_translate_input_issue_key`) says has no surviving counterpart among ``working``'s
    own deferred issues, in ``diagram``'s own (pre-rewrite) coordinates. Step 8 refuses to
    introduce a new hard-error issue kind, but it never looks at deferred issues at all -- a
    diagram-level unify assumption
    (:class:`~qufzx.diagram.validate.IssueKind.DIMENSION_DEFERRED`) a rewrite resolves (by
    consuming or overwriting the leg it was recorded against) would otherwise simply vanish,
    with nothing on the certificate to say a pre-existing assumption was ever there, let
    alone that this rewrite is the one that made it disappear. A non-empty value is not an
    error -- see :mod:`qufzx.rewrite.rules_library`'s module docstring, "Phase 5 judgement
    call", for why spider_fusion is allowed to fire on a ``DEFERRED`` dimension pair at all.
    Enforced by ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.

    The compare is multiset, not set/dict-keyed: two distinct input issues that translate to
    the same key -- e.g. one node-anchored ``DIMENSION_DEFERRED`` issue on each of two nodes
    a fusion consumes, both mapped onto the same surviving node id -- are each counted and,
    if both lack a surviving counterpart, both reported, never collapsed to one by a
    last-write-wins dict lookup.
    """

    introduced_deferred_issues: tuple[ValidationIssue, ...] = ()
    """The other direction of the same :class:`~collections.Counter` difference: every
    deferred issue ``working`` carries that has no counterpart among ``diagram``'s own
    (translated) deferred issues, in ``working``'s own post-rewrite coordinates.

    A rewrite can *create* a deferred assumption as readily as it resolves one -- forcing a
    surviving leg onto ``shared_dim`` can leave a neighbouring wire that was an exact match
    before merely deferred after (see :mod:`qufzx.rewrite.rules_library`'s module docstring,
    "Dimension of the merged node"). That is expected, not a defect; but the argument for
    recording removals -- that a silently-changed assumption is a loss of information the
    certificate should not paper over -- is direction-symmetric, so this field exists too.
    Neither field is a gate; both are certificate facts.
    Enforced by ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.
    """

    phase_substitutions: Mapping[NodeId, Mapping[str, Dim]] = MappingProxyType({})
    """Per-node bindings the builder actually substituted into a phase's entries -- see
    :attr:`~qufzx.rewrite.rule.BuildResult.verified_phase_substitutions`. Empty when the
    builder supplied ``None`` (nothing re-derived) or genuinely substituted nothing.
    """

    deferred_issue_identity_ambiguous: bool = False
    """Whether the *identity* of the reported deferred issues above is meaningful, or only
    their count.

    The contract, stated rather than left to be inferred. Both fields are populated by
    walking the source report's issues in :func:`~qufzx.diagram.validate.validate` order and
    taking the first ``n`` occurrences of each translated key, where ``n`` is that key's
    Counter surplus. When each key's surplus equals its total occurrence count, that
    selection is forced and every reported issue is uniquely the one with no counterpart.
    When several issues collide on one translated key and only *some* of them lack a
    counterpart, which of the colliding issues to name is genuinely unrecoverable: the
    collision arises because :func:`_translate_input_issue_key` maps every consumed node's
    identity onto the one surviving merged node, and nothing in a
    :class:`~qufzx.diagram.validate.ValidationIssue` distinguishes two issues that agree on
    ``(kind, ref)`` beyond a message this comparison deliberately never reads (a message
    embeds node ids that legitimately change across a rewrite). In that case the selection
    is arbitrary but deterministic -- first in ``validate`` order -- the count is the
    meaningful part, and this flag is ``True`` so a reader is told so in the data rather
    than left to assume the named issue was chosen for a reason.
    Enforced by
    ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``\\
    ``::test_colliding_keys_are_flagged_ambiguous_and_pinned_to_validate_order``.

    "First in ``validate`` order" is a value-and-order contract, and is now actually true
    across processes, not merely within one (Phase 5 post-closing audit round 18, Defect 1):
    :func:`~qufzx.diagram.validate.validate` used to iterate ``diagram.wires``, a frozenset
    whose hash (via ``Wire`` -> ``PortRef`` -> ``Direction``, an ``Enum`` hashed by member
    name) is ``PYTHONHASHSEED``-dependent, so which issue ``validate`` reported *first* --
    and therefore which of several colliding issues this field names when ambiguous -- could
    differ between two otherwise-identical runs in different processes, contradicting this
    docstring's own promise. Fixed by sorting every such iteration in ``validate.py`` (see
    that module's docstring); this field's selection is deterministic by *value and order*
    across processes now, proven (not merely asserted) by
    ``tests/test_engine.py::TestCrossProcessDeterminism``, which runs an identical rewrite in
    two subprocesses under two different ``PYTHONHASHSEED`` values and compares this field
    (among others) by ``repr()`` equality -- never by ``hash()``, which remains legitimately
    process-dependent for reasons unrelated to this fix (see :meth:`RewriteStep.__hash__`'s
    own docstring).
    """

    def __hash__(self) -> int:
        """Explicit, since the dataclass-generated one would raise on ``port_mapping``.

        ``@dataclass(frozen=True)`` with the default ``eq=True`` would otherwise generate a
        ``__hash__`` that hashes every field verbatim, including ``port_mapping`` -- a
        :class:`~types.MappingProxyType`, which is unhashable (its backing ``dict`` is
        mutable even though the proxy itself is read-only). Defining ``__hash__`` here
        explicitly, in the class body, makes ``dataclass`` leave it alone rather than
        overwrite it with the broken auto-generated one. Every other field is hashed as-is;
        ``port_mapping`` is hashed as ``frozenset(port_mapping.items())`` -- order-independent,
        matching the dataclass-generated ``__eq__``, which compares ``port_mapping`` via
        plain mapping equality (also order-independent) -- so ``a == b`` still implies
        ``hash(a) == hash(b)``, the contract Phase 12's cache (which will key on
        ``RewriteStep``) needs.

        Cross-process disclaimer, stated explicitly (Phase 5 post-closing audit round 18,
        Defect 1 -- corrected from a prior version of this docstring, which cited Phase 12's
        cache-keying need without qualification and thereby over-promised): the
        ``a == b`` implies ``hash(a) == hash(b)`` contract above holds *within one process*,
        but ``hash()`` itself is not guaranteed stable *across* processes, and this fix does
        not and cannot change that. ``IssueKind`` and ``Direction`` (reached transitively
        through ``consumed_wires``, ``port_mapping``, and the deferred-issue tuples, all of
        which embed ``PortRef``/``Wire``/``ValidationIssue`` values) are ``enum.Enum``
        subclasses hashed by member *name*, and Python randomizes string hashing per process
        by default (``PYTHONHASHSEED``) -- so ``hash(step)`` computed for equal ``RewriteStep``
        values in two different processes may legitimately differ, permanently, for reasons
        that have nothing to do with this class's own hash implementation. What round 18 did
        fix is a different property entirely: the *value and order* of every field this hash
        is computed over (in particular ``removed_deferred_issues``/
        ``introduced_deferred_issues``/``deferred_issue_identity_ambiguous``, populated from
        :mod:`qufzx.diagram.validate`'s issue order) is now itself deterministic across
        processes -- see that module's docstring and ``tests/test_engine.py
        ::TestCrossProcessDeterminism``, which compares field values and order directly, by
        ``repr()``, never by ``hash()``.
        """
        return hash(
            (
                self.rule_name,
                self.match,
                self.consumed_node_ids,
                self.consumed_wires,
                self.side_condition_outcomes,
                self.dimension_constraints,
                self.scalar_introduced,
                frozenset(self.port_mapping.items()),
                self.new_node_ids,
                self.removed_deferred_issues,
                self.introduced_deferred_issues,
                frozenset(
                    (node_id, frozenset(subs.items()))
                    for node_id, subs in self.phase_substitutions.items()
                ),
                self.deferred_issue_identity_ambiguous,
            )
        )


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """The outcome of :func:`apply`: the new diagram, its new node(s), and the step provenance."""

    diagram: Diagram
    new_node_ids: tuple[NodeId, ...]
    step: RewriteStep


def _issue_key(issue: ValidationIssue) -> tuple[IssueKind, object]:
    """A ``(kind, offending ref)`` key identifying one hard-error issue for step 8's compare.

    Never the issue's ``message``: a node id embedded in a message legitimately changes
    across a rewrite (a merged node gets a fresh id), so comparing messages would flag
    every rewrite as introducing a "new" issue merely because its wording mentions a
    different id, even when the underlying defect is the pre-existing one carried over
    unchanged. ``port_ref``, ``wire``, and ``node_id`` are checked in that order -- per
    :class:`~qufzx.diagram.validate.ValidationIssue`'s own docstring, at most one is set
    for a given issue in practice, so the order only matters as a deterministic tie-break.
    """
    ref: object = issue.port_ref
    if ref is None:
        ref = issue.wire
    if ref is None:
        ref = issue.node_id
    return (issue.kind, ref)


def _translate_input_issue_key(
    issue: ValidationIssue,
    consumed_node_ids: frozenset[NodeId],
    port_mapping: Mapping[PortRef, PortRef],
    new_node_ids: tuple[NodeId, ...],
) -> tuple[IssueKind, object]:
    """``_issue_key`` of an *input*-diagram issue, translated into post-rewrite coordinates.

    Step 8 must compare like with like: ``result_hard_counts`` is already keyed on
    references as they exist in ``working`` (the post-rewrite diagram), but a naive
    ``_issue_key`` of an input-diagram issue is keyed on references as they existed
    *before* the rewrite. A consumed node's ports and node id are gone from ``working``
    entirely -- the merged node gets a fresh :class:`NodeId` and fresh port indices -- so
    an issue anchored on one of them would silently never match its post-rewrite
    counterpart even when the rewrite carried it over faithfully, making step 8 flag a
    rewrite that introduced nothing (see the module docstring, step 8).

    A ``port_ref`` or a ``wire`` endpoint on a node *not* being consumed passes through
    unchanged -- that node survives with the same id and the same port indices. A
    ``port_ref`` or wire endpoint on a *consumed* node is translated via ``port_mapping``
    when present there (a surviving leg the builder remapped); a consumed port that is
    *not* in ``port_mapping`` is the matched port itself (or a foreign match's malformed
    port) -- it has no post-rewrite counterpart at all, so it is left unchanged, which
    correctly makes it match nothing in ``result_hard_counts`` (the issue did not carry
    over because the port it was anchored on no longer exists) rather than being dropped
    silently.

    A ``node_id`` on a consumed node is translated to the sole entry of ``new_node_ids``
    when there is exactly one -- true for every Phase 5 rule (spider fusion always merges
    its two consumed nodes into exactly one new node) -- since that is the only node the
    identity could plausibly have carried over to. When ``new_node_ids`` has zero or more
    than one entries, which new node (if any) a consumed node's identity maps to is
    genuinely undecidable from the information ``apply`` has, so the id is left unchanged
    rather than guessed at: this deliberately makes the translated key impossible to match
    against anything in ``result_hard_counts`` (no node in ``working`` carries the
    original, now-removed id), so a node-id-anchored input issue on a consumed node is
    conservatively treated as *not* carried over whenever the mapping is ambiguous -- the
    same fail-closed posture step 8 already takes toward every other unrecognised case,
    at the cost (for a future multi-new-node rule only; no Phase 5 rule triggers this) of
    occasionally blocking a rewrite that did carry the issue over faithfully.
    """

    def _translate_ref(ref: PortRef) -> PortRef:
        if ref.node_id not in consumed_node_ids:
            return ref
        return port_mapping.get(ref, ref)

    if issue.port_ref is not None:
        return (issue.kind, _translate_ref(issue.port_ref))
    if issue.wire is not None:
        translated_a = _translate_ref(issue.wire.a)
        translated_b = _translate_ref(issue.wire.b)
        if translated_a == translated_b:
            # Step 5 now rejects any collapsing remap of a live wire, so this branch is
            # reachable only for an input-issue wire listed in consumed_wires (dropped,
            # never spliced) whose endpoints a foreign builder mapped anyway; falling back
            # to the untranslated wire is the fail-closed choice (it then matches nothing
            # on the result side).
            return (issue.kind, issue.wire)
        return (issue.kind, Wire(translated_a, translated_b))
    if issue.node_id is not None:
        node_id = issue.node_id
        if node_id in consumed_node_ids and len(new_node_ids) == 1:
            node_id = new_node_ids[0]
        return (issue.kind, node_id)
    return (issue.kind, None)


def _select_by_key_surplus(
    keyed_issues: tuple[tuple[tuple[IssueKind, object], ValidationIssue], ...],
    surplus: Counter[tuple[IssueKind, object]],
) -> tuple[tuple[ValidationIssue, ...], bool]:
    """Take ``surplus[key]`` issues per key from ``keyed_issues``, in the order given.

    The one selection routine behind both :attr:`RewriteStep.removed_deferred_issues` (input
    issues keyed into post-rewrite coordinates, surplus = input - result) and
    :attr:`RewriteStep.introduced_deferred_issues` (result issues in their own coordinates,
    surplus = result - input) -- the two directions of the same
    :class:`~collections.Counter` difference, so they are computed by the same code rather
    than by two similar-looking loops.

    Returns the selected issues (always the actual issue objects from ``keyed_issues``,
    never translated stand-ins) and whether the selection was *ambiguous* for any key: True
    iff some key's surplus is non-zero but strictly smaller than how many issues carry that
    key, i.e. several interchangeable issues collided and only some of them are being
    reported. See :attr:`RewriteStep.deferred_issue_identity_ambiguous` for the contract
    that flag states.
    """
    remaining = dict(surplus)
    totals = Counter(key for key, _issue in keyed_issues)
    selected: list[ValidationIssue] = []
    for key, issue in keyed_issues:
        if remaining.get(key, 0) > 0:
            selected.append(issue)
            remaining[key] -= 1
    ambiguous = any(count > 0 and count < totals[key] for key, count in surplus.items())
    return tuple(selected), ambiguous


def _remap_endpoint(
    ref: PortRef,
    consumed_node_ids: frozenset[NodeId],
    port_mapping: Mapping[PortRef, PortRef],
    rule_name: str,
) -> PortRef:
    """Remap a wire endpoint, raising if a consumed node's port was left unmapped.

    An endpoint on a node *not* being consumed is passed through unchanged -- that fallback
    is correct and required. An endpoint on a *consumed* node must appear in ``port_mapping``;
    if it does not, leaving the fallback in place would silently point the wire at a node
    step 6 is about to remove (whose removal cascade then silently drops the wire) instead of
    surfacing the problem. This also fires for a *consumed* port that a second wire or a
    boundary entry still names alongside the matched wire that consumed it -- a builder never
    maps a consumed port. Phase 5 post-closing audit round 23, Task 2: this used to be
    guarded only by a bare ``continue`` filter inside :func:`~qufzx.rewrite.match.find_matches`
    (a check that existed in code but appeared nowhere in the certificate); it is now
    ``consumed_ports_singly_claimed``, a real, re-verified side condition
    :func:`~qufzx.rewrite.match.resolve_fusion_match` decides and
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` re-checks before ever calling
    this function -- so this branch is unreachable not merely for any match ``find_matches``
    produces, but for any match the builder itself accepts at all, via either path (see
    :mod:`qufzx.rewrite.match`'s module docstring, "Match-implies-applicable and
    multiply-claimed ports"). It remains only as a defensive check against a hand-built or
    foreign ``Match`` that bypasses the builder's own re-verification -- which is not
    possible through ``spider_fusion_builder`` itself, but this function has no way to know
    it was called from there rather than some future caller that skips that step.
    """
    if ref.node_id not in consumed_node_ids:
        return ref
    if ref not in port_mapping:
        raise RewriteDomainError(
            f"rule {rule_name!r}: port {ref!r} is on a consumed node but is absent from "
            f"the builder's port_mapping; either the builder forgot to map a surviving "
            f"port, or {ref!r} names a port the match's own consumed wire already claimed "
            f"while the input diagram also listed it on a boundary"
        )
    return port_mapping[ref]


def apply(diagram: Diagram, rule: Rule, match: Match) -> RewriteResult:
    """Apply ``rule`` at ``match`` against ``diagram``, returning a new diagram and provenance.

    Never mutates ``diagram``; see the module docstring for the algorithm the step numbers
    below refer to. ``test_engine.py::TestApplyDocstringMatchesRaiseSites`` pins the count
    of raise statements lexically inside this body, so a new one added without updating
    this list fails loudly rather than drifting.

    Raises :class:`~qufzx.rewrite.rule.RewriteDomainError`:

    * Step 1: ``match``'s ``side_condition_outcomes`` do not exactly cover
      ``rule.side_conditions``, or include a failed one.
    * Step 4: the builder's ``scalar_introduced`` disagrees with ``rule.scalar_introduced``.
    * Step 5 (from :func:`_remap_endpoint`, so not counted by the meta-test): a wire or
      boundary entry names a port on a consumed node absent from ``port_mapping``.
    * Step 8: the result carries a hard-failure validation issue the input did not, under
      the multiset compare over :func:`_translate_input_issue_key`/:func:`_issue_key`.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError`:

    * Step 3: ``BuildResult.diagram`` is not, by identity, the working diagram.
    * Step 4: a reported consumed wire or node is absent from the working diagram;
      ``consumed_node_ids`` or ``new_node_ids`` repeats an id; a ``new_node_ids`` entry
      names no node in the working diagram; a ``port_mapping`` value names no real port; or
      ``port_mapping`` is not injective.
    * Step 5: ``port_mapping`` collapses both endpoints of one surviving wire onto a single
      port; or the working diagram's wire count after remapping is not the expected count.

    Further builder-output validation is deferred to Phase 11, when a second rule gives the
    generic ``BuildResult`` contract a real second consumer. Still unchecked: a builder that
    mutates ``working`` directly rather than through the returned fields (undetectable from
    ``BuildResult`` alone); a ``new_node_ids`` entry naming a pre-existing node rather than
    a created one (existence is checked, freshness is not); a duplicate in
    ``consumed_wires``; and an unused ``port_mapping`` key. The last two are inert today.
    """
    check_side_condition_coverage(match, rule.side_conditions, rule.name)

    working = diagram.copy()
    build_result = rule.builder(working, match)

    if build_result.diagram is not working:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder returned a BuildResult.diagram that is not the "
            "working diagram it was given; a builder must mutate and return that same "
            "object, never substitute a different one (see BuildResult's docstring)"
        )

    if build_result.scalar_introduced != rule.scalar_introduced:
        raise RewriteDomainError(
            f"rule {rule.name!r} declares scalar_introduced={rule.scalar_introduced!r}, "
            f"but its builder returned {build_result.scalar_introduced!r} for this match"
        )

    # Snapshotted once, as a set, rather than testing membership against ``working.wires``
    # (whatever collection backs that property) once per consumed wire -- the latter
    # re-materialises the full wire collection on every ``in`` test, making this check
    # quadratic in the number of consumed wires times the diagram's wire count.
    working_wire_set = frozenset(working.wires)
    missing_wires = [wire for wire in build_result.consumed_wires if wire not in working_wire_set]
    missing_node_ids = [
        node_id for node_id in build_result.consumed_node_ids if node_id not in working.nodes
    ]
    if missing_wires or missing_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: match does not belong to the diagram it is applied to "
            f"(consumed wire(s) absent: {missing_wires!r}; consumed node id(s) absent: "
            f"{missing_node_ids!r})"
        )

    # A3 (Phase 5 round-12 audit): a repeated entry in consumed_node_ids passes the
    # membership check above (every entry, including the repeat, is a real node id) but
    # would make step 6's ``working.remove_node(node_id)`` loop call ``remove_node`` twice
    # on the same, by-then-already-removed id -- raising
    # ``qufzx.diagram.graph.GraphGrammarError``, a different module's exception, escaping
    # this function's declared ``RewriteError`` hierarchy entirely (``apply``'s own
    # docstring promises only ``RewriteDomainError``/``RewriteGrammarError``). A duplicate
    # is a malformed request -- the same node cannot legitimately be consumed twice by one
    # match -- so it is rejected here, at the same point every other ``BuildResult`` field
    # is validated, rather than left to surface as a foreign error class deep in step 6.
    duplicate_node_ids = [
        node_id
        for node_id, count in Counter(build_result.consumed_node_ids).items()
        if count > 1
    ]
    if duplicate_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: build_result.consumed_node_ids names the same node id "
            f"more than once: {sorted(duplicate_node_ids)!r} -- a match cannot legitimately "
            "consume the same node twice"
        )

    # Every id the builder claims to have created must actually exist in ``working``, and
    # every port_mapping *value* (a builder-reported "new" port) must name a real port on a
    # node that is actually there -- these two fields are otherwise taken on faith: step 5
    # below feeds port_mapping values straight into every surviving wire and boundary entry
    # without ever checking they name anything real, and step 9's ``RewriteStep`` publishes
    # ``new_node_ids`` verbatim for Phase 6's certificate to replay against. An unvalidated
    # builder bug here (an id that was never added, or a port_mapping value with a stale or
    # out-of-range index) would otherwise surface much later as a confusing KeyError/mismatch
    # deep in remapping or certificate replay, or -- if the corrupted ref happens to alias a
    # real port by coincidence -- not surface at all, silently splicing a wire onto the wrong
    # port. Checked the same way every other ``BuildResult`` field already is in this
    # function: fail fast, close to the builder that produced the bad value.
    missing_new_node_ids = tuple(
        node_id for node_id in build_result.new_node_ids if node_id not in working.nodes
    )
    invalid_port_mapping_values = tuple(
        ref
        for ref in build_result.port_mapping.values()
        if ref.node_id not in working.nodes
        or ref.index >= len(working.nodes[ref.node_id].legs(ref.direction))
    )
    if missing_new_node_ids or invalid_port_mapping_values:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder-reported BuildResult fields do not name real "
            f"nodes/ports in the working diagram (new_node_ids absent: "
            f"{missing_new_node_ids!r}; port_mapping value(s) naming no real port: "
            f"{invalid_port_mapping_values!r})"
        )

    # Class 2 sweep (Phase 5 post-closing audit round 19, Task 4): consumed_node_ids gets a
    # duplicate check (A3, above) because a repeat there drives step 6's imperative removal
    # loop into a crash. new_node_ids drives no imperative loop, so a repeat cannot crash
    # apply() the same way -- but it is a structurally identical reference kind (a tuple of
    # NodeId the builder reports about this one rewrite) left with no duplicate check of its
    # own before this fix, and no entry in this module's own BuildResult-field table (see
    # the module docstring) recording that as a deliberate choice the way consumed_wires
    # duplicates and unused port_mapping keys are. It is not harmless: step 9 publishes
    # new_node_ids verbatim for Phase 6's certificate to replay against, and a duplicate
    # entry would misreport how many new nodes a rewrite actually created.
    duplicate_new_node_ids = [
        node_id for node_id, count in Counter(build_result.new_node_ids).items() if count > 1
    ]
    if duplicate_new_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: build_result.new_node_ids names the same node id more "
            f"than once: {sorted(duplicate_new_node_ids)!r} -- a builder creates each new "
            "node once; a repeated id here would misreport how many new nodes exist to "
            "the certificate"
        )

    # Hardening 5 (Phase 5 post-closing audit round 18): port_mapping's injectivity is
    # load-bearing but was previously unchecked. Step 5 below rejects a *single* surviving
    # wire whose two endpoints collapse onto one port (new_a == new_b), but says nothing
    # about two *different* surviving wires whose four endpoints map onto only two distinct
    # ports between them -- e.g. port_mapping sending both old_x and old_y to the same
    # new_z. Diagram._wires is a Python set (Wire equality/hash is order-independent), so
    # two such remapped wires that happen to produce the identical Wire object would
    # silently collapse into one entry in that set at ``working.add_wire`` time in step 5,
    # losing a wire with no exception anywhere -- step 5's own per-wire check cannot see
    # this, since each wire is remapped and re-added independently, one at a time.
    # ``spider_fusion_builder`` is injective by construction (each surviving old port maps
    # to exactly one new port, and ``_surviving_legs`` never emits the same old port twice),
    # so this never fires for Phase 5's one registered rule -- but a foreign or future
    # builder need not be, and the engine's generic contract in this module's own docstring
    # must not rely on every builder happening to get this right unchecked.
    if len(set(build_result.port_mapping.values())) != len(build_result.port_mapping):
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder's port_mapping is not injective -- two or more "
            "old ports map to the same new port, which would silently collapse two "
            "distinct surviving wires into one once remapped"
        )

    consumed_wire_set = frozenset(build_result.consumed_wires)
    consumed_node_ids = frozenset(build_result.consumed_node_ids)
    port_mapping = build_result.port_mapping

    # Sorted, not the raw frozenset (Phase 5 post-closing audit round 18, Defect 1):
    # ``working.wires`` is a frozenset whose hash (via Wire -> PortRef -> Direction) is
    # PYTHONHASHSEED-dependent. This loop can raise (a collapsing port_mapping, or an
    # unmapped surviving port via ``_remap_endpoint``); with more than one such wire in a
    # diagram, an unsorted iteration would surface a different one -- a different exception
    # message -- across processes. The final *set* of wires this loop leaves in ``working``
    # is unaffected by order (each wire's own fate is independent of every other's), but the
    # order in which a possible failure is reported is not, so it is sorted here too.
    for wire in sorted(working.wires, key=lambda w: w.sort_key()):
        if wire in consumed_wire_set:
            working.remove_wire(wire.a, wire.b)
            continue
        touches_consumed = (
            wire.a.node_id in consumed_node_ids or wire.b.node_id in consumed_node_ids
        )
        if not touches_consumed:
            continue
        new_a = _remap_endpoint(wire.a, consumed_node_ids, port_mapping, rule.name)
        new_b = _remap_endpoint(wire.b, consumed_node_ids, port_mapping, rule.name)
        if new_a == new_b:
            raise RewriteGrammarError(
                f"rule {rule.name!r}: port_mapping collapses wire {wire!r} onto a "
                f"single port {new_a!r}; a builder must map a surviving wire's two "
                f"endpoints to two distinct ports"
            )
        working.remove_wire(wire.a, wire.b)
        working.add_wire(new_a, new_b)

    # Hardening 6 (Phase 5 post-closing audit round 18): a cheap structural postcondition
    # on step 5 as a whole, the same posture step 8 already takes toward the *result*
    # diagram's validation issues -- catch a silently-lost wire from any cause, not just
    # the one path (a collapsing port_mapping) step 5's own per-wire check already guards.
    # Every wire in ``working`` either survived untouched (not touching a consumed node),
    # was dropped as one of ``consumed_wires`` (absorbed into the merged node, never
    # replaced), or was removed and re-added exactly once (remapped) -- so the wire count
    # can only ever shrink by exactly the number of *distinct* consumed wires, never more,
    # never less. ``working_wire_set`` is the pre-step-5 snapshot (taken above, right after
    # the builder ran and before any of steps 4-5's own mutations); distinct, not a bare
    # ``len(build_result.consumed_wires)``, since a duplicate entry there is inert (see the
    # validation contract table's own note on this) and must not be double-counted here.
    expected_wire_count = len(working_wire_set) - len(consumed_wire_set)
    actual_wire_count = len(working.wires)
    if actual_wire_count != expected_wire_count:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: wire-count postcondition violated after remapping -- "
            f"expected {expected_wire_count} wire(s) ({len(working_wire_set)} before this "
            f"rewrite minus {len(consumed_wire_set)} distinct consumed), got "
            f"{actual_wire_count}; a wire was silently lost or gained during remapping"
        )

    working.set_boundary_inputs(
        tuple(
            _remap_endpoint(ref, consumed_node_ids, port_mapping, rule.name)
            for ref in working.boundary_inputs
        )
    )
    working.set_boundary_outputs(
        tuple(
            _remap_endpoint(ref, consumed_node_ids, port_mapping, rule.name)
            for ref in working.boundary_outputs
        )
    )

    for node_id in build_result.consumed_node_ids:
        working.remove_node(node_id)

    working.multiply_scalar(build_result.scalar_introduced)

    # Round 20, Task 8: validate(diagram) and validate(working) each used to be called twice
    # -- once for .errors here, again for .deferred below -- making validate, the single most
    # expensive operation in apply, run four times total for two logical inputs. Hoisted to
    # one call per diagram; .errors/.deferred are cheap tuple filters over the same
    # ValidationReport. Also a correctness property, not only a saving: the hard-error and
    # deferred views below are now guaranteed to read off the very same validation snapshot
    # of each diagram, so they can never be computed against two differently-timed
    # (im)possible re-validations of what is nominally "the same" diagram.
    input_report = validate(diagram)
    result_report = validate(working)

    input_hard_counts = Counter(
        _translate_input_issue_key(
            issue, consumed_node_ids, port_mapping, build_result.new_node_ids
        )
        for issue in input_report.errors
    )
    result_hard_counts = Counter(_issue_key(issue) for issue in result_report.errors)
    introduced_counts = result_hard_counts - input_hard_counts
    if introduced_counts:
        # Name the actual offending (kind, ref) pairs, not merely the set of kinds -- a bare
        # kind (e.g. "dimension_policy_violation") does not say *where*, which is exactly
        # the information a user hitting this needs to find the offending node/port/wire in
        # ``working``. Sorted by ``(kind.value, repr(ref))`` for determinism, since ``ref``
        # is a heterogeneous mix of ``PortRef | Wire | NodeId | None`` with no natural order
        # of its own.
        offending = sorted(
            ((kind.value, ref, count) for (kind, ref), count in introduced_counts.items()),
            key=lambda item: (item[0], repr(item[1])),
        )
        detail = "; ".join(
            f"{kind} at {ref!r}" + (f" (x{count})" if count > 1 else "")
            for kind, ref, count in offending
        )
        raise RewriteDomainError(
            f"rule {rule.name!r}: rewrite introduced hard-error issue kind(s) not present "
            f"in the input diagram: {detail}"
        )

    # Judgement call 1 (Phase 5 post-closing audit): a rewrite is allowed to fire across a
    # DEFERRED dimension pair (see rules_library.py's module docstring, "Phase 5 judgement
    # call") -- this is not new here, and step 8 above does not change it. What step 8 never
    # did is notice when firing silently drops the resulting deferred assumption from the
    # diagram entirely (e.g. a d*e leg consumed by fusion, or overwritten onto shared_dim):
    # not a hard-error regression (deferred issues are explicitly outside step 8's multiset
    # compare, by design), but a loss of information the certificate should not paper over
    # in silence. Same translation machinery as the hard-error compare, reused for the
    # deferred set instead.
    input_deferred_keyed = tuple(
        (
            _translate_input_issue_key(
                issue, consumed_node_ids, port_mapping, build_result.new_node_ids
            ),
            issue,
        )
        for issue in input_report.deferred
    )
    result_deferred_keyed = tuple(
        (_issue_key(issue), issue) for issue in result_report.deferred
    )
    input_deferred_key_counts = Counter(key for key, _issue in input_deferred_keyed)
    result_deferred_key_counts = Counter(key for key, _issue in result_deferred_keyed)
    removed_deferred_issues, removed_ambiguous = _select_by_key_surplus(
        input_deferred_keyed, input_deferred_key_counts - result_deferred_key_counts
    )
    introduced_deferred_issues, introduced_ambiguous = _select_by_key_surplus(
        result_deferred_keyed, result_deferred_key_counts - input_deferred_key_counts
    )

    # Defect 2 (Phase 5 post-closing audit): prefer the builder's independently re-derived
    # facts over the match's own claims whenever the builder supplied them -- see
    # BuildResult's docstring. A builder that never re-verifies anything (leaves these
    # fields None) falls back to match's own fields, the pre-fix behavior, unchanged.
    step_side_condition_outcomes = (
        build_result.verified_side_condition_outcomes
        if build_result.verified_side_condition_outcomes is not None
        else match.side_condition_outcomes
    )
    step_dimension_constraints = (
        build_result.verified_dimension_constraints
        if build_result.verified_dimension_constraints is not None
        else match.dimension_constraints
    )
    step_phase_substitutions = (
        build_result.verified_phase_substitutions
        if build_result.verified_phase_substitutions is not None
        else MappingProxyType({})
    )

    step = RewriteStep(
        rule_name=rule.name,
        match=match,
        consumed_node_ids=build_result.consumed_node_ids,
        consumed_wires=build_result.consumed_wires,
        side_condition_outcomes=step_side_condition_outcomes,
        dimension_constraints=step_dimension_constraints,
        scalar_introduced=build_result.scalar_introduced,
        port_mapping=MappingProxyType(dict(port_mapping)),
        new_node_ids=build_result.new_node_ids,
        removed_deferred_issues=removed_deferred_issues,
        introduced_deferred_issues=introduced_deferred_issues,
        phase_substitutions=step_phase_substitutions,
        deferred_issue_identity_ambiguous=removed_ambiguous or introduced_ambiguous,
    )
    return RewriteResult(diagram=working, new_node_ids=build_result.new_node_ids, step=step)
