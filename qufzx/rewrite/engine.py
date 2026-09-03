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
   more, no duplicates -- or whose outcomes are not all ``passed``. Coverage is checked as
   well as passedness because ``all(...)`` over an empty tuple is vacuously True.
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
   leaving a wire pointing at a node step 6 is about to remove. Both endpoints directed to
   a single port raises ``RewriteGrammarError``. One uniform rule covers a wire to a third
   node, a pre-existing self-loop on a consumed node, and the consumed wire itself. Both
   ordered boundary lists are rebuilt through the same ``_remap_endpoint``, in place, so
   position is preserved and a boundary ref is held to a wire endpoint's standard.

   ``_remap_endpoint``'s raise is unreachable for any match
   :func:`~qufzx.rewrite.match.find_matches` returned; it stays as a defensive check
   against a hand-built or foreign ``Match``.
6. Remove the consumed nodes, only once every reference to them has been replaced.
   :meth:`~qufzx.diagram.graph.Diagram.remove_node`'s cascade is then a no-op; removing
   before step 5 would silently drop wires to third nodes before they were remapped.
7. Multiply the scalar, after every structural change.
8. Verify the rewrite is not a relative regression. :func:`~qufzx.diagram.validate.validate`
   runs on both the input and the finished ``working``; a hard-failure issue in ``working``
   not accounted for in the input raises ``RewriteDomainError``. The comparison is a
   *multiset* over ``(kind, offending ref)`` (:func:`_issue_key`) -- a set comparison
   cannot see a second independent issue of an already-present kind -- and never by
   message, since a node id in a message legitimately changes across a rewrite.

   Input-side keys are first mapped into ``working``'s coordinate space by
   :func:`_translate_input_issue_key` (via ``port_mapping`` for a port on a consumed node,
   ``new_node_ids`` for a node id, fail-closed when a rule's consumed-to-new cardinality is
   not one); otherwise any reference anchored on a consumed node would read as
   "introduced".

   The check is relative, never absolute: a diagram that already carries a hard error is
   legitimately rewritable, and only a rewrite that makes things worse is blocked.

   The compare covers ``.errors`` only. ``.deferred`` issues are an assumed diagram state,
   not a regression, so step 8 never blocks on them; the same translation machinery runs
   over both sides' ``.deferred`` to populate :attr:`RewriteStep.removed_deferred_issues`
   and :attr:`RewriteStep.introduced_deferred_issues`. That compare is a multiset too. When
   several issues collide on one key and only some have a counterpart, the one reported is
   arbitrary but deterministic (first in that side's own order), and
   :attr:`RewriteStep.deferred_issue_identity_ambiguous` says so.
9. Record provenance. A :class:`RewriteStep` carrying the rule name, the ``match`` verbatim
   (so Phase 6 replays from it rather than re-running the matcher), the match location as
   it was in the input diagram, every side condition and dimension constraint, the scalar
   introduced, and the full old-port -> new-port remapping. Side conditions and constraints
   come from ``build_result``'s ``verified_*`` fields when the builder supplied them,
   falling back to the match's own. Phase 6 implements replay; this module does not.

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
    diagram, so this record describes the rewrite as it was applied to the input.
    ``match`` is stored verbatim so Phase 6 can look ``rule_name`` up (via
    :func:`~qufzx.rewrite.rules_library.lookup_rule`) and re-apply at this match directly,
    without re-running the matcher.
    """

    rule_name: str
    match: Match
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...]
    """Every dimension equality this rewrite assumed rather than verified as a syntactic
    identity. See :attr:`~qufzx.rewrite.rule.Match.dimension_constraints` for the recording
    contract (source-keyed, at most one entry per
    :class:`~qufzx.rewrite.rule.ConstraintSource`).

    A ``DEFERRED`` entry is a recorded assumption, not a claim that it is satisfiable on
    anything but a degenerate point -- a surviving leg of dimension ``d**2`` forced onto a
    shared dimension ``d`` records ``d**2 == d``, true only at ``d = 1``. The rewrite is
    sound relative to what it recorded; distinguishing a broadly satisfiable constraint
    from a degenerate one is Phase 10's job.
    """
    scalar_introduced: Scalar
    port_mapping: Mapping[PortRef, PortRef]
    new_node_ids: tuple[NodeId, ...]
    removed_deferred_issues: tuple[ValidationIssue, ...] = ()
    """Every :attr:`~qufzx.diagram.validate.ValidationReport.deferred` issue ``diagram``
    carried with no surviving counterpart among ``working``'s own deferred issues, in
    ``diagram``'s pre-rewrite coordinates.

    Step 8 never looks at deferred issues, so a diagram-level unify assumption a rewrite
    resolves (by consuming or overwriting the leg it was recorded against) would otherwise
    vanish with nothing on the certificate to say it was there. A non-empty value is not an
    error.

    The compare is a multiset over translated keys (:func:`_translate_input_issue_key`), not
    set- or dict-keyed: two input issues translating to the same key -- one on each of two
    nodes a fusion consumes -- are each counted and, if both lack a counterpart, both
    reported. Enforced by
    ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.
    """

    introduced_deferred_issues: tuple[ValidationIssue, ...] = ()
    """The other direction of the same :class:`~collections.Counter` difference: every
    deferred issue ``working`` carries with no counterpart among ``diagram``'s own
    translated deferred issues, in ``working``'s post-rewrite coordinates.

    A rewrite can create a deferred assumption as readily as it resolves one -- forcing a
    surviving leg onto ``shared_dim`` can leave a neighbouring wire that was an exact match
    before merely deferred after. That is expected, not a defect. Neither this field nor
    :attr:`removed_deferred_issues` is a gate; both are certificate facts. Enforced by
    ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.
    """

    phase_substitutions: Mapping[NodeId, Mapping[str, Dim]] = MappingProxyType({})
    """Per-node bindings the builder actually substituted into a phase's entries -- see
    :attr:`~qufzx.rewrite.rule.BuildResult.verified_phase_substitutions`. Empty when the
    builder supplied ``None`` (nothing re-derived) or genuinely substituted nothing.
    """

    deferred_issue_identity_ambiguous: bool = False
    """Whether the identity of the reported deferred issues above is meaningful, or only
    their count.

    Both fields are populated by walking the source report's issues in
    :func:`~qufzx.diagram.validate.validate` order and taking the first ``n`` occurrences of
    each translated key, where ``n`` is that key's Counter surplus. When each key's surplus
    equals its total occurrence count, the selection is forced and every reported issue is
    uniquely the one with no counterpart.

    When several issues collide on one key and only some lack a counterpart, which to name
    is unrecoverable: the collision arises because :func:`_translate_input_issue_key` maps
    every consumed node's identity onto the one surviving merged node, and nothing in a
    :class:`~qufzx.diagram.validate.ValidationIssue` distinguishes two issues agreeing on
    ``(kind, ref)`` beyond a message this comparison never reads. The selection is then
    arbitrary but deterministic -- first in ``validate`` order, which is itself deterministic
    across processes -- the count is the meaningful part, and this flag is ``True``.

    Enforced by ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``\\
    ``::test_colliding_keys_are_flagged_ambiguous_and_pinned_to_validate_order`` and by
    ``TestCrossProcessDeterminism``.
    """

    def __hash__(self) -> int:
        """Hash every field, with ``port_mapping`` hashed as an order-independent frozenset.

        Defined explicitly because the dataclass-generated ``__hash__`` would hash
        ``port_mapping`` verbatim, and a :class:`~types.MappingProxyType` is unhashable. The
        frozenset matches the generated ``__eq__``'s mapping equality, so ``a == b`` implies
        ``hash(a) == hash(b)`` -- the contract Phase 12's cache needs.

        Within-process only: ``IssueKind`` and ``Direction`` are reached transitively
        through ``consumed_wires``, ``port_mapping`` and the deferred-issue tuples, and
        ``enum.Enum`` members hash by name, which Python randomizes per process. The *field
        values and their order* are deterministic across processes -- see
        ``tests/test_engine.py::TestCrossProcessDeterminism``, which compares by ``repr()``,
        never by ``hash()``.
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
    across a rewrite, so comparing messages would flag every rewrite as introducing a "new"
    issue merely because its wording mentions a different id. ``port_ref``, ``wire`` and
    ``node_id`` are checked in that order; at most one is set in practice, so the order is
    only a deterministic tie-break.

    An issue naming none of the three -- today only
    :attr:`~qufzx.diagram.validate.IssueKind.SYMBOL_ROLE_COLLISION`, which reports a symbol
    name and no diagram reference -- keys as ``(kind, None)``. Every such issue in one
    report therefore shares a key, so the compare below sees only how many there are, not
    which names collided.
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

    Step 8 must compare like with like: ``result_hard_counts`` is keyed on references as
    they exist in ``working``, but a naive ``_issue_key`` of an input issue is keyed on
    references as they existed before the rewrite. A consumed node's ports and node id are
    gone from ``working`` entirely, so an issue anchored on one would never match its
    post-rewrite counterpart even when the rewrite carried it over faithfully.

    A ``port_ref`` or ``wire`` endpoint on a node not being consumed passes through
    unchanged. One on a consumed node is translated via ``port_mapping`` when present there;
    a consumed port absent from ``port_mapping`` is the matched port itself and has no
    post-rewrite counterpart, so it is left unchanged and correctly matches nothing.

    A ``node_id`` on a consumed node is translated to the sole entry of ``new_node_ids``
    when there is exactly one -- true for every Phase 5 rule. With zero or several entries
    the mapping is undecidable from what ``apply`` has, so the id is left unchanged, making
    the key match nothing: fail-closed, at the cost (for a future multi-new-node rule only)
    of occasionally blocking a rewrite that did carry the issue over.
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
            # Step 5 rejects any collapsing remap of a live wire, so this is reachable only
            # for an input-issue wire listed in consumed_wires whose endpoints a foreign
            # builder mapped anyway. Falling back to the untranslated wire is fail-closed.
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

    Shared by :attr:`RewriteStep.removed_deferred_issues` (input issues keyed into
    post-rewrite coordinates, surplus = input - result) and
    :attr:`RewriteStep.introduced_deferred_issues` (result issues in their own coordinates,
    surplus = result - input) -- the two directions of one
    :class:`~collections.Counter` difference.

    Returns the selected issues -- always the actual objects from ``keyed_issues``, never
    translated stand-ins -- and whether the selection was ambiguous for any key: True iff
    some key's surplus is non-zero but smaller than how many issues carry that key. See
    :attr:`RewriteStep.deferred_issue_identity_ambiguous`.
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

    An endpoint on a node not being consumed passes through unchanged. One on a consumed
    node must appear in ``port_mapping``; otherwise the wire would silently be left pointing
    at a node step 6 is about to remove, whose removal cascade would drop it. This also
    covers a consumed port a second wire or boundary entry still names, since a builder
    never maps a consumed port.

    The raise is unreachable for any match the ``consumed_ports_singly_claimed`` side
    condition accepted, which both
    :func:`~qufzx.rewrite.match.resolve_fusion_match` and
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` decide. It remains as a
    defensive check against a hand-built or foreign ``Match``.
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
    of raise statements lexically inside this body against this list.

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

    Further builder-output validation is deferred to Phase 11. Still unchecked: a builder
    that mutates ``working`` directly rather than through the returned fields; a
    ``new_node_ids`` entry naming a pre-existing node (existence is checked, freshness is
    not); a duplicate in ``consumed_wires``; and an unused ``port_mapping`` key. The last
    two are inert today.
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

    # Snapshotted once as a set: working.wires re-materialises on every access, which would
    # make this check quadratic in consumed wires times diagram wires.
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

    # A repeated entry passes the membership check above but would make step 6's removal
    # loop call remove_node twice on an already-removed id, raising GraphGrammarError --
    # a foreign exception class escaping this function's declared hierarchy.
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

    # Step 5 feeds port_mapping values straight into every surviving wire and boundary
    # entry, and step 9 publishes new_node_ids verbatim for Phase 6 to replay against, so
    # both are checked here rather than surfacing later as a KeyError deep in remapping --
    # or, if a corrupted ref happens to alias a real port, not surfacing at all.
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

    # new_node_ids drives no imperative loop, so a repeat cannot crash apply() the way a
    # duplicate consumed_node_id does -- but step 9 publishes it verbatim for Phase 6's
    # certificate, where a duplicate would misreport how many new nodes were created.
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

    # Step 5 rejects a single surviving wire whose two endpoints collapse onto one port, but
    # says nothing about two different wires whose four endpoints map onto only two ports
    # between them. Diagram._wires is a set, so two such remapped wires producing the
    # identical Wire would silently collapse into one entry at add_wire time, losing a wire
    # with no exception. spider_fusion_builder is injective by construction; a foreign or
    # future builder need not be.
    if len(set(build_result.port_mapping.values())) != len(build_result.port_mapping):
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder's port_mapping is not injective -- two or more "
            "old ports map to the same new port, which would silently collapse two "
            "distinct surviving wires into one once remapped"
        )

    consumed_wire_set = frozenset(build_result.consumed_wires)
    consumed_node_ids = frozenset(build_result.consumed_node_ids)
    port_mapping = build_result.port_mapping

    # Sorted, not the raw frozenset: working.wires' hash is PYTHONHASHSEED-dependent, and
    # this loop can raise. The final wire set is order-independent, but which of several
    # offending wires gets reported is not.
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

    # A structural postcondition on step 5 as a whole, catching a silently-lost wire from
    # any cause rather than only the collapsing-port_mapping path checked per-wire above.
    # Every wire either survived untouched, was dropped as a consumed wire, or was removed
    # and re-added exactly once, so the count can only shrink by the number of *distinct*
    # consumed wires -- distinct, since a duplicate entry there is inert.
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

    # One validate call per diagram, not one per view: .errors/.deferred are cheap tuple
    # filters over the same report, and this guarantees the hard-error and deferred compares
    # below read off the same validation snapshot.
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
        # Name the offending (kind, ref) pairs, not just the kinds: a bare kind does not say
        # where. Sorted by (kind.value, repr(ref)), since ref is a heterogeneous mix of
        # PortRef | Wire | NodeId | None with no natural order.
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

    # A rewrite may fire across a DEFERRED dimension pair (see rules_library's module
    # docstring), and step 8 above deliberately ignores deferred issues. But firing can
    # silently drop the resulting assumption from the diagram -- not a regression, but a
    # loss of information the certificate records here, reusing the same translation
    # machinery as the hard-error compare.
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
        side_condition_outcomes=match.side_condition_outcomes,
        dimension_constraints=match.dimension_constraints,
        scalar_introduced=build_result.scalar_introduced,
        port_mapping=MappingProxyType(dict(port_mapping)),
        new_node_ids=build_result.new_node_ids,
        removed_deferred_issues=removed_deferred_issues,
        introduced_deferred_issues=introduced_deferred_issues,
        phase_substitutions=step_phase_substitutions,
        deferred_issue_identity_ambiguous=removed_ambiguous or introduced_ambiguous,
    )
    return RewriteResult(diagram=working, new_node_ids=build_result.new_node_ids, step=step)
