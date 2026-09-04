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

:func:`apply` is the single entry point, generic over any :class:`~qufzx.rewrite.rule.Rule`:
it reads only :class:`~qufzx.rewrite.rule.Match` and
:class:`~qufzx.rewrite.rule.BuildResult`, never a match's rule-specific fields. All
rule-specific work happens in the rule's own builder.

Algorithm.

1. :func:`~qufzx.rewrite.rule.check_side_condition_coverage` against ``rule.side_conditions``.
2. Work on ``diagram.copy()``; never mutate the diagram passed in.
3. Call ``rule.builder(working, match)``. ``build_result.diagram`` must ``is``-match
   ``working``, and the builder must have left ``working``'s wire set and both boundary
   lists exactly as ``diagram`` had them -- every other change it makes is reported through
   :class:`~qufzx.rewrite.rule.BuildResult`, never applied directly.
4. Validate the build result against ``working``: the scalar agrees with the rule's; every
   consumed wire and node exists; every ``new_node_ids`` entry exists; every
   ``port_mapping`` value names a real port on a node that outlives the rewrite; neither id
   tuple repeats; ``port_mapping`` is injective; ``verified_phase_substitutions`` names only
   consumed nodes.
5. Remap every reference, before any node is removed. A consumed wire is dropped; any other
   wire with an endpoint on a consumed node is re-added through ``port_mapping``. An
   endpoint on a consumed node absent from ``port_mapping`` raises, as does a remap
   collapsing one wire's two endpoints onto a single port. Both boundary lists are rebuilt
   through the same ``_remap_endpoint``, in place, so position is preserved -- and both are
   the input's own lists, step 3 having established the builder did not touch them. A
   wire-count postcondition, anchored on ``diagram``'s wire set rather than the
   post-builder one, then catches a wire lost during remapping.
6. Remove the consumed nodes. Step 4 rejects a ``port_mapping`` value on a consumed node,
   so no surviving reference can point at one by the time the cascade runs.
7. Multiply the scalar.
8. Verify the rewrite is not a relative regression. :func:`~qufzx.diagram.validate.validate`
   runs on the input and on the finished ``working``; a hard-failure issue in ``working``
   not accounted for in the input raises. The comparison is a *multiset* over
   ``(kind, offending ref)`` (:func:`_issue_key`), never over messages, with input-side keys
   first mapped into ``working``'s coordinates by :func:`_translate_input_issue_key`. The
   check is relative: a diagram that already carries a hard error is legitimately
   rewritable.

   ``.deferred`` issues never block. The same machinery runs over both sides' ``.deferred``
   to populate :attr:`RewriteStep.removed_deferred_issues` and
   :attr:`RewriteStep.introduced_deferred_issues`.
9. Record a :class:`RewriteStep`. Phase 6 implements replay; this module does not.

This module does not search for matches, choose which rule or match to apply, iterate to a
fixpoint, or evaluate a diagram numerically -- nothing here imports
:mod:`qufzx.semantics`.
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

    Every field comes from the ``Match`` that was applied, the ``BuildResult`` its builder
    produced, or step 8's before/after validation compare -- the three deferred-issue fields
    are the compare's, and are the only ones that read the finished working diagram.
    ``match`` is stored verbatim so Phase 6 can resolve ``rule_name`` through
    :func:`~qufzx.rewrite.rules_library.lookup_rule` and re-apply at this match directly,
    without re-running the matcher.
    """

    rule_name: str
    match: Match
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...]
    """Every dimension equality this rewrite assumed rather than verified as a syntactic
    identity, source-keyed (see :attr:`~qufzx.rewrite.rule.Match.dimension_constraints`).

    A ``DEFERRED`` entry is a recorded assumption, not a claim of satisfiability on anything
    but a degenerate point: a surviving leg of ``d**2`` forced onto a shared ``d`` records
    ``d**2 == d``, true only at ``d = 1``. Discharging such a constraint is Phase 10's job.
    """
    scalar_introduced: Scalar
    port_mapping: Mapping[PortRef, PortRef]
    new_node_ids: tuple[NodeId, ...]
    removed_deferred_issues: tuple[ValidationIssue, ...] = ()
    """Every deferred issue ``diagram`` carried with no surviving counterpart among
    ``working``'s own, in ``diagram``'s pre-rewrite coordinates.

    A multiset difference over translated keys (:func:`_translate_input_issue_key`), never
    set- or dict-keyed: two input issues translating to one key are each counted and, if
    both lack a counterpart, both reported. Not a gate; a certificate fact. Enforced by
    ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.
    """

    introduced_deferred_issues: tuple[ValidationIssue, ...] = ()
    """The other direction of the same :class:`~collections.Counter` difference: every
    deferred issue ``working`` carries with no counterpart among ``diagram``'s translated
    ones, in ``working``'s post-rewrite coordinates. Not a gate either.
    """

    phase_substitutions: Mapping[NodeId, Mapping[str, Dim]] = MappingProxyType({})
    """Per-node bindings the builder actually substituted into a phase's entries -- see
    :attr:`~qufzx.rewrite.rule.BuildResult.verified_phase_substitutions`. Empty when the
    builder supplied ``None`` (nothing re-derived) or genuinely substituted nothing.
    """

    deferred_issue_identity_ambiguous: bool = False
    """Whether the identity of the reported deferred issues above is meaningful, or only
    their count.

    Both fields take the first ``n`` occurrences of each translated key in
    :func:`~qufzx.diagram.validate.validate` order, where ``n`` is that key's Counter
    surplus. When a key's surplus equals its total occurrence count the selection is forced;
    when several issues collide on one key and only some lack a counterpart, which to name is
    unrecoverable, the selection is arbitrary but deterministic, and this flag is ``True``.

    Enforced by ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``\\
    ``::test_colliding_keys_are_flagged_ambiguous_and_pinned_to_validate_order`` and by
    ``TestCrossProcessDeterminism``.
    """

    def __hash__(self) -> int:
        """Hash every field, with ``port_mapping`` hashed as an order-independent frozenset.

        Explicit: the generated ``__hash__`` would hash a
        :class:`~types.MappingProxyType`, which is unhashable. The frozenset matches the
        generated ``__eq__``'s mapping equality, so ``a == b`` implies ``hash(a) == hash(b)``.

        Within-process only: ``enum.Enum`` members reached through ``consumed_wires``,
        ``port_mapping`` and the deferred-issue tuples hash by name, which Python randomizes
        per process. The field values and their order are deterministic across processes --
        ``tests/test_engine.py::TestCrossProcessDeterminism`` compares by ``repr()``.
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
    """A ``(kind, offending ref)`` key identifying one issue for step 8's compare.

    Never the ``message``: a node id embedded in one legitimately changes across a rewrite.
    ``port_ref``, ``wire`` and ``node_id`` are checked in that order; at most one is set in
    practice, so the order is a deterministic tie-break.

    An issue naming none of the three -- today only
    :attr:`~qufzx.diagram.validate.IssueKind.SYMBOL_ROLE_COLLISION` -- keys as
    ``(kind, None)``, so every such issue in one report shares a key and the compare sees
    only how many there are, not which names collided.
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

    A ``port_ref`` or ``wire`` endpoint on a node not being consumed passes through
    unchanged. One on a consumed node goes through ``port_mapping`` when present there; a
    consumed port absent from it is the matched port itself, has no post-rewrite
    counterpart, and is left unchanged so it matches nothing.

    A ``node_id`` on a consumed node becomes the sole entry of ``new_node_ids`` when there is
    exactly one -- true for every Phase 5 rule. With zero or several the mapping is
    undecidable here, so the id is left unchanged and the key matches nothing: fail-closed.
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

    Shared by the two directions of step 8's deferred-issue Counter difference. Returns the
    selected issues -- always the actual objects from ``keyed_issues``, never translated
    stand-ins -- and whether the selection was ambiguous for any key: True iff some key's
    surplus is non-zero but smaller than how many issues carry that key.
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

    An endpoint on a node not being consumed passes through unchanged; one on a consumed node
    must appear in ``port_mapping``. Unreachable for any match the
    ``consumed_ports_singly_claimed`` side condition accepted; a defensive check against a
    hand-built or foreign ``Match``.
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

    Never mutates ``diagram``; the module docstring numbers the steps below.
    ``test_engine.py::TestApplyDocstringMatchesRaiseSites`` pins the count of raise
    statements lexically inside this body against this list.

    Raises :class:`~qufzx.rewrite.rule.RewriteDomainError` at step 1 (``match``'s
    ``side_condition_outcomes`` do not exactly cover ``rule.side_conditions``, or include a
    failed one), step 4 (the builder's ``scalar_introduced`` disagrees with the rule's),
    step 5 (a wire or boundary entry names a port on a consumed node absent from
    ``port_mapping``, from :func:`_remap_endpoint`, so not counted by the meta-test), and
    step 8 (the result carries a hard-failure validation issue the input did not).

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` at step 3
    (``BuildResult.diagram`` is not, by identity, the working diagram; the builder edited
    the working diagram's wire set or either boundary list), step 4 (a consumed
    wire or node absent from the working diagram; ``consumed_node_ids`` or ``new_node_ids``
    repeating an id; a ``new_node_ids`` entry naming no node; a ``port_mapping`` value naming
    no real port or one on a consumed node; a non-injective ``port_mapping``;
    ``verified_phase_substitutions`` naming a node the rewrite does not consume), and step 5
    (``port_mapping`` collapsing one wire's endpoints onto a single port; the wire count not
    matching its postcondition).

    Still unchecked, and deferred to Phase 11: a builder that edits a pre-existing node's
    phase or port dimensions in place (its wire and boundary edits are caught at step 3); a
    ``new_node_ids`` entry naming a pre-existing node (existence is checked, freshness is
    not); a duplicate in ``consumed_wires``; and an unused ``port_mapping`` key.
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

    # BuildResult's contract: the builder adds the replacement node(s) and reports every
    # other change through its fields. Checked against `diagram`, the pre-builder state, so
    # the wire-count postcondition below is anchored to a baseline the builder cannot move,
    # and so a boundary edit cannot reach step 5's remap and be adopted as ground truth.
    if working.wires != diagram.wires:
        added = sorted(working.wires - diagram.wires, key=lambda w: w.sort_key())
        removed = sorted(diagram.wires - working.wires, key=lambda w: w.sort_key())
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder mutated the working diagram's wires directly "
            f"(added {added!r}; removed {removed!r}); a builder reports every wire it "
            "consumes through BuildResult.consumed_wires and never edits the wire set"
        )
    if (
        working.boundary_inputs != diagram.boundary_inputs
        or working.boundary_outputs != diagram.boundary_outputs
    ):
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder mutated the working diagram's boundary lists "
            f"directly (inputs {diagram.boundary_inputs!r} -> "
            f"{working.boundary_inputs!r}; outputs {diagram.boundary_outputs!r} -> "
            f"{working.boundary_outputs!r}); a builder never touches a boundary list, and "
            "step 5 rebuilds both from the input's through port_mapping"
        )

    if build_result.scalar_introduced != rule.scalar_introduced:
        raise RewriteDomainError(
            f"rule {rule.name!r} declares scalar_introduced={rule.scalar_introduced!r}, "
            f"but its builder returned {build_result.scalar_introduced!r} for this match"
        )

    # diagram.wires, not working.wires: equal by the step-3b check above, and naming the
    # pre-builder set keeps the postcondition's baseline independent of the builder.
    # Snapshotted once, since the property re-materialises on every access.
    working_wire_set = frozenset(diagram.wires)
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
        node_id for node_id, count in Counter(build_result.consumed_node_ids).items() if count > 1
    ]
    if duplicate_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: build_result.consumed_node_ids names the same node id "
            f"more than once: {sorted(duplicate_node_ids)!r} -- a match cannot legitimately "
            "consume the same node twice"
        )

    # Checked here rather than surfacing later as a KeyError deep in remapping -- or, if a
    # corrupted ref aliases a real port, not surfacing at all.
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

    consumed_node_ids = frozenset(build_result.consumed_node_ids)

    # A value on a consumed node names a real port here and survives step 5's remap, and
    # step 6's remove_node cascade then drops every reference to it -- a wire below the
    # wire-count postcondition's snapshot, or a boundary entry, with no exception raised.
    port_mapping_onto_consumed = tuple(
        ref for ref in build_result.port_mapping.values() if ref.node_id in consumed_node_ids
    )
    if port_mapping_onto_consumed:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder's port_mapping sends a surviving port onto "
            f"{port_mapping_onto_consumed!r}, which lies on a node this rewrite consumes; "
            "a surviving reference must be remapped onto a port that outlives the rewrite"
        )

    # Recorded verbatim onto RewriteStep.phase_substitutions, so a node id no phase was
    # read from puts a claim in the certificate the rewrite never made.
    if build_result.verified_phase_substitutions is not None:
        foreign_phase_nodes = tuple(
            node_id
            for node_id in build_result.verified_phase_substitutions
            if node_id not in consumed_node_ids
        )
        if foreign_phase_nodes:
            raise RewriteGrammarError(
                f"rule {rule.name!r}: build_result.verified_phase_substitutions names node "
                f"id(s) {sorted(foreign_phase_nodes)!r} that this rewrite does not consume; "
                "a builder substitutes into a matched node's own phase, so every key must "
                "be a consumed node id"
            )

    # Step 9 publishes new_node_ids verbatim, where a duplicate would misreport how many
    # new nodes were created.
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

    # Diagram._wires is a set, so two remapped wires producing the identical Wire would
    # collapse into one entry at add_wire time, losing a wire with no exception.
    if len(set(build_result.port_mapping.values())) != len(build_result.port_mapping):
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder's port_mapping is not injective -- two or more "
            "old ports map to the same new port, which would silently collapse two "
            "distinct surviving wires into one once remapped"
        )

    consumed_wire_set = frozenset(build_result.consumed_wires)
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

    # Every wire either survived untouched, was dropped as a consumed wire, or was removed
    # and re-added exactly once, so the count can only shrink by the number of *distinct*
    # consumed wires.
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

    # Firing across a DEFERRED pair can drop the resulting assumption from the diagram --
    # not a regression, but a loss of information the certificate records here.
    input_deferred_keyed = tuple(
        (
            _translate_input_issue_key(
                issue, consumed_node_ids, port_mapping, build_result.new_node_ids
            ),
            issue,
        )
        for issue in input_report.deferred
    )
    result_deferred_keyed = tuple((_issue_key(issue), issue) for issue in result_report.deferred)
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
