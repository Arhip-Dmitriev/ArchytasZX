"""Rewrite engine: applies rules at matches, returns new diagrams, and records step provenance.

:func:`apply` is the single entry point. It is generic over any future
:class:`~qufzx.rewrite.rule.Rule` -- it never inspects a match's rule-specific fields
(e.g. ``FusionMatch.a_id``) directly, only the generic contract
:class:`~qufzx.rewrite.rule.Match` and :class:`~qufzx.rewrite.rule.BuildResult` expose. All
of the rule-specific work (which nodes are consumed, what the replacement node's legs and
phase are, what scalar is introduced) happens inside the rule's own builder, in
:mod:`qufzx.rewrite.rules_library`; this module only knows how to splice a
:class:`~qufzx.rewrite.rule.BuildResult` into the rest of a diagram.

Algorithm.

1. Verify the match's certificate. :func:`~qufzx.rewrite.rule.check_side_condition_coverage`
   rejects (raising :class:`~qufzx.rewrite.rule.RewriteDomainError`) a match whose
   ``side_condition_outcomes`` do not name exactly ``rule.side_conditions`` -- no fewer, no
   more, no duplicates -- or whose named outcomes are not all ``passed``; a bare
   ``all(outcome.passed for outcome in ())`` is vacuously True, so checking coverage, not
   only passedness, is what actually closes that hole (see that function's docstring for the
   full account). This runs before the builder is even called; each rule's own builder
   applies the identical check against its own declared conditions (see
   ``spider_fusion_builder`` in :mod:`qufzx.rewrite.rules_library`), since a builder is
   reachable directly and not only through this function.
2. Copy. Never mutate the diagram passed in -- work entirely on ``diagram.copy()``
   (``working`` below) and return that. ``tests/test_engine.py`` asserts the original is
   byte-for-byte unchanged (same nodes, wires, boundaries, scalar) after a rewrite.
3. Build. Call ``rule.builder(working, match)``. Per :class:`~qufzx.rewrite.rule.BuildResult`'s
   contract, this mutates ``working`` by adding the replacement node(s) only -- it does not
   touch any wire, any boundary entry, or remove the consumed nodes.
4. Verify the build result belongs to this diagram. The builder's introduced scalar must
   agree with the rule's declared ``scalar_introduced``, and every one of
   ``build_result.consumed_wires`` and ``build_result.consumed_node_ids`` must actually be
   present in the working diagram -- raising :class:`~qufzx.rewrite.rule.RewriteGrammarError`
   for the latter, since a match (or a builder) naming a wire or node the diagram does not
   have is a malformed request, not a domain violation a builder computed correctly and then
   this step rejected.
5. Remap every reference. This is the single most failure-prone part of a rewrite (see
   :mod:`qufzx.semantics.check`'s interface check, which fails on a boundary that lost its
   order or its entries before ever comparing a tensor). For every wire in ``working``
   *before* any node is removed: if it is one of ``build_result.consumed_wires``, drop it
   without replacement (it has been absorbed into the merged node). Otherwise, if either
   endpoint's node id is in ``build_result.consumed_node_ids``, remove the wire and re-add
   it with that endpoint replaced via ``build_result.port_mapping`` (an endpoint on a node
   *not* being consumed is left untouched). An endpoint on a node that *is* being consumed
   must appear in ``port_mapping``; if it does not, this step raises
   :class:`~qufzx.rewrite.rule.RewriteDomainError` naming the rule and the unmapped port
   rather than silently leaving a wire pointing at a node step 6 is about to remove (whose
   removal cascade would then silently drop that wire). This single rule, applied uniformly,
   handles every case the build plan calls out: a wire to a third node, a pre-existing
   self-loop on one of the consumed nodes (both endpoints get remapped, yielding a self-loop
   on the merged node), and the consumed wire itself (dropped, never remapped). The two
   ordered boundary lists are rebuilt through this exact same ``_remap_endpoint`` helper, one
   entry at a time, in place, so position is preserved exactly and a boundary ref is held to
   the identical standard as a wire endpoint -- a ref on a node *not* being consumed passes
   through unchanged (that node survives, so the ref still names a live port), a ref on a
   *consumed* node must appear in ``port_mapping`` or this step raises. Boundaries and wires
   used to diverge here (an early version silently fell back to ``port_mapping.get(ref, ref)``
   for boundary entries, so a builder that forgot to map a surviving boundary port would not
   raise -- the ref would survive the rebuild unchanged, still naming a soon-to-be-removed
   node, and step 6's ``remove_node`` cascade would then delete it from the boundary with no
   exception, silently shrinking the returned diagram's boundary arity below the input's);
   that silent-drop path is now ruled out identically for both.
6. Remove the consumed nodes. Only after every wire and boundary entry that referenced them
   has already been replaced. :meth:`~qufzx.diagram.graph.Diagram.remove_node`'s cascade
   (see that module's docstring) is therefore a no-op on wires and boundary entries at this
   point -- both node ids are attached to nothing else. Removing before step 5 would corrupt
   things: the cascade would silently drop wires to third nodes before they were remapped.
7. Multiply the scalar. ``working.multiply_scalar(build_result.scalar_introduced)``, after
   every structural change, so the returned diagram's scalar accumulator is exactly the
   input's times the rule's introduced factor.
8. Verify the rewrite is not a relative regression. :func:`~qufzx.diagram.validate.validate`
   runs on both the original ``diagram`` and the finished ``working``; if ``working`` carries
   a hard-failure :class:`~qufzx.diagram.validate.IssueKind` that ``diagram`` did not already
   carry, this raises :class:`~qufzx.rewrite.rule.RewriteDomainError`. Compared by kind, not
   by message (a node id embedded in an issue's message legitimately changes across a
   rewrite), and relative to the input, never absolute -- a diagram that already carries a
   hard error (e.g. an unwired non-boundary leg) is legitimately rewritable, and this step
   must not block that; it only catches a rewrite that made things *worse*. One check,
   independent of steps 5 and 6 getting their own bookkeeping right, standing in for the
   whole family of structural regressions a rewrite could otherwise introduce (a dropped
   wire, a shrunk boundary, a mixed-dimension leg, a lost dimension) instead of guarding each
   one individually.
9. Record provenance. A :class:`RewriteStep` carrying the rule name, the located ``match``
   exactly as it was applied (stored verbatim, so Phase 6 replays directly from it rather
   than re-running the matcher and re-selecting a candidate by node id), the match location
   (``build_result.consumed_node_ids`` and ``consumed_wires``, *as they were in the input
   diagram* -- these are read from ``build_result``, not re-derived from ``working`` after
   mutation), every side condition the match checked with its outcome, every dimension
   constraint the match assumed, the scalar introduced, and the full old-port -> new-port
   remapping. Phase 6's certificate module must be able to replay a rewrite from this record
   alone (look the rule up by name via
   :func:`~qufzx.rewrite.rules_library.lookup_rule`, re-apply it at the stored ``match``,
   and confirm the replay reproduces ``diagram`` and passes the oracle) -- these fields are
   shaped for that consumer. This module does not implement replay or verification itself;
   that is Phase 6's job.

What this module deliberately does not do. It does not search for matches (that is
:mod:`qufzx.rewrite.match`'s job -- callers pass an already-located ``Match`` in); it does
not choose which rule or which match to apply, or iterate to a fixpoint (that is Phase 11's
strategy layer); and, per ``CLAUDE.md``, it never contracts or evaluates a diagram
numerically -- nothing in this module imports :mod:`qufzx.semantics`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, NodeId, PortRef, Wire
from qufzx.diagram.validate import validate
from qufzx.rewrite.rule import (
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
    matched_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[tuple[Dim, Dim], ...]
    scalar_introduced: Scalar
    port_mapping: Mapping[PortRef, PortRef]
    new_node_ids: tuple[NodeId, ...]


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """The outcome of :func:`apply`: the new diagram, its new node(s), and the step provenance."""

    diagram: Diagram
    new_node_ids: tuple[NodeId, ...]
    step: RewriteStep


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
    surfacing the problem.
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

    Never mutates ``diagram`` -- see the module docstring for the full algorithm. Raises
    :class:`~qufzx.rewrite.rule.RewriteDomainError` if ``match``'s side-condition outcomes
    do not exactly cover ``rule.side_conditions`` or include a failed one, if the builder's
    introduced scalar disagrees with ``rule.scalar_introduced``, or if the result carries a
    hard-failure validation issue kind ``diagram`` did not already carry (step 8). Raises
    :class:`~qufzx.rewrite.rule.RewriteGrammarError` if the match does not belong to
    ``diagram`` -- i.e. a consumed wire or node the builder reported is not actually
    present.
    """
    check_side_condition_coverage(match, rule.side_conditions, rule.name)

    working = diagram.copy()
    build_result = rule.builder(working, match)

    if build_result.scalar_introduced != rule.scalar_introduced:
        raise RewriteDomainError(
            f"rule {rule.name!r} declares scalar_introduced={rule.scalar_introduced!r}, "
            f"but its builder returned {build_result.scalar_introduced!r} for this match"
        )

    missing_wires = [wire for wire in build_result.consumed_wires if wire not in working.wires]
    missing_node_ids = [
        node_id for node_id in build_result.consumed_node_ids if node_id not in working.nodes
    ]
    if missing_wires or missing_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: match does not belong to the diagram it is applied to "
            f"(consumed wire(s) absent: {missing_wires!r}; consumed node id(s) absent: "
            f"{missing_node_ids!r})"
        )

    consumed_wire_set = frozenset(build_result.consumed_wires)
    consumed_node_ids = frozenset(build_result.consumed_node_ids)
    port_mapping = build_result.port_mapping

    for wire in tuple(working.wires):
        if wire in consumed_wire_set:
            continue
        touches_consumed = (
            wire.a.node_id in consumed_node_ids or wire.b.node_id in consumed_node_ids
        )
        if not touches_consumed:
            continue
        new_a = _remap_endpoint(wire.a, consumed_node_ids, port_mapping, rule.name)
        new_b = _remap_endpoint(wire.b, consumed_node_ids, port_mapping, rule.name)
        working.remove_wire(wire.a, wire.b)
        working.add_wire(new_a, new_b)

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

    input_hard_kinds = {issue.kind for issue in validate(diagram).errors}
    result_hard_kinds = {issue.kind for issue in validate(working).errors}
    introduced_kinds = result_hard_kinds - input_hard_kinds
    if introduced_kinds:
        raise RewriteDomainError(
            f"rule {rule.name!r}: rewrite introduced hard-error issue kind(s) "
            f"{sorted(kind.value for kind in introduced_kinds)} not present in the input "
            "diagram"
        )

    step = RewriteStep(
        rule_name=rule.name,
        match=match,
        matched_node_ids=build_result.consumed_node_ids,
        consumed_wires=build_result.consumed_wires,
        side_condition_outcomes=match.side_condition_outcomes,
        dimension_constraints=match.dimension_constraints,
        scalar_introduced=build_result.scalar_introduced,
        port_mapping=MappingProxyType(dict(port_mapping)),
        new_node_ids=build_result.new_node_ids,
    )
    return RewriteResult(diagram=working, new_node_ids=build_result.new_node_ids, step=step)
