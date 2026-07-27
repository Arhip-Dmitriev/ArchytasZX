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

1. Copy. Never mutate the diagram passed in -- work entirely on ``diagram.copy()``
   (``working`` below) and return that. ``tests/test_engine.py`` asserts the original is
   byte-for-byte unchanged (same nodes, wires, boundaries, scalar) after a rewrite.
2. Build. Call ``rule.builder(working, match)``. Per :class:`~qufzx.rewrite.rule.BuildResult`'s
   contract, this mutates ``working`` by adding the replacement node(s) only -- it does not
   touch any wire, any boundary entry, or remove the consumed nodes.
3. Verify the match and the introduced scalar. Reject (raise
   :class:`~qufzx.rewrite.rule.RewriteDomainError`) a match with a failing side condition,
   or a builder whose ``scalar_introduced`` disagrees with the rule's declared
   ``scalar_introduced`` -- both are checked here, once, rather than trusted from the
   builder or duplicated in every rule's own builder code.
4. Remap every reference. This is the single most failure-prone part of a rewrite (see
   :mod:`qufzx.semantics.check`'s interface check, which fails on a boundary that lost its
   order or its entries before ever comparing a tensor). For every wire in ``working``
   *before* any node is removed: if it is one of ``build_result.consumed_wires``, drop it
   without replacement (it has been absorbed into the merged node). Otherwise, if either
   endpoint's node id is in ``build_result.consumed_node_ids``, remove the wire and re-add
   it with that endpoint replaced via ``build_result.port_mapping`` (an endpoint on a node
   *not* being consumed is left untouched). This single rule, applied uniformly, handles
   every case the build plan calls out: a wire to a third node, a pre-existing self-loop on
   one of the consumed nodes (both endpoints get remapped, yielding a self-loop on the
   merged node), and the consumed wire itself (dropped, never remapped). The two ordered
   boundary lists are rebuilt the same way -- ``port_mapping.get(ref, ref)`` for every
   entry, in place, so position is preserved exactly.
5. Remove the consumed nodes. Only after every wire and boundary entry that referenced them
   has already been replaced. :meth:`~qufzx.diagram.graph.Diagram.remove_node`'s cascade
   (see that module's docstring) is therefore a no-op on wires and boundary entries at this
   point -- both node ids are attached to nothing else. Removing before step 4 would corrupt
   things: the cascade would silently drop wires to third nodes before they were remapped.
6. Multiply the scalar. ``working.multiply_scalar(build_result.scalar_introduced)``, after
   every structural change, so the returned diagram's scalar accumulator is exactly the
   input's times the rule's introduced factor.
7. Record provenance. A :class:`RewriteStep` carrying the rule name, the match location
   (``build_result.consumed_node_ids`` and ``consumed_wires``, *as they were in the input
   diagram* -- these are read from ``build_result``, not re-derived from ``working`` after
   mutation), every side condition the match checked with its outcome, every dimension
   constraint the match assumed, the scalar introduced, and the full old-port -> new-port
   remapping. Phase 6's certificate module must be able to replay a rewrite from this record
   alone (rebuild the match, look up the same rule by name, and confirm the replay reproduces
   ``diagram`` and passes the oracle) -- these fields are shaped for that consumer. This
   module does not implement replay or verification itself; that is Phase 6's job.

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
from qufzx.rewrite.rule import (
    Match,
    RewriteDomainError,
    Rule,
    SideConditionOutcome,
)


@dataclass(frozen=True, slots=True)
class RewriteStep:
    """Structured provenance for one rewrite application. See the module docstring, step 7.

    Every field is drawn either from the ``Match`` that was applied or from the
    ``BuildResult`` its rule's builder produced, never re-derived from the mutated working
    diagram, so this record describes the rewrite *as it was applied to the input*.
    """

    rule_name: str
    matched_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[tuple[Dim, Dim], ...]
    scalar_introduced: Scalar
    port_mapping: Mapping[PortRef, PortRef]
    new_node_ids: tuple[NodeId, ...]


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """The outcome of :func:`apply`: the new diagram, its new node, and the step provenance."""

    diagram: Diagram
    new_node_id: NodeId
    step: RewriteStep


def apply(diagram: Diagram, rule: Rule, match: Match) -> RewriteResult:
    """Apply ``rule`` at ``match`` against ``diagram``, returning a new diagram and provenance.

    Never mutates ``diagram`` -- see the module docstring for the full algorithm. Raises
    :class:`~qufzx.rewrite.rule.RewriteDomainError` if ``match`` carries a failed side
    condition, or if the builder's introduced scalar disagrees with ``rule.scalar_introduced``.
    """
    if not match.all_side_conditions_passed:
        failed = tuple(o.name for o in match.side_condition_outcomes if not o.passed)
        raise RewriteDomainError(
            f"cannot apply rule {rule.name!r}: match has failing side condition(s) {failed}"
        )

    working = diagram.copy()
    build_result = rule.builder(working, match)

    if build_result.scalar_introduced != rule.scalar_introduced:
        raise RewriteDomainError(
            f"rule {rule.name!r} declares scalar_introduced={rule.scalar_introduced!r}, "
            f"but its builder returned {build_result.scalar_introduced!r} for this match"
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
        new_a = port_mapping.get(wire.a, wire.a)
        new_b = port_mapping.get(wire.b, wire.b)
        working.remove_wire(wire.a, wire.b)
        working.add_wire(new_a, new_b)

    working.set_boundary_inputs(
        tuple(port_mapping.get(ref, ref) for ref in working.boundary_inputs)
    )
    working.set_boundary_outputs(
        tuple(port_mapping.get(ref, ref) for ref in working.boundary_outputs)
    )

    for node_id in build_result.consumed_node_ids:
        working.remove_node(node_id)

    working.multiply_scalar(build_result.scalar_introduced)

    step = RewriteStep(
        rule_name=rule.name,
        matched_node_ids=build_result.consumed_node_ids,
        consumed_wires=build_result.consumed_wires,
        side_condition_outcomes=match.side_condition_outcomes,
        dimension_constraints=match.dimension_constraints,
        scalar_introduced=build_result.scalar_introduced,
        port_mapping=MappingProxyType(dict(port_mapping)),
        new_node_ids=(build_result.new_node_id,),
    )
    return RewriteResult(diagram=working, new_node_id=build_result.new_node_id, step=step)
