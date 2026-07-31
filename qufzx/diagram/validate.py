"""Diagram well-formedness checks: per-port dimension agreement, boundary consistency,
port usage, and generator policy conformance.

:func:`validate` never mutates the diagram it is given -- it is a pure read function
from a :class:`~qufzx.diagram.graph.Diagram` to a :class:`ValidationReport`. This is
deliberate: :mod:`qufzx.diagram.graph`'s mutators are intentionally permissive (see
that module's docstring), so this is the one place all of a diagram's cross-cutting
invariants are checked together, in one pass, and reported as a structured list of
typed issues rather than a bare pass/fail bool. Phase 5's rewrite matcher and Phase 6's
certificates both need that structure (which port, which node, which wire, and why) to
do their own work, not just a yes/no.

Every port of every node must be exactly one of: an endpoint of exactly one wire, or an
entry in the matching boundary list. Ports that are wired or claimed by the boundary
more than once are reported by the existing over-use kinds
(:class:`IssueKind.PORT_WIRED_TWICE`, :class:`IssueKind.PORT_WIRED_AND_BOUNDARY`,
:class:`IssueKind.DUPLICATE_BOUNDARY_ENTRY`); a port claimed by neither is
:class:`IssueKind.PORT_UNUSED`, a hard error, since a dangling port has no meaning to
give the diagram. This under-use check is skipped for a port on a node already
implicated in an :class:`IssueKind.UNKNOWN_NODE` or
:class:`IssueKind.PORT_INDEX_OUT_OF_RANGE` issue, so one structural mistake (e.g. a wire
naming an out-of-range index) is reported once rather than cascading into spurious
PORT_UNUSED findings for that node's real legs.

Dimension checking is layered exactly the way :meth:`~qufzx.algebra.dimension.Dim.unify`
is layered, and this layering applies uniformly whether the two dimensions being
compared are joined by a wire or shared by one node's legs (or its phase). A pair of
dimensions that are unequal and non-unifiable is a hard error --
:class:`IssueKind.DIMENSION_MISMATCH` for a wire, :class:`IssueKind.DIMENSION_POLICY_VIOLATION`
for a node's legs, :class:`IssueKind.PHASE_DIMENSION_MISMATCH` for a node's phase. A pair
whose dimensions cannot yet be resolved (``unify`` returns ``DEFERRED``, e.g. ``Dim("d")``
against ``Dim("d") * Dim("e")``, where ``d`` occurs as a proper subterm of the other side
and is therefore not bound) is recorded as :class:`IssueKind.DIMENSION_DEFERRED` in every
one of these cases, an *assumed* constraint, not silently accepted as valid and not
reported as an error either -- this is exactly the seam Phase 10's real unifier is meant
to drop into: replacing the placeholder in ``Dim.unify`` changes what gets deferred here
without this module changing at all. A bare symbol against an unrelated symbol or
concrete value -- e.g. ``Dim("d")`` against ``Dim("e")``, or against ``Dim(3)`` -- is not
this case: ``unify`` reports ``SUCCESS`` with a binding (``d := e``, or ``d := 3``), and
this module records no issue at all for it, the same as any other syntactic-identity
success; only the *unresolved* residual shape above reaches ``DIMENSION_DEFERRED``.

What this module does not do. It does not contract, evaluate, or attach any numeric
meaning to a diagram (that is Phase 4's oracle); it does not attempt to fix or rewrite
anything it finds wrong (that is Phase 5); and it does not yet know about bang boxes
(Phase 7 extends this module's scoping checks when that generator arrives).
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field

from qufzx.algebra.dimension import Dim
from qufzx.diagram.generators import DimensionPolicy, PhaseSchema
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef, Wire


class ValidateError(Exception):
    """Base class for all errors raised by this module."""


class ValidationFailedError(ValidateError):
    """Raised by :func:`validate_or_raise` when a diagram's report contains an error-level issue.

    Carries the offending :class:`ValidationReport` as :attr:`report`.
    """

    def __init__(self, report: ValidationReport) -> None:
        """Build the error from the failing report, formatting its issues into the message."""
        self.report = report
        summary = "; ".join(issue.message for issue in report.errors)
        super().__init__(f"diagram failed validation with {len(report.errors)} issue(s): {summary}")


class IssueKind(enum.Enum):
    """The machine-readable kind of a single validation finding."""

    UNKNOWN_NODE = "unknown_node"
    PORT_INDEX_OUT_OF_RANGE = "port_index_out_of_range"
    DIMENSION_MISMATCH = "dimension_mismatch"
    DIMENSION_DEFERRED = "dimension_deferred"
    PORT_WIRED_TWICE = "port_wired_twice"
    PORT_WIRED_AND_BOUNDARY = "port_wired_and_boundary"
    PORT_UNUSED = "port_unused"
    DUPLICATE_BOUNDARY_ENTRY = "duplicate_boundary_entry"
    BOUNDARY_DIRECTION_MISMATCH = "boundary_direction_mismatch"
    LEG_POLICY_VIOLATION = "leg_policy_violation"
    DIMENSION_POLICY_VIOLATION = "dimension_policy_violation"
    PHASE_DIMENSION_MISMATCH = "phase_dimension_mismatch"
    PHASE_NOT_PERMITTED = "phase_not_permitted"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One finding: a machine-readable kind, the offending reference(s), and a message.

    Exactly one of ``node_id``, ``port_ref``, or ``wire`` is typically the primary
    offender for a given ``kind``; the others are left ``None``. ``deferred`` is True
    only for :attr:`IssueKind.DIMENSION_DEFERRED`, marking this as an assumed
    constraint rather than a hard failure -- see the module docstring.
    """

    kind: IssueKind
    message: str
    node_id: NodeId | None = None
    port_ref: PortRef | None = None
    wire: Wire | None = None
    deferred: bool = False


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The full set of findings from one :func:`validate` call."""

    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """The non-deferred (hard-failure) issues."""
        return tuple(issue for issue in self.issues if not issue.deferred)

    @property
    def deferred(self) -> tuple[ValidationIssue, ...]:
        """The deferred (assumed-constraint) issues."""
        return tuple(issue for issue in self.issues if issue.deferred)

    @property
    def is_valid(self) -> bool:
        """True iff there are no hard-failure issues (deferred constraints do not fail validation)."""
        return not self.errors


def _resolve(diagram: Diagram, ref: PortRef, issues: list[ValidationIssue]) -> Port | None:
    """Resolve a PortRef against the diagram, appending an issue and returning None on failure."""
    node = diagram.nodes.get(ref.node_id)
    if node is None:
        issues.append(
            ValidationIssue(
                kind=IssueKind.UNKNOWN_NODE,
                message=f"{ref} refers to unknown node {ref.node_id!r}",
                port_ref=ref,
            )
        )
        return None
    legs = node.legs(ref.direction)
    if ref.index >= len(legs):
        issues.append(
            ValidationIssue(
                kind=IssueKind.PORT_INDEX_OUT_OF_RANGE,
                message=(
                    f"{ref} index {ref.index} out of range for node {ref.node_id!r} "
                    f"{ref.direction.value} legs (has {len(legs)})"
                ),
                port_ref=ref,
            )
        )
        return None
    return legs[ref.index]


def _check_wire_dimensions(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    for wire in diagram.wires:
        port_a = _resolve(diagram, wire.a, issues)
        port_b = _resolve(diagram, wire.b, issues)
        if port_a is None or port_b is None:
            continue
        if port_a.dim == port_b.dim:
            continue
        result = port_a.dim.unify(port_b.dim)
        if result.is_failure:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DIMENSION_MISMATCH,
                    message=f"wire {wire!r} joins mismatched dimensions {port_a.dim} and {port_b.dim}",
                    wire=wire,
                )
            )
        elif result.is_deferred:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DIMENSION_DEFERRED,
                    message=(
                        f"wire {wire!r} assumes {port_a.dim} == {port_b.dim} "
                        "(deferred, not yet decided)"
                    ),
                    wire=wire,
                    deferred=True,
                )
            )
        # SUCCESS: dimensions unify without contradiction; nothing to report.


def _check_port_usage(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    wired_refs: list[PortRef] = []
    for wire in diagram.wires:
        wired_refs.append(wire.a)
        wired_refs.append(wire.b)
    wired_counts = Counter(wired_refs)
    for ref, count in wired_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PORT_WIRED_TWICE,
                    message=f"{ref} is wired {count} times",
                    port_ref=ref,
                )
            )

    wired_set = set(wired_refs)
    boundary_counts = Counter(diagram.boundary_inputs) + Counter(diagram.boundary_outputs)
    for ref, count in boundary_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DUPLICATE_BOUNDARY_ENTRY,
                    message=f"{ref} appears {count} times across the boundary lists",
                    port_ref=ref,
                )
            )
        if ref in wired_set:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PORT_WIRED_AND_BOUNDARY,
                    message=f"{ref} is both wired and on the boundary",
                    port_ref=ref,
                )
            )

    for ref in diagram.boundary_inputs:
        if ref.direction is not Direction.INPUT:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.BOUNDARY_DIRECTION_MISMATCH,
                    message=f"{ref} is in boundary_inputs but is not an INPUT-direction port",
                    port_ref=ref,
                )
            )
    for ref in diagram.boundary_outputs:
        if ref.direction is not Direction.OUTPUT:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.BOUNDARY_DIRECTION_MISMATCH,
                    message=f"{ref} is in boundary_outputs but is not an OUTPUT-direction port",
                    port_ref=ref,
                )
            )
    for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs):
        _resolve(diagram, ref, issues)

    broken_node_ids = {
        issue.port_ref.node_id
        for issue in issues
        if issue.kind in (IssueKind.UNKNOWN_NODE, IssueKind.PORT_INDEX_OUT_OF_RANGE)
        and issue.port_ref is not None
    }
    boundary_set = set(boundary_counts)
    for node_id, node in diagram.nodes.items():
        if node_id in broken_node_ids:
            continue
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index in range(len(node.legs(direction))):
                ref = PortRef(node_id, direction, index)
                if ref in wired_set or ref in boundary_set:
                    continue
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.PORT_UNUSED,
                        message=(
                            f"{ref} is neither wired nor present on the boundary"
                        ),
                        port_ref=ref,
                    )
                )


def _check_generator_policy(node: Node, issues: list[ValidationIssue]) -> None:
    gen = node.generator_type
    if not gen.leg_policy.allows(node.num_inputs, node.num_outputs):
        issues.append(
            ValidationIssue(
                kind=IssueKind.LEG_POLICY_VIOLATION,
                message=(
                    f"node {node.id!r} ({gen.name}) has {node.num_inputs} inputs / "
                    f"{node.num_outputs} outputs, violating its leg policy"
                ),
                node_id=node.id,
            )
        )

    all_ports = (*node.inputs, *node.outputs)
    shared_dim: Dim | None = all_ports[0].dim if all_ports else None

    if gen.dimension_policy is DimensionPolicy.ALL_LEGS_EQUAL and shared_dim is not None:
        strongest: ValidationIssue | None = None
        for port in all_ports:
            if port.dim == shared_dim:
                continue
            result = port.dim.unify(shared_dim)
            if result.is_failure:
                strongest = ValidationIssue(
                    kind=IssueKind.DIMENSION_POLICY_VIOLATION,
                    message=(
                        f"node {node.id!r} ({gen.name}) requires all legs to share one "
                        f"dimension, but found {shared_dim} and {port.dim}"
                    ),
                    node_id=node.id,
                )
                break
            if result.is_deferred and strongest is None:
                strongest = ValidationIssue(
                    kind=IssueKind.DIMENSION_DEFERRED,
                    message=(
                        f"node {node.id!r} ({gen.name}) assumes {shared_dim} == {port.dim} "
                        "across its legs (deferred, not yet decided)"
                    ),
                    node_id=node.id,
                    deferred=True,
                )
        if strongest is not None:
            issues.append(strongest)

    if node.phase is not None:
        if gen.phase_schema is PhaseSchema.NONE:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PHASE_NOT_PERMITTED,
                    message=f"node {node.id!r} ({gen.name}) carries a phase but its type is phase-free",
                    node_id=node.id,
                )
            )
        elif (
            gen.phase_schema is PhaseSchema.TIED_TO_LEG_DIM
            and shared_dim is not None
            and node.phase.dim != shared_dim
        ):
            result = node.phase.dim.unify(shared_dim)
            if result.is_failure:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.PHASE_DIMENSION_MISMATCH,
                        message=(
                            f"node {node.id!r} ({gen.name}) phase vector is over "
                            f"{node.phase.dim}, but its legs share dimension {shared_dim}"
                        ),
                        node_id=node.id,
                    )
                )
            elif result.is_deferred:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.DIMENSION_DEFERRED,
                        message=(
                            f"node {node.id!r} ({gen.name}) assumes phase dimension "
                            f"{node.phase.dim} == leg dimension {shared_dim} "
                            "(deferred, not yet decided)"
                        ),
                        node_id=node.id,
                        deferred=True,
                    )
                )


def validate(diagram: Diagram) -> ValidationReport:
    """Run every well-formedness check against ``diagram`` and return the full report.

    Pure: never mutates ``diagram``. See the module docstring for what is checked and
    the layering of hard-failure versus deferred dimension issues.
    """
    issues: list[ValidationIssue] = []
    _check_wire_dimensions(diagram, issues)
    _check_port_usage(diagram, issues)
    for node in diagram.nodes.values():
        _check_generator_policy(node, issues)
    return ValidationReport(tuple(issues))


def validate_or_raise(diagram: Diagram) -> ValidationReport:
    """Run :func:`validate` and raise ValidationFailedError if any hard-failure issue is found.

    Returns the report (which may still carry deferred issues) on success.
    """
    report = validate(diagram)
    if not report.is_valid:
        raise ValidationFailedError(report)
    return report
