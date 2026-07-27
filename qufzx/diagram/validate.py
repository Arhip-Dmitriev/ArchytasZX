"""Diagram well-formedness checks: per-port dimension agreement, boundary consistency,
and generator policy conformance.

:func:`validate` never mutates the diagram it is given -- it is a pure read function
from a :class:`~qufzx.diagram.graph.Diagram` to a :class:`ValidationReport`. This is
deliberate: :mod:`qufzx.diagram.graph`'s mutators are intentionally permissive (see
that module's docstring), so this is the one place all of a diagram's cross-cutting
invariants are checked together, in one pass, and reported as a structured list of
typed issues rather than a bare pass/fail bool. Phase 5's rewrite matcher and Phase 6's
certificates both need that structure (which port, which node, which wire, and why) to
do their own work, not just a yes/no.

Dimension checking is layered exactly the way :meth:`~qufzx.algebra.dimension.Dim.unify`
is layered. A wire whose two ports carry unequal, non-unifiable dimensions is a hard
:class:`IssueKind.DIMENSION_MISMATCH` error. A wire whose dimensions cannot yet be
resolved (``unify`` returns ``DEFERRED``, e.g. ``Dim("d")`` against ``Dim("e")``) is
recorded as :class:`IssueKind.DIMENSION_DEFERRED`, an *assumed* constraint, not silently
accepted as valid and not reported as an error either -- this is exactly the seam Phase
10's real unifier is meant to drop into: replacing the placeholder in ``Dim.unify``
changes what gets deferred here without this module changing at all.

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
        for port in all_ports:
            if port.dim != shared_dim:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.DIMENSION_POLICY_VIOLATION,
                        message=(
                            f"node {node.id!r} ({gen.name}) requires all legs to share one "
                            f"dimension, but found {shared_dim} and {port.dim}"
                        ),
                        node_id=node.id,
                    )
                )
                break

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
