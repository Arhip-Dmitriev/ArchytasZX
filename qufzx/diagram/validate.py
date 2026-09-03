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

"""Diagram well-formedness checks: per-port dimension agreement, boundary consistency,
port usage, generator policy conformance, and symbol-role collisions.

:func:`validate` never mutates the diagram it is given: a pure read function from a
:class:`~qufzx.diagram.graph.Diagram` to a :class:`ValidationReport`.
:mod:`qufzx.diagram.graph`'s mutators are permissive, so this is the one place a diagram's
cross-cutting invariants are checked together, in one pass, and reported as typed issues
rather than a bool.

Port usage. Every port of every node must be exactly one of: an endpoint of exactly one
wire, or an entry in the matching boundary list. Over-use is reported by
:class:`IssueKind.PORT_WIRED_TWICE`, :class:`IssueKind.PORT_WIRED_AND_BOUNDARY`, or
:class:`IssueKind.DUPLICATE_BOUNDARY_ENTRY`; a port claimed by neither is
:class:`IssueKind.PORT_UNUSED`, a hard error, since a dangling port gives the diagram no
meaning. The under-use check is skipped for a node already implicated in an
:class:`IssueKind.UNKNOWN_NODE` or :class:`IssueKind.PORT_INDEX_OUT_OF_RANGE` issue, so one
structural mistake is reported once rather than cascading.

Dimension checking is layered the way :meth:`~qufzx.algebra.dimension.Dim.unify` is,
uniformly for dimensions joined by a wire, shared by one node's legs, or tied to its phase.
Unequal and non-unifiable is a hard error -- :class:`IssueKind.DIMENSION_MISMATCH`,
:class:`IssueKind.DIMENSION_POLICY_VIOLATION`, or
:class:`IssueKind.PHASE_DIMENSION_MISMATCH` respectively. A pair ``unify`` cannot yet
resolve (``DEFERRED``, e.g. ``Dim("d")`` against ``Dim("d") * Dim("e")``, where ``d`` is a
proper subterm and so not bound) is recorded as :class:`IssueKind.DIMENSION_DEFERRED`: an
assumed constraint, neither silently accepted nor reported as an error. Phase 10's real
unifier drops in at ``Dim.unify``, changing what is deferred here without changing this
module. A bare symbol against an unrelated symbol or concrete value is not that case:
``unify`` reports ``SUCCESS`` with a binding and nothing is recorded.

``ALL_LEGS_EQUAL`` resolves a node's whole leg set through
:func:`~qufzx.algebra.dimension.unify_all`, a monotone bindings fixpoint, rather than
unifying each leg against ``all_ports[0].dim`` and discarding bindings, which would let a
jointly-unsatisfiable leg set pass and make the verdict leg-order-dependent.
``TIED_TO_LEG_DIM``'s phase/leg check resolves through those same bindings. Each residual
``DEFERRED`` pair gets its own issue. Bindings do not propagate from one node's legs to
another's -- diagram-global propagation is FULL_PLAN.md Phase 10 item (i), pinned by
``tests/test_unify_all.py::TestCrossNodePropagationDeferredToPhase10``.

:class:`IssueKind.NODE_DIMENSION_UNDETERMINED` rejects a node with no legs and no phase
vector, which carries its dimension nowhere (dimension is stored per port, never as one
global parameter). It keeps ``validate(d).is_valid`` implying every node in ``d`` is
denotable, which :mod:`qufzx.rewrite.engine`'s step 8 rests on.

:class:`IssueKind.SYMBOL_ROLE_COLLISION` (:func:`_check_symbol_role_collisions`) rejects a
name used in two symbol roles in one diagram. Substitution here is keyed by name, so
:meth:`~qufzx.algebra.phase.PhaseVector.substitute` cannot tell such a collision from the
legal case of a phase entry citing its own container dimension's symbol. The roles are
distinguished by the sympy assumptions each of :mod:`qufzx.algebra`'s four symbol
constructors stamps -- including a dimension's exponent, which is its own role, not a
phase's.

Determinism. Every pass whose issue-append order is observable iterates a snapshot sorted
by :meth:`~qufzx.diagram.graph.Wire.sort_key` /
:meth:`~qufzx.diagram.graph.PortRef.sort_key`, never a frozenset directly:
:class:`~qufzx.diagram.graph.Direction` is an ``enum.Enum`` hashed by member name, so a set
containing one iterates in a ``PYTHONHASHSEED``-dependent order.
:attr:`ValidationReport.issues`'s order is relied on downstream by
:mod:`qufzx.rewrite.engine`'s deferred-issue selection.

What this module does not do. It does not contract, evaluate, or attach numeric meaning to
a diagram (Phase 4's oracle); it does not fix anything it finds wrong (Phase 5); and it
does not yet know about bang boxes (Phase 7).
"""

from __future__ import annotations

import enum
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import DimSubstituteValue, DimSymbolKey, unify_all
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
    NODE_DIMENSION_UNDETERMINED = "node_dimension_undetermined"
    SYMBOL_ROLE_COLLISION = "symbol_role_collision"
    DIMENSION_RESOLUTION_EXHAUSTED = "dimension_resolution_exhausted"


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
        """True iff there are no hard-failure issues (deferred constraints do not fail
        validation)."""
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
    # Sorted by the hash-independent Wire.sort_key(): diagram.wires is a frozenset whose
    # hash folds in Direction's member-name hash, so iterating it directly would append
    # issues in a PYTHONHASHSEED-dependent order, breaking the "first in validate order"
    # selection qufzx.rewrite.engine relies on.
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
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
                    message=(
                        f"wire {wire!r} joins mismatched dimensions {port_a.dim} and {port_b.dim}"
                    ),
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
    # Sorted for the same reason as _check_wire_dimensions: an unsorted pass would make the
    # insertion order into wired_counts below process-dependent, and with it the order
    # PORT_WIRED_TWICE issues are appended in.
    wired_refs: list[PortRef] = []
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        wired_refs.append(wire.a)
        wired_refs.append(wire.b)
    wired_counts = Counter(wired_refs)
    for ref, count in sorted(wired_counts.items(), key=lambda item: item[0].sort_key()):
        if count > 1:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PORT_WIRED_TWICE,
                    message=f"{ref} is wired {count} times",
                    port_ref=ref,
                )
            )

    wired_set = set(wired_refs)
    # No sort needed: boundary_inputs/boundary_outputs are ordered tuples, and Counter
    # iteration order is insertion order.
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
    # A ref on both boundary lists (already reported above as DUPLICATE_BOUNDARY_ENTRY) is
    # resolved only once, or one malformed reference would append two identical issues,
    # inflating counts the engine's deferred-issue bookkeeping treats as load-bearing.
    # dict.fromkeys dedupes while preserving first-appearance order, inputs before outputs.
    for ref in dict.fromkeys((*diagram.boundary_inputs, *diagram.boundary_outputs)):
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
                        message=(f"{ref} is neither wired nor present on the boundary"),
                        port_ref=ref,
                    )
                )


def _classify_symbol_role(symbol: sp.Symbol) -> str | None:
    """Which namespace ``symbol`` belongs to, from its assumptions.

    ``qufzx.algebra``'s four symbol constructors each stamp a distinct assumption signature,
    matched here against sympy's computed closure. Round-tripped per constructor by
    ``tests/test_validate.py``'s ``TestSymbolConstructorRolesRoundTrip``:

    * :meth:`~qufzx.algebra.dimension.Dim.symbol` (``positive=True, integer=True``) --
      "dimension". Signature ``integer and positive``.
    * a dimension's exponent, from :meth:`~qufzx.algebra.dimension.Dim.__pow__` via
      ``_exponent_symbol`` (``integer=True, nonnegative=True``, never ``positive``: an
      exponent of 0 is legal, a dimension of 0 is not) -- "exponent". Signature ``integer
      and not positive``; ``nonnegative`` holds for both and does not discriminate.
    * :meth:`~qufzx.algebra.phase.Phase.symbol` (``real=True``) -- "phase". Signature ``real
      and not integer``; a dimension or exponent symbol is ``real`` by closure.
    * :meth:`~qufzx.algebra.scalar.Scalar.symbol` (``complex=True``) -- "scalar". Signature
      ``complex and not real``; the other three are ``complex`` by closure.

    Each branch tests only the keys its constructor sets, never a derived one, so a fifth
    constructor setting a different pair falls through to ``None`` -- unclassified rather
    than aliased into an existing role. ``None`` also covers a bare, assumption-free
    ``Symbol``.

    :func:`_check_symbol_role_collisions` decides which pairs of roles sharing one name are
    a genuine collision.
    """
    assumptions = symbol.assumptions0
    is_integer = bool(assumptions.get("integer"))
    is_positive = bool(assumptions.get("positive"))
    is_real = bool(assumptions.get("real"))
    is_complex = bool(assumptions.get("complex"))
    if is_integer and is_positive:
        return "dimension"
    if is_integer and not is_positive:
        return "exponent"
    if is_real and not is_integer:
        return "phase"
    if is_complex and not is_real:
        return "scalar"
    return None


def _check_symbol_role_collisions(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    # A same-named symbol of two different roles is two distinct sympy Symbol objects, so a
    # by-name substitution (every substitute() here is) silently rewrites both. All six
    # unordered cross-role pairs over {dimension, exponent, phase, scalar} are genuine
    # collisions, since the four roles accept different substitution domains: positive
    # integers, nonnegative integers, reals mod one turn, and arbitrary complex.
    #
    # The same role used twice under one name is legitimate reuse, not a collision (two
    # ports sharing a dimension symbol, or a root-of-unity phase entry over its own node's
    # dimension symbol). The setdefault below never records a second entry for a role
    # already seen under that name, so that case never becomes a pair for len(by_role) > 1.
    roles: dict[str, dict[str, sp.Symbol]] = {}

    def _note(expr: sp.Expr) -> None:
        for symbol in expr.free_symbols:
            role = _classify_symbol_role(symbol)
            if role is None:
                continue
            roles.setdefault(str(symbol.name), {}).setdefault(role, symbol)

    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            _note(port.dim.to_sympy())
        if node.phase is not None:
            _note(node.phase.dim.to_sympy())
            for entry in node.phase.entries().values():
                _note(entry.to_sympy_turns())
    _note(diagram.scalar.to_sympy())

    for name, by_role in sorted(roles.items()):
        if len(by_role) > 1:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.SYMBOL_ROLE_COLLISION,
                    message=(
                        f"symbol {name!r} is used as more than one role in this diagram: "
                        f"{sorted(by_role)}"
                    ),
                )
            )


def _check_generator_policy(node: Node, issues: list[ValidationIssue]) -> None:
    gen = node.generator_type

    # Dimension is stored per port, so a node with zero legs and no phase vector carries it
    # nowhere at all. qufzx.semantics.denote already refuses such a node; stating the same
    # fact here, as a hard error, is what makes validate(d).is_valid imply every node in d
    # is denotable -- the invariant qufzx.rewrite.engine's apply step 8 depends on.
    if node.num_inputs == 0 and node.num_outputs == 0 and node.phase is None:
        issues.append(
            ValidationIssue(
                kind=IssueKind.NODE_DIMENSION_UNDETERMINED,
                message=(
                    f"node {node.id!r} ({gen.name}) has no legs and no phase vector; its "
                    "dimension cannot be determined"
                ),
                node_id=node.id,
            )
        )

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

    # leg_unify is the only place this function resolves what dimension a node's legs
    # jointly agree on; both the DIMENSION_POLICY_VIOLATION/DEFERRED branch below and the
    # phase-vs-legs branch after it read from it, so there is one leg-resolution
    # computation, not two that can drift apart.
    all_ports = (*node.inputs, *node.outputs)
    leg_unify = unify_all([port.dim for port in all_ports]) if all_ports else None

    if gen.dimension_policy is DimensionPolicy.ALL_LEGS_EQUAL and leg_unify is not None:
        if leg_unify.is_failure:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DIMENSION_POLICY_VIOLATION,
                    message=(
                        f"node {node.id!r} ({gen.name}) requires all legs to share one "
                        "dimension, but its leg dimensions do not jointly unify: "
                        f"{sorted(str(port.dim) for port in all_ports)}"
                    ),
                    node_id=node.id,
                )
            )
        elif leg_unify.exhausted:
            # unify_all's pass budget ran out before its bindings fixpoint stabilised: an
            # undecided node, not a decided-and-fine one. A hard error, not deferred -- a
            # deferred issue means the question itself is genuinely open, whereas an
            # exhausted budget has not reached that question at all.
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DIMENSION_RESOLUTION_EXHAUSTED,
                    message=(
                        f"node {node.id!r} ({gen.name}) leg-dimension resolution did not "
                        "stabilise within unify_all's pass budget; "
                        f"{len(leg_unify.residual_pairs)} pair(s) were still unresolved on "
                        "the final pass -- nothing about this node's legs was decided"
                    ),
                    node_id=node.id,
                )
            )
        elif leg_unify.is_deferred:
            for assumed, equal_to in leg_unify.residual_pairs:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.DIMENSION_DEFERRED,
                        message=(
                            f"node {node.id!r} ({gen.name}) assumes {assumed} == {equal_to} "
                            "across its legs (deferred, not yet decided)"
                        ),
                        node_id=node.id,
                        deferred=True,
                    )
                )
        # A SUCCESS with non-empty leg_unify.declined_bindings (legs `d` and `e` unifying
        # only by binding d := e) falls through without an issue. This module has no
        # reporting path for "SUCCESS, but only under an assumption", unlike
        # qufzx.rewrite.match, which records it as a BOUND DimensionConstraint. Closing that
        # asymmetry is a certificate-shape change deferred to Phase 10; the assumption is
        # still carried on UnifyAllResult.declined_bindings, and
        # tests/test_symbolic_dimension_sweep.py pins today's behavior.

    if node.phase is not None:
        if gen.phase_schema is PhaseSchema.NONE:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PHASE_NOT_PERMITTED,
                    message=(
                        f"node {node.id!r} ({gen.name}) carries a phase but its type is phase-free"
                    ),
                    node_id=node.id,
                )
            )
        elif (
            gen.phase_schema is PhaseSchema.TIED_TO_LEG_DIM
            and leg_unify is not None
            and not leg_unify.is_failure
            and not leg_unify.exhausted
        ):
            # A FAILURE or exhausted leg set already has its own finding above, and neither
            # leaves a coherent shared leg dimension to check the phase against, so this
            # branch is skipped rather than manufacturing a second, arbitrary finding.
            #
            # resolved_leg_dim is the node's first leg (input-then-output, original order --
            # an arbitrary but fixed seed, the role shared_dim plays in match.py) with
            # leg_unify's accumulated bindings substituted in. Under SUCCESS this equals what
            # substituting into any leg would give; under DEFERRED it is one representative
            # among a residual-equal set.
            resolved_leg_dim = all_ports[0].dim
            resolved_phase_dim = node.phase.dim
            if leg_unify.bindings:
                bindings = cast(Mapping[DimSymbolKey, DimSubstituteValue], leg_unify.bindings)
                resolved_leg_dim = resolved_leg_dim.substitute(bindings)
                resolved_phase_dim = resolved_phase_dim.substitute(bindings)
            if resolved_phase_dim != resolved_leg_dim:
                result = resolved_phase_dim.unify(resolved_leg_dim)
                if result.is_failure:
                    issues.append(
                        ValidationIssue(
                            kind=IssueKind.PHASE_DIMENSION_MISMATCH,
                            message=(
                                f"node {node.id!r} ({gen.name}) phase vector is over "
                                f"{resolved_phase_dim}, but its legs share dimension "
                                f"{resolved_leg_dim}"
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
                                f"{resolved_phase_dim} == leg dimension {resolved_leg_dim} "
                                "(deferred, not yet decided)"
                            ),
                            node_id=node.id,
                            deferred=True,
                        )
                    )
                # A binding this phase check produces is deliberately not fed back into
                # leg_unify/resolved_leg_dim. Unlike match.py's condition 7, this function
                # decides no applicability: the legs' question is already fully settled
                # before the phase is examined, and nothing later re-reads resolved_leg_dim.
                # Feeding the binding back could only sharpen the wording of an
                # already-emitted DIMENSION_DEFERRED residual, never change a verdict.


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
    _check_symbol_role_collisions(diagram, issues)
    return ValidationReport(tuple(issues))


def validate_or_raise(diagram: Diagram) -> ValidationReport:
    """Run :func:`validate` and raise ValidationFailedError if any hard-failure issue is found.

    Returns the report (which may still carry deferred issues) on success.
    """
    report = validate(diagram)
    if not report.is_valid:
        raise ValidationFailedError(report)
    return report
