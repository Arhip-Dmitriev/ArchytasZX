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
port usage, generator policy conformance, symbol-role collisions, and the parameter
environment.

:func:`validate` is a pure read function from a :class:`~archytaszx.diagram.graph.Diagram` to a
:class:`ValidationReport`. It is the one place a diagram's cross-cutting invariants are
checked together, in one pass, and reported as typed issues rather than a bool.

Port usage. Every port of every node must be exactly one of: an endpoint of exactly one
wire, or an entry in the matching boundary list. Over-use is
:class:`IssueKind.PORT_WIRED_TWICE`, :class:`IssueKind.PORT_WIRED_AND_BOUNDARY`, or
:class:`IssueKind.DUPLICATE_BOUNDARY_ENTRY`; a port claimed by neither is
:class:`IssueKind.PORT_UNUSED`, a hard error. The under-use check is skipped for a node
already implicated in an :class:`IssueKind.UNKNOWN_NODE` or
:class:`IssueKind.PORT_INDEX_OUT_OF_RANGE` issue.

Dimension checking is layered the way :meth:`~archytaszx.algebra.dimension.Dim.unify` is,
uniformly for dimensions joined by a wire, shared by one node's legs, or tied to its phase.
Unequal and non-unifiable is a hard error -- :class:`IssueKind.DIMENSION_MISMATCH`,
:class:`IssueKind.DIMENSION_POLICY_VIOLATION`, or
:class:`IssueKind.PHASE_DIMENSION_MISMATCH`. A pair ``unify`` cannot resolve is
:class:`IssueKind.DIMENSION_DEFERRED`; a pair holding only under a binding is
:class:`IssueKind.DIMENSION_BOUND`. Both carry ``deferred=True`` and never fail validation,
and a resolution can report both at once.

``ALL_LEGS_EQUAL`` resolves a node's whole leg set through
:func:`~archytaszx.algebra.dimension.unify_all`, a monotone bindings fixpoint;
``TIED_TO_LEG_DIM``'s phase/leg check resolves through those same bindings. Each residual
``DEFERRED`` pair gets its own issue. Bindings do not propagate from one node's legs to
another's -- diagram-global propagation is FULL_PLAN.md Phase 10 item (i), pinned by
``tests/test_unify_all.py::TestCrossNodePropagationDeferredToPhase10``.

:class:`IssueKind.NODE_DIMENSION_UNDETERMINED` rejects a node with no legs and no phase
vector, which carries its dimension nowhere. It keeps ``validate(d).is_valid`` implying
every node in ``d`` is denotable, which :mod:`archytaszx.rewrite.engine`'s step 8 rests on.

:class:`IssueKind.SYMBOL_ROLE_COLLISION` rejects a name used in two symbol roles in one
diagram. The roles are read off the sympy assumptions each of :mod:`archytaszx.algebra`'s four
symbol constructors stamps -- including a dimension's exponent, which is its own role.

Parameter environment. Every name :attr:`~archytaszx.diagram.graph.Diagram.parameters` binds
must be a symbol the diagram carries, in exactly one role, at a value inside that role's
domain. A name no symbol carries is the deferred
:class:`IssueKind.PARAMETER_UNKNOWN_SYMBOL`; a value outside the role's domain is the hard
:class:`IssueKind.PARAMETER_VALUE_OUT_OF_DOMAIN`. A name already reported as a role
collision is skipped, there being no single domain to check it against.

Determinism. Every pass whose issue-append order is observable iterates a snapshot sorted
by :meth:`~archytaszx.diagram.graph.Wire.sort_key` /
:meth:`~archytaszx.diagram.graph.PortRef.sort_key`, never a frozenset directly.
:attr:`ValidationReport.issues`'s order is relied on by :mod:`archytaszx.rewrite.engine`'s
deferred-issue selection.

Out of scope: contraction and numeric meaning (Phase 4's oracle), repair, and bang boxes
(Phase 7).
"""

from __future__ import annotations

import enum
import itertools
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import DimSubstituteValue, DimSymbolKey, unify_all
from archytaszx.diagram.bangbox import BangBox
from archytaszx.diagram.generators import DimensionPolicy, PhaseSchema
from archytaszx.diagram.graph import (
    BangBoxId,
    Diagram,
    Direction,
    Node,
    NodeId,
    Port,
    PortRef,
    Wire,
)


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
    DIMENSION_BOUND = "dimension_bound"
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
    PARAMETER_UNKNOWN_SYMBOL = "parameter_unknown_symbol"
    PARAMETER_VALUE_OUT_OF_DOMAIN = "parameter_value_out_of_domain"
    BANGBOX_SCOPE_UNKNOWN_NODE = "bangbox_scope_unknown_node"
    BANGBOX_PORT_UNKNOWN = "bangbox_port_unknown"
    BANGBOX_PORT_NOT_BOUNDARY = "bangbox_port_not_boundary"
    BANGBOX_SCOPE_OVERLAP = "bangbox_scope_overlap"
    BANGBOX_UNKNOWN_PARENT = "bangbox_unknown_parent"
    BANGBOX_PARENT_CYCLE = "bangbox_parent_cycle"
    BANGBOX_NESTING_MISMATCH = "bangbox_nesting_mismatch"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One finding: a machine-readable kind, the offending reference(s), and a message.

    Exactly one of ``node_id``, ``port_ref``, or ``wire`` is typically the primary
    offender for a given ``kind``; the others are left ``None``. ``deferred`` is True
    for :attr:`IssueKind.DIMENSION_DEFERRED` and :attr:`IssueKind.DIMENSION_BOUND`,
    marking these as assumed constraints rather than hard failures -- see the module
    docstring.
    """

    kind: IssueKind
    message: str
    node_id: NodeId | None = None
    port_ref: PortRef | None = None
    wire: Wire | None = None
    bang_box_id: BangBoxId | None = None
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
    # selection archytaszx.rewrite.engine relies on.
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        port_a = _resolve(diagram, wire.a, issues)
        port_b = _resolve(diagram, wire.b, issues)
        if port_a is None or port_b is None:
            continue
        if port_a.dim == port_b.dim:
            continue
        resolved_a = diagram.resolve_dim(port_a.dim)
        resolved_b = diagram.resolve_dim(port_b.dim)
        if (resolved_a, resolved_b) != (port_a.dim, port_b.dim) and resolved_a.unify(
            resolved_b
        ).is_failure:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DIMENSION_MISMATCH,
                    message=(
                        f"wire {wire!r} joins {port_a.dim} and {port_b.dim}, which the "
                        f"parameter environment resolves to {resolved_a} and {resolved_b}"
                    ),
                    wire=wire,
                )
            )
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
        elif result.bindings:
            bound = ", ".join(
                f"{name} := {value}" for name, value in sorted(result.bindings.items())
            )
            issues.append(
                ValidationIssue(
                    kind=IssueKind.DIMENSION_BOUND,
                    message=(
                        f"wire {wire!r} assumes {port_a.dim} == {port_b.dim}, which holds "
                        f"only under the binding(s) {bound}"
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

    ``archytaszx.algebra``'s four symbol constructors each stamp a distinct assumption signature,
    matched here against sympy's computed closure. Round-tripped per constructor by
    ``tests/test_validate.py``'s ``TestSymbolConstructorRolesRoundTrip``:

    * :meth:`~archytaszx.algebra.dimension.Dim.symbol` (``positive=True, integer=True``) --
      "dimension". Signature ``integer and positive``.
    * a dimension's exponent, from :meth:`~archytaszx.algebra.dimension.Dim.__pow__` via
      ``_exponent_symbol`` (``integer=True, nonnegative=True``, never ``positive``: an
      exponent of 0 is legal, a dimension of 0 is not) -- "exponent". Signature ``integer
      and not positive``; ``nonnegative`` holds for both and does not discriminate.
    * :meth:`~archytaszx.algebra.phase.Phase.symbol` (``real=True``) -- "phase". Signature ``real
      and not integer``; a dimension or exponent symbol is ``real`` by closure.
    * :meth:`~archytaszx.algebra.scalar.Scalar.symbol` (``complex=True``) -- "scalar". Signature
      ``complex and not real``; the other three are ``complex`` by closure.
    * :meth:`~archytaszx.diagram.bangbox.Mult.symbol` (Phase 7) -- "multiplicity". Same real
      assumptions as an exponent, discriminated only by an inert ``multiplicity`` marker
      sympy's own closure never sets, so it survives into ``assumptions0`` untouched.

    Each branch tests only the keys its constructor sets, never a derived one, so a fifth
    constructor setting a different pair falls through to ``None``, unclassified rather than
    aliased into an existing role. ``None`` also covers a bare, assumption-free ``Symbol``.
    """
    assumptions = symbol.assumptions0
    is_integer = bool(assumptions.get("integer"))
    is_positive = bool(assumptions.get("positive"))
    is_real = bool(assumptions.get("real"))
    is_complex = bool(assumptions.get("complex"))
    is_multiplicity = bool(assumptions.get("multiplicity"))
    if is_integer and is_positive:
        return "dimension"
    if is_integer and is_multiplicity:
        return "multiplicity"
    if is_integer and not is_positive:
        return "exponent"
    if is_real and not is_integer:
        return "phase"
    if is_complex and not is_real:
        return "scalar"
    return None


def _symbol_roles(diagram: Diagram) -> dict[str, dict[str, sp.Symbol]]:
    """Every symbol name the diagram carries, mapped to the roles it appears in.

    The same role twice under one name records one entry, so a name in legitimate reuse
    (two ports sharing a dimension symbol, a root-of-unity entry over its own node's
    dimension symbol) never reaches ``len(by_role) > 1``. Shared by
    :func:`_check_symbol_role_collisions` and :func:`_check_parameter_environment`.
    """
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
    for box in diagram.bang_boxes.values():
        _note(box.multiplicity.to_sympy())
    return roles


_ROLE_MINIMUM: Mapping[str, int | None] = MappingProxyType(
    {"dimension": 1, "exponent": 0, "phase": None, "scalar": None, "multiplicity": 0}
)
"""The lower bound each symbol role imposes on an integer parameter value, ``None`` for a
role that bounds it nowhere."""


def _check_parameter_environment(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    # A same-named symbol of two roles is already SYMBOL_ROLE_COLLISION, and there is no one
    # domain to check the value against, so such a name is skipped here rather than reported
    # twice under two different kinds.
    roles = _symbol_roles(diagram)
    for name, value in sorted(diagram.parameters.items()):
        by_role = roles.get(name)
        if not by_role:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PARAMETER_UNKNOWN_SYMBOL,
                    message=(
                        f"parameter environment binds {name!r} := {value}, which no symbol "
                        "in this diagram carries; the substitution it records is pending "
                        "against nothing"
                    ),
                    deferred=True,
                )
            )
            continue
        if len(by_role) > 1:
            continue
        (role,) = by_role
        minimum = _ROLE_MINIMUM[role]
        if minimum is not None and value < minimum:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.PARAMETER_VALUE_OUT_OF_DOMAIN,
                    message=(
                        f"parameter environment binds {name!r} := {value}, outside the "
                        f"domain of its {role} role (>= {minimum})"
                    ),
                )
            )


def _check_symbol_role_collisions(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    # A same-named symbol of two different roles is two distinct sympy Symbol objects, so a
    # by-name substitution (every substitute() here is) silently rewrites both. All six
    # unordered cross-role pairs over {dimension, exponent, phase, scalar} are genuine
    # collisions, since the four roles accept different substitution domains: positive
    # integers, nonnegative integers, reals mod one turn, and arbitrary complex.
    for name, by_role in sorted(_symbol_roles(diagram).items()):
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
    # nowhere at all. archytaszx.semantics.denote already refuses such a node; stating the same
    # fact here, as a hard error, is what makes validate(d).is_valid imply every node in d
    # is denotable -- the invariant archytaszx.rewrite.engine's apply step 8 depends on.
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
        else:
            # A binding and a residual pair are independent findings, not alternatives: a
            # DEFERRED resolution can also have bound a symbol, and its residual pairs are
            # stated at operands that binding already resolved.
            if leg_unify.bindings or leg_unify.declined_bindings:
                bound = ", ".join(
                    f"{name} := {value}"
                    for name, value in sorted(
                        {**dict(leg_unify.bindings), **dict(leg_unify.declined_bindings)}.items()
                    )
                )
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.DIMENSION_BOUND,
                        message=(
                            f"node {node.id!r} ({gen.name}) legs "
                            f"{sorted(str(port.dim) for port in all_ports)} agree only under "
                            f"the binding(s) {bound}"
                        ),
                        node_id=node.id,
                        deferred=True,
                    )
                )
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
                elif result.bindings:
                    bound = ", ".join(
                        f"{name} := {value}" for name, value in sorted(result.bindings.items())
                    )
                    issues.append(
                        ValidationIssue(
                            kind=IssueKind.DIMENSION_BOUND,
                            message=(
                                f"node {node.id!r} ({gen.name}) assumes phase dimension "
                                f"{resolved_phase_dim} == leg dimension {resolved_leg_dim}, "
                                f"which holds only under the binding(s) {bound}"
                            ),
                            node_id=node.id,
                            deferred=True,
                        )
                    )
                # A binding this phase check produces is deliberately not fed back into
                # leg_unify/resolved_leg_dim. Unlike match.py's condition 8, this function
                # decides no applicability: the legs' question is already fully settled
                # before the phase is examined, and nothing later re-reads resolved_leg_dim.
                # Feeding the binding back could only sharpen the wording of an
                # already-emitted DIMENSION_DEFERRED residual, never change a verdict.


def _check_bangbox_scopes(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    """Every node-scope node and every port-scope port must resolve against ``diagram``
    (Phase 7).

    A port-scope box's port must additionally currently be a diagram boundary slot --
    :mod:`archytaszx.diagram.bangbox`'s instantiate/kill mechanism has no other way to grow
    or remove it (see that module's docstring). Boxes failing either check are excluded
    from :func:`_check_bangbox_nesting`, mirroring :func:`_check_port_usage`'s
    ``broken_node_ids`` skip pattern.
    """
    for box_id, box in sorted(diagram.bang_boxes.items()):
        for node_id in sorted(box.node_scope):
            if node_id not in diagram.nodes:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.BANGBOX_SCOPE_UNKNOWN_NODE,
                        message=(
                            f"bang box {box_id!r} node_scope references unknown node {node_id!r}"
                        ),
                        bang_box_id=box_id,
                    )
                )
        for ref in sorted(box.port_scope, key=lambda r: r.sort_key()):
            node = diagram.nodes.get(ref.node_id)
            if node is None or ref.index >= len(node.legs(ref.direction)):
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.BANGBOX_PORT_UNKNOWN,
                        message=(
                            f"bang box {box_id!r} port_scope references unresolvable port {ref!r}"
                        ),
                        bang_box_id=box_id,
                        port_ref=ref,
                    )
                )
                continue
            if ref not in diagram.boundary_inputs and ref not in diagram.boundary_outputs:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.BANGBOX_PORT_NOT_BOUNDARY,
                        message=(
                            f"bang box {box_id!r} port_scope port {ref!r} is not a diagram "
                            "boundary slot; instantiate/kill can only grow or remove a "
                            "scoped port that is currently on the boundary"
                        ),
                        bang_box_id=box_id,
                        port_ref=ref,
                    )
                )


def _broken_bangbox_ids(issues: list[ValidationIssue]) -> frozenset[BangBoxId]:
    return frozenset(
        issue.bang_box_id
        for issue in issues
        if issue.kind
        in (
            IssueKind.BANGBOX_SCOPE_UNKNOWN_NODE,
            IssueKind.BANGBOX_PORT_UNKNOWN,
            IssueKind.BANGBOX_PORT_NOT_BOUNDARY,
            IssueKind.BANGBOX_UNKNOWN_PARENT,
            IssueKind.BANGBOX_PARENT_CYCLE,
        )
        and issue.bang_box_id is not None
    )


def _bangbox_footprint(box: BangBox) -> frozenset[NodeId]:
    """The set of node ids a box "occupies", for overlap/nesting purposes: its own
    node_scope, plus the node each of its port_scope entries sits on."""
    return box.node_scope | frozenset(ref.node_id for ref in box.port_scope)


def _render_ports(port_scope: frozenset[PortRef]) -> list[tuple[int, str, int]]:
    """A port scope as a list of sort keys, ordered by ``PortRef.sort_key``."""
    return [ref.sort_key() for ref in sorted(port_scope, key=lambda r: r.sort_key())]


def _nesting_refusal(box: BangBox, parent: BangBox) -> str | None:
    """Why ``box`` is not strictly finer than ``parent``, or None when it is.

    Containment is judged on footprints; strictness is judged on the child's own scope,
    a port-scope child under a node-scope parent refining that parent's ports.
    """
    footprint = _bangbox_footprint(box)
    parent_footprint = _bangbox_footprint(parent)
    if not footprint <= parent_footprint:
        return (
            f"its footprint {sorted(footprint)} is not contained in the parent's "
            f"{sorted(parent_footprint)}"
        )
    if box.is_node_scope:
        if footprint == parent_footprint:
            return (
                f"its node scope {sorted(footprint)} is not a proper subset of the "
                f"parent's footprint {sorted(parent_footprint)}"
            )
        return None
    if parent.is_node_scope:
        return None
    if box.port_scope < parent.port_scope:
        return None
    return (
        f"its port scope {_render_ports(box.port_scope)} is not a proper subset of the "
        f"parent's {_render_ports(parent.port_scope)}"
    )


def _check_bangbox_nesting(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    """Parent references form a cycle-free forest, and a declared parent strictly
    contains its child (Phase 7).

    Two boxes with no declared parent/child relationship along the ``parent`` chain
    must have disjoint footprints -- an undeclared overlap can only mean two
    independent boxes were built over the same node in error, since a legitimate nested
    relationship is always recorded via ``parent`` (see
    :mod:`archytaszx.diagram.bangbox`'s module docstring).
    """
    broken = _broken_bangbox_ids(issues)
    boxes = {box_id: box for box_id, box in diagram.bang_boxes.items() if box_id not in broken}

    ancestors: dict[BangBoxId, tuple[BangBoxId, ...]] = {}
    for box_id, box in sorted(boxes.items()):
        chain: list[BangBoxId] = []
        current = box.parent
        seen = {box_id}
        cyclic = False
        for _ in range(len(boxes) + 1):
            if current is None:
                break
            if current not in boxes and current not in diagram.bang_boxes:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.BANGBOX_UNKNOWN_PARENT,
                        message=(
                            f"bang box {box_id!r} declares parent {current!r}, which does not exist"
                        ),
                        bang_box_id=box_id,
                    )
                )
                cyclic = True  # not a cycle, but stops the walk from trusting `chain`
                chain = []
                break
            if current in seen:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.BANGBOX_PARENT_CYCLE,
                        message=(
                            f"bang box {box_id!r} has a cyclic parent chain reaching "
                            f"{current!r} again"
                        ),
                        bang_box_id=box_id,
                    )
                )
                cyclic = True
                chain = []
                break
            seen.add(current)
            chain.append(current)
            current = diagram.bang_boxes[current].parent
        if not cyclic:
            ancestors[box_id] = tuple(chain)

    ok_ids = frozenset(ancestors) & frozenset(boxes)
    for box_id in sorted(ok_ids):
        box = boxes[box_id]
        if box.parent is None:
            continue
        parent_id = box.parent
        if parent_id not in boxes:
            continue  # the parent itself is broken/cyclic; already reported against it
        parent = boxes[parent_id]
        refusal = _nesting_refusal(box, parent)
        if refusal is not None:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.BANGBOX_NESTING_MISMATCH,
                    message=(f"bang box {box_id!r} declares parent {parent_id!r}, but {refusal}"),
                    bang_box_id=box_id,
                )
            )

    for box_id_1, box_id_2 in itertools.combinations(sorted(ok_ids), 2):
        box_1, box_2 = boxes[box_id_1], boxes[box_id_2]
        related = box_id_2 in ancestors.get(box_id_1, ()) or box_id_1 in ancestors.get(box_id_2, ())
        if related:
            continue
        overlap = _bangbox_footprint(box_1) & _bangbox_footprint(box_2)
        if overlap:
            issues.append(
                ValidationIssue(
                    kind=IssueKind.BANGBOX_SCOPE_OVERLAP,
                    message=(
                        f"bang boxes {box_id_1!r} and {box_id_2!r} overlap on node(s) "
                        f"{sorted(overlap)} with no declared parent/child relationship "
                        "between them"
                    ),
                    bang_box_id=box_id_1,
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
    _check_bangbox_scopes(diagram, issues)
    _check_bangbox_nesting(diagram, issues)
    _check_symbol_role_collisions(diagram, issues)
    _check_parameter_environment(diagram, issues)
    return ValidationReport(tuple(issues))


def validate_or_raise(diagram: Diagram) -> ValidationReport:
    """Run :func:`validate` and raise ValidationFailedError if any hard-failure issue is found.

    Returns the report (which may still carry deferred issues) on success.
    """
    report = validate(diagram)
    if not report.is_valid:
        raise ValidationFailedError(report)
    return report
