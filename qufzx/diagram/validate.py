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

Phase 5 audit round 18: process-independent issue ordering. The class of defect this round
closed here: this module's issue-producing passes (:func:`_check_wire_dimensions`,
:func:`_check_port_usage`) used to iterate ``diagram.wires``, a frozenset, directly. Any
value whose hash is not a pure function of its own content -- concretely,
:class:`~qufzx.diagram.graph.Direction`, an ``enum.Enum`` hashed by member name, and
therefore ``PYTHONHASHSEED``-dependent, reached transitively through
:class:`~qufzx.diagram.graph.PortRef` and :class:`~qufzx.diagram.graph.Wire`'s own hashes --
makes a ``set``/``frozenset``'s iteration order vary by process, not merely by content. Every
pass here whose issue-append order is observable (an exception raised partway through, or
the order of :attr:`ValidationReport.issues` itself, which downstream consumers such as
:mod:`qufzx.rewrite.engine`'s ``RewriteStep.removed_deferred_issues``/
``introduced_deferred_issues`` selection explicitly rely on being deterministic) now iterates
a snapshot sorted by :meth:`~qufzx.diagram.graph.Wire.sort_key` /
:meth:`~qufzx.diagram.graph.PortRef.sort_key` -- a plain tuple key with no ``Enum`` and no
seed-dependent hash anywhere in its own comparison path -- instead. A future pass added to
this module should default to the same discipline for any new set/frozenset it introduces,
not merely for the two fixed here: the question to ask is not "does this look like it needs
sorting" but "could this collection's construction ever route a value's own hash into
something this function returns, raises, or appends to a list", since that is the actual
condition that makes ordering observable rather than incidental.

Phase 5 post-closing audit round 20: a validator whose "valid" was weaker than the
denotation it is meant to gate. A node with zero input legs, zero output legs, and no phase
vector carries its dimension nowhere at all (per ``CLAUDE.md``, "dimension is stored per
port, not as one global parameter") -- :mod:`qufzx.semantics.denote` already refused such a
node (``DenoteGrammarError``, "has no legs and no phase vector; its dimension cannot be
determined"), but this module accepted it as valid, resting :mod:`qufzx.rewrite.engine`'s
``apply`` step 8 -- which uses this module as its sole structural postcondition on a
rewritten diagram -- on one builder's (``rules_library``'s) good behaviour never producing
such a node, rather than on a check. Closed by :class:`IssueKind.NODE_DIMENSION_UNDETERMINED`,
a hard error (not deferred) for exactly this shape, worded to name the same fact ``denote``
does. The invariant this establishes, and that ``tests/test_phase5_exhaustive_oracle.py``'s
exhaustive sweep now checks directly rather than merely assumes: ``validate(d).is_valid``
implies every node in ``d`` is denotable (at whatever concrete substitution makes every
dimension in scope concrete -- ``validate`` itself still accepts a well-formed *symbolic*
diagram, which is not yet denotable for an unrelated, expected reason; the invariant is about
the shape gap this round closed, not about concreteness, which was never this module's job to
enforce). The general question for a future check added to this module: does "valid" here
actually imply everything a downstream consumer (a rewrite's postcondition, a certificate
replay, an oracle contraction) assumes it does, or only the subset this module happened to
check first?

Phase 5 post-closing audit round 21 (same class as round 20's, reopened). Two more gaps
in "valid implies denotable"/"valid implies well-formed" closed:

* ``DimensionPolicy.ALL_LEGS_EQUAL`` used to unify each leg against ``all_ports[0].dim``
  independently and discard every binding, so a jointly-unsatisfiable leg set (e.g. one leg
  binding a symbol to ``2``, another to ``3``) passed as valid, and the verdict depended on
  leg order. Closed by :func:`~qufzx.algebra.dimension.unify_all`, which resolves the whole
  leg set to one shared value via a monotone bindings fixpoint (mirroring
  :mod:`qufzx.rewrite.match`'s own, but not sharing its implementation -- the two are
  pinned to agree on the question they share, a lone connecting pair with no surviving
  legs, by ``tests/test_unify_all.py::TestAgreesWithResolveFusionMatch``); FAILURE is now a
  hard :class:`IssueKind.DIMENSION_POLICY_VIOLATION`, and every residual ``DEFERRED`` pair
  gets its own :class:`IssueKind.DIMENSION_DEFERRED` rather than one collapsed
  "strongest" issue for the whole node. What this still does not do: propagate a binding
  from one node's legs to a *different* node's -- a ``d``-vs-``2`` wire on one node and a
  ``d``-vs-``3`` wire on another is jointly unsatisfiable across the diagram but each
  node's own check binds ``d`` independently and reports nothing, since diagram-global
  dimension-constraint propagation is FULL_PLAN.md Phase 10 item (i)'s job, not this
  module's local, per-node one; pinned by
  ``tests/test_unify_all.py::TestCrossNodePropagationDeferredToPhase10``.
* A name used as both a dimension symbol and a phase parameter in one diagram is a
  different defect from a dimension disagreement: substitution in this codebase is keyed
  by name, so :meth:`~qufzx.algebra.phase.PhaseVector.substitute` cannot tell such a
  collision apart from the ordinary, legal case of a phase entry legitimately citing its
  own container dimension's symbol (e.g. a root-of-unity entry over a symbolic dim).
  :class:`IssueKind.SYMBOL_ROLE_COLLISION` (see :func:`_check_symbol_role_collisions`)
  makes the ambiguous diagram itself invalid, using the distinguishing sympy assumptions
  each of :mod:`qufzx.algebra.dimension`/:mod:`qufzx.algebra.phase`/
  :mod:`qufzx.algebra.scalar`'s symbol constructors already stamps on its own symbols, as
  the structural half of closing this; see :func:`~qufzx.rewrite.match.reattach_phase` for
  the certificate half (which bindings a rewrite actually substituted into a phase's
  entries, recorded rather than left implicit).
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim, unify_all
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
    # Phase 5 post-closing audit round 18, Defect 1: ``diagram.wires`` is a frozenset, and
    # PortRef's (and therefore Wire's) hash folds in Direction's member-name hash, which is
    # PYTHONHASHSEED-dependent -- so iterating it directly here would append issues in a
    # process-dependent order. ``validate()``'s own module docstring, and
    # ``qufzx.rewrite.engine``'s ``RewriteStep.deferred_issue_identity_ambiguous``, both
    # promise a deterministic "first in validate order" selection downstream; sorting by
    # ``Wire.sort_key()`` (hash-independent) is what actually makes that promise true across
    # processes, not merely within one. See ``tests/test_engine.py::TestCrossProcessDeterminism``.
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
                        f"wire {wire!r} joins mismatched dimensions {port_a.dim} and "
                        f"{port_b.dim}"
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
    # Same fix as _check_wire_dimensions, and for the same reason: ``diagram.wires`` is a
    # frozenset whose iteration order is PYTHONHASHSEED-dependent (Direction's member-name
    # hash), so building ``wired_refs`` from an unsorted pass would make the *insertion*
    # order into ``wired_counts`` below (a ``Counter``, itself a plain dict -- iteration
    # order is insertion order, not hash order, once built) process-dependent, and with it
    # the order PORT_WIRED_TWICE issues are appended in.
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
    # boundary_inputs/boundary_outputs are ordered tuples (Diagram.boundary_inputs/
    # boundary_outputs), never a set/frozenset, and Counter/dict iteration order is
    # insertion order, not hash order -- so this loop's order is already deterministic and
    # process-independent without a sort, unlike the wire-derived loop above.
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


def _classify_symbol_role(symbol: sp.Symbol) -> str | None:
    """Which namespace ``symbol`` belongs to (dimension/phase/scalar), from its assumptions.

    ``None`` for a symbol matching none of ``qufzx.algebra``'s three constructors.
    """
    assumptions = symbol.assumptions0
    if assumptions.get("integer") and assumptions.get("positive"):
        return "dimension"
    if "real" in assumptions and assumptions.get("real"):
        return "phase"
    if assumptions.get("complex"):
        return "scalar"
    return None


def _check_symbol_role_collisions(diagram: Diagram, issues: list[ValidationIssue]) -> None:
    # A same-named dimension and phase symbol are distinct sympy Symbol objects, so a
    # by-name substitution (every substitute() here is) silently rewrites both.
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

    # Round 20, Task 9: per CLAUDE.md, "dimension is stored per port, not as one global
    # parameter" -- a node with zero legs and no phase vector carries its dimension nowhere
    # at all, so it is not well-formed, yet this module accepted it as valid before this
    # fix. :mod:`qufzx.semantics.denote` already refused such a node
    # (``DenoteGrammarError``, "has no legs and no phase vector; its dimension cannot be
    # determined") -- this check states the identical fact at validation time, as a hard
    # error rather than a deferred one, so that ``validate(d).is_valid`` actually implies
    # every node in ``d`` is denotable, which is the invariant :mod:`qufzx.rewrite.engine`'s
    # ``apply`` step 8 depends on when it uses this module as its sole structural
    # postcondition on a rewritten diagram. See ``tests/test_phase5_exhaustive_oracle.py``
    # and ``tests/test_fusion_properties.py`` for the property sweep asserting this
    # cross-module invariant holds over every diagram either generator produces, not merely
    # over the one hand-built case that motivated the fix.
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

    all_ports = (*node.inputs, *node.outputs)
    shared_dim: Dim | None = all_ports[0].dim if all_ports else None

    if gen.dimension_policy is DimensionPolicy.ALL_LEGS_EQUAL and shared_dim is not None:
        all_result = unify_all([port.dim for port in all_ports])
        if all_result.is_failure:
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
        elif all_result.is_deferred:
            for assumed, equal_to in all_result.residual_pairs:
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
                        f"node {node.id!r} ({gen.name}) carries a phase but its type is "
                        "phase-free"
                    ),
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
