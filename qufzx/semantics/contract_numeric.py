"""Numeric contraction of a fully concrete diagram into a tensor, carrying the exact scalar.

This module is the only place, besides :mod:`qufzx.semantics.denote`, that is allowed to
construct a dense array: it denotes every node (via :func:`~qufzx.semantics.denote.denote`)
and contracts the results along the diagram's wires. It never rewrites, simplifies, or
reorders the diagram it is given -- it only reads and evaluates it, per ``CLAUDE.md``'s
"rewriting never contracts" rule read the other way around: contraction never rewrites.

Algorithm. Refuse first: run :func:`qufzx.diagram.validate.validate` and refuse a diagram
with any hard-failure issue (:class:`ContractValidationError`, carrying the report so the
caller can see why); a diagram carrying a *deferred* dimension issue is refused the same
way, since a deferred issue exists only because some dimension pair could not be decided,
which cannot happen once every dimension in scope is concrete (checked next) -- seeing one
here means something upstream (e.g. a caller skipping :mod:`qufzx.semantics.check`
instantiation) left a symbol in. Then refuse if any port dimension, any node phase vector,
or the diagram's own :class:`~qufzx.algebra.scalar.Scalar` is non-concrete. Both refusals
happen before any array is allocated.

Each node's axes get their own integer label; a :class:`~qufzx.diagram.graph.Wire`, which
names exactly two :class:`~qufzx.diagram.graph.PortRef`\\ s (regardless of whether either
side is an input or an output port -- the graph model permits wiring two outputs together,
and this module does not assume wires run input-to-output), unifies its two ports' labels.
A self-loop -- a wire whose two ports both belong to the same node -- unifies two labels
that already sit on the same tensor, which is exactly a partial trace; nothing special-
cases it. Free (boundary) ports keep their own distinct label. The whole contraction is
then one call to ``numpy.einsum`` in its *interleaved* (operand, axis-label-list, ...,
output-label-list) form, with axis labels as plain Python ints rather than the 52-letter
subscript-string alphabet -- chosen specifically so the diagram's leg count is never
bounded by that alphabet; there is accordingly no separate error path for "too many legs
for einsum" here; see the size guard below for the orthogonal "too many elements" limit.
Output axes are ordered ``diagram.boundary_outputs`` then ``diagram.boundary_inputs``, per
the axis convention fixed in :mod:`qufzx.semantics.denote`. :func:`~qufzx.diagram.validate.validate`
guarantees every port is wired exactly once or on exactly one boundary list, so every port
gets exactly one label and every node axis is accounted for; this module asserts that
rather than assuming it silently. The diagram's exact scalar is multiplied in as the very
last step, via ``Scalar.to_complex()`` -- the only sanctioned Scalar-to-number path -- so
no factor is ever normalized, dropped, or quotiented along the way.

An empty diagram (no nodes) has nothing to denote or contract; it evaluates directly to
the rank-0 array holding ``diagram.scalar.to_complex()``.

Size guard. The result and every intermediate is ``d ** (number of axes)`` complex
numbers, and a diagram that looks small on the page (few nodes, a symbolic-looking but
fully concrete ``d``) can still be catastrophic once contracted -- e.g. a single 20-leg
spider at ``d = 17``. ``max_elements`` (default ``10_000_000``, about 160 MB of
``complex128``, an arbitrary but explicit budget for what this "boringly correct" oracle
should attempt without a caller opting into more) is checked against both the size of
each node's own tensor, before it is denoted, and the size of the final output tensor,
before contraction runs, raising :class:`ContractSizeError` with a clear message rather
than exhausting memory silently.

Return type. :func:`contract` returns a :class:`ContractionResult` -- the tensor plus the
ordered tuple of :class:`~qufzx.diagram.graph.PortRef`\\ s that produced its axes -- rather
than a bare array, matching this codebase's preference for structured returns
(``ValidationReport``, ``UnifyResult``): a bare array cannot answer "which axis is which
boundary port", which :mod:`qufzx.semantics.check` and any future caller comparing two
contractions need in order to interpret the result at all.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import ValidationReport, validate
from qufzx.semantics.denote import denote, resolve_dimension

DEFAULT_MAX_ELEMENTS = 10_000_000
"""The default cap on the element count of any single tensor this module allocates."""


class ContractError(Exception):
    """Base class for all errors raised by this module."""


class ContractDomainError(ContractError):
    """A value is outside the mathematical domain this module requires.

    Raised for a non-concrete port dimension, node phase, or diagram scalar -- always
    before any array is allocated.
    """


class ContractGrammarError(ContractError):
    """A request is malformed in a way independent of concreteness.

    Raised if the label-assignment bookkeeping ever fails to cover every port (an
    internal consistency assertion, not expected to be reachable from a validated
    diagram).
    """


class ContractValidationError(ContractDomainError):
    """Raised when the diagram fails :func:`qufzx.diagram.validate.validate`.

    A subclass of :class:`ContractDomainError`: a diagram that validate rejects, or that
    still carries a deferred dimension constraint, is not a diagram this module's domain
    (fully concrete, well-formed graphs) accepts. Carries the offending
    :class:`~qufzx.diagram.validate.ValidationReport` as :attr:`report`.
    """

    def __init__(self, report: ValidationReport) -> None:
        """Build the error from the failing (or still-deferred) report."""
        self.report = report
        summary = "; ".join(issue.message for issue in (report.errors or report.deferred))
        super().__init__(f"diagram is not contractible: {summary}")


class ContractSizeError(ContractDomainError):
    """Raised when a tensor this module would allocate exceeds the configured size cap."""


@dataclass(frozen=True, slots=True)
class ContractionResult:
    """The result of :func:`contract`: a tensor plus the axis order that produced it.

    ``axis_refs[i]`` is the :class:`~qufzx.diagram.graph.PortRef` that ``tensor``'s axis
    ``i`` came from, in ``diagram.boundary_outputs`` then ``diagram.boundary_inputs``
    order (the axis convention fixed in :mod:`qufzx.semantics.denote`).
    """

    tensor: np.ndarray
    axis_refs: tuple[PortRef, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor's shape, for convenience."""
        return tuple(self.tensor.shape)


def _check_concrete(diagram: Diagram) -> None:
    if not diagram.scalar.is_concrete:
        raise ContractDomainError(
            f"diagram scalar {diagram.scalar} is not concrete; cannot contract numerically"
        )
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            if not port.dim.is_concrete:
                raise ContractDomainError(
                    f"node {node.id!r} has non-concrete port dimension {port.dim}; "
                    "cannot contract numerically"
                )
        if node.phase is not None and not node.phase.is_concrete:
            raise ContractDomainError(
                f"node {node.id!r} has non-concrete phase vector {node.phase}; "
                "cannot contract numerically"
            )


def _assign_labels(diagram: Diagram) -> dict[PortRef, int]:
    """Assign one integer axis label per port, unifying the two ends of every wire."""
    counter = itertools.count()
    labels: dict[PortRef, int] = {}
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        labels.setdefault(ref, next(counter))
    for wire in diagram.wires:
        a_label = labels.get(wire.a)
        b_label = labels.get(wire.b)
        if a_label is None and b_label is None:
            shared = next(counter)
            labels[wire.a] = shared
            labels[wire.b] = shared
        elif a_label is None:
            assert b_label is not None
            labels[wire.a] = b_label
        elif b_label is None:
            labels[wire.b] = a_label
        else:
            labels[wire.a] = a_label
            labels[wire.b] = a_label
    return labels


def _check_size(elements: int, *, max_elements: int, what: str) -> None:
    if elements > max_elements:
        raise ContractSizeError(
            f"{what} would have {elements} elements, exceeding the cap of {max_elements}; "
            "pass a larger max_elements to contract() if this is intentional"
        )


def contract(diagram: Diagram, *, max_elements: int = DEFAULT_MAX_ELEMENTS) -> ContractionResult:
    """Contract a fully concrete, bang-box-free diagram into one tensor.

    See the module docstring for the full algorithm, the size guard, and why the return
    type is a :class:`ContractionResult` rather than a bare array.
    """
    report = validate(diagram)
    if not report.is_valid or report.deferred:
        raise ContractValidationError(report)
    _check_concrete(diagram)

    axis_refs = (*diagram.boundary_outputs, *diagram.boundary_inputs)

    if not diagram.nodes:
        tensor = np.array(diagram.scalar.to_complex(), dtype=np.complex128)
        return ContractionResult(tensor=tensor, axis_refs=axis_refs)

    labels = _assign_labels(diagram)

    einsum_args: list[Any] = []
    for node_id, node in diagram.nodes.items():
        rank = node.num_outputs + node.num_inputs
        d = resolve_dimension(node)
        _check_size(d**rank, max_elements=max_elements, what=f"node {node_id!r}'s tensor")
        tensor = denote(node)
        axis_labels = [
            labels[PortRef(node_id, Direction.OUTPUT, i)] for i in range(node.num_outputs)
        ] + [labels[PortRef(node_id, Direction.INPUT, i)] for i in range(node.num_inputs)]
        if len(axis_labels) != tensor.ndim:
            raise ContractGrammarError(
                f"node {node_id!r} has {tensor.ndim} tensor axes but {len(axis_labels)} "
                "port labels; this indicates an internal bookkeeping inconsistency"
            )
        einsum_args.append(tensor)
        einsum_args.append(axis_labels)

    missing = [ref for ref in axis_refs if ref not in labels]
    if missing:
        raise ContractGrammarError(
            f"the following boundary ports were never assigned an axis label: {missing}"
        )
    output_labels = [labels[ref] for ref in axis_refs]

    output_elements = 1
    for ref in axis_refs:
        node = diagram.nodes[ref.node_id]
        port = node.legs(ref.direction)[ref.index]
        output_elements *= port.dim.to_int()
    _check_size(output_elements, max_elements=max_elements, what="the contracted output tensor")

    raw_tensor = np.einsum(*einsum_args, output_labels)
    tensor = np.asarray(raw_tensor, dtype=np.complex128) * diagram.scalar.to_complex()
    return ContractionResult(tensor=tensor, axis_refs=axis_refs)
