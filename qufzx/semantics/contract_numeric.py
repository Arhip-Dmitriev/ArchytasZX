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

"""Numeric contraction of a fully concrete diagram into a tensor, carrying the exact scalar.

Besides :mod:`qufzx.semantics.denote`, the only place allowed to construct a dense array:
it denotes every node and contracts the results along the diagram's wires. It never
rewrites, simplifies, or reorders the diagram -- the spec's "rewriting never contracts"
read the other way around.

Algorithm. Refuse first: :func:`qufzx.diagram.validate.validate` runs, and a diagram with
any hard-failure issue is refused with :class:`ContractValidationError` carrying the
report. A *deferred* dimension issue is refused the same way -- it exists only because a
dimension pair could not be decided, which cannot happen once every dimension is concrete,
so seeing one means something upstream left a symbol in. Then refuse any non-concrete port
dimension, phase vector, or diagram :class:`~qufzx.algebra.scalar.Scalar`. Both refusals
precede any allocation.

Each node's axes get their own integer label, and a :class:`~qufzx.diagram.graph.Wire`
unifies its two ports' labels -- regardless of direction, since the graph model permits
wiring two outputs together. A self-loop unifies two labels already on the same tensor,
which is exactly a partial trace; nothing special-cases it. Free ports keep a distinct
label. The contraction is one ``numpy.einsum`` call in interleaved form with plain ``int``
labels rather than the 52-letter subscript alphabet, so leg count is never bounded by that
alphabet and there is no "too many legs" error path. Output axes are ordered
``boundary_outputs`` then ``boundary_inputs``, per ``denote``'s axis convention.
``validate`` guarantees every port is wired exactly once or on exactly one boundary list,
so every node axis is accounted for. Two consistency checks stand behind that guarantee
rather than resting on it: each node's port-label count is compared against its tensor's
rank, and every boundary ref is confirmed to have been labelled, each raising
:class:`ContractGrammarError`. The exact scalar is multiplied in last via
``Scalar.to_complex()``, the only sanctioned Scalar-to-number path, so no factor is ever
normalized away.

An empty diagram evaluates directly to the rank-0 array holding
``diagram.scalar.to_complex()``.

Size guard. The result and every intermediate is ``d ** (number of axes)`` complex numbers,
so a diagram that looks small on the page can still be catastrophic -- a single 20-leg
spider at ``d = 17``. ``max_elements`` (default ``10_000_000``, about 160 MB of
``complex128``: an arbitrary but explicit budget) is checked against each node's own tensor
before it is denoted and against the output tensor before contraction, raising
:class:`ContractSizeError` rather than exhausting memory silently.

Return type. :func:`contract` returns a :class:`ContractionResult` -- the tensor, the
ordered :class:`~qufzx.diagram.graph.PortRef`\\ s that produced its axes, and the count of
leading axes that are boundary outputs -- since a bare array cannot answer "which axis is
which port" or "where does the output/input split fall", which any caller comparing two
contractions needs. The split count is carried rather than recomputed from
``len(diagram.boundary_outputs)``, since the diagram may no longer be at hand by then.
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
    ``num_boundary_outputs`` records where that split falls: ``axis_refs[:num_boundary_outputs]``
    are the boundary outputs and ``axis_refs[num_boundary_outputs:]`` are the boundary
    inputs, so a caller can recover the output/input arity split without re-deriving it
    from a diagram it may not still have on hand.
    """

    tensor: np.ndarray
    axis_refs: tuple[PortRef, ...]
    num_boundary_outputs: int

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
    """Assign one integer axis label per port, unifying the two ends of every wire.

    A proper union-find over ports, not a per-wire two-endpoint patch: when a wire's two
    ends already carry different labels, every port wearing the higher-numbered (absorbed)
    label is rewritten to the lower-numbered (surviving) one, not merely ``wire.a`` and
    ``wire.b``. A third port sharing the absorbed label would otherwise keep a stale one,
    splitting an equivalence class in two and producing a wrong contraction silently.

    That merge path is unreachable from :func:`contract`, which refuses the multiply-claimed
    ports needed to reach it, but this function is also callable directly on an unvalidated
    wire set, so it is correct on its own terms.
    """
    counter = itertools.count()
    labels: dict[PortRef, int] = {}
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        labels.setdefault(ref, next(counter))
    # diagram.wires is a frozenset with PYTHONHASHSEED-dependent iteration order. That
    # order cannot affect the returned tensor -- the integers are dummy labels, and
    # contraction is invariant under any consistent relabeling of them -- but it is sorted
    # anyway so a dump of `labels` itself is reproducible.
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
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
        elif a_label != b_label:
            # This wire merges two equivalence classes built up independently by earlier
            # wires, so every port wearing either label must end up on the same one. The
            # lower integer survives, keeping the result deterministic.
            survivor, absorbed = (a_label, b_label) if a_label < b_label else (b_label, a_label)
            # Collected first, then rewritten, rather than reassigned while iterating
            # labels.items(): mutating values mid-iteration happens to work in CPython, but
            # it is one added key away from a "dictionary changed size" crash.
            absorbed_ports = [port for port, label in labels.items() if label == absorbed]
            for port in absorbed_ports:
                labels[port] = survivor
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
    num_boundary_outputs = len(diagram.boundary_outputs)

    if not diagram.nodes:
        tensor = np.array(diagram.scalar.to_complex(), dtype=np.complex128)
        return ContractionResult(
            tensor=tensor, axis_refs=axis_refs, num_boundary_outputs=num_boundary_outputs
        )

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
    return ContractionResult(
        tensor=tensor, axis_refs=axis_refs, num_boundary_outputs=num_boundary_outputs
    )
