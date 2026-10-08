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

Besides :mod:`archytaszx.semantics.denote`, the only place allowed to construct a dense array:
it denotes every node and contracts the results along the diagram's wires, never rewriting,
simplifying, or reordering.

Algorithm. Refuse first: :func:`archytaszx.diagram.validate.validate` runs, and a diagram with
any hard-failure issue is refused with :class:`ContractValidationError` carrying the
report. A *deferred* dimension issue is refused the same way -- it exists only when a
dimension pair could not be decided, which cannot happen once every dimension is concrete.
Then refuse any non-concrete port dimension, phase vector, or diagram
:class:`~archytaszx.algebra.scalar.Scalar`. Both refusals precede any allocation.

Each node's axes get their own integer label, and a :class:`~archytaszx.diagram.graph.Wire`
unifies its two ports' labels, regardless of direction. A self-loop unifies two labels
already on the same tensor, which is exactly a partial trace. Free ports keep a distinct
label. The network is contracted pair by pair: extent-one axes are dropped, then the two
tensors sharing a label whose result is smallest are merged by one two-operand
``numpy.einsum`` over locally renumbered labels, so the diagram's label count is unbounded.
Output axes are ordered ``boundary_outputs`` then ``boundary_inputs``, per ``denote``'s axis
convention.
``validate`` guarantees every port is wired exactly once or on exactly one boundary list;
two consistency checks stand behind that guarantee rather than resting on it -- each node's
port-label count against its tensor's rank, and every boundary ref confirmed labelled --
each raising :class:`ContractGrammarError`. The exact scalar is multiplied in last via
``Scalar.to_complex()``, the only sanctioned Scalar-to-number path.

An empty diagram evaluates directly to the rank-0 array holding
``diagram.scalar.to_complex()``.

Size guard. A node's element count is the product of its per-leg dimensions.
``max_elements`` (default ``10_000_000``, about 160 MB of ``complex128``) is checked
against each node's own tensor before it is denoted, against the output tensor before
contraction, and against every intermediate, raising :class:`ContractSizeError`.

Return type. :func:`contract` returns a :class:`ContractionResult`: the tensor, the ordered
:class:`~archytaszx.diagram.graph.PortRef`\\ s that produced its axes, and the count of leading
axes that are boundary outputs. The split count is carried rather than recomputed from
``len(diagram.boundary_outputs)``, the diagram not necessarily being at hand by then.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.validate import ValidationReport, validate
from archytaszx.semantics.denote import denote, leg_dimensions

DEFAULT_MAX_ELEMENTS = 10_000_000
"""The default cap on the element count of any single tensor this module allocates, inputs,
intermediates and output alike."""


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
    """Raised when the diagram fails :func:`archytaszx.diagram.validate.validate`.

    A subclass of :class:`ContractDomainError`: a diagram validate rejects, or that still
    carries a deferred dimension constraint, is outside this module's domain of fully
    concrete, well-formed graphs. Carries the offending
    :class:`~archytaszx.diagram.validate.ValidationReport` as :attr:`report`.
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

    ``axis_refs[i]`` is the :class:`~archytaszx.diagram.graph.PortRef` that ``tensor``'s axis
    ``i`` came from, in ``diagram.boundary_outputs`` then ``diagram.boundary_inputs``
    order (the axis convention fixed in :mod:`archytaszx.semantics.denote`).
    ``axis_refs[:num_boundary_outputs]`` are the boundary outputs and the rest the boundary
    inputs, giving a caller the output/input arity split without a diagram on hand.
    """

    tensor: np.ndarray
    axis_refs: tuple[PortRef, ...]
    num_boundary_outputs: int

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor's shape, for convenience."""
        return tuple(self.tensor.shape)


def _check_concrete(diagram: Diagram) -> None:
    if diagram.bang_boxes:
        raise ContractDomainError(
            f"diagram still carries bang box(es) {sorted(diagram.bang_boxes)}; expand them "
            "(archytaszx.semantics.check.instantiate) before contracting numerically"
        )
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

    A union-find over ports: when a wire's two ends already carry different labels, every
    port wearing the higher-numbered (absorbed) label is rewritten to the lower-numbered
    (surviving) one, not merely ``wire.a`` and ``wire.b``. That merge path is unreachable
    from :func:`contract`, which refuses the multiply-claimed ports needed to reach it; this
    function is also callable directly on an unvalidated wire set.
    """
    counter = itertools.count()
    labels: dict[PortRef, int] = {}
    for ref in (*diagram.boundary_outputs, *diagram.boundary_inputs):
        labels.setdefault(ref, next(counter))
    # diagram.wires is a frozenset with PYTHONHASHSEED-dependent iteration order. The
    # tensor is invariant under any consistent relabeling, but sorting keeps a dump of
    # `labels` itself reproducible.
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
            # This wire merges two equivalence classes built up by earlier wires: every
            # port wearing either label ends up on the lower one, deterministically.
            survivor, absorbed = (a_label, b_label) if a_label < b_label else (b_label, a_label)
            # Collected first, then rewritten, never reassigned while iterating
            # labels.items().
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


def _squeezed(tensor: np.ndarray, axis_labels: list[int]) -> tuple[np.ndarray, list[int]]:
    """``tensor`` without its extent-one axes, and the labels of the axes kept."""
    kept = [i for i, extent in enumerate(tensor.shape) if extent != 1]
    return tensor.reshape([tensor.shape[i] for i in kept]), [axis_labels[i] for i in kept]


def _einsum_local(
    operands: list[tuple[np.ndarray, list[int]]], output_labels: list[int]
) -> np.ndarray:
    """One ``numpy.einsum`` over ``operands`` with labels renumbered from zero."""
    local: dict[int, int] = {}
    arguments: list[Any] = []
    for tensor, axis_labels in operands:
        arguments.append(tensor)
        arguments.append([local.setdefault(label, len(local)) for label in axis_labels])
    arguments.append([local.setdefault(label, len(local)) for label in output_labels])
    return np.asarray(np.einsum(*arguments, optimize=len(operands) > 1))


def _contract_network(
    operands: list[tuple[np.ndarray, list[int]]],
    output_labels: list[int],
    extents: dict[int, int],
    max_elements: int,
) -> np.ndarray:
    """Contract ``operands`` into one tensor over ``output_labels``, greedily pair by pair.

    Each step merges the two live tensors sharing a label whose merged tensor is smallest,
    keeping every label still read by another tensor or by the output; with no shared label
    left, the two smallest tensors are merged by outer product.
    """
    live: dict[int, tuple[np.ndarray, list[int]]] = dict(
        enumerate(_squeezed(tensor, axis_labels) for tensor, axis_labels in operands)
    )
    wanted = [label for label in output_labels if extents[label] != 1]
    readers: dict[int, int] = {label: 1 for label in wanted}
    for _tensor, axis_labels in live.values():
        for label in set(axis_labels):
            readers[label] = readers.get(label, 0) + 1
    next_key = len(live)

    def merged_labels(first: int, second: int) -> list[int]:
        own = set(live[first][1]) | set(live[second][1])
        local = {label: (label in live[first][1]) + (label in live[second][1]) for label in own}
        result: list[int] = []
        for label in (*live[first][1], *live[second][1]):
            if readers[label] > local[label] and label not in result:
                result.append(label)
        return result

    while len(live) > 1:
        owners: dict[int, list[int]] = {}
        for key in sorted(live):
            for label in set(live[key][1]):
                owners.setdefault(label, []).append(key)
        pairs = {
            (keys[i], keys[j])
            for keys in owners.values()
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        }
        if not pairs:
            first, second = sorted(live, key=lambda key: (live[key][0].size, key))[:2]
            pairs = {(min(first, second), max(first, second))}
        best: tuple[int, int, int, list[int]] | None = None
        for first, second in sorted(pairs):
            result = merged_labels(first, second)
            cost = math.prod(extents[label] for label in result)
            if best is None or cost < best[0]:
                best = (cost, first, second, result)
        assert best is not None
        cost, first, second, result = best
        _check_size(cost, max_elements=max_elements, what="an intermediate tensor")
        pair = [live.pop(first), live.pop(second)]
        for _tensor, axis_labels in pair:
            for label in set(axis_labels):
                readers[label] -= 1
        for label in result:
            readers[label] += 1
        live[next_key] = (_einsum_local(pair, result), result)
        next_key += 1

    tensor, axis_labels = next(iter(live.values()))
    tensor = _einsum_local([(tensor, axis_labels)], wanted)
    return tensor.reshape([extents[label] for label in output_labels])


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

    operands: list[tuple[np.ndarray, list[int]]] = []
    for node_id, node in diagram.nodes.items():
        elements = math.prod(leg_dimensions(node))
        _check_size(elements, max_elements=max_elements, what=f"node {node_id!r}'s tensor")
        tensor = denote(node)
        port_refs = [PortRef(node_id, Direction.OUTPUT, i) for i in range(node.num_outputs)] + [
            PortRef(node_id, Direction.INPUT, i) for i in range(node.num_inputs)
        ]
        unlabelled = [ref for ref in port_refs if ref not in labels]
        if unlabelled:
            raise ContractGrammarError(
                f"node {node_id!r} has port(s) {unlabelled!r} that were never assigned an "
                "axis label; validate() rejects an unused port, so a labelled-port gap here "
                "is an internal bookkeeping inconsistency"
            )
        axis_labels = [labels[ref] for ref in port_refs]
        if len(axis_labels) != tensor.ndim:
            raise ContractGrammarError(
                f"node {node_id!r} has {tensor.ndim} tensor axes but {len(axis_labels)} "
                "port labels; this indicates an internal bookkeeping inconsistency"
            )
        operands.append((tensor, axis_labels))

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

    extents: dict[int, int] = {}
    for tensor, axis_labels in operands:
        extents.update(zip(axis_labels, tensor.shape, strict=True))
    raw_tensor = _contract_network(operands, output_labels, extents, max_elements)
    tensor = np.asarray(raw_tensor, dtype=np.complex128) * diagram.scalar.to_complex()
    return ContractionResult(
        tensor=tensor, axis_refs=axis_refs, num_boundary_outputs=num_boundary_outputs
    )
