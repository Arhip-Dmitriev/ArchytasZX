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


"""Symbolic contraction of an arbitrary diagram with the dimension kept formal.

A dense array is unavailable once ``d`` is symbolic, so a tensor is represented as an
ordered list of axes plus one exact :class:`~qufzx.algebra.scalar.Scalar` entry expression
over per-axis index symbols. Wires become bound summation indices and every Kronecker delta
is written as the character sum ``delta(a, b) = d^-1 * Sum_t omega_d^{t*(a-b)}``, so the
Phase 9 simplifier in :mod:`qufzx.algebra.scalar` is what closes a contraction rather than
a separate tensor engine.

Substituting a parameter environment into the returned entry is the path that answers a
user-supplied concrete ``d`` or ``n`` at any size; this module never converts a ``Dim`` to
an ``int``, never calls ``Scalar.to_complex``, and never imports numpy.

Determinism rests on five orderings: bang boxes and nodes by their integer ids, free axes as
``boundary_outputs`` then ``boundary_inputs``, wires by ``Wire.sort_key``, and sympy
sub-terms by ``srepr``. No set is iterated unsorted and no ``sp.Dummy`` is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from qufzx.algebra.scalar import (
    DEFAULT_MAX_SIMPLIFY_STEPS,
    Scalar,
    ScalarSubstituteValue,
    ScalarSymbolKey,
)
from qufzx.diagram.bangbox import expand_concrete_boxes, scope_is_closed
from qufzx.diagram.generators import FOURIER_BOX, REGISTRY, X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import BangBoxId, Diagram, Direction, Node, NodeId, PortRef, Wire
from qufzx.diagram.validate import ValidationReport, validate
from qufzx.semantics.denote import resolve_dim


class SymbolicContractionError(Exception):
    """Base class for all errors raised by this module."""


class SymbolicContractionDomainError(SymbolicContractionError):
    """A value is outside the domain an operation requires."""


class SymbolicContractionGrammarError(SymbolicContractionError):
    """A diagram is malformed in a way validate() should already have refused."""


class SymbolicContractionValidationError(SymbolicContractionDomainError):
    """The diagram failed validation. Carries the report."""

    def __init__(self, report: ValidationReport) -> None:
        """Build the error from the report whose errors caused the refusal."""
        self.report = report
        super().__init__(
            "cannot contract a diagram that fails validation: "
            + "; ".join(str(issue) for issue in report.errors)
        )


class SymbolicContractionUnsupportedError(SymbolicContractionDomainError):
    """A diagram shape this module does not cover, such as a variable rank."""


@dataclass(frozen=True, slots=True)
class SymbolicAxis:
    """One tensor axis: the boundary port it came from, its dimension, and its index name."""

    port: PortRef
    dim: Dim
    index: str


@dataclass(frozen=True, slots=True)
class SymbolicTensor:
    """A tensor with formal extents: ordered axes and one Scalar entry over their indices."""

    axes: tuple[SymbolicAxis, ...]
    entry: Scalar

    @property
    def rank(self) -> int:
        """The number of axes."""
        return len(self.axes)

    def dims(self) -> tuple[Dim, ...]:
        """The per-axis dimensions, in axis order."""
        return tuple(axis.dim for axis in self.axes)

    def substitute(self, mapping: Mapping[str, int]) -> SymbolicTensor:
        """Push a symbol-to-integer mapping into every axis dimension and into the entry."""
        dim_mapping: dict[DimSymbolKey, DimSubstituteValue] = {
            name: value for name, value in mapping.items()
        }
        scalar_mapping: dict[ScalarSymbolKey, ScalarSubstituteValue] = {
            name: value for name, value in mapping.items()
        }
        axes = tuple(
            SymbolicAxis(axis.port, axis.dim.substitute(dim_mapping), axis.index)
            for axis in self.axes
        )
        return SymbolicTensor(axes, self.entry.substitute(scalar_mapping))

    def simplify(self, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS) -> SymbolicTensor:
        """Run the character-sum simplifier over the entry."""
        return SymbolicTensor(self.axes, self.entry.simplify(max_steps=max_steps))

    def to_dense(self, *, max_elements: int = 10_000_000) -> object:
        """Enumerate every entry into a numpy tensor. Requires every axis dimension concrete."""
        import numpy as np

        extents: list[int] = []
        for axis in self.axes:
            if not axis.dim.is_concrete:
                raise SymbolicContractionDomainError(
                    f"axis {axis.index} has non-concrete dimension {axis.dim}; "
                    "substitute the parameter environment before going dense"
                )
            extents.append(axis.dim.to_int())
        total = 1
        for extent in extents:
            total *= extent
        if total > max_elements:
            raise SymbolicContractionDomainError(
                f"dense form would hold {total} elements, over the {max_elements} cap"
            )
        symbols = [sp.Symbol(axis.index, integer=True, nonnegative=True) for axis in self.axes]
        expr = self.entry.to_sympy()
        out = np.zeros(tuple(extents), dtype=np.complex128)
        for flat in np.ndindex(*extents):
            bound = dict(zip(symbols, (sp.Integer(v) for v in flat), strict=True))
            out[flat] = Scalar(expr.xreplace(bound)).simplify().to_complex()
        return out


class _Indices:
    """A deterministic allocator of engine index symbols named _k0, _k1, ..."""

    def __init__(self) -> None:
        """Start the counter at zero."""
        self._next = 0

    def fresh(self) -> sp.Symbol:
        """Allocate the next index symbol."""
        symbol = sp.Symbol(f"_k{self._next}", integer=True, nonnegative=True)
        self._next += 1
        return symbol


def _delta(dim: Dim, a: sp.Expr, b: sp.Expr, indices: _Indices) -> sp.Expr:
    """The Kronecker delta of two index expressions, written as a character sum over dim.

    The pair is ordered by ``srepr`` before the difference is formed, so the symmetry
    ``delta(a, b) == delta(b, a)`` holds syntactically and two diagrams that agree up to
    the orientation of a delta canonicalize to the same expression.
    """
    first, second = sorted((a, b), key=sp.srepr)
    t = indices.fresh()
    d_expr = dim.to_sympy()
    return sp.Pow(d_expr, -1) * sp.Sum(
        sp.exp(2 * sp.pi * sp.I * t * (first - second) / d_expr), (t, 0, d_expr - 1)
    )


def _phase_corrections(node: Node) -> tuple[tuple[int, sp.Expr], ...]:
    """Each stored phase entry as (concrete index, e^{i*alpha_k} - 1), in ascending index order.

    A :class:`~qufzx.algebra.phase.PhaseVector` is sparse over concrete integer indices with
    ``Phase.zero()`` everywhere else, so subtracting the phaseless spider leaves a finite
    correction whose every index is a concrete integer. That is what lets a phased spider
    contract with ``d`` symbolic: nothing is ever indexed by a symbolic summation variable.
    """
    if node.phase is None:
        return ()
    corrections = []
    for index in sorted(node.phase.entries()):
        factor = Scalar.from_phase(node.phase.get(index)).to_sympy() - sp.Integer(1)
        corrections.append((index, factor))
    return tuple(corrections)


def _z_entry(
    dim: Dim, legs: list[sp.Symbol], corrections: tuple[tuple[int, sp.Expr], ...], indices: _Indices
) -> sp.Expr:
    """The Z spider's entry: the phaseless spider plus one finite term per stored phase."""
    d_expr = dim.to_sympy()
    if legs:
        entry: sp.Expr = sp.Integer(1)
        for leg in legs[1:]:
            entry = entry * _delta(dim, legs[0], leg, indices)
    else:
        entry = d_expr
    for index, factor in corrections:
        term = factor
        for leg in legs:
            term = term * _delta(dim, leg, sp.Integer(index), indices)
        entry = entry + term
    return entry


def _x_entry(
    dim: Dim,
    legs: list[sp.Symbol],
    total: sp.Expr,
    corrections: tuple[tuple[int, sp.Expr], ...],
    indices: _Indices,
) -> sp.Expr:
    """The X spider's entry: the Z spider conjugated by a Fourier box on every leg.

    Composing the two collapses every per-leg delta, leaving one character sum over the
    signed index total, plus the same finite phase correction the Z spider carries.
    """
    d_expr = dim.to_sympy()
    k = indices.fresh()
    entry: sp.Expr = sp.Sum(sp.exp(2 * sp.pi * sp.I * k * total / d_expr), (k, 0, d_expr - 1))
    for index, factor in corrections:
        entry = entry + factor * sp.exp(2 * sp.pi * sp.I * sp.Integer(index) * total / d_expr)
    return sp.Pow(d_expr, sp.Rational(-len(legs), 2)) * entry


def _node_entry(
    diagram: Diagram,
    node_id: NodeId,
    port_index: Mapping[PortRef, sp.Symbol],
    indices: _Indices,
) -> sp.Expr:
    """The symbolic denotation of one node, over the index symbols its ports carry."""
    node = diagram.nodes[node_id]
    if not REGISTRY.is_registered(node.generator_type):
        raise SymbolicContractionGrammarError(
            f"node {node_id!r} carries generator type {node.generator_type.name!r}, which is "
            "not the type registered under that name"
        )
    dim = resolve_dim(node)
    d_expr = dim.to_sympy()
    outputs = [port_index[PortRef(node_id, Direction.OUTPUT, i)] for i in range(node.num_outputs)]
    inputs = [port_index[PortRef(node_id, Direction.INPUT, i)] for i in range(node.num_inputs)]

    if node.generator_type.name == FOURIER_BOX.name:
        return sp.Pow(d_expr, sp.Rational(-1, 2)) * sp.exp(
            2 * sp.pi * sp.I * outputs[0] * inputs[0] / d_expr
        )

    corrections = _phase_corrections(node)

    if node.generator_type.name == X_SPIDER.name:
        total = sum(outputs, sp.Integer(0)) - sum(inputs, sp.Integer(0))
        return _x_entry(dim, outputs + inputs, total, corrections, indices)

    if node.generator_type.name != Z_SPIDER.name:
        raise SymbolicContractionGrammarError(
            f"node {node_id!r} has generator type {node.generator_type.name!r}, which "
            "contract_symbolic() does not know how to denote"
        )

    return _z_entry(dim, outputs + inputs, corrections, indices)


def _port_dim(diagram: Diagram, ref: PortRef) -> Dim:
    """The dimension carried by one port."""
    node = diagram.nodes[ref.node_id]
    return node.legs(ref.direction)[ref.index].dim


def _closed_scope_subdiagram(
    diagram: Diagram, node_scope: frozenset[NodeId], exclude: BangBoxId
) -> Diagram:
    """The closed sub-diagram on ``node_scope``: its nodes, its wires, no boundary, scalar one.

    ``exclude`` is the box being expanded; every other box lying inside the scope is carried
    over, so a nested count is still there for the recursive contraction to meet.
    """
    extracted = Diagram()
    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(node_scope):
        node = diagram.nodes[old_id]
        id_map[old_id] = extracted.add_node(
            node.generator_type,
            [port.dim for port in node.inputs],
            [port.dim for port in node.outputs],
            phase=node.phase,
        )
    for wire in sorted(diagram.wires, key=Wire.sort_key):
        if wire.a.node_id in node_scope and wire.b.node_id in node_scope:
            extracted.add_wire(
                PortRef(id_map[wire.a.node_id], wire.a.direction, wire.a.index),
                PortRef(id_map[wire.b.node_id], wire.b.direction, wire.b.index),
            )
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        if box_id != exclude and box.node_scope and box.node_scope <= node_scope:
            extracted.add_bang_box(
                box.multiplicity, node_scope=frozenset(id_map[n] for n in box.node_scope)
            )
    extracted.set_parameters(dict(diagram.parameters))
    return extracted


def contract_symbolic(
    diagram: Diagram, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS
) -> SymbolicTensor:
    """Contract a diagram with every dimension formal, into one closed Scalar entry."""
    report = validate(diagram)
    if report.errors:
        raise SymbolicContractionValidationError(report)

    # A concrete multiplicity is expanded outright: k copies of a scope contribute k times
    # over, and contracting the scope once would silently drop the rest.
    diagram = expand_concrete_boxes(diagram)

    # What survives carries a symbolic multiplicity. A box closed off from the rest of the
    # diagram contributes its own scalar raised to that multiplicity, which stays closed-form
    # with the count formal; one meeting a wire or a boundary slot has a rank that varies
    # with the count, which is out of scope for Phase 9.
    box_factor: sp.Expr = sp.Integer(1)
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        if not box.is_node_scope or not scope_is_closed(diagram, box.node_scope):
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id} has symbolic multiplicity {box.multiplicity} and meets "
                "the rest of the diagram at a wire or a boundary slot; symbolic contraction "
                "over a variable rank is out of scope for Phase 9"
            )
    if diagram.bang_boxes:
        working = diagram.copy()
        for box_id in sorted(diagram.bang_boxes):
            box = diagram.bang_boxes[box_id]
            if box_id not in working.bang_boxes:
                continue
            scope = box.node_scope
            inner = contract_symbolic(
                _closed_scope_subdiagram(diagram, scope, box_id), max_steps=max_steps
            )
            box_factor = box_factor * sp.Pow(inner.entry.to_sympy(), box.multiplicity.to_sympy())
            working.remove_bang_box(box_id)
            for node_id in sorted(scope):
                if node_id in working.nodes:
                    working.remove_node(node_id)
        diagram = working

    indices = _Indices()
    port_index: dict[PortRef, sp.Symbol] = {}

    free_refs = (*diagram.boundary_outputs, *diagram.boundary_inputs)
    axes: list[SymbolicAxis] = []
    for position, ref in enumerate(free_refs):
        symbol = sp.Symbol(f"_i{position}", integer=True, nonnegative=True)
        port_index[ref] = symbol
        axes.append(SymbolicAxis(ref, _port_dim(diagram, ref), str(symbol.name)))

    wire_indices: list[tuple[sp.Symbol, Dim]] = []
    for wire in sorted(diagram.wires, key=Wire.sort_key):
        first, second = wire.a, wire.b
        symbol = indices.fresh()
        port_index[first] = symbol
        port_index[second] = symbol
        wire_indices.append((symbol, _port_dim(diagram, first)))

    entry: sp.Expr = diagram.scalar.to_sympy() * box_factor
    for node_id in sorted(diagram.nodes):
        node = diagram.nodes[node_id]
        for direction in (Direction.OUTPUT, Direction.INPUT):
            for position in range(len(node.legs(direction))):
                ref = PortRef(node_id, direction, position)
                if ref not in port_index:
                    raise SymbolicContractionGrammarError(
                        f"port {ref} carries no index; validate should have refused this diagram"
                    )
        entry = entry * _node_entry(diagram, node_id, port_index, indices)

    for symbol, dim in reversed(wire_indices):
        entry = sp.Sum(entry, (symbol, 0, dim.to_sympy() - 1))

    return SymbolicTensor(tuple(axes), Scalar(entry).simplify(max_steps=max_steps))
