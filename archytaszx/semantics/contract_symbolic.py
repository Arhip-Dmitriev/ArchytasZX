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
ordered list of axes plus one exact :class:`~archytaszx.algebra.scalar.Scalar` entry expression
over per-axis index symbols. Wires become bound summation indices and every Kronecker delta
is a :class:`~archytaszx.algebra.scalar.ModDelta`, ``delta(a, b) = [a - b == 0 mod d]``, so the
simplifier in :mod:`archytaszx.algebra.scalar` is what closes a contraction rather than a
separate tensor engine.

A bang box whose count is symbolic and which meets the boundary makes the rank vary: its
copies become a :class:`ReplicatedFactor` and its boundary slots a :class:`ReplicatedBlock`.
:meth:`SymbolicTensor.substitute` of a parameter environment, then
:meth:`SymbolicTensor.value_at`, is the path that answers a user-supplied concrete ``d`` or
``n`` at any size; contraction never converts a ``Dim`` to an ``int`` and never calls
``Scalar.to_complex``.

Determinism rests on five orderings: bang boxes and nodes by their integer ids, free axes as
``boundary_outputs`` then ``boundary_inputs``, wires by ``Wire.sort_key``, and sympy
sub-terms by ``srepr``. No set is iterated unsorted and no ``sp.Dummy`` is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from archytaszx.algebra.scalar import (
    DEFAULT_MAX_SIMPLIFY_STEPS,
    ModDelta,
    Scalar,
    ScalarSubstituteValue,
    ScalarSymbolKey,
)
from archytaszx.diagram.bangbox import (
    Mult,
    crossing_wires,
    expand_concrete_boxes,
    scope_is_closed,
)
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    REGISTRY,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, Node, NodeId, PortRef, Wire
from archytaszx.diagram.validate import ValidationReport, validate
from archytaszx.semantics.denote import resolve_dim


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
class ReplicatedBlock:
    """A run of boundary slots laid out copy-major at one slot: per copy of group ``group``, one
    slot per template slot in ``axes``; a nested block names a group of that group's ``inner``."""

    group: int
    axes: tuple[LayoutSlot, ...]


@dataclass(frozen=True, slots=True)
class ReplicatedFactor:
    """``factor`` over its template indices, once per copy for ``multiplicity`` copies, each
    copy reading its own copy of every template index.

    With ``marked``, an entry term carrying the group's mark symbol ``_m<g>`` takes
    ``Sum_p marked(p) prod_{q != p} factor(q)`` in place of the product. With ``inner``, each
    copy is a copy of that self-contained variable-rank tensor and ``factor`` is unused.
    """

    multiplicity: Mult
    factor: Scalar
    marked: Scalar | None = None
    inner: SymbolicTensor | None = None


@dataclass(frozen=True, slots=True)
class SharedIndex:
    """An index summed over ``0 .. dim - 1`` around the entry and every replicated factor."""

    name: str
    dim: Dim


LayoutSlot: TypeAlias = "SymbolicAxis | ReplicatedBlock"


@dataclass(frozen=True, slots=True)
class SymbolicTensor:
    """A tensor with formal extents: ordered axes and one Scalar entry over their indices.

    A variable-rank tensor also carries ``groups``, ``shared`` and a ``layout`` interleaving
    ``axes`` with :class:`ReplicatedBlock` runs; its value is the sum over every
    :class:`SharedIndex` of ``entry`` times each group's factor at each of its copies.
    """

    axes: tuple[SymbolicAxis, ...]
    entry: Scalar
    layout: tuple[LayoutSlot, ...] | None = None
    groups: tuple[ReplicatedFactor, ...] = ()
    shared: tuple[SharedIndex, ...] = ()

    @property
    def rank(self) -> int:
        """The number of axes; for a variable-rank tensor, of its fixed axes only."""
        return len(self.axes)

    @property
    def is_variable_rank(self) -> bool:
        """True when some axes repeat once per copy of a replicated factor."""
        return bool(self.groups)

    def slots(self) -> tuple[LayoutSlot, ...]:
        """The boundary layout: :attr:`layout`, or :attr:`axes` when there is none."""
        return self.axes if self.layout is None else self.layout

    def dims(self) -> tuple[Dim, ...]:
        """The per-axis dimensions, in axis order."""
        return tuple(axis.dim for axis in self.axes)

    def signature(self) -> tuple[object, ...]:
        """Every slot's dimensions and directions, with each block's group multiplicity."""
        return tuple(self._slot_signature(slot) for slot in self.slots())

    def _slot_signature(self, slot: LayoutSlot) -> object:
        """One slot's dimension and direction, or a block's multiplicity and nested slots."""
        if isinstance(slot, SymbolicAxis):
            return (slot.dim, slot.port.direction)
        group = self.groups[slot.group]
        owner = self if group.inner is None else group.inner
        return (group.multiplicity, tuple(owner._slot_signature(inner) for inner in slot.axes))

    def multiplicities_concrete(self) -> bool:
        """True when every multiplicity, nested ones included, is concrete."""
        return all(
            group.multiplicity.is_concrete
            and (group.inner is None or group.inner.multiplicities_concrete())
            for group in self.groups
        )

    def substitute(self, mapping: Mapping[str, int | sp.Rational]) -> SymbolicTensor:
        """Push a symbol-to-value mapping into every dimension, multiplicity, entry and factor;
        only integer values reach dimensions and multiplicities. Once every multiplicity is
        concrete the result is :meth:`flatten`-ed."""
        result = self._substituted(mapping)
        if result.groups and result.multiplicities_concrete():
            return result.flatten()
        return result

    def _substituted(self, mapping: Mapping[str, int | sp.Rational]) -> SymbolicTensor:
        """:meth:`substitute` without the final flatten, recursing into nested groups."""
        dim_mapping: dict[DimSymbolKey, DimSubstituteValue] = {
            name: value
            for name, value in mapping.items()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 1
        }
        mult_mapping = {
            name: value
            for name, value in mapping.items()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }
        scalar_mapping: dict[ScalarSymbolKey, ScalarSubstituteValue] = {
            name: value for name, value in mapping.items()
        }

        def slot_at(slot: LayoutSlot) -> LayoutSlot:
            if isinstance(slot, SymbolicAxis):
                return SymbolicAxis(slot.port, slot.dim.substitute(dim_mapping), slot.index)
            return ReplicatedBlock(slot.group, tuple(slot_at(a) for a in slot.axes))

        return SymbolicTensor(
            tuple(cast(SymbolicAxis, slot_at(axis)) for axis in self.axes),
            self.entry.substitute(scalar_mapping),
            None if self.layout is None else tuple(slot_at(slot) for slot in self.layout),
            tuple(
                ReplicatedFactor(
                    group.multiplicity.substitute(
                        {
                            k: v
                            for k, v in mult_mapping.items()
                            if k in group.multiplicity.free_symbols
                        }
                    ),
                    group.factor.substitute(scalar_mapping),
                    None if group.marked is None else group.marked.substitute(scalar_mapping),
                    None if group.inner is None else group.inner._substituted(mapping),
                )
                for group in self.groups
            ),
            tuple(
                SharedIndex(index.name, index.dim.substitute(dim_mapping)) for index in self.shared
            ),
        )

    def flatten(self, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS) -> SymbolicTensor:
        """The fixed-rank tensor of a variable-rank one whose every multiplicity is concrete:
        axes ``_i0, _i1, ...`` in layout order, every copy of every factor multiplied in, every
        shared index summed."""
        if not self.groups:
            return self
        counts = []
        for group in self.groups:
            if not group.multiplicity.is_concrete:
                raise SymbolicContractionDomainError(
                    f"multiplicity {group.multiplicity} is symbolic; substitute it first"
                )
            counts.append(group.multiplicity.to_int())
        flat_inner = [
            None if group.inner is None else group.inner.flatten(max_steps=max_steps)
            for group in self.groups
        ]
        axes: list[SymbolicAxis] = []
        fixed_names: dict[str, sp.Symbol] = {}
        copy_names: list[list[dict[str, sp.Symbol]]] = [[{} for _ in range(n)] for n in counts]

        def fresh(axis: SymbolicAxis) -> sp.Symbol:
            symbol = _engine_index(f"_i{len(axes)}")
            axes.append(SymbolicAxis(axis.port, axis.dim, str(symbol.name)))
            return symbol

        for slot in self.slots():
            if isinstance(slot, SymbolicAxis):
                fixed_names[slot.index] = fresh(slot)
                continue
            inner = flat_inner[slot.group]
            for copy in range(counts[slot.group]):
                if inner is None:
                    for axis in cast(tuple[SymbolicAxis, ...], slot.axes):
                        copy_names[slot.group][copy][axis.index] = fresh(axis)
                    continue
                direction = _block_direction(slot)
                for axis in inner.axes:
                    if axis.port.direction is direction:
                        copy_names[slot.group][copy][axis.index] = fresh(axis)
        expr = _rename(self.entry.to_sympy(), fixed_names)
        for g, (group, copies) in enumerate(zip(self.groups, copy_names, strict=True)):
            inner = flat_inner[g]
            factor = group.factor.to_sympy() if inner is None else inner.entry.to_sympy()
            per_copy = [_rename(factor, {**fixed_names, **names}) for names in copies]
            product = sp.Mul(*per_copy)
            mark = _mark_symbol(g)
            if group.marked is None or mark not in expr.free_symbols:
                expr = expr * product
                continue
            marked = group.marked.to_sympy()
            chosen = sp.Add(
                *(
                    _rename(marked, {**fixed_names, **copies[p]})
                    * sp.Mul(*(per_copy[:p] + per_copy[p + 1 :]))
                    for p in range(len(copies))
                )
            )
            expanded = sp.expand(expr)
            expr = (
                expanded.xreplace({mark: sp.Integer(0)}) * product
                + sp.diff(expanded, mark) * chosen
            )
        for index in reversed(self.shared):
            expr = sp.Sum(expr, (_engine_index(index.name), 0, index.dim.to_sympy() - 1))
        return SymbolicTensor(tuple(axes), Scalar(expr).simplify(max_steps=max_steps))

    def simplify(self, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS) -> SymbolicTensor:
        """Run the character-sum simplifier over the entry and every replicated factor."""
        return SymbolicTensor(
            self.axes,
            self.entry.simplify(max_steps=max_steps, ranges=self.ranges()),
            self.layout,
            tuple(
                ReplicatedFactor(
                    group.multiplicity,
                    group.factor.simplify(max_steps=max_steps),
                    None if group.marked is None else group.marked.simplify(max_steps=max_steps),
                    None if group.inner is None else group.inner.simplify(max_steps=max_steps),
                )
                for group in self.groups
            ),
            self.shared,
        )

    def ranges(self) -> dict[str, Dim]:
        """The dimension each fixed axis index and shared index runs over."""
        sized = {axis.index: axis.dim for axis in self.axes}
        sized.update({index.name: index.dim for index in self.shared})
        return sized

    def _require_fixed_rank(self) -> None:
        """Raise unless this tensor has a fixed rank."""
        if self.groups:
            raise SymbolicContractionDomainError(
                "the tensor's rank varies with a multiplicity; substitute every multiplicity first"
            )

    def value_at(
        self, index: Sequence[int], *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS
    ) -> Scalar:
        """The entry at one boundary index tuple, in axis order, simplified."""
        self._require_fixed_rank()
        if len(index) != len(self.axes):
            raise SymbolicContractionGrammarError(
                f"index tuple has {len(index)} value(s) for {len(self.axes)} axis/axes"
            )
        for axis, value in zip(self.axes, index, strict=True):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SymbolicContractionGrammarError(
                    f"axis {axis.index} needs a non-negative int, got {value!r}"
                )
            if axis.dim.is_concrete and value >= axis.dim.to_int():
                raise SymbolicContractionDomainError(
                    f"axis {axis.index} has dimension {axis.dim}, so {value} is out of range"
                )
        values = {axis.index: v for axis, v in zip(self.axes, index, strict=True)}
        return _entry_at(self.entry, values).simplify(max_steps=max_steps)

    def to_dense(self, *, max_elements: int = 10_000_000) -> object:
        """Enumerate every entry into a numpy tensor. Requires every axis dimension concrete."""
        import numpy as np

        self._require_fixed_rank()
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


def _block_direction(block: ReplicatedBlock) -> Direction:
    """The direction of a block's first axis, nested blocks included."""
    first = block.axes[0]
    return first.port.direction if isinstance(first, SymbolicAxis) else _block_direction(first)


def _mark_symbol(group: int) -> sp.Symbol:
    """The formal symbol marking an entry term that takes group ``group``'s marked copy."""
    return _engine_index(f"_m{group}")


def _engine_index(name: str) -> sp.Symbol:
    """The non-negative integer index symbol named ``name``."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def _rename(expr: sp.Expr, names: Mapping[str, sp.Expr]) -> sp.Expr:
    """``expr`` with every free symbol named in ``names`` replaced, simultaneously."""
    return cast(
        sp.Expr,
        expr.xreplace(
            {
                symbol: names[str(symbol.name)]
                for symbol in expr.free_symbols
                if str(symbol.name) in names
            }
        ),
    )


def _entry_at(entry: Scalar, values: Mapping[str, int]) -> Scalar:
    """``entry`` with each named free index symbol replaced by its integer value."""
    expr = entry.to_sympy()
    bound = {
        symbol: sp.Integer(values[str(symbol.name)])
        for symbol in expr.free_symbols
        if str(symbol.name) in values
    }
    return Scalar(expr.xreplace(bound))


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


def _delta(dim: Dim, a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """The Kronecker delta of two index expressions, as ``[a - b == 0 mod dim]``."""
    return cast(sp.Expr, ModDelta(a - b, dim.to_sympy()))


def _phase_corrections(node: Node) -> tuple[tuple[int, sp.Expr], ...]:
    """Each stored phase entry as (concrete index, e^{i*alpha_k} - 1), in ascending index order.

    A :class:`~archytaszx.algebra.phase.PhaseVector` is sparse over concrete integer indices with
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
    dim: Dim, legs: list[sp.Symbol], corrections: tuple[tuple[int, sp.Expr], ...]
) -> sp.Expr:
    """The Z spider's entry: the phaseless spider plus one finite term per stored phase."""
    d_expr = dim.to_sympy()
    if legs:
        entry: sp.Expr = sp.Integer(1)
        for leg in legs[1:]:
            entry = entry * _delta(dim, legs[0], leg)
    else:
        entry = d_expr
    for index, factor in corrections:
        term = factor
        for leg in legs:
            term = term * _delta(dim, leg, sp.Integer(index))
        entry = entry + term
    return entry


def _x_entry(
    dim: Dim,
    legs: list[sp.Symbol],
    total: sp.Expr,
    corrections: tuple[tuple[int, sp.Expr], ...],
) -> sp.Expr:
    """The X spider's entry: the Z spider conjugated by a Fourier box on every leg.

    Composing the two collapses every per-leg delta, leaving one delta on the signed index
    total, plus the same finite phase correction the Z spider carries.
    """
    d_expr = dim.to_sympy()
    entry: sp.Expr = d_expr * ModDelta(total, d_expr)
    for index, factor in corrections:
        entry = entry + factor * sp.exp(2 * sp.pi * sp.I * sp.Integer(index) * total / d_expr)
    return sp.Pow(d_expr, sp.Rational(-len(legs), 2)) * entry


def _connective_entry(
    node: Node,
    node_id: NodeId,
    port_index: Mapping[PortRef, sp.Symbol],
) -> sp.Expr:
    """B's or S's entry: one delta pairing the joint index against ``a * t + b``."""
    dims = _node_leg_dims(node)
    if len(dims) != 3:
        raise SymbolicContractionGrammarError(
            f"node {node_id!r} ({node.generator_type.name}) has {len(dims)} leg(s); a "
            "dimension connective takes exactly three"
        )
    if node.generator_type.name == DIM_BINDER.name:
        _, s_dim, t_dim = dims
        _check_connective_product(node, node_id, s_dim, t_dim, dims[0])
        out = port_index[PortRef(node_id, Direction.OUTPUT, 0)]
        in0 = port_index[PortRef(node_id, Direction.INPUT, 0)]
        in1 = port_index[PortRef(node_id, Direction.INPUT, 1)]
        return _delta(s_dim * t_dim, out, in0 * t_dim.to_sympy() + in1)
    s_dim, t_dim, _ = dims
    _check_connective_product(node, node_id, s_dim, t_dim, dims[2])
    out0 = port_index[PortRef(node_id, Direction.OUTPUT, 0)]
    out1 = port_index[PortRef(node_id, Direction.OUTPUT, 1)]
    inp = port_index[PortRef(node_id, Direction.INPUT, 0)]
    return _delta(s_dim * t_dim, inp, out0 * t_dim.to_sympy() + out1)


def _check_connective_product(
    node: Node, node_id: NodeId, s_dim: Dim, t_dim: Dim, joint: Dim
) -> None:
    """Raise when ``joint`` and ``s_dim * t_dim`` are known not to be equal."""
    if (s_dim * t_dim).unify(joint).is_failure:
        raise SymbolicContractionDomainError(
            f"node {node_id!r} ({node.generator_type.name}) pairs {s_dim} and {t_dim} "
            f"against joint dimension {joint}, which is not their product"
        )


def _node_leg_dims(node: Node) -> tuple[Dim, ...]:
    """Every leg dimension of ``node``, outputs then inputs, with no agreement check."""
    return tuple(port.dim for port in (*node.outputs, *node.inputs))


def _node_entry(
    diagram: Diagram,
    node_id: NodeId,
    port_index: Mapping[PortRef, sp.Symbol],
) -> sp.Expr:
    """The symbolic denotation of one node, over the index symbols its ports carry."""
    node = diagram.nodes[node_id]
    if not REGISTRY.is_registered(node.generator_type):
        raise SymbolicContractionGrammarError(
            f"node {node_id!r} carries generator type {node.generator_type.name!r}, which is "
            "not the type registered under that name"
        )
    name = node.generator_type.name
    if name in (DIM_BINDER.name, DIM_SPLITTER.name):
        return _connective_entry(node, node_id, port_index)
    dim = resolve_dim(node)
    d_expr = dim.to_sympy()
    outputs = [port_index[PortRef(node_id, Direction.OUTPUT, i)] for i in range(node.num_outputs)]
    inputs = [port_index[PortRef(node_id, Direction.INPUT, i)] for i in range(node.num_inputs)]

    if node.generator_type.name == FOURIER_BOX.name:
        return sp.Pow(d_expr, sp.Rational(-1, 2)) * sp.exp(
            2 * sp.pi * sp.I * outputs[0] * inputs[0] / d_expr
        )

    if name == TRIANGLE.name:
        return _delta(dim, outputs[0], inputs[0]) + _delta(dim, outputs[0], sp.Integer(0)) * (
            1 - _delta(dim, inputs[0], sp.Integer(0))
        )

    if name == TRIANGLE_INVERSE.name:
        return _delta(dim, outputs[0], inputs[0]) - _delta(dim, outputs[0], sp.Integer(0)) * (
            1 - _delta(dim, inputs[0], sp.Integer(0))
        )

    if name == W_NODE.name:
        zero = sp.Integer(0)
        vacuum = [_delta(dim, leg, zero) for leg in outputs]
        entry = -(len(outputs) - 1) * sp.Mul(*vacuum) * _delta(dim, inputs[0], zero)
        for position, leg in enumerate(outputs):
            rest = vacuum[:position] + vacuum[position + 1 :]
            entry = entry + _delta(dim, leg, inputs[0]) * sp.Mul(*rest)
        return entry

    corrections = _phase_corrections(node)

    if node.generator_type.name == X_SPIDER.name:
        total = sum(outputs, sp.Integer(0)) - sum(inputs, sp.Integer(0))
        return _x_entry(dim, outputs + inputs, total, corrections)

    if node.generator_type.name != Z_SPIDER.name:
        raise SymbolicContractionGrammarError(
            f"node {node_id!r} has generator type {node.generator_type.name!r}, which "
            "contract_symbolic() does not know how to denote"
        )

    return _z_entry(dim, outputs + inputs, corrections)


def _port_dim(diagram: Diagram, ref: PortRef) -> Dim:
    """The dimension carried by one port."""
    node = diagram.nodes[ref.node_id]
    return node.legs(ref.direction)[ref.index].dim


def _shared_spider_entry(
    node: Node,
    node_id: NodeId,
    port_index: Mapping[PortRef, sp.Symbol],
    boxed: Mapping[PortRef, sp.Symbol],
    shared: sp.Symbol,
) -> tuple[sp.Expr, dict[PortRef, sp.Expr]]:
    """A Z or X spider some of whose legs are replicated, written over the spider's value
    ``shared``: the unreplicated part, and each replicated leg's per-copy factor over its
    template index in ``boxed``.

    Z: ``w(s) prod [leg = s]``; X: ``d^(-L/2) w(s) omega^(s * signed leg total)``; ``w(s)`` is
    one plus each stored phase's correction at ``s``.
    """
    dim = resolve_dim(node)
    d_expr = dim.to_sympy()
    weight: sp.Expr = sp.Integer(1)
    for index, factor in _phase_corrections(node):
        weight = weight + factor * _delta(dim, shared, sp.Integer(index))
    legs = [
        (PortRef(node_id, direction, position), sign)
        for direction, sign in ((Direction.OUTPUT, 1), (Direction.INPUT, -1))
        for position in range(len(node.legs(direction)))
    ]
    factors: dict[PortRef, sp.Expr] = {}
    if node.generator_type.name == Z_SPIDER.name:
        entry = weight
        for ref, _sign in legs:
            if ref in boxed:
                factors[ref] = _delta(dim, boxed[ref], shared)
            else:
                entry = entry * _delta(dim, port_index[ref], shared)
        return entry, factors
    total: sp.Expr = sp.Integer(0)
    fixed = 0
    for ref, sign in legs:
        if ref in boxed:
            factors[ref] = sp.Pow(d_expr, sp.Rational(-1, 2)) * sp.exp(
                2 * sp.pi * sp.I * sign * shared * boxed[ref] / d_expr
            )
        else:
            total = total + sign * port_index[ref]
            fixed += 1
    entry = (
        sp.Pow(d_expr, sp.Rational(-fixed, 2))
        * weight
        * sp.exp(2 * sp.pi * sp.I * shared * total / d_expr)
    )
    return entry, factors


def _shared_w_entry(
    node: Node,
    node_id: NodeId,
    port_index: Mapping[PortRef, sp.Symbol],
    boxed: Mapping[PortRef, sp.Symbol],
    shared: sp.Symbol,
    marks: Sequence[sp.Symbol],
) -> tuple[sp.Expr, dict[PortRef, tuple[sp.Expr, sp.Expr]]]:
    """A W node some of whose outputs are replicated, written over its input value ``shared``:
    the unreplicated part, and each replicated leg's (vacuum, excited) per-copy factors.

    The part is ``[in = s] (W_F(s) + [s != 0] prod_F [f = 0] sum marks)``, ``W_F`` the W
    entry over the fixed outputs ``F``.
    """
    dim = resolve_dim(node)
    zero = sp.Integer(0)
    fixed = [
        port_index[PortRef(node_id, Direction.OUTPUT, position)]
        for position in range(len(node.outputs))
        if PortRef(node_id, Direction.OUTPUT, position) not in boxed
    ]
    vacuum = [_delta(dim, leg, zero) for leg in fixed]
    w_fixed = -(len(fixed) - 1) * sp.Mul(*vacuum) * _delta(dim, shared, zero)
    for position, leg in enumerate(fixed):
        w_fixed = w_fixed + _delta(dim, leg, shared) * sp.Mul(
            *(vacuum[:position] + vacuum[position + 1 :])
        )
    excited = (1 - _delta(dim, shared, zero)) * sp.Mul(*vacuum) * sp.Add(*marks)
    entry = _delta(dim, port_index[PortRef(node_id, Direction.INPUT, 0)], shared) * (
        w_fixed + excited
    )
    factors = {
        ref: (_delta(dim, symbol, zero), _delta(dim, symbol, shared))
        for ref, symbol in boxed.items()
        if ref.node_id == node_id
    }
    return entry, factors


def _pins(term: sp.Expr, shared: sp.Symbol, modulus: sp.Expr) -> list[sp.Expr]:
    """Every value ``e``, free of ``shared``, that a delta factor of ``term`` pins ``shared`` to
    with a unit coefficient."""
    found = []
    for factor in sp.Mul.make_args(term):
        if not isinstance(factor, ModDelta) or factor.args[1] != modulus:
            continue
        argument = sp.expand(factor.args[0])
        if sp.degree(argument, shared) != 1:
            continue
        coefficient = argument.coeff(shared)
        rest = sp.expand(argument - coefficient * shared)
        if coefficient in (1, -1) and shared not in rest.free_symbols:
            found.append(sp.expand(-rest * coefficient))
    return found


def _pinned_value(entry: Scalar, index: SharedIndex, max_steps: int) -> sp.Expr | None:
    """The value every term of ``entry`` pins ``index`` to, or None when there is none."""
    symbol = _engine_index(index.name)
    modulus = index.dim.to_sympy()
    terms = sp.Add.make_args(entry.to_sympy())
    candidates: list[sp.Expr] = []
    for term in terms:
        for pin in _pins(term, symbol, modulus):
            if pin not in candidates:
                candidates.append(pin)
    for value in candidates:
        pinned = cast(sp.Expr, ModDelta(symbol - value, modulus))
        if all(
            Scalar(term - pinned * term.xreplace({symbol: value}))
            .simplify(max_steps=max_steps)
            .is_zero
            for term in terms
        ):
            return value
    return None


def _scope_subdiagram(
    diagram: Diagram,
    node_scope: frozenset[NodeId],
    exclude: BangBoxId,
    outputs: Sequence[PortRef] = (),
    inputs: Sequence[PortRef] = (),
) -> Diagram:
    """The sub-diagram on ``node_scope``: its nodes, its wires, ``outputs`` and ``inputs`` as its
    boundary, scalar one.

    ``exclude`` is the box being expanded; its descendants are carried over, its children
    becoming top-level boxes.
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
    box_map: dict[BangBoxId, BangBoxId | None] = {exclude: None}
    pending = True
    while pending:
        pending = False
        for box_id in sorted(diagram.bang_boxes):
            box = diagram.bang_boxes[box_id]
            if box_id in box_map or box.parent not in box_map:
                continue
            box_map[box_id] = extracted.add_bang_box(
                box.multiplicity,
                node_scope=frozenset(id_map[n] for n in box.node_scope),
                port_scope=frozenset(
                    PortRef(id_map[r.node_id], r.direction, r.index) for r in box.port_scope
                ),
                parent=box_map[box.parent],
            )
            pending = True
    extracted.set_boundary_outputs(
        [PortRef(id_map[r.node_id], r.direction, r.index) for r in outputs]
    )
    extracted.set_boundary_inputs(
        [PortRef(id_map[r.node_id], r.direction, r.index) for r in inputs]
    )
    extracted.set_parameters(dict(diagram.parameters))
    return extracted


@dataclass(frozen=True, slots=True)
class _BoxPlan:
    """Every top-level symbolic box of a diagram, sorted by how it contracts."""

    closed: tuple[BangBoxId, ...]
    replicated: tuple[BangBoxId, ...]
    ported: tuple[BangBoxId, ...]


def _plan_boxes(diagram: Diagram) -> _BoxPlan:
    """Sort every top-level box: a node scope closed off, a node scope meeting only the
    boundary, or a port scope on Z and X spiders and at most one W node; raise on any other
    box."""
    closed: list[BangBoxId] = []
    replicated: list[BangBoxId] = []
    ported: list[BangBoxId] = []
    boxed_nodes = {
        node_id
        for box in diagram.bang_boxes.values()
        if box.is_node_scope
        for node_id in box.node_scope
    }
    for box_id in sorted(diagram.bang_boxes):
        box = diagram.bang_boxes[box_id]
        if box.parent is not None:
            parent = diagram.bang_boxes.get(box.parent)
            if parent is not None and parent.is_node_scope:
                continue
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id} is nested in port-scope bang box {box.parent}"
            )
        if box.is_node_scope:
            if scope_is_closed(diagram, box.node_scope):
                closed.append(box_id)
            elif not crossing_wires(diagram, box.node_scope):
                replicated.append(box_id)
            else:
                raise SymbolicContractionUnsupportedError(
                    f"bang box {box_id} has symbolic multiplicity {box.multiplicity} and meets "
                    "the rest of the diagram at a wire"
                )
            continue
        w_nodes = {
            ref.node_id
            for ref in box.port_scope
            if diagram.nodes[ref.node_id].generator_type.name == W_NODE.name
        }
        if len(w_nodes) > 1:
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id} replicates legs of {len(w_nodes)} W nodes"
            )
        for ref in sorted(box.port_scope, key=lambda r: r.sort_key()):
            node = diagram.nodes[ref.node_id]
            if node.generator_type.name not in (Z_SPIDER.name, X_SPIDER.name, W_NODE.name):
                raise SymbolicContractionUnsupportedError(
                    f"bang box {box_id} replicates a leg of node {ref.node_id!r} "
                    f"({node.generator_type.name}), which is not a Z or X spider or a W node"
                )
            if ref.node_id in boxed_nodes:
                raise SymbolicContractionUnsupportedError(
                    f"bang box {box_id} replicates a leg of node {ref.node_id!r}, which also "
                    "lies in a node-scope bang box"
                )
        ported.append(box_id)
    return _BoxPlan(tuple(closed), tuple(replicated), tuple(ported))


def contract_symbolic(
    diagram: Diagram, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS
) -> SymbolicTensor:
    """Contract a diagram with every dimension and multiplicity formal.

    A symbolic box closed off from the rest contributes its scope's scalar to the power of
    its multiplicity. A node-scope box meeting only the boundary, and a port-scope box on Z
    and X spiders, become :class:`ReplicatedFactor` groups of a variable-rank tensor; a
    spider with a replicated leg is summed over a :class:`SharedIndex`, eliminated when the
    entry pins it.
    """
    report = validate(diagram)
    if report.errors:
        raise SymbolicContractionValidationError(report)

    diagram = expand_concrete_boxes(diagram)
    plan = _plan_boxes(diagram)

    box_factor: sp.Expr = sp.Integer(1)
    working = diagram.copy()
    for box_id in plan.closed:
        box = diagram.bang_boxes[box_id]
        inner = contract_symbolic(
            _scope_subdiagram(diagram, box.node_scope, box_id), max_steps=max_steps
        )
        box_factor = box_factor * sp.Pow(inner.entry.to_sympy(), box.multiplicity.to_sympy())
        _drop_scope(working, box.node_scope)

    scope_of: dict[NodeId, BangBoxId] = {
        node_id: box_id
        for box_id in plan.replicated
        for node_id in diagram.bang_boxes[box_id].node_scope
    }
    port_box: dict[PortRef, BangBoxId] = {
        ref: box_id for box_id in plan.ported for ref in diagram.bang_boxes[box_id].port_scope
    }

    group_of: dict[BangBoxId, int] = {}
    group_boxes: list[BangBoxId] = []
    templates: list[list[SymbolicAxis]] = []
    layout: list[LayoutSlot] = []
    fixed_axes: list[SymbolicAxis] = []
    port_index: dict[PortRef, sp.Symbol] = {}
    boxed: dict[PortRef, sp.Symbol] = {}

    def group(box_id: BangBoxId) -> int:
        if box_id not in group_of:
            group_of[box_id] = len(group_boxes)
            group_boxes.append(box_id)
            templates.append([])
        return group_of[box_id]

    def template(g: int, ref: PortRef) -> SymbolicAxis:
        axis = SymbolicAxis(ref, _port_dim(diagram, ref), f"_t{g}_{len(templates[g])}")
        templates[g].append(axis)
        return axis

    for refs in (diagram.boundary_outputs, diagram.boundary_inputs):
        emitted: set[BangBoxId] = set()
        for ref in refs:
            owner = scope_of.get(ref.node_id)
            if owner is not None:
                if owner not in emitted:
                    emitted.add(owner)
                    g = group(owner)
                    run = [r for r in refs if scope_of.get(r.node_id) == owner]
                    layout.append(ReplicatedBlock(g, tuple(template(g, r) for r in run)))
                continue
            if ref in port_box:
                g = group(port_box[ref])
                axis = template(g, ref)
                boxed[ref] = _engine_index(axis.index)
                layout.append(ReplicatedBlock(g, (axis,)))
                continue
            symbol = _engine_index(f"_i{len(fixed_axes)}")
            port_index[ref] = symbol
            axis = SymbolicAxis(ref, _port_dim(diagram, ref), str(symbol.name))
            fixed_axes.append(axis)
            layout.append(axis)

    factors: list[sp.Expr] = [sp.Integer(1) for _ in group_boxes]
    inner_of: dict[int, SymbolicTensor] = {}
    for box_id in plan.replicated:
        box = diagram.bang_boxes[box_id]
        g = group(box_id)
        if len(factors) <= g:
            factors.append(sp.Integer(1))
        run_out = [a.port for a in templates[g] if a.port.direction is Direction.OUTPUT]
        run_in = [a.port for a in templates[g] if a.port.direction is Direction.INPUT]
        inner = contract_symbolic(
            _scope_subdiagram(diagram, box.node_scope, box_id, run_out, run_in),
            max_steps=max_steps,
        )
        if inner.is_variable_rank:
            inner_of[g] = inner
            _drop_scope(working, box.node_scope)
            continue
        names = {axis.index: _engine_index(t.index) for axis, t in zip(inner.axes, templates[g])}
        factors[g] = _rename(inner.entry.to_sympy(), names)
        _drop_scope(working, box.node_scope)
    for box_id in plan.ported:
        working.remove_bang_box(box_id)

    marked: list[sp.Expr | None] = [None for _ in group_boxes]
    w_legs: dict[int, list[tuple[sp.Expr, sp.Expr]]] = {}
    shared: list[SharedIndex] = []
    shared_of: dict[NodeId, sp.Symbol] = {}
    for ref in boxed:
        if ref.node_id not in shared_of:
            symbol = _engine_index(f"_s{len(shared)}")
            shared_of[ref.node_id] = symbol
            shared.append(SharedIndex(str(symbol.name), resolve_dim(working.nodes[ref.node_id])))

    indices = _Indices()
    wire_indices: list[tuple[sp.Symbol, Dim]] = []
    for wire in sorted(working.wires, key=Wire.sort_key):
        symbol = indices.fresh()
        port_index[wire.a] = symbol
        port_index[wire.b] = symbol
        wire_indices.append((symbol, _port_dim(working, wire.a)))

    entry: sp.Expr = working.scalar.to_sympy() * box_factor
    for node_id in sorted(working.nodes):
        node = working.nodes[node_id]
        for direction in (Direction.OUTPUT, Direction.INPUT):
            for position in range(len(node.legs(direction))):
                ref = PortRef(node_id, direction, position)
                if ref not in port_index and ref not in boxed:
                    raise SymbolicContractionGrammarError(
                        f"port {ref} carries no index; validate should have refused this diagram"
                    )
        if node_id in shared_of and node.generator_type.name == W_NODE.name:
            touched = sorted({group_of[port_box[ref]] for ref in boxed if ref.node_id == node_id})
            part, legs = _shared_w_entry(
                node,
                node_id,
                port_index,
                boxed,
                shared_of[node_id],
                [_mark_symbol(g) for g in touched],
            )
            entry = entry * part
            for ref, pair in legs.items():
                w_legs.setdefault(group_of[port_box[ref]], []).append(pair)
            continue
        if node_id in shared_of:
            part, per_leg = _shared_spider_entry(
                node, node_id, port_index, boxed, shared_of[node_id]
            )
            entry = entry * part
            for ref, factor in per_leg.items():
                g = group_of[port_box[ref]]
                factors[g] = factors[g] * factor
            continue
        entry = entry * _node_entry(working, node_id, port_index)

    for g, pairs in sorted(w_legs.items()):
        vacuum = [zero for zero, _excited in pairs]
        excited = sp.Add(
            *(
                pair[1] * sp.Mul(*(vacuum[:position] + vacuum[position + 1 :]))
                for position, pair in enumerate(pairs)
            )
        )
        marked[g] = factors[g] * excited
        factors[g] = factors[g] * sp.Mul(*vacuum)

    for symbol, dim in reversed(wire_indices):
        entry = sp.Sum(entry, (symbol, 0, dim.to_sympy() - 1))
    closed = Scalar(entry).simplify(max_steps=max_steps)
    if not group_boxes:
        return SymbolicTensor(tuple(fixed_axes), closed)

    kept: list[SharedIndex] = []
    for index in shared:
        value = _pinned_value(closed, index, max_steps)
        if value is None:
            kept.append(index)
            continue
        pin = {_engine_index(index.name): value}
        closed = Scalar(closed.to_sympy().xreplace(pin)).simplify(max_steps=max_steps)
        factors = [factor.xreplace(pin) for factor in factors]
        marked = [None if m is None else m.xreplace(pin) for m in marked]

    def nested(slot: LayoutSlot) -> LayoutSlot:
        if isinstance(slot, SymbolicAxis) or inner_of.get(slot.group) is None:
            return slot
        direction = _block_direction(slot)
        return ReplicatedBlock(
            slot.group,
            tuple(
                inner
                for inner in inner_of[slot.group].slots()
                if (
                    inner.port.direction
                    if isinstance(inner, SymbolicAxis)
                    else _block_direction(inner)
                )
                is direction
            ),
        )

    return SymbolicTensor(
        tuple(fixed_axes),
        closed,
        tuple(nested(slot) for slot in layout),
        tuple(
            ReplicatedFactor(
                diagram.bang_boxes[box_id].multiplicity,
                Scalar(factor).simplify(max_steps=max_steps),
                None if marked[g] is None else Scalar(marked[g]).simplify(max_steps=max_steps),
                inner_of.get(g),
            )
            for g, (box_id, factor) in enumerate(zip(group_boxes, factors, strict=True))
        ),
        tuple(kept),
    )


def _drop_scope(working: Diagram, scope: frozenset[NodeId]) -> None:
    """Remove from ``working`` every node of ``scope`` and every box lying wholly inside it."""
    for other_id, other in sorted(working.bang_boxes.items()):
        footprint = other.node_scope | {ref.node_id for ref in other.port_scope}
        if footprint <= scope:
            working.remove_bang_box(other_id)
    for node_id in sorted(scope):
        if node_id in working.nodes:
            working.remove_node(node_id)
