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

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey, format_count
from archytaszx.algebra.phase import INDEX, PhaseVector
from archytaszx.algebra.scalar import (
    DEFAULT_MAX_SIMPLIFY_STEPS,
    ModDelta,
    Scalar,
    ScalarBudgetError,
    ScalarSubstituteValue,
    ScalarSymbolKey,
)
from archytaszx.diagram.bangbox import (
    BangBox,
    Mult,
    boundary_refs_in_scope,
    crossing_wires,
    expand_concrete_boxes,
    outside_crossings,
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
    copy is a copy of that self-contained variable-rank tensor and ``factor`` is unused. With
    ``index``, copy ``m`` (``m = 0 .. multiplicity - 1``) reads that symbol as ``m``. With
    ``pairs``, every pair of copies ``j < k`` also contributes ``pairs`` with copy ``j``'s
    template indices as ``_t<g>_<i>``, copy ``k``'s as ``_u<g>_<i>``, ``_j<g> = j`` and
    ``_k<g> = k``.
    """

    multiplicity: Mult
    factor: Scalar
    marked: Scalar | None = None
    inner: SymbolicTensor | None = None
    index: str | None = None
    pairs: Scalar | None = None


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
                    group.index,
                    None if group.pairs is None else group.pairs.substitute(scalar_mapping),
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
            at = _copy_numbers(group.index, len(copies))
            per_copy = [
                _rename(factor, {**fixed_names, **names, **at[c]}) for c, names in enumerate(copies)
            ]
            product = sp.Mul(*per_copy, *_pair_terms(group, g, copies, fixed_names))
            mark = _mark_symbol(g)
            if group.marked is None or mark not in expr.free_symbols:
                expr = expr * product
                continue
            marked = group.marked.to_sympy()
            chosen = sp.Add(
                *(
                    _rename(marked, {**fixed_names, **copies[p], **at[p]})
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
                    group.index,
                    None if group.pairs is None else group.pairs.simplify(max_steps=max_steps),
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
                f"dense form would hold {format_count(total)} elements, over the "
                f"{format_count(max_elements)} cap"
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


def _linear_table(node: Node, dim: Dim) -> int | None:
    """The integer ``a`` when ``node``'s stored phases at a concrete dimension ``D`` are exactly
    ``a*k/D`` turns mod 1 at every index ``k`` (``a`` nonzero mod ``D``, entry 1 giving ``a``),
    else None."""
    phase = node.phase
    if phase is None or not dim.is_concrete:
        return None
    size = dim.to_int()
    entries = phase.entries()
    if size < 3 or len(entries) != size - 1:
        return None
    first = entries.get(1)
    if first is None or not sp.sympify(first.to_sympy_turns()).is_Rational:
        return None
    slope = sp.Rational(first.to_sympy_turns()) * size
    if not slope.is_Integer or int(slope) % size == 0:
        return None
    a = int(slope) % size
    for k in range(1, size):
        stored = entries.get(k)
        if stored is None or not sp.sympify(stored.to_sympy_turns()).is_Rational:
            return None
        if not (sp.Rational(stored.to_sympy_turns()) - sp.Rational(a * k, size)).is_Integer:
            return None
    return a


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


def _partner_port(diagram: Diagram, ref: PortRef) -> PortRef | None:
    """The port wired to ``ref``, or None."""
    return next((w.b if w.a == ref else w.a for w in diagram.wires if ref in (w.a, w.b)), None)


def _bilinear_table(
    diagram: Diagram, node_id: NodeId
) -> tuple[NodeId, int, int, tuple[sp.Rational, sp.Rational, sp.Rational]] | None:
    """For a Z(1 -> 1) phase table fed by a binder B(d1, d2): the binder, d1, d2 and (a, b, c)
    with every stored entry at index x*d2 + y equal to ``a x y + b x + c y`` mod 1 turns, a,
    b, c read from the entries at (1, 1), (1, 0), (0, 1); None otherwise."""
    from fractions import Fraction

    node = diagram.nodes[node_id]
    phase = node.phase
    if (
        node.generator_type.name != Z_SPIDER.name
        or node.num_inputs != 1
        or node.num_outputs != 1
        or phase is None
        or phase.rule is not None
        or phase.is_weighted
    ):
        return None
    dim = resolve_dim(node)
    if not dim.is_concrete:
        return None
    entries = phase.entries()
    if not entries:
        return None
    source = _partner_port(diagram, PortRef(node_id, Direction.INPUT, 0))
    if source is None or source.direction is not Direction.OUTPUT:
        return None
    binder = diagram.nodes[source.node_id]
    if binder.generator_type.name != DIM_BINDER.name or len(binder.inputs) != 2:
        return None
    d1_dim = diagram.resolve_dim(binder.inputs[0].dim)
    d2_dim = diagram.resolve_dim(binder.inputs[1].dim)
    if not (d1_dim.is_concrete and d2_dim.is_concrete):
        return None
    d1, d2 = d1_dim.to_int(), d2_dim.to_int()
    if d1 * d2 != dim.to_int():
        return None
    turns: dict[int, Fraction] = {}
    for index, value in entries.items():
        t = value.to_sympy_turns()
        if not t.is_Rational:
            return None
        turns[index] = Fraction(int(t.p), int(t.q))

    def at(j: int) -> Fraction:
        return turns.get(j, Fraction(0))

    b = at(d2) if d1 > 1 else Fraction(0)
    c = at(1) if d2 > 1 else Fraction(0)
    a = at(d2 + 1) - b - c if d1 > 1 and d2 > 1 else Fraction(0)
    for x in range(d1):
        for y in range(d2):
            if (a * x * y + b * x + c * y - at(x * d2 + y)) % 1:
                return None

    def rational(f: Fraction) -> sp.Rational:
        return sp.Rational(f.numerator, f.denominator)

    return source.node_id, d1, d2, (rational(a), rational(b), rational(c))


def _binder_digits(
    binder: NodeId, port_index: Mapping[PortRef, sp.Symbol]
) -> tuple[sp.Symbol, sp.Symbol] | None:
    """The index symbols on a binder's two inputs, or None when either has none."""
    xs = port_index.get(PortRef(binder, Direction.INPUT, 0))
    ys = port_index.get(PortRef(binder, Direction.INPUT, 1))
    return None if xs is None or ys is None else (xs, ys)


def _bilinear_table_entry(
    diagram: Diagram, node_id: NodeId, port_index: Mapping[PortRef, sp.Symbol]
) -> sp.Expr | None:
    """A phase table :func:`_bilinear_table` recognises, as ``[out = in] exp(2 pi i P(x, y))``
    over its binder's inputs (x, y); None otherwise."""
    found = _bilinear_table(diagram, node_id)
    if found is None:
        return None
    binder, _d1, _d2, (a, b, c) = found
    digits = _binder_digits(binder, port_index)
    if digits is None:
        return None
    xs, ys = digits
    out = port_index[PortRef(node_id, Direction.OUTPUT, 0)]
    inp = port_index[PortRef(node_id, Direction.INPUT, 0)]
    dim = resolve_dim(diagram.nodes[node_id])
    return cast(
        sp.Expr, _delta(dim, out, inp) * sp.exp(2 * sp.pi * sp.I * (a * xs * ys + b * xs + c * ys))
    )


def _table_splitter_entry(
    diagram: Diagram, node_id: NodeId, port_index: Mapping[PortRef, sp.Symbol]
) -> sp.Expr | None:
    """A splitter S(d1, d2) fed by a recognised phase table fed by B(d1, d2), as
    ``[out0 = x][out1 = y]`` over the binder's inputs (x, y); None otherwise."""
    node = diagram.nodes[node_id]
    if node.generator_type.name != DIM_SPLITTER.name or len(node.outputs) != 2:
        return None
    feed = _partner_port(diagram, PortRef(node_id, Direction.INPUT, 0))
    if feed is None or feed.direction is not Direction.OUTPUT:
        return None
    found = _bilinear_table(diagram, feed.node_id)
    if found is None:
        return None
    binder, d1, d2, _coefficients = found
    s1 = diagram.resolve_dim(node.outputs[0].dim)
    s2 = diagram.resolve_dim(node.outputs[1].dim)
    if not (s1.is_concrete and s2.is_concrete) or (s1.to_int(), s2.to_int()) != (d1, d2):
        return None
    digits = _binder_digits(binder, port_index)
    if digits is None:
        return None
    out0 = port_index[PortRef(node_id, Direction.OUTPUT, 0)]
    out1 = port_index[PortRef(node_id, Direction.OUTPUT, 1)]
    return cast(sp.Expr, _delta(s1, out0, digits[0]) * _delta(s2, out1, digits[1]))


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
        split = _table_splitter_entry(diagram, node_id, port_index)
        return split if split is not None else _connective_entry(node, node_id, port_index)
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

    if node.phase is not None and (node.phase.rule is not None or node.phase.is_weighted):
        return _ruled_entry(node, node_id, dim, outputs, inputs)

    bilinear = _bilinear_table_entry(diagram, node_id, port_index)
    if bilinear is not None:
        return bilinear

    slope = _linear_table(node, dim)
    legs = outputs + inputs
    if slope is not None and node.generator_type.name == X_SPIDER.name:
        total = sum(outputs, sp.Integer(0)) - sum(inputs, sp.Integer(0))
        return sp.Pow(d_expr, sp.Rational(-len(legs), 2)) * d_expr * ModDelta(total + slope, d_expr)
    if slope is not None and node.generator_type.name == Z_SPIDER.name and legs:
        entry = sp.exp(2 * sp.pi * sp.I * slope * legs[0] / d_expr)
        for leg in legs[1:]:
            entry = entry * _delta(dim, legs[0], leg)
        return cast(sp.Expr, entry)

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


def _rule_weight(node: Node, value: sp.Expr) -> sp.Expr:
    """The coefficient ``w * e^{i*alpha}`` of ``node``'s phase vector at a possibly symbolic
    ``value``."""
    phase = node.phase
    assert phase is not None
    if not phase.is_weighted:
        return cast(sp.Expr, sp.exp(2 * sp.pi * sp.I * phase.at(value)))
    dim = phase.dim
    rule = phase.weight_rule

    def base(index: sp.Expr) -> sp.Expr:
        return sp.Integer(1) if rule is None else rule.to_sympy().xreplace({INDEX: index})

    weight = base(value) + _delta(dim, value, sp.Integer(0)) * (1 - base(sp.Integer(0)))
    for index, entry in sorted(phase.weights().items()):
        weight = weight + _delta(dim, value, sp.Integer(index)) * (
            entry.to_sympy() - base(sp.Integer(index))
        )
    factor: sp.Expr = sp.Integer(1)
    if phase.rule is not None:
        factor = sp.exp(2 * sp.pi * sp.I * phase.rule.to_sympy_turns().xreplace({INDEX: value}))
    for index, turns in sorted(phase.entries().items()):
        factor = factor * (
            1
            + _delta(dim, value, sp.Integer(index))
            * (Scalar.from_phase(turns).to_sympy() - sp.Integer(1))
        )
    return cast(sp.Expr, weight * factor)


def _ruled_entry(
    node: Node, node_id: NodeId, dim: Dim, outputs: list[sp.Symbol], inputs: list[sp.Symbol]
) -> sp.Expr:
    """A Z or X spider entry whose phase vector carries a rule, summed over the spider value."""
    d_expr = dim.to_sympy()
    legs = outputs + inputs
    if node.generator_type.name == X_SPIDER.name:
        s = sp.Symbol("_phase_s", integer=True, nonnegative=True)
        total = sum(outputs, sp.Integer(0)) - sum(inputs, sp.Integer(0))
        body = _rule_weight(node, s) * sp.exp(2 * sp.pi * sp.I * s * total / d_expr)
        return sp.Pow(d_expr, sp.Rational(-len(legs), 2)) * sp.Sum(body, (s, 0, d_expr - 1))
    if node.generator_type.name != Z_SPIDER.name:
        raise SymbolicContractionGrammarError(
            f"node {node_id!r} has generator type {node.generator_type.name!r}, which "
            "contract_symbolic() does not know how to denote"
        )
    if not legs:
        k = sp.Symbol("_phase_s", integer=True, nonnegative=True)
        return cast(sp.Expr, sp.Sum(_rule_weight(node, k), (k, 0, d_expr - 1)))
    entry = _rule_weight(node, legs[0])
    for leg in legs[1:]:
        entry = entry * _delta(dim, legs[0], leg)
    return entry


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
    ruled = node.phase is not None and (node.phase.rule is not None or node.phase.is_weighted)
    if ruled:
        weight = _rule_weight(node, shared)
    for index, factor in _phase_corrections(node) if not ruled else ():
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


def _periodic(expr: sp.Expr, symbol: sp.Symbol, modulus: sp.Expr) -> bool:
    """Whether ``expr`` is unchanged by ``symbol -> symbol + modulus``, read off its structure:
    deltas mod ``modulus`` and ``exp(2 pi i P / modulus)`` over integer polynomials ``P``."""
    if symbol not in expr.free_symbols:
        return True
    if isinstance(expr, (sp.Add, sp.Mul)):
        return all(_periodic(arg, symbol, modulus) for arg in expr.args)
    if isinstance(expr, sp.Pow):
        return symbol not in expr.exp.free_symbols and _periodic(expr.base, symbol, modulus)
    if isinstance(expr, ModDelta):
        argument, m = expr.args
        return bool(m == modulus) and _integer_poly(sp.expand(argument), symbol)
    if isinstance(expr, sp.exp):
        return _integer_poly(sp.expand(expr.args[0] * modulus / (2 * sp.pi * sp.I)), symbol)
    return False


def _integer_poly(expr: sp.Expr, symbol: sp.Symbol) -> bool:
    """Whether ``expr`` is a polynomial in ``symbol`` with integer-valued coefficients."""
    try:
        poly = sp.Poly(expr, symbol)
    except sp.PolynomialError:
        return False
    return all(c.is_integer for c in poly.all_coeffs())


def _pinned_by_delta(term: sp.Expr, symbol: sp.Symbol, value: sp.Expr, modulus: sp.Expr) -> bool:
    """Whether ``term`` is a delta pinning ``symbol`` to ``value`` times a part periodic in
    ``symbol``."""
    for factor in sp.Mul.make_args(term):
        if value in _pins(factor, symbol, modulus):
            rest = sp.Mul(*(f for f in sp.Mul.make_args(term) if f is not factor))
            return _periodic(rest, symbol, modulus)
    return False


def _pinned_value(
    entry: Scalar, index: SharedIndex, max_steps: int, others: Sequence[sp.Expr] = ()
) -> sp.Expr | None:
    """The value every term of ``entry`` pins ``index`` to, or None when there is none; a term
    passes by simplification, or as a delta times a periodic part when ``others`` are periodic.
    A value other than 0 or a bare index needs ``others`` periodic."""
    symbol = _engine_index(index.name)
    modulus = index.dim.to_sympy()
    terms = sp.Add.make_args(entry.to_sympy())
    candidates: list[sp.Expr] = []
    for term in terms:
        for pin in _pins(term, symbol, modulus):
            if pin not in candidates:
                candidates.append(pin)
    periodic_others = all(_periodic(e, symbol, modulus) for e in others)
    for value in candidates:
        if not (periodic_others or value == 0 or value.is_Symbol):
            continue
        pinned = cast(sp.Expr, ModDelta(symbol - value, modulus))
        structural = periodic_others and bool(value.is_integer)
        if all(
            (structural and _pinned_by_delta(term, symbol, value, modulus))
            or Scalar(term - pinned * term.xreplace({symbol: value}))
            .simplify(max_steps=max_steps)
            .is_zero
            for term in terms
        ):
            return value
    return None


def _split_pinned_terms(
    closed: Scalar,
    shared: Sequence[SharedIndex],
    factors: Sequence[sp.Expr],
    multiplicities: Sequence[sp.Expr],
    max_steps: int,
) -> tuple[Scalar, list[sp.Expr]] | None:
    """The entry and group factors with every shared index summed out, when each term of the
    entry pins every shared index and each group factor at every term's pins is a constant
    multiple of its value at the first term's pins: entry ``Sum_k e_k prod_g lambda_kg^m_g``,
    factors the first term's; None otherwise."""
    terms = sp.Add.make_args(closed.to_sympy())
    pinned: list[tuple[sp.Expr, dict[sp.Symbol, sp.Expr]]] = []
    for term in terms:
        pins: dict[sp.Symbol, sp.Expr] = {}
        for index in shared:
            symbol = _engine_index(index.name)
            modulus = index.dim.to_sympy()
            found = [
                v
                for v in _pins(term, symbol, modulus)
                if v.is_Integer
                and _pinned_by_delta(term, symbol, v, modulus)
                and all(_periodic(f, symbol, modulus) for f in factors)
            ]
            if not found:
                return None
            pins[symbol] = found[0]
        pinned.append((term, pins))
    if any(s in v.free_symbols for _, pins in pinned for v in pins.values() for s in pins):
        return None

    def at(expr: sp.Expr, pins: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
        return Scalar(expr.xreplace(dict(pins))).simplify(max_steps=max_steps).to_sympy()

    base = [at(f, pinned[0][1]) for f in factors]
    if any(b == 0 for b in base):
        return None
    entry: sp.Expr = sp.Integer(0)
    for term, pins in pinned:
        weight = at(term, pins)
        for factor, first, m in zip(factors, base, multiplicities, strict=True):
            ratio = Scalar(at(factor, pins) / first).simplify(max_steps=max_steps).to_sympy()
            ratio = ratio.rewrite(sp.exp)
            if ratio.has(sp.zoo, sp.nan) or any(
                str(x.name).startswith("_t") for x in ratio.free_symbols
            ):
                return None
            weight = weight * sp.Pow(ratio, m)
        entry = entry + weight
    return Scalar(entry).simplify(max_steps=max_steps), base


def _pin_boundary_in_factors(
    closed: Scalar,
    factors: list[sp.Expr],
    marked: list[sp.Expr | None],
    boundary: Sequence[sp.Symbol],
) -> tuple[list[sp.Expr], list[sp.Expr | None]]:
    """``factors`` and ``marked`` with each boundary index the entry pins by a delta factor
    replaced by its pinned value, where every factor is periodic in that index mod the delta's
    modulus."""
    deltas = [f for f in sp.Mul.make_args(closed.to_sympy()) if isinstance(f, ModDelta)]
    for delta in deltas:
        modulus = delta.args[1]
        for symbol in boundary:
            present = [e for e in (*factors, *(m for m in marked if m is not None))]
            if not any(symbol in e.free_symbols for e in present):
                continue
            pins = [
                v
                for v in _pins(delta, symbol, modulus)
                if not any(str(x.name).startswith("_t") for x in v.free_symbols)
            ]
            if not pins or not all(_periodic(e, symbol, modulus) for e in present):
                continue
            pin = {symbol: pins[0]}
            factors = [f.xreplace(pin) for f in factors]
            marked = [None if m is None else m.xreplace(pin) for m in marked]
            break
    return factors, marked


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

    def moved(r: PortRef) -> PortRef:
        return PortRef(id_map[r.node_id], r.direction, r.index)

    pending = True
    while pending:
        pending = False
        for box_id in sorted(diagram.bang_boxes):
            box = diagram.bang_boxes[box_id]
            if box_id in box_map or box.parent not in box_map:
                continue
            if box.pair_of is not None and box_map.get(box.pair_of) is None:
                continue
            box_map[box_id] = extracted.add_bang_box(
                box.multiplicity,
                node_scope=frozenset(id_map[n] for n in box.node_scope),
                port_scope=frozenset(moved(r) for r in box.port_scope),
                parent=box_map[box.parent],
                links=tuple((moved(p), moved(q)) for p, q in box.links),
                cyclic=box.cyclic,
                index=box.index,
                copy_scalar=box.copy_scalar,
                pair_of=None if box.pair_of is None else box_map[box.pair_of],
                pair_index=box.pair_index,
                left_ports=frozenset(moved(r) for r in box.left_ports),
                right_ports=frozenset(moved(r) for r in box.right_ports),
                tree=tuple(moved(r) for r in box.tree),
                tree_parent=None if box.tree_parent is None else moved(box.tree_parent),
                level=box.level,
                position=box.position,
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


def _pair_terms(
    group: ReplicatedFactor,
    g: int,
    copies: Sequence[Mapping[str, sp.Symbol]],
    fixed: Mapping[str, sp.Symbol],
) -> list[sp.Expr]:
    """``group``'s pair factor at every pair of copies ``j < k``."""
    if group.pairs is None:
        return []
    pair = group.pairs.to_sympy()
    terms: list[sp.Expr] = []
    for k in range(len(copies)):
        for j in range(k):
            names: dict[str, sp.Expr] = {**fixed, **copies[j]}
            names.update(
                {f"_u{name[2:]}": v for name, v in copies[k].items() if name.startswith("_t")}
            )
            names.update({f"_j{g}": sp.Integer(j), f"_k{g}": sp.Integer(k)})
            terms.append(_rename(pair, names))
    return terms


def _detach_pairs(diagram: Diagram, max_steps: int) -> tuple[Diagram, dict[BangBoxId, list[_Pair]]]:
    """``diagram`` without its pair boxes, each left as its tensor over the parent's ports it
    met, by parent."""
    working = diagram.copy()
    found: dict[BangBoxId, list[_Pair]] = {}
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if box.pair_of is None:
            continue
        parent = diagram.bang_boxes[box.pair_of]
        if parent.parent is not None or parent.index is None or box.pair_index is None:
            raise SymbolicContractionUnsupportedError(f"pair box {box_id}: a nested parent")
        if any(b.parent == box_id for b in diagram.bang_boxes.values()):
            raise SymbolicContractionUnsupportedError(f"pair box {box_id} holds a child box")
        attached: list[tuple[PortRef, PortRef, str]] = []
        for wire in sorted(crossing_wires(diagram, box.node_scope), key=Wire.sort_key):
            inside = _inside_end(diagram, box.node_scope, wire)
            outside = _outside_end(diagram, box.node_scope, wire)
            side = "left" if outside in box.left_ports else "right"
            if outside not in box.left_ports and outside not in box.right_ports:
                raise SymbolicContractionUnsupportedError(
                    f"pair box {box_id} meets a port that is neither left nor right"
                )
            node = diagram.nodes[outside.node_id]
            if node.generator_type.name != Z_SPIDER.name:
                raise SymbolicContractionUnsupportedError(
                    f"pair box {box_id} meets node {outside.node_id}, not a Z spider"
                )
            attached.append((inside, outside, side))
        outs = [a[0] for a in attached if a[0].direction is Direction.OUTPUT]
        ins = [a[0] for a in attached if a[0].direction is Direction.INPUT]
        sub = _scope_subdiagram(diagram, box.node_scope, box_id, outs, ins)
        tensor = contract_symbolic(sub, max_steps=max_steps)
        if tensor.is_variable_rank:
            raise SymbolicContractionUnsupportedError(f"pair box {box_id}: a variable-rank part")
        axis_of = {
            ref: _engine_index(a.index) for ref, a in zip([*outs, *ins], tensor.axes, strict=True)
        }
        entry = tensor.entry.to_sympy()
        if box.copy_scalar is not None:
            entry = entry * box.copy_scalar.to_sympy()
        legs = tuple((axis_of[inside], outside, side) for inside, outside, side in attached)
        found.setdefault(box.pair_of, []).append(_Pair(entry, legs, parent.index, box.pair_index))
        for node_id in sorted(box.node_scope):
            working.remove_node(node_id)
        working.remove_bang_box(box_id)
    return working, found


@dataclass(frozen=True, slots=True)
class _Pair:
    """A detached pair box: its tensor, each leg's axis symbol with the parent port and side
    it met, and the parent's and its own index names."""

    entry: sp.Expr
    legs: tuple[tuple[sp.Symbol, PortRef, str], ...]
    index: str
    pair_index: str


def _pair_factor(pairs: Sequence[_Pair], pinned: Mapping[PortRef, sp.Expr], g: int) -> sp.Expr:
    """The product of ``pairs``' tensors with each leg at its parent port's pinned value, copy
    ``j``'s template indices on a left leg and copy ``k``'s (``_u<g>_<i>``) on a right one,
    the indices renamed ``_j<g>`` and ``_k<g>``."""
    out: sp.Expr = sp.Integer(1)
    for pair in pairs:
        values: dict[sp.Symbol, sp.Expr] = {}
        for symbol, ref, side in pair.legs:
            value = pinned[ref]
            if side == "right":
                value = _rename(
                    value,
                    {
                        str(x): _engine_index(f"_u{str(x)[2:]}")
                        for x in value.free_symbols
                        if str(x).startswith(f"_t{g}_")
                    },
                )
            values[symbol] = value
        entry = pair.entry.xreplace(values)
        out = out * _rename(
            entry,
            {pair.index: _engine_index(f"_j{g}"), pair.pair_index: _engine_index(f"_k{g}")},
        )
    return out


def _hidden_pins(
    entry: sp.Expr, hidden: Mapping[PortRef, sp.Symbol], dims: Mapping[PortRef, Dim], max_steps: int
) -> tuple[sp.Expr, dict[PortRef, sp.Expr]]:
    """``entry`` with each hidden leg replaced by the value a delta factor pins it to, free of
    every hidden leg, and those values; raise when one is not pinned."""
    values: dict[PortRef, sp.Expr] = {}
    others = set(hidden.values())
    for ref, symbol in hidden.items():
        modulus = dims[ref].to_sympy()
        for value in _pins(entry, symbol, modulus):
            if value.free_symbols & others:
                continue
            folded = entry.xreplace({symbol: value})
            if (
                Scalar(entry - ModDelta(symbol - value, modulus) * folded)
                .simplify(max_steps=max_steps)
                .is_zero
            ):
                entry, values[ref] = folded, value
                break
        else:
            raise SymbolicContractionUnsupportedError(
                f"the spider at {ref} that a pair box meets is not pinned by the boundary"
            )
    return entry, values


def _copy_numbers(index: str | None, count: int) -> list[dict[str, sp.Expr]]:
    """Per copy, the index symbol's value: ``{index: m}`` for copy ``m``, or nothing."""
    if index is None:
        return [{} for _ in range(count)]
    return [{index: sp.Integer(m)} for m in range(count)]


_COPY = sp.Symbol("_copy", integer=True, nonnegative=True)


def _index_power(expr: sp.Expr, box: BangBox) -> sp.Expr:
    """``expr`` over the copies of ``box``: ``expr^n``, or with an index ``I`` read by ``expr``
    the closed form of ``prod_{m=0}^{n-1} expr(I = m)``; raise when it has none."""
    count = box.multiplicity.to_sympy()
    named = [x for x in expr.free_symbols if box.index is not None and str(x) == box.index]
    if not named:
        return sp.Pow(expr, count)
    product = _closed_product(expr.xreplace({named[0]: _COPY}), count)
    if product is None:
        raise SymbolicContractionUnsupportedError(
            f"bang box {box.id}: the product over copies of {expr} has no closed form"
        )
    return product


def _closed_product(expr: sp.Expr, count: sp.Expr) -> sp.Expr | None:
    """``prod_{_copy=0}^{count-1} expr`` in closed form: constant factors to the power
    ``count``, exponentials and powers of constant bases through a closed sum of their
    exponents, anything else through sympy's product; None when one stays unevaluated."""
    out: sp.Expr = sp.Integer(1)
    for factor in sp.Mul.make_args(sp.powsimp(sp.expand_power_base(expr, force=True))):
        if _COPY not in factor.free_symbols:
            out = out * sp.Pow(factor, count)
            continue
        base, exponent = (
            (sp.E, factor.args[0]) if isinstance(factor, sp.exp) else factor.as_base_exp()
        )
        if _COPY not in base.free_symbols:
            total = sp.summation(exponent, (_COPY, 0, count - 1))
            if total.has(sp.Sum):
                return None
            out = out * sp.Pow(base, sp.expand(total))
            continue
        product = sp.product(factor, (_COPY, 0, count - 1))
        if product.has(sp.Product):
            return None
        out = out * product
    return out


def _partner_or_slot(diagram: Diagram, ref: PortRef) -> PortRef | None:
    """The port wired to ``ref``, or None when ``ref`` is a boundary slot."""
    for wire in diagram.wires:
        if wire.a == ref:
            return wire.b
        if wire.b == ref:
            return wire.a
    return None


def _open_copy(diagram: Diagram, box_id: BangBoxId) -> tuple[Diagram, list[PortRef]]:
    """One copy of a chain box's scope with every link port, crossing and boundary slot
    open: outputs then inputs, each by port order; the list of open ports in axis order."""
    box = diagram.bang_boxes[box_id]
    working = diagram.copy()
    for p_ref, q_ref in box.links:
        for ref in (p_ref, q_ref):
            partner = _partner_or_slot(working, ref)
            if partner is not None:
                working.remove_wire(ref, partner)
    scope = box.node_scope
    inside = {
        ref
        for wire in working.wires
        if wire.a.node_id in scope and wire.b.node_id in scope
        for ref in (wire.a, wire.b)
    }
    open_ports = sorted(
        (
            PortRef(node_id, direction, i)
            for node_id in scope
            for direction in (Direction.OUTPUT, Direction.INPUT)
            for i in range(len(working.nodes[node_id].legs(direction)))
            if PortRef(node_id, direction, i) not in inside
        ),
        key=lambda r: (r.direction is Direction.INPUT, r.sort_key()),
    )
    outputs = [r for r in open_ports if r.direction is Direction.OUTPUT]
    inputs = [r for r in open_ports if r.direction is Direction.INPUT]
    return _scope_subdiagram(working, scope, box_id, outputs, inputs), [*outputs, *inputs]


def _carries_copy(entry: sp.Expr, p: sp.Symbol, q: sp.Symbol, dim: Dim, max_steps: int) -> bool:
    """Whether ``entry`` is ``delta(p - q)`` times ``entry`` at ``p = q``."""
    modulus = dim.to_sympy()
    for arg in sp.Mul.make_args(entry):
        if isinstance(arg, ModDelta) and sp.simplify(arg.args[1] - modulus) == 0:
            gap = sp.expand(arg.args[0])
            if sp.expand(gap - (p - q)) == 0 or sp.expand(gap + (p - q)) == 0:
                return True
    folded = entry.xreplace({p: q}) * ModDelta(p - q, modulus)
    return Scalar(entry - folded).simplify(max_steps=max_steps).is_zero


def _resolve_chains(diagram: Diagram, max_steps: int) -> Diagram:
    """``diagram`` with each top-level carry-copy chain box replaced by a plain box whose
    link ports meet one phaseless Z spider per link, the spider also taking the head and tail
    ends; raise on any other chain box."""
    working = diagram
    for box_id in sorted(diagram.bang_boxes):
        box = working.bang_boxes[box_id]
        if not box.links or box.parent is not None:
            continue
        if box.cyclic:
            constant = sp.expand(box.multiplicity.to_sympy()).as_coeff_Add()[0]
            if not (constant.is_Integer and constant >= 1):
                raise SymbolicContractionUnsupportedError(
                    f"cyclic chain box {box_id}: multiplicity {box.multiplicity} may be 0"
                )
        copy, ports = _open_copy(working, box_id)
        tensor = contract_symbolic(copy, max_steps=max_steps)
        if tensor.is_variable_rank:
            raise SymbolicContractionUnsupportedError(
                f"chain box {box_id}: one copy holds a variable-rank part"
            )
        axis_of = {ref: _engine_index(a.index) for ref, a in zip(ports, tensor.axes, strict=True)}
        entry = tensor.entry.to_sympy()
        for p_ref, q_ref in box.links:
            if not _carries_copy(
                entry, axis_of[p_ref], axis_of[q_ref], _port_dim(working, p_ref), max_steps
            ):
                raise SymbolicContractionUnsupportedError(
                    f"chain box {box_id}: the transfer tensor is neither carry-copy nor neighbour"
                )
        working = _spider_links(working, box_id)
    return working


MAX_PERIOD = 8
"""The largest power :func:`chain_period` tries."""

MAX_TRANSFER_SIZE = 64
"""The largest transfer matrix side :func:`chain_period` builds."""


def chain_period(diagram: Diagram, box_id: BangBoxId, max_steps: int = 64) -> tuple[int, sp.Expr]:
    """The least ``p <= MAX_PERIOD`` and the nonzero ``c`` with ``T^p = c I``, ``T`` the exact
    matrix of one copy of chain box ``box_id`` (copy scalar included) from its link-in to its
    link-out values; raise SymbolicContractionUnsupportedError when the copy has another open
    port, an index, a symbol or a non-concrete link dimension, or no such ``p`` exists."""
    box = diagram.bang_boxes[box_id]
    if not box.links or box.cyclic or box.index is not None or box.is_tree:
        raise SymbolicContractionUnsupportedError(f"chain box {box_id}: not a plain open chain")
    if any(other.parent == box_id for other in diagram.bang_boxes.values()):
        raise SymbolicContractionUnsupportedError(f"chain box {box_id} has a child box")
    copy, ports = _open_copy(diagram, box_id)
    outs = [p for p, _q in box.links]
    ins = [q for _p, q in box.links]
    if sorted(ports, key=PortRef.sort_key) != sorted([*outs, *ins], key=PortRef.sort_key):
        raise SymbolicContractionUnsupportedError(
            f"chain box {box_id}: a copy has an open port other than its links"
        )
    tensor = contract_symbolic(copy, max_steps=max_steps)
    if tensor.is_variable_rank:
        raise SymbolicContractionUnsupportedError(f"chain box {box_id}: a variable-rank copy")
    axis_of = {ref: _engine_index(a.index) for ref, a in zip(ports, tensor.axes, strict=True)}
    entry = tensor.entry.to_sympy()
    if box.copy_scalar is not None:
        entry = entry * box.copy_scalar.to_sympy()
    if entry.free_symbols - set(axis_of.values()):
        raise SymbolicContractionUnsupportedError(f"chain box {box_id}: a copy reads a symbol")
    dims = [_port_dim(diagram, p).to_sympy() for p in outs]
    if not all(d.is_Integer for d in dims) or math.prod(int(d) for d in dims) > MAX_TRANSFER_SIZE:
        raise SymbolicContractionUnsupportedError(
            f"chain box {box_id}: link dimensions not concrete or above {MAX_TRANSFER_SIZE}"
        )
    values = list(itertools.product(*(range(int(d)) for d in dims)))
    size = len(values)
    transfer = sp.Matrix(
        size,
        size,
        lambda i, j: sp.simplify(
            entry.xreplace(
                {
                    **{axis_of[p]: v for p, v in zip(outs, values[i], strict=True)},
                    **{axis_of[q]: v for q, v in zip(ins, values[j], strict=True)},
                }
            )
        ),
    )
    power = sp.eye(size)
    for p in range(1, MAX_PERIOD + 1):
        power = (power * transfer).applyfunc(sp.simplify)
        c = power[0, 0]
        if c != 0 and (power - c * sp.eye(size)).applyfunc(sp.simplify).is_zero_matrix:
            return p, c
    raise SymbolicContractionUnsupportedError(
        f"chain box {box_id}: no power up to {MAX_PERIOD} of the transfer matrix is scalar"
    )


def _resolve_trees(diagram: Diagram, max_steps: int) -> Diagram:
    """``diagram`` with each top-level carry-copy tree box (one copy's tensor is
    prod_i delta(p_i - q) times a constant c, no other open port) replaced by a phaseless Z
    spider from the head end whose one leaf leg is port-boxed b^k times, the diagram scalar
    times c^((b^k - 1)/(b - 1)); raise on any other tree box."""
    working = diagram
    for box_id in sorted(diagram.bang_boxes):
        box = working.bang_boxes[box_id]
        if not box.is_tree or box.parent is not None:
            continue
        assert box.tree_parent is not None
        copy, ports = _open_copy(working, box_id)
        if set(ports) != {box.tree_parent, *box.tree}:
            raise SymbolicContractionUnsupportedError(
                f"tree box {box_id}: a copy meets more than its tree ports"
            )
        tensor = contract_symbolic(copy, max_steps=max_steps)
        if tensor.is_variable_rank:
            raise SymbolicContractionUnsupportedError(f"tree box {box_id}: a variable-rank copy")
        axis_of = {ref: _engine_index(a.index) for ref, a in zip(ports, tensor.axes, strict=True)}
        q = axis_of[box.tree_parent]
        dim = _port_dim(working, box.tree_parent)
        entry = tensor.entry.to_sympy()
        folded = entry.xreplace({axis_of[p]: q for p in box.tree})
        product = sp.Mul(*(ModDelta(axis_of[p] - q, dim.to_sympy()) for p in box.tree))
        constant = Scalar(folded).simplify(max_steps=max_steps).to_sympy()
        names = {str(x) for x in constant.free_symbols}
        if (
            not Scalar(entry - product * folded).simplify(max_steps=max_steps).is_zero
            or q in constant.free_symbols
            or names & {box.level, box.position, box.index}
        ):
            raise SymbolicContractionUnsupportedError(
                f"tree box {box_id}: the copy is not a carry-copy with a constant factor"
            )
        working = _spider_tree(working, box_id, constant)
    return working


def _fan_candidate(diagram: Diagram, box: BangBox) -> tuple[PortRef, PortRef] | None:
    """The ``(inside, outside)`` crossing of a top-level plain node-scope box without children
    whose one crossing meets a Z spider outside every box, its other open ports boundary slots
    facing that spider leg's way; None otherwise."""
    if (
        not box.is_node_scope
        or box.parent is not None
        or box.links
        or box.pair_of is not None
        or box.tree
        or box.feed is not None
        or box.index is not None
        or any(
            other.parent == box.id or other.pair_of == box.id
            for other in diagram.bang_boxes.values()
        )
    ):
        return None
    crossings = outside_crossings(diagram, box.node_scope)
    if len(crossings) != 1:
        return None
    inside, outside = crossings[0]
    spider = diagram.nodes[outside.node_id]
    if spider.generator_type is not Z_SPIDER:
        return None
    if any(
        outside.node_id in other.node_scope or outside in other.port_scope
        for other in diagram.bang_boxes.values()
    ):
        return None
    slots = boundary_refs_in_scope(diagram, box.node_scope)
    if not slots or any(r.direction is not outside.direction for r in slots):
        return None
    return inside, outside


def _resolve_fans(diagram: Diagram, max_steps: int) -> Diagram:
    """``diagram`` with each box fanned into a Z spider whose copy is a carry-copy (one copy's
    tensor is prod_i delta(p_i - q) times a constant c, q its crossing, the p_i its b boundary
    slots) replaced by that spider's leg port-boxed b m times in the box's boundary block, the
    scalar times (c s)^m for multiplicity m and copy scalar s."""
    working = diagram
    for box_id in sorted(diagram.bang_boxes):
        box = working.bang_boxes.get(box_id)
        if box is None:
            continue
        found = _fan_candidate(working, box)
        if found is None:
            continue
        inside, outside = found
        copy, ports = _open_copy(working, box_id)
        slots = boundary_refs_in_scope(working, box.node_scope)
        if set(ports) != {inside, *slots}:
            continue
        try:
            tensor = contract_symbolic(copy, max_steps=max_steps)
        except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError):
            continue
        if tensor.is_variable_rank:
            continue
        axis_of = {ref: _engine_index(a.index) for ref, a in zip(ports, tensor.axes, strict=True)}
        q = axis_of[inside]
        dim = _port_dim(working, inside)
        entry = tensor.entry.to_sympy()
        folded = entry.xreplace({axis_of[p]: q for p in slots})
        product = sp.Mul(*(ModDelta(axis_of[p] - q, dim.to_sympy()) for p in slots))
        constant = Scalar(folded).simplify(max_steps=max_steps).to_sympy()
        if (
            not Scalar(entry - product * folded).simplify(max_steps=max_steps).is_zero
            or q in constant.free_symbols
        ):
            continue
        working = working.copy()
        refs = list(
            working.boundary_outputs
            if outside.direction is Direction.OUTPUT
            else working.boundary_inputs
        )
        first = min(refs.index(r) for r in slots)
        kept = [r for r in refs[:first] if r not in slots]
        rest = [r for r in refs[first:] if r not in slots]
        working.remove_wire(inside, outside)
        for node_id in sorted(box.node_scope):
            working.remove_node(node_id)
        working.remove_bang_box(box_id)
        setter = (
            working.set_boundary_outputs
            if outside.direction is Direction.OUTPUT
            else working.set_boundary_inputs
        )
        setter([*kept, outside, *rest])
        m = box.multiplicity.to_sympy()
        working.add_bang_box(
            Mult._from_expr(sp.powsimp(sp.expand(len(slots) * m), force=True)),
            port_scope=frozenset({outside}),
        )
        factor = constant if box.copy_scalar is None else constant * box.copy_scalar.to_sympy()
        working.multiply_scalar(Scalar(sp.Pow(factor, m)))
    return working


def _spider_tree(diagram: Diagram, box_id: BangBoxId, constant: sp.Expr) -> Diagram:
    """``diagram`` with tree box ``box_id`` and its scope replaced by a phaseless Z spider from
    the head end, its leaf leg in the leaf run's place port-boxed b^k times."""
    working = diagram.copy()
    box = working.bang_boxes[box_id]
    assert box.tree_parent is not None
    b = len(box.tree)
    dim = _port_dim(working, box.tree_parent)
    head = _partner_or_slot(working, box.tree_parent)
    leaf_out = box.tree[0].direction is Direction.OUTPUT
    spider = working.add_node(Z_SPIDER, [dim], [dim], phase=PhaseVector(dim))
    near = PortRef(spider, Direction.INPUT if leaf_out else Direction.OUTPUT, 0)
    leaf = PortRef(spider, Direction.OUTPUT if leaf_out else Direction.INPUT, 0)
    refs = list(working.boundary_outputs if leaf_out else working.boundary_inputs)
    first = refs.index(box.tree[0])
    refs[first : first + b] = [leaf]
    (working.set_boundary_outputs if leaf_out else working.set_boundary_inputs)(refs)
    if head is not None:
        working.remove_wire(box.tree_parent, head)
        working.add_wire(head, near)
    else:
        others = list(working.boundary_inputs if leaf_out else working.boundary_outputs)
        others = [near if r == box.tree_parent else r for r in others]
        (working.set_boundary_inputs if leaf_out else working.set_boundary_outputs)(others)
    k = box.multiplicity.to_sympy()
    for child_id, child in sorted(working.bang_boxes.items()):
        if child.parent == box_id:
            raise SymbolicContractionUnsupportedError(f"tree box {box_id} holds a child box")
    for node_id in sorted(box.node_scope):
        working.remove_node(node_id)
    working.remove_bang_box(box_id)
    leaves = sp.Integer(b) ** k
    working.add_bang_box(Mult._from_expr(leaves), port_scope=frozenset({leaf}))
    copies = k if b == 1 else (leaves - 1) / (b - 1)
    working.multiply_scalar(Scalar(sp.Pow(constant, copies)))
    return working


def _flip(direction: Direction) -> Direction:
    """The other direction."""
    return Direction.INPUT if direction is Direction.OUTPUT else Direction.OUTPUT


def _spider_links(diagram: Diagram, box_id: BangBoxId) -> Diagram:
    """``diagram`` with box ``box_id``'s links dropped, each link's ``p``, ``q``, head and tail
    ends wired to one new phaseless Z spider."""
    working = diagram.copy()
    box = working.bang_boxes[box_id]
    for p_ref, q_ref in box.links:
        dim = _port_dim(working, p_ref)
        ends: list[tuple[Direction, PortRef, str | None]] = []
        for ref in (p_ref, q_ref):
            partner = _partner_or_slot(working, ref)
            if partner is not None:
                working.remove_wire(ref, partner)
            ends.append((_flip(ref.direction), ref, None))
            if box.cyclic:
                continue
            if partner is not None:
                ends.append((_flip(partner.direction), partner, None))
            elif ref in working.boundary_inputs:
                ends.append((Direction.INPUT, ref, "in"))
            else:
                ends.append((Direction.OUTPUT, ref, "out"))
        ins = [e for e in ends if e[0] is Direction.INPUT]
        outs = [e for e in ends if e[0] is Direction.OUTPUT]
        spider = working.add_node(
            Z_SPIDER, [dim] * len(ins), [dim] * len(outs), phase=PhaseVector(dim)
        )
        for direction, group in ((Direction.INPUT, ins), (Direction.OUTPUT, outs)):
            for i, (_d, ref, slot) in enumerate(group):
                leg = PortRef(spider, direction, i)
                if slot == "in":
                    working.set_boundary_inputs(
                        [leg if r == ref else r for r in working.boundary_inputs]
                    )
                elif slot == "out":
                    working.set_boundary_outputs(
                        [leg if r == ref else r for r in working.boundary_outputs]
                    )
                else:
                    working.add_wire(leg, ref)
    working.set_bang_box_extension(box_id, links=(), cyclic=False)
    return working


@dataclass(frozen=True, slots=True)
class _BoxPlan:
    """Every top-level symbolic box of a diagram, sorted by how it contracts."""

    closed: tuple[BangBoxId, ...]
    replicated: tuple[BangBoxId, ...]
    ported: tuple[BangBoxId, ...]
    crossing: tuple[BangBoxId, ...] = ()
    bridged: tuple[BangBoxId, ...] = ()


def _plan_boxes(diagram: Diagram) -> _BoxPlan:
    """Sort every top-level box: a node scope closed off, a node scope meeting only the
    boundary, a node scope meeting the rest only at wires into Z and X spiders outside every
    box, one meeting the boundary and such wires, or a port scope on Z and X spiders and at
    most one W node; raise on any other box."""
    closed: list[BangBoxId] = []
    replicated: list[BangBoxId] = []
    ported: list[BangBoxId] = []
    crossing: list[BangBoxId] = []
    bridged: list[BangBoxId] = []
    boxed_nodes = {
        node_id
        for box in diagram.bang_boxes.values()
        if box.is_node_scope
        for node_id in box.node_scope
    }
    ported_nodes = {
        ref.node_id
        for box in diagram.bang_boxes.values()
        if not box.is_node_scope
        for ref in box.port_scope
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
            elif _crosses_into_spiders(diagram, box.node_scope, boxed_nodes, ported_nodes):
                crossing.append(box_id)
            elif _wires_into_spiders(diagram, box.node_scope, boxed_nodes, ported_nodes):
                bridged.append(box_id)
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
    meeting = [*crossing, *bridged]
    touched = [
        (box_id, _outside_end(diagram, diagram.bang_boxes[box_id].node_scope, w).node_id)
        for box_id in meeting
        for w in crossing_wires(diagram, diagram.bang_boxes[box_id].node_scope)
    ]
    owners: dict[NodeId, set[BangBoxId]] = {}
    for box_id, node_id in touched:
        owners.setdefault(node_id, set()).add(box_id)
    if (len(crossing) > 1 and len({n for _b, n in touched}) < len(touched)) or any(
        len(boxes) > 1 for boxes in owners.values()
    ):
        raise SymbolicContractionUnsupportedError(
            "two node-scope bang boxes meet one spider at wires"
        )
    return _BoxPlan(
        tuple(closed), tuple(replicated), tuple(ported), tuple(crossing), tuple(bridged)
    )


def _outside_end(diagram: Diagram, scope: frozenset[NodeId], wire: Wire) -> PortRef:
    """The end of a crossing ``wire`` outside ``scope``."""
    return wire.b if wire.a.node_id in scope else wire.a


def _inside_end(diagram: Diagram, scope: frozenset[NodeId], wire: Wire) -> PortRef:
    """The end of a crossing ``wire`` inside ``scope``."""
    return wire.a if wire.a.node_id in scope else wire.b


def _crosses_into_spiders(
    diagram: Diagram,
    scope: frozenset[NodeId],
    boxed_nodes: set[NodeId],
    ported_nodes: set[NodeId],
) -> bool:
    """Whether ``scope`` meets no boundary slot and every wire leaving it ends at a Z or X
    spider outside every box and every port scope."""
    if boundary_refs_in_scope(diagram, scope):
        return False
    for wire in crossing_wires(diagram, scope):
        end = _outside_end(diagram, scope, wire)
        node = diagram.nodes[end.node_id]
        if node.generator_type.name not in (Z_SPIDER.name, X_SPIDER.name):
            return False
        if end.node_id in boxed_nodes or end.node_id in ported_nodes:
            return False
    return True


def _wires_into_spiders(
    diagram: Diagram,
    scope: frozenset[NodeId],
    boxed_nodes: set[NodeId],
    ported_nodes: set[NodeId],
) -> bool:
    """Whether ``scope`` meets a boundary slot and every wire leaving it ends at a Z or X
    spider outside every box and every port scope."""
    if not boundary_refs_in_scope(diagram, scope):
        return False
    for wire in crossing_wires(diagram, scope):
        end = _outside_end(diagram, scope, wire)
        node = diagram.nodes[end.node_id]
        if node.generator_type.name not in (Z_SPIDER.name, X_SPIDER.name):
            return False
        if end.node_id in boxed_nodes or end.node_id in ported_nodes:
            return False
    return True


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
    from archytaszx.diagram.mixed import is_pure  # local: mixed imports bangbox

    if not is_pure(diagram):
        raise SymbolicContractionUnsupportedError(
            "the diagram has a discard; contract the doubled diagram (contract_symbolic_pair)"
        )

    diagram = expand_concrete_boxes(diagram)
    if not diagram.bang_boxes:
        reduced = _path_sum_tensor(diagram, max_steps)
        if reduced is not None:
            return reduced
    if any(box.links for box in diagram.bang_boxes.values()):
        diagram = _resolve_chains(diagram, max_steps)
    if any(box.is_fed for box in diagram.bang_boxes.values()):
        raise SymbolicContractionUnsupportedError("a tree box feeds a box")
    if any(box.is_tree for box in diagram.bang_boxes.values()):
        diagram = _resolve_trees(diagram, max_steps)
    if any(_fan_candidate(diagram, box) is not None for box in diagram.bang_boxes.values()):
        diagram = _resolve_fans(diagram, max_steps)
    pair_boxes: dict[BangBoxId, list[_Pair]] = {}
    if any(box.pair_of is not None for box in diagram.bang_boxes.values()):
        diagram, pair_boxes = _detach_pairs(diagram, max_steps)
    box_factor: sp.Expr = sp.Integer(1)
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if box.pair_of is not None or box.links or box.is_tree:
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id} is a pair or nested chain box"
            )
        if box.index is not None and any(
            box.index in child.multiplicity.free_symbols
            for child in diagram.bang_boxes.values()
            if child.parent == box_id and not child.is_node_scope
        ):
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id}: a port-scope child's multiplicity reads its index"
            )
        if box.copy_scalar is not None and box.parent is None:
            box_factor = box_factor * _index_power(box.copy_scalar.to_sympy(), box)
    plan = _plan_boxes(diagram)
    if set(pair_boxes) - set(plan.replicated):
        raise SymbolicContractionUnsupportedError(
            "a box with pair boxes does not meet only the boundary"
        )

    working = diagram.copy()
    for box_id in plan.closed:
        box = diagram.bang_boxes[box_id]
        inner = contract_symbolic(
            _scope_subdiagram(diagram, box.node_scope, box_id), max_steps=max_steps
        )
        box_factor = box_factor * _index_power(inner.entry.to_sympy(), box)
        _drop_scope(working, box.node_scope)

    crossing_leg: dict[PortRef, BangBoxId] = {}
    crossing_inner: dict[BangBoxId, tuple[SymbolicTensor, dict[PortRef, sp.Symbol]]] = {}
    for box_id in plan.crossing:
        box = diagram.bang_boxes[box_id]
        wires = sorted(crossing_wires(diagram, box.node_scope), key=Wire.sort_key)
        inner_out = [
            _inside_end(diagram, box.node_scope, w)
            for w in wires
            if _inside_end(diagram, box.node_scope, w).direction is Direction.OUTPUT
        ]
        inner_in = [
            _inside_end(diagram, box.node_scope, w)
            for w in wires
            if _inside_end(diagram, box.node_scope, w).direction is Direction.INPUT
        ]
        inner = contract_symbolic(
            _scope_subdiagram(diagram, box.node_scope, box_id, inner_out, inner_in),
            max_steps=max_steps,
        )
        if inner.is_variable_rank:
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id} meets spiders at wires and holds a variable-rank part"
            )
        legs: dict[PortRef, sp.Symbol] = {}
        for axis_ref, axis in zip([*inner_out, *inner_in], inner.axes, strict=True):
            wire = next(w for w in wires if _inside_end(diagram, box.node_scope, w) == axis_ref)
            outside = _outside_end(diagram, box.node_scope, wire)
            legs[outside] = _engine_index(f"_c{len(crossing_leg)}")
            crossing_leg[outside] = box_id
            inner = SymbolicTensor(
                inner.axes,
                Scalar(_rename(inner.entry.to_sympy(), {axis.index: legs[outside]})),
            )
        crossing_inner[box_id] = (inner, legs)
        _drop_scope(working, box.node_scope)

    scope_of: dict[NodeId, BangBoxId] = {
        node_id: box_id
        for box_id in (*plan.replicated, *plan.bridged)
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
        emitted: set[tuple[BangBoxId, int]] = set()
        half = len(refs) // 2

        def part(owner: BangBoxId, position: int, half: int = half) -> int:
            return int(diagram.bang_boxes[owner].sheets and position >= half)

        for position, ref in enumerate(refs):
            owner = scope_of.get(ref.node_id)
            if owner is not None:
                key = (owner, part(owner, position))
                if key not in emitted:
                    emitted.add(key)
                    g = group(owner)
                    run = [
                        r
                        for i, r in enumerate(refs)
                        if scope_of.get(r.node_id) == owner and part(owner, i) == key[1]
                    ]
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
    pair_of: dict[int, sp.Expr] = {}
    for box_id in plan.replicated:
        box = diagram.bang_boxes[box_id]
        g = group(box_id)
        if len(factors) <= g:
            factors.append(sp.Integer(1))
        run_out = [a.port for a in templates[g] if a.port.direction is Direction.OUTPUT]
        run_in = [a.port for a in templates[g] if a.port.direction is Direction.INPUT]
        met = sorted(
            {leg[1] for pair in pair_boxes.get(box_id, ()) for leg in pair.legs},
            key=PortRef.sort_key,
        )
        hidden_out = [r for r in met if r.direction is Direction.OUTPUT]
        hidden_in = [r for r in met if r.direction is Direction.INPUT]
        inner = contract_symbolic(
            _scope_subdiagram(
                diagram, box.node_scope, box_id, [*run_out, *hidden_out], [*run_in, *hidden_in]
            ),
            max_steps=max_steps,
        )
        if inner.is_variable_rank:
            if met:
                raise SymbolicContractionUnsupportedError(
                    f"bang box {box_id}: pair boxes over a variable-rank copy"
                )
            inner_of[g] = inner
            _drop_scope(working, box.node_scope)
            continue
        template_of = {t.port: _engine_index(t.index) for t in templates[g]}
        names: dict[str, sp.Expr] = {}
        hidden: dict[PortRef, sp.Symbol] = {}
        axis_refs = [*run_out, *hidden_out, *run_in, *hidden_in]
        for ref, axis in zip(axis_refs, inner.axes, strict=True):
            if ref in template_of:
                names[axis.index] = template_of[ref]
            else:
                hidden[ref] = _engine_index(f"_h{len(hidden)}")
                names[axis.index] = hidden[ref]
        factors[g] = _rename(inner.entry.to_sympy(), names)
        if met:
            factors[g], pinned = _hidden_pins(
                factors[g], hidden, {r: _port_dim(diagram, r) for r in met}, max_steps
            )
            pair_of[g] = _pair_factor(pair_boxes[box_id], pinned, g)
        _drop_scope(working, box.node_scope)
    bridged_leg: dict[PortRef, int] = {}
    bridged_sums: dict[int, list[tuple[sp.Symbol, Dim]]] = {}
    for box_id in plan.bridged:
        box = diagram.bang_boxes[box_id]
        g = group(box_id)
        if len(factors) <= g:
            factors.append(sp.Integer(1))
        wires = sorted(crossing_wires(diagram, box.node_scope), key=Wire.sort_key)
        ends = [_inside_end(diagram, box.node_scope, w) for w in wires]
        cross_out = [e for e in ends if e.direction is Direction.OUTPUT]
        cross_in = [e for e in ends if e.direction is Direction.INPUT]
        run_out = [a.port for a in templates[g] if a.port.direction is Direction.OUTPUT]
        run_in = [a.port for a in templates[g] if a.port.direction is Direction.INPUT]
        inner = contract_symbolic(
            _scope_subdiagram(
                diagram, box.node_scope, box_id, [*run_out, *cross_out], [*run_in, *cross_in]
            ),
            max_steps=max_steps,
        )
        if inner.is_variable_rank:
            raise SymbolicContractionUnsupportedError(
                f"bang box {box_id} meets the boundary and spiders and holds a variable-rank part"
            )
        template_of = {t.port: t.index for t in templates[g]}
        leg_names: dict[str, sp.Symbol] = {}
        sums: list[tuple[sp.Symbol, Dim]] = []
        for ref, axis in zip([*run_out, *cross_out, *run_in, *cross_in], inner.axes, strict=True):
            if ref in template_of:
                leg_names[axis.index] = _engine_index(template_of[ref])
                continue
            wire = next(w for w in wires if _inside_end(diagram, box.node_scope, w) == ref)
            outside = _outside_end(diagram, box.node_scope, wire)
            leg = _engine_index(f"_b{len(bridged_leg)}")
            leg_names[axis.index] = leg
            boxed[outside] = leg
            bridged_leg[outside] = g
            sums.append((leg, _port_dim(diagram, ref)))
        factors[g] = _rename(inner.entry.to_sympy(), leg_names)
        bridged_sums[g] = sums
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
    summed_shared: list[tuple[sp.Symbol, Dim]] = []
    for box_id in plan.crossing:
        for ref, symbol in crossing_inner[box_id][1].items():
            boxed[ref] = symbol
            if ref.node_id not in shared_of:
                shared_symbol = _engine_index(f"_s{len(shared) + len(summed_shared)}")
                shared_of[ref.node_id] = shared_symbol
                summed_shared.append((shared_symbol, resolve_dim(working.nodes[ref.node_id])))
    copy_factor: dict[BangBoxId, sp.Expr] = {box_id: sp.Integer(1) for box_id in plan.crossing}

    indices = _Indices()
    wire_indices: list[tuple[sp.Symbol, Dim]] = []
    for wire in sorted(working.wires, key=Wire.sort_key):
        symbol = indices.fresh()
        port_index[wire.a] = symbol
        port_index[wire.b] = symbol
        wire_indices.append((symbol, _port_dim(working, wire.a)))

    head: sp.Expr = working.scalar.to_sympy() * box_factor
    parts: list[sp.Expr] = []
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
            parts.append(part)
            for ref, pair in legs.items():
                w_legs.setdefault(group_of[port_box[ref]], []).append(pair)
            continue
        if node_id in shared_of:
            part, per_leg = _shared_spider_entry(
                node, node_id, port_index, boxed, shared_of[node_id]
            )
            parts.append(part)
            for ref, factor in per_leg.items():
                if ref in crossing_leg:
                    copy_factor[crossing_leg[ref]] = copy_factor[crossing_leg[ref]] * factor
                    continue
                if ref in bridged_leg:
                    factors[bridged_leg[ref]] = factors[bridged_leg[ref]] * factor
                    continue
                g = group_of[port_box[ref]]
                factors[g] = factors[g] * factor
            continue
        parts.append(_node_entry(working, node_id, port_index))

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

    for g, sums in sorted(bridged_sums.items()):
        summed: sp.Expr = factors[g]
        for symbol, dim in reversed(sums):
            summed = sp.Sum(summed, (symbol, 0, dim.to_sympy() - 1))
        factors[g] = Scalar(summed).simplify(max_steps=max_steps).to_sympy()

    for box_id in plan.crossing:
        inner, legs = crossing_inner[box_id]
        copy: sp.Expr = inner.entry.to_sympy() * copy_factor[box_id]
        for ref, symbol in reversed(list(legs.items())):
            copy = sp.Sum(copy, (symbol, 0, _port_dim(diagram, ref).to_sympy() - 1))
        copy = Scalar(copy).simplify(max_steps=max_steps).to_sympy()
        parts.append(_index_power(copy, diagram.bang_boxes[box_id]))
    closed = _sum_out(
        head,
        parts,
        [*summed_shared, *wire_indices],
        max_steps,
        {axis.index: axis.dim for axis in fixed_axes},
    )
    if not group_boxes:
        return SymbolicTensor(tuple(fixed_axes), closed)

    kept: list[SharedIndex] = []
    for index in shared:
        others = [*factors, *(m for m in marked if m is not None)]
        value = _pinned_value(closed, index, max_steps, others)
        if value is None:
            kept.append(index)
            continue
        pin = {_engine_index(index.name): value}
        closed = Scalar(closed.to_sympy().xreplace(pin)).simplify(max_steps=max_steps)
        factors = [factor.xreplace(pin) for factor in factors]
        marked = [None if m is None else m.xreplace(pin) for m in marked]
    in_factors: set[sp.Basic] = set()
    for expr in (*factors, *(m for m in marked if m is not None)):
        in_factors |= expr.free_symbols
    for index in list(kept):
        symbol = _engine_index(index.name)
        if symbol in in_factors:
            continue
        summed = sp.Sum(closed.to_sympy(), (symbol, 0, index.dim.to_sympy() - 1))
        closed = Scalar(summed).simplify(max_steps=max_steps)
        kept.remove(index)
    indexed = any(diagram.bang_boxes[b].index is not None for b in group_boxes)
    if kept and all(m is None for m in marked) and not indexed:
        split = _split_pinned_terms(
            closed,
            kept,
            factors,
            [diagram.bang_boxes[b].multiplicity.to_sympy() for b in group_boxes],
            max_steps,
        )
        if split is not None:
            closed, factors = split
            kept = []
    boundary = [_engine_index(axis.index) for axis in fixed_axes]
    factors, marked = _pin_boundary_in_factors(closed, factors, marked, boundary)

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

    groups: list[ReplicatedFactor] = []
    for g, (box_id, factor) in enumerate(zip(group_boxes, factors, strict=True)):
        box = diagram.bang_boxes[box_id]
        mark = marked[g]
        copy_index: str | None = None
        if box.index is not None:
            if g in inner_of:
                raise SymbolicContractionUnsupportedError(
                    f"bang box {box_id}: an indexed box holding a variable-rank part"
                )
            copy_index = f"_j{g}"
            to_index = {box.index: _engine_index(copy_index)}
            factor = _rename(factor, to_index)
            mark = None if mark is None else _rename(mark, to_index)
        pair_factor = pair_of.get(g)
        groups.append(
            ReplicatedFactor(
                box.multiplicity,
                Scalar(factor).simplify(max_steps=max_steps),
                None if mark is None else Scalar(mark).simplify(max_steps=max_steps),
                inner_of.get(g),
                copy_index,
                None if pair_factor is None else Scalar(pair_factor).simplify(max_steps=max_steps),
            )
        )
    return SymbolicTensor(
        tuple(fixed_axes),
        closed,
        tuple(nested(slot) for slot in layout),
        tuple(groups),
        tuple(kept),
    )


def _cyclo_expr(value: object) -> sp.Expr:
    """A cyclotomic number as its sum of rational multiples of roots of unity."""
    from archytaszx.algebra.pathsum import Cyclo

    assert isinstance(value, Cyclo)
    return sp.Add(
        *(
            sp.Rational(c.numerator, c.denominator)
            * sp.exp(2 * sp.pi * sp.I * sp.Rational(k, value.n))
            for k, c in sorted(value.c.items())
        )
    )


def _parity_expr(poly: frozenset[frozenset[int]], var: Mapping[int, sp.Expr]) -> sp.Expr:
    """An F_2 polynomial as the integer sum of its monomials, whose parity is its value."""
    return sp.Add(*(sp.Mul(*(var[v] for v in sorted(mono))) for mono in sorted(poly, key=sorted)))


def _path_sum_tensor(diagram: Diagram, max_steps: int) -> SymbolicTensor | None:
    """The contraction of a diagram of Z, X, F, B and S nodes at concrete power-of-two
    dimensions with boundary legs of dimension at most 2, read from its reduced path sum:
    constant, sqrt 2 power, a mod-2 delta per support constraint, a sum over each path variable
    left, the phase as one exponential and each amplitude atom ``a^F`` as
    ``[F even] + a [F odd]``; None outside that class or when the simplifier exceeds its
    budget."""
    from archytaszx.algebra.pathsum import PathSum, PathSumUnsupported, reduce
    from archytaszx.semantics.pathsum_diagram import diagram_base, from_diagram

    refs = (*diagram.boundary_outputs, *diagram.boundary_inputs)
    dims = [_port_dim(diagram, ref) for ref in refs]
    if not all(d.is_concrete and d.to_int() in (1, 2) for d in dims):
        return None
    try:
        if diagram_base(diagram) != 2:
            return None
        raw = from_diagram(diagram)
    except PathSumUnsupported:
        return None
    if not isinstance(raw, PathSum):
        return None
    ps = reduce(raw)
    axes = tuple(
        SymbolicAxis(ref, dim, f"_i{k}")
        for k, (ref, dim) in enumerate(zip(refs, dims, strict=True))
    )
    two = [axis for axis in axes if axis.dim.to_int() == 2]
    if len(two) != len(ps.boundary):
        return None
    var: dict[int, sp.Expr] = {
        v: _engine_index(axis.index) for v, axis in zip(ps.boundary, two, strict=True)
    }
    path = sorted(ps.path)
    for v in path:
        var[v] = _engine_index(f"_p{v}")
    used = ps.phase.variables() | ps.amp_vars()
    used |= {v for c in ps.constraints for m in c for v in m}
    if not used <= var.keys():
        return None
    phase = sp.Add(
        *(
            sp.Rational(c.numerator, c.denominator) * sp.Mul(*(var[v] for v in sorted(mono)))
            for mono, c in sorted(ps.phase.terms.items(), key=lambda item: sorted(item[0]))
        )
    )
    entry: sp.Expr = (
        _cyclo_expr(ps.const) * sp.sqrt(2) ** ps.sqrt2 * sp.exp(2 * sp.pi * sp.I * phase)
    )
    for constraint in ps.constraints:
        entry = entry * ModDelta(_parity_expr(constraint, var), 2)
    for base, exponent in ps.amps:
        even = ModDelta(_parity_expr(exponent, var), 2)
        entry = entry * (even + _cyclo_expr(base) * (1 - even))
    for v in reversed(path):
        entry = sp.Sum(entry, (var[v], 0, 1))
    try:
        return SymbolicTensor(axes, Scalar(entry).simplify(max_steps=max_steps))
    except ScalarBudgetError:
        return None


def _sum_out(
    head: sp.Expr,
    parts: Sequence[sp.Expr],
    summed: Sequence[tuple[sp.Symbol, Dim]],
    max_steps: int,
    boundary: Mapping[str, Dim] | None = None,
) -> Scalar:
    """``head`` times the product of ``parts`` summed over ``summed``, simplified: at concrete
    dimensions one index at a time over only the parts it occurs in, the index whose parts
    carry the fewest other summed indices first, each partial sum simplified with the
    ``boundary`` indices and the summed indices still open sized; otherwise one nested sum
    simplified once."""
    if not all(dim.is_concrete for _symbol, dim in summed):
        entry = head * sp.Mul(*parts)
        for symbol, dim in reversed(summed):
            entry = sp.Sum(entry, (symbol, 0, dim.to_sympy() - 1))
        return Scalar(entry).simplify(max_steps=max_steps)
    dims = dict(summed)
    live: dict[int, tuple[sp.Expr, frozenset[sp.Symbol]]] = {}
    where: dict[sp.Symbol, set[int]] = {symbol: set() for symbol in dims}
    serial = 0

    def add(expr: sp.Expr) -> None:
        nonlocal serial
        symbols = frozenset(s for s in expr.free_symbols if s in dims)
        live[serial] = (expr, symbols)
        for symbol in symbols:
            where[symbol].add(serial)
        serial += 1

    for part in parts:
        add(part)

    def width(symbol: sp.Symbol) -> int:
        return len(frozenset().union(*(live[i][1] for i in where[symbol])))

    widths = {symbol: width(symbol) for symbol in dims}
    deferred: set[sp.Symbol] = set()
    while widths:
        ready = [s for s in widths if s not in deferred] or list(widths)
        symbol = min(ready, key=lambda s: (widths[s], str(s.name)))
        final = symbol in deferred
        bucket = sorted(where[symbol])
        touched: frozenset[sp.Symbol] = frozenset().union(*(live[i][1] for i in bucket))
        top = dims[symbol].to_sympy() - 1
        if not bucket:
            del widths[symbol]
            where.pop(symbol)
            head = head * (top + 1)
            continue
        open_ranges = {**(boundary or {}), **{str(s.name): dims[s] for s in touched - {symbol}}}
        product = sp.Mul(*(live[i][0] for i in bucket))
        closed = (
            Scalar(sp.Sum(product, (symbol, 0, top)))
            .simplify(max_steps=max_steps, ranges=open_ranges, enumerate_concrete=final)
            .to_sympy()
        )
        if not final and len(closed.atoms(sp.Sum)) > len(product.atoms(sp.Sum)):
            deferred.add(symbol)
            continue
        del widths[symbol]
        where.pop(symbol)
        deferred -= touched
        for i in bucket:
            _expr, symbols = live.pop(i)
            for other in symbols - {symbol}:
                where[other].discard(i)
        add(closed)
        for other in touched - {symbol}:
            widths[other] = width(other)
    return Scalar(head * sp.Mul(*(expr for expr, _symbols in live.values()))).simplify(
        max_steps=max_steps, ranges=boundary
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


def contract_mixed_symbolic(
    diagram: Diagram, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS
) -> SymbolicTensor:
    """The CPM of a mixed diagram, formal in every dimension and multiplicity: the symbolic
    contraction of its doubled diagram, axes plain outputs, conjugate outputs, plain inputs,
    conjugate inputs."""
    from archytaszx.diagram.mixed import double  # local: mixed imports bangbox

    return contract_symbolic(double(diagram), max_steps=max_steps)


def contract_symbolic_pair(
    left: Diagram, right: Diagram, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS
) -> tuple[SymbolicTensor, SymbolicTensor]:
    """The symbolic contractions of ``left`` and ``right``, of both doubled diagrams when either
    has a discard."""
    from archytaszx.diagram.mixed import double, is_pure  # local: mixed imports bangbox

    if not (is_pure(left) and is_pure(right)):
        left, right = double(left), double(right)
    return (
        contract_symbolic(left, max_steps=max_steps),
        contract_symbolic(right, max_steps=max_steps),
    )
