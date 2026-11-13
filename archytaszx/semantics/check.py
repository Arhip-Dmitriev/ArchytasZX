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

"""Oracle equality check: instantiate symbols, contract concretely, and compare exactly.

The top-level entry point of the Phase 4 oracle: :func:`score` denotes what one diagram
means at a concrete symbol assignment, :func:`compare` decides whether two mean the same
thing at a shared one. Everything builds on :mod:`archytaszx.semantics.contract_numeric`.

Instantiation. :func:`instantiate` substitutes every dimension, phase, and scalar symbol
via the node-id-preserving :meth:`~archytaszx.diagram.graph.Diagram.substitute`. An unsupplied
symbol falls back to its :attr:`~archytaszx.diagram.graph.Diagram.parameters` binding and a
supplied value overrides that binding, so the oracle can spot-check a diagram at a value
other than the one its user typed. Only a symbol neither source supplies is refused, never
defaulted.

Comparison modes. ``EXACT`` is the default everywhere; ``UP_TO_GLOBAL_PHASE`` and
``UP_TO_POSITIVE_SCALAR`` are opt-in, never inferred. ``EXACT`` requires matching shapes and
entrywise agreement within ``tolerance``, including the overall scalar.
``UP_TO_GLOBAL_PHASE`` asks whether some unit-modulus ``lambda`` has ``b == lambda * a``;
``lambda`` is recovered from ``a``'s largest-magnitude entry and then verified against the
whole tensor. A recovered ``lambda`` with ``|lambda| != 1`` is a non-match: a rescaling by
2 is not a global phase, and this mode is not an up-to-scale escape hatch. All-zero tensors
match each other; one zero and one nonzero never match. ``UP_TO_POSITIVE_SCALAR`` recovers
``lambda`` the same way and requires it to be a positive real.

Mixed operands. :func:`compare` on a diagram with a discard compares the doubled diagrams of
both sides; there ``UP_TO_GLOBAL_PHASE`` is run as ``EXACT``, the two coinciding on CPMs.

Tolerance is a single explicit parameter defaulting to ``1e-9``, an absolute entrywise
bound, with no path that silently loosens it.

:class:`ComparisonResult` carries the mode, a matched flag, a reason, the max absolute
deviation observed, and the recovered ``lambda``.

Interface check, before tensors are compared. :func:`compare_tensors` sees bare arrays, so
it can do no better than ``a.shape == b.shape``. :func:`compare` therefore first checks
that the interfaces correspond: the same number of boundary outputs, the same number of
boundary inputs, and the same dimension per axis. A mismatch is reported as its own result
rather than as a numeric deviation.

What this cannot do. Genuine leg correspondence -- that boundary axis ``i`` of A and axis
``j`` of B denote the same leg, not merely the same dimension -- is not establishable here:
two diagrams denoting the same map may distribute boundary legs over an entirely different
node structure. A silently reordered boundary is therefore invisible whenever the tensors
agree regardless of order. Real correspondence needs a rule to declare the map between its
input and output boundaries and the engine to assert it, feeding Phase 6.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import (
    DEFAULT_MAX_SIMPLIFY_STEPS,
    ModDelta,
    ModGcd,
    ModResidue,
    Scalar,
    ScalarError,
)
from archytaszx.diagram.bangbox import (
    BangBoxDomainError,
    expand_concrete_boxes,
    free_mult_symbols,
    index_symbols,
    instantiate_symbol,
)
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractionResult, contract
from archytaszx.semantics.contract_symbolic import (
    LayoutSlot,
    ReplicatedBlock,
    ReplicatedFactor,
    SymbolicAxis,
    SymbolicContractionError,
    SymbolicTensor,
)

CheckAssignmentValue: TypeAlias = "int | sp.Rational"
DEFAULT_TOLERANCE = 1e-9
"""The default absolute entrywise tolerance used throughout this module. See the module
docstring."""


class CheckError(Exception):
    """Base class for all errors raised by this module."""


class CheckDomainError(CheckError):
    """The domain half of this module's domain/grammar error split. Currently unraised.

    Every refusal this module makes today is a malformed request (an incomplete assignment,
    an unknown mode), so it raises :class:`CheckGrammarError`; a non-concrete value reaches
    :mod:`archytaszx.semantics.contract_numeric` and is refused there instead. Declared so the
    split matches every other module here, and so a future domain refusal has its class
    already in the hierarchy.
    """


class CheckGrammarError(CheckError):
    """A request is malformed: an unknown equality mode, or an incomplete symbol assignment.

    Raised when an ``assignment`` leaves a diagram symbol uninstantiated -- this module
    never defaults a missing symbol -- and when an unrecognized :class:`EqualityMode` is
    passed to :func:`compare_tensors`.
    """


def _diagram_free_symbols(diagram: Diagram) -> frozenset[str]:
    """Every free symbol (dimension, phase, scalar, copy scalar, or bang-box multiplicity, Phase 7)
    appearing anywhere in ``diagram``."""
    symbols: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            symbols |= port.dim.free_symbols
        if node.phase is not None:
            symbols |= node.phase.free_symbols
    for box in diagram.bang_boxes.values():
        if box.copy_scalar is not None:
            symbols |= box.copy_scalar.free_symbols
    symbols |= free_mult_symbols(diagram)
    return frozenset(symbols - index_symbols(diagram))


def _expand_bang_boxes(diagram: Diagram, resolved: Mapping[str, CheckAssignmentValue]) -> Diagram:
    """Instantiate every bang-box multiplicity symbol named in ``resolved``, in turn, then
    expand every box whose multiplicity is concrete.

    Must run before :meth:`~archytaszx.diagram.graph.Diagram.substitute`: expansion can build
    fresh port/node structure whose dimensions still carry the very symbols ``resolved``
    is about to substitute, so dimension/phase/scalar substitution runs once, last, over
    the fully box-free result (Phase 7).

    A name whose boxes an earlier instantiation already killed is skipped: a multiplicity
    of 0 removes its box's children, leaving their symbols with nothing to expand.
    """
    working = diagram
    for name in sorted(free_mult_symbols(diagram)):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BangBoxDomainError(
                f"multiplicity symbol {name!r} requires a non-negative int, got {value!r}"
            )
        if name not in free_mult_symbols(working):
            continue
        working = instantiate_symbol(working, name, value)
    return expand_concrete_boxes(working)


def instantiate(diagram: Diagram, assignment: Mapping[str, CheckAssignmentValue]) -> Diagram:
    """Substitute every symbol in ``diagram`` per ``assignment``, falling back to
    ``diagram.parameters``.

    A symbol ``assignment`` does not mention takes its value from the diagram's parameter
    environment; a symbol ``assignment`` does mention takes that value, overriding the
    environment. A symbol neither supplies raises CheckGrammarError -- never defaulted. A
    key naming no symbol in this diagram is dropped: :func:`compare` and the sweeps pass one
    shared assignment to two diagrams, and a rewrite can eliminate a symbol from one side (a
    phase binding the shared leg dimension leaves the merged node concrete). See the module
    docstring. Every bang-box multiplicity symbol (Phase 7) is instantiated the same way,
    before the ordinary dimension/phase/scalar substitution runs -- see
    :func:`_expand_bang_boxes`.
    """
    free_symbols = _diagram_free_symbols(diagram)
    resolved: dict[str, CheckAssignmentValue] = {
        name: value for name, value in diagram.parameters.items() if name in free_symbols
    }
    resolved.update({name: value for name, value in assignment.items() if name in free_symbols})
    missing = free_symbols - set(resolved)
    if missing:
        raise CheckGrammarError(
            f"assignment leaves symbol(s) uninstantiated: {sorted(missing)}; neither the "
            "assignment nor the diagram's parameter environment supplies them, and "
            "instantiate() never defaults a missing symbol"
        )
    expanded = _expand_bang_boxes(diagram, resolved)
    mult_names = free_mult_symbols(diagram)
    still_free = _diagram_free_symbols(expanded)
    remaining = {
        name: value
        for name, value in resolved.items()
        if name not in mult_names or name in still_free
    }
    return expanded.substitute(remaining)


def score(
    diagram: Diagram,
    assignment: Mapping[str, CheckAssignmentValue],
    *,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ContractionResult:
    """Instantiate ``diagram`` at ``assignment`` and contract it. The oracle's "evaluate"
    entry point."""
    instantiated = instantiate(diagram, assignment)
    return contract(instantiated, max_elements=max_elements)


class EqualityMode(enum.Enum):
    """How two contracted tensors are compared. See the module docstring."""

    EXACT = "exact"
    UP_TO_GLOBAL_PHASE = "up_to_global_phase"
    UP_TO_POSITIVE_SCALAR = "up_to_positive_scalar"


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of one :func:`compare_tensors` or :func:`compare` call. See the module
    docstring."""

    mode: EqualityMode
    matched: bool
    reason: str
    max_abs_deviation: float
    recovered_lambda: complex | None = None
    doubled: bool = False
    difference: Scalar | None = None
    """The nonzero entry difference of a symbolic comparison that found one."""
    index: tuple[int, ...] | None = None
    """The index tuple (outputs, then inputs) of the largest entrywise deviation, when an
    EXACT dense comparison failed."""


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ComparisonResult:
    """Compare two already-contracted tensors under ``mode``. ``mode`` defaults to EXACT.

    See the module docstring for the exact contract of each mode and for the tolerance
    default.
    """
    if a.shape != b.shape:
        return ComparisonResult(
            mode=mode,
            matched=False,
            reason=f"shape mismatch: {a.shape} vs {b.shape}",
            max_abs_deviation=float("inf"),
        )

    if mode is EqualityMode.EXACT:
        gap = np.abs(a - b)
        deviation = float(np.max(gap)) if a.size else 0.0
        matched = deviation <= tolerance
        worst = (
            None
            if matched or not a.size
            else tuple(int(v) for v in np.unravel_index(int(np.argmax(gap)), gap.shape))
        )
        reason = (
            "tensors agree entrywise within tolerance"
            if matched
            else f"max abs deviation {deviation} exceeds tolerance {tolerance}"
        )
        return ComparisonResult(
            mode=mode, matched=matched, reason=reason, max_abs_deviation=deviation, index=worst
        )

    if mode in (EqualityMode.UP_TO_GLOBAL_PHASE, EqualityMode.UP_TO_POSITIVE_SCALAR):
        return _compare_up_to_global_phase(a, b, tolerance=tolerance, mode=mode)

    raise CheckGrammarError(f"unknown equality mode {mode!r}")


def _compare_up_to_global_phase(
    a: np.ndarray,
    b: np.ndarray,
    *,
    tolerance: float,
    mode: EqualityMode = EqualityMode.UP_TO_GLOBAL_PHASE,
) -> ComparisonResult:
    """``b == lambda a`` with ``|lambda| = 1`` (global phase) or ``lambda > 0`` (positive
    scalar), ``lambda`` recovered at ``a``'s largest entry."""
    if a.size == 0:
        return ComparisonResult(
            mode=mode, matched=True, reason="both tensors are empty", max_abs_deviation=0.0
        )

    norm_a = float(np.max(np.abs(a)))
    norm_b = float(np.max(np.abs(b)))
    a_zero = norm_a <= tolerance
    b_zero = norm_b <= tolerance
    if a_zero and b_zero:
        return ComparisonResult(
            mode=mode, matched=True, reason="both tensors are zero", max_abs_deviation=0.0
        )
    if a_zero or b_zero:
        return ComparisonResult(
            mode=mode,
            matched=False,
            reason="one tensor is zero and the other is not",
            max_abs_deviation=float("inf"),
        )

    flat_index = int(np.argmax(np.abs(a)))
    idx = np.unravel_index(flat_index, a.shape)
    a_entry = complex(a[idx])
    b_entry = complex(b[idx])
    lam = b_entry / a_entry
    deviation = float(np.max(np.abs(b - lam * a)))

    if mode is EqualityMode.UP_TO_POSITIVE_SCALAR:
        if abs(lam.imag) > tolerance or lam.real <= tolerance:
            return ComparisonResult(
                mode=mode,
                matched=False,
                reason=f"recovered factor {lam} is not a positive real",
                max_abs_deviation=deviation,
                recovered_lambda=lam,
            )
        matched = deviation <= tolerance
        reason = (
            f"tensors agree up to the positive scalar {lam.real}"
            if matched
            else (
                f"max abs deviation {deviation} after removing the scalar {lam.real} exceeds "
                f"tolerance {tolerance}"
            )
        )
        return ComparisonResult(
            mode=mode,
            matched=matched,
            reason=reason,
            max_abs_deviation=deviation,
            recovered_lambda=lam,
        )

    if abs(abs(lam) - 1.0) > tolerance:
        return ComparisonResult(
            mode=mode,
            matched=False,
            reason=(
                f"recovered factor {lam} has magnitude {abs(lam)}, not unit modulus; "
                "a rescaling is not a global phase"
            ),
            max_abs_deviation=deviation,
            recovered_lambda=lam,
        )

    matched = deviation <= tolerance
    reason = (
        f"tensors agree up to global phase {lam}"
        if matched
        else (
            f"max abs deviation {deviation} after removing global phase {lam} exceeds "
            f"tolerance {tolerance}"
        )
    )
    return ComparisonResult(
        mode=mode, matched=matched, reason=reason, max_abs_deviation=deviation, recovered_lambda=lam
    )


def _axis_dimensions(diagram: Diagram, result: ContractionResult) -> tuple[int, ...]:
    """The concrete dimension of each axis in ``result``, in ``axis_refs`` order.

    Resolves each :class:`~archytaszx.diagram.graph.PortRef` against ``diagram``, following the
    same ``node.legs(ref.direction)[ref.index]`` pattern
    :mod:`archytaszx.semantics.contract_numeric` uses for its own output-size guard, rather than
    inventing a second way to resolve a ``PortRef`` against its node.
    """
    dimensions = []
    for ref in result.axis_refs:
        node = diagram.nodes[ref.node_id]
        port = node.legs(ref.direction)[ref.index]
        dimensions.append(port.dim.to_int())
    return tuple(dimensions)


def _interface_mismatch(
    diagram_a: Diagram,
    result_a: ContractionResult,
    diagram_b: Diagram,
    result_b: ContractionResult,
    *,
    mode: EqualityMode,
) -> ComparisonResult | None:
    """Check that two contractions' boundaries correspond, or explain why they don't.

    Returns ``None`` when the interfaces correspond; otherwise a non-matching
    :class:`ComparisonResult` naming the mismatch, so a caller never mistakes an interface
    problem for a numeric deviation. Two things are checked, in order:

    1. The boundary output/input split (``result.num_boundary_outputs`` and the
       remainder) -- a diagram with 2 outputs and 1 input must never match one with 1
       output and 2 inputs merely on both having three free legs.
    2. Per axis, the concrete dimension of the port that produced it, position by
       position.

    This module has no way to establish genuine leg correspondence beyond that -- see the
    module docstring's "What this cannot do" paragraph. In particular this check
    does not catch a silently reordered boundary when the reordering leaves the tensor
    unchanged (e.g. any diagram symmetric under that swap); that is out of scope for a
    numeric oracle and belongs to the rewrite engine's own certificates instead.
    """
    outputs_a = result_a.num_boundary_outputs
    outputs_b = result_b.num_boundary_outputs
    inputs_a = len(result_a.axis_refs) - outputs_a
    inputs_b = len(result_b.axis_refs) - outputs_b
    if outputs_a != outputs_b or inputs_a != inputs_b:
        return ComparisonResult(
            mode=mode,
            matched=False,
            reason=(
                f"boundary arity mismatch: {outputs_a} output(s)/{inputs_a} input(s) vs "
                f"{outputs_b} output(s)/{inputs_b} input(s)"
            ),
            max_abs_deviation=float("inf"),
        )

    dims_a = _axis_dimensions(diagram_a, result_a)
    dims_b = _axis_dimensions(diagram_b, result_b)
    if dims_a != dims_b:
        return ComparisonResult(
            mode=mode,
            matched=False,
            reason=f"boundary axis dimension mismatch: {dims_a} vs {dims_b}",
            max_abs_deviation=float("inf"),
        )

    return None


def compare(
    diagram_a: Diagram,
    diagram_b: Diagram,
    assignment: Mapping[str, CheckAssignmentValue],
    *,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ComparisonResult:
    """Instantiate both diagrams at the shared ``assignment``, contract, and compare.

    The oracle's "are these equal" entry point. ``mode`` defaults to EXACT, per the
    module docstring's standing rule that up-to-global-phase comparison is opt-in only.
    Before the contracted tensors are compared, the two contractions' interfaces are
    checked to correspond (same boundary output/input split, same per-axis dimensions);
    an interface mismatch is reported as its own non-match result rather than surfacing as
    a numeric deviation. See the module docstring.
    """
    from archytaszx.diagram.mixed import double, is_pure  # local: mixed imports bangbox

    if not (is_pure(diagram_a) and is_pure(diagram_b)):
        exact = mode is EqualityMode.UP_TO_GLOBAL_PHASE
        result = compare(
            double(instantiate(diagram_a, assignment)),
            double(instantiate(diagram_b, assignment)),
            {},
            mode=EqualityMode.EXACT if exact else mode,
            tolerance=tolerance,
            max_elements=max_elements,
        )
        note = "; on CPMs UP_TO_GLOBAL_PHASE coincides with EXACT" if exact else ""
        return replace(result, reason=f"doubled: {result.reason}{note}", doubled=True)
    instantiated_a = instantiate(diagram_a, assignment)
    instantiated_b = instantiate(diagram_b, assignment)
    result_a = contract(instantiated_a, max_elements=max_elements)
    result_b = contract(instantiated_b, max_elements=max_elements)

    interface_mismatch = _interface_mismatch(
        instantiated_a, result_a, instantiated_b, result_b, mode=mode
    )
    if interface_mismatch is not None:
        return interface_mismatch

    return compare_tensors(result_a.tensor, result_b.tensor, mode=mode, tolerance=tolerance)


def _group_ranges(t: SymbolicTensor, group: int) -> dict[str, Dim]:
    """The dimension of each template index of ``group``'s top-level blocks."""
    out: dict[str, Dim] = {}
    for slot in t.slots():
        if isinstance(slot, ReplicatedBlock) and slot.group == group:
            for axis in slot.axes:
                if isinstance(axis, SymbolicAxis):
                    out[axis.index] = axis.dim
    return out


def _by_name(expr: sp.Expr, mapping: Mapping[str, sp.Expr]) -> sp.Expr:
    """``expr`` with each free symbol whose name ``mapping`` holds replaced."""
    return expr.xreplace(
        {x: mapping[str(x.name)] for x in expr.free_symbols if str(x.name) in mapping}
    )


def _split_parts(
    t: SymbolicTensor, group: ReplicatedFactor, positions: list[int]
) -> list[sp.Expr] | None:
    """One factor per block of a group laid out at ``positions``, or None when its factor,
    after substituting each template index a delta pins, couples two blocks."""
    from archytaszx.semantics.contract_symbolic import _periodic, _pins

    assert t.layout is not None
    block_of: dict[str, int] = {}
    for k, position in enumerate(positions):
        for axis in cast(ReplicatedBlock, t.layout[position]).axes:
            if not isinstance(axis, SymbolicAxis):
                return None
            block_of[axis.index] = k
    args = list(sp.Mul.make_args(group.factor.to_sympy()))
    for i, arg in enumerate(args):
        if not isinstance(arg, ModDelta):
            continue
        modulus = arg.args[1]
        for symbol in sorted(arg.free_symbols, key=lambda x: str(x.name)):
            if str(symbol.name) not in block_of:
                continue
            pins = [
                v
                for v in _pins(arg, symbol, modulus)
                if not any(str(x.name) in block_of for x in v.free_symbols)
            ]
            others = args[:i] + args[i + 1 :]
            if pins and all(_periodic(o, symbol, modulus) for o in others):
                args = [a if j == i else a.xreplace({symbol: pins[0]}) for j, a in enumerate(args)]
                break
    per_block: list[list[sp.Expr]] = [[] for _ in positions]
    for arg in args:
        touched = {block_of[str(x.name)] for x in arg.free_symbols if str(x.name) in block_of}
        if len(touched) > 1:
            return None
        per_block[min(touched) if touched else 0].append(arg)
    return [sp.Mul(*part) for part in per_block]


def _split_groups(t: SymbolicTensor, max_steps: int) -> SymbolicTensor:
    """``t`` with each plain group laid out in several top-level blocks split into one group
    per block when its factor separates by block (:func:`_split_parts`); groups are numbered by
    first appearance in the layout, template indices renamed ``_t<g>_<k>`` to match."""
    if not t.groups or t.layout is None:
        return t
    blocks: dict[int, list[int]] = {}
    for position, slot in enumerate(t.layout):
        if isinstance(slot, ReplicatedBlock):
            blocks.setdefault(slot.group, []).append(position)
    split: dict[int, list[sp.Expr]] = {}
    for g, positions in blocks.items():
        group = t.groups[g]
        plain = group.marked is None and group.inner is None and group.pairs is None
        if len(positions) > 1 and plain:
            found = _split_parts(t, group, positions)
            if found is not None:
                split[g] = found
    if not split:
        return t
    groups: list[ReplicatedFactor] = []
    layout: list[LayoutSlot] = []
    seen: dict[int, int] = {}
    for position, slot in enumerate(t.layout):
        if not isinstance(slot, ReplicatedBlock):
            layout.append(slot)
            continue
        old = t.groups[slot.group]
        if slot.group in split:
            factor = split[slot.group][blocks[slot.group].index(position)]
        elif slot.group in seen:
            layout.append(ReplicatedBlock(seen[slot.group], slot.axes))
            continue
        else:
            factor = old.factor.to_sympy()
        g_new = len(groups)
        if slot.group not in split:
            seen[slot.group] = g_new
        rename: dict[str, sp.Expr] = {}
        axes: list[LayoutSlot] = []
        for k, axis in enumerate(slot.axes):
            if not isinstance(axis, SymbolicAxis):
                return t
            name = f"_t{g_new}_{k}"
            rename[axis.index] = sp.Symbol(name, integer=True, nonnegative=True)
            axes.append(SymbolicAxis(axis.port, axis.dim, name))
        groups.append(
            replace(old, factor=Scalar(_by_name(factor, rename)).simplify(max_steps=max_steps))
        )
        layout.append(ReplicatedBlock(g_new, tuple(axes)))
    return replace(t, layout=tuple(layout), groups=tuple(groups))


def _normalise_groups(t: SymbolicTensor, max_steps: int) -> SymbolicTensor:
    """``t`` with each plain group's index-free factor constant moved into the entry as its
    power by the multiplicity."""
    if not t.groups:
        return t
    shared = {index.name for index in t.shared}
    entry = t.entry.to_sympy()
    groups: list[ReplicatedFactor] = []
    moved = False
    for g, group in enumerate(t.groups):
        if group.marked is not None or group.inner is not None:
            groups.append(group)
            continue
        names = set(_group_ranges(t, g)) | shared
        prefix = f"_t{g}_"
        constant: list[sp.Expr] = []
        rest: list[sp.Expr] = []
        for arg in sp.Mul.make_args(group.factor.to_sympy()):
            symbols = {str(symbol.name) for symbol in arg.free_symbols}
            bound = symbols & names or any(name.startswith(prefix) for name in symbols)
            (rest if bound else constant).append(arg)
        product: sp.Expr | None = sp.Mul(*constant) ** group.multiplicity.to_sympy()
        if group.index is not None and constant:
            from archytaszx.semantics.contract_symbolic import _COPY, _closed_product

            at = {x: _COPY for x in sp.Mul(*constant).free_symbols if str(x) == group.index}
            product = _closed_product(sp.Mul(*constant).xreplace(at), group.multiplicity.to_sympy())
            if product is None:
                rest, constant = rest + constant, []
        if not constant or product is None:
            groups.append(group)
            continue
        moved = True
        entry = entry * product
        factor = Scalar(sp.Mul(*rest)).simplify(max_steps=max_steps)
        groups.append(replace(group, factor=factor))
    if not moved:
        return t
    simplified = Scalar(entry).simplify(max_steps=max_steps, ranges=t.ranges())
    return replace(t, entry=simplified, groups=tuple(groups))


def _same_groups(left: SymbolicTensor, right: SymbolicTensor, max_steps: int) -> bool:
    """Whether every group of ``left`` and ``right`` has the same multiplicity, mark, inner
    tensor and, syntactically or after simplifying the difference, factor."""
    if len(left.groups) != len(right.groups):
        return False
    for g, (a, b) in enumerate(zip(left.groups, right.groups, strict=True)):
        if (a.multiplicity, a.marked, a.inner, a.index) != (
            b.multiplicity,
            b.marked,
            b.inner,
            b.index,
        ):
            return False
        if a.pairs != b.pairs:
            if a.pairs is None or b.pairs is None:
                return False
            if not (a.pairs - b.pairs).simplify(max_steps=max_steps).is_zero:
                return False
        if a.factor == b.factor:
            continue
        if a.marked is not None or a.inner is not None:
            return False
        ranges = {**_group_ranges(left, g), **left.ranges()}
        if not (a.factor - b.factor).simplify(max_steps=max_steps, ranges=ranges).is_zero:
            return False
    return True


def compare_symbolic(
    left: SymbolicTensor,
    right: SymbolicTensor,
    *,
    mode: EqualityMode = EqualityMode.EXACT,
    max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS,
    dimension_floors: Mapping[str, int] | None = None,
) -> ComparisonResult:
    """Compare two symbolic tensors by simplifying the difference of their entries.

    Three outcomes, not two: equal, definitely unequal, and indeterminate -- the difference
    still carries an index sum whose character-sum verdict was undecidable. Variable-rank
    tensors match only with equal layouts, shared indices and replicated factors. An indeterminate
    result reports ``matched=False`` with a reason naming the residual sum, so a caller that
    treats "not matched" as "unequal" is wrong and must read ``reason``. ``dimension_floors``
    gives a least value per dimension symbol, applied through
    :meth:`~archytaszx.algebra.scalar.Scalar.with_dimension_floors`.
    """
    if mode is not EqualityMode.EXACT:
        raise CheckDomainError(
            "up-to-global-phase comparison requires concrete entries; substitute the "
            "parameter environment first"
        )
    if left.is_variable_rank or right.is_variable_rank:
        if left.signature() != right.signature() or left.shared != right.shared:
            return ComparisonResult(
                mode,
                False,
                "indeterminate: the variable-rank layouts or shared indices differ",
                float("inf"),
            )
        left = _normalise_groups(_split_groups(left, max_steps), max_steps)
        right = _normalise_groups(_split_groups(right, max_steps), max_steps)
        if not _same_groups(left, right, max_steps):
            return ComparisonResult(
                mode, False, "indeterminate: the replicated factors differ", float("inf")
            )
    if left.rank != right.rank:
        return ComparisonResult(mode, False, f"rank {left.rank} != rank {right.rank}", float("inf"))
    if left.dims() != right.dims():
        return ComparisonResult(
            mode, False, f"axis dimensions {left.dims()} != {right.dims()}", float("inf")
        )
    ranges = left.ranges() if left.ranges() == right.ranges() else None
    difference = (left.entry - right.entry).simplify(max_steps=max_steps, ranges=ranges)
    if difference.is_zero or _vanishes(difference.to_sympy(), ranges):
        return ComparisonResult(mode, True, "entries are exactly equal with d formal", 0.0)
    if dimension_floors:
        difference = difference.with_dimension_floors(dimension_floors).simplify(
            max_steps=max_steps, ranges=ranges
        )
        if difference.is_zero:
            floors = ", ".join(f"{name} >= {low}" for name, low in sorted(dimension_floors.items()))
            return ComparisonResult(
                mode, True, f"entries are exactly equal with d formal, given {floors}", 0.0
            )
    if left.is_variable_rank:
        return ComparisonResult(
            mode,
            False,
            f"indeterminate: the entries differ by {difference} under shared replicated factors",
            float("inf"),
            difference=difference,
        )
    if difference.to_sympy().atoms(sp.Sum, ModDelta, ModGcd, ModResidue):
        return ComparisonResult(
            mode,
            False,
            f"indeterminate: the difference {difference} still carries an unevaluated "
            "index sum or divisibility atom, so no conclusion is drawn",
            float("inf"),
        )
    return ComparisonResult(
        mode, False, f"entries differ by {difference}", float("inf"), difference=difference
    )


@dataclass(frozen=True, slots=True)
class EntryWitness:
    """One symbol assignment and boundary index tuple, in axis order, at which two symbolic
    tensors' entries differ, with both entries."""

    assignment: Mapping[str, CheckAssignmentValue]
    index: tuple[int, ...]
    left: complex
    right: complex

    @property
    def deviation(self) -> float:
        """The modulus of the entries' difference."""
        return abs(self.left - self.right)


_MAX_CASES = 4096


def _vanishes(expr: sp.Expr, ranges: Mapping[str, Dim] | None = None, cases: int = 1) -> bool:
    """Whether ``expr`` is zero once powers are written through exp, exponentials split and
    reduced mod one turn, every complete set of residue deltas mod a concrete modulus with one
    coefficient is replaced by that coefficient, or, over one symbol, by :func:`_uw_zero`, and,
    failing that, in each case of an index whose concrete range equals the modulus of a delta it
    appears in alone."""
    if expr.atoms(sp.Sum):
        return False
    single = {
        s: sp.Integer(0)
        for s in expr.free_symbols
        if (dim := (ranges or {}).get(str(s.name))) is not None
        and dim.is_concrete
        and dim.to_int() == 1
    }
    if single:
        expr = sp.expand(expr.xreplace(single))
    expr = _square_moduli(_constant_bases(expr))
    folded = _fold_residues(_reduce_exponents(sp.expand(expr.rewrite(sp.exp), power_exp=True)))
    folded = sp.expand(folded)
    if folded == 0:
        return True
    if not folded.free_symbols:
        return _cyclotomic_zero(folded)
    if _grouped_zero(folded):
        return True
    if len(folded.free_symbols) == 1 and not folded.atoms(ModDelta) and _uw_zero(folded):
        return True
    split = _case_index(folded, ranges or {})
    if split is None:
        return False
    symbol, size = split
    if cases * size > _MAX_CASES:
        return False
    return all(
        _vanishes(folded.xreplace({symbol: sp.Integer(v)}), ranges, cases * size)
        for v in range(size)
    )


def _grouped_zero(expr: sp.Expr) -> bool:
    """Whether the terms of ``expr``, grouped by their symbol-dependent factor, have constant
    coefficients each summing to zero (:func:`_cyclotomic_zero`)."""
    symbols = tuple(expr.free_symbols)
    groups: dict[sp.Expr, list[sp.Expr]] = {}
    for term in sp.Add.make_args(expr):
        coefficient, dependent = term.as_independent(*symbols, as_Add=False)
        groups.setdefault(dependent, []).append(coefficient)
    if len(groups) == len(sp.Add.make_args(expr)):
        return False
    return all(_cyclotomic_zero(sp.Add(*coefficients)) for coefficients in groups.values())


def _cyclotomic_zero(expr: sp.Expr) -> bool:
    """Whether the constant ``expr``, read exactly as a cyclotomic number times a power of
    sqrt 2, is zero; False when it is not one."""
    from archytaszx.algebra.pathsum import PathSumUnsupported, constant, with_sqrt2

    try:
        value, half_powers = constant(expr)
    except (PathSumUnsupported, ValueError, TypeError, ZeroDivisionError):
        return False
    return with_sqrt2(value, half_powers).is_zero


def _constant_bases(expr: sp.Expr) -> sp.Expr:
    """``expr`` with the base of each power to a symbolic exponent, when the base is a constant,
    written in its simplified exact form."""

    def simple(base: sp.Expr) -> sp.Expr:
        return sp.radsimp(sp.expand_complex(base.rewrite(sp.exp)))

    return expr.replace(
        lambda e: (
            isinstance(e, sp.Pow)
            and not e.base.free_symbols
            and bool(e.exp.free_symbols)
            and not e.base.is_Rational
        ),
        lambda e: sp.Pow(simple(e.base), e.exp),
    )


def _square_moduli(expr: sp.Expr) -> sp.Expr:
    """``expr`` with each ``sqrt(X) conjugate(sqrt(X))`` replaced by ``X`` when ``X`` is real
    (:func:`_is_real`) and bounded below by zero (:func:`_lower_bound`)."""

    def fold(term: sp.Expr) -> sp.Expr:
        factors = list(sp.Mul.make_args(term))
        for i, f in enumerate(factors):
            if not (isinstance(f, sp.Pow) and f.exp == sp.Rational(1, 2)):
                continue
            partner = sp.conjugate(f)
            if partner not in factors:
                continue
            x = f.base
            bound = _lower_bound(x)
            if (bound is None or not _is_real(x) or not bool(bound >= 0)) and not (
                _real_nonnegative_uw(x)
            ):
                continue
            rest = list(factors)
            rest.pop(i)
            rest.remove(partner)
            return sp.Mul(x, *rest)
        return _fold_product(factors)

    if not expr.atoms(sp.conjugate):
        return expr
    return sp.Add(*(fold(t) for t in sp.Add.make_args(sp.expand(expr, power_exp=False))))


def _nonnegative_real(x: sp.Expr) -> bool:
    """Whether ``x`` is real and at least 0 by :func:`_lower_bound` or
    :func:`_real_nonnegative_uw`."""
    bound = _lower_bound(x)
    if bound is not None and _is_real(x) and bool(bound >= 0):
        return True
    return _real_nonnegative_uw(x)


def _differ_at_samples(a: sp.Expr, b: sp.Expr) -> bool:
    """Whether ``a`` and ``b`` evaluate apart at one of a few integer points of their free
    symbols (30 digits, relative gap above 1e-20); False when an evaluation fails."""
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda s: str(s))
    for value in (1, 2, 3, 5):
        point = {s: sp.Integer(value + i) for i, s in enumerate(symbols)}
        try:
            x, y = sp.N(a.xreplace(point), 30), sp.N(b.xreplace(point), 30)
            gap, size = abs(complex(sp.N(x - y, 30))), abs(complex(x)) + abs(complex(y))
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if gap > 1e-20 * (1 + size):
            return True
    return False


def _fold_product(factors: list[sp.Expr]) -> sp.Expr:
    """The product of ``factors`` with ``sqrt(Z) * conj(sqrt(X1)) * ... * conj(sqrt(Xk))``
    replaced by ``Z`` when ``Z == X1 * ... * Xk`` and every ``Xi`` is real and at least 0."""
    partners = [
        g
        for g in factors
        if isinstance(g, sp.conjugate)
        and isinstance(g.args[0], sp.Pow)
        and g.args[0].exp == sp.Rational(1, 2)
    ]
    if len(partners) < 2:
        return sp.Mul(*factors)
    bases = [g.args[0].base for g in partners]
    for i, f in enumerate(factors):
        if not (isinstance(f, sp.Pow) and f.exp == sp.Rational(1, 2)):
            continue
        product = sp.Mul(*bases)
        if _differ_at_samples(f.base, product):
            continue
        if not _uw_zero(f.base - product) and sp.cancel(sp.together(f.base - product)) != 0:
            continue
        if not all(_nonnegative_real(b) for b in bases):
            continue
        rest = [g for j, g in enumerate(factors) if j != i and g not in partners]
        return sp.Mul(f.base, *rest)
    return sp.Mul(*factors)


def _uw_form(x: sp.Expr) -> tuple[sp.Expr, sp.Symbol, sp.Symbol, int] | None:
    """``x`` over one nonnegative integer symbol ``n`` as a rational function of
    ``U = b^(n/qu)`` and ``W = exp(i pi n/qw)``, with ``2*qw``, when every ``n``-dependence is
    such a power or exponential; None otherwise."""
    symbols = x.free_symbols
    if len(symbols) != 1:
        return None
    (n,) = symbols
    if not (n.is_integer and n.is_nonnegative):
        return None
    exps: dict[sp.Expr, sp.Rational] = {}
    pows: dict[sp.Expr, sp.Rational] = {}
    base: sp.Expr | None = None
    for atom in x.atoms(sp.exp, sp.Pow):
        if n not in atom.free_symbols:
            continue
        if isinstance(atom, sp.exp):
            c = sp.cancel(atom.args[0] / (sp.I * sp.pi * n))
            if c.is_Rational:
                exps[atom] = sp.Rational(c)
                continue
            r, log = sp.cancel(atom.args[0] / n).as_coeff_Mul()
            if not (r.is_Rational and isinstance(log, sp.log) and log.args[0].is_Integer):
                return None
            b = log.args[0]
        else:
            if n not in atom.exp.free_symbols:
                continue
            b, r = atom.base, sp.cancel(atom.exp / n)
        if not (b.is_Integer and b > 1) or not r.is_Rational:
            return None
        if base is not None and b != base:
            return None
        base = b
        pows[atom] = sp.Rational(r)
    qw = math.lcm(*(int(c.q) for c in exps.values())) if exps else 1
    qu = math.lcm(*(int(r.q) for r in pows.values())) if pows else 1
    u = sp.Symbol("_U", positive=True)
    w = sp.Symbol("_W", real=True)
    mapping = {a: w ** int(c * qw) for a, c in exps.items()}
    mapping.update({a: u ** int(r * qu) for a, r in pows.items()})
    form = x.xreplace(mapping)
    if n in form.free_symbols:
        return None
    return form, u, w, 2 * qw


def _uw_zero(x: sp.Expr) -> bool:
    """Whether ``x``, read by :func:`_uw_form` as a rational function of ``U`` and ``W``, has a
    numerator vanishing mod ``W^period - 1``."""
    found = _uw_form(x)
    if found is None:
        return False
    form, _u, w, period = found
    try:
        numerator, _denominator = sp.fraction(sp.together(form))
        reduced = _laurent_mod(sp.expand(numerator), w, period)
    except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
        return False
    return reduced is not None and reduced.is_zero


def _laurent_mod(expr: sp.Expr, w: sp.Symbol, period: int) -> sp.Poly | None:
    """``expr``, a Laurent polynomial in ``w``, reduced mod ``w^period - 1`` as a polynomial in
    ``w``; None when it is not one."""
    expr = sp.expand(expr)
    low = min(int(t.as_powers_dict().get(w, 0)) for t in sp.Add.make_args(expr))
    shift = period * ((-low + period - 1) // period) if low < 0 else 0
    try:
        poly = sp.Poly(sp.expand(expr * w**shift), w)
    except sp.PolynomialError:
        return None
    return sp.Poly(sp.rem(poly.as_expr(), w**period - 1, w), w)


def _real_nonnegative_uw(x: sp.Expr) -> bool:
    """Whether ``x``, read by :func:`_uw_form` as ``P/Q`` in ``U >= 1`` and ``|W| = 1`` with
    ``W^period = 1``, equals its conjugate and has ``P conj(Q)`` bounded below by zero: its top
    ``U`` coefficient's real constant less the other moduli, against every lower coefficient's
    modulus sum, with no root of that bound above ``U = 1``."""
    found = _uw_form(x)
    if found is None:
        return False
    form, u, w, period = found
    try:
        r = sp.cancel(sp.together(form))
        p, q = sp.fraction(r)
        conj_q = sp.conjugate(q).xreplace({w: 1 / w})
        conj_r = sp.conjugate(r).xreplace({w: 1 / w})
        num, _den = sp.fraction(sp.together(r - conj_r))
        reduced = _laurent_mod(num, w, period)
        if reduced is None or not reduced.is_zero:
            return False
        product = _laurent_mod(sp.expand(p * conj_q), w, period)
        if product is None:
            return False
        by_u: dict[int, list[sp.Expr]] = {}
        for (j,), coefficient in product.terms():
            for (k,), c in sp.Poly(coefficient, u).terms():
                by_u.setdefault(k, []).append((j, c))
        if not by_u:
            return True
        top = max(by_u)
        lead = sum((sp.re(c) for j, c in by_u[top] if j == 0), sp.Integer(0)) - sum(
            (sp.Abs(c) for j, c in by_u[top] if j != 0), sp.Integer(0)
        )
        bound = lead * u**top - sum(
            (sp.Abs(c) * u**k for k in by_u if k != top for _j, c in by_u[k]), sp.Integer(0)
        )
        bound = sp.nsimplify(sp.expand(bound))
        if (
            bound.free_symbols <= {u}
            and not sp.simplify(bound.subs(u, 1)) < 0
            and sp.solveset(bound < 0, u, sp.Interval(1, sp.oo)) is sp.S.EmptySet
        ):
            return True
        return _shifted_nonnegative(sp.expand(p * conj_q), u, w, period)
    except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
        return False


def _shifted_nonnegative(product: sp.Expr, u: sp.Symbol, w: sp.Symbol, period: int) -> bool:
    """Whether ``product`` is at least 0 for every ``U >= 1`` at each ``W = exp(2 pi i r/period)``:
    every coefficient of its polynomial in ``t = U - 1`` real and at least 0."""
    for r in range(period):
        value = sp.exp(2 * sp.pi * sp.I * sp.Rational(r, period))
        at = sp.expand(product.xreplace({w: value}).xreplace({u: u + 1}))
        try:
            coefficients = sp.Poly(at, u).all_coeffs()
        except sp.PolynomialError:
            return False
        for c in coefficients:
            v = complex(sp.N(c, 50))
            if abs(v.imag) > 1e-30 and sp.simplify(sp.im(sp.expand_complex(c))) != 0:
                return False
            if v.real < -1e-30:
                return False
            if v.real < 1e-30 and sp.simplify(sp.expand_complex(c)) != 0:
                return False
    return True


def _is_real(x: sp.Expr) -> bool:
    """Whether ``x`` equals its conjugate, exponents over integer symbols reduced mod 1."""
    difference = sp.expand(sp.expand(x - sp.conjugate(x)).rewrite(sp.exp), power_exp=True)
    return bool(sp.expand(_reduce_exponents(difference)) == 0)


def _unimodular(f: sp.Expr) -> bool:
    """Whether ``f`` is ``exp(i r)`` or ``(-1)^q``, r real and q rational."""
    if isinstance(f, sp.exp):
        arg = f.args[0]
        return bool((arg / sp.I).is_real)
    return bool(isinstance(f, sp.Pow) and f.base == -1 and f.exp.is_Rational)


def _modulus_sup(c: sp.Expr) -> sp.Expr | None:
    """The supremum over non-negative integer symbols of a positive modulus ``c`` built from
    constants and powers ``b^e`` (b > 1, e linear with non-positive coefficients, or b < 1
    with non-negative ones): ``c`` at every symbol 0; None for any other shape."""
    for f in sp.Mul.make_args(c):
        if not f.free_symbols:
            if not (f.is_real and f.is_positive):
                return None
            continue
        if not isinstance(f, sp.Pow) or f.base.free_symbols or not f.base.is_positive:
            return None
        try:
            poly = sp.Poly(f.exp, *sorted(f.exp.free_symbols, key=str))
        except sp.PolynomialError:
            return None
        if poly.total_degree() > 1:
            return None
        if not all(x.is_integer and x.is_nonnegative for x in f.exp.free_symbols):
            return None
        coefficients = [poly.coeff_monomial(g) for g in poly.gens]
        growing = f.base > 1
        if any((cf > 0) if growing else (cf < 0) for cf in coefficients):
            return None
    return c.xreplace({x: sp.Integer(0) for x in c.free_symbols})


def _lower_bound(x: sp.Expr) -> sp.Expr | None:
    """A lower bound of the real part of ``x``: its real constant terms minus the supremum of
    each other term's modulus (unimodular factors dropped); None when a term has no such
    form."""
    bound: sp.Expr = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(x, power_exp=False)):
        if not term.free_symbols and term.is_real:
            bound += term
            continue
        modulus = sp.Mul(*(f for f in sp.Mul.make_args(term) if not _unimodular(f)))
        sup = _modulus_sup(sp.Abs(modulus) if modulus.is_negative else modulus)
        if sup is None:
            return None
        bound -= sup
    return sp.simplify(bound)


def _case_index(expr: sp.Expr, ranges: Mapping[str, Dim]) -> tuple[sp.Symbol, int] | None:
    """An index symbol of concrete range m that a delta mod m reads alone, plus a constant;
    failing that, the first by name of the index symbols of concrete range m that a delta mod
    m reads among others."""
    deltas = sorted(expr.atoms(ModDelta), key=str)
    shared: list[tuple[str, sp.Symbol, int]] = []
    for delta in deltas:
        argument, modulus = delta.args
        if not modulus.is_Integer:
            continue
        for symbol in argument.free_symbols:
            dim = ranges.get(str(symbol.name))
            if dim is None or not dim.is_concrete or dim.to_int() != int(modulus):
                continue
            if len(argument.free_symbols) == 1 and (
                sp.expand(argument - symbol).is_Integer or sp.expand(argument + symbol).is_Integer
            ):
                return symbol, int(modulus)
            shared.append((str(symbol.name), symbol, int(modulus)))
    if shared:
        _name, symbol, size = min(shared, key=lambda item: item[0])
        return symbol, size
    for delta in deltas:
        for symbol in sorted(delta.args[0].free_symbols, key=lambda s: str(s.name)):
            dim = ranges.get(str(symbol.name))
            if (
                dim is not None
                and dim.is_concrete
                and sp.expand(delta.args[0]).coeff(symbol).free_symbols
            ):
                return symbol, dim.to_int()
    return None


def _reduce_exponents(expr: sp.Expr) -> sp.Expr:
    """``expr`` with each ``exp(2 pi i P)``, P a polynomial over integer symbols, rewritten with
    every coefficient of P reduced mod 1."""

    def reduce(arg: sp.Expr) -> sp.Expr:
        turns = sp.expand(arg / (2 * sp.pi * sp.I))
        if not all(x.is_integer for x in turns.free_symbols):
            return sp.exp(arg)
        if not turns.free_symbols:
            if not turns.is_Rational:
                return sp.exp(arg)
            return sp.exp(2 * sp.pi * sp.I * (turns - sp.floor(turns)))
        try:
            poly = sp.Poly(turns, *sorted(turns.free_symbols, key=str))
        except sp.PolynomialError:
            return sp.exp(arg)
        if not all(c.is_Rational for c in poly.coeffs()):
            return sp.exp(arg)
        reduced = sp.Add(
            *(
                (c - sp.floor(c))
                * sp.Mul(*(g**e for g, e in zip(poly.gens, monomial, strict=True)))
                for monomial, c in poly.terms()
            )
        )
        return sp.exp(2 * sp.pi * sp.I * reduced)

    return expr.replace(lambda e: isinstance(e, sp.exp), lambda e: reduce(e.args[0]))


def _fold_residues(expr: sp.Expr) -> sp.Expr:
    """``expr`` with each sum ``c [x + r_0] + ... + c [x + r_(m-1)]`` over all residues mod a
    concrete m replaced by ``c``."""
    groups: dict[tuple[sp.Expr, sp.Symbol, int], dict[int, sp.Expr]] = {}
    rest: list[sp.Expr] = []
    for term in sp.Add.make_args(expr):
        deltas = [f for f in sp.Mul.make_args(term) if isinstance(f, ModDelta)]
        key = None
        if len(deltas) == 1 and deltas[0].args[1].is_Integer:
            argument, modulus = deltas[0].args
            free = sorted(argument.free_symbols, key=lambda x: str(x))
            if len(free) == 1 and sp.expand(argument).coeff(free[0]) == 1:
                offset = sp.expand(argument - free[0])
                if offset.is_Integer:
                    coeff = sp.Mul(*(f for f in sp.Mul.make_args(term) if f is not deltas[0]))
                    key = (coeff, free[0], int(modulus))
                    residue = int(offset) % int(modulus)
                    if residue in groups.setdefault(key, {}):
                        key = None
                    else:
                        groups[key][residue] = term
        if key is None:
            rest.append(term)
    out = list(rest)
    for (coeff, _symbol, modulus), by_residue in groups.items():
        if len(by_residue) == modulus:
            out.append(coeff)
        else:
            out.extend(by_residue.values())
    return sp.Add(*out)


def _pin(delta: ModDelta, pins: Mapping[sp.Symbol, int]) -> tuple[sp.Symbol, int] | None:
    """The one unpinned index ``delta`` fixes with a unit coefficient, and its residue."""
    argument, modulus = delta.args
    if not modulus.is_Integer:
        return None
    argument = sp.expand(argument.xreplace({s: sp.Integer(v) for s, v in pins.items()}))
    free = sorted(argument.free_symbols, key=lambda symbol: str(symbol.name))
    if len(free) != 1:
        return None
    (symbol,) = free
    coefficient = argument.coeff(symbol)
    rest = sp.expand(argument - coefficient * symbol)
    if coefficient not in (1, -1) or not rest.is_Integer:
        return None
    return symbol, int(-rest * coefficient) % int(modulus)


def support_points(
    difference: Scalar, indices: Sequence[str], extents: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Index tuples at which ``difference`` is likely nonzero: all zeros, all ones, then per
    term the point its unit-coefficient deltas pin, every other index 0, deduplicated."""
    expr = difference.to_sympy()
    symbols = {str(symbol.name): symbol for symbol in expr.free_symbols}
    ordered = [symbols.get(name) for name in indices]
    points: list[tuple[int, ...]] = [
        tuple(0 for _ in extents),
        tuple(1 % extent for extent in extents),
    ]
    for term in sp.Add.make_args(sp.expand(expr)):
        deltas = [factor for factor in sp.Mul.make_args(term) if isinstance(factor, ModDelta)]
        pins: dict[sp.Symbol, int] = {}
        progress = True
        while progress:
            progress = False
            for delta in deltas:
                pinned = _pin(delta, pins)
                if pinned is not None and pinned[0] not in pins:
                    pins[pinned[0]] = pinned[1]
                    progress = True
        points.append(
            tuple(
                (pins.get(symbol, 0) if symbol is not None else 0) % extent
                for symbol, extent in zip(ordered, extents, strict=True)
            )
        )
    return tuple(dict.fromkeys(points))


def find_entry_witness(
    left: SymbolicTensor,
    right: SymbolicTensor,
    assignments: Sequence[Mapping[str, CheckAssignmentValue]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS,
) -> EntryWitness | None:
    """The first entry, over ``assignments`` and then :func:`support_points` of the simplified
    difference, where the tensors differ by more than ``tolerance`` times the larger entry
    modulus (at least 1); None when none does.

    An assignment leaving a dimension or entry symbol free, or whose axis dimensions differ
    between the sides, is skipped.
    """
    for assignment in assignments:
        try:
            at_l = left.substitute(assignment)
            at_r = right.substitute(assignment)
        except (SymbolicContractionError, ScalarError, TypeError, ValueError):
            continue
        dims = at_l.dims()
        if at_r.dims() != dims or not all(dim.is_concrete for dim in dims):
            continue
        names = tuple(axis.index for axis in at_l.axes)
        if tuple(axis.index for axis in at_r.axes) != names:
            continue
        if (at_l.entry.free_symbols | at_r.entry.free_symbols) - set(names):
            continue
        try:
            difference = (at_l.entry - at_r.entry).simplify(max_steps=max_steps)
        except ScalarError:
            continue
        if difference.is_zero:
            continue
        extents = tuple(dim.to_int() for dim in dims)
        for point in support_points(difference, names, extents):
            try:
                value_l = at_l.value_at(point, max_steps=max_steps).to_complex()
                value_r = at_r.value_at(point, max_steps=max_steps).to_complex()
            except (SymbolicContractionError, ScalarError, TypeError, ValueError):
                continue
            scale = max(1.0, abs(value_l), abs(value_r))
            if abs(value_l - value_r) > tolerance * scale:
                return EntryWitness(MappingProxyType(dict(assignment)), point, value_l, value_r)
    return None
