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

Comparison modes. Exactly two, and ``EXACT`` is the default everywhere;
``UP_TO_GLOBAL_PHASE`` is opt-in, never inferred. ``EXACT`` requires matching shapes and
entrywise agreement within ``tolerance``, including the overall scalar.
``UP_TO_GLOBAL_PHASE`` asks whether some unit-modulus ``lambda`` has ``b == lambda * a``;
``lambda`` is recovered from ``a``'s largest-magnitude entry and then verified against the
whole tensor. A recovered ``lambda`` with ``|lambda| != 1`` is a non-match: a rescaling by
2 is not a global phase, and this mode is not an up-to-scale escape hatch. All-zero tensors
match each other; one zero and one nonzero never match.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import (
    DEFAULT_MAX_SIMPLIFY_STEPS,
    ModDelta,
    ModGcd,
    Scalar,
    ScalarError,
)
from archytaszx.diagram.bangbox import (
    BangBoxDomainError,
    expand_concrete_boxes,
    free_mult_symbols,
    instantiate_symbol,
)
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractionResult, contract
from archytaszx.semantics.contract_symbolic import SymbolicContractionError, SymbolicTensor

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
    """Every free symbol (dimension, phase, scalar, or bang-box multiplicity, Phase 7)
    appearing anywhere in ``diagram``."""
    symbols: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            symbols |= port.dim.free_symbols
        if node.phase is not None:
            symbols |= node.phase.free_symbols
    symbols |= free_mult_symbols(diagram)
    return frozenset(symbols)


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


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of one :func:`compare_tensors` or :func:`compare` call. See the module
    docstring."""

    mode: EqualityMode
    matched: bool
    reason: str
    max_abs_deviation: float
    recovered_lambda: complex | None = None


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
        deviation = float(np.max(np.abs(a - b))) if a.size else 0.0
        matched = deviation <= tolerance
        reason = (
            "tensors agree entrywise within tolerance"
            if matched
            else f"max abs deviation {deviation} exceeds tolerance {tolerance}"
        )
        return ComparisonResult(
            mode=mode, matched=matched, reason=reason, max_abs_deviation=deviation
        )

    if mode is EqualityMode.UP_TO_GLOBAL_PHASE:
        return _compare_up_to_global_phase(a, b, tolerance=tolerance)

    raise CheckGrammarError(f"unknown equality mode {mode!r}")


def _compare_up_to_global_phase(
    a: np.ndarray, b: np.ndarray, *, tolerance: float
) -> ComparisonResult:
    mode = EqualityMode.UP_TO_GLOBAL_PHASE
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
        if tuple((g.factor, g.marked, g.inner) for g in left.groups) != tuple(
            (g.factor, g.marked, g.inner) for g in right.groups
        ):
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
    if difference.is_zero:
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
        )
    if difference.to_sympy().atoms(sp.Sum, ModDelta, ModGcd):
        return ComparisonResult(
            mode,
            False,
            f"indeterminate: the difference {difference} still carries an unevaluated "
            "index sum or divisibility atom, so no conclusion is drawn",
            float("inf"),
        )
    return ComparisonResult(mode, False, f"entries differ by {difference}", float("inf"))


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
