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
thing at a shared one. Everything builds on :mod:`qufzx.semantics.contract_numeric`;
nothing here touches a dense array directly.

Instantiation. :func:`instantiate` substitutes every dimension, phase, and scalar symbol
via the node-id-preserving :meth:`~qufzx.diagram.graph.Diagram.substitute`. It refuses --
rather than defaulting a missing symbol to something "reasonable" -- whenever the
assignment does not mention every free symbol the diagram carries. The spec's "never
construct a matrix or dense tensor while any dimension or count in scope is symbolic" rule
is enforced downstream in ``contract_numeric``, but catching a missing symbol here gives a
far more specific error.

Comparison modes. Exactly two, and ``EXACT`` is the default everywhere;
``UP_TO_GLOBAL_PHASE`` is opt-in, never inferred. ``EXACT`` requires matching shapes and
entrywise agreement within ``tolerance``, including the overall scalar.
``UP_TO_GLOBAL_PHASE`` asks whether some unit-modulus ``lambda`` has ``b == lambda * a``;
``lambda`` is recovered from ``a``'s largest-magnitude entry (least sensitive to
floating-point noise) and then verified against the whole tensor. A recovered ``lambda``
with ``|lambda| != 1`` is a non-match: a rescaling by 2 is not a global phase, and this
mode is deliberately not an up-to-scale escape hatch. All-zero tensors match each other;
one zero and one nonzero never match.

Tolerance is a single explicit parameter defaulting to ``1e-9``, an absolute entrywise
bound, with no path that silently loosens it -- comfortably above float64 accumulation
noise at the tensor sizes this oracle targets, far below any meaningful difference.

:class:`ComparisonResult` carries the mode, a matched flag, a reason, the max absolute
deviation observed, and the recovered ``lambda`` -- what a developer needs when a later
phase's rewrite turns out wrong, not a yes/no.

Interface check, before tensors are compared. :func:`compare_tensors` sees bare arrays, so
it can do no better than ``a.shape == b.shape``; it cannot know that axis ``i`` of each is
meant to describe the same boundary leg. :func:`compare` therefore first checks that the
interfaces correspond: the same number of boundary outputs, the same number of boundary
inputs (not merely the same total, so a 2-output/1-input diagram never matches a
1-output/2-input one), and the same dimension per axis. A mismatch is reported as its own
result rather than being misreported as a numeric deviation.

What this cannot do. Genuine leg correspondence -- that boundary axis ``i`` of A and axis
``j`` of B denote the same leg, not merely the same dimension -- is not establishable here.
Two diagrams denoting the same map may distribute boundary legs over an entirely different
node structure, so no per-axis identity survives a rewrite. A silently reordered boundary
is therefore invisible whenever the tensors agree regardless of order. Establishing real
correspondence requires a rule to declare the map between its input and output boundaries
and the engine to assert it -- a :mod:`qufzx.rewrite.engine` concern feeding Phase 6.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.diagram.graph import Diagram
from qufzx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractionResult, contract

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
    :mod:`qufzx.semantics.contract_numeric` and is refused there instead. Declared so the
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
    """Every free symbol (dimension, phase, or scalar) appearing anywhere in ``diagram``."""
    symbols: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            symbols |= port.dim.free_symbols
        if node.phase is not None:
            symbols |= node.phase.free_symbols
    return frozenset(symbols)


def instantiate(diagram: Diagram, assignment: Mapping[str, CheckAssignmentValue]) -> Diagram:
    """Substitute every symbol in ``diagram`` per ``assignment``.

    Raises CheckGrammarError if ``assignment`` does not cover every free symbol the
    diagram carries -- a missing symbol is never defaulted. A key naming no symbol in this
    diagram is dropped: :func:`compare` and the sweeps pass one shared assignment to two
    diagrams, and a rewrite can eliminate a symbol from one side (a phase binding the
    shared leg dimension leaves the merged node concrete). See the module docstring.
    """
    missing = _diagram_free_symbols(diagram) - set(assignment)
    if missing:
        raise CheckGrammarError(
            f"assignment leaves symbol(s) uninstantiated: {sorted(missing)}; "
            "instantiate() never defaults a missing symbol"
        )
    return diagram.substitute(assignment)


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

    Resolves each :class:`~qufzx.diagram.graph.PortRef` against ``diagram``, following the
    same ``node.legs(ref.direction)[ref.index]`` pattern
    :mod:`qufzx.semantics.contract_numeric` uses for its own output-size guard, rather than
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
