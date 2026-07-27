"""Oracle equality check: instantiate symbols, contract concretely, and compare exactly.

This is the top-level entry point of the Phase 4 oracle: :func:`score` denotes what a
single diagram means at a concrete symbol assignment, and :func:`compare` decides whether
two diagrams mean the same thing at a shared assignment. Everything below builds on
:mod:`qufzx.semantics.contract_numeric`; this module adds nothing that touches a dense
array directly.

Instantiation. :func:`instantiate` substitutes every dimension symbol, every phase
symbol, and every scalar symbol in a diagram to a concrete value, via the node-id-
preserving :meth:`~qufzx.diagram.graph.Diagram.substitute` added to
:mod:`qufzx.diagram.graph` for this purpose (see that module's docstring for why a plain
rebuild through ``add_node`` cannot do this). It refuses -- rather than defaulting a
missing symbol to some "reasonable" value -- whenever the assignment does not mention
every free symbol the diagram actually carries; ``CLAUDE.md``'s "never construct a
matrix... while any dimension or count in scope is symbolic" rule is enforced further
downstream by :mod:`qufzx.semantics.contract_numeric`, but catching a missing symbol
here, before any substitution happens, gives a much more specific error.

Comparison modes. Exactly two are defined, and ``EXACT`` is the default in every
signature in this module -- ``UP_TO_GLOBAL_PHASE`` is opt-in only, never inferred from
context. ``EXACT`` requires matching shapes and every entry to agree within
``tolerance``, including the overall scalar: nothing is normalized away.
``UP_TO_GLOBAL_PHASE`` asks whether there is a unit-modulus ``lambda`` (``|lambda| == 1``,
within ``tolerance``) with ``b == lambda * a``; ``lambda`` is recovered from the entry of
``a`` with the largest magnitude (least sensitive to floating-point noise) and then
verified against the *whole* tensor, not just that one entry. A recovered ``lambda`` with
``|lambda| != 1`` is a documented **non-match** in this mode -- a rescaling by 2 is not a
global phase, and this mode is deliberately not an up-to-scale escape hatch. All-zero
tensors match each other (vacuously unit-modulus-compatible); one zero and one nonzero
tensor never match.

Tolerance. A single explicit parameter, ``tolerance``, defaults to ``1e-9`` (an absolute
bound on entrywise deviation) everywhere in this module; there is no path that silently
loosens it. ``1e-9`` was chosen as comfortably above float64 accumulation noise for the
tensor sizes this oracle targets (see :mod:`qufzx.semantics.contract_numeric`'s
``max_elements`` default) while remaining far below any physically meaningful difference.

Structured results, not bare booleans. :class:`ComparisonResult` carries the mode used, a
matched flag, a human-readable reason, the max absolute deviation actually observed, and
the recovered ``lambda`` (when relevant) -- this is what a developer needs to stare at
when a later phase's rewrite turns out to be wrong, not just a yes/no.
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
"""The default absolute entrywise tolerance used throughout this module. See the module docstring."""


class CheckError(Exception):
    """Base class for all errors raised by this module."""


class CheckDomainError(CheckError):
    """A value is outside the mathematical domain this module requires."""


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
    diagram carries -- a missing symbol is never defaulted. See the module docstring.
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
    """Instantiate ``diagram`` at ``assignment`` and contract it. The oracle's "evaluate" entry point."""
    instantiated = instantiate(diagram, assignment)
    return contract(instantiated, max_elements=max_elements)


class EqualityMode(enum.Enum):
    """How two contracted tensors are compared. See the module docstring."""

    EXACT = "exact"
    UP_TO_GLOBAL_PHASE = "up_to_global_phase"


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of one :func:`compare_tensors` or :func:`compare` call. See the module docstring."""

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
        return ComparisonResult(mode=mode, matched=matched, reason=reason, max_abs_deviation=deviation)

    if mode is EqualityMode.UP_TO_GLOBAL_PHASE:
        return _compare_up_to_global_phase(a, b, tolerance=tolerance)

    raise CheckGrammarError(f"unknown equality mode {mode!r}")


def _compare_up_to_global_phase(a: np.ndarray, b: np.ndarray, *, tolerance: float) -> ComparisonResult:
    mode = EqualityMode.UP_TO_GLOBAL_PHASE
    if a.size == 0:
        return ComparisonResult(mode=mode, matched=True, reason="both tensors are empty", max_abs_deviation=0.0)

    norm_a = float(np.max(np.abs(a)))
    norm_b = float(np.max(np.abs(b)))
    a_zero = norm_a <= tolerance
    b_zero = norm_b <= tolerance
    if a_zero and b_zero:
        return ComparisonResult(mode=mode, matched=True, reason="both tensors are zero", max_abs_deviation=0.0)
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
        else f"max abs deviation {deviation} after removing global phase {lam} exceeds tolerance {tolerance}"
    )
    return ComparisonResult(
        mode=mode, matched=matched, reason=reason, max_abs_deviation=deviation, recovered_lambda=lam
    )


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
    """
    result_a = score(diagram_a, assignment, max_elements=max_elements)
    result_b = score(diagram_b, assignment, max_elements=max_elements)
    return compare_tensors(result_a.tensor, result_b.tensor, mode=mode, tolerance=tolerance)
