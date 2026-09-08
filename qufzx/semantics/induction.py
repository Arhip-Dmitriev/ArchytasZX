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

"""Symbolic-n proof by induction over bang-box multiplicities, one index at a time.

Builds a base case plus a hypothesis/successor pair for one multiplicity index, then
discharges the step case through a ladder of tiers (rewrite, induction-rewrite, an oracle
window, or a symbolic contraction) until one settles or all decline.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from qufzx.algebra.scalar import ScalarBudgetError
from qufzx.diagram.bangbox import BangBoxError, Mult, free_mult_symbols, peel_one
from qufzx.diagram.graph import Diagram, NodeId, PortRef
from qufzx.diagram.validate import ValidateError, validate_or_raise
from qufzx.rewrite.engine import RewriteStep, apply
from qufzx.rewrite.rule import Match, Rule
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.certificate import (
    Derivation,
    DerivationKind,
    InductionClaim,
    compare_structure,
)
from qufzx.semantics.check import (
    DEFAULT_TOLERANCE,
    CheckAssignmentValue,
    CheckError,
    ComparisonResult,
    EqualityMode,
    compare,
    compare_symbolic,
    compare_tensors,
    instantiate,
    score,
)
from qufzx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractError
from qufzx.semantics.contract_symbolic import (
    SymbolicContractionDomainError,
    SymbolicContractionUnsupportedError,
    contract_symbolic,
)
from qufzx.semantics.denote import DenoteError


class InductionError(Exception):
    """Base class for all errors raised by this module."""


class InductionDomainError(InductionError):
    """A value is outside the mathematical domain this module accepts."""


class InductionGrammarError(InductionError):
    """A request is malformed, independent of any diagram's concreteness."""


class InductionUnavailableError(InductionError):
    """A discharge tier's backing machinery is not implemented."""


class Verdict(enum.Enum):
    """What an induction attempt established about the claimed equality."""

    PROVED_UNIFORM = "proved_uniform"
    PROVED_INDUCTION = "proved_induction"
    SCHEMA_CHECKED = "schema_checked"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class StepDischarge(enum.Enum):
    """Which machinery a tier uses to settle the step case."""

    UNIFORM_REWRITE = "uniform_rewrite"
    INDUCTION_REWRITE = "induction_rewrite"
    ORACLE_WINDOW = "oracle_window"
    SYMBOLIC_CONTRACTION = "symbolic_contraction"


DEFAULT_WINDOW_WIDTH = 4
DEFAULT_MAX_STEPS = 64
DEFAULT_STEP_SYMBOL = "k"
DEFAULT_LADDER: tuple[StepDischarge, ...] = (
    StepDischarge.UNIFORM_REWRITE,
    StepDischarge.INDUCTION_REWRITE,
    StepDischarge.SYMBOLIC_CONTRACTION,
    StepDischarge.ORACLE_WINDOW,
)


@dataclass(frozen=True, slots=True, eq=False)
class InductionObligation:
    """The six diagrams an induction over one index is discharged against, and the bindings they
    were built under."""

    index: str
    base_value: int
    step_symbol: str
    left_at_base: Diagram
    right_at_base: Diagram
    left_at_k: Diagram
    right_at_k: Diagram
    left_at_successor: Diagram
    right_at_successor: Diagram
    witness: Mapping[str, CheckAssignmentValue] = MappingProxyType({})
    held_symbolic: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class TierOutcome:
    """The result of running one discharge tier."""

    discharge: StepDischarge
    settled: bool
    reason: str
    comparisons: tuple[ComparisonResult, ...] = ()
    steps: tuple[RewriteStep, ...] = ()
    counterexample: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class InductionResult:
    """A verdict on one equality over one multiplicity index, the tiers that produced it, and the
    obligation they ran against."""

    verdict: Verdict
    index: str
    base_value: int
    reason: str
    obligation: InductionObligation
    witness: Mapping[str, CheckAssignmentValue] = MappingProxyType({})
    base: ComparisonResult | None = None
    tiers: tuple[TierOutcome, ...] = ()
    comparison: ComparisonResult | None = None
    counterexample: int | None = None
    derivation: Derivation | None = None

    @property
    def proved(self) -> bool:
        """True when the verdict is one of the two proved verdicts."""
        return self.verdict in (Verdict.PROVED_UNIFORM, Verdict.PROVED_INDUCTION)

    @property
    def discharge(self) -> StepDischarge | None:
        """The tier that settled the verdict, or None when none did."""
        for tier in reversed(self.tiers):
            if tier.settled:
                return tier.discharge
        return None

    @property
    def held_symbolic(self) -> frozenset[str]:
        """The symbols the settling tier left universally quantified."""
        symbolic_tiers = (
            StepDischarge.UNIFORM_REWRITE,
            StepDischarge.INDUCTION_REWRITE,
            StepDischarge.SYMBOLIC_CONTRACTION,
        )
        if self.discharge in symbolic_tiers:
            return self.obligation.held_symbolic
        return frozenset()


# -- construction ---------------------------------------------------------------------


def _all_free_symbols(diagram: Diagram) -> frozenset[str]:
    """Every free symbol (dimension, phase, scalar, or bang-box multiplicity) in ``diagram``."""
    symbols: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            symbols |= port.dim.free_symbols
        if node.phase is not None:
            symbols |= node.phase.free_symbols
    symbols |= free_mult_symbols(diagram)
    return frozenset(symbols)


def _fresh_symbol_name(stem: str, avoid: frozenset[str]) -> str:
    """The first of ``stem``, ``stem1``, ``stem2``, ... not in ``avoid``."""
    name = stem
    suffix = 0
    while name in avoid:
        suffix += 1
        name = f"{stem}{suffix}"
    return name


def _fresh_step_symbol(left: Diagram, right: Diagram, index: str, step_symbol: str) -> str:
    """``step_symbol`` itself, or a freshening of it against both diagrams' free symbols."""
    avoid = (_all_free_symbols(left) | _all_free_symbols(right)) - {index}
    return _fresh_symbol_name(step_symbol, avoid)


def choose_index(left: Diagram, right: Diagram, *, index: str | None = None) -> str:
    """The lexicographically least multiplicity symbol free in either diagram, or ``index`` when
    given."""
    live = free_mult_symbols(left) | free_mult_symbols(right)
    if not live:
        raise InductionGrammarError("diagram carries no bang box; there is no index to induct on")
    if index is not None:
        if index not in live:
            raise InductionGrammarError(f"no live bang box has multiplicity symbol {index!r}")
        return index
    return min(live)


def rename_multiplicity(diagram: Diagram, index: str, replacement: Mult) -> Diagram:
    """Replace every live bang box's multiplicity ``index`` with ``replacement``, together,
    dropping any parameter binding for ``index``."""
    working = diagram.copy()
    for box_id, box in sorted(working.bang_boxes.items()):
        if index not in box.multiplicity.free_symbols:
            continue
        if not (box.multiplicity.is_bare_symbol and box.multiplicity.bare_symbol_name() == index):
            raise InductionDomainError(
                f"induction requires a bare multiplicity symbol, got {box.multiplicity!r}"
            )
        working.set_bang_box_multiplicity(box_id, replacement)
    if index in working.parameters:
        working.set_parameters(
            {name: value for name, value in working.parameters.items() if name != index}
        )
    return working


def hypothesis_diagram(
    diagram: Diagram, index: str, *, step_symbol: str = DEFAULT_STEP_SYMBOL
) -> Diagram:
    """``diagram`` with its induction index renamed to the bare step symbol."""
    return rename_multiplicity(diagram, index, Mult(step_symbol))


def successor_diagram(
    diagram: Diagram, index: str, *, step_symbol: str = DEFAULT_STEP_SYMBOL
) -> Diagram:
    """``diagram`` with its induction index replaced by the step symbol plus one."""
    fresh = _fresh_step_symbol(diagram, diagram, index, step_symbol)
    return rename_multiplicity(diagram, index, Mult(fresh) + 1)


def choose_base_value(
    left: Diagram,
    right: Diagram,
    index: str,
    witness: Mapping[str, CheckAssignmentValue],
    *,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> int:
    """Zero when both diagrams evaluate there, otherwise one."""
    reasons: dict[int, str] = {}
    for candidate in (0, 1):
        try:
            compare(
                left,
                right,
                {**witness, index: candidate},
                mode=mode,
                tolerance=tolerance,
                max_elements=max_elements,
            )
        except (BangBoxError, CheckError, DenoteError, ContractError, ValidateError) as exc:
            reasons[candidate] = str(exc)
            continue
        return candidate
    raise InductionDomainError(
        "neither multiplicity 0 nor 1 yields an evaluable instance; "
        f"at 0: {reasons[0]}; at 1: {reasons[1]}"
    )


def build_obligation(
    left: Diagram,
    right: Diagram,
    *,
    index: str | None = None,
    base: int | None = None,
    witness: Mapping[str, CheckAssignmentValue] = MappingProxyType({}),
    step_symbol: str = DEFAULT_STEP_SYMBOL,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> InductionObligation:
    """Construct and validate the base, hypothesis and successor diagrams for one index."""
    idx = choose_index(left, right, index=index)
    held = (free_mult_symbols(left) | free_mult_symbols(right)) - {idx}
    required = (_all_free_symbols(left) | _all_free_symbols(right)) - {idx} - held
    supplied = set(witness)
    missing = required - supplied
    if missing:
        raise InductionGrammarError(
            f"induction over {idx!r} leaves symbol(s) uninstantiated: {sorted(missing)}; "
            "name them in the witness to hold them at a value"
        )
    if base is not None and (isinstance(base, bool) or base not in (0, 1)):
        raise InductionDomainError(f"base index must be 0 or 1, got {base!r}")
    if base is None:
        base = choose_base_value(
            left, right, idx, witness, mode=mode, tolerance=tolerance, max_elements=max_elements
        )
    step = _fresh_step_symbol(left, right, idx, step_symbol)
    obligation = InductionObligation(
        index=idx,
        base_value=base,
        step_symbol=step,
        left_at_base=instantiate(left, {**witness, idx: base}),
        right_at_base=instantiate(right, {**witness, idx: base}),
        left_at_k=hypothesis_diagram(left, idx, step_symbol=step),
        right_at_k=hypothesis_diagram(right, idx, step_symbol=step),
        left_at_successor=successor_diagram(left, idx, step_symbol=step),
        right_at_successor=successor_diagram(right, idx, step_symbol=step),
        witness=MappingProxyType(dict(witness)),
        held_symbolic=required | held,
    )
    for diagram in (
        obligation.left_at_base,
        obligation.right_at_base,
        obligation.left_at_k,
        obligation.right_at_k,
        obligation.left_at_successor,
        obligation.right_at_successor,
    ):
        validate_or_raise(diagram)
    return obligation


_VERDICT: dict[StepDischarge, Verdict] = {
    StepDischarge.UNIFORM_REWRITE: Verdict.PROVED_UNIFORM,
    StepDischarge.INDUCTION_REWRITE: Verdict.PROVED_INDUCTION,
    StepDischarge.SYMBOLIC_CONTRACTION: Verdict.PROVED_INDUCTION,
    StepDischarge.ORACLE_WINDOW: Verdict.SCHEMA_CHECKED,
}


# -- discharge tiers: discharge_base, hypothesis_rule, the four discharge_* (L2-2) ------


def discharge_base(
    obligation: InductionObligation,
    *,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ComparisonResult:
    """Compare the two base diagrams through the numeric oracle."""
    left_tensor = score(obligation.left_at_base, {}, max_elements=max_elements).tensor
    right_tensor = score(obligation.right_at_base, {}, max_elements=max_elements).tensor
    return compare_tensors(left_tensor, right_tensor, mode=mode, tolerance=tolerance)


def _port_scope_key(ref: PortRef) -> tuple[int, str, int]:
    """A total order on a bang box's port-scope entries."""
    return (ref.node_id, ref.direction.value, ref.index)


def _same_bang_boxes(a: Diagram, b: Diagram) -> bool:
    """True when both diagrams carry the same bang boxes, field for field."""
    if sorted(a.bang_boxes) != sorted(b.bang_boxes):
        return False
    for box_id in sorted(a.bang_boxes):
        box_a, box_b = a.bang_boxes[box_id], b.bang_boxes[box_id]
        if box_a.multiplicity != box_b.multiplicity or box_a.parent != box_b.parent:
            return False
        if sorted(box_a.node_scope) != sorted(box_b.node_scope):
            return False
        if sorted(box_a.port_scope, key=_port_scope_key) != sorted(
            box_b.port_scope, key=_port_scope_key
        ):
            return False
    return True


DEFAULT_RULES: tuple[Rule, ...] = (SPIDER_FUSION,)


def _rewrite_to(
    source: Diagram,
    target: Diagram,
    rules: Sequence[Rule],
    max_steps: int,
    guard_symbols: frozenset[str] | None,
) -> tuple[bool, tuple[RewriteStep, ...], str]:
    """Apply up to ``max_steps`` rule firings from ``source`` toward ``target``.

    Tries each rule in the order given and, within one rule, its matches in the order
    :meth:`~qufzx.rewrite.rule.Pattern.find_matches` returns them; the first applicable
    pair fires. ``guard_symbols``, when given, rejects a step whose result changes
    :func:`~qufzx.diagram.bangbox.free_mult_symbols` away from it. A
    :class:`~qufzx.rewrite.rule.RewriteError` from :func:`~qufzx.rewrite.engine.apply`
    propagates uncaught.
    """
    working = source.copy()
    steps: list[RewriteStep] = []
    for _firing in range(max_steps):
        if compare_structure(working, target).identical and _same_bang_boxes(working, target):
            return True, tuple(steps), "structurally reached the target"
        candidate: tuple[Rule, Match] | None = None
        for rule in rules:
            matches = rule.pattern.find_matches(working)
            if matches:
                candidate = (rule, matches[0])
                break
        if candidate is None:
            return False, tuple(steps), "no applicable rule"
        chosen_rule, chosen_match = candidate
        result = apply(working, chosen_rule, chosen_match)
        if guard_symbols is not None and free_mult_symbols(result.diagram) != guard_symbols:
            return False, tuple(steps), "a step consumed the multiplicity symbol"
        working = result.diagram
        steps = [*steps, result.step]
    return False, tuple(steps), f"step budget {max_steps} exhausted"


def _hypothesis_inexpressible_reason(obligation: InductionObligation) -> str:
    """How the hypothesis diagram fails to occur in the successor diagram as a pattern."""
    boxes_k = obligation.left_at_k.bang_boxes
    boxes_successor = obligation.left_at_successor.bang_boxes
    if sorted(boxes_k) != sorted(boxes_successor):
        return "the hypothesis and successor left sides carry different bang-box ids"
    for box_id in sorted(boxes_k):
        hypothesis_mult = boxes_k[box_id].multiplicity
        successor_mult = boxes_successor[box_id].multiplicity
        if hypothesis_mult == successor_mult:
            continue
        return (
            f"bang box {box_id!r} carries multiplicity {hypothesis_mult!r} in the hypothesis "
            f"and {successor_mult!r} in the successor; every Pattern here matches structure "
            "only and none unifies Mult arithmetic, and a match ignoring the multiplicity "
            "would rewrite the successor straight onto its own target"
        )
    return "the hypothesis left side already equals the successor left side field for field"


def _extract(diagram: Diagram, node_ids: frozenset[NodeId]) -> Diagram:
    """The sub-diagram on ``node_ids``: their wires among themselves, their boundary refs in
    the enclosing order."""
    extracted = Diagram()
    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(node_ids):
        node = diagram.nodes[old_id]
        id_map[old_id] = extracted.add_node(
            node.generator_type,
            [port.dim for port in node.inputs],
            [port.dim for port in node.outputs],
            phase=node.phase,
        )
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if wire.a.node_id in node_ids and wire.b.node_id in node_ids:
            extracted.add_wire(
                PortRef(id_map[wire.a.node_id], wire.a.direction, wire.a.index),
                PortRef(id_map[wire.b.node_id], wire.b.direction, wire.b.index),
            )
    extracted.set_boundary_inputs(
        [
            PortRef(id_map[ref.node_id], ref.direction, ref.index)
            for ref in diagram.boundary_inputs
            if ref.node_id in node_ids
        ]
    )
    extracted.set_boundary_outputs(
        [
            PortRef(id_map[ref.node_id], ref.direction, ref.index)
            for ref in diagram.boundary_outputs
            if ref.node_id in node_ids
        ]
    )
    extracted.set_parameters(dict(diagram.parameters))
    return extracted


def _residual_is(peeled: Diagram, copy_node_ids: frozenset[NodeId], expected: Diagram) -> bool:
    """Whether ``peeled`` minus its peeled copy is ``expected``, field for field."""
    kept = frozenset(peeled.nodes) - copy_node_ids
    if kept != frozenset(expected.nodes):
        return False
    residual = _extract(peeled, kept)
    reference = _extract(expected, frozenset(expected.nodes))
    if not compare_structure(residual, reference).identical:
        return False
    peeled_boxes = {
        box_id: box
        for box_id, box in peeled.bang_boxes.items()
        if not (box.node_scope & copy_node_ids)
        and not any(ref.node_id in copy_node_ids for ref in box.port_scope)
    }
    if sorted(peeled_boxes) != sorted(expected.bang_boxes):
        return False
    for box_id, box in peeled_boxes.items():
        other = expected.bang_boxes[box_id]
        if box.multiplicity != other.multiplicity or box.parent != other.parent:
            return False
        if sorted(box.node_scope) != sorted(other.node_scope):
            return False
        if sorted(box.port_scope, key=_port_scope_key) != sorted(
            other.port_scope, key=_port_scope_key
        ):
            return False
    return True


def _peel_step_index(diagram: Diagram, index: str) -> tuple[Diagram, frozenset[NodeId]] | None:
    """Peel one copy off the single box whose multiplicity carries ``index``."""
    owners = [
        box_id
        for box_id, box in sorted(diagram.bang_boxes.items())
        if index in box.multiplicity.free_symbols
    ]
    if len(owners) != 1:
        return None
    try:
        result = peel_one(diagram, owners[0])
    except BangBoxError:
        return None
    if not result.separable or not result.copy_node_ids:
        return None
    return result.diagram, result.copy_node_ids


def hypothesis_rule(obligation: InductionObligation) -> Rule | None:
    """A rule rewriting the hypothesis diagram's left side to its right side, or None when that
    pattern is not matchable."""
    return None


def discharge_uniform_rewrite(
    obligation: InductionObligation,
    *,
    rules: Sequence[Rule] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> TierOutcome:
    """Rewrite the hypothesis diagrams into each other with the multiplicity symbol never
    instantiated."""
    rules_in_use = rules if rules is not None else DEFAULT_RULES
    ok, steps, reason = _rewrite_to(
        obligation.left_at_k,
        obligation.right_at_k,
        sorted(rules_in_use, key=lambda rule: rule.name),
        max_steps,
        guard_symbols=free_mult_symbols(obligation.left_at_k),
    )
    return TierOutcome(StepDischarge.UNIFORM_REWRITE, ok, reason, steps=steps)


def discharge_peeled_hypothesis(
    obligation: InductionObligation, *, max_steps: int = DEFAULT_MAX_STEPS
) -> TierOutcome:
    """Settle the step case by peeling one copy off each successor and discharging the rest.

    ``!_{k+1}(G)`` splits into ``!_k(G)`` beside one bare copy of ``G``
    (:func:`~qufzx.diagram.bangbox.peel_one`). When each peeled residual is exactly its own
    hypothesis diagram, the hypothesis settles the residuals and the step reduces to the two
    peeled copies, which carry no bang box and are compared by symbolic contraction with ``d``
    formal. Requires a separable (node-scope) box, and one owner of the index per side.
    """
    index = obligation.step_symbol
    left = _peel_step_index(obligation.left_at_successor, index)
    right = _peel_step_index(obligation.right_at_successor, index)
    if left is None or right is None:
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the step index is not carried by a single separable bang box on each side, so no "
            "copy can be peeled off to expose the hypothesis",
        )
    left_peeled, left_copy = left
    right_peeled, right_copy = right

    if not _residual_is(left_peeled, left_copy, obligation.left_at_k):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the left successor's peeled residual is not the left hypothesis diagram",
        )
    if not _residual_is(right_peeled, right_copy, obligation.right_at_k):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the right successor's peeled residual is not the right hypothesis diagram",
        )

    left_extra = _extract(left_peeled, left_copy)
    right_extra = _extract(right_peeled, right_copy)
    try:
        contracted_left = contract_symbolic(left_extra, max_steps=max_steps)
        contracted_right = contract_symbolic(right_extra, max_steps=max_steps)
    except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError) as exc:
        return TierOutcome(StepDischarge.INDUCTION_REWRITE, False, str(exc), ())
    except ScalarBudgetError as exc:
        return TierOutcome(StepDischarge.INDUCTION_REWRITE, False, str(exc), ())

    comparison = compare_symbolic(contracted_left, contracted_right)
    if comparison.matched:
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            True,
            "peeled one copy off each successor; the residuals are the hypothesis diagrams and "
            "the peeled copies agree under symbolic contraction with d formal",
            (comparison,),
        )
    return TierOutcome(
        StepDischarge.INDUCTION_REWRITE,
        False,
        f"the peeled copies are not equal: {comparison.reason}",
        (comparison,),
    )


def discharge_induction_rewrite(
    obligation: InductionObligation,
    *,
    rules: Sequence[Rule] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> TierOutcome:
    """Rewrite the successor diagrams into each other with the induction hypothesis available as a
    rule, falling back to :func:`discharge_peeled_hypothesis` when no such rule exists."""
    hypothesis = hypothesis_rule(obligation)
    if hypothesis is None:
        peeled = discharge_peeled_hypothesis(obligation, max_steps=max_steps)
        if peeled.settled:
            return peeled
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "hypothesis not expressible as a rule: "
            + _hypothesis_inexpressible_reason(obligation)
            + f"; peeling did not settle it either: {peeled.reason}",
        )
    rules_in_use = rules if rules is not None else DEFAULT_RULES
    ok, steps, reason = _rewrite_to(
        obligation.left_at_successor,
        obligation.right_at_successor,
        [*sorted(rules_in_use, key=lambda rule: rule.name), hypothesis],
        max_steps,
        guard_symbols=None,
    )
    if ok and not any(step.rule_name == hypothesis.name for step in steps):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "derivation never used the hypothesis",
            steps=steps,
        )
    return TierOutcome(StepDischarge.INDUCTION_REWRITE, ok, reason, steps=steps)


def discharge_oracle_window(
    obligation: InductionObligation,
    *,
    width: int = DEFAULT_WINDOW_WIDTH,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> TierOutcome:
    """Compare the two diagrams through the oracle at every multiplicity in a contiguous window
    above the base."""
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise InductionDomainError(f"window width must be a positive int, got {width!r}")
    unbound = sorted(obligation.held_symbolic - set(obligation.witness))
    if unbound:
        raise InductionGrammarError(
            f"a numeric tier leaves symbol(s) uninstantiated: {unbound}; name them in the "
            "witness to hold them at a value"
        )
    b = obligation.base_value
    comparisons: list[ComparisonResult] = []
    for v in range(b, b + width):
        comparison = compare(
            obligation.left_at_k,
            obligation.right_at_k,
            {**obligation.witness, obligation.step_symbol: v},
            mode=mode,
            tolerance=tolerance,
            max_elements=max_elements,
        )
        comparisons.append(comparison)
        if not comparison.matched:
            return TierOutcome(
                StepDischarge.ORACLE_WINDOW,
                False,
                f"mismatch at {obligation.index}={v}",
                tuple(comparisons),
                (),
                counterexample=v,
            )
    return TierOutcome(
        StepDischarge.ORACLE_WINDOW,
        True,
        f"equality confirmed at {obligation.index} = {b}..{b + width - 1}; a finite schema "
        "check, not a proof",
        tuple(comparisons),
    )


def discharge_symbolic_contraction(
    obligation: InductionObligation, *, max_steps: int = DEFAULT_MAX_STEPS
) -> TierOutcome:
    """Settle the step case by contracting both successor diagrams with every symbol kept formal.

    A shape the symbolic contractor does not cover, or a budget it exhausts, returns an
    unsettled outcome rather than raising, so the ladder falls through to the tier below.
    A malformed obligation still propagates: no lower tier can rescue it.
    """
    try:
        left = contract_symbolic(obligation.left_at_successor, max_steps=max_steps)
        right = contract_symbolic(obligation.right_at_successor, max_steps=max_steps)
    except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError) as exc:
        return TierOutcome(StepDischarge.SYMBOLIC_CONTRACTION, False, str(exc), ())
    except ScalarBudgetError as exc:
        return TierOutcome(StepDischarge.SYMBOLIC_CONTRACTION, False, str(exc), ())
    comparison = compare_symbolic(left, right)
    if comparison.matched:
        return TierOutcome(
            StepDischarge.SYMBOLIC_CONTRACTION,
            True,
            "successor case closed by symbolic contraction with every symbol formal",
            (comparison,),
        )
    return TierOutcome(StepDischarge.SYMBOLIC_CONTRACTION, False, comparison.reason, (comparison,))


# -- prove_by_induction, _build_derivation and the certificate wiring (L2-3) ------------


_DISPATCH: dict[StepDischarge, Callable[..., TierOutcome]] = {
    StepDischarge.UNIFORM_REWRITE: discharge_uniform_rewrite,
    StepDischarge.INDUCTION_REWRITE: discharge_induction_rewrite,
    StepDischarge.ORACLE_WINDOW: discharge_oracle_window,
    StepDischarge.SYMBOLIC_CONTRACTION: discharge_symbolic_contraction,
}


def _run_tier(
    discharge: StepDischarge,
    obligation: InductionObligation,
    *,
    rules: Sequence[Rule] | None,
    max_steps: int,
    width: int,
    mode: EqualityMode,
    tolerance: float,
    max_elements: int,
) -> TierOutcome:
    """Call one tier's discharge function with the keyword arguments its signature takes."""
    run = _DISPATCH[discharge]
    if discharge in (StepDischarge.UNIFORM_REWRITE, StepDischarge.INDUCTION_REWRITE):
        return run(obligation, rules=rules, max_steps=max_steps)
    if discharge is StepDischarge.ORACLE_WINDOW:
        return run(
            obligation, width=width, mode=mode, tolerance=tolerance, max_elements=max_elements
        )
    return run(obligation, max_steps=max_steps)


def _concrete_child(
    obligation: InductionObligation,
    value: int,
    rules: Sequence[Rule] | None,
    max_steps: int,
) -> tuple[Derivation | None, str]:
    """The step-sequence derivation over the hypothesis pair instantiated at ``value``, or None
    and the rewrite's reason."""
    binding = {**dict(obligation.witness), obligation.step_symbol: value}
    left = instantiate(obligation.left_at_k, binding)
    right = instantiate(obligation.right_at_k, binding)
    rules_in_use = rules if rules is not None else DEFAULT_RULES
    reached, steps, reason = _rewrite_to(left, right, rules_in_use, max_steps, None)
    if not reached:
        return None, f"at {obligation.step_symbol}={value}: {reason}"
    return (
        Derivation(kind=DerivationKind.STEP_SEQUENCE, initial=left, final=right, steps=steps),
        "",
    )


def _build_derivation(
    obligation: InductionObligation,
    outcome: TierOutcome,
    *,
    rules: Sequence[Rule] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[Derivation | None, str]:
    """The induction derivation over the hypothesis pair carrying a base and a successor child,
    or None with the rewrite failure that blocked it."""
    base_child, base_reason = _concrete_child(obligation, obligation.base_value, rules, max_steps)
    if base_child is None:
        return None, base_reason
    step_child, step_reason = _concrete_child(
        obligation, obligation.base_value + 1, rules, max_steps
    )
    if step_child is None:
        return None, step_reason
    claim = InductionClaim(
        symbol=obligation.index,
        base_value=obligation.base_value,
        held_symbolic=tuple(sorted(obligation.held_symbolic)),
        step_discharge=outcome.discharge.value,
    )
    return (
        Derivation(
            kind=DerivationKind.INDUCTION,
            initial=obligation.left_at_k,
            final=obligation.right_at_k,
            steps=outcome.steps,
            children=(base_child, step_child),
            induction=claim,
        ),
        "",
    )


def prove_by_induction(
    left: Diagram,
    right: Diagram,
    *,
    index: str | None = None,
    base: int | None = None,
    witness: Mapping[str, CheckAssignmentValue] = MappingProxyType({}),
    ladder: Sequence[StepDischarge] = DEFAULT_LADDER,
    rules: Sequence[Rule] | None = None,
    step_symbol: str = DEFAULT_STEP_SYMBOL,
    width: int = DEFAULT_WINDOW_WIDTH,
    max_steps: int = DEFAULT_MAX_STEPS,
    certify_proof: bool = True,
    mode: EqualityMode = EqualityMode.EXACT,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> InductionResult:
    """Discharge an equality over one multiplicity index by running the base case then each
    discharge tier in ladder order until one settles."""
    obligation = build_obligation(
        left,
        right,
        index=index,
        base=base,
        witness=witness,
        step_symbol=step_symbol,
        mode=mode,
        tolerance=tolerance,
        max_elements=max_elements,
    )
    base_comparison = discharge_base(
        obligation, mode=mode, tolerance=tolerance, max_elements=max_elements
    )
    if not base_comparison.matched:
        return InductionResult(
            verdict=Verdict.REFUTED,
            index=obligation.index,
            base_value=obligation.base_value,
            reason=(
                f"the base case at {obligation.index}={obligation.base_value} does not hold: "
                f"{base_comparison.reason}"
            ),
            obligation=obligation,
            witness=obligation.witness,
            base=base_comparison,
            comparison=base_comparison,
            counterexample=obligation.base_value,
        )
    tiers: list[TierOutcome] = []
    for discharge in ladder:
        outcome = _run_tier(
            discharge,
            obligation,
            rules=rules,
            max_steps=max_steps,
            width=width,
            mode=mode,
            tolerance=tolerance,
            max_elements=max_elements,
        )
        tiers.append(outcome)
        if outcome.counterexample is not None:
            return InductionResult(
                verdict=Verdict.REFUTED,
                index=obligation.index,
                base_value=obligation.base_value,
                reason=outcome.reason,
                obligation=obligation,
                witness=obligation.witness,
                base=base_comparison,
                tiers=tuple(tiers),
                comparison=outcome.comparisons[-1] if outcome.comparisons else None,
                counterexample=outcome.counterexample,
            )
        if outcome.settled:
            verdict = _VERDICT[discharge]
            derivation: Derivation | None = None
            reason = outcome.reason
            if certify_proof and verdict in (
                Verdict.PROVED_UNIFORM,
                Verdict.PROVED_INDUCTION,
            ):
                derivation, certificate_reason = _build_derivation(
                    obligation, outcome, rules=rules, max_steps=max_steps
                )
                if derivation is None:
                    reason = (
                        f"{outcome.reason}; the verdict stands and no certificate was "
                        f"built: {certificate_reason}"
                    )
            return InductionResult(
                verdict=verdict,
                index=obligation.index,
                base_value=obligation.base_value,
                reason=reason,
                obligation=obligation,
                witness=obligation.witness,
                base=base_comparison,
                tiers=tuple(tiers),
                derivation=derivation,
            )
    return InductionResult(
        verdict=Verdict.INCONCLUSIVE,
        index=obligation.index,
        base_value=obligation.base_value,
        reason="no discharge tier settled the step case",
        obligation=obligation,
        witness=obligation.witness,
        base=base_comparison,
        tiers=tuple(tiers),
    )
