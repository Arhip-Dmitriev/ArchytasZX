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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.algebra.scalar import Scalar, ScalarBudgetError
from archytaszx.diagram.bangbox import (
    BangBox,
    BangBoxError,
    Mult,
    _dim_at,
    _port_multiplicity,
    add_box_image,
    aligned_members,
    fed_box,
    free_mult_symbols,
    index_owners,
    index_symbols,
    peel_one,
    peel_owners,
    peels_first,
    substitute_index_phases,
)
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import W_NODE, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import ValidateError, validate_or_raise
from archytaszx.rewrite.engine import RewriteStep, apply
from archytaszx.rewrite.rule import Match, RewriteDomainError, Rule
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.semantics.certificate import (
    Derivation,
    DerivationKind,
    InductionClaim,
    compare_structure,
)
from archytaszx.semantics.check import (
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
from archytaszx.semantics.contract_numeric import (
    DEFAULT_MAX_ELEMENTS,
    ContractDomainError,
    ContractError,
)
from archytaszx.semantics.contract_symbolic import (
    SymbolicContractionDomainError,
    SymbolicContractionUnsupportedError,
    contract_symbolic_pair,
)
from archytaszx.semantics.denote import DenoteError


class InductionError(Exception):
    """Base class for all errors raised by this module."""


class InductionDomainError(InductionError):
    """A value is outside the mathematical domain this module accepts."""


class InductionGrammarError(InductionError):
    """A request is malformed, independent of any diagram's concreteness."""


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
    CHAIN_WINDOW = "chain_window"
    TREE_WINDOW = "tree_window"


DEFAULT_WINDOW_WIDTH = 4
DEFAULT_MAX_STEPS = 64
DEFAULT_STEP_SYMBOL = "k"
DEFAULT_LADDER: tuple[StepDischarge, ...] = (
    StepDischarge.UNIFORM_REWRITE,
    StepDischarge.INDUCTION_REWRITE,
    StepDischarge.SYMBOLIC_CONTRACTION,
    StepDischarge.CHAIN_WINDOW,
    StepDischarge.TREE_WINDOW,
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
    base_proved: bool = False


@dataclass(frozen=True, slots=True)
class TierOutcome:
    """The result of running one discharge tier; a settled peeling tier carries each peeled
    successor and its peeled copy's node ids."""

    discharge: StepDischarge
    settled: bool
    reason: str
    comparisons: tuple[ComparisonResult, ...] = ()
    steps: tuple[RewriteStep, ...] = ()
    counterexample: int | None = None
    left_peeled: Diagram | None = None
    left_copy: frozenset[NodeId] | None = None
    right_peeled: Diagram | None = None
    right_copy: frozenset[NodeId] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class InductionResult:
    """A verdict on one equality over one multiplicity index, the tiers that produced it, and the
    obligation they ran against; a restricted run carries its ``restriction``, the unrestricted
    ``family`` result, ``family_holds`` when that result proved the claim, the
    ``restricted_pair`` it ran on otherwise and the counterexample in the original symbols."""

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
    restriction: object | None = None
    family: InductionResult | None = None
    family_holds: bool = False
    restricted_pair: tuple[Diagram, Diagram] | None = None
    counterexample_original: Mapping[str, CheckAssignmentValue] | None = None

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
            StepDischarge.CHAIN_WINDOW,
        )
        if self.discharge in symbolic_tiers:
            return self.obligation.held_symbolic
        return frozenset()


# -- construction ---------------------------------------------------------------------


def _all_free_symbols(diagram: Diagram) -> frozenset[str]:
    """Every free symbol (dimension, phase, scalar, copy scalar, or bang-box multiplicity) in
    ``diagram``."""
    symbols: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            symbols |= port.dim.free_symbols
        if node.phase is not None:
            symbols |= node.phase.free_symbols
    symbols |= free_mult_symbols(diagram)
    for box in diagram.bang_boxes.values():
        if box.copy_scalar is not None:
            symbols |= box.copy_scalar.free_symbols
    return frozenset(symbols - index_symbols(diagram))


def _fresh_symbol_name(stem: str, avoid: frozenset[str]) -> str:
    """The first of ``stem``, ``stem1``, ``stem2``, ... not in ``avoid``."""
    name = stem
    suffix = 0
    while name in avoid:
        suffix += 1
        name = f"{stem}{suffix}"
    return name


def _fresh_step_symbol(left: Diagram, right: Diagram, index: str, step_symbol: str) -> str:
    """``step_symbol`` itself, or a freshening of it against both diagrams' free symbols and
    ``index``."""
    avoid = _all_free_symbols(left) | _all_free_symbols(right) | {index}
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


def _offset(expr: sp.Expr, symbol: sp.Symbol) -> int | None:
    """``c`` when ``expr`` is ``symbol + c`` for an integer ``c``, else None."""
    rest = sp.expand(expr - symbol)
    return int(rest) if rest.is_Integer else None


def _least_offset(left: Diagram, right: Diagram, index: str) -> int:
    """The least ``c`` over the multiplicities ``index + c`` of both diagrams' boxes (0 when
    none)."""
    found = [0]
    for diagram in (left, right):
        for box in diagram.bang_boxes.values():
            expr = box.multiplicity.to_sympy()
            for symbol in expr.free_symbols:
                if str(symbol) == index and (c := _offset(expr, symbol)) is not None:
                    found.append(c)
    return min(found)


def rename_multiplicity(diagram: Diagram, index: str, replacement: Mult) -> Diagram:
    """Replace every live bang box's multiplicity ``index`` with ``replacement``, together, and
    ``index`` in the scalar and node phases, dropping any parameter binding for ``index``."""
    working = diagram.copy()
    for box_id, box in sorted(working.bang_boxes.items()):
        if index not in box.multiplicity.free_symbols:
            continue
        if box.multiplicity.is_bare_symbol and box.multiplicity.bare_symbol_name() == index:
            working.set_bang_box_multiplicity(box_id, replacement)
            continue
        expr = box.multiplicity.to_sympy()
        symbol = next(s for s in expr.free_symbols if str(s) == index)
        if _offset(expr, symbol) is None and not (
            expr.is_Pow and expr.base.is_Integer and expr.exp == symbol
        ):
            raise InductionDomainError(
                "induction requires a multiplicity n + c (c an integer) or a power b^n, got "
                f"{box.multiplicity!r}"
            )
        try:
            renamed = Mult._from_expr(
                sp.powsimp(sp.expand(expr.xreplace({symbol: replacement.to_sympy()})), force=True)
            )
        except BangBoxError as exc:
            raise InductionDomainError(
                f"multiplicity {box.multiplicity!r} does not take {index} = {replacement}: {exc}"
            ) from exc
        working.set_bang_box_multiplicity(box_id, renamed)
    if index in working.scalar.free_symbols:
        working.set_scalar(working.scalar.substitute({index: Scalar(replacement.to_sympy())}))
    for box_id, box in sorted(working.bang_boxes.items()):
        copy_scalar = box.copy_scalar
        if copy_scalar is not None and index in copy_scalar.free_symbols:
            working.set_bang_box_extension(
                box_id,
                copy_scalar=copy_scalar.substitute({index: Scalar(replacement.to_sympy())}),
            )
    working = substitute_index_phases(working, index, replacement)
    value = replacement.to_sympy()
    for node_id, node in sorted(working.nodes.items()):
        if any(index in p.dim.free_symbols for p in (*node.inputs, *node.outputs)):
            working.set_port_dims(
                node_id,
                [_dim_at(p.dim, index, value) for p in node.inputs],
                [_dim_at(p.dim, index, value) for p in node.outputs],
            )
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
    base_proved: bool = False,
) -> InductionObligation:
    """Construct and validate the base, hypothesis and successor diagrams for one index;
    ``base_proved`` builds the base diagrams with only the index instantiated, symbols the
    witness leaves out held formal, and no numeric check at the base."""
    idx = choose_index(left, right, index=index)
    held = (free_mult_symbols(left) | free_mult_symbols(right)) - {idx}
    required = (_all_free_symbols(left) | _all_free_symbols(right)) - {idx} - held
    supplied = set(witness)
    missing = required - supplied
    if missing and not base_proved:
        raise InductionGrammarError(
            f"induction over {idx!r} leaves symbol(s) uninstantiated: {sorted(missing)}; "
            "name them in the witness to hold them at a value"
        )
    if base is not None and (isinstance(base, bool) or base not in (0, 1)):
        raise InductionDomainError(f"base index must be 0 or 1, got {base!r}")
    if base_proved and base is None:
        raise InductionGrammarError("a proved base needs its base value")
    if base is not None and not base_proved:
        try:
            compare(
                left,
                right,
                {**witness, idx: base},
                mode=mode,
                tolerance=tolerance,
                max_elements=max_elements,
            )
        except (BangBoxError, CheckError, DenoteError, ContractError, ValidateError) as exc:
            raise InductionDomainError(f"base {base} has no evaluable instance: {exc}") from exc
    if base is None:
        base = choose_base_value(
            left, right, idx, witness, mode=mode, tolerance=tolerance, max_elements=max_elements
        )
    if base + _least_offset(left, right, idx) < 0:
        raise InductionDomainError(
            f"a box of multiplicity {idx} + c is negative at the base {idx} = {base}"
        )
    step = _fresh_step_symbol(left, right, idx, step_symbol)
    obligation = InductionObligation(
        index=idx,
        base_value=base,
        step_symbol=step,
        left_at_base=_at_base(left, idx, base, witness, base_proved),
        right_at_base=_at_base(right, idx, base, witness, base_proved),
        left_at_k=hypothesis_diagram(left, idx, step_symbol=step),
        right_at_k=hypothesis_diagram(right, idx, step_symbol=step),
        left_at_successor=successor_diagram(left, idx, step_symbol=step),
        right_at_successor=successor_diagram(right, idx, step_symbol=step),
        witness=MappingProxyType(dict(witness)),
        held_symbolic=required | held,
        base_proved=base_proved,
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


def _at_base(
    diagram: Diagram,
    index: str,
    value: int,
    witness: Mapping[str, CheckAssignmentValue],
    index_only: bool,
) -> Diagram:
    """``diagram`` at ``index = value``: with only the index set when ``index_only``, else
    with every symbol per ``witness``."""
    if not index_only:
        return instantiate(diagram, {**witness, index: value})
    from archytaszx.semantics.decide import _instantiate_index

    return _instantiate_index(diagram, index, value)


_VERDICT: dict[StepDischarge, Verdict] = {
    StepDischarge.UNIFORM_REWRITE: Verdict.PROVED_UNIFORM,
    StepDischarge.INDUCTION_REWRITE: Verdict.PROVED_INDUCTION,
    StepDischarge.SYMBOLIC_CONTRACTION: Verdict.PROVED_INDUCTION,
    StepDischarge.CHAIN_WINDOW: Verdict.PROVED_INDUCTION,
    StepDischarge.TREE_WINDOW: Verdict.PROVED_INDUCTION,
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
        if _extension(box_a) != _extension(box_b):
            return False
    return True


def _extension(box: BangBox) -> tuple[object, ...]:
    """A box's chain, index, copy-scalar, pair, tree, feed, align and group fields."""
    return (
        box.links,
        box.cyclic,
        box.index,
        box.copy_scalar,
        box.pair_of,
        box.pair_index,
        sorted(box.left_ports, key=_port_scope_key),
        sorted(box.right_ports, key=_port_scope_key),
        box.tree,
        box.tree_parent,
        box.level,
        box.position,
        box.feed,
        box.feed_port,
        box.align,
        tuple(tuple(sorted(g)) for g in box.groups),
    )


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
    :meth:`~archytaszx.rewrite.rule.Pattern.find_matches` returns them; the first applicable
    pair fires. ``guard_symbols``, when given, rejects a step whose result changes
    :func:`~archytaszx.diagram.bangbox.free_mult_symbols` away from it. A
    :class:`~archytaszx.rewrite.rule.RewriteError` from :func:`~archytaszx.rewrite.engine.apply`
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


def _box_footprint(box: BangBox) -> frozenset[NodeId]:
    """The nodes a box's scope touches."""
    return box.node_scope | frozenset(ref.node_id for ref in box.port_scope)


def _extract(diagram: Diagram, node_ids: frozenset[NodeId], *, with_boxes: bool = False) -> Diagram:
    """The sub-diagram on ``node_ids``: their wires among themselves, their boundary refs in
    the enclosing order, and with ``with_boxes`` every box lying wholly inside them (aligned
    only to a box also inside) and the part inside of each plain straddling box; raises
    InductionDomainError for any other box whose footprint meets ``node_ids``."""
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
    if with_boxes:
        box_map: dict[BangBoxId, BangBoxId] = {}
        inside = [
            (
                box_id,
                box
                if _box_footprint(box) <= node_ids
                else box.with_node_scope(part)
                if _plain_straddle(diagram, box)
                else _group_part(box, part),
            )
            for box_id, box in sorted(diagram.bang_boxes.items())
            for part in (box.node_scope & node_ids,)
            if _box_footprint(box) <= node_ids
            or (
                part
                and (_plain_straddle(diagram, box) or _grouped_straddle(diagram, box, node_ids))
            )
        ]
        inside_ids = {box_id for box_id, _ in inside}
        for box_id, box in diagram.bang_boxes.items():
            if box_id not in inside_ids and _box_footprint(box) & node_ids:
                raise InductionDomainError(f"bang box {box_id!r} lies partly inside the region")
        for _box_id, box in inside:
            if box.pair_of is not None and box.pair_of not in inside_ids:
                raise InductionDomainError("a pair box lies inside a region its parent leaves")
        while len(box_map) < len(inside):
            for box_id, box in inside:
                parent = box.parent if box.parent in inside_ids else None
                if box_id in box_map or (parent is not None and parent not in box_map):
                    continue
                if box.pair_of is not None and box.pair_of not in box_map:
                    continue
                if box.align in inside_ids and box.align not in box_map:
                    continue
                box_map[box_id] = add_box_image(
                    extracted,
                    box if box.align in inside_ids else replace(box, align=None),
                    id_map,
                    None if parent is None else box_map[parent],
                    None if box.pair_of is None else box_map[box.pair_of],
                    align=box_map.get(box.align) if box.align is not None else None,
                )
    extracted.set_parameters(dict(diagram.parameters))
    return extracted


def connected_regions(diagram: Diagram) -> tuple[frozenset[NodeId], ...]:
    """The node sets closed under wires and bang-box footprints, smallest id first."""
    parent: dict[NodeId, NodeId] = {node_id: node_id for node_id in diagram.nodes}

    def root(node_id: NodeId) -> NodeId:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def join(a: NodeId, b: NodeId) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for wire in diagram.wires:
        join(wire.a.node_id, wire.b.node_id)
    for box in diagram.bang_boxes.values():
        footprint = sorted(_box_footprint(box))
        for node_id in footprint[1:]:
            join(footprint[0], node_id)
    groups: dict[NodeId, set[NodeId]] = {}
    for node_id in diagram.nodes:
        groups.setdefault(root(node_id), set()).add(node_id)
    return tuple(frozenset(group) for _, group in sorted(groups.items()))


def _bare(diagram: Diagram) -> Diagram:
    """``diagram`` with scalar one and no parameter environment."""
    working = diagram.copy()
    working.set_scalar(Scalar.one())
    working.set_parameters({})
    return working


def find_hypothesis_region(
    diagram: Diagram, hypothesis: Diagram, *, max_regions: int = 12
) -> frozenset[NodeId] | None:
    """The smallest union of :func:`connected_regions` whose extraction is isomorphic to
    ``hypothesis`` up to scalar and spider leg order, or None; single regions only past
    ``max_regions``."""
    from itertools import combinations

    regions = connected_regions(diagram)
    target = _bare(hypothesis)
    wanted = sorted(node.generator_type.name for node in target.nodes.values())
    sizes = range(1, len(regions) + 1) if len(regions) <= max_regions else range(1, 2)
    for size in sizes:
        for chosen in combinations(regions, size):
            region = frozenset().union(*chosen)
            if len(region) != len(target.nodes):
                continue
            names = sorted(diagram.nodes[n].generator_type.name for n in region)
            if names != wanted:
                continue
            try:
                extracted = _bare(_extract(diagram, region, with_boxes=True))
            except InductionDomainError:
                continue
            if isomorphic(extracted, target, symmetric_legs=True):
                return region
    return None


def replace_region(
    diagram: Diagram, region: frozenset[NodeId], hypothesis: Diagram, replacement: Diagram
) -> Diagram:
    """``diagram`` with ``region`` swapped for ``replacement`` in its boundary slots, the scalar
    times ``replacement.scalar / hypothesis.scalar``."""
    if len(replacement.boundary_inputs) != sum(
        ref.node_id in region for ref in diagram.boundary_inputs
    ) or len(replacement.boundary_outputs) != sum(
        ref.node_id in region for ref in diagram.boundary_outputs
    ):
        raise InductionDomainError("the replacement's interface does not fit the region's slots")
    layout = boundary_layout(hypothesis)
    if layout is None or layout != boundary_layout(replacement):
        raise InductionDomainError("the replacement's boundary layout differs from the region's")
    if boundary_layout(diagram) is None:
        raise InductionDomainError("a box's boundary slots are not contiguous")
    working = diagram.copy()
    inputs = list(working.boundary_inputs)
    outputs = list(working.boundary_outputs)
    inside = [box_id for box_id, box in working.bang_boxes.items() if _box_footprint(box) & region]
    for box_id in inside:
        if not _box_footprint(working.bang_boxes[box_id]) <= region:
            raise InductionDomainError(f"bang box {box_id!r} straddles the replaced region")
    for box_id in sorted(inside, reverse=True):
        working.remove_bang_box(box_id)
    for node_id in sorted(region):
        working.remove_node(node_id)
    id_map = _graft(working, replacement)

    def refill(slots: list[PortRef], fresh: tuple[PortRef, ...]) -> list[PortRef]:
        queue = [PortRef(id_map[r.node_id], r.direction, r.index) for r in fresh]
        return [queue.pop(0) if ref.node_id in region else ref for ref in slots]

    working.set_boundary_inputs(refill(inputs, replacement.boundary_inputs))
    working.set_boundary_outputs(refill(outputs, replacement.boundary_outputs))
    working.multiply_scalar(replacement.scalar * hypothesis.scalar**-1)
    return working


def _residual_is(
    peeled: Diagram, copy_node_ids: frozenset[NodeId], expected: Diagram, index: str = ""
) -> bool:
    """Whether ``peeled`` minus its peeled copy is ``expected``, field for field, no box carrying
    ``index`` a chain."""
    kept = frozenset(peeled.nodes) - copy_node_ids
    if kept != frozenset(expected.nodes):
        return False
    residual = _extract(peeled, kept)
    reference = _extract(expected, frozenset(expected.nodes))
    if not compare_structure(residual, reference).identical:
        return False
    if any(
        _box_footprint(box) & copy_node_ids and not _box_footprint(box) <= copy_node_ids
        for box in peeled.bang_boxes.values()
    ):
        return False
    peeled_boxes = {
        box_id: box
        for box_id, box in peeled.bang_boxes.items()
        if not _box_footprint(box) & copy_node_ids
    }
    if sorted(peeled_boxes) != sorted(expected.bang_boxes):
        return False
    for box_id, box in peeled_boxes.items():
        other = expected.bang_boxes[box_id]
        if box.multiplicity != other.multiplicity or box.parent != other.parent:
            return False
        owns = not index or index in box.multiplicity.free_symbols
        if (box.links and owns) or _extension(box) != _extension(other):
            return False
        if sorted(box.node_scope) != sorted(other.node_scope):
            return False
        if sorted(box.port_scope, key=_port_scope_key) != sorted(
            other.port_scope, key=_port_scope_key
        ):
            return False
    return True


def _peel_step_index(diagram: Diagram, index: str) -> tuple[Diagram, frozenset[NodeId]] | None:
    """Peel one copy off every box whose multiplicity carries ``index`` (:func:`peel_owners`)."""
    try:
        result = peel_owners(diagram, index)
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


def _boundary_positions(
    diagram: Diagram, node_ids: frozenset[NodeId]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Which slots of each boundary list ``node_ids`` occupies."""
    inputs = tuple(
        position for position, ref in enumerate(diagram.boundary_inputs) if ref.node_id in node_ids
    )
    outputs = tuple(
        position for position, ref in enumerate(diagram.boundary_outputs) if ref.node_id in node_ids
    )
    return inputs, outputs


def _interface(diagram: Diagram) -> tuple[tuple[Dim, ...], tuple[Dim, ...]]:
    """The dimensions of a diagram's boundary inputs and outputs, in order."""
    return (
        tuple(_ref_dim(diagram, ref) for ref in diagram.boundary_inputs),
        tuple(_ref_dim(diagram, ref) for ref in diagram.boundary_outputs),
    )


def _ref_dim(diagram: Diagram, ref: PortRef) -> Dim:
    """The dimension carried by one boundary port."""
    return diagram.nodes[ref.node_id].legs(ref.direction)[ref.index].dim


def discharge_peeled_hypothesis(
    obligation: InductionObligation, *, max_steps: int = DEFAULT_MAX_STEPS
) -> TierOutcome:
    """Settle the step case by peeling one copy off each successor and discharging the rest.

    ``!_{k+1}(G)`` splits into ``!_k(G)`` beside one bare copy of ``G``
    (:func:`~archytaszx.diagram.bangbox.peel_one`). When each peeled residual is exactly its own
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

    if not _residual_is(left_peeled, left_copy, obligation.left_at_k, index):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the left successor's peeled residual is not the left hypothesis diagram",
        )
    if not _residual_is(right_peeled, right_copy, obligation.right_at_k, index):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the right successor's peeled residual is not the right hypothesis diagram",
        )

    # The reduction is "residual tensored with copy" on both sides, so the copy must sit in
    # the same boundary slots on each: otherwise the two sides differ by a permutation the
    # hypothesis says nothing about.
    if _boundary_positions(left_peeled, left_copy) != _boundary_positions(right_peeled, right_copy):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the peeled copies occupy different boundary slots on the two sides, so the two "
            "successors differ by a permutation the hypothesis does not settle",
        )

    if left_peeled.scalar != right_peeled.scalar:
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            f"the two successors carry different scalars, {left_peeled.scalar} and "
            f"{right_peeled.scalar}",
        )

    left_extra = _extract(left_peeled, left_copy, with_boxes=True)
    right_extra = _extract(right_peeled, right_copy, with_boxes=True)

    # compare_symbolic sees a rank and the axis dimensions, not which axes are inputs and
    # which outputs, so the interfaces are compared here before contracting.
    if _interface(left_extra) != _interface(right_extra):
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            False,
            "the peeled copies do not share one boundary interface: "
            f"{_interface(left_extra)} against {_interface(right_extra)}",
        )
    if compare_structure(left_extra, right_extra).identical:
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            True,
            "peeled one copy off each successor; the residuals are the hypothesis diagrams and "
            "the peeled copies are structurally identical",
            left_peeled=left_peeled,
            left_copy=left_copy,
            right_peeled=right_peeled,
            right_copy=right_copy,
        )
    lemma = TierOutcome(
        StepDischarge.INDUCTION_REWRITE,
        True,
        "peeled one copy off each successor; the residuals are the hypothesis diagrams and "
        "the peeled copies are equal by a nested decision (lemma)",
        left_peeled=left_peeled,
        left_copy=left_copy,
        right_peeled=right_peeled,
        right_copy=right_copy,
    )
    try:
        contracted_left, contracted_right = contract_symbolic_pair(
            left_extra, right_extra, max_steps=max_steps
        )
    except (
        SymbolicContractionUnsupportedError,
        SymbolicContractionDomainError,
        ScalarBudgetError,
    ) as exc:
        if lemma_decision(left_extra, right_extra) is not None:
            return lemma
        return TierOutcome(StepDischarge.INDUCTION_REWRITE, False, str(exc), ())

    comparison = compare_symbolic(contracted_left, contracted_right)
    if not comparison.matched and lemma_decision(left_extra, right_extra) is not None:
        return lemma
    if comparison.matched:
        return TierOutcome(
            StepDischarge.INDUCTION_REWRITE,
            True,
            "peeled one copy off each successor; the residuals are the hypothesis diagrams and "
            "the peeled copies agree under symbolic contraction with d formal",
            (comparison,),
            left_peeled=left_peeled,
            left_copy=left_copy,
            right_peeled=right_peeled,
            right_copy=right_copy,
        )
    return TierOutcome(
        StepDischarge.INDUCTION_REWRITE,
        False,
        f"the peeled copies are not equal: {comparison.reason}",
        (comparison,),
    )


_MAX_CUT_PLACEMENTS = 2000


def _slot_owners(diagram: Diagram, ref: PortRef) -> tuple[BangBoxId, ...]:
    """The boxes replicating boundary slot ``ref``, outermost first: the node-scope boxes over
    its node (a chain box not when ``ref`` is one of its link ports), then the port-scope boxes
    holding it."""
    owners: list[BangBoxId] = []
    for box_id, box in diagram.bang_boxes.items():
        in_copy = ref.node_id in box.node_scope and not any(ref in link for link in box.links)
        if in_copy or ref in box.port_scope:
            owners.append(box_id)
    return tuple(sorted(owners, key=lambda b: (_box_depth(diagram, b), b)))


def _box_depth(diagram: Diagram, box_id: BangBoxId) -> int:
    """How many boxes enclose ``box_id``."""
    depth, parent = 0, diagram.bang_boxes[box_id].parent
    while parent is not None:
        depth, parent = depth + 1, diagram.bang_boxes[parent].parent
    return depth


def _end_dim(diagram: Diagram, ref: PortRef) -> Dim:
    """The dimension of boundary slot ``ref``; on an open link end of an indexed chain, read at
    the copy holding it (the last copy for a link's ``p`` port, the first for its ``q`` port)."""
    dim = _ref_dim(diagram, ref)
    for box in diagram.bang_boxes.values():
        if box.index is None or box.cyclic:
            continue
        for p_ref, q_ref in box.links:
            if ref not in (p_ref, q_ref):
                continue
            expr = dim.to_sympy()
            symbol = next((s for s in expr.free_symbols if s.name == box.index), None)
            if symbol is None:
                return dim
            at = sp.expand(box.multiplicity.to_sympy() - 1) if ref == p_ref else sp.Integer(0)
            return Dim.from_sympy(sp.expand(expr.xreplace({symbol: at})))
    return dim


def boundary_layout(diagram: Diagram) -> tuple[tuple[object, ...], ...] | None:
    """Per boundary list, each slot's replicating multiplicities and, per enclosing box, whether
    the slot opens a new run of that box; None when some box's slots in one list are not
    contiguous."""
    out: list[tuple[object, ...]] = []
    for refs in (diagram.boundary_inputs, diagram.boundary_outputs):
        rows: list[object] = []
        closed: set[BangBoxId] = set()
        previous: tuple[BangBoxId, ...] = ()
        for ref in refs:
            owners = _slot_owners(diagram, ref)
            for box_id in previous:
                if box_id not in owners:
                    closed.add(box_id)
            if any(box_id in closed for box_id in owners):
                return None
            mults = tuple(diagram.bang_boxes[b].multiplicity for b in owners)
            opens = tuple(b not in previous for b in owners)
            rows.append((mults, opens, _end_dim(diagram, ref)))
            previous = owners
        out.append(tuple(rows))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CutRegion:
    """A region of a diagram read as an open diagram: its boundary lists, each slot either one
    of the diagram's boundary slots on the region or the region's end of a cut wire."""

    region: frozenset[NodeId]
    inputs: tuple[PortRef, ...]
    outputs: tuple[PortRef, ...]
    cuts: Mapping[PortRef, PortRef]


def _cut_wires(diagram: Diagram, region: frozenset[NodeId]) -> dict[PortRef, PortRef]:
    """Each wire leaving ``region``, as its region end to its far end."""
    out: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        if (wire.a.node_id in region) != (wire.b.node_id in region):
            inside, far = (wire.a, wire.b) if wire.a.node_id in region else (wire.b, wire.a)
            out[inside] = far
    return out


def _cut_extract(diagram: Diagram, cut: CutRegion) -> Diagram:
    """The region of ``cut`` as a diagram with ``cut``'s boundary lists, scalar one."""
    extracted = _extract(diagram, cut.region, with_boxes=True)
    order = sorted(cut.region)
    id_map = {old: NodeId(new) for old, new in zip(order, sorted(extracted.nodes), strict=True)}
    extracted.set_boundary_inputs([_moved_ref(r, id_map) for r in cut.inputs])
    extracted.set_boundary_outputs([_moved_ref(r, id_map) for r in cut.outputs])
    return extracted


def _moved_ref(ref: PortRef, id_map: Mapping[NodeId, NodeId]) -> PortRef:
    """``ref`` on its node's image."""
    return PortRef(id_map[ref.node_id], ref.direction, ref.index)


def find_cut_region(
    diagram: Diagram, region: frozenset[NodeId], hypothesis: Diagram
) -> CutRegion | None:
    """A placement of ``region``'s cut wires among its boundary slots under which the region
    is isomorphic to ``hypothesis`` up to scalar: each cut end takes the place of a slot of
    ``diagram`` held off the region, or follows the region's slots, per direction; or None."""
    from itertools import permutations

    cuts = _cut_wires(diagram, region)
    target = _bare(hypothesis)
    lists = (list(diagram.boundary_inputs), list(diagram.boundary_outputs))
    wanted = (len(hypothesis.boundary_inputs), len(hypothesis.boundary_outputs))
    per_list: list[list[tuple[PortRef, ...]]] = []
    for side, (refs, want) in enumerate(zip(lists, wanted, strict=True)):
        direction = (Direction.INPUT, Direction.OUTPUT)[side]
        own = [cut for cut in sorted(cuts, key=PortRef.sort_key) if cut.direction is direction]
        kept = [i for i, r in enumerate(refs) if r.node_id in region]
        if len(kept) + len(own) != want:
            return None
        holes = [i for i, r in enumerate(refs) if r.node_id not in region]
        spots = [*holes, *(len(refs) + j for j in range(len(own)))]
        options: list[tuple[PortRef, ...]] = []
        for chosen in permutations(spots, len(own)):
            placed = dict(zip(chosen, own, strict=True))
            options.append(
                tuple(placed[i] if i in placed else refs[i] for i in sorted({*kept, *chosen}))
            )
            if len(options) > _MAX_CUT_PLACEMENTS:
                return None
        per_list.append(options)
    for inputs in per_list[0]:
        for outputs in per_list[1]:
            cut = CutRegion(region, inputs, outputs, MappingProxyType(cuts))
            try:
                extracted = _cut_extract(diagram, cut)
            except InductionDomainError:
                return None
            if isomorphic(_bare(extracted), target):
                return cut
    return None


def replace_cut_region(
    diagram: Diagram,
    cut: CutRegion,
    hypothesis: Diagram,
    replacement: Diagram,
    *,
    contiguous: bool = True,
) -> Diagram:
    """``diagram`` with ``cut``'s region swapped for ``replacement``: slot ``i`` of each of
    ``replacement``'s boundary lists takes slot ``i`` of ``cut``'s, a cut end's far port wired
    to it, the scalar times ``replacement.scalar / hypothesis.scalar``, a box off the region
    aligned to one in it aligned with the grafted box its cut wires reach (:func:`_realign`);
    ``contiguous=False`` accepts a ``diagram`` whose box slots are not contiguous."""
    if (len(replacement.boundary_inputs), len(replacement.boundary_outputs)) != (
        len(cut.inputs),
        len(cut.outputs),
    ):
        raise InductionDomainError("the replacement's interface does not fit the cut region")
    layout = boundary_layout(hypothesis)
    if layout is None or layout != boundary_layout(replacement):
        raise InductionDomainError("the replacement's boundary layout differs from the region's")
    if contiguous and boundary_layout(diagram) is None:
        raise InductionDomainError("a box's boundary slots are not contiguous")
    working = diagram.copy()
    inside = [b for b, box in working.bang_boxes.items() if _box_footprint(box) & cut.region]
    outside: dict[BangBoxId, BangBox] = {}
    for box_id in inside:
        box = working.bang_boxes[box_id]
        if not _box_footprint(box) <= cut.region:
            if not (
                _straddle_merges(working, box, cut.region)
                or _grouped_straddle(working, box, cut.region)
            ):
                raise InductionDomainError(f"bang box {box_id!r} straddles the replaced region")
            outside[box_id] = box
    partners = _outside_partners(working, frozenset(inside))
    for members, _root in partners:
        for box_id in members:
            working.set_bang_box_extension(box_id, align=None)
    for box_id in sorted(inside, reverse=True):
        working.remove_bang_box(box_id)
    slot_of: dict[PortRef, PortRef] = {}
    box_count = len(working.bang_boxes)
    fresh = _graft(working, replacement)
    grafted = sorted(working.bang_boxes)[box_count:]
    for mine, theirs in (
        (cut.inputs, replacement.boundary_inputs),
        (cut.outputs, replacement.boundary_outputs),
    ):
        for ref, other in zip(mine, theirs, strict=True):
            slot_of[ref] = _moved_ref(other, fresh)
    inputs = [slot_of.get(r, r) for r in working.boundary_inputs]
    outputs = [slot_of.get(r, r) for r in working.boundary_outputs]
    for node_id in sorted(cut.region):
        working.remove_node(node_id)
    for inside_end, far in sorted(cut.cuts.items(), key=lambda item: item[0].sort_key()):
        working.add_wire(slot_of[inside_end], far)
    for box in outside.values():
        if box.groups:
            _merge_grouped(working, box, cut, slot_of, grafted)
        else:
            _merge_straddle(working, box, cut, slot_of, grafted)
    for members, root in partners:
        _realign(working, members, root, cut, slot_of, grafted)
    working.set_boundary_inputs(inputs)
    working.set_boundary_outputs(outputs)
    working.multiply_scalar(replacement.scalar * hypothesis.scalar**-1)
    return working


def _outside_partners(
    diagram: Diagram, inside: frozenset[BangBoxId]
) -> list[tuple[list[BangBoxId], BangBoxId | None]]:
    """Per alignment group with boxes both in ``inside`` and off it: its boxes off ``inside``
    and its root when that is off ``inside`` (else None)."""
    out: dict[BangBoxId, tuple[list[BangBoxId], BangBoxId | None]] = {}
    for box_id in sorted(inside):
        group = aligned_members(diagram, box_id)
        members = [b for b in group if b not in inside]
        if len(group) > 1 and members:
            out[group[0]] = (members, None if group[0] in inside else group[0])
    return list(out.values())


def _ancestors(diagram: Diagram, box_id: BangBoxId) -> frozenset[BangBoxId]:
    """The boxes enclosing ``box_id``."""
    out: set[BangBoxId] = set()
    parent = diagram.bang_boxes[box_id].parent
    while parent is not None:
        out.add(parent)
        parent = diagram.bang_boxes[parent].parent
    return frozenset(out)


def _realign(
    working: Diagram,
    members: Sequence[BangBoxId],
    root: BangBoxId | None,
    cut: CutRegion,
    slot_of: Mapping[PortRef, PortRef],
    grafted: Sequence[BangBoxId],
) -> None:
    """Align ``members`` (boxes off ``cut``'s region, ``root`` their old root when off it) with
    the one grafted box their cut wires reach, and with its alignment group; with no such wire,
    with each other only."""
    scopes = {n for m in members for n in working.bang_boxes[m].node_scope}
    targets: set[BangBoxId] = set()
    for inside_end, far in cut.cuts.items():
        if far.node_id not in scopes:
            continue
        end = slot_of[inside_end].node_id
        owners = [b for b in grafted if end in working.bang_boxes[b].node_scope]
        if len(owners) != 1:
            raise InductionDomainError(f"aligned boxes {list(members)!r} meet no single box")
        targets.add(owners[0])
    if len(targets) > 1:
        raise InductionDomainError(f"aligned boxes {list(members)!r} meet several boxes")
    group = list(members)
    new_root = root if root is not None else min(members)
    if targets:
        (target,) = targets
        image = working.bang_boxes[target]
        image_group = aligned_members(working, target if image.align is None else image.align)
        group += image_group
        if root is None:
            new_root = image_group[0]
    mult = working.bang_boxes[new_root].multiplicity
    for box_id in group:
        box = working.bang_boxes[box_id]
        if box.multiplicity != mult or not _ancestors(working, new_root) <= _ancestors(
            working, box_id
        ):
            raise InductionDomainError(f"bang box {box_id!r} cannot align with {new_root!r}")
        if not box.is_node_scope or box.is_pair or box.is_tree or box.is_fed:
            raise InductionDomainError(f"bang box {box_id!r} cannot align with {new_root!r}")
    for box_id in group:
        working.set_bang_box_extension(box_id, align=None if box_id == new_root else new_root)


def _plain_straddle(diagram: Diagram, box: BangBox) -> bool:
    """Whether ``box`` is a top-level node-scope box with no links, copy scalar, tree, feed,
    pairing, alignment or groups, aligned to by no box."""
    return not (
        not box.node_scope
        or box.parent is not None
        or box.links
        or box.copy_scalar is not None
        or box.is_tree
        or box.is_fed
        or box.is_pair
        or box.align is not None
        or box.groups
        or any(other.align == box.id for other in diagram.bang_boxes.values())
    )


def _blocks(box: BangBox) -> list[frozenset[NodeId]]:
    """``box``'s groups, then its ungrouped scope nodes when there are any."""
    grouped = frozenset(n for g in box.groups for n in g)
    rest = box.node_scope - grouped
    return [*box.groups, *([rest] if rest else [])]


def _grouped_straddle(diagram: Diagram, box: BangBox, region: frozenset[NodeId]) -> bool:
    """Whether ``box``, partly in ``region``, is a top-level node-scope box with groups and no
    child, index, copy scalar, tree, feed, pairing or alignment, each of its blocks
    (:func:`_blocks`) wholly in or wholly off ``region``, and each link's two ports and the
    ports they are wired to on one side."""
    if not box.groups or not box.node_scope or box.parent is not None:
        return False
    if box.index is not None or box.copy_scalar is not None or box.align is not None:
        return False
    if box.is_tree or box.is_fed or box.is_pair or box.cyclic:
        return False
    for other in diagram.bang_boxes.values():
        if other.id != box.id and box.id in (other.parent, other.align, other.pair_of, other.feed):
            return False
    if any(bool(g & region) and not g <= region for g in _blocks(box)):
        return False
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    for link in box.links:
        ends = [*link, *(partner[r] for r in link if r in partner)]
        if len({r.node_id in region for r in ends}) != 1:
            return False
    return True


def _group_part(box: BangBox, part: frozenset[NodeId]) -> BangBox:
    """``box`` cut to ``part``: its links and groups within ``part``, no groups when one is
    left."""
    groups = tuple(g for g in box.groups if g <= part)
    links = tuple(link for link in box.links if link[0].node_id in part)
    return replace(box, node_scope=part, links=links, groups=groups if len(groups) > 1 else ())


def _merge_grouped(
    working: Diagram,
    box: BangBox,
    cut: CutRegion,
    slot_of: Mapping[PortRef, PortRef],
    grafted: Sequence[BangBoxId],
) -> None:
    """Put the part of grouped straddling ``box`` off ``cut``'s region (:func:`_grouped_straddle`)
    into the one top-level grafted box of ``box``'s multiplicity, with no index, copy scalar,
    alignment or child, that holds alone the slot of every cut wire from that part, each of
    that part's cut wires ending in ``box``: its blocks, then that part's blocks as groups,
    and that part's links."""
    rest = box.node_scope - cut.region
    targets: set[BangBoxId] = set()
    for inside_end, far in cut.cuts.items():
        if far.node_id not in rest:
            continue
        if inside_end.node_id not in box.node_scope:
            raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
        end = slot_of[inside_end].node_id
        owners = [b for b, other in working.bang_boxes.items() if end in other.node_scope]
        if len(owners) != 1 or owners[0] not in grafted:
            raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
        targets.add(owners[0])
    if len(targets) != 1:
        raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
    (target,) = targets
    into = working.bang_boxes[target]
    tied = any(
        target in (other.parent, other.align, other.pair_of, other.feed)
        for other in working.bang_boxes.values()
        if other.id != target
    )
    if (
        into.multiplicity != box.multiplicity
        or into.parent is not None
        or into.index is not None
        or into.copy_scalar is not None
        or into.align is not None
        or into.is_tree
        or into.is_fed
        or into.is_pair
        or into.cyclic
        or into.sheets != box.sheets
        or tied
    ):
        raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
    moved = [g for g in _blocks(box) if g <= rest]
    links = tuple(link for link in box.links if link[0].node_id in rest)
    working.set_bang_box_extension(
        target,
        node_scope=into.node_scope | rest,
        links=(*into.links, *links),
        groups=(*_blocks(into), *moved),
    )


def _straddle_merges(diagram: Diagram, box: BangBox, region: frozenset[NodeId]) -> bool:
    """Whether ``box``, partly in ``region``, has a part off ``region`` that
    :func:`_merge_straddle` can put into a replacement box: :func:`_plain_straddle`, every box
    inside or paired with it within ``region``, its index read by no node off ``region``."""
    if not _plain_straddle(diagram, box):
        return False
    for other in diagram.bang_boxes.values():
        if other.id == box.id:
            continue
        tied = other.parent == box.id or other.pair_of == box.id or other.align == box.id
        if tied and not _box_footprint(other) <= region:
            return False
    if box.index is not None:
        for node_id in box.node_scope - region:
            if box.index in _node_symbols(diagram, node_id):
                return False
    return True


def _node_symbols(diagram: Diagram, node_id: NodeId) -> frozenset[str]:
    """The symbol names a node's leg dimensions and phase read."""
    node = diagram.nodes[node_id]
    names: set[str] = set()
    for port in (*node.inputs, *node.outputs):
        names |= port.dim.free_symbols
    if node.phase is not None:
        names |= node.phase.free_symbols
    return frozenset(names)


def _merge_straddle(
    working: Diagram,
    box: BangBox,
    cut: CutRegion,
    slot_of: Mapping[PortRef, PortRef],
    grafted: Sequence[BangBoxId],
) -> None:
    """Put each wire-connected part of the nodes of ``box`` off ``cut``'s region into the one
    grafted box of ``box``'s multiplicity that every cut wire from it reaches."""
    rest = box.node_scope - cut.region
    for part in _parts(working, rest):
        targets: set[BangBoxId] = set()
        for inside_end, far in cut.cuts.items():
            if far.node_id not in part:
                continue
            end = slot_of[inside_end].node_id
            owners = [b for b in grafted if end in working.bang_boxes[b].node_scope]
            top = [b for b in owners if working.bang_boxes[b].parent is None]
            if len(owners) != 1 or len(top) != 1:
                raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
            targets.add(top[0])
        if len(targets) != 1:
            raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
        (target,) = targets
        into = working.bang_boxes[target]
        if into.multiplicity != box.multiplicity or into.is_tree or into.is_fed or into.is_pair:
            raise InductionDomainError(f"bang box {box.id!r} straddles the replaced region")
        working.set_bang_box_node_scope(target, into.node_scope | part)


def _parts(diagram: Diagram, nodes: frozenset[NodeId]) -> list[frozenset[NodeId]]:
    """The parts of ``nodes`` connected by wires among them, smallest id first."""
    parts: list[frozenset[NodeId]] = []
    left = set(nodes)
    while left:
        todo, seen = [min(left)], set()
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            for wire in diagram.wires:
                for a, b in ((wire.a.node_id, wire.b.node_id), (wire.b.node_id, wire.a.node_id)):
                    if a == n and b in left and b not in seen:
                        todo.append(b)
        parts.append(frozenset(seen))
        left -= seen
    return parts


def _graft(working: Diagram, replacement: Diagram) -> dict[NodeId, NodeId]:
    """Add ``replacement``'s nodes, wires and boxes to ``working``; its node map."""
    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(replacement.nodes):
        node = replacement.nodes[old_id]
        id_map[old_id] = working.add_node(
            node.generator_type,
            [port.dim for port in node.inputs],
            [port.dim for port in node.outputs],
            phase=node.phase,
        )
    for wire in sorted(replacement.wires, key=lambda w: w.sort_key()):
        working.add_wire(_moved_ref(wire.a, id_map), _moved_ref(wire.b, id_map))
    box_map: dict[BangBoxId, BangBoxId] = {}
    pending = sorted(replacement.bang_boxes.items())
    while len(box_map) < len(pending):
        for box_id, box in pending:
            if box_id in box_map or (box.parent is not None and box.parent not in box_map):
                continue
            if box.pair_of is not None and box.pair_of not in box_map:
                continue
            if box.align is not None and box.align not in box_map:
                continue
            box_map[box_id] = add_box_image(
                working,
                box,
                id_map,
                None if box.parent is None else box_map[box.parent],
                None if box.pair_of is None else box_map[box.pair_of],
                align=None if box.align is None else box_map[box.align],
            )
    return id_map


def _fan_split(
    diagram: Diagram, copy: frozenset[NodeId]
) -> Iterator[tuple[Diagram, frozenset[NodeId]]]:
    """Per spider of ``copy`` wired to a boxed node off ``copy``, and per boundary slot it holds:
    ``diagram`` with the spider split into a hub off the copy (its phase and other legs, the
    slot's place wired to the leaf) and a phaseless leaf (the slot and the legs into ``copy``),
    and the copy with the leaf in the spider's place."""
    boxed = frozenset(n for box in diagram.bang_boxes.values() for n in box.node_scope)
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    slots = [*diagram.boundary_inputs, *diagram.boundary_outputs]
    for v in sorted(copy):
        node = diagram.nodes[v]
        if node.generator_type not in (Z_SPIDER, X_SPIDER):
            continue
        legs = [
            PortRef(v, direction, i)
            for direction in (Direction.INPUT, Direction.OUTPUT)
            for i in range(len(node.legs(direction)))
        ]
        far = {leg: partner.get(leg) for leg in legs}
        inward = [leg for leg, f in far.items() if f is not None and f.node_id in copy]
        if not inward or not any(
            f is not None and f.node_id not in copy and f.node_id in boxed for f in far.values()
        ):
            continue
        for slot in (leg for leg in legs if leg in slots):
            yield _split_spider(diagram, v, slot, inward, copy)
    yield from _gather_split(diagram, copy, boxed, partner, slots)
    yield from _boxed_split(diagram, copy, partner, slots)


def copy_candidates(
    peeled: Diagram, old: frozenset[NodeId], copy: frozenset[NodeId]
) -> list[frozenset[NodeId]]:
    """``copy``, then, when it differs, the nodes new to ``peeled`` (not in ``old``) in no box
    together with the new nodes of a box other than a pair box wired to one of them (a peeled
    copy's pair nodes placed in the residual box), then ``copy`` without its spiders wired to
    a node of ``old`` off it."""
    new = frozenset(peeled.nodes) - old
    boxed = frozenset(n for box in peeled.bang_boxes.values() for n in box.node_scope)
    paired = frozenset(
        n for box in peeled.bang_boxes.values() if box.is_pair for n in box.node_scope
    )
    bare = new - boxed
    extra = set()
    for wire in peeled.wires:
        for a, b in ((wire.a.node_id, wire.b.node_id), (wire.b.node_id, wire.a.node_id)):
            if a in new and a in boxed and a not in paired and b in bare:
                extra.add(a)
    out = [copy]
    if extra and bare | extra != copy:
        out.append(bare | frozenset(extra))
    regrown = set()
    for wire in peeled.wires:
        for a, b in ((wire.a.node_id, wire.b.node_id), (wire.b.node_id, wire.a.node_id)):
            spider = peeled.nodes[a].generator_type in (Z_SPIDER, X_SPIDER)
            if a in bare and spider and b in old:
                regrown.add(a)
    if extra and regrown:
        out.append((bare - frozenset(regrown)) | frozenset(extra))
    hubs = frozenset(
        a
        for wire in peeled.wires
        for a, b in ((wire.a.node_id, wire.b.node_id), (wire.b.node_id, wire.a.node_id))
        if a in copy and b in old - copy and peeled.nodes[a].generator_type in (Z_SPIDER, X_SPIDER)
    )
    if hubs and copy - hubs not in out:
        out.append(copy - hubs)
    return out


@dataclass(frozen=True, slots=True)
class ExposeStep:
    """One exposing step: ``after`` from ``before`` by a :func:`_fan_split` split (kind split)
    or by :func:`leg_moves` (kind move), with the copies before and after."""

    kind: str
    before: Diagram
    after: Diagram
    copy_before: frozenset[NodeId]
    copy_after: frozenset[NodeId]


def expose_cut(
    peeled: Diagram, copies: Sequence[frozenset[NodeId]], hypothesis: Diagram
) -> tuple[Diagram, CutRegion, tuple[ExposeStep, ...]] | None:
    """The first of ``copies`` whose complement in ``peeled``, in a :func:`_fan_split` of it,
    or after all :func:`_greedy_steps`, then after their shorter prefixes, reads as
    ``hypothesis`` through cut wires: that diagram, the cut and the steps to it; else None."""
    for copy in copies:
        cut = find_cut_region(peeled, frozenset(peeled.nodes) - copy, hypothesis)
        if cut is not None:
            return peeled, cut, ()
        for split, split_copy in _fan_split(peeled, copy):
            cut = find_cut_region(split, frozenset(split.nodes) - split_copy, hypothesis)
            if cut is not None:
                return split, cut, (ExposeStep("split", peeled, split, copy, split_copy),)
    for copy in copies[1:]:
        steps = _greedy_steps(peeled, copy)
        for end in (len(steps), *range(1, len(steps))) if steps else ():
            last = steps[end - 1]
            region = frozenset(last.after.nodes) - last.copy_after
            cut = find_cut_region(last.after, region, hypothesis)
            if cut is not None:
                return last.after, cut, steps[:end]
    return None


_MAX_GREEDY_STEPS = 16


def _greedy_steps(peeled: Diagram, copy: frozenset[NodeId]) -> tuple[ExposeStep, ...]:
    """Steps from ``peeled``: every boxed split (:func:`boxed_splits`) in turn, then every leg
    move (:func:`leg_moves`), then every unboxed split at a slot (:func:`_gather_split`)."""
    steps: list[ExposeStep] = []
    current, held = peeled, copy
    for stage in ("boxed", "move", "gather"):
        while len(steps) < _MAX_GREEDY_STEPS:
            if stage == "boxed":
                found = next(boxed_splits(current, held), None)
            elif stage == "move":
                moved = next(leg_moves(current, held), None)
                found = None if moved is None else (moved, held)
            else:
                found = next(_gather_splits(current, held), None)
            if found is None:
                break
            kind = "move" if stage == "move" else "split"
            steps.append(ExposeStep(kind, current, found[0], held, found[1]))
            current, held = found
    return tuple(steps)


def _gather_splits(
    diagram: Diagram, copy: frozenset[NodeId]
) -> Iterator[tuple[Diagram, frozenset[NodeId]]]:
    """:func:`_gather_split` of ``diagram`` and ``copy``."""
    boxed = frozenset(n for box in diagram.bang_boxes.values() for n in box.node_scope)
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    slots = [*diagram.boundary_inputs, *diagram.boundary_outputs]
    yield from _gather_split(diagram, copy, boxed, partner, slots)


def _box_referenced(diagram: Diagram) -> frozenset[NodeId]:
    """Nodes whose ports a box names (links, pair, tree, feed or port-scope ports) or that a
    box holds."""
    out: set[NodeId] = set()
    for box in diagram.bang_boxes.values():
        out |= box.node_scope
        refs = [*box.port_scope, *box.left_ports, *box.right_ports, *box.tree]
        refs += [r for link in box.links for r in link]
        refs += [r for r in (box.tree_parent, box.feed_port) if r is not None]
        out |= {r.node_id for r in refs}
    return frozenset(out)


def leg_moves(diagram: Diagram, copy: frozenset[NodeId]) -> Iterator[Diagram]:
    """Per unboxed spider ``v`` off ``copy`` holding no boundary slot, with legs into ``copy``,
    and per unboxed spider ``w`` of its type off ``copy`` holding a slot and wired to it:
    ``diagram`` with those legs moved from ``v`` onto new legs of ``w`` (spider fusion and
    unfusion along the wire)."""
    fixed = _box_referenced(diagram)
    slots = {r.node_id for r in (*diagram.boundary_inputs, *diagram.boundary_outputs)}
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    for v in sorted(diagram.nodes):
        node = diagram.nodes[v]
        if v in copy or v in fixed or v in slots or node.generator_type not in (Z_SPIDER, X_SPIDER):
            continue
        legs = [
            PortRef(v, d, i)
            for d in (Direction.INPUT, Direction.OUTPUT)
            for i in range(len(node.legs(d)))
        ]
        inward = [leg for leg in legs if (f := partner.get(leg)) is not None and f.node_id in copy]
        if not inward:
            continue
        for leg in legs:
            far = partner.get(leg)
            if far is None or far.node_id in copy or far.node_id in fixed:
                continue
            if far.node_id not in slots:
                continue
            other = diagram.nodes[far.node_id]
            if far.node_id == v or other.generator_type is not node.generator_type:
                continue
            yield _move_legs(diagram, v, inward, far.node_id)


def _move_legs(diagram: Diagram, v: NodeId, legs: Sequence[PortRef], w: NodeId) -> Diagram:
    """``diagram`` with ``legs`` of ``v`` removed and added, in their directions, after
    ``w``'s legs, each wired to the same far end."""
    working = diagram.copy()
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    gone = frozenset(legs)
    for leg in legs:
        working.remove_wire(leg, partner[leg])
    node, target = diagram.nodes[v], diagram.nodes[w]
    kept = {
        d: [i for i in range(len(node.legs(d))) if PortRef(v, d, i) not in gone]
        for d in (Direction.INPUT, Direction.OUTPUT)
    }
    new_v = working.add_node(
        node.generator_type,
        [node.inputs[i].dim for i in kept[Direction.INPUT]],
        [node.outputs[i].dim for i in kept[Direction.OUTPUT]],
        phase=node.phase,
    )
    port_map = {PortRef(v, d, i): PortRef(new_v, d, j) for d in kept for j, i in enumerate(kept[d])}
    grown = {
        d: [p.dim for p in target.legs(d)]
        + [node.legs(d)[r.index].dim for r in legs if r.direction is d]
        for d in (Direction.INPUT, Direction.OUTPUT)
    }
    new_w = working.add_node(
        target.generator_type, grown[Direction.INPUT], grown[Direction.OUTPUT], phase=target.phase
    )
    for d in (Direction.INPUT, Direction.OUTPUT):
        for i in range(len(target.legs(d))):
            port_map[PortRef(w, d, i)] = PortRef(new_w, d, i)
    added = {d: len(target.legs(d)) for d in (Direction.INPUT, Direction.OUTPUT)}
    for leg in legs:
        port_map[leg] = PortRef(new_w, leg.direction, added[leg.direction])
        added[leg.direction] += 1
    for wire in sorted(working.wires, key=lambda x: x.sort_key()):
        if {wire.a.node_id, wire.b.node_id} & {v, w}:
            working.remove_wire(wire.a, wire.b)
            working.add_wire(port_map.get(wire.a, wire.a), port_map.get(wire.b, wire.b))
    for leg in legs:
        working.add_wire(port_map[leg], partner[leg])
    working.set_boundary_inputs([port_map.get(r, r) for r in working.boundary_inputs])
    working.set_boundary_outputs([port_map.get(r, r) for r in working.boundary_outputs])
    working.remove_node(v)
    working.remove_node(w)
    return working


def _box_ports(diagram: Diagram) -> frozenset[PortRef]:
    """The ports a box names: links, pair, tree, feed and port-scope ports."""
    out: set[PortRef] = set()
    for box in diagram.bang_boxes.values():
        out |= {*box.port_scope, *box.left_ports, *box.right_ports, *box.tree}
        out |= {r for link in box.links for r in link}
        out |= {r for r in (box.tree_parent, box.feed_port) if r is not None}
    return frozenset(out)


def _holders(diagram: Diagram, node_id: NodeId) -> frozenset[BangBoxId]:
    """The node-scope boxes holding ``node_id``."""
    return frozenset(b for b, box in diagram.bang_boxes.items() if node_id in box.node_scope)


def identity_slides(diagram: Diagram, copy: frozenset[NodeId]) -> Iterator[Diagram]:
    """Per phaseless Z(1 -> 1) ``v`` off ``copy`` with one leg a boundary slot and the other
    wired to ``p`` held by the boxes holding ``v``, and per wire from off ``copy`` to a node
    ``b`` of ``copy`` held by those boxes, no box naming ``p``'s or ``b``'s port on it nor,
    other than as a link port, the far port: ``diagram`` with ``p``'s port in the slot, ``v``
    on that wire beside ``b``, and ``p``, in each box where its group is not ``v``'s, in a
    group of its own."""
    named = _box_ports(diagram)
    linked = {r for box in diagram.bang_boxes.values() for link in box.links for r in link}
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    slots = [*diagram.boundary_inputs, *diagram.boundary_outputs]
    for v in sorted(diagram.nodes):
        node = diagram.nodes[v]
        if v in copy or node.generator_type is not Z_SPIDER:
            continue
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            continue
        if node.inputs[0].dim != node.outputs[0].dim:
            continue
        if node.phase is not None and not node.phase.is_zero:
            continue
        legs = (PortRef(v, Direction.INPUT, 0), PortRef(v, Direction.OUTPUT, 0))
        held = [leg for leg in legs if leg in slots]
        if len(held) != 1 or any(leg in named for leg in legs):
            continue
        (slot,) = held
        (inner,) = (leg for leg in legs if leg != slot)
        pp = partner.get(inner)
        boxes = _holders(diagram, v)
        if pp is None or pp in named or pp in slots or _holders(diagram, pp.node_id) != boxes:
            continue
        for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
            for b_ref, a_ref in ((wire.a, wire.b), (wire.b, wire.a)):
                if b_ref.node_id not in copy or a_ref.node_id in copy or v == a_ref.node_id:
                    continue
                if b_ref in named or (a_ref in named and a_ref not in linked):
                    continue
                if _holders(diagram, b_ref.node_id) != boxes:
                    continue
                if diagram.nodes[b_ref.node_id].legs(b_ref.direction)[b_ref.index].dim != (
                    node.inputs[0].dim
                ):
                    continue
                yield _slide(diagram, v, slot, pp, a_ref, b_ref)


def _slide(
    diagram: Diagram, v: NodeId, slot: PortRef, pp: PortRef, a_ref: PortRef, b_ref: PortRef
) -> Diagram:
    """:func:`identity_slides`' result for ``v``, its slot leg, its partner port and the wire
    ``a_ref`` -- ``b_ref``."""
    working = diagram.copy()
    for wire in list(working.wires):
        if v in (wire.a.node_id, wire.b.node_id):
            working.remove_wire(wire.a, wire.b)
    working.remove_wire(a_ref, b_ref)
    working.set_boundary_inputs([pp if r == slot else r for r in working.boundary_inputs])
    working.set_boundary_outputs([pp if r == slot else r for r in working.boundary_outputs])
    out_end, in_end = (a_ref, b_ref) if a_ref.direction is Direction.OUTPUT else (b_ref, a_ref)
    working.add_wire(out_end, PortRef(v, Direction.INPUT, 0))
    working.add_wire(PortRef(v, Direction.OUTPUT, 0), in_end)
    p = pp.node_id
    for box_id in sorted(_holders(working, v)):
        box = working.bang_boxes[box_id]
        if not box.groups:
            continue
        at = [next((i for i, g in enumerate(box.groups) if n in g), -1) for n in (p, v)]
        if at[0] == at[1]:
            continue
        kept = tuple(g - {p} for g in box.groups if g - {p})
        working.set_bang_box_extension(box_id, groups=(*kept, frozenset({p})))
    return working


def boxed_splits(
    diagram: Diagram, copy: frozenset[NodeId]
) -> Iterator[tuple[Diagram, frozenset[NodeId]]]:
    """:func:`_boxed_split` of ``diagram`` and ``copy``."""
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    slots = [*diagram.boundary_inputs, *diagram.boundary_outputs]
    yield from _boxed_split(diagram, copy, partner, slots)


def _boxed_split(
    diagram: Diagram,
    copy: frozenset[NodeId],
    partner: Mapping[PortRef, PortRef],
    slots: Sequence[PortRef],
) -> Iterator[tuple[Diagram, frozenset[NodeId]]]:
    """Per boxed spider off ``copy`` wired to nodes of ``copy`` in its own boxes, or in a box
    aligned with its one box, per such home and per boundary slot it holds: ``diagram`` with
    a phaseless leaf split off the spider (the slot and the legs into ``copy``, in the boxes
    :func:`_leaf_home` names; the spider keeps its id, its other legs and a connector in the
    slot's place), and ``copy`` with the leaf; its own boxes first."""
    for v in sorted(diagram.nodes):
        node = diagram.nodes[v]
        if v in copy or node.generator_type not in (Z_SPIDER, X_SPIDER):
            continue
        owners = [b for b, box in diagram.bang_boxes.items() if v in box.node_scope]
        if not owners:
            continue
        legs = [
            PortRef(v, direction, i)
            for direction in (Direction.INPUT, Direction.OUTPUT)
            for i in range(len(node.legs(direction)))
        ]
        homes: dict[tuple[BangBoxId, ...], list[PortRef]] = {}
        for leg in legs:
            far = partner.get(leg)
            if far is None or far.node_id not in copy:
                continue
            home = _leaf_home(diagram, owners, far.node_id)
            other = diagram.nodes[far.node_id]
            if home != tuple(owners) and other.generator_type is node.generator_type:
                continue
            if home is not None:
                homes.setdefault(home, []).append(leg)
        for home, inward in sorted(homes.items(), key=lambda item: item[0] != tuple(owners)):
            for slot in (leg for leg in legs if leg in slots):
                split = _split_boxed_spider(diagram, v, slot, inward, home)
                if split is not None:
                    yield split[0], copy | {split[1]}


def _leaf_home(
    diagram: Diagram, owners: Sequence[BangBoxId], far: NodeId
) -> tuple[BangBoxId, ...] | None:
    """The boxes a leaf split off a spider in ``owners`` toward ``far`` goes into: ``owners``
    when ``far`` lies in all of them; the top-level box holding ``far`` when ``owners`` is one
    top-level box it is aligned with and ``far`` lies in no other box; else None."""
    if all(far in diagram.bang_boxes[b].node_scope for b in owners):
        return tuple(owners)
    holding = [b for b, box in diagram.bang_boxes.items() if far in box.node_scope]
    if len(owners) != 1 or len(holding) != 1:
        return None
    (owner,), (partner,) = owners, holding
    top = diagram.bang_boxes[owner].parent is None and diagram.bang_boxes[partner].parent is None
    if not top or partner not in aligned_members(diagram, owner):
        return None
    return (partner,)


def _split_boxed_spider(
    diagram: Diagram,
    v: NodeId,
    slot: PortRef,
    inward: Sequence[PortRef],
    into: Sequence[BangBoxId],
) -> tuple[Diagram, NodeId] | None:
    """:func:`_boxed_split`'s split of ``v`` at ``slot``, the leaf in the boxes ``into``, and
    the leaf's id; None when an ``inward`` leg precedes a kept leg of its direction."""
    node = diagram.nodes[v]
    moved = frozenset(inward)
    for direction in (Direction.INPUT, Direction.OUTPUT):
        flags = [PortRef(v, direction, i) in moved for i in range(len(node.legs(direction)))]
        if any(a and not b for a, b in pairwise(flags)):
            return None
    working = diagram.copy()
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    for leg in (slot, *inward):
        far = partner.get(leg)
        if far is not None:
            working.remove_wire(leg, far)
    dims = {d: [p.dim for p in node.legs(d)] for d in (Direction.INPUT, Direction.OUTPUT)}
    kept = {d: [x for i, x in enumerate(dims[d]) if PortRef(v, d, i) not in moved] for d in dims}
    hub = working.add_node(
        node.generator_type, kept[Direction.INPUT], kept[Direction.OUTPUT], phase=node.phase
    )
    _move_node(working, v, hub)
    connector_end = PortRef(hub, slot.direction, slot.index)
    away = slot.direction
    toward = Direction.OUTPUT if away is Direction.INPUT else Direction.INPUT
    leaf_legs: dict[Direction, list[PortRef | None]] = {away: [slot], toward: [None]}
    for leg in inward:
        leaf_legs[leg.direction].append(leg)
    slot_dim = dims[away][slot.index]

    def leg_dims(direction: Direction) -> list[Dim]:
        return [slot_dim if r is None else dims[r.direction][r.index] for r in leaf_legs[direction]]

    leaf = working.add_node(
        node.generator_type, leg_dims(Direction.INPUT), leg_dims(Direction.OUTPUT)
    )
    leaf_of = {r: PortRef(leaf, d, j) for d in leaf_legs for j, r in enumerate(leaf_legs[d]) if r}
    for leg in inward:
        far = partner.get(leg)
        if far is not None:
            working.add_wire(leaf_of[leg], far)
    working.add_wire(PortRef(leaf, toward, 0), connector_end)
    swap = {connector_end: leaf_of[slot]}
    working.set_boundary_inputs([swap.get(r, r) for r in working.boundary_inputs])
    working.set_boundary_outputs([swap.get(r, r) for r in working.boundary_outputs])
    for box_id in into:
        scope = working.bang_boxes[box_id].node_scope
        working.set_bang_box_node_scope(box_id, scope | {leaf})
    return working, leaf


def _move_node(working: Diagram, v: NodeId, hub: NodeId) -> None:
    """Rewire every wire, boundary slot and box reference of ``v`` onto ``hub`` (same
    direction and index) and remove ``v``."""

    def moved(ref: PortRef) -> PortRef:
        return PortRef(hub, ref.direction, ref.index) if ref.node_id == v else ref

    def moved_set(refs: frozenset[PortRef]) -> frozenset[PortRef]:
        return frozenset(moved(r) for r in refs)

    for wire in sorted(working.wires, key=lambda w: w.sort_key()):
        if v in (wire.a.node_id, wire.b.node_id):
            working.remove_wire(wire.a, wire.b)
            working.add_wire(moved(wire.a), moved(wire.b))
    working.set_boundary_inputs([moved(r) for r in working.boundary_inputs])
    working.set_boundary_outputs([moved(r) for r in working.boundary_outputs])
    for box_id, box in sorted(working.bang_boxes.items()):
        fields: dict[str, object] = {
            "node_scope": frozenset(hub if n == v else n for n in box.node_scope),
            "links": tuple((moved(p), moved(q)) for p, q in box.links),
            "groups": tuple(frozenset(hub if n == v else n for n in g) for g in box.groups),
            "left_ports": moved_set(box.left_ports),
            "right_ports": moved_set(box.right_ports),
            "tree": tuple(moved(r) for r in box.tree),
            "tree_parent": None if box.tree_parent is None else moved(box.tree_parent),
            "feed_port": None if box.feed_port is None else moved(box.feed_port),
        }
        working.set_bang_box_extension(box_id, **fields)
        if box.port_scope:
            working.set_bang_box_port_scope(box_id, moved_set(box.port_scope))
    working.remove_node(v)


def _gather_split(
    diagram: Diagram,
    copy: frozenset[NodeId],
    boxed: frozenset[NodeId],
    partner: Mapping[PortRef, PortRef],
    slots: Sequence[PortRef],
) -> Iterator[tuple[Diagram, frozenset[NodeId]]]:
    """Per unboxed spider off ``copy`` wired to ``copy`` and to a boxed node off it, and per
    boundary slot it holds: ``diagram`` with the spider split into a hub off the copy (its phase
    and its legs off the copy, the slot's place wired to the leaf) and a phaseless leaf in the
    copy (the slot and the legs into ``copy``)."""
    for v in sorted(diagram.nodes):
        node = diagram.nodes[v]
        if v in copy or v in boxed or node.generator_type not in (Z_SPIDER, X_SPIDER):
            continue
        legs = [
            PortRef(v, direction, i)
            for direction in (Direction.INPUT, Direction.OUTPUT)
            for i in range(len(node.legs(direction)))
        ]
        far = {leg: partner.get(leg) for leg in legs}
        inward = [leg for leg, f in far.items() if f is not None and f.node_id in copy]
        if not inward or not any(
            f is not None and f.node_id not in copy and f.node_id in boxed for f in far.values()
        ):
            continue
        for slot in (leg for leg in legs if leg in slots):
            yield _split_spider(diagram, v, slot, inward, copy)


def _split_spider(
    diagram: Diagram, v: NodeId, slot: PortRef, inward: Sequence[PortRef], copy: frozenset[NodeId]
) -> tuple[Diagram, frozenset[NodeId]]:
    """:func:`_fan_split`'s split of ``v`` at ``slot``."""
    node = diagram.nodes[v]
    working = diagram.copy()
    partner: dict[PortRef, PortRef] = {}
    for wire in diagram.wires:
        partner[wire.a], partner[wire.b] = wire.b, wire.a
    moved = frozenset(inward)
    dims = {d: [p.dim for p in node.legs(d)] for d in (Direction.INPUT, Direction.OUTPUT)}
    keep = {
        d: [i for i in range(len(dims[d])) if PortRef(v, d, i) not in moved]
        for d in (Direction.INPUT, Direction.OUTPUT)
    }
    hub = working.add_node(
        node.generator_type,
        [dims[Direction.INPUT][i] for i in keep[Direction.INPUT]],
        [dims[Direction.OUTPUT][i] for i in keep[Direction.OUTPUT]],
        phase=node.phase,
    )
    hub_of = {PortRef(v, d, i): PortRef(hub, d, j) for d in keep for j, i in enumerate(keep[d])}
    away = slot.direction
    toward = Direction.OUTPUT if away is Direction.INPUT else Direction.INPUT
    leaf_legs: dict[Direction, list[PortRef | None]] = {away: [slot], toward: [None]}
    for leg in inward:
        leaf_legs[leg.direction].append(leg)
    slot_dim = dims[away][slot.index]
    leaf = working.add_node(
        node.generator_type,
        [slot_dim if r is None else dims[r.direction][r.index] for r in leaf_legs[Direction.INPUT]],
        [
            slot_dim if r is None else dims[r.direction][r.index]
            for r in leaf_legs[Direction.OUTPUT]
        ],
    )
    leaf_of = {r: PortRef(leaf, d, j) for d in leaf_legs for j, r in enumerate(leaf_legs[d]) if r}
    connector = PortRef(leaf, toward, 0)
    new_of = {**hub_of, **leaf_of}
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        if v in (wire.a.node_id, wire.b.node_id):
            working.remove_wire(wire.a, wire.b)
    done: set[PortRef] = set()
    for leg, far in sorted(partner.items(), key=lambda item: item[0].sort_key()):
        if leg.node_id != v or leg in done:
            continue
        done.add(leg)
        other = new_of[far] if far.node_id == v else far
        if far.node_id == v:
            done.add(far)
        working.add_wire(new_of[leg], other)
    working.add_wire(connector, hub_of[slot])
    working.set_boundary_inputs([new_of.get(r, r) for r in diagram.boundary_inputs])
    working.set_boundary_outputs([new_of.get(r, r) for r in diagram.boundary_outputs])
    working.remove_node(v)
    return working, (copy - {v}) | {leaf}


def _chain_owner(diagram: Diagram, index: str) -> BangBoxId | None:
    """The single box whose multiplicity carries ``index`` when it is a chain box."""
    owners = [b for b, box in diagram.bang_boxes.items() if index in box.multiplicity.free_symbols]
    if len(owners) != 1 or not diagram.bang_boxes[owners[0]].links:
        return None
    return owners[0]


def _chain_owned(diagram: Diagram, index: str) -> bool:
    """Whether a box of :func:`index_owners` is a chain box."""
    return any(diagram.bang_boxes[b].links for b in index_owners(diagram, index))


def discharge_chain_window(
    obligation: InductionObligation, *, max_steps: int = DEFAULT_MAX_STEPS
) -> TierOutcome:
    """Settle the step case on a chain box: peel the last copy off one successor, read the
    residual as that side's hypothesis diagram through its cut wires, replace it by the other
    side's hypothesis diagram, and compare the result with the other successor (peeled or
    not) structurally or by symbolic contraction. A chain that peels its first copy and reads
    an index is tried again on :func:`offset_obligation`."""
    settled, why = _chain_window(obligation, max_steps)
    if settled:
        return TierOutcome(StepDischarge.CHAIN_WINDOW, True, why)
    offset = offset_obligation(obligation)
    if offset is not None:
        shifted, symbol = offset
        settled, inner = _chain_window(shifted, max_steps)
        if settled:
            return TierOutcome(
                StepDischarge.CHAIN_WINDOW,
                True,
                f"successors reading index + {symbol}, hypotheses index + {symbol} + 1 for "
                f"every {symbol}: {inner}",
            )
        why = f"{why}; offset {symbol}: {inner}"
    return TierOutcome(StepDischarge.CHAIN_WINDOW, False, why)


def offset_obligation(obligation: InductionObligation) -> tuple[InductionObligation, str] | None:
    """``obligation`` with both successors reading their copy index as ``index + s`` and both
    hypothesis diagrams as ``index + s + 1``, for a fresh ``s``; None unless a successor's box
    peels its first copy and reads an index."""
    from archytaszx.semantics.prove import induction_offset, offset_symbol, peels_shifted

    k = obligation.step_symbol
    pair = (obligation.left_at_successor, obligation.right_at_successor)
    if not any(peels_shifted(d, k) for d in pair):
        return None
    sym = offset_symbol(
        obligation.left_at_successor,
        obligation.right_at_successor,
        obligation.left_at_k,
        obligation.right_at_k,
    )
    try:
        shifted = replace(
            obligation,
            left_at_successor=induction_offset(obligation.left_at_successor, k, sym),
            right_at_successor=induction_offset(obligation.right_at_successor, k, sym),
            left_at_k=induction_offset(obligation.left_at_k, k, sym + 1),
            right_at_k=induction_offset(obligation.right_at_k, k, sym + 1),
        )
    except (BangBoxError, InductionError):
        return None
    return shifted, str(sym)


def _chain_window(obligation: InductionObligation, max_steps: int) -> tuple[bool, str]:
    """Whether :func:`discharge_chain_window`'s peel-and-replace settles ``obligation``; and
    how, or why not."""
    index = obligation.step_symbol
    reasons: list[str] = []
    paired = _compare_chain_copies(obligation, max_steps)
    if paired[0]:
        return True, paired[1]
    reasons.append(paired[1])
    sides = (
        (
            obligation.left_at_successor,
            obligation.left_at_k,
            obligation.right_at_k,
            obligation.right_at_successor,
        ),
        (
            obligation.right_at_successor,
            obligation.right_at_k,
            obligation.left_at_k,
            obligation.left_at_successor,
        ),
    )
    for own_successor, own_k, other_k, other_successor in sides:
        if not _chain_owned(own_successor, index):
            reasons.append("no chain box carries the step index")
            continue
        replaced = _side_replaced(own_successor, own_k, other_k, index)
        if isinstance(replaced, str):
            reasons.append(replaced)
            continue
        targets = [other_successor]
        try:
            targets.insert(0, peel_owners(other_successor, index).diagram)
        except BangBoxError:
            pass
        for target in targets:
            settled, why = _close_window(replaced, target, max_steps)
            if not settled:
                reasons.append(why)
                settled, why = _close_by_chain_core(replaced, target, other_k, index, max_steps)
            if settled:
                return True, (
                    "peeled the last chain copy; the residual is the hypothesis diagram through "
                    f"its cut wires, replaced by the other side's; {why}"
                )
            reasons.append(why)
    for last, (own_successor, own_k, other_k, other_successor) in (
        (False, sides[0]),
        (False, sides[1]),
        (True, sides[0]),
        (True, sides[1]),
    ):
        if not _chain_owned(other_successor, index):
            continue
        replaced = _side_replaced(own_successor, own_k, other_k, index, last=last)
        if isinstance(replaced, str):
            continue
        peeled_other = peel_peelable(other_successor, index)
        if peeled_other is None:
            continue
        settled, why = _close_by_lemmas(replaced, peeled_other, other_successor, index)
        if settled:
            return True, (
                f"peeled one copy{' (chains at their last copy)' if last else ''}; the residual "
                f"is the hypothesis diagram through its cut wires, replaced by the other side's; "
                f"{why}"
            )
        reasons.append(why)
    return False, "; ".join(reasons)


def peel_peelable(diagram: Diagram, index: str) -> Diagram | None:
    """:func:`peel_owners` of ``diagram``, else of its boxes carrying ``index`` whose
    multiplicity has a positive constant term (the others kept); None when neither peels."""
    try:
        return peel_owners(diagram, index).diagram
    except BangBoxError:
        pass
    try:
        return peel_owners(diagram, index, partial=True).diagram
    except BangBoxError:
        return None


def first_peelers(diagram: Diagram, index: str) -> frozenset[BangBoxId]:
    """The boxes carrying ``index`` that peel their first copy."""
    return frozenset(b for b in index_owners(diagram, index) if peels_first(diagram.bang_boxes[b]))


def _side_replaced(
    own_successor: Diagram, own_k: Diagram, other_k: Diagram, index: str, *, last: bool = False
) -> Diagram | str:
    """``own_successor`` peeled (:func:`_peel_step_index`; with ``last``, every first-peeling
    chain at its last copy), its residual read as ``own_k`` through cut wires
    (:func:`expose_cut`) and replaced by ``other_k``; else why not."""
    if last:
        firsts = first_peelers(own_successor, index)
        if not firsts:
            return "no chain peels its first copy"
        try:
            result = peel_owners(own_successor, index, last=firsts)
        except BangBoxError as exc:
            return str(exc)
        peeled = (result.diagram, result.copy_node_ids) if result.copy_node_ids else None
    else:
        peeled = _peel_step_index(own_successor, index)
    if peeled is None:
        return "the chain box does not peel"
    exposed, copy = peeled
    candidates = copy_candidates(exposed, frozenset(own_successor.nodes), copy)
    found = expose_cut(exposed, candidates, own_k)
    if found is None:
        return "the residual is not the hypothesis diagram through its cut wires"
    try:
        return replace_cut_region(found[0], found[1], own_k, other_k)
    except InductionDomainError as exc:
        return str(exc)


def _close_by_lemmas(
    replaced: Diagram, target: Diagram, source: Diagram, index: str
) -> tuple[bool, str]:
    """Whether :func:`~archytaszx.semantics.lemmas.close_with_lemmas` proves ``replaced`` equal
    to ``target``, the joint peel of ``source``; and how, or why not."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.lemmas import LemmaError, close_with_lemmas

    try:
        close_with_lemmas(replaced, target, source, index)
    except (kernel.KernelError, LemmaError, InductionError, BangBoxError, ValidateError) as exc:
        return False, f"lemmas: {exc}"
    return True, (
        "library lemmas rewritten on both sides (the other side re-peeled at its chains' last "
        "copies) leave parts off the chain cores that agree under symbolic contraction"
    )


def _chain_core(diagram: Diagram, index: str) -> frozenset[NodeId] | None:
    """The scope of the chain box carrying ``index`` with every node off it reached from a
    link-in end without passing a box node; None when there is no such box or those nodes
    also reach a link-out end."""
    owner = _chain_owner(diagram, index)
    if owner is None:
        return None
    return box_core(diagram, owner)


def box_core(diagram: Diagram, owner: BangBoxId) -> frozenset[NodeId] | None:
    """The scope of chain box ``owner`` with every node off it reached from a link-in end
    without passing a scope node; None when those nodes also reach a link-out end."""
    box = diagram.bang_boxes[owner]
    scope = box.node_scope
    neighbours: dict[NodeId, set[NodeId]] = {n: set() for n in diagram.nodes}
    for wire in diagram.wires:
        neighbours[wire.a.node_id].add(wire.b.node_id)
        neighbours[wire.b.node_id].add(wire.a.node_id)

    def reached(ends: Sequence[PortRef]) -> set[NodeId]:
        seen: set[NodeId] = set()
        todo = [
            far.node_id
            for end in ends
            for far in (_partner_of(diagram, end),)
            if far is not None and far.node_id not in scope
        ]
        while todo:
            node_id = todo.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            todo += [n for n in neighbours[node_id] if n not in scope and n not in seen]
        return seen

    head = reached([q_ref for _p, q_ref in box.links])
    tail = reached([p_ref for p_ref, _q in box.links])
    if head & tail:
        return None
    return frozenset(scope | head)


def _partner_of(diagram: Diagram, ref: PortRef) -> PortRef | None:
    """The port wired to ``ref``, or None."""
    for wire in diagram.wires:
        if wire.a == ref:
            return wire.b
        if wire.b == ref:
            return wire.a
    return None


def _open_region(diagram: Diagram, region: frozenset[NodeId]) -> Diagram:
    """``region`` of ``diagram`` as an open diagram: its boundary slots, then its ends of cut
    wires by port order; scalar ``diagram``'s."""
    rest = frozenset(diagram.nodes) - region
    cuts = _cut_wires(diagram, rest)
    far_ends = sorted(cuts.values(), key=PortRef.sort_key)
    others = {far: inside for inside, far in cuts.items()}
    cut = CutRegion(
        rest,
        tuple(
            [r for r in diagram.boundary_inputs if r.node_id in rest]
            + [others[f] for f in far_ends if f.direction is Direction.INPUT]
        ),
        tuple(
            [r for r in diagram.boundary_outputs if r.node_id in rest]
            + [others[f] for f in far_ends if f.direction is Direction.OUTPUT]
        ),
        MappingProxyType(cuts),
    )
    return _copy_side(diagram, cut, diagram.scalar)[0]


def chain_cores(
    hypothesis: Diagram, a: Diagram, b: Diagram, index: str
) -> tuple[Diagram, frozenset[NodeId], frozenset[NodeId]] | str:
    """``hypothesis``'s chain core (:func:`_chain_core`) as an open diagram and the chain cores
    of ``a`` and ``b``, each chain box's multiplicity ``hypothesis``'s; else why not."""
    cores = [_chain_core(d, index) for d in (hypothesis, a, b)]
    core, core_a, core_b = cores
    if core is None or core_a is None or core_b is None:
        return "no chain box with separate head and tail on both sides"
    mults = [
        d.bang_boxes[owner].multiplicity
        for d in (hypothesis, a, b)
        for owner in (_chain_owner(d, index),)
        if owner is not None
    ]
    if any(m != mults[0] for m in mults[1:]):
        return "the chain boxes have multiplicities " + ", ".join(str(m) for m in mults)
    return _open_region(hypothesis, core), core_a, core_b


def _close_by_chain_core(
    replaced: Diagram, target: Diagram, hypothesis: Diagram, index: str, max_steps: int
) -> tuple[bool, str]:
    """Whether ``replaced`` and ``target`` hold ``hypothesis``'s chain core (:func:`_chain_core`)
    through cut wires in the same places, the parts off it agreeing with their cut ends open;
    and how, or why not."""
    from archytaszx.semantics import kernel

    found = chain_cores(hypothesis, replaced, target, index)
    if isinstance(found, str):
        return False, f"chain core: {found}"
    opened, core_a, core_b = found
    try:
        kernel.local_contraction(replaced, core_a, target, core_b, opened, max_steps=max_steps)
    except (kernel.KernelError, InductionDomainError, ValidateError) as exc:
        return False, f"chain core: {exc}"
    return True, (
        "both hold the other side's chain core (box and head) in the same places and the parts "
        "off it agree under symbolic contraction"
    )


def _tree_owner(diagram: Diagram, index: str) -> BangBoxId | None:
    """The single box whose multiplicity carries ``index`` when it is a tree box."""
    owners = [b for b, box in diagram.bang_boxes.items() if index in box.multiplicity.free_symbols]
    if len(owners) != 1 or not diagram.bang_boxes[owners[0]].is_tree:
        return None
    return owners[0]


def _slot_counts(diagram: Diagram, refs: Sequence[PortRef]) -> list[object] | None:
    """The expanded slot count of each boundary ref, or None when one does not resolve."""
    out: list[object] = []
    for ref in refs:
        count = _port_multiplicity(diagram, ref)
        if count is None:
            return None
        out.append(sp.expand(count))
    return out


def unfed(exposed: Diagram, tree_id: BangBoxId) -> Diagram:
    """``exposed`` without the box tree ``tree_id`` feeds, the tree's child ports in that box's
    boundary block."""
    tree = exposed.bang_boxes[tree_id]
    fed = fed_box(exposed, tree_id)
    if fed is None:
        raise InductionDomainError(f"tree box {tree_id!r} feeds no box")
    working = exposed.copy()
    for refs, setter in (
        (list(working.boundary_outputs), working.set_boundary_outputs),
        (list(working.boundary_inputs), working.set_boundary_inputs),
    ):
        block = [i for i, r in enumerate(refs) if r.node_id in fed.node_scope]
        if not block:
            continue
        if block != list(range(block[0], block[0] + len(tree.tree))):
            raise InductionDomainError("the fed box's boundary slots are not its leaf block")
        refs[block[0] : block[0] + len(tree.tree)] = list(tree.tree)
        setter(refs)
    working.remove_bang_box(fed.id)
    for node_id in sorted(fed.node_scope):
        working.remove_node(node_id)
    return working


def fed_replacement(
    exposed: Diagram, tree_id: BangBoxId, own_k: Diagram, other_k: Diagram
) -> Diagram:
    """``other_k`` with the box tree ``tree_id`` feeds in ``exposed`` grafted on: the leaf run
    of ``own_k`` read at the same slot counts in ``other_k``, as one leg port-boxed b^k times
    (the leg fanned into the box) or a tree box's child run (the box fed by that tree)."""
    tree = exposed.bang_boxes[tree_id]
    fed = fed_box(exposed, tree_id)
    if fed is None or fed.feed_port is None:
        raise InductionDomainError(f"tree box {tree_id!r} feeds no box")
    b = len(tree.tree)
    residual = unfed(exposed, tree_id)
    if not isomorphic(residual, own_k):
        raise InductionDomainError("the residual is not the hypothesis diagram")
    outputs = tree.tree[0].direction is Direction.OUTPUT
    own = list(residual.boundary_outputs if outputs else residual.boundary_inputs)
    other = list(other_k.boundary_outputs if outputs else other_k.boundary_inputs)
    s = own.index(tree.tree[0])
    width = len(other) - len(own) + b
    own_counts = _slot_counts(residual, own)
    other_counts = _slot_counts(other_k, other)
    if own_counts is None or other_counts is None or width < 1:
        raise InductionDomainError("the boundary slot counts do not resolve")
    if own_counts[:s] != other_counts[:s] or own_counts[s + b :] != other_counts[s + width :]:
        raise InductionDomainError("the two hypothesis diagrams place the leaf run differently")
    middle = other[s : s + width]
    working = other_k.copy()
    id_map: dict[NodeId, NodeId] = {}
    for old_id in sorted(fed.node_scope):
        node = exposed.nodes[old_id]
        id_map[old_id] = working.add_node(
            node.generator_type,
            [p.dim for p in node.inputs],
            [p.dim for p in node.outputs],
            phase=node.phase,
        )
    for wire in sorted(exposed.wires, key=lambda w: w.sort_key()):
        if wire.a.node_id in fed.node_scope and wire.b.node_id in fed.node_scope:
            working.add_wire(_moved_ref(wire.a, id_map), _moved_ref(wire.b, id_map))
    own_list = list(exposed.boundary_outputs if outputs else exposed.boundary_inputs)
    block = [_moved_ref(r, id_map) for r in own_list if r.node_id in fed.node_scope]
    leaves = sp.Integer(b) ** tree.multiplicity.to_sympy()
    feed: BangBoxId | None = None
    port_box = [
        box_id
        for box_id, box in working.bang_boxes.items()
        if len(middle) == 1 and box.port_scope == frozenset(middle)
    ]
    trees = [
        box_id
        for box_id, box in working.bang_boxes.items()
        if box.is_tree and list(box.tree) == middle and fed_box(working, box_id) is None
    ]
    if (
        port_box
        and sp.expand(working.bang_boxes[port_box[0]].multiplicity.to_sympy() - leaves) == 0
    ):
        box = working.bang_boxes[port_box[0]]
        if box.parent is not None or any(o.parent == box.id for o in working.bang_boxes.values()):
            raise InductionDomainError("the leaf leg's port box is nested")
        working.remove_bang_box(box.id)
        working.add_wire(middle[0], _moved_ref(fed.feed_port, id_map))
    elif (
        trees
        and len(working.bang_boxes[trees[0]].tree) == b
        and (working.bang_boxes[trees[0]].multiplicity == tree.multiplicity)
    ):
        feed = trees[0]
    else:
        raise InductionDomainError(
            "the other hypothesis diagram's leaf run is neither one leg port-boxed b^k times "
            "nor a tree box's child run"
        )
    refs = [*other[:s], *block, *other[s + width :]]
    (working.set_boundary_outputs if outputs else working.set_boundary_inputs)(refs)
    image = add_box_image(working, replace(fed, feed=None, feed_port=None), id_map, None)
    if feed is not None:
        working.set_bang_box_extension(
            image, feed=feed, feed_port=_moved_ref(fed.feed_port, id_map)
        )
    working.multiply_scalar(exposed.scalar * residual.scalar**-1)
    validate_or_raise(working)
    return working


def _w_with_weights(dim: Dim, legs: int, weights: Mapping[int, Scalar], scalar: Scalar) -> Diagram:
    """``scalar`` times W(1 -> ``legs``) after a Z(1 -> 1) with ``weights``."""
    out = Diagram()
    phase = PhaseVector(dim, weights=weights) if weights else None
    gate = out.add_node(Z_SPIDER, [dim], [dim], phase=phase)
    w = out.add_node(W_NODE, [dim], [dim] * legs)
    out.add_wire(PortRef(gate, Direction.OUTPUT, 0), PortRef(w, Direction.INPUT, 0))
    out.set_boundary_inputs([PortRef(gate, Direction.INPUT, 0)])
    out.set_boundary_outputs([PortRef(w, Direction.OUTPUT, i) for i in range(legs)])
    out.set_scalar(scalar)
    return out


def _w_body(
    diagram: Diagram, box: BangBox, inside: PortRef, outs: Sequence[PortRef]
) -> tuple[Scalar, dict[int, Scalar]] | None:
    """``(c, w)`` with the copy of ``box`` (input ``inside``, outputs ``outs``) exactly equal to
    ``c`` W(1 -> len(outs)) after a Z(1 -> 1) with weights ``w``; None when it is not."""
    from archytaszx.semantics.contract_symbolic import _engine_index, contract_symbolic

    dim = _ref_dim(diagram, inside)
    if not dim.is_concrete or any(_ref_dim(diagram, r) != dim for r in outs):
        return None
    body = _extract(diagram, box.node_scope)
    id_map = dict(zip(sorted(box.node_scope), sorted(body.nodes), strict=True))
    body.set_boundary_inputs([_moved_ref(inside, id_map)])
    body.set_boundary_outputs([_moved_ref(r, id_map) for r in outs])
    if _all_free_symbols(body):
        return None
    try:
        tensor = contract_symbolic(body)
    except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError):
        return None
    if tensor.is_variable_rank:
        return None
    axes = [_engine_index(a.index) for a in tensor.axes]
    entry = tensor.entry.to_sympy()
    ratios: list[sp.Expr] = []
    for j in range(dim.to_int()):
        values = [j if i == 0 else 0 for i in range(len(outs))] + [j]
        ratios.append(sp.simplify(entry.xreplace(dict(zip(axes, values, strict=True)))))
    if ratios[0] == 0:
        return None
    weights = {j: Scalar(sp.simplify(ratios[j] / ratios[0])) for j in range(1, len(ratios))}
    weights = {j: w for j, w in weights.items() if w != Scalar.one()}
    candidate = _w_with_weights(dim, len(outs), weights, Scalar(ratios[0]))
    try:
        left, right = contract_symbolic_pair(body, candidate)
    except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError):
        return None
    if not compare_symbolic(left, right).matched:
        return None
    return Scalar(ratios[0]), weights


def w_fused(diagram: Diagram) -> Diagram:
    """``diagram`` with a top-level node box, fed by one wire from the output ``q`` of a W node
    and otherwise held by a contiguous run of boundary outputs, whose copy is ``c`` W(1 -> m)
    after a Z with weights ``w`` (:func:`_w_body`), fused into the W node: the box removed,
    ``q`` port-boxed ``M m`` times in the run's place, a Z with weights ``w`` before the W node
    (the W node's only output then ``q``), the scalar times ``(c s)^M`` (``M`` the box's
    multiplicity, ``s`` its copy scalar); InductionDomainError when there is no such box."""
    for box_id, box in sorted(diagram.bang_boxes.items()):
        scope = box.node_scope
        if (
            not scope
            or box.parent is not None
            or box.links
            or box.is_tree
            or box.feed is not None
            or box.pair_of is not None
            or any(o.parent == box_id or o.pair_of == box_id for o in diagram.bang_boxes.values())
            or any(r.node_id in scope for r in diagram.boundary_inputs)
        ):
            continue
        cuts = _cut_wires(diagram, scope)
        if len(cuts) != 1:
            continue
        ((inside, far),) = cuts.items()
        w_node = diagram.nodes[far.node_id]
        if (
            inside.direction is not Direction.INPUT
            or w_node.generator_type is not W_NODE
            or any(_box_footprint(b) & {far.node_id} for b in diagram.bang_boxes.values())
        ):
            continue
        slots = [i for i, r in enumerate(diagram.boundary_outputs) if r.node_id in scope]
        if not slots or slots != list(range(slots[0], slots[0] + len(slots))):
            continue
        outs = [diagram.boundary_outputs[i] for i in slots]
        copy_scalar = sp.Integer(1) if box.copy_scalar is None else box.copy_scalar.to_sympy()
        if box.index is not None and any(str(s) == box.index for s in copy_scalar.free_symbols):
            continue
        found = _w_body(diagram, box, inside, outs)
        if found is None:
            continue
        factor, weights = found
        if weights and w_node.num_outputs != 1:
            continue
        fused = diagram.copy()
        fused.remove_bang_box(box_id)
        for node_id in sorted(scope):
            fused.remove_node(node_id)
        if weights:
            w_in = PortRef(far.node_id, Direction.INPUT, 0)
            gate = fused.add_node(
                Z_SPIDER,
                [_ref_dim(diagram, w_in)],
                [_ref_dim(diagram, w_in)],
                phase=PhaseVector(_ref_dim(diagram, w_in), weights=weights),
            )
            source = _partner_of(diagram, w_in)
            if source is None:
                fused.set_boundary_inputs(
                    [
                        PortRef(gate, Direction.INPUT, 0) if r == w_in else r
                        for r in fused.boundary_inputs
                    ]
                )
            else:
                fused.remove_wire(source, w_in)
                fused.add_wire(source, PortRef(gate, Direction.INPUT, 0))
            fused.add_wire(PortRef(gate, Direction.OUTPUT, 0), w_in)
        count = box.multiplicity.to_sympy()
        fused.add_bang_box(
            Mult._from_expr(sp.powsimp(sp.expand(count * len(outs)))), port_scope=frozenset({far})
        )
        refs = list(diagram.boundary_outputs)
        refs[slots[0] : slots[-1] + 1] = [far]
        fused.set_boundary_outputs(refs)
        fused.set_scalar(
            Scalar(diagram.scalar.to_sympy() * (factor.to_sympy() * copy_scalar) ** count)
        )
        validate_or_raise(fused)
        return fused
    raise InductionDomainError("no box is a W fan on one W leg")


def discharge_tree_window(
    obligation: InductionObligation, *, max_steps: int = DEFAULT_MAX_STEPS
) -> TierOutcome:
    """Settle the step case on a tree box over its depth: peel the leaf level off one
    successor, read the rest as that side's hypothesis diagram, replace it by the other side's
    with the leaf level grafted on its leaf run, and compare the result with the other successor
    (peeled or not) structurally or by symbolic contraction."""
    index = obligation.step_symbol
    reasons: list[str] = []
    sides = (
        (
            obligation.left_at_successor,
            obligation.left_at_k,
            obligation.right_at_k,
            obligation.right_at_successor,
        ),
        (
            obligation.right_at_successor,
            obligation.right_at_k,
            obligation.left_at_k,
            obligation.left_at_successor,
        ),
    )
    for own_successor, own_k, other_k, other_successor in sides:
        owner = _tree_owner(own_successor, index)
        if owner is None:
            reasons.append("no tree box carries the step index")
            continue
        try:
            exposed = peel_one(own_successor, owner).diagram
            replaced = fed_replacement(exposed, owner, own_k, other_k)
        except (BangBoxError, InductionDomainError, ValidateError) as exc:
            reasons.append(str(exc))
            continue
        targets = [other_successor]
        other_owner = _tree_owner(other_successor, index)
        if other_owner is not None:
            try:
                targets.insert(0, peel_one(other_successor, other_owner).diagram)
            except BangBoxError:
                pass
        for target in targets:
            settled, why = _close_window(replaced, target, max_steps)
            if settled:
                return TierOutcome(
                    StepDischarge.TREE_WINDOW,
                    True,
                    "peeled the tree's leaf level; the rest is the hypothesis diagram, replaced "
                    f"by the other side's with the leaf level on its leaf run; {why}",
                    left_peeled=exposed,
                )
            reasons.append(why)
        try:
            fused = w_fused(replaced)
        except (BangBoxError, InductionDomainError, ValidateError) as exc:
            reasons.append(f"W fusion: {exc}")
            continue
        for target in targets:
            settled, why = _close_window(fused, target, max_steps)
            if settled:
                return TierOutcome(
                    StepDischarge.TREE_WINDOW,
                    True,
                    "peeled the tree's leaf level; the rest is the hypothesis diagram, replaced "
                    "by the other side's with the leaf level on its leaf run; the leaf copies are "
                    "a weighted W fused into the W leg feeding them; "
                    f"{why}",
                    left_peeled=exposed,
                )
            reasons.append(f"W fusion: {why}")
    return TierOutcome(StepDischarge.TREE_WINDOW, False, "; ".join(reasons))


def _exposed_chain(
    successor: Diagram, hypothesis: Diagram, index: str
) -> tuple[Diagram, CutRegion] | None:
    """The successor with its chain box's last copy peeled, and the residual read as
    ``hypothesis`` through its cut wires; None when either fails."""
    if not _chain_owned(successor, index):
        return None
    peeled = _peel_step_index(successor, index)
    if peeled is None:
        return None
    exposed, copy = peeled
    cut = find_cut_region(exposed, frozenset(exposed.nodes) - copy, hypothesis)
    if cut is None or boundary_layout(exposed) is None:
        return None
    return exposed, cut


def _copy_side(exposed: Diagram, cut: CutRegion, scalar: Scalar) -> tuple[Diagram, object]:
    """The part of ``exposed`` off ``cut``'s region as an open diagram (its own boundary slots
    in order, then each cut wire's far end by the place of its region end in ``cut``'s lists),
    scalar ``scalar``; and the positions the region holds in ``exposed``'s lists."""
    rest = frozenset(exposed.nodes) - cut.region
    place = {r: (0, i) for i, r in enumerate(cut.inputs)}
    place.update({r: (1, i) for i, r in enumerate(cut.outputs)})
    ends = sorted(cut.cuts.items(), key=lambda item: place[item[0]])
    extracted = _extract(exposed, rest, with_boxes=True)
    id_map = {old: new for old, new in zip(sorted(rest), sorted(extracted.nodes), strict=True)}
    inputs = [_moved_ref(r, id_map) for r in exposed.boundary_inputs if r.node_id in rest]
    outputs = [_moved_ref(r, id_map) for r in exposed.boundary_outputs if r.node_id in rest]
    for _inside, far in ends:
        (inputs if far.direction is Direction.INPUT else outputs).append(_moved_ref(far, id_map))
    extracted.set_boundary_inputs(inputs)
    extracted.set_boundary_outputs(outputs)
    extracted.set_scalar(scalar)
    held = (
        tuple(i for i, r in enumerate(exposed.boundary_inputs) if r.node_id in cut.region),
        tuple(i for i, r in enumerate(exposed.boundary_outputs) if r.node_id in cut.region),
        tuple(place[inside] for inside, _far in ends),
        tuple(far.direction for _inside, far in ends),
    )
    return extracted, held


def _compare_chain_copies(obligation: InductionObligation, max_steps: int) -> tuple[bool, str]:
    """Whether both successors are chains whose residuals are their hypothesis diagrams in the
    same places and whose peeled copies, with their cut ends open, agree."""
    index = obligation.step_symbol
    left = _exposed_chain(obligation.left_at_successor, obligation.left_at_k, index)
    right = _exposed_chain(obligation.right_at_successor, obligation.right_at_k, index)
    if left is None or right is None:
        return False, "the two successors are not both peelable chains over their hypotheses"
    if boundary_layout(obligation.left_at_k) != boundary_layout(obligation.right_at_k):
        return False, "the two hypothesis diagrams lay out their boundaries differently"
    left_copy, left_held = _copy_side(
        left[0], left[1], left[0].scalar * obligation.left_at_k.scalar**-1
    )
    right_copy, right_held = _copy_side(
        right[0], right[1], right[0].scalar * obligation.right_at_k.scalar**-1
    )
    if left_held != right_held:
        return False, "the residuals sit in different boundary places on the two sides"
    if _interface(left_copy) != _interface(right_copy):
        return False, "the peeled chain copies have different interfaces"
    settled, why = _close_window(left_copy, right_copy, max_steps)
    if not settled and lemma_decision(left_copy, right_copy) is not None:
        settled, why = True, "they are equal by a nested decision (lemma)"
    if settled:
        return True, (
            "peeled the last copy off both chains; the residuals are the hypothesis diagrams in "
            f"the same places and the copies agree: {why}"
        )
    return False, f"the peeled chain copies differ: {why}"


_LEMMA_DEPTH: ContextVar[int] = ContextVar("induction_lemma_depth", default=0)
_MAX_LEMMA_DEPTH = 2


def lemma_decision(left: Diagram, right: Diagram) -> object | None:
    """The EQUAL decision of ``left`` against ``right``, both carrying a live bang box, by
    :func:`~archytaszx.semantics.decide.decide_equal` with no oracle sample, at most
    ``_MAX_LEMMA_DEPTH`` such decisions deep; else None."""
    if not (free_mult_symbols(left) and free_mult_symbols(right)):
        return None
    depth = _LEMMA_DEPTH.get()
    if depth >= _MAX_LEMMA_DEPTH:
        return None
    from archytaszx.semantics.decide import DecideError, EqualityVerdict, decide_equal

    token = _LEMMA_DEPTH.set(depth + 1)
    try:
        decision = decide_equal(left, right, samples=[])
    except (DecideError, InductionError, BangBoxError, ValidateError):
        return None
    finally:
        _LEMMA_DEPTH.reset(token)
    return decision if decision.verdict is EqualityVerdict.EQUAL else None


def rest_cut(exposed: Diagram, cut: CutRegion) -> CutRegion:
    """The part of ``exposed`` off ``cut``'s region as a cut region, its lists ordered as
    :func:`_copy_side` orders the open diagram's."""
    rest = frozenset(exposed.nodes) - cut.region
    place = {r: (0, i) for i, r in enumerate(cut.inputs)}
    place.update({r: (1, i) for i, r in enumerate(cut.outputs)})
    ends = sorted(cut.cuts.items(), key=lambda item: place[item[0]])
    inputs = [r for r in exposed.boundary_inputs if r.node_id in rest]
    outputs = [r for r in exposed.boundary_outputs if r.node_id in rest]
    for _inside, far in ends:
        (inputs if far.direction is Direction.INPUT else outputs).append(far)
    return CutRegion(
        rest, tuple(inputs), tuple(outputs), MappingProxyType(_cut_wires(exposed, rest))
    )


def _close_window(replaced: Diagram, target: Diagram, max_steps: int) -> tuple[bool, str]:
    """Whether ``replaced`` equals ``target``, structurally or by symbolic contraction; and how,
    or why not."""
    if compare_structure(replaced, target).identical and _same_bang_boxes(replaced, target):
        return True, "the result is the other successor"
    linked = any(box.links for d in (replaced, target) for box in d.bang_boxes.values())
    if isomorphic(replaced, target, symmetric_legs=not linked):
        return True, "the result is isomorphic to the other successor"
    try:
        left, right = contract_symbolic_pair(replaced, target, max_steps=max_steps)
    except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError) as exc:
        return False, f"symbolic contraction: {exc}"
    except ScalarBudgetError as exc:
        return False, f"symbolic contraction: {exc}"
    comparison = compare_symbolic(left, right)
    if comparison.matched:
        return True, "the result and the other successor agree under symbolic contraction"
    return False, f"symbolic contraction: {comparison.reason}"


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
    """Settle the step case by contracting both successor diagrams (both doubled when either has
    a discard) with every symbol kept formal.

    A shape the symbolic contractor does not cover, or a budget it exhausts, returns an
    unsettled outcome rather than raising, so the ladder falls through to the tier below.
    A malformed obligation still propagates: no lower tier can rescue it.
    """
    try:
        left, right = contract_symbolic_pair(
            obligation.left_at_successor, obligation.right_at_successor, max_steps=max_steps
        )
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
    StepDischarge.CHAIN_WINDOW: discharge_chain_window,
    StepDischarge.TREE_WINDOW: discharge_tree_window,
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
    """Call one tier's discharge function with the keyword arguments its signature takes.

    A tier that refuses the obligation outright returns an unsettled outcome here, so a
    ladder falls through to the tier below instead of taking the whole call down; a direct
    call to that tier still raises, where the refusal is the caller's own error.
    """
    run = _DISPATCH[discharge]
    try:
        if discharge in (StepDischarge.UNIFORM_REWRITE, StepDischarge.INDUCTION_REWRITE):
            return run(obligation, rules=rules, max_steps=max_steps)
        if discharge is StepDischarge.ORACLE_WINDOW:
            return run(
                obligation, width=width, mode=mode, tolerance=tolerance, max_elements=max_elements
            )
        return run(obligation, max_steps=max_steps)
    except InductionGrammarError as exc:
        return TierOutcome(discharge, False, f"tier refused the obligation: {exc}")
    except (
        InductionDomainError,
        RewriteDomainError,
        SymbolicContractionDomainError,
        ContractDomainError,
    ) as exc:
        return TierOutcome(discharge, False, f"tier hit a domain error: {exc}")


def _concrete_child(
    obligation: InductionObligation,
    value: int,
    rules: Sequence[Rule] | None,
    max_steps: int,
) -> tuple[Derivation | None, str]:
    """The step-sequence derivation over the hypothesis pair instantiated at ``value``, or None
    and the rewrite's reason."""
    k = obligation.step_symbol
    left = _at_base(obligation.left_at_k, k, value, obligation.witness, obligation.base_proved)
    right = _at_base(obligation.right_at_k, k, value, obligation.witness, obligation.base_proved)
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
    restrict: object | None = None,
    family_first: bool = True,
    base_proved: bool = False,
) -> InductionResult:
    """Discharge an equality over one multiplicity index by running the base case then each
    discharge tier in ladder order until one settles; under ``restrict``, the family first,
    kept when proved or refuted in range, otherwise the restricted pair inducted over the fresh
    symbol of the index's clause from 0; ``family_first=False`` inducts the restricted pair
    only. ``base_proved`` (no ``restrict``) takes the base case at ``base`` as proved by the
    caller and holds formal every symbol ``witness`` leaves out."""
    if not isinstance(family_first, bool):
        raise InductionGrammarError(f"family_first must be a bool, got {family_first!r}")
    if base_proved and restrict is not None:
        raise InductionGrammarError("a proved base is not taken under a restriction")
    if restrict is not None:
        options = {
            "base": base,
            "ladder": ladder,
            "rules": rules,
            "step_symbol": step_symbol,
            "width": width,
            "max_steps": max_steps,
            "certify_proof": certify_proof,
            "mode": mode,
            "tolerance": tolerance,
            "max_elements": max_elements,
        }
        return _restricted_induction(
            left, right, index, witness, restrict, options, family_first=family_first
        )
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
        base_proved=base_proved,
    )
    base_comparison = (
        None
        if base_proved
        else discharge_base(obligation, mode=mode, tolerance=tolerance, max_elements=max_elements)
    )
    if base_comparison is not None and not base_comparison.matched:
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


def _restricted_induction(
    left: Diagram,
    right: Diagram,
    index: str | None,
    witness: Mapping[str, CheckAssignmentValue],
    restrict: object,
    options: Mapping[str, object],
    family_first: bool = True,
) -> InductionResult:
    """The three steps of :func:`prove_by_induction` under a restriction; every clause symbol
    other than the index is held at its clause's value or a witness value inside it; the
    family step skipped when ``family_first`` is False."""
    from archytaszx.semantics.decide import DecideGrammarError
    from archytaszx.semantics.restrict import (
        EQ,
        in_domain,
        parse_restriction,
        subst_pair,
        to_original,
        to_restricted,
        validate,
    )

    try:
        restriction = validate(parse_restriction(restrict), left, right)  # type: ignore[arg-type]
    except DecideGrammarError as exc:
        raise InductionGrammarError(str(exc)) from exc
    if restriction.empty:
        return prove_by_induction(left, right, index=index, witness=witness, **options)  # type: ignore[arg-type]
    chosen = choose_index(left, right, index=index)
    held = dict(witness)
    for clause in restriction.clauses:
        if clause.symbol == chosen:
            if clause.form == EQ:
                raise InductionGrammarError(
                    f"the induction index {chosen} is fixed by {clause.text()}"
                )
            continue
        if clause.form == EQ:
            if clause.symbol in held and held[clause.symbol] != clause.c:
                raise InductionGrammarError(
                    f"{clause.symbol} = {held[clause.symbol]} lies outside {clause.text()}"
                )
            held[clause.symbol] = clause.c
        elif clause.symbol in held and not clause.contains(held[clause.symbol]):
            raise InductionGrammarError(
                f"{clause.symbol} = {held[clause.symbol]} lies outside {clause.text()}"
            )
    family = (
        prove_by_induction(left, right, index=chosen, witness=held, **options)  # type: ignore[arg-type]
        if family_first
        else None
    )
    if family is not None and family.proved:
        return replace(family, restriction=restriction, family=family, family_holds=True)
    if family is not None and family.counterexample is not None:
        point = {**dict(family.witness), family.index: family.counterexample}
        if in_domain(restriction, point):
            return replace(
                family,
                restriction=restriction,
                family=family,
                counterexample_original=MappingProxyType(point),
            )
    try:
        pair = subst_pair(left, right, restriction)
    except DecideGrammarError as exc:
        raise InductionGrammarError(str(exc)) from exc
    own = restriction.by_symbol.get(chosen)
    inner_index = own.fresh if own is not None else chosen
    inner_witness = to_restricted(restriction, held) or {}
    inner = prove_by_induction(*pair, index=inner_index, witness=inner_witness, **options)  # type: ignore[arg-type]
    original = None
    if inner.counterexample is not None:
        point = {**dict(inner.witness), inner.index: inner.counterexample}
        original = MappingProxyType(to_original(restriction, point))
    return replace(
        inner,
        restriction=restriction,
        family=family,
        family_holds=False,
        restricted_pair=pair,
        counterexample_original=original,
    )
