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
from dataclasses import dataclass, replace
from types import MappingProxyType

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar, ScalarBudgetError
from archytaszx.diagram.bangbox import (
    BangBox,
    BangBoxError,
    Mult,
    _port_multiplicity,
    add_box_image,
    fed_box,
    free_mult_symbols,
    index_symbols,
    peel_one,
    substitute_index_phases,
)
from archytaszx.diagram.compare import isomorphic
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
        if not (expr.is_Pow and expr.base.is_Integer and expr.exp == symbol):
            raise InductionDomainError(
                "induction requires a bare multiplicity symbol or a power b^n of one, got "
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
    if base is not None:
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
    """A box's chain, index, copy-scalar and pair fields."""
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
    the enclosing order, and with ``with_boxes`` every box lying wholly inside them."""
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
            (box_id, box)
            for box_id, box in sorted(diagram.bang_boxes.items())
            if _box_footprint(box) <= node_ids
        ]
        inside_ids = {box_id for box_id, _ in inside}
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
                box_map[box_id] = add_box_image(
                    extracted,
                    box,
                    id_map,
                    None if parent is None else box_map[parent],
                    None if box.pair_of is None else box_map[box.pair_of],
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
            extracted = _bare(_extract(diagram, region, with_boxes=True))
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
        working.add_wire(
            PortRef(id_map[wire.a.node_id], wire.a.direction, wire.a.index),
            PortRef(id_map[wire.b.node_id], wire.b.direction, wire.b.index),
        )
    box_map: dict[BangBoxId, BangBoxId] = {}
    pending = sorted(replacement.bang_boxes.items())
    while len(box_map) < len(pending):
        for box_id, box in pending:
            if box_id in box_map or (box.parent is not None and box.parent not in box_map):
                continue
            if box.pair_of is not None and box.pair_of not in box_map:
                continue
            box_map[box_id] = add_box_image(
                working,
                box,
                id_map,
                None if box.parent is None else box_map[box.parent],
                None if box.pair_of is None else box_map[box.pair_of],
            )

    def refill(slots: list[PortRef], fresh: tuple[PortRef, ...]) -> list[PortRef]:
        queue = [PortRef(id_map[r.node_id], r.direction, r.index) for r in fresh]
        return [queue.pop(0) if ref.node_id in region else ref for ref in slots]

    working.set_boundary_inputs(refill(inputs, replacement.boundary_inputs))
    working.set_boundary_outputs(refill(outputs, replacement.boundary_outputs))
    working.multiply_scalar(replacement.scalar * hypothesis.scalar**-1)
    return working


def _residual_is(peeled: Diagram, copy_node_ids: frozenset[NodeId], expected: Diagram) -> bool:
    """Whether ``peeled`` minus its peeled copy is ``expected``, field for field."""
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
        if box.links or _extension(box) != _extension(other):
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
    try:
        contracted_left, contracted_right = contract_symbolic_pair(
            left_extra, right_extra, max_steps=max_steps
        )
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
            rows.append((mults, opens, _ref_dim(diagram, ref)))
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
            if isomorphic(_bare(_cut_extract(diagram, cut)), target):
                return cut
    return None


def replace_cut_region(
    diagram: Diagram, cut: CutRegion, hypothesis: Diagram, replacement: Diagram
) -> Diagram:
    """``diagram`` with ``cut``'s region swapped for ``replacement``: slot ``i`` of each of
    ``replacement``'s boundary lists takes slot ``i`` of ``cut``'s, a cut end's far port wired
    to it, the scalar times ``replacement.scalar / hypothesis.scalar``."""
    if (len(replacement.boundary_inputs), len(replacement.boundary_outputs)) != (
        len(cut.inputs),
        len(cut.outputs),
    ):
        raise InductionDomainError("the replacement's interface does not fit the cut region")
    layout = boundary_layout(hypothesis)
    if layout is None or layout != boundary_layout(replacement):
        raise InductionDomainError("the replacement's boundary layout differs from the region's")
    if boundary_layout(diagram) is None:
        raise InductionDomainError("a box's boundary slots are not contiguous")
    working = diagram.copy()
    inside = [b for b, box in working.bang_boxes.items() if _box_footprint(box) & cut.region]
    for box_id in inside:
        if not _box_footprint(working.bang_boxes[box_id]) <= cut.region:
            raise InductionDomainError(f"bang box {box_id!r} straddles the replaced region")
    for box_id in sorted(inside, reverse=True):
        working.remove_bang_box(box_id)
    slot_of: dict[PortRef, PortRef] = {}
    fresh = _graft(working, replacement)
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
    working.set_boundary_inputs(inputs)
    working.set_boundary_outputs(outputs)
    working.multiply_scalar(replacement.scalar * hypothesis.scalar**-1)
    return working


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
            box_map[box_id] = add_box_image(
                working,
                box,
                id_map,
                None if box.parent is None else box_map[box.parent],
                None if box.pair_of is None else box_map[box.pair_of],
            )
    return id_map


def _chain_owner(diagram: Diagram, index: str) -> BangBoxId | None:
    """The single box whose multiplicity carries ``index`` when it is a chain box."""
    owners = [b for b, box in diagram.bang_boxes.items() if index in box.multiplicity.free_symbols]
    if len(owners) != 1 or not diagram.bang_boxes[owners[0]].links:
        return None
    return owners[0]


def discharge_chain_window(
    obligation: InductionObligation, *, max_steps: int = DEFAULT_MAX_STEPS
) -> TierOutcome:
    """Settle the step case on a chain box: peel the last copy off one successor, read the
    residual as that side's hypothesis diagram through its cut wires, replace it by the other
    side's hypothesis diagram, and compare the result with the other successor (peeled or
    not) structurally or by symbolic contraction."""
    index = obligation.step_symbol
    reasons: list[str] = []
    paired = _compare_chain_copies(obligation, max_steps)
    if paired[0]:
        return TierOutcome(StepDischarge.CHAIN_WINDOW, True, paired[1])
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
        if _chain_owner(own_successor, index) is None:
            reasons.append("no chain box carries the step index")
            continue
        peeled = _peel_step_index(own_successor, index)
        if peeled is None:
            reasons.append("the chain box does not peel")
            continue
        exposed, copy = peeled
        cut = find_cut_region(exposed, frozenset(exposed.nodes) - copy, own_k)
        if cut is None:
            reasons.append("the residual is not the hypothesis diagram through its cut wires")
            continue
        try:
            replaced = replace_cut_region(exposed, cut, own_k, other_k)
        except InductionDomainError as exc:
            reasons.append(str(exc))
            continue
        targets = [other_successor]
        owners = [
            b
            for b, box in other_successor.bang_boxes.items()
            if index in box.multiplicity.free_symbols
        ]
        if len(owners) == 1:
            try:
                targets.insert(0, peel_one(other_successor, owners[0]).diagram)
            except BangBoxError:
                pass
        for target in targets:
            settled, why = _close_window(replaced, target, max_steps)
            if settled:
                return TierOutcome(
                    StepDischarge.CHAIN_WINDOW,
                    True,
                    "peeled the last chain copy; the residual is the hypothesis diagram through "
                    f"its cut wires, replaced by the other side's; {why}",
                )
            reasons.append(why)
    return TierOutcome(StepDischarge.CHAIN_WINDOW, False, "; ".join(reasons))


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
    return TierOutcome(StepDischarge.TREE_WINDOW, False, "; ".join(reasons))


def _exposed_chain(
    successor: Diagram, hypothesis: Diagram, index: str
) -> tuple[Diagram, CutRegion] | None:
    """The successor with its chain box's last copy peeled, and the residual read as
    ``hypothesis`` through its cut wires; None when either fails."""
    if _chain_owner(successor, index) is None:
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
    if settled:
        return True, (
            "peeled the last copy off both chains; the residuals are the hypothesis diagrams in "
            f"the same places and the copies agree: {why}"
        )
    return False, f"the peeled chain copies differ: {why}"


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
    except (RewriteDomainError, SymbolicContractionDomainError, ContractDomainError) as exc:
        return TierOutcome(discharge, False, f"tier hit a domain error: {exc}")


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
    restrict: object | None = None,
    family_first: bool = True,
) -> InductionResult:
    """Discharge an equality over one multiplicity index by running the base case then each
    discharge tier in ladder order until one settles; under ``restrict``, the family first,
    kept when proved or refuted in range, otherwise the restricted pair inducted over the fresh
    symbol of the index's clause from 0; ``family_first=False`` inducts the restricted pair
    only."""
    if not isinstance(family_first, bool):
        raise InductionGrammarError(f"family_first must be a bool, got {family_first!r}")
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
