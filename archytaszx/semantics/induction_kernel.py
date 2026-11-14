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

"""The kernel theorem of an induction-rung EQUAL, built from the rung's own data.

A step case settled by uniform rewrite gives a chain of rule inferences on the original pair,
each step re-applied where the matcher finds the same nodes. One settled by induction rewrite
gives a :func:`~archytaszx.semantics.kernel.induction` theorem: the base case from the base
decision's certificates (else a :func:`~archytaszx.semantics.prove.prove` search), the
tier's peeled successor as the exposed diagram, its residual as the hypothesis region, and a
step from a rewrite derivation between the two peeled copies (normal forms meeting, else a
search) re-applied inside the successors. One settled by chain_window or tree_window gives an
induction theorem over the chain successor's cut region or the tree successor's unfed residual,
the step from the replaced diagram to the other peeled successor by isomorphism, meeting
normal forms, or symbolic contraction of the parts off the regions
(:func:`~archytaszx.semantics.kernel.local_contraction`) or of the whole
(:func:`~archytaszx.semantics.kernel.contraction`). One settled by symbolic contraction gives
an induction theorem over a chain's cut region, a tree's unfed residual or a closed region, the
step by isomorphism or by symbolic contraction. A peeled step whose copies have no rewrite
derivation falls back to symbolic contraction of the replaced diagram, then to
:func:`~archytaszx.semantics.kernel.congr` of a nested decision's theorem equating the parts
off the regions (:func:`~archytaszx.semantics.induction.lemma_decision`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.engine import RewriteStep, apply
from archytaszx.rewrite.match import FusionMatch
from archytaszx.rewrite.normal_form import normal_form
from archytaszx.rewrite.rule import RewriteError, Rule
from archytaszx.rewrite.rules_library import SPIDER_FUSION, lookup_rule
from archytaszx.semantics import kernel
from archytaszx.semantics.certificate import compare_structure
from archytaszx.semantics.induction import (
    CutRegion,
    ExposeStep,
    InductionDomainError,
    InductionResult,
    StepDischarge,
    TierOutcome,
    _copy_side,
    _extract,
    _fan_split,
    copy_candidates,
    expose_cut,
    fed_replacement,
    find_cut_region,
    first_peelers,
    lemma_decision,
    replace_cut_region,
    replace_region,
    rest_cut,
    unfed,
)
from archytaszx.semantics.prove import (
    InductionSide,
    induction_base,
    induction_hypothesis,
    induction_offset,
    induction_renamed,
    induction_successor,
    offset_symbol,
    peels_shifted,
    prove,
)


class NoInductionTheorem(Exception):
    """Why an induction verdict has no kernel theorem."""


def induction_theorem(
    left: Diagram,
    right: Diagram,
    result: InductionResult,
    reversed_: bool,
    base_pair: tuple[Diagram, Diagram],
    base_decision: object,
    rules: Sequence[Rule] | None = None,
) -> tuple[kernel.Theorem | None, str]:
    """``left = right`` from a proved induction ``result`` (run on ``right, left`` when
    ``reversed_``) and the decision of ``base_pair`` at index 0, or None and why."""
    try:
        first, second = (right, left) if reversed_ else (left, right)
        tier = _settling_tier(result)
        if tier.discharge is StepDischarge.UNIFORM_REWRITE:
            theorem = _uniform(first, second, result, tier)
        elif tier.discharge is StepDischarge.INDUCTION_REWRITE:
            base = _base(first, second, result.index, base_pair, base_decision, reversed_)
            theorem = _peeled(first, second, result, tier, base, rules)
        elif tier.discharge in (StepDischarge.CHAIN_WINDOW, StepDischarge.TREE_WINDOW):
            base = _base(first, second, result.index, base_pair, base_decision, reversed_)
            window = "cut" if tier.discharge is StepDischarge.CHAIN_WINDOW else "tree"
            theorem = _windowed(first, second, result, window, base, rules)
        elif tier.discharge is StepDischarge.SYMBOLIC_CONTRACTION:
            base = _base(first, second, result.index, base_pair, base_decision, reversed_)
            theorem = _contracted(first, second, result, base, rules)
        else:
            raise NoInductionTheorem(
                f"the step case was settled by {tier.discharge.value}, which records no "
                "rewrite steps"
            )
        if reversed_:
            theorem = kernel.symm(theorem)
        return theorem, ""
    except NoInductionTheorem as exc:
        return None, str(exc)
    except kernel.KernelError as exc:
        return None, f"the kernel refused a piece: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"building it raised {type(exc).__name__}: {exc}"


def _settling_tier(result: InductionResult) -> TierOutcome:
    """The last settled tier of ``result``."""
    for tier in reversed(result.tiers):
        if tier.settled:
            return tier
    raise NoInductionTheorem("no tier settled the step case")


def _uniform(
    first: Diagram, second: Diagram, result: InductionResult, tier: TierOutcome
) -> kernel.Theorem:
    """``first = second`` by the tier's steps re-applied to ``first``."""
    source = result.obligation.left_at_k
    if frozenset(source.nodes) != frozenset(first.nodes):
        raise NoInductionTheorem("the hypothesis diagram does not keep the left side's node ids")
    steps = transport(first, tier.steps, {n: n for n in first.nodes})
    chained = kernel.chain(first, steps)
    return kernel.trans(chained, kernel.iso(chained.rhs, second))


def _base(
    start: Diagram,
    goal: Diagram,
    index: str,
    pair: tuple[Diagram, Diagram],
    decision: object,
    reversed_: bool,
) -> kernel.Theorem:
    """``start = goal`` at ``index = 0`` from the base decision of ``pair``, else a search."""
    try:
        theorem = kernel.from_decision(decision, *pair)
    except kernel.KernelError as exc:
        outcome = prove(*pair)
        artefact = (
            None
            if not outcome.proved
            else outcome.certificate
            if outcome.certificate is not None
            else outcome.induction
        )
        if artefact is None:
            raise NoInductionTheorem(
                f"base case: {exc}; search: {outcome.reason or 'no certificate'}"
            ) from exc
        theorem = kernel.check(artefact)
    if reversed_:
        theorem = kernel.symm(theorem)
    at_start, at_goal = induction_base(start, index), induction_base(goal, index)
    return kernel.trans(
        kernel.iso(at_start, theorem.lhs),
        kernel.trans(theorem, kernel.iso(theorem.rhs, at_goal)),
    )


def _peeled(
    start: Diagram,
    goal: Diagram,
    result: InductionResult,
    tier: TierOutcome,
    base: kernel.Theorem,
    rules: Sequence[Rule] | None,
) -> kernel.Theorem:
    """``start = goal`` by the kernel's induction rule over the tier's peeled successors."""
    if (
        tier.left_peeled is None
        or tier.left_copy is None
        or tier.right_peeled is None
        or tier.right_copy is None
    ):
        raise NoInductionTheorem("the peeling tier recorded no peeled successors")
    index, k = result.index, result.obligation.step_symbol
    peeled = induction_successor(start, index, k)
    other_peeled = induction_successor(goal, index, k)
    if peeled is None or other_peeled is None:
        raise NoInductionTheorem("a successor has no single box to peel")
    exposed, right_peeled = tier.left_peeled, tier.right_peeled
    expose = (
        kernel.refl(peeled)
        if compare_structure(peeled, exposed).identical
        else kernel.iso(peeled, exposed)
    )
    region = frozenset(exposed.nodes) - tier.left_copy
    replaced = replace_region(
        exposed, region, induction_hypothesis(start, index, k), induction_hypothesis(goal, index, k)
    )
    left_copy = _extract(exposed, tier.left_copy, with_boxes=True)
    right_copy = _extract(right_peeled, tier.right_copy, with_boxes=True)
    try:
        forward, backward = copy_derivation(left_copy, right_copy, rules)
    except NoInductionTheorem as exc:
        try:
            contracted = kernel.contraction(replaced, right_peeled)
        except kernel.KernelError as inner:
            closed = CutRegion(
                tier.left_copy,
                tuple(r for r in replaced.boundary_inputs if r.node_id in tier.left_copy),
                tuple(r for r in replaced.boundary_outputs if r.node_id in tier.left_copy),
                MappingProxyType({}),
            )
            try:
                contracted = _lemma_congr(replaced, closed, left_copy, right_copy, right_peeled)
            except (NoInductionTheorem, kernel.KernelError) as last:
                raise NoInductionTheorem(f"{exc}; {inner}; lemma: {last}") from exc
        step = kernel.trans(contracted, kernel.iso(right_peeled, other_peeled))
    else:
        there = kernel.chain(
            replaced, transport(replaced, forward, _ids(tier.left_copy, left_copy))
        )
        back = kernel.chain(
            right_peeled, transport(right_peeled, backward, _ids(tier.right_copy, right_copy))
        )
        step = kernel.trans(
            there,
            kernel.trans(
                kernel.iso(there.rhs, back.rhs),
                kernel.trans(kernel.symm(back), kernel.iso(right_peeled, other_peeled)),
            ),
        )
    return kernel.induction(start, goal, index, k, base, InductionSide.START, expose, region, step)


def _contracted(
    start: Diagram,
    goal: Diagram,
    result: InductionResult,
    base: kernel.Theorem,
    rules: Sequence[Rule] | None,
) -> kernel.Theorem:
    """``start = goal`` by the kernel's induction rule over a chain's cut region, a tree's unfed
    residual or a closed region, whichever the peeled successor admits, the step closed as in
    :func:`_windowed`."""
    reasons: list[str] = []
    for window in ("cut", "tree", "region"):
        try:
            return _windowed(start, goal, result, window, base, rules, contract=True)
        except NoInductionTheorem as exc:
            reasons.append(f"{window}: {exc}")
    raise NoInductionTheorem("; ".join(reasons))


def _windowed(
    start: Diagram,
    goal: Diagram,
    result: InductionResult,
    window: str,
    base: kernel.Theorem,
    rules: Sequence[Rule] | None,
    *,
    contract: bool = False,
) -> kernel.Theorem:
    """``start = goal`` by the kernel's induction rule over a chain's cut region (``window``
    cut), a tree's unfed residual (tree) or a closed region (region), the step from the
    replaced diagram to the other peeled successor by :func:`_step_theorem`; each side's
    successor tried as the exposed one, then, for a box that peels its first copy and reads an
    index, both again with an index offset (:func:`~archytaszx.semantics.kernel.induction`)."""
    index, k = result.index, result.obligation.step_symbol
    reasons: list[str] = []
    offsets: list[sp.Symbol | None] = [None]
    if peels_shifted(start, index) or peels_shifted(goal, index):
        obligation = result.obligation
        offsets.append(
            offset_symbol(start, goal, obligation.left_at_successor, obligation.right_at_successor)
        )
    for sym, side in [(o, s) for o in offsets for s in (InductionSide.START, InductionSide.GOAL)]:
        own, other = (start, goal) if side is InductionSide.START else (goal, start)
        label = side.value if sym is None else f"{side.value}, offset {sym}"
        try:
            own_s, other_s = own, other
            if sym is not None:
                own_s = induction_offset(own, index, sym)
                other_s = induction_offset(other, index, sym)
            peeled = induction_successor(own_s, index, k)
            other_peeled = induction_successor(other_s, index, k)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{label}: peeling raised {type(exc).__name__}: {exc}")
            continue
        if peeled is None or other_peeled is None:
            reasons.append(f"{label}: a successor has no single box to peel")
            continue
        own_k = induction_hypothesis(own, index, k)
        other_k = induction_hypothesis(other, index, k)
        if sym is not None:
            own_k = induction_offset(own_k, k, sym + 1)
            other_k = induction_offset(other_k, k, sym + 1)
        try:
            tree_id = None
            cut = None
            expose = kernel.refl(peeled)
            if window == "cut":
                copy = frozenset(peeled.nodes) - frozenset(own_k.nodes)
                region = frozenset(peeled.nodes) - copy
                cut = find_cut_region(peeled, region, own_k)
                for split, split_copy in () if cut is not None else _fan_split(peeled, copy):
                    cut = find_cut_region(split, frozenset(split.nodes) - split_copy, own_k)
                    if cut is not None:
                        expose = _split_theorem(peeled, split, copy, split_copy)
                        region = cut.region
                        break
                if cut is None:
                    later = copy_candidates(peeled, frozenset(own_k.nodes), copy)[1:]
                    found = expose_cut(peeled, [copy, *later], own_k) if later else None
                    if found is not None:
                        _split, cut, moves = found
                        expose = _steps_theorem(peeled, moves)
                        region = cut.region
                if cut is None:
                    raise NoInductionTheorem("the residual is not the hypothesis through cut wires")
                replaced = replace_cut_region(expose.rhs, cut, own_k, other_k)
            elif window == "region":
                region = frozenset(peeled.nodes) & frozenset(own_k.nodes)
                replaced = replace_region(peeled, region, own_k, other_k)
            else:
                fed = [box for box in peeled.bang_boxes.values() if box.feed is not None]
                if len(fed) != 1 or fed[0].feed is None:
                    raise NoInductionTheorem("the peeled successor holds no single fed box")
                tree_id = fed[0].feed
                region = frozenset(unfed(peeled, tree_id).nodes)
                replaced = fed_replacement(peeled, tree_id, own_k, other_k)
            local = None
            if tree_id is None:
                rest = frozenset(expose.rhs.nodes) - region
                local = (
                    frozenset(replaced.nodes) - rest,
                    frozenset(other_peeled.nodes) & frozenset(other_k.nodes),
                    other_k,
                )
            chain = (other_k, k) if window == "cut" else None
            try:
                step = _step_theorem(
                    replaced, other_peeled, rules, contract=contract, local=local, chain=chain
                )
            except NoInductionTheorem as exc:
                if tree_id is None and window == "cut":
                    source = induction_renamed(other_s, index, k)
                    step = _lemma_step(replaced, other_peeled, source, k, exc)
                    return kernel.induction(
                        start,
                        goal,
                        index,
                        k,
                        base,
                        side,
                        expose,
                        region,
                        step,
                        cut=cut,
                        offset=None if sym is None else str(sym),
                    )
                if tree_id is None:
                    raise
                try:
                    fused = kernel.w_fusion(replaced)
                except kernel.KernelError as inner:
                    raise NoInductionTheorem(f"{exc}; W fusion: {inner}") from exc
                closing = _step_theorem(fused.rhs, other_peeled, rules, contract=contract)
                step = kernel.trans(fused, closing)
            return kernel.induction(
                start,
                goal,
                index,
                k,
                base,
                side,
                expose,
                region,
                step,
                cut=cut,
                tree=tree_id,
                offset=None if sym is None else str(sym),
            )
        except (NoInductionTheorem, kernel.KernelError, InductionDomainError) as exc:
            reasons.append(f"{label}: {exc}")
    if window == "cut":
        for side in (InductionSide.START, InductionSide.GOAL):
            try:
                return _last_window(start, goal, index, k, base, side)
            except (NoInductionTheorem, kernel.KernelError, InductionDomainError) as exc:
                reasons.append(f"{side.value}, chains at their last copy: {exc}")
    raise NoInductionTheorem("; ".join(reasons))


def _last_window(
    start: Diagram,
    goal: Diagram,
    index: str,
    k: str,
    base: kernel.Theorem,
    side: InductionSide,
) -> kernel.Theorem:
    """``start = goal`` by the kernel's induction rule over a cut region of ``side``'s successor
    with every first-peeling chain re-peeled at its last copy
    (:func:`~archytaszx.semantics.kernel.repeel`), the step by
    :func:`~archytaszx.semantics.lemmas.close_with_lemmas`."""
    own, other = (start, goal) if side is InductionSide.START else (goal, start)
    source = induction_renamed(own, index, k)
    firsts = first_peelers(source, k)
    peeled = induction_successor(own, index, k)
    other_peeled = induction_successor(other, index, k)
    if not firsts or peeled is None or other_peeled is None:
        raise NoInductionTheorem("no chain peels its first copy")
    moved = kernel.repeel(source, k, frozenset(), firsts)
    if not kernel._same(moved.lhs, peeled):
        raise NoInductionTheorem("the re-peel does not start at the peeled successor")
    exposed = moved.rhs
    own_k = induction_hypothesis(own, index, k)
    other_k = induction_hypothesis(other, index, k)
    copy = frozenset(exposed.nodes) - frozenset(own_k.nodes)
    found = expose_cut(exposed, copy_candidates(exposed, frozenset(own_k.nodes), copy), own_k)
    if found is None:
        raise NoInductionTheorem("the residual is not the hypothesis through cut wires")
    _split, cut, steps = found
    expose = kernel.trans(moved, _steps_theorem(exposed, steps))
    replaced = replace_cut_region(expose.rhs, cut, own_k, other_k)
    step = _lemma_step(
        replaced,
        other_peeled,
        induction_renamed(other, index, k),
        k,
        NoInductionTheorem("chains at their last copy"),
    )
    return kernel.induction(start, goal, index, k, base, side, expose, cut.region, step, cut=cut)


def _lemma_step(
    replaced: Diagram, target: Diagram, source: Diagram, k: str, failed: NoInductionTheorem
) -> kernel.Theorem:
    """``replaced = target`` by :func:`~archytaszx.semantics.lemmas.close_with_lemmas`; raises
    ``failed`` extended with why not."""
    from archytaszx.semantics.lemmas import LemmaError, close_with_lemmas

    try:
        return cast("kernel.Theorem", close_with_lemmas(replaced, target, source, k))
    except (kernel.KernelError, LemmaError, InductionDomainError) as exc:
        raise NoInductionTheorem(f"{failed}; lemmas: {exc}") from exc


def _steps_theorem(peeled: Diagram, steps: Sequence[ExposeStep]) -> kernel.Theorem:
    """``peeled = d`` through ``steps`` (:func:`~archytaszx.semantics.induction.expose_cut`):
    each split by :func:`_split_theorem`, each leg move by
    :func:`~archytaszx.semantics.kernel.leg_move`."""
    theorem = kernel.refl(peeled)
    for step in steps:
        if step.kind == "move":
            piece = kernel.leg_move(step.before, step.after, step.copy_before)
        else:
            piece = _split_theorem(step.before, step.after, step.copy_before, step.copy_after)
        theorem = kernel.trans(theorem, piece)
    return theorem


def _split_theorem(
    peeled: Diagram, split: Diagram, copy: frozenset[NodeId], split_copy: frozenset[NodeId]
) -> kernel.Theorem:
    """``peeled = split`` for a :func:`~archytaszx.semantics.induction._fan_split` result: the
    fusion of its two halves, reversed; a boxed split by
    :func:`~archytaszx.semantics.kernel.box_split`."""
    if any(split_copy - copy <= box.node_scope for box in split.bang_boxes.values()):
        return kernel.box_split(peeled, split, copy)
    (leaf,) = split_copy - copy
    (hub,) = frozenset(split.nodes) - frozenset(peeled.nodes) - {leaf}
    for match in SPIDER_FUSION.pattern.find_matches(split):
        if not isinstance(match, FusionMatch) or {match.a_id, match.b_id} != {hub, leaf}:
            continue
        fused = kernel.rule(split, apply(split, SPIDER_FUSION, match).step)
        return kernel.symm(kernel.trans(fused, kernel.iso(fused.rhs, peeled)))
    raise NoInductionTheorem("the split spider does not fuse back")


def _step_theorem(
    replaced: Diagram,
    target: Diagram,
    rules: Sequence[Rule] | None,
    *,
    contract: bool = False,
    local: tuple[frozenset[NodeId], frozenset[NodeId], Diagram] | None = None,
    chain: tuple[Diagram, str] | None = None,
) -> kernel.Theorem:
    """``replaced = target`` by an isomorphism; else, unless ``contract``, by a rewrite
    derivation (:func:`copy_derivation`, between the parts off the two regions of ``local``
    when given, :func:`_local_derivation`); else by
    :func:`~archytaszx.semantics.kernel.local_contraction` over ``local`` (the two regions and
    the diagram they read as), else over the chain core of ``chain`` (a hypothesis diagram and
    the step symbol), else by :func:`~archytaszx.semantics.kernel.contraction`."""
    try:
        return kernel.iso(replaced, target)
    except kernel.KernelError:
        pass
    reasons: list[str] = []
    if not contract and local is not None:
        try:
            there, back = _local_derivation(replaced, target, local, rules)
            return kernel.trans(
                there, kernel.trans(kernel.iso(there.rhs, back.rhs), kernel.symm(back))
            )
        except (NoInductionTheorem, kernel.KernelError) as exc:
            reasons.append(f"local derivation: {exc}")
    elif not contract:
        try:
            forward, backward = copy_derivation(replaced, target, rules)
        except NoInductionTheorem as exc:
            reasons.append(str(exc))
        else:
            there = kernel.chain(replaced, forward)
            back = kernel.chain(target, backward)
            return kernel.trans(
                there, kernel.trans(kernel.iso(there.rhs, back.rhs), kernel.symm(back))
            )
    if local is not None:
        try:
            return kernel.local_contraction(replaced, local[0], target, local[1], local[2])
        except kernel.KernelError as exc:
            reasons.append(f"local: {exc}")
        try:
            return _local_lemma(replaced, target, local)
        except (NoInductionTheorem, kernel.KernelError, InductionDomainError) as exc:
            reasons.append(f"lemma: {exc}")
    if chain is not None:
        from archytaszx.semantics.induction import chain_cores

        found = chain_cores(chain[0], replaced, target, chain[1])
        if isinstance(found, str):
            reasons.append(f"chain core: {found}")
        else:
            try:
                return kernel.local_contraction(replaced, found[1], target, found[2], found[0])
            except kernel.KernelError as exc:
                reasons.append(f"chain core: {exc}")
    try:
        return kernel.contraction(replaced, target)
    except kernel.KernelError as exc:
        reasons.append(str(exc))
    raise NoInductionTheorem("; ".join(reasons))


def _lemma_congr(
    context: Diagram, cut: CutRegion, part: Diagram, other_part: Diagram, target: Diagram
) -> kernel.Theorem:
    """``context = target`` by :func:`~archytaszx.semantics.kernel.congr` of the theorem
    ``part = other_part`` (from :func:`~archytaszx.semantics.induction.lemma_decision`) at
    ``cut``, then an isomorphism."""
    decision = lemma_decision(part, other_part)
    if decision is None:
        raise NoInductionTheorem("no nested decision proves the parts equal")
    lemma = kernel.from_decision(decision, part, other_part)
    plugged = kernel.congr(lemma, context, cut)
    return kernel.trans(plugged, kernel.iso(plugged.rhs, target))


def _local_lemma(
    replaced: Diagram,
    target: Diagram,
    local: tuple[frozenset[NodeId], frozenset[NodeId], Diagram],
) -> kernel.Theorem:
    """``replaced = target`` when the regions of ``local`` read as its hypothesis through cut
    wires and the parts off them, cut ends open, are equal by :func:`_lemma_congr`."""
    hypothesis = local[2]
    pairs = ((replaced, local[0]), (target, local[1]))
    cuts = [find_cut_region(d, region, hypothesis) for d, region in pairs]
    if cuts[0] is None or cuts[1] is None:
        raise NoInductionTheorem("a region is not the hypothesis through cut wires")
    part, _ = _copy_side(replaced, cuts[0], replaced.scalar * hypothesis.scalar**-1)
    other_part, _ = _copy_side(target, cuts[1], target.scalar * hypothesis.scalar**-1)
    return _lemma_congr(replaced, rest_cut(replaced, cuts[0]), part, other_part, target)


def _local_derivation(
    replaced: Diagram,
    target: Diagram,
    local: tuple[frozenset[NodeId], frozenset[NodeId], Diagram],
    rules: Sequence[Rule] | None,
) -> tuple[kernel.Theorem, kernel.Theorem]:
    """Rewrite chains from ``replaced`` and ``target``: a :func:`copy_derivation` between their
    parts off the regions of ``local`` (cut ends open), transported back onto each."""
    parts: list[tuple[frozenset[NodeId], Diagram]] = []
    for d, region in ((replaced, local[0]), (target, local[1])):
        part, _held = kernel._plugged(d, region, local[2])
        parts.append((frozenset(d.nodes) - region, part))
    forward, backward = copy_derivation(parts[0][1], parts[1][1], rules)
    there = kernel.chain(replaced, transport(replaced, forward, _ids(parts[0][0], parts[0][1])))
    back = kernel.chain(target, transport(target, backward, _ids(parts[1][0], parts[1][1])))
    return there, back


def _ids(original: frozenset[NodeId], extracted: Diagram) -> dict[NodeId, NodeId]:
    """The extracted diagram's node ids mapped to the ids they were extracted from."""
    return dict(zip(sorted(extracted.nodes), sorted(original), strict=True))


def copy_derivation(
    left: Diagram, right: Diagram, rules: Sequence[Rule] | None
) -> tuple[tuple[RewriteStep, ...], tuple[RewriteStep, ...]]:
    """Steps from ``left`` and from ``right`` to diagrams with isomorphic comparison views:
    their normal forms, else a :func:`~archytaszx.semantics.prove.prove` search."""
    reasons: list[str] = []
    try:
        nf_l, nf_r = normal_form(left, rules=rules), normal_form(right, rules=rules)
        kernel.iso(nf_l.diagram, nf_r.diagram)
        return tuple(r.step for r in nf_l.results), tuple(r.step for r in nf_r.results)
    except (RewriteError, kernel.KernelError) as exc:
        reasons.append(f"normal forms: {exc}")
    outcome = prove(left, right)
    if outcome.proved and outcome.certificate is not None:
        return outcome.certificate.forward.steps, outcome.certificate.backward.steps
    reasons.append(f"search: {outcome.reason}")
    raise NoInductionTheorem(
        f"no rewrite derivation between the peeled copies ({'; '.join(reasons)})"
    )


def transport(
    start: Diagram, steps: Sequence[RewriteStep], ids: Mapping[NodeId, NodeId]
) -> tuple[RewriteStep, ...]:
    """``steps``, recorded on a diagram whose node ids ``ids`` sends into ``start``, re-applied
    to ``start`` at the match of each rule consuming the corresponding nodes."""
    working, mapping, out = start, dict(ids), []
    for position, step in enumerate(steps):
        try:
            wanted = frozenset(mapping[n] for n in step.consumed_node_ids)
        except KeyError as exc:
            raise NoInductionTheorem(f"step {position} consumes a node outside the copy") from exc
        rule = lookup_rule(step.rule_name)
        chosen = None
        for match in rule.pattern.find_matches(working):
            try:
                candidate = apply(working, rule, match)
            except RewriteError:
                continue
            if frozenset(candidate.step.consumed_node_ids) == wanted and len(
                candidate.step.new_node_ids
            ) == len(step.new_node_ids):
                chosen = candidate
                break
        if chosen is None:
            raise NoInductionTheorem(
                f"step {position} ({step.rule_name}) has no match on the corresponding nodes"
            )
        mapping.update(zip(step.new_node_ids, chosen.step.new_node_ids, strict=True))
        working = chosen.diagram
        out.append(chosen.step)
    return tuple(out)
