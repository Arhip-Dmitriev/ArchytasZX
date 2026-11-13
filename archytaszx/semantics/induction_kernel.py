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
derivation falls back to symbolic contraction of the replaced diagram.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.engine import RewriteStep, apply
from archytaszx.rewrite.normal_form import normal_form
from archytaszx.rewrite.rule import RewriteError, Rule
from archytaszx.rewrite.rules_library import lookup_rule
from archytaszx.semantics import kernel
from archytaszx.semantics.certificate import compare_structure
from archytaszx.semantics.induction import (
    InductionDomainError,
    InductionResult,
    StepDischarge,
    TierOutcome,
    _extract,
    fed_replacement,
    find_cut_region,
    replace_cut_region,
    replace_region,
    unfed,
)
from archytaszx.semantics.prove import (
    InductionSide,
    induction_base,
    induction_hypothesis,
    induction_offset,
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
            raise NoInductionTheorem(f"{exc}; {inner}") from exc
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
            if window == "cut":
                copy = frozenset(peeled.nodes) - frozenset(own_k.nodes)
                region = frozenset(peeled.nodes) - copy
                cut = find_cut_region(peeled, region, own_k)
                if cut is None:
                    raise NoInductionTheorem("the residual is not the hypothesis through cut wires")
                replaced = replace_cut_region(peeled, cut, own_k, other_k)
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
                rest = frozenset(peeled.nodes) - region
                local = (
                    frozenset(replaced.nodes) - rest,
                    frozenset(other_peeled.nodes) & frozenset(other_k.nodes),
                    other_k,
                )
            chain = (other_k, k) if window == "cut" else None
            step = _step_theorem(
                replaced, other_peeled, rules, contract=contract, local=local, chain=chain
            )
            return kernel.induction(
                start,
                goal,
                index,
                k,
                base,
                side,
                kernel.refl(peeled),
                region,
                step,
                cut=cut,
                tree=tree_id,
                offset=None if sym is None else str(sym),
            )
        except (NoInductionTheorem, kernel.KernelError, InductionDomainError) as exc:
            reasons.append(f"{label}: {exc}")
    raise NoInductionTheorem("; ".join(reasons))


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
    derivation (:func:`copy_derivation`); else by
    :func:`~archytaszx.semantics.kernel.local_contraction` over ``local`` (the two regions and
    the diagram they read as), else over the chain core of ``chain`` (a hypothesis diagram and
    the step symbol), else by :func:`~archytaszx.semantics.kernel.contraction`."""
    try:
        return kernel.iso(replaced, target)
    except kernel.KernelError:
        pass
    reasons: list[str] = []
    if not contract:
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
