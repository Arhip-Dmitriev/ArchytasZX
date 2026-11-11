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
search) re-applied inside the successors. A step case settled by symbolic contraction gives
none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.engine import RewriteStep, apply
from archytaszx.rewrite.normal_form import normal_form
from archytaszx.rewrite.rule import RewriteError, Rule
from archytaszx.rewrite.rules_library import lookup_rule
from archytaszx.semantics import kernel
from archytaszx.semantics.certificate import compare_structure
from archytaszx.semantics.induction import (
    InductionResult,
    StepDischarge,
    TierOutcome,
    _extract,
    replace_region,
)
from archytaszx.semantics.prove import (
    InductionSide,
    induction_base,
    induction_hypothesis,
    induction_successor,
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
    forward, backward = copy_derivation(left_copy, right_copy, rules)
    there = kernel.chain(replaced, transport(replaced, forward, _ids(tier.left_copy, left_copy)))
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
