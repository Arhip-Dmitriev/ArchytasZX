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

"""The proof kernel: :class:`Theorem` values built only by its inference rules, and their
export to Lean 4.

A :class:`Theorem` states ``lhs = rhs`` for diagrams, every free multiplicity universally
quantified over the naturals; for diagrams with a discard, equality of their CPMs. Only the
rules below construct one:

* :func:`refl` -- ``d = d``;
* :func:`rule` -- ``d = d'`` for one recorded rewrite step, re-applied by the engine at a
  match the matcher rediscovers in ``d``; a pure rule's step may consume no discard;
* :func:`iso` -- ``a = b`` when their comparison views are isomorphic;
* :func:`contraction` -- ``a = b`` when their symbolic contractions agree, every symbol formal;
  :func:`local_contraction` -- the same for the parts of ``a`` and ``b`` off two regions that
  read as one diagram through cut wires in the same places;
* :func:`congr` -- ``c = c'`` from ``a = b`` when a region of ``c`` reads as ``a`` through
  its cut wires and ``c'`` has it replaced by ``b``;
* :func:`symm`, :func:`trans` -- ``trans`` needs the middle diagrams identical id for id;
* :func:`induction` -- ``start = goal`` from a base theorem at index 0, a theorem exposing
  one side's peeled successor at ``k + 1`` as a context around that side at ``k``, and a
  theorem from that context around the other side at ``k`` to the other peeled successor;
  with an offset ``s``, over both sides reading their copy index as ``copy index + s`` for
  every ``s``, the hypothesis at ``s + 1`` (inference kind ``induction_offset``);
* :func:`periodic` -- ``start = goal`` from theorems at ``index = 0 .. p - 1`` when both sides
  scale by one constant from ``index`` to ``index + p`` (the index read only in the scalar and
  one chain whose copy transfer matrix has a scalar power).

:func:`check` turns a step-sequence :class:`~archytaszx.semantics.certificate.Certificate`, a
:class:`~archytaszx.semantics.prove.ProofCertificate` or an
:class:`~archytaszx.semantics.prove.InductionProof` into a Theorem, raising
:class:`KernelError` at the first piece that does not check. An induction certificate of
:mod:`archytaszx.semantics.induction` carries no re-checkable step and is refused.

:func:`lean` emits Lean 4 source. A theorem without free multiplicities goes through
:mod:`archytaszx.semantics.lean_net`: diagrams as tensor networks over ℂ in the bundled
``ArchytasZX`` library, each rule step an instance of a lemma proved there. Otherwise, or when
that export does not cover the theorem, the source has no imports: an opaque ``Diagram``
type, one constant per diagram (a function of its free multiplicities), one axiom per rule
instance, isomorphism, algebraic meet, symbolic contraction, instantiation at 0, peeling and
context plug the kernel checked, and
the theorem as their composition by ``Eq.trans``, ``Eq.symm``, ``congrArg`` and ``induction``.

:func:`symbolic` proves ``left = right`` for every value of the dimensions and phases by the
normaliser of symbolic sums in ``ArchytasZX.SymNet`` (:class:`Symbolic`), through
:mod:`archytaszx.semantics.lean_sym`.
:func:`refutation` reads an UNEQUAL decision's counterexample as a :class:`Refutation`, which
:func:`lean` writes through :mod:`archytaszx.semantics.lean_refute` as a Lean proof that the
diagrams are not equal for every parameter value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx import trace as _trace
from archytaszx.algebra import numfield
from archytaszx.algebra.algebraic import NotAlgebraicError
from archytaszx.algebra.phase import PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import free_mult_symbols
from archytaszx.diagram.compare import isomorphic, isomorphism
from archytaszx.diagram.generators import MIXED_ONLY
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId
from archytaszx.rewrite.engine import RewriteStep
from archytaszx.rewrite.normal_form import comparison_view, views_isomorphic
from archytaszx.rewrite.rule import assumed
from archytaszx.semantics import lean_net, lean_refute, lean_sym
from archytaszx.semantics.certificate import (
    Certificate,
    DerivationKind,
    _replay_one,
    compare_structure,
)
from archytaszx.semantics.induction import (
    CutRegion,
    _bare,
    _cut_extract,
    _extract,
    fed_replacement,
    replace_cut_region,
    replace_region,
    unfed,
)
from archytaszx.semantics.lean_scalar import ScalarUnsupported
from archytaszx.semantics.prove import (
    _INDUCTION_ERRORS,
    InductionProof,
    InductionSide,
    ProofCertificate,
    _crosses,
    induction_base,
    induction_hypothesis,
    induction_successor,
)
from archytaszx.semantics.restrict import Clause, Restriction


class KernelError(Exception):
    """A proof piece the kernel does not accept."""


_TOKEN = object()


@dataclass(frozen=True, eq=False)
class Inference:
    """The rule that built a theorem: ``kind`` is refl, rule, iso, algebraic, contraction,
    congr, symm, trans, induction, induction_offset, subst or shift."""

    kind: str
    premises: tuple[Theorem, ...] = ()
    rule: str = ""
    detail: str = ""
    data: dict[str, object] = field(default_factory=dict)


class Theorem:
    """``lhs = rhs`` for every value of the free multiplicities, under ``assumptions``;
    ``domain`` holds the restriction clauses that produced it and ``origin`` the unrestricted
    pair they were substituted into."""

    __slots__ = ("assumptions", "domain", "inference", "lhs", "origin", "rhs")

    def __init__(
        self,
        token: object,
        lhs: Diagram,
        rhs: Diagram,
        inference: Inference,
        assumptions: tuple[str, ...] = (),
        domain: tuple[Clause, ...] = (),
        origin: tuple[Diagram, Diagram] | None = None,
    ) -> None:
        if token is not _TOKEN:
            raise KernelError("a Theorem is built only by the kernel's inference rules")
        self.lhs, self.rhs = lhs, rhs
        self.inference = inference
        self.assumptions = assumptions
        self.domain = domain
        self.origin = origin

    def statement(self) -> str:
        """The quantifier part of the printed claim: ``for all ...`` over the free
        multiplicities, each restricted symbol by its clause."""
        return statement(self.domain, self.variables, self.assumptions)

    @property
    def variables(self) -> tuple[str, ...]:
        """The free multiplicities of both sides, sorted."""
        return tuple(sorted(free_mult_symbols(self.lhs) | free_mult_symbols(self.rhs)))

    def size(self) -> int:
        """The number of inferences."""
        return 1 + sum(p.size() for p in self.inference.premises)

    def axioms(self) -> list[str]:
        """The rule names of every rule inference, in proof order, repeats kept."""
        if self.inference.kind == "rule":
            return [self.inference.rule]
        return [name for p in self.inference.premises for name in p.axioms()]


def statement(
    domain: Sequence[Clause], variables: Sequence[str], assumptions: Sequence[str] = ()
) -> str:
    """``for all n >= 2``, ``at n = 2``, ``for all n = 2*m + 1 (m >= 0)``, then the dimension
    clauses; empty when nothing is quantified or restricted."""
    from archytaszx.semantics.restrict import DIMENSION, EQ, FINITE, GE, SHIFT, SIDE

    fresh = {c.fresh for c in domain if c.fresh}
    free = [v for v in variables if v not in fresh]
    parts: list[str] = []
    for c in domain:
        if c.form in (SHIFT, SIDE, FINITE):
            parts.append(_hypothesis(c))
            continue
        if c.kind == DIMENSION:
            continue
        if c.form == EQ:
            parts.append(f"at {c.symbol} = {c.c}")
        elif c.form == GE:
            parts.append(f"for all {c.symbol} >= {c.c} ({c.symbol} = {c.expression()})")
        else:
            parts.append(f"for all {c.symbol} = {c.expression()} ({c.fresh} >= 0)")
    if free:
        parts.insert(0, f"for all {', '.join(free)}")
    for c in domain:
        if c.kind != DIMENSION or c.form in (SHIFT, SIDE, FINITE):
            continue
        if c.form == EQ:
            parts.append(f"at {c.symbol} = {c.c}")
        elif f"{c.c} <= {c.symbol}" in assumptions:
            parts.append(f"for {c.symbol} >= {c.c}")
        else:
            parts.append(f"for every {c.symbol}")
    return ", ".join(parts)


def _range_text(name: str, domain: object) -> str:
    """A Domain's interval, lattice and excluded points over ``name``, comma separated."""
    lo, hi = getattr(domain, "lo", None), getattr(domain, "hi", None)
    out: list[str] = []
    if lo is not None:
        out.append(f"{name} {'>' if getattr(domain, 'lo_strict', False) else '>='} {lo}")
    if hi is not None:
        out.append(f"{name} {'<' if getattr(domain, 'hi_strict', False) else '<='} {hi}")
    q = getattr(domain, "q", 0)
    if q:
        out.append(f"{name} in Z" if q == 1 else f"{name} in (1/{q})Z")
    step = getattr(domain, "step", None)
    if step is not None:
        out.append(f"{name} in {getattr(domain, 'offset', 0)} + {step}Z")
    out += [f"{name} != {v}" for v in getattr(domain, "excluded", ())]
    return ", ".join(out)


def _hypothesis(c: Clause) -> str:
    """A shift, side condition or finite clause as the hypothesis the theorem states."""
    from archytaszx.semantics.restrict import FINITE, SHIFT

    if c.form == SHIFT:
        inner = _range_text(c.fresh, c.fresh_domain)
        side = _range_text(c.symbol, c.domain) if c.domain is not None else ""
        extra = f"; {side}" if side else ""
        return f"for all {c.symbol} = {c.expression()} ({inner}{extra})"
    if c.form == FINITE:
        return f"at {c.symbol} in {{{', '.join(str(v) for v in c.values)}}}"
    return f"for {_range_text(c.symbol, c.domain)}"


def _merge(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for group in groups:
        out += [a for a in group if a not in out]
    return tuple(out)


def _same(a: Diagram, b: Diagram) -> bool:
    return compare_structure(a, b).identical


@_trace.traced("kernel.refl")
def refl(d: Diagram) -> Theorem:
    """``d = d``."""
    return Theorem(_TOKEN, d, d, Inference("refl"))


@_trace.traced("kernel.rule")
def rule(d: Diagram, step: RewriteStep) -> Theorem:
    """``d = d'`` by re-applying ``step`` to ``d`` at its rediscovered match; a step of a pure
    rule must consume no discard."""
    from archytaszx.rewrite.rules_discard import MIXED_RULES

    if step.rule_name not in MIXED_RULES and any(
        n in d.nodes and d.nodes[n].generator_type.name in MIXED_ONLY
        for n in step.consumed_node_ids
    ):
        raise KernelError(f"the pure rule {step.rule_name} consumes a discard")
    failure, after = _replay_one(d.copy(), 0, step, True)
    if failure is not None:
        raise KernelError(failure)
    nodes = ", ".join(f"n{int(n)}" for n in sorted(step.consumed_node_ids, key=int))
    equations = tuple(c.equation for c in assumed(step.dimension_constraints))
    inference = Inference("rule", rule=step.rule_name, detail=nodes, data={"step": step})
    return Theorem(_TOKEN, d, after, inference, equations)


@_trace.traced("kernel.iso")
def iso(a: Diagram, b: Diagram) -> Theorem:
    """``a = b`` for isomorphic comparison views."""
    if not (_same(a, b) or views_isomorphic(comparison_view(a), comparison_view(b))):
        raise KernelError("the diagrams have different comparison views")
    return Theorem(_TOKEN, a, b, Inference("iso"))


def _values(d: Diagram) -> list[sp.Expr]:
    """The coefficients of every Z and X spider of ``d`` and its scalar."""
    out = [d.scalar.to_sympy()]
    for node in d.nodes.values():
        if node.phase is None:
            continue
        if not node.phase.dim.is_concrete:
            raise KernelError("a spider parameter at a symbolic dimension")
        out += [node.phase.coefficient(k).to_sympy() for k in range(1, node.phase.dim.to_int())]
    return out


def _keyed(d: Diagram, cert: numfield.FieldCertificate, tokens: dict[str, int]) -> Diagram:
    """A copy of ``d`` with each spider coefficient replaced by an integer naming its field key,
    scalar one and no parameters."""
    out = d.copy()
    out.set_parameters({})
    out.set_scalar(Scalar.one())
    for nid, node in d.nodes.items():
        if node.phase is None:
            continue
        dim = node.phase.dim
        weights = {}
        for k in range(1, dim.to_int()):
            key = str(numfield.key(cert, node.phase.coefficient(k).to_sympy()))
            weights[k] = Scalar(sp.Integer(tokens.setdefault(key, len(tokens) + 2)))
        out.set_phase(nid, PhaseVector.from_weights(dim, weights))
    return out


@_trace.traced("kernel.algebraic")
def algebraic(a: Diagram, b: Diagram, cert: numfield.FieldCertificate | None = None) -> Theorem:
    """``a = b`` when a node renaming carries ``a`` onto ``b`` with every spider coefficient
    and the scalar equal in the number field of ``cert``: ``a = a'`` (``a`` with ``b``'s
    values), then :func:`iso`. ``cert`` is built from the values when not given; it is
    checked either way."""
    if a.bang_boxes or b.bang_boxes:
        raise KernelError("an algebraic meet of diagrams with bang boxes")
    values = _values(a) + _values(b)
    try:
        if cert is None:
            cert = numfield.certificate(values)
        numfield.check(cert)
        tokens: dict[str, int] = {}
        ka, kb = _keyed(a, cert, tokens), _keyed(b, cert, tokens)
        same_scalar = numfield.key(cert, a.scalar.to_sympy()) == numfield.key(
            cert, b.scalar.to_sympy()
        )
    except (numfield.NumberFieldError, NotAlgebraicError) as exc:
        raise KernelError(f"no number-field certificate: {exc}") from exc
    nodes = isomorphism(ka, kb, symmetric_legs=True)
    if nodes is None or not same_scalar:
        raise KernelError("the diagrams differ beyond equal algebraic values")
    target = a.copy()
    for nid in a.nodes:
        target.set_phase(nid, b.nodes[nodes[nid]].phase)
    target.set_scalar(b.scalar)
    step = Theorem(_TOKEN, a, target, Inference("algebraic", data={"field": cert}))
    return trans(step, iso(target, b))


@_trace.traced("kernel.contraction")
def contraction(a: Diagram, b: Diagram, max_steps: int = 64) -> Theorem:
    """``a = b`` when the symbolic contractions of both (both doubled when either has a
    discard), every symbol formal, agree."""
    from archytaszx.algebra.scalar import ScalarBudgetError
    from archytaszx.semantics.check import compare_symbolic
    from archytaszx.semantics.contract_symbolic import (
        SymbolicContractionDomainError,
        SymbolicContractionUnsupportedError,
        contract_symbolic_pair,
    )

    try:
        left, right = contract_symbolic_pair(a, b, max_steps=max_steps)
    except (
        SymbolicContractionUnsupportedError,
        SymbolicContractionDomainError,
        ScalarBudgetError,
    ) as exc:
        raise KernelError(f"no symbolic contraction: {exc}") from exc
    comparison = compare_symbolic(left, right)
    if not comparison.matched:
        raise KernelError(f"the symbolic contractions differ: {comparison.reason}")
    return Theorem(_TOKEN, a, b, Inference("contraction"))


@_trace.traced("kernel.w_fusion")
def w_fusion(d: Diagram) -> Theorem:
    """``d = d'`` for ``d'`` the :func:`~archytaszx.semantics.induction.w_fused` image of ``d``:
    a top-level box fed by one output leg of a W node, its copy re-checked by exact symbolic
    contraction to be ``c`` W(1 -> m) after a Z with weights ``w`` (``w_0 = 1``), fused into
    that leg (W associativity; the weighted Z moved before the W node, its only output)."""
    from archytaszx.semantics.induction import w_fused

    try:
        after = w_fused(d)
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"w_fusion raised {type(exc).__name__}: {exc}") from exc
    return Theorem(_TOKEN, d, after, Inference("w_fusion"))


def _plugged(d: Diagram, region: frozenset[NodeId], hypothesis: Diagram) -> tuple[Diagram, object]:
    """The part of ``d`` off ``region`` with its cut ends open, ``region`` read as
    ``hypothesis`` through cut wires; and where the region sits in ``d``'s boundary lists."""
    from archytaszx.semantics.induction import _copy_side, find_cut_region

    cut = find_cut_region(d, region, hypothesis)
    if cut is None:
        raise KernelError("the region is not the hypothesis diagram through cut wires")
    copy, held = _copy_side(d, cut, d.scalar * hypothesis.scalar**-1)
    place = {r: (0, i) for i, r in enumerate(cut.inputs)}
    place.update({r: (1, i) for i, r in enumerate(cut.outputs)})
    slots = tuple(
        place[r] for refs in (d.boundary_inputs, d.boundary_outputs) for r in refs if r in place
    )
    return copy, (held, slots)


@_trace.traced("kernel.contraction")
def local_contraction(
    a: Diagram,
    region_a: frozenset[NodeId],
    b: Diagram,
    region_b: frozenset[NodeId],
    hypothesis: Diagram,
    max_steps: int = 64,
) -> Theorem:
    """``a = b`` when ``region_a`` of ``a`` and ``region_b`` of ``b`` read as ``hypothesis``
    through cut wires in the same places, and the rest of each, cut ends open, meet by
    :func:`contraction`."""
    from archytaszx.semantics.induction import _interface

    copy_a, at_a = _plugged(a, region_a, hypothesis)
    copy_b, at_b = _plugged(b, region_b, hypothesis)
    if at_a != at_b:
        raise KernelError("the regions sit in different places of the two diagrams")
    if _interface(copy_a) != _interface(copy_b):
        raise KernelError("the parts off the regions have different interfaces")
    try:
        contraction(copy_a, copy_b, max_steps)
    except KernelError as exc:
        if "step budget" not in str(exc):
            raise
        contraction(copy_a, copy_b, 4 * max_steps)
    data: dict[str, object] = {
        "region_a": region_a,
        "region_b": region_b,
        "hypothesis": hypothesis,
        "parts": (copy_a, copy_b),
    }
    return Theorem(_TOKEN, a, b, Inference("contraction", data=data))


@_trace.traced("kernel.congr")
def congr(t: Theorem, context: Diagram, cut: CutRegion) -> Theorem:
    """``context = context'`` where ``cut``'s region of ``context`` reads as ``t.lhs`` through
    its cut wires and ``context'`` has it replaced by ``t.rhs``."""
    from archytaszx.semantics.induction import _cut_wires

    region = cut.region
    if not region <= frozenset(context.nodes):
        raise KernelError("congr: the region is not part of the context")
    cuts = _cut_wires(context, region)
    if dict(cut.cuts) != cuts:
        raise KernelError("congr: the cut wires are not the region's")
    held = [r for refs in (context.boundary_inputs, context.boundary_outputs) for r in refs]
    for refs, direction in ((cut.inputs, Direction.INPUT), (cut.outputs, Direction.OUTPUT)):
        wanted = [r for r in held if r.node_id in region and r.direction is direction]
        wanted += [r for r in cuts if r.direction is direction]
        if len(refs) != len(wanted) or set(refs) != set(wanted):
            raise KernelError("congr: the region's boundary is not its slots and cut ends")
    if t.lhs.scalar.is_zero:
        raise KernelError("congr: the theorem's left side has scalar zero")
    try:
        if not isomorphic(_bare(_cut_extract(context, cut)), _bare(t.lhs)):
            raise KernelError("congr: the region is not the theorem's left side")
        plugged = replace_cut_region(context, cut, t.lhs, t.rhs, contiguous=False)
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"congr raised {type(exc).__name__}: {exc}") from exc
    data: dict[str, object] = {"cut": cut}
    return Theorem(_TOKEN, context, plugged, Inference("congr", (t,), data=data), t.assumptions)


@_trace.traced("kernel.symm")
def symm(t: Theorem) -> Theorem:
    """``rhs = lhs`` from ``lhs = rhs``."""
    return Theorem(_TOKEN, t.rhs, t.lhs, Inference("symm", (t,)), t.assumptions)


@_trace.traced("kernel.trans")
def trans(a: Theorem, b: Theorem) -> Theorem:
    """``a.lhs = b.rhs`` when ``a.rhs`` is ``b.lhs``."""
    if not _same(a.rhs, b.lhs):
        raise KernelError("trans: the middle diagrams differ")
    return Theorem(
        _TOKEN, a.lhs, b.rhs, Inference("trans", (a, b)), _merge(a.assumptions, b.assumptions)
    )


def chain(start: Diagram, steps: Sequence[RewriteStep]) -> Theorem:
    """``start = d_n`` through every step in order."""
    out: Theorem | None = None
    current = start
    for step in steps:
        nxt = rule(current, step)
        out = nxt if out is None else trans(out, nxt)
        current = nxt.rhs
    return out if out is not None else refl(start)


@_trace.traced("kernel.induction")
def induction(
    start: Diagram,
    goal: Diagram,
    index: str,
    step_symbol: str,
    base: Theorem,
    side: InductionSide,
    expose: Theorem,
    region: frozenset[NodeId],
    step: Theorem,
    *,
    cut: CutRegion | None = None,
    tree: BangBoxId | None = None,
    offset: str | None = None,
) -> Theorem:
    """``start = goal`` for every value of ``index`` (see the module docstring); with ``cut``
    the region meets the rest at cut wires (a chain), with ``tree`` (a tree box id) it is the
    exposed diagram without the box that tree feeds. With ``offset`` (a fresh symbol ``s``)
    the claim inducted on is both sides with the box carrying ``index`` reading its copy index
    as ``copy index + s``, for every ``s``: at 0 both sides must not depend on ``s``, the
    successors read ``+ s`` and the hypothesis diagrams ``+ s + 1``."""
    live = free_mult_symbols(start) | free_mult_symbols(goal)
    if index not in live or step_symbol in live:
        raise KernelError(f"{index!r} is not a free multiplicity or {step_symbol!r} is not fresh")
    k = step_symbol
    try:
        if not (_same(base.lhs, induction_base(start, index))):
            raise KernelError("the base theorem does not start at the start side at 0")
        if not _same(base.rhs, induction_base(goal, index)):
            raise KernelError("the base theorem does not end at the goal side at 0")
        own, other = (start, goal) if side is InductionSide.START else (goal, start)
        shift = _offset_shift(start, goal, index, k, offset)
        own_s, other_s = own, other
        if shift is not None:
            own_s, other_s = shift(own, index, 0), shift(other, index, 0)
        peeled = induction_successor(own_s, index, k)
        other_peeled = induction_successor(other_s, index, k)
        if peeled is None or other_peeled is None:
            raise KernelError("a successor has no single box to peel")
        if not _same(expose.lhs, peeled):
            raise KernelError("the exposing theorem does not start at the peeled successor")
        exposed = expose.rhs
        hypothesis = induction_hypothesis(own, index, k)
        other_k = induction_hypothesis(other, index, k)
        if shift is not None:
            hypothesis, other_k = shift(hypothesis, k, 1), shift(other_k, k, 1)
        if cut is not None:
            if cut.region != region or not region <= frozenset(exposed.nodes):
                raise KernelError("the cut region is not the hypothesis region")
            if not isomorphic(_bare(_cut_extract(exposed, cut)), _bare(hypothesis)):
                raise KernelError("the cut region is not the hypothesis diagram")
            replaced = replace_cut_region(exposed, cut, hypothesis, other_k)
        elif tree is not None:
            tree_id = tree
            rest = unfed(exposed, tree_id)
            if region != frozenset(rest.nodes) or not isomorphic(rest, hypothesis):
                raise KernelError("the unfed exposed diagram is not the hypothesis diagram")
            replaced = fed_replacement(exposed, tree_id, hypothesis, other_k)
        else:
            if not region <= frozenset(exposed.nodes) or _crosses(exposed, region):
                raise KernelError(
                    "the hypothesis region is not a closed part of the exposed diagram"
                )
            if not isomorphic(_bare(_extract(exposed, region, with_boxes=True)), _bare(hypothesis)):
                raise KernelError("the hypothesis region is not the hypothesis diagram")
            replaced = replace_region(exposed, region, hypothesis, other_k)
        if not (_same(step.lhs, replaced) and _same(step.rhs, other_peeled)):
            raise KernelError("the step theorem does not run from the replaced to the other side")
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"rebuilding the induction raised {type(exc).__name__}: {exc}") from exc
    data: dict[str, object] = {
        "index": index,
        "step": k,
        "side": side,
        "peeled": peeled,
        "other_peeled": other_peeled,
        "own_k": hypothesis,
        "other_k": other_k,
        "replaced": replaced,
        "region": region,
        "exposed": exposed,
        "window": "cut" if cut is not None else "tree" if tree is not None else "region",
    }
    kind = "induction"
    if offset is not None:
        data["offset"] = offset
        kind = "induction_offset"
    assumptions = _merge(base.assumptions, expose.assumptions, step.assumptions)
    return Theorem(
        _TOKEN, start, goal, Inference(kind, (base, expose, step), data=data), assumptions
    )


@_trace.traced("kernel.periodic")
def periodic(start: Diagram, goal: Diagram, index: str, bases: Sequence[Theorem]) -> Theorem:
    """``start = goal`` for every value of ``index`` from ``bases[r]``, ``start = goal`` at
    ``index = r`` for ``r < p = len(bases)``, when both sides scale by one constant from
    ``index`` to ``index + p`` (:func:`~archytaszx.semantics.periodic.scaling`: ``index`` read
    only in the scalar and the multiplicity of at most one top-level chain box whose copy
    transfer matrix has a scalar power with period dividing ``p``)."""
    from archytaszx.semantics.decide import _instantiate_index
    from archytaszx.semantics.periodic import PeriodicError, scaling

    live = free_mult_symbols(start) | free_mult_symbols(goal)
    if index not in live or not bases:
        raise KernelError(f"{index!r} is not a free multiplicity or there is no base theorem")
    if any(b.domain for b in bases):
        raise KernelError("a base theorem holds under a restriction")
    step = len(bases)
    try:
        for value, base in enumerate(bases):
            if not _same(base.lhs, _instantiate_index(start, index, value)):
                raise KernelError(f"base {value} does not start at the start side at {value}")
            if not _same(base.rhs, _instantiate_index(goal, index, value)):
                raise KernelError(f"base {value} does not end at the goal side at {value}")
        ratios = [scaling(side, index, step) for side in (start, goal)]
    except PeriodicError as exc:
        raise KernelError(f"periodic: {exc}") from exc
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"rebuilding a base raised {type(exc).__name__}: {exc}") from exc
    if sp.simplify(ratios[0] - ratios[1]) != 0:
        raise KernelError(f"periodic: the sides scale by {ratios[0]} and {ratios[1]}")
    data: dict[str, object] = {"index": index, "step": step, "ratio": ratios[0]}
    assumptions = _merge(*(b.assumptions for b in bases))
    return Theorem(_TOKEN, start, goal, Inference("periodic", tuple(bases), data=data), assumptions)


def _offset_shift(
    start: Diagram, goal: Diagram, index: str, step_symbol: str, offset: str | None
) -> Callable[[Diagram, str, int], Diagram] | None:
    """None without ``offset``; else, once ``offset`` is checked fresh and both sides at
    ``index = 0`` are unchanged by it, a function taking a diagram, the multiplicity its box
    carries and ``c`` to that diagram reading ``copy index + offset + c``."""
    from archytaszx.semantics.induction import _all_free_symbols
    from archytaszx.semantics.prove import induction_offset

    if offset is None:
        return None
    names = _all_free_symbols(start) | _all_free_symbols(goal) | {index, step_symbol}
    names |= {b.index for d in (start, goal) for b in d.bang_boxes.values() if b.index}
    if offset in names:
        raise KernelError(f"the offset {offset!r} is not fresh")
    sym = sp.Symbol(offset, integer=True, nonnegative=True)
    for d in (start, goal):
        at_zero = induction_base(induction_offset(d, index, sym), index)
        if not _same(at_zero, induction_base(d, index)):
            raise KernelError("a side at 0 depends on the offset")

    def shifted(d: Diagram, carried: str, c: int) -> Diagram:
        return induction_offset(d, carried, sym + c)

    return shifted


@_trace.traced("kernel.subst")
def subst(
    t: Theorem, clause: Clause, lhs: Diagram | None = None, rhs: Diagram | None = None
) -> Theorem:
    """``lhs' = rhs'`` from ``t`` and one clause ``n = e`` or ``d = v``: both sides recomputed
    from ``t``'s by :func:`~archytaszx.semantics.restrict.apply_clause` and compared id for id
    with ``lhs`` and ``rhs`` when stated."""
    from archytaszx.semantics.restrict import _all_symbols, apply_clause

    if not clause.substitutes:
        raise KernelError(f"subst: {clause.text()} substitutes nothing")
    if clause.fresh and clause.fresh in _all_symbols(t.lhs, t.rhs):
        raise KernelError(f"subst: {clause.fresh} is not fresh for the theorem")
    try:
        new_l, new_r = apply_clause(t.lhs, clause), apply_clause(t.rhs, clause)
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"subst raised {type(exc).__name__}: {exc}") from exc
    for stated, computed in ((lhs, new_l), (rhs, new_r)):
        if stated is not None and not _same(stated, computed):
            raise KernelError(f"subst: a stated side is not the theorem's side at {clause.text()}")
    inference = Inference("subst", (t,), detail=clause.text(), data={"clause": clause})
    origin = t.origin if t.origin is not None else (t.lhs, t.rhs)
    return Theorem(
        _TOKEN,
        new_l if lhs is None else lhs,
        new_r if rhs is None else rhs,
        inference,
        t.assumptions,
        (*t.domain, clause),
        origin,
    )


def restricted(
    t: Theorem, left: Diagram, right: Diagram, restriction: Restriction, used: Sequence[str] = ()
) -> Theorem:
    """``t`` read as the restricted claim of ``left = right``: its sides must be the
    restricted pair, or isomorphic to it; ``used`` names the floors the proof assumed."""
    from archytaszx.semantics.restrict import subst_pair

    try:
        pair = subst_pair(left, right, restriction)
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"restricting raised {type(exc).__name__}: {exc}") from exc
    for side, target in ((t.lhs, pair[0]), (t.rhs, pair[1])):
        if not (
            _same(side, target) or views_isomorphic(comparison_view(side), comparison_view(target))
        ):
            raise KernelError("the theorem is not about the restricted pair")
    floors = _floors_used(restriction, used)
    return Theorem(
        _TOKEN,
        t.lhs,
        t.rhs,
        t.inference,
        _merge(t.assumptions, floors),
        restriction.clauses,
        (left, right),
    )


def _floors_used(restriction: Restriction, used: Sequence[str]) -> tuple[str, ...]:
    """The ``c <= d`` floors of the dimension clauses ``d >= c`` a proof assumed."""
    from archytaszx.semantics.restrict import DIMENSION, GE

    return tuple(
        f"{c.c} <= {c.symbol}"
        for c in restriction.clauses
        if c.kind == DIMENSION and c.form == GE and c.symbol in used
    )


@_trace.traced("kernel.shift")
def shift(t: Theorem, left: Diagram, right: Diagram, clause: Clause) -> Theorem:
    """``left = right`` at ``n = a*m + c`` for every ``m``, from ``t`` about both sides with the
    clause ``n >= c`` or ``n = a*m + c`` substituted: each side rebuilt by
    :func:`~archytaszx.semantics.restrict.apply_clause` and matched to ``t``'s side id for id or
    by isomorphic comparison views; at ``n = c``, :func:`lift`."""
    from archytaszx.semantics.restrict import EQ, MULTIPLICITY, _all_symbols, apply_clause

    if clause.kind == MULTIPLICITY and clause.form == EQ:
        return lift(t, left, right, (clause,))
    if clause.kind != MULTIPLICITY or not clause.fresh or (clause.a == 1 and clause.c == 0):
        raise KernelError(f"shift: {clause.text()} is not n = a*m + c with a > 1 or c >= 1")
    live = free_mult_symbols(left) | free_mult_symbols(right)
    if clause.symbol not in live or clause.fresh in _all_symbols(left, right):
        raise KernelError(
            f"{clause.symbol!r} is not a free multiplicity or {clause.fresh!r} is not fresh"
        )
    try:
        peeled = apply_clause(left, clause)
        other = apply_clause(right, clause)
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"rebuilding the successor raised {type(exc).__name__}: {exc}") from exc
    body = _onto(t, peeled, other)
    data: dict[str, object] = {
        "index": clause.symbol,
        "step": clause.fresh,
        "count": clause.c,
        "scale": clause.a,
        "peeled": peeled,
        "other_peeled": other,
    }
    domain = (clause,)
    inference = Inference("shift", (body,), data=data)
    return Theorem(_TOKEN, left, right, inference, body.assumptions, domain, (left, right))


@_trace.traced("kernel.lift")
def lift(t: Theorem, left: Diagram, right: Diagram, clauses: Sequence[Clause]) -> Theorem:
    """``left = right`` under the multiplicity clauses ``clauses`` (``n = c``, ``n >= c``,
    ``n = a*m + c``), from ``t`` about both sides with the clauses substituted in order, matched
    to ``t``'s sides id for id or by isomorphic comparison views; the inference ``shift`` with
    scale 0, each clause as ``(n, a, c, m)`` under ``lift`` (``a = 0`` for ``n = c``)."""
    from archytaszx.semantics.restrict import EQ, MULTIPLICITY, _all_symbols, apply_clause

    live = free_mult_symbols(left) | free_mult_symbols(right)
    taken = _all_symbols(left, right)
    for clause in clauses:
        if clause.kind != MULTIPLICITY:
            raise KernelError(f"lift: {clause.text()} is not a multiplicity clause")
        if clause.symbol not in live:
            raise KernelError(f"{clause.symbol!r} is not a free multiplicity")
        if clause.form != EQ and (not clause.fresh or clause.fresh in taken):
            raise KernelError(f"lift: {clause.text()} has no fresh symbol")
    if len({c.symbol for c in clauses}) != len(clauses) or not clauses:
        raise KernelError("lift: one clause per multiplicity")
    try:
        peeled, other = left, right
        for clause in clauses:
            peeled, other = apply_clause(peeled, clause), apply_clause(other, clause)
    except _INDUCTION_ERRORS as exc:
        raise KernelError(f"rebuilding the instance raised {type(exc).__name__}: {exc}") from exc
    body = _onto(t, peeled, other)
    data: dict[str, object] = {
        "index": clauses[0].symbol,
        "step": None,
        "count": clauses[0].c,
        "scale": 0,
        "fixed": tuple((c.symbol, c.c) for c in clauses if c.form == EQ),
        "lift": tuple(
            (c.symbol, 0, c.c, None) if c.form == EQ else (c.symbol, c.a, c.c, c.fresh)
            for c in clauses
        ),
        "peeled": peeled,
        "other_peeled": other,
    }
    inference = Inference("shift", (body,), data=data)
    return Theorem(_TOKEN, left, right, inference, body.assumptions, tuple(clauses), (left, right))


@dataclass(frozen=True, eq=False)
class RestrictedProof:
    """A proof of ``left = right`` under ``restriction``: ``family`` set when ``proof`` proves
    the unrestricted claim (specialised by subst), otherwise a proof about the restricted
    pair; ``floors_used`` names the floors a normaliser proof assumed."""

    proof: object
    restriction: Restriction
    left: Diagram
    right: Diagram
    family: bool
    floors_used: tuple[str, ...] = ()


@_trace.traced("proof.kernel_check")
def check(artefact: object) -> Theorem:
    """The theorem a certificate, proof certificate, induction proof or restricted proof
    establishes."""
    if isinstance(artefact, Theorem):
        return artefact
    if isinstance(artefact, RestrictedProof):
        inner = check(artefact.proof)
        if not artefact.family:
            return restricted(
                inner, artefact.left, artefact.right, artefact.restriction, artefact.floors_used
            )
        out = inner
        for clause in artefact.restriction.clauses:
            if clause.substitutes:
                out = subst(out, clause)
        return Theorem(
            _TOKEN,
            out.lhs,
            out.rhs,
            out.inference,
            out.assumptions,
            artefact.restriction.clauses,
            (inner.lhs, inner.rhs),
        )
    if isinstance(artefact, Certificate):
        if artefact.derivation.kind is not DerivationKind.STEP_SEQUENCE:
            raise KernelError("an induction certificate carries no re-checkable step")
        out = chain(artefact.initial, artefact.steps)
        if not _same(out.rhs, artefact.final):
            raise KernelError("the steps do not reach the recorded final diagram")
        return out
    if isinstance(artefact, ProofCertificate):
        for half, end in ((artefact.forward, artefact.start), (artefact.backward, artefact.goal)):
            if not _same(half.initial, end):
                raise KernelError("a proof half does not begin at its end of the goal")
        forward = check(artefact.forward)
        backward = check(artefact.backward)
        try:
            meet = iso(forward.rhs, backward.rhs)
        except KernelError as exc:
            try:
                meet = algebraic(forward.rhs, backward.rhs)
            except KernelError:
                raise exc from None
        return trans(forward, trans(meet, symm(backward)))
    if isinstance(artefact, InductionProof):
        return induction(
            artefact.start,
            artefact.goal,
            artefact.index,
            artefact.step_symbol,
            check(artefact.base),
            artefact.side,
            check(artefact.expose),
            artefact.region,
            check(artefact.step),
        )
    raise KernelError(f"cannot check a {type(artefact).__name__}")


@dataclass(frozen=True, eq=False)
class Refutation:
    """The claim ``lhs = rhs`` for every parameter value, refuted at the values ``point``;
    ``source`` is its Lean export when already written."""

    lhs: Diagram
    rhs: Diagram
    point: Mapping[str, object]
    source: str = ""


@dataclass(frozen=True, eq=False)
class Symbolic:
    """``lhs = rhs`` for every value of the dimensions and phases; ``source`` is its Lean
    proof by the normaliser of symbolic sums; ``domain`` holds the restriction clauses it was
    read under and ``assumptions`` the floors it assumed."""

    lhs: Diagram
    rhs: Diagram
    source: str
    domain: tuple[Clause, ...] = ()
    assumptions: tuple[str, ...] = ()

    @property
    def variables(self) -> tuple[str, ...]:
        """The free multiplicities of both sides, sorted."""
        return tuple(sorted(free_mult_symbols(self.lhs) | free_mult_symbols(self.rhs)))

    def statement(self) -> str:
        """The quantifier part of the printed claim, as :meth:`Theorem.statement`."""
        return statement(self.domain, self.variables, self.assumptions)


def restricted_symbolic(
    proof: RestrictedProof, describe: Callable[[Diagram], str] | None = None
) -> Symbolic:
    """The restricted claim of a :class:`RestrictedProof` over a normaliser proof: the family
    proof followed by its corollary at the clauses, or the restricted pair's proof under the
    floors it assumed."""
    from archytaszx.semantics.lean_restrict import RestrictUnsupported, corollary, header

    inner = proof.proof
    if not isinstance(inner, Symbolic):
        raise KernelError("the restricted proof is not a normaliser proof")
    text = proof.restriction.text()
    if proof.family:
        source = header(
            inner.source, f"restricted to {text}: the family theorem, then {_RESTRICTED}"
        )
        try:
            source = corollary(source, proof.restriction.clauses, "main", _var)
        except RestrictUnsupported as exc:
            raise KernelError(f"no corollary: {exc}") from exc
        return Symbolic(inner.lhs, inner.rhs, source, proof.restriction.clauses)
    floors = _floors_used(proof.restriction, proof.floors_used)
    shifted = _symbolic_shift(proof, describe)
    if shifted is not None:
        source = header(
            shifted, _pair_note(proof.restriction) + f"; {_lifted_note(proof.restriction)}"
        )
        return Symbolic(inner.lhs, inner.rhs, source, proof.restriction.clauses, floors)
    source = header(inner.source, _pair_note(proof.restriction) + _unlinked(proof.restriction))
    return Symbolic(inner.lhs, inner.rhs, source, proof.restriction.clauses, floors)


def _symbolic_shift(
    proof: RestrictedProof, describe: Callable[[Diagram], str] | None
) -> str | None:
    """The normaliser's proof over the original diagrams: the family lemma of both networks,
    along each multiplicity clause in order split ``c`` times, and zeroed at ``n = c``; lifted
    by ``Equal.of_split_at`` and ``Equal.of_zero_at``, with the corollary over each ``n`` of a
    clause ``n >= c`` (``n = m + c``); at its clauses ``d = v``, restated over the original
    diagrams. None at a clause ``n = a*m + c`` with ``a > 1``, or when the normaliser
    declines."""
    from archytaszx.semantics.decide import dimension_floors
    from archytaszx.semantics.lean_restrict import RestrictUnsupported, shift_corollary
    from archytaszx.semantics.restrict import DIMENSION, EQ, MULTIPLICITY, apply_clause

    subs = [c for c in proof.restriction.clauses if c.substitutes]
    dims = [c for c in subs if c.kind == DIMENSION and c.form == EQ]
    mults = [c for c in subs if c not in dims]
    if not subs or any(c.kind != MULTIPLICITY or (c.form != EQ and c.a != 1) for c in mults):
        return None
    steps = [c for c in mults if c.form != EQ]
    single = len(mults) == 1 and bool(steps)
    if single and steps[0].c < 1:
        return None
    floors = dimension_floors(proof.left, proof.right)
    for name, low in proof.restriction.floors.items():
        if name in proof.floors_used:
            floors[name] = max(floors.get(name, 1), low)
    try:
        left, right = proof.left, proof.right
        for c in dims:
            left, right = apply_clause(left, c), apply_clause(right, c)
        for c in dims:
            floors.pop(c.symbol, None)
        original = (
            ({c.symbol: c.c for c in dims}, [(left, proof.left), (right, proof.right)])
            if dims
            else None
        )
        if not mults and original is None:
            return None
        source = lean_sym.export_symbolic(
            left,
            right,
            describe=describe,
            floors=floors,
            shift=(steps[0].symbol, steps[0].c) if single else None,
            original=original,
            fixed=None if single or not mults else [(c.symbol, c.c, c.form == EQ) for c in mults],
        )
        return shift_corollary(source, steps) if steps else source
    except (lean_net.NetUnsupported, ScalarUnsupported, RestrictUnsupported, KernelError):
        return None


_RESTRICTED = "its corollary at the clauses"


def _pair_note(restriction: Restriction) -> str:
    subs = [f"{c.symbol} = {c.expression()}" for c in restriction.clauses if c.substitutes]
    return f"restricted to {restriction.text()}: the theorem about the restricted pair" + (
        f", substituted {', '.join(subs)}" if subs else ""
    )


def symbolic(
    left: Diagram, right: Diagram, describe: Callable[[Diagram], str] | None = None
) -> Symbolic:
    """The normaliser's proof of ``left = right``."""
    try:
        return Symbolic(left, right, lean_sym.export_symbolic(left, right, describe=describe))
    except (lean_net.NetUnsupported, ScalarUnsupported) as exc:
        raise KernelError(f"no symbolic proof: {exc}") from exc


def refutation(decision: object, left: Diagram, right: Diagram) -> Refutation:
    """The refutation an UNEQUAL decision's counterexample gives."""
    from archytaszx.semantics.decide import Decision, EqualityVerdict

    if not isinstance(decision, Decision):
        raise KernelError(f"cannot read a {type(decision).__name__}")
    if decision.verdict is not EqualityVerdict.UNEQUAL or decision.counterexample is None:
        raise KernelError("the decision carries no counterexample")
    return Refutation(left, right, dict(decision.counterexample))


@_trace.traced("proof.build")
def from_decision(decision: object, left: Diagram, right: Diagram) -> Theorem:
    """``left = right`` from a decision's normal-form or saturation certificates, its induction
    theorem, or :func:`contraction` for a symbolic-contraction verdict. A DOUBLED decision gives
    the theorem over its doubled pair, read through its settling rung; ``left`` and ``right``
    are that pair or the diagrams it doubles."""
    from dataclasses import replace

    from archytaszx.diagram.mixed import double
    from archytaszx.semantics.decide import Decision, DecisionMethod

    if not isinstance(decision, Decision):
        raise KernelError(f"cannot read a {type(decision).__name__}")
    if decision.method is DecisionMethod.DOUBLED:
        pair, inner = decision.doubled, decision.inner_method
        if pair is None or inner is None or inner is DecisionMethod.DOUBLED:
            raise KernelError("a doubled verdict without its doubled pair and settling rung")
        if not (_same(left, pair[0]) and _same(right, pair[1])):
            for given, held in ((left, pair[0]), (right, pair[1])):
                if not (_same(double(given), held) or _same(given, held)):
                    raise KernelError("the doubled pair is not the doubling of left and right")
        return from_decision(replace(decision, method=inner, inner_method=None), *pair)
    if decision.method is DecisionMethod.INDUCTION and decision.induction_proof is not None:
        proved = check(decision.induction_proof)
        _remember_induction(proved)
        return trans(trans(iso(left, proved.lhs), proved), iso(proved.rhs, right))
    if decision.method is DecisionMethod.NORMAL_FORM and len(decision.certificates) == 2:
        to_l, to_r = (check(c) for c in decision.certificates)
        return trans(
            trans(iso(left, to_l.lhs), to_l),
            trans(iso(to_l.rhs, to_r.rhs), trans(symm(to_r), iso(to_r.lhs, right))),
        )
    if decision.method is DecisionMethod.SATURATION:
        out = refl(left)
        for certificate in decision.certificates:
            edge = check(certificate)
            cur = out.rhs
            if views_isomorphic(comparison_view(cur), comparison_view(edge.lhs)):
                out = trans(out, trans(iso(cur, edge.lhs), edge))
            elif views_isomorphic(comparison_view(cur), comparison_view(edge.rhs)):
                out = trans(out, trans(iso(cur, edge.rhs), symm(edge)))
            else:
                raise KernelError("a saturation edge does not continue the path")
        return trans(out, iso(out.rhs, right))
    if decision.method is DecisionMethod.SYMBOLIC_CONTRACTION:
        return contraction(left, right)
    raise KernelError(f"a {decision.method.value} verdict carries no re-checkable proof")


_LEAN_KEYWORDS = frozenset(
    {"at", "axiom", "by", "def", "do", "else", "end", "from", "fun", "have", "if", "in"}
    | {"let", "match", "show", "then", "theorem", "with"}
)


def _var(name: str) -> str:
    """A Lean binder for a multiplicity name."""
    plain = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not re.match(r"[A-Za-z_]", plain) or plain in _LEAN_KEYWORDS:
        plain = "v_" + plain
    return plain


def _comment(text: str) -> str:
    return text.replace("-/", "- /").replace("/-", "/ -")


class _Lean:
    """Accumulates declarations while proof terms are built."""

    def __init__(self, describe: Callable[[Diagram], str], semantic: bool = False) -> None:
        self.describe = describe
        self.semantic = semantic
        self.decls: list[str] = []
        self.diagrams: list[tuple[Diagram, str, tuple[str, ...]]] = []
        self.stated: dict[str, str] = {}
        self.count = 0
        self.subs: list[str] = []
        self.imports: list[str] = []
        self.depth = 0

    def inducted(self, t: Theorem, x: Diagram, y: Diagram, binders: Sequence[str]) -> str | None:
        """A contraction step over boxed diagrams through the induction theorem of
        :func:`_contraction_induction`, stated as a lemma ``fam<i>`` over its pair: at a shift
        ``m = n + c`` applied at ``n + c`` between instantiation axioms onto ``x`` and ``y``;
        the term itself when ``x``, ``y`` are the step's sides, else the axiom ``net<i>`` from
        a lemma ``fam<j> : x = y`` to the step; None at nesting depth :data:`INDUCTED_DEPTH`
        or when there is no such theorem."""
        if self.depth >= INDUCTED_DEPTH:
            return None
        found = _contraction_induction(x, y)
        if found is None:
            return None
        proof, shift = found
        self.depth += 1
        try:
            pb = [_var(v) for v in proof.variables]
            body = self.term(proof)
        finally:
            self.depth -= 1
        lemma = self.fresh("fam")
        stmt = f"{self.call(proof.lhs)} = {self.call(proof.rhs)}"
        head = f"∀ ({' '.join(pb)} : Nat), {stmt}" if pb else stmt
        value = f"fun {' '.join(pb)} => {body}" if pb else body
        self.decls.append(f"theorem {lemma} : {head} :=\n  {value}")
        xb = [_var(v) for v in sorted(free_mult_symbols(x) | free_mult_symbols(y))]
        if shift is None:
            applied = f"({lemma} {' '.join(pb)})" if pb else lemma
        else:
            index, fresh, c = shift
            at = {fresh: f"({_var(index)} + {c})"}
            args = [at.get(v, _var(v)) for v in proof.variables]
            note = f"instantiation at {fresh} = {index} + {c}"
            inst_l = self.axiom("inst", note, xb, self.call(proof.lhs, at), self.call(x))
            inst_r = self.axiom("inst", note, xb, self.call(proof.rhs, at), self.call(y))
            applied = f"({inst_l}).symm.trans (({lemma} {' '.join(args)}).trans ({inst_r}))"
        if "parts" not in t.inference.data:
            return applied
        if shift is not None:
            lemma = self.fresh("fam")
            stmt = f"{self.call(x)} = {self.call(y)}"
            head = f"∀ ({' '.join(xb)} : Nat), {stmt}" if xb else stmt
            value = f"fun {' '.join(xb)} => {applied}" if xb else applied
            self.decls.append(f"theorem {lemma} : {head} :=\n  {value}")
        name = self.fresh("net")
        lhs, rhs = self.call(t.lhs), self.call(t.rhs)
        quant = f"∀ ({' '.join(binders)} : Nat), " if binders else ""
        note = (
            f"the diagrams are the parts off the two regions, plugged into one context, which "
            f"{lemma} proves equal"
        )
        self.decls.append(
            f"/-- {_comment(note)} -/\naxiom {name} : (type_of% @{lemma}) → {quant}{lhs} = {rhs}"
        )
        return f"({name} {lemma} {' '.join(binders)})" if binders else f"({name} {lemma})"

    def contraction(self, t: Theorem, binders: Sequence[str]) -> str | None:
        """A contraction step through a semantic proof of the two contracted diagrams (the
        parts off the regions for a local contraction): a sub-proof ``C<i>.main`` over their
        networks and the axiom ``net<i>`` from its statement to the step; None when neither
        diagram pair has a semantic proof."""
        x, y = cast("tuple[Diagram, Diagram]", t.inference.data.get("parts", (t.lhs, t.rhs)))
        if x.bang_boxes or y.bang_boxes or free_mult_symbols(x) or free_mult_symbols(y):
            return self.inducted(t, x, y, binders)
        return self.networks(x, y, t.lhs, t.rhs, binders, "parts" in t.inference.data)

    def base(self, t: Theorem) -> str:
        """An induction's base case through a semantic proof of its two sides, free of
        multiplicity symbols (:meth:`networks`), else its proof term."""
        x, y = t.lhs, t.rhs
        free = free_mult_symbols(x) or free_mult_symbols(y)
        if self.semantic and t.inference.kind != "refl" and not free:
            binders = [_var(v) for v in t.variables]
            proved = self.networks(x, y, x, y, binders, False)
            if proved is not None:
                return proved
        return self.term(t)

    def networks(
        self,
        x: Diagram,
        y: Diagram,
        a: Diagram,
        b: Diagram,
        binders: Sequence[str],
        local: bool,
    ) -> str | None:
        """``a = b`` through a sub-proof ``C<i>.main`` over the networks of ``x`` and ``y``
        (the parts off the regions of ``a`` and ``b`` when ``local``) and the axiom ``net<i>``
        from its statement; None when the pair has no semantic proof."""
        tag = f"C{len(self.subs) + 1}"
        source = _semantic_pair(x, y)
        if source is None:
            return None
        body, imports, ns = _nest(source, tag)
        self.subs.append(body)
        self.imports += [i for i in imports if i not in self.imports]
        lhs, rhs = self.call(a), self.call(b)
        head = f"∀ ({' '.join(binders)} : Nat), " if binders else ""
        note = (
            "the diagrams are the parts off the two regions, plugged into one context, whose "
            f"networks {ns}.main proves equal"
            if local
            else f"the diagrams' networks are those {ns}.main proves equal"
        )
        name = self.fresh("net")
        self.decls.append(
            f"/-- {_comment(note)} -/\naxiom {name} : (type_of% @{ns}.main) → {head}{lhs} = {rhs}"
        )
        applied = f"{name} @{ns}.main"
        return f"({applied} {' '.join(binders)})" if binders else f"({applied})"

    def fresh(self, stem: str) -> str:
        self.count += 1
        return f"{stem}{self.count}"

    def const(self, d: Diagram) -> tuple[str, tuple[str, ...]]:
        params = tuple(sorted(free_mult_symbols(d)))
        for other, name, ps in self.diagrams:
            if ps == params and _same(other, d):
                return name, ps
        name = f"d{len(self.diagrams)}"
        self.diagrams.append((d, name, params))
        kind = " → ".join(["Nat"] * len(params) + ["Diagram"])
        self.decls.append(f"/-- {_comment(self.describe(d))} -/\naxiom {name} : {kind}")
        return name, params

    def call(self, d: Diagram, subst: dict[str, str] | None = None) -> str:
        name, params = self.const(d)
        if not params:
            return name
        args = [(subst or {}).get(p, _var(p)) for p in params]
        return f"({name} {' '.join(args)})"

    def axiom(self, stem: str, note: str, binders: Sequence[str], lhs: str, rhs: str) -> str:
        if lhs == rhs:
            return "rfl"
        head = f"∀ ({' '.join(binders)} : Nat), " if binders else ""
        statement = f"{head}{lhs} = {rhs}"
        name = self.stated.get(statement)
        if name is None:
            name = self.fresh(stem)
            self.stated[statement] = name
            self.decls.append(f"/-- {_comment(note)} -/\naxiom {name} : {statement}")
        return name if not binders else f"({name} {' '.join(binders)})"

    def term(self, t: Theorem) -> str:
        inf = t.inference
        binders = [_var(v) for v in t.variables]
        if inf.kind == "refl":
            return "rfl"
        if inf.kind == "rule":
            note = f"rule {inf.rule}" + (f" at {inf.detail}" if inf.detail else "")
            stem = "ax"
            if inf.rule is not None and inf.rule.startswith("discard_"):
                stem = f"unproved_DISCARD_{inf.rule.removeprefix('discard_')}_"
            return self.axiom(stem, note, binders, self.call(t.lhs), self.call(t.rhs))
        if inf.kind == "iso":
            note = "isomorphic comparison views"
            return self.axiom("iso", note, binders, self.call(t.lhs), self.call(t.rhs))
        if inf.kind == "contraction" and self.semantic:
            proved = self.contraction(t, binders)
            if proved is not None:
                return proved
        if inf.kind in ("algebraic", "contraction"):
            note = (
                "equal values in a checked number field"
                if inf.kind == "algebraic"
                else "symbolic contractions agree, every symbol formal"
            )
            return self.axiom(inf.kind, note, binders, self.call(t.lhs), self.call(t.rhs))
        if inf.kind == "symm":
            inner = self.term(inf.premises[0])
            return inner if inner == "rfl" else f"({inner}).symm"
        if inf.kind == "trans":
            a, b = (self.term(p) for p in inf.premises)
            return b if a == "rfl" else a if b == "rfl" else f"({a}).trans ({b})"
        if inf.kind == "subst":
            return self.subst(t)
        if inf.kind == "congr":
            return self.congr(t)
        if inf.kind == "shift":
            raise KernelError("a shift has no opaque export")
        if inf.kind != "induction":
            raise KernelError(f"an {inf.kind} inference has no opaque export")
        return self.induction(t)

    def subst(self, t: Theorem) -> str:
        """The premise stated as a lemma over its multiplicities, applied at the clause's
        value, between instantiation axioms onto the substituted sides."""
        from archytaszx.semantics.restrict import DIMENSION, EQ

        (prem,) = t.inference.premises
        clause = cast("Clause", t.inference.data["clause"])
        pb = [_var(v) for v in prem.variables]
        body = self.term(prem)
        stmt = f"{self.call(prem.lhs)} = {self.call(prem.rhs)}"
        lemma = self.fresh("fam")
        head = f"∀ ({' '.join(pb)} : Nat), {stmt}" if pb else stmt
        proof = f"fun {' '.join(pb)} => {body}" if pb else body
        self.decls.append(f"theorem {lemma} : {head} :=\n  {proof}")
        at: dict[str, str] = {}
        if clause.kind != DIMENSION:
            value = (
                str(clause.c)
                if clause.form == EQ
                else (f"({clause.a} * {_var(clause.fresh)} + {clause.c})")
            )
            at[clause.symbol] = value
        args = [at.get(v, _var(v)) for v in prem.variables]
        applied = f"({lemma} {' '.join(args)})" if args else lemma
        binders = [_var(v) for v in t.variables]
        note = f"instantiation at {clause.text()}"
        inst_l = self.axiom("inst", note, binders, self.call(prem.lhs, at), self.call(t.lhs))
        inst_r = self.axiom("inst", note, binders, self.call(prem.rhs, at), self.call(t.rhs))
        return f"({inst_l}).symm.trans (({applied}).trans ({inst_r}))"

    def congr(self, t: Theorem) -> str:
        """The premise under a context axiom, between plug axioms onto both sides."""
        (prem,) = t.inference.premises
        binders = [_var(v) for v in t.variables]
        hole = self.fresh("hole")
        arity = " → ".join(["Nat"] * len(binders) + ["Diagram", "Diagram"])
        self.decls.append(f"/-- a context around a region -/\naxiom {hole} : {arity}")
        ctx = f"({hole} {' '.join(binders)})" if binders else hole
        plug_l = self.axiom(
            "plug", "the region", binders, self.call(t.lhs), f"({ctx} {self.call(prem.lhs)})"
        )
        plug_r = self.axiom(
            "plug",
            "the region replaced",
            binders,
            self.call(t.rhs),
            f"({ctx} {self.call(prem.rhs)})",
        )
        return f"({plug_l}).trans ((congrArg {ctx} {self.term(prem)}).trans ({plug_r}).symm)"

    def induction(self, t: Theorem, name: str | None = None) -> str:
        inf, data = t.inference, t.inference.data
        base, expose, step = inf.premises
        index, k = str(data["index"]), str(data["step"])
        side = data["side"]
        own, other = (t.lhs, t.rhs) if side is InductionSide.START else (t.rhs, t.lhs)
        others = [_var(v) for v in t.variables if v != index]
        n, kv = _var(index), _var(k)
        z, s = {index: "0"}, {index: f"({kv} + 1)"}
        at_k = {index: kv}
        inst_s = self.axiom(
            "inst", f"{index} = 0", others, self.call(t.lhs, z), self.call(base.lhs)
        )
        inst_g = self.axiom(
            "inst", f"{index} = 0", others, self.call(t.rhs, z), self.call(base.rhs)
        )
        ok = [*others, kv]
        succ_own = self.axiom(
            "peel",
            f"{index} = {k} + 1",
            ok,
            self.call(own, s),
            self.call(data["peeled"]),  # type: ignore[arg-type]
        )
        succ_other = self.axiom(
            "peel",
            f"{index} = {k} + 1",
            ok,
            self.call(other, s),
            self.call(data["other_peeled"]),  # type: ignore[arg-type]
        )
        hole = self.fresh("hole")
        arity = " → ".join(["Nat"] * len(ok) + ["Diagram", "Diagram"])
        self.decls.append(f"/-- the exposed successor around a hole -/\naxiom {hole} : {arity}")
        ctx = f"({hole} {' '.join(ok)})"
        plug_own = self.axiom(
            "plug",
            "the hypothesis region",
            ok,
            self.call(expose.rhs),
            f"({ctx} {self.call(own, at_k)})",
        )
        plug_other = self.axiom(
            "plug",
            "the region replaced",
            ok,
            self.call(data["replaced"]),  # type: ignore[arg-type]
            f"({ctx} {self.call(other, at_k)})",
        )
        hyp = "ih" if side is InductionSide.START else "ih.symm"
        zero = f"({inst_s}).trans (({self.base(base)}).trans ({inst_g}).symm)"
        succ = (
            f"({succ_own}).trans (({self.term(expose)}).trans (({plug_own}).trans "
            f"((congrArg {ctx} {hyp}).trans (({plug_other}).symm.trans "
            f"(({self.term(step)}).trans ({succ_other}).symm)))))"
        )
        if side is not InductionSide.START:
            succ = f"({succ}).symm"
        name = name or self.fresh("ind")
        binders = f"∀ ({' '.join([*others, n])} : Nat), "
        intro = " ".join([*others, n])
        self.decls.append(
            f"theorem {name} : {binders}{self.call(t.lhs)} = {self.call(t.rhs)} := by\n"
            f"  intro {intro}\n  induction {n} with\n  | zero => exact {zero}\n"
            f"  | succ {kv} ih => exact {succ}"
        )
        return f"({name} {' '.join([*others, n])})"


def _summary(d: Diagram) -> str:
    return f"a diagram of {len(d.nodes)} nodes and {len(d.wires)} wires"


def lean(
    theorem: Theorem | Refutation | Symbolic,
    name: str = "main",
    describe: Callable[[Diagram], str] | None = None,
    *,
    semantic: bool = True,
) -> str:
    """Lean 4 source stating and proving ``theorem`` (see the module docstring); ``describe``
    gives each diagram's doc comment, ``semantic=False`` forces the opaque export."""
    if isinstance(theorem, Symbolic):
        if name == "main":
            return theorem.source
        try:
            return lean_sym.export_symbolic(theorem.lhs, theorem.rhs, name, describe)
        except lean_net.NetUnsupported as exc:
            raise KernelError(f"no symbolic proof: {exc}") from exc
    if isinstance(theorem, Refutation):
        if theorem.source and name == "main":
            return theorem.source
        try:
            return lean_refute.export_refutation(
                theorem.lhs, theorem.rhs, theorem.point, name, describe
            )
        except lean_net.NetUnsupported as exc:
            raise KernelError(f"no Lean refutation: {exc}") from exc
    if not isinstance(theorem, Theorem):
        raise KernelError(f"lean takes a Theorem, got {type(theorem).__name__}")
    if theorem.domain:
        return _lean_restricted(theorem, name, describe, semantic)
    if semantic:
        source: str | None = None
        try:
            source = lean_net.export(theorem, name, describe)
        except lean_net.NetUnsupported:
            pass
        if source is not None and "\naxiom unproved_" not in source:
            return source
        try:
            return lean_sym.export_symbolic(theorem.lhs, theorem.rhs, name, describe)
        except lean_net.NetUnsupported:
            pass
        if source is not None:
            return source
    out = _Lean(describe or _summary, semantic)
    if theorem.inference.kind == "induction":
        out.induction(theorem, name)
    else:
        binders = [_var(v) for v in theorem.variables]
        body = out.term(theorem)
        stmt = f"{out.call(theorem.lhs)} = {out.call(theorem.rhs)}"
        if binders:
            head = f"∀ ({' '.join(binders)} : Nat), {stmt}"
            body = f"fun {' '.join(binders)} => {body}"
        else:
            head = stmt
        out.decls.append(f"theorem {name} : {head} :=\n  {body}")
    notes = [f"  assumes {a}" for a in theorem.assumptions]
    if out.subs:
        header = [
            "/-",
            "  ArchytasZX proof export. Each axiom states one rewrite-rule instance, isomorphism,",
            "  algebraic meet, instantiation, peeling or context plug the ArchytasZX kernel",
            "  checked; each symbolic contraction is a theorem C<i>.main over the networks of the",
            "  two contracted diagrams, carried to the step by the axiom net<i> (the diagrams'",
            "  translation into those networks); Lean checks how they compose.",
            *notes,
            "-/",
            "",
            *out.imports,
            "",
            "\n\n".join(out.subs),
            "",
            f"namespace {OPAQUE_NS}",
            "",
            "axiom Diagram : Type",
        ]
        tail = f"end {OPAQUE_NS}"
    else:
        header = [
            "/-",
            "  ArchytasZX proof export. Each axiom states one rewrite-rule instance, isomorphism,",
            "  algebraic meet, symbolic contraction, instantiation, peeling or context plug the",
            "  ArchytasZX kernel checked; Lean checks how they compose.",
            *notes,
            "-/",
            "",
            "namespace ArchytasZX",
            "",
            "axiom Diagram : Type",
        ]
        tail = "end ArchytasZX"
    return "\n".join([*header, "", "\n\n".join(out.decls), "", tail, ""])


INDUCTED_DEPTH = 2
"""The deepest nesting of contraction steps re-proved by induction in one opaque export."""

INDUCTED_SAMPLES = 4
"""How many default samples :func:`_contraction_induction` tries as the induction witness."""


def _contraction_induction(
    x: Diagram, y: Diagram
) -> tuple[Theorem, tuple[str, str, int] | None] | None:
    """``x = y`` by :func:`_induction_rung` and no shift; when every box multiplicity reading
    ``n`` is ``n + c`` (one integer ``c >= 1``), that rung's theorem over both sides at
    ``n = m - c`` (``m`` fresh) and the shift ``(n, m, c)``; None when neither gives a
    theorem."""
    from archytaszx.semantics.restrict import MULTIPLICITY, SHIFT, _all_symbols, apply_clause

    shifted = _offset_index(x, y)
    if shifted is None:
        proof = _induction_rung(x, y)
        return None if proof is None else (proof, None)
    index, c = shifted
    taken = _all_symbols(x, y)
    fresh = next(f"m{i}" for i in range(len(taken) + 1) if f"m{i}" not in taken)
    back = Clause(index, MULTIPLICITY, SHIFT, a=1, c=-c, fresh=fresh)
    try:
        xs, ys = apply_clause(x, back), apply_clause(y, back)
    except Exception:  # noqa: BLE001
        return None
    if free_mult_symbols(xs) | free_mult_symbols(ys) != {fresh}:
        return None
    proof = _induction_rung(xs, ys)
    return None if proof is None else (proof, (index, fresh, c))


def _offset_index(x: Diagram, y: Diagram) -> tuple[str, int] | None:
    """``(n, c)`` when ``n`` is the one free multiplicity of ``x`` and ``y`` and every box
    multiplicity reading it is ``n + c`` for one integer ``c >= 1``; else None."""
    names = free_mult_symbols(x) | free_mult_symbols(y)
    if len(names) != 1:
        return None
    (index,) = names
    offsets = set()
    for d in (x, y):
        for box in d.bang_boxes.values():
            expr = box.multiplicity.to_sympy()
            if index not in {str(s) for s in expr.free_symbols}:
                continue
            const, rest = sp.expand(expr).as_coeff_Add()
            if str(rest) != index or not const.is_Integer:
                return None
            offsets.add(int(const))
    if len(offsets) != 1:
        return None
    (c,) = offsets
    return (index, c) if c >= 1 else None


INDUCTED_KEPT = 16
"""How many induction theorems :data:`_INDUCTED` keeps, latest last."""

_INDUCTED: list[tuple[Diagram, Diagram, Theorem | None]] = []
"""Induction theorems by their sides: those of decisions read by :func:`from_decision` and
those :func:`_induction_rung` built or failed to build."""


def _remember_induction(
    proof: Theorem | None, x: Diagram | None = None, y: Diagram | None = None
) -> None:
    """``proof`` kept under its sides (or ``x``, ``y``)."""
    a = x if x is not None else proof.lhs if proof is not None else None
    b = y if y is not None else proof.rhs if proof is not None else None
    if a is None or b is None:
        return
    _INDUCTED.append((a, b, proof))
    del _INDUCTED[:-INDUCTED_KEPT]


def _recalled_induction(x: Diagram, y: Diagram) -> tuple[bool, Theorem | None]:
    """Whether an induction theorem of ``x = y`` (or its failure) is kept, and that theorem."""
    for a, b, proof in reversed(_INDUCTED):
        if _same(a, x) and _same(b, y):
            return True, proof
    return False, None


def _induction_rung(x: Diagram, y: Diagram) -> Theorem | None:
    """``x = y`` by the kept induction theorem of that pair (:data:`_INDUCTED`), else by the
    induction rung of :func:`~archytaszx.semantics.decide.decide_equal` (base case decided,
    step case by its step ladder, the kernel theorem built from both, the other symbols held at
    the first of :data:`INDUCTED_SAMPLES` default samples the oracle evaluates); None when the
    rung does not settle EQUAL with a kernel theorem of ``x = y``."""
    kept, proof = _recalled_induction(x, y)
    if kept:
        return proof
    proof = _induction_rung_run(x, y)
    _remember_induction(proof, x, y)
    return proof


def _induction_rung_run(x: Diagram, y: Diagram) -> Theorem | None:
    """:func:`_induction_rung` without the kept theorems."""
    from archytaszx.rewrite.engine import DEFAULT_GUARD
    from archytaszx.semantics import decide as dec
    from archytaszx.semantics.check import DEFAULT_TOLERANCE
    from archytaszx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS

    try:
        settled = dec._decide_by_induction(
            x,
            y,
            evaluated=dec.sample_grid(x, y)[:INDUCTED_SAMPLES],
            samples=None,
            rules=None,
            guard=DEFAULT_GUARD,
            max_samples=24,
            use_symbolic=True,
            max_depth=2,
            tolerance=DEFAULT_TOLERANCE,
            max_elements=DEFAULT_MAX_ELEMENTS,
            use_saturation=True,
            saturation_limits=dec.DECIDE_SATURATION_LIMITS,
        )
    except Exception:  # noqa: BLE001
        return None
    if isinstance(settled, str) or settled[0] is not dec.EqualityVerdict.EQUAL:
        return None
    proof = settled[3].get("induction_proof")
    if not isinstance(proof, Theorem) or not (_same(proof.lhs, x) and _same(proof.rhs, y)):
        return None
    return proof


OPAQUE_NS = "ArchytasZX.Opaque"
"""The namespace of the opaque diagrams in an export that imports the Lean library."""


def _semantic_pair(x: Diagram, y: Diagram) -> str | None:
    """Lean source whose theorem ``main`` proves ``x`` and ``y`` equal over their networks: the
    normaliser of symbolic sums, else the qubit or odd-prime path-sum network proof; None when
    none applies."""
    from archytaszx.algebra.pathsum import PathSumError
    from archytaszx.semantics.lean_pathsum import export_net, export_qudit_net

    builders: tuple[Callable[[], str], ...] = (
        lambda: lean_sym.export_symbolic(x, y, "main"),
        lambda: export_net(x, y),
        lambda: export_qudit_net(x, y),
    )
    for build in builders:
        try:
            return build()
        except (lean_net.NetUnsupported, ScalarUnsupported, PathSumError):
            continue
    return None


def _nest(source: str, tag: str) -> tuple[str, list[str], str]:
    """``source`` without its imports, its namespace ``N`` renamed ``N.tag`` and its tactic
    macros local; the imports and the new namespace."""
    lines = source.splitlines()
    imports = [ln for ln in lines if ln.startswith("import ")]
    ns = next(ln.split()[1] for ln in lines if ln.startswith("namespace "))
    body: list[str] = []
    for ln in lines:
        if ln.startswith("import "):
            continue
        if ln == f"namespace {ns}":
            ln = f"namespace {ns}.{tag}"
        elif ln == f"end {ns}":
            ln = f"end {ns}.{tag}"
        elif ln.startswith('macro "'):
            ln = "local " + ln
        body.append(ln)
    return "\n".join(body).strip(), imports, f"{ns}.{tag}"


def _lean_restricted(
    theorem: Theorem, name: str, describe: Callable[[Diagram], str] | None, semantic: bool
) -> str:
    """A restricted theorem's export: the family theorem with its corollary at the clauses, or
    the theorem about the restricted pair with the restriction in its header."""
    from archytaszx.semantics.lean_restrict import RestrictUnsupported, corollary, header

    base = theorem
    while base.inference.kind == "subst":
        base = base.inference.premises[0]
    plain = Theorem(_TOKEN, base.lhs, base.rhs, base.inference, base.assumptions)
    source = lean(plain, name, describe, semantic=semantic)
    origin = theorem.origin
    text = ", ".join(c.text() for c in theorem.domain)
    if origin is not None and base.lhs is origin[0] and base.rhs is origin[1]:
        source = header(source, f"restricted to {text}: the family theorem, then {_RESTRICTED}")
        try:
            return corollary(source, theorem.domain, name, _var)
        except RestrictUnsupported as exc:
            raise KernelError(f"no corollary: {exc}") from exc
    restriction = Restriction(theorem.domain)
    shifted = _lean_shift(plain, theorem, name, describe) if semantic else None
    if shifted is not None:
        return header(shifted, _pair_note(restriction) + f"; {_lifted_note(restriction)}")
    return header(source, _pair_note(restriction) + _unlinked(restriction))


def _unlinked(restriction: Restriction) -> str:
    """``; UNLINKED`` when a clause substitutes, else empty."""
    return f"; {UNLINKED}" if any(c.substitutes for c in restriction.clauses) else ""


UNLINKED = (
    "UNCERTIFIED for the original diagrams: the theorem is about the restricted diagrams, "
    "their link to the original diagrams not proved"
)
"""The header note of a restricted-pair export with no statement over the original
diagrams."""

_SHIFTED = "stated over the original diagrams by peeling (Equal.of_split_at)"


def _lifted_note(restriction: Restriction) -> str:
    """How a restricted-pair theorem is carried to the original diagrams."""
    from archytaszx.semantics.restrict import DIMENSION, EQ

    subs = [c for c in restriction.clauses if c.substitutes]
    dims = [c.symbol for c in subs if c.kind == DIMENSION and c.form == EQ]
    parts = [_SHIFTED] if len(dims) < len(subs) else []
    if dims:
        parts.append(
            f"restated over the original diagrams with {', '.join(dims)} free "
            "(main_diagrams_original)"
        )
    return "; ".join(parts)


def _lean_shift(
    plain: Theorem, theorem: Theorem, name: str, describe: Callable[[Diagram], str] | None
) -> str | None:
    """The restricted-pair theorem carried to the original diagrams: through :func:`shift` at
    one clause ``n >= c`` or ``n = a*m + c``, through :func:`lift` at several multiplicity
    clauses or clauses ``n = c``, with the corollary over each ``n``, and restated over the
    original diagrams at its clauses ``d = v``; None when the one clause is ``n = m`` or the
    export declines."""
    from archytaszx.semantics.lean_restrict import RestrictUnsupported, shift_corollary
    from archytaszx.semantics.restrict import DIMENSION, EQ, MULTIPLICITY, apply_clause

    subs = [c for c in theorem.domain if c.substitutes]
    if theorem.origin is None or not subs:
        return None
    dims = [c for c in subs if c.kind == DIMENSION and c.form == EQ]
    mults = [c for c in subs if c not in dims]
    if any(c.kind != MULTIPLICITY for c in mults):
        return None
    single = len(mults) == 1 and mults[0].form != EQ
    if single and mults[0].a == 1 and mults[0].c == 0:
        return None
    left0, right0 = theorem.origin
    try:
        mid_l, mid_r = left0, right0
        for c in dims:
            mid_l, mid_r = apply_clause(mid_l, c), apply_clause(mid_r, c)
        if single:
            body = shift(plain, mid_l, mid_r, mults[0])
        elif mults:
            body = lift(plain, mid_l, mid_r, mults)
        else:
            body = _onto(plain, mid_l, mid_r)
        original = (
            ({c.symbol: c.c for c in dims}, [(body.lhs, left0), (body.rhs, right0)])
            if dims
            else None
        )
        source = lean_net.export(body, name, describe, original=original)
        steps = [c for c in mults if c.form != EQ]
        return shift_corollary(source, steps, name) if steps else source
    except (KernelError, lean_net.NetUnsupported, ScalarUnsupported, RestrictUnsupported):
        return None


def _onto(t: Theorem, left: Diagram, right: Diagram) -> Theorem:
    """``t`` with its sides replaced by ``left`` and ``right`` through isomorphisms."""
    body = t
    if not _same(t.lhs, left):
        body = trans(iso(left, t.lhs), body)
    if not _same(t.rhs, right):
        body = trans(body, iso(t.rhs, right))
    return body
