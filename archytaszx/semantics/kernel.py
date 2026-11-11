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
* :func:`symm`, :func:`trans` -- ``trans`` needs the middle diagrams identical id for id;
* :func:`induction` -- ``start = goal`` from a base theorem at index 0, a theorem exposing
  one side's peeled successor at ``k + 1`` as a context around that side at ``k``, and a
  theorem from that context around the other side at ``k`` to the other peeled successor.

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
instance, isomorphism, instantiation at 0, peeling and context plug the kernel checked, and
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
from archytaszx.diagram.graph import Diagram, NodeId
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
from archytaszx.semantics.induction import _bare, _extract, replace_region
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
    """The rule that built a theorem: ``kind`` is refl, rule, iso, algebraic, symm, trans,
    induction, subst or shift."""

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
    from archytaszx.semantics.restrict import DIMENSION, EQ, GE

    fresh = {c.fresh for c in domain if c.fresh}
    free = [v for v in variables if v not in fresh]
    parts: list[str] = []
    for c in domain:
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
        if c.kind != DIMENSION:
            continue
        if c.form == EQ:
            parts.append(f"at {c.symbol} = {c.c}")
        elif f"{c.c} <= {c.symbol}" in assumptions:
            parts.append(f"for {c.symbol} >= {c.c}")
        else:
            parts.append(f"for every {c.symbol}")
    return ", ".join(parts)


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
) -> Theorem:
    """``start = goal`` for every value of ``index`` (see the module docstring)."""
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
        peeled = induction_successor(own, index, k)
        other_peeled = induction_successor(other, index, k)
        if peeled is None or other_peeled is None:
            raise KernelError("a successor has no single box to peel")
        if not _same(expose.lhs, peeled):
            raise KernelError("the exposing theorem does not start at the peeled successor")
        exposed = expose.rhs
        hypothesis = induction_hypothesis(own, index, k)
        if not region <= frozenset(exposed.nodes) or _crosses(exposed, region):
            raise KernelError("the hypothesis region is not a closed part of the exposed diagram")
        if not isomorphic(_bare(_extract(exposed, region, with_boxes=True)), _bare(hypothesis)):
            raise KernelError("the hypothesis region is not the hypothesis diagram")
        replaced = replace_region(
            exposed, region, hypothesis, induction_hypothesis(other, index, k)
        )
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
        "other_k": induction_hypothesis(other, index, k),
        "replaced": replaced,
        "region": region,
        "exposed": exposed,
    }
    assumptions = _merge(base.assumptions, expose.assumptions, step.assumptions)
    return Theorem(
        _TOKEN, start, goal, Inference("induction", (base, expose, step), data=data), assumptions
    )


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
    floors = tuple(
        f"{c.c} <= {c.symbol}"
        for c in restriction.clauses
        if not c.substitutes and c.symbol in used
    )
    return Theorem(
        _TOKEN,
        t.lhs,
        t.rhs,
        t.inference,
        _merge(t.assumptions, floors),
        restriction.clauses,
        (left, right),
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
    floors = tuple(
        f"{c.c} <= {c.symbol}"
        for c in proof.restriction.clauses
        if not c.substitutes and c.symbol in proof.floors_used
    )
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
    """``left = right`` from a decision's normal-form or saturation certificates or its
    induction theorem."""
    from archytaszx.semantics.decide import Decision, DecisionMethod

    if not isinstance(decision, Decision):
        raise KernelError(f"cannot read a {type(decision).__name__}")
    if decision.method is DecisionMethod.INDUCTION and decision.induction_proof is not None:
        proved = check(decision.induction_proof)
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

    def __init__(self, describe: Callable[[Diagram], str]) -> None:
        self.describe = describe
        self.decls: list[str] = []
        self.diagrams: list[tuple[Diagram, str, tuple[str, ...]]] = []
        self.stated: dict[str, str] = {}
        self.count = 0

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
        if inf.kind == "symm":
            inner = self.term(inf.premises[0])
            return inner if inner == "rfl" else f"({inner}).symm"
        if inf.kind == "trans":
            a, b = (self.term(p) for p in inf.premises)
            return b if a == "rfl" else a if b == "rfl" else f"({a}).trans ({b})"
        if inf.kind == "subst":
            return self.subst(t)
        if inf.kind == "shift":
            raise KernelError("a shift has no opaque export")
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
        zero = f"({inst_s}).trans (({self.term(base)}).trans ({inst_g}).symm)"
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
    out = _Lean(describe or _summary)
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
    header = [
        "/-",
        "  ArchytasZX proof export. Each axiom states one rewrite-rule instance, isomorphism,",
        "  instantiation, peeling or context plug the ArchytasZX kernel checked; Lean checks",
        "  how they compose.",
        *notes,
        "-/",
        "",
        "namespace ArchytasZX",
        "",
        "axiom Diagram : Type",
    ]
    return "\n".join([*header, "", "\n\n".join(out.decls), "", "end ArchytasZX", ""])


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
