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
quantified over the naturals. Only the rules below construct one:

* :func:`refl` -- ``d = d``;
* :func:`rule` -- ``d = d'`` for one recorded rewrite step, re-applied by the engine at a
  match the matcher rediscovers in ``d``;
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

from archytaszx.diagram.bangbox import free_mult_symbols
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.graph import Diagram, NodeId
from archytaszx.rewrite.engine import RewriteStep
from archytaszx.rewrite.normal_form import comparison_view, views_isomorphic
from archytaszx.rewrite.rule import ConstraintOutcome
from archytaszx.semantics import lean_net, lean_refute, lean_sym
from archytaszx.semantics.certificate import (
    Certificate,
    DerivationKind,
    _replay_one,
    compare_structure,
)
from archytaszx.semantics.induction import _bare, _extract, replace_region
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


class KernelError(Exception):
    """A proof piece the kernel does not accept."""


_TOKEN = object()


@dataclass(frozen=True, eq=False)
class Inference:
    """The rule that built a theorem: ``kind`` is refl, rule, iso, symm, trans or induction."""

    kind: str
    premises: tuple[Theorem, ...] = ()
    rule: str = ""
    detail: str = ""
    data: dict[str, object] = field(default_factory=dict)


class Theorem:
    """``lhs = rhs`` for every value of the free multiplicities, under ``assumptions``."""

    __slots__ = ("assumptions", "inference", "lhs", "rhs")

    def __init__(
        self,
        token: object,
        lhs: Diagram,
        rhs: Diagram,
        inference: Inference,
        assumptions: tuple[str, ...] = (),
    ) -> None:
        if token is not _TOKEN:
            raise KernelError("a Theorem is built only by the kernel's inference rules")
        self.lhs, self.rhs = lhs, rhs
        self.inference = inference
        self.assumptions = assumptions

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


def _merge(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for group in groups:
        out += [a for a in group if a not in out]
    return tuple(out)


def _same(a: Diagram, b: Diagram) -> bool:
    return compare_structure(a, b).identical


def refl(d: Diagram) -> Theorem:
    """``d = d``."""
    return Theorem(_TOKEN, d, d, Inference("refl"))


def rule(d: Diagram, step: RewriteStep) -> Theorem:
    """``d = d'`` by re-applying ``step`` to ``d`` at its rediscovered match."""
    failure, after = _replay_one(d.copy(), 0, step, True)
    if failure is not None:
        raise KernelError(failure)
    nodes = ", ".join(f"n{int(n)}" for n in sorted(step.consumed_node_ids, key=int))
    deferred = tuple(
        str(c) for c in step.dimension_constraints if c.outcome is ConstraintOutcome.DEFERRED
    )
    inference = Inference("rule", rule=step.rule_name, detail=nodes, data={"step": step})
    return Theorem(_TOKEN, d, after, inference, deferred)


def iso(a: Diagram, b: Diagram) -> Theorem:
    """``a = b`` for isomorphic comparison views."""
    if not (_same(a, b) or views_isomorphic(comparison_view(a), comparison_view(b))):
        raise KernelError("the diagrams have different comparison views")
    return Theorem(_TOKEN, a, b, Inference("iso"))


def symm(t: Theorem) -> Theorem:
    """``rhs = lhs`` from ``lhs = rhs``."""
    return Theorem(_TOKEN, t.rhs, t.lhs, Inference("symm", (t,)), t.assumptions)


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


def check(artefact: object) -> Theorem:
    """The theorem a certificate, proof certificate or induction proof establishes."""
    if isinstance(artefact, Theorem):
        return artefact
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
        return trans(forward, trans(iso(forward.rhs, backward.rhs), symm(backward)))
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
    proof by the normaliser of symbolic sums."""

    lhs: Diagram
    rhs: Diagram
    source: str


def symbolic(
    left: Diagram, right: Diagram, describe: Callable[[Diagram], str] | None = None
) -> Symbolic:
    """The normaliser's proof of ``left = right``."""
    try:
        return Symbolic(left, right, lean_sym.export_symbolic(left, right, describe=describe))
    except lean_net.NetUnsupported as exc:
        raise KernelError(f"no symbolic proof: {exc}") from exc


def refutation(decision: object, left: Diagram, right: Diagram) -> Refutation:
    """The refutation an UNEQUAL decision's counterexample gives."""
    from archytaszx.semantics.decide import Decision, EqualityVerdict

    if not isinstance(decision, Decision):
        raise KernelError(f"cannot read a {type(decision).__name__}")
    if decision.verdict is not EqualityVerdict.UNEQUAL or decision.counterexample is None:
        raise KernelError("the decision carries no counterexample")
    return Refutation(left, right, dict(decision.counterexample))


def from_decision(decision: object, left: Diagram, right: Diagram) -> Theorem:
    """``left = right`` from a decision's normal-form or saturation certificates."""
    from archytaszx.semantics.decide import Decision, DecisionMethod

    if not isinstance(decision, Decision):
        raise KernelError(f"cannot read a {type(decision).__name__}")
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
            return self.axiom("ax", note, binders, self.call(t.lhs), self.call(t.rhs))
        if inf.kind == "iso":
            note = "isomorphic comparison views"
            return self.axiom("iso", note, binders, self.call(t.lhs), self.call(t.rhs))
        if inf.kind == "symm":
            inner = self.term(inf.premises[0])
            return inner if inner == "rfl" else f"({inner}).symm"
        if inf.kind == "trans":
            a, b = (self.term(p) for p in inf.premises)
            return b if a == "rfl" else a if b == "rfl" else f"({a}).trans ({b})"
        return self.induction(t)

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
