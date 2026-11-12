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

"""Restrictions of a claim to instances and restricted families of its free symbols.

A :class:`Restriction` is a set of :class:`Clause` values, at most one per symbol:

* ``n = v``: the multiplicity ``n`` at the value ``v``;
* ``n >= c``: ``n`` replaced by ``m + c``, ``m`` fresh;
* ``n = a*m + c``: ``n`` replaced by ``a*m + c``, ``m`` fresh;
* ``d = v``: the dimension ``d`` at the value ``v``;
* ``d >= c``: a floor on ``d``, the diagrams unchanged.

:func:`apply_clause` substitutes one clause into a diagram: a concrete multiplicity expands
its boxes, an affine one is substituted into every box carrying the symbol and each box whose
multiplicity keeps a constant term ``c >= 1`` is peeled ``c`` times. :func:`subst_pair`
applies every clause, in order, to both sides.

Phase symbols (free in a phase vector only) and exponent symbols (free in a scalar, in no
phase, multiplicity or dimension) take rational clauses too, and every kind takes ``x != v``,
``x > c``, ``x <= c``, ``x < c``, chained ``a <= x <= b`` (strict forms too), ``x in Z``,
``x in (1/q)Z`` and ``x in Z/q``; several clauses on one symbol of these forms are intersected
into an interval, a lattice and excluded points. The intersection is planned as an instance
(one value), finitely many instances (a bounded integer or lattice domain, or a lattice phase
symbol whose phases are 1-periodic in it), or a shift ``x = c + a*s`` with ``s`` fresh and its
range kept as a side condition; excluded points are side conditions only. :func:`in_domain`
checks every clause and side condition.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import (
    Mult,
    expand_concrete_boxes,
    free_mult_symbols,
    instantiate_symbol,
    peel_one,
)
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.decide import DecideGrammarError, _free_symbols

MULTIPLICITY = "multiplicity"
DIMENSION = "dimension"
PHASE = "phase"
EXPONENT = "exponent"
EQ = "eq"
GE = "ge"
AFFINE = "affine"
NE = "ne"
GT = "gt"
LE = "le"
LT = "lt"
RANGE = "range"
LATTICE = "lattice"
SHIFT = "shift"
FINITE = "finite"
SIDE = "side"
_NEW_FORMS = frozenset({NE, GT, LE, LT, RANGE, LATTICE})

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CLAUSE_RE = re.compile(rf"\s*({_IDENT})\s*(!=|≠|>=|≥|<=|≤|=|>|<)\s*(.*?)\s*")
_INT_RE = re.compile(r"[+-]?\d+")
_NUM = r"[+-]?\d+(?:/\d+)?"
_NUM_RE = re.compile(_NUM)
_AFFINE_RE = re.compile(rf"(?:({_NUM})\s*\*?\s*)?({_IDENT})\s*(?:\+\s*({_NUM}))?")
_RANGE_RE = re.compile(rf"\s*({_NUM})\s*(<=|≤|<)\s*({_IDENT})\s*(<=|≤|<)\s*({_NUM})\s*")
_LATTICE_RE = re.compile(
    rf"\s*({_IDENT})\s+in\s+(?:Z|\(\s*1\s*/\s*(\d+)\s*\)\s*Z|Z\s*/\s*(\d+))\s*"
)
_OPS = {"!=": NE, "≠": NE, ">=": GE, "≥": GE, "<=": LE, "≤": LE, ">": GT, "<": LT}


@dataclass(frozen=True, slots=True)
class Domain:
    """A set of values: an interval (``None`` ends unbounded), a lattice ``(1/q)Z`` (q = 0 for
    the reals), an integer flag and excluded points."""

    lo: Fraction | None = None
    lo_strict: bool = False
    hi: Fraction | None = None
    hi_strict: bool = False
    q: int = 0
    excluded: tuple[Fraction, ...] = ()
    step: Fraction | None = None
    offset: Fraction = Fraction(0)

    def contains(self, value: object) -> bool:
        """Whether ``value`` lies in the domain."""
        x = _as_fraction(value)
        if x is None:
            return False
        if self.lo is not None and (x < self.lo or (self.lo_strict and x == self.lo)):
            return False
        if self.hi is not None and (x > self.hi or (self.hi_strict and x == self.hi)):
            return False
        if self.q and (x * self.q).denominator != 1:
            return False
        if self.step is not None and ((x - self.offset) / self.step).denominator != 1:
            return False
        return x not in self.excluded


def _as_fraction(value: object) -> Fraction | None:
    """``value`` as an exact rational, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    try:
        expr = sp.nsimplify(value) if isinstance(value, float) else sp.sympify(value)
    except (TypeError, ValueError, sp.SympifyError):
        return None
    if not getattr(expr, "is_Rational", False):
        return None
    return Fraction(int(expr.p), int(expr.q))


@dataclass(frozen=True, slots=True)
class Clause:
    """One constraint on one symbol: ``form`` is eq (``symbol = c``), ge (``symbol >= c``) or
    affine (``symbol = a*fresh + c``); ``kind`` is multiplicity, dimension, or empty before
    :func:`validate`; a multiplicity ge clause carries its fresh symbol too."""

    symbol: str
    kind: str
    form: str
    a: int = 1
    c: int = 0
    fresh: str = ""
    hi: Fraction | None = None
    lo_strict: bool = False
    hi_strict: bool = False
    q: int = 0
    source: str = ""
    domain: Domain | None = None
    fresh_domain: Domain | None = None
    values: tuple[Fraction, ...] = ()
    raw: str = field(default="", compare=False)
    """``a`` and ``c`` hold rationals (Fraction) on phase and exponent clauses. The extended
    clause data: ``hi``, strictness and ``q`` of a parsed new form, its
    ``source`` text, and after :func:`validate` the symbol's ``domain``, the shift's
    ``fresh_domain`` and a finite clause's ``values``."""

    @property
    def substitutes(self) -> bool:
        """True when the clause changes the diagrams (every clause but a dimension floor and
        a side condition)."""
        if self.form in (SIDE, FINITE):
            return False
        return not (self.kind == DIMENSION and self.form == GE)

    def expression(self) -> str:
        """The value the symbol takes: ``c``, or ``a*m + c`` in its shortest form."""
        if self.form == EQ:
            return str(self.c)
        if self.form == SHIFT:
            scale = "" if self.a == 1 else "-" if self.a == -1 else f"{self.a}*"
            head = f"{scale}{self.fresh}"
            return f"{self.c} + {head}" if self.c else head
        head = self.fresh if self.a == 1 else f"{self.a}*{self.fresh}"
        return f"{head} + {self.c}" if self.c else head

    def text(self) -> str:
        """The clause as written in the grammar; a side condition planned with a base clause
        has none."""
        if self.source:
            return self.source
        if self.form == SIDE:
            return ""
        if self.form == GE:
            return f"{self.symbol} >= {self.c}"
        if self.form == EQ:
            return f"{self.symbol} = {self.c}"
        return f"{self.symbol} = {self.expression()}"

    def contains(self, value: object) -> bool:
        """Whether ``value`` lies in the clause's set of values."""
        if self.domain is not None:
            return self.domain.contains(value)
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                value = int(sp.Integer(value))
            except (TypeError, ValueError, sp.SympifyError):
                return False
        if self.form == EQ:
            return value == self.c
        if self.form == GE and self.kind == DIMENSION:
            return value >= self.c
        return value >= self.c and (value - self.c) % self.a == 0


@dataclass(frozen=True, slots=True)
class Restriction:
    """The clauses of a restricted claim, in order."""

    clauses: tuple[Clause, ...] = ()

    @property
    def by_symbol(self) -> Mapping[str, Clause]:
        """Each constrained symbol's clause, side conditions left out."""
        return MappingProxyType({c.symbol: c for c in self.clauses if c.form != SIDE})

    @property
    def certifiable(self) -> bool:
        """True when every clause is a multiplicity or dimension clause of the forms the
        kernel states (``= v``, ``>= c``, ``= a*m + c``)."""
        return all(c.domain is None and c.form in (EQ, GE, AFFINE) for c in self.clauses)

    @property
    def finite(self) -> tuple[Clause, ...]:
        """The clauses with finitely many values, decided instance by instance."""
        return tuple(c for c in self.clauses if c.form == FINITE)

    def instance_at(self, values: Mapping[str, Fraction]) -> Restriction:
        """The restriction with each finite clause fixed at its value in ``values``."""
        out = []
        for c in self.clauses:
            if c.form == FINITE:
                v = values[c.symbol]
                value: int | Fraction = int(v) if v.denominator == 1 else v
                out.append(Clause(c.symbol, c.kind, EQ, c=_slot(value), domain=c.domain))
            else:
                out.append(c)
        return Restriction(tuple(out))

    @property
    def empty(self) -> bool:
        """True when no clause constrains anything."""
        return not self.clauses

    @property
    def floors(self) -> Mapping[str, int]:
        """The floor of each dimension with a ``d >= c`` clause."""
        return MappingProxyType(
            {c.symbol: c.c for c in self.clauses if c.kind == DIMENSION and c.form == GE}
        )

    @property
    def instance(self) -> bool:
        """True when every clause is an ``= v`` clause."""
        return bool(self.clauses) and all(c.form == EQ for c in self.clauses)

    def text(self) -> str:
        """The clauses as written in the grammar, comma separated."""
        return ", ".join(t for t in (c.text() for c in self.clauses) if t)


def _literal(text: str, clause: str) -> int:
    if not _INT_RE.fullmatch(text):
        raise DecideGrammarError(f"clause {clause!r}: {text!r} is not an integer literal")
    value = int(text)
    if value < 0:
        raise DecideGrammarError(f"clause {clause!r}: negative literal {value}")
    return value


def _slot(value: int | Fraction) -> int:
    """A rational in a Clause's int-typed ``a`` or ``c`` slot (phase and exponent clauses)."""
    return cast(int, value)


def _number(text: str) -> int | Fraction:
    """A signed integer or p/q literal; an int when integral."""
    value = Fraction(text)
    return int(value) if value.denominator == 1 else value


def parse_clause(text: str) -> Clause:
    """One clause of the grammar, its kind left to :func:`validate`."""
    clause = text.strip()
    lattice = _LATTICE_RE.fullmatch(clause)
    if lattice is not None:
        q = int(lattice[2] or lattice[3] or 1)
        if q < 1:
            raise DecideGrammarError(f"clause {clause!r}: the lattice needs q >= 1")
        return Clause(lattice[1], "", LATTICE, q=q, source=clause, raw=clause)
    chained = _RANGE_RE.fullmatch(clause)
    if chained is not None:
        lo, hi = _number(chained[1]), _number(chained[5])
        return Clause(
            chained[3],
            "",
            RANGE,
            c=_slot(lo),
            hi=Fraction(hi),
            lo_strict=chained[2] == "<",
            hi_strict=chained[4] == "<",
            source=clause,
            raw=clause,
        )
    match = _CLAUSE_RE.fullmatch(clause)
    if match is None or not match[3]:
        raise DecideGrammarError(
            f"clause {clause!r}: expected NAME = INT, NAME >= INT or NAME = A*M + C"
        )
    symbol, op, rhs = match[1], match[2], match[3]
    if op != "=":
        form = _OPS[op]
        if not _NUM_RE.fullmatch(rhs):
            if form == GE:
                raise DecideGrammarError(f"clause {clause!r}: {rhs!r} is not an integer literal")
            raise DecideGrammarError(f"clause {clause!r}: {rhs!r} is not a rational literal")
        if form == GE:
            return Clause(symbol, "", GE, c=_slot(_number(rhs)), raw=clause)
        return Clause(symbol, "", form, c=_slot(_number(rhs)), source=clause, raw=clause)
    if _NUM_RE.fullmatch(rhs):
        return Clause(symbol, "", EQ, c=_slot(_number(rhs)), raw=clause)
    affine = _AFFINE_RE.fullmatch(rhs)
    if affine is None:
        raise DecideGrammarError(
            f"clause {clause!r}: {rhs!r} is neither an integer literal nor A*M + C"
        )
    a = _number(affine[1]) if affine[1] is not None else 1
    c = _number(affine[3]) if affine[3] is not None else 0
    if a == 0:
        raise DecideGrammarError(f"clause {clause!r}: a = 0; write {symbol} = {c}")
    return Clause(symbol, "", AFFINE, a=_slot(a), c=_slot(c), fresh=affine[2], raw=clause)


def parse_restriction(text: str | Mapping[str, str] | Restriction | None) -> Restriction:
    """A :class:`Restriction` from comma-separated clauses, a mapping from symbol to ``"= v"``
    or ``">= c"`` text, or a Restriction; kinds left to :func:`validate`."""
    if text is None:
        return Restriction()
    if isinstance(text, Restriction):
        return text
    if isinstance(text, Mapping):
        parts = [f"{name} {str(value).strip()}" for name, value in text.items()]
    elif isinstance(text, str):
        parts = [p for p in text.split(",")] if text.strip() else []
    else:
        raise DecideGrammarError(
            f"a restriction is clause text or a mapping, got {type(text).__name__}"
        )
    return Restriction(tuple(parse_clause(p) for p in parts))


def _all_symbols(*diagrams: Diagram) -> frozenset[str]:
    names: set[str] = set()
    for diagram in diagrams:
        dims, mults, others = _free_symbols(diagram)
        names |= dims | mults | others
    return frozenset(names)


def _fresh(stem: str, avoid: set[str]) -> str:
    name, suffix = stem, 0
    while name in avoid:
        suffix += 1
        name = f"{stem}{suffix}"
    return name


def _phase_scalar_symbols(*diagrams: Diagram) -> tuple[frozenset[str], frozenset[str]]:
    """The free symbols of every phase vector and of every scalar of ``diagrams``."""
    phases: set[str] = set()
    scalars: set[str] = set()
    for diagram in diagrams:
        scalars |= diagram.scalar.free_symbols
        for node in diagram.nodes.values():
            if node.phase is not None:
                phases |= node.phase.free_symbols
    return frozenset(phases), frozenset(scalars)


_OLD_FORMS = (EQ, GE, AFFINE)
_MAX_INSTANCES = 64


def _integral_literals(clause: Clause) -> None:
    """The literal checks of a multiplicity or dimension clause."""
    raw = clause.raw or clause.text()
    for value in (clause.c, clause.a) if clause.form == AFFINE else (clause.c,):
        if isinstance(value, Fraction):
            raise DecideGrammarError(f"clause {raw!r}: {str(value)!r} is not an integer literal")
        if value < 0:
            raise DecideGrammarError(f"clause {raw!r}: negative literal {value}")


def validate(restriction: Restriction, left: Diagram, right: Diagram) -> Restriction:
    """``restriction`` with every clause's kind resolved against both diagrams, clauses equal to
    no constraint (``n >= 0``, ``d >= 1``) dropped and a fresh symbol chosen for each ``n >= c``;
    clauses of the new forms and on phase or exponent symbols intersected per symbol and
    planned (module docstring); DecideGrammarError naming the clause on any rule broken."""
    if not isinstance(restriction, Restriction):
        raise DecideGrammarError(
            f"restrict must be a Restriction, got {type(restriction).__name__}"
        )
    dims_l, mults_l, _ = _free_symbols(left)
    dims_r, mults_r, _ = _free_symbols(right)
    mults = mults_l | mults_r
    dims = (dims_l | dims_r) - mults
    phase_syms, scalar_syms = _phase_scalar_symbols(left, right)
    exps = scalar_syms - mults - dims
    phases = phase_syms - mults - dims - exps
    free = _all_symbols(left, right)
    allowed = ", ".join(sorted(mults | dims | exps | phases)) or "none"
    for clause in restriction.clauses:
        if clause.symbol not in exps | phases and clause.form in _OLD_FORMS:
            _integral_literals(clause)
    new_symbols = {
        c.symbol for c in restriction.clauses if c.form in _NEW_FORMS or c.symbol in exps | phases
    }
    seen: set[str] = set()
    taken: set[str] = set(free)
    for clause in restriction.clauses:
        if clause.form == AFFINE:
            taken.add(clause.fresh)
    out: list[Clause] = []
    fresh_used: set[str] = set()
    planned: set[str] = set()
    for clause in restriction.clauses:
        text = clause.text()
        if clause.symbol in new_symbols:
            if clause.symbol in planned:
                continue
            planned.add(clause.symbol)
            if clause.symbol in mults:
                kind = MULTIPLICITY
            elif clause.symbol in dims:
                kind = DIMENSION
            elif clause.symbol in exps:
                kind = EXPONENT
            elif clause.symbol in phases:
                kind = PHASE
            else:
                raise DecideGrammarError(
                    f"clause {text!r}: {clause.symbol} is free in neither diagram "
                    f"(may be constrained: {allowed})"
                )
            group = [c for c in restriction.clauses if c.symbol == clause.symbol]
            avoid = taken | fresh_used | {c.symbol for c in restriction.clauses}
            plan = _plan(clause.symbol, kind, group, avoid, left, right)
            for item in plan:
                if item.fresh:
                    taken.add(item.fresh)
                    fresh_used.add(item.fresh)
            out.extend(plan)
            continue
        if clause.symbol in seen:
            raise DecideGrammarError(f"clause {text!r}: a second clause on {clause.symbol}")
        seen.add(clause.symbol)
        if clause.symbol in mults:
            kind = MULTIPLICITY
        elif clause.symbol in dims:
            kind = DIMENSION
        else:
            raise DecideGrammarError(
                f"clause {text!r}: {clause.symbol} is free in neither diagram "
                f"(may be constrained: {allowed})"
            )
        if kind == DIMENSION:
            if clause.form == AFFINE:
                raise DecideGrammarError(f"clause {text!r}: a dimension takes only d = v or d >= c")
            if clause.c < 1:
                raise DecideGrammarError(f"clause {text!r}: a dimension needs a positive literal")
            if clause.form == GE and clause.c == 1:
                continue
            out.append(Clause(clause.symbol, kind, clause.form, c=clause.c))
            continue
        if clause.form == GE and clause.c == 0:
            continue
        if clause.form == AFFINE:
            if clause.fresh in free or clause.fresh in seen or clause.fresh in fresh_used:
                raise DecideGrammarError(
                    f"clause {text!r}: {clause.fresh} is not fresh (free symbols: "
                    f"{', '.join(sorted(free))})"
                )
            if any(c.symbol == clause.fresh for c in restriction.clauses):
                raise DecideGrammarError(
                    f"clause {text!r}: {clause.fresh} is constrained by another clause"
                )
            fresh_used.add(clause.fresh)
            out.append(Clause(clause.symbol, kind, AFFINE, clause.a, clause.c, clause.fresh))
            continue
        if clause.form == GE:
            fresh = _fresh("m", taken | fresh_used | {c.symbol for c in restriction.clauses})
            taken.add(fresh)
            fresh_used.add(fresh)
            out.append(Clause(clause.symbol, kind, GE, 1, clause.c, fresh))
            continue
        out.append(Clause(clause.symbol, kind, EQ, c=clause.c))
    result = Restriction(tuple(out))
    _check_bindings(left, result, "left")
    _check_bindings(right, result, "right")
    return result


def _ceil_on(x: Fraction, q: int, strict: bool) -> Fraction:
    """The least point of (1/q)Z at or (``strict``) above ``x``."""
    k = -((-x.numerator * q) // x.denominator)
    if strict and Fraction(k, q) == x:
        k += 1
    return Fraction(k, q)


def _floor_on(x: Fraction, q: int, strict: bool) -> Fraction:
    """The largest point of (1/q)Z at or (``strict``) below ``x``."""
    k = (x.numerator * q) // x.denominator
    if strict and Fraction(k, q) == x:
        k -= 1
    return Fraction(k, q)


def _periodic(symbol: str, *diagrams: Diagram) -> bool:
    """Whether every phase expression of ``diagrams`` carrying ``symbol`` changes by an
    integer number of turns when ``symbol`` grows by 1."""
    from archytaszx.algebra.phase import INDEX

    for diagram in diagrams:
        for node in diagram.nodes.values():
            phase = node.phase
            if phase is None or symbol not in phase.free_symbols:
                continue
            exprs = [phase.get(k).to_sympy_turns() for k in phase.entries()]
            if phase.rule is not None:
                exprs.append(phase.rule.to_sympy_turns())
            if phase.is_weighted:
                return False
            for expr in exprs:
                by_name = {str(x): x for x in expr.free_symbols}
                if symbol not in by_name:
                    continue
                x = by_name[symbol]
                diff = sp.expand(expr.subs(x, x + 1) - expr)
                index = {str(v): v for v in diff.free_symbols}.get(str(INDEX))
                if index is not None:
                    diff = diff.subs(index, sp.Symbol("k_int", integer=True))
                if diff.free_symbols or not diff.is_integer:
                    return False
    return True


def _plan(
    symbol: str,
    kind: str,
    group: Sequence[Clause],
    avoid: set[str],
    left: Diagram,
    right: Diagram,
) -> list[Clause]:
    """The intersection of one symbol's clauses as the clauses that apply it."""
    texts = ", ".join(c.text() if c.form in _NEW_FORMS else (c.raw or c.text()) for c in group)
    integer = kind in (MULTIPLICITY, DIMENSION)

    def bad(reason: str) -> DecideGrammarError:
        return DecideGrammarError(f"clauses {texts!r}: {reason}")

    lo: Fraction | None = (
        Fraction(0) if kind == MULTIPLICITY else Fraction(1) if kind == DIMENSION else None
    )
    hi: Fraction | None = None
    lo_strict = hi_strict = False
    q = 1 if integer else 0
    point: Fraction | None = None
    affine: Clause | None = None
    excluded: list[Fraction] = []

    def raise_lo(value: Fraction, strict: bool) -> None:
        nonlocal lo, lo_strict
        if lo is None or value > lo or (value == lo and strict):
            lo, lo_strict = value, strict

    def lower_hi(value: Fraction, strict: bool) -> None:
        nonlocal hi, hi_strict
        if hi is None or value < hi or (value == hi and strict):
            hi, hi_strict = value, strict

    for c in group:
        value = Fraction(c.c)
        if c.form == EQ:
            if point is not None and point != value:
                raise bad("the intersection is empty")
            point = value
        elif c.form == GE:
            raise_lo(value, False)
        elif c.form == GT:
            raise_lo(value, True)
        elif c.form == LE:
            lower_hi(value, False)
        elif c.form == LT:
            lower_hi(value, True)
        elif c.form == RANGE:
            raise_lo(value, c.lo_strict)
            assert c.hi is not None
            lower_hi(c.hi, c.hi_strict)
        elif c.form == LATTICE:
            if not integer:
                q = c.q if q == 0 else math.gcd(q, c.q)
        elif c.form == NE:
            excluded.append(value)
        elif c.form == AFFINE:
            if affine is not None:
                raise bad("two A*M + C clauses")
            affine = c
    if affine is not None:
        if (
            point is not None
            or hi is not None
            or any(c.form in (GE, GT, LATTICE, RANGE) for c in group)
        ):
            raise bad("A*M + C combines only with x != v")
        if integer:
            if kind == DIMENSION:
                raise bad("a dimension takes no A*M + C")
            if affine.fresh in avoid:
                raise bad(f"{affine.fresh} is not fresh")
            affine_base = Clause(
                symbol, kind, AFFINE, affine.a, affine.c, affine.fresh, source=texts
            )
            return [affine_base, *_side(symbol, kind, excluded, "")]
        if affine.fresh in avoid:
            raise bad(f"{affine.fresh} is not fresh")
        lattice = Domain(
            excluded=tuple(excluded), step=Fraction(affine.a), offset=Fraction(affine.c)
        )
        affine_shift = Clause(
            symbol,
            kind,
            SHIFT,
            a=affine.a,
            c=affine.c,
            fresh=affine.fresh,
            q=1,
            source=texts,
            domain=lattice,
            fresh_domain=Domain(q=1),
        )
        return [affine_shift]
    if q:
        if lo is not None:
            lo, lo_strict = _ceil_on(lo, q, lo_strict), False
        if hi is not None:
            hi, hi_strict = _floor_on(hi, q, hi_strict), False
    domain = Domain(lo, lo_strict, hi, hi_strict, q, tuple(sorted(set(excluded))))
    if lo is not None and hi is not None and (lo > hi or (lo == hi and (lo_strict or hi_strict))):
        raise bad("the intersection is empty")
    if point is not None:
        if not domain.contains(point):
            raise bad("the intersection is empty")
        exact = int(point) if point.denominator == 1 else point
        return [Clause(symbol, kind, EQ, c=_slot(exact), source=texts, domain=domain)]
    if q and lo is not None and hi is not None:
        points = [Fraction(k, q) for k in range(int(lo * q), int(hi * q) + 1)]
        points = [v for v in points if domain.contains(v)]
        if not points:
            raise bad("the intersection is empty")
        if len(points) == 1:
            v = points[0]
            single = int(v) if v.denominator == 1 else v
            return [Clause(symbol, kind, EQ, c=_slot(single), source=texts, domain=domain)]
        if len(points) > _MAX_INSTANCES:
            raise bad(f"{len(points)} instances, over {_MAX_INSTANCES}")
        return [Clause(symbol, kind, FINITE, source=texts, domain=domain, values=tuple(points))]
    if kind == PHASE and q and _periodic(symbol, left, right):
        values = tuple(Fraction(j, q) for j in range(q))
        rep_domain = Domain(q=q, excluded=domain.excluded)
        return [Clause(symbol, kind, FINITE, source=texts, domain=rep_domain, values=values)]
    if integer:
        assert lo is not None
        floor = int(lo)
        base: list[Clause] = []
        if kind == DIMENSION:
            if floor > 1:
                base = [Clause(symbol, kind, GE, c=floor, source=texts)]
        elif floor != 0:
            base = [Clause(symbol, kind, GE, 1, floor, _fresh("m", avoid), source=texts)]
        side = _side(symbol, kind, domain.excluded, "" if base else texts)
        if side:
            side = [replace(side[0], domain=domain)]
        return [*base, *side]
    stem = "k" if q else "s"
    fresh = _fresh(stem, avoid)
    unit = Fraction(1, q) if q else Fraction(1)
    scale = Fraction(1, unit.denominator) if q else Fraction(1)

    def shifted(a: Fraction, c: Fraction, fd: Domain) -> list[Clause]:
        return [
            Clause(
                symbol,
                kind,
                SHIFT,
                a=_slot(a),
                c=_slot(c),
                fresh=fresh,
                q=q,
                source=texts,
                domain=domain,
                fresh_domain=fd,
            )
        ]

    if lo is not None:
        span = None if hi is None else (hi - lo) / scale
        return shifted(scale, lo, Domain(Fraction(0), lo_strict, span, hi_strict, 1 if q else 0))
    if hi is not None:
        return shifted(-scale, hi, Domain(Fraction(0), hi_strict, None, False, 1 if q else 0))
    if q:
        return shifted(scale, Fraction(0), Domain(q=1))
    return [Clause(symbol, kind, SIDE, source=texts, domain=domain)]


def _side(symbol: str, kind: str, excluded: Sequence[Fraction], texts: str) -> list[Clause]:
    """A side condition excluding ``excluded``, when there is one."""
    if not excluded:
        return []
    return [Clause(symbol, kind, SIDE, source=texts, domain=Domain(excluded=tuple(excluded)))]


def in_domain(restriction: Restriction, assignment: Mapping[str, object]) -> bool:
    """Whether ``assignment`` gives every constrained symbol a value its clause covers."""
    return all(
        clause.symbol in assignment and clause.contains(assignment[clause.symbol])
        for clause in restriction.clauses
    )


def to_restricted(
    restriction: Restriction, assignment: Mapping[str, object]
) -> dict[str, object] | None:
    """``assignment`` over the restricted pair's symbols (``m = (n - c)/a``, substituted symbols
    dropped), or None when it lies outside the clauses."""
    out = dict(assignment)
    for clause in restriction.clauses:
        if clause.symbol not in out:
            continue
        value = out[clause.symbol]
        if not clause.contains(value):
            return None
        if not clause.substitutes:
            continue
        del out[clause.symbol]
        if clause.form == SHIFT:
            x = _as_fraction(value)
            assert x is not None
            step = (x - Fraction(clause.c)) / Fraction(clause.a)
            if clause.fresh_domain is not None and not clause.fresh_domain.contains(step):
                return None
            out[clause.fresh] = int(step) if step.denominator == 1 else step
            continue
        if clause.kind in (PHASE, EXPONENT):
            continue
        if clause.kind == MULTIPLICITY and clause.form != EQ:
            out[clause.fresh] = (int(value) - clause.c) // clause.a  # type: ignore[call-overload]
    return out


def to_original(restriction: Restriction, assignment: Mapping[str, object]) -> dict[str, object]:
    """``assignment`` over the restricted pair read in the original symbols: ``n = a*m + c``,
    ``n = v`` and ``d = v`` added, each fresh symbol dropped."""
    out = dict(assignment)
    for clause in restriction.clauses:
        if not clause.substitutes:
            continue
        if clause.form == EQ:
            out[clause.symbol] = clause.c
            continue
        if clause.form == SHIFT and clause.fresh in out:
            step = _as_fraction(out.pop(clause.fresh))
            assert step is not None
            value = Fraction(clause.c) + Fraction(clause.a) * step
            out[clause.symbol] = int(value) if value.denominator == 1 else value
            continue
        if clause.fresh in out:
            m = out.pop(clause.fresh)
            out[clause.symbol] = clause.a * int(m) + clause.c  # type: ignore[call-overload]
    return dict(sorted(out.items()))


def _constant(multiplicity: Mult) -> int:
    expr = sp.expand(multiplicity.to_sympy())
    constant = expr.as_coeff_Add()[0]
    return int(constant) if constant.is_Integer else 0


def _settle(diagram: Diagram) -> Diagram:
    """``diagram`` with concrete boxes expanded and every box whose symbolic multiplicity has a
    constant term ``c >= 1`` peeled until it has none, least box id first."""
    working = expand_concrete_boxes(diagram)
    while True:
        target = next(
            (
                box_id
                for box_id, box in sorted(working.bang_boxes.items())
                if not box.multiplicity.is_concrete and _constant(box.multiplicity) >= 1
            ),
            None,
        )
        if target is None:
            return working
        working = expand_concrete_boxes(peel_one(working, target).diagram)


def _drop_parameter(diagram: Diagram, name: str) -> Diagram:
    if name not in diagram.parameters:
        return diagram
    out = diagram.copy()
    out.set_parameters({k: v for k, v in diagram.parameters.items() if k != name})
    return out


def apply_clause(diagram: Diagram, clause: Clause) -> Diagram:
    """``diagram`` with one clause substituted (see the module docstring); a new Diagram."""
    if clause.form == FINITE:
        raise DecideGrammarError(
            f"clause {clause.text()!r}: {clause.symbol} takes finitely many values; the claim is "
            "decided instance by instance"
        )
    if not clause.substitutes:
        return diagram.copy()
    if clause.kind in (PHASE, EXPONENT):
        return _apply_real(diagram, clause)
    if clause.kind == DIMENSION:
        dims, _mults, _others = _free_symbols(diagram)
        if clause.symbol not in dims:
            return _drop_parameter(diagram.copy(), clause.symbol)
        return _drop_parameter(diagram.substitute({clause.symbol: clause.c}), clause.symbol)
    working = _drop_parameter(diagram.copy(), clause.symbol)
    if clause.form == EQ:
        if clause.symbol in free_mult_symbols(working):
            working = instantiate_symbol(working, clause.symbol, clause.c)
        if clause.symbol in _all_symbols(working):
            working = working.substitute({clause.symbol: clause.c})
        return working
    if clause.symbol not in _all_symbols(working):
        return working
    n = sp.Symbol(clause.symbol, integer=True, nonnegative=True, multiplicity=True)
    value = clause.a * Mult(clause.fresh).to_sympy() + clause.c
    for box_id, box in sorted(working.bang_boxes.items()):
        if clause.symbol in box.multiplicity.free_symbols:
            expr = sp.expand(box.multiplicity.to_sympy().subs(n, value))
            working.set_bang_box_multiplicity(box_id, Mult._from_expr(expr))
    if clause.symbol in working.scalar.free_symbols:
        working.set_scalar(working.scalar.substitute({clause.symbol: Scalar(value)}))
    if clause.symbol in _all_symbols(working):
        raise DecideGrammarError(
            f"clause {clause.text()!r}: {clause.symbol} occurs outside the bang boxes and the "
            "scalar"
        )
    return _settle(working)


def _apply_real(diagram: Diagram, clause: Clause) -> Diagram:
    """A phase or exponent clause substituted: ``x = v`` by its rational, ``x = c + a*s`` by
    the expression over the fresh symbol (an integer symbol on a lattice)."""
    from archytaszx.algebra.phase import Phase

    working = _drop_parameter(diagram.copy(), clause.symbol)
    if clause.symbol not in _all_symbols(working):
        return working
    if clause.form == EQ:
        value = Fraction(clause.c)
        rational = sp.Rational(value.numerator, value.denominator)
        return working.substitute(
            {clause.symbol: int(value) if value.denominator == 1 else rational}
        )
    fresh = (
        sp.Symbol(clause.fresh, integer=True) if clause.q else sp.Symbol(clause.fresh, real=True)
    )
    a, c = Fraction(clause.a), Fraction(clause.c)
    expr = sp.Rational(c.numerator, c.denominator) + sp.Rational(a.numerator, a.denominator) * fresh
    working = working.substitute({clause.symbol: Phase(expr)})
    if clause.symbol in working.scalar.free_symbols:
        working = working.substitute({clause.symbol: Scalar(expr)})
    return working


def subst_one(diagram: Diagram, restriction: Restriction) -> Diagram:
    """``diagram`` with every clause of ``restriction`` substituted in order."""
    out = diagram
    for clause in restriction.clauses:
        out = apply_clause(out, clause)
    return out


def _check_bindings(diagram: Diagram, restriction: Restriction, side: str) -> None:
    for clause in restriction.clauses:
        if clause.form != EQ or clause.symbol not in diagram.parameters:
            continue
        bound = diagram.parameters[clause.symbol]
        if bound != clause.c:
            raise DecideGrammarError(
                f"clause {clause.text()!r}: the {side} side binds {clause.symbol} = {bound}"
            )


def subst_pair(left: Diagram, right: Diagram, restriction: Restriction) -> tuple[Diagram, Diagram]:
    """Both sides with every clause substituted; DecideGrammarError when a side binds a clause's
    symbol to another value."""
    _check_bindings(left, restriction, "left")
    _check_bindings(right, restriction, "right")
    return subst_one(left, restriction), subst_one(right, restriction)


def restricted_samples(
    restriction: Restriction, samples: Sequence[Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    """``samples`` inside the clauses, read over the restricted pair's symbols, in order and
    deduplicated."""
    out: list[Mapping[str, object]] = []
    for sample in samples:
        point = to_restricted(restriction, sample)
        if point is not None and point not in out:
            out.append(MappingProxyType(point))
    return tuple(out)
