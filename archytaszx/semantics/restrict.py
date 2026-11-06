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
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

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
EQ = "eq"
GE = "ge"
AFFINE = "affine"

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CLAUSE_RE = re.compile(rf"\s*({_IDENT})\s*(>=|≥|=)\s*(.*?)\s*")
_INT_RE = re.compile(r"[+-]?\d+")
_AFFINE_RE = re.compile(rf"(?:([+-]?\d+)\s*\*?\s*)?({_IDENT})\s*(?:\+\s*([+-]?\d+))?")


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

    @property
    def substitutes(self) -> bool:
        """True when the clause changes the diagrams (every clause but a dimension floor)."""
        return not (self.kind == DIMENSION and self.form == GE)

    def expression(self) -> str:
        """The value the symbol takes: ``c``, or ``a*m + c`` in its shortest form."""
        if self.form == EQ:
            return str(self.c)
        head = self.fresh if self.a == 1 else f"{self.a}*{self.fresh}"
        return f"{head} + {self.c}" if self.c else head

    def text(self) -> str:
        """The clause as written in the grammar."""
        if self.form == GE:
            return f"{self.symbol} >= {self.c}"
        if self.form == EQ:
            return f"{self.symbol} = {self.c}"
        return f"{self.symbol} = {self.expression()}"

    def contains(self, value: object) -> bool:
        """Whether ``value`` lies in the clause's set of values."""
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
        """Each constrained symbol's clause."""
        return MappingProxyType({c.symbol: c for c in self.clauses})

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
        return ", ".join(c.text() for c in self.clauses)


def _literal(text: str, clause: str) -> int:
    if not _INT_RE.fullmatch(text):
        raise DecideGrammarError(f"clause {clause!r}: {text!r} is not an integer literal")
    value = int(text)
    if value < 0:
        raise DecideGrammarError(f"clause {clause!r}: negative literal {value}")
    return value


def parse_clause(text: str) -> Clause:
    """One clause of the grammar, its kind left to :func:`validate`."""
    clause = text.strip()
    match = _CLAUSE_RE.fullmatch(clause)
    if match is None or not match[3]:
        raise DecideGrammarError(
            f"clause {clause!r}: expected NAME = INT, NAME >= INT or NAME = A*M + C"
        )
    symbol, op, rhs = match[1], match[2], match[3]
    if op != "=":
        return Clause(symbol, "", GE, c=_literal(rhs, clause))
    if _INT_RE.fullmatch(rhs):
        return Clause(symbol, "", EQ, c=_literal(rhs, clause))
    affine = _AFFINE_RE.fullmatch(rhs)
    if affine is None:
        raise DecideGrammarError(
            f"clause {clause!r}: {rhs!r} is neither an integer literal nor A*M + C"
        )
    a = _literal(affine[1], clause) if affine[1] is not None else 1
    c = _literal(affine[3], clause) if affine[3] is not None else 0
    if a == 0:
        raise DecideGrammarError(f"clause {clause!r}: a = 0; write {symbol} = {c}")
    return Clause(symbol, "", AFFINE, a=a, c=c, fresh=affine[2])


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


def validate(restriction: Restriction, left: Diagram, right: Diagram) -> Restriction:
    """``restriction`` with every clause's kind resolved against both diagrams, clauses equal to
    no constraint (``n >= 0``, ``d >= 1``) dropped and a fresh symbol chosen for each ``n >= c``;
    DecideGrammarError naming the clause on any rule of the grammar broken."""
    if not isinstance(restriction, Restriction):
        raise DecideGrammarError(
            f"restrict must be a Restriction, got {type(restriction).__name__}"
        )
    dims_l, mults_l, _ = _free_symbols(left)
    dims_r, mults_r, _ = _free_symbols(right)
    mults = mults_l | mults_r
    dims = (dims_l | dims_r) - mults
    free = _all_symbols(left, right)
    allowed = ", ".join(sorted(mults | dims)) or "none"
    seen: set[str] = set()
    taken: set[str] = set(free)
    for clause in restriction.clauses:
        if clause.form == AFFINE:
            taken.add(clause.fresh)
    out: list[Clause] = []
    fresh_used: set[str] = set()
    for clause in restriction.clauses:
        text = clause.text()
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
    if not clause.substitutes:
        return diagram.copy()
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
