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

"""Text input: graph listings, the spider/wire DSL, Dirac expressions, and rule steps.

:func:`parse` reads one source string in any :class:`InputForm`; no function takes a notation
mode. The forms:

* LISTING, :func:`parse_listing`: the exact inverse of
  :func:`~archytaszx.repl.printer.render_graph`. Node and box ids are kept; nothing is
  abstracted.
* DSL, :func:`parse_dsl`: node, ``wire``, ``inputs``/``outputs``, ``scalar``, ``box`` and
  ``dim``/``count``/``let`` statements, split by newline or ``;``, ``#`` comments.
* DIRAC, :func:`parse_dirac`; :func:`parse_dirac_source` is the Phase 5 slice:
  ``sum_{k=0}^{D-1} |k,k,...>`` or ``|k>^{n}``, optionally followed by ``; copy``.
* STEP, :func:`parse_step` and :func:`apply_step`: ``apply RULE [at INT | at nI, nJ, ...]``.

Expressions (dims, multiplicities, phase turns, scalars) go through one recursive-descent
parser to an AST, then a role evaluator builds the value; no user text reaches sympy's own
parser.

Abstraction on entry (DSL, Dirac): a numeral dimension becomes a symbol bound in the parameter
environment, one symbol per value per parse; a concrete count becomes a fresh box symbol per
occurrence. Minted names avoid every identifier in the source. ``literal N`` keeps ``N``.

Every returned diagram passes :func:`~archytaszx.diagram.validate.validate` with no error and
no unknown-parameter issue. Every failure is a :class:`ParseError`.
"""

from __future__ import annotations

import enum
import functools
import math
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ParamSpec, TypeAlias, TypeVar

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionDomainError, DimensionError
from archytaszx.algebra.phase import INDEX, Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import BangBoxError, Mult, abstract_port_count
from archytaszx.diagram.generators import REGISTRY, X_SPIDER, Z_SPIDER, GeneratorType
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, Node, NodeId, PortRef
from archytaszx.diagram.validate import IssueKind, validate
from archytaszx.rewrite.engine import RewriteResult, apply
from archytaszx.rewrite.rules_library import RULES


class ParseError(Exception):
    """Base of every error this module raises."""


class ParseGrammarError(ParseError):
    """The source does not match the grammar of the form it was read as."""


class ParseDomainError(ParseError):
    """The source parses, but names a value or a diagram outside the accepted domain."""


DiracError = ParseError
DiracGrammarError = ParseGrammarError
DiracDomainError = ParseDomainError


# -- legacy Phase 5 Dirac slice ------------------------------------------------------------

_SUMMATION_INDEX = "k"
"""The bound summation variable of the legacy slice."""

_ASCII_DIGITS = "[0-9]+"
"""The decimal-literal shape, ASCII only, as a regex fragment. ``07`` is 7."""

_ASCII_DIGITS_RE = re.compile(rf"^{_ASCII_DIGITS}$")
"""Whole-token form of :data:`_ASCII_DIGITS`."""

_IDENTIFIER = r"[A-Za-z_]\w*"
"""The bare-identifier shape, as a regex fragment."""

_IDENTIFIER_RE = re.compile(rf"^{_IDENTIFIER}$")
"""Whole-token form of :data:`_IDENTIFIER`."""

_MAX_KET_LEG_COUNT = 1024
"""Cap on an explicitly expanded leg count."""

_MAX_EXPANDED_LEGS = 65536
"""Cap on the legs one Dirac parse expands."""

_LITERAL_MARKER = "literal"
"""Keyword before a numeral that suppresses abstraction on entry."""

_KET_SUM_RE = re.compile(
    rf"^sum_\{{{_SUMMATION_INDEX}=0\}}\^\{{(?:(?P<literal>{_LITERAL_MARKER})\s+)?"
    rf"(?P<dim>{_IDENTIFIER}|{_ASCII_DIGITS})-1\}}\s*"
    rf"\|(?P<body>[^>]*)>"
    rf"(?:\^\{{(?:\\otimes|⊗)?\s*(?:(?P<power_literal>{_LITERAL_MARKER})\s+)?"
    rf"(?P<power>{_ASCII_DIGITS})\s*\}})?$"
)
"""Matches ``sum_{k=0}^{D-1} |BODY>`` with an optional ``^{n}``/``^{\\otimes n}``/``^{⊗n}``."""


_MAX_NUMERAL_DIGITS = 18
"""Longest digit run converted with ``int()``."""


def _numeral(token: str, what: str) -> int:
    """The int a checked digit run names, as this module's own error when it is too long."""
    if len(token) > _MAX_NUMERAL_DIGITS:
        raise DiracDomainError(
            f"{what} {token[:16]}... is {len(token)} digits long, above this parser's bound "
            f"of {_MAX_NUMERAL_DIGITS}"
        )
    return int(token)


def _leg_count_from_body(body: str, power: str | None, *, eager: bool = True) -> int:
    """The output-leg count of a ket body (one leg per ``k``, commas and spaces ignored), or
    of the ``power`` suffix on a single-``k`` body."""
    stripped = body.replace(",", "").replace(" ", "")
    if not stripped or any(ch != "k" for ch in stripped):
        raise DiracGrammarError(
            f"unsupported ket body {body!r}: this slice only accepts repeated 'k' indices, "
            "e.g. '|k,k>' or '|kk>'"
        )
    if power is not None:
        if stripped != "k":
            raise DiracGrammarError(
                f"ket body {body!r} combined with a tensor-power suffix ^{{{power}}} is "
                f"ambiguous: use either '|k>^{{n}}' or '|k,k,...>', not both"
            )
        leg_count = _numeral(power, "tensor-power count")
    else:
        leg_count = len(stripped)
    if leg_count < 1:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs; at least 1 is required"
        )
    if eager and leg_count > _MAX_KET_LEG_COUNT:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs, above this parser's sanity bound "
            f"of {_MAX_KET_LEG_COUNT} (see _MAX_KET_LEG_COUNT's docstring)"
        )
    return leg_count


def _parse_dim(token: str, *, literal: bool) -> tuple[Dim, Mapping[str, int]]:
    """The dimension a token names, and the parameter binding it records.

    A numeral is abstracted through :meth:`~archytaszx.algebra.dimension.Dim.abstract`
    unless ``literal``; an identifier is a symbolic ``Dim`` with no binding.
    """
    if _ASCII_DIGITS_RE.match(token):
        try:
            concrete = Dim.concrete(_numeral(token, "dimension token"))
        except DimensionDomainError as exc:
            raise DiracDomainError(
                f"dimension token {token!r} is outside Dim's domain: {exc}"
            ) from exc
        if literal:
            return concrete, {}
        return concrete.abstract()
    if literal:
        raise DiracGrammarError(
            f"the {_LITERAL_MARKER!r} marker suppresses abstraction of a numeric dimension, "
            f"but {token!r} is not a numeral; a symbolic dimension is never abstracted"
        )
    if token == _SUMMATION_INDEX:
        raise DiracDomainError(
            f"dimension token {token!r} is the bound summation index; a dimension symbol "
            "must be free, not the variable the enclosing sum binds"
        )
    if not _IDENTIFIER_RE.match(token):
        raise DiracGrammarError(
            f"dimension token {token!r} is neither a decimal literal ({_ASCII_DIGITS}) nor "
            f"a bare identifier ({_IDENTIFIER})"
        )
    try:
        return Dim.symbol(token), {}
    except DimensionError as exc:
        raise DiracGrammarError(
            f"dimension token {token!r} matches this module's identifier shape "
            f"({_IDENTIFIER}, Unicode-aware) but is not a name Dim accepts: {exc}"
        ) from exc


def _parse_ket_sum(text: str) -> tuple[Dim, Mapping[str, int], int, bool]:
    """Parse ``sum_{k=0}^{D-1} |...>`` into dim, parameter binding, leg count and whether
    the count is to be abstracted into a bang box."""
    match = _KET_SUM_RE.match(text.strip())
    if match is None:
        raise DiracGrammarError(
            f"{text!r} does not match this slice's grammar: expected "
            "'sum_{k=0}^{D-1} |k,k,...>' (or the '|k>^{n}' shorthand)"
        )
    dim, binding = _parse_dim(match.group("dim"), literal=match.group("literal") is not None)
    power = match.group("power")
    abstract_count = power is not None and match.group("power_literal") is None
    leg_count = _leg_count_from_body(match.group("body"), power, eager=not abstract_count)
    return dim, binding, leg_count, abstract_count


def parse_dirac_source(source: str) -> Diagram:
    """The legacy Phase 5 slice: one Z state ``sum_{k=0}^{D-1} |k,k,...>`` with boundary
    outputs in ket order, or that state's first output wired into a 1->2 copy spider
    (``; copy``), boundary the state's other outputs then the copy's two. A non-literal
    ``^{n}`` becomes one leg in a port-scope box bound to ``n``."""
    terms = [term.strip() for term in source.split(";")]
    if len(terms) not in (1, 2) or any(not term for term in terms):
        raise DiracGrammarError(
            f"{source!r}: expected a single ket-sum term, optionally followed by one "
            "';' and the keyword 'copy'"
        )
    dim, binding, leg_count, abstract_count = _parse_ket_sum(terms[0])

    diagram = Diagram()
    diagram.set_parameters(binding)
    built_legs = 1 if abstract_count else leg_count
    if abstract_count and len(terms) == 2:
        built_legs = 2
    state_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim] * built_legs)

    if len(terms) == 1:
        diagram.set_boundary_outputs(
            [PortRef(state_id, Direction.OUTPUT, i) for i in range(built_legs)]
        )
        if abstract_count:
            diagram, _box_id, _mult = abstract_port_count(
                diagram, PortRef(state_id, Direction.OUTPUT, 0), leg_count, stem="n"
            )
        return diagram

    if terms[1] != "copy":
        raise DiracGrammarError(
            f"{terms[1]!r}: this slice recognizes only the 'copy' keyword after ';' "
            "(a general spider-declaration syntax is Phase 18's, not this slice's)"
        )
    copy_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(PortRef(state_id, Direction.OUTPUT, 0), PortRef(copy_id, Direction.INPUT, 0))
    boundary_outputs = [PortRef(state_id, Direction.OUTPUT, i) for i in range(1, built_legs)] + [
        PortRef(copy_id, Direction.OUTPUT, 0),
        PortRef(copy_id, Direction.OUTPUT, 1),
    ]
    diagram.set_boundary_outputs(boundary_outputs)
    if abstract_count:
        diagram, _box_id, _mult = abstract_port_count(
            diagram, PortRef(state_id, Direction.OUTPUT, 1), leg_count - 1, stem="n"
        )
    return diagram


# -- shared infrastructure -----------------------------------------------------------------

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wrap(exc: Exception) -> ParseError:
    """``exc`` as a :class:`ParseDomainError` (a ``*DomainError`` class or a resource or
    limit failure) or a :class:`ParseGrammarError` (anything else)."""
    if isinstance(exc, ParseError):
        return exc
    if isinstance(exc, RecursionError):
        return ParseGrammarError("input nests too deeply")
    limit = isinstance(exc, OverflowError | MemoryError) or (
        type(exc) is ValueError and "limit" in str(exc)
    )
    kind = ParseDomainError if limit or "Domain" in type(exc).__name__ else ParseGrammarError
    return kind(f"{type(exc).__name__}: {exc}")


def _guarded(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """``func`` with every exception re-raised as a :class:`ParseError`."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return func(*args, **kwargs)
        except ParseError:
            raise
        except Exception as exc:
            raise _wrap(exc) from exc

    return wrapper


def _require_text(source: object) -> str:
    if not isinstance(source, str):
        raise ParseGrammarError(f"source must be a str, got {type(source).__name__}")
    return source


def _resolved(dim: Dim, parameters: Mapping[str, int]) -> sp.Expr:
    """``dim`` with ``parameters`` substituted under the expression size bounds."""

    def walk(expr: sp.Expr) -> sp.Expr:
        if expr.is_Symbol:
            value = parameters.get(str(expr.name))
            return expr if value is None else sp.Integer(value)
        if expr.is_Pow:
            return _power(walk(expr.base), walk(expr.exp))
        if expr.is_Mul:
            return _product([walk(arg) for arg in expr.args])
        if expr.is_Add:
            return _bounded(sp.Add(*(walk(arg) for arg in expr.args)))
        return expr

    return walk(dim.to_sympy())


def _finish(diagram: Diagram) -> Diagram:
    """``diagram``, once its dimensions resolve within the size bounds and :func:`validate`
    reports no error and no unknown parameter."""
    seen: set[Dim] = set()
    for node in diagram.nodes.values():
        for dim in (*(p.dim for p in (*node.inputs, *node.outputs)), *_phase_dims(node)):
            if dim not in seen:
                seen.add(dim)
                _resolved(dim, diagram.parameters)
    bad = [
        issue
        for issue in validate(diagram).issues
        if not issue.deferred or issue.kind is IssueKind.PARAMETER_UNKNOWN_SYMBOL
    ]
    if bad:
        raise ParseDomainError(
            "input is not a valid diagram: " + "; ".join(issue.message for issue in bad)
        )
    return diagram


def _phase_dims(node: Node) -> tuple[Dim, ...]:
    return () if node.phase is None else (node.phase.dim,)


def _split_top(text: str, sep: str = ",") -> list[str]:
    """``text`` split at ``sep`` outside brackets, parts stripped; ``[]`` for blank text."""
    if not text.strip():
        return []
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise ParseGrammarError(f"unbalanced {ch!r} in {text!r}")
        elif ch == sep and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    if depth != 0:
        raise ParseGrammarError(f"unbalanced brackets in {text!r}")
    parts.append(text[start:].strip())
    if any(not part for part in parts):
        raise ParseGrammarError(f"empty item in {text!r}")
    return parts


def _int_token(token: str, what: str) -> int:
    """The int an ASCII digit run names."""
    token = token.strip()
    if not _ASCII_DIGITS_RE.match(token):
        raise ParseGrammarError(f"{what} must be a decimal integer, got {token!r}")
    return _numeral(token, what)


_ANY_IDENTIFIER_RE = re.compile(r"[^\W\d]\w*")


class _Abstraction:
    """Abstraction on entry for one parse: one dimension symbol per value (stem ``d``), a
    fresh count symbol per call (stem ``n``), names avoiding every identifier in the source.
    :attr:`bindings` records every minted symbol's value."""

    def __init__(self, source: str, reuse: Mapping[int, str] | None = None) -> None:
        self._taken: set[str] = set(_ANY_IDENTIFIER_RE.findall(source))
        self._dims: dict[int, str] = dict(reuse or {})
        self.bindings: dict[str, int] = {}

    def dim(self, value: int, literal: bool = False) -> Dim:
        """The dimension a numeral stands for: shared symbol, or concrete under ``literal``."""
        concrete = Dim.concrete(value)
        if literal:
            return concrete
        name = self._dims.get(value)
        if name is None:
            symbol, binding = concrete.abstract(avoid=self._taken, stem="d")
            name = next(iter(binding))
            self._taken.add(name)
            self._dims[value] = name
            self.bindings.update(binding)
            return symbol
        return Dim.symbol(name)

    def count(self, value: int) -> Mult:
        """A fresh multiplicity symbol bound to ``value``."""
        symbol, binding = Mult(value).abstract(avoid=self._taken, stem="n")
        self._taken.update(binding)
        self.bindings.update(binding)
        return symbol


# -- expressions ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Num:
    value: int


@dataclass(frozen=True, slots=True)
class _Lit:
    value: int


@dataclass(frozen=True, slots=True)
class _Name:
    name: str


@dataclass(frozen=True, slots=True)
class _Const:
    name: str


@dataclass(frozen=True, slots=True)
class _Neg:
    arg: _Expr


@dataclass(frozen=True, slots=True)
class _Sum:
    terms: tuple[tuple[bool, _Expr], ...]
    """``(negated, term)`` pairs."""


@dataclass(frozen=True, slots=True)
class _Prod:
    factors: tuple[tuple[bool, _Expr], ...]
    """``(divided, factor)`` pairs."""


@dataclass(frozen=True, slots=True)
class _Pow:
    base: _Expr
    exponent: _Expr


@dataclass(frozen=True, slots=True)
class _Call:
    fn: str
    arg: _Expr


_Expr: TypeAlias = _Num | _Lit | _Name | _Const | _Neg | _Sum | _Prod | _Pow | _Call

_EXPR_TOKEN_RE = re.compile(r"(?P<int>[0-9]+)|(?P<name>[^\W\d]\w*)|(?P<op>\*\*|[-+*/^(),])")
_SPACE_RE = re.compile(r"\s*")
_CONSTANTS = {"I": sp.I, "pi": sp.pi, "π": sp.pi, "E": sp.E}
_FUNCTIONS = frozenset({"exp", "sqrt"})
_MAX_EXPR_DEPTH = 100
_MAX_EXPR_DIGITS = 1000
_INT_LIMIT = 10**_MAX_EXPR_DIGITS
_MAX_POWER_BITS = 4096
_MAX_ROOT_BITS = 256
_MAX_TERMS = 4096


def _expr_numeral(token: str) -> int:
    """The int of an expression digit run of at most :data:`_MAX_EXPR_DIGITS` digits."""
    if len(token) > _MAX_EXPR_DIGITS:
        raise ParseDomainError(
            f"numeral {token[:16]}... is {len(token)} digits long, above the bound of "
            f"{_MAX_EXPR_DIGITS}"
        )
    return int(token)


class _ExprParser:
    """Recursive descent over ``text`` from ``pos``: :meth:`expression` reads the longest
    expression prefix and leaves :attr:`pos` just after it. ``^``/``**`` bind tighter than
    unary minus and associate right."""

    def __init__(self, text: str, pos: int = 0) -> None:
        self.text = text
        self.pos = pos
        self._depth = 0
        self._peeked: tuple[int, tuple[str, str, int, int]] | None = None

    def peek(self) -> tuple[str, str, int, int]:
        """``(kind, text, start, end)`` of the next token; kind ``end`` or ``other`` past
        the expression alphabet."""
        if self._peeked is not None and self._peeked[0] == self.pos:
            return self._peeked[1]
        pos = _SPACE_RE.match(self.text, self.pos).end()  # type: ignore[union-attr]
        token: tuple[str, str, int, int]
        if pos >= len(self.text):
            token = ("end", "", pos, pos)
        elif (match := _EXPR_TOKEN_RE.match(self.text, pos)) is None:
            token = ("other", self.text[pos], pos, pos + 1)
        else:
            token = (match.lastgroup or "op", match.group(), pos, match.end())
        self._peeked = (self.pos, token)
        return token

    def _error(self, message: str) -> ParseGrammarError:
        kind, text, start, _ = self.peek()
        found = "end of input" if kind == "end" else repr(text)
        return ParseGrammarError(f"{message} at column {start + 1} of {self.text!r}, found {found}")

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > _MAX_EXPR_DEPTH:
            raise ParseGrammarError(f"expression nests deeper than {_MAX_EXPR_DEPTH} levels")

    def _op(self, *ops: str) -> str | None:
        kind, text, _, end = self.peek()
        if kind == "op" and text in ops:
            self.pos = end
            return text
        return None

    def expression(self) -> _Expr:
        """``term (('+' | '-') term)*``."""
        self._enter()
        terms: list[tuple[bool, _Expr]] = [(False, self._term())]
        while (op := self._op("+", "-")) is not None:
            if len(terms) >= _MAX_TERMS:
                raise ParseDomainError(f"sum of more than {_MAX_TERMS} terms")
            terms.append((op == "-", self._term()))
        self._depth -= 1
        return terms[0][1] if len(terms) == 1 else _Sum(tuple(terms))

    def _term(self) -> _Expr:
        factors: list[tuple[bool, _Expr]] = [(False, self._unary())]
        while (op := self._op("*", "/")) is not None:
            factors.append((op == "/", self._unary()))
        return factors[0][1] if len(factors) == 1 else _Prod(tuple(factors))

    def _unary(self) -> _Expr:
        self._enter()
        result: _Expr
        if self._op("-") is not None:
            result = _Neg(self._unary())
        else:
            result = self._atom()
            if self._op("^", "**") is not None:
                result = _Pow(result, self._unary())
        self._depth -= 1
        return result

    def _atom(self) -> _Expr:
        kind, text, _, end = self.peek()
        if kind == "int":
            self.pos = end
            return _Num(_expr_numeral(text))
        if kind == "op" and text == "(":
            self.pos = end
            inner = self.expression()
            if self._op(")") is None:
                raise self._error("expected ')'")
            return inner
        if kind != "name":
            raise self._error("expected a number, a name or '('")
        if not text.isidentifier():
            raise self._error("not an identifier")
        self.pos = end
        if text == _LITERAL_MARKER and self.peek()[0] == "int":
            _, digits, _, after = self.peek()
            self.pos = after
            return _Lit(_expr_numeral(digits))
        if text in _CONSTANTS:
            return _Const(text)
        if text in _FUNCTIONS and self._op("(") is not None:
            arg = self.expression()
            if self._op(")") is None:
                raise self._error("expected ')'")
            return _Call(text, arg)
        return _Name(text)


def _parse_expr(text: str, what: str = "expression") -> _Expr:
    """The AST of the whole of ``text``."""
    parser = _ExprParser(text)
    if parser.peek()[0] == "end":
        raise ParseGrammarError(f"empty {what}")
    node = parser.expression()
    if parser.peek()[0] != "end":
        raise parser._error(f"unexpected text after the {what}")
    return node


def _bits(value: sp.Rational) -> int:
    return max(int(value.p).bit_length(), int(value.q).bit_length())


def _numeric_bits(expr: sp.Expr) -> int:
    """Total bit size of the rational atoms of ``expr`` other than 0 and +-1."""
    return sum(_bits(a) for a in expr.atoms(sp.Rational) if a not in (0, 1, -1))


def _bounded(expr: sp.Expr) -> sp.Expr:
    """``expr``, once every rational atom has at most :data:`_MAX_EXPR_DIGITS` digits."""
    for atom in expr.atoms(sp.Rational):
        if abs(atom.p) >= _INT_LIMIT or atom.q >= _INT_LIMIT:
            raise ParseDomainError(f"a value has more than {_MAX_EXPR_DIGITS} digits")
    return expr


def _product(values: Sequence[sp.Expr]) -> sp.Expr:
    """``Mul`` of ``values`` in one call, refused when the numeric atoms are too large."""
    if sum(_numeric_bits(v) for v in values) > _MAX_POWER_BITS:
        raise ParseDomainError("product is too large")
    return _bounded(sp.Mul(*values))


def _power(base: sp.Expr, exponent: sp.Expr) -> sp.Expr:
    """``base ** exponent``; refused when a numeric base grows too large or a fractional
    power would factor a large integer."""
    if exponent.is_Rational and exponent.is_negative and base.is_zero:
        raise ParseDomainError("division by zero")
    bits = _numeric_bits(base)
    if bits:
        size = 1 + sum(abs(a.p) // a.q for a in exponent.atoms(sp.Rational))
        if bits * size > _MAX_POWER_BITS:
            raise ParseDomainError(f"power ({base})^({exponent}) is too large")
        if any(not a.is_Integer for a in exponent.atoms(sp.Rational)) and any(
            _bits(a) > _MAX_ROOT_BITS for a in base.atoms(sp.Rational)
        ):
            raise ParseDomainError(f"fractional power of an integer above {_MAX_ROOT_BITS} bits")
    return _bounded(base**exponent)


def _term_count(expr: sp.Expr) -> int:
    """An upper estimate of the term count of ``expand(expr)``, capped past :data:`_MAX_TERMS`."""
    cap = _MAX_TERMS + 1
    if expr.is_Add:
        return min(cap, sum(_term_count(a) for a in expr.args))
    if expr.is_Mul:
        total = 1
        for arg in expr.args:
            total = min(cap, total * _term_count(arg))
        return total
    if expr.is_Pow:
        base, exponent = expr.args
        inner = _term_count(exponent)
        t = _term_count(base)
        if t > 1 and exponent.is_Rational:
            k = abs(exponent.p) // exponent.q + (0 if exponent.is_Integer else 1)
            t = cap if k >= cap or t >= cap else min(cap, math.comb(t + k - 1, k))
        return max(t, inner)
    if expr.args:
        return max(_term_count(a) for a in expr.args)
    return 1


def _check_terms(expr: sp.Expr, role: str) -> sp.Expr:
    if _term_count(expr) > _MAX_TERMS:
        raise ParseDomainError(f"{role} expands to more than {_MAX_TERMS} terms")
    return expr


def _sympy_of(node: _Expr, resolve: Callable[[str], sp.Expr]) -> sp.Expr:
    """The sympy value of ``node``, names through ``resolve``."""
    if isinstance(node, _Num):
        return sp.Integer(node.value)
    if isinstance(node, _Lit):
        raise ParseGrammarError("'literal' marks only a dimension or count numeral")
    if isinstance(node, _Name):
        return resolve(node.name)
    if isinstance(node, _Const):
        return _CONSTANTS[node.name]
    if isinstance(node, _Neg):
        return -_sympy_of(node.arg, resolve)
    if isinstance(node, _Sum):
        if len(node.terms) > _MAX_TERMS:
            raise ParseDomainError(f"sum of more than {_MAX_TERMS} terms")
        values = [_sympy_of(term, resolve) for _, term in node.terms]
        return _bounded(
            sp.Add(*(-v if neg else v for (neg, _), v in zip(node.terms, values, strict=True)))
        )
    if isinstance(node, _Prod):
        factors = []
        for divided, factor in node.factors:
            value = _sympy_of(factor, resolve)
            if divided:
                if value.is_zero:
                    raise ParseDomainError("division by zero")
                value = sp.Pow(value, -1)
            factors.append(value)
        return _product(factors)
    if isinstance(node, _Pow):
        return _power(_sympy_of(node.base, resolve), _sympy_of(node.exponent, resolve))
    value = _sympy_of(node.arg, resolve)
    return sp.exp(value) if node.fn == "exp" else _power(value, sp.Rational(1, 2))


def _shape_error(node: _Expr, role: str) -> ParseGrammarError:
    return ParseGrammarError(
        f"{type(node).__name__.lstrip('_').lower()} is not allowed in a {role}"
    )


_Numeral: TypeAlias = Callable[[int, bool], Dim]


def _exponent_of(node: _Expr, mults: Collection[str] = ()) -> sp.Expr:
    """A dimension exponent: integers and names under ``+``, ``*`` and ``^``; a name in
    ``mults`` is that multiplicity symbol, any other an exponent symbol."""
    if isinstance(node, _Num):
        return sp.Integer(node.value)
    if isinstance(node, _Name | _Const):
        if node.name in mults:
            return Mult.symbol(node.name).to_sympy()
        Dim.symbol(node.name)
        return sp.Symbol(node.name, integer=True, nonnegative=True)
    if isinstance(node, _Sum) and not any(negated for negated, _ in node.terms):
        return _bounded(sp.Add(*(_exponent_of(term, mults) for _, term in node.terms)))
    if isinstance(node, _Prod) and not any(divided for divided, _ in node.factors):
        return _product([_exponent_of(factor, mults) for _, factor in node.factors])
    if isinstance(node, _Pow):
        return _power(_exponent_of(node.base, mults), _exponent_of(node.exponent, mults))
    raise _shape_error(node, "dimension exponent")


def _dim_of(node: _Expr, numeral: _Numeral, mults: Collection[str] = ()) -> Dim:
    """A dimension: numerals through ``numeral(value, literal)``, names as dimension symbols,
    ``*``, and ``^`` by an exponent."""
    if isinstance(node, _Num | _Lit):
        return numeral(node.value, isinstance(node, _Lit))
    if isinstance(node, _Name | _Const):
        return Dim.symbol(node.name)
    if isinstance(node, _Prod) and not any(divided for divided, _ in node.factors):
        factors = [_dim_of(factor, numeral, mults).to_sympy() for _, factor in node.factors]
        return Dim.from_sympy(_product(factors))
    if isinstance(node, _Pow):
        base = _dim_of(node.base, numeral, mults)
        exponent = _check_terms(_exponent_of(node.exponent, mults), "exponent")
        return Dim.from_sympy(_power(base.to_sympy(), exponent))
    raise _shape_error(node, "dimension")


def _concrete_dim(value: int, literal: bool) -> Dim:
    return Dim.concrete(value)


def _slot_dim(node: _Expr, numeral: _Numeral, mults: Collection[str] = ()) -> Dim:
    """A whole dimension slot: a bare numeral through ``numeral``; numerals inside a compound
    expression stay concrete."""
    if isinstance(node, _Num | _Lit):
        return numeral(node.value, isinstance(node, _Lit))
    return _dim_of(node, _concrete_dim, mults)


def _mult_expr(node: _Expr) -> sp.Expr:
    if isinstance(node, _Num):
        return sp.Integer(node.value)
    if isinstance(node, _Name | _Const):
        return Mult.symbol(node.name).to_sympy()
    if isinstance(node, _Sum) and not any(negated for negated, _ in node.terms):
        return _bounded(sp.Add(*(_mult_expr(term) for _, term in node.terms)))
    if isinstance(node, _Prod) and not any(divided for divided, _ in node.factors):
        return _product([_mult_expr(factor) for _, factor in node.factors])
    raise _shape_error(node, "multiplicity")


def _mult_of(node: _Expr) -> Mult:
    """A multiplicity: integers and names under ``+`` and ``*``."""
    expr = _mult_expr(node)
    try:
        return Mult._from_expr(expr)
    except BangBoxError as exc:
        raise ParseGrammarError(
            f"multiplicity {expr} is outside the multiplicity grammar: non-negative integers "
            "and names under '+' and '*', no name repeated as a factor (n*n folds to n**2)"
        ) from exc


def _mult_names(node: _Expr) -> set[str]:
    """The names a multiplicity AST uses, or none when it is not one."""
    try:
        return set(_mult_of(node).free_symbols)
    except ParseError:
        return set()


def _with_mults(dim: Dim, mults: Collection[str]) -> Dim:
    """``dim`` with each exponent symbol named in ``mults`` replaced by that multiplicity
    symbol."""
    expr = dim.to_sympy()
    swap = {
        s: Mult.symbol(s.name).to_sympy()
        for s in expr.free_symbols
        if s.name in mults and not s.is_positive
    }
    return Dim.from_sympy(expr.xreplace(swap)) if swap else dim


_ROLES: dict[str, Callable[[str], sp.Symbol]] = {
    "dimension": lambda name: Dim.symbol(name).to_sympy(),
    "multiplicity": lambda name: Mult.symbol(name).to_sympy(),
    "exponent": lambda name: sp.Symbol(name, integer=True, nonnegative=True),
    "phase": lambda name: Phase.symbol(name).to_sympy_turns(),
    "scalar": lambda name: Scalar.symbol(name).to_sympy(),
}
"""Symbol constructor per ``roles:`` keyword."""


def _role_table(
    dims: Iterable[Dim],
    mults: Iterable[Mult],
    phases: Iterable[Phase] = (),
    seeds: Mapping[str, sp.Symbol] | None = None,
) -> dict[str, sp.Symbol]:
    """Symbol per name, first role wins: ``seeds``, dimension, multiplicity, exponent,
    phase."""
    dim_symbols = [s for dim in dims for s in dim.to_sympy().free_symbols]
    groups = [
        [s for s in dim_symbols if s.is_positive],
        [s for mult in mults for s in mult.to_sympy().free_symbols],
        [s for s in dim_symbols if not s.is_positive],
        [s for phase in phases for s in phase.to_sympy_turns().free_symbols],
    ]
    table: dict[str, sp.Symbol] = dict(seeds or {})
    for group in groups:
        for symbol in sorted(group, key=lambda s: str(s.name)):
            table.setdefault(str(symbol.name), symbol)
    return table


def _check_finite(expr: sp.Expr, role: str) -> sp.Expr:
    if expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ParseDomainError(f"{role} {expr} is not finite")
    return expr


def _phase_expr(node: _Expr, table: Mapping[str, sp.Symbol]) -> sp.Expr:
    """Phase turns of ``node``; unknown names become :meth:`Phase.symbol`."""

    @functools.cache
    def resolve(name: str) -> sp.Expr:
        known = table.get(name)
        return known if known is not None else Phase.symbol(name).to_sympy_turns()

    return _check_finite(_sympy_of(node, resolve), "phase")


def _phase_value(expr: sp.Expr) -> Phase:
    _check_terms(expr, "phase")
    if expr.has(sp.I):
        raise ParseDomainError(f"phase {expr} is not real")
    return Phase(expr)


def _phase_of(node: _Expr, table: Mapping[str, sp.Symbol]) -> Phase:
    """A phase in turns; unknown names become :meth:`Phase.symbol`."""
    return _phase_value(_phase_expr(node, table))


def _scalar_expr(node: _Expr, table: Mapping[str, sp.Symbol]) -> sp.Expr:
    """The scalar value of ``node``; unknown names become :meth:`Scalar.symbol`."""

    @functools.cache
    def resolve(name: str) -> sp.Expr:
        known = table.get(name)
        return known if known is not None else Scalar.symbol(name).to_sympy()

    return _check_finite(_sympy_of(node, resolve), "scalar")


def _scalar_of(node: _Expr, table: Mapping[str, sp.Symbol]) -> Scalar:
    """A scalar; unknown names become :meth:`Scalar.symbol`."""
    return Scalar(_check_terms(_scalar_expr(node, table), "scalar"))


def _phase_entries(text: str, table: Mapping[str, sp.Symbol]) -> dict[int, Phase]:
    """``i: expr, ...`` as index -> Phase, without a ``k: rule`` item."""
    return _phase_vector_parts(text, table)[0]


def _phase_vector_parts(
    text: str, table: Mapping[str, sp.Symbol]
) -> tuple[dict[int, Phase], Phase | None]:
    """``i: expr, ..., k: rule`` as index -> Phase and the rule in :data:`INDEX`."""
    entries: dict[int, Phase] = {}
    rule: Phase | None = None
    for item in _split_top(text):
        index, colon, expr = item.partition(":")
        if not colon:
            raise ParseGrammarError(f"phase entry {item!r} is not 'INDEX: EXPR'")
        if index.strip() == "k":
            if rule is not None:
                raise ParseGrammarError("the phase rule k appears twice")
            turns = _phase_of(_parse_expr(expr, "phase"), {**table, "k": INDEX})
            rule = Phase(turns.to_sympy_turns().xreplace({sp.Symbol("k", real=True): INDEX}))
            continue
        i = _int_token(index, "phase index")
        if i in entries:
            raise ParseGrammarError(f"phase index {i} appears twice")
        entries[i] = _phase_of(_parse_expr(expr, "phase"), table)
    return entries, rule


def _carried(diagram: Diagram) -> frozenset[str]:
    """Every symbol name ``diagram`` carries."""
    names: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            names |= port.dim.free_symbols
        if node.phase is not None:
            names |= node.phase.free_symbols
    for box in diagram.bang_boxes.values():
        names |= box.multiplicity.free_symbols
    return frozenset(names)


def _generator(name: str) -> GeneratorType:
    if name not in REGISTRY.names():
        raise ParseGrammarError(f"unknown generator {name!r}; known: {sorted(REGISTRY.names())}")
    return REGISTRY.get(name)


@dataclass(frozen=True, slots=True)
class _NodeDraft:
    """A node ready to add, phase still as text."""

    generator: GeneratorType
    inputs: tuple[Dim, ...]
    outputs: tuple[Dim, ...]
    phase_dim: Dim | None = None
    phase_entries: str | None = None


def _place_nodes(diagram: Diagram, drafts: Mapping[int, _NodeDraft]) -> None:
    """Add ``drafts`` at exactly their ids, phases left unset."""
    for i, draft in sorted(drafts.items()):
        diagram.advance_ids(node_id=i)
        diagram.add_node(draft.generator, draft.inputs, draft.outputs)


@dataclass(frozen=True, slots=True)
class _BoxDraft:
    multiplicity: Mult
    node_scope: frozenset[NodeId] = frozenset()
    port_scope: frozenset[PortRef] = frozenset()
    parent: BangBoxId | None = None


def _place_boxes(diagram: Diagram, drafts: Mapping[int, _BoxDraft]) -> None:
    """Add ``drafts`` at exactly their ids."""
    for i, draft in sorted(drafts.items()):
        diagram.advance_ids(bang_box_id=i)
        diagram.add_bang_box(
            draft.multiplicity,
            node_scope=draft.node_scope,
            port_scope=draft.port_scope,
            parent=draft.parent,
        )


def _set_phases(
    diagram: Diagram, drafts: Mapping[int, _NodeDraft], table: Mapping[str, sp.Symbol]
) -> None:
    for i, draft in sorted(drafts.items()):
        if draft.phase_dim is not None:
            entries, rule = _phase_vector_parts(draft.phase_entries or "", table)
            diagram.set_phase(NodeId(i), PhaseVector(draft.phase_dim, entries, rule))


# -- form 1: graph listing -----------------------------------------------------------------

_LISTING_HEADER_RE = re.compile(
    r"graph:\s+(?P<nodes>[0-9]+) nodes?,\s+(?P<wires>[0-9]+) wires?,\s+"
    r"(?P<inputs>[0-9]+) inputs?\s+->\s+(?P<outputs>[0-9]+) outputs?"
)
_LISTING_REF_RE = re.compile(r"n(?P<node>[0-9]+)\.(?P<side>out|in)(?P<index>[0-9]+)")
_LISTING_NODE_RE = re.compile(
    r"n(?P<id>[0-9]+)\s+(?P<gen>\S+)\s+in:\s*\[(?P<ins>[^\]]*)\]\s+out:\s*\[(?P<outs>[^\]]*)\]"
    r"(?:\s+phase\[(?P<pdim>[^\]]*)\]:\s*(?:(?P<pzero>0)|\{(?P<pentries>[^{}]*)\}\s*turns))?"
    r"(?:\s+boxes:\s*(?P<boxes>!.*))?"
)
_LISTING_BOX_RE = re.compile(
    r"!(?P<id>[0-9]+)\s+x\s+(?P<mult>.+?)\s+(?:nodes:\s*(?P<nodes>.+?)|ports:\s*(?P<ports>.+?))"
    r"(?:\s+parent:\s*!(?P<parent>[0-9]+)\s+\((?P<flag>missing|cycle)\))?"
)
_UNICODE_IDENTIFIER = r"[^\W\d]\w*"
_LISTING_PARAM_RE = re.compile(rf"(?P<name>{_UNICODE_IDENTIFIER})\s*=\s*(?P<value>-?[0-9]+)")
_LISTING_ROLE_RE = re.compile(rf"(?P<name>{_UNICODE_IDENTIFIER})\s*=\s*(?P<role>[a-z]+)")
_MARK_RE = re.compile(r"!([0-9]+)")


def _signed(token: str) -> int:
    return -_expr_numeral(token[1:]) if token.startswith("-") else _expr_numeral(token)


def _listing_ref(text: str) -> PortRef:
    match = _LISTING_REF_RE.fullmatch(text.strip())
    if match is None:
        raise ParseGrammarError(f"{text!r} is not a port reference 'nI.outJ' or 'nI.inJ'")
    direction = Direction.OUTPUT if match["side"] == "out" else Direction.INPUT
    return PortRef(
        NodeId(_numeral(match["node"], "node id")),
        direction,
        _numeral(match["index"], "port index"),
    )


def _listing_refs(text: str) -> list[PortRef]:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ParseGrammarError(f"{text!r} is not a bracketed port list")
    return [_listing_ref(item) for item in _split_top(text[1:-1])]


def _marks(text: str) -> list[int]:
    """``!a > !b`` or ``!a!b`` as box ids."""
    found = _MARK_RE.findall(text)
    if re.sub(r"[\s>]", "", text) != "".join(f"!{m}" for m in found):
        raise ParseGrammarError(f"{text!r} is not a list of box marks")
    return [_numeral(m, "box id") for m in found]


class _Cursor:
    """Sequential reader over non-blank listing lines."""

    def __init__(self, lines: Sequence[str]) -> None:
        self.lines = lines
        self.i = 1

    def has(self, label: str) -> bool:
        """Whether the next line opens with ``label:``."""
        return self.i < len(self.lines) and self.lines[self.i].strip().startswith(label + ":")

    def field(self, label: str) -> str:
        """The text after ``label:`` on the next line."""
        if self.i >= len(self.lines) or not self.lines[self.i].strip().startswith(label + ":"):
            found = self.lines[self.i].strip() if self.i < len(self.lines) else "end of input"
            raise ParseGrammarError(f"listing: expected '{label}:', found {found!r}")
        self.i += 1
        return self.lines[self.i - 1].strip()[len(label) + 1 :].strip()

    def block(self, label: str, stop: str | None) -> list[str]:
        """The item lines under ``label:`` (or none after ``label: none``)."""
        rest = self.field(label)
        if rest == "none":
            return []
        if rest:
            raise ParseGrammarError(f"listing: unexpected {rest!r} after '{label}:'")
        items = []
        while self.i < len(self.lines) and (
            stop is None or not self.lines[self.i].strip().startswith(stop + ":")
        ):
            items.append(self.lines[self.i])
            self.i += 1
        return items


def _listing_node(
    line: str,
    port_marks: dict[PortRef, list[int]],
    node_chains: dict[int, list[int]],
    mults: Collection[str],
) -> tuple[int, _NodeDraft]:
    match = _LISTING_NODE_RE.fullmatch(line.strip())
    if match is None:
        raise ParseGrammarError(f"listing: {line.strip()!r} is not a node line")
    node_id = _numeral(match["id"], "node id")
    sides: list[tuple[Dim, ...]] = []
    for group, direction in (("ins", Direction.INPUT), ("outs", Direction.OUTPUT)):
        dims = []
        for index, item in enumerate(_split_top(match[group])):
            text, bang, marks = item.partition("!")
            dims.append(_dim_of(_parse_expr(text, "dimension"), _concrete_dim, mults))
            if bang:
                port_marks[PortRef(NodeId(node_id), direction, index)] = _marks(bang + marks)
        sides.append(tuple(dims))
    if match["boxes"] is not None:
        node_chains[node_id] = _marks(match["boxes"])
    phase_dim = None
    if match["pdim"] is not None:
        phase_dim = _dim_of(_parse_expr(match["pdim"], "dimension"), _concrete_dim, mults)
    draft = _NodeDraft(
        _generator(match["gen"]), sides[0], sides[1], phase_dim, match["pentries"] or ""
    )
    return node_id, draft


def _listing_boxes(lines: Sequence[str]) -> tuple[dict[int, _BoxDraft], dict[int, int]]:
    """Box drafts by id, and each box's depth, parents read off indentation."""
    drafts: dict[int, _BoxDraft] = {}
    depth: dict[int, int] = {}
    stack: list[tuple[int, int]] = []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        match = _LISTING_BOX_RE.fullmatch(line.strip())
        if match is None:
            raise ParseGrammarError(f"listing: {line.strip()!r} is not a bang-box line")
        box_id = _numeral(match["id"], "box id")
        if box_id in drafts:
            raise ParseGrammarError(f"listing: box !{box_id} appears twice")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else None
        if match["parent"] is not None:
            if parent is not None:
                raise ParseGrammarError(f"listing: nested box !{box_id} names a parent")
            parent = _numeral(match["parent"], "box id")
        depth[box_id] = len(stack)
        stack.append((indent, box_id))
        mult_node = _parse_expr(match["mult"], "multiplicity")
        mult = Mult(mult_node.value) if isinstance(mult_node, _Lit) else _mult_of(mult_node)
        if match["nodes"] is not None:
            nodes = []
            for item in _split_top(match["nodes"]):
                if not re.fullmatch(r"n[0-9]+", item):
                    raise ParseGrammarError(f"listing: {item!r} is not a node id")
                nodes.append(NodeId(_numeral(item[1:], "node id")))
            draft = _BoxDraft(mult, node_scope=frozenset(nodes))
        else:
            refs = [_listing_ref(item) for item in _split_top(match["ports"])]
            draft = _BoxDraft(mult, port_scope=frozenset(refs))
        drafts[box_id] = _BoxDraft(
            draft.multiplicity,
            draft.node_scope,
            draft.port_scope,
            None if parent is None else BangBoxId(parent),
        )
    return drafts, depth


def _check_marks(
    boxes: Mapping[int, _BoxDraft],
    depth: Mapping[int, int],
    port_marks: Mapping[PortRef, list[int]],
    node_chains: Mapping[int, list[int]],
) -> None:
    """Port marks and node chains must equal the box lines' scopes, in (depth, id) order."""
    want_ports: dict[PortRef, list[int]] = {}
    want_nodes: dict[int, list[int]] = {}
    for i in sorted(boxes, key=lambda i: (depth[i], i)):
        for ref in boxes[i].port_scope:
            want_ports.setdefault(ref, []).append(i)
        for nid in boxes[i].node_scope:
            want_nodes.setdefault(int(nid), []).append(i)
    if dict(port_marks) != want_ports:
        raise ParseGrammarError("listing: port '!' marks disagree with the bang-box lines")
    if dict(node_chains) != want_nodes:
        raise ParseGrammarError("listing: node 'boxes:' chains disagree with the bang-box lines")


@_guarded
def parse_listing(source: str) -> Diagram:
    """The diagram a :func:`~archytaszx.repl.printer.render_graph` listing shows, ids kept."""
    lines = [line.rstrip() for line in _require_text(source).splitlines() if line.strip()]
    header = _LISTING_HEADER_RE.fullmatch(lines[0].strip()) if lines else None
    if header is None:
        raise ParseGrammarError(
            "listing: expected 'graph: N nodes, W wires, I inputs -> O outputs'"
        )
    cursor = _Cursor(lines)
    scalar_text = cursor.field("scalar")
    parameters: dict[str, int] = {}
    parameter_text = cursor.field("parameters")
    if parameter_text != "none":
        for item in _split_top(parameter_text):
            match = _LISTING_PARAM_RE.fullmatch(item)
            if match is None or match["name"] in parameters or not match["name"].isidentifier():
                raise ParseGrammarError(f"listing: bad parameter entry {item!r}")
            parameters[match["name"]] = _signed(match["value"])
    seeds: dict[str, sp.Symbol] = {}
    if cursor.has("roles"):
        for item in _split_top(cursor.field("roles")):
            match = _LISTING_ROLE_RE.fullmatch(item)
            if (
                match is None
                or match["role"] not in _ROLES
                or match["name"] in seeds
                or not match["name"].isidentifier()
            ):
                raise ParseGrammarError(f"listing: bad role entry {item!r}")
            seeds[match["name"]] = _ROLES[match["role"]](match["name"])
    node_lines = cursor.block("nodes", "wires")
    wires = []
    for line in cursor.block("wires", "inputs"):
        left, sep, right = line.strip().partition(" -- ")
        if not sep:
            raise ParseGrammarError(f"listing: {line.strip()!r} is not a wire line")
        wires.append((_listing_ref(left), _listing_ref(right)))
    inputs = _listing_refs(cursor.field("inputs"))
    outputs = _listing_refs(cursor.field("outputs"))
    boxes, depth = _listing_boxes(cursor.block("bang boxes", None))
    if cursor.i < len(lines):
        raise ParseGrammarError(f"listing: unexpected {lines[cursor.i].strip()!r} at the end")
    mults = {name for box in boxes.values() for name in box.multiplicity.free_symbols}
    port_marks: dict[PortRef, list[int]] = {}
    node_chains: dict[int, list[int]] = {}
    drafts: dict[int, _NodeDraft] = {}
    for line in node_lines:
        node_id, draft = _listing_node(line, port_marks, node_chains, mults)
        if node_id in drafts:
            raise ParseGrammarError(f"listing: node n{node_id} appears twice")
        drafts[node_id] = draft
    _check_marks(boxes, depth, port_marks, node_chains)

    diagram = Diagram()
    _place_nodes(diagram, drafts)
    for a, b in wires:
        diagram.add_wire(a, b)
    diagram.set_boundary_inputs(inputs)
    diagram.set_boundary_outputs(outputs)
    _place_boxes(diagram, boxes)
    diagram.set_parameters(parameters)
    counts = (len(drafts), len(diagram.wires), len(inputs), len(outputs))
    if len(wires) != len(diagram.wires) or counts != tuple(
        int(header[k]) for k in ("nodes", "wires", "inputs", "outputs")
    ):
        raise ParseGrammarError("listing: the header counts disagree with the body")
    box_mults = [b.multiplicity for b in boxes.values()]
    _set_phases(diagram, drafts, _role_table(_draft_dims(drafts), box_mults, seeds=seeds))
    table = _role_table(_draft_dims(drafts), box_mults, _phases(diagram), seeds)
    diagram.set_scalar(_scalar_of(_parse_expr(scalar_text, "scalar"), table))
    return _finish(diagram)


def _draft_dims(drafts: Mapping[int, _NodeDraft]) -> list[Dim]:
    dims: list[Dim] = []
    for draft in drafts.values():
        dims.extend((*draft.inputs, *draft.outputs))
        if draft.phase_dim is not None:
            dims.append(draft.phase_dim)
    return dims


def _phases(diagram: Diagram) -> list[Phase]:
    return [
        phase
        for node in diagram.nodes.values()
        if node.phase is not None
        for phase in node.phase.entries().values()
    ]


# -- form 2: DSL ---------------------------------------------------------------------------

_DSL_KEYWORDS = frozenset({"dim", "count", "let", "wire", "inputs", "outputs", "scalar", "box"})
_RESERVED_SYMBOLS = _DSL_KEYWORDS | {_LITERAL_MARKER, "apply"}
_RESERVED_NAMES = _RESERVED_SYMBOLS | set(_CONSTANTS) | _FUNCTIONS
_DSL_START_RE = re.compile(rf"(?:{'|'.join(sorted(_DSL_KEYWORDS))})\b")
_DSL_NODE_RE = re.compile(rf"(?P<name>{_IDENTIFIER})\s*=\s*(?P<gen>{_IDENTIFIER})\s*\(")
_DSL_REF_RE = re.compile(rf"(?P<node>{_IDENTIFIER})\.(?P<side>out|in)(?P<index>[0-9]+)")
_DSL_PHASE_RE = re.compile(
    r"phase\s*(?:\[(?P<dim>[^\]]*)\])?\s*(?:(?P<zero>0)|\{(?P<entries>[^{}]*)\})"
)
_DSL_EXPLICIT_RE = re.compile(r"in:\s*\[(?P<ins>[^\]]*)\]\s*,\s*out:\s*\[(?P<outs>[^\]]*)\]")
_DSL_BOX_RE = re.compile(
    rf"box\s+(?P<name>{_IDENTIFIER})\s+x\s+(?P<mult>.+?)\s+(?P<kind>nodes|ports)\s+"
    rf"(?P<scope>.+?)(?:\s+in\s+(?P<parent>{_IDENTIFIER}))?"
)
_DSL_BINDING_RE = re.compile(rf"(?P<name>{_UNICODE_IDENTIFIER})(?:\s*=\s*(?P<value>-?[0-9]+))?")
_ONE_TO_ONE = frozenset({"F", "T", "Ti"})


def _statements(source: str) -> list[str]:
    """Non-empty statements: ``#`` comments cut, split by newline and ``;``."""
    statements: list[str] = []
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        statements.extend(part.strip() for part in code.split(";") if part.strip())
    return statements


def _keyword(text: str) -> tuple[str, str]:
    """The first word of a statement and the rest, split at any whitespace."""
    parts = text.split(None, 1)
    return parts[0], parts[1].strip() if len(parts) > 1 else ""


def _closing_paren(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ParseGrammarError(f"unbalanced '(' in {text!r}")


def _name(name: str, what: str) -> str:
    reserved = _RESERVED_SYMBOLS if what == "symbol" else _RESERVED_NAMES
    if name in reserved or not name.isidentifier():
        raise ParseGrammarError(f"{what} name {name!r} is a keyword or not an identifier")
    return name


@dataclass(frozen=True, slots=True)
class _DslBox:
    multiplicity: Mult
    kind: str
    scope: str
    parent: str | None


@dataclass
class _Dsl:
    """One DSL parse: statements read in order, references resolved by :meth:`build`."""

    abstraction: _Abstraction
    mults: frozenset[str] = frozenset()
    nodes: dict[str, _NodeDraft] = field(default_factory=dict)
    wires: list[tuple[str, str]] = field(default_factory=list)
    boundary: dict[str, list[str]] = field(default_factory=dict)
    scalar: str | None = None
    boxes: dict[str, _DslBox] = field(default_factory=dict)
    declared_dims: set[str] = field(default_factory=set)
    declared_counts: set[str] = field(default_factory=set)
    bindings: dict[str, int] = field(default_factory=dict)

    def dim(self, text: str) -> Dim:
        return _slot_dim(_parse_expr(text, "dimension"), self.abstraction.dim, self.mults)

    def statement(self, text: str) -> None:
        node = _DSL_NODE_RE.match(text)
        if node is not None:
            self.node(text, node)
            return
        keyword, rest = _keyword(text)
        if keyword in ("dim", "count", "let"):
            self.binding(keyword, rest)
        elif keyword == "wire":
            for item in _split_top(rest):
                left, sep, right = item.partition("--")
                if not sep:
                    raise ParseGrammarError(f"wire {item!r} is not 'a.outI -- b.inJ'")
                self.wires.append((left.strip(), right.strip()))
        elif keyword in ("inputs", "outputs"):
            if keyword in self.boundary:
                raise ParseGrammarError(f"'{keyword}' appears twice")
            self.boundary[keyword] = _split_top(rest)
        elif keyword == "scalar":
            if self.scalar is not None:
                raise ParseGrammarError("'scalar' appears twice")
            self.scalar = rest
        elif keyword == "box":
            self.box(text)
        else:
            raise ParseGrammarError(f"unknown DSL statement {text!r}")

    def node(self, text: str, match: re.Match[str]) -> None:
        name = _name(match["name"], "node")
        if name in self.nodes:
            raise ParseGrammarError(f"node {name!r} is declared twice")
        generator = _generator(match["gen"])
        close = _closing_paren(text, match.end() - 1)
        inputs, outputs, default = self.legs(generator, text[match.end() : close])
        rest = text[close + 1 :].strip()
        phase_dim = None
        entries = None
        if rest:
            phase = _DSL_PHASE_RE.fullmatch(rest)
            if phase is None:
                raise ParseGrammarError(f"node {name!r}: {rest!r} is not a phase suffix")
            if phase["dim"] is not None:
                phase_dim = self.dim(phase["dim"])
            elif default is not None or inputs or outputs:
                phase_dim = default if default is not None else (*inputs, *outputs)[0]
            else:
                raise ParseDomainError(f"node {name!r} has no legs; its phase needs 'phase[D]'")
            entries = phase["entries"] or ""
        self.nodes[name] = _NodeDraft(generator, inputs, outputs, phase_dim, entries)

    def legs(
        self, generator: GeneratorType, args: str
    ) -> tuple[tuple[Dim, ...], tuple[Dim, ...], Dim | None]:
        """Input dims, output dims, and the shorthand dim if one was given."""
        parts = _split_top(args)
        if parts and "->" in parts[0]:
            if len(parts) != 2:
                raise ParseGrammarError(f"{args!r} is not 'm -> n, D'")
            ins_text, _, outs_text = parts[0].partition("->")
            counts = (_int_token(ins_text, "input count"), _int_token(outs_text, "output count"))
            if max(counts) > _MAX_KET_LEG_COUNT:
                raise ParseDomainError(f"leg count above the bound of {_MAX_KET_LEG_COUNT}")
            dim = self.dim(parts[1])
            return (dim,) * counts[0], (dim,) * counts[1], dim
        explicit = _DSL_EXPLICIT_RE.fullmatch(args.strip())
        if explicit is not None:
            inputs = tuple(self.dim(t) for t in _split_top(explicit["ins"]))
            outputs = tuple(self.dim(t) for t in _split_top(explicit["outs"]))
            return inputs, outputs, None
        if generator.name in _ONE_TO_ONE and len(parts) == 1:
            dim = self.dim(parts[0])
            return (dim,), (dim,), dim
        if generator.name in ("B", "S") and len(parts) == 2:
            s, t = self.dim(parts[0]), self.dim(parts[1])
            if generator.name == "B":
                return (s, t), (s * t,), None
            return (s * t,), (s, t), None
        raise ParseGrammarError(f"{generator.name}({args}) matches no node form")

    def binding(self, keyword: str, rest: str) -> None:
        items = _split_top(rest)
        if not items:
            raise ParseGrammarError(f"'{keyword}' names nothing")
        for item in items:
            match = _DSL_BINDING_RE.fullmatch(item)
            if match is None:
                raise ParseGrammarError(f"{keyword} {item!r} is not 'NAME' or 'NAME = INT'")
            name = _name(match["name"], "symbol")
            if keyword == "dim":
                self.declared_dims.add(name)
            elif keyword == "count":
                self.declared_counts.add(name)
            if match["value"] is None:
                if keyword == "let":
                    raise ParseGrammarError(f"let {name!r} needs '= INT'")
                continue
            if name in self.bindings:
                raise ParseGrammarError(f"{name!r} is bound twice")
            self.bindings[name] = _signed(match["value"])

    def box(self, text: str) -> None:
        match = _DSL_BOX_RE.fullmatch(text)
        if match is None:
            raise ParseGrammarError(f"{text!r} is not 'box NAME x MULT nodes|ports ... [in P]'")
        name = _name(match["name"], "box")
        if name in self.boxes:
            raise ParseGrammarError(f"box {name!r} is declared twice")
        node = _parse_expr(match["mult"], "multiplicity")
        if isinstance(node, _Num):
            mult = self.abstraction.count(node.value)
        elif isinstance(node, _Lit):
            mult = Mult(node.value)
        else:
            mult = _mult_of(node)
        self.boxes[name] = _DslBox(mult, match["kind"], match["scope"], match["parent"])

    def ref(self, text: str, ids: Mapping[str, NodeId]) -> PortRef:
        match = _DSL_REF_RE.fullmatch(text.strip())
        if match is None:
            raise ParseGrammarError(f"{text!r} is not a port 'NAME.outI' or 'NAME.inJ'")
        if match["node"] not in ids:
            raise ParseGrammarError(f"unknown node {match['node']!r}")
        direction = Direction.OUTPUT if match["side"] == "out" else Direction.INPUT
        return PortRef(ids[match["node"]], direction, _numeral(match["index"], "port index"))

    def build(self) -> Diagram:
        diagram = Diagram()
        drafts = dict(enumerate(self.nodes.values()))
        _place_nodes(diagram, drafts)
        ids = {name: NodeId(i) for i, name in enumerate(self.nodes)}
        used: set[PortRef] = set()
        for left, right in self.wires:
            a, b = self.ref(left, ids), self.ref(right, ids)
            diagram.add_wire(a, b)
            used |= {a, b}
        sides = {}
        for keyword in ("inputs", "outputs"):
            sides[keyword] = [self.ref(text, ids) for text in self.boundary.get(keyword, [])]
            used.update(sides[keyword])
        for keyword, direction in (("inputs", Direction.INPUT), ("outputs", Direction.OUTPUT)):
            if keyword not in self.boundary:
                sides[keyword] = [
                    ref
                    for i, draft in drafts.items()
                    for ref in (
                        PortRef(NodeId(i), direction, j)
                        for j in range(
                            len(draft.inputs if direction is Direction.INPUT else draft.outputs)
                        )
                    )
                    if ref not in used
                ]
        diagram.set_boundary_inputs(sides["inputs"])
        diagram.set_boundary_outputs(sides["outputs"])
        box_ids = {name: BangBoxId(i) for i, name in enumerate(self.boxes)}
        for box in self.boxes.values():
            if box.parent is not None and box.parent not in box_ids:
                raise ParseGrammarError(f"unknown parent box {box.parent!r}")
            parent = None if box.parent is None else box_ids[box.parent]
            items = _split_top(box.scope)
            if box.kind == "nodes":
                unknown = [item for item in items if item not in ids]
                if unknown:
                    raise ParseGrammarError(f"unknown node(s) {unknown} in a box scope")
                scope = frozenset(ids[item] for item in items)
                diagram.add_bang_box(box.multiplicity, node_scope=scope, parent=parent)
            else:
                ports = frozenset(self.ref(item, ids) for item in items)
                diagram.add_bang_box(box.multiplicity, port_scope=ports, parent=parent)
        dims = [*_draft_dims(drafts), *(Dim.symbol(n) for n in sorted(self.declared_dims))]
        mults = [
            *(box.multiplicity for box in self.boxes.values()),
            *(Mult.symbol(n) for n in sorted(self.declared_counts)),
        ]
        _set_phases(diagram, drafts, _role_table(dims, mults))
        if self.scalar is not None:
            table = _role_table(dims, mults, _phases(diagram))
            diagram.set_scalar(_scalar_of(_parse_expr(self.scalar, "scalar"), table))
        uncarried = sorted(set(self.bindings) - _carried(diagram))
        if uncarried:
            raise ParseDomainError(f"binding(s) {uncarried} name no symbol the diagram carries")
        diagram.set_parameters({**self.abstraction.bindings, **self.bindings})
        return diagram


def _dsl_prepass(statements: Sequence[str]) -> tuple[dict[int, str], frozenset[str]]:
    """Values bound by ``dim``/``let`` (value -> first name), and every count or box
    multiplicity name."""
    reuse: dict[int, str] = {}
    mults: set[str] = set()
    for text in statements:
        if _DSL_NODE_RE.match(text):
            continue
        keyword, rest = _keyword(text)
        if keyword in ("dim", "let", "count"):
            for item in _split_top(rest):
                match = _DSL_BINDING_RE.fullmatch(item)
                if match is None:
                    continue
                if keyword == "count":
                    mults.add(match["name"])
                elif match["value"] is not None:
                    reuse.setdefault(_signed(match["value"]), match["name"])
        elif keyword == "box" and (box := _DSL_BOX_RE.fullmatch(text)) is not None:
            mults |= _mult_names(_parse_expr(box["mult"], "multiplicity"))
    return reuse, frozenset(mults)


def _check_phase_range(diagram: Diagram) -> Diagram:
    """``diagram``, once every phase index is below its dimension where the parameters make
    that dimension concrete."""
    for node_id, node in sorted(diagram.nodes.items()):
        if node.phase is None or not node.phase.entries():
            continue
        value = _resolved(node.phase.dim, diagram.parameters)
        top = max(node.phase.entries())
        if value.is_Integer and top > int(value) - 1:
            raise ParseDomainError(
                f"phase index {top} of n{int(node_id)} is outside dimension {value}"
            )
    return diagram


@_guarded
def parse_dsl(source: str) -> Diagram:
    """The diagram a DSL program declares; node and box ids follow declaration order."""
    statements = _statements(_require_text(source))
    if not statements:
        raise ParseGrammarError("empty DSL program")
    reuse, mults = _dsl_prepass(statements)
    dsl = _Dsl(_Abstraction(source, reuse), mults)
    for text in statements:
        dsl.statement(text)
    return _finish(_check_phase_range(dsl.build()))


# -- form 3: Dirac -------------------------------------------------------------------------

_DIRAC_WORDS = frozenset({"plus", "id", "copy", "zero", "F", "triangle", "bind", "split"})
_DIRAC_RESERVED = _DIRAC_WORDS | {"GHZ", "Z", "X", "W", "Z_phase"}
_DIRAC_ATOM_RE = re.compile(r"(?P<sub>GHZ|copy|W|Z|X)_\{|(?P<word>[A-Za-z_]\w*)")
_DIRAC_SUM_RE = re.compile(r"(?:sum|Σ)_\{")
_DIRAC_NAMES_RE = re.compile(rf"\s*(?P<names>{_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})*)\s*=\s*0\s*")
_DIRAC_LEGACY_BOUND_RE = re.compile(
    rf"(?:(?P<literal>{_LITERAL_MARKER})\s+)?(?P<dim>{_IDENTIFIER}|{_ASCII_DIGITS})-1"
)
_DIRAC_DELTA_RE = re.compile(rf"\s*(?P<a>{_IDENTIFIER})\s*=\s*(?P<b>{_IDENTIFIER})\s*")
_DIRAC_PHASE_RE = re.compile(r"\s*2\s*(?:pi|π)\s*i\s*\((?P<body>.*)\)\s*", re.DOTALL)
_DIRAC_AT_RE = re.compile(rf"\s*(?P<index>{_IDENTIFIER})\s*=\s*(?P<value>[0-9]+)\s*")
_DIRAC_ONE_RE = re.compile(r"1(?!\w)")
_DIRAC_PAIRS = {")": "(", "]": "[", "}": "{"}


class _Var:
    """A dimension unification variable: union-find parent, optional value, owning atom."""

    __slots__ = ("label", "parent", "value")

    def __init__(self, label: str, value: Dim | None = None) -> None:
        self.label = label
        self.parent: _Var | None = None
        self.value = value

    def root(self) -> _Var:
        """The representative of this variable's class."""
        var = self
        while var.parent is not None:
            var = var.parent
        return var


_DimTerm: TypeAlias = tuple[_Var, ...]
"""A leg dimension: the product of its variables."""


class _Leg:
    """One port of a :class:`_DiracNode`: plain, a pending concrete fan, or a symbolic fan."""

    __slots__ = ("count", "dim", "mult", "owner")

    def __init__(
        self, owner: _DiracNode, dim: _DimTerm, count: int | None = None, mult: Mult | None = None
    ) -> None:
        self.owner = owner
        self.dim = dim
        self.count = count
        self.mult = mult


class _DiracNode:
    """A node under construction; ``phase`` maps a value to its summed turn ASTs."""

    __slots__ = ("generator", "inputs", "label", "outputs", "phase", "var")

    def __init__(self, generator: GeneratorType, label: str, var: _Var | None) -> None:
        self.generator = generator
        self.label = label
        self.var = var
        self.inputs: list[_Leg] = []
        self.outputs: list[_Leg] = []
        self.phase: dict[int, list[_Expr]] | None = None


@dataclass
class _Piece:
    """A parsed sub-expression: nodes in creation order, wires, open legs and scalar ASTs."""

    nodes: list[_DiracNode] = field(default_factory=list)
    wires: list[tuple[_Leg, _Leg]] = field(default_factory=list)
    inputs: list[_Leg] = field(default_factory=list)
    outputs: list[_Leg] = field(default_factory=list)
    scalars: list[_Expr] = field(default_factory=list)

    def absorb(self, other: _Piece) -> _Piece:
        """``self`` extended in place by ``other`` (tensor product)."""
        self.nodes.extend(other.nodes)
        self.wires.extend(other.wires)
        self.inputs.extend(other.inputs)
        self.outputs.extend(other.outputs)
        self.scalars.extend(other.scalars)
        return self


_Count: TypeAlias = tuple[int | None, Mult | None]
"""A leg template: ``(None, None)`` plain, ``(c, None)`` pending fan, ``(None, m)`` box."""


def _bracket_pairs(text: str) -> dict[int, int]:
    """Open-bracket position -> matching close position over ``()[]{}``."""
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch in "([{":
            stack.append(i)
        elif ch in _DIRAC_PAIRS:
            if not stack or text[stack[-1]] != _DIRAC_PAIRS[ch]:
                raise ParseGrammarError(f"unbalanced {ch!r} at column {i + 1}")
            pairs[stack.pop()] = i
    if stack:
        raise ParseGrammarError(f"unclosed {text[stack[-1]]!r} at column {stack[-1] + 1}")
    return pairs


def _strip_name(text: str) -> str:
    """``text`` after its first top-level `` = `` when the left side is a chain of catalog
    atoms, else all of ``text``."""
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(" = ", i - 1) and i > 0:
            if not _is_atom_chain(text[: i - 1]):
                raise ParseGrammarError(
                    f"{text[: i - 1].strip()!r} before ' = ' is not a catalog name"
                )
            return text[i + 2 :]
    return text


def _is_atom_chain(text: str) -> bool:
    """Whether ``text`` is catalog atoms joined by ``(x)`` or ``⊗``, syntax only."""
    try:
        pairs = _bracket_pairs(text)
    except ParseGrammarError:
        return False
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        match = _DIRAC_ATOM_RE.match(text, pos)
        if match is None:
            return False
        pos = match.end()
        if match["sub"] is not None:
            pos = pairs[pos - 1] + 1
            if (
                match["sub"] in ("Z", "X")
                and text.startswith("(", pos)
                and not text.startswith("(x)", pos)
            ):
                pos = pairs[pos] + 1
        elif match["word"] not in _DIRAC_WORDS | {"Z_phase"}:
            return False
        for suffix in ("^dagger", "^-1"):
            if text.startswith(suffix, pos):
                pos += len(suffix)
                break
        if text.startswith("[", pos):
            pos = pairs[pos] + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            return True
        op = next((op for op in ("(x)", "⊗") if text.startswith(op, pos)), None)
        if op is None:
            return False
        pos += len(op)


def _names(node: _Expr) -> Iterable[str]:
    """Every identifier in ``node``."""
    if isinstance(node, _Name):
        yield node.name
    elif isinstance(node, _Neg | _Call):
        yield from _names(node.arg)
    elif isinstance(node, _Sum):
        for _, term in node.terms:
            yield from _names(term)
    elif isinstance(node, _Prod):
        for _, factor in node.factors:
            yield from _names(factor)
    elif isinstance(node, _Pow):
        yield from _names(node.base)
        yield from _names(node.exponent)


def _scalar_only(text: str) -> _Expr | None:
    """The AST of ``text`` when all of it is one scalar expression naming no catalog atom."""
    parser = _ExprParser(text)
    try:
        if parser.peek()[0] == "end":
            return None
        node = parser.expression()
    except ParseGrammarError:
        return None
    if parser.peek()[0] != "end" or any(n in _DIRAC_RESERVED for n in _names(node)):
        return None
    return node


class _Dirac:
    """Recursive descent over a Dirac expression into one :class:`_Piece`, then a diagram."""

    def __init__(self, text: str, abstraction: _Abstraction) -> None:
        self.text = text
        self.pos = 0
        self.pairs = _bracket_pairs(text)
        self.depth = 0
        self.abstraction = abstraction
        self.equations: list[tuple[_DimTerm, _DimTerm]] = []
        self.expanded = 0

    # -- scanning

    def ws(self) -> int:
        """Skip whitespace; the new position."""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        return self.pos

    def error(self, message: str) -> ParseGrammarError:
        found = repr(self.text[self.pos : self.pos + 12]) if self.pos < len(self.text) else "end"
        return ParseGrammarError(f"{message} at column {self.pos + 1}, found {found}")

    def braced(self, opener: str = "{") -> str:
        """The text inside the bracket at :attr:`pos`, which moves past its close."""
        if not self.text.startswith(opener, self.pos):
            raise self.error(f"expected {opener!r}")
        close = self.pairs[self.pos]
        inner = self.text[self.pos + 1 : close]
        self.pos = close + 1
        return inner

    def tensor_op(self) -> bool:
        """Consume ``(x)`` or ``⊗`` after whitespace."""
        start = self.pos
        self.ws()
        for op in ("(x)", "⊗"):
            if self.text.startswith(op, self.pos):
                self.pos += len(op)
                return True
        self.pos = start
        return False

    # -- grammar

    def read(self) -> Diagram:
        """The diagram of the whole text."""
        piece = self.feed()
        if self.ws() < len(self.text):
            raise self.error("unexpected text")
        return self.build(piece)

    def feed(self) -> _Piece:
        """``tensor (';' tensor)*``."""
        piece = self.tensor()
        while self.ws() < len(self.text) and self.text[self.pos] == ";":
            self.pos += 1
            piece = self.compose(piece, self.tensor())
        return piece

    def tensor(self) -> _Piece:
        """``term (('(x)' | '⊗') term)*``."""
        piece = self.term()
        while self.tensor_op():
            piece = piece.absorb(self.term())
        return piece

    def term(self) -> _Piece:
        """``['(' scalar ')'] primary``: a parenthesis followed by a term is a scalar."""
        start = self.ws()
        if start < len(self.text) and self.text[start] == "(":
            close = self.pairs[start]
            self.pos = close + 1
            after = self.ws()
            rest = self.text[after:]
            if rest and rest[0] not in ";)" and not rest.startswith(("(x)", "⊗")):
                scalar = _parse_expr(self.text[start + 1 : close], "scalar")
                reserved = sorted(set(_names(scalar)) & _DIRAC_RESERVED)
                if reserved:
                    raise ParseGrammarError(f"catalog name {reserved[0]!r} is not a scalar")
                piece = self.primary()
                piece.scalars.insert(0, scalar)
                return piece
            self.pos = start
        return self.primary()

    def primary(self) -> _Piece:
        """``ketsum | atom | '(' feed ')'``."""
        start = self.ws()
        if start >= len(self.text):
            raise self.error("expected a term")
        if self.text[start] == "(":
            self.depth += 1
            if self.depth > _MAX_EXPR_DEPTH:
                raise ParseGrammarError(f"groups nest deeper than {_MAX_EXPR_DEPTH} levels")
            self.pos += 1
            piece = self.feed()
            if self.ws() >= len(self.text) or self.text[self.pos] != ")":
                raise self.error("expected ')'")
            self.pos += 1
            self.depth -= 1
            return piece
        if _DIRAC_SUM_RE.match(self.text, start):
            return self.ketsum()
        return self.atom()

    # -- composition

    def spend(self, legs: int) -> None:
        """Count ``legs`` expanded legs against the per-parse bound."""
        self.expanded += legs
        if self.expanded > _MAX_EXPANDED_LEGS:
            raise ParseDomainError(f"input expands to more than {_MAX_EXPANDED_LEGS} legs")

    def same(self, a: Dim, b: Dim) -> bool:
        """Whether ``a`` and ``b`` agree once the minted bindings are substituted."""
        if a == b:
            return True
        bindings = self.abstraction.bindings
        return bool(_resolved(a, bindings) == _resolved(b, bindings))

    def unify(self, a: _DimTerm, b: _DimTerm) -> None:
        """Record ``a == b``; single variables are merged at once, or kept apart when their
        values agree only under the minted bindings."""
        if len(a) != 1 or len(b) != 1:
            self.equations.append((a, b))
            return
        ra, rb = a[0].root(), b[0].root()
        if ra is rb:
            return
        if ra.value is not None and rb.value is not None and ra.value != rb.value:
            if self.same(ra.value, rb.value):
                return
            raise ParseDomainError(
                f"dimension {ra.value} of {ra.label!r} is fed into {rb.value} of {rb.label!r}"
            )
        if ra.value is None:
            ra.value = rb.value
        rb.parent = ra

    def compose(self, a: _Piece, b: _Piece) -> _Piece:
        """``a ; b``: ``b``'s expanded inputs wired to ``a``'s leading outputs, peeling fans."""
        needed = 0
        for leg in b.inputs:
            if leg.mult is not None:
                raise ParseDomainError(
                    f"input fan x {leg.mult} of {leg.owner.label!r} cannot be fed: its count "
                    "is symbolic"
                )
            if leg.count is not None and leg.count > _MAX_KET_LEG_COUNT:
                raise ParseDomainError(
                    f"fan of {leg.count} legs is above the bound of {_MAX_KET_LEG_COUNT}"
                )
            needed += 1 if leg.count is None else leg.count
        available = 0
        for leg in a.outputs:
            if available >= needed:
                break
            if leg.mult is not None:
                raise ParseDomainError(
                    f"output fan x {leg.mult} of {leg.owner.label!r} cannot feed: its "
                    "count is symbolic"
                )
            available += 1 if leg.count is None else leg.count
        if available < needed:
            raise ParseDomainError(f"feed needs {needed} outputs, the left side has {available}")
        expansion: dict[int, list[_Leg]] = {}
        inputs: list[_Leg] = []
        for leg in b.inputs:
            if leg.count is None:
                inputs.append(leg)
                continue
            self.spend(leg.count)
            plain = [_Leg(leg.owner, leg.dim) for _ in range(leg.count)]
            expansion[id(leg)] = plain
            inputs.extend(plain)
        _rebuild({id(leg.owner): leg.owner for leg in b.inputs}, "inputs", expansion, False)
        supplied: list[_Leg] = []
        rest: list[_Leg] = []
        peeled: dict[int, list[_Leg]] = {}
        for leg in a.outputs:
            want = len(inputs) - len(supplied)
            if want and leg.count is None:
                supplied.append(leg)
                continue
            if want and leg.count is not None:
                take = min(leg.count, want)
                self.spend(take)
                fresh = [_Leg(leg.owner, leg.dim) for _ in range(take)]
                peeled[id(leg)] = fresh
                supplied.extend(fresh)
                leg.count -= take
            rest.append(leg)
        _rebuild({id(leg.owner): leg.owner for leg in a.outputs}, "outputs", peeled, True)
        for out_leg, in_leg in zip(supplied, inputs, strict=True):
            self.unify(out_leg.dim, in_leg.dim)
        a.nodes.extend(b.nodes)
        a.wires.extend(b.wires)
        a.wires.extend(zip(supplied, inputs, strict=True))
        a.outputs = rest + b.outputs
        a.scalars.extend(b.scalars)
        return a

    # -- ket sums

    def bound(self, text: str) -> Dim:
        """The ``D`` of a ``D-1`` sum bound."""
        legacy = _DIRAC_LEGACY_BOUND_RE.fullmatch(text)
        if legacy is not None:
            return _parse_dim_token(legacy["dim"], legacy["literal"] is not None, self.abstraction)
        node = _parse_expr(text, "sum bound")
        if not (
            isinstance(node, _Sum)
            and len(node.terms) == 2
            and not node.terms[0][0]
            and node.terms[1] == (True, _Num(1))
        ):
            raise ParseGrammarError(f"sum bound {text!r} is not 'D-1'")
        return _slot_dim(node.terms[0][1], self.abstraction.dim)

    def ketsum(self) -> _Piece:
        """``(SUM_{names=0}^{D-1})+ factors kets bras``; one Z spider per index."""
        indices: dict[str, _DiracNode] = {}
        while _DIRAC_SUM_RE.match(self.text, self.ws()):
            self.pos = self.text.index("_", self.pos) + 1
            names = _DIRAC_NAMES_RE.fullmatch(self.braced())
            if names is None:
                raise self.error("expected '{name, ...=0}' after the sum")
            if not self.text.startswith("^", self.pos):
                raise self.error("expected '^{D-1}'")
            self.pos += 1
            dim = self.bound(self.braced())
            for name in (n.strip() for n in names["names"].split(",")):
                if name in indices or not name.isidentifier():
                    raise ParseGrammarError(f"index {name!r} is bound twice or not a name")
                indices[name] = _DiracNode(Z_SPIDER, name, _Var(name, dim))
        for node in indices.values():
            assert node.var is not None and node.var.value is not None
            clash = node.var.value.free_symbols & set(indices)
            if clash:
                raise ParseDomainError(
                    f"dimension {node.var.value} names the bound index {min(clash)!r}"
                )
        piece = _Piece(nodes=list(indices.values()))
        delta_out: dict[str, list[_Leg]] = {name: [] for name in indices}
        delta_in: dict[str, list[_Leg]] = {name: [] for name in indices}
        while self.factor(indices, piece, delta_out, delta_in):
            pass
        for ket in (True, False):
            while self.ws() < len(self.text) and self.text[self.pos] in ("|" if ket else "<⟨"):
                for node, template in self.slots(indices, ket=ket):
                    side = node.outputs if ket else node.inputs
                    leg = _Leg(node, (_var(node),), *template)
                    side.append(leg)
                    (piece.outputs if ket else piece.inputs).append(leg)
        for name, node in indices.items():
            node.outputs.extend(delta_out[name])
            node.inputs.extend(delta_in[name])
        return piece

    def index(self, name: str, indices: Mapping[str, _DiracNode]) -> _DiracNode:
        if name not in indices:
            raise ParseGrammarError(f"index {name!r} is not bound by the sum")
        return indices[name]

    def factor(
        self,
        indices: Mapping[str, _DiracNode],
        piece: _Piece,
        delta_out: dict[str, list[_Leg]],
        delta_in: dict[str, list[_Leg]],
    ) -> bool:
        """Consume one ``[a = b]``, ``e^{2 pi i (...)}`` or ``1`` factor; False if none."""
        start = self.ws()
        if self.text.startswith("[", start):
            delta = _DIRAC_DELTA_RE.fullmatch(self.braced("["))
            if delta is None:
                raise self.error("expected '[a = b]'")
            a, b = self.index(delta["a"], indices), self.index(delta["b"], indices)
            self.unify((_var(a),), (_var(b),))
            out_leg, in_leg = _Leg(a, (_var(a),)), _Leg(b, (_var(b),))
            delta_out[a.label].append(out_leg)
            delta_in[b.label].append(in_leg)
            piece.wires.append((out_leg, in_leg))
            return True
        if self.text.startswith("e^{", start):
            self.pos += 2
            match = _DIRAC_PHASE_RE.fullmatch(self.braced())
            if match is None:
                raise self.error("expected 'e^{2 pi i (...)}'")
            for name, value, turns in _phase_terms(match["body"]):
                node = self.index(name, indices)
                bound = sorted(set(_names(turns)) & set(indices))
                if bound:
                    raise ParseDomainError(f"phase coefficient names the bound index {bound[0]!r}")
                if node.phase is None:
                    node.phase = {}
                node.phase.setdefault(value, []).append(turns)
            return True
        if _DIRAC_ONE_RE.match(self.text, start):
            self.pos += 1
            return True
        return False

    def slots(
        self, indices: Mapping[str, _DiracNode], *, ket: bool
    ) -> list[tuple[_DiracNode, _Count]]:
        """The slots of one ket (or bra) and its power: index node and leg template each."""
        self.pos += 1
        closers = ">⟩" if ket else "|"
        end = self.pos
        while end < len(self.text) and self.text[end] not in closers:
            if self.text[end] in "|<⟨>⟩":
                break
            end += 1
        if end >= len(self.text) or self.text[end] not in closers:
            raise self.error("unterminated ket" if ket else "unterminated bra")
        body = self.text[self.pos : end]
        self.pos = end + 1
        nodes: list[_DiracNode] = []
        for token in (t for t in re.split(r"[,\s]+", body) if t):
            if token in indices:
                nodes.append(indices[token])
            elif "[" in token:
                raise ParseGrammarError(f"copy families like {token!r} are not input syntax")
            elif len(set(token)) == 1 and token[0] in indices:
                nodes.extend([indices[token[0]]] * len(token))
            else:
                raise ParseGrammarError(f"index {token!r} is not bound by the sum")
        if not nodes:
            raise ParseGrammarError(f"empty {'ket' if ket else 'bra'}")
        plain: _Count = (None, None)
        if not self.text.startswith("^", self.pos):
            return [(node, plain) for node in nodes]
        self.pos += 1
        power = self.braced().strip()
        if len(nodes) != 1:
            raise ParseGrammarError(f"a power needs a single-slot ket or bra, got {body!r}")
        for prefix in ("\\otimes", "⊗"):
            if power.startswith(prefix):
                power = power[len(prefix) :]
                break
        node = _parse_expr(power, "power")
        if isinstance(node, _Lit):
            legs = _literal_count(node.value, minimum=1)
            self.spend(legs)
            return [(nodes[0], plain)] * legs
        count = _mult_of(node)
        if count.is_concrete:
            if count.to_int() < 1:
                raise ParseDomainError("a tensor power must be at least 1")
            return [(nodes[0], (count.to_int(), None))]
        return [(nodes[0], (None, count))]

    # -- catalog atoms

    def atom(self) -> _Piece:
        """One catalog atom with its optional phase, dagger and ``[dims]``."""
        start = self.pos
        match = _DIRAC_ATOM_RE.match(self.text, start)
        if match is None:
            raise self.error("expected a ket sum, a catalog atom or '('")
        name = match["sub"] or match["word"]
        self.pos = match.end()
        ins: list[_Count] = []
        outs: list[_Count] = []
        phase = None
        generator = Z_SPIDER
        if match["sub"] is not None:
            self.pos -= 1
            subscript = self.braced()
            if name in ("Z", "X"):
                left, arrow, right = subscript.partition("->")
                if not arrow:
                    raise ParseGrammarError(f"{name}_{{{subscript}}} is not '{name}_{{m->n}}'")
                ins, outs = _counts(left), _counts(right)
                generator = X_SPIDER if name == "X" else Z_SPIDER
                if self.at("(") and not self.at("(x)"):
                    phase = _dirac_phase_entries(self.braced("("))
            else:
                counted = _counts(subscript)
                if name == "W":
                    generator = REGISTRY.get("W")
                ins, outs = ([] if name == "GHZ" else [(None, None)]), counted
        elif name == "Z_phase":
            raise ParseGrammarError("Z_phase is a printer label; write Z_{1->1}(1: ...)[D]")
        elif name not in _DIRAC_WORDS:
            raise ParseGrammarError(f"unknown catalog atom {name!r}")
        else:
            generator, ins, outs = _word_atom(name)
        dagger = name in ("GHZ", "copy", "plus") and self.text.startswith("^dagger", self.pos)
        if dagger:
            self.pos += len("^dagger")
            ins, outs = outs, ins
        if name == "triangle" and self.text.startswith("^-1", self.pos):
            self.pos += len("^-1")
            generator = REGISTRY.get("Ti")
        label = self.text[start : self.pos]
        self.spend(sum(1 for c in (*ins, *outs) if c == (None, None)))
        dims: list[Dim] = []
        if self.at("["):
            inner = self.braced("[")
            if not inner.strip():
                raise ParseGrammarError(f"empty dimension list '[]' after {label!r}")
            dims = [self.dim(t) for t in _split_top(inner)]
        return self.place(generator, label, ins, outs, dims, phase)

    def at(self, text: str) -> bool:
        return self.text.startswith(text, self.pos)

    def dim(self, text: str) -> Dim:
        return _slot_dim(_parse_expr(text, "dimension"), self.abstraction.dim)

    def place(
        self,
        generator: GeneratorType,
        label: str,
        ins: Sequence[_Count],
        outs: Sequence[_Count],
        dims: Sequence[Dim],
        phase: dict[int, list[_Expr]] | None,
    ) -> _Piece:
        """The one-node piece of an atom; ``dims`` give its shared or ``[s, t]`` dimensions."""
        if generator.name in ("B", "S"):
            if generator.name == "S" and len(dims) != 2:
                raise ParseGrammarError(f"{label!r} needs explicit dimensions '[s, t]'")
            if dims and len(dims) != 2:
                raise ParseGrammarError(f"{label!r} takes two dimensions '[s, t]'")
            s, t = (
                (_Var(label, dims[0]), _Var(label, dims[1]))
                if dims
                else (
                    _Var(label),
                    _Var(label),
                )
            )
            node = _DiracNode(generator, label, None)
            pair = [_Leg(node, (s,)), _Leg(node, (t,))]
            whole = [_Leg(node, (s, t))]
            node.inputs, node.outputs = (pair, whole) if generator.name == "B" else (whole, pair)
        else:
            if len(dims) > 1:
                raise ParseGrammarError(f"{label!r} takes one dimension '[D]'")
            node = _DiracNode(generator, label, _Var(label, dims[0] if dims else None))
            node.inputs = [_Leg(node, (_var(node),), *c) for c in ins]
            node.outputs = [_Leg(node, (_var(node),), *c) for c in outs]
            node.phase = phase
        return _Piece([node], [], list(node.inputs), list(node.outputs))

    # -- finalization

    def value(self, term: _DimTerm) -> Dim | None:
        result = Dim.concrete(1)
        for var in term:
            value = var.root().value
            if value is None:
                return None
            result = value if len(term) == 1 else result * value
        return result

    def solve(self) -> None:
        """Propagate known products into single variables until stable, then check all."""
        changed = True
        while changed:
            changed = False
            for a, b in self.equations:
                va, vb = self.value(a), self.value(b)
                for known, other in ((va, b), (vb, a)):
                    if known is not None and len(other) == 1 and other[0].root().value is None:
                        other[0].root().value = known
                        changed = True
        for a, b in self.equations:
            va, vb = self.value(a), self.value(b)
            if va is not None and vb is not None and not self.same(va, vb):
                raise ParseDomainError(f"dimension {va} is fed into {vb}")

    def build(self, piece: _Piece) -> Diagram:
        """Nodes in creation order, boxes by node id then inputs, outputs and port index."""
        self.solve()
        sides: list[tuple[list[Dim], list[Dim]]] = []
        for node in piece.nodes:
            found = [[self.value(leg.dim) for leg in legs] for legs in (node.inputs, node.outputs)]
            phase_dim = None if node.var is None else node.var.root().value
            if any(d is None for side in found for d in side) or (
                node.var is not None and phase_dim is None
            ):
                hint = "[s, t]" if node.generator.name in ("B", "S") else "[D]"
                raise ParseDomainError(
                    f"cannot infer the dimension of {node.label!r}; give it as {node.label}{hint}"
                )
            sides.append(([d for d in found[0] if d], [d for d in found[1] if d]))
        names = {
            name
            for node in piece.nodes
            for leg in (*node.inputs, *node.outputs)
            if leg.mult is not None
            for name in leg.mult.free_symbols
        }
        sides = [
            ([_with_mults(d, names) for d in i], [_with_mults(d, names) for d in o])
            for i, o in sides
        ]
        diagram = Diagram()
        refs: dict[int, PortRef] = {}
        for node, (ins, outs) in zip(piece.nodes, sides, strict=True):
            node_id = diagram.add_node(node.generator, ins, outs)
            for direction, legs in (
                (Direction.INPUT, node.inputs),
                (Direction.OUTPUT, node.outputs),
            ):
                for i, leg in enumerate(legs):
                    refs[id(leg)] = PortRef(node_id, direction, i)
        for out_leg, in_leg in piece.wires:
            diagram.add_wire(refs[id(out_leg)], refs[id(in_leg)])
        diagram.set_boundary_inputs([refs[id(leg)] for leg in piece.inputs])
        diagram.set_boundary_outputs([refs[id(leg)] for leg in piece.outputs])
        mults: list[Mult] = []
        for node in piece.nodes:
            for leg in (*node.inputs, *node.outputs):
                if leg.count is None and leg.mult is None:
                    continue
                mult = leg.mult if leg.mult is not None else self.abstraction.count(leg.count or 0)
                mults.append(mult)
                diagram.add_bang_box(mult, port_scope=frozenset({refs[id(leg)]}))
        dims = [d for ins, outs in sides for d in (*ins, *outs)]
        dims += [
            _with_mults(v, names)
            for n in piece.nodes
            if n.var is not None and (v := n.var.root().value)
        ]
        table = _role_table(dims, mults)
        for node_id, node in zip(sorted(diagram.nodes), piece.nodes, strict=True):
            if node.var is None or node.generator.name not in ("Z", "X"):
                continue
            dim = node.var.root().value
            assert dim is not None
            dim = _with_mults(dim, names)
            if node.phase is not None:
                entries = {
                    value: _phase_value(sp.Add(*(_phase_expr(t, table) for t in terms)))
                    for value, terms in node.phase.items()
                }
                diagram.set_phase(node_id, PhaseVector(dim, entries))
            elif not node.inputs and not node.outputs:
                diagram.set_phase(node_id, PhaseVector(dim))
        if piece.scalars:
            table = _role_table(dims, mults, _phases(diagram))
            product = _product([_scalar_expr(ast, table) for ast in piece.scalars])
            diagram.set_scalar(Scalar(_check_terms(product, "scalar")))
        diagram.set_parameters(self.abstraction.bindings)
        return diagram


def _rebuild(
    owners: Mapping[int, _DiracNode], side: str, legs: Mapping[int, list[_Leg]], keep: bool
) -> None:
    """Each owner's ``side`` list in one pass: a leg in ``legs`` becomes its list, followed
    by the leg itself when ``keep``."""
    if not legs:
        return
    for owner in owners.values():
        rebuilt: list[_Leg] = []
        for leg in getattr(owner, side):
            if id(leg) in legs:
                rebuilt.extend(legs[id(leg)])
                if keep:
                    rebuilt.append(leg)
            else:
                rebuilt.append(leg)
        setattr(owner, side, rebuilt)


def _var(node: _DiracNode) -> _Var:
    assert node.var is not None
    return node.var


def _position(legs: Sequence[_Leg], leg: _Leg) -> int:
    """The index of ``leg`` in ``legs`` by identity."""
    return next(i for i, other in enumerate(legs) if other is leg)


def _parse_dim_token(token: str, literal: bool, abstraction: _Abstraction) -> Dim:
    """A legacy-shaped dimension token: a numeral through ``abstraction``, or a symbol."""
    if _ASCII_DIGITS_RE.match(token):
        return abstraction.dim(_numeral(token, "dimension token"), literal)
    if literal:
        raise ParseGrammarError(f"{_LITERAL_MARKER!r} needs a numeral, got {token!r}")
    return Dim.symbol(token)


def _literal_count(value: int, minimum: int = 0) -> int:
    if value < minimum:
        raise ParseDomainError(f"leg count {value} is below {minimum}")
    if value > _MAX_KET_LEG_COUNT:
        raise ParseDomainError(f"leg count {value} is above the bound of {_MAX_KET_LEG_COUNT}")
    return value


def _counts(text: str) -> list[_Count]:
    """The legs of a catalog subscript: ``literal c`` plain legs; ``c`` none, one plain leg, or
    one pending fan; ``c + rest`` c plain legs and a fan of ``rest``."""
    node = _parse_expr(text, "count")
    plain: _Count = (None, None)
    if isinstance(node, _Lit):
        return [plain] * _literal_count(node.value)
    whole = _mult_of(node)
    if whole.is_concrete:
        c = whole.to_int()
        return [] if c == 0 else [plain] if c == 1 else [(c, None)]
    constant = 0
    rest: _Expr = node
    if isinstance(node, _Sum) and not any(negated for negated, _ in node.terms):
        numbers = [t.value for _, t in node.terms if isinstance(t, _Num)]
        others = tuple((False, t) for _, t in node.terms if not isinstance(t, _Num))
        constant = sum(numbers)
        if others:
            rest = others[0][1] if len(others) == 1 else _Sum(others)
        else:
            rest = _Num(constant)
            constant = 0
    mult = _mult_of(rest)
    if mult.is_concrete:
        c = mult.to_int()
        return [] if c == 0 else [plain] if c == 1 else [(c, None)]
    return [plain] * _literal_count(constant) + [(None, mult)]


def _word_atom(name: str) -> tuple[GeneratorType, list[_Count], list[_Count]]:
    plain: _Count = (None, None)
    shapes = {
        "plus": ("Z", 0, 1),
        "id": ("Z", 1, 1),
        "copy": ("Z", 1, 2),
        "zero": ("X", 0, 1),
        "F": ("F", 1, 1),
        "triangle": ("T", 1, 1),
        "bind": ("B", 2, 1),
        "split": ("S", 1, 2),
    }
    gen, m, n = shapes[name]
    return REGISTRY.get(gen), [plain] * m, [plain] * n


def _dirac_phase_entries(text: str) -> dict[int, list[_Expr]]:
    """``i: expr, ...`` as value -> one turn AST."""
    entries: dict[int, list[_Expr]] = {}
    items = _split_top(text)
    if not items:
        raise ParseGrammarError("empty phase entry list")
    for item in items:
        index, colon, expr = item.partition(":")
        if not colon:
            raise ParseGrammarError(f"phase entry {item!r} is not 'INDEX: EXPR'")
        i = _int_token(index, "phase index")
        if i in entries:
            raise ParseGrammarError(f"phase index {i} appears twice")
        entries[i] = [_parse_expr(expr, "phase")]
    return entries


def _phase_terms(body: str) -> list[tuple[str, int, _Expr]]:
    """``expr [k=j] + ...`` as (index name, value, turn AST) triples."""
    terms: list[tuple[str, int, _Expr]] = []
    pairs = _bracket_pairs(body)
    i = 0
    while True:
        depth = 0
        j = i
        while j < len(body) and not (body[j] == "[" and depth == 0):
            depth += body[j] in "({"
            depth -= body[j] in ")}"
            j += 1
        if j >= len(body):
            raise ParseGrammarError(f"phase term {body[i:].strip()!r} lacks '[k=j]'")
        at = _DIRAC_AT_RE.fullmatch(body[j + 1 : pairs[j]])
        if at is None:
            raise ParseGrammarError(f"{body[j : pairs[j] + 1]!r} is not '[k=j]'")
        terms.append(
            (at["index"], _numeral(at["value"], "phase value"), _parse_expr(body[i:j], "phase"))
        )
        i = pairs[j] + 1
        rest = body[i:].lstrip()
        if not rest:
            return terms
        if not rest.startswith("+"):
            raise ParseGrammarError(f"expected '+' between phase terms, found {rest[:12]!r}")
        i = len(body) - len(rest) + 1


@_guarded
def parse_dirac(source: str) -> Diagram:
    """The diagram a Dirac expression denotes; a lone scalar expression is the empty diagram
    with that scalar, and a leading ``NAME = `` is dropped."""
    text = _require_text(source)
    formula = _strip_name(text)
    scalar = _scalar_only(formula)
    if isinstance(scalar, _Name):
        raise ParseGrammarError(
            f"unknown catalog atom or lone scalar name {scalar.name!r}; write a scalar as an "
            "expression such as '1*x'"
        )
    if scalar is not None:
        diagram = Diagram()
        diagram.set_scalar(_scalar_of(scalar, {}))
        return _finish(diagram)
    return _finish(_check_phase_range(_Dirac(formula, _Abstraction(text)).read()))


# -- form 4: rule steps --------------------------------------------------------------------

_STEP_RE = re.compile(
    rf"apply\s+(?P<rule>{_IDENTIFIER})"
    r"(?:\s+at\s+(?:(?P<index>[0-9]+)|(?P<nodes>n[0-9]+(?:\s*,\s*n[0-9]+)*)))?"
)


@dataclass(frozen=True, slots=True)
class RuleStep:
    """``apply RULE [at INDEX | at nI, nJ, ...]``: a rule name and an optional match selector."""

    rule: str
    index: int | None
    nodes: tuple[int, ...] | None
    source: str

    def __post_init__(self) -> None:
        """Type and range checks: ParseGrammarError on a bad field."""

        def count(value: object) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0

        if not isinstance(self.rule, str):
            raise ParseGrammarError(f"rule must be a str, got {self.rule!r}")
        if self.index is not None and not count(self.index):
            raise ParseGrammarError(f"index must be None or an int >= 0, got {self.index!r}")
        if self.nodes is not None and (
            not isinstance(self.nodes, tuple)
            or not all(count(n) for n in self.nodes)
            or len(set(self.nodes)) != len(self.nodes)
        ):
            raise ParseGrammarError(
                f"nodes must be None or a tuple of distinct ints >= 0, got {self.nodes!r}"
            )
        if self.index is not None and self.nodes is not None:
            raise ParseGrammarError("a rule step selects by index or by nodes, not both")


@_guarded
def parse_step(source: str) -> RuleStep:
    """The :class:`RuleStep` an ``apply`` line names; the rule must be in ``RULES``."""
    text = _require_text(source).strip()
    match = _STEP_RE.fullmatch(text)
    if match is None:
        raise ParseGrammarError(f"{text!r} is not 'apply RULE [at INT | at nI, nJ, ...]'")
    if match["rule"] not in RULES:
        raise ParseDomainError(f"no rule {match['rule']!r}; known: {sorted(RULES)}")
    index = None if match["index"] is None else _numeral(match["index"], "match index")
    nodes = None
    if match["nodes"] is not None:
        nodes = tuple(_numeral(t.strip()[1:], "node id") for t in match["nodes"].split(","))
        if len(set(nodes)) != len(nodes):
            raise ParseGrammarError(f"{text!r} names a node twice")
    return RuleStep(match["rule"], index, nodes, text)


@_guarded
def apply_step(diagram: Diagram, step: RuleStep) -> RewriteResult:
    """``step``'s rule applied at the selected match of ``rule.pattern.find_matches``: the
    match whose support equals ``step.nodes``, else the match at ``step.index`` (default 0)."""
    if not isinstance(diagram, Diagram) or not isinstance(step, RuleStep):
        raise ParseGrammarError("apply_step takes a Diagram and a RuleStep")
    if step.rule not in RULES:
        raise ParseDomainError(f"no rule {step.rule!r}")
    rule = RULES[step.rule]
    matches = rule.pattern.find_matches(diagram)
    if step.nodes is not None:
        wanted = set(step.nodes)
        found = [m for m in matches if set(m.support_node_ids) == wanted]
        if len(found) != 1:
            what = "no" if not found else f"{len(found)}"
            raise ParseDomainError(f"{what} {step.rule} matches on nodes {sorted(wanted)}")
        return apply(diagram, rule, found[0])
    index = 0 if step.index is None else step.index
    if index >= len(matches):
        raise ParseDomainError(f"{step.rule} has {len(matches)} match(es); no match {index}")
    return apply(diagram, rule, matches[index])


# -- dispatch ------------------------------------------------------------------------------


class InputForm(enum.Enum):
    """The four input forms :func:`parse` accepts."""

    DIRAC = "dirac"
    DSL = "dsl"
    LISTING = "listing"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class ParsedDiagram:
    """A parsed diagram and the form it was read as."""

    diagram: Diagram
    form: InputForm


def _form(source: str) -> InputForm:
    """``graph:`` prefix: LISTING; first word ``apply``: STEP; a first statement opening
    with a DSL keyword or ``NAME = GEN(`` (GEN a generator, ``(`` not ``(x)``): DSL; else
    DIRAC."""
    text = source.strip()
    if text.startswith("graph:"):
        return InputForm.LISTING
    if re.match(r"apply\b", text):
        return InputForm.STEP
    statements = _statements(text)
    if not statements:
        return InputForm.DIRAC
    first = statements[0]
    if _DSL_START_RE.match(first):
        return InputForm.DSL
    node = _DSL_NODE_RE.match(first)
    if (
        node is not None
        and node["gen"] in REGISTRY.names()
        and not first.startswith("(x)", node.end() - 1)
    ):
        return InputForm.DSL
    return InputForm.DIRAC


@_guarded
def parse(source: str) -> ParsedDiagram | RuleStep:
    """``source`` with ``#`` comments cut, read in the form :func:`_form` picks."""
    source = _require_text(source)
    if "#" in source:
        source = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    form = _form(source)
    if form is InputForm.STEP:
        return parse_step(source)
    reader = {
        InputForm.LISTING: parse_listing,
        InputForm.DSL: parse_dsl,
        InputForm.DIRAC: parse_dirac,
    }[form]
    return ParsedDiagram(reader(source), form)


@_guarded
def parse_diagram(source: str) -> Diagram:
    """The diagram ``source`` denotes; a rule step is a grammar error."""
    result = parse(source)
    if isinstance(result, RuleStep):
        raise ParseGrammarError("a rule step names no diagram")
    return result.diagram
