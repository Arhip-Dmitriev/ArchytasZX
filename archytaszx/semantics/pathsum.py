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

"""Symbolic path-sum states and operators over qudit registers.

A :class:`State` is a tuple of registers and a sum of :class:`Term` s; a term is
``amp * sum_{bound} |k_1>|k_2>...`` with sympy expressions for the amplitude, the register
values and the bound ranges. Register r holds ``widths[r]`` qudits of dimension ``dims[r]``
(2 by default); its value is a big-endian base-``dims[r]`` integer in ``0..d^w-1``. Widths,
dimensions and values may be symbolic.

Operators (:class:`Gate`, :class:`Identity`, :class:`Power`, :class:`Tensor`,
:class:`Compose`, :class:`Outer`, :class:`Scaled`, :class:`OpSum`) act through
:func:`apply`. A gate is a callable from register values, widths and dimensions to a state,
so applying it is substitution. The tensor power of a one-qudit gate or state factors its
amplitude digit by digit: exponents linear in the input digit a, the output digit y and
``a*y`` become :class:`dsum` and :class:`ddot` (``pop`` and ``bitdot`` for qubits).

While :func:`apply` runs, a free symbol of an operator that meets a register of another
width or dimension is bound to that value (:func:`bind`), and the bound values are substituted
into the operator's output.

:func:`measure` measures every register: numerically (sampled with a numpy generator) when
the state is concrete and has at most :data:`PATH_LIMIT` basis paths, otherwise symbolically (an outcome symbol, the projected state and the
constraints it implies). :func:`inner` contracts a bra with a ket.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TypeAlias

import numpy as np
import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

Expr: TypeAlias = sympy.Expr
GateFn: TypeAlias = Callable[[tuple[Expr, ...], tuple[Expr, ...], tuple[Expr, ...]], "State"]


def expand(e: object) -> Expr:
    """``sympy.expand`` keeping logarithms, power sums and power bases intact."""
    return sympy.expand(e, log=False, power_exp=False, power_base=False)


class PathSumError(Exception):
    """A path-sum operation that is undefined on its inputs."""


def _bit_count(v: int) -> int:
    return int(v).bit_count()


def _digits(v: int, d: int) -> Iterator[int]:
    v = int(v)
    while v:
        v, r = divmod(v, d)
        yield r


def _ints(*args: Expr) -> bool:
    return all(a.is_Integer for a in args)


class _IntFunction(sympy.Function):  # type: ignore[misc]
    """An integer-valued function."""

    def _eval_is_integer(self) -> bool:
        return True


class xor(_IntFunction):
    """Bitwise exclusive or of two non-negative integers."""

    nargs = 2

    @classmethod
    def eval(cls, a: Expr, b: Expr) -> Expr | None:
        if _ints(a, b):
            return sympy.Integer(int(a) ^ int(b))
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return sympy.S.Zero
        return None


class pop(_IntFunction):
    """Population count of a non-negative integer."""

    nargs = 1

    @classmethod
    def eval(cls, v: Expr) -> Expr | None:
        return sympy.Integer(_bit_count(int(v))) if v.is_Integer else None


class bitdot(_IntFunction):
    """Population count of the bitwise and of two non-negative integers."""

    nargs = 2

    @classmethod
    def eval(cls, a: Expr, b: Expr) -> Expr | None:
        if _ints(a, b):
            return sympy.Integer(_bit_count(int(a) & int(b)))
        if a == 0 or b == 0:
            return sympy.S.Zero
        return None


class dsum(_IntFunction):
    """``dsum(v, d)``: the sum of the base-d digits of v; :class:`pop` for d = 2."""

    nargs = 2

    @classmethod
    def eval(cls, v: Expr, d: Expr) -> Expr | None:
        if d == 2:
            return pop(v)
        if _ints(v, d):
            return sympy.Integer(sum(_digits(int(v), int(d))))
        return sympy.S.Zero if v == 0 else None


class ddot(_IntFunction):
    """``ddot(a, b, d)``: the sum of products of aligned base-d digits; :class:`bitdot` for
    d = 2."""

    nargs = 3

    @classmethod
    def eval(cls, a: Expr, b: Expr, d: Expr) -> Expr | None:
        if d == 2:
            return bitdot(a, b)
        if _ints(a, b, d):
            pairs = itertools.zip_longest(
                _digits(int(a), int(d)), _digits(int(b), int(d)), fillvalue=0
            )
            return sympy.Integer(sum(x * y for x, y in pairs))
        return sympy.S.Zero if a == 0 or b == 0 else None


class bits(_IntFunction):
    """``bits(z, lo, hi, w)``: bits ``lo..hi-1`` of the width-``w`` big-endian bitstring z."""

    nargs = 4

    @classmethod
    def eval(cls, z: Expr, lo: Expr, hi: Expr, w: Expr) -> Expr | None:
        if lo == 0 and expand(hi - w) == 0:
            return z
        if _ints(z, lo, hi, w):
            return sympy.Integer((int(z) >> (int(w) - int(hi))) & ((1 << (int(hi) - int(lo))) - 1))
        return None


class digits(_IntFunction):
    """``digits(z, lo, hi, w, d)``: digits ``lo..hi-1`` of the width-``w`` base-d string z;
    :class:`bits` for d = 2."""

    nargs = 5

    @classmethod
    def eval(cls, z: Expr, lo: Expr, hi: Expr, w: Expr, d: Expr) -> Expr | None:
        if d == 2:
            return bits(z, lo, hi, w)
        if lo == 0 and expand(hi - w) == 0:
            return z
        if _ints(z, lo, hi, w, d):
            base = int(d)
            return sympy.Integer(
                (int(z) // base ** (int(w) - int(hi))) % base ** (int(hi) - int(lo))
            )
        return None


class part(_IntFunction):
    """``part(z, k, below, levels)``: register k of a mixed-radix outcome z."""

    nargs = 4

    @classmethod
    def eval(cls, z: Expr, k: Expr, below: Expr, levels: Expr) -> Expr | None:
        if _ints(z, below, levels):
            return sympy.Integer((int(z) // int(below)) % int(levels))
        return None


class shr(_IntFunction):
    """``shr(v, k, d)``: v with its k lowest base-d digits dropped."""

    nargs = 3

    @classmethod
    def eval(cls, v: Expr, k: Expr, d: Expr) -> Expr | None:
        if _ints(v, k, d):
            return sympy.Integer(int(v) // int(d) ** int(k))
        if v == 0 or k == 0:
            return v
        return None


class dshift(_IntFunction):
    """``dshift(v, k, d, w)``: every base-d digit of the w-digit v increased by k mod d."""

    nargs = 4

    @classmethod
    def eval(cls, v: Expr, k: Expr, d: Expr, w: Expr) -> Expr | None:
        if k == 0:
            return v
        small = k.is_Integer and int(k) > 0 and (int(k) == 1 or (d.is_Integer and int(k) < int(d)))
        if v == 0 and small and not _ints(d, w):
            return k * (d**w - 1) / (d - 1)
        if _ints(v, k, d, w):
            base, out = int(d), 0
            for pos in range(int(w)):
                digit = (int(v) // base**pos) % base
                out += ((digit + int(k)) % base) * base**pos
            return sympy.Integer(out)
        return None


class bit(_IntFunction):
    """``bit(x, j)``: bit j of x, counted from the least significant bit."""

    nargs = 2

    @classmethod
    def eval(cls, x: Expr, j: Expr) -> Expr | None:
        if _ints(x, j):
            return sympy.Integer((int(x) >> int(j)) & 1)
        return sympy.S.Zero if x == 0 else None


class powmod(_IntFunction):
    """``powmod(b, e, m)``: ``b**e % m`` by modular exponentiation."""

    nargs = 3

    @classmethod
    def eval(cls, b: Expr, e: Expr, m: Expr) -> Expr | None:
        if _ints(b, e, m) and int(e) >= 0 and int(m) != 0:
            return sympy.Integer(pow(int(b), int(e), int(m)))
        return None


class igcd(_IntFunction):
    """Greatest common divisor of integers; unevaluated on symbolic arguments."""

    nargs = 2

    @classmethod
    def eval(cls, a: Expr, b: Expr) -> Expr | None:
        if _ints(a, b):
            return sympy.Integer(sympy.igcd(int(a), int(b)))
        if a == b:
            return a
        return None


class denom(_IntFunction):
    """``denom(x, limit)``: the denominator of the best approximation of x with denominator at
    most limit (continued fractions)."""

    nargs = 2

    @classmethod
    def eval(cls, x: Expr, limit: Expr) -> Expr | None:
        if x.is_Rational and limit.is_Integer and int(limit) >= 1:
            frac = Fraction(int(x.p), int(x.q)).limit_denominator(int(limit))
            return sympy.Integer(frac.denominator)
        return None


def to_powmod(expr: Expr) -> Expr:
    """``Mod(b**e, m)`` subterms rewritten as :class:`powmod`."""
    return expr.replace(
        lambda e: isinstance(e, sympy.Mod) and e.args[0].is_Pow,
        lambda e: powmod(e.args[0].base, e.args[0].exp, e.args[1]),
    )


def _obj(x: object) -> np.ndarray:
    return np.asarray(x, dtype=object)


_IMPLS: dict[str, object] = {
    "xor": lambda a, b: np.bitwise_xor(_obj(a), _obj(b)),
    "pop": np.frompyfunc(_bit_count, 1, 1),
    "bitdot": np.frompyfunc(lambda a, b: _bit_count(int(a) & int(b)), 2, 1),
    "dsum": np.frompyfunc(
        lambda v, d: int(dsum(sympy.Integer(int(v)), sympy.Integer(int(d)))), 2, 1
    ),
    "ddot": np.frompyfunc(
        lambda a, b, d: int(
            ddot(sympy.Integer(int(a)), sympy.Integer(int(b)), sympy.Integer(int(d)))
        ),
        3,
        1,
    ),
    "bits": lambda z, lo, hi, w: (_obj(z) >> (w - hi)) & ((1 << (hi - lo)) - 1),
    "digits": lambda z, lo, hi, w, d: (_obj(z) // d ** (w - hi)) % d ** (hi - lo),
    "part": lambda z, k, below, levels: (_obj(z) // below) % levels,
    "shr": lambda v, k, d: _obj(v) // _obj(d) ** k,
    "dshift": np.frompyfunc(
        lambda v, k, d, w: int(dshift(*(sympy.Integer(int(x)) for x in (v, k, d, w)))), 4, 1
    ),
    "powmod": np.frompyfunc(lambda b, e, m: pow(int(b), int(e), int(m)), 3, 1),
    "bit": lambda x, j: (_obj(x) >> j) & 1,
}


def _scalar_of(cls: type) -> Callable[..., int]:
    def run(*args: object) -> int:
        return int(cls(*(sympy.Integer(int(a)) for a in args)))  # type: ignore[call-overload]

    return run


_SCALAR: dict[str, object] = {
    "xor": lambda a, b: int(a) ^ int(b),
    "pop": _bit_count,
    "bitdot": lambda a, b: _bit_count(int(a) & int(b)),
    "dsum": _scalar_of(dsum),
    "ddot": _scalar_of(ddot),
    "bits": lambda z, lo, hi, w: (int(z) >> (int(w) - int(hi))) & ((1 << (int(hi) - int(lo))) - 1),
    "digits": _scalar_of(digits),
    "part": lambda z, k, below, levels: (int(z) // int(below)) % int(levels),
    "shr": lambda v, k, d: int(v) // int(d) ** int(k),
    "dshift": _scalar_of(dshift),
    "powmod": lambda b, e, m: pow(int(b), int(e), int(m)),
    "bit": lambda x, j: (int(x) >> int(j)) & 1,
}


def scalar_impls(expr: Expr) -> dict[str, object]:
    """Python-int lambdify namespace for the integer functions in ``expr``."""
    names = dict(_SCALAR)
    for app in expr.atoms(sympy.Function):
        impl = getattr(type(app), "_scalar", None)
        if impl is not None:
            names[type(app).__name__] = impl
    return names


def numeric_impls(expr: Expr) -> dict[str, object]:
    """Lambdify namespace for the integer functions in ``expr``, user functions included."""
    names = dict(_IMPLS)
    for app in expr.atoms(sympy.Function):
        impl = getattr(type(app), "_numeric", None)
        if impl is not None:
            names[type(app).__name__] = impl
    return names


@dataclass(frozen=True)
class Bound:
    """A summation index ``symbol`` over ``lo..hi`` inclusive."""

    symbol: sympy.Symbol
    lo: Expr
    hi: Expr


@dataclass(frozen=True)
class Term:
    """``amp * sum_{bound} |kets>``."""

    amp: Expr
    bound: tuple[Bound, ...]
    kets: tuple[Expr, ...]


TWO = sympy.Integer(2)

PATH_LIMIT = 1 << 22
"""The most basis paths a concrete state is enumerated over numerically."""

LIST_LIMIT = 1 << 12
"""The most basis paths a concrete state is listed over, one term per basis ket."""


@dataclass(frozen=True)
class State:
    """Register widths and dimensions (``None`` while unresolved) and a sum of terms; a bra
    when ``bra``. Omitted dimensions are 2."""

    widths: tuple[Expr | None, ...]
    terms: tuple[Term, ...]
    bra: bool = False
    dims: tuple[Expr | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.dims and self.widths:
            object.__setattr__(self, "dims", (TWO,) * len(self.widths))
        if len(self.dims) != len(self.widths):
            raise PathSumError("one dimension per register")

    @property
    def total_width(self) -> Expr:
        if any(w is None for w in self.widths):
            raise PathSumError("a register width is unresolved")
        return sympy.Add(*self.widths) if self.widths else sympy.S.Zero

    def levels(self, k: int) -> Expr:
        """``d^w`` of register k."""
        d, w = self.dims[k], self.widths[k]
        if d is None or w is None:
            raise PathSumError("a register width or dimension is unresolved")
        return d**w

    def uniform_dim(self) -> Expr | None:
        """The common dimension of every register, or None when they differ."""
        if not self.dims:
            return TWO
        first = self.dims[0]
        return first if all(same(d, first) for d in self.dims) else None


def _with(
    s: State,
    widths: tuple[Expr | None, ...] | None = None,
    terms: Sequence[Term] | None = None,
    bra: bool | None = None,
    dims: tuple[Expr | None, ...] | None = None,
) -> State:
    return State(
        s.widths if widths is None else widths,
        s.terms if terms is None else tuple(terms),
        s.bra if bra is None else bra,
        s.dims if dims is None else dims,
    )


def new_bound(name: str, lo: Expr, hi: Expr) -> Bound:
    """A bound index with a fresh integer dummy named ``name``."""
    return Bound(sympy.Dummy(name, integer=True), sympy.sympify(lo), sympy.sympify(hi))


def same(a: Expr | None, b: Expr | None) -> bool:
    """Symbolic equality of two widths, dimensions or values."""
    if a is None or b is None:
        return False
    diff = expand(sympy.sympify(a) - sympy.sympify(b))
    return bool(diff == 0 or sympy.simplify(diff) == 0)


@dataclass
class Params:
    """Values bound to free symbols during one :func:`apply`; ``fixed`` symbols never bind."""

    fixed: frozenset[sympy.Symbol] = frozenset()
    values: dict[sympy.Symbol, Expr] = field(default_factory=dict)

    def resolve(self, e: Expr) -> Expr:
        """``e`` with the bound values substituted."""
        return sympy.sympify(e).xreplace(self.values)

    def unify(self, want: Expr, have: Expr) -> bool:
        """Whether ``want`` equals ``have``, binding the one free symbol of ``want`` if needed."""
        want, have = self.resolve(want), sympy.sympify(have)
        if same(want, have):
            return True
        free = [x for x in want.free_symbols if not isinstance(x, sympy.Dummy)]
        if len(free) != 1 or free[0] in self.fixed or free[0] in have.free_symbols:
            return False
        sols = sympy.solve(sympy.Eq(want, have), free[0])
        if len(sols) != 1 or any(isinstance(x, sympy.Dummy) for x in sols[0].free_symbols):
            return False
        self.values[free[0]] = sols[0]
        return True


_SCOPE: ContextVar[Params | None] = ContextVar("pathsum_params", default=None)


def bind(want: Expr | None, have: Expr | None) -> bool:
    """``same(want, have)``, binding a free symbol of ``want`` inside a running :func:`apply`."""
    params = _SCOPE.get()
    if params is None or want is None or have is None:
        return same(want, have)
    return params.unify(want, have)


def resolve(e: Expr) -> Expr:
    """``e`` with the values bound in the running :func:`apply` substituted."""
    params = _SCOPE.get()
    return sympy.sympify(e) if params is None else params.resolve(e)


def free_symbols(s: State) -> frozenset[sympy.Symbol]:
    """The non-dummy symbols of every width, dimension, range, amplitude and value of ``s``."""
    exprs: list[Expr] = [x for x in (*s.widths, *s.dims) if x is not None]
    for t in s.terms:
        exprs.extend((t.amp, *t.kets, *(b.lo for b in t.bound), *(b.hi for b in t.bound)))
    return frozenset(
        x
        for e in exprs
        for x in sympy.sympify(e).free_symbols
        if isinstance(x, sympy.Symbol) and not isinstance(x, sympy.Dummy)
    )


def substitute(s: State, mapping: dict[sympy.Symbol, Expr]) -> State:
    """``s`` with ``mapping`` applied to its widths, dimensions and terms."""
    if not mapping:
        return s

    def sub(x: Expr | None) -> Expr | None:
        return None if x is None else sympy.sympify(x).xreplace(mapping)

    return State(
        tuple(map(sub, s.widths)),
        tuple(_subs_term(t, mapping) for t in s.terms),  # type: ignore[arg-type]
        s.bra,
        tuple(map(sub, s.dims)),
    )


def _resolved(s: State) -> State:
    params = _SCOPE.get()
    return s if params is None else substitute(s, params.values)


def _same_layout(a: State, b: State) -> bool:
    return (
        len(a.widths) == len(b.widths)
        and all(map(same, a.widths, b.widths))
        and all(map(same, a.dims, b.dims))
    )


def _subs_term(term: Term, mapping: dict[sympy.Basic, Expr]) -> Term:
    return Term(
        term.amp.xreplace(mapping),
        tuple(
            Bound(b.symbol, b.lo.xreplace(mapping), b.hi.xreplace(mapping))
            for b in term.bound
            if b.symbol not in mapping
        ),
        tuple(k.xreplace(mapping) for k in term.kets),
    )


def fresh(term: Term) -> Term:
    """``term`` with new dummies for its bound indices."""
    if not term.bound:
        return term
    mapping = {b.symbol: sympy.Dummy(b.symbol.name, integer=True) for b in term.bound}
    return Term(
        term.amp.xreplace(mapping),
        tuple(
            Bound(mapping[b.symbol], b.lo.xreplace(mapping), b.hi.xreplace(mapping))
            for b in term.bound
        ),
        tuple(k.xreplace(mapping) for k in term.kets),
    )


def basis(value: Expr, width: Expr | None, dim: Expr | None = TWO) -> State:
    """The basis ket ``|value>`` on one register."""
    return State((width,), (Term(sympy.S.One, (), (sympy.sympify(value),)),), False, (dim,))


def basis_kets(
    values: Sequence[Expr], widths: Sequence[Expr | None], dims: Sequence[Expr | None] = ()
) -> State:
    """The basis ket with one register per value."""
    return State(tuple(widths), (Term(sympy.S.One, (), tuple(values)),), False, tuple(dims))


EMPTY = State((), (Term(sympy.S.One, (), ()),))
"""The zero-register state ``1``."""


def tensor(a: State, b: State) -> State:
    """``a (x) b``; both kets or both bras."""
    if a.bra != b.bra:
        raise PathSumError("cannot tensor a ket with a bra")
    right = [fresh(t) for t in b.terms]
    terms = tuple(
        Term(ta.amp * tb.amp, ta.bound + tb.bound, ta.kets + tb.kets)
        for ta in a.terms
        for tb in right
    )
    return State(a.widths + b.widths, terms, a.bra, a.dims + b.dims)


def scale(c: Expr, s: State) -> State:
    """``c * s``."""
    return _with(s, terms=[Term(c * t.amp, t.bound, t.kets) for t in s.terms])


def add(a: State, b: State) -> State:
    """``a + b``; register layouts must agree or join into one register of one dimension."""
    if a.bra != b.bra:
        raise PathSumError("cannot add a ket and a bra")
    if not _same_layout(a, b):
        da, db = a.uniform_dim(), b.uniform_dim()
        if not (same(a.total_width, b.total_width) and same(da, db)):
            raise PathSumError("cannot add states of different widths or dimensions")
        a, b = merge(a, 0, len(a.widths)), merge(b, 0, len(b.widths))
    return _with(a, terms=a.terms + b.terms)


def conjugate(expr: Expr) -> Expr:
    """The complex conjugate, with non-negative symbols read as positive."""
    reps = {
        x: sympy.Dummy(x.name, positive=True, integer=x.is_integer)
        for x in expr.free_symbols
        if x.is_nonnegative and not isinstance(x, sympy.Dummy)
    }
    return sympy.conjugate(expr.xreplace(reps)).xreplace({v: k for k, v in reps.items()})


def dagger(s: State) -> State:
    """The conjugate transpose: ket to bra and bra to ket."""
    terms = [Term(conjugate(t.amp), t.bound, t.kets) for t in s.terms]
    return _with(s, terms=terms, bra=not s.bra)


def merge(s: State, i: int, j: int) -> State:
    """Registers ``i..j-1``, of one dimension, joined into one big-endian register."""
    if j - i <= 1:
        return s
    widths, dims = s.widths[i:j], s.dims[i:j]
    if any(w is None for w in widths) or any(d is None for d in dims):
        raise PathSumError("a register width or dimension is unresolved")
    d = dims[0]
    if not all(same(x, d) for x in dims):
        raise PathSumError("cannot join registers of different dimensions")
    shifts = [sympy.Add(*widths[k + 1 :]) for k in range(len(widths))]
    terms = []
    for t in s.terms:
        value = sympy.Add(*(v * d**sh for v, sh in zip(t.kets[i:j], shifts, strict=True)))
        terms.append(Term(t.amp, t.bound, t.kets[:i] + (value,) + t.kets[j:]))
    return _with(
        s,
        widths=s.widths[:i] + (sympy.Add(*widths),) + s.widths[j:],
        terms=terms,
        dims=s.dims[:i] + (d,) + s.dims[j:],
    )


def _full_range(b: Bound, width: Expr | None, dim: Expr | None) -> bool:
    return width is not None and dim is not None and b.lo == 0 and same(b.hi, dim**width - 1)


def split(s: State, i: int, high: Expr) -> State:
    """Register ``i`` split into a ``high``-qudit register and the remaining low qudits."""
    width, d = s.widths[i], s.dims[i]
    if width is None or d is None:
        raise PathSumError("a register width or dimension is unresolved")
    low = expand(width - high)
    terms = []
    for t in s.terms:
        v = t.kets[i]
        bound = next((b for b in t.bound if b.symbol == v), None)
        if bound is not None and _full_range(bound, width, d):
            hb, lb = new_bound(v.name, 0, d**high - 1), new_bound(v.name, 0, d**low - 1)
            t = _subs_term(t, {v: hb.symbol * d**low + lb.symbol})
            t = Term(t.amp, t.bound + (hb, lb), t.kets)
            parts: tuple[Expr, ...] = (hb.symbol, lb.symbol)
        else:
            parts = (shr(v, low, d), sympy.Mod(v, d**low))
        terms.append(Term(t.amp, t.bound, t.kets[:i] + parts + t.kets[i + 1 :]))
    return _with(
        s,
        widths=s.widths[:i] + (high, low) + s.widths[i + 1 :],
        terms=terms,
        dims=s.dims[:i] + (d, d) + s.dims[i + 1 :],
    )


def _ranges(term: Term) -> list[range]:
    out = []
    for b in term.bound:
        if not (b.lo.is_Integer and b.hi.is_Integer):
            raise PathSumError("a bound range is symbolic")
        out.append(range(int(b.lo), int(b.hi) + 1))
    return out


def expand_terms(s: State) -> Iterator[tuple[Expr, tuple[Expr, ...]]]:
    """Every ``(amp, kets)`` of ``s`` with its bound indices enumerated; ranges concrete."""
    for t in s.terms:
        syms = [b.symbol for b in t.bound]
        for point in itertools.product(*_ranges(t)):
            mapping = {sym: sympy.Integer(v) for sym, v in zip(syms, point, strict=True)}
            yield t.amp.xreplace(mapping), tuple(k.xreplace(mapping) for k in t.kets)


def _prime_powers(b: Expr) -> dict[int, Expr] | None:
    """``b`` as ``{prime: exponent}`` when it is a positive product of rational powers."""
    out: dict[int, Expr] = {}
    for factor in sympy.Mul.make_args(b):
        base, e = factor.as_base_exp()
        if not (base.is_Rational and base > 0 and e.is_Rational):
            return None
        for p, k in sympy.factorint(base.p).items():
            out[p] = out.get(p, 0) + k * e
        for p, k in sympy.factorint(base.q).items():
            out[p] = out.get(p, 0) - k * e
    return out


def denest(expr: Expr) -> Expr:
    """``(b^e1)^e2`` as ``b^(e1 e2)`` when b is non-negative or e2 is an integer."""

    def ok(p: sympy.Basic) -> bool:
        return bool(p.is_Pow and p.base.is_Pow and (p.base.base.is_nonnegative or p.exp.is_integer))

    return expr.replace(ok, lambda p: p.base.base ** expand(p.base.exp * p.exp))


def tidy(expr: Expr) -> Expr:
    """``expr`` with numeric bases of symbolic powers split into prime powers."""

    def rewrite(p: sympy.Pow) -> Expr:
        primes = _prime_powers(p.base)
        if primes is None:
            return p
        return sympy.Mul(*(sympy.Integer(q) ** expand(k * p.exp) for q, k in primes.items()))

    expr = sympy.powsimp(denest(sympy.sympify(expr)), combine="exp")
    return expr.replace(
        lambda e: e.is_Pow and e.base.is_number and not e.exp.is_number and e.base != -1,
        rewrite,
    )


def _digitwise(
    expr: Expr, n: Expr, a: sympy.Symbol | None, v: Expr, y: sympy.Symbol | None, yv: Expr, d: Expr
) -> Expr | None:
    """``prod_j expr(a = v_j, y = Y_j)`` over n base-d digits; None without a closed form."""
    syms = {s for s in (a, y) if s is not None}
    out = sympy.S.One
    for f in sympy.Mul.make_args(sympy.powsimp(expr)):
        if not f.free_symbols & syms:
            out *= f**n
            continue
        if isinstance(f, sympy.exp):
            base, e = sympy.E, f.args[0]
        elif f.is_Pow and not f.base.free_symbols & syms:
            base, e = f.base, f.exp
        else:
            return None
        gens = [s for s in (a, y) if s is not None]
        try:
            poly = sympy.Poly(expand(e), *gens)
        except sympy.PolynomialError:
            return None
        coeffs = {
            (0, 0): sympy.S.Zero,
            (1, 0): sympy.S.Zero,
            (0, 1): sympy.S.Zero,
            (1, 1): sympy.S.Zero,
        }
        for monom, c in poly.terms():
            key = (monom[0], monom[1] if len(monom) > 1 else 0) if a is not None else (0, monom[0])
            if key not in coeffs or any(m > 1 for m in monom):
                return None
            coeffs[key] = c
        integral = all(coeffs[k].is_integer for k in ((1, 0), (0, 1), (1, 1)))
        if base != sympy.E and not base.is_positive and not integral:
            return None
        exponent = (
            coeffs[(1, 0)] * dsum(v, d)
            + coeffs[(0, 1)] * dsum(yv, d)
            + coeffs[(1, 1)] * ddot(v, yv, d)
        )
        if coeffs[(0, 0)] != 0:
            out *= (base ** coeffs[(0, 0)]) ** n
        out *= base**exponent
    return out


def _repeat_digit(k: Expr, d: Expr, n: Expr) -> Expr:
    if k == 0:
        return sympy.S.Zero
    return sympy.factor(k * (d**n - 1) / (d - 1)) if d != 2 else k * (2**n - 1)


def power_state(s: State, n: Expr, hint: str = "x") -> State:
    """``s^(x)n``; closed form for one-qudit states, repeated tensor for concrete n."""
    n = sympy.sympify(n)
    if len(s.widths) == 1 and same(s.widths[0], 1):
        out = _power_one(simplify(s), n, hint)
        if out is not None:
            return out
    if n.is_Integer and n >= 0:
        acc = State((), (Term(sympy.S.One, (), ()),), s.bra)
        for _ in range(int(n)):
            acc = tensor(acc, s)
        return acc
    raise PathSumError("a symbolic tensor power needs a one-qudit state with a closed form")


def _power_one(s: State, n: Expr, hint: str) -> State | None:
    d = s.dims[0]
    assert d is not None
    if len(s.terms) == 1:
        t = s.terms[0]
        if not t.bound:
            value = _repeat_digit(t.kets[0], d, n)
            return State((n,), (Term(tidy(t.amp**n), (), (value,)),), s.bra, (d,))
        if len(t.bound) == 1 and t.kets[0] == t.bound[0].symbol and _full_range(t.bound[0], 1, d):
            y = new_bound(hint, 0, d**n - 1)
            amp = _digitwise(t.amp, n, None, sympy.S.Zero, t.bound[0].symbol, y.symbol, d)
            if amp is not None:
                return State((n,), (Term(tidy(amp), (y,), (y.symbol,)),), s.bra, (d,))
        return None
    if not (d.is_Integer and is_concrete(s)):
        return None
    row = [sympy.S.Zero] * int(d)
    for amp, kets in expand_terms(s):
        row[int(kets[0])] += amp
    row = [sympy.simplify(x) for x in row]
    nonzero = [k for k, x in enumerate(row) if x != 0]
    if len(nonzero) == 1:
        k = nonzero[0]
        return State(
            (n,),
            (Term(tidy(row[k] ** n), (), (_repeat_digit(sympy.Integer(k), d, n),)),),
            s.bra,
            (d,),
        )
    if len(nonzero) != len(row):
        return None
    r = sympy.simplify(row[1] / row[0])
    if any(sympy.simplify(row[k] - row[0] * r**k) != 0 for k in range(len(row))):
        return None
    y = new_bound(hint, 0, d**n - 1)
    amp = tidy(row[0] ** n * r ** dsum(y.symbol, d))
    return State((n,), (Term(amp, (y,), (y.symbol,)),), s.bra, (d,))


class Operator:
    """A linear map on registers; see :func:`apply`."""


@dataclass(frozen=True, eq=False)
class Gate(Operator):
    """``fn(values, widths, dims) -> State`` on ``arity`` registers; ``qubit`` gates take single
    qubits; an ``oracle`` gate that maps basis states to basis states is checked reversible."""

    name: str
    arity: int
    fn: GateFn
    qubit: bool = False
    oracle: bool = False
    source: str = ""
    checked: set[tuple[int, ...]] = field(default_factory=set)


@dataclass(frozen=True)
class Identity(Operator):
    """The identity on ``width`` qudits, or on whatever it is aligned with."""

    width: Expr | None = None


@dataclass(frozen=True)
class Power(Operator):
    """``op^(x)count``."""

    op: Operator
    count: Expr


@dataclass(frozen=True)
class Tensor(Operator):
    """``f_1 (x) f_2 (x) ...``."""

    factors: tuple[Operator, ...]


@dataclass(frozen=True)
class Compose(Operator):
    """``ops[0] ops[1] ...``, applied right to left."""

    ops: tuple[Operator, ...]


@dataclass(frozen=True)
class Outer(Operator):
    """``|ket><bra|``."""

    ket: State
    bra: State


@dataclass(frozen=True)
class Scaled(Operator):
    """``factor * op``."""

    factor: Expr
    op: Operator


@dataclass(frozen=True)
class OpSum(Operator):
    """``ops[0] + ops[1] + ...``."""

    ops: tuple[Operator, ...]


def apply(op: Operator, s: State) -> State:
    """``op |s>`` over all registers of the ket ``s``."""
    if s.bra:
        raise PathSumError("operators act on kets; apply to |psi>, not <psi|")
    token = _SCOPE.set(Params(free_symbols(s))) if _SCOPE.get() is None else None
    try:
        return simplify(_resolved(_apply(op, s, 0, len(s.widths))))
    finally:
        if token is not None:
            _SCOPE.reset(token)


SliceFn: TypeAlias = Callable[[tuple[Expr, ...], tuple[Expr, ...], tuple[Expr, ...]], State]


def _apply_map(s: State, i: int, j: int, fn: SliceFn) -> State:
    out: State | None = None
    terms: list[Term] = []
    for t in s.terms:
        res = fn(t.kets[i:j], s.widths[i:j], s.dims[i:j])
        if out is None:
            out = res
        elif not _same_layout(res, out):
            raise PathSumError("an operator gave different output registers on different terms")
        for r in (fresh(x) for x in res.terms):
            terms.append(Term(t.amp * r.amp, t.bound + r.bound, t.kets[:i] + r.kets + t.kets[j:]))
    assert out is not None
    return _with(
        s,
        widths=s.widths[:i] + out.widths + s.widths[j:],
        terms=terms,
        dims=s.dims[:i] + out.dims + s.dims[j:],
    )


def _apply(op: Operator, s: State, i: int, j: int) -> State:
    if isinstance(op, Gate):
        return _apply_gate(op, s, i, j)
    if isinstance(op, Identity):
        if op.width is not None and not bind(op.width, sympy.Add(*s.widths[i:j])):
            raise PathSumError(f"I^@{op.width} does not match the register width")
        return s
    if isinstance(op, Power):
        return _apply_power(op, s, i, j)
    if isinstance(op, Tensor):
        s, segments = _align(op.factors, s, i, j)
        for f, a, b in reversed(segments):
            s = _apply(f, s, a, b)
        return s
    if isinstance(op, Compose):
        for o in reversed(op.ops):
            before = len(s.widths)
            s = _apply(o, s, i, j)
            j += len(s.widths) - before
        return s
    if isinstance(op, Outer):
        bra = op.bra
        return _apply_map(
            s, i, j, lambda vals, ws, ds: scale(inner(bra, basis_kets(vals, ws, ds)), op.ket)
        )
    if isinstance(op, Scaled):
        return scale(op.factor, _apply(op.op, s, i, j))
    if isinstance(op, OpSum):
        results = [_apply(o, s, i, j) for o in op.ops]
        out = results[0]
        for r in results[1:]:
            out = add(out, r)
        return out
    raise PathSumError(f"not an operator: {op!r}")


def _apply_gate(g: Gate, s: State, i: int, j: int) -> State:
    if g.arity == 1 and j - i > 1:
        s, j = merge(s, i, j), i + 1
    if j - i != g.arity:
        raise PathSumError(f"{g.name} acts on {g.arity} register(s), got {j - i}")
    if g.qubit and not all(
        same(w, 1) and same(d, 2) for w, d in zip(s.widths[i:j], s.dims[i:j], strict=True)
    ):
        raise PathSumError(f"{g.name} acts on single qubits; use {g.name}^@w for a register")

    classical = [True]

    def run(vals: tuple[Expr, ...], ws: tuple[Expr, ...], ds: tuple[Expr, ...]) -> State:
        res = _resolved(g.fn(vals, ws, ds))
        if g.oracle and classical[0]:
            classical[0] = (
                len(res.terms) == 1
                and not res.terms[0].bound
                and sympy.simplify(res.terms[0].amp) == 1
            )
        return res

    out = _apply_map(s, i, j, run)
    if g.oracle and classical[0]:
        _check_oracle(g, s.widths[i:j], s.dims[i:j], run)
    return out


def _check_oracle(
    g: Gate, widths: Sequence[Expr | None], dims: Sequence[Expr | None], run: SliceFn
) -> None:
    """Raises unless the oracle permutes basis states; concrete widths and dimensions only."""
    pairs = list(zip(widths, dims, strict=True))
    if not all(w is not None and w.is_Integer and d is not None and d.is_Integer for w, d in pairs):
        return
    key = tuple(int(x) for pair in pairs for x in pair)  # type: ignore[arg-type]
    if key in g.checked:
        return
    bounds = [new_bound("x", 0, d**w - 1) for w, d in pairs]  # type: ignore[operator]
    probe = State(
        tuple(widths),
        (Term(sympy.S.One, tuple(bounds), tuple(b.symbol for b in bounds)),),
        False,
        tuple(dims),
    )
    out = _apply_map(probe, 0, len(pairs), run)
    if not is_concrete(out):
        return
    keys, _ = enumerate_state(out)
    if len(np.unique(keys)) != len(keys):
        raise PathSumError(f"oracle {g.name} is not reversible on widths {tuple(widths)}")
    g.checked.add(key)


def _shift_amount(val: Expr, a: sympy.Symbol, d: Expr) -> Expr | None:
    if isinstance(val, sympy.Mod) and same(val.args[1], d):
        k = expand(val.args[0] - a)
        if a not in k.free_symbols:
            return k
    return None


def _power_gate(g: Gate, v: Expr, w: Expr, d: Expr, hint: str) -> State | None:
    """``g^(x)w |v>`` in closed form, or None."""
    a = sympy.Dummy("a", integer=True, nonnegative=True)
    res = simplify(_resolved(g.fn((a,), (sympy.S.One,), (d,))))
    if len(res.terms) != 1 or len(res.widths) != 1:
        return None
    t = res.terms[0]
    if not t.bound:
        amp = _digitwise(t.amp, w, a, v, None, sympy.S.Zero, d)
        val = t.kets[0]
        if amp is None:
            return None
        if val == a:
            out = v
        elif same(val, d - 1 - a):
            out = d**w - 1 - v
        elif (k := _shift_amount(val, a, d)) is not None:
            out = dshift(v, k, d, w)
        else:
            return None
        return State((w,), (Term(tidy(amp), (), (out,)),), False, (d,))
    if len(t.bound) == 1 and t.kets[0] == t.bound[0].symbol and _full_range(t.bound[0], 1, d):
        y = new_bound(hint, 0, d**w - 1)
        amp = _digitwise(t.amp, w, a, v, t.bound[0].symbol, y.symbol, d)
        if amp is not None:
            return State((w,), (Term(tidy(amp), (y,), (y.symbol,)),), False, (d,))
    return None


def _apply_power(p: Power, s: State, i: int, j: int) -> State:
    inner_op = p.op
    if isinstance(inner_op, Identity):
        return s
    count = sympy.sympify(p.count)
    if isinstance(inner_op, Gate) and inner_op.arity == 1:
        s = merge(s, i, j)
        width, d = s.widths[i], s.dims[i]
        if not bind(count, width):
            raise PathSumError(f"{inner_op.name}^@{count} does not match register width {width}")
        count = resolve(count)
        if inner_op.qubit and not same(d, 2):
            raise PathSumError(f"{inner_op.name} acts on qubits, not dimension {d}")
        assert d is not None
        probe = _power_gate(inner_op, sympy.Dummy("v", integer=True), count, d, "x")
        if probe is not None:

            def closed(vals: tuple[Expr, ...], ws: tuple[Expr, ...], ds: tuple[Expr, ...]) -> State:
                out = _power_gate(inner_op, vals[0], ws[0], ds[0], "x")
                assert out is not None
                return out

            return _apply_map(s, i, i + 1, closed)
        if count.is_Integer and count >= 0:
            return _apply(Tensor((inner_op,) * int(count)), s, i, i + 1)
        raise PathSumError(f"no closed form for the symbolic tensor power of {inner_op.name}")
    if count.is_Integer and count >= 0:
        return _apply(Tensor((inner_op,) * int(count)), s, i, j)
    raise PathSumError("a symbolic tensor power needs a one-register gate")


def _span_width(op: Operator) -> Expr | None:
    if isinstance(op, Power):
        if isinstance(op.op, Gate) and op.op.arity == 1:
            return sympy.sympify(op.count)
        inner_w = _span_width(op.op)
        return None if inner_w is None else expand(inner_w * op.count)
    if isinstance(op, Identity):
        return op.width
    if isinstance(op, Gate) and op.qubit:
        return sympy.Integer(op.arity)
    if isinstance(op, Outer):
        return op.bra.total_width
    if isinstance(op, Scaled):
        return _span_width(op.op)
    if isinstance(op, Compose):
        return _span_width(op.ops[-1])
    if isinstance(op, OpSum):
        return _span_width(op.ops[0])
    if isinstance(op, Tensor):
        parts = [_span_width(f) for f in op.factors]
        return None if any(w is None for w in parts) else sympy.Add(*parts)
    return None


def _align(
    factors: Sequence[Operator], s: State, i: int, j: int
) -> tuple[State, list[tuple[Operator, int, int]]]:
    """Register segments ``(factor, start, end)`` covering ``i..j``, splitting as needed."""
    flat: list[Operator] = []
    for f in factors:
        flat.extend(f.factors if isinstance(f, Tensor) else [f])
    segments: list[tuple[Operator, int, int]] = []
    k = i
    for idx, f in enumerate(flat):
        last = idx == len(flat) - 1
        width = _span_width(f)
        if width is None:
            if isinstance(f, Identity) and last:
                end = j
            elif isinstance(f, Gate):
                end = k + f.arity
            else:
                end = k + 1
        else:
            s, end, j = _consume(s, k, j, width)
        if end > j or (end == k and not (isinstance(f, Identity) and last)):
            raise PathSumError("the operator does not cover the state's registers")
        segments.append((f, k, end))
        k = end
    if k != j:
        raise PathSumError(f"the operator covers {k - i} of {j - i} registers")
    return s, segments


def _consume(s: State, k: int, j: int, width: Expr) -> tuple[State, int, int]:
    acc: Expr = sympy.S.Zero
    r = k
    while r < j:
        w = s.widths[r]
        if w is None:
            raise PathSumError("a register width is unresolved")
        acc = expand(acc + w)
        r += 1
        if same(acc, width):
            return s, r, j
        over = expand(acc - width)
        if over.is_Integer and over > 0:
            s = split(s, r - 1, expand(w - over))
            return s, r, j + 1
    raise PathSumError(f"no registers of total width {width} to act on")


def _bound_ok(sym: Expr, other: Expr, bound: list[Bound], width: Expr, dim: Expr) -> bool:
    b = next((x for x in bound if x.symbol == sym), None)
    return b is not None and _full_range(b, width, dim) and sym not in other.free_symbols


def inner(a: State, b: State) -> Expr:
    """``<a|b>`` for a bra ``a`` and a ket ``b``."""
    if not a.bra or b.bra:
        raise PathSumError("an inner product needs a bra then a ket")
    if not _same_layout(a, b):
        if not (same(a.total_width, b.total_width) and same(a.uniform_dim(), b.uniform_dim())):
            raise PathSumError("cannot contract states of different widths or dimensions")
        a, b = merge(a, 0, len(a.widths)), merge(b, 0, len(b.widths))
    if enumerable(a) and enumerable(b):
        return _inner_numeric(a, b)
    total: Expr = sympy.S.Zero
    for ta in a.terms:
        for tb in (fresh(t) for t in b.terms):
            total += _contract(ta, tb, a.widths, a.dims)
    return sympy.simplify(total)


def _contract(
    ta: Term, tb: Term, widths: Sequence[Expr | None], dims: Sequence[Expr | None]
) -> Expr:
    amp = ta.amp * tb.amp
    bound = list(ta.bound + tb.bound)
    eqs = list(zip(ta.kets, tb.kets, widths, dims, strict=True))
    deltas: list[Expr] = []
    while eqs:
        lhs, rhs, w, d = eqs.pop()
        if expand(lhs - rhs) == 0:
            continue
        pick = next(
            ((x, y) for x, y in ((lhs, rhs), (rhs, lhs)) if _bound_ok(x, y, bound, w, d)), None
        )
        if pick is not None:
            sym, val = pick
            mapping = {sym: val}
            amp = amp.xreplace(mapping)
            eqs = [(x.xreplace(mapping), y.xreplace(mapping), ww, dd) for x, y, ww, dd in eqs]
            deltas = [e.xreplace(mapping) for e in deltas]
            bound = [x for x in bound if x.symbol != sym]
            continue
        if lhs.is_Integer and rhs.is_Integer:
            return sympy.S.Zero
        deltas.append(sympy.KroneckerDelta(lhs, rhs))
    return _sum_out(amp * sympy.Mul(*deltas), bound)


def _geometric(amp: Expr, b: Bound) -> Expr | None:
    """``sum_b amp`` for ``amp = C e^{2 pi i b L / R}`` over ``0..R-1``; else None."""
    if b.lo != 0:
        return None
    sym = b.symbol
    exps = [f for f in sympy.Mul.make_args(amp) if isinstance(f, sympy.exp)]
    rest = sympy.Mul(*(f for f in sympy.Mul.make_args(amp) if not isinstance(f, sympy.exp)))
    if sym in rest.free_symbols or not exps:
        return None
    arg = expand(sympy.Add(*(e.args[0] for e in exps)) / (2 * sympy.pi * sympy.I))
    coeff = expand(sympy.diff(arg, sym))
    if sym in coeff.free_symbols or coeff.has(sympy.Derivative):
        return None
    count = b.hi + 1
    level = expand(coeff * count)
    if not level.is_integer:
        return None
    level = sympy.Add(*(p for p in sympy.Add.make_args(level) if not (p / count).is_integer))
    other = sympy.exp(2 * sympy.pi * sympy.I * expand(arg - coeff * sym))
    parts = sympy.Add.make_args(level)
    signs = [p.as_coeff_Mul() for p in parts]
    if not level.is_Integer and len(parts) <= 2 and all(abs(c) == 1 for c, _ in signs):
        pos = [x for c, x in signs if c == 1]
        neg = [x for c, x in signs if c == -1]
        if len(parts) == 1:
            return rest * other * count * sympy.KroneckerDelta(parts[0] * signs[0][0], 0)
        if len(pos) == 1 and len(neg) == 1:
            return rest * other * count * sympy.KroneckerDelta(pos[0], neg[0])
    terms = [p for p in parts if not p.is_Integer]
    const = sympy.Add(*(p for p in parts if p.is_Integer))
    if len(terms) == 1 and const != 0 and abs(terms[0].as_coeff_Mul()[0]) == 1:
        c, u = terms[0].as_coeff_Mul()
        target = -c * const
        if target.is_negative:
            target = expand(count + target)
        return rest * other * count * sympy.KroneckerDelta(u, target)
    if level.is_Integer and count.is_Integer:
        return rest * other * (count if int(level) % int(count) == 0 else 0)
    return None


def _sum_out(amp: Expr, bound: list[Bound]) -> Expr:
    for b in bound:
        if b.symbol not in amp.free_symbols:
            amp = amp * (b.hi - b.lo + 1)
            continue
        geo = _geometric(amp, b)
        amp = geo if geo is not None else sympy.Sum(amp, (b.symbol, b.lo, b.hi)).doit()
    return amp


def _character(f: Expr, sym: sympy.Symbol) -> Expr | None:
    """The partner v of a factor ``(-1)^bitdot(sym, v)`` or ``e^{+-2 pi i ddot(sym, v, d)/d}``."""
    if f.is_Pow and f.base == -1 and isinstance(f.exp, bitdot) and sym in f.exp.args:
        return next((x for x in f.exp.args if x != sym), sym)
    if isinstance(f, sympy.exp):
        hits = [x for x in f.args[0].atoms(ddot) if sym in x.args[:2]]
        if len(hits) == 1:
            dd = hits[0]
            c = sympy.simplify(f.args[0] / dd * dd.args[2] / (2 * sympy.pi * sympy.I))
            if c in (1, -1):
                return next((x for x in dd.args[:2] if x != sym), sym)
    return None


def _parity(amp: Expr, b: Bound, bound: Sequence[Bound]) -> Expr | None:
    """``sum_b C chi(b, v)`` over a full ``d^w`` range, for a digit character chi, as
    ``d^w C [v = 0]``; else None."""
    sym = b.symbol
    if b.lo != 0:
        return None
    factors = sympy.Mul.make_args(amp)
    hits = [(f, v) for f in factors if (v := _character(f, sym)) is not None]
    if len(hits) != 1:
        return None
    hit, v = hits[0]
    rest = sympy.Mul(*(f for f in factors if f is not hit))
    if sym in rest.free_symbols or v == sym:
        return None
    partner = next((x for x in bound if x.symbol == v), None)
    if not (v.is_Integer or (partner is not None and partner.lo == 0 and same(partner.hi, b.hi))):
        return None
    return rest * (b.hi + 1) * sympy.KroneckerDelta(v, 0)


def _eliminate_deltas(t: Term) -> Term:
    """``[i = c]`` factors with ``i`` a bound index and ``c`` in its range substituted."""
    changed = True
    while changed:
        changed = False
        for f in sympy.Mul.make_args(t.amp):
            if not isinstance(f, sympy.KroneckerDelta):
                continue
            for sym, val in (f.args, f.args[::-1]):
                b = next((x for x in t.bound if x.symbol == sym), None)
                if b is None or sym in val.free_symbols:
                    continue
                other = next((x for x in t.bound if x.symbol == val), None)
                inside = (
                    (val.is_Integer and b.lo.is_Integer and b.hi.is_Integer and b.lo <= val <= b.hi)
                    or val == b.lo
                    or (other is not None and same(other.lo, b.lo) and same(other.hi, b.hi))
                    or same(val, b.hi)
                )
                if inside:
                    t = _subs_term(t, {sym: val})
                    changed = True
                    break
            if changed:
                break
    return t


def _reduce_mods(t: Term) -> Term:
    """``Mod(b, D)`` with b a bound index over ``0..D-1`` replaced by b."""
    ranges = {b.symbol: b for b in t.bound if b.lo == 0}

    def plain(e: sympy.Mod) -> bool:
        b = ranges.get(e.args[0])
        return b is not None and same(b.hi + 1, e.args[1])

    def rewrite(e: Expr) -> Expr:
        return e.replace(lambda x: isinstance(x, sympy.Mod) and plain(x), lambda x: x.args[0])

    return Term(rewrite(t.amp), t.bound, tuple(rewrite(k) for k in t.kets))


def simplify(s: State) -> State:
    """Terms tidied, zero terms dropped, and bound indices outside every ket summed out."""
    terms = []
    for t in s.terms:
        t = _reduce_mods(t)
        amp = tidy(t.amp)
        bound = list(t.bound)
        for b in list(bound):
            if any(b.symbol in k.free_symbols for k in t.kets):
                continue
            if b.symbol not in amp.free_symbols:
                amp = amp * (b.hi - b.lo + 1)
                bound.remove(b)
                continue
            geo = _geometric(amp, b)
            if geo is None:
                geo = _parity(amp, b, bound)
            if geo is not None:
                amp = geo
                bound.remove(b)
        t = _eliminate_deltas(Term(amp, tuple(bound), t.kets))
        amp = tidy(t.amp)
        if amp == 0:
            continue
        terms.append(Term(amp, t.bound, t.kets))
    if not terms:
        terms = [Term(sympy.S.Zero, (), s.terms[0].kets if s.terms else ())]
    out = _with(s, terms=terms)
    listable = len(terms) > 1 and is_concrete(out) and paths(out) <= LIST_LIMIT
    return _combine(out) if listable else out


def _combine(s: State) -> State:
    """A concrete state as one exact term per basis ket, in basis order."""
    acc: dict[tuple[int, ...], Expr] = {}
    for amp, kets in expand_terms(s):
        key = tuple(int(k) for k in kets)
        acc[key] = acc.get(key, sympy.S.Zero) + amp
    terms = []
    for key in sorted(acc):
        amp = sympy.simplify(acc[key])
        if amp != 0:
            terms.append(Term(amp, (), tuple(sympy.Integer(k) for k in key)))
    if not terms:
        terms = [Term(sympy.S.Zero, (), s.terms[0].kets)]
    return _with(s, terms=terms)


def is_concrete(s: State) -> bool:
    """Every width, dimension, range, amplitude and value is a number once bound indices are
    fixed."""
    for x in (*s.widths, *s.dims):
        if x is None or not sympy.sympify(x).is_Integer:
            return False
    for t in s.terms:
        syms = {b.symbol for b in t.bound}
        for b in t.bound:
            if not (b.lo.is_Integer and b.hi.is_Integer):
                return False
        for e in (t.amp, *t.kets):
            if not e.free_symbols <= syms:
                return False
            for app in e.atoms(sympy.Function):
                if getattr(type(app), "_free", None):
                    return False
    return True


def paths(s: State) -> int:
    """The number of basis paths of ``s``; its bound ranges are concrete."""
    return sum(math.prod(max(0, int(b.hi) - int(b.lo) + 1) for b in t.bound) for t in s.terms)


def enumerable(s: State) -> bool:
    """Whether ``s`` is concrete with at most :data:`PATH_LIMIT` basis paths."""
    return is_concrete(s) and paths(s) <= PATH_LIMIT


def _grid(term: Term) -> tuple[list[np.ndarray], int]:
    ranges = _ranges(term)
    if not ranges:
        return [], 1
    axes = [np.arange(r.start, r.stop, dtype=object) for r in ranges]
    mesh = np.meshgrid(*axes, indexing="ij")
    size = int(np.prod([len(r) for r in ranges]))
    return [m.reshape(-1) for m in mesh], size


def _evaluate(
    expr: Expr, syms: list[sympy.Symbol], grid: list[np.ndarray], size: int, dtype: object
) -> np.ndarray:
    expr = to_powmod(expr)
    if dtype is object:
        exact = sympy.lambdify(syms, expr, modules=[scalar_impls(expr), "math"])
        if not syms:
            return np.full(size, int(exact()), dtype=object)
        out = np.frompyfunc(exact, len(syms), 1)(*grid)
        return np.broadcast_to(out, (size,)).copy() if np.shape(out) != (size,) else out
    fn = sympy.lambdify(syms, expr, modules=[numeric_impls(expr), "numpy"])
    out = np.asarray(fn(*[_as_numeric(g) for g in grid]), dtype=dtype)  # type: ignore[call-overload]
    return np.broadcast_to(out, (size,)).copy() if out.shape != (size,) else out


def _as_numeric(g: np.ndarray) -> np.ndarray:
    if g.size and max(abs(int(g.max())), abs(int(g.min()))) < 2**62:
        return g.astype(np.int64)
    return g.astype(float)


def _suffixes(levels: Sequence[int]) -> list[int]:
    out, acc = [], 1
    for lv in reversed(levels):
        out.append(acc)
        acc *= lv
    return out[::-1]


def concrete_levels(s: State) -> list[int]:
    """``d^w`` per register of a concrete state."""
    return [int(s.levels(k)) for k in range(len(s.widths))]


def enumerate_state(s: State) -> tuple[np.ndarray, np.ndarray]:
    """``(keys, amps)`` for every basis path of a concrete state; keys are mixed-radix
    concatenations of the registers."""
    shifts = _suffixes(concrete_levels(s))
    all_keys, all_amps = [], []
    for t in s.terms:
        syms = [b.symbol for b in t.bound]
        grid, size = _grid(t)
        key = np.zeros(size, dtype=object)
        for k, sh in zip(t.kets, shifts, strict=True):
            key = key + _evaluate(k, syms, grid, size, object) * sh
        all_keys.append(key)
        all_amps.append(_evaluate(t.amp, syms, grid, size, complex))
    keys = np.concatenate(all_keys) if all_keys else np.zeros(0, dtype=object)
    amps = np.concatenate(all_amps) if all_amps else np.zeros(0, dtype=complex)
    return keys, amps


def _accumulate(s: State) -> tuple[np.ndarray, np.ndarray]:
    keys, amps = enumerate_state(s)
    uniq, inv = np.unique(keys, return_inverse=True)
    acc = np.zeros(len(uniq), dtype=complex)
    np.add.at(acc, inv.reshape(-1), amps)
    return uniq, acc


def _inner_numeric(a: State, b: State) -> Expr:
    ka, aa = _accumulate(a)
    kb, ab = _accumulate(b)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    value = complex(np.sum(aa[ia] * ab[ib])) if len(common) else 0j
    re, im = (x if abs(x) > 1e-12 else 0.0 for x in (value.real, value.imag))
    exact = [
        sympy.nsimplify(x, [sympy.pi], tolerance=1e-10) if x else sympy.S.Zero for x in (re, im)
    ]
    return exact[0] + sympy.I * exact[1]


def decode(key: int, levels: Sequence[int]) -> tuple[sympy.Integer, ...]:
    """Register values of a mixed-radix key."""
    return tuple(
        sympy.Integer((key // sh) % lv) for lv, sh in zip(levels, _suffixes(levels), strict=True)
    )


@dataclass(frozen=True)
class Measurement:
    """A measured outcome, the post-measurement state and the facts the outcome satisfies."""

    outcome: Expr
    width: Expr
    post: State
    constraints: tuple[tuple[Expr, Expr], ...] = ()
    random: bool = False


def measure(
    s: State,
    symbol: sympy.Symbol,
    rng: np.random.Generator,
    facts: Sequence[tuple[Expr, Expr]] = (),
) -> Measurement:
    """Measures every register of the ket ``s``; ``symbol`` names a symbolic outcome.

    ``facts`` are ``(slice, value)`` pairs from earlier outcomes, used to express the outcome
    of an already-collapsed state through those outcomes.
    """
    if s.bra:
        raise PathSumError("measure a ket, not a bra")
    s = simplify(s)
    width = s.total_width
    if len(s.terms) == 1 and not s.terms[0].bound:
        values = s.terms[0].kets
        named = tuple(v.xreplace({b: a for a, b in facts}) for v in values)
        return Measurement(_concat(named, s), width, basis_kets(values, s.widths, s.dims))
    if enumerable(s):
        return _measure_numeric(s, rng)
    return _measure_symbolic(s, symbol)


def _slices(s: State, z: Expr) -> list[Expr]:
    total = s.total_width
    d = s.uniform_dim()
    if d is not None:
        offsets = [sympy.Add(*s.widths[:k]) for k in range(len(s.widths))]
        return [
            digits(z, off, expand(off + w), total, d)
            for off, w in zip(offsets, s.widths, strict=True)
        ]
    levels = [s.levels(k) for k in range(len(s.widths))]
    return [part(z, k, sympy.Mul(*levels[k + 1 :]), levels[k]) for k in range(len(levels))]


def _concat(values: Sequence[Expr], s: State) -> Expr:
    z = sympy.Dummy("z")
    pattern = _slices(s, z)
    heads = {v.args[0] for v in values if isinstance(v, (bits, digits, part))}
    if len(heads) == 1:
        head = next(iter(heads))
        if all(same(v, p.xreplace({z: head})) for v, p in zip(values, pattern, strict=True)):
            return head
    levels = [s.levels(k) for k in range(len(s.widths))]
    shifts = [sympy.Mul(*levels[k + 1 :]) for k in range(len(levels))]
    return sympy.Add(*(v * sh for v, sh in zip(values, shifts, strict=True)))


def _measure_numeric(s: State, rng: np.random.Generator) -> Measurement:
    keys, acc = _accumulate(s)
    probs = np.abs(acc) ** 2
    total = probs.sum()
    if total <= 0:
        raise PathSumError("cannot measure the zero vector")
    pick = int(rng.choice(len(keys), p=probs / total))
    key = int(keys[pick])
    values = decode(key, concrete_levels(s))
    post = basis_kets(values, s.widths, s.dims)
    return Measurement(sympy.Integer(key), s.total_width, post, random=True)


def _measure_symbolic(s: State, z: sympy.Symbol) -> Measurement:
    widths, dims = s.widths, s.dims
    total = s.total_width
    slices = _slices(s, z)
    if len(s.terms) != 1:
        return Measurement(z, total, basis_kets(slices, widths, dims))
    t = s.terms[0]
    mapping: dict[sympy.Basic, Expr] = {}
    for v, w, d, sl in zip(t.kets, widths, dims, slices, strict=True):
        b = next((x for x in t.bound if x.symbol == v), None)
        if b is not None and v not in mapping and _full_range(b, w, d):
            mapping[v] = sl
    bound_syms = {b.symbol for b in t.bound}
    values, constraints = [], []
    for v, sl in zip(t.kets, slices, strict=True):
        val = v.xreplace(mapping)
        if val.free_symbols & bound_syms:
            val = sl
        elif val != sl:
            constraints.append((sl, val))
        values.append(val)
    return Measurement(z, total, basis_kets(values, widths, dims), tuple(constraints))


def inline(expr: Expr) -> Expr:
    """``expr`` with user-function applications replaced by their bodies."""
    while True:
        apps = [a for a in expr.atoms(sympy.Function) if hasattr(type(a), "_body")]
        if not apps:
            return expr
        expr = expr.xreplace(
            {
                a: type(a)._body.xreplace(dict(zip(type(a)._params, a.args, strict=True)))
                for a in apps
            }
        )


def _multiplicative(w: Expr, u: sympy.Symbol) -> tuple[Expr, Expr] | None:
    """``(G, M)`` when ``w = Mod(u * G, M)`` (a ``u < M`` guard allowed), G free of u."""
    if isinstance(w, sympy.Piecewise) and len(w.args) == 2:
        (inside, cond), (outside, rest) = w.args
        if not (rest == sympy.true and outside == u and isinstance(cond, sympy.StrictLessThan)):
            return None
        if cond.args[0] != u:
            return None
        w = inside
        guard = cond.args[1]
    else:
        guard = None
    if not isinstance(w, sympy.Mod) or (guard is not None and not same(w.args[1], guard)):
        return None
    g = sympy.simplify(w.args[0] / u)
    if u in g.free_symbols:
        return None
    return g, w.args[1]


def _bit_sum(e: Expr, j: sympy.Symbol, lo: Expr, hi: Expr) -> Expr:
    """``sum_{j=lo}^{hi} e``, with ``sum_{j=0}^{hi} 2^j bit(x, j) = x mod 2^(hi+1)``."""
    e = expand(e)
    for b in e.atoms(bit):
        if b.args[1] == j and lo == 0 and sympy.simplify(e - 2**j * b) == 0:
            return sympy.Mod(b.args[0], 2 ** expand(hi + 1))
    return sympy.Sum(e, (j, lo, hi)).doit()


def _product_mod(g: Expr, j: sympy.Symbol, lo: Expr, hi: Expr, modulus: Expr) -> Expr:
    """``prod_{j=lo}^{hi} g`` up to congruence mod ``modulus``."""
    g = g.replace(
        lambda e: isinstance(e, powmod) and same(e.args[2], modulus),
        lambda e: e.args[0] ** e.args[1],
    )
    out = sympy.S.One
    for f in sympy.Mul.make_args(sympy.powsimp(denest(g))):
        if j not in f.free_symbols:
            out *= f ** expand(hi - lo + 1)
            continue
        base, e = f.as_base_exp()
        if j in base.free_symbols:
            return sympy.Product(g, (j, lo, hi))
        out *= base ** _bit_sum(e, j, lo, hi)
    return out


def _below(value: Expr, modulus: Expr) -> bool:
    """Whether a register value is known to lie in ``0..modulus-1`` (modulus at least 2)."""
    if isinstance(value, sympy.Mod) and same(value.args[1], modulus):
        return True
    if value.is_Integer and modulus.is_Integer:
        return 0 <= int(value) < int(modulus)
    return bool(value.is_Integer and int(value) in (0, 1))


def fold(
    cur: State, step: State, probes: Sequence[sympy.Symbol], j: sympy.Symbol, lo: Expr, hi: Expr
) -> State:
    """``cur`` after ``step`` for ``j = lo..hi``; ``step`` is one iteration on the probe basis
    state ``|probes>``. Registers must stay fixed or update as ``y -> y G(j) mod M``."""
    if step.bra or len(step.terms) != 1 or step.terms[0].bound or not _same_layout(cur, step):
        raise PathSumError("a symbolic loop step must map basis states to basis states")
    t = step.terms[0]
    updates: dict[int, tuple[Expr, Expr]] = {}
    for k, (u, w) in enumerate(zip(probes, t.kets, strict=True)):
        w = inline(w)
        if w == u:
            continue
        found = _multiplicative(w, u)
        if found is None:
            raise PathSumError(f"register {k} does not update as y -> y*G mod M in the loop")
        updates[k] = found
    moving = {probes[k] for k in updates}
    if any(g.free_symbols & moving for g, _ in updates.values()):
        raise PathSumError("loop updates of different registers depend on each other")
    amp = inline(t.amp)
    if amp.free_symbols & moving:
        raise PathSumError("the loop phase depends on a register the loop changes")
    terms = []
    for term in cur.terms:
        fixed = {probes[k]: v for k, v in enumerate(term.kets) if k not in updates}
        kets = list(term.kets)
        for k, (g, modulus) in updates.items():
            if not _below(term.kets[k], modulus):
                raise PathSumError(f"register {k} is not known to start below {modulus}")
            product = _product_mod(g.xreplace(fixed), j, lo, hi, modulus)
            kets[k] = sympy.Mod(term.kets[k] * product, modulus)
        total = sympy.Product(amp.xreplace(fixed), (j, lo, hi)).doit() if amp != 1 else 1
        terms.append(Term(term.amp * total, term.bound, tuple(kets)))
    return _with(cur, terms=terms)
