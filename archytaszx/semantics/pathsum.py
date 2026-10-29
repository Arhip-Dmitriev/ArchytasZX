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

:func:`distribution` gives the Born distribution of measuring every register, derived from
the path sum by substitution and closed-form sums (never by enumerating paths); a
:class:`Mixed` state holds a density matrix as its vectorisation. :func:`inner` contracts a
bra with a ket.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, TypeAlias

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
    """Lambdify namespace for the integer functions in ``expr``, user functions included,
    each returning machine integers (floats beyond them) for numpy arithmetic."""
    names = dict(_IMPLS)
    for app in expr.atoms(sympy.Function):
        impl = getattr(type(app), "_numeric", None)
        if impl is not None:
            names[type(app).__name__] = impl
    return {k: _machine(v) for k, v in names.items()}


def _machine(fn: Any) -> Callable[..., np.ndarray]:
    def run(*args: object) -> np.ndarray:
        out = np.asarray(fn(*args), dtype=object)
        flat = out.reshape(-1)
        if flat.size and max(abs(int(x)) for x in flat) < 2**62:
            return out.astype(np.int64)
        return out.astype(float)

    return run


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
_RANGES: ContextVar[dict[sympy.Symbol, tuple[Expr, Expr]] | None] = ContextVar(
    "pathsum_ranges", default=None
)
_DIVISIBLE: ContextVar[tuple[tuple[Expr, Expr], ...]] = ContextVar("pathsum_divisible", default=())


@contextmanager
def divisible(facts: Sequence[tuple[Expr, Expr]]) -> Iterator[None]:
    """A scope in which each ``(a, b)`` of ``facts`` has b dividing a."""
    token = _DIVISIBLE.set((*_DIVISIBLE.get(), *facts))
    try:
        yield
    finally:
        _DIVISIBLE.reset(token)


@contextmanager
def ranged(ranges: dict[sympy.Symbol, tuple[Expr, Expr]]) -> Iterator[None]:
    """A scope in which each symbol of ``ranges`` lies in its inclusive ``(lo, hi)``."""
    token = _RANGES.set({**(_RANGES.get() or {}), **ranges})
    try:
        yield
    finally:
        _RANGES.reset(token)


def _in_known_range(val: Expr, b: Bound) -> bool:
    """Whether ``val`` is a symbol of the running :func:`ranged` scope within ``b``'s range."""
    span = (_RANGES.get() or {}).get(val)
    if span is None:
        return False
    low, high = sympy.simplify(span[0] - b.lo), sympy.simplify(b.hi - span[1])
    return bool(low.is_nonnegative) and bool(high.is_nonnegative)


def _apart(a: Expr, b: Expr) -> bool:
    """Whether ``a - b`` has a known sign other than 0 over the running :func:`ranged` scope."""
    shift = {}
    for x, (lo, _) in (_RANGES.get() or {}).items():
        if lo.is_finite and x in (a - b).free_symbols:
            shift[x] = lo + sympy.Dummy(str(x), integer=True, nonnegative=True)
    if not shift:
        return False
    diff = sympy.simplify(sympy.sympify(a - b).xreplace(shift))
    return bool(diff.is_positive or diff.is_negative)


def _nonnegative(e: Expr) -> bool:
    """Whether ``e`` is known non-negative, its symbols shifted to the lower bounds of the
    running :func:`ranged` scope."""
    shift = {
        x: lo + sympy.Dummy(str(x), integer=True, nonnegative=True)
        for x, (lo, _) in (_RANGES.get() or {}).items()
        if lo.is_finite and x in sympy.sympify(e).free_symbols
    }
    return bool(sympy.simplify(sympy.sympify(e).xreplace(shift)).is_nonnegative)


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
        tuple(_subs_term(t, mapping) for t in s.terms),
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

    expr = sympy.sympify(expr)
    if expr.has(sympy.Piecewise):
        expr = expr.replace(lambda e: isinstance(e, sympy.Piecewise), piecewise_delta)
    expr = sympy.powsimp(denest(expr), combine="exp")
    if expr.has(sympy.KroneckerDelta):
        expr = expr.replace(_delta_power, _split_delta_power)
    expr = expr.replace(lambda e: isinstance(e, sympy.exp) and e.has(sympy.Mod), _unmod_phase)
    expr = expr.replace(
        lambda e: e.is_Pow and e.base.is_number and not e.exp.is_number and e.base != -1,
        rewrite,
    )
    return expr.replace(
        lambda e: e.is_Pow and e.base == -1 and not e.exp.is_number and _fraction(e.exp) != 0,
        lambda e: (
            sympy.Pow(-1, _fraction(e.exp), evaluate=True) * (-1) ** (e.exp - _fraction(e.exp))
        ),
    )


def _indicator(c: sympy.Basic) -> Expr | None:
    """``[c]`` as a delta expression for an equation or inequation of integers; else None."""
    if isinstance(c, (sympy.Eq, sympy.Ne)) and expand(c.lhs - c.rhs).is_integer:
        delta = sympy.KroneckerDelta(c.lhs, c.rhs)
        return delta if isinstance(c, sympy.Eq) else 1 - delta
    if isinstance(c, (sympy.And, sympy.Or, sympy.Not)):
        parts = [_indicator(a) for a in c.args]
        if any(p is None for p in parts):
            return None
        if isinstance(c, sympy.Not):
            return 1 - parts[0]
        if isinstance(c, sympy.And):
            return sympy.Mul(*parts)
        return 1 - sympy.Mul(*(1 - p for p in parts))
    return None


def piecewise_delta(p: sympy.Piecewise) -> Expr:
    """A Piecewise whose conditions are integer (in)equations as a sum of delta terms."""
    out: Expr = sympy.S.Zero
    rest: Expr = sympy.S.One
    for value, cond in p.args:
        if cond is sympy.true:
            return expand(out + rest * value)
        ind = _indicator(cond)
        if ind is None:
            return p
        out += rest * ind * value
        rest *= 1 - ind
    return p


def _delta_power(e: sympy.Basic) -> bool:
    ex = e.args[0] if isinstance(e, sympy.exp) else e.exp if e.is_Pow else None
    return ex is not None and any(
        isinstance(t.as_independent(sympy.KroneckerDelta, as_Add=False)[1], sympy.KroneckerDelta)
        for t in sympy.Add.make_args(expand(ex))
    )


def _split_delta_power(e: sympy.Basic) -> Expr:
    """``b^(c [p] + r)`` as ``b^r (1 + (b^c - 1) [p])``."""
    base, ex = (sympy.E, e.args[0]) if isinstance(e, sympy.exp) else (e.base, e.exp)
    rest: Expr = sympy.S.Zero
    out: Expr = sympy.S.One
    for t in sympy.Add.make_args(expand(ex)):
        c, f = t.as_independent(sympy.KroneckerDelta, as_Add=False)
        if isinstance(f, sympy.KroneckerDelta):
            out *= 1 + (base**c - 1) * f
        else:
            rest += t
    return base**rest * out


def _unmod_phase(f: sympy.exp) -> Expr:
    """``e^{2 pi i c Mod(a, D)}`` as ``e^{2 pi i c a}`` for integer a and integer c D."""
    turns = expand(f.args[0] / (2 * sympy.pi * sympy.I))
    out: Expr = sympy.S.Zero
    for term in sympy.Add.make_args(turns):
        mods = [m for m in sympy.Mul.make_args(term) if isinstance(m, sympy.Mod)]
        if len(mods) == 1:
            m = mods[0]
            c = term / m
            if m.args[0].is_integer and (c * m.args[1]).is_integer:
                term = expand(c * m.args[0])
        out += term
    return sympy.exp(2 * sympy.pi * sympy.I * out)


def _fraction(e: Expr) -> Expr:
    """The non-integer numeric part of an additive exponent."""
    c = sympy.Add(*(x for x in sympy.Add.make_args(e) if x.is_number))
    return c - sympy.floor(c) if c.is_Rational else (c if c.is_number else sympy.S.Zero)


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
    for t in map(fresh, s.terms):
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
    if j - i < g.arity and same(sympy.Add(*s.widths[i:j]), g.arity):
        r = i
        while r < j:
            if same(s.widths[r], 1):
                r += 1
                continue
            s, j = split(s, r, sympy.S.One), j + 1
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
    stack = list(reversed(factors))
    while stack:
        f = stack.pop()
        if isinstance(f, Tensor):
            stack.extend(reversed(f.factors))
        else:
            flat.append(f)
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
        if (over.is_Integer and over > 0) or (not over.is_Integer and _nonnegative(over)):
            s = split(s, r - 1, expand(w - over))
            return s, r, j + 1
    hint = (
        f"; assume each register width is at least {width}"
        if any(not sympy.sympify(x).is_Integer for x in s.widths[k:j] if x is not None)
        else ""
    )
    raise PathSumError(f"no registers of total width {width} to act on{hint}")


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
    total: Expr = sympy.S.Zero
    for ta in a.terms:
        for tb in (fresh(t) for t in b.terms):
            total += _contract(ta, tb, a.widths, a.dims)
    return _simp(_delta_tidy(total))


def _contract(
    ta: Term, tb: Term, widths: Sequence[Expr | None], dims: Sequence[Expr | None]
) -> Expr:
    amp, bound = _pinned(ta, tb, widths, dims)
    if amp == 0:
        return amp
    return _close(_signs(amp, [b.symbol for b in bound]), bound, {})


def _geometric(amp: Expr, b: Bound, others: Sequence[Bound] = ()) -> Expr | None:
    """``sum_b amp`` for ``amp = C e^{2 pi i b L / R}`` over ``0..R-1``; else None. An index
    of ``others`` over ``0..R-1`` with unit weight in L is pinned to the rest of L mod R.
    Under :func:`divisible`, a dividend a of b | a is read as ``b q`` with q an integer."""
    done = _geometric_core(amp, b, others)
    if done is not None or b.symbol not in amp.free_symbols:
        return done
    for a, d in _DIVISIBLE.get():
        if not a.free_symbols <= amp.free_symbols:
            continue
        q = sympy.Dummy("q", integer=True, positive=True)
        to, back = {a: d * q}, {q: a / d}
        done = _geometric_core(
            expand(amp.subs(to)),
            Bound(b.symbol, b.lo, expand(b.hi.subs(to))),
            [Bound(x.symbol, x.lo, expand(x.hi.subs(to))) for x in others],
        )
        if done is not None:
            return done.xreplace(back)
    return None


def _geometric_core(amp: Expr, b: Bound, others: Sequence[Bound]) -> Expr | None:
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
    if coeff == 0:
        return rest * sympy.exp(2 * sympy.pi * sympy.I * arg) * count
    level = expand(coeff * count)
    if not level.is_integer:
        return None
    level = sympy.Add(*(p for p in sympy.Add.make_args(level) if not (p / count).is_integer))
    other = sympy.exp(2 * sympy.pi * sympy.I * expand(arg - coeff * sym))
    parts = sympy.Add.make_args(level)
    signs = [p.as_coeff_Mul() for p in parts]
    for p, (c, u) in zip(parts, signs, strict=True):
        partner = next((x for x in others if x.symbol == u and x.symbol != sym), None)
        if partner is None or abs(c) != 1 or partner.lo != 0 or not same(partner.hi + 1, count):
            continue
        residue = expand(-c * (level - p))
        if u in residue.free_symbols or not residue.is_integer:
            continue
        return rest * other * count * sympy.KroneckerDelta(u, sympy.Mod(residue, count))
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
        if geo is None:
            geo = _bit_character(amp, b, {})
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


def _split_characters(amp: Expr, sym: sympy.Symbol) -> Expr:
    """``amp`` with each exponential's ``ddot`` terms in ``sym`` split off into a factor of
    their own."""
    out = sympy.S.One
    for f in sympy.Mul.make_args(amp):
        if not isinstance(f, sympy.exp):
            out *= f
            continue
        terms = sympy.Add.make_args(expand(f.args[0]))
        hit = sympy.Add(*(x for x in terms if any(sym in dd.args[:2] for dd in x.atoms(ddot))))
        if hit == 0 or hit == f.args[0]:
            out *= f
            continue
        out *= sympy.Mul(sympy.exp(hit, evaluate=False), sympy.exp(f.args[0] - hit), evaluate=False)
    return out


def _parity(amp: Expr, b: Bound, bound: Sequence[Bound]) -> Expr | None:
    """``sum_b C chi(b, v)`` over a full ``d^w`` range, for a digit character chi, as
    ``d^w C [v = 0]``; else None."""
    sym = b.symbol
    if b.lo != 0:
        return None
    factors = sympy.Mul.make_args(_split_characters(amp, sym))
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
                    _inside(val, b, t.bound)
                    or _in_known_range(val, b)
                    or (
                        val.is_Integer
                        and b.lo.is_Integer
                        and b.hi.is_Integer
                        and b.lo <= val <= b.hi
                    )
                    or (val.is_Integer and b.lo.is_Integer and b.lo <= val and _at_least(b.hi, val))
                    or val == b.lo
                    or (other is not None and same(other.lo, b.lo) and same(other.hi, b.hi))
                    or same(val, b.hi)
                )
                if inside:
                    t = _subs_term(t, {sym: val})
                    changed = True
                    break
            if not changed:
                solved = _solve_shift_delta(f, t.bound)
                if solved is not None:
                    t = _subs_term(t, solved)
                    changed = True
            if changed:
                break
    return t


def _solve_shift_delta(
    f: sympy.KroneckerDelta, bound: Sequence[Bound]
) -> dict[sympy.Basic, Expr] | None:
    """``{b: Mod(v - s, D)}`` for ``[v = Mod(b + s, D)]``, b a bound index over ``0..D-1`` and
    v known to lie in that range."""
    for val, m in (f.args, f.args[::-1]):
        if not isinstance(m, sympy.Mod):
            continue
        top, modulus = m.args
        for b in bound:
            shift = expand(top - b.symbol)
            if (
                b.lo == 0
                and same(b.hi + 1, modulus)
                and b.symbol not in shift.free_symbols
                and b.symbol not in val.free_symbols
                and (_in_known_range(val, b) or _inside(val, b, bound))
            ):
                return {b.symbol: sympy.Mod(val - shift, modulus)}
    return None


def _span(e: Expr, bounds: Sequence[Bound]) -> tuple[Expr, Expr] | None:
    """``(lo, hi)`` of ``e`` over the ranges of ``bounds``: sums, constant multiples and
    ``Mod(u, m)`` (in ``0..m-1``, m non-negative and so non-zero) of bound indices."""
    table = {b.symbol: b for b in bounds}
    if not e.free_symbols:
        return e, e
    if e in table:
        return table[e].lo, table[e].hi
    if (
        isinstance(e, sympy.Mod)
        and e.args[1].is_nonnegative
        and not e.args[1].free_symbols & set(table)
    ):
        return sympy.S.Zero, e.args[1] - 1
    if isinstance(e, sympy.Add):
        lo = hi = sympy.S.Zero
        for a in e.args:
            part = _span(a, bounds)
            if part is None:
                return None
            lo, hi = lo + part[0], hi + part[1]
        return lo, hi
    if isinstance(e, sympy.Mul):
        coeff, rest = e.as_independent(*table, as_Add=False)
        part = _span(rest, bounds) if rest != e else None
        if part is None or not (coeff.is_nonnegative or coeff.is_nonpositive):
            return None
        lo, hi = coeff * part[0], coeff * part[1]
        return (lo, hi) if coeff.is_nonnegative else (hi, lo)
    return None


def _inside(val: Expr, b: Bound, bounds: Sequence[Bound]) -> bool:
    """Whether ``val`` provably stays in the range of ``b`` over the other ``bounds``, also
    with a dividend a of each :func:`divisible` fact b | a read as ``b q``."""
    if _inside_plain(val, b, bounds):
        return True
    for a, d in _DIVISIBLE.get():
        q = sympy.Dummy("q", integer=True, positive=True)
        to = {a: d * q}

        def sub(x: Bound, to: dict[Expr, Expr] = to) -> Bound:
            return Bound(x.symbol, expand(x.lo.subs(to)), expand(x.hi.subs(to)))

        if _inside_plain(expand(val.subs(to)), sub(b), [sub(x) for x in bounds]):
            return True
    return False


def _inside_plain(val: Expr, b: Bound, bounds: Sequence[Bound]) -> bool:
    span = _span(val, [x for x in bounds if x.symbol != b.symbol])
    if span is None:
        return False
    lo, hi = (sympy.simplify(x) for x in span)
    low, high = sympy.simplify(lo - b.lo), sympy.simplify(b.hi - hi)
    return bool(low.is_nonnegative) and bool(high.is_nonnegative)


def _at_least(hi: Expr, v: Expr) -> bool:
    """Whether ``hi >= v`` for every value of its symbols of at least 1 (register widths and
    dimensions), ``hi`` being ``d^w - 1``-like and growing in each symbol."""
    syms = [x for x in hi.free_symbols if not isinstance(x, sympy.Dummy)]
    if not syms or not all(x.is_nonnegative for x in syms):
        return False
    low = hi.xreplace({x: sympy.S.One for x in syms})
    if not low.is_Integer or low < v:
        return False
    return all(sympy.diff(hi, x).xreplace({y: sympy.S.One for y in syms}).is_positive for x in syms)


def _reduce_mods(t: Term) -> Term:
    """``Mod(b, D)`` with b a bound index over ``0..D-1`` replaced by b."""
    ranges = {b.symbol: b for b in t.bound if b.lo == 0}

    def plain(e: sympy.Mod) -> bool:
        b = ranges.get(e.args[0])
        if b is None:
            span = (_RANGES.get() or {}).get(e.args[0])
            return span is not None and span[0] == 0 and same(span[1] + 1, e.args[1])
        return same(b.hi + 1, e.args[1])

    def rewrite(e: Expr) -> Expr:
        return e.replace(lambda x: isinstance(x, sympy.Mod) and plain(x), lambda x: x.args[0])

    return Term(rewrite(t.amp), t.bound, tuple(rewrite(k) for k in t.kets))


def _reindex_shifts(t: Term) -> Term:
    """A ket ``Mod(b + s, D)``, b a bound index over ``0..D-1`` in no other ket, re-indexed to
    b when the amplitude at ``Mod(b - s, D)`` is free of Mod in b."""
    for r, k in enumerate(t.kets):
        if not isinstance(k, sympy.Mod):
            continue
        top, modulus = k.args
        for b in t.bound:
            shift = expand(top - b.symbol)
            if not (
                b.lo == 0
                and same(b.hi + 1, modulus)
                and b.symbol not in shift.free_symbols
                and shift.is_integer
                and not any(b.symbol in x.free_symbols for j, x in enumerate(t.kets) if j != r)
            ):
                continue
            amp = t.amp.xreplace({b.symbol: sympy.Mod(b.symbol - shift, modulus)})
            amp = tidy(_reduce_mods(Term(amp, t.bound, ())).amp)
            if any(b.symbol in m.free_symbols for m in amp.atoms(sympy.Mod)):
                continue
            kets = t.kets[:r] + (b.symbol,) + t.kets[r + 1 :]
            return _reindex_shifts(Term(amp, t.bound, kets))
    return t


def simplify(s: State) -> State:
    """Terms tidied, zero terms dropped, and bound indices outside every ket summed out."""
    terms = []
    for t in (p for t in s.terms for p in _delta_parts(t)):
        t = _reindex_shifts(_reduce_mods(t))
        amp = tidy(t.amp)
        bound = list(t.bound)
        for b in list(bound):
            if any(b.symbol in k.free_symbols for k in t.kets):
                continue
            if b.symbol not in amp.free_symbols:
                amp = amp * (b.hi - b.lo + 1)
                bound.remove(b)
                continue
            geo = _geometric(amp, b, bound)
            if geo is None:
                geo = _parity(amp, b, bound)
            if geo is None:
                limits = {x.symbol: x.hi + 1 for x in bound if x.lo == 0}
                geo = _bit_character(amp, b, limits)
            if geo is None:
                geo = _digit_character(amp, b)
            if geo is not None:
                amp = _delta_tidy(tidy(geo))
                bound.remove(b)
        t = _eliminate_deltas(Term(amp, tuple(bound), t.kets))
        amp = tidy(t.amp)
        if amp == 0:
            continue
        terms.append(_pin_digits(Term(amp, t.bound, t.kets), s))
    terms = [t for t in _merge_terms(terms, s) if t.amp != 0]
    if terms and all(not t.bound and all(k.is_Integer for k in t.kets) for t in terms):
        terms.sort(key=lambda t: tuple(int(k) for k in t.kets))
    if not terms:
        terms = [Term(sympy.S.Zero, (), s.terms[0].kets if s.terms else ())]
    return _with(s, terms=terms)


def _delta_parts(t: Term) -> list[Term]:
    """``t`` split into one term per additive part of its amplitude when a delta on a bound
    index occurs in it."""
    syms = {b.symbol for b in t.bound}
    amp = tidy(t.amp)
    if not any(syms & d.free_symbols for d in amp.atoms(sympy.KroneckerDelta)):
        return [t]
    parts = sympy.Add.make_args(expand(amp))
    return [Term(p, t.bound, t.kets) for p in parts] if len(parts) > 1 else [t]


_CANON = [sympy.Symbol(f"_canon{k}", integer=True) for k in range(64)]


def _merge_terms(terms: Sequence[Term], s: State) -> list[Term]:
    """Terms equal up to the names of their summation indices joined, amplitudes added, when
    the joined term is simpler: a digit it fixes or an amplitude that is no longer a sum."""
    out: list[Term] = []
    keys: dict[tuple[object, ...], int] = {}
    for t in terms:
        if len(t.bound) > len(_CANON):
            out.append(t)
            continue
        names = {b.symbol: _CANON[k] for k, b in enumerate(t.bound)}
        key = (
            tuple((b.lo.xreplace(names), b.hi.xreplace(names)) for b in t.bound),
            tuple(sympy.sympify(k).xreplace(names) for k in t.kets),
        )
        if key not in keys:
            keys[key] = len(out)
            out.append(t)
            continue
        first = out[keys[key]]
        back = {_CANON[k]: b.symbol for k, b in enumerate(first.bound)}
        amp = tidy(first.amp + t.amp.xreplace(names).xreplace(back))
        joined = _pin_digits(Term(amp, first.bound, first.kets), s)
        if not first.bound or len(joined.bound) < len(first.bound) or not amp.is_Add:
            out[keys[key]] = joined
            continue
        out.append(t)
    return out


def _pin_digits(t: Term, s: State) -> Term:
    """A summation index that is the whole value of a one-digit register of concrete
    dimension, with an amplitude vanishing at every digit value but one, fixed to that
    value."""
    for b in t.bound:
        if not (b.lo.is_Integer and b.hi.is_Integer):
            continue
        regs = [
            r
            for r, k in enumerate(t.kets)
            if k == b.symbol
            and same(s.widths[r], 1)
            and s.dims[r] is not None
            and same(s.dims[r], b.hi - b.lo + 1)
        ]
        if not regs or b.symbol not in t.amp.free_symbols:
            continue
        v = _pinned_digit(t.amp, b.symbol, b.lo, int(b.hi - b.lo) + 1)
        if v is not None:
            t = _subs_term(t, {b.symbol: v})
            return _pin_digits(Term(tidy(sympy.simplify(t.amp)), t.bound, t.kets), s)
    return t


def _pinned_digit(amp: Expr, sym: sympy.Symbol, lo: Expr, size: int) -> Expr | None:
    """The one value of ``sym`` in ``lo .. lo + size - 1`` where ``amp`` does not vanish, when
    ``amp`` is a character sum ``sum_j c_j w^(j (sym - lo))``, ``w = e^(2 pi i / size)``, with
    ``c_j = c_0 w^(-j k)`` for every ``j``."""
    coeffs: dict[int, Expr] = {}
    for term in sympy.Add.make_args(sympy.expand(amp)):
        c, dep = term.as_independent(sym, as_Add=False)
        dep = sympy.powsimp(dep.rewrite(sympy.exp), combine="exp")
        if dep == 1:
            j, phase = 0, sympy.S.Zero
        elif isinstance(dep, sympy.exp):
            turns = sympy.expand(dep.args[0] / (2 * sympy.pi * sympy.I))
            a = turns.coeff(sym, 1)
            phase = sympy.expand(turns - a * sym)
            j_val = sympy.nsimplify(a * size)
            if phase.has(sym) or not j_val.is_Integer:
                return None
            phase += a * lo
            j = int(j_val) % size
        else:
            return None
        coeffs[j] = coeffs.get(j, sympy.S.Zero) + c * sympy.exp(2 * sympy.pi * sympy.I * phase)
    if len(coeffs) < size or any(sympy.simplify(c) == 0 for c in coeffs.values()):
        return None
    ratio = sympy.simplify(coeffs[1] / coeffs[0]) if size > 1 else sympy.S.One
    turns = sympy.nsimplify(sympy.arg(ratio) / (2 * sympy.pi))
    if not turns.is_Rational or sympy.simplify(sympy.Abs(ratio) - 1) != 0:
        return None
    shift = -turns * size
    if not shift.is_Integer:
        return None
    k = int(shift) % size
    w = sympy.exp(-2 * sympy.pi * sympy.I * k / size)
    if any(sympy.simplify(coeffs[j] - coeffs[0] * w**j) != 0 for j in range(size)):
        return None
    return lo + k


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


def decode(key: int, levels: Sequence[int]) -> tuple[sympy.Integer, ...]:
    """Register values of a mixed-radix key."""
    return tuple(
        sympy.Integer((key // sh) % lv) for lv, sh in zip(levels, _suffixes(levels), strict=True)
    )


@dataclass(frozen=True)
class Branch:
    """Outcomes ``value`` with probability ``prob`` for every value of the indices ``free``;
    ``registers`` holds the value of each measured register."""

    value: Expr
    prob: Expr
    free: tuple[Bound, ...] = ()
    registers: tuple[Expr, ...] = ()


@dataclass(frozen=True)
class Distribution:
    """A Born distribution as branches of outcomes."""

    branches: tuple[Branch, ...]


@dataclass(frozen=True)
class Mixed:
    """A density matrix ``sum rho_kb |k><b|`` held as its vectorisation ``vec``: the ket
    registers, then as many bra registers, with amplitudes ``rho_kb``; ``factors`` holds
    ``(k, b)`` when ``vec`` is the product of a ket part and a bra part."""

    vec: State
    factors: tuple[State, State] | None = None

    @property
    def count(self) -> int:
        """The number of ket registers."""
        return len(self.vec.widths) // 2

    def uniform_dim(self) -> Expr | None:
        """The common dimension of every register, or None when they differ."""
        return self.layout().uniform_dim()

    def layout(self) -> State:
        """A ket with the registers of the density matrix and no terms of interest."""
        k = self.count
        return State(
            self.vec.widths[:k],
            (Term(sympy.S.One, (), (sympy.S.Zero,) * k),),
            False,
            self.vec.dims[:k],
        )


def density_of(op: Operator) -> Mixed | None:
    """The density matrix of a sum of scaled outer products ``c |k><b|``; None for any other
    operator."""
    if isinstance(op, Outer):
        if op.ket.bra or not op.bra.bra:
            return None
        bra = State(op.bra.widths, op.bra.terms, False, op.bra.dims)
        return _product(simplify(op.ket), simplify(bra))
    if isinstance(op, Scaled):
        inner_m = density_of(op.op)
        if inner_m is None:
            return None
        if inner_m.factors is not None:
            a, b = inner_m.factors
            return _product(scale(op.factor, a), b)
        return Mixed(scale(op.factor, inner_m.vec))
    if isinstance(op, OpSum):
        parts = [density_of(o) for o in op.ops]
        if any(m is None for m in parts):
            return None
        vec = parts[0].vec  # type: ignore[union-attr]
        for m in parts[1:]:
            vec = add(vec, m.vec)  # type: ignore[union-attr]
        return Mixed(simplify(vec))
    return None


def conj(s: State) -> State:
    """``s`` with complex-conjugated amplitudes, still a ket."""
    return _with(s, terms=[Term(conjugate(t.amp), t.bound, t.kets) for t in s.terms])


def density(s: State) -> Mixed:
    """``|s><s|`` of a ket."""
    if s.bra:
        raise PathSumError("a density matrix is made from a ket")
    s = simplify(s)
    return _product(s, conj(s))


def _product(a: State, b: State) -> Mixed:
    """The density matrix with ket part ``a`` and bra part ``b``."""
    return Mixed(simplify(tensor(a, b)), (a, b))


def apply_mixed(op: Operator, m: Mixed) -> Mixed:
    """``op rho op^dagger``."""
    vec = m.vec
    token = _SCOPE.set(Params(free_symbols(vec))) if _SCOPE.get() is None else None
    try:
        if m.factors is not None:
            a, b = m.factors
            a = simplify(_resolved(_apply(op, a, 0, len(a.widths))))
            b = conj(simplify(_resolved(_apply(op, conj(b), 0, len(b.widths)))))
            return _product(a, b)
        k = m.count
        v = _apply(op, vec, 0, k)
        start = len(v.widths) - k
        v = conj(_apply(op, conj(v), start, len(v.widths)))
        return Mixed(simplify(_resolved(v)))
    finally:
        if token is not None:
            _SCOPE.reset(token)


def trace(m: Mixed) -> Expr:
    """``tr rho`` by symbolic contraction."""
    layout = m.layout()
    bounds = [new_bound("z", 0, layout.levels(k) - 1) for k in range(m.count)]
    zs = [b.symbol for b in bounds]
    total: Expr = sympy.S.Zero
    for t in m.vec.terms:
        bra = Term(sympy.S.One, tuple(bounds), tuple(zs + zs))
        amp, bound = _pinned(bra, fresh(t), m.vec.widths, m.vec.dims)
        if amp != 0:
            total += _close(_signs(amp, [b.symbol for b in bound]), bound, {})
    return _delta_tidy(tidy(total))


def collapse(d: Distribution, layout: State) -> Mixed:
    """The mixture ``sum p |v><v|`` of the outcomes of ``d`` on the registers of ``layout``."""
    terms = [Term(b.prob, b.free, tuple(b.registers) + tuple(b.registers)) for b in d.branches]
    vec = State(layout.widths * 2, tuple(terms), False, layout.dims * 2)
    return Mixed(simplify(vec))


def _simp(e: Expr) -> Expr:
    """``sympy.simplify`` of an expression without unevaluated sums; others unchanged, since
    simplify evaluates a sum over a concrete range term by term."""
    if e.has(sympy.Sum):
        e = e.replace(
            lambda x: isinstance(x, sympy.Sum),
            lambda x: _close(x.function, [Bound(v, lo, hi) for v, lo, hi in x.limits], {}),
        )
    if e.has(sympy.Sum):
        return e
    out = sympy.simplify(e)
    cancelled = _cancel_powers(out)
    return cancelled if sympy.count_ops(cancelled) < sympy.count_ops(out) else out


def _cancel_powers(e: Expr) -> Expr:
    """``e`` as a cancelled rational function of ``p^(g n)`` for each prime p and symbol n
    of its powers ``p^(c n + b)``, g the step of the coefficients c."""
    e = tidy(e)
    pows = [
        x
        for x in e.atoms(sympy.Pow)
        if x.base.is_Integer and x.base > 1 and sympy.isprime(int(x.base)) and x.exp.free_symbols
    ]
    if not pows:
        return e
    steps: dict[tuple[sympy.Integer, sympy.Symbol], list[sympy.Rational]] = {}
    for x in pows:
        terms = sympy.Add.make_args(expand(x.exp))
        syms = x.exp.free_symbols
        if len(syms) != 1:
            return e
        (n,) = syms
        for t in terms:
            c = t.coeff(n) if t.has(n) else None
            if c is not None and not c.is_Rational:
                return e
        c = expand(x.exp).coeff(n)
        if not c.is_Rational or expand(x.exp - c * n).has(n):
            return e
        steps.setdefault((x.base, n), []).append(c)
    mapping: dict[sympy.Basic, Expr] = {}
    back: dict[sympy.Symbol, Expr] = {}
    for (q, n), cs in steps.items():
        g = sympy.Rational(sympy.gcd([c.p for c in cs]), sympy.lcm([c.q for c in cs]))
        t = sympy.Dummy(f"{q}_{n}", positive=True)
        back[t] = q ** (g * n)
        for x in pows:
            if x.base == q and n in x.exp.free_symbols:
                c = expand(x.exp).coeff(n)
                mapping[x] = q ** expand(x.exp - c * n) * t ** (c / g)
    return tidy(sympy.factor(sympy.cancel(e.xreplace(mapping))).xreplace(back))


def _xor_leaves(e: Expr) -> list[Expr]:
    if isinstance(e, xor):
        return _xor_leaves(e.args[0]) + _xor_leaves(e.args[1])
    return [e]


def _xor_of(leaves: Sequence[Expr]) -> Expr:
    out: Expr = sympy.S.Zero
    for leaf in leaves:
        out = xor(out, leaf)
    return out


def _solve_xor(
    lhs: Expr, rhs: Expr, bound: list[Bound], width: Expr, dim: Expr
) -> tuple[Expr, Expr] | None:
    """``(u, value)`` solving ``lhs = rhs`` for a full-range bound qubit index u that is one
    leaf of the xor ``lhs``."""
    if not isinstance(lhs, xor) or not same(dim, TWO):
        return None
    leaves = _xor_leaves(lhs)
    for k, u in enumerate(leaves):
        rest = leaves[:k] + leaves[k + 1 :]
        if _bound_ok(u, sympy.Add(rhs, *rest), bound, width, dim):
            return u, _xor_of([rhs, *rest])
    return None


def _sign_exponent(
    e: Expr, summed: Sequence[sympy.Symbol], bits: frozenset[sympy.Symbol] = frozenset()
) -> Expr:
    """An exponent of -1 rewritten mod 2 as a sum of distinct atoms: xor as a sum, ``bit(x, k)``
    and a summed index x as characters ``bitdot(x, c)``, characters split over xor partners;
    a product with a one-bit index b in ``bits`` read as ``bitdot(b, other)``."""
    e = inline(e)
    counts: dict[Expr, int] = {}
    stack = list(sympy.Add.make_args(expand(e)))
    while stack:
        term = stack.pop()
        coeff, rest = term.as_coeff_Mul()
        if coeff.is_Integer:
            if int(coeff) % 2 == 0:
                continue
            term = rest
        elif coeff != 1:
            counts[term] = counts.get(term, 0) + 1
            continue
        if isinstance(term, sympy.Add):
            stack.extend(sympy.Add.make_args(term))
            continue
        if term == 1:
            counts[sympy.S.One] = counts.get(sympy.S.One, 0) + 1
            continue
        if isinstance(term, xor):
            stack.extend(_xor_leaves(term))
            continue
        if isinstance(term, sympy.Mod) and term.args[1] == 2:
            stack.append(term.args[0])
            continue
        if isinstance(term, bit) and term.args[1].is_Integer:
            stack.append(bitdot(term.args[0], 2 ** term.args[1]))
            continue
        if term in summed:
            stack.append(bitdot(term, 1))
            continue
        if isinstance(term, sympy.Mul) and len(term.args) == 2:
            hit = next((a for a in term.args if a in bits), None)
            if hit is not None:
                other = next(a for a in term.args if a is not hit)
                if other.is_integer:
                    stack.append(bitdot(hit, other))
                    continue
        if isinstance(term, bitdot):
            a, b = term.args
            if a == b:
                term = pop(a)
            elif len(_xor_leaves(b)) > 1:
                stack.extend(bitdot(a, leaf) for leaf in _xor_leaves(b))
                continue
            elif len(_xor_leaves(a)) > 1:
                stack.extend(bitdot(leaf, b) for leaf in _xor_leaves(a))
                continue
            else:
                term = bitdot(*sorted((a, b), key=sympy.default_sort_key))
        counts[term] = counts.get(term, 0) + 1
    return sympy.Add(*(k for k, c in counts.items() if c % 2))


def _signs(
    amp: Expr, summed: Sequence[sympy.Symbol], bits: frozenset[sympy.Symbol] = frozenset()
) -> Expr:
    """``amp`` with its factors of -1 joined and their exponents in :func:`_sign_exponent`
    form."""
    exponent: Expr = sympy.S.Zero
    rest = sympy.S.One
    for f in sympy.Mul.make_args(sympy.powsimp(amp)):
        if f.is_Pow and f.base == -1:
            exponent += f.exp
        elif f == -1:
            exponent += 1
        else:
            rest *= f
    if exponent == 0:
        return amp
    exponent = _sign_exponent(exponent, summed, bits)
    fixed = sympy.Add(*(x for x in sympy.Add.make_args(exponent) if x.is_number))
    return rest * sympy.Pow(-1, fixed) * (-1) ** (exponent - fixed)


def _bit_character(amp: Expr, b: Bound, limits: dict[sympy.Symbol, Expr]) -> Expr | None:
    """``sum_b C (-1)^bitdot(b, v)`` over ``0..2^w - 1`` as ``2^w C [v = 0]`` when v is
    free of b and below ``2^w`` (its symbols bounded by ``limits``); else None."""
    sym = b.symbol
    if b.lo != 0:
        return None
    size = b.hi + 1
    w = sympy.log(size, 2) if not size.is_Integer else None
    if size.is_Integer and int(size) & (int(size) - 1):
        return None
    if w is not None and not same(TWO ** sympy.simplify(w), size):
        return None
    amp = _signs(amp, [sym], frozenset({sym}) if same(size, 2) else frozenset())
    factors = sympy.Mul.make_args(amp)
    sign = next((f for f in factors if f.is_Pow and f.base == -1), None)
    rest = sympy.Mul(*(f for f in factors if f is not sign))
    if sym in rest.free_symbols:
        return None
    if sign is None:
        return rest * size

    def top_ok(v: Expr, size: Expr) -> bool:
        for leaf in _xor_leaves(v):
            if leaf.is_Integer:
                if int(leaf) < 0 or (size.is_Integer and int(leaf) >= int(size)):
                    return False
                continue
            if same(leaf, size - 1):
                continue
            top = limits.get(leaf) if isinstance(leaf, sympy.Symbol) else None
            if top is None or not same(top, size):
                if not _in_known_range(leaf, Bound(sym, sympy.S.Zero, size - 1)):
                    return False
        return True

    terms = sympy.Add.make_args(sign.exp)
    mine = [t for t in terms if sym in t.free_symbols]
    partners: list[Expr] = []
    for term in mine:
        if isinstance(term, pop) and term.args[0] == sym:
            partners.append(size - 1)
        elif isinstance(term, bitdot) and sym in term.args and term.args[0] != term.args[1]:
            v = term.args[1] if term.args[0] == sym else term.args[0]
            if sym in v.free_symbols:
                return None
            partners.append(v)
        else:
            return None
    v = _xor_of(partners)
    other = sympy.Add(*(t for t in terms if t not in mine))
    if all(top_ok(p, size) for p in partners):
        return rest * (-1) ** other * size * sympy.KroneckerDelta(v, 0)
    return None


def _digit_character(amp: Expr, b: Bound) -> Expr | None:
    """``sum_b C e^{2 pi i sum_j c_j ddot(b, v_j, d)}`` over ``0..d^w - 1``: ``d^w C`` when the
    digitwise combination of the v_j vanishes (``[v = 0]`` for one partner of unit weight),
    else 0; None outside that form."""
    sym = b.symbol
    if b.lo != 0:
        return None
    size = b.hi + 1
    factors = sympy.Mul.make_args(amp)
    exps = [f for f in factors if isinstance(f, sympy.exp)]
    rest = sympy.Mul(*(f for f in factors if not isinstance(f, sympy.exp)))
    if not exps or sym in rest.free_symbols:
        return None
    arg = expand(sympy.Add(*(e.args[0] for e in exps)) / (2 * sympy.pi * sympy.I))
    parts: list[tuple[Expr, Expr]] = []
    other: Expr = sympy.S.Zero
    base: Expr | None = None
    plain: list[Expr] = []
    for term in sympy.Add.make_args(arg):
        if sym not in term.free_symbols:
            other += term
            continue
        hits = [a for a in sympy.Mul.make_args(term) if isinstance(a, ddot) and sym in a.args[:2]]
        if not hits and sym not in (term / sym).free_symbols:
            plain.append(term / sym)
            continue
        if len(hits) != 1:
            return None
        dd = hits[0]
        c = sympy.simplify(term / dd)
        if sym in c.free_symbols:
            return None
        v = dd.args[1] if dd.args[0] == sym else dd.args[0]
        if sym in v.free_symbols or (base is not None and not same(base, dd.args[2])):
            return None
        base = dd.args[2]
        m = sympy.simplify(c * base)
        if not m.is_Integer:
            return None
        parts.append((m, v))
    if plain and base is None:
        return None
    for c in plain:
        m = sympy.simplify(c * base)
        if not m.is_Integer:
            return None
        parts.append((m, sympy.S.One))
    if base is None:
        return None
    if not (base.is_Integer and size.is_Integer):
        if len(parts) == 1 and parts[0][0] % base in (1, base - 1):
            return (
                rest
                * sympy.exp(2 * sympy.pi * sympy.I * other)
                * size
                * (sympy.KroneckerDelta(parts[0][1], 0))
            )
        if len(parts) == 2 and parts[0][0] in (1, -1) and parts[0][0] + parts[1][0] == 0:
            return (
                rest
                * sympy.exp(2 * sympy.pi * sympy.I * other)
                * size
                * sympy.KroneckerDelta(parts[0][1], parts[1][1])
            )
        return None
    d = int(base)
    width, top = 0, 1
    while top < int(size):
        top, width = top * d, width + 1
    if top != int(size):
        return None
    spread = _repeated_partners(parts, d, width, size)
    if spread is not None:
        return rest * sympy.exp(2 * sympy.pi * sympy.I * other) * size * spread
    if not all(v.is_Integer for _, v in parts):
        if len(parts) == 1 and int(parts[0][0]) % d in (1, d - 1):
            return (
                rest
                * sympy.exp(2 * sympy.pi * sympy.I * other)
                * size
                * (sympy.KroneckerDelta(parts[0][1], 0))
            )
        return None
    total = [0] * width
    for m, v in parts:
        value = int(v)
        for k in range(width):
            value, digit = divmod(value, d)
            total[k] = (total[k] + int(m) * digit) % d
    if any(total):
        return sympy.S.Zero
    return rest * sympy.exp(2 * sympy.pi * sympy.I * other) * size


def _repeated_partners(
    parts: Sequence[tuple[Expr, Expr]], d: int, width: int, size: Expr
) -> Expr | None:
    """The delta of ``sum_j m_j ddot(b, v_j)`` vanishing digitwise, the partners being
    integers or repeated digits ``c (d^w - 1)/(d - 1)`` (not both) but at most one, a
    whole-register value y of unit weight: ``[y = digitwise negated total]``, else
    ``[total = 0 mod d]``."""
    unit = (d**width - 1) // (d - 1)
    total: Expr = sympy.S.Zero
    digits = [0] * width
    integers = False
    free: list[tuple[Expr, Expr]] = []
    for m, v in parts:
        c = sympy.cancel(v / unit)
        if c.free_symbols and sympy.denom(c) == 1 and c.is_integer is not False:
            total += m * c
        elif v.is_Integer:
            integers = True
            value = int(v)
            for k in range(width):
                value, digit = divmod(value, d)
                digits[k] = (digits[k] + int(m) * digit) % d
        else:
            free.append((m, v))
    if len(free) > 1 or (integers and total != 0):
        return None
    if not free:
        if integers:
            return None
        return sympy.KroneckerDelta(sympy.Mod(total, d), 0)
    m, y = free[0]
    if not y.is_Symbol or int(m) % d not in (1, d - 1):
        return None
    sign = 1 if int(m) % d == 1 else -1
    if integers:
        target = sum(((-sign * digit) % d) * d**k for k, digit in enumerate(digits))
        return sympy.KroneckerDelta(y, sympy.Integer(target))
    return sympy.KroneckerDelta(y, sympy.Mod(-sign * total, d) * unit)


def _delta_tidy(e: Expr) -> Expr:
    """``[xor(a, c) = 0]`` as ``[a = c]``, squared deltas as deltas, and deltas whose sides
    differ over the running :func:`ranged` scope as 0."""
    if _RANGES.get():
        e = e.replace(
            lambda x: isinstance(x, sympy.KroneckerDelta) and _apart(x.args[0], x.args[1]),
            lambda x: sympy.S.Zero,
        )
    e = e.replace(
        lambda x: isinstance(x, sympy.KroneckerDelta) and 0 in x.args,
        lambda x: _delta_zero(next(a for a in x.args if a != 0) if x.args[0] != x.args[1] else 0),
    )
    return e.replace(
        lambda x: x.is_Pow and isinstance(x.base, sympy.KroneckerDelta) and x.exp.is_positive,
        lambda x: x.base,
    )


def _delta_zero(v: Expr) -> Expr:
    leaves = _xor_leaves(v)
    if len(leaves) == 2:
        return sympy.KroneckerDelta(*leaves)
    if len(leaves) > 2:
        return sympy.KroneckerDelta(leaves[0], _xor_of(leaves[1:]))
    if isinstance(v, sympy.Mul):
        c, rest = v.as_independent(*(x for x in v.free_symbols if x.is_Dummy), as_Add=False)
        if c != 1 and rest != 1 and c.is_nonzero:
            return sympy.KroneckerDelta(rest, 0)
    return sympy.KroneckerDelta(v, 0)


def _close(
    amp: Expr, bound: list[Bound], limits: dict[sympy.Symbol, Expr], digits_ok: bool = True
) -> Expr:
    """``sum_bound amp`` by closed forms: deltas substituted, geometric sums, sign characters
    and constants, until none applies; an unevaluated Sum for the rest."""
    syms = {b.symbol for b in bound}
    if syms & sympy.sympify(amp).free_symbols and isinstance(expand(amp), sympy.Add):
        parts = sympy.Add.make_args(expand(amp))
        return sympy.Add(*(_close(p, bound, limits, digits_ok) for p in parts))
    bound = list(bound)
    limits = {**limits, **{b.symbol: b.hi + 1 for b in bound if b.lo == 0}}
    progress = True
    while progress and bound:
        progress = False
        t = _eliminate_deltas(Term(amp, tuple(bound), ()))
        if len(t.bound) < len(bound):
            amp, bound, progress = _delta_tidy(tidy(t.amp)), list(t.bound), True
            continue
        for b in bound:
            if b.symbol not in amp.free_symbols:
                done = amp * (b.hi - b.lo + 1)
            else:
                done = _geometric(amp, b, bound)
                if done is None:
                    done = _bit_character(amp, b, limits)
                if done is None:
                    done = _digit_character(amp, b)
            if done is not None:
                amp = _delta_tidy(tidy(done))
                bound.remove(b)
                progress = True
                break
    if bound and digits_ok:
        done = _close_by_digits(amp, bound, limits)
        if done is not None:
            return done
        done = _close_by_parity(amp, bound, limits)
        if done is not None:
            return done
    for b in bound:
        amp = sympy.Sum(amp, (b.symbol, b.lo, b.hi))
    return amp


def _is_bit(e: Expr, limits: dict[sympy.Symbol, Expr], bound: Sequence[Bound]) -> bool:
    """Whether ``e`` takes only the values 0 and 1."""
    if e.is_Integer:
        return e in (0, 1)
    if isinstance(e, sympy.Symbol):
        b = next((x for x in bound if x.symbol == e), None)
        if b is not None:
            return b.lo == 0 and b.hi == 1
        span = (_RANGES.get() or {}).get(e)
        return same(limits.get(e), 2) or (span is not None and span == (0, 1))
    if isinstance(e, bit) or (isinstance(e, sympy.Mod) and e.args[1] == 2):
        return True
    if isinstance(e, xor):
        return all(_is_bit(a, limits, bound) for a in e.args)
    return False


def _close_by_parity(
    amp: Expr, bound: list[Bound], limits: dict[sympy.Symbol, Expr]
) -> Expr | None:
    """``sum_bound amp`` closed with each delta of two bits on a summed index written as the
    character sum ``(1 + (-1)^(a + b)) / 2``."""
    syms = {b.symbol for b in bound}
    hits = [
        f
        for f in sympy.sympify(amp).atoms(sympy.KroneckerDelta)
        if f.free_symbols & syms and all(_is_bit(a, limits, bound) for a in f.args)
    ]
    if not hits:
        return None
    amp = amp.xreplace(
        {f: (1 + (-1) ** (_parity_sum(f.args[0]) + _parity_sum(f.args[1]))) / 2 for f in hits}
    )
    parts = sympy.Add.make_args(expand(amp))
    done = sympy.Add(*(_close(_signs(p, list(syms)), bound, limits, digits_ok=False) for p in parts))
    return None if done.has(sympy.Sum) else _delta_tidy(tidy(done))


def _parity_sum(e: Expr) -> Expr:
    """A bit expression with the same parity as ``e``: xor as a sum, ``Mod(v, 2)`` as v."""
    if isinstance(e, xor):
        return sympy.Add(*(_parity_sum(a) for a in e.args))
    if isinstance(e, sympy.Mod) and e.args[1] == 2:
        return e.args[0]
    return e


_DIGIT_FUNCS = (shr, sympy.Mod, bit, sympy.floor)


def _digit_base(amp: Expr) -> tuple[sympy.Symbol, int, int] | None:
    """``(x, d, w)`` for a symbol x of the running :func:`ranged` scope with a concrete range
    ``0 .. d^w - 1`` or below, met by a digit function of base d in ``amp``."""
    ranges = _RANGES.get() or {}
    for f in amp.atoms(*_DIGIT_FUNCS):
        if isinstance(f, shr):
            d = f.args[2]
        elif isinstance(f, bit):
            d = TWO
        else:
            m = f.args[1] if isinstance(f, sympy.Mod) else 1 / f.args[0].as_coeff_Mul()[0]
            primes = sympy.factorint(m) if m.is_Integer and m > 1 else {}
            if len(primes) != 1:
                continue
            d = sympy.Integer(next(iter(primes)))
        for x in f.free_symbols:
            lo, hi = ranges.get(x, (None, None))
            if lo == 0 and hi is not None and hi.is_Integer and d.is_Integer and d > 1:
                w = 1
                while int(d) ** w <= int(hi):
                    w += 1
                return x, int(d), w
    return None


def _digit_vector(v: Expr, digits: Sequence[sympy.Symbol], d: int) -> dict[int, sympy.Symbol] | None:
    """``{j: digit}`` when ``v`` is ``sum_j d^j digit_j`` over some of ``digits``."""
    out: dict[int, sympy.Symbol] = {}
    for t in sympy.Add.make_args(expand(v)):
        if t == 0:
            continue
        c, sym = t.as_coeff_Mul()
        if sym not in digits or not c.is_Integer or c <= 0:
            return None
        j = sympy.multiplicity(d, int(c)) if int(c) > 1 else 0
        if d**j != int(c) or j in out:
            return None
        out[j] = sym
    return out


def _eval_digits(e: Expr, digits: Sequence[sympy.Symbol], d: int) -> Expr:
    """``e`` with digit functions of digit sums evaluated to digit sums."""
    if not e.args:
        return e
    e = e.func(*(_eval_digits(a, digits, d) for a in e.args))

    def keep(vec: dict[int, sympy.Symbol], lo: int, hi: int | None) -> Expr:
        return sympy.Add(*(d ** (j - lo) * b for j, b in vec.items() if j >= lo and (hi is None or j < hi)))

    if isinstance(e, shr) and e.args[1].is_Integer and e.args[2] == d:
        vec = _digit_vector(e.args[0], digits, d)
        if vec is not None:
            return keep(vec, int(e.args[1]), None)
    if isinstance(e, sympy.Mod) and e.args[1].is_Integer:
        m = int(e.args[1])
        k = sympy.multiplicity(d, m) if m > 1 else 0
        vec = _digit_vector(e.args[0], digits, d)
        if vec is not None and d**k == m:
            return keep(vec, 0, k)
    if isinstance(e, bit) and d == 2 and e.args[1].is_Integer:
        vec = _digit_vector(e.args[0], digits, d)
        if vec is not None:
            return vec.get(int(e.args[1]), sympy.S.Zero)
    if isinstance(e, sympy.floor):
        q = sympy.denom(sympy.together(e.args[0]))
        if q.is_Integer and q > 1:
            k = sympy.multiplicity(d, int(q))
            vec = _digit_vector(expand(e.args[0] * q), digits, d)
            if vec is not None and d**k == q:
                return keep(vec, k, None)
    return e


def _close_by_digits(
    amp: Expr, bound: list[Bound], limits: dict[sympy.Symbol, Expr]
) -> Expr | None:
    """``sum_bound amp`` closed with a symbol of concrete range written in its digits, when
    the digit functions of it then evaluate; the digits read back as digit functions."""
    found = _digit_base(amp)
    if found is None:
        return None
    x, d, w = found
    digits = [sympy.Dummy(f"{x}{j}", integer=True, nonnegative=True) for j in range(w)]
    whole = sympy.Add(*(d**j * b for j, b in enumerate(digits)))
    amp = _eval_digits(amp.xreplace({x: whole}), digits, d)
    if any(x in f.free_symbols or set(digits) & f.free_symbols for f in amp.atoms(*_DIGIT_FUNCS)):
        return None
    amp = amp.replace(
        lambda e: e.is_Pow and e.base == -1, lambda e: sympy.exp(sympy.pi * sympy.I * e.exp)
    )
    amp = tidy(sympy.powsimp(expand(amp), combine="exp"))
    done = _close(amp, bound, {**limits, **{b: d for b in digits}}, digits_ok=False)
    if done.has(sympy.Sum):
        return None
    back = {b: shr(sympy.Mod(x, d ** (j + 1)), j, d) for j, b in enumerate(digits)}
    return tidy(_delta_tidy(done.xreplace(back)))


def amplitude(
    s: State, values: Sequence[Expr], limits: dict[sympy.Symbol, Expr] | None = None
) -> Expr:
    """``<values|s>`` for one value per register, by substitution and closed-form sums;
    ``limits`` bounds the symbols of ``values`` (exclusive)."""
    limits = dict(limits or {})
    bra = Term(sympy.S.One, (), tuple(values))
    total: Expr = sympy.S.Zero
    for t in s.terms:
        t = fresh(t)
        amp, bound = _pinned(bra, t, s.widths, s.dims)
        if amp == 0:
            continue
        total += _close(_signs(amp, [b.symbol for b in bound]), bound, limits)
    return _delta_tidy(tidy(total))


def _xor_expand(e: Expr) -> Expr:
    """``e`` with the arguments of every xor expanded."""
    return e.replace(lambda x: isinstance(x, xor), lambda x: xor(*map(expand, x.args)))


def _pinned(
    ta: Term, tb: Term, widths: Sequence[Expr | None], dims: Sequence[Expr | None]
) -> tuple[Expr, list[Bound]]:
    """The amplitude of ``<ta|tb>`` with every solvable ket equation substituted, other
    equations as deltas, and the bound indices left to sum."""
    amp = ta.amp * tb.amp
    bound = list(ta.bound + tb.bound)
    eqs = list(zip(ta.kets, tb.kets, widths, dims, strict=True))
    deltas: list[Expr] = []
    while eqs:
        lhs, rhs, w, d = eqs.pop()
        if expand(lhs - rhs) == 0 or _xor_expand(lhs) == _xor_expand(rhs):
            continue
        pick = next(
            ((x, y) for x, y in ((lhs, rhs), (rhs, lhs)) if _bound_ok(x, y, bound, w, d)), None
        )
        if pick is None:
            pick = _solve_xor(lhs, rhs, bound, w, d) or _solve_xor(rhs, lhs, bound, w, d)
        if pick is not None:
            sym, val = pick
            mapping = {sym: val}
            amp = amp.xreplace(mapping)
            eqs = [(x.xreplace(mapping), y.xreplace(mapping), ww, dd) for x, y, ww, dd in eqs]
            deltas = [e.xreplace(mapping) for e in deltas]
            bound = [x for x in bound if x.symbol != sym]
            continue
        if lhs.is_Integer and rhs.is_Integer:
            return sympy.S.Zero, []
        deltas.append(sympy.KroneckerDelta(lhs, rhs))
    return amp * sympy.Mul(*deltas), bound


def _abs2(a: Expr) -> Expr:
    """``|a|^2``, factor by factor where the factors are signs, phases, deltas or positive."""
    if isinstance(a, sympy.Add) or a.has(sympy.Sum):
        return tidy(expand(a * conjugate(a)))
    out = sympy.S.One
    for f in sympy.Mul.make_args(a):
        if f.is_Pow and f.base == -1 and f.exp.is_real is not False:
            continue
        if f in (sympy.I, -sympy.I) or f == -1:
            continue
        if isinstance(f, sympy.exp) and (f.args[0] / sympy.I).is_real:
            continue
        if isinstance(f, sympy.KroneckerDelta):
            out *= f
            continue
        if f.is_positive or f.is_negative:
            out *= f**2
            continue
        out *= f * conjugate(f)
    return tidy(out)


def _register_symbols(s: State) -> list[sympy.Symbol]:
    return [sympy.Dummy(f"z{k}", integer=True, nonnegative=True) for k in range(len(s.widths))]


def _concat_values(values: Sequence[Expr], s: State) -> Expr:
    levels = [s.levels(k) for k in range(len(s.widths))]
    total: Expr = sympy.S.Zero
    for v, n in zip(values, levels, strict=True):
        total = total * n + v
    return expand(total)


def _pins(prob: Expr, zs: Sequence[sympy.Symbol]) -> tuple[Expr, dict[sympy.Symbol, Expr]]:
    """``prob`` with its delta factors that fix register values substituted, and the fixed
    values, constants first, then values of the other registers."""
    pins: dict[sympy.Symbol, Expr] = {}
    for dependent in (False, True):
        prob = _pin_pass(prob, zs, pins, dependent)
    return prob, {z: v.xreplace(pins) for z, v in pins.items()}


def _pin_pass(
    prob: Expr, zs: Sequence[sympy.Symbol], pins: dict[sympy.Symbol, Expr], dependent: bool
) -> Expr:
    changed = True
    while changed:
        changed = False
        for f in sympy.Mul.make_args(prob):
            if not isinstance(f, sympy.KroneckerDelta):
                continue
            for z, v in (f.args, f.args[::-1]):
                others = v.free_symbols & set(zs)
                if z in zs and z not in pins and z not in others and (dependent or not others):
                    pins[z] = v
                    prob = _delta_tidy(tidy(prob.xreplace({z: v})))
                    changed = True
                    break
            if changed:
                break
    return prob


def _born(s: State, prob: Expr, zs: Sequence[sympy.Symbol]) -> Distribution:
    """The distribution with probability ``prob`` at register values ``zs``: each additive
    part becomes a branch whose deltas pin registers, the other registers free indices."""
    prob = _delta_tidy(tidy(prob))
    parts = sympy.Add.make_args(expand(prob)) if prob.has(sympy.KroneckerDelta) else (prob,)
    grouped: dict[tuple[tuple[int, Expr], ...], Expr] = {}
    for part in parts:
        rest, pins = _pins(part, zs)
        if rest == 0:
            continue
        key = tuple(
            sorted(((k, pins[z]) for k, z in enumerate(zs) if z in pins), key=lambda kv: kv[0])
        )
        grouped[key] = grouped.get(key, sympy.S.Zero) + rest
    branches = []
    shared = [
        new_bound("k" if len(zs) == 1 else f"k{k}", 0, s.levels(k) - 1) for k in range(len(zs))
    ]
    for key, rest in grouped.items():
        pins = {zs[k]: v for k, v in key}
        free: list[Bound] = []
        registers: list[Expr] = []
        mapping: dict[sympy.Basic, Expr] = {}
        for k, z in enumerate(zs):
            if z in pins:
                registers.append(pins[z])
                continue
            b = shared[k]
            mapping[z] = b.symbol
            free.append(b)
            registers.append(b.symbol)
        registers = [r.xreplace(mapping) for r in registers]
        p = _delta_tidy(tidy(_simp(rest).xreplace(mapping)))
        if p == 0:
            continue
        if p.free_symbols and not p.has(sympy.Sum, sympy.KroneckerDelta, sympy.Piecewise):
            p = sympy.factor(p)
        regs = tuple(registers)
        branches.append(Branch(_concat_values(regs, s), p, tuple(free), regs))
    branches = _cover(s, branches, shared)
    branches.sort(key=lambda br: sympy.default_sort_key(br.value))
    return Distribution(tuple(branches))


def _cover(s: State, branches: list[Branch], shared: Sequence[Bound]) -> list[Branch]:
    """Branches that differ only in one register, pinned to every one of its values with the
    same probability, merged into one branch with that register free."""
    changed = True
    while changed:
        changed = False
        for r in range(len(s.widths)):
            levels = s.levels(r)
            if not levels.is_Integer:
                continue
            groups: dict[tuple[object, ...], list[Branch]] = {}
            for br in branches:
                v = br.registers[r]
                if not v.is_Integer:
                    continue
                key: tuple[object, ...] = (
                    tuple(x for k, x in enumerate(br.registers) if k != r),
                    br.prob,
                    br.free,
                )
                groups.setdefault(key, []).append(br)
            for key, group in groups.items():
                if len(group) != int(levels):
                    continue
                values = sorted(int(br.registers[r]) for br in group)
                if values != list(range(int(levels))):
                    continue
                first = group[0]
                b = shared[r]
                regs = tuple(b.symbol if k == r else x for k, x in enumerate(first.registers))
                free = tuple(x for x in shared if x in first.free or x is b)
                merged = Branch(_concat_values(regs, s), first.prob, free, regs)
                branches = [br for br in branches if br not in group] + [merged]
                changed = True
                break
            if changed:
                break
    return branches


def joined(d: Distribution, layout: State) -> Distribution:
    """``d`` with each branch's adjacent free registers whose probability does not tell them
    apart read as one index over their joint values."""
    out = []
    for br in d.branches:
        regs = list(br.registers)
        free = {
            b.symbol: b
            for b in br.free
            if sum(1 for r in regs if b.symbol in sympy.sympify(r).free_symbols) == 1
        }
        runs: list[list[int]] = []
        for k, v in enumerate(regs):
            prev = regs[k - 1] if k else None
            if (
                v in free
                and v not in br.prob.free_symbols
                and runs
                and runs[-1][-1] == k - 1
                and prev in free
                and prev not in br.prob.free_symbols
            ):
                runs[-1].append(k)
                continue
            runs.append([k])
        if all(len(run) == 1 for run in runs):
            out.append(br)
            continue
        value = sympy.S.Zero
        bounds: list[Bound] = []
        for run in runs:
            size = sympy.Mul(*[layout.levels(k) for k in run])
            if len(run) == 1:
                value = value * size + regs[run[0]]
                bounds.extend(
                    b
                    for b in br.free
                    if b.symbol in sympy.sympify(regs[run[0]]).free_symbols and b not in bounds
                )
                continue
            b = new_bound("k", 0, size - 1)
            value = value * size + b.symbol
            bounds.append(b)
        out.append(Branch(expand(value), br.prob, tuple(bounds)))
    return Distribution(tuple(out))


def distribution(s: State | Mixed) -> Distribution:
    """The Born distribution of measuring every register of the ket or density matrix ``s``,
    derived from its path sum."""
    if isinstance(s, Mixed):
        layout = s.layout()
        zs = _register_symbols(layout)
        limits = {z: layout.levels(k) for k, z in enumerate(zs)}
        amp = amplitude(s.vec, zs + zs, limits)
        tr = trace(s)
        if tr == 0:
            raise PathSumError("cannot measure the zero matrix")
        return _born(layout, amp / tr, zs)
    if s.bra:
        raise PathSumError("measure a ket, not a bra")
    s = simplify(s)
    if all(t.amp == 0 for t in s.terms):
        raise PathSumError("cannot measure the zero vector")
    zs = _register_symbols(s)
    limits = {z: s.levels(k) for k, z in enumerate(zs)}
    amp = amplitude(s, zs, limits)
    norm = norm2(s)
    if norm == 0:
        raise PathSumError("cannot measure the zero vector")
    return _born(s, _abs2(amp) / norm, zs)


def _interval(e: Expr, free: dict[sympy.Symbol, Bound]) -> tuple[Expr, Expr] | None:
    """``(lo, hi)`` of an expression linear in the indices ``free`` with sign-known
    coefficients."""
    e = expand(e)
    lo = hi = sympy.S.Zero
    for term in sympy.Add.make_args(e):
        syms = term.free_symbols & set(free)
        if not syms:
            lo, hi = lo + term, hi + term
            continue
        if len(syms) != 1:
            return None
        (x,) = syms
        c = sympy.simplify(term / x)
        if x in c.free_symbols:
            return None
        b = free[x]
        if c.is_nonnegative:
            lo, hi = lo + c * b.lo, hi + c * b.hi
        elif c.is_nonpositive:
            lo, hi = lo + c * b.hi, hi + c * b.lo
        else:
            return None
    return lo, hi


def settle(e: Expr, free: Sequence[Bound]) -> Expr:
    """``e`` with each ``floor(u)`` replaced by the integer it takes over the ranges of the
    indices ``free``, when u less an integer part stays within one unit interval, and each
    ``Mod(u, m)`` by u less a multiple of m when that stays in ``0..m-1``."""
    table = {b.symbol: b for b in free}
    sizes = {
        b.hi + 1: sympy.Dummy(str(b.hi + 1), integer=True, positive=True)
        for b in free
        if b.lo == 0 and (b.hi + 1).is_Symbol
    }

    sized = {
        k: Bound(b.symbol, b.lo.xreplace(sizes), b.hi.xreplace(sizes)) for k, b in table.items()
    }
    unsized = {v: k for k, v in sizes.items()}

    def fold(f: sympy.floor) -> Expr:
        for whole, part in _integer_parts(f.args[0], 1):
            span = _interval(part.xreplace(sizes), sized)
            if span is None:
                continue
            lo, hi = span
            base = sympy.floor(lo)
            if base.has(sympy.floor):
                continue
            gap = sympy.simplify(hi - base - 1)
            if gap.is_negative and sympy.simplify(lo - base).is_nonnegative:
                return whole + base.xreplace(unsized)
        return f

    def fold_mod(f: sympy.Mod) -> Expr:
        top, m = f.args
        for _, part in _integer_parts(top, m):
            span = _interval(part, table)
            if span is None:
                continue
            lo, hi = span
            if sympy.simplify(lo).is_nonnegative and sympy.simplify(m - 1 - hi).is_nonnegative:
                return part
        return f

    e = e.replace(lambda x: isinstance(x, sympy.floor), fold)
    return e.replace(lambda x: isinstance(x, sympy.Mod), fold_mod)


def _integer_parts(u: Expr, m: Expr) -> Iterator[tuple[Expr, Expr]]:
    """``(w, u - m w)`` with w the sum of the terms of ``u / m`` that are integers, plainly
    and with a dividend a of each :func:`divisible` fact b | a read as ``b q``."""
    yield sympy.S.Zero, u
    for a, d in [(None, None), *_DIVISIBLE.get()]:
        q = sympy.Dummy("q", integer=True, positive=True)
        to, back = ({a: d * q}, {q: a / d}) if a is not None else ({}, {})
        terms = sympy.Add.make_args(expand(sympy.sympify(u).subs(to) / m))
        whole = sympy.Add(*(t for t in terms if t.is_integer))
        if whole == 0:
            continue
        whole = expand(whole.xreplace(back))
        yield whole, expand(u - m * whole)


def map_outcomes(d: Distribution, fn: Callable[[Expr], Expr]) -> Distribution:
    """The distribution of ``fn(value)``: indices the new values no longer depend on summed
    out, equal outcomes merged."""
    out: list[Branch] = []
    for br in d.branches:
        value = settle(sympy.sympify(fn(br.value)), br.free)
        prob, free = br.prob, list(br.free)
        for b in list(free):
            if b.symbol not in value.free_symbols:
                prob = _close(prob, [b], {})
                free.remove(b)
        prob = _delta_tidy(tidy(prob))
        if prob == 0:
            continue
        same_branch = next(
            (k for k, o in enumerate(out) if not o.free and not free and same(o.value, value)),
            None,
        )
        if same_branch is not None:
            o = out[same_branch]
            out[same_branch] = Branch(o.value, tidy(o.prob + prob))
            continue
        out.append(Branch(value, prob, tuple(free)))
    return Distribution(tuple(_merge_overlaps(out)))


def _outcomes(br: Branch, cap: int = 64) -> list[dict[sympy.Symbol, Expr]] | None:
    """Every valuation of the free indices of ``br`` over their concrete ranges, or None when
    a range is symbolic or there are more than ``cap``."""
    vals: list[dict[sympy.Symbol, Expr]] = [{}]
    for b in br.free:
        if not (b.lo.is_Integer and b.hi.is_Integer):
            return None
        vals = [{**v, b.symbol: sympy.Integer(k)} for v in vals for k in range(b.lo, b.hi + 1)]
        if len(vals) > cap:
            return None
    return vals


def _merge_overlaps(branches: list[Branch]) -> list[Branch]:
    """Branches over concrete ranges whose outcomes meet another branch's split into single
    outcomes, and equal outcomes merged."""
    sets: list[set[Expr] | None] = []
    for br in branches:
        vals = _outcomes(br)
        sets.append(None if vals is None else {br.value.xreplace(v) for v in vals})
    hit = [
        a is not None and any(b is not None and k != j and a & b for j, b in enumerate(sets))
        for k, a in enumerate(sets)
    ]
    if not any(hit):
        return branches
    out: list[Branch] = [br for br, h in zip(branches, hit, strict=True) if not h]
    merged: dict[Expr, Expr] = {}
    for br, h in zip(branches, hit, strict=True):
        if h:
            for v in _outcomes(br) or []:
                value = br.value.xreplace(v)
                merged[value] = merged.get(value, sympy.S.Zero) + br.prob.xreplace(v)
    for value, prob in merged.items():
        prob = _simp(_delta_tidy(tidy(prob)))
        if prob != 0:
            out.append(Branch(value, prob))
    return out


def certain(d: Distribution) -> Expr | None:
    """The value of a distribution with one outcome of probability 1, else None."""
    if len(d.branches) == 1 and not d.branches[0].free and same(d.branches[0].prob, 1):
        return d.branches[0].value
    return None


def norm2(s: State) -> Expr:
    """``<s|s>`` by symbolic contraction."""
    ket = dagger(s) if s.bra else s
    return inner(dagger(ket), ket)


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


def _multiplicative(w: Expr, u: sympy.Symbol) -> tuple[Expr, Expr, bool] | None:
    """``(G, M, guarded)`` when ``w = Mod(u * G, M)``, or ``w`` is that below M and u
    otherwise, G free of u."""
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
    return g, w.args[1], guard is not None


def _bit_sum(e: Expr, j: sympy.Symbol, lo: Expr, hi: Expr) -> Expr:
    """``sum_{j=lo}^{hi} e``, with ``sum_{j=0}^{hi} 2^j bit(x, j) = x mod 2^(hi+1)`` and
    ``sum_{j=0}^{hi} d^j (floor(x / d^j) mod d) = x mod d^(hi+1)``, per term."""
    out: Expr = sympy.S.Zero
    for term in sympy.Add.make_args(expand(e)):
        out += _digit_term_sum(term, j, lo, hi)
    return out


def _digit_term_sum(term: Expr, j: sympy.Symbol, lo: Expr, hi: Expr) -> Expr:
    if j not in term.free_symbols:
        return term * (hi - lo + 1)
    if lo == 0:
        for b in term.atoms(bit):
            c = sympy.simplify(term / (2**j * b))
            if b.args[1] == j and j not in c.free_symbols:
                return c * sympy.Mod(b.args[0], 2 ** expand(hi + 1))
        for m in term.atoms(sympy.Mod):
            f, d = m.args
            if not (isinstance(f, sympy.floor) and j not in d.free_symbols):
                continue
            x = sympy.simplify(f.args[0] * d**j)
            c = sympy.simplify(term / (d**j * m))
            if j not in x.free_symbols and j not in c.free_symbols:
                return c * sympy.Mod(x, d ** expand(hi + 1))
    return sympy.Sum(term, (j, lo, hi)).doit()


def _loop_phase(amp: Expr, j: sympy.Symbol, lo: Expr, hi: Expr) -> Expr:
    """``prod_{j=lo}^{hi} amp``, exponents of j-free bases summed by :func:`_bit_sum`."""
    out = sympy.S.One
    for f in sympy.Mul.make_args(sympy.powsimp(amp)):
        if j not in f.free_symbols:
            out *= f ** expand(hi - lo + 1)
        elif isinstance(f, sympy.exp):
            out *= sympy.exp(_bit_sum(f.args[0], j, lo, hi))
        elif f.is_Pow and j not in f.base.free_symbols:
            out *= f.base ** _bit_sum(f.exp, j, lo, hi)
        else:
            return sympy.Product(amp, (j, lo, hi)).doit()
    return out


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


def _xor_mask(w: Expr, u: sympy.Symbol, j: sympy.Symbol) -> Expr | None:
    """h when ``w = xor(u, h 2^j)`` with h free of u and j, else None."""
    if not isinstance(w, xor) or u not in w.args:
        return None
    other = w.args[1] if w.args[0] == u else w.args[0]
    h = sympy.simplify(other / 2**j)
    if h.free_symbols & {u, j}:
        return None
    return h


def fold(
    cur: State, step: State, probes: Sequence[sympy.Symbol], j: sympy.Symbol, lo: Expr, hi: Expr
) -> State:
    """``cur`` after ``step`` for ``j = lo..hi``; ``step`` is one iteration on the probe basis
    state ``|probes>``. Registers must stay fixed or update as ``y -> y G(j) mod M`` or
    ``y -> y xor h 2^j`` with h a one-qubit register."""
    if step.bra or len(step.terms) != 1 or step.terms[0].bound or not _same_layout(cur, step):
        raise PathSumError("a symbolic loop step must map basis states to basis states")
    t = step.terms[0]
    updates: dict[int, tuple[Expr, Expr, bool]] = {}
    masks: dict[int, Expr] = {}
    for k, (u, w) in enumerate(zip(probes, t.kets, strict=True)):
        w = inline(w)
        if w == u:
            continue
        h = _xor_mask(w, u, j)
        if h is not None and h in probes:
            r = probes.index(h)
            if same(cur.widths[r], 1) and same(cur.dims[r], TWO):
                masks[k] = h
                continue
        found = _multiplicative(w, u)
        if found is None:
            raise PathSumError(
                f"register {k} does not update as y -> y*G mod M or y -> y xor h*2^j in the loop"
            )
        updates[k] = found
    moving = {probes[k] for k in (*updates, *masks)}
    if any(g.free_symbols & moving for g, _, _ in updates.values()) or set(masks.values()) & moving:
        raise PathSumError("loop updates of different registers depend on each other")
    amp = inline(t.amp)
    if amp.free_symbols & moving:
        raise PathSumError("the loop phase depends on a register the loop changes")
    terms = []
    for term in cur.terms:
        fixed = {
            probes[k]: v for k, v in enumerate(term.kets) if k not in updates and k not in masks
        }
        kets = list(term.kets)
        for k, h in masks.items():
            kets[k] = xor(term.kets[k], h.xreplace(fixed) * (2 ** expand(hi + 1) - 2**lo))
        for k, (g, modulus, guarded) in updates.items():
            below = _below(term.kets[k], modulus)
            if not (below or guarded):
                raise PathSumError(f"register {k} is not known to start below {modulus}")
            product = _product_mod(g.xreplace(fixed), j, lo, hi, modulus)
            if same(product, 1) or same(product, modulus + 1):
                continue
            kets[k] = sympy.Mod(term.kets[k] * product, modulus)
            if not below:
                kets[k] = sympy.Piecewise(
                    (kets[k], sympy.StrictLessThan(term.kets[k], modulus)), (term.kets[k], True)
                )
        total = _loop_phase(amp.xreplace(fixed), j, lo, hi) if amp != 1 else 1
        terms.append(Term(term.amp * total, term.bound, tuple(kets)))
    return _with(cur, terms=terms)
