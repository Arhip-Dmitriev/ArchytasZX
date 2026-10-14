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

"""Symbolic path-sum states and operators over qubit registers.

A :class:`State` is a tuple of register widths and a sum of :class:`Term` s; a term is
``amp * sum_{bound} |k_1>|k_2>...`` with sympy expressions for the amplitude, the register
values and the bound ranges. Register values are big-endian integers of their register's
width. Widths and values may be symbolic.

Operators (:class:`Gate`, :class:`Identity`, :class:`Power`, :class:`Tensor`,
:class:`Compose`, :class:`Outer`, :class:`Scaled`, :class:`OpSum`) act through
:func:`apply`. A gate is a callable from register values to a state, so applying it is
substitution. The tensor power of a one-qubit gate or state uses the closed form over bits
(``pop`` and ``bitdot`` exponents).

:func:`measure` measures every register: numerically (sampled with a numpy generator) when
the state is concrete, otherwise symbolically (an outcome symbol, the projected state and the
constraints it implies). :func:`inner` contracts a bra with a ket.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

Expr: TypeAlias = sympy.Expr


class PathSumError(Exception):
    """A path-sum operation that is undefined on its inputs."""


def _bit_count(v: int) -> int:
    return int(v).bit_count()


class xor(sympy.Function):  # type: ignore[misc]
    """Bitwise exclusive or of two non-negative integers."""

    nargs = 2

    @classmethod
    def eval(cls, a: Expr, b: Expr) -> Expr | None:
        if a.is_Integer and b.is_Integer:
            return sympy.Integer(int(a) ^ int(b))
        if a == 0:
            return b
        if b == 0:
            return a
        if a == b:
            return sympy.S.Zero
        return None

    def _eval_is_integer(self) -> bool:
        return True


class pop(sympy.Function):  # type: ignore[misc]
    """Population count of a non-negative integer."""

    nargs = 1

    @classmethod
    def eval(cls, v: Expr) -> Expr | None:
        return sympy.Integer(_bit_count(int(v))) if v.is_Integer else None

    def _eval_is_integer(self) -> bool:
        return True


class bitdot(sympy.Function):  # type: ignore[misc]
    """Population count of the bitwise and of two non-negative integers."""

    nargs = 2

    @classmethod
    def eval(cls, a: Expr, b: Expr) -> Expr | None:
        if a.is_Integer and b.is_Integer:
            return sympy.Integer(_bit_count(int(a) & int(b)))
        if a == 0 or b == 0:
            return sympy.S.Zero
        return None

    def _eval_is_integer(self) -> bool:
        return True


class bits(sympy.Function):  # type: ignore[misc]
    """``bits(z, lo, hi, w)``: bits ``lo..hi-1`` of the width-``w`` big-endian bitstring z."""

    nargs = 4

    @classmethod
    def eval(cls, z: Expr, lo: Expr, hi: Expr, w: Expr) -> Expr | None:
        if lo == 0 and sympy.expand(hi - w) == 0:
            return z
        if all(a.is_Integer for a in (z, lo, hi, w)):
            return sympy.Integer((int(z) >> (int(w) - int(hi))) & ((1 << (int(hi) - int(lo))) - 1))
        return None

    def _eval_is_integer(self) -> bool:
        return True


class shr(sympy.Function):  # type: ignore[misc]
    """``shr(v, k)``: v shifted right by k bits."""

    nargs = 2

    @classmethod
    def eval(cls, v: Expr, k: Expr) -> Expr | None:
        if v.is_Integer and k.is_Integer:
            return sympy.Integer(int(v) >> int(k))
        if v == 0 or k == 0:
            return v
        return None

    def _eval_is_integer(self) -> bool:
        return True


class powmod(sympy.Function):  # type: ignore[misc]
    """``powmod(b, e, m)``: ``b**e % m`` by modular exponentiation."""

    nargs = 3

    @classmethod
    def eval(cls, b: Expr, e: Expr, m: Expr) -> Expr | None:
        if b.is_Integer and e.is_Integer and m.is_Integer and int(e) >= 0 and int(m) != 0:
            return sympy.Integer(pow(int(b), int(e), int(m)))
        return None

    def _eval_is_integer(self) -> bool:
        return True


def to_powmod(expr: Expr) -> Expr:
    """``Mod(b**e, m)`` subterms rewritten as :class:`powmod`."""
    return expr.replace(
        lambda e: isinstance(e, sympy.Mod) and e.args[0].is_Pow,
        lambda e: powmod(e.args[0].base, e.args[0].exp, e.args[1]),
    )


_POP = np.frompyfunc(_bit_count, 1, 1)
_BITDOT = np.frompyfunc(lambda a, b: _bit_count(int(a) & int(b)), 2, 1)
_POWMOD = np.frompyfunc(lambda b, e, m: pow(int(b), int(e), int(m)), 3, 1)


def _bits_impl(z: object, lo: object, hi: object, w: object) -> object:
    zs = np.asarray(z, dtype=object)
    return (zs >> (w - hi)) & ((1 << (hi - lo)) - 1)  # type: ignore[operator]


_IMPLS: dict[str, object] = {
    "xor": lambda a, b: np.bitwise_xor(np.asarray(a, dtype=object), np.asarray(b, dtype=object)),
    "pop": _POP,
    "bitdot": _BITDOT,
    "bits": _bits_impl,
    "shr": lambda v, k: np.right_shift(np.asarray(v, dtype=object), np.asarray(k, dtype=object)),
    "powmod": _POWMOD,
}


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


@dataclass(frozen=True)
class State:
    """Register widths (``None`` while unresolved) and a sum of terms; a bra when ``bra``."""

    widths: tuple[Expr | None, ...]
    terms: tuple[Term, ...]
    bra: bool = False

    @property
    def total_width(self) -> Expr:
        if any(w is None for w in self.widths):
            raise PathSumError("a register width is unresolved")
        return sympy.Add(*self.widths) if self.widths else sympy.S.Zero


def new_bound(name: str, lo: Expr, hi: Expr) -> Bound:
    """A bound index with a fresh integer dummy named ``name``."""
    return Bound(sympy.Dummy(name, integer=True), sympy.sympify(lo), sympy.sympify(hi))


def same(a: Expr | None, b: Expr | None) -> bool:
    """Symbolic equality of two widths or values."""
    if a is None or b is None:
        return False
    diff = sympy.expand(sympy.sympify(a) - sympy.sympify(b))
    return bool(diff == 0 or sympy.simplify(diff) == 0)


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
    renamed = Term(
        term.amp.xreplace(mapping),
        tuple(
            Bound(mapping[b.symbol], b.lo.xreplace(mapping), b.hi.xreplace(mapping))
            for b in term.bound
        ),
        tuple(k.xreplace(mapping) for k in term.kets),
    )
    return renamed


def basis(value: Expr, width: Expr | None) -> State:
    """The basis ket ``|value>`` on one register."""
    return State((width,), (Term(sympy.S.One, (), (sympy.sympify(value),)),))


def basis_kets(values: Sequence[Expr], widths: Sequence[Expr | None]) -> State:
    """The basis ket with one register per value."""
    return State(tuple(widths), (Term(sympy.S.One, (), tuple(values)),))


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
    return State(a.widths + b.widths, terms, a.bra)


def scale(c: Expr, s: State) -> State:
    """``c * s``."""
    return State(s.widths, tuple(Term(c * t.amp, t.bound, t.kets) for t in s.terms), s.bra)


def add(a: State, b: State) -> State:
    """``a + b``; register layouts must agree or have the same total width."""
    if a.bra != b.bra:
        raise PathSumError("cannot add a ket and a bra")
    if len(a.widths) != len(b.widths) or not all(map(same, a.widths, b.widths)):
        if not same(a.total_width, b.total_width):
            raise PathSumError("cannot add states of different widths")
        a, b = merge(a, 0, len(a.widths)), merge(b, 0, len(b.widths))
    return State(a.widths, a.terms + b.terms, a.bra)


def dagger(s: State) -> State:
    """The conjugate transpose: ket to bra and bra to ket."""
    return State(
        s.widths,
        tuple(Term(sympy.conjugate(t.amp), t.bound, t.kets) for t in s.terms),
        not s.bra,
    )


def merge(s: State, i: int, j: int) -> State:
    """Registers ``i..j-1`` joined into one big-endian register."""
    if j - i <= 1:
        return s
    widths = s.widths[i:j]
    if any(w is None for w in widths):
        raise PathSumError("a register width is unresolved")
    shifts = [sympy.Add(*widths[k + 1 :]) for k in range(len(widths))]
    terms = []
    for t in s.terms:
        value = sympy.Add(*(v * 2**sh for v, sh in zip(t.kets[i:j], shifts, strict=True)))
        terms.append(Term(t.amp, t.bound, t.kets[:i] + (value,) + t.kets[j:]))
    return State(s.widths[:i] + (sympy.Add(*widths),) + s.widths[j:], tuple(terms), s.bra)


def _full_range(b: Bound, width: Expr) -> bool:
    return b.lo == 0 and same(b.hi, 2**width - 1)


def split(s: State, i: int, high: Expr) -> State:
    """Register ``i`` split into a ``high``-bit register and the remaining low bits."""
    width = s.widths[i]
    if width is None:
        raise PathSumError("a register width is unresolved")
    low = sympy.expand(width - high)
    terms = []
    for t in s.terms:
        v = t.kets[i]
        bound = next((b for b in t.bound if b.symbol == v), None)
        if bound is not None and _full_range(bound, width):
            hb, lb = new_bound(v.name, 0, 2**high - 1), new_bound(v.name, 0, 2**low - 1)
            t = _subs_term(t, {v: hb.symbol * 2**low + lb.symbol})
            t = Term(t.amp, t.bound + (hb, lb), t.kets)
            parts = (hb.symbol, lb.symbol)
        else:
            parts = (shr(v, low), sympy.Mod(v, 2**low))
        terms.append(Term(t.amp, t.bound, t.kets[:i] + parts + t.kets[i + 1 :]))
    return State(s.widths[:i] + (high, low) + s.widths[i + 1 :], tuple(terms), s.bra)


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


def table(s0: State, s1: State) -> list[list[Expr]]:
    """``A[a][b]``: the amplitude of ``|b>`` in the one-qubit states ``s0`` (a=0), ``s1``."""
    out = []
    for s in (s0, s1):
        row = [sympy.S.Zero, sympy.S.Zero]
        for amp, kets in expand_terms(s):
            if len(kets) != 1 or kets[0] not in (0, 1):
                raise PathSumError("not a one-qubit map")
            row[int(kets[0])] += amp
        out.append([sympy.simplify(x) for x in row])
    return out


def _power_from_table(a: list[list[Expr]], v: Expr, w: Expr, hint: str) -> State:
    a00, a01, a10, a11 = a[0][0], a[0][1], a[1][0], a[1][1]
    if all(x != 0 for x in (a00, a01, a10, a11)):
        y = new_bound(hint, 0, 2**w - 1)
        t = sympy.simplify(a11 * a00 / (a10 * a01))
        amp = (
            a00**w * (a10 / a00) ** pop(v) * (a01 / a00) ** pop(y.symbol) * t ** bitdot(v, y.symbol)
        )
        return State((w,), (Term(tidy(amp), (y,), (y.symbol,)),))
    if a01 == 0 and a10 == 0 and a00 != 0:
        return State((w,), (Term(tidy(a00**w * (a11 / a00) ** pop(v)), (), (v,)),))
    if a00 == 0 and a11 == 0 and a01 != 0:
        amp = a01**w * (a10 / a01) ** pop(v)
        return State((w,), (Term(tidy(amp), (), (2**w - 1 - v,)),))
    raise PathSumError("no closed-form tensor power for this one-qubit map")


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


def tidy(expr: Expr) -> Expr:
    """``expr`` with numeric bases of symbolic powers split into prime powers."""

    def rewrite(p: sympy.Pow) -> Expr:
        primes = _prime_powers(p.base)
        if primes is None:
            return p
        return sympy.Mul(*(sympy.Integer(q) ** sympy.expand(k * p.exp) for q, k in primes.items()))

    expr = sympy.sympify(expr)
    return expr.replace(
        lambda e: e.is_Pow and e.base.is_number and not e.exp.is_number and e.base != -1,
        rewrite,
    )


def power_state(s: State, n: Expr, hint: str = "x") -> State:
    """``s^(x)n``; closed form for one-qubit states, repeated tensor for concrete n."""
    n = sympy.sympify(n)
    if len(s.widths) == 1 and same(s.widths[0], 1):
        row = [sympy.S.Zero, sympy.S.Zero]
        for amp, kets in expand_terms(s):
            row[int(kets[0])] += amp
        a0, a1 = (sympy.simplify(x) for x in row)
        if a0 != 0 and a1 != 0:
            y = new_bound(hint, 0, 2**n - 1)
            amp = tidy(a0**n * (a1 / a0) ** pop(y.symbol))
            return State((n,), (Term(amp, (y,), (y.symbol,)),), s.bra)
        value, amp = (sympy.S.Zero, a0) if a1 == 0 else (2**n - 1, a1)
        return State((n,), (Term(tidy(amp**n), (), (value,)),), s.bra)
    if n.is_Integer and n >= 0:
        out = State((), (Term(sympy.S.One, (), ()),), s.bra)
        for _ in range(int(n)):
            out = tensor(out, s)
        return out
    raise PathSumError("a symbolic tensor power needs a one-qubit state")


class Operator:
    """A linear map on registers; see :func:`apply`."""


@dataclass(frozen=True, eq=False)
class Gate(Operator):
    """``fn(values, widths) -> State`` on ``arity`` registers; ``qubit`` gates take width 1."""

    name: str
    arity: int
    fn: Callable[[tuple[Expr, ...], tuple[Expr, ...]], State]
    qubit: bool = False
    oracle: bool = False
    source: str = ""
    checked: set[tuple[int, ...]] = field(default_factory=set)


@dataclass(frozen=True)
class Identity(Operator):
    """The identity on ``width`` qubits, or on whatever it is aligned with."""

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
    return simplify(_apply(op, s, 0, len(s.widths)))


def _apply_map(
    s: State, i: int, j: int, fn: Callable[[tuple[Expr, ...], tuple[Expr, ...]], State]
) -> State:
    out_widths: tuple[Expr | None, ...] | None = None
    terms: list[Term] = []
    for t in s.terms:
        res = fn(t.kets[i:j], s.widths[i:j])
        if out_widths is None:
            out_widths = res.widths
        elif len(res.widths) != len(out_widths) or not all(map(same, res.widths, out_widths)):
            raise PathSumError("an operator gave different output widths on different terms")
        for r in (fresh(x) for x in res.terms):
            terms.append(Term(t.amp * r.amp, t.bound + r.bound, t.kets[:i] + r.kets + t.kets[j:]))
    assert out_widths is not None
    return State(s.widths[:i] + out_widths + s.widths[j:], tuple(terms), s.bra)


def _apply(op: Operator, s: State, i: int, j: int) -> State:
    if isinstance(op, Gate):
        return _apply_gate(op, s, i, j)
    if isinstance(op, Identity):
        if op.width is not None and not same(sympy.Add(*s.widths[i:j]), op.width):
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
        return _apply_map(s, i, j, lambda vals, ws: scale(inner(bra, basis_kets(vals, ws)), op.ket))
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
    if g.qubit and not all(same(w, 1) for w in s.widths[i:j]):
        raise PathSumError(f"{g.name} acts on single qubits; use {g.name}^@w for a register")

    def run(vals: tuple[Expr, ...], ws: tuple[Expr, ...]) -> State:
        res = g.fn(vals, ws)
        if g.oracle and (
            len(res.terms) != 1 or res.terms[0].bound or sympy.simplify(res.terms[0].amp) != 1
        ):
            raise PathSumError(f"oracle {g.name} must map basis states to basis states")
        return res

    out = _apply_map(s, i, j, run)
    if g.oracle:
        _check_oracle(g, s.widths[i:j], run)
    return out


def _check_oracle(g: Gate, widths: Sequence[Expr | None], run: Callable[..., State]) -> None:
    """Raises unless the oracle permutes basis states; concrete widths only."""
    if not all(w is not None and w.is_Integer for w in widths):
        return
    key = tuple(int(w) for w in widths)  # type: ignore[arg-type]
    if key in g.checked:
        return
    bounds = [new_bound("x", 0, 2**w - 1) for w in key]
    probe = State(
        tuple(widths), (Term(sympy.S.One, tuple(bounds), tuple(b.symbol for b in bounds)),)
    )
    out = _apply_map(probe, 0, len(key), run)
    if not is_concrete(out):
        return
    keys, _ = enumerate_state(out)
    if len(np.unique(keys)) != len(keys):
        raise PathSumError(f"oracle {g.name} is not reversible on widths {key}")
    g.checked.add(key)


def _apply_power(p: Power, s: State, i: int, j: int) -> State:
    inner_op = p.op
    if isinstance(inner_op, Identity):
        return s
    if isinstance(inner_op, Gate) and inner_op.arity == 1 and inner_op.qubit:
        s = merge(s, i, j)
        if not same(s.widths[i], p.count):
            raise PathSumError(f"{inner_op.name}^@{p.count} does not match register width")
        one = sympy.Integer(1)
        a = table(inner_op.fn((sympy.S.Zero,), (one,)), inner_op.fn((one,), (one,)))
        return _apply_map(s, i, i + 1, lambda vals, ws: _power_from_table(a, vals[0], ws[0], "x"))
    count = sympy.sympify(p.count)
    if count.is_Integer and count >= 0:
        return _apply(Tensor((inner_op,) * int(count)), s, i, j)
    raise PathSumError("a symbolic tensor power needs a one-qubit gate")


def _span_width(op: Operator) -> Expr | None:
    if isinstance(op, Power):
        inner_w = _span_width(op.op)
        return None if inner_w is None else sympy.expand(inner_w * op.count)
    if isinstance(op, Identity):
        return op.width
    if isinstance(op, Gate) and op.qubit:
        return sympy.Integer(op.arity)
    if isinstance(op, Outer):
        return op.bra.total_width
    if isinstance(op, (Scaled,)):
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
        acc = sympy.expand(acc + w)
        r += 1
        if same(acc, width):
            return s, r, j
        over = sympy.expand(acc - width)
        if over.is_Integer and over > 0:
            s = split(s, r - 1, sympy.expand(w - over))
            return s, r, j + 1
    raise PathSumError(f"no registers of total width {width} to act on")


def _bound_ok(sym: Expr, other: Expr, bound: list[Bound], width: Expr) -> bool:
    b = next((x for x in bound if x.symbol == sym), None)
    return b is not None and _full_range(b, width) and sym not in other.free_symbols


def inner(a: State, b: State) -> Expr:
    """``<a|b>`` for a bra ``a`` and a ket ``b``."""
    if not a.bra or b.bra:
        raise PathSumError("an inner product needs a bra then a ket")
    if len(a.widths) != len(b.widths) or not all(map(same, a.widths, b.widths)):
        if not same(a.total_width, b.total_width):
            raise PathSumError("cannot contract states of different widths")
        a, b = merge(a, 0, len(a.widths)), merge(b, 0, len(b.widths))
    widths = a.widths
    if is_concrete(a) and is_concrete(b):
        return _inner_numeric(a, b)
    total: Expr = sympy.S.Zero
    for ta in a.terms:
        for tb in (fresh(t) for t in b.terms):
            total += _contract(ta, tb, widths)
    return sympy.simplify(total)


def _contract(ta: Term, tb: Term, widths: Sequence[Expr | None]) -> Expr:
    amp = ta.amp * tb.amp
    bound = list(ta.bound + tb.bound)
    eqs = [(lk, rk, w) for lk, rk, w in zip(ta.kets, tb.kets, widths, strict=True)]
    deltas: list[Expr] = []
    while eqs:
        lhs, rhs, w = eqs.pop()
        if sympy.expand(lhs - rhs) == 0:
            continue
        pick = next(
            ((x, y) for x, y in ((lhs, rhs), (rhs, lhs)) if _bound_ok(x, y, bound, w)), None
        )
        if pick is not None:
            sym, val = pick
            mapping = {sym: val}
            amp = amp.xreplace(mapping)
            eqs = [(x.xreplace(mapping), y.xreplace(mapping), ww) for x, y, ww in eqs]
            deltas = [d.xreplace(mapping) for d in deltas]
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
    arg = sympy.expand(sympy.Add(*(e.args[0] for e in exps)) / (2 * sympy.pi * sympy.I))
    coeff = sympy.expand(sympy.diff(arg, sym))
    if sym in coeff.free_symbols:
        return None
    count = b.hi + 1
    level = sympy.expand(coeff * count)
    if not level.is_integer:
        return None
    level = sympy.Add(*(p for p in sympy.Add.make_args(level) if not (p / count).is_integer))
    other = sympy.exp(2 * sympy.pi * sympy.I * sympy.expand(arg - coeff * sym))
    parts = sympy.Add.make_args(level)
    signs = [p.as_coeff_Mul() for p in parts]
    if not level.is_Integer and len(parts) <= 2 and all(abs(c) == 1 for c, _ in signs):
        pos = [x for c, x in signs if c == 1]
        neg = [x for c, x in signs if c == -1]
        if len(parts) == 1:
            return rest * other * count * sympy.KroneckerDelta(parts[0] * signs[0][0], 0)
    terms = [p for p in parts if not p.is_Integer]
    const = sympy.Add(*(p for p in parts if p.is_Integer))
    if len(terms) == 1 and const != 0 and abs(terms[0].as_coeff_Mul()[0]) == 1:
        c, u = terms[0].as_coeff_Mul()
        target = -c * const
        if target.is_negative:
            target = sympy.expand(count + target)
        return rest * other * count * sympy.KroneckerDelta(u, target)
        if len(pos) == 1 and len(neg) == 1:
            return rest * other * count * sympy.KroneckerDelta(pos[0], neg[0])
    if level.is_Integer:
        return (
            rest * other * (count if int(level) % int(count) == 0 else 0)
            if count.is_Integer
            else None
        )
    return None


def _sum_out(amp: Expr, bound: list[Bound]) -> Expr:
    for b in bound:
        if b.symbol not in amp.free_symbols:
            amp = amp * (b.hi - b.lo + 1)
            continue
        geo = _geometric(amp, b)
        amp = geo if geo is not None else sympy.Sum(amp, (b.symbol, b.lo, b.hi)).doit()
    return amp


def _parity(amp: Expr, b: Bound, bound: Sequence[Bound]) -> Expr | None:
    """``sum_b C (-1)^{b.v}`` over a full ``2^w`` range as ``2^w C [v = 0]``; else None."""
    sym = b.symbol
    if b.lo != 0:
        return None
    factors = sympy.Mul.make_args(amp)
    hits = [
        f
        for f in factors
        if f.is_Pow and f.base == -1 and isinstance(f.exp, bitdot) and sym in f.exp.args
    ]
    if len(hits) != 1:
        return None
    rest = sympy.Mul(*(f for f in factors if f is not hits[0]))
    v = next((x for x in hits[0].exp.args if x != sym), sym)
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


def simplify(s: State) -> State:
    """Terms tidied, zero terms dropped, and bound indices outside every ket summed out."""
    terms = []
    for t in s.terms:
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
    out = State(s.widths, tuple(terms), s.bra)
    return _combine(out) if len(terms) > 1 and is_concrete(out) else out


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
    return State(s.widths, tuple(terms), s.bra)


def is_concrete(s: State) -> bool:
    """Every width, range, amplitude and value is a number once bound indices are fixed."""
    if not all(w is not None and sympy.sympify(w).is_Integer for w in s.widths):
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
    fn = sympy.lambdify(syms, expr, modules=[numeric_impls(expr), "numpy"])
    args = grid if dtype is object else [_as_numeric(g) for g in grid]
    out = np.asarray(fn(*args), dtype=dtype)  # type: ignore[call-overload]
    return np.broadcast_to(out, (size,)).copy() if out.shape != (size,) else out


def _as_numeric(g: np.ndarray) -> np.ndarray:
    if g.size and max(abs(int(g.max())), abs(int(g.min()))) < 2**62:
        return g.astype(np.int64)
    return g.astype(float)


def enumerate_state(s: State) -> tuple[np.ndarray, np.ndarray]:
    """``(keys, amps)`` for every basis path of a concrete state; keys concatenate registers."""
    widths = [int(w) for w in s.widths]  # type: ignore[arg-type]
    shifts = [sum(widths[k + 1 :]) for k in range(len(widths))]
    all_keys, all_amps = [], []
    for t in s.terms:
        syms = [b.symbol for b in t.bound]
        grid, size = _grid(t)
        key = np.zeros(size, dtype=object)
        for k, sh in zip(t.kets, shifts, strict=True):
            vals = _evaluate(k, syms, grid, size, object)
            key = key + np.left_shift(vals, sh)
        amps = _evaluate(t.amp, syms, grid, size, complex)
        all_keys.append(key)
        all_amps.append(amps)
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


def decode(key: int, widths: Sequence[int]) -> tuple[sympy.Integer, ...]:
    """Register values of a concatenated key."""
    out = []
    shift = sum(widths)
    for w in widths:
        shift -= w
        out.append(sympy.Integer((key >> shift) & ((1 << w) - 1)))
    return tuple(out)


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
        return Measurement(_concat(named, s.widths), width, basis_kets(values, s.widths))
    if is_concrete(s):
        return _measure_numeric(s, rng)
    return _measure_symbolic(s, symbol)


def _concat(values: Sequence[Expr], widths: Sequence[Expr | None]) -> Expr:
    total = sympy.Add(*widths)
    offsets = [sympy.Add(*widths[:k]) for k in range(len(widths))]
    zs = {v.args[0] for v in values if isinstance(v, bits)}
    if len(zs) == 1 and all(
        isinstance(v, bits)
        and same(v.args[1], off)
        and same(v.args[2], off + w)
        and same(v.args[3], total)
        for v, off, w in zip(values, offsets, widths, strict=True)
    ):
        return next(iter(zs))
    shifts = [sympy.Add(*widths[k + 1 :]) for k in range(len(widths))]
    return sympy.Add(*(v * 2**sh for v, sh in zip(values, shifts, strict=True)))


def _measure_numeric(s: State, rng: np.random.Generator) -> Measurement:
    keys, acc = _accumulate(s)
    probs = np.abs(acc) ** 2
    total = probs.sum()
    if total <= 0:
        raise PathSumError("cannot measure the zero vector")
    pick = int(rng.choice(len(keys), p=probs / total))
    key = int(keys[pick])
    widths = [int(w) for w in s.widths]  # type: ignore[arg-type]
    values = decode(key, widths)
    return Measurement(sympy.Integer(key), s.total_width, basis_kets(values, s.widths), random=True)


def _measure_symbolic(s: State, z: sympy.Symbol) -> Measurement:
    widths = s.widths
    total = s.total_width
    offsets = [sympy.Add(*widths[:k]) for k in range(len(widths))]
    slices = [
        bits(z, off, sympy.expand(off + w), total) for off, w in zip(offsets, widths, strict=True)
    ]
    if len(s.terms) != 1:
        return Measurement(z, total, basis_kets(slices, widths))
    t = s.terms[0]
    mapping: dict[sympy.Basic, Expr] = {}
    for v, w, sl in zip(t.kets, widths, slices, strict=True):
        b = next((x for x in t.bound if x.symbol == v), None)
        if b is not None and v not in mapping and _full_range(b, w):
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
    return Measurement(z, total, basis_kets(values, widths), tuple(constraints))
