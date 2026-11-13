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

"""Naive translator: circuit IR to an ArchytasZX DSL body, one generator group per gate, every
node named and every wire explicit, no simplification.

Gate encodings (``D`` the wire dimension; scalars as written into the ``scalar`` line):

  Init v      X(0 -> 1) phase{k: -v k/D}, D^(-1/2)      Post v   X(1 -> 0) phase{k: v k/D}, D^(-1/2)
  Discard     G(1 -> 0)                                   Measure  Z(1 -> 2), out1 to G
  X a         X(1 -> 1) phase{k: -a k/D}                  Z a      Z(1 -> 1) phase{k: a k/D}
  F p         p mod 4 F nodes                             ZPhase   Z(1 -> 1) phase{...}
  XPhase      X(1 -> 1) phase{...}                        Swap     no node
  CX s        Z(1 -> 2) on c, X(2 -> 1) on t, joined (through F F when s = -1), D^(1/2)
  CZ s        Z(1 -> 2) on c, Z(2 -> 1) on t, joined through F (F^3 when s = -1), D^(1/2)
  PhaseFn     binders to one leg of the product dimension, a Z spider with the phase table,
              splitters back (concrete dimensions only)
  MCZ, MCX    PhaseFn on controls and target (MCX between F and F^3 on the target)
  Oracle      F, PhaseFn f(x) y / D, F^3 on y
  Perm        a fresh Z(0 -> 1) leg t, PhaseFn -t p(x)/D on (t, x), Z(1 -> 0) on x, F on t,
              D^(-1/2) (concrete dimensions only)

A register's nodes go in one node-scope box per count; a gate between a register and a single
wire puts the single wire's spider outside the box, where it takes one leg per copy. A phase
table in a box takes each single wire through a Z(1 -> 2) outside the box, its out1 into the
copy's binder and the copy's splitter end into a Z(1 -> 0). A wire never touched gets an
identity Z(1 -> 1).

A Chain is one named box over its body's nodes, with its index, linked from each carry's end
after the exchange to the first port the carry entered in the copy (chain NAME: p -> q; a
reversed Chain writes q -> p); a carry or register with no node in the copy gets an identity
Z(1 -> 1) there. A carry that ends in the copy takes the register's end, and the register's
copy has no output. Gates on a chain's registers before or after it go in its box. Pairs puts
Z(2 -> 1) (its in1 the right port) then Z(1 -> 2) (its out1 the left port) on the register's
wire in the register's box, which takes the index, and the CZ's F nodes in a pair box with
copy scalar D^(1/2). LowerMCX (D = 2) puts, per copy k in the register's box (index j), a
Z(1 -> 2) fan on the wire (before the target, after it with ``reverse``), the target X(2 -> 1)
(scalar 2^(1/2) per copy) fed by Ti after a Z AND spider, and one T per single control into
the AND (through X(pi) for value 0); a pair box (index m) holds one T from copy j's fan leg to
copy k's AND, so the AND reads T|0> + prod x |1> = T|prod x>.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Any

import sympy as sp

from benchmarks import ir


class Unsupported(Exception):
    """The circuit cannot be translated at this version; the message says why."""


@dataclass(frozen=True)
class Context:
    """How symbols are written: ``values`` binds d, n, ... to ints (absent: symbolic); a
    symbol in ``literal`` is written as its value (``literal V`` where a dimension or count
    token is expected)."""

    values: Mapping[str, int] = field(default_factory=dict)
    literal: frozenset[str] = frozenset()

    def _subs(self) -> dict[sp.Symbol, int]:
        return {sp.Symbol(k): v for k, v in self.values.items() if k in self.literal}

    def _key(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((k, v) for k, v in self.values.items() if k in self.literal))

    def dim_token(self, dim: Any) -> str:
        """The DSL token of a dimension expression."""
        return _memo(_dim_token, self._key(), dim)

    def value(self, expr: Any) -> sp.Expr:
        """``expr`` with the literal symbols substituted."""
        return _memo(_value, self._key(), expr)

    def concrete(self, expr: Any) -> int | None:
        """The integer value of ``expr`` once every literal symbol is substituted, or None."""
        return _memo(_concrete, self._key(), expr)


def _subs_of(key: tuple[tuple[str, int], ...]) -> dict[sp.Symbol, int]:
    return {sp.Symbol(k): v for k, v in key}


def _dim_token(key: tuple[tuple[str, int], ...], dim: Any) -> str:
    e = sp.sympify(dim)
    if e.is_Integer:
        return f"literal {int(e)}"
    v = e.subs(_subs_of(key))
    if v.is_Integer:
        return f"literal {int(v)}"
    return _expr(v)


def _value(key: tuple[tuple[str, int], ...], expr: Any) -> sp.Expr:
    e = sp.sympify(expr).subs(_subs_of(key))
    return sp.nsimplify(e) if e.is_number else e


def _concrete(key: tuple[tuple[str, int], ...], expr: Any) -> int | None:
    e = sp.sympify(expr).subs(_subs_of(key))
    return int(e) if e.is_Integer else None


@functools.lru_cache(maxsize=65536)
def _cached(fn: Callable[..., Any], key: tuple[tuple[str, int], ...], kind: type, expr: Any) -> Any:
    return fn(key, expr)


def _memo(fn: Callable[..., Any], key: tuple[tuple[str, int], ...], expr: Any) -> Any:
    """``fn(key, expr)``, cached on (fn, key, type of expr, expr) when ``expr`` is hashable."""
    try:
        hash(expr)
    except TypeError:
        return fn(key, expr)
    return _cached(fn, key, type(expr), expr)


def _expr(e: Any) -> str:
    """A sympy expression as DSL text."""
    s = str(sp.sympify(e)).replace("**", "^")
    return s


_LONG_BITS = 160
_SMALL_PRIMES = tuple(sp.primerange(2, 1000))


def _prime_powers(m: int) -> str | None:
    """``m`` as a product of powers of primes below 1000, or None when it has a larger factor."""
    parts: list[str] = []
    for p in _SMALL_PRIMES:
        if m == 1:
            break
        if m % p:
            continue
        e = int(sp.multiplicity(p, m))
        m //= p**e
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    return ("*".join(parts) or "1") if m == 1 else None


def _scalar_text(e: Any) -> str:
    """A scalar as DSL text, a rational coefficient above :data:`_LONG_BITS` bits written as
    prime powers."""
    coeff, rest = sp.sympify(e).as_coeff_Mul()
    if not coeff.is_Rational or max(abs(coeff.p), coeff.q).bit_length() <= _LONG_BITS:
        return _expr(e)
    num, den = _prime_powers(abs(int(coeff.p))), _prime_powers(int(coeff.q))
    if num is None or den is None:
        return _expr(e)
    text = ("-" if coeff.p < 0 else "") + f"({num})"
    if den != "1":
        text += f"/({den})"
    if rest != 1:
        text += f"*({_expr(rest)})"
    return text


def _turns(value: Any) -> Fraction | None:
    """A concrete number of turns reduced into [0, 1), or None when symbolic."""
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        return Fraction(value) % 1
    e = sp.sympify(value)
    if e.is_Float:
        e = sp.nsimplify(e, rational=True)
    if not e.is_Rational:
        return None
    f = Fraction(int(e.p), int(e.q)) % 1
    return f


@dataclass
class _Box:
    count: ir.Count
    nodes: list[str] = field(default_factory=list)
    name: str | None = None
    index: str | None = None
    extra: list[str] = field(default_factory=list)
    per_copy: list[sp.Expr] | None = None


@dataclass(frozen=True)
class _Head:
    """A chain carry not yet used in its copy: the carry and the port it continues from."""

    carry: str
    outer: Any


@dataclass
class _ChainScope:
    key: str
    count: str
    carries: frozenset[str]


class _Builder:
    """Accumulates nodes, wires, boundary and scalar factors."""

    def __init__(self, c: ir.Circuit, ctx: Context) -> None:
        self.c = c
        self.ctx = ctx
        self.decls: list[str] = []
        self.wires: list[str] = []
        self.k = 0
        self.cur: dict[str, Any] = {q: ("input", q) for q in c.wires}
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.closed: set[str] = set()
        self.scalar: list[sp.Expr] = []
        self.boxes: dict[str, _Box] = {}
        self.chain: _ChainScope | None = None
        self.heads: dict[str, str] = {}
        self.chained: set[str] = set()
        self.after: dict[str, str] = {}
        self.before: dict[str, str] = {}
        self.pair_lines: list[str] = []
        for i, g in enumerate(c.gates):
            if not isinstance(g, ir.Chain):
                continue
            regs = {q for q in ir.gate_wires(g) if c.count(q) is not None}
            used = {q for h in c.gates[:i] for q in ir.gate_wires(h)} & regs
            for q in used - set(self.before):
                self.before[q] = f"_pre{i}"
                self.boxes.setdefault(f"_pre{i}", _Box(g.count))

    def dim(self, q: str) -> str:
        return self.ctx.dim_token(self.c.dim(q))

    def dval(self, q: str) -> sp.Expr:
        return self.ctx.value(self.c.dim(q))

    def boxed(self, q: str) -> bool:
        """Whether ``q``'s nodes go in a box: a register, or a carry inside its chain."""
        return self.c.count(q) is not None or (self.chain is not None and q in self.chain.carries)

    def box_of(self, qs: tuple[str, ...]) -> str | None:
        counts = {str(self.c.count(q)) for q in qs if self.c.count(q) is not None}
        if len(counts) > 1:
            raise Unsupported(f"gate across registers of different counts {sorted(counts)}")
        if self.chain is not None:
            if counts - {self.chain.count}:
                raise Unsupported("a gate in a chain on a register of another count")
            return self.chain.key if any(self.boxed(q) for q in qs) else None
        if any(q in self.chained and q not in self.after for q in qs):
            raise Unsupported("a chain register used outside its chain")
        for side, text in ((self.after, "after"), (self.before, "before")):
            keys = {side[q] for q in qs if q in side}
            if keys:
                if len(keys) > 1 or any(self.c.count(q) is not None and q not in side for q in qs):
                    raise Unsupported(f"a gate {text} a chain on registers of another box")
                return keys.pop()
        return counts.pop() if counts else None

    def box_count(self, box: str) -> sp.Expr:
        """The multiplicity of box ``box``."""
        count = self.boxes[box].count if box in self.boxes else box
        return self.ctx.value(sp.sympify(count))

    def node(self, text: str, box: str | None) -> str:
        name = f"g{self.k}"
        self.k += 1
        self.decls.append(f"{name} = {text}")
        if box is not None:
            self.boxes.setdefault(box, _Box(box)).nodes.append(name)
        return name

    def factor(self, value: sp.Expr, box: str | None) -> None:
        if box is None:
            self.scalar.append(value)
        else:
            per_copy = self.boxes[box].per_copy if box in self.boxes else None
            if per_copy is not None:
                per_copy.append(value)
            else:
                self.scalar.append(value ** self.box_count(box))

    def take(self, q: str, port: str) -> None:
        """Wires the current end of ``q`` into ``port``."""
        if q in self.closed:
            raise ValueError(f"wire {q} is used after it ended")
        cur = self.cur[q]
        if isinstance(cur, _Head):
            self.heads[cur.carry] = port
            cur = cur.outer
        if isinstance(cur, tuple):
            self.inputs[cur[1]] = port
        else:
            self.wires.append(f"{cur} -- {port}")

    def gives(self, q: str, port: str) -> None:
        self.cur[q] = port

    def close(self, q: str) -> None:
        self.closed.add(q)
        self.cur[q] = None

    def unary(self, q: str, gen: str, box: str | None) -> None:
        n = self.node(gen, box)
        self.take(q, f"{n}.in0")
        self.gives(q, f"{n}.out0")

    def phase_rule(self, rule: Any, q: str) -> str:
        e = self.ctx.value(sp.sympify(rule))
        e = sp.simplify(e)
        if e == 0:
            return ""
        return f" phase{{k: {_expr(e)}}}"

    def phase_entries(self, entries: Mapping[int, Any]) -> str:
        parts = []
        for i in sorted(entries):
            v = self.ctx.value(entries[i])
            t = _turns(v)
            if t is not None:
                if t:
                    parts.append(f"{i}: {t}")
            else:
                parts.append(f"{i}: {_expr(v)}")
        return f" phase{{{', '.join(parts)}}}" if parts else ""


def _digits(index: int, dims: list[int]) -> tuple[int, ...]:
    out = []
    for dv in reversed(dims):
        out.append(index % dv)
        index //= dv
    return tuple(reversed(out))


def _ravel(x: tuple[int, ...], ds: list[int]) -> int:
    """The joint index of digits ``x`` (first most significant) in dimensions ``ds``."""
    index = 0
    for v, dv in zip(x, ds):
        index = index * dv + v
    return index


def _inside(x: tuple[int, ...], ds: list[int]) -> bool:
    return len(x) == len(ds) and all(0 <= v < dv for v, dv in zip(x, ds))


def _phase_fn(
    b: _Builder,
    qs: tuple[str, ...],
    f: Callable[[tuple[int, ...]], Any],
    support: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """The binder, phase-table, splitter group of :class:`ir.PhaseFn`, in the box of its wires;
    with ``support`` only those inputs are evaluated."""
    box = b.box_of(qs)
    fanned: dict[str, str] = {}
    if box is not None:
        for q in qs:
            if not b.boxed(q):
                z = b.node(f"Z(1 -> 2, {b.dim(q)})", None)
                b.take(q, f"{z}.in0")
                b.gives(q, f"{z}.out0")
                fanned[q] = f"{z}.out1"

    def take(q: str, port: str) -> None:
        if q in fanned:
            b.wires.append(f"{fanned[q]} -- {port}")
        else:
            b.take(q, port)

    def gives(q: str, port: str) -> None:
        if q in fanned:
            end = b.node(f"Z(1 -> 0, {b.dim(q)})", box)
            b.wires.append(f"{port} -- {end}.in0")
        else:
            b.gives(q, port)

    dims = [b.ctx.concrete(b.c.dim(q)) for q in qs]
    if any(v is None for v in dims):
        raise Unsupported("a phase table needs literal concrete dimensions")
    ds = [int(v) for v in dims if v is not None]
    total = math.prod(ds)
    entries: dict[int, Fraction] = {}
    if support is None:
        points: Any = enumerate(product(*[range(v) for v in ds]))
    else:
        points = sorted((_ravel(x, ds), tuple(x)) for x in set(support) if _inside(x, ds))
    for index, x in points:
        t = _turns(f(tuple(x)))
        if t is None:
            raise Unsupported("a phase table entry is not a rational number of turns")
        if t:
            entries[index] = t
    t0 = entries.pop(0, Fraction(0))
    if t0:
        shifted = {i: (entries.get(i, Fraction(0)) - t0) % 1 for i in range(1, total)}
        entries = {i: t for i, t in shifted.items() if t}
        b.factor(sp.exp(2 * sp.pi * sp.I * sp.Rational(t0.numerator, t0.denominator)), box)
    body = ", ".join(f"{i}: {t}" for i, t in entries.items())
    phase = f" phase{{{body}}}" if body else ""
    if len(qs) == 1:
        b.unary(qs[0], f"Z(1 -> 1, literal {total}){phase}", box)
        return
    acc = ds[0]
    prev = qs[0]
    binders: list[tuple[str, int, int]] = []
    for q, dv in zip(qs[1:], ds[1:]):
        name = b.node(f"B(literal {acc}, literal {dv})", box)
        if prev is not None:
            take(prev, f"{name}.in0")
        else:
            b.wires.append(f"{binders[-1][0]}.out0 -- {name}.in0")
        take(q, f"{name}.in1")
        binders.append((name, acc, dv))
        acc *= dv
        prev = None
    z = b.node(f"Z(1 -> 1, literal {total}){phase}", box)
    b.wires.append(f"{binders[-1][0]}.out0 -- {z}.in0")
    port = f"{z}.out0"
    for (_, acc_k, dv), q in zip(reversed(binders), reversed(qs[1:])):
        s = b.node(f"S(literal {acc_k}, literal {dv})", box)
        b.wires.append(f"{port} -- {s}.in0")
        gives(q, f"{s}.out1")
        port = f"{s}.out0"
    gives(qs[0], port)


def _gate(b: _Builder, g: ir.Gate) -> None:
    qs = ir.gate_wires(g)
    if isinstance(g, ir.Chain):
        _chain(b, g)
        return
    box = b.box_of(qs)
    if isinstance(g, ir.Init):
        d = b.dval(g.q)
        n = b.node(
            f"X(0 -> 1, {b.dim(g.q)}){b.phase_rule(sp.sympify(-g.v) * sp.Symbol('k') / d, g.q)}",
            box,
        )
        if not isinstance(b.cur[g.q], tuple):
            raise ValueError(f"Init on wire {g.q} after it started")  # noqa: TRY004
        b.cur[g.q] = f"{n}.out0"
        b.factor(d ** sp.Rational(-1, 2), box)
        return
    if isinstance(g, ir.Post):
        d = b.dval(g.q)
        n = b.node(
            f"X(1 -> 0, {b.dim(g.q)}){b.phase_rule(sp.sympify(g.v) * sp.Symbol('k') / d, g.q)}", box
        )
        b.take(g.q, f"{n}.in0")
        b.close(g.q)
        b.factor(d ** sp.Rational(-1, 2), box)
        return
    if isinstance(g, ir.Discard):
        n = b.node(f"G(1 -> 0, {b.dim(g.q)})", box)
        b.take(g.q, f"{n}.in0")
        b.close(g.q)
        return
    if isinstance(g, ir.Measure):
        z = b.node(f"Z(1 -> 2, {b.dim(g.q)})", box)
        gd = b.node(f"G(1 -> 0, {b.dim(g.q)})", box)
        b.take(g.q, f"{z}.in0")
        b.wires.append(f"{z}.out1 -- {gd}.in0")
        b.gives(g.q, f"{z}.out0")
        return
    if isinstance(g, (ir.X, ir.Z)):
        d = b.dval(g.q)
        sign = -1 if isinstance(g, ir.X) else 1
        rule = b.phase_rule(sign * sp.sympify(g.a) * sp.Symbol("k") / d, g.q)
        b.unary(g.q, f"{type(g).__name__}(1 -> 1, {b.dim(g.q)}){rule}", box)
        return
    if isinstance(g, ir.F):
        for _ in range(g.p % 4):
            b.unary(g.q, f"F({b.dim(g.q)})", box)
        return
    if isinstance(g, (ir.ZPhase, ir.XPhase)):
        gen = "Z" if isinstance(g, ir.ZPhase) else "X"
        phase = b.phase_rule(g.rule, g.q) if g.rule is not None else b.phase_entries(g.entries)
        b.unary(g.q, f"{gen}(1 -> 1, {b.dim(g.q)}){phase}", box)
        return
    if isinstance(g, ir.Swap):
        b.cur[g.a], b.cur[g.b] = b.cur[g.b], b.cur[g.a]
        return
    if isinstance(g, (ir.CX, ir.CZ)):
        if g.s not in (1, -1):
            raise ValueError("CX and CZ take s = +1 or -1")
        if b.c.dim(g.c) != b.c.dim(g.t):
            raise Unsupported("a controlled gate between wires of different dimensions")
        cb = box if b.boxed(g.c) else None
        tb = box if b.boxed(g.t) else None
        mid = cb if cb is not None else tb
        dt = b.dim(g.c)
        zc = b.node(f"Z(1 -> 2, {dt})", cb)
        b.take(g.c, f"{zc}.in0")
        b.gives(g.c, f"{zc}.out0")
        gen = "X" if isinstance(g, ir.CX) else "Z"
        tn = b.node(f"{gen}(2 -> 1, {dt})", tb)
        b.take(g.t, f"{tn}.in0")
        b.gives(g.t, f"{tn}.out0")
        hops = (2 if g.s == -1 else 0) if isinstance(g, ir.CX) else (3 if g.s == -1 else 1)
        port = f"{zc}.out1"
        for _ in range(hops):
            f = b.node(f"F({dt})", mid)
            b.wires.append(f"{port} -- {f}.in0")
            port = f"{f}.out0"
        b.wires.append(f"{port} -- {tn}.in1")
        b.factor(b.dval(g.c) ** sp.Rational(1, 2), mid)
        return
    if isinstance(g, ir.PhaseFn):
        _phase_fn(b, g.qs, g.f, g.support)
        return
    if isinstance(g, (ir.MCZ, ir.MCX)):
        dt = b.ctx.concrete(b.c.dim(g.t))
        if dt is None:
            raise Unsupported("a multi-controlled gate needs literal concrete dimensions")
        values = g.values if g.values is not None else (1,) * len(g.cs)
        a = g.a

        def f(x: tuple[int, ...], values: tuple[int, ...] = values, a: int = a) -> Any:
            return sp.Rational(a * x[-1], dt) if tuple(x[:-1]) == tuple(values) else 0

        if isinstance(g, ir.MCX):
            _gate(b, ir.F(g.t, 1))
        _phase_fn(b, (*g.cs, g.t), f)
        if isinstance(g, ir.MCX):
            _gate(b, ir.F(g.t, 3))
        return
    if isinstance(g, ir.Oracle):
        dy = b.ctx.concrete(b.c.dim(g.y))
        if dy is None:
            raise Unsupported("an oracle needs literal concrete dimensions")
        fn = g.f

        def h(x: tuple[int, ...], fn: Callable[..., int] = fn, dy: int = dy) -> Any:
            return Fraction(int(fn(tuple(x[:-1]))) * x[-1], dy)

        _gate(b, ir.F(g.y, 1))
        _phase_fn(b, (*g.xs, g.y), h)
        _gate(b, ir.F(g.y, 3))
        return
    if isinstance(g, ir.Perm):
        _perm(b, g)
        return
    if isinstance(g, (ir.Unitary, ir.Prepare)):
        raise Unsupported("a dense reference gate has no ZXW translation")
    if isinstance(g, ir.Chain):
        _chain(b, g)
        return
    if isinstance(g, ir.Pairs):
        _pairs(b, g)
        return
    if isinstance(g, ir.Tree):
        _tree(b, g)
        return
    if isinstance(g, ir.LowerMCX):
        _lower_mcx(b, g)
        return
    raise TypeError(f"unknown gate {g!r}")


def _lower_mcx(b: _Builder, g: ir.LowerMCX) -> None:
    """The fan, AND and target nodes of :class:`ir.LowerMCX` (see the module docstring)."""
    q = g.over
    if b.chain is not None or b.c.count(q) is None or q in b.chained:
        raise Unsupported("a lower-copy MCX inside a loop or on a wire that is not a free register")
    if any(b.ctx.concrete(b.c.dim(w)) != 2 for w in (q, *g.controls)):
        raise Unsupported("a lower-copy MCX needs dimension 2")
    if len(g.values) != len(g.controls) or any(v not in (0, 1) for v in g.values):
        raise ValueError("a lower-copy MCX takes one value 0 or 1 per control")
    key = str(b.c.count(q))
    box = b.boxes.setdefault(key, _Box(key))
    if box.name is None:
        box.name = f"C{sum(1 for x in b.boxes.values() if x.name is not None)}"
    if box.index not in (None, "j"):
        raise Unsupported("a lower-copy MCX on a box reading another index")
    box.index = "j"
    dt = "literal 2"

    def fan() -> str:
        z = b.node(f"Z(1 -> 2, {dt})", key)
        b.take(q, f"{z}.in0")
        b.gives(q, f"{z}.out0")
        return f"{z}.out1"

    def target() -> str:
        x = b.node(f"X(2 -> 1, {dt})", key)
        b.take(q, f"{x}.in0")
        b.gives(q, f"{x}.out0")
        return f"{x}.in1"

    if g.reverse:
        flip = target()
        left = fan()
    else:
        left = fan()
        flip = target()
    conj = b.node(f"Z({1 + len(g.controls)} -> 1, {dt})", key)
    inverse = b.node(f"Ti({dt})", key)
    b.wires += [f"{conj}.out0 -- {inverse}.in0", f"{inverse}.out0 -- {flip}"]
    b.factor(sp.sqrt(2), key)
    for i, (c, v) in enumerate(zip(g.controls, g.values, strict=True)):
        z = b.node(f"Z(1 -> 2, {dt})", None)
        b.take(c, f"{z}.in0")
        b.gives(c, f"{z}.out0")
        port = f"{z}.out1"
        if v == 0:
            n = b.node(f"X(1 -> 1, {dt}) phase{{1: 1/2}}", key)
            b.wires.append(f"{port} -- {n}.in0")
            port = f"{n}.out0"
        t = b.node(f"T({dt})", key)
        b.wires += [f"{port} -- {t}.in0", f"{t}.out0 -- {conj}.in{1 + i}"]
    t = b.node(f"T({dt})", None)
    b.wires += [f"{left} -- {t}.in0", f"{t}.out0 -- {conj}.in0"]
    name = f"P{len(b.pair_lines)}"
    b.pair_lines += [
        f"pairs {name} of {box.name} index m: {t}",
        f"left {name}: {left}",
        f"right {name}: {conj}.in0",
    ]


def _tree(b: _Builder, g: ir.Tree) -> None:
    """The tree box of :class:`ir.Tree` (see the module docstring)."""
    if b.chain is not None or b.c.count(g.parent) is not None or g.parent in b.chained:
        raise Unsupported("a tree inside a loop or on a register")
    local = sorted({w for h in g.body for w in ir.gate_wires(h)} - {g.parent})
    if any(w in b.cur for w in local):
        raise Unsupported("a tree body names a circuit wire other than its parent")
    key = f"C{sum(1 for box in b.boxes.values() if box.name is not None)}"
    b.boxes[key] = _Box(str(g.count), name=key, per_copy=[])
    b.chain = _ChainScope(key, str(g.count), frozenset({g.parent, *local}))
    for w in local:
        b.cur[w] = ("local", w)
    b.cur[g.parent] = _Head(g.parent, b.cur[g.parent])
    b.heads.pop(g.parent, None)
    for h in g.body:
        _gate(b, h)
    for w in (g.parent, *g.children):
        if w in b.closed:
            raise Unsupported("a tree child or parent that ends inside the tree")
        if isinstance(b.cur[w], (tuple, _Head)):
            b.unary(w, f"Z(1 -> 1, {b.dim(g.parent)})", key)
    if set(local) - set(g.children) - b.closed:
        raise Unsupported("a tree's local wire neither ends nor is a child")
    children = [str(b.cur[w]) for w in g.children]
    line = f"tree {key}: {b.heads[g.parent]} <- " + ", ".join(children)
    line += "" if g.level is None else f" level {g.level}"
    line += "" if g.position is None else f" position {g.position}"
    b.boxes[key].extra.append(line)
    b.chain = None
    for w in local:
        del b.cur[w]
        b.closed.discard(w)
    b.cur[g.parent] = ", ".join(children)
    b.chained.add(g.parent)


def _chain(b: _Builder, g: ir.Chain) -> None:
    """The chain box of :class:`ir.Chain` (see the module docstring)."""
    if b.chain is not None:
        raise Unsupported("a chain inside a chain")
    count = str(g.count)
    regs = [q for q in ir.gate_wires(g) if b.c.count(q) is not None]
    if any(sp.sympify(b.c.count(q)) != sp.sympify(count) for q in regs):
        raise Unsupported("a chain over a register of another count")
    early = {q for q in regs if q in b.before}
    pre = {b.before.pop(q) for q in early}
    if any(
        (not isinstance(b.cur[q], tuple) and q not in early) or q in b.chained or q in b.closed
        for q in regs
    ):
        raise Unsupported("a chain register used before its chain")
    singles = [carry for carry, _reg in g.carries] + list(g.through)
    if any(b.c.count(q) is not None for q in singles):
        raise Unsupported("a chain carry that is a register")
    key = f"C{sum(1 for box in b.boxes.values() if box.name is not None)}"
    b.boxes[key] = _Box(count, name=key, index=g.index)
    for p in pre:
        b.boxes[key].nodes += b.boxes.pop(p).nodes
    b.chain = _ChainScope(key, count, frozenset(singles))
    for q in singles:
        b.cur[q] = _Head(q, b.cur[q])
        b.heads.pop(q, None)
    for h in g.body:
        _gate(b, h)
    links = []
    if any(q in b.closed for q in g.through):
        raise Unsupported("a chain's through wire that ends inside its chain")
    for carry, reg in g.carries:
        if reg in b.closed:
            raise Unsupported("a chain register that ends inside its chain")
    for q in [x for pair in g.carries for x in pair] + list(g.through):
        if q not in b.closed and isinstance(b.cur[q], (tuple, _Head)):
            b.unary(q, f"Z(1 -> 1, {b.dim(q)})", key)
    for carry, reg in g.carries:
        if carry in b.closed:
            b.closed.discard(carry)
            b.cur[carry] = b.cur[reg]
            b.close(reg)
        else:
            b.cur[carry], b.cur[reg] = b.cur[reg], b.cur[carry]
    for q in singles:
        ends = (b.heads[q], b.cur[q]) if g.reverse else (b.cur[q], b.heads[q])
        links.append(" -> ".join(map(str, ends)))
    if links:
        b.boxes[key].extra.append(f"chain {key}: " + ", ".join(links))
    b.chain = None
    b.chained |= set(regs)
    b.after.update({q: key for q in regs if q not in b.closed})


def _pairs(b: _Builder, g: ir.Pairs) -> None:
    """The anchors and pair box of :class:`ir.Pairs` (see the module docstring)."""
    if b.chain is not None or len(g.body) != 1 or not isinstance(g.body[0], ir.CZ):
        raise Unsupported("a pair loop other than one CZ outside a chain")
    cz = g.body[0]
    if {cz.c, cz.t} != {"left", "right"}:
        raise Unsupported("a pair CZ not between the two copies")
    q = g.over
    if b.c.count(q) is None or q in b.chained:
        raise Unsupported("a pair loop over a wire that is not a free register")
    key = str(b.c.count(q))
    box = b.boxes.setdefault(key, _Box(key))
    if box.name is None:
        box.name = f"C{sum(1 for x in b.boxes.values() if x.name is not None)}"
    if box.index not in (None, g.index):
        raise Unsupported("two pair loops reading different indices")
    box.index = g.index
    dt = b.dim(q)
    tn = b.node(f"Z(2 -> 1, {dt})", key)
    b.take(q, f"{tn}.in0")
    b.gives(q, f"{tn}.out0")
    zc = b.node(f"Z(1 -> 2, {dt})", key)
    b.take(q, f"{zc}.in0")
    b.gives(q, f"{zc}.out0")
    left, right = (f"{zc}.out1", f"{tn}.in1")
    if cz.c == "right":
        left, right = right, left
    hops = 3 if cz.s == -1 else 1
    name = f"P{len(b.pair_lines)}"
    port = f"{zc}.out1"
    fs = []
    for _ in range(hops):
        f = b.node(f"F({dt})", None)
        b.wires.append(f"{port} -- {f}.in0")
        port = f"{f}.out0"
        fs.append(f)
    b.wires.append(f"{port} -- {tn}.in1")
    b.pair_lines += [
        f"pairs {name} of {box.name} index {g.pair_index}: " + ", ".join(fs),
        f"left {name}: {left}",
        f"right {name}: {right}",
        f"copy scalar {name}: {_expr(b.dval(q) ** sp.Rational(1, 2))}",
    ]


def _perm(b: _Builder, g: ir.Perm) -> None:
    """The fresh-leg encoding of :class:`ir.Perm` (see the module docstring)."""
    if b.box_of(g.qs) is not None:
        raise Unsupported("a permutation on a register")
    dims = [b.ctx.concrete(b.c.dim(q)) for q in g.qs]
    if any(v is None for v in dims):
        raise Unsupported("a permutation needs literal concrete dimensions")
    ds = [int(v) for v in dims if v is not None]
    total = math.prod(ds)
    if len(g.qs) > 1:
        x = b.node(f"B(literal {ds[0]}, literal {ds[1]})", None)
        b.take(g.qs[0], f"{x}.in0")
        b.take(g.qs[1], f"{x}.in1")
        port, acc = f"{x}.out0", ds[0] * ds[1]
        for q, dv in zip(g.qs[2:], ds[2:]):
            y = b.node(f"B(literal {acc}, literal {dv})", None)
            b.wires.append(f"{port} -- {y}.in0")
            b.take(q, f"{y}.in1")
            port, acc = f"{y}.out0", acc * dv
    else:
        port = None
    t = b.node(f"Z(0 -> 1, literal {total})", None)
    pair = b.node(f"B(literal {total}, literal {total})", None)
    b.wires.append(f"{t}.out0 -- {pair}.in0")
    if port is None:
        b.take(g.qs[0], f"{pair}.in1")
    else:
        b.wires.append(f"{port} -- {pair}.in1")
    entries = []
    for index in range(total * total):
        tv, xv = divmod(index, total)
        turn = Fraction(-tv * g.p(xv), total) % 1
        if turn:
            entries.append(f"{index}: {turn}")
    phase = f" phase{{{', '.join(entries)}}}" if entries else ""
    z = b.node(f"Z(1 -> 1, literal {total * total}){phase}", None)
    b.wires.append(f"{pair}.out0 -- {z}.in0")
    split = b.node(f"S(literal {total}, literal {total})", None)
    b.wires.append(f"{z}.out0 -- {split}.in0")
    counit = b.node(f"Z(1 -> 0, literal {total})", None)
    b.wires.append(f"{split}.out1 -- {counit}.in0")
    f = b.node(f"F(literal {total})", None)
    b.wires.append(f"{split}.out0 -- {f}.in0")
    b.factor(sp.Integer(total) ** sp.Rational(-1, 2), None)
    port = f"{f}.out0"
    if len(g.qs) == 1:
        b.gives(g.qs[0], port)
        return
    rest = ds[1:]
    for q, dv in zip(reversed(g.qs[1:]), reversed(rest)):
        acc = math.prod(ds[: g.qs.index(q)])
        s = b.node(f"S(literal {acc}, literal {dv})", None)
        b.wires.append(f"{port} -- {s}.in0")
        b.gives(q, f"{s}.out1")
        port = f"{s}.out0"
    b.gives(g.qs[0], port)


@dataclass(frozen=True)
class Translation:
    """The DSL body (no dim or count lines) and its node count."""

    body: str
    nodes: int
    counts: tuple[str, ...]


def translate(c: ir.Circuit, ctx: Context) -> Translation:
    """The naive DSL body of ``c`` under ``ctx`` (see the module docstring)."""
    _check_layout(c)
    b = _Builder(c, ctx)
    for g in c.gates:
        _gate(b, g)
    for q in c.wires:
        cur = b.cur[q]
        if isinstance(cur, tuple):
            box = b.box_of((q,))
            n = b.node(f"Z(1 -> 1, {b.dim(q)})", box)
            b.inputs[cur[1]] = f"{n}.in0"
            b.outputs[q] = f"{n}.out0"
        elif cur is not None:
            b.outputs[q] = cur
    lines = list(b.decls)
    if b.wires:
        lines.append("wire " + ", ".join(b.wires))
    for box in b.boxes.values():
        count = str(box.count)
        bound = (
            ctx.concrete(count)
            if set(map(str, sp.sympify(count).free_symbols)) <= ctx.literal
            else None
        )
        token = (
            f"literal {count}"
            if count.isdigit()
            else (f"literal {bound}" if bound is not None else count)
        )
        head = "" if box.name is None else f" as {box.name}"
        head += "" if box.index is None else f" index {box.index}"
        lines.append(f"repeat {token}{head}: " + ", ".join(box.nodes))
        lines += box.extra
        if box.per_copy:
            factor = sp.powsimp(sp.Mul(*box.per_copy), force=True)
            if factor != 1:
                lines.append(f"copy scalar {box.name}: {_scalar_text(factor)}")
    lines += b.pair_lines
    ins = [b.inputs[q] for q in c.wires if q in b.inputs]
    outs = [b.outputs[q] for q in c.output_order() if q in b.outputs]
    if ins:
        lines.append("inputs " + ", ".join(ins))
    if outs:
        lines.append("outputs " + ", ".join(outs))
    total = sp.powsimp(sp.Mul(*b.scalar, ctx.value(c.scalar)), force=True)
    if total != 1:
        lines.append("scalar " + _scalar_text(total))
    counts = tuple(sorted({str(v) for v in c.registers.values() if not str(v).isdigit()}))
    return Translation("\n".join(lines), len(b.decls), counts)


def _check_layout(c: ir.Circuit) -> None:
    """Registers of one count must be contiguous in the wire order and in the output order."""
    for order in (c.wires, c.output_order()):
        seen: list[str] = []
        for q in order:
            key = str(c.count(q)) if c.count(q) is not None else None
            if key is not None and seen and seen[-1] != key and key in seen:
                raise ValueError(f"registers of count {key} are not contiguous")
            if key is not None and (not seen or seen[-1] != key):
                seen.append(key)
            if key is None:
                seen.append("")
