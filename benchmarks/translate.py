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
wire puts the single wire's spider outside the box, where it takes one leg per copy. A wire
never touched gets an identity Z(1 -> 1).
"""

from __future__ import annotations

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

    def dim_token(self, dim: Any) -> str:
        """The DSL token of a dimension expression."""
        e = sp.sympify(dim)
        if e.is_Integer:
            return f"literal {int(e)}"
        v = e.subs(self._subs())
        if v.is_Integer:
            return f"literal {int(v)}"
        return _expr(v)

    def value(self, expr: Any) -> sp.Expr:
        """``expr`` with the literal symbols substituted."""
        e = sp.sympify(expr).subs(self._subs())
        return sp.nsimplify(e) if e.is_number else e

    def concrete(self, expr: Any) -> int | None:
        """The integer value of ``expr`` once every literal symbol is substituted, or None."""
        e = sp.sympify(expr).subs(self._subs())
        return int(e) if e.is_Integer else None


def _expr(e: Any) -> str:
    """A sympy expression as DSL text."""
    s = str(sp.sympify(e)).replace("**", "^")
    return s


def _turns(value: Any) -> Fraction | None:
    """A concrete number of turns reduced into [0, 1), or None when symbolic."""
    e = sp.nsimplify(sp.sympify(value))
    if not e.is_Rational:
        return None
    f = Fraction(int(e.p), int(e.q)) % 1
    return f


@dataclass
class _Box:
    count: ir.Count
    nodes: list[str] = field(default_factory=list)


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

    def dim(self, q: str) -> str:
        return self.ctx.dim_token(self.c.dim(q))

    def dval(self, q: str) -> sp.Expr:
        return self.ctx.value(self.c.dim(q))

    def box_of(self, qs: tuple[str, ...]) -> str | None:
        counts = {str(self.c.count(q)) for q in qs if self.c.count(q) is not None}
        if len(counts) > 1:
            raise Unsupported(f"gate across registers of different counts {sorted(counts)}")
        return counts.pop() if counts else None

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
            count = self.ctx.value(sp.Symbol(box)) if not box.isdigit() else sp.Integer(box)
            self.scalar.append(value**count)

    def take(self, q: str, port: str) -> None:
        """Wires the current end of ``q`` into ``port``."""
        if q in self.closed:
            raise ValueError(f"wire {q} is used after it ended")
        cur = self.cur[q]
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


def _phase_fn(b: _Builder, qs: tuple[str, ...], f: Callable[[tuple[int, ...]], Any]) -> None:
    """The binder, phase-table, splitter group of :class:`ir.PhaseFn`."""
    if b.box_of(qs) is not None:
        raise Unsupported("a phase table on a register")
    dims = [b.ctx.concrete(b.c.dim(q)) for q in qs]
    if any(v is None for v in dims):
        raise Unsupported("a phase table needs literal concrete dimensions")
    ds = [int(v) for v in dims if v is not None]
    total = math.prod(ds)
    entries: dict[int, Fraction] = {}
    for index, x in enumerate(product(*[range(v) for v in ds])):
        t = _turns(f(tuple(x)))
        if t is None:
            raise Unsupported("a phase table entry is not a rational number of turns")
        if t:
            entries[index] = t
    t0 = entries.pop(0, Fraction(0))
    if t0:
        shifted = {i: (entries.get(i, Fraction(0)) - t0) % 1 for i in range(1, total)}
        entries = {i: t for i, t in shifted.items() if t}
        b.factor(sp.exp(2 * sp.pi * sp.I * sp.Rational(t0.numerator, t0.denominator)), None)
    body = ", ".join(f"{i}: {t}" for i, t in entries.items())
    phase = f" phase{{{body}}}" if body else ""
    if len(qs) == 1:
        b.unary(qs[0], f"Z(1 -> 1, literal {total}){phase}", None)
        return
    acc = ds[0]
    prev = qs[0]
    binders: list[tuple[str, int, int]] = []
    for q, dv in zip(qs[1:], ds[1:]):
        name = b.node(f"B(literal {acc}, literal {dv})", None)
        if prev is not None:
            b.take(prev, f"{name}.in0")
        else:
            b.wires.append(f"{binders[-1][0]}.out0 -- {name}.in0")
        b.take(q, f"{name}.in1")
        binders.append((name, acc, dv))
        acc *= dv
        prev = None
    z = b.node(f"Z(1 -> 1, literal {total}){phase}", None)
    b.wires.append(f"{binders[-1][0]}.out0 -- {z}.in0")
    port = f"{z}.out0"
    for (_, acc_k, dv), q in zip(reversed(binders), reversed(qs[1:])):
        s = b.node(f"S(literal {acc_k}, literal {dv})", None)
        b.wires.append(f"{port} -- {s}.in0")
        b.gives(q, f"{s}.out1")
        port = f"{s}.out0"
    b.gives(qs[0], port)


def _gate(b: _Builder, g: ir.Gate) -> None:
    qs = ir.gate_wires(g)
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
        cb = box if b.c.count(g.c) is not None else None
        tb = box if b.c.count(g.t) is not None else None
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
        _phase_fn(b, g.qs, g.f)
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
            return sp.Rational(fn(tuple(x[:-1])) * x[-1], dy)

        _gate(b, ir.F(g.y, 1))
        _phase_fn(b, (*g.xs, g.y), h)
        _gate(b, ir.F(g.y, 3))
        return
    if isinstance(g, ir.Perm):
        _perm(b, g)
        return
    if isinstance(g, (ir.Unitary, ir.Prepare)):
        raise Unsupported("a dense reference gate has no ZXW translation")
    raise TypeError(f"unknown gate {g!r}")


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
    for count, box in b.boxes.items():
        bound = ctx.values.get(count) if count in ctx.literal else None
        token = (
            f"literal {count}"
            if count.isdigit()
            else (f"literal {bound}" if bound is not None else count)
        )
        lines.append(f"repeat {token}: " + ", ".join(box.nodes))
    ins = [b.inputs[q] for q in c.wires if q in b.inputs]
    outs = [b.outputs[q] for q in c.wires if q in b.outputs]
    if ins:
        lines.append("inputs " + ", ".join(ins))
    if outs:
        lines.append("outputs " + ", ".join(outs))
    total = sp.powsimp(sp.Mul(*b.scalar, ctx.value(c.scalar)), force=True)
    if total != 1:
        lines.append("scalar " + _expr(total))
    counts = tuple(sorted({str(v) for v in c.registers.values() if not str(v).isdigit()}))
    return Translation("\n".join(lines), len(b.decls), counts)


def _check_layout(c: ir.Circuit) -> None:
    """Registers of one count must be contiguous in the wire order."""
    seen: list[str] = []
    for q in c.wires:
        key = str(c.count(q)) if c.count(q) is not None else None
        if key is not None and seen and seen[-1] != key and key in seen:
            raise ValueError(f"registers of count {key} are not contiguous")
        if key is not None and (not seen or seen[-1] != key):
            seen.append(key)
        if key is None:
            seen.append("")
