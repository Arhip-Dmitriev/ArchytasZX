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


"""ZXW diagrams of :mod:`archytaszx.semantics.pathsum` states and density matrices, built
from the path sum's structure rather than from its amplitudes.

One term is translated by :mod:`archytaszx.repl.zxsym`: digit indices are Z spiders in bang
boxes of the register widths, any other index or amplitude factor a gadget, the same graph
for every value of the sizes. Several terms of a concrete state are summed exactly by a
selector: a W node fans ``|1, payload>`` out to one copy per term (see :func:`_sum`), each
term a block of digit spiders (:class:`_Planner`); a sum without that form is folded into one
term by :mod:`~archytaszx.repl.zxsym`. A one-register ket of literal terms is drawn from its
amplitude vector: pick the first s with ``c_s != 0`` and set ``u_k = c_{k+s mod D} / c_s``; a
W node read backwards adds the components of the Z effects on its outputs::

    u = W^T (z_1 (x) ... (x) z_m),  z_j = sum_k e^{i alpha_j[k]} |k>,  u_k = sum_j e^{i alpha_j[k]}

with m = 0 for ``u = |0>``, m = 1 when every ``|u_k| = 1``, and otherwise m even, m >= max
``|u_k|``, and pairs of phases ``arg u_k +- acos(|u_k| / m)``.

A density matrix is drawn from its vectorisation, the ket registers then the bra registers.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.repl import zxsym
from archytaszx.repl.printer import render_graph
from archytaszx.semantics import pathsum as P

MAX_TABLE = 256
"""The largest digit range whose weights are tabulated one value at a time."""


class ZXViewError(Exception):
    """A state with no ZX diagram here."""


class _NoForm(Exception):
    """A term outside the structural translation."""


@dataclass
class View:
    """A diagram with the nodes of each summed term grouped: ``groups`` maps a node to its
    term, ``labels`` names each term, ``outputs`` names the outputs (``out k`` when empty) and
    ``notes`` define dimension names."""

    diagram: Diagram
    groups: dict[NodeId, int] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# -- dense construction ----------------------------------------------------------------------


def amplitudes(s: P.State) -> list[sympy.Expr]:
    """The exact amplitude of every basis state of a concrete ket, in mixed-radix order, by
    enumerating its paths (a reference for tests, not used to draw)."""
    if s.bra:
        raise ZXViewError("a bra")
    if not P.is_concrete(s):
        raise ZXViewError("symbolic widths, dimensions or ranges")
    levels = P.concrete_levels(s)
    size = math.prod(levels)
    out = [sympy.S.Zero] * size
    for t in s.terms:
        ranges = [range(int(b.lo), int(b.hi) + 1) for b in t.bound]
        for point in itertools.product(*ranges):
            env = {b.symbol: sympy.Integer(v) for b, v in zip(t.bound, point, strict=True)}
            key = 0
            for k, n in zip(t.kets, levels, strict=True):
                key = key * n + int(sympy.sympify(k).xreplace(env))
            out[key] += t.amp.xreplace(env)
    return [
        sympy.nsimplify(sympy.simplify(a)) if a.has(sympy.Float) else sympy.simplify(a) for a in out
    ]


def _polar(z: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    """``(|z|, arg(z) / 2 pi)`` of a nonzero ``z``, read factor by factor."""
    mag, turns = sympy.S.One, sympy.S.Zero
    for f in sympy.Mul.make_args(z):
        if f.is_positive:
            mag *= f
        elif f.is_negative:
            mag, turns = mag * -f, turns + sympy.S.Half
        elif f == sympy.I:
            turns += sympy.Rational(1, 4)
        elif isinstance(f, sympy.exp) and (t := f.args[0] / (2 * sympy.pi * sympy.I)).is_real:
            turns += t
        elif isinstance(f, sympy.Pow) and f.base == -1 and f.exp.is_real:
            turns += f.exp / 2
        elif isinstance(f, sympy.Pow) and f.exp.is_integer:
            m, t = _polar(f.base)
            mag, turns = mag * m**f.exp, turns + t * f.exp
        else:
            re, im = f.as_real_imag()
            mag *= sympy.sqrt(sympy.expand(re**2 + im**2))
            turns += sympy.atan2(im, re) / (2 * sympy.pi)
    return mag, turns


def _unit(m: sympy.Expr) -> bool:
    """Whether the magnitude ``m`` is exactly 1."""
    if m == 1:
        return True
    if abs(complex(sympy.N(m, 30)) - 1) > 1e-12:
        return False
    return bool(sympy.simplify(m - 1) == 0)


def _turns(z: sympy.Expr) -> sympy.Expr:
    return _polar(z)[1] if z != 0 else sympy.S.Zero


def _effects(u: Sequence[sympy.Expr]) -> list[dict[int, sympy.Expr]]:
    """Phase maps (index -> turns) of the Z effects whose W^T sum is ``u``; ``u[0] == 1``."""
    return _effects_polar(
        [(sympy.S.Zero, sympy.S.Zero) if z == 0 else _polar(z) for z in u[1:]]
    )


def _effects_polar(rest: Sequence[tuple[sympy.Expr, sympy.Expr]]) -> list[dict[int, sympy.Expr]]:
    """:func:`_effects` of ``u = (1, m_1 e^{2 pi i t_1}, ...)`` given as ``(m_k, t_k)``."""
    if all(m == 0 for m, _ in rest):
        return []
    if all(_unit(m) for m, _ in rest):
        return [{k: t for k, (_, t) in enumerate(rest, 1)}]
    top = max(abs(complex(sympy.N(m, 30))) for m, _ in rest)
    n = max(2, 2 * math.ceil(top / 2 - 1e-12))
    plus, minus = {}, {}
    for k, (m, t) in enumerate(rest, 1):
        beta = sympy.acos(m / n) / (2 * sympy.pi)
        plus[k], minus[k] = t + beta, t - beta
    return [plus, minus] * (n // 2)


def _qudits(s: P.State) -> list[int]:
    return [
        int(d)  # type: ignore[arg-type]
        for d, w in zip(s.dims, s.widths, strict=True)
        for _ in range(int(w))  # type: ignore[arg-type]
    ]


def _vector(dim: Dim, phases: dict[int, sympy.Expr]) -> PhaseVector:
    return PhaseVector(dim, {k: Phase(t) for k, t in phases.items()})


# -- term plans ------------------------------------------------------------------------------


@dataclass
class _Digit:
    """One summed digit: its range, its phase per value in turns and its other weight."""

    radix: int
    turns: list[sympy.Expr]
    mags: list[sympy.Expr]


@dataclass
class _Plan:
    """One term as digits, per-qudit outputs (``("c", v)`` literal, ``("v", i)`` digit i),
    ``omega^{m x y}`` couplings and the constant factor."""

    scalar: sympy.Expr
    digits: list[_Digit] = field(default_factory=list)
    outputs: list[tuple[str, int]] = field(default_factory=list)
    couplings: dict[tuple[int, int], int] = field(default_factory=dict)
    radices: list[int] = field(default_factory=list)


def _digits_of(v: int, d: int, w: int) -> list[int]:
    """The w base-d digits of v, most significant first."""
    out = []
    for _ in range(w):
        v, r = divmod(v, d)
        out.append(r)
    if v:
        raise _NoForm(f"{v} does not fit")
    return out[::-1]


class _Planner:
    """Builds the :class:`_Plan` of one concrete term."""

    def __init__(self, t: P.Term, widths: Sequence[int], dims: Sequence[int]) -> None:
        self.t = t
        self.bounds = {b.symbol: b for b in t.bound}
        self.plan = _Plan(sympy.S.One)
        self.split: dict[sympy.Symbol, list[int]] = {}
        self.base: dict[sympy.Symbol, int] = {}
        for e, w, d in zip(t.kets, widths, dims, strict=True):
            self._ket(sympy.sympify(e), w, d)
        for sym, b in self.bounds.items():
            if sym not in self.split:
                size = int(b.hi) - int(b.lo) + 1
                if size > MAX_TABLE:
                    raise _NoForm("a long index outside the kets")
                self._new(sym, 1, size)
        for f in sympy.Mul.make_args(t.amp):
            self._factor(f)

    def _new(self, sym: sympy.Symbol, w: int, d: int) -> list[int]:
        first = len(self.plan.digits)
        for _ in range(w):
            self.plan.digits.append(_Digit(d, [sympy.S.Zero] * d, [sympy.S.One] * d))
        self.split[sym] = list(range(first, first + w))
        self.base[sym] = d
        return self.split[sym]

    def _ket(self, e: sympy.Expr, w: int, d: int) -> None:
        out = self.plan.outputs
        if e.is_Integer:
            out.extend(("c", v) for v in _digits_of(int(e), d, w))
            return
        if e in self.bounds:
            b = self.bounds[e]
            if b.lo != 0:
                raise _NoForm("an index not starting at 0")
            if e in self.split:
                if self.base[e] != d or len(self.split[e]) != w:
                    raise _NoForm("an index on registers of different shapes")
                digits = self.split[e]
            elif int(b.hi) == d**w - 1:
                digits = self._new(e, w, d)
            elif w == 1 and int(b.hi) < d:
                digits = self._new(e, 1, d)
                for v in range(int(b.hi) + 1, d):
                    self.plan.digits[digits[0]].mags[v] = sympy.S.Zero
            else:
                raise _NoForm("an index over part of a register")
            out.extend(("v", i) for i in digits)
            return
        for k in sorted(e.free_symbols & set(self.bounds), key=str):
            b = self.bounds[k]
            if b.lo == 0 and int(b.hi) < d and P.expand(e - k * (d**w - 1) / (d - 1)) == 0:
                if k in self.split:
                    if self.base[k] != d or len(self.split[k]) != 1:
                        raise _NoForm("an index on registers of different shapes")
                    digit = self.split[k][0]
                else:
                    digit = self._new(k, 1, d)[0]
                    for v in range(int(b.hi) + 1, d):
                        self.plan.digits[digit].mags[v] = sympy.S.Zero
                out.extend(("v", digit) for _ in range(w))
                return
        raise _NoForm(f"the ket value {e}")

    def _value(self, sym: sympy.Symbol) -> dict[tuple[int, ...], sympy.Expr]:
        digits, d = self.split[sym], self.base[sym]
        lo = self.bounds[sym].lo
        poly: dict[tuple[int, ...], sympy.Expr] = {(): lo} if lo != 0 else {}
        for j, i in enumerate(digits):
            poly[(i,)] = sympy.Integer(d) ** (len(digits) - 1 - j)
        return poly

    def _poly(self, e: sympy.Expr) -> dict[tuple[int, ...], sympy.Expr]:
        """``e`` as a polynomial of degree at most 2 in digit variables."""
        if not e.free_symbols & set(self.bounds):
            return {(): e}
        if e in self.bounds:
            return self._value(e)
        if isinstance(e, sympy.Add):
            acc: dict[tuple[int, ...], sympy.Expr] = {}
            for a in e.args:
                acc = _padd(acc, self._poly(a))
            return acc
        if isinstance(e, sympy.Mul):
            acc = {(): sympy.S.One}
            for a in e.args:
                acc = _pmul(acc, self._poly(a))
            return acc
        if isinstance(e, sympy.Pow) and e.exp.is_Integer and 0 <= e.exp <= 2:
            acc = {(): sympy.S.One}
            for _ in range(int(e.exp)):
                acc = _pmul(acc, self._poly(e.base))
            return acc
        if isinstance(e, (P.dsum, P.pop)):
            d = 2 if isinstance(e, P.pop) else e.args[1]
            return {(i,): sympy.S.One for i in self._digit_list(e.args[0], d)}
        if isinstance(e, (P.ddot, P.bitdot)):
            a, b = e.args[0], e.args[1]
            d = 2 if isinstance(e, P.bitdot) else e.args[2]
            if not a.free_symbols & set(self.bounds):
                a, b = b, a
            da = self._digit_list(a, d)
            if not b.free_symbols & set(self.bounds):
                if not b.is_Integer:
                    raise _NoForm(f"the digit product {e}")
                cs = _digits_of(int(b) % int(d) ** len(da), int(d), len(da))
                return {(i,): sympy.Integer(c) for i, c in zip(da, cs, strict=True) if c}
            db = self._digit_list(b, d)
            if len(da) != len(db):
                raise _NoForm(f"the digit product {e}")
            return {tuple(sorted((i, j))): sympy.S.One for i, j in zip(da, db, strict=True)}
        raise _NoForm(f"the exponent part {e}")

    def _digit_list(self, v: sympy.Expr, d: sympy.Expr) -> list[int]:
        if v not in self.split or self.base[v] != d or self.bounds[v].lo != 0:
            raise _NoForm(f"a digit function of {v}")
        return self.split[v]

    def _factor(self, f: sympy.Expr) -> None:
        plan = self.plan
        syms = f.free_symbols & set(self.bounds)
        if not syms:
            plan.scalar *= f
            return
        if isinstance(f, sympy.exp):
            base, expo = sympy.E, f.args[0]
        elif isinstance(f, sympy.Pow) and not f.base.free_symbols & set(self.bounds):
            base, expo = f.base, f.exp
        else:
            base = None
        if base is not None:
            try:
                poly = self._poly(sympy.expand(expo))
            except _NoForm:
                poly = None
            if poly is not None:
                for mono, coef in poly.items():
                    self._monomial(base, mono, coef)
                return
        if len(syms) == 1:
            (sym,) = syms
            if len(self.split[sym]) == 1:
                digit = plan.digits[self.split[sym][0]]
                lo = self.bounds[sym].lo
                for v in range(digit.radix):
                    if digit.mags[v] != 0:
                        digit.mags[v] *= f.xreplace({sym: lo + v})
                return
        raise _NoForm(f"the amplitude factor {f}")

    def _monomial(self, base: sympy.Expr, mono: tuple[int, ...], coef: sympy.Expr) -> None:
        plan = self.plan
        turns = sympy.simplify(coef * sympy.log(base) / (2 * sympy.pi * sympy.I))
        phase = turns.is_real is True
        if not mono:
            plan.scalar *= base**coef
            return
        if len(mono) == 1 or mono[0] == mono[1]:
            digit = plan.digits[mono[0]]
            for v in range(digit.radix):
                x = v if len(mono) == 1 else v * v
                if phase:
                    digit.turns[v] += turns * x
                elif digit.mags[v] != 0:
                    digit.mags[v] *= base ** (coef * x)
            return
        i, j = mono
        r = plan.digits[i].radix
        m = sympy.simplify(turns * r)
        if plan.digits[j].radix != r or not m.is_Integer:
            raise _NoForm("a two-digit phase that is not a power of omega")
        key = (i, j)
        plan.couplings[key] = (plan.couplings.get(key, 0) + int(m)) % r


def _padd(a: dict, b: dict) -> dict:  # type: ignore[type-arg]
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, sympy.S.Zero) + v
    return out


def _pmul(a: dict, b: dict) -> dict:  # type: ignore[type-arg]
    out: dict[tuple[int, ...], sympy.Expr] = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(sorted(ka + kb))
            if len(k) > 2:
                raise _NoForm("a phase of degree above 2")
            out[k] = out.get(k, sympy.S.Zero) + va * vb
    return out


def _plan(t: P.Term, s: P.State) -> _Plan:
    widths = [int(w) for w in s.widths]  # type: ignore[arg-type]
    dims = [int(d) for d in s.dims]  # type: ignore[arg-type]
    plan = _Planner(t, widths, dims).plan
    plan.radices = _qudits(s)
    return plan


# -- building blocks -------------------------------------------------------------------------


def _out(n: NodeId, k: int) -> PortRef:
    return PortRef(n, Direction.OUTPUT, k)


def _in(n: NodeId, k: int) -> PortRef:
    return PortRef(n, Direction.INPUT, k)


def _basis(g: Diagram, v: int, r: int) -> tuple[PortRef, sympy.Expr]:
    """``|v>`` on dimension r as an X spider, and the scalar that normalizes it."""
    dim = Dim.concrete(r)
    phases = {k: sympy.Rational(-k * v, r) for k in range(1, r)} if v else {}
    x = g.add_node(X_SPIDER, [], [dim], _vector(dim, phases))
    return _out(x, 0), 1 / sympy.sqrt(r)


def _weigh(g: Diagram, leg: PortRef, u: Sequence[tuple[sympy.Expr, sympy.Expr]], r: int) -> None:
    """Attaches the weight ``(1, m_1 e^{2 pi i t_1}, ...)``, given as ``(m_k, t_k)``, to the
    spider leg ``leg``."""
    dim = Dim.concrete(r)
    effects = _effects_polar(u)
    w = g.add_node(W_NODE, [dim], [dim] * len(effects))
    g.add_wire(leg, _in(w, 0))
    for j, phases in enumerate(effects):
        z = g.add_node(Z_SPIDER, [dim], [], _vector(dim, phases))
        g.add_wire(_out(w, j), _in(z, 0))


def _bind(g: Diagram, legs: Sequence[PortRef], radices: Sequence[int]) -> PortRef:
    """The legs joined into one wire, most significant first."""
    out, size = legs[0], radices[0]
    for leg, r in zip(legs[1:], radices[1:], strict=True):
        b = g.add_node(DIM_BINDER, [Dim.concrete(size), Dim.concrete(r)], [Dim.concrete(size * r)])
        g.add_wire(out, _in(b, 0))
        g.add_wire(leg, _in(b, 1))
        out, size = _out(b, 0), size * r
    return out


def _split(g: Diagram, out: PortRef, radices: Sequence[int]) -> list[PortRef]:
    """The wire ``out`` cut into wires of ``radices``, most significant first."""
    outputs: list[PortRef] = []
    rest = math.prod(radices)
    for q in radices[:-1]:
        rest //= q
        split = g.add_node(
            DIM_SPLITTER, [Dim.concrete(q * rest)], [Dim.concrete(q), Dim.concrete(rest)]
        )
        g.add_wire(out, _in(split, 0))
        outputs.append(_out(split, 0))
        out = _out(split, 1)
    return [*outputs, out]




@dataclass
class _Block:
    """A built term: its per-qudit output legs, the scalar of its gadgets and the factor its
    digit weights take out at 0."""

    outputs: list[PortRef]
    scalar: sympy.Expr
    active: sympy.Expr


def _block(
    g: Diagram,
    plan: _Plan,
    feeds: Sequence[PortRef] | None = None,
    actives: dict[int, list[PortRef]] | None = None,
) -> _Block:
    """The diagram of ``plan``; with ``feeds`` each digit reads its value from that leg and
    each literal digit on a dimension-r qudit is multiplied by the bit ``actives[r]``."""
    scalar = sympy.S.One
    active = sympy.S.One
    uses: dict[int, list[int]] = {}
    for q, (kind, v) in enumerate(plan.outputs):
        if kind == "v":
            uses.setdefault(v, []).append(q)
    weights: dict[int, list[tuple[sympy.Expr, sympy.Expr]]] = {}
    phases: dict[int, dict[int, sympy.Expr]] = {}
    for i, digit in enumerate(plan.digits):
        u0 = digit.mags[0] * sympy.exp(2 * sympy.pi * sympy.I * digit.turns[0])
        if all(m == 1 for m in digit.mags):
            phases[i] = {k: digit.turns[k] - digit.turns[0] for k in range(1, digit.radix)}
        else:
            if digit.mags[0] == 0:
                raise _NoForm("a digit weight that vanishes at 0")
            ratio = []
            for k in range(1, digit.radix):
                if digit.mags[k] == 0:
                    ratio.append((sympy.S.Zero, sympy.S.Zero))
                    continue
                m, t = _polar(digit.mags[k] / digit.mags[0])
                ratio.append((m, t + digit.turns[k] - digit.turns[0]))
            if all(_unit(m) for m, _ in ratio):
                phases[i] = {k: t for k, (_, t) in enumerate(ratio, 1)}
            else:
                weights[i] = ratio
        active *= u0
    sources = {i: [j for (a, j), m in plan.couplings.items() if a == i for _ in range(m)]
               for i in range(len(plan.digits))}
    targets = {j: sum(m for (_, b), m in plan.couplings.items() if b == j)
               for j in range(len(plan.digits))}
    spiders: list[NodeId] = []
    free: dict[int, int] = {}
    for i, digit in enumerate(plan.digits):
        dim = Dim.concrete(digit.radix)
        n_in = targets[i] + (feeds is not None)
        n_out = len(uses.get(i, [])) + (i in weights) + len(sources[i])
        vector = _vector(dim, phases[i]) if i in phases else None
        spiders.append(g.add_node(Z_SPIDER, [dim] * n_in, [dim] * n_out, vector))
        free[i] = 0
    outputs: list[PortRef | None] = [None] * len(plan.outputs)
    taken_in = {i: 0 for i in range(len(plan.digits))}
    for i, node in enumerate(spiders):
        for q in uses.get(i, []):
            outputs[q] = _out(node, free[i])
            free[i] += 1
        if i in weights:
            _weigh(g, _out(node, free[i]), weights[i], plan.digits[i].radix)
            free[i] += 1
        if feeds is not None:
            g.add_wire(feeds[i], _in(node, taken_in[i]))
            taken_in[i] += 1
    for i, js in sources.items():
        for j in js:
            r = plan.digits[i].radix
            dim = Dim.concrete(r)
            f = g.add_node(FOURIER_BOX, [dim], [dim])
            g.add_wire(_out(spiders[i], free[i]), _in(f, 0))
            free[i] += 1
            g.add_wire(_out(f, 0), _in(spiders[j], taken_in[j]))
            taken_in[j] += 1
            scalar *= sympy.sqrt(r)
    for q, (kind, v) in enumerate(plan.outputs):
        if kind != "c":
            continue
        r = plan.radices[q]
        if actives is None or v == 0:
            outputs[q], norm = _basis(g, v, r)
            scalar *= norm
            continue
        dim = Dim.concrete(r)
        x = g.add_node(X_SPIDER, [dim] * v, [dim])
        copy = g.add_node(Z_SPIDER, [dim], [dim] * v)
        g.add_wire(actives[r].pop(), _in(copy, 0))
        for k in range(v):
            g.add_wire(_out(copy, k), _in(x, k))
        outputs[q] = _out(x, 0)
        scalar /= sympy.Integer(r) ** sympy.Rational(1 - v, 2)
    return _Block([o for o in outputs if o is not None], scalar, active)


# -- whole states ----------------------------------------------------------------------------


def _single(s: P.State, plan: _Plan) -> Diagram:
    g = Diagram()
    block = _block(g, plan)
    g.set_boundary_outputs(block.outputs)
    g.set_scalar(Scalar(sympy.simplify(plan.scalar * block.active * block.scalar)))
    return g


def _slots(plans: Sequence[_Plan]) -> tuple[list[int], list[list[int]]]:
    """Shared payload slot radices and, per plan, the slot of each of its digits."""
    slots: list[int] = []
    owners: list[list[int]] = []
    for plan in plans:
        used: set[int] = set()
        mine: list[int] = []
        for digit in plan.digits:
            k = next((j for j, r in enumerate(slots) if r == digit.radix and j not in used), None)
            if k is None:
                slots.append(digit.radix)
                k = len(slots) - 1
            used.add(k)
            mine.append(k)
        owners.append(mine)
    return slots, owners


def _cup(g: Diagram, dim: Dim) -> tuple[PortRef, PortRef]:
    cup = g.add_node(Z_SPIDER, [], [dim, dim])
    return _out(cup, 0), _out(cup, 1)


def _label(plan: _Plan) -> str:
    """The ket of a term with ``x`` for a summed digit, cut after eight qudits."""
    marks = [str(v) if kind == "c" else "x" for kind, v in plan.outputs]
    more = ",…" if len(marks) > 8 else ""
    return "|" + ",".join(marks[:8]) + more + "⟩"


def _sum(s: P.State, plans: list[_Plan], view: View) -> Diagram:
    g = view.diagram
    radices = plans[0].radices
    bits = sorted(set(radices)) or [2]
    slots, owners = _slots(plans)
    head = bits + slots
    size = math.prod(head)
    scalar = sympy.S.One
    legs: list[PortRef] = []
    for r in bits:
        leg, norm = _basis(g, 1, r)
        legs.append(leg)
        scalar *= norm
    for r in slots:
        legs.append(_out(g.add_node(Z_SPIDER, [], [Dim.concrete(r)]), 0))
    source = _bind(g, legs, head)
    big = Dim.concrete(size)
    fan = g.add_node(W_NODE, [big], [big] * len(plans))
    g.add_wire(source, _in(fan, 0))
    total = math.prod(radices)
    joined = Dim.concrete(total)
    out_a, out_b = _cup(g, joined)
    merge = g.add_node(W_NODE, [joined], [joined] * len(plans))
    g.add_wire(out_a, _in(merge, 0))
    for t, (plan, mine) in enumerate(zip(plans, owners, strict=True)):
        before = set(g.nodes)
        parts = _split(g, _out(fan, t), head) if len(head) > 1 else [_out(fan, t)]
        actives: dict[int, list[PortRef]] = {}
        weight = plan.scalar
        for r, part in zip(bits, parts[: len(bits)], strict=True):
            dim = Dim.concrete(r)
            n = sum(1 for q, (k, v) in enumerate(plan.outputs) if k == "c" and v and radices[q] == r)
            carries = r == bits[0]
            node = g.add_node(Z_SPIDER, [dim], [dim] * (n + carries))
            g.add_wire(part, _in(node, 0))
            actives[r] = [_out(node, k) for k in range(carries, n + carries)]
            if carries:
                carrier = _out(node, 0)
        slot_legs = parts[len(bits) :]
        feeds = [slot_legs[k] for k in mine]
        for k, leg in enumerate(slot_legs):
            if k not in mine:
                g.add_wire(leg, _in(g.add_node(Z_SPIDER, [Dim.concrete(slots[k])], []), 0))
                weight /= slots[k]
        block = _block(g, plan, feeds, actives)
        scalar *= block.scalar
        mag, turn = _polar(weight * block.active)
        r0 = bits[0]
        if _unit(mag):
            vector = _vector(Dim.concrete(r0), {1: turn})
            g.add_wire(carrier, _in(g.add_node(Z_SPIDER, [Dim.concrete(r0)], [], vector), 0))
        else:
            one = (sympy.S.One, sympy.S.Zero)
            _weigh(g, carrier, [(mag, turn)] + [one] * (r0 - 2), r0)
        if len(block.outputs) > 1:
            joined_leg = _bind(g, block.outputs, radices)
        else:
            joined_leg = block.outputs[0]
        cap = g.add_node(Z_SPIDER, [joined, joined], [])
        g.add_wire(joined_leg, _in(cap, 0))
        g.add_wire(_out(merge, t), _in(cap, 1))
        view.groups.update({n: t for n in set(g.nodes) - before})
        view.labels[t] = _label(plan)
    g.set_boundary_outputs(_split(g, out_b, radices) if len(radices) > 1 else [out_b])
    g.set_scalar(Scalar(sympy.simplify(scalar)))
    return g


def diagram(s: P.State | P.Mixed) -> Diagram:
    """A ZXW diagram whose outputs are the qudits of the ket or density matrix ``s``."""
    return view(s).diagram


def _literal_vector(s: P.State) -> list[sympy.Expr] | None:
    """The amplitude vector of a one-register ket of concrete dimension made of literal terms;
    None for any other ket."""
    if len(s.widths) != 1 or not P.is_concrete(s) or any(t.bound for t in s.terms):
        return None
    size = int(s.levels(0))
    out = [sympy.S.Zero] * size
    for t in s.terms:
        out[int(t.kets[0])] += t.amp
    return [sympy.simplify(a) for a in out]


def view(s: P.State | P.Mixed) -> View:
    """:func:`diagram` with the nodes of each summed term grouped and the outputs named:
    ``bra k`` for a bra, ``ket k`` and ``bra k`` for the two sides of a density matrix."""
    names: list[str] = []
    if isinstance(s, P.Mixed) and s.factors is not None:
        a, b = s.factors
        left, right = view(a), view(b)
        out = _juxtapose(left, right)
        out.outputs = [n.replace("out", "ket") for n in _names(left)] + [
            n.replace("out", "bra") for n in _names(right)
        ]
        return out
    if isinstance(s, P.Mixed):
        k = s.count
        widths = s.vec.widths
        per = [_count(w) for w in widths]
        names = [f"ket {r}" for r in range(k) for _ in range(per[r])]
        names += [f"bra {r}" for r in range(k) for _ in range(per[r + k])]
        s = s.vec
    elif s.bra:
        s = P.dagger(s)
        names = [f"bra {r}" for r in range(len(s.widths)) for _ in range(_count(s.widths[r]))]
    s = P.simplify(s)
    out = _view(s)
    if names and len(names) == len(out.diagram.boundary_outputs):
        out.outputs = names
    return out


def _names(v: View) -> list[str]:
    return v.outputs or [f"out {k}" for k in range(len(v.diagram.boundary_outputs))]


def _juxtapose(a: View, b: View) -> View:
    """The two diagrams side by side: nodes, wires, boxes and outputs of ``a`` then ``b``,
    scalars multiplied."""
    g = Diagram()
    outputs: list[PortRef] = []
    for part in (a.diagram, b.diagram):
        ids = {
            nid: g.add_node(
                node.generator_type,
                [port.dim for port in node.inputs],
                [port.dim for port in node.outputs],
                node.phase,
            )
            for nid, node in sorted(part.nodes.items())
        }
        for wire in sorted(part.wires, key=lambda w: w.sort_key()):
            g.add_wire(
                PortRef(ids[wire.a.node_id], wire.a.direction, wire.a.index),
                PortRef(ids[wire.b.node_id], wire.b.direction, wire.b.index),
            )
        for _, box in sorted(part.bang_boxes.items()):
            g.add_bang_box(box.multiplicity, node_scope=frozenset(ids[n] for n in box.node_scope))
        outputs.extend(PortRef(ids[r.node_id], r.direction, r.index) for r in part.boundary_outputs)
    g.set_boundary_outputs(outputs)
    g.set_scalar(Scalar(sympy.simplify(a.diagram.scalar.to_sympy() * b.diagram.scalar.to_sympy())))
    return View(g, notes=a.notes + [n for n in b.notes if n not in a.notes])


def _count(w: sympy.Expr | None) -> int:
    return int(w) if w is not None and sympy.sympify(w).is_Integer else 1


def _view(s: P.State) -> View:
    vector = _literal_vector(s)
    if vector is not None and any(a != 0 for a in vector):
        return View(_vector_diagram(vector, s))
    if len(s.terms) > 1 and P.is_concrete(s):
        try:
            plans = [_plan(t, s) for t in s.terms]
            out = View(Diagram())
            _sum(s, plans, out)
            return out
        except _NoForm:
            pass
    d, notes = zxsym.diagram(s)
    return View(d, notes=notes)


# -- symbolic widths and dimensions ----------------------------------------------------------


def _vector_diagram(c: list[sympy.Expr], s: P.State) -> Diagram:
    """A ZX diagram of one register from its literal amplitude vector."""
    size = len(c)
    nonzero = [k for k, z in enumerate(c) if z != 0]
    shift = nonzero[0]
    u = [sympy.simplify(c[(k + shift) % size] / c[shift]) for k in range(size)]
    dim = Dim.concrete(size)
    g = Diagram()
    effects = _effects(u)
    if not effects:
        out, norm = _basis(g, shift, size)
        g.set_boundary_outputs(_split(g, out, _qudits(s)))
        g.set_scalar(Scalar(c[shift] * norm))
        return g
    if len(effects) == 1:
        out = _out(g.add_node(Z_SPIDER, [], [dim], _vector(dim, effects[0])), 0)
    else:
        cup = g.add_node(Z_SPIDER, [], [dim, dim])
        w = g.add_node(W_NODE, [dim], [dim] * len(effects))
        g.add_wire(PortRef(cup, Direction.OUTPUT, 0), PortRef(w, Direction.INPUT, 0))
        for j, phases in enumerate(effects):
            z = g.add_node(Z_SPIDER, [dim], [], _vector(dim, phases))
            g.add_wire(PortRef(w, Direction.OUTPUT, j), PortRef(z, Direction.INPUT, 0))
        out = PortRef(cup, Direction.OUTPUT, 1)
    if shift:
        vector = PhaseVector(
            dim, {k: Phase(sympy.Rational(-k * shift, size)) for k in range(1, size)}
        )
        x = g.add_node(X_SPIDER, [dim], [dim], vector)
        g.add_wire(out, PortRef(x, Direction.INPUT, 0))
        out = PortRef(x, Direction.OUTPUT, 0)
    g.set_boundary_outputs(_split(g, out, _qudits(s)))
    g.set_scalar(Scalar(c[shift]))
    return g


def listing(s: P.State | P.Mixed) -> str:
    """The graph listing of :func:`view`."""
    return render_graph(view(s).diagram)
