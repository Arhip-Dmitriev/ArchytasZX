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


"""Structural ZXW diagrams of path-sum states of any size, built the same way whatever the
sizes are.

Registers of one shape (w digits of base d) form a family; the digits of a family share bang
boxes of multiplicity w, split where a single digit is singled out (``bit(x, k)``, a literal
or ``bitdot(x, c)``). A summation index over a whole register is a digit index: one Z spider
per digit in each box. Phases acting digit by digit (``dsum``, ``ddot``, a digit's own value)
are phases on those spiders and Fourier boxes between them; a qubit ket that is an xor of
digits is an X spider; a literal digit is an X spider.

Every other index is a wide index: one Z spider on a wire of its whole range. Any other
amplitude factor is a gadget: the wires of the indices it reads, bound into one, end in a Z
effect whose phase rule is the factor's turns (complex turns for a weight that is not a
phase). Any other ket value v on a register of L levels is an output spider u read through
``[u = v] = 1/L sum_t omega^{t (u - v)}``: a Fourier box between u and t and a gadget on t and
the indices of v.

Several terms fold into one with a selector index s: term i's factors read ``s == i``, and a
register's value becomes a function of s and the term indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, cast

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionError
from archytaszx.algebra.phase import INDEX, Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, _mult_symbol
from archytaszx.diagram.generators import (
    DIM_BINDER,
    FOURIER_BOX,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.semantics import pathsum as P

Expr: TypeAlias = sympy.Expr


class Unsupported(Exception):
    """A term this translation does not cover."""


# -- abstract nodes --------------------------------------------------------------------------


@dataclass
class _Node:
    """A node before it is placed: generator, wire dimension, phase and the box it sits in."""

    kind: GeneratorType
    dim: Dim
    phase: PhaseVector | None = None
    box: int | None = None
    ins: list[Dim] = field(default_factory=list)
    outs: list[Dim] = field(default_factory=list)


class _Builder:
    """Collects nodes, wires and outputs, then places them in a :class:`Diagram`."""

    def __init__(self) -> None:
        self.nodes: list[_Node] = []
        self.wires: list[tuple[int, int, int, int]] = []
        self.outputs: list[tuple[int, int]] = []
        self.boxes: list[Expr] = []
        self.scalar: Expr = sympy.S.One
        self.notes: list[str] = []
        self._dims: dict[Expr, Dim] = {}

    def box(self, mult: Expr) -> int | None:
        """A new box of multiplicity ``mult``; None when it is 1."""
        if P.same(mult, 1):
            return None
        self.boxes.append(mult)
        return len(self.boxes) - 1

    def node(
        self,
        kind: GeneratorType,
        dim: Dim,
        phase: PhaseVector | None = None,
        box: int | None = None,
    ) -> int:
        self.nodes.append(_Node(kind, dim, phase, box))
        return len(self.nodes) - 1

    def out_leg(self, n: int, dim: Dim | None = None) -> int:
        node = self.nodes[n]
        node.outs.append(dim or node.dim)
        return len(node.outs) - 1

    def in_leg(self, n: int, dim: Dim | None = None) -> int:
        node = self.nodes[n]
        node.ins.append(dim or node.dim)
        return len(node.ins) - 1

    def wire(self, a: int, b: int) -> None:
        """A wire from a new output leg of ``a`` to a new input leg of ``b``."""
        self.wires.append((a, self.out_leg(a), b, self.in_leg(b)))

    def wire_legs(self, a: int, ka: int, b: int, kb: int) -> None:
        self.wires.append((a, ka, b, kb))

    def output(self, n: int, dim: Dim | None = None) -> None:
        self.outputs.append((n, self.out_leg(n, dim)))

    def dim(self, e: Expr) -> Dim:
        """The Dim of a size expression; a fresh named dimension outside the Dim grammar."""
        e = sympy.sympify(e)
        if e in self._dims:
            return self._dims[e]
        exps = {s for p in e.atoms(sympy.Pow) for s in p.exp.free_symbols}
        reps = {
            s: _mult_symbol(s.name)
            if s in exps
            else sympy.Symbol(s.name, positive=True, integer=True)
            for s in e.free_symbols
        }
        try:
            d = Dim.from_sympy(e.xreplace(reps))
        except (DimensionError, TypeError, ValueError):
            name = f"D{len([x for x in self._dims.values() if not x.is_concrete]) + 1}"
            d = Dim.symbol(name)
            self.notes.append(f"{name} = {sympy.sstr(e).replace('**', '^')}")
        self._dims[e] = d
        return d

    def build(self) -> Diagram:
        g = Diagram()
        names = {s.name for m in self.boxes for s in sympy.sympify(m).free_symbols}
        dims: set[str] = set()
        for node in self.nodes:
            for d in (node.dim, *node.ins, *node.outs):
                for s in d.to_sympy().free_symbols:
                    (dims if s.is_positive else names).add(s.name)

        def rename(e: Expr) -> Expr:
            e = sympy.sympify(e)
            reps = {}
            for s in e.free_symbols:
                if s.name in names:
                    reps[s] = _mult_symbol(s.name)
                elif s.name in dims:
                    reps[s] = sympy.Symbol(s.name, positive=True, integer=True)
            return e.xreplace(reps)

        ids: list[NodeId | None] = []
        for node in self.nodes:
            if node.phase is not None and names:
                pv = node.phase
                rule = pv.rule
                node.phase = PhaseVector(
                    pv.dim,
                    {k: Phase(rename(x.to_sympy_turns())) for k, x in pv.entries().items()},
                    Phase(rename(rule.to_sympy_turns())) if rule is not None else None,
                    expand=False,
                )
            if not node.ins and not node.outs and node.phase is None and node.kind is Z_SPIDER:
                mult = self.boxes[node.box] if node.box is not None else sympy.S.One
                self.scalar *= node.dim.to_sympy() ** mult
                ids.append(None)
                continue
            ids.append(g.add_node(node.kind, node.ins, node.outs, node.phase))
        for a, ka, b, kb in self.wires:
            g.add_wire(
                PortRef(ids[a], Direction.OUTPUT, ka),  # type: ignore[arg-type]
                PortRef(ids[b], Direction.INPUT, kb),  # type: ignore[arg-type]
            )
        g.set_boundary_outputs(
            [PortRef(ids[n], Direction.OUTPUT, k) for n, k in self.outputs]  # type: ignore[arg-type]
        )
        for k, mult in enumerate(self.boxes):
            scope = frozenset(
                nid
                for nid, node in zip(ids, self.nodes, strict=True)
                if node.box == k and nid is not None
            )
            if scope:
                g.add_bang_box(_mult(mult), node_scope=scope)
        g.set_scalar(Scalar(rename(P.tidy(sympy.simplify(self.scalar)))))
        return g


def _mult(e: Expr) -> Mult:
    e = sympy.sympify(e)
    if e.is_Integer:
        return Mult.concrete(int(e))
    return Mult._from_expr(e.xreplace({s: _mult_symbol(s.name) for s in e.free_symbols}))


def _vector(dim: Dim, turns: Expr, expand: bool = True) -> tuple[PhaseVector | None, Expr]:
    """The phase vector of ``k -> turns(INDEX = k)`` gauge-fixed at 0, and the factor
    ``e^{2 pi i turns(0)}`` it leaves."""
    turns = sympy.sympify(turns)
    zero = turns.xreplace({INDEX: 0})
    rule = turns - zero
    factor = sympy.exp(2 * sympy.pi * sympy.I * zero) if zero != 0 else sympy.S.One
    if rule == 0:
        return None, factor
    if dim.is_concrete and expand and dim.to_int() <= 64:
        entries = {}
        for k in range(1, dim.to_int()):
            value = (
                sympy.nsimplify(rule.xreplace({INDEX: k}))
                if rule.has(sympy.Float)
                else (rule.xreplace({INDEX: k}))
            )
            entries[k] = Phase(value)
        return PhaseVector(dim, entries), factor
    return PhaseVector(dim, {}, Phase(rule), expand=False), factor


# -- term analysis ---------------------------------------------------------------------------


@dataclass
class _Family:
    """Registers of one shape: width, base, and the digit positions (from the least
    significant) that stand alone."""

    width: Expr
    radix: Expr
    singles: set[int] = field(default_factory=set)
    segments: list[Expr] = field(default_factory=list)
    boxes: list[int | None] = field(default_factory=list)
    high: bool = False

    def layout(self) -> None:
        """Segments, most significant first: a box over the high digits, then one segment per
        digit at or below the highest single."""
        if not self.singles:
            self.segments = [self.width]
            self.high = not P.same(self.width, 1)
            return
        top = max(self.singles)
        if self.width.is_Integer:
            top = min(top, int(self.width) - 1)
        high = sympy.expand(self.width - (top + 1))
        self.high = not (high.is_Integer and int(high) <= 0)
        self.segments = ([high] if self.high else []) + [sympy.S.One] * (top + 1)

    def position(self, seg: int) -> int | None:
        """The digit position (from the least significant) of a one-digit segment."""
        if self.high and seg == 0:
            return None
        return len(self.segments) - 1 - seg


@dataclass
class _Index:
    """A summation index: its range, its family when it is a digit index, and its nodes."""

    symbol: sympy.Symbol
    lo: Expr
    size: Expr
    family: _Family | None = None
    nodes: list[int] = field(default_factory=list)
    wide: int | None = None
    linear: Expr = sympy.S.Zero


def _repeated(v: Expr, w: Expr, d: Expr) -> Expr | None:
    """c when ``v = c (d^w - 1)/(d - 1)``, a repeated digit; None otherwise."""
    if v == 0:
        return sympy.S.Zero
    if P.same(w, 1):
        return v
    c = sympy.simplify(v * (d - 1) / (d**w - 1))
    if c.free_symbols - v.free_symbols - w.free_symbols - d.free_symbols:
        return None
    if c.is_integer is False or c.has(w):
        return None
    return c


class _Term:
    """The diagram of one path-sum term on the registers of ``s``."""

    def __init__(self, t: P.Term, s: P.State, b: _Builder) -> None:
        self.b = b
        self.s = s
        self.amp = P.inline(t.amp)
        self.kets = [sympy.sympify(P.inline(k)) for k in t.kets]
        self.indices = {
            x.symbol: _Index(x.symbol, x.lo, sympy.expand(x.hi - x.lo + 1)) for x in t.bound
        }
        self.amp = P._signs(self.amp, list(self.indices))
        self.families: list[_Family] = []
        self.reg_family: list[_Family] = []
        for w, d in zip(s.widths, s.dims, strict=True):
            if w is None or d is None:
                raise Unsupported("an unresolved register")
            fam = next(
                (f for f in self.families if P.same(f.width, w) and P.same(f.radix, d)), None
            )
            if fam is None:
                fam = _Family(sympy.sympify(w), sympy.sympify(d))
                self.families.append(fam)
            self.reg_family.append(fam)
        self.wide: set[sympy.Symbol] = set()
        self._classify()

    # classification -------------------------------------------------------------------------

    def _digit_family(self, x: sympy.Symbol) -> _Family | None:
        if x in self.wide:
            return None
        idx = self.indices[x]
        if idx.lo != 0:
            return None
        for r, k in enumerate(self.kets):
            fam = self.reg_family[r]
            if k == x and P.same(idx.size, fam.radix**fam.width):
                return fam
        for fam in self.families:
            if P.same(idx.size, fam.radix**fam.width):
                return fam
        return None

    def _classify(self) -> None:
        """Marks wide every index a non-digit use needs whole, until nothing changes."""
        while True:
            before = set(self.wide)
            for x in self.indices:
                self.indices[x].family = self._digit_family(x)
            for f in self.families:
                f.singles = set()
            for r, k in enumerate(self.kets):
                if not self._ket_plan(r, k, dry=True):
                    self.wide |= k.free_symbols & set(self.indices)
            for f in sympy.Mul.make_args(self.amp):
                self._factor(f, dry=True)
            if self.wide == before:
                return

    # kets -------------------------------------------------------------------------------

    def _ket_plan(self, r: int, k: Expr, dry: bool) -> bool:
        """Whether register r's value has a digit form; builds it unless ``dry``."""
        fam = self.reg_family[r]
        if k in self.indices:
            idx = self.indices[k]
            if idx.family is fam:
                if not dry:
                    for seg, node in enumerate(idx.nodes):
                        self.b.output(node)
                return True
            if k in self.wide and P.same(idx.size, fam.radix**fam.width) and idx.lo == 0:
                if not dry:
                    self.b.output(self._wide(k))
                return True
            return False
        if not k.free_symbols & set(self.indices):
            if k.is_Integer and fam.radix.is_Integer:
                if dry:
                    v, pos = int(k), 0
                    while v:
                        v //= int(fam.radix)
                        fam.singles.add(pos)
                        pos += 1
                else:
                    self._literal(fam, k)
                return True
            rep = _repeated(k, fam.width, fam.radix)
            if rep is not None:
                if not dry:
                    self._repeated_literal(fam, rep)
                return True
            if P.same(fam.width, 1):
                if not dry:
                    self._repeated_literal(fam, k)
                return True
            return False
        if isinstance(k, P.xor) and P.same(fam.radix, 2):
            return self._xor(fam, k, dry)
        rep = _repeated(k, fam.width, fam.radix)
        if rep is not None and rep in self.indices:
            idx = self.indices[rep]
            if idx.lo == 0 and P.same(idx.size, fam.radix):
                if not dry:
                    self._repeated_index(idx, fam)
                return True
        return False

    def _repeated_index(self, idx: _Index, fam: _Family) -> None:
        """Every digit of a register equal to the one-digit index ``idx``: a boxed Z spider
        per digit wired to the index's spider."""
        if idx.family is not None and P.same(idx.family.width, 1):
            src = idx.nodes[0]
        else:
            src = self._wide(idx.symbol)
        for seg, mult in enumerate(fam.segments):
            copy = self.b.node(Z_SPIDER, self.b.dim(fam.radix), None, fam.boxes[seg])
            self.b.wire(src, copy)
            self.b.output(copy)

    def _literal(self, fam: _Family, v: Expr) -> None:
        d = int(fam.radix)
        digits = {}
        value, pos = int(v), 0
        while value:
            value, digits[pos] = divmod(value, d)[0], value % d
            pos += 1
        for seg, mult in enumerate(fam.segments):
            p = fam.position(seg)
            self._x_literal(fam, seg, sympy.Integer(digits.get(p, 0) if p is not None else 0))

    def _repeated_literal(self, fam: _Family, c: Expr) -> None:
        for seg in range(len(fam.segments)):
            self._x_literal(fam, seg, c)

    def _x_literal(self, fam: _Family, seg: int, v: Expr) -> None:
        d = self.b.dim(fam.radix)
        phase, factor = _vector(d, -INDEX * v / fam.radix)
        node = self.b.node(X_SPIDER, d, phase, fam.boxes[seg])
        self.b.output(node)
        self.b.scalar *= (factor / sympy.sqrt(fam.radix)) ** fam.segments[seg]

    def _xor_leaves(self, fam: _Family, k: Expr) -> list[tuple[str, object]] | None:
        """Per leaf of the qubit xor ``k``: ``("index", x)``, ``("bit", (x, pos))`` or
        ``("const", c)``; None when a leaf has no digit form."""
        out: list[tuple[str, object]] = []
        for leaf in P._xor_leaves(k):
            if leaf in self.indices and self.indices[leaf].family is fam:
                out.append(("index", leaf))
            elif (
                isinstance(leaf, P.bit) and leaf.args[0] in self.indices and leaf.args[1].is_Integer
            ):
                x = leaf.args[0]
                idx = self.indices[x]
                if (
                    idx.family is None
                    or not P.same(idx.family.radix, 2)
                    or not P.same(fam.width, 1)
                ):
                    return None
                out.append(("bit", (x, int(leaf.args[1]))))
            elif leaf.is_Integer and P.same(fam.width, 1) and int(leaf) in (0, 1):
                out.append(("const", int(leaf)))
            elif leaf.is_Integer and int(leaf) == 0:
                continue
            else:
                return None
        return out

    def _xor(self, fam: _Family, k: Expr, dry: bool) -> bool:
        leaves = self._xor_leaves(fam, k)
        if leaves is None:
            return False
        if dry:
            for kind, val in leaves:
                if kind == "bit":
                    x, pos = cast(tuple[sympy.Symbol, int], val)
                    fam_x = self.indices[x].family
                    assert fam_x is not None
                    fam_x.singles.add(pos)
            return True
        flips = sum(int(v) for kind, v in leaves if kind == "const") % 2  # type: ignore[call-overload]
        for seg in range(len(fam.segments)):
            d = self.b.dim(2)
            phase, factor = _vector(d, INDEX * sympy.Rational(flips, 2))
            node = self.b.node(X_SPIDER, d, phase, fam.boxes[seg])
            ins = 0
            for kind, val in leaves:
                if kind == "index":
                    self.b.wire(self.indices[cast(sympy.Symbol, val)].nodes[seg], node)
                    ins += 1
                elif kind == "bit":
                    x, pos = cast(tuple[sympy.Symbol, int], val)
                    src = self._digit_node(x, pos)
                    self.b.wire(src, node)
                    ins += 1
            self.b.output(node)
            self.b.scalar *= (factor * sympy.sqrt(2) ** (ins - 1)) ** fam.segments[seg]
        return True

    def _digit_node(self, x: sympy.Symbol, pos: int) -> int:
        idx = self.indices[x]
        fam = idx.family
        assert fam is not None
        for seg in range(len(fam.segments)):
            if fam.position(seg) == pos:
                return idx.nodes[seg]
        raise Unsupported(f"digit {pos} of {x}")

    # amplitude --------------------------------------------------------------------------

    def _factor(self, f: Expr, dry: bool) -> None:
        syms = f.free_symbols & set(self.indices)
        if not syms:
            if not dry:
                self.b.scalar *= f
            return
        turns = _turns(f)
        if turns is not None:
            rest: Expr = sympy.S.Zero
            for mono in sympy.Add.make_args(sympy.expand(turns)):
                if not mono.free_symbols & set(self.indices):
                    if not dry:
                        self.b.scalar *= sympy.exp(2 * sympy.pi * sympy.I * mono)
                elif not self._monomial(mono, dry):
                    rest += mono
            if rest != 0:
                self._gadget(rest, dry, phase=True)
            return
        if isinstance(f, sympy.KroneckerDelta):
            self._delta(f, dry)
            return
        if len(syms) == 1:
            (x,) = syms
            idx = self.indices[x]
            fam = idx.family
            if fam is not None and P.same(fam.width, 1) and fam.radix.is_Integer:
                if not dry:
                    self._weigh(idx, f)
                return
        self._gadget(f, dry, phase=False)

    def _monomial(self, mono: Expr, dry: bool) -> bool:
        """Places one additive part of a phase in turns on digit spiders; False when it has
        no digit form."""
        coeff = sympy.Mul(
            *[a for a in sympy.Mul.make_args(mono) if not a.free_symbols & set(self.indices)]
        )
        body = sympy.simplify(mono / coeff)
        if body in self.indices:
            idx = self.indices[body]
            if body in self.wide:
                if not dry:
                    self._wide_phase(idx, coeff)
                return True
            fam = idx.family
            if fam is not None and P.same(fam.width, 1):
                if not dry:
                    self._digit_phase(idx, 0, coeff * INDEX)
                return True
            return False
        if isinstance(body, (P.pop, P.dsum)):
            x = body.args[0]
            d = P.TWO if isinstance(body, P.pop) else body.args[1]
            found = self.indices.get(x)
            if found is None or found.family is None or not P.same(found.family.radix, d):
                return False
            if not dry:
                for seg in range(len(found.nodes)):
                    self._digit_phase(found, seg, coeff * INDEX)
            return True
        if isinstance(body, (P.bitdot, P.ddot)):
            a, c = body.args[0], body.args[1]
            d = P.TWO if isinstance(body, P.bitdot) else body.args[2]
            if a not in self.indices:
                a, c = c, a
            ia = self.indices.get(a)
            if ia is None or ia.family is None or not P.same(ia.family.radix, d):
                return False
            if c.is_Integer and d.is_Integer:
                pos, value, places = 0, int(c), []
                while value:
                    value, digit = divmod(value, int(d))
                    if digit:
                        places.append((pos, digit))
                    pos += 1
                if dry:
                    ia.family.singles.update(p for p, _ in places)
                    return True
                for p, digit in places:
                    seg = next(j for j in range(len(ia.nodes)) if ia.family.position(j) == p)
                    self._digit_phase(ia, seg, coeff * digit * INDEX)
                return True
            if not c.free_symbols & set(self.indices):
                rep = _repeated(c, ia.family.width, d)
                if rep is None or rep.free_symbols & set(self.indices):
                    return False
                if not dry:
                    for seg in range(len(ia.nodes)):
                        self._digit_phase(ia, seg, coeff * rep * INDEX)
                return True
            partner = self.indices.get(c)
            if partner is None or partner.family is not ia.family:
                return False
            m = sympy.simplify(coeff * d)
            if not m.is_integer:
                return False
            if not dry:
                for seg in range(len(ia.nodes)):
                    self._couple(ia, partner, seg, m)
            return True
        if isinstance(body, P.bit) and body.args[0] in self.indices and body.args[1].is_Integer:
            idx = self.indices[body.args[0]]
            if idx.family is None or not P.same(idx.family.radix, 2):
                return False
            if dry:
                idx.family.singles.add(int(body.args[1]))
                return True
            self._digit_phase_at(idx, int(body.args[1]), coeff * INDEX)
            return True
        if body.is_Pow and body.base in self.indices and body.exp == 2:
            idx = self.indices[body.base]
            if (
                idx.family is not None
                and P.same(idx.family.width, 1)
                and body.base not in self.wide
            ):
                if not dry:
                    self._digit_phase(idx, 0, coeff * INDEX**2)
                return True
            return False
        if body.is_Mul and len(body.args) == 2 and all(a in self.indices for a in body.args):
            ia, ic = (self.indices[a] for a in body.args)
            fam = ia.family
            if fam is not None and ic.family is fam and P.same(fam.width, 1):
                m = sympy.simplify(coeff * fam.radix)
                if m.is_integer:
                    if not dry:
                        self._couple(ia, ic, 0, m)
                    return True
            return False
        return False

    def _digit_phase(self, idx: _Index, seg: int, turns: Expr) -> None:
        node = self.b.nodes[idx.nodes[seg]]
        phase, factor = _vector(node.dim, turns)
        if phase is not None:
            node.phase = phase if node.phase is None else _add(node.phase, phase)
        assert idx.family is not None
        self.b.scalar *= factor ** idx.family.segments[seg]

    def _digit_phase_at(self, idx: _Index, pos: int, turns: Expr) -> None:
        assert idx.family is not None
        seg = next(j for j in range(len(idx.nodes)) if idx.family.position(j) == pos)
        self._digit_phase(idx, seg, turns)

    def _couple(self, a: _Index, c: _Index, seg: int, m: Expr) -> None:
        """``omega^{m x y}`` between the digit spiders of a and c in segment ``seg``."""
        assert a.family is not None
        d = a.family.radix
        count = int(m) % int(d) if d.is_Integer and m.is_Integer else None
        if count is None:
            raise Unsupported("a digit coupling of symbolic strength")
        box = a.family.boxes[seg]
        for _ in range(count):
            f = self.b.node(FOURIER_BOX, self.b.dim(d), None, box)
            self.b.wire(a.nodes[seg], f)
            self.b.wire(f, c.nodes[seg])
            self.b.scalar *= sympy.sqrt(d) ** a.family.segments[seg]

    def _weigh(self, idx: _Index, f: Expr) -> None:
        """A non-phase weight on a one-digit index of concrete base: a W node over Z
        effects whose sum is the weight divided by its value at 0."""
        from archytaszx.repl import zxview

        d = int(idx.family.radix)  # type: ignore[union-attr]
        values = [sympy.simplify(f.xreplace({idx.symbol: idx.lo + v})) for v in range(d)]
        if values[0] == 0:
            raise Unsupported("a digit weight that vanishes at 0")
        self.b.scalar *= values[0]
        ratio = []
        for v in values[1:]:
            ratio.append((sympy.S.Zero, sympy.S.Zero) if v == 0 else zxview._polar(v / values[0]))
        effects = zxview._effects_polar(ratio)
        dim = self.b.dim(d)
        if not effects:
            return
        if len(effects) == 1:
            phase = PhaseVector(dim, {k: Phase(t) for k, t in effects[0].items()})
            z = self.b.node(Z_SPIDER, dim, phase)
            self.b.wire(idx.nodes[0], z)
            return
        w = self.b.node(W_NODE, dim)
        self.b.wire(idx.nodes[0], w)
        for phases in effects:
            z = self.b.node(
                Z_SPIDER, dim, PhaseVector(dim, {k: Phase(t) for k, t in phases.items()})
            )
            self.b.wire(w, z)

    def _delta(self, f: sympy.KroneckerDelta, dry: bool) -> None:
        """``[a = b]`` as ``1/L sum_t omega^{t (a - b)}`` with L above both sides."""
        a, c = f.args
        top = self._bound(a - c)
        if top is None:
            self._gadget(f, dry, phase=False)
            return
        self._constraint(a - c, top, None, dry)

    def _bound(self, e: Expr) -> Expr | None:
        """A modulus L with ``|e| < L`` over the index ranges, when they bound it."""
        total = sympy.S.Zero
        for term in sympy.Add.make_args(sympy.expand(e)):
            syms = term.free_symbols & set(self.indices)
            if not syms:
                if not term.is_Integer:
                    return None
                total += abs(term)
                continue
            if len(syms) != 1:
                return None
            (x,) = syms
            c = sympy.simplify(term / x)
            if not c.is_Integer:
                return None
            idx = self.indices[x]
            total += abs(c) * (idx.lo + idx.size - 1)
        return sympy.expand(total + 1)

    def _constraint(self, e: Expr, size: Expr, out: int | None, dry: bool) -> None:
        """``[e (+ u) = 0]`` mod ``size`` through a Fourier index t; with ``out`` the output
        spider u is wired to t by a Fourier box."""
        if dry:
            self.wide |= e.free_symbols & set(self.indices)
            return
        dim = self.b.dim(size)
        t = self.b.node(Z_SPIDER, dim)
        if out is not None:
            fb = self.b.node(FOURIER_BOX, dim)
            self.b.wire(t, fb)
            self.b.wire(fb, out)
            self.b.scalar *= 1 / sympy.sqrt(size)
        else:
            self.b.scalar *= 1 / size
        tsym = sympy.Dummy("t", integer=True)
        self.indices[tsym] = _Index(tsym, sympy.S.Zero, size, None, [], t)
        self._gadget_build(-tsym * e / size, phase=True)

    def _gadget(self, f: Expr, dry: bool, phase: bool) -> None:
        if dry:
            self.wide |= f.free_symbols & set(self.indices)
            return
        self._gadget_build(f, phase)

    def _gadget_build(self, f: Expr, phase: bool) -> None:
        """A Z effect on the bound wires of the indices of ``f`` with the phase rule ``f``
        (turns when ``phase``, else the weight's complex turns)."""
        syms = sorted(f.free_symbols & set(self.indices), key=sympy.default_sort_key)
        idxs = [self.indices[x] for x in syms]
        sizes = [i.size for i in idxs]
        legs = [self._wide(x) for x in syms]
        values: dict[sympy.Basic, Expr] = {}
        below = sympy.S.One
        for i, x in zip(reversed(idxs), reversed(syms), strict=True):
            part = INDEX if len(idxs) == 1 else sympy.Mod(sympy.floor(INDEX / below), i.size)
            values[x] = i.lo + part
            below = below * i.size
        g = f.xreplace(values)
        turns = g if phase else sympy.log(g) / (2 * sympy.pi * sympy.I)
        total = sympy.Mul(*sizes)
        dim = self.b.dim(total)
        source = legs[0]
        size = sizes[0]
        for leg, s in zip(legs[1:], sizes[1:], strict=True):
            bnode = self.b.node(DIM_BINDER, self.b.dim(size * s))
            self.b.wire_legs(
                source, self.b.out_leg(source), bnode, self.b.in_leg(bnode, self.b.dim(size))
            )
            self.b.wire_legs(leg, self.b.out_leg(leg), bnode, self.b.in_leg(bnode, self.b.dim(s)))
            source, size = bnode, size * s
        pv, factor = _vector(dim, turns, expand=False)
        effect = self.b.node(Z_SPIDER, dim, pv)
        self.b.wire(source, effect)
        self.b.scalar *= factor

    def _wide_phase(self, idx: _Index, coeff: Expr) -> None:
        """Adds the phase ``coeff * value`` in turns to a wide index."""
        if idx.wide is None:
            idx.linear += coeff
            return
        node = self.b.nodes[idx.wide]
        pv, factor = _vector(node.dim, coeff * (idx.lo + INDEX), expand=False)
        if pv is not None:
            node.phase = pv if node.phase is None else _add(node.phase, pv)
        self.b.scalar *= factor

    def _wide(self, x: sympy.Symbol) -> int:
        idx = self.indices[x]
        if idx.wide is None:
            dim = self.b.dim(idx.size)
            pv, factor = _vector(dim, idx.linear * (idx.lo + INDEX), expand=False)
            idx.wide = self.b.node(Z_SPIDER, dim, pv)
            self.b.scalar *= factor
        return idx.wide

    # building ---------------------------------------------------------------------------

    def build(self) -> None:
        for fam in self.families:
            fam.layout()
            fam.boxes = [self.b.box(m) for m in fam.segments]
        for x, idx in self.indices.items():
            if idx.family is None:
                continue
            d = self.b.dim(idx.family.radix)
            idx.nodes = [self.b.node(Z_SPIDER, d, None, box) for box in idx.family.boxes]
        for f in sympy.Mul.make_args(self.amp):
            self._factor(f, dry=False)
        for x, idx in self.indices.items():
            if x in self.wide and idx.linear != 0:
                self._wide(x)
        for r, k in enumerate(self.kets):
            if self._ket_plan(r, k, dry=False):
                continue
            fam = self.reg_family[r]
            size = fam.radix**fam.width
            dim = self.b.dim(size)
            u = self.b.node(Z_SPIDER, dim)
            self.b.output(u)
            self._constraint(k, size, u, dry=False)
        for x, idx in self.indices.items():
            if idx.family is None and idx.wide is None:
                self.b.scalar *= idx.size


def _add(a: PhaseVector, b: PhaseVector) -> PhaseVector:
    if a.rule is not None or b.rule is not None:
        rules = [r for r in (a.rule, b.rule) if r is not None]
        rule = rules[0] if len(rules) == 1 else rules[0] + rules[1]
        merged = {k: a.get(k) + b.get(k) for k in set(a.entries()) | set(b.entries())}
        return PhaseVector(a.dim, merged, rule, expand=False)
    return a + b


def _turns(f: Expr) -> Expr | None:
    """The turns of a factor that is a pure phase ``b^e``, ``e^x`` or ``i^e``; None otherwise."""
    if isinstance(f, sympy.exp):
        t = sympy.expand(f.args[0] / (2 * sympy.pi * sympy.I))
        return t if t.is_real is not False else None
    if f.is_Pow and not f.base.free_symbols:
        base = f.base
        if base == -1:
            return f.exp / 2
        if base == sympy.I:
            return f.exp / 4
        if base == -sympy.I:
            return -f.exp / 4
        if abs(complex(base)) == 1:
            arg = sympy.arg(base) / (2 * sympy.pi)
            return f.exp * sympy.nsimplify(arg)
    return None


# -- states ----------------------------------------------------------------------------------


def _fold(s: P.State) -> P.State:
    """Several terms as one with a selector index over the terms."""
    terms = [P.fresh(t) for t in s.terms]
    sel = P.new_bound("s", 0, len(terms) - 1)
    sv = sel.symbol
    amp = sympy.S.One
    bound: list[P.Bound] = [sel]
    for i, t in enumerate(terms):
        size = sympy.Mul(*[b.hi - b.lo + 1 for b in t.bound])
        amp *= sympy.Piecewise((t.amp, sympy.Eq(sv, i)), (1 / size, True))
        bound.extend(t.bound)
    kets = []
    for r in range(len(s.widths)):
        values = [t.kets[r] for t in terms]
        if all(v == values[0] for v in values):
            kets.append(values[0])
            continue
        kets.append(sympy.Add(*[sympy.KroneckerDelta(sv, i) * v for i, v in enumerate(values)]))
    return P.State(s.widths, (P.Term(amp, tuple(bound), tuple(kets)),), False, s.dims)


def diagram(s: P.State) -> tuple[Diagram, list[str]]:
    """A ZXW diagram of the ket ``s`` and notes naming the dimensions outside the Dim grammar."""
    b = _Builder()
    if len(s.terms) > 1:
        s = _fold(s)
    term = _Term(s.terms[0], s, b)
    term.build()
    return b.build(), b.notes
