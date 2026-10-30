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

"""The Python embedding of the engine: one :class:`Program` per session.

::

    from archytaszx import azx                 # or azx = archytaszx.Program()
    n = azx.count("n")
    data = (azx.F ^ 4 | azx.X(0, 1)) ** n      # n data qudits, F^4 on |0>
    target = azx.X(2, 1) | azx.F ^ 3 | azx.X(0, 1, phase="-k/d")
    circuit = data @ target * azx.d ** (-n / 2)
    azx.decide(circuit, goal)                  # equal, for every n and d
    azx.lean("dj.lean")
    azx("rules spider_fusion")                 # any engine command; returns what it printed

A :class:`Piece` is a diagram fragment. ``a | b`` feeds b's leading outputs into a's leading
inputs (Dirac order: b first), ``@`` is the tensor product, ``p ** k`` the tensor power (a
!-box when k is a symbol), ``p ^ k`` k copies in series and ``*`` a scalar factor. A
:class:`Sym` is a symbolic expression with ``+ - * / **``. Every piece compiles to the DSL
(:meth:`Piece.dsl`) and is read by :func:`~archytaszx.repl.parser.parse_dsl`, so node ids
follow the order in which nodes happen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

from archytaszx.diagram.graph import Diagram
from archytaszx.repl import zxdraw
from archytaszx.repl.commands import Outcome, _Pair
from archytaszx.repl.parser import parse_dsl
from archytaszx.repl.printer import NotationMode, render
from archytaszx.repl.shell import Shell, _logical_lines
from archytaszx.semantics.decide import Decision


class AzxError(Exception):
    """A piece that cannot be built or compiled."""


_RESERVED = frozenset({"I", "E", "pi", "exp", "sqrt", "k"})
_IDENT = re.compile(r"[A-Za-z_]\w*")


class Sym:
    """A symbolic expression: its text and its precedence (4 atom, 3 power, 2 product, 1 sum)."""

    __slots__ = ("prec", "text")

    def __init__(self, text: object, prec: int = 4) -> None:
        self.text = str(text)
        self.prec = prec

    def _op(self, op: str, prec: int, other: object, swap: bool) -> Sym:
        if isinstance(other, Piece):
            return NotImplemented  # type: ignore[no-any-return]
        a, b = self, _sym(other)
        if swap:
            a, b = b, a
        left = a.text if a.prec >= prec + (op == "^") else f"({a.text})"
        right = b.text if b.prec > prec or (b.prec == prec and op in "+*") else f"({b.text})"
        return Sym(f"{left} {op} {right}" if prec == 1 else f"{left}{op}{right}", prec)

    def __add__(self, o: object) -> Sym:
        return self._op("+", 1, o, False)

    def __radd__(self, o: object) -> Sym:
        return self._op("+", 1, o, True)

    def __sub__(self, o: object) -> Sym:
        return self._op("-", 1, o, False)

    def __rsub__(self, o: object) -> Sym:
        return self._op("-", 1, o, True)

    def __mul__(self, o: object) -> Sym:
        return self._op("*", 2, o, False)

    def __rmul__(self, o: object) -> Sym:
        return self._op("*", 2, o, True)

    def __truediv__(self, o: object) -> Sym:
        return self._op("/", 2, o, False)

    def __rtruediv__(self, o: object) -> Sym:
        return self._op("/", 2, o, True)

    def __pow__(self, o: object) -> Sym:
        return self._op("^", 3, o, False)

    def __rpow__(self, o: object) -> Sym:
        return self._op("^", 3, o, True)

    def __neg__(self) -> Sym:
        return Sym(f"-{self.text}" if self.prec >= 2 else f"-({self.text})", 2)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<azx sym {self.text}>"


def _sym(value: object) -> Sym:
    """``value`` as a :class:`Sym`: Sym, int, Fraction or expression text."""
    if isinstance(value, Sym):
        return value
    if isinstance(value, bool):
        raise AzxError(f"not an expression: {value!r}")
    if isinstance(value, int):
        return Sym(value, 4 if value >= 0 else 2)
    if isinstance(value, Fraction):
        return Sym(value, 4) if value.denominator == 1 and value >= 0 else Sym(value, 2)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        return Sym(text, 4 if _IDENT.fullmatch(text) or text.isdigit() else 1)
    raise AzxError(f"not an expression: {value!r}")


@dataclass(frozen=True, slots=True)
class _Node:
    """One node: its DSL text after ``NAME = ``, its leg counts, its port !-boxes
    ``(side, index, mult)`` and whether it is a bare wire."""

    text: str
    nin: int
    nout: int
    fans: tuple[tuple[str, int, str], ...] = ()
    wire: bool = False


Port = tuple[int, int]
"""A node index and a leg index."""


@dataclass(frozen=True, slots=True)
class Piece:
    """A diagram fragment: nodes, wires ``(out port, in port)``, node !-boxes ``(mult,
    parent)``, each node's innermost box, the free inputs and outputs, and scalar factors."""

    program: Program
    nodes: tuple[_Node, ...] = ()
    wires: tuple[tuple[Port, Port], ...] = ()
    boxes: tuple[tuple[str, int | None], ...] = ()
    owner: tuple[int | None, ...] = ()
    ins: tuple[Port, ...] = ()
    outs: tuple[Port, ...] = ()
    scalar: tuple[str, ...] = ()

    # -- combinators ---------------------------------------------------------------------

    def _shifted(self, nodes: int, boxes: int) -> Piece:
        def port(p: Port) -> Port:
            return (p[0] + nodes, p[1])

        return replace(
            self,
            wires=tuple((port(a), port(b)) for a, b in self.wires),
            boxes=tuple((m, None if p is None else p + boxes) for m, p in self.boxes),
            owner=tuple(None if o is None else o + boxes for o in self.owner),
            ins=tuple(map(port, self.ins)),
            outs=tuple(map(port, self.outs)),
        )

    def _beside(self, other: Piece) -> Piece:
        b = other._shifted(len(self.nodes), len(self.boxes))
        return Piece(
            self.program,
            self.nodes + b.nodes,
            self.wires + b.wires,
            self.boxes + b.boxes,
            self.owner + b.owner,
            self.ins + b.ins,
            self.outs + b.outs,
            self.scalar + b.scalar,
        )

    def _chain(self, node: int) -> list[int]:
        out: list[int] = []
        box = self.owner[node]
        while box is not None:
            out.append(box)
            box = self.boxes[box][1]
        return out

    def _fanned(self, port: Port, side: str) -> bool:
        return any(s == side and i == port[1] for s, i, _ in self.nodes[port[0]].fans)

    def _merge_boxes(self, keep: int, drop: int) -> Piece:
        def fix(b: int | None) -> int | None:
            return keep if b == drop else b

        return replace(
            self,
            boxes=tuple((m, fix(p)) for m, p in self.boxes),
            owner=tuple(map(fix, self.owner)),
        )

    def __or__(self, other: object) -> Piece:
        """``self`` applied to ``other``: other's leading outputs feed self's leading inputs."""
        if not isinstance(other, Piece):
            return NotImplemented
        if other.program is not self.program:
            raise AzxError("pieces of different programs")
        joined = other._beside(self)
        a = self._shifted(len(other.nodes), len(other.boxes))
        k = min(len(a.ins), len(other.outs))
        pairs = list(zip(other.outs[:k], a.ins[:k]))
        for src, dst in pairs:
            if joined._fanned(src, "out") or joined._fanned(dst, "in"):
                raise AzxError("a !-box leg cannot be wired here; use azx(...) with the DSL")
            while True:
                left, right = joined._chain(src[0]), joined._chain(dst[0])
                if left == right:
                    break
                diff = [(x, y) for x, y in zip(left[::-1], right[::-1]) if x != y]
                if (
                    len(left) != len(right)
                    or joined.boxes[diff[0][0]][0] != joined.boxes[diff[0][1]][0]
                ):
                    raise AzxError(
                        "a wire crosses a !-box boundary; repeat the whole chain instead"
                    )
                joined = joined._merge_boxes(*diff[0])
        return replace(
            joined,
            wires=joined.wires + tuple(pairs),
            ins=other.ins + a.ins[k:],
            outs=a.outs + other.outs[k:],
        )

    def __matmul__(self, other: object) -> Piece:
        """The tensor product: ``self`` above ``other``."""
        if not isinstance(other, Piece):
            return NotImplemented
        return self._beside(other)

    def __pow__(self, count: object) -> Piece:
        """The tensor power: ``count`` copies side by side, one !-box when it is symbolic."""
        if isinstance(count, int) and not isinstance(count, bool):
            if count < 0:
                raise AzxError("a tensor power must be >= 0")
            out = Piece(self.program)
            for _ in range(count):
                out = out._beside(self)
            return out
        mult = _sym(count).text
        top = len(self.boxes)
        return replace(
            self,
            boxes=tuple((m, top if p is None else p) for m, p in self.boxes) + ((mult, None),),
            owner=tuple(top if o is None else o for o in self.owner),
        )

    def __xor__(self, count: object) -> Piece:
        """``count`` copies of ``self`` in series."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise AzxError("a series power must be an int >= 1")
        out = self
        for _ in range(count - 1):
            out = self | out
        return out

    def __mul__(self, factor: object) -> Piece:
        if isinstance(factor, Piece):
            return NotImplemented
        return replace(self, scalar=self.scalar + (_sym(factor).text,))

    __rmul__ = __mul__

    def __truediv__(self, factor: object) -> Piece:
        return self * (1 / _sym(factor))

    # -- output --------------------------------------------------------------------------

    def _compiled(self) -> Piece:
        """``self`` with every bare wire node spliced out, or kept as ``Z(1 -> 1)`` when it
        joins an input straight to an output."""
        p = self
        while True:
            hit = next(
                (
                    i
                    for i, n in enumerate(p.nodes)
                    if n.wire and not ((i, 0) in p.ins and (i, 0) in p.outs)
                ),
                None,
            )
            if hit is None:
                break
            src = next((a for a, b in p.wires if b == (hit, 0)), None)
            dst = next((b for a, b in p.wires if a == (hit, 0)), None)
            wires = [w for w in p.wires if hit not in (w[0][0], w[1][0])]
            ins, outs = list(p.ins), list(p.outs)
            if src is not None and dst is not None:
                wires.append((src, dst))
            elif src is not None:
                outs[outs.index((hit, 0))] = src
            elif dst is not None:
                ins[ins.index((hit, 0))] = dst
            else:
                ins = [x for x in ins if x[0] != hit]
                outs = [x for x in outs if x[0] != hit]

            def port(q: Port, hit: int = hit) -> Port:
                return (q[0] - (q[0] > hit), q[1])

            p = replace(
                p,
                nodes=p.nodes[:hit] + p.nodes[hit + 1 :],
                owner=p.owner[:hit] + p.owner[hit + 1 :],
                wires=tuple((port(a), port(b)) for a, b in wires),
                ins=tuple(map(port, ins)),
                outs=tuple(map(port, outs)),
            )
        return p

    def dsl(self) -> str:
        """The piece as DSL text."""
        p = self._compiled()
        texts = [n.text for n in p.nodes] + [m for m, _ in p.boxes] + list(p.scalar)
        names = {w for t in texts for w in _IDENT.findall(t)} | set(self.program._bindings)
        stem = "v"
        while any(re.fullmatch(rf"{stem}[bf]?\d+", w) for w in names):
            stem = "_" + stem
        node = [f"{stem}{i}" for i in range(len(p.nodes))]
        box = [f"{stem}b{i}" for i in range(len(p.boxes))]
        used = names - _RESERVED - {w for t in texts for w in re.findall(r"\b[A-Z]\w*(?=\()", t)}
        mults = [w for m, _ in p.boxes for w in _IDENT.findall(m)]
        mults += [w for n in p.nodes for _, _, m in n.fans for w in _IDENT.findall(m)]
        counts = list(dict.fromkeys(w for w in mults if w not in _RESERVED))
        lines: list[str] = []
        if counts:
            lines.append("count " + ", ".join(self.program._binding(w) for w in counts))
        lets = [w for w in sorted(used) if w not in counts and w in self.program._values]
        if lets:
            lines.append("let " + ", ".join(self.program._binding(w) for w in lets))
        lines += [
            f"{node[i]} = {n.text.replace('wire', 'Z(1 -> 1)', 1) if n.wire else n.text}"
            for i, n in enumerate(p.nodes)
        ]
        if p.wires:
            lines.append(
                "wire "
                + ", ".join(f"{node[a]}.out{i} -- {node[b]}.in{j}" for (a, i), (b, j) in p.wires)
            )
        for b, (mult, parent) in enumerate(p.boxes):
            members = [node[i] for i, o in enumerate(p.owner) if o == b]
            if members:
                tail = "" if parent is None else f" in {box[parent]}"
                lines.append(f"box {box[b]} x {mult} nodes {', '.join(members)}{tail}")
        fan = 0
        for i, n in enumerate(p.nodes):
            for side, leg, mult in n.fans:
                owner = p.owner[i]
                tail = "" if owner is None else f" in {box[owner]}"
                lines.append(f"box {stem}f{fan} x {mult} ports {node[i]}.{side}{leg}{tail}")
                fan += 1
        lines.append("inputs " + ", ".join(f"{node[a]}.in{i}" for a, i in p.ins))
        lines.append("outputs " + ", ".join(f"{node[a]}.out{i}" for a, i in p.outs))
        if p.scalar:
            lines.append("scalar " + " * ".join(f"({s})" for s in p.scalar))
        return "\n".join(line.rstrip() for line in lines)

    def diagram(self) -> Diagram:
        """The engine diagram."""
        return parse_dsl(self.dsl())

    def draw(self, style: str = "unicode") -> str:
        """The ZX drawing: matplot, unicode or ascii."""
        return zxdraw.draw(zxdraw.View(self.diagram()), style)

    def _repr_png_(self) -> bytes:
        from archytaszx.repl import terminal as T

        lay = zxdraw.layout(zxdraw.View(self.diagram()))
        return zxdraw.figure(lay, dark_theme=False, opaque=True, palette=T.WHITE)

    def __str__(self) -> str:
        return render(self.diagram(), NotationMode.DIRAC, unicode=True)

    def __repr__(self) -> str:
        return f"<azx piece: {len(self.ins)} -> {len(self.outs)}, {len(self.nodes)} nodes>"


@dataclass(frozen=True, slots=True)
class Result:
    """A command's printed text, its success, and the decision behind it when there is one."""

    text: str
    ok: bool
    decision: Decision | None = None

    @property
    def verdict(self) -> str | None:
        return None if self.decision is None else self.decision.verdict.value

    @property
    def method(self) -> str | None:
        return None if self.decision is None else self.decision.method.value

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<azx result: {self.text.splitlines()[0] if self.text else ''}>"


_OWN = frozenset({"_shell", "_bindings", "_values", "_named"})


def _legs(count: object, side: str) -> tuple[int, tuple[tuple[str, int, str], ...]]:
    """A leg count: an int gives that many legs, a symbol one leg in a port !-box."""
    if isinstance(count, int) and not isinstance(count, bool):
        if count < 0:
            raise AzxError("a leg count must be >= 0")
        return count, ()
    return 1, ((side, 0, _sym(count).text),)


def _vector(word: str, value: object) -> str:
    """A ``phase`` or ``weight`` suffix: 0, one expression at every index k, or {index: expr}."""
    if value is None:
        return ""
    if isinstance(value, int) and not isinstance(value, bool) and value == 0 and word == "phase":
        return " phase 0"
    if isinstance(value, Mapping):
        body = ", ".join(f"{i}: {_sym(e).text}" for i, e in value.items())
        return f" {word}{{{body}}}"
    return f" {word}{{k: {_sym(value).text}}}"


class Program:
    """An engine session reached from Python; ``azx.NAME = piece`` stores ``@NAME``."""

    _shell: Shell
    _bindings: dict[str, bool]
    _values: dict[str, int]
    _named: dict[str, Piece | str]

    def __init__(self, *, unicode: bool = True) -> None:
        object.__setattr__(self, "_shell", Shell(unicode=unicode))
        object.__setattr__(self, "_bindings", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_named", {})

    # -- commands ------------------------------------------------------------------------

    def __call__(self, source: str) -> str:
        """Runs engine commands (indented lines continue a command) and returns their output."""
        out = [self._shell.run_line(line) for line in _logical_lines(source.splitlines())]
        return "\n".join(o for o in out if o)

    def _run(self, line: str) -> Result:
        text = self._shell.run_line(line)
        outcome: Outcome | None = self._shell.last_outcome
        ok = outcome is not None and outcome.ok
        detail = outcome.detail if outcome is not None else None
        decision = detail.payload if isinstance(detail, _Pair) else None
        return Result(text, ok, decision if isinstance(decision, Decision) else None)

    def _source(self, value: Piece | str) -> str:
        if isinstance(value, Piece):
            return "\n" + value.dsl()
        if isinstance(value, str):
            return value if value.startswith("@") else "\n" + value
        raise AzxError(f"not a diagram: {value!r}")

    def load(self, piece: Piece | str) -> Result:
        """Makes ``piece`` (a Piece, DSL text or ``@NAME``) the current diagram."""
        return self._run("load " + self._source(piece))

    def _pair(self, command: str, left: Piece | str, right: Piece | str | None) -> Result:
        loaded = self.load(left)
        if not loaded.ok:
            return loaded
        if right is None:
            return self._run(command)
        if not (isinstance(right, str) and right.startswith("@")):
            self._run("@azx_target = " + self._source(right))
            right = "@azx_target"
        return self._run(f"{command} against {right}")

    def decide(self, left: Piece | str, right: Piece | str | None = None) -> Result:
        """Decides ``left == right`` for every value of their symbols."""
        return self._pair("decide", left, right)

    def prove(
        self, left: Piece | str, right: Piece | str | None = None, *, exhaustive: bool = False
    ) -> Result:
        """Searches for a certified derivation of ``left == right``."""
        return self._pair("prove exhaustive" if exhaustive else "prove", left, right)

    def check(self, left: Piece | str, right: Piece | str, **values: object) -> Result:
        """Compares ``left`` and ``right`` with the numeric oracle, at ``values`` when given."""
        given = " ".join(f"{k}={v}" for k, v in values.items())
        return self._pair(f"check {given}".strip(), left, right)

    def lean(self, path: str | None = None) -> Result:
        """Exports the latest proof to Lean 4, written to ``path`` or printed."""
        return self._run("certificate lean" + (f" {path}" if path else ""))

    def kernel(self) -> Result:
        """Re-checks the latest proof in the proof kernel."""
        return self._run("certificate kernel")

    # -- symbols -------------------------------------------------------------------------

    def _binding(self, name: str) -> str:
        value = self._values.get(name)
        return name if value is None else f"{name} = {value}"

    def _symbol(self, name: str, value: int | None) -> Sym:
        if not _IDENT.fullmatch(name) or name in _RESERVED:
            raise AzxError(f"not a symbol name: {name!r}")
        self._bindings[name] = True
        if value is not None:
            self._values[name] = int(value)
        return Sym(name)

    def count(self, name: str, value: int | None = None) -> Sym:
        """A multiplicity symbol, optionally bound to a value substituted after the proof."""
        return self._symbol(name, value)

    def dim(self, name: str = "d", value: int | None = None) -> Sym:
        """A dimension symbol, optionally bound."""
        return self._symbol(name, value)

    @property
    def d(self) -> Sym:
        """The default dimension."""
        return Sym("d")

    # -- generators ----------------------------------------------------------------------

    def _piece(self, node: _Node) -> Piece:
        return Piece(
            self,
            (node,),
            owner=(None,),
            ins=tuple((0, i) for i in range(node.nin)),
            outs=tuple((0, i) for i in range(node.nout)),
        )

    def _spider(
        self, gen: str, m: object, n: object, phase: object, weight: object, dim: object
    ) -> Piece:
        nin, fin = _legs(m, "in")
        nout, fout = _legs(n, "out")
        d = "" if dim is None else f", {_sym(dim).text}"
        text = f"{gen}({nin} -> {nout}{d}){_vector('phase', phase)}{_vector('weight', weight)}"
        return self._piece(_Node(text, nin, nout, fin + fout))

    def Z(
        self,
        m: object = 0,
        n: object = 1,
        *,
        phase: object = None,
        weight: object = None,
        dim: object = None,
    ) -> Piece:
        """A Z spider; a symbolic leg count is one leg in a port !-box."""
        return self._spider("Z", m, n, phase, weight, dim)

    def X(self, m: object = 0, n: object = 1, *, phase: object = None, dim: object = None) -> Piece:
        """An X spider."""
        return self._spider("X", m, n, phase, None, dim)

    def W(self, m: object = 1, n: object = 2, *, dim: object = None) -> Piece:
        """A W node."""
        return self._spider("W", m, n, None, None, dim)

    def _box(self, gen: str, dim: object) -> Piece:
        text = gen if dim is None else f"{gen}({_sym(dim).text})"
        return self._piece(_Node(text, 1, 1))

    def fourier(self, dim: object = None) -> Piece:
        """The Fourier box."""
        return self._box("F", dim)

    def triangle(self, dim: object = None, *, inverse: bool = False) -> Piece:
        """The triangle, or its inverse."""
        return self._box("Ti" if inverse else "T", dim)

    F = property(fourier)
    T = property(triangle)
    Ti = property(lambda self: self.triangle(inverse=True))

    def B(self, s: object, t: object) -> Piece:
        """The binder: inputs s and t, output s*t."""
        return self._piece(_Node(f"B({_sym(s).text}, {_sym(t).text})", 2, 1))

    def S(self, s: object, t: object) -> Piece:
        """The splitter: input s*t, outputs s and t."""
        return self._piece(_Node(f"S({_sym(s).text}, {_sym(t).text})", 1, 2))

    def node(
        self,
        gen: str,
        ins: list[object],
        outs: list[object],
        *,
        phase: object = None,
        weight: object = None,
    ) -> Piece:
        """Any generator with explicit leg dimensions."""
        body = ", ".join(_sym(x).text for x in ins), ", ".join(_sym(x).text for x in outs)
        text = f"{gen}(in: [{body[0]}], out: [{body[1]}])"
        text += _vector("phase", phase) + _vector("weight", weight)
        return self._piece(_Node(text, len(ins), len(outs)))

    @property
    def id(self) -> Piece:
        """A bare wire."""
        return self._piece(_Node("wire", 1, 1, wire=True))

    @property
    def swap(self) -> Piece:
        """Two crossing wires."""
        two = self.id @ self.id
        return replace(two, outs=(two.outs[1], two.outs[0]))

    def scalar(self, value: object) -> Piece:
        """The empty diagram times ``value``."""
        return Piece(self, scalar=(_sym(value).text,))

    # -- names ---------------------------------------------------------------------------

    def __setattr__(self, name: str, value: object) -> None:
        if name in _OWN or name.startswith("_"):
            raise AttributeError(name)
        if not isinstance(value, (Piece, str)):
            raise AzxError(f"azx.{name} takes a piece or DSL text")
        result = self._run(f"@{name} = " + self._source(value))
        if not result.ok:
            raise AzxError(result.text)
        self._named[name] = value

    def __getattr__(self, name: str) -> Piece | str:
        named: dict[str, Piece | str] = self.__dict__.get("_named", {})
        if name in named:
            return named[name]
        raise AttributeError(name)


azx = Program()
"""The default program."""
