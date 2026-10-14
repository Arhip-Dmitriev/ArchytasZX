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

"""Dirac rendering of :mod:`archytaszx.repl.lang` values in three styles.

* ``ascii``: the input syntax (``|0>^@n @ |0>^@m``, ``sum(x in 0..q-1){ ... }``);
* ``unicode``: ``|0⟩^⊗n ⊗ |0⟩^⊗m``, ``Σ_{x=0}^{q-1} ...``;
* ``plot``: matplotlib mathtext, one line per entry, drawn off-screen and shown inline in the
  terminal (kitty, iTerm2 or sixel image protocol).

A :class:`Renderer` renders entries and records the defined variables, functions and
measurement outcomes they mention; :meth:`Renderer.where` lists those as a ``where`` line.
Defined variables are folded back into expressions (``2^(-n/2)`` shows as ``1/sqrt(q)``
under ``var q = 2^n``). :func:`terminal_image` returns ``plot`` output as an inline image escape
sequence, or None when the terminal has no image protocol.
"""

from __future__ import annotations

import base64
import io
import itertools
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker
from sympy.printing.latex import LatexPrinter  # type: ignore[import-untyped]
from sympy.printing.str import StrPrinter  # type: ignore[import-untyped]

from archytaszx.semantics import pathsum as P

STYLES = ("plot", "unicode", "ascii")
"""The display styles, most to least rich."""

_GREEK = {
    name: (uni, "\\" + name)
    for name, uni in {
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "epsilon": "ε",
        "zeta": "ζ",
        "eta": "η",
        "theta": "θ",
        "iota": "ι",
        "kappa": "κ",
        "lambda": "λ",
        "mu": "μ",
        "nu": "ν",
        "xi": "ξ",
        "rho": "ρ",
        "sigma": "σ",
        "tau": "τ",
        "phi": "φ",
        "chi": "χ",
        "psi": "ψ",
        "omega": "ω",
        "Gamma": "Γ",
        "Delta": "Δ",
        "Theta": "Θ",
        "Lambda": "Λ",
        "Xi": "Ξ",
        "Pi": "Π",
        "Sigma": "Σ",
        "Phi": "Φ",
        "Psi": "Ψ",
        "Omega": "Ω",
    }.items()
}


def name_text(name: str, style: str) -> str:
    """An identifier with a leading Greek letter name shown as the letter."""
    head, sep, tail = name.partition("_")
    if style == "ascii" or head not in _GREEK:
        greek = head
    else:
        greek = _GREEK[head][0] if style == "unicode" else _GREEK[head][1]
    if style == "plot":
        return f"{greek}_{{{tail}}}" if sep else greek
    return greek + sep + tail


_TEXT_FUNCS = {"xor", "bits", "shr", "pop", "bitdot", "powmod"}


def _user(e: sympy.Basic) -> bool:
    return hasattr(type(e), "_body")


class _Text(StrPrinter):  # type: ignore[misc]
    """ASCII (``uni=False``) or Unicode expression text."""

    def __init__(self, uni: bool) -> None:
        super().__init__()
        self.uni = uni

    def _paren(self, e: sympy.Basic) -> str:
        text = str(self._print(e))
        if e.is_Symbol or (e.is_Integer and e >= 0) or e in (sympy.pi, sympy.E, sympy.I):
            return text
        if isinstance(e, sympy.Function) and not isinstance(e, P.xor):
            return text
        return f"({text})"

    def _print_ImaginaryUnit(self, e: sympy.Basic) -> str:
        return "i"

    def _print_Pi(self, e: sympy.Basic) -> str:
        return "π" if self.uni else "pi"

    def _print_Exp1(self, e: sympy.Basic) -> str:
        return "e"

    def _print_Dummy(self, e: sympy.Dummy) -> str:
        return name_text(e.name, "unicode" if self.uni else "ascii")

    def _print_Symbol(self, e: sympy.Symbol) -> str:
        return name_text(e.name, "unicode" if self.uni else "ascii")

    def _sqrt(self, b: sympy.Basic) -> str:
        if not self.uni:
            return f"sqrt({self._print(b)})"
        return f"√{self._print(b)}" if b.is_Atom else f"√({self._print(b)})"

    def _exp(self, arg: sympy.Expr) -> str:
        c = sympy.factor_terms(sympy.expand(arg / (2 * sympy.pi * sympy.I)))
        if c.has(sympy.I) or c.has(sympy.pi):
            body = self._print(arg)
        else:
            sign, c = ("-", -c) if c.could_extract_minus_sign() else ("", c)
            twopi = "2πi" if self.uni else "2*pi*i"
            dot = "·" if self.uni else "*"
            if c.is_Rational and c != 1:
                num = "" if c.p == 1 else dot + str(c.p)
                body = f"{sign}{twopi}{num}/{c.q}"
            else:
                ctext = self._print(c)
                wrapped = f"({ctext})" if c.is_Add else ctext
                body = sign + twopi + ("" if c == 1 else dot + wrapped)
        return f"e^{{{body}}}" if self.uni else f"e^({body})"

    def _print_exp(self, e: sympy.exp) -> str:
        return self._exp(e.args[0])

    def _print_Pow(self, e: sympy.Pow, rational: bool = False) -> str:
        b, x = e.base, e.exp
        if x == sympy.Rational(1, 2):
            return self._sqrt(b)
        if x == -sympy.Rational(1, 2):
            return "1/" + self._sqrt(b)
        if b == sympy.E:
            return self._exp(x)
        if x == -1:
            return "1/" + self._paren(b)
        exp_text = self._print(x)
        if not (x.is_Symbol or (x.is_Integer and x >= 0)):
            exp_text = f"({exp_text})"
        return f"{self._paren(b)}^{exp_text}"

    def _print_Mul(self, e: sympy.Mul) -> str:
        text = str(super()._print_Mul(e))
        return text.replace("*", "·") if self.uni else text

    def _print_Mod(self, e: sympy.Mod) -> str:
        a, m = e.args
        left = self._paren(a) if a.is_Add else self._print(a)
        return f"{left} {'mod' if self.uni else '%'} {self._paren(m)}"

    def _print_KroneckerDelta(self, e: sympy.KroneckerDelta) -> str:
        return f"[{self._print(e.args[0])} = {self._print(e.args[1])}]"

    def _print_Function(self, e: sympy.Function) -> str:
        name = type(e).__name__
        args = e.args
        if name == "xor":
            return f"({self._print(args[0])} {'⊕' if self.uni else 'xor'} {self._print(args[1])})"
        if name == "bits":
            z, lo, hi = (self._print(a) for a in args[:3])
            if args[1] == 0:
                return f"{z}[:{hi}]"
            if P.same(args[2], args[3]):
                return f"{z}[{lo}:]"
            return f"{z}[{lo}:{hi}]"
        if name == "shr":
            return f"({self._print(args[0])} >> {self._print(args[1])})"
        if name == "pop":
            return f"popcount({self._print(args[0])})"
        if name == "bitdot":
            return f"{self._paren(args[0])}{'·' if self.uni else '.'}{self._paren(args[1])}"
        if name == "powmod":
            return str(self._print(sympy.Mod(sympy.Pow(args[0], args[1], evaluate=False), args[2])))
        return str(super()._print_Function(e))


class _Latex(LatexPrinter):  # type: ignore[misc]
    """Matplotlib-mathtext expression text."""

    def __init__(self, settings: dict[str, object] | None = None) -> None:
        super().__init__(settings or {"imaginary_unit": "i"})

    def _print_Symbol(self, e: sympy.Symbol, style: str = "plain") -> str:
        return name_text(e.name, "plot")

    _print_Dummy = _print_Symbol

    def _print_Mod(self, e: sympy.Mod, exp: str | None = None) -> str:
        a, m = e.args
        return rf"{self._print(a)}\ \mathrm{{mod}}\ {self._print(m)}"

    def _print_exp(self, e: sympy.exp, exp: str | None = None) -> str:
        arg = e.args[0]
        c = sympy.factor_terms(sympy.expand(arg / (2 * sympy.pi * sympy.I)))
        if c.has(sympy.I) or c.has(sympy.pi):
            return str(super()._print_exp(e, exp))
        sign, c = ("-", -c) if c.could_extract_minus_sign() else ("", c)
        inline = _Latex({"fold_short_frac": True, "imaginary_unit": "i"})
        body = "" if c == 1 else r"\," + inline.doprint(c)
        return rf"e^{{{sign}2\pi i{body}}}"

    def _print_KroneckerDelta(self, e: sympy.KroneckerDelta, exp: str | None = None) -> str:
        return rf"[{self._print(e.args[0])} = {self._print(e.args[1])}]"

    def _print_Function(self, e: sympy.Function, exp: str | None = None) -> str:
        name = type(e).__name__
        args = e.args
        if name == "xor":
            return rf"({self._print(args[0])} \oplus {self._print(args[1])})"
        if name == "bits":
            z, lo, hi = (self._print(a) for a in args[:3])
            if args[1] == 0:
                return f"{z}[:{hi}]"
            return f"{z}[{lo}:]" if P.same(args[2], args[3]) else f"{z}[{lo}:{hi}]"
        if name == "shr":
            return rf"({self._print(args[0])} \gg {self._print(args[1])})"
        if name == "pop":
            return rf"\mathrm{{pop}}({self._print(args[0])})"
        if name == "bitdot":
            return rf"{self._print(args[0])} \cdot {self._print(args[1])}"
        if name == "powmod":
            return str(self._print(sympy.Mod(sympy.Pow(args[0], args[1], evaluate=False), args[2])))
        if _user(e):
            inner = ", ".join(self._print(a) for a in args)
            return f"{name}({inner})"
        return str(super()._print_Function(e, exp))

    def doprint(self, expr: sympy.Basic) -> str:
        return str(super().doprint(expr)).replace(r"\left", "").replace(r"\right", "")


@dataclass(frozen=True)
class Label:
    """How an entry is named: ``kind`` is ket, bra, var, func, op or list."""

    kind: str
    name: str


@dataclass
class Context:
    """Interpreter facts a renderer consults."""

    sugar: Sequence[tuple[sympy.Expr, sympy.Symbol]] = ()
    measured: dict[
        sympy.Symbol, tuple[sympy.Expr, str, tuple[tuple[sympy.Expr, sympy.Expr], ...]]
    ] = field(default_factory=dict)
    taken: frozenset[str] = frozenset()


class Renderer:
    """Renders entries in one style and collects the names they mention."""

    def __init__(self, style: str, context: Context | None = None) -> None:
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}")
        self.style = style
        self.ctx = context or Context()
        self._text = _Latex() if style == "plot" else _Text(style == "unicode")
        self.seen_symbols: list[sympy.Symbol] = []
        self.seen_funcs: list[type] = []

    def _record(self, e: sympy.Basic) -> None:
        for s in sorted(e.free_symbols, key=lambda s: s.name):
            if s not in self.seen_symbols:
                self.seen_symbols.append(s)
        for app in e.atoms(sympy.Function):
            if _user(app) and type(app) not in self.seen_funcs:
                self.seen_funcs.append(type(app))
            for arg in app.args:
                self._record(arg)

    def sugar(self, e: sympy.Basic, skip: sympy.Symbol | None = None) -> sympy.Basic:
        """``e`` with defined variables other than ``skip`` folded back in."""
        for value, sym in self.ctx.sugar:
            if sym != skip:
                e = e.subs(value, sym)
        return e

    def expr(self, e: sympy.Basic, skip: sympy.Symbol | None = None) -> str:
        """Expression text; records what it mentions."""
        e = self.sugar(sympy.sympify(e), skip)
        self._record(e)
        return self._text.doprint(e)

    def _wrap(self, e: sympy.Basic) -> str:
        text = self.expr(e)
        return f"({text})" if sympy.sympify(e).is_Add else text

    def label(self, label: Label) -> str:
        """``|psi>``, ``<psi|`` or the bare name."""
        name = name_text(label.name, self.style)
        if label.kind == "ket":
            return self._ket(name)
        if label.kind == "bra":
            return self._bra(name)
        return name

    def _ket(self, inner: str) -> str:
        return {"ascii": f"|{inner}>", "unicode": f"|{inner}⟩", "plot": rf"|{inner}\rangle"}[
            self.style
        ]

    def _bra(self, inner: str) -> str:
        return {"ascii": f"<{inner}|", "unicode": f"⟨{inner}|", "plot": rf"\langle {inner}|"}[
            self.style
        ]

    def _tpow(self, base: str, w: sympy.Expr) -> str:
        wt = self.expr(w)
        if self.style == "plot":
            return rf"{base}^{{\otimes {wt}}}"
        if not (sympy.sympify(self.sugar(w)).is_Atom):
            wt = f"({wt})"
        return f"{base}^{'⊗' if self.style == 'unicode' else '@'}{wt}"

    def _register(self, v: sympy.Expr, w: sympy.Expr | None, bra: bool) -> str:
        wrap = self._bra if bra else self._ket
        if w is not None and not P.same(w, 1):
            if v == 0:
                return self._tpow(wrap("0"), w)
            if P.same(v, 2**w - 1):
                return self._tpow(wrap("1"), w)
        return wrap(self.expr(v))

    def _tensor(self) -> str:
        return {"ascii": " @ ", "unicode": " ⊗ ", "plot": r" \otimes "}[self.style]

    def _sum(self, names: list[tuple[str, str, str]], body: str) -> str:
        if self.style == "ascii":
            ranges = ", ".join(f"{n} in {lo}..{hi}" for n, lo, hi in names)
            return f"sum({ranges}){{ {body} }}"
        sigma = "Σ" if self.style == "unicode" else r"\sum"
        return "".join(f"{sigma}_{{{n}={lo}}}^{{{hi}}} " for n, lo, hi in names) + body

    def state(self, s: P.State) -> str:
        """A state as a sum of path-sum terms."""
        parts = [self._term(t, s) for t in s.terms]
        text = parts[0]
        for p in parts[1:]:
            text += f" - {p[1:]}" if p.startswith("-") else f" + {p}"
        return text

    def _term(self, t: P.Term, s: P.State) -> str:
        own = {b.symbol for b in t.bound}
        taken = {x.name for e in (t.amp, *t.kets) for x in e.free_symbols if x not in own}
        taken |= {sym.name for _, sym in self.ctx.sugar} | set(self.ctx.taken)
        mapping: dict[sympy.Basic, sympy.Expr] = {}
        names = []
        for b in t.bound:
            base = b.symbol.name
            name = base
            while name in taken:
                name += "'"
            taken.add(name)
            plain = sympy.Symbol(name, integer=True)
            mapping[b.symbol] = plain
            names.append(
                (
                    name_text(name, self.style),
                    self.expr(b.lo.xreplace(mapping)),
                    self.expr(b.hi.xreplace(mapping)),
                )
            )
        amp = t.amp.xreplace(mapping)
        kets = [k.xreplace(mapping) for k in t.kets]
        syms = [mapping[b.symbol] for b in t.bound]
        scalar, rest = amp.as_independent(*syms, as_Add=False) if syms else (amp, sympy.S.One)
        regs = self._tensor().join(
            self._register(k, w, s.bra) for k, w in zip(kets, s.widths, strict=True)
        )
        body = regs
        if rest != 1:
            body = f"{self._wrap(rest)} {regs}" if regs else self._wrap(rest)
        if names:
            body = self._sum(names, body)
        if not regs and not names:
            return self.expr(amp)
        if scalar == 1:
            return body
        if scalar == -1:
            return "-" + body
        return f"{self._wrap(scalar)} {body}"

    def value(self, v: object) -> str:
        """Any interpreter value."""
        from archytaszx.repl import lang

        if isinstance(v, P.State):
            return self.state(v)
        if isinstance(v, lang.Bits):
            return f"{v} ({int(v)})"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, P.Operator):
            return self.operator(v)
        if isinstance(v, type) and _user(v):
            return self.function(v)
        if isinstance(v, sympy.Basic):
            return self.expr(v)
        return str(v)

    def function(self, f: type) -> str:
        """``f(x) = body``."""
        params = ", ".join(name_text(p.name, self.style) for p in f._params)  # type: ignore[attr-defined]
        return f"{f.__name__}({params}) = {self.expr(f._body)}"  # type: ignore[attr-defined]

    def operator(self, op: P.Operator) -> str:
        """Operator text in input syntax."""
        tens = self._tensor()
        if isinstance(op, P.Gate):
            return op.name
        if isinstance(op, P.Identity):
            return "I" if op.width is None else self._tpow("I", op.width)
        if isinstance(op, P.Power):
            return self._tpow(self._op_atom(op.op), op.count)
        if isinstance(op, P.Tensor):
            return tens.join(self._op_atom(f) for f in op.factors)
        if isinstance(op, P.Compose):
            return " ".join(self._op_atom(f) for f in op.ops)
        if isinstance(op, P.Outer):
            return f"({self.state(op.ket)})({self.state(op.bra)})"
        if isinstance(op, P.Scaled):
            return f"{self._wrap(op.factor)} {self._op_atom(op.op)}"
        if isinstance(op, P.OpSum):
            return " + ".join(self.operator(o) for o in op.ops)
        return repr(op)

    def _op_atom(self, op: P.Operator) -> str:
        text = self.operator(op)
        simple = isinstance(op, (P.Gate, P.Identity, P.Power))
        return text if simple else f"({text})"

    def measured(self, z: sympy.Symbol) -> str:
        """``z in {0,1}^(w)``."""
        width = self.ctx.measured[z][0]
        zt = self.expr(z)
        if P.same(width, 1):
            dom = {"ascii": "{0,1}", "unicode": "{0,1}", "plot": r"\{0,1\}"}[self.style]
        else:
            wt = self.expr(width)
            if self.style == "plot":
                dom = rf"\{{0,1\}}^{{{wt}}}"
            else:
                dom = "{0,1}^" + (wt if sympy.sympify(width).is_Atom else f"({wt})")
        mem = {"ascii": " in ", "unicode": " ∈ ", "plot": r" \in "}[self.style]
        return zt + mem + dom

    def where(self) -> str | None:
        """The ``where`` line for everything rendered so far, or None."""
        items: list[str] = []
        done_syms: set[sympy.Symbol] = set()
        done_funcs: set[type] = set()
        progress = True
        while progress:
            progress = False
            for value, sym in self.ctx.sugar:
                if sym in self.seen_symbols and sym not in done_syms:
                    done_syms.add(sym)
                    items.append(f"{self.expr(sym)} = {self.expr(value, sym)}")
                    progress = True
            for f in list(self.seen_funcs):
                if f not in done_funcs:
                    done_funcs.add(f)
                    items.append(self.function(f))
                    progress = True
            for z, (_, source, facts) in self.ctx.measured.items():
                if z in self.seen_symbols and z not in done_syms:
                    done_syms.add(z)
                    src = self._ket(name_text(source, self.style))
                    word = (
                        r"\ \mathrm{measured\ from}\ "
                        if self.style == "plot"
                        else " measured from "
                    )
                    items.append(f"{self.measured(z)}{word}{src}")
                    items.extend(f"{self.expr(a)} = {self.expr(b)}" for a, b in facts)
                    progress = True
        if not items:
            return None
        word = r"\mathrm{where}\ " if self.style == "plot" else "where "
        return "  " + word + (r",\ " if self.style == "plot" else ", ").join(items)

    def entry(self, label: Label | None, value: object) -> str:
        """``label = value`` or the bare value."""
        if label is not None and label.kind == "func":
            return self.function(value)  # type: ignore[arg-type]
        if (
            label is not None
            and isinstance(value, sympy.Symbol)
            and value in self.ctx.measured
            and value.name == label.name
        ):
            return self.measured(value)
        text = self.value(value)
        return f"{self.label(label)} = {text}" if label is not None else text


Plotter = Callable[[list[str]], "str | None"]
"""Turns mathtext lines into terminal output, or None to fall back to Unicode text."""

PROTOCOLS = ("kitty", "iterm", "sixel", "none")
"""Terminal image protocols; ``ARCHYTASZX_IMAGES`` selects one explicitly."""

_SIXEL_LEVELS = 16


def protocol(environ: Mapping[str, str] | None = None, tty: bool | None = None) -> str:
    """The image protocol of the terminal on standard output, ``none`` when unsupported."""
    env = os.environ if environ is None else environ
    forced = env.get("ARCHYTASZX_IMAGES", "").lower()
    if forced in PROTOCOLS:
        return forced
    if not (sys.stdout.isatty() if tty is None else tty):
        return "none"
    term, program = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    if "KITTY_WINDOW_ID" in env or term in ("xterm-kitty", "xterm-ghostty") or program == "ghostty":
        return "kitty"
    if program in ("iTerm.app", "WezTerm", "vscode") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    if "sixel" in term or term.startswith(("foot", "mlterm")) or program == "mlterm":
        return "sixel"
    return "none"


def dark(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the terminal background is dark: ``ARCHYTASZX_IMAGE_THEME``, then ``COLORFGBG``."""
    env = os.environ if environ is None else environ
    theme = env.get("ARCHYTASZX_IMAGE_THEME", "").lower()
    if theme in ("dark", "light"):
        return theme == "dark"
    bg = env.get("COLORFGBG", "").rpartition(";")[2]
    return not (bg.isdigit() and int(bg) in (7, 15))


def render_png(lines: list[str], *, dark_theme: bool, opaque: bool = False) -> bytes:
    """Off-screen PNG of mathtext lines, one per row, left aligned."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fg, bg = ("#e6e6e6", "#1e1e1e") if dark_theme else ("#111111", "#ffffff")
    fig = Figure(figsize=(1, 0.6 * len(lines) + 0.1), dpi=110)
    FigureCanvasAgg(fig)
    for k, line in enumerate(lines):
        y = 1 - (k + 0.5) / len(lines)
        fig.text(0, y, f"${line.strip()}$", fontsize=13, color=fg, va="center")
    out = io.BytesIO()
    fig.savefig(
        out,
        format="png",
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=not opaque,
        facecolor=bg if opaque else "none",
    )
    return out.getvalue()


def _kitty(png: bytes) -> str:
    data = base64.standard_b64encode(png).decode("ascii")
    chunks = [data[k : k + 4096] for k in range(0, len(data), 4096)] or [""]
    out = []
    for k, chunk in enumerate(chunks):
        more = 1 if k < len(chunks) - 1 else 0
        head = f"a=T,f=100,m={more}" if k == 0 else f"m={more}"
        out.append(f"\x1b_G{head};{chunk}\x1b\\")
    return "".join(out)


def _iterm(png: bytes) -> str:
    data = base64.standard_b64encode(png).decode("ascii")
    return f"\x1b]1337;File=inline=1;size={len(png)};preserveAspectRatio=1:{data}\x07"


def _sixel(png: bytes) -> str:
    """Grey-quantised sixel image."""
    import matplotlib.image as mpimg
    import numpy as np

    rgba = mpimg.imread(io.BytesIO(png), format="png")
    grey = rgba[..., :3].mean(axis=2)
    idx = np.rint(grey * (_SIXEL_LEVELS - 1)).astype(int)
    height, width = idx.shape
    palette = "".join(
        f"#{c};2;{round(100 * c / (_SIXEL_LEVELS - 1))};"
        f"{round(100 * c / (_SIXEL_LEVELS - 1))};{round(100 * c / (_SIXEL_LEVELS - 1))}"
        for c in range(_SIXEL_LEVELS)
    )
    weights = (1 << np.arange(6))[:, None]
    bands = []
    for y0 in range(0, height, 6):
        band = idx[y0 : y0 + 6]
        rows = []
        for c in np.unique(band):
            mask = band == c
            codes = (mask * weights[: len(band)]).sum(axis=0)
            text = []
            for ch, run in itertools.groupby(chr(63 + int(v)) for v in codes):
                n = len(list(run))
                text.append(f"!{n}{ch}" if n > 3 else ch * n)
            rows.append(f"#{c}" + "".join(text))
        bands.append("$".join(rows))
    return f'\x1bPq"1;1;{width};{height}{palette}' + "-".join(bands) + "\x1b\\"


def terminal_image(lines: list[str]) -> str | None:
    """Mathtext lines as an inline terminal image; None when the terminal shows no images."""
    proto = protocol()
    if proto == "none":
        return None
    try:
        png = render_png(lines, dark_theme=dark(), opaque=proto == "sixel")
    except Exception:  # noqa: BLE001
        return None
    return {"kitty": _kitty, "iterm": _iterm, "sixel": _sixel}[proto](png)
