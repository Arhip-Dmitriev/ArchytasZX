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

* ``ascii``: the input syntax (``|0>^@n @ |0>^@m``, ``sum(x in 0..q){ ... }``);
* ``unicode``: ``|0⟩^⊗n ⊗ |0⟩^⊗m``, ``Σ_{x=0}^{q-1} ...``;
* ``plot``: matplotlib mathtext, one line per entry, drawn off-screen and shown inline in the
  terminal (kitty, iTerm2 or sixel image protocol);
* ``latex``: the ``plot`` text as LaTeX source.

A :class:`Renderer` renders entries and records the defined variables and functions they
mention; :meth:`Renderer.where` lists those as a ``where`` line. A measured distribution is
``[(value, probability), ...]``, a branch over free indices with their ranges after a colon.
Defined variables are folded back into expressions (``2^(-n/2)`` shows as ``1/sqrt(q)``
under ``var q = 2^n``). :func:`terminal_image` returns ``plot`` output as an inline image escape
sequence, or None when the terminal has no image protocol.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker
from sympy.printing.latex import LatexPrinter  # type: ignore[import-untyped]
from sympy.printing.str import StrPrinter  # type: ignore[import-untyped]

from archytaszx.repl import terminal as T
from archytaszx.semantics import pathsum as P

STYLES = ("plot", "unicode", "ascii", "latex")
"""The display styles: mathtext images, Unicode text, ASCII input syntax, LaTeX source."""

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


_UNICODE_SOURCE = [
    ("==", "="),
    ("!=", "≠"),
    ("<=", "≤"),
    (">=", "≥"),
    ("*", "·"),
    ("%", "mod"),
    ("@", "⊗"),
    ("sqrt", "√"),
    ("pi", "π"),
]
_LATEX_SOURCE = [
    ("==", "="),
    ("!=", r"\neq "),
    ("<=", r"\leq "),
    (">=", r"\geq "),
    ("*", r"\cdot "),
    ("%", r"\ \mathrm{mod}\ "),
    ("@", r"\otimes "),
    (" and ", r"\ \mathrm{and}\ "),
    (" or ", r"\ \mathrm{or}\ "),
    ("pi", r"\pi "),
]


def _braces(text: str, opener: str, replacement: str) -> str:
    """``opener(...)`` rewritten as ``replacement{...}`` with matched parentheses."""
    out, i = [], 0
    while i < len(text):
        if text.startswith(opener + "(", i):
            depth, j = 0, i + len(opener)
            while j < len(text):
                depth += text[j] == "("
                depth -= text[j] == ")"
                if depth == 0:
                    break
                j += 1
            inner = _braces(text[i + len(opener) + 1 : j], opener, replacement)
            out.append(f"{replacement}{{{inner}}}")
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def source_text(source: str, style: str) -> str:
    """Input text shown in a style: kets, operators and powers in that style's notation."""
    if style == "ascii":
        return source
    if style == "unicode":
        text = re.sub(r"\|([^|<>]*)>", r"|\1⟩", source)
        text = re.sub(r"<([^\s|<>=][^|<>]*)\|", r"⟨\1|", text)
        for a, b in _UNICODE_SOURCE:
            text = text.replace(a, b)
        return text
    text = re.sub(r"\|([^|<>]*)>", r"|\1\\rangle ", source)
    text = re.sub(r"<([^\s|<>=][^|<>]*)\|", r"\\langle \1|", text)
    for a, b in _LATEX_SOURCE:
        text = text.replace(a, b)
    return _braces(_braces(text, "sqrt", r"\sqrt"), "^", "^")


def _repeated_digit(v: sympy.Expr, w: sympy.Expr, d: sympy.Expr) -> sympy.Expr | None:
    """k when the w-digit base-d value v is ``kk...k``, else None."""
    if v == 0:
        return sympy.S.Zero
    if v.is_Integer and w.is_Integer and d.is_Integer:
        found = {(int(v) // int(d) ** p) % int(d) for p in range(int(w))}
        return sympy.Integer(found.pop()) if len(found) == 1 else None
    k = sympy.simplify(v * (d - 1) / (d**w - 1))
    if k.free_symbols & w.free_symbols or not (k.is_Integer or k == d - 1):
        return None
    return k


def _fold_linear(e: sympy.Basic, value: sympy.Basic, sym: sympy.Symbol) -> sympy.Basic:
    """``e`` rewritten through ``sym = c*t + d`` when that removes the term t and shortens e."""
    if not isinstance(e, sympy.Expr) or not isinstance(value, sympy.Add):
        return e
    terms = [x for x in value.args if not x.is_number]
    if len(terms) != 1:
        return e
    c, t = terms[0].as_coeff_Mul()
    if t.is_Symbol or t not in e.atoms(type(t)) or e.is_Atom:
        return e
    d = value - terms[0]
    out = P.expand(e.xreplace({t: (sym - d) / c}))
    if t in out.atoms(type(t)) or sympy.count_ops(out) >= sympy.count_ops(e):
        return e
    return out


def _slice(show: Callable[[sympy.Basic], str], args: tuple[sympy.Basic, ...], name: str) -> str:
    """``z[lo:hi]`` for bits and digits slices, ``z[#k]`` for a mixed-radix register."""
    z = show(args[0])
    if name == "part":
        return f"{z}[#{show(args[1])}]"
    lo, hi = show(args[1]), show(args[2])
    if args[1] == 0:
        return f"{z}[:{hi}]"
    return f"{z}[{lo}:]" if P.same(args[2], args[3]) else f"{z}[{lo}:{hi}]"


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
        c = sympy.factor_terms(P.expand(arg / (2 * sympy.pi * sympy.I)))
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

    def _print_conjugate(self, e: sympy.conjugate) -> str:
        return f"conj({self._print(e.args[0])})"

    def _print_Sum(self, e: sympy.Sum) -> str:
        used = {self._print(x) for x in e.free_symbols}
        body = e.function
        heads = []
        for v, lo, hi in e.limits:
            name = self._print(v)
            while name in used:
                name += "'"
            used.add(name)
            if name != self._print(v):
                new = sympy.Symbol(name, integer=True)
                body = body.xreplace({v: new})
                v = new
            heads.append((name, lo, hi))
        text = self._print(body)
        for name, lo, hi in heads:
            if self.uni:
                atomic = body.is_Atom or isinstance(body, (sympy.Function, sympy.Pow))
                inner = text if atomic and len(heads) == 1 else f"({text})"
                text = f"Σ_{{{name}={self._print(lo)}}}^{{{self._print(hi)}}} {inner}"
            else:
                text = f"sum({name} in {self._print(lo)}..={self._print(hi)}){{ {text} }}"
            body = sympy.Symbol("_")
        return str(text)

    def _print_Equality(self, e: sympy.Eq) -> str:
        return f"{self._print(e.lhs)} = {self._print(e.rhs)}"

    def _print_Unequality(self, e: sympy.Ne) -> str:
        return f"{self._print(e.lhs)} {'≠' if self.uni else '!='} {self._print(e.rhs)}"

    def _print_Piecewise(self, e: sympy.Piecewise) -> str:
        parts = [
            f"{self._print(v)} otherwise"
            if c is sympy.true
            else f"{self._print(v)} if {self._print(c)}"
            for v, c in e.args
        ]
        return "{" + "; ".join(parts) + "}"

    def _print_Function(self, e: sympy.Function) -> str:
        name = type(e).__name__
        args = e.args
        if name == "xor":
            return f"({self._print(args[0])} {'⊕' if self.uni else 'xor'} {self._print(args[1])})"
        if name in ("bits", "digits", "part"):
            return _slice(self._print, args, name)
        if name == "shr":
            if args[2] == 2:
                return f"({self._print(args[0])} >> {self._print(args[1])})"
            return f"floor({self._print(args[0])}/{self._paren(args[2] ** args[1])})"
        if name == "igcd":
            return f"gcd({self._print(args[0])}, {self._print(args[1])})"
        if name == "pop":
            return f"popcount({self._print(args[0])})"
        if name == "dsum":
            return f"dsum({self._print(args[0])})"
        if name in ("bitdot", "ddot"):
            return f"{self._paren(args[0])}{'⊙' if self.uni else '.'}{self._paren(args[1])}"
        if name == "dshift":
            op = "⊞" if self.uni else "++"
            return f"({self._print(args[0])} {op} {self._print(args[1])})"
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
        c = sympy.factor_terms(P.expand(arg / (2 * sympy.pi * sympy.I)))
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
        if name in ("bits", "digits", "part"):
            return _slice(self._print, args, name)
        if name == "shr":
            if args[2] == 2:
                return rf"({self._print(args[0])} \gg {self._print(args[1])})"
            return rf"\lfloor {self._print(args[0])}/{self._print(args[2] ** args[1])} \rfloor"
        if name == "igcd":
            return rf"\gcd({self._print(args[0])}, {self._print(args[1])})"
        if name in ("pop", "dsum"):
            return rf"\mathrm{{{name}}}({self._print(args[0])})"
        if name in ("bitdot", "ddot"):
            return rf"{self._print(args[0])} \odot {self._print(args[1])}"
        if name == "dshift":
            return rf"({self._print(args[0])} \boxplus {self._print(args[1])})"
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
    """How an entry is named: ``kind`` is ket, bra, var, func, op or list; ``dim`` is shown as
    ``_dim`` on a qudit state."""

    kind: str
    name: str
    dim: sympy.Expr | None = None


@dataclass(frozen=True)
class Text:
    """A literal line of text."""

    text: str


@dataclass(frozen=True)
class Action:
    """An operator shown as the image of a basis ket of symbolic value."""

    probe: P.State
    image: P.State


@dataclass
class Context:
    """Interpreter facts a renderer consults."""

    sugar: Sequence[tuple[sympy.Expr, sympy.Symbol]] = ()
    taken: frozenset[str] = frozenset()
    assumptions: tuple[sympy.Basic, ...] = ()
    order: dict[sympy.Symbol, int] = field(default_factory=dict)


class Renderer:
    """Renders entries in one style and collects the names they mention."""

    def __init__(self, style: str, context: Context | None = None) -> None:
        if style not in STYLES:
            raise ValueError(f"unknown style {style!r}")
        self.style = "plot" if style == "latex" else style
        self.ctx = context or Context()
        self._text = _Latex() if self.style == "plot" else _Text(style == "unicode")
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
        """``e`` with defined variables folded back in: plain substitutions, then the shortest
        linear rewrite. With ``skip``, only variables defined before ``skip`` are used."""
        order = self.ctx.order
        limit = order.get(skip, len(order)) if skip is not None else len(order) + 1
        items = [(v, s) for v, s in self.ctx.sugar if s != skip and order.get(s, 0) < limit]
        normal = []
        for value, sym in items:
            for other, osym in items:
                if osym != sym:
                    value = value.subs(other, osym)
            normal.append((value, sym))
        for _ in range(4):
            before = e
            for (value, sym), (norm, _) in zip(items, normal, strict=True):
                e = e.subs(value, sym).subs(norm, sym)
            if e == before:
                break
        best = e
        for value, sym in normal:
            cand = _fold_linear(e, value, sym)
            if cand is not e and sympy.count_ops(cand) < sympy.count_ops(best):
                best = cand
        return best

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
        if label.kind == "expr":
            return source_text(label.name, self.style)
        name = name_text(label.name, self.style)
        if label.kind == "ket":
            return self._ket(name) + self._dim(label.dim)
        if label.kind == "bra":
            return self._bra(name) + self._dim(label.dim)
        return name

    def _dim(self, d: sympy.Expr | None) -> str:
        if d is None or P.same(d, 2):
            return ""
        text = self.expr(d)
        if self.style == "plot":
            return f"_{{{text}}}"
        return f"_{text}" if sympy.sympify(self.sugar(d)).is_Atom else f"_({text})"

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

    def _register(
        self, v: sympy.Expr, w: sympy.Expr | None, d: sympy.Expr | None, bra: bool
    ) -> str:
        suffix = self._dim(d)

        def wrap(text: str) -> str:
            return (self._bra(text) if bra else self._ket(text)) + suffix

        if w is not None and d is not None and not P.same(w, 1):
            digit = _repeated_digit(v, w, d)
            if digit is not None:
                return self._tpow(wrap(self.expr(digit)), w)
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

    def mixed(self, m: P.Mixed) -> str:
        """A density matrix as a sum of path-sum terms of ``|k><b|`` per register."""
        k = m.count
        parts = [self._term(t, m.vec, pairs=k) for t in m.vec.terms]
        text = parts[0]
        for p in parts[1:]:
            text += f" - {p[1:]}" if p.startswith("-") else f" + {p}"
        return text

    def _names(
        self, bound: Sequence[P.Bound], taken: set[str]
    ) -> tuple[dict[sympy.Basic, sympy.Expr], list[tuple[str, str, str]]]:
        """Display symbols for summation indices and their ``(name, lo, hi)`` texts."""
        taken |= {sym.name for _, sym in self.ctx.sugar} | set(self.ctx.taken)
        mapping: dict[sympy.Basic, sympy.Expr] = {}
        names = []
        for b in bound:
            name = b.symbol.name
            while name in taken:
                name += "'"
            taken.add(name)
            mapping[b.symbol] = sympy.Symbol(name, integer=True)
            names.append(
                (
                    name_text(name, self.style),
                    self.expr(b.lo.xreplace(mapping)),
                    self.expr(b.hi.xreplace(mapping) + int(self.style == "ascii")),
                )
            )
        return mapping, names

    def probability(self, p: sympy.Expr) -> str:
        """A probability, a fraction with a terminating decimal expansion as that decimal."""
        if p.is_Rational and not p.is_Integer:
            q = int(p.q)
            while q % 2 == 0:
                q //= 2
            while q % 5 == 0:
                q //= 5
            if q == 1:
                digits = len(str(int(p.q))) + 2
                text = format(sympy.Float(p, digits), "f").rstrip("0")
                return text
        return self.expr(p)

    def distribution(self, d: P.Distribution) -> str:
        """``[(value, p), ...]``; a branch over free indices as ``[(v(k), p(k)) : k in ...]``."""
        items = []
        for br in d.branches:
            taken = {x.name for x in (br.value.free_symbols | br.prob.free_symbols)}
            taken -= {b.symbol.name for b in br.free}
            mapping, names = self._names(br.free, taken)
            value = self.expr(sympy.sympify(br.value).xreplace(mapping))
            prob = self.probability(sympy.sympify(br.prob).xreplace(mapping))
            pair = f"({value}, {prob})"
            if names:
                ranges = ", ".join(self._range(n, lo, hi) for n, lo, hi in names)
                colon = r" \colon " if self.style == "plot" else " : "
                pair = f"{pair}{colon}{ranges}"
            items.append(pair)
        if self.style == "plot":
            return r"\left[" + ", ".join(items) + r"\right]"
        return "[" + ", ".join(items) + "]"

    def _range(self, name: str, lo: str, hi: str) -> str:
        if self.style == "ascii":
            return f"{name} in {lo}..{hi}"
        if self.style == "plot":
            if (lo, hi) == ("0", "1"):
                return rf"{name} \in \{{0, 1\}}"
            return rf"{name} \in \{{{lo}, \ldots, {hi}\}}"
        if (lo, hi) == ("0", "1"):
            return f"{name} ∈ {{0,1}}"
        return f"{name} ∈ {{{lo},…,{hi}}}"

    def _term(self, t: P.Term, s: P.State, pairs: int | None = None) -> str:
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
                    self.expr(b.hi.xreplace(mapping) + int(self.style == "ascii")),
                )
            )
        amp = t.amp.xreplace(mapping)
        kets = [k.xreplace(mapping) for k in t.kets]
        syms = [mapping[b.symbol] for b in t.bound]
        scalar, rest = amp.as_independent(*syms, as_Add=False) if syms else (amp, sympy.S.One)
        if pairs is None:
            regs = self._tensor().join(
                self._register(k, w, d, s.bra)
                for k, w, d in zip(kets, s.widths, s.dims, strict=True)
            )
        else:
            regs = self._tensor().join(
                self._register(kets[j], s.widths[j], s.dims[j], False)
                + self._register(kets[j + pairs], s.widths[j], s.dims[j], True)
                for j in range(pairs)
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
        if isinstance(v, Text):
            if self.style != "plot":
                return v.text
            escaped = re.sub(r"([#$%&_{}])", r"\\\1", v.text).replace(" ", r"\ ")
            return rf"\mathrm{{{escaped}}}"
        if isinstance(v, P.State):
            return self.state(v)
        if isinstance(v, P.Mixed):
            return self.mixed(v)
        if isinstance(v, P.Distribution):
            return self.distribution(v)
        if isinstance(v, (bool, sympy.logic.boolalg.BooleanAtom)):
            return "true" if v else "false"
        if isinstance(v, P.Operator):
            return self.operator(v)
        if isinstance(v, Action):
            arrow = {"plot": r"\mapsto", "latex": r"\mapsto", "unicode": "↦"}.get(self.style, "->")
            return f"{self.state(v.probe)} {arrow} {self.state(v.image)}"
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
        if not items:
            return None
        word = r"\mathrm{where}\ " if self.style == "plot" else "where "
        return "  " + word + (r",\ " if self.style == "plot" else ", ").join(items)

    def entry(self, label: Label | None, value: object, record: bool = True) -> str:
        """``label = value`` or the bare value; with ``record=False`` its names stay out of
        :meth:`where`."""
        if not record:
            symbols, funcs = list(self.seen_symbols), list(self.seen_funcs)
            try:
                return self.entry(label, value)
            finally:
                self.seen_symbols, self.seen_funcs = symbols, funcs
        if label is not None and label.kind == "func":
            return self.function(value)  # type: ignore[arg-type]
        if label is not None and label.kind == "var" and isinstance(value, sympy.Basic):
            if isinstance(value, sympy.Symbol) and value.name == label.name:
                word = r"\ (\mathrm{symbolic})" if self.style == "plot" else " (symbolic)"
                return f"{self.label(label)}{word}"
            return f"{self.label(label)} = {self.expr(value, sympy.Symbol(label.name))}"
        if isinstance(value, sympy.logic.boolalg.Boolean) and not isinstance(
            value, sympy.logic.boolalg.BooleanAtom
        ):
            status = "assumed" if any(value == a for a in self.ctx.assumptions) else "open"
            self._record(self.sugar(value))
            shown = self.label(label) if label is not None else self.expr(value)
            colon = r"\colon\ " if self.style == "plot" else ": "
            return f"{shown}{colon}{status}"
        text = self.value(value)
        if label is None:
            return text
        if isinstance(value, P.Mixed) and label.kind == "ket":
            name = name_text(label.name, self.style)
            return f"{self._ket(name)}{self._bra(name)}{self._dim(label.dim)} = {text}"
        if isinstance(value, (bool, sympy.logic.boolalg.BooleanAtom)):
            colon = r"\colon\ " if self.style == "plot" else ": "
            return f"{self.label(label)}{colon}{text}"
        return f"{self.label(label)} = {text}"


Plotter = Callable[[list[str]], "str | None"]
"""Turns mathtext lines into terminal output, or None to fall back to Unicode text."""


def terminal_image(lines: list[str]) -> str | None:
    """Mathtext lines as an inline terminal image; None when the terminal shows no images."""
    proto = T.protocol()
    if proto == "none" and T.SAVE_DIR is None:
        return None
    density = T._DENSITY if proto == "none" or T.scalable(proto) else 1
    try:
        if T.SAVE_WHITE:
            T.keep(
                T.render_png(
                    lines, dark_theme=False, opaque=True, density=T._DENSITY, palette=T.WHITE
                ),
                "dirac",
            )
            if proto == "none":
                return None
            png = T.render_png(lines, dark_theme=T.dark(), opaque=proto == "sixel", density=density)
        else:
            opaque = proto == "sixel" or T.SAVE_DIR is not None
            png = T.render_png(lines, dark_theme=T.dark(), opaque=opaque, density=density)
            T.keep(png, "dirac")
    except Exception:  # noqa: BLE001
        return None
    return T.encode_image(png, proto, density)
