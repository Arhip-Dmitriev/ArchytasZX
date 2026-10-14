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

"""The Dirac input language: lexer, parser and :class:`Interpreter`.

Statements (newline or ``;`` separated, ``#`` comments)::

    var n, m                  var q = 2^n          var t: real
    func f(x) = a^x % N
    oracle U = |x>|y> -> |x>|y xor f(x)>
    gate QFT = |x> -> 1/sqrt(q) sum(y in 0..q-1){ e^(2*pi*i*x*y/q) |y> }
    |psi> = (H^@n @ I)|0>^@n @ |0>^@m          <phi| = ...       |k : n>
    var z = measure|psi>      measure|psi>
    list L = {|psi>, <phi|, z}
    show(L)  show(|psi>, ascii)                seed 7

Every state is named ``|name>``; ``<name|`` is its conjugate transpose. Classical values are
declared with ``var`` (types ``nat`` (default), ``int``, ``real``, ``complex``); measurement
results are booleans (one qubit) or :class:`Bits`, both usable as numbers. ``|v : w>`` gives
the basis ket v a register of width w.

Expression precedence, loosest first: ``xor``; ``+ -``; ``@``; juxtaposition (operator
application, scalar multiplication, ``|a>|b>`` tensor, ``<a|b>``, ``|a><b|``; grouped from
the right, so ``U |x>|y>`` applies U to both kets); ``* / %``;
unary ``-``; ``^`` (right associative) and postfix ``^@``. Constants ``pi``, ``e``, ``i``;
functions ``sqrt exp abs floor ceil`` and user ``func`` s. Built-in gates ``H X Y Z S T
CNOT SWAP`` (one-qubit registers; ``H^@n`` for a register) and the identity ``I`` (absorbs
the remaining registers when last in a tensor). Built-in states ``|+>``, ``|->``.

Only ``show`` prints. Values are symbolic exactly when their inputs are.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.repl import display as D
from archytaszx.semantics import pathsum as P


class LangError(Exception):
    """A statement that cannot be executed."""


class LangSyntaxError(LangError):
    """Text that does not parse."""


class LangTypeError(LangError):
    """A value used where its type is not allowed."""


@dataclass(frozen=True)
class Bits:
    """A measured bitstring; acts as its integer value."""

    value: int
    width: int

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return format(self.value, f"0{self.width}b")


@dataclass(frozen=True)
class Tok:
    kind: str
    text: str
    pos: int


_SIMPLE = re.compile(
    r"(?P<ws>\s+)|(?P<num>\d+(?:\.\d+)?)|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>->|\.\.|\^@|[-+*/%^@(){}\[\],=:])"
)
_BRAKET_TAIL = re.compile(r"([^\s|<>][^|<>]*)>")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(src: str) -> list[Tok]:
    """Tokens of one statement; ``|..>`` and ``<..|`` are single tokens."""
    out: list[Tok] = []
    i = 0
    while i < len(src):
        c = src[i]
        if c == "|":
            j = src.find(">", i + 1)
            if j < 0 or "|" in src[i + 1 : j] or "<" in src[i + 1 : j]:
                raise LangSyntaxError(f"unclosed ket at column {i + 1}")
            out.append(Tok("ket", src[i + 1 : j], i))
            i = j + 1
            continue
        if c == "<":
            j = src.find("|", i + 1)
            if j < 0 or ">" in src[i + 1 : j]:
                raise LangSyntaxError(f"unclosed bra at column {i + 1}")
            out.append(Tok("bra", src[i + 1 : j], i))
            i = j + 1
            tail = _BRAKET_TAIL.match(src, i)
            if tail:
                out.append(Tok("ket", tail.group(1), i))
                i = tail.end()
            continue
        m = _SIMPLE.match(src, i)
        if not m:
            raise LangSyntaxError(f"unexpected {c!r} at column {i + 1}")
        if m.lastgroup != "ws":
            out.append(Tok(m.lastgroup or "", m.group(), i))
        i = m.end()
    return out


@dataclass(frozen=True)
class Num:
    value: sympy.Expr


@dataclass(frozen=True)
class Name:
    id: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[object, ...]


@dataclass(frozen=True)
class Bin:
    op: str
    left: object
    right: object


@dataclass(frozen=True)
class Neg:
    arg: object


@dataclass(frozen=True)
class Juxt:
    left: object
    right: object


@dataclass(frozen=True)
class Braket:
    """``|content>`` (``bra=False``) or ``<content|``."""

    content: str
    bra: bool


@dataclass(frozen=True)
class SumNode:
    var: str
    lo: object
    hi: object
    body: object


_KEYWORDS = {"xor", "in"}
_STARTERS = {"num", "ket", "bra"}


class Parser:
    """Recursive-descent expression parser over :func:`tokenize` tokens."""

    def __init__(self, toks: list[Tok], is_func: Callable[[str], bool]) -> None:
        self.toks = toks
        self.i = 0
        self.is_func = is_func

    def peek(self, k: int = 0) -> Tok | None:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def at(self, text: str) -> bool:
        t = self.peek()
        return t is not None and t.kind in ("op", "ident") and t.text == text

    def take(self, text: str | None = None) -> Tok:
        t = self.peek()
        if t is None:
            raise LangSyntaxError(
                f"expected {text!r}, got end of input" if text else "unexpected end"
            )
        if text is not None and not (t.kind in ("op", "ident") and t.text == text):
            raise LangSyntaxError(f"expected {text!r} at column {t.pos + 1}, got {t.text!r}")
        self.i += 1
        return t

    def done(self) -> bool:
        return self.i >= len(self.toks)

    def expr(self) -> object:
        node = self.add()
        while self.at("xor"):
            self.take()
            node = Bin("xor", node, self.add())
        return node

    def add(self) -> object:
        node = self.tensor()
        while self.at("+") or self.at("-"):
            op = self.take().text
            node = Bin(op, node, self.tensor())
        return node

    def tensor(self) -> object:
        node = self.juxt()
        while self.at("@"):
            self.take()
            node = Bin("@", node, self.juxt())
        return node

    def _starts_operand(self) -> bool:
        t = self.peek()
        if t is None:
            return False
        if t.kind in _STARTERS:
            return True
        if t.kind == "ident":
            return t.text not in _KEYWORDS
        return t.kind == "op" and t.text == "("

    def juxt(self) -> object:
        nodes = [self.mul()]
        while self._starts_operand():
            nodes.append(self.mul())
        node = nodes.pop()
        while nodes:
            node = Juxt(nodes.pop(), node)
        return node

    def mul(self) -> object:
        node = self.unary()
        while self.at("*") or self.at("/") or self.at("%"):
            op = self.take().text
            node = Bin(op, node, self.unary())
        return node

    def unary(self) -> object:
        if self.at("-"):
            self.take()
            return Neg(self.unary())
        if self.at("+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self) -> object:
        node = self.postfix()
        if self.at("^"):
            self.take()
            return Bin("^", node, self.unary())
        return node

    def postfix(self) -> object:
        node = self.atom()
        while self.at("^@"):
            self.take()
            node = Bin("^@", node, self.atom())
        return node

    def atom(self) -> object:
        t = self.peek()
        if t is None:
            raise LangSyntaxError("unexpected end of input")
        if t.kind == "num":
            self.i += 1
            return Num(sympy.Rational(t.text) if "." in t.text else sympy.Integer(t.text))
        if t.kind in ("ket", "bra"):
            self.i += 1
            return Braket(t.text, t.kind == "bra")
        if t.kind == "ident":
            if t.text == "sum":
                return self.sum()
            self.i += 1
            if self.at("(") and self.is_func(t.text):
                self.take("(")
                args = [] if self.at(")") else self.args()
                self.take(")")
                return Call(t.text, tuple(args))
            return Name(t.text)
        if t.kind == "op" and t.text == "(":
            self.take("(")
            node = self.expr()
            self.take(")")
            return node
        raise LangSyntaxError(f"unexpected {t.text!r} at column {t.pos + 1}")

    def args(self) -> list[object]:
        out = [self.expr()]
        while self.at(","):
            self.take()
            out.append(self.expr())
        return out

    def sum(self) -> SumNode:
        self.take("sum")
        self.take("(")
        var = self.take()
        if var.kind != "ident":
            raise LangSyntaxError(f"expected an index name at column {var.pos + 1}")
        self.take("in")
        lo = self.add()
        self.take("..")
        hi = self.add()
        self.take(")")
        self.take("{")
        body = self.expr()
        self.take("}")
        return SumNode(var.text, lo, hi, body)


@dataclass(frozen=True)
class Local:
    """A pattern variable, function parameter or summation index in scope."""

    value: sympy.Expr
    width: sympy.Expr | None = None
    bound: P.Bound | None = None


@dataclass
class Ctx:
    locals: dict[str, Local] = field(default_factory=dict)
    body: bool = False

    def child(self, name: str, local: Local) -> Ctx:
        return Ctx({**self.locals, name: local}, self.body)


_TYPES = {
    "nat": {"integer": True, "nonnegative": True},
    "int": {"integer": True},
    "real": {"real": True},
    "complex": {},
}
_FUNCS: dict[str, Callable[..., sympy.Expr]] = {
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
}
_CONSTANTS = {"pi": sympy.pi, "e": sympy.E, "i": sympy.I}
_STATEMENTS = {"var", "func", "oracle", "gate", "list", "show", "measure", "seed"}
_RESERVED = _STATEMENTS | _KEYWORDS | set(_FUNCS) | set(_CONSTANTS) | {"sum"} | set(D.STYLES)

PRELUDE = """
gate H = |x> -> 1/sqrt(2) sum(y in 0..1){ (-1)^(x*y) |y> }
gate X = |x> -> |1 - x>
gate Y = |x> -> i*(-1)^x |1 - x>
gate Z = |x> -> (-1)^x |x>
gate S = |x> -> i^x |x>
gate T = |x> -> e^(i*pi*x/4) |x>
gate CNOT = |c>|t> -> |c>|t xor c>
gate SWAP = |a>|b> -> |b>|a>
"""
"""Built-in gates, defined in the language itself."""


def _split_statements(source: str) -> Iterator[str]:
    for line in source.splitlines():
        line = line.split("#", 1)[0]
        depth = 0
        start = 0
        for k, c in enumerate(line):
            depth += c in "({["
            depth -= c in ")}]"
            if c == ";" and depth == 0:
                yield line[start:k]
                start = k + 1
        yield line[start:]


def _range_width(lo: sympy.Expr, hi: sympy.Expr) -> sympy.Expr:
    if lo != 0:
        if hi.is_Integer:
            return sympy.Integer(max(1, int(hi).bit_length()))
        raise LangTypeError("a ket over a symbolic range needs the range to start at 0")
    count = sympy.expand(hi + 1)
    if count.is_Integer:
        return sympy.Integer(max(1, (int(count) - 1).bit_length()))
    w = sympy.simplify(sympy.expand_log(sympy.log(count, 2), force=True))
    if w.has(sympy.log):
        raise LangTypeError(f"cannot infer a register width from the range 0..{hi}")
    return w


def make_function(name: str, params: list[sympy.Dummy], body: sympy.Expr) -> type:
    """A sympy function class for ``name(params) = body``, evaluated on integer arguments."""
    free = body.free_symbols - set(params)
    fast = P.to_powmod(body)

    def _eval(cls: type, *args: sympy.Expr) -> sympy.Expr | None:
        if free or not all(a.is_Integer for a in args):
            return None
        return fast.xreplace(dict(zip(params, args, strict=True)))

    def _numeric(*args: object) -> object:
        fn = sympy.lambdify(params, fast, modules=[P.numeric_impls(fast), "numpy"])
        return fn(*(np.asarray(a, dtype=object) for a in args))

    cls = type(name, (sympy.Function,), {"eval": classmethod(_eval), "nargs": len(params)})
    cls._params = tuple(params)  # type: ignore[attr-defined]
    cls._body = body  # type: ignore[attr-defined]
    cls._free = free  # type: ignore[attr-defined]
    cls._numeric = staticmethod(_numeric)  # type: ignore[attr-defined]
    return cls


class Interpreter:
    """Executes statements against named states, variables, functions and operators."""

    def __init__(
        self, *, style: str = "plot", seed: int | None = None, plotter: D.Plotter | None = None
    ) -> None:
        if style not in D.STYLES:
            raise ValueError(f"unknown display style {style!r}")
        self.style = style
        self.plotter = plotter if plotter is not None else D.terminal_image
        self.rng = np.random.default_rng(seed)
        self.values: dict[str, object] = {}
        self.sugar: list[tuple[str, sympy.Expr]] = []
        self.funcs: dict[str, type] = {}
        self.ops: dict[str, P.Operator] = {"I": P.Identity()}
        self.states: dict[str, P.State] = {}
        self.lists: dict[str, list[tuple[D.Label | None, object]]] = {}
        self.measured: dict[
            sympy.Symbol, tuple[sympy.Expr, str, tuple[tuple[sympy.Expr, sympy.Expr], ...]]
        ] = {}
        self._count = 0
        for stmt in _split_statements(PRELUDE):
            if stmt.strip():
                self._statement(stmt.strip(), builtin=True)
        self.builtins = set(self.ops)

    def execute(self, source: str) -> str:
        """Runs every statement of ``source``; returns what ``show`` printed."""
        out: list[str] = []
        for stmt in _split_statements(source):
            text = stmt.strip()
            if text:
                printed = self._statement(text)
                if printed:
                    out.append(printed)
        return "\n".join(out)

    def _parser(self, toks: list[Tok]) -> Parser:
        return Parser(toks, lambda name: name in _FUNCS or name in self.funcs)

    def _statement(self, text: str, builtin: bool = False) -> str:
        toks = tokenize(text)
        head = toks[0]
        if head.kind == "ident" and head.text in _STATEMENTS:
            return getattr(self, f"_stmt_{head.text}")(toks, text, builtin) or ""
        if head.kind in ("ket", "bra") and len(toks) > 1 and toks[1].text == "=":
            self._assign_state(head, toks[2:])
            return ""
        raise LangSyntaxError(
            "a statement starts with var, func, oracle, gate, list, show, measure, seed, "
            "|name> = or <name| ="
        )

    def _check_name(self, name: str, kind: str) -> None:
        if not _IDENT.fullmatch(name):
            raise LangSyntaxError(f"{name!r} is not a name")
        if name in _RESERVED:
            raise LangSyntaxError(f"{name!r} is reserved")
        tables: dict[str, Mapping[str, object]] = {
            "var": self.values,
            "func": self.funcs,
            "op": self.ops,
            "state": self.states,
            "list": self.lists,
        }
        for k, table in tables.items():
            if k != kind and name in table:
                raise LangTypeError(f"{name!r} is already a {k}")
        if kind == "op" and name in getattr(self, "builtins", ()):
            raise LangTypeError(f"{name!r} is a built-in operator")

    def _stmt_var(self, toks: list[Tok], text: str, builtin: bool) -> None:
        p = self._parser(toks[1:])
        while True:
            t = p.take()
            if t.kind != "ident":
                raise LangSyntaxError(f"expected a variable name at column {t.pos + 1}")
            name = t.text
            self._check_name(name, "var")
            kind = "nat"
            if p.at(":"):
                p.take()
                kind = p.take().text
                if kind not in _TYPES:
                    raise LangTypeError(f"unknown type {kind!r}; use {', '.join(_TYPES)}")
            if p.at("="):
                p.take()
                if p.at("measure"):
                    p.take()
                    self._measure(p, name)
                else:
                    self._define(name, p.expr())
            else:
                self._forget(name)
                self.values[name] = sympy.Symbol(name, **_TYPES[kind])
            if p.done():
                return
            p.take(",")

    def _forget(self, name: str) -> None:
        self.sugar = [(n, v) for n, v in self.sugar if n != name]

    def _define(self, name: str, node: object) -> None:
        value = self._scalar(self._eval(node, Ctx()), f"var {name}")
        self._forget(name)
        self.values[name] = value
        if value.free_symbols and not value.is_Symbol:
            self.sugar.append((name, value))

    def _measure(self, p: Parser, name: str | None) -> None:
        t = p.take()
        if t.kind != "ket" or not _IDENT.fullmatch(t.text.strip()):
            raise LangSyntaxError("measure takes a named ket, e.g. measure|psi>")
        label = t.text.strip()
        if label not in self.states:
            raise LangTypeError(f"|{label}> is not defined")
        if not p.done() and not p.at(","):
            raise LangSyntaxError("measure takes exactly one named ket")
        if name is None:
            self._count += 1
            sym_name = f"m{self._count}"
        else:
            sym_name = name
        symbol = sympy.Symbol(sym_name, integer=True, nonnegative=True)
        facts = [f for info in self.measured.values() for f in info[2]]
        try:
            m = P.measure(self.states[label], symbol, self.rng, facts)
        except P.PathSumError as exc:
            raise LangError(str(exc)) from exc
        self.states[label] = m.post
        if name is None:
            return
        self._forget(name)
        outcome = m.outcome
        if outcome.is_Integer and m.width.is_Integer:
            self.values[name] = bool(outcome) if m.width == 1 else Bits(int(outcome), int(m.width))
            return
        self.values[name] = outcome
        if outcome == symbol:
            self.measured[symbol] = (m.width, label, m.constraints)

    def _stmt_measure(self, toks: list[Tok], text: str, builtin: bool) -> None:
        p = self._parser(toks[1:])
        self._measure(p, None)

    def _stmt_seed(self, toks: list[Tok], text: str, builtin: bool) -> None:
        if len(toks) != 2 or toks[1].kind != "num" or "." in toks[1].text:
            raise LangSyntaxError("usage: seed INT")
        self.rng = np.random.default_rng(int(toks[1].text))

    def _stmt_func(self, toks: list[Tok], text: str, builtin: bool) -> None:
        p = self._parser(toks[1:])
        name = p.take().text
        self._check_name(name, "func")
        p.take("(")
        params: list[str] = []
        while not p.at(")"):
            params.append(p.take().text)
            if not p.at(")"):
                p.take(",")
        p.take(")")
        p.take("=")
        dummies = [sympy.Dummy(q, integer=True, nonnegative=True) for q in params]
        ctx = Ctx({q: Local(d) for q, d in zip(params, dummies, strict=True)})
        self.funcs[name] = make_function(name, dummies, sympy.Integer(0))
        try:
            body = self._scalar(self._eval(p.expr(), ctx), f"func {name}")
        except Exception:
            del self.funcs[name]
            raise
        if not p.done():
            raise LangSyntaxError(f"unexpected {p.peek().text!r} in func {name}")  # type: ignore[union-attr]
        self.funcs[name] = make_function(name, dummies, body)

    def _stmt_gate(self, toks: list[Tok], text: str, builtin: bool, oracle: bool = False) -> None:
        p = self._parser(toks[1:])
        name = p.take().text
        if not builtin:
            self._check_name(name, "op")
        p.take("=")
        pattern = p.expr()
        p.take("->")
        body = p.expr()
        if not p.done():
            raise LangSyntaxError(f"unexpected {p.peek().text!r} in {name}")  # type: ignore[union-attr]
        names = self._pattern(pattern)
        source = text.split("=", 1)[1].strip()

        def fn(vals: tuple[sympy.Expr, ...], widths: tuple[sympy.Expr, ...]) -> P.State:
            ctx = Ctx({q: Local(v, w) for q, v, w in zip(names, vals, widths, strict=True)}, True)
            res = self._eval(body, ctx)
            if not isinstance(res, P.State) or res.bra:
                raise LangTypeError(f"the body of {name} must be a ket")
            return self._fix_widths(res, widths)

        self.ops[name] = P.Gate(name, len(names), fn, qubit=builtin, oracle=oracle, source=source)

    def _stmt_oracle(self, toks: list[Tok], text: str, builtin: bool) -> None:
        self._stmt_gate(toks, text, builtin, oracle=True)

    def _pattern(self, node: object) -> list[str]:
        if isinstance(node, Braket) and not node.bra:
            content = node.content.strip()
            if not _IDENT.fullmatch(content) or content in _RESERVED:
                raise LangSyntaxError(f"a pattern ket holds one variable name, got |{content}>")
            return [content]
        if isinstance(node, Juxt) or (isinstance(node, Bin) and node.op == "@"):
            names = self._pattern(node.left) + self._pattern(node.right)
            if len(set(names)) != len(names):
                raise LangSyntaxError("pattern variables must be distinct")
            return names
        raise LangSyntaxError("a gate pattern is a product of kets, e.g. |x>|y>")

    def _fix_widths(self, s: P.State, defaults: tuple[sympy.Expr | None, ...]) -> P.State:
        if all(w is not None for w in s.widths):
            return s
        widths = list(s.widths)
        for k, w in enumerate(widths):
            if w is not None:
                continue
            if len(widths) == len(defaults):
                widths[k] = defaults[k]
                continue
            values = [t.kets[k] for t in s.terms]
            if all(v.is_Integer for v in values):
                widths[k] = sympy.Integer(max(1, max(int(v) for v in values).bit_length()))
            else:
                raise LangTypeError("cannot infer an output register width")
        return P.State(tuple(widths), s.terms, s.bra)

    def _stmt_list(self, toks: list[Tok], text: str, builtin: bool) -> None:
        p = self._parser(toks[1:])
        name = p.take().text
        self._check_name(name, "list")
        p.take("=")
        p.take("{")
        items: list[tuple[D.Label | None, object]] = []
        while not p.at("}"):
            items.extend(self._entries(p.expr()))
            if not p.at("}"):
                p.take(",")
        p.take("}")
        if not p.done():
            raise LangSyntaxError("unexpected text after the list")
        self.lists[name] = items

    def _stmt_show(self, toks: list[Tok], text: str, builtin: bool) -> str:
        p = self._parser(toks[1:])
        paren = p.at("(")
        if paren:
            p.take("(")
        nodes = [] if p.done() or p.at(")") else p.args()
        if paren:
            p.take(")")
        if not p.done():
            raise LangSyntaxError("unexpected text after show(...)")
        style = self.style
        if nodes and isinstance(nodes[-1], Name) and nodes[-1].id in D.STYLES:
            style = nodes.pop().id  # type: ignore[attr-defined]
        entries: list[tuple[D.Label | None, object]] = []
        for node in nodes:
            entries.extend(self._entries(node))
        return self._render(entries, style)

    def _context(self) -> D.Context:
        sugar = sorted(
            ((v, sympy.Symbol(n)) for n, v in self.sugar),
            key=lambda pair: -sympy.count_ops(pair[0]),
        )
        taken = frozenset(self.values) | frozenset(self.states) | frozenset(self.funcs)
        return D.Context(sugar, self.measured, taken | frozenset(self.ops))

    def _render(self, entries: list[tuple[D.Label | None, object]], style: str) -> str:
        if style == "plot":
            r = D.Renderer("plot", self._context())
            lines = [r.entry(lab, v) for lab, v in entries]
            where = r.where()
            image = self.plotter(lines + ([where] if where else []))
            if image is not None:
                return image
            style = "unicode"
        r = D.Renderer(style, self._context())
        lines = [r.entry(lab, v) for lab, v in entries]
        where = r.where()
        return "\n".join(lines + ([where] if where else []))

    def _entries(self, node: object) -> list[tuple[D.Label | None, object]]:
        if isinstance(node, Braket) and _IDENT.fullmatch(node.content.strip()):
            name = node.content.strip()
            if name in self.states:
                s = self.states[name]
                return [
                    (D.Label("bra" if node.bra else "ket", name), P.dagger(s) if node.bra else s)
                ]
        if isinstance(node, Name):
            n = node.id
            if n in self.lists:
                return list(self.lists[n])
            if n in self.funcs:
                return [(D.Label("func", n), self.funcs[n])]
            if n in self.ops:
                return [(D.Label("op", n), self.ops[n])]
            if n in self.values:
                return [(D.Label("var", n), self.values[n])]
        return [(None, self._eval(node, Ctx()))]

    def _assign_state(self, head: Tok, rest: list[Tok]) -> None:
        name = head.text.strip()
        self._check_name(name, "state")
        p = self._parser(rest)
        if p.done():
            raise LangSyntaxError("missing the value after =")
        value = self._eval(p.expr(), Ctx())
        if not p.done():
            raise LangSyntaxError(f"unexpected {p.peek().text!r}")  # type: ignore[union-attr]
        if not isinstance(value, P.State):
            raise LangTypeError(
                f"{'<' + name + '|' if head.kind == 'bra' else '|' + name + '>'} "
                f"must be assigned a state, got {self._kind(value)}"
            )
        want_bra = head.kind == "bra"
        if value.bra != want_bra:
            raise LangTypeError(
                f"cannot assign a {'bra' if value.bra else 'ket'} to "
                f"{'a bra' if want_bra else 'a ket'} name"
            )
        if any(w is None for w in value.widths):
            raise LangTypeError(
                "cannot infer every register width; use a sum range or |v> inside a gate"
            )
        state = P.dagger(value) if want_bra else value
        self.states[name] = P.simplify(state)

    @staticmethod
    def _kind(v: object) -> str:
        if isinstance(v, P.State):
            return "a bra" if v.bra else "a ket"
        if isinstance(v, P.Operator):
            return "an operator"
        return "a scalar"

    def _scalar(self, v: object, where: str) -> sympy.Expr:
        if isinstance(v, sympy.Basic):
            return v
        raise LangTypeError(f"{where} must be a scalar, got {self._kind(v)}")

    def _eval(self, node: object, ctx: Ctx) -> object:
        try:
            return self._eval_node(node, ctx)
        except P.PathSumError as exc:
            raise LangError(str(exc)) from exc

    def _eval_node(self, node: object, ctx: Ctx) -> object:
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Name):
            return self._name(node.id, ctx)
        if isinstance(node, Call):
            args = [
                self._scalar(self._eval_node(a, ctx), f"an argument of {node.name}")
                for a in node.args
            ]
            fn = _FUNCS.get(node.name) or self.funcs[node.name]
            try:
                return fn(*args)
            except (TypeError, ValueError) as exc:
                raise LangTypeError(f"{node.name}: {exc}") from exc
        if isinstance(node, Neg):
            return self._mul(sympy.Integer(-1), self._eval_node(node.arg, ctx))
        if isinstance(node, Braket):
            return self._braket(node, ctx)
        if isinstance(node, SumNode):
            return self._sum(node, ctx)
        if isinstance(node, Juxt):
            return self._juxt(self._eval_node(node.left, ctx), self._eval_node(node.right, ctx))
        if isinstance(node, Bin):
            return self._binary(
                node.op, self._eval_node(node.left, ctx), self._eval_node(node.right, ctx)
            )
        raise LangSyntaxError(f"cannot evaluate {node!r}")

    def _name(self, name: str, ctx: Ctx) -> object:
        if name in ctx.locals:
            return ctx.locals[name].value
        if name in self.values:
            v = self.values[name]
            if isinstance(v, (Bits, bool)):
                return sympy.Integer(int(v))
            return v
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        if name in self.ops:
            return self.ops[name]
        if name in self.funcs:
            raise LangTypeError(f"{name} is a function; call it as {name}(...)")
        if name in self.states:
            raise LangTypeError(f"{name} is a state; write |{name}> or <{name}|")
        if name in self.lists:
            raise LangTypeError(f"{name} is a list; only show takes lists")
        raise LangError(f"{name!r} is not defined; declare it with var")

    def _braket(self, node: Braket, ctx: Ctx) -> P.State:
        content = node.content.strip()
        if _IDENT.fullmatch(content) and content in self.states and content not in ctx.locals:
            s = self.states[content]
        elif content in ("+", "-"):
            sign = 1 if content == "+" else -1
            s = P.scale(1 / sympy.sqrt(2), P.add(P.basis(0, 1), P.scale(sign, P.basis(1, 1))))
        else:
            s = P.EMPTY
            for item in _split_items(content):
                s = P.tensor(s, self._basis(item, ctx))
            if not s.widths:
                raise LangSyntaxError("an empty ket")
        return P.dagger(s) if node.bra else s

    def _basis(self, text: str, ctx: Ctx) -> P.State:
        p = self._parser(tokenize(text))
        ast = p.expr()
        width: sympy.Expr | None = None
        if p.at(":"):
            p.take()
            width = self._scalar(self._eval_node(p.expr(), ctx), f"the width in |{text}>")
            if width.is_integer is False or (width.is_Integer and width < 1):
                raise LangTypeError(f"|{text}> needs a positive integer width")
        if not p.done():
            raise LangSyntaxError(f"cannot read the ket |{text}>")
        value = self._scalar(self._eval_node(ast, ctx), f"|{text}>")
        if value.is_integer is False or (value.is_Integer and value < 0):
            raise LangTypeError(f"|{text}> needs a non-negative integer value")
        if width is not None:
            if value.is_Integer and width.is_Integer and int(value) >= 2 ** int(width):
                raise LangTypeError(f"|{text}>: {value} does not fit in {width} bits")
        elif isinstance(ast, Name) and ast.id in ctx.locals:
            local = ctx.locals[ast.id]
            if local.width is not None:
                width = local.width
            elif local.bound is not None:
                width = _range_width(local.bound.lo, local.bound.hi)
        elif value.is_Integer and not ctx.body:
            width = sympy.Integer(max(1, int(value).bit_length()))
        if width is None and not ctx.body:
            raise LangTypeError(f"cannot infer the register width of |{text}>; write |{text} : w>")
        return P.basis(value, width)

    def _sum(self, node: SumNode, ctx: Ctx) -> object:
        lo = self._scalar(self._eval_node(node.lo, ctx), "a sum bound")
        hi = self._scalar(self._eval_node(node.hi, ctx), "a sum bound")
        b = P.new_bound(node.var, lo, hi)
        body = self._eval_node(node.body, ctx.child(node.var, Local(b.symbol, bound=b)))
        if isinstance(body, P.State):
            terms = tuple(P.Term(t.amp, (b,) + t.bound, t.kets) for t in body.terms)
            return P.simplify(P.State(body.widths, terms, body.bra))
        if isinstance(body, sympy.Basic):
            return sympy.Sum(body, (b.symbol, lo, hi)).doit()
        raise LangTypeError("a sum body must be a state or a scalar")

    def _mul(self, a: object, b: object) -> object:
        if isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic):
            return a * b
        if isinstance(a, sympy.Basic):
            a, b = b, a
        if isinstance(b, sympy.Basic):
            if isinstance(a, P.State):
                return P.scale(b, a)
            if isinstance(a, P.Operator):
                return P.Scaled(b, a)
        raise LangTypeError(f"cannot multiply {self._kind(a)} by {self._kind(b)}")

    def _juxt(self, a: object, b: object) -> object:
        if isinstance(a, sympy.Basic) or isinstance(b, sympy.Basic):
            return self._mul(a, b)
        if isinstance(a, P.Operator) and isinstance(b, P.State):
            if b.bra:
                raise LangTypeError("an operator applies to a ket; write U|psi>")
            return P.apply(a, b)
        if isinstance(a, P.Operator) and isinstance(b, P.Operator):
            return P.Compose((a, b))
        if isinstance(a, P.State) and isinstance(b, P.State):
            if a.bra and not b.bra:
                return P.inner(a, b)
            if not a.bra and b.bra:
                return P.Outer(a, b)
            return P.tensor(a, b)
        raise LangTypeError(f"cannot apply {self._kind(a)} to {self._kind(b)}")

    def _binary(self, op: str, a: object, b: object) -> object:
        scalars = isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic)
        if op == "*":
            if isinstance(a, P.Operator) and isinstance(b, P.Operator):
                return P.Compose((a, b))
            return self._mul(a, b)
        if op == "/":
            if not isinstance(b, sympy.Basic):
                raise LangTypeError(f"cannot divide by {self._kind(b)}")
            return self._mul(a, 1 / b)
        if op in ("+", "-"):
            if scalars:
                return a + b if op == "+" else a - b  # type: ignore[operator]
            if op == "-":
                b = self._mul(sympy.Integer(-1), b)
            if isinstance(a, P.State) and isinstance(b, P.State):
                return P.simplify(P.add(a, b))
            if isinstance(a, P.Operator) and isinstance(b, P.Operator):
                return P.OpSum((a, b))
            raise LangTypeError(f"cannot add {self._kind(a)} and {self._kind(b)}")
        if op == "@":
            if isinstance(a, P.State) and isinstance(b, P.State):
                return P.tensor(a, b)
            if isinstance(a, P.Operator) and isinstance(b, P.Operator):
                return P.Tensor((a, b))
            raise LangTypeError(f"cannot tensor {self._kind(a)} with {self._kind(b)}")
        if op == "^@":
            n = self._scalar(b, "a tensor power")
            if isinstance(a, P.State):
                return P.power_state(a, n)
            if isinstance(a, P.Operator):
                return P.Power(a, n)
            raise LangTypeError("^@ applies to a state or an operator")
        if op == "^" and isinstance(a, P.Operator):
            k = self._scalar(b, "an operator power")
            if not (k.is_Integer and k >= 0):
                raise LangTypeError("an operator power needs a concrete non-negative integer")
            return P.Compose((a,) * int(k)) if int(k) else P.Identity()
        if not scalars:
            raise LangTypeError(f"{op} needs scalars, got {self._kind(a)} and {self._kind(b)}")
        if op == "^":
            return a**b  # type: ignore[operator]
        if op == "%":
            return sympy.Mod(a, b)
        if op == "xor":
            return P.xor(a, b)
        raise LangSyntaxError(f"unknown operator {op}")


def _split_items(content: str) -> list[str]:
    items, depth, start = [], 0, 0
    for k, c in enumerate(content):
        depth += c in "({["
        depth -= c in ")}]"
        if c == "," and depth == 0:
            items.append(content[start:k])
            start = k + 1
    items.append(content[start:])
    return [x.strip() for x in items if x.strip()]
