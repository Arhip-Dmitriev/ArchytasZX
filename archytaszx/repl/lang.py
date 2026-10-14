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
    gate U = |x>|y> -> |x>|y xor f(x)>
    gate QFT = |x> -> 1/sqrt(q) sum(y in 0..q-1){ e^(2*pi*i*x*y/q) |y> }
    |psi> = (H^@n @ I)|0>^@n @ |0>^@m          <phi| = ...       |k : n>
    gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d-1){ e^(2*pi*i*x*y/d) |y>_d }
    |chi>_d = F^@n |0>_d^@n                    |2>_3
    var z = measure|psi>      measure|psi>
    list L = {|psi>, <phi|, z}
    show(L)  show(|psi>, ascii)  print("z =", z)  seed 7

Every state is named ``|name>``; ``<name|`` is its conjugate transpose. Classical values are
declared with ``var`` (types ``nat`` (default), ``int``, ``real``, ``complex``); measurement
results are booleans (one qubit) or :class:`Bits`, both usable as numbers. ``|v : w>`` gives
the basis ket v a register of width w. A ``_d`` suffix (a numeral or a variable) makes a
ket, bra or pattern a dimension-d qudit; a state whose registers share a dimension other than
2 is always written ``|name>_d``. A bare expression is a statement too.

Expression precedence, loosest first: ``xor``; ``+ -``; ``@``; juxtaposition (operator
application, scalar multiplication, ``|a>|b>`` tensor, ``<a|b>``, ``|a><b|``; grouped from
the right, so ``U |x>|y>`` applies U to both kets); ``* / %``;
unary ``-``; ``^`` (right associative) and postfix ``^@``. Constants ``pi``, ``e``, ``i``;
functions ``sqrt exp abs floor ceil`` and user ``func`` s. Built-in gates ``H X Y Z S T
CNOT SWAP`` (one-qubit registers; ``H^@n`` for a register) and the identity ``I`` (absorbs
the remaining registers when last in a tensor); ``AB`` is ``A B`` for operator names A, B.
Built-in states ``|+>``, ``|->``, ``|i>``, ``|-i>``.

``show`` prints mathematical values, ``print`` strings and concrete values. Values are
symbolic exactly when their inputs are. :attr:`Interpreter.last` is the :class:`Echo` of the
latest statement.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.repl import display as D
from archytaszx.repl import zxview
from archytaszx.repl.printer import DEFAULT_MODE, NotationMode
from archytaszx.semantics import pathsum as P


class LangError(Exception):
    """A statement that cannot be executed."""


class LangSyntaxError(LangError):
    """Text that does not parse."""


class LangTypeError(LangError):
    """A value used where its type is not allowed."""


@dataclass(frozen=True)
class Bits:
    """A measured digit string, one radix per digit (all 2 for qubits); acts as its integer
    value."""

    value: int
    radix: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.radix)

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __str__(self) -> str:
        out, v = [], self.value
        for r in reversed(self.radix):
            v, digit = divmod(v, r)
            out.append(digit)
        sep = "" if all(r <= 10 for r in self.radix) else "."
        return sep.join(str(x) for x in reversed(out))


@dataclass(frozen=True)
class Echo:
    """A statement's result: ``val label : type = value``, or ``- : type = value`` without a
    label; ``value`` is None for a bare symbol."""

    label: D.Label | None
    type: str
    value: object = None


@dataclass(frozen=True)
class Tok:
    kind: str
    text: str
    pos: int
    dim: str | None = None


_SIMPLE = re.compile(
    r"(?P<ws>\s+)|(?P<num>\d+(?:\.\d+)?)|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>->|\.\.|\^@|==|!=|>=|[-+*/%^@(){}\[\],=:>])"
)
_BRAKET_TAIL = re.compile(r"([^\s|<>][^|<>]*)>")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIM = re.compile(r"_(\d+|[A-Za-z_][A-Za-z0-9_]*)")


def _with_dim(src: str, i: int, kind: str, text: str, pos: int) -> tuple[Tok, int]:
    m = _DIM.match(src, i)
    if m:
        return Tok(kind, text, pos, m.group(1)), m.end()
    return Tok(kind, text, pos), i


def tokenize(src: str) -> list[Tok]:
    """Tokens of one statement; ``|..>`` and ``<..|``, with an optional ``_dim``, are single
    tokens."""
    out: list[Tok] = []
    i = 0
    while i < len(src):
        c = src[i]
        if c == "|":
            j = src.find(">", i + 1)
            if j < 0 or "|" in src[i + 1 : j] or "<" in src[i + 1 : j]:
                raise LangSyntaxError(f"unclosed ket at column {i + 1}")
            tok, i = _with_dim(src, j + 1, "ket", src[i + 1 : j], i)
            out.append(tok)
            continue
        if c == "<" and (src[i + 1 : i + 2].isspace() or src[i + 1 : i + 2] == "="):
            op = "<=" if src[i + 1 : i + 2] == "=" else "<"
            out.append(Tok("op", op, i))
            i += len(op)
            continue
        if c == "<":
            j = src.find("|", i + 1)
            if j < 0 or ">" in src[i + 1 : j]:
                raise LangSyntaxError(f"unclosed bra at column {i + 1}")
            tok, i = _with_dim(src, j + 1, "bra", src[i + 1 : j], i)
            out.append(tok)
            tail = _BRAKET_TAIL.match(src, i) if tok.dim is None else None
            if tail:
                tok, i = _with_dim(src, tail.end(), "ket", tail.group(1), i)
                if tok.dim is not None:
                    bra = out.pop()
                    out.append(Tok(bra.kind, bra.text, bra.pos, tok.dim))
                out.append(tok)
            continue
        if c == '"':
            j = src.find('"', i + 1)
            if j < 0:
                raise LangSyntaxError(f"unclosed string at column {i + 1}")
            out.append(Tok("str", src[i + 1 : j], i))
            i = j + 1
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
class Str:
    text: str


@dataclass(frozen=True)
class Not:
    arg: object


@dataclass(frozen=True)
class Neg:
    arg: object


@dataclass(frozen=True)
class Juxt:
    left: object
    right: object


@dataclass(frozen=True)
class Braket:
    """``|content>_dim`` (``bra=False``) or ``<content|_dim``; ``dim`` is None when absent."""

    content: str
    bra: bool
    dim: str | None = None


@dataclass(frozen=True)
class SumNode:
    var: str
    lo: object
    hi: object
    body: object


_KEYWORDS = {"xor", "in", "and", "or", "not", "else"}
_NAMED_STATES = {"+": sympy.Integer(1), "-": sympy.Integer(-1), "i": sympy.I, "-i": -sympy.I}
"""Built-in qubit states ``(|0> + c|1>)/sqrt(2)`` by their coefficient c."""
_COMPARE = {"==", "!=", "<", "<=", ">", ">="}
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
        node = self.conj()
        while self.at("or"):
            self.take()
            node = Bin("or", node, self.conj())
        return node

    def conj(self) -> object:
        node = self.negation()
        while self.at("and"):
            self.take()
            node = Bin("and", node, self.negation())
        return node

    def negation(self) -> object:
        if self.at("not"):
            self.take()
            return Not(self.negation())
        return self.compare()

    def compare(self) -> object:
        node = self.xor()
        t = self.peek()
        if t is not None and t.kind == "op" and t.text in _COMPARE:
            self.take()
            node = Bin(t.text, node, self.xor())
        return node

    def xor(self) -> object:
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
            return Braket(t.text, t.kind == "bra", t.dim)
        if t.kind == "str":
            self.i += 1
            return Str(t.text)
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
    dim: sympy.Expr | None = None


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


def _bitlen(v: sympy.Expr) -> sympy.Expr:
    if v.is_Integer:
        return sympy.Integer(int(v).bit_length())
    return sympy.floor(sympy.log(v, 2)) + 1


def _known(v: sympy.Expr) -> sympy.Basic:
    return sympy.true if not v.free_symbols else sympy.false


_FUNCS: dict[str, Callable[..., sympy.Expr]] = {
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "gcd": P.igcd,
    "min": sympy.Min,
    "max": sympy.Max,
    "bit": P.bit,
    "powmod": P.powmod,
    "bitlen": _bitlen,
    "denom": P.denom,
    "known": _known,
    "ifelse": lambda c, a, b: sympy.Piecewise((a, c), (b, True)),
}
_SPECIAL = {"arg", "rand"}
_CONSTANTS = {"pi": sympy.pi, "e": sympy.E, "i": sympy.I}
_STATEMENTS = {
    "var",
    "func",
    "gate",
    "list",
    "show",
    "print",
    "measure",
    "seed",
    "for",
    "while",
    "if",
    "assert",
    "assume",
}
_RESERVED = (
    _STATEMENTS | _KEYWORDS | set(_FUNCS) | _SPECIAL | set(_CONSTANTS) | {"sum"} | set(D.STYLES)
)

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
    """Statements split at newlines and ``;`` outside brackets; ``#`` comments dropped."""
    text = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    depth, start = 0, 0
    for k, c in enumerate(text):
        depth += c in "({["
        depth -= c in ")}]"
        if c in ";\n" and depth <= 0:
            yield text[start:k]
            start = k + 1
    yield text[start:]


def brace_depth(source: str) -> int:
    """Unclosed ``{`` count of ``source``, comments ignored."""
    text = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    return text.count("{") - text.count("}")


def _block(text: str, start: int) -> tuple[str, int]:
    """The body of the ``{...}`` opening at ``text[start]`` and the index after its ``}``."""
    depth = 0
    for k in range(start, len(text)):
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[start + 1 : k], k + 1
    raise LangSyntaxError("unclosed {")


def _concrete_width(top: int, d: int) -> sympy.Expr:
    """Fewest base-d digits holding ``0..top``."""
    w, cap = 1, d
    while cap <= top:
        w, cap = w + 1, cap * d
    return sympy.Integer(w)


def _range_width(lo: sympy.Expr, hi: sympy.Expr, d: sympy.Expr) -> sympy.Expr:
    if lo != 0:
        if hi.is_Integer and d.is_Integer:
            return _concrete_width(int(hi), int(d))
        raise LangTypeError("a ket over a symbolic range needs the range to start at 0")
    count = hi + 1
    if count.is_Integer and d.is_Integer:
        return _concrete_width(int(count) - 1, int(d))
    power = sympy.powsimp(count)
    if power.is_Pow and P.same(power.base, d):
        return power.exp
    w = sympy.simplify(sympy.expand_log(sympy.log(count, d), force=True))
    if not P.same(d**w, count):
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

    exact = sympy.lambdify(params, fast, modules=[P.scalar_impls(fast), "math"])

    def _scalar(*args: int) -> object:
        out = exact(*(int(a) for a in args))
        if isinstance(out, int) or not float(out).is_integer():
            return out
        return int(out)

    cls = type(name, (sympy.Function,), {"eval": classmethod(_eval), "nargs": len(params)})
    cls._params = tuple(params)  # type: ignore[attr-defined]
    cls._body = body  # type: ignore[attr-defined]
    cls._free = free  # type: ignore[attr-defined]
    cls._scalar = staticmethod(_scalar)  # type: ignore[attr-defined]
    cls._numeric = np.frompyfunc(_scalar, len(params), 1)  # type: ignore[attr-defined]
    return cls


class Interpreter:
    """Executes statements against named states, variables, functions and operators."""

    def __init__(
        self,
        *,
        style: str = "plot",
        seed: int | None = None,
        plotter: D.Plotter | None = None,
        args: tuple[str, ...] = (),
        check_oracles: bool = False,
        mode: NotationMode = DEFAULT_MODE,
    ) -> None:
        """``args`` are the command-line arguments read by ``arg(k)``; ``check_oracles`` checks
        every gate that maps basis states to basis states for reversibility; ``mode`` picks
        Dirac text, the ZX listing or both for ``show``."""
        if style not in D.STYLES:
            raise ValueError(f"unknown display style {style!r}")
        self.style = style
        self.check_oracles = check_oracles
        self.mode = mode
        self.last: Echo | None = None
        self.plotter = plotter if plotter is not None else D.terminal_image
        self.rng = np.random.default_rng(seed)
        self.args = tuple(args)
        self.assumptions: list[sympy.Basic] = []
        self.values: dict[str, object] = {}
        self.sugar: list[tuple[str, sympy.Expr]] = []
        self.funcs: dict[str, type] = {}
        self.ops: dict[str, P.Operator] = {"I": P.Identity()}
        self.states: dict[str, P.State] = {}
        self.lists: dict[str, list[tuple[D.Label | None, object]]] = {}
        self.measured: dict[sympy.Symbol, D.Outcome] = {}
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
        return Parser(toks, lambda name: name in _FUNCS or name in _SPECIAL or name in self.funcs)

    def _statement(self, text: str, builtin: bool = False) -> str:
        word = text.split(None, 1)[0]
        if word in ("for", "while", "if"):
            out: str = getattr(self, f"_stmt_{word}")(text)
            self.last = None
            return out
        if word == "else" or word.startswith("else{"):
            raise LangSyntaxError("else must follow the closing } of an if on the same line")
        self.last = None
        if word == "oracle":
            raise LangSyntaxError("oracle is gone; write gate (every gate is an oracle)")
        toks = tokenize(text)
        head = toks[0]
        if head.kind == "ident" and head.text in _STATEMENTS:
            return getattr(self, f"_stmt_{head.text}")(toks, text, builtin) or ""
        if head.kind in ("ket", "bra") and len(toks) > 1 and toks[1].text == "=":
            self._assign_state(head, toks[2:])
            return ""
        p = self._parser(toks)
        value = self._eval(p.expr(), Ctx())
        if not p.done():
            raise LangSyntaxError(
                f"unexpected {p.peek().text!r}; a statement starts with var, func, gate, "  # type: ignore[union-attr]
                "list, show, print, measure, seed, for, while, if, assert, assume, |name> =, "
                "<name| = or is an expression"
            )
        self.last = Echo(None, self._type(value), value)
        return ""

    def _type(self, v: object) -> str:
        """The type word of a value: ket, bra, op, bool, bits, nat, int, real or complex."""
        if isinstance(v, P.State):
            return "bra" if v.bra else "ket"
        if isinstance(v, P.Operator):
            return "op"
        if isinstance(v, Bits):
            return "bits"
        if isinstance(v, (bool, sympy.logic.boolalg.Boolean)):
            return "bool"
        if isinstance(v, sympy.Basic):
            if v.is_integer and v.is_nonnegative:
                return "nat"
            if v.is_integer:
                return "int"
            return "real" if v.is_real else "complex"
        return type(v).__name__

    def _header(self, text: str, word: str) -> tuple[str, str, str]:
        """``(header, body, rest)`` of ``WORD header { body } rest``."""
        brace = text.find("{")
        if brace < 0:
            raise LangSyntaxError(f"{word} needs a {{ block }}")
        body, end = _block(text, brace)
        return text[len(word) : brace].strip(), body, text[end:].strip()

    def _truth(self, node: object, what: str) -> bool:
        value = self._eval(node, Ctx())
        if isinstance(value, (bool, sympy.logic.boolalg.BooleanAtom)):
            return bool(value)
        if any(value == known for known in self.assumptions):
            return True
        if isinstance(value, sympy.Basic) and value.is_number:
            return bool(value != 0)
        raise LangTypeError(f"the {what} is not a concrete truth value: {value}")

    def _condition(self, text: str, what: str) -> object:
        p = self._parser(tokenize(text))
        node = p.expr()
        if not p.done():
            raise LangSyntaxError(f"unexpected {p.peek().text!r} in the {what}")  # type: ignore[union-attr]
        return node

    def _stmt_for(self, text: str) -> str:
        header, body, rest = self._header(text, "for")
        if rest:
            raise LangSyntaxError(f"unexpected {rest!r} after the for block")
        p = self._parser(tokenize(header))
        var = p.take()
        p.take("in")
        lo = self._scalar(self._eval(p.add(), Ctx()), "a for bound")
        p.take("..")
        hi = self._scalar(self._eval(p.add(), Ctx()), "a for bound")
        if not p.done():
            raise LangSyntaxError("usage: for NAME in LO..HI { ... }")
        self._check_name(var.text, "var")
        if not (lo.is_Integer and hi.is_Integer):
            self._fold_loop(var.text, lo, hi, body)
            return ""
        out = []
        for k in range(int(lo), int(hi) + 1):
            self._forget(var.text)
            self.values[var.text] = sympy.Integer(k)
            out.append(self.execute(body))
        return "\n".join(x for x in out if x)

    def _stmt_while(self, text: str) -> str:
        header, body, rest = self._header(text, "while")
        if rest:
            raise LangSyntaxError(f"unexpected {rest!r} after the while block")
        cond = self._condition(header, "while condition")
        out = []
        while self._truth(cond, "while condition"):
            out.append(self.execute(body))
        return "\n".join(x for x in out if x)

    def _stmt_if(self, text: str) -> str:
        header, body, rest = self._header(text, "if")
        if self._truth(self._condition(header, "if condition"), "if condition"):
            return self.execute(body)
        if not rest:
            return ""
        if not rest.startswith("else"):
            raise LangSyntaxError(f"unexpected {rest!r} after the if block")
        rest = rest[4:].strip()
        if rest.startswith("if"):
            return self._stmt_if(rest)
        if not rest.startswith("{"):
            raise LangSyntaxError("else needs a { block } or another if")
        inner, end = _block(rest, 0)
        if rest[end:].strip():
            raise LangSyntaxError(f"unexpected {rest[end:].strip()!r} after the else block")
        return self.execute(inner)

    def _fold_loop(self, var: str, lo: sympy.Expr, hi: sympy.Expr, body: str) -> None:
        """A loop over a symbolic range: each ``|s> = ...`` step is run once with the loop
        variable symbolic on a probe basis state, then folded over the range."""
        j = sympy.Symbol(var, integer=True, nonnegative=True)
        for stmt in _split_statements(body):
            text = stmt.strip()
            if not text:
                continue
            toks = tokenize(text)
            if not (toks[0].kind == "ket" and len(toks) > 1 and toks[1].text == "="):
                raise LangError(
                    f"a loop over the symbolic range {lo}..{hi} may only update states (|s> = ...)"
                )
            name = toks[0].text.strip()
            if name not in self.states:
                raise LangTypeError(f"|{name}> is not defined")
            cur = self.states[name]
            probes = [
                sympy.Dummy(f"u{k}", integer=True, nonnegative=True) for k in range(len(cur.widths))
            ]
            saved = self.values.get(var)
            self.values[var] = j
            self.states[name] = P.basis_kets(probes, cur.widths, cur.dims)
            try:
                step = self._eval(self._parser(toks[2:]).expr(), Ctx())
            finally:
                self.states[name] = cur
                if saved is None:
                    self.values.pop(var, None)
                else:
                    self.values[var] = saved
            if not isinstance(step, P.State):
                raise LangTypeError(f"|{name}> must be assigned a state")
            try:
                self.states[name] = P.simplify(P.fold(cur, step, probes, j, lo, hi))
            except P.PathSumError as exc:
                raise LangError(f"cannot run the loop symbolically: {exc}") from exc

    def _stmt_assume(self, toks: list[Tok], text: str, builtin: bool) -> None:
        source = text[len("assume") :].strip()
        value = self._eval(self._condition(source, "assumption"), Ctx())
        if isinstance(value, (bool, sympy.logic.boolalg.BooleanAtom)):
            if not value:
                raise LangError(f"assumption is false: {source}")
            return
        self.assumptions.append(value)

    def _stmt_assert(self, toks: list[Tok], text: str, builtin: bool) -> None:
        source = text[len("assert") :].strip()
        if not self._truth(self._condition(source, "assertion"), "assertion"):
            raise LangError(f"assertion failed: {source}")

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
                value = self.values[name]
                self.last = Echo(D.Label("var", name), self._type(value), value)
            else:
                self._forget(name)
                self.values[name] = sympy.Symbol(name, **_TYPES[kind])
                self.last = Echo(D.Label("var", name), kind)
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

    def _measure(self, p: Parser, name: str | None) -> object:
        """Measures the named ket of ``p``; returns the outcome and binds it to ``name``."""
        t = p.take()
        if t.kind != "ket" or not _IDENT.fullmatch(t.text.strip()):
            raise LangSyntaxError("measure takes a named ket, e.g. measure|psi>")
        label = t.text.strip()
        if label not in self.states:
            raise LangTypeError(f"|{label}> is not defined")
        self._check_dim(label, self._dim(t.dim, Ctx()))
        if not p.done() and not p.at(","):
            raise LangSyntaxError("measure takes exactly one named ket")
        if name is None:
            self._count += 1
            sym_name = f"m{self._count}"
        else:
            sym_name = name
        symbol = sympy.Symbol(sym_name, integer=True, nonnegative=True)
        facts = [f for info in self.measured.values() for f in info.facts]
        state = self.states[label]
        try:
            m = P.measure(state, symbol, self.rng, facts)
        except P.PathSumError as exc:
            raise LangError(str(exc)) from exc
        self.states[label] = m.post
        outcome = m.outcome
        value: object = outcome
        if outcome.is_Integer and P.is_concrete(m.post):
            radix = tuple(
                int(d)  # type: ignore[arg-type]
                for d, w in zip(m.post.dims, m.post.widths, strict=True)
                for _ in range(int(w))  # type: ignore[arg-type]
            )
            value = bool(outcome) if radix == (2,) else Bits(int(outcome), radix)
        if name is not None:
            self._forget(name)
            self.values[name] = value
        if outcome == symbol:
            dim = state.uniform_dim()
            size = (
                m.width
                if dim is not None
                else sympy.Mul(*(state.levels(k) for k in range(len(state.widths))))
            )
            self.measured[symbol] = D.Outcome(size, label, m.constraints, dim)
        return value

    def _stmt_measure(self, toks: list[Tok], text: str, builtin: bool) -> None:
        p = self._parser(toks[1:])
        value = self._measure(p, None)
        self.last = Echo(None, self._type(value), value)

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
        self.last = Echo(D.Label("func", name), "func", self.funcs[name])

    def _stmt_gate(self, toks: list[Tok], text: str, builtin: bool) -> None:
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
        slots = self._pattern(pattern)
        names = [q for q, _ in slots]
        source = text.split("=", 1)[1].strip()

        def fn(
            vals: tuple[sympy.Expr, ...],
            widths: tuple[sympy.Expr, ...],
            dims: tuple[sympy.Expr, ...],
        ) -> P.State:
            for (q, want), d in zip(slots, dims, strict=True):
                if want is not None and not P.same(self._dim(want, Ctx()), d):
                    raise LangTypeError(f"{name}: |{q}>_{want} applied to a dimension-{d} register")
            ctx = Ctx(
                {
                    q: Local(v, w, dim=d)
                    for q, v, w, d in zip(names, vals, widths, dims, strict=True)
                },
                True,
            )
            res = self._eval(body, ctx)
            if not isinstance(res, P.State) or res.bra:
                raise LangTypeError(f"the body of {name} must be a ket")
            return self._fix_widths(res, widths, dims)

        oracle = self.check_oracles and not builtin
        self.ops[name] = P.Gate(name, len(names), fn, qubit=builtin, oracle=oracle, source=source)
        self.last = Echo(D.Label("op", name), "gate", self.ops[name])

    def _pattern(self, node: object) -> list[tuple[str, str | None]]:
        if isinstance(node, Braket) and not node.bra:
            content = node.content.strip()
            if not _IDENT.fullmatch(content) or content in _RESERVED:
                raise LangSyntaxError(f"a pattern ket holds one variable name, got |{content}>")
            return [(content, node.dim)]
        if isinstance(node, Juxt) or (isinstance(node, Bin) and node.op == "@"):
            slots = self._pattern(node.left) + self._pattern(node.right)
            names = [q for q, _ in slots]
            if len(set(names)) != len(names):
                raise LangSyntaxError("pattern variables must be distinct")
            return slots
        raise LangSyntaxError("a gate pattern is a product of kets, e.g. |x>|y>")

    def _fix_widths(
        self,
        s: P.State,
        defaults: tuple[sympy.Expr | None, ...],
        default_dims: tuple[sympy.Expr | None, ...],
    ) -> P.State:
        positional = len(s.widths) == len(defaults)
        dims = [
            d if d is not None else (default_dims[k] if positional else P.TWO)
            for k, d in enumerate(s.dims)
        ]
        widths = list(s.widths)
        for k, w in enumerate(widths):
            if w is not None:
                continue
            if positional:
                widths[k] = defaults[k]
                continue
            values = [t.kets[k] for t in s.terms]
            d = dims[k]
            if all(v.is_Integer for v in values) and d is not None and d.is_Integer:
                widths[k] = _concrete_width(max(int(v) for v in values), int(d))
            else:
                raise LangTypeError("cannot infer an output register width")
        return P.State(tuple(widths), s.terms, s.bra, tuple(dims))

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
        self.last = Echo(D.Label("list", name), "list", items)

    def _arguments(self, toks: list[Tok], text: str, word: str) -> list[tuple[object, str]]:
        """``(node, source text)`` per argument of ``WORD(a, b, ...)`` or ``WORD a, b``."""
        p = self._parser(toks[1:])
        paren = p.at("(")
        if paren:
            p.take("(")
        nodes: list[tuple[object, str]] = []
        while not (p.done() or p.at(")")):
            start = p.i
            node = p.expr()
            stop = p.toks[p.i].pos if p.i < len(p.toks) else len(text)
            nodes.append((node, text[p.toks[start].pos : stop].strip()))
            if p.at(","):
                p.take(",")
        if paren:
            p.take(")")
        if not p.done():
            raise LangSyntaxError(f"unexpected text after {word}(...)")
        return nodes

    def _stmt_show(self, toks: list[Tok], text: str, builtin: bool) -> str:
        nodes = self._arguments(toks, text, "show")
        style = self.style
        if nodes and isinstance(nodes[-1][0], Name) and nodes[-1][0].id in D.STYLES:
            style = nodes.pop()[0].id  # type: ignore[attr-defined]
        entries: list[tuple[D.Label | None, object]] = []
        for node, source in nodes:
            if isinstance(node, Str):
                raise LangTypeError(f'show takes values; write print("{node.text}")')
            entries.extend(self._entries(node, source))
        return self.render(entries, style)

    def _stmt_print(self, toks: list[Tok], text: str, builtin: bool) -> str:
        parts: list[str] = []
        for node, source in self._arguments(toks, text, "print"):
            if isinstance(node, Str):
                parts.append(node.text)
                continue
            for _, value in self._entries(node, source):
                parts.append(_plain(value, source))
        return " ".join(parts)

    def _context(self) -> D.Context:
        sugar = sorted(
            ((v, sympy.Symbol(n)) for n, v in self.sugar),
            key=lambda pair: -sympy.count_ops(pair[0]),
        )
        taken = frozenset(self.values) | frozenset(self.states) | frozenset(self.funcs)
        order = {sympy.Symbol(n): k for k, (n, _) in enumerate(self.sugar)}
        return D.Context(
            sugar, self.measured, taken | frozenset(self.ops), tuple(self.assumptions), order
        )

    def _dim(self, text: str | None, ctx: Ctx) -> sympy.Expr | None:
        """The dimension named by a ``_dim`` suffix, or None without one."""
        if text is None:
            return None
        d = sympy.Integer(text) if text.isdigit() else self._name(text, ctx)
        if not isinstance(d, sympy.Basic) or d.is_integer is False or (d.is_Integer and d < 2):
            raise LangTypeError(f"_{text} must name an integer dimension of at least 2")
        return d

    def _check_dim(self, name: str, dim: sympy.Expr | None) -> None:
        """Requires ``_dim`` exactly when the state's registers share a dimension other than 2."""
        have = self.states[name].uniform_dim()
        if dim is None:
            if have is not None and not P.same(have, 2):
                raise LangTypeError(
                    f"|{name}> holds dimension-{have} qudits; write |{name}>_{have}"
                )
            return
        if have is None or not P.same(have, dim):
            got = "mixed dimensions" if have is None else f"dimension {have}"
            raise LangTypeError(f"|{name}>_{dim} does not match its {got}")

    def _label(self, kind: str, name: str) -> D.Label:
        have = self.states[name].uniform_dim()
        return D.Label(kind, name, None if have is None or P.same(have, 2) else have)

    def render(
        self,
        entries: list[tuple[D.Label | None, object]],
        style: str | None = None,
        mode: NotationMode | None = None,
    ) -> str:
        """Entries as Dirac lines (an image in ``plot`` style), ZX listings of states, or both,
        then a ``where`` line for what the non-state entries mention."""
        style = self.style if style is None else style
        mode = self.mode if mode is None else mode
        if style == "plot":
            image = self._blocks(entries, "plot", mode)
            if image is not None:
                return image
            style = "unicode"
        return self._blocks(entries, style, mode) or ""

    def _blocks(
        self, entries: list[tuple[D.Label | None, object]], style: str, mode: NotationMode
    ) -> str | None:
        """Rendered entries; None when a plot cannot be drawn."""
        r = D.Renderer(style, self._context())
        out: list[str] = []
        math: list[str] = []

        def flush() -> bool:
            if math and style == "plot":
                image = self.plotter(list(math))
                if image is None:
                    return False
                out.append(image)
            else:
                out.extend(math)
            math.clear()
            return True

        for lab, v in entries:
            state = isinstance(v, P.State)
            if not state or mode is not NotationMode.ZX:
                math.append(r.entry(lab, v, record=not state))
            if state and mode is not NotationMode.DIRAC:
                if not flush():
                    return None
                out.append(self._zx_text(lab, v))  # type: ignore[arg-type]
        where = r.where()
        if where:
            math.append(where)
        if not flush():
            return None
        return "\n".join(out)

    def echo_parts(self, e: Echo, style: str | None = None) -> tuple[str, str, str | None]:
        """``(name, type, value text)`` of ``e`` in a text style; the name is ``-`` without a
        label and the value text None for a bare symbol."""
        style = self.style if style is None else style
        style = "unicode" if style == "plot" else style
        r = D.Renderer(style, self._context())
        lab, v = e.label, e.value
        name = "-" if lab is None else r.label(lab)
        if lab is not None and lab.kind == "func":
            params = ", ".join(D.name_text(q.name, style) for q in v._params)  # type: ignore[attr-defined]
            return f"{name}({params})", e.type, r.expr(v._body)  # type: ignore[attr-defined]
        if lab is not None and lab.kind == "op" and isinstance(v, P.Gate):
            return name, e.type, D.source_text(v.source, style)
        if lab is not None and lab.kind == "list":
            items = [r.label(x) if x is not None else r.value(y) for x, y in v]  # type: ignore[attr-defined]
            return name, e.type, "{" + ", ".join(items) + "}"
        if v is None:
            return name, e.type, None
        if isinstance(v, sympy.Symbol) and v in self.measured:
            return name, e.type, r.measured(v)
        if lab is not None and lab.kind == "var" and isinstance(v, sympy.Expr):
            return name, e.type, r.expr(v, sympy.Symbol(lab.name))
        if isinstance(v, P.State) and self.mode is not NotationMode.DIRAC:
            listing = self._zx_body(v)
            if self.mode is NotationMode.ZX:
                return name, e.type, listing
            return name, e.type, r.value(v) + "\n" + listing
        return name, e.type, r.value(v)

    def _zx_text(self, label: D.Label | None, s: P.State) -> str:
        """``label =`` then the indented ZX listing of ``s``."""
        style = "ascii" if self.style == "ascii" else "unicode"
        head = D.Renderer(style).label(label) if label is not None else ""
        return f"{head} =\n{self._zx_body(s)}".lstrip()

    @staticmethod
    def _zx_body(s: P.State) -> str:
        """The ZX listing of ``s`` indented by two spaces, or ``no ZX form (reason)``."""
        try:
            body = zxview.listing(s)
        except zxview.ZXViewError as exc:
            return f"  no ZX form ({exc})"
        return "\n".join("  " + line for line in body.splitlines())

    def _entries(self, node: object, source: str = "") -> list[tuple[D.Label | None, object]]:
        if isinstance(node, Braket) and _IDENT.fullmatch(node.content.strip()):
            name = node.content.strip()
            if name in self.states:
                s = self.states[name]
                self._check_dim(name, self._dim(node.dim, Ctx()))
                kind = "bra" if node.bra else "ket"
                return [(self._label(kind, name), P.dagger(s) if node.bra else s)]
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
        if isinstance(node, Str):
            return [(None, D.Text(node.text))]
        label = D.Label("expr", source) if source and not isinstance(node, Num) else None
        return [(label, self._eval(node, Ctx()))]

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
        dim = self._dim(head.dim, Ctx())
        have = value.uniform_dim()
        if dim is not None and (have is None or not P.same(have, dim)):
            got = "mixed dimensions" if have is None else f"dimension {have}"
            raise LangTypeError(f"cannot assign a state of {got} to |{name}>_{dim}")
        if dim is None and have is not None and not P.same(have, 2):
            raise LangTypeError(f"the state holds dimension-{have} qudits; write |{name}>_{have} =")
        state = P.dagger(value) if want_bra else value
        self.states[name] = P.simplify(state)
        shown = P.dagger(self.states[name]) if want_bra else self.states[name]
        self.last = Echo(self._label(head.kind, name), head.kind, shown)

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
        if isinstance(node, Call) and node.name in _SPECIAL:
            return self._special(node, ctx)
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
        if isinstance(node, Str):
            raise LangTypeError("a string can only be shown")
        if isinstance(node, Not):
            return sympy.Not(_boolean(self._scalar(self._eval_node(node.arg, ctx), "not")))
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
            raise LangTypeError(f"{name} is a list; only show and print take lists")
        ops = self._split_ops(name)
        if ops is not None:
            return P.Compose(tuple(self.ops[o] for o in ops))
        raise LangError(f"{name!r} is not defined; declare it with var")

    def _split_ops(self, name: str) -> list[str] | None:
        """The only way to write ``name`` as two or more operator names, or None."""
        ways: dict[int, list[list[str]]] = {0: [[]]}
        for end in range(1, len(name) + 1):
            ways[end] = [
                w + [name[k:end]] for k in range(end) if name[k:end] in self.ops for w in ways[k]
            ][:2]
        found = [w for w in ways[len(name)] if len(w) > 1]
        if len(found) > 1:
            raise LangError(f"{name!r} splits into operators in more than one way; use spaces")
        return found[0] if found else None

    def _braket(self, node: Braket, ctx: Ctx) -> P.State:
        content = node.content.strip()
        dim = self._dim(node.dim, ctx)
        if _IDENT.fullmatch(content) and content in self.states and content not in ctx.locals:
            self._check_dim(content, dim)
            s = self.states[content]
        elif content in _NAMED_STATES and not (content == "i" and "i" in ctx.locals):
            if dim is not None and not P.same(dim, 2):
                raise LangTypeError(f"|{content}> is a qubit state")
            one = P.scale(_NAMED_STATES[content], P.basis(1, 1))
            s = P.scale(1 / sympy.sqrt(2), P.add(P.basis(0, 1), one))
        else:
            if not content:
                raise LangSyntaxError("an empty ket")
            if len(_split_items(content)) > 1:
                parts = "".join(f"|{x}>" for x in _split_items(content))
                raise LangSyntaxError(f"|{content}> is not a ket; write {parts}")
            s = self._basis(content, ctx, dim)
        return P.dagger(s) if node.bra else s

    def _basis(self, text: str, ctx: Ctx, dim: sympy.Expr | None) -> P.State:
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
        local = ctx.locals.get(ast.id) if isinstance(ast, Name) else None
        explicit = dim is not None
        if dim is None and local is not None and local.dim is not None:
            dim = local.dim
        if dim is None and not ctx.body:
            dim = P.TWO
        if width is not None:
            if (
                dim is not None
                and _ints(value, width, dim)
                and int(value) >= int(dim) ** int(width)
            ):
                raise LangTypeError(
                    f"|{text}>: {value} does not fit in {width} digits of base {dim}"
                )
        elif local is not None and local.width is not None:
            width = local.width
        elif local is not None and local.bound is not None:
            width = _range_width(local.bound.lo, local.bound.hi, dim if dim is not None else P.TWO)
        elif value.is_Integer and dim is not None and dim.is_Integer and not ctx.body:
            width = _concrete_width(int(value), int(dim))
        elif explicit:
            width = sympy.Integer(1)
        if width is None and not ctx.body:
            raise LangTypeError(f"cannot infer the register width of |{text}>; write |{text} : w>")
        return P.basis(value, width, dim)

    def _sum(self, node: SumNode, ctx: Ctx) -> object:
        lo = self._scalar(self._eval_node(node.lo, ctx), "a sum bound")
        hi = self._scalar(self._eval_node(node.hi, ctx), "a sum bound")
        b = P.new_bound(node.var, lo, hi)
        body = self._eval_node(node.body, ctx.child(node.var, Local(b.symbol, bound=b)))
        if isinstance(body, P.State):
            terms = tuple(P.Term(t.amp, (b,) + t.bound, t.kets) for t in body.terms)
            return P.simplify(P.State(body.widths, terms, body.bra, body.dims))
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

    def _special(self, node: Call, ctx: Ctx) -> sympy.Expr:
        args = [
            self._scalar(self._eval_node(a, ctx), f"an argument of {node.name}") for a in node.args
        ]
        if not all(a.is_Integer for a in args):
            raise LangTypeError(f"{node.name} takes concrete integers")
        if node.name == "rand":
            if len(args) != 2 or args[0] > args[1]:
                raise LangTypeError("usage: rand(LO, HI) with LO <= HI")
            return sympy.Integer(int(self.rng.integers(int(args[0]), int(args[1]) + 1)))
        if len(args) not in (1, 2) or args[0] < 1:
            raise LangTypeError("usage: arg(K) or arg(K, DEFAULT), K counting from 1")
        k = int(args[0])
        if k > len(self.args):
            if len(args) == 2:
                return args[1]
            raise LangError(f"missing command-line argument {k}")
        text = self.args[k - 1]
        if text.lstrip("-").isdigit():
            return sympy.Integer(int(text))
        if not _IDENT.fullmatch(text) or text in _RESERVED:
            raise LangTypeError(f"argument {k} is neither an integer nor a name: {text!r}")
        return sympy.Symbol(text, integer=True, positive=True)

    def _binary(self, op: str, a: object, b: object) -> object:
        scalars = isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic)
        if op in _COMPARE or op in ("and", "or"):
            if not scalars:
                raise LangTypeError(
                    f"{op} compares scalars, got {self._kind(a)} and {self._kind(b)}"
                )
            if op == "and":
                return sympy.And(_boolean(a), _boolean(b))
            if op == "or":
                return sympy.Or(_boolean(a), _boolean(b))
            rel = {
                "==": sympy.Eq,
                "!=": sympy.Ne,
                "<": sympy.Lt,
                "<=": sympy.Le,
                ">": sympy.Gt,
                ">=": sympy.Ge,
            }[op]
            return rel(a, b)
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


def _plain(v: object, source: str) -> str:
    """A concrete value as plain text: true/false, integers, fractions or decimals."""
    if isinstance(v, D.Text):
        return v.text
    if isinstance(v, (bool, sympy.logic.boolalg.BooleanAtom)):
        return "true" if v else "false"
    if isinstance(v, Bits):
        return str(int(v))
    if isinstance(v, sympy.Basic) and v.is_number:
        if v.is_Rational:
            return str(v)
        z = complex(v.evalf())

        def num(x: float) -> str:
            return str(int(x)) if x.is_integer() else repr(x)

        if z.imag == 0:
            return num(z.real)
        sign = "-" if z.imag < 0 else "+"
        return f"{num(z.real)} {sign} {num(abs(z.imag))}i" if z.real else f"{num(z.imag)}i"
    what = "symbolic" if isinstance(v, sympy.Basic) else "not a concrete value"
    raise LangTypeError(f"print takes strings and concrete values; {source} is {what}, use show")


def _boolean(v: sympy.Basic) -> sympy.Basic:
    """A truth value from a boolean or a number (nonzero is true)."""
    if isinstance(v, sympy.logic.boolalg.Boolean):
        return v
    if v.is_number:
        return sympy.true if v != 0 else sympy.false
    return sympy.Ne(v, 0)


def _ints(*xs: sympy.Expr) -> bool:
    return all(x.is_Integer for x in xs)


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
