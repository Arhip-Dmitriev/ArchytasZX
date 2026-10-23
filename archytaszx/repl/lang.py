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
    gate QFT = |x> -> 1/sqrt(q) sum(y in 0..q){ e^(2*pi*i*x*y/q) |y> }
    |psi> = (H^@n @ I)|0>^@n @ |0>^@m          <phi| = ...       |psi> - 2 |w><w|psi>
    gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }
    gate R = [[cos(t), -sin(t)], [sin(t), cos(t)]]      |v> = [a, b]
    gate G = matrix(r, c in 0..d){ e^(2*pi*i*r*c/d) / sqrt(d) }
    |chi>_d = F^@n |0>_d^@n                    |2>_3     |0> + |2>     |010`>
    (H^@n @ I)|psi>           var z = measure|psi>      var x = floor(measure|psi> / 2)
    list L = {|psi>, <phi|, z}
    show(L)  show(|psi>, ascii)  print(f"z = {z}")  seed 7
    for j in 0..n { ... }     sum(y in 0..=q-1){ ... }

Every state is named ``|name>``; ``<name|`` is its conjugate transpose. A statement that is
operators applied to a named state, ``U V|name>``, replaces that state by the result. Classical
values are declared with ``var`` (types ``nat`` (default), ``int``, ``real``, ``complex``).
``measure|name>`` is an expression: the Born distribution of every register, derived from the
path sum, as ``[(value, probability), ...]`` (a branch over free indices stands for all their
values); the state becomes the density matrix of the mixture of its outcomes, which gates act
on as ``U rho U^dagger``. ``|rho> = |a><a| + ...`` (scaled outer products) names a density
matrix directly. Scalar functions and operators map over a distribution's outcomes,
merging equal ones. A literal ``|v>`` is one register of the lowest dimension holding v (of
unresolved width for a symbolic v), laid out as a sum, inner product, gate or ``|name>_d =``
needs: ``|000>`` is ``|0>``. A trailing backtick makes a digit string, one register per
digit: ``|010`>`` is ``|0>|1>|0>``. A ``_d`` suffix (a numeral or a variable) fixes a ket,
bra or pattern to dimension d. A matrix gate maps ``|c>`` to
``sum_r M[r][c] |r>`` over the registers it meets; ``|x>|y> -> [[...]]`` fixes them. Ranges
``a..b`` exclude b, ``a..=b`` include it. A bare expression is a statement too.

Expression precedence, loosest first: ``xor``; ``+ -``; ``@``; juxtaposition (operator
application, scalar multiplication, ``|a>|b>`` tensor, ``<a|b>``, ``|a><b|``; grouped from
the right, so ``U |x>|y>`` applies U to both kets); ``* / %``;
unary ``-``; ``^`` (right associative) and postfix ``^@``. Constants ``pi``, ``e``, ``i``;
functions ``sqrt exp abs floor ceil cos sin tan log`` and user ``func`` s. Built-in gates
``H X Y Z S T CNOT SWAP`` (one-qubit registers; ``H^@n`` for a register) and the identity
``I`` (absorbs the remaining registers when last in a tensor). Built-in states ``|+>``,
``|->``, ``|i>``, ``|-i>``. ``normalise(s)`` divides a state by its norm.

``show`` prints mathematical values, ``print`` strings and concrete values. Values are
symbolic exactly when their inputs are. :attr:`Interpreter.last` is the :class:`Echo` of the
latest statement.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.repl import display as D
from archytaszx.repl import zxdraw, zxsym, zxview
from archytaszx.repl.printer import DEFAULT_MODE, NotationMode
from archytaszx.semantics import pathsum as P


class LangError(Exception):
    """A statement that cannot be executed."""


class LangSyntaxError(LangError):
    """Text that does not parse."""


class LangTypeError(LangError):
    """A value used where its type is not allowed."""


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
    r"|(?P<op>->|\.\.=|\.\.|\^@|==|!=|>=|[-+*/%^@(){}\[\],=:>])"
)
_BRAKET_TAIL = re.compile(r"([^\s|<>][^|<>]*)>")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIM = re.compile(r"_(\d+|[A-Za-z_][A-Za-z0-9_]*)")
_DIGIT_STRING = re.compile(r"\d*`")


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
        fmt = c == "f" and src[i + 1 : i + 2] == '"' and not src[i - 1 : i].isalnum()
        if c == '"' or fmt:
            q = i + fmt
            j = src.find('"', q + 1)
            if j < 0:
                raise LangSyntaxError(f"unclosed string at column {i + 1}")
            out.append(Tok("fstr" if fmt else "str", src[q + 1 : j], i))
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
class Fstr:
    """``f"..."``: literal text and ``{expr[:spec]}`` fields."""

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
class MeasureNode:
    """``measure|name>``."""

    target: Braket


@dataclass(frozen=True)
class MatrixNode:
    """``[[a, b], [c, d]]``: rows of entry expressions."""

    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class VectorNode:
    """``[a, b, ...]``: amplitudes of a state."""

    items: tuple[object, ...]


@dataclass(frozen=True)
class MatrixFor:
    """``matrix(row, col in lo..hi){ entry }``; ``hi`` excluded unless ``inclusive``."""

    row: str
    col: str
    lo: object
    hi: object
    body: object
    inclusive: bool = False


@dataclass(frozen=True)
class SumNode:
    """``sum(var in lo..hi){ body }``; ``hi`` excluded unless ``inclusive`` (``..=``)."""

    var: str
    lo: object
    hi: object
    body: object
    inclusive: bool = False


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
        return t.kind == "op" and t.text in ("(", "[")

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
        if t.kind == "fstr":
            self.i += 1
            return Fstr(t.text)
        if t.kind == "ident":
            if t.text == "sum":
                return self.sum()
            if t.text == "matrix":
                return self.matrix()
            if t.text == "measure":
                self.i += 1
                target = self.peek()
                if target is None or target.kind != "ket":
                    raise LangSyntaxError("measure takes a named ket, e.g. measure|psi>")
                self.i += 1
                return MeasureNode(Braket(target.text, False, target.dim))
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
        if t.kind == "op" and t.text == "[":
            return self.brackets()
        raise LangSyntaxError(f"unexpected {t.text!r} at column {t.pos + 1}")

    def brackets(self) -> MatrixNode | VectorNode:
        """``[[...], ...]`` (a matrix, rows of equal length) or ``[a, b, ...]`` (a vector)."""
        self.take("[")
        if not self.at("["):
            items = self.args()
            self.take("]")
            return VectorNode(tuple(items))
        rows: list[tuple[object, ...]] = []
        while True:
            self.take("[")
            rows.append(tuple(self.args()))
            self.take("]")
            if not self.at(","):
                break
            self.take(",")
        self.take("]")
        if any(len(row) != len(rows) for row in rows):
            raise LangSyntaxError(
                f"a matrix needs {len(rows)} entries in each of its {len(rows)} rows"
            )
        return MatrixNode(tuple(rows))

    def matrix(self) -> MatrixFor:
        self.take("matrix")
        self.take("(")
        names = []
        for k in range(2):
            t = self.take()
            if t.kind != "ident":
                raise LangSyntaxError(f"expected an index name at column {t.pos + 1}")
            names.append(t.text)
            if k == 0:
                self.take(",")
        self.take("in")
        lo = self.add()
        inclusive = self.range_op()
        hi = self.add()
        self.take(")")
        self.take("{")
        body = self.expr()
        self.take("}")
        return MatrixFor(names[0], names[1], lo, hi, body, inclusive)

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
        inclusive = self.range_op()
        hi = self.add()
        self.take(")")
        self.take("{")
        body = self.expr()
        self.take("}")
        return SumNode(var.text, lo, hi, body, inclusive)

    def range_op(self) -> bool:
        """Takes ``..`` (False) or ``..=`` (True)."""
        if self.at("..="):
            self.take()
            return True
        self.take("..")
        return False


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
    "cos": sympy.cos,
    "sin": sympy.sin,
    "tan": sympy.tan,
    "log": sympy.log,
}
_SPECIAL = {"arg", "rand"}
_STATE_FUNCS = {"normalise"}
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
    _STATEMENTS
    | _KEYWORDS
    | set(_FUNCS)
    | _SPECIAL
    | _STATE_FUNCS
    | set(_CONSTANTS)
    | {"sum", "matrix"}
    | set(D.STYLES)
)

PRELUDE = """
gate H = |x> -> 1/sqrt(2) sum(y in 0..2){ (-1)^(x*y) |y> }
gate X = |x> -> |1 - x>
gate Y = |x> -> i*(-1)^x |1 - x>
gate Z = |x> -> (-1)^x |x>
gate S = |x> -> i^x |x>
gate T = |x> -> e^(i*pi*x/4) |x>
gate CNOT = |c>|t> -> |c>|t xor c>
gate SWAP = |a>|b> -> |b>|a>
"""
"""Built-in gates, defined in the language itself."""


_BLOCK_WORDS = ("for", "while", "if")
_ELSE = re.compile(r"else\b")


def _code(source: str) -> str:
    """``source`` without ``#`` comments; string contents blanked to spaces."""
    out: list[str] = []
    for line in source.splitlines():
        quoted = False
        for c in line:
            if c == '"':
                quoted = not quoted
            elif c == "#" and not quoted:
                break
            out.append(" " if quoted and c != '"' else c)
        out.append("\n")
    return "".join(out)


def _uncommented(source: str) -> str:
    """``source`` without ``#`` comments outside strings."""
    code = _code(source)
    lines = source.splitlines()
    kept = code.split("\n")
    return "\n".join(line[: len(c)] for line, c in zip(lines, kept, strict=False))


def _open_header(stmt: str) -> bool:
    """Whether ``stmt`` is a for/while/if header still waiting for its ``{``."""
    words = stmt.split(None, 1)
    return bool(words) and words[0] in _BLOCK_WORDS and "{" not in stmt


def _split_statements(source: str) -> Iterator[str]:
    """Statements split at newlines and ``;`` outside brackets and strings; ``#`` comments
    dropped; a newline before ``else`` or before a block header's ``{`` does not split."""
    text = _uncommented(source)
    code = _code(source)
    depth, start = 0, 0
    for k, c in enumerate(code):
        depth += c in "({["
        depth -= c in ")}]"
        if c in ";\n" and depth <= 0:
            if c == "\n":
                head = code[start:k].strip()
                after = code[k + 1 :].lstrip()
                first = head.split(None, 1)[0] if head else ""
                if _open_header(head) or (first == "if" and _ELSE.match(after)):
                    continue
            yield text[start:k]
            start = k + 1
    yield text[start:]


def brace_depth(source: str) -> int:
    """Unclosed ``{`` count of ``source``, comments and strings ignored."""
    code = _code(source)
    return code.count("{") - code.count("}")


def is_else(line: str) -> bool:
    """Whether ``line`` starts with the word ``else``."""
    return bool(_ELSE.match(line.strip()))


def incomplete(source: str) -> bool:
    """Whether ``source`` needs more lines: an unclosed ``{`` or a block header without one."""
    if brace_depth(source) > 0:
        return True
    stmts = [x.strip() for x in _split_statements(source) if x.strip()]
    return bool(stmts) and _open_header(stmts[-1])


def may_continue(source: str) -> bool:
    """Whether an ``else`` on a later line could still extend ``source`` (its last statement
    is a closed if)."""
    stmts = [x.strip() for x in _split_statements(source) if x.strip()]
    if not stmts or stmts[-1].split(None, 1)[0] != "if" or not stmts[-1].endswith("}"):
        return False
    code, depth, tail = _code(stmts[-1]), 0, ""
    for k, c in enumerate(code):
        depth += c == "{"
        depth -= c == "}"
        if depth == 0 and _ELSE.match(code, k) and not code[k - 1 : k].isalnum():
            tail = code[k + 4 :].lstrip()
    return not tail.startswith("{")


def _block(text: str, start: int) -> tuple[str, int]:
    """The body of the ``{...}`` opening at ``text[start]`` and the index after its ``}``."""
    depth, quoted = 0, False
    for k in range(start, len(text)):
        quoted ^= text[k] == '"'
        depth += text[k] == "{" and not quoted
        depth -= text[k] == "}" and not quoted
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


@dataclass(frozen=True, eq=False)
class _Flex(P.State):
    """A state whose ``flex`` registers come from literal kets without ``_d``."""

    flex: tuple[bool, ...] = ()


def _flags(s: P.State) -> tuple[bool, ...]:
    """Per register, whether its dimension may still grow."""
    return s.flex if isinstance(s, _Flex) else (False,) * len(s.widths)


def _flex(s: P.State, flex: Sequence[bool]) -> P.State:
    """``s`` with the flexible registers ``flex``; a plain State when there are none."""
    if len(flex) != len(s.widths) or not any(flex):
        return P.State(s.widths, s.terms, s.bra, s.dims) if isinstance(s, _Flex) else s
    return _Flex(s.widths, s.terms, s.bra, s.dims, tuple(flex))


def _least_dim(value: int, width: int) -> sympy.Integer:
    """The smallest dimension of at least 2 whose ``width`` digits hold ``value``."""
    d = max(2, round(value ** (1 / width)) - 1) if width > 1 else max(2, value + 1)
    while d**width <= value:
        d += 1
    return sympy.Integer(d)


def _promote(s: P.State, k: int, d: sympy.Expr) -> P.State:
    """``s`` with register k in dimension d; LangTypeError when one of its values cannot fit."""
    return _relayout(s, k, s.widths[k], d)


def _relayout(s: P.State, k: int, w: sympy.Expr | None, d: sympy.Expr | None) -> P.State:
    """``s`` with register k as w digits of base d; LangTypeError when a value cannot fit."""
    if w is None or d is None:
        raise LangTypeError("cannot infer a register width; give the ket a dimension |v>_d")
    for t in s.terms:
        v = t.kets[k]
        if not v.is_Integer:
            continue
        if not w.is_Integer and not d.is_Integer:
            raise LangTypeError(f"cannot give a symbolic register dimension {d}")
        cap = d**w if d.is_Integer and w.is_Integer else None
        if cap is not None and v >= cap:
            where = f"dimension {d}" if P.same(w, 1) else f"{w} digits of base {d}"
            raise LangTypeError(f"|{v}> does not fit in {where}")
    widths = s.widths[:k] + (w,) + s.widths[k + 1 :]
    dims = s.dims[:k] + (d,) + s.dims[k + 1 :]
    return _Flex(widths, s.terms, s.bra, dims, _flags(s))


def _merged(s: P.State) -> tuple[sympy.Expr, sympy.Expr] | None:
    """``(total width, dimension)`` of a state with resolved widths and one dimension."""
    d = s.uniform_dim()
    if d is None or any(w is None for w in s.widths):
        return None
    return sympy.Add(*s.widths), d


def _unify(a: P.State, b: P.State) -> tuple[P.State, P.State, list[bool]]:
    """``a`` and ``b`` with flexible registers raised to their counterpart's layout, the
    lower dimension giving way; and the registers that stay flexible. A lone flexible register
    against several registers takes their joint width."""
    if len(a.widths) != len(b.widths):
        for one, many in ((a, b), (b, a)):
            if len(one.widths) == 1 and _flags(one)[0] and _merged(many) is not None:
                w, d = _merged(many)  # type: ignore[misc]
                one = _relayout(one, 0, w, d)
                return (one, b, []) if many is b else (a, one, [])
        return a, b, []
    flags: list[bool] = []
    for k in range(len(a.widths)):
        fa, fb = _flags(a)[k], _flags(b)[k]
        da, db = a.dims[k], b.dims[k]
        wa, wb = a.widths[k], b.widths[k]
        if P.same(da, db) and P.same(wa, wb):
            flags.append(fa and fb)
            continue
        if fa and fb and wa is not None and wb is not None and P.same(wa, wb):
            if int(da) < int(db):  # type: ignore[arg-type]
                a = _promote(a, k, db)
            else:
                b = _promote(b, k, da)
        elif fa and fb and wa is None and wb is None:
            pass
        elif fa and (not fb or wa is None or (wb is not None and P.same(wa, 1))):
            a = _relayout(a, k, wb, db)
        elif fb:
            b = _relayout(b, k, wa, da)
        flags.append(fa and fb)
    return a, b, flags


@dataclass(frozen=True, eq=False)
class _MatrixGate(P.Gate):
    """A gate given by a matrix of ``size`` levels; ``slots`` is its pattern, if any."""

    size: sympy.Expr | None = None
    slots: tuple[tuple[str, str | None], ...] | None = None


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
        zx_style: str = "matplot",
    ) -> None:
        """``args`` are the command-line arguments read by ``arg(k)``; ``check_oracles`` checks
        every gate that maps basis states to basis states for reversibility; ``mode`` picks
        Dirac text, the ZX drawing, both, or the raw graph listing for ``show``; ``zx_style``
        draws ZX diagrams with matplotlib, Unicode or ASCII."""
        if style not in D.STYLES:
            raise ValueError(f"unknown display style {style!r}")
        if zx_style not in zxdraw.STYLES:
            raise ValueError(f"unknown ZX drawing style {zx_style!r}")
        self.zx_style = zx_style
        self.listing = False
        self.style = style
        self.styles: tuple[str, ...] = (style,)
        """The styles each ``show`` is rendered in, in order."""
        self._out: list[tuple[str | None, str]] = []
        self.parts: list[tuple[str | None, str]] = []
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
        self.slots: dict[str, list[tuple[str, str | None]]] = {}
        self.states: dict[str, P.State | P.Mixed] = {}
        self.lists: dict[str, list[tuple[D.Label | None, object]]] = {}
        for stmt in _split_statements(PRELUDE):
            if stmt.strip():
                self._statement(stmt.strip(), builtin=True)
        self.builtins = set(self.ops)

    def execute(self, source: str) -> str:
        """Runs every statement of ``source``; returns what ``show`` and ``print`` printed,
        each ``show`` once per style of :attr:`styles`."""
        return "\n".join(text for _, text in self.execute_parts(source))

    def execute_parts(self, source: str) -> list[tuple[str | None, str]]:
        """Runs ``source``; returns ``(style, text)`` per printed piece, style None for text
        shared by every style. :attr:`parts` keeps the pieces printed before an error."""
        start = len(self._out)
        try:
            self._run(source)
        finally:
            self.parts = self._out[start:]
            del self._out[start:]
        return list(self.parts)

    def _run(self, source: str) -> None:
        for stmt in _split_statements(source):
            text = stmt.strip()
            if text:
                printed = self._statement(text)
                if printed:
                    self._out.append((None, printed))

    def _parser(self, toks: list[Tok]) -> Parser:
        return Parser(
            toks,
            lambda name: name in _FUNCS
            or name in _SPECIAL
            or name in _STATE_FUNCS
            or name in self.funcs,
        )

    def _statement(self, text: str, builtin: bool = False) -> str:
        word = text.split(None, 1)[0]
        if word in ("for", "while", "if"):
            out: str = getattr(self, f"_stmt_{word}")(text)
            self.last = None
            return out
        if _ELSE.match(word):
            raise LangSyntaxError("else without an if before it")
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
        node = p.expr()
        if not p.done():
            raise LangSyntaxError(
                f"unexpected {p.peek().text!r}; a statement starts with var, func, gate, "  # type: ignore[union-attr]
                "list, show, print, measure, seed, for, while, if, assert, assume, |name> =, "
                "<name| = or is an expression"
            )
        if self._apply_in_place(node):
            return ""
        value = self._eval(node, Ctx())
        self.last = Echo(None, self._type(value), value)
        return ""

    def _apply_in_place(self, node: object) -> bool:
        """``U V |name>`` as a statement: the operators applied to the named state, which is
        replaced by the result; False for any other expression."""
        lefts: list[object] = []
        while isinstance(node, Juxt):
            lefts.append(node.left)
            node = node.right
        if not (lefts and isinstance(node, Braket) and not node.bra):
            return False
        name = node.content.strip()
        if not _IDENT.fullmatch(name) or name not in self.states:
            return False
        ops = [self._eval(left, Ctx()) for left in lefts]
        if not all(isinstance(op, P.Operator) for op in ops):
            return False
        value: object = self._eval(node, Ctx())
        for op in reversed(ops):
            try:
                value = self._juxt(op, value)
            except P.PathSumError as exc:
                raise LangError(str(exc)) from exc
        if isinstance(value, P.Mixed):
            self.states[name] = value
            self.last = Echo(self._label("ket", name), "density", value)
            return True
        if not isinstance(value, P.State) or value.bra:
            return False
        self.states[name] = P.simplify(_flex(value, ()))
        self.last = Echo(self._label("ket", name), "ket", self.states[name])
        return True

    def _type(self, v: object) -> str:
        """The type word of a value: ket, density, dist, bra, op, bool, nat, int, real or
        complex."""
        if isinstance(v, P.State):
            return "bra" if v.bra else "ket"
        if isinstance(v, P.Mixed):
            return "density"
        if isinstance(v, P.Distribution):
            return "dist"
        if isinstance(v, P.Operator):
            return "op"
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
        if isinstance(value, P.Distribution):
            certain = P.certain(value)
            if certain is None:
                raise LangTypeError(f"the {what} depends on a measured outcome")
            value = certain
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
        inclusive = p.range_op()
        hi = self._scalar(self._eval(p.add(), Ctx()), "a for bound")
        if not p.done():
            raise LangSyntaxError("usage: for NAME in LO..HI { ... } or LO..=HI")
        if not inclusive:
            hi = hi - 1
        self._check_name(var.text, "var")
        if not (lo.is_Integer and hi.is_Integer):
            self._fold_loop(var.text, lo, hi, body)
            return ""
        for k in range(int(lo), int(hi) + 1):
            self._forget(var.text)
            self.values[var.text] = sympy.Integer(k)
            self._run(body)
        return ""

    def _stmt_while(self, text: str) -> str:
        header, body, rest = self._header(text, "while")
        if rest:
            raise LangSyntaxError(f"unexpected {rest!r} after the while block")
        cond = self._condition(header, "while condition")
        while self._truth(cond, "while condition"):
            self._run(body)
        return ""

    def _stmt_if(self, text: str) -> str:
        header, body, rest = self._header(text, "if")
        if self._truth(self._condition(header, "if condition"), "if condition"):
            self._run(body)
            return ""
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
        self._run(inner)
        return ""

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
                    f"a loop over the symbolic range {lo}..={hi} may only update states (|s> = ...)"
                )
            name = toks[0].text.strip()
            if name not in self.states:
                raise LangTypeError(f"|{name}> is not defined")
            cur = self.states[name]
            if isinstance(cur, P.Mixed):
                raise LangError(
                    f"a loop over the symbolic range {lo}..={hi} cannot update the density "
                    f"matrix |{name}>"
                )
            probes = [
                sympy.Dummy(f"u{k}", integer=True, nonnegative=True) for k in range(len(cur.widths))
            ]
            saved = self.values.get(var)
            self.values[var] = j
            self.states[name] = P.basis_kets(probes, cur.widths, cur.dims)
            ranges = {
                u: (sympy.S.Zero, d**w - 1)
                for u, w, d in zip(probes, cur.widths, cur.dims, strict=True)
                if w is not None and d is not None
            }
            try:
                with P.ranged(ranges):
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
        raw = self._eval(node, Ctx())
        if isinstance(raw, P.Distribution):
            self._forget(name)
            self.values[name] = raw
            return
        value = self._scalar(raw, f"var {name}")
        self._forget(name)
        self.values[name] = value
        if value.free_symbols and not value.is_Symbol:
            self.sugar.append((name, value))

    def _measure(self, node: Braket) -> P.Distribution:
        """Measures every register of the named state ``node``: returns the Born distribution
        and leaves the state as the mixture of its outcomes."""
        label = node.content.strip()
        if node.bra or not _IDENT.fullmatch(label):
            raise LangSyntaxError("measure takes a named ket, e.g. measure|psi>")
        if label not in self.states:
            raise LangTypeError(f"|{label}> is not defined")
        self._check_dim(label, self._dim(node.dim, Ctx()))
        state = self.states[label]
        try:
            dist = P.distribution(state)
            layout = state.layout() if isinstance(state, P.Mixed) else state
            self.states[label] = P.collapse(dist, layout)
        except P.PathSumError as exc:
            raise LangError(str(exc)) from exc
        return P.joined(dist, layout)

    def _stmt_measure(self, toks: list[Tok], text: str, builtin: bool) -> None:
        p = self._parser(toks)
        value = self._eval(p.expr(), Ctx())
        if not p.done():
            raise LangSyntaxError("measure takes exactly one named ket")
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
        source = text.split("=", 1)[1].strip()
        if isinstance(pattern, (MatrixNode, MatrixFor)) and p.done():
            self.ops[name] = self._matrix_gate(name, pattern, Ctx(), None, source)
            self.last = Echo(D.Label("op", name), "gate", self.ops[name])
            return
        p.take("->")
        body = p.expr()
        if not p.done():
            raise LangSyntaxError(f"unexpected {p.peek().text!r} in {name}")  # type: ignore[union-attr]
        slots = self._pattern(pattern)
        names = [q for q, _ in slots]
        if isinstance(body, (MatrixNode, MatrixFor)):
            self.ops[name] = self._matrix_gate(name, body, Ctx(), slots, source)
            self.slots[name] = slots
            self.last = Echo(D.Label("op", name), "gate", self.ops[name])
            return

        def fn(
            vals: tuple[sympy.Expr, ...],
            widths: tuple[sympy.Expr, ...],
            dims: tuple[sympy.Expr, ...],
        ) -> P.State:
            for (q, want), d in zip(slots, dims, strict=True):
                if want is not None and not P.bind(self._dim(want, Ctx()), d):
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
        self.slots[name] = slots
        self.last = Echo(D.Label("op", name), "gate", self.ops[name])

    def _matrix_gate(
        self,
        name: str,
        node: MatrixNode | MatrixFor,
        ctx: Ctx,
        slots: list[tuple[str, str | None]] | None,
        source: str = "",
    ) -> _MatrixGate:
        """The gate ``|c> -> sum_r M[r][c] |r>`` over the registers it is applied to (one
        merged register without ``slots``); entries are evaluated at each application."""
        size = sympy.Integer(len(node.rows)) if isinstance(node, MatrixNode) else None
        if isinstance(node, MatrixFor):
            lo = self._scalar(self._eval_node(node.lo, ctx), "a matrix bound")
            hi = self._scalar(self._eval_node(node.hi, ctx), "a matrix bound")
            if lo != 0:
                raise LangTypeError("a matrix index range starts at 0")
            size = hi + 1 if node.inclusive else hi

        def fn(
            vals: tuple[sympy.Expr, ...],
            widths: tuple[sympy.Expr, ...],
            dims: tuple[sympy.Expr, ...],
        ) -> P.State:
            for (q, want), d in zip(slots or (), dims, strict=False):
                if want is not None and not P.bind(self._dim(want, ctx), d):
                    raise LangTypeError(f"{name}: |{q}>_{want} applied to a dimension-{d} register")
            if any(w is None or d is None for w, d in zip(widths, dims, strict=True)):
                raise LangTypeError(f"{name}: a register width is unresolved")
            levels = [d**w for w, d in zip(widths, dims, strict=True)]
            total = sympy.Mul(*levels)
            if not (P.same(total, size) or P.bind(size, total)):
                raise LangTypeError(f"{name}: a matrix of {size} levels applied to {total} levels")
            col = _mixed(vals, levels)
            if isinstance(node, MatrixFor):
                bounds = [P.new_bound(node.row, 0, n - 1) for n in levels]
                row = _mixed([b.symbol for b in bounds], levels)
                inner = Ctx({**ctx.locals, node.row: Local(row), node.col: Local(col)}, ctx.body)
                amp = self._scalar(self._eval_node(node.body, inner), "a matrix entry")
                term = P.Term(amp, tuple(bounds), tuple(b.symbol for b in bounds))
                return P.State(widths, (term,), False, dims)
            sizes = [int(n) for n in levels]
            rows = [
                [self._scalar(self._eval_node(e, ctx), "a matrix entry") for e in r]
                for r in node.rows
            ]
            terms = []
            for r, entries in enumerate(rows):
                if col.is_Integer:
                    amp = entries[int(col)]
                else:
                    amp = sympy.Add(
                        *(m * sympy.KroneckerDelta(col, c) for c, m in enumerate(entries) if m != 0)
                    )
                if amp != 0:
                    terms.append(P.Term(amp, (), _split(r, sizes)))
            if not terms:
                terms.append(P.Term(sympy.S.Zero, (), _split(0, sizes)))
            return P.State(widths, tuple(terms), False, dims)

        arity = len(slots) if slots else 1
        frozen = tuple(slots) if slots else None
        return _MatrixGate(name, arity, fn, source=source or name, size=size, slots=frozen)

    def _vector(self, node: VectorNode, ctx: Ctx) -> P.State:
        """``[a, b, ...]`` as ``a|0> + b|1> + ...`` on one register of that many levels."""
        amps = [self._scalar(self._eval_node(e, ctx), "a vector entry") for e in node.items]
        if len(amps) < 2:
            raise LangTypeError("a vector needs at least 2 entries")
        w, d = _matrix_layout(sympy.Integer(len(amps)))
        terms = [P.Term(a, (), (sympy.Integer(k),)) for k, a in enumerate(amps) if a != 0]
        if not terms:
            terms = [P.Term(sympy.S.Zero, (), (sympy.S.Zero,))]
        return P.State((w,), tuple(terms), False, (d,))

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
        style = None
        if nodes and isinstance(nodes[-1][0], Name) and nodes[-1][0].id in D.STYLES:
            style = nodes.pop()[0].id  # type: ignore[attr-defined]
        entries: list[tuple[D.Label | None, object]] = []
        for node, source in nodes:
            if isinstance(node, (Str, Fstr)):
                raise LangTypeError(f'show takes values; write print("{node.text}")')
            entries.extend(self._entries(node, source))
        if style is not None:
            return self.render(entries, style)
        for each in self.styles:
            self._out.append((each, self.render(entries, each)))
        return ""

    def _stmt_print(self, toks: list[Tok], text: str, builtin: bool) -> str:
        parts: list[str] = []
        for node, source in self._arguments(toks, text, "print"):
            if isinstance(node, Str):
                parts.append(node.text)
                continue
            if isinstance(node, Fstr):
                parts.append(self._format(node.text))
                continue
            for _, value in self._entries(node, source):
                parts.append(_plain(value, source))
        return " ".join(parts)

    def _format(self, text: str) -> str:
        """An f-string's text with each ``{expr[:spec]}`` field filled in; ``{{`` and ``}}``
        are literal braces."""
        out: list[str] = []
        k = 0
        while k < len(text):
            c = text[k]
            if c in "{}" and text[k + 1 : k + 2] == c:
                out.append(c)
                k += 2
                continue
            if c == "}":
                raise LangSyntaxError("a single } in an f-string; write }}")
            if c != "{":
                out.append(c)
                k += 1
                continue
            depth, j = 1, k + 1
            while j < len(text) and depth:
                depth += text[j] == "{"
                depth -= text[j] == "}"
                j += 1
            if depth:
                raise LangSyntaxError("an unclosed { in an f-string")
            out.append(self._field(text[k + 1 : j - 1]))
            k = j
        return "".join(out)

    def _field(self, field: str) -> str:
        """One f-string field: a concrete value plainly or by its format spec, anything else as
        text."""
        source, spec = field.strip(), None
        head, sep, tail = field.rpartition(":")
        if sep and _SPEC.fullmatch(tail):
            try:
                p = self._parser(tokenize(head))
                p.expr()
                if p.done() and head.strip():
                    source, spec = head.strip(), tail
            except LangSyntaxError:
                pass
        if not source:
            raise LangSyntaxError("an empty {} in an f-string")
        p = self._parser(tokenize(source))
        node = p.expr()
        if not p.done():
            raise LangSyntaxError(f"cannot read {{{source}}} in an f-string")
        texts = []
        r = D.Renderer("ascii" if self.style == "ascii" else "unicode", self._context())
        for _, value in self._entries(node, source):
            if spec is not None:
                texts.append(format(_python(value, source), spec))
                continue
            try:
                texts.append(_plain(value, source))
            except LangTypeError:
                texts.append(r.value(value))
        return ", ".join(texts)

    def _context(self) -> D.Context:
        sugar = sorted(
            ((v, sympy.Symbol(n)) for n, v in self.sugar),
            key=lambda pair: -sympy.count_ops(pair[0]),
        )
        taken = frozenset(self.values) | frozenset(self.states) | frozenset(self.funcs)
        order = {sympy.Symbol(n): k for k, (n, _) in enumerate(self.sugar)}
        return D.Context(
            sugar, taken | frozenset(self.ops), tuple(self.assumptions), order
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
        """Requires a given ``_dim`` to match the state's registers."""
        if dim is None:
            return
        have = self.states[name].uniform_dim()
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
        """Entries as Dirac lines (an image in ``plot`` style, ``$...$`` LaTeX lines in
        ``latex`` style), ZX drawings of states, or both."""
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
            elif style == "latex":
                out.extend(f"${line.strip()}$" for line in math)
            else:
                out.extend(math)
            math.clear()
            return True

        for lab, v in entries:
            state = isinstance(v, (P.State, P.Mixed))
            if not state or mode in (NotationMode.DIRAC, NotationMode.BOTH):
                math.append(r.entry(lab, v, record=not state))
            if state and mode is not NotationMode.DIRAC:
                if not flush():
                    return None
                raw = mode is NotationMode.RAW
                out.append(self._zx_text(lab, v, raw=raw, style=style))  # type: ignore[arg-type]
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
        if lab is not None and lab.kind == "var" and isinstance(v, sympy.Expr):
            return name, e.type, r.expr(v, sympy.Symbol(lab.name))
        if isinstance(v, (P.State, P.Mixed)) and self.mode is not NotationMode.DIRAC:
            listing = self._zx_body(v)
            if self.mode in (NotationMode.ZX, NotationMode.RAW):
                return name, e.type, listing
            return name, e.type, r.value(v) + "\n" + listing
        return name, e.type, r.value(v)

    def _zx_text(
        self, label: D.Label | None, s: P.State, raw: bool = False, style: str | None = None
    ) -> str:
        """``label =`` then the ZX drawing of ``s``, and its listing when :attr:`listing`;
        with ``raw`` only its uncapped listing."""
        style = self.style if style is None else style
        text = "ascii" if style == "ascii" else "unicode"
        head = D.Renderer(text).label(label) if label is not None else ""
        if raw:
            return "\n".join([f"{head} =".lstrip(), self._zx_body(s)])
        parts = [f"{head} =".lstrip(), self._zx_drawing(s, style)]
        if self.listing:
            parts.append(self._zx_body(s))
        return "\n".join(parts)

    def _zx_drawing(self, s: P.State, display: str | None = None) -> str:
        """The ZX drawing of ``s`` in :attr:`zx_style`, ASCII under the ascii display style;
        text drawings indented by two spaces."""
        display = self.style if display is None else display
        style = "ascii" if display == "ascii" else self.zx_style
        try:
            drawing = zxdraw.draw(zxview.view(s), style)
        except (zxview.ZXViewError, zxsym.Unsupported) as exc:
            return f"  no ZX form ({exc})"
        if drawing.startswith("\x1b"):
            return drawing
        return "\n".join("  " + line for line in drawing.splitlines())

    @staticmethod
    def _zx_body(s: P.State | P.Mixed) -> str:
        """The ZX listing of ``s`` indented by two spaces, or ``no ZX form (reason)``."""
        try:
            body = zxview.listing(s)
        except (zxview.ZXViewError, zxsym.Unsupported) as exc:
            return f"  no ZX form ({exc})"
        return "\n".join("  " + line for line in body.splitlines())

    def _entries(self, node: object, source: str = "") -> list[tuple[D.Label | None, object]]:
        if isinstance(node, Braket) and _IDENT.fullmatch(node.content.strip()):
            name = node.content.strip()
            if name in self.states:
                s = self.states[name]
                self._check_dim(name, self._dim(node.dim, Ctx()))
                kind = "bra" if node.bra else "ket"
                if isinstance(s, P.Mixed):
                    if node.bra:
                        raise LangTypeError(f"|{name}> is a density matrix; it has no bra")
                    return [(self._label(kind, name), s)]
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
        if re.fullmatch(r"\d+`?|`", name):
            raise LangSyntaxError(
                f"|{name}> is a literal ket, not a name; a state name starts with a letter or _"
            )
        self._check_name(name, "state")
        p = self._parser(rest)
        if p.done():
            raise LangSyntaxError("missing the value after =")
        value = self._eval(p.expr(), Ctx())
        if not p.done():
            raise LangSyntaxError(f"unexpected {p.peek().text!r}")  # type: ignore[union-attr]
        if isinstance(value, P.Operator) and head.kind == "ket":
            density = P.density_of(value)
            if density is not None:
                value = density
        if isinstance(value, P.Mixed) and head.kind == "ket" and head.dim is None:
            self.states[name] = value
            self.last = Echo(self._label("ket", name), "density", value)
            return
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
        if dim is not None:
            for k, flexible in enumerate(_flags(value)):
                if flexible and not P.same(value.dims[k], dim):
                    value = _promote(value, k, dim)
        have = value.uniform_dim()
        if dim is not None and (have is None or not P.same(have, dim)):
            got = "mixed dimensions" if have is None else f"dimension {have}"
            raise LangTypeError(f"cannot assign a state of {got} to |{name}>_{dim}")
        value = _flex(value, ())
        state = P.simplify(P.dagger(value) if want_bra else value)
        self.states[name] = state
        shown = P.dagger(state) if want_bra else state
        self.last = Echo(self._label(head.kind, name), head.kind, shown)

    @staticmethod
    def _kind(v: object) -> str:
        if isinstance(v, P.State):
            return "a bra" if v.bra else "a ket"
        if isinstance(v, P.Mixed):
            return "a density matrix"
        if isinstance(v, P.Operator):
            return "an operator"
        if isinstance(v, P.Distribution):
            return "a measured distribution"
        return "a scalar"

    def _scalar(self, v: object, where: str) -> sympy.Expr:
        if isinstance(v, sympy.Basic):
            return v
        if isinstance(v, P.Distribution):
            value = P.certain(v)
            if value is not None:
                return value
            raise LangTypeError(
                f"{where} must be a scalar, got a measured distribution with several outcomes"
            )
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
        if isinstance(node, Call) and node.name == "normalise":
            return self._normalise(node, ctx)
        if isinstance(node, MeasureNode):
            return self._measure(node.target)
        if isinstance(node, Call):
            fn = _FUNCS.get(node.name) or self.funcs[node.name]

            def call(*args: object) -> object:
                vals = [self._scalar(a, f"an argument of {node.name}") for a in args]
                try:
                    return fn(*vals)
                except (TypeError, ValueError) as exc:
                    raise LangTypeError(f"{node.name}: {exc}") from exc

            return self._lift(call, [self._eval_node(a, ctx) for a in node.args])
        if isinstance(node, Neg):
            return self._lift(
                lambda v: self._mul(sympy.Integer(-1), v), [self._eval_node(node.arg, ctx)]
            )
        if isinstance(node, (Str, Fstr)):
            raise LangTypeError("a string can only be printed")
        if isinstance(node, Not):
            return self._lift(
                lambda v: sympy.Not(_boolean(self._scalar(v, "not"))),
                [self._eval_node(node.arg, ctx)],
            )
        if isinstance(node, Braket):
            return self._braket(node, ctx)
        if isinstance(node, SumNode):
            return self._sum(node, ctx)
        if isinstance(node, (MatrixNode, MatrixFor)):
            return self._matrix_gate(_matrix_text(node), node, ctx, None)
        if isinstance(node, VectorNode):
            return self._vector(node, ctx)
        if isinstance(node, Juxt):
            return self._juxt(self._eval_node(node.left, ctx), self._eval_node(node.right, ctx))
        if isinstance(node, Bin):
            return self._binary(
                node.op, self._eval_node(node.left, ctx), self._eval_node(node.right, ctx)
            )
        raise LangSyntaxError(f"cannot evaluate {node!r}")

    def _lift(self, fn: Callable[..., object], args: list[object]) -> object:
        """``fn(*args)``, mapped over the outcomes of the one argument that is a measured
        distribution; a certain distribution counts as its value."""
        args = [
            P.certain(a) if isinstance(a, P.Distribution) and P.certain(a) is not None else a
            for a in args
        ]
        dists = [k for k, a in enumerate(args) if isinstance(a, P.Distribution)]
        if not dists:
            return fn(*args)
        if len(dists) > 1:
            raise LangTypeError(
                "cannot combine two measured distributions; their joint distribution is unknown"
            )
        k = dists[0]

        def one(x: sympy.Expr) -> sympy.Expr:
            out = fn(*args[:k], x, *args[k + 1 :])
            return self._scalar(out, "a function of a measured outcome")

        return P.map_outcomes(args[k], one)  # type: ignore[arg-type]

    def _name(self, name: str, ctx: Ctx) -> object:
        if name in ctx.locals:
            return ctx.locals[name].value
        if name in self.values:
            v = self.values[name]
            if isinstance(v, bool):
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
        raise LangError(f"{name!r} is not defined; declare it with var")

    def _braket(self, node: Braket, ctx: Ctx) -> P.State | P.Mixed:
        content = node.content.strip()
        dim = self._dim(node.dim, ctx)
        if _IDENT.fullmatch(content) and content in self.states and content not in ctx.locals:
            self._check_dim(content, dim)
            s = self.states[content]
            if isinstance(s, P.Mixed):
                if node.bra:
                    raise LangTypeError(f"|{content}> is a density matrix; it has no bra")
                return s
        elif content in _NAMED_STATES and not (content == "i" and "i" in ctx.locals):
            if dim is not None and not P.same(dim, 2):
                raise LangTypeError(f"|{content}> is a qubit state")
            one = P.scale(_NAMED_STATES[content], P.basis(1, 1))
            s = P.scale(1 / sympy.sqrt(2), P.add(P.basis(0, 1), one))
        elif _DIGIT_STRING.fullmatch(content):
            if content == "`":
                raise LangSyntaxError("|`> is not a ket; write digits before the `, e.g. |01`>")
            digit_kets = [self._basis(c, ctx, dim) for c in content[:-1]]
            s = digit_kets[0]
            for part in digit_kets[1:]:
                s = _flex(P.tensor(s, part), _flags(s) + _flags(part))
        else:
            if not content:
                raise LangSyntaxError("an empty ket")
            if len(_split_items(content)) > 1:
                parts = "".join(f"|{x}>" for x in _split_items(content))
                raise LangSyntaxError(f"|{content}> is not a ket; write {parts}")
            s = self._basis(content, ctx, dim)
        return _flex(P.dagger(s), _flags(s)) if node.bra else s

    def _basis(self, text: str, ctx: Ctx, dim: sympy.Expr | None) -> P.State:
        p = self._parser(tokenize(text))
        ast = p.expr()
        if p.at(":"):
            raise LangSyntaxError(
                f"|{text}> is not a ket; write |v> (raised to fit where it is used), |v>_d "
                "or |v>_d^@n"
            )
        if not p.done():
            raise LangSyntaxError(f"cannot read the ket |{text}>")
        value = self._scalar(self._eval_node(ast, ctx), f"|{text}>")
        if value.is_integer is False or (value.is_Integer and value < 0):
            raise LangTypeError(f"|{text}> needs a non-negative integer value")
        local = ctx.locals.get(ast.id) if isinstance(ast, Name) else None
        explicit = dim is not None
        if dim is None and local is not None and local.dim is not None:
            dim = local.dim
        literal = local is None or (local.width is None and local.bound is None)
        if dim is None and not ctx.body and literal:
            if value.is_Integer:
                return _flex(P.basis(value, sympy.Integer(1), _least_dim(int(value), 1)), [True])
            return _flex(P.basis(value, None, None), [True])
        if dim is None and not ctx.body:
            dim = P.TWO
        width: sympy.Expr | None = None
        if local is not None and local.width is not None:
            width = local.width
        elif local is not None and local.bound is not None:
            width = _range_width(local.bound.lo, local.bound.hi, dim if dim is not None else P.TWO)
        elif explicit:
            width = sympy.Integer(1)
        if width is None and not ctx.body:
            raise LangTypeError(f"cannot infer the register width of |{text}>; write |{text}>_d")
        concrete = dim is not None and width is not None and _ints(value, width, dim)
        if concrete and int(value) >= int(dim) ** int(width):  # type: ignore[arg-type]
            digits = "one digit" if width == 1 else f"{width} digits"
            raise LangTypeError(f"|{text}>: {value} does not fit in {digits} of base {dim}")
        return P.basis(value, width, dim)

    def _sum(self, node: SumNode, ctx: Ctx) -> object:
        lo = self._scalar(self._eval_node(node.lo, ctx), "a sum bound")
        hi = self._scalar(self._eval_node(node.hi, ctx), "a sum bound")
        if not node.inclusive:
            hi = hi - 1
        b = P.new_bound(node.var, lo, hi)
        body = self._eval_node(node.body, ctx.child(node.var, Local(b.symbol, bound=b)))
        if isinstance(body, P.State):
            terms = tuple(P.Term(t.amp, (b,) + t.bound, t.kets) for t in body.terms)
            return P.simplify(P.State(body.widths, terms, body.bra, body.dims))
        if isinstance(body, sympy.Basic):
            return sympy.Sum(body, (b.symbol, lo, hi)).doit()
        raise LangTypeError("a sum body must be a state or a scalar")

    def _mixed_binary(self, op: str, a: object, b: object) -> object:
        """``+ - * /`` on density matrices: sums of equal layouts and scalar multiples."""
        if op in ("+", "-") and isinstance(a, P.Mixed) and isinstance(b, P.Mixed):
            other = b.vec if op == "+" else P.scale(sympy.Integer(-1), b.vec)
            return P.Mixed(P.simplify(P.add(a.vec, other)))
        if op == "*":
            return self._mul(a, b)
        if op == "/" and isinstance(b, sympy.Basic):
            return self._mul(a, 1 / b)
        raise LangTypeError(f"{op} on {self._kind(a)} and {self._kind(b)} is not defined")

    def _mul(self, a: object, b: object) -> object:
        if isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic):
            return a * b
        if isinstance(a, sympy.Basic):
            a, b = b, a
        if isinstance(b, sympy.Basic):
            if isinstance(a, P.Mixed):
                return P.Mixed(P.scale(b, a.vec))
            if isinstance(a, P.State):
                return _flex(P.scale(b, a), _flags(a))
            if isinstance(a, P.Operator):
                return P.Scaled(b, a)
        raise LangTypeError(f"cannot multiply {self._kind(a)} by {self._kind(b)}")

    def _juxt(self, a: object, b: object) -> object:
        if isinstance(a, P.Distribution) or isinstance(b, P.Distribution):
            return self._lift(self._juxt, [a, b])
        if isinstance(a, sympy.Basic) or isinstance(b, sympy.Basic):
            return self._mul(a, b)
        if isinstance(b, P.Mixed):
            if not isinstance(a, P.Operator):
                raise LangTypeError(f"cannot apply {self._kind(a)} to a density matrix")
            return P.apply_mixed(a, b)
        if isinstance(a, P.Mixed):
            raise LangTypeError("a density matrix is acted on from the left: U|rho>")
        if isinstance(a, P.Operator) and isinstance(b, P.State):
            if b.bra:
                raise LangTypeError("an operator applies to a ket; write U|psi>")
            b = self._fit_gate(a, b)
            if any(w is None for w in b.widths):
                raise LangTypeError("cannot infer a register width; give the ket a dimension |v>_d")
            try:
                return _collect(P.apply(_fit_outer(a, b), b))
            except P.PathSumError:
                power = _power_of_digit(a, b)
                if power is None:
                    raise
                return power
        if isinstance(a, P.Operator) and isinstance(b, P.Operator):
            return P.Compose((a, b))
        if isinstance(a, P.State) and isinstance(b, P.State):
            if a.bra and not b.bra:
                a, b, _ = _unify(a, b)
                return P.inner(a, b)
            if not a.bra and b.bra:
                return P.Outer(a, b)
            return _flex(P.tensor(a, b), _flags(a) + _flags(b))
        raise LangTypeError(f"cannot apply {self._kind(a)} to {self._kind(b)}")

    def _fit_gate(self, op: P.Operator, s: P.State) -> P.State:
        """``s`` with flexible registers raised to the layout a gate's pattern or matrix
        names; an unresolved one becomes one digit of the slot's dimension (2 by default)."""
        if isinstance(op, _MatrixGate) and op.slots is None:
            return _fit_matrix(op, s)
        if not isinstance(op, P.Gate):
            return s
        known = self.slots.get(op.name)
        slots: Sequence[tuple[str, str | None]]
        if isinstance(op, _MatrixGate) and op.slots is not None:
            slots = op.slots
        elif known is None or self.ops.get(op.name) is not op:
            slots = [("", None)] * op.arity
        else:
            slots = known
        if len(slots) != len(s.widths):
            return s
        for k, (_, want) in enumerate(slots):
            d = self._dim(want, Ctx())
            if not _flags(s)[k]:
                continue
            if s.widths[k] is None:
                s = _relayout(s, k, sympy.Integer(1), d if d is not None else P.TWO)
            elif d is not None and d.is_Integer and not P.same(d, s.dims[k]):
                s = _promote(s, k, d)
        return s

    def _normalise(self, node: Call, ctx: Ctx) -> P.State:
        """``normalise(s)``: the state ``s`` divided by its norm."""
        if len(node.args) != 1:
            raise LangTypeError("usage: normalise(STATE)")
        s = self._eval_node(node.args[0], ctx)
        if isinstance(s, P.Mixed):
            tr = P.tidy(P.trace(s))
            if tr == 0:
                raise LangTypeError("cannot normalise the zero matrix")
            return P.Mixed(P.simplify(P.scale(1 / tr, s.vec)))  # type: ignore[return-value]
        if not isinstance(s, P.State):
            raise LangTypeError(f"normalise takes a state, got {self._kind(s)}")
        ket = P.dagger(s) if s.bra else s
        norm = P.tidy(sympy.sqrt(P.inner(P.dagger(ket), ket)))
        if norm == 0:
            raise LangTypeError("cannot normalise the zero vector")
        return _flex(P.simplify(P.scale(1 / norm, s)), _flags(s))

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
        if isinstance(a, P.Distribution) or isinstance(b, P.Distribution):
            return self._lift(lambda x, y: self._binary(op, x, y), [a, b])
        if isinstance(a, P.Mixed) or isinstance(b, P.Mixed):
            return self._mixed_binary(op, a, b)
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
                a, b, flags = _unify(a, b)
                total = P.simplify(P.add(a, b))
                return _flex(total if any(flags) else _collect(total), flags)
            if isinstance(a, P.Operator) and isinstance(b, P.Operator):
                return P.OpSum((a, b))
            raise LangTypeError(f"cannot add {self._kind(a)} and {self._kind(b)}")
        if op == "@":
            if isinstance(a, P.State) and isinstance(b, P.State):
                return _flex(P.tensor(a, b), _flags(a) + _flags(b))
            if isinstance(a, P.Operator) and isinstance(b, P.Operator):
                return P.Tensor((a, b))
            raise LangTypeError(f"cannot tensor {self._kind(a)} with {self._kind(b)}")
        if op == "^@":
            n = self._scalar(b, "a tensor power")
            if isinstance(a, P.State):
                power = P.power_state(a, n)
                return _flex(power, _flags(a) * int(n) if n.is_Integer else ())
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


def _power_of_digit(op: P.Operator, s: P.State) -> P.State | None:
    """``U^@n |v>^@n`` as ``(U|v>)^@n`` for a one-register gate U and a repeated digit v."""
    if not (isinstance(op, P.Power) and isinstance(op.op, P.Gate) and op.op.arity == 1):
        return None
    if len(s.widths) != 1 or len(s.terms) != 1 or s.terms[0].bound:
        return None
    w, d, t = s.widths[0], s.dims[0], s.terms[0]
    digit = None if w is None or d is None else D._repeated_digit(t.kets[0], w, d)
    if digit is None or not P.same(op.count, w):
        return None
    one = _collect(P.apply(op.op, P.basis(digit, sympy.Integer(1), d)))
    return P.scale(t.amp, P.power_state(one, w))


def _collect(s: P.State) -> P.State:
    """``s`` with the unsummed terms on equal kets added together."""
    if isinstance(s, _Flex) or sum(not t.bound for t in s.terms) < 2:
        return s
    amps: dict[tuple[sympy.Expr, ...], sympy.Expr] = {}
    rest: list[P.Term] = []
    for t in s.terms:
        if t.bound:
            rest.append(t)
        else:
            amps[t.kets] = amps.get(t.kets, sympy.S.Zero) + t.amp
    terms = [P.Term(P.tidy(a), (), k) for k, a in amps.items()]
    terms = [t for t in terms if t.amp != 0] + rest
    if not terms:
        terms = [P.Term(sympy.S.Zero, (), s.terms[0].kets)]
    return P.State(s.widths, tuple(terms), s.bra, s.dims)


def _matrix_text(node: MatrixNode | MatrixFor) -> str:
    """A short name for an anonymous matrix."""
    if isinstance(node, MatrixFor):
        return f"matrix({node.row}, {node.col})"
    return f"M{len(node.rows)}"


def _fit_outer(op: P.Operator, s: P.State) -> P.Operator:
    """``op`` with the flexible kets and bras of its outer products laid out like ``s``."""
    if isinstance(op, P.Outer):
        bra, _, _ = _unify(op.bra, s)
        ket, _, _ = _unify(op.ket, s)
        return P.Outer(ket, bra)
    if isinstance(op, P.Scaled):
        return P.Scaled(op.factor, _fit_outer(op.op, s))
    if isinstance(op, P.OpSum):
        return P.OpSum(tuple(_fit_outer(o, s) for o in op.ops))
    if isinstance(op, P.Compose):
        return P.Compose(tuple(_fit_outer(o, s) for o in op.ops))
    return op


def _matrix_layout(size: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr]:
    """``(width, dim)`` of one register with ``size`` levels: qubits for a power of 2."""
    power = sympy.powsimp(sympy.sympify(size))
    if power.is_Integer and int(power) >= 2 and int(power) & (int(power) - 1) == 0:
        return sympy.Integer(int(power).bit_length() - 1), P.TWO
    if power.is_Pow and P.same(power.base, 2):
        return power.exp, P.TWO
    return sympy.Integer(1), power


def _fit_matrix(op: _MatrixGate, s: P.State) -> P.State:
    """``s`` with a lone flexible register laid out to the matrix size."""
    if len(s.widths) != 1 or not _flags(s)[0] or op.size is None:
        return s
    w, d = s.widths[0], s.dims[0]
    if w is not None and d is not None and P.same(d**w, op.size):
        return s
    return _relayout(s, 0, *_matrix_layout(op.size))


def _mixed(values: Sequence[sympy.Expr], levels: Sequence[sympy.Expr]) -> sympy.Expr:
    """The index of ``values`` read as digits of the radices ``levels``, first most
    significant."""
    total: sympy.Expr = sympy.Integer(0)
    for v, n in zip(values, levels, strict=True):
        total = total * n + v
    return total


def _split(index: int, levels: Sequence[int]) -> tuple[sympy.Integer, ...]:
    """``index`` as one digit per radix of ``levels``, first most significant."""
    out = []
    for n in reversed(levels):
        index, digit = divmod(index, n)
        out.append(sympy.Integer(digit))
    return tuple(reversed(out))


def _plain(v: object, source: str) -> str:
    """A concrete value as plain text: true/false, integers, fractions or decimals; a measured
    distribution as its outcome list."""
    if isinstance(v, D.Text):
        return v.text
    if isinstance(v, P.Distribution):
        return D.Renderer("unicode").value(v)
    if isinstance(v, (bool, sympy.logic.boolalg.BooleanAtom)):
        return "true" if v else "false"
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


_SPEC = re.compile(r"(?:[^{}]?[<>=^])?[-+ ]?z?#?0?\d*[,_]?(?:\.\d+)?[bcdeEfFgGnosxX%]?")
"""A Python format spec."""


def _python(v: object, source: str) -> object:
    """A concrete value as a Python bool, int, float or complex."""
    if isinstance(v, (bool, sympy.logic.boolalg.BooleanAtom)):
        return bool(v)
    if isinstance(v, P.Distribution) and P.certain(v) is not None:
        v = P.certain(v)
    if isinstance(v, sympy.Basic) and v.is_number:
        if v.is_Integer:
            return int(v)
        z = complex(v.evalf())
        return z.real if z.imag == 0 else z
    raise LangTypeError(f"a format spec needs a concrete value; {source} is not one")


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
