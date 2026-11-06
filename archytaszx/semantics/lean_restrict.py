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

"""Lean exports of restricted theorems.

:func:`corollary` reads the final theorem of an export (``main_diagrams`` when present, else
the named theorem): its binders and statement, split at the top-level ``:=``. The corollary
states that theorem at each clause of a restriction: ``n = v`` drops the binder of ``n`` and
writes ``v`` for it, ``n >= c`` adds the hypothesis ``c ≤ n`` after it, ``n = a*m + c`` binds
``m`` and writes ``(a * m + c)``, ``d = v`` writes ``v`` for ``d``; a hypothesis binder left
without free variables is discharged by ``norm_num``. Its proof applies the final theorem.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from archytaszx.semantics.restrict import DIMENSION, EQ, GE, Clause

_OPEN = "([{⟨"
_CLOSE = ")]}⟩"


class RestrictUnsupported(Exception):
    """An export whose final theorem the corollary does not read."""


@dataclass(frozen=True, slots=True)
class _Binder:
    """One binder group ``(names : type)``, its bracket kept."""

    open: str
    names: tuple[str, ...]
    type: str

    def text(self) -> str:
        close = _CLOSE[_OPEN.index(self.open)]
        return f"{self.open}{' '.join(self.names)} : {self.type}{close}"


def _group(text: str, start: int) -> int:
    """The index just past the bracket group opening at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
            if depth == 0:
                return i + 1
    raise RestrictUnsupported("unbalanced brackets in the theorem")


def _binders(text: str) -> tuple[list[_Binder], str]:
    """The leading binder groups of ``text`` and the rest."""
    out: list[_Binder] = []
    rest = text.lstrip()
    while rest and rest[0] in "({[":
        end = _group(rest, 0)
        inner = rest[1 : end - 1]
        names, sep, kind = inner.partition(":")
        if not sep:
            raise RestrictUnsupported(f"a binder without a type: {rest[:end]}")
        out.append(_Binder(rest[0], tuple(names.split()), kind.strip()))
        rest = rest[end:].lstrip()
    return out, rest


def _split_proof(text: str) -> str:
    """``text`` up to its first ``:=`` outside brackets."""
    depth = 0
    for i, ch in enumerate(text):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        elif depth == 0 and text.startswith(":=", i):
            return text[:i]
    raise RestrictUnsupported("the theorem has no proof")


def _final(source: str, name: str) -> tuple[str, list[_Binder], str]:
    """The final theorem's name, all its binders (signature, then a leading ``∀``) and its
    statement body."""
    for candidate in (f"{name}_diagrams", name):
        match = re.search(
            rf"^theorem {re.escape(candidate)}\b(.*)", source, re.MULTILINE | re.DOTALL
        )
        if match is not None:
            break
    else:
        raise RestrictUnsupported(f"no theorem {name} in the export")
    head = _split_proof(match[1])
    binders, rest = _binders(head)
    if not rest.startswith(":"):
        raise RestrictUnsupported("the theorem's statement does not follow its binders")
    body = rest[1:].strip()
    if body.startswith("∀"):
        more, after = _binders(body[1:])
        if not more:
            plain = re.match(r"\s*([^:,()]+?)\s*:\s*([^,]+?)\s*,", body[1:])
            if plain is None:
                raise RestrictUnsupported("a ∀ statement this reader does not parse")
            more = [_Binder("(", tuple(plain[1].split()), plain[2])]
            after = body[1 + plain.end() - 1 :]
        if not after.startswith(","):
            raise RestrictUnsupported("a ∀ statement without a comma")
        binders += more
        body = after[1:].strip()
    return candidate, binders, body


def _sub(text: str, var: str, value: str) -> str:
    return re.sub(rf"(?<![\w'.₀-₉]){re.escape(var)}(?![\w'₀-₉])", value, text)


def corollary(
    source: str,
    clauses: Sequence[Clause],
    name: str = "main",
    var: Callable[[str], str] = lambda s: s,
) -> str:
    """``source`` followed by the final theorem stated at ``clauses`` (see the module
    docstring)."""
    final, binders, body = _final(source, name)
    values: dict[str, str] = {}
    after: dict[str, list[_Binder]] = {}
    fresh: dict[str, str] = {}
    for clause in clauses:
        if not clause.substitutes:
            continue
        v = var(clause.symbol)
        if clause.form == EQ:
            values[v] = str(clause.c)
        elif clause.kind != DIMENSION and clause.form == GE:
            after[v] = [_Binder("(", (f"h_{v}_ge",), f"{clause.c} ≤ {v}")]
        else:
            m = var(clause.fresh)
            fresh[v] = m
            values[v] = f"({clause.a} * {m} + {clause.c})"
    stated: list[_Binder] = []
    args: list[str] = []
    for b in binders:
        kind = b.type
        for v, value in values.items():
            kind = _sub(kind, v, value)
        closed = b.open == "(" and kind != b.type and not re.search(r"[A-Za-z_]", kind)
        kept: list[str] = []
        for n in b.names:
            if closed:
                args.append("(by norm_num)")
                continue
            if n in fresh:
                kept.append(fresh[n])
                arg = values[n]
            elif n in values:
                arg = values[n]
            else:
                kept.append(n)
                arg = n
            if b.open == "(":
                args.append(arg)
        if kept:
            stated.append(_Binder(b.open, tuple(kept), kind))
        for n in b.names:
            stated.extend(after.get(n, []))
    for v, value in values.items():
        body = _sub(body, v, value)
    label = "at" if all(c.form == EQ for c in clauses if c.substitutes) else "restricted"
    text = ", ".join(c.text() for c in clauses)
    sig = " ".join(b.text() for b in stated)
    decl = (
        f"/-- The final theorem at {text}. -/\n"
        f"theorem {name}_{label}{' ' + sig if sig else ''} :\n    {body} :=\n"
        f"  {final}{' ' + ' '.join(args) if args else ''}\n"
    )
    marker = re.search(r"^end [^\n]*\s*\Z", source, re.MULTILINE)
    if marker is None:
        return f"{source.rstrip()}\n\n{decl}"
    return f"{source[: marker.start()].rstrip()}\n\n{decl}\n{source[marker.start() :]}"


def header(source: str, note: str) -> str:
    """``source`` with ``note`` added as the last lines of its leading comment."""
    if source.startswith("/-"):
        end = source.index("-/")
        lines = "".join(f"  {line}\n" for line in note.splitlines())
        return f"{source[:end].rstrip()}\n{lines}{source[end:]}"
    return f"/-\n{note}\n-/\n{source}"


def shift_corollary(source: str, clause: Clause, name: str = "main") -> str:
    """``source`` followed by its final theorem at ``n = k + c`` restated over ``n``: the binder
    of ``k`` (the last one) replaced by ``n`` and the hypothesis ``c ≤ n``, proved by writing
    ``n`` as ``k + c``."""
    final, binders, body = _final(source, name)
    if not binders or binders[-1].open != "(" or len(binders[-1].names) != 1:
        raise RestrictUnsupported("the final theorem does not end in its step binder")
    k = binders[-1].names[0]
    if clause.a != 1:
        sig = " ".join(b.text() for b in binders)
        applied = " ".join(x for b in binders if b.open == "(" for x in b.names)
        decl = (
            f"/-- The final theorem at {clause.text()}, over {k}. -/\n"
            f"theorem {name}_restricted {sig} :\n    {body} :=\n  {final} {applied}\n"
        )
        return _before_end(source, decl)
    taken = {x for b in binders for x in b.names}
    n = clause.symbol if clause.symbol not in taken else f"{clause.symbol}'"
    plus = f"({k} + {clause.c})"
    if plus not in body:
        raise RestrictUnsupported("the final theorem is not stated at k + c")
    stated = [
        *binders[:-1],
        _Binder("(", (n,), "ℕ"),
        _Binder("(", (f"h_{n}_ge",), f"{clause.c} ≤ {n}"),
    ]
    args = [x for b in binders[:-1] if b.open == "(" for x in b.names]
    sig = " ".join(b.text() for b in stated)
    decl = (
        f"/-- The final theorem at {clause.text()}, over {n}. -/\n"
        f"theorem {name}_restricted {sig} :\n    {body.replace(plus, n)} := by\n"
        f"  obtain ⟨{k}, rfl⟩ : ∃ {k}, {n} = {k} + {clause.c} := ⟨{n} - {clause.c}, by omega⟩\n"
        f"  exact {final} {' '.join([*args, k])}\n"
    )
    return _before_end(source, decl)


def _before_end(source: str, decl: str) -> str:
    """``source`` with ``decl`` before its closing ``end``."""
    marker = re.search(r"^end [^\n]*\s*\Z", source, re.MULTILINE)
    if marker is None:
        return f"{source.rstrip()}\n\n{decl}"
    return f"{source[: marker.start()].rstrip()}\n\n{decl}\n{source[marker.start() :]}"
