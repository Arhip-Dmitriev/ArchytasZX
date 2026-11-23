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

"""One Lean file of the instance exports of a claim with finitely many values, closed by the
case split over those values."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

_LIBRARY_ROOT = "ArchytasZX"


class FiniteUnsupported(ValueError):
    """An instance export without a final theorem."""


def _scan(lines: Sequence[str]) -> tuple[list[str], str, str]:
    """The top-level namespaces of ``lines``, and the last theorem's namespace and name."""
    stack: list[str] = []
    roots: list[str] = []
    last_ns, last_name = "", ""
    for ln in lines:
        head = ln.split()
        if not head or ln != ln.lstrip():
            continue
        if head[0] == "namespace" and len(head) > 1:
            if not stack:
                roots.append(head[1])
            stack.append(head[1])
        elif head[0] == "section" or head[:2] == ["noncomputable", "section"]:
            stack.append("")
        elif head[0] == "end" and stack:
            stack.pop()
        elif head[0] == "theorem" and len(head) > 1:
            last_ns = ".".join(s for s in stack if s)
            last_name = head[1]
    return roots, last_ns, last_name


def _statement(source: str, name: str) -> str:
    """The proposition of theorem ``name`` in ``source``: ``∀ BINDERS, TYPE``."""
    start = None
    for start in re.finditer(rf"^theorem {re.escape(name)}\b", source, re.MULTILINE):
        pass
    if start is None:
        raise FiniteUnsupported(f"no theorem {name}")
    text = source[start.end() :]
    depth = 0
    colon = -1
    for k, ch in enumerate(text):
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        elif depth == 0 and text.startswith(":=", k):
            body = " ".join(text[colon + 1 : k].split())
            binders = " ".join(text[:colon].split())
            return f"∀ {binders}, {body}" if binders else body
        elif depth == 0 and ch == ":" and colon < 0:
            colon = k
    raise FiniteUnsupported(f"theorem {name} has no body")


def nest(source: str, tag: str) -> tuple[str, list[str], str, str, str]:
    """``source`` without its imports, each top-level namespace ``N`` renamed ``N.tag`` (its
    qualified references too) and its macros local; its imports, the namespace and the name of
    its final theorem, and that theorem's proposition ``∀ BINDERS, TYPE``."""
    lines = source.splitlines()
    imports = [ln for ln in lines if ln.startswith("import ")]
    body = [ln for ln in lines if not ln.startswith("import ")]
    roots, last_ns, last_name = _scan(body)
    if not last_name:
        raise FiniteUnsupported("an instance export states no theorem")
    prop = _statement("\n".join(body), last_name)
    plain = sorted({r for r in roots if r != _LIBRARY_ROOT}, key=len)
    keep = [r for r in plain if not any(r.startswith(f"{o}.") for o in plain if o != r)]

    def rename(text: str) -> str:
        for r in keep:
            text = re.sub(rf"(?<![\w.]){re.escape(r)}(?![\w])", f"{r}.{tag}", text)
        return text

    out: list[str] = []
    for ln in body:
        if _LIBRARY_ROOT in roots and ln in (f"namespace {_LIBRARY_ROOT}", f"end {_LIBRARY_ROOT}"):
            ln = f"{ln}.{tag}"
        ln = rename(ln)
        if ln.startswith('macro "'):
            ln = "local " + ln
        out.append(ln)
    ns = rename(last_ns)
    if ns == _LIBRARY_ROOT and _LIBRARY_ROOT in roots:
        ns = f"{_LIBRARY_ROOT}.{tag}"
    return "\n".join(out).strip(), imports, ns, last_name, rename(prop)


def _opens(body: str) -> list[str]:
    """The distinct top-level ``open`` lines of ``body``."""
    return list(dict.fromkeys(ln for ln in body.splitlines() if ln.startswith("open ")))


def _uniform(props: Sequence[str], values: Sequence[int], var: str) -> str | None:
    """The common proposition of ``props`` with each instance's value written ``var``, or
    None when they differ or one names no value."""
    common: str | None = None
    for prop, v in zip(props, values, strict=True):
        out, hits = re.subn(rf"\({v} : (?:ℕ|Nat)\)", var, prop)
        if not hits or (common is not None and out != common):
            return None
        common = out
    return common


def combine(
    instances: Sequence[tuple[Mapping[str, Fraction], str]],
    integral: Mapping[str, bool],
    text: str,
    var: Callable[[str], str],
    shared: bool,
) -> str:
    """The instance exports with namespaces tagged ``I1``, ``I2``, ..., then ``main_finite``
    in the first instance's namespace. When ``shared`` (every instance a corollary of one
    family proof) and one finite symbol has natural values over which the propositions agree:
    that proposition for every value, by ``rcases`` on the values (``omega`` from the bounds).
    Otherwise each instance's proposition as ``finite_claim`` in its namespace, and
    ``main_finite`` their conjunction."""
    if not instances:
        raise FiniteUnsupported("no instance")
    bodies: list[str] = []
    imports: list[str] = []
    spaces: list[str] = []
    names: list[str] = []
    props: list[str] = []
    for k, (_at, source) in enumerate(instances, 1):
        body, imp, ns, name, prop = nest(source, f"I{k}")
        if not ns:
            raise FiniteUnsupported("an instance's final theorem is outside a namespace")
        bodies.append(body)
        imports += [i for i in imp if i not in imports]
        spaces.append(ns)
        names.append(f"{ns}.{name}")
        props.append(prop)
    symbols = list(instances[0][0])
    split: str | None = None
    if shared and len(symbols) == 1 and integral.get(symbols[0], False):
        sym = symbols[0]
        values = [int(at[sym]) for at, _ in instances]
        v = var(sym)
        common = _uniform(props, values, v)
        if common is not None and len(set(values)) == len(values):
            lo, hi = min(values), max(values)
            cases = " ∨ ".join(f"{v} = {x}" for x in values)
            if sorted(values) == list(range(lo, hi + 1)):
                hyp = f"{lo} ≤ {v} → {v} ≤ {hi} → "
                intro = f"  intro {v} h_lo h_hi\n  have h : {cases} := by omega\n"
            else:
                hyp = f"{cases} → "
                intro = f"  intro {v} h\n"
            alts = " | ".join("rfl" for _ in values)
            split = f"theorem main_finite : ∀ {v} : ℕ, {hyp}{common} := by\n{intro}"
            split += f"  rcases h with {alts}\n"
            split += "".join(f"  · exact {n}\n" for n in names)
    parts = ["", *imports, ""]
    for k, body in enumerate(bodies):
        parts += [body, ""]
        if split is None:
            parts += [
                f"namespace {spaces[k]}",
                *_opens(body),
                "",
                f"/-- Instance {k + 1}'s claim. -/",
                f"def finite_claim : Prop :=\n  {props[k]}",
                "",
                f"theorem finite_holds : finite_claim := {names[k]}",
                "",
                f"end {spaces[k]}",
                "",
            ]
    parts += [f"namespace {spaces[0]}", *_opens(bodies[0]), ""]
    if split is not None:
        how = "the case split over its values"
        parts.append(split)
    else:
        how = "the conjunction of the instance claims"
        claims = " ∧ ".join(f"{ns}.finite_claim" for ns in spaces)
        holds = ", ".join(f"{ns}.finite_holds" for ns in spaces)
        parts.append(f"theorem main_finite :\n    {claims} :=\n  ⟨{holds}⟩\n")
    parts += [f"end {spaces[0]}", ""]
    note = (
        f"/-\n  ArchytasZX proof export: restricted to {text}, decided instance by instance; "
        f"each\n  instance's export with its namespaces tagged I<k>, then main_finite by {how}."
        "\n-/"
    )
    return note + "\n".join(parts)
