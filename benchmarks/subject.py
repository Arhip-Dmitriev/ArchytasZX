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

"""Subject files, equations, versions and the engine script of one equation at one version. A
subject file is a module in the catalogue directory defining ``SUBJECT``, a :class:`Subject`. An
:class:`Equation` holds the circuit (``(n, d) -> ir.Circuit``), the specification (``SpecContext
-> DSL body``) and optionally a reference circuit realising the specification for circuit-only
tools; ``literal`` writes every dimension as a literal. ``form`` says how n enters the circuit:
``boxed`` (a register count, symbolic allowed), ``unrolled`` (an int, one wire per qudit) or
``fixed`` (no n)."""

from __future__ import annotations

import fnmatch
import functools
import importlib.util
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from benchmarks import ir, translate

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("algorithm", "code", "protocol", "circuit")
VERSIONS = ("symbolic", "concrete", "restricted")


def is_prime(v: int) -> bool:
    """Whether ``v`` is prime."""
    if v < 2:
        return False
    k = 2
    while k * k <= v:
        if v % k == 0:
            return False
        k += 1
    return True


D_CONDITIONS: dict[str, Callable[[int], bool]] = {
    "any": lambda d: d >= 1,
    "ge2": lambda d: d >= 2,
    "prime": is_prime,
    "odd": lambda d: d % 2 == 1,
    "odd_prime": lambda d: d % 2 == 1 and is_prime(d),
    "even": lambda d: d % 2 == 0,
}


@dataclass(frozen=True)
class Domain:
    """Where a claim is true: n in [n_min, n_max] (step ``n_step`` from n_min), d in [d_min,
    d_max] satisfying ``d_condition``."""

    n_min: int = 0
    n_max: int | None = None
    n_step: int = 1
    d_min: int = 1
    d_max: int | None = None
    d_condition: str = "any"

    def has_n(self, n: int) -> bool:
        """Whether ``n`` is in the domain."""
        if n < self.n_min or (self.n_max is not None and n > self.n_max):
            return False
        return (n - self.n_min) % self.n_step == 0

    def has_d(self, d: int) -> bool:
        """Whether ``d`` is in the domain."""
        if d < self.d_min or (self.d_max is not None and d > self.d_max):
            return False
        return D_CONDITIONS[self.d_condition](d)

    def __contains__(self, point: tuple[int, int]) -> bool:
        return self.has_n(point[0]) and self.has_d(point[1])


@dataclass(frozen=True)
class SpecContext:
    """How a specification writes symbols: ``D`` the node dimension token (``d`` or
    ``literal 5``), ``d`` the dimension in phases and scalars, ``n`` the count in expressions,
    ``N`` the box multiplicity token, ``nv`` and ``dv`` the ints (None when symbolic)."""

    D: str
    d: str
    n: str
    N: str
    nv: int | None
    dv: int | None


@dataclass(frozen=True)
class Equation:
    """One equation of a subject (see the module docstring)."""

    key: str
    circuit: Callable[[Any, Any], ir.Circuit]
    specification: Callable[[SpecContext], str]
    form: str = "boxed"
    reference: Callable[[int, int], ir.Circuit] | None = None
    literal: bool = False
    note: str = ""


@dataclass(frozen=True)
class Subject:
    """A subject file's declaration (BENCHMARKS.txt, section 3)."""

    id: str
    name: str
    category: str
    source: str
    size_parameter: str | None
    dimension: str | int
    domain: Domain
    equations: tuple[Equation, ...]
    restricted: tuple[tuple[str, ...], ...] = ()
    clifford: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.size_parameter is None and self.domain.n_min != self.domain.n_max:
            object.__setattr__(self, "domain", replace(self.domain, n_min=0, n_max=0))

    @property
    def general_d(self) -> bool:
        """Whether the subject has a general dimension d."""
        return self.dimension == "general"


@dataclass(frozen=True)
class Instance:
    """One equation at one version: ``restriction`` the clause tuple of a restricted claim."""

    subject: Subject
    equation: Equation
    restriction: tuple[str, ...] = ()
    falsified: Any = None

    @property
    def id(self) -> str:
        """``subject/equation[/r<k>]``, with ``!`` appended for an F equation."""
        base = f"{self.subject.id}/{self.equation.key}"
        if self.restriction:
            k = self.subject.restricted.index(self.restriction)
            base += f"/r{k}"
        return base + ("!" if self.falsified is not None else "")

    @property
    def domain(self) -> Domain:
        """The subject's domain cut by the restriction's clauses."""
        dom = self.subject.domain
        for clause in self.restriction:
            dom = restrict_domain(dom, clause)
        floor = (
            getattr(self.falsified, "params", {}).get("d_min")
            if self.falsified is not None
            else None
        )
        if floor is not None and dom.d_min < floor:
            dom = replace(dom, d_min=floor)
        return dom

    @property
    def truth(self) -> bool:
        """True for an S equation, False for an F equation."""
        return self.falsified is None


def restrict_domain(dom: Domain, clause: str) -> Domain:
    """``dom`` cut by one engine clause (n = v, n >= c, n = a*m + c, d = v, d >= c)."""
    text = clause.replace(" ", "")
    sym, op, rhs = (text[0], text[1:3], text[3:]) if text[1:3] == ">=" else (text[0], "=", text[2:])
    if sym == "d":
        v = int(rhs)
        return (
            replace(dom, d_min=v, d_max=v) if op == "=" else replace(dom, d_min=max(dom.d_min, v))
        )
    if op == ">=":
        return replace(dom, n_min=max(dom.n_min, int(rhs)))
    if "m" in rhs:
        a_part, _, c_part = rhs.partition("+")
        a = int(a_part.replace("*m", "").replace("m", "") or 1)
        c = int(c_part or 0)
        start = c
        while start < dom.n_min:
            start += a
        return replace(dom, n_min=start, n_step=a)
    v = int(rhs)
    return replace(dom, n_min=v, n_max=v)


@dataclass(frozen=True)
class Script:
    """An engine script: the text, whether dimensions were written literally, the circuit's
    naive node count."""

    text: str
    literal: bool
    nodes: int


def spec_context(subject: Subject, nv: int | None, dv: int | None, literal_d: bool) -> SpecContext:
    """The :class:`SpecContext` at (nv, dv); symbolic where None."""
    if not subject.general_d:
        dv = int(subject.dimension)
        literal_d = True
    if literal_d and dv is not None:
        D, d = f"literal {dv}", str(dv)
    else:
        D, d = "d", "d"
    n = "n"
    return SpecContext(D=D, d=d, n=n, N=n, nv=nv, dv=dv)


def _header(
    subject: Subject, eq: Equation, nv: int | None, dv: int | None, literal_d: bool
) -> list[str]:
    out = []
    if subject.general_d and not literal_d:
        out.append("dim d" + (f" = {dv}" if dv is not None else ""))
    if eq.form == "boxed" and subject.size_parameter is not None:
        out.append("count n" + (f" = {nv}" if nv is not None else ""))
    return out


def _indent(text: str) -> str:
    return "\n".join("    " + ln if ln.strip() else ln for ln in text.splitlines())


def circuit_at(
    inst: Instance, nv: int | None, dv: int | None, *, expand: bool = False
) -> ir.Circuit:
    """The instance's circuit at (nv, dv), the F break applied; ``expand`` makes register counts
    and the scalar concrete in n first."""
    subject, eq = inst.subject, inst.equation
    d_arg: Any = dv if dv is not None else ("d" if subject.general_d else int(subject.dimension))
    if not subject.general_d:
        d_arg = int(subject.dimension)
    if eq.form == "unrolled":
        if nv is None:
            raise translate.Unsupported("the circuit has no general form in n")
        n_arg: Any = nv
    elif eq.form == "boxed":
        n_arg = "n"
    else:
        n_arg = None
    if eq.form == "boxed" and subject.size_parameter is None:
        n_arg = None
    c = eq.circuit(n_arg, d_arg)
    if expand and nv is not None:
        import sympy as sp

        regs = {q: int(sp.sympify(v).subs(sp.Symbol("n"), nv)) for q, v in c.registers.items()}
        scalar = str(sp.sympify(c.scalar).subs(sp.Symbol("n"), nv))
        c = replace(c, registers=regs, scalar=scalar)
    if inst.falsified is not None:
        c = inst.falsified.apply(c, inst, nv, dv)
    return c


def script(inst: Instance, nv: int | None, dv: int | None, *, command: str = "decide") -> Script:
    """The engine script of ``inst`` at (nv, dv), None meaning symbolic: the specification as
    ``@rhs``, the naive circuit loaded, then ``decide against @rhs`` with the restriction."""
    subject, eq = inst.subject, inst.equation
    if dv is None and not subject.general_d:
        dv = int(subject.dimension)
    c = circuit_at(inst, nv, dv)
    values = {k: v for k, v in (("n", nv), ("d", dv)) if v is not None}
    literal = not subject.general_d or eq.literal
    if literal and dv is None:
        raise translate.Unsupported("the equation needs a concrete dimension")
    tr: translate.Translation | None = None
    if not literal:
        try:
            tr = translate.translate(c, translate.Context(values=values, literal=frozenset()))
        except translate.Unsupported as exc:
            if "literal" not in str(exc) or dv is None:
                raise
            literal = True
    if tr is None:
        tr = translate.translate(c, translate.Context(values=values, literal=frozenset({"d"})))
    ctx = spec_context(subject, nv, dv, literal)
    spec = eq.specification(ctx)
    if inst.falsified is not None:
        spec = inst.falsified.apply_spec(spec, inst, nv, dv, ctx)
    head = _header(subject, eq, nv, dv, literal)
    blocks = [
        "@rhs =",
        _indent("\n".join([*head, spec])),
        "",
        "load",
        _indent("\n".join([*head, tr.body])),
        "",
    ]
    clause = f" for {', '.join(inst.restriction)}" if inst.restriction else ""
    blocks.append(f"{command} against @rhs{clause}")
    return Script("\n".join(blocks) + "\n", literal, tr.nodes)


def _load(path: Path) -> Subject:
    name = f"benchmarks_subject_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    subject = module.SUBJECT
    if subject.id != path.stem:
        raise ValueError(f"{path}: id {subject.id!r} differs from the file name")
    return subject


@functools.cache
def catalogue(directory: str = "benchmarks/subjects") -> tuple[Subject, ...]:
    """Every subject file of ``directory`` (relative to the repository root), by id."""
    base = Path(directory)
    base = base if base.is_absolute() else ROOT / base
    return tuple(_load(p) for p in sorted(base.glob("*.py")) if not p.name.startswith("_"))


def select(
    subjects: tuple[Subject, ...], include: list[str], exclude: list[str], categories: list[str]
) -> list[Subject]:
    """The subjects matching an include glob and no exclude glob, in the given categories."""
    return [
        s
        for s in subjects
        if s.category in categories
        and any(fnmatch.fnmatch(s.id, g) for g in include)
        and not any(fnmatch.fnmatch(s.id, g) for g in exclude)
    ]


def instances(subjects: list[Subject]) -> Iterator[Instance]:
    """The S equations: every equation of every subject, unrestricted then per restriction."""
    for s in subjects:
        for eq in s.equations:
            yield Instance(s, eq)
            for r in s.restricted:
                yield Instance(s, eq, r)


def has_symbolic(inst: Instance) -> bool:
    """Whether the instance has a symbolic version (a general form in n or d)."""
    eq = inst.equation
    if eq.form == "unrolled":
        return False
    return inst.subject.general_d or (
        eq.form == "boxed" and inst.subject.size_parameter is not None
    )


def axes(inst: Instance) -> tuple[bool, bool]:
    """Whether the instance varies in n and in d."""
    return inst.subject.size_parameter is not None, inst.subject.general_d


def bindings(mapping: Mapping[str, int | None]) -> dict[str, int]:
    """``mapping`` without its None entries."""
    return {k: v for k, v in mapping.items() if v is not None}


__all__ = [
    "Domain",
    "Equation",
    "Instance",
    "Script",
    "SpecContext",
    "Subject",
    "catalogue",
    "field",
    "instances",
    "script",
    "select",
]
