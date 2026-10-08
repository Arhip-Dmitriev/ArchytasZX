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

"""Test-only numeric evaluator of the printer's Dirac AST, and the oracle it is compared with.

Each copy vector of each IndexVar family is one summed member; every factor instance (after
expanding Products, fans and Blocks at concrete multiplicities) becomes a small dense tensor
over its distinct members, and the sum over all members is contracted pairwise. Axes are
the expanded kets then the expanded bras.
"""

from __future__ import annotations

import cmath
import itertools
import math
import string
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.graph import Diagram
from archytaszx.repl import printer as P
from archytaszx.semantics.check import score

Env = Mapping[str, object]
MAX_ELEMENTS = 1 << 24
Member = tuple[str, tuple[int, ...]]
Copies = dict[str, int]


class EvalSizeError(ValueError):
    """A dense tensor the evaluator would build exceeds ``MAX_ELEMENTS``."""


def _guard(shape: Sequence[int]) -> None:
    if math.prod(shape) > MAX_ELEMENTS:
        raise EvalSizeError(f"tensor of shape {tuple(shape)} exceeds {MAX_ELEMENTS} elements")


def _int_env(env: Env, names: frozenset[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        if name not in env:
            raise KeyError(f"env lacks symbol {name!r}")
        value = env[name]
        out[name] = int(value)  # type: ignore[call-overload]
    return out


def dim_value(dim: Dim, env: Env) -> int:
    """``dim`` at ``env`` as an int."""
    if dim.is_concrete:
        return dim.to_int()
    return dim.substitute(_int_env(env, dim.free_symbols)).to_int()  # type: ignore[arg-type]


def mult_value(mult: Mult, env: Env) -> int:
    """``mult`` at ``env`` as an int."""
    if mult.is_concrete:
        return mult.to_int()
    return mult.substitute(_int_env(env, mult.free_symbols)).to_int()


def phase_value(phase: PhaseVector | None, env: Env, k: int) -> complex:
    """e^{2 pi i alpha(k)} for ``phase`` at ``env``."""
    if phase is None:
        return 1.0 + 0j
    names = phase.free_symbols | phase.dim.free_symbols
    concrete = phase.substitute({n: env[n] for n in names}) if names else phase
    return complex(concrete.get(k).to_complex())


def scalar_value(scalar: Scalar, env: Env) -> complex:
    """``scalar`` at ``env``."""
    names = scalar.free_symbols
    concrete = scalar.substitute({n: env[n] for n in names}) if names else scalar
    return complex(concrete.to_complex())


class _Evaluator:
    """Expands one DiracTerm at one env into tensor operands over members."""

    def __init__(self, term: P.DiracTerm, env: Env) -> None:
        self.env = env
        self.dims: dict[str, int] = {}
        self.copy_counts: dict[str, int] = {}
        self.members: dict[Member, int] = {}
        for var in term.indices:
            assert var.name not in self.dims, f"duplicate IndexVar {var.name}"
            self.dims[var.name] = dim_value(var.dim, env)
            for cv in var.copies:
                self._count(cv)
            ranges = [range(1, self.copy_counts[cv.name] + 1) for cv in var.copies]
            for vector in itertools.product(*ranges):
                self.members[(var.name, tuple(vector))] = self.dims[var.name]
        self.operands: list[tuple[np.ndarray, list[object]]] = []
        self.used: set[Member] = set()

    def _count(self, cv: P.CopyVar) -> int:
        value = mult_value(cv.count, self.env)
        previous = self.copy_counts.setdefault(cv.name, value)
        assert previous == value, f"copy var {cv.name} has two counts"
        return value

    def _range(self, cv: P.CopyVar) -> range:
        return range(1, self._count(cv) + 1)

    def _assignments(self, over: Sequence[P.CopyVar], ctx: Copies) -> Iterator[Copies]:
        for values in itertools.product(*(self._range(cv) for cv in over)):
            inner = dict(ctx)
            for cv, v in zip(over, values, strict=True):
                assert cv.name not in ctx, f"copy var {cv.name} rebound"
                inner[cv.name] = v
            yield inner

    def member(self, at: P.IndexAt, ctx: Copies) -> Member:
        assert tuple(c.name for c in at.at) == tuple(c.name for c in at.var.copies), at
        key = (at.var.name, tuple(ctx[c.name] for c in at.at))
        assert key in self.members, f"undeclared member {key}"
        self.used.add(key)
        return key

    def legs(self, legs: Sequence[P.Leg], ctx: Copies) -> list[Member]:
        out: list[Member] = []
        for leg in legs:
            for inner in self._assignments(leg.fan, ctx):
                out.append(self.member(leg.index, inner))
        return out

    def add(self, members: Sequence[Member], fn: Callable[[Mapping[Member, int]], complex]) -> None:
        distinct = list(dict.fromkeys(members))
        shape = tuple(self.members[m] for m in distinct)
        _guard(shape)
        tensor = np.zeros(shape, dtype=np.complex128)
        for values in itertools.product(*(range(s) for s in shape)):
            tensor[values] = fn(dict(zip(distinct, values, strict=True)))
        self.operands.append((tensor, list(distinct)))

    def factor(self, f: object, ctx: Copies) -> None:
        env = self.env
        if isinstance(f, P.Product):
            for inner in self._assignments(f.over, ctx):
                for sub in f.factors:
                    self.factor(sub, inner)
        elif isinstance(f, P.Delta):
            a, b = self.member(f.left, ctx), self.member(f.right, ctx)
            self.add([a, b], lambda v: float(v[a] == v[b]))
        elif isinstance(f, P.ZFactor):
            k = self.member(f.index, ctx)
            phase = f.phase
            self.add([k], lambda v: phase_value(phase, env, v[k]))
        elif isinstance(f, P.XFactor):
            q = self.member(f.kappa, ctx)
            outs, ins = self.legs(f.outputs, ctx), self.legs(f.inputs, ctx)
            d = dim_value(f.dim, env)
            norm = d ** (-(len(outs) + len(ins)) / 2)
            x_phase = f.phase

            def x_entry(v: Mapping[Member, int]) -> complex:
                total = sum(v[o] for o in outs) - sum(v[i] for i in ins)
                return complex(
                    norm
                    * phase_value(x_phase, env, v[q])
                    * cmath.exp(2j * math.pi * v[q] * total / d)
                )

            self.add([q, *outs, *ins], x_entry)
        elif isinstance(f, P.FourierFactor):
            o, i = self.member(f.out, ctx), self.member(f.inp, ctx)
            d = dim_value(f.dim, env)
            self.add([o, i], lambda v: cmath.exp(2j * math.pi * v[o] * v[i] / d) / math.sqrt(d))
        elif isinstance(f, P.TriangleFactor):
            o, i = self.member(f.out, ctx), self.member(f.inp, ctx)
            self.add([o, i], lambda v: float(v[o] == 0 or v[o] == v[i]))
        elif isinstance(f, P.TriangleInverseFactor):
            o, i = self.member(f.out, ctx), self.member(f.inp, ctx)
            self.add([o, i], lambda v: float(v[o] == v[i]) - float(v[o] == 0 and v[i] >= 1))
        elif isinstance(f, P.WFactor):
            i = self.member(f.inp, ctx)
            outs = self.legs(f.outputs, ctx)
            self.add(
                [i, *outs],
                lambda v: float(
                    v[i] == sum(v[o] for o in outs) and sum(1 for o in outs if v[o]) <= 1
                ),
            )
        elif isinstance(f, P.BinderFactor):
            o, a, b = (self.member(x, ctx) for x in (f.out, f.a, f.b))
            t = dim_value(f.t, env)
            self.add([o, a, b], lambda v: float(v[o] == v[a] * t + v[b]))
        elif isinstance(f, P.SplitterFactor):
            i, a, b = (self.member(x, ctx) for x in (f.inp, f.a, f.b))
            t = dim_value(f.t, env)
            self.add([i, a, b], lambda v: float(v[i] == v[a] * t + v[b]))
        else:
            raise TypeError(f"unknown factor {f!r}")

    def ket(self, items: Sequence[object], ctx: Copies) -> list[Member]:
        out: list[Member] = []
        for item in items:
            if isinstance(item, P.Slot):
                out.append(self.member(item.index, ctx))
            elif isinstance(item, P.Fan):
                for inner in self._assignments(item.over, ctx):
                    out.append(self.member(item.index, inner))
            elif isinstance(item, P.Block):
                for inner in self._assignments((item.over,), ctx):
                    out.extend(self.ket(item.items, inner))
            else:
                raise TypeError(f"unknown ket item {item!r}")
        return out


def _letters(labels: Sequence[object], table: dict[object, str]) -> str:
    for label in labels:
        if label not in table:
            table[label] = string.ascii_letters[len(table)]
    return "".join(table[label] for label in labels)


def _contract(operands: list[tuple[np.ndarray, list[object]]], output: list[object]) -> np.ndarray:
    """Pairwise contraction, summing a label once no other operand or the output holds it."""
    work = list(operands)
    keep = set(output)
    if not work:
        return np.array(1.0 + 0j)
    while len(work) > 1:
        best = None
        for i, j in itertools.combinations(range(len(work)), 2):
            shared = len(set(work[i][1]) & set(work[j][1]))
            size = len(set(work[i][1]) | set(work[j][1])) - shared
            key = (-shared, size)
            if best is None or key < best[0]:
                best = (key, i, j)
        assert best is not None
        _, i, j = best
        (ta, la), (tb, lb) = work[i], work[j]
        rest = [op for n, op in enumerate(work) if n not in (i, j)]
        live = keep.union(*(set(op[1]) for op in rest))
        result = [lab for lab in dict.fromkeys([*la, *lb]) if lab in live]
        dims = {**dict(zip(la, ta.shape, strict=True)), **dict(zip(lb, tb.shape, strict=True))}
        _guard([dims[lab] for lab in result])
        table: dict[object, str] = {}
        spec = f"{_letters(la, table)},{_letters(lb, table)}->{_letters(result, table)}"
        work = [*rest, (np.einsum(spec, ta, tb), result)]
    tensor, labels = work[0]
    table = {}
    final = [lab for lab in labels if lab in keep]
    tensor = np.einsum(f"{_letters(labels, table)}->{_letters(final, table)}", tensor)
    order = [final.index(lab) for lab in output]
    return np.asarray(np.transpose(tensor, order), dtype=np.complex128)


def evaluate(term: P.DiracTerm, env: Env) -> np.ndarray:
    """The tensor ``term`` denotes at ``env`` (kets then bras, complex128)."""
    ev = _Evaluator(term, env)
    for f in term.factors:
        ev.factor(f, {})
    positions = [*ev.ket(term.kets, {}), *ev.ket(term.bras, {})]
    _guard([ev.members[m] for m in positions])
    operands = list(ev.operands)
    output: list[object] = []
    for n, member in enumerate(positions):
        label = ("boundary", n)
        output.append(label)
        operands.append((np.eye(ev.members[member], dtype=np.complex128), [label, member]))
    free = 1
    referenced = {m for _, labels in operands for m in labels}
    for member, dim in ev.members.items():
        if member not in referenced:
            free *= dim
    tensor = _contract(operands, output) * free
    return tensor * scalar_value(term.scalar, env)


def full_env(diagram: Diagram, env: Env) -> dict[str, object]:
    """``diagram.parameters`` overridden by ``env``."""
    return {**diagram.parameters, **env}


def oracle_tensor(diagram: Diagram, env: Env) -> np.ndarray:
    """The numeric oracle's tensor of ``diagram`` at ``env`` (bang boxes instantiated)."""
    return np.asarray(score(diagram, dict(env)).tensor, dtype=np.complex128)


def rational(p: int, q: int) -> sp.Rational:
    """A sympy rational, for phase-symbol env values."""
    return sp.Rational(p, q)
