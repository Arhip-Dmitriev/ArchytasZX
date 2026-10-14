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

"""ZX diagrams of concrete :mod:`archytaszx.semantics.pathsum` states.

A state with amplitudes ``c`` on one wire of dimension D (the product of its qudit dimensions)
is drawn as follows. Pick the first s with ``c_s != 0`` and set ``u_k = c_{k+s mod D} / c_s``,
so ``u_0 = 1``. A W node read backwards (a cup feeds its input) adds the components of the
Z states on its outputs, each drawn as a Z effect::

    u = W^T (z_1 (x) ... (x) z_m),  z_j = sum_k e^{i alpha_j[k]} |k>,  u_k = sum_j e^{i alpha_j[k]}

with m = 0 for ``u = |0>``, m = 1 when every ``|u_k| = 1``, and otherwise m even, m >= max
``|u_k|``, and pairs of phases ``arg u_k +- acos(|u_k| / m)``. An X spider with phases
``-k s / D`` turns shifts by s, the scalar is ``c_s``, and splitters cut the wire into the
state's qudits, most significant first.
"""

from __future__ import annotations

import itertools
import math

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import DIM_SPLITTER, W_NODE, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.repl.printer import render_graph
from archytaszx.semantics import pathsum as P

MAX_LEVELS = 64
"""The largest total dimension D drawn."""

MAX_PATHS = 4096
"""The largest number of basis paths summed."""


class ZXViewError(Exception):
    """A state with no ZX drawing here: symbolic, a bra, or too large."""


def amplitudes(s: P.State) -> list[sympy.Expr]:
    """The exact amplitude of every basis state of a concrete ket, in mixed-radix order."""
    if s.bra:
        raise ZXViewError("a bra")
    if not P.is_concrete(s):
        raise ZXViewError("symbolic widths, dimensions or ranges")
    levels = P.concrete_levels(s)
    size = math.prod(levels)
    if size > MAX_LEVELS:
        raise ZXViewError(f"{size} basis states, more than {MAX_LEVELS}")
    out = [sympy.S.Zero] * size
    paths = 0
    for t in s.terms:
        ranges = [range(int(b.lo), int(b.hi) + 1) for b in t.bound]
        paths += math.prod(len(r) for r in ranges)
        if paths > MAX_PATHS:
            raise ZXViewError(f"more than {MAX_PATHS} basis paths")
        for point in itertools.product(*ranges):
            env = {b.symbol: sympy.Integer(v) for b, v in zip(t.bound, point, strict=True)}
            key = 0
            for k, n in zip(t.kets, levels, strict=True):
                key = key * n + int(sympy.sympify(k).xreplace(env))
            out[key] += t.amp.xreplace(env)
    return [
        sympy.nsimplify(sympy.simplify(a)) if a.has(sympy.Float) else sympy.simplify(a) for a in out
    ]


def _turns(z: sympy.Expr) -> sympy.Expr:
    return sympy.simplify(sympy.arg(z) / (2 * sympy.pi)) if z != 0 else sympy.S.Zero


def _effects(u: list[sympy.Expr]) -> list[dict[int, sympy.Expr]]:
    """Phase maps (index -> turns) of the Z effects whose W^T sum is ``u``."""
    rest = list(enumerate(u))[1:]
    if all(z == 0 for _, z in rest):
        return []
    mags = [sympy.simplify(sympy.Abs(z)) for _, z in rest]
    if all(m == 1 for m in mags):
        return [{k: _turns(z) for k, z in rest}]
    top = max(float(m) for m in mags)
    m = max(2, 2 * math.ceil(top / 2 - 1e-12))
    plus, minus = {}, {}
    for (k, z), r in zip(rest, mags, strict=True):
        phi, beta = _turns(z), sympy.simplify(sympy.acos(r / m) / (2 * sympy.pi))
        plus[k], minus[k] = phi + beta, phi - beta
    return [plus, minus] * (m // 2)


def _qudits(s: P.State) -> list[int]:
    return [
        int(d)  # type: ignore[arg-type]
        for d, w in zip(s.dims, s.widths, strict=True)
        for _ in range(int(w))  # type: ignore[arg-type]
    ]


def diagram(s: P.State) -> Diagram:
    """A ZX diagram whose outputs are the qudits of the concrete ket ``s``."""
    c = amplitudes(s)
    size = len(c)
    nonzero = [k for k, z in enumerate(c) if z != 0]
    if not nonzero:
        raise ZXViewError("the zero vector")
    shift = nonzero[0]
    u = [sympy.simplify(c[(k + shift) % size] / c[shift]) for k in range(size)]
    dim = Dim.concrete(size)
    g = Diagram()
    cup = g.add_node(Z_SPIDER, [], [dim, dim])
    effects = _effects(u)
    w = g.add_node(W_NODE, [dim], [dim] * len(effects))
    g.add_wire(PortRef(cup, Direction.OUTPUT, 0), PortRef(w, Direction.INPUT, 0))
    for j, phases in enumerate(effects):
        vector = PhaseVector(dim, {k: Phase(t) for k, t in phases.items()})
        z = g.add_node(Z_SPIDER, [dim], [], vector)
        g.add_wire(PortRef(w, Direction.OUTPUT, j), PortRef(z, Direction.INPUT, 0))
    out = PortRef(cup, Direction.OUTPUT, 1)
    if shift:
        vector = PhaseVector(
            dim, {k: Phase(sympy.Rational(-k * shift, size)) for k in range(1, size)}
        )
        x = g.add_node(X_SPIDER, [dim], [dim], vector)
        g.add_wire(out, PortRef(x, Direction.INPUT, 0))
        out = PortRef(x, Direction.OUTPUT, 0)
    outputs: list[PortRef] = []
    rest = size
    for q in _qudits(s)[:-1]:
        rest //= q
        split = g.add_node(
            DIM_SPLITTER, [Dim.concrete(q * rest)], [Dim.concrete(q), Dim.concrete(rest)]
        )
        g.add_wire(out, PortRef(split, Direction.INPUT, 0))
        outputs.append(PortRef(split, Direction.OUTPUT, 0))
        out = PortRef(split, Direction.OUTPUT, 1)
    g.set_boundary_outputs([*outputs, out])
    g.set_scalar(Scalar(c[shift]))
    return g


def listing(s: P.State) -> str:
    """The graph listing of :func:`diagram`."""
    return render_graph(diagram(s))
