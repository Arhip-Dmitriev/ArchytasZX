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

"""Dense numpy semantics of the benchmark circuit IR, written from the gate definitions in
benchmarks/ir.py, for checking translators and building rival gate matrices.

``evaluate(c, n, d)`` returns the map as a tensor with one axis per boundary output then one per
boundary input (registers expanded copy-major), or, when the circuit discards or measures, the
CPM ``T[O, O', I, I']`` with each group flattened row-major.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp

from benchmarks import ir


def _num(expr: Any, n: int, d: int, k: int | None = None) -> complex:
    subs = {sp.Symbol("n"): n, sp.Symbol("d"): d}
    if k is not None:
        subs[sp.Symbol("k")] = k
    return complex(sp.N(sp.sympify(expr).subs(subs), 30))


def _turn(t: complex) -> complex:
    return complex(np.exp(2j * np.pi * t.real))


class _State:
    def __init__(self) -> None:
        self.t = np.ones((), dtype=complex)
        self.labels: list[tuple[str, Any]] = []
        self.dims: dict[tuple[str, Any], int] = {}
        self.env = 0

    def add(self, tensor: np.ndarray, labels: list[tuple[str, Any]], dims: list[int]) -> None:
        self.t = np.multiply.outer(self.t, tensor)
        self.labels += labels
        for lab, dv in zip(labels, dims):
            self.dims[lab] = dv

    def apply(self, op: np.ndarray, wires: list[str]) -> None:
        """``op`` with axes (outs..., ins...) applied to the live ends of ``wires``."""
        m = len(wires)
        axes = [self.labels.index(("live", q)) for q in wires]
        self.t = np.tensordot(op, self.t, axes=(list(range(m, 2 * m)), axes))
        rest = [lab for i, lab in enumerate(self.labels) if i not in axes]
        self.labels = [("live", q) for q in wires] + rest

    def contract(self, vec: np.ndarray, q: str) -> None:
        ax = self.labels.index(("live", q))
        self.t = np.tensordot(vec, self.t, axes=([0], [ax]))
        self.labels.pop(ax)

    def rename(self, old: tuple[str, Any], new: tuple[str, Any]) -> None:
        i = self.labels.index(old)
        self.labels[i] = new
        self.dims[new] = self.dims[old]


def _expand(c: ir.Circuit, n: int) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Wire order with registers copy-major, the copies of each register, the dimension
    expression of each expanded wire."""
    order: list[str] = []
    copies: dict[str, list[str]] = {}
    group: list[str] = []
    gkey: str | None = None

    def flush() -> None:
        if group:
            cnt = _count(c.count(group[0]), n)
            for i in range(cnt):
                for q in group:
                    order.append(f"{q}[{i}]")
            for q in group:
                copies[q] = [f"{q}[{i}]" for i in range(cnt)]
            group.clear()

    for q in c.wires:
        key = str(c.count(q)) if c.count(q) is not None else None
        if key is None:
            flush()
            gkey = None
            order.append(q)
            copies[q] = [q]
        else:
            if gkey != key:
                flush()
                gkey = key
            group.append(q)
    flush()
    dims = {}
    for q in c.wires:
        for e in copies[q]:
            dims[e] = c.dim(q)
    return order, copies, dims


def _count(v: Any, n: int) -> int:
    return int(sp.sympify(v).subs(sp.Symbol("n"), n))


def _first(c: ir.Circuit, q: str) -> ir.Gate | None:
    for g in c.gates:
        if q in ir.gate_wires(g):
            return g
    return None


def _single(g: ir.Gate, dv: int, n: int, d: int) -> np.ndarray:
    w = np.exp(2j * np.pi / dv)
    if isinstance(g, ir.X):
        a = round(_num(g.a, n, d).real)
        return np.array(
            [[1.0 if (j + a) % dv == i else 0 for j in range(dv)] for i in range(dv)], dtype=complex
        )
    if isinstance(g, ir.Z):
        a = _num(g.a, n, d)
        return np.diag([np.exp(2j * np.pi * (a * k).real / dv) for k in range(dv)])
    if isinstance(g, ir.F):
        f = np.array([[w ** (j * k) for k in range(dv)] for j in range(dv)]) / math.sqrt(dv)
        return np.linalg.matrix_power(f, g.p % 4)
    if isinstance(g, (ir.ZPhase, ir.XPhase)):
        ph = []
        for k in range(dv):
            if k == 0:
                ph.append(1.0)
            elif g.rule is not None:
                ph.append(_turn(_num(g.rule, n, d, k)))
            else:
                ph.append(_turn(_num(g.entries.get(k, 0), n, d)))
        z = np.diag(ph)
        if isinstance(g, ir.ZPhase):
            return z
        f = np.array([[w ** (j * k) for k in range(dv)] for j in range(dv)]) / math.sqrt(dv)
        return f @ z @ f.conj().T
    raise TypeError(g)


def _joint(dims: list[int]) -> list[tuple[int, ...]]:
    return [tuple(x) for x in np.ndindex(*dims)]


def _op_from(fn: Any, dims: list[int]) -> np.ndarray:
    """The tensor (outs..., ins...) of the basis map x -> (coefficient, y) given by ``fn``."""
    op = np.zeros(dims + dims, dtype=complex)
    for x in _joint(dims):
        coef, y = fn(x)
        op[tuple(y) + tuple(x)] += coef
    return op


def evaluate(c: ir.Circuit, n: int, d: int) -> np.ndarray:
    """The circuit at (n, d) (see the module docstring)."""
    if any(isinstance(g, (ir.Chain, ir.Pairs)) for g in c.gates):
        from benchmarks.tools.common import expand

        c = expand(c, n)
    order, copies, dexpr = _expand(c, n)
    subs = {sp.Symbol("d"): d, sp.Symbol("n"): n}
    dim = {q: int(sp.sympify(e).subs(subs)) for q, e in dexpr.items()}
    st = _State()
    started: set[str] = set()
    inputs: list[str] = []
    for q in c.wires:
        if not isinstance(_first(c, q), (ir.Init, ir.Prepare)):
            for e in copies[q]:
                eye = np.eye(dim[e], dtype=complex)
                st.add(eye, [("live", e), ("in", e)], [dim[e], dim[e]])
                started.add(e)
    for e in order:
        if e in started:
            inputs.append(e)
    for g in c.gates:
        _apply(st, g, c, copies, dim, n, d)
    outs = [e for e in order if ("live", e) in st.labels]
    ins = [e for e in order if ("in", e) in st.labels]
    envs = [lab for lab in st.labels if lab[0] == "env"]
    perm = (
        [st.labels.index(("live", e)) for e in outs]
        + [st.labels.index(lab) for lab in envs]
        + [st.labels.index(("in", e)) for e in ins]
    )
    t = np.transpose(st.t, perm) * _num(c.scalar, n, d)
    if not envs:
        return t
    do = math.prod(dim[e] for e in outs)
    de = math.prod(st.dims[lab] for lab in envs)
    di = math.prod(dim[e] for e in ins)
    v = t.reshape(do, de, di)
    return np.einsum("aeb,ced->acbd", v, v.conj())


def _apply(
    st: _State,
    g: ir.Gate,
    c: ir.Circuit,
    copies: dict[str, list[str]],
    dim: dict[str, int],
    n: int,
    d: int,
) -> None:
    qs = ir.gate_wires(g)
    reg = [q for q in qs if c.count(q) is not None]
    if reg:
        cnt = len(copies[reg[0]])
        for i in range(cnt):
            sub = {q: (copies[q][i] if c.count(q) is not None else q) for q in qs}
            _apply_one(st, g, sub, dim, n, d)
    else:
        _apply_one(st, g, {q: q for q in qs}, dim, n, d)


def _apply_one(
    st: _State, g: ir.Gate, sub: dict[str, str], dim: dict[str, int], n: int, d: int
) -> None:
    if isinstance(g, ir.Prepare):
        ws = [sub[q] for q in g.qs]
        vec = np.array([complex(a, b) for a, b in g.vector]).reshape([dim[e] for e in ws])
        st.add(vec, [("live", e) for e in ws], [dim[e] for e in ws])
        return
    if isinstance(g, ir.Init):
        e = sub[g.q]
        v = np.zeros(dim[e], dtype=complex)
        v[g.v % dim[e]] = 1
        st.add(v, [("live", e)], [dim[e]])
        return
    if isinstance(g, ir.Post):
        e = sub[g.q]
        v = np.zeros(dim[e], dtype=complex)
        v[g.v % dim[e]] = 1
        st.contract(v, e)
        return
    if isinstance(g, ir.Discard):
        st.rename(("live", sub[g.q]), ("env", st.env))
        st.env += 1
        return
    if isinstance(g, ir.Measure):
        e = sub[g.q]
        dv = dim[e]
        cp = np.zeros((dv, dv, dv), dtype=complex)
        for k in range(dv):
            cp[k, k, k] = 1
        ax = st.labels.index(("live", e))
        st.t = np.tensordot(cp, st.t, axes=([2], [ax]))
        rest = [lab for i, lab in enumerate(st.labels) if i != ax]
        lab = ("env", st.env)
        st.env += 1
        st.labels = [("live", e), lab, *rest]
        st.dims[lab] = dv
        return
    if isinstance(g, (ir.X, ir.Z, ir.F, ir.ZPhase, ir.XPhase)):
        e = sub[g.q]
        st.apply(_single(g, dim[e], n, d), [e])
        return
    if isinstance(g, ir.Swap):
        a, b = sub[g.a], sub[g.b]
        ia, ib = st.labels.index(("live", a)), st.labels.index(("live", b))
        st.labels[ia], st.labels[ib] = ("live", b), ("live", a)
        return
    ws = [sub[q] for q in ir.gate_wires(g)]
    op = gate_tensor(g, [dim[e] for e in ws], n, d)
    st.apply(op, ws)


def gate_tensor(g: ir.Gate, ds: list[int], n: int, d: int) -> np.ndarray:
    """The tensor (outputs..., inputs...) of a unitary gate on wires of dimensions ``ds``."""
    if isinstance(g, (ir.X, ir.Z, ir.F, ir.ZPhase, ir.XPhase)):
        return _single(g, ds[0], n, d)
    if isinstance(g, ir.CX):
        dv = ds[0]
        op = _op_from(lambda x: (1.0, (x[0], (x[1] + g.s * x[0]) % dv)), ds)
    elif isinstance(g, ir.CZ):
        dv = ds[0]
        op = _op_from(lambda x: (np.exp(2j * np.pi * g.s * x[0] * x[1] / dv), x), ds)
    elif isinstance(g, ir.PhaseFn):
        op = _op_from(
            lambda x: (_turn(complex(float(sp.Rational(g.f(x)))) if g.f(x) != 0 else 0), x), ds
        )
    elif isinstance(g, (ir.MCX, ir.MCZ)):
        vals = g.values if g.values is not None else (1,) * len(g.cs)
        dt = ds[-1]

        def fn(x: tuple[int, ...]) -> tuple[complex, tuple[int, ...]]:
            hit = tuple(x[:-1]) == tuple(vals)
            if isinstance(g, ir.MCX):
                return 1.0, (*x[:-1], (x[-1] + g.a * hit) % dt)
            return np.exp(2j * np.pi * g.a * x[-1] * hit / dt), x

        op = _op_from(fn, ds)
    elif isinstance(g, ir.Perm):
        total = math.prod(ds)

        def fn(x: tuple[int, ...]) -> tuple[complex, tuple[int, ...]]:
            idx = int(np.ravel_multi_index(x, ds))
            return 1.0, tuple(int(v) for v in np.unravel_index(g.p(idx) % total, ds))

        op = _op_from(fn, ds)
    elif isinstance(g, ir.Unitary):
        m = np.array([[complex(a, b) for a, b in row] for row in g.matrix])
        op = m.reshape(ds + ds)
    elif isinstance(g, ir.Oracle):
        dy = ds[-1]
        op = _op_from(lambda x: (1.0, (*x[:-1], (x[-1] + g.f(tuple(x[:-1]))) % dy)), ds)
    else:
        raise TypeError(g)
    return op
