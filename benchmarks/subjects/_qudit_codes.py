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

"""Qudit codes over Z_d from integer data. CSS: |x> -> d^(-r/2) sum_s |x L + s S>, encoded by
carrier fans and F-then-fan on stabilizer pivots, specified by the integer CSS normal form.
Graph codes: |x> -> Z^(x N) |G>, encoded by CZ from each carrier then F on it then the graph's
CZ edges, specified by Z spiders joined by F edges."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks import ir
from benchmarks.ir import CX, CZ, F, Init
from benchmarks.subject import SpecContext
from benchmarks.subjects._spec import Dsl


def _fan(c: str, t: str, a: int) -> list[ir.Gate]:
    """CX^a from c onto t as |a| CX gates of the sign of a."""
    s = 1 if a > 0 else -1
    return [CX(c, t, s) for _ in range(abs(a))]


def css_encoder(
    n: int,
    stabs: Sequence[Sequence[int]],
    logicals: Sequence[Sequence[int]],
    carriers: Sequence[int],
) -> list[ir.Gate]:
    """The encoder gates on wires q0..q{n-1}: stabilizer rows reduced to unit pivots off the
    carriers, logical rows reduced on those pivots, carrier fans in dependency order, then F
    and a fan from each stabilizer pivot."""
    S = [list(r) for r in stabs]
    L = [list(r) for r in logicals]
    pivots: list[int] = []
    for i in range(len(S)):
        cands = [j for j in range(n) if j not in carriers and j not in pivots and abs(S[i][j]) == 1]
        col = min(cands, key=lambda j: sum(1 for row in L if row[j]))
        if S[i][col] < 0:
            S[i] = [-v for v in S[i]]
        for rows in (S, L):
            for j, row in enumerate(rows):
                if (rows is not S or j != i) and row[col]:
                    a = row[col]
                    rows[j] = [u - a * w for u, w in zip(row, S[i])]
        pivots.append(col)
    for l, c in enumerate(carriers):
        if L[l][c] != 1:
            raise ValueError(f"logical row {l} is not 1 on its carrier {c}")
    q = [f"q{j}" for j in range(n)]
    order: list[int] = []
    pending = list(range(len(L)))
    while pending:
        ready = [m for m in pending if all(L[m][carriers[l]] == 0 for l in pending if l != m)]
        if not ready:
            raise ValueError("carrier dependencies are cyclic")
        order.append(ready[0])
        pending.remove(ready[0])
    gates: list[ir.Gate] = [Init(q[j]) for j in range(n) if j not in carriers]
    for m in order:
        for j in range(n):
            if j != carriers[m] and L[m][j]:
                gates += _fan(q[carriers[m]], q[j], L[m][j])
    for i, p in enumerate(pivots):
        gates.append(F(q[p]))
        for j in range(n):
            if j != p and S[i][j]:
                gates += _fan(q[p], q[j], S[i][j])
    return gates


def css_spec(
    c: SpecContext,
    n: int,
    stabs: Sequence[Sequence[int]],
    logicals: Sequence[Sequence[int]],
    carriers: Sequence[int],
) -> str:
    """The integer CSS normal form: a Z spider per logical row (an input, in carrier order)
    and per stabilizer row (a state), an X spider per qudit; an entry a is |a| wires, through
    the antipode F^2 when negative."""
    b = Dsl()
    feeds: dict[int, list[str]] = {j: [] for j in range(n)}
    rows: list[tuple[Sequence[int], bool]] = [
        (logicals[l], True) for l in sorted(range(len(logicals)), key=lambda l: carriers[l])
    ]
    rows += [(s, False) for s in stabs]
    for row, is_input in rows:
        legs = sum(abs(v) for v in row)
        z = b.node(f"Z({1 if is_input else 0} -> {legs}, {c.D})")
        if is_input:
            b.ins.append(f"{z}.in0")
        k = 0
        for j, v in enumerate(row):
            for _ in range(abs(v)):
                port = f"{z}.out{k}"
                k += 1
                if v < 0:
                    f1 = b.node(f"F({c.D})")
                    f2 = b.node(f"F({c.D})")
                    b.wire(port, f"{f1}.in0")
                    b.wire(f"{f1}.out0", f"{f2}.in0")
                    port = f"{f2}.out0"
                feeds[j].append(port)
    exp = -len(stabs)
    for j in range(n):
        m = len(feeds[j])
        x = b.node(f"X({m} -> 1, {c.D})")
        for i, port in enumerate(feeds[j]):
            b.wire(port, f"{x}.in{i}")
        b.outs.append(f"{x}.out0")
        exp += m - 1
    if exp:
        b.scalar.append(f"{c.d}^(({exp})/2)")
    return b.text()


def graph_encoder(
    n: int,
    edges: Sequence[tuple[int, int]],
    logicals: Sequence[Sequence[int]],
    carriers: Sequence[int],
) -> list[ir.Gate]:
    """The graph-code encoder on q0..q{n-1}: |+> on the non-carriers, CZ from each carrier to
    the rest of its logical set, F on the carriers, the CZ edges of the graph."""
    q = [f"q{j}" for j in range(n)]
    gates: list[ir.Gate] = []
    for j in range(n):
        if j not in carriers:
            gates += [Init(q[j]), F(q[j])]
    for N, cj in zip(logicals, carriers):
        for j in N:
            if j != cj:
                if j in carriers:
                    raise ValueError("a logical set contains another carrier")
                gates.append(CZ(q[cj], q[j]))
    gates += [F(q[cj]) for cj in carriers]
    gates += [CZ(q[a], q[b]) for a, b in edges]
    return gates


def graph_spec(
    c: SpecContext,
    n: int,
    edges: Sequence[tuple[int, int]],
    logicals: Sequence[Sequence[int]],
    carriers: Sequence[int],
) -> str:
    """Z^(x N) |G>: a Z spider per vertex, an F edge per graph edge, an input Z spider per
    logical joined by F edges to its set."""
    b = Dsl()
    deg = {j: 0 for j in range(n)}
    for a, bb in edges:
        deg[a] += 1
        deg[bb] += 1
    for N in logicals:
        for j in N:
            deg[j] += 1
    vert = {j: b.node(f"Z({deg[j]} -> 1, {c.D})") for j in range(n)}
    used = {j: 0 for j in range(n)}

    def port(j: int) -> str:
        p = f"{vert[j]}.in{used[j]}"
        used[j] += 1
        return p

    nf = 0
    for l in sorted(range(len(logicals)), key=lambda l: carriers[l]):
        N = logicals[l]
        z = b.node(f"Z(1 -> {len(N)}, {c.D})")
        b.ins.append(f"{z}.in0")
        for i, j in enumerate(N):
            f = b.node(f"F({c.D})")
            b.wire(f"{z}.out{i}", f"{f}.in0")
            b.wire(f"{f}.out0", port(j))
            nf += 1
    for a, bb in edges:
        u = b.node(f"Z(0 -> 2, {c.D})")
        f = b.node(f"F({c.D})")
        b.wire(f"{u}.out0", port(a))
        b.wire(f"{u}.out1", f"{f}.in0")
        b.wire(f"{f}.out0", port(bb))
        nf += 1
    b.outs = [f"{vert[j]}.out0" for j in range(n)]
    exp = nf - n
    if exp:
        b.scalar.append(f"{c.d}^(({exp})/2)")
    return b.text()


__all__ = ["css_encoder", "css_spec", "graph_encoder", "graph_spec"]
