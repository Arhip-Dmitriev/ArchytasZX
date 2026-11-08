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

"""Qubit Clifford synthesis from Pauli images and stabilizer-state contraction. Synthesis: the
image rows (X_j -> P_j, Z_j -> Q_j) are reduced qubit by qubit to X_j, Z_j by H, S, S^dag, CX,
X and Z conjugations, and the circuit is the inverse of the reducing gates. Contraction: the
stabilizer group of <Phi+|_{uv} |psi> keeps the combinations of generators commuting with X_u X_v
and Z_u Z_v, each multiplied by those to clear legs u and v."""

from __future__ import annotations

from collections.abc import Sequence

from benchmarks import ir
from benchmarks.ir import CX, X, Z
from benchmarks.subjects._gates import H, S, Sdg
from benchmarks.subjects._stab import Pauli, commute, mul

Gate = tuple[str, int, int]


def _set(t: tuple[int, ...], j: int, v: int) -> tuple[int, ...]:
    return (*t[:j], v, *t[j + 1 :])


def conjugate(p: Pauli, g: Gate) -> Pauli:
    """U p U^dag for the gate g = (name, a, b): H, S, Sdg, X, Z on qubit a, CX from a to b."""
    name, a, b = g
    x, z, ph = p.x, p.z, p.p
    if name == "H":
        xa, za = x[a], z[a]
        return Pauli(_set(x, a, za), _set(z, a, xa), (ph + 2 * xa * za) % 4)
    if name in ("S", "Sdg"):
        xa = x[a]
        return Pauli(x, _set(z, a, z[a] ^ xa), (ph + (1 if name == "S" else 3) * xa) % 4)
    if name == "X":
        return Pauli(x, z, (ph + 2 * z[a]) % 4)
    if name == "Z":
        return Pauli(x, z, (ph + 2 * x[a]) % 4)
    if name == "CX":
        return Pauli(_set(x, b, x[b] ^ x[a]), _set(z, a, z[a] ^ z[b]), ph)
    raise ValueError(name)


def _unit(n: int, j: int, letter: str) -> Pauli:
    x = tuple(int(i == j and letter == "X") for i in range(n))
    z = tuple(int(i == j and letter == "Z") for i in range(n))
    return Pauli(x, z, 0)


def synthesize(images: Sequence[tuple[Pauli, Pauli]]) -> list[Gate]:
    """Gates of a Clifford U with U X_j U^dag = P_j and U Z_j U^dag = Q_j for the pairs
    images[j] = (P_j, Q_j)."""
    rows = [list(pq) for pq in images]
    n = len(rows)
    red: list[Gate] = []

    def apply(g: Gate) -> None:
        red.append(g)
        for r in rows:
            r[0], r[1] = conjugate(r[0], g), conjugate(r[1], g)

    for j in range(n):
        P = rows[j][0]
        if not any(P.x[k] for k in range(j, n)):
            apply(("H", next(k for k in range(j, n) if P.z[k]), 0))
        for k in range(j, n):
            P = rows[j][0]
            if P.x[k] and P.z[k]:
                apply(("Sdg", k, 0))
            elif P.z[k]:
                apply(("H", k, 0))
        P = rows[j][0]
        if not P.x[j]:
            k = next(k for k in range(j + 1, n) if P.x[k])
            apply(("CX", j, k))
            apply(("CX", k, j))
            apply(("CX", j, k))
        for k in range(j + 1, n):
            if rows[j][0].x[k]:
                apply(("CX", j, k))
        for k in range(j + 1, n):
            Q = rows[j][1]
            if Q.x[k] and Q.z[k]:
                apply(("Sdg", k, 0))
                apply(("H", k, 0))
            elif Q.x[k]:
                apply(("H", k, 0))
        for k in range(j + 1, n):
            if rows[j][1].z[k]:
                apply(("CX", k, j))
        if rows[j][1].x[j]:
            apply(("H", j, 0))
            apply(("S", j, 0))
            apply(("H", j, 0))
        if rows[j][0].sign() < 0:
            apply(("Z", j, 0))
        if rows[j][1].sign() < 0:
            apply(("X", j, 0))
    for j, (P, Q) in enumerate(rows):
        if (P, Q) != (_unit(n, j, "X"), _unit(n, j, "Z")):
            raise ValueError(f"synthesis left row {j} at {P}, {Q}")
    inv = {"S": "Sdg", "Sdg": "S"}
    return [(inv.get(g[0], g[0]), g[1], g[2]) for g in reversed(red)]


def ir_gates(gates: Sequence[Gate], wires: Sequence[str]) -> list[ir.Gate]:
    """The IR gates of a synthesized gate list on the named wires."""
    out: list[ir.Gate] = []
    for name, a, b in gates:
        q = wires[a]
        if name == "CX":
            out.append(CX(q, wires[b]))
        else:
            out.append({"H": H, "S": S, "Sdg": Sdg, "X": X, "Z": Z}[name](q))
    return out


def _kernel(rows: list[list[int]], m: int) -> list[list[int]]:
    """A basis of the 0/1 vectors c with sum_i c_i rows[i] = 0 (mod 2)."""
    aug = [r[:] + [int(i == j) for j in range(m)] for i, r in enumerate(rows)]
    w = len(rows[0]) if rows else 0
    r = 0
    for col in range(w):
        piv = next((i for i in range(r, m) if aug[i][col]), None)
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        for i in range(m):
            if i != r and aug[i][col]:
                aug[i] = [u ^ v for u, v in zip(aug[i], aug[r])]
        r += 1
    return [row[w:] for row in aug[r:]]


def _product(gens: Sequence[Pauli], c: Sequence[int]) -> Pauli:
    n = gens[0].n
    out = Pauli((0,) * n, (0,) * n, 0)
    for g, ci in zip(gens, c):
        if ci:
            out = mul(out, g)
    return out


def restrict(gens: Sequence[Pauli], keep: Sequence[int]) -> list[Pauli]:
    """The Paulis on the qubits ``keep`` (in that order) of each generator."""
    return [Pauli(tuple(g.x[j] for j in keep), tuple(g.z[j] for j in keep), g.p) for g in gens]


def trivial_on(gens: Sequence[Pauli], legs: Sequence[int]) -> list[Pauli]:
    """Generators of the subgroup acting as the identity on ``legs``."""
    rows = [[g.x[j] for j in legs] + [g.z[j] for j in legs] for g in gens]
    return [_product(gens, c) for c in _kernel(rows, len(gens))]


def contract(gens: Sequence[Pauli], u: int, v: int) -> list[Pauli]:
    """Generators of the stabilizer group of <Phi+|_{uv} |psi> for |psi> stabilized by
    ``gens``, legs u and v removed."""
    n = gens[0].n
    xx = Pauli(tuple(int(j in (u, v)) for j in range(n)), (0,) * n, 0)
    zz = Pauli((0,) * n, tuple(int(j in (u, v)) for j in range(n)), 0)
    rows = [[int(not commute(g, xx)), int(not commute(g, zz))] for g in gens]
    out = []
    for c in _kernel(rows, len(gens)):
        g = _product(gens, c)
        if g.x[u]:
            g = mul(g, xx)
        if g.z[u]:
            g = mul(g, zz)
        if g.x[v] or g.z[v]:
            raise ValueError("legs not cleared")
        out.append(g)
    keep = [j for j in range(n) if j not in (u, v)]
    return restrict(out, keep)


__all__ = ["conjugate", "contract", "ir_gates", "restrict", "synthesize", "trivial_on"]
