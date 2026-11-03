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

"""Benchmark: one symbolic family proof against instance-by-instance checking.

Workloads: the seven families of scaling.py and GHZ (examples/ghz_proof.azx). Tools:

- ``azx-free``: ArchytasZX with n and d free, one run per family (``n`` is ``"free"``).
- ``azx-bound``: ArchytasZX at ``dim d = 2`` with n bound.
- ``pyzx``: PyZX at d = 2 on the same diagram built at each n; both sides as states (inputs
  bent to outputs), ``full_reduce`` of the inner products <L|L>, <R|R>, <R|L>; equal when
  ||L - R||^2 = 0 with exact scalars, a mismatch is ``wrong``.
- ``qiskit`` (optional arm): Qiskit at each n, the left side as a circuit, its Statevector
  (Operator for F^4) compared with the target written from the algorithm's definition.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

from benchmarks import workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "n"
NS = [2**k for k in range(21)]
QUICK_NS = [1, 2, 4]
TOOLS = ("azx-free", "azx-bound", "pyzx", "qiskit")
RECORD_ENV = True
QISKIT_UNSUPPORTED = {
    "spider fusion": "both sides are spider presentations of one state; no circuit reads the "
    "Z spider with an input and n outputs as a different preparation",
    "Deutsch-Jozsa constant": "the equation is a map with the free input c; the arm compares "
    "states",
}


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every family: ``azx-free`` once, every other tool at every n."""
    ns = QUICK_NS if quick else NS
    out: list[dict[str, Any]] = []
    for family in workloads.FAMILIES:
        out.append({"suite": "families", "tool": "azx-free", "family": family, "n": "free"})
        out += [
            {"suite": "families", "tool": tool, "family": family, "n": n}
            for tool in TOOLS[1:]
            for n in ns
        ]
    return out


class Net:
    """A ZX diagram at d = 2: Z/X spiders with phases in units of pi, boundary vertices, plain
    and Hadamard edges, ordered outputs and inputs, and a scalar sqrt(2)^pow2."""

    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.phases: list[Fraction] = []
        self.edges: list[tuple[int, int, bool]] = []
        self.outs: list[int] = []
        self.ins: list[int] = []
        self.pow2 = 0

    def _add(self, kind: str, phase: Fraction) -> int:
        self.kinds.append(kind)
        self.phases.append(phase)
        return len(self.kinds) - 1

    def z(self, phase: Fraction = Fraction(0)) -> int:
        """A Z spider."""
        return self._add("Z", phase)

    def x(self, phase: Fraction = Fraction(0)) -> int:
        """An X spider."""
        return self._add("X", phase)

    def wire(self, a: int, b: int, hadamard: bool = False) -> None:
        """An edge."""
        self.edges.append((a, b, hadamard))

    def out(self, v: int) -> None:
        """A boundary output on ``v``."""
        b = self._add("B", Fraction(0))
        self.wire(v, b)
        self.outs.append(b)

    def inp(self) -> int:
        """A boundary input vertex."""
        b = self._add("B", Fraction(0))
        self.ins.append(b)
        return b

    def f(self, v: int, k: int = 1) -> int:
        """``k`` Fourier boxes after ``v``: each a Hadamard edge into an identity Z spider."""
        for _ in range(k):
            w = self.z()
            self.wire(v, w, True)
            v = w
        return v


PI = Fraction(1)


def nets(family: str, n: int) -> tuple[Net, Net]:
    """The family's two sides at d = 2 and size ``n``."""
    lhs, rhs = Net(), Net()
    if family == "state copy":
        z = lhs.z()
        lhs.wire(lhs.x(), z)
        for _ in range(n):
            lhs.out(z)
        for _ in range(n):
            rhs.out(rhs.x())
        rhs.pow2 = 1 - n
    elif family == "spider fusion":
        b = lhs.z()
        lhs.wire(lhs.z(), b)
        for _ in range(n):
            lhs.out(b)
        z = rhs.z()
        for _ in range(n):
            rhs.out(z)
    elif family == "Fourier order 4":
        for _ in range(n):
            lhs.out(lhs.f(lhs.inp(), 4))
            v = rhs.z()
            rhs.wire(rhs.inp(), v)
            rhs.out(v)
    elif family == "Fourier superposition":
        for _ in range(n):
            lhs.out(lhs.f(lhs.x()))
            rhs.out(rhs.z())
        lhs.pow2 = rhs.pow2 = -n
    elif family == "Bernstein-Vazirani":
        for _ in range(n):
            p = lhs.z(PI)
            lhs.wire(lhs.f(lhs.x()), p)
            lhs.out(lhs.f(p, 3))
            rhs.out(rhs.x(PI))
        lhs.pow2 = rhs.pow2 = -n
    elif family == "Deutsch-Jozsa constant":
        for _ in range(n):
            lhs.out(lhs.f(lhs.x(), 4))
        u = lhs.x()
        lhs.wire(lhs.f(lhs.x(PI), 3), u)
        lhs.wire(lhs.inp(), u)
        lhs.out(u)
        lhs.pow2 = -n
        for _ in range(n):
            rhs.out(rhs.x())
        rhs.out(rhs.z(PI))
        e = rhs.z(PI)
        rhs.wire(rhs.inp(), e)
        rhs.pow2 = -(n + 1)
    elif family == "Deutsch-Jozsa balanced":
        for _ in range(n):
            lhs.out(lhs.f(lhs.x(), 4))
        c = lhs.z()
        lhs.wire(lhs.f(lhs.x()), c)
        lhs.out(lhs.f(c, 3))
        u = lhs.x()
        lhs.wire(lhs.f(lhs.x(PI), 3), u)
        lhs.wire(c, u)
        lhs.out(u)
        lhs.pow2 = -(n + 1)
        for _ in range(n):
            rhs.out(rhs.x())
        rhs.out(rhs.x(PI))
        rhs.out(rhs.z(PI))
        rhs.pow2 = -(n + 2)
    elif family == "GHZ":
        z = lhs.z()
        lhs.wire(lhs.f(lhs.x()), z)
        lhs.out(z)
        for _ in range(n):
            t = lhs.x()
            lhs.wire(z, t)
            lhs.wire(lhs.x(), t)
            lhs.out(t)
        lhs.pow2 = -1
        g = rhs.z()
        for _ in range(n + 1):
            rhs.out(g)
        rhs.pow2 = -1
    else:
        raise ValueError(family)
    return lhs, rhs


def to_pyzx(net: Net, state: bool = True) -> Any:
    """The PyZX graph of ``net``; with ``state``, every boundary vertex is an output (outputs,
    then inputs)."""
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    kinds = {"Z": VertexType.Z, "X": VertexType.X, "B": VertexType.BOUNDARY}
    g = zx.Graph()  # type: ignore[attr-defined]
    ids = [
        g.add_vertex(kinds[k], 0, i, phase=p) for i, (k, p) in enumerate(zip(net.kinds, net.phases))
    ]
    for a, b, h in net.edges:
        g.add_edge((ids[a], ids[b]), EdgeType.HADAMARD if h else EdgeType.SIMPLE)
    if state:
        g.set_outputs(tuple(ids[v] for v in [*net.outs, *net.ins]))
        g.set_inputs(())
    else:
        g.set_outputs(tuple(ids[v] for v in net.outs))
        g.set_inputs(tuple(ids[v] for v in net.ins))
    g.scalar.add_power(net.pow2)
    return g


def _inner(a: Any, b: Any) -> complex:
    """<a|b> by ``full_reduce``; a remainder of at most 30 vertices is contracted by PyZX."""
    import pyzx as zx

    g = b.copy()
    g.compose(a.adjoint())
    zx.full_reduce(g)
    if g.num_vertices() == 0:
        return complex(g.scalar.to_number())
    if g.num_vertices() <= 30:
        return complex(g.to_tensor())
    raise Inconclusive(f"full_reduce left {g.num_vertices()} vertices in a closed diagram")


def pyzx_check(family: str, n: int) -> dict[str, Any]:
    """PyZX on the family at size ``n`` (see the module docstring)."""
    import pyzx  # noqa: F401

    c0, w0 = time.process_time(), time.perf_counter()
    lhs, rhs = nets(family, n)
    gl, gr = to_pyzx(lhs), to_pyzx(rhs)
    size = gl.num_vertices() + gr.num_vertices()
    wires = gl.num_edges() + gr.num_edges()
    ll, rr, rl = _inner(gl, gl), _inner(gr, gr), _inner(gr, gl)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    gap = ll.real + rr.real - 2 * rl.real
    if abs(gap) > 1e-9 * max(1.0, abs(ll), abs(rr)):
        raise Wrong(f"<L|L> = {ll}, <R|R> = {rr}, <R|L> = {rl}")
    return {
        "cpu": cpu,
        "wall": wall,
        "nodes": size,
        "wires": wires,
        "method": "full_reduce inner products",
    }


def _label(bits: list[str]) -> str:
    """A Qiskit label for qubits 0, 1, ... carrying ``bits`` (Qiskit reads labels right to
    left)."""
    return "".join(reversed(bits))


def qiskit_circuit(family: str, n: int) -> tuple[Any, Any]:
    """The left side as a Qiskit circuit and the target from the algorithm's definition: a
    Statevector, or an Operator for F^4."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector

    if family == "state copy":
        qc = QuantumCircuit(n)
        for q in range(1, n):
            qc.cx(0, q)
        return qc, Statevector.from_label("0" * n)
    if family == "Fourier order 4":
        qc = QuantumCircuit(n)
        for q in range(n):
            for _ in range(4):
                qc.h(q)
        return qc, Operator(QuantumCircuit(n))
    if family == "Fourier superposition":
        qc = QuantumCircuit(n)
        qc.h(range(n))
        return qc, Statevector.from_label("+" * n)
    if family == "Bernstein-Vazirani":
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.z(range(n))
        qc.h(range(n))
        return qc, Statevector.from_label("1" * n)
    if family == "Deutsch-Jozsa balanced":
        qc = QuantumCircuit(n + 2)
        for q in range(n):
            for _ in range(4):
                qc.h(q)
        qc.h(n)
        qc.x(n + 1)
        qc.h(n + 1)
        qc.cx(n, n + 1)
        qc.h(n)
        return qc, Statevector.from_label(_label(["0"] * n + ["1", "-"]))
    if family == "GHZ":
        qc = QuantumCircuit(n + 1)
        qc.h(0)
        for q in range(1, n + 1):
            qc.cx(0, q)
        v = np.zeros(2 ** (n + 1), dtype=complex)
        v[0] = v[-1] = 2**-0.5
        return qc, Statevector(v)
    raise Inconclusive(QISKIT_UNSUPPORTED[family])


def qiskit_check(family: str, n: int) -> dict[str, Any]:
    """Qiskit on the family at size ``n`` (see the module docstring)."""
    if family in QISKIT_UNSUPPORTED:
        return {
            "status": "unsupported",
            "detail": QISKIT_UNSUPPORTED[family],
            "cpu": 0.0,
            "wall": 0.0,
        }
    from qiskit.quantum_info import Operator, Statevector

    c0, w0 = time.process_time(), time.perf_counter()
    qc, target = qiskit_circuit(family, n)
    got = Operator(qc) if isinstance(target, Operator) else Statevector(qc)
    same = got == target
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    if not same:
        raise Wrong("the circuit's state differs from the target")
    method = "Operator ==" if isinstance(target, Operator) else "Statevector =="
    return {"cpu": cpu, "wall": wall, "nodes": qc.size(), "wires": qc.num_qubits, "method": method}


def azx(family: str, n: object, d: object) -> dict[str, Any]:
    """ArchytasZX on the family's command file (``decide_script`` fields and ``verdict``)."""
    from benchmarks import engine

    c0, w0 = time.process_time(), time.perf_counter()
    run_ = engine.run_script(workloads.family_script(family, n, d))
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    v = engine.verdict(run_)
    if v["verdict"] == "unequal":
        raise Wrong(f"refuted by {v['method']}: {v['reason']}")
    if v["verdict"] != "equal":
        raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])
    return {
        "cpu": cpu,
        "wall": wall,
        **{k: v[k] for k in ("method", "nodes", "wires", "verdict_cpu")},
    }


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One check of the case's family by its tool."""
    tool, family, n = case["tool"], case["family"], case["n"]
    if tool == "azx-free":
        return azx(family, "free", "free")
    if tool == "azx-bound":
        return azx(family, n, 2)
    if tool == "pyzx":
        return pyzx_check(family, n)
    return qiskit_check(family, n)


def crossover(rows: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    """Per family and concrete tool: the smallest n at which ``azx-free`` is faster than the
    tool or the tool exceeds a cap; None when neither happens."""
    out: dict[str, dict[str, object]] = {}
    for family in workloads.FAMILIES:
        free = [r for r in rows if r["family"] == family and r["tool"] == "azx-free"]
        if not free or free[0]["status"] != "ok":
            continue
        t = free[0]["time"]
        out[family] = {}
        for tool in TOOLS[1:]:
            mine = sorted(
                (r for r in rows if r["family"] == family and r["tool"] == tool),
                key=lambda r: r["n"],
            )
            hit = next(
                (
                    r["n"]
                    for r in mine
                    if r["status"] in ("timeout", "oom", "skipped")
                    or (r["status"] == "ok" and (r["time"] or 0) > t)
                ),
                None,
            )
            out[family][tool] = hit
    return out
