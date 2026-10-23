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

"""Computing algorithm outputs over a concrete (qudit count n, dimension d) grid.

Algorithms: GHZ, Deutsch-Jozsa (f(x) = x mod d), one Grover iteration (marked w = 1), and Shor
order finding for N = 15, a = 7 (n count qudits, a mod-15 work register, inverse Fourier
transform over d^n; Qiskit and PyZX use controlled multiply-by-7^(2^k) mod 15 from SWAPs and Xs).
ArchytasZX runs each as a Dirac-language program; Qiskit computes the statevector and PyZX
runs ``full_reduce`` on the |0>-plugged circuit graph, both at d = 2 only. An ArchytasZX
result at d = 2 and n <= 8 is compared with Qiskit's statevector outside the timed region.
"""

from __future__ import annotations

from benchmarks.circuit import Unsupported, qiskit_to_pyzx
from benchmarks.harness import Wrong, row, timed

TOOLS = ("archytaszx", "qiskit", "pyzx")
ALGORITHMS = ("ghz", "dj", "grover", "shor")
VALIDATE_MAX_N = 8

_F = "gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }"
SCRIPTS = {
    "ghz": _F
    + """
var M = d^(n-1)
gate FAN = |x>_d|y>_M -> |x>_d|(y + x*(M - 1)/(d - 1)) % M>_M
|p> = |0>_d @ |0>_M
(F @ I)|p>
FAN|p>""",
    "dj": _F
    + """
func f(x) = x % d
gate U = |x>|y> -> |x>|(y + f(x)) % d>
|p>_d = |0>_d^@n @ |1>_d
(F^@n @ F)|p>
U|p>
(F^@n @ I)|p>""",
    "grover": _F
    + """
var N = d^n, w = 1
|s>_d = F^@n |0>_d^@n
|p> = |s>
|p> = |p> - 2 |w><w|p>
|p> = 2 |s><s|p> - |p>""",
    "shor": """
var N = 15, a = 7
var Q = d^n
gate FQ = |x>_Q -> 1/sqrt(Q) sum(k in 0..Q){ e^(2*pi*i*x*k/Q) |k>_Q }
gate IQFT = |x>_Q -> 1/sqrt(Q) sum(k in 0..Q){ e^(-2*pi*i*x*k/Q) |k>_Q }
gate U = |x>_Q|y>_N -> |x>_Q|(y * powmod(a, x, N)) % N>_N
|p> = |0>_Q @ |1>_N
(FQ @ I)|p>
U|p>
(IQFT @ I)|p>""",
}
SHOR_N = 15

SIZES = [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 48, 64, 128, 256, 1024]
DIMS = [2, 3, 4, 5, 6, 7, 8, 11, 16]


def cases(quick: bool) -> list[dict]:
    sizes = SIZES[:12] if quick else SIZES
    dims = DIMS[:4] if quick else DIMS
    out = []
    for alg in ALGORITHMS:
        for d in dims:
            for n in sizes:
                for tool in TOOLS:
                    if tool == "archytaszx" or d == 2:
                        out.append(
                            {"suite": "algorithms", "family": alg, "tool": tool, "n": n, "d": d}
                        )
    return out


def supports(case: dict) -> str | None:
    if case["tool"] != "archytaszx" and case["d"] != 2:
        return f"{case['tool']} is qubit-only"
    return None


def _archytaszx(case: dict, repeat: int) -> dict:
    from archytaszx import Program

    src = f"var n = {case['n']}, d = {case['d']}\n{SCRIPTS[case['family']]}\nshow(|p>)"
    holder: list[Program] = []

    def go() -> str:
        holder[:] = [Program()]
        return holder[0](src)

    text, times = timed(go, repeat)
    if case["d"] == 2 and case["n"] <= VALIDATE_MAX_N:
        _validate(case, holder[0].p.vector())
    return row(case, "ok", times, str(text)[:400])


def _validate(case: dict, vec: object) -> None:
    import numpy as np
    from qiskit.quantum_info import Statevector

    ref = Statevector(_qiskit_circuit(case)).data
    got = np.asarray(vec)
    if case["family"] == "shor":
        q = 2 ** case["n"]
        full = np.zeros(q * 16, dtype=complex)
        for x in range(q):
            full[x + q * np.arange(SHOR_N)] = got[x * SHOR_N : (x + 1) * SHOR_N]
        got = full
    overlap = abs(np.vdot(ref, got)) if got.shape == ref.shape else 0.0
    if abs(overlap - 1) > 1e-8:
        raise Wrong(f"differs from the Qiskit statevector (|<ref|got>| = {overlap:.6g})")


def _qiskit_circuit(case: dict):  # type: ignore[no-untyped-def]
    from qiskit import QuantumCircuit

    n, fam = case["n"], case["family"]
    if fam == "ghz":
        qc = QuantumCircuit(n)
        qc.h(0)
        for k in range(n - 1):
            qc.cx(k, k + 1)
        return qc
    if fam == "dj":
        qc = QuantumCircuit(n + 1)
        qc.x(0)
        qc.h(range(n + 1))
        qc.cx(1, 0)
        qc.h(range(1, n + 1))
        return qc
    if fam == "grover":
        qc = QuantumCircuit(n)
        qc.h(range(n))
        _flip(qc, [q for q in range(n) if q != 0])
        qc.h(range(n))
        _flip(qc, range(n))
        qc.h(range(n))
        return qc
    if fam == "shor":
        return _shor_circuit(n)
    raise Unsupported(fam)


def _shor_circuit(n: int):  # type: ignore[no-untyped-def]
    """n count qubits, work qubits n..n+3 at |1>; controlled 7^(2^k) mod 15, inverse QFT."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate

    times7 = QuantumCircuit(4, name="x7")
    times7.swap(0, 1)
    times7.swap(1, 2)
    times7.swap(2, 3)
    times7.x(range(4))
    controlled = times7.to_gate().control(1)
    qc = QuantumCircuit(n + 4)
    qc.x(n)
    qc.h(range(n))
    for k in range(n):
        for _ in range(pow(2, k, 4)):
            qc.append(controlled, [k, n, n + 1, n + 2, n + 3])
    qc.append(QFTGate(n).inverse(), range(n))
    return transpile(
        qc, basis_gates=["h", "cx", "p", "x", "swap", "cswap", "ccx"], optimization_level=0
    )


def _flip(qc, zeros) -> None:  # type: ignore[no-untyped-def]
    """Phase -1 on the basis state with ``zeros`` at 0 and every other qubit at 1."""
    zeros = list(zeros)
    n = qc.num_qubits
    if zeros:
        qc.x(zeros)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    if zeros:
        qc.x(zeros)


def _qiskit(case: dict, repeat: int) -> dict:
    from qiskit.quantum_info import Statevector

    qc = _qiskit_circuit(case)
    _, times = timed(lambda: Statevector(qc), repeat)
    return row(case, "ok", times, f"statevector of {qc.num_qubits} qubits")


def _pyzx(case: dict, repeat: int) -> dict:
    import pyzx as zx

    circ = qiskit_to_pyzx(_qiskit_circuit(case))

    def go() -> int:
        g = circ.to_graph()
        g.apply_state("0" * circ.qubits)
        zx.full_reduce(g)
        return int(g.num_vertices())

    after, times = timed(go, repeat)
    return row(case, "ok", times, f"full_reduce: {len(circ.gates)} gates -> {after} vertices")


def run(case: dict, repeat: int) -> dict:
    fn = {"archytaszx": _archytaszx, "qiskit": _qiskit, "pyzx": _pyzx}[case["tool"]]
    return fn(case, repeat)
