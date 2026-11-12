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

"""Shared rival preparation (untimed, in the worker) and the driver call (timed, in the tool's
own environment).

Preparation turns the expanded qubit circuit into a Qiskit circuit and QASM 2 (Init and Post
wires recorded beside it), the specification into its dense tensor or its PyZX graph, and the
IR gates into dense matrices for qudit tools. A driver reads one JSON payload, does the tool's
work and prints one JSON line with its CPU and wall seconds.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import resource
import subprocess
import tempfile
import time
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import ir, stages, subject
from benchmarks.tools import common

VENVS = Path(__file__).resolve().parent.parent / ".venvs"
DRIVERS = Path(__file__).resolve().parent / "drivers"
DENSE_CAP = 1 << 24
MAX_AXES = 64


class Unsupported(Exception):
    """The tool cannot run this instance; the message says why."""


def python(env: str) -> Path:
    """The interpreter of a tool environment."""
    return VENVS / env / "bin" / "python"


def installed(env: str) -> str | None:
    """None when the environment exists, else the reason."""
    return None if python(env).exists() else f"environment benchmarks/.venvs/{env} not installed"


def qubit_ir(inst: Any, n: int | None) -> ir.Circuit:
    """The expanded circuit at d = 2."""
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        raise Unsupported(f"fixed dimension {inst.subject.dimension}")
    c = common.concrete_circuit(inst, n, 2)
    for g in c.gates:
        if isinstance(g, (ir.Measure, ir.Discard)):
            raise Unsupported("measurement or discard (a channel, not a pure map)")
    if any(str(c.dim(q)) not in ("d", "2") for q in c.wires):
        raise Unsupported("a wire of dimension other than d")
    return c


def _num(expr: Any) -> float:
    import sympy as sp

    return float(sp.N(sp.sympify(expr).subs({sp.Symbol("d"): 2})))


def qiskit_circuit(
    c: ir.Circuit, unit_scalar: bool = True
) -> tuple[Any, dict[int, int], dict[int, int]]:
    """The unitary part of a qubit IR circuit as a Qiskit circuit (qubit i is wire i), the Init
    values and the Post values by wire index; the circuit scalar's argument is the global
    phase, a scalar of modulus other than 1 refused unless ``unit_scalar`` is False (its
    magnitude then left to the caller, :func:`scalar_parts` of ``c.scalar``)."""
    import sympy as sp
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate, UnitaryGate, XGate, ZGate

    idx = {q: i for i, q in enumerate(c.wires)}
    qc = QuantumCircuit(len(c.wires))
    inits: dict[int, int] = {}
    posts: dict[int, int] = {}
    for g in c.gates:
        if isinstance(g, ir.Init):
            inits[idx[g.q]] = g.v % 2
        elif isinstance(g, ir.Post):
            posts[idx[g.q]] = g.v % 2
        elif isinstance(g, ir.Prepare):
            from qiskit.circuit.library import StatePreparation

            ws = list(g.qs)
            vec = np.array([complex(a, b) for a, b in g.vector])
            vec = vec.reshape([2] * len(ws)).transpose(list(range(len(ws)))[::-1]).reshape(-1)
            for q in ws:
                inits[idx[q]] = 0
            qc.append(StatePreparation(vec / np.linalg.norm(vec)), [idx[q] for q in ws])
        elif isinstance(g, ir.Unitary):
            ws = list(g.qs)
            mat = np.array([[complex(a, b) for a, b in row] for row in g.matrix])
            qc.append(UnitaryGate(_little(mat, len(ws))), [idx[q] for q in ws])
        elif isinstance(g, ir.X):
            if round(_num(g.a)) % 2:
                qc.x(idx[g.q])
        elif isinstance(g, ir.Z):
            a = _num(g.a)
            if abs(a - round(a)) > 1e-12:
                qc.p(math.pi * a, idx[g.q])
            elif round(a) % 2:
                qc.z(idx[g.q])
        elif isinstance(g, ir.F):
            for _ in range(g.p % 4):
                qc.h(idx[g.q])
        elif isinstance(g, (ir.ZPhase, ir.XPhase)):
            t = _num(g.rule.replace("k", "1") if g.rule else g.entries.get(1, 0))
            if isinstance(g, ir.XPhase):
                qc.h(idx[g.q])
            qc.p(2 * math.pi * t, idx[g.q])
            if isinstance(g, ir.XPhase):
                qc.h(idx[g.q])
        elif isinstance(g, ir.CX):
            qc.cx(idx[g.c], idx[g.t])
        elif isinstance(g, ir.CZ):
            qc.cz(idx[g.c], idx[g.t])
        elif isinstance(g, ir.Swap):
            qc.swap(idx[g.a], idx[g.b])
        elif isinstance(g, (ir.MCX, ir.MCZ)):
            vals = g.values if g.values is not None else (1,) * len(g.cs)
            state = "".join(str(v % 2) for v in reversed(vals))
            base = XGate() if isinstance(g, ir.MCX) else ZGate()
            if g.a % 2:
                qc.append(
                    base.control(len(g.cs), ctrl_state=state), [*(idx[q] for q in g.cs), idx[g.t]]
                )
        elif isinstance(g, ir.PhaseFn):
            qs = list(g.qs)
            m = len(qs)
            diag = []
            for k in range(2**m):
                x = tuple(reversed([(k >> i) & 1 for i in range(m)]))
                diag.append(complex(np.exp(2j * np.pi * float(sp.Rational(g.f(x))))))
            qc.append(DiagonalGate(diag), [idx[q] for q in reversed(qs)])
        elif isinstance(g, (ir.Perm, ir.Oracle)):
            from benchmarks.tools import dense

            ws = list(ir.gate_wires(g))
            op = dense._op_from(_basis_fn(g, [2] * len(ws)), [2] * len(ws))
            mat = op.reshape(2 ** len(ws), 2 ** len(ws))
            qc.append(UnitaryGate(_little(mat, len(ws))), [idx[q] for q in ws])
        else:
            raise Unsupported(f"gate {type(g).__name__}")
    if not unit_scalar:
        zero, _, arg = scalar_parts(c.scalar)
        qc.global_phase = 0.0 if zero else arg
        return qc, inits, posts
    val = complex(sp.N(sp.sympify(c.scalar)))
    if abs(abs(val) - 1) > 1e-12:
        raise Unsupported("a circuit scalar of modulus other than 1")
    qc.global_phase = float(np.angle(val))
    return qc, inits, posts


def _basis_fn(g: Any, ds: list[int]) -> Any:
    if isinstance(g, ir.Oracle):
        return lambda x: (1.0, (*x[:-1], (x[-1] + g.f(tuple(x[:-1]))) % ds[-1]))
    total = math.prod(ds)
    return lambda x: (
        1.0,
        tuple(int(v) for v in np.unravel_index(g.p(int(np.ravel_multi_index(x, ds))) % total, ds)),
    )


def _little(mat: np.ndarray, m: int) -> np.ndarray:
    """A matrix on big-endian wire order rewritten for Qiskit's little-endian order."""
    t = mat.reshape([2] * (2 * m))
    perm = list(range(m))[::-1] + [m + i for i in range(m)][::-1]
    return t.transpose(perm).reshape(2**m, 2**m)


def qasm(qc: Any) -> str:
    """OpenQASM 2 of ``qc`` in the gates PyZX and QuiZX read."""
    return qasm_and_phase(qc)[0]


def qasm_and_phase(qc: Any) -> tuple[str, float]:
    """OpenQASM 2 of ``qc`` transpiled to qelib1 gates, and the transpiled circuit's global
    phase (QASM 2 has none)."""
    from qiskit import qasm2, transpile

    t = transpile(qc, basis_gates=QELIB_BASIS, optimization_level=0)
    return exact_angles(qasm2.dumps(t)), float(t.global_phase)


QELIB_BASIS = ["cx", "cz", "h", "x", "z", "s", "sdg", "t", "tdg", "rz", "ccx"]


def qasm_and_zx_phase(qc: Any) -> tuple[str, float]:
    """As :func:`qasm_and_phase`, the phase for a reader taking rz(theta) as diag(1, e^(i
    theta)) (PyZX, QuiZX): Qiskit's global phase minus theta/2 for every rz."""
    from qiskit import qasm2, transpile

    t = transpile(qc, basis_gates=QELIB_BASIS, optimization_level=0)
    shift = sum(float(ins.operation.params[0]) / 2 for ins in t.data if ins.operation.name == "rz")
    return exact_angles(qasm2.dumps(t)), float(t.global_phase) - shift


_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_INT64 = 2**63


def angle_text(a: float) -> str:
    """``a`` radians as ``p*pi/q`` when a/pi has a convergent p/q within 2^-50 relative with
    |p|, q < 2^63, else as a fixed-point decimal without exponent."""
    r = Fraction(a / math.pi)
    if r == 0:
        return "0"
    h0, h1, k0, k1 = 0, 1, 1, 0
    x = r
    while True:
        m = math.floor(x)
        h0, h1 = h1, m * h1 + h0
        k0, k1 = k1, m * k1 + k0
        if abs(h1) >= _INT64 or k1 >= _INT64:
            break
        if abs(Fraction(h1, k1) - r) <= abs(r) * Fraction(1, 2**50):
            p, q = h1, k1
            num = "pi" if abs(p) == 1 else f"{abs(p)}*pi"
            return ("-" if p < 0 else "") + num + ("" if q == 1 else f"/{q}")
        if x == m:
            break
        x = 1 / (x - m)
    return format(Decimal(repr(a)), "f")


def exact_angles(text: str) -> str:
    """QASM 2 ``text`` with every gate parameter that is a bare number rewritten by
    :func:`angle_text`; other parameters unchanged."""

    def one(par: str) -> str:
        return angle_text(float(par)) if _FLOAT.fullmatch(par.strip()) else par

    def params(mt: re.Match[str]) -> str:
        return "(" + ",".join(one(par) for par in mt.group(1).split(",")) + ")"

    out = []
    for line in text.splitlines(keepends=True):
        head = line.lstrip()
        if head.startswith(("OPENQASM", "include", "qreg", "creg", "//")):
            out.append(line)
        else:
            out.append(re.sub(r"\(([^()]*)\)", params, line, count=1))
    return "".join(out)


class SpecBound(Exception):
    """The engine's loader refused the specification script (an input bound or another load
    error); the message is the loader's."""


def spec_diagram(inst: Any, n: int | None, d: int) -> Any:
    """The specification diagram at (n, d), instantiated; SpecBound when the loader refuses
    the script."""
    from archytaszx.semantics import check

    sc = subject.script(inst, n, d)
    try:
        _, right = stages.diagrams(sc.text)
    except RuntimeError as exc:
        raise SpecBound(str(exc).splitlines()[0][:400]) from exc
    return check.instantiate(right, {})


def spec_tensor(inst: Any, n: int | None, d: int) -> np.ndarray:
    """The specification's dense tensor (outputs then inputs) by the ArchytasZX oracle."""
    from archytaszx.semantics import check
    from archytaszx.semantics.contract_numeric import ContractSizeError

    right = spec_diagram(inst, n, d)
    if not right.is_pure():
        raise Unsupported("a mixed specification")
    if len(right.boundary_outputs) + len(right.boundary_inputs) > MAX_AXES:
        raise Unsupported(f"the specification has more than {MAX_AXES} axes")
    try:
        return np.asarray(check.contract(right, max_elements=DENSE_CAP).tensor)
    except ContractSizeError as exc:
        raise Unsupported("the specification tensor is too large") from exc


def spec_pyzx(inst: Any, n: int | None) -> dict[str, Any]:
    """The specification at d = 2 as a ZX graph (vertex types and phases in units of pi, edges
    with their types, boundary vertices): Z and X spiders with their phases, each F a Z vertex
    with a Hadamard edge on its input side, and the diagram's exact scalar as
    :func:`scalar_parts`; any other generator is unsupported."""
    from archytaszx.diagram.graph import Direction

    right = spec_diagram(inst, n, 2)
    verts: list[list[Any]] = []
    vid: dict[Any, int] = {}
    for nid, node in right.nodes.items():
        name = node.generator_type.name
        if name not in ("Z", "X", "F"):
            raise Unsupported(f"generator {name} in the specification")
        ph = node.phase
        if ph is not None and getattr(ph, "is_weighted", False):
            raise Unsupported("a weighted spider in the specification")
        t = (
            Fraction(str(ph.get(1).to_sympy_turns()))
            if (ph is not None and name != "F")
            else Fraction(0)
        )
        vid[nid] = len(verts)
        verts.append(["X" if name == "X" else "Z", str((2 * t) % 2)])

    def htype(ref: Any) -> int:
        return (
            2
            if right.nodes[ref.node_id].generator_type.name == "F"
            and ref.direction is Direction.INPUT
            else 1
        )

    edges = [
        [vid[w.a.node_id], vid[w.b.node_id], 2 if 2 in (htype(w.a), htype(w.b)) else 1]
        for w in right.wires
    ]
    ins, outs = [], []
    for ref in right.boundary_inputs:
        ins.append(len(verts))
        edges.append([len(verts), vid[ref.node_id], htype(ref)])
        verts.append(["B", "0"])
    for ref in right.boundary_outputs:
        outs.append(len(verts))
        edges.append([vid[ref.node_id], len(verts), 1])
        verts.append(["B", "0"])
    return {
        "vertices": verts,
        "edges": edges,
        "inputs": ins,
        "outputs": outs,
        "scalar": scalar_parts(right.scalar.to_sympy()),
    }


def scalar_parts(value: Any) -> list[Any]:
    """[zero, log|value|, arg value] of an exact scalar, by sympy at 30 digits."""
    import sympy as sp

    v = sp.sympify(value)
    if v.is_zero:
        return [True, 0.0, 0.0]
    return [False, float(sp.N(sp.log(sp.Abs(v)), 30)), float(sp.N(sp.arg(v), 30))]


def reference_circuit(inst: Any, n: int | None, d: int) -> ir.Circuit:
    """The equation's reference circuit at (n, d): its own, else the specification as one
    dense gate (Prepare for a state, Unitary for a square unitary map)."""
    eq = inst.equation
    if eq.reference is not None:
        return common.expand(eq.reference(n, d), n or 0)
    t = spec_tensor(inst, n, d)
    right = spec_diagram(inst, n, d)
    no, ni = len(right.boundary_outputs), len(right.boundary_inputs)
    if no == 0:
        raise Unsupported("the specification has no wires")
    wires = tuple(f"o{i}" for i in range(no))
    if ni == 0:
        vec = t.reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            raise Unsupported("the specification is the zero state")
        pairs = tuple((float(v.real), float(v.imag)) for v in vec / norm)
        return ir.Circuit(wires, (ir.Prepare(wires, pairs),), {}, {}, "1")
    if ni != no:
        raise Unsupported("the specification is an isometry, not a unitary")
    mat = t.reshape(d**no, d**ni)
    if not np.allclose(mat @ mat.conj().T, np.eye(d**no), atol=1e-8):
        raise Unsupported("the specification is not unitary")
    rows = tuple(tuple((float(v.real), float(v.imag)) for v in row) for row in mat)
    return ir.Circuit(wires, (ir.Unitary(wires, rows),), {}, {}, "1")


def _rival_call(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """``fn`` with the preparation timer paused (worker.end_prepare) and the rival stage
    announced before it, and the timer resumed with the budget left after it."""

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        from benchmarks.worker import end_prepare, mark, resume_prepare

        end_prepare()
        mark("rival")
        try:
            return fn(*args, **kwargs)
        finally:
            resume_prepare()

    return call


@_rival_call
def run_driver(
    env: str, driver: str, payload: dict[str, Any], limit: float, wall_factor: float = 4.0
) -> dict[str, Any]:
    """The driver's JSON result, or a status: timeout, oom, error."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        path = fh.name

    def caps() -> None:
        cpu = max(1, int(limit))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))

    w0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(python(env)), "-I", str(DRIVERS / f"{driver}.py"), path],
            capture_output=True,
            text=True,
            timeout=limit * wall_factor,
            preexec_fn=caps,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "limit": "wall", "wall": time.perf_counter() - w0}
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        status = (
            "timeout"
            if proc.returncode in (-24, 152)
            else (
                "oom" if proc.returncode in (-9, 137) or "MemoryError" in proc.stderr else "error"
            )
        )
        out = {
            "status": status,
            "detail": proc.stderr.strip()[-2000:],
            "wall": time.perf_counter() - w0,
        }
        if status == "timeout":
            out["limit"] = "cpu"
        return out
    return {"status": "ok", **json.loads(proc.stdout.strip().splitlines()[-1])}


def binary(env: str, name: str) -> Path:
    """A built tool binary under benchmarks/.venvs/ENV/bin."""
    return VENVS / env / "bin" / name


def lift_address_space() -> None:
    """Raises the soft RLIMIT_AS of the calling process to its hard limit."""
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


@_rival_call
def run_binary(
    cmd: list[str], limit: float, wall_factor: float = 4.0, lift_memory: bool = False
) -> dict[str, Any]:
    """A tool binary's stdout, its CPU (children's rusage) and wall seconds, or a status;
    ``lift_memory`` runs it without the worker's soft address-space cap."""

    def caps() -> None:
        cpu = max(1, int(limit))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        if lift_memory:
            lift_address_space()

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    w0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=limit * wall_factor,
            preexec_fn=caps,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "limit": "wall", "wall": time.perf_counter() - w0}
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    out: dict[str, Any] = {
        "status": "ok",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cpu": cpu,
        "wall": time.perf_counter() - w0,
        "peak_mb": round(after.ru_maxrss / 1024, 1),
    }
    if proc.returncode not in (0, 1):
        killed = proc.returncode in (-24, 152)
        oom = proc.returncode in (-9, 137)
        out["status"] = "timeout" if killed else ("oom" if oom else "error")
        if killed:
            out["limit"] = "cpu"
        out["detail"] = (proc.stderr or proc.stdout).strip()[-400:]
    return out


def explicit_reference(inst: Any, n: int | None, d: int) -> ir.Circuit:
    """The equation's own reference circuit, expanded; unsupported when it has none."""
    if inst.equation.reference is None:
        raise Unsupported("no gate-level reference circuit for this equation")
    return common.expand(inst.equation.reference(n, d), n or 0)


def transpiled(qc: Any, basis: list[str]) -> Any:
    """``qc`` transpiled to ``basis``; unsupported when it cannot be expressed exactly."""
    from qiskit import transpile

    try:
        return transpile(qc, basis_gates=basis, optimization_level=0)
    except Exception as exc:
        raise Unsupported(f"not expressible in the gate set {basis}: {str(exc)[:120]}") from exc


def finish(
    inst: Any, res: dict[str, Any], prep_cpu: float, versions: dict[str, Any]
) -> dict[str, Any]:
    """The run record of a rival: the check stage and the outcome."""
    verdict = res.get("verdict")
    stage = {k: v for k, v in res.items() if k != "verdict"}
    stage.setdefault("cpu", 0.0)
    stage.setdefault("wall", 0.0)
    return {
        "expected": common.expected(inst),
        "verdict": verdict,
        "outcome": common.outcome(inst, res["status"], verdict),
        "wrong": common.outcome(inst, res["status"], verdict) == "wrong",
        "stages": {"prepare": {"status": "ok", "cpu": prep_cpu, "wall": prep_cpu}, "check": stage},
        "tool_version": versions,
    }


def version(env: str, package: str) -> str | None:
    """The installed version of ``package`` in a tool environment."""
    try:
        out = subprocess.run(
            [
                str(python(env)),
                "-c",
                f"import importlib.metadata as m; print(m.version('{package}'))",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None
