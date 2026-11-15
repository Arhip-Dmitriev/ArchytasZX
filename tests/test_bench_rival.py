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

"""Circuit conversions of benchmarks.tools.rival against Qiskit operators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _qram_circuit() -> Any:
    """The Qiskit circuit of qram/main at n = 2 (7 qubits, idle wires included)."""
    from benchmarks import subject
    from benchmarks.tools import rival

    inst = next(i for i in subject.instances(list(subject.catalogue())) if i.id == "qram/main")
    return rival.qiskit_circuit(rival.qubit_ir(inst, 2))[0]


def test_qasm_keeps_the_map() -> None:
    """qasm_and_phase and transpiled give the operator of the circuit on every input."""
    pytest.importorskip("qiskit")
    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    from benchmarks.tools import rival

    qc = _qram_circuit()
    want = Operator(qc).data
    text, phase = rival.qasm_and_phase(qc)
    back = qasm2.loads(text)
    back.global_phase = phase
    assert np.abs(Operator(back).data - want).max() < 1e-8
    t = rival.transpiled(qc, rival.QELIB_BASIS)
    assert np.abs(Operator(t).data - want).max() < 1e-8


def test_phase_functions_keep_exact_angles() -> None:
    """A PhaseFn on 3 wires is its diagonal, and its rz angles are exact multiples of pi down to
    a turn of 2^-70."""
    pytest.importorskip("qiskit")
    from fractions import Fraction

    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    from benchmarks import ir
    from benchmarks.tools import rival

    vals = [Fraction(0), Fraction(1, 3), Fraction(2, 7), Fraction(1, 2**70), Fraction(5, 8)]

    def f(x: tuple[int, ...]) -> Fraction:
        return vals[(x[0] + 2 * x[1] + 3 * x[2]) % len(vals)] + Fraction(x[0] * x[2], 2**70)

    c = ir.Circuit(("a", "b", "e"), (ir.PhaseFn(("a", "b", "e"), f),))
    qc = rival.qiskit_circuit(c)[0]
    diag = []
    for k in range(8):
        x = (k >> 2 & 1, k >> 1 & 1, k & 1)
        diag.append(np.exp(2j * np.pi * float(f(x))))
    want = rival._little(np.diag(diag), 3)
    assert np.abs(Operator(qc).data - want).max() < 1e-12
    text, phase = rival.qasm_and_phase(qc)
    angles = [line.split("(")[1].split(")")[0] for line in text.splitlines() if "rz(" in line]
    assert angles and all("pi" in a for a in angles)
    assert any("/" in a and int(a.split("/")[1]) % 2**69 == 0 for a in angles)
    back = qasm2.loads(text)
    back.global_phase = phase
    assert np.abs(Operator(back).data - want).max() < 1e-12


def test_scalar_parts_drop_an_imaginary_residue() -> None:
    """The qubitization_chebyshev scalar -(-1)^(1/4)/2 + (-1)^(3/4)/2 is -1/sqrt2."""
    import math

    import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

    from benchmarks.tools import rival

    v = -((-1) ** sp.Rational(1, 4)) / 2 + (-1) ** sp.Rational(3, 4) / 2
    zero, lg, arg = rival.scalar_parts(v)
    assert not zero and abs(lg - math.log(0.5) / 2) < 1e-12 and abs(arg - math.pi) < 1e-12
    with pytest.raises(ValueError, match="not real"):
        rival._real(sp.I)


def test_phase_rules_take_the_point_n() -> None:
    """A ZPhase rule in n and k at n = 4 is diag(1, -1); ZPhase/XPhase entries keep the k = 0
    phase as a global phase."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from benchmarks import ir
    from benchmarks.tools import rival

    rule = ir.ZPhase("a", rule="k*n*(n-1)*(n-2)*(n-3)/(24*d)")
    qc = rival.qiskit_circuit(rival._at_n(ir.Circuit(("a",), (rule,)), 4))[0]
    assert np.abs(Operator(qc).data - np.diag([1, -1])).max() < 1e-12
    ent = ir.Circuit(("a",), (ir.ZPhase("a", entries={0: "1/4", 1: "1/2"}),))
    qc = rival.qiskit_circuit(ent)[0]
    assert np.abs(Operator(qc).data - np.diag([1j, -1])).max() < 1e-12
    xent = ir.Circuit(("a",), (ir.XPhase("a", entries={0: "1/4", 1: "1/2"}),))
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    qc = rival.qiskit_circuit(xent)[0]
    assert np.abs(Operator(qc).data - h @ np.diag([1j, -1]) @ h).max() < 1e-12


def test_allocation_failures_are_oom() -> None:
    """A worker abort on a failed Rust allocation, a worker MemoryError chain and an mmap
    ImportError in prepare classify as oom; an exit without such text stays error."""
    from benchmarks import outcomes

    def killed(detail: str, code: int) -> dict[str, Any]:
        rec = {"tool": "quizx", "outcome": "error", "killed_in": "prepare", "exit": code}
        return outcomes.classify({**rec, "kill": f"exit {code}", "detail": detail})

    rust = killed("memory allocation of 32768 bytes failed\nnote: run with RUST_BACKTRACE=1", -6)
    chain = killed("Error\n\nDuring handling of the above exception:\n\nMemoryError", 1)
    assert (rust["outcome"], chain["outcome"]) == ("oom", "oom")
    assert killed("", -11)["outcome"] == "error"
    msg = "ImportError: /x/mtrand.so: failed to map segment from shared object"
    rec = {"tool": "autoq_para", "outcome": "harness_error", "reason": f"prepare: {msg}"}
    assert outcomes.classify({**rec, "detail": msg})["outcome"] == "oom"


def test_feynman_refuses_a_boundary_mismatch() -> None:
    """feynman._interface tells a 3-wire state from a 2-wire one and an input from an Init."""
    from benchmarks import ir
    from benchmarks.tools import feynman

    two = ir.Circuit(("a", "b"), (ir.Init("a", 0), ir.Init("b", 0)))
    three = ir.Circuit(("a", "b", "c"), (ir.Init("a", 0), ir.Init("b", 0), ir.Init("c", 0)))
    half = ir.Circuit(("a", "b"), (ir.Init("a", 0),))
    assert feynman._interface(two) == (2, frozenset({0, 1}))
    assert feynman._interface(three) != feynman._interface(two) != feynman._interface(half)


def test_engine_memory_errors_are_oom() -> None:
    """An ArchytasZX decide stage failing with the engine's RuntimeError around a MemoryError
    (plain, in a load line, in an @rhs line) is oom; another load error stays load_refused."""
    from benchmarks import outcomes

    def run(detail: str) -> tuple[str, str]:
        rec: dict[str, Any] = {"tool": "archytaszx", "outcome": "error"}
        rec["stages"] = {
            "generate": {"status": "ok"},
            "decide": {"status": "error", "detail": detail},
        }
        out = outcomes.classify(rec)
        return out["outcome"], out["cause"]

    for text in ("error: MemoryError", "load: error: MemoryError", "@rhs =: error: MemoryError:"):
        assert run(f"RuntimeError: {text}\n(detail)") == ("oom", "memory.error")
    assert run("RuntimeError: load: error: bad token") == ("error", "engine.load_refused")


def test_dizx_amounts_take_the_point_n() -> None:
    """DiZX Clifford ops of Z^(n(n+1)/2) X^n at n = 4, d = 3: Z^1, and X^1 as HAD, Z, HAD^3."""
    from benchmarks import ir
    from benchmarks.tools import dizx, rival

    c = ir.Circuit(("a",), (ir.Z("a", "n*(n+1)/2"), ir.X("a", "n")))
    ops = dizx.clifford_ops(rival._at_n(c, 4), 3)
    assert ops == [["Z", [0], 1], ["HAD", [0], 1], ["Z", [0], 1], ["HAD", [0], 3]]


def test_result_encoding_survives_a_memory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker.encode_result drops the span trees when encoding the whole record raises
    MemoryError; an ArchytasZX worker dying of a MemoryError traceback is oom."""
    import json

    from benchmarks import outcomes, worker

    real = json.dumps

    def dumps(obj: Any, **kw: Any) -> str:
        if "spans" in real(obj, **kw):
            raise MemoryError
        return real(obj, **kw)

    monkeypatch.setattr(json, "dumps", dumps)
    out = {"outcome": "timeout", "stages": {"decide": {"status": "timeout", "spans": [1] * 9}}}
    rec = json.loads(worker.encode_result(out)[len(worker.RESULT) :])
    assert rec["result_encode"] == "MemoryError"
    assert rec["stages"] == {"decide": {"status": "timeout"}}
    assert worker.encode_spans([{"name": "spans"}]) is None
    assert worker.encode_spans([{"name": "x"}]) == worker.SPANS + '[{"name": "x"}]'
    killed = {"tool": "archytaszx", "outcome": "error", "killed_in": "decide", "exit": 1}
    detail = 'File "json/encoder.py", line 258, in iterencode\nMemoryError'
    out2 = outcomes.classify({**killed, "kill": "exit 1", "detail": detail})
    assert (out2["outcome"], out2["cause"]) == ("oom", "memory.error")
