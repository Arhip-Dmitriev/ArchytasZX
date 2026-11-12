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

"""Qiskit adapter: d = 2, concrete n small enough to simulate; the circuit's Operator (or
Statevector when every input is initialised) against the specification's dense tensor, exactly
and up to a global phase (drivers/sim.py). Simulation, not proof."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import numpy as np

from benchmarks.tools import common, rival

ENV = "qiskit"
PACKAGE = "qiskit"
PROBE = ("qiskit.qasm2", "qiskit.quantum_info", "numpy")
DIMENSIONS = (2,)
MAX_STATE_QUBITS = 26
MAX_MAP_QUBITS = 13


def supports(inst: Any, point: Any) -> str | None:
    """d = 2 subjects or general-d ones, the environment installed."""
    why = rival.installed(ENV)
    if why:
        return why
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One Qiskit simulation run."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    if d != 2:
        return common.unsupported("qubits only (d = 2)")
    c0 = time.process_time()
    try:
        c = rival.qubit_ir(inst, n)
        m = len(c.wires)
        state = sum(1 for g in c.gates if type(g).__name__ == "Init") == m
        if m > (MAX_STATE_QUBITS if state else MAX_MAP_QUBITS):
            return common.unsupported(f"{m} qubits: too large to simulate")
        qc, inits, posts = rival.qiskit_circuit(c)
        rhs = rival.spec_tensor(inst, n, 2)
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
        np.save(fh, rhs)
        path = fh.name
    text, phase = rival.qasm_and_phase(qc)
    payload = {
        "tool": "qiskit",
        "qasm": text,
        "global_phase": phase,
        "wires": m,
        "inits": {str(k): v for k, v in inits.items()},
        "posts": {str(k): v for k, v in posts.items()},
        "rhs": path,
    }
    prep = time.process_time() - c0
    try:
        res = rival.run_driver(
            ENV, "sim", payload, float(job["config"]["stages"]["timeout_seconds"])
        )
    finally:
        os.unlink(path)
    return rival.finish(inst, res, prep, {PACKAGE: rival.version(ENV, PACKAGE)})
