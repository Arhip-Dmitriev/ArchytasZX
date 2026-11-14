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

"""MQT QCEC adapter: d = 2, concrete n; the circuit against the equation's reference circuit
(its own, else the specification as one dense gate, rival.reference_circuit), both through QASM
with the global phase appended as gates, Init wires ancillary (prepared |0>, X where Init is 1),
full equivalence checked (drivers/qcec.py). Post-selected wires, and a circuit and reference
of different input or output counts, are unsupported."""

from __future__ import annotations

import math
import time
from typing import Any

from benchmarks.tools import common, rival

ENV = "qcec"
PACKAGE = "mqt.qcec"
PROBE = ("mqt.core.ir", "mqt.qcec")
DIMENSIONS = (2,)


def supports(inst: Any, point: Any) -> str | None:
    """d = 2 subjects or general-d ones, the environment installed."""
    why = rival.installed(ENV)
    if why:
        return why
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def _side(c: Any) -> tuple[str, list[int]]:
    """QASM of a qubit IR circuit with Init wires prepared and its global phase appended
    (:func:`phase_gates`), and its ancillary wires."""
    qc, inits, posts = rival.qiskit_circuit(c)
    if posts:
        raise rival.Unsupported("post-selected wires")
    prep = qc.copy_empty_like()
    prep.global_phase = 0
    for i, v in inits.items():
        if v:
            prep.x(i)
    text, phase = rival.qasm_and_phase(prep.compose(qc))
    return text.rstrip("\n") + "\n" + phase_gates(phase), sorted(inits)


def phase_gates(phase: float) -> str:
    """QASM 2 gates on q[0] whose product is e^(i phase) I: u1, x, u1, x; empty for a phase
    of 0 mod 2 pi."""
    phase = math.remainder(phase, 2 * math.pi)
    if abs(phase) < 1e-12:
        return ""
    a = f"{phase:.20f}"
    return f"u1({a}) q[0];\nx q[0];\nu1({a}) q[0];\nx q[0];\n"


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One equivalence check."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    if d != 2:
        return common.unsupported("qubits only (d = 2)")
    c0 = time.process_time()
    try:
        c, r = rival.qubit_ir(inst, n), rival.reference_circuit(inst, n, 2)
        circ, anc = _side(c)
        ref, ref_anc = _side(r)
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    shape = (len(c.wires) - len(anc), len(c.wires))
    ref_shape = (len(r.wires) - len(ref_anc), len(r.wires))
    if shape != ref_shape:
        return common.unsupported(
            f"the circuit ({shape[0]} inputs, {shape[1]} outputs) and the reference "
            f"({ref_shape[0]} inputs, {ref_shape[1]} outputs) differ in boundary"
        )
    payload = {"qasm": circ, "ancillary": anc, "reference": ref, "reference_ancillary": ref_anc}
    prep = time.process_time() - c0
    limit = float(job["config"]["stages"]["timeout_seconds"])
    res = rival.run_driver(ENV, "qcec", payload, limit)
    return rival.finish(inst, res, prep, {PACKAGE: rival.version(ENV, PACKAGE)})
