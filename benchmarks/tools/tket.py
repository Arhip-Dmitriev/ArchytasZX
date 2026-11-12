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

"""tket adapter, B3 only: the qubit circuit (QASM) through FullPeepholeOptimise; gate and
two-qubit-gate counts before and after (drivers/tket.py). A run that optimises is ok; it gives
no verdict."""

from __future__ import annotations

import time
from typing import Any

from benchmarks.tools import common, rival

ENV = "tket"
PACKAGE = "pytket"
PROBE = ("pytket.passes", "pytket.qasm")
DIMENSIONS = (2,)


def supports(inst: Any, point: Any) -> str | None:
    """d = 2 subjects or general-d ones, the environment installed."""
    why = rival.installed(ENV)
    if why:
        return why
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One optimisation run."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    if d != 2:
        return common.unsupported("qubits only (d = 2)")
    c0 = time.process_time()
    try:
        qc, _, _ = rival.qiskit_circuit(rival.qubit_ir(inst, n))
        payload = {"qasm": rival.qasm(qc)}
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    prep = time.process_time() - c0
    res = rival.run_driver(ENV, "tket", payload, float(job["config"]["stages"]["timeout_seconds"]))
    out = rival.finish(inst, {**res, "verdict": None}, prep, {PACKAGE: rival.version(ENV, PACKAGE)})
    out["outcome"] = res["status"]
    out["b3_only"] = True
    if res["status"] == "ok":
        out["b3"] = {
            "tool": {k: res[k] for k in ("gates_after", "two_qubit_after")},
            "naive": {"gates": res["gates_before"], "two_qubit": res["two_qubit_before"]},
        }
    return out
