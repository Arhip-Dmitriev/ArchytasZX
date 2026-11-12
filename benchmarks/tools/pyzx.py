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

"""PyZX adapter: d = 2 and concrete n; the circuit through QASM, the specification as a PyZX
graph; full_reduce of circuit . spec^dagger to the identity for maps, inner products for states
(driver drivers/zx.py)."""

from __future__ import annotations

import time
from typing import Any

from benchmarks.tools import common, rival

ENV = "pyzx"
PACKAGE = "pyzx"


def supports(inst: Any, point: Any) -> str | None:
    """d = 2 subjects or general-d ones, the environment installed."""
    why = rival.installed(ENV)
    if why:
        return why
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def zx_run(tool: str, env: str, package: str, inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One PyZX or QuiZX run at the job's point (symbolic n or d unsupported)."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    if d != 2:
        return common.unsupported("qubits only (d = 2)")
    c0 = time.process_time()
    try:
        c = rival.qubit_ir(inst, n)
        qc, inits, posts = rival.qiskit_circuit(c)
        text, phase = rival.qasm_and_zx_phase(qc)
        payload = {
            "tool": tool,
            "qasm": text,
            "phase": phase,
            "inits": {str(k): v for k, v in inits.items()},
            "posts": {str(k): v for k, v in posts.items()},
            "wires": len(c.wires),
            "rhs": rival.spec_pyzx(inst, n),
        }
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    prep = time.process_time() - c0
    res = rival.run_driver(env, "zx", payload, float(job["config"]["stages"]["timeout_seconds"]))
    out = rival.finish(inst, res, prep, {package: rival.version(env, package)})
    if res.get("vertices_after") is not None:
        out["b3"] = {
            "tool": {"vertices": res["vertices_after"], "edges": res.get("edges_after")},
            "naive": {"vertices": res.get("vertices_before")},
        }
    return out


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One PyZX run."""
    return zx_run("pyzx", ENV, PACKAGE, inst, job)
