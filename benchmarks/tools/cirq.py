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

"""Cirq adapter: qudit circuits at concrete n and d small enough to simulate; every IR gate a
cirq.MatrixGate on LineQids of dimension d, the final state (or unitary) against the
specification's dense tensor (drivers/sim.py). Simulation, not proof."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import numpy as np

from benchmarks import ir
from benchmarks.tools import common, dense, rival

ENV = "cirq"
PACKAGE = "cirq-core"
MAX_ENTRIES = 1 << 26


def supports(inst: Any, point: Any) -> str | None:
    """Every subject, the environment installed."""
    return rival.installed(ENV)


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One Cirq simulation run."""
    import sympy as sp

    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    c0 = time.process_time()
    try:
        c = common.concrete_circuit(inst, n, d)
        if any(isinstance(g, (ir.Measure, ir.Discard)) for g in c.gates):
            raise rival.Unsupported("measurement or discard (a channel, not a pure map)")
        if any(int(sp.sympify(c.dim(q)).subs(sp.Symbol("d"), d)) != d for q in c.wires):
            raise rival.Unsupported("wires of mixed dimensions")
        m = len(c.wires)
        idx = {q: i for i, q in enumerate(c.wires)}
        inits = {idx[g.q]: g.v % d for g in c.gates if isinstance(g, ir.Init)}
        posts = {idx[g.q]: g.v % d for g in c.gates if isinstance(g, ir.Post)}
        state = len(inits) == m
        axes = m if state else 2 * m
        if axes > rival.MAX_AXES:
            raise rival.Unsupported(f"{axes} tensor axes: above numpy's {rival.MAX_AXES}")
        if d**axes > MAX_ENTRIES:
            raise rival.Unsupported(f"{m} qudits of dimension {d}: too large to simulate")
        gates = []
        for g in c.gates:
            if isinstance(g, (ir.Init, ir.Post)):
                continue
            if isinstance(g, ir.Swap):
                op = np.eye(d * d).reshape(d, d, d, d).transpose(1, 0, 2, 3)
            else:
                op = dense.gate_tensor(g, [d] * len(ir.gate_wires(g)), n or 0, d)
            k = len(ir.gate_wires(g))
            mat = op.reshape(d**k, d**k)
            gates.append([[idx[q] for q in ir.gate_wires(g)], mat.real.tolist(), mat.imag.tolist()])
        scalar = complex(sp.N(sp.sympify(c.scalar).subs(sp.Symbol("d"), d)))
        rhs = rival.spec_tensor(inst, n, d)
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
        np.save(fh, rhs)
        path = fh.name
    payload = {
        "tool": "cirq",
        "wires": m,
        "d": d,
        "gates": gates,
        "scalar": [scalar.real, scalar.imag],
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
