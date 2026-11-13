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

"""Feynman adapter: d = 2, concrete n; the circuit and the equation's own reference circuit,
both Clifford+T, written in .qc (Init wires ancillas, X where Init is 1) and compared by
feynver (Equal, Not equal, Inconclusive). Post-selected wires are unsupported."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks.tools import common, rival

ENV = "feynman"
DIMENSIONS = (2,)
BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "cx", "ccx", "swap", "p"]
EIGHTHS = {1: ["T"], 2: ["P"], 3: ["P", "T"], 4: ["Z"], 5: ["Z", "T"], 6: ["P*"], 7: ["T*"]}
NAMES = {
    "h": "H",
    "x": "X",
    "z": "Z",
    "s": "P",
    "sdg": "P*",
    "t": "T",
    "tdg": "T*",
    "cx": "cnot",
    "ccx": "tof",
    "swap": "swap",
}


def installed() -> str | None:
    """None when feynver is built, else the reason."""
    if not rival.binary(ENV, "feynver").exists():
        return "not installed: run benchmarks/tools/setup_rivals.sh haskell"
    return None


def supports(inst: Any, point: Any) -> str | None:
    """Equations with a reference circuit at d = 2, feynver built."""
    why = installed()
    if why:
        return why
    if inst.equation.reference is None:
        return "no gate-level reference circuit for this equation"
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def qc_text(c: Any) -> str:
    """The .qc file of a qubit IR circuit."""
    qc, inits, posts = rival.qiskit_circuit(c)
    if posts:
        raise rival.Unsupported("post-selected wires")
    t = rival.transpiled(qc, BASIS)
    names = [f"q{i}" for i in range(len(c.wires))]
    lines = [
        ".v " + " ".join(names),
        ".i " + " ".join(n for i, n in enumerate(names) if i not in inits),
        "",
        "BEGIN",
    ]
    lines += [f"X {names[i]}" for i, v in inits.items() if v]
    for inst_ in t.data:
        op = inst_.operation.name
        if op == "barrier":
            continue
        if op == "p":
            k = float(inst_.operation.params[0]) / (math.pi / 4)
            if abs(k - round(k)) > 1e-9:
                raise rival.Unsupported("a phase that is not a multiple of pi/4")
            q = names[t.find_bit(inst_.qubits[0]).index]
            lines += [f"{g} {q}" for g in EIGHTHS.get(round(k) % 8, [])]
            continue
        if op not in NAMES:
            raise rival.Unsupported(f"gate {op} outside Clifford+T")
        if op == "x":
            lines.append(f"tof {names[t.find_bit(inst_.qubits[0]).index]}")
            continue
        lines.append(NAMES[op] + " " + " ".join(names[t.find_bit(q).index] for q in inst_.qubits))
    lines.append("END")
    return "\n".join(lines) + "\n"


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One feynver equivalence check."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("no symbolic n or d")
    why = supports(inst, None)
    if why or d != 2:
        return common.unsupported(why or "qubits only (d = 2)")
    c0 = time.process_time()
    try:
        a = qc_text(rival.qubit_ir(inst, n))
        b = qc_text(rival.explicit_reference(inst, n, 2))
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    prep = time.process_time() - c0
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = Path(tmp) / "a.qc", Path(tmp) / "b.qc"
        pa.write_text(a)
        pb.write_text(b)
        res = rival.run_binary(
            [str(rival.binary(ENV, "feynver")), str(pa), str(pb)],
            float(job["config"]["stages"]["timeout_seconds"]),
        )
    text = res.pop("stdout", "")
    if res["status"] == "ok":
        res["verdict"] = (
            "equal"
            if text.startswith("Equal")
            else ("unequal" if text.startswith("Not equal") else "unknown")
        )
    return rival.finish(inst, res, prep, {"feynman": "d2c382a"})
