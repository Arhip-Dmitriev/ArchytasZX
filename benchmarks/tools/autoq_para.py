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

"""AutoQ adapter: AutoQ 2.0 (the parameterised AutoQ-Para has no public release, so symbolic n
is unsupported). d = 2, concrete n, states (every wire initialised): autoq ver with the Init
basis state as precondition, the circuit in its gate set, and the specification state as the
postcondition in .hsl (amplitudes relative to the first, each a power of sqrt2 times an eighth
of a turn; anything else unsupported). AutoQ checks up to a nonzero scalar."""

from __future__ import annotations

import cmath
import math
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.tools import common, rival

ENV = "autoq"
DIMENSIONS = (2,)
BASIS = ["x", "z", "h", "s", "sdg", "t", "tdg", "cx", "cz", "ccx", "swap"]


def supports(inst: Any, point: Any) -> str | None:
    """Qubit subjects, autoq built."""
    if not rival.binary(ENV, "autoq").exists():
        return "not installed: run benchmarks/tools/setup_rivals.sh autoq"
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def _amp(z: complex) -> str:
    """An .hsl constant for z, or unsupported."""
    r, phi = abs(z), cmath.phase(z) / (2 * math.pi)
    j = round(2 * math.log2(r)) if r > 0 else 0
    k = Fraction(round(phi * 8) % 8, 8)
    if abs(r - 2 ** (j / 2)) > 1e-9 or abs((phi * 8) - round(phi * 8)) > 1e-9:
        raise rival.Unsupported("an amplitude outside sqrt2 powers and eighth turns")
    mag = "" if j == 0 else (f"*sqrt2^{j}" if j > 0 else f"/sqrt2^{-j}")
    return f"ei2pi({k.numerator}/{k.denominator}){mag}"


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One autoq ver run."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("symbolic n needs AutoQ-Para, which has no public release")
    why = supports(inst, None)
    if why or d != 2:
        return common.unsupported(why or "qubits only (d = 2)")
    c0 = time.process_time()
    try:
        c = rival.qubit_ir(inst, n)
        qc, inits, posts = rival.qiskit_circuit(c)
        m = len(c.wires)
        if len(inits) != m or posts:
            raise rival.Unsupported("only states (every wire initialised, none post-selected)")
        from qiskit import qasm2

        text = qasm2.dumps(rival.transpiled(qc, BASIS))
        vec = rival.spec_tensor(inst, n, 2).reshape(-1)
        nz = np.flatnonzero(np.abs(vec) > 1e-12)
        if not len(nz):
            raise rival.Unsupported("the zero state")
        a0 = vec[nz[0]]
        terms = [f"{_amp(vec[i] / a0)}|{format(int(i), f'0{m}b')}>" for i in nz]
        pre = "Extended Dirac\n{|" + "".join(str(inits[i]) for i in range(m)) + ">}\n"
        post = "Extended Dirac\n{" + " + ".join(terms) + "}\n"
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    prep = time.process_time() - c0
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "pre.hsl").write_text(pre)
        (p / "c.qasm").write_text(text)
        (p / "post.hsl").write_text(post)
        cmd = [
            str(rival.binary(ENV, "autoq")),
            "ver",
            str(p / "pre.hsl"),
            str(p / "c.qasm"),
            str(p / "post.hsl"),
        ]
        res = rival.run_binary(cmd, float(job["config"]["stages"]["timeout_seconds"]))
    out = res.pop("stdout", "")
    if res["status"] == "ok":
        res["verdict"] = (
            "equal" if "[OK]" in out else ("unequal" if "[failed]" in out else "unknown")
        )
    return rival.finish(inst, res, prep, {"autoq": "3d92eb3"})
