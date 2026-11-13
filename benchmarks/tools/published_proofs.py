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

"""Published-proofs adapter, B1 only: for a subject with a published machine-checked proof, the
proof files re-checked by their own checker (coqc with the project's _CoqProject load paths, in
the opam root of setup_rivals.sh), timed, with their lines and bytes. Proofs are at the size
their authors state, so only the symbolic run is made. The verdict is equal when every file is
accepted."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks.tools import common, rival

COQ_HOME = Path(
    os.environ.get("BENCH_COQ_HOME", str(Path.home() / ".cache" / "archytaszx-bench" / "coq"))
)
SRC = COQ_HOME / "src"
PROOFS: dict[str, list[tuple[str, str]]] = {
    "bernstein_vazirani": [("SQIR-2023", "examples/examples/BernsteinVazirani.v")],
    "deutsch": [("SQIR", "examples/examples/Deutsch.v")],
    "deutsch_jozsa": [("SQIR-2023", "examples/examples/DeutschJozsa.v")],
    "grover": [
        ("SQIR-2023", "examples/examples/Grover.v"),
        ("CoqQ", "src/example/coqq_paper/example.v"),
    ],
    "phase_estimation_digit": [("SQIR-2023", "examples/examples/QPE.v")],
    "superdense_coding": [("SQIR", "examples/examples/Superdense.v")],
    "teleportation": [
        ("SQIR", "examples/examples/Teleport.v"),
        ("VyZX", "src/Examples/Teleportation.v"),
    ],
    "ghz": [("SQIR-2023", "examples/ghz/GHZ.v")],
    "simon": [("SQIR-2023", "examples/examples/Simon.v")],
    "wiesner_money": [("SQIR-2023", "examples/examples/Wiesner.v")],
    "qft": [("CoqQ", "src/example/coqq_paper/example.v")],
    "phase_estimation": [("CoqQ", "src/example/coqq_paper/example.v")],
    "hhl": [("CoqQ", "src/example/coqq_paper/example.v")],
    "hidden_linear_function": [("CoqQ", "src/example/coqq_paper/example.v")],
    "repetition_code": [("SQIR", "examples/error-correction/ThreeQubitCode.v")],
    "shor_code": [("SQIR", "examples/error-correction/NineQubitCode.v")],
    "bell_state": [("VyZX", "src/Examples/Bell.v")],
    "qudit_swap": [("VyZX", "src/Examples/ThreeCNOTs.v")],
}


SWITCH = {"CoqQ": "coqq", "SQIR-2023": "sqir23"}


def coqc(project: str = "SQIR") -> Path:
    """coqc of the project's opam switch (setup_rivals.sh coq)."""
    return COQ_HOME / "opam" / SWITCH.get(project, "bench") / "bin" / "coqc"


def installed() -> str | None:
    """None when every project's coqc and source tree are in place, else the reason."""
    for p in sorted({p for files in PROOFS.values() for p, _ in files}):
        if not (coqc(p).exists() and (SRC / p).is_dir()):
            return f"{p} not installed: run benchmarks/tools/setup_rivals.sh coq"
    return None


def supports(inst: Any, point: Any) -> str | None:
    """Subjects in PROOFS, S equations only, coqc installed."""
    if inst.subject.id not in PROOFS:
        return "no published machine-checked proof of this subject"
    if not inst.truth:
        return "published proofs exist for true equations only (B1)"
    if not all(coqc(p).exists() for p, _ in PROOFS[inst.subject.id]):
        return "not installed: run benchmarks/tools/setup_rivals.sh coq"
    return None


def _flags(project: Path) -> list[str]:
    """The -R, -Q and -I load paths of the project's _CoqProject."""
    out: list[str] = []
    for line in (project / "_CoqProject").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ("-R", "-Q") and len(parts) == 3:
            out += [parts[0], str(project / parts[1]), parts[2]]
        elif parts and parts[0] == "-I" and len(parts) == 2:
            out += ["-I", str(project / parts[1])]
    return out


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """The checker on every proof file of the subject."""
    why = supports(inst, None)
    if why:
        return common.unsupported(why)
    if job.get("point") is not None:
        return common.unsupported("published proofs are checked once, at their stated size")
    limit = float(job["config"]["stages"]["timeout_seconds"])
    files, cpu, wall, ok, detail = [], 0.0, 0.0, True, ""
    path0 = os.environ.get("PATH", "")
    c0 = time.process_time()
    for project, rel in PROOFS[inst.subject.id]:
        root = SRC / project
        path = root / rel
        if not path.exists():
            return common.unsupported(f"{project} proof file {rel} missing")
        text = path.read_bytes()
        os.environ["PATH"] = str(coqc(project).parent) + os.pathsep + path0
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / (path.stem + ".vo"))
            cmd = [str(coqc(project)), *_flags(root), "-o", target, str(path)]
            res = rival.run_binary(cmd, limit)
        cpu += res.get("cpu", 0.0)
        wall += res.get("wall", 0.0)
        accepted = res["status"] == "ok" and res.get("returncode") == 0
        ok &= accepted
        if not accepted:
            detail = (res.get("detail") or res.get("stderr") or res.get("stdout", ""))[-300:]
        files.append(
            {
                "project": project,
                "file": rel,
                "lines": text.count(b"\n"),
                "bytes": len(text),
                "cpu": res.get("cpu"),
                "status": res["status"],
            }
        )
    prep = time.process_time() - c0 - cpu
    res = {
        "status": "ok",
        "verdict": "equal" if ok else "unknown",
        "cpu": cpu,
        "wall": wall,
        "files": files,
        "proof_lines": sum(f["lines"] for f in files),
        "proof_bytes": sum(f["bytes"] for f in files),
    }
    if detail:
        res["detail"] = detail
    return rival.finish(
        inst,
        res,
        max(prep, 0.0),
        {p: "pinned in setup_rivals.sh" for p, _ in PROOFS[inst.subject.id]},
    )
