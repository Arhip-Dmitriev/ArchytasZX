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
the opam root of setup_rivals.sh), timed, with their lines and bytes. The check runs once
(CHECK_ONCE, K = 1): the symbolic run when the proofs cover all of it (:func:`symbolic_covered`),
else the stated point (:func:`stated`). The verdict is equal when some file is accepted; when none
is, the outcome follows the files' failures (:func:`summary`)."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks import points
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
CHECK_ONCE = True
STATED_N: dict[str, int] = {"repetition_code": 2}
STATED_D = 2
PROOF_DIMENSIONS: dict[str, tuple[int, ...]] = {subject: (STATED_D,) for subject in PROOFS}
"""The dimensions d each subject's proofs state."""
DIMENSIONS: tuple[int, ...] = tuple(sorted({d for ds in PROOF_DIMENSIONS.values() for d in ds}))


def proof_dimensions(subject_id: str) -> tuple[int, ...]:
    """The dimensions the subject's proofs state, (STATED_D,) when not declared."""
    return PROOF_DIMENSIONS.get(subject_id, (STATED_D,))


def stated(inst: Any, cfg: dict[str, Any]) -> points.Point | None:
    """The proof's stated point: d the least of :func:`proof_dimensions`, n from STATED_N (the
    proof's fixed size) or the least domain n (a proof for every n)."""
    sid = inst.subject.id
    return points.stated(inst, cfg, STATED_N.get(sid), min(proof_dimensions(sid)))


def symbolic_covered(subject: Any) -> bool:
    """Whether the proofs cover every point of the subject's symbolic run: d fixed at one of
    :func:`proof_dimensions` and no fixed n in STATED_N."""
    return (
        not subject.general_d
        and int(subject.dimension) in proof_dimensions(subject.id)
        and subject.id not in STATED_N
    )


def summary(files: list[dict[str, Any]]) -> tuple[str, str, str | None]:
    """(status, verdict, limit) of the files' checks: ok and equal when some file is accepted;
    else timeout (limit stage_cpu) when every file timed out, oom when one ran out of memory,
    error otherwise."""
    if any(f.get("accepted", f.get("status") == "ok") for f in files):
        return "ok", "equal", None
    statuses = {f.get("status") for f in files}
    if statuses == {"timeout"}:
        return "timeout", "unknown", "stage_cpu"
    if "oom" in statuses:
        return "oom", "unknown", None
    return "error", "unknown", None


def rederive(rec: dict[str, Any]) -> dict[str, Any]:
    """A logged run record with its check status, verdict and outcome recomputed from its
    files by :func:`summary`; records without files unchanged."""
    check = (rec.get("stages") or {}).get("check") or {}
    if not check.get("files"):
        return rec
    status, verdict, lim = summary(check["files"])
    check = {**check, "status": status}
    if lim:
        check["limit"] = lim
    oc = common.verdict_outcome(bool(rec.get("truth")), status, verdict)
    return {
        **rec,
        "stages": {**rec["stages"], "check": check},
        "verdict": verdict,
        "outcome": oc,
        "wrong": oc == "wrong",
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
    point = job.get("point")
    if point is not None and tuple(point) != stated(inst, job["config"]):
        return common.unsupported("published proofs are checked once, at their stated size")
    limit = float(job["config"]["stages"]["timeout_seconds"])
    files, cpu, wall, detail = [], 0.0, 0.0, ""
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
                "accepted": accepted,
            }
        )
    prep = time.process_time() - c0 - cpu
    status, verdict, lim = summary(files)
    res = {
        "status": status,
        "verdict": verdict,
        "cpu": cpu,
        "wall": wall,
        "files": files,
        "proof_lines": sum(f["lines"] for f in files),
        "proof_bytes": sum(f["bytes"] for f in files),
    }
    if lim:
        res["limit"] = lim
    if detail:
        res["detail"] = detail
    return rival.finish(
        inst,
        res,
        max(prep, 0.0),
        {p: "pinned in setup_rivals.sh" for p, _ in PROOFS[inst.subject.id]},
    )
