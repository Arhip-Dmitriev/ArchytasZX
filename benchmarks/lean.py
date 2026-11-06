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

"""Lean runner: ``lake env lean FILE`` in archytaszx/lean, in a memory scope of its own (the
worker's cap would evict the mapped Mathlib files), under the worker's CPU cap, at most
:data:`SLOTS` at a time across every worker; and the ``lean`` suite: every exportable workload
exported by ``certificate lean`` and checked by the runner.

The runner reports ``lean_status`` (accepted, rejected, timeout, oom), CPU and wall seconds,
peak memory, file bytes and lines, the ``#print axioms`` result of the final theorem as
``axiom_class`` (clean: only propext, Classical.choice and Quot.sound; unproved: some
``unproved_*`` axiom; opaque: the fallback export of LEAN.txt section 4.5; other: any other
axiom; for a file Lean does not accept, its ``lean_status``) and ``statement_level``
(source, diagram, network, opaque).
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LEAN_DIR = ROOT / "archytaszx" / "lean"
SLOTS = 5
LEAN_MEMORY_FRACTION = 0.85
SLOT_DIR = Path(tempfile.gettempdir())
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_DECL = re.compile(r"^(?:private\s+)?(?:theorem|lemma)\s+([^\s(:{\[]+)", re.MULTILINE)
_AXIOMS = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]")
_NO_AXIOMS = re.compile(r"'([^']+)' does not depend on any axioms")


def _scoped(cmd: list[str]) -> list[str]:
    """``cmd`` in a systemd scope of its own capped at :data:`LEAN_MEMORY_FRACTION` of the total
    memory, outside the calling worker's scope; ``cmd`` itself without systemd."""
    from benchmarks import harness

    if not harness._cgroups():
        return cmd
    with open("/proc/meminfo", encoding="ascii") as fh:
        total_kb = next(int(ln.split()[1]) for ln in fh if ln.startswith("MemTotal:"))
    cap = int(LEAN_MEMORY_FRACTION * total_kb / 1024)
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap}M", *cmd]


def lake() -> str:
    """The ``lake`` executable: on PATH, else elan's."""
    return shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")


def _slot() -> Any:
    """An exclusive lock on one of :data:`SLOTS` slot files, waited for."""
    while True:
        for k in range(SLOTS):
            fh = open(SLOT_DIR / f"archytaszx-bench-lean-{k}.lock", "w")  # noqa: SIM115
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            return fh
        time.sleep(0.5)


def final_theorem(source: str) -> str | None:
    """The full name of the last theorem declared in ``source``."""
    last: tuple[int, str] | None = None
    for m in _DECL.finditer(source):
        last = (m.start(), m.group(1))
    if last is None:
        return None
    stack: list[str] = []
    for line in source[: last[0]].splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == "namespace":
            stack.append(words[1])
        elif len(words) >= 2 and words[0] == "end" and stack and words[1] == stack[-1]:
            stack.pop()
    return ".".join([*stack, last[1]])


def statement(source: str) -> str:
    """The final theorem's statement: from its name to ``:=``."""
    starts = [m.start() for m in _DECL.finditer(source)]
    if not starts:
        return ""
    return source[starts[-1] :].split(":=", 1)[0]


def statement_level(source: str) -> str:
    """source, diagram, network or opaque (see the module docstring)."""
    if "import ArchytasZX" not in source:
        return "opaque"
    text = statement(source)
    if '"' in text or re.search(r"\b(elab\w*|parse\w*|Src\w*|src\w*|source\w*)\b", text):
        return "source"
    if re.search(r"\bDEqual\w*\b", text):
        return "diagram"
    return "network"


def classify(axioms: list[str] | None, source: str) -> str:
    """The axiom class of the final theorem's axioms (see the module docstring)."""
    if "import ArchytasZX" not in source:
        return "opaque"
    if axioms is None:
        return "unknown"
    rest = [a for a in axioms if a not in STANDARD_AXIOMS]
    if not rest:
        return "clean"
    if all(a.rsplit(".", 1)[-1].startswith("unproved_") for a in rest):
        return "unproved"
    return "other"


def parse_axioms(output: str, name: str) -> list[str] | None:
    """The axioms ``#print axioms name`` printed, or None when absent."""
    flat = " ".join(output.split())
    for m in _AXIOMS.finditer(flat):
        if m.group(1) == name:
            return [a.strip() for a in m.group(2).split(",") if a.strip()]
    for m in _NO_AXIOMS.finditer(flat):
        if m.group(1) == name:
            return []
    return None


def check(path: Path, *, timeout: float | None = None) -> dict[str, Any]:
    """Runs Lean on ``path`` (a copy with ``#print axioms`` of the final theorem appended)
    and returns the runner's fields (see the module docstring)."""
    source = path.read_text(encoding="utf-8")
    name = final_theorem(source)
    out: dict[str, Any] = {
        "file_bytes": len(source.encode("utf-8")),
        "file_lines": source.count("\n") + 1,
        "final_theorem": name,
        "statement_level": statement_level(source),
        "unproved_declared": len(re.findall(r"^axiom unproved_", source, re.MULTILINE)),
    }
    probe = path.with_name(path.stem + "_axioms.lean")
    probe.write_text(source + (f"\n#print axioms {name}\n" if name else ""), encoding="utf-8")
    slot = _slot()
    try:
        w0 = time.perf_counter()
        proc = subprocess.Popen(
            _scoped([lake(), "env", "lean", str(probe)]),
            cwd=LEAN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        killed = threading.Event()

        def kill() -> None:
            killed.set()
            proc.kill()

        timer = threading.Timer(timeout, kill) if timeout else None
        if timer is not None:
            timer.start()
        text = proc.stdout.read() if proc.stdout is not None else ""
        _, status, usage = os.wait4(proc.pid, 0)
        if timer is not None:
            timer.cancel()
        wall = time.perf_counter() - w0
    finally:
        slot.close()
    code = os.waitstatus_to_exitcode(status)
    cpu = usage.ru_utime + usage.ru_stime
    out |= {
        "lean_cpu": cpu,
        "lean_wall": wall,
        "lean_peak_mb": round(usage.ru_maxrss / 1024, 1),
    }
    if killed.is_set() or code in (-signal.SIGXCPU, 128 + signal.SIGXCPU):
        out["lean_status"] = "timeout"
    elif code in (-signal.SIGKILL, 128 + signal.SIGKILL):
        out["lean_status"] = "oom"
    elif code == 0 and not re.search(r"(^|\s)error:", text):
        out["lean_status"] = "accepted"
    else:
        out["lean_status"] = "rejected"
        errors = [ln for ln in text.splitlines() if "error" in ln]
        out["lean_detail"] = (errors[0] if errors else text.strip()[-300:])[:300]
    axioms = parse_axioms(text, name) if name and out["lean_status"] == "accepted" else None
    out["axioms"] = axioms
    out["axiom_class"] = (
        classify(axioms, source) if out["lean_status"] == "accepted" else out["lean_status"]
    )
    out["unproved_axioms"] = (
        sorted(a for a in axioms if a.rsplit(".", 1)[-1].startswith("unproved_"))
        if axioms is not None
        else None
    )
    out["unproved"] = len(out["unproved_axioms"]) if axioms is not None else None
    probe.unlink(missing_ok=True)
    return out


SIZE = "size"
REPEAT = 3
RECORD_ENV = True
QUICK_WORKLOADS = ("examples/copy_induction.azx #1", "spider fusion")


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every proof workload of :func:`benchmarks.workloads.unit_scripts`; ``quick`` keeps
    two."""
    from benchmarks import workloads

    return [
        {"suite": "lean", "group": group, "workload": name, "size": 1}
        for group, name, _, _ in workloads.unit_scripts()
        if not quick or name in QUICK_WORKLOADS
    ]


def export_and_check(script: str, base: str, path: Path) -> dict[str, Any]:
    """Runs ``script`` with ``certificate kernel`` and ``certificate lean PATH`` appended, then
    the Lean runner on the file written; ``cpu`` and ``wall`` are Lean's (0 without a file);
    an unknown verdict is ``inconclusive`` and not exported."""
    from benchmarks import engine, workloads

    run_ = engine.run_script(workloads.with_lean(script, path), base=base)
    v = engine.verdict(run_)
    cpu, wall = engine.until_verdict(run_)
    out: dict[str, Any] = {
        "cpu": 0.0,
        "wall": 0.0,
        "script_cpu": cpu,
        "script_wall": wall,
        **{k: v[k] for k in ("verdict", "method", "nodes", "wires", "derivation_length")},
        **engine.kernel_fields(run_),
        **engine.lean_export_fields(run_),
    }
    if v["verdict"] == "unknown":
        return out | {"status": "inconclusive", "detail": str(v["reason"])[:400]}
    if not path.exists():
        out["lean_status"] = "no export"
        return out
    out |= check(path)
    return out | {"cpu": out["lean_cpu"], "wall": out["lean_wall"]}


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Export one workload and check it in Lean."""
    from benchmarks import workloads

    match = [w for w in workloads.unit_scripts() if w[1] == case["workload"]]
    if not match:
        raise ValueError(f"no workload {case['workload']!r}")
    _, _, script, base = match[0]
    with tempfile.TemporaryDirectory(prefix="bench-lean-") as tmp:
        return export_and_check(script, base, Path(tmp) / "export.lean")
