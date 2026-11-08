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

"""Lean runner: ``lake env lean FILE`` in archytaszx/lean, in a systemd scope of its own capped
at a fraction of the total memory, at most :func:`slots` at a time across every worker.

The file is checked as a probe copy with ``set_option profiler true``, a ``#print`` marker
before every declaration and ``#print axioms`` of the final theorem appended. The runner
reports ``lean_status`` (accepted, rejected, timeout, oom), CPU and wall seconds, peak memory,
file bytes and lines, ``decl_ms`` (profiler milliseconds per declaration), ``axiom_class``
(clean: only propext, Classical.choice and Quot.sound; unproved: some ``unproved_*`` axiom;
opaque: no ``import ArchytasZX``; other: any other axiom; else the ``lean_status``) and
``statement_level`` (source, diagram, network, opaque).
"""

from __future__ import annotations

import fcntl
import functools
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
LEAN_PEAK_MB = 7000
MAX_SLOTS = 5
SLOT_DIR = Path(tempfile.gettempdir())
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_DECL = re.compile(r"^(?:private\s+)?(?:theorem|lemma)\s+([^\s(:{\[]+)", re.MULTILINE)
_ANY_DECL = re.compile(
    r"^(?:(?:/--(?:(?!-/)[\s\S])*-/|set_option[^\n]*\sin)[ \t]*\n)*"
    r"(?:@\[[^\]]*\]\s*)?(?:private\s+|noncomputable\s+|protected\s+)*"
    r"(theorem|lemma|def|abbrev|instance|example)\b\s*([^\s(:{\[]*)",
    re.MULTILINE,
)
_AXIOMS = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]")
_NO_AXIOMS = re.compile(r"'([^']+)' does not depend on any axioms")
_TOOK = re.compile(r"^(.*?) took ([0-9.]+)(ms|s)$")
MARK = "@@DECL "


@functools.cache
def cgroups() -> bool:
    """True when ``systemd-run --user --scope`` can hold a process to a memory cap."""
    if shutil.which("systemd-run") is None:
        return False
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64M", "true"]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def slurm_memory_mb() -> float | None:
    """The Slurm job's memory allocation in MiB (SLURM_MEM_PER_NODE, else SLURM_MEM_PER_CPU
    times SLURM_CPUS_PER_TASK), None outside Slurm."""
    node = os.environ.get("SLURM_MEM_PER_NODE")
    if node:
        return float(node)
    per_cpu, cpus = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_PER_TASK")
    return float(per_cpu) * int(cpus) if per_cpu and cpus else None


def total_memory_mb() -> float:
    """MemTotal in MiB, at most the Slurm allocation."""
    with open("/proc/meminfo", encoding="ascii") as fh:
        total = next(int(ln.split()[1]) for ln in fh if ln.startswith("MemTotal:")) / 1024
    alloc = slurm_memory_mb()
    return min(total, alloc) if alloc else total


def scoped(cmd: list[str], fraction: float) -> list[str]:
    """``cmd`` in a systemd scope of its own capped at ``fraction`` of the total memory, outside
    the calling worker's scope; ``cmd`` itself without systemd."""
    if not cgroups():
        return cmd
    cap = int(fraction * total_memory_mb())
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap}M", *cmd]


def _lift_address_space() -> None:
    """Raises the soft RLIMIT_AS to the hard limit (the worker's cap is soft only)."""
    import resource

    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


def lake() -> str:
    """The ``lake`` executable: on PATH, else elan's."""
    return shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")


@functools.cache
def slots() -> int:
    """Concurrent Lean checks: the checks of LEAN_PEAK_MB that fit in 85 % of the total memory,
    between 1 and MAX_SLOTS (BENCH_LEAN_SLOTS overrides)."""
    if os.environ.get("BENCH_LEAN_SLOTS"):
        return max(1, int(os.environ["BENCH_LEAN_SLOTS"]))
    return max(1, min(MAX_SLOTS, int(0.85 * total_memory_mb() // LEAN_PEAK_MB)))


def _slot() -> Any:
    """An exclusive lock on one of :func:`slots` slot files, waited for."""
    while True:
        for k in range(slots()):
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


def instrument(source: str, name: str | None) -> str:
    """``source`` with the profiler on after its imports, a marker before every declaration and
    ``#print axioms name`` appended."""
    lines = source.splitlines()
    head = max((i + 1 for i, ln in enumerate(lines) if ln.startswith("import ")), default=0)
    body = "\n".join(lines[head:])
    body = _ANY_DECL.sub(
        lambda m: f'#print "{MARK}{m.group(1)} {m.group(2) or "_"}"\n{m.group(0)}', body
    )
    tail = f'\n#print "{MARK}end _"\n' + (f"#print axioms {name}\n" if name else "")
    profiler = "set_option profiler true\nset_option profiler.threshold 0\n"
    return "\n".join(lines[:head]) + "\n" + profiler + body + tail


def declaration_times(output: str) -> list[dict[str, Any]]:
    """Per marked declaration: ``kind``, ``name`` and the profiler milliseconds that follow its
    marker, by category and in total."""
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith(MARK):
            kind, _, name = line[len(MARK) :].partition(" ")
            cur = None if kind == "end" else {"kind": kind, "name": name, "ms": 0.0, "by": {}}
            if cur is not None:
                out.append(cur)
            continue
        m = _TOOK.match(line)
        if m and cur is not None:
            ms = float(m.group(2)) * (1000.0 if m.group(3) == "s" else 1.0)
            cat = re.sub(r" of .*$", "", m.group(1))
            cur["ms"] += ms
            cur["by"][cat] = cur["by"].get(cat, 0.0) + ms
    return out


def check(
    path: Path, *, timeout: float | None = None, memory_fraction: float = 0.85
) -> dict[str, Any]:
    """Runs Lean on an instrumented copy of ``path`` and returns the runner's fields (see the
    module docstring)."""
    source = path.read_text(encoding="utf-8")
    name = final_theorem(source)
    out: dict[str, Any] = {
        "file_bytes": len(source.encode("utf-8")),
        "file_lines": source.count("\n") + 1,
        "final_theorem": name,
        "statement_level": statement_level(source),
        "unproved_declared": len(re.findall(r"^axiom unproved_", source, re.MULTILINE)),
    }
    probe = path.with_name(path.stem + "_probe.lean")
    probe.write_text(instrument(source, name), encoding="utf-8")
    slot = _slot()
    try:
        w0 = time.perf_counter()
        proc = subprocess.Popen(
            scoped([lake(), "env", "lean", str(probe)], memory_fraction),
            cwd=LEAN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=_lift_address_space,  # noqa: PLW1509  # noqa: PLW1509
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
    out |= {
        "lean_cpu": usage.ru_utime + usage.ru_stime,
        "lean_wall": wall,
        "lean_peak_mb": round(usage.ru_maxrss / 1024, 1),
        "decl_ms": declaration_times(text),
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
    out["unproved"] = (
        sum(1 for a in axioms if a.rsplit(".", 1)[-1].startswith("unproved_"))
        if axioms is not None
        else None
    )
    probe.unlink(missing_ok=True)
    return out
