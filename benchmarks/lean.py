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

"""Lean runner: ``lake env lean FILE`` in archytaszx/lean, at most :func:`slots` at a time across
every worker of a job, each check killed when its anonymous memory passes :func:`budget_mb` (and
under a memcap.Cap of OLEAN_MB plus that budget: a systemd scope, a cgroup, or a Slurm step
whose wrapper's stats replace the rusage CPU and peak memory).

The file is checked as a probe copy with ``set_option profiler true``, a ``#print`` marker
before every declaration and ``#print axioms`` of the final theorem appended. The runner
reports ``lean_status`` (accepted, rejected, timeout, oom), CPU and wall seconds, peak memory,
file bytes and lines, ``decl_ms`` (profiler milliseconds per declaration), ``axiom_class``
(clean: only propext, Classical.choice and Quot.sound; unproved: some ``unproved_*`` axiom;
opaque: no ``import ArchytasZX`` or opaque diagrams; other: any other axiom; else the
``lean_status``) and
``statement_level`` (source, diagram, network, opaque).

A file whose header imports ArchytasZX is first checked with that line replaced by
``import ArchytasZX.Light``; when that attempt is not accepted with the final theorem's axioms
printed, the file as written is checked and gives the result. ``lean_attempts`` lists each
attempt (import, lean_status, CPU, wall, peak memory, lean_kill, lean_detail) and
``lean_import`` names the one reported (light or full).
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks import memcap

ROOT = Path(__file__).resolve().parent.parent
LEAN_DIR = ROOT / "archytaszx" / "lean"
OLEAN_MB = 5120
"""The memory-mapped .olean files of one check (file pages, shared by concurrent checks)."""
LEAN_WORK_MB = 8192
"""The smallest anonymous-memory budget of one check."""
WORKER_RESERVE_MB = 1024
"""The memory kept per worker outside the Lean slots."""
MAX_SLOTS = 5
WATCH_SECONDS = 0.25
SHM = Path("/dev/shm")
SLOT_DIR = SHM if SHM.is_dir() and os.access(SHM, os.W_OK) else Path(tempfile.gettempdir())
_OOM_TEXT = re.compile(r"out of memory|std::bad_alloc|Cannot allocate memory")
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
FULL_IMPORT = "import ArchytasZX"
LIGHT_IMPORT = "import ArchytasZX.Light"


def cgroups() -> bool:
    """True when ``systemd-run --user --scope`` can hold a process to a memory cap."""
    return memcap.systemd_ok()


def slurm_memory_mb() -> float | None:
    """The Slurm job's memory allocation in MiB (SLURM_MEM_PER_NODE, else SLURM_MEM_PER_CPU
    times SLURM_CPUS_PER_TASK), None outside Slurm."""
    node = os.environ.get("SLURM_MEM_PER_NODE")
    if node:
        return float(node)
    per_cpu, cpus = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_PER_TASK")
    return float(per_cpu) * int(cpus) if per_cpu and cpus else None


def cgroup_memory_mb(
    root: Path = Path("/sys/fs/cgroup"), proc: Path = Path("/proc/self/cgroup")
) -> float | None:
    """The smallest memory limit in MiB of this process's memory cgroup and its ancestors
    (cgroup v2 memory.max, v1 memory.limit_in_bytes), None when there is none."""
    try:
        lines = proc.read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    best: float | None = None
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrl, rel = parts
        if ctrl == "":
            base, name = root, "memory.max"
        elif "memory" in ctrl.split(","):
            base, name = root / "memory", "memory.limit_in_bytes"
        else:
            continue
        path = Path(rel.lstrip("/"))
        for anc in (path, *path.parents):
            try:
                text = (base / anc / name).read_text(encoding="ascii").strip()
            except OSError:
                continue
            if text.isdigit() and int(text) < 2**60:
                mb = int(text) / 2**20
                best = mb if best is None else min(best, mb)
    return best


def total_memory_mb() -> float:
    """MemTotal in MiB, at most the Slurm allocation and the cgroup memory limit."""
    with open("/proc/meminfo", encoding="ascii") as fh:
        total = next(int(ln.split()[1]) for ln in fh if ln.startswith("MemTotal:")) / 1024
    for cap in (slurm_memory_mb(), cgroup_memory_mb()):
        if cap:
            total = min(total, cap)
    return total


def cap_mb(fraction: float, budget: float | None = None) -> int:
    """The cgroup cap of one check: ``fraction`` of the total memory, at most OLEAN_MB plus
    ``budget``."""
    cap = int(fraction * total_memory_mb())
    return min(cap, int(OLEAN_MB + budget)) if budget is not None else cap


def scoped(cmd: list[str], fraction: float, budget: float | None = None) -> list[str]:
    """``cmd`` in a systemd scope of its own capped at ``fraction`` of the total memory and at
    OLEAN_MB plus ``budget``, outside the calling worker's scope; ``cmd`` itself without
    systemd."""
    if not cgroups():
        return cmd
    cap = cap_mb(fraction, budget)
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        f"MemoryMax={cap}M",
        *cmd,
    ]


_ACTIVE: set[int] = set()


def _child_limits(cpu: float | None) -> Callable[[], None]:
    """A preexec function: the soft RLIMIT_AS raised to the hard limit (the worker's cap is soft
    only) and RLIMIT_CPU at ``cpu`` seconds per process."""

    def apply() -> None:
        import resource

        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
        if cpu:
            c = max(1, int(cpu))
            resource.setrlimit(resource.RLIMIT_CPU, (c, c + 5))

    return apply


def _children(pid: int) -> list[int]:
    """The child pids of every thread of ``pid`` (/proc/PID/task/TID/children)."""
    out: list[int] = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/children", encoding="ascii") as fh:
                out.extend(int(c) for c in fh.read().split())
        except OSError:
            continue
    return out


def _cpu_ticks(pid: int) -> int:
    """utime + stime + cutime + cstime of ``pid`` in clock ticks (/proc/PID/stat), 0 when gone."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return 0
    return sum(int(f) for f in fields[11:15])


def tree_usage(pid: int) -> tuple[float, float]:
    """(RssAnon in MiB, CPU seconds with reaped children) summed over ``pid`` and its
    descendants."""
    anon, ticks, stack, seen = 0, 0, [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/status", encoding="ascii", errors="replace") as fh:
                anon += next((int(ln.split()[1]) for ln in fh if ln.startswith("RssAnon:")), 0)
        except OSError:
            continue
        ticks += _cpu_ticks(p)
        stack.extend(_children(p))
    return anon / 1024, ticks / os.sysconf("SC_CLK_TCK")


def tree_anon_mb(pid: int) -> float:
    """RssAnon in MiB summed over ``pid`` and its descendants."""
    return tree_usage(pid)[0]


def watch(
    pid: int,
    limit_mb: float,
    done: threading.Event,
    kill: Callable[[], None],
    stats: dict[str, float] | None = None,
) -> threading.Event:
    """A daemon thread that calls ``kill`` once the tree RssAnon of ``pid`` passes ``limit_mb``,
    polled every WATCH_SECONDS until ``done``; the returned event is set on that kill.
    ``stats`` gets ``peak_anon_mb`` and ``cpu_tree``, the largest tree RssAnon and tree CPU
    seconds seen over the polls."""
    over = threading.Event()
    seen = stats if stats is not None else {}
    seen.setdefault("peak_anon_mb", 0.0)
    seen.setdefault("cpu_tree", 0.0)

    def loop() -> None:
        while not done.wait(WATCH_SECONDS):
            anon, cpu = tree_usage(pid)
            seen["peak_anon_mb"] = max(seen["peak_anon_mb"], anon)
            seen["cpu_tree"] = max(seen["cpu_tree"], cpu)
            if anon > limit_mb:
                over.set()
                kill()
                return

    threading.Thread(target=loop, daemon=True).start()
    return over


def kill_active() -> None:
    """SIGKILL to the process group of every running Lean check."""
    for pg in list(_ACTIVE):
        try:
            os.killpg(pg, signal.SIGKILL)
        except OSError:
            pass


def lake() -> str:
    """The ``lake`` executable: on PATH, else elan's."""
    return shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")


def plan(total_mb: float, jobs: int, fixed: int | None = None) -> tuple[int, int]:
    """(slots, per-check budget in MiB): the pool is ``total_mb`` less OLEAN_MB and
    WORKER_RESERVE_MB per worker; slots = floor(pool / LEAN_WORK_MB) between 1 and MAX_SLOTS
    (``fixed`` when given), budget = max(LEAN_WORK_MB, floor(pool / slots))."""
    pool = total_mb - OLEAN_MB - max(1, jobs) * WORKER_RESERVE_MB
    n = max(1, fixed) if fixed else max(1, min(MAX_SLOTS, int(pool // LEAN_WORK_MB)))
    return n, max(LEAN_WORK_MB, int(pool // n))


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def slots(jobs: int | None = None) -> int:
    """Concurrent Lean checks per job: :func:`plan` of the total memory and ``jobs`` workers
    (BENCH_JOBS, else 1); BENCH_LEAN_SLOTS overrides."""
    fixed = _env_int("BENCH_LEAN_SLOTS")
    if fixed:
        return max(1, fixed)
    return plan(total_memory_mb(), jobs or _env_int("BENCH_JOBS") or 1)[0]


def budget_mb(jobs: int | None = None) -> int:
    """The anonymous-memory budget in MiB of one Lean check: :func:`plan`'s for :func:`slots`
    slots; BENCH_LEAN_BUDGET_MB overrides."""
    fixed = _env_int("BENCH_LEAN_BUDGET_MB")
    if fixed:
        return fixed
    n_jobs = jobs or _env_int("BENCH_JOBS") or 1
    return plan(total_memory_mb(), n_jobs, slots(n_jobs))[1]


def _slot_scope() -> str:
    """The slot pool's name: the Slurm job and array task (each has its own allocation), else
    the machine's single pool."""
    job = os.environ.get("SLURM_JOB_ID")
    return f"{job}-{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}" if job else "local"


def slot_path(k: int) -> Path:
    """The lock file of slot ``k`` of this job's pool."""
    return SLOT_DIR / f"archytaszx-bench-lean-{_slot_scope()}-{k}.lock"


def _slot() -> Any:
    """An exclusive lock on one of :func:`slots` slot files, waited for."""
    n = slots()
    while True:
        for k in range(n):
            fh = open(slot_path(k), "w")  # noqa: SIM115
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


def _opaque(source: str) -> bool:
    """Whether the final theorem is over opaque diagrams."""
    return "import ArchytasZX" not in source or "namespace ArchytasZX.Opaque" in source


def statement_level(source: str) -> str:
    """source, diagram, network or opaque (see the module docstring)."""
    if _opaque(source):
        return "opaque"
    text = statement(source)
    if '"' in text or re.search(r"\b(elab\w*|parse\w*|Src\w*|src\w*|source\w*)\b", text):
        return "source"
    if re.search(r"\bDEqual\w*\b", text):
        return "diagram"
    return "network"


def classify(axioms: list[str] | None, source: str) -> str:
    """The axiom class of the final theorem's axioms (see the module docstring); ``unknown``
    when no axiom list was printed."""
    if _opaque(source):
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


def _decode(data: bytes) -> str:
    """``data`` as UTF-8, each invalid byte replaced."""
    return data.decode("utf-8", errors="replace")


def _drain(stream: Any, chunks: list[bytes]) -> None:
    """Appends every chunk read from ``stream`` to ``chunks`` until its end."""
    if stream is None:
        return
    chunks.extend(iter(lambda: stream.read(65536), b""))


def light_source(source: str) -> str | None:
    """``source`` with its header line ``import ArchytasZX`` replaced by LIGHT_IMPORT, None when
    its header has no such line."""
    lines = source.splitlines(keepends=True)
    head = max((i + 1 for i, ln in enumerate(lines) if ln.startswith("import ")), default=0)
    for i in range(head):
        if lines[i].strip() == FULL_IMPORT:
            end = lines[i][len(lines[i].rstrip("\r\n")) :]
            return "".join([*lines[:i], LIGHT_IMPORT + end, *lines[i + 1 :]])
    return None


def _attempt(
    path: Path, source: str, name: str | None, timeout: float | None, memory_fraction: float
) -> dict[str, Any]:
    """One Lean run on an instrumented copy of ``source`` next to ``path``: lean_status, CPU,
    wall, peak memory, decl_ms, lean_detail and lean_kill when set, and the axioms of ``name``."""
    probe = path.with_name(path.stem + "_probe.lean")
    probe.write_text(instrument(source, name), encoding="utf-8")
    budget = budget_mb()
    cap = memcap.Cap(cap_mb(memory_fraction, budget), tag="lean", no_swap=False)
    out: dict[str, Any] = {"lean_budget_mb": budget, "lean_memcap": cap.mech}
    slot = _slot()
    proc: subprocess.Popen[bytes] | None = None
    done = threading.Event()
    try:
        w0 = time.perf_counter()
        proc = subprocess.Popen(
            cap.command([lake(), "env", "lean", str(probe)]),
            cwd=LEAN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=memcap.chain(cap.preexec(), _child_limits(timeout)),  # noqa: PLW1509
        )
        err_chunks: list[bytes] = []
        err_reader = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
        err_reader.start()
        pgid = proc.pid
        _ACTIVE.add(pgid)
        killed = threading.Event()

        def killpg() -> None:
            cap.kill_group(pgid)

        def kill() -> None:
            killed.set()
            killpg()

        stats: dict[str, float] = {}
        over = watch(pgid, budget, done, killpg, stats)
        timer = threading.Timer(timeout, kill) if timeout else None
        if timer is not None:
            timer.start()
        messages = _decode(proc.stdout.read() if proc.stdout is not None else b"")
        _, status, usage = os.wait4(proc.pid, 0)
        err_reader.join()
        errors_text = _decode(b"".join(err_chunks))
        text = messages + ("\n" + errors_text if errors_text else "")
        if timer is not None:
            timer.cancel()
        wall = time.perf_counter() - w0
        code = os.waitstatus_to_exitcode(status)
        cgroup_oom = cap.oom(code, text)
        step = cap.usage()
    finally:
        cap.close()
        cap.discard()
        done.set()
        if proc is not None:
            _ACTIVE.discard(proc.pid)
        slot.close()
        probe.unlink(missing_ok=True)
    rusage_cpu = usage.ru_utime + usage.ru_stime
    maxrss_kb: float = usage.ru_maxrss
    if step is not None:
        rusage_cpu, maxrss_kb = step["cpu"], step["maxrss_kb"]
        for k in ("peak_anon_mb", "cpu_tree"):
            stats[k] = max(stats.get(k, 0.0), step.get(k, 0.0))
    out |= {
        "lean_cpu": max(rusage_cpu, stats.get("cpu_tree", 0.0)),
        "lean_cpu_rusage": rusage_cpu,
        "lean_cpu_tree": round(stats.get("cpu_tree", 0.0), 2),
        "lean_wall": wall,
        "lean_peak_mb": round(maxrss_kb / 1024, 1),
        "lean_peak_anon_mb": round(stats.get("peak_anon_mb", 0.0), 1),
        "decl_ms": declaration_times(messages),
    }
    if over.is_set():
        out["lean_status"] = "oom"
        out["lean_kill"] = "budget"
    elif cgroup_oom:
        out["lean_status"] = "oom"
        out["lean_kill"] = f"memcap.{cap.mech}"
    elif killed.is_set() or code in (-signal.SIGXCPU, 128 + signal.SIGXCPU):
        out["lean_status"] = "timeout"
    elif code in (-signal.SIGKILL, 128 + signal.SIGKILL):
        out["lean_status"] = "oom"
    elif code == 0 and not re.search(r"(^|\s)error:", text):
        out["lean_status"] = "accepted"
    elif _OOM_TEXT.search(text):
        out["lean_status"] = "oom"
        out["lean_detail"] = text.strip()[-300:]
    else:
        out["lean_status"] = "rejected"
        errors = [ln for ln in text.splitlines() if "error" in ln]
        out["lean_detail"] = (errors[0] if errors else text.strip()[-300:])[:300]
    out["axioms"] = parse_axioms(text, name) if name and out["lean_status"] == "accepted" else None
    return out


ATTEMPT_FIELDS = (
    "lean_status",
    "lean_cpu",
    "lean_wall",
    "lean_peak_mb",
    "lean_peak_anon_mb",
    "lean_kill",
    "lean_detail",
)


def check(
    path: Path, *, timeout: float | None = None, memory_fraction: float = 0.85
) -> dict[str, Any]:
    """Runs Lean on ``path`` with the light import first, when its header imports ArchytasZX,
    and on the file as written whenever that attempt is not accepted with its axioms printed;
    returns the runner's fields of the final attempt (see the module docstring)."""
    source = path.read_text(encoding="utf-8")
    name = final_theorem(source)
    meta: dict[str, Any] = {
        "file_bytes": len(source.encode("utf-8")),
        "file_lines": source.count("\n") + 1,
        "final_theorem": name,
        "statement_level": statement_level(source),
        "unproved_declared": len(re.findall(r"^axiom unproved_", source, re.MULTILINE)),
    }
    attempts: list[dict[str, Any]] = []
    light = light_source(source)
    res: dict[str, Any] | None = None
    if light is not None:
        res = _attempt(path, light, name, timeout, memory_fraction)
        attempts.append({"import": "light", **{k: res[k] for k in ATTEMPT_FIELDS if k in res}})
        if res["lean_status"] != "accepted" or (name and res["axioms"] is None):
            res = None
    used = "light"
    if res is None:
        used = "full"
        res = _attempt(path, source, name, timeout, memory_fraction)
        attempts.append({"import": "full", **{k: res[k] for k in ATTEMPT_FIELDS if k in res}})
    out = meta | res | {"lean_import": used, "lean_attempts": attempts}
    axioms = out["axioms"]
    out["axiom_class"] = (
        classify(axioms, source) if out["lean_status"] == "accepted" else out["lean_status"]
    )
    out["unproved"] = (
        sum(1 for a in axioms if a.rsplit(".", 1)[-1].startswith("unproved_"))
        if axioms is not None
        else None
    )
    return out
