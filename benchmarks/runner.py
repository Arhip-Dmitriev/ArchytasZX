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

"""The runner (BENCHMARKS.txt, section 10): every (equation, tool) unit through the symbolic
run, the phase 1 ladder and phase 2 fill, each run a fresh worker process under a memory cap,
each finished run appended to this worker's log before the next starts.

Units run ``jobs`` at a time. A unit's runs go one after another, since each phase 1 step and
each phase 2 draw depends on the outcomes before it. On start every log is read and every key
already present is reused, not re-run; a unit whose ``unit`` record exists is skipped whole.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks import breaks, config, lean, log, outcomes, points, sources, stats, subject

ROOT = Path(__file__).resolve().parent.parent
STAGE_MARK = "@@STAGE "
RESULT_MARK = "@@RESULT "
OOMKILL_MARK = "@@OOMKILL "
SPANS_MARK = "@@SPANS "
JOURNAL_WAIT = 3.0
CALIBRATION = """@rhs =
    dim d
    count n
    g = Z(0 -> 2, d)
    repeat n: g.out1
    scalar d^(-1/2)

load
    dim d
    count n
    g0 = X(0 -> 1, d)
    g1 = F(d)
    g2 = X(0 -> 1, d)
    g3 = Z(1 -> 2, d)
    g4 = X(2 -> 1, d)
    wire g0.out0 -- g1.in0, g1.out0 -- g3.in0, g2.out0 -- g4.in0, g3.out1 -- g4.in1
    repeat n: g2, g4
    outputs g3.out0, g4.out0
    scalar 1/sqrt(d)

decide against @rhs
"""
_stop = threading.Event()
_children: set[subprocess.Popen[str]] = set()
_children_lock = threading.Lock()


@dataclass
class Unit:
    """One (equation, tool): the instance id, the F record of an F equation, the tool."""

    equation: str
    tool: str
    false: dict[str, Any] | None = None
    restricted: bool = False

    @property
    def id(self) -> str:
        return f"{self.equation}{'!' if self.false is not None else ''}|{self.tool}"


@dataclass
class Context:
    """What every unit shares: config, its hash, results directory, the log index, writers,
    the per-run memory cap and the machine record."""

    cfg: dict[str, Any]
    cfg_hash: str
    results: Path
    done: dict[str, dict[str, Any]]
    machine: dict[str, Any]
    calibration: dict[str, Any] | None
    mem_mb: int
    cgroup: bool
    git: dict[str, Any] = field(default_factory=dict)
    writers: dict[int, log.Writer] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stale: set[str] = field(default_factory=set)

    def writer(self) -> log.Writer:
        tid = threading.get_ident()
        with self.lock:
            if tid not in self.writers:
                self.writers[tid] = log.Writer(self.results, f"w{len(self.writers)}")
            return self.writers[tid]


def machine() -> dict[str, Any]:
    """Node name, CPU model, frequency governor, memory, OS and Python."""
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as fh:
            cpu = next(
                (ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), ""
            )
    except OSError:
        pass
    gov = ""
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    return {
        "node": socket.gethostname(),
        "cpu_model": cpu,
        "governor": gov,
        "mem_total_mb": round(lean.total_memory_mb()),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
    }


def environment(cfg: dict[str, Any]) -> dict[str, Any]:
    """The job's environment record: config, git commit and dirty flag, machine, software and
    Lean versions."""

    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return r.stdout.strip()

    versions = {}
    for mod in ("archytaszx", "sympy", "numpy", "matplotlib"):
        try:
            from importlib.metadata import version

            versions[mod] = version(mod)
        except Exception:  # noqa: BLE001
            versions[mod] = None
    lean_dir = lean.LEAN_DIR
    manifest = {}
    try:
        manifest = json.loads((lean_dir / "lake-manifest.json").read_text())
    except OSError:
        pass
    mathlib = next(
        (p.get("rev") for p in manifest.get("packages", []) if p.get("name") == "mathlib"), None
    )
    toolchain = (
        (lean_dir / "lean-toolchain").read_text().strip()
        if (lean_dir / "lean-toolchain").exists()
        else None
    )
    return {
        "config": cfg,
        "config_hash": config.digest(cfg),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "machine": machine(),
        "software": versions,
        "lean_toolchain": toolchain,
        "mathlib": mathlib,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
    }


def memory_cap_mb(cfg: dict[str, Any], jobs: int) -> int:
    """memory_fraction of the available memory (at most the Slurm allocation), shared by the
    workers."""
    with open("/proc/meminfo", encoding="ascii") as fh:
        info = {ln.split(":")[0]: int(ln.split()[1]) for ln in fh}
    avail = info["MemAvailable"] / 1024
    alloc = lean.slurm_memory_mb()
    avail = min(avail, alloc) if alloc else avail
    return int(cfg["stages"]["memory_fraction"] * avail / max(1, jobs))


def terminate(proc: subprocess.Popen[str], grace: float = 5.0) -> None:
    """SIGTERM to the worker's process group (the worker kills its Lean checks), then SIGKILL
    after ``grace`` seconds."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            return
        try:
            proc.wait(grace)
            return
        except subprocess.TimeoutExpired:
            continue


def killed_status(timer: bool, code: int, oom_rise: int) -> tuple[str, str]:
    """(outcome, kill source) of a worker that ended without a result: timeout by the stage
    timer, oom by SIGKILL or a rise of the scope's oom_kill counter, else error."""
    if timer:
        return "timeout", "stage timer"
    if code in (-signal.SIGKILL, 137):
        return "oom", "SIGKILL"
    if oom_rise > 0:
        return "oom", "cgroup oom_kill"
    return "error", f"exit {code}"


def _journal_oom(pid: int, started: float) -> bool:
    """Whether the journal shows an OOM-kill failure of the worker's scope (run-p<pid>-*)
    since its start (``started``, a perf_counter time), polled for up to JOURNAL_WAIT
    seconds."""
    from benchmarks import reclassify

    t0 = time.time() - (time.perf_counter() - started)
    deadline = time.perf_counter() + JOURNAL_WAIT
    while True:
        if reclassify.journal_oom_times(since=t0 - 1, scope=f"run-p{pid}-"):
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(0.5)


def launch(job: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """One worker process on ``job``: its result, or the stage it was killed in."""
    st = ctx.cfg["stages"]
    n_stages = sum(1 for s in ("decide", "certificate", "lean_export", "lean_check") if st[s]) + 1
    wall_limit = float(st["timeout_seconds"]) * n_stages * float(st.get("wall_factor", 4))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, prefix="bench-job-") as fh:
        json.dump(job, fh, default=str)
        job_path = fh.name
    cmd = [sys.executable, "-m", "benchmarks.worker", job_path]
    if ctx.cgroup:
        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "-p",
            f"MemoryMax={ctx.mem_mb}M",
            "-p",
            "MemorySwapMax=0",
            *cmd,
        ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "BENCH_MEMORY_MB": "" if ctx.cgroup else str(ctx.mem_mb),
        "BENCH_SPAN_FILE": job_path + ".spans",
    }

    w0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with _children_lock:
        _children.add(proc)
    killed = threading.Event()

    def kill() -> None:
        killed.set()
        terminate(proc)

    timer = threading.Timer(wall_limit, kill)
    timer.start()
    stage = "start"
    result: dict[str, Any] | None = None
    oom_rise = 0
    spans: list[Any] | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(STAGE_MARK):
            stage = line[len(STAGE_MARK) :].strip()
        elif line.startswith(RESULT_MARK):
            result = json.loads(line[len(RESULT_MARK) :])
        elif line.startswith(OOMKILL_MARK):
            oom_rise = int(line[len(OOMKILL_MARK) :])
        elif line.startswith(SPANS_MARK):
            spans = json.loads(line[len(SPANS_MARK) :])
    err = proc.stderr.read() if proc.stderr is not None else ""
    code = proc.wait()
    timer.cancel()
    with _children_lock:
        _children.discard(proc)
    os.unlink(job_path)
    source = "sigterm" if spans is not None else None
    side = Path(job_path + ".spans")
    if spans is None and result is None and side.exists():
        try:
            spans = json.loads(side.read_text(encoding="utf-8"))["spans"]
            source = "snapshot"
        except (OSError, ValueError, KeyError):
            spans = None
    for f in (side, side.with_name(side.name + ".tmp")):
        f.unlink(missing_ok=True)
    wall = time.perf_counter() - w0
    if _stop.is_set() and result is None:
        raise InterruptedError
    if result is not None:
        result["process_wall"] = wall
        return result
    status, kill = killed_status(killed.is_set(), code, oom_rise)
    term = code in (143, -signal.SIGTERM)
    if status == "error" and term and ctx.cgroup and _journal_oom(proc.pid, w0):
        status, kill = "oom", "cgroup oom_kill"
    out: dict[str, Any] = {
        "outcome": status,
        "killed_in": stage,
        "exit": code,
        "kill": kill,
        "detail": err.strip()[-400:],
        "process_wall": wall,
    }
    if spans is not None:
        from archytaszx.trace import open_path

        out["spans"] = spans
        out["spans_source"] = source
        out["open_span"] = "/".join(open_path(spans)) or None
    return out


def _success(rec: dict[str, Any]) -> bool:
    return rec.get("outcome") == "ok"


def _published_once(rec: dict[str, Any]) -> bool:
    """Whether a logged run is a check-once tool's refusal of a point (re-run on resume)."""
    return (
        rec.get("outcome") == "unsupported"
        and (rec.get("unsupported_code") or outcomes.unsupported_code(str(rec.get("detail") or "")))
        == "published_once"
    )


def _main_cpu(rec: dict[str, Any]) -> float | None:
    st = rec.get("stages", {})
    main = st.get("decide") or st.get("check") or next(iter(st.values()), None)
    return None if main is None else main.get("cpu")


def _sampled(key: str, share: float) -> bool:
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return h / 2**64 < share


class UnitRun:
    """The schedule of one unit (BENCHMARKS.txt, sections 5 and 9)."""

    def __init__(self, unit: Unit, inst: Any, ctx: Context) -> None:
        self.unit, self.inst, self.ctx = unit, inst, ctx
        self.cfg = ctx.cfg
        self.eq = inst.id
        self.source = sources.source_digest(inst, unit.false, self.cfg["subjects"]["catalogue"])

    def version(self, point: Any) -> str:
        if self.inst.restriction:
            return "restricted"
        return "symbolic" if point is None else "concrete"

    def run(self, point: Any, repeat: int, trace: bool, phase: str) -> dict[str, Any]:
        """One run, from the log when present."""
        version = self.version(point)
        k = log.key(
            self.ctx.cfg_hash,
            f"{self.eq}@{self.source}",
            version,
            self.unit.tool,
            point,
            repeat,
            trace,
        )
        old = self.ctx.done.get(k)
        if old is not None and not _published_once(old):
            return old if "cause" in old else self.classify(old)
        if _stop.is_set():
            raise InterruptedError
        job = {
            "equation": self.unit.equation,
            "false": self.unit.false,
            "tool": self.unit.tool,
            "point": list(point) if point is not None else None,
            "repeat": repeat,
            "trace": trace,
            "config": self.cfg,
        }
        t0 = time.time()
        res = launch(job, self.ctx)
        rec = {
            "key": k,
            "config": self.ctx.cfg_hash,
            "equation": self.eq,
            "source": self.source,
            "truth": self.inst.truth,
            "kind": self.unit.false["kind"] if self.unit.false else None,
            "subject": self.inst.subject.id,
            "category": self.inst.subject.category,
            "version": version,
            "tool": self.unit.tool,
            "n": point[0] if point is not None else None,
            "d": point[1] if point is not None else None,
            "repeat": repeat,
            "trace": trace,
            "phase": phase,
            "started": t0,
            "finished": time.time(),
            "machine": self.ctx.machine,
            "calibration": self.ctx.calibration,
            "git": self.ctx.git,
            **res,
        }
        rec = self.classify(rec)
        self.ctx.writer().append(rec)
        self.ctx.done[k] = rec
        return rec

    def classify(self, rec: dict[str, Any]) -> dict[str, Any]:
        """``rec`` with outcomes.classify's fields for this unit."""
        return outcomes.classify(rec, self.unit.false, outcomes.free_axes(self.inst, rec))

    def repeats(self, point: Any, first: dict[str, Any], phase: str) -> None:
        """The repeats of a successful point."""
        rc = config.repeats(self.cfg)
        trace = bool(self.cfg["trace"]["enabled"])
        cpus = [c for c in [_main_cpu(first)] if c is not None]
        if rc["mode"] == "fixed":
            for r in range(1, int(rc["runs"])):
                self.run(point, r, trace, phase)
        else:
            r = 1
            floor = max(int(rc["min_runs"]), 1)
            while r < int(rc["max_runs"]):
                if (
                    r >= floor
                    and r >= stats.min_runs(rc["confidence"])
                    and stats.precise(cpus, rc["confidence"], rc["precision"])
                ):
                    break
                rec = self.run(point, r, trace, phase)
                c = _main_cpu(rec)
                if _success(rec) and c is not None:
                    cpus.append(c)
                r += 1
        share = float(self.cfg["trace"]["overhead_sample"])
        if trace and share > 0 and _sampled(f"{self.eq}|{self.unit.tool}|{point}", share):
            self.run(point, 0, False, phase)

    def execute(self) -> None:
        """Symbolic run, phase 1, phase 2, then the unit record."""
        tool = self.unit.tool
        unit_key = f"{self.ctx.cfg_hash}|{self.eq}@{self.source}|unit|{tool}"
        if unit_key in self.ctx.done and unit_key not in self.ctx.stale:
            return
        from benchmarks import tools

        trace = bool(self.cfg["trace"]["enabled"])
        adapter = tools.adapter(tool)
        why = adapter.supports(self.inst, None)
        if why is None and getattr(adapter, "CHECK_ONCE", False):
            self.check_once(unit_key, adapter, trace)
            return
        if why is not None:
            self._finish(
                unit_key,
                {
                    "unsupported": why,
                    "unsupported_code": outcomes.unsupported_code(why),
                    "successes": 0,
                },
            )
            return
        pc = self.cfg["points"]
        if pc["symbolic"] and subject.has_symbolic(self.inst):
            rec = self.run(None, 0, trace, "symbolic")
            if _success(rec):
                self.repeats(None, rec, "symbolic")
        dims = getattr(tools.adapter(tool), "DIMENSIONS", None)
        ax = points.axes(self.inst, self.cfg, self.unit.false, dims)
        k_target = int(pc["K"])
        size = points.domain_size(ax, k_target)
        if dims and size == 0:
            why = "qubits only (d = 2)" if tuple(dims) == (2,) else f"dimensions {dims} only"
            self._finish(
                unit_key,
                {
                    "unsupported": why,
                    "unsupported_code": outcomes.unsupported_code(why),
                    "successes": 0,
                },
            )
            return
        used: set[Any] = set()
        failed: list[Any] = []
        fails: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        succ = 0
        if size < k_target:
            for p in points.enumerate_domain(ax, k_target):
                used.add(p)
                rec = self.run(p, 0, trace, "all")
                if _success(rec):
                    succ += 1
                    self.repeats(p, rec, "all")
                elif outcomes.counted(rec):
                    fails.append({"point": p, "outcome": rec.get("outcome")})
                else:
                    skipped.append({"point": p, "outcome": rec.get("outcome")})
            self._finish(
                unit_key,
                {
                    "successes": succ,
                    "failures": fails,
                    "skipped": skipped,
                    "domain": size,
                    "region_exhausted": False,
                },
            )
            return
        for p in points.ladder(ax, self.cfg):
            used.add(p)
            rec = self.run(p, 0, trace, "ladder")
            if not _success(rec) and not outcomes.counted(rec):
                skipped.append({"point": p, "outcome": rec.get("outcome"), "phase": 1})
                continue
            if not _success(rec):
                failed.append(p)
                fails.append({"point": p, "outcome": rec.get("outcome"), "phase": 1})
                break
            succ += 1
            self.repeats(p, rec, "ladder")
        exhausted = False
        if succ < k_target:
            sampler = points.Phase2.of(ax, self.cfg, self.eq, tool)
            p2_fail = 0
            while succ < k_target:
                p = sampler.next(failed, used)
                if p is None:
                    exhausted = True
                    break
                used.add(p)
                rec = self.run(p, 0, trace, "random")
                if _success(rec):
                    succ += 1
                    self.repeats(p, rec, "random")
                elif not outcomes.counted(rec):
                    skipped.append({"point": p, "outcome": rec.get("outcome"), "phase": 2})
                else:
                    failed.append(p)
                    fails.append({"point": p, "outcome": rec.get("outcome"), "phase": 2})
                    p2_fail += 1
                    if p2_fail >= int(pc["phase2_extra_failures"]):
                        exhausted = True
                        break
        self._finish(
            unit_key,
            {
                "successes": succ,
                "failures": fails,
                "skipped": skipped,
                "region_exhausted": exhausted,
            },
        )

    def check_once(self, unit_key: str, adapter: Any, trace: bool) -> None:
        """The single check of a CHECK_ONCE tool, K = 1: the symbolic run when the instance has
        one the adapter's ``symbolic_covered`` accepts, else the adapter's stated point."""
        p: points.Point | None = None
        covered = getattr(adapter, "symbolic_covered", lambda _subject: True)
        if (
            self.cfg["points"]["symbolic"]
            and subject.has_symbolic(self.inst)
            and covered(self.inst.subject)
        ):
            phase = "symbolic"
        else:
            p, phase = adapter.stated(self.inst, self.cfg), "stated"
            if p is None:
                why = "the published proof's stated size lies outside the domain"
                self._finish(
                    unit_key,
                    {
                        "K": 1,
                        "unsupported": why,
                        "unsupported_code": outcomes.unsupported_code(why),
                        "successes": 0,
                    },
                )
                return
        rec = self.run(p, 0, trace, phase)
        ok = _success(rec)
        if ok:
            self.repeats(p, rec, phase)
        row = [{"point": p, "outcome": rec.get("outcome"), "phase": phase}]
        self._finish(
            unit_key,
            {
                "K": 1,
                "check_phase": phase,
                "successes": int(ok),
                "failures": [] if ok or not outcomes.counted(rec) else row,
                "skipped": [] if ok or outcomes.counted(rec) else row,
                "region_exhausted": False,
            },
        )

    def _finish(self, unit_key: str, fields: dict[str, Any]) -> None:
        rec = {
            "key": unit_key,
            "config": self.ctx.cfg_hash,
            "unit": True,
            "equation": self.eq,
            "source": self.source,
            "truth": self.inst.truth,
            "subject": self.inst.subject.id,
            "category": self.inst.subject.category,
            "tool": self.unit.tool,
            "kind": self.unit.false["kind"] if self.unit.false else None,
            "restricted": bool(self.inst.restriction),
            "K": self.cfg["points"]["K"],
            "differs_checked": self.unit.false is not None,
            **fields,
        }
        self.ctx.writer().append(rec)
        self.ctx.done[unit_key] = rec


def calibrate(ctx_cfg: dict[str, Any], results: Path) -> dict[str, Any]:
    """The pinned calibration script, run three times in fresh processes: CPU seconds each and
    their median."""
    from benchmarks import stages

    _ = stages
    times = []
    for _ in range(3):
        code = (
            "import time; from benchmarks import stages; r = stages.new_run(); "
            "c = time.process_time(); "
            "stages.decide_stage(r, open(__import__('sys').argv[1]).read()); "
            "print(time.process_time() - c)"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".azx", delete=False) as fh:
            fh.write(CALIBRATION)
            path = fh.name
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        out = subprocess.run(
            [sys.executable, "-c", code, path],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        os.unlink(path)
        try:
            times.append(float(out.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            pass
    med = sorted(times)[len(times) // 2] if times else None
    return {"script": "GHZ symbolic decide", "cpu": times, "median": med}


def units_of(cfg: dict[str, Any], false_records: list[dict[str, Any]]) -> list[tuple[Unit, Any]]:
    """Every (S or F equation, tool) unit with its instance."""
    from benchmarks import tools

    sc = cfg["subjects"]
    subjects = subject.select(
        subject.catalogue(sc["catalogue"]), sc["include"], sc["exclude"], sc["categories"]
    )
    insts = {i.id: i for i in subject.instances(subjects)}
    out = []
    tool_names = [t for t in cfg["tools"]["enabled"] if t in tools.NAMES]
    for i in insts.values():
        for t in tool_names:
            out.append((Unit(i.id, t, None, bool(i.restriction)), i))
    if cfg["false_set"]["enabled"]:
        for rec in false_records:
            if rec.get("kind") is None:
                continue
            fi = breaks.false_instance(rec, insts)
            for t in tool_names:
                out.append((Unit(rec["equation"], t, rec, bool(fi.restriction)), fi))
    return out


def stale_units(done: dict[str, dict[str, Any]], false_records: list[dict[str, Any]]) -> set[str]:
    """The unit keys of finished F units whose unit record predates the differs check and
    that ran a point where their pair is equal, and of a CHECK_ONCE tool's units with K other
    than 1 or without a check_phase whose subject the adapter's ``symbolic_covered`` refuses."""
    from benchmarks import tools
    from benchmarks.differs import differs

    falses = {f"{r['equation']}!": r for r in false_records if r.get("kind") is not None}
    subjects = {s.id: s for s in subject.catalogue()}
    out: set[str] = set()
    once: dict[str, bool] = {}
    for rec in done.values():
        tool = str(rec.get("tool"))
        if rec.get("unit") and tool in tools.NAMES and tool not in once:
            once[tool] = bool(getattr(tools.adapter(tool), "CHECK_ONCE", False))
        if rec.get("unit") and once.get(tool) and rec.get("K") != 1:
            out.add(rec["key"])
            continue
        if rec.get("unit") and once.get(tool) and "check_phase" not in rec:
            covered = getattr(tools.adapter(tool), "symbolic_covered", None)
            subj = subjects.get(str(rec.get("subject")))
            if covered is not None and subj is not None and not covered(subj):
                out.add(rec["key"])
                continue
        false = falses.get(rec.get("equation", ""))
        if rec.get("unit") or false is None or rec.get("d") is None:
            continue
        if not differs(false, rec.get("n"), rec["d"]):
            out.add(f"{rec['config']}|{rec['equation']}@{rec['source']}|unit|{rec['tool']}")
    return {k for k in out if k in done and not done[k].get("differs_checked")}


FALSE_SET_POLL = 30.0


def false_set(
    cfg: dict[str, Any], results: Path, jobs: int, shard: tuple[int, int]
) -> list[dict[str, Any]]:
    """The F records of results/false_set.json: shard 0 generates them under an exclusive lock
    on false_set.lock when the file does not hold this config's set; every other shard waits,
    polling every FALSE_SET_POLL seconds, until it does."""
    import fcntl

    path = results / "false_set.json"
    if shard[0] == 0:
        with (results / "false_set.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return breaks.generate(cfg, path, jobs)
    waited = 0.0
    while True:
        records = breaks.load(cfg, path)
        if records is not None:
            return records
        if waited % 600 == 0:
            print(f"benchmarks: shard {shard[0]} waits for {path}", flush=True)
        time.sleep(FALSE_SET_POLL)
        waited += FALSE_SET_POLL


def run(
    cfg: dict[str, Any], shard: tuple[int, int] = (0, 1), only: list[str] | None = None
) -> None:
    """Runs every unit of ``cfg`` (this shard's), resuming from the logs."""
    import fnmatch

    results = config.path(cfg, "results")
    results.mkdir(parents=True, exist_ok=True)
    cfg_hash = config.digest(cfg)
    (results / "configs").mkdir(exist_ok=True)
    (results / "configs" / f"{cfg_hash}.json").write_text(json.dumps(cfg, indent=1, default=str))
    jobs = int(cfg["platform"]["jobs"])
    from benchmarks import tools

    problems = tools.preflight([t for t in cfg["tools"]["enabled"] if t in tools.NAMES])
    if problems:
        print("benchmarks: preflight failed, not starting:", flush=True)
        for p in problems:
            print(f"  {p}", flush=True)
        sys.exit(2)
    env = environment(cfg)
    calib = calibrate(cfg, results)
    env["calibration"] = calib
    envdir = results / "environment"
    envdir.mkdir(exist_ok=True)
    job = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    tag = f"{socket.gethostname().split('.')[0]}-{job}-{task}-{int(time.time())}"
    (envdir / f"{tag}.json").write_text(json.dumps(env, indent=1, default=str))
    false_records = false_set(cfg, results, jobs, shard) if cfg["false_set"]["enabled"] else []
    ctx = Context(
        cfg,
        cfg_hash,
        results,
        log.index(results, cfg_hash),
        env["machine"],
        {"median": calib["median"], "tag": tag},
        memory_cap_mb(cfg, jobs),
        lean.cgroups(),
        {"commit": env["git_commit"], "dirty": env["git_dirty"]},
    )
    ctx.stale = stale_units(ctx.done, false_records)
    units = units_of(cfg, false_records)
    i, total = shard
    units = [
        (u, inst)
        for u, inst in units
        if int(hashlib.sha256(u.id.encode()).hexdigest(), 16) % total == i
    ]
    if only:
        units = [(u, inst) for u, inst in units if any(fnmatch.fnmatch(u.id, g) for g in only)]
    print(
        f"benchmarks: {len(units)} units, config {cfg_hash}, {jobs} workers, "
        f"{ctx.mem_mb} MiB each, "
        f"{len(ctx.done)} runs already logged",
        flush=True,
    )

    def handler(signum: int, frame: Any) -> None:
        _stop.set()
        with _children_lock:
            procs = list(_children)
        for p in procs:
            threading.Thread(target=terminate, args=(p,), daemon=True).start()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    done = 0

    def one(item: tuple[Unit, Any]) -> None:
        nonlocal done
        u, inst = item
        try:
            UnitRun(u, inst, ctx).execute()
        except InterruptedError:
            return
        except Exception as exc:  # noqa: BLE001
            print(f"unit {u.id}: {type(exc).__name__}: {exc}", flush=True)
            return
        with ctx.lock:
            done += 1
            print(f"[{done}/{len(units)}] {u.id}", flush=True)

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(one, units))
    if _stop.is_set():
        print("benchmarks: interrupted; finished runs are logged, rerun to resume", flush=True)
        sys.exit(130)
    _ = shutil
