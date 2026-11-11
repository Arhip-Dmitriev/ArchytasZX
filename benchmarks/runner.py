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

from benchmarks import breaks, config, lean, log, points, sources, stats, subject

ROOT = Path(__file__).resolve().parent.parent
STAGE_MARK = "@@STAGE "
RESULT_MARK = "@@RESULT "
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
        "git_dirty": bool(git("status", "--porcelain")),
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
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(STAGE_MARK):
            stage = line[len(STAGE_MARK) :].strip()
        elif line.startswith(RESULT_MARK):
            result = json.loads(line[len(RESULT_MARK) :])
    err = proc.stderr.read() if proc.stderr is not None else ""
    code = proc.wait()
    timer.cancel()
    with _children_lock:
        _children.discard(proc)
    os.unlink(job_path)
    wall = time.perf_counter() - w0
    if _stop.is_set() and result is None:
        raise InterruptedError
    if result is not None:
        result["process_wall"] = wall
        return result
    status = (
        "timeout" if killed.is_set() else ("oom" if code in (-signal.SIGKILL, 137) else "error")
    )
    return {
        "outcome": status,
        "killed_in": stage,
        "exit": code,
        "detail": err.strip()[-400:],
        "process_wall": wall,
    }


def _success(rec: dict[str, Any]) -> bool:
    return rec.get("outcome") == "ok"


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
        if k in self.ctx.done:
            return self.ctx.done[k]
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
            **res,
        }
        self.ctx.writer().append(rec)
        self.ctx.done[k] = rec
        return rec

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
        why = tools.adapter(tool).supports(self.inst, None)
        if why is not None:
            self._finish(unit_key, {"unsupported": why, "successes": 0})
            return
        pc = self.cfg["points"]
        if pc["symbolic"] and subject.has_symbolic(self.inst):
            rec = self.run(None, 0, trace, "symbolic")
            if _success(rec):
                self.repeats(None, rec, "symbolic")
        ax = points.axes(self.inst, self.cfg, self.unit.false)
        k_target = int(pc["K"])
        size = points.domain_size(ax, k_target)
        used: set[Any] = set()
        failed: list[Any] = []
        fails: list[dict[str, Any]] = []
        succ = 0
        if size < k_target:
            for p in points.enumerate_domain(ax, k_target):
                used.add(p)
                rec = self.run(p, 0, trace, "all")
                if _success(rec):
                    succ += 1
                    self.repeats(p, rec, "all")
                else:
                    fails.append({"point": p, "outcome": rec.get("outcome")})
            self._finish(
                unit_key,
                {"successes": succ, "failures": fails, "domain": size, "region_exhausted": False},
            )
            return
        for p in points.ladder(ax, self.cfg):
            used.add(p)
            rec = self.run(p, 0, trace, "ladder")
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
                else:
                    failed.append(p)
                    fails.append({"point": p, "outcome": rec.get("outcome"), "phase": 2})
                    p2_fail += 1
                    if p2_fail >= int(pc["phase2_extra_failures"]):
                        exhausted = True
                        break
        self._finish(
            unit_key, {"successes": succ, "failures": fails, "region_exhausted": exhausted}
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
    that ran a point where their pair is equal."""
    from benchmarks.differs import differs

    falses = {f"{r['equation']}!": r for r in false_records if r.get("kind") is not None}
    out: set[str] = set()
    for rec in done.values():
        false = falses.get(rec.get("equation", ""))
        if rec.get("unit") or false is None or rec.get("d") is None:
            continue
        if not differs(false, rec.get("n"), rec["d"]):
            out.add(f"{rec['config']}|{rec['equation']}@{rec['source']}|unit|{rec['tool']}")
    return {k for k in out if k in done and not done[k].get("differs_checked")}


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
    env = environment(cfg)
    calib = calibrate(cfg, results)
    env["calibration"] = calib
    envdir = results / "environment"
    envdir.mkdir(exist_ok=True)
    job = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    tag = f"{socket.gethostname().split('.')[0]}-{job}-{task}-{int(time.time())}"
    (envdir / f"{tag}.json").write_text(json.dumps(env, indent=1, default=str))
    false_records = (
        breaks.generate(cfg, results / "false_set.json", jobs)
        if cfg["false_set"]["enabled"]
        else []
    )
    ctx = Context(
        cfg,
        cfg_hash,
        results,
        log.index(results, cfg_hash),
        env["machine"],
        {"median": calib["median"], "tag": tag},
        memory_cap_mb(cfg, jobs),
        lean.cgroups(),
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
