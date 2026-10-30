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

"""Case runner: every run is a fresh worker process; a case's runs go back to back, a sweep's
cases in ascending size, ``jobs`` sweeps at a time.

A suite module exposes ``SIZE`` (the swept field), ``cases(quick) -> list[dict]`` and
``run(case) -> dict``. A sweep is the cases equal on every field but ``SIZE``. Each run is
stopped by the kernel after ``timeout`` CPU seconds, killed after :data:`WALL_FACTOR` times that
in wall time, and held to a memory cap (a systemd cgroup with no swap, else RLIMIT_AS). A run
that times out or runs out of memory ends its case and every larger case of its sweep.

A case's row holds ``status`` (``ok`` when every run finished ``ok``, else the commonest other
status), ``time`` (median CPU seconds), ``time_q1``/``time_q3``, ``times``, ``wall``, ``walls``,
``runs`` and the extra fields of its first run. Every run is appended to
``results/<suite>.runs.jsonl``; finished cases are saved to ``results/<suite>.json``.
"""

from __future__ import annotations

import collections
import importlib
import json
import math
import os
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WALL_FACTOR = 4
SAFE_FRACTION = 0.85
MIN_MEMORY_GB = 0.75
_MARK = "@@RESULT "
_FAILED = ("timeout", "oom")
_OUTPUTS = ("status", "cpu", "wall", "detail", "run")


class Wrong(Exception):
    """A tool returned an incorrect answer."""


class Inconclusive(Exception):
    """A tool finished without settling the question."""


def measure(fn: Callable[[], object]) -> tuple[object, float, float]:
    """``fn()``, its CPU seconds and its wall seconds."""
    c, w = time.process_time(), time.perf_counter()
    out = fn()
    return out, time.process_time() - c, time.perf_counter() - w


def _worker(case: dict) -> dict:
    module = importlib.import_module(f"benchmarks.{case['suite']}")
    try:
        return {"status": "ok", **module.run(case)}  # type: ignore[no-any-return]
    except Wrong as exc:
        return {"status": "wrong", "detail": str(exc)[:500]}
    except Inconclusive as exc:
        return {"status": "inconclusive", "detail": str(exc)[:500]}
    except MemoryError:
        return {"status": "oom", "detail": "MemoryError"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": traceback.format_exception_only(exc)[-1][:500]}


def safe_memory_gb(jobs: int) -> float:
    """:data:`SAFE_FRACTION` of the available memory, shared by ``jobs`` runs."""
    with open("/proc/meminfo", encoding="ascii") as fh:
        info = {ln.split(":")[0]: int(ln.split()[1]) for ln in fh}
    return SAFE_FRACTION * info["MemAvailable"] / 2**20 / max(1, jobs)


def _cgroups() -> bool:
    """True when ``systemd-run --user --scope`` can hold a process to a memory cap."""
    if shutil.which("systemd-run") is None:
        return False
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64M", "true"]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def _limit(mem_gb: float, cpu_s: float, cgroup: bool) -> Callable[[], None]:
    def apply() -> None:
        if not cgroup:
            cap = int(mem_gb * 2**30)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        cpu = max(1, math.ceil(cpu_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 10))

    return apply


def _one_run(case: dict, timeout: float, mem_gb: float, cgroup: bool) -> dict:
    """One fresh worker process on ``case``."""
    cmd = [sys.executable, "-m", "benchmarks.harness", json.dumps(case)]
    if cgroup:
        caps = ["-p", f"MemoryMax={int(mem_gb * 1024)}M", "-p", "MemorySwapMax=0"]
        cmd = ["systemd-run", "--user", "--scope", "--quiet", *caps, *cmd]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_limit(mem_gb, timeout, cgroup),  # noqa: PLW1509
    )
    try:
        stdout, stderr = proc.communicate(timeout=WALL_FACTOR * timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return {"status": "timeout", "detail": f"> {WALL_FACTOR * timeout:g} s wall"}
    lines = [ln for ln in stdout.splitlines() if ln.startswith(_MARK)]
    if lines:
        return json.loads(lines[-1][len(_MARK) :])  # type: ignore[no-any-return]
    if proc.returncode in (-signal.SIGXCPU, 128 + signal.SIGXCPU):
        return {"status": "timeout", "detail": f"> {timeout:g} s cpu"}
    if "MemoryError" in stderr or proc.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL):
        return {"status": "oom", "detail": f"> {mem_gb:.2f} GB"}
    return {"status": "error", "detail": stderr.strip()[-300:]}


def _combine(case: dict, runs: list[dict], repeat: int) -> dict:
    """One row from a case's runs."""
    statuses = [r["status"] for r in runs]
    bad = [s for s in statuses if s != "ok"]
    status = collections.Counter(bad).most_common(1)[0][0] if bad else "ok"
    ok = [r for r in runs if r["status"] == "ok"]
    cpu = [r["cpu"] for r in ok]
    wall = [r["wall"] for r in ok]
    q1, q3 = statistics.quantiles(cpu, n=4)[::2] if len(cpu) > 1 else (cpu or [None]) * 2
    first = next((r for r in runs if r["status"] == status), runs[0])
    extra = {k: v for k, v in first.items() if k not in _OUTPUTS}
    return {
        **case,
        **extra,
        "status": status,
        "detail": first.get("detail", ""),
        "time": statistics.median(cpu) if cpu else None,
        "time_q1": q1,
        "time_q3": q3,
        "times": cpu,
        "wall": statistics.median(wall) if wall else None,
        "walls": wall,
        "runs": len(ok),
        "planned_runs": repeat,
        "run_status": statuses,
    }


def _key(case: dict) -> str:
    return json.dumps(case, sort_keys=True)


def _sweeps(cases: list[dict], size: str) -> list[list[dict]]:
    """The cases grouped by every field but ``size``, each group in ascending ``size``."""
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(_key({k: v for k, v in case.items() if k != size}), []).append(case)
    return [sorted(g, key=lambda c: c[size]) for g in groups.values()]


def run_suite(
    suite: str, *, quick: bool, jobs: int, timeout: float, repeat: int, resume: bool
) -> list[dict]:
    """Every case's row of ``suite`` (see the module docstring)."""
    module = importlib.import_module(f"benchmarks.{suite}")
    cases = module.cases(quick)
    RESULTS.mkdir(exist_ok=True)
    rows_path = RESULTS / f"{suite}.json"
    runs_path = RESULTS / f"{suite}.runs.jsonl"
    done: dict[str, dict] = {}
    if resume and rows_path.exists():
        for r in json.loads(rows_path.read_text(encoding="utf-8")):
            done[_key({k: r[k] for k in cases[0] if k in r})] = r
    elif runs_path.exists():
        runs_path.unlink()
    mem_gb = safe_memory_gb(jobs)
    if mem_gb < MIN_MEMORY_GB:
        raise SystemExit(
            f"{suite}: {mem_gb:.2f} GB per run with {jobs} jobs is under {MIN_MEMORY_GB} GB; "
            "free memory or lower --jobs"
        )
    cgroup = _cgroups()
    lock = threading.Lock()
    sweeps = _sweeps(cases, module.SIZE)
    total = len(cases)
    counter = [len([c for c in cases if _key(c) in done])]

    def save() -> None:
        rows = sorted(done.values(), key=lambda r: [str(r.get(k)) for k in cases[0]])
        rows_path.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")

    def sweep_worker(queue: list[list[dict]]) -> None:
        while True:
            with lock:
                if not queue:
                    return
                sweep = queue.pop(0)
            stopped: str | None = None
            for case in sweep:
                if _key(case) in done:
                    if done[_key(case)]["status"] in _FAILED:
                        stopped = f"not run: {module.SIZE} = {case[module.SIZE]} failed earlier"
                    continue
                if stopped is not None:
                    row = {**case, "status": "skipped", "detail": stopped, "time": None}
                else:
                    runs: list[dict] = []
                    for k in range(repeat):
                        r = _one_run(case, timeout, mem_gb, cgroup) | {"run": k}
                        runs.append(r)
                        with lock, runs_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({**case, **r}, default=str) + "\n")
                        if r["status"] in _FAILED:
                            break
                    row = _combine(case, runs, repeat)
                    if row["status"] in _FAILED:
                        stopped = f"not run: {module.SIZE} = {case[module.SIZE]} {row['status']}"
                with lock:
                    done[_key(case)] = row
                    counter[0] += 1
                    save()
                    _log(f"{counter[0]}/{total}", row, case)

    queue = list(sweeps)
    threads = [threading.Thread(target=sweep_worker, args=(queue,)) for _ in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    save()
    return list(done.values())


def _log(where: str, r: dict, case: dict) -> None:
    t = "" if r.get("time") is None else f"{r['time']:.3g}s cpu"
    label = " ".join(f"{k}={v}" for k, v in case.items() if k != "suite")
    print(f"[{case['suite']} {where}] {r['status']:<12} {t:>13}  {label}", flush=True)


def load(suite: str) -> list[dict]:
    """The saved rows of ``suite``."""
    path = RESULTS / f"{suite}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


if __name__ == "__main__":
    from benchmarks import harness as _self  # noqa: PLW0406

    print(_MARK + json.dumps(_self._worker(json.loads(sys.argv[1])), default=str), flush=True)
