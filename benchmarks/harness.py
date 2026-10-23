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

"""Case runner: ``repeat`` fresh worker processes per case (one run each, cold caches), ``jobs``
processes at a time.

Each run is stopped by the kernel after ``timeout`` CPU seconds (RLIMIT_CPU), killed after
:data:`WALL_FACTOR` times that in wall time, and held to a memory cap on resident memory (a
systemd cgroup with no swap; RLIMIT_AS when systemd-run is missing). Run ``k`` of every case
forms its own sweep over ``n``: a timeout or memory failure of run ``k`` at ``n`` ends run ``k``
for every larger ``n`` with the same other keys.

A suite module exposes ``cases(quick) -> list[dict]``, ``supports(case) -> str | None`` (the
reason a tool cannot run it) and ``run(case, repeat) -> dict``. The row written back for a case
adds ``status`` (``ok`` when at least half its runs finished, else the commonest failure),
``time`` (median CPU seconds over the finished runs), ``time_q1``/``time_q3`` (quartiles),
``time_mean``, ``time_std``, ``times``, ``wall``, ``walls``, ``runs`` (finished),
``failed_runs``, ``shot_status`` (per run) and ``detail``. CPU time is ``time.process_time``:
every thread of the worker, none of the time it waits for a core. Runs go in passes: run 1 of
every case, then run 2, and so on. Every finished run is appended to
``results/<suite>.shots.jsonl`` and every finished case saved at once; ``resume`` keeps the
logged runs, otherwise the run log starts empty.
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
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from benchmarks.circuit import Unsupported

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
STATUSES = ("ok", "inconclusive", "wrong", "unsupported", "timeout", "oom", "error")
_MARK = "@@RESULT "
WALL_FACTOR = 4
SAFE_FRACTION = 0.85
_FAILED = ("timeout", "oom")


class Wrong(Exception):
    """A tool returned an incorrect answer."""


class Inconclusive(Exception):
    """A tool finished without settling the question."""


def timed(fn: Callable[[], object], repeat: int) -> tuple[object, list[tuple[float, float]]]:
    """``fn`` run ``repeat`` times; its last value and every (CPU, wall) time in seconds."""
    times = []
    out: object = None
    for _ in range(max(1, repeat)):
        c, w = time.process_time(), time.perf_counter()
        out = fn()
        times.append((time.process_time() - c, time.perf_counter() - w))
    return out, times


def row(
    case: dict,
    status: str,
    times: list[tuple[float, float]] | None = None,
    detail: str = "",
    **extra: object,
) -> dict:
    cpu = [c for c, _ in times or []]
    wall = [w for _, w in times or []]
    q1, q3 = statistics.quantiles(cpu, n=4)[::2] if len(cpu) > 1 else (cpu or [None]) * 2
    return {
        **case,
        **extra,
        "status": status,
        "time": statistics.median(cpu) if cpu else None,
        "time_q1": q1,
        "time_q3": q3,
        "time_mean": statistics.fmean(cpu) if cpu else None,
        "time_std": statistics.stdev(cpu) if len(cpu) > 1 else 0.0,
        "times": cpu,
        "wall": statistics.median(wall) if wall else None,
        "walls": wall,
        "runs": len(cpu),
        "detail": detail,
    }


def _combine(case: dict, shots: dict[int, dict]) -> dict:
    """One row from a case's runs: ``ok`` when at least half finished, else the commonest
    failure."""
    ordered = [shots[k] for k in sorted(shots)]
    ok = [s for s in ordered if s["status"] == "ok"]
    bad = [s for s in ordered if s["status"] != "ok"]
    if len(ok) * 2 >= len(ordered):
        status, first = "ok", ok[0]
    else:
        status = collections.Counter(s["status"] for s in bad).most_common(1)[0][0]
        first = next(s for s in bad if s["status"] == status)
    pairs = [(c, w) for s in ok for c, w in zip(s["times"], s["walls"])]
    extra = {k: v for k, v in first.items() if k not in case and k not in _OUTPUTS}
    extra |= {"failed_runs": len(bad), "shot_status": [s["status"] for s in ordered]}
    return row(case, status, pairs, first["detail"], **extra)


def _worker(case: dict, repeat: int) -> dict:
    module = importlib.import_module(f"benchmarks.{case['suite']}")
    try:
        return module.run(case, repeat)  # type: ignore[no-any-return]
    except Unsupported as exc:
        return row(case, "unsupported", detail=str(exc))
    except Wrong as exc:
        return row(case, "wrong", detail=str(exc))
    except Inconclusive as exc:
        return row(case, "inconclusive", detail=str(exc))
    except MemoryError:
        return row(case, "oom", detail="MemoryError")
    except RecursionError as exc:
        return row(case, "error", detail=f"RecursionError: {exc}")
    except Exception as exc:  # noqa: BLE001
        last = traceback.format_exception_only(exc)[-1].strip()
        return row(case, "error", detail=last[:500])


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
        if mem_gb > 0 and not cgroup:
            cap = int(mem_gb * 2**30)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        cpu = max(1, math.ceil(cpu_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 10))
        os.environ["OMP_NUM_THREADS"] = "1"

    return apply


def _group(case: dict) -> str:
    """The case without its ``n``: the sweep a failure at some ``n`` ends."""
    return json.dumps({k: v for k, v in case.items() if k != "n"}, sort_keys=True, default=str)


def _skip(case: dict, shot: int, failed: dict[tuple[str, int], tuple[int, str]]) -> dict | None:
    """The row of run ``shot`` of a case whose sweep already failed at a smaller ``n``, with
    that failure's status."""
    first = failed.get((_group(case), shot))
    if first is None or not isinstance(case.get("n"), int) or case["n"] <= first[0]:
        return None
    return row(case, first[1], detail=f"not run: n = {first[0]} failed ({first[1]}) in this run")


def _fail(
    failed: dict[tuple[str, int], tuple[int, str]], key: tuple[str, int], n: int, status: str
) -> None:
    if key not in failed or n < failed[key][0]:
        failed[key] = (n, status)


def _shot_result(case: dict, proc: subprocess.Popen[str], timeout: float, mem_gb: float) -> dict:
    """The row of one finished worker process."""
    stdout, stderr = proc.communicate()
    lines = [ln for ln in stdout.splitlines() if ln.startswith(_MARK)]
    if lines:
        return json.loads(lines[-1][len(_MARK) :])  # type: ignore[no-any-return]
    code = proc.returncode
    if code in (-signal.SIGXCPU, 128 + signal.SIGXCPU):
        return row(case, "timeout", detail=f"> {timeout:g} s cpu")
    if "MemoryError" in stderr or code in (-signal.SIGKILL, 128 + signal.SIGKILL):
        return row(case, "oom", detail=f"> {mem_gb:.2f} GB")
    return row(case, "error", detail=stderr.strip()[-300:])


def run_cases(
    cases: list[dict],
    *,
    jobs: int,
    timeout: float,
    repeat: int,
    mem_gb: float,
    resume: bool = False,
    log: bool = True,
) -> list[dict]:
    """Every case's row, in case order (see the module docstring)."""
    repeat = max(1, repeat)
    out: list[dict | None] = [None] * len(cases)
    shots: dict[int, dict[int, dict]] = {}
    failed: dict[tuple[str, int], tuple[int, str]] = {}
    suites = {c["suite"] for c in cases}
    kept = {(_key(r), r["shot"]): r for s in suites for r in load_shots(s)} if resume else {}
    if not resume:
        for suite in suites:
            _shots_path(suite).unlink(missing_ok=True)
    for (_, k), r in kept.items():
        if r["status"] in _FAILED and isinstance(r.get("n"), int) and k < repeat:
            _fail(failed, (_group(_case_of(r)), k), r["n"], r["status"])
    for i, case in enumerate(cases):
        module = importlib.import_module(f"benchmarks.{case['suite']}")
        reason = module.supports(case)
        if reason is not None:
            out[i] = row(case, "unsupported", detail=reason)
            save(case["suite"], [out[i]])
            continue
        shots[i] = {k: kept[_key(case), k] for k in range(repeat) if (_key(case), k) in kept}
        if len(shots[i]) == repeat:
            out[i] = _combine(case, shots[i])
            save(case["suite"], [out[i]])
    pending = [(i, k) for k in range(repeat) for i in shots if k not in shots[i]]
    total = len(pending)
    cgroup = mem_gb > 0 and _cgroups()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    running: dict[int, tuple[int, int, subprocess.Popen[str], float]] = {}
    slot = 0
    finished = 0

    def record(i: int, k: int, shot: dict) -> None:
        nonlocal finished
        shot = shot | {"shot": k}
        shots[i][k] = shot
        append_shot(cases[i]["suite"], shot)
        finished += 1
        if log:
            _log(f"run {k + 1}/{repeat}  {finished}/{total}", shot, cases[i])
        if len(shots[i]) == repeat:
            out[i] = _combine(cases[i], shots[i])
            save(cases[i]["suite"], [out[i]])

    pending.reverse()
    while pending or running:
        while pending and len(running) < jobs:
            i, k = pending.pop()
            if (skipped := _skip(cases[i], k, failed)) is not None:
                record(i, k, skipped)
                continue
            cmd = [sys.executable, "-m", "benchmarks.harness", json.dumps(cases[i]), "1"]
            if cgroup:
                mem = f"MemoryMax={int(mem_gb * 1024)}M"
                cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", mem]
                cmd += ["-p", "MemorySwapMax=0", sys.executable, "-m", "benchmarks.harness"]
                cmd += [json.dumps(cases[i]), "1"]
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_limit(mem_gb, timeout, cgroup),  # noqa: PLW1509
            )
            running[slot] = (i, k, proc, time.monotonic())
            slot += 1
        time.sleep(0.02)
        for key, (i, k, proc, start) in list(running.items()):
            if proc.poll() is None:
                if time.monotonic() - start <= WALL_FACTOR * timeout:
                    continue
                proc.kill()
                proc.communicate()
                shot = row(cases[i], "timeout", detail=f"> {WALL_FACTOR * timeout:g} s wall")
            else:
                shot = _shot_result(cases[i], proc, timeout, mem_gb)
            del running[key]
            if shot["status"] in _FAILED and isinstance(cases[i].get("n"), int):
                _fail(failed, (_group(cases[i]), k), cases[i]["n"], shot["status"])
            record(i, k, shot)
    return [r for r in out if r is not None]


def _log(where: str, r: dict, case: dict) -> None:
    t = "" if r["time"] is None else f"{r['time']:.3g}s cpu"
    label = " ".join(f"{k}={v}" for k, v in case.items() if k != "suite")
    print(f"[{where}] {r['status']:<12} {t:>13}  {label}", flush=True)


def _case_of(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in _OUTPUTS and k != "shot"}


def _shots_path(suite: str) -> Path:
    return RESULTS / f"{suite}.shots.jsonl"


def append_shot(suite: str, shot: dict) -> None:
    """One run's row appended to the suite's run log."""
    RESULTS.mkdir(exist_ok=True)
    with _shots_path(suite).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(shot, default=str) + "\n")


def load_shots(suite: str) -> list[dict]:
    """Every complete line of the suite's run log."""
    path = _shots_path(suite)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


_OUTPUTS = frozenset(
    {
        "status", "time", "time_q1", "time_q3", "time_mean", "time_std", "times", "wall",
        "walls", "runs", "failed_runs", "shot_status", "detail", "verdict", "zx_nodes",
        "nodes_before", "nodes_after", "shot",
    }
)  # fmt: skip


def _key(r: dict) -> str:
    return json.dumps(
        {k: v for k, v in r.items() if k not in _OUTPUTS}, sort_keys=True, default=str
    )


def save(suite: str, rows: list[dict]) -> Path:
    """Rows merged into the saved ones; a rerun case replaces its old row."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{suite}.json"
    merged = {_key(r): r for r in load(suite)}
    merged.update({_key(r): r for r in rows})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(list(merged.values()), indent=1, default=str))
    tmp.replace(path)
    return path


def load(suite: str) -> list[dict]:
    path = RESULTS / f"{suite}.json"
    return json.loads(path.read_text()) if path.exists() else []


if __name__ == "__main__":
    sys.setrecursionlimit(20000)
    result = importlib.import_module("benchmarks.harness")._worker(
        json.loads(sys.argv[1]), int(sys.argv[2])
    )
    print(_MARK + json.dumps(result, default=str), flush=True)
