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

"""Case runner: one worker process per case, ``jobs`` at a time, killed at ``timeout``.

A suite module exposes ``cases(quick) -> list[dict]``, ``supports(case) -> str | None`` (the
reason a tool cannot run it) and ``run(case, repeat) -> dict``. A case is a JSON dict with at
least ``suite``, ``family`` and ``tool``; the row written back adds ``status``, ``time`` (median
seconds over ``repeat`` runs), ``times`` and ``detail``.
"""

from __future__ import annotations

import importlib
import json
import os
import resource
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


class Wrong(Exception):
    """A tool returned an incorrect answer."""


class Inconclusive(Exception):
    """A tool finished without settling the question."""


def timed(fn: Callable[[], object], repeat: int) -> tuple[object, list[float]]:
    """``fn`` run ``repeat`` times; its last value and every wall time."""
    times = []
    out: object = None
    for _ in range(max(1, repeat)):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return out, times


def row(
    case: dict, status: str, times: list[float] | None = None, detail: str = "", **extra: object
) -> dict:
    t = statistics.median(times) if times else None
    return {**case, **extra, "status": status, "time": t, "times": times or [], "detail": detail}


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


def _limit(mem_gb: float) -> Callable[[], None]:
    def apply() -> None:
        if mem_gb > 0:
            cap = int(mem_gb * 2**30)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.environ["OMP_NUM_THREADS"] = "1"

    return apply


def run_cases(
    cases: list[dict], *, jobs: int, timeout: float, repeat: int, mem_gb: float, log: bool = True
) -> list[dict]:
    """Every case's row, in case order; unsupported cases are not spawned."""
    out: list[dict | None] = [None] * len(cases)
    pending: list[int] = []
    for i, case in enumerate(cases):
        module = importlib.import_module(f"benchmarks.{case['suite']}")
        reason = module.supports(case)
        if reason is None:
            pending.append(i)
        else:
            out[i] = row(case, "unsupported", detail=reason)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    running: dict[int, tuple[subprocess.Popen[str], float]] = {}
    done = len(cases) - len(pending)
    pending.reverse()
    while pending or running:
        while pending and len(running) < jobs:
            i = pending.pop()
            cmd = [sys.executable, "-m", "benchmarks.harness", json.dumps(cases[i]), str(repeat)]
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_limit(mem_gb),  # noqa: PLW1509
            )
            running[i] = (proc, time.monotonic())
        time.sleep(0.02)
        for i, (proc, start) in list(running.items()):
            if proc.poll() is None:
                if time.monotonic() - start > timeout:
                    proc.kill()
                    proc.communicate()
                    out[i] = row(cases[i], "timeout", detail=f"> {timeout:g} s")
                    del running[i]
                else:
                    continue
            else:
                stdout, stderr = proc.communicate()
                lines = [ln for ln in stdout.splitlines() if ln.startswith(_MARK)]
                if lines:
                    out[i] = json.loads(lines[-1][len(_MARK) :])
                elif "MemoryError" in stderr or proc.returncode in (-9, 137):
                    out[i] = row(cases[i], "oom", detail=stderr.strip()[-300:])
                else:
                    out[i] = row(cases[i], "error", detail=stderr.strip()[-300:])
                del running[i]
            done += 1
            if log:
                r = out[i]
                assert r is not None
                t = "" if r["time"] is None else f"{r['time']:.3g}s"
                label = " ".join(f"{k}={v}" for k, v in cases[i].items() if k not in ("suite",))
                print(f"[{done}/{len(cases)}] {r['status']:<12} {t:>9}  {label}", flush=True)
    return [r for r in out if r is not None]


_OUTPUTS = frozenset(
    {"status", "time", "times", "detail", "verdict", "zx_nodes", "nodes_before", "nodes_after"}
)


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
    path.write_text(json.dumps(list(merged.values()), indent=1, default=str))
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
