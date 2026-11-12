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

"""One run in a fresh process: ``python -m benchmarks.worker JOB_JSON_FILE``.

The job names the equation, version, tool, point, repeat and the resolved config. The worker
prints ``@@STAGE`` before each stage and ``@@RESULT`` with the run's fields at the end, so the
parent can tell which stage a killed run was in.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from typing import Any

STAGE = "@@STAGE "
RESULT = "@@RESULT "
OOMKILL = "@@OOMKILL "
_oom_start: int | None = None
_prepare = {"on": False, "expired": False}


class PrepareTimeout(Exception):
    """A rival's preparation used up its CPU limit."""


def _prepare_alarm(signum: int, frame: Any) -> None:
    """Raises PrepareTimeout, re-armed every 0.5 s CPU until it escapes."""
    if not _prepare["on"]:
        return
    _prepare["expired"] = True
    signal.setitimer(signal.ITIMER_PROF, 0.5)
    raise PrepareTimeout()


def start_prepare(limit: float) -> None:
    """Arms a CPU timer of ``limit`` seconds over a rival's preparation."""
    _prepare.update(on=True, expired=False)
    signal.signal(signal.SIGPROF, _prepare_alarm)
    signal.setitimer(signal.ITIMER_PROF, limit)


def end_prepare() -> None:
    """Disarms the preparation timer; raises PrepareTimeout when it had expired."""
    was = _prepare["on"]
    _prepare["on"] = False
    signal.setitimer(signal.ITIMER_PROF, 0)
    if was and _prepare["expired"]:
        raise PrepareTimeout()


def mark(stage: str) -> None:
    """Announces the stage about to start."""
    print(STAGE + stage, flush=True)


def instance(job: dict[str, Any]) -> Any:
    """The S or F instance a job names."""
    from benchmarks import breaks, subject

    cfg = job["config"]
    sc = cfg["subjects"]
    subjects = subject.select(
        subject.catalogue(sc["catalogue"]), sc["include"], sc["exclude"], sc["categories"]
    )
    insts = {i.id: i for i in subject.instances(subjects)}
    if job.get("false") is not None:
        return breaks.false_instance(job["false"], insts)
    return insts[job["equation"]]


def oom_kills() -> int | None:
    """The oom_kill counter of this process's own systemd scope (cgroup v2 memory.events),
    None outside a scope."""
    try:
        with open("/proc/self/cgroup", encoding="ascii") as fh:
            path = next(ln.split("::", 1)[1].strip() for ln in fh if ln.startswith("0::"))
        if not path.endswith(".scope"):
            return None
        with open(f"/sys/fs/cgroup{path}/memory.events", encoding="ascii") as fh:
            return next(int(ln.split()[1]) for ln in fh if ln.startswith("oom_kill "))
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _terminate(signum: int, frame: Any) -> None:
    """Writes the scope's oom_kill rise since start, kills the running Lean checks, then
    exits."""
    now = oom_kills()
    if now is not None and _oom_start is not None:
        os.write(1, f"{OOMKILL}{now - _oom_start}\n".encode())
    lean = sys.modules.get("benchmarks.lean")
    if lean is not None:
        lean.kill_active()
    os._exit(128 + signum)


def failure(tool: str, exc: BaseException) -> dict[str, Any]:
    """The record of an exception raised in the worker: unsupported for a translate or rival
    Unsupported, oom for MemoryError, error for ArchytasZX, harness_error (reason
    "prepare: ...") for a rival, whose own call runs in a subprocess."""
    import traceback

    from benchmarks import translate
    from benchmarks.tools import common, rival

    if isinstance(exc, (translate.Unsupported, rival.Unsupported)):
        return common.unsupported(str(exc))
    msg = traceback.format_exception_only(exc)[-1].strip()[:500]
    if isinstance(exc, MemoryError):
        return {"outcome": "oom", "killed_in": "prepare", "detail": msg}
    if tool == "archytaszx":
        return {"outcome": "error", "detail": msg}
    return {"outcome": "harness_error", "reason": f"prepare: {msg}", "detail": msg}


def main() -> None:
    global _oom_start
    sys.set_int_max_str_digits(0)
    if not os.environ.get("BENCH_MEMORY_MB"):
        _oom_start = oom_kills()
    signal.signal(signal.SIGTERM, _terminate)
    with open(sys.argv[1], encoding="utf-8") as fh:
        job = json.load(fh)
    os.environ["ARCHYTASZX_TRACE"] = "1" if job.get("trace") else "0"
    cap = os.environ.get("BENCH_MEMORY_MB")
    if cap:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (int(cap) * 2**20, resource.RLIM_INFINITY))
    from benchmarks import tools

    inst = instance(job)
    adapter = tools.adapter(job["tool"])
    rival = job["tool"] != "archytaszx"
    st = job["config"]["stages"]
    if rival:
        mark("prepare")
        start_prepare(float(st.get("prepare_timeout_seconds", st["timeout_seconds"])))
    try:
        out = adapter.run(inst, job)
        if rival:
            end_prepare()
    except PrepareTimeout:
        out = {"outcome": "timeout", "killed_in": "prepare", "limit": "prepare_cpu"}
    except Exception as exc:  # noqa: BLE001
        out = failure(job["tool"], exc)
    finally:
        _prepare["on"] = False
        signal.setitimer(signal.ITIMER_PROF, 0)
    if rival and _prepare["expired"] and out.get("outcome") != "timeout":
        out = {"outcome": "timeout", "killed_in": "prepare", "limit": "prepare_cpu"}
    print(RESULT + json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
