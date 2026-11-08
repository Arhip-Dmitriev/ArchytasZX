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


def _terminate(signum: int, frame: Any) -> None:
    """Kills the running Lean checks, then exits."""
    lean = sys.modules.get("benchmarks.lean")
    if lean is not None:
        lean.kill_active()
    os._exit(128 + signum)


def main() -> None:
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
    try:
        out = adapter.run(inst, job)
    except Exception as exc:  # noqa: BLE001
        import traceback

        out = {"outcome": "error", "detail": traceback.format_exception_only(exc)[-1].strip()[:500]}
    print(RESULT + json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
