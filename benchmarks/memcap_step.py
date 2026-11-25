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

"""In-step wrapper, standard library only: ``python -I memcap_step.py STATS CMD...`` runs CMD as
its child, polls the child tree's RssAnon and CPU every 0.25 s, and on its end writes to STATS
the JSON ``{"code", "cpu", "maxrss_kb", "peak_anon_mb", "cpu_tree"}`` (cpu and maxrss_kb from
the reaped children's rusage); it exits with CMD's code, 128 + N for a signal N."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading


def _children(pid: int) -> list[int]:
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


def tree_usage(pid: int) -> tuple[float, float]:
    """(RssAnon MiB, CPU seconds with reaped children) over ``pid`` and its descendants."""
    anon, ticks, stack, seen = 0, 0, [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/status", encoding="ascii", errors="replace") as fh:
                anon += next((int(ln.split()[1]) for ln in fh if ln.startswith("RssAnon:")), 0)
            with open(f"/proc/{p}/stat", encoding="ascii", errors="replace") as fh:
                ticks += sum(int(f) for f in fh.read().rsplit(")", 1)[1].split()[11:15])
        except (OSError, IndexError, ValueError):
            continue
        stack.extend(_children(p))
    return anon / 1024, ticks / os.sysconf("SC_CLK_TCK")


def main() -> int:
    stats, cmd = sys.argv[1], sys.argv[2:]
    proc = subprocess.Popen(cmd)
    seen = {"peak_anon_mb": 0.0, "cpu_tree": 0.0}
    done = threading.Event()

    def poll() -> None:
        while not done.wait(0.25):
            anon, cpu = tree_usage(proc.pid)
            seen["peak_anon_mb"] = max(seen["peak_anon_mb"], anon)
            seen["cpu_tree"] = max(seen["cpu_tree"], cpu)

    threading.Thread(target=poll, daemon=True).start()
    code = proc.wait()
    done.set()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    code = 128 - code if code < 0 else code
    row = {
        "code": code,
        "cpu": usage.ru_utime + usage.ru_stime,
        "maxrss_kb": usage.ru_maxrss,
        **seen,
    }
    with open(stats, "w", encoding="utf-8") as fh:
        json.dump(row, fh)
    return code


if __name__ == "__main__":
    sys.exit(main())
