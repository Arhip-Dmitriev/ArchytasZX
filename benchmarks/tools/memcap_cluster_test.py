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

"""Cluster memory-cap test, run inside a Slurm job: ``python -m benchmarks.tools.memcap_cluster_test
[--export FILE] [--cap-mb M]``. It prints one JSON line per check and a summary line, and exits 0
only when all hold:

(a) ``lean.check`` of a small file importing ArchytasZX (or FILE) is accepted;
(b) ``lean.check`` of it with OLEAN_MB 0 and a 64 MiB budget is oom with lean_kill memcap.<mech>
    (or budget, the RssAnon watcher, when the mechanism is not srun);
(c) ``rival.run_binary`` of a child touching 2 M MiB under a cap of M MiB is oom with kill
    memcap.<mech>;
(d) a neighbour child touching M / 2 MiB under its own cap of M MiB, running alongside (b) and
    (c), exits 0 and is not oom."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from benchmarks import lean, memcap

SMALL = "import ArchytasZX\n\ntheorem memcap_cluster_t : (1 : Nat) + 1 = 2 := rfl\n"


def child(mb: float, hold: float) -> list[str]:
    """A Python child touching ``mb`` MiB of anonymous memory and holding it ``hold`` seconds."""
    code = f"import time; b = b'\\x01' * {int(mb * 2**20)}; time.sleep({hold}); print('done')"
    return [sys.executable, "-c", code]


def emit(row: dict[str, Any]) -> bool:
    """Prints ``row`` as one JSON line; returns its ``ok``."""
    print(json.dumps(row, default=str), flush=True)
    return bool(row["ok"])


def _env(extra: dict[str, str]) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    return old


def _restore(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def lean_fields(res: dict[str, Any]) -> dict[str, Any]:
    """The reported subset of a lean.check record."""
    keys = ("lean_status", "lean_kill", "lean_memcap", "lean_import", "axiom_class", "axioms")
    return {k: res.get(k) for k in (*keys, "lean_cpu", "lean_peak_mb", "lean_wall")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path)
    ap.add_argument("--cap-mb", type=float, default=300.0)
    args = ap.parse_args()
    mech = memcap.mechanism()
    good = emit(
        {
            "check": "env",
            "ok": bool(os.environ.get("SLURM_JOB_ID")),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            **memcap.describe(),
        }
    )
    with tempfile.TemporaryDirectory(prefix="azx-memcap-") as tmp:
        path = Path(tmp) / "memcap_cluster.lean"
        source = args.export.read_text(encoding="utf-8") if args.export else SMALL
        path.write_text(source, encoding="utf-8")

        res = lean.check(path, timeout=1800)
        ok = res["lean_status"] == "accepted" and res["lean_memcap"] == mech
        good &= emit({"check": "a_lean_under_budget", "ok": ok, **lean_fields(res)})

        neighbour: dict[str, Any] = {}

        def run_neighbour() -> None:
            proc, oom = memcap.run(child(args.cap_mb / 2, 60), memcap.Cap(args.cap_mb), 600)
            neighbour.update(exit=proc.returncode, oom=oom, stdout=proc.stdout.strip()[-80:])

        thread = threading.Thread(target=run_neighbour)
        thread.start()

        olean, lean.OLEAN_MB = lean.OLEAN_MB, 0
        old = _env({"BENCH_LEAN_BUDGET_MB": "64"})
        try:
            low = lean.check(path, timeout=1800)
        finally:
            lean.OLEAN_MB = olean
            _restore(old)
        kills = {f"memcap.{mech}"} | ({"budget"} if mech != "srun" else set())
        ok = low["lean_status"] == "oom" and low.get("lean_kill") in kills
        good &= emit({"check": "b_lean_over_budget", "ok": ok, **lean_fields(low)})

        from benchmarks.tools import rival

        old = _env({"BENCH_MEMORY_MB": str(int(args.cap_mb))})
        try:
            riv = rival.run_binary.__wrapped__(  # type: ignore[attr-defined]
                child(2 * args.cap_mb, 10), 120, lift_memory=True
            )
        finally:
            _restore(old)
        ok = riv["status"] == "oom" and riv.get("kill") == f"memcap.{mech}"
        riv.pop("stdout", None)
        good &= emit({"check": "c_rival_over_budget", "ok": ok, **riv})

        thread.join()
        ok = neighbour.get("exit") == 0 and not neighbour.get("oom")
        good &= emit({"check": "d_neighbour_alive", "ok": ok, **neighbour})
    emit({"check": "summary", "ok": good, "mechanism": mech})
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
