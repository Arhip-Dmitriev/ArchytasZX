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

"""benchmarks.memcap: children over the cap are killed as oom by every usable mechanism, children
under it finish, rivals carry the mechanism, a Lean check's record does not depend on it."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

CAP_MB = 128.0
LOCAL = ("watchdog", "cgroup_v2", "systemd")


def _child(mb: float) -> list[str]:
    code = f"import time; b = b'\\x01' * {int(mb * 2**20)}; time.sleep(3)"
    return [sys.executable, "-c", code]


def _forced(mech: str) -> dict[str, str]:
    return {"BENCH_MEMCAP": mech, "BENCH_MEMORY_MB": str(int(CAP_MB))}


def _with_env(env: dict[str, str], fn: Any) -> Any:
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_a_child_over_the_cap_is_killed_as_oom() -> None:
    """A child touching twice the cap is killed and oom under each usable local mechanism; one
    touching half of it exits 0."""
    from benchmarks import memcap

    for mech in LOCAL:
        if not memcap.usable(mech):
            continue
        proc, oom = memcap.run(_child(2 * CAP_MB), memcap.Cap(CAP_MB, mech), 60)
        assert oom or proc.returncode == -9, (mech, proc.returncode)
        proc, oom = memcap.run(_child(CAP_MB / 2), memcap.Cap(CAP_MB, mech), 60)
        assert proc.returncode == 0 and not oom, (mech, proc.returncode)


def test_an_over_budget_rival_is_killed() -> None:
    """run_binary, address space lifted, of a child over the worker cap is oom with its
    mechanism and kill recorded; under the cap it is ok."""
    from benchmarks import memcap
    from benchmarks.tools import rival

    for mech in LOCAL:
        if not memcap.usable(mech):
            continue

        def over() -> dict[str, Any]:
            return rival.run_binary(_child(2 * CAP_MB), 30, lift_memory=True)

        def under() -> dict[str, Any]:
            return rival.run_binary(_child(CAP_MB / 2), 30, lift_memory=True)

        res = _with_env(_forced(mech), over)
        assert res["status"] == "oom" and res["memcap"] == mech, (mech, res)
        fine = _with_env(_forced(mech), under)
        assert fine["status"] == "ok" and fine["memcap"] == mech, (mech, fine)


def test_the_worker_mechanism() -> None:
    """worker_memcap: systemd under ctx.cgroup, cgroup_v2 when chosen, else rlimit_as."""
    from benchmarks import runner

    ctx = runner.Context({}, "t", Path("."), {}, {}, None, 0, True)
    assert runner.worker_memcap(ctx) == "systemd"
    ctx.cgroup, ctx.memcap = False, "cgroup_v2"
    assert runner.worker_memcap(ctx) == "cgroup_v2"
    ctx.memcap = "watchdog"
    assert runner.worker_memcap(ctx) == "rlimit_as"


LEAN_FIELDS = (
    "lean_status",
    "axioms",
    "axiom_class",
    "statement_level",
    "final_theorem",
    "lean_import",
    "file_bytes",
    "file_lines",
    "unproved",
)


def test_an_under_budget_lean_check_does_not_depend_on_the_mechanism() -> None:
    """lean.check of a small file gives the same record fields under each usable local
    mechanism, with lean_memcap naming it."""
    from benchmarks import lean, memcap

    if not (lean.LEAN_DIR / ".lake" / "packages").is_dir():
        return
    src = "theorem memcap_probe_t : 1 + 1 = 2 := rfl\n"
    seen: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memcap_probe.lean"
        path.write_text(src, encoding="utf-8")
        for mech in LOCAL:
            if not memcap.usable(mech):
                continue

            def one() -> dict[str, Any]:
                return lean.check(path, timeout=300)

            res = _with_env({"BENCH_MEMCAP": mech}, one)
            assert res["lean_memcap"] == mech, res
            seen[mech] = {k: res.get(k) for k in LEAN_FIELDS}
    first = next(iter(seen.values()))
    assert first["lean_status"] == "accepted", first
    assert all(v == first for v in seen.values()), seen


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name, flush=True)
