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
    from benchmarks import memcap

    saved = memcap.srun_ok
    memcap.srun_ok = lambda: True
    try:
        assert _with_env({"BENCH_WORKER_STEP": "srun"}, lambda: runner.worker_memcap(ctx)) == "srun"
        assert (
            _with_env({"BENCH_WORKER_STEP": ""}, lambda: runner.worker_memcap(ctx)) == "rlimit_as"
        )
        cmd = memcap.Cap(512, "srun", tag="worker").command(["w"])
        assert cmd[:6] == ["srun", "--overlap", "--quiet", "-n1", "-N1", "--mem=512M"], cmd
    finally:
        memcap.srun_ok = saved


def test_the_batch_memory_probe() -> None:
    """batchmem_probe: memory.stat fields parsed in plain and total_ forms; a snapshot holds
    meminfo and the /tmp filesystem; dropcache counts the files under a directory."""
    import tempfile

    from benchmarks.tools import batchmem_probe as bp

    st = bp.parse_stat("cache 10\nrss 5\nshmem 7\ntotal_shmem 9\npgfault 3\n")
    assert st == {"cache": 10, "rss": 5, "shmem": 7, "total_shmem": 9}, st
    snap = bp.snapshot("t")
    assert snap["label"] == "t" and "MemTotal" in snap["meminfo"], snap
    assert snap["fs"]["/tmp"] is None or snap["fs"]["/tmp"]["size"] > 0, snap["fs"]
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            Path(d, f"f{i}").write_text("x")
        assert bp.dropcache([d]) == 3


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


SHIM = """#!/bin/bash
mem=""
while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do
  case "$1" in --mem=*M) mem="${1#--mem=}"; mem="${mem%M}";; esac
  shift
done
if [ -n "$FAKE_SRUN_ENFORCE" ] && [ -n "$mem" ]; then
  ( ulimit -v $((mem * 1024)); exec "$@" )
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "slurmstepd: error: Detected 1 oom_kill event in StepId=1.0. Out Of Memory" >&2
    exit 137
  fi
  exit 0
fi
exec "$@"
"""


def _exec_dir() -> Path:
    """A directory for the shim outside a possibly noexec /tmp."""
    path = Path.home() / ".cache" / "archytaszx-tests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_srun(tmp: str, enforce: bool) -> dict[str, str]:
    """The environment of a Slurm job whose ``srun`` is a shim running the command directly
    (under ``ulimit -v`` of --mem and reporting a step oom when ``enforce``)."""
    shim = Path(tmp) / "srun"
    shim.write_text(SHIM, encoding="utf-8")
    shim.chmod(0o755)
    env = {
        "PATH": f"{tmp}:{os.environ.get('PATH', '')}",
        "SLURM_JOB_ID": "4242",
        "BENCH_MEMCAP": "srun",
        "BENCH_MEMORY_MB": str(int(CAP_MB)),
    }
    return env | ({"FAKE_SRUN_ENFORCE": "1"} if enforce else {"FAKE_SRUN_ENFORCE": ""})


def test_srun_is_the_default_in_slurm_without_systemd_or_cgroups() -> None:
    """mechanism(): srun inside a Slurm job where systemd and cgroup_v2 are unusable,
    BENCH_MEMCAP overriding it; outside Slurm the AUTO order."""
    from benchmarks import memcap

    saved = (memcap.systemd_ok, memcap.delegated_parent, memcap.srun_ok)
    memcap.systemd_ok = lambda: False  # type: ignore[assignment]
    memcap.delegated_parent = lambda: None  # type: ignore[assignment]
    try:
        memcap.srun_ok = lambda: True
        assert _with_env({"BENCH_MEMCAP": ""}, memcap.mechanism) == "srun"
        assert _with_env({"BENCH_MEMCAP": "watchdog"}, memcap.mechanism) == "watchdog"
        memcap.srun_ok = lambda: False
        assert _with_env({"BENCH_MEMCAP": ""}, memcap.mechanism) == "watchdog"
    finally:
        memcap.systemd_ok, memcap.delegated_parent, memcap.srun_ok = saved


def test_a_rival_under_a_fake_srun() -> None:
    """run_binary under the srun shim: ok with the step's CPU and peak memory; a step oom is
    oom with kill memcap.srun."""
    from benchmarks.tools import rival

    with tempfile.TemporaryDirectory(dir=_exec_dir()) as tmp:
        burn = [sys.executable, "-c", "b = b'\\x01' * 50 * 2**20; sum(range(3 * 10**6))"]
        res = _with_env(_fake_srun(tmp, False), lambda: rival.run_binary(burn, 30))
        assert res["status"] == "ok" and res["memcap"] == "srun", res
        assert res["cpu"] > 0 and res["peak_mb"] >= 50, res
        over = _with_env(_fake_srun(tmp, True), lambda: rival.run_binary(_child(2 * CAP_MB), 30))
        assert over["status"] == "oom" and over["kill"] == "memcap.srun", over


def test_a_lean_check_under_a_fake_srun() -> None:
    """lean.check of a small file under the srun shim gives the watchdog record's fields and
    nonzero CPU and peak memory; with a cap too small it is oom with lean_kill memcap.srun."""
    from benchmarks import lean

    if not (lean.LEAN_DIR / ".lake" / "packages").is_dir():
        return
    src = "theorem memcap_probe_t : 1 + 1 = 2 := rfl\n"
    with tempfile.TemporaryDirectory(dir=_exec_dir()) as tmp:
        path = Path(tmp) / "memcap_probe.lean"
        path.write_text(src, encoding="utf-8")
        ref = _with_env({"BENCH_MEMCAP": "watchdog"}, lambda: lean.check(path, timeout=300))
        res = _with_env(_fake_srun(tmp, False), lambda: lean.check(path, timeout=300))
        assert res["lean_memcap"] == "srun", res
        assert {k: res.get(k) for k in LEAN_FIELDS} == {k: ref.get(k) for k in LEAN_FIELDS}
        assert set(res) == set(ref), set(res) ^ set(ref)
        assert res["lean_cpu"] > 0 and res["lean_peak_mb"] > 0, res
        olean = lean.OLEAN_MB
        lean.OLEAN_MB = 0
        try:
            env = _fake_srun(tmp, True) | {"BENCH_LEAN_BUDGET_MB": "64"}
            low = _with_env(env, lambda: lean.check(path, timeout=300))
        finally:
            lean.OLEAN_MB = olean
        assert low["lean_status"] == "oom" and low["lean_kill"] == "memcap.srun", low


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name, flush=True)
