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

"""Lean slots of benchmarks.lean: slot count and budget per allocation, one lock pool per job,
the budget kill of a check."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from benchmarks import lean

ROOT = Path(__file__).resolve().parent.parent
HOLD = """
import sys
from pathlib import Path
from benchmarks import lean
lean.SLOT_DIR = Path(sys.argv[1])
fh = lean._slot()
print(Path(fh.name).name, flush=True)
sys.stdin.read()
"""
ALLOC = "import time\nb = b'\\x01' * ({mb} << 20)\nprint('error: none')\ntime.sleep({sleep})\n"


@pytest.mark.parametrize(
    ("total", "jobs", "fixed", "expected"),
    [
        (32768, 8, None, (2, 9728)),
        (65536, 8, None, (5, 10444)),
        (15732, 6, None, (1, 8192)),
        (16384, 16, None, (1, 8192)),
        (131072, 32, None, (5, 18636)),
        (32768, 8, 3, (3, 8192)),
    ],
)
def test_plan(total: float, jobs: int, fixed: int | None, expected: tuple[int, int]) -> None:
    assert lean.plan(total, jobs, fixed) == expected


def test_slots_from_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLURM_MEM_PER_NODE", "BENCH_LEAN_SLOTS", "BENCH_LEAN_BUDGET_MB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert lean.slurm_memory_mb() == 32768
    monkeypatch.setattr(lean, "total_memory_mb", lambda: 32768.0)
    assert (lean.slots(8), lean.budget_mb(8)) == (2, 9728)
    monkeypatch.setenv("BENCH_JOBS", "8")
    assert (lean.slots(), lean.budget_mb()) == (2, 9728)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "65536")
    assert lean.slurm_memory_mb() == 65536
    monkeypatch.setenv("BENCH_LEAN_SLOTS", "1")
    monkeypatch.setenv("BENCH_LEAN_BUDGET_MB", "12000")
    assert (lean.slots(8), lean.budget_mb(8)) == (1, 12000)


def test_cgroup_limits(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    job = v1 / "memory" / "slurm" / "uid_1" / "job_7" / "step_batch"
    job.mkdir(parents=True)
    (job / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
    (job.parent / "memory.limit_in_bytes").write_text(str(32 * 2**30))
    (v1 / "cgroup").write_text("5:cpuset:/slurm\n4:memory:/slurm/uid_1/job_7/step_batch\n")
    assert lean.cgroup_memory_mb(v1, v1 / "cgroup") == 32768
    v2 = tmp_path / "v2"
    scope = v2 / "user.slice" / "run-1.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text("max\n")
    (scope.parent / "memory.max").write_text(str(12 * 2**30))
    (v2 / "cgroup").write_text("0::/user.slice/run-1.scope\n")
    assert lean.cgroup_memory_mb(v2, v2 / "cgroup") == 12288
    (v2 / "none").write_text("0::/elsewhere\n")
    assert lean.cgroup_memory_mb(v2, v2 / "none") is None


def _hold(tmp_path: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), **env},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    return proc


def _release(proc: subprocess.Popen[str]) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait(30)


def test_lock_shared_by_one_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"SLURM_JOB_ID": "77", "SLURM_ARRAY_TASK_ID": "3", "BENCH_LEAN_SLOTS": "2"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(lean, "SLOT_DIR", tmp_path)
    first, second = _hold(tmp_path, env), _hold(tmp_path, env)
    try:
        assert first.stdout is not None and second.stdout is not None
        names = {first.stdout.readline().strip(), second.stdout.readline().strip()}
        assert names == {lean.slot_path(0).name, lean.slot_path(1).name}
        assert lean.slot_path(0).name == "archytaszx-bench-lean-77-3-0.lock"
        got: list[Any] = []
        waiter = threading.Thread(target=lambda: got.append(lean._slot()))
        waiter.start()
        waiter.join(1.5)
        assert waiter.is_alive() and not got
        _release(first)
        waiter.join(10)
        assert got and Path(got[0].name).name in names
        got[0].close()
        other = {**env, "SLURM_JOB_ID": "78"}
        third = _hold(tmp_path, other)
        assert third.stdout is not None
        assert third.stdout.readline().strip() == "archytaszx-bench-lean-78-3-0.lock"
        _release(third)
    finally:
        for p in (first, second):
            if p.poll() is None:
                _release(p)


def test_slot_lock_is_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"SLURM_JOB_ID": "90", "SLURM_ARRAY_TASK_ID": "0", "BENCH_LEAN_SLOTS": "1"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(lean, "SLOT_DIR", tmp_path)
    holder = _hold(tmp_path, env)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip()
        with open(lean.slot_path(0), "w") as fh, pytest.raises(OSError):
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        _release(holder)


def _fake_lake(tmp_path: Path, mb: int, sleep: float) -> str:
    """A ``lake`` stand-in: Python running ``env`` in the working directory, which starts a
    Python child holding ``mb`` MiB."""
    child = tmp_path / "alloc.py"
    child.write_text(ALLOC.format(mb=mb, sleep=sleep))
    (tmp_path / "env").write_text(
        f"import subprocess, sys\nsys.exit(subprocess.call([sys.executable, {str(child)!r}]))\n"
    )
    return sys.executable


@pytest.mark.parametrize(("mb", "status"), [(300, "oom"), (20, "rejected")])
def test_budget_kills_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mb: int, status: str
) -> None:
    monkeypatch.setattr(lean, "SLOT_DIR", tmp_path)
    monkeypatch.setattr(lean, "cgroups", lambda: False)
    monkeypatch.setattr(lean, "LEAN_DIR", tmp_path)
    monkeypatch.setattr(lean, "lake", lambda: _fake_lake(tmp_path, mb, 1.5))
    monkeypatch.setenv("BENCH_LEAN_SLOTS", "1")
    monkeypatch.setenv("BENCH_LEAN_BUDGET_MB", "120")
    src = tmp_path / "t.lean"
    src.write_text("theorem t : True := trivial\n")
    out = lean.check(src, timeout=60)
    assert out["lean_budget_mb"] == 120
    assert out["lean_status"] == status
    assert ("lean_kill" in out) == (status == "oom")
    if status == "oom":
        assert out["lean_wall"] < 1.5


def test_scope_cap_is_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lean, "cgroups", lambda: True)
    monkeypatch.setattr(lean, "total_memory_mb", lambda: 32768.0)
    cmd = lean.scoped(["lean", "x"], 0.85, 9728)
    assert f"MemoryMax={lean.OLEAN_MB + 9728}M" in cmd and cmd[-2:] == ["lean", "x"]
    cmd = lean.scoped(["lean", "x"], 0.25, 9728)
    assert "MemoryMax=8192M" in cmd


PROFILED = r"""
import sys
out, err = sys.stdout.buffer, sys.stderr.buffer
out.write(b"@@DECL theorem t\nelaboration took 2ms\nerror: Tactic ")
out.flush()
err.write(b"parsing took 0.5ms\n\xcc")
err.flush()
out.write(b"`decide` failed \xcc\n@@DECL end _\n")
out.flush()
sys.exit(1)
"""


def test_profiler_stderr_and_bad_bytes_stay_out_of_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check whose Lean writes profiler lines to stderr mid-message and invalid UTF-8 keeps the
    message whole and its declaration time."""
    monkeypatch.setattr(lean, "SLOT_DIR", tmp_path)
    monkeypatch.setattr(lean, "cgroups", lambda: False)
    monkeypatch.setattr(lean, "LEAN_DIR", tmp_path)
    monkeypatch.setattr(lean, "lake", lambda: sys.executable)
    monkeypatch.setenv("BENCH_LEAN_SLOTS", "1")
    monkeypatch.setenv("BENCH_LEAN_BUDGET_MB", "120")
    (tmp_path / "env").write_text(PROFILED)
    src = tmp_path / "t.lean"
    src.write_text("theorem t : True := trivial\n")
    out = lean.check(src, timeout=60)
    assert out["lean_status"] == "rejected"
    assert out["lean_detail"] == "error: Tactic `decide` failed �"
    assert out["decl_ms"] == [
        {"kind": "theorem", "name": "t", "ms": 2.0, "by": {"elaboration": 2.0}}
    ]


def test_budget_kill_records_the_tree_peak_and_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget-killed check reports the Lean tree peak RssAnon (at least the budget) and its CPU
    seconds read from /proc; lake rusage never sees the killed child."""
    monkeypatch.setattr(lean, "SLOT_DIR", tmp_path)
    monkeypatch.setattr(lean, "cgroups", lambda: False)
    monkeypatch.setattr(lean, "LEAN_DIR", tmp_path)
    monkeypatch.setattr(lean, "lake", lambda: _fake_lake(tmp_path, 300, 1.5))
    monkeypatch.setenv("BENCH_LEAN_SLOTS", "1")
    monkeypatch.setenv("BENCH_LEAN_BUDGET_MB", "120")
    src = tmp_path / "t.lean"
    src.write_text("theorem t : True := trivial\n")
    out = lean.check(src, timeout=60)
    assert out["lean_status"] == "oom" and out["lean_kill"] == "budget"
    assert out["lean_peak_anon_mb"] >= 120
    assert out["lean_cpu_tree"] > 0
    assert out["lean_cpu"] == max(out["lean_cpu_rusage"], out["lean_cpu_tree"])


BURN = """
import subprocess, sys
child = (
    "import time\\nb = b'\\\\x01' * (64 << 20)\\nt = time.process_time()\\n"
    "while time.process_time() - t < 0.3: pass\\ntime.sleep(30)\\n"
)
subprocess.Popen([sys.executable, "-c", child]).wait()
"""


def test_watch_sums_the_tree(tmp_path: Path) -> None:
    """watch records the RssAnon and CPU seconds of a process and its child."""
    proc = subprocess.Popen([sys.executable, "-c", BURN], start_new_session=True)
    try:
        done = threading.Event()
        stats: dict[str, float] = {}
        over = lean.watch(proc.pid, 10_000, done, lambda: None, stats)
        deadline = time.monotonic() + 20
        while stats.get("cpu_tree", 0.0) < 0.3 and time.monotonic() < deadline:
            time.sleep(0.1)
        done.set()
        anon, cpu = lean.tree_usage(proc.pid)
        assert not over.is_set()
        assert anon >= 64 and cpu >= 0.3
        assert stats["peak_anon_mb"] >= 64 and stats["cpu_tree"] >= 0.3
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(10)
