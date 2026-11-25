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

"""Per-child memory caps, the mechanism chosen at run time: ``systemd`` (a ``systemd-run --user
--scope`` with MemoryMax), ``cgroup_v2`` (a child cgroup made in a delegated cgroup v2 subtree,
memory.max written, the child moved in before exec), ``srun`` (a Slurm step with --mem running
the child under the memcap_step wrapper, whose stats file gives the child's CPU and peak memory)
or ``watchdog`` (the child's process-tree RssAnon polled and the tree killed past the cap).
BENCH_MEMCAP names one; otherwise srun inside a Slurm job where systemd and cgroup_v2 are not
usable, else the first usable of AUTO."""

from __future__ import annotations

import functools
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CGROUP_ROOT = Path("/sys/fs/cgroup")
STEP = Path(__file__).resolve().with_name("memcap_step.py")
ALL = ("systemd", "cgroup_v2", "srun", "watchdog")
AUTO = ("systemd", "cgroup_v2", "watchdog")
_SRUN_OOM = re.compile(
    r"oom[-_ ]kill|Exceeded (?:step|job) memory limit|out of memory", re.IGNORECASE
)
_COUNT = itertools.count()


def own_cgroup(proc: Path = Path("/proc/self/cgroup")) -> Path | None:
    """This process's cgroup v2 directory, None without a ``0::`` line."""
    try:
        for ln in proc.read_text(encoding="ascii").splitlines():
            if ln.startswith("0::"):
                return CGROUP_ROOT / ln[3:].strip().lstrip("/")
    except OSError:
        return None
    return None


def _write(path: Path, text: str) -> bool:
    try:
        path.write_text(text, encoding="ascii")
    except OSError:
        return False
    return True


def make_cgroup(parent: Path, name: str, limit_mb: float, no_swap: bool = True) -> Path | None:
    """The child cgroup ``parent/name`` with memory.max at ``limit_mb`` MiB, memory.swap.max 0
    (when ``no_swap``) and memory.oom.group 1; None when it cannot be made."""
    path = parent / name
    try:
        path.mkdir()
    except OSError:
        return None
    if not _write(path / "memory.max", str(int(limit_mb * 2**20))):
        remove_cgroup(path)
        return None
    if no_swap:
        _write(path / "memory.swap.max", "0")
    _write(path / "memory.oom.group", "1")
    return path


def remove_cgroup(path: Path) -> None:
    """rmdir of ``path``, retried for up to 2 s while its last processes exit."""
    for _ in range(20):
        try:
            path.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.1)


def join(path: Path) -> Callable[[], None]:
    """A preexec function moving the calling process into the cgroup ``path``."""
    procs = str(path / "cgroup.procs")

    def apply() -> None:
        with open(procs, "w", encoding="ascii") as fh:
            fh.write("0")

    return apply


def chain(*fns: Callable[[], None] | None) -> Callable[[], None]:
    """A preexec function calling each given function in order."""
    todo = [f for f in fns if f is not None]

    def apply() -> None:
        for f in todo:
            f()

    return apply


def oom_kills(path: Path) -> int:
    """The oom_kill counter of the cgroup ``path`` (memory.events), 0 when unreadable."""
    try:
        for ln in (path / "memory.events").read_text(encoding="ascii").splitlines():
            if ln.startswith("oom_kill "):
                return int(ln.split()[1])
    except (OSError, ValueError):
        return 0
    return 0


@functools.cache
def delegated_parent() -> Path | None:
    """A cgroup v2 directory where a child cgroup with memory.max can be made and a process
    moved into it (BENCH_CGROUP_PARENT, else the parent of this process's cgroup, else the
    cgroup itself), None when there is none."""
    own = own_cgroup()
    cands: list[Path] = []
    if os.environ.get("BENCH_CGROUP_PARENT"):
        cands.append(Path(os.environ["BENCH_CGROUP_PARENT"]))
    if own is not None and own != CGROUP_ROOT:
        cands += [own.parent, own]
    for cand in cands:
        path = make_cgroup(cand, f"azx-probe-{os.getpid()}-{next(_COUNT)}", 64)
        if path is None:
            continue
        try:
            ok = subprocess.run(["true"], preexec_fn=join(path), check=False).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        remove_cgroup(path)
        if ok:
            return cand
    return None


@functools.cache
def systemd_ok() -> bool:
    """True when ``systemd-run --user --scope`` can hold a process to a memory cap."""
    if shutil.which("systemd-run") is None:
        return False
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M"]
    try:
        return subprocess.run([*cmd, "true"], capture_output=True, check=False).returncode == 0
    except OSError:
        return False


def srun_ok() -> bool:
    """True inside a Slurm job with ``srun`` on PATH."""
    return bool(os.environ.get("SLURM_JOB_ID")) and shutil.which("srun") is not None


def usable(mech: str) -> bool:
    """Whether ``mech`` can cap a child here."""
    if mech == "systemd":
        return systemd_ok()
    if mech == "cgroup_v2":
        return delegated_parent() is not None
    if mech == "srun":
        return srun_ok()
    return mech == "watchdog"


def mechanism() -> str:
    """BENCH_MEMCAP when usable; else srun inside a Slurm job where systemd and cgroup_v2 are
    not usable; else the first usable mechanism of AUTO."""
    forced = os.environ.get("BENCH_MEMCAP", "")
    if forced in ALL and usable(forced):
        return forced
    if srun_ok() and not usable("systemd") and not usable("cgroup_v2"):
        return "srun"
    return next(m for m in AUTO if usable(m))


def tree_anon_mb(pid: int) -> float:
    """RssAnon in MiB summed over ``pid`` and its descendants."""
    from benchmarks import lean

    return lean.tree_anon_mb(pid)


class Cap:
    """A memory cap of ``limit_mb`` MiB on one child process tree by ``mech`` (default
    :func:`mechanism`; cgroup_v2 falls back to watchdog when the cgroup cannot be made)."""

    def __init__(
        self, limit_mb: float, mech: str | None = None, tag: str = "child", no_swap: bool = True
    ) -> None:
        self.limit_mb = limit_mb
        self.mech = mech or mechanism()
        self.no_swap = no_swap
        self.path: Path | None = None
        self.stats: Path | None = None
        self.fired = threading.Event()
        self.done = threading.Event()
        if self.mech == "cgroup_v2":
            parent = delegated_parent()
            name = f"azx-{tag}-{os.getpid()}-{next(_COUNT)}"
            self.path = make_cgroup(parent, name, limit_mb, no_swap) if parent else None
            if self.path is None:
                self.mech = "watchdog"
        if self.mech == "srun":
            fd, name = tempfile.mkstemp(prefix=f"azx-{tag}-", suffix=".json")
            os.close(fd)
            self.stats = Path(name)

    def command(self, cmd: list[str]) -> list[str]:
        """``cmd`` wrapped for systemd, or for srun under the memcap_step wrapper, else
        ``cmd``."""
        mb = int(self.limit_mb)
        if self.mech == "systemd":
            swap = ["-p", "MemorySwapMax=0"] if self.no_swap else []
            head = ["systemd-run", "--user", "--scope", "--quiet", "--collect"]
            return [*head, "-p", f"MemoryMax={mb}M", *swap, *cmd]
        if self.mech == "srun":
            head = ["srun", "--overlap", "--quiet", "-n1", "-N1", f"--mem={mb}M"]
            return [*head, sys.executable, "-I", str(STEP), str(self.stats), *cmd]
        return cmd

    def preexec(self) -> Callable[[], None] | None:
        """The cgroup join of cgroup_v2, else None."""
        return join(self.path) if self.path is not None else None

    def watch(self, pid: int, kill: Callable[[], None], every: float = 0.25) -> None:
        """Under watchdog, a daemon thread calling ``kill`` once the tree RssAnon of ``pid``
        passes the cap, polled every ``every`` seconds until :meth:`close`."""
        if self.mech != "watchdog":
            return

        def loop() -> None:
            while not self.done.wait(every):
                if tree_anon_mb(pid) > self.limit_mb:
                    self.fired.set()
                    kill()
                    return

        threading.Thread(target=loop, daemon=True).start()

    def oom(self, code: int | None = None, text: str = "") -> bool:
        """Whether the cap killed the child: the watchdog fired, the cgroup's oom_kill rose, or
        (srun) a nonzero exit reporting a memory kill."""
        if self.fired.is_set():
            return True
        if self.path is not None and oom_kills(self.path) > 0:
            return True
        return self.mech == "srun" and code not in (0, None) and bool(_SRUN_OOM.search(text))

    def kill_group(self, pgid: int) -> None:
        """SIGKILL to the process group ``pgid``; under srun SIGTERM first (srun passes it to
        the step) and SIGKILL 3 s later."""

        def sig(n: int) -> None:
            try:
                os.killpg(pgid, n)
            except OSError:
                pass

        if self.mech != "srun":
            sig(signal.SIGKILL)
            return
        sig(signal.SIGTERM)
        threading.Timer(3.0, sig, args=(signal.SIGKILL,)).start()

    def usage(self) -> dict[str, float] | None:
        """Under srun, the wrapper's stats (code, cpu, maxrss_kb, peak_anon_mb, cpu_tree);
        None when absent."""
        if self.stats is None:
            return None
        try:
            row = json.loads(self.stats.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return row if isinstance(row, dict) else None

    def close(self) -> None:
        """Stops the watchdog and removes the cgroup."""
        self.done.set()
        if self.path is not None:
            remove_cgroup(self.path)

    def discard(self) -> None:
        """Removes the srun stats file."""
        if self.stats is not None:
            self.stats.unlink(missing_ok=True)


def worker_mechanism() -> str:
    """The mechanism capping the calling worker (BENCH_MEMCAP_WORKER, else rlimit_as)."""
    return os.environ.get("BENCH_MEMCAP_WORKER") or "rlimit_as"


def child_cap(tag: str) -> Cap | None:
    """A cap at the worker's BENCH_MEMORY_MB for a rival child, None when the worker's own
    cgroup holds its children."""
    mb = os.environ.get("BENCH_MEMORY_MB")
    return Cap(float(mb), tag=tag) if mb else None


def run(
    cmd: list[str],
    cap: Cap | None,
    timeout: float,
    preexec: Callable[[], None] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """``cmd`` run to its end in a session of its own under ``cap`` (text output captured):
    (the completed process, whether the cap killed it); TimeoutExpired after ``timeout``
    wall seconds, the process group killed."""
    full = cap.command(cmd) if cap is not None else cmd
    proc = subprocess.Popen(
        full,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
        preexec_fn=chain(cap.preexec() if cap else None, preexec),  # noqa: PLW1509
    )

    def kill() -> None:
        if cap is not None:
            cap.kill_group(proc.pid)
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass

    try:
        if cap is not None:
            cap.watch(proc.pid, kill)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill()
            proc.communicate()
            if cap is not None:
                cap.discard()
            raise
        done = subprocess.CompletedProcess(full, proc.returncode, out, err)
        return done, cap is not None and cap.oom(proc.returncode, (err or "") + (out or ""))
    finally:
        if cap is not None:
            cap.close()


def describe() -> dict[str, Any]:
    """Each mechanism's usability and the chosen one."""
    return {"usable": {m: usable(m) for m in ALL}, "chosen": mechanism()}
