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

"""Batch-step memory probe, standard library only:

``python -I batchmem_probe.py sample OUT [--every S] [--label L]`` appends one JSON line to OUT
every S seconds (default 5) until killed; ``... once [--label L]`` prints one line;
``... dropcache PATH...`` posix_fadvise(DONTNEED)s every regular file under each PATH.

A line holds: time, label, this process's cgroup v1 memory directory (step) and its parent
(job) with usage, max_usage, limit and memory.stat fields (cache, rss, rss_huge, shmem,
mapped_file, active_file, inactive_file and their total_ forms; v2 memory.current, memory.peak,
anon, file, shmem, file_mapped), the RSS / PSS / anonymous / shared sums over the step's tasks
with the top 5 processes by RSS, /proc/meminfo Shmem, Cached, MemAvailable, and the
filesystem type, size and use of /tmp, /dev/shm, $TMPDIR and the node-local candidates."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

V1 = Path("/sys/fs/cgroup/memory")
V2 = Path("/sys/fs/cgroup")
STAT_KEYS = (
    "cache",
    "rss",
    "rss_huge",
    "shmem",
    "mapped_file",
    "active_file",
    "inactive_file",
    "anon",
    "file",
    "file_mapped",
)
LOCAL = ("/tmp", "/dev/shm", "/local", "/scratch/local", "/lscratch", "/var/tmp", "/localscratch")


def cgroup_dir(proc: str = "/proc/self/cgroup") -> tuple[str, Path | None]:
    """("v1" or "v2", this process's memory cgroup directory), None when not found."""
    try:
        lines = Path(proc).read_text(encoding="ascii").splitlines()
    except OSError:
        return "", None
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            return "v1", V1 / parts[2].lstrip("/")
    for ln in lines:
        if ln.startswith("0::"):
            return "v2", V2 / ln[3:].strip().lstrip("/")
    return "", None


def _int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def parse_stat(text: str) -> dict[str, int]:
    """The STAT_KEYS fields of a memory.stat text, plain and total_ forms."""
    out: dict[str, int] = {}
    for ln in text.splitlines():
        k, _, v = ln.partition(" ")
        base = k.removeprefix("total_")
        if base in STAT_KEYS and v.strip().isdigit():
            out[k] = int(v)
    return out


def cg_record(path: Path) -> dict[str, Any]:
    """Usage, peak, limit and memory.stat fields of the cgroup ``path`` (v1 or v2 names)."""
    rec: dict[str, Any] = {"path": str(path)}
    for name in (
        "memory.usage_in_bytes",
        "memory.max_usage_in_bytes",
        "memory.limit_in_bytes",
        "memory.memsw.max_usage_in_bytes",
        "memory.current",
        "memory.peak",
        "memory.max",
    ):
        v = _int(path / name)
        if v is not None:
            rec[name.removeprefix("memory.")] = v
    try:
        rec["stat"] = parse_stat((path / "memory.stat").read_text(encoding="ascii"))
    except OSError:
        rec["stat"] = None
    return rec


def tasks(path: Path) -> list[int]:
    """The pids in the cgroup ``path`` (cgroup.procs, else tasks)."""
    for name in ("cgroup.procs", "tasks"):
        try:
            text = (path / name).read_text(encoding="ascii")
        except OSError:
            continue
        return sorted({int(x) for x in text.split()})
    return []


def proc_mem(pid: int) -> dict[str, Any] | None:
    """kB fields Rss, Pss, Anonymous, Shared_Clean, Shared_Dirty (smaps_rollup) and the
    command of ``pid``."""
    row: dict[str, Any] = {"pid": pid}
    try:
        for ln in Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="ascii").splitlines():
            k, _, v = ln.partition(":")
            if k in ("Rss", "Pss", "Anonymous", "Shared_Clean", "Shared_Dirty"):
                row[k] = int(v.split()[0])
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError, IndexError):
        return None
    row["cmd"] = raw.replace(b"\0", b" ").decode(errors="replace")[:160]
    return row


def meminfo() -> dict[str, int]:
    """Shmem, Cached, MemAvailable, MemTotal of /proc/meminfo in kB."""
    out: dict[str, int] = {}
    try:
        for ln in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            k, _, v = ln.partition(":")
            if k in ("Shmem", "Cached", "MemAvailable", "MemTotal"):
                out[k] = int(v.split()[0])
    except OSError:
        pass
    return out


def mounts() -> dict[str, str]:
    """Mount point to filesystem type, from /proc/mounts."""
    out: dict[str, str] = {}
    try:
        for ln in Path("/proc/mounts").read_text(encoding="ascii", errors="replace").splitlines():
            f = ln.split()
            if len(f) >= 3:
                out[f[1]] = f[2]
    except OSError:
        pass
    return out


def fs_of(path: str, table: dict[str, str]) -> dict[str, Any] | None:
    """The mount point, type, size and used bytes holding ``path``; None when absent."""
    p = os.path.realpath(path)
    if not os.path.isdir(p):
        return None
    mp = max(
        (m for m in table if p == m or p.startswith(m.rstrip("/") + "/")), key=len, default="/"
    )
    st = os.statvfs(p)
    return {
        "mount": mp,
        "type": table.get(mp, ""),
        "size": st.f_blocks * st.f_frsize,
        "used": (st.f_blocks - st.f_bfree) * st.f_frsize,
    }


def snapshot(label: str = "") -> dict[str, Any]:
    """One probe line (see the module docstring)."""
    ver, path = cgroup_dir()
    rec: dict[str, Any] = {"t": time.time(), "label": label, "cgroup": ver}
    if path is not None:
        rec["step"] = cg_record(path)
        rec["job"] = cg_record(path.parent)
        rows = [r for r in map(proc_mem, tasks(path)) if r is not None]
        rec["procs"] = {
            k: sum(int(r.get(k, 0)) for r in rows) for k in ("Rss", "Pss", "Anonymous")
        } | {"n": len(rows)}
        rec["top"] = sorted(rows, key=lambda r: -int(r.get("Rss", 0)))[:5]
    rec["meminfo"] = meminfo()
    table = mounts()
    places = dict.fromkeys([*LOCAL, os.environ.get("TMPDIR", "")])
    rec["fs"] = {p: fs_of(p, table) for p in places if p}
    return rec


def dropcache(paths: list[str]) -> int:
    """posix_fadvise(DONTNEED) on every regular file under ``paths``; the count of files."""
    n = 0
    for top in paths:
        for root, _dirs, files in os.walk(top):
            for name in files:
                f = os.path.join(root, name)
                try:
                    if os.path.islink(f):
                        continue
                    fd = os.open(f, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return n


def _opt(args: list[str], name: str, default: str) -> str:
    return args[args.index(name) + 1] if name in args else default


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    label = _opt(rest, "--label", "")
    if cmd == "once":
        print(json.dumps(snapshot(label)), flush=True)
        return 0
    if cmd == "dropcache":
        print(json.dumps({"dropcache": dropcache(rest)}), flush=True)
        return 0
    if cmd == "sample" and rest:
        every = float(_opt(rest, "--every", "5"))
        while True:
            with open(rest[0], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(snapshot(label)) + "\n")
            time.sleep(every)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
