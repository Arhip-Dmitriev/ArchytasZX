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

"""The results log: one JSON line per finished run, one file per worker under
``results/raw``, appended with flush and fsync and never edited.

A run is keyed by (config hash, equation, version, tool, point, repeat, trace). The reader
reads every log, ignores a truncated or malformed line, and keeps the first record of a key.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

RAW = "raw"


def key(
    cfg_hash: str,
    equation: str,
    version: str,
    tool: str,
    point: tuple[Any, Any] | None,
    repeat: int,
    trace: bool,
) -> str:
    """The run key as one string."""
    n, d = point if point is not None else ("sym", "sym")
    return f"{cfg_hash}|{equation}|{version}|{tool}|n={n}|d={d}|r={repeat}|t={int(trace)}"


class Writer:
    """Appends records to this worker's log file."""

    def __init__(self, results: Path, worker: str) -> None:
        raw = results / RAW
        raw.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname().split(".")[0]
        job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
        task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        self.path = raw / f"{host}-{job}-{task}-{worker}.jsonl"
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        """One record as one line, flushed and fsynced."""
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def lines(results: Path) -> Iterator[dict[str, Any]]:
    """Every well-formed record of every log under ``results/raw``."""
    raw = results / RAW
    if not raw.exists():
        return
    for p in sorted(raw.glob("*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.endswith("\n"):
                    break
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "key" in rec:
                    yield rec


def index(results: Path, cfg_hash: str | None = None) -> dict[str, dict[str, Any]]:
    """The first record of every key, a unit record with ``differs_checked`` preferred (of one
    config hash when given)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in lines(results):
        if cfg_hash is not None and rec.get("config") != cfg_hash:
            continue
        old = out.get(rec["key"])
        if old is None or (
            rec.get("unit") and rec.get("differs_checked") and not old.get("differs_checked")
        ):
            out[rec["key"]] = rec
    return out
