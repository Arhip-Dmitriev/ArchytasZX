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

"""Logged run records reclassified by benchmarks.outcomes, without re-running:
``python -m benchmarks.reclassify RAW_DIR FALSE_SET OUT_JSONL [--journal]``.

Reads every RAW_DIR/*.jsonl (never writes there), gives each run record the classifier's
fields (F records from FALSE_SET for invalid_point), writes all records to OUT_JSONL and prints
the outcome counts per tool before and after. ``--journal`` marks a worker that died with exit
143 as ``journal_oom_kill`` when a systemd user-scope "Failed with result 'oom-kill'" journal
entry lies from 10 s before to 2 s after its record's ``finished`` (the record predates the
worker's own oom_kill report).
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks import outcomes

WINDOW = 10.0
AFTER = 2.0


def journal_oom_times() -> list[float]:
    """Sorted times of every user-scope OOM-kill failure in the journal."""
    out = subprocess.run(
        [
            "journalctl",
            "--user",
            "-o",
            "short-unix",
            "--no-pager",
            "-g",
            "Failed with result 'oom-kill'",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    times = []
    for ln in out.splitlines():
        head = ln.split(" ", 1)[0]
        if ".scope:" in ln and head.replace(".", "", 1).isdigit():
            times.append(float(head))
    return sorted(times)


def oom_near(times: list[float], t: float) -> bool:
    """Whether an OOM-kill time lies in [t - WINDOW, t + AFTER]."""
    i = bisect.bisect_left(times, t - WINDOW)
    return i < len(times) and times[i] <= t + AFTER


def load(raw: Path) -> list[dict[str, Any]]:
    """Every record of every RAW/*.jsonl, in file order."""
    recs: list[dict[str, Any]] = []
    for f in sorted(raw.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            recs.extend(json.loads(ln) for ln in fh if ln.strip())
    return recs


def reclassify(
    recs: list[dict[str, Any]], falses: dict[str, dict[str, Any]], times: list[float] | None
) -> list[dict[str, Any]]:
    """Every run record classified afresh from its logged fields; unit records unchanged."""
    insts = outcomes.instances()
    out = []
    for r in recs:
        if r.get("unit") or "tool" not in r:
            out.append(r)
            continue
        base = {**r, "outcome": r.get("raw_outcome", r.get("outcome"))}
        for k in ("stage", "cause", "cause_message", "raw_outcome", "point_invalid"):
            base.pop(k, None)
        if (
            times is not None
            and base.get("exit") == 143
            and base.get("kill") is None
            and base.get("outcome") == "error"
            and oom_near(times, float(base.get("finished") or 0))
        ):
            base["journal_oom_kill"] = True
        out.append(outcomes.reclassify_record(base, insts, falses))
    return out


def counts(recs: list[dict[str, Any]], field: str = "outcome") -> dict[str, Counter[str]]:
    """Per tool, the count of each value of ``field`` over run records."""
    acc: dict[str, Counter[str]] = {}
    for r in recs:
        if r.get("unit") or "tool" not in r:
            continue
        acc.setdefault(r["tool"], Counter())[str(r.get(field))] += 1
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.reclassify")
    ap.add_argument("raw", type=Path)
    ap.add_argument("false_set", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--journal", action="store_true")
    a = ap.parse_args()
    if a.out.resolve().parent == a.raw.resolve():
        raise SystemExit("OUT_JSONL must lie outside RAW_DIR")
    recs = load(a.raw)
    data = json.loads(a.false_set.read_text(encoding="utf-8"))
    falses = {f"{r['equation']}!": r for r in data.get("records", []) if r.get("kind")}
    new = reclassify(recs, falses, journal_oom_times() if a.journal else None)
    with a.out.open("w", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps(r, default=str) + "\n")
    before = counts(recs)
    after = counts(new)
    names = sorted(
        set(outcomes.OUTCOMES) | {k for c in (*before.values(), *after.values()) for k in c}
    )
    for tool in sorted(after):
        b, n = before.get(tool, Counter()), after[tool]
        cells = [f"{k}={b[k]}->{n[k]}" for k in names if b[k] or n[k]]
        print(f"{tool}: " + ", ".join(cells))


if __name__ == "__main__":
    main()
