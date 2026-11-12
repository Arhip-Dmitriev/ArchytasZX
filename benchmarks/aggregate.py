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

"""Aggregates from the logs (BENCHMARKS.txt, section 12.1), written to results/aggregate as CSV
and JSON: points, units, counts, causes, breakdown, b3, overhead, calibration, wrong and
harness_errors.

A point's outcome is its first run's; its statistics are over its successful runs; it is
``unstable`` when a later repeat failed and ``noisy`` when it reached the cluster maximum of
runs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks import config, log, stats
from benchmarks.outcomes import NOT_SCORED

B3_ONLY = ("tket",)
STAGES = ("generate", "decide", "certificate", "lean_export", "lean_check", "check")
SIZE_FIELDS = (
    ("decide", "derivation_steps"),
    ("certificate", "inferences"),
    ("certificate", "certificate_bytes"),
    ("lean_export", "lean_lines"),
    ("lean_export", "lean_bytes"),
    ("lean_check", "lean_cpu"),
    ("check", "nodes_after"),
    ("check", "proof_lines"),
    ("check", "proof_bytes"),
)


def n_bin(n: int | None) -> str:
    """The decade of n: 0, 1-9, 10-99, ...; ``none`` without n."""
    if n is None:
        return "none"
    if n == 0:
        return "0"
    k = int(math.log10(n))
    return f"1e{k}"


def d_bin(d: int | None) -> str:
    """The binary octave of d: 1, 2, 3-4, 5-8, ..."""
    if d is None:
        return "none"
    if d <= 2:
        return str(d)
    k = math.ceil(math.log2(d))
    return f"{2 ** (k - 1) + 1}-{2**k}"


def upgrade(rec: dict[str, Any]) -> dict[str, Any]:
    """An ArchytasZX run record from before the certified verdicts, its verdict mapped onto
    the five and its outcome recomputed; any other record unchanged."""
    from benchmarks.tools import common

    if rec.get("tool") != "archytaszx" or "certified" in rec or rec.get("verdict") is None:
        return rec
    stages = rec.get("stages") or {}
    verdict = rec["verdict"]
    kernel = (stages.get("certificate") or {}).get("kernel")
    reason = str((stages.get("decide") or {}).get("reason") or "")
    if verdict == "equal" and kernel == "rejected":
        verdict = "equal_uncertified"
    elif verdict == "unknown" and "not reported unequal:" in reason:
        verdict = "unequal_uncertified"
    status = (stages.get("decide") or {}).get("status", "ok")
    oc = common.verdict_outcome(bool(rec.get("truth")), status, verdict)
    certified = verdict in ("equal", "unequal")
    return {
        **rec,
        "verdict": verdict,
        "certified": certified,
        "outcome": oc,
        "wrong": oc == "wrong",
    }


def _point_key(r: dict[str, Any]) -> tuple[Any, ...]:
    return (r["equation"], r["version"], r["tool"], r.get("n"), r.get("d"))


def flatten(
    spans: Iterable[dict[str, Any]], prefix: str = ""
) -> Iterable[tuple[str, str | None, dict[str, Any]]]:
    """Every span as (path, label, node)."""
    for s in spans:
        path = f"{prefix}/{s['name']}" if prefix else s["name"]
        yield path, s.get("label"), s
        yield from flatten(s.get("children", []), path)


def breakdown(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per rung, per rule and per inference kind: CPU seconds and calls of one run's spans."""
    out: dict[str, dict[str, Any]] = {
        "rung": {},
        "rule": {},
        "match": {},
        "inference": {},
        "lean": {},
        "other": {},
    }
    for stage, st in (rec.get("stages") or {}).items():
        for _path, label, node in flatten(st.get("spans") or []):
            name = node["name"]
            if name.startswith(("decide.", "saturation.")):
                group, key = "rung", name
            elif name == "rewrite.apply":
                group, key = "rule", label or "?"
            elif name == "rewrite.match":
                group, key = "match", label or "?"
            elif name.startswith("kernel."):
                group, key = "inference", name.split(".", 1)[1]
            elif name.startswith("lean."):
                group, key = "lean", f"{name}[{label}]" if label else name
            else:
                group, key = "other", f"{stage}:{name}"
            slot = out[group].setdefault(key, {"cpu": 0.0, "calls": 0})
            slot["cpu"] += node["cpu"]
            slot["calls"] += node["calls"]
    for stage, st in (rec.get("stages") or {}).items():
        for decl in st.get("decl_ms") or []:
            key = f"lean_check:{decl['kind']}"
            slot = out["lean"].setdefault(key, {"cpu": 0.0, "calls": 0})
            slot["cpu"] += decl["ms"] / 1000
            slot["calls"] += 1
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
            )


def _write(out: Path, name: str, rows: list[dict[str, Any]]) -> None:
    _write_csv(out / f"{name}.csv", rows)
    (out / f"{name}.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")


def points_table(runs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (equation, version, tool, point): outcome, flags and per-stage statistics."""
    rc = config.repeats(cfg)
    conf = rc.get("confidence") if rc["mode"] == "adaptive" else None
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r.get("trace") is False and cfg["trace"]["enabled"]:
            continue
        groups[_point_key(r)].append(r)
    rows = []
    for key, rs in groups.items():
        rs.sort(key=lambda r: r["repeat"])
        first = rs[0]
        ok = [r for r in rs if r.get("outcome") == "ok"]
        row: dict[str, Any] = {
            "equation": key[0],
            "version": key[1],
            "tool": key[2],
            "n": key[3],
            "d": key[4],
            "subject": first.get("subject"),
            "category": first.get("category"),
            "truth": first.get("truth"),
            "kind": first.get("kind"),
            "phase": first.get("phase"),
            "outcome": first.get("outcome"),
            "verdict": first.get("verdict"),
            "wrong": first.get("outcome") == "wrong",
            "runs": len(rs),
            "unstable": first.get("outcome") == "ok" and len(ok) < len(rs),
            "noisy": rc["mode"] == "adaptive" and len(rs) >= int(rc["max_runs"]),
            "node": (first.get("machine") or {}).get("node"),
            "cpu_model": (first.get("machine") or {}).get("cpu_model"),
            "calibration": (first.get("calibration") or {}).get("median"),
            "detail": (first.get("detail") or "")[:200],
            "reason": first.get("reason"),
            "stage": first.get("stage"),
            "cause": first.get("cause"),
        }
        for stage in STAGES:
            cpus = [
                r["stages"][stage]["cpu"]
                for r in ok
                if stage in (r.get("stages") or {}) and r["stages"][stage].get("status") == "ok"
            ]
            if not cpus and stage in (first.get("stages") or {}):
                row[f"{stage}_status"] = first["stages"][stage].get("status")
            if cpus:
                s = stats.summary(cpus, conf)
                row |= {f"{stage}_cpu_{k}": v for k, v in s.items()}
                walls = [
                    r["stages"][stage]["wall"]
                    for r in ok
                    if stage in r["stages"] and r["stages"][stage].get("status") == "ok"
                ]
                row[f"{stage}_wall_median"] = stats.summary(walls)["median"]
                peaks = [r["stages"][stage].get("peak_mb") for r in ok if stage in r["stages"]]
                row[f"{stage}_peak_mb"] = max((p for p in peaks if p is not None), default=None)
                row[f"{stage}_status"] = "ok"
                if row["calibration"]:
                    row[f"{stage}_cpu_scaled"] = s["median"] / row["calibration"]
        totals = [
            sum(st.get("cpu", 0) for st in r["stages"].values()) for r in ok if r.get("stages")
        ]
        if totals:
            row |= {
                f"run_cpu_{k}": v for k, v in stats.summary(totals, conf).items() if k != "values"
            }
        src = first.get("stages") or {}
        for stage, field in SIZE_FIELDS:
            if stage in src and src[stage].get(field) is not None:
                row[f"{stage}.{field}"] = src[stage][field]
        if "lean_check" in src:
            row["axiom_class"] = src["lean_check"].get("axiom_class")
        rows.append(row)
    rows.sort(
        key=lambda r: (
            r["equation"],
            r["tool"],
            r["version"],
            r["n"] if r["n"] is not None else -1,
            r["d"] if r["d"] is not None else -1,
        )
    )
    return rows


def counts_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Points proved (S) and refuted (F) per category, version, tool, break kind and n, d bins;
    wrongly accepted F points and unknown points counted beside; harness_error points counted
    and invalid_point points in their own columns and left out of ``points``."""
    acc: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["tool"] in B3_ONLY:
            continue
        key = (
            r["category"],
            r["version"],
            r["tool"],
            "S" if r["truth"] else "F",
            r["kind"],
            n_bin(r["n"]),
            d_bin(r["d"]),
        )
        slot = acc[key]
        oc = r["outcome"]
        if oc not in NOT_SCORED:
            slot["points"] += 1
        if oc == "ok":
            slot["proved" if r["truth"] else "refuted"] += 1
        elif oc == "wrong":
            slot["wrongly_refuted" if r["truth"] else "wrongly_accepted"] += 1
        else:
            slot[oc or "none"] += 1
    out = []
    for k, v in sorted(acc.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append(
            {
                "category": k[0],
                "version": k[1],
                "tool": k[2],
                "set": k[3],
                "kind": k[4],
                "n_bin": k[5],
                "d_bin": k[6],
                **v,
            }
        )
    return out


def harness_errors_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The points whose first run failed in the harness's own preparation, with the reason."""
    return [
        {
            "equation": r["equation"],
            "version": r["version"],
            "tool": r["tool"],
            "n": r["n"],
            "d": r["d"],
            "reason": r.get("reason"),
        }
        for r in rows
        if r["outcome"] == "harness_error"
    ]


def equations_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Equations settled per category, version, tool and set: an equation counts once when any
    of its points settled it."""
    acc: dict[tuple[Any, ...], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["tool"] in B3_ONLY:
            continue
        key = (r["category"], r["version"], r["tool"], "S" if r["truth"] else "F")
        acc[key]["all"].add(r["equation"])
        if r["outcome"] == "ok":
            acc[key]["settled"].add(r["equation"])
        if r["outcome"] == "wrong":
            acc[key]["wrong"].add(r["equation"])
    return [
        {
            "category": k[0],
            "version": k[1],
            "tool": k[2],
            "set": k[3],
            "equations": len(v["all"]),
            "settled": len(v["settled"]),
            "wrong": len(v["wrong"]),
        }
        for k, v in sorted(acc.items())
    ]


def units_table(units: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (equation, tool): K reached, region_exhausted, unsupported reason, the frontier
    (failed points and how) and the largest successful n and d."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r["outcome"] != "ok" or r["version"] == "symbolic":
            continue
        b = best.setdefault((r["equation"], r["tool"]), {"n_max": None, "d_max": None})
        if r["n"] is not None:
            b["n_max"] = max(b["n_max"] or 0, r["n"])
        if r["d"] is not None:
            b["d_max"] = max(b["d_max"] or 0, r["d"])
    out = []
    for u in units:
        b = best.get((u["equation"], u["tool"]), {})
        out.append(
            {
                "equation": u["equation"],
                "tool": u["tool"],
                "subject": u.get("subject"),
                "category": u.get("category"),
                "truth": u.get("truth"),
                "kind": u.get("kind"),
                "K": u.get("K"),
                "successes": u.get("successes"),
                "K_reached": u.get("successes") == u.get("K"),
                "region_exhausted": u.get("region_exhausted"),
                "unsupported": u.get("unsupported"),
                "frontier": u.get("failures"),
                "n_max_ok": b.get("n_max"),
                "d_max_ok": b.get("d_max"),
            }
        )
    return out


def breakdown_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (equation, tool, point, group, key): CPU seconds, calls and mean per call over the
    point's traced successful runs (medians of totals)."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r.get("trace") and r.get("outcome") == "ok":
            groups[
                (r["equation"], r["tool"], r["version"], r.get("n"), r.get("d"), r.get("category"))
            ].append(r)
    out = []
    for key, rs in groups.items():
        per: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        for r in rs:
            for group, items in breakdown(r).items():
                for name, v in items.items():
                    per[(group, name)].append((v["cpu"], v["calls"]))
        for (group, name), vals in per.items():
            cpu = stats.summary([v[0] for v in vals])["median"]
            calls = int(stats.summary([v[1] for v in vals])["median"])
            out.append(
                {
                    "equation": key[0],
                    "tool": key[1],
                    "version": key[2],
                    "n": key[3],
                    "d": key[4],
                    "category": key[5],
                    "group": group,
                    "name": name,
                    "cpu": cpu,
                    "calls": calls,
                    "mean": cpu / calls if calls else None,
                }
            )
    return out


def b3_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """B3 counts per (equation, tool, point): naive, smallest and their ratio, from repeat 0."""
    out = []
    for r in runs:
        b = r.get("b3")
        if not b or r.get("repeat") != 0 or not r.get("trace", True) or "naive" not in b:
            continue
        row = {
            "equation": r["equation"],
            "tool": r["tool"],
            "version": r["version"],
            "n": r.get("n"),
            "d": r.get("d"),
            "category": r.get("category"),
            "ratio": b.get("ratio"),
        }
        row |= {f"naive.{k}": v for k, v in b["naive"].items()}
        row |= {f"smallest.{k}": v for k, v in (b.get("smallest") or {}).items()}
        out.append(row)
    return out


def overhead_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Traced over untraced decide CPU at the sampled points (repeat 0 of each)."""
    on: dict[tuple[Any, ...], float] = {}
    off: dict[tuple[Any, ...], float] = {}
    for r in runs:
        if r.get("repeat") != 0 or r.get("outcome") != "ok":
            continue
        st = (r.get("stages") or {}).get("decide") or (r.get("stages") or {}).get("check")
        if not st:
            continue
        (on if r.get("trace") else off)[_point_key(r)] = st["cpu"]
    out = []
    for k, v in off.items():
        if k in on and v > 0:
            out.append(
                {
                    "equation": k[0],
                    "version": k[1],
                    "tool": k[2],
                    "n": k[3],
                    "d": k[4],
                    "traced_cpu": on[k],
                    "untraced_cpu": v,
                    "ratio": on[k] / v,
                }
            )
    return out


def calibration_table(results: Path) -> list[dict[str, Any]]:
    """The calibration of every job, from the environment records."""
    out = []
    for p in sorted((results / "environment").glob("*.json")):
        env = json.loads(p.read_text())
        m = env.get("machine", {})
        c = env.get("calibration") or {}
        out.append(
            {
                "job": p.stem,
                "node": m.get("node"),
                "cpu_model": m.get("cpu_model"),
                "governor": m.get("governor"),
                "calibration_median": c.get("median"),
                "calibration_runs": c.get("cpu"),
                "git_commit": env.get("git_commit"),
                "git_dirty": env.get("git_dirty"),
            }
        )
    return out


def current(cfg: dict[str, Any], recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records whose ``source`` digest matches today's sources of their instance (runs of
    an edited or removed subject dropped)."""
    from benchmarks import breaks, sources, subject

    sc = cfg["subjects"]
    insts = {i.id: i for i in subject.instances(list(subject.catalogue(sc["catalogue"])))}
    falses = {}
    if cfg["false_set"]["enabled"]:
        fs = config.path(cfg, "results") / "false_set.json"
        if fs.exists():
            for rec in json.loads(fs.read_text(encoding="utf-8")).get("records", []):
                if rec.get("kind") is not None and rec["equation"] in insts:
                    fi = breaks.false_instance(rec, insts)
                    falses[fi.id] = (fi, rec)
    want: dict[str, str] = {}
    for iid, inst in insts.items():
        want[iid] = sources.source_digest(inst, None, sc["catalogue"])
    for fid, (fi, rec) in falses.items():
        want[fid] = sources.source_digest(fi, rec, sc["catalogue"])
    return [r for r in recs if want.get(r.get("equation", "")) == r.get("source")]


def false_records(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The F record of every F equation id (``EQUATION!``) in results/false_set.json."""
    fs = config.path(cfg, "results") / "false_set.json"
    if not cfg["false_set"]["enabled"] or not fs.exists():
        return {}
    return {
        f"{r['equation']}!": r
        for r in json.loads(fs.read_text(encoding="utf-8")).get("records", [])
        if r.get("kind") is not None
    }


def classified(
    recs: list[dict[str, Any]], falses: dict[str, dict[str, Any]], catalogue: str
) -> list[dict[str, Any]]:
    """Every run record given outcomes.classify's fields when it has none; unit records
    unchanged."""
    from benchmarks import outcomes

    insts = outcomes.instances(catalogue)
    return [
        r if r.get("unit") or "cause" in r else outcomes.reclassify_record(r, insts, falses)
        for r in recs
    ]


def causes_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Points per tool, version, set, outcome, stage, cause code and tags."""
    acc: dict[tuple[Any, ...], int] = defaultdict(int)
    for r in rows:
        key = (
            r["tool"],
            r["version"],
            "S" if r["truth"] else "F",
            r["outcome"],
            r.get("stage"),
            r.get("cause"),
            ",".join(r.get("tags") or ()) or None,
        )
        acc[key] += 1
    return [
        {
            "tool": k[0],
            "version": k[1],
            "set": k[2],
            "outcome": k[3],
            "stage": k[4],
            "cause": k[5],
            "tags": k[6],
            "points": v,
        }
        for k, v in sorted(acc.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
    ]


def main(cfg: dict[str, Any]) -> None:
    """Writes every aggregate of the config's logs."""
    results = config.path(cfg, "results")
    h = config.digest(cfg)
    recs = [upgrade(r) for r in current(cfg, list(log.index(results, h).values()))]
    recs = classified(recs, false_records(cfg), cfg["subjects"]["catalogue"])
    excluded = {
        id(r) for r in recs if r.get("outcome") == "invalid_point" or r.get("point_invalid")
    }
    recs = [r for r in recs if id(r) not in excluded]
    runs = [r for r in recs if not r.get("unit")]
    units = [r for r in recs if r.get("unit")]
    out = results / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    rows = points_table(runs, cfg)
    _write(out, "points", rows)
    _write(out, "units", units_table(units, rows))
    _write(out, "counts", counts_table(rows))
    _write(out, "causes", causes_table(rows))
    _write(out, "equations", equations_table(rows))
    _write(out, "breakdown", breakdown_table(runs))
    _write(out, "b3", b3_table(runs))
    _write(out, "overhead", overhead_table(runs))
    _write(out, "calibration", calibration_table(results))
    wrong = [r for r in rows if r["wrong"]]
    _write(out, "wrong", wrong)
    harness = harness_errors_table(rows)
    _write(out, "harness_errors", harness)
    print(
        f"aggregate: {len(runs)} runs, {len(rows)} points, {len(units)} units, "
        f"{len(wrong)} wrong verdicts, {len(harness)} harness errors, {len(excluded)} "
        f"runs at points where the F pair is equal (invalid_point, point_invalid) excluded "
        f"-> {out}"
    )
    for r in rows:
        if r["tool"] == "archytaszx" and r["outcome"] in ("wrong", "wrong (uncertified)"):
            tag = "" if r["wrong"] else " (uncertified)"
            print(
                f"  ENGINE WRONG VERDICT{tag}: {r['equation']} {r['version']} "
                f"n={r['n']} d={r['d']}: {r['verdict']}"
            )
