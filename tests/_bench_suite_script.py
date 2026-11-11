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

"""Tests of the benchmark suite, ``python -m tests._bench_suite_script [false] [schedule]
[rivals] [stats] [resume] [trace] [--jobs J]`` (all when none is named; default 2 workers,
each capped at 6 GiB of address space; BENCHMARKS.txt, section 13):

- false: every F record differs at its stored witness and is equal wherever its break kind
  says; the boxed beyond_N gadget equals its closed form at N = 1, 2, 3 (its equal-where check:
  the gadget is 1 below N + 1 and the S equation holds);
- rivals: the shared Quantomatic red_green rule set (each rule's sides agree up to a scalar under
  the oracle), then every installed rival on every S equation at n <= 3 (d = 2, and d = 3 for
  Cirq) gives no wrong verdict, and Qiskit and Cirq reproduce the specification tensor exactly;
- schedule: the point schedule of the runner on simulated outcomes: same seed same points,
  ladder steps distinct and in the domain, phase 2 points in the valid region, counts add up;
- stats: the median interval against scipy's binomial quantiles;
- resume: a runner killed at random moments and resumed gives the uninterrupted key set and no
  partial record; a second complete run adds nothing;
- trace: tracing off gives the verdicts of tracing on; span trees nest.

Prints one line per failure and a summary per test; exits 1 on any failure.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-8


WORKER_MEMORY = 6 << 30


def _limit_memory() -> None:
    """Caps a pool worker's address space at WORKER_MEMORY bytes (JVM rivals included)."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY, WORKER_MEMORY))


# -- false -----------------------------------------------------------------------------------


def _false_job(rec: dict[str, Any]) -> list[str]:
    from archytaszx.semantics import check
    from benchmarks import breaks, config, subject
    from benchmarks.differs import doubling_mismatch

    cfg = config.load()
    insts = {i.id: i for i in subject.instances(breaks._selected(cfg))}
    fi = breaks.false_instance(rec, insts)
    tag = f"{rec['equation']}! [{rec['kind']}: {rec['label']}]"
    out: list[str] = []
    w = rec["witness"]
    try:
        left, right = breaks._pair(fi, w["n"], w["d"])
        mixed = not (left.is_pure() and right.is_pure())
        if w.get("method") == "factored":
            g = breaks.gadget_value(rec["params"], w["n"])
            g = abs(g) ** 2 if mixed else g
            e = breaks._symbolic_entry(right, tuple(w["outs"]), tuple(w["ins"]), mixed)
            if abs(e) <= TOL or abs(g - 1) <= TOL:
                out.append(f"{tag}: factored witness does not differ (e={e}, g={g})")
        elif w.get("interface"):
            lo = breaks._boundary_dims(breaks._pure(left, mixed))
            ro = breaks._boundary_dims(breaks._pure(right, mixed))
            if lo == ro or doubling_mismatch(w["interface"]):
                out.append(f"{tag}: interfaces agree at the witness")
        else:
            pl, pr = breaks._pure(left, mixed), breaks._pure(right, mixed)
            x = _entry(left, pl, w["outs"], w["ins"], mixed)
            y = _entry(right, pr, w["outs"], w["ins"], mixed)
            if abs(x - y) <= TOL:
                out.append(f"{tag}: equal at the stored witness")
    except Exception as exc:  # noqa: BLE001
        out.append(f"{tag}: witness check error {type(exc).__name__}: {str(exc)[:200]}")
    ew = rec["equal_where"]
    pts: list[tuple[Any, int]] = []
    mode = check.EqualityMode.EXACT
    if ew.get("grid") == "oracle":
        pts = breaks.oracle_grid(fi)
    elif ew.get("d_prime"):
        pts = [p for p in breaks.oracle_grid(fi) if subject.is_prime(p[1])]
        pts += [(p[0], 5) for p in pts[:1] if fi.domain.has_d(5)]
    elif "d_max" in ew:
        pts = [p for p in breaks.oracle_grid(fi) if p[1] <= ew["d_max"]]
    elif "n_max" in ew and "index" in rec["params"]:
        pts = [p for p in breaks.oracle_grid(fi) if p[0] is None or p[0] <= ew["n_max"]]
    elif ew.get("up_to_global_phase"):
        pts = breaks.oracle_grid(fi)[:4]
        mode = check.EqualityMode.UP_TO_GLOBAL_PHASE
    elif "n_gt" in ew:
        pts = [p for p in breaks.oracle_grid(fi) if p[0] is not None and p[0] > ew["n_gt"]]
    for n, d in pts:
        try:
            left, right = breaks._pair(fi, n, d)
            c = check.compare(left, right, {}, mode=mode)
            if not c.matched:
                out.append(f"{tag}: not equal at n={n} d={d} though {ew} says so: {c.reason[:120]}")
        except Exception as exc:  # noqa: BLE001
            skip = "SKIP " if type(exc).__name__ in ("ContractSizeError", "MemoryError") else ""
            out.append(f"{skip}{tag}: equal-where check error at n={n} d={d}: {str(exc)[:160]}")
    if "input_zero" in ew or "outputs_not_all" in ew:
        out += _slice_check(fi, tag, ew)
    return out


def _entry(raw: Any, pure: Any, outs: list[int], ins: list[int], mixed: bool = False) -> complex:
    from benchmarks import breaks

    arity = max((len(n.inputs) + len(n.outputs) for n in pure.nodes.values()), default=0)
    lo, li = breaks._boundary_dims(pure)
    if math.prod(lo) * math.prod(li) <= breaks.DENSE_LIMIT:
        from archytaszx.semantics import check

        t = np.asarray(check.contract(pure).tensor)
        return complex(t[tuple(outs) + tuple(ins)]) if t.ndim else complex(t)
    if arity <= 12:
        return breaks.plugged(pure, tuple(outs), tuple(ins))
    return breaks._symbolic_entry(raw, tuple(outs), tuple(ins), mixed)


def _slice_check(fi: Any, tag: str, ew: dict[str, Any]) -> list[str]:
    """state_not_map: equal on input 0 of the broken wire; single_entry: equal wherever some
    output is not 1."""

    from benchmarks import breaks

    out = []
    for n, d in [p for p in breaks.oracle_grid(fi) if p[1] >= 2][:4]:
        try:
            out += _slice_one(fi, tag, ew, n, d)
        except Exception as exc:  # noqa: BLE001
            skip = "SKIP " if type(exc).__name__ in ("ContractSizeError", "MemoryError") else ""
            out.append(f"{skip}{tag}: slice check error at n={n} d={d}: {str(exc)[:160]}")
    return out


def _slice_one(fi: Any, tag: str, ew: dict[str, Any], n: Any, d: int) -> list[str]:
    from archytaszx.semantics import check
    from benchmarks import breaks, subject

    out: list[str] = []
    if True:
        left, right = breaks._pair(fi, n, d)
        pl, pr = breaks._pure(left), breaks._pure(right)
        if not left.is_pure() or not right.is_pure():
            return out
        lo, li = breaks._boundary_dims(pl)
        a = np.asarray(check.contract(pl).tensor).reshape([*lo, *li])
        b = np.asarray(check.contract(pr).tensor).reshape([*lo, *li])
        if "input_zero" in ew:
            c = subject.circuit_at(fi, n, d)
            ins = [q for q in breaks._inputs(c)]
            k = ins.index(ew["input_zero"]) if ew["input_zero"] in ins else None
            if k is None or c.count(ew["input_zero"]) is not None:
                return out
            sl = (slice(None),) * len(lo) + tuple(
                0 if j == k else slice(None) for j in range(len(li))
            )
            if not np.allclose(a[sl], b[sl], atol=TOL):
                out.append(f"{tag}: differs on input 0 at n={n} d={d}")
        else:
            mask = np.ones(a.shape, dtype=bool)
            idx = tuple(1 for _ in lo)
            if lo and all(v > 1 for v in lo):
                mask[idx] = False
            if not np.allclose(a[mask], b[mask], atol=TOL):
                out.append(f"{tag}: differs away from the all-ones outputs at n={n} d={d}")
    return out


def test_false(jobs: int) -> list[str]:
    from archytaszx.semantics import check
    from benchmarks import breaks, config

    cfg = config.load()
    recs = breaks.generate(cfg, config.path(cfg, "results") / "false_set.json", jobs)
    fails: list[str] = []
    for big in (1, 2, 3):
        for n in range(big + 4):
            v = complex(
                np.asarray(check.score(breaks.gadget_diagram({"N": big}, n), {}).tensor).reshape(
                    -1
                )[0]
            )
            if abs(v - breaks.gadget_value({"N": big}, n)) > TOL:
                fails.append(f"gadget N={big} n={n}: {v} != closed form")
    for r in recs:
        if r.get("kind") is None:
            reason = r.get("reason")
            skip = "SKIP " if reason in ("time budget", "oracle size") else ""
            fails.append(f"{skip}{r['equation']}: no break kind applies ({reason or 'none'})")
    with ProcessPoolExecutor(jobs, initializer=_limit_memory) as pool:
        for res in pool.map(_false_job, [r for r in recs if r.get("kind")], chunksize=1):
            fails += res
    return fails


# -- rivals ----------------------------------------------------------------------------------


def _rival_job(job: tuple[str, str, Any, int]) -> list[str]:
    from benchmarks import config, subject
    from benchmarks.tools import adapter

    tool, eq, n, d = job
    cfg = config.load(sets=["stages.timeout_seconds=120"])
    inst = {i.id: i for i in subject.instances(list(subject.catalogue()))}[eq]
    try:
        r = adapter(tool).run(inst, {"point": [n, d], "config": cfg})
    except Exception as exc:  # noqa: BLE001
        return [f"{tool} {eq} n={n} d={d}: adapter error {type(exc).__name__}: {str(exc)[:160]}"]
    tag = f"{tool} {eq} n={n} d={d}"
    if r.get("outcome") in ("wrong", "error"):
        return [f"{tag}: {r.get('outcome')} ({_why(r)})"]
    if tool in ("qiskit", "cirq") and r.get("outcome") == "ok" and r.get("verdict") != "equal":
        return [f"{tag}: simulation differs from the specification tensor"]
    return []


def _rival_isolated(job: tuple[str, str, Any, int]) -> list[str]:
    """_rival_job in a child interpreter under the worker memory cap; a child killed by a
    signal or an abort is a SKIP line."""
    p = subprocess.run(
        [sys.executable, "-m", "tests._bench_suite_script", "--rival-one", json.dumps(job)],
        capture_output=True,
        text=True,
        preexec_fn=_limit_memory,
        check=False,
    )
    tool, eq, n, d = job
    try:
        return list(json.loads(p.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError):
        why = (p.stderr.strip().splitlines() or ["no output"])[0][:120]
        return [f"SKIP {tool} {eq} n={n} d={d}: run died ({p.returncode}): {why}"]


def _why(r: dict[str, Any]) -> str:
    """The detail of a run record, last 160 characters."""
    return (r.get("detail") or r.get("stages", {}).get("check", {}).get("detail") or "")[-160:]


def test_quanto_rules() -> list[str]:
    """Every rule of the shared red_green rule set: both sides agree up to a nonzero scalar under
    the oracle, !-boxes at 0 and 1 copies, alpha and beta at 1/3 and 1/7 (units of pi)."""
    from fractions import Fraction

    from archytaszx.semantics import check
    from benchmarks import stages
    from benchmarks.tools import quantomatic, rival

    path = quantomatic.rules()
    if not path.exists():
        path = rival.VENVS / "src" / "quantomatic" / "rulesets" / "red_green" / "default.rules"
    if not path.exists():
        return ["the red_green rule set is not present"]
    env = {"alpha": Fraction(1, 3), "beta": Fraction(1, 7)}
    fails = []
    for name, rule in json.loads(path.read_text())["rules"].items():
        for keep in (False, True):
            try:
                tens = []
                for side in ("lhs", "rhs"):
                    body, order = quantomatic.rule_side_dsl(rule[side], keep, env)
                    if all(ln.startswith("scalar") for ln in body.splitlines()):
                        import sympy as sp

                        expr = " * ".join(ln[len("scalar ") :] for ln in body.splitlines()) or "1"
                        tens.append(np.array([complex(sp.N(sp.sympify(expr)))]))
                        continue
                    run_ = stages.new_run()
                    out = stages.execute(
                        run_, "load\n" + "\n".join("    " + ln for ln in body.splitlines())
                    )
                    if not out.ok:
                        raise RuntimeError(stages.text_of(out)[:160])
                    t = np.asarray(
                        check.contract(check.instantiate(run_.session.current, {})).tensor
                    )
                    perm = [order.index(w) for w in sorted(order)]
                    tens.append(np.transpose(t, perm).reshape(-1) if perm else t.reshape(-1))
                a, b = tens
                k = int(np.argmax(np.abs(b)))
                if abs(b[k]) < TOL:
                    ok = bool(np.allclose(a, 0, atol=TOL))
                else:
                    lam = a[k] / b[k]
                    ok = abs(lam) > TOL and bool(np.allclose(a, lam * b, atol=1e-7))
                if not ok:
                    fails.append(
                        f"quantomatic rule {name} (boxes {'kept' if keep else 'dropped'}): "
                        "sides differ"
                    )
            except rival.Unsupported as exc:
                print(f"quantomatic rule {name}: skipped ({exc})", flush=True)
            except Exception as exc:  # noqa: BLE001
                fails.append(f"quantomatic rule {name}: {type(exc).__name__}: {str(exc)[:160]}")
    return fails


def test_rivals(jobs: int) -> list[str]:
    from benchmarks import subject
    from benchmarks.tools import NAMES, adapter

    work = []
    for inst in subject.instances(list(subject.catalogue())):
        for tool in NAMES[1:]:
            if adapter(tool).supports(inst, None) is not None:
                continue
            ns = (
                [n for n in range(4) if inst.domain.has_n(n)][:3]
                if inst.subject.size_parameter
                else [None]
            )
            ds = [2] if tool != "cirq" else [d for d in (2, 3) if inst.domain.has_d(d)]
            if not inst.subject.general_d:
                ds = [int(inst.subject.dimension)]
            work += [(tool, inst.id, n, d) for n in ns for d in ds]
    fails: list[str] = test_quanto_rules()
    with ThreadPoolExecutor(jobs) as pool:
        for res in pool.map(_rival_isolated, work):
            fails += res
    print(f"rivals: {len(work)} runs", flush=True)
    return fails


# -- schedule --------------------------------------------------------------------------------


class _FakeRun:
    """A UnitRun whose runs succeed below a fake frontier."""

    def __init__(
        self, inst: Any, cfg: dict[str, Any], frontier: tuple[int, int], tool: str = "archytaszx"
    ) -> None:
        from benchmarks import runner

        self.ctx = runner.Context(cfg, "test", Path(tempfile.mkdtemp()), {}, {}, None, 0, False)
        self.unit = runner.UnitRun(runner.Unit(inst.id, tool), inst, self.ctx)
        self.frontier = frontier
        self.calls: list[tuple[Any, int, str]] = []
        self.unit.run = self.run  # type: ignore[method-assign]

    def run(self, point: Any, repeat: int, trace: bool, phase: str) -> dict[str, Any]:
        self.calls.append((point, repeat, phase))
        if point is None:
            return {"outcome": "ok", "stages": {"decide": {"cpu": 1.0}}}
        n, d = point
        ok = (n is None or n < self.frontier[0]) and d < self.frontier[1]
        return {
            "outcome": "ok" if ok else "timeout",
            "stages": {"decide": {"cpu": 1.0 + random.random() * 1e-3}},
        }

    def execute(self) -> dict[str, Any]:
        recs: list[dict[str, Any]] = []
        self.unit._finish = lambda key, fields: recs.append(fields)  # type: ignore[method-assign]
        self.unit.execute()
        return recs[0]


def test_schedule() -> list[str]:
    from benchmarks import config, points, subject

    cfg = config.load(sets=["points.K=8", "trace.overhead_sample=0"])
    fails = []
    insts = list(subject.instances(list(subject.catalogue())))
    for inst in insts:
        ax = points.axes(inst, cfg)
        lad = points.ladder(ax, cfg)
        if len(set(lad)) != len(lad):
            fails.append(f"{inst.id}: ladder repeats a point")
        if any(not ax.contains(p) for p in lad):
            fails.append(f"{inst.id}: ladder leaves the domain")
        if lad != points.ladder(ax, cfg):
            fails.append(f"{inst.id}: ladder not deterministic")
        s1 = points.Phase2.of(ax, cfg, inst.id, "t")
        s2 = points.Phase2.of(ax, cfg, inst.id, "t")
        if [s1.draw() for _ in range(20)] != [s2.draw() for _ in range(20)]:
            fails.append(f"{inst.id}: phase 2 draws differ under one seed")
        for frontier in ((5, 3), (10**9, 10**9), (1, 2)):
            fake = _FakeRun(inst, cfg, frontier)
            fields = fake.execute()
            done = [c for c in fake.calls if c[0] is not None and c[1] == 0]
            ok = [
                c
                for c in done
                if (c[0][0] is None or c[0][0] < frontier[0]) and c[0][1] < frontier[1]
            ]
            if fields["successes"] != len(ok):
                fails.append(
                    f"{inst.id} frontier {frontier}: successes {fields['successes']} != {len(ok)}"
                )
            size = points.domain_size(ax, 8)
            if fields["successes"] > 8:
                fails.append(f"{inst.id}: more than K successes")
            if fields["successes"] < 8 and size >= 8 and not fields.get("region_exhausted"):
                fails.append(
                    f"{inst.id} frontier {frontier}: fewer than K without region_exhausted"
                )
            failed: list[Any] = []
            for p, _r, ph in done:
                if ph == "random" and points.dominated(p, failed):
                    fails.append(f"{inst.id}: phase 2 point {p} in the failed region")
                if not ((p[0] is None or p[0] < frontier[0]) and p[1] < frontier[1]):
                    failed.append(p)
            reps = [c for c in fake.calls if c[1] > 0]
            if any(c[1] >= 3 for c in reps):
                fails.append(f"{inst.id}: more than 3 laptop runs")
    return fails


# -- stats -----------------------------------------------------------------------------------


def test_stats() -> list[str]:
    from scipy import stats as sst

    from benchmarks import stats

    fails = []
    for conf in (0.95, 0.99):
        for m in range(1, 60):
            ranks = stats.median_ci_ranks(m, conf)
            best = None
            for j in range(1, m // 2 + 1):
                cover = 1 - 2 * sst.binom.cdf(j - 1, m, 0.5)
                if cover >= conf - 1e-12:
                    best = j
            ref = None if best is None else (best, m + 1 - best)
            if ranks != ref:
                fails.append(f"m={m} conf={conf}: {ranks} != {ref}")
    if stats.min_runs(0.99) != 8:
        fails.append(f"min runs at 99% is {stats.min_runs(0.99)}, not 8")
    vals = [float(v) for v in range(1, 21)]
    if stats.median_ci(vals, 0.99) != (4.0, 17.0):
        fails.append(f"CI of 1..20 at 99% is {stats.median_ci(vals, 0.99)}, not (4, 17)")
    return fails


# -- resume ----------------------------------------------------------------------------------


def _resume_cmd(res: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks",
        "run",
        "--set",
        'tools.enabled=["archytaszx"]',
        "--set",
        "points.K=3",
        "--set",
        'subjects.include=["ghz", "bell_state", "plus_state"]',
        "--set",
        'false_set.kinds=["global_phase", "wrong_phase", "extra_gate"]',
        "--set",
        "stages.lean_check=false",
        "--set",
        "stages.timeout_seconds=600",
        "--set",
        "platform.jobs=3",
        "--set",
        "repeats.laptop.runs=2",
        "--set",
        f"output.results={res}",
    ]


def _keys(res: Path) -> tuple[set[str], int, list[str]]:
    from benchmarks import log

    bad = []
    count = 0
    for p in (res / "raw").glob("*.jsonl"):
        for ln in p.read_text().splitlines(keepends=True):
            count += 1
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                bad.append(f"{p.name}: malformed line")
                continue
            if not rec.get("unit") and "outcome" not in rec:
                bad.append(f"{p.name}: record without outcome")
    return set(log.index(res).keys()), count, bad


def test_resume() -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    tmp = Path(tempfile.mkdtemp(prefix="bench-resume-"))
    ref, cut = tmp / "ref", tmp / "cut"
    fails = []
    subprocess.run(_resume_cmd(ref), cwd=ROOT, env=env, capture_output=True, check=False)
    rng = random.Random(7)
    for k in range(6):
        p = subprocess.Popen(
            _resume_cmd(cut),
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(rng.uniform(4, 25))
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL if k % 2 else signal.SIGINT)
        p.wait()
    subprocess.run(_resume_cmd(cut), cwd=ROOT, env=env, capture_output=True, check=False)
    kr, _, bad_r = _keys(ref)
    kc, lines, bad_c = _keys(cut)
    fails += bad_r + bad_c
    if kr != kc:
        fails.append(
            f"key sets differ: {len(kr - kc)} only uninterrupted, {len(kc - kr)} only resumed"
        )
    subprocess.run(_resume_cmd(cut), cwd=ROOT, env=env, capture_output=True, check=False)
    _, lines2, _ = _keys(cut)
    if lines2 != lines:
        fails.append(f"a second complete run added {lines2 - lines} lines")
    shutil.rmtree(tmp, ignore_errors=True)
    return fails


# -- trace -----------------------------------------------------------------------------------

_TRACE_PROBE = """
import json, sys
from benchmarks import subject as S, stages
out = {}
for inst in S.instances(list(S.catalogue())):
    if inst.subject.id not in ("ghz", "deutsch_jozsa", "teleportation", "repetition_code"):
        continue
    for nv, dv in ((None, None), (3, 3)):
        try:
            sc = S.script(inst, nv, dv)
        except Exception:
            continue
        r = stages.new_run()
        res = stages.timed(lambda: stages.decide_stage(r, sc.text), 120)
        out[f"{inst.id} {nv} {dv}"] = {
            "verdict": res.get("verdict"), "method": res.get("method"), "spans": res.get("spans")
        }
print(json.dumps(out))
"""


def _nested(spans: list[dict[str, Any]], parent_cpu: float | None, where: str) -> list[str]:
    out = []
    total = 0.0
    for s in spans:
        if s["calls"] < 1 or s["cpu"] < 0:
            out.append(f"{where}/{s['name']}: bad totals")
        total += s["cpu"]
        out += _nested(s.get("children", []), s["cpu"], f"{where}/{s['name']}")
    if parent_cpu is not None and total > parent_cpu + 1e-3 + 0.05 * parent_cpu:
        out.append(f"{where}: children CPU {total:.4f} exceeds the parent's {parent_cpu:.4f}")
    return out


def test_trace() -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    res = {}
    for t in ("0", "1"):
        p = subprocess.run(
            [sys.executable, "-c", _TRACE_PROBE],
            cwd=ROOT,
            env={**env, "ARCHYTASZX_TRACE": t},
            capture_output=True,
            text=True,
            check=False,
        )
        res[t] = json.loads(p.stdout.strip().splitlines()[-1])
    fails = []
    for k in res["1"]:
        a, b = res["0"].get(k), res["1"][k]
        if a is None or (a["verdict"], a["method"]) != (b["verdict"], b["method"]):
            fails.append(
                f"{k}: traced {b['verdict']}/{b['method']} "
                f"vs untraced {a and a['verdict']}/{a and a['method']}"
            )
        if a is not None and a.get("spans"):
            fails.append(f"{k}: spans recorded with tracing off")
        if not b.get("spans"):
            fails.append(f"{k}: no spans with tracing on")
        fails += _nested(b.get("spans") or [], None, k)
    return fails


def main() -> None:
    if "--rival-one" in sys.argv:
        job = json.loads(sys.argv[sys.argv.index("--rival-one") + 1])
        print(json.dumps(_rival_job(tuple(job))), flush=True)
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = 2
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
        args = [a for a in args if a != str(jobs)]
    tests = {
        "false": lambda: test_false(jobs),
        "rivals": lambda: test_rivals(jobs),
        "schedule": test_schedule,
        "stats": test_stats,
        "resume": test_resume,
        "trace": test_trace,
    }
    names = args or list(tests)
    total = 0
    for name in names:
        lines = tests[name]()
        for f in lines:
            print(f"[{name}] {f}", flush=True)
        fails = [f for f in lines if not f.startswith("SKIP ")]
        print(f"{name}: {len(fails)} failures", flush=True)
        total += len(fails)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
