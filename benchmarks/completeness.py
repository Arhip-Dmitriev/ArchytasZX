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

"""Benchmark: the lemmas of [P1] and [P2] (archytaszx/papers/*/LEMMAS.txt), and every template
rule once.

Lemma rows, one per encoded lemma and mode: ``auto`` (``prove`` naming no rule),
``exhaustive`` (``prove_exhaustive``; default 3600 CPU s, one run; layers and diagrams visited
reported, also on timeout), ``kernel`` (``certificate kernel`` on the automatic proof) and
``lean`` (``certificate lean`` and the Lean runner). A lemma with no command file is one row,
status ``unsupported`` ("not encoded"). Lemma files: examples/completeness/lemmas/ and
benchmarks/workloads/completeness/.

Rule rows (``paper`` "rules", ``mode`` "rule"): every rule of rules_axioms.by_name applied once
to a seeded random instance of its matched side, certified, replayed, kernel-checked, exported
and checked by Lean.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks import engine, harness, lean, workloads
from benchmarks.harness import Inconclusive, Wrong

SIZE = "size"
RECORD_ENV = True
MODES = ("auto", "exhaustive", "kernel", "lean")
EXHAUSTIVE_TIMEOUT = 3600.0
QUICK_LEMMAS = (("P1", "8"), ("P1", "11"), ("P1", "38"))
QUICK_RULES = ("zxw_s2", "zxw_k0")


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every lemma in every mode, every unencoded lemma once, every rule once."""
    from archytaszx.rewrite.rules_axioms import by_name

    out: list[dict[str, Any]] = []
    for e in workloads.lemma_entries():
        if quick and (e["paper"], e["lemma"]) not in QUICK_LEMMAS:
            continue
        base = {"suite": "completeness", "paper": e["paper"], "lemma": str(e["lemma"])}
        if e["file"] is None:
            out.append({**base, "mode": "auto", "size": 1})
            continue
        for mode in MODES:
            if quick and mode == "exhaustive" and e["lemma"] != "38":
                continue
            out.append({**base, "mode": mode, "size": 1})
    for name in sorted(by_name()):
        if quick and name not in QUICK_RULES:
            continue
        out.append(
            {"suite": "completeness", "paper": "rules", "lemma": name, "mode": "rule", "size": 1}
        )
    return out


def case_limits(case: dict[str, Any], timeout: float, repeat: int) -> tuple[float, int]:
    """Exhaustive rows: :data:`EXHAUSTIVE_TIMEOUT` and one run."""
    if case["mode"] == "exhaustive":
        return EXHAUSTIVE_TIMEOUT, 1
    return timeout, repeat


def _entry(case: dict[str, Any]) -> dict[str, object]:
    for e in workloads.lemma_entries():
        if e["paper"] == case["paper"] and str(e["lemma"]) == case["lemma"]:
            return e
    raise ValueError(f"no lemma {case['paper']} {case['lemma']}")


def _settled(v: dict[str, Any]) -> None:
    if v["verdict"] == "unequal":
        raise Wrong(f"refuted by {v['method']}: {v['reason']}"[:400])
    if v["verdict"] != "equal":
        raise Inconclusive(f"{v['method']}: {v['reason']}"[:400])


def _exhaustive(script: str, base: str) -> dict[str, Any]:
    """``prove_exhaustive`` on the pair the script's verdict line names, with every layer
    reported by :func:`benchmarks.harness.progress`."""
    from archytaszx.repl.commands import _pair_args
    from archytaszx.semantics.prove import ProofStatus, prove_exhaustive

    text, line = workloads.split_verdict(script)
    prefix = engine.run_script(text, base=base)
    args = line.split(None, 1)[1] if " " in line else ""
    args = args.removeprefix("exhaustive").strip()
    left, right = _pair_args(prefix.session, args)
    seen: dict[str, Any] = {"layers_start": 0, "layers_goal": 0, "visited": 0, "reached": {}}

    def on_layer(report: Any) -> None:
        seen[f"layers_{report.side}"] = report.depth
        seen["reached"][report.side] = report.reached
        seen["visited"] = sum(seen["reached"].values())
        harness.progress({k: v for k, v in seen.items() if k != "reached"})

    c0, w0 = time.process_time(), time.perf_counter()
    outcome = prove_exhaustive(left, right, on_layer=on_layer)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    fields = {k: v for k, v in seen.items() if k != "reached"}
    if outcome.status is ProofStatus.REFUTED:
        raise Wrong(f"refuted: {outcome.reason}"[:400])
    if outcome.status is not ProofStatus.PROVED or outcome.certificate is None:
        return {
            "status": "inconclusive",
            "detail": outcome.reason[:400],
            "cpu": cpu,
            "wall": wall,
            **fields,
        }
    return {
        "cpu": cpu,
        "wall": wall,
        "derivation_length": outcome.certificate.length,
        "method": "exhaustive",
        **fields,
    }


def _lemma(case: dict[str, Any]) -> dict[str, Any]:
    e = _entry(case)
    if e["file"] is None:
        return {"status": "unsupported", "detail": "not encoded", "cpu": 0.0, "wall": 0.0}
    path = workloads.ROOT / str(e["file"])
    script = workloads.units(path.read_text(encoding="utf-8"))[0]
    base = str(path.parent)
    plain = "\n".join(ln for ln in script.splitlines() if not ln.startswith("certificate")) + "\n"
    out: dict[str, Any] = {"file": e["file"]}
    if case["mode"] == "exhaustive":
        return out | _exhaustive(plain, base)
    if case["mode"] == "lean":
        with tempfile.TemporaryDirectory(prefix="bench-lemma-") as tmp:
            out |= lean.export_and_check(plain, base, Path(tmp) / "lemma.lean")
        if out.get("status") is None and out.get("lean_status") != "accepted":
            out["detail"] = f"lean {out.get('lean_status')}: {out.get('lean_detail', '')}"[:400]
        return out
    kernel_line = "certificate kernel\n" if case["mode"] == "kernel" else ""
    run_ = engine.run_script(plain + kernel_line, base=base)
    v = engine.verdict(run_)
    _settled(v)
    cpu, wall = engine.until_verdict(run_)
    out |= {k: v[k] for k in ("method", "nodes", "wires", "derivation_length")}
    if case["mode"] == "auto":
        return out | {"cpu": cpu, "wall": wall}
    k = engine.kernel_fields(run_)
    if k["kernel"] in (None, "rejected"):
        return (
            out
            | k
            | {
                "status": "inconclusive",
                "detail": str(k.get("kernel_detail"))[:400],
                "cpu": 0.0,
                "wall": 0.0,
            }
        )
    line = run_.last("certificate", "certificate kernel")
    assert line is not None
    return out | k | {"cpu": line.cpu, "wall": line.wall}


def _rule(case: dict[str, Any]) -> dict[str, Any]:
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.rules_library import RULES
    from archytaszx.semantics import kernel
    from archytaszx.semantics.certificate import certify, replay
    from benchmarks.axiom_instances import instance

    name = case["lemma"]
    diagram = instance(name, 0)
    if diagram is None:
        return {
            "status": "unsupported",
            "detail": "no instance from the generator",
            "cpu": 0.0,
            "wall": 0.0,
        }
    rule = RULES[name]
    c0, w0 = time.process_time(), time.perf_counter()
    result = apply(diagram, rule, rule.pattern.find_matches(diagram)[0])
    certificate = certify(diagram, (result,), label=name)
    apply_cpu = time.process_time() - c0
    c = time.process_time()
    replayed = replay(certificate)
    replay_cpu = time.process_time() - c
    c = time.process_time()
    theorem = kernel.check(certificate)
    kernel_cpu = time.process_time() - c
    c = time.process_time()
    source = kernel.lean(theorem, name=name)
    export_cpu = time.process_time() - c
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    out: dict[str, Any] = {
        "cpu": cpu,
        "wall": wall,
        "apply_cpu": apply_cpu,
        "replay_cpu": replay_cpu,
        "replayed": replayed.reproduced,
        "kernel_cpu": kernel_cpu,
        "inferences": theorem.size(),
        "export_cpu": export_cpu,
        "nodes": len(diagram.nodes),
        "wires": len(diagram.wires),
    }
    with tempfile.TemporaryDirectory(prefix="bench-rule-") as tmp:
        path = Path(tmp) / "rule.lean"
        path.write_text(source, encoding="utf-8")
        out |= lean.check(path)
    return out


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One lemma in one mode, or one rule."""
    return _rule(case) if case["mode"] == "rule" else _lemma(case)
