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

"""ArchytasZX adapter: generation of the script, then the four stages of BENCHMARKS.txt
section 8 in this process, and the B3 counts of a decided run."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from benchmarks import stages, subject, translate
from benchmarks.tools import common


def supports(inst: Any, point: tuple[int | None, int | None] | None) -> str | None:
    """Every instance is supported; a version the translator refuses is reported by run."""
    return None


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One run: generate, decide, certificate, lean_export, lean_check."""
    from benchmarks.worker import mark

    cfg = job["config"]
    st = cfg["stages"]
    limit = float(st["timeout_seconds"])
    nv, dv = common.point(job)
    out: dict[str, Any] = {"expected": common.expected(inst), "stages": {}}
    holder: dict[str, Any] = {}

    def gen() -> dict[str, Any]:
        from archytaszx import trace

        with trace.span("generate.script"):
            sc = subject.script(inst, nv, dv)
        holder["script"] = sc
        return {
            "script_bytes": len(sc.text),
            "naive_nodes": sc.nodes,
            "literal": sc.literal,
            "point_claim": sc.point_claim,
        }

    mark("generate")
    try:
        g = stages.timed(gen, limit)
    except translate.Unsupported as exc:
        return common.unsupported(str(exc))
    if g["status"] == "error" and "Unsupported" in str(g.get("detail", "")):
        return common.unsupported(g["detail"])
    out["stages"]["generate"] = g
    if g["status"] != "ok":
        out["outcome"] = g["status"]
        return out
    script = holder["script"].text
    run_ = stages.new_run()
    mark("decide")
    dec = stages.timed(lambda: stages.decide_stage(run_, script), limit)
    out["stages"]["decide"] = dec
    verdict = dec.get("verdict")
    out["verdict"] = verdict
    out["certified"] = dec.get("certified")
    out["outcome"] = common.outcome(inst, dec["status"], verdict)
    out["wrong"] = out["outcome"] == "wrong"
    if dec["status"] == "ok" and verdict == "equal" and inst.truth:
        try:
            out["b3"] = _b3(run_)
        except Exception as exc:  # noqa: BLE001
            out["b3"] = {"error": str(exc)[:200]}
    decided = dec["status"] == "ok" and verdict in ("equal", "unequal", "equal_uncertified")
    reason = "no decided verdict" if not decided else None
    if st["certificate"]:
        if reason is None:
            mark("certificate")
            cert = stages.timed(lambda: stages.certificate_stage(run_), limit)
            if cert["status"] == "ok":
                cert["certificate_bytes"] = stages.certificate_bytes(run_)
            out["stages"]["certificate"] = cert
        else:
            out["stages"]["certificate"] = stages.not_run(reason)
    with tempfile.TemporaryDirectory(prefix="bench-lean-") as tmp:
        path = Path(tmp) / "export.lean"
        if st["lean_export"]:
            if reason is None:
                mark("lean_export")
                exp = stages.timed(lambda: stages.lean_export_stage(run_, path), limit)
                out["stages"]["lean_export"] = exp
                if exp["status"] != "ok":
                    reason = "no Lean file"
            else:
                out["stages"]["lean_export"] = stages.not_run(reason)
        else:
            reason = reason or "lean_export off"
        if st["lean_check"]:
            if reason is None:
                mark("lean_check")
                frac = float(st["memory_fraction"])
                out["stages"]["lean_check"] = stages.timed(
                    lambda: stages.lean_check_stage(path, limit, frac), None
                )
            else:
                out["stages"]["lean_check"] = stages.not_run(reason)
    return out


def _b3(run_: Any) -> dict[str, Any]:
    """The B3 counts from the session's last decide."""
    from archytaszx.repl.commands import _Pair
    from benchmarks import size

    last = next(o for o in reversed(run_.outcomes) if o.command == "decide")
    pair = last.detail
    if not isinstance(pair, _Pair):
        raise RuntimeError("no decision pair")  # noqa: TRY004
    return size.b3(pair.payload, pair.left, pair.right)
