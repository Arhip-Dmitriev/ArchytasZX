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

"""Quantomatic adapter: d = 2, concrete n, pure equations; the naive circuit and the
specification as red_green graphs (Z and X spiders with angles in units of pi, each F a hadamard
vertex, boundaries as wire vertices), each rewritten to a fixpoint with the size-reducing rules
of Quantomatic's red_green rule set, equal when the normal forms match (drivers/QuantoBench.scala,
run on the jar built by setup_rivals.sh). Boundary order and scalars are not compared; W,
triangle, binder, splitter, discard and weighted spiders are unsupported."""

from __future__ import annotations

import json
import os
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from benchmarks import stages, subject
from benchmarks.tools import common, rival

ENV = "quantomatic"
DIMENSIONS = (2,)
MAX_STEPS = 100_000


def jar() -> Path:
    """The built Quantomatic jar with the driver."""
    return rival.VENVS / ENV / "quanto.jar"


def rules() -> Path:
    """The shared red_green rule set copied by the setup script."""
    return rival.VENVS / ENV / "red_green.rules"


def java() -> str:
    """The JDK 8 java of the setup script, else java on PATH."""
    p = rival.VENVS / "jdk8" / "bin" / "java"
    return str(p) if p.exists() else "java"


def supports(inst: Any, point: Any) -> str | None:
    """Qubit-capable subjects, the jar built."""
    if not jar().exists():
        return "not installed: run benchmarks/tools/setup_rivals.sh quantomatic"
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def angle(t: Fraction) -> dict[str, Any]:
    """A red_green angle value of t turns (2 t pi)."""
    a = (2 * t) % 2
    if a == 0:
        return {"pretty": "0"}
    pretty = (
        "\\pi"
        if a == 1
        else f"{a.numerator}/{a.denominator} \\pi"
        if a.denominator != 1
        else f"{a.numerator} \\pi"
    )
    return {
        "pi": {"num": a.numerator, "denom": a.denominator},
        "pretty": pretty.replace("\\\\", "\\"),
    }


def qgraph(diagram: Any) -> dict[str, Any]:
    """A concrete pure diagram at d = 2 as a red_green graph."""
    from archytaszx.semantics import check

    g = check.instantiate(diagram, {})
    if not g.is_pure():
        raise rival.Unsupported("a mixed diagram")
    nodes: dict[str, Any] = {}
    for nid, node in g.nodes.items():
        name = node.generator_type.name
        if name == "F":
            nodes[f"v{nid}"] = {"data": {"type": "hadamard", "value": ""}}
            continue
        if name not in ("Z", "X"):
            raise rival.Unsupported(f"generator {name}")
        ph = node.phase
        if ph is not None and getattr(ph, "is_weighted", False):
            raise rival.Unsupported("a weighted spider")
        t = Fraction(str(ph.get(1).to_sympy_turns())) if ph is not None else Fraction(0)
        nodes[f"v{nid}"] = {"data": {"type": name, "value": angle(t)}}
    edges: dict[str, Any] = {}
    for k, w in enumerate(g.wires):
        edges[f"e{k}"] = {"src": f"v{w.a.node_id}", "tgt": f"v{w.b.node_id}"}
    wires: dict[str, Any] = {}
    for tag, refs in (("i", g.boundary_inputs), ("o", g.boundary_outputs)):
        for k, ref in enumerate(refs):
            wires[f"{tag}{k}"] = {"annotation": {"boundary": True}}
            edges[f"b{tag}{k}"] = {"src": f"{tag}{k}", "tgt": f"v{ref.node_id}"}
    return {"wire_vertices": wires, "node_vertices": nodes, "undir_edges": edges}


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One normalise-and-match run."""
    n, d = common.point(job)
    if job.get("point") is None:
        return common.unsupported("symbolic n through Quantomatic's own !-boxes is not built")
    why = supports(inst, None)
    if why or d != 2:
        return common.unsupported(why or "qubits only (d = 2)")
    c0 = time.process_time()
    try:
        left, right = stages.diagrams(subject.script(inst, n, 2).text)
        payload = {
            "theory": "red_green",
            "rules": str(rules()),
            "max_steps": MAX_STEPS,
            "lhs": qgraph(left),
            "rhs": qgraph(right),
        }
    except rival.Unsupported as exc:
        return common.unsupported(str(exc))
    prep = time.process_time() - c0
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        path = fh.name
    try:
        cmd = [java(), "-Xmx2g", "-Xss64m", "-cp", str(jar()), "quanto.bench.QuantoBench", path]
        res = rival.run_binary(
            cmd, float(job["config"]["stages"]["timeout_seconds"]), lift_memory=True
        )
    finally:
        os.unlink(path)
    text = res.pop("stdout", "")
    if res["status"] == "ok":
        try:
            res |= json.loads(text[text.index("{") :])
        except (json.JSONDecodeError, ValueError):
            res = {"status": "error", "detail": text[-300:]}
    out = rival.finish(inst, res, prep, {"quantomatic": "0762b5f"})
    if res.get("vertices_after") is not None:
        out["b3"] = {
            "tool": {"vertices": res["vertices_after"], "edges": res.get("edges_after")},
            "naive": {"vertices": res.get("vertices_before")},
        }
    return out


def _turns(value: Any, env: dict[str, Fraction]) -> Fraction:
    """The turns of a red_green angle value, variables substituted from env (in units of pi)."""
    if not isinstance(value, dict):
        return Fraction(0)
    total = Fraction(0)
    if "pi" in value:
        total += Fraction(value["pi"]["num"], value["pi"]["denom"])
    for name, c in value.get("vars", {}).items():
        total += Fraction(c["num"], c["denom"]) * env[name.lstrip("\\")]
    return total / 2


def rule_side_dsl(
    graph: dict[str, Any], keep_boxes: bool, env: dict[str, Fraction]
) -> tuple[str, list[str]]:
    """One side of a red_green rule as an engine DSL body at d = 2 (!-box contents kept once or
    dropped; a spider with no legs a scalar factor) and the boundary names in the order of its
    tensor axes (outputs, then inputs)."""
    drop: set[str] = set()
    if not keep_boxes:
        for box in graph.get("bang_boxes", {}).values():
            drop |= set(box.get("contents", []))
    wires = sorted(w for w in graph.get("wire_vertices", {}) if w not in drop)
    nodes = {k: v for k, v in graph.get("node_vertices", {}).items() if k not in drop}
    edges = [
        e
        for e in graph.get("undir_edges", {}).values()
        if e["src"] not in drop and e["tgt"] not in drop
    ]
    legs: dict[str, int] = {k: 0 for k in nodes}
    lines: list[str] = []
    bare = 0
    ends: dict[str, str] = {}
    wire_links: list[str] = []
    for e in edges:
        s, t = e["src"], e["tgt"]
        if s in nodes and t in nodes:
            ps = legs[s]
            legs[s] += 1
            pt = legs[t]
            legs[t] += 1
            wire_links.append(f"{s}.{ps} -- {t}.{pt}")
        elif s in nodes or t in nodes:
            node, w = (s, t) if s in nodes else (t, s)
            ends[w] = f"{node}.{legs[node]}"
            legs[node] += 1
        else:
            name = f"bare{bare}"
            bare += 1
            nodes[name] = {"data": {"type": "Z", "value": {"pretty": "0"}}}
            legs[name] = 2
            ends[s], ends[t] = f"{name}.0", f"{name}.1"
    decl: dict[str, str] = {}
    factors: list[str] = []
    for k, v in nodes.items():
        kind = (v.get("data") or {}).get("type", "Z")
        if legs[k] == 0 and kind in ("Z", "X"):
            t = _turns((v.get("data") or {}).get("value"), env) % 1
            factors.append(f"(1 + exp(2*pi*I*{t}))")
            continue
        if kind == "hadamard":
            if legs[k] != 2:
                raise rival.Unsupported("a hadamard vertex of arity other than 2")
            decl[k] = "F"
            lines.append(f"{k} = F(literal 2)")
            continue
        t = _turns((v.get("data") or {}).get("value"), env)
        ph = f" phase{{1: {t % 1}}}" if t % 1 else ""
        decl[k] = kind
        lines.append(f"{k} = {kind}(0 -> {legs[k]}, literal 2){ph}")

    def port(ref: str) -> str:
        node, k = ref.split(".")
        if decl.get(node) == "F":
            return f"{node}.in0" if k == "0" else f"{node}.out0"
        return f"{node}.out{k}"

    if wire_links:
        lines.append(
            "wire "
            + ", ".join(" -- ".join(port(x) for x in link.split(" -- ")) for link in wire_links)
        )
    named = [(w, port(ends[w])) for w in wires if w in ends]
    outs = [(w, p) for w, p in named if not p.endswith(".in0")]
    ins = [(w, p) for w, p in named if p.endswith(".in0")]
    if ins:
        lines.append("inputs " + ", ".join(p for _, p in ins))
    if outs:
        lines.append("outputs " + ", ".join(p for _, p in outs))
    if factors:
        lines.append("scalar " + " * ".join(factors))
    return "\n".join(lines), [w for w, _ in outs] + [w for w, _ in ins]
