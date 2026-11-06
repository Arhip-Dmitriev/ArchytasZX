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

"""Test, ``python -m tests._trust_mutations_script [--quick] [OUT_DIR]``: the mutation table.
Each mutation is a deliberate translation bug, applied in the worker's memory
to the exporter (archytaszx.semantics.lean_net._Export) or to the diagrams the parser
returns to the session, never to the installed files; every exported proof or
refutation of every example and battery file is then checked by Lean. Rejected is ``ok``,
accepted is ``wrong`` unless the accepted theorem depends on more ``unproved_*`` axioms than the
unmutated export (then ``ok``: the mutation surfaced as an unproved step); a mutation with
nothing to act on, one that leaves the export unchanged, and one whose export falls back to the
opaque form or fails are ``unsupported``.

Mutations (all consistent: every network the export builds carries the same bug):

- ``swap legs``: the labels of two legs on different nodes exchanged (two wires crossed).
- ``drop wire``: the lowest internal label cut into two labels, each summed alone.
- ``swap boundary``: the first two boundary labels exchanged.
- ``chain off by one``: one Bun layer removed from the first boxed node.
- ``bundle multiplicity``: the first Bun layer's multiplicity m replaced by m + 1.
- ``phase entry``: 1/7 turn added at index 1 to every phase function of the table.
- ``scalar``: the first scalar the export writes doubled.
- ``dimension``: the first dimension of the table increased by 1.
- ``source literal``: the scalar of the first diagram the parser returns doubled.

Battery files: benchmarks/workloads/trust/*.azx and .trust_work/battery*.azx when present.
"""

from __future__ import annotations

import csv
import io
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks import engine, lean, workloads

SIZE = "size"
REPEAT = 1
RECORD_ENV = True
MUTATIONS = (
    "swap legs",
    "drop wire",
    "swap boundary",
    "chain off by one",
    "bundle multiplicity",
    "phase entry",
    "scalar",
    "dimension",
    "source literal",
)
QUICK_MUTATIONS = ("swap legs", "drop wire", "source literal")
QUICK_WORKLOADS = ("examples/bernstein_vazirani_proof.azx #1",)


def battery_files() -> list[Path]:
    """The battery files present."""
    out = sorted((workloads.WORKLOADS / "trust").glob("*.azx"))
    return out + sorted((workloads.ROOT / ".trust_work").glob("battery*.azx"))


def unit_list() -> list[tuple[str, str, str]]:
    """(name, script, base) for every verdict of every example and battery file."""
    out = []
    files = [p for p in workloads.example_files() if "completeness" not in p.parts]
    for path in [*files, *battery_files()]:
        for k, script in enumerate(workloads.units(path.read_text(encoding="utf-8"))):
            out.append((f"{workloads.rel(path)} #{k + 1}", script, str(path.parent)))
    return out


def cases(quick: bool) -> list[dict[str, Any]]:
    """Every mutation on every unit."""
    return [
        {"suite": "trust", "mutation": m, "workload": name, "size": 1}
        for m in MUTATIONS
        for name, _, _ in unit_list()
        if not quick or (m in QUICK_MUTATIONS and name in QUICK_WORKLOADS)
    ]


class _Hit:
    """Whether a mutation acted."""

    acted = False


def _bun_depth(kind: Any) -> int:
    return 1 + _bun_depth(kind[2]) if isinstance(kind, tuple) and kind[:1] == ("Bun",) else 0


def _mutate_net(self: Any, net: Any, mutation: str) -> Any:
    """``net`` with ``mutation`` applied, or ``net`` when it has nothing to act on."""
    nodes = list(net.nodes)
    if mutation == "swap legs":
        legs = [
            (i, side, j, x)
            for i, n in enumerate(nodes)
            for side, row in (("o", n.outs), ("i", n.ins))
            for j, x in enumerate(row)
        ]
        for a in legs:
            b = next((b for b in legs if b[0] != a[0] and b[3] != a[3]), None)
            if b is None:
                continue
            for (i, side, j, _), x in ((a, b[3]), (b, a[3])):
                n = nodes[i]
                row = list(n.outs if side == "o" else n.ins)
                row[j] = x
                nodes[i] = replace(n, **{"outs" if side == "o" else "ins": tuple(row)})
            _Hit.acted = True
            return replace(net, nodes=tuple(nodes))
        return net
    if mutation == "drop wire":
        if not net.summed:
            return net
        cut = min(net.summed)
        fresh = max([*net.summed, *net.bnd, *(x for n in nodes for x in n.labels())]) + 1
        dim = self.labs.get(cut)
        if dim is not None:
            self.labs.setdefault(fresh, dim)
        seen = False
        for i, n in enumerate(nodes):
            outs, ins = list(n.outs), list(n.ins)
            for row in (outs, ins):
                for j, x in enumerate(row):
                    if x == cut:
                        if seen:
                            row[j] = fresh
                        seen = True
            nodes[i] = replace(n, outs=tuple(outs), ins=tuple(ins))
        _Hit.acted = True
        return replace(net, nodes=tuple(nodes), summed=(*net.summed, fresh))
    if mutation == "swap boundary":
        if len(net.bnd) < 2:
            return net
        _Hit.acted = True
        bnd = list(net.bnd)
        bnd[0], bnd[1] = bnd[1], bnd[0]
        return replace(net, bnd=tuple(bnd))
    if mutation in ("chain off by one", "bundle multiplicity"):
        for i, n in enumerate(nodes):
            if _bun_depth(n.kind) == 0:
                continue
            _, m, inner = n.kind
            if mutation == "chain off by one":
                kind = inner
            else:
                kind = ("Bun", self.mult_of(self.mult_exprs[m] + 1), inner)
            nodes[i] = replace(n, kind=kind)
            _Hit.acted = True
            return replace(net, nodes=tuple(nodes))
        return net
    return net


def _install(mutation: str) -> None:
    """Patches the exporter, or the session's parser, in this process."""
    from archytaszx.repl import commands
    from archytaszx.semantics import lean_net

    E = lean_net._Export
    if mutation in (
        "swap legs",
        "drop wire",
        "swap boundary",
        "chain off by one",
        "bundle multiplicity",
    ):
        original = E.translate

        def translate(self: Any, *args: Any, **kwargs: Any) -> Any:
            net, order, port = original(self, *args, **kwargs)
            return _mutate_net(self, net, mutation), order, port

        E.translate = translate  # type: ignore[method-assign]
    elif mutation == "phase entry":
        original_phase = E.phase

        def phase(self: Any, pv: Any) -> int:
            i = original_phase(self, pv)
            if i and not self.phases[i].startswith("fun k => (if k = 1 then 1/7"):
                self.phases[i] = (
                    "fun k => (if k = 1 then 1/7 else 0) + (" + self.phases[i][9:] + ")"
                )
                _Hit.acted = True
            return i

        E.phase = phase  # type: ignore[method-assign]
    elif mutation == "scalar":
        original_scalar = E.scalar

        def scalar(self: Any, s: Any) -> Any:
            out = original_scalar(self, s)
            if not getattr(self, "_bench_scaled", False):
                self._bench_scaled = True
                _Hit.acted = True
                return 2 * out
            return out

        E.scalar = scalar  # type: ignore[method-assign]
    elif mutation == "dimension":
        original_dim = E.dim

        def dim(self: Any, d: Any) -> int:
            before = len(self.dims)
            i = original_dim(self, d)
            if before == 0 and self.dims:
                self.dims[0] = f"({self.dims[0]} + 1)"
                _Hit.acted = True
            return i

        E.dim = dim  # type: ignore[method-assign]
    elif mutation == "source literal":
        original_parse = getattr(commands, "parse")  # noqa: B009

        def parse(source: str) -> Any:
            parsed = original_parse(source)
            diagram = getattr(parsed, "diagram", None)
            if diagram is not None and not _Hit.acted:
                from archytaszx.algebra.scalar import Scalar

                diagram.multiply_scalar(Scalar(2))
                _Hit.acted = True
            return parsed

        setattr(commands, "parse", parse)  # noqa: B010
    else:
        raise ValueError(mutation)


def run(case: dict[str, Any]) -> dict[str, Any]:
    """One mutated export of one unit, checked by Lean."""
    match = [u for u in unit_list() if u[0] == case["workload"]]
    if not match:
        raise ValueError(f"no workload {case['workload']!r}")
    _, script, base = match[0]
    with tempfile.TemporaryDirectory(prefix="bench-trust-") as tmp:
        clean, path = Path(tmp) / "clean.lean", Path(tmp) / "mutated.lean"
        engine.run_script(workloads.with_lean(script, clean, kernel=False), base=base)
        _install(case["mutation"])
        run_ = engine.run_script(workloads.with_lean(script, path, kernel=False), base=base)
        out: dict[str, Any] = {"cpu": 0.0, "wall": 0.0, **engine.lean_export_fields(run_)}
        out["verdict"] = engine.verdict(run_)["verdict"]
        if not _Hit.acted:
            return out | {"status": "unsupported", "detail": "the mutation had nothing to act on"}
        if not path.exists():
            detail = f"no mutated export: {out['lean_semantics']}"
            return out | {"status": "unsupported", "detail": detail[:400]}
        text = path.read_text(encoding="utf-8")
        if clean.exists() and text == clean.read_text(encoding="utf-8"):
            return out | {"status": "unsupported", "detail": "the mutated export is unchanged"}
        if "import ArchytasZX" not in text:
            return out | {"status": "unsupported", "detail": "the mutated export is opaque"}
        out |= lean.check(path)
        out |= {"cpu": out["lean_cpu"], "wall": out["lean_wall"]}
        before = (
            clean.read_text(encoding="utf-8").count("\naxiom unproved_") if clean.exists() else 0
        )
        if out["lean_status"] == "accepted" and (out["unproved"] or 0) > before:
            more = (out["unproved"] or 0) - before
            return out | {"detail": f"accepted only with {more} new unproved step(s)"}
        if out["lean_status"] == "accepted":
            return out | {"status": "wrong", "detail": "Lean accepted the mutated export"}
        if out["lean_status"] != "rejected":
            return out | {"status": out["lean_status"]}
    return out


def table(rows: list[dict[str, Any]]) -> tuple[str, str]:
    """The mutation table as aligned plain text and as CSV."""
    head = ("mutation", "workload", "status", "lean_status", "time", "detail")
    data = [
        (
            r["mutation"],
            r["workload"],
            r["status"],
            str(r.get("lean_status") or ""),
            "" if r.get("time") is None else f"{r['time']:.2f}",
            str(r.get("detail") or "")[:80],
        )
        for r in sorted(rows, key=lambda r: (MUTATIONS.index(r["mutation"]), r["workload"]))
    ]
    widths = [
        max(len(h), *(len(row[i]) for row in data)) if data else len(h) for i, h in enumerate(head)
    ]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in [head, *data]]
    lines.insert(1, "  ".join("-" * w for w in widths))
    buf = io.StringIO()
    csv.writer(buf).writerows([head, *data])
    return "\n".join(lines) + "\n", buf.getvalue()


def _one(case: dict[str, Any]) -> dict[str, Any]:
    """One case in this (fresh) process: its row."""
    try:
        return {**case, "status": "ok", **run(case)}
    except Exception as exc:  # noqa: BLE001
        return {**case, "status": "error", "detail": f"{type(exc).__name__}: {exc}"[:400]}


def main() -> int:
    """Every case in its own process; writes trust_mutations.txt and .csv to OUT_DIR (default:
    the current directory); exit status 1 when a mutation was accepted."""
    import multiprocessing
    import sys

    args = [a for a in sys.argv[1:] if a != "--quick"]
    out = Path(args[0]) if args else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=4, maxtasksperchild=1) as pool:
        rows = pool.map(_one, cases("--quick" in sys.argv), chunksize=1)
    text, csv_text = table(rows)
    (out / "trust_mutations.txt").write_text(text, encoding="utf-8")
    (out / "trust_mutations.csv").write_text(csv_text, encoding="utf-8")
    print(text)
    return 1 if any(r.get("status") == "wrong" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
