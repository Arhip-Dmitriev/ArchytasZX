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

"""Plots from results/aggregate (BENCHMARKS.txt, section 12.2): Okabe-Ito hues in fixed order
(yellow and black left out), a marker shape per tool beside its hue, white background, solid
lines, embedded fonts, 300 dpi PNG and PDF."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks import config

HUES = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s")
TOOLS = (
    "archytaszx",
    "pyzx",
    "quizx",
    "qcec",
    "feynman",
    "dizx",
    "qiskit",
    "cirq",
    "autoq_para",
    "tket",
    "quantomatic",
    "published_proofs",
)
CATEGORIES = ("algorithm", "code", "protocol", "circuit")
STAGES = ("decide", "certificate", "lean_export", "lean_check")
SET_COLORS = {"refuted": HUES[0], "wrongly_accepted": HUES[1], "unknown": HUES[2], "other": HUES[3]}


def style(tool: str) -> dict[str, Any]:
    """The fixed hue and marker of a tool."""
    k = TOOLS.index(tool) if tool in TOOLS else len(TOOLS) - 1
    return {"color": HUES[k % len(HUES)], "marker": MARKERS[(k // len(HUES)) % len(MARKERS)]}


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.grid": True,
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.5,
            "axes.edgecolor": "#888888",
            "lines.linewidth": 1.5,
            "lines.linestyle": "-",
            "legend.frameon": False,
        }
    )
    return plt


def _save(fig: Any, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _load(agg: Path, name: str) -> list[dict[str, Any]]:
    p = agg / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else []


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2


def time_in_axis(rows: list[dict[str, Any]], out: Path, axis: str, stage: str) -> None:
    """Stage CPU time against n (or d): one panel per category, each equation a faint line per
    tool, the median over equations in bold."""
    plt = _mpl()
    field = f"{stage}_cpu_median"
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    any_data = False
    for ax, cat in zip(axes, CATEGORIES):
        by_tool: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in rows:
            if (
                r["category"] != cat
                or not r["truth"]
                or r.get(field) is None
                or r.get(axis) is None
                or r["version"] == "symbolic"
            ):
                continue
            by_tool[r["tool"]][r["equation"]].append((r[axis], r[field]))
        for tool, eqs in by_tool.items():
            st = style(tool)
            agg: dict[float, list[float]] = defaultdict(list)
            for pts in eqs.values():
                pts.sort()
                ax.plot(
                    [p[0] + (1 if axis == "n" else 0) for p in pts],
                    [p[1] for p in pts],
                    color=st["color"],
                    alpha=0.18,
                    linewidth=0.7,
                )
                for x, y in pts:
                    agg[x].append(y)
            xs = sorted(agg)
            if xs:
                any_data = True
                ax.plot(
                    [x + (1 if axis == "n" else 0) for x in xs],
                    [_median(agg[x]) for x in xs],
                    color=st["color"],
                    marker=st["marker"],
                    markersize=3,
                    linewidth=1.8,
                    label=tool,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(cat)
        ax.set_xlabel("n + 1" if axis == "n" else "d")
    axes[0].set_ylabel(f"{stage} CPU s")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            bbox_to_anchor=(0.5, -0.12),
        )
    if any_data:
        _save(fig, out, f"time_{stage}_in_{axis}")
    else:
        plt.close(fig)


def surface(rows: list[dict[str, Any]], out: Path) -> None:
    """ArchytasZX decide CPU time against n and d per category, as 3D scatter."""
    plt = _mpl()
    fig = plt.figure(figsize=(11, 3))
    for k, cat in enumerate(CATEGORIES):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        pts = [
            (r["n"], r["d"], r["decide_cpu_median"])
            for r in rows
            if r["category"] == cat
            and r["tool"] == "archytaszx"
            and r["truth"]
            and r.get("decide_cpu_median")
            and r.get("n") is not None
            and r.get("d") is not None
        ]
        if pts:
            ax.scatter(
                [math.log10(p[0] + 1) for p in pts],
                [math.log10(p[1]) for p in pts],
                [math.log10(p[2]) for p in pts],
                color=HUES[0],
                s=4,
            )
        ax.set_title(cat)
        ax.set_xlabel("log10(n+1)")
        ax.set_ylabel("log10 d")
        ax.set_zlabel("log10 CPU s")
    _save(fig, out, "time_3d")


def frontier(units: list[dict[str, Any]], out: Path) -> None:
    """Per tool, the largest successful (n, d) of each equation."""
    plt = _mpl()
    tools = sorted(
        {
            u["tool"]
            for u in units
            if u.get("n_max_ok") is not None or u.get("d_max_ok") is not None
        },
        key=lambda t: TOOLS.index(t) if t in TOOLS else 99,
    )
    if not tools:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for t in tools:
        st = style(t)
        pts = [
            ((u.get("n_max_ok") or 0) + 1, u.get("d_max_ok") or 1)
            for u in units
            if u["tool"] == t and u.get("truth")
        ]
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=st["color"],
            marker=st["marker"],
            s=10,
            label=t,
        )
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("largest n + 1 reached")
    ax.set_ylabel("largest d reached")
    ax.legend(loc="best")
    _save(fig, out, "frontier")


def lean_sizes(rows: list[dict[str, Any]], out: Path) -> None:
    """Lean check time and Lean lines against n and against d (ArchytasZX, S)."""
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    sel = [
        r for r in rows if r["tool"] == "archytaszx" and r["truth"] and r["version"] != "symbolic"
    ]
    for j, axis in enumerate(("n", "d")):
        for i, (field, lab) in enumerate(
            (
                ("lean_check_cpu_median", "Lean check CPU s"),
                ("lean_export.lean_lines", "Lean lines"),
            )
        ):
            ax = axes[i][j]
            for k, cat in enumerate(CATEGORIES):
                pts = [
                    (r[axis] + (1 if axis == "n" else 0), r[field])
                    for r in sel
                    if r["category"] == cat and r.get(field) is not None and r.get(axis) is not None
                ]
                if pts:
                    ax.scatter(
                        [p[0] for p in pts], [p[1] for p in pts], s=6, color=HUES[k], label=cat
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("n + 1" if axis == "n" else "d")
            ax.set_ylabel(lab)
    axes[0][0].legend()
    _save(fig, out, "lean_sizes")


def b3_plots(b3: list[dict[str, Any]], out: Path) -> None:
    """B3: smallest total elements against n and d, and the naive-to-smallest ratio."""
    plt = _mpl()
    if not b3:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for k, cat in enumerate(CATEGORIES):
        sel = [r for r in b3 if r["category"] == cat]
        for j, axis in enumerate(("n", "d")):
            pts = [
                (r[axis] + (1 if axis == "n" else 0), r.get("smallest.total"))
                for r in sel
                if r.get(axis) is not None and r.get("smallest.total") is not None
            ]
            if pts:
                axes[j].scatter(
                    [p[0] for p in pts], [p[1] for p in pts], s=6, color=HUES[k], label=cat
                )
        ratios = [r["ratio"] for r in sel if r.get("ratio")]
        if ratios:
            axes[2].bar(k, _median(ratios), color=HUES[k], width=0.6)
    for j, axis in enumerate(("n", "d")):
        axes[j].set_xscale("log")
        axes[j].set_yscale("log")
        axes[j].set_xlabel("n + 1" if axis == "n" else "d")
        axes[j].set_ylabel("smallest diagram, elements")
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(CATEGORIES)
    axes[2].set_ylabel("naive / smallest (median)")
    axes[0].legend()
    _save(fig, out, "b3_sizes")


def breakdown_bars(rows: list[dict[str, Any]], out: Path, group: str, top: int = 8) -> None:
    """Share of traced CPU per category by rung, rule or inference kind: the ``top`` largest
    keys in fixed hue order, the rest as other."""
    plt = _mpl()
    sel = [r for r in rows if r["group"] == group and r["tool"] == "archytaszx"]
    if not sel:
        return
    totals: dict[str, float] = defaultdict(float)
    for r in sel:
        totals[r["name"]] += r["cpu"]
    keys = [k for k, _ in sorted(totals.items(), key=lambda kv: -kv[1])][: min(top, len(HUES))]
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, cat in enumerate(CATEGORIES):
        per: dict[str, float] = defaultdict(float)
        for r in sel:
            if r["category"] == cat:
                per[r["name"] if r["name"] in keys else "other"] += r["cpu"]
        tot = sum(per.values()) or 1
        left = 0.0
        for k, name in enumerate([*keys, "other"]):
            w = per.get(name, 0) / tot
            ax.barh(
                i,
                w,
                left=left,
                color=HUES[k] if name != "other" else "#bbbbbb",
                edgecolor="white",
                linewidth=1,
                label=name if i == 0 else None,
            )
            left += w
    ax.set_yticks(range(4))
    ax.set_yticklabels(CATEGORIES)
    ax.set_xlabel(f"share of traced CPU by {group}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=6)
    _save(fig, out, f"breakdown_{group}")


def b2_counts(counts: list[dict[str, Any]], out: Path) -> None:
    """F points per break kind and tool: refuted, wrongly accepted, unknown."""
    plt = _mpl()
    sel = [c for c in counts if c["set"] == "F"]
    if not sel:
        return
    tools = sorted({c["tool"] for c in sel}, key=lambda t: TOOLS.index(t) if t in TOOLS else 99)
    kinds = sorted({c["kind"] for c in sel if c["kind"]})
    fig, axes = plt.subplots(
        1, len(tools), figsize=(3 * len(tools), 3.5), sharey=True, squeeze=False
    )
    for ax, t in zip(axes[0], tools):
        for i, kind in enumerate(kinds):
            agg: dict[str, int] = defaultdict(int)
            for c in sel:
                if c["tool"] == t and c["kind"] == kind:
                    agg["refuted"] += c.get("refuted", 0)
                    agg["wrongly_accepted"] += c.get("wrongly_accepted", 0)
                    agg["unknown"] += c.get("unknown", 0)
                    agg["other"] += (
                        c.get("points", 0)
                        - c.get("refuted", 0)
                        - c.get("wrongly_accepted", 0)
                        - c.get("unknown", 0)
                    )
            left = 0
            for name in ("refuted", "wrongly_accepted", "unknown", "other"):
                ax.barh(
                    i,
                    agg[name],
                    left=left,
                    color=SET_COLORS[name],
                    edgecolor="white",
                    linewidth=1,
                    label=name if i == 0 else None,
                )
                left += agg[name]
        ax.set_title(t)
        ax.set_yticks(range(len(kinds)))
        ax.set_yticklabels(kinds, fontsize=6)
        ax.set_xlabel("F points")
    axes[0][0].legend(loc="lower right", fontsize=6)
    _save(fig, out, "b2_kinds")


def proved_bins(counts: list[dict[str, Any]], out: Path) -> None:
    """S points proved per n bin and per d bin, per tool."""
    plt = _mpl()
    sel = [c for c in counts if c["set"] == "S"]
    if not sel:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for j, b in enumerate(("n_bin", "d_bin")):
        bins = sorted({c[b] for c in sel}, key=_bin_key)
        tools = sorted({c["tool"] for c in sel}, key=lambda t: TOOLS.index(t) if t in TOOLS else 99)
        for t in tools:
            st = style(t)
            ys = [
                sum(c.get("proved", 0) for c in sel if c["tool"] == t and c[b] == x) for x in bins
            ]
            axes[j].plot(
                range(len(bins)), ys, color=st["color"], marker=st["marker"], markersize=3, label=t
            )
        axes[j].set_xticks(range(len(bins)))
        axes[j].set_xticklabels(bins, rotation=45, fontsize=6)
        axes[j].set_xlabel(b.replace("_", " "))
        axes[j].set_ylabel("points proved")
    axes[0].legend()
    _save(fig, out, "proved_bins")


def _bin_key(b: str) -> float:
    if b == "none":
        return -2
    if b.startswith("1e"):
        return 10 ** int(b[2:])
    return float(b.split("-")[-1])


def main(cfg: dict[str, Any]) -> None:
    """Every plot from the aggregates."""
    agg = config.path(cfg, "results") / "aggregate"
    out = config.path(cfg, "plots")
    rows = _load(agg, "points")
    for stage in STAGES:
        for axis in ("n", "d"):
            time_in_axis(rows, out, axis, stage)
    surface(rows, out)
    frontier(_load(agg, "units"), out)
    lean_sizes(rows, out)
    b3_plots(_load(agg, "b3"), out)
    bd = _load(agg, "breakdown")
    for group in ("rung", "rule", "inference"):
        breakdown_bars(bd, out, group)
    counts = _load(agg, "counts")
    b2_counts(counts, out)
    proved_bins(counts, out)
    print(f"plots -> {out}")
