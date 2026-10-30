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

"""Print-style plots: white background, Okabe-Ito colours, solid lines, PDF and 300 dpi PNG."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402

from benchmarks import harness, scaling, size  # noqa: E402

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
TOOL_COLOURS = {"azx-family": "#0072b2", "azx-qubit": "#009e73", "pyzx": "#d55e00"}
TOOL_LABELS = {
    "azx-family": "ArchytasZX, d free, n copies",
    "azx-qubit": "ArchytasZX, d = 2",
    "pyzx": "PyZX, d = 2",
}
FAILED = ("timeout", "oom", "skipped", "inconclusive", "wrong", "error")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _grid(count: int) -> tuple[plt.Figure, list[plt.Axes]]:
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.9 * cols + 0.6, 1.7 * rows + 0.5), squeeze=False
    )
    flat = [a for row in axes for a in row]
    for a in flat[count:]:
        a.set_visible(False)
    return fig, flat[:count]


def _decades(rows: list[dict]) -> tuple[float, float]:
    """The whole decades around every ``ok`` row's median, at least two decades wide."""
    ok = [r["time"] for r in rows if r["status"] == "ok"] or [1.0]
    lo = math.floor(math.log10(min(ok)))
    hi = max(math.ceil(math.log10(max(ok))), lo + 2)
    return 10.0**lo, 10.0**hi


def plot_scaling() -> None:
    """One panel per family: median CPU seconds against n, one line per d, the free case as a
    grey line; failed cases as crosses on the top edge."""
    rows = harness.load("scaling")
    if not rows:
        return
    _style()
    families = [f for f in scaling.FAMILIES if any(r["family"] == f for r in rows)]
    ds = sorted({r["d"] for r in rows if r["d"] != "free"})
    ramp = colormaps["Blues"]
    colour = {d: ramp(0.45 + 0.55 * k / max(1, len(ds) - 1)) for k, d in enumerate(ds)}
    fig, axes = _grid(len(families))
    top = _decades(rows)[1] / 1.6
    for ax, family in zip(axes, families):
        mine = [r for r in rows if r["family"] == family]
        for d in ds:
            line = sorted((r for r in mine if r["d"] == d), key=lambda r: r["n"])
            ok = [r for r in line if r["status"] == "ok"]
            ax.plot([r["n"] for r in ok], [r["time"] for r in ok], "-o", color=colour[d],
                    label=f"d = {d}" if d < 2**16 else f"d = 2^{int(math.log2(d))}")
            bad = [r for r in line if r["status"] in FAILED]
            ax.plot([r["n"] for r in bad], [top * 1.5] * len(bad), "x", color=colour[d])
        free = [r for r in mine if r["d"] == "free" and r["status"] == "ok"]
        if free:
            ax.axhline(free[0]["time"], color="#555555", lw=1.0, label="n, d free")
        ax.set_xscale("log", base=4)
        ax.set_yscale("log")
        ax.set_ylim(*_decades(rows))
        ax.set_title(family, fontsize=8)
        ax.set_xlabel("n")
        ax.set_ylabel("CPU s (median)")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    _save(fig, "scaling_n_d")


def plot_size() -> None:
    """One panel per width: median over seeds of each seed's median CPU seconds against the
    gate count of C C^-1, one line per tool; failed cases as crosses on the top edge."""
    rows = harness.load("size")
    if not rows:
        return
    _style()
    widths = sorted({r["w"] for r in rows})
    fig, axes = _grid(len(widths))
    top = max((r["time"] or 0) for r in rows if r["status"] == "ok") or 1
    for ax, w in zip(axes, widths):
        for tool in size.TOOLS:
            mine = [r for r in rows if r["w"] == w and r["tool"] == tool]
            xs, ys, lo, hi, bad = [], [], [], [], []
            for s in sorted({r["s"] for r in mine}):
                at = [r for r in mine if r["s"] == s]
                ok = [r["time"] for r in at if r["status"] == "ok"]
                if len(ok) * 2 > len(at):
                    xs.append(2 * s)
                    ys.append(statistics.median(ok))
                    lo.append(min(ok))
                    hi.append(max(ok))
                elif at:
                    bad.append(2 * s)
            colour = TOOL_COLOURS[tool]
            ax.plot(xs, ys, "-o", color=colour, label=TOOL_LABELS[tool])
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.2, lw=0)
            ax.plot(bad, [top * 1.5] * len(bad), "x", color=colour)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"w = {w} wires", fontsize=8)
        ax.set_xlabel("gates in C C^-1")
        ax.set_ylabel("CPU s (median)")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    _save(fig, "size_gates")


def plot_scaling_flatness() -> None:
    """Every bound (n, d) case's median CPU seconds over its family's free case, one strip per
    family; the line at 1 is the free case."""
    rows = harness.load("scaling")
    free = {r["family"]: r["time"] for r in rows if r["d"] == "free" and r["status"] == "ok"}
    families = [f for f in scaling.FAMILIES if f in free]
    if not families:
        return
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    for k, family in enumerate(families):
        ratios = [
            r["time"] / free[family]
            for r in rows
            if r["family"] == family and r["d"] != "free" and r["status"] == "ok"
        ]
        jitter = [k + 0.25 * ((j * 0.618) % 1 - 0.5) for j in range(len(ratios))]
        ax.plot(jitter, ratios, "o", color="#0072b2", alpha=0.45, mec="none")
        if ratios:
            ax.plot([k - 0.3, k + 0.3], [statistics.median(ratios)] * 2, "-", color="#222222")
    ax.axhline(1, color="#888888", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(families)), families, rotation=25, ha="right")
    ax.set_ylabel("bound time / free time")
    ax.set_title("Every bound n in 1..4^10 and d in 2..2^20, against n and d free", fontsize=8)
    fig.tight_layout()
    _save(fig, "scaling_flatness")


def plot_scaling_heatmap() -> None:
    """One panel per family: median CPU seconds over the (n, d) grid, one shared colour scale."""
    rows = [r for r in harness.load("scaling") if r["d"] != "free"]
    if not rows:
        return
    _style()
    families = [f for f in scaling.FAMILIES if any(r["family"] == f for r in rows)]
    ns = sorted({r["n"] for r in rows})
    ds = sorted({r["d"] for r in rows})
    norm = matplotlib.colors.LogNorm(*_decades(rows))
    fig, axes = _grid(len(families))
    image = None
    for ax, family in zip(axes, families):
        cell = {(r["n"], r["d"]): r for r in rows if r["family"] == family}
        grid = [
            [
                cell[n, d]["time"] if (n, d) in cell and cell[n, d]["status"] == "ok" else math.nan
                for n in ns
            ]
            for d in ds
        ]
        image = ax.imshow(grid, cmap="Blues", norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(ns))[::2], [f"4^{int(math.log(n, 4))}" for n in ns][::2])
        ax.set_yticks(range(len(ds)), [str(d) if d < 2**16 else "2^20" for d in ds])
        ax.set_xlabel("n")
        ax.set_ylabel("d")
        ax.set_title(family, fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(image, ax=axes, label="CPU s (median)", shrink=0.8)
    _save(fig, "scaling_heatmap")


def plot_size_nodes() -> None:
    """Every solved case's median CPU seconds against its diagram's node count, by tool."""
    rows = [r for r in harness.load("size") if r["status"] == "ok"]
    if not rows:
        return
    _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for tool in size.TOOLS:
        mine = [r for r in rows if r["tool"] == tool]
        ax.plot([r["nodes"] for r in mine], [r["time"] for r in mine], "o",
                color=TOOL_COLOURS[tool], alpha=0.6, mec="none", label=TOOL_LABELS[tool])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes in the checked diagram(s)")
    ax.set_ylabel("CPU s (median)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    _save(fig, "size_nodes")


def plot_size_solved() -> None:
    """The share of (w, seed) cases each tool settles within the timeout, against gate count."""
    rows = harness.load("size")
    if not rows:
        return
    _style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for tool in size.TOOLS:
        mine = [r for r in rows if r["tool"] == tool]
        xs = sorted({r["s"] for r in mine})
        share = [
            sum(r["status"] == "ok" for r in mine if r["s"] == s)
            / max(1, sum(1 for r in mine if r["s"] == s))
            for s in xs
        ]
        ax.plot([2 * s for s in xs], share, "-o", color=TOOL_COLOURS[tool], label=TOOL_LABELS[tool])
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("gates in C C^-1")
    ax.set_ylabel("share settled")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    _save(fig, "size_solved")


def _all(*plots: Callable[[], None]) -> Callable[[], None]:
    return lambda: [p() for p in plots] and None


PLOTS: dict[str, Callable[[], None]] = {
    "scaling": _all(plot_scaling, plot_scaling_flatness, plot_scaling_heatmap),
    "size": _all(plot_size, plot_size_nodes, plot_size_solved),
}
