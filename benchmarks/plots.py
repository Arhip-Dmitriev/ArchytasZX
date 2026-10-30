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

"""Print-style plots: white background, Okabe-Ito colours, solid lines, PDF and 300 dpi PNG.

``scaling_n_d``: 3D, log n, d and log time, one line per d plus the free case.
``scaling_times``: every bound case's time, by family.
``size_gates``: time against the gate count of C C^-1, one panel per width.
``size_solved``: the share of cases each tool settles, against the gate count.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

from benchmarks import harness, scaling, size

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
SURFACE = "#ffffff"
INK = "#000000"
INK_2 = "#333333"
GRID = "#d9d9d9"
FREE = "#555555"
BLUE_RAMP = [
    matplotlib.colors.to_hex(matplotlib.colormaps["Blues"](0.6 + 0.4 * i / 9)) for i in range(10)
]
TOOL_COLOURS = {"azx-family": "#0072b2", "azx-qubit": "#009e73", "pyzx": "#d55e00"}
TOOL_MARKERS = {"azx-family": "o", "azx-qubit": "s", "pyzx": "^"}
TOOL_LABELS = {
    "azx-family": "ArchytasZX, d free, n copies",
    "azx-qubit": "ArchytasZX, d = 2",
    "pyzx": "PyZX, d = 2",
}
FAILED = ("timeout", "oom", "skipped", "inconclusive", "wrong", "error")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": "medium",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str, tight: bool = True) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    box = "tight" if tight else None
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches=box)
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches=box)
    plt.close(fig)


def _log_ticks(lo: float, hi: float) -> list[int]:
    return list(range(math.floor(lo), math.ceil(hi) + 1))


def _d_label(d: object) -> str:
    if d == "free":
        return "free"
    return f"$2^{{{int(math.log2(d))}}}$" if d >= 2**16 else str(d)


def plot_scaling() -> None:
    """One 3D panel per family: log10 n, d (and a free row) and log10 median CPU seconds; one
    line per d along n; a cross on the cap plane for every case that did not finish."""
    rows = harness.load("scaling")
    if not rows:
        return
    _style()
    plt.rcParams.update(
        {"font.size": 12, "axes.labelsize": 12.5, "xtick.labelsize": 11, "ytick.labelsize": 11}
    )
    families = [f for f in scaling.FAMILIES if any(r["family"] == f for r in rows)]
    bound = [r for r in rows if r["d"] != "free"]
    ns = sorted({r["n"] for r in bound})
    ds = sorted({r["d"] for r in bound})
    times = [r["time"] for r in rows if r["status"] == "ok" and r["time"]]
    zmin = math.floor(math.log10(min(times)))
    cap = max(math.log10(max(times)) + 0.5, zmin + 2)
    cols = 2
    nrows = -(-len(families) // cols)
    height = 5.6 * nrows + 1.0
    fig = plt.figure(figsize=(17.5, height))
    fig.subplots_adjust(
        left=0.0, right=0.96, bottom=0.01, top=1 - 1.0 / height, wspace=0.04, hspace=0.06
    )
    for k, family in enumerate(families):
        ax = fig.add_subplot(nrows, cols, k + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        mine = [r for r in rows if r["family"] == family]
        for y, d in enumerate(ds):
            colour = BLUE_RAMP[round(y * (len(BLUE_RAMP) - 1) / max(1, len(ds) - 1))]
            line = sorted((r for r in mine if r["d"] == d), key=lambda r: r["n"])
            done = [r for r in line if r["status"] == "ok"]
            ax.plot(
                [math.log10(r["n"]) for r in done],
                [y] * len(done),
                [math.log10(r["time"]) for r in done],
                color=colour,
                marker="o",
                markeredgecolor=SURFACE,
                markeredgewidth=0.5,
            )
            failed = [r for r in line if r["status"] in FAILED]
            ax.scatter(
                [math.log10(r["n"]) for r in failed],
                [y] * len(failed),
                [cap] * len(failed),
                color=colour,
                marker="x",
                s=26,
                linewidths=1.4,
                depthshade=False,
            )
        free = [r for r in mine if r["d"] == "free"]
        if free and free[0]["status"] == "ok":
            z = math.log10(free[0]["time"])
            ax.plot(
                [math.log10(ns[0]), math.log10(ns[-1])], [len(ds)] * 2, [z, z], color=FREE, lw=2.2
            )
        elif free:
            ax.scatter([math.log10(ns[0])], [len(ds)], [cap], color=FREE, marker="x", s=26)
        xt = list(range(len(ns)))[::2]
        ax.set_xticks([math.log10(ns[i]) for i in xt], [f"$4^{{{i}}}$" for i in xt])
        ax.set_xlim(math.log10(ns[0]), math.log10(ns[-1]))
        ax.set_yticks(range(len(ds) + 1), [_d_label(d) for d in [*ds, "free"]])
        ax.set_ylim(0, len(ds))
        zt = _log_ticks(zmin, cap)
        ax.set_zticks(zt, [f"{10.0**t:g}" for t in zt])
        ax.set_zlim(zt[0], cap)
        ax.set_xlabel("qudits n (log)")
        ax.set_ylabel("dimension d")
        ax.set_zlabel("CPU time (s, log)")
        ax.set_title(family, y=0.96, fontsize=15)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((2.1, 1.0, 0.85), zoom=1.3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(SURFACE)
            axis.pane.set_edgecolor(GRID)
            axis._axinfo["grid"]["color"] = GRID
    handles = [
        Line2D(
            [],
            [],
            color=BLUE_RAMP[-1],
            marker="o",
            label="n, d bound (light to dark: d = 2 to $2^{20}$)",
        ),
        Line2D([], [], color=FREE, label="n, d free"),
        Line2D(
            [], [], color=INK_2, marker="x", linestyle="none", label="did not finish (top plane)"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1 - 0.45 / height),
        fontsize=12.5,
    )
    fig.suptitle(
        "ArchytasZX: time over qudit count n and dimension d",
        y=1 - 0.1 / height,
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    _save(fig, "scaling_n_d", tight=False)


def _strip(ax: plt.Axes, rows: list[dict], names: dict[str, str], ms: float) -> None:
    """Per family: every bound (n, d) case's median CPU seconds as a dot and the family's median
    as a black bar; log seconds."""
    families = [f for f in scaling.FAMILIES if any(r["family"] == f for r in rows)]
    times = []
    for k, family in enumerate(families):
        mine = [
            r["time"]
            for r in rows
            if r["family"] == family and r["d"] != "free" and r["status"] == "ok"
        ]
        times += mine
        jitter = [k + 0.5 * ((j * 0.618) % 1 - 0.5) for j in range(len(mine))]
        ax.plot(jitter, mine, "o", color="#0072b2", alpha=0.35, mec="none", ms=ms)
        if mine:
            ax.plot([k - 0.32, k + 0.32], [statistics.median(mine)] * 2, "-", color=INK, lw=1.2)
    ax.set_yscale("log")
    lo = math.floor(math.log10(min(times)))
    hi = math.ceil(math.log10(max(times)))
    ax.set_ylim(10.0**lo, 10.0**hi)
    ax.set_yticks([10.0**k for k in range(lo, hi + 1)], [f"{10.0**k:g}" for k in range(lo, hi + 1)])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(families)), [names[f] for f in families], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(families) - 0.4)
    ax.set_ylabel("CPU s (median)")


def plot_scaling_times() -> None:
    """:func:`_strip` over every family, with full names."""
    rows = harness.load("scaling")
    if not rows:
        return
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    _strip(ax, rows, {f: f for f in scaling.FAMILIES}, 3.5)
    ax.set_title("n = 1 to $4^{10}$ and d = 2 to $2^{20}$", fontsize=9)
    fig.tight_layout()
    _save(fig, "scaling_times")


def _size_line(rows: list[dict]) -> tuple[list[int], list[float], list[int]]:
    """Gate counts and the median over seeds where most seeds settled; the rest's counts."""
    xs: list[int] = []
    ys: list[float] = []
    bad: list[int] = []
    for s in sorted({r["s"] for r in rows}):
        at = [r for r in rows if r["s"] == s]
        ok = [r["time"] for r in at if r["status"] == "ok"]
        if len(ok) * 2 > len(at):
            xs.append(2 * s)
            ys.append(statistics.median(ok))
        else:
            bad.append(2 * s)
    return xs, ys, bad


def plot_size() -> None:
    """One panel per width: the median over seeds of each case's median CPU seconds against
    the gate count of C C^-1, one line per tool; crosses on the top edge where most seeds did
    not settle."""
    rows = harness.load("size")
    if not rows:
        return
    _style()
    widths = sorted({r["w"] for r in rows})
    times = [r["time"] for r in rows if r["status"] == "ok" and r["time"]]
    if not times:
        return
    lo = 10.0 ** math.floor(math.log10(min(times)))
    top = 10.0 ** math.ceil(math.log10(max(times)) + 0.3)
    fig, axes = plt.subplots(1, len(widths), figsize=(2.6 * len(widths) + 0.4, 2.7), sharey=True)
    axes = list(axes) if len(widths) > 1 else [axes]
    for ax, w in zip(axes, widths):
        for tool in size.TOOLS:
            xs, ys, bad = _size_line([r for r in rows if r["w"] == w and r["tool"] == tool])
            colour = TOOL_COLOURS[tool]
            ax.plot(xs, ys, "-", marker=TOOL_MARKERS[tool], color=colour, label=TOOL_LABELS[tool])
            ax.plot(bad, [top] * len(bad), "x", color=colour, ms=5, mew=1.4, clip_on=False)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(lo, top)
        ax.set_title(f"{w} wires", fontsize=9)
        ax.set_xlabel("gates in $C\\,C^{-1}$")
    axes[0].set_ylabel("CPU s (median)")
    handles = [
        Line2D([], [], color=TOOL_COLOURS[t], marker=TOOL_MARKERS[t], label=TOOL_LABELS[t])
        for t in size.TOOLS
    ]
    handles.append(Line2D([], [], color=INK_2, marker="x", linestyle="none", label="not settled"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    _save(fig, "size_gates")


def plot_size_solved() -> None:
    """The share of (w, seed) cases each tool settles within the timeout, against gate count."""
    rows = harness.load("size")
    if not rows:
        return
    _style()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    for tool in size.TOOLS:
        mine = [r for r in rows if r["tool"] == tool]
        xs = sorted({r["s"] for r in mine})
        share = [
            sum(r["status"] == "ok" for r in mine if r["s"] == s)
            / max(1, sum(1 for r in mine if r["s"] == s))
            for s in xs
        ]
        ax.plot(
            [2 * s for s in xs],
            share,
            "-",
            marker=TOOL_MARKERS[tool],
            color=TOOL_COLOURS[tool],
            label=TOOL_LABELS[tool],
        )
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("gates in $C\\,C^{-1}$")
    ax.set_ylabel("cases settled")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "size_solved")


PAPER_WIDTH = 2.7
PAPER_HEIGHT = 2.0
PAPER_SIZE_WIDTH = 4
SHORT = {
    "state copy": "copy",
    "spider fusion": "fusion",
    "Fourier order 4": "$F^4$",
    "Fourier superposition": "Fourier sup.",
    "Bernstein-Vazirani": "BV",
    "Deutsch-Jozsa constant": "DJ const.",
    "Deutsch-Jozsa balanced": "DJ bal.",
}


def _paper_style() -> None:
    _style()
    plt.rcParams.update(
        {"font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.2}
    )


def plot_paper_scaling() -> None:
    """:func:`_strip` at half the paper's text width, with short family names."""
    rows = harness.load("scaling")
    if not rows:
        return
    _paper_style()
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH, PAPER_HEIGHT))
    _strip(ax, rows, SHORT, 2.2)
    fig.tight_layout(pad=0.2)
    _save(fig, "paper_scaling")


def plot_paper_size() -> None:
    """``size_gates`` for :data:`PAPER_SIZE_WIDTH` wires at half the paper's text width."""
    rows = [r for r in harness.load("size") if r["w"] == PAPER_SIZE_WIDTH]
    if not rows:
        return
    _paper_style()
    times = [r["time"] for r in rows if r["status"] == "ok" and r["time"]]
    if not times:
        return
    lo = 10.0 ** math.floor(math.log10(min(times)))
    top = 10.0 ** math.ceil(math.log10(max(times)) + 0.3)
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH, PAPER_HEIGHT))
    for tool in size.TOOLS:
        xs, ys, bad = _size_line([r for r in rows if r["tool"] == tool])
        colour = TOOL_COLOURS[tool]
        ax.plot(xs, ys, "-", marker=TOOL_MARKERS[tool], ms=3, color=colour, label=TOOL_LABELS[tool])
        ax.plot(bad, [top] * len(bad), "x", color=colour, ms=4, mew=1.1, clip_on=False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.set_xlabel("gates in $C\\,C^{-1}$")
    ax.set_ylabel("CPU s (median)")
    ax.legend(fontsize=5.5, loc="lower right")
    fig.tight_layout(pad=0.2)
    _save(fig, "paper_size")


def _all(*plots: Callable[[], None]) -> Callable[[], None]:
    def go() -> None:
        for p in plots:
            p()

    return go


PLOTS: dict[str, Callable[[], None]] = {
    "scaling": _all(plot_scaling, plot_scaling_times, plot_paper_scaling),
    "size": _all(plot_size, plot_size_solved, plot_paper_size),
}
