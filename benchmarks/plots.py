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
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from benchmarks import harness, scaling, size

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
Row = dict[str, Any]
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
TOOL_COLOURS.update(
    {"azx-free": "#0072b2", "azx-bound": "#009e73", "qiskit": "#cc79a7", "dizx": "#e69f00"}
)
TOOL_MARKERS.update({"azx-free": "o", "azx-bound": "s", "qiskit": "D", "dizx": "v"})
TOOL_LABELS.update(
    {
        "azx-free": "ArchytasZX, n, d free",
        "azx-bound": "ArchytasZX, n and d bound",
        "qiskit": "Qiskit (optional arm), d = 2",
        "dizx": "DiZX, odd prime d",
    }
)
NOT_SETTLED = (*FAILED, "unsupported")
OKABE_ITO = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#f0e442", "#000000"]


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


def plots_dir() -> Path:
    """``$BENCH_PLOTS`` when set, else :data:`PLOTS_DIR`."""
    override = os.environ.get("BENCH_PLOTS")
    return Path(override).resolve() if override else PLOTS_DIR


def _save(fig: Figure, name: str, tight: bool = True) -> None:
    out = plots_dir()
    out.mkdir(parents=True, exist_ok=True)
    box = "tight" if tight else None
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches=box)
    fig.savefig(out / f"{name}.pdf", bbox_inches=box)
    plt.close(fig)


def _log_ticks(lo: float, hi: float) -> list[int]:
    return list(range(math.floor(lo), math.ceil(hi) + 1))


def _d_label(d: Any) -> str:
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


def _strip(ax: Axes, rows: list[Row], names: dict[str, str], ms: float) -> None:
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


def _size_line(rows: list[Row]) -> tuple[list[int], list[float], list[int]]:
    """Gate counts and the median over the seeds that settled; the counts no seed settled."""
    xs: list[int] = []
    ys: list[float] = []
    bad: list[int] = []
    for s in sorted({r["s"] for r in rows}):
        at = [r for r in rows if r["s"] == s]
        ok = [r["time"] for r in at if r["status"] == "ok"]
        if ok:
            xs.append(2 * s)
            ys.append(statistics.median(ok))
        else:
            bad.append(2 * s)
    return xs, ys, bad


def plot_size() -> None:
    """One panel per width: the median over seeds of each case's median CPU seconds against
    the gate count of C C^-1, one line per tool; crosses on the top edge where no seed
    settled."""
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
    _save(fig, "paper_scaling", tight=False)


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
        row = top / 2.2 ** size.TOOLS.index(tool)
        ax.plot(bad, [row] * len(bad), "x", color=colour, ms=4, mew=1.1, clip_on=False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.set_xlabel("gates in $C\\,C^{-1}$")
    ax.set_ylabel("CPU s (median)")
    ax.legend(fontsize=5.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout(pad=0.2)
    _save(fig, "paper_size", tight=False)


def _rows(suite: str) -> list[Row]:
    """The suite's rows; one line printed and none returned when there are none."""
    rows = harness.load(suite)
    if not rows:
        print(f"{suite}: no results in {harness.results_dir()}; nothing plotted")
    return rows


def _time_range(times: list[float]) -> tuple[float, float]:
    """A log axis range around ``times`` with room for the failure row on top."""
    good = [t for t in times if t and t > 0] or [1.0]
    lo = 10.0 ** math.floor(math.log10(min(good)))
    top = 10.0 ** math.ceil(math.log10(max(good)) + 0.3)
    return lo, top


def _colour(key: str, keys: list[str]) -> str:
    return TOOL_COLOURS.get(key) or OKABE_ITO[keys.index(key) % len(OKABE_ITO)]


def _failure_handle() -> Line2D:
    return Line2D([], [], color=INK_2, marker="x", linestyle="none", label="not settled (top)")


def _families_panel(ax: Axes, rows: list[Row], family: str, top: float) -> None:
    from benchmarks import families

    mine = [r for r in rows if r["family"] == family]
    for tool in families.TOOLS[1:]:
        line = sorted((r for r in mine if r["tool"] == tool), key=lambda r: r["n"])
        ok = [r for r in line if r["status"] == "ok"]
        bad = [r for r in line if r["status"] in NOT_SETTLED]
        colour = TOOL_COLOURS[tool]
        ax.plot(
            [r["n"] for r in ok],
            [r["time"] for r in ok],
            "-",
            marker=TOOL_MARKERS[tool],
            color=colour,
        )
        ax.plot(
            [r["n"] for r in bad], [top] * len(bad), "x", color=colour, ms=5, mew=1.2, clip_on=False
        )
    free = [r for r in mine if r["tool"] == "azx-free"]
    if free and free[0]["status"] == "ok":
        ax.axhline(free[0]["time"], color=TOOL_COLOURS["azx-free"], lw=1.6)
    elif free:
        ax.plot([1], [top], "x", color=TOOL_COLOURS["azx-free"], ms=6, mew=1.4, clip_on=False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")


def plot_families() -> None:
    """``families_crossover``: CPU seconds against n (log-log), one panel per family, azx-free
    as a horizontal line; ``families_paper``: every family in one panel (azx-bound, pyzx and
    the free lines); ``families_crossover.txt``: per family the smallest n at which azx-free
    is faster than each concrete tool."""
    rows = _rows("families")
    if not rows:
        return
    from benchmarks import families, workloads

    _style()
    lo, top = _time_range([r["time"] for r in rows if r["status"] == "ok"])
    names = [f for f in workloads.FAMILIES if any(r["family"] == f for r in rows)]
    cols = 4
    nrows = -(-len(names) // cols)
    fig, axes = plt.subplots(
        nrows, cols, figsize=(2.4 * cols, 2.5 * nrows + 0.5), sharey=True, squeeze=False
    )
    for ax, family in zip(axes.flat, names):
        _families_panel(ax, rows, family, top)
        ax.set_ylim(lo, top)
        ax.set_title(SHORT.get(family, family), fontsize=9)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("CPU s (median)")
    for ax in axes[-1, :]:
        ax.set_xlabel("n")
    handles = [
        Line2D(
            [],
            [],
            color=TOOL_COLOURS[t],
            marker=TOOL_MARKERS[t] if t != "azx-free" else None,
            label=TOOL_LABELS[t],
        )
        for t in families.TOOLS
    ]
    handles.append(_failure_handle())
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout()
    _save(fig, "families_crossover")
    _paper_style()
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH * 2, PAPER_HEIGHT * 1.3))
    for k, family in enumerate(names):
        colour = OKABE_ITO[k % len(OKABE_ITO)]
        for tool, style in (("azx-bound", "-"), ("pyzx", ":")):
            line = sorted(
                (r for r in rows if r["family"] == family and r["tool"] == tool),
                key=lambda r: r["n"],
            )
            ok = [r for r in line if r["status"] == "ok"]
            bad = [r for r in line if r["status"] in NOT_SETTLED]
            ax.plot(
                [r["n"] for r in ok],
                [r["time"] for r in ok],
                style,
                color=colour,
                marker=TOOL_MARKERS[tool],
                ms=2.5,
            )
            ax.plot([r["n"] for r in bad], [top] * len(bad), "x", color=colour, ms=3, clip_on=False)
        free = [
            r
            for r in rows
            if r["family"] == family and r["tool"] == "azx-free" and r["status"] == "ok"
        ]
        if free:
            ax.axhline(free[0]["time"], color=colour, lw=0.8, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.set_xlabel("n")
    ax.set_ylabel("CPU s (median)")
    handles = [
        Line2D([], [], color=OKABE_ITO[k % len(OKABE_ITO)], label=SHORT.get(f, f))
        for k, f in enumerate(names)
    ]
    handles += [
        Line2D([], [], color=INK_2, linestyle="-", marker="s", ms=2.5, label="ArchytasZX bound"),
        Line2D([], [], color=INK_2, linestyle=":", marker="^", ms=2.5, label="PyZX"),
        Line2D([], [], color=INK_2, lw=0.8, label="ArchytasZX free"),
    ]
    ax.legend(handles=handles, fontsize=5, ncol=2, loc="upper left")
    fig.tight_layout(pad=0.2)
    _save(fig, "families_paper", tight=False)
    lines = ["family                     tool        smallest n with azx-free faster"]
    for family, per in families.crossover(rows).items():
        for tool, n in per.items():
            lines.append(f"{family:<26} {tool:<11} {n if n is not None else 'none in the sweep'}")
    out = plots_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "families_crossover.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_bars(
    ax: Axes, labels: list[str], rows: list[Row], top: float, methods: list[str]
) -> None:
    """One horizontal bar per row at its time, coloured by method; a cross at ``top`` for an
    unsettled row."""
    for y, r in enumerate(rows):
        if r["status"] in NOT_SETTLED or r.get("time") is None:
            ax.plot([top], [y], "x", color=INK_2, ms=5, mew=1.2, clip_on=False)
            continue
        ax.barh(y, r["time"], color=_colour(str(r.get("method")), methods), height=0.7)
    ax.set_yticks(range(len(rows)), labels, fontsize=5.5)
    ax.set_xscale("log")
    ax.invert_yaxis()


def plot_refutations() -> None:
    """``refutations_methods``: the verdict's CPU seconds per variant (free n and d), coloured
    by the rung that refuted it; a cross where it was not refuted."""
    rows = _rows("refutations")
    if not rows:
        return
    _style()
    free = sorted((r for r in rows if r["n"] == "free"), key=lambda r: r["variant"])
    if not free:
        print("refutations: no free-n rows; nothing plotted")
        return
    methods = sorted({str(r.get("method")) for r in free})
    lo, top = _time_range([r["time"] for r in free if r.get("time")])
    fig, ax = plt.subplots(figsize=(5.2, 0.16 * len(free) + 1.0))
    _method_bars(ax, [r["variant"] for r in free], free, top, methods)
    ax.set_xlim(lo, top)
    ax.set_xlabel("CPU s to refute (median)")
    handles = [Line2D([], [], color=_colour(m, methods), lw=6, label=m) for m in methods]
    handles.append(_failure_handle())
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    fig.tight_layout()
    _save(fig, "refutations_methods")


def plot_breadth() -> None:
    """``breadth_methods``: one row per workload (n and d free): the rung that settled it and
    its CPU seconds; a cross for unknown, unsupported and failed rows."""
    rows = _rows("breadth")
    if not rows:
        return
    _style()
    free = sorted((r for r in rows if r["n"] == "free"), key=lambda r: r["workload"])
    methods = sorted({str(r.get("method")) for r in free if r["status"] == "ok"})
    lo, top = _time_range([r["time"] for r in free if r["status"] == "ok"])
    fig, ax = plt.subplots(figsize=(5.0, 0.2 * len(free) + 1.0))
    labels = [
        r["workload"].replace("_", " ") + (" (unsupported)" if r["status"] == "unsupported" else "")
        for r in free
    ]
    _method_bars(ax, labels, free, top, methods)
    ax.set_xlim(lo, top)
    ax.set_xlabel("CPU s (median)")
    handles = [Line2D([], [], color=_colour(m, methods), lw=6, label=m) for m in methods]
    handles.append(_failure_handle())
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    fig.tight_layout()
    _save(fig, "breadth_methods")


def plot_completeness() -> None:
    """``completeness_coverage``: per paper, the lemmas proved automatically, by exhaustive mode
    only, not found, and not encoded (stacked bars); ``completeness_cost``: CPU seconds against
    derivation length for auto and exhaustive, crosses on top for lemmas not found."""
    rows = _rows("completeness")
    if not rows:
        return
    _style()
    papers = ["P1", "P2"]
    kinds = ["auto", "exhaustive only", "not found", "not encoded"]
    colours = dict(zip(kinds, ["#0072b2", "#009e73", "#d55e00", "#bbbbbb"]))
    counts = {p: dict.fromkeys(kinds, 0) for p in papers}
    lemmas = {(r["paper"], r["lemma"]) for r in rows if r["paper"] in papers}
    for paper, lemma in sorted(lemmas):
        mine = {r["mode"]: r for r in rows if r["paper"] == paper and r["lemma"] == lemma}
        auto = mine.get("auto")
        if auto is not None and auto["status"] == "unsupported":
            kind = "not encoded"
        elif auto is not None and auto["status"] == "ok":
            kind = "auto"
        elif mine.get("exhaustive", {}).get("status") == "ok":
            kind = "exhaustive only"
        else:
            kind = "not found"
        counts[paper][kind] += 1
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    left = [0] * len(papers)
    for kind in kinds:
        widths = [counts[p][kind] for p in papers]
        ax.barh(range(len(papers)), widths, left=left, color=colours[kind], label=kind, height=0.6)
        left = [a + b for a, b in zip(left, widths)]
    ax.set_yticks(range(len(papers)), ["[P1]", "[P2]"])
    ax.set_xlabel("lemmas")
    ax.legend(fontsize=6, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "completeness_coverage")
    lemma_rows = [r for r in rows if r["mode"] in ("auto", "exhaustive")]
    lo, top = _time_range([r["time"] for r in lemma_rows if r["status"] == "ok"])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for mode, colour, marker in (("auto", "#0072b2", "o"), ("exhaustive", "#009e73", "s")):
        of_mode = [r for r in lemma_rows if r["mode"] == mode]
        ok = [r for r in of_mode if r["status"] == "ok" and r.get("derivation_length") is not None]
        bad = [r for r in of_mode if r["status"] in NOT_SETTLED and r["status"] != "unsupported"]
        ax.plot(
            [r["derivation_length"] for r in ok],
            [r["time"] for r in ok],
            marker,
            color=colour,
            ls="none",
            label=mode,
        )
        ax.plot([0] * len(bad), [top] * len(bad), "x", color=colour, ms=5, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.set_xlabel("derivation length (steps; x at 0: not found)")
    ax.set_ylabel("CPU s (median)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "completeness_cost")


def plot_lean() -> None:
    """``lean_cost``: Lean CPU seconds against file lines, coloured by Lean status;
    ``lean_coverage``: the share of exports per axiom class and per statement level."""
    rows = _rows("lean")
    if not rows:
        return
    _style()
    statuses = sorted({str(r.get("lean_status")) for r in rows})
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    times = [r["time"] for r in rows if r.get("time")]
    lo, top = _time_range(times)
    for status in statuses:
        mine = [r for r in rows if str(r.get("lean_status")) == status]
        ok = [r for r in mine if r.get("time") and r.get("file_lines")]
        bad = [r for r in mine if not (r.get("time") and r.get("file_lines"))]
        colour = _colour(status, statuses)
        ax.plot(
            [r["file_lines"] for r in ok],
            [r["time"] for r in ok],
            "o",
            color=colour,
            ms=3,
            label=status,
        )
        ax.plot([1] * len(bad), [top] * len(bad), "x", color=colour, ms=5, clip_on=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.set_xlabel("Lean file lines")
    ax.set_ylabel("Lean CPU s (median)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "lean_cost")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    for ax, field, title in (
        (axes[0], "axiom_class", "axiom class"),
        (axes[1], "statement_level", "statement level"),
    ):
        keys = sorted({str(r.get(field) or r.get("lean_status") or r["status"]) for r in rows})
        values = [
            sum(1 for r in rows if str(r.get(field) or r.get("lean_status") or r["status"]) == k)
            / len(rows)
            for k in keys
        ]
        ax.barh(range(len(keys)), values, color=[_colour(k, keys) for k in keys], height=0.6)
        ax.set_yticks(range(len(keys)), keys, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "lean_coverage")


def plot_trust() -> None:
    """``trust_mutations.txt`` and ``trust_mutations.csv``: the mutation table."""
    rows = _rows("trust")
    if not rows:
        return
    from benchmarks import trust

    text, csv_text = trust.table(rows)
    out = plots_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "trust_mutations.txt").write_text(text, encoding="utf-8")
    (out / "trust_mutations.csv").write_text(csv_text, encoding="utf-8")


def plot_ablation() -> None:
    """``ablation_matrix``: workloads by configuration; a filled cell is settled, its shade the
    log CPU seconds; an empty cell with a cross is not settled; "n/a" is unsupported."""
    rows = _rows("ablation")
    if not rows:
        return
    from benchmarks import ablation

    _style()
    configs = [c for c in ablation.CONFIGS if any(r["configuration"] == c for r in rows)]
    names = sorted({r["workload"] for r in rows})
    cell = {(r["workload"], r["configuration"]): r for r in rows}
    times = [r["time"] for r in rows if r["status"] == "ok" and r.get("time")]
    lo, hi = (math.log10(min(times)), math.log10(max(times))) if times else (0.0, 1.0)
    cmap = matplotlib.colormaps["Blues"]
    fig, ax = plt.subplots(figsize=(0.75 * len(configs) + 2.2, 0.2 * len(names) + 1.0))
    for y, name in enumerate(names):
        for x, config in enumerate(configs):
            r = cell.get((name, config))
            if r is None:
                continue
            if r["status"] == "ok" and r.get("time"):
                shade = 0.3 + 0.7 * (math.log10(r["time"]) - lo) / max(1e-9, hi - lo)
                ax.add_patch(
                    matplotlib.patches.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=cmap(shade))
                )
            elif r["status"] == "unsupported":
                ax.text(x, y, "n/a", ha="center", va="center", fontsize=5, color=INK_2)
            else:
                ax.plot([x], [y], "x", color="#d55e00", ms=4)
    ax.set_xlim(-0.5, len(configs) - 0.5)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xticks(range(len(configs)), configs, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=5.5)
    ax.set_title("ablation (not a user run): settled, shade = log CPU s", fontsize=8)
    fig.tight_layout()
    _save(fig, "ablation_matrix")


def plot_qudits() -> None:
    """``qudits_crossover``: CPU seconds against n (log-log), one panel per workload and d,
    azx-free as a horizontal line, crosses on top for unsettled and unsupported rows."""
    rows = _rows("qudits")
    if not rows:
        return
    from benchmarks import qudits

    _style()
    lo, top = _time_range([r["time"] for r in rows if r["status"] == "ok"])
    ds = sorted({r["d"] for r in rows if r["d"] != "free"})
    names = [w for w in qudits.WORKLOADS if any(r["workload"] == w for r in rows)]
    fig, axes = plt.subplots(
        len(names),
        len(ds),
        figsize=(2.3 * len(ds) + 0.5, 1.9 * len(names) + 0.5),
        sharey=True,
        squeeze=False,
    )
    for i, w in enumerate(names):
        free = [r for r in rows if r["workload"] == w and r["tool"] == "azx-free"]
        for j, d in enumerate(ds):
            ax = axes[i, j]
            for tool in ("azx-bound", "dizx"):
                line = sorted(
                    (r for r in rows if r["workload"] == w and r["tool"] == tool and r["d"] == d),
                    key=lambda r: r["n"],
                )
                ok = [r for r in line if r["status"] == "ok"]
                bad = [r for r in line if r["status"] in NOT_SETTLED]
                ax.plot(
                    [r["n"] for r in ok],
                    [r["time"] for r in ok],
                    "-",
                    marker=TOOL_MARKERS[tool],
                    color=TOOL_COLOURS[tool],
                )
                ax.plot(
                    [r["n"] for r in bad],
                    [top] * len(bad),
                    "x",
                    color=TOOL_COLOURS[tool],
                    ms=5,
                    clip_on=False,
                )
            if free and free[0]["status"] == "ok":
                ax.axhline(free[0]["time"], color=TOOL_COLOURS["azx-free"], lw=1.4)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(lo, top)
            ax.set_title(f"{w}, d = {d}", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("CPU s")
    for ax in axes[-1, :]:
        ax.set_xlabel("n")
    handles = [Line2D([], [], color=TOOL_COLOURS[t], label=TOOL_LABELS[t]) for t in qudits.TOOLS]
    handles.append(_failure_handle())
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), fontsize=7)
    fig.tight_layout()
    _save(fig, "qudits_crossover")


def plot_exhaustive() -> None:
    """``exhaustive_reach``: CPU seconds and diagrams visited against k (log scale), one line
    per seed; crosses on top where the target was not reached."""
    rows = _rows("exhaustive")
    if not rows:
        return
    _style()
    seeds = sorted({r["seed"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, field, label in (
        (axes[0], "time", "CPU s (median)"),
        (axes[1], "visited", "diagrams visited"),
    ):
        values = [float(r[field]) for r in rows if r.get(field)]
        lo, top = _time_range(values)
        for k, seed in enumerate(seeds):
            line = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["k"])
            ok = [r for r in line if r["status"] == "ok" and r.get(field)]
            bad = [r for r in line if r["status"] in NOT_SETTLED]
            colour = OKABE_ITO[k % len(OKABE_ITO)]
            ax.plot(
                [r["k"] for r in ok], [r[field] for r in ok], "-o", color=colour, ms=2.5, label=seed
            )
            ax.plot([r["k"] for r in bad], [top] * len(bad), "x", color=colour, ms=4, clip_on=False)
        ax.set_yscale("log")
        ax.set_ylim(lo, top)
        ax.set_xlabel("random rule steps k")
        ax.set_ylabel(label)
    axes[1].legend(fontsize=4.5, ncol=2)
    fig.tight_layout()
    _save(fig, "exhaustive_reach")


def plot_certificates() -> None:
    """``certificates_size``: kernel inferences and Lean lines against node count, coloured by
    workload group."""
    rows = _rows("certificates")
    if not rows:
        return
    _style()
    groups = sorted({r["group"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, field, label in (
        (axes[0], "inferences", "kernel inferences"),
        (axes[1], "lean_lines", "Lean lines"),
    ):
        values = [float(r[field]) for r in rows if r.get(field)]
        lo, top = _time_range(values)
        for group in groups:
            mine = [r for r in rows if r["group"] == group]
            ok = [r for r in mine if r["status"] == "ok" and r.get(field) and r.get("nodes")]
            bad = [r for r in mine if r["status"] in NOT_SETTLED or not r.get(field)]
            colour = _colour(group, groups)
            ax.plot(
                [r["nodes"] for r in ok],
                [r[field] for r in ok],
                "o",
                color=colour,
                ms=3,
                label=group,
            )
            ax.plot(
                [r.get("nodes") or 1 for r in bad],
                [top] * len(bad),
                "x",
                color=colour,
                ms=4,
                clip_on=False,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(lo, top)
        ax.set_xlabel("nodes (both sides)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "certificates_size")


def plot_limits() -> None:
    """``limits``: one panel per kind: CPU seconds against the parameter (contraction,
    nesting, symbols), and settled/conditional/unsettled counts (deferred, prime)."""
    rows = _rows("limits")
    if not rows:
        return
    _style()
    fig, axes = plt.subplots(
        1, 5, figsize=(15.0, 2.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.3, 1]}
    )
    lo, top = _time_range([r["time"] for r in rows if r["status"] == "ok"])
    for ax, kind, xlabel in zip(
        axes[:3],
        ("contraction", "nesting", "symbols"),
        ("symbolic phases", "box depth", "(count, dimension) pairs"),
    ):
        mine = [r for r in rows if r["kind"] == kind]
        names = sorted({r["workload"] for r in mine})
        for k, name in enumerate(names):
            line = sorted((r for r in mine if r["workload"] == name), key=lambda r: r["param"])
            ok = [r for r in line if r["status"] == "ok"]
            bad = [r for r in line if r["status"] in NOT_SETTLED]
            colour = OKABE_ITO[k % len(OKABE_ITO)]
            ax.plot(
                [r["param"] for r in ok],
                [r["time"] for r in ok],
                "-o",
                color=colour,
                ms=3,
                label=name,
            )
            ax.plot(
                [r["param"] for r in bad], [top] * len(bad), "x", color=colour, ms=5, clip_on=False
            )
        ax.set_yscale("log")
        ax.set_ylim(lo, top)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_title(kind, fontsize=9)
        if names:
            ax.legend(fontsize=5)
    axes[0].set_ylabel("CPU s (median)")
    deferred = [r for r in rows if r["kind"] == "deferred"]
    labels = [r["workload"] for r in deferred]
    outcome = [
        "conditional" if (r.get("assumptions") or 0) > 0 else str(r.get("verdict") or r["status"])
        for r in deferred
    ]
    keys = sorted(set(outcome))
    axes[3].barh(range(len(labels)), [1] * len(labels), color=[_colour(o, keys) for o in outcome])
    for y, o in enumerate(outcome):
        axes[3].text(0.5, y, o, ha="center", va="center", fontsize=6, color=SURFACE)
    axes[3].set_yticks(range(len(labels)), labels, fontsize=6)
    axes[3].set_xticks([])
    axes[3].set_title("deferred constraints", fontsize=9)
    axes[3].invert_yaxis()
    prime = [r for r in rows if r["kind"] == "prime"]
    params = [0, 3, 4]
    names_p = ["d free", "d = 3 (prime)", "d = 4"]
    settled = [sum(1 for r in prime if r["param"] == p and r["status"] == "ok") for p in params]
    unsettled = [sum(1 for r in prime if r["param"] == p and r["status"] != "ok") for p in params]
    axes[4].bar(range(3), settled, color="#0072b2", label="settled")
    axes[4].bar(range(3), unsettled, bottom=settled, color="#d55e00", label="not settled")
    axes[4].set_xticks(range(3), names_p, fontsize=6)
    axes[4].set_ylabel("workloads")
    axes[4].set_title("prime-only rules", fontsize=9)
    axes[4].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    _save(fig, "limits")


def _all(*plots: Callable[[], None]) -> Callable[[], None]:
    def go() -> None:
        for p in plots:
            p()

    return go


PLOTS: dict[str, Callable[[], None]] = {
    "scaling": _all(plot_scaling, plot_scaling_times, plot_paper_scaling),
    "size": _all(plot_size, plot_size_solved, plot_paper_size),
    "families": plot_families,
    "refutations": plot_refutations,
    "lean": plot_lean,
    "completeness": plot_completeness,
    "trust": plot_trust,
    "breadth": plot_breadth,
    "certificates": plot_certificates,
    "exhaustive": plot_exhaustive,
    "limits": plot_limits,
    "ablation": plot_ablation,
    "qudits": plot_qudits,
}
