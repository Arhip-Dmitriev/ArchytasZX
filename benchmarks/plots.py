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

"""3D figures of time over (qudit count n, dimension d) from ``benchmarks/results/``.

One panel per algorithm: log10 n, d and log10 time on the three axes; one line per (tool, d)
along n; a cross on the cap plane for every run that did not finish.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmarks.harness import PLOTS, load

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
TOOL_STYLE = {
    "archytaszx": ("#2a78d6", "o", "ArchytasZX (light to dark: d = 2 to 16)"),
    "qiskit": ("#eb6834", "s", "Qiskit (d = 2 only)"),
    "pyzx": ("#1baf7a", "^", "PyZX (d = 2 only)"),
}
BLUE_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
FIG_WIDTH = 17.5
ASPECT_W, ASPECT_H = 3.0, 2.0
TITLES = {
    "ghz": "GHZ",
    "dj": "Deutsch-Jozsa",
    "grover": "Grover, one iteration",
    "shor": "Shor order finding, N = 15, a = 7",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 11,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
    }
)


def _log_ticks(lo: float, hi: float) -> list[int]:
    return list(range(math.floor(lo), math.ceil(hi) + 1))


def plot_algorithms() -> list[Path]:
    rows = [r for r in load("algorithms") if r["status"] != "unsupported"]
    if not rows:
        return []
    algs = [a for a in TITLES if any(r["family"] == a for r in rows)]
    times = [r["time"] for r in rows if r["status"] == "ok" and r["time"]]
    zmin, zmax = math.log10(min(times)), math.log10(max(times))
    cap = zmax + 0.5
    ns = sorted({r["n"] for r in rows})
    ds = sorted({r["d"] for r in rows})
    cols = 2 if len(algs) > 3 else len(algs)
    nrows = -(-len(algs) // cols)
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * ASPECT_H / ASPECT_W))
    fig.subplots_adjust(left=0.0, right=0.96, bottom=0.03, top=0.93, wspace=0.04, hspace=0.06)
    for k, alg in enumerate(algs):
        ax = fig.add_subplot(nrows, cols, k + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        lines: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for r in rows:
            if r["family"] == alg:
                lines[(r["tool"], r["d"])].append(r)
        for (tool, d), rs in sorted(
            lines.items(), key=lambda kv: (list(TOOL_STYLE).index(kv[0][0]), kv[0][1])
        ):
            color, marker, _ = TOOL_STYLE[tool]
            if tool == "archytaszx":
                color = BLUE_RAMP[round(ds.index(d) * (len(BLUE_RAMP) - 1) / max(1, len(ds) - 1))]
            y = ds.index(d)
            rs.sort(key=lambda r: r["n"])
            done = [r for r in rs if r["status"] == "ok" and r["time"]]
            xs = [math.log10(r["n"]) for r in done]
            ax.plot(
                xs,
                [y] * len(done),
                [math.log10(r["time"]) for r in done],
                color=color,
                marker=marker,
                markeredgecolor=SURFACE,
                markeredgewidth=0.6,
            )
            failed = [r for r in rs if r["status"] != "ok"]
            ax.scatter(
                [math.log10(r["n"]) for r in failed],
                [y] * len(failed),
                [cap] * len(failed),
                color=color,
                marker="x",
                s=22,
                depthshade=False,
            )
        xt = [t for t in _log_ticks(math.log10(ns[0]), math.log10(ns[-1])) if 10**t >= ns[0]]
        ax.set_xticks(xt, [f"{10**t:g}" for t in xt])
        ax.set_xlim(math.log10(ns[0]), math.log10(ns[-1]))
        ax.set_yticks(range(len(ds)), [str(d) for d in ds])
        ax.set_ylim(0, len(ds) - 1)
        zt = _log_ticks(zmin, cap)
        ax.set_zticks(zt, [f"{10.0**t:g}" for t in zt])
        ax.set_zlim(zt[0], cap)
        ax.set_xlabel("qudits n (log)")
        ax.set_ylabel("dimension d")
        ax.set_zlabel("CPU time (s, log)")
        ax.set_title(TITLES[alg], y=0.96)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((2.1, 1.0, 0.85), zoom=1.3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(SURFACE)
            axis.pane.set_edgecolor(GRID)
            axis._axinfo["grid"]["color"] = GRID
    handles = [
        Line2D([], [], color=c, marker=m, label=label) for c, m, label in TOOL_STYLE.values()
    ]
    handles.append(
        Line2D(
            [], [], color=INK_2, marker="x", linestyle="none", label="did not finish (top plane)"
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Algorithm output: time over qudit count n and dimension d (all concrete)",
        y=0.99,
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    PLOTS.mkdir(exist_ok=True)
    path = PLOTS / "algorithms_n_d.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def plot(suite: str) -> list[Path]:
    return {"algorithms": plot_algorithms}[suite]()
