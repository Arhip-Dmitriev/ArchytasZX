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

"""Repeat statistics: the exact order-statistic (binomial) confidence interval of the median,
the adaptive stopping rule and the summary row of a sample."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _binom_cdf(k: int, m: int) -> float:
    """P(Binomial(m, 1/2) <= k)."""
    if k < 0:
        return 0.0
    return sum(math.comb(m, i) for i in range(min(k, m) + 1)) / 2**m


def median_ci_ranks(m: int, confidence: float) -> tuple[int, int] | None:
    """1-based ranks (j, m + 1 - j) of the order statistics bracketing the median with
    coverage at least ``confidence``, j as large as possible; None when no j covers."""
    alpha = 1.0 - confidence
    best = None
    for j in range(1, m // 2 + 1):
        if 2 * _binom_cdf(j - 1, m) <= alpha:
            best = j
        else:
            break
    return None if best is None else (best, m + 1 - best)


def median_ci(values: Sequence[float], confidence: float) -> tuple[float, float] | None:
    """The exact confidence interval of the median, or None with too few values."""
    ranks = median_ci_ranks(len(values), confidence)
    if ranks is None:
        return None
    s = sorted(values)
    return s[ranks[0] - 1], s[ranks[1] - 1]


def min_runs(confidence: float) -> int:
    """The least sample size for which :func:`median_ci_ranks` exists."""
    m = 1
    while median_ci_ranks(m, confidence) is None:
        m += 1
    return m


def precise(values: Sequence[float], confidence: float, precision: float) -> bool:
    """Whether the interval's half-width is within ``precision`` of the median."""
    ci = median_ci(values, confidence)
    if ci is None:
        return False
    med = float(np.median(values))
    half = (ci[1] - ci[0]) / 2
    return half <= precision * abs(med) if med else half == 0


def summary(values: Sequence[float], confidence: float | None = None) -> dict[str, Any]:
    """median, q1, q3, min, max, n, the raw values and, with ``confidence``, the median CI."""
    if not values:
        return {"n": 0, "values": []}
    arr = np.asarray(values, dtype=float)
    out: dict[str, Any] = {
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(values),
        "values": [float(v) for v in values],
    }
    if confidence is not None:
        ci = median_ci(values, confidence)
        out["ci_lo"], out["ci_hi"] = ci if ci is not None else (None, None)
    return out
