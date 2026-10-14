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

"""Tests for :mod:`archytaszx.repl.display`: the three styles and mathtext validity."""

from __future__ import annotations

import io

import pytest
import sympy

from archytaszx.repl import display as D
from archytaszx.repl.lang import Interpreter
from tests.test_lang import SHOR


def plot_lines(source: str) -> list[str]:
    drawn: list[str] = []
    Interpreter(style="plot", plotter=lambda lines: drawn.extend(lines) or True).execute(source)
    return drawn


def test_shor_lines_render_in_mathtext() -> None:
    mathtext = pytest.importorskip("matplotlib.mathtext")
    lines = plot_lines(SHOR + "show(y, z, f, QFT)\n")
    assert len(lines) == 11
    for line in lines:
        mathtext.math_to_image(f"${line.strip()}$", io.BytesIO(), format="png")


def test_figure_builds_offscreen() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    fig = D.figure(plot_lines(SHOR))
    fig.canvas.draw()  # type: ignore[attr-defined]
    plt.close("all")
    assert D.plot_lines([r"|0\rangle"]) is False


@pytest.mark.parametrize(
    ("style", "expected"),
    [("ascii", "|psi_1>"), ("unicode", "|ψ_1⟩"), ("plot", r"|\psi_{1}\rangle")],
)
def test_labels(style: str, expected: str) -> None:
    assert D.Renderer(style).label(D.Label("ket", "psi_1")) == expected


@pytest.mark.parametrize(
    ("expr", "ascii", "uni"),
    [
        (sympy.exp(sympy.pi * sympy.I * sympy.Symbol("x") / 2), "e^(2*pi*i*x/4)", "e^{2πi·x/4}"),
        (1 / sympy.sqrt(sympy.Symbol("q")), "1/sqrt(q)", "1/√q"),
        (
            sympy.Mod(sympy.Symbol("a") ** sympy.Symbol("x"), sympy.Symbol("N")),
            "a^x % N",
            "a^x mod N",
        ),
        ((-1) ** (sympy.Symbol("x") * sympy.Symbol("y")), "(-1)^(x*y)", "(-1)^(x·y)"),
    ],
)
def test_expressions(expr: sympy.Expr, ascii: str, uni: str) -> None:
    assert D.Renderer("ascii").expr(expr) == ascii
    assert D.Renderer("unicode").expr(expr) == uni


def test_unknown_style() -> None:
    with pytest.raises(ValueError):
        D.Renderer("latex")
    with pytest.raises(ValueError):
        Interpreter(style="latex")
