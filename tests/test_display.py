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
    Interpreter(style="plot", plotter=lambda lines: drawn.extend(lines) or "").execute(source)
    return drawn


def test_shor_lines_render_in_mathtext() -> None:
    mathtext = pytest.importorskip("matplotlib.mathtext")
    lines = plot_lines(SHOR + "show(y, z, f, QFT)\n")
    assert len(lines) == 10
    for line in lines:
        mathtext.math_to_image(f"${line.strip()}$", io.BytesIO(), format="png")


@pytest.mark.parametrize(
    ("env", "tty", "expected"),
    [
        ({"TERM": "xterm-kitty"}, True, "kitty"),
        ({"TERM": "xterm-ghostty"}, True, "kitty"),
        ({"TERM_PROGRAM": "iTerm.app"}, True, "iterm"),
        ({"TERM_PROGRAM": "vscode"}, True, "iterm"),
        ({"TERM": "foot"}, True, "sixel"),
        ({"TERM": "xterm-256color"}, True, "none"),
        ({"TERM": "xterm-kitty"}, False, "none"),
        ({"TERM": "xterm-256color", "ARCHYTASZX_IMAGES": "sixel"}, False, "sixel"),
    ],
)
def test_protocol_detection(env: dict[str, str], tty: bool, expected: str) -> None:
    assert D.protocol(env, tty) == expected


def test_theme() -> None:
    assert D.dark({}) and D.dark({"COLORFGBG": "15;0"})
    assert not D.dark({"COLORFGBG": "0;15"}) and not D.dark({"ARCHYTASZX_IMAGE_THEME": "light"})


@pytest.mark.parametrize("proto", ["kitty", "iterm", "sixel"])
def test_image_escapes(proto: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("ARCHYTASZX_IMAGES", proto)
    out = D.terminal_image(plot_lines(SHOR))
    assert out is not None
    prefix = {"kitty": "\x1b_Ga=T,f=100", "iterm": "\x1b]1337;File=inline=1", "sixel": "\x1bPq"}
    assert out.startswith(prefix[proto])
    png = D.render_png([r"|0\rangle"], dark_theme=False)
    assert png.startswith(b"\x89PNG")


def test_no_image_when_not_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARCHYTASZX_IMAGES", raising=False)
    assert D.terminal_image([r"|0\rangle"]) is None
    assert Interpreter(style="plot").execute("|a> = |0>\nshow(|a>)") == "|a⟩ = |0⟩"


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
