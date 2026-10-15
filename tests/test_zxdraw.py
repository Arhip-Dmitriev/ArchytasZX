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

"""Tests for :mod:`archytaszx.repl.zxdraw`: one layout for every renderer, the fallbacks, and the
ZX drawing flags of the shell."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from archytaszx.repl import display as D
from archytaszx.repl import zxdraw, zxview
from archytaszx.repl.lang import Interpreter
from archytaszx.repl.shell import VIEWS, _argument_parser, main

HORRIBLE = str(Path(__file__).resolve().parent.parent / "examples" / "horrible_state.azx")
FOURIER = "gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }\n"


def view(source: str) -> zxview.View:
    interp = Interpreter(style="ascii")
    interp.execute(source)
    return zxview.view(interp.states["s"])


def test_layout_is_deterministic() -> None:
    source = "|s> = 3/5 |0>|1> - 4/5 i |1>|0>"
    a, b = zxdraw.layout(view(source)), zxdraw.layout(view(source))
    assert [(k, i.col, i.row) for k, i in sorted(a.items.items())] == [
        (k, i.col, i.row) for k, i in sorted(b.items.items())
    ]
    assert zxdraw.grid(a) == zxdraw.grid(b)


def test_renderers_share_the_layout() -> None:
    lay = zxdraw.layout(view("|s> = CNOT (H@I)|0>|0>"))
    uni, asc = zxdraw.grid(lay, unicode=True), zxdraw.grid(lay, unicode=False)
    assert len(uni.splitlines()) == len(asc.splitlines())
    assert "●" in uni and "(z)" in asc and "out 1" in uni and "out 1" in asc
    assert zxdraw.figure(lay, dark_theme=True).startswith(b"\x89PNG")


def test_symbolic_fourier_draws_a_bang_box() -> None:
    lay = zxdraw.layout(view("var d, n\n" + FOURIER + "|s> = F^@n |1>_d^@n"))
    text = zxdraw.grid(lay)
    assert "┌── n ──┐" in text and "●k/d" in text and "d^(-n/2)" in text
    assert "+-- n --+" in zxdraw.grid(lay, unicode=False)


def test_large_states_are_summarised() -> None:
    kets = "".join(f"|{k % 7}>_7" for k in range(22))
    other = "".join(f"|{(3 * k) % 7}>_7" for k in range(22))
    v = view("var d, n\n" + FOURIER + f"|s> = F^@n (|0>_7^@22 + 2 {kets} - {other})")
    lay = zxdraw.layout(v, max_nodes=50)
    assert lay.summary and sum(i.kind == "group" for i in lay.items.values()) == 3
    assert "[term 1:" in zxdraw.grid(lay)


@pytest.fixture
def utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zxdraw, "_utf8", lambda: True)


def test_no_image_protocol_falls_back_to_unicode(
    monkeypatch: pytest.MonkeyPatch, utf8: None
) -> None:
    monkeypatch.setenv("ARCHYTASZX_IMAGES", "none")
    assert "●" in zxdraw.draw(view("|s> = H|0>"), "matplot")


def test_matplotlib_failure_falls_back_to_unicode(
    monkeypatch: pytest.MonkeyPatch, utf8: None
) -> None:
    monkeypatch.setenv("ARCHYTASZX_IMAGES", "kitty")

    def broken(*_a: object, **_k: object) -> bytes:
        raise ImportError("no matplotlib")

    monkeypatch.setattr(zxdraw, "figure", broken)
    assert "●" in zxdraw.draw(view("|s> = H|0>"), "matplot")


def test_image_protocol_draws_an_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHYTASZX_IMAGES", "kitty")
    assert zxdraw.draw(view("|s> = H|0>"), "matplot").startswith("\x1b_G")


def test_non_utf8_terminal_falls_back_to_ascii(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdout", io.TextIOWrapper(io.BytesIO(), encoding="latin-1"))
    out = zxdraw.draw(view("|s> = H|0>"), "unicode")
    assert "(z" in out and "●" not in out


def test_zx_is_zxmatplot() -> None:
    parser = _argument_parser()
    a, b = parser.parse_args(["--zx"]), parser.parse_args(["--zxmatplot"])
    assert VIEWS[a.mode] == VIEWS[b.mode]
    assert VIEWS[parser.parse_args(["--both"]).mode][1] == "matplot"
    assert VIEWS[parser.parse_args(["--raw"]).mode][0].value == "raw"


@pytest.mark.parametrize(("flag", "mark"), [("--zxunicode", "▷W"), ("--zxascii", ">W")])
def test_horrible_state_grids(
    flag: str, mark: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zxdraw, "_utf8", lambda: True)
    assert main([HORRIBLE, flag]) == 0
    out = capsys.readouterr().out
    assert mark in out and "[term 1:" in out and "out 21" in out


def test_horrible_state_zx_draws(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHYTASZX_IMAGES", "kitty")
    monkeypatch.setattr(D, "prepare", lambda *a, **k: None)
    assert main([HORRIBLE, "--zx"]) == 0
    assert "\x1b_G" in capsys.readouterr().out
