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

"""Tests for :mod:`archytaszx.repl.zxdraw` on engine diagrams: one layout for every renderer
and the bang-box frame."""

from __future__ import annotations

import pytest

from archytaszx.repl import zxdraw
from archytaszx.repl.parser import parse_dirac_source

EXAMPLE = "sum_{k=0}^{3-1} |k>^{30}; copy"


def view() -> zxdraw.View:
    return zxdraw.View(parse_dirac_source(EXAMPLE))


def test_layout_is_deterministic() -> None:
    assert zxdraw.layout(view()) == zxdraw.layout(view())


def test_text_renderers_draw_the_bang_box() -> None:
    unicode = zxdraw.draw(view(), "unicode")
    ascii_ = zxdraw.draw(view(), "ascii")
    assert "Ⓩ" in unicode and "── n ──" in unicode
    assert "(Z)" in ascii_ and "-- n --" in ascii_
    assert unicode.count("\n") == ascii_.count("\n")


def test_unknown_style() -> None:
    with pytest.raises(ValueError, match="unknown drawing style"):
        zxdraw.draw(view(), "nope")
