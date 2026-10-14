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

"""Tests for :mod:`archytaszx.repl.zxview`: each drawn diagram contracts to the state's own
amplitudes; symbolic and oversized states are refused."""

from __future__ import annotations

import numpy as np
import pytest

from archytaszx.diagram.validate import validate
from archytaszx.repl import zxview
from archytaszx.repl.lang import Interpreter
from archytaszx.semantics.contract_numeric import contract

STATES = [
    "|s> = |0>",
    "|s> = |1>",
    "|s> = H|0>",
    "|s> = |i>",
    "|s> = |-i>",
    "|s> = CNOT (H@I)|0>|0>",
    "|s> = 1/2|0>|0> + sqrt(3)/2 |1>|1>",
    "|s> = 3/5 |0>|1> - 4/5 i |1>|0>",
    "|s> = 2|0>|1>|1> + |1>|0>|1>",
    "|s> = 5|1>|1> + |0>|1>",
    "|s> = H^@3 |0>^@3",
    "|s>_3 = |2>_3 @ |0>_3 + |1>_3 @ |1>_3",
    "|s> = |1>|0>_3 + |0>|2>_3",
]


def state(source: str):  # type: ignore[no-untyped-def]
    interp = Interpreter(style="ascii")
    interp.execute(source)
    return interp.states["s"]


@pytest.mark.parametrize("source", STATES)
def test_diagram_contracts_to_the_state(source: str) -> None:
    s = state(source)
    expected = np.array([complex(a) for a in zxview.amplitudes(s)])
    d = zxview.diagram(s)
    assert validate(d).is_valid
    assert np.allclose(contract(d).tensor.reshape(-1), expected)


def test_listing_names_the_generators() -> None:
    text = zxview.listing(state("|s> = 3/5 |0>|1> - 4/5 i |1>|0>"))
    assert text.startswith("graph:")
    for generator in ("  W  ", "  Z  ", "  X  ", "  S  "):
        assert generator in text


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("var n\n|s> = |0>^@n", "symbolic"),
        ("|s> = H^@7 |0>^@7", "basis states"),
    ],
)
def test_refusals(source: str, match: str) -> None:
    with pytest.raises(zxview.ZXViewError, match=match):
        zxview.diagram(state(source))
