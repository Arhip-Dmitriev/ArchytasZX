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

"""The Python embedding: Program, Value operators, naming and measurement."""

from __future__ import annotations

import numpy as np
import pytest

import archytaszx
from archytaszx.embed import Program, Value
from archytaszx.repl.lang import LangError


def test_default_program_is_shared() -> None:
    assert isinstance(archytaszx.azx, Program) and archytaszx.azx is archytaszx.azx
    assert archytaszx.Program is Program
    with pytest.raises(AttributeError):
        archytaszx.nothing  # noqa: B018


def test_symbolic_state_by_attribute_and_item() -> None:
    azx = Program()
    azx("var n")
    azx.psi = azx.H**azx.n | azx.ket(0) ** azx.n
    assert azx.psi.ascii() == "|psi> = 2^(-n/2) sum(x in 0..2^n){ |x> }"
    azx["|phi>"] = azx["|psi> - 2 |0><0|psi>"]
    assert azx.phi.unicode().startswith("|φ⟩ = 2^(-n/2)")
    assert azx.phi.latex().startswith("$|\\phi\\rangle")
    assert azx.phi._repr_latex_() == azx.phi.latex()


def test_operators() -> None:
    p = Program()
    zero, one = p.ket(0), p.ket(1)
    assert np.allclose((p.H | zero).vector(), [2**-0.5, 2**-0.5])
    assert (p.X | zero).ascii() == "|1>"
    assert (p.X @ p.X | zero @ one).ascii() == "|1> @ |0>"
    assert float(p.bra(0) | p.H | zero) == pytest.approx(2**-0.5)
    assert (2 * one + zero).ascii() == "|0> + 2 |1>"
    assert ((one - one) | p.bra(0)).value is not None
    assert int(p["2 + 3"]) == 5 and complex(p["i"]) == 1j
    assert (zero.dag() | zero).sympy() == 1


def test_matrix_and_names() -> None:
    p = Program()
    p.U = p["[[0, 1], [1, 0]]"]
    assert (p.U | p.ket(0)).ascii() == "|1>"
    p.t = 0.5
    assert str(p.t) == "1/2"
    p("var k")
    assert set(p.names()) >= {"U", "t", "k"}


def test_measure_returns_the_distribution_named_and_anonymous() -> None:
    p = Program()
    p.q = p.H | p.ket(0)
    y = p.measure | p.q
    assert y.ascii() == "[(k, 0.5) : k in 0..2]"
    assert p.q.ascii() == "|q><q| = 1/2 sum(k in 0..2){ |k><k| }"
    z = p.measure(p.ket(1) @ p.ket(0))
    assert z.ascii() == "[(2, 1)]" and int(z) == 2


def test_errors() -> None:
    p = Program()
    with pytest.raises(AttributeError):
        p.missing  # noqa: B018
    with pytest.raises(LangError):
        p.H | p.ket(0, 3)
    with pytest.raises(LangError):
        p.measure(p["2"])
    assert isinstance(p.H, Value)
