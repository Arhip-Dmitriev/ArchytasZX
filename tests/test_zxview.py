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

from archytaszx.diagram.bangbox import expand_concrete_boxes, instantiate_symbol
from archytaszx.diagram.validate import validate
from archytaszx.repl import zxview
from archytaszx.repl.lang import Interpreter
from archytaszx.semantics import pathsum as P
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


def tensor(d):  # type: ignore[no-untyped-def]
    return contract(expand_concrete_boxes(d)).tensor.reshape(-1)


@pytest.mark.parametrize("source", STATES)
def test_diagram_contracts_to_the_state(source: str) -> None:
    s = state(source)
    expected = np.array([complex(a) for a in zxview.amplitudes(s)])
    d = zxview.diagram(s)
    assert validate(d).is_valid
    assert np.allclose(tensor(d), expected)


def test_listing_names_the_generators() -> None:
    text = zxview.listing(state("|s> = 3/5 |0>|1> - 4/5 i |1>|0>"))
    assert text.startswith("graph:")
    for generator in ("  W  ", "  Z  ", "  X  ", "  S  "):
        assert generator in text


FOURIER = "gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }\n"

STRUCTURED = [
    "|s>_3 = sum(y in 0..3){ e^(2*pi*i*y/3) |y>_3 }",
    "|s>_3 = sum(k in 0..3){ |k>_3^@3 / (k + sqrt(2)) }",
    "var d, n\n" + FOURIER + "|s> = F^@n (|1>_3 - 2 i |2>_3)",
    "|s> = sum(x in 0..4){ sum(y in 0..2){ (-1)^(x*y) |x>|y> } }",
    "|s> = sum(x in 0..2){ sum(y in 0..2){ (-1)^(x*y) / (x + 2) |x>|y>|x> } } + |1>|1>|0>",
    "|s>_5 = sum(k in 0..3){ e^(i*k^2) |k>_5 } + 2|4>_5",
    "|s> = H^@7 |0>^@7",
]


def dense(s) -> np.ndarray:  # type: ignore[no-untyped-def]
    keys, amps = P.enumerate_state(s)
    out = np.zeros(int(np.prod(P.concrete_levels(s))), dtype=complex)
    np.add.at(out, keys.astype(np.int64), amps)
    return out


@pytest.mark.parametrize("source", STRUCTURED)
def test_structural_diagram_contracts_to_the_state(source: str) -> None:
    s = state(source)
    d = zxview.diagram(s)
    assert validate(d).is_valid
    assert np.allclose(tensor(d), dense(s))


@pytest.mark.parametrize(
    "source",
    [
        "|s> = H^@2 |0>|1> - 2 H^@2 |1>|1>",
        "|s> = sum(k in 0..2){ |k>^@2 / (k + 3) } + i H^@2 |1>|0>",
    ],
)
def test_sum_of_closed_forms_contracts(source: str) -> None:
    s = state(source)
    assert len(s.terms) == 2 and all(t.bound for t in s.terms)
    d = zxview.diagram(s)
    assert validate(d).is_valid
    assert np.allclose(tensor(d), dense(s))


DJ = (
    "func f(x) = bit(x, 0)\ngate U = |x>|y> -> |x>|y xor f(x)>\n"
    "|s> = (H^@n @ I) U (H^@n @ H) (|0>^@n @ |1>)\n"
)


@pytest.mark.parametrize(
    "source",
    [
        DJ,
        "|s> = |0>^@n + |1>^@n",
        "|s> = sum(x in 0..2^n){ e^(2*pi*i*x/2^n) |x> }",
        "|s> = sum(x in 0..2^n){ 1/(x + 1) |x> }",
        (
            "gate QFT = |x> -> 1/sqrt(2^n) sum(y in 0..2^n){ e^(2*pi*i*x*y/2^n) |y> }\n"
            "|s> = QFT |1>^@n"
        ),
    ],
)
def test_symbolic_diagram_instantiates_to_every_size(source: str) -> None:
    sym = state("var n\n" + source)
    d = zxview.diagram(sym)
    assert validate(d).is_valid
    for k in (1, 2, 3):
        concrete = state(f"var n = {k}\n" + source)
        boxed = {s for b in d.bang_boxes.values() for s in b.multiplicity.free_symbols}
        e = instantiate_symbol(d, "n", k) if "n" in boxed else d
        assert np.allclose(tensor(e.substitute({"n": k})), dense(concrete)), k


def test_drawing_does_not_depend_on_the_size() -> None:
    kinds = []
    for head in ("var n\n", "var n = 7\n", "var n = 60\n"):
        d = zxview.diagram(state(head + DJ))
        kinds.append(sorted(n.generator_type.name for n in d.nodes.values()))
        assert len(d.bang_boxes) == 1
    assert kinds[0] == kinds[1] == kinds[2]


def test_large_sum_has_a_diagram_of_term_size() -> None:
    kets = "".join(f"|{k % 7}>_7" for k in range(22))
    s = state("var d, n\n" + FOURIER + f"|s> = F^@n (|0>_7^@22 + 2 {kets})")
    d = zxview.diagram(s)
    assert validate(d).is_valid
    assert len(d.nodes) < 400


def test_symbolic_fourier_is_a_boxed_spider() -> None:
    s = state("var d, n\n" + FOURIER + "|s> = F^@n |1>_d^@n")
    d = zxview.diagram(s)
    assert len(d.bang_boxes) == 1 and len(d.nodes) == 1
    (node,) = d.nodes.values()
    assert node.phase is not None and node.phase.rule is not None
    concrete = instantiate_symbol(d, "n", 2).substitute({"d": 3, "n": 2})
    want = state("var d = 3\n" + FOURIER + "|s> = F^@2 |1>_3^@2")
    expected = np.array([complex(a) for a in zxview.amplitudes(want)])
    assert np.allclose(contract(concrete).tensor.reshape(-1), expected)


def test_bra_outputs_are_named() -> None:
    v = zxview.view(P.dagger(state("|s> = |0>|1>")))
    assert v.outputs == ["bra 0", "bra 1"]


def test_measured_state_is_the_decohered_graph() -> None:
    interp = Interpreter(style="ascii")
    interp.execute("var n\n|p> = H^@n |0>^@n\nvar z = measure|p>")
    v = zxview.view(interp.states["p"])
    assert v.outputs == ["ket 0", "bra 0"]
    (node,) = v.diagram.nodes.values()
    assert node.generator_type.name == "Z" and len(node.outputs) == 2
    assert len(v.diagram.bang_boxes) == 1


def test_density_matrix_of_a_product_draws_both_sides() -> None:
    interp = Interpreter(style="ascii")
    interp.execute("|r> = |+><+|")
    v = zxview.view(interp.states["r"])
    assert v.outputs == ["ket 0", "bra 0"] and len(v.diagram.nodes) == 2


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("|s> = |0> + |1>", "Z"),
        ("|s> = |0> - i |1>", "Z"),
        ("|s> = |1>_3 + |1>_3", "X"),
    ],
)
def test_small_sum_is_one_spider(source: str, kind: str) -> None:
    s = state(source)
    d = zxview.diagram(s)
    assert validate(d).is_valid
    assert [n.generator_type.name for n in d.nodes.values()] == [kind]
    assert np.allclose(contract(d).tensor.reshape(-1), dense(s))
