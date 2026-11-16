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

"""Qubit networks proved by the F_2 term checker of ``ArchytasZX.SymF2``."""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_diagram
from archytaszx.semantics import lean_f2 as Q
from archytaszx.semantics import lean_net as L
from archytaszx.semantics import lean_sym as S

PAIRS: dict[str, tuple[str, str]] = {
    "hadamard_twice": (
        "a = F(literal 2); b = F(literal 2); wire a.out0 -- b.in0",
        "a = Z(1 -> 1, literal 2)",
    ),
    "z_fusion": (
        "a = Z(1 -> 2, literal 2); b = Z(1 -> 1, literal 2); wire a.out0 -- b.in0",
        "a = Z(1 -> 2, literal 2)",
    ),
    "x_fusion": (
        "a = X(1 -> 2, literal 2); b = X(1 -> 1, literal 2); wire a.out0 -- b.in0",
        "a = X(1 -> 2, literal 2)",
    ),
    "colour_change": (
        (
            "a = F(literal 2); b = Z(1 -> 1, literal 2); c = F(literal 2); "
            "wire a.out0 -- b.in0; wire b.out0 -- c.in0"
        ),
        "a = X(1 -> 1, literal 2)",
    ),
}


def _pair(name: str) -> tuple[Diagram, Diagram]:
    left, right = PAIRS[name]
    return parse_diagram(left), parse_diagram(right)


def _par(nu: dict[int, int], m: int) -> int:
    out = m & 1
    for v, b in nu.items():
        if Q.tb(m, v + 1):
            out ^= b
    return out


def _term(t: Q.QTm | None, nu: dict[int, int]) -> float:
    """A term's value at the bits ``nu``."""
    if t is None:
        return 0.0
    q, k, dels, phs = t
    if any(_par(nu, d) for d in dels):
        return 0.0
    sign = -1.0 if sum(_par(nu, u) * _par(nu, w) for u, w in phs) % 2 else 1.0
    return float(q * math.sqrt(2) ** k * sign)


def _node(kind: str, legs: list[int]) -> float:
    if kind == "Z":
        return 1.0 if len(set(legs)) <= 1 else 0.0
    if kind == "X":
        return math.sqrt(2) ** (-len(legs)) * (2.0 if sum(legs) % 2 == 0 else 0.0)
    return (-1.0 if legs[0] * legs[1] else 1.0) / math.sqrt(2)


def _net(nodes: Sequence[Q.QNode], internal: tuple[int, ...], nu: dict[int, int]) -> float:
    """A network's entry at the boundary bits ``nu``, by enumeration."""
    total = 0.0
    for vals in itertools.product((0, 1), repeat=len(internal)):
        env = {**nu, **dict(zip(internal, vals, strict=True))}
        v = 1.0
        for kind, os_, is_, _, _ in nodes:
            v *= _node(kind, [env[lab] for lab in (*os_, *is_)])
        total += v
    return total


@pytest.mark.parametrize("name", list(PAIRS))
def test_a_qubit_pair_takes_the_f2_checker(name: str) -> None:
    source = S.export_symbolic(*_pair(name))
    assert "qchain q0 IM0 R0" in source and "bridge_ok" in source and "equal_q" in source
    assert "sorry" not in source and "\naxiom " not in source and "symCheck" not in source


@pytest.mark.parametrize("name", list(PAIRS))
def test_the_final_term_is_the_network(name: str) -> None:
    left, right = _pair(name)
    ex = S._SymExport(left, right, None, {})
    for lnet in (ex.a.net, ex.b.net):
        nodes, internal, bnd = ex.net(lnet)
        for order in (nodes, nodes[::-1]):
            got = Q.run_q(order, internal, Q.hints(order, internal))
            assert got is not False
            t, sizes, _ = got
            assert sum(sizes) == len(nodes) and all(0 < k <= Q.CHAIN for k in sizes)
            for bits in itertools.product((0, 1), repeat=len(bnd)):
                nu = dict(zip(bnd, bits, strict=True))
                assert abs(_term(t, nu) - _net(nodes, internal, nu)) < 1e-9
                u = None if t is None else Q.canon_q(t)
                assert abs(_term(u, nu) - _term(t, nu)) < 1e-9


def test_an_unequal_qubit_pair_has_no_f2_proof() -> None:
    left = parse_diagram("a = Z(0 -> 1, literal 2)")
    right = parse_diagram("a = X(0 -> 1, literal 2)")
    with pytest.raises(L.NetUnsupported):
        S.export_symbolic(left, right)


def test_symbolic_dimensions_keep_the_normaliser() -> None:
    left = parse_diagram(
        "a = F(d); b = F(d); c = F(d); e = F(d); wire a.out0 -- b.in0; "
        "wire b.out0 -- c.in0; wire c.out0 -- e.in0"
    )
    right = parse_diagram("a = Z(1 -> 1, d)")
    source = S.export_symbolic(left, right)
    assert ("symCheck" in source or "equal_symS" in source) and "qchain" not in source


def test_the_packed_wires_read_back() -> None:
    wires = [((0, True, 0), (3, False, 1)), ((2, True, 1), (1, False, 0))]
    width, bits = 2, 5
    wb = Q.pack_w(width, bits, wires)
    for j, (a, b) in enumerate(wires):
        e = (wb >> (2 * bits * j)) % (1 << (2 * bits))
        assert e % (1 << bits) == Q.pkey(width, *a) and e >> bits == Q.pkey(width, *b)


# Lean


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


needs_lean = pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")


@pytest.mark.slow
@needs_lean
@pytest.mark.parametrize("name", list(PAIRS))
def test_lean_checks_a_qubit_pair(name: str, tmp_path: Path) -> None:
    source = S.export_symbolic(*_pair(name)) + "\n#print axioms ArchytasZX.Export.main_diagrams\n"
    passed, message = L.check(source, str(tmp_path / f"{name}.lean"))
    assert passed, message
    assert "[propext, Classical.choice, Quot.sound]" in message


@pytest.mark.slow
@needs_lean
def test_lean_rejects_a_tampered_scalar(tmp_path: Path) -> None:
    source = S.export_symbolic(*_pair("z_fusion"))
    head, tail = source.split("theorem qc : qCheck N0 N1 H0 H1 q0_fin q1_fin (", 1)
    bad = head + "theorem qc : qCheck N0 N1 H0 H1 q0_fin q1_fin (3 * " + tail
    passed, _ = L.check(bad, str(tmp_path / "bad.lean"))
    assert not passed
