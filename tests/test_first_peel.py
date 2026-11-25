"""Tests for first-copy peels of indexed boxes with pair boxes and of out-to-in chains."""

from __future__ import annotations

import numpy as np
import pytest

from archytaszx.diagram.bangbox import Mult, instantiate_symbol, peel_one
from archytaszx.diagram.graph import BangBoxId, Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel
from archytaszx.semantics.contract_numeric import contract
from archytaszx.semantics.induction import rename_multiplicity
from archytaszx.semantics.prove import _with_scalar_index, induction_renamed, induction_successor

_PAIRS = """dim d
count n
v = Z(1 -> 3, d)
p = Zp(2 -> 0, d) phase{x, y: x*y/d^(m-j+1)}
repeat n as V index j: v
pairs P of V index m: p
left P: v.out1
right P: v.out2
wire v.out1 -- p.in0, v.out2 -- p.in1
inputs v.in0
outputs v.out0
"""

_FCA = """dim d = 2
count n
g0 = Z(2 -> 1, d)
g1 = F(d)
g2 = Z(1 -> 2, d)
g3 = Zp(1 -> 1, d) phase{x, y: d^(j - m - 1)*x*y}
g4 = Z(1 -> 1, d) phase{k: 2*d^(j - n)*k}
g5 = Z(1 -> 2, d)
g6 = F(d)
g7 = F(d)
g8 = F(d)
g9 = Z(2 -> 1, d)
g10 = Zp(1 -> 1, d) phase{x, y: -d^(j - m - 1)*x*y}
wire g0.out0 -- g1.in0, g1.out0 -- g2.in0, g2.out1 -- g3.in0, g3.out0 -- g0.in1
wire g2.out0 -- g4.in0, g4.out0 -- g5.in0, g5.out0 -- g6.in0, g6.out0 -- g7.in0
wire g7.out0 -- g8.in0, g8.out0 -- g9.in0, g5.out1 -- g10.in0, g10.out0 -- g9.in1
repeat n as C0 index j: g0, g1, g2, g4, g5, g6, g7, g8, g9
pairs P0 of C0 index m: g3
left P0: g2.out1
right P0: g0.in1
pairs P3 of C0 index m: g10
left P3: g5.out1
right P3: g9.in1
inputs g0.in0
outputs g9.out0
"""

_CHAIN = """dim d = 2
count n
h = Z(0 -> 1, literal 1)
b = B(d^j, d)
s = X(1 -> 1, d^n) phase{k: -2*k/d^n}
wire h.out0 -- b.in0, b.out0 -- s.in0
repeat n as V index j: b
chain V: b.out0 -> b.in0
inputs b.in1
outputs s.out0
"""


def _same_instances(text: str, box: str, d: int, top: int) -> None:
    value = Mult("k") + 1
    diagram = _with_scalar_index(rename_multiplicity(parse_dsl(text), "n", value), "n", value)
    owner = next(b for b, x in sorted(diagram.bang_boxes.items()) if x.pair_of is None)
    peeled = peel_one(diagram, owner, first=True).diagram
    for k in range(top):
        want = contract(instantiate_symbol(diagram, "k", k).substitute({"d": d})).tensor
        got = contract(instantiate_symbol(peeled, "k", k).substitute({"d": d})).tensor
        assert np.linalg.norm(want) > 0.5 and np.allclose(got, want), (box, k, d)


def test_first_peel_of_pair_boxes_commutes_with_instantiation() -> None:
    for d in (2, 3):
        _same_instances(_PAIRS, "V", d, 5 if d == 2 else 4)


def test_first_peel_of_fca_box_commutes_with_instantiation() -> None:
    _same_instances(_FCA, "C0", 2, 5)


def test_first_peel_of_out_to_in_chain_commutes_with_instantiation() -> None:
    _same_instances(_CHAIN, "V", 2, 5)


def _fca_source() -> tuple[Diagram, BangBoxId]:
    source = induction_renamed(parse_dsl(_FCA), "n", "k")
    owner = next(b for b, x in sorted(source.bang_boxes.items()) if x.pair_of is None)
    return source, owner


def test_repeel_relates_first_and_last_copy_peels() -> None:
    source, owner = _fca_source()
    moved = kernel.repeel(source, "k", frozenset(), frozenset(), frozenset({owner}))
    assert kernel._same(moved.lhs, peel_one(source, owner, first=True).diagram)
    last = induction_successor(parse_dsl(_FCA), "n", "k")
    assert last is not None and kernel._same(moved.rhs, last)
    for k in range(3):
        a = contract(instantiate_symbol(moved.lhs, "k", k).substitute({"d": 2})).tensor
        b = contract(instantiate_symbol(moved.rhs, "k", k).substitute({"d": 2})).tensor
        assert np.allclose(a, b)


def test_tampered_repeel_is_refused() -> None:
    source, owner = _fca_source()
    tampered = source.copy()
    node = next(n for n, x in sorted(tampered.nodes.items()) if x.phase is not None)
    tampered.set_phase(node, None)
    moved = kernel.repeel(tampered, "k", frozenset(), frozenset(), frozenset({owner}))
    last = induction_successor(parse_dsl(_FCA), "n", "k")
    assert last is not None
    real = kernel.refl(last)
    with pytest.raises(kernel.KernelError):
        kernel.trans(moved, real)
    with pytest.raises(kernel.KernelError):
        kernel.Theorem(object(), moved.lhs, real.rhs, moved.inference)
    chain = induction_renamed(parse_dsl(_CHAIN), "n", "k")
    (box,) = list(chain.bang_boxes)
    with pytest.raises(kernel.KernelError):
        kernel.repeel(chain, "k", frozenset({box}), frozenset(), frozenset({box}))


if __name__ == "__main__":
    test_first_peel_of_pair_boxes_commutes_with_instantiation()
    test_first_peel_of_fca_box_commutes_with_instantiation()
    test_first_peel_of_out_to_in_chain_commutes_with_instantiation()
    test_repeel_relates_first_and_last_copy_peels()
    test_tampered_repeel_is_refused()
    print("ok")
