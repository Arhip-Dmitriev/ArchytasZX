"""Tests for the pair-phase generator Zp and PairPhase."""

from __future__ import annotations

import itertools

import numpy as np

from archytaszx.algebra.phase import PairPhase
from archytaszx.diagram.bangbox import instantiate_symbol
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics.contract_numeric import contract
from archytaszx.semantics.decide import EqualityVerdict, decide_equal

_ZP = "dim d\np = Zp(2 -> 0, d) phase{{x, y: {expr}}}\ninputs p.in0, p.in1\n"

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


def test_parse_reads_a_pair_phase() -> None:
    node = next(iter(parse_dsl(_ZP.format(expr="x*y/d")).nodes.values()))
    assert node.generator_type.name == "Zp"
    assert isinstance(node.phase, PairPhase)
    assert node.phase.free_symbols == frozenset({"d"})


def test_pair_box_expansion_denotes_the_pair_phases() -> None:
    diagram = parse_dsl(_PAIRS)
    for n, d in ((2, 2), (3, 2), (2, 3)):
        tensor = contract(instantiate_symbol(diagram, "n", n).substitute({"d": d})).tensor
        diag = np.diagonal(tensor.reshape(d**n, d**n))
        for xs in itertools.product(range(d), repeat=n):
            turns = sum(xs[j] * xs[m] / d ** (m - j + 1) for j in range(n) for m in range(j + 1, n))
            index = int(np.ravel_multi_index(xs, (d,) * n))
            assert np.isclose(diag[index], np.exp(2j * np.pi * turns))


def test_decide_reads_zp_at_symbolic_d() -> None:
    edge = "dim d\nf = F(d)\nc = Z(2 -> 0, d)\nwire f.out0 -- c.in0\ninputs f.in0, c.in1\n"
    left = parse_dsl(_ZP.format(expr="x*y/d"))
    assert decide_equal(left, parse_dsl(edge + "scalar d^(1/2)\n")).verdict is EqualityVerdict.EQUAL
    assert decide_equal(left, parse_dsl(_ZP.format(expr="(x*y + d*x*y)/d"))).verdict is (
        EqualityVerdict.EQUAL
    )
    unequal = decide_equal(left, parse_dsl(_ZP.format(expr="2*x*y/d"))).verdict
    assert unequal is not EqualityVerdict.EQUAL


_TWO_PAIRS = """dim d
count n
v = Z(1 -> 5, d)
p = Zp(2 -> 0, d) phase{x, y: x*y/d^(m-j+1)}
q = Zp(2 -> 0, d) phase{x, y: -2*x*y/d^(m-j+1)}
repeat n as V index j: v
pairs P of V index m: p
pairs Q of V index m: q
left P: v.out1
right P: v.out2
left Q: v.out3
right Q: v.out4
wire v.out1 -- p.in0, v.out2 -- p.in1, v.out3 -- q.in0, v.out4 -- q.in1
inputs v.in0
outputs v.out0
"""


def test_two_pair_boxes_on_one_node_expand() -> None:
    diagram = parse_dsl(_TWO_PAIRS)
    for n, d in ((2, 2), (3, 2), (3, 3)):
        tensor = contract(instantiate_symbol(diagram, "n", n).substitute({"d": d})).tensor
        diag = np.diagonal(tensor.reshape(d**n, d**n))
        for xs in itertools.product(range(d), repeat=n):
            turns = -sum(
                xs[j] * xs[m] / d ** (m - j + 1) for j in range(n) for m in range(j + 1, n)
            )
            index = int(np.ravel_multi_index(xs, (d,) * n))
            assert np.isclose(diag[index], np.exp(2j * np.pi * turns))
