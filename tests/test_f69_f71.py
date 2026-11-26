"""Regression tests for F69 (pair-phase conjugation), F70 (box copy scalar conjugation) and F71
(tree-box copies of port dims and child copy scalars)."""

from __future__ import annotations

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import INDEX, INDEX2, PairPhase, Phase
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, expand_concrete_boxes, instantiate_symbol
from archytaszx.diagram.generators import PAIR_PHASE, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.mixed import conj, double
from archytaszx.repl.parser import parse_listing
from archytaszx.semantics.contract_numeric import contract


def _pair_phase_diagram() -> Diagram:
    d = Dim(3)
    D = Diagram()
    n = D.add_node(PAIR_PHASE, [d], [d], phase=PairPhase(d, d, Phase(INDEX * INDEX2 / 3)))
    D.set_boundary_inputs([PortRef(n, Direction.INPUT, 0)])
    D.set_boundary_outputs([PortRef(n, Direction.OUTPUT, 0)])
    return D


def test_conj_negates_a_pair_phase() -> None:
    D = _pair_phase_diagram()
    assert np.allclose(contract(conj(D)).tensor, np.conj(contract(D).tensor))


def test_double_of_a_pair_phase_is_t_beside_conj_t() -> None:
    D = _pair_phase_diagram()
    T = contract(D).tensor
    got = contract(double(D)).tensor
    assert np.allclose(got, np.einsum("ai,bj->abij", T, np.conj(T)))


def _boxed_state(c: sp.Expr) -> Diagram:
    D = Diagram()
    n = D.add_node(Z_SPIDER, [], [Dim(2)])
    D.set_boundary_outputs([PortRef(n, Direction.OUTPUT, 0)])
    D.add_bang_box(Mult(3), node_scope=frozenset({n}), copy_scalar=Scalar(c))
    return D


def test_conj_conjugates_a_box_copy_scalar() -> None:
    D = _boxed_state(sp.I)
    T = contract(expand_concrete_boxes(D)).tensor
    U = contract(expand_concrete_boxes(conj(D))).tensor
    assert np.allclose(U, np.conj(T))


def test_double_box_copy_scalar_is_c_times_conj_c() -> None:
    D = _boxed_state(1 + sp.I)
    (box,) = double(D).bang_boxes.values()
    assert box.copy_scalar is not None
    assert sp.simplify(box.copy_scalar.to_sympy() - 2) == 0


TREE_DIMS = """graph: 3 nodes, 1 wires, 1 input -> 2 outputs
  scalar: 1
  parameters: none
  nodes:
    n1  Z  in: [2]  out: [2, 2]  boxes: !0
    n2  Z  in: []  out: [2^L]  boxes: !0
    n3  Z  in: [2^L]  out: []  boxes: !0
  wires:
    n2.out0 -- n3.in0
  inputs: [n1.in0]
  outputs: [n1.out0, n1.out1]
  bang boxes:
    !0  x n + 1  nodes: n1, n2, n3  tree: n1.in0 <- n1.out0, n1.out1  level: L  position: P"""


def test_a_tree_copy_sets_its_level_in_port_dims() -> None:
    got = instantiate_symbol(parse_listing(TREE_DIMS), "n", 1)
    dims = sorted(str(p.dim) for n in got.nodes.values() for p in (*n.inputs, *n.outputs))
    assert dims == ["2"] * 11 + ["4"] * 4


TREE_CHILD = """graph: 2 nodes, 0 wires, 1 input -> 2 outputs
  scalar: 1
  parameters: none
  nodes:
    n1  Z  in: [2]  out: [2, 2]  boxes: !0
    n2  Z  in: []  out: []  phase[2]: {}  turns  boxes: !0 > !1
  wires: none
  inputs: [n1.in0]
  outputs: [n1.out0, n1.out1]
  bang boxes:
    !0  x n + 1  nodes: n1, n2  tree: n1.in0 <- n1.out0, n1.out1  level: L  position: P
      !1  x 1  nodes: n2  copy scalar: 3^L"""


def test_a_tree_copy_sets_its_level_in_child_copy_scalars() -> None:
    got = expand_concrete_boxes(instantiate_symbol(parse_listing(TREE_CHILD), "n", 1))
    assert sp.simplify(got.scalar.to_sympy() - 3 * 9 * 9) == 0
