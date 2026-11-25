"""Regression tests for the remaining F61 engine items."""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, peel_one
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.mixed import outcome
from archytaszx.diagram.validate import _classify_symbol_role


def test_outcome_symbol_is_an_exponent_symbol() -> None:
    d = outcome("k", 3)
    (node,) = d.nodes.values()
    assert node.phase is not None
    (sym,) = node.phase.sympy_symbols()
    assert sym == sp.Symbol("k", integer=True, nonnegative=True)
    assert _classify_symbol_role(sym) == "exponent"


def test_extended_peel_copies_leave_out_a_regrown_hub() -> None:
    dim = Dim.concrete(2)
    for index, copy_scalar in (("j", None), (None, Scalar(sp.Integer(2)))):
        diagram = Diagram()
        hub = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
        inner = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
        diagram.add_wire(PortRef(hub, Direction.OUTPUT, 0), PortRef(inner, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(hub, Direction.INPUT, 0)])
        box = diagram.add_bang_box(
            Mult.symbol("n") + Mult.concrete(1),
            node_scope=frozenset({inner}),
            index=index,
            copy_scalar=copy_scalar,
        )
        result = peel_one(diagram, box)
        assert len(result.copy_node_ids) == 1, (index, copy_scalar)
        (copy,) = result.copy_node_ids
        assert result.diagram.nodes[copy].generator_type.name == X_SPIDER.name
