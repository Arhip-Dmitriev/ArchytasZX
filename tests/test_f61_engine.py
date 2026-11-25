"""Regression tests for the F61 engine items."""

from __future__ import annotations

from fractions import Fraction

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.pathsum import SQRT2, Cyclo
from archytaszx.algebra.phase import INDEX, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import (
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite import rules_axioms as R
from archytaszx.rewrite.match import find_cap_matches, find_triangle_matches
from archytaszx.rewrite.match_zxw import find_w_identity_matches
from archytaszx.semantics.contract_symbolic import (
    ReplicatedBlock,
    ReplicatedFactor,
    SymbolicTensor,
    _mark_symbol,
)


def test_index_sum_nested_keeps_both_indices() -> None:
    d = Dim.concrete(3)
    w = Scalar.omega(d)
    nested = Scalar.index_sum(
        d,
        lambda k: Scalar.index_sum(
            d, lambda m: Scalar(w.to_sympy() ** (k.to_sympy() * m.to_sympy()))
        ),
    )
    expr = nested.to_sympy()
    assert len(expr.function.free_symbols) == 2
    assert sp.simplify(nested.simplify().to_sympy() - 3) == 0


def test_cyclo_equal_values_hash_equal() -> None:
    pairs = [
        (Cyclo(1, {0: Fraction(1)}), Cyclo(4, {0: Fraction(1)})),
        (Cyclo(3, {1: Fraction(1), 2: Fraction(1)}), Cyclo(1, {0: Fraction(-1)})),
        (SQRT2, SQRT2.lift(24)),
        (Cyclo(6, {1: Fraction(1)}), Cyclo(12, {2: Fraction(1)})),
    ]
    for a, b in pairs:
        assert a == b
        assert len({a, b}) == 1


def test_unit_index_compares_bound_value() -> None:
    three = Dim.concrete(3)
    unit = R.unit_vector(1, three)
    assert R.UnitIndex("j").match(unit, three, {"j": sp.Integer(2)}) is None
    assert R.UnitIndex("j").match(unit, three, {"j": sp.Integer(1)}) == {"j": 1}
    assert R.UnitIndex("j", -1).match(unit, three, {}) == {"j": 2}


def test_mixed_var_and_unit_index_equality() -> None:
    assert R.UnitIndex("j") == R.UnitIndex("j")
    assert R.UnitIndex("j") != R.UnitIndex("j", -1)
    assert len({R.MixedVar("a", 2), R.MixedVar("a", 2)}) == 1
    assert R.MixedVar("a", 2) != R.MixedVar("a", 3)


def test_symbolic_unit_vector_reduces_index_mod_d() -> None:
    d = Dim.symbol("d")
    top = d.to_sympy() - 1
    for index in (sp.Integer(-1), top):
        rule = R._rule_coefficient(R.unit_vector(index, d))
        assert rule.xreplace({INDEX: top}) == 1
    j = sp.Symbol("j", integer=True)
    assert R._rule_coefficient(R.unit_vector(-j, d)).xreplace({INDEX: d.to_sympy() - j}) == 1


def test_flatten_keeps_pair_factors_on_the_marked_branch() -> None:
    marked = ReplicatedFactor(Mult.concrete(2), Scalar(2), marked=Scalar(3), pairs=Scalar(5))
    tensor = SymbolicTensor((), Scalar(_mark_symbol(0)), (ReplicatedBlock(0, ()),), (marked,))
    assert tensor.flatten().entry.to_sympy() == 60
    plain = ReplicatedFactor(Mult.concrete(2), Scalar(2), pairs=Scalar(5))
    tensor = SymbolicTensor((), Scalar(1), (ReplicatedBlock(0, ()),), (plain,))
    assert tensor.flatten().entry.to_sympy() == 20


def test_diagram_substitute_reaches_box_multiplicities() -> None:
    diagram = Diagram()
    node = diagram.add_node(Z_SPIDER, [], [Dim.concrete(2)])
    box = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({node}))
    diagram.bind_parameter("n", 3)
    out = diagram.substitute({"n": 3})
    assert out.bang_boxes[box].multiplicity == Mult.concrete(3)
    assert "n" not in out.parameters


def _splice_onto_an_output(first: GeneratorType, second: GeneratorType | None) -> Diagram:
    """``first`` (then ``second``) from a boundary input into one output of a Z state."""
    dim = Dim.concrete(2)
    diagram = Diagram()
    head = diagram.add_node(first, input_dims=[dim], output_dims=[dim])
    tail = head
    if second is not None:
        tail = diagram.add_node(second, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(PortRef(head, Direction.OUTPUT, 0), PortRef(tail, Direction.INPUT, 0))
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    diagram.add_wire(PortRef(tail, Direction.OUTPUT, 0), PortRef(state, Direction.OUTPUT, 0))
    diagram.set_boundary_inputs([PortRef(head, Direction.INPUT, 0)])
    diagram.set_boundary_outputs([PortRef(state, Direction.OUTPUT, 1)])
    return diagram


def test_triangle_and_w_identity_keep_boundary_directions() -> None:
    assert find_triangle_matches(_splice_onto_an_output(TRIANGLE, TRIANGLE_INVERSE)) == ()
    assert find_triangle_matches(_splice_onto_an_output(TRIANGLE_INVERSE, TRIANGLE)) == ()
    assert find_w_identity_matches(_splice_onto_an_output(W_NODE, None)) == ()


def test_cap_matches_an_all_zero_phase() -> None:
    dim = Dim.concrete(3)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim], phase=PhaseVector(dim))
    effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    assert len(find_cap_matches(diagram)) == 1


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
