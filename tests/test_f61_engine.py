"""Regression tests for the F61 engine items."""

from __future__ import annotations

from fractions import Fraction

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.pathsum import SQRT2, Cyclo
from archytaszx.algebra.phase import INDEX, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, chain_splitted, instantiate_symbol, merge, peel_one
from archytaszx.diagram.generators import (
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.mixed import double
from archytaszx.diagram.scalable import ScalableGrammarError, strip, to_scalable
from archytaszx.diagram.validate import validate
from archytaszx.rewrite import rules_axioms as R
from archytaszx.rewrite.match import find_cap_matches, find_triangle_matches
from archytaszx.rewrite.match_template import TemplateAxiom
from archytaszx.rewrite.match_zxw import find_w_identity_matches
from archytaszx.rewrite.rule import RewriteError
from archytaszx.rewrite.rules_library import Z_FUSION_PRIME_D, ZX_CAP_SWAPPED
from archytaszx.semantics.check import compare
from archytaszx.semantics.contract_symbolic import (
    ReplicatedBlock,
    ReplicatedFactor,
    SymbolicTensor,
    _mark_symbol,
)
from archytaszx.semantics.periodic import PeriodicError, chain_period


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


def _states(count: int) -> tuple[Diagram, list[NodeId]]:
    """``count`` one-output Z states on the boundary."""
    dim = Dim.concrete(2)
    diagram = Diagram()
    ids = [diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim]) for _ in range(count)]
    diagram.set_boundary_outputs([PortRef(i, Direction.OUTPUT, 0) for i in ids])
    return diagram, ids


def test_merge_moves_children_to_the_merged_box() -> None:
    diagram, (a, c, b) = _states(3)
    one = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({a, c}))
    two = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({b}))
    child = diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({a}), parent=one)
    assert not validate(diagram).errors
    out = merge(diagram, one, two)
    assert out.bang_boxes[child].parent in out.bang_boxes
    assert not validate(out).errors


def test_peel_one_on_a_multi_port_box() -> None:
    dim = Dim.concrete(2)
    diagram = Diagram()
    a = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim, dim])
    x = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    diagram.set_boundary_inputs([PortRef(a, Direction.INPUT, 0)])
    diagram.set_boundary_outputs(
        [PortRef(a, Direction.OUTPUT, i) for i in range(3)]
        + [PortRef(x, Direction.OUTPUT, 1), PortRef(x, Direction.OUTPUT, 0)]
    )
    scope = frozenset(
        {
            PortRef(a, Direction.OUTPUT, 0),
            PortRef(a, Direction.OUTPUT, 2),
            PortRef(x, Direction.OUTPUT, 1),
        }
    )
    box = diagram.add_bang_box(Mult.symbol("n") + Mult.concrete(1), port_scope=scope)
    peeled = peel_one(diagram, box).diagram
    assert len(peeled.bang_boxes[box].port_scope) == 3
    for k in (0, 2):
        assert compare(diagram, peeled, {"n": k}).matched


def test_peel_one_copies_leave_out_a_regrown_hub() -> None:
    dim = Dim.concrete(2)
    diagram = Diagram()
    hub = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    inner = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(hub, Direction.OUTPUT, 0), PortRef(inner, Direction.INPUT, 0))
    diagram.set_boundary_inputs([PortRef(hub, Direction.INPUT, 0)])
    box = diagram.add_bang_box(Mult.symbol("n") + Mult.concrete(1), node_scope=frozenset({inner}))
    result = peel_one(diagram, box)
    assert len(result.copy_node_ids) == 1
    (copy,) = result.copy_node_ids
    assert result.diagram.nodes[copy].generator_type.name == X_SPIDER.name


def test_chain_splitted_keeps_the_parent_link() -> None:
    diagram, (a, b, c) = _states(3)
    parent = diagram.add_bang_box(Mult.symbol("m"), node_scope=frozenset({a, b, c}))
    box = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({a, b}), parent=parent)
    out = chain_splitted(diagram, box, [frozenset({a}), frozenset({b})])
    made = [i for i in out.bang_boxes if i not in diagram.bang_boxes]
    assert [out.bang_boxes[i].parent for i in made] == [parent]


def test_strip_gives_a_legless_spider_its_zero_phase() -> None:
    dim = Dim.concrete(3)
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    diagram.set_boundary_outputs([PortRef(z, Direction.OUTPUT, 0)])
    diagram.add_bang_box(Mult.symbol("n"), port_scope=frozenset({PortRef(z, Direction.OUTPUT, 0)}))
    stripped = strip(to_scalable(diagram), {"n": 0})
    (node,) = stripped.nodes.values()
    assert node.phase == PhaseVector(dim)
    assert compare(stripped, instantiate_symbol(diagram, "n", 0), {}).matched


def test_aligned_boxes_are_refused_by_to_scalable_and_kept_by_doubling() -> None:
    diagram, (a, b) = _states(2)
    root = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({a}))
    other = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({b}), align=root)
    try:
        to_scalable(diagram)
    except ScalableGrammarError:
        pass
    else:
        raise AssertionError("to_scalable dropped the alignment")
    doubled = double(diagram)
    aligned = [box for box in doubled.bang_boxes.values() if box.align is not None]
    assert len(aligned) == 1
    assert aligned[0].align in doubled.bang_boxes
    assert other in diagram.bang_boxes


def test_chain_period_sees_an_index_in_a_copy_scalar() -> None:
    diagram, (z,) = _states(1)
    diagram.add_bang_box(
        Mult.concrete(2), node_scope=frozenset({z}), copy_scalar=Scalar.symbol("n")
    )
    try:
        chain_period(diagram, "n")
    except PeriodicError:
        return
    raise AssertionError("an index read by a copy scalar passed as unread")


def test_converse_of_a_converse_keeps_the_free_defaults() -> None:
    axiom = next(a for a in vars(R).values() if isinstance(a, TemplateAxiom) and a.free)
    there = axiom.converse("there")
    assert set(there.converse_free) == set(axiom.free)
    assert set(there.converse("back").free) == set(axiom.free)


def test_z_fusion_prime_d_fuses_only_z_spiders() -> None:
    dim = Dim.concrete(3)
    for generator, count in ((X_SPIDER, 0), (Z_SPIDER, 1)):
        diagram = Diagram()
        a = diagram.add_node(generator, input_dims=[dim], output_dims=[dim])
        b = diagram.add_node(generator, input_dims=[dim], output_dims=[dim])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        diagram.set_boundary_inputs([PortRef(a, Direction.INPUT, 0)])
        diagram.set_boundary_outputs([PortRef(b, Direction.OUTPUT, 0)])
        assert len(Z_FUSION_PRIME_D.pattern.find_matches(diagram)) == count


def test_zx_cap_swapped_refuses_an_unswapped_match() -> None:
    dim = Dim.concrete(3)
    diagram = Diagram()
    state = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim])
    effect = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[])
    diagram.add_wire(PortRef(state, Direction.OUTPUT, 0), PortRef(effect, Direction.INPUT, 0))
    (match,) = find_cap_matches(diagram)
    try:
        ZX_CAP_SWAPPED.builder(diagram.copy(), match)
    except RewriteError:
        return
    raise AssertionError("zx_cap_swapped built an unswapped cap")


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
