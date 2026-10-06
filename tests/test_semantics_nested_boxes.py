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

"""Concrete and nested bang boxes through instantiate, contraction, peeling and induction,
checked against hand-built box-free expansions."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.diagram.bangbox import Mult, instantiate_symbol, kill, peel_one
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.rewrite.rule import RewriteDomainError
from archytaszx.semantics import induction
from archytaszx.semantics.check import instantiate, score
from archytaszx.semantics.contract_numeric import ContractDomainError, contract
from archytaszx.semantics.contract_symbolic import contract_symbolic

D = Dim(3)
S = induction.StepDischarge


def _loop(g: Diagram, dim: Dim = D) -> frozenset[NodeId]:
    a = g.add_node(Z_SPIDER, [], [dim])
    b = g.add_node(Z_SPIDER, [dim], [])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    return frozenset({a, b})


def _zx(g: Diagram) -> frozenset[NodeId]:
    a = g.add_node(Z_SPIDER, [], [D])
    b = g.add_node(X_SPIDER, [D], [])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    return frozenset({a, b})


def _loops(count: int) -> Diagram:
    g = Diagram()
    for _ in range(count):
        _loop(g)
    return g


def _value(g: Diagram, assignment: dict[str, int] | None = None) -> complex:
    return complex(score(g, assignment or {}).tensor)


def _maybe_instantiate(g: Diagram, name: str, value: int) -> Diagram:
    if any(name in box.multiplicity.free_symbols for box in g.bang_boxes.values()):
        return instantiate_symbol(g, name, value)
    return g


def _nested(outer: Mult, inner: Mult, *, inner_first: bool = False) -> Diagram:
    """``!outer(!inner(loop) ⊗ loop)``."""
    g = Diagram()
    if inner_first:
        a = _loop(g)
        b = _loop(g)
    else:
        b = _loop(g)
        a = _loop(g)
    o = g.add_bang_box(outer, node_scope=a | b)
    g.add_bang_box(inner, node_scope=a, parent=o)
    return g


def _three_level() -> Diagram:
    """``!a(!b(!c(loop) ⊗ loop) ⊗ loop)``."""
    g = Diagram()
    x, y, z = _loop(g), _loop(g), _loop(g)
    a = g.add_bang_box(Mult("a"), node_scope=x | y | z)
    b = g.add_bang_box(Mult("b"), node_scope=x | y, parent=a)
    g.add_bang_box(Mult("c"), node_scope=x, parent=b)
    return g


class TestConcreteBoxes:
    """A concrete multiplicity is expanded before numeric contraction."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_flat_concrete_box_matches_manual_copies(self, k: int) -> None:
        g = Diagram()
        g.add_bang_box(Mult(k), node_scope=_loop(g))
        assert np.isclose(_value(g), _value(_loops(k)))
        assert np.isclose(_value(g), 3**k)

    @pytest.mark.parametrize("b", [0, 1, 2])
    def test_concrete_outer_over_symbolic_inner(self, b: int) -> None:
        assert np.isclose(_value(_nested(Mult(0), Mult("b")), {"b": b}), 1)
        assert np.isclose(_value(_nested(Mult(2), Mult("b")), {"b": b}), _value(_loops(2 * b + 2)))

    @pytest.mark.parametrize("a", [0, 1, 2])
    def test_symbolic_outer_over_concrete_inner(self, a: int) -> None:
        assert np.isclose(_value(_nested(Mult("a"), Mult(2)), {"a": a}), _value(_loops(3 * a)))

    def test_contract_refuses_a_remaining_box(self) -> None:
        g = Diagram()
        g.add_bang_box(Mult(2), node_scope=_loop(g))
        with pytest.raises(ContractDomainError):
            contract(g)


class TestNestedInstantiation:
    """Nested boxes instantiate to the same expansion in either symbol order."""

    @pytest.mark.parametrize("inner_first", [False, True])
    @pytest.mark.parametrize("n,m", list(itertools.product(range(3), range(3))))
    def test_inner_symbol_sorting_first(self, inner_first: bool, n: int, m: int) -> None:
        g = _nested(Mult("n"), Mult("a"), inner_first=inner_first)
        assert np.isclose(_value(g, {"n": n, "a": m}), _value(_loops(n * m + n)))

    @pytest.mark.parametrize("n,m", list(itertools.product(range(3), range(3))))
    def test_both_orders_agree(self, n: int, m: int) -> None:
        g = _nested(Mult("n"), Mult("m"))
        outer_first = _maybe_instantiate(_maybe_instantiate(g, "n", n), "m", m)
        inner_first = _maybe_instantiate(_maybe_instantiate(g, "m", m), "n", n)
        assert not outer_first.bang_boxes and not inner_first.bang_boxes
        assert len(outer_first.nodes) == len(inner_first.nodes) == 2 * (n * m + n)

    @pytest.mark.parametrize("a,b,c", list(itertools.product(range(3), range(3), range(3))))
    def test_three_levels_match_closed_form(self, a: int, b: int, c: int) -> None:
        expected = _value(_loops(a * (b * (c + 1) + 1)))
        assert np.isclose(_value(_three_level(), {"a": a, "b": b, "c": c}), expected)

    def test_killing_a_box_with_a_nested_node_child(self) -> None:
        g = Diagram()
        y = g.add_node(
            Z_SPIDER, [], [], phase=PhaseVector(Dim(2), {1: Phase.turns(sp.Rational(1, 2))})
        )
        ab = _loop(g, Dim(1))
        o = g.add_bang_box(Mult("a"), node_scope=ab | {y})
        g.add_bang_box(Mult("b"), node_scope=ab, parent=o)
        killed = kill(g, BangBoxId(o))
        assert not killed.nodes and not killed.bang_boxes
        assert np.isclose(_value(g, {"a": 0, "b": 3}), 1)
        assert np.isclose(_value(g, {"a": 1, "b": 3}), 0)

    def test_peeling_an_inner_box_keeps_the_copy_in_the_outer_box(self) -> None:
        g = _nested(Mult("n"), Mult("m") + 1)
        inner = next(i for i, box in g.bang_boxes.items() if box.parent is not None)
        peeled = peel_one(g, inner).diagram
        for n, m in itertools.product(range(3), range(3)):
            assert np.isclose(_value(peeled, {"n": n, "m": m}), _value(g, {"n": n, "m": m}))

    def test_multiplicity_symbol_in_a_dimension_is_substituted(self) -> None:
        g = Diagram()
        _loop(g, Dim("n"))
        g.add_bang_box(Mult("n"), node_scope=_loop(g, Dim(2)))
        result = instantiate(g, {"n": 3})
        assert not result.bang_boxes
        assert all(port.dim.is_concrete for node in result.nodes.values() for port in node.outputs)
        assert np.isclose(_value(result), 3 * 2**3)


class TestSymbolicNested:
    """Symbolic contraction counts each nested box once."""

    @pytest.mark.parametrize("inner_first", [False, True])
    def test_two_levels(self, inner_first: bool) -> None:
        g = _nested(Mult("n"), Mult("m"), inner_first=inner_first)
        n, m = (sp.Symbol(s, integer=True, nonnegative=True, multiplicity=True) for s in "nm")
        assert sp.simplify(contract_symbolic(g).entry.to_sympy() - 3 ** (n * m + n)) == 0
        for nv, mv in itertools.product(range(3), range(3)):
            entry = contract_symbolic(g).entry.to_sympy().subs({n: nv, m: mv})
            assert np.isclose(complex(entry), _value(g, {"n": nv, "m": mv}))

    def test_three_levels(self) -> None:
        g = _three_level()
        entry = contract_symbolic(g).entry.to_sympy()
        symbols = {str(s): s for s in entry.free_symbols}
        for a, b, c in itertools.product(range(3), range(3), range(3)):
            value = entry.subs({symbols[k]: v for k, v in zip("abc", (a, b, c)) if k in symbols})
            assert np.isclose(complex(value), _value(g, {"a": a, "b": b, "c": c}))


class TestInductionNested:
    """The peeled-copy tier never proves a false nested family."""

    @staticmethod
    def _pair() -> tuple[Diagram, Diagram]:
        left = Diagram()
        a, b = _zx(left), _zx(left)
        o = left.add_bang_box(Mult("a"), node_scope=a | b)
        left.add_bang_box(Mult("b"), node_scope=a, parent=o)
        right = Diagram()
        a, b = _zx(right), _zx(right)
        right.add_bang_box(Mult("a"), node_scope=a | b)
        return left, right

    @pytest.mark.parametrize(
        "ladder",
        [(S.UNIFORM_REWRITE, S.INDUCTION_REWRITE, S.SYMBOLIC_CONTRACTION), (S.INDUCTION_REWRITE,)],
    )
    def test_dropped_inner_box_is_not_proved(
        self, ladder: tuple[induction.StepDischarge, ...]
    ) -> None:
        left, right = self._pair()
        assert not np.isclose(_value(left, {"a": 1, "b": 0}), _value(right, {"a": 1, "b": 0}))
        result = induction.prove_by_induction(
            left, right, index="a", base=1, witness={"b": 1}, ladder=ladder
        )
        assert result.verdict is not induction.Verdict.PROVED_INDUCTION

    def test_dropped_inner_port_box_is_not_proved(self) -> None:
        def build(nested: bool) -> Diagram:
            g = Diagram()
            z = g.add_node(Z_SPIDER, [], [D])
            ref = PortRef(z, Direction.OUTPUT, 0)
            g.set_boundary_outputs([ref])
            o = g.add_bang_box(Mult("a"), node_scope=frozenset({z}))
            if nested:
                g.add_bang_box(Mult("b"), port_scope=frozenset({ref}), parent=o)
            return g

        result = induction.prove_by_induction(
            build(True),
            build(False),
            index="a",
            base=0,
            witness={"b": 1},
            ladder=(S.UNIFORM_REWRITE, S.INDUCTION_REWRITE, S.SYMBOLIC_CONTRACTION),
        )
        assert result.verdict is not induction.Verdict.PROVED_INDUCTION

    def test_a_tier_domain_error_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: object, **_kwargs: object) -> induction.TierOutcome:
            raise RewriteDomainError("tier failed")

        monkeypatch.setitem(induction._DISPATCH, S.UNIFORM_REWRITE, boom)
        g = _nested(Mult("a"), Mult("b"))
        result = induction.prove_by_induction(
            g,
            g.copy(),
            index="a",
            base=0,
            witness={"b": 1},
            ladder=(S.UNIFORM_REWRITE, S.SYMBOLIC_CONTRACTION),
        )
        assert result.proved
        assert not result.tiers[0].settled
