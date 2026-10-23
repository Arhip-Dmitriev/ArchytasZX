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

"""Lean proofs over symbolic dimensions by the verified normaliser of symbolic sums."""

from __future__ import annotations

import cmath
import itertools
import math
import os
import random
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, _mult_symbol
from archytaszx.diagram.generators import W_NODE, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.repl.parser import parse_diagram
from archytaszx.semantics import kernel as K
from archytaszx.semantics import lean_net as L
from archytaszx.semantics import lean_sym as S
from archytaszx.semantics.check import compare

from .test_phase14_oracle import seeded_diagram
from .test_phase15_oracle import PROVED

PAIRS: dict[str, tuple[str, str]] = {
    "phase_fusion": (
        "a = Z(1 -> 1, d) phase {1: t}; b = Z(1 -> 1, d) phase {1: s}; wire a.out0 -- b.in0",
        "a = Z(1 -> 1, d) phase {1: t + s}",
    ),
    "x_fusion": (
        "a = X(1 -> 1, d) phase {1: t}; b = X(1 -> 1, d) phase {2: 1/3}; wire a.out0 -- b.in0",
        "a = X(1 -> 1, d) phase {1: t, 2: 1/3}",
    ),
    "fourier": (
        (
            "a = F(d); b = F(d); c = F(d); e = F(d); wire a.out0 -- b.in0; "
            "wire b.out0 -- c.in0; wire c.out0 -- e.in0"
        ),
        "a = Z(1 -> 1, d)",
    ),
    "triangle": ("a = T(d); b = Ti(d); wire a.out0 -- b.in0", "a = Z(1 -> 1, d)"),
    "two_dims": ("a = Z(1 -> 1, d); b = X(1 -> 1, e)", "a = Z(1 -> 1, d); b = Z(1 -> 1, e)"),
    "literal": (
        "a = X(1 -> 1, literal 3); b = X(1 -> 1, literal 3); wire a.out0 -- b.in0",
        "a = X(1 -> 1, literal 3)",
    ),
    "phase_rule": (
        (
            "a = Z(1 -> 1, d) phase {k: k/d}; b = Z(1 -> 1, d) phase {k: k^2/d}; "
            "wire a.out0 -- b.in0"
        ),
        "a = Z(1 -> 1, d) phase {k: (k + k^2)/d}",
    ),
}

CATALOGUE = ("bialgebra_4", "bialgebra_5", "bialgebra_6", "boxed_fusion", "two_index_fusion")

D = Dim.symbol("d")


def _loop(g: Diagram, fused: bool) -> list[NodeId]:
    """A closed Z loop worth ``d``: a state into an effect, or one legless spider."""
    if fused:
        return [g.add_node(Z_SPIDER, [], [], phase=PhaseVector(D))]
    a = g.add_node(Z_SPIDER, [], [D])
    b = g.add_node(Z_SPIDER, [D], [])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    return [a, b]


def _boxed(fused: bool, shape: str) -> Diagram:
    """A wire beside two boxed loops: nested, side by side, or nested in one symbol."""
    g = Diagram()
    w = g.add_node(Z_SPIDER, [D], [D])
    g.set_boundary_inputs([PortRef(w, Direction.INPUT, 0)])
    g.set_boundary_outputs([PortRef(w, Direction.OUTPUT, 0)])
    p, q = _loop(g, fused), _loop(g, fused)
    if shape == "two":
        g.add_bang_box(Mult("n"), node_scope=frozenset(p))
        g.add_bang_box(Mult.symbol("m") + 1, node_scope=frozenset(q))
    else:
        outer = g.add_bang_box(Mult("n"), node_scope=frozenset(p + q))
        g.add_bang_box(
            Mult("m" if shape == "nested" else "n"), node_scope=frozenset(q), parent=outer
        )
    return g


def _x_port(fused: bool) -> Diagram:
    """An X spider whose output a port box replicates, its input fed through another X."""
    g = Diagram()
    x = g.add_node(X_SPIDER, [D], [D], phase=PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))}))
    if fused:
        g.set_boundary_inputs([PortRef(x, Direction.INPUT, 0)])
    else:
        y = g.add_node(X_SPIDER, [D], [D])
        g.add_wire(PortRef(y, Direction.OUTPUT, 0), PortRef(x, Direction.INPUT, 0))
        g.set_boundary_inputs([PortRef(y, Direction.INPUT, 0)])
    ref = PortRef(x, Direction.OUTPUT, 0)
    g.set_boundary_outputs([ref])
    g.add_bang_box(Mult("k"), port_scope=frozenset({ref}))
    return g


def _w_port(fed: bool) -> Diagram:
    """A W node whose second output a port box replicates, its input fed through a Z wire."""
    g = Diagram()
    w = g.add_node(W_NODE, [D], [D, D])
    if fed:
        z = g.add_node(Z_SPIDER, [D], [D])
        g.add_wire(PortRef(z, Direction.OUTPUT, 0), PortRef(w, Direction.INPUT, 0))
        g.set_boundary_inputs([PortRef(z, Direction.INPUT, 0)])
    else:
        g.set_boundary_inputs([PortRef(w, Direction.INPUT, 0)])
    g.set_boundary_outputs([PortRef(w, Direction.OUTPUT, 0), PortRef(w, Direction.OUTPUT, 1)])
    g.add_bang_box(Mult("n"), port_scope=frozenset({PortRef(w, Direction.OUTPUT, 1)}))
    return g


def _scaled_family(boxed: bool) -> Diagram:
    """A wire beside n boxed loops, or beside the scalar d^n."""
    g = Diagram()
    w = g.add_node(Z_SPIDER, [D], [D])
    g.set_boundary_inputs([PortRef(w, Direction.INPUT, 0)])
    g.set_boundary_outputs([PortRef(w, Direction.OUTPUT, 0)])
    if boxed:
        g.add_bang_box(Mult("n"), node_scope=frozenset(_loop(g, False)))
    else:
        power = sp.Symbol("d", integer=True, positive=True) ** _mult_symbol("n")
        g.multiply_scalar(Scalar.from_dim(Dim.from_sympy(power)))
    return g


def _with_two(g: Diagram) -> Diagram:
    g.multiply_scalar(Scalar.rational(2))
    return g


def _with_pi(g: Diagram) -> Diagram:
    g.multiply_scalar(Scalar(sp.pi))
    return g


def _mixed_legs() -> Diagram:
    """A spider whose legs have dimensions d^p and d^q, one of them capped."""
    g = Diagram()
    p, q = D ** Dim.symbol("p"), D ** Dim.symbol("q")
    a = g.add_node(Z_SPIDER, [], [p, q])
    b = g.add_node(Z_SPIDER, [p], [])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    g.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 1)])
    return g


def _integer_phase(phased: bool) -> Diagram:
    """A qutrit state with an integer-symbol phase, or without it."""
    g = Diagram()
    j = Phase(sp.Symbol("j", integer=True))
    n = g.add_node(
        Z_SPIDER,
        [],
        [Dim.concrete(3)],
        phase=PhaseVector(Dim.concrete(3), {1: j}) if phased else None,
    )
    g.set_boundary_outputs([PortRef(n, Direction.OUTPUT, 0)])
    return g


BUILT = {
    "nested_boxes": lambda: (_boxed(False, "nested"), _boxed(True, "nested")),
    "two_multiplicities": lambda: (_boxed(False, "two"), _boxed(True, "two")),
    "nested_one_symbol": lambda: (_boxed(False, "same"), _boxed(True, "same")),
    "x_port_box": lambda: (_x_port(False), _x_port(True)),
    "integer_phase": lambda: (_integer_phase(True), _integer_phase(False)),
    "w_port_box": lambda: (_w_port(True), _w_port(False)),
    "scalar_power": lambda: (_scaled_family(True), _scaled_family(False)),
    "shared_opaque": lambda: tuple(_with_pi(parse_diagram(t)) for t in PAIRS["phase_fusion"]),
    "opaque_ratio": lambda: (
        _with_pi(_with_two(parse_diagram(PAIRS["phase_fusion"][0]))),
        _with_two(_with_pi(parse_diagram(PAIRS["phase_fusion"][1]))),
    ),
    "mixed_legs": lambda: (_mixed_legs(), _mixed_legs()),
}


def _pair(name: str):  # type: ignore[no-untyped-def]
    if name in PAIRS:
        return tuple(parse_diagram(t) for t in PAIRS[name])
    if name in BUILT:
        return BUILT[name]()
    return PROVED[name]()


def _source(name: str) -> str:
    return S.export_symbolic(*_pair(name))


@pytest.mark.parametrize("name", [*CATALOGUE, *PAIRS, *BUILT])
def test_states_and_proves_the_equality(name: str) -> None:
    source = _source(name)
    assert "theorem main" in source and "equal_sym" in source and "decide +kernel" in source
    assert "sorry" not in source and "\naxiom " not in source


def test_phases_are_quantified_and_floored() -> None:
    source = _source("phase_fusion")
    assert "theorem main (d : ℕ) (t : ℝ) (s : ℝ) (h_d : 2 ≤ d)" in source


def test_a_family_is_proved_by_induction() -> None:
    source = _source("boxed_fusion")
    assert "∀ m : ℕ, Equal ((F d).env (ksOf [m])) 1 N0 1 N1" in source
    assert "Equal.of_split_at" in source and "symCheck" in source


def test_port_boxes_step_by_the_port_lemma() -> None:
    for name in ("two_index_fusion", "x_port_box"):
        source = _source(name)
        assert "Equal.of_port_at" in source and "Spec.portOK_struct" in source


def test_every_multiplicity_is_inducted_on() -> None:
    source = _source("two_multiplicities")
    assert "∀ n m : ℕ, Equal ((F d).env (ksOf [n, m, (1 + m)]))" in source
    assert source.count("\n  induction ") == 2


def test_nested_boxes_of_one_symbol_get_one_value() -> None:
    source = _source("nested_one_symbol")
    assert "∀ n : ℕ, Equal ((F d).env (ksOf [n, n]))" in source


def test_new_shapes_take_their_lemmas() -> None:
    assert "rule_ex" in _source("phase_rule")
    assert "Equal.of_port_at" in _source("w_port_box")
    assert "Equal.congr_s" in _source("scalar_power")
    assert "Equal.smul" in _source("shared_opaque")


def test_a_dimension_constraint_is_a_hypothesis() -> None:
    source = _source("mixed_legs")
    assert "(hc0 : (d ^ p) = (d ^ q))" in source
    assert "assumes the dimension constraints" in source.split("-/")[0]


def test_a_theorem_falls_back_to_the_normaliser(monkeypatch: pytest.MonkeyPatch) -> None:
    from archytaszx.semantics.decide import decide_equal

    left, right = _pair("phase_fusion")
    theorem = K.from_decision(decide_equal(left, right), left, right)

    def refuse(*_args: object, **_kw: object) -> str:
        raise L.NetUnsupported("no lemma")

    monkeypatch.setattr(L, "export", refuse)
    assert "verified normaliser" in K.lean(theorem).split("-/")[0]


def test_integer_phase_symbols_drop_whole_turns() -> None:
    source = _source("integer_phase")
    assert "(p_j : ℤ)" in source and "phC_den_int" in source


def test_a_box_meeting_a_wire_is_refused() -> None:
    g = Diagram()
    a, _ = _loop(g, False)
    g.add_bang_box(Mult("n"), node_scope=frozenset({a}))
    with pytest.raises(L.NetUnsupported, match="meeting the rest at a wire"):
        S.export_symbolic(g, g.copy())


def test_unequal_diagrams_have_no_proof() -> None:
    left, right = parse_diagram("a = Z(0 -> 1, d)"), parse_diagram("a = X(0 -> 1, d)")
    with pytest.raises(L.NetUnsupported, match="normal forms"):
        S.export_symbolic(left, right)
    with pytest.raises(K.KernelError, match="no symbolic proof"):
        K.symbolic(left, right)


def test_a_missing_scalar_is_caught() -> None:
    left, right = PROVED["bialgebra_4"]()
    right = right.copy()
    right.multiply_scalar(right.scalar.__class__.rational(2))
    with pytest.raises(L.NetUnsupported):
        S.export_symbolic(left, right)


def test_kernel_artefact_carries_the_source() -> None:
    left, right = PROVED["bialgebra_5"]()
    sym = K.symbolic(left, right)
    assert K.lean(sym) == sym.source


# the mirror against numeric contraction


def _value(t: S.STm, dims: list[int], nu: dict[int, int]) -> complex:
    """A term's value by enumeration."""

    def dm(m: S.DMon) -> int:
        return S.dm_den(m, lambda i: dims[i] if i < len(dims) else 1)

    def poly(p: S.IPoly, env: dict[int, int]) -> int:
        out = 0
        for mon, c in p:
            v = c
            for x in mon:
                v *= env[x]
            out += v
        return out

    c = t[0]
    coef = complex(float(c[0])) * cmath.exp(2j * cmath.pi * float(c[1]))
    for i, k in enumerate(c[2]):
        coef *= (dims[i] ** 0.5) ** k
    for x in c[3]:
        coef *= x**0.5
    for a, m in c[5]:
        coef *= math.gcd(abs(a), dm(m))
    assert not c[4]
    total = 0j
    ranges = [range(dm(m)) for _, m in t[1]]
    for values in itertools.product(*ranges):
        env = dict(nu)
        env.update({x: v for (x, _), v in zip(t[1], values, strict=True)})
        body = 1 + 0j
        for m, p in t[2]:
            body *= 1 if poly(p, env) % dm(m) == 0 else 0
        for m, p in t[3]:
            body *= cmath.exp(2j * cmath.pi * poly(p, env) / dm(m))
        total += body
    return coef * total


def _ex_value(e: list[S.STm], dims: list[int], nu: dict[int, int]) -> complex:
    return sum((_value(t, dims, nu) for t in e), 0j)


@pytest.mark.parametrize("seed", range(12))
def test_the_mirror_keeps_the_value(seed: int) -> None:
    diagram = seeded_diagram(seed)
    if diagram is None:
        pytest.skip("no diagram")
    ex = S._SymExport(diagram, diagram, None, {})
    nodes, internal, bnd = ex.net(ex.a.net)
    e = S.tr_net(ex.fl, ex.dtf, ex.ptf, dict(ex.out.labs), nodes, internal)
    assert e is not None
    nf = list(S.nf_ex(ex.fl, e))
    rng = random.Random(seed)
    for d in (2, 3):
        nu = {b: rng.randrange(d) for b in bnd}
        assert abs(_ex_value(e, [d], nu) - _ex_value(nf, [d], nu)) < 1e-6


def test_the_translation_is_the_tensor() -> None:
    left, right = PROVED["bialgebra_4"]()
    assert compare(left, right, {"d": 3}).matched
    ex = S._SymExport(left, right, None, {})
    for lnet, scalar in ((ex.a.net, ex.s0), (ex.b.net, ex.s1)):
        nodes, internal, _ = ex.net(lnet)
        e = S.tr_net(ex.fl, ex.dtf, ex.ptf, dict(ex.out.labs), nodes, internal)
        assert e is not None
        values = [_ex_value(S.ex_mul(scalar, e), [3], {0: v}) for v in range(3)]
        if lnet is ex.a.net:
            first = values
        else:
            assert all(abs(a - b) < 1e-9 for a, b in zip(first, values, strict=True))
    assert S.rnorm(Fraction(1), Fraction(3, 4)) == (Fraction(-1), Fraction(1, 4))


# Lean


def _mathlib() -> bool:
    return os.path.isdir(os.path.join(L.lean_project(), ".lake", "packages", "mathlib"))


needs_lean = pytest.mark.skipif(not _mathlib(), reason="the ArchytasZX Lean project has no Mathlib")


@pytest.mark.slow
@needs_lean
@pytest.mark.parametrize("name", [*CATALOGUE, *PAIRS, *BUILT])
def test_lean_checks_the_equality(name: str, tmp_path: Path) -> None:
    source = _source(name) + "\n#print axioms ArchytasZX.Export.main\n"
    passed, message = L.check(source, str(tmp_path / f"{name}.lean"))
    assert passed, message
    assert "[propext, Classical.choice, Quot.sound]" in message
