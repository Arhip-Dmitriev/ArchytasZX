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

"""Phase 17: the structural Dirac term evaluates to the oracle's tensor, and the plan's
done-whens (fusion round trip, single Z spider sum, Dirac rendering of every derivation state)."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable

import numpy as np
import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import BangBoxError, Mult, free_mult_symbols
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.repl import printer as P
from archytaszx.repl.parser import parse_dirac_source
from archytaszx.rewrite.engine import RewriteResult, apply, apply_until_fixpoint
from archytaszx.rewrite.rules_library import IDENTITY_REMOVAL, SPIDER_FUSION
from archytaszx.semantics.check import CheckError
from archytaszx.semantics.contract_numeric import ContractSizeError

from .dirac_eval import EvalSizeError, evaluate, full_env, oracle_tensor
from .test_scalable import BANG_BOX_FAMILIES

D = Dim("d")
DS = (2, 3)
SYMBOLS = {"theta": sp.Rational(1, 5), "phi": sp.Rational(2, 7), "c": 2}
Builder = Callable[[], Diagram]


def out(node: NodeId, i: int) -> PortRef:
    return PortRef(node, Direction.OUTPUT, i)


def inp(node: NodeId, i: int) -> PortRef:
    return PortRef(node, Direction.INPUT, i)


def node(
    d: Diagram,
    gen: GeneratorType,
    ins: int,
    outs: int,
    dim: Dim = D,
    phase: PhaseVector | None = None,
) -> NodeId:
    """Add a node; Z/X get a zero phase vector when none is given."""
    if phase is None and gen in (Z_SPIDER, X_SPIDER):
        phase = PhaseVector(dim)
    return d.add_node(gen, [dim] * ins, [dim] * outs, phase=phase)


def close(d: Diagram) -> Diagram:
    """Put every unwired port on the boundary in node/index order."""
    used = {end for w in d.wires for end in (w.a, w.b)}
    ins, outs = [], []
    for nid, n in sorted(d.nodes.items()):
        ins += [inp(nid, i) for i in range(n.num_inputs) if inp(nid, i) not in used]
        outs += [out(nid, i) for i in range(n.num_outputs) if out(nid, i) not in used]
    d.set_boundary_inputs(ins)
    d.set_boundary_outputs(outs)
    return d


def assert_oracle(diagram: Diagram, env: dict[str, object]) -> None:
    """dirac_term(diagram) evaluated at env equals the oracle tensor."""
    env = full_env(diagram, env)
    term = P.dirac_term(diagram)
    got = evaluate(term, env)
    want = oracle_tensor(diagram, env)
    assert got.shape == want.shape, (got.shape, want.shape)
    np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)


# -- generators ---------------------------------------------------------------------------


def spider(gen: GeneratorType, ins: int, outs: int, phase: PhaseVector | None = None) -> Builder:
    def build() -> Diagram:
        d = Diagram()
        node(d, gen, ins, outs, D, phase)
        return close(d)

    return build


PHASES: dict[str, PhaseVector | None] = {
    "free": None,
    "rational": PhaseVector(D, {1: Phase.turns(sp.Rational(1, 3))}),
    "symbolic": PhaseVector(D, {1: Phase.symbol("theta")}),
    "root": PhaseVector(D, {1: Phase.root_of_unity(1, D)}),
}

SPIDER_CASES = [
    pytest.param(gen, m, n, ph, id=f"{gen.name}{m}->{n}-{ph}")
    for gen in (Z_SPIDER, X_SPIDER)
    for m in range(4)
    for n in range(4)
    for ph in PHASES
    if ph == "free" or (m + n) % 2 == 1 or m + n == 0
]


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize(("gen", "m", "n", "ph"), SPIDER_CASES)
def test_spider(gen: GeneratorType, m: int, n: int, ph: str, d: int) -> None:
    assert_oracle(spider(gen, m, n, PHASES[ph])(), {"d": d, **SYMBOLS})


@pytest.mark.parametrize("d", DS)
def test_two_entry_phase_at_d3_and_concrete_dim(d: int) -> None:
    dim = Dim(3)
    pv = PhaseVector(dim, {1: Phase.symbol("theta"), 2: Phase.turns(sp.Rational(1, 4))})
    for gen in (Z_SPIDER, X_SPIDER):
        diagram = Diagram()
        node(diagram, gen, 1, 2, dim, pv)
        assert_oracle(close(diagram), {**SYMBOLS})
    diagram = Diagram()
    node(diagram, Z_SPIDER, 1, 1, Dim(d), PhaseVector(Dim(d), {1: Phase.turns(sp.Rational(1, 6))}))
    assert_oracle(close(diagram), {})


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("gen", (FOURIER_BOX, TRIANGLE, TRIANGLE_INVERSE), ids=lambda g: g.name)
def test_one_to_one_boxes(gen: GeneratorType, d: int) -> None:
    assert_oracle(spider(gen, 1, 1)(), {"d": d})


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("outs", range(4))
def test_w(outs: int, d: int) -> None:
    assert_oracle(spider(W_NODE, 1, outs)(), {"d": d})


CONNECTIVE_DIMS = [(2, 3), (3, 2), (2, 2), (3, 3), (1, 6)]


@pytest.mark.parametrize(("s", "t"), CONNECTIVE_DIMS)
def test_binder_and_splitter(s: int, t: int) -> None:
    d = Diagram()
    d.add_node(DIM_BINDER, [Dim(s), Dim(t)], [Dim(s * t)])
    assert_oracle(close(d), {})
    d = Diagram()
    d.add_node(DIM_SPLITTER, [Dim(s * t)], [Dim(s), Dim(t)])
    assert_oracle(close(d), {})


@pytest.mark.parametrize("d", DS)
def test_symbolic_binder(d: int) -> None:
    diagram = Diagram()
    diagram.add_node(DIM_BINDER, [D, Dim(2)], [D * 2])
    assert_oracle(close(diagram), {"d": d})
    diagram = Diagram()
    diagram.add_node(DIM_SPLITTER, [Dim(3) * D], [Dim(3), D])
    assert_oracle(close(diagram), {"d": d})


SCALARS = {
    "half": Scalar.rational(3, 2),
    "gauss": Scalar.gaussian_rational(1, -2),
    "symbol": Scalar.symbol("c"),
    "zero": Scalar.zero(),
}


@pytest.mark.parametrize("name", sorted(SCALARS))
def test_nonunit_scalar(name: str) -> None:
    diagram = spider(Z_SPIDER, 1, 2)()
    diagram.multiply_scalar(SCALARS[name])
    assert_oracle(diagram, {"d": 2, **SYMBOLS})
    empty = Diagram()
    empty.multiply_scalar(SCALARS[name])
    assert_oracle(empty, {**SYMBOLS})


# -- wired diagrams -----------------------------------------------------------------------


def pair(
    a: GeneratorType, b: GeneratorType, a_shape: tuple[int, int], b_shape: tuple[int, int]
) -> Builder:
    """``a``'s out0 wired to ``b``'s in0."""

    def build() -> Diagram:
        d = Diagram()
        x = node(
            d,
            a,
            *a_shape,
            phase=PhaseVector(D, {1: Phase.symbol("theta")}) if a in (Z_SPIDER, X_SPIDER) else None,
        )
        y = node(d, b, *b_shape)
        d.add_wire(out(x, 0), inp(y, 0))
        return close(d)

    return build


def z_self_loop() -> Diagram:
    d = Diagram()
    z = node(d, Z_SPIDER, 1, 3)
    d.add_wire(out(z, 1), inp(z, 0))
    return close(d)


def x_self_loop() -> Diagram:
    d = Diagram()
    x = node(d, X_SPIDER, 2, 2, phase=PhaseVector(D, {1: Phase.symbol("phi")}))
    d.add_wire(out(x, 0), inp(x, 1))
    return close(d)


def x_cap() -> Diagram:
    d = Diagram()
    x = node(d, X_SPIDER, 0, 3)
    d.add_wire(out(x, 0), out(x, 2))
    return close(d)


def double_wire() -> Diagram:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 3)
    b = node(d, X_SPIDER, 2, 1)
    d.add_wire(out(a, 0), inp(b, 0))
    d.add_wire(out(a, 2), inp(b, 1))
    return close(d)


def bs_chain() -> Diagram:
    d = Diagram()
    b = d.add_node(DIM_BINDER, [Dim(2), Dim(3)], [Dim(6)])
    s = d.add_node(DIM_SPLITTER, [Dim(6)], [Dim(3), Dim(2)])
    d.add_wire(out(b, 0), inp(s, 0))
    return close(d)


def sb_chain() -> Diagram:
    d = Diagram()
    s = d.add_node(DIM_SPLITTER, [Dim(6)], [Dim(2), Dim(3)])
    b = d.add_node(DIM_BINDER, [Dim(3), Dim(2)], [Dim(6)])
    z = node(d, Z_SPIDER, 1, 1, Dim(2))
    d.add_wire(out(s, 0), inp(z, 0))
    d.add_wire(out(z, 0), inp(b, 1))
    d.add_wire(out(s, 1), inp(b, 0))
    return close(d)


def zx_triangle() -> Diagram:
    d = Diagram()
    z = node(d, Z_SPIDER, 1, 2)
    t = node(d, TRIANGLE, 1, 1)
    x = node(d, X_SPIDER, 2, 1)
    ti = node(d, TRIANGLE_INVERSE, 1, 1)
    d.add_wire(out(z, 0), inp(t, 0))
    d.add_wire(out(t, 0), inp(x, 0))
    d.add_wire(out(z, 1), inp(ti, 0))
    d.add_wire(out(ti, 0), inp(x, 1))
    return close(d)


def disconnected() -> Diagram:
    d = Diagram()
    node(d, X_SPIDER, 1, 1)
    node(d, Z_SPIDER, 0, 2)
    node(d, W_NODE, 1, 2)
    d = close(d)
    d.set_boundary_outputs(list(reversed(d.boundary_outputs)))
    return d


WIRED: dict[str, Builder] = {
    "Z-Z": pair(Z_SPIDER, Z_SPIDER, (0, 2), (1, 2)),
    "Z-X": pair(Z_SPIDER, X_SPIDER, (1, 2), (1, 2)),
    "X-Z": pair(X_SPIDER, Z_SPIDER, (1, 1), (2, 1)),
    "X-X": pair(X_SPIDER, X_SPIDER, (1, 2), (1, 1)),
    "X-F": pair(X_SPIDER, FOURIER_BOX, (0, 2), (1, 1)),
    "F-X": pair(FOURIER_BOX, X_SPIDER, (1, 1), (1, 2)),
    "W-Z": pair(W_NODE, Z_SPIDER, (1, 2), (1, 1)),
    "Z-W": pair(Z_SPIDER, W_NODE, (0, 2), (1, 2)),
    "T-Ti": pair(TRIANGLE, TRIANGLE_INVERSE, (1, 1), (1, 1)),
    "W-X": pair(W_NODE, X_SPIDER, (1, 3), (1, 1)),
    "z_self_loop": z_self_loop,
    "x_self_loop": x_self_loop,
    "x_cap": x_cap,
    "double_wire": double_wire,
    "bs_chain": bs_chain,
    "sb_chain": sb_chain,
    "zx_triangle": zx_triangle,
    "disconnected": disconnected,
}


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("name", sorted(WIRED))
def test_wired(name: str, d: int) -> None:
    assert_oracle(WIRED[name](), {"d": d, **SYMBOLS})


# -- bang boxes ---------------------------------------------------------------------------


def port_box(
    gen: GeneratorType, ins: int, outs: int, ref: Callable[[NodeId], list[PortRef]]
) -> Builder:
    def build() -> Diagram:
        d = Diagram()
        x = node(
            d,
            gen,
            ins,
            outs,
            phase=PhaseVector(D, {1: Phase.symbol("theta")})
            if gen in (Z_SPIDER, X_SPIDER)
            else None,
        )
        close(d)
        d.add_bang_box(Mult("n"), port_scope=frozenset(ref(x)))
        return d

    return build


def node_box_internal() -> Diagram:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 2, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
    b = node(d, X_SPIDER, 1, 1)
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    return d


def node_box_with_outside() -> Diagram:
    d = Diagram()
    a = node(d, X_SPIDER, 1, 2)
    b = node(d, Z_SPIDER, 1, 1)
    w = node(d, W_NODE, 1, 1)
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    del w
    return d


def nested_two_index() -> Diagram:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 2)
    b = node(d, Z_SPIDER, 1, 1, phase=PhaseVector(D, {1: Phase.symbol("phi")}))
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    outer = d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("m"), node_scope=frozenset({b}), parent=outer)
    return d


def nested_x_inner() -> Diagram:
    d = Diagram()
    a = node(d, X_SPIDER, 1, 2)
    b = node(d, X_SPIDER, 1, 1)
    d.add_wire(out(a, 0), inp(b, 0))
    close(d)
    outer = d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("m"), node_scope=frozenset({b}), parent=outer)
    return d


def port_in_node_box() -> Diagram:
    d = Diagram()
    z = node(d, Z_SPIDER, 1, 2)
    close(d)
    outer = d.add_bang_box(Mult("n"), node_scope=frozenset({z}))
    d.add_bang_box(Mult("m"), port_scope=frozenset({out(z, 1)}), parent=outer)
    return d


def x_port_in_node_box() -> Diagram:
    d = Diagram()
    x = node(d, X_SPIDER, 1, 1)
    close(d)
    outer = d.add_bang_box(Mult("n"), node_scope=frozenset({x}))
    d.add_bang_box(Mult("m"), port_scope=frozenset({inp(x, 0)}), parent=outer)
    return d


def hub(gen: GeneratorType, hub_shape: tuple[int, int], sat: GeneratorType) -> Builder:
    """Hub's out0 wired to a boxed satellite's in0."""

    def build() -> Diagram:
        d = Diagram()
        h = node(d, gen, *hub_shape)
        s = node(d, sat, 1, 1)
        d.add_wire(out(h, 0), inp(s, 0))
        close(d)
        d.add_bang_box(Mult("n"), node_scope=frozenset({s}))
        return d

    return build


def hub_input_side() -> Diagram:
    d = Diagram()
    h = node(d, X_SPIDER, 2, 1)
    s = node(d, Z_SPIDER, 0, 2)
    d.add_wire(out(s, 0), inp(h, 1))
    close(d)
    d.add_bang_box(Mult("n"), node_scope=frozenset({s}))
    return d


BOXED: dict[str, Builder] = {
    "ghz_z": port_box(Z_SPIDER, 0, 1, lambda x: [out(x, 0)]),
    "ghz_z_in_out": port_box(Z_SPIDER, 1, 2, lambda x: [inp(x, 0), out(x, 1)]),
    "port_x": port_box(X_SPIDER, 1, 2, lambda x: [out(x, 0)]),
    "port_x_in": port_box(X_SPIDER, 1, 1, lambda x: [inp(x, 0)]),
    "port_w": port_box(W_NODE, 1, 2, lambda x: [out(x, 1)]),
    "node_box_internal": node_box_internal,
    "node_box_with_outside": node_box_with_outside,
    "nested_two_index": nested_two_index,
    "nested_x_inner": nested_x_inner,
    "port_in_node_box": port_in_node_box,
    "x_port_in_node_box": x_port_in_node_box,
    "hub_z": hub(Z_SPIDER, (1, 1), X_SPIDER),
    "hub_x": hub(X_SPIDER, (0, 1), Z_SPIDER),
    "hub_w": hub(W_NODE, (1, 1), Z_SPIDER),
    "hub_z_to_z": hub(Z_SPIDER, (0, 2), Z_SPIDER),
    "hub_x_to_x": hub(X_SPIDER, (1, 1), X_SPIDER),
    "hub_input_side": hub_input_side,
}


def mult_envs(diagram: Diagram, values: tuple[int, ...] = (0, 1, 2, 3)) -> list[dict[str, int]]:
    names = sorted(free_mult_symbols(diagram))
    if len(names) > 1:
        values = (0, 1, 2)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(values, repeat=len(names))
    ]


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("name", sorted(BOXED))
def test_boxed(name: str, d: int) -> None:
    diagram = BOXED[name]()
    assert validate(diagram).is_valid, validate(diagram)
    for env in mult_envs(diagram):
        assert_oracle(diagram, {"d": d, **SYMBOLS, **env})


@pytest.mark.parametrize("name", sorted(BANG_BOX_FAMILIES))
def test_scalable_families(name: str) -> None:
    diagram = BANG_BOX_FAMILIES[name]()
    for env in mult_envs(diagram, (0, 1, 2)):
        for d in DS:
            assert_oracle(diagram, {"d": d, "alpha": sp.Rational(1, 3), **env})


# -- random sweep -------------------------------------------------------------------------

_SMALL = (Z_SPIDER, Z_SPIDER, X_SPIDER, X_SPIDER, FOURIER_BOX, TRIANGLE, TRIANGLE_INVERSE, W_NODE)


def _random_node(rng: random.Random, d: Diagram, dim: Dim) -> NodeId:
    roll = rng.random()
    if roll < 0.08:
        return d.add_node(DIM_BINDER, [Dim(2), Dim(3)], [Dim(6)])
    if roll < 0.16:
        return d.add_node(DIM_SPLITTER, [Dim(6)], [Dim(2), Dim(3)])
    gen = rng.choice(_SMALL)
    if gen in (Z_SPIDER, X_SPIDER):
        phase = rng.choice(
            (
                PhaseVector(dim),
                PhaseVector(dim, {1: Phase.turns(sp.Rational(rng.randint(1, 5), 6))}),
                PhaseVector(dim, {1: Phase.symbol("theta")}),
            )
        )
        return node(d, gen, rng.randint(0, 2), rng.randint(0, 2), dim, phase)
    if gen is W_NODE:
        return node(d, gen, 1, rng.randint(0, 3), dim)
    return node(d, gen, 1, 1, dim)


def random_diagram(rng: random.Random) -> Diagram:
    """1-4 random nodes, random dim-matching wires, shuffled boundary, random boxes."""
    d = Diagram()
    dim = rng.choice((Dim(2), Dim(3), D))
    ids = [_random_node(rng, d, dim) for _ in range(rng.randint(1, 4))]
    ports = [
        (PortRef(nid, direction, i), port.dim)
        for nid in ids
        for direction in (Direction.INPUT, Direction.OUTPUT)
        for i, port in enumerate(d.nodes[nid].legs(direction))
    ]
    rng.shuffle(ports)
    free: list[PortRef] = []
    while ports:
        ref, pdim = ports.pop()
        partner = next((p for p in ports if p[1] == pdim), None)
        if partner is not None and rng.random() < 0.55:
            ports.remove(partner)
            d.add_wire(ref, partner[0])
        else:
            free.append(ref)
    rng.shuffle(free)
    d.set_boundary_inputs([r for r in free if r.direction is Direction.INPUT])
    d.set_boundary_outputs([r for r in free if r.direction is Direction.OUTPUT])
    owner = {}
    if rng.random() < 0.5:
        scope = frozenset(rng.sample(ids, rng.randint(1, len(ids))))
        box = d.add_bang_box(Mult("n") + rng.randint(0, 1), node_scope=scope)
        owner = dict.fromkeys(scope, box)
        if len(scope) > 1 and rng.random() < 0.4:
            inner = frozenset(rng.sample(sorted(scope), 1))
            owner.update(
                dict.fromkeys(inner, d.add_bang_box(Mult("m"), node_scope=inner, parent=box))
            )
    for ref in free:
        if rng.random() < 0.2:
            d.add_bang_box(Mult("k"), port_scope=frozenset({ref}), parent=owner.get(ref.node_id))
    if rng.random() < 0.2:
        d.multiply_scalar(Scalar.rational(rng.randint(1, 4), rng.randint(1, 3)))
    return d


ATTEMPTS = 30


def run_seed(seed: int) -> bool:
    """Check one seeded random diagram at a few envs; False if none was valid."""
    rng = random.Random(seed)
    for _ in range(ATTEMPTS):
        diagram = random_diagram(rng)
        if not validate(diagram).is_valid:
            continue
        envs = [
            {
                "d": rng.choice(DS),
                "theta": sp.Rational(1, 7),
                "n": rng.randint(0, 2),
                "m": rng.randint(0, 2),
                "k": rng.randint(0, 2),
            }
            for _ in range(3)
        ]
        checked = False
        for env in envs:
            env = full_env(diagram, env)
            try:
                want = oracle_tensor(diagram, env)
            except (BangBoxError, CheckError, ContractSizeError):
                continue
            try:
                got = evaluate(P.dirac_term(diagram), env)
            except EvalSizeError:
                continue
            assert got.shape == want.shape, (seed, env)
            np.testing.assert_allclose(
                got, want, atol=1e-9, rtol=0, err_msg=f"seed {seed} env {env}"
            )
            checked = True
        if checked:
            return True
    return False


@pytest.mark.parametrize("seed", range(40))
def test_random_small(seed: int) -> None:
    if not run_seed(seed):
        pytest.skip("no valid diagram")


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(40, 3040))
def test_random_large(seed: int) -> None:
    if not run_seed(seed):
        pytest.skip("no valid diagram")


def test_random_sweep_mostly_valid() -> None:
    assert sum(run_seed(seed) for seed in range(1000, 1012)) >= 10


# -- plan done-whens ----------------------------------------------------------------------


def _fuse(diagram: Diagram) -> Diagram:
    match = SPIDER_FUSION.pattern.find_matches(diagram)[0]
    return apply(diagram, SPIDER_FUSION, match).diagram


def _nospace(text: str | None) -> str | None:
    return None if text is None else text.replace(" ", "")


class TestFusionRoundTrip:
    @pytest.mark.parametrize(
        "source", ["sum_{k=0}^{d-1} |k,k>; copy", "sum_{k=0}^{3-1} |k,k>; copy"]
    )
    def test_round_trip(self, source: str) -> None:
        pre = parse_dirac_source(source)
        assert P.to_dirac_source(pre) is None
        fused = _fuse(pre)
        text = P.to_dirac_source(fused)
        assert text is not None
        back = parse_dirac_source(text)
        assert isomorphic(back, fused)
        assert dict(back.parameters) == dict(fused.parameters)
        for d in DS:
            env = full_env(fused, {"d": d})
            np.testing.assert_allclose(
                evaluate(P.dirac_term(pre), env), oracle_tensor(fused, env), atol=1e-9
            )

    def test_symbolic_source_text(self) -> None:
        fused = _fuse(parse_dirac_source("sum_{k=0}^{d-1} |k,k>; copy"))
        assert _nospace(P.to_dirac_source(fused)) == "sum_{k=0}^{d-1}|k,k,k>"
        assert P.render_term(P.dirac_term(fused)) == "sum_{k=0}^{d-1} |k, k, k>"

    def test_concrete_source_text_uses_bound_value(self) -> None:
        fused = _fuse(parse_dirac_source("sum_{k=0}^{3-1} |k,k>; copy"))
        assert _nospace(P.to_dirac_source(fused)) == "sum_{k=0}^{3-1}|k,k,k>"

    def test_boxed_state_round_trip_is_oracle_equal(self) -> None:
        diagram = parse_dirac_source("sum_{k=0}^{d-1} |k>^{3}")
        text = P.to_dirac_source(diagram)
        assert text == "sum_{k=0}^{d-1} |k>^{3}"
        back = parse_dirac_source(text)
        for d in DS:
            np.testing.assert_allclose(
                oracle_tensor(back, full_env(back, {"d": d})),
                oracle_tensor(diagram, full_env(diagram, {"d": d})),
                atol=1e-9,
            )


class TestSingleZSpider:
    def test_ghz3(self) -> None:
        diagram = spider(Z_SPIDER, 0, 3)()
        assert P.render_term(P.dirac_term(diagram)) == "sum_{k=0}^{d-1} |k, k, k>"
        assert P.render_dirac(diagram) == "GHZ_{3} = sum_{k=0}^{d-1} |k, k, k>"
        rendering = P.dirac(diagram)
        assert rendering.name == "GHZ_{3}"
        assert rendering.text == "GHZ_{3} = sum_{k=0}^{d-1} |k, k, k>"
        assert rendering.instance_text is None

    def test_copy_and_identity(self) -> None:
        assert P.render_dirac(spider(Z_SPIDER, 1, 2)()) == "copy_{2} = sum_{k=0}^{d-1} |k, k><k|"
        assert P.render_dirac(spider(Z_SPIDER, 1, 1)()) == "id = sum_{k=0}^{d-1} |k><k|"
        assert P.render_dirac(spider(Z_SPIDER, 2, 0)()) == "GHZ_{2}^dagger = sum_{k=0}^{d-1} <k, k|"

    def test_unicode(self) -> None:
        text = P.render_dirac(spider(Z_SPIDER, 0, 3)(), unicode=True)
        assert "Σ" in text and "⟩" in text and "sum" not in text and ">" not in text

    def test_phased_z_mentions_phase(self) -> None:
        text = P.render_term(P.dirac_term(spider(Z_SPIDER, 1, 1, PHASES["symbolic"])()))
        assert text.startswith("sum_{k=0}^{d-1} e^{2 pi i (")
        assert "theta" in text and "[k=1]" in text.replace(" ", "")
        assert text.endswith("|k><k|")

    def test_concrete_instance_line(self) -> None:
        diagram = parse_dirac_source("sum_{k=0}^{3-1} |k,k,k>")
        text = P.render_dirac(diagram, instance=True)
        lines = text.splitlines()
        assert len(lines) == 2
        assert lines == [
            "GHZ_{3} = sum_{k=0}^{d-1} |k, k, k>",
            "at d = 3: sum_{k=0}^{3-1} |k, k, k>",
        ]


def _derivation() -> tuple[Diagram, list[RewriteResult]]:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 2)
    b = node(d, Z_SPIDER, 1, 1)
    c = node(d, Z_SPIDER, 1, 2)
    e = node(d, Z_SPIDER, 1, 1)
    d.add_wire(out(a, 0), inp(b, 0))
    d.add_wire(out(b, 0), inp(c, 0))
    d.add_wire(out(c, 0), inp(e, 0))
    close(d)
    results: list[RewriteResult] = []
    apply_until_fixpoint(d.copy(), [IDENTITY_REMOVAL, SPIDER_FUSION], on_result=results.append)
    return d, results


class TestDerivationRendering:
    def test_every_state_has_dirac_in_dirac_mode(self) -> None:
        initial, results = _derivation()
        assert len(results) >= 2
        text = P.render_derivation(initial, results, P.NotationMode.DIRAC)
        assert "graph:" not in text and "scalable:" not in text
        for i in range(len(results) + 1):
            assert f"state {i}:" in text
        for i in range(1, len(results) + 1):
            assert f"step {i}:" in text
        for state in (initial, *(r.diagram for r in results)):
            assert P.render_term(P.dirac_term(state)) in text
        assert P.render_term(P.dirac_term(results[-1].diagram)) == "sum_{k=0}^{d-1} |k, k, k>"

    def test_zx_mode_has_both(self) -> None:
        initial, results = _derivation()
        text = P.render_derivation(initial, results, P.NotationMode.ZX)
        assert text.count("graph:") == len(results) + 1
        for state in (initial, *(r.diagram for r in results)):
            assert P.render_term(P.dirac_term(state)) in text

    def test_render_modes_on_a_diagram(self) -> None:
        diagram = spider(Z_SPIDER, 0, 3)()
        dirac_text = P.render(diagram)
        assert not any(line.startswith(("graph:", "scalable:")) for line in dirac_text.splitlines())
        assert dirac_text == P.render(diagram, P.NotationMode.DIRAC)
        zx_text = P.render(diagram, P.NotationMode.ZX)
        assert zx_text.startswith("graph:")
        assert P.render_dirac(diagram) in zx_text


def _sibling(gen_a: GeneratorType, gen_b: GeneratorType, mult_b: str) -> Builder:
    """a.out0 -- b.in0 with a and b in sibling node-scope boxes."""

    def build() -> Diagram:
        d = Diagram()
        a = node(d, gen_a, 0, 2)
        b = node(d, gen_b, 1, 1, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
        d.add_wire(out(a, 0), inp(b, 0))
        close(d)
        d.add_bang_box(Mult("n"), node_scope=frozenset({a}))
        d.add_bang_box(Mult(mult_b), node_scope=frozenset({b}))
        return d

    return build


def _colliding_names() -> Diagram:
    d = Diagram()
    k0 = Dim("k0")
    z = node(d, Z_SPIDER, 0, 2, k0)
    x = node(d, X_SPIDER, 1, 1, k0, PhaseVector(k0, {1: Phase.symbol("q1")}))
    d.add_wire(out(z, 0), inp(x, 0))
    close(d)
    d.add_bang_box(Mult("i0") + 1, node_scope=frozenset({x}))
    return d


CORE_EXTRA: dict[str, Builder] = {
    "sibling_z_z": _sibling(Z_SPIDER, Z_SPIDER, "m"),
    "sibling_x_x": _sibling(X_SPIDER, X_SPIDER, "m"),
    "sibling_x_z_shared_symbol": _sibling(X_SPIDER, Z_SPIDER, "n"),
    "colliding_names": _colliding_names,
}


class TestCoreExtraOracle:
    @pytest.mark.parametrize("name", sorted(CORE_EXTRA))
    def test_oracle(self, name: str) -> None:
        diagram = CORE_EXTRA[name]()
        assert validate(diagram).is_valid, validate(diagram)
        for env in mult_envs(diagram, (0, 1, 2)):
            for d in DS:
                assert_oracle(diagram, {"d": d, "k0": d, "q1": sp.Rational(1, 3), **SYMBOLS, **env})


def _deep_cross(gen_b: GeneratorType) -> Builder:
    """Hub a in O; b in O>I and c in O>J (siblings under O), both wired to a."""

    def build() -> Diagram:
        d = Diagram()
        a = node(d, X_SPIDER, 0, 2)
        b = node(d, gen_b, 1, 1, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
        c = node(d, Z_SPIDER, 1, 1)
        d.add_wire(out(a, 0), inp(b, 0))
        d.add_wire(out(a, 1), inp(c, 0))
        close(d)
        outer = d.add_bang_box(Mult("n"), node_scope=frozenset({a, b, c}))
        d.add_bang_box(Mult("m"), node_scope=frozenset({b}), parent=outer)
        d.add_bang_box(Mult("k"), node_scope=frozenset({c}), parent=outer)
        return d

    return build


def _boxed_self_loops() -> Diagram:
    d = Diagram()
    z = node(d, Z_SPIDER, 1, 3, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
    x = node(d, X_SPIDER, 1, 2, phase=PhaseVector(D, {1: Phase.symbol("phi")}))
    d.add_wire(out(z, 1), inp(z, 0))
    d.add_wire(out(x, 0), inp(x, 0))
    close(d)
    d.add_bang_box(Mult("n"), node_scope=frozenset({z, x}))
    return d


def _legless_boxed() -> Diagram:
    d = Diagram()
    a = node(d, Z_SPIDER, 0, 0, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
    b = node(d, X_SPIDER, 0, 0)
    c = node(d, Z_SPIDER, 0, 0)
    d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("m"), node_scope=frozenset({c}))
    d.multiply_scalar(Scalar.symbol("c"))
    return d


def _boxed_splitter() -> Diagram:
    d = Diagram()
    s = d.add_node(DIM_SPLITTER, [Dim(6)], [Dim(2), Dim(3)])
    z = node(d, Z_SPIDER, 1, 2, Dim(3))
    d.add_wire(out(s, 1), inp(z, 0))
    close(d)
    d.add_bang_box(Mult("n"), node_scope=frozenset({s, z}))
    return d


def _interleaved_boundary() -> Diagram:
    """Nested node boxes and a port box whose boundary refs interleave with outside ones."""
    d = Diagram()
    a = node(d, Z_SPIDER, 1, 2)
    b = node(d, X_SPIDER, 1, 2, phase=PhaseVector(D, {1: Phase.symbol("theta")}))
    c = node(d, Z_SPIDER, 0, 1)
    d.set_boundary_outputs([out(b, 1), out(c, 0), out(a, 0), out(b, 0), out(a, 1)])
    d.set_boundary_inputs([inp(b, 0), inp(a, 0)])
    outer = d.add_bang_box(Mult("n"), node_scope=frozenset({a, b}))
    d.add_bang_box(Mult("m"), node_scope=frozenset({b}), parent=outer)
    d.add_bang_box(Mult("k"), port_scope=frozenset({out(a, 1)}), parent=outer)
    return d


BOXED_EXTRA: dict[str, Builder] = {
    "deep_cross_x": _deep_cross(X_SPIDER),
    "deep_cross_z": _deep_cross(Z_SPIDER),
    "boxed_self_loops": _boxed_self_loops,
    "legless_boxed": _legless_boxed,
    "boxed_splitter": _boxed_splitter,
}


class TestBoxedExtraOracle:
    @pytest.mark.parametrize("name", sorted(BOXED_EXTRA))
    def test_oracle(self, name: str) -> None:
        diagram = BOXED_EXTRA[name]()
        assert validate(diagram).is_valid, validate(diagram)
        for env in mult_envs(diagram, (0, 1, 2)):
            for d in DS:
                assert_oracle(diagram, {"d": d, **SYMBOLS, **env})

    def test_interleaved_boundary(self) -> None:
        diagram = _interleaved_boundary()
        assert validate(diagram).is_valid, validate(diagram)
        for env in mult_envs(diagram, (0, 1, 2)):
            if sum(env.values()) <= 4:
                assert_oracle(diagram, {"d": 2, **SYMBOLS, **env})
        assert_oracle(diagram, {"d": 3, **SYMBOLS, "n": 2, "m": 1, "k": 1})

    def test_bound_parameters_used(self) -> None:
        diagram = parse_dirac_source("sum_{k=0}^{3-1} |k>^{2}")
        assert dict(diagram.parameters) == {"d": 3, "n": 2}
        assert_oracle(diagram, {})

    def test_fanned_fixed_arity_crossing_rejected(self) -> None:
        d = Diagram()
        b = d.add_node(DIM_BINDER, [Dim(2), Dim(3)], [Dim(6)])
        z = node(d, Z_SPIDER, 1, 1, Dim(6))
        d.add_wire(out(b, 0), inp(z, 0))
        close(d)
        d.add_bang_box(Mult("n"), node_scope=frozenset({z}))
        with pytest.raises(P.PrinterDomainError):
            P.dirac_term(d)
