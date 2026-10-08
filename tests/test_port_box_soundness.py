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

"""Rules, validation, instantiation and decide_equal on port-scope bang boxes."""

from __future__ import annotations

import random

import pytest

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, instantiate_symbol
from archytaszx.diagram.generators import (
    FOURIER_BOX,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    GeneratorType,
)
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import IssueKind, validate
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import RULES
from archytaszx.semantics.contract_numeric import contract
from archytaszx.semantics.decide import EqualityVerdict, decide_equal, refute_by_oracle

D2 = Dim(2)
N_SAMPLES = [{"n": k} for k in range(4)]


def _out(node: NodeId, index: int = 0) -> PortRef:
    return PortRef(node, Direction.OUTPUT, index)


def _in(node: NodeId, index: int = 0) -> PortRef:
    return PortRef(node, Direction.INPUT, index)


def _boxed_output(d: Diagram, ref: PortRef) -> Diagram:
    d.set_boundary_outputs([ref])
    d.add_bang_box(Mult("n"), port_scope=frozenset({ref}))
    return d


def x_state_into_boxed_z() -> Diagram:
    """An X state wired into a Z spider whose single output is port-boxed."""
    d = Diagram()
    x = d.add_node(X_SPIDER, [], [D2])
    z = d.add_node(Z_SPIDER, [D2], [D2])
    d.add_wire(_out(x), _in(z))
    return _boxed_output(d, _out(z))


def connected_x_family() -> Diagram:
    """One X spider with a port-boxed output and scalar 1."""
    d = Diagram()
    x = d.add_node(X_SPIDER, [], [D2])
    return _boxed_output(d, _out(x))


def w_into_boxed_x() -> Diagram:
    """A W node feeding a 1-to-1 X spider whose output is port-boxed."""
    d = Diagram()
    w = d.add_node(W_NODE, [D2], [D2])
    x = d.add_node(X_SPIDER, [D2], [D2])
    d.add_wire(_out(w), _in(x))
    d.set_boundary_inputs([_in(w)])
    return _boxed_output(d, _out(x))


def boxed_w() -> Diagram:
    """A W node with its output port-boxed."""
    d = Diagram()
    w = d.add_node(W_NODE, [D2], [D2])
    d.set_boundary_inputs([_in(w)])
    return _boxed_output(d, _out(w))


def z_into_boxed_w() -> Diagram:
    """A 1-to-1 Z spider feeding a W node whose output is port-boxed."""
    d = Diagram()
    z = d.add_node(Z_SPIDER, [D2], [D2])
    w = d.add_node(W_NODE, [D2], [D2])
    d.add_wire(_out(z), _in(w))
    d.set_boundary_inputs([_in(z)])
    return _boxed_output(d, _out(w))


def boxed_z() -> Diagram:
    """A 1-to-1 Z spider with its output port-boxed."""
    d = Diagram()
    z = d.add_node(Z_SPIDER, [D2], [D2])
    d.set_boundary_inputs([_in(z)])
    return _boxed_output(d, _out(z))


def boxed_z_with_w_input() -> Diagram:
    """A W node feeding a 1-to-1 Z spider whose output is port-boxed."""
    d = Diagram()
    w = d.add_node(W_NODE, [D2], [D2])
    z = d.add_node(Z_SPIDER, [D2], [D2])
    d.add_wire(_out(w), _in(z))
    d.set_boundary_inputs([_in(w)])
    return _boxed_output(d, _out(z))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (x_state_into_boxed_z, connected_x_family),
        (w_into_boxed_x, boxed_w),
        (z_into_boxed_w, boxed_z),
    ],
)
def test_decide_equal_refutes_wrongly_moved_boxes(left, right) -> None:  # type: ignore[no-untyped-def]
    assert refute_by_oracle(left(), right(), samples=N_SAMPLES).refuted
    decision = decide_equal(left(), right(), samples=N_SAMPLES)
    assert decision.verdict is EqualityVerdict.UNEQUAL


@pytest.mark.parametrize(
    "build", [x_state_into_boxed_z, w_into_boxed_x, boxed_w, z_into_boxed_w, boxed_z_with_w_input]
)
def test_rules_on_reported_cases_agree_with_oracle(build) -> None:  # type: ignore[no-untyped-def]
    d = build()
    for rule in RULES.values():
        for match in rule.pattern.find_matches(d):
            if match.all_side_conditions_passed:
                after = apply(d, rule, match).diagram
                assert not refute_by_oracle(d, after, samples=N_SAMPLES).refuted, rule.name


def test_bialgebra_skips_port_boxed_leg() -> None:
    d = Diagram()
    z = d.add_node(Z_SPIDER, [D2], [D2, D2])
    x = d.add_node(X_SPIDER, [D2, D2], [D2])
    d.add_wire(_out(z, 0), _in(x, 0))
    d.set_boundary_inputs([_in(z), _in(x, 1)])
    d.set_boundary_outputs([_out(z, 1), _out(x)])
    d.add_bang_box(Mult("n"), port_scope=frozenset({_out(x)}))
    for name in ("bialgebra", "bialgebra_swapped"):
        rule = RULES[name]
        assert not [m for m in rule.pattern.find_matches(d) if m.all_side_conditions_passed]


_GENERATORS: tuple[GeneratorType, ...] = (
    Z_SPIDER,
    X_SPIDER,
    Z_SPIDER,
    X_SPIDER,
    W_NODE,
    W_NODE,
    TRIANGLE,
    TRIANGLE_INVERSE,
    FOURIER_BOX,
)


def _random_boxed(rng: random.Random) -> Diagram | None:
    d = Diagram()
    ids: list[NodeId] = []
    for _ in range(rng.randint(1, 3)):
        generator = rng.choice(_GENERATORS)
        if generator in (TRIANGLE, TRIANGLE_INVERSE, FOURIER_BOX):
            ins, outs = 1, 1
        elif generator is W_NODE:
            ins, outs = 1, rng.randint(0, 3)
        else:
            ins, outs = rng.randint(0, 2), rng.randint(0, 2)
        ids.append(d.add_node(generator, [D2] * ins, [D2] * outs))

    def free() -> list[PortRef]:
        used = {end for w in d.wires for end in (w.a, w.b)}
        return [
            PortRef(n, direction, k)
            for n in ids
            for direction in (Direction.INPUT, Direction.OUTPUT)
            for k in range(len(d.nodes[n].legs(direction)))
            if PortRef(n, direction, k) not in used
        ]

    for _ in range(rng.randint(0, 4)):
        outs = [p for p in free() if p.direction is Direction.OUTPUT]
        ins = [p for p in free() if p.direction is Direction.INPUT]
        if outs and ins:
            a, b = rng.choice(outs), rng.choice(ins)
            if a.node_id != b.node_id:
                d.add_wire(a, b)
    ports = free()
    rng.shuffle(ports)
    d.set_boundary_inputs([p for p in ports if p.direction is Direction.INPUT])
    d.set_boundary_outputs([p for p in ports if p.direction is Direction.OUTPUT])
    boxed = False
    for p in ports:
        generator = d.nodes[p.node_id].generator_type
        growable = generator in (Z_SPIDER, X_SPIDER) or (
            generator is W_NODE and p.direction is Direction.OUTPUT
        )
        if growable and rng.random() < 0.5:
            d.add_bang_box(Mult("n"), port_scope=frozenset({p}))
            boxed = True
    if not boxed or not validate(d).is_valid:
        return None
    return d


@pytest.mark.parametrize("chunk", range(4))
def test_every_rule_agrees_with_oracle_on_port_boxed_diagrams(chunk: int) -> None:
    failures = []
    for seed in range(chunk * 750, (chunk + 1) * 750):
        d = _random_boxed(random.Random(seed))
        if d is None:
            continue
        for name, rule in RULES.items():
            for match in rule.pattern.find_matches(d):
                if not match.all_side_conditions_passed:
                    continue
                after = apply(d, rule, match).diagram
                if refute_by_oracle(d, after, samples=N_SAMPLES).refuted:
                    failures.append((seed, name))
    assert not failures


@pytest.mark.parametrize(
    ("generator", "direction"),
    [
        (TRIANGLE, Direction.OUTPUT),
        (FOURIER_BOX, Direction.INPUT),
        (W_NODE, Direction.INPUT),
    ],
)
def test_validate_rejects_port_box_on_fixed_arity_leg(
    generator: GeneratorType, direction: Direction
) -> None:
    d = Diagram()
    node = d.add_node(generator, [D2], [D2])
    d.set_boundary_inputs([_in(node)])
    d.set_boundary_outputs([_out(node)])
    d.add_bang_box(Mult("n"), port_scope=frozenset({PortRef(node, direction, 0)}))
    kinds = {issue.kind for issue in validate(d).errors}
    assert IssueKind.BANGBOX_PORT_FIXED_ARITY in kinds


def test_validate_accepts_port_box_on_w_output() -> None:
    assert validate(boxed_w()).is_valid


@pytest.mark.parametrize("generator", [Z_SPIDER, X_SPIDER])
def test_killing_last_leg_keeps_spider_contractible(generator: GeneratorType) -> None:
    d = Diagram()
    node = d.add_node(generator, [], [Dim(3)])
    _boxed_output(d, _out(node))
    killed = instantiate_symbol(d, "n", 0)
    assert contract(killed).tensor == pytest.approx(3)
    assert not refute_by_oracle(d, d, samples=N_SAMPLES).refusals


def test_decide_equal_still_accepts_sound_normal_form() -> None:
    d = boxed_z_with_w_input()
    decision = decide_equal(d, d.copy(), samples=N_SAMPLES)
    assert decision.verdict is EqualityVerdict.EQUAL
    assert decision.samples_checked > 0
    assert Scalar.one() == d.scalar


def test_killing_parent_kills_sibling_port_boxes_on_one_node() -> None:
    d = Diagram()
    z = d.add_node(Z_SPIDER, [], [D2, D2])
    d.set_boundary_outputs([_out(z, 0), _out(z, 1)])
    parent = d.add_bang_box(Mult("m"), node_scope=frozenset({z}))
    d.add_bang_box(Mult("n"), port_scope=frozenset({_out(z, 0)}), parent=parent)
    d.add_bang_box(Mult("k"), port_scope=frozenset({_out(z, 1)}), parent=parent)
    assert validate(d).is_valid
    killed = instantiate_symbol(d, "m", 0)
    assert not killed.nodes
    assert not killed.bang_boxes
    assert validate(killed).is_valid
