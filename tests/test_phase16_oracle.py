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

"""Phase 16 done-when: users may work in whichever notation suits a construction, losslessly.

Every catalogue family and seeded random bang-boxed diagrams round-trip through scalable
notation with diagram and denotation preserved, and every sheet operation applicable to them
checks against the oracle.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Callable
from dataclasses import replace

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.diagram.bangbox import Mult, peel_one
from archytaszx.diagram.compare import compare_structure, isomorphic
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import BangBoxId, Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.scalable import (
    ScalableDiagram,
    ScaleId,
    ScaleKind,
    from_scalable,
    is_bundle_normal,
    to_scalable,
)
from archytaszx.diagram.validate import validate
from archytaszx.rewrite.sheet import (
    SHEET_OPERATIONS,
    SheetError,
    SheetStep,
    join_scales,
)
from archytaszx.semantics.interop import check_round_trip, check_sheet_step, default_samples

from .test_scalable import BANG_BOX_FAMILIES, SCALABLE_FAMILIES

SEEDS = 60
ATTEMPTS = 20
RANDOM_SAMPLE_LIMIT = 6
MULTIPLICITIES: tuple[Callable[[], Mult], ...] = (
    lambda: Mult("k"),
    lambda: Mult("k") + 1,
    lambda: Mult("k") * 2,
    lambda: Mult("j"),
)


def _bang_box_form(build: Callable[[], ScalableDiagram]) -> Diagram:
    return from_scalable(build())


def _families() -> dict[str, Callable[[], Diagram]]:
    families = dict(BANG_BOX_FAMILIES)
    for name, build in SCALABLE_FAMILIES.items():
        families[name] = functools.partial(_bang_box_form, build)
    return families


FAMILIES = _families()


# -- seeded random bang-boxed diagrams ---------------------------------------------------


def _component(rng: random.Random, d: Diagram) -> list[NodeId]:
    """A random connected tree of 1 to 3 spiders; free ports stay unassigned."""
    nodes: list[NodeId] = []
    for position in range(rng.randint(1, 3)):
        generator = rng.choice((Z_SPIDER, X_SPIDER))
        dim = Dim(2)
        phase = None
        if rng.random() < 0.4:
            phase = PhaseVector(dim, {1: Phase(sp.Rational(rng.randint(1, 7), 8))})
        ins = (1 if position else 0) + rng.randint(0, 1)
        node = d.add_node(generator, [dim] * ins, [dim] * rng.randint(1, 2), phase=phase)
        if position:
            parent = rng.choice(nodes)
            d.add_wire(PortRef(parent, Direction.OUTPUT, 0), PortRef(node, Direction.INPUT, 0))
        nodes.append(node)
    return nodes


def _free_ports(d: Diagram) -> list[PortRef]:
    used = {end for w in d.wires for end in (w.a, w.b)}
    refs = [
        PortRef(nid, direction, i)
        for nid, node in sorted(d.nodes.items())
        for direction in (Direction.INPUT, Direction.OUTPUT)
        for i in range(len(node.legs(direction)))
    ]
    return [r for r in refs if r not in used]


def random_family(rng: random.Random) -> Diagram:
    """Disjoint random components, shuffled boundaries, random node- and port-scope boxes."""
    d = Diagram()
    components = [_component(rng, d) for _ in range(rng.randint(1, 3))]
    free = _free_ports(d)
    rng.shuffle(free)
    d.set_boundary_inputs([r for r in free if r.direction is Direction.INPUT])
    d.set_boundary_outputs([r for r in free if r.direction is Direction.OUTPUT])
    owner: dict[NodeId, BangBoxId] = {}
    outer: BangBoxId | None = None
    if len(components) >= 2 and rng.random() < 0.5:
        scope = frozenset(components[0] + components[1])
        outer = d.add_bang_box(rng.choice(MULTIPLICITIES)(), node_scope=scope)
        owner.update(dict.fromkeys(scope, outer))
    for component in components:
        if rng.random() < 0.5:
            parent = owner.get(component[0])
            box = d.add_bang_box(
                rng.choice(MULTIPLICITIES)(), node_scope=frozenset(component), parent=parent
            )
            owner.update(dict.fromkeys(component, box))
    for ref in free:
        if rng.random() < 0.25:
            d.add_bang_box(
                Mult("n") + rng.randint(0, 1),
                port_scope=frozenset({ref}),
                parent=owner.get(ref.node_id),
            )
    return d


def seeded_family(seed: int) -> Diagram | None:
    """The first valid :func:`random_family` of seed ``seed`` with a bang box, or ``None``."""
    rng = random.Random(seed)
    for _ in range(ATTEMPTS):
        diagram = random_family(rng)
        if diagram.bang_boxes and validate(diagram).is_valid:
            return diagram
    return None


# -- catalogue round trips ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FAMILIES))
class TestCatalogueRoundTrip:
    """Every catalogue family keeps its structure and its denotation through both notations."""

    def test_diagram_and_denotation_are_preserved(self, name: str) -> None:
        diagram = FAMILIES[name]()
        report = check_round_trip(diagram, default_samples(diagram, limit=27))
        assert report.ok, report.reason
        assert report.samples_evaluated > 0
        assert report.identical is is_bundle_normal(diagram)

    def test_scalable_form_survives_a_second_trip(self, name: str) -> None:
        s = to_scalable(FAMILIES[name]())
        assert to_scalable(from_scalable(s)) == s.renumbered()
        twice = check_round_trip(from_scalable(s))
        assert twice.ok and twice.identical, twice.reason


# -- random round trips ------------------------------------------------------------------


@pytest.mark.slow
class TestRandomRoundTrip:
    """Seeded random bang-boxed diagrams round-trip losslessly."""

    @pytest.mark.parametrize("seed", range(SEEDS))
    def test_round_trip_is_lossless(self, seed: int) -> None:
        diagram = seeded_family(seed)
        if diagram is None:
            pytest.skip(f"no valid boxed diagram in {ATTEMPTS} attempts")
        report = check_round_trip(diagram, default_samples(diagram, limit=RANDOM_SAMPLE_LIMIT))
        assert report.ok, report.reason
        if is_bundle_normal(diagram):
            assert report.identical, report.reason
        regrouped = from_scalable(to_scalable(report.back))
        assert compare_structure(regrouped, report.back).identical

    def test_sweep_exercises_every_box_shape(self) -> None:
        shapes: set[str] = set()
        evaluated = 0
        non_normal = 0
        for seed in range(SEEDS):
            diagram = seeded_family(seed)
            assert diagram is not None, seed
            for box in diagram.bang_boxes.values():
                kind = "node" if box.is_node_scope else "port"
                shapes.add(kind + ("-nested" if box.parent is not None else ""))
            non_normal += not is_bundle_normal(diagram)
            samples = default_samples(diagram, limit=RANDOM_SAMPLE_LIMIT)
            evaluated += check_round_trip(diagram, samples).samples_evaluated
        assert shapes == {"node", "node-nested", "port", "port-nested"}
        assert non_normal >= SEEDS // 10
        assert evaluated >= SEEDS * 3


# -- sheet operations --------------------------------------------------------------------


def _components(s: ScalableDiagram) -> list[tuple[NodeId, ...]]:
    """The node sets of the wire-connected components of ``s``, in ascending id order."""
    root = {n.id: n.id for n in s.nodes}

    def find(node: NodeId) -> NodeId:
        while root[node] != node:
            node = root[node]
        return node

    for wire in sorted(s.wires, key=lambda w: w.sort_key()):
        a, b = sorted((find(wire.a.node_id), find(wire.b.node_id)))
        root[b] = a
    groups: dict[NodeId, list[NodeId]] = {}
    for node in s.nodes:
        groups.setdefault(find(node.id), []).append(node.id)
    return [tuple(group) for _, group in sorted(groups.items())]


def _split_candidates(multiplicity: Mult) -> list[Mult]:
    return [Mult(1), *(Mult(name) for name in sorted(multiplicity.free_symbols))]


def _zeroed(s: ScalableDiagram, scale: ScaleId) -> ScalableDiagram:
    """``s`` with ``scale`` at multiplicity 0 and bindings of vanished symbols dropped."""
    scales = tuple(replace(sc, multiplicity=Mult(0)) if sc.id == scale else sc for sc in s.scales)
    zeroed = replace(s, scales=scales)
    live = zeroed.free_mult_symbols()
    mults = s.free_mult_symbols()
    parameters = {k: v for k, v in s.parameters.items() if k in live or k not in mults}
    return replace(zeroed, parameters=parameters)


def applicable_steps(s: ScalableDiagram) -> list[SheetStep]:
    """Every sheet step the operations accept on ``s``, plus a kill of each zeroed scale."""
    steps: list[SheetStep] = []

    def attempt(state: ScalableDiagram, name: str, **arguments: object) -> None:
        try:
            steps.append(SHEET_OPERATIONS[name](state, **arguments))
        except SheetError:
            return

    for scale in s.scales:
        if scale.kind is ScaleKind.COPIES:
            for first in _split_candidates(scale.multiplicity):
                for unit in (True, False):
                    attempt(s, "split_scale", scale=scale.id, first=first, dissolve_unit=unit)
        attempt(s, "dissolve", scale=scale.id)
        attempt(_zeroed(s, scale.id), "kill_scale", scale=scale.id)
    for wire in sorted(s.wires, key=lambda w: w.sort_key()):
        attempt(s, "fuse_sheet", wire=wire)
    for component in _components(s):
        attempt(s, "enclose", nodes=component)
    first_round = tuple(steps)
    for step in first_round:
        new_ids = sorted({sc.id for sc in step.after.scales} - {sc.id for sc in s.scales})
        if step.operation == "enclose":
            attempt(step.after, "dissolve", scale=new_ids[0])
        if step.operation == "split_scale" and not dict(step.arguments)["dissolve_unit"]:
            attempt(
                step.after, "join_scales", first=dict(step.arguments)["scale"], second=new_ids[0]
            )
    return steps


def assert_sheet_laws(s: ScalableDiagram, sample_limit: int) -> list[str]:
    """Check every applicable step, split-join identity, and split-equals-peel; the names."""
    steps = applicable_steps(s)
    for step in steps:
        samples = default_samples(step.before, step.after, limit=sample_limit)
        check = check_sheet_step(step, samples)
        assert check.ok, (step.operation, step.arguments, check.reason)
        if step.operation != "split_scale":
            continue
        arguments = dict(step.arguments)
        scale = s.scale(arguments["scale"])  # type: ignore[arg-type]
        if not arguments["dissolve_unit"]:
            new = ScaleId(max(int(sc.id) for sc in s.scales) + 1)
            assert join_scales(step.after, scale.id, new).after == s
        rest = scale.multiplicity.to_sympy() - arguments["first"].to_sympy()  # type: ignore[attr-defined]
        if rest == 1 and arguments["dissolve_unit"] and scale.parent is None:
            rank = BangBoxId([sc.id for sc in s.scales].index(scale.id))
            peeled = peel_one(from_scalable(s), rank).diagram
            assert isomorphic(from_scalable(step.after), peeled), step.arguments
    return sorted({step.operation for step in steps})


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_applicable_sheet_step_checks(name: str) -> None:
    s = to_scalable(FAMILIES[name]())
    assert_sheet_laws(s, sample_limit=8)


def test_catalogue_exercises_every_operation() -> None:
    seen: set[str] = set()
    for build in FAMILIES.values():
        seen.update(
            op for op in (step.operation for step in applicable_steps(to_scalable(build())))
        )
    assert seen == set(SHEET_OPERATIONS)


@pytest.mark.slow
class TestRandomSheetSteps:
    """Every applicable sheet step on seeded random bang-boxed diagrams checks."""

    @pytest.mark.parametrize("seed", range(SEEDS))
    def test_steps_check(self, seed: int) -> None:
        diagram = seeded_family(seed)
        if diagram is None:
            pytest.skip(f"no valid boxed diagram in {ATTEMPTS} attempts")
        assert_sheet_laws(to_scalable(diagram), sample_limit=RANDOM_SAMPLE_LIMIT)
