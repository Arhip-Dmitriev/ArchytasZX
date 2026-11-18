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

"""Kernel peel steps exported over linked diagrams (REMAINING_BUGS_SPEC 79.16a): the
cluster_state_2d peel proved by ``evalL_peel``, its reverse refused, the semantic export linking it
by a ``net`` axiom, and Lean accepting the proof and rejecting a wrong copy count; link ports on
the boundary behind an identity spider (79.16e); identity slides through
``evalL_slide`` (79.16d); both sides of a peel or slide written with each boundary link port the
ghost of its slot, proved equal to the identity forms, the phaseless index stated as zero, slides
read through the slot data of both sides' box groups (79.16f); the engine's instantiated boundary
in the key order of ``LinkedSplice.lean`` (``flatAt``), peels and slides read at those flat
positions, a peel whose copy slots moved refused here and in Lean (79.16h, 79.16i); peels of the
open grids, the chain end on the boundary held in the box of the identity the peel puts behind
it, at the engine's flat positions, checked in Lean, and peels of the reversed row chain
(the first copy peeled) through ``evalL_peel_first`` (79.16l); a peel of the binder chains of
BINDERS, whose box index is read through ``cph`` and ``cdim`` and whose copies read their template
at the constant copy, through ``evalX_peel_flat`` and ``evalGX_ghostAt`` (79.16l); a peel of the
first copy of a reversed chain whose box has an index, its kept copies read one copy higher,
through ``evalX_peel_first``, and a copy reading an inner box's index one level up (79.16l)."""

from __future__ import annotations

import itertools
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import cast

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram import bangbox
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.graph import Diagram, Direction, NodeId
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel, lean_linked
from archytaszx.semantics.decide import decide_equal
from archytaszx.semantics.induction import rename_multiplicity
from archytaszx.semantics.lean_net import check, lean_project
from archytaszx.semantics.prove import InductionSide, _with_scalar_index
from tests.test_nested_chain import BINDERS, GRID, _grid_pair, _open_grid


def _peels(t: kernel.Theorem, seen: set[int] | None = None) -> Iterator[kernel.Theorem]:
    seen = set() if seen is None else seen
    if id(t) in seen:
        return
    seen.add(id(t))
    if t.inference.kind == "peel":
        yield t
    for p in t.inference.premises:
        yield from _peels(p, seen)


def _cluster() -> kernel.Theorem:
    left, right = _grid_pair(1)
    return kernel.from_decision(decide_equal(left, right, samples=[]), left, right)


def test_the_cluster_state_2d_peel_exports_through_evalL_peel() -> None:
    theorem = _cluster()
    peels = list(_peels(theorem))
    assert peels
    for t in peels:
        source = lean_linked.export_peel(t.lhs, t.rhs)
        assert source is not None and "evalL_peel_flat ok flat" in source
        assert lean_linked.export_peel(t.rhs, t.lhs) is None
    source = kernel.lean(theorem, "main", semantic=True)
    assert "namespace ArchytasZX.Export.P" in source
    assert "a joint peel of one box family" not in source


def _succ_peels(t: kernel.Theorem) -> Iterator[tuple[Diagram, Diagram]]:
    """The successor peels over ``m`` of the induction steps under ``t``: ``d`` at
    ``m = k + 1`` and its peel."""
    seen: set[int] = set()
    stack = [t]
    while stack:
        s = stack.pop()
        if id(s) in seen:
            continue
        seen.add(id(s))
        stack.extend(s.inference.premises)
        data = s.inference.data
        if s.inference.kind != "induction" or str(data["index"]) != "m":
            continue
        own, other = (s.lhs, s.rhs) if data["side"] is InductionSide.START else (s.rhs, s.lhs)
        value = Mult(str(data["step"])) + 1
        for d, peeled in ((own, data["peeled"]), (other, data["other_peeled"])):
            before = rename_multiplicity(d, "m", value)
            yield _with_scalar_index(before, "m", value), cast(Diagram, peeled)


def test_successor_peels_with_boundary_link_ports_export() -> None:
    theorem = _cluster()
    pairs = list(_succ_peels(theorem))
    assert len(pairs) == 4
    for before, peeled in pairs:
        source = lean_linked.export_peel(before, peeled)
        assert source is not None and "evalL_peel_flat ok flat" in source
    assert "\naxiom peel" not in kernel.lean(theorem, "main", semantic=True)


def test_peel_sides_are_written_with_ghosts_not_identities() -> None:
    for before, peeled in _succ_peels(_cluster()):
        source = lean_linked.export_peel(before, peeled)
        assert source is not None
        assert "evalL_dropAt okk1" in source and "evalG_ghostAt okb1" in source
        sides = source.split("def L : ")[1].split("def zero")[0]
        idents = re.findall(r"abbrev x\w+ : DNode := ⟨(\d+),", source)
        assert idents and all(f"⟨{x}, " not in sides for x in idents)
        assert re.search(r"abbrev gsL : Ghosts := \[\(", source)
        assert "E.phase zero k = 0" in source


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_a_boundary_link_peel_and_refuses_a_moved_end(tmp_path: object) -> None:
    before, peeled = next(_succ_peels(_cluster()))
    source = lean_linked.export_peel(before, peeled)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/peel.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    found = re.search(r"links := \[⟨0, (⟨[^⟩]*⟩, ){2}(⟨[^⟩]*⟩)", source.split("def L' ")[1])
    assert found is not None
    bad = source.replace(found[0], found[0].replace(found[2], "⟨0, false, 0⟩"))
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_peel_and_refuses_a_wrong_copy_count(tmp_path: object) -> None:
    (t, *_) = _peels(_cluster())
    source = lean_linked.export_peel(t.lhs, t.rhs)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/peel.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("+ 1 →", "+ 2 →")
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


def _slides(t: kernel.Theorem) -> list[kernel.Theorem]:
    """The identity slide steps under ``t``."""
    seen: set[int] = set()
    out, stack = [], [t]
    while stack:
        s = stack.pop()
        if id(s) not in seen:
            seen.add(id(s))
            stack.extend(s.inference.premises)
            if s.inference.kind == "identity_slide":
                out.append(s)
    return out


def test_the_cluster_state_2d_identity_slide_exports_through_one_dropped_diagram() -> None:
    theorem = _cluster()
    slides = _slides(theorem)
    assert slides
    for t in slides:
        source = lean_linked.export_slide(t.lhs, t.rhs)
        assert source is not None and "evalL_dropAt oksb1" in source
        assert "evalL_dropAt oksk1" in source and "evalG_flatAt layout" in source
    source = kernel.lean(theorem, "main", semantic=True)
    assert "namespace ArchytasZX.Export.S" in source
    assert "a boxed identity spider moved" not in source


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_the_slide_and_refuses_a_phased_identity(tmp_path: object) -> None:
    (t, *_) = _slides(_cluster())
    source = lean_linked.export_slide(t.lhs, t.rhs)
    assert source is not None
    ok, output = check(source, f"{tmp_path}/slide.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = re.sub(r"def zero : ℕ := (\d+)", lambda m: f"def zero : ℕ := {int(m[1]) + 1}", source)
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok
    blocks = re.search(r"def G : List \(ℕ × List \(List ℕ\)\) := (.*)\n", source)
    assert blocks is not None
    merged = re.sub(
        r"(def G' : List \(ℕ × List \(List ℕ\)\) := ).*\n", rf"\g<1>{blocks[1]}\n", source
    )
    assert merged != source
    ok, output = check(merged, f"{tmp_path}/merged.lean")
    assert not ok


def test_a_slide_whose_slot_blocks_differ_is_not_exported() -> None:
    (t, *_) = _slides(_cluster())
    after = t.rhs.copy()
    box = next(b for b, x in after.bang_boxes.items() if x.groups)
    after.set_bang_box_extension(box, groups=())
    assert lean_linked.export_slide(t.lhs, after) is None


NEST = """dim d
count n
count m
a = Z(1 -> 1, d)
b = Z(1 -> 1, d)
e = Z(1 -> 1, d)
f = Z(1 -> 1, d)
repeat n as P: a, b, e
repeat m as Q in P: b
outputs b.out0, f.out0, a.out0, e.out0
inputs e.in0, f.in0, b.in0, a.in0"""
"""Three nodes in a box, one of them in a box within, beside an idle wire, legs on both sides."""

Slot = tuple[int, tuple[int, ...]]


def _engine_flat(
    d: Diagram,
    order: Sequence[str],
    values: Mapping[str, int],
    held: Mapping[lean_linked.Port, lean_linked.LBox] | None = None,
) -> list[Slot]:
    """The instantiated boundary of ``d`` (symbols instantiated in ``order``) as slots of its
    linked ghost form (boundary link ports of ``held`` in their held boxes) with copy indices
    outermost box first; a boundary node the instantiation adds reads the slot and copy index of
    the leg it is wired to, its link's box left out."""
    prov: dict[NodeId, tuple[NodeId, dict[int, int]]] = {}
    scopes = {int(b): set(x.node_scope) for b, x in d.bang_boxes.items()}
    real = bangbox._instantiate_node_scope

    def hooked(diagram: Diagram, box: bangbox.BangBox, k: int) -> list[dict[NodeId, NodeId]]:
        held = {prov.get(y, (y, {}))[0] for y in box.node_scope}
        maps = real(diagram, box, k)
        for j, m in enumerate(maps):
            for old, new in m.items():
                o, c = prov.get(old, (old, {}))
                top = max(
                    (b for b, s in scopes.items() if o in s and s <= held),
                    key=lambda b: len(scopes[b]),
                    default=None,
                )
                prov[new] = (o, c if top is None else {**c, top: j})
        return maps

    bangbox._instantiate_node_scope = hooked
    try:
        w = d
        for name in order:
            w = bangbox.instantiate_symbol(w, name, values[name])
        w = bangbox.expand_concrete_boxes(w)
    finally:
        bangbox._instantiate_node_scope = real
    tables = lean_linked._Tables()
    ghost = 1 + max(int(x) for x in d.nodes)
    D = lean_linked.linked(d, tables, ghost=ghost)
    ports = [lean_linked._port(r) for r in [*d.boundary_outputs, *d.boundary_inputs]]
    link_box = {
        lean_linked._port(r): int(b) for b, x in d.bang_boxes.items() for pq in x.links for r in pq
    }
    out = []
    for r in [*w.boundary_outputs, *w.boundary_inputs]:
        o, c = prov.get(r.node_id, (r.node_id, {}))
        port = (int(o), r.direction is Direction.OUTPUT, r.index)
        if o not in d.nodes:
            other = next(
                x.b if x.a.node_id == r.node_id else x.a
                for x in w.wires
                if r.node_id in (x.a.node_id, x.b.node_id)
            )
            o, c = prov[other.node_id]
            port = (int(o), other.direction is Direction.OUTPUT, other.index)
            i = ports.index(port)
            boxes = [b.id for b in D.node_boxes(int(o)) if b.id != link_box[port]]
            out.append((i, tuple(c[b] for b in boxes)))
            continue
        i = ports.index(port)
        ghosted = D.bnd()[i][0] == ghost and port not in (held or {})
        out.append((i, () if ghosted else tuple(c[b.id] for b in D.node_boxes(int(o)))))
    return out


def _model_flat(
    d: Diagram,
    values: Mapping[str, int],
    held: Callable[[lean_linked._Tables], Mapping[lean_linked.Port, lean_linked.LBox]]
    | None = None,
) -> list[Slot]:
    """The slots of ``d``'s linked ghost form (boundary link ports of ``held`` in their held
    boxes) in increasing ``key`` (``LinkedSplice.lean``)."""
    tables = lean_linked._Tables()
    D = lean_linked.linked(
        d, tables, ghost=1 + max(int(x) for x in d.nodes), held=held(tables) if held else None
    )
    sd = lean_linked.slot_data(D, lean_linked.groups(d))
    ks = {m: int(sp.sympify(e).subs(values)) for e, m in tables.mults.items()}

    def key(v: Slot) -> list[int]:
        anch = lean_linked.anchors(sd, v[0])[0]
        return [x for pair in itertools.zip_longest(anch, v[1]) for x in pair if x is not None]

    return sorted(
        (
            (i, t)
            for i in range(sd[0] + sd[1])
            for t in itertools.product(*(range(ks[m]) for m in lean_linked.anchors(sd, i)[1]))
        ),
        key=key,
    )


def _nest_grouped(groups: Sequence[Sequence[int]]) -> Diagram:
    d = parse_dsl(NEST)
    box = next(b for b, x in d.bang_boxes.items() if x.parent is None)
    nodes = sorted(d.bang_boxes[box].node_scope)
    d.set_bang_box_extension(box, groups=tuple(frozenset(nodes[k] for k in g) for g in groups))
    return d


def test_the_flat_positions_are_the_engine_boundary_order() -> None:
    theorem = _cluster()
    sides = [(t.lhs, t.rhs) for t in [*_peels(theorem), *_slides(theorem)]]
    cases = [d for pair in [*sides, *_succ_peels(theorem)] for d in pair]
    cases.append(parse_dsl(NEST))
    for d in cases:
        names = sorted(bangbox.free_mult_symbols(d))
        values = {s: 2 + j for j, s in enumerate(names)}
        for order in itertools.permutations(names):
            assert _engine_flat(d, order, values) == _model_flat(d, values), (names, order)
    for groups in ([[0, 1], [2]], [[2], [0, 1]], [[0], [1, 2]], [[1], [2]]):
        d = _nest_grouped(groups)
        values = {"n": 2, "m": 3}
        assert _engine_flat(d, ["n", "m"], values) == _model_flat(d, values), groups


def _open_peel(head: bool, tail: bool, reverse: bool = False) -> tuple[Diagram, Diagram]:
    """An open grid (with ``reverse`` its row chain reversed, with neither end open the closed
    grid) at ``s + 1`` rows and its peel of the row box."""
    text = _open_grid(head, tail) if head or tail else GRID
    if reverse:
        text = text.replace("chain R: v.out2 -> hv.in0", "chain R: hv.in0 -> v.out2")
    d = rename_multiplicity(parse_dsl(text), "n", Mult("s") + 1)
    row = next(b for b in d.bang_boxes.values() if b.parent is None and b.links)
    return d, bangbox.peel_one(d, row.id).diagram


OPEN = [(True, False), (False, True), (True, True)]
"""The open grids: top, bottom or both vertical legs on the boundary."""


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("head", "tail"), OPEN)
def test_open_grid_flat_positions_are_the_engine_boundary_order(
    head: bool, tail: bool, reverse: bool
) -> None:
    before, after = _open_peel(head, tail, reverse)

    def held(tables: lean_linked._Tables) -> Mapping[lean_linked.Port, lean_linked.LBox]:
        pe = lean_linked.peel_ends(before, after, tables, reverse)
        assert pe is not None and pe.held
        return pe.held

    ports = held(lean_linked._Tables())
    values = {"s": 2, "l": 3}
    for order in (["s", "l"], ["l", "s"]):
        assert _engine_flat(before, order, values, ports) == _model_flat(before, values, held)
        assert _engine_flat(after, order, values) == _model_flat(after, values)


@pytest.mark.parametrize(("head", "tail"), OPEN)
def test_a_peel_of_an_open_grid_exports_through_boxed_ghosts(head: bool, tail: bool) -> None:
    before, after = _open_peel(head, tail)
    source = lean_linked.export_peel(before, after)
    assert source is not None and "evalL_peel_flat ok flat" in source
    assert "evalG_ghostAt okb1" in source
    assert lean_linked.export_peel(after, before) is None


@pytest.mark.parametrize(("head", "tail"), [(False, False), *OPEN])
def test_a_peel_of_a_reversed_chain_exports_through_evalL_peel_first(
    head: bool, tail: bool
) -> None:
    before, after = _open_peel(head, tail, True)
    source = lean_linked.export_peel(before, after)
    assert source is not None and "evalL_peel_first rb rb' rk rk' ok flat hcp" in source
    assert "theorem flat : PeelFirst Lb Lk" in source
    assert lean_linked.export_peel(after, before) is None


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_a_reversed_chain_peel_and_refuses_a_wrong_copy_count(tmp_path: object) -> None:
    source = lean_linked.export_peel(*_open_peel(False, False, True))
    assert source is not None
    ok, output = check(source, f"{tmp_path}/rev.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("+ 1 →", "+ 2 →")
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_an_open_grid_peel_and_refuses_a_wrong_copy_count(tmp_path: object) -> None:
    source = lean_linked.export_peel(*_open_peel(True, False))
    assert source is not None
    ok, output = check(source, f"{tmp_path}/open.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = source.replace("+ 1 →", "+ 2 →")
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


def _binders_peel() -> tuple[Diagram, Diagram]:
    """BINDERS at ``s + 1`` rows and its peel of the row box."""
    d = rename_multiplicity(parse_dsl(BINDERS), "n", Mult("s") + 1)
    row = next(b for b in d.bang_boxes.values() if b.parent is None and b.links)
    return d, bangbox.peel_one(d, row.id).diagram


def test_a_binder_chain_peel_exports_through_evalX_peel_flat() -> None:
    before, after = _binders_peel()
    source = lean_linked.export_peel(before, after)
    assert source is not None and "(evalX_peel_flat ok flat E ks hk y)" in source
    assert "theorem ok : PeelX Lb Lk" in source and "evalGX_ghostAt okb1" in source
    assert "L'.evalGX gsL' E ks (flatAt (L'.slotData G') ks y)" in source
    copies = re.findall(r"\(⟨\d+, \w+, \d+⟩, \d+, 1, \d+\)", source.split("def L' ")[1])
    assert len(copies) >= 2
    assert lean_linked.export_peel(after, before) is None


def test_a_binder_copy_whose_literal_is_not_its_template_at_the_copy_is_not_exported() -> None:
    before, after = _binders_peel()
    copy = next(x for x, n in after.nodes.items() if x not in before.nodes and len(n.inputs) == 2)
    node = after.nodes[copy]
    after.set_port_dims(copy, [node.inputs[0].dim, node.outputs[0].dim], [node.outputs[0].dim])
    assert lean_linked.export_peel(before, after) is None


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_a_binder_chain_peel_and_refuses_a_copy_read_at_zero(tmp_path: object) -> None:
    source = lean_linked.export_peel(*_binders_peel())
    assert source is not None
    ok, output = check(source, f"{tmp_path}/binders.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    head, tail = source.split("def L' : LDiagram :=", 1)
    found = re.search(r"\((⟨\d+, \w+, \d+⟩), (\d+), 1, (\d+)\)", tail)
    assert found is not None
    bad = f"({found[1]}, {found[2]}, 0, {found[3]})"
    ok, output = check(
        head + "def L' : LDiagram :=" + tail.replace(found[0], bad, 1), f"{tmp_path}/bad.lean"
    )
    assert not ok


REV_BINDER = (
    GRID.replace("repeat n as R: v, hv, hh, hs, he", "repeat n as R index m: v, hv, hh, hs, he, q")
    .replace("hv = F(d)", "hv = F(d)\nq = B(literal 2, literal 2^m)")
    .replace("outputs v.out0", "inputs q.in0, q.in1\noutputs v.out0, q.out0")
    .replace("chain R: v.out2 -> hv.in0", "chain R: hv.in0 -> v.out2")
)
"""GRID with the row chain reversed, the row box indexed by ``m`` and holding a binder
B(2, 2^m) with its legs on the boundary."""


def _rev_binder_peel() -> tuple[Diagram, Diagram]:
    """REV_BINDER at ``s + 1`` rows and its peel of the row box (the first copy)."""
    d = rename_multiplicity(parse_dsl(REV_BINDER), "n", Mult("s") + 1)
    row = next(b for b in d.bang_boxes.values() if b.parent is None and b.links)
    return d, bangbox.peel_one(d, row.id).diagram


def test_a_reversed_chain_peel_with_an_index_exports_through_evalX_peel_first() -> None:
    before, after = _rev_binder_peel()
    source = lean_linked.export_peel(before, after)
    assert source is not None and "(evalX_peel_first fo okx flat E ks hk hm y)" in source
    assert "theorem okx : FirstX Lb Lk" in source and "theorem rb : RevOK Lb.strip Lbr" in source
    m1 = re.search(r"ks \d+ = ks \d+ \+ 1 → ks (\d+) = 1 →", source)
    assert m1 is not None
    kept = re.findall(rf"\(⟨\d+, \w+, \d+⟩, \d+, 1, {m1[1]}\)", source.split("def L' ")[1])
    assert len(kept) >= 2
    assert lean_linked.export_peel(after, before) is None


def test_a_kept_binder_not_its_template_one_copy_higher_is_not_exported() -> None:
    before, after = _rev_binder_peel()
    kept = next(x for x, n in before.nodes.items() if n.generator_type.name == "B")
    node = after.nodes[kept]
    after.set_port_dims(kept, [node.inputs[0].dim, node.outputs[0].dim], [node.outputs[0].dim])
    assert lean_linked.export_peel(before, after) is None


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_a_reversed_chain_peel_with_an_index_and_refuses_a_kept_copy_read_at_t(
    tmp_path: object,
) -> None:
    source = lean_linked.export_peel(*_rev_binder_peel())
    assert source is not None
    ok, output = check(source, f"{tmp_path}/revx.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    head, tail = source.split("def L' : LDiagram :=", 1)
    found = re.search(r"\((⟨\d+, \w+, \d+⟩), (\d+), 1, (\d+)\)", tail)
    assert found is not None
    bad = f"({found[1]}, {found[2]}, 0, {found[3]})"
    ok, output = check(
        head + "def L' : LDiagram :=" + tail.replace(found[0], bad, 1), f"{tmp_path}/bad.lean"
    )
    assert not ok


INNER = (
    BINDERS.replace("repeat n as R index m: b, u, w", "repeat n as R index m: b, u, w, q")
    .replace("repeat l as C in R: b", "repeat l as C index k in R: b, q")
    .replace("w = Z(1 -> 0, literal 2)", "w = Z(1 -> 0, literal 2)\nq = B(literal 2, literal 2^k)")
    .replace("inputs b.in1, b.in0", "inputs b.in1, b.in0, q.in0, q.in1")
    .replace("outputs b.out0", "outputs b.out0, q.out0")
)
"""BINDERS with the column box indexed by ``k`` and holding a binder B(2, 2^k) with its legs on
the boundary."""


def _inner_peel() -> tuple[Diagram, Diagram]:
    """INNER at ``s + 1`` rows and its peel of the row box."""
    d = rename_multiplicity(parse_dsl(INNER), "n", Mult("s") + 1)
    row = next(b for b in d.bang_boxes.values() if b.parent is None and b.links)
    return d, bangbox.peel_one(d, row.id).diagram


def _inner_copy_reading(source: str) -> tuple[str, str]:
    """The phase reading of the copy of the inner binder in ``L'`` and that reading at the
    template's box."""
    cph = [
        re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", m) for m in re.findall(r"cph := .*", source)
    ]
    before, after = set(cph[0]), set(cph[1])
    (x, y, a, m), *_ = sorted(q for q in after - before if q[2] == "0")
    (tx, ty, *_), *_ = sorted(q for q in before if q[1] != "0")
    assert tx != x
    return f"({x}, {y}, {a}, {m})", f"({x}, {ty}, {a}, {m})"


def test_a_peel_reading_an_inner_box_index_exports_with_the_copy_one_level_up() -> None:
    before, after = _inner_peel()
    source = lean_linked.export_peel(before, after)
    assert source is not None and "(evalX_peel_flat ok flat E ks hk y)" in source
    good, _ = _inner_copy_reading(source)
    assert good in source.split("def L' ")[1]


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_an_inner_box_reading_on_a_copy_and_refuses_it_at_the_template_box(
    tmp_path: object,
) -> None:
    source = lean_linked.export_peel(*_inner_peel())
    assert source is not None
    ok, output = check(source, f"{tmp_path}/inner.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    good, bad = _inner_copy_reading(source)
    ok, output = check(source.replace(good, bad), f"{tmp_path}/bad.lean")
    assert not ok


PLAIN = """count n
b = X(1 -> 1, literal 2)
q = B(literal 2, literal 2^m)
repeat n as R index m: b, q
chain R: b.out0 -> b.in0
inputs b.in0, q.in0, q.in1
outputs b.out0, q.out0"""
"""A chain of X(1 -> 1) of dimension 2 through a box of index m that also holds a binder
B(2, 2^m), both chain ends and the binder's legs on the boundary."""


def _plain_peel() -> tuple[Diagram, Diagram]:
    """PLAIN at ``s + 1`` copies and its peel of the box."""
    d = rename_multiplicity(parse_dsl(PLAIN), "n", Mult("s") + 1)
    row = next(b for b in d.bang_boxes.values() if b.parent is None and b.links)
    return d, bangbox.peel_one(d, row.id).diagram


def _read_dropped_leg(source: str) -> str:
    """``source`` with the leg a dropped identity meets given a leg-dimension reading in ``Lk``."""
    found = re.search(r"theorem okk1 : DropOKX .* (⟨\d+, \w+, \d+⟩) :=", source)
    assert found is not None
    head, tail = source.split("def Lk : LDiagram :=", 1)
    tail = tail.replace("cdim := [", f"cdim := [({found[1]}, 0, 0, 0), ", 1)
    return head + "def Lk : LDiagram :=" + tail


def test_an_identity_wired_to_a_node_leg_on_a_side_with_an_index_drops_through_evalX_dropAt() -> (
    None
):
    before, after = _plain_peel()
    source = lean_linked.export_peel(before, after)
    assert source is not None and "(evalX_peel_flat ok flat E ks hk y)" in source
    assert "theorem okk1 : DropOKX (Lk.moveLast xk1.id)" in source
    assert "(evalX_dropAt okk1 E hE ks σ)" in source
    assert _read_dropped_leg(source) != source


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_checks_an_identity_drop_with_an_index_and_refuses_it_at_a_read_leg(
    tmp_path: object,
) -> None:
    source = lean_linked.export_peel(*_plain_peel())
    assert source is not None
    ok, output = check(source, f"{tmp_path}/drop.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    ok, output = check(_read_dropped_leg(source), f"{tmp_path}/bad.lean")
    assert not ok


BINPLAIN = """count n
q = B(literal 2, literal 2^m)
repeat n as R index m: q
chain R: q.out0 -> q.in1
inputs q.in1, q.in0
outputs q.out0"""
"""A chain of binders B(2, 2^m) through a box of index m, its ends and the binders' first inputs
on the boundary."""

BINROW = BINDERS.replace("count n, l", "count n").replace("repeat l as C in R: b\n", "")
"""BINDERS with one column."""


def _binder_peel(text: str) -> tuple[Diagram, Diagram]:
    """``text`` at ``s + 1`` copies and its peel of the chain box."""
    d = rename_multiplicity(parse_dsl(text), "n", Mult("s") + 1)
    row = next(b for b in d.bang_boxes.values() if b.parent is None and b.links)
    return d, bangbox.peel_one(d, row.id).diagram


@pytest.mark.parametrize("text", [BINPLAIN, BINROW])
def test_an_identity_wired_to_a_read_leg_drops_over_evalN(text: str) -> None:
    source = lean_linked.export_peel(*_binder_peel(text))
    assert source is not None and "(evalN_peel_flat ok flat E ks hk y)" in source
    assert "theorem okk1 : DropN (Lk.moveLast xk1.id)" in source
    assert "(evalN_dropAt okk1 E hE ks σ)" in source and "(evalGXR_ghostAt okb1" in source
    assert "L.evalGXR gsL grL E ks (flatAt (L.slotData G) ks y)" in source
    assert re.search(r"cnd := \[\(\d+, 0, 1, \d+\)\]", source)


def _unread_bound(source: str) -> str:
    """``source`` with the read bounds of ``L``'s ghosts dropped."""
    found = re.search(r"abbrev grL : GReads := \[.+\]", source)
    assert found is not None
    return source.replace(found[0], "abbrev grL : GReads := []")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
@pytest.mark.parametrize("text", [BINPLAIN, BINROW])
def test_lean_checks_an_identity_drop_at_a_read_leg_and_refuses_an_unread_bound(
    text: str, tmp_path: object
) -> None:
    source = lean_linked.export_peel(*_binder_peel(text))
    assert source is not None
    ok, output = check(source, f"{tmp_path}/read.lean")
    assert ok and "error" not in output and "sorry" not in output, output
    bad = _unread_bound(source)
    assert bad != source
    ok, output = check(bad, f"{tmp_path}/bad.lean")
    assert not ok


def _moved_copy_slots() -> tuple[Diagram, Diagram]:
    """A successor peel whose peeled side has its first and last outputs swapped."""
    before, after = next(_succ_peels(_cluster()))
    outs = list(after.boundary_outputs)
    outs[0], outs[-1] = outs[-1], outs[0]
    moved = after.copy()
    moved.set_boundary_outputs(outs)
    return before, moved


def test_peels_read_flat_positions_and_refuse_moved_copy_slots() -> None:
    for before, after in _succ_peels(_cluster()):
        source = lean_linked.export_peel(before, after)
        assert source is not None and "evalL_peel_flat ok flat" in source
        assert "flatAt (L'.slotData G') ks y" in source
    assert lean_linked.export_peel(*_moved_copy_slots()) is None


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
def test_lean_refuses_a_peel_whose_copy_slots_moved(tmp_path: object) -> None:
    real = lean_linked.peel_flat_ok
    lean_linked.peel_flat_ok = lambda *_: True
    try:
        source = lean_linked.export_peel(*_moved_copy_slots())
    finally:
        lean_linked.peel_flat_ok = real
    assert source is not None
    ok, output = check(source, f"{tmp_path}/moved.lean")
    assert not ok and "PeelFlat" in source, output
