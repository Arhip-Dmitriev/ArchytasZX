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

"""Chain links through a child box (REMAINING_BUGS_SPEC 72.6): copy j of the child in one chain
copy is linked to copy j in the next, whichever count is instantiated first."""

from __future__ import annotations

import os

import pytest

from archytaszx.diagram.bangbox import BangBoxGrammarError, instantiate_symbol, peel_one
from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics.check import compare
from archytaszx.semantics.lean_net import check, lean_project

RING = """dim d
count n, l
v = Z(1 -> 2, d)
u = Z(1 -> 1, d)
wire v.out1 -- v.in0
repeat n as R: v, u
repeat l as C in R: v
chain R: v.out1 -> v.in0 cyclic
inputs u.in0
outputs v.out0, u.out0"""
"""n rings of l spiders: spider j of row m wired to spider j of row m + 1 (mod n)."""

GRID = """dim d
count n, l
v = Z(2 -> 3, d)
hv = F(d)
hh = F(d)
hs = X(0 -> 1, d)
he = Z(1 -> 0, d)
vs = X(0 -> 1, d)
ve = Z(1 -> 0, d)
wire hv.out0 -- v.in0, hh.out0 -- v.in1, hs.out0 -- hh.in0, v.out1 -- he.in0
wire vs.out0 -- hv.in0, v.out2 -- ve.in0
repeat n as R: v, hv, hh, hs, he
repeat l as C in R: v, hv, hh
chain C: v.out1 -> hh.in0
chain R: v.out2 -> hv.in0
repeat l as PH: vs
repeat l as PT: ve
outputs v.out0"""
"""The n x l grid graph state: rows chained through the column box C, the vertical ends in the
boxes PH and PT beside the row chain."""


def _ring(n: int, l: int) -> str:
    lines = ["dim d"] + [f"u{r} = Z(1 -> 1, d)" for r in range(n)]
    lines += [f"v{r}_{c} = Z(1 -> 2, d)" for r in range(n) for c in range(l)]
    lines.append(
        "wire "
        + ", ".join(f"v{r}_{c}.out1 -- v{(r + 1) % n}_{c}.in0" for r in range(n) for c in range(l))
    )
    lines.append("inputs " + ", ".join(f"u{r}.in0" for r in range(n)))
    outs = [x for r in range(n) for x in [*(f"v{r}_{c}.out0" for c in range(l)), f"u{r}.out0"]]
    lines.append("outputs " + ", ".join(outs))
    return "\n".join(lines)


def _grid(n: int, l: int, shift: int = 0, head: bool = False, tail: bool = False) -> str:
    """The flat n x l grid; with ``head`` (``tail``) the top (bottom) vertical legs are
    boundary inputs (outputs)."""
    lines = ["dim d"]
    wires, inputs, ends = [], [], []
    for r in range(n):
        lines += [f"hs{r} = X(0 -> 1, d)", f"he{r} = Z(1 -> 0, d)"]
        for c in range(l):
            lines += [f"v{r}_{c} = Z(2 -> 3, d)", f"hv{r}_{c} = F(d)", f"hh{r}_{c} = F(d)"]
            wires += [f"hv{r}_{c}.out0 -- v{r}_{c}.in0", f"hh{r}_{c}.out0 -- v{r}_{c}.in1"]
            wires.append(f"{f'hs{r}.out0' if c == 0 else f'v{r}_{c - 1}.out1'} -- hh{r}_{c}.in0")
            if r == 0 and head:
                inputs.append(f"hv0_{c}.in0")
                continue
            up = f"vs{c}.out0" if r == 0 else f"v{r - 1}_{(c + shift) % l}.out2"
            wires.append(f"{up} -- hv{r}_{c}.in0")
        wires.append(f"v{r}_{l - 1}.out1 -- he{r}.in0")
    for c in range(l):
        lines += [] if head else [f"vs{c} = X(0 -> 1, d)"]
        lines += [] if tail else [f"ve{c} = Z(1 -> 0, d)"]
        top = f"v{n - 1}_{c}.out2" if n else (None if head else f"vs{c}.out0")
        bottom = None if tail else f"ve{c}.in0"
        if top is None and bottom is None:
            lines.append(f"id{c} = Z(1 -> 1, d)")
            inputs.append(f"id{c}.in0")
            ends.append(f"id{c}.out0")
        elif top is None:
            inputs.append(f"{bottom}")
        elif bottom is None:
            ends.append(top)
        else:
            wires.append(f"{top} -- {bottom}")
    if wires:
        lines.append("wire " + ", ".join(wires))
    if inputs:
        lines.append("inputs " + ", ".join(inputs))
    outputs = [f"v{r}_{c}.out0" for r in range(n) for c in range(l)]
    lines.append("outputs " + ", ".join(outputs + ends))
    return "\n".join(lines)


def _at(text: str, order: tuple[str, str], n: int, l: int) -> Diagram:
    diagram = parse_dsl(text)
    for name in order:
        diagram = instantiate_symbol(diagram, name, {"n": n, "l": l}[name])
    return diagram


@pytest.mark.parametrize("order", [("n", "l"), ("l", "n")])
@pytest.mark.parametrize(("n", "l"), [(1, 1), (2, 2), (2, 3), (3, 2)])
def test_a_cyclic_chain_through_a_child_links_copy_to_copy(
    order: tuple[str, str], n: int, l: int
) -> None:
    result = compare(_at(RING, order, n, l), parse_dsl(_ring(n, l)), {"d": 2})
    assert result.matched, result.reason


@pytest.mark.parametrize("order", [("n", "l"), ("l", "n")])
@pytest.mark.parametrize(("n", "l"), [(1, 1), (2, 2), (2, 3), (3, 2), (3, 1)])
def test_an_open_chain_through_a_child_is_the_grid(order: tuple[str, str], n: int, l: int) -> None:
    result = compare(_at(GRID, order, n, l), parse_dsl(_grid(n, l)), {"d": 2})
    assert result.matched, result.reason


def test_a_shifted_grid_differs() -> None:
    assert not compare(_at(GRID, ("n", "l"), 2, 2), parse_dsl(_grid(2, 2, 1)), {"d": 2}).matched


def test_the_column_chain_has_no_peel() -> None:
    diagram = parse_dsl(GRID)
    column = next(box for box in diagram.bang_boxes.values() if box.parent is not None)
    with pytest.raises(BangBoxGrammarError, match="through a child box"):
        peel_one(diagram, column.id)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("k", "l"), [(0, 1), (1, 2), (2, 2), (1, 3)])
def test_the_row_chain_peels_through_its_child(reverse: bool, k: int, l: int) -> None:
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics.induction import rename_multiplicity

    text = (
        GRID.replace("chain R: v.out2 -> hv.in0", "chain R: hv.in0 -> v.out2") if reverse else GRID
    )
    diagram = rename_multiplicity(parse_dsl(text), "n", Mult("s") + 1)
    row = next(box for box in diagram.bang_boxes.values() if box.parent is None and box.links)
    peeled = peel_one(diagram, row.id).diagram
    at = {"s": k, "l": l, "d": 2}
    expected = instantiate_symbol(instantiate_symbol(diagram, "s", k), "l", l)
    result = compare(instantiate_symbol(instantiate_symbol(peeled, "s", k), "l", l), expected, at)
    assert result.matched, result.reason
    if not reverse:
        assert compare(expected, parse_dsl(_grid(k + 1, l)), at).matched


def _open_grid(head: bool, tail: bool) -> str:
    """GRID with the top (``head``) or bottom (``tail``) vertical legs on the boundary."""
    text = GRID
    inputs, outputs = "", ["v.out0"]
    if head:
        text = text.replace("repeat l as PH: vs\n", "").replace("vs = X(0 -> 1, d)\n", "")
        text = text.replace("vs.out0 -- hv.in0, ", "")
        inputs = "inputs hv.in0\n"
    if tail:
        text = text.replace("repeat l as PT: ve\n", "").replace("ve = Z(1 -> 0, d)\n", "")
        text = text.replace(", v.out2 -- ve.in0", "").replace("wire v.out2 -- ve.in0\n", "")
        outputs.append("v.out2")
    return text.replace("outputs v.out0", inputs + "outputs " + ", ".join(outputs))


@pytest.mark.parametrize(("head", "tail"), [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("order", [("n", "l"), ("l", "n")])
@pytest.mark.parametrize(("n", "l"), [(0, 2), (1, 1), (2, 3)])
def test_a_chain_end_at_a_boundary_slot_is_the_open_grid(
    head: bool, tail: bool, order: tuple[str, str], n: int, l: int
) -> None:
    diagram = _at(_open_grid(head, tail), order, n, l)
    result = compare(diagram, parse_dsl(_grid(n, l, head=head, tail=tail)), {"d": 2})
    assert result.matched, result.reason
    if n >= 2:
        shifted = parse_dsl(_grid(n, l, 1, head=head, tail=tail))
        assert not compare(diagram, shifted, {"d": 2}).matched


@pytest.mark.parametrize(("head", "tail"), [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_chain_with_a_boundary_end_peels(head: bool, tail: bool, reverse: bool) -> None:
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics.induction import rename_multiplicity

    text = _open_grid(head, tail)
    if reverse:
        text = text.replace("chain R: v.out2 -> hv.in0", "chain R: hv.in0 -> v.out2")
    diagram = rename_multiplicity(parse_dsl(text), "n", Mult("s") + 1)
    row = next(box for box in diagram.bang_boxes.values() if box.parent is None and box.links)
    peeled = peel_one(diagram, row.id).diagram
    for k, l in [(0, 1), (2, 3)]:
        at = {"s": k, "l": l, "d": 2}
        got = instantiate_symbol(instantiate_symbol(peeled, "s", k), "l", l)
        expected = instantiate_symbol(instantiate_symbol(diagram, "s", k), "l", l)
        assert compare(got, expected, at).matched


BINDERS = """count n, l
b = B(literal 2, literal 2^m)
u = X(0 -> 1, literal 2)
w = Z(1 -> 0, literal 2)
wire u.out0 -- w.in0
repeat n as R index m: b, u, w
repeat l as C in R: b
chain R: b.out0 -> b.in1
inputs b.in1, b.in0
outputs b.out0"""
"""l binder chains B(2, 2^m) side by side, their dimension 1 heads and 2^n tails on the
boundary."""


def _binders(n: int, l: int, shift: int = 0) -> str:
    lines = [f"b{r}_{c} = B(literal 2, literal {2**r})" for c in range(l) for r in range(n)]
    wires, inputs, outputs = [], [], []
    for r in range(n):
        lines += [f"u{r} = X(0 -> 1, literal 2)", f"w{r} = Z(1 -> 0, literal 2)"]
        wires.append(f"u{r}.out0 -- w{r}.in0")
    for c in range(l):
        if n == 0:
            lines.append(f"i{c} = Z(1 -> 1, literal 1)")
        inputs.append(f"b0_{c}.in1" if n else f"i{c}.in0")
        outputs.append(f"b{n - 1}_{c}.out0" if n else f"i{c}.out0")
    for r in range(n):
        for c in range(l):
            inputs.append(f"b{r}_{c}.in0")
            if r + 1 < n:
                wires.append(f"b{r}_{c}.out0 -- b{r + 1}_{(c + shift) % l}.in1")
    if wires:
        lines.append("wire " + ", ".join(wires))
    lines += ["inputs " + ", ".join(inputs), "outputs " + ", ".join(outputs)]
    return "\n".join(lines)


@pytest.mark.parametrize("order", [("n", "l"), ("l", "n")])
@pytest.mark.parametrize(("n", "l"), [(0, 1), (1, 2), (2, 1), (2, 2), (3, 2)])
def test_boundary_chain_ends_reading_the_index_take_its_end_values(
    order: tuple[str, str], n: int, l: int
) -> None:
    diagram = _at(BINDERS, order, n, l)
    result = compare(diagram, parse_dsl(_binders(n, l)), {"n": n})
    assert result.matched, result.reason
    if n >= 2 and l >= 2:
        assert not compare(diagram, parse_dsl(_binders(n, l, 1)), {"n": n}).matched


def test_boundary_chain_ends_reading_the_index_peel() -> None:
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics.induction import rename_multiplicity

    diagram = rename_multiplicity(parse_dsl(BINDERS), "n", Mult("s") + 1)
    row = next(box for box in diagram.bang_boxes.values() if box.parent is None and box.links)
    peeled = peel_one(diagram, row.id).diagram
    for k, l in [(0, 1), (1, 2), (2, 2)]:
        got = instantiate_symbol(instantiate_symbol(peeled, "s", k), "l", l)
        assert compare(got, parse_dsl(_binders(k + 1, l)), {"s": k}).matched


def _grid_pair(s: int) -> tuple[Diagram, Diagram]:
    """Boxed cluster_state_2d's translation (vertical CZ^s) and its specification."""
    from pathlib import Path

    from benchmarks import ir, translate
    from benchmarks.subject import _load, spec_context

    subject = _load(Path(__file__).parents[1] / "benchmarks/subjects/cluster_state_2d.py")
    eq = subject.equations[0]
    c = eq.circuit("n", "d")
    gates = tuple(
        ir.Chain(
            g.count,
            tuple(ir.CZ(h.c, h.t, s) if isinstance(h, ir.CZ) else h for h in g.body),
            g.carries,
        )
        if isinstance(g, ir.Chain)
        else g
        for g in c.gates
    )
    c = ir.Circuit(c.wires, gates, c.registers, c.dims, c.scalar)
    body = translate.translate(c, translate.Context(values={}, literal=frozenset())).body
    spec = eq.specification(spec_context(subject, None, None, False))
    return parse_dsl("dim d\ncount n\n" + body), parse_dsl("dim d\ncount n\n" + spec)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_the_boxed_grid_cluster_state_is_its_graph_state(n: int) -> None:
    from archytaszx.semantics.check import instantiate

    left, right = _grid_pair(1)
    assert compare(
        instantiate(left, {"n": n, "d": 3}), instantiate(right, {"n": n, "d": 3}), {}
    ).matched
    if n >= 2:
        left, right = _grid_pair(-1)
        both = (instantiate(left, {"n": n, "d": 3}), instantiate(right, {"n": n, "d": 3}))
        assert not compare(*both, {}).matched


def test_instantiating_a_count_fills_it_into_node_dimensions() -> None:
    from archytaszx.diagram.bangbox import Mult
    from archytaszx.semantics.induction import rename_multiplicity

    diagram = rename_multiplicity(parse_dsl(BINDERS), "n", Mult("s") + 1)
    row = next(box for box in diagram.bang_boxes.values() if box.parent is None and box.links)
    peeled = peel_one(diagram, row.id).diagram
    for order in [("s", "l"), ("l", "s")]:
        got = peeled
        for name in order:
            got = instantiate_symbol(got, name, 2)
        for node in got.nodes.values():
            assert all(not p.dim.free_symbols for p in (*node.inputs, *node.outputs))
            assert node.phase is None or not node.phase.free_symbols


def _link_identity(text: str, gen: str) -> str:
    """``text`` (GRID or an open grid) with the node ``gen`` between the row link and hv.in0."""
    text = text.replace("hv = F(d)\n", f"hv = F(d)\ni = {gen}\n")
    text = text.replace("repeat n as R: v, hv, hh, hs, he", "repeat n as R: v, hv, hh, hs, he, i")
    text = text.replace("repeat l as C in R: v, hv, hh", "repeat l as C in R: v, hv, hh, i")
    text = text.replace("chain R: v.out2 -> hv.in0", "chain R: v.out2 -> i.in0")
    text = text.replace("wire hv.out0 -- v.in0,", "wire i.out0 -- hv.in0, hv.out0 -- v.in0,")
    text = text.replace("vs.out0 -- hv.in0", "vs.out0 -- i.in0")
    return text.replace("inputs hv.in0", "inputs i.in0")


def _binder_identity(gen: str, tail: bool = False) -> str:
    """BINDERS with the node ``gen`` between the row link and b.in1 (b.out0 with ``tail``)."""
    text = BINDERS.replace(
        "b = B(literal 2, literal 2^m)\n", f"b = B(literal 2, literal 2^m)\ni = {gen}\n"
    )
    text = text.replace("index m: b, u, w", "index m: b, u, w, i")
    text = text.replace("repeat l as C in R: b", "repeat l as C in R: b, i")
    if tail:
        text = text.replace("chain R: b.out0 -> b.in1", "chain R: i.out0 -> b.in1")
        text = text.replace("wire u.out0 -- w.in0", "wire u.out0 -- w.in0, b.out0 -- i.in0")
        return text.replace("outputs b.out0", "outputs i.out0")
    text = text.replace("chain R: b.out0 -> b.in1", "chain R: b.out0 -> i.in0")
    text = text.replace("wire u.out0 -- w.in0", "wire u.out0 -- w.in0, i.out0 -- b.in1")
    return text.replace("inputs b.in1, b.in0", "inputs i.in0, b.in0")


def _link_identity_pair(case: int) -> tuple[str, str, str]:
    """A diagram, it with an identity on its zipped chain link, and with a non-identity there."""
    if case >= 4:
        dim = "2^m" if case == 4 else "2^(m+1)"
        return (
            BINDERS,
            _binder_identity(f"Z(1 -> 1, literal {dim})", case == 5),
            _binder_identity(f"F(literal {dim})", case == 5),
        )
    base = GRID if case == 0 else _open_grid(case != 2, case != 1)
    return base, _link_identity(base, "Z(1 -> 1, d)"), _link_identity(base, "F(d)")


@pytest.mark.parametrize("case", range(6))
def test_an_identity_on_a_zipped_chain_link_is_removed(case: int) -> None:
    from archytaszx.semantics.decide import decide_equal

    base, identity, other = _link_identity_pair(case)
    decision = decide_equal(parse_dsl(base), parse_dsl(identity), samples=[])
    assert decision.verdict.value == "equal", decision.reason
    assert decision.proof is not None
    assert decide_equal(parse_dsl(base), parse_dsl(other), samples=[]).verdict.value != "equal"
    swapped = identity.replace("hv = F(d)", "hv = Z(1 -> 1, d)")
    if swapped != identity:
        assert decide_equal(parse_dsl(base), parse_dsl(swapped)).verdict.value != "equal"


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(lean_project(), ".lake", "packages", "mathlib")),
    reason="the ArchytasZX Lean project has no Mathlib",
)
@pytest.mark.parametrize("case", [4, 5])
def test_lean_checks_the_link_identity_removal_export(case: int, tmp_path: object) -> None:
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import decide_equal

    base, identity, _ = _link_identity_pair(case)
    left, right = parse_dsl(base), parse_dsl(identity)
    theorem = kernel.from_decision(decide_equal(left, right, samples=[]), left, right)
    for semantic in (False, True):
        source = kernel.lean(theorem, "main", semantic=semantic)
        ok, output = check(source, f"{tmp_path}/link{case}{semantic}.lean")
        assert ok, output
        assert ": error" not in output and "sorry" not in output, output


SCALED = """dim d
count n, l
v = Z(1 -> 1, d)
u = Z(1 -> 1, d)
repeat n as R index i: v, u
repeat l as C in R: v
copy scalar C: 2^n
scalar 3^n"""
"""Copies of a box whose copy scalar reads the parent count, under a diagram scalar reading it."""


def test_instantiating_a_count_fills_it_into_the_scalar_and_copy_scalars() -> None:
    from archytaszx.semantics.decide import _instantiate_index
    from archytaszx.semantics.prove import induction_base

    d = parse_dsl(SCALED)
    got = instantiate_symbol(d, "n", 2)
    assert str(got.scalar) == "9"
    assert [str(b.copy_scalar) for b in got.bang_boxes.values()] == ["4", "4"]
    assert str(instantiate_symbol(got, "l", 2).scalar) == "2304"
    got = _instantiate_index(d, "n", 3)
    assert str(got.scalar) == "27"
    assert [str(b.copy_scalar) for b in got.bang_boxes.values()] == ["8", "8", "8"]
    assert str(induction_base(d, "n").scalar) == "1"


def test_a_child_copy_scalar_is_taken_per_parent_copy_in_either_order() -> None:
    from archytaszx.semantics.check import instantiate

    d = parse_dsl(SCALED)
    for order in [("n", "l"), ("l", "n")]:
        got = d
        for name in order:
            got = instantiate_symbol(got, name, 2)
        assert str(got.scalar) == "2304"
    assert str(instantiate(d, {"n": 2, "l": 2, "d": 2}).scalar) == "2304"
    assert str(instantiate(d, {"n": 2, "l": 3, "d": 2}).scalar) == str(9 * 2 ** (2 * 2 * 3))


NESTED_PAIRS = (
    """graph: 2 nodes, 2 wires, 0 inputs -> 1 output
  scalar: 1
  parameters: none
  nodes:
    n0  Z  in: []  out: [d, d, d]  boxes: !0 > !1
    n1  F  in: [d]  out: [d]  boxes: !0 > !2
  wires:
    n0.out1 -- n1.in0
    n0.out2 -- n1.out0
  inputs: []
  outputs: [n0.out0]
  bang boxes:
    !0  x l  nodes: n0, n1
      !1  x n + 1  nodes: n0  index: j  copy scalar: 3^j
      !2  x 1  nodes: n1  copy scalar: 2^(j + m)  pairs of: !1 index: m  left: n0.out1"""
    "  right: n0.out2"
)
"""A pair box with a copy scalar over an indexed box, both under a live box."""


def test_a_nested_pair_copy_scalar_is_taken_per_parent_copy_in_either_order() -> None:
    from archytaszx.diagram.compare import isomorphic
    from archytaszx.diagram.graph import BangBoxId
    from archytaszx.diagram.validate import validate
    from archytaszx.repl.parser import parse_listing
    from archytaszx.semantics.check import instantiate

    d = parse_listing(NESTED_PAIRS)
    expected = str(2 ** (2 * 6) * 3 ** (2 * 3))
    results = []
    for order in [("n", "l"), ("l", "n")]:
        got = d
        for name in order:
            got = instantiate_symbol(got, name, 2)
        assert str(got.scalar) == expected
        assert not validate(got).errors
        assert len(got.nodes) == 12
        results.append(got)
    assert isomorphic(results[0], results[1])
    assert str(instantiate(d, {"n": 2, "l": 2, "d": 2}).scalar) == expected
    peeled = peel_one(d, BangBoxId(1)).diagram
    for order in [("n", "l"), ("l", "n")]:
        got = peeled
        for name in order:
            got = instantiate_symbol(got, name, 2)
        assert str(got.scalar) == expected
        assert isomorphic(got, results[0])
