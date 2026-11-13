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

import pytest

from archytaszx.diagram.bangbox import BangBoxGrammarError, instantiate_symbol, peel_one
from archytaszx.diagram.graph import Diagram
from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics.check import compare

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


def _grid(n: int, l: int, shift: int = 0) -> str:
    lines = ["dim d"]
    wires = []
    for r in range(n):
        lines += [f"hs{r} = X(0 -> 1, d)", f"he{r} = Z(1 -> 0, d)"]
        for c in range(l):
            lines += [f"v{r}_{c} = Z(2 -> 3, d)", f"hv{r}_{c} = F(d)", f"hh{r}_{c} = F(d)"]
            wires += [f"hv{r}_{c}.out0 -- v{r}_{c}.in0", f"hh{r}_{c}.out0 -- v{r}_{c}.in1"]
            wires.append(f"{f'hs{r}.out0' if c == 0 else f'v{r}_{c - 1}.out1'} -- hh{r}_{c}.in0")
            up = f"vs{c}.out0" if r == 0 else f"v{r - 1}_{(c + shift) % l}.out2"
            wires.append(f"{up} -- hv{r}_{c}.in0")
        wires.append(f"v{r}_{l - 1}.out1 -- he{r}.in0")
    for c in range(l):
        lines += [f"vs{c} = X(0 -> 1, d)", f"ve{c} = Z(1 -> 0, d)"]
        wires.append(f"v{n - 1}_{c}.out2 -- ve{c}.in0")
    lines.append("wire " + ", ".join(wires))
    lines.append("outputs " + ", ".join(f"v{r}_{c}.out0" for r in range(n) for c in range(l)))
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


def test_a_chain_end_at_a_boundary_slot_is_refused() -> None:
    text = GRID.replace("repeat l as PH: vs\n", "").replace("vs = X(0 -> 1, d)\n", "")
    text = text.replace("vs.out0 -- hv.in0, ", "")
    text = text.replace("outputs v.out0", "inputs hv.in0\noutputs v.out0")
    with pytest.raises(Exception, match="not wired to a box of multiplicity"):
        parse_dsl(text)


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
