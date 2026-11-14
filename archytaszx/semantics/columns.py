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

"""A column count generalised to a free multiplicity, and induction over the rows.

:func:`generalise` takes a pair over one free multiplicity ``n`` in which each side holds one
rows box (a box holding a box of multiplicity ``n``, its own multiplicity ``n + c``) and
counts every other box of multiplicity ``n`` by a fresh ``m``. :func:`induct_rows` proves such
a pair for every ``n >= c`` and ``m``: ``n`` renamed ``s + c``, each side peeled
(:func:`~archytaszx.semantics.kernel.peel`) until its rows box has multiplicity ``s``, then
:func:`~archytaszx.semantics.kernel.induction` over ``s`` with the base decided with no oracle
sample, one side's peeled successor exposed as is or by an
:func:`~archytaszx.semantics.kernel.identity_slide`, its hypothesis replaced by the other
side's, and the parts off the rows boxes' chain cores met by a decision (box groups dropped,
then put back by :func:`~archytaszx.semantics.kernel.regroup`) placed by
:func:`~archytaszx.semantics.kernel.congr`. :func:`specialised` carries the theorem back at
``m = n`` by :func:`~archytaszx.semantics.kernel.instance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult, free_mult_symbols
from archytaszx.diagram.graph import BangBoxId, Diagram, NodeId

if TYPE_CHECKING:
    from archytaszx.semantics.induction import CutRegion
    from archytaszx.semantics.kernel import Theorem


class ColumnsError(Exception):
    """The column count or the row induction does not apply or does not check."""


@dataclass(frozen=True, slots=True)
class Generalised:
    """``left`` and ``right`` with their column boxes counted by the free multiplicity
    ``symbol``; ``index`` counts the rows."""

    left: Diagram
    right: Diagram
    index: str
    symbol: str


def _rows_box(diagram: Diagram, index: str) -> BangBoxId | None:
    """The one box holding a box of multiplicity ``index``, when its multiplicity is ``index``
    plus a constant; else None."""
    n = Mult(index).to_sympy()
    holders = {
        box.parent
        for box in diagram.bang_boxes.values()
        if box.parent is not None and box.multiplicity == Mult(index)
    }
    if len(holders) != 1:
        return None
    (rows,) = holders
    assert rows is not None
    shift = sp.expand(diagram.bang_boxes[rows].multiplicity.to_sympy() - n)
    return rows if shift.is_Integer else None


def _fresh(left: Diagram, right: Diagram, stem: str) -> str:
    """``stem``, ``stem1``, ... named nowhere in ``left`` or ``right``."""
    from archytaszx.semantics.induction import _all_free_symbols

    taken = _all_free_symbols(left) | _all_free_symbols(right)
    taken |= {b.index for d in (left, right) for b in d.bang_boxes.values() if b.index}
    return next(name for i in range(len(taken) + 1) if (name := f"{stem}{i or ''}") not in taken)


def generalise(left: Diagram, right: Diagram) -> Generalised | None:
    """The pair with every box of multiplicity ``n`` other than each side's rows box
    (:func:`_rows_box`) counted by a fresh ``m``; None unless ``n`` is the pair's one free
    multiplicity and each side has a rows box and a box so counted."""
    names = free_mult_symbols(left) | free_mult_symbols(right)
    if len(names) != 1:
        return None
    (index,) = names
    symbol = _fresh(left, right, "m")
    out: list[Diagram] = []
    for diagram in (left, right):
        rows = _rows_box(diagram, index)
        if rows is None:
            return None
        counted = diagram.copy()
        columns = [
            b
            for b, box in diagram.bang_boxes.items()
            if b != rows and box.multiplicity == Mult(index)
        ]
        if not columns:
            return None
        for box_id in columns:
            counted.set_bang_box_multiplicity(box_id, Mult(symbol))
        out.append(counted)
    return Generalised(out[0], out[1], index, symbol)


def specialised(theorem: Theorem, found: Generalised, left: Diagram, right: Diagram) -> Theorem:
    """``left = right`` from ``theorem`` (``found.left = found.right``) at ``m = n``."""
    from archytaszx.semantics import kernel

    t = kernel.instance(theorem, found.symbol, Mult(found.index))
    return kernel.trans(kernel.iso(left, t.lhs), kernel.trans(t, kernel.iso(t.rhs, right)))


def _flat(diagram: Diagram) -> Diagram:
    """``diagram`` with no box groups."""
    out = diagram.copy()
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if box.groups:
            out.set_bang_box_extension(box_id, groups=())
    return out


def _decided(left: Diagram, right: Diagram) -> Theorem:
    """``left = right`` from a decision with no oracle sample."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import EqualityVerdict, decide_equal

    decision = decide_equal(left, right, samples=[])
    if decision.verdict is not EqualityVerdict.EQUAL:
        raise ColumnsError(f"{decision.verdict.value}: {decision.reason}")
    return kernel.from_decision(decision, left, right)


def _rests(a: Diagram, b: Diagram) -> Theorem:
    """``a = b`` by :func:`_decided` over both with box groups dropped, then
    :func:`~archytaszx.semantics.kernel.regroup`."""
    from archytaszx.semantics import kernel

    flat_a, flat_b = _flat(a), _flat(b)
    if kernel._same(flat_a, a) and kernel._same(flat_b, b):
        return _decided(a, b)
    return kernel.regroup(_decided(flat_a, flat_b), a, b)


def _core(diagram: Diagram, index: str) -> frozenset[NodeId]:
    """The chain core (:func:`~archytaszx.semantics.induction.box_core`) of the one top-level
    box carrying ``index``."""
    from archytaszx.diagram.bangbox import index_owners
    from archytaszx.semantics.induction import box_core

    owners = [b for b in index_owners(diagram, index) if diagram.bang_boxes[b].parent is None]
    if len(owners) != 1 or not diagram.bang_boxes[owners[0]].links:
        raise ColumnsError(f"no single top-level chain box carries {index}")
    core = box_core(diagram, owners[0])
    if core is None:
        raise ColumnsError("the chain box's head and tail meet")
    return core


def close_rows(replaced: Diagram, target: Diagram, index: str) -> Theorem:
    """``replaced = target`` by an isomorphism, else by :func:`_rests` over the parts off both
    chain cores (:func:`_core`), scalars one and ``target``'s over ``replaced``'s, placed in
    ``replaced`` by :func:`~archytaszx.semantics.kernel.congr`."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import _open_region, find_cut_region

    try:
        return kernel.iso(replaced, target)
    except kernel.KernelError:
        pass
    rest_a = frozenset(replaced.nodes) - _core(replaced, index)
    rest_b = frozenset(target.nodes) - _core(target, index)
    a, b = _open_region(replaced, rest_a), _open_region(target, rest_b)
    a.set_scalar(Scalar.one())
    b.set_scalar(target.scalar * replaced.scalar**-1)
    rests = _rests(a, b)
    placed = find_cut_region(replaced, rest_a, rests.lhs)
    if placed is None:
        raise ColumnsError("the part off the chain core is not its open diagram")
    moved = kernel.congr(rests, replaced, placed)
    return kernel.trans(moved, kernel.iso(moved.rhs, target))


def _exposures(
    peeled: Diagram, copy: frozenset[NodeId], hypothesis: Diagram
) -> list[tuple[Theorem, CutRegion]]:
    """``peeled = exposed`` with the cut reading ``hypothesis`` in ``exposed`` off ``copy``:
    ``peeled`` itself, then each identity slide."""
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import find_cut_region, identity_slides

    out: list[tuple[Theorem, CutRegion]] = []
    cut = find_cut_region(peeled, frozenset(peeled.nodes) - copy, hypothesis)
    if cut is not None:
        out.append((kernel.refl(peeled), cut))
    for slid in identity_slides(peeled, copy):
        cut = find_cut_region(slid, frozenset(slid.nodes) - copy, hypothesis)
        if cut is not None:
            out.append((kernel.identity_slide(peeled, slid, copy), cut))
    return out


def _peeled_to(diagram: Diagram, index: str) -> Theorem:
    """``diagram = d'`` by :func:`~archytaszx.semantics.kernel.peel` over ``index`` until no box
    carrying it has a positive constant term."""
    from archytaszx.diagram.bangbox import index_owners
    from archytaszx.semantics import kernel

    out = kernel.refl(diagram)
    for _ in range(4):
        d = out.rhs
        owners = index_owners(d, index)
        constants = [
            sp.expand(d.bang_boxes[b].multiplicity.to_sympy()).as_coeff_Add()[0] for b in owners
        ]
        if not any(c.is_Integer and c >= 1 for c in constants):
            return out
        out = kernel.trans(out, kernel.peel(d, index))
    raise ColumnsError(f"the boxes carrying {index} do not peel down")


def induct_rows(left: Diagram, right: Diagram, index: str) -> Theorem:
    """``left = right`` for every value of ``index`` (see the module docstring), ``index``
    read in each side's rows box with no negative constant."""
    from archytaszx.diagram.bangbox import BangBoxError
    from archytaszx.diagram.validate import ValidateError
    from archytaszx.semantics import kernel
    from archytaszx.semantics.induction import (
        InductionError,
        _fresh_step_symbol,
        replace_cut_region,
    )
    from archytaszx.semantics.prove import (
        InductionSide,
        induction_base,
        induction_hypothesis,
        induction_successor,
    )

    errors = (kernel.KernelError, ColumnsError, InductionError, BangBoxError, ValidateError)
    try:
        to_a, to_b = _peeled_to(left, index), _peeled_to(right, index)
    except errors as exc:
        raise ColumnsError(f"peel: {exc}") from exc
    a, b = to_a.rhs, to_b.rhs
    k = _fresh_step_symbol(a, b, index, "k")
    reasons: list[str] = []
    base: Theorem | None = None
    for side, own, other in ((InductionSide.START, a, b), (InductionSide.GOAL, b, a)):
        peeled = induction_successor(own, index, k)
        other_peeled = induction_successor(other, index, k)
        if peeled is None or other_peeled is None:
            reasons.append(f"{side.value}: no single box to peel")
            continue
        own_k, other_k = (induction_hypothesis(d, index, k) for d in (own, other))
        copy = frozenset(peeled.nodes) - frozenset(own_k.nodes)
        try:
            found = _exposures(peeled, copy, own_k)
        except errors as exc:
            reasons.append(f"{side.value}: {exc}")
            continue
        if not found:
            reasons.append(f"{side.value}: the hypothesis is not exposed")
        for expose, cut in found:
            try:
                replaced = replace_cut_region(expose.rhs, cut, own_k, other_k)
                step = close_rows(replaced, other_peeled, k)
                if base is None:
                    base = _decided(induction_base(a, index), induction_base(b, index))
                proved = kernel.induction(
                    a, b, index, k, base, side, expose, cut.region, step, cut=cut
                )
                return kernel.trans(to_a, kernel.trans(proved, kernel.symm(to_b)))
            except errors as exc:
                reasons.append(f"{side.value}: {exc}")
    raise ColumnsError("; ".join(reasons))


__all__ = [
    "ColumnsError",
    "Generalised",
    "close_rows",
    "generalise",
    "induct_rows",
    "specialised",
]
