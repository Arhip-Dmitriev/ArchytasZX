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

"""A power box count ``b^e`` generalised to a free value count ``N``.

:func:`generalise` replaces, on both sides of a pair, every box multiplicity ``b^e`` (one
power for the whole pair) and that power in the scalar by a fresh multiplicity ``N``.
:func:`specialised` carries a kernel theorem of the generalised pair back to the pair by
:func:`~archytaszx.semantics.kernel.instance` at ``N = b^e`` and isomorphisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.graph import Diagram

if TYPE_CHECKING:
    from archytaszx.semantics.kernel import Theorem


@dataclass(frozen=True, slots=True)
class Generalised:
    """``left`` and ``right`` with the power ``power`` counted by the free multiplicity
    ``symbol``."""

    left: Diagram
    right: Diagram
    symbol: str
    power: Mult


def _powers(diagram: Diagram) -> set[sp.Expr]:
    """The box multiplicities ``b^e`` of ``diagram`` with ``e`` symbolic."""
    out = set()
    for box in diagram.bang_boxes.values():
        expr = box.multiplicity.to_sympy()
        if expr.is_Pow and expr.base.is_Integer and expr.exp.free_symbols:
            out.add(expr)
    return out


def _fresh(left: Diagram, right: Diagram) -> str:
    """``N``, ``N1``, ... named nowhere in ``left`` or ``right``."""
    from archytaszx.semantics.induction import _all_free_symbols

    taken = _all_free_symbols(left) | _all_free_symbols(right)
    taken |= {b.index for d in (left, right) for b in d.bang_boxes.values() if b.index}
    return next(name for i in range(len(taken) + 1) if (name := f"N{i or ''}") not in taken)


def _counted(diagram: Diagram, power: sp.Expr, symbol: str) -> Diagram:
    """``diagram`` with each box of multiplicity ``power`` and ``power`` in the scalar counted
    by ``symbol``."""
    out = diagram.copy()
    for box_id, box in sorted(out.bang_boxes.items()):
        if box.multiplicity.to_sympy() == power:
            out.set_bang_box_multiplicity(box_id, Mult(symbol))
    scalar = out.scalar.to_sympy()
    if scalar.has(power):
        out.set_scalar(Scalar(scalar.xreplace({power: Mult(symbol).to_sympy()})))
    return out


def generalise(left: Diagram, right: Diagram) -> Generalised | None:
    """The pair with its one power box count counted by a fresh ``N``; None unless the boxes of
    the pair hold one power ``b^e`` and no other."""
    powers = _powers(left) | _powers(right)
    if len(powers) != 1:
        return None
    (power,) = powers
    symbol = _fresh(left, right)
    left_n, right_n = (_counted(d, power, symbol) for d in (left, right))
    return Generalised(left_n, right_n, symbol, Mult._from_expr(power))


def specialised(theorem: Theorem, found: Generalised, left: Diagram, right: Diagram) -> Theorem:
    """``left = right`` from ``theorem`` (``found.left = found.right``) at ``N = b^e``."""
    from archytaszx.semantics import kernel

    t = kernel.instance(theorem, found.symbol, found.power)
    return kernel.trans(kernel.iso(left, t.lhs), kernel.trans(t, kernel.iso(t.rhs, right)))


__all__ = ["Generalised", "generalise", "specialised"]
