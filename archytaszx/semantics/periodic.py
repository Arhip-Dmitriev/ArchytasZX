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

"""Sides that scale by a constant when their index grows by a step: the index read only in the
scalar and the multiplicity of at most one top-level chain box whose copy transfer matrix ``T``
has a scalar power ``T^p = c I``
(:func:`~archytaszx.semantics.contract_symbolic.chain_period`)."""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.graph import Diagram


class PeriodicError(Exception):
    """A side that does not scale by a constant over its index's period."""


def chain_period(diagram: Diagram, index: str) -> tuple[int, sp.Expr] | None:
    """``(p, c)`` of the chain box over ``index`` (``T^p = c I``), or None when no box reads
    ``index``; PeriodicError when ``index`` is read outside the scalar and one such box."""
    from archytaszx.semantics.contract_symbolic import (
        SymbolicContractionDomainError,
        SymbolicContractionUnsupportedError,
    )
    from archytaszx.semantics.contract_symbolic import chain_period as transfer_period

    owners = [b for b, box in diagram.bang_boxes.items() if index in box.multiplicity.free_symbols]
    carried = any(
        index in node.phase.free_symbols
        for node in diagram.nodes.values()
        if node.phase is not None
    ) or any(
        index in port.dim.free_symbols
        for node in diagram.nodes.values()
        for port in (*node.inputs, *node.outputs)
    )
    if carried or len(owners) > 1:
        raise PeriodicError(f"{index} is read outside the scalar and one box")
    if not owners:
        return None
    box = diagram.bang_boxes[owners[0]]
    if box.multiplicity.to_sympy() != Mult.symbol(index).to_sympy() or box.parent is not None:
        raise PeriodicError("the box is not a top-level box of multiplicity the index")
    if box.copy_scalar is not None and index in box.copy_scalar.free_symbols:
        raise PeriodicError("a copy scalar reads the index")
    try:
        return transfer_period(diagram, owners[0])
    except (SymbolicContractionUnsupportedError, SymbolicContractionDomainError) as exc:
        raise PeriodicError(str(exc)) from exc


def shift_ratio(diagram: Diagram, index: str, step: int) -> sp.Expr:
    """The constant ``a`` with ``s(index + step) = a s(index)`` for ``diagram``'s scalar ``s``;
    PeriodicError when there is none."""
    scalar = diagram.scalar.to_sympy()
    symbol = next((s for s in scalar.free_symbols if str(s) == index), None)
    if symbol is None:
        return sp.Integer(1)
    shifted = scalar.xreplace({symbol: symbol + step})
    for point in range(6):
        at = sp.simplify(scalar.xreplace({symbol: point}))
        if at != 0:
            ratio = sp.simplify(shifted.xreplace({symbol: point}) / at)
            break
    else:
        raise PeriodicError("the scalar vanishes at index 0 .. 5")
    gap = sp.expand(shifted - ratio * scalar, power_exp=True, power_base=True)
    if sp.simplify(gap) != 0:
        raise PeriodicError(f"the scalar at {index} + {step} is not a constant times the scalar")
    return ratio


def scaling(diagram: Diagram, index: str, step: int) -> sp.Expr:
    """The constant ``a`` with ``diagram`` at ``index + step`` equal to ``a`` times ``diagram``
    at ``index``: the scalar's ratio times ``c^(step/p)`` for its chain's ``(p, c)``;
    PeriodicError when ``step`` is not a multiple of ``p`` or the side does not scale."""
    found = chain_period(diagram, index)
    ratio = shift_ratio(diagram, index, step)
    if found is None:
        return ratio
    period, c = found
    if step % period:
        raise PeriodicError(f"{step} is not a multiple of the chain's period {period}")
    return sp.simplify(ratio * c ** (step // period))
