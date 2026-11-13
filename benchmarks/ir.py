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

"""Circuit IR: qudit gate lists over named wires and registers.

A wire is a name. A register is a name bound to a count (an int or a symbol name such as
``"n"``); a gate naming a register acts on every copy, the same gate on each. Registers that
share a count are laid out together, copy-major. A wire with no ``Init`` is a boundary input; a
wire with no ``Post`` or ``Discard`` at the end is a boundary output. Amounts (``a`` of X, Z)
and phase rules are integers, fractions or sympy expressions in ``d``, ``n`` and the index
``k``; phases are in turns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

Amount = int | str
Count = int | str


@dataclass(frozen=True)
class Init:
    """The basis state |v>."""

    q: str
    v: int = 0


@dataclass(frozen=True)
class Post:
    """The basis effect <v|."""

    q: str
    v: int = 0


@dataclass(frozen=True)
class Discard:
    """Discard (trace out) the wire."""

    q: str


@dataclass(frozen=True)
class Measure:
    """Computational-basis measurement: the wire is dephased and stays classical."""

    q: str


@dataclass(frozen=True)
class X:
    """The shift |k> -> |k + a>."""

    q: str
    a: Amount = 1


@dataclass(frozen=True)
class Z:
    """The clock |k> -> w^(a k) |k>."""

    q: str
    a: Amount = 1


@dataclass(frozen=True)
class F:
    """The Fourier gate F^p, F[j][k] = w^(j k) / sqrt(d)."""

    q: str
    p: int = 1


@dataclass(frozen=True)
class ZPhase:
    """The diagonal gate |k> -> e^(2 pi i phase_k) |k>: ``entries`` {k: turns} or ``rule`` in k."""

    q: str
    entries: Mapping[int, Any] = field(default_factory=dict)
    rule: str | None = None


@dataclass(frozen=True)
class XPhase:
    """The Fourier conjugate F . ZPhase . F^dagger of a diagonal gate."""

    q: str
    entries: Mapping[int, Any] = field(default_factory=dict)
    rule: str | None = None


@dataclass(frozen=True)
class CX:
    """Controlled add: |c, t> -> |c, t + s c>, s = +1 or -1."""

    c: str
    t: str
    s: int = 1


@dataclass(frozen=True)
class CZ:
    """Controlled phase: |c, t> -> w^(s c t) |c, t>, s = +1 or -1."""

    c: str
    t: str
    s: int = 1


@dataclass(frozen=True)
class Swap:
    """Exchange two wires."""

    a: str
    b: str


@dataclass(frozen=True)
class PhaseFn:
    """The diagonal gate |x> -> e^(2 pi i f(x)) |x> on several wires, f in turns; ``support``
    lists the only inputs where f may be nonzero (None: any input)."""

    qs: tuple[str, ...]
    f: Callable[[tuple[int, ...]], Any]
    support: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class MCX:
    """Multi-controlled add: |c, t> -> |c, t + a [c = values]>."""

    cs: tuple[str, ...]
    t: str
    values: tuple[int, ...] | None = None
    a: int = 1


@dataclass(frozen=True)
class MCZ:
    """Multi-controlled clock: |c, t> -> w^(a t [c = values]) |c, t>."""

    cs: tuple[str, ...]
    t: str
    values: tuple[int, ...] | None = None
    a: int = 1


@dataclass(frozen=True)
class Perm:
    """The permutation |x> -> |p(x)> of the joint basis of several wires (first wire most
    significant)."""

    qs: tuple[str, ...]
    p: Callable[[int], int]


@dataclass(frozen=True)
class Oracle:
    """The standard oracle |x, y> -> |x, y + f(x)> (x on ``xs``, y one wire)."""

    xs: tuple[str, ...]
    y: str
    f: Callable[[tuple[int, ...]], int]


@dataclass(frozen=True)
class Unitary:
    """A dense unitary on several wires (first wire most significant), given as nested lists
    of [real, imag] pairs; used by reference circuits only."""

    qs: tuple[str, ...]
    matrix: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class Prepare:
    """A dense state on several initially |0> wires (first wire most significant), as
    [real, imag] pairs; used by reference circuits only."""

    qs: tuple[str, ...]
    vector: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Chain:
    """A loop over ``count`` copies, copy ``m`` reading ``index`` as ``m`` in amounts and phase
    rules: ``body`` acts on copy ``m`` of each register of that count and on the single wires
    of ``carries``; after each copy every ``(carry, register)`` pair exchanges values, so the
    register's copy keeps the carry's value and the carry goes on with the register's. With
    ``reverse`` the copies run from ``count - 1`` down to 0."""

    count: Count
    body: tuple[Gate, ...]
    carries: tuple[tuple[str, str], ...] = ()
    index: str | None = None
    reverse: bool = False


@dataclass(frozen=True)
class Tree:
    """A tree of depth ``count``: copy c(L, P) (L = 1 .. count, P < b^(L-1)) runs ``body`` on
    its parent value as wire ``parent`` and on fresh local wires; the final values of the
    ``children`` wires (b of them) feed c(L + 1, b P + i), and the leaves' ``children`` take
    ``parent``'s place in the boundary, leaf-major. ``level`` and ``position`` read L and P."""

    count: Count
    body: tuple[Gate, ...]
    parent: str
    children: tuple[str, ...]
    level: str | None = None
    position: str | None = None


@dataclass(frozen=True)
class Pairs:
    """For every pair of copies ``j < k`` of register ``over``, ``body`` on the wires ``"left"``
    (copy j) and ``"right"`` (copy k), ``index`` reading j and ``pair_index`` reading k, at
    this point of both copies' wires. The translator takes one CZ body."""

    over: str
    body: tuple[Gate, ...]
    index: str = "j"
    pair_index: str = "m"


Gate = Union[  # noqa: UP007
    Init,
    Post,
    Discard,
    Measure,
    X,
    Z,
    F,
    ZPhase,
    XPhase,
    CX,
    CZ,
    Swap,
    PhaseFn,
    MCX,
    MCZ,
    Perm,
    Oracle,
    Unitary,
    Prepare,
    Chain,
    Pairs,
    Tree,
]


@dataclass(frozen=True)
class Circuit:
    """Wires in input boundary order, registers with their counts, per-wire dimensions, gates, a
    global scalar (an expression, 1 by default) and the output boundary order (``wires`` when
    None)."""

    wires: tuple[str, ...]
    gates: tuple[Gate, ...]
    registers: Mapping[str, Count] = field(default_factory=dict)
    dims: Mapping[str, Any] = field(default_factory=dict)
    scalar: str = "1"
    outputs: tuple[str, ...] | None = None

    def output_order(self) -> tuple[str, ...]:
        """The wires in output boundary order: ``outputs``, each wire it omits placed after the
        wire before it in ``wires`` (first when none)."""
        if self.outputs is None:
            return self.wires
        order = [q for q in self.outputs if q in self.wires]
        for i, q in enumerate(self.wires):
            if q not in order:
                order.insert(order.index(self.wires[i - 1]) + 1 if i else 0, q)
        return tuple(order)

    def dim(self, q: str) -> Any:
        """The dimension of wire ``q`` (``"d"`` by default)."""
        return self.dims.get(q, "d")

    def count(self, q: str) -> Count | None:
        """The register count of ``q``, or None for a single wire."""
        return self.registers.get(q)


def gate_wires(g: Gate) -> tuple[str, ...]:
    """Every wire ``g`` acts on, in argument order."""
    if isinstance(g, (CX, CZ)):
        return (g.c, g.t)
    if isinstance(g, Swap):
        return (g.a, g.b)
    if isinstance(g, (PhaseFn, Perm, Unitary, Prepare)):
        return tuple(g.qs)
    if isinstance(g, (MCX, MCZ)):
        return (*g.cs, g.t)
    if isinstance(g, Oracle):
        return (*g.xs, g.y)
    if isinstance(g, Chain):
        seen = [q for h in g.body for q in gate_wires(h)]
        seen += [q for pair in g.carries for q in pair]
        return tuple(dict.fromkeys(seen))
    if isinstance(g, Pairs):
        return (g.over,)
    if isinstance(g, Tree):
        return (g.parent,)
    return (g.q,)


def at_index(g: Gate, values: Mapping[str, int]) -> Gate:
    """``g`` with each index symbol of ``values`` replaced by its value in amounts, phase
    entries and phase rules."""
    from dataclasses import replace as _replace

    import sympy as sp

    subs = {sp.Symbol(k): v for k, v in values.items()}

    def at(v: Any) -> Any:
        if isinstance(v, (int, bool)) or v is None:
            return v
        e = sp.sympify(v)
        return e.subs(subs) if e.free_symbols & set(subs) else v

    if isinstance(g, (X, Z)):
        return _replace(g, a=at(g.a))
    if isinstance(g, (ZPhase, XPhase)):
        rule = None if g.rule is None else str(at(g.rule))
        return _replace(g, entries={i: at(v) for i, v in g.entries.items()}, rule=rule)
    if isinstance(g, Chain):
        return _replace(g, body=tuple(at_index(h, values) for h in g.body))
    return g


def circuit(
    wires: Sequence[str],
    gates: Sequence[Gate],
    *,
    registers: Mapping[str, Count] | None = None,
    dims: Mapping[str, Any] | None = None,
    scalar: str = "1",
    outputs: Sequence[str] | None = None,
) -> Circuit:
    """A :class:`Circuit`, checking that every gate names declared wires and that ``outputs``
    orders the wires."""
    known = set(wires)
    for g in gates:
        bad = [q for q in gate_wires(g) if q not in known]
        if bad:
            raise ValueError(f"{type(g).__name__} names undeclared wire(s) {bad}")
    if outputs is not None and sorted(outputs) != sorted(wires):
        raise ValueError(f"outputs {list(outputs)} is not an order of the wires {list(wires)}")
    outs = tuple(outputs) if outputs is not None else None
    return Circuit(
        tuple(wires), tuple(gates), dict(registers or {}), dict(dims or {}), scalar, outs
    )


def unrolled(name: str, n: int) -> list[str]:
    """The wire names name0 .. name{n-1}."""
    return [f"{name}{i}" for i in range(n)]
