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
]


@dataclass(frozen=True)
class Circuit:
    """Wires in boundary order, registers with their counts, per-wire dimensions, gates and a
    global scalar (an expression, 1 by default)."""

    wires: tuple[str, ...]
    gates: tuple[Gate, ...]
    registers: Mapping[str, Count] = field(default_factory=dict)
    dims: Mapping[str, Any] = field(default_factory=dict)
    scalar: str = "1"

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
    return (g.q,)


def circuit(
    wires: Sequence[str],
    gates: Sequence[Gate],
    *,
    registers: Mapping[str, Count] | None = None,
    dims: Mapping[str, Any] | None = None,
    scalar: str = "1",
) -> Circuit:
    """A :class:`Circuit`, checking that every gate names declared wires."""
    known = set(wires)
    for g in gates:
        bad = [q for q in gate_wires(g) if q not in known]
        if bad:
            raise ValueError(f"{type(g).__name__} names undeclared wire(s) {bad}")
    return Circuit(tuple(wires), tuple(gates), dict(registers or {}), dict(dims or {}), scalar)


def unrolled(name: str, n: int) -> list[str]:
    """The wire names name0 .. name{n-1}."""
    return [f"{name}{i}" for i in range(n)]
