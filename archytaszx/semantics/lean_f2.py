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

"""Mirror of ``ArchytasZX.SymF2`` and ``ArchytasZX.SymF2Bridge`` on Python integers.

A form over F_2 is a mask: bit 0 the constant, bit ``v + 1`` the variable ``v``. A term is
``(q, k, dels, phs)``: ``q √2^k Π_d [d = 0] Π_{(u, w)} (-1)^(u w)``; a state ``(S, t)`` holds
the mask of the summed labels and the term, ``None`` when zero. Each function computes what
its Lean namesake computes, in the same order, so that the literal states agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

CHAIN = 40
"""Steps per ``qchain`` block."""

QTm = tuple[int, int, list[int], list[tuple[int, int]]]
St = tuple[int, QTm | None]
QNode = tuple[str, Sequence[int], Sequence[int], int, int]


def vb(v: int) -> int:
    return 1 << (v + 1)


def tb(m: int, i: int) -> bool:
    return (m >> i) & 1 == 1


def sub(x: int, v: int, m: int) -> int:
    return m ^ vb(x) ^ v if tb(m, x + 1) else m


def clr(x: int, m: int) -> int:
    return m ^ vb(x) if tb(m, x + 1) else m


def tr_q(kind: str, os: Sequence[int], is_: Sequence[int]) -> QTm | None:
    """``trQ``."""
    legs = [*os, *is_]
    if kind == "Z":
        if not legs:
            return (1, 2, [], [])
        return (1, 0, [vb(lab) ^ vb(legs[0]) for lab in legs[1:]], [])
    if kind == "X":
        m = 0
        for lab in reversed(legs):
            m ^= vb(lab)
        return (1, 2 - len(legs), [m], [])
    if kind == "F" and len(os) == 1 and len(is_) == 1:
        return (1, -1, [], [(vb(os[0]), vb(is_[0]))])
    return None


def sum_x(x: int, t: QTm) -> QTm | None:
    """``sumX``: through the first delta reading ``x``, else the character sum."""
    q, k, dels, phs = t
    for i, d in enumerate(dels):
        if tb(d, x + 1):
            v = d ^ vb(x)
            kept = []
            for e in (*dels[:i], *dels[i + 1 :]):
                e2 = sub(x, v, e)
                if e2 == 1:
                    return None
                if e2 != 0:
                    kept.append(e2)
            return (q, k, kept[::-1], [(sub(x, v, a), sub(x, v, b)) for a, b in phs])
    big_a = 0
    nps = []
    for u, w in reversed(phs):
        a, b = tb(u, x + 1), tb(w, x + 1)
        u2, w2 = clr(x, u), clr(x, w)
        big_a ^= (1 if a and b else 0) ^ (w2 if a else 0) ^ (u2 if b else 0)
    for u, w in phs:
        u2, w2 = clr(x, u), clr(x, w)
        if u2 != 0 and w2 != 0:
            nps.append((u2, w2))
    if big_a == 1:
        return None
    return (q, k + 2, dels if big_a == 0 else [big_a, *dels], nps)


def hints(nodes: Sequence[QNode], internal: Sequence[int]) -> list[list[int]]:
    """Each node's labels to sum after it: the summed labels it reads last."""
    ints = set(internal)
    last: dict[int, int] = {}
    for i, nd in enumerate(nodes):
        for lab in (*nd[1], *nd[2]):
            if lab in ints:
                last[lab] = i
    hs: list[list[int]] = [[] for _ in nodes]
    for lab in internal:
        if lab in last:
            hs[last[lab]].append(lab)
    return hs


def sum_ls(mask: int, labs: Sequence[int], s: St) -> St | None:
    """``sumLs``."""
    big_s, t = s
    for lab in labs:
        if not tb(mask, lab) or tb(big_s, lab):
            return None
        big_s |= 1 << lab
        if t is not None:
            t = sum_x(lab, t)
    return (big_s, t)


def step_q(mask: int, node: QNode, h: Sequence[int], s: St) -> St | None:
    """``stepQ``."""
    big_s, t = s
    if any(tb(big_s, lab) for lab in (*node[1], *node[2])):
        return None
    a = tr_q(node[0], node[1], node[2])
    if a is None:
        return None
    if t is not None:
        t = (a[0] * t[0], a[1] + t[1], a[2] + t[2], a[3] + t[3])
    return sum_ls(mask, h, (big_s, t))


def mask_of(labs: Sequence[int]) -> int:
    m = 0
    for lab in labs:
        m |= 1 << lab
    return m


def run_q(nodes: Sequence[QNode], internal: Sequence[int], hs: Sequence[Sequence[int]]) -> Any:
    """The final term after ``runQ`` and ``finQ``; ``False`` when a check fails."""
    mask = mask_of(internal)
    s: St | None = (0, (1, 0, [], []))
    for node, h in zip(nodes, hs, strict=True):
        assert s is not None
        s = step_q(mask, node, h, s)
        if s is None:
            return False
    assert s is not None
    rest = [lab for lab in internal if not tb(s[0], lab)]
    fin = sum_ls(mask, rest, s)
    return False if fin is None else fin[1]


def piv(r: int) -> int:
    return r.bit_length() - 1


def red_r(rows: Sequence[int], m: int) -> int:
    for r in rows:
        if tb(m, piv(r)):
            m ^= r
    return m


def canon_q(t: QTm) -> QTm | None:
    """``canonQ``: reduced echelon deltas, sorted; phases reduced, cleaned and sorted."""
    q, k, dels, phs = t
    rows: list[int] = []
    for a in reversed(dels):
        a2 = red_r(rows, a)
        if a2 == 0:
            continue
        if a2 == 1:
            return None
        p = piv(a2)
        rows = [a2, *(r ^ a2 if tb(r, p) else r for r in rows)]
    neg = 0
    kept: list[tuple[int, int]] = []
    for u, w in reversed(phs):
        u, w = red_r(rows, u), red_r(rows, w)
        u, w = (u, w) if u <= w else (w, u)
        if u == 0 or w == 0:
            continue
        if u == 1 and w == 1:
            neg += 1
            continue
        kept.insert(0, (u, w))
    return (q if neg % 2 == 0 else -q, k, sorted(rows, reverse=True), sorted(kept))


def scal_eq(a0: int, a1: int, e: int) -> bool:
    """``scalEq``."""
    if e % 2 != 0:
        return False
    if e >= 0:
        return a0 << (e // 2) == a1
    return a0 == a1 << (-e // 2)


def cmp_q(s0: tuple[int, int, int], s1: tuple[int, int, int], t0: Any, t1: Any) -> bool:
    """``cmpQ`` on the scalars ``(a, b, K)`` and the final terms."""
    (a0, b0, k0), (a1, b1, k1) = s0, s1
    if b0 == 0 or b1 == 0:
        return False
    u0 = None if t0 is None else canon_q(t0)
    u1 = None if t1 is None else canon_q(t1)
    if u0 is None or u1 is None:
        return u0 is None and u1 is None
    return (
        u0[2] == u1[2]
        and u0[3] == u1[3]
        and scal_eq(a0 * u0[0] * b1, a1 * u1[0] * b0, (k0 + u0[1]) - (k1 + u1[1]))
    )


def scalar_of(model: Any) -> tuple[int, int, int] | None:
    """``(a, b, K)`` with ``model = a / b √2^K``, else None."""
    import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

    from archytaszx.semantics.lean_scalar import Inv, Rt

    q, k = Fraction(1), 0
    for f in sp.Mul.make_args(sp.sympify(model)):
        base, e = f.as_base_exp()
        if not e.is_Integer:
            return None
        e = int(e)
        if base.is_Rational:
            q *= Fraction(int(base.p), int(base.q)) ** e
        elif isinstance(base, Rt) and base.args[0] == 2:
            k += e
        elif isinstance(base, Inv) and isinstance(base.args[0], Rt) and base.args[0].args[0] == 2:
            k -= e
        else:
            return None
    return q.numerator, q.denominator, k


def lean_hints(hs: Sequence[Sequence[int]]) -> str:
    return "[" + ", ".join("[" + ", ".join(map(str, h)) + "]" for h in hs) + "]"


# the bridge


def pkey(width: int, node: int, out: bool, pos: int) -> int:
    """``pkey``."""
    return (2 * node + (0 if out else 1)) * width + pos


def pack_w(width: int, bits: int, wires: Sequence[tuple[Any, Any]]) -> int:
    """``packW`` on wires of ports ``(node, out, pos)``."""
    acc = 0
    for a, b in reversed(wires):
        acc = pkey(width, *a) + (1 << bits) * pkey(width, *b) + (1 << (2 * bits)) * acc
    return acc
