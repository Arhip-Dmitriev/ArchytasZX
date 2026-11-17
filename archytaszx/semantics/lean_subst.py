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

"""Residue contraction by substitution and geometric sums, a mirror of `RefuteSym.lean`.

A plan sums out the labels a binder, splitter, Z spider or constraint fixes (delta steps) and
the labels shared by exactly two Fourier boxes (geometric steps), then contracts the remaining
labels with residue tables. With the refuter's ``sparse_all`` the plan first replaces each X
spider with zero phases by its divisibility constraint (X steps) and its bindings read the
boundary labels from the assignment. :func:`sym_plan` builds the plan at one boundary assignment and
:func:`sym_decls` writes its Lean declarations.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from archytaszx.semantics.lean_net import LNet, NetUnsupported, _head

Ex = tuple[Any, ...]

MAX_SYM_STEPS = 4000
MAX_XSTEP = 4096
RUN_STEPS = 64


def ev(e: Ex, t: Callable[[int], int]) -> int:
    """Mirror of `LX.eval`."""
    h = e[0]
    if h == "v":
        return t(e[1])
    if h == "c":
        return int(e[1])
    if h == "add":
        return ev(e[1], t) + ev(e[2], t)
    if h == "mul":
        return int(e[1]) * ev(e[2], t)
    if h == "div":
        return ev(e[1], t) // e[2] if e[2] else 0
    if h == "mod":
        return ev(e[1], t) % e[2] if e[2] else ev(e[1], t)
    n = e[2]
    return (n - ev(e[1], t) % n) % n if n else 0


def vars_(e: Ex) -> list[int]:
    """Mirror of `LX.vars`."""
    h = e[0]
    if h == "v":
        return [e[1]]
    if h == "c":
        return []
    if h == "add":
        return vars_(e[1]) + vars_(e[2])
    if h == "mul":
        return vars_(e[2])
    return vars_(e[1])


def subst(e: Ex, l: int, r: Ex) -> Ex:
    """Mirror of `LX.subst`."""
    h = e[0]
    if h == "v":
        return r if e[1] == l else e
    if h == "c":
        return e
    if h == "add":
        return ("add", subst(e[1], l, r), subst(e[2], l, r))
    if h == "mul":
        return ("mul", e[1], subst(e[2], l, r))
    return (h, subst(e[1], l, r), e[2])


def ub(e: Ex, d: Callable[[int], int]) -> int:
    """Mirror of `LX.ub`."""
    h = e[0]
    if h == "v":
        return max(d(e[1]) - 1, 0)
    if h == "c":
        return int(e[1])
    if h == "add":
        return ub(e[1], d) + ub(e[2], d)
    if h == "mul":
        return int(e[1]) * ub(e[2], d)
    if h == "div":
        return ub(e[1], d) // e[2] if e[2] else 0
    if h == "mod":
        return ub(e[1], d) if e[2] == 0 else int(e[2]) - 1
    return max(int(e[2]) - 1, 0)


def _sadd(a: Ex, b: Ex) -> Ex:
    if a[0] == "c" and b[0] == "c":
        return ("c", a[1] + b[1])
    if a == ("c", 0):
        return b
    if b == ("c", 0):
        return a
    return ("add", a, b)


def _smod0(a: Ex, k: int) -> Ex:
    if a[0] == "c":
        return ("c", a[1] % k if k else a[1])
    return ("mod", a, k)


def _smod(a: Ex, k: int) -> Ex:
    if a[0] == "c":
        return ("c", a[1] % k if k else a[1])
    if a[0] == "mod":
        return a if a[2] == k else ("mod", a, k)
    if a[0] == "add" and a[1][0] == "mul":
        return _smod0(a[2], k) if a[1][1] == k else ("mod", a, k)
    if a[0] == "add" and a[2][0] == "mul":
        return _smod0(a[1], k) if a[2][1] == k else ("mod", a, k)
    return ("mod", a, k)


def _sdiv0(a: Ex, k: int) -> Ex:
    if a[0] == "c":
        return ("c", a[1] // k if k else 0)
    if a[0] == "mod":
        return ("c", 0) if a[2] == k and k > 0 else ("div", a, k)
    return ("div", a, k)


def _sdiv(a: Ex, k: int) -> Ex:
    if a[0] in ("c", "mod"):
        return _sdiv0(a, k)
    if a[0] == "add" and a[1][0] == "mul":
        m, x, r = a[1][1], a[1][2], a[2]
        return _sadd(x, _sdiv0(r, k)) if m == k and k > 0 else ("div", a, k)
    if a[0] == "add" and a[2][0] == "mul":
        m, x, r = a[2][1], a[2][2], a[1]
        return _sadd(_sdiv0(r, k), x) if m == k and k > 0 else ("div", a, k)
    return ("div", a, k)


def lnorm(e: Ex, u: Sequence[int] | set[int], sigma: Mapping[int, int]) -> Ex:
    """Mirror of `LX.norm`."""
    h = e[0]
    if h == "v":
        return ("c", sigma.get(e[1], 0)) if e[1] in u else e
    if h == "c":
        return e
    if h == "add":
        return _sadd(lnorm(e[1], u, sigma), lnorm(e[2], u, sigma))
    if h == "mul":
        a = lnorm(e[2], u, sigma)
        return ("c", e[1] * a[1]) if a[0] == "c" else ("mul", e[1], a)
    a = lnorm(e[1], u, sigma)
    if h == "div":
        return _sdiv(a, e[2])
    if h == "mod":
        return _smod(a, e[2])
    n = e[2]
    return ("c", (n - a[1] % n) % n if n else 0) if a[0] == "c" else ("neg", a, n)


def text(e: Ex) -> str:
    """An expression as a Lean `LX` term."""
    h = e[0]
    if h == "v":
        return f"(LX.v {e[1]})"
    if h == "c":
        return f"(LX.c {e[1]})"
    if h == "add":
        return f"(LX.add {text(e[1])} {text(e[2])})"
    if h == "mul":
        return f"(LX.mul {e[1]} {text(e[2])})"
    return f"(LX.{h} {text(e[1])} {e[2]})"


@dataclass
class State:
    """Mirror of `SSt`: bindings newest first, node indices, constraints and a constant."""

    bs: list[tuple[int, Ex]] = field(default_factory=list)
    ns: list[int] = field(default_factory=list)
    cs: list[tuple[int, Ex]] = field(default_factory=list)
    c: int = 1
    ix: dict[int, Ex] | None = field(default=None, repr=False, compare=False)

    def index(self) -> dict[int, Ex]:
        if self.ix is None:
            self.ix = {}
            for a, e in self.bs:
                self.ix.setdefault(a, e)
        return self.ix

    def texpr(self, x: int) -> Ex:
        return self.index().get(x, ("v", x))

    def bound(self, x: int) -> bool:
        return x in self.index()


@dataclass
class Plan:
    """The states and steps of a substitution contraction and its final residue tables."""

    states: list[State]
    frees: list[list[int]]
    steps: list[tuple[Any, ...]]
    leaves: list[Any]
    groups: list[list[int]]
    scopes: list[tuple[int, ...]]
    cscopes: list[tuple[int, ...]]
    final: Any
    value: int


def _legs(net: LNet, i: int) -> list[int]:
    n = net.nodes[i]
    return [*n.outs, *n.ins]


def _spair(st: State, net: LNet, dimv: Callable[[int], int], src: tuple[Any, ...]) -> Any:
    """Mirror of `spair`."""
    if src[0] == "zz":
        _, i, x, y = src
        if _head(net.nodes[i].kind) == "Z" and x in _legs(net, i) and y in _legs(net, i):
            return st.texpr(x), st.texpr(y)
        return None
    _, i, pos = src
    n = net.nodes[i]
    h = _head(n.kind)
    t = dimv(n.ph)
    if h == "B" and len(n.outs) == 1 and len(n.ins) == 2:
        (k,), (a, b) = n.outs, n.ins
    elif h == "S" and len(n.outs) == 2 and len(n.ins) == 1:
        (a, b), (k,) = n.outs, n.ins
        pos = {2: 0, 0: 1, 1: 2}[pos] if pos in (0, 1, 2) else 3
    else:
        return None
    ek, ea, eb = st.texpr(k), st.texpr(a), st.texpr(b)
    if pos == 0:
        return ek, ("add", ("mul", t, ea), eb)
    if pos == 1:
        return ea, ("div", ek, t)
    if pos == 2:
        return eb, ("mod", ek, t)
    return None


def _solvep(pair: Any, l: int, d: Callable[[int], int]) -> Ex | None:
    """Mirror of `solveP`."""
    if pair is None:
        return None
    a, t = pair
    if a == ("v", l):
        return tuple(t)
    if a[0] == "neg" and a[1] == ("v", l) and a[2] == d(l):
        return ("neg", t, a[2])
    return None


def _csolve(n: int, a: Ex, b: Ex, l: int, d: Callable[[int], int]) -> Ex | None:
    """Mirror of `csolve`."""
    if a == ("v", l) and n == d(l):
        return ("neg", b, n)
    if a[0] == "neg" and a[1] == ("v", l) and a[2] == n and n == d(l):
        return ("mod", b, n)
    return None


def _solve(
    st: State, net: LNet, dimv: Any, d: Any, l: int, src: tuple[Any, ...], bare: bool = False
) -> Ex | None:
    """Mirror of `solve`; ``bare`` admits a constraint on one label."""
    if src[0] == "cn":
        n, e = src[1]
        if e[0] == "v" and bare:
            return _csolve(n, e, ("c", 0), l, d)
        if e[0] != "add":
            return None
        return _csolve(n, e[1], e[2], l, d) or _csolve(n, e[2], e[1], l, d)
    return _solvep(_spair(st, net, dimv, src), l, d)


def _fleg(st: State, net: LNet, i: int, l: int) -> Ex | None:
    """Mirror of `fleg`."""
    n = net.nodes[i]
    if _head(n.kind) != "F" or len(n.outs) != 1 or len(n.ins) != 1:
        return None
    eo, ei = st.texpr(n.outs[0]), st.texpr(n.ins[0])
    if eo == ("v", l) and l not in vars_(ei):
        return ei
    if ei == ("v", l) and l not in vars_(eo):
        return eo
    return None


def sym_plan(refuter: Any, net: LNet, sigma: Mapping[int, int]) -> Plan:
    """The substitution contraction of ``net`` at the boundary assignment ``sigma``."""
    d = refuter.labdim
    dimv = refuter.dimv
    p, m = refuter.p, refuter.M
    bnd = list(net.bnd)
    st = State(ns=list(range(len(net.nodes))))
    free = list(dict.fromkeys(net.summed))
    fset, bset = set(free), set(bnd)
    states: list[State] = [st]
    frees: list[list[int]] = [list(free)]
    steps: list[tuple[Any, ...]] = []
    pinned = bool(getattr(refuter, "sparse_all", False))
    pu: set[int] = bset if pinned else set()
    psig: Mapping[int, int] = sigma if pinned else {}
    if pinned:
        for i in list(st.ns):
            n = net.nodes[i]
            dn = dimv(n.dim)
            if _head(n.kind) != "X" or not 0 < dn <= MAX_XSTEP or m % dn:
                continue
            if any(refuter.P(n.ph, k)[0] for k in range(dn)):
                continue
            rest = [j for j in st.ns if j != i]
            e = _xexpr(st, n, dn)
            st = State(list(st.bs), rest, [*st.cs, (dn, e)], st.c * dn % p)
            steps.append(("X", i, rest))
            states.append(st)
            frees.append(list(free))

    def opened(s: State, a: Ex) -> bool:
        if a[0] == "neg" and a[1][0] == "v":
            a = a[1]
        return a[0] == "v" and a[1] in fset and not s.bound(a[1])

    def ncands(s: State, i: int) -> Any:
        n = net.nodes[i]
        h = _head(n.kind)
        if h == "Z":
            L = list(dict.fromkeys(_legs(net, i)))
            for x in L:
                if not opened(s, s.texpr(x)):
                    continue
                for y in L:
                    if x != y:
                        yield ("zz", i, x, y)
        elif h in ("B", "S"):
            for pos in (0, 1, 2):
                yield ("bs", i, pos)

    def dok(s: State, l: int, src: tuple[Any, ...]) -> Ex | None:
        if l not in fset or s.bound(l):
            return None
        e = _solve(s, net, dimv, d, l, src, pinned)
        if e is None or l in vars_(e):
            return None
        if any(x not in fset and x not in bset for x in vars_(e)):
            return None
        if ub(e, d) >= d(l):
            return None
        return e

    def leg_label(s: State, src: tuple[Any, ...]) -> int | None:
        if src[0] == "cn":
            e = src[1][1]
            if e[0] == "v":
                return int(e[1]) if pinned and e[1] in fset else None
            if e[0] != "add":
                return None
            for a in (e[1], e[2]):
                if a[0] == "v" and a[1] in fset:
                    return int(a[1])
                if a[0] == "neg" and a[1][0] == "v" and a[1][1] in fset:
                    return int(a[1][1])
            return None
        pr = _spair(s, net, dimv, src)
        if pr is None:
            return None
        a = pr[0]
        if a[0] == "v":
            return int(a[1])
        if a[0] == "neg" and a[1][0] == "v":
            return int(a[1][1])
        return None

    def first(s: State, closed: bool, srcs: Any) -> tuple[Any, ...] | None:
        for src in srcs:
            l = leg_label(s, src)
            if l is None:
                continue
            e = dok(s, l, src)
            if e is None:
                continue
            if closed and any(x in fset for x in vars_(e)):
                continue
            return ("D", l, src, e)
        return None

    memo: dict[Any, tuple[Any, ...] | None] = {}

    def delta(s: State, closed: bool, only_cn: bool = False) -> tuple[Any, ...] | None:
        for i in [] if only_cn else s.ns:
            key: Any = (closed, i, tuple(s.texpr(x) for x in _legs(net, i)))
            if key not in memo:
                memo[key] = first(s, closed, ncands(s, i))
            if memo[key] is not None:
                return memo[key]
        for q in s.cs:
            key = (closed, "cn", q)
            if key not in memo:
                memo[key] = first(s, closed, [("cn", q)])
            if memo[key] is not None:
                return memo[key]
        return None

    def geom(s: State) -> tuple[Any, ...] | None:
        mens: dict[int, list[int]] = {}
        for i in s.ns:
            for v in dict.fromkeys(v for x in _legs(net, i) for v in vars_(s.texpr(x))):
                mens.setdefault(v, []).append(i)
        cvars = {v for q in s.cs for v in vars_(q[1])}
        for l in free:
            if s.bound(l):
                continue
            ms = mens.get(l, [])
            if len(ms) != 2 or l in cvars:
                continue
            n = d(l)
            if n <= 0 or m % n:
                continue
            if any(dimv(net.nodes[i].dim) != n for i in ms):
                continue
            x, z = _fleg(s, net, ms[0], l), _fleg(s, net, ms[1], l)
            if x is None or z is None:
                continue
            return ("G", l, ms[0], ms[1], [i for i in s.ns if i not in ms], x, z)
        return None

    while free and len(steps) < MAX_SYM_STEPS:
        step = delta(st, True) or delta(st, False, True) or geom(st) or delta(st, False)
        if step is None:
            break
        l = step[1]
        if step[0] == "D":
            e = step[3]
            st = State(
                [
                    (l, lnorm(e, pu, psig)),
                    *((a, lnorm(subst(v, l, e), pu, psig)) for a, v in st.bs),
                ],
                list(st.ns),
                [(nq, lnorm(subst(v, l, e), pu, psig)) for nq, v in st.cs],
                st.c,
            )
        else:
            _, _, _, _, rest, x, z = step
            st = State(list(st.bs), list(rest), [*st.cs, (d(l), ("add", x, z))], st.c * d(l) % p)
        free.remove(l)
        fset.discard(l)
        steps.append(step)
        states.append(st)
        frees.append(list(free))
    return _finish(refuter, net, sigma, st, states, frees, steps, free)


def _xexpr(st: State, n: Any, d: int) -> Ex:
    """Mirror of `xexpr`."""
    es = [st.texpr(x) for x in n.outs]
    es += [st.texpr(x) if d == 2 else ("mul", d - 1, st.texpr(x)) for x in n.ins]
    out: Ex = ("c", 0)
    for e in reversed(es):
        out = ("add", e, out)
    return out


def _finish(
    refuter: Any,
    net: LNet,
    sigma: Mapping[int, int],
    st: State,
    states: list[State],
    frees: list[list[int]],
    steps: list[tuple[Any, ...]],
    free: list[int],
) -> Plan:
    """The residue tables of the last state over the labels left, a mirror of `leafS`,
    `leafQ` and `kleaf`, and their elimination."""
    d = refuter.labdim
    p = refuter.p
    u = set(net.bnd)

    def scope_of(es: Sequence[Ex]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(x for e in es for x in vars_(lnorm(e, u, sigma))))

    def table(scope: Sequence[int], entry: Callable[[Callable[[int], int]], int]) -> list[int]:
        dims = [d(x) for x in scope]
        size = math.prod(dims)
        if size > refuter.MAX_TABLE:
            raise NetUnsupported("a residue table is too large after substitution")
        out = []
        for idx in range(size):
            tau, rest = {}, idx
            for x, dx in zip(scope, dims, strict=True):
                tau[x], rest = rest % dx, rest // dx

            def at(y: int, tau: dict[int, int] = tau) -> int:
                return sigma.get(y, 0) if y in u else tau.get(y, 0)

            out.append(entry(at))
        return out

    by_scope: dict[frozenset[int], tuple[tuple[int, ...], list[int]]] = {}
    for i in st.ns:
        sc = scope_of([st.texpr(x) for x in _legs(net, i)])
        by_scope[frozenset([i])] = (sc, [i])
    groups = [g for _, g in by_scope.values()]
    scopes = [sc for sc, _ in by_scope.values()]
    cscopes: list[tuple[int, ...]] = []
    leaves = []
    for sc, g in by_scope.values():

        def group_entry(at: Callable[[int], int], g: list[int] = g) -> int:
            acc = 1
            for i in reversed(g):
                n = net.nodes[i]
                vals = {x: ev(st.texpr(x), at) for x in _legs(net, i)}
                acc = int(refuter.res(refuter.node(n, vals))) * acc % p
            return acc

        leaves.append(_reduce(sc, table(sc, group_entry), d))
    for nq, e in st.cs:

        def cons_entry(at: Callable[[int], int], nq: int = nq, e: Ex = e) -> int:
            return 1 if ev(e, at) % nq == 0 else 0

        sc = scope_of([e])
        cscopes.append(sc)
        leaves.append(_reduce(sc, table(sc, cons_entry), d))
    leaves.append(((), [st.c % p]))
    plan = refuter.res_eliminate(leaves, free)
    value = refuter.res_final(plan[2], sigma)
    return Plan(states, frees, steps, plan[0], groups, scopes, cscopes, plan, value)


def _reduce(
    scope: Sequence[int], tab: list[int], d: Callable[[int], int]
) -> tuple[tuple[int, ...], list[int]]:
    """The table restricted to the labels of `scope` it varies in, first label least
    significant."""
    scope, dims = list(scope), [d(x) for x in scope]
    k = 0
    while k < len(scope):
        inner = math.prod(dims[:k])
        dk = dims[k]
        block = inner * dk
        if all(tab[i] == tab[i % inner + (i // block) * block] for i in range(len(tab))):
            tab = [tab[i] for i in range(len(tab)) if (i // inner) % dk == 0]
            del scope[k], dims[k]
        else:
            k += 1
    return tuple(scope), tab


def _segments(steps: Sequence[tuple[Any, ...]], pinned: bool) -> list[tuple[int, int]]:
    """The steps as ranges ``(a, b)``: with ``pinned`` the runs of at most `RUN_STEPS` delta and
    X steps between geometric steps, else one step each."""
    out: list[tuple[int, int]] = []
    k = 0
    while k < len(steps):
        b = k + 1
        if pinned and steps[k][0] != "G":
            while b < len(steps) and steps[b][0] != "G" and b - k < RUN_STEPS:
                b += 1
        out.append((k, b))
        k = b
    return out


def _sstep_text(tag: str, step: tuple[Any, ...]) -> str:
    """A delta or X step as a Lean `SStep`."""
    if step[0] == "X":
        return f"SStep.x {tag}k{step[1]}"
    _, l, src, e = step
    return f"SStep.d {l} {_src_text(tag, src)} {text(e)}"


def _src_text(tag: str, src: tuple[Any, ...]) -> str:
    """A delta step's source as a Lean `Src`."""
    if src[0] == "zz":
        return f"(Src.zz {tag}k{src[1]} {src[2]} {src[3]})"
    if src[0] == "bs":
        return f"(Src.bs {tag}k{src[1]} {src[2]})"
    return f"(Src.cn ({src[1][0]}, {text(src[1][1])}))"


def sym_decls(
    refuter: Any, tag: str, net: LNet, plan: Plan, env: str, unbnd: str
) -> tuple[list[str], str]:
    """The declarations of ``plan`` and the term proving the network's formal entry mod `p`
    equal to its final residue."""
    M, p, w = refuter.M, refuter.p, refuter.w
    pinned = bool(getattr(refuter, "sparse_all", False))
    shape = f"{env}.shape"
    dim = f"{shape}.dim"
    sv = f"sval {shape} {M} Pt Ct {p} {w}"
    out: list[str] = []
    for i, n in enumerate(net.nodes):
        out.append(f"def {tag}k{i} : Node := {n.lean()}")

    def ns_text(ns: Sequence[int]) -> str:
        return "[" + ", ".join(f"{tag}k{i}" for i in ns) + "]" if ns else "([] : List Node)"

    def bs_text(bs: Sequence[tuple[int, Ex]]) -> str:
        if not bs:
            return "([] : List (ℕ × LX))"
        return "[" + ", ".join(f"({a}, {text(e)})" for a, e in bs) + "]"

    def cs_text(cs: Sequence[tuple[int, Ex]]) -> str:
        if not cs:
            return "([] : List (ℕ × LX))"
        return "[" + ", ".join(f"({a}, {text(e)})" for a, e in cs) + "]"

    def nat_list(xs: Sequence[int]) -> str:
        return "[" + ", ".join(str(x) for x in xs) + "]" if xs else "([] : List ℕ)"

    segs = _segments(plan.steps, pinned)
    keep = {0, len(plan.steps)} | {a for a, _ in segs} | {b for _, b in segs}
    for k, (s, fr) in enumerate(zip(plan.states, plan.frees, strict=True)):
        if k not in keep:
            continue
        body = f"⟨{bs_text(s.bs)}, {ns_text(s.ns)}, {cs_text(s.cs)}, {s.c}⟩"
        out.append(f"def {tag}s{k} : SSt := {body}")
        out.append(f"def {tag}R{k} : List ℕ := {nat_list(fr)}")
    out.append(f"theorem {tag}hU : ∀ x ∈ ({unbnd} : List ℕ), σ0 x < {dim} x := by decide +kernel")
    links = [f"(sym_start {shape} {M} Pt Ct {p} {w} {tag} σ0)"]
    for k, b in segs:
        step = plan.steps[k]
        head = (
            f"sumR {dim} {tag}R{k} ({sv} {tag}s{k}) σ0 = sumR {dim} {tag}R{b} ({sv} {tag}s{b}) σ0"
        )
        if pinned and step[0] != "G":
            run = ", ".join(_sstep_text(tag, st) for st in plan.steps[k:b])
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {tag}t{k} [Fact (Nat.Prime {p})] "
                f"(hw : IsPrimitiveRoot (({w} : ℕ) : ZMod {p}) {M}) : {head} :=\n"
                f"  sym_run {shape} {M} Pt Ct {p} {w} hw (by norm_num) {unbnd} {tag}R{k} "
                f"{tag}R{b} {tag}s{k} {tag}s{b} [{run}] σ0 {tag}hU (by decide +kernel) "
                "(by decide +kernel)"
            )
            links.append(f"({tag}t{k} hw)")
        elif step[0] == "D":
            _, l, src, e = step
            if src[0] == "zz":
                stext = f"(Src.zz {tag}k{src[1]} {src[2]} {src[3]})"
            elif src[0] == "bs":
                stext = f"(Src.bs {tag}k{src[1]} {src[2]})"
            else:
                stext = f"(Src.cn ({src[1][0]}, {text(src[1][1])}))"
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {tag}t{k} : {head} :=\n"
                f"  sym_dstep {shape} {M} Pt Ct {p} {w} {unbnd} {tag}R{k} {tag}R{k + 1} "
                f"{tag}s{k} {tag}s{k + 1} {l} {stext} {text(e)} σ0 {tag}hU "
                f"(by decide +kernel) (by decide +kernel) (by decide +kernel)"
            )
            links.append(f"{tag}t{k}")
        else:
            _, l, i1, i2, rest, x, z = step
            out.append(
                f"set_option maxHeartbeats 0 in\ntheorem {tag}t{k} [Fact (Nat.Prime {p})] "
                f"(hw : IsPrimitiveRoot (({w} : ℕ) : ZMod {p}) {M}) : {head} :=\n"
                f"  sym_gstep {shape} {M} Pt Ct {p} {w} hw (by norm_num) {tag}R{k} {tag}R{k + 1} "
                f"{tag}s{k} {tag}s{k + 1} {l} {tag}k{i1} {tag}k{i2} {ns_text(rest)} {text(x)} "
                f"{text(z)} σ0 (by decide +kernel) (by decide +kernel) (by decide +kernel)"
            )
            links.append(f"({tag}t{k} hw)")
    xt = f"{tag}x"
    out += refuter.res_decls(xt, plan.final, dim, packed=True)
    last = len(plan.steps)
    nl = len(plan.groups)
    names = [f"{xt}a{i}" for i in range(len(plan.final[0]))]
    gss = (
        "[" + ", ".join(ns_text(g) for g in plan.groups) + "]" if nl else "([] : List (List Node))"
    )
    scs = (
        "[" + ", ".join(nat_list(sc) for sc in plan.scopes) + "]" if nl else "([] : List (List ℕ))"
    )
    ln = "[" + ", ".join(names[:nl]) + "]" if nl else "([] : List RFac)"
    sqs = (
        "[" + ", ".join(nat_list(sc) for sc in plan.cscopes) + "]"
        if plan.cscopes
        else "([] : List (List ℕ))"
    )
    lq_names = names[nl:-1]
    lq = "[" + ", ".join(lq_names) + "]" if lq_names else "([] : List RFac)"
    lemmas = []
    st_last = f"{tag}s{last}"
    for gi, g in enumerate(plan.groups):
        out.append(
            f"set_option maxHeartbeats 0 in\ntheorem {xt}L{gi} : leafS {shape} {M} Pt Ct {p} {w} "
            f"{unbnd} σ0 {st_last}.bs {ns_text(g)} {nat_list(plan.scopes[gi])} {names[gi]} = true "
            ":= by decide +kernel"
        )
        lemmas.append(f"{xt}L{gi}")
    for qi, (nq, e) in enumerate(plan.states[-1].cs):
        out.append(
            f"set_option maxHeartbeats 0 in\ntheorem {xt}Q{qi} : leafQ {p} {dim} {unbnd} σ0 "
            f"({nq}, {text(e)}) {nat_list(plan.cscopes[qi])} {names[nl + qi]} = true "
            ":= by decide +kernel"
        )
        lemmas.append(f"{xt}Q{qi}")
    zs = (
        "by simp only [List.zip_cons_cons, List.zip_nil_left, List.zip_nil_right, List.all_cons, "
        "List.all_nil, Bool.and_true"
    )
    zs1 = zs + "".join(f", {x}" for x in lemmas[:nl]) + "]"
    zs2 = zs + f", {st_last}" + "".join(f", {x}" for x in lemmas[nl:]) + "]"
    out.append(
        f"set_option maxHeartbeats 0 in\ntheorem {tag}fin "
        f"(hw1 : (({w} : ℕ) : ZMod {p}) ^ {M} = 1) :\n"
        f"    sumR {dim} {tag}R{last} ({sv} {st_last}) σ0 = "
        f"(rfold {p} FN{xt} σ0 : ZMod {p}) :=\n"
        f"  sym_final {shape} {M} Pt Ct {p} {w} (by norm_num) hw1 (by norm_num) (by norm_num) "
        f"{unbnd} {tag}R{last} σ0 {st_last} {gss} {scs} {ln} {lq} (by decide +kernel) "
        f"(by decide +kernel) (by decide +kernel) ({zs1}) {sqs} (by decide +kernel) "
        f"(by decide +kernel) ({zs2}) ST{xt} FN{xt} (by decide +kernel) CH{xt} "
        f"(by decide +kernel) (by decide +kernel) (by decide +kernel) {tag}hU"
    )
    links.append(f"({tag}fin hw1)")
    return out, " |>.trans ".join(links)


__all__ = ["Plan", "State", "sym_decls", "sym_plan"]
