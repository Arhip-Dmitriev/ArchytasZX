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

"""Lean 4 refutations over the tensor-network semantics.

:func:`export_refutation` states that two diagrams are not equal for every value of their
parameters and proves it at one point. There both networks evaluate, by ``eval_cyc``, to a
product of ``(√d)⁻¹`` factors times a polynomial in ``ζ = om M`` whose coefficients the Lean
kernel computes. At one boundary assignment the scaled difference, rewritten over the
relations of its atoms and of ``ζ``, is a nonzero integer vector of length at most ``φ(M)``,
which ``cyc_ne`` shows does not vanish.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.phase import INDEX
from archytaszx.diagram.graph import Diagram
from archytaszx.semantics.lean_net import (
    LNet,
    LNode,
    NetUnsupported,
    _Export,
    _head,
    _incs,
    _join,
    _outcs,
    _paren,
)
from archytaszx.semantics.lean_scalar import II, NC, Ex, Inv, Om, Rt, ScalarUnsupported

K = "(by decide +kernel)"
MAX_ASSIGNMENTS = 4096
MAX_POINTS = 16
BRUTE_LIMIT = 256
MAX_TERMS = 200_000

_Z = sp.Symbol("zeta_lean")

Key = tuple[tuple[int, ...], int]
Vec = dict[Key, int]


def export_refutation(
    left: Diagram,
    right: Diagram,
    point: Mapping[str, object],
    name: str = "main",
    describe: Callable[[Diagram], str] | None = None,
) -> str:
    """Lean 4 source proving that ``left`` and ``right`` are not equal for every value of
    their parameters, at ``point`` or else at the first default sample point that works."""
    from archytaszx.semantics.decide import sample_grid

    if _interfaces_differ(left, right) and (left.bang_boxes or right.bang_boxes or any(
            a != b for (a, _), (b, _) in zip(_legs(left), _legs(right), strict=False))):
        try:
            return export_interface_refutation(left, right, point, name)
        except NetUnsupported:
            if not (left.bang_boxes or right.bang_boxes):
                raise
        return _instantiated_refutation(left, right, point, name, describe)
    first: NetUnsupported | None = None
    candidates = [dict(point)]
    with contextlib.suppress(Exception):
        candidates += [dict(p) for p in sample_grid(left, right)][:MAX_POINTS]
    seen: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    for candidate in seen:
        try:
            return _Refuter(left, right, candidate, describe).run(name)
        except ScalarUnsupported as exc:
            first = first or NetUnsupported(str(exc))
        except NetUnsupported as exc:
            first = first or exc
    from archytaszx.semantics.lean_numeric import export_numeric_refutation

    for candidate in seen:
        try:
            return export_numeric_refutation(left, right, candidate, name, describe)
        except NetUnsupported:
            pass
    assert first is not None
    raise first


def _legs(d: Diagram) -> list[tuple[sp.Expr, sp.Expr]]:
    """Each boundary slot's dimension and number of legs, outputs then inputs."""
    from archytaszx.semantics.lean_net import _port_dim

    out = []
    for ref in (*d.boundary_outputs, *d.boundary_inputs):
        count = sp.Integer(1)
        for box in d.bang_boxes.values():
            if ref.node_id in box.node_scope or ref in box.port_scope:
                count *= box.multiplicity.to_sympy()
        out.append((_port_dim(d, ref).to_sympy(), count))
    return out


def export_interface_refutation(
    left: Diagram, right: Diagram, point: Mapping[str, object], name: str = "main",
) -> str:
    """Lean 4 source proving that the leg dimensions of ``left`` and ``right`` differ for some
    parameter values, at ``point`` or a default sample point."""
    from archytaszx.semantics.decide import sample_grid

    ll, lr = _legs(left), _legs(right)
    syms = sorted({s for dim, n in (*ll, *lr) for e in (dim, n) for s in e.free_symbols}, key=str)
    candidates = [dict(point)] + [dict(p) for p in sample_grid(left, right)][:MAX_POINTS]
    for cand in candidates:
        vals = {s: sp.nsimplify(cand[str(s)]) for s in syms if str(s) in cand}
        if len(vals) != len(syms):
            continue

        def flat(legs: list[tuple[sp.Expr, sp.Expr]],
                 vals: dict[sp.Symbol, sp.Expr] = vals) -> list[int] | None:
            out = []
            for dim, n in legs:
                dv, nv = dim.xreplace(vals), n.xreplace(vals)
                if not (dv.is_Integer and nv.is_Integer) or int(dv) <= 0 or int(nv) < 0:
                    return None
                out += [int(dv)] * int(nv)
            return out

        fl, fr = flat(ll), flat(lr)
        if fl is None or fr is None or fl == fr:
            continue
        names = {s: _lean_name(str(s)) for s in syms}

        def text(legs: list[tuple[sp.Expr, sp.Expr]],
                 names: dict[sp.Symbol, str] = names) -> str:
            parts = [f"List.replicate {_nat(n, names)} {_nat(dim, names)}" for dim, n in legs]
            return " ++ ".join(parts) if parts else "([] : List ℕ)"

        dims = sorted({s for dim, _ in (*ll, *lr) for s in dim.free_symbols}, key=str)
        binders = " ".join(f"({names[s]} : ℕ)" for s in syms)
        hyps = " ".join(f"(h_{names[s]} : 0 < {names[s]})" for s in dims)
        args = " ".join(str(int(vals[s])) for s in syms)
        args += " " + " ".join("(by norm_num)" for _ in dims) if dims else ""
        quant = f"∀ {binders} {hyps}, " if syms else ""
        lead = ", ".join(f"{s} = {vals[s]}" for s in syms) or "no parameters"
        return "\n".join([
            "/-",
            "  ArchytasZX interface refutation: the boundary leg dimensions of the two",
            f"  diagrams, outputs then inputs, differ at {lead}.",
            "-/", "", "import ArchytasZX", "", "namespace ArchytasZX.Export", "",
            f"/-- The diagrams' leg dimensions differ at {lead}. -/",
            f"theorem {name} :",
            f"    ¬ {quant}({text(ll)}) = ({text(lr)}) := by",
            f"  intro h; exact absurd (h {args}) (by decide)" if syms else "  decide",
            "", "end ArchytasZX.Export", ""])
    raise NetUnsupported("the leg dimensions agree at every sample point")


def _instantiated_refutation(left: Diagram, right: Diagram, point: Mapping[str, object],
                             name: str, describe: Callable[[Diagram], str] | None) -> str:
    """The refutation of the two families instantiated at a point's multiplicities."""
    from archytaszx.diagram.bangbox import instantiate_symbol
    from archytaszx.semantics.decide import sample_grid

    mults = sorted({str(s) for d in (left, right) for box in d.bang_boxes.values()
                    for s in box.multiplicity.free_symbols})
    first: NetUnsupported | None = None
    for cand in [dict(point)] + [dict(p) for p in sample_grid(left, right)][:MAX_POINTS]:
        if any(m not in cand for m in mults):
            continue
        pair = []
        for d in (left, right):
            for m in mults:
                if any(m in {str(x) for x in b.multiplicity.free_symbols}
                       for b in d.bang_boxes.values()):
                    d = instantiate_symbol(d, m, int(cand[m]))
            pair.append(d)
        if pair[0].bang_boxes or pair[1].bang_boxes:
            continue
        rest = {k: v for k, v in cand.items() if k not in mults}
        try:
            src = export_refutation(pair[0], pair[1], rest, name, describe)
        except NetUnsupported as exc:
            first = first or exc
            continue
        at = ", ".join(f"{m} = {cand[m]}" for m in mults)
        return src.replace("/-\n", f"/-\n  the families instantiated at {at}\n", 1)
    raise first or NetUnsupported("no sample point instantiates both families")


def _lean_name(raw: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    return out if out and not out[0].isdigit() else f"v_{out}"


def _nat(e: sp.Expr, names: Mapping[sp.Symbol, str]) -> str:
    e = sp.sympify(e)
    if e.is_Integer:
        return str(int(e))
    if e.is_Symbol:
        return names[e]
    if e.is_Add:
        return "(" + " + ".join(_nat(a, names) for a in e.args) + ")"
    if e.is_Mul:
        return "(" + " * ".join(_nat(a, names) for a in e.args) + ")"
    if e.is_Pow and e.args[1].is_Integer and int(e.args[1]) >= 0:
        return f"({_nat(e.args[0], names)} ^ {int(e.args[1])})"
    if e.is_Pow:
        return f"({_nat(e.args[0], names)} ^ {_nat(e.args[1], names)})"
    raise NetUnsupported(f"{e} is not a natural number expression")


def _interfaces_differ(left: Diagram, right: Diagram) -> bool:
    ll, lr = _legs(left), _legs(right)
    return len(ll) != len(lr) or any(
        sp.expand(a - b) != 0 or sp.expand(m - n) != 0 for (a, m), (b, n) in zip(ll, lr,
                                                                              strict=False))


class _Refuter:
    """Evaluates both networks at one point and writes the refutation."""

    def __init__(self, left: Diagram, right: Diagram, point: Mapping[str, object],
                 describe: Callable[[Diagram], str] | None, opaque: bool = False) -> None:
        self.out = _Export(describe or (lambda d: f"a diagram of {len(d.nodes)} nodes"))
        self.point = {str(k): sp.nsimplify(v) for k, v in point.items()}
        for d in (left, right):
            for box in d.bang_boxes.values():
                for s in sorted(box.multiplicity.to_sympy().free_symbols, key=str):
                    self.out.mult_index(str(s))
        self.a = self.out.labelled(left)
        n_l = len(left.boundary_outputs) + len(left.boundary_inputs)
        n_r = len(right.boundary_outputs) + len(right.boundary_inputs)
        self.arity = n_l != n_r
        if self.arity and (left.bang_boxes or right.bang_boxes):
            raise NetUnsupported("the boundaries differ in size within a family")
        if self.arity:
            base = self.out.next_label
            self.b = self.out.labelled(right, list(range(base, base + n_r)))
        else:
            self.b = self.out.labelled(right)
        self.out.register(left)
        self.out.register(right)
        self.s0 = self.out.scalar(left.scalar)
        self.s1 = self.out.scalar(right.scalar)
        if self.out.sc.opaques and not opaque:
            raise NetUnsupported("a diagram scalar has no exact Lean form")
        self.out.set_params()

    # values at the point

    def val(self, expr: object) -> sp.Expr:
        e = sp.sympify(expr)
        subs = {}
        for s in e.free_symbols:
            if str(s) not in self.point:
                raise NetUnsupported(f"the counterexample gives no value for {s}")
            subs[s] = self.point[str(s)]
        return sp.nsimplify(e.xreplace(subs))

    def nat(self, expr: object) -> int:
        v = self.val(expr)
        if not v.is_Integer or int(v) < 0:
            raise NetUnsupported(f"{expr} = {v} is not a natural number")
        return int(v)

    def setup_env(self) -> None:
        out = self.out
        self.ks = [self.nat(e) for e in out.mult_exprs] if out.family else []
        base = [self.nat(e) for e in out.dim_exprs]
        self.dimv_tab: dict[int, int] = {}
        for i in range(len(out.dims)):
            if out.family and i in out.bund_of:
                self.dimv_tab[i] = base[out.root(i)] ** math.prod(
                    self.mult(m) for m in out.chain(i))
            else:
                self.dimv_tab[i] = base[i]

    def dimv(self, i: int) -> int:
        return self.dimv_tab.get(i, 1)

    def mult(self, m: int) -> int:
        return self.ks[m] if m < len(self.ks) else 0

    def labdim(self, lab: int) -> int:
        return self.dimv(self.out.labs.get(lab, 0))

    def phase(self, ph: int, k: int) -> sp.Rational:
        specs = self.out.phase_specs
        entries, rule = specs[ph] if ph < len(specs) else ({}, None)
        v = entries.get(k, sp.Integer(0))
        if rule is not None and k != 0:
            v = v + rule.xreplace({INDEX: k})
        out = self.val(v)
        if not out.is_Rational:
            raise NetUnsupported(f"the phase {out} is not a rational number of turns")
        return sp.Rational(out)

    # the exact evaluator, a mirror of `evalT` and `sqL`

    def bundled(self, d: int, vs: Sequence[int], cs: Sequence[tuple[int, ...]]) -> list[int]:
        n = max(len(vs) - len(cs), 0)
        return list(vs[:n]) + [
            y for v, c in zip(vs[n:], cs, strict=False) for y in self.digs_c(d, c, v)]

    def digs_c(self, d: int, c: tuple[int, ...], v: int) -> list[int]:
        if not c:
            return [v]
        b = d ** math.prod(self.mult(x) for x in c[1:])
        return [y for i in range(self.mult(c[0])) for y in self.digs_c(d, c[1:], v // b**i % b)]

    def copy_legs(self, d: int, i: int, cs: Sequence[tuple[int, ...]],
                  vs: Sequence[int]) -> list[int]:
        out = []
        for c, v in zip(cs, vs, strict=False):
            b = d ** math.prod(self.mult(x) for x in c)
            out.append(v // b**i % b)
        return out

    def tz(self, ph: int, d: int, vs: Sequence[int]) -> Vec:
        out: Vec = {}
        for k in range(d):
            if all(v == k for v in vs):
                e, ks = self.P(ph, k)
                _add(out, (ks, e), 1, self.M)
        return out

    def tx(self, ph: int, d: int, os: Sequence[int], is_: Sequence[int]) -> Vec:
        q = self.M // d
        out: Vec = {}
        for k in range(d):
            e, ks = self.P(ph, k)
            _add(out, (ks, e + q * (k * sum(os)) + q * ((d - 1) * (k * sum(is_)))), 1, self.M)
        return out

    @staticmethod
    def twz(d: int, os: Sequence[int], i: int) -> int:
        if i == 0:
            return 1 if all(o == 0 for o in os) else 0
        return 1 if i < d and [o for o in os if o != 0] == [i] else 0

    def tensor(self, kind: Any, d: int, ph: int, os: Sequence[int], is_: Sequence[int]) -> Vec:
        h = _head(kind)
        one = len(os) == 1 and len(is_) == 1
        if h == "Z":
            return self.tz(ph, d, [*os, *is_])
        if h == "X":
            return self.tx(ph, d, os, is_)
        if h == "F":
            return {((), (self.M // d * (os[0] * is_[0])) % self.M): 1} if one else {}
        if h == "T":
            return _const(1 if os[0] == 0 or os[0] == is_[0] else 0) if one else {}
        if h == "Ti":
            if not one:
                return {}
            o, i = os[0], is_[0]
            return _const(1 if o == i else -1 if o == 0 and i >= 1 else 0)
        if h == "W":
            return _const(self.twz(d, os, is_[0])) if len(is_) == 1 else {}
        if h == "I":
            return _const(1 if one and os[0] == is_[0] and os[0] < d else 0) if one else {}
        if h == "ZB":
            return self.tz(ph, d, [*os, *self.bundled(d, is_, kind[1])])
        if h == "XB":
            return self.tx(ph, d, self.bundled(d, os, kind[1]), self.bundled(d, is_, kind[2]))
        if h == "WB":
            if len(is_) != 1:
                return {}
            return _const(self.twz(d, self.bundled(d, os, kind[1]), is_[0]))
        if h == "Bun":
            b = kind[2]
            acc: Vec = {((), 0): 1}
            for i in range(self.mult(kind[1])):
                acc = _mul(acc, self.tensor(b, d, ph, self.copy_legs(d, i, _outcs(b, len(os)), os),
                                            self.copy_legs(d, i, _incs(b, len(is_)), is_)), self.M)
                if not acc:
                    break
            return acc
        return {}

    def node(self, n: LNode, sigma: Mapping[int, int]) -> Vec:
        os = [sigma.get(x, 0) for x in n.outs]
        is_ = [sigma.get(x, 0) for x in n.ins]
        h = _head(n.kind)
        if h in ("B", "S"):
            s, t = self.dimv(n.dim), self.dimv(n.ph)
            if h == "B" and len(os) == 1 and len(is_) == 2:
                k, (a, b) = os[0], is_
            elif h == "S" and len(os) == 2 and len(is_) == 1:
                (a, b), k = os, is_[0]
            else:
                return {}
            return _const(1 if a < s and b < t and k == a * t + b else 0)
        return self.tensor(n.kind, self.dimv(n.dim), n.ph, os, is_)

    def sqc(self, kind: Any, d: int, no: int, ni: int) -> int:
        h = _head(kind)
        if h == "X":
            return no + ni
        if h == "F":
            return 1
        if h == "XB":
            return len(self.bundled(d, [0] * no, kind[1])) + len(self.bundled(d, [0] * ni, kind[2]))
        if h == "Bun":
            return self.mult(kind[1]) * self.sqc(kind[2], d, no, ni)
        return 0

    def sq_list(self, net: LNet) -> list[tuple[int, int]]:
        return [(self.dimv(n.dim), 0 if _head(n.kind) in ("B", "S") else
                 self.sqc(n.kind, self.dimv(n.dim), len(n.outs), len(n.ins))) for n in net.nodes]

    def evaluate(self, net: LNet, sigma: Mapping[int, int]) -> list[int]:
        """The entry without atoms, as `evalG`'s single vector."""
        groups = self.groups(net, sigma)
        if any(k for k in groups):
            raise NetUnsupported("the entry depends on a phase that is not a rational number")
        return groups.get((), [0] * self.M)

    def groups(self, net: LNet, sigma: Mapping[int, int]) -> dict[tuple[int, ...], list[int]]:
        """The entry as groups, by contraction for a large network."""
        if self.use_plan(net):
            return self.final_groups(self.plan_of(net)[2], sigma)
        return self.groups_brute(net, sigma)

    def plan_of(self, net: LNet) -> tuple[list[Any], list[Any], list[Any]]:
        cache = self.__dict__.setdefault("_plans", {})
        if id(net) not in cache:
            cache[id(net)] = self.plan(net)
        return cache[id(net)]

    def eval_rewrites(self, env: str, gs: str, keys: str, nets: Sequence[tuple[str, LNet, Any,
                      Any]], sigma: str = "σ0") -> tuple[str, str, list[str]]:
        """The `rw` terms for each network's entry, the `show` terms for its constant and
        groups, and the declarations of its contraction."""
        M = self.M
        evals, shows, decls = [], [], []
        for name, net, sq, g in nets:
            shows.append(f"show sqL {_paren(env)}.shape {name} = {_list_pairs(sq)} from by "
                         f"decide +kernel")
            if self.use_plan(net):
                decls += self.plan_decls(name, self.plan_of(net))
                evals.append(
                    f"eval_ve {_paren(env)} {M} {gs} Pt {keys} {name} (by norm_num) "
                    f"{K} hP L{name} {K} {K} ST{name} FN{name} {K} {K} {K} {sigma} {K}")
                evals.append(f"eval_ve_groups {_paren(env)} {M} {gs} (by norm_num) FN{name} "
                             f"{sigma}")
                shows.append(f"show gnorm {M} (ffold {_paren(env)}.shape.dim FN{name} {sigma}) = "
                             f"{_groups_text(g)} from by decide +kernel")
            else:
                evals.append(f"eval_cyc {_paren(env)} {M} {gs} Pt {keys} {name} {sigma} "
                             f"(by norm_num) (by decide +kernel) hP")
                shows.append(f"show evalG {_paren(env)}.shape {M} Pt {name} {sigma} = "
                             f"{_groups_text(g)} from by decide +kernel")
        return ", ".join(evals), ", ".join(shows), decls

    def groups_brute(self, net: LNet, sigma: Mapping[int, int]) -> dict[tuple[int, ...],
                                                                         list[int]]:
        """Mirror of `evalG`: one vector of length `M` per atom monomial, zero ones dropped."""
        total: Vec = {}
        ranges = [range(self.labdim(x)) for x in net.summed]
        if math.prod(len(r) for r in ranges) * max(len(net.nodes), 1) > MAX_TERMS:
            raise NetUnsupported("the networks are too large to evaluate at the counterexample")
        for values in itertools.product(*ranges):
            tau = dict(sigma)
            for x, v in zip(net.summed, values, strict=True):
                tau[x] = v
            acc: Vec = {((), 0): 1}
            for n in net.nodes:
                acc = _mul(acc, self.node(n, tau), self.M)
                if not acc:
                    break
            for key, c in acc.items():
                _add(total, key, c, self.M)
        out: dict[tuple[int, ...], list[int]] = {}
        for (ks, e), c in total.items():
            out.setdefault(ks, [0] * self.M)[e] += c
        return {k: v for k, v in sorted(out.items()) if any(v)}

    # contraction, a mirror of `leafOK`, `elimOK` and `chainOK`

    def table(self, scope: Sequence[int], entry: Callable[[dict[int, int]], Vec]) -> list[Vec]:
        dims = [self.labdim(x) for x in scope]
        out = []
        for i in range(math.prod(dims)):
            tau, rest = {}, i
            for x, d in zip(scope, dims, strict=True):
                tau[x], rest = rest % d, rest // d
            out.append(entry(tau))
        return out

    def at(self, fac: tuple[tuple[int, ...], list[Vec]], tau: Mapping[int, int]) -> Vec:
        scope, tab = fac
        i, mul = 0, 1
        for x in scope:
            i += tau.get(x, 0) * mul
            mul *= self.labdim(x)
        return tab[i] if i < len(tab) else {}

    def plan(self, net: LNet) -> tuple[list[Any], list[tuple[int, list[Any]]], list[Any]]:
        """Leaves, elimination steps and final factors of ``net``."""
        leaves = []
        for n in net.nodes:
            scope = tuple(dict.fromkeys((*n.outs, *n.ins)))
            leaves.append((scope, self.table(scope, lambda tau, n=n: self.node(n, tau))))
        facs = list(leaves)
        steps = []
        left = list(dict.fromkeys(net.summed))
        while left:
            def cost(l: int, facs: list[Any] = facs) -> int:
                sc: set[int] = set()
                for f in facs:
                    if l in f[0]:
                        sc |= set(f[0])
                sc.discard(l)
                return math.prod(self.labdim(x) for x in sc)
            l = min(left, key=lambda x: (cost(x), x))
            left.remove(l)
            ins = [f for f in facs if l in f[0]]
            outs = [f for f in facs if l not in f[0]]
            scope = tuple(dict.fromkeys(x for f in ins for x in f[0] if x != l))
            if math.prod(self.labdim(x) for x in scope) * self.labdim(l) > MAX_TERMS:
                raise NetUnsupported("an intermediate factor is too large")

            def entry(tau: dict[int, int], ins=ins, l=l) -> Vec:
                total: Vec = {}
                for j in range(self.labdim(l)):
                    t2 = {**tau, l: j}
                    acc: Vec = {((), 0): 1}
                    for f in ins:
                        acc = _mul(acc, self.at(f, t2), self.M)
                        if not acc:
                            break
                    for key, c in acc.items():
                        _add(total, key, c, self.M)
                return total
            new = (scope, self.table(scope, entry))
            facs = outs + [new]
            steps.append((l, list(facs)))
        return leaves, steps, facs

    def final_groups(self, final: list[Any], sigma: Mapping[int, int]) -> dict[tuple[int, ...],
                                                                                list[int]]:
        acc: Vec = {((), 0): 1}
        for f in final:
            acc = _mul(acc, self.at(f, sigma), self.M)
        out: dict[tuple[int, ...], list[int]] = {}
        for (ks, e), c in acc.items():
            out.setdefault(ks, [0] * self.M)[e] += c
        return {k: v for k, v in sorted(out.items()) if any(v)}

    def use_plan(self, net: LNet) -> bool:
        return math.prod(self.labdim(x) for x in dict.fromkeys(net.summed)) > BRUTE_LIMIT

    def plan_decls(self, tag: str, plan: tuple[list[Any], list[Any], list[Any]]) -> list[str]:
        leaves, steps, final = plan
        steps_t = ", ".join(f"({l}, {_facs(fs)})" for l, fs in steps)
        return [
            f"/-- The leaves of {tag}. -/\ndef L{tag} : List Fac := {_facs(leaves)}",
            (f"/-- The eliminations of {tag}. -/\ndef ST{tag} : List (ℕ × List Fac) := "
             f"[{steps_t}]"),
            f"/-- The final factors of {tag}. -/\ndef FN{tag} : List Fac := {_facs(final)}",
        ]

    # the order of ζ

    def choose_m(self) -> None:
        needs = {1}
        self.keys: list[tuple[int, int]] = []
        dens: list[sp.Rational] = []
        for net in (self.a.net, self.b.net):
            for n in net.nodes:
                if _head(n.kind) in ("B", "S"):
                    continue
                d = self.dimv(n.dim)
                if d <= 0:
                    raise NetUnsupported("a node of dimension 0")
                needs.add(d)
                for j in range(d):
                    if (n.ph, j) not in self.keys:
                        self.keys.append((n.ph, j))
                        dens.append(self.phase(n.ph, j))
        for t in dens:
            needs.add(int(t.q))
        self.atoms_seen: set[sp.Expr] = set()
        for e in (self.at_point(self.s0), self.at_point(self.s1)):
            needs |= self.atom_needs(e)
        for net in (self.a.net, self.b.net):
            for d, c in self.sq_list(net):
                if c:
                    needs |= self.atom_needs(Rt(d))
        self.M = math.lcm(*needs)
        self.ptab = {}
        for (ph, j), t in zip(self.keys, dens, strict=True):
            self.ptab[(ph, j)] = (int((t * self.M) % self.M), ())

    def P(self, ph: int, k: int) -> tuple[int, tuple[int, ...]]:
        return self.ptab.get((ph, k), (0, ()))

    def at_point(self, model: sp.Expr) -> sp.Expr:
        e = sp.sympify(model)
        subs = {s: self.point[str(s)] for s in e.free_symbols
                if s != II and str(s) in self.point}
        missing = [s for s in e.free_symbols if s != II and str(s) not in self.point]
        if missing:
            raise NetUnsupported(f"the counterexample gives no value for {missing[0]}")
        return e.xreplace(subs)

    def atom_needs(self, e: sp.Expr) -> set[int]:
        needs: set[int] = set()
        for a in sp.preorder_traversal(sp.sympify(e)):
            if a == II:
                needs.add(4)
            elif isinstance(a, Rt):
                _, s = _square_split(_int(a.args[0]))
                for q in sp.primefactors(s):
                    needs.add({2: 8, 3: 12}.get(q, q if q % 4 == 1 else 4 * q))
            elif isinstance(a, Om):
                needs.add(_int(a.args[0]))
            elif isinstance(a, Ex):
                raise NetUnsupported(f"the exponential {a} at the counterexample")
        return needs

    # exact values in ℚ(ζ)

    def phi(self) -> sp.Poly:
        return sp.Poly(sp.cyclotomic_poly(self.M, _Z), _Z, domain="QQ")

    def poly(self, e: sp.Expr) -> sp.Poly:
        """``e`` as a polynomial in ζ reduced modulo Φ_M."""
        phi = self.phi()
        return self.poly_expr(e).rem(phi)

    def poly_expr(self, e: sp.Expr) -> sp.Poly:
        e = sp.sympify(e)
        phi = self.phi()
        if e.is_Rational:
            return sp.Poly(e, _Z, domain="QQ")
        if e == _Z:
            return sp.Poly(_Z, _Z, domain="QQ")
        if e.is_Add:
            out = sp.Poly(0, _Z, domain="QQ")
            for x in e.args:
                out = out + self.poly_expr(x)
            return out.rem(phi)
        if e.is_Mul:
            out = sp.Poly(1, _Z, domain="QQ")
            for x in e.args:
                out = (out * self.poly_expr(x)).rem(phi)
            return out
        if e.is_Pow and e.args[1].is_Integer:
            n = int(e.args[1])
            base = self.poly_expr(e.args[0])
            if n < 0:
                base, n = sp.Poly(sp.invert(base.as_expr(), phi.as_expr(), _Z), _Z,
                                  domain="QQ"), -n
            return (base**n).rem(phi)
        if isinstance(e, NC):
            return sp.Poly(_int(e.args[0]), _Z, domain="QQ")
        if isinstance(e, Inv):
            inner = self.poly_expr(e.args[0])
            if inner.is_zero:
                return inner
            return sp.Poly(sp.invert(inner.as_expr(), phi.as_expr(), _Z), _Z, domain="QQ")
        sub = self.substitute(e, emit=False)
        if sub is None:
            raise NetUnsupported(f"no exact value for {e}")
        return self.poly_expr(sub[0])

    def substitute(self, a: sp.Expr, emit: bool = True) -> tuple[sp.Expr, str] | None:
        """``a`` in terms of other atoms or ζ, with a Lean proof of ``a = that``; ``emit``
        declares the facts that proof uses."""
        M = self.M
        if a == II:
            return _Z ** (M // 4), f"I_sub {M} {M // 4} (by norm_num) (by norm_num)"
        if isinstance(a, Om):
            m = _int(a.args[0])
            if m == M:
                return None
            return _Z ** (M // m), f"om_sub {M} {m} {M // m} (by norm_num) (by norm_num)"
        if isinstance(a, Rt):
            x = _int(a.args[0])
            k, s = _square_split(x)
            if x == 1:
                return sp.Integer(1), "rt_one"
            if k != 1:
                return k * Rt(s), f"rt_split {x} {k} {s} (by norm_num)"
            if s == 2:
                return _Z ** (M // 8) + _Z ** (7 * (M // 8)), \
                    f"rt_sub2 {M} {M // 8} (by norm_num) (by norm_num)"
            if s == 3:
                return _Z ** (M // 12) + _Z ** (11 * (M // 12)), \
                    f"rt_sub3 {M} {M // 12} (by norm_num) (by norm_num)"
            if not sp.isprime(s):
                q = min(sp.primefactors(s))
                return Rt(q) * Rt(s // q), f"rt_prod {s} {q} {s // q} (by norm_num)"
            return self.gauss(s, emit)
        return None

    def gauss(self, p: int, emit: bool = True) -> tuple[sp.Expr, str]:
        """√p for an odd prime p as a quadratic Gauss sum in ζ, with its Lean proof."""
        from archytaszx.semantics.lean_numeric import ball_tree

        M = self.M
        if not emit:
            g = sum(sp.legendre_symbol(a, p) * _Z ** (a * (M // p)) for a in range(1, p))
            h = _Z ** (M // 4) * g if p % 4 == 3 else g
            val = complex(sp.N(h.xreplace({_Z: sp.exp(2 * sp.pi * sp.I / M)}), 30))
            return sp.expand(-h if val.real < 0 else h), ""
        g = sum(sp.legendre_symbol(a, p) * _Z ** (a * (M // p)) for a in range(1, p))
        h = _Z ** (M // 4) * g if p % 4 == 3 else g
        val = complex(sp.N(h.xreplace({_Z: sp.exp(2 * sp.pi * sp.I / M)}), 30))
        if val.real < 0:
            h = -h
        h = sp.expand(h)
        lines, text, ball, x, _y, r = ball_tree(M, self.out, h, f"g{p}_")
        if not x > r:
            raise NetUnsupported(f"the Gauss sum for √{p} is not separated from zero")
        cert = self.combination(sp.expand(h**2 - p), [])
        self.facts += lines
        return h, (f"(sqrt_of_sq {text} {p} (by {cert}) (re_pos_of_near {ball} "
                   f"(by norm_num))).symm")

    # the certificate

    def symbolize(self, e: sp.Expr) -> sp.Expr:
        """``e`` with every atom a symbol of :attr:`atom_sym`, ``om M`` as ζ."""
        e = sp.sympify(e)
        if e.is_Rational or e.is_Symbol and e != II:
            return e
        if e == _Z:
            return e
        if e.is_Add:
            return sp.Add(*(self.symbolize(x) for x in e.args))
        if e.is_Mul:
            return sp.Mul(*(self.symbolize(x) for x in e.args))
        if e.is_Pow and e.args[1].is_Integer and int(e.args[1]) >= 0:
            return self.symbolize(e.args[0]) ** int(e.args[1])
        if isinstance(e, NC):
            return sp.Integer(_int(e.args[0]))
        if isinstance(e, Om) and _int(e.args[0]) == self.M:
            return _Z
        if e == II or isinstance(e, (Rt, Om, Inv)):
            if e not in self.atom_sym:
                self.atom_sym[e] = sp.Symbol(f"atom_lean_{len(self.atom_sym)}")
                self.sym_atom[self.atom_sym[e]] = e
            return self.atom_sym[e]
        raise NetUnsupported(f"no polynomial model for {e}")

    def fact(self, atom: sp.Expr) -> None:
        """Declares ``h_i : atom = rhs`` for ``atom`` and every atom its right side needs."""
        if atom in self.fact_of:
            return
        if isinstance(atom, Inv):
            inner = atom.args[0]
            for x in sp.preorder_traversal(inner):
                if (x == II or isinstance(x, (Rt, Om, Inv))) and not (
                        isinstance(x, Om) and _int(x.args[0]) == self.M):
                    self.fact(x)
            f = self.poly(atom).as_expr()
            goal = self.symbolize(inner) * f - 1
            proof = (f"inv_eq_of_mul_eq_one_right (by {self.combination(goal, [])})")
            rhs = f
        else:
            sub = self.substitute(atom)
            if sub is None:
                return
            rhs, lemma = sub
            proof = f"by have e := {lemma}; linear_combination e"
            for x in sp.preorder_traversal(sp.sympify(rhs)):
                if x == II or isinstance(x, (Rt, Om, Inv)):
                    self.fact(x)
        name = f"hf{len(self.facts)}"
        self.symbolize(atom)
        self.fact_of[atom] = (name, rhs)
        stmt = f"{self.text(atom)} = {self.text(rhs)}"
        self.facts.append(f"have {name} : {stmt} := {proof}")

    def combination(self, target: sp.Expr, hyps: Sequence[tuple[sp.Expr, str]]) -> str:
        """``linear_combination`` proving ``target = 0`` from ``hyps`` (``lhs - rhs`` with
        a coefficient each) and the atom facts, ``ζ^M = 1`` and the prime-gon sums."""
        terms = [(c, name) for c, name in hyps]
        rest = sp.expand(target - sum(c * self.hyp_diff[name] for c, name in hyps))
        free = set(self.free)
        for _ in range(64):
            rest = self.reduce_squares(rest, terms)
            atoms = sorted((s for s in rest.free_symbols if s != _Z and s not in free), key=str)
            if not atoms:
                break
            s = atoms[0]
            atom = self.sym_atom[s]
            self.fact(atom)
            name, rhs = self.fact_of[atom]
            rhs_s = self.symbolize(rhs)
            new = sp.expand(rest.xreplace({s: rhs_s}))
            q, r = sp.div(sp.Poly(rest - new, s), sp.Poly(s - rhs_s, s))
            if not r.is_zero:
                raise NetUnsupported("an atom substitution left a remainder")
            terms.append((q.as_expr(), name))
            rest = new
        else:
            raise NetUnsupported("atom substitution did not terminate")
        for x, xi, name in self.inv_pairs:
            if rest.has(x) and rest.has(xi):
                q, r = sp.reduced(rest, [x * xi - 1], x, xi)
                if q[0] != 0:
                    terms.append((q[0], name))
                rest = sp.expand(r)
        if rest != 0:
            q, r = sp.div(rest, _Z**self.M - 1, _Z)
            if sp.expand(q) != 0:
                terms.append((sp.expand(q), "hzM"))
            rest = sp.expand(r)
        if rest != 0:
            quot, rem = sp.div(rest, self.phi().as_expr(), _Z)
            if sp.expand(rem) != 0:
                raise NetUnsupported("the scalar identity does not hold in ℚ(ζ)")
            for coeff, name in zip(self.bezout, self.rel_names, strict=True):
                c = sp.expand(quot * coeff.as_expr())
                if c != 0:
                    terms.append((c, name))
        parts = [f"({self.text(self.unsym(c))}) * {name}" for c, name in terms
                 if sp.expand(c) != 0]
        return "linear_combination " + (" + ".join(parts) if parts else "0")

    def reduce_squares(self, rest: sp.Expr, terms: list[tuple[sp.Expr, str]]) -> sp.Expr:
        """``rest`` with each √x atom and its inverse of degree below 2, by ``a² = c``."""
        for s in sorted((s for s in rest.free_symbols if s != _Z), key=str):
            sq = self.square(self.sym_atom[s])
            if sq is None or sp.degree(rest, s) < 2:
                continue
            name, c = sq
            q, r = sp.div(sp.Poly(rest, s), sp.Poly(s**2 - c, s))
            terms.append((q.as_expr(), name))
            rest = sp.expand(r.as_expr())
        return rest

    def square(self, atom: sp.Expr) -> tuple[str, sp.Expr] | None:
        """``h : atom ^ 2 = c`` for ``√x`` and ``(√x)⁻¹``, declared once."""
        if atom in self.sq_of:
            return self.sq_of[atom]
        if isinstance(atom, Rt):
            x = _int(atom.args[0])
            c, proof = sp.Integer(x), f"by have e := rt_sq' {x}; linear_combination e"
        elif isinstance(atom, Inv) and isinstance(atom.args[0], Rt):
            x = _int(atom.args[0].args[0])
            c, proof = sp.Rational(1, x), f"by rw [inv_pow, rt_sq' {x}]; ring"
        else:
            return None
        name = f"hsq{len(self.sq_of)}"
        self.sq_of[atom] = (name, c)
        self.hyp_diff[name] = self.symbolize(atom) ** 2 - c
        self.facts.append(f"have {name} : {self.text(atom)} ^ 2 = {self.text(c)} := {proof}")
        return self.sq_of[atom]

    def unsym(self, e: sp.Expr) -> sp.Expr:
        return sp.sympify(e).xreplace(self.sym_atom).xreplace({_Z: Om(self.M)})

    def text(self, e: sp.Expr) -> str:
        e = sp.sympify(e).xreplace({_Z: Om(self.M)})
        return self.lean(sp.expand(e) if e.is_Add else e)

    def lean(self, e: sp.Expr) -> str:
        """``e`` as a Lean ℂ term, free symbols by their texts."""
        if e in self.free:
            return self.free[e]
        if not (e.free_symbols & set(self.free)):
            return self.out.sc.text(e)
        if e.is_Add:
            return "(" + " + ".join(self.lean(a) for a in e.args) + ")"
        if e.is_Mul:
            return "(" + " * ".join(self.lean(a) for a in e.args) + ")"
        if e.is_Pow and e.args[1].is_Integer and int(e.args[1]) >= 0:
            return f"({self.lean(e.args[0])} ^ {int(e.args[1])})"
        raise NetUnsupported(f"cannot write {e} in Lean")

    def relations(self) -> None:
        """``ζ^M = 1`` and each prime ``p ∣ M``'s ``Σ_{j<p} ζ^{jM/p} = 0``, with Bézout
        coefficients writing Φ_M over them."""
        M = self.M
        rels = [sp.Poly(_Z**M - 1, _Z, domain="QQ")]
        names = ["hzM"]
        self.rel_lines = [f"have hzM : om {M} ^ {M} = 1 := om_pow_self {M} (by norm_num)"]
        self.hyp_diff = {"hzM": _Z**M - 1}
        for p in sp.primefactors(M):
            step = M // p
            g = sum(_Z ** (j * step) for j in range(p))
            name = f"hg{p}"
            text = " + ".join("(1 : ℂ)" if j == 0 else f"om {M} ^ {j * step}" for j in range(p))
            self.rel_lines.append(
                f"have {name} : {text} = 0 := by simpa [Finset.sum_range_succ] using "
                f"om_geom {M} {p} (by norm_num) (by norm_num) (by norm_num)")
            rels.append(sp.Poly(g, _Z, domain="QQ"))
            names.append(name)
            self.hyp_diff[name] = g
        cur = rels[0]
        coeffs = [sp.Poly(1, _Z, domain="QQ")] + [sp.Poly(0, _Z, domain="QQ")] * (len(rels) - 1)
        for i, g in enumerate(rels[1:], start=1):
            s, t, h = sp.gcdex(cur, g)
            coeffs = [c * s for c in coeffs]
            coeffs[i] = coeffs[i] + t
            cur = h
        lead = cur.LC()
        coeffs = [c * (1 / lead) for c in coeffs]
        if (cur * (1 / lead)) != self.phi():
            raise NetUnsupported("the relations of ζ do not generate Φ_M")
        self.bezout = coeffs
        self.rel_names = names

    # the source

    def init_cert(self) -> None:
        """Empty fact tables and the relations of ζ."""
        self.atom_sym: dict[sp.Expr, sp.Symbol] = {}
        self.sym_atom: dict[sp.Symbol, sp.Expr] = {}
        self.fact_of: dict[sp.Expr, tuple[str, sp.Expr]] = {}
        self.sq_of: dict[sp.Expr, tuple[str, sp.Expr]] = {}
        self.free: dict[sp.Symbol, str] = getattr(self, "free", {})
        self.inv_pairs: list[tuple[sp.Symbol, sp.Symbol, str]] = getattr(self, "inv_pairs", [])
        self.facts: list[str] = []
        self.relations()

    def run(self, name: str) -> str:
        if self.arity:
            return self.arity_source(name)
        self.setup_env()
        self.choose_m()
        bnd = self.a.net.bnd
        if sorted(bnd) != sorted(self.b.net.bnd):
            raise NetUnsupported("the diagrams have different boundaries")
        sq0, sq1 = self.sq_list(self.a.net), self.sq_list(self.b.net)
        k0 = sp.Mul(*(Inv(Rt(d)) ** c for d, c in sq0))
        k1 = sp.Mul(*(Inv(Rt(d)) ** c for d, c in sq1))
        s0, s1 = self.at_point(self.s0), self.at_point(self.s1)
        f0, f1 = self.poly(s0 * k0), self.poly(s1 * k1)
        ranges = [range(self.labdim(x)) for x in bnd]
        found = None
        for count, values in enumerate(itertools.product(*ranges)):
            if count >= MAX_ASSIGNMENTS:
                break
            sigma = dict(zip(bnd, values, strict=True))
            c0, c1 = self.evaluate(self.a.net, sigma), self.evaluate(self.b.net, sigma)
            a0 = sp.Poly(sum(c * _Z**i for i, c in enumerate(c0)), _Z, domain="QQ")
            a1 = sp.Poly(sum(c * _Z**i for i, c in enumerate(c1)), _Z, domain="QQ")
            diff = (f0 * a0 - f1 * a1).rem(self.phi())
            if not diff.is_zero:
                found = sigma, c0, c1, diff
                break
        if found is None:
            raise NetUnsupported("the networks agree at every assignment checked")
        sigma, c0, c1, diff = found
        self.init_cert()
        coeffs = [sp.Rational(c) for c in reversed(diff.all_coeffs())]
        scale = math.lcm(*(int(c.q) for c in coeffs))
        r = [int(c * scale) for c in coeffs]
        a0 = sum(c * _Z**i for i, c in enumerate(c0))
        a1 = sum(c * _Z**i for i, c in enumerate(c1))
        d_atoms = sp.expand(self.symbolize(s0) * self.symbolize(k0) * a0
                            - self.symbolize(s1) * self.symbolize(k1) * a1)
        self.hyp_diff["h0"] = d_atoms
        goal = scale * sum(c * _Z**i for i, c in enumerate(coeffs))
        final = self.combination(sp.expand(goal), [(sp.Integer(scale), "h0")])
        return self.source(name, sigma, sq0, sq1, c0, c1, r, final)

    def env_at(self) -> str:
        """The environment term the statement's instance has at the point."""
        out = self.out
        if out.family:
            text = out.env_text(out.ks_vals)
        else:
            text = out.env
        return self.instantiate(text)

    def instantiate(self, text: str) -> str:
        for raw, (n, ty) in self.out.sym.params.items():
            if raw == "k_lean_index":
                continue
            text = re.sub(rf"(?<![\w.']){re.escape(n)}(?![\w'])", self.value_text(raw, ty), text)
        return text

    def value_text(self, raw: str, ty: str) -> str:
        if raw not in self.point:
            raise NetUnsupported(f"the counterexample gives no value for {raw}")
        v = sp.Rational(self.point[raw])
        if ty == "ℕ":
            return str(int(v))
        if v.q == 1:
            return f"({int(v.p)} : {ty})"
        return f"(({int(v.p)} : {ty}) / {int(v.q)})"

    def source(self, name: str, sigma: Mapping[int, int], sq0: list[tuple[int, int]],
               sq1: list[tuple[int, int]], c0: list[int], c1: list[int], r: list[int],
               final: str) -> str:
        out = self.out
        head = self.statement()
        params = [(raw, n, ty) for raw, (n, ty) in out.sym.params.items()
                  if raw != "k_lean_index" and raw not in out.mults]
        bare = [(str(e), n) for n, e, b in zip(out.mult_names, out.mult_exprs, out.mult_bare,
                                               strict=True) if b]
        args = [self.value_text(raw, ty) for raw, _, ty in params]
        args += [self.value_text(raw, "ℕ") for raw, _ in bare]
        args += ["(by norm_num)" for _, _, ty in params if ty == "ℕ"]
        env = self.instantiate(out.env)
        binders = _join(out.binders, out.hyps)
        M = self.M
        pt = ", ".join(f"(({ph}, {j}), ({self.P(ph, j)[0]}, []))" for ph, j in self.keys)
        keys = _list_pairs(self.keys)
        sig = ", ".join(f"({x}, {v})" for x, v in sorted(sigma.items()))
        unfold = "Fam.env, F, Φ" if out.family else "E, Φ"
        phok = ", ".join(self.phase_proof(ph, j, unfold) for ph, j in self.keys)
        phok_args = phok + ", " if phok else ""
        decls = [
            (f"/-- Each phase entry's power of `om {M}`. -/\n"
             f"def Pt (i k : ℕ) : ℕ × List ℤ := ((([{pt}] : List ((ℕ × ℕ) × (ℕ × List ℤ)))"
             f".lookup (i, k)).getD (0, []))"),
            (f"/-- The boundary assignment where the diagrams differ. -/\n"
             f"def σ0 (l : ℕ) : ℕ := ((([{sig}] : List (ℕ × ℕ)).lookup l).getD 0)"),
        ]
        nets = [(self.a.name, self.a.net, sq0, {(): c0} if any(c0) else {})]
        if self.b.name != self.a.name:
            nets.append((self.b.name, self.b.net, sq1, {(): c1} if any(c1) else {}))
        evals, shows, pdecls = self.eval_rewrites(env, "[]", keys, nets)
        decls += pdecls
        body = [
            "  intro h",
            f"  have he := h {' '.join(args)}" if args else "  have he := h",
            "  refine refute σ0 (inRange_of _ _ _ (by decide +kernel)) ?_ he",
            "  intro h0",
            f"  have hP : PhOK {_paren(env)} {M} [] Pt {keys} := ⟨{phok_args}trivial⟩",
            f"  rw [{evals}] at h0",
            "  unfold sqK at h0",
            f"  rw [{shows}] at h0",
            ("  simp only [sqK, gval, mono_nil, List.map_cons, List.map_nil, List.prod_cons,"
             " List.prod_nil, List.sum_cons, List.sum_nil, one_mul, cvz] at h0"),
            "  try push_cast at h0",
            *(f"  {line}" for line in self.rel_lines),
            *(f"  {line}" for line in self.facts),
            f"  exact cyc_ne {M} (by norm_num) {_ints(r)} (by decide +kernel) (by decide +kernel)",
            f"    (by simp only [cvz]; push_cast; {final})",
        ]
        if not binders:
            body[1] = "  have he := h"
            body[0] = "  intro h"
        theorem = ("set_option maxHeartbeats 0 in\n"
                   f"/-- The diagrams are not equal: they differ at {self.point_note()}. -/\n"
                   f"theorem {name} :\n    {head} := by\n" + "\n".join(body))
        out.decls += [*decls, theorem]
        return out.source(lead=f"  refutation at {self.point_note()}, ζ = om {M}")

    def statement(self) -> str:
        """The refuted claim, prepared after the family's binders."""
        out = self.out
        if out.family:
            out.sp_binders, out.sp_args = out.binders, out.args
            out.sp_hyps, out.sp_hyp_args = out.hyps, out.hyp_args
            bare = [n for n, b in zip(out.mult_names, out.mult_bare, strict=True) if b]
            out.ks_vals = list(out.mult_names)
            out.ks_syms = list(out.mult_exprs)
            out.binders = _join(out.binders, *(f"({n} : ℕ)" for n in bare))
            out.args = _join(out.args, *bare)
            out.env = out.env_text(out.ks_vals)
        stmt = out.statement(self.scalar_text(self.s0), self.a, self.scalar_text(self.s1),
                             self.b)
        binders = _join(out.binders, out.hyps)
        return f"¬ ∀ {binders}, {stmt}" if binders else f"¬ {stmt}"

    def scalar_text(self, model: sp.Expr) -> str:
        """A diagram scalar as Lean."""
        return self.out.text(model)

    def arity_source(self, name: str) -> str:
        """The refutation of diagrams whose boundaries have different sizes."""
        head = self.statement()
        args = self.zero_args()
        apply = f"(h {args})" if args else "h"
        theorem = ("set_option maxHeartbeats 0 in\n"
                   "/-- The diagrams are not equal: their boundaries differ in size. -/\n"
                   f"theorem {name} :\n    {head} := by\n"
                   f"  intro h\n  exact refute_bnd (by decide) {apply}")
        self.out.decls.append(theorem)
        return self.out.source(lead="  refutation: the boundaries differ in size")

    def zero_args(self) -> str:
        """Arguments instantiating every binder at 1."""
        params = [(n, ty) for raw, (n, ty) in self.out.sym.params.items()
                  if raw != "k_lean_index" and raw not in self.out.mults]
        bare = [n for n, b in zip(self.out.mult_names, self.out.mult_bare, strict=True) if b]
        args = ["1" if ty == "ℕ" else f"(0 : {ty})" for _, ty in params]
        args += ["0"] * len(bare)
        args += ["(by norm_num)"] * sum(ty == "ℕ" for _, ty in params)
        return " ".join(args)

    def phase_proof(self, ph: int, j: int, unfold: str) -> str:
        t = self.phase(ph, j)
        p = self.P(ph, j)[0]
        z = (t * self.M - p) / self.M
        return f"ex_atoms {self.M} {p} ({int(z)}) [] [] (by norm_num) (by norm_num [{unfold}])"

    def point_note(self) -> str:
        return ", ".join(f"{k} = {v}" for k, v in sorted(self.point.items())) or "no parameters"


def _add(v: Vec, key: Key, c: int, m: int) -> None:
    key = (_ktrim(key[0]), key[1] % m)
    n = v.get(key, 0) + c
    if n:
        v[key] = n
    else:
        v.pop(key, None)


def _mul(a: Vec, b: Vec, m: int) -> Vec:
    out: Vec = {}
    for (ka, ea), ca in a.items():
        for (kb, eb), cb in b.items():
            _add(out, (_kadd(ka, kb), ea + eb), ca * cb, m)
    return out


def _const(c: int) -> Vec:
    return {((), 0): c} if c else {}


def _kadd(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    n = max(len(a), len(b))
    return tuple((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n))


def _ktrim(k: Sequence[int]) -> tuple[int, ...]:
    k = list(k)
    while k and k[-1] == 0:
        k.pop()
    return tuple(k)


def _int(e: object) -> int:
    v = sp.sympify(e)
    if not v.is_Integer:
        raise NetUnsupported(f"{e} is not an integer at the counterexample")
    return int(v)


def _square_split(x: int) -> tuple[int, int]:
    """``(k, s)`` with ``x = k² s`` and ``s`` squarefree."""
    k, s = 1, 1
    for p, e in sp.factorint(x).items():
        k *= p ** (e // 2)
        s *= p ** (e % 2)
    return k, s


def _unsupported(a: sp.Expr) -> int:
    raise NetUnsupported(f"{a} at the counterexample")


def _list_pairs(ps: Sequence[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"({a}, {b})" for a, b in ps) + "]"


def _terms(v: Vec) -> str:
    """A vector as the formal sum `compact` gives: groups by monomial, exponents ascending."""
    by: dict[tuple[int, ...], dict[int, int]] = {}
    for (ks, e), c in v.items():
        if c:
            by.setdefault(ks, {})[e] = c
    items = [f"({c}, {e}, {_ints(list(k))})" for k in sorted(by) for e, c in sorted(by[k].items())]
    return "[" + ", ".join(items) + "]" if items else "([] : Terms)"


def _facs(fs: Sequence[Any]) -> str:
    """Factors as a Lean list."""
    out = [f"⟨{_ints_nat(list(sc))}, [{', '.join(_terms(v) for v in tab)}]⟩" for sc, tab in fs]
    return "[" + ", ".join(out) + "]" if out else "([] : List Fac)"


def _ints_nat(xs: Sequence[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]" if xs else "([] : List ℕ)"


def _groups_text(g: Mapping[tuple[int, ...], Sequence[int]]) -> str:
    """Groups as the Lean literal `evalG` gives."""
    if not g:
        return "([] : Grp)"
    return "[" + ", ".join(f"({_ints(list(k))}, {_ints(list(v))})" for k, v in g.items()) + "]"


def _groups(c: Sequence[int]) -> str:
    """A vector as the groups `evalG` gives without atoms."""
    return f"[([], {_ints(c)})]" if any(c) else "([] : Grp)"


def _ints(xs: Sequence[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]" if xs else "([] : List ℤ)"


__all__ = ["export_interface_refutation", "export_refutation"]
