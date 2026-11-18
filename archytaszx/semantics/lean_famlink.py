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

"""Lean family proofs of inductions whose sides' phases read the induction index.

:func:`linked_indices` finds such inductions in a kernel theorem. :class:`Linked` proves one
inside a :mod:`lean_net` family export, over that export's tables, in its own namespace: the
family's phase parameter ``table_n`` is the index itself,

    theorem P : ∀ n : ℕ, Equal ((F n).env (ksOf [n])) s₀ X s₁ Y

by induction on ``n``. The zero case is the normaliser's family lemma (:mod:`lean_sym`, for
every ``table_n``) on the zeroed networks, ``Y``'s phases read at ``n = 0``, carried to
``F 0`` by ``Equal.of_phRows``. The step splits one copy off ``X``'s box, unfuses the one Z
spider the copies meet (``fusionZG``), plugs the hypothesis carried from ``F k`` to
``F (k + 1)`` by ``Equal.of_phRows`` (``Y``'s phases read at ``n - 1``), and closes on ``Y``'s
split by the normaliser's family lemma at ``table_n = k + 1``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.diagram.graph import Diagram
from archytaszx.semantics import lean_sym
from archytaszx.semantics.lean_net import (
    LNet,
    LNode,
    NetUnsupported,
    _head,
    _join,
    _list,
    _pairs,
    _paren,
    _perm_to_bijection,
    _rename_net,
    _split_net,
    _tail,
    _zero_net,
    _zk,
)


def linked_indices(theorem: object) -> set[str]:
    """The index of every induction inference in ``theorem`` whose sides' phases read it."""
    from archytaszx.semantics.kernel import Theorem

    found: set[str] = set()
    stack = [theorem]
    while stack:
        t = stack.pop()
        assert isinstance(t, Theorem)
        inf = t.inference
        if inf.kind == "induction":
            idx = str(inf.data.get("index"))
            if any(idx in _phase_names(d) for d in (t.lhs, t.rhs)):
                found.add(idx)
        stack.extend(inf.premises)
    return found


def _phase_names(d: Diagram) -> set[str]:
    """The symbol names in ``d``'s node phases."""
    return {str(x) for n in d.nodes.values() if n.phase is not None for x in n.phase.free_symbols}


def _map_ph(net: LNet, g: dict[int, int]) -> LNet:
    """Python mirror of ``Net.mapPh (phMap g)``."""
    nodes = tuple(
        n if _head(n.kind) in ("B", "S") else LNode(n.kind, n.outs, n.ins, n.dim, g.get(n.ph, n.ph))
        for n in net.nodes
    )
    return LNet(nodes, net.summed, net.bnd)


def _rows(net: LNet) -> list[int]:
    """The phase rows ``net``'s nodes other than B and S read."""
    return sorted({n.ph for n in net.nodes if _head(n.kind) not in ("B", "S")})


def _mem_cases(qs: Sequence[int], tactic: str) -> str:
    """A proof of ``∀ q ∈ qs, goal q`` running ``tactic`` on each literal ``q``."""
    if not qs:
        return "(fun q hq => by simp at hq)"
    alts = " | ".join(["rfl"] * len(qs))
    return (
        "(by intro q hq; simp only [List.mem_cons, List.not_mem_nil, or_false] at hq; "
        f"rcases hq with {alts} <;> {tactic})"
    )


_ROW_EQ = (
    "first | rfl | (funext j; simp [Fam.env, F, Φ, phMap] <;> split_ifs <;> push_cast <;> ring)"
)
"""A tactic proving two phase rows of the family equal at their parameters."""


class Linked(lean_sym._SymExport):
    """The induction lemma of a :mod:`lean_net` family export whose goal's phases read the
    index ``idx``; ``start`` and ``goal`` are the export's networks of its sides."""

    def __init__(
        self,
        out: Any,
        start: Any,
        goal: Any,
        lhs: Diagram,
        rhs: Diagram,
        idx: str,
        tag: str,
    ) -> None:
        self.out = out
        self.alias: dict[str, str] = {}
        self.a, self.b = start, goal
        self.idx, self.tag = idx, tag
        self.mults = [n for n, bare in zip(out.mult_ix, out.mult_bare, strict=True) if bare]
        if self.mults != [idx] or idx not in out.mult_ix:
            raise NetUnsupported("a linked induction beside other multiplicities")
        self.tsym = sp.Symbol(f"table_{idx}", integer=True, nonnegative=True)
        if str(self.tsym) not in out.sym.params:
            raise NetUnsupported("the induction index reads no phase table")
        self.tname = out.sym.params[str(self.tsym)][0]
        if idx in _phase_names(lhs):
            raise NetUnsupported("a linked induction whose start's phases read the index")
        self.step_g: dict[int, int] = {}
        self.zero_g: dict[int, int] = {}
        for node in rhs.nodes.values():
            pv = node.phase
            if pv is None or idx not in pv.free_symbols:
                continue
            p = out.phase(pv)
            self.step_g[p] = out.phase(pv.substitute({idx: sp.Symbol(idx) - 1}))
            self.zero_g[p] = out.phase(pv.substitute({idx: 0}))
        if not self.step_g:
            raise NetUnsupported("the goal's phases do not read the index")
        if not hasattr(out, "mixed"):
            out.mixed = set()
        saved = (out.binders, out.args, out.hyps, out.hyp_args, out.env)
        try:
            self.tables(out.scalar(lhs.scalar), out.scalar(rhs.scalar), {})
        finally:
            out.binders, out.args, out.hyps, out.hyp_args, out.env = saved
        if self.common is not None or self.constraints:
            raise NetUnsupported("a linked induction with a common scalar or constraints")

    def phase_value(self, v: sp.Expr) -> sp.Expr:
        """``v`` with the index read as the family's phase parameter."""
        v = sp.sympify(v)
        return v.xreplace({x: self.tsym for x in v.free_symbols if str(x) == self.idx})

    def at(self, text: str, value: str) -> str:
        """``text`` with the phase parameter set to ``value``."""
        return self.retext(text, self.tname, value)

    def transfer(
        self,
        e0: str,
        e1: str,
        gm: dict[int, int],
        n: str,
        nn: Any,
        m: str,
        mn: Any,
        m1: str,
        hd: str = "rfl",
    ) -> str:
        """``Equal.of_phRows`` from ``e0`` to ``e1`` (``hd`` their dimension values equal):
        ``n`` kept, ``m`` read through ``gm`` as ``m1``; the premise follows as the next
        argument."""
        pairs = _pairs(sorted(gm.items()))
        keep = _mem_cases(_rows(nn), "rfl")
        moved = _mem_cases(_rows(mn), _ROW_EQ)
        ok = "(by decide +kernel)"
        return (
            f"Equal.of_phRows {e0} {e1} (phMap []) (phMap {pairs}) {_list(_rows(nn))} "
            f"{_list(_rows(mn))} {hd} rfl rfl {keep} (fun _ _ => rfl) {moved} (fun _ _ => rfl) "
            f"{n} {m} {n} {m1} {ok} {ok} {ok} {ok}"
        )

    def build(self) -> list[str]:
        """The namespace ``tag`` holding the normaliser's tables, the lemmas and ``P``."""
        out = self.out
        self.texts()
        m = out.mult_ix[self.idx]
        x, xt, y, yt = self.a.net, self.a.name, self.b.net, self.b.name
        vs = self.var_names()
        name = vs[m]
        for q in _rows(x):
            entries = out.phase_specs[q][0] if q < len(out.phase_specs) else {}
            if any(
                self.idx in {str(s) for s in sp.sympify(v).free_symbols} for v in entries.values()
            ):
                raise NetUnsupported("the start reads a phase row of the index")
        sc = (sp.sympify(self.m0), sp.sympify(self.m1), self.s0t, self.s1t)
        labels = [
            lb
            for net in (x, y)
            for lb in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels()))
        ]
        H = 1 << max(16, max(labels, default=0).bit_length())
        for net in (x, y):
            for lab in (*net.summed, *net.bnd, *(q for n in net.nodes for q in n.labels())):
                if out.label_chain(lab)[:1] == (m,):
                    out.peel_reg(lab, H)
        isb = out.isb(m)
        spc = f"(SpC {m})"
        i = len(self.fam_decls)
        tag = self.tag

        y0 = _map_ph(y, self.zero_g)
        back = {q: p for p, q in self.zero_g.items()}
        if _map_ph(y0, back) != y or len(back) != len(self.zero_g):
            raise NetUnsupported("the goal's phases at zero share a row with its other phases")
        ys = _map_ph(y, self.step_g)
        tz, tys = f"L{i}z", f"L{i}y"
        zx, zy0 = _zero_net(x, m, isb), _zero_net(y0, m, isb)
        if _map_ph(zy0, back) != _zero_net(y, m, isb):
            raise NetUnsupported("the zeroed goal does not read its rows back")
        self.net_decls += [
            (
                f"/-- The goal's zero network, its phases at `{name} = 0`. -/\ndef {tz} : Net := "
                f"{zy0.lean()}"
            ),
            f"/-- The goal, its phases at `{name} - 1`. -/\ndef {tys} : Net := {ys.lean()}",
        ]
        lz = self.fam(zx, f"(Net.zero {spc} {xt})", zy0, tz, self.scal_at(sc, m, "0", 0))

        spiders = [
            n
            for n in x.nodes
            if _head(n.kind) == "ZB" and any(isb(lab) and lab in x.summed for lab in n.ins)
        ]
        if len(spiders) != 1:
            raise NetUnsupported("a linked node box meeting other than one Z spider")
        zn = spiders[0]
        if any(_head(n.kind) == "Bun" and n.kind[1] != m for n in x.nodes):
            raise NetUnsupported("towers along two multiplicities")
        o = next((lab for lab in zn.outs if lab in x.bnd), None)
        if o is None:
            raise NetUnsupported("the spider meeting the node box has no boundary output")
        sp0, sp1 = _split_net(x, m, isb, H), _split_net(y, m, isb, H)
        others = [n for n in x.nodes if n is not zn]
        rest = list(sp0.nodes)
        for n in others:
            if n not in rest:
                raise NetUnsupported("the split network does not contain the network")
            rest.remove(n)
        split_z = [n for n in rest if _head(n.kind) == "ZB" and n.outs == zn.outs]
        if len(split_z) != 1:
            raise NetUnsupported("the split spider is not found")
        zs = split_z[0]
        copy = [n for n in rest if n is not zs]
        cs1 = tuple(zn.kind[1])
        t1 = _tail(zn.ins, cs1)
        i1 = tuple(zn.ins[: len(zn.ins) - len(cs1)])
        extra = tuple(zs.ins[len(zn.ins) :])
        cs2 = ((),) * len(extra)
        if not extra or tuple(zs.ins) != i1 + tuple(t1) + extra or tuple(zs.kind[1]) != cs1 + cs2:
            raise NetUnsupported("the split spider's legs are not the unfused form")
        w = out.fresh_like(o)
        o1 = tuple(w if lab == o else lab for lab in zn.outs)
        za = LNode(zn.kind, o1, zn.ins, zn.dim, zn.ph)
        zb = LNode(_zk(cs2), (o,), (w, *extra), zs.dim, 0)
        kvs = [("k_" if j == m else v) for j, v in enumerate(vs)]
        zvs = [("0" if j == m else v) for j, v in enumerate(vs)]
        k1 = "(k_ + 1)"
        env0 = self.at(self.env_at(kvs), "k_")
        env1 = self.at(self.env_at(kvs), k1)
        envz = self.at(self.env_at(zvs), "0")
        sk = self.scal_at(sc, m, "k_", sp.Symbol("k_"))
        sk1 = self.scal_at(sc, m, k1, sp.Symbol("k_") + 1)
        al, be = self.ratio(sc[0], m), self.ratio(sc[1], m)
        alt, bet = _paren(out.text(al)), _paren(out.text(be))
        bnd = sp0.bnd
        bl = _list(bnd)

        def iso_s(a_text: str, a: Any, b_text: str, b: Any) -> str:
            return out.iso_b(env1, a_text, a, b_text, b)

        def lit(nodes: Sequence[Any]) -> str:
            return "[" + ", ".join(n.lean() for n in nodes) + "]"

        ctx1 = tuple(copy) + tuple(others)
        n1 = LNet(ctx1 + (zs,), sp0.summed, bnd)
        n2 = LNet(ctx1 + (za, zb), sp0.summed + (w,), bnd)
        n1t, n2t = f"L{i}s", f"L{i}u"
        self.net_decls += [
            f"/-- The split network, its spider last. -/\ndef {n1t} : Net := {n1.lean()}",
            f"/-- The split network with its spider unfused. -/\ndef {n2t} : Net := {n2.lean()}",
        ]
        iface = tuple(dict.fromkeys(zs.labels()))
        fusion = (
            f"fusionZG {env1} {self.fmt_cs(cs1)} {self.fmt_cs(cs2)} {za.ph} {zb.ph} {zs.ph} "
            f"{zs.dim} {w} {_list(o1)} {_list(i1)} {_list(t1)} {_list(zb.outs)} {_list((w,))} "
            f"{_list(extra)} {_list(zs.outs)} {_list(i1)} {_list(iface)} (by chk) (by chk) "
            "(by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by phs)"
        )
        unfuse = (
            f"(plug {env1} {lit(ctx1)} {lit([zs])} {lit([za, zb])} {_list(sp0.summed)} [] "
            f"[{w}] {bl} {_list(iface)} 1 1 (fun τ hτ => (({fusion}) τ hτ).symm) (by chk) "
            "(by chk) (by chk))"
        )
        mapping: dict[int, int] = {o: w}
        for lab in ys.summed:
            mapping.setdefault(lab, out.fresh_like(lab))
        pairs = _perm_to_bijection(mapping)
        pt = _pairs(pairs)
        rx, ry = _rename_net(x, pairs), _rename_net(ys, pairs)
        ctx3 = tuple(copy) + (zb,)
        ic3 = tuple(lab for lab in n2.summed if lab not in set(rx.summed))
        n3 = LNet(ctx3 + rx.nodes, ic3 + rx.summed, bnd)
        n4 = LNet(ctx3 + ry.nodes, ic3 + ry.summed, bnd)

        def plug_text(t: str) -> str:
            return (
                f"⟨{lit(ctx3)} ++ (Net.rename {pt} {t}).nodes, {_list(ic3)} ++ "
                f"(Net.rename {pt} {t}).int, {bl}⟩"
            )

        fam_k, fam_k1 = self.at(f"(F {self.args})", "k_"), self.at(f"(F {self.args})", k1)
        hd = (
            f"(Fam.env_dimv_congr {fam_k} {fam_k1} _ (by show Δ {self.at(self.args, 'k_')} = "
            f"Δ {self.at(self.args, k1)}; funext i; unfold Δ; rfl) rfl)"
        )
        moved = f"({self.transfer(env0, env1, self.step_g, xt, x, yt, y, tys, hd)} ih)"
        hr = f"(Equal.rename {env1} {pt} {xt} {tys} (by chk) (by chk) {moved})"
        plug = (
            f"(plug {env1} {lit(ctx3)} (Net.rename {pt} {xt}).nodes (Net.rename {pt} {tys}).nodes "
            f"{_list(ic3)} (Net.rename {pt} {xt}).int (Net.rename {pt} {tys}).int {bl} "
            f"(Net.rename {pt} {xt}).bnd {sk[2]} {sk[3]} (Equal.toLocal _ _ _ {hr}) (by chk) "
            "(by chk) (by chk))"
        )
        rt = f"L{i}r"
        self.net_decls.append(
            f"/-- The rest after the hypothesis. -/\ndef {rt} : Net := {n4.lean()}"
        )
        lt = self.fam(n4, rt, sp1, f"(Net.split {spc} {H} {yt})", (al, be, alt, bet))
        chain = (
            "Equal.congr_s (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans "
            f"(Equal.mul_trans {iso_s(f'(Net.split {spc} {H} {xt})', sp0, n1t, n1)}\n"
            f"    {unfuse})\n    {iso_s(n2t, n2, plug_text(xt), n3)})\n    {plug})\n"
            f"    {iso_s(plug_text(tys), n4, rt, n4)})\n    "
            f"{self.at(self.call(lt, *kvs), k1)}) (by ring) (by ring)"
        )
        ok0 = lean_sym._split_ok(m, H, xt)
        ok1 = lean_sym._split_ok(m, H, yt)
        fam = f"(F {self.args})" if self.args else "F"
        hyp0 = " ".join(f"(by omega : 0 < {n})" for n in self.dnames)
        fpos = f"(Fpos {_join(self.args, hyp0)} _)" if (self.args or hyp0) else "(Fpos _)"
        bh = self.bh.replace(f"({self.tname} : ℕ)", "").strip()
        zxt, zyt = f"(Net.zero {spc} {xt})", f"(Net.zero {spc} {yt})"
        zero = (
            f"({self.transfer(envz, envz, back, zxt, zx, tz, zy0, zyt)} "
            f"(Equal.congr_s {self.at(self.call(lz, *zvs), '0')} (by ring) (by ring)))"
        )
        s0n, s1n = self.s0t, self.s1t
        envn = self.at(self.env_at(vs), name)
        lemma = "\n\n".join(
            [
                (
                    f"/-- The step at `{name} = k_ + 1`. -/\n"
                    f"theorem Q {_join(bh, '(k_ : ℕ)')}\n"
                    f"    (ih : Equal {env0} {sk[2]} {xt} {sk[3]} {yt}) :\n"
                    f"    Equal {env1} {sk1[2]} (Net.split {spc} {H} {xt}) {sk1[3]} "
                    f"(Net.split {spc} {H} {yt}) :=\n  {chain}"
                ),
                (
                    "/-- The induction, the family's phase parameter at the index. -/\n"
                    f"theorem P {bh} : ∀ {name} : ℕ, Equal {envn} {s0n} {xt} {s1n} {yt} := by\n"
                    f"  intro {name}\n  induction {name} with\n"
                    f"  | zero => exact Equal.of_zero_at {self.at(fam, '0')} {m} {H} "
                    f"[{', '.join(zvs)}] (by simp) {xt} {yt} {ok0} {ok1} (by decide +kernel) "
                    f"{zero}\n"
                    f"  | succ k_ ih => exact Equal.of_split_at {self.at(fam, k1)} {m} {H} k_ "
                    f"[{', '.join(kvs)}] (by simp) {xt} {yt} {ok0} {ok1} "
                    f"(fun _ _ _ => Fam.dimv_pos _ _ _ {self.at(fpos, k1)}) (Q k_ ih)"
                ),
            ]
        )
        return [
            f"namespace {tag}",
            *self.prelude(),
            *self.stab_decls,
            *self.net_decls,
            *reversed(self.fam_decls),
            lemma,
            f"end {tag}",
        ]
