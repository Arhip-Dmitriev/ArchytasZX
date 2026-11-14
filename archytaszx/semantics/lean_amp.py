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

"""Amplitude atoms in the Lean path-sum export (ArchytasZX/PathSumAmp.lean). ``Coefs`` numbers
the atoms' cyclotomic bases as the coefficient vector ``A``; ``AmpRun`` follows a reduction
trace of a path sum with atoms and writes the ``AStep`` list replaying it: inner steps on the
path of variables no atom reads, [Average] (plain and through a half phase) on held variables,
pins along a constraint, folds of atoms on 0 and 1, [Quarter] and [Turn] on inner variables,
releases of held variables no atom reads, and after the canonical pivots the atoms'
substitution, normalisation, merging and complementary pairs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.pathsum import (
    ONE,
    Cyclo,
    F2Poly,
    Mono,
    PathSumUnsupported,
    Step,
    f2_substitute,
)

Atom = tuple[int, F2Poly]
"""A coefficient index and an F_2 polynomial."""


def f2_text(poly: F2Poly) -> str:
    """An F_2 polynomial as a Lean list of sorted monomials."""
    return "[" + ", ".join(str(sorted(m)) for m in sorted(poly, key=sorted)) + "]"


@dataclass
class Coefs:
    """The atoms' bases by index over zeta_m; equal bases share an index."""

    m: int
    values: list[Cyclo] = field(default_factory=list)

    def index(self, c: Cyclo) -> int:
        """The index of ``c``, added when new."""
        for i, v in enumerate(self.values):
            if v == c:
                return i
        self.values.append(c)
        return len(self.values) - 1

    def model(self, i: int) -> sp.Expr:
        """Base ``i`` over the lean_scalar atom ``Om(m)``."""
        from archytaszx.semantics.lean_scalar import Om

        c = self.values[i]
        if self.m % c.n:
            raise PathSumUnsupported(f"an atom base {c} off zeta_{self.m}")
        return sp.expand(
            sum(
                (
                    sp.Rational(v.numerator, v.denominator) * Om(self.m) ** (e * self.m // c.n)
                    for e, v in c.c.items()
                ),
                sp.Integer(0),
            )
        )

    def lean_def(self, sc: object) -> list[str]:
        """The coefficient vector ``A``."""
        text = sc.text  # type: ignore[attr-defined]
        return [
            "/-- The atoms' coefficients. -/",
            "noncomputable def A : ℕ → ℂ",
            *(f"  | {i} => {text(self.model(i))}" for i in range(len(self.values))),
            "  | _ => 0",
            "",
        ]


Req = tuple[int, int, tuple[tuple[int, ...], ...]]
"""``k * A j = Sum_ps Prod_{i in ps} A i``."""


@dataclass
class AmpRun:
    """The Lean state of a path sum with atoms: inner path, held variables and atoms, with the
    AStep texts written so far, their coefficient identities and factors."""

    coefs: Coefs
    cur: list[int]
    held: list[int]
    atoms: list[Atom]
    steps: list[str] = field(default_factory=list)
    reqs: list[Req] = field(default_factory=list)
    facs: list[sp.Expr] = field(default_factory=list)
    omega: bool = False
    qreqs: list[tuple[int, int, int]] = field(default_factory=list)

    def avars(self) -> set[int]:
        """The variables the atoms read."""
        return {v for _j, f in self.atoms for m in f for v in m}

    def inner(self, texts: Iterable[str]) -> None:
        """PSem steps on the inner path sum."""
        from archytaszx.semantics.lean_scalar import Om, Rt

        m = self.coefs.m
        for st in texts:
            self.steps.append(f".inner ({st})")
            if st in (".elim", ".hhCons", ".hhPivot", ".hhNl"):
                self.facs.append(sp.Integer(2))
            elif st == ".omega":
                self.facs.append(Rt(2) * Om(m) ** (m // 8))
                self.omega = True
            elif st == ".omegaConj":
                self.facs.append(Rt(2) * Om(m) ** (7 * m // 8))
                self.omega = True

    def trace(self, step: Step) -> None:
        """One trace step of the inner path sum."""
        from archytaszx.semantics.lean_pathsum import trace_step

        texts, self.cur = trace_step(step, self.cur)
        self.inner(texts)

    def _hfront(self, y: int) -> None:
        rest = [v for v in self.held if v != y]
        if self.held[:1] != [y]:
            self.steps.append(f".hperm {[y, *rest]}")
        self.held = [y, *rest]

    def release(self) -> None:
        """Every held variable no atom reads moved to the inner path."""
        read = self.avars()
        for y in [v for v in self.held if v not in read]:
            self._hfront(y)
            self.steps.append(".release")
            self.held = self.held[1:]
            self.cur = [y, *self.cur]

    def _pick(self, atom: Atom) -> None:
        i = self.atoms.index(atom)
        if i:
            self.steps.append(f".apick {i}")
        self.atoms = [atom, *(a for k, a in enumerate(self.atoms) if k != i)]

    def fold(self, path: set[int], unit: bool = False) -> None:
        """Atoms on 0 dropped, on 1 into the constant, of base 0 free of ``path`` made
        constraints and, when ``unit``, of base 1 dropped, in list order."""
        if any(not f for _j, f in self.atoms):
            self.steps.append(".adrop")
            self.atoms = [a for a in self.atoms if a[1]]
        for atom in list(self.atoms):
            j, f = atom
            base = self.coefs.values[j]
            if f == ONE:
                self._pick(atom)
                self.steps.append(f".aconst {j}")
                self.facs.append(self.coefs.model(j))
            elif unit and base == Cyclo.rational(1):
                self._pick(atom)
                self.steps.append(f".aone {j}")
                self.reqs.append((1, j, ((),)))
            elif base.is_zero and not any(v in path for m in f for v in m):
                self._pick(atom)
                self.steps.append(f".azero {j}")
                self.reqs.append((1, j, ()))
            else:
                continue
            self.atoms = self.atoms[1:]

    def average(self, y: int, guard: F2Poly) -> None:
        """[Average] on the held variable ``y`` with the guard ``guard``."""
        one_y = frozenset(guard | {m | {y} for m in guard})
        only_y = frozenset(m | {y} for m in guard)
        reading = [a for a in self.atoms if any(y in m for m in a[1])]
        alpha = [j for j, f in reading if f == one_y]
        beta = [j for j, f in reading if f != one_y]
        if any(f not in (one_y, only_y) for _j, f in reading):
            raise PathSumUnsupported(f"an atom on {y} is not on (1 + y) g or y g")
        a, b = Cyclo.rational(1), Cyclo.rational(1)
        for j in alpha:
            a = a * self.coefs.values[j]
        for j in beta:
            b = b * self.coefs.values[j]
        new = self.coefs.index((a + b).scale(Fraction(1, 2)))
        self._hfront(y)
        self.steps.append(f".average {y} {f2_text(guard)} {new} {alpha} {beta}")
        self.reqs.append((2, new, (tuple(alpha), tuple(beta))))
        self.facs.append(sp.Integer(2))
        self.held = self.held[1:]
        self.atoms = [a for a in self.atoms if not any(y in m for m in a[1])] + [(new, guard)]

    def average_half(self, y: int, q: F2Poly) -> None:
        """[Average] through the half phase ``y q / 2`` on the held variable ``y``."""
        one_y = frozenset({frozenset(), frozenset({y})})
        only_y = frozenset({frozenset({y})})
        reading = [a for a in self.atoms if any(y in m for m in a[1])]
        if any(f not in (one_y, only_y) for _j, f in reading):
            raise PathSumUnsupported(f"an atom on {y} is not on 1 + y or y")
        alpha = [j for j, f in reading if f == one_y]
        beta = [j for j, f in reading if f != one_y]
        a, b = Cyclo.rational(1), Cyclo.rational(1)
        for j in alpha:
            a = a * self.coefs.values[j]
        for j in beta:
            b = b * self.coefs.values[j]
        j0 = self.coefs.index((a + b).scale(Fraction(1, 2)))
        j1 = self.coefs.index((a - b).scale(Fraction(1, 2)))
        jn = self.coefs.index(Cyclo.rational(-1))
        self._hfront(y)
        self.inner([".norm"])
        self.steps.append(f".averageH {y} {j0} {j1} {jn} {alpha} {beta}")
        self.reqs += [
            (2, j0, (tuple(alpha), tuple(beta))),
            (2, j1, (tuple(alpha), (*beta, jn))),
            (0, jn, ((), (jn,))),
        ]
        self.facs.append(sp.Integer(2))
        self.held = self.held[1:]
        self.atoms = [atom for atom in self.atoms if not any(y in m for m in atom[1])] + [
            (j0, frozenset(q ^ {frozenset()})),
            (j1, q),
        ]

    def quarter(self, y: int, b0: F2Poly, b1: F2Poly) -> None:
        """[Quarter] on the inner path variable ``y`` with the digits ``b0``, ``b1``: the atom
        ``((1 + i) / 2)^b0`` appended."""
        from archytaszx.algebra.pathsum import QUARTER_BASE

        if self.coefs.m % 4:
            raise PathSumUnsupported(f"[Quarter] over zeta_{self.coefs.m}")
        rest = [v for v in self.cur if v != y]
        self.inner([f".perm {[y, *rest]}"])
        self.cur = rest
        j = self.coefs.index(QUARTER_BASE)
        self.steps.append(f".quarter {f2_text(b0)} {f2_text(b1)} {j}")
        self.qreqs.append((self.coefs.m // 4, j, 1))
        self.facs.append(sp.Integer(2))
        self.atoms = [*self.atoms, (j, b0)]

    def turn(self, y: int, bs: Sequence[F2Poly]) -> None:
        """[Turn] on the inner path variable ``y`` with the digits ``bs``: one atom
        ``((1 + zeta^s) / 2)^[bs = s]`` appended per ``s`` of turn_values."""
        from archytaszx.algebra.pathsum import turn_base, turn_indicator, turn_values

        k = len(bs)
        if self.coefs.m % (1 << k):
            raise PathSumUnsupported(f"[Turn] by 2^{k} over zeta_{self.coefs.m}")
        rest = [v for v in self.cur if v != y]
        self.inner([f".perm {[y, *rest]}"])
        self.cur = rest
        pairs = [(s, self.coefs.index(turn_base(s, k))) for s in turn_values(k)]
        digits = "[" + ", ".join(f2_text(b) for b in bs) + "]"
        listed = "[" + ", ".join(f"({s}, {j})" for s, j in pairs) + "]"
        self.steps.append(f".turn {digits} {listed}")
        self.qreqs += [(s * (self.coefs.m >> k), j, 1) for s, j in pairs]
        self.facs.append(sp.Integer(2))
        self.atoms = [*self.atoms, *((j, turn_indicator(bs, s)) for s, j in pairs)]

    def pin(self, y: int, value: F2Poly) -> None:
        """[Pin] of ``y`` along its constraint ``y + value``: when an atom reads ``y``, the
        path variables of ``value`` held and ``y`` substituted in the atoms first."""
        self.inner([".norm"])
        if y in self.avars():
            for v in sorted({v for m in value for v in m} & set(self.cur)):
                rest = [x for x in self.cur if x != v]
                self.inner([f".perm {[v, *rest]}"])
                self.steps.append(".hold")
                self.cur, self.held = rest, [v, *self.held]
            self.steps.append(f".asubB {y} {f2_text(value)}")
            self.atoms = [(j, f2_substitute(f, y, value)) for j, f in self.atoms]
            self.release()
        rest = [x for x in self.cur if x != y]
        self.inner([f".perm {[y, *rest]}", f".pin {f2_text(value)}"])
        self.cur = rest

    def substitute(self, rows: Sequence[tuple[int, F2Poly]]) -> None:
        """The atoms' pivots set by the rows, then every atom normalised and atoms on one
        polynomial merged."""
        for pivot, rest in rows:
            if pivot in self.avars():
                self.steps.append(f".asub {pivot}")
                self.atoms = [(j, f2_substitute(f, pivot, rest)) for j, f in self.atoms]
        if self.atoms:
            self.steps.append(".anorm")
        self.fold(set(), True)
        while True:
            seen: dict[F2Poly, Atom] = {}
            pair = None
            for atom in self.atoms:
                if atom[1] in seen:
                    pair = (seen[atom[1]], atom)
                    break
                seen[atom[1]] = atom
            if pair is None:
                break
            first, second = pair
            self._pick(second)
            self._pick(first)
            j = self.coefs.index(self.coefs.values[first[0]] * self.coefs.values[second[0]])
            self.steps.append(f".amerge {j} {first[0]} {second[0]}")
            self.reqs.append((1, j, ((first[0], second[0]),)))
            self.atoms = [(j, first[1]), *self.atoms[2:]]
            self.fold(set(), True)
        self.complement()

    def complement(self) -> None:
        """Two atoms of one index on polynomials adding to 1 moved into the constant, until
        none is left."""
        while True:
            pair = next(
                (
                    (a, b)
                    for i, a in enumerate(self.atoms)
                    for b in self.atoms[i + 1 :]
                    if a[0] == b[0] and a[1] ^ b[1] == ONE
                ),
                None,
            )
            if pair is None:
                return
            first, second = pair
            self._pick(second)
            self._pick(first)
            j = first[0]
            self.steps.append(f".acomp {j}")
            self.facs.append(self.coefs.model(j))
            self.atoms = self.atoms[2:]

    def normalize(self, phase: Sequence[tuple[Mono, Fraction]], support: F2Poly) -> None:
        """The final sum's phase ``phase`` and atoms in atom form: with the constraint ``a + a b``
        (support_lift of ``support``) the phase monomials holding ``a`` times ``b`` (psup), every
        nonconstant phase term moved to an atom (aphase), every atom on a polynomial with a
        constant term ``x^(1 + g)`` written ``x (1 / x)^g`` (aflip), then atoms normalised,
        merged and paired."""
        from archytaszx.algebra.pathsum import EMPTY, support_lift

        m = self.coefs.m
        lift = support_lift(support)
        terms = {mono: c % 1 for mono, c in phase if mono and c % 1}
        if lift is not None:
            a, b = lift
            self.steps.append(f".psup {sorted(a)} {sorted(b)}")
            terms = _lifted(terms, a, b)
            self.inner([".norm"])
        for mono, c in sorted(terms.items(), key=lambda t: sorted(t[0])):
            j = self.coefs.index(Cyclo.root(c))
            e = int(c * m) % m
            self.steps.append(f".aphase {sorted(mono)} {e} {j}")
            self.qreqs.append((e, j, 0))
            self.atoms = [*self.atoms, (j, frozenset({mono}))]
        one = self.coefs.index(Cyclo.rational(1))
        for atom in list(self.atoms):
            j, f = atom
            if EMPTY not in f or f == ONE:
                continue
            self._pick(atom)
            inv = self.coefs.index(self.coefs.values[j].inverse())
            self.steps.append(f".aflip {j} {inv} {one}")
            self.reqs += [(1, one, ((),)), (1, one, ((j, inv),))]
            self.facs.append(self.coefs.model(j))
            self.atoms = [(inv, frozenset(f ^ {EMPTY})), *self.atoms[1:]]
        self.substitute([])

    def final_key(self) -> list[tuple[int, F2Poly]]:
        """The final atoms as a sorted multiset."""
        return sorted(self.atoms, key=lambda a: (a[0], sorted(sorted(m) for m in a[1])))


def _lifted(terms: dict[Mono, Fraction], a: Mono, b: Mono) -> dict[Mono, Fraction]:
    """Phase terms with every monomial holding ``a`` times ``b``, equal monomials added."""
    out: dict[Mono, Fraction] = {}
    for mono, c in terms.items():
        key = mono | b if a <= mono else mono
        out[key] = (out.get(key, Fraction(0)) + c) % 1
    return {k: v for k, v in out.items() if v}


def reduce_run(run: AmpRun, trace: Sequence[Step]) -> None:
    """``run`` through a reduction trace: [Average] steps on held variables, the others on the
    inner path, folds and releases after each."""
    run.fold(set(run.cur) | set(run.held))
    run.release()
    for step in trace:
        if step.rule == "average":
            if step.value is None:
                raise PathSumUnsupported("an [Average] step without its guard")
            run.average(step.var, step.value)
            run.fold(set(run.cur) | set(run.held))
        elif step.rule in ("average_half", "pin"):
            if step.value is None:
                raise PathSumUnsupported(f"a [{step.rule}] step without its polynomial")
            if step.rule == "pin":
                run.pin(step.var, step.value)
            else:
                run.average_half(step.var, step.value)
            run.fold(set(run.cur) | set(run.held))
        elif step.rule == "quarter":
            b0, b1 = step.values
            run.quarter(step.var, b0, b1)
            run.fold(set(run.cur) | set(run.held))
        elif step.rule == "turn":
            run.turn(step.var, step.values)
            run.fold(set(run.cur) | set(run.held))
        else:
            run.trace(step)
        run.release()
    if run.held:
        raise PathSumUnsupported(f"the held variables {run.held} remain")


def reqs_lines(tag: str, reqs: Sequence[Req], coefs: Coefs, sc: object, steps: str) -> list[str]:
    """The coefficient identities of a step list and the theorem collecting them."""
    prove = sc.prove  # type: ignore[attr-defined]
    text = sc.text  # type: ignore[attr-defined]
    lines: list[str] = []
    names = []
    for i, (k, j, terms) in enumerate(reqs):
        lhs = k * coefs.model(j)
        rhs = sp.Add(*(sp.Mul(*(coefs.model(x) for x in ps)) for ps in terms), sp.Integer(0))
        name = f"req{tag}{i}"
        names.append(name)
        lines += [
            f"theorem {name} : reqHolds A ({k}, {j}, {[list(ps) for ps in terms]}) := by",
            f"  have key : ({text(sp.expand(lhs))} : ℂ) = {text(sp.expand(rhs))} :=",
            *("    " + ln for ln in prove(sp.expand(lhs), sp.expand(rhs)).splitlines()),
            (
                "  simp only [reqHolds, A, List.map_cons, List.map_nil, List.prod_cons, "
                "List.prod_nil, List.sum_cons, List.sum_nil]"
            ),
            "  all_goals push_cast",
            "  all_goals linear_combination key",
            "",
        ]
    listed = ", ".join(f"({k}, {j}, {[list(ps) for ps in terms]})" for k, j, terms in reqs)
    lines += [
        f"theorem reqs{tag} : ∀ r ∈ areqs {steps}, reqHolds A r := by",
        f"  have e : areqs {steps} = [{listed}] := by decide",
        "  rw [e]",
        "  intro r hr",
        "  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr",
    ]
    if names:
        lines += [
            "  rcases hr with " + " | ".join("rfl" for _ in names),
            "  exacts [" + ", ".join(names) + "]",
        ]
    return [*lines, ""]


def qreqs_lines(tag: str, run: AmpRun, sc: object, steps: str) -> list[str]:
    """The theorem that every triple ``(e, j, b)`` a [Quarter], [Turn] or phase-atom step of a
    step list assumes has ``(b + 1) * A j = b + om m ^ e``."""
    from archytaszx.semantics.lean_scalar import Om

    prove = sc.prove  # type: ignore[attr-defined]
    text = sc.text  # type: ignore[attr-defined]
    m = run.coefs.m
    triples = list(run.qreqs)
    listed = "[" + ", ".join(f"({e}, {j}, {b})" for e, j, b in triples) + "]"
    head = [
        f"theorem qreqs{tag} : ∀ p ∈ aqreqs {m} {steps}, qreqHolds {m} A p := by",
        f"  have e : aqreqs {m} {steps} = {listed} := by decide",
        "  rw [e]",
        "  intro p hp",
    ]
    if not triples:
        return [*head, "  simp at hp", ""]
    lines: list[str] = []
    for e, j, b in sorted(set(triples)):
        lhs = sp.expand((b + 1) * run.coefs.model(j))
        rhs = sp.expand(b + Om(m) ** e)
        lines += [
            f"theorem qreq{tag}{j}e{e}b{b} : qreqHolds {m} A ({e}, {j}, {b}) := by",
            f"  have key : ({text(lhs)} : ℂ) = {text(rhs)} :=",
            *("    " + ln for ln in prove(lhs, rhs).splitlines()),
            "  simp only [qreqHolds, A]",
            "  all_goals norm_num",
            "  all_goals linear_combination key",
            "",
        ]
    cases = []
    for e, j, b in triples:
        cases += ["  rcases List.mem_cons.1 hp with rfl | hp", f"  · exact qreq{tag}{j}e{e}b{b}"]
    return [*lines, *head, *cases, "  simp at hp", ""]


def afac_lines(tag: str, run: AmpRun, sc: object) -> tuple[list[str], sp.Expr]:
    """The theorem that a run's step factors multiply to their model, and the model."""
    text = sc.text  # type: ignore[attr-defined]
    model = sp.Mul(*run.facs, sp.Integer(1))
    m = run.coefs.m
    lines = [
        f"theorem fac{tag} : afacs A steps{tag} = {text(model)} := by",
        f"  simp only [steps{tag}, afacs, AStep.fac, fac, A]",
        *(
            [f"  rw [om_sub {m} 8 {m // 8} (by norm_num) (by norm_num)]"]
            if m != 8 and run.omega
            else []
        ),
        "  all_goals ring",
        "",
    ]
    return lines, model


def initial(coefs: Coefs, path: set[int], amps: Sequence[tuple[Cyclo, F2Poly]]) -> AmpRun:
    """The run of a path sum: held the path variables an atom reads, the others inner."""
    atoms = [(coefs.index(a), f) for a, f in amps]
    read = {v for _a, f in amps for m in f for v in m}
    return AmpRun(coefs, sorted(path - read), sorted(path & read), atoms)


def atoms_text(atoms: Sequence[Atom]) -> str:
    """Atoms as a Lean list."""
    return "[" + ", ".join(f"({j}, {f2_text(f)})" for j, f in atoms) + "]"


__all__ = [
    "AmpRun",
    "Coefs",
    "afac_lines",
    "atoms_text",
    "f2_text",
    "initial",
    "qreqs_lines",
    "reduce_run",
    "reqs_lines",
]
