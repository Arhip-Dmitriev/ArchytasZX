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

"""Qubit stabilizer codes from their generators: Pauli algebra, Gottesman's standard form and
logical operators, the standard-form encoding circuit, a syndrome-and-correct circuit, and two
specifications written from the code's definition: the projector form 2^(r/2) P_S Xbar^x |b>
(P_S a product of parity gadgets) and, for CSS codes, the phase-free normal form (one Z spider
per logical or X-stabilizer row, one X spider per qubit)."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from benchmarks import ir
from benchmarks.ir import CX, CZ, MCX, MCZ, F, Init, X, Z, ZPhase
from benchmarks.subject import Domain, Equation, SpecContext, Subject
from benchmarks.subjects._gates import inverse
from benchmarks.subjects._spec import Dsl

_LETTERS = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}


@dataclass(frozen=True)
class Pauli:
    """i^p X^x Z^z, the X and Z bit vectors over the qubits."""

    x: tuple[int, ...]
    z: tuple[int, ...]
    p: int = 0

    @property
    def n(self) -> int:
        """The number of qubits."""
        return len(self.x)

    def support(self) -> list[int]:
        """The qubits acted on."""
        return [j for j in range(self.n) if self.x[j] or self.z[j]]

    def letter(self, j: int) -> str:
        """The Pauli letter at qubit j."""
        return {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(self.x[j], self.z[j])]

    def sign(self) -> int:
        """The sign s of the Hermitian operator s (x) P_j, the letters P_j in {X, Y, Z}."""
        ys = sum(1 for j in range(self.n) if self.x[j] and self.z[j])
        e = (self.p - ys) % 4
        if e not in (0, 2):
            raise ValueError("not a Hermitian Pauli operator")
        return 1 if e == 0 else -1

    def is_x_type(self) -> bool:
        """Whether the operator has no Z part."""
        return not any(self.z)

    def is_z_type(self) -> bool:
        """Whether the operator has no X part."""
        return not any(self.x)

    def __str__(self) -> str:
        return ("-" if self.sign() < 0 else "+") + "".join(self.letter(j) for j in range(self.n))


def pauli(text: str) -> Pauli:
    """The Hermitian Pauli operator written as ``[+-]P_1...P_n``."""
    sign = 0
    if text[0] in "+-":
        sign = 2 if text[0] == "-" else 0
        text = text[1:]
    x = tuple(_LETTERS[ch][0] for ch in text)
    z = tuple(_LETTERS[ch][1] for ch in text)
    return Pauli(x, z, (sign + text.count("Y")) % 4)


def mul(a: Pauli, b: Pauli) -> Pauli:
    """The product a b."""
    swap = sum(za * xb for za, xb in zip(a.z, b.x))
    x = tuple(u ^ v for u, v in zip(a.x, b.x))
    z = tuple(u ^ v for u, v in zip(a.z, b.z))
    return Pauli(x, z, (a.p + b.p + 2 * swap) % 4)


def commute(a: Pauli, b: Pauli) -> bool:
    """Whether a and b commute."""
    return sum(u * v + w * t for u, v, w, t in zip(a.x, b.z, a.z, b.x)) % 2 == 0


@dataclass(frozen=True)
class Code:
    """A code in standard form: generators, the qubit order (first r X pivots, then the m - r
    Z pivots, then the k carriers), the reference basis state b and the logical operators."""

    n: int
    gens: tuple[Pauli, ...]
    defining: tuple[Pauli, ...]
    order: tuple[int, ...]
    r: int
    ref: tuple[int, ...]
    xbar: tuple[Pauli, ...]
    zbar: tuple[Pauli, ...]

    @property
    def m(self) -> int:
        """The number of generators."""
        return len(self.gens)

    @property
    def k(self) -> int:
        """The number of logical qubits."""
        return self.n - self.m

    @property
    def carriers(self) -> tuple[int, ...]:
        """The qubit carrying each logical input."""
        return self.order[self.m :]

    def is_css(self) -> bool:
        """Whether every defining generator is X type or Z type."""
        return all(g.is_x_type() or g.is_z_type() for g in self.defining)


def standard_form(generators: Sequence[str]) -> Code:
    """Gottesman's standard form of independent commuting generators, with its logical
    operators and reference state."""
    rows = [pauli(g) for g in generators]
    defining = tuple(rows)
    n, m = rows[0].n, len(rows)
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if not commute(a, b):
                raise ValueError(f"generators {a} and {b} anticommute")
    order = list(range(n))

    def pivot(start: int, part: str) -> tuple[int, int] | None:
        for i in range(start, m):
            vec = rows[i].x if part == "x" else rows[i].z
            for pos in range(start, n):
                if vec[order[pos]]:
                    return i, pos
        return None

    def eliminate(s: int, part: str) -> None:
        q = order[s]
        for j in range(m):
            vec = rows[j].x if part == "x" else rows[j].z
            if j != s and vec[q]:
                rows[j] = mul(rows[j], rows[s])

    r = 0
    while (hit := pivot(r, "x")) is not None:
        i, pos = hit
        rows[r], rows[i] = rows[i], rows[r]
        order[r], order[pos] = order[pos], order[r]
        eliminate(r, "x")
        r += 1
    for s in range(r, m):
        hit = pivot(s, "z")
        if hit is None:
            raise ValueError("dependent generators")
        i, pos = hit
        rows[s], rows[i] = rows[i], rows[s]
        order[s], order[pos] = order[pos], order[s]
        eliminate(s, "z")
    first, mid, last = order[:r], order[r:m], order[m:]
    ref = [0] * n
    for a, q in enumerate(mid):
        if rows[r + a].sign() < 0:
            ref[q] = 1
    xbar, zbar = [], []
    for j, c in enumerate(last):
        x = [0] * n
        z = [0] * n
        x[c] = 1
        for a, q in enumerate(mid):
            x[q] = rows[r + a].z[c]
        for i, f in enumerate(first):
            v = rows[i].z[c]
            for a, q in enumerate(mid):
                v ^= rows[r + a].z[c] & rows[i].z[q]
            z[f] = v
        xbar.append(Pauli(tuple(x), tuple(z), 0))
        z2 = [0] * n
        z2[c] = 1
        for i, f in enumerate(first):
            z2[f] = rows[i].x[c]
        zbar.append(Pauli(tuple([0] * n), tuple(z2), 0))
    code = Code(n, tuple(rows), defining, tuple(order), r, tuple(ref), tuple(xbar), tuple(zbar))
    _verify(code)
    return code


def _verify(code: Code) -> None:
    for g in code.gens:
        for lo in (*code.xbar, *code.zbar):
            if not commute(g, lo):
                raise ValueError(f"logical {lo} does not commute with {g}")
    for i, a in enumerate(code.xbar):
        for j, b in enumerate(code.zbar):
            if commute(a, b) != (i != j):
                raise ValueError("logical operators are not paired")


def wires(n: int) -> list[str]:
    """The data wire names."""
    return [f"q{j}" for j in range(n)]


def encoder_gates(code: Code) -> tuple[list[ir.Gate], list[ir.Gate]]:
    """The Inits and the unitary gates of the standard-form encoder: carriers copied onto the
    Z-pivot block by Xbar, then H, the phase i^p and the controlled rest of each X-pivot
    generator."""
    q = wires(code.n)
    carriers = set(code.carriers)
    inits = [Init(q[j], code.ref[j]) for j in range(code.n) if j not in carriers]
    gates: list[ir.Gate] = []
    for lo, c in zip(code.xbar, code.carriers):
        gates += [CX(q[c], q[j]) for j in range(code.n) if lo.x[j] and j != c]
    for i in range(code.r):
        g, f = code.gens[i], code.order[i]
        gates.append(F(q[f]))
        if g.p % 4:
            gates.append(ZPhase(q[f], {1: Fraction(g.p % 4, 4)}))
        for j in range(code.n):
            if j == f:
                continue
            if g.z[j]:
                gates.append(CZ(q[f], q[j]))
            if g.x[j]:
                gates.append(CX(q[f], q[j]))
    return inits, gates


def encoder(code: Code) -> ir.Circuit:
    """The standard-form encoding circuit."""
    inits, gates = encoder_gates(code)
    return ir.circuit(wires(code.n), [*inits, *gates])


def _inputs_and_copies(code: Code, b: Dsl, D: str) -> dict[int, list[str]]:
    """Input spiders in carrier order; the ports each data qubit takes from them."""
    feeds: dict[int, list[str]] = {j: [] for j in range(code.n)}
    by_carrier = sorted(range(code.k), key=lambda j: code.carriers[j])
    for j in by_carrier:
        lo = code.xbar[j]
        supp = [t for t in range(code.n) if lo.x[t]]
        phase = (lo.p + 2 * sum(lo.z[t] * code.ref[t] for t in range(code.n))) % 4
        ph = f" phase{{1: {Fraction(phase, 4)}}}" if phase else ""
        s = b.node(f"Z(1 -> {len(supp)}, {D}){ph}")
        b.ins.append(f"{s}.in0")
        for i, t in enumerate(supp):
            feeds[t].append(f"{s}.out{i}")
    return feeds


def projector_spec(code: Code, c: SpecContext) -> str:
    """2^(r/2) P_S Xbar^x |b>: Xbar copied from the inputs onto |b>, then one parity gadget
    (I + g)/2 per defining generator."""
    D = c.D
    b = Dsl()
    for i, lo in enumerate(code.xbar):
        for lo2 in code.xbar:
            if sum(z * x for z, x in zip(lo.z, lo2.x)) % 2:
                raise ValueError("logical X operators with crossing Z parts")
    feeds = _inputs_and_copies(code, b, D)
    cur: dict[int, str] = {}
    exp = code.r
    for j in range(code.n):
        m = len(feeds[j])
        v = code.ref[j]
        ph = " phase{1: 1/2}" if v else ""
        s = b.node(f"X({m} -> 1, {D}){ph}")
        for i, port in enumerate(feeds[j]):
            b.wire(port, f"{s}.in{i}")
        cur[j] = f"{s}.out0"
        exp += m - 1
    for g in code.defining:
        supp = g.support()
        legs = []
        for j in supp:
            letter = g.letter(j)
            port = cur[j]
            if letter == "Y":
                sd = b.node(f"Z(1 -> 1, {D}) phase{{1: 3/4}}")
                b.wire(port, f"{sd}.in0")
                port = f"{sd}.out0"
            if letter in "XY":
                h = b.node(f"F({D})")
                b.wire(port, f"{h}.in0")
                port = f"{h}.out0"
            leg = b.node(f"Z(1 -> 2, {D})")
            b.wire(port, f"{leg}.in0")
            port = f"{leg}.out0"
            legs.append(f"{leg}.out1")
            if letter in "XY":
                h = b.node(f"F({D})")
                b.wire(port, f"{h}.in0")
                port = f"{h}.out0"
            if letter == "Y":
                s_ = b.node(f"Z(1 -> 1, {D}) phase{{1: 1/4}}")
                b.wire(port, f"{s_}.in0")
                port = f"{s_}.out0"
            cur[j] = port
        ph = " phase{1: 1/2}" if g.sign() < 0 else ""
        hub = b.node(f"X({len(legs)} -> 0, {D}){ph}")
        for i, port in enumerate(legs):
            b.wire(port, f"{hub}.in{i}")
        exp += len(legs) - 2
    b.outs = [cur[j] for j in range(code.n)]
    if exp:
        b.scalar.append(f"2^({exp}/2)")
    return b.text()


def css_spec(code: Code, c: SpecContext) -> str:
    """The phase-free CSS normal form as DSL text (see :func:`css_into`)."""
    b = Dsl()
    css_into(code, c, b)
    return b.text()


def css_into(code: Code, c: SpecContext, b: Dsl) -> None:
    """Adds the phase-free CSS normal form to ``b``: a Z spider per logical X row (with an
    input) and per defining X-stabilizer (a state), an X spider per qubit joined to the rows
    covering it."""
    D = c.D
    feeds = _inputs_and_copies(code, b, D)
    xs = [g for g in code.defining if g.is_x_type()]
    for g in xs:
        if g.sign() < 0:
            raise ValueError("a negative X stabilizer")
        supp = g.support()
        s = b.node(f"Z(0 -> {len(supp)}, {D})")
        for i, t in enumerate(supp):
            feeds[t].append(f"{s}.out{i}")
    exp = -len(xs)
    for j in range(code.n):
        m = len(feeds[j])
        ph = " phase{1: 1/2}" if code.ref[j] else ""
        s = b.node(f"X({m} -> 1, {D}){ph}")
        for i, port in enumerate(feeds[j]):
            b.wire(port, f"{s}.in{i}")
        b.outs.append(f"{s}.out0")
        exp += m - 1
    if exp:
        b.scalar.append(f"2^({exp}/2)")


def syndrome_gates(code: Code, anc: Sequence[str]) -> list[ir.Gate]:
    """Each defining generator measured coherently onto its ancilla: H, the controlled
    operator, H; the ancilla ends in |s> for eigenvalue (-1)^s."""
    q = wires(code.n)
    out: list[ir.Gate] = []
    for g, a in zip(code.defining, anc):
        out += [Init(a), F(a)]
        if g.p % 4:
            out.append(ZPhase(a, {1: Fraction(g.p % 4, 4)}))
        for j in range(code.n):
            if g.z[j]:
                out.append(CZ(a, q[j]))
            if g.x[j]:
                out.append(CX(a, q[j]))
        out.append(F(a))
    return out


def syndrome(code: Code, e: Pauli) -> tuple[int, ...]:
    """The syndrome bits of the error e."""
    return tuple(0 if commute(g, e) else 1 for g in code.defining)


def single_errors(n: int) -> list[tuple[int, str]]:
    """Every single-qubit Pauli error as (qubit, letter)."""
    return [(j, letter) for j in range(n) for letter in "XZY"]


def _error_pauli(n: int, j: int, letter: str) -> Pauli:
    return pauli("".join(letter if t == j else "I" for t in range(n)))


def correction_circuit(code: Code, error: tuple[int, str]) -> ir.Circuit:
    """Encode, apply the error, measure the syndrome onto ancillas, correct every single-qubit
    error by multi-controlled gates on its syndrome, decode."""
    q = wires(code.n)
    anc = [f"a{i}" for i in range(code.m)]
    inits, enc = encoder_gates(code)
    j0, l0 = error
    gates: list[ir.Gate] = [*inits, *enc]
    gates += {"X": [X(q[j0])], "Z": [Z(q[j0])], "Y": [Z(q[j0]), X(q[j0])]}[l0]
    gates += syndrome_gates(code, anc)
    seen: set[tuple[int, ...]] = set()
    for j, letter in [error, *single_errors(code.n)]:
        s = syndrome(code, _error_pauli(code.n, j, letter))
        if not any(s) or s in seen:
            continue
        seen.add(s)
        if letter in "XY":
            gates.append(MCX(tuple(anc), q[j], s))
        if letter in "ZY":
            gates.append(MCZ(tuple(anc), q[j], s))
    gates += inverse(enc)
    allw = [*q, *anc]
    return ir.circuit(allw, gates)


def correction_spec(code: Code, error: tuple[int, str], c: SpecContext) -> str:
    """The carriers unchanged, the other data qubits in |b>, the ancillas in the syndrome."""
    s = syndrome(code, _error_pauli(code.n, *error))
    b = Dsl()
    carriers = set(code.carriers)
    count = 0
    for j in range(code.n):
        if j in carriers:
            node = b.node(f"Z(1 -> 1, {c.D})")
            b.ins.append(f"{node}.in0")
        else:
            ph = " phase{1: 1/2}" if code.ref[j] else ""
            node = b.node(f"X(0 -> 1, {c.D}){ph}")
            count += 1
        b.outs.append(f"{node}.out0")
    for bit in s:
        ph = " phase{1: 1/2}" if bit else ""
        node = b.node(f"X(0 -> 1, {c.D}){ph}")
        b.outs.append(f"{node}.out0")
        count += 1
    if count:
        b.scalar.append(f"2^(-{count}/2)")
    return b.text()


def sorted_logicals(code: Code) -> list[int]:
    """The logical indices in input order (by carrier qubit)."""
    return sorted(range(code.k), key=lambda j: code.carriers[j])


def logical_phase(code: Code, turns: Sequence[Fraction]) -> Callable[[tuple[int, ...]], Fraction]:
    """The logical diagonal of the transversal phase gate P(turns[q]) per qubit on a CSS code:
    the phase of every codeword of each logical coset, checked constant over the coset."""
    xs = [g.x for g in code.defining if g.is_x_type()]
    order = sorted_logicals(code)
    table: dict[tuple[int, ...], Fraction] = {}
    for x in product((0, 1), repeat=code.k):
        base = list(code.ref)
        for i, v in enumerate(x):
            if v:
                base = [a ^ b for a, b in zip(base, code.xbar[order[i]].x)]
        seen = set()
        for sel in product((0, 1), repeat=len(xs)):
            w = list(base)
            for use, row in zip(sel, xs):
                if use:
                    w = [a ^ b for a, b in zip(w, row)]
            seen.add(sum((Fraction(t) * v for t, v in zip(turns, w)), Fraction(0)) % 1)
        if len(seen) != 1:
            raise ValueError("the transversal phase gate is not logical on this code")
        table[x] = seen.pop()
    return lambda x: table[tuple(x)]


def pre_unary(text: str) -> Callable[[Dsl, SpecContext, list[str]], list[str]]:
    """A one-leg node (``text`` formatted with D) before every input."""

    def pre(b: Dsl, c: SpecContext, ins: list[str]) -> list[str]:
        out = []
        for port in ins:
            node = b.node(text.format(D=c.D))
            b.wire(f"{node}.out0", port)
            out.append(f"{node}.in0")
        return out

    return pre


def pre_diagonal(
    f: Callable[[tuple[int, ...]], Fraction], k: int
) -> Callable[[Dsl, SpecContext, list[str]], list[str]]:
    """The diagonal e^(2 pi i f(x)) on the k qubit inputs: binders, a phase table, splitters."""

    def pre(b: Dsl, c: SpecContext, ins: list[str]) -> list[str]:
        from benchmarks.subjects._spec import bind, split

        new, port = bind(b, k, 2)
        body = ", ".join(
            f"{i}: {t}"
            for i, x in enumerate(product((0, 1), repeat=k))
            if (t := Fraction(f(x)) % 1)
        )
        z = b.node(f"Z(1 -> 1, literal {2**k})" + (f" phase{{{body}}}" if body else ""))
        b.wire(port, f"{z}.in0")
        for o, i in zip(split(b, f"{z}.out0", k, 2), ins):
            b.wire(o, i)
        return new

    return pre


@dataclass(frozen=True)
class Logical:
    """A logical-gate equation: the physical gates after the encoder, the logical gate written
    before the specification's inputs."""

    key: str
    physical: Callable[[Code], list[ir.Gate]]
    pre: Callable[[Code], Callable[[Dsl, SpecContext, list[str]], list[str]]]


def logical_spec(code: Code, c: SpecContext, lo: Logical) -> str:
    """The logical gate composed with the encoding specification."""
    if not code.is_css():
        raise ValueError("logical equations need a CSS code")
    b = Dsl()
    css_into(code, c, b)
    b.ins = lo.pre(code)(b, c, b.ins)
    return b.text()


def code_subject(
    id: str,
    name: str,
    source: str,
    generators: Sequence[str],
    *,
    correct: tuple[int, str] | None = None,
    logical: Sequence[Logical] = (),
    note: str = "",
) -> Subject:
    """A fixed-size qubit code subject: the encoding equation (CSS normal form for a CSS
    code, the projector form otherwise), the correction equation when ``correct`` names an
    error, and one equation per logical gate."""
    code = standard_form(generators)
    spec = css_spec if code.is_css() else projector_spec
    eqs = [
        Equation(
            "encoding",
            lambda n, d: encoder(code),
            lambda c: spec(code, c),
            form="fixed",
        )
    ]
    if correct is not None:
        eqs.append(
            Equation(
                "correction",
                lambda n, d: correction_circuit(code, correct),
                lambda c: correction_spec(code, correct, c),
                form="fixed",
                literal=True,
            )
        )
    for lo in logical:
        eqs.append(
            Equation(
                lo.key,
                lambda n, d, lo=lo: ir.circuit(
                    wires(code.n),
                    [*encoder_gates(code)[0], *encoder_gates(code)[1], *lo.physical(code)],
                ),
                lambda c, lo=lo: logical_spec(code, c, lo),
                form="fixed",
                literal=True,
            )
        )
    return Subject(
        id=id,
        name=name,
        category="code",
        source=source,
        size_parameter=None,
        dimension=2,
        domain=Domain(d_min=2, d_max=2),
        equations=tuple(eqs),
        clifford=True,
        note=note,
    )


__all__ = [
    "Code",
    "Logical",
    "Pauli",
    "code_family",
    "code_subject",
    "commute",
    "correction_circuit",
    "correction_spec",
    "css_into",
    "css_spec",
    "encoder",
    "logical_phase",
    "mul",
    "pauli",
    "pre_diagonal",
    "pre_unary",
    "projector_spec",
    "sorted_logicals",
    "standard_form",
]


def css_generators(xrows: Sequence[Sequence[int]], zrows: Sequence[Sequence[int]]) -> list[str]:
    """Generator strings from the 0/1 supports of X-type and Z-type checks."""
    out = ["".join("X" if v else "I" for v in row) for row in xrows]
    out += ["".join("Z" if v else "I" for v in row) for row in zrows]
    return out


def support_rows(n: int, supports: Sequence[Sequence[int]]) -> list[list[int]]:
    """0/1 rows of length n from lists of qubit indices."""
    return [[1 if j in set(s) else 0 for j in range(n)] for s in supports]


def hadamard_conjugate(gens: Sequence[str], qubits: Sequence[int]) -> list[str]:
    """The generators with X and Z exchanged on ``qubits``."""
    sw = {"X": "Z", "Z": "X", "Y": "Y", "I": "I"}
    out = []
    for g in gens:
        sign = g[0] if g[0] in "+-" else ""
        body = g[1:] if sign else g
        out.append(sign + "".join(sw[ch] if j in set(qubits) else ch for j, ch in enumerate(body)))
    return out


@functools.cache
def _family_code(make: Callable[[int], tuple[str, ...]], n: int) -> Code:
    return standard_form(make(n))


def code_family(
    id: str,
    name: str,
    source: str,
    make: Callable[[int], tuple[str, ...]],
    size_parameter: str,
    domain: Domain,
    note: str = "",
) -> Subject:
    """A qubit code family indexed by n (``make(n)`` its generators): the encoding equation at
    every n, one wire per qubit."""

    def build(n: int, d: int) -> ir.Circuit:
        return encoder(_family_code(make, n))

    def spec(c: SpecContext) -> str:
        code = _family_code(make, c.nv)
        return (css_spec if code.is_css() else projector_spec)(code, c)

    return Subject(
        id=id,
        name=name,
        category="code",
        source=source,
        size_parameter=size_parameter,
        dimension=2,
        domain=domain,
        equations=(Equation("encoding", build, spec, form="unrolled"),),
        clifford=True,
        note=note,
    )
