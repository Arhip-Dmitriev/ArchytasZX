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

"""Entanglement-assisted [[4,1,3;1]] code: the five-qubit code with one qubit held noiselessly by
Bob. Alice holds the message, two |0> ancillas and her half of a shared Bell pair, and encodes by a
Clifford on her four qubits alone, synthesized from the images Z_anc -> stabilizers trivial on
Bob, (X_e, Z_e) -> the stabilizers with X and Z on Bob, (X_m, Z_m) -> the logical operators made
trivial on Bob; all restricted to Alice. The specification is the projector form of the
five-qubit code."""

from itertools import product

from benchmarks.ir import CX, F, Init, circuit
from benchmarks.subject import Domain, Equation, Subject
from benchmarks.subjects._clifford import ir_gates, restrict, synthesize, trivial_on
from benchmarks.subjects._stab import Pauli, commute, mul, projector_spec, standard_form, wires

_CODE = standard_form(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
_BOB = _CODE.order[0]
_ALICE = [j for j in range(5) if j != _BOB]
_MSG = _CODE.carriers[0]


def _group():
    """Every element of the stabilizer group."""
    out = []
    for c in product((0, 1), repeat=4):
        g = Pauli((0,) * 5, (0,) * 5, 0)
        for gi, ci in zip(_CODE.defining, c):
            if ci:
                g = mul(g, gi)
        out.append(g)
    return out


def _with_bob(letter):
    """A stabilizer whose Bob factor is ``letter``."""
    want = {"I": (0, 0), "X": (1, 0), "Z": (0, 1)}[letter]
    return next(g for g in _group() if (g.x[_BOB], g.z[_BOB]) == want)


def _clear_bob(lo):
    """lo times the stabilizer making its Bob factor the identity."""
    return next(mul(lo, g) for g in _group() if not mul(lo, g).x[_BOB] and not mul(lo, g).z[_BOB])


def _on_alice(ps):
    return restrict(ps, _ALICE)


def _images():
    """(X, Z) images per Alice qubit in Alice order, and the ebit and ancilla qubits."""
    iso = _on_alice(trivial_on(list(_CODE.defining), [_BOB]))
    sx, sz = _on_alice([_with_bob("X"), _with_bob("Z")])
    xm, zm = _on_alice([_clear_bob(_CODE.xbar[0]), _clear_bob(_CODE.zbar[0])])
    rest = [j for j in _ALICE if j != _MSG]
    e, anc = rest[0], rest[1:]
    fixed = [sx, sz, xm, zm]
    dest = []
    for i in range(2):
        for xs, zs in product(product((0, 1), repeat=4), repeat=2):
            d = Pauli(xs, zs, sum(a & b for a, b in zip(xs, zs)) % 4)
            if commute(d, iso[i]) or not commute(d, iso[1 - i]):
                continue
            if all(commute(d, f) for f in fixed + dest):
                dest.append(d)
                break
    pos = {q: i for i, q in enumerate(_ALICE)}
    img = [None] * 4
    img[pos[_MSG]] = (xm, zm)
    img[pos[e]] = (sx, sz)
    for i, a in enumerate(anc):
        img[pos[a]] = (dest[i], iso[i])
    return img, e, anc


_IMG, _E, _ANC = _images()
_Q = wires(5)


def _circuit(n, d):
    """Ancillas |0>, a Bell pair between Alice's e and Bob, Alice's Clifford, the phase -i."""
    gates = [Init(_Q[a]) for a in _ANC]
    gates += [Init(_Q[_E]), Init(_Q[_BOB]), F(_Q[_E]), CX(_Q[_E], _Q[_BOB])]
    gates += ir_gates(synthesize(_IMG), [_Q[j] for j in _ALICE])
    return circuit(_Q, gates, scalar="-I")


SUBJECT = Subject(
    id="ea_code",
    name="Entanglement-assisted [[4,1,3;1]] code",
    category="code",
    source="brun2006correcting",
    size_parameter=None,
    dimension=2,
    domain=Domain(d_min=2, d_max=2),
    equations=(Equation("encoding", _circuit, lambda c: projector_spec(_CODE, c), form="fixed"),),
)
