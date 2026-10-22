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

"""Tests for :mod:`archytaszx.semantics.pathsum`, checked against dense numpy vectors."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from archytaszx.semantics import pathsum as P

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def dense(s: P.State) -> np.ndarray:
    keys, amps = P.enumerate_state(s)
    out = np.zeros(2 ** int(s.total_width), dtype=complex)
    np.add.at(out, keys.astype(np.int64), amps)
    return out


HADAMARD = P.Gate(
    "H",
    1,
    lambda vals, ws, ds: P.State(
        (sympy.Integer(1),),
        (
            P.Term(
                (-1) ** (vals[0] * (b := P.new_bound("y", 0, 1)).symbol) / sympy.sqrt(2),
                (b,),
                (b.symbol,),
            ),
        ),
    ),
    qubit=True,
)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_hadamard_power_matches_kron(n: int) -> None:
    zero = P.basis(0, sympy.Integer(n))
    got = dense(P.apply(P.Power(HADAMARD, sympy.Integer(n)), zero))
    want = np.zeros(2**n)
    want[0] = 1
    k = H
    for _ in range(n - 1):
        k = np.kron(k, H)
    assert np.allclose(got, k @ want)


@pytest.mark.parametrize("value", [0, 1, 5, 7])
def test_hadamard_power_on_basis(value: int) -> None:
    s = P.apply(P.Power(HADAMARD, sympy.Integer(3)), P.basis(value, sympy.Integer(3)))
    k = np.kron(np.kron(H, H), H)
    assert np.allclose(dense(s), k[:, value])


def test_symbolic_power_then_instantiate() -> None:
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    s = P.apply(P.Power(HADAMARD, n), P.basis(0, n))
    concrete = P.State(
        tuple(w.subs(n, 3) for w in s.widths),
        tuple(
            P.Term(
                t.amp.subs(n, 3),
                tuple(P.Bound(b.symbol, b.lo, b.hi.subs(n, 3)) for b in t.bound),
                t.kets,
            )
            for t in s.terms
        ),
    )
    assert np.allclose(dense(concrete), np.full(8, 1 / np.sqrt(8)))


def test_tensor_and_split_alignment() -> None:
    s = P.tensor(P.basis(0, sympy.Integer(2)), P.basis(1, sympy.Integer(1)))
    op = P.Tensor((HADAMARD, P.Identity()))
    out = P.apply(op, s)
    assert out.widths == (1, 1, 1)
    want = np.kron(np.kron(H @ [1, 0], [1, 0]), [0, 1])
    assert np.allclose(dense(out), want)


def test_power_state_plus() -> None:
    plus = P.scale(1 / sympy.sqrt(2), P.add(P.basis(0, 1), P.basis(1, 1)))
    s = P.power_state(plus, sympy.Integer(3))
    assert np.allclose(dense(s), np.full(8, 1 / np.sqrt(8)))


def test_inner_symbolic_and_numeric() -> None:
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    s = P.apply(P.Power(HADAMARD, n), P.basis(0, n))
    assert P.inner(P.dagger(s), s) == 1
    zero = P.basis(0, n)
    assert sympy.simplify(P.inner(P.dagger(zero), s) - 2 ** (-n / 2)) == 0
    c = P.apply(P.Power(HADAMARD, sympy.Integer(2)), P.basis(0, sympy.Integer(2)))
    assert P.inner(P.dagger(c), c) == 1


def test_measurement_distribution_is_derived() -> None:
    s = P.apply(P.Power(HADAMARD, sympy.Integer(2)), P.basis(0, sympy.Integer(2)))
    (branch,) = P.distribution(s).branches
    (k,) = branch.free
    assert branch.value == k.symbol and branch.prob == sympy.Rational(1, 4)
    assert (k.lo, k.hi) == (0, 3)


def test_symbolic_measurement_pins_dependent_registers() -> None:
    n, m = sympy.symbols("n m", integer=True, nonnegative=True)
    b = P.new_bound("x", 0, 2**n - 1)
    f = sympy.Function("f")
    s = P.State((n, m), (P.Term(2 ** (-n / 2), (b,), (b.symbol, f(b.symbol))),))
    d = P.distribution(s)
    (branch,) = d.branches
    (k,) = branch.free
    assert branch.registers == (k.symbol, f(k.symbol))
    assert P.same(branch.prob, 2**-n)
    mixed = P.collapse(d, s)
    again = P.distribution(mixed)
    (other,) = again.branches
    assert P.same(other.prob, 2**-n) and len(other.free) == 1


def test_measure_mixture_is_stable_under_remeasurement() -> None:
    s = P.add(P.basis(0, sympy.Integer(1)), P.scale(sympy.I, P.basis(1, sympy.Integer(1))))
    d = P.distribution(s)
    m = P.collapse(d, s)
    for dist in (P.distribution(m), P.distribution(P.collapse(P.distribution(m), m.layout()))):
        (branch,) = dist.branches
        (k,) = branch.free
        assert branch.value == k.symbol and branch.prob == sympy.Rational(1, 2)
    assert P.same(P.trace(m), 1)


def test_integer_functions() -> None:
    x = sympy.Symbol("x", integer=True)
    assert P.xor(5, 3) == 6 and P.xor(0, x) == x and P.xor(x, x) == 0
    assert P.pop(7) == 3 and P.bitdot(6, 3) == 1 and P.bitdot(0, x) == 0
    assert (
        P.bits(0b1011, 1, 3, 4) == 0b01
        and P.shr(12, 2, 2) == 3
        and P.dsum(5, 3) == 3
        and P.ddot(4, 5, 3) == 3
    )
    assert P.powmod(7, 4, 15) == 1


def test_oracle_must_be_reversible() -> None:
    bad = P.Gate("B", 1, lambda vals, ws, ds: P.basis(0, ws[0]), oracle=True)
    with pytest.raises(P.PathSumError, match="not reversible"):
        P.apply(bad, P.basis(1, sympy.Integer(2)))


def test_errors() -> None:
    with pytest.raises(P.PathSumError):
        P.tensor(P.basis(0, 1), P.dagger(P.basis(0, 1)))
    with pytest.raises(P.PathSumError):
        P.apply(HADAMARD, P.dagger(P.basis(0, 1)))
    with pytest.raises(P.PathSumError, match="single qubits"):
        P.apply(HADAMARD, P.basis(0, sympy.Integer(2)))
