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

"""Phase 2 numeric-oracle stand-in.

Per the spec, every build phase ends with a numeric oracle check. The real semantics
oracle arrives in Phase 4; here we instantiate all symbols to concrete values, evaluate
through the gated numeric methods (Phase.to_complex / Scalar.to_complex), and confirm
against cmath for representative cases. No simplifier is invoked -- the character-sum
identity is Phase 9's job, so the d=0..d-1 sum below is computed by direct cmath
summation, not by any method on Scalar or Phase.
"""

import cmath

import sympy as sp  # type: ignore[import-untyped]

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase
from qufzx.algebra.scalar import Scalar


class TestPhaseAdditionHomomorphism:
    def test_several_symbolic_phase_pairs(self) -> None:
        cases = [
            (sp.Rational(1, 3), sp.Rational(1, 5)),
            (sp.Rational(2, 7), sp.Rational(3, 7)),
            (sp.Rational(-1, 4), sp.Rational(5, 8)),
        ]
        for turns_a, turns_b in cases:
            a, b = Phase.turns(turns_a), Phase.turns(turns_b)
            lhs = (a + b).to_complex()
            rhs = a.to_complex() * b.to_complex()
            assert abs(lhs - rhs) < 1e-12

    def test_symbolic_expressions_instantiated(self) -> None:
        alpha, beta = Phase.symbol("alpha"), Phase.symbol("beta")
        fused = alpha + beta
        instantiated = fused.substitute({"alpha": sp.Rational(1, 6), "beta": sp.Rational(1, 9)})
        lhs = instantiated.to_complex()
        rhs = cmath.exp(2j * cmath.pi * (1 / 6 + 1 / 9))
        assert abs(lhs - rhs) < 1e-9


class TestRootOfUnityOracle:
    def test_omega_d_to_d_is_one(self) -> None:
        for d in (2, 3, 4, 5, 7):
            s = Scalar.omega(Dim.concrete(d), d)
            assert abs(s.to_complex() - 1.0) < 1e-12

    def test_character_sum_at_j_zero_by_direct_cmath(self) -> None:
        for d in (2, 3, 4, 5, 7):
            total = sum(Scalar.omega(Dim.concrete(d), k).to_complex() for k in range(d))
            assert abs(total - 0.0) < 1e-9

    def test_character_sum_at_j_nonzero_multiple_of_d_via_substitution(self) -> None:
        d_sym = Dim.symbol("d")
        j_sym = sp.Symbol("j", integer=True)
        s = Scalar.omega(d_sym, j_sym)
        for d in (2, 3, 4, 5, 7):
            total = sum(s.substitute({"d": d, "j": k}).to_complex() for k in range(d))
            assert abs(total - 0.0) < 1e-9


class TestScalarArithmeticOracle:
    def test_product_and_sum_agree_with_complex(self) -> None:
        d = Dim.concrete(6)
        a = Scalar.omega(d, 2) * Scalar.rational(1, 2)
        b = Scalar.gaussian_rational(1, -1) + Scalar.omega(d, 1)
        prod_lhs = (a * b).to_complex()
        prod_rhs = a.to_complex() * b.to_complex()
        sum_lhs = (a + b).to_complex()
        sum_rhs = a.to_complex() + b.to_complex()
        assert abs(prod_lhs - prod_rhs) < 1e-9
        assert abs(sum_lhs - sum_rhs) < 1e-9

    def test_symbolic_scalar_instantiated_matches_cmath(self) -> None:
        d_sym = Dim.symbol("d")
        s = Scalar.omega(d_sym, 3) * Scalar.symbol("k")
        instantiated = s.substitute({"d": 8, "k": 2})
        got = instantiated.to_complex()
        expected = cmath.exp(2j * cmath.pi * 3 / 8) * 2
        assert abs(got - expected) < 1e-9
