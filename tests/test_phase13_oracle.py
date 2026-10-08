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


"""Phase 13 done-when: equal diagrams share a normal form, unequal ones do not, and every
decision is cross-checked against the numeric oracle."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.normal_form import normal_form, same_normal_form
from archytaszx.rewrite.rules_library import BIALGEBRA
from archytaszx.semantics.decide import DecisionMethod, EqualityVerdict, decide_equal

from . import test_decide as TD
from . import test_phase8_oracle as T8
from .test_rules_library_phase11 import (
    bialgebra_diagram,
    fourier_state_diagram,
    hopf_diagram,
    identity_chain,
    state_copy_right_hand_side,
    triangle_chain,
)

D = Dim("d")
EQUAL = EqualityVerdict.EQUAL
UNEQUAL = EqualityVerdict.UNEQUAL
UNKNOWN = EqualityVerdict.UNKNOWN
NF = DecisionMethod.NORMAL_FORM
ORACLE = DecisionMethod.ORACLE_COUNTEREXAMPLE

Builder = Callable[[], TD.Pair]

TABLE: list[tuple[str, Builder, dict[str, object], EqualityVerdict, DecisionMethod]] = [
    ("hopf_self", TD.NF_EQUAL["hopf_self"], {}, EQUAL, NF),
    ("identity_z_vs_x", TD.NF_EQUAL["identity_z_vs_x"], {}, EQUAL, NF),
    ("triangle_orders", TD.NF_EQUAL["triangle_orders"], {}, EQUAL, NF),
    ("fourier_rebuilt", TD.NF_EQUAL["fourier_rebuilt"], {}, EQUAL, NF),
    ("ghz_fused", TD.NF_EQUAL["ghz_fused"], {}, EQUAL, NF),
    ("state_copy_rewrite", TD.NF_EQUAL["state_copy_rewrite"], {}, EQUAL, NF),
    ("fusion_node_orders", TD.NF_EQUAL["fusion_node_orders"], {}, EQUAL, NF),
    ("boxed_fusion_nf", T8._build_boxed_fusion_family, {}, EQUAL, NF),
    ("two_index_fusion_nf", T8._build_two_index_fusion_family, {}, EQUAL, NF),
    ("arity", TD.UNEQUAL_INTERFACE["arity"], {}, UNEQUAL, DecisionMethod.INTERFACE),
    ("dimension", TD.UNEQUAL_INTERFACE["dimension"], {}, UNEQUAL, DecisionMethod.INTERFACE),
    ("false_near_identity", T8._build_false_near_identity, {}, UNEQUAL, ORACLE),
    ("different_phases", TD.UNEQUAL_ORACLE["different_phases"], {}, UNEQUAL, ORACLE),
    ("extra_scalar", TD.UNEQUAL_ORACLE["extra_scalar"], {}, UNEQUAL, ORACLE),
    ("z_cup_vs_x_cup", TD.UNEQUAL_ORACLE["z_cup_vs_x_cup"], {}, UNEQUAL, ORACLE),
    (
        "zero_phase_vector",
        TD.zero_phase_pair,
        {},
        EQUAL,
        DecisionMethod.SYMBOLIC_CONTRACTION,
    ),
    (
        "full_turn_phase",
        TD.SYMBOLIC_EQUAL["full_turn_phase"],
        {},
        EQUAL,
        DecisionMethod.SYMBOLIC_CONTRACTION,
    ),
    ("bialgebra_saturation", TD.bialgebra_pair, {}, EQUAL, DecisionMethod.SATURATION),
    (
        "boxed_fusion_induction",
        T8._build_boxed_fusion_family,
        {"guard": TD.NO_FUSION, "use_saturation": False, "use_symbolic": False},
        EQUAL,
        DecisionMethod.INDUCTION,
    ),
    (
        "boxed_fusion_symbolic",
        T8._build_boxed_fusion_family,
        {"guard": TD.NO_FUSION, "use_saturation": False},
        EQUAL,
        DecisionMethod.SYMBOLIC_CONTRACTION,
    ),
    (
        "two_index_fusion_induction",
        T8._build_two_index_fusion_family,
        {"guard": TD.NO_FUSION, "use_saturation": False},
        EQUAL,
        DecisionMethod.INDUCTION,
    ),
    (
        "boxed_fusion_induction_swapped",
        lambda: T8._build_boxed_fusion_family()[1::-1],
        {"guard": TD.NO_FUSION, "use_saturation": False, "use_symbolic": False},
        EQUAL,
        DecisionMethod.INDUCTION,
    ),
    (
        "base_case_refutation",
        TD.concrete_false_near_identity,
        {"samples": [{}]},
        UNEQUAL,
        ORACLE,
    ),
    (
        "refused_ghz_phase",
        lambda: TD.wide_ghz_phase_pair(4),
        {"max_elements": 1},
        UNEQUAL,
        DecisionMethod.SYMBOLIC_WITNESS,
    ),
    (
        "zero_phase_rungs_disabled",
        TD.zero_phase_pair,
        {"use_symbolic": False, "use_induction": False},
        UNKNOWN,
        DecisionMethod.NONE,
    ),
]

KNOWN_EQUAL = [row[0] for row in TABLE if row[4] is NF]
KNOWN_UNEQUAL = [row[0] for row in TABLE if row[4] is ORACLE]
BY_NAME = {row[0]: row for row in TABLE}


def concrete_pairs(d: int) -> dict[str, tuple[TD.Pair, EqualityVerdict, DecisionMethod]]:
    """Every concrete-dimension pair at dimension ``d``, with its expected verdict and rung.

    At ``d = 2`` the Z and X cups coincide and the symbolic rung proves them equal.
    """
    dim = Dim.concrete(d)
    other = Dim.concrete(d + 1)
    phase_a = TD.phase_turns(dim, sp.Rational(1, 3))
    phase_b = TD.phase_turns(dim, sp.Rational(1, 4))
    return {
        "hopf_self": ((hopf_diagram(d, 1, 1, 1, 1), hopf_diagram(d, 1, 1, 1, 1)), EQUAL, NF),
        "hopf_phased": (
            (hopf_diagram(d, 2, 1, 1, 2, phases=True), hopf_diagram(d, 2, 1, 1, 2, phases=True)),
            EQUAL,
            NF,
        ),
        "identity_z_vs_x": ((identity_chain(d), identity_chain(d, X_SPIDER)), EQUAL, NF),
        "triangle_orders": (
            (triangle_chain(d), triangle_chain(d, inverse_first=True)),
            EQUAL,
            NF,
        ),
        "fourier_effect": (
            (fourier_state_diagram(d, is_state=False), fourier_state_diagram(d, is_state=False)),
            EQUAL,
            NF,
        ),
        "ghz_fused": (TD.ghz_pair(dim), EQUAL, NF),
        "state_copy_rewrite": (TD.state_copy_pair(d, 3), EQUAL, NF),
        "fusion_node_orders": (
            (TD.fusion_chain("sme", dim), TD.fusion_chain("mes", dim)),
            EQUAL,
            NF,
        ),
        "arity": (
            (TD.ghz_spider(dim), state_copy_right_hand_side(d, 2)),
            UNEQUAL,
            DecisionMethod.INTERFACE,
        ),
        "dimension": ((TD.z_state(dim), TD.z_state(other)), UNEQUAL, DecisionMethod.INTERFACE),
        "different_phases": (
            (TD.z_state(dim, phase_a), TD.z_state(dim, phase_b)),
            UNEQUAL,
            ORACLE,
        ),
        "extra_scalar": (
            (TD.ghz_spider(dim), TD.scaled(TD.ghz_spider(dim), Scalar.rational(2))),
            UNEQUAL,
            ORACLE,
        ),
        "z_cup_vs_x_cup": (
            (TD.cup(Z_SPIDER, dim), TD.cup(X_SPIDER, dim)),
            *((EQUAL, DecisionMethod.SYMBOLIC_CONTRACTION) if d == 2 else (UNEQUAL, ORACLE)),
        ),
        "zero_phase_vector": (
            TD.zero_phase_pair(dim),
            EQUAL,
            DecisionMethod.SYMBOLIC_CONTRACTION,
        ),
        "self_loop": (
            (TD.self_loop(dim), TD.z_state(dim)),
            EQUAL,
            DecisionMethod.SYMBOLIC_CONTRACTION,
        ),
    }


def check(
    left_right: TD.Pair,
    options: dict[str, object],
    verdict: EqualityVerdict,
    method: DecisionMethod,
) -> None:
    """Decide the pair, assert the verdict and rung, and cross-check against the oracle."""
    left, right = left_right
    decision = decide_equal(left, right, **options)  # type: ignore[arg-type]
    assert decision.verdict is verdict, decision.reason
    assert decision.method is method, decision.reason
    TD.assert_oracle_agrees(left, right, decision)
    if verdict is UNKNOWN:
        assert all(matched for _, matched in TD.oracle_samples(left, right))
    if method is not DecisionMethod.INTERFACE:
        TD.assert_certificates_verify(decision)


class TestDoneWhen:
    """The phase's three done-when clauses."""

    @pytest.mark.parametrize("name", KNOWN_EQUAL)
    def test_known_equal_diagrams_share_a_normal_form(self, name: str) -> None:
        left, right = BY_NAME[name][1]()
        nf_l, nf_r = normal_form(left), normal_form(right)
        assert nf_l.reached_fixpoint and nf_r.reached_fixpoint
        assert nf_l.key == nf_r.key
        assert same_normal_form(nf_l, nf_r)

    @pytest.mark.parametrize("name", KNOWN_UNEQUAL)
    def test_known_unequal_diagrams_have_distinct_normal_forms(self, name: str) -> None:
        left, right = BY_NAME[name][1]()
        nf_l, nf_r = normal_form(left), normal_form(right)
        assert nf_l.key != nf_r.key
        assert not same_normal_form(nf_l, nf_r)

    @pytest.mark.parametrize(
        ("name", "builder", "options", "verdict", "method"),
        TABLE,
        ids=[row[0] for row in TABLE],
    )
    def test_every_decision_agrees_with_the_oracle(
        self,
        name: str,
        builder: Builder,
        options: dict[str, object],
        verdict: EqualityVerdict,
        method: DecisionMethod,
    ) -> None:
        check(builder(), options, verdict, method)

    def test_the_table_covers_every_rung(self) -> None:
        assert {row[4] for row in TABLE} == set(DecisionMethod)


@pytest.mark.slow
class TestConcreteDimensionSweep:
    """The concrete pairs re-decided at several dimensions."""

    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("name", sorted(concrete_pairs(2)))
    def test_pair_at_dimension(self, name: str, d: int) -> None:
        pair, verdict, method = concrete_pairs(d)[name]
        check(pair, {}, verdict, method)

    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_state_copy_leg_counts(self, d: int, n: int) -> None:
        check(TD.state_copy_pair(d, n), {}, EQUAL, NF)

    def test_bialgebra_rewrite_is_proved_by_symbolic_contraction(self) -> None:
        left = bialgebra_diagram(2)
        right = apply(left, BIALGEBRA, BIALGEBRA.pattern.find_matches(left)[0]).diagram
        decision = decide_equal(left, right, use_saturation=False)
        assert decision.verdict is EQUAL, decision.reason
        assert decision.method is DecisionMethod.SYMBOLIC_CONTRACTION
        assert decision.samples_checked == 1
        assert decision.assumptions == ()
        TD.assert_oracle_agrees(left, right, decision)
