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

"""The induction hypothesis as a rule: L(k) matched inside a peeled successor it is a proper
part of, replaced by R(k)."""

from __future__ import annotations

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import induction
from archytaszx.semantics.check import compare

PAIRS = (
    "dim d\ncount n\ng0 = Z(1 -> 1, d)\ng1 = Z(1 -> 1, d)\nwire g0.out0 -- g1.in0\n"
    "repeat n as C: g0, g1\nchain C: g1.out0 -> g0.in0\ninputs g0.in0\noutputs g1.out0"
)
"""A chain of n copies of two phaseless Z spiders in a row."""

SINGLES = (
    "dim d\ncount n\ng = Z(1 -> 1, d){phase}\nrepeat n as C: g\nchain C: g.out0 -> g.in0\n"
    "inputs g.in0\noutputs g.out0"
)
"""A chain of n copies of one Z spider."""


def _obligation(phase: str = "") -> induction.InductionObligation:
    return induction.build_obligation(
        parse_dsl(PAIRS), parse_dsl(SINGLES.format(phase=phase)), witness={"d": 2}
    )


def test_the_hypothesis_is_a_rule() -> None:
    assert induction.hypothesis_rule(_obligation()) is not None


def test_a_chain_step_is_settled_through_the_hypothesis() -> None:
    obligation = _obligation()
    assert not induction.discharge_peeled_hypothesis(obligation).settled
    outcome = induction.discharge_hypothesis_rewrite(obligation)
    assert outcome.settled, outcome.reason
    assert any(step.rule_name == induction.HYPOTHESIS_RULE_NAME for step in outcome.steps)


def _headed(text: str) -> str:
    """``text`` with an X spider ahead of the chain's input."""
    lines = text.split("\n")
    port = lines[-2].removeprefix("inputs ")
    return "\n".join(
        [*lines[:-2], "h = X(1 -> 1, d)", f"wire h.out0 -- {port}", "inputs h.in0", lines[-1]]
    )


def test_a_headed_chain_step_is_settled_through_the_hypothesis() -> None:
    obligation = induction.build_obligation(
        parse_dsl(_headed(PAIRS)), parse_dsl(_headed(SINGLES.format(phase=""))), witness={"d": 2}
    )
    outcome = induction.discharge_hypothesis_rewrite(obligation)
    assert outcome.settled, outcome.reason
    assert any(step.rule_name == induction.HYPOTHESIS_RULE_NAME for step in outcome.steps)


def test_a_hypothesis_rule_step_certifies() -> None:
    from archytaszx.semantics import kernel
    from archytaszx.semantics.decide import decide_equal
    from archytaszx.semantics.induction_kernel import induction_theorem
    from archytaszx.semantics.prove import induction_base

    for text_l, text_r in (
        (PAIRS, SINGLES.format(phase="")),
        (_headed(PAIRS), _headed(SINGLES.format(phase=""))),
    ):
        left, right = parse_dsl(text_l), parse_dsl(text_r)
        result = induction.prove_by_induction(
            left, right, witness={"d": 2}, ladder=(induction.StepDischarge.HYPOTHESIS_REWRITE,)
        )
        assert result.verdict is induction.Verdict.PROVED_INDUCTION
        pair = (induction_base(left, result.index), induction_base(right, result.index))
        theorem, why = induction_theorem(left, right, result, False, pair, decide_equal(*pair))
        assert isinstance(theorem, kernel.Theorem), why
        assert theorem.inference.kind == "induction"
        assert kernel.lean(theorem, semantic=False)


def test_a_phase_difference_is_not_settled() -> None:
    obligation = _obligation(" phase {1: 1/2}")
    assert not compare(obligation.left_at_k, obligation.right_at_k, {"d": 2, "k": 1}).matched
    assert not induction.discharge_hypothesis_rewrite(obligation).settled


def test_the_hypothesis_does_not_match_the_unpeeled_goal() -> None:
    obligation = _obligation()
    rule = induction.hypothesis_rule(obligation)
    assert rule is not None
    assert rule.pattern.find_matches(obligation.left_at_k)
    for match in rule.pattern.find_matches(obligation.left_at_successor):
        assert getattr(match, "peels", 0) >= 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
