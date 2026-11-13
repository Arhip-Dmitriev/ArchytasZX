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

"""Kernel induction over a tree box against a leg port-boxed 2^n times."""

from __future__ import annotations

from archytaszx.repl.parser import parse_dsl
from archytaszx.semantics import kernel as K
from archytaszx.semantics.decide import EqualityVerdict, decide_equal
from archytaszx.semantics.prove import induction_successor

TREE = "\n".join(  # noqa: FLY002
    [
        "dim d",
        "count n",
        "g0 = X(0 -> 1, d)",
        "g1 = F(d)",
        "g2 = X(0 -> 1, d)",
        "g3 = Z(1 -> 2, d)",
        "g4 = X(2 -> 1, d)",
        "wire g0.out0 -- g1.in0, g1.out0 -- g3.in0, g2.out0 -- g4.in0, g3.out1 -- g4.in1",
        "repeat n as C0: g2, g3, g4",
        "tree C0: g3.in0 <- g3.out0, g4.out0",
        "outputs g3.out0, g4.out0",
        "scalar 1/sqrt(d)",
    ]
)
"""The doubling GHZ tree of depth n (ghz_log_depth's circuit)."""

SPEC = "dim d\ncount n\nz = Z(0 -> 1, d)\nrepeat 2^n: z.out0\nscalar d^(-1/2)"
"""Z(0 -> 2^n)."""


def _windows(t: K.Theorem) -> list[str]:
    own = [str(t.inference.data["window"])] if t.inference.kind == "induction" else []
    return own + [w for p in t.inference.premises for w in _windows(p)]


def test_a_power_multiplicity_successor_is_unpeeled() -> None:
    spec = parse_dsl(SPEC)
    succ = induction_successor(spec, "n", "k")
    assert succ is not None and frozenset(succ.nodes) == frozenset(spec.nodes)


def test_the_tree_window_gives_a_kernel_theorem() -> None:
    decision = decide_equal(parse_dsl(TREE), parse_dsl(SPEC), use_symbolic=False)
    assert decision.verdict is EqualityVerdict.EQUAL
    assert isinstance(decision.proof, K.Theorem) and "tree" in _windows(decision.proof)
    broken = SPEC.replace("scalar d^(-1/2)", "scalar d^(-1)")
    verdict = decide_equal(parse_dsl(TREE), parse_dsl(broken), use_symbolic=False).verdict
    assert verdict is EqualityVerdict.UNEQUAL
