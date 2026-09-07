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

"""Standalone driver for ``TestGroupD_Determinism`` (see ``test_phase8_oracle.py``).

Not a pytest module itself -- run as a plain script, once per ``PYTHONHASHSEED`` value, via
``subprocess``. Builds the false near-identity pair (the bang-boxed GHZ family against the
boxed product family), refutes it with :func:`prove_by_induction`, and prints a stable
serialization of every field PHASE8_SPEC.md section 9.2 clause 5 promises deterministic:
``verdict``, ``index``, ``base_value``, ``counterexample``, ``reason``, the failing
``ComparisonResult``, the witness in force, ``held_symbolic``, whether a certificate was
emitted, and every ``TierOutcome``.

Every set is ``sorted`` and every enum is printed through its ``.value`` string before it
reaches the output. This script prints no ``hash()`` and no ``id()``, both of which
legitimately vary across processes.
"""

from __future__ import annotations

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseVector
from qufzx.diagram.bangbox import abstract_port_count, abstract_subgraph_count
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.semantics.induction import prove_by_induction


def _build_false_near_identity() -> tuple[Diagram, Diagram]:
    """The bang-boxed GHZ family against the boxed product family, both over ``Dim("d")``.

    The left side port-scope boxes the single Z-spider output under stem ``n``; the right
    side node-scope boxes one independent single-leg Z spider under the same stem, so the
    right side grows ``n`` unwired copies. Both sides mint exactly ``n``, so the pair
    presents one multiplicity index.
    """
    d = Dim("d")

    left = Diagram()
    s = left.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    ref = PortRef(s, Direction.OUTPUT, 0)
    left.set_boundary_outputs([ref])
    left, _left_box_id, _left_n = abstract_port_count(left, ref, 1, stem="n")

    right = Diagram()
    t = right.add_node(Z_SPIDER, input_dims=[], output_dims=[d], phase=PhaseVector(d, {}))
    right.set_boundary_outputs([PortRef(t, Direction.OUTPUT, 0)])
    right, _right_box_id, _right_n = abstract_subgraph_count(right, frozenset({t}), 1, stem="n")

    return left, right


def main() -> None:
    d_value = 2
    left, right = _build_false_near_identity()
    result = prove_by_induction(left, right, witness={"d": d_value})

    comparison = result.comparison
    lines = [
        f"verdict={result.verdict.value!r}",
        f"proved={result.proved!r}",
        f"index={result.index!r}",
        f"base_value={result.base_value!r}",
        f"counterexample={result.counterexample!r}",
        f"reason={result.reason!r}",
        f"comparison.matched={(None if comparison is None else comparison.matched)!r}",
        f"comparison.reason={(None if comparison is None else comparison.reason)!r}",
        f"comparison.mode={(None if comparison is None else comparison.mode.value)!r}",
        f"base.matched={(None if result.base is None else result.base.matched)!r}",
        f"base.reason={(None if result.base is None else result.base.reason)!r}",
        f"witness={sorted(result.witness.items())!r}",
        f"held_symbolic={sorted(result.held_symbolic)!r}",
        f"discharge={(None if result.discharge is None else result.discharge.value)!r}",
        f"derivation_is_none={(result.derivation is None)!r}",
    ]
    for tier in result.tiers:
        tier_tuple = (tier.discharge.value, tier.settled, tier.reason, tier.counterexample)
        lines.append(f"tier={tier_tuple!r}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
