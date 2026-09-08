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

"""Runnable tour of every engine layer, one short section per public entry point.

Each ``stage_*`` function builds on the previous one's return value and prints what the
engine handed back. Calls only public ``archytaszx.*`` API -- no test helpers. Deterministic and
non-interactive; run with no arguments and no config:

    python examples/api_tour.py

The worked example throughout is "A into B": a state-prep Z spider ``sum_k |kk>`` whose
first output feeds a copy spider, over a symbolic dimension ``d``. Section 7 boxes that same
graph under a multiplicity ``m`` and proves the fusion identity for every ``m`` at once.
Companion to ``docs/TUTORIAL.md``, whose sections match these numbers.
"""

from __future__ import annotations

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.bangbox import abstract_subgraph_count
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, NodeId, PortRef
from archytaszx.diagram.validate import validate
from archytaszx.repl.parser import parse_dirac_source
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.semantics.certificate import certify, verify
from archytaszx.semantics.check import compare
from archytaszx.semantics.contract_symbolic import contract_symbolic
from archytaszx.semantics.induction import prove_by_induction


def build_ghz_with_copy(dim: Dim) -> tuple[Diagram, NodeId, NodeId]:
    """Build the "A into B" graph over ``dim`` and return it with both node ids.

    A is a 0-input/2-output Z spider; B is a 1-input/2-output Z spider. A's output 0 is
    wired to B's input 0, leaving three free outputs as the ordered boundary.
    """
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [
            PortRef(a_id, Direction.OUTPUT, 1),
            PortRef(b_id, Direction.OUTPUT, 0),
            PortRef(b_id, Direction.OUTPUT, 1),
        ]
    )
    return diagram, a_id, b_id


def stage_1_validate(diagram: Diagram) -> None:
    """Print the validation report's errors and non-fatal findings."""
    report = validate(diagram)
    print("1. validate")
    print(f"     errors:   {[i.kind.value for i in report.errors] or 'none'}")
    print(f"     findings: {[i.kind.value for i in report.issues] or 'none'}")


def stage_2_match_and_rewrite(diagram: Diagram) -> Diagram:
    """Find the fusion match, apply it, and print the step's provenance."""
    matches = find_matches(diagram)
    result = apply(diagram, SPIDER_FUSION, matches[0])
    print("2. match and rewrite")
    print(f"     matches found:  {len(matches)}")
    print(f"     nodes:          {len(diagram.nodes)} -> {len(result.diagram.nodes)}")
    print(f"     scalar:         {result.diagram.scalar}")
    print(f"     side conditions:{len(result.step.side_condition_outcomes)} all passed")
    print(f"     constraints:    {result.step.dimension_constraints or 'none assumed'}")
    return result.diagram


def stage_3_certificate(diagram: Diagram) -> None:
    """Certify the one-step derivation and replay-verify it at a concrete dimension."""
    matches = find_matches(diagram)
    result = apply(diagram, SPIDER_FUSION, matches[0])
    certificate = certify(diagram, [result], label="fuse A into B")
    report = verify(certificate, {"d": 3})
    print("3. certificate")
    print(f"     steps:    {len(certificate.steps)}")
    print(f"     verified: {report.verified} at d=3")


def stage_4_numeric_oracle(before: Diagram, after: Diagram) -> None:
    """Instantiate both diagrams at a concrete dimension and compare contracted tensors."""
    print("4. numeric oracle")
    for value in (2, 3, 4):
        print(f"     d={value}: matched={compare(before, after, {'d': value}).matched}")


def stage_5_symbolic_contraction(diagram: Diagram) -> None:
    """Contract with the dimension left formal and print the closed entry."""
    print("5. symbolic contraction, d formal")
    print(f"     entry: {contract_symbolic(diagram).entry}")


def stage_6_dirac() -> None:
    """Parse both Dirac source forms and print node counts and parameter environments."""
    print("6. Dirac front end")
    for source in ("sum_{k=0}^{d-1} |k,k>", "sum_{k=0}^{3-1} |k>^{30}; copy"):
        parsed = parse_dirac_source(source)
        print(
            f"     {source!r}"
            f" -> {len(parsed.nodes)} node(s), {len(parsed.bang_boxes)} box(es),"
            f" env {dict(parsed.parameters)}"
        )


def stage_7_induction(diagram: Diagram, a_id: NodeId, b_id: NodeId) -> None:
    """Box the graph under a multiplicity and prove the fusion identity for every value."""
    boxed, _box_id, mult = abstract_subgraph_count(diagram, frozenset({a_id, b_id}), 1, stem="m")
    fused = apply(boxed, SPIDER_FUSION, find_matches(boxed)[0]).diagram
    result = prove_by_induction(boxed, fused, witness={"d": 2})
    print("7. induction over a symbolic multiplicity")
    print(f"     multiplicity:  {mult}")
    print(f"     proved:        {result.proved}")
    print(f"     verdict:       {result.verdict.value}")
    print(f"     index:         {result.index}")
    print(f"     settling tier: {result.discharge.value if result.discharge else 'none'}")
    print(f"     held symbolic: {sorted(result.held_symbolic) or 'none'}")


def main() -> None:
    """Run every stage in order against the "A into B" example over a symbolic ``d``."""
    dim = Dim.symbol("d")
    before, a_id, b_id = build_ghz_with_copy(dim)

    stage_1_validate(before)
    after = stage_2_match_and_rewrite(before)
    stage_3_certificate(before)
    stage_4_numeric_oracle(before, after)
    stage_5_symbolic_contraction(after)
    stage_6_dirac()
    stage_7_induction(before, a_id, b_id)


if __name__ == "__main__":
    main()
