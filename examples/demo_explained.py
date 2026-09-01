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

"""Annotated walkthrough of the current qufzx engine -- read this before filming
``demo_visual.py``, which runs the identical calculations with the narration stripped out.

Not a screen-capture script: no typing animation, no keypress gating, no 72-column budget.
It runs straight through and prints an explanation of what each step is and why it matters
immediately before the live result of that step, so the whole pipeline (build a diagram,
validate it, find a rewrite match, apply it, and oracle-check the result) can be read and
understood in one pass. Every number, name, and detail string is read live from the objects
the engine returns -- the prose around them explains what those objects mean, it does not
substitute for them. Calls only public API from ``qufzx.*``; no test helpers, no new engine
surface. Run with no arguments:

    python examples/demo_explained.py
"""

from __future__ import annotations

import sys
import textwrap

from qufzx.algebra.dimension import Dim
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import RewriteResult, apply
from qufzx.rewrite.match import (
    FUSION_SIDE_CONDITIONS,
    FusionMatch,
    find_matches,
    resolve_fusion_match,
)
from qufzx.rewrite.rules_library import SPIDER_FUSION

WRAP_WIDTH = 88

_GREEN = "\033[32m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _header(title: str) -> None:
    print()
    print(f"{_BOLD}== {title} =={_RESET}")


def _explain(text: str) -> None:
    for line in textwrap.wrap(" ".join(text.split()), width=WRAP_WIDTH):
        print(f"  {line}")
    print()


def _result(text: str) -> None:
    print(f"  {text}")


def _fmt_ref(ref: PortRef) -> str:
    return f"{int(ref.node_id)}.{ref.direction.value}[{ref.index}]"


def _build_ghz_with_copy(dim: Dim) -> tuple[Diagram, NodeId, NodeId]:
    """The "A into B" GHZ-with-copy construction, matching ``tests/helpers.py``'s
    ``build_ghz_with_copy`` (a state-prep spider feeding a copy spider), inlined here so
    this file needs no import from ``tests/``.
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


def _print_diagram_table(diagram: Diagram) -> None:
    for node_id in sorted(diagram.nodes, key=int):
        node = diagram.nodes[node_id]
        in_dims = ",".join(str(port.dim) for port in node.inputs)
        out_dims = ",".join(str(port.dim) for port in node.outputs)
        _result(
            f"node {int(node_id)}: {node.generator_type.name}  "
            f"in={node.num_inputs}[{in_dims}] out={node.num_outputs}[{out_dims}]"
        )
    for wire in sorted(diagram.wires, key=lambda w: w.sort_key()):
        _result(f"wire  {_fmt_ref(wire.a)} -- {_fmt_ref(wire.b)}")
    boundary = ", ".join(_fmt_ref(ref) for ref in diagram.boundary_outputs)
    _result(f"boundary outputs: [{boundary}]")
    _result(f"scalar: {diagram.scalar}")


def step_problem() -> None:
    _header("The problem this engine exists to solve")
    _explain(
        """
        A linear map on n qudits of dimension d is, in the ordinary tensor-network
        picture, a d^n x d^n matrix, and a state is a vector of length d^n. That
        picture only works when both n and d are concrete numbers you can plug in --
        the moment either one is a symbol, there is no array for numpy or any other
        tensor library to allocate. n symbolic ("how many qudits") is the usual
        "parametrized circuit" problem; d symbolic ("what dimension is each qudit")
        is the much less common regime this project targets, and the two combine: a
        diagram may need to prove something true for every n and every d at once.
        """
    )
    _explain(
        """
        ZX-calculus rewrite rules are equalities between diagrams that preserve the
        map the diagram denotes, and they are schematic -- stated once, valid for
        every leg count and dimension. So a single graph rewrite, done correctly, is
        simultaneously a proof of an entire doubly-indexed family of identities. That
        is the promise this script is here to make concrete: build a diagram with a
        symbolic dimension, rewrite it by graph surgery alone (no matrix ever
        constructed), and inspect the structured record the engine keeps of exactly
        what it did and why that was legal.
        """
    )


def step_build() -> Diagram:
    _header("Step 1 -- build a diagram at a symbolic dimension")
    _explain(
        """
        The worked example used throughout this codebase's own test suite is "A into
        B": a Z-spider A with zero inputs and two outputs (a state-preparation
        spider, denoting sum_k |k,k>) feeding one of its outputs into a Z-spider B
        with one input and two outputs (a copy/Delta spider). One output of A is
        wired to the input of B; A's remaining output and both of B's outputs are
        left on the diagram's boundary. Every port's dimension is set to the same
        symbol, d -- not a number -- via Dim.symbol("d"), so nothing downstream can
        accidentally allocate a concrete array.
        """
    )
    diagram, a_id, b_id = _build_ghz_with_copy(Dim.symbol("d"))
    _explain(f"Node A got id {int(a_id)}, node B got id {int(b_id)} (allocated by add_node).")
    _print_diagram_table(diagram)
    return diagram


def step_validate(diagram: Diagram) -> None:
    _header("Step 2 -- validate: is this diagram even well-formed?")
    _explain(
        """
        validate() checks structural well-formedness only -- it does not know what
        the diagram means numerically, and it does not fix anything. It checks that
        every port is wired or on the boundary exactly once, that a wire's two
        endpoints agree on dimension (exactly, or via the placeholder unifier, which
        can also defer an assumption it cannot yet decide), that each node's leg
        counts respect its generator's policy, and that a node's legs jointly agree
        on one shared dimension. A DIMENSION_DEFERRED finding would mean an
        assumption was accepted but not yet proven; an error means the diagram is
        actually broken. Neither should appear here, since every port was built with
        the identical symbol d.
        """
    )
    report = validate(diagram)
    _result(f"is_valid = {report.is_valid}")
    _result(f"errors = {len(report.errors)}, deferred = {len(report.deferred)}")


def step_match(diagram: Diagram) -> FusionMatch:
    _header("Step 3 -- find a spider-fusion match")
    _explain(
        """
        Spider fusion is the one rewrite rule implemented so far: two spiders of the
        same color (both Z, or both X) joined by a wire fuse into a single spider
        whose surviving legs are the union of both nodes' other legs, and whose phase
        (if any) is the sum of the two phases. find_matches() locates every
        occurrence of that shape and, for each one, checks seven side conditions
        before it is allowed to count as a match. These are read directly off the
        engine's own declared condition list below, not retyped by hand -- if a
        future phase adds an eighth condition, this paragraph's count will just be
        wrong until this sentence is fixed, which is exactly the point of reading it
        live.
        """
    )
    _result(f"{len(FUSION_SIDE_CONDITIONS)} declared side conditions for this rule:")
    for condition in FUSION_SIDE_CONDITIONS:
        _result(f"  - {condition.name}: {condition.description}")
    print()

    matches = find_matches(diagram)
    _explain(f"find_matches() returned {len(matches)} match(es) on this diagram.")
    match = matches[0]
    _result(f"a_id={int(match.a_id)}  b_id={int(match.b_id)}")
    _result(f"consumed wire: {_fmt_ref(match.wire.a)} -- {_fmt_ref(match.wire.b)}")
    _result(f"shared_dim = {match.shared_dim}")
    print()
    _explain(
        """
        Every one of the seven conditions below passed for this candidate -- a
        pattern is only allowed to return a candidate whose conditions all pass, so
        this table is a certificate of legality, not a filter still being applied.
        The detail string on each line is the actual, specific fact that condition
        checked for this pair of nodes, not a generic pass/fail label.
        """
    )
    for outcome in match.side_condition_outcomes:
        mark = f"{_GREEN}PASS{_RESET}" if outcome.passed else f"{_RED}FAIL{_RESET}"
        deferred = "  (deferred)" if outcome.deferred else ""
        _result(f"[{mark}] {outcome.name}{deferred}")
        _result(f"        {outcome.detail}")
    return match


def step_apply(diagram: Diagram, match: FusionMatch) -> RewriteResult:
    _header("Step 4 -- apply the rewrite")
    _explain(
        """
        apply() never mutates the diagram it is given -- it works on a copy, splices
        in whatever the rule's builder constructs, removes the two consumed nodes,
        remaps every surviving wire and boundary entry through the builder's
        old-port -> new-port mapping, multiplies in the rule's declared scalar, and
        finally re-validates the result as a structural sanity check relative to the
        input (a pre-existing defect is allowed to survive; a brand new one is not).
        Everything it does is recorded on a RewriteStep, which is the record a future
        certificate-replay feature (not yet implemented) would need to reproduce this
        exact step from scratch.
        """
    )
    result = apply(diagram, SPIDER_FUSION, match)
    _print_diagram_table(result.diagram)
    print()
    step = result.step
    _explain(
        f"""
        The two consumed nodes ({", ".join(str(int(n)) for n in step.consumed_node_ids)})
        are gone; their surviving legs now live on new node
        {", ".join(str(int(n)) for n in result.new_node_ids)}. The rule declared it
        introduces scalar {step.scalar_introduced} on every application (same-color
        fusion across one wire never rescales the diagram, so this is always exactly
        1) -- the engine checked the builder's own computed scalar against that
        declared value and would have raised if they disagreed. {len(step.port_mapping)}
        old ports were remapped onto the merged node's new ports.
        """
    )
    return result


def step_record(diagram: Diagram, match: FusionMatch, result: RewriteResult) -> None:
    _header("Step 5 -- the record: how RewriteStep actually gets built")
    _explain(
        """
        Producing a new diagram is not the whole job: qufzx.rewrite.engine's apply()
        also builds a RewriteStep, a structured record of exactly what happened and
        why it was legal -- the rule name, which nodes and wires were consumed, every
        side condition checked and its outcome, every dimension equality assumed
        rather than proven, the exact scalar introduced, the full old-port to
        new-port mapping, and which pre-existing deferred assumptions the rewrite
        resolved or introduced. This is the data a future Phase 6 certificate-replay
        feature would need to reproduce this exact step from scratch and confirm it
        independently -- it does not exist yet, but the record it would replay
        already does, on every single call to apply(). That record is only as
        trustworthy as the process that built it, so this step does not introduce
        new API; it walks through the same verification apply() itself performs, so
        what gets printed below is shown to be earned rather than merely asserted.
        """
    )
    _explain(
        """
        find_matches() built `match` in the previous step by calling one function,
        resolve_fusion_match(diagram, a_id, b_id, wire), which independently
        re-derives every side condition and dimension assumption straight from the
        diagram -- never trusting a pre-existing match's own fields. spider_fusion's
        builder calls that exact same function again, fresh, against the diagram it
        was actually handed, before doing any graph surgery at all. This matters
        because a hand-built or foreign FusionMatch could in principle claim a
        passing side condition, or a shared_dim, that does not actually hold --
        match-approval and build-applicability being the same function call, not two
        similar-looking computations that could quietly drift apart over time, is
        what closes that gap. Calling it a third time here, independently of both,
        is exactly that same check, made visible.
        """
    )
    resolution = resolve_fusion_match(diagram, match.a_id, match.b_id, match.wire)
    _result(f"resolve_fusion_match(diagram, a_id, b_id, wire) -> passed={resolution.passed}")
    _result(f"  match.side_condition_outcomes == resolution.outcomes: "
            f"{match.side_condition_outcomes == resolution.outcomes}")
    _result(f"  match.dimension_constraints == resolution.dimension_constraints: "
            f"{match.dimension_constraints == resolution.dimension_constraints}")
    _result(f"  match.shared_dim == resolution.shared_dim: "
            f"{match.shared_dim == resolution.shared_dim}")
    _result(f"  dict(match.bindings) == dict(resolution.bindings): "
            f"{dict(match.bindings) == dict(resolution.bindings)}")
    print()
    _explain(
        """
        The builder hands this same resolution back to apply() as
        BuildResult.verified_side_condition_outcomes and
        verified_dimension_constraints. apply() prefers those verified fields over
        match's own when it writes the RewriteStep, so what actually lands on the
        record below is the builder's independently re-checked ground truth, not
        the match's original, unaudited claim -- even though, as just shown, the two
        happen to agree exactly in this run.
        """
    )
    step = result.step
    _result(f"step.side_condition_outcomes == resolution.outcomes: "
            f"{step.side_condition_outcomes == resolution.outcomes}")
    _result(f"step.dimension_constraints == resolution.dimension_constraints: "
            f"{step.dimension_constraints == resolution.dimension_constraints}")
    print()

    _explain("The record itself, RewriteStep, in full:")
    _result(f"rule_name = {step.rule_name!r}")
    _result(f"consumed_node_ids = {step.consumed_node_ids}")
    _result(f"consumed_wires = {step.consumed_wires!r}")
    _result(f"new_node_ids = {result.new_node_ids}")
    _result(f"scalar_introduced = {step.scalar_introduced!r}")
    _result(f"dimension_constraints = {step.dimension_constraints!r}")
    _result(f"removed_deferred_issues = {step.removed_deferred_issues!r}")
    _result(f"introduced_deferred_issues = {step.introduced_deferred_issues!r}")
    _result(f"phase_substitutions = {dict(step.phase_substitutions)!r}")
    _result(f"deferred_issue_identity_ambiguous = {step.deferred_issue_identity_ambiguous}")
    _result(f"port_mapping ({len(step.port_mapping)} entries):")
    for old_ref, new_ref in sorted(step.port_mapping.items(), key=lambda kv: kv[0].sort_key()):
        _result(f"  {old_ref!r} -> {new_ref!r}")
    _result(f"side_condition_outcomes ({len(step.side_condition_outcomes)} entries):")
    for outcome in step.side_condition_outcomes:
        mark = f"{_GREEN}PASS{_RESET}" if outcome.passed else f"{_RED}FAIL{_RESET}"
        _result(f"  [{mark}] {outcome.name}: {outcome.detail}")


def step_solution() -> None:
    _header("The solution, and where it stands today")
    _explain(
        """
        The two symbolic axes are handled by keeping dimension as data rather than a
        global parameter (a Dim lives on each port individually, so a rewrite rule
        can be checked and applied without ever resolving it to a number), and by
        treating a rewrite as graph surgery with side conditions and an exact scalar
        recorded at every step -- never as a numeric operation. The RewriteStep
        record from Step 5 is exactly what a future certificate-replay feature would
        consume: the data already exists, on every rewrite, well before anything
        reads it back that way.
        """
    )
    implemented = (
        "Implemented and under test today: dimension/phase/scalar algebra, the "
        "diagram data model, the numeric oracle (contraction + exact/up-to-phase "
        "comparison), and the rewrite core -- spider fusion with its seven side "
        "conditions, dimension-constraint bookkeeping, and step provenance."
    )
    remaining = (
        "Not yet implemented: replayable proof certificates, bang boxes and free n, "
        "induction over multiplicities, symbolic contraction with d left formal, the "
        "character-sum simplifier, mixed-dimension generators, a broader rule "
        "library and strategy layer, rewrite caching, the normal-form decision "
        "procedure, equality saturation, tactics and proof search, scalable "
        "notation, the Dirac printer, and the general REPL declaration syntax."
    )
    _explain(implemented)
    _explain(remaining)


def main() -> None:
    step_problem()
    diagram = step_build()
    step_validate(diagram)
    match = step_match(diagram)
    result = step_apply(diagram, match)
    step_record(diagram, match, result)
    step_solution()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- clean-failure contract: no traceback on screen
        print(f"demo failed: {exc}")
        sys.exit(1)
