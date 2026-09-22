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

"""
Reformat an InductionResult into mathematical proof notation.

Takes a proof result emitted by prove_by_induction() and renders it in formal
mathematical style suitable for papers and posters. Outputs proof structure,
obligatory diagrams, verdict, and discharge method.

Handles two distinct proof types:
  - UNIFORM_REWRITE: Direct proof via rule that fires at all index values
  - PROVED_INDUCTION: Classical induction with hypothesis and step

Usage:
    from archytaszx.semantics.induction import prove_by_induction
    from examples.proof_formatter import ProofFormatter

    proof = prove_by_induction(left, right, witness={"d": 2})
    formatter = ProofFormatter(proof)
    print(formatter.render_proof())
"""

from __future__ import annotations

from archytaszx.semantics.induction import InductionResult, Verdict


class ProofFormatter:
    """Convert an InductionResult into mathematical proof statements."""

    def __init__(self, proof: InductionResult) -> None:
        """Initialize formatter with an InductionResult from prove_by_induction()."""
        self.proof = proof
        self.obligation = proof.obligation

    def _theorem_statement(self) -> str:
        """Render the universally quantified theorem statement (no witness values)."""
        index = self.proof.index
        held = sorted(self.proof.held_symbolic) if self.proof.held_symbolic else []

        lines = [f"∀ {index} ∈ ℕ"]
        if held:
            held_str = ", ".join(held)
            lines.append(f"∀ {held_str} ∈ ℕ")

        lines.append(f"  LHS({index}) = RHS({index})")
        return "\n".join(lines)

    def _witness_note(self) -> str:
        """Explain witness values used for oracle verification only."""
        if not self.proof.witness:
            return ""

        witness_str = ", ".join(f"{k}={v}" for k, v in sorted(self.proof.witness.items()))
        return (
            f"[Witness: {witness_str}. These are concrete instantiations "
            f"used only for numeric oracle verification; the proof holds "
            f"for all values of the held-symbolic parameters.]"
        )

    def _uniform_rewrite_proof(self) -> str:
        """Render a UNIFORM_REWRITE proof (direct, not induction)."""
        index = self.proof.index

        lines = [
            "THEOREM",
            "-" * 80,
            self._theorem_statement(),
            "",
        ]

        witness_note = self._witness_note()
        if witness_note:
            lines.extend([witness_note, ""])

        lines.extend(
            [
                "PROOF",
                "-" * 80,
                "The spider fusion rule is quantified over all multiplicities and dimensions:",
                "",
                "  ∀ d, ∀ m: SPIDER_FUSION(diagram) → result",
                "",
                f"This rule does not depend on the parameter {index} itself. It fires uniformly",
                f"at every value of {index}:",
                "",
                "  ∀ m: LHS[m] --SPIDER_FUSION→ RHS[m]",
                "",
                f"Therefore, the identity holds for all multiplicities {index} without need for",
                "an induction hypothesis. The proof is direct and uniform. □",
                "",
            ]
        )

        return "\n".join(lines)

    def _induction_proof(self) -> str:
        """Render a PROVED_INDUCTION proof (classical induction)."""
        index = self.proof.index
        base = self.proof.base_value
        step = self.obligation.step_symbol

        lines = [
            "THEOREM",
            "-" * 80,
            self._theorem_statement(),
            "",
        ]

        witness_note = self._witness_note()
        if witness_note:
            lines.extend([witness_note, ""])

        lines.extend(
            [
                "PROOF (by induction on " + index + ")",
                "-" * 80,
                f"Base case ({index} = {base}):",
                f"  LHS[{index}={base}] = RHS[{index}={base}]",
                "",
                f"Inductive hypothesis (assume true for {index} = {step}):",
                f"  LHS[{index}={step}] = RHS[{index}={step}]",
                "",
                f"Inductive step (prove for {index} = {step} + 1):",
                f"  Goal: LHS[{index}={step}+1] = RHS[{index}={step}+1]",
                "",
                f"  By {self.proof.discharge.value if self.proof.discharge else 'the tier'},",
                "  the successor diagrams are rewritten to each other, with the residual",
                "  matching the hypothesis by assumption.",
                "",
                (
                    f"By mathematical induction, LHS({index}) = RHS({index}) "
                    f"for all {index} ≥ {base}. □"
                ),
                "",
            ]
        )

        return "\n".join(lines)

    def _refuted_proof(self) -> str:
        """Render a REFUTED proof (disproof)."""
        index = self.proof.index
        cx = self.proof.counterexample

        lines = [
            "THEOREM (REFUTED)",
            "-" * 80,
            self._theorem_statement(),
            "",
            "COUNTEREXAMPLE",
            "-" * 80,
            f"The claimed equality fails at {index} = {cx}:",
            f"  LHS[{index}={cx}] ≠ RHS[{index}={cx}]",
            "",
            f"Reason: {self.proof.reason}",
            "",
            "The theorem is false. ✗",
            "",
        ]

        return "\n".join(lines)

    def _inconclusive_proof(self) -> str:
        """Render an INCONCLUSIVE proof (no tier settled)."""
        lines = [
            "THEOREM (INCONCLUSIVE)",
            "-" * 80,
            self._theorem_statement(),
            "",
            "STATUS",
            "-" * 80,
            "No discharge tier proved or refuted the equality.",
            f"Reason: {self.proof.reason}",
            "",
            "Verdict unknown. The proof is incomplete.",
            "",
        ]

        return "\n".join(lines)

    def render_proof(self) -> str:
        """Render the proof in formal mathematical notation."""
        # Route to appropriate proof type
        if self.proof.verdict == Verdict.PROVED_UNIFORM:
            base = self._uniform_rewrite_proof()
        elif self.proof.verdict in (Verdict.PROVED_INDUCTION,):
            base = self._induction_proof()
        elif self.proof.verdict == Verdict.REFUTED:
            base = self._refuted_proof()
        else:  # INCONCLUSIVE, SCHEMA_CHECKED
            base = self._inconclusive_proof()

        # Obligation structure
        ob = self.obligation
        obligation_section = "\n".join(
            [
                "OBLIGATION STRUCTURE",
                "-" * 80,
                f"Index: {self.proof.index}",
                f"Base value: {self.proof.base_value}",
                f"Step symbol: {ob.step_symbol}",
                f"Held symbolic: {sorted(self.proof.held_symbolic) or 'none'}",
                "",
                "Diagrams constructed:",
                (
                    f"  LHS at base ({self.proof.index}={self.proof.base_value}): "
                    f"{len(ob.left_at_base.nodes)} nodes"
                ),
                (
                    f"  RHS at base ({self.proof.index}={self.proof.base_value}): "
                    f"{len(ob.right_at_base.nodes)} nodes"
                ),
                (
                    f"  LHS at hypothesis ({self.proof.index}={ob.step_symbol}): "
                    f"{len(ob.left_at_k.nodes)} nodes, "
                    f"{len(ob.left_at_k.bang_boxes)} bang box(es)"
                ),
                (
                    f"  RHS at hypothesis ({self.proof.index}={ob.step_symbol}): "
                    f"{len(ob.right_at_k.nodes)} nodes, "
                    f"{len(ob.right_at_k.bang_boxes)} bang box(es)"
                ),
                (
                    f"  LHS at successor ({self.proof.index}={ob.step_symbol}+1): "
                    f"{len(ob.left_at_successor.nodes)} nodes"
                ),
                (
                    f"  RHS at successor ({self.proof.index}={ob.step_symbol}+1): "
                    f"{len(ob.right_at_successor.nodes)} nodes"
                ),
                "",
            ]
        )

        # Proof metadata
        metadata_section = "\n".join(
            [
                "PROOF METADATA",
                "-" * 80,
                f"Verdict: {self.proof.verdict.value}",
                f"Proved: {self.proof.proved}",
                f"Discharge tier: {self.proof.discharge.value if self.proof.discharge else 'N/A'}",
                f"Certificate: {'Yes' if self.proof.derivation else 'No'}",
                f"Reason: {self.proof.reason}",
                "",
                "=" * 80,
            ]
        )

        return "=" * 80 + "\n" + base + obligation_section + metadata_section


if __name__ == "__main__":
    """Example usage: format the spider fusion proof."""

    from archytaszx.algebra.dimension import Dim
    from archytaszx.diagram.bangbox import abstract_subgraph_count
    from archytaszx.diagram.generators import Z_SPIDER
    from archytaszx.diagram.graph import Diagram, Direction, PortRef
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.match import find_matches
    from archytaszx.rewrite.rules_library import SPIDER_FUSION
    from archytaszx.semantics.induction import prove_by_induction

    # Build problem diagram
    d = Dim.symbol("d")
    g = Diagram()
    a = g.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
    b = g.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    g.set_boundary_outputs(
        [
            PortRef(a, Direction.OUTPUT, 1),
            PortRef(b, Direction.OUTPUT, 0),
            PortRef(b, Direction.OUTPUT, 1),
        ]
    )

    # Box and prove by induction
    a_id, b_id = list(g.nodes.keys())
    boxed, box_id, mult = abstract_subgraph_count(g, frozenset({a_id, b_id}), 1, stem="m")
    fused_boxed = apply(boxed, SPIDER_FUSION, find_matches(boxed)[0]).diagram
    proof = prove_by_induction(boxed, fused_boxed, witness={"d": 2})

    # Format and emit
    formatter = ProofFormatter(proof)
    print(formatter.render_proof())
