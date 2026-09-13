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

Usage:
    from archytaszx.semantics.induction import prove_by_induction
    from examples.proof_formatter import ProofFormatter

    proof = prove_by_induction(left, right, witness={"d": 2})
    formatter = ProofFormatter(proof)
    print(formatter.render_proof())
    print(formatter.render_latex())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from archytaszx.semantics.induction import InductionResult, Verdict, StepDischarge


@dataclass(frozen=True)
class ProofStatement:
    """One named component of a proof rendered in mathematical notation."""

    name: str
    content: str


class ProofFormatter:
    """Convert an InductionResult into mathematical proof statements and LaTeX."""

    def __init__(self, proof: InductionResult) -> None:
        """Initialize formatter with an InductionResult from prove_by_induction()."""
        self.proof = proof
        self.obligation = proof.obligation

    def _base_case_statement(self) -> str:
        """Render the base case as a mathematical assertion."""
        index = self.proof.index
        base = self.proof.base_value
        witness_str = ", ".join(f"{k}={v}" for k, v in sorted(self.proof.witness.items()))

        if witness_str:
            return f"LHS[{index}={base}, {witness_str}] = RHS[{index}={base}, {witness_str}]"
        else:
            return f"LHS[{index}={base}] = RHS[{index}={base}]"

    def _hypothesis_statement(self) -> str:
        """Render the inductive hypothesis."""
        index = self.proof.index
        step = self.obligation.step_symbol

        held = sorted(self.proof.held_symbolic) if self.proof.held_symbolic else []
        if held:
            held_str = ", ".join(held)
            return f"Assume: LHS[{index}={step}, {held_str}] = RHS[{index}={step}, {held_str}]"
        else:
            return f"Assume: LHS[{index}={step}] = RHS[{index}={step}]"

    def _successor_statement(self) -> str:
        """Render the goal for the inductive step."""
        index = self.proof.index
        step = self.obligation.step_symbol

        held = sorted(self.proof.held_symbolic) if self.proof.held_symbolic else []
        if held:
            held_str = ", ".join(held)
            return f"Goal: LHS[{index}={step}+1, {held_str}] = RHS[{index}={step}+1, {held_str}]"
        else:
            return f"Goal: LHS[{index}={step}+1] = RHS[{index}={step}+1]"

    def _discharge_explanation(self) -> str:
        """Explain which tier settled the proof and why it works."""
        if not self.proof.discharge:
            return "No discharge tier settled the proof."

        discharge = self.proof.discharge
        index = self.proof.index

        explanations = {
            StepDischarge.UNIFORM_REWRITE: (
                f"A single rewrite rule fires uniformly at every value of {index}. "
                f"The rule is quantified over all ({index}, d) pairs and does not depend on "
                f"the multiplicity parameter itself. Therefore, the rule applies at {index}={index}+1 "
                f"just as it does at {index}=k, proving the step without an induction hypothesis."
            ),
            StepDischarge.INDUCTION_REWRITE: (
                f"The successor diagrams are rewritten to each other using the induction hypothesis "
                f"as a rewrite rule. One copy is peeled off each successor, exposing the hypothesis "
                f"diagram as the residual, which by assumption is equal on both sides. "
                f"The peeled copies are then verified through symbolic contraction."
            ),
            StepDischarge.SYMBOLIC_CONTRACTION: (
                f"Both successor diagrams are contracted symbolically with every symbol formal, "
                f"including {index} and all dimensions d. The two resulting scalar expressions "
                f"are algebraically simplified through character-sum reduction and found to be equal."
            ),
            StepDischarge.ORACLE_WINDOW: (
                f"The equality is verified numerically at a window of concrete {index} values "
                f"above the base. This is a finite schema check, not a proof."
            ),
        }

        return explanations.get(
            discharge,
            f"Settled by {discharge.value}."
        )

    def _verdict_summary(self) -> str:
        """Render the proof verdict and overall result."""
        if self.proof.verdict == Verdict.PROVED_UNIFORM:
            return (
                f"PROVED (UNIFORM): The identity holds for all values of {self.proof.index} "
                f"with all other symbols universally quantified."
            )
        elif self.proof.verdict == Verdict.PROVED_INDUCTION:
            return (
                f"PROVED (INDUCTION): Base case and inductive step both discharge. "
                f"The identity holds for all values of {self.proof.index} ≥ {self.proof.base_value}."
            )
        elif self.proof.verdict == Verdict.SCHEMA_CHECKED:
            return (
                f"SCHEMA CHECKED (NOT A PROOF): The identity verified at a finite window of "
                f"{self.proof.index} values, but this is not a mathematical proof."
            )
        elif self.proof.verdict == Verdict.REFUTED:
            return f"REFUTED: The identity fails at {self.proof.index}={self.proof.counterexample}."
        else:
            return f"INCONCLUSIVE: No discharge tier settled the step case."

    def render_proof(self) -> str:
        """Render the proof in formal mathematical notation as a string."""
        lines = [
            "=" * 80,
            "PROOF BY INDUCTION",
            "=" * 80,
            "",
        ]

        # Theorem statement
        lines.extend([
            "THEOREM",
            "-" * 80,
            "∀ " + self.proof.index + " ∈ ℕ",
        ])
        if self.proof.held_symbolic:
            held_str = ", ".join(sorted(self.proof.held_symbolic))
            lines.append(f"∀ {held_str} ∈ ℕ")
        lines.extend([
            "  LHS(" + self.proof.index + ") = RHS(" + self.proof.index + ")",
            "",
        ])

        # Base case
        lines.extend([
            "BASE CASE: " + self.proof.index + " = " + str(self.proof.base_value),
            "-" * 80,
            self._base_case_statement(),
            "",
        ])

        # Inductive step
        lines.extend([
            "INDUCTIVE STEP",
            "-" * 80,
            self._hypothesis_statement(),
            self._successor_statement(),
            "",
            "Discharge:",
            self._discharge_explanation(),
            "",
        ])

        # Conclusion
        lines.extend([
            "CONCLUSION",
            "-" * 80,
            self._verdict_summary(),
            "",
        ])

        # Obligation details
        lines.extend([
            "OBLIGATION STRUCTURE",
            "-" * 80,
            f"Index: {self.proof.index}",
            f"Base value: {self.proof.base_value}",
            f"Step symbol: {self.obligation.step_symbol}",
            f"Held symbolic: {sorted(self.proof.held_symbolic) or 'none'}",
            f"Witness: {dict(self.proof.witness)}",
            "",
            f"Diagrams constructed:",
            f"  LHS at base: {len(self.obligation.left_at_base.nodes)} nodes",
            f"  RHS at base: {len(self.obligation.right_at_base.nodes)} nodes",
            f"  LHS at k: {len(self.obligation.left_at_k.nodes)} nodes, "
            f"{len(self.obligation.left_at_k.bang_boxes)} box(es)",
            f"  RHS at k: {len(self.obligation.right_at_k.nodes)} nodes, "
            f"{len(self.obligation.right_at_k.bang_boxes)} box(es)",
            f"  LHS at k+1: {len(self.obligation.left_at_successor.nodes)} nodes",
            f"  RHS at k+1: {len(self.obligation.right_at_successor.nodes)} nodes",
            "",
        ])

        # Proof metadata
        lines.extend([
            "PROOF METADATA",
            "-" * 80,
            f"Verdict: {self.proof.verdict.value}",
            f"Proved: {self.proof.proved}",
            f"Discharge tier: {self.proof.discharge.value if self.proof.discharge else 'N/A'}",
            f"Certificate: {'Yes' if self.proof.derivation else 'No'}",
            f"Reason: {self.proof.reason}",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    def render_latex(self) -> str:
        """Render the proof in LaTeX suitable for a paper or poster."""
        lines = [
            r"\section*{Proof by Induction}",
            "",
            r"\subsection*{Theorem}",
            "",
            r"\[",
            r"  \forall " + self.proof.index + r" \in \mathbb{N}",
        ]

        if self.proof.held_symbolic:
            held_list = ", ".join(sorted(self.proof.held_symbolic))
            lines.append(r"  \forall " + held_list + r" \in \mathbb{N}")

        lines.extend([
            r"  \text{LHS}(" + self.proof.index + r") = \text{RHS}(" + self.proof.index + r")",
            r"\]",
            "",
            r"\subsection*{Base Case: $" + self.proof.index + " = " + str(self.proof.base_value) + r"$}",
            "",
            r"\[",
            "  " + self._base_case_statement().replace("=", r"=").replace("[", "_{").replace("]", "}"),
            r"\]",
            "",
            r"\subsection*{Inductive Step}",
            "",
            r"\textbf{Hypothesis:}",
            "",
            r"\[",
            "  " + self._hypothesis_statement()
            .replace("Assume: ", "")
            .replace("[", "_{")
            .replace("]", "}"),
            r"\]",
            "",
            r"\textbf{Goal:}",
            "",
            r"\[",
            "  " + self._successor_statement()
            .replace("Goal: ", "")
            .replace("[", "_{")
            .replace("]", "}"),
            r"\]",
            "",
            r"\textbf{Proof:}",
            "",
        ])

        # Discharge explanation in LaTeX
        discharge_text = self._discharge_explanation()
        lines.append(discharge_text)
        lines.extend([
            "",
            r"\subsection*{Conclusion}",
            "",
        ])

        verdict_text = self._verdict_summary()
        lines.append(r"\textbf{" + verdict_text + r"}")
        lines.append("")

        return "\n".join(lines)

    def emit_all(self) -> tuple[str, str]:
        """Return both text and LaTeX renderings as a tuple."""
        return self.render_proof(), self.render_latex()


if __name__ == "__main__":
    """Example usage: format the spider fusion proof."""

    from archytaszx.algebra.dimension import Dim
    from archytaszx.diagram.generators import Z_SPIDER
    from archytaszx.diagram.graph import Diagram, Direction, PortRef
    from archytaszx.diagram.bangbox import abstract_subgraph_count
    from archytaszx.rewrite.engine import apply
    from archytaszx.rewrite.match import find_matches
    from archytaszx.rewrite.rules_library import SPIDER_FUSION

    # Build problem diagram
    d = Dim.symbol("d")
    g = Diagram()
    a = g.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
    b = g.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
    g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
    g.set_boundary_outputs([
        PortRef(a, Direction.OUTPUT, 1),
        PortRef(b, Direction.OUTPUT, 0),
        PortRef(b, Direction.OUTPUT, 1),
    ])

    # Box and prove by induction
    a_id, b_id = list(g.nodes.keys())
    boxed, box_id, mult = abstract_subgraph_count(g, frozenset({a_id, b_id}), 1, stem="m")
    fused_boxed = apply(boxed, SPIDER_FUSION, find_matches(boxed)[0]).diagram
    proof = prove_by_induction(boxed, fused_boxed, witness={"d": 2})

    # Format and emit
    formatter = ProofFormatter(proof)
    print(formatter.render_proof())
    print("\n\n")
    print("LATEX OUTPUT:")
    print("-" * 80)
    print(formatter.render_latex())
