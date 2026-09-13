#!/usr/bin/env python3
"""
Run the proof formatter on the spider fusion example and emit the formatted proof.

Usage:
    python examples/format_spider_fusion_proof.py

Output:
    - Text-formatted mathematical proof to stdout
"""

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.bangbox import abstract_subgraph_count
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.rules_library import SPIDER_FUSION
from archytaszx.semantics.induction import prove_by_induction
from examples.proof_formatter import ProofFormatter


def main():
    """Build spider fusion diagram, prove by induction, format and emit proof."""
    
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


if __name__ == "__main__":
    main()
