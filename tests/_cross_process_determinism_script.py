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

"""Standalone driver for ``TestCrossProcessDeterminism`` (see ``test_engine.py``).

Not a pytest module itself -- run as a plain script, once per ``PYTHONHASHSEED`` value, via
``subprocess``. Builds a diagram engineered so more than one pre-existing, node-anchored
``DIMENSION_DEFERRED`` issue collides onto the same translated key once fusion merges its
two nodes, applies the one fusion match at it, and prints a stable serialization of every
field promised deterministic across processes: ``removed_deferred_issues``,
``introduced_deferred_issues``, ``deferred_issue_identity_ambiguous``,
``dimension_constraints``, ``side_condition_outcomes``, and ``validate(...)``'s errors and
deferred issues on both diagrams.

Printing ``repr()`` is safe because every value object on this path defines a ``__repr__``
built from literal attribute names and values -- ``PortRef.__repr__`` uses
``direction.value``, a fixed string, never the ``Direction`` member's own ``repr`` -- and
none embeds a raw ``id()`` or ``hash()``. This script deliberately prints no ``hash()``,
which legitimately varies across processes.
"""

from __future__ import annotations

from qufzx.algebra.dimension import Dim
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION


def _build_diagram() -> Diagram:
    """Two Z spiders, each with its own pre-existing node-level DIMENSION_DEFERRED issue.

    A: outputs [d, d*e, d]; ALL_LEGS_EQUAL compares against the first leg's dim (d), and the
    middle leg (d*e) defers -- one DIMENSION_DEFERRED issue anchored on A.
    B: inputs [d, d], outputs [d*e]; symmetric shape, one DIMENSION_DEFERRED issue anchored
    on B. A.outputs[0] -- B.inputs[0] and A.outputs[2] -- B.inputs[1] join them by two
    parallel wires (a condition-3 pair), so find_matches returns two candidates; the lower
    (direction, index) one is applied. Once merged, every surviving leg is forced onto the
    resolved shared_dim (d) exactly, so both pre-existing per-node deferred issues vanish --
    both translate (via the single-new-node-id fallback) onto the identical post-rewrite key,
    testing the multiset/collision machinery, not just a single scalar count.
    """
    d = Dim.symbol("d")
    e = Dim.symbol("e")
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d * e, d])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[d, d], output_dims=[d * e])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 2), PortRef(b_id, Direction.INPUT, 1))
    diagram.set_boundary_outputs([PortRef(b_id, Direction.OUTPUT, 0)])
    return diagram


def main() -> None:
    diagram = _build_diagram()
    matches = find_matches(diagram)
    match = matches[0]
    result = apply(diagram, SPIDER_FUSION, match)
    step = result.step

    pre_report = validate(diagram)
    post_report = validate(result.diagram)

    lines = [
        f"num_matches={len(matches)!r}",
        f"dimension_constraints={step.dimension_constraints!r}",
        f"side_condition_outcomes={step.side_condition_outcomes!r}",
        f"removed_deferred_issues={step.removed_deferred_issues!r}",
        f"introduced_deferred_issues={step.introduced_deferred_issues!r}",
        f"deferred_issue_identity_ambiguous={step.deferred_issue_identity_ambiguous!r}",
        f"pre_errors={pre_report.errors!r}",
        f"pre_deferred={pre_report.deferred!r}",
        f"post_errors={post_report.errors!r}",
        f"post_deferred={post_report.deferred!r}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
