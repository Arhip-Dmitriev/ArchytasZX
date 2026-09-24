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

"""Standalone driver for ``TestStrategyCrossProcessDeterminism`` (see ``test_strategy.py``).

Not a pytest module itself -- run as a plain script, once per ``PYTHONHASHSEED`` value, via
``subprocess``. Drives :func:`~archytaszx.rewrite.engine.toward_normal_form` over a chain of
four Z spiders that admits several fusion matches at once, and prints a stable
serialization of the whole run: the resolved and skipped rule names, the stop reason, the
accumulated scalar, and every step's rule name, consumed ids, port mapping and dimension
constraints.
"""

from __future__ import annotations

from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.engine import (
    missing_normal_form_rule_names,
    normal_form_rules,
    toward_normal_form,
)


def _build_diagram() -> Diagram:
    """A chain of four one-in-one-out Z spiders over ``d``, the first a state and the last an
    effect."""
    d = Dim.symbol("d")
    diagram = Diagram()
    head = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d])
    middle_a = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    middle_b = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    tail = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d])
    diagram.add_wire(PortRef(head, Direction.OUTPUT, 0), PortRef(middle_a, Direction.INPUT, 0))
    diagram.add_wire(PortRef(middle_a, Direction.OUTPUT, 0), PortRef(middle_b, Direction.INPUT, 0))
    diagram.add_wire(PortRef(middle_b, Direction.OUTPUT, 0), PortRef(tail, Direction.INPUT, 0))
    diagram.set_boundary_outputs([PortRef(tail, Direction.OUTPUT, 0)])
    return diagram


def main() -> None:
    diagram = _build_diagram()
    outcome = toward_normal_form(diagram)

    lines = [
        f"rules={[rule.name for rule in normal_form_rules()]!r}",
        f"missing={missing_normal_form_rule_names()!r}",
        f"stop_reason={outcome.stop_reason.value!r}",
        f"steps_attempted={outcome.steps_attempted!r}",
        f"num_steps={len(outcome.steps)!r}",
        f"scalar_accumulated={outcome.scalar_accumulated!r}",
        f"final_scalar={outcome.diagram.scalar!r}",
        f"final_nodes={sorted(outcome.diagram.nodes)!r}",
        f"final_boundary_outputs={outcome.diagram.boundary_outputs!r}",
    ]
    for index, step in enumerate(outcome.steps):
        port_mapping = sorted(step.port_mapping.items(), key=lambda kv: kv[0].sort_key())
        lines.append(
            f"step[{index}]=({step.rule_name!r}, {step.consumed_node_ids!r}, "
            f"{sorted(step.consumed_wires, key=lambda w: w.sort_key())!r}, "
            f"{step.new_node_ids!r}, {port_mapping!r}, {step.scalar_introduced!r}, "
            f"{step.dimension_constraints!r}, {step.side_condition_outcomes!r})"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
