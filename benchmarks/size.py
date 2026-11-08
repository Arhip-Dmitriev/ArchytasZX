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

"""B3 element counts of a diagram, and the smallest diagram of a decision's derivations
(BENCHMARKS.txt, section 6, B3).

Counted: nodes per generator, node-scope and port-scope boxes and the deepest nesting, phase
entries (a rule counts once), weight entries, wires, self-loops, boundary inputs and outputs,
and the scalar's sympy operation count. ``total`` is nodes + boxes + phases + weights + wires;
the smallest diagram has the least total, ties broken by fewer wires.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

GENERATORS = ("Z", "X", "W", "F", "T", "Ti", "B", "S", "G")


def counts(diagram: Any) -> dict[str, int]:
    """The element counts of one diagram."""
    import sympy as sp

    out: dict[str, int] = {f"node_{g}": 0 for g in GENERATORS}
    phases = weights = 0
    for node in diagram.nodes.values():
        name = node.generator_type.name
        out[f"node_{name}"] = out.get(f"node_{name}", 0) + 1
        ph = node.phase
        if ph is not None:
            phases += len(ph.entries()) + (1 if ph.rule is not None else 0)
            w = getattr(ph, "weights", None)
            if callable(w):
                weights += len(w()) + (1 if getattr(ph, "weight_rule", None) is not None else 0)
    boxes = list(diagram.bang_boxes.values())
    node_boxes = sum(1 for b in boxes if b.node_scope)
    port_boxes = sum(1 for b in boxes if b.port_scope)
    depth = 0
    by_id = {b.id: b for b in boxes}
    for b in boxes:
        k, cur = 1, b
        while cur.parent is not None and cur.parent in by_id:
            k += 1
            cur = by_id[cur.parent]
        depth = max(depth, k)
    wires = list(diagram.wires)
    loops = sum(1 for w in wires if w.a.node_id == w.b.node_id)
    try:
        scalar_size = int(sp.count_ops(diagram.scalar.to_sympy())) + 1
    except Exception:  # noqa: BLE001
        scalar_size = len(str(diagram.scalar))
    nodes = len(diagram.nodes)
    out |= {
        "nodes": nodes,
        "node_boxes": node_boxes,
        "port_boxes": port_boxes,
        "box_depth": depth,
        "phases": phases,
        "weights": weights,
        "wires": len(wires),
        "self_loops": loops,
        "inputs": len(diagram.boundary_inputs),
        "outputs": len(diagram.boundary_outputs),
        "scalar_size": scalar_size,
    }
    out["total"] = nodes + node_boxes + port_boxes + phases + weights + len(wires)
    return out


def _derivation_diagrams(derivation: Any) -> Iterator[Any]:
    from archytaszx.semantics.certificate import _replay_one

    yield derivation.initial
    working = derivation.initial.copy()
    for k, step in enumerate(derivation.steps):
        failure, working = _replay_one(working, k, step, True)
        if failure is not None:
            break
        yield working
    yield derivation.final
    for child in derivation.children:
        yield from _derivation_diagrams(child)


def diagrams_of(decision: Any, left: Any, right: Any) -> Iterator[Any]:
    """Both sides, both normal forms and every diagram of every certificate's derivation."""
    yield left
    yield right
    for nf in (decision.left_nf, decision.right_nf):
        if nf is not None:
            yield nf
    for cert in decision.certificates or ():
        yield from _derivation_diagrams(cert.derivation)


def smallest(diagrams: Iterator[Any]) -> dict[str, int] | None:
    """The counts of the smallest diagram (least total, then fewest wires)."""
    best: dict[str, int] | None = None
    for d in diagrams:
        try:
            c = counts(d)
        except Exception:  # noqa: BLE001, S112
            continue
        if best is None or (c["total"], c["wires"]) < (best["total"], best["wires"]):
            best = c
    return best


def b3(decision: Any, left: Any, right: Any) -> dict[str, Any]:
    """``naive`` (the circuit's counts), ``smallest`` and ``ratio`` naive / smallest total."""
    naive = counts(left)
    small = smallest(diagrams_of(decision, left, right))
    ratio = naive["total"] / small["total"] if small and small["total"] else None
    return {"naive": naive, "smallest": small, "ratio": ratio}
