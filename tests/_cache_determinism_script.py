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

"""Standalone driver for ``TestCacheCrossProcessDeterminism`` (see ``test_cache.py``).

Not a pytest module itself -- run as a plain script, once per ``PYTHONHASHSEED`` value, via
``subprocess``. Prints every cache key promised stable across processes: each node's
``node_digest`` and ``incidence_digest``, every :class:`DiagramFingerprint` field, the
``pattern_key`` of each pattern in ``match.py``, the ``canonical`` key, and the ``repr`` of a
cached ``MatchCache.matches`` result.
"""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.bangbox import Mult
from archytaszx.diagram.generators import X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.cache import (
    MatchCache,
    fingerprint,
    incidence_digest,
    node_digest,
    pattern_key,
)
from archytaszx.rewrite.match import (
    BialgebraPattern,
    CapPattern,
    FourierCancellationPattern,
    FourierStateColorChangePattern,
    FusionPattern,
    HopfPattern,
    IdentityRemovalPattern,
    StateCopyPattern,
    TriangleInverseCancellationPattern,
)

DIM = Dim.concrete(2)

PATTERNS = (
    FusionPattern(),
    FourierCancellationPattern(),
    CapPattern(),
    IdentityRemovalPattern(),
    TriangleInverseCancellationPattern(),
    StateCopyPattern(),
    HopfPattern(),
    BialgebraPattern(),
    FourierStateColorChangePattern(),
)


def _build_diagram() -> Diagram:
    """Two wired Z spiders plus an X spider, a phase, a scalar, a parameter and two bang boxes."""
    diagram = Diagram()
    a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[DIM, DIM])
    b_id = diagram.add_node(Z_SPIDER, input_dims=[DIM], output_dims=[DIM, DIM])
    c_id = diagram.add_node(X_SPIDER, input_dims=[DIM], output_dims=[DIM])
    diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
    diagram.add_wire(PortRef(b_id, Direction.OUTPUT, 0), PortRef(c_id, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [
            PortRef(b_id, Direction.OUTPUT, 1),
            PortRef(a_id, Direction.OUTPUT, 1),
            PortRef(c_id, Direction.OUTPUT, 0),
        ]
    )
    diagram.set_phase(b_id, PhaseVector(DIM, {1: Phase.turns(sp.Rational(1, 3))}))
    diagram.multiply_scalar(Scalar.rational(3))
    diagram.bind_parameter("n", 2)
    outer = diagram.add_bang_box(Mult.symbol("n"), node_scope=frozenset({a_id, b_id}))
    diagram.add_bang_box(
        Mult.concrete(2), port_scope=frozenset({PortRef(c_id, Direction.OUTPUT, 0)}), parent=outer
    )
    return diagram


def main() -> None:
    diagram = _build_diagram()
    lines: list[str] = []
    for node_id in sorted(diagram.nodes):
        lines.append(f"node_digest[{node_id!r}]={node_digest(diagram, node_id)!r}")
        lines.append(f"incidence_digest[{node_id!r}]={incidence_digest(diagram, node_id)!r}")

    fp = fingerprint(diagram)
    lines.append(f"nodes={tuple((nid, fp.nodes[nid]) for nid in sorted(fp.nodes))!r}")
    lines.append(f"global_key={fp.global_key!r}")
    lines.append(f"boundary_key={fp.boundary_key!r}")
    lines.append(f"env_key={fp.env_key!r}")
    lines.append(f"boxes_key={fp.boxes_key!r}")

    for pattern in PATTERNS:
        lines.append(f"pattern_key={pattern_key(pattern)!r}")

    cache = MatchCache()
    matches = cache.matches(FusionPattern(), diagram, fingerprint=fp)
    lines.append(f"matches={matches!r}")
    lines.append(f"cached_matches={cache.matches(FusionPattern(), diagram, fingerprint=fp)!r}")
    lines.append(f"stats={cache.stats!r}")
    lines.append(f"canonical={cache.canonical(diagram, fingerprint=fp)!r}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
