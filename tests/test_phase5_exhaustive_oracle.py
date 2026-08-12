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

"""EXHAUSTIVE (not random) spider-fusion oracle sweep. Phase 5 round-12 audit, B3.

Every prior property harness in ``test_fusion_properties.py`` is a *random* sample of a
much larger space, no matter how many seeds it runs -- a shape the sampler never happens to
draw stays untested forever, and rounds 3-11 of the audit chased shape-specific defects one
at a time precisely because of this. This module instead enumerates the *entire* finite
space of every 2-node, 0-2-legs-per-side, 1- or 2-wire spider pair at ``d in {2, 3}`` with a
phase present or absent on each node, and asserts oracle equality on every fusion match this
space produces. If this test passes, no fusion shape in that space -- however unlikely a
random generator would be to draw it -- can silently disagree with the oracle.

Space and runtime. Per node: 3 input-leg counts (0, 1, 2) x 3 output-leg counts (0, 1, 2) x
2 phase choices (absent, or a fixed concrete phase at entry index 1) = 18 node shapes; two
nodes, two colours each, two ``d`` values gives 18 * 18 * 2 * 2 * 2 = 2,592 base
(shape, colour, d) configurations before wiring is even chosen. For each, every distinct
1-wire and 2-wire matching between the two nodes' ports is enumerated (no port reused across
wires, matching the diagram model). The 2-wire arm at ``d = 3`` is by far the most expensive
slice (most matchings per shape pair, largest dense tensors) and is the one deliberately
subsampled away -- see ``_SKIP_TWO_WIRE_AT_D3`` below -- per this task's explicit "subsample
d=3 or the 2-wire arm if needed" allowance; the 1-wire arm still runs fully at every ``d``,
and the 2-wire arm still runs fully at ``d = 2``. Do not "optimise" this into a random
sample: the entire point is that it is not one.

Scope boundary (Phase 5 audit round 15, N1): every node here is built at a single, shared
*concrete* ``Dim.concrete(d_value)`` (see :func:`_build_node`) -- no symbol, no product, no
power ever appears anywhere in this space, so :func:`_is_cleanly_contractible` holds
trivially for every diagram this module constructs (asserted, never merely assumed) and
every ``dimension_agreement``/``phase_dimension_agreement`` outcome this space can produce is
a bare syntactic identity, never a ``DEFERRED`` or ``BOUND`` one. This sweep is therefore
exhaustive over *leg/wire topology* at fixed concrete ``d``, not over the deferred/binding
resolution path (the fixpoint in :func:`~qufzx.rewrite.match.resolve_fusion_match` that D1
lived in) -- a defect confined to that path, such as D1, is structurally invisible here by
this module's own design, not a gap introduced by an insufficiently wide palette the way the
other two sweeps' had. See ``tests/test_match.py``'s ``TestD1FixpointTerminationSoundness``
and ``TestStructuralSatisfiabilityOfEveryMatch``, ``tests/test_fusion_properties.py``'s
``TestBindingsSubstitutionIsCleanAndOracleEqual``/``TestStructuralSatisfiabilityAtScale``, and
``test_phase5_certificate_sweep.py``'s widened ``_SURVIVING_PALETTE`` for the harnesses that
actually cover the symbolic/deferred/binding space this one does not.
"""

from __future__ import annotations

import functools
import itertools
from unittest.mock import patch

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER, GeneratorType
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import validate
from qufzx.rewrite import match as match_module
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import find_matches
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import EqualityMode, compare
from qufzx.semantics.contract_numeric import ContractSizeError
from qufzx.semantics.denote import DenoteError, denote

_LEG_COUNTS = (0, 1, 2)
_COLORS = (Z_SPIDER, X_SPIDER)
_D_VALUES = (2, 3)
_PHASE_TURNS = sp.Rational(1, 3)

# See the module docstring: the 2-wire arm at d=3 is the single most expensive slice, and is
# the one this task's own instructions permit subsampling away to keep the whole sweep under
# about a minute.
_SKIP_TWO_WIRE_AT_D3 = True

_NodeShape = tuple[int, int, bool]  # (n_in, n_out, phase_present)


def _node_shapes() -> tuple[_NodeShape, ...]:
    return tuple(
        (n_in, n_out, phase_present)
        for n_in in _LEG_COUNTS
        for n_out in _LEG_COUNTS
        for phase_present in (False, True)
    )


def _build_node(
    diagram: Diagram, color: GeneratorType, shape: _NodeShape, dim: Dim
) -> list[PortRef]:
    n_in, n_out, phase_present = shape
    phase: PhaseVector | None = None
    if phase_present:
        phase = PhaseVector(dim, {1: Phase.turns(_PHASE_TURNS)})
    elif n_in == 0 and n_out == 0:
        # No legs and no phase leaves nothing to carry the node's own dimension at all --
        # give it an explicit zero phase, mirroring every other generator in this suite and
        # rules_library.py's own "no legs survive" corner case for a merged node.
        phase = PhaseVector(dim, {})
    node_id = diagram.add_node(
        color, input_dims=[dim] * n_in, output_dims=[dim] * n_out, phase=phase
    )
    ports = [PortRef(node_id, Direction.INPUT, i) for i in range(n_in)]
    ports += [PortRef(node_id, Direction.OUTPUT, i) for i in range(n_out)]
    return ports


@functools.cache
def _wiring_templates(
    n_in_a: int, n_out_a: int, n_in_b: int, n_out_b: int, skip_two_wire: bool
) -> tuple[tuple[tuple[PortRef, PortRef], ...], ...]:
    """Every distinct 1-wire and (unless ``skip_two_wire``) 2-wire matching from A to B.

    Placeholder ``PortRef``\\ s (node id 0 for A, 1 for B) -- this depends only on each
    node's leg counts, never on its colour, phase, or ``d``, so it is cached per leg-count
    quadruple (9 x 9 = 81 distinct calls total, reused across every colour/phase/``d``
    combination that shares the same shape) rather than recomputed per diagram. No port is
    ever reused across two wires in the same wiring, matching the diagram model (a port
    carries at most one wire in a well-formed diagram).
    """
    node_a_id = NodeId(0)
    node_b_id = NodeId(1)
    ports_a = [PortRef(node_a_id, Direction.INPUT, i) for i in range(n_in_a)]
    ports_a += [PortRef(node_a_id, Direction.OUTPUT, i) for i in range(n_out_a)]
    ports_b = [PortRef(node_b_id, Direction.INPUT, i) for i in range(n_in_b)]
    ports_b += [PortRef(node_b_id, Direction.OUTPUT, i) for i in range(n_out_b)]

    wirings: list[tuple[tuple[PortRef, PortRef], ...]] = [
        ((a, b),) for a in ports_a for b in ports_b
    ]
    if not skip_two_wire and len(ports_a) >= 2 and len(ports_b) >= 2:
        for pair_a in itertools.combinations(ports_a, 2):
            for pair_b in itertools.combinations(ports_b, 2):
                for perm_b in itertools.permutations(pair_b):
                    wirings.append(((pair_a[0], perm_b[0]), (pair_a[1], perm_b[1])))
    return tuple(wirings)


def _translate_wiring(
    template: tuple[tuple[PortRef, PortRef], ...], ports_a: list[PortRef], ports_b: list[PortRef]
) -> tuple[tuple[PortRef, PortRef], ...]:
    """A template wiring (placeholder node ids 0/1) translated onto real, live ports.

    Translation is by position (``(direction, index)`` uniquely identifies a port within
    one node), since the placeholder and real port lists are built in the exact same order
    by :func:`_build_node`/``_wiring_templates``'s own construction.
    """

    def _lookup(ref: PortRef, ports: list[PortRef]) -> PortRef:
        return next(p for p in ports if p.direction is ref.direction and p.index == ref.index)

    return tuple(
        (_lookup(ref_a, ports_a), _lookup(ref_b, ports_b)) for ref_a, ref_b in template
    )


def _is_cleanly_contractible(diagram: Diagram) -> bool:
    report = validate(diagram)
    return report.is_valid and not report.deferred


class TestExhaustiveSpiderFusionOracleSweep:
    """The full B3 sweep. One test, so a single wall-clock/comparison-count report covers it."""

    def test_every_match_in_the_finite_space_agrees_with_the_oracle(self) -> None:
        # Phase 5 post-closing audit round 18, Defect 3: _verify_fixpoint_closure's own
        # docstring claims the post-loop closure re-check it performs is unreachable (every
        # one of its own checks was already verified, this same pass, by the time the loop
        # reaches its convergence break) -- but a claim of unreachability is worth nothing
        # if nothing ever actually confirms it holds at the scale this sweep exercises.
        # Wrapping (not merely mocking away) the real function lets this test both count how
        # many times it fires and assert every single call returned True, over the entire
        # exhaustive finite space this module enumerates -- the guard is *not* a no-op here:
        # every fusion match in the whole sweep runs through it, exactly the leg/wire
        # topology space this module exists to cover exhaustively.
        closure_results: list[bool] = []
        real_closure = match_module._verify_fixpoint_closure

        def _wrapped_closure(*args: object, **kwargs: object) -> bool:
            result = real_closure(*args, **kwargs)  # type: ignore[arg-type]
            closure_results.append(result)
            return result

        with patch.object(match_module, "_verify_fixpoint_closure", _wrapped_closure):
            self._run_sweep()

        assert closure_results, (
            "_verify_fixpoint_closure was never called at all -- this instrumentation "
            "would then be vacuously proving nothing"
        )
        assert all(closure_results), (
            f"_verify_fixpoint_closure returned False {closure_results.count(False)} "
            f"time(s) out of {len(closure_results)} calls across the exhaustive sweep -- "
            "its own docstring's unreachability claim does not actually hold"
        )

    def _run_sweep(self) -> None:
        shapes = _node_shapes()
        total_diagrams = 0
        total_matches = 0
        total_comparisons = 0
        size_skips = 0

        for d_value in _D_VALUES:
            dim = Dim.concrete(d_value)
            skip_two_wire = _SKIP_TWO_WIRE_AT_D3 and d_value == 3
            for color_a, color_b in itertools.product(_COLORS, _COLORS):
                for shape_a, shape_b in itertools.product(shapes, shapes):
                    templates = _wiring_templates(
                        shape_a[0], shape_a[1], shape_b[0], shape_b[1], skip_two_wire
                    )
                    for template in templates:
                        diagram = Diagram()
                        ports_a = _build_node(diagram, color_a, shape_a, dim)
                        ports_b = _build_node(diagram, color_b, shape_b, dim)
                        wiring = _translate_wiring(template, ports_a, ports_b)
                        for ref_a, ref_b in wiring:
                            diagram.add_wire(ref_a, ref_b)
                        wired = frozenset(ref for pair in wiring for ref in pair)
                        all_ports = (*ports_a, *ports_b)
                        diagram.set_boundary_inputs(
                            [
                                ref
                                for ref in all_ports
                                if ref not in wired and ref.direction is Direction.INPUT
                            ]
                        )
                        diagram.set_boundary_outputs(
                            [
                                ref
                                for ref in all_ports
                                if ref not in wired and ref.direction is Direction.OUTPUT
                            ]
                        )
                        total_diagrams += 1
                        assert _is_cleanly_contractible(diagram), (
                            "an exhaustively-constructed, fully concrete diagram must "
                            "always be cleanly contractible"
                        )

                        # Round 20, Task 9: validate(d).is_valid must imply every node in d
                        # is denotable -- qufzx.diagram.validate's NODE_DIMENSION_UNDETERMINED
                        # check exists specifically so this holds, rather than resting on one
                        # builder's (rules_library's) good behaviour never producing the gap.
                        # Every diagram in this exhaustive space is already asserted valid
                        # (via _is_cleanly_contractible, above) at fully concrete dimensions,
                        # so this is the natural place to also confirm denote() never raises
                        # for a node such a diagram contains -- covering every leg-count x
                        # phase-presence x colour x d combination this module enumerates, not
                        # a single hand-picked shape.
                        for node in diagram.nodes.values():
                            try:
                                denote(node)
                            except DenoteError as exc:  # pragma: no cover - invariant guard
                                raise AssertionError(
                                    f"validate(diagram).is_valid but denote() raised for "
                                    f"node {node.id!r} ({node.generator_type.name}, "
                                    f"{node.num_inputs} in / {node.num_outputs} out, "
                                    f"phase={node.phase!r}): {exc}"
                                ) from exc

                        for match in find_matches(diagram):
                            total_matches += 1
                            result = apply(diagram, SPIDER_FUSION, match)
                            post = result.diagram
                            try:
                                comparison = compare(diagram, post, {})
                            except ContractSizeError:
                                size_skips += 1
                                continue
                            assert comparison.mode is EqualityMode.EXACT
                            assert comparison.matched, (
                                f"oracle mismatch: colors=({color_a.name}, {color_b.name}), "
                                f"shapes=({shape_a}, {shape_b}), d={d_value}, "
                                f"wiring={wiring}: {comparison.reason}"
                            )
                            total_comparisons += 1

        assert total_diagrams > 1000, (
            f"only {total_diagrams} diagrams were constructed; the exhaustive space "
            "collapsed unexpectedly small"
        )
        assert total_matches > 0, "the exhaustive space never produced a single fusion match"
        assert size_skips == 0, (
            f"{size_skips} comparison(s) were skipped for ContractSizeError; the exhaustive "
            "space at d in {2, 3} with <=2 legs per side should never be too large to "
            "densely contract"
        )
        assert total_comparisons >= 1000, (
            f"only {total_comparisons} oracle comparisons actually executed; the exhaustive "
            "sweep may be silently skipping most of its matches"
        )
