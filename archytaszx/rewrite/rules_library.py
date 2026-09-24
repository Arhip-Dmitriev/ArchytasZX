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

"""Concrete rewrite rules, starting with spider fusion, each recording its exact scalar.

Phase 5 registers one rule, :data:`SPIDER_FUSION`; Phase 11 adds :data:`IDENTITY_REMOVAL`,
:data:`TRIANGLE_INVERSE_CANCELLATION`, :data:`STATE_COPY`, :data:`HOPF`, :data:`BIALGEBRA`
and :data:`FOURIER_STATE_COLOR_CHANGE`, each carrying its own scalar derivation.
:data:`RULES` and :func:`lookup_rule`
resolve a :class:`~archytaszx.rewrite.engine.RewriteStep`'s ``rule_name`` back to its
:class:`~archytaszx.rewrite.rule.Rule`, keeping :mod:`archytaszx.rewrite.engine` generic.

Scalar. Same-color fusion across one wire introduces no factor in either wire shape
condition 4 (``consumed_wire_direction_permitted_for_color``) permits, so both land on
:meth:`~archytaszx.algebra.scalar.Scalar.one`:

* Alternating output-to-input, either color. Z: both spiders are diagonal with entry
  ``e^{i*angle(k)}`` at the all-axes-``k`` position, so contracting an output leg against
  an input leg identifies their ``k``. X: ``X_{m->n} = F^{ox n} . Z_{m->n} .
  (conj(F))^{ox m}``, and the wire contracts an ``F`` against a ``conj(F)`` on the shared
  axis, which cancel to the identity, ``F`` being unitary and symmetric.
* Same-direction, Z only. ``_z_tensor`` is diagonal in every axis and
  :mod:`archytaszx.semantics.contract_numeric` applies no conjugation at contraction time, so
  the same index ``k`` is identified. A same-direction X wire contracts ``F`` against ``F``,
  giving a permutation matrix, and is not this rule.

Neither derivation depends on the consumed wire being the pair's only wire: a further wire
between the same nodes is never contracted by this rule.

Merged leg ordering, a choice rather than a derivation: the merged node's inputs are A's
surviving inputs in original index order, then B's; outputs likewise. "A" is the lower
:class:`~archytaszx.diagram.graph.NodeId`.

Merged dimension. Every surviving port is built at
:attr:`~archytaszx.rewrite.match.FusionMatch.shared_dim`, never its own original ``Dim``.
Condition 6 unifies every surviving leg against the resolved ``shared_dim`` before a match
is returned, and this builder calls the same
:func:`~archytaszx.rewrite.match.resolve_fusion_match` fresh against the diagram it was handed.

Fusion may fire on a ``DEFERRED`` dimension pair, though FULL_PLAN.md's Phase 5 states the
pattern as spiders "sharing a dimension". A ``d``/``d*e`` leg pair is already legal,
non-hard-error input under ``ALL_LEGS_EQUAL``
(:class:`~archytaszx.diagram.validate.IssueKind.DIMENSION_DEFERRED`), and the assumption is
recorded either way. Two consequences: a neighbouring wire that was an exact match before
the fusion may be merely deferred after, which :mod:`archytaszx.rewrite.engine`'s step-8
relative postcondition permits; and a surviving leg on a boundary is rebuilt at
``shared_dim``, so the finished diagram's interface holds only under the same recorded
assumption.

Recorded is not satisfiable. A surviving leg of ``d**2`` forced onto ``shared_dim = d`` is
a legal ``DEFERRED`` unify recording ``d**2 == d``, which holds over the positive integers
only at ``d = 1``. Discharging such a constraint is Phase 10's job.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import PhaseDomainError, PhaseVector
from archytaszx.algebra.scalar import Scalar
from archytaszx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef, Wire
from archytaszx.rewrite.match import (
    BIALGEBRA_SIDE_CONDITIONS,
    CAP_SIDE_CONDITIONS,
    FOURIER_SIDE_CONDITIONS,
    FOURIER_STATE_SIDE_CONDITIONS,
    FUSION_SIDE_CONDITIONS,
    HOPF_SIDE_CONDITIONS,
    IDENTITY_SIDE_CONDITIONS,
    STATE_COPY_SIDE_CONDITIONS,
    TRIANGLE_SIDE_CONDITIONS,
    BialgebraMatch,
    BialgebraPattern,
    CapMatch,
    CapPattern,
    FourierCancellationPattern,
    FourierMatch,
    FourierStateColorChangePattern,
    FourierStateMatch,
    FusionMatch,
    FusionPattern,
    HopfMatch,
    HopfPattern,
    IdentityMatch,
    IdentityRemovalPattern,
    StateCopyMatch,
    StateCopyPattern,
    TriangleInverseCancellationPattern,
    TriangleMatch,
    find_bialgebra_matches,
    find_cap_matches,
    find_fourier_matches,
    find_fourier_state_matches,
    find_hopf_matches,
    find_identity_matches,
    find_state_copy_matches,
    find_triangle_matches,
    innermost_node_scope_box,
    reattach_phase,
    resolve_fusion_match,
)
from archytaszx.rewrite.rule import (
    BuildResult,
    DimensionGuard,
    DimensionGuardKind,
    Match,
    Quantifiers,
    RewriteDomainError,
    RewriteGrammarError,
    Rule,
    check_side_condition_coverage,
)


def _surviving_legs(
    node_id: NodeId, node: Node, direction: Direction, consumed_ref: PortRef
) -> list[tuple[PortRef, Port]]:
    """Every ``(PortRef, Port)`` of ``node`` on ``direction`` except ``consumed_ref``, in order."""
    legs = node.legs(direction)
    surviving = []
    for index, port in enumerate(legs):
        ref = PortRef(node_id, direction, index)
        if ref == consumed_ref:
            continue
        surviving.append((ref, port))
    return surviving


def _over_shared_dim(
    phase: PhaseVector | None, shared_dim: Dim, bindings: Mapping[str, Dim]
) -> tuple[PhaseVector, Mapping[str, Dim]]:
    """``phase``'s entries, with ``bindings`` substituted in, reattached to ``shared_dim``
    -- or an all-zero vector over ``shared_dim`` if ``phase`` is absent.

    Delegates to :func:`archytaszx.rewrite.match.reattach_phase`, returning its vector and the
    subset of ``bindings`` it substituted into an entry's value. Raises
    :class:`RewriteDomainError`, not :class:`~archytaszx.algebra.phase.PhaseDomainError`, if an
    entry index falls outside ``shared_dim``'s range.
    """
    if phase is None:
        return PhaseVector(shared_dim, {}), MappingProxyType({})
    try:
        return reattach_phase(phase, shared_dim, bindings)
    except PhaseDomainError as exc:
        raise RewriteDomainError(
            f"spider_fusion cannot reattach a phase vector to shared dimension {shared_dim}: {exc}"
        ) from exc


def _merged_phase(
    node_a: Node,
    node_b: Node,
    a_id: NodeId,
    b_id: NodeId,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
    *,
    any_legs_survive: bool,
) -> tuple[PhaseVector | None, Mapping[NodeId, Mapping[str, Dim]]]:
    """The merged node's phase: componentwise sum, both operands read over ``shared_dim``.

    A `None` phase on both sides stays `None`, except when ``any_legs_survive`` is `False`: a
    merged node with no surviving legs has no port to carry ``shared_dim``, so an all-zero
    ``PhaseVector(shared_dim, {})`` is returned instead. Otherwise both operands are read via
    :func:`_over_shared_dim` before adding, since
    :meth:`~archytaszx.algebra.phase.PhaseVector.__add__` demands exactly equal ``Dim``\\ s.

    The second return value is every node whose phase had a binding substituted into an
    entry, keyed by node id.
    """
    if node_a.phase is None and node_b.phase is None:
        if not any_legs_survive:
            return PhaseVector(shared_dim, {}), {}
        return None, {}
    phase_a, applied_a = _over_shared_dim(node_a.phase, shared_dim, bindings)
    phase_b, applied_b = _over_shared_dim(node_b.phase, shared_dim, bindings)
    substitutions: dict[NodeId, Mapping[str, Dim]] = {}
    if applied_a:
        substitutions[a_id] = applied_a
    if applied_b:
        substitutions[b_id] = applied_b
    return phase_a + phase_b, substitutions


PRIME_DIMENSION_GUARDS: tuple[DimensionGuard, ...] = (DimensionGuard(DimensionGuardKind.PRIME),)
"""The guard tuple :data:`Z_FUSION_PRIME_D` declares: the shared dimension must be prime."""


def spider_fusion_builder(
    diagram: Diagram,
    match: Match,
    *,
    dimension_guards: tuple[DimensionGuard, ...] = (),
    context: str = "spider_fusion",
) -> BuildResult:
    """The right-hand side of :data:`SPIDER_FUSION`: merge the two matched spiders.

    Mutates ``diagram`` in place by adding the merged node (module docstring: leg ordering,
    scalar, merged dimension) and returns the :class:`BuildResult`
    :mod:`archytaszx.rewrite.engine` needs to splice it in; never removes the matched nodes or
    touches any wire or boundary entry.

    Trusts nothing about ``match`` for graph surgery until it has been re-derived. In order:

    1. ``isinstance(match, FusionMatch)``.
    2. :func:`~archytaszx.rewrite.rule.check_side_condition_coverage` against the module-level
       :data:`FUSION_SIDE_CONDITIONS`; this builder is reachable directly.
    3. :func:`~archytaszx.rewrite.match.resolve_fusion_match`, called fresh against ``diagram``.
       Everything downstream builds from ``resolution``'s fields, never ``match``'s.
    4. ``match.shared_dim``, ``bindings``, ``dimension_constraints`` and
       ``side_condition_outcomes``, each checked for exact agreement with ``resolution``'s.
       ``apply`` records the match's own copies, so this equality is what makes them the
       certificate's ground truth.

    Raises :class:`~archytaszx.rewrite.rule.RewriteDomainError` on any disagreement at step 3 or
    4, and :class:`~archytaszx.rewrite.rule.RewriteGrammarError` for a foreign match type or a
    structurally malformed one (equal node ids, a node absent from ``diagram``, a wire not
    incident on both or not in ``diagram.wires``).
    """
    if not isinstance(match, FusionMatch):
        raise RewriteGrammarError(f"{context} requires a FusionMatch, got {type(match).__name__}")
    # The module-level constant, not spider_fusion_builder.side_conditions, which would be a
    # self-reference to this function's own global name. That attribute exists solely for
    # Rule.__post_init__.
    check_side_condition_coverage(match, FUSION_SIDE_CONDITIONS, context)

    resolution = resolve_fusion_match(
        diagram, match.a_id, match.b_id, match.wire, dimension_guards=dimension_guards
    )
    if not resolution.passed:
        failed = [outcome.name for outcome in resolution.outcomes if not outcome.passed]
        raise RewriteDomainError(
            f"{context}: match at ({match.a_id!r}, {match.b_id!r}) over wire "
            f"{match.wire!r} fails side condition(s) {failed} when re-verified fresh "
            "against the diagram it is being applied to; match.side_condition_outcomes "
            "claimed every condition passed, but a match's own outcomes are never taken "
            "on faith for graph surgery -- see resolve_fusion_match"
        )
    assert resolution.shared_dim is not None  # invariant: passed implies shared_dim is set
    if match.shared_dim != resolution.shared_dim:
        raise RewriteDomainError(
            f"{context}: match.shared_dim {match.shared_dim!r} disagrees with the "
            f"shared dimension {resolution.shared_dim!r} resolve_fusion_match derives "
            "fresh from the diagram for this wire; a match's own shared_dim is never "
            "trusted for graph surgery without this agreement"
        )
    if dict(match.bindings) != dict(resolution.bindings):
        raise RewriteDomainError(
            f"{context}: match.bindings {dict(match.bindings)!r} disagrees with the "
            f"bindings {dict(resolution.bindings)!r} resolve_fusion_match derives fresh "
            "from the diagram for this wire"
        )
    # apply records these two fields verbatim onto RewriteStep, so a fabricated pair must be
    # rejected here or the certificate records a claim the rewrite never made.
    if match.dimension_constraints != resolution.dimension_constraints:
        raise RewriteDomainError(
            f"{context}: match.dimension_constraints {match.dimension_constraints!r} "
            f"disagrees with {resolution.dimension_constraints!r}, which "
            "resolve_fusion_match derives fresh from the diagram for this wire -- a "
            "match's own dimension_constraints is never trusted for the certificate "
            "without this agreement"
        )
    if match.side_condition_outcomes != resolution.outcomes:
        raise RewriteDomainError(
            f"{context}: match.side_condition_outcomes disagrees with the outcomes "
            "resolve_fusion_match derives fresh from the diagram for this wire -- a "
            "match's own side_condition_outcomes is never trusted for the certificate "
            "without this agreement"
        )

    node_a = diagram.nodes[match.a_id]
    node_b = diagram.nodes[match.b_id]
    wire = match.wire
    consumed_ref_a = wire.a if wire.a.node_id == match.a_id else wire.b
    consumed_ref_b = wire.b if wire.a.node_id == match.a_id else wire.a

    surviving_inputs_a = _surviving_legs(match.a_id, node_a, Direction.INPUT, consumed_ref_a)
    surviving_outputs_a = _surviving_legs(match.a_id, node_a, Direction.OUTPUT, consumed_ref_a)
    surviving_inputs_b = _surviving_legs(match.b_id, node_b, Direction.INPUT, consumed_ref_b)
    surviving_outputs_b = _surviving_legs(match.b_id, node_b, Direction.OUTPUT, consumed_ref_b)

    merged_inputs = surviving_inputs_a + surviving_inputs_b
    merged_outputs = surviving_outputs_a + surviving_outputs_b

    any_legs_survive = bool(merged_inputs or merged_outputs)
    merged_phase, phase_substitutions = _merged_phase(
        node_a,
        node_b,
        match.a_id,
        match.b_id,
        resolution.shared_dim,
        resolution.bindings,
        any_legs_survive=any_legs_survive,
    )

    # node_a.generator_type alone: resolution.passed confirmed it equals node_b's.
    new_node_id = diagram.add_node(
        node_a.generator_type,
        input_dims=[resolution.shared_dim] * len(merged_inputs),
        output_dims=[resolution.shared_dim] * len(merged_outputs),
        phase=merged_phase,
    )

    port_mapping: dict[PortRef, PortRef] = {}
    for new_index, (old_ref, _) in enumerate(merged_inputs):
        port_mapping[old_ref] = PortRef(new_node_id, Direction.INPUT, new_index)
    for new_index, (old_ref, _) in enumerate(merged_outputs):
        port_mapping[old_ref] = PortRef(new_node_id, Direction.OUTPUT, new_index)

    # Bang box "left intact" (Phase 7): condition 6 already required both matched nodes
    # to share one innermost node-scope box, or neither to have one. If they do, that
    # box's scope drops the two consumed nodes and gains the merged one, in place --
    # multiplicity, and every other field, untouched, since this fusion is a single
    # symbolic rewrite standing for one fusion per future instantiated copy.
    enclosing_box = innermost_node_scope_box(diagram, match.a_id)
    if enclosing_box is not None:
        box = diagram.bang_boxes[enclosing_box]
        new_scope = (box.node_scope - {match.a_id, match.b_id}) | {new_node_id}
        diagram.set_bang_box_node_scope(enclosing_box, frozenset(new_scope))

    # A port-scope box names a leg of one of the merged nodes; that leg survives on the
    # merged node, so the box follows it through the same port_mapping the boundary does.
    # A consumed port is never boxed: a port-scope box's port must be a boundary slot, and
    # condition 5 already required neither consumed port to be one.
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if not any(ref in port_mapping for ref in box.port_scope):
            continue
        diagram.set_bang_box_port_scope(
            box_id, frozenset(port_mapping.get(ref, ref) for ref in box.port_scope)
        )

    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_node_id,),
        consumed_node_ids=(match.a_id, match.b_id),
        consumed_wires=(wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
        phase_substitutions=MappingProxyType(phase_substitutions),
    )


def z_fusion_prime_d_builder(diagram: Diagram, match: Match) -> BuildResult:
    """:func:`spider_fusion_builder` under :data:`PRIME_DIMENSION_GUARDS`."""
    return spider_fusion_builder(
        diagram,
        match,
        dimension_guards=PRIME_DIMENSION_GUARDS,
        context="z_fusion_prime_d",
    )


spider_fusion_builder.side_conditions = FUSION_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The single declared side-condition tuple this builder is meant to be paired with.

Read only by :class:`~archytaszx.rewrite.rule.Rule`'s constructor-time consistency check, which
makes two contradicting tuples for one builder impossible to construct. The builder's own
body reads the module-level constant.
"""


SPIDER_FUSION = Rule(
    name="spider_fusion",
    pattern=FusionPattern(),
    builder=spider_fusion_builder,
    side_conditions=FUSION_SIDE_CONDITIONS,
    quantifiers=Quantifiers(
        leg_counts=("m_a", "n_a", "m_b", "n_b"),
        dimensions=("d",),
    ),
    scalar_introduced=Scalar.one(),
)
"""Same-color spider fusion across one wire -- any direction for Z, output-to-input only for X.

Any further wire joining the same pair is not consumed: it survives as a self-loop on the
merged spider (condition 3 in :mod:`archytaszx.rewrite.match`).
"""


z_fusion_prime_d_builder.side_conditions = FUSION_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


Z_FUSION_PRIME_D = Rule(
    name="z_fusion_prime_d",
    pattern=FusionPattern(dimension_guards=PRIME_DIMENSION_GUARDS),
    builder=z_fusion_prime_d_builder,
    side_conditions=FUSION_SIDE_CONDITIONS,
    quantifiers=Quantifiers(
        leg_counts=("m_a", "n_a", "m_b", "n_b"),
        dimensions=("d",),
    ),
    scalar_introduced=Scalar.one(),
    dimension_guards=PRIME_DIMENSION_GUARDS,
)
"""Spider fusion restricted to a prime shared dimension.

Same builder and scalar as :data:`SPIDER_FUSION`; a composite or non-concrete shared
dimension is not a match.
"""


def lookup_rule(name: str) -> Rule:
    """Resolve a rule name back to its :class:`Rule`.

    Raises :class:`~archytaszx.rewrite.rule.RewriteGrammarError` if ``name`` is not in
    :data:`RULES`.
    """
    try:
        return RULES[name]
    except KeyError:
        raise RewriteGrammarError(f"no such rule: {name!r}") from None


def fourier_cancellation_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`FOURIER_CANCELLATION`: one identity spider for four F boxes.

    Adds a phaseless one-in-one-out Z spider (the identity on a wire) and maps the chain's
    free input and output onto its legs; never removes the matched nodes or touches a wire.

    Trusts nothing about ``match`` for graph surgery until it has been re-derived: the match
    must be among those :func:`~archytaszx.rewrite.match.find_fourier_matches` finds afresh in
    ``diagram``, which settles ``node_ids``, ``wires``, ``shared_dim``,
    ``dimension_constraints`` and ``side_condition_outcomes`` in one equality. This builder
    is reachable directly, so a fabricated match must not reach surgery.
    """
    if not isinstance(match, FourierMatch):
        raise RewriteGrammarError(
            f"fourier_cancellation_builder requires a FourierMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, FOURIER_SIDE_CONDITIONS, "fourier_cancellation_builder")
    if match not in find_fourier_matches(diagram):
        raise RewriteDomainError(
            "fourier_cancellation_builder: the match is not among those rediscovered in this "
            "diagram, so it is not evidence of an F^4 chain"
        )
    for node_id in match.node_ids:
        node = diagram.nodes.get(node_id)
        if node is None or node.generator_type.name != FOURIER_BOX.name:
            raise RewriteGrammarError(
                f"fourier_cancellation_builder: node {node_id!r} is not an F box in this diagram"
            )
    dim = match.shared_dim
    enclosing_boxes = {innermost_node_scope_box(diagram, node_id) for node_id in match.node_ids}
    if len(enclosing_boxes) != 1:
        found = sorted(box for box in enclosing_boxes if box is not None)
        raise RewriteDomainError(
            "fourier_cancellation_builder: the four F boxes do not share one innermost "
            f"node-scope bang box (found {found!r}); the chain must be wholly inside one "
            "box or wholly outside every box"
        )
    new_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    first, last = match.node_ids[0], match.node_ids[-1]
    port_mapping = {
        PortRef(first, Direction.INPUT, 0): PortRef(new_id, Direction.INPUT, 0),
        PortRef(last, Direction.OUTPUT, 0): PortRef(new_id, Direction.OUTPUT, 0),
    }

    # Bang box "left intact" (Phase 7), as in spider_fusion_builder: the enclosing box drops
    # the four consumed nodes and gains the identity spider, every other field untouched.
    (enclosing_box,) = enclosing_boxes
    if enclosing_box is not None:
        box = diagram.bang_boxes[enclosing_box]
        new_scope = (box.node_scope - set(match.node_ids)) | {new_id}
        diagram.set_bang_box_node_scope(enclosing_box, frozenset(new_scope))

    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_id,),
        consumed_node_ids=tuple(match.node_ids),
        consumed_wires=tuple(match.wires),
        port_mapping=port_mapping,
        scalar_introduced=FOURIER_CANCELLATION_SCALAR,
    )


FOURIER_CANCELLATION_SCALAR = Scalar.one()
"""The exact scalar F^4 cancellation introduces: one.

The character sum contributes a factor of ``d`` twice and the four boxes contribute
``d^{-1/2}`` each, so the product is exactly one. An implementation that drops either
factor lands on ``d``, ``d^{-1}`` or ``d^{-2}`` instead, which is a wrong global factor.
"""


FOURIER_CANCELLATION = Rule(
    name="fourier_cancellation",
    pattern=FourierCancellationPattern(),
    builder=fourier_cancellation_builder,
    side_conditions=FOURIER_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=FOURIER_CANCELLATION_SCALAR,
)
"""Four Fourier boxes in series collapse to the identity wire, introducing exactly one."""


def zx_cap_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`ZX_CAP`: nothing, with ``d ** (1/2)`` recorded.

    Removes both nodes and the wire joining them, leaving whatever else the diagram holds.

    Trusts nothing about ``match`` for graph surgery until it has been re-derived: the match
    must be among those :func:`~archytaszx.rewrite.match.find_cap_matches` finds afresh in
    ``diagram``.
    """
    if not isinstance(match, CapMatch):
        raise RewriteGrammarError(f"zx_cap_builder requires a CapMatch, got {type(match).__name__}")
    check_side_condition_coverage(match, CAP_SIDE_CONDITIONS, "zx_cap_builder")
    if match not in find_cap_matches(diagram):
        raise RewriteDomainError(
            "zx_cap_builder: the match is not among those rediscovered in this diagram, so "
            "it is not evidence of a Z state wired into an X effect"
        )
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=(match.state_id, match.effect_id),
        consumed_wires=(match.wire,),
        port_mapping={},
        scalar_introduced=zx_cap_scalar(match.shared_dim),
    )


def zx_cap_scalar(dim: Dim) -> Scalar:
    """The exact scalar the cap introduces at ``dim``: ``dim ** (1/2)``."""
    return Scalar.dim_power(dim, 1, 2)


ZX_CAP = Rule(
    name="zx_cap",
    pattern=CapPattern(),
    builder=zx_cap_builder,
    side_conditions=CAP_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=zx_cap_scalar(Dim.symbol("d")),
    scalar_in_dim=zx_cap_scalar,
)
"""A phaseless Z state capped by a phaseless X effect is the empty diagram times ``d ** (1/2)``.

The X effect reads ``sum_j conj(omega_d^{j k}) / sqrt(d)``, which the character sum closes to
``sqrt(d) * [k == 0 mod d]``; summing that against the Z state's all-ones vector leaves exactly
``sqrt(d)``. Unlike the other two rules, the scalar this introduces is not one, so a dropped
factor changes the answer.
"""


def _splice_out_box(diagram: Diagram, node_ids: tuple[NodeId, ...], context: str) -> None:
    """Drop ``node_ids`` from their shared innermost node-scope bang box, if they have one."""
    boxes = {innermost_node_scope_box(diagram, node_id) for node_id in node_ids}
    if len(boxes) != 1:
        found = sorted(box for box in boxes if box is not None)
        raise RewriteDomainError(
            f"{context}: the matched nodes do not share one innermost node-scope bang box "
            f"(found {found!r})"
        )
    (box_id,) = boxes
    if box_id is None:
        return
    box = diagram.bang_boxes[box_id]
    diagram.set_bang_box_node_scope(box_id, frozenset(box.node_scope - set(node_ids)))


def _adopt_into_box(
    diagram: Diagram, consumed: tuple[NodeId, ...], created: tuple[NodeId, ...]
) -> None:
    """Replace ``consumed`` by ``created`` in their shared innermost node-scope bang box."""
    box_id = innermost_node_scope_box(diagram, consumed[0])
    if box_id is None:
        return
    box = diagram.bang_boxes[box_id]
    diagram.set_bang_box_node_scope(
        box_id, frozenset((box.node_scope - set(consumed)) | set(created))
    )


def _follow_port_scopes(diagram: Diagram, port_mapping: Mapping[PortRef, PortRef]) -> None:
    """Send every port-scope bang box entry through ``port_mapping``."""
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if not any(ref in port_mapping for ref in box.port_scope):
            continue
        diagram.set_bang_box_port_scope(
            box_id, frozenset(port_mapping.get(ref, ref) for ref in box.port_scope)
        )


def _require_rediscovered(match: Match, found: tuple[Match, ...], context: str) -> None:
    """Raise unless ``match`` is among the matches rediscovered in the diagram."""
    if match not in found:
        raise RewriteDomainError(
            f"{context}: the match is not among those rediscovered in this diagram, so it "
            "is not evidence of this rule's pattern"
        )


def identity_removal_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`IDENTITY_REMOVAL`: nothing, the wire spliced through.

    Consumes the spider and the wire on one of its legs, mapping the other leg onto that
    wire's far port.
    """
    if not isinstance(match, IdentityMatch):
        raise RewriteGrammarError(
            f"identity_removal_builder requires an IdentityMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, IDENTITY_SIDE_CONDITIONS, "identity_removal_builder")
    _require_rediscovered(match, find_identity_matches(diagram), "identity_removal_builder")
    port_mapping = {match.surviving_ref: match.far_ref}
    _splice_out_box(diagram, (match.node_id,), "identity_removal_builder")
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=(match.node_id,),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


identity_removal_builder.side_conditions = IDENTITY_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


IDENTITY_REMOVAL = Rule(
    name="identity_removal",
    pattern=IdentityRemovalPattern(),
    builder=identity_removal_builder,
    side_conditions=IDENTITY_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A phaseless one-in-one-out spider is the identity on its wire, introducing exactly one."""


def triangle_inverse_cancellation_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`TRIANGLE_INVERSE_CANCELLATION`: the bare wire.

    Consumes both boxes, the wire between them, and the wire on one outer leg, mapping the
    other outer leg onto that wire's far port.
    """
    if not isinstance(match, TriangleMatch):
        raise RewriteGrammarError(
            "triangle_inverse_cancellation_builder requires a TriangleMatch, got "
            f"{type(match).__name__}"
        )
    check_side_condition_coverage(
        match, TRIANGLE_SIDE_CONDITIONS, "triangle_inverse_cancellation_builder"
    )
    _require_rediscovered(
        match, find_triangle_matches(diagram), "triangle_inverse_cancellation_builder"
    )
    consumed = (match.first_id, match.second_id)
    port_mapping = {match.surviving_ref: match.far_ref}
    _splice_out_box(diagram, consumed, "triangle_inverse_cancellation_builder")
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=consumed,
        consumed_wires=(match.wire, match.spliced_wire),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


triangle_inverse_cancellation_builder.side_conditions = TRIANGLE_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


TRIANGLE_INVERSE_CANCELLATION = Rule(
    name="triangle_inverse_cancellation",
    pattern=TriangleInverseCancellationPattern(),
    builder=triangle_inverse_cancellation_builder,
    side_conditions=TRIANGLE_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A T and a Ti in series are the identity wire at every dimension, introducing exactly one.

``_triangle_inverse_tensor`` in :mod:`archytaszx.semantics.denote` is T's exact inverse, so
no factor appears in either order.
"""


def state_copy_scalar(dim: Dim, output_count: int) -> Scalar:
    """The exact scalar state copy introduces: ``dim ** ((1 - output_count) / 2)``."""
    return Scalar.dim_power(dim, 1 - output_count, 2)


def _state_copy_scalar_for(match: Match) -> Scalar:
    """:func:`state_copy_scalar` read off a :class:`StateCopyMatch`."""
    if not isinstance(match, StateCopyMatch):
        raise RewriteGrammarError(
            f"state_copy requires a StateCopyMatch, got {type(match).__name__}"
        )
    return state_copy_scalar(match.shared_dim, match.output_count)


def state_copy_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`STATE_COPY`: one phaseless X state per former Z output."""
    if not isinstance(match, StateCopyMatch):
        raise RewriteGrammarError(
            f"state_copy_builder requires a StateCopyMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, STATE_COPY_SIDE_CONDITIONS, "state_copy_builder")
    _require_rediscovered(match, find_state_copy_matches(diagram), "state_copy_builder")
    dim = match.shared_dim
    new_ids = tuple(
        diagram.add_node(X_SPIDER, input_dims=[], output_dims=[dim])
        for _ in range(match.output_count)
    )
    port_mapping = {
        PortRef(match.spider_id, Direction.OUTPUT, index): PortRef(new_id, Direction.OUTPUT, 0)
        for index, new_id in enumerate(new_ids)
    }
    consumed = (match.state_id, match.spider_id)
    _adopt_into_box(diagram, consumed, new_ids)
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=new_ids,
        consumed_node_ids=consumed,
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=state_copy_scalar(dim, match.output_count),
    )


state_copy_builder.side_conditions = STATE_COPY_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


STATE_COPY = Rule(
    name="state_copy",
    pattern=StateCopyPattern(),
    builder=state_copy_builder,
    side_conditions=STATE_COPY_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("n",), dimensions=("d",)),
    scalar_introduced=state_copy_scalar(Dim.symbol("d"), 0),
    scalar_in_match=_state_copy_scalar_for,
)
"""A phaseless X state copies through a phaseless Z spider into one X state per output.

``X_{0->1} = sqrt(d)|0>``, so the left side is ``sqrt(d) |0>^{ox n}`` and ``n`` fresh X
states are ``d^{n/2} |0>^{ox n}``: the factor is ``d ** ((1 - n) / 2)``, which
``scalar_introduced`` writes at ``n = 0`` and ``scalar_in_match`` evaluates per match.
"""


def _spider_without_legs(
    diagram: Diagram,
    node: Node,
    dropped_inputs: frozenset[int],
    dropped_outputs: frozenset[int],
    dim: Dim,
) -> tuple[NodeId, dict[PortRef, PortRef]]:
    """A copy of ``node`` less the named legs, and the old-to-new map of the legs kept."""
    kept_inputs = [i for i in range(node.num_inputs) if i not in dropped_inputs]
    kept_outputs = [i for i in range(node.num_outputs) if i not in dropped_outputs]
    phase = node.phase
    if phase is None and not kept_inputs and not kept_outputs:
        phase = PhaseVector(dim, {})
    new_id = diagram.add_node(
        node.generator_type,
        input_dims=[dim] * len(kept_inputs),
        output_dims=[dim] * len(kept_outputs),
        phase=phase,
    )
    mapping = {
        PortRef(node.id, Direction.INPUT, old): PortRef(new_id, Direction.INPUT, new)
        for new, old in enumerate(kept_inputs)
    }
    mapping.update(
        {
            PortRef(node.id, Direction.OUTPUT, old): PortRef(new_id, Direction.OUTPUT, new)
            for new, old in enumerate(kept_outputs)
        }
    )
    return new_id, mapping


def hopf_scalar(dim: Dim) -> Scalar:
    """The exact scalar the Hopf law introduces at ``dim``: ``dim ** -1``."""
    return Scalar.dim_power(dim, -1, 1)


def hopf_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`HOPF`: the two spiders, disconnected, each two legs lighter."""
    if not isinstance(match, HopfMatch):
        raise RewriteGrammarError(f"hopf_builder requires a HopfMatch, got {type(match).__name__}")
    check_side_condition_coverage(match, HOPF_SIDE_CONDITIONS, "hopf_builder")
    _require_rediscovered(match, find_hopf_matches(diagram), "hopf_builder")
    dim = match.shared_dim
    new_z, z_mapping = _spider_without_legs(
        diagram, diagram.nodes[match.z_id], frozenset(), frozenset(match.z_leg_indices), dim
    )
    new_x, x_mapping = _spider_without_legs(
        diagram, diagram.nodes[match.x_id], frozenset(match.x_leg_indices), frozenset(), dim
    )
    port_mapping = {**z_mapping, **x_mapping}
    consumed = (match.z_id, match.x_id, *match.fourier_ids)
    _adopt_into_box(diagram, consumed, (new_z, new_x))
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_z, new_x),
        consumed_node_ids=consumed,
        consumed_wires=match.wires,
        port_mapping=port_mapping,
        scalar_introduced=hopf_scalar(dim),
    )


hopf_builder.side_conditions = HOPF_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


HOPF = Rule(
    name="hopf",
    pattern=HopfPattern(),
    builder=hopf_builder,
    side_conditions=HOPF_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("m", "n", "p", "q"), dimensions=("d",)),
    scalar_introduced=hopf_scalar(Dim.symbol("d")),
    scalar_in_dim=hopf_scalar,
)
"""A Z and an X joined by a plain wire and by an F^2 antipode wire fall apart, times ``d ** -1``.

The two X legs contract against ``|k>`` and ``|-k>``, whose two ``conj(F)`` entries multiply
to ``d ** -1`` independently of every index, leaving the two spiders with their remaining
legs. Without the F pair the same double edge contracts to ``d ** (-1/2) [o == 2 i]``, which
is not disconnected, so the plain double edge is not a match.
"""


def bialgebra_scalar(dim: Dim) -> Scalar:
    """The exact scalar the bialgebra law introduces at ``dim``: ``dim ** (1/2)``."""
    return Scalar.dim_power(dim, 1, 2)


def bialgebra_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`BIALGEBRA`: two Z_{1->2} and two X_{2->1}, each Z feeding both.

    The only builder reporting :attr:`~archytaszx.rewrite.rule.BuildResult.new_wires`.
    """
    if not isinstance(match, BialgebraMatch):
        raise RewriteGrammarError(
            f"bialgebra_builder requires a BialgebraMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, BIALGEBRA_SIDE_CONDITIONS, "bialgebra_builder")
    _require_rediscovered(match, find_bialgebra_matches(diagram), "bialgebra_builder")
    dim = match.shared_dim
    z_ids = tuple(
        diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim]) for _ in range(2)
    )
    x_ids = tuple(
        diagram.add_node(X_SPIDER, input_dims=[dim, dim], output_dims=[dim]) for _ in range(2)
    )
    port_mapping = {
        PortRef(match.x_id, Direction.INPUT, 0): PortRef(z_ids[0], Direction.INPUT, 0),
        PortRef(match.x_id, Direction.INPUT, 1): PortRef(z_ids[1], Direction.INPUT, 0),
        PortRef(match.z_id, Direction.OUTPUT, 0): PortRef(x_ids[0], Direction.OUTPUT, 0),
        PortRef(match.z_id, Direction.OUTPUT, 1): PortRef(x_ids[1], Direction.OUTPUT, 0),
    }
    new_wires = tuple(
        Wire(
            PortRef(z_ids[z_index], Direction.OUTPUT, x_index),
            PortRef(x_ids[x_index], Direction.INPUT, z_index),
        )
        for z_index in range(2)
        for x_index in range(2)
    )
    consumed = (match.x_id, match.z_id)
    _adopt_into_box(diagram, consumed, z_ids + x_ids)
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=z_ids + x_ids,
        consumed_node_ids=consumed,
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=bialgebra_scalar(dim),
        new_wires=new_wires,
    )


bialgebra_builder.side_conditions = BIALGEBRA_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


BIALGEBRA = Rule(
    name="bialgebra",
    pattern=BialgebraPattern(),
    builder=bialgebra_builder,
    side_conditions=BIALGEBRA_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=bialgebra_scalar(Dim.symbol("d")),
    scalar_in_dim=bialgebra_scalar,
)
"""X_{2->1} into Z_{1->2} crosses over into two Z_{1->2} and two X_{2->1}, times ``d ** (1/2)``.

The left side is ``d ** (-1/2) [o1 == o2 == i1 + i2]`` and the right ``d ** -1 [o1 == i1 +
i2] [o2 == i1 + i2]``. This rule grows the diagram from two nodes to four, so it is excluded
from :func:`~archytaszx.rewrite.engine.toward_normal_form`'s default set.
"""


def fourier_state_color_change_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`FOURIER_STATE_COLOR_CHANGE`: a phaseless X state or effect."""
    if not isinstance(match, FourierStateMatch):
        raise RewriteGrammarError(
            "fourier_state_color_change_builder requires a FourierStateMatch, got "
            f"{type(match).__name__}"
        )
    check_side_condition_coverage(
        match, FOURIER_STATE_SIDE_CONDITIONS, "fourier_state_color_change_builder"
    )
    _require_rediscovered(
        match, find_fourier_state_matches(diagram), "fourier_state_color_change_builder"
    )
    dim = match.shared_dim
    direction = Direction.OUTPUT if match.is_state else Direction.INPUT
    new_id = diagram.add_node(
        X_SPIDER,
        input_dims=[] if match.is_state else [dim],
        output_dims=[dim] if match.is_state else [],
    )
    port_mapping = {match.free_ref: PortRef(new_id, direction, 0)}
    consumed = (match.spider_id, match.fourier_id)
    _adopt_into_box(diagram, consumed, (new_id,))
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_id,),
        consumed_node_ids=consumed,
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


fourier_state_color_change_builder.side_conditions = FOURIER_STATE_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


FOURIER_STATE_COLOR_CHANGE = Rule(
    name="fourier_state_color_change",
    pattern=FourierStateColorChangePattern(),
    builder=fourier_state_color_change_builder,
    side_conditions=FOURIER_STATE_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""An F box on a phaseless Z state's leg is a phaseless X state, introducing exactly one.

``F . (sum_k |k>) = sqrt(d) |0> = X_{0->1}`` exactly, and the dual form -- an F box on a
phaseless Z effect's leg -- carries the same unit factor.
"""

RULES: Mapping[str, Rule] = MappingProxyType(
    {
        SPIDER_FUSION.name: SPIDER_FUSION,
        Z_FUSION_PRIME_D.name: Z_FUSION_PRIME_D,
        FOURIER_CANCELLATION.name: FOURIER_CANCELLATION,
        ZX_CAP.name: ZX_CAP,
        IDENTITY_REMOVAL.name: IDENTITY_REMOVAL,
        TRIANGLE_INVERSE_CANCELLATION.name: TRIANGLE_INVERSE_CANCELLATION,
        STATE_COPY.name: STATE_COPY,
        HOPF.name: HOPF,
        BIALGEBRA.name: BIALGEBRA,
        FOURIER_STATE_COLOR_CHANGE.name: FOURIER_STATE_COLOR_CHANGE,
    }
)
"""Every rule this module registers, keyed by :attr:`~archytaszx.rewrite.rule.Rule.name`.

A ``MappingProxyType``, so a caller cannot mutate the registry through it.
"""
