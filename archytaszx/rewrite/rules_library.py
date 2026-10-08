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
and :data:`FOURIER_STATE_COLOR_CHANGE`, each carrying its own scalar derivation, and every
colour-paired rule has a ``_swapped`` twin with Z and X exchanged. The W, triangle and
connective rules, matched in :mod:`archytaszx.rewrite.match_zxw`, follow them.
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
from archytaszx.diagram.generators import FOURIER_BOX, TRIANGLE, W_NODE, X_SPIDER, Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef, Wire
from archytaszx.rewrite.match import (
    BIALGEBRA_SIDE_CONDITIONS,
    BIALGEBRA_SWAPPED_SIDE_CONDITIONS,
    CAP_SIDE_CONDITIONS,
    CAP_SWAPPED_SIDE_CONDITIONS,
    FOURIER_SIDE_CONDITIONS,
    FOURIER_STATE_SIDE_CONDITIONS,
    FOURIER_STATE_SWAPPED_SIDE_CONDITIONS,
    FUSION_SIDE_CONDITIONS,
    HOPF_SIDE_CONDITIONS,
    HOPF_SWAPPED_SIDE_CONDITIONS,
    IDENTITY_SIDE_CONDITIONS,
    STATE_COPY_SIDE_CONDITIONS,
    STATE_COPY_SWAPPED_SIDE_CONDITIONS,
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
from archytaszx.rewrite.match_box import (
    PORT_BOX_UNFUSION_SIDE_CONDITIONS,
    PortBoxUnfusionMatch,
    PortBoxUnfusionPattern,
    find_port_box_unfusion_matches,
)
from archytaszx.rewrite.match_zxw import (
    CONNECTIVE_INVERSE_SIDE_CONDITIONS,
    CONNECTIVE_STATES_SIDE_CONDITIONS,
    CONNECTIVE_STATES_SWAPPED_SIDE_CONDITIONS,
    TRIANGLE_EFFECT_SIDE_CONDITIONS,
    TRIANGLE_STATE_SIDE_CONDITIONS,
    W_COPY_SIDE_CONDITIONS,
    W_FUSION_SIDE_CONDITIONS,
    W_IDENTITY_SIDE_CONDITIONS,
    W_Z_EFFECT_SIDE_CONDITIONS,
    W_ZERO_EFFECT_SIDE_CONDITIONS,
    WZ_BIALGEBRA_SIDE_CONDITIONS,
    ConnectiveInverseMatch,
    ConnectiveInversePattern,
    ConnectiveStatesMatch,
    ConnectiveStatesPattern,
    FedStateMatch,
    FedStatePattern,
    TriangleEffectMatch,
    TriangleEffectPattern,
    WEffectMatch,
    WEffectPattern,
    WFusionMatch,
    WFusionPattern,
    WIdentityPattern,
    WZBialgebraMatch,
    WZBialgebraPattern,
    find_connective_inverse_matches,
    find_connective_states_matches,
    find_fed_state_matches,
    find_triangle_effect_matches,
    find_w_effect_matches,
    find_w_fusion_matches,
    find_w_identity_matches,
    find_wz_bialgebra_matches,
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
    # to share one innermost node-scope box, or neither to have one. That box and every box
    # enclosing it drop the two consumed nodes and gain the merged one, in place --
    # multiplicity, and every other field, untouched, since this fusion is a single
    # symbolic rewrite standing for one fusion per future instantiated copy.
    _rescope_boxes(diagram, (match.a_id, match.b_id), (new_node_id,), context)

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
    new_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim])
    first, last = match.node_ids[0], match.node_ids[-1]
    port_mapping = {
        PortRef(first, Direction.INPUT, 0): PortRef(new_id, Direction.INPUT, 0),
        PortRef(last, Direction.OUTPUT, 0): PortRef(new_id, Direction.OUTPUT, 0),
    }

    # Bang box "left intact" (Phase 7), as in spider_fusion_builder: the enclosing boxes drop
    # the four consumed nodes and gain the identity spider, every other field untouched.
    _rescope_boxes(diagram, tuple(match.node_ids), (new_id,), "fourier_cancellation_builder")

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
    conditions = CAP_SWAPPED_SIDE_CONDITIONS if match.swapped else CAP_SIDE_CONDITIONS
    check_side_condition_coverage(match, conditions, "zx_cap_builder")
    if match not in find_cap_matches(diagram, swapped=match.swapped):
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

ZX_CAP_SWAPPED = Rule(
    name="zx_cap_swapped",
    pattern=CapPattern(swapped=True),
    builder=zx_cap_builder,
    side_conditions=CAP_SWAPPED_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=zx_cap_scalar(Dim.symbol("d")),
    scalar_in_dim=zx_cap_scalar,
)
"""A phaseless X state capped by a phaseless Z effect: the X state is ``sqrt(d) |0>`` and the
Z effect sums every entry, so the empty diagram times ``d ** (1/2)`` again."""


def _rescope_boxes(
    diagram: Diagram,
    consumed: tuple[NodeId, ...],
    created: tuple[NodeId, ...],
    context: str,
) -> None:
    """Replace ``consumed`` by ``created`` in every node-scope bang box holding all of ``consumed``.

    These are the consumed nodes' shared innermost node-scope box and every box enclosing it.
    Raises :class:`~archytaszx.rewrite.rule.RewriteDomainError` if the consumed nodes do not
    share one innermost node-scope box.
    """
    boxes = {innermost_node_scope_box(diagram, node_id) for node_id in consumed}
    if len(boxes) != 1:
        found = sorted(box for box in boxes if box is not None)
        raise RewriteDomainError(
            f"{context}: the matched nodes do not share one innermost node-scope bang box "
            f"(found {found!r})"
        )
    removed = frozenset(consumed)
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if box.is_node_scope and removed <= box.node_scope:
            diagram.set_bang_box_node_scope(box_id, (box.node_scope - removed) | set(created))


def _follow_port_scopes(diagram: Diagram, port_mapping: Mapping[PortRef, PortRef]) -> None:
    """Send every port-scope bang box entry through ``port_mapping``."""
    for box_id, box in sorted(diagram.bang_boxes.items()):
        if not any(ref in port_mapping for ref in box.port_scope):
            continue
        diagram.set_bang_box_port_scope(
            box_id, frozenset(port_mapping.get(ref, ref) for ref in box.port_scope)
        )


def _require_swapped(match: Match, context: str) -> None:
    """Raise unless ``match`` is a colour-swapped match."""
    if not getattr(match, "swapped", False):
        raise RewriteGrammarError(f"{context} requires a colour-swapped match")


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
    _rescope_boxes(diagram, (match.node_id,), (), "identity_removal_builder")
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
    _rescope_boxes(diagram, consumed, (), "triangle_inverse_cancellation_builder")
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
    """The right-hand side of :data:`STATE_COPY`: one phaseless state of the copied colour per
    former spider output."""
    if not isinstance(match, StateCopyMatch):
        raise RewriteGrammarError(
            f"state_copy_builder requires a StateCopyMatch, got {type(match).__name__}"
        )
    conditions = STATE_COPY_SWAPPED_SIDE_CONDITIONS if match.swapped else STATE_COPY_SIDE_CONDITIONS
    check_side_condition_coverage(match, conditions, "state_copy_builder")
    found = find_state_copy_matches(diagram, swapped=match.swapped)
    _require_rediscovered(match, found, "state_copy_builder")
    dim = match.shared_dim
    state = diagram.nodes[match.state_id].generator_type
    new_ids = tuple(
        diagram.add_node(state, input_dims=[], output_dims=[dim]) for _ in range(match.output_count)
    )
    port_mapping = {
        PortRef(match.spider_id, Direction.OUTPUT, index): PortRef(new_id, Direction.OUTPUT, 0)
        for index, new_id in enumerate(new_ids)
    }
    consumed = (match.state_id, match.spider_id)
    _rescope_boxes(diagram, consumed, new_ids, "state_copy_builder")
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


def state_copy_swapped_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`STATE_COPY_SWAPPED`: :func:`state_copy_builder` on a
    swapped match."""
    _require_swapped(match, "state_copy_swapped_builder")
    return state_copy_builder(diagram, match)


state_copy_swapped_builder.side_conditions = STATE_COPY_SWAPPED_SIDE_CONDITIONS  # type: ignore[attr-defined]
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

STATE_COPY_SWAPPED = Rule(
    name="state_copy_swapped",
    pattern=StateCopyPattern(swapped=True),
    builder=state_copy_swapped_builder,
    side_conditions=STATE_COPY_SWAPPED_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("n",), dimensions=("d",)),
    scalar_introduced=state_copy_scalar(Dim.symbol("d"), 0),
    scalar_in_match=_state_copy_scalar_for,
)
"""A phaseless Z state copies through a phaseless X spider into one Z state per output.

The X spider contracted against the all-ones Z state is ``d^{-(n+1)/2} * d`` on every output
tuple summing to zero, and ``n`` Z states are all-ones: the factor is ``d ** ((1 - n) / 2)``.
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
    conditions = HOPF_SWAPPED_SIDE_CONDITIONS if match.swapped else HOPF_SIDE_CONDITIONS
    check_side_condition_coverage(match, conditions, "hopf_builder")
    _require_rediscovered(match, find_hopf_matches(diagram, swapped=match.swapped), "hopf_builder")
    dim = match.shared_dim
    new_z, z_mapping = _spider_without_legs(
        diagram, diagram.nodes[match.z_id], frozenset(), frozenset(match.z_leg_indices), dim
    )
    new_x, x_mapping = _spider_without_legs(
        diagram, diagram.nodes[match.x_id], frozenset(match.x_leg_indices), frozenset(), dim
    )
    port_mapping = {**z_mapping, **x_mapping}
    consumed = (match.z_id, match.x_id, *match.fourier_ids)
    _rescope_boxes(diagram, consumed, (new_z, new_x), "hopf_builder")
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


def hopf_swapped_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`HOPF_SWAPPED`: :func:`hopf_builder` on a swapped match."""
    _require_swapped(match, "hopf_swapped_builder")
    return hopf_builder(diagram, match)


hopf_swapped_builder.side_conditions = HOPF_SWAPPED_SIDE_CONDITIONS  # type: ignore[attr-defined]
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

HOPF_SWAPPED = Rule(
    name="hopf_swapped",
    pattern=HopfPattern(swapped=True),
    builder=hopf_swapped_builder,
    side_conditions=HOPF_SWAPPED_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("m", "n", "p", "q"), dimensions=("d",)),
    scalar_introduced=hopf_scalar(Dim.symbol("d")),
    scalar_in_dim=hopf_scalar,
)
"""An X above a Z, joined by a plain wire and an F^2 antipode wire, fall apart times ``d ** -1``:
the X's two legs carry ``z`` and ``-z`` and drop out of its total, and its normalisation loses
two factors of ``d ** (-1/2)``."""


def bialgebra_scalar(dim: Dim) -> Scalar:
    """The exact scalar the bialgebra law introduces at ``dim``: ``dim ** (1/2)``."""
    return Scalar.dim_power(dim, 1, 2)


def bialgebra_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`BIALGEBRA`: two one-to-two and two two-to-one spiders of
    the matched colours, each one-to-two feeding both.

    The only builder reporting :attr:`~archytaszx.rewrite.rule.BuildResult.new_wires`.
    """
    if not isinstance(match, BialgebraMatch):
        raise RewriteGrammarError(
            f"bialgebra_builder requires a BialgebraMatch, got {type(match).__name__}"
        )
    conditions = BIALGEBRA_SWAPPED_SIDE_CONDITIONS if match.swapped else BIALGEBRA_SIDE_CONDITIONS
    check_side_condition_coverage(match, conditions, "bialgebra_builder")
    found = find_bialgebra_matches(diagram, swapped=match.swapped)
    _require_rediscovered(match, found, "bialgebra_builder")
    dim = match.shared_dim
    fan_out = diagram.nodes[match.z_id].generator_type
    fan_in = diagram.nodes[match.x_id].generator_type
    z_ids = tuple(
        diagram.add_node(fan_out, input_dims=[dim], output_dims=[dim, dim]) for _ in range(2)
    )
    x_ids = tuple(
        diagram.add_node(fan_in, input_dims=[dim, dim], output_dims=[dim]) for _ in range(2)
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
    _rescope_boxes(diagram, consumed, z_ids + x_ids, "bialgebra_builder")
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


def bialgebra_swapped_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`BIALGEBRA_SWAPPED`: :func:`bialgebra_builder` on a
    swapped match."""
    _require_swapped(match, "bialgebra_swapped_builder")
    return bialgebra_builder(diagram, match)


bialgebra_swapped_builder.side_conditions = BIALGEBRA_SWAPPED_SIDE_CONDITIONS  # type: ignore[attr-defined]
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

BIALGEBRA_SWAPPED = Rule(
    name="bialgebra_swapped",
    pattern=BialgebraPattern(swapped=True),
    builder=bialgebra_swapped_builder,
    side_conditions=BIALGEBRA_SWAPPED_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=bialgebra_scalar(Dim.symbol("d")),
    scalar_in_dim=bialgebra_scalar,
)
"""Z_{2->1} into X_{1->2} crosses over into two X_{1->2} and two Z_{2->1}, times ``d ** (1/2)``.

The left side is ``d ** (-1/2) [i1 == i2] [o1 + o2 == i1]`` and the right ``d ** -1
[o1 + o2 == i1] [o1 + o2 == i2]``; like :data:`BIALGEBRA` it is outside the normal-form set.
"""


def fourier_state_color_change_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`FOURIER_STATE_COLOR_CHANGE`: a phaseless state or effect of
    the other colour."""
    if not isinstance(match, FourierStateMatch):
        raise RewriteGrammarError(
            "fourier_state_color_change_builder requires a FourierStateMatch, got "
            f"{type(match).__name__}"
        )
    conditions = (
        FOURIER_STATE_SWAPPED_SIDE_CONDITIONS if match.swapped else FOURIER_STATE_SIDE_CONDITIONS
    )
    check_side_condition_coverage(match, conditions, "fourier_state_color_change_builder")
    _require_rediscovered(
        match,
        find_fourier_state_matches(diagram, swapped=match.swapped),
        "fourier_state_color_change_builder",
    )
    dim = match.shared_dim
    direction = Direction.OUTPUT if match.is_state else Direction.INPUT
    new_id = diagram.add_node(
        Z_SPIDER if match.swapped else X_SPIDER,
        input_dims=[] if match.is_state else [dim],
        output_dims=[dim] if match.is_state else [],
    )
    port_mapping = {match.free_ref: PortRef(new_id, direction, 0)}
    consumed = (match.spider_id, match.fourier_id)
    _rescope_boxes(diagram, consumed, (new_id,), "fourier_state_color_change_builder")
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


def fourier_state_color_change_swapped_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`FOURIER_STATE_COLOR_CHANGE_SWAPPED`:
    :func:`fourier_state_color_change_builder` on a swapped match."""
    _require_swapped(match, "fourier_state_color_change_swapped_builder")
    return fourier_state_color_change_builder(diagram, match)


fourier_state_color_change_swapped_builder.side_conditions = FOURIER_STATE_SWAPPED_SIDE_CONDITIONS  # type: ignore[attr-defined]
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

FOURIER_STATE_COLOR_CHANGE_SWAPPED = Rule(
    name="fourier_state_color_change_swapped",
    pattern=FourierStateColorChangePattern(swapped=True),
    builder=fourier_state_color_change_swapped_builder,
    side_conditions=FOURIER_STATE_SWAPPED_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""An F box on a phaseless X state's leg is a phaseless Z state, introducing exactly one:
``F . sqrt(d) |0> = sum_k |k>``, and dually for an X effect."""


def w_fusion_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`W_FUSION`: one W whose outputs are the first's, with the
    joined output replaced by the second's outputs in order."""
    if not isinstance(match, WFusionMatch):
        raise RewriteGrammarError(
            f"w_fusion_builder requires a WFusionMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, W_FUSION_SIDE_CONDITIONS, "w_fusion_builder")
    _require_rediscovered(match, find_w_fusion_matches(diagram), "w_fusion_builder")
    first = diagram.nodes[match.first_id]
    second = diagram.nodes[match.second_id]
    outer = [
        PortRef(match.first_id, Direction.OUTPUT, index)
        for index in range(first.num_outputs)
        if index != match.position
    ]
    inner = [PortRef(match.second_id, Direction.OUTPUT, i) for i in range(second.num_outputs)]
    ordered = outer[: match.position] + inner + outer[match.position :]
    fused = diagram.add_node(
        W_NODE, input_dims=[match.shared_dim], output_dims=[match.shared_dim] * len(ordered)
    )
    port_mapping = {PortRef(match.first_id, Direction.INPUT, 0): PortRef(fused, Direction.INPUT, 0)}
    for index, ref in enumerate(ordered):
        port_mapping[ref] = PortRef(fused, Direction.OUTPUT, index)
    consumed = (match.first_id, match.second_id)
    _rescope_boxes(diagram, consumed, (fused,), "w_fusion_builder")
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(fused,),
        consumed_node_ids=consumed,
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


w_fusion_builder.side_conditions = W_FUSION_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


W_FUSION = Rule(
    name="w_fusion",
    pattern=WFusionPattern(),
    builder=w_fusion_builder,
    side_conditions=W_FUSION_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("m", "n"), dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A W fed by a W's output is one W, introducing exactly one; with a zero-output second W,
the counit ``<0|``, the joined output is dropped."""


def w_identity_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`W_IDENTITY`: the wire spliced through."""
    if not isinstance(match, IdentityMatch):
        raise RewriteGrammarError(
            f"w_identity_builder requires an IdentityMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, W_IDENTITY_SIDE_CONDITIONS, "w_identity_builder")
    _require_rediscovered(match, find_w_identity_matches(diagram), "w_identity_builder")
    port_mapping = {match.surviving_ref: match.far_ref}
    _rescope_boxes(diagram, (match.node_id,), (), "w_identity_builder")
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=(match.node_id,),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


w_identity_builder.side_conditions = W_IDENTITY_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


W_IDENTITY = Rule(
    name="w_identity",
    pattern=WIdentityPattern(),
    builder=w_identity_builder,
    side_conditions=W_IDENTITY_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A one-output W is the identity on its wire, introducing exactly one."""


def _w_copy_scalar_for(match: Match) -> Scalar:
    """:func:`state_copy_scalar` read off a :class:`FedStateMatch`."""
    if not isinstance(match, FedStateMatch):
        raise RewriteGrammarError(
            f"w_zero_copy requires a FedStateMatch, got {type(match).__name__}"
        )
    return state_copy_scalar(match.shared_dim, match.output_count)


def w_zero_copy_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`W_ZERO_COPY`: one phaseless X state per W output."""
    if not isinstance(match, FedStateMatch):
        raise RewriteGrammarError(
            f"w_zero_copy_builder requires a FedStateMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, W_COPY_SIDE_CONDITIONS, "w_zero_copy_builder")
    found = find_fed_state_matches(diagram, triangle=False)
    _require_rediscovered(match, found, "w_zero_copy_builder")
    dim = match.shared_dim
    new_ids = tuple(
        diagram.add_node(X_SPIDER, input_dims=[], output_dims=[dim])
        for _ in range(match.output_count)
    )
    port_mapping = {
        PortRef(match.target_id, Direction.OUTPUT, index): PortRef(new_id, Direction.OUTPUT, 0)
        for index, new_id in enumerate(new_ids)
    }
    consumed = (match.state_id, match.target_id)
    _rescope_boxes(diagram, consumed, new_ids, "w_zero_copy_builder")
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=new_ids,
        consumed_node_ids=consumed,
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=state_copy_scalar(dim, match.output_count),
    )


w_zero_copy_builder.side_conditions = W_COPY_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


W_ZERO_COPY = Rule(
    name="w_zero_copy",
    pattern=FedStatePattern(),
    builder=w_zero_copy_builder,
    side_conditions=W_COPY_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("n",), dimensions=("d",)),
    scalar_introduced=state_copy_scalar(Dim.symbol("d"), 0),
    scalar_in_match=_w_copy_scalar_for,
)
"""A phaseless X state into an n-output W is n phaseless X states, scaled by
``d ** ((1 - n) / 2)``: ``W |0> = |0...0>`` and ``X_{0->1} = sqrt(d) |0>``."""


def triangle_zero_state_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`TRIANGLE_ZERO_STATE`: the state, wired where the
    triangle's output was."""
    if not isinstance(match, FedStateMatch):
        raise RewriteGrammarError(
            f"triangle_zero_state_builder requires a FedStateMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(
        match, TRIANGLE_STATE_SIDE_CONDITIONS, "triangle_zero_state_builder"
    )
    found = find_fed_state_matches(diagram, triangle=True)
    _require_rediscovered(match, found, "triangle_zero_state_builder")
    port_mapping = {
        PortRef(match.target_id, Direction.OUTPUT, 0): PortRef(match.state_id, Direction.OUTPUT, 0)
    }
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=(match.target_id,),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


triangle_zero_state_builder.side_conditions = TRIANGLE_STATE_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


TRIANGLE_ZERO_STATE = Rule(
    name="triangle_zero_state",
    pattern=FedStatePattern(triangle=True),
    builder=triangle_zero_state_builder,
    side_conditions=TRIANGLE_STATE_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A T or Ti fixes the phaseless X state ``sqrt(d) |0>``, introducing exactly one."""


def triangle_zero_effect_scalar(dim: Dim) -> Scalar:
    """The exact scalar :data:`TRIANGLE_ZERO_EFFECT` introduces: ``dim ** (1/2)``."""
    return Scalar.dim_power(dim, 1, 2)


def _triangle_effect_scalar_for(match: Match) -> Scalar:
    """:func:`triangle_zero_effect_scalar` read off a :class:`TriangleEffectMatch`."""
    if not isinstance(match, TriangleEffectMatch):
        raise RewriteGrammarError(
            f"triangle_zero_effect requires a TriangleEffectMatch, got {type(match).__name__}"
        )
    return triangle_zero_effect_scalar(match.shared_dim)


def triangle_zero_effect_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`TRIANGLE_ZERO_EFFECT`: a phaseless Z effect on the
    triangle's input."""
    if not isinstance(match, TriangleEffectMatch):
        raise RewriteGrammarError(
            "triangle_zero_effect_builder requires a TriangleEffectMatch, got "
            f"{type(match).__name__}"
        )
    check_side_condition_coverage(
        match, TRIANGLE_EFFECT_SIDE_CONDITIONS, "triangle_zero_effect_builder"
    )
    _require_rediscovered(
        match, find_triangle_effect_matches(diagram), "triangle_zero_effect_builder"
    )
    effect = diagram.add_node(Z_SPIDER, input_dims=[match.shared_dim], output_dims=[])
    port_mapping = {
        PortRef(match.triangle_id, Direction.INPUT, 0): PortRef(effect, Direction.INPUT, 0)
    }
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(effect,),
        consumed_node_ids=(match.triangle_id, match.effect_id),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=triangle_zero_effect_scalar(match.shared_dim),
    )


triangle_zero_effect_builder.side_conditions = TRIANGLE_EFFECT_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


TRIANGLE_ZERO_EFFECT = Rule(
    name="triangle_zero_effect",
    pattern=TriangleEffectPattern(),
    builder=triangle_zero_effect_builder,
    side_conditions=TRIANGLE_EFFECT_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("d",)),
    scalar_introduced=triangle_zero_effect_scalar(Dim.symbol("d")),
    scalar_in_match=_triangle_effect_scalar_for,
)
"""A T into a phaseless X effect is a phaseless Z effect times ``d ** (1/2)``:
``<0| T`` is the all-ones row, and the X effect is ``sqrt(d) <0|``."""


def connective_inverse_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`CONNECTIVE_INVERSE` and its B-first twin: one spliced
    wire per outer pair of legs."""
    if not isinstance(match, ConnectiveInverseMatch):
        raise RewriteGrammarError(
            "connective_inverse_builder requires a ConnectiveInverseMatch, got "
            f"{type(match).__name__}"
        )
    check_side_condition_coverage(
        match, CONNECTIVE_INVERSE_SIDE_CONDITIONS, "connective_inverse_builder"
    )
    found = find_connective_inverse_matches(diagram, bind_first=match.bind_first)
    _require_rediscovered(match, found, "connective_inverse_builder")
    port_mapping = {splice.surviving_ref: splice.far_ref for splice in match.splices}
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(),
        consumed_node_ids=(match.first_id, match.second_id),
        consumed_wires=(*match.joining, *(splice.wire for splice in match.splices)),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


connective_inverse_builder.side_conditions = CONNECTIVE_INVERSE_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


CONNECTIVE_INVERSE = Rule(
    name="connective_inverse",
    pattern=ConnectiveInversePattern(),
    builder=connective_inverse_builder,
    side_conditions=CONNECTIVE_INVERSE_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("s", "t")),
    scalar_introduced=Scalar.one(),
)
"""An S splitting ``s*t`` into ``s`` and ``t``, both fed in order into a B, is the identity on
``s*t``, introducing exactly one."""

CONNECTIVE_INVERSE_BIND_FIRST = Rule(
    name="connective_inverse_bind_first",
    pattern=ConnectiveInversePattern(bind_first=True),
    builder=connective_inverse_builder,
    side_conditions=CONNECTIVE_INVERSE_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("s", "t")),
    scalar_introduced=Scalar.one(),
)
"""A B binding ``s`` and ``t`` into ``s*t``, fed into an S, is the identity on ``s`` and on
``t``, introducing exactly one."""


def connective_states_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`CONNECTIVE_STATES` and its twin: one phaseless state, or
    effect, of the same colour on the joint leg."""
    if not isinstance(match, ConnectiveStatesMatch):
        raise RewriteGrammarError(
            "connective_states_builder requires a ConnectiveStatesMatch, got "
            f"{type(match).__name__}"
        )
    conditions = (
        CONNECTIVE_STATES_SWAPPED_SIDE_CONDITIONS
        if match.swapped
        else CONNECTIVE_STATES_SIDE_CONDITIONS
    )
    check_side_condition_coverage(match, conditions, "connective_states_builder")
    found = find_connective_states_matches(diagram, swapped=match.swapped)
    _require_rediscovered(match, found, "connective_states_builder")
    colour = X_SPIDER if match.swapped else Z_SPIDER
    if match.effects:
        merged = diagram.add_node(colour, input_dims=[match.joint_dim], output_dims=[])
        joint = PortRef(match.connective_id, Direction.INPUT, 0)
        port_mapping = {joint: PortRef(merged, Direction.INPUT, 0)}
    else:
        merged = diagram.add_node(colour, input_dims=[], output_dims=[match.joint_dim])
        joint = PortRef(match.connective_id, Direction.OUTPUT, 0)
        port_mapping = {joint: PortRef(merged, Direction.OUTPUT, 0)}
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(merged,),
        consumed_node_ids=(match.connective_id, *match.end_ids),
        consumed_wires=match.wires,
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
    )


connective_states_builder.side_conditions = CONNECTIVE_STATES_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


def connective_states_swapped_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`CONNECTIVE_STATES_SWAPPED`:
    :func:`connective_states_builder` on a swapped match."""
    _require_swapped(match, "connective_states_swapped_builder")
    return connective_states_builder(diagram, match)


connective_states_swapped_builder.side_conditions = CONNECTIVE_STATES_SWAPPED_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


CONNECTIVE_STATES = Rule(
    name="connective_states",
    pattern=ConnectiveStatesPattern(),
    builder=connective_states_builder,
    side_conditions=CONNECTIVE_STATES_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("s", "t")),
    scalar_introduced=Scalar.one(),
)
"""Phaseless Z states on both inputs of a B are one phaseless Z state on its ``s*t`` output,
and dually for Z effects on an S, introducing exactly one."""

CONNECTIVE_STATES_SWAPPED = Rule(
    name="connective_states_swapped",
    pattern=ConnectiveStatesPattern(swapped=True),
    builder=connective_states_swapped_builder,
    side_conditions=CONNECTIVE_STATES_SWAPPED_SIDE_CONDITIONS,
    quantifiers=Quantifiers(dimensions=("s", "t")),
    scalar_introduced=Scalar.one(),
)
"""Phaseless X states on both inputs of a B are one phaseless X state on its output:
``sqrt(s) |0> (x) sqrt(t) |0>`` binds to ``sqrt(s*t) |0>``; dually for X effects on an S."""


def wz_bialgebra_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`WZ_BIALGEBRA`: one W_{1->m} per Z input and one phaseless
    Z_{n->1} per W output, W ``a``'s output ``b`` feeding Z ``b``'s input ``a``."""
    if not isinstance(match, WZBialgebraMatch):
        raise RewriteGrammarError(
            f"wz_bialgebra_builder requires a WZBialgebraMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(match, WZ_BIALGEBRA_SIDE_CONDITIONS, "wz_bialgebra_builder")
    _require_rediscovered(match, find_wz_bialgebra_matches(diagram), "wz_bialgebra_builder")
    dim = match.shared_dim
    n, m = match.input_count, match.output_count
    w_ids = tuple(
        diagram.add_node(W_NODE, input_dims=[dim], output_dims=[dim] * m) for _ in range(n)
    )
    z_ids = tuple(
        diagram.add_node(Z_SPIDER, input_dims=[dim] * n, output_dims=[dim]) for _ in range(m)
    )
    port_mapping = {
        **{
            PortRef(match.z_id, Direction.INPUT, a): PortRef(w_ids[a], Direction.INPUT, 0)
            for a in range(n)
        },
        **{
            PortRef(match.w_id, Direction.OUTPUT, b): PortRef(z_ids[b], Direction.OUTPUT, 0)
            for b in range(m)
        },
    }
    new_wires = tuple(
        Wire(PortRef(w_ids[a], Direction.OUTPUT, b), PortRef(z_ids[b], Direction.INPUT, a))
        for a in range(n)
        for b in range(m)
    )
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=w_ids + z_ids,
        consumed_node_ids=(match.z_id, match.w_id),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
        new_wires=new_wires,
    )


wz_bialgebra_builder.side_conditions = WZ_BIALGEBRA_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


WZ_BIALGEBRA = Rule(
    name="wz_bialgebra",
    pattern=WZBialgebraPattern(),
    builder=wz_bialgebra_builder,
    side_conditions=WZ_BIALGEBRA_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("n", "m"), dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A phaseless Z_{n->1} into W_{1->m} is n W_{1->m} crossed into m Z_{n->1}, scalar 1."""


def _remaining_w(diagram: Diagram, match: WEffectMatch) -> tuple[NodeId, dict[PortRef, PortRef]]:
    """A W with every output of ``match``'s W but the capped one, and the outputs' mapping."""
    dim = match.shared_dim
    kept = [k for k in range(match.output_count) if k != match.position]
    w = diagram.add_node(W_NODE, input_dims=[dim], output_dims=[dim] * len(kept))
    mapping = {
        PortRef(match.w_id, Direction.OUTPUT, old): PortRef(w, Direction.OUTPUT, new)
        for new, old in enumerate(kept)
    }
    return w, mapping


def w_z_effect_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`W_Z_EFFECT`: a T feeding a W without the capped output."""
    if not isinstance(match, WEffectMatch) or match.zero:
        raise RewriteGrammarError(
            f"w_z_effect_builder requires a Z-effect WEffectMatch, got {match!r}"
        )
    check_side_condition_coverage(match, W_Z_EFFECT_SIDE_CONDITIONS, "w_z_effect_builder")
    _require_rediscovered(match, find_w_effect_matches(diagram, zero=False), "w_z_effect_builder")
    dim = match.shared_dim
    triangle = diagram.add_node(TRIANGLE, input_dims=[dim], output_dims=[dim])
    w, port_mapping = _remaining_w(diagram, match)
    port_mapping[PortRef(match.w_id, Direction.INPUT, 0)] = PortRef(triangle, Direction.INPUT, 0)
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(triangle, w),
        consumed_node_ids=(match.w_id, match.effect_id),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
        new_wires=(Wire(PortRef(triangle, Direction.OUTPUT, 0), PortRef(w, Direction.INPUT, 0)),),
    )


w_z_effect_builder.side_conditions = W_Z_EFFECT_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


W_Z_EFFECT = Rule(
    name="w_z_effect",
    pattern=WEffectPattern(),
    builder=w_z_effect_builder,
    side_conditions=W_Z_EFFECT_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("m",), dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A W_{1->m} output into a phaseless Z effect is T followed by W_{1->m-1}, scalar 1."""


def w_zero_effect_scalar(dim: Dim) -> Scalar:
    """The exact scalar :data:`W_ZERO_EFFECT` introduces: ``dim ** (1/2)``."""
    return Scalar.dim_power(dim, 1, 2)


def _w_zero_effect_scalar_for(match: Match) -> Scalar:
    """:func:`w_zero_effect_scalar` read off a :class:`WEffectMatch`."""
    if not isinstance(match, WEffectMatch):
        raise RewriteGrammarError(
            f"w_zero_effect requires a WEffectMatch, got {type(match).__name__}"
        )
    return w_zero_effect_scalar(match.shared_dim)


def w_zero_effect_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`W_ZERO_EFFECT`: the W without the capped output."""
    if not isinstance(match, WEffectMatch) or not match.zero:
        raise RewriteGrammarError(
            f"w_zero_effect_builder requires an X-effect WEffectMatch, got {match!r}"
        )
    check_side_condition_coverage(match, W_ZERO_EFFECT_SIDE_CONDITIONS, "w_zero_effect_builder")
    _require_rediscovered(match, find_w_effect_matches(diagram, zero=True), "w_zero_effect_builder")
    w, port_mapping = _remaining_w(diagram, match)
    port_mapping[PortRef(match.w_id, Direction.INPUT, 0)] = PortRef(w, Direction.INPUT, 0)
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(w,),
        consumed_node_ids=(match.w_id, match.effect_id),
        consumed_wires=(match.wire,),
        port_mapping=port_mapping,
        scalar_introduced=w_zero_effect_scalar(match.shared_dim),
    )


w_zero_effect_builder.side_conditions = W_ZERO_EFFECT_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


W_ZERO_EFFECT = Rule(
    name="w_zero_effect",
    pattern=WEffectPattern(zero=True),
    builder=w_zero_effect_builder,
    side_conditions=W_ZERO_EFFECT_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("m",), dimensions=("d",)),
    scalar_introduced=w_zero_effect_scalar(Dim.symbol("d")),
    scalar_in_match=_w_zero_effect_scalar_for,
)
"""A W_{1->m} output into a phaseless X effect is W_{1->m-1} times ``d ** (1/2)``."""


def port_box_unfusion_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`PORT_BOX_UNFUSION`: the spider without its boxed leg plus
    one new leg, wired to a new phaseless spider of its colour holding the boxed leg."""
    if not isinstance(match, PortBoxUnfusionMatch):
        raise RewriteGrammarError(
            f"port_box_unfusion_builder requires a PortBoxUnfusionMatch, got {type(match).__name__}"
        )
    check_side_condition_coverage(
        match, PORT_BOX_UNFUSION_SIDE_CONDITIONS, "port_box_unfusion_builder"
    )
    _require_rediscovered(
        match, find_port_box_unfusion_matches(diagram), "port_box_unfusion_builder"
    )
    node = diagram.nodes[match.node_id]
    dim = match.shared_dim
    boxed = match.boxed
    is_output = boxed.direction is Direction.OUTPUT
    inputs = [port.dim for port in node.inputs]
    outputs = [port.dim for port in node.outputs]
    kept = outputs if is_output else inputs
    del kept[boxed.index]
    kept.append(dim)
    main = diagram.add_node(node.generator_type, input_dims=inputs, output_dims=outputs)
    if node.phase is not None:
        diagram.set_phase(main, node.phase)
    side = diagram.add_node(
        node.generator_type,
        input_dims=[dim],
        output_dims=[dim],
    )
    port_mapping: dict[PortRef, PortRef] = {}
    for direction, legs in ((Direction.INPUT, node.inputs), (Direction.OUTPUT, node.outputs)):
        for index in range(len(legs)):
            old = PortRef(match.node_id, direction, index)
            if old == boxed:
                port_mapping[old] = PortRef(side, direction, 0)
            elif direction is boxed.direction and index > boxed.index:
                port_mapping[old] = PortRef(main, direction, index - 1)
            else:
                port_mapping[old] = PortRef(main, direction, index)
    joint = len(kept) - 1
    if is_output:
        wire = Wire(PortRef(main, Direction.OUTPUT, joint), PortRef(side, Direction.INPUT, 0))
    else:
        wire = Wire(PortRef(side, Direction.OUTPUT, 0), PortRef(main, Direction.INPUT, joint))
    _follow_port_scopes(diagram, port_mapping)
    return BuildResult(
        diagram=diagram,
        new_node_ids=(main, side),
        consumed_node_ids=(match.node_id,),
        consumed_wires=(),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
        new_wires=(wire,),
    )


port_box_unfusion_builder.side_conditions = PORT_BOX_UNFUSION_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The declared side-condition tuple this builder is meant to be paired with."""


PORT_BOX_UNFUSION = Rule(
    name="port_box_unfusion",
    pattern=PortBoxUnfusionPattern(),
    builder=port_box_unfusion_builder,
    side_conditions=PORT_BOX_UNFUSION_SIDE_CONDITIONS,
    quantifiers=Quantifiers(leg_counts=("n",), dimensions=("d",)),
    scalar_introduced=Scalar.one(),
)
"""A spider with a port-boxed leg and two other legs is two spiders of its colour joined by
one wire, the boxed leg on a new phaseless one, scalar 1."""


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
        ZX_CAP_SWAPPED.name: ZX_CAP_SWAPPED,
        STATE_COPY_SWAPPED.name: STATE_COPY_SWAPPED,
        HOPF_SWAPPED.name: HOPF_SWAPPED,
        BIALGEBRA_SWAPPED.name: BIALGEBRA_SWAPPED,
        FOURIER_STATE_COLOR_CHANGE_SWAPPED.name: FOURIER_STATE_COLOR_CHANGE_SWAPPED,
        W_FUSION.name: W_FUSION,
        W_IDENTITY.name: W_IDENTITY,
        W_ZERO_COPY.name: W_ZERO_COPY,
        TRIANGLE_ZERO_STATE.name: TRIANGLE_ZERO_STATE,
        TRIANGLE_ZERO_EFFECT.name: TRIANGLE_ZERO_EFFECT,
        CONNECTIVE_INVERSE.name: CONNECTIVE_INVERSE,
        CONNECTIVE_INVERSE_BIND_FIRST.name: CONNECTIVE_INVERSE_BIND_FIRST,
        CONNECTIVE_STATES.name: CONNECTIVE_STATES,
        CONNECTIVE_STATES_SWAPPED.name: CONNECTIVE_STATES_SWAPPED,
        WZ_BIALGEBRA.name: WZ_BIALGEBRA,
        W_Z_EFFECT.name: W_Z_EFFECT,
        W_ZERO_EFFECT.name: W_ZERO_EFFECT,
        PORT_BOX_UNFUSION.name: PORT_BOX_UNFUSION,
    }
)
"""Every rule this module registers, keyed by :attr:`~archytaszx.rewrite.rule.Rule.name`.

A ``MappingProxyType``, so a caller cannot mutate the registry through it.
"""
