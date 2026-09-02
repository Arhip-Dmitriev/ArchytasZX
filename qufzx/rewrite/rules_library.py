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

Phase 5 registers exactly one rule, :data:`SPIDER_FUSION`, built from
:class:`~qufzx.rewrite.match.FusionPattern` and :func:`spider_fusion_builder`.

Rule registry. A :class:`~qufzx.rewrite.engine.RewriteStep` records only a rule's ``name``,
so Phase 6's certificate replay needs a way to resolve that name back to the rule.
:data:`RULES` and :func:`lookup_rule` are that path, which keeps
:mod:`qufzx.rewrite.engine` generic over any future rule.

Scalar derivation (not assertion). Same-color fusion across one wire introduces no scalar
factor in either wire shape that condition 4
(``consumed_wire_direction_permitted_for_color``) in :mod:`qufzx.rewrite.match` permits:

* Alternating output-to-input, either color. Z: both spiders are diagonal with entry
  ``e^{i*angle(k)}`` at the all-axes-``k`` position, so contracting an output leg against
  an input leg identifies their ``k``, giving the merged spider's tensor with phase
  ``alpha + beta`` and no extra factor. X: ``X_{m->n} = F^{ox n} . Z_{m->n} .
  (conj(F))^{ox m}``, and the wire contracts an ``F`` against a ``conj(F)`` on the shared
  axis; ``F`` is unitary and symmetric, so these cancel to the identity, leaving the Z
  argument sandwiched between the surviving Fourier factors.
* Same-direction, Z only. ``_z_tensor`` is diagonal in every axis and
  :mod:`qufzx.semantics.contract_numeric` applies no conjugation at contraction time, so
  the same index ``k`` is identified and the derivation is unchanged. It does not carry
  over to X, whose tensor is not diagonal: a same-direction X-X wire contracts ``F``
  against ``F``, giving ``F^T F``, a permutation matrix, not the identity.

Neither derivation depends on the consumed wire being the pair's only wire: a further wire
between the same nodes is never contracted by this rule -- both endpoints survive and are
remapped onto the merged node as a self-loop, contributing the same factor to both sides.

Both shapes land on :meth:`~qufzx.algebra.scalar.Scalar.one`, agreeing with the oracle in
``tests/test_phase5_oracle.py`` and the fuzz comparisons in
``tests/test_fusion_properties.py``.

Merged leg-ordering convention. A choice, not a derivation: the merged node's inputs are
A's surviving inputs in original index order, then B's; outputs likewise. "A" is the lower
:class:`~qufzx.diagram.graph.NodeId`, per :class:`~qufzx.rewrite.match.FusionMatch`.

Dimension of the merged node. Every surviving port is built at
:attr:`~qufzx.rewrite.match.FusionMatch.shared_dim`, never its own original ``Dim``. Z and
X being ``ALL_LEGS_EQUAL`` is not sufficient justification: :mod:`qufzx.diagram.validate`
reports nothing for two leg dims that unify by binding a free symbol, so forcing legs onto
``shared_dim`` unchecked could silently overwrite one. What makes it sound is that
condition 6 (``dimension_agreement``) unifies every surviving leg against the resolved
``shared_dim`` before a match is returned, and this builder calls the same
:func:`~qufzx.rewrite.match.resolve_fusion_match` fresh against the diagram it was handed,
building only after confirming agreement with ``match.shared_dim``. A foreign match whose
``shared_dim`` does not relate to the ports it names is rejected with
:class:`~qufzx.rewrite.rule.RewriteDomainError`.

When agreement only deferred, the assumed equality is carried into the diagram: a
neighbouring wire that was an exact match before the fusion may be merely deferred after
it. That is expected, and permitted by :mod:`qufzx.rewrite.engine`'s step-8 relative
postcondition. See :func:`_merged_phase` for the legless corner case, where dimension can
survive only via the phase slot.

Phase 5 judgement call: fusion may fire on a ``DEFERRED`` dimension pair, though
FULL_PLAN.md's Phase 5 states the pattern as spiders "sharing a dimension". A ``d``/``d*e``
leg pair is already legal, non-hard-error input under ``ALL_LEGS_EQUAL``
(:class:`~qufzx.diagram.validate.IssueKind.DIMENSION_DEFERRED`), so refusing to fuse across
it would make this pattern stricter than the diagram format it operates on for no soundness
gain -- the assumption is recorded either way, as a diagram-level finding before and a
``dimension_constraints`` entry after, and the removed finding is surfaced in
:attr:`~qufzx.rewrite.engine.RewriteStep.removed_deferred_issues`.

Recorded is not satisfiable. A surviving leg of ``d**2`` forced onto ``shared_dim = d`` is
a legal ``DEFERRED`` unify recording ``d**2 == d``, which holds over the positive integers
only at ``d = 1``. Phase 5's placeholder ``Dim.unify`` cannot tell such a constraint from
one satisfiable on an interesting set, so a ``DEFERRED`` entry is always an assumption a
real unifier (Phase 10) must discharge.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import PhaseDomainError, PhaseVector
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, Port, PortRef
from qufzx.rewrite.match import (
    FUSION_SIDE_CONDITIONS,
    FusionMatch,
    FusionPattern,
    reattach_phase,
    resolve_fusion_match,
)
from qufzx.rewrite.rule import (
    BuildResult,
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

    Delegates to :func:`qufzx.rewrite.match.reattach_phase`, and returns its reattached
    vector together with the subset of ``bindings`` it substituted into an entry's value
    (empty when ``phase`` is absent).

    Raises :class:`RewriteDomainError`, not :class:`~qufzx.algebra.phase.PhaseDomainError`,
    if an entry index falls outside ``shared_dim``'s range.
    """
    if phase is None:
        return PhaseVector(shared_dim, {}), MappingProxyType({})
    try:
        return reattach_phase(phase, shared_dim, bindings)
    except PhaseDomainError as exc:
        raise RewriteDomainError(
            f"spider_fusion cannot reattach a phase vector to shared dimension "
            f"{shared_dim}: {exc}"
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

    A `None` phase on both sides stays `None`, except when ``any_legs_survive`` is `False`:
    a merged node with no surviving legs has no port left to carry ``shared_dim``, since
    dimension lives per port and nowhere else, so an all-zero ``PhaseVector(shared_dim,
    {})`` is returned to give it a place to live. Otherwise both operands are read via
    :func:`_over_shared_dim` before adding, since
    :meth:`~qufzx.algebra.phase.PhaseVector.__add__` demands exactly equal ``Dim``\\ s.

    The second return value is every node whose phase actually had a binding substituted
    into an entry, keyed by node id.
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


def spider_fusion_builder(diagram: Diagram, match: Match) -> BuildResult:
    """The right-hand side of :data:`SPIDER_FUSION`: merge the two matched spiders.

    Mutates ``diagram`` in place by adding the merged node (see the module docstring for
    the leg-ordering and scalar conventions) and returns the :class:`BuildResult`
    :mod:`qufzx.rewrite.engine` needs to splice it in; never removes the matched nodes or
    touches any wire or boundary entry itself.

    Trusts nothing about ``match`` for graph surgery until it has been independently
    re-derived. In order:

    1. ``isinstance(match, FusionMatch)`` -- a foreign match type is a malformed request.
    2. :func:`~qufzx.rewrite.rule.check_side_condition_coverage` against the module-level
       :data:`FUSION_SIDE_CONDITIONS`. This builder is reachable directly, not only through
       :func:`qufzx.rewrite.engine.apply`, so it cannot rely on that function having
       checked.
    3. :func:`~qufzx.rewrite.match.resolve_fusion_match`, called fresh against ``diagram``
       -- the same function :func:`~qufzx.rewrite.match.find_matches` uses. This verifies
       that the two generator types agree (coverage alone catches only a missing or
       duplicate outcome, not a fabricated-passing one) and that
       ``match.shared_dim``/``match.bindings`` really relate to the ports being assigned
       them. Any disagreement raises :class:`~qufzx.rewrite.rule.RewriteDomainError`.
       Everything downstream builds from ``resolution``'s fields, never ``match``'s.
    4. ``match.dimension_constraints`` and ``match.side_condition_outcomes``, checked for
       exact agreement with ``resolution``'s, so a match cannot claim to have assumed
       nothing and have ``apply`` record that verbatim. This builder returns
       ``resolution``'s as
       :attr:`~qufzx.rewrite.rule.BuildResult.verified_side_condition_outcomes` /
       ``verified_dimension_constraints``, which ``apply`` prefers when recording.

    A structurally malformed match -- equal node ids, a node absent from ``diagram``, or a
    wire not incident on both or not in ``diagram.wires`` -- surfaces as
    :class:`~qufzx.rewrite.rule.RewriteGrammarError` from step 3.
    """
    if not isinstance(match, FusionMatch):
        raise RewriteGrammarError(
            f"spider_fusion requires a FusionMatch, got {type(match).__name__}"
        )
    # Reads the module-level constant, not spider_fusion_builder.side_conditions, which
    # would be a self-reference to this function's own global name and would break under a
    # rename or wrap. That attribute exists solely for Rule.__post_init__.
    check_side_condition_coverage(match, FUSION_SIDE_CONDITIONS, "spider_fusion")

    resolution = resolve_fusion_match(diagram, match.a_id, match.b_id, match.wire)
    if not resolution.passed:
        failed = [outcome.name for outcome in resolution.outcomes if not outcome.passed]
        raise RewriteDomainError(
            f"spider_fusion: match at ({match.a_id!r}, {match.b_id!r}) over wire "
            f"{match.wire!r} fails side condition(s) {failed} when re-verified fresh "
            "against the diagram it is being applied to; match.side_condition_outcomes "
            "claimed every condition passed, but a match's own outcomes are never taken "
            "on faith for graph surgery -- see resolve_fusion_match"
        )
    assert resolution.shared_dim is not None  # invariant: passed implies shared_dim is set
    if match.shared_dim != resolution.shared_dim:
        raise RewriteDomainError(
            f"spider_fusion: match.shared_dim {match.shared_dim!r} disagrees with the "
            f"shared dimension {resolution.shared_dim!r} resolve_fusion_match derives "
            "fresh from the diagram for this wire; a match's own shared_dim is never "
            "trusted for graph surgery without this agreement"
        )
    if dict(match.bindings) != dict(resolution.bindings):
        raise RewriteDomainError(
            f"spider_fusion: match.bindings {dict(match.bindings)!r} disagrees with the "
            f"bindings {dict(resolution.bindings)!r} resolve_fusion_match derives fresh "
            "from the diagram for this wire"
        )
    # The certificate must record what this rewrite actually assumed, not what the match
    # claims. These two fields must agree exactly with what resolve_fusion_match derived
    # fresh, or a fabricated pair would reach apply and be recorded verbatim onto
    # RewriteStep.
    if match.dimension_constraints != resolution.dimension_constraints:
        raise RewriteDomainError(
            f"spider_fusion: match.dimension_constraints {match.dimension_constraints!r} "
            f"disagrees with {resolution.dimension_constraints!r}, which "
            "resolve_fusion_match derives fresh from the diagram for this wire -- a "
            "match's own dimension_constraints is never trusted for the certificate "
            "without this agreement"
        )
    if match.side_condition_outcomes != resolution.outcomes:
        raise RewriteDomainError(
            "spider_fusion: match.side_condition_outcomes disagrees with the outcomes "
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

    # Safe to build from node_a.generator_type only because resolution.passed above
    # confirmed it equals node_b's (condition 2).
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

    return BuildResult(
        diagram=diagram,
        new_node_ids=(new_node_id,),
        consumed_node_ids=(match.a_id, match.b_id),
        consumed_wires=(wire,),
        port_mapping=port_mapping,
        scalar_introduced=Scalar.one(),
        verified_side_condition_outcomes=resolution.outcomes,
        verified_dimension_constraints=resolution.dimension_constraints,
        verified_phase_substitutions=MappingProxyType(phase_substitutions),
    )


spider_fusion_builder.side_conditions = FUSION_SIDE_CONDITIONS  # type: ignore[attr-defined]
"""The single declared side-condition tuple this builder is meant to be paired with.

Set on the function object so :class:`~qufzx.rewrite.rule.Rule`'s constructor-time
consistency check can compare it against the ``side_conditions`` a wrapping ``Rule``
declares, making two contradicting tuples for one builder impossible to construct. That
check is its only reader; the builder's own body reads the module-level constant.
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
merged spider (see condition 3 in :mod:`qufzx.rewrite.match`). See the module docstring for
the scalar derivation and condition 4 there for exactly which direction combinations each
color permits.
"""


RULES: Mapping[str, Rule] = MappingProxyType({SPIDER_FUSION.name: SPIDER_FUSION})
"""Every rule this module registers, keyed by :attr:`~qufzx.rewrite.rule.Rule.name`.

A ``MappingProxyType`` so a caller cannot mutate the registry through the reference
:func:`lookup_rule` or this constant hands out.
"""


def lookup_rule(name: str) -> Rule:
    """Resolve a rule name (e.g. ``"spider_fusion"``) back to its :class:`Rule` object.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` if ``name`` is not registered
    in :data:`RULES` -- an unknown rule name is a malformed replay request, not a
    mathematical domain violation.
    """
    try:
        return RULES[name]
    except KeyError:
        raise RewriteGrammarError(f"no such rule: {name!r}") from None
