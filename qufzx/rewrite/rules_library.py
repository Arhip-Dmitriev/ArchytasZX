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
:class:`~qufzx.rewrite.match.FusionPattern` and :func:`spider_fusion_builder` below.

Rule registry. A :class:`~qufzx.rewrite.engine.RewriteStep` records only a rule's
``name`` (a plain string), not the :class:`~qufzx.rewrite.rule.Rule` object itself, so
Phase 6's certificate replay needs a way to resolve that name back to the actual rule it
names before it can re-apply anything. :data:`RULES` and :func:`lookup_rule` are that
resolution path: every rule this module defines is registered there, keyed by
:attr:`~qufzx.rewrite.rule.Rule.name`, so :mod:`qufzx.rewrite.engine` (which must stay
generic over any future rule, per its own module docstring) never needs to import a
specific rule module itself.

Scalar derivation (not assertion). Same-color fusion across one wire introduces no scalar
factor in either shape that wire can take -- read directly off
:mod:`qufzx.semantics.denote`'s formulas, the merged node's denotation already equals the
pre-fusion diagram's contraction with no leftover coefficient. Two distinct wire shapes are
possible, gated by :mod:`qufzx.rewrite.match`'s condition 4
(``consumed_wire_direction_permitted_for_color`` -- see that module's docstring for the full
account of exactly which direction combinations each color permits, and why): an alternating
output-to-input wire, valid fusion for both colors, and a same-direction (output-output or
input-input) wire, valid fusion for Z only.

* Alternating output-to-input, either color. Z: both spiders are diagonal, entry
  ``e^{i*angle(k)}`` at the all-axes-``k`` position; contracting an output leg of one
  against an input leg of the other identifies their ``k`` in the sum, giving exactly the
  merged spider's tensor with phase vector ``alpha + beta``
  (:meth:`~qufzx.algebra.phase.PhaseVector.__add__`) and no extra factor. X:
  ``X_{m->n} = F^{ox n} . Z_{m->n} . (conj(F))^{ox m}``; an output-to-input wire contracts
  an ``F`` against a ``conj(F)`` on the shared axis, and since ``F`` is unitary and
  symmetric these cancel to the identity, leaving exactly the Z-spider argument above
  sandwiched between the surviving Fourier factors on every other leg.
* Same-direction, Z only. ``_z_tensor`` is diagonal in every axis regardless of direction,
  and :mod:`qufzx.semantics.contract_numeric` contracts a wire by assigning its two
  endpoints the same einsum axis label unconditionally, applying no conjugation at
  contraction time at all (for X, conjugation is applied only at ``denote()`` time, when an
  input axis's Fourier factor is built as ``conj(F)`` rather than ``F`` -- it plays no part
  in contraction itself). A same-direction Z-Z wire therefore identifies the same basis
  index ``k`` on both endpoints exactly as the alternating case does, yielding the merged
  spider's tensor with phase ``alpha + beta`` and no leftover coefficient -- the identical
  derivation to the Z case above, merely without requiring one endpoint to be an input and
  the other an output. This argument does not carry over to X: X's own tensor is not
  diagonal (its diagonal Z core sits sandwiched between Fourier factors), so a
  same-direction X-X wire would contract ``F`` against ``F`` (or ``conj(F)`` against
  ``conj(F)``) on the shared axis rather than ``F`` against ``conj(F)`` -- giving
  ``F^T F`` (or its conjugate), a nontrivial permutation matrix, not the identity. That is a
  different (and, for Phase 5, unimplemented) rule, which is exactly why condition 4
  restricts same-direction fusion to Z.

Neither derivation depends on the consumed wire being the pair's only wire: a further wire
between the same two nodes is never contracted by this rule -- both its endpoints are
surviving legs, remapped onto the merged node as a self-loop, so it still contracts the
images of those same two legs and contributes the same factor to both sides.

Both shapes land on :meth:`~qufzx.algebra.scalar.Scalar.one`, which is what
:data:`SPIDER_FUSION` declares and what :func:`spider_fusion_builder` returns per
application, agreeing with the Phase 4 oracle in ``tests/test_phase5_oracle.py`` and with
the fuzz-tested oracle comparisons in ``tests/test_fusion_properties.py``.

Merged leg-ordering convention. A choice, stated once here, not a derivation: the merged
node's inputs are A's surviving inputs, original index order, then B's; its outputs follow
the same rule. "A" and "B" are :class:`~qufzx.rewrite.match.FusionMatch`'s own convention
(A is always the lower :class:`~qufzx.diagram.graph.NodeId`); "surviving" means every leg
except the one consumed by the matched wire.

Dimension of the merged node. Every surviving port -- A's and B's alike -- is built at
exactly :attr:`~qufzx.rewrite.match.FusionMatch.shared_dim`, never each leg's own original
``Dim``. This is *not* sound merely because Z and X are ``ALL_LEGS_EQUAL``
(:mod:`qufzx.diagram.generators`): ``ALL_LEGS_EQUAL`` is enforced by
:mod:`qufzx.diagram.validate` via :meth:`~qufzx.algebra.dimension.Dim.unify`, and
``validate`` reports no issue at all -- not even a deferred one -- for two leg dims that
unify by binding a free symbol (e.g. a leg stated over ``Dim(2)`` sitting beside a leg
stated over the still-free symbol ``Dim("d")``); forcing every surviving leg onto
``shared_dim`` without first checking it against each leg's own dim would silently
overwrite such a leg -- or, worse, a leg whose dim plainly does not unify with
``shared_dim`` at all -- with no record and no rejected match. What actually makes this
construction sound is that :mod:`qufzx.rewrite.match`'s ``dimension_agreement`` condition
(condition 5 in that module's docstring) itself unifies every surviving leg of both nodes
against the resolved ``shared_dim`` before a :class:`~qufzx.rewrite.match.FusionMatch` is
ever returned: a leg that fails to unify makes the candidate a non-match, and a leg that
only unifies by deferring or binding a symbol is recorded in
:attr:`~qufzx.rewrite.match.FusionMatch.dimension_constraints`, with ``shared_dim`` itself
possibly refined further by that leg's binding. This claim is only as sound as the fixpoint
that resolves ``shared_dim``/``bindings`` in the first place: D1 (Phase 5 audit round 15)
was exactly a case where every surviving leg individually passed this check against *some*
value of ``shared_dim``, yet the recorded assumptions were jointly unsatisfiable, because the
fixpoint exited before every leg had been re-checked against the fully-accumulated
``bindings``. :func:`~qufzx.rewrite.match.resolve_fusion_match` now closes this structurally
(terminate only when a full pass adds nothing to either ``shared_dim`` or ``bindings``, and a
post-loop closure check that re-verifies every leg, phase, and the connecting pair against
the *final* state) rather than this builder re-deriving its own separate guarantee -- see
that function's own inline commentary. This builder does not merely take that
verification on faith, either (Phase 5 round-12 audit, A2): :func:`spider_fusion_builder`
calls :func:`~qufzx.rewrite.match.resolve_fusion_match` -- the very same function
:func:`~qufzx.rewrite.match.find_matches` calls to decide whether a candidate is a match at
all -- fresh against the diagram it was actually handed, and builds from *its* returned
``shared_dim`` only after confirming it agrees exactly with ``match.shared_dim``; a
foreign or hand-built match whose ``shared_dim`` does not relate to the ports it names is
rejected with :class:`~qufzx.rewrite.rule.RewriteDomainError` rather than trusted. This
is what makes "match-approval and build-applicability are the same predicate by
construction" (see :mod:`qufzx.rewrite.match`'s module docstring) literally true -- the
same function object, called from both places, not a policy this builder re-derives
independently and hopes stays in sync. When the matched wire's ``dimension_agreement`` only
deferred (on the connecting pair or on some surviving leg), the affected leg's assumed
equality with ``shared_dim`` is
carried into the diagram -- a neighbouring wire that was an exact match before this fusion
may become merely deferred after it. That is expected, not a defect. See
:func:`_merged_phase` for the legless corner case, where dimension can only survive via
the phase slot.

Phase 5 judgement call, decided (Phase 5 post-closing audit, judgement call 1): fusion is
permitted to fire on a ``DEFERRED`` dimension pair at all, even though FULL_PLAN.md's Phase
5 item (ii) states the pattern as "two same-color spiders joined by a wire and sharing a
dimension" -- a ``DEFERRED`` unify means it is not actually known that the two legs share
one, only that :meth:`~qufzx.algebra.dimension.Dim.unify`'s deliberately weak placeholder
could not decide either way. Concretely, a node with legs ``[d, d*e]`` fused against a node
with leg ``d`` produces a merged node whose surviving port is ``Dim(d)``; the ``d*e`` label
is gone from the diagram entirely, surviving only as a ``dimension_constraints`` entry on
the :class:`~qufzx.rewrite.engine.RewriteStep` certificate.

The alternative (refuse to match on ``DEFERRED``, reading FULL_PLAN.md's "sharing a
dimension" literally) was rejected, not merely left unexamined: ``DEFERRED`` is
:mod:`qufzx.diagram.validate`'s own deferral posture too -- a ``d``/``d*e`` leg pair is
already legal, non-hard-error input under ``ALL_LEGS_EQUAL`` before any rewrite touches it
(:class:`~qufzx.diagram.validate.IssueKind.DIMENSION_DEFERRED`, not
``DIMENSION_POLICY_VIOLATION``), so refusing to fuse across it would make this pattern
*stricter* than the diagram format it operates on already accepts, for no soundness gain:
the merged node's tensor is exactly as well-defined under the assumption as the pre-fusion
diagram was, and the assumption is recorded either way (as a diagram-level
``DIMENSION_DEFERRED`` finding before the rewrite, as a ``dimension_constraints`` certificate
entry after it). Refusing would also invalidate a large, deliberate slice of existing
coverage built specifically to exercise this path (:mod:`tests.test_fusion_properties`'s
``d*e``/``d**2`` palette entries, added precisely so ``Dim.unify``'s ``DEFERRED`` branch --
and :func:`_unify_surviving_legs`'s dedicated handling of it -- was exercised at all), for a
literal reading of one phrase in a phase-level spec summary against a placeholder unifier
that is explicitly provisional (see ``Dim.unify``'s own docstring) and superseded by Phase
10 regardless of which choice Phase 5 makes now.

What was a genuine, unaddressed gap -- not the firing itself -- is that the resulting
assumption used to be recorded *only* on the certificate, never surfaced anywhere near the
diagram-level bookkeeping that already tracks deferred assumptions
(:class:`~qufzx.diagram.validate.ValidationReport.deferred`): a rewrite consuming or
overwriting a ``d*e``-typed leg makes that diagram-level ``DIMENSION_DEFERRED`` finding
vanish with nothing announcing that this specific rewrite is the one that made it disappear
-- step 8 (see :mod:`qufzx.rewrite.engine`) never even looked at deferred issues, only hard
ones, so a removed deferred issue was invisible by construction, not merely unblocked.
:attr:`~qufzx.rewrite.engine.RewriteStep.removed_deferred_issues` closes this: every
deferred issue ``diagram`` carried whose translated key finds no counterpart among
``working``'s own deferred issues is now recorded there, not as a raised error (removing a
deferred issue by resolving it is not itself wrong -- that is the entire point of allowing
fusion to fire across one) but as an explicit, always-present certificate field a Phase 6
reader can inspect rather than a silent gap :func:`~qufzx.rewrite.match.find_matches`'s
``dimension_constraints`` only indirectly hinted at. See that field's own docstring, and
:mod:`qufzx.rewrite.engine`'s module docstring, step 8, for the mechanism.

Round 20, Task 11 -- what judgement call 1 does *not* claim. Firing on a ``DEFERRED`` pair
records an assumption, but "recorded" and "satisfiable" are different claims, and this
module does not conflate them. A surviving leg of dimension ``d**2`` forced onto
``shared_dim = d`` (a legal ``DEFERRED`` unify: ``d`` occurs as a proper subterm of ``d**2``)
records the constraint ``d**2 == d``, which holds over the positive integers only at ``d =
1`` -- the rewrite is sound in the narrow sense that it asserts equality under exactly the
assumption it recorded, and the oracle can confirm that assumption at any concrete
substitution satisfying it, but Phase 5's placeholder :meth:`~qufzx.algebra.dimension.Dim
.unify` has no way to tell a recorded constraint that is satisfiable on an interesting
(infinite, or large) subset of assignments from one that is satisfiable only at such a
single degenerate point. A ``DEFERRED`` entry in
:attr:`~qufzx.rewrite.engine.RewriteStep.dimension_constraints` is therefore always an
assumption a real unifier (Phase 10) must eventually discharge -- not a claim this module
can itself distinguish as "probably fine" versus "vacuous but technically not FAILURE". No
machinery is added here to make that distinction: per ``Dim.unify``'s own docstring, that is
explicitly Phase 10's job, and Phase 5 recording the assumption honestly (rather than
silently accepting or rejecting based on a guess at its satisfiability) is the whole point of
recording it as ``DEFERRED`` rather than as a bare pass.
:class:`~qufzx.rewrite.rule.RewriteStep.dimension_constraints`'s own docstring states the
identical caveat, so a reader who reaches it from either direction sees the same warning.

Phase 5 judgement call 2, decided (Phase 5 post-closing audit): whether
``phase_dimension_agreement``'s pre-fix plain-``Dim``-equality conservatism (never calling
:meth:`~qufzx.algebra.dimension.Dim.unify`, and additionally requiring two present phases'
raw ``Dim``\\ s to equal each other) was still wanted now that :func:`reattach_phase`
substitutes bindings into a phase's entries before reattaching it to ``shared_dim``. Decided
no, on both counts, and fixed in :mod:`qufzx.rewrite.match` rather than re-documented as an
accepted limitation: the raw-dim-agreement check between two present phases was never
actually load-bearing once ``reattach_phase`` forces both operands onto the identical
``shared_dim`` regardless of their raw ``Dim``\\ s, and the plain-equality check (versus a
real ``unify`` call) silently missed a phase whose ``Dim`` unifies with ``shared_dim`` only
via a binding this condition itself would have to produce. See
:mod:`qufzx.rewrite.match`'s module docstring, condition 6, for the full account of both
retired checks and why neither bought any soundness.
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
    """``phase``'s entries, with ``bindings`` substituted in, reattached to ``shared_dim``.

    Returns the reattached vector together with the subset of ``bindings`` that
    :func:`~qufzx.rewrite.match.reattach_phase` actually substituted into an entry's value
    -- empty when ``phase`` is absent.

    Or an all-zero vector if ``phase`` is absent. Delegates to
    :func:`qufzx.rewrite.match.reattach_phase` -- the same function
    ``phase_dimension_agreement`` uses as its own trial construction while matching -- so
    that match-approval and build-applicability are the same predicate by construction, not
    two similar-looking computations kept in sync by hand. See that function's docstring for
    the full account of why substituting ``bindings`` into the entries (rather than
    reattaching them unchanged, the pre-fix behavior) is the correct resolution of the
    ``_over_shared_dim`` defect family: an entry stated in terms of a dimension symbol that
    ``shared_dim`` resolution has since bound (e.g. ``1/d turns`` once ``d := 2``) denotes a
    different, and wrong, angle if left unsubstituted once the container ``Dim`` becomes the
    concrete ``2``.

    Raises :class:`RewriteDomainError`, not :class:`~qufzx.algebra.phase.PhaseDomainError`,
    if an entry index falls outside ``shared_dim``'s valid range -- this builder is
    reachable directly, not only through :func:`qufzx.rewrite.engine.apply`, so a foreign or
    hand-built match must not leak a different module's exception hierarchy through it.
    ``phase_dimension_agreement`` performs this exact same construction (with the same
    ``bindings``) as part of matching, so for any match ``find_matches`` actually returned,
    this call cannot raise -- the ``except`` branch below is unreachable from such a match
    and exists only to keep this function safe against a foreign or hand-built one.
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

    A `None` phase on both sides stays `None` -- *except* when ``any_legs_survive`` is
    `False`: a merged node with no surviving legs has no port left to carry ``shared_dim``
    (dimension lives per port and nowhere else, see ``claude.md``), so this returns an
    explicit all-zero ``PhaseVector(shared_dim, {})`` purely to give it a place to live.
    Otherwise, both operands are read via :func:`_over_shared_dim` (which substitutes
    ``bindings`` into each operand's entries before reattaching to ``shared_dim`` -- see
    that function's docstring) before adding, since
    :meth:`~qufzx.algebra.phase.PhaseVector.__add__` demands its two operands' ``Dim``\\ s
    be exactly equal. The second return value is every node whose phase actually had a
    binding substituted into an entry, keyed by node id -- empty for a node with no phase
    or whose phase entries mentioned no bound symbol.
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
    touches any wire or boundary entry itself -- see :class:`~qufzx.rewrite.rule.BuildResult`.

    Trusts nothing about ``match`` for graph surgery until it has been independently
    re-derived (Phase 5 round-12 audit, defects A1/A2/A4). In order:

    1. ``isinstance(match, FusionMatch)`` -- a foreign match type is a malformed request.
    2. :func:`~qufzx.rewrite.rule.check_side_condition_coverage` against
       ``spider_fusion_builder.side_conditions`` (the same object
       :data:`SPIDER_FUSION.side_conditions <SPIDER_FUSION>` is built from -- see
       :class:`~qufzx.rewrite.rule.Rule`'s constructor-time consistency check, which makes
       two contradicting tuples impossible to wire up in the first place, not merely
       undesirable). This builder is reachable directly, not only through
       :func:`qufzx.rewrite.engine.apply`, so it cannot rely on that function having
       already checked this.
    3. :func:`~qufzx.rewrite.match.resolve_fusion_match`, called fresh against ``diagram``
       at ``match.a_id``, ``match.b_id``, ``match.wire`` -- the exact same function
       :func:`~qufzx.rewrite.match.find_matches` calls to decide whether this is a fusion
       candidate at all. This is what actually verifies ``node_a.generator_type ==
       node_b.generator_type`` (A1: a match's ``side_condition_outcomes`` claiming
       ``same_generator_type`` passed, for an actual Z/X pair, is caught here -- coverage
       alone cannot catch a fabricated-passing outcome, only a missing or duplicate one)
       and that ``match.shared_dim``/``match.bindings`` actually relate to the ports being
       assigned them (A2: re-derived fresh, then checked for exact agreement with what
       ``match`` itself claims, rather than the pre-fix behavior of assigning
       ``match.shared_dim`` to every surviving port on faith). Any disagreement, or a
       freshly-failing side condition, raises
       :class:`~qufzx.rewrite.rule.RewriteDomainError` -- a match asserting something that
       does not hold is a domain violation, the same category
       :func:`check_side_condition_coverage` already uses for a failed or incomplete
       outcome tuple, not a different error class for what is the same kind of defect.
       Everything from here on builds from ``resolution``'s fields, never ``match.shared_dim``
       or ``match.bindings`` directly, even though they are now known to agree.
    4. ``match.dimension_constraints`` and ``match.side_condition_outcomes`` themselves,
       checked for exact agreement with ``resolution.dimension_constraints``/``resolution
       .outcomes`` (Phase 5 post-closing audit, Defect 2) -- the same disagreement-is-
       ``RewriteDomainError`` policy step 3 already applies to ``shared_dim``/``bindings``,
       extended to the two fields :mod:`qufzx.rewrite.engine`'s certificate actually reads.
       Before this check existed, a match could claim (via a fabricated
       ``side_condition_outcomes`` tuple, or a hand-edited ``dimension_constraints``) to
       have assumed nothing, or something different from what was actually assumed, and
       ``apply`` would record that unaudited claim onto ``RewriteStep`` verbatim -- this
       builder now returns ``resolution.outcomes``/``resolution.dimension_constraints`` as
       :attr:`~qufzx.rewrite.rule.BuildResult.verified_side_condition_outcomes`/
       :attr:`~qufzx.rewrite.rule.BuildResult.verified_dimension_constraints` instead, which
       ``apply`` prefers over ``match``'s own fields when recording the certificate.

    A structurally malformed match -- ``match.a_id == match.b_id``, either node id absent
    from ``diagram``, ``match.wire`` not actually incident on both, or ``match.wire`` not
    actually an element of ``diagram.wires`` -- surfaces as
    :class:`~qufzx.rewrite.rule.RewriteGrammarError` from step 3 itself (via
    :func:`~qufzx.rewrite.match.resolve_fusion_match`), the same class this builder raised
    for that case before this refactor.
    """
    if not isinstance(match, FusionMatch):
        raise RewriteGrammarError(
            f"spider_fusion requires a FusionMatch, got {type(match).__name__}"
        )
    # Reads the module-level FUSION_SIDE_CONDITIONS constant directly, not
    # spider_fusion_builder.side_conditions (Phase 5 post-closing audit round 19, Task 4):
    # the latter is a self-reference to this function's own global name from inside its own
    # body, which breaks under any rename/wrap of spider_fusion_builder that does not also
    # update this call site -- a class of brittleness worth closing structurally rather
    # than accepting. The function-object attribute itself is kept (see its own docstring,
    # just below the function) purely for Rule.__post_init__'s constructor-time consistency
    # check, which is a plain module-level assignment, not a self-reference, and so does
    # not share this fragility.
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
    # Defect 2 (Phase 5 post-closing audit): the certificate must record what this rewrite
    # actually assumed, not merely what the match claims it assumed -- the same class of gap
    # A2 already closed for shared_dim/bindings. match.dimension_constraints and
    # match.side_condition_outcomes are never trusted for the certificate either: they must
    # agree exactly with what resolve_fusion_match just derived fresh, or this raises here,
    # before graph surgery, rather than letting a fabricated pair of fields reach
    # qufzx.rewrite.engine.apply and be recorded verbatim onto RewriteStep.
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

    # node_a.generator_type is only safe to build the merged node from because
    # ``resolution.passed`` above already confirmed node_a.generator_type ==
    # node_b.generator_type (condition 2) -- see step 3 in this function's docstring.
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

Set on the function object itself so :class:`~qufzx.rewrite.rule.Rule`'s constructor-time
consistency check can compare it against whatever ``side_conditions`` a ``Rule`` wrapping
this builder declares -- see that check's docstring and A5 in the Phase 5 round-12 audit
brief: two contradicting tuples for the same builder must be impossible to construct, not
merely undocumented. This is its only reader: :func:`spider_fusion_builder`'s own body
checks coverage against the module-level ``FUSION_SIDE_CONDITIONS`` constant directly
(Phase 5 post-closing audit round 19), not this attribute, so a rename of
``spider_fusion_builder`` cannot silently desync the runtime check from this one -- only
``Rule.__post_init__``'s already-enforced construction-time comparison depends on this
attribute now.
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

See the module docstring's "Rule registry" paragraph for why this exists. A
``MappingProxyType`` so a caller cannot mutate the registry through the reference
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
