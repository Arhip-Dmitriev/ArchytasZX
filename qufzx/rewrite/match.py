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

"""The fusion matcher: finds occurrences of same-color spider fusion, and only that pattern.

Phase 5 implements exactly one :class:`~qufzx.rewrite.rule.Pattern`: two spiders of the
same generator type (both Z or both X), joined by a wire whose connected legs agree on
dimension -- not necessarily by only that wire: a pair joined by k wires yields one
candidate per wire, each fusing across its own and leaving the rest as self-loops on the
merged node (condition 3 below). Every other pattern shape (bialgebra, Hopf, copy,
identity removal, ...) is out of scope until Phase 11.

Side conditions checked, in the order applied below (see ``FUSION_SIDE_CONDITIONS`` for
their declared names and one-line descriptions, and :class:`FusionMatch` for where their
per-candidate outcomes are recorded):

1. ``distinct_nodes`` -- the two nodes are not the same node. Enforced structurally: a
   wire whose two endpoints share a node id (a self-loop on one spider) is excluded from
   candidate grouping entirely, before any other condition is even reached, since a
   self-loop is not a joining wire between two nodes at all.
2. ``same_generator_type`` -- both nodes carry the identical, registered
   :class:`~qufzx.diagram.generators.GeneratorType` (Z/Z or X/X; Z/X never matches). "identical"
   and "registered" are both this condition's own responsibility, reported as one fact, not
   two: a same-typed pair whose shared type is not in this pattern's registered fusable set
   (``Z_SPIDER``/``X_SPIDER`` today) fails here, with that reason, rather than passing this
   condition and having the real cause surface only as an aside inside condition 4's detail
   (Phase 5 post-closing audit -- the pre-fix behavior folded two distinct facts, "same type"
   and "a fusable type", into this condition's True/False alone, mislabeling the latter as a
   pass here and burying its actual reason in a condition whose own declared description is
   about wire direction, not fusability).
3. ``parallel_wires_become_self_loops`` -- a pair joined by k wires yields up to k
   candidates, one per wire: each fuses across that one (consumed) wire, leaving the other
   k-1 as self-loops on the merged node. Both endpoints of a leftover wire are on the two
   consumed nodes, hence both land in the builder's ``port_mapping``, so ``apply``'s step-5
   remap turns it into an ordinary self-loop -- verified against the oracle at d = 2, 3, 5
   for both colors, with ``Scalar.one()``. Every other condition still applies per
   candidate, independently, to that candidate's own consumed wire only -- condition 4 in
   particular, so a pair joined by wires of different directions can yield fewer candidates
   than wires (e.g. an X pair with one OUTPUT-INPUT and one OUTPUT-OUTPUT wire matches once,
   not twice).
4. ``consumed_wire_direction_permitted_for_color`` -- color-conditioned, per the Phase 5
   final fix round's Step 4 decision. For X, the consumed wire must run from an OUTPUT port of one
   node to an INPUT port of the other; a same-direction (OUTPUT-OUTPUT or INPUT-INPUT)
   wire between two X spiders is refused outright (not matched). This half is load-bearing,
   not cosmetic: per :mod:`qufzx.semantics.denote`'s axis convention, X applies the Fourier
   matrix ``F`` to output axes and ``conj(F)`` to input axes. An output-to-input
   contraction pairs one ``F`` with one ``conj(F)`` on the shared leg, giving
   ``F^dagger F = I`` -- fusion is scalar-free. An output-to-output (or input-to-input)
   contraction instead pairs ``F`` with ``F``, giving ``F^T F``, a nontrivial permutation
   matrix, not the identity -- that is a different (and, for Phase 5, unimplemented) rule,
   not fusion. For Z, by contrast, the spider tensor is diagonal in every axis (see
   :func:`~qufzx.semantics.denote._z_tensor`) and :mod:`qufzx.semantics.contract_numeric`
   contracts a wire by assigning its two endpoints the same einsum axis label regardless of
   direction -- no conjugation is ever applied at contraction time, only at ``denote()``
   time, and only for X. A Z-Z wire of *any* direction (OUTPUT-OUTPUT, INPUT-INPUT, or
   OUTPUT-INPUT) therefore identifies the same basis index ``k`` on both sides and is
   genuinely, numerically valid fusion -- verified against the oracle at ``d = 2``,
   ``d = 3``, and ``d = 5`` -- so this condition permits any direction combination for Z.
   Before this decision, the condition was applied uniformly to both colors (a
   same-direction Z-Z wire was silently never matched), which was sound but incomplete
   relative to FULL_PLAN.md's
   Phase 5 spec ("two same-color spiders joined by a wire and sharing a dimension" states
   no direction restriction). :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`
   needed no change for this widening: it selects the consumed ref by identity against
   ``match.wire``, never by direction, so it already builds the correct merged node
   (correct surviving legs, correct phase, ``Scalar.one()``) for a same-direction Z-Z match
   exactly as it does for an alternating one.
5. ``consumed_ports_singly_claimed`` -- neither consumed port (``ref_a``/``ref_b``) is
   claimed by a second wire, and neither is listed on either boundary list. Decided from
   ``diagram`` alone inside :func:`resolve_fusion_match`, before the dimension fixpoint
   (conditions 6-7) runs at all, since a candidate that fails this has no legal
   ``port_mapping`` for :mod:`qufzx.rewrite.engine`'s ``apply`` to build regardless of what
   the dimension checks would have found -- see the module docstring's "Match-implies-
   applicable and multiply-claimed ports" section below for the full account of why this
   check exists and why it now lives here rather than only in :func:`find_matches`.
6. ``dimension_agreement`` -- the two connected legs' :class:`~qufzx.algebra.dimension.Dim`
   are equal, or :meth:`~qufzx.algebra.dimension.Dim.unify` defers or succeeds only by
   binding a symbol (both recorded as a :class:`FusionMatch` dimension constraint -- see
   below -- not silently accepted). A ``FAILURE`` from ``unify`` is a non-match. The
   match's ``shared_dim`` is *not* simply the A-side leg's raw ``Dim``: when ``unify``
   succeeds by binding a symbol (e.g. leg dims ``d`` and ``3`` bind ``d := 3``), the
   binding is substituted into the A-side leg's ``Dim`` (see
   :meth:`~qufzx.algebra.dimension.Dim.substitute`) to produce a provisional ``shared_dim``,
   so it is ``3`` in that example, not the still-unbound ``d``. When ``unify`` only defers,
   or succeeds with no binding at all (a bare syntactic identity), the provisional
   ``shared_dim`` is the A-side leg's raw ``Dim``, unchanged. This condition does not stop
   at the connecting pair: every *surviving* leg of both nodes (every leg other than the
   one consumed by the matched wire) is then unified against the provisional
   ``shared_dim``, in turn, per node in input-then-output, original-index order (A's legs,
   then B's) -- not the same global order
   :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` builds the merged node's
   ports in (A's inputs, B's inputs, then A's outputs, B's outputs); the difference is
   confined to the recorded order of ``dimension_constraints`` and which symbol in a chain
   binds first, both already deterministic, never to a match/non-match decision. A later
   leg is checked against whatever ``shared_dim`` an earlier leg's binding refined it to, so
   a chain such as leg dims ``d``, ``d``, ``2``
   resolves to a final ``shared_dim`` of ``2`` and every step along the way is recorded. A
   ``FAILURE`` on any surviving leg makes the whole candidate a non-match, exactly like a
   ``FAILURE`` on the connecting pair itself: forcing a genuinely non-unifiable surviving
   leg onto ``shared_dim`` anyway would silently rewrite that leg's dimension (or erase a
   pre-existing hard dimension conflict on it) rather than report the conflict. A
   ``DEFERRED`` or binding-only ``SUCCESS`` on a surviving leg is recorded as a dimension
   constraint exactly like the connecting pair's own (see "Dimension constraints" below),
   and, if it binds, the newly-refined ``shared_dim`` carries forward to every leg checked
   after it. Only once every surviving leg has been checked is ``shared_dim`` final.

   Forcing every surviving leg onto ``shared_dim`` is exactly what can make a wire that was
   an exact dimension match *before* this fusion merely deferred *after* it: a neighbouring
   node's wire to a surviving leg agreed with that leg's own original ``Dim`` exactly, but
   the merged node now presents that leg at ``shared_dim`` instead, and the neighbour's own
   wire dimension was never touched -- so :mod:`qufzx.diagram.validate`, re-run at step 8,
   may see a pair that no longer unifies as a bare identity. This is a routine outcome of
   this condition, not a marginal edge case: instrumented over
   ``tests/test_fusion_properties.py``'s random property harness, roughly one in ten
   applications this module's own matches produce trips exactly this path (measured: 22 of
   193 applications over the harness's seed range) and is caught, and correctly permitted,
   by :mod:`qufzx.rewrite.engine`'s own step-8 relative postcondition -- see that module's
   docstring for why a *newly introduced* hard-error issue like this is legitimate rewrite
   behavior, not a bug, and ``tests/test_match.py::TestSurvivingLegOverwriteIntroducesDeferral``
   for a deliberately constructed, named regression pinning the shape rather than leaving it
   to be rediscovered as folklore. Widening the matcher to reject these candidates outright
   would make this pattern stricter than FULL_PLAN.md's Phase 5 statement requires and is
   Phase 10/11 territory (a real diagram-level unifier, and/or a strategy layer that can
   choose a fusion order that avoids the conflict) -- not this phase's job.
7. ``phase_dimension_agreement`` -- every phase vector actually present (on either node, or
   both) must :meth:`~qufzx.algebra.dimension.Dim.unify` with the resolved ``shared_dim``
   from condition 6 (Phase 5 post-closing audit, judgement call 2, decided: this condition
   now calls ``unify`` itself, the same way condition 6 does for a leg -- see below for why
   the earlier plain-``Dim``-equality version was retired rather than merely re-documented).
   Unlike condition 6, though, a ``DEFERRED`` outcome here is treated exactly like a
   ``FAILURE`` -- rejected, not accepted-and-recorded -- because a leg and a phase are not
   symmetric: a leg carries no internal expression tied to its own dim, so forcing it onto
   ``shared_dim`` under an unproven ``DEFERRED`` assumption is safe (the leg *is* its dim;
   the assumption is simply recorded in ``dimension_constraints`` and the leg's new value is
   ``shared_dim``, full stop). A phase's entries, by contrast, can directly reference its own
   ``dim``'s free symbols (e.g. a root-of-unity entry ``index/d`` on a ``PhaseVector`` whose
   own ``dim`` is ``d``), and reattaching those entries to a *different*, more-resolved
   ``shared_dim`` is only correct when there is an actual concrete binding to
   :meth:`~qufzx.algebra.phase.PhaseVector.substitute` through them first; a ``DEFERRED``
   unify produces no binding, so reattaching anyway would silently leave a stale symbol
   baked into an entry under a container ``dim`` that no longer mentions it -- exactly the
   ``_over_shared_dim`` defect family this module and :mod:`qufzx.rewrite.rules_library`
   fight elsewhere. (This asymmetry was found empirically, not derived up front: an earlier
   version of this fix let ``DEFERRED`` through uniformly with condition 6, and the
   randomized oracle harness in ``tests/test_fusion_properties.py`` caught the resulting
   stale-symbol construction on its very next run.) Only ``SUCCESS`` -- a bare identity or a
   binding -- is accepted. A binding-only ``SUCCESS`` is recorded as a
   :class:`~qufzx.rewrite.rule.ConstraintSourceKind.NODE_PHASE`-sourced
   :class:`~qufzx.rewrite.rule.DimensionConstraint` in the same record condition 6's own leg
   checks write into -- see "Dimension constraints" below -- and folded into the
   whole-candidate ``bindings`` accumulator for :func:`reattach_phase` to use, exactly like a
   surviving leg's own binding.

   Round-12 audit defects 1 and 2, fixed: a phase's own binding *is* used to further refine
   ``shared_dim`` -- the claim that "``shared_dim`` is already final by the time this
   condition runs, since nothing about a node's phase determines what its legs share" was
   false for these generators (see :attr:`~qufzx.diagram.generators.GeneratorType.phase_schema`
   ``TIED_TO_LEG_DIM``, :mod:`qufzx.diagram.validate`, which ties a node's phase dimension to
   its legs' shared dimension explicitly) and was the root cause of two defects: two present
   phases could each bind the same symbol against the same stale, unrefined ``shared_dim``
   (last-write-wins, silently accepting a contradiction the equivalent all-leg shape already
   refused), and a phase's own binding was reattached into the merged phase's entries via
   :func:`reattach_phase` without that same binding ever refining the merged node's *legs*,
   so the built node's legs and its phase could disagree on the very value the binding
   assumed. Fixed by giving this condition the same accumulator discipline condition 6 has
   (:func:`_unify_phase_dims`: each present phase's ``Dim`` is resolved through the running
   ``bindings`` accumulator first, then unified against the *current* ``shared_dim``,
   refining both in place on a concrete binding before the next phase is examined), and by
   folding conditions 6 and 7 into one bounded fixpoint that re-runs the surviving-leg sweep
   whenever a phase's binding has refined ``shared_dim`` past what the legs were last checked
   against -- see :func:`resolve_fusion_match`'s own inline commentary for the fixpoint's
   mechanics and termination argument.

   For every phase actually present, this condition also verifies that :func:`reattach_phase`
   -- substituting the accumulated concrete bindings (every one this fixpoint accumulated,
   from condition 6's leg checks and condition 7's own per-phase unifies alike) into the
   phase's entries (via :meth:`~qufzx.algebra.phase.PhaseVector.substitute`, never leaving a
   stale dimension symbol baked into an entry once its own binding has resolved
   ``shared_dim`` past it; see that function's docstring for why a substituting builder, not
   a stricter matcher, is the chosen resolution of the ``_over_shared_dim`` defect family)
   then reattaching the result to the fixpoint's *final* ``shared_dim`` -- succeeds;
   substitution changes only an entry's *value*, never its index, so an index that falls
   outside ``shared_dim``'s range once a phase's own symbol resolves (e.g. a phase legally
   stated over symbolic ``d`` with an entry at index 5, binding ``d := 2``, where index 5 is
   out of range once ``d`` really is ``2``) is still caught here, exactly as before, and
   against the value the merged node will actually be built at, not an intermediate one from
   an earlier fixpoint pass. A :class:`~qufzx.algebra.phase.PhaseDomainError` here is a
   failed condition (non-match), never let escape from the builder later. A failure of this
   condition -- a non-unifying phase dimension, or a :class:`PhaseDomainError` from
   :func:`reattach_phase` -- routes through the same placeholder-producing failure path
   (``_failed``) every other failing condition in this function uses, rather than falling
   through to a bespoke return: see :class:`FusionResolution`'s own docstring for the
   invariant this preserves (``shared_dim`` is ``None``, and ``bindings``/
   ``dimension_constraints`` are empty, on any failure without exception).
   :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls the very same
   :func:`reattach_phase` to actually build the merged phase, so this makes match-approval
   and build-applicability the same predicate by construction, not two predicates kept in
   sync by hand: the invariant is that every match this function returns can be applied by
   :func:`~qufzx.rewrite.engine.apply` without raising anything except the step-8 relative-
   postcondition :class:`~qufzx.rewrite.rule.RewriteDomainError`.

   This condition's own ``SideConditionOutcome.deferred`` is unconditionally ``False`` on a
   passing outcome: since a genuinely ``DEFERRED`` per-phase unify is rejected outright above
   rather than accepted-and-flagged (unlike condition 6's own ``leg_deferred``), a *passing*
   ``phase_dimension_agreement`` outcome is never itself resting on an undecided unify -- at
   most on a binding, which ``dimension_constraints`` records but which, following condition
   6's own convention (a binding-only ``SUCCESS`` does not set ``leg_deferred`` either), this
   flag does not count as "deferred".

   Decided, not merely re-documented (Phase 5 post-closing audit, judgement call 2): the
   prior version checked plain ``Dim`` equality after substituting only condition 6's own
   already-accumulated bindings, and additionally required two *present* phases' raw,
   unsubstituted ``Dim``\\ s to equal each other outright. Both restrictions were audited and
   rejected as unnecessary conservatism, not merely documented as such, once
   :func:`reattach_phase` existed to do the substitution:

   * Plain equality silently refused a phase whose ``Dim`` unifies with ``shared_dim`` only
     via a *new* binding this condition itself would have to produce -- e.g. a phase stated
     over a free symbol ``e`` that appears on no leg at all, where ``shared_dim`` is
     concrete ``3``: :mod:`qufzx.diagram.validate` itself accepts this diagram outright
     (``unify`` binds ``e := 3``, no issue reported), yet the pre-fix matcher silently
     refused it as a non-match, since ``e`` and ``3`` are not equal as raw expressions and
     condition 6 never had any reason to bind ``e`` itself. Calling ``unify`` here, exactly
     as condition 6 already does for a leg, closes this for real rather than re-describing it
     as Phase 10's job.
   * The "both present phases' raw Dims must equal each other" check was never actually
     load-bearing: :func:`reattach_phase` forces *both* operands' container ``Dim`` to the
     same ``shared_dim`` regardless of what their raw, unsubstituted ``Dim``\\ s were (see
     that function's docstring), so :meth:`~qufzx.algebra.phase.PhaseVector.__add__`'s
     equal-``Dim`` requirement is already guaranteed to hold by the time the builder adds
     them -- a phase over ``d`` and a phase over ``3``, with ``d`` bound to ``3`` elsewhere,
     reattach to the identical ``PhaseVector(3, ...)`` shape either way. The extra check
     bought no soundness (nothing it rejected could otherwise have crashed the builder or
     denoted something unintended) and cost real completeness, so it is removed rather than
     kept and merely re-labeled "stricter than necessary".

Malformed wire references. :mod:`qufzx.diagram.graph` is deliberately permissive about
what a :class:`~qufzx.diagram.graph.Wire` may name (see that module's docstring on
validation ownership), so an un-validated diagram can hold a wire endpoint naming a node
id absent from the diagram, or one present but with an out-of-range port index for that
side. :func:`find_matches` checks both endpoints of *every* wire in the diagram for both
faults -- via :func:`_validate_wire_endpoint` -- in a first pass that runs before any
candidate grouping or filtering: before the self-loop skip, before wires are grouped by
node pair, before the parallel-wire-pair (``len(connecting_wires) != 1``) filter, and
before any other candidate property (generator color, fusable-color-ness, wire direction)
is even reached. It raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` naming the
offending :class:`~qufzx.diagram.graph.PortRef` (and, for the index case, the node's
actual leg count) -- the same treatment :mod:`qufzx.diagram.validate` gives both as hard
errors (``UNKNOWN_NODE``, ``PORT_INDEX_OUT_OF_RANGE``), rather than letting either escape
this module's declared error hierarchy as a bare ``KeyError``/``IndexError`` or passing
silently as a non-match. This ordering is deliberate and load-bearing: detection of a
malformed wire must not depend on any other property of the wire or the candidate pair it
happens to sit on, so a wire with an out-of-range port index (or an unknown node id) is
rejected identically whether it joins two Z spiders, a Z and an X, a pair with matching
wire directions, a pair also joined by a second parallel wire, or even both of its own
endpoints on the very same node (a self-loop) -- every one of those shapes used to make
the malformed-wire check unreachable for that wire specifically, since each is dropped
(via `continue`, without ever raising) by a filter that used to run first.

Malformed boundary references (Phase 5 post-closing audit round 18, Defect 2). The exact
same permissiveness :mod:`qufzx.diagram.graph` extends to a wire endpoint applies equally
to a ``boundary_inputs``/``boundary_outputs`` entry: nothing in that module stops a caller
from listing a boundary ``PortRef`` naming an unknown node id or an out-of-range index.
:mod:`qufzx.rewrite.engine`'s ``_remap_endpoint`` (see that module's docstring, step 5)
holds a boundary entry to the identical standard as a wire endpoint -- both are ``PortRef``
lookups against the same working diagram -- yet, before this fix, only the wire side of that
symmetry was checked here; a malformed boundary entry reached ``apply`` unexamined unless it
happened to sit on a port a match's builder was about to remap, in which case it surfaced
late, as a *different* error (``RewriteDomainError`` from ``apply`` step 5, not
``RewriteGrammarError`` from this module) than the one this module's own contract promises
for a malformed reference. Fixed by extending the pre-pass above: every entry of both
boundary lists is checked via the same :func:`_validate_wire_endpoint` (its
``wire_or_boundary_ref`` parameter accepts a bare :class:`~qufzx.diagram.graph.PortRef` for
exactly this call shape, not only a :class:`~qufzx.diagram.graph.Wire`) before any candidate
grouping, so a malformed boundary entry is now structurally excluded the same way a
malformed wire endpoint already was -- the same fix, applied to the other reference kind the
"match-implies-applicable" contract below was silently narrower than it claimed to be.
``_remap_endpoint``'s own raise remains the defensive check against a foreign or hand-built
``Match`` that never went through this pre-pass at all -- see its docstring in
:mod:`qufzx.rewrite.engine`.

This is not in tension with :mod:`qufzx.rewrite.engine`'s own step 8, which says a diagram
that already carries a hard-error *validation issue* (e.g. an unwired non-boundary leg,
``IssueKind.PORT_UNUSED``) is legitimately rewritable and must not be blocked from firing --
the two policies are about different failure categories entirely, and this module's own
:mod:`qufzx.rewrite.rule` already draws exactly this line for the package as a whole
(``RewriteDomainError`` for a value or state outside the mathematical domain a rewrite
requires, ``RewriteGrammarError`` for a malformed request). A hard-error validation issue is
a defect in an otherwise-coherent *request*: every reference involved still names a real
port, so the request can be evaluated, and evaluating it (finding a match, or applying one)
is exactly how a rewrite is allowed to carry a pre-existing defect forward, resolve it, or
leave it alone -- step 8 is precisely the mechanism that lets that happen without blocking.
A boundary entry (or wire endpoint) naming no real port at all is a different kind of thing:
there is no diagram state to evaluate a request against, only a reference that resolves to
nothing, so it cannot be "carried forward" by a rewrite in any sense -- it is not a
mathematical fact about the diagram that a rewrite might legitimately act on or ignore, it is
a malformed description of one. The wire-endpoint pass already drew this line (a malformed
wire endpoint has always raised ``RewriteGrammarError`` here, never been treated as "a
pre-existing issue to carry forward"); this fix is that same, already-settled call applied
uniformly to the boundary case, not a new policy question.

Match-implies-applicable and multiply-claimed ports. A port that is claimed by more than
one wire (:class:`~qufzx.diagram.validate.IssueKind.PORT_WIRED_TWICE`), or that is both
wired and listed on a boundary
(:class:`~qufzx.diagram.validate.IssueKind.PORT_WIRED_AND_BOUNDARY`), is not treated as a
fusion occurrence even when it happens to be the port a candidate wire would consume.
:mod:`qufzx.rewrite.engine`'s ``apply`` requires every consumed port to appear in the
builder's ``port_mapping``, but a builder only ever maps *surviving* ports -- the consumed
port itself is deliberately absent, since it no longer exists on the merged node. If that
same port is also named by a second wire (to a third node) or a boundary entry, ``apply``
would need to remap that second reference too, and cannot: the port is gone, and nothing
in the match or the builder says what it should become.

Phase 5 post-closing audit round 23, Task 2: this used to be enforced by a bare ``continue``
inside :func:`find_matches` alone -- a candidate whose consumed port was multiply-claimed
was silently dropped from the returned tuple, with nothing about the check appearing in
``FUSION_SIDE_CONDITIONS``, ``side_condition_outcomes``, or anywhere a Phase 6 certificate
could show it was ever evaluated. Two consequences that admission tracked but did not fix:
:func:`~qufzx.rewrite.rules_library.spider_fusion_builder`'s "match-approval and
build-applicability are the same predicate by construction -- literally the same function
object" claim was false for exactly this condition (:func:`resolve_fusion_match` never
decided it, so a hand-built or foreign :class:`FusionMatch` carrying a multiply-claimed
consumed port sailed through the builder and only failed in ``apply`` step 5, as a
:class:`~qufzx.rewrite.rule.RewriteDomainError` -- a different error class, from a different
module, than this module's own "malformed reference" contract promises for the shape); and
the certificate simply had no row for it. Promoted to condition 5,
``consumed_ports_singly_claimed``, decided by :func:`resolve_fusion_match` itself from
``(diagram, a_id, b_id, wire)`` alone, exactly like every other condition -- placed *before*
the dimension fixpoint (conditions 6-7) since it gates whether a legal ``port_mapping`` can
even exist, independent of any dimension question, and a candidate that fails it should not
burn fixpoint work deciding shared-dimension resolution for a match that could never be
applied regardless of the answer. :func:`find_matches`'s own ``continue`` filter is deleted;
the candidate is now dropped by ``resolution.passed`` being ``False``, the same single
decision point every other condition already goes through -- see :func:`_remap_endpoint`
in :mod:`qufzx.rewrite.engine` for the matching update on that side (its own defensive
raise is now provably unreachable via any match this module's :func:`find_matches` builds).
This is what makes "every match this function returns can be applied by
:func:`~qufzx.rewrite.engine.apply` without raising anything except the step-8
relative-postcondition" (see condition 7 above) true without qualification, by construction
rather than by a filter a reader would otherwise have to know to go looking for outside the
resolution function itself: :mod:`qufzx.rewrite.engine`'s own docstring states the matching
half of this resolution on its side.

Dimension constraints. A :class:`FusionMatch`'s ``dimension_constraints`` records every
dimension equality that :func:`find_matches` did not verify as a syntactic identity but
still accepted: both a ``DEFERRED`` outcome from :meth:`Dim.unify` (truly undecided, per
that method's contract) and a ``SUCCESS`` outcome that only holds because ``unify`` bound a
free symbol (decided, but only under that binding -- e.g. leg dims ``d`` and ``3`` unify by
binding ``d := 3``, and the fusion is valid only at that value). This applies to the
connecting pair (:class:`~qufzx.rewrite.rule.ConstraintSourceKind.CONNECTING_PAIR`), to
every surviving leg condition 6 checks against the running ``shared_dim``
(``SURVIVING_LEG``, identified by its own ``(NodeId, Direction, index)``), and to every
present phase's own ``Dim`` condition 7 checks against the (by then final) ``shared_dim``
(``NODE_PHASE``, identified by its node id -- but, unlike the connecting pair and every
surviving leg, only for a binding-only ``SUCCESS``, never for ``DEFERRED``: condition 7
rejects a genuinely ``DEFERRED`` phase-dim outright rather than accepting and recording it,
see condition 7 above for why a phase cannot tolerate the same assumption a leg can). All
are assumed equalities a diagram-level unifier (Phase 10) must eventually justify, so all
belong in the certificate; only a unify outcome that is a bare syntactic identity (no
binding, nothing deferred) is left out, since nothing was assumed.

Each entry is a :class:`~qufzx.rewrite.rule.DimensionConstraint`, keyed by its
:class:`~qufzx.rewrite.rule.ConstraintSource` -- see that class and
:class:`~qufzx.rewrite.rule.ConstraintSourceKind` for the discriminator and why the record
is source-keyed rather than one bare ``(Dim, Dim)`` pair per check. Conditions 6 and 7 run
as one bounded fixpoint (see :func:`resolve_fusion_match`'s own inline commentary), so
"the connecting pair", "every surviving leg", and "every present phase" above each mean
every check across *every* pass of that fixpoint, not only its first or its last -- the
connecting pair is re-derived every pass exactly like a surviving leg or phase (D2, Phase 5
audit round 15; :func:`_unify_connecting_pair`), not recorded once up front and then never
revisited. A source checked on more than one pass is recorded exactly once, at its
most-resolved statement of itself: :class:`_ConstraintRecord` (see its own docstring)
replaces the entry for a source a later pass re-derives, keyed by
:class:`~qufzx.rewrite.rule.ConstraintSource`, rather than appending a second one. This
holds for a source whose every re-check comes back a bound value (the entry is replaced with
the same fact, harmlessly) and for one whose re-check comes back a bare identity once an
earlier pass' binding has since discharged it (the entry is dropped -- see
:meth:`_ConstraintRecord.record_identity`), so a duplicate-of-itself assumption is never
double-counted either way. Enforced exhaustively, over the whole finite
space this module's own hand-picked unit tests only sample a few shapes of, by
``tests/test_phase5_certificate_sweep.py::TestCertificateStructuralProperties`` (no
duplicate sources, no bare-identity entries) and
``TestOracleTiesBackToRecordedConstraints`` (every recorded constraint is load-bearing: the
pre- and post-fusion diagrams agree with the oracle at a substitution satisfying it, and
disagree at one violating it). Pinned at specific shapes by
``tests/test_match.py::TestDimensionConstraintsRecording`` and
``tests/test_engine.py::TestDimensionConstraintsExactContent``.

The corresponding ``SideConditionOutcome.deferred`` flag on ``dimension_agreement``
(condition 6) is computed from the *finished* record (:meth:`_ConstraintRecord
.any_leg_deferred`), not from a flag accumulated pass-by-pass, so a leg that deferred on one
pass and was discharged (bound, or resolved to identity) on a later one does not leave a
stale ``deferred=True`` -- see ``tests/test_match.py
::TestPhaseDimensionAgreementDeferredFidelity``. It is ``True`` iff the finished record
still holds a ``DEFERRED`` connecting-pair or surviving-leg entry; a check that only ever
bound symbols, with none deferred, is reported with ``deferred=False`` even though it, too,
is in ``dimension_constraints``. On ``phase_dimension_agreement`` (condition 7), by
contrast, this flag is unconditionally ``False`` on a passing outcome -- see condition 7
above.

Non-concrete bindings. :meth:`~qufzx.algebra.dimension.Dim.unify` can succeed by binding a
symbol to another still-symbolic ``Dim`` (e.g. ``d`` against ``e`` binds ``d := e``), not
only to a concrete one. :meth:`~qufzx.algebra.dimension.Dim.substitute` accepts only a
concrete replacement value (by contract, not merely by current limitation -- see that
method's own docstring), and :class:`~qufzx.algebra.phase.PhaseVector.substitute`'s
``PhaseSubstituteValue`` is likewise ``int | Rational | Phase``, so a symbol-to-symbol
binding is not expressible through the Phase 1/2 substitution APIs at all; solving it is
:meth:`Dim.unify`'s own docstring's explicit Phase 10 carve-out ("the real unifier ... must
actually solve constraints such as d = d1 * d2"), not an oversight here. Every site in this
module that consumes a binding -- :func:`_resolve_with_bindings`, :func:`_merge_bindings`,
and (with one addition below) :func:`_unify_phase_dims` -- therefore treats a non-concrete
binding identically: it is dropped from ``bindings``/``shared_dim`` and never substituted
through, exactly as if nothing had been bound. This is silent only in the sense that it does
not raise; it is not silent in the sense of "lost" at the moment the check runs: whatever
check produced the binding records it as a :class:`~qufzx.rewrite.rule.DimensionConstraint`
(``BOUND``, per source) in the same record every other assumption goes into.

Phase 5 post-closing audit round 23, Task 1 correction: that entry is not guaranteed to
survive to the *finished* record under its own source's key, though -- a later pass that
re-derives the same source (its own resolved operands now more concrete, because some other
source's own concrete binding has since been folded into ``bindings``) overwrites it, and
the overwritten form need not itself be a ``BOUND`` restating the same non-concrete binding
(see :class:`_ConstraintRecord`'s own docstring for the full displacement policy this
provoked, and why the assumption is not lost even when the specific entry that first stated
it is). What is still true, and is the property that actually matters here: the assumption
is *adequate* -- implied by the finished ``dimension_constraints`` together with the
finished ``bindings`` -- not that it is recorded verbatim, under its original source, at its
first-derived non-concrete form.

For a surviving leg, that is the end of it: a leg is its own ``Dim`` and references no
internal structure tied to it, so leaving a non-concrete binding unused costs completeness
(condition 6 might resolve ``shared_dim`` less than a real solver could) but nothing more.
For a phase (condition 7), :func:`_unify_phase_dims` goes one step further and rejects the
whole candidate outright on a non-concrete binding, not merely on a ``DEFERRED`` unify --
because a phase's own entries can directly reference its ``dim``'s free symbols (a
root-of-unity entry ``index/d`` under a ``PhaseVector`` whose own ``dim`` is ``d``), so
:func:`reattach_phase` reattaching such a phase to a *different*, more-resolved
``shared_dim`` is only correct with an actual concrete binding to substitute through first;
without one, silently reattaching would leave a stale symbol baked into an entry under a
container ``dim`` that no longer mentions it. A leg carries no such entries, so it has
nothing analogous to protect. This asymmetry is deliberate, not an inconsistency to be
flattened: recorded as a Phase 10 item once, here, rather than as three separate asides.

No :class:`FusionMatch` is ever constructed for a failing candidate. Conditions 1 and 3 are
structural: :func:`find_matches` excludes a self-loop from candidate grouping entirely
(condition 1), and condition 3 is a statement about what happens to the *other* wires
joining an accepted pair, so neither can fail for a candidate that reaches
:func:`resolve_fusion_match` at all -- both are reported unconditionally True. Conditions 2
and 4-7 are decided per candidate by :func:`resolve_fusion_match`, which reports the failing
one (and every condition after it, as "not evaluated") in its :class:`FusionResolution`;
:func:`find_matches` then drops that candidate on ``resolution.passed`` being ``False``,
never returning it as a match with a False outcome. Every :class:`FusionMatch` this module
returns therefore has ``all_side_conditions_passed`` True by construction. This mirrors
:mod:`qufzx.diagram.validate`'s existing deferred/hard-failure split for dimension issues.

Round 24: this paragraph used to split the seven conditions as "1, 2, or 3 dropped before
any FusionMatch is constructed" versus "4-7 checked per surviving candidate", which described
the pre-round-23 code -- condition 2 (``same_generator_type``) has been decided by
:func:`resolve_fusion_match`, and reported as an ordinary failing outcome, since that round
folded every non-structural condition into the one predicate. The observable contract (a
failing candidate is never returned) was and is unchanged; only the account of *where* each
condition is decided had drifted.

Determinism. :func:`find_matches` sorts its result by node ids, then -- since a pair's node
ids no longer uniquely determine a candidate once condition 3 permits several parallel
wires -- by the consumed wire's own (direction, index) on each side, never by set or dict
iteration order, since certificates and Phase 12's cache tests will compare match lists
directly.

This is one instance of a whole-certificate discipline, not an isolated concern of this
function's own return value (Phase 5 post-closing audit round 18, Defect 1's sweep). Every
set/frozenset iteration in this module whose order could reach a returned value, a recorded
certificate field, or an exception message is sorted the same way, by the same
hash-independent key (:meth:`~qufzx.diagram.graph.Wire.sort_key`,
:meth:`~qufzx.diagram.graph.PortRef.sort_key`) -- never left to a frozenset's own iteration
order, which is ``PYTHONHASHSEED``-dependent because :class:`~qufzx.diagram.graph.Direction`
is an ``enum.Enum`` hashed by member name: :func:`find_matches`' own internal passes over
``diagram.wires`` (malformed-endpoint check, pair grouping) and its boundary-ref validation
pass (Defect 2, below) all iterate a pre-sorted snapshot for exactly this reason -- see each
site's own comment. This closes the same class of defect
:mod:`qufzx.diagram.validate` closes on its side (see that module's docstring): before both
fixes, the *value* of a fusion match's fields was always correct, but the *order* in which a
malformed reference was detected (and, one layer further out, the order
:mod:`qufzx.rewrite.engine`'s ``RewriteStep.removed_deferred_issues``/
``introduced_deferred_issues`` select colliding issues in) could vary by process --
contradicting :attr:`~qufzx.rewrite.engine.RewriteStep.deferred_issue_identity_ambiguous`'s
own "first in validate order" promise. ``tests/test_engine.py::TestCrossProcessDeterminism``
is the end-to-end regression test for the whole chain, comparing a full certificate's worth
of fields, by value and order, across two subprocesses under two different
``PYTHONHASHSEED`` values.

One verification predicate, not two kept in sync by hand (Phase 5 round-12 audit). Prior
rounds' claim that "match-approval and build-applicability are the same predicate by
construction" (condition 7, above) was true of the phase check alone (both callers went
through :func:`_reattach_phase`/:func:`reattach_phase`) but false of the rest of a
candidate's legality: :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` built the
merged node straight from ``node_a.generator_type`` and ``match.shared_dim`` with no check
that ``node_a`` and ``node_b`` actually share a generator type, or that ``shared_dim``
actually related to the ports it was about to be assigned to. A hand-built or foreign
``FusionMatch`` carrying a fabricated-passing ``side_condition_outcomes`` tuple, a Z/X pair,
or a ``shared_dim`` unrelated to the matched legs was applied anyway, producing a diagram
:mod:`qufzx.diagram.validate` called clean and denoting something else entirely.
:func:`resolve_fusion_match` closes this: it is the one function that decides conditions 2
(``same_generator_type``) and 4-7 (``consumed_wire_direction_permitted_for_color``,
``consumed_ports_singly_claimed``, ``dimension_agreement``, ``phase_dimension_agreement``)
from ``(diagram, a_id, b_id, wire)`` alone, never from a pre-existing match's own fields.
:func:`find_matches` calls it once per
candidate wire to decide whether to return a match at all, and to populate that match's
``shared_dim``/``bindings``/``dimension_constraints``/``side_condition_outcomes``.
:func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls it again, fresh, against
the diagram it was actually handed, and uses *only* its return value -- never
``match.shared_dim``, ``match.bindings``, or an unchecked ``node_a.generator_type`` -- for
graph surgery, raising :class:`~qufzx.rewrite.rule.RewriteDomainError` if the match's own
claimed ``shared_dim``/``bindings`` disagree with what this function independently derives.
So "the same predicate by construction" is now literally the same function object called
from both places, not a documented intention two similar-looking computations could drift
out of sync with. Conditions 1 (``distinct_nodes``) and 3
(``parallel_wires_become_self_loops``) remain :func:`find_matches`' own responsibility to
report (though :func:`resolve_fusion_match` still recomputes and reports them itself, from
``diagram`` alone, so a builder-side re-verify sees a complete, independently-derived
seven-outcome picture too) -- they gate whether a wire is a fusion candidate in the first
place (a self-loop is excluded from candidate grouping entirely; the count of "other" wires
joining a pair is a property of the pair, not of any one candidate wire's own legality).

Phase 5 audit round 18 summary. Three classes of defect closed in this module, each
described here at the class level so a hypothetical round 19 knows where to keep looking,
not merely which three instances were fixed:

* Process-dependent ordering (Defect 1). See :mod:`qufzx.diagram.validate`'s own "Phase 5
  audit round 18" note for the general statement; this module's instance was its three
  internal passes over ``diagram.wires``, now sorted (see :func:`find_matches`'s own
  comment).
* A stated invariant enforced on one reference kind but not a structurally identical other
  (Defect 2). "Malformed wire references" was a load-bearing contract this module enforced
  for wire endpoints alone; boundary entries are held to the identical
  ``_remap_endpoint``/``apply`` standard downstream but were not checked here at all. The
  general question for a future reference kind this package grows (e.g. a Phase 7 bang-box
  scope boundary, or a Phase 11 rule's own new match-location fields): does something
  downstream treat this reference the same way it treats one already validated here, and if
  so, is it actually validated here too, or merely assumed to be?
* A guard documented as unreachable that was reachable on a path its own reasoning did not
  cover (Defect 3). :func:`_verify_fixpoint_closure`'s unreachability argument was sound for
  the fixpoint's convergence exit but was invoked unconditionally on every exit, including a
  phase-dim ``FAILURE`` that ``break``s out for an unrelated reason. The general lesson: an
  "unreachable given X's own termination guarantee" claim is a claim about one specific exit
  path, and every other exit from the same loop needs its own, separately-argued path to
  whatever code runs after the loop -- never inherited by proximity.
* A human-readable detail derived from a different source of truth than the machine-readable
  record it describes (Defect 4). :func:`_connecting_pair_detail` intersected the raw legs'
  free symbols with ``bindings`` -- a collection filtered for a *different* purpose (concrete
  substitution) than what the detail needed to describe (whether an outcome was assumed at
  all). The general question for any future detail string in this package: is it derived
  from the same record its own docstring says it describes, or from some other collection
  that happens to usually agree with that record?

Phase 5 post-closing audit round 19 summary. Round 18's Defect 4 fix routed only the
``outcome`` discriminator through :class:`_ConstraintRecord`; ``assumed``/``equal_to`` and
the "bound to what" clause were still recomputed from final state and from
``free_symbols & bindings.keys()`` respectively -- the identical class of bug, still open in
the same function its own round-18 fix docstring named. Closed by giving
:class:`~qufzx.rewrite.rule.DimensionConstraint` a ``bound_here`` field (the raw
``UnifyResult.bindings`` the recording check actually produced) so every detail string in
this module (:func:`_connecting_pair_detail` and the passing ``phase_dimension_agreement``
"assuming ..." clause, both audited; ``phase_detail``, ``direction_detail``, and
``same_type_detail`` were also checked and derive from a single fresh computation each, not
from a record a later step could disagree with) reads its operands and bindings directly off
a :class:`DimensionConstraint` entry, never recomputes or symbol-matches them. A fifth class,
this round's own meta-lesson: a regression test written at the shape that produced the bug
rather than across the domain the fix's docstring claims to cover. Round 18's own
regression test for Defect 4
(``tests/test_match.py::TestDimensionConstraintsRecording::test_non_concrete_binding_detail_says_something_was_assumed``)
used the one pair shape (no surviving legs at either node) where recomputation from final
state coincidentally agrees with the record -- which is exactly why the recurrence went
undetected for a whole round.
``tests/test_phase5_certificate_sweep.py::TestCertificateDetailFidelity`` is the
sweep, over a six-member concrete/symbolic/product/power palette crossed with consumed and
surviving legs on both sides, that should have existed from the start. Two smaller items
noted but not filed as defects in round 18 were also resolved this round:
:func:`~qufzx.rewrite.rules_library.spider_fusion_builder` no longer reads its own
``side_conditions`` back off its own function-object attribute from inside its own body (a
rename-fragile self-reference; it now reads the module-level ``FUSION_SIDE_CONDITIONS``
constant directly, leaving the attribute solely for :class:`~qufzx.rewrite.rule.Rule`'s
construction-time consistency check), and :func:`_verify_fixpoint_closure`'s call-site
failure message now names the specific convergence path it is unreachable on, rather than
citing the termination guarantee unqualified. Class 2 (a structurally identical reference
kind left unvalidated) was found once more, in :mod:`qufzx.rewrite.engine` rather than this
module: ``consumed_node_ids`` has had a duplicate-entry check since Phase 5's round-12 audit
(A3), but ``new_node_ids`` -- the same shape of field, reported by the same builder about
the same rewrite -- never did, and was not even listed as a deliberately-open row the way
``consumed_wires`` duplicates and unused ``port_mapping`` keys are in that module's own
``BuildResult``-field table. Closed there; see that module's docstring table.

Phase 5 post-closing audit round 20 summary. Three defect classes closed in this module:

* A documented invariant with no structural enforcement.
  :class:`~qufzx.rewrite.rule.DimensionConstraint.bound_here`'s docstring stated its
  empty/non-empty contract against ``outcome`` since round 19, but nothing checked it --
  exactly round 18's Defect 3 class (a guard whose correctness argument is asserted rather
  than structural), recurring in the sibling class right next to the one round 18 already
  fixed the same way (:class:`~qufzx.rewrite.rule.ConstraintSource.__post_init__`). Closed by
  giving ``DimensionConstraint`` its own ``__post_init__``; see that class's docstring.
* A rendered value derived from a different collection than the record it describes -- this
  time the *keys*, not only the values. Round 19's Defect 4 fix made every detail string's
  *values* read off :class:`_ConstraintRecord`; ``phase_dimension_agreement``'s "assuming
  ..." clause still built the *names* it indexed that record with from a second, separately
  threaded accumulator (``_unify_phase_dims``'s returned ``bound_names`` list), agreeing with
  the record only because Phase 5's placeholder ``Dim.unify`` binds at most one symbol per
  call and a phase binding is only ever recorded once fully concrete -- an accident of the
  current unifier's contract, not a property this module's own certificate-fidelity
  discipline should rest on. Closed by deleting the second accumulator entirely
  (``_unify_phase_dims`` now returns ``Dim | None``, not ``tuple[Dim, list[str]] | None``)
  and deriving both the names and their values from ``record.entries()`` alone. The general
  question a future rendered-detail addition to this module should ask, sharpened from round
  19's version: not just "does this value come from the record", but "does *every* collection
  this detail iterates -- keys as well as values -- come from that same record"?
* A validator whose "valid" was weaker than the denotation it is meant to gate. Not a defect
  in this module itself, but :mod:`qufzx.diagram.validate`'s corresponding fix
  (accepting a node with no legs and no phase, which :mod:`qufzx.semantics.denote` correctly
  refuses) is recorded here too because this module's own fixpoint is exactly the kind of
  caller that must not be able to hand :mod:`qufzx.rewrite.engine` a diagram ``validate``
  calls clean but ``denote`` cannot handle -- see :mod:`qufzx.diagram.validate`'s module
  docstring for the fix itself.

Phase 5 post-closing audit round 23 summary. An independent verification round found no
functional defect (699/699 tests, 6,000- and 12,000-seed adversarial sweeps, zero
violations) but five defects of a class this repo's own audit history keeps rediscovering --
named here at the class level, per rounds 18-20's own convention, so round 24 knows where to
keep looking:

* An invariant stated as a proxy for the property that actually matters, not the property
  itself. :class:`_ConstraintRecord`'s docstring claimed a ``BOUND`` entry is "never
  displaced" -- false (:meth:`record` overwrites unconditionally; only
  :meth:`record_identity` is asymmetric) -- but the claim was never actually load-bearing:
  what matters is *adequacy* (the finished record plus finished bindings implies every pair
  ever asserted), which held throughout despite the displacement, for a structural reason
  (only concrete bindings ever propagate between sources). Closed by replacing the false
  invariant with the true one, stated precisely and enforced by
  ``tests/test_phase5_certificate_sweep.py::TestConstraintRecordAdequacy``. The general
  question this sharpens from round 20's "a documented invariant with no structural
  enforcement": before writing down *an* invariant, ask whether it is the property a
  downstream consumer actually depends on, or a proxy for it that happens to usually agree.
* A load-bearing check that exists in the code but not in the certificate. The
  wired-twice-or-boundary-claimed filter lived only as a bare ``continue`` inside
  :func:`find_matches`, invisible to ``FUSION_SIDE_CONDITIONS``, ``side_condition_outcomes``,
  and therefore to any certificate consumer -- and, because :func:`resolve_fusion_match`
  never decided it, the "match-approval and build-applicability are the same predicate by
  construction" claim was false for exactly this condition. Closed by promoting it to
  ``consumed_ports_singly_claimed``, condition 5, decided by ``resolve_fusion_match`` itself.
  The general question: does every candidate-rejection this package performs correspond to a
  named, reported side condition, or does one still live as an anonymous filter a certificate
  has no way to ask about?
* A human-readable detail derived from *which call site* returned a failure, not from *what
  failed*. ``_unify_surviving_legs``, ``_unify_phase_dims``, and ``_unify_connecting_pair``
  each conflated a non-unifying ``Dim`` with a unifying one whose binding contradicts an
  earlier assumption, reporting both with the same generic wording -- round 18's Defect 4
  class, recurring at the failure-path level rather than the passing-detail level it was
  first found at. Closed by :class:`_ResolutionFailure`, a discriminated result each helper
  returns and each call site renders from directly. The general question, extended from
  round 18/19's "does this value come from the record": for a *failure* path, does the
  detail come from a value that names the actual cause, or merely from the shape of the
  branch that produced it?
* A guard credited with a correctness role it does not play. ``_merge_bindings``'s docstring
  called it "D1's soundness fix" and the source of ``bindings``' monotonicity; instrumented
  across 15,000 seeds, its contradiction branch has 0 hits, and reasoning through why
  reveals both claims were wrong -- D1's own example is caught by fixpoint pass repetition,
  and monotonicity falls out of :func:`_resolve_with_bindings`'s pre-resolution discipline.
  The guard is correct and kept (Phase 10's real unifier may make it reachable), but its
  docstring now attributes what it actually does, not what it was once believed to do. The
  general question: when a guard has zero observed hits, is that because it is doing its job
  silently, or because something *else* is doing that job and the guard is measuring a
  different, coincidentally-never-triggered thing?
* An algorithm that is wrong but currently unreachable, found outside this module
  (:func:`qufzx.semantics.contract_numeric._assign_labels`'s four-branch label merge silently
  splits an equivalence class when two already-labelled classes merge, reachable only from a
  wire set ``contract()``'s own callers do not build, but directly callable by a future
  Phase 7 caller that does). Recorded here too, per round 20's precedent for a cross-module
  fix this module's own callers depend on staying correct: "unreachable from every current
  caller" is not the same claim as "correct", and a Phase 7 caller handed a wire set it did
  not validate itself is exactly the shape this repo's "strong foundations first" principle
  exists for.

Phase 5 post-closing audit round 24 summary. A cold review (no prior context, largest
adversarial sweeps this code has seen: 20,000 seeds of deliberately contended-port diagrams,
plus 4,000 chained fusion sequences at 3-5 nodes oracle-checked at every step, on top of the
existing suite) found **no functional defect in the rewrite core**. That is this round's
primary result, and it is reported as a result rather than as an absence of one. What it did
find, per rounds 18-23's own convention of naming the class rather than only the instance:

* A safety lemma that is false, reaching a true conclusion for a different reason -- the same
  class as round 23's ``_merge_bindings`` correction, recurring in
  :mod:`qufzx.repl.parser`, the module round 20 declared closed out. ``_parse_dim`` gated
  ``int(token)`` on ``token.isdigit()`` while the module's own outbound-call audit justified
  it as "the token matches ``\\d+``". Those are different predicates -- ``str.isdigit()``
  additionally admits Unicode category ``No``, on which ``int()`` raises -- so the guard
  leaked a bare ``ValueError`` through that module's ``DiracError`` boundary, the Task 1 class
  round 20 closed everywhere else there. Unreachable through the public entry point, because
  the *regex* gated it; the conclusion held, the stated reason did not. Closed structurally:
  guard and grammar are now built from one shared constant, so they cannot diverge again.
  The general question: when a guard and its documented justification are stated as different
  predicates, which one is actually load-bearing -- and would the other one still hold if the
  caller changed?
* One fact stated in two places and kept in sync by hand -- the class this history keeps
  rediscovering, this time as *numbering*. Round 23 inserted ``consumed_ports_singly_claimed``
  at position 5 and left seven references across three modules stating the pre-insertion
  numbers, one degraded to the meaningless "condition 6/6" and one
  ("condition 5's own convention ... ``leg_deferred``") that no plausible grep would have
  surfaced. Also: ``_merge_bindings``' docstring naming the wrong test module and a return
  convention two rounds stale, ``spider_fusion_builder``'s numbered contract contradicting
  its own call-site comment, and this module's account of *where* each condition is decided
  still describing the pre-round-23 code. Closed by fixing each, and -- since the numbering
  is machine-readable -- by
  ``tests/test_engine.py::TestConditionNumberingMatchesDeclaredOrder``, which pins the
  authoritative list exactly and nets adjacent prose cross-references. It found the seventh
  instance itself, after the other six had been fixed by hand.
* A load-bearing condition with no property-scale coverage. Every diagram generator in
  ``tests/test_fusion_properties.py`` wires ports by popping them out of a pool, so no port
  is ever claimed twice and the boundaries are exactly the leftovers -- meaning round 23's
  headline change, ``consumed_ports_singly_claimed``, could not fail for any candidate the
  harness produced, and was pinned only by hand-picked unit tests. Closed by
  ``_build_contended_diagram`` and its arm, which exercises the condition at 6,663 rejections
  and 1,208 acceptances over 20,000 seeds, cross-checking each verdict against an
  independently re-derived notion of contention. The general question: for each side
  condition, does *some* generator here actually produce a diagram that fails it?

Where this round did not go, and why it matters more than where it did. FULL_PLAN.md's Phase
5 states its own exit criterion -- "the full path Dirac to graph to fuse to graph runs and the
oracle confirms exact equality" -- and that criterion was not satisfiable at all until round
19 added :mod:`qufzx.repl.parser`, because no Dirac front end existed (see that module's own
docstring). Rounds 1-18 audited this phase against everything except its stated
done-when. The criterion has been met since round 19. A reader arriving at round 25 should
weigh a further pass over these four files against Phase 6, where the certificate fields
these rounds refined finally acquire a consumer that can replay and check them -- which is
evidence of a different and stronger kind than another reading of this docstring.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from qufzx.algebra.dimension import Dim, DimSubstituteValue, DimSymbolKey
from qufzx.algebra.phase import PhaseDomainError, PhaseSubstituteValue, PhaseSymbolKey, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, Node, NodeId, PortRef, Wire
from qufzx.rewrite.rule import (
    ConstraintOutcome,
    ConstraintSource,
    ConstraintSourceKind,
    DimensionConstraint,
    Match,
    Pattern,
    RewriteGrammarError,
    SideCondition,
    SideConditionOutcome,
)

FUSION_SIDE_CONDITIONS: tuple[SideCondition, ...] = (
    SideCondition("distinct_nodes", "the two matched nodes are not the same node"),
    SideCondition("same_generator_type", "both nodes are the same registered spider color"),
    SideCondition(
        "parallel_wires_become_self_loops",
        "every other wire joining the two nodes survives as a self-loop on the merged spider",
    ),
    SideCondition(
        "consumed_wire_direction_permitted_for_color",
        "for X, the consumed wire runs OUTPUT to INPUT; for Z, any direction combination "
        "is valid fusion",
    ),
    SideCondition(
        "consumed_ports_singly_claimed",
        "neither consumed port is claimed by a second wire, and neither is listed on "
        "either boundary list",
    ),
    SideCondition(
        "dimension_agreement",
        "the connecting pair and every surviving leg of both nodes unify, in a bounded "
        "fixpoint, against the shared dimension -- equal outright, or unify defers or "
        "binds a symbol",
    ),
    SideCondition(
        "phase_dimension_agreement",
        "every phase vector present unifies with the resolved shared leg dimension -- equal "
        "outright, or unify binds a symbol to a concrete value (never merely defers, and "
        "never binds to another still-symbolic Dim -- see the module docstring's "
        "'Non-concrete bindings' note)",
    ),
)
"""The declared side-condition specs for :class:`FusionPattern`. See the module docstring."""


@dataclass(frozen=True, slots=True)
class FusionMatch:
    """One located fusion occurrence: the two spiders, the consumed wire, and the shared dim.

    ``a_id`` is always the lower :class:`~qufzx.diagram.graph.NodeId` of the pair and
    ``b_id`` the higher -- a fixed, deterministic convention (not a claim about which node
    was "created first") that :mod:`qufzx.rewrite.rules_library` reuses as its merged-leg
    ordering convention ("A's surviving legs, then B's").

    ``bindings`` is the whole-candidate accumulator of every concrete symbol binding
    conditions 6 and 7 (``dimension_agreement``, ``phase_dimension_agreement``) produced
    while resolving ``shared_dim`` -- the connecting pair's own, every surviving leg's on
    either node, and every present phase's own, across every pass of their shared fixpoint
    (see :func:`find_matches`'s local ``bindings`` dict, which this field is built from
    verbatim). :mod:`qufzx.rewrite.rules_library`'s builder substitutes it into a present
    phase's entries, via :func:`reattach_phase`, before reattaching them to ``shared_dim``
    -- see that function's docstring for why this, not a stricter matcher, is the
    resolution of the ``_over_shared_dim`` defect family.
    """

    a_id: NodeId
    b_id: NodeId
    wire: Wire
    shared_dim: Dim
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...] = ()
    bindings: Mapping[str, Dim] = MappingProxyType({})

    def __hash__(self) -> int:
        """Explicit, mirroring :meth:`~qufzx.rewrite.engine.RewriteStep.__hash__` exactly.

        ``@dataclass(frozen=True)`` with the default ``eq=True`` would otherwise generate a
        ``__hash__`` that hashes every field verbatim, including ``bindings`` -- a
        :class:`~types.MappingProxyType`, which is unhashable (its backing ``dict`` is
        mutable even though the proxy itself is read-only). Defining ``__hash__`` here
        explicitly, in the class body, makes ``dataclass`` leave it alone rather than
        overwrite it with the broken auto-generated one. Every other field is hashed as-is
        (``Dim``, ``Wire``, ``NodeId``, and the generated-``__hash__`` ``SideConditionOutcome``
        and its tuple are all already hashable); ``bindings`` is hashed as
        ``frozenset(bindings.items())`` -- order-independent, matching the dataclass-generated
        ``__eq__``, which compares ``bindings`` via plain mapping equality (also
        order-independent) -- so ``a == b`` still implies ``hash(a) == hash(b)``, the same
        contract :class:`~qufzx.rewrite.engine.RewriteStep` needs for Phase 12's cache, which
        embeds a ``FusionMatch`` in its own ``match`` field and therefore needs this to hold
        transitively.

        Same cross-process disclaimer as :meth:`RewriteStep.__hash__` (Phase 5 post-closing
        audit round 18, Defect 1): this contract is a within-process one. ``Wire`` (via
        ``PortRef`` -> ``Direction``) and ``DimensionConstraint`` (via ``ConstraintSourceKind``
        / ``ConstraintOutcome``) are reached transitively here, and every one of those is an
        ``enum.Enum`` hashed by member name -- ``PYTHONHASHSEED``-dependent, so
        ``hash(match)`` legitimately differs across processes for two values this module's own
        ``find_matches`` would otherwise report identically. What round 18 actually fixed is
        the *value and order* of every field this hash is computed over (in particular
        ``dimension_constraints`` and ``side_condition_outcomes``, whose upstream inputs used
        to depend on ``diagram.wires``' own hash-seed-dependent iteration order -- see
        :mod:`qufzx.diagram.validate`'s docstring), not ``hash()`` stability itself, which was
        never a promise this method could make and does not make now.
        """
        return hash(
            (
                self.a_id,
                self.b_id,
                self.wire,
                self.shared_dim,
                self.side_condition_outcomes,
                self.dimension_constraints,
                frozenset(self.bindings.items()),
            )
        )

    @property
    def all_side_conditions_passed(self) -> bool:
        """True iff every recorded side condition passed. See :class:`qufzx.rewrite.rule.Match`."""
        return all(outcome.passed for outcome in self.side_condition_outcomes)


_FUSABLE_GENERATOR_NAMES = frozenset((Z_SPIDER.name, X_SPIDER.name))
_SAME_DIRECTION_FUSABLE_GENERATOR_NAMES = frozenset((Z_SPIDER.name,))
"""Generator names for which a same-direction (OUTPUT-OUTPUT or INPUT-INPUT) connecting
wire is still valid fusion -- see condition 4
(``consumed_wire_direction_permitted_for_color``) in the module docstring. Z only: X's
Fourier-conjugate structure makes a same-direction wire a
different, unimplemented rule, not fusion."""

_MAX_FIXPOINT_PASSES = 32
"""Iteration budget for :func:`resolve_fusion_match`'s joint condition-5/6 fixpoint.

Module-level, not a function local, so a test can patch it low and actually exercise the
exhaustion path (``tests/test_match.py::TestFixpointBudgetExhaustion``).

Unreachability argument (D1's fix restates this in terms of ``bindings``, not
``shared_dim``'s own concretization -- the pre-fix version of this docstring argued from
the latter, which is exactly the false converse D1 was): ``bindings`` is monotone -- a key
is only ever added, never rebound to a different value. Round 23 correction: this is not
because :func:`_merge_bindings` rejects a contradictory rebind (instrumented across 15,000
seeds, that guard has 0 hits -- see its own docstring); it is because every operand this
fixpoint unifies is first resolved through :func:`_resolve_with_bindings`, which substitutes
every symbol ``bindings`` already names, so an already-bound symbol is never free in what
reaches ``Dim.unify`` and therefore never reappears as a fresh binding key. ``_merge_bindings``
is kept as a structural guard for that same invariant, not as the mechanism that produces it
-- and every key ``bindings`` can ever hold is drawn from the finite set of free symbols
appearing in
node_a's legs and phase, node_b's legs and phase, and the connecting pair (equivalently,
``shared_dim``'s own lineage, which starts at ``port_a.dim`` and is only ever refined by
substituting members of that same finite set). A pass that does not stabilise therefore adds
at least one fresh key to ``bindings``, so the number of non-stabilising passes is bounded by
that finite symbol count -- unreachable for any diagram with fewer distinct dimension symbols
in play than the cap. The guard is kept anyway, conservatively refusing the candidate rather
than looping forever, because that bound rests on :meth:`~qufzx.algebra.dimension.Dim.unify`'s
current placeholder contract, which Phase 10 replaces."""


def _resolve_with_bindings(dim: Dim, bindings: Mapping[str, Dim]) -> Dim:
    """Substitute ``bindings`` into ``dim``, or return it unchanged if none apply.

    ``bindings`` is empty both when :meth:`Dim.unify` deferred and when it succeeded via a
    bare syntactic identity with nothing bound -- in both cases this is the identity
    function, which is exactly the "keep the raw Dim unchanged" behavior condition 6 in
    the module docstring calls for. A bare symbol can also unify by binding to *another*
    still-symbolic ``Dim`` (e.g. ``d`` against ``e`` binds ``d := e``); :meth:`Dim.substitute`
    only ever accepts a concrete replacement value, so such a binding is dropped here rather
    than resolving through it -- ``dim`` stays raw and unchanged for that symbol, the same
    treatment a deferred pair gets, rather than crashing on a substitution ``Dim`` was never
    built to perform.
    """
    concrete_bindings = {name: value for name, value in bindings.items() if value.is_concrete}
    if not concrete_bindings:
        return dim
    return dim.substitute(cast(Mapping[DimSymbolKey, DimSubstituteValue], concrete_bindings))


class _FailureReason(enum.Enum):
    """Why one of the fixpoint's unify helpers (:func:`_unify_surviving_legs`,
    :func:`_unify_phase_dims`, :func:`_unify_connecting_pair`) failed this pass.

    Phase 5 post-closing audit round 23, Task 3: introduced because a failure's detail
    string used to be derived from *which call site* returned ``None``/``(None, ...)``, not
    from *what actually went wrong* -- a leg/pair that never unifies at all (``UNIFY_FAILURE``)
    and one that unifies fine but whose binding contradicts an earlier one
    (``CONTRADICTORY_REBIND``) were reported with the identical generic wording, even though
    they are different failures with different remedies. ``PHASE_DEFERRED`` and
    ``PHASE_NON_CONCRETE_BINDING`` are :func:`_unify_phase_dims`-only: a leg or the connecting
    pair tolerates a deferral or a non-concrete binding (see the module docstring's
    "Non-concrete bindings" section), a phase does not.
    """

    UNIFY_FAILURE = "unify_failure"
    """:meth:`~qufzx.algebra.dimension.Dim.unify` returned ``FAILURE`` outright."""

    CONTRADICTORY_REBIND = "contradictory_rebind"
    """The unify succeeded, but :func:`_merge_bindings` rejected its binding as a
    contradictory rebind of a symbol already bound to a different concrete value."""

    PHASE_DEFERRED = "phase_deferred"
    """A phase's own dimension unify ``DEFERRED`` against the shared leg dimension -- tolerated
    for a leg or the connecting pair, but not for a phase (module docstring, condition 7)."""

    PHASE_NON_CONCRETE_BINDING = "phase_non_concrete_binding"
    """A phase's own dimension unify succeeded only by binding to a non-concrete ``Dim`` --
    tolerated for a leg (left unused for shared-dimension resolution) but not for a phase,
    since a phase's own entries can reference its ``dim``'s free symbols directly (module
    docstring, "Non-concrete bindings")."""


@dataclass(frozen=True, slots=True)
class _ResolutionFailure:
    """One unify helper's failure this pass: why, and the two operands actually involved.

    ``assumed``/``equal_to`` are the operands *as checked this pass* (already resolved
    through the running ``bindings`` accumulator where applicable) -- the same pair a
    successful check would have recorded as a :class:`~qufzx.rewrite.rule.DimensionConstraint`
    had it not failed. A call site renders its detail string from these fields and ``reason``
    directly, never by re-deriving "what failed" from which branch of its own code returned
    this value (the same discipline :func:`_connecting_pair_detail` already applies to a
    *passing* check's own detail -- see that function's docstring).
    """

    reason: _FailureReason
    assumed: Dim
    equal_to: Dim


def _merge_bindings(
    bindings: dict[str, Dim], new_bindings: Mapping[str, Dim]
) -> tuple[str, Dim, Dim] | None:
    """Merge the concrete entries of ``new_bindings`` into ``bindings``, in place.

    Only concrete-valued bindings are ever stored (see :func:`_resolve_with_bindings` for why
    a binding to a non-concrete ``Dim`` is dropped rather than resolved through). Returns
    ``(name, existing_value, new_value)``, leaving ``bindings`` completely unmodified, iff
    some name already bound to a different concrete ``Dim`` would be rebound to a new one --
    a contradictory assumption (e.g. two surviving legs forcing the same symbol to two
    different concrete values); returns ``None`` on a clean merge. The conflict tuple (Phase
    5 post-closing audit round 23, Task 3) is what lets a call site's failure detail name the
    actual symbol and both values a contradictory rebind involved, rather than folding this
    failure into the same generic message a plain ``Dim.unify`` ``FAILURE`` gets -- the two
    are genuinely different failures (a leg/pair that does not unify at all, versus one that
    unifies fine but whose binding conflicts with an earlier one), see
    :class:`_ResolutionFailure`. A non-``None`` return makes the whole candidate a non-match
    at the call site, exactly like a ``FAILURE`` from ``Dim.unify`` itself.

    Round 23 correction of two prior misattributions in this docstring. First: this guard is
    *not* what catches D1's original unsatisfiable-constraint-set example (``e*f == 2 and
    e == 2 and f == 2``, from legs ``[e*f, e, f]``) -- that is caught by
    :func:`resolve_fusion_match`'s fixpoint requiring a full pass to add nothing to either
    ``shared_dim`` or ``bindings`` before exiting (see that function's own inline commentary),
    which forces a further pass to re-resolve ``e*f`` through the now-bound ``e``/``f`` and
    re-unify it, surfacing the contradiction as an ordinary ``Dim.unify`` ``FAILURE`` -- not
    as a rebind this function rejects. Second: this function is not what makes ``bindings``
    monotone either -- monotonicity falls out of :func:`_resolve_with_bindings` substituting
    every already-known binding into an operand *before* it reaches ``Dim.unify``, which
    means an already-bound symbol is never free in what gets unified and so can never
    reappear as a fresh ``UnifyResult.bindings`` key for this function to see rebound.

    Consequence, confirmed by instrumenting every call across the same 15,000-seed sweep
    that found the round-23 ``_ConstraintRecord`` displacement (see that class's docstring):
    this guard's own contradiction branch has 0 hits -- it does not currently fire, for the
    structural reason just given, not by coincidence. It is kept anyway: it is the correct
    guard for the case where a caller updates ``bindings`` with something that was *not*
    first passed through ``_resolve_with_bindings`` (not true of any current call site, but
    also not an invariant this function itself can see or enforce), and :meth:`Dim.unify`'s
    contract is a Phase 5 placeholder Phase 10's real unifier replaces -- at which point the
    pre-resolution argument above may no longer hold and this guard becomes reachable.
    ``tests/test_fusion_properties.py::TestSpiderFusionProperties
    ::test_random_diagrams_fuse_soundly`` wraps this function to assert it returns ``None``
    -- a clean merge -- on every call across its sweeps, so a Phase 10 change that makes it
    reachable is announced by a newly-failing assertion rather than discovered by accident.
    (Round 24: this sentence named the wrong test module and the wrong return convention. It
    predates Task 3's change of this function's return type from a bare ``bool`` to
    ``tuple[str, Dim, Dim] | None``, and survived the round-23 rewrite of the paragraphs
    above it; the test itself already carried a comment correcting the convention, which is
    exactly the "two statements of one fact kept in sync by hand" shape the record-keyed
    constraint discipline elsewhere in this module exists to avoid.)
    """
    concrete = {name: value for name, value in new_bindings.items() if value.is_concrete}
    for name, value in concrete.items():
        existing = bindings.get(name)
        if existing is not None and existing != value:
            return (name, existing, value)
    bindings.update(concrete)
    return None


class _ConstraintRecord:
    """The source-keyed, insertion-ordered record of one candidate's dimension assumptions.

    One entry per :class:`~qufzx.rewrite.rule.ConstraintSource`, never one per check: the
    leg/phase fixpoint re-checks the same source once per pass, and each re-check
    :meth:`record`\\ s over the previous entry *in place* (a ``dict`` assignment to an
    existing key keeps that key's original position), so the finished sequence is in
    first-derivation order.

    Phase 5 post-closing audit round 23, Task 1: a prior version of this docstring claimed a
    ``BOUND`` entry is "pinned at the pass that made the binding and never displaced by a
    later pass's bare-identity re-check of the same source". That claim is false: only
    :meth:`record_identity` honours any such asymmetry; :meth:`record` itself overwrites its
    key unconditionally, for *every* outcome, including a later pass re-deriving a
    previously-``BOUND`` source as ``DEFERRED``. Instrumented across 15,000 generator seeds
    (``tests/test_fusion_properties.py``'s three generators), this displaces a ``BOUND``
    entry 43 times: 16 ``BOUND`` -> ``DEFERRED``, 27 ``BOUND`` -> a different ``BOUND``. The
    invariant that actually needs to hold is not "a ``BOUND`` entry is never displaced" --
    that was a proxy for the property that matters and does not, itself, matter -- it is:

        **Adequacy.** The conjunction of the *finished* ``dimension_constraints`` (this
        record's entries, once resolution completes) together with the *finished*
        ``bindings`` must imply every ``(assumed, equal_to)`` pair any check in the whole
        fixpoint ever asserted -- including one a later pass's re-derivation subsequently
        replaced or dropped.

    This holds -- proven mechanically by
    ``tests/test_phase5_certificate_sweep.py::TestConstraintRecordAdequacy`` over a
    concrete-palette-crossed substitution space plus the three random generators, not merely
    argued here -- because of one structural fact about how ``bindings`` is built: only a
    *concrete* binding ever enters the shared ``bindings`` accumulator (see
    :func:`_merge_bindings`; a binding to a non-concrete ``Dim`` is dropped, per the module
    docstring's "Non-concrete bindings" section). No source's ``assumed``/``equal_to`` is
    therefore ever built by resolving through another source's *non-concrete* binding --
    only through concrete ones, which are never later contradicted (:func:`_merge_bindings`
    rejects a contradictory rebind) and which are themselves always separately recorded under
    their own source. So a displaced entry can only ever become implied transitively through
    a concrete binding that is itself still on the finished record (or itself the finished
    ``bindings`` mapping directly) -- never through the very non-concrete binding that got
    displaced, since that binding was never propagated to begin with. This also answers
    whether a displacement can break adequacy transitively (source A's binding is displaced,
    and source B's entry was stated in terms of it): it cannot, for exactly this reason -- B
    can only ever have been stated in terms of a *concrete* value A contributed to
    ``bindings``, never in terms of A's own discarded non-concrete ``assumed``/``equal_to``.

    The full policy, all nine (previous entry, this check's outcome) cells:

    * (none recorded yet, ``DEFERRED``): record it -- the first assumption from this source.
    * (none recorded yet, ``BOUND``): record it -- same reasoning.
    * (none recorded yet, bare identity via :meth:`record_identity`): no-op, no entry created
      -- nothing was ever assumed, so nothing belongs on the certificate for this source.
    * (``DEFERRED``, ``DEFERRED``): overwrite with the new ``DEFERRED`` entry -- still an
      open assumption, restated at its currently-resolved operands; the same shape, so this
      is harmless even though it is, technically, a displacement.
    * (``DEFERRED``, ``BOUND``): overwrite with ``BOUND`` -- the deferral has been discharged
      into a decided-under-binding fact, strictly more informative than what it replaces.
    * (``DEFERRED``, bare identity): drop the entry (:meth:`record_identity`) -- the deferral
      has been discharged into a bare syntactic identity; nothing is assumed any more.
    * (``BOUND``, ``DEFERRED``): overwrite with ``DEFERRED`` -- the disputed cell above. Not
      a bug: adequacy holds transitively (see above), and the record's job is to hold each
      source's single most-resolved current statement, not a history of what it once was.
    * (``BOUND``, ``BOUND``): overwrite with the new ``BOUND`` entry -- possibly the same
      fact restated, possibly a refined ``bound_here``; same-source dedup applies uniformly.
    * (``BOUND``, bare identity): **keep** the existing ``BOUND`` entry, via
      :meth:`record_identity`'s own asymmetric handling -- do not drop it. The binding *is*
      the assumption: a later bare identity holds only because that binding was made, and
      dropping the entry would erase the very assumption that makes the identity true. This
      is the one cell where :meth:`record_identity` deliberately does not mirror
      :meth:`record`'s "always overwrite" rule.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[ConstraintSource, DimensionConstraint] = {}

    def record(
        self,
        source: ConstraintSource,
        assumed: Dim,
        equal_to: Dim,
        outcome: ConstraintOutcome,
        bound_here: Mapping[str, Dim] | None = None,
    ) -> None:
        """Record (or re-record, in place) ``source``'s assumed equality.

        ``bound_here`` is the raw ``UnifyResult.bindings`` this specific check produced --
        required (non-empty) when ``outcome`` is ``BOUND``, omitted (empty) otherwise. See
        :attr:`~qufzx.rewrite.rule.DimensionConstraint.bound_here`.
        """
        self._entries[source] = DimensionConstraint(
            assumed=assumed,
            equal_to=equal_to,
            source=source,
            outcome=outcome,
            bound_here=tuple(sorted((bound_here or {}).items())),
        )

    def record_identity(self, source: ConstraintSource) -> None:
        """Note that ``source`` re-checked as a bare identity. See the class docstring."""
        existing = self._entries.get(source)
        if existing is not None and existing.outcome is ConstraintOutcome.DEFERRED:
            del self._entries[source]

    def entries(self) -> tuple[DimensionConstraint, ...]:
        """Every recorded constraint, in first-derivation order."""
        return tuple(self._entries.values())

    def entry_for(self, source: ConstraintSource) -> DimensionConstraint | None:
        """The current entry for ``source``, or ``None`` if never recorded, or since discharged."""
        return self._entries.get(source)

    def leg_count(self) -> int:
        """How many entries came from a surviving leg (not the connecting pair or a phase)."""
        return sum(
            1
            for entry in self._entries.values()
            if entry.source.kind is ConstraintSourceKind.SURVIVING_LEG
        )

    def any_leg_deferred(self) -> bool:
        """True iff a connecting-pair or surviving-leg entry is, finally, a ``DEFERRED`` one.

        Computed from the finished record rather than from a flag accumulated across passes,
        so a leg that deferred on one pass and bound (or resolved to an identity) on a later
        one does not leave a stale ``deferred=True`` on ``dimension_agreement``.
        """
        return any(
            entry.deferred
            and entry.source.kind
            in (ConstraintSourceKind.CONNECTING_PAIR, ConstraintSourceKind.SURVIVING_LEG)
            for entry in self._entries.values()
        )


def _unify_surviving_legs(
    node: Node,
    node_id: NodeId,
    consumed_ref: PortRef,
    shared_dim: Dim,
    bindings: dict[str, Dim],
    record: _ConstraintRecord,
) -> Dim | _ResolutionFailure:
    """Unify every surviving leg of ``node`` (both directions) against ``shared_dim`` in turn.

    "Surviving" means every leg of ``node`` except ``consumed_ref``, checked in
    input-then-output, original-index order for this one node -- the same per-node order
    :mod:`qufzx.rewrite.rules_library`'s ``_surviving_legs`` uses, though the two callers
    (this function for A, then again for B) do not combine into the same global order the
    builder assembles the merged node's ports in; see the module docstring, condition 6.

    ``bindings`` is the running, whole-candidate accumulator of every concrete symbol
    binding seen so far (starting with the connecting pair's own) -- this function updates
    it in place with each surviving leg's own concrete binding, on top of using it (via
    :func:`_resolve_with_bindings`) to refine ``shared_dim`` as it goes. Each leg's own
    ``Dim`` is resolved through that accumulator *before* being unified against
    ``shared_dim``, never in its raw form, exactly as :func:`_unify_phase_dims` does for a
    phase.

    Returns the (possibly refined) shared dimension, or a :class:`_ResolutionFailure` if any
    surviving leg's resolved dim is non-unifiable with it, or unifies but only via a binding
    that contradicts an earlier one -- either makes the whole candidate a non-match, exactly
    like a ``FAILURE`` on the connecting-leg pair itself, since forcing that leg onto
    ``shared_dim`` anyway would destroy a real dimension conflict rather than report it.
    Every leg's outcome is written into ``record`` under its own
    :meth:`~qufzx.rewrite.rule.ConstraintSource.surviving_leg` key -- deferred, bound, or
    (via :meth:`_ConstraintRecord.record_identity`) a bare identity.
    """
    for direction in (Direction.INPUT, Direction.OUTPUT):
        for index, port in enumerate(node.legs(direction)):
            ref = PortRef(node_id, direction, index)
            if ref == consumed_ref:
                continue
            source = ConstraintSource.surviving_leg(ref)
            leg_dim = _resolve_with_bindings(port.dim, bindings)
            result = leg_dim.unify(shared_dim)
            if result.is_failure:
                return _ResolutionFailure(_FailureReason.UNIFY_FAILURE, leg_dim, shared_dim)
            bound_this_pass = result.is_success and bool(result.bindings)
            if result.is_deferred:
                record.record(source, leg_dim, shared_dim, ConstraintOutcome.DEFERRED)
            elif bound_this_pass:
                record.record(
                    source, leg_dim, shared_dim, ConstraintOutcome.BOUND,
                    bound_here=result.bindings,
                )
            else:
                record.record_identity(source)
            if bound_this_pass:
                conflict = _merge_bindings(bindings, result.bindings)
                if conflict is not None:
                    return _ResolutionFailure(
                        _FailureReason.CONTRADICTORY_REBIND, leg_dim, shared_dim
                    )
                shared_dim = _resolve_with_bindings(shared_dim, result.bindings)
    return shared_dim


def _unify_phase_dims(
    node_a: Node,
    node_b: Node,
    a_id: NodeId,
    b_id: NodeId,
    shared_dim: Dim,
    bindings: dict[str, Dim],
    record: _ConstraintRecord,
) -> Dim | _ResolutionFailure:
    """Unify every phase vector actually present (A's, then B's) against ``shared_dim``, in turn.

    Mirrors :func:`_unify_surviving_legs`'s accumulator discipline exactly, extended to
    phases (module docstring, condition 7): each phase's own ``Dim`` is first resolved
    through the running ``bindings`` accumulator (a phase whose symbol a *later* leg or an
    earlier phase in this same call already bound concretely must be checked against its
    resolved value, never its stale raw one), then unified against the *current*
    ``shared_dim``; a concrete binding refines both ``bindings`` and ``shared_dim`` in place
    before the next phase (if any) is examined. Two present phases therefore cannot each
    bind the same symbol against the same stale ``shared_dim``, and a binding a phase alone
    produces is folded into ``shared_dim`` immediately rather than held back as though a
    phase's binding could never matter to what a leg shares.

    Unlike a leg, a genuinely ``DEFERRED`` result, or a result whose binding is not
    concrete, is never accepted here -- see the module docstring, condition 7, for why a
    phase's own entries make reattaching under an unproven or non-concrete assumption
    unsafe. Returns a :class:`_ResolutionFailure` on any of those (or a contradictory
    rebind), making the whole candidate a non-match -- its ``equal_to`` is the ``shared_dim``
    value actually checked against the failing phase, not necessarily the caller's pre-call
    ``shared_dim``. Returns the (possibly refined) ``shared_dim`` on success, having written
    each present phase's binding-only success into ``record`` under its own
    :meth:`~qufzx.rewrite.rule.ConstraintSource.node_phase` key.

    Round 20, Task 7: previously returned ``(resolved_shared_dim, bound_names)``, threading a
    second, parallel accumulator of bound symbol names out to the caller alongside ``record``
    itself. ``resolve_fusion_match`` then read the *values* for those names back out of
    ``record.entries()`` while reading the *names* from this separately-threaded list --
    two different collections that happened to agree only because ``Dim.unify`` (Phase 5's
    placeholder) binds at most one symbol per call and a phase binding is always concrete, so
    ``phase_bound_names`` could never name anything ``record`` had not *also* just bound. See
    the module docstring's "Round 20" section for why that agreement is not something a
    future ``Dim.unify`` (Phase 10's real unifier) can be trusted to preserve, and why
    ``record`` is now the *only* source the caller reads for both.
    """
    for node_id, phase in ((a_id, node_a.phase), (b_id, node_b.phase)):
        if phase is None:
            continue
        source = ConstraintSource.node_phase(node_id)
        phase_dim = _resolve_with_bindings(phase.dim, bindings)
        phase_unify = phase_dim.unify(shared_dim)
        if phase_unify.is_failure:
            return _ResolutionFailure(_FailureReason.UNIFY_FAILURE, phase_dim, shared_dim)
        if phase_unify.is_deferred:
            return _ResolutionFailure(_FailureReason.PHASE_DEFERRED, phase_dim, shared_dim)
        if not phase_unify.bindings:
            record.record_identity(source)
            continue
        if not all(value.is_concrete for value in phase_unify.bindings.values()):
            return _ResolutionFailure(
                _FailureReason.PHASE_NON_CONCRETE_BINDING, phase_dim, shared_dim
            )
        record.record(
            source, phase_dim, shared_dim, ConstraintOutcome.BOUND,
            bound_here=phase_unify.bindings,
        )
        new_concrete = dict(phase_unify.bindings)
        conflict = _merge_bindings(bindings, new_concrete)
        if conflict is not None:
            return _ResolutionFailure(
                _FailureReason.CONTRADICTORY_REBIND, phase_dim, shared_dim
            )
        shared_dim = _resolve_with_bindings(shared_dim, phase_unify.bindings)
    return shared_dim


def _unify_connecting_pair(
    port_a_dim: Dim,
    port_b_dim: Dim,
    shared_dim: Dim,
    bindings: dict[str, Dim],
    record: _ConstraintRecord,
) -> Dim | _ResolutionFailure:
    """Re-derive the connecting pair's own equality, at its most-resolved form, this pass.

    Unlike every ``SURVIVING_LEG`` and ``NODE_PHASE`` check, the connecting pair relates its
    own two legs to *each other*, not to ``shared_dim`` (see
    :attr:`~qufzx.rewrite.rule.ConstraintSourceKind.CONNECTING_PAIR`): it is what seeds
    ``shared_dim`` in the first place, on the fixpoint's first pass, when both ports are
    still resolved through an empty ``bindings`` accumulator (so ``resolved_a`` there is
    ``port_a_dim`` itself, unify-ing trivially against ``shared_dim``, which
    :func:`resolve_fusion_match` also seeds at ``port_a_dim`` -- and ``resolved_b`` reduces
    to exactly the pre-fixpoint ``port_a_dim.unify(port_b_dim)`` this replaces).

    D2's fix: called once per pass, not once before the loop, so a later pass sees whatever
    a leg or phase check elsewhere in this same fixpoint has since bound -- the same
    treatment every other source already had. Both legs are resolved through the running
    ``bindings`` first, then unified against each other; a ``FAILURE`` here is a non-match,
    exactly like a leg's own. Returns the (possibly refined) shared dimension, or a
    :class:`_ResolutionFailure` on ``FAILURE`` or a contradictory rebind
    (:func:`_merge_bindings`).

    Unlike :func:`_unify_surviving_legs` and :func:`_unify_phase_dims`, *both* of this
    function's unify operands (``resolved_a``, ``resolved_b``) are pre-resolved through
    ``bindings`` before ``Dim.unify`` ever sees them -- there is no second, raw operand
    (those two functions each unify their own pre-resolved value against the caller's
    ``shared_dim`` parameter directly, unresolved). A ``CONTRADICTORY_REBIND`` here is
    therefore not merely unreached in practice (like the other two, and like
    :func:`_merge_bindings` itself) but unconstructible by any input to this function at
    all: any symbol free in ``resolved_a``/``resolved_b`` is, by definition, not already a
    key in ``bindings`` -- if it were, resolution would have already replaced it with its
    bound value, leaving nothing free for ``Dim.unify`` to (re)bind.
    """
    resolved_a = _resolve_with_bindings(port_a_dim, bindings)
    resolved_b = _resolve_with_bindings(port_b_dim, bindings)
    result = resolved_a.unify(resolved_b)
    if result.is_failure:
        return _ResolutionFailure(_FailureReason.UNIFY_FAILURE, resolved_a, resolved_b)
    source = ConstraintSource.connecting_pair()
    bound_this_pass = result.is_success and bool(result.bindings)
    if result.is_deferred:
        record.record(source, resolved_a, resolved_b, ConstraintOutcome.DEFERRED)
    elif bound_this_pass:
        record.record(
            source, resolved_a, resolved_b, ConstraintOutcome.BOUND,
            bound_here=result.bindings,
        )
    else:
        record.record_identity(source)
    if bound_this_pass:
        conflict = _merge_bindings(bindings, result.bindings)
        if conflict is not None:
            return _ResolutionFailure(
                _FailureReason.CONTRADICTORY_REBIND, resolved_a, resolved_b
            )
        shared_dim = _resolve_with_bindings(shared_dim, result.bindings)
    return shared_dim


def _verify_fixpoint_closure(
    node_a: Node,
    node_b: Node,
    a_id: NodeId,
    b_id: NodeId,
    ref_a: PortRef,
    ref_b: PortRef,
    port_a_dim: Dim,
    port_b_dim: Dim,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
) -> bool:
    """Re-verify, from scratch, that the finished fixpoint's own claim actually holds.

    The fixpoint's own definition made explicit, as a structural guarantee rather than a
    property only tests assert (D1's required post-loop check): the connecting pair's two
    legs, every surviving leg of both nodes, and every present phase -- each resolved under
    the *final* ``bindings`` -- must unify with the *final* ``shared_dim`` without
    ``FAILURE``. Given the fixpoint only exits *via its own convergence break* (a full pass
    that adds nothing to either ``shared_dim`` or ``bindings`` -- D1's fixed termination
    signal) and every ``bindings`` update is contradiction-checked (:func:`_merge_bindings`),
    this is unreachable on that path: the loop's last pass already re-checked every one of
    these against the state this function re-checks them against again. It is called anyway,
    unconditionally on that path, rather than left to be unreachable in principle only.

    Where this now runs (Phase 5 post-closing audit round 18, Defect 3 -- corrected from a
    prior version of this docstring, which claimed unreachability unconditionally):
    :func:`resolve_fusion_match` calls this function *only* on the stabilised-convergence
    path just described. A phase-dim ``FAILURE`` inside the fixpoint loop, and the
    ``_MAX_FIXPOINT_PASSES`` budget-exhaustion path, both ``return`` directly from within
    the loop -- neither one reaches this call at all any more. Before this fix, every exit
    from the loop (a genuine convergence break, *and* a phase-dim ``FAILURE`` that ``break``
    out of the loop with ``phase_dims_agree = False``) fell through to this same
    unconditional call; for the ``FAILURE`` case, that re-verified phases under the exact
    unify contract that had just failed them, so this function necessarily returned
    ``False`` -- which was not evidence of some new problem, but this function faithfully
    reporting the failure that was already known and had already been given its own
    dedicated report. Reaching it from that path was the defect (see
    :func:`resolve_fusion_match`'s inline commentary at its phase-failure return for the
    full account), not anything wrong with this function itself, whose own re-verification
    logic is unchanged. Its unreachability claim is now proven, not merely asserted, by
    ``tests/test_phase5_exhaustive_oracle.py`` and
    ``tests/test_fusion_properties.py::TestSpiderFusionProperties
    ::test_random_diagrams_fuse_soundly``, both of which wrap this function to assert it
    returns ``True`` on every call across their respective sweeps.
    """
    for dim in (port_a_dim, port_b_dim):
        if _resolve_with_bindings(dim, bindings).unify(shared_dim).is_failure:
            return False
    for node, node_id, consumed_ref in ((node_a, a_id, ref_a), (node_b, b_id, ref_b)):
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index, port in enumerate(node.legs(direction)):
                ref = PortRef(node_id, direction, index)
                if ref == consumed_ref:
                    continue
                if _resolve_with_bindings(port.dim, bindings).unify(shared_dim).is_failure:
                    return False
    for phase in (node_a.phase, node_b.phase):
        if phase is None:
            continue
        if _resolve_with_bindings(phase.dim, bindings).unify(shared_dim).is_failure:
            return False
    return True


def _connecting_pair_detail(
    port_a_dim: Dim, port_b_dim: Dim, bindings: Mapping[str, Dim], record: _ConstraintRecord
) -> str:
    """Human-readable summary of the connecting pair's *finished* record entry.

    Built from the finished record, not the fixpoint's first pass alone (D2's knock-on fix):
    since the connecting pair is now re-derived every pass, whether it ended up deferred,
    bound, or (once a later pass's binding discharges it) a bare identity is only known once
    the fixpoint itself has finished -- reading it off the first pass alone, as the pre-fix
    code did, could contradict the finished ``dimension_constraints`` it sits beside in the
    certificate.

    Every operand and binding rendered below is read directly off ``entry`` -- the same
    :class:`~qufzx.rewrite.rule.DimensionConstraint` ``dimension_constraints`` itself is
    built from -- never recomputed against the final ``port_a_dim``/``port_b_dim``/
    ``bindings`` state (Phase 5 post-closing audit round 19, Defect 4 continued). Recomputing
    was the round-18 fix's own residual bug: this pass's ``resolved_a == resolved_b`` need
    not equal ``entry.assumed == entry.equal_to`` (a surviving leg elsewhere can concretize
    both sides *after* the connecting pair's own check ran), and a symbol-occurrence
    intersection against ``bindings`` can attribute a binding some other check made to this
    entry, or drop a real one that never reached ``bindings`` because it was non-concrete.
    ``port_a_dim``/``port_b_dim``/``bindings`` are used for exactly one case: no entry was
    ever recorded at all, which only happens when the pair was a bare identity on every pass
    -- there ``resolved_a`` and ``resolved_b`` are trivially equal, so there is no second
    source of truth to diverge from.
    """
    entry = record.entry_for(ConstraintSource.connecting_pair())
    if entry is None:
        resolved_a = _resolve_with_bindings(port_a_dim, bindings)
        resolved_b = _resolve_with_bindings(port_b_dim, bindings)
        return f"{resolved_a} == {resolved_b}"
    if entry.outcome is ConstraintOutcome.DEFERRED:
        return f"{entry.assumed} == {entry.equal_to} (deferred, assumed)"
    # entry.outcome is BOUND: render exactly what this check's own unify bound
    # (entry.bound_here), not a value looked up by symbol coincidence.
    #
    # Round 20, Task 6: entry.bound_here being non-empty here is no longer merely assumed --
    # DimensionConstraint.__post_init__ now enforces, structurally, that a BOUND outcome
    # always carries a non-empty bound_here (see that class's docstring). This is exactly the
    # class round 18's Defect 3 named: a guard whose correctness argument used to rest on
    # every caller happening to get it right is now backed by a constructor that rejects the
    # violating case outright. The `entry.bound_here and` half of the condition below is
    # therefore dead in the sense that it can no longer be False for a BOUND entry -- kept
    # anyway as a direct, readable mirror of the invariant it relies on, rather than deleted
    # and replaced with an assert (which would just relocate the same trust one line up
    # without making it any more visible). The fall-through branch's claim -- "bound to a
    # non-concrete Dim" -- is therefore now provably reachable only for a genuine
    # non-concrete binding, not for an unrelated bug that happened to leave bound_here empty.
    if entry.bound_here and all(value.is_concrete for _, value in entry.bound_here):
        binding_desc = ", ".join(f"{name} := {value}" for name, value in entry.bound_here)
        return f"{entry.assumed} == {entry.equal_to} (bound: {binding_desc})"
    return (
        f"{entry.assumed} == {entry.equal_to} (bound to a non-concrete Dim; left unused for "
        "shared-dimension resolution, see the module docstring's 'Non-concrete bindings')"
    )


def _dimension_agreement_outcome(
    port_a_dim: Dim,
    port_b_dim: Dim,
    shared_dim: Dim,
    bindings: Mapping[str, Dim],
    record: _ConstraintRecord,
) -> SideConditionOutcome:
    """Build condition 6's (``dimension_agreement``) passing outcome from a leg-sweep state.

    Shared by :func:`resolve_fusion_match`'s stabilised-success path and its phase-failure
    path (Defect 3, Phase 5 post-closing audit round 18): both must report condition 6 from
    the *leg* sweep's own state -- ``shared_dim``/``bindings`` exactly as the connecting
    pair and every surviving leg were actually checked against -- never from a state a later
    phase check has since advanced past what the legs saw. See ``resolve_fusion_match``'s
    own inline commentary at its call sites for why the two states can differ within one
    fixpoint pass.
    """
    leg_detail = _connecting_pair_detail(port_a_dim, port_b_dim, bindings, record)
    leg_constraint_count = record.leg_count()
    return SideConditionOutcome(
        "dimension_agreement",
        True,
        leg_detail
        + (
            ""
            if not leg_constraint_count
            else (
                f"; surviving leg(s) resolved to shared_dim={shared_dim} with "
                f"{leg_constraint_count} additional assumed dimension equality/ies"
            )
        ),
        deferred=record.any_leg_deferred(),
    )


def reattach_phase(
    phase: PhaseVector, shared_dim: Dim, bindings: Mapping[str, Dim]
) -> tuple[PhaseVector, Mapping[str, Dim]]:
    """Substitute ``bindings`` into ``phase``'s entries, then reattach the result to ``shared_dim``.

    Returns the reattached vector together with the subset of ``bindings`` that was
    actually substituted into an entry's *value* -- distinct from ``shared_dim``, which
    every caller reattaches to regardless of whether any entry mentioned it.

    Public (not ``_reattach_phase``, its pre-round-12 name): both this module's own
    :func:`resolve_fusion_match` and :mod:`qufzx.rewrite.rules_library`'s
    :func:`~qufzx.rewrite.rules_library._over_shared_dim` treat it as the shared
    match-approval / build-applicability contract their own docstrings describe -- that
    makes it public API in all but name, so a leading underscore claiming otherwise is
    corrected here rather than left as a private name two modules quietly depend on.

    This is the one, shared resolution -- used identically here (as a trial construction
    ``dimension_agreement`` performs to decide whether a candidate is a match at all) and by
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` (to actually build the merged
    node's phase) -- of the ``_over_shared_dim`` defect family documented at length in that
    module's docstring, "Dimension of the merged node": a phase legally stated over a
    symbolic dimension (e.g. ``PhaseVector(d, {1: Phase.root_of_unity(1, d)})``) that
    condition 6 resolves ``shared_dim`` past via a binding (e.g. ``d := 2``) must not be
    reattached to that concrete ``shared_dim`` with its entries left verbatim -- an entry
    ``1/d turns`` sitting on a container dimension of concrete ``2`` denotes a different
    (and wrong) angle once ``d``'s binding is substituted in, and silently keeps citing a
    symbol its own container dimension has already been resolved past.

    The alternative considered was to make ``dimension_agreement`` itself refuse any
    candidate whose phase entries reference a dimension symbol shared_dim resolution has
    bound, rather than have the builder substitute through it. That would satisfy
    ``claude.md``'s "phases are first-class symbolic objects" only in the thinnest sense
    (never touching them), at the cost of rejecting a fusion that is, in fact, perfectly
    well-defined at the assumed binding -- exactly the binding :attr:`FusionMatch.bindings`
    already records as an assumption the certificate carries forward. Substituting is the
    behavior :meth:`~qufzx.algebra.phase.PhaseVector.substitute` already exists to perform
    (see that method's docstring: dimension and phase symbols are substituted through the
    same mapping, and it re-checks each entry's index bound against the newly-concrete
    dimension), and it is what keeps scalars and phases exact rather than merely refusing to
    look at them: the resulting phase is the actual angle implied by the binding, not an
    approximation or a discarded one. This also keeps match-approval and build-applicability
    the same predicate by construction (this function's return value, or the
    :class:`~qufzx.algebra.phase.PhaseDomainError` it raises, is identical whether called
    from here or from the builder), rather than two similar-looking checks kept in sync by
    hand.

    Raises :class:`~qufzx.algebra.phase.PhaseDomainError` if, after substitution, an entry's
    index falls outside ``shared_dim``'s valid range (substitution only changes an entry's
    *value*, never its index, so this is exactly the same index-bound check either caller
    would hit reattaching the original, unsubstituted entries -- substituting first does not
    change which candidates this can reject).
    """
    concrete_bindings = {name: dim.to_int() for name, dim in bindings.items() if dim.is_concrete}
    substituted = (
        phase.substitute(cast(Mapping[PhaseSymbolKey, PhaseSubstituteValue], concrete_bindings))
        if concrete_bindings
        else phase
    )
    entry_symbols: set[str] = set()
    for entry in phase.entries().values():
        entry_symbols |= entry.free_symbols
    applied = {name: bindings[name] for name in concrete_bindings if name in entry_symbols}
    return PhaseVector(shared_dim, substituted.entries()), MappingProxyType(applied)


def _validate_wire_endpoint(
    diagram: Diagram, wire_or_boundary_ref: Wire | PortRef, ref: PortRef
) -> None:
    """Raise ``RewriteGrammarError`` if ``ref`` names an unknown node or out-of-range index.

    Called for both endpoints of every wire in the diagram, and (Phase 5 post-closing audit
    round 18, Defect 2) for every ``boundary_inputs``/``boundary_outputs`` entry -- see
    "Malformed wire references" in the module docstring for why this must not depend on any
    other property of the wire (or, now, the boundary list) or the candidate pair it might
    otherwise sit on. ``wire_or_boundary_ref`` is only used to phrase the raised message: a
    ``Wire`` for the wire-endpoint call sites (``ref`` is one of its own two endpoints, so
    the message can name the whole wire for context), or the bare ``PortRef`` itself for a
    boundary entry (there is no enclosing ``Wire`` to name -- the ref *is* the reference,
    passed as both parameters at the boundary call site).
    """
    node = diagram.nodes.get(ref.node_id)
    if node is None:
        if isinstance(wire_or_boundary_ref, Wire):
            context = f"wire {wire_or_boundary_ref!r}"
            explanation = "a live wire can never legitimately name a removed node"
        else:
            context = f"boundary entry {wire_or_boundary_ref!r}"
            explanation = "a live boundary entry can never legitimately name a removed node"
        raise RewriteGrammarError(
            f"{context} references node id {ref.node_id!r} absent from the diagram; "
            f"Diagram.remove_node cascades, so {explanation}"
        )
    legs = node.legs(ref.direction)
    if ref.index >= len(legs):
        kind = "wire endpoint" if isinstance(wire_or_boundary_ref, Wire) else "boundary entry"
        raise RewriteGrammarError(
            f"{kind} {ref!r} is out of range for node {ref.node_id!r}: it has only "
            f"{len(legs)} {ref.direction.value} leg(s)"
        )


@dataclass(frozen=True, slots=True)
class FusionResolution:
    """The one verification predicate behind every :data:`FUSION_SIDE_CONDITIONS` entry.

    Returned by :func:`resolve_fusion_match`, computed fresh from ``(diagram, a_id, b_id,
    wire)`` alone -- never from a pre-existing :class:`FusionMatch`'s own fields. See the
    module docstring's "One verification predicate" paragraph for why this exists and who
    calls it.

    ``outcomes`` covers exactly the seven :data:`FUSION_SIDE_CONDITIONS` names, in that
    order, each independently derived. ``passed`` is ``True`` iff every one of them passed.
    When ``True``, ``shared_dim``, ``bindings``, and ``dimension_constraints`` are the ground
    truth to build a merged node from -- the only values
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` may use for graph surgery,
    never a pre-existing match's own same-named fields, which a hand-built or foreign
    ``FusionMatch`` could have fabricated. When ``False``, ``bindings`` and
    ``dimension_constraints`` are best-effort placeholders (computed only as far as
    resolution got before the first failing condition) and must not be used for anything but
    diagnostics -- a condition that was never reached because an earlier one already failed
    is still recorded, as a failing outcome whose detail says so, so ``outcomes`` always has
    exactly seven entries regardless of where resolution stopped.

    ``shared_dim`` is ``Dim | None``, not a placeholder ``Dim`` (Phase 5 post-closing audit):
    a failed resolution has no shared dimension to report -- the two legs may not even agree
    on one, which is exactly why resolution failed -- so unlike ``bindings``/
    ``dimension_constraints`` (which have a natural, harmless "nothing accumulated yet"
    empty value), there is no ``Dim`` that means "no shared dimension" without also being a
    plausible-looking real one. ``None`` makes "resolution failed, don't read this" a type
    error at every call site rather than a caller-side discipline of checking ``passed``
    first and hoping every caller remembers to. ``shared_dim`` is only ever non-``None`` when
    ``passed`` is ``True``, and every reader of this field (``find_matches``,
    ``spider_fusion_builder``) is downstream of its own ``if not resolution.passed`` guard.
    """

    passed: bool
    shared_dim: Dim | None
    bindings: Mapping[str, Dim]
    dimension_constraints: tuple[DimensionConstraint, ...]
    outcomes: tuple[SideConditionOutcome, ...]


def _connecting_pair_failure_detail(failure: _ResolutionFailure) -> str:
    """Render a connecting-pair :class:`_ResolutionFailure`, distinguishing its two causes.

    Phase 5 post-closing audit round 23, Task 3: a non-unifying pair and a pair that unifies
    fine but whose binding contradicts an earlier one are different failures -- the former
    means the two legs are provably incompatible, the latter means they are compatible with
    each other but not with an assumption a different check already made -- so they get
    genuinely different wording, derived from ``failure.reason`` rather than folded into one
    generic "does not unify" string regardless of which one actually happened.
    """
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"{failure.assumed} == {failure.equal_to} unifies, but only by binding a "
            "symbol to a value that contradicts an earlier binding accumulated this "
            "resolution (see _merge_bindings)"
        )
    return f"{failure.assumed} != {failure.equal_to}: the connecting pair does not unify"


def _leg_failure_detail(side: str, failure: _ResolutionFailure) -> str:
    """Render a surviving-leg :class:`_ResolutionFailure` for node ``side`` ('A' or 'B').

    Same distinction as :func:`_connecting_pair_failure_detail`, for the leg-sweep call
    sites.
    """
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"a surviving leg of the {side}-side node ({failure.assumed}) unifies with "
            f"shared_dim ({failure.equal_to}) only by binding a symbol to a value that "
            "contradicts an earlier binding accumulated this resolution"
        )
    return (
        f"a surviving leg of the {side}-side node ({failure.assumed}) does not unify with "
        f"shared_dim ({failure.equal_to})"
    )


def _phase_failure_detail(failure: _ResolutionFailure) -> str:
    """Render a phase-dimension :class:`_ResolutionFailure`, distinguishing all four causes.

    A phase has more ways to fail than a leg or the connecting pair (module docstring,
    condition 7): a genuine ``FAILURE``, a ``DEFERRED`` unify (tolerated for a leg, not for a
    phase), a binding to a non-concrete ``Dim`` (also tolerated for a leg, not for a phase),
    and a contradictory rebind. Each gets its own wording -- this is the *in-loop* phase
    failure detail; the distinct post-loop ``reattach_phase`` detail (an entry falling out of
    range once every binding is substituted in, rather than a non-unifying ``Dim``) is built
    separately at its own call site in :func:`resolve_fusion_match`, deliberately, since it
    is a genuinely different failure this function never sees.
    """
    if failure.reason is _FailureReason.PHASE_DEFERRED:
        return (
            f"a present phase dimension ({failure.assumed}) unifies only as DEFERRED "
            f"against the resolved shared leg dimension {failure.equal_to} -- a DEFERRED "
            "unify is not accepted for a phase, see the module docstring, condition 7"
        )
    if failure.reason is _FailureReason.PHASE_NON_CONCRETE_BINDING:
        return (
            f"a present phase dimension ({failure.assumed}) unifies with the resolved "
            f"shared leg dimension {failure.equal_to} only by binding to a non-concrete "
            "Dim -- not accepted for a phase, see the module docstring's 'Non-concrete "
            "bindings' section"
        )
    if failure.reason is _FailureReason.CONTRADICTORY_REBIND:
        return (
            f"a present phase dimension ({failure.assumed}) unifies with the resolved "
            f"shared leg dimension {failure.equal_to}, but only by binding a symbol to a "
            "value that contradicts an earlier binding accumulated this resolution"
        )
    return (
        f"a present phase dimension ({failure.assumed}) does not unify with the resolved "
        f"shared leg dimension {failure.equal_to}"
    )


def _consumed_port_claim_conflict(
    diagram: Diagram, ref: PortRef, consuming_wire: Wire
) -> str | None:
    """``None`` if ``ref`` is claimed only by ``consuming_wire`` and is on no boundary list.

    Otherwise, a human-readable description of what else claims it: a second wire, or a
    boundary entry (or both). Recomputed from ``diagram`` alone on every call -- consistent
    with every other condition :func:`resolve_fusion_match` decides -- rather than threaded
    in from a precomputed whole-diagram accumulator the way :func:`find_matches` used to
    build one (see the module docstring, "Match-implies-applicable and multiply-claimed
    ports", Phase 5 post-closing audit round 23, Task 2).
    """
    other_wire_claims = sum(
        1 for wire in diagram.wires if wire != consuming_wire and ref in (wire.a, wire.b)
    )
    on_boundary = ref in diagram.boundary_inputs or ref in diagram.boundary_outputs
    if not other_wire_claims and not on_boundary:
        return None
    parts = []
    if other_wire_claims:
        parts.append(f"claimed by {other_wire_claims} other wire(s)")
    if on_boundary:
        parts.append("listed on a boundary list")
    return f"{ref} is " + " and ".join(parts)


def resolve_fusion_match(
    diagram: Diagram, a_id: NodeId, b_id: NodeId, wire: Wire
) -> FusionResolution:
    """Decide, from ``diagram`` alone, whether ``wire`` is a legal fusion of ``a_id``/``b_id``.

    The single shared predicate behind all seven conditions in the module docstring -- see "One
    verification predicate" there for the full account of why this function exists and the
    Phase 5 round-12 audit defect (A1/A2/A4) it closes. :func:`find_matches` calls this once
    per candidate wire to decide whether to report a match at all, and to populate the
    :class:`FusionMatch` it returns. :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`
    calls it again, fresh, against the diagram it was actually handed, and trusts only this
    function's return value for graph surgery -- never ``match.shared_dim``,
    ``match.bindings``, or an unverified ``node_a.generator_type`` -- so a foreign or
    hand-built match cannot smuggle a fabricated value past the builder by simply asserting
    it in a passing-looking ``side_condition_outcomes`` tuple.

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError` for a structurally malformed
    request: ``a_id == b_id``, either node id absent from ``diagram``, ``wire`` not actually
    incident on both ``a_id`` and ``b_id``, or either of ``wire``'s own endpoints naming an
    unknown node id or an out-of-range port index (via :func:`_validate_wire_endpoint`).
    These are requests that cannot even be evaluated, not candidates that evaluate to
    "no" -- the same domain/grammar split :mod:`qufzx.rewrite.rule`'s module docstring
    states for this package as a whole.

    Never mutates ``diagram``.
    """
    if a_id == b_id:
        raise RewriteGrammarError(
            f"resolve_fusion_match: a_id and b_id must be distinct, both were {a_id!r}"
        )
    node_a = diagram.nodes.get(a_id)
    node_b = diagram.nodes.get(b_id)
    if node_a is None or node_b is None:
        raise RewriteGrammarError(
            f"resolve_fusion_match: node id(s) {a_id!r}, {b_id!r} not both present in the diagram"
        )
    wire_node_ids = {wire.a.node_id, wire.b.node_id}
    if wire_node_ids != {a_id, b_id}:
        raise RewriteGrammarError(
            f"resolve_fusion_match: wire {wire!r} does not connect {a_id!r} and {b_id!r} "
            f"(it connects {sorted(wire_node_ids)!r})"
        )
    _validate_wire_endpoint(diagram, wire, wire.a)
    _validate_wire_endpoint(diagram, wire, wire.b)

    # Defect 1 (Phase 5 post-closing audit): everything above validates that ``wire``
    # *looks* like it could join a_id and b_id -- its own endpoints name real nodes and
    # ports, and its node ids are exactly {a_id, b_id} -- but none of that establishes
    # that ``wire`` is actually an element of ``diagram.wires`` rather than a freestanding
    # ``Wire`` object a caller merely constructed to look like one. A wire ghost-written
    # against two real, correctly-incident ports is otherwise accepted by every check
    # above and passed on to graph surgery, which then consumes a wire the diagram never
    # had. This is a malformed request, the same category as non-incidence above, so it
    # is checked here, before any side condition is evaluated, and raises the same
    # RewriteGrammarError. ``all_wires`` is snapshotted once (``Diagram.wires`` rebuilds a
    # fresh ``frozenset`` on every access) and reused below for ``other_wire_count``.
    all_wires = diagram.wires
    if wire not in all_wires:
        raise RewriteGrammarError(
            f"resolve_fusion_match: wire {wire!r} is not an element of diagram.wires -- "
            "a wire naming two correctly-incident ports is not itself proof that the "
            "diagram actually contains it"
        )

    ref_a = wire.a if wire.a.node_id == a_id else wire.b
    ref_b = wire.b if wire.a.node_id == a_id else wire.a

    other_wire_count = sum(
        1
        for other in all_wires
        if other != wire and {other.a.node_id, other.b.node_id} == {a_id, b_id}
    )

    outcomes: list[SideConditionOutcome] = [
        SideConditionOutcome("distinct_nodes", True, f"{a_id!r} != {b_id!r}"),
    ]

    # Two distinct facts, reported under the condition that actually decides them (Phase 5
    # post-closing audit): "do the two nodes carry the identical generator type" and "is
    # that generator type one this pattern is registered to fuse at all" used to be folded
    # into one -- an unregistered same-typed pair reported same_generator_type=True (the
    # types genuinely are equal) with the real reason ("not a registered fusable generator
    # type") buried inside consumed_wire_direction_permitted_for_color's detail instead,
    # a condition whose own declared description is about wire direction, not fusability.
    # same_generator_type's own declared description ("both nodes are the same *registered*
    # spider color") already promised this; the implementation now actually checks it.
    generator_types_match = node_a.generator_type == node_b.generator_type
    is_fusable_type = (
        generator_types_match and node_a.generator_type.name in _FUSABLE_GENERATOR_NAMES
    )
    if not generator_types_match:
        same_type_detail = (
            f"{a_id!r} is {node_a.generator_type.name!r} but {b_id!r} is "
            f"{node_b.generator_type.name!r}"
        )
    elif not is_fusable_type:
        same_type_detail = (
            f"both nodes are {node_a.generator_type.name!r}, but that is not a registered "
            "fusable generator type"
        )
    else:
        same_type_detail = f"both nodes are {node_a.generator_type.name!r}"
    same_type = is_fusable_type
    outcomes.append(SideConditionOutcome("same_generator_type", same_type, same_type_detail))

    outcomes.append(
        SideConditionOutcome(
            "parallel_wires_become_self_loops",
            True,
            f"{other_wire_count} other wire(s) join the two nodes, surviving as "
            "self-loop(s) on the merged spider",
        )
    )

    def _failed(remaining_names: tuple[str, ...], reason: str = "") -> FusionResolution:
        for name in remaining_names:
            outcomes.append(SideConditionOutcome(name, False, reason))
        return FusionResolution(
            passed=False,
            # None, not a placeholder concrete Dim: a failed resolution has no shared
            # dimension a caller may read (see FusionResolution's own docstring) -- making
            # this unrepresentable at the type level, rather than a caller-side "check
            # passed first" discipline, closes the Phase 5 post-closing audit's
            # "_failed() returns a meaningless dimension a caller ignoring passed could
            # silently read" defect.
            shared_dim=None,
            bindings=MappingProxyType({}),
            dimension_constraints=(),
            outcomes=tuple(outcomes),
        )

    if not same_type:
        return _failed(
            (
                "consumed_wire_direction_permitted_for_color",
                "consumed_ports_singly_claimed",
                "dimension_agreement",
                "phase_dimension_agreement",
            ),
            "not evaluated: same_generator_type failed first",
        )

    # By construction, same_type above already guarantees node_a.generator_type.name is in
    # _FUSABLE_GENERATOR_NAMES -- resolution returns early via _failed() otherwise, so this
    # condition is only ever reached for a registered fusable, same-typed pair. No separate
    # "not fusable" branch is needed here any more (Phase 5 post-closing audit).
    same_direction = ref_a.direction == ref_b.direction
    direction_ok = (
        not same_direction
        or node_a.generator_type.name in _SAME_DIRECTION_FUSABLE_GENERATOR_NAMES
    )

    direction_detail = (
        f"{ref_a} (direction={ref_a.direction.value}) -> "
        f"{ref_b} (direction={ref_b.direction.value})"
    )
    if same_direction:
        direction_detail += (
            f" (same-direction {ref_a.direction.value}-{ref_b.direction.value} wire, "
            f"permitted for {node_a.generator_type.name!r} only -- see the module "
            "docstring, condition 4)"
            if direction_ok
            else (
                f" (same-direction {ref_a.direction.value}-{ref_b.direction.value} wire "
                f"is not permitted for {node_a.generator_type.name!r} -- see the module "
                "docstring, condition 4)"
            )
        )
    outcomes.append(
        SideConditionOutcome(
            "consumed_wire_direction_permitted_for_color", direction_ok, direction_detail
        )
    )

    if not direction_ok:
        return _failed(
            ("consumed_ports_singly_claimed", "dimension_agreement", "phase_dimension_agreement"),
            "not evaluated: consumed_wire_direction_permitted_for_color failed first",
        )

    # Condition 5: decided from (diagram, a_id, b_id, wire) alone, before the dimension
    # fixpoint below, exactly as every other condition is -- see the module docstring,
    # "Match-implies-applicable and multiply-claimed ports" (Phase 5 post-closing audit
    # round 23, Task 2). A candidate that fails this has no legal port_mapping regardless of
    # what the fixpoint would find, so it is checked first rather than after paying for a
    # fixpoint whose result would be discarded either way.
    claim_conflict_a = _consumed_port_claim_conflict(diagram, ref_a, wire)
    claim_conflict_b = _consumed_port_claim_conflict(diagram, ref_b, wire)
    claims_ok = claim_conflict_a is None and claim_conflict_b is None
    claim_detail = (
        f"neither {ref_a} nor {ref_b} is claimed by another wire or listed on a boundary"
        if claims_ok
        else "; ".join(d for d in (claim_conflict_a, claim_conflict_b) if d is not None)
    )
    outcomes.append(
        SideConditionOutcome("consumed_ports_singly_claimed", claims_ok, claim_detail)
    )
    if not claims_ok:
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            "not evaluated: consumed_ports_singly_claimed failed first",
        )

    legs_a = node_a.legs(ref_a.direction)
    legs_b = node_b.legs(ref_b.direction)
    port_a = legs_a[ref_a.index]
    port_b = legs_b[ref_b.index]

    record = _ConstraintRecord()
    shared_dim = port_a.dim
    bindings: dict[str, Dim] = {}

    # Conditions 6 and 7 run as one bounded fixpoint: each pass re-derives the connecting
    # pair's own equality, then re-unifies every surviving leg of both nodes, then every
    # present phase's own Dim -- each against shared_dim (or, for the connecting pair,
    # against the other connected leg) as of the point reached *so far in that same pass*,
    # refining `bindings` and shared_dim in place on any concrete binding. The pass repeats
    # until a full pass adds nothing: both shared_dim and bindings are unchanged from the
    # pass's own start (D1's fix -- see below for why shared_dim stopping alone is not a
    # sound termination signal). `bindings` is not pass-scoped: it is the single
    # whole-candidate accumulator every pass reads from and writes into, so each successive
    # pass is strictly more informed than the last. It is also monotone -- a key is only
    # ever added, never rebound to a different value -- but not because _merge_bindings
    # rejects a contradictory rebind (round 23 correction: it never actually gets the
    # chance to. Every operand reaching a unify call in this fixpoint has already been
    # passed through _resolve_with_bindings first, which substitutes every symbol `bindings`
    # already names with its bound value; a symbol that is already bound is therefore no
    # longer *free* in what gets unified, so Dim.unify's own UnifyResult.bindings can never
    # name it again. Monotonicity is a consequence of that pre-resolution discipline, not of
    # _merge_bindings' own contradiction check -- see that function's docstring for what its
    # guard is actually for. Duplicate assumptions are ruled out by the *record*, not by
    # this loop: `record` is keyed by ConstraintSource, so a source re-derived on a later
    # pass replaces its own entry rather than appending a second one (see _ConstraintRecord).
    #
    # D1's root cause: "shared_dim stopped changing" is not the same fact as "bindings
    # stopped changing". bindings can grow on a pass whose new binding does not touch any
    # symbol appearing in shared_dim itself (e.g. binding a symbol that occurs only in a
    # surviving leg's own dim, or rebinding a symbol to the value it already holds) --
    # exiting on shared_dim alone then leaves whatever was checked earlier in that very pass
    # unre-checked against the newly accumulated bindings, which is exactly how an
    # unsatisfiable constraint set (e.g. e*f == 2 and e == 2 and f == 2, from legs
    # [e*f, e, f]) went undetected. Checking both signals -- i.e. requiring an entire pass to
    # add nothing to either shared_dim or bindings before exiting -- is what actually catches
    # it: the pass that binds e := 2 and f := 2 does not itself stabilise (bindings grew), so
    # a further pass runs and re-resolves e*f through those bindings before re-unifying it,
    # surfacing the contradiction as an ordinary Dim.unify FAILURE on that later pass -- not
    # by _merge_bindings. _verify_fixpoint_closure is the same check restated as a
    # structural post-loop guard on the convergence path, rather than something left to pass
    # repetition alone to have caught.
    fixpoint_budget_exhausted = False

    for _pass_index in range(_MAX_FIXPOINT_PASSES):
        pass_start_dim = shared_dim
        pass_start_bindings = dict(bindings)

        next_dim = _unify_connecting_pair(port_a.dim, port_b.dim, shared_dim, bindings, record)
        if isinstance(next_dim, _ResolutionFailure):
            return _failed(
                ("dimension_agreement", "phase_dimension_agreement"),
                _connecting_pair_failure_detail(next_dim),
            )
        shared_dim = next_dim

        next_dim = _unify_surviving_legs(node_a, a_id, ref_a, shared_dim, bindings, record)
        if isinstance(next_dim, _ResolutionFailure):
            return _failed(
                ("dimension_agreement", "phase_dimension_agreement"),
                _leg_failure_detail("A", next_dim),
            )
        shared_dim = next_dim

        next_dim = _unify_surviving_legs(node_b, b_id, ref_b, shared_dim, bindings, record)
        if isinstance(next_dim, _ResolutionFailure):
            return _failed(
                ("dimension_agreement", "phase_dimension_agreement"),
                _leg_failure_detail("B", next_dim),
            )
        shared_dim = next_dim

        # Defect 3 (Phase 5 post-closing audit round 18): snapshotted *before* calling
        # _unify_phase_dims, not read back out of `shared_dim`/`bindings` after it returns.
        # This pass's leg sweep (connecting pair, A's surviving legs, B's surviving legs,
        # all three just above) has, as of this point, been fully verified against exactly
        # this shared_dim/bindings state -- that is what condition 6 (dimension_agreement)
        # must be reported against if a phase now fails. _unify_phase_dims can bind phase
        # A's own symbol -- refining both `bindings` and `shared_dim` in place -- before
        # failing on phase B in that same call (see its own docstring); reporting condition
        # 6 against the state *after* that call would claim the legs were verified against
        # a shared_dim they were never actually checked against.
        leg_verified_shared_dim = shared_dim
        leg_verified_bindings = dict(bindings)

        phase_result = _unify_phase_dims(node_a, node_b, a_id, b_id, shared_dim, bindings, record)
        if isinstance(phase_result, _ResolutionFailure):
            # Root cause (Defect 3): a phase-dim FAILURE is decided -- condition 7 does not
            # hold -- and is reported as exactly that, directly, rather than falling through
            # to _verify_fixpoint_closure. A prior version of this function let every break
            # out of this loop (phase failure or genuine convergence alike) fall through to
            # that post-loop closure check; closure re-verifies phases from scratch under
            # the same unify contract that just failed, so it necessarily failed too,
            # reporting BOTH dimension_agreement and phase_dimension_agreement as failed
            # with the closure guard's own "this is unreachable" message -- which was false
            # exactly here: the guard's unreachability argument (every check the closure
            # re-verifies was already verified this same pass, against this same state) only
            # holds when the loop reaches its termination condition below, never when it
            # exits via a phase failure. See _verify_fixpoint_closure's own docstring, fixed
            # to describe only the path it now actually runs on. dimension_agreement is
            # reported True (from the leg-verified snapshot above, via
            # _dimension_agreement_outcome -- a leg-sweep FAILURE already returned above,
            # via _failed, so reaching here means every leg genuinely did unify) with a
            # leg-accurate detail; phase_dimension_agreement is reported False with the
            # dedicated per-phase detail below -- never the closure-guard string, which
            # therefore appears in neither outcome for this path.
            outcomes.append(
                _dimension_agreement_outcome(
                    port_a.dim, port_b.dim, leg_verified_shared_dim, leg_verified_bindings, record
                )
            )
            phase_detail = _phase_failure_detail(phase_result)
            outcomes.append(
                SideConditionOutcome(
                    "phase_dimension_agreement", False, phase_detail, deferred=False
                )
            )
            # Same placeholder convention as _failed() (Phase 5 post-closing audit,
            # pre-round-18 Defect 3): shared_dim=None, bindings/dimension_constraints empty
            # on any failure -- see FusionResolution's own docstring. Not routed through
            # _failed() itself since that helper always marks its ``remaining_names`` False
            # and appends nothing for dimension_agreement; this path needs the opposite mix
            # (dimension_agreement True, phase_dimension_agreement False), which _failed()
            # cannot express.
            return FusionResolution(
                passed=False,
                shared_dim=None,
                bindings=MappingProxyType({}),
                dimension_constraints=(),
                outcomes=tuple(outcomes),
            )
        shared_dim = phase_result

        if shared_dim == pass_start_dim and bindings == pass_start_bindings:
            break
    else:
        # The cap is a resolver *iteration budget*, not a dimension disagreement, and it is
        # reported as exactly that. bindings is monotone (never shrinks, never rebinds --
        # _merge_bindings) and is drawn from the finite free-symbol set of node_a's legs and
        # phase, node_b's legs and phase, and shared_dim's own lineage from port_a.dim; a
        # pass that does not stabilise therefore strictly grows bindings by at least one
        # fresh key, which bounds the number of non-stabilising passes by that finite symbol
        # count. The guard is kept anyway, conservatively refusing the candidate rather than
        # looping forever, because that bound rests on Dim.unify's current contract and
        # Phase 10 replaces its body. Both conditions 6 and 7 are reported failed, with the
        # same detail: the fixpoint decides them jointly, so when it does not terminate
        # neither one was decided, and blaming either alone would send a reader to the wrong
        # place.
        fixpoint_budget_exhausted = True

    if fixpoint_budget_exhausted:
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            f"the bounded leg/phase resolution fixpoint did not stabilise within "
            f"{_MAX_FIXPOINT_PASSES} passes (_MAX_FIXPOINT_PASSES): this is a resolver "
            "iteration budget, not a dimension or phase-dimension disagreement -- neither "
            "condition was decided",
        )

    # Reached only via the loop's own convergence break above (a phase failure returns
    # directly, above; budget exhaustion returns directly, just above) -- see
    # _verify_fixpoint_closure's own docstring for why its "unreachable" claim is actually
    # true on exactly this path, and only this path (Defect 3, Phase 5 post-closing audit
    # round 18).
    if not _verify_fixpoint_closure(
        node_a, node_b, a_id, b_id, ref_a, ref_b, port_a.dim, port_b.dim, shared_dim, bindings
    ):
        return _failed(
            ("dimension_agreement", "phase_dimension_agreement"),
            "post-loop closure check failed: a resolved leg, phase, or the connecting pair "
            "does not unify with the final shared_dim under the final bindings -- see "
            "_verify_fixpoint_closure; this is unreachable on the convergence path this "
            "call site is reached from (a phase failure or budget exhaustion both return "
            "before reaching here) given the fixpoint's own termination guarantee, and is "
            "checked anyway as a structural guard, not a property left to tests alone",
        )

    outcomes.append(
        _dimension_agreement_outcome(port_a.dim, port_b.dim, shared_dim, bindings, record)
    )

    phase_a_dim = node_a.phase.dim if node_a.phase is not None else None
    phase_b_dim = node_b.phase.dim if node_b.phase is not None else None
    phase_dims_present = tuple(d for d in (phase_a_dim, phase_b_dim) if d is not None)

    # reattach_phase's own index-bound check must run against the FINAL shared_dim
    # (post-fixpoint), not an intermediate one from an earlier pass: substitution only
    # changes an entry's *value*, never its index, so an entry that falls out of range only
    # once the fixpoint's own later bindings resolve is still caught here. This is a
    # genuinely different failure mode than a phase-dim FAILURE inside the loop above (see
    # _phase_failure_detail: that is a non-unifying Dim, a DEFERRED unify, or a non-concrete
    # binding) -- this one is a *unifying* Dim whose own entries fall out of range once
    # substituted, discoverable only after the fixpoint has fully converged and
    # reattach_phase can be tried for real. Round 23 correction: a prior version of this
    # detail string listed every possible cause verbatim (including the in-loop ones this
    # path cannot actually be reached by, since a phase-dim FAILURE/DEFERRED/non-concrete
    # binding already returns above), making the two phase_dimension_agreement failure
    # details near-identical copies of each other -- see the module docstring's Task 3 note.
    for phase in (node_a.phase, node_b.phase):
        if phase is None:
            continue
        try:
            reattach_phase(phase, shared_dim, bindings)
        except PhaseDomainError as exc:
            phase_detail = (
                f"a present phase dimension unifies with the resolved shared leg dimension "
                f"{shared_dim}, but at least one of its own entries falls out of range once "
                f"every binding this fixpoint accumulated is substituted in ({exc})"
            )
            outcomes.append(
                SideConditionOutcome(
                    "phase_dimension_agreement", False, phase_detail, deferred=False
                )
            )
            return _failed(())

    # Round 20, Task 7: both the rendered *names* and their *values* are now read off the
    # single source this detail claims to describe -- ``record.entries()`` -- rather than
    # names from one accumulator (the now-deleted ``phase_bound_names``, threaded out of
    # ``_unify_phase_dims`` across passes) and values from another (``record`` itself). The
    # two used to agree only because ``Dim.unify``'s current placeholder body binds at most
    # one symbol per call and a phase binding is only ever recorded once fully concrete --
    # true today, not a property this rendering step should depend on (see the module
    # docstring's "Round 20" section, completing round 18's Defect 4 / round 19's Task 1
    # class: a rendered detail must read every operand it prints off the same record it
    # claims to describe, including the keys it iterates, not only the values it looks up).
    # Walked in ``record.entries()``'s own first-derivation order (see that method's
    # docstring) and de-duplicated with ``dict.fromkeys`` to preserve that order while
    # dropping a name a later pass re-bound under the same key (D6's original concern,
    # preserved here even though the accumulator it was patching over is gone).
    phase_bound_values: dict[str, Dim] = {}
    for phase_entry in record.entries():
        if (
            phase_entry.source.kind is ConstraintSourceKind.NODE_PHASE
            and phase_entry.outcome is ConstraintOutcome.BOUND
        ):
            phase_bound_values.update(phase_entry.bound_here)
    unique_bound_names = list(dict.fromkeys(phase_bound_values))
    phase_detail = (
        "no phase present on either node"
        if not phase_dims_present
        else (
            "present phase dimension(s) unify with the resolved shared leg dimension "
            f"{shared_dim}"
            + (
                "; assuming "
                + ", ".join(
                    f"{name} := {phase_bound_values[name]}" for name in unique_bound_names
                )
                if unique_bound_names
                else ""
            )
        )
    )
    outcomes.append(
        # Always deferred=False: unlike condition 6, a genuinely DEFERRED phase-dim
        # unify is rejected outright (see _unify_phase_dims) rather than
        # accepted-and-flagged, so a *passing* phase_dimension_agreement outcome is
        # never itself resting on an undecided unify -- only, at most, on a binding
        # (which dimension_constraints records, but which this flag -- following
        # condition 6's own convention -- does not count as "deferred").
        SideConditionOutcome("phase_dimension_agreement", True, phase_detail, deferred=False)
    )
    return FusionResolution(
        passed=True,
        shared_dim=shared_dim,
        bindings=MappingProxyType(dict(bindings)),
        dimension_constraints=record.entries(),
        outcomes=tuple(outcomes),
    )


def find_matches(diagram: Diagram) -> tuple[FusionMatch, ...]:
    """Find every same-color spider fusion occurrence in ``diagram``. See the module docstring.

    Never mutates ``diagram``, and does not require ``diagram`` to be well-formed --
    :func:`~qufzx.diagram.validate.validate` is never called here. Returns matches sorted
    by ``(a_id, b_id)``, tiebroken by the consumed wire's own per-side (direction, index).
    """
    # Malformed-wire detection (an unknown node id or an out-of-range port index) must be
    # independent of every other property of the wire or the candidate pair it happens to
    # sit on -- generator color, fusable-color-ness, wire direction, single-vs-parallel
    # wiring, and self-loop-ness alike -- so every wire's both endpoints are checked here,
    # before any grouping or filtering. Checking this only after grouping (as an earlier
    # version did) let a malformed wire escape undetected as a bare non-match whenever it
    # happened to be dropped first by the self-loop skip or the parallel-wire-pair filter
    # below, masking the same structural defect differently depending on unrelated shape.
    # Snapshotted once, not re-read from ``diagram.wires`` on every pass below: the two
    # passes over the wire set (malformed-endpoint check, pair grouping) used to each
    # re-materialise ``diagram.wires`` independently, doubling the cost of building whatever
    # collection backs that property for no reason -- neither pass needs a live view, and
    # neither mutates ``diagram``.
    #
    # Sorted, not the raw frozenset (Phase 5 post-closing audit round 18, Defect 1):
    # ``diagram.wires`` is a frozenset, and ``Wire``'s hash folds in ``Direction``'s
    # member-name hash, which is PYTHONHASHSEED-dependent. The malformed-endpoint pass below
    # raises on the *first* offending wire it finds -- with more than one malformed wire in
    # the diagram, an unsorted iteration would report a different one (a different exception
    # message) across processes. The later pass (pair grouping) would still produce the same
    # *matches* even unsorted, since ``find_matches`` sorts its returned tuple explicitly
    # below regardless -- but sorting once, up front, keeps both passes uniformly
    # deterministic rather than leaving readers to work out which one needs it and which
    # merely happens not to.
    wires = tuple(sorted(diagram.wires, key=lambda w: w.sort_key()))

    for wire in wires:
        _validate_wire_endpoint(diagram, wire, wire.a)
        _validate_wire_endpoint(diagram, wire, wire.b)

    # Defect 2 (Phase 5 post-closing audit round 18): a boundary entry naming an unknown
    # node id or an out-of-range port index is held to the identical standard as a wire
    # endpoint above -- both are references ``_remap_endpoint`` (in
    # :mod:`qufzx.rewrite.engine`) treats identically once a match reaches ``apply``, so a
    # malformed one of either kind must be caught here, before any candidate is even
    # grouped, rather than only on the wire side. Before this fix, a boundary_inputs/
    # boundary_outputs entry naming no real port at all (e.g. an out-of-range index on a
    # node with fewer legs, or an unknown node id -- both legitimately constructible on an
    # un-validated diagram, since :mod:`qufzx.diagram.graph` is deliberately permissive, see
    # that module's docstring) reached ``apply`` unexamined as long as it did not happen to
    # sit on a consumed port; when it did, ``apply``'s step 5 (via ``_remap_endpoint``)
    # raised ``RewriteDomainError`` -- a *different* error, from a *different* step, than
    # this module's own "malformed wire reference" contract promises, and one this module's
    # own module docstring's "Match-implies-applicable" section did not cover at all,
    # despite ``find_matches`` returning a match whose ``apply`` was not, in fact,
    # guaranteed to succeed cleanly.
    #
    # Sorted for the same reason the wire pass above is (Phase 5 post-closing audit round
    # 18, Defect 1): ``PortRef``'s hash is PYTHONHASHSEED-dependent (via ``Direction``), and
    # ``boundary_inputs``/``boundary_outputs`` are already ordered tuples -- but a malformed
    # entry on one list should not have its reported order interleaved arbitrarily with the
    # other, so both lists are walked in their own declared order, boundary_inputs first
    # (the same order ``Diagram.boundary_inputs``/``boundary_outputs`` themselves are always
    # read in elsewhere in this codebase), each entry checked via the same
    # :func:`_validate_wire_endpoint` machinery the wire pass uses -- reusing that function
    # (renamed in spirit only; it validates *any* ``PortRef``, not just a wire's own) rather
    # than a second, hand-duplicated existence/range check that could drift out of sync with
    # it. ``wire`` is only used by :func:`_validate_wire_endpoint` to name the offending
    # object in its raised message; passing the ref itself there (rather than a real
    # ``Wire``) is what the ``wire_or_boundary_ref`` parameter name below documents.
    for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs):
        _validate_wire_endpoint(diagram, ref, ref)

    # Defect 2 (match-implies-applicable), Phase 5 post-closing audit round 23, Task 2: a
    # port claimed by more than one wire, or both wired and listed on a boundary, is not a
    # legitimate fusion occurrence -- fusing across it would ask the builder to remap a
    # consumed port that a *third* reference (another wire, or a boundary entry) also still
    # names, which qufzx.rewrite.engine.apply's port_mapping coverage check (step 5)
    # correctly refuses to do silently. This used to be enforced by a bare filter here,
    # before any FusionMatch was even constructed -- it is now condition 5
    # (``consumed_ports_singly_claimed``), decided by :func:`resolve_fusion_match` itself
    # from ``diagram`` alone, the same single decision point every other condition already
    # goes through (see the module docstring, "Match-implies-applicable and multiply-claimed
    # ports"). No separate filter is needed here any more: a candidate that fails it is
    # simply a resolution whose ``passed`` is ``False``, dropped by the ordinary
    # ``if not resolution.passed: continue`` below.
    candidates_by_pair: dict[frozenset[NodeId], list[Wire]] = {}
    for wire in wires:
        if wire.a.node_id == wire.b.node_id:
            continue
        key = frozenset((wire.a.node_id, wire.b.node_id))
        candidates_by_pair.setdefault(key, []).append(wire)

    # Flattened once so the loop below stays single-level: every wire in every candidate
    # pair. How many *other* wires join that same pair (condition 3,
    # ``parallel_wires_become_self_loops``) is recomputed, from ``diagram`` alone, inside
    # :func:`resolve_fusion_match` below -- not threaded through here -- so that function
    # stays the single source of truth for every one of the seven side conditions, not six
    # of them plus one still assembled by this loop.
    wire_candidates = [
        wire for connecting_wires in candidates_by_pair.values() for wire in connecting_wires
    ]

    matches: list[FusionMatch] = []
    for wire in wire_candidates:
        a_id, b_id = _ordered_pair(wire)

        # See the module docstring, "One verification predicate": conditions 2 and 4-7 are
        # decided by exactly this call, the same function
        # :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` calls again to
        # re-verify the match before trusting any of its fields -- not a second,
        # independently-maintained copy of this logic.
        resolution = resolve_fusion_match(diagram, a_id, b_id, wire)
        if not resolution.passed:
            continue
        assert resolution.shared_dim is not None  # invariant: passed implies shared_dim is set

        matches.append(
            FusionMatch(
                a_id=a_id,
                b_id=b_id,
                wire=wire,
                shared_dim=resolution.shared_dim,
                side_condition_outcomes=resolution.outcomes,
                dimension_constraints=resolution.dimension_constraints,
                bindings=MappingProxyType(dict(resolution.bindings)),
            )
        )

    matches.sort(
        key=lambda m: (
            int(m.a_id),
            int(m.b_id),
            (m.wire.a if m.wire.a.node_id == m.a_id else m.wire.b).direction.value,
            (m.wire.a if m.wire.a.node_id == m.a_id else m.wire.b).index,
            (m.wire.b if m.wire.a.node_id == m.a_id else m.wire.a).direction.value,
            (m.wire.b if m.wire.a.node_id == m.a_id else m.wire.a).index,
        )
    )
    return tuple(matches)


def _ordered_pair(wire: Wire) -> tuple[NodeId, NodeId]:
    """The wire's two node ids as ``(lower, higher)``. See :class:`FusionMatch`."""
    if wire.a.node_id <= wire.b.node_id:
        return wire.a.node_id, wire.b.node_id
    return wire.b.node_id, wire.a.node_id


class FusionPattern(Pattern):
    """The :class:`~qufzx.rewrite.rule.Pattern` implementation for same-color spider fusion."""

    def find_matches(self, diagram: Diagram) -> tuple[Match, ...]:
        """Delegate to the module-level :func:`find_matches`. See the module docstring."""
        return find_matches(diagram)
