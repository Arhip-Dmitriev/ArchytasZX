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

"""Rewrite engine: applies rules at matches, returns new diagrams, and records step provenance.

:func:`apply` is the single entry point. It is generic over any future
:class:`~qufzx.rewrite.rule.Rule` -- it never inspects a match's rule-specific fields
(e.g. ``FusionMatch.a_id``) directly, only the generic contract
:class:`~qufzx.rewrite.rule.Match` and :class:`~qufzx.rewrite.rule.BuildResult` expose. All
of the rule-specific work (which nodes are consumed, what the replacement node's legs and
phase are, what scalar is introduced) happens inside the rule's own builder, in
:mod:`qufzx.rewrite.rules_library`; this module only knows how to splice a
:class:`~qufzx.rewrite.rule.BuildResult` into the rest of a diagram.

Algorithm.

1. Verify the match's certificate. :func:`~qufzx.rewrite.rule.check_side_condition_coverage`
   rejects (raising :class:`~qufzx.rewrite.rule.RewriteDomainError`) a match whose
   ``side_condition_outcomes`` do not name exactly ``rule.side_conditions`` -- no fewer, no
   more, no duplicates -- or whose named outcomes are not all ``passed``; a bare
   ``all(outcome.passed for outcome in ())`` is vacuously True, so checking coverage, not
   only passedness, is what actually closes that hole (see that function's docstring for the
   full account). This runs before the builder is even called; each rule's own builder
   applies the identical check against its own declared conditions (see
   ``spider_fusion_builder`` in :mod:`qufzx.rewrite.rules_library`), since a builder is
   reachable directly and not only through this function.
2. Copy. Never mutate the diagram passed in -- work entirely on ``diagram.copy()``
   (``working`` below) and return that. ``tests/test_engine.py`` asserts the original is
   byte-for-byte unchanged (same nodes, wires, boundaries, scalar) after a rewrite.
3. Build. Call ``rule.builder(working, match)``. Per :class:`~qufzx.rewrite.rule.BuildResult`'s
   contract, this mutates ``working`` by adding the replacement node(s) only -- it does not
   touch any wire, any boundary entry, or remove the consumed nodes. ``build_result.diagram``
   is the engine/builder contract field for *which* diagram was mutated: this step verifies
   it ``is`` (identity, not just equality) ``working``, raising
   :class:`~qufzx.rewrite.rule.RewriteGrammarError` otherwise, rather than declaring the
   field and never reading it.
4. Verify the build result belongs to this diagram. The builder's introduced scalar must
   agree with the rule's declared ``scalar_introduced``, and every one of
   ``build_result.consumed_wires`` and ``build_result.consumed_node_ids`` must actually be
   present in the working diagram -- raising :class:`~qufzx.rewrite.rule.RewriteGrammarError`
   for the latter, since a match (or a builder) naming a wire or node the diagram does not
   have is a malformed request, not a domain violation a builder computed correctly and then
   this step rejected. This step also validates the two remaining builder-reported fields
   step 5 and step 9 otherwise take on faith: every id in ``build_result.new_node_ids`` must
   actually exist in the working diagram, and every value in ``build_result.port_mapping``
   must name a real port (a node that exists, at an in-range index for its direction) --
   step 5 feeds ``port_mapping`` values directly into every surviving wire and boundary
   entry with no existence check of its own, and step 9 publishes ``new_node_ids`` verbatim
   for Phase 6 to replay against, so an unvalidated bad value from either field would
   otherwise surface far from its cause (a confusing failure deep in remapping, or in a
   certificate replay), or -- if it happens to alias a real port by coincidence -- not
   surface at all. It also rejects a repeated entry in ``build_result.consumed_node_ids``
   (Phase 5 round-12 audit, A3): every entry there names a real node individually, so the
   membership check above would not catch a duplicate, but step 6's removal loop would call
   :meth:`~qufzx.diagram.graph.Diagram.remove_node` twice on the same (by-then-already-
   removed) id, raising :class:`~qufzx.diagram.graph.GraphGrammarError` -- a different
   module's exception, escaping the ``RewriteError`` hierarchy this function's own docstring
   promises. A duplicate is a malformed request (a match cannot legitimately consume the
   same node twice), so it is rejected here, at the same point every other malformed
   ``BuildResult`` field is.
5. Remap every reference. This is the single most failure-prone part of a rewrite (see
   :mod:`qufzx.semantics.check`'s interface check, which fails on a boundary that lost its
   order or its entries before ever comparing a tensor). For every wire in ``working``
   *before* any node is removed: if it is one of ``build_result.consumed_wires``, drop it
   without replacement (it has been absorbed into the merged node). Otherwise, if either
   endpoint's node id is in ``build_result.consumed_node_ids``, remove the wire and re-add
   it with that endpoint replaced via ``build_result.port_mapping`` (an endpoint on a node
   *not* being consumed is left untouched). An endpoint on a node that *is* being consumed
   must appear in ``port_mapping``; if it does not, this step raises
   :class:`~qufzx.rewrite.rule.RewriteDomainError` naming the rule and the unmapped port
   rather than silently leaving a wire pointing at a node step 6 is about to remove (whose
   removal cascade would then silently drop that wire). If ``port_mapping`` directs both
   endpoints of a surviving wire to a single port, this step raises
   :class:`~qufzx.rewrite.rule.RewriteGrammarError` rather than constructing a degenerate
   wire. This single rule, applied uniformly,
   handles every case the build plan calls out: a wire to a third node, a pre-existing
   self-loop on one of the consumed nodes (both endpoints get remapped, yielding a self-loop
   on the merged node), and the consumed wire itself (dropped, never remapped). The two
   ordered boundary lists are rebuilt through this exact same ``_remap_endpoint`` helper, one
   entry at a time, in place, so position is preserved exactly and a boundary ref is held to
   the identical standard as a wire endpoint -- a ref on a node *not* being consumed passes
   through unchanged (that node survives, so the ref still names a live port), a ref on a
   *consumed* node must appear in ``port_mapping`` or this step raises. Boundaries and wires
   used to diverge here (an early version silently fell back to ``port_mapping.get(ref, ref)``
   for boundary entries, so a builder that forgot to map a surviving boundary port would not
   raise -- the ref would survive the rebuild unchanged, still naming a soon-to-be-removed
   node, and step 6's ``remove_node`` cascade would then delete it from the boundary with no
   exception, silently shrinking the returned diagram's boundary arity below the input's);
   that silent-drop path is now ruled out identically for both. ``_remap_endpoint``'s raise is
   also the failure mode a *consumed* (not surviving) port would hit if it were still named by
   a second wire or a boundary entry -- a builder never maps a consumed port, since it no
   longer exists once the match is applied. :mod:`qufzx.rewrite.match`'s ``find_matches``
   resolves this on its side: it rejects any candidate whose consumed port is claimed by more
   than one wire or is on a boundary list before ever returning it as a match (see that
   module's docstring, "Match-implies-applicable and multiply-claimed ports"), so this branch
   of ``_remap_endpoint`` is unreachable for any match ``find_matches`` actually returned -- it
   remains here only as a defensive check against a hand-built or foreign ``Match``, the same
   posture every other coverage check in this module and in
   :mod:`qufzx.rewrite.rules_library` takes toward inputs it did not itself produce.

   Match-implies-applicable gains a boundary-ref leg (Phase 5 post-closing audit round 18,
   Defect 2): the claim in the paragraph above -- and in ``qufzx.rewrite.match``'s own
   "Match-implies-applicable and multiply-claimed ports" section -- used to hold for a
   *malformed* reference (unknown node id, out-of-range index) only on the wire-endpoint
   side; a boundary entry naming no real port at all could still reach this function
   unexamined, surfacing here as ``RewriteDomainError`` rather than the
   ``RewriteGrammarError`` a malformed reference should raise. ``find_matches`` now validates
   every boundary entry the identical way it already validated every wire endpoint (see
   ``qufzx.rewrite.match``'s "Malformed boundary references"), so this function's own
   unreachability claim now covers both reference kinds, not only one.
6. Remove the consumed nodes. Only after every wire and boundary entry that referenced them
   has already been replaced. :meth:`~qufzx.diagram.graph.Diagram.remove_node`'s cascade
   (see that module's docstring) is therefore a no-op on wires and boundary entries at this
   point -- both node ids are attached to nothing else. Removing before step 5 would corrupt
   things: the cascade would silently drop wires to third nodes before they were remapped.
7. Multiply the scalar. ``working.multiply_scalar(build_result.scalar_introduced)``, after
   every structural change, so the returned diagram's scalar accumulator is exactly the
   input's times the rule's introduced factor.
8. Verify the rewrite is not a relative regression. :func:`~qufzx.diagram.validate.validate`
   runs on both the original ``diagram`` and the finished ``working``; if ``working`` carries
   a hard-failure issue this step cannot account for as already present in ``diagram``, this
   raises :class:`~qufzx.rewrite.rule.RewriteDomainError`. The comparison is a *multiset*
   over ``(kind, offending ref)`` pairs (via :func:`_issue_key`), not a set of bare
   :class:`~qufzx.diagram.validate.IssueKind`\\ s: a set comparison cannot see a second,
   independent issue of a kind the input already carried once (it would vanish into the same
   set element), and cannot see an issue the rewrite *removed* either, since set difference in
   the wrong direction hides exactly that -- both blind spots let a rewrite launder a
   pre-existing hard error into a *different* one of the same kind with nothing to show for
   it (see the "Dimension of the merged node" paragraph in
   :mod:`qufzx.rewrite.rules_library`'s module docstring for a worked example). Never
   compared by message (a node id embedded in an issue's message legitimately changes across
   a rewrite, e.g. because a merged node gets a fresh id) -- ``_issue_key`` reads the issue's
   actual offending reference (``port_ref``, ``wire``, or ``node_id``) instead.

   The input side of the comparison is not simply ``_issue_key`` of each of ``diagram``'s
   own issues, though: a reference anchored on a *consumed* node -- its node id, or any of
   its ports -- is guaranteed to differ from anything ``working`` could possibly carry,
   since that node id and those port indices are gone once the match is applied (the merged
   node gets a fresh id and fresh port indices). Comparing such a reference's raw
   ``_issue_key`` against ``working``'s post-rewrite keys would therefore *always* read as
   "introduced", regardless of whether the rewrite actually carried the underlying defect
   forward -- which is exactly the false-positive failure mode this step exists to avoid,
   not to cause (Phase 5 round-7 audit's Defect 1). :func:`_translate_input_issue_key`
   closes this: it maps each of ``diagram``'s hard-error issues into the coordinate space
   ``working`` actually uses, via ``build_result.port_mapping`` for a port on a consumed
   node (falling back to the original, now-nonexistent port when the port is the matched
   one itself and so absent from ``port_mapping`` -- deliberately left untranslated, since
   there is nothing to translate it to and it correctly then matches nothing on the result
   side) and via ``build_result.new_node_ids`` for a node id on a consumed node, but only
   when that tuple has exactly one entry (true for spider fusion's two-consumed-into-one
   shape; see that function's docstring for the fail-closed policy a future rule with a
   different consumed-to-new node cardinality would hit). Still relative to the input, never
   absolute -- a diagram that already carries a hard error (e.g. an unwired non-boundary leg)
   is legitimately rewritable, and this step must not block that; it only catches a rewrite
   that made things *worse*, now counted precisely, in the right coordinate space, rather
   than merely by kind. One check, independent of steps 5 and 6 getting their own bookkeeping
   right, standing in for the whole family of structural regressions a rewrite could
   otherwise introduce (a dropped wire, a shrunk boundary, a mixed-dimension leg, a lost
   dimension) instead of guarding each one individually.

   This step's multiset compare is over ``.errors`` (hard-failure issues) only, by design
   -- ``.deferred`` issues (:class:`~qufzx.diagram.validate.IssueKind.DIMENSION_DEFERRED`)
   are an assumed, non-hard-error diagram state, not a regression, so step 8 never blocks a
   rewrite for changing them. That leaves a related but distinct question step 8's own
   compare cannot answer: did a rewrite that fired across a deferred dimension assumption
   (see :mod:`qufzx.rewrite.rules_library`'s module docstring, "Phase 5 judgement call")
   silently make that assumption disappear from the diagram entirely, with nothing to show
   for it? Immediately after step 8's compare, the same translation machinery
   (:func:`_translate_input_issue_key`) is reused over ``validate(diagram).deferred`` and
   ``validate(working).deferred`` instead of ``.errors`` -- not to raise (a rewrite is
   allowed to resolve a deferred assumption; that is the entire point of firing across one),
   but to populate :attr:`RewriteStep.removed_deferred_issues` and, from the same
   ``Counter`` difference read the other way, :attr:`RewriteStep.introduced_deferred_issues`
   -- a rewrite can create a deferred assumption as readily as it resolves one (see
   :mod:`qufzx.rewrite.rules_library`'s module docstring, "Dimension of the merged node"),
   and the argument for recording removals applies equally to introductions, so both are
   populated from :func:`_select_by_key_surplus`, the one selection routine shared by both
   directions.

   This compare is a *multiset* over translated keys, mirroring step 8's own discipline
   rather than diverging from it: ``_translate_input_issue_key`` maps every consumed node's
   deferred issues onto the same single surviving node id
   (:func:`_translate_input_issue_key`'s one-new-node-id fallback, above), so two distinct,
   node-anchored deferred issues -- one on each of two fused spiders, an ordinary shape --
   legitimately translate to the *same* key; a plain dict-by-key comparison would silently
   drop one to last-write-wins, exactly the bug a ``Counter`` difference (as step 8 already
   uses) does not have. Each direction's surviving count per key,
   ``Counter(keyed) - Counter(other side's keyed)``, is satisfied by walking that side's own
   issues in their original order and taking the first that-many occurrences of each key, so
   every reported issue is an actual issue object in its own (pre- or post-rewrite,
   respectively) coordinates -- never a translated stand-in. When several issues collide on
   one key and only *some* of them have a counterpart, the choice of which to report is
   arbitrary but deterministic (first in that side's own order) and
   :attr:`RewriteStep.deferred_issue_identity_ambiguous` says so explicitly rather than
   leaving a reader to assume the reported one was chosen for a reason -- see that field's
   own docstring for the contract, and
   ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric`` /
   ``TestRemovedDeferredIssuesMultisetCompare`` for the tests pinning both directions and
   the collision behavior.
9. Record provenance. A :class:`RewriteStep` carrying the rule name, the located ``match``
   exactly as it was applied (stored verbatim, so Phase 6 replays directly from it rather
   than re-running the matcher and re-selecting a candidate by node id), the match location
   (``build_result.consumed_node_ids`` and ``consumed_wires``, *as they were in the input
   diagram* -- these are read from ``build_result``, not re-derived from ``working`` after
   mutation), every side condition checked with its outcome, every dimension constraint
   assumed, the scalar introduced, and the full old-port -> new-port remapping. The side
   condition outcomes and dimension constraints recorded are ``build_result
   .verified_side_condition_outcomes``/``verified_dimension_constraints`` whenever the
   builder supplied them (not ``None``), falling back to ``match``'s own same-named fields
   otherwise (Phase 5 post-closing audit, Defect 2 -- see
   :class:`~qufzx.rewrite.rule.BuildResult`'s docstring): a builder that independently
   re-derives these facts (e.g. :func:`~qufzx.rewrite.rules_library.spider_fusion_builder`,
   via :func:`~qufzx.rewrite.match.resolve_fusion_match`) hands back the ground truth it
   computed, not the match's own unaudited claim, so the certificate records what the
   rewrite actually assumed even for a foreign or hand-built match whose claimed fields a
   builder did not happen to check bit-for-bit before this fix existed. Phase 6's certificate
   module must be able to replay a rewrite from this record alone (look the rule up by name
   via :func:`~qufzx.rewrite.rules_library.lookup_rule`, re-apply it at the stored ``match``,
   and confirm the replay reproduces ``diagram`` and passes the oracle) -- these fields are
   shaped for that consumer. This module does not implement replay or verification itself;
   that is Phase 6's job.

What this module deliberately does not do. It does not search for matches (that is
:mod:`qufzx.rewrite.match`'s job -- callers pass an already-located ``Match`` in); it does
not choose which rule or which match to apply, or iterate to a fixpoint (that is Phase 11's
strategy layer); and, per ``CLAUDE.md``, it never contracts or evaluates a diagram
numerically -- nothing in this module imports :mod:`qufzx.semantics`.

Phase 5 audit round 18: two structural postconditions on step 5 as a whole, not one more
field-by-field check. Hardening 5 (``port_mapping`` injectivity) and Hardening 6 (the
wire-count postcondition) both close the same class of gap: step 5's own per-wire checks
(the unmapped-port raise, the single-wire collapse raise) each reason about *one* wire or
*one* port_mapping entry at a time, so neither can see a problem that only exists across
several of them jointly -- two different old ports both mapping to the same new port, or a
remapped wire silently aliasing an already-live one. The general lesson for a future
per-field or per-entry validation check added to this function: ask whether the property it
establishes is actually preserved in aggregate (injectivity, a conserved count, an invariant
that only makes sense summed or compared across every entry), not only for each entry
examined in isolation -- a check that is individually sound at every one of N points can
still admit a defect only visible from the outside once all N are considered together.
``spider_fusion_builder`` happens to satisfy both by construction (see :mod:`qufzx.rewrite
.rules_library`'s module docstring, "Merged leg-ordering convention"), which is exactly why
neither gap was ever observed in practice -- and exactly why a foreign or future builder
cannot be trusted to share that property merely because the one rule tested so far does.

The validation contract table (Phase 5 round-12 audit, B1).

Rounds 4, 6, 7, and 9 of the Phase 5 audit each closed one hole in what ``apply()`` and
``spider_fusion_builder`` check on an incoming :class:`~qufzx.rewrite.rule.Match` or an
outgoing :class:`~qufzx.rewrite.rule.BuildResult`, without ever writing down the complete
set of fields either one consumes. That let the same kind of hole (an untrusted field taken
on faith) recur field-by-field across rounds instead of being closed once, for every field,
in one pass. This table is that one pass: every field either function reads, who checks it,
at which step (numbered per the Algorithm section above), with which error class, and the
test that proves it. A row marked "not validated" says so explicitly and names the phase
that owns closing it, rather than leaving the gap to be rediscovered as a "defect" later.

Reference-kind coverage (Phase 5 post-closing audit round 18, Defect 2's class sweep). Every
reference kind :mod:`qufzx.rewrite` consumes, and who checks it before it can name nothing:
wire endpoints and boundary entries are both checked by :func:`~qufzx.rewrite.match
.find_matches` before a match is even constructed (via ``_validate_wire_endpoint``, held to
the identical standard for both kinds since round 18 -- see that module's docstring,
"Malformed wire references" / "Malformed boundary references"); ``port_mapping`` keys are
never validated for existence (row below, re-justified there); ``port_mapping`` values,
``consumed_node_ids``, and ``new_node_ids`` are all checked by ``apply`` itself, at step 4,
in the ``BuildResult`` table below. Boundary entries were the one reference kind missing
from this enumeration before round 18 -- held to a *different* (weaker) standard than wire
endpoints despite ``_remap_endpoint`` treating the two identically once a match reaches
``apply``; they are now structurally excluded at the same point and by the same mechanism
wire endpoints already were.

Generic ``Match`` protocol fields (consumed by ``apply`` itself, for any future rule):

======================================  =================================  ====  =================  ============================================================
Field                                   Validated by                       Step  Error class          Proof
======================================  =================================  ====  =================  ============================================================
``side_condition_outcomes``             ``check_side_condition_coverage``  1     RewriteDomainError   test_engine.py::TestApplyEnforcesSideConditionCoverage,
(coverage: exact name-set, no dupes,    (exact name-set coverage, no                                 test_engine.py::TestApplyRejectsAFailingMatch
all passed -- content is a              dupes, all passed)
separate question, see below)
``dimension_constraints``               Not validated by ``apply`` itself  9     n/a                  n/a (see below) -- but see the ``FusionMatch``-specific table:
                                         -- recorded into ``RewriteStep``,                             for spider_fusion specifically, its *content* is checked
                                         preferring ``build_result                                     against ``resolve_fusion_match``'s fresh derivation before
                                         .verified_dimension_constraints``                             ``apply`` ever sees it (Defect 2)
                                         when the builder supplied it
``all_side_conditions_passed``          Not read by ``apply`` at all --    n/a   n/a                  test_match.py (``all_side_conditions_passed`` is vacuously True
                                         ``check_side_condition_coverage``                             over an empty tuple; superseded by explicit coverage checking,
                                         is the real gate, precisely                                   never relied on as a gate)
                                         because this property alone
                                         cannot see a missing outcome
======================================  =================================  ====  =================  ============================================================

``apply`` itself still never checks a *generic* match's ``dimension_constraints`` against
anything -- it has no rule-independent way to re-derive what a future, unknown rule should
have assumed, so this stays a rule-local concern (as it is for spider_fusion, below) rather
than something the generic engine can enforce. What changed (Defect 2, Phase 5 post-closing
audit) is that ``apply`` no longer *automatically* records the match's own claimed
``side_condition_outcomes``/``dimension_constraints`` onto the certificate verbatim: it
prefers ``build_result.verified_side_condition_outcomes``/``verified_dimension_constraints``
whenever the builder supplied them (see :class:`~qufzx.rewrite.rule.BuildResult`'s
docstring), so a rule whose builder does independently verify these facts (spider_fusion
does) gets its ground truth recorded instead of the match's unaudited claim, without
``apply`` itself needing to know how that verification works. A future rule whose builder
never re-derives anything is unaffected -- ``None`` falls back to the pre-fix behavior
exactly. See ``FusionMatch.dimension_constraints``'s own docstring for the full account of
why this field exists at all.

``FusionMatch``-specific fields (consumed only by ``spider_fusion_builder``, which is
reachable directly and not only through ``apply`` -- so it cannot rely on ``apply`` having
already checked anything):

======================================  =================================  ====  =================  ============================================================
Field                                   Validated by                       Step  Error class          Proof
======================================  =================================  ====  =================  ============================================================
``isinstance(match, FusionMatch)``      ``spider_fusion_builder`` itself    B.1   RewriteGrammarError  test_rules_library.py::TestBuilderTypedErrors
                                                                                                        ::test_rejects_a_non_fusion_match
``side_condition_outcomes`` (coverage,  ``check_side_condition_coverage``   B.2   RewriteDomainError   test_rules_library.py::TestSideConditionCoverageEnforced
again, against ``spider_fusion          against ``spider_fusion_builder``
_builder``'s own declared side_         .side_conditions -- kept
conditions -- see A5 below)             identical to ``SPIDER_FUSION``
                                         .side_conditions by ``Rule``
                                         .__post_init__ itself (A5)
``a_id``, ``b_id``, ``wire``            ``resolve_fusion_match``, called    B.3   RewriteGrammarError  test_match.py::TestResolveFusionMatchIsTheSharedPredicate
(structural: distinct, present in       fresh against ``diagram`` --              (structural)         (distinct/absent/non-incident/not-an-element-of-
``diagram``, wire incident on both,     including, since the Phase 5                                   ``diagram.wires`` cases)
and an actual element of                post-closing audit, that ``wire``
``diagram.wires``)                      is actually in ``diagram.wires``,
                                         not merely incident-looking
                                         (Defect 1)
same_generator_type /                   ``resolve_fusion_match``, the       B.3   RewriteDomainError   test_rules_library.py::TestPhase5Round12AuditDefects
parallel_wires_become_self_loops /      exact function ``find_matches``           (domain)             ::test_a1_fabricated_same_generator_type_outcome_on_a_z_x_pair
consumed_wire_direction_permitted_      calls to decide a candidate in                                 _is_rejected
for_color / consumed_ports_singly_      the first place -- not a second,
claimed / dimension_agreement /         independently-maintained copy of
phase_dimension_agreement (the seven    this logic (A1). The fifth of these
side conditions, re-verified fresh,     (``consumed_ports_singly_claimed``)
not merely trusted from                 was promoted from a bare filter
``side_condition_outcomes``)            inside ``find_matches`` to a real,
                                         re-verified side condition here in
                                         Phase 5 post-closing audit round 23
                                         (Task 2) -- see
                                         ``qufzx.rewrite.match``'s module
                                         docstring, "Match-implies-
                                         applicable and multiply-claimed
                                         ports"
``shared_dim``                          Checked for exact agreement with    B.4   RewriteDomainError   test_rules_library.py::TestPhase5Round12AuditDefects
                                         ``resolve_fusion_match``'s own                                 ::test_a2_shared_dim_unrelated_to_the_matched_legs_is_rejected
                                         freshly-derived value (A2)
``bindings``                            Checked for exact agreement with    B.4   RewriteDomainError   test_rules_library.py::TestPhase5Round12AuditDefects
                                         ``resolve_fusion_match``'s own                                 ::test_a2_bindings_unrelated_to_the_matched_legs_is_rejected
                                         freshly-derived value (A2)
``dimension_constraints`` (content,     Checked for exact agreement with    B.5   RewriteDomainError   test_engine.py::TestCertificateRecordsTheReDerivedFacts
not merely coverage of its own          ``resolve_fusion_match``'s own
container field, which B.2 already      freshly-derived value; the
checks)                                 agreeing value is then returned
                                         as ``BuildResult
                                         .verified_dimension_constraints``
                                         for ``apply`` to record (Defect 2)
``side_condition_outcomes`` (content,   Checked for exact agreement with    B.5   RewriteDomainError   test_engine.py::TestCertificateRecordsTheReDerivedFacts
not merely coverage/passedness,         ``resolve_fusion_match``'s own
which B.2 already checks)               freshly-derived value; the
                                         agreeing value is then returned
                                         as ``BuildResult
                                         .verified_side_condition_outcomes``
                                         for ``apply`` to record (Defect 2)
======================================  =================================  ====  =================  ============================================================

``BuildResult`` fields (consumed by ``apply`` itself, for any future rule; step numbers per
the Algorithm section above):

======================================  =================================  ====  =================  ============================================================
Field                                   Validated by                       Step  Error class          Proof
======================================  =================================  ====  =================  ============================================================
``diagram``                             ``is working`` identity check       3     RewriteGrammarError  (no dedicated regression test; every other engine test would
                                                                                                        fail immediately if this check were removed, since a builder
                                                                                                        that substituted a different diagram would desync every
                                                                                                        later step from ``working``)
``scalar_introduced``                   Exact equality against              4     RewriteDomainError   test_engine.py::TestApplyRejectsAScalarMismatch
                                         ``rule.scalar_introduced``
``consumed_wires`` (membership)         Every entry must be present in       4     RewriteGrammarError  test_engine.py::TestApplyRejectsAForeignMatch
                                         ``working.wires``                                                    ::test_raises_when_the_matched_wire_is_absent_from_the_diagram
``consumed_wires`` (duplicates)         Not validated -- collapsed          n/a   n/a                  n/a (see below)
                                         harmlessly by ``frozenset``
``consumed_node_ids`` (membership)      Every entry must be present in       4     RewriteGrammarError  test_engine.py::TestApplyRejectsAForeignMatch
                                         ``working.nodes``                                                    ::test_raises_when_the_build_result_names_a_missing_node_id
``consumed_node_ids`` (duplicates)      No entry may repeat (A3)             4     RewriteGrammarError  test_engine.py::TestApplyWithAnIndependentlyScriptedBuilder
                                                                                                        ::test_a3_duplicate_consumed_node_ids_raises_rewrite_grammar_error
``new_node_ids`` (membership)           Every entry must be present in       4     RewriteGrammarError  test_engine.py::TestApplyRejectsAForeignMatch
                                         ``working.nodes``                                                    ::test_raises_when_new_node_ids_names_a_node_never_added
``new_node_ids`` (duplicates, Phase 5   No entry may repeat -- would         4     RewriteGrammarError  test_engine.py::TestApplyWithAnIndependentlyScriptedBuilder
post-closing audit round 19)            misreport how many new nodes                                         ::test_duplicate_new_node_ids_raises_rewrite_grammar_error
                                         a rewrite actually created to
                                         the certificate
``port_mapping`` (values)               Every value must name a real,        4     RewriteGrammarError  test_engine.py::TestApplyRejectsAForeignMatch
                                         in-range port in ``working``                                        ::test_raises_when_port_mapping_names_an_out_of_range_port
``port_mapping`` (keys, existence)      Not validated -- an extra key       n/a   n/a                  n/a (see below)
                                         naming a port nothing ever looks
                                         up is silently unused
``port_mapping`` (injectivity)          ``len(set(values)) ==                4     RewriteGrammarError  test_engine.py::TestApplyWithAnIndependentlyScriptedBuilder
(Hardening 5, Phase 5 post-closing      len(port_mapping)`` -- two old                                      ::test_hardening_5_non_injective_port_mapping_raises_
audit round 18)                         ports must never map to the                                         rewrite_grammar_error
                                         same new port
``port_mapping`` (covers every          Every wire/boundary endpoint on a    5     RewriteDomainError   test_engine.py::TestApplyRejectsAnUnmappedSurvivingPort,
consumed-node endpoint that             consumed node must appear as a                                 TestApplyRejectsAnUnmappedSurvivingBoundaryPort
survives) -- boundary refs, too,        key (``_remap_endpoint``); the
not only wire endpoints (Defect 2)      identical treatment now also
                                         applies before a match is even
                                         returned (see
                                         :mod:`qufzx.rewrite.match`'s own
                                         "Malformed boundary references")
``port_mapping`` (no collapse)          Both endpoints of a surviving wire   5     RewriteGrammarError  test_engine.py::TestApplyWithAnIndependentlyScriptedBuilder
within one wire                         must map to two distinct ports                                       ::test_collapsing_port_mapping_raises_rewrite_grammar_error
Wire-count postcondition (Hardening 6,  ``len(working.wires) ==              5     RewriteGrammarError  test_engine.py::TestApplyWithAnIndependentlyScriptedBuilder
Phase 5 post-closing audit round 18)    len(pre_wires) -                                                     ::test_hardening_6_wire_count_postcondition_catches_a_
                                         len(set(consumed_wires))``,                                         silently_lost_wire
                                         checked once after all of step 5's
                                         per-wire remapping -- catches a
                                         wire silently lost by any
                                         mechanism, not only the two this
                                         step's own per-wire checks and
                                         injectivity above already guard
``verified_side_condition_outcomes`` /  Not validated by ``apply`` itself   9     n/a                  test_engine.py::TestCertificateRecordsTheReDerivedFacts
``verified_dimension_constraints``      -- a builder that supplies these
(Defect 2)                              is trusted to have done its own
                                         checking (as spider_fusion_builder
                                         does, against B.5 above) before
                                         handing them back; ``apply`` only
                                         chooses whether to prefer them
                                         (not-``None``) over ``match``'s
                                         own fields when recording the step
``verified_phase_substitutions``        Not validated by ``apply`` itself   9     n/a                  test_rules_library.py::TestPhaseSubstitutionsCertificate
(F2, Phase 5 post-closing audit         -- same trust posture as the row
round 21)                               above; ``None`` falls back to an
                                         empty mapping (see
                                         ``RewriteStep.phase_substitutions``)
Overall structural non-regression       Multiset ``(kind, ref)`` comparison  8     RewriteDomainError   test_engine.py::TestStep8CatchesAnExtraIssueOfAnAlreadyPresentKind,
(not one field -- the combined          of ``validate(diagram)`` vs.                                   TestStep8DoesNotBlockAPreExistingIssueOnAConsumedNode
effect of every field above)            ``validate(working)``
Removed deferred issues (judgement      Not a gate -- multiset compare of    8     n/a (recorded,      test_engine.py::TestRemovedDeferredIssuesAreRecorded
call 1, Phase 5 post-closing audit)     ``.deferred`` issue keys, mirroring        never raised)
                                         step 8's own discipline; issues
                                         with no surviving counterpart are
                                         recorded onto ``RewriteStep
                                         .removed_deferred_issues``, never
                                         blocked -- see that field's
                                         docstring
======================================  =================================  ====  =================  ============================================================

``consumed_wires`` duplicates and unused ``port_mapping`` keys are the two rows left
deliberately open: a duplicate consumed wire is inert (``frozenset(build_result
.consumed_wires)`` is used purely for set membership in the removal loop, so a repeat has
no observable effect -- unlike a duplicate *node* id, which drives an imperative removal
loop and so does have one, see A3), and an extra, unused ``port_mapping`` key can never
corrupt a rewrite since nothing ever looks it up. Both are harmless-by-construction rather
than silently-wrong-by-construction, which is the distinction that makes leaving them
unchecked a deliberate choice rather than an oversight -- unlike A1/A2/A3, which were all
silently wrong.

Phase 5 post-closing audit round 20 summary. Two defect classes closed in this module (a
third, shared with :mod:`qufzx.rewrite.match`, is recorded in that module's own docstring):

* A documented contract drifted out of date against the code it describes, with nothing
  checking the two stayed in sync. ``apply``'s own docstring (the prose an actual caller
  reads, as opposed to the module-level validation contract table above, which stayed
  current) named only 4 of its 11 real raise conditions -- rounds 18 and 19 each added raise
  sites to the function body without updating it. Closed by rewriting the docstring complete
  and grouped by error class, each entry naming its step, and by adding
  ``test_engine.py::TestApplyDocstringMatchesRaiseSites``, which parses ``apply``'s own AST
  and pins the count of ``RewriteGrammarError``/``RewriteDomainError`` raise sites lexically
  inside its body -- a future raise site added without a docstring update now fails there,
  naming the offending line, rather than drifting a third time. The general lesson: a
  docstring is not "documentation that happens to describe the code" but a claim that needs
  the same kind of enforcement as any other invariant this module cares about, once it has
  drifted wrong twice.
* Redundant work standing in for a correctness property nobody had stated. ``validate(diagram)``
  and ``validate(working)`` were each called twice -- once for ``.errors`` (step 8's hard-error
  compare), again for ``.deferred`` (the removed/introduced-deferred-issue compare) -- making
  ``validate``, the single most expensive operation in ``apply``, run four times for two
  logical inputs. Hoisted to one call per diagram (``input_report``/``result_report``), which
  is also a correctness property and not merely a saving: the hard-error and deferred views
  are now guaranteed to read off the very same validation snapshot of each diagram, rather
  than two separately-timed (and, after a future concurrent-mutation change, potentially
  different) re-validations of what is nominally "the same" diagram.

Round 20, and round 21 again (a jointly-unsatisfiable ``ALL_LEGS_EQUAL`` leg set; a name
colliding between a dimension symbol and a phase parameter), each closed a
validator/denotation gap this module's own step 8 depends on -- not a defect in this
module, but recorded here because step 8 uses ``validate`` as its sole structural
postcondition on a rewritten diagram, so a gap in what "valid" means is a gap in what step
8 actually guarantees. See :mod:`qufzx.diagram.validate`'s module docstring for both fixes.

Phase 5 post-closing audit round 23: a load-bearing check that existed in
:mod:`qufzx.rewrite.match` but appeared nowhere in the certificate (a port claimed by a
second wire, or also on a boundary list, was rejected by a bare filter inside
``find_matches``, never a named, reported side condition). Not a defect in this module
either, but the validation-contract table above named "the six side conditions" and this
module's own :func:`_remap_endpoint` carried a defensive raise whose docstring described
that filter as the reason it was unreachable -- both now updated to the seven-condition,
one-decision-point picture: see :mod:`qufzx.rewrite.match`'s own module docstring,
"Match-implies-applicable and multiply-claimed ports", for the fix itself. The general
lesson, sharpened from round 20's "a documented contract drifted out of date": a
cross-module table naming another module's own invariants ("the six side conditions") is
exactly the kind of claim that drifts silently when that other module changes shape --
worth a search for every place this module names a specific count or list belonging to
another module's contract, not just its own, whenever that other module's shape changes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.graph import Diagram, NodeId, PortRef, Wire
from qufzx.diagram.validate import IssueKind, ValidationIssue, validate
from qufzx.rewrite.rule import (
    DimensionConstraint,
    Match,
    RewriteDomainError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
    check_side_condition_coverage,
)


@dataclass(frozen=True, slots=True)
class RewriteStep:
    """Structured provenance for one rewrite application. See the module docstring, step 9.

    Every field is drawn either from the ``Match`` that was applied or from the
    ``BuildResult`` its rule's builder produced, never re-derived from the mutated working
    diagram, so this record describes the rewrite *as it was applied to the input*.
    ``match`` is the field Phase 6 replays from: the exact ``Match`` object ``apply`` was
    given, stored verbatim rather than re-derived, so a replayer never needs to re-run the
    matcher and re-select a candidate by node id -- it can look ``rule_name`` up (via
    :func:`~qufzx.rewrite.rules_library.lookup_rule`) and re-apply it at this stored match
    directly.
    """

    rule_name: str
    match: Match
    consumed_node_ids: tuple[NodeId, ...]
    consumed_wires: tuple[Wire, ...]
    side_condition_outcomes: tuple[SideConditionOutcome, ...]
    dimension_constraints: tuple[DimensionConstraint, ...]
    """Every dimension equality this rewrite assumed rather than verified as a syntactic
    identity -- see :attr:`~qufzx.rewrite.rule.Match.dimension_constraints` for the recording
    contract itself (source-keyed, at most one entry per
    :class:`~qufzx.rewrite.rule.ConstraintSource`).

    Round 20, Task 11: a ``DEFERRED`` entry here is a recorded *assumption*, not a claim that
    the assumption is satisfiable on anything but a single degenerate point -- e.g. a
    surviving leg of dimension ``d**2`` forced onto a shared dimension ``d`` records ``d**2
    == d``, true only at ``d = 1``. The rewrite is sound relative to what it recorded (the
    oracle can confirm it at any substitution that actually satisfies the assumption), but
    Phase 5's placeholder :meth:`~qufzx.algebra.dimension.Dim.unify` cannot itself distinguish
    a constraint satisfiable broadly from one satisfiable only degenerately -- discharging
    that distinction is explicitly Phase 10's job (see ``Dim.unify``'s own docstring), not
    something this field or this phase's matcher fabricates a guess at. See
    :mod:`qufzx.rewrite.rules_library`'s module docstring, "Round 20, Task 11", for the fuller
    account and the worked example.
    """
    scalar_introduced: Scalar
    port_mapping: Mapping[PortRef, PortRef]
    new_node_ids: tuple[NodeId, ...]
    removed_deferred_issues: tuple[ValidationIssue, ...] = ()
    """Every :attr:`~qufzx.diagram.validate.ValidationReport.deferred` issue ``diagram``
    carried that a *multiset* compare (translated key, via
    :func:`_translate_input_issue_key`) says has no surviving counterpart among ``working``'s
    own deferred issues, in ``diagram``'s own (pre-rewrite) coordinates. Step 8 refuses to
    introduce a new hard-error issue kind, but it never looks at deferred issues at all -- a
    diagram-level unify assumption
    (:class:`~qufzx.diagram.validate.IssueKind.DIMENSION_DEFERRED`) a rewrite resolves (by
    consuming or overwriting the leg it was recorded against) would otherwise simply vanish,
    with nothing on the certificate to say a pre-existing assumption was ever there, let
    alone that this rewrite is the one that made it disappear. A non-empty value is not an
    error -- see :mod:`qufzx.rewrite.rules_library`'s module docstring, "Phase 5 judgement
    call", for why spider_fusion is allowed to fire on a ``DEFERRED`` dimension pair at all.
    Enforced by ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.

    The compare is multiset, not set/dict-keyed: two distinct input issues that translate to
    the same key -- e.g. one node-anchored ``DIMENSION_DEFERRED`` issue on each of two nodes
    a fusion consumes, both mapped onto the same surviving node id -- are each counted and,
    if both lack a surviving counterpart, both reported, never collapsed to one by a
    last-write-wins dict lookup.
    """

    introduced_deferred_issues: tuple[ValidationIssue, ...] = ()
    """The other direction of the same :class:`~collections.Counter` difference: every
    deferred issue ``working`` carries that has no counterpart among ``diagram``'s own
    (translated) deferred issues, in ``working``'s own post-rewrite coordinates.

    A rewrite can *create* a deferred assumption as readily as it resolves one -- forcing a
    surviving leg onto ``shared_dim`` can leave a neighbouring wire that was an exact match
    before merely deferred after (see :mod:`qufzx.rewrite.rules_library`'s module docstring,
    "Dimension of the merged node"). That is expected, not a defect; but the argument for
    recording removals -- that a silently-changed assumption is a loss of information the
    certificate should not paper over -- is direction-symmetric, so this field exists too.
    Neither field is a gate; both are certificate facts.
    Enforced by ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``.
    """

    phase_substitutions: Mapping[NodeId, Mapping[str, Dim]] = MappingProxyType({})
    """Per-node bindings the builder actually substituted into a phase's entries -- see
    :attr:`~qufzx.rewrite.rule.BuildResult.verified_phase_substitutions`. Empty when the
    builder supplied ``None`` (nothing re-derived) or genuinely substituted nothing.
    """

    deferred_issue_identity_ambiguous: bool = False
    """Whether the *identity* of the reported deferred issues above is meaningful, or only
    their count.

    The contract, stated rather than left to be inferred. Both fields are populated by
    walking the source report's issues in :func:`~qufzx.diagram.validate.validate` order and
    taking the first ``n`` occurrences of each translated key, where ``n`` is that key's
    Counter surplus. When each key's surplus equals its total occurrence count, that
    selection is forced and every reported issue is uniquely the one with no counterpart.
    When several issues collide on one translated key and only *some* of them lack a
    counterpart, which of the colliding issues to name is genuinely unrecoverable: the
    collision arises because :func:`_translate_input_issue_key` maps every consumed node's
    identity onto the one surviving merged node, and nothing in a
    :class:`~qufzx.diagram.validate.ValidationIssue` distinguishes two issues that agree on
    ``(kind, ref)`` beyond a message this comparison deliberately never reads (a message
    embeds node ids that legitimately change across a rewrite). In that case the selection
    is arbitrary but deterministic -- first in ``validate`` order -- the count is the
    meaningful part, and this flag is ``True`` so a reader is told so in the data rather
    than left to assume the named issue was chosen for a reason.
    Enforced by
    ``tests/test_engine.py::TestDeferredIssueProvenanceIsSymmetric``\\
    ``::test_colliding_keys_are_flagged_ambiguous_and_pinned_to_validate_order``.

    "First in ``validate`` order" is a value-and-order contract, and is now actually true
    across processes, not merely within one (Phase 5 post-closing audit round 18, Defect 1):
    :func:`~qufzx.diagram.validate.validate` used to iterate ``diagram.wires``, a frozenset
    whose hash (via ``Wire`` -> ``PortRef`` -> ``Direction``, an ``Enum`` hashed by member
    name) is ``PYTHONHASHSEED``-dependent, so which issue ``validate`` reported *first* --
    and therefore which of several colliding issues this field names when ambiguous -- could
    differ between two otherwise-identical runs in different processes, contradicting this
    docstring's own promise. Fixed by sorting every such iteration in ``validate.py`` (see
    that module's docstring); this field's selection is deterministic by *value and order*
    across processes now, proven (not merely asserted) by
    ``tests/test_engine.py::TestCrossProcessDeterminism``, which runs an identical rewrite in
    two subprocesses under two different ``PYTHONHASHSEED`` values and compares this field
    (among others) by ``repr()`` equality -- never by ``hash()``, which remains legitimately
    process-dependent for reasons unrelated to this fix (see :meth:`RewriteStep.__hash__`'s
    own docstring).
    """

    def __hash__(self) -> int:
        """Explicit, since the dataclass-generated one would raise on ``port_mapping``.

        ``@dataclass(frozen=True)`` with the default ``eq=True`` would otherwise generate a
        ``__hash__`` that hashes every field verbatim, including ``port_mapping`` -- a
        :class:`~types.MappingProxyType`, which is unhashable (its backing ``dict`` is
        mutable even though the proxy itself is read-only). Defining ``__hash__`` here
        explicitly, in the class body, makes ``dataclass`` leave it alone rather than
        overwrite it with the broken auto-generated one. Every other field is hashed as-is;
        ``port_mapping`` is hashed as ``frozenset(port_mapping.items())`` -- order-independent,
        matching the dataclass-generated ``__eq__``, which compares ``port_mapping`` via
        plain mapping equality (also order-independent) -- so ``a == b`` still implies
        ``hash(a) == hash(b)``, the contract Phase 12's cache (which will key on
        ``RewriteStep``) needs.

        Cross-process disclaimer, stated explicitly (Phase 5 post-closing audit round 18,
        Defect 1 -- corrected from a prior version of this docstring, which cited Phase 12's
        cache-keying need without qualification and thereby over-promised): the
        ``a == b`` implies ``hash(a) == hash(b)`` contract above holds *within one process*,
        but ``hash()`` itself is not guaranteed stable *across* processes, and this fix does
        not and cannot change that. ``IssueKind`` and ``Direction`` (reached transitively
        through ``consumed_wires``, ``port_mapping``, and the deferred-issue tuples, all of
        which embed ``PortRef``/``Wire``/``ValidationIssue`` values) are ``enum.Enum``
        subclasses hashed by member *name*, and Python randomizes string hashing per process
        by default (``PYTHONHASHSEED``) -- so ``hash(step)`` computed for equal ``RewriteStep``
        values in two different processes may legitimately differ, permanently, for reasons
        that have nothing to do with this class's own hash implementation. What round 18 did
        fix is a different property entirely: the *value and order* of every field this hash
        is computed over (in particular ``removed_deferred_issues``/
        ``introduced_deferred_issues``/``deferred_issue_identity_ambiguous``, populated from
        :mod:`qufzx.diagram.validate`'s issue order) is now itself deterministic across
        processes -- see that module's docstring and ``tests/test_engine.py
        ::TestCrossProcessDeterminism``, which compares field values and order directly, by
        ``repr()``, never by ``hash()``.
        """
        return hash(
            (
                self.rule_name,
                self.match,
                self.consumed_node_ids,
                self.consumed_wires,
                self.side_condition_outcomes,
                self.dimension_constraints,
                self.scalar_introduced,
                frozenset(self.port_mapping.items()),
                self.new_node_ids,
                self.removed_deferred_issues,
                self.introduced_deferred_issues,
                frozenset(
                    (node_id, frozenset(subs.items()))
                    for node_id, subs in self.phase_substitutions.items()
                ),
                self.deferred_issue_identity_ambiguous,
            )
        )


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """The outcome of :func:`apply`: the new diagram, its new node(s), and the step provenance."""

    diagram: Diagram
    new_node_ids: tuple[NodeId, ...]
    step: RewriteStep


def _issue_key(issue: ValidationIssue) -> tuple[IssueKind, object]:
    """A ``(kind, offending ref)`` key identifying one hard-error issue for step 8's compare.

    Never the issue's ``message``: a node id embedded in a message legitimately changes
    across a rewrite (a merged node gets a fresh id), so comparing messages would flag
    every rewrite as introducing a "new" issue merely because its wording mentions a
    different id, even when the underlying defect is the pre-existing one carried over
    unchanged. ``port_ref``, ``wire``, and ``node_id`` are checked in that order -- per
    :class:`~qufzx.diagram.validate.ValidationIssue`'s own docstring, at most one is set
    for a given issue in practice, so the order only matters as a deterministic tie-break.
    """
    ref: object = issue.port_ref
    if ref is None:
        ref = issue.wire
    if ref is None:
        ref = issue.node_id
    return (issue.kind, ref)


def _translate_input_issue_key(
    issue: ValidationIssue,
    consumed_node_ids: frozenset[NodeId],
    port_mapping: Mapping[PortRef, PortRef],
    new_node_ids: tuple[NodeId, ...],
) -> tuple[IssueKind, object]:
    """``_issue_key`` of an *input*-diagram issue, translated into post-rewrite coordinates.

    Step 8 must compare like with like: ``result_hard_counts`` is already keyed on
    references as they exist in ``working`` (the post-rewrite diagram), but a naive
    ``_issue_key`` of an input-diagram issue is keyed on references as they existed
    *before* the rewrite. A consumed node's ports and node id are gone from ``working``
    entirely -- the merged node gets a fresh :class:`NodeId` and fresh port indices -- so
    an issue anchored on one of them would silently never match its post-rewrite
    counterpart even when the rewrite carried it over faithfully, making step 8 flag a
    rewrite that introduced nothing (see the module docstring, step 8).

    A ``port_ref`` or a ``wire`` endpoint on a node *not* being consumed passes through
    unchanged -- that node survives with the same id and the same port indices. A
    ``port_ref`` or wire endpoint on a *consumed* node is translated via ``port_mapping``
    when present there (a surviving leg the builder remapped); a consumed port that is
    *not* in ``port_mapping`` is the matched port itself (or a foreign match's malformed
    port) -- it has no post-rewrite counterpart at all, so it is left unchanged, which
    correctly makes it match nothing in ``result_hard_counts`` (the issue did not carry
    over because the port it was anchored on no longer exists) rather than being dropped
    silently.

    A ``node_id`` on a consumed node is translated to the sole entry of ``new_node_ids``
    when there is exactly one -- true for every Phase 5 rule (spider fusion always merges
    its two consumed nodes into exactly one new node) -- since that is the only node the
    identity could plausibly have carried over to. When ``new_node_ids`` has zero or more
    than one entries, which new node (if any) a consumed node's identity maps to is
    genuinely undecidable from the information ``apply`` has, so the id is left unchanged
    rather than guessed at: this deliberately makes the translated key impossible to match
    against anything in ``result_hard_counts`` (no node in ``working`` carries the
    original, now-removed id), so a node-id-anchored input issue on a consumed node is
    conservatively treated as *not* carried over whenever the mapping is ambiguous -- the
    same fail-closed posture step 8 already takes toward every other unrecognised case,
    at the cost (for a future multi-new-node rule only; no Phase 5 rule triggers this) of
    occasionally blocking a rewrite that did carry the issue over faithfully.
    """

    def _translate_ref(ref: PortRef) -> PortRef:
        if ref.node_id not in consumed_node_ids:
            return ref
        return port_mapping.get(ref, ref)

    if issue.port_ref is not None:
        return (issue.kind, _translate_ref(issue.port_ref))
    if issue.wire is not None:
        translated_a = _translate_ref(issue.wire.a)
        translated_b = _translate_ref(issue.wire.b)
        if translated_a == translated_b:
            # Step 5 now rejects any collapsing remap of a live wire, so this branch is
            # reachable only for an input-issue wire listed in consumed_wires (dropped,
            # never spliced) whose endpoints a foreign builder mapped anyway; falling back
            # to the untranslated wire is the fail-closed choice (it then matches nothing
            # on the result side).
            return (issue.kind, issue.wire)
        return (issue.kind, Wire(translated_a, translated_b))
    if issue.node_id is not None:
        node_id = issue.node_id
        if node_id in consumed_node_ids and len(new_node_ids) == 1:
            node_id = new_node_ids[0]
        return (issue.kind, node_id)
    return (issue.kind, None)


def _select_by_key_surplus(
    keyed_issues: tuple[tuple[tuple[IssueKind, object], ValidationIssue], ...],
    surplus: Counter[tuple[IssueKind, object]],
) -> tuple[tuple[ValidationIssue, ...], bool]:
    """Take ``surplus[key]`` issues per key from ``keyed_issues``, in the order given.

    The one selection routine behind both :attr:`RewriteStep.removed_deferred_issues` (input
    issues keyed into post-rewrite coordinates, surplus = input - result) and
    :attr:`RewriteStep.introduced_deferred_issues` (result issues in their own coordinates,
    surplus = result - input) -- the two directions of the same
    :class:`~collections.Counter` difference, so they are computed by the same code rather
    than by two similar-looking loops.

    Returns the selected issues (always the actual issue objects from ``keyed_issues``,
    never translated stand-ins) and whether the selection was *ambiguous* for any key: True
    iff some key's surplus is non-zero but strictly smaller than how many issues carry that
    key, i.e. several interchangeable issues collided and only some of them are being
    reported. See :attr:`RewriteStep.deferred_issue_identity_ambiguous` for the contract
    that flag states.
    """
    remaining = dict(surplus)
    totals = Counter(key for key, _issue in keyed_issues)
    selected: list[ValidationIssue] = []
    for key, issue in keyed_issues:
        if remaining.get(key, 0) > 0:
            selected.append(issue)
            remaining[key] -= 1
    ambiguous = any(count > 0 and count < totals[key] for key, count in surplus.items())
    return tuple(selected), ambiguous


def _remap_endpoint(
    ref: PortRef,
    consumed_node_ids: frozenset[NodeId],
    port_mapping: Mapping[PortRef, PortRef],
    rule_name: str,
) -> PortRef:
    """Remap a wire endpoint, raising if a consumed node's port was left unmapped.

    An endpoint on a node *not* being consumed is passed through unchanged -- that fallback
    is correct and required. An endpoint on a *consumed* node must appear in ``port_mapping``;
    if it does not, leaving the fallback in place would silently point the wire at a node
    step 6 is about to remove (whose removal cascade then silently drops the wire) instead of
    surfacing the problem. This also fires for a *consumed* port that a second wire or a
    boundary entry still names alongside the matched wire that consumed it -- a builder never
    maps a consumed port. Phase 5 post-closing audit round 23, Task 2: this used to be
    guarded only by a bare ``continue`` filter inside :func:`~qufzx.rewrite.match.find_matches`
    (a check that existed in code but appeared nowhere in the certificate); it is now
    ``consumed_ports_singly_claimed``, a real, re-verified side condition
    :func:`~qufzx.rewrite.match.resolve_fusion_match` decides and
    :func:`~qufzx.rewrite.rules_library.spider_fusion_builder` re-checks before ever calling
    this function -- so this branch is unreachable not merely for any match ``find_matches``
    produces, but for any match the builder itself accepts at all, via either path (see
    :mod:`qufzx.rewrite.match`'s module docstring, "Match-implies-applicable and
    multiply-claimed ports"). It remains only as a defensive check against a hand-built or
    foreign ``Match`` that bypasses the builder's own re-verification -- which is not
    possible through ``spider_fusion_builder`` itself, but this function has no way to know
    it was called from there rather than some future caller that skips that step.
    """
    if ref.node_id not in consumed_node_ids:
        return ref
    if ref not in port_mapping:
        raise RewriteDomainError(
            f"rule {rule_name!r}: port {ref!r} is on a consumed node but is absent from "
            f"the builder's port_mapping; either the builder forgot to map a surviving "
            f"port, or {ref!r} names a port the match's own consumed wire already claimed "
            f"while the input diagram also listed it on a boundary"
        )
    return port_mapping[ref]


def apply(diagram: Diagram, rule: Rule, match: Match) -> RewriteResult:
    """Apply ``rule`` at ``match`` against ``diagram``, returning a new diagram and provenance.

    Never mutates ``diagram`` -- see the module docstring for the full algorithm (step
    numbers below cross-reference that Algorithm section). This list was found, round 20, to
    have drifted badly out of date against the actual raise sites (it named 4 of the 11 real
    ones); it is now complete, grouped by error class, and each entry names its step so the
    two can be cross-checked by eye. A meta-test,
    ``test_engine.py::TestApplyDocstringMatchesRaiseSites``, parses this function's AST and
    pins the count of ``raise RewriteGrammarError(...)``/``raise RewriteDomainError(...)``
    statements lexically inside ``apply``'s own body, so a future raise site added here
    without a matching docstring update fails loudly instead of drifting again.

    Raises :class:`~qufzx.rewrite.rule.RewriteDomainError`:

    * Step 1: ``match``'s ``side_condition_outcomes`` do not exactly cover
      ``rule.side_conditions``, or include a failed one
      (:func:`~qufzx.rewrite.rule.check_side_condition_coverage`).
    * Step 4: the builder's ``scalar_introduced`` disagrees with ``rule.scalar_introduced``.
    * Step 5 (raised by :func:`_remap_endpoint`, called from this step; not a literal raise
      site inside ``apply``'s own body, so not counted by the meta-test above): a wire or
      boundary entry names a port on a consumed node that is absent from
      ``build_result.port_mapping`` -- an unmapped surviving port.
    * Step 8: the result carries a hard-failure validation issue kind ``diagram`` did not
      already carry, relative to the input (the multiset compare over
      :func:`_translate_input_issue_key`/:func:`_issue_key`).

    Raises :class:`~qufzx.rewrite.rule.RewriteGrammarError`:

    * Step 3: the builder returned a ``BuildResult.diagram`` that is not (by identity) the
      working diagram it was given.
    * Step 4: a consumed wire or consumed node the builder reported is not actually present
      in the working diagram; ``build_result.consumed_node_ids`` names the same node id more
      than once (Phase 5 round-12 audit, A3); ``build_result.new_node_ids`` names a node id
      absent from the working diagram, or a ``port_mapping`` value names no real port
      (existence/in-range checks on both fields, closed together since round 19's Task 4
      sweep of the same reference-kind class); ``build_result.new_node_ids`` names the same
      node id more than once (round 19); ``build_result.port_mapping`` is not injective --
      two or more old ports map to the same new port (Hardening 5, round 18).
    * Step 5: ``port_mapping`` collapses both endpoints of one surviving wire onto a single
      port; or, as a postcondition on step 5 as a whole rather than any one wire, the working
      diagram's wire count after remapping does not equal the expected count (Hardening 6,
      round 18).

    Builder-output validation beyond these checks is deliberately still deferred to Phase 11,
    when a second rule gives the generic ``BuildResult`` contract a real second consumer to
    validate against. As of round 20, what remains deferred (re-scoped from an earlier,
    now-stale version of this sentence that named checks round 18-19 already closed) is: a
    builder that mutates ``working``'s wires, boundaries, or scalar directly instead of
    through the returned ``BuildResult`` fields (undetectable from ``BuildResult`` alone); a
    ``new_node_ids`` entry naming a node that pre-existed the builder's call rather than one
    it created (existence is checked; freshness is not); a duplicate entry in
    ``consumed_wires`` (inert today -- ``working_wire_set`` membership and the frozenset
    dedup in step 5 make a repeat harmless -- but not rejected as malformed); and an unused
    key in ``port_mapping`` naming a port no surviving wire or boundary entry actually reads
    (harmless today, since only values are consumed, but not rejected as extraneous input).
    """
    check_side_condition_coverage(match, rule.side_conditions, rule.name)

    working = diagram.copy()
    build_result = rule.builder(working, match)

    if build_result.diagram is not working:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder returned a BuildResult.diagram that is not the "
            "working diagram it was given; a builder must mutate and return that same "
            "object, never substitute a different one (see BuildResult's docstring)"
        )

    if build_result.scalar_introduced != rule.scalar_introduced:
        raise RewriteDomainError(
            f"rule {rule.name!r} declares scalar_introduced={rule.scalar_introduced!r}, "
            f"but its builder returned {build_result.scalar_introduced!r} for this match"
        )

    # Snapshotted once, as a set, rather than testing membership against ``working.wires``
    # (whatever collection backs that property) once per consumed wire -- the latter
    # re-materialises the full wire collection on every ``in`` test, making this check
    # quadratic in the number of consumed wires times the diagram's wire count.
    working_wire_set = frozenset(working.wires)
    missing_wires = [wire for wire in build_result.consumed_wires if wire not in working_wire_set]
    missing_node_ids = [
        node_id for node_id in build_result.consumed_node_ids if node_id not in working.nodes
    ]
    if missing_wires or missing_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: match does not belong to the diagram it is applied to "
            f"(consumed wire(s) absent: {missing_wires!r}; consumed node id(s) absent: "
            f"{missing_node_ids!r})"
        )

    # A3 (Phase 5 round-12 audit): a repeated entry in consumed_node_ids passes the
    # membership check above (every entry, including the repeat, is a real node id) but
    # would make step 6's ``working.remove_node(node_id)`` loop call ``remove_node`` twice
    # on the same, by-then-already-removed id -- raising
    # ``qufzx.diagram.graph.GraphGrammarError``, a different module's exception, escaping
    # this function's declared ``RewriteError`` hierarchy entirely (``apply``'s own
    # docstring promises only ``RewriteDomainError``/``RewriteGrammarError``). A duplicate
    # is a malformed request -- the same node cannot legitimately be consumed twice by one
    # match -- so it is rejected here, at the same point every other ``BuildResult`` field
    # is validated, rather than left to surface as a foreign error class deep in step 6.
    duplicate_node_ids = [
        node_id
        for node_id, count in Counter(build_result.consumed_node_ids).items()
        if count > 1
    ]
    if duplicate_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: build_result.consumed_node_ids names the same node id "
            f"more than once: {sorted(duplicate_node_ids)!r} -- a match cannot legitimately "
            "consume the same node twice"
        )

    # Every id the builder claims to have created must actually exist in ``working``, and
    # every port_mapping *value* (a builder-reported "new" port) must name a real port on a
    # node that is actually there -- these two fields are otherwise taken on faith: step 5
    # below feeds port_mapping values straight into every surviving wire and boundary entry
    # without ever checking they name anything real, and step 9's ``RewriteStep`` publishes
    # ``new_node_ids`` verbatim for Phase 6's certificate to replay against. An unvalidated
    # builder bug here (an id that was never added, or a port_mapping value with a stale or
    # out-of-range index) would otherwise surface much later as a confusing KeyError/mismatch
    # deep in remapping or certificate replay, or -- if the corrupted ref happens to alias a
    # real port by coincidence -- not surface at all, silently splicing a wire onto the wrong
    # port. Checked the same way every other ``BuildResult`` field already is in this
    # function: fail fast, close to the builder that produced the bad value.
    missing_new_node_ids = tuple(
        node_id for node_id in build_result.new_node_ids if node_id not in working.nodes
    )
    invalid_port_mapping_values = tuple(
        ref
        for ref in build_result.port_mapping.values()
        if ref.node_id not in working.nodes
        or ref.index >= len(working.nodes[ref.node_id].legs(ref.direction))
    )
    if missing_new_node_ids or invalid_port_mapping_values:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder-reported BuildResult fields do not name real "
            f"nodes/ports in the working diagram (new_node_ids absent: "
            f"{missing_new_node_ids!r}; port_mapping value(s) naming no real port: "
            f"{invalid_port_mapping_values!r})"
        )

    # Class 2 sweep (Phase 5 post-closing audit round 19, Task 4): consumed_node_ids gets a
    # duplicate check (A3, above) because a repeat there drives step 6's imperative removal
    # loop into a crash. new_node_ids drives no imperative loop, so a repeat cannot crash
    # apply() the same way -- but it is a structurally identical reference kind (a tuple of
    # NodeId the builder reports about this one rewrite) left with no duplicate check of its
    # own before this fix, and no entry in this module's own BuildResult-field table (see
    # the module docstring) recording that as a deliberate choice the way consumed_wires
    # duplicates and unused port_mapping keys are. It is not harmless: step 9 publishes
    # new_node_ids verbatim for Phase 6's certificate to replay against, and a duplicate
    # entry would misreport how many new nodes a rewrite actually created.
    duplicate_new_node_ids = [
        node_id for node_id, count in Counter(build_result.new_node_ids).items() if count > 1
    ]
    if duplicate_new_node_ids:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: build_result.new_node_ids names the same node id more "
            f"than once: {sorted(duplicate_new_node_ids)!r} -- a builder creates each new "
            "node once; a repeated id here would misreport how many new nodes exist to "
            "the certificate"
        )

    # Hardening 5 (Phase 5 post-closing audit round 18): port_mapping's injectivity is
    # load-bearing but was previously unchecked. Step 5 below rejects a *single* surviving
    # wire whose two endpoints collapse onto one port (new_a == new_b), but says nothing
    # about two *different* surviving wires whose four endpoints map onto only two distinct
    # ports between them -- e.g. port_mapping sending both old_x and old_y to the same
    # new_z. Diagram._wires is a Python set (Wire equality/hash is order-independent), so
    # two such remapped wires that happen to produce the identical Wire object would
    # silently collapse into one entry in that set at ``working.add_wire`` time in step 5,
    # losing a wire with no exception anywhere -- step 5's own per-wire check cannot see
    # this, since each wire is remapped and re-added independently, one at a time.
    # ``spider_fusion_builder`` is injective by construction (each surviving old port maps
    # to exactly one new port, and ``_surviving_legs`` never emits the same old port twice),
    # so this never fires for Phase 5's one registered rule -- but a foreign or future
    # builder need not be, and the engine's generic contract in this module's own docstring
    # must not rely on every builder happening to get this right unchecked.
    if len(set(build_result.port_mapping.values())) != len(build_result.port_mapping):
        raise RewriteGrammarError(
            f"rule {rule.name!r}: builder's port_mapping is not injective -- two or more "
            "old ports map to the same new port, which would silently collapse two "
            "distinct surviving wires into one once remapped"
        )

    consumed_wire_set = frozenset(build_result.consumed_wires)
    consumed_node_ids = frozenset(build_result.consumed_node_ids)
    port_mapping = build_result.port_mapping

    # Sorted, not the raw frozenset (Phase 5 post-closing audit round 18, Defect 1):
    # ``working.wires`` is a frozenset whose hash (via Wire -> PortRef -> Direction) is
    # PYTHONHASHSEED-dependent. This loop can raise (a collapsing port_mapping, or an
    # unmapped surviving port via ``_remap_endpoint``); with more than one such wire in a
    # diagram, an unsorted iteration would surface a different one -- a different exception
    # message -- across processes. The final *set* of wires this loop leaves in ``working``
    # is unaffected by order (each wire's own fate is independent of every other's), but the
    # order in which a possible failure is reported is not, so it is sorted here too.
    for wire in sorted(working.wires, key=lambda w: w.sort_key()):
        if wire in consumed_wire_set:
            working.remove_wire(wire.a, wire.b)
            continue
        touches_consumed = (
            wire.a.node_id in consumed_node_ids or wire.b.node_id in consumed_node_ids
        )
        if not touches_consumed:
            continue
        new_a = _remap_endpoint(wire.a, consumed_node_ids, port_mapping, rule.name)
        new_b = _remap_endpoint(wire.b, consumed_node_ids, port_mapping, rule.name)
        if new_a == new_b:
            raise RewriteGrammarError(
                f"rule {rule.name!r}: port_mapping collapses wire {wire!r} onto a "
                f"single port {new_a!r}; a builder must map a surviving wire's two "
                f"endpoints to two distinct ports"
            )
        working.remove_wire(wire.a, wire.b)
        working.add_wire(new_a, new_b)

    # Hardening 6 (Phase 5 post-closing audit round 18): a cheap structural postcondition
    # on step 5 as a whole, the same posture step 8 already takes toward the *result*
    # diagram's validation issues -- catch a silently-lost wire from any cause, not just
    # the one path (a collapsing port_mapping) step 5's own per-wire check already guards.
    # Every wire in ``working`` either survived untouched (not touching a consumed node),
    # was dropped as one of ``consumed_wires`` (absorbed into the merged node, never
    # replaced), or was removed and re-added exactly once (remapped) -- so the wire count
    # can only ever shrink by exactly the number of *distinct* consumed wires, never more,
    # never less. ``working_wire_set`` is the pre-step-5 snapshot (taken above, right after
    # the builder ran and before any of steps 4-5's own mutations); distinct, not a bare
    # ``len(build_result.consumed_wires)``, since a duplicate entry there is inert (see the
    # validation contract table's own note on this) and must not be double-counted here.
    expected_wire_count = len(working_wire_set) - len(consumed_wire_set)
    actual_wire_count = len(working.wires)
    if actual_wire_count != expected_wire_count:
        raise RewriteGrammarError(
            f"rule {rule.name!r}: wire-count postcondition violated after remapping -- "
            f"expected {expected_wire_count} wire(s) ({len(working_wire_set)} before this "
            f"rewrite minus {len(consumed_wire_set)} distinct consumed), got "
            f"{actual_wire_count}; a wire was silently lost or gained during remapping"
        )

    working.set_boundary_inputs(
        tuple(
            _remap_endpoint(ref, consumed_node_ids, port_mapping, rule.name)
            for ref in working.boundary_inputs
        )
    )
    working.set_boundary_outputs(
        tuple(
            _remap_endpoint(ref, consumed_node_ids, port_mapping, rule.name)
            for ref in working.boundary_outputs
        )
    )

    for node_id in build_result.consumed_node_ids:
        working.remove_node(node_id)

    working.multiply_scalar(build_result.scalar_introduced)

    # Round 20, Task 8: validate(diagram) and validate(working) each used to be called twice
    # -- once for .errors here, again for .deferred below -- making validate, the single most
    # expensive operation in apply, run four times total for two logical inputs. Hoisted to
    # one call per diagram; .errors/.deferred are cheap tuple filters over the same
    # ValidationReport. Also a correctness property, not only a saving: the hard-error and
    # deferred views below are now guaranteed to read off the very same validation snapshot
    # of each diagram, so they can never be computed against two differently-timed
    # (im)possible re-validations of what is nominally "the same" diagram.
    input_report = validate(diagram)
    result_report = validate(working)

    input_hard_counts = Counter(
        _translate_input_issue_key(
            issue, consumed_node_ids, port_mapping, build_result.new_node_ids
        )
        for issue in input_report.errors
    )
    result_hard_counts = Counter(_issue_key(issue) for issue in result_report.errors)
    introduced_counts = result_hard_counts - input_hard_counts
    if introduced_counts:
        # Name the actual offending (kind, ref) pairs, not merely the set of kinds -- a bare
        # kind (e.g. "dimension_policy_violation") does not say *where*, which is exactly
        # the information a user hitting this needs to find the offending node/port/wire in
        # ``working``. Sorted by ``(kind.value, repr(ref))`` for determinism, since ``ref``
        # is a heterogeneous mix of ``PortRef | Wire | NodeId | None`` with no natural order
        # of its own.
        offending = sorted(
            ((kind.value, ref, count) for (kind, ref), count in introduced_counts.items()),
            key=lambda item: (item[0], repr(item[1])),
        )
        detail = "; ".join(
            f"{kind} at {ref!r}" + (f" (x{count})" if count > 1 else "")
            for kind, ref, count in offending
        )
        raise RewriteDomainError(
            f"rule {rule.name!r}: rewrite introduced hard-error issue kind(s) not present "
            f"in the input diagram: {detail}"
        )

    # Judgement call 1 (Phase 5 post-closing audit): a rewrite is allowed to fire across a
    # DEFERRED dimension pair (see rules_library.py's module docstring, "Phase 5 judgement
    # call") -- this is not new here, and step 8 above does not change it. What step 8 never
    # did is notice when firing silently drops the resulting deferred assumption from the
    # diagram entirely (e.g. a d*e leg consumed by fusion, or overwritten onto shared_dim):
    # not a hard-error regression (deferred issues are explicitly outside step 8's multiset
    # compare, by design), but a loss of information the certificate should not paper over
    # in silence. Same translation machinery as the hard-error compare, reused for the
    # deferred set instead.
    input_deferred_keyed = tuple(
        (
            _translate_input_issue_key(
                issue, consumed_node_ids, port_mapping, build_result.new_node_ids
            ),
            issue,
        )
        for issue in input_report.deferred
    )
    result_deferred_keyed = tuple(
        (_issue_key(issue), issue) for issue in result_report.deferred
    )
    input_deferred_key_counts = Counter(key for key, _issue in input_deferred_keyed)
    result_deferred_key_counts = Counter(key for key, _issue in result_deferred_keyed)
    removed_deferred_issues, removed_ambiguous = _select_by_key_surplus(
        input_deferred_keyed, input_deferred_key_counts - result_deferred_key_counts
    )
    introduced_deferred_issues, introduced_ambiguous = _select_by_key_surplus(
        result_deferred_keyed, result_deferred_key_counts - input_deferred_key_counts
    )

    # Defect 2 (Phase 5 post-closing audit): prefer the builder's independently re-derived
    # facts over the match's own claims whenever the builder supplied them -- see
    # BuildResult's docstring. A builder that never re-verifies anything (leaves these
    # fields None) falls back to match's own fields, the pre-fix behavior, unchanged.
    step_side_condition_outcomes = (
        build_result.verified_side_condition_outcomes
        if build_result.verified_side_condition_outcomes is not None
        else match.side_condition_outcomes
    )
    step_dimension_constraints = (
        build_result.verified_dimension_constraints
        if build_result.verified_dimension_constraints is not None
        else match.dimension_constraints
    )
    step_phase_substitutions = (
        build_result.verified_phase_substitutions
        if build_result.verified_phase_substitutions is not None
        else MappingProxyType({})
    )

    step = RewriteStep(
        rule_name=rule.name,
        match=match,
        consumed_node_ids=build_result.consumed_node_ids,
        consumed_wires=build_result.consumed_wires,
        side_condition_outcomes=step_side_condition_outcomes,
        dimension_constraints=step_dimension_constraints,
        scalar_introduced=build_result.scalar_introduced,
        port_mapping=MappingProxyType(dict(port_mapping)),
        new_node_ids=build_result.new_node_ids,
        removed_deferred_issues=removed_deferred_issues,
        introduced_deferred_issues=introduced_deferred_issues,
        phase_substitutions=step_phase_substitutions,
        deferred_issue_identity_ambiguous=removed_ambiguous or introduced_ambiguous,
    )
    return RewriteResult(diagram=working, new_node_ids=build_result.new_node_ids, step=step)
