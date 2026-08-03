"""Deterministic randomized property harness over spider fusion.

Generates small random diagrams (fixed seed list, ``random.Random(seed)``, never
unseeded), applies every fusion match :func:`~qufzx.rewrite.match.find_matches` reports
against each, and checks three properties: no unexpected exception escapes, the
relative-validity post-condition :func:`~qufzx.rewrite.engine.apply` itself enforces
(re-derived independently here, not merely trusted), and oracle equality at concrete
substitutions via :mod:`qufzx.semantics.check`. This is the harness the Phase 5 audit's
manual fix rounds are meant to replace; see ``CLAUDE.md`` and ``FULL_PLAN.md`` Phase 5.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping
from unittest.mock import patch

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseDomainError, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef
from qufzx.diagram.validate import IssueKind, ValidationIssue, ValidationReport, validate
from qufzx.rewrite.engine import RewriteResult, apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import Match, RewriteDomainError, RewriteError, Rule
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import compare
from qufzx.semantics.contract_numeric import ContractSizeError

_SEEDS: tuple[int, ...] = tuple(range(2500))
_DIM_D = Dim.symbol("d")
_DIM_E = Dim.symbol("e")
# ``d*e`` and ``d**2`` are here so Dim.unify's DEFERRED branch (a symbol occurring as a
# proper subterm of the other side, e.g. ``d`` against ``d*e``) is actually exercised by
# the generator -- previously only bare symbols and concrete ints appeared in the palette,
# so a leg dim could unify with another leg dim only via the concrete/concrete,
# syntactic-identity, or bare-symbol-binding branches, never the deferred one, even though
# ``_unify_surviving_legs`` has a dedicated code path for exactly this case.
_DIM_PALETTE = (Dim.concrete(2), Dim.concrete(3), _DIM_D, _DIM_E, _DIM_D * _DIM_E, _DIM_D**2)
_DIM_SYMBOL_NAMES = frozenset({"d", "e"})
_CONCRETE_TURNS = (sp.Integer(0), sp.Rational(1, 3), sp.Rational(2, 5))
# (d, e) oracle substitution pairs -- see _substitution_for. d and e are substituted
# independently (never collapsed to the same value): three pairs hold d == e (at each of
# 2, 3, 5, matching the old single-value coverage) and two hold d != e, so mixed-symbolic-
# dimension diagrams are actually exercised at both equal and unequal bindings, without
# the combinatorial (and dense-tensor-size) blowup of a full cross product.
_ORACLE_DIM_PAIRS: tuple[tuple[int, int], ...] = ((2, 2), (3, 3), (5, 5), (2, 3), (3, 2))
_PHASE_SUBSTITUTE_TURNS = sp.Rational(1, 3)
_COLORS = (Z_SPIDER, X_SPIDER)
# Entry indices used for randomly generated phase vectors. Deliberately includes indices
# (4, 6) that are out of range for the smaller members of _ORACLE_DIM_VALUES (2, 3) --
# and 6 is out of range for all of them -- so a phase legally stated over a symbolic
# dimension can carry an entry index that becomes invalid once that dimension is bound to
# a small concrete value. This is exactly the shape of the Task 1 defect: match.py's
# phase_dimension_agreement side condition binding a symbol without checking whether the
# phase's own entries remain in range at that binding.
_PHASE_INDEX_POOL = (1, 2, 3, 4, 6)

# apply()'s own message for the step-8 relative post-condition (see engine.py) -- the one
# RewriteDomainError a matcher-approved match may legitimately raise, since it reflects a
# genuine downstream conflict (e.g. a symbol forced two different concrete values by two
# independent fusions), not a builder bug. Anything else escaping apply() for a match
# find_matches() itself returned is exactly the class of defect this harness exists to
# catch (see defects 1 and 2 in the Phase 5 audit), so it must not be swallowed.
_RELATIVE_POSTCONDITION_MARKER = "rewrite introduced hard-error issue kind"


def _random_phase_index(rng: random.Random, phase_dim: Dim) -> int:
    """An entry index from ``_PHASE_INDEX_POOL``, capped to what ``phase_dim`` allows.

    When ``phase_dim`` is concrete, :class:`~qufzx.algebra.phase.PhaseVector` enforces the
    valid range at construction time, so the pool is filtered down to what is actually
    constructible. When ``phase_dim`` is symbolic, the full pool is available -- including
    indices that will turn out to be out of range once a leg-unify binding (see
    ``match.py``'s ``phase_dimension_agreement``) or an oracle substitution resolves the
    symbol to a small concrete value. That gap is exactly what this harness exists to
    exercise.
    """
    if phase_dim.is_concrete:
        max_index = phase_dim.to_int() - 1
        candidates = [i for i in _PHASE_INDEX_POOL if i <= max_index]
        return rng.choice(candidates) if candidates else 1
    return rng.choice(_PHASE_INDEX_POOL)


def _random_phase(rng: random.Random, dim: Dim, node_index: int) -> PhaseVector | None:
    """Sometimes absent, sometimes concrete, sometimes symbolic, sometimes root-of-unity;
    always over ``dim`` -- or, when ``dim`` is symbolic, sometimes over a concrete value it
    could unify to.

    The root-of-unity branch (``Phase.root_of_unity(index, phase_dim)``, turns
    ``index / phase_dim``) is the only one whose entry's free symbols can include a
    *dimension* symbol rather than a pure phase parameter -- before it existed, every entry
    this generator produced was built from :meth:`Phase.turns` (no free symbols at all) or
    :meth:`Phase.symbol` (a ``theta_i`` phase parameter, never a dimension name), so a phase
    entry referencing ``d`` or ``e`` directly (as opposed to only via its container ``Dim``)
    was never generated at all, and the defect family in
    :func:`~qufzx.rewrite.rules_library._over_shared_dim` (which reattaches entries to a
    resolved ``shared_dim`` verbatim, without substituting a binding into them) could never
    be observed disagreeing with anything.
    """
    choice = rng.random()
    if choice < 0.2:
        return None
    # Weighted 60/20/20 towards keeping ``dim`` itself (rather than overriding to a fixed
    # concrete value) when ``dim`` is symbolic, so a root-of-unity entry over ``dim`` is
    # common enough that a later leg-unify binding of that same symbol (see below) actually
    # collides with it within the fixed seed list, rather than needing an implausibly long
    # one to find a single occurrence.
    phase_dim = (
        dim
        if dim.is_concrete
        else rng.choices((dim, Dim.concrete(2), Dim.concrete(3)), weights=(3, 1, 1))[0]
    )
    index = _random_phase_index(rng, phase_dim)
    if choice < 0.45:
        return PhaseVector(phase_dim, {index: Phase.turns(rng.choice(_CONCRETE_TURNS))})
    if choice < 0.65:
        return PhaseVector(phase_dim, {index: Phase.symbol(f"theta_{node_index}")})
    return PhaseVector(phase_dim, {index: Phase.root_of_unity(index, phase_dim)})


def _build_random_diagram(rng: random.Random) -> Diagram:
    """2-4 nodes, 0-3 legs per side, colour Z/X, mostly one dim per node, a random wiring.

    Roughly a third of nodes instead draw each leg's dimension independently from
    ``_DIM_PALETTE`` (see the ``mixed`` branch below), so a single node can legitimately
    carry mixed leg dims -- e.g. two concrete dims that plainly disagree (already a hard
    ``DIMENSION_POLICY_VIOLATION`` on that node alone), or a concrete leg beside a symbolic
    one that only unifies by binding. Before this, every node had exactly one dim shared by
    every one of its legs, so a fusion's surviving legs were always already equal to
    ``shared_dim`` by construction -- Defect 1 (Phase 5 audit), an un-unified overwrite of a
    surviving leg's dim, could never be observed disagreeing with anything, since nothing
    generated here ever gave it the chance to disagree.
    """
    diagram = Diagram()
    all_ports: list[PortRef] = []
    for node_index in range(rng.randint(2, 4)):
        color = rng.choice(_COLORS)
        dim = rng.choice(_DIM_PALETTE)
        n_in = rng.randint(0, 3)
        n_out = rng.randint(0, 3)
        mixed = rng.random() < 0.35
        if mixed:
            input_dims = [rng.choice(_DIM_PALETTE) for _ in range(n_in)]
            output_dims = [rng.choice(_DIM_PALETTE) for _ in range(n_out)]
        else:
            input_dims = [dim] * n_in
            output_dims = [dim] * n_out
        phase = _random_phase(rng, dim, node_index)
        if n_in == 0 and n_out == 0 and phase is None:
            # A node with no legs and no phase has no port or slot left to carry its
            # dimension at all, so semantics/denote.py cannot resolve it -- an explicit
            # all-zero PhaseVector gives it somewhere to live, mirroring
            # rules_library.py's own "no legs survive" corner case for a merged node.
            phase = PhaseVector(dim, {})
        node_id = diagram.add_node(
            color,
            input_dims=input_dims,
            output_dims=output_dims,
            phase=phase,
        )
        all_ports.extend(PortRef(node_id, Direction.INPUT, i) for i in range(n_in))
        all_ports.extend(PortRef(node_id, Direction.OUTPUT, i) for i in range(n_out))

    rng.shuffle(all_ports)
    unwired = list(all_ports)
    while len(unwired) >= 2 and rng.random() < 0.7:
        a = unwired.pop()
        b = unwired.pop(rng.randrange(len(unwired)))
        diagram.add_wire(a, b)

    diagram.set_boundary_inputs([ref for ref in unwired if ref.direction is Direction.INPUT])
    diagram.set_boundary_outputs([ref for ref in unwired if ref.direction is Direction.OUTPUT])
    return diagram


def _free_symbol_names(diagram: Diagram) -> frozenset[str]:
    """Every dim/phase/scalar symbol in ``diagram``, mirroring ``check.py``'s own helper."""
    names: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.inputs, *node.outputs):
            names |= port.dim.free_symbols
        if node.phase is not None:
            names |= node.phase.free_symbols
    return frozenset(names)


def _substitution_for(
    names: Iterable[str], d_value: int, e_value: int
) -> dict[str, int | sp.Rational]:
    """``d`` -> ``d_value``, ``e`` -> ``e_value`` (independently -- never collapsed to one
    value, so mixed-symbolic-dimension diagrams are actually exercised at ``d != e``);
    every other (phase) symbol -> a fixed rational."""
    dim_values = {"d": d_value, "e": e_value}
    return {
        name: (dim_values[name] if name in _DIM_SYMBOL_NAMES else _PHASE_SUBSTITUTE_TURNS)
        for name in names
    }


def _assert_phase_entries_consistent_with_dim(diagram: Diagram, seed: int) -> None:
    """Every node's phase entries reference only its own dim's symbols or a phase parameter.

    A free symbol appearing in a phase entry's turns-expression is legitimate in exactly
    two cases: it is one of that phase vector's own container ``dim``'s free symbols (e.g.
    a root-of-unity entry ``index / d`` sitting on a ``PhaseVector`` whose ``dim`` is still
    ``d``), or it is a genuine phase parameter (a ``theta_i`` symbol from
    :meth:`Phase.symbol`, which never denotes a dimension at all). What must never happen is
    a *dimension* symbol (one of ``_DIM_SYMBOL_NAMES``) surviving in an entry after the
    container ``dim`` has already been resolved past it -- e.g. an entry ``1/d`` sitting on
    a ``PhaseVector`` whose ``dim`` is the concrete ``2`` (because a fusion's shared_dim
    resolution bound ``d := 2`` and the merged phase's container dim was updated to match,
    while the pre-fusion entry that still says ``1/d`` was carried over unchanged). This is
    exactly the shape of the ``_over_shared_dim`` defect family (see
    ``rules_library.py``'s module docstring, "Dimension of the merged node", and Task 2 of
    the Phase 5 final fix round): a phase entry frozen in terms of a symbol its own
    container dimension no longer mentions denotes a different (and wrong) angle once that
    symbol's binding is substituted in.
    """
    for node_id, node in diagram.nodes.items():
        phase = node.phase
        if phase is None:
            continue
        dim_symbols = phase.dim.free_symbols
        for index, entry in phase.entries().items():
            stale_dim_symbols = (entry.free_symbols & _DIM_SYMBOL_NAMES) - dim_symbols
            assert not stale_dim_symbols, (
                f"seed {seed}: node {node_id!r} phase entry at index {index} references "
                f"dimension symbol(s) {sorted(stale_dim_symbols)} not present in its own "
                f"PhaseVector's dim {phase.dim} -- the container dimension has already been "
                "resolved past a symbol the entry itself still depends on"
            )


def _hard_error_kinds(diagram: Diagram) -> frozenset[IssueKind]:
    return frozenset(issue.kind for issue in validate(diagram).errors)


def _is_cleanly_contractible(diagram: Diagram) -> bool:
    report = validate(diagram)
    return report.is_valid and not report.deferred


def _apply_ignoring_step8(diagram: Diagram, rule: Rule, match: Match) -> RewriteResult:
    """Re-run ``apply`` with its step-8 relative post-condition disarmed, for re-derivation.

    Patches ``qufzx.rewrite.engine.validate`` (the name ``apply`` actually calls, per its
    module-level import) to report no issues at all for the duration of this one call, so
    ``input_hard_counts`` and ``result_hard_counts`` are both empty and their difference can
    never be non-empty -- ``apply`` runs every other step exactly as normal and returns its
    result unconditionally. This exists solely so ``_check_one_match`` can independently
    re-derive, via the real (unpatched) ``validate``, whether a step-8 raise from the normal
    call was actually justified -- see that function and Defect 1 (Phase 5 round-7 audit),
    which is exactly the class of bug a message-substring whitelist alone cannot catch.
    """
    with patch("qufzx.rewrite.engine.validate", return_value=ValidationReport(())):
        return apply(diagram, rule, match)


def _independent_issue_key(
    issue: ValidationIssue,
    consumed_node_ids: frozenset[NodeId],
    port_mapping: Mapping[PortRef, PortRef],
) -> tuple[IssueKind, object]:
    """A ``(kind, ref)`` key for one *input*-diagram hard-error issue, in post-rewrite terms.

    A from-scratch reimplementation of the same idea
    :func:`qufzx.rewrite.engine._translate_input_issue_key` embodies -- written
    independently here (not by importing and calling that private function) so this
    harness gives real cross-check value against a regression in ``apply``'s own step-8
    bookkeeping, rather than trivially agreeing with it by construction. A ``port_ref`` or
    wire endpoint on a node *not* being consumed passes through unchanged; one on a
    consumed node is looked up in ``port_mapping`` (falling back to itself -- the matched
    port itself is never in ``port_mapping`` and has no post-rewrite counterpart, so it is
    left as something that will correctly match nothing on the result side). A wire's two
    endpoints are compared as an unordered ``frozenset`` pair, matching
    :class:`~qufzx.diagram.graph.Wire`'s own order-independent equality. A ``node_id`` on a
    consumed node has no principled translation from only this function's inputs (spider
    fusion always merges into exactly one new node, but nothing here is told which) and is
    left unchanged -- the same fail-closed posture the module under test documents for that
    case; it deliberately then cannot match anything in :func:`_post_issue_key`'s output.
    """

    def _translate(ref: PortRef) -> PortRef:
        if ref.node_id not in consumed_node_ids:
            return ref
        return port_mapping.get(ref, ref)

    if issue.port_ref is not None:
        return (issue.kind, _translate(issue.port_ref))
    if issue.wire is not None:
        wire = issue.wire
        return (issue.kind, frozenset((_translate(wire.a), _translate(wire.b))))
    if issue.node_id is not None:
        return (issue.kind, issue.node_id)
    return (issue.kind, None)


def _post_issue_key(issue: ValidationIssue) -> tuple[IssueKind, object]:
    """The post-rewrite-diagram counterpart of :func:`_independent_issue_key`'s keying."""
    if issue.port_ref is not None:
        return (issue.kind, issue.port_ref)
    if issue.wire is not None:
        wire = issue.wire
        return (issue.kind, frozenset((wire.a, wire.b)))
    if issue.node_id is not None:
        return (issue.kind, issue.node_id)
    return (issue.kind, None)


def _check_one_match(diagram: Diagram, match: FusionMatch, seed: int) -> int:
    """Apply ``match`` and check it soundly; returns how many oracle comparisons actually ran.

    A zero return (no comparison ever reached line "assert comparison.matched") is not by
    itself a failure here -- ``test_random_diagrams_fuse_soundly`` sums this across every
    checked match and asserts the *total* clears a meaningful floor, which is what actually
    closes the hole where an always-skipped oracle arm would let the test pass vacuously.
    """
    try:
        result = apply(diagram, SPIDER_FUSION, match)
    except RewriteDomainError as exc:
        assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
            f"seed {seed}: apply() raised a RewriteDomainError that is not the relative "
            f"post-condition, for a match find_matches() itself returned: {exc}"
        )
        # Re-derive whether the block was actually justified, independently of apply()'s
        # own step-8 bookkeeping, instead of trusting the message alone (a message-substring
        # whitelist here is exactly what let Defect 1 -- a false-positive step-8 block --
        # survive three prior audit rounds undetected). ``_apply_ignoring_step8`` forces the
        # rewrite through regardless of step 8; ``_independent_issue_key`` /
        # ``_post_issue_key`` then build the same *multiset* comparison ``apply`` itself
        # makes (not merely a set-of-kinds one -- a coarser check cannot tell a false-
        # positive block apart from a second, independent issue of a kind the input already
        # carried once, which is a real regression apply() is right to catch), computed
        # from scratch rather than by calling apply()'s own private helpers, so this is a
        # genuine cross-check and not a tautology.
        forced = _apply_ignoring_step8(diagram, SPIDER_FUSION, match)
        pre_counts = Counter(
            _independent_issue_key(
                issue, frozenset(forced.step.matched_node_ids), forced.step.port_mapping
            )
            for issue in validate(diagram).errors
        )
        post_counts = Counter(_post_issue_key(issue) for issue in validate(forced.diagram).errors)
        introduced_issues = post_counts - pre_counts
        assert introduced_issues, (
            f"seed {seed}: apply() raised the step-8 relative post-condition, but "
            f"re-deriving independently found no hard-error issue introduced that was not "
            f"already present (in translated form) in the input diagram -- apply() blocked "
            f"a rewrite that introduced nothing: {exc}"
        )
        return 0
    except RewriteError as exc:  # pragma: no cover - documents the intended failure mode
        raise AssertionError(f"seed {seed}: unexpected {type(exc).__name__}: {exc}") from exc

    post = result.diagram
    _assert_phase_entries_consistent_with_dim(post, seed)

    introduced = _hard_error_kinds(post) - _hard_error_kinds(diagram)
    assert not introduced, (
        f"seed {seed}: rewrite introduced hard-error issue kind(s) "
        f"{sorted(k.value for k in introduced)} not present in the input diagram"
    )

    oracle_runs = 0
    all_names = _free_symbol_names(diagram) | _free_symbol_names(post)
    for d_value, e_value in _ORACLE_DIM_PAIRS:
        subs = _substitution_for(all_names, d_value, e_value)
        try:
            pre_concrete = diagram.substitute(subs)
        except PhaseDomainError:
            # Some node's phase (possibly unrelated to this match) carries an entry
            # index that this particular (d_value, e_value) choice makes invalid --
            # a legitimate mismatch between a well-formed symbolic diagram and one
            # arbitrary concrete substitution, not a matcher or builder defect.
            continue
        if not _is_cleanly_contractible(pre_concrete):
            continue
        try:
            comparison = compare(diagram, post, subs)
        except ContractSizeError:
            # This (d_value, e_value) choice makes the concrete diagram too large to
            # densely contract at all -- an oracle scale limit, not a matcher or
            # builder defect, so this substitution is skipped like any other
            # not-cleanly-contractible one.
            continue
        assert comparison.matched, (
            f"seed {seed}, d={d_value}, e={e_value}: oracle mismatch: {comparison.reason}"
        )
        oracle_runs += 1
    return oracle_runs


_MIN_ORACLE_COMPARISONS = 40
"""Floor for the total oracle comparisons summed across every checked match.

Without this, an always-skipped oracle arm (e.g. every substitution failing
``_is_cleanly_contractible`` or raising ``PhaseDomainError``) would let the test pass
while never actually calling :func:`~qufzx.semantics.check.compare` -- see the module
docstring and this file's Task 2 history. The actual count on the current seed list and
generator was around 260 before an earlier Phase 5 audit round's Defect 1 fix added mixed
per-node leg dims to the generator (see ``_build_random_diagram``); mixed legs make more
candidates fail ``_is_cleanly_contractible`` pre-rewrite (a node with a hard
``DIMENSION_POLICY_VIOLATION`` already on it is exactly the case that check is meant to
skip), so fewer oracle comparisons actually run -- around 130 since, and still around 130
after the Phase 5 round-7 audit's Defect 1 fix in ``engine.py`` (see that module's
docstring; that fix unblocked matches that were never going to contribute an oracle
comparison regardless, blocked or not).

The Phase 5 final fix round's Step 1 harness extension (``d*e`` and ``d**2`` added to
``_DIM_PALETTE``, and ``_random_phase`` reweighted towards symbolic-dim root-of-unity
entries -- both needed to actually reach the ``_over_shared_dim`` defect family; see this
module's docstring and ``_random_phase``'s) dropped the measured total further, to around
57: a product or power dim more often fails to unify with a bare symbol or a mismatched
concrete value under ``Dim.unify``'s deliberately weak placeholder (deferring rather than
solving), which makes more nodes carry a hard ``DIMENSION_POLICY_VIOLATION`` that
``_is_cleanly_contractible`` is meant to skip, and a root-of-unity phase entry more often
has an index that some ``(d_value, e_value)`` substitution puts out of range, raising
``PhaseDomainError`` before ``compare`` is ever reached. 40 leaves headroom against
incidental generator changes while still failing hard if the oracle arm silently stops
running.
"""


class TestSpiderFusionProperties:
    def test_random_diagrams_fuse_soundly(self) -> None:
        checked_any_match = False
        total_oracle_comparisons = 0
        for seed in _SEEDS:
            rng = random.Random(seed)
            diagram = _build_random_diagram(rng)
            for match in find_matches(diagram):
                checked_any_match = True
                total_oracle_comparisons += _check_one_match(diagram, match, seed)
        assert checked_any_match, "the generator never produced a single fusion match"
        assert total_oracle_comparisons >= _MIN_ORACLE_COMPARISONS, (
            f"only {total_oracle_comparisons} oracle comparisons actually executed "
            f"(floor is {_MIN_ORACLE_COMPARISONS}); the oracle arm may be silently "
            "skipping every substitution instead of exercising compare()"
        )

    def test_phase_index_out_of_range_under_binding_is_not_a_match(self) -> None:
        """Regression test for the Task 1 defect: see match.py's module docstring, condition 6.

        A phase legally stated over symbolic ``d`` with an entry at index 5 must not be
        reported as a match once leg-unify binds ``d := 2`` -- index 5 is out of range at
        that binding. Before the fix, ``phase_dimension_agreement`` checked only the phase
        vector's container ``Dim`` against the resolved shared dimension, never its entry
        indices, so this candidate passed matching and then ``spider_fusion_builder``
        raised ``RewriteDomainError`` trying to build it -- violating the invariant that
        every match ``find_matches`` returns is applicable by ``apply`` without raising.
        """
        d = Dim.symbol("d")
        diagram = Diagram()
        a = diagram.add_node(
            Z_SPIDER,
            [],
            [d],
            phase=PhaseVector(d, {5: Phase.turns(sp.Rational(1, 3))}),
        )
        b = diagram.add_node(Z_SPIDER, [Dim.concrete(2)], [])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))

        matches = find_matches(diagram)
        assert matches == (), (
            "a phase entry out of range under the leg-unify binding must be excluded as a "
            f"non-match, not returned and left for apply() to fail on later; got {matches!r}"
        )

    def test_phase_entry_over_bound_symbol_is_substituted_not_left_stale(self) -> None:
        """Regression test for the Task 2 defect: see rules_library.py's ``_over_shared_dim``.

        A phase legally stated over symbolic ``d`` (``1/d`` turns, from
        ``Phase.root_of_unity``) fuses against a spider whose leg is the concrete ``2``,
        binding ``d := 2``. Before the fix, ``_over_shared_dim`` reattached the entry
        unchanged onto the resolved (now concrete) ``shared_dim``, producing
        ``PhaseVector[2]({1: 1/d turns})`` -- a phase vector whose container dimension no
        longer mentions the symbol its own entry still depends on, silently discarding the
        ``d := 2`` constraint that made the fusion well-formed in the first place. The fixed
        builder substitutes the accumulated binding into the entry before reattaching it, so
        the merged phase is the concrete ``1/2`` turns the binding actually implies, and the
        oracle agrees exactly at ``d = 2``.
        """
        d = Dim.symbol("d")
        diagram = Diagram()
        a = diagram.add_node(
            Z_SPIDER,
            [],
            [d, d],
            phase=PhaseVector(d, {1: Phase.root_of_unity(1, d)}),
        )
        b = diagram.add_node(Z_SPIDER, [Dim.concrete(2)], [])
        diagram.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
        diagram.set_boundary_outputs([PortRef(a, Direction.OUTPUT, 1)])

        matches = find_matches(diagram)
        assert len(matches) == 1
        result = apply(diagram, SPIDER_FUSION, matches[0])
        merged = result.diagram.nodes[result.new_node_ids[0]]
        assert merged.phase == PhaseVector(
            Dim.concrete(2), {1: Phase.turns(sp.Rational(1, 2))}
        ), f"expected the binding d := 2 substituted into the entry, got {merged.phase!r}"

        comparison = compare(diagram, result.diagram, {"d": 2})
        assert comparison.matched, f"oracle mismatch at d=2: {comparison.reason}"
