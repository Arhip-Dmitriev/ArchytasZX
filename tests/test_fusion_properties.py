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
_CLEAN_SEEDS: tuple[int, ...] = tuple(range(20000))
_DIM_D = Dim.symbol("d")
_DIM_E = Dim.symbol("e")
# ``d*e`` and ``d**2`` are here so Dim.unify's DEFERRED branch (a symbol occurring as a
# proper subterm of the other side, e.g. ``d`` against ``d*e``) is actually exercised by
# the generator -- previously only bare symbols and concrete ints appeared in the palette,
# so a leg dim could unify with another leg dim only via the concrete/concrete,
# syntactic-identity, or bare-symbol-binding branches, never the deferred one, even though
# ``_unify_surviving_legs`` has a dedicated code path for exactly this case.
_DIM_PALETTE = (
    Dim.concrete(2),
    Dim.concrete(3),
    Dim.concrete(4),
    Dim.concrete(6),
    _DIM_D,
    _DIM_E,
    _DIM_D * _DIM_E,
    _DIM_D**2,
)
_DIM_SYMBOL_NAMES = frozenset({"d", "e"})
_CONCRETE_TURNS = (sp.Integer(0), sp.Rational(1, 3), sp.Rational(2, 5))
# (d, e) oracle substitution pairs -- see _substitution_for. d and e are substituted
# independently (never collapsed to the same value): three pairs hold d == e (at each of
# 2, 3, 5, matching the old single-value coverage), two hold d != e, and (2, 1)/(3, 1) let
# a ``d*e`` or ``d**2`` leg agree with a concrete 2/3 leg at e=1 (Fix 4(a): without these,
# ``_DIM_PALETTE``'s product/power dims almost never landed on a value any concrete leg
# in the palette could also take, so a candidate carrying one almost always failed
# ``_is_cleanly_contractible`` before the oracle ran).
_ORACLE_DIM_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 2),
    (3, 3),
    (5, 5),
    (2, 3),
    (3, 2),
    (2, 1),
    (3, 1),
)
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


_CLEAN_DIM_VALUES = (2, 3, 4, 5)


def _random_clean_phase(rng: random.Random, dim: Dim, node_index: int) -> PhaseVector | None:
    """A phase for a cleanly-contractible node: always fully concrete, entry always in range.

    Unlike :func:`_random_phase`, this never produces an entry over a dimension symbol or a
    phase parameter (``Phase.symbol``) -- every free symbol in the diagram would otherwise
    need to appear in the oracle's ``assignment``, and the whole point of this generator is
    a diagram with *no* free symbols at all, so :func:`~qufzx.semantics.check.compare` can be
    called with an empty assignment and never raise ``CheckGrammarError`` for a missing one.
    """
    if rng.random() < 0.2:
        return None
    max_index = dim.to_int() - 1
    if max_index < 1:
        return PhaseVector(dim, {})
    index = rng.randint(1, max_index)
    return PhaseVector(dim, {index: Phase.turns(rng.choice(_CONCRETE_TURNS))})


def _build_clean_diagram(rng: random.Random) -> Diagram:
    """2-3 nodes, 0-2 legs per side, colour Z/X, *one* concrete dim shared by every leg of
    every node in the whole diagram, a fully concrete phase (or none) per node, and random
    wiring drawn from the flat shuffled port pool -- so a self-loop (two ports of the same
    node landing adjacent in the shuffle) or a same-direction wire (two output ports, or two
    input ports, landing adjacent) arises freely, exactly like :func:`_build_random_diagram`'s
    own wiring mechanism, just never blocked by a mismatched dimension since there is only
    ever one dimension in play. Every diagram this produces is cleanly contractible by
    construction: :mod:`qufzx.diagram.validate` reports no issue at all (hard or deferred),
    since every leg agrees syntactically (not merely by unification) and nothing is
    symbolic -- there is no draw from ``_DIM_PALETTE``'s mixed, unify-only, or deferred-unify
    dimensions the way :func:`_build_random_diagram` deliberately includes.

    The leg-count and node-count ranges are deliberately smaller than
    :func:`_build_random_diagram`'s (which mostly never reaches ``compare`` at all, so its
    own boundary size barely matters for wall time): since here essentially every match
    *does* reach :func:`~qufzx.semantics.check.compare`, an unconstrained boundary size
    would let the free (unwired) leg count -- and so the dense contracted tensor's element
    count, ``dim ** boundary_legs`` -- grow large enough to dominate this test's runtime
    (measured: 0-3 legs per side across up to 4 nodes occasionally left enough boundary legs
    that a single ``compare()`` call took over 100ms, and a few tripped
    :class:`~qufzx.semantics.contract_numeric.ContractSizeError` outright). This tighter
    range keeps typical boundaries small while still leaving plenty of room for a
    fusion-eligible pair (each node needs only one spare leg on the wired side to become a
    fusion candidate), and still produces self-loops, same-direction wires, and all six
    colour/direction combinations freely.

    This is the generator Task 4 (Phase 5 closing round) adds to give the oracle-equality
    arm of this harness a diagram population that actually reaches ``compare()`` on (almost)
    every match, rather than being dropped by :func:`_is_cleanly_contractible` before the
    oracle ever runs -- see ``_MIN_CLEAN_ORACLE_COMPARISONS``'s docstring for why the
    existing generator cannot be widened to do this instead without losing its own (still
    needed) mixed-dimension coverage.
    """
    diagram = Diagram()
    dim = Dim.concrete(rng.choice(_CLEAN_DIM_VALUES))
    all_ports: list[PortRef] = []
    for node_index in range(rng.randint(2, 3)):
        color = rng.choice(_COLORS)
        n_in = rng.randint(0, 2)
        n_out = rng.randint(0, 2)
        phase = _random_clean_phase(rng, dim, node_index)
        if n_in == 0 and n_out == 0 and phase is None:
            phase = PhaseVector(dim, {})
        node_id = diagram.add_node(
            color,
            input_dims=[dim] * n_in,
            output_dims=[dim] * n_out,
            phase=phase,
        )
        all_ports.extend(PortRef(node_id, Direction.INPUT, i) for i in range(n_in))
        all_ports.extend(PortRef(node_id, Direction.OUTPUT, i) for i in range(n_out))

    rng.shuffle(all_ports)
    unwired = list(all_ports)
    # A higher continue-probability than _build_random_diagram's 0.7: since this generator's
    # matches overwhelmingly reach compare(), keeping the leftover boundary small keeps the
    # dense contracted tensor (dim ** boundary_legs) small too -- see the docstring above.
    while len(unwired) >= 2 and rng.random() < 0.85:
        a = unwired.pop()
        b = unwired.pop(rng.randrange(len(unwired)))
        diagram.add_wire(a, b)

    diagram.set_boundary_inputs([ref for ref in unwired if ref.direction is Direction.INPUT])
    diagram.set_boundary_outputs([ref for ref in unwired if ref.direction is Direction.OUTPUT])
    return diagram


_MIXED_DIM_PALETTE = (_DIM_D, Dim.concrete(2), Dim.concrete(3))
_MIXED_ORACLE_D_VALUES = (2, 3, 4, 5)


def _build_mixed_diagram(rng: random.Random) -> Diagram:
    """Fix 4(b): 2-4 nodes, Z/X mix, one dim per node from ``(d, 2, 3)``, phases only from
    ``Phase.turns`` or ``Phase.root_of_unity(1, pdim)`` over a palette member, wiring from a
    shuffled flat port pool (so self-loops and same-direction wires arise freely). Unlike
    :func:`_build_random_diagram`'s deliberately hard-to-contract ``d*e``/``d**2`` legs (see
    ``_MIN_ORACLE_COMPARISONS``'s docstring), every leg here is drawn from a 3-member
    palette small enough that a symbolic leg and a concrete one frequently coexist on a
    fusable pair, exercising a ``shared_dim`` refined by a leg-unify *binding* -- the region
    the last several audit rounds' defects lived in -- while still reaching the oracle at
    d in {2, 3, 4, 5} on most matches.
    """
    diagram = Diagram()
    all_ports: list[PortRef] = []
    for node_index in range(rng.randint(2, 4)):
        color = rng.choice(_COLORS)
        dim = rng.choice(_MIXED_DIM_PALETTE)
        n_in = rng.randint(0, 3)
        n_out = rng.randint(0, 3)
        phase = None
        if rng.random() >= 0.2:
            pdim = rng.choice(_MIXED_DIM_PALETTE)
            entry = (
                Phase.turns(rng.choice(_CONCRETE_TURNS))
                if rng.random() < 0.5
                else Phase.root_of_unity(1, pdim)
            )
            phase = PhaseVector(pdim, {1: entry})
        if n_in == 0 and n_out == 0 and phase is None:
            phase = PhaseVector(dim, {})
        node_id = diagram.add_node(
            color, input_dims=[dim] * n_in, output_dims=[dim] * n_out, phase=phase
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


def _check_mixed_diagram_chain(rng: random.Random, seed: int) -> tuple[int, int]:
    """Fix 4(c): repeatedly find-and-apply a match on a fresh mixed diagram, to a fixpoint
    (up to 10 steps), oracle-checking each step at every d in ``_MIXED_ORACLE_D_VALUES`` --
    skipping a (diagram, d) where either the substituted pre- or post-diagram is not
    cleanly contractible, and any ``ContractSizeError`` -- so a defect that only appears on
    the second (or later) fusion of a chain is exercised, not only a single fusion against
    a fresh diagram. Returns ``(comparisons_ran, chain_steps)``.
    """
    diagram = _build_mixed_diagram(rng)
    comparisons = 0
    steps = 0
    for _ in range(10):
        matches = find_matches(diagram)
        if not matches:
            break
        match = rng.choice(matches)
        try:
            result = apply(diagram, SPIDER_FUSION, match)
        except RewriteDomainError as exc:
            assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
                f"seed {seed}, step {steps}: unexpected RewriteDomainError: {exc}"
            )
            break
        post = result.diagram
        _assert_phase_entries_consistent_with_dim(post, seed)
        introduced = _hard_error_kinds(post) - _hard_error_kinds(diagram)
        assert not introduced, (
            f"seed {seed}, step {steps}: rewrite introduced hard-error issue kind(s) "
            f"{sorted(k.value for k in introduced)} not present in the input diagram"
        )
        for d_value in _MIXED_ORACLE_D_VALUES:
            subs = {"d": d_value}
            try:
                pre_concrete = diagram.substitute(subs)
                post_concrete = post.substitute(subs)
            except PhaseDomainError:
                continue
            if not (_is_cleanly_contractible(pre_concrete) and _is_cleanly_contractible(post_concrete)):
                continue
            try:
                comparison = compare(diagram, post, subs)
            except ContractSizeError:
                continue
            assert comparison.matched, (
                f"seed {seed}, step {steps}, d={d_value}: oracle mismatch: {comparison.reason}"
            )
            comparisons += 1
        diagram = post
        steps += 1
    return comparisons, steps


def _match_color_direction(diagram: Diagram, match: FusionMatch) -> tuple[str, str, str]:
    """``(generator name, a-side direction, b-side direction)`` for one located match.

    Read directly off the diagram and the match's own wire, not off any side-condition
    outcome's free-text detail -- so the coverage assertion in
    ``test_clean_diagrams_fuse_soundly`` checks the actual located structure, not a string.
    """
    node_a = diagram.nodes[match.a_id]
    ref_a = match.wire.a if match.wire.a.node_id == match.a_id else match.wire.b
    ref_b = match.wire.b if match.wire.a.node_id == match.a_id else match.wire.a
    return (node_a.generator_type.name, ref_a.direction.value, ref_b.direction.value)


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


def _check_one_clean_match(diagram: Diagram, match: FusionMatch, seed: int) -> tuple[int, int]:
    """Apply ``match``, assert oracle equality, and return ``(comparisons_ran, size_skips)``.

    Unlike :func:`_check_one_match`, ``diagram`` here is already fully concrete (one
    dimension, concrete phases only, no symbols anywhere per :func:`_build_clean_diagram`),
    so there is exactly one substitution to try -- the empty one -- rather than a sweep over
    ``_ORACLE_DIM_PAIRS``, and no ``PhaseDomainError`` branch is needed (an out-of-range
    phase entry cannot arise: every entry was drawn in range for its node's one fixed
    concrete dim to begin with). ``ContractSizeError`` is still a genuine oracle scale
    limit, not a defect, so it is skipped -- but counted in the returned ``size_skips`` so a
    generator change that started tripping it on every match could not silently masquerade
    as "no failures" the way an uncounted skip would.
    """
    result = apply(diagram, SPIDER_FUSION, match)
    post = result.diagram
    _assert_phase_entries_consistent_with_dim(post, seed)

    introduced = _hard_error_kinds(post) - _hard_error_kinds(diagram)
    assert not introduced, (
        f"seed {seed}: rewrite introduced hard-error issue kind(s) "
        f"{sorted(k.value for k in introduced)} not present in the input diagram"
    )

    try:
        comparison = compare(diagram, post, {})
    except ContractSizeError:
        return 0, 1
    assert comparison.matched, f"seed {seed}: oracle mismatch: {comparison.reason}"
    return 1, 0


_MIN_CLEAN_ORACLE_COMPARISONS = 3000
"""Floor for the total oracle comparisons summed across every checked clean-diagram match.

Set well above what :class:`TestSpiderFusionProperties`'s own reference run actually
achieves (a few thousand over ``_CLEAN_SEEDS``), the same margin-for-safety posture as
``_MIN_ORACLE_COMPARISONS`` above but at a scale that means something for *this* arm: the
existing generator's floor was walked down from 260 to 130 to 57 across prior audit rounds
because its deliberately mixed, deferred-unify, and product/power dimensions make almost
every candidate fail :func:`_is_cleanly_contractible` before the oracle ever runs (~57 out
of 2,500 seeds' worth of matches), so a low floor there says nothing about whether the
oracle-equality property -- Phase 5's stated completion condition -- is actually being
exercised at scale. ``_build_clean_diagram`` is cleanly contractible *by construction* (one
concrete dim, no symbols anywhere), so essentially every match this generator produces
reaches :func:`~qufzx.semantics.check.compare` -- measured at 7,656 comparisons over
``_CLEAN_SEEDS``'s 20,000 seeds, in about 20s wall time (well within the existing test's own
~65s), with zero ``ContractSizeError`` skips at the current leg-count/wiring-probability
tuning (see :func:`_build_clean_diagram`'s docstring for why those are kept small). 3,000
leaves comfortable headroom against incidental generator tuning while still failing hard if
this arm's oracle calls silently stopped running.
"""

_MIN_ORACLE_COMPARISONS = 82
"""Floor for the total oracle comparisons summed across every checked match.

Without this, an always-skipped oracle arm (e.g. every substitution failing
``_is_cleanly_contractible`` or raising ``PhaseDomainError``) would let the test pass
while never actually calling :func:`~qufzx.semantics.check.compare`. Fix 4(a) (Phase 5
post-closing audit) added ``Dim.concrete(4)``/``Dim.concrete(6)`` to ``_DIM_PALETTE`` and
``(2, 1)``/``(3, 1)`` to ``_ORACLE_DIM_PAIRS`` so a ``d*e`` or ``d**2`` leg can actually
agree with a concrete leg at some oracle substitution instead of almost always failing
``_is_cleanly_contractible`` first; this measured 118 comparisons over ``_SEEDS`` (up from
~57 before). 82 is roughly 70% of that measurement, leaving headroom against incidental
generator changes while still failing hard if the oracle arm silently stops running.
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

    def test_clean_diagrams_fuse_soundly(self) -> None:
        """Task 4 (Phase 5 closing round): the oracle-equality arm, at real scale.

        Uses :func:`_build_clean_diagram` (cleanly contractible by construction, unlike
        ``_build_random_diagram``'s deliberately mixed-dimension population) so that
        essentially every match reaches :func:`~qufzx.semantics.check.compare`, not just the
        ~57-out-of-2,500-seeds' worth the other arm's own floor documents. Also asserts the
        six colour/direction shapes ``consumed_wire_direction_permitted_for_color`` actually permits (see
        ``match.py``'s condition 4) are all still being generated -- so a future change to
        this generator that stopped producing, say, same-direction Z-Z wires would fail this
        test directly, rather than silently losing coverage of the Z-widening commit.
        """
        checked_any_match = False
        total_oracle_comparisons = 0
        total_size_skips = 0
        seen_combinations: set[tuple[str, str, str]] = set()
        for seed in _CLEAN_SEEDS:
            rng = random.Random(seed)
            diagram = _build_clean_diagram(rng)
            for match in find_matches(diagram):
                checked_any_match = True
                seen_combinations.add(_match_color_direction(diagram, match))
                comparisons, size_skips = _check_one_clean_match(diagram, match, seed)
                total_oracle_comparisons += comparisons
                total_size_skips += size_skips
        assert checked_any_match, "the clean generator never produced a single fusion match"
        assert total_oracle_comparisons >= _MIN_CLEAN_ORACLE_COMPARISONS, (
            f"only {total_oracle_comparisons} oracle comparisons actually executed "
            f"(floor is {_MIN_CLEAN_ORACLE_COMPARISONS}, {total_size_skips} skipped for "
            "ContractSizeError); the oracle arm may be silently skipping every match "
            "instead of exercising compare()"
        )
        expected_combinations = {
            (Z_SPIDER.name, "input", "input"),
            (Z_SPIDER.name, "input", "output"),
            (Z_SPIDER.name, "output", "input"),
            (Z_SPIDER.name, "output", "output"),
            (X_SPIDER.name, "input", "output"),
            (X_SPIDER.name, "output", "input"),
        }
        missing_combinations = expected_combinations - seen_combinations
        assert not missing_combinations, (
            f"the clean generator never produced these colour/direction combinations: "
            f"{sorted(missing_combinations)} -- coverage of the Z same-direction widening "
            "(or the X alternating-only restriction) may have silently regressed"
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


_MIXED_SEEDS: tuple[int, ...] = tuple(range(40000))
_MIN_MIXED_ORACLE_COMPARISONS = 2200
"""Floor for the total oracle comparisons summed across every checked step of every chain.

Fix 4(b)/(c) (Phase 5 post-closing audit): :func:`_build_mixed_diagram`'s small
``(d, 2, 3)`` palette is chosen so a symbolic and a concrete leg frequently coexist on a
fusable pair, reaching a ``shared_dim`` refined by a leg-unify binding -- unlike
``_build_random_diagram``'s ``d*e``/``d**2`` legs, which mostly fail
``_is_cleanly_contractible`` before the oracle runs (see ``_MIN_ORACLE_COMPARISONS``'s
docstring). Measured 3,154 comparisons over 3,103 fixpoint-chain steps across
``_MIXED_SEEDS``'s 40,000 seeds, in about 27s wall time. 2,200 is roughly 70% of that
measurement, leaving headroom against incidental generator tuning while still failing hard
if this arm's oracle calls silently stopped running.
"""


class TestMixedSymbolicConcreteFusionProperties:
    def test_mixed_diagrams_fuse_soundly_to_a_fixpoint(self) -> None:
        total_oracle_comparisons = 0
        total_chain_steps = 0
        for seed in _MIXED_SEEDS:
            rng = random.Random(seed)
            comparisons, steps = _check_mixed_diagram_chain(rng, seed)
            total_oracle_comparisons += comparisons
            total_chain_steps += steps
        assert total_chain_steps > 0, "the mixed generator never produced a single fusion match"
        assert total_oracle_comparisons >= _MIN_MIXED_ORACLE_COMPARISONS, (
            f"only {total_oracle_comparisons} oracle comparisons actually executed "
            f"(floor is {_MIN_MIXED_ORACLE_COMPARISONS}); the oracle arm may be silently "
            "skipping every substitution instead of exercising compare()"
        )
