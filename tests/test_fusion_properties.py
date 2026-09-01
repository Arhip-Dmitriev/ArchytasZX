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
manual fix rounds are meant to replace; see the spec and ``FULL_PLAN.md`` Phase 5.
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from unittest.mock import patch

import pytest
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseDomainError, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, NodeId, PortRef, Wire
from qufzx.diagram.validate import IssueKind, ValidationIssue, ValidationReport, validate
from qufzx.rewrite import match as match_module
from qufzx.rewrite.engine import RewriteResult, apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import (
    BuildResult,
    ConstraintOutcome,
    ConstraintSource,
    DimensionConstraint,
    Match,
    RewriteDomainError,
    RewriteError,
    RewriteGrammarError,
    Rule,
    SideConditionOutcome,
)
from qufzx.rewrite.rules_library import SPIDER_FUSION, spider_fusion_builder
from qufzx.semantics.check import compare
from qufzx.semantics.contract_numeric import ContractSizeError, ContractValidationError
from qufzx.semantics.denote import DenoteError, denote

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
    _maybe_corrupt_a_boundary_ref(rng, diagram)
    return diagram


_MALFORMED_BOUNDARY_PROBABILITY = 0.03
"""Chance :func:`_maybe_corrupt_a_boundary_ref` replaces one boundary entry with a malformed
``PortRef``. Low enough that this generator's existing oracle-comparison floors
(``_MIN_ORACLE_COMPARISONS``) are not meaningfully eaten into (a corrupted diagram is
rejected by ``find_matches`` before any match is ever returned, forfeiting that seed's
comparisons entirely -- see ``test_random_diagrams_fuse_soundly``), while still being high
enough that, summed over ``_SEEDS``' several thousand seeds, the malformed-boundary path is
exercised many times over."""


def _maybe_corrupt_a_boundary_ref(rng: random.Random, diagram: Diagram) -> bool:
    """With low probability, replace one boundary entry in place with a malformed ``PortRef``.

    Phase 5 post-closing audit round 18, Defect 2: :func:`~qufzx.rewrite.match.find_matches`
    must reject a malformed boundary entry (an unknown node id, or an out-of-range index)
    exactly as it already rejects a malformed wire endpoint -- see that module's docstring,
    "Malformed boundary references". Widening this generator to sometimes produce one is
    what lets :class:`TestSpiderFusionProperties` exercise that rejection at the property
    harness's own scale and diagram variety, not only via the hand-picked unit tests in
    ``test_match.py``'s ``TestOutOfRangeBoundaryRefRaises``. Returns whether a corruption was
    actually made (there may be no boundary entry at all to corrupt), so the caller can track
    how often this path is really exercised across the whole seed range.
    """
    if rng.random() >= _MALFORMED_BOUNDARY_PROBABILITY:
        return False
    inputs = list(diagram.boundary_inputs)
    outputs = list(diagram.boundary_outputs)
    pool: list[tuple[str, int]] = [("in", i) for i in range(len(inputs))]
    pool += [("out", i) for i in range(len(outputs))]
    if not pool:
        return False
    which, index = rng.choice(pool)
    target_list = inputs if which == "in" else outputs
    ref = target_list[index]
    if rng.random() < 0.5:
        target_list[index] = PortRef(NodeId(999999), ref.direction, ref.index)
    else:
        target_list[index] = PortRef(ref.node_id, ref.direction, ref.index + 50)
    diagram.set_boundary_inputs(inputs)
    diagram.set_boundary_outputs(outputs)
    return True


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
            if not (
                _is_cleanly_contractible(pre_concrete) and _is_cleanly_contractible(post_concrete)
            ):
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
                issue, frozenset(forced.step.consumed_node_ids), forced.step.port_mapping
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
        # Phase 5 post-closing audit round 22: "validate(d).is_valid implies every node in
        # d is denotable" (module docstrings of qufzx.diagram.validate and
        # qufzx.semantics.denote) is asserted as a property here, not only in
        # tests/test_phase5_exhaustive_oracle.py's exhaustive-but-single-dim-per-node sweep
        # -- this harness's mixed-leg-dimension diagrams (see _build_random_diagram's
        # ``mixed`` branch) are exactly the shape Defect 1 (a validator that let a jointly-
        # unsatisfiable leg/phase disagreement through) needed to be observed at all, so this
        # is where a future gap of that same shape would actually be caught.
        for node in pre_concrete.nodes.values():
            try:
                denote(node)
            except DenoteError as exc:  # pragma: no cover - invariant guard
                raise AssertionError(
                    f"seed {seed}, d={d_value}, e={e_value}: validate(pre_concrete).is_valid "
                    f"but denote() raised for node {node.id!r} "
                    f"({node.generator_type.name}, {node.num_inputs} in / "
                    f"{node.num_outputs} out, phase={node.phase!r}): {exc}"
                ) from exc
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

def _resolved_dim(dim: Dim, bindings: Mapping[str, Dim]) -> Dim:
    """``dim`` with every concrete entry of ``bindings`` substituted in, non-concrete dropped."""
    concrete: dict[str | Dim, int | Dim] = {
        name: value for name, value in bindings.items() if value.is_concrete
    }
    return dim.substitute(concrete) if concrete else dim


def _assert_constraints_satisfiable(
    constraints: Iterable[DimensionConstraint], bindings: Mapping[str, Dim]
) -> None:
    """Every recorded ``(assumed, equal_to)`` pair, resolved under ``bindings``, must unify."""
    for entry in constraints:
        resolved_assumed = _resolved_dim(entry.assumed, bindings)
        resolved_equal_to = _resolved_dim(entry.equal_to, bindings)
        assert not resolved_assumed.unify(resolved_equal_to).is_failure, (
            f"recorded constraint {entry} is not simultaneously satisfiable with the rest "
            f"of match.bindings={dict(bindings)!r}"
        )


def _assert_match_structurally_satisfiable(diagram: Diagram, match: FusionMatch) -> None:
    """N1's required structural-satisfiability arm (cheap, broad, no oracle call).

    Every surviving leg dim, the connecting pair's own two dims, and every present phase dim
    -- each resolved under ``match.bindings`` -- must unify with ``match.shared_dim`` without
    ``FAILURE``, and ``match.dimension_constraints`` must be simultaneously satisfiable. This
    is exactly :func:`~qufzx.rewrite.match._verify_fixpoint_closure`'s own check, re-derived
    independently here (not by importing and calling that private function) as a genuine
    cross-check against a regression in the fixpoint's own termination or bindings-merge
    logic (D1, Phase 5 audit round 15) -- not a tautological re-assertion of it.
    """
    bindings = dict(match.bindings)
    node_a = diagram.nodes[match.a_id]
    node_b = diagram.nodes[match.b_id]
    ref_a = match.wire.a if match.wire.a.node_id == match.a_id else match.wire.b
    ref_b = match.wire.b if match.wire.a.node_id == match.a_id else match.wire.a

    for node, node_id, consumed_ref in (
        (node_a, match.a_id, ref_a),
        (node_b, match.b_id, ref_b),
    ):
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index, port in enumerate(node.legs(direction)):
                ref = PortRef(node_id, direction, index)
                if ref == consumed_ref:
                    continue
                resolved = _resolved_dim(port.dim, bindings)
                assert not resolved.unify(match.shared_dim).is_failure, (
                    f"surviving leg {ref} (resolved {resolved}) does not unify with "
                    f"shared_dim {match.shared_dim}"
                )

    legs_a = node_a.legs(ref_a.direction)
    legs_b = node_b.legs(ref_b.direction)
    for dim in (legs_a[ref_a.index].dim, legs_b[ref_b.index].dim):
        resolved = _resolved_dim(dim, bindings)
        assert not resolved.unify(match.shared_dim).is_failure, (
            f"connecting-pair leg (resolved {resolved}) does not unify with shared_dim "
            f"{match.shared_dim}"
        )

    for node in (node_a, node_b):
        if node.phase is None:
            continue
        resolved = _resolved_dim(node.phase.dim, bindings)
        assert not resolved.unify(match.shared_dim).is_failure, (
            f"phase dim (resolved {resolved}) does not unify with shared_dim "
            f"{match.shared_dim}"
        )

    _assert_constraints_satisfiable(match.dimension_constraints, bindings)


def _bindings_as_int_subs(bindings: Mapping[str, Dim]) -> dict[str, int]:
    return {name: dim.to_int() for name, dim in bindings.items() if dim.is_concrete}


def _isolate_match_pair(diagram: Diagram, match: FusionMatch) -> Diagram:
    """A fresh 2-node diagram containing only ``match``'s own pair, its connecting wire, and
    every surviving leg as a boundary port.

    :func:`~qufzx.rewrite.match.resolve_fusion_match` decides everything about a match from
    ``(diagram, a_id, b_id, wire)`` alone (see that function's own docstring), so this
    isolated diagram reproduces the identical match -- but with any unrelated third node
    ``_build_random_diagram`` may also have generated (and any dimension symbol it happens
    to share with this pair) removed entirely. This is what lets
    ``TestBindingsSubstitutionIsCleanAndOracleEqual`` assert whole-diagram clean
    contractibility under ``match.bindings`` without being confounded by some unrelated
    node's own pre-existing conflict becoming concrete at the same substitution -- a false
    positive this test found in its first draft (seed 508, 1266: an unrelated node's phase
    legitimately disagreed with its own legs only once a symbol *this* match's bindings
    happened to also use was forced concrete, which is `_build_random_diagram` sharing
    symbols across unrelated nodes, not a defect in the fusion under test).
    """
    node_a = diagram.nodes[match.a_id]
    node_b = diagram.nodes[match.b_id]
    isolated = Diagram()
    a_id = isolated.add_node(
        node_a.generator_type,
        input_dims=[p.dim for p in node_a.inputs],
        output_dims=[p.dim for p in node_a.outputs],
        phase=node_a.phase,
    )
    b_id = isolated.add_node(
        node_b.generator_type,
        input_dims=[p.dim for p in node_b.inputs],
        output_dims=[p.dim for p in node_b.outputs],
        phase=node_b.phase,
    )
    ref_a = match.wire.a if match.wire.a.node_id == match.a_id else match.wire.b
    ref_b = match.wire.b if match.wire.a.node_id == match.a_id else match.wire.a
    isolated.add_wire(
        PortRef(a_id, ref_a.direction, ref_a.index), PortRef(b_id, ref_b.direction, ref_b.index)
    )
    boundary_inputs: list[PortRef] = []
    boundary_outputs: list[PortRef] = []
    for new_id, orig_node, consumed_ref in ((a_id, node_a, ref_a), (b_id, node_b, ref_b)):
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index in range(len(orig_node.legs(direction))):
                if direction == consumed_ref.direction and index == consumed_ref.index:
                    continue
                ref = PortRef(new_id, direction, index)
                (boundary_inputs if direction is Direction.INPUT else boundary_outputs).append(
                    ref
                )
    isolated.set_boundary_inputs(boundary_inputs)
    isolated.set_boundary_outputs(boundary_outputs)
    return isolated


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


_MIN_MALFORMED_BOUNDARY_HITS = 10
"""Floor for how many seeds' diagrams actually got a malformed boundary ref and were
correctly rejected by ``find_matches`` -- see :func:`_maybe_corrupt_a_boundary_ref`. Without
this, a change that silently broke the corruption path (e.g. always corrupting into a
still-valid ref) could leave ``test_random_diagrams_fuse_soundly`` passing vacuously on this
arm. Set well below the ~ ``_MALFORMED_BOUNDARY_PROBABILITY * len(_SEEDS)`` expectation
(~75 over 2,500 seeds), leaving headroom for seeds whose diagram has no boundary entry at
all to corrupt."""


def _find_matches_tolerating_malformed_boundary(
    diagram: Diagram, seed: int
) -> tuple[FusionMatch, ...] | None:
    """``find_matches(diagram)``, or ``None`` if it raised for the malformed-boundary reason
    :func:`_maybe_corrupt_a_boundary_ref` deliberately introduces.

    Shared by every property-harness arm that iterates ``_build_random_diagram``'s output
    (Phase 5 post-closing audit round 18, Defect 2): once that generator sometimes produces
    a diagram with a malformed boundary entry, every consumer of it must handle
    ``find_matches`` rejecting the whole diagram outright, not only the one arm
    (``test_random_diagrams_fuse_soundly``) that motivated the widening. Re-raises any other
    exception, and re-raises if the message does not match the expected malformed-reference
    wording, so an unrelated regression is never mistaken for this deliberate corruption.
    """
    try:
        return find_matches(diagram)
    except RewriteGrammarError as exc:
        assert "out of range" in str(exc) or "absent from the diagram" in str(exc), (
            f"seed {seed}: find_matches raised RewriteGrammarError for an unexpected "
            f"reason: {exc}"
        )
        return None


_CONTENDED_SEEDS: tuple[int, ...] = tuple(range(20000))
"""Seeds for :func:`_build_contended_diagram`. See :data:`_MIN_CONDITION5_REJECTIONS`."""


def _build_contended_diagram(rng: random.Random) -> Diagram:
    """A clean diagram whose ports are deliberately *contended*: wired twice, or wired and
    on a boundary.

    Round 24. Every other generator in this module wires ports by popping them out of a
    pool (:func:`_build_random_diagram`, :func:`_build_clean_diagram`,
    :func:`_build_mixed_diagram` all do), so a port is claimed by at most one wire and the
    boundary lists are exactly the leftovers. That is a perfectly reasonable shape for a
    *well-formed* diagram -- and it means ``consumed_ports_singly_claimed``, the seventh
    side condition (:mod:`qufzx.rewrite.match`'s condition 5, promoted from a bare
    ``find_matches`` filter to a real certificate-visible condition in round 23), could
    never fail for any candidate this module generated. Round 23's headline change was
    therefore exercised only by the hand-picked unit tests in ``test_match.py``, never at
    property-harness scale.

    This generator closes that gap by sampling wire endpoints *with* replacement across
    wires (so one port can be claimed by two wires, a
    :class:`~qufzx.diagram.validate.IssueKind.PORT_WIRED_TWICE`) and by putting boundary
    entries on already-wired ports (a
    :class:`~qufzx.diagram.validate.IssueKind.PORT_WIRED_AND_BOUNDARY`). Both are hard
    validation errors, which is exactly the point: :func:`~qufzx.rewrite.match.find_matches`
    does not require a well-formed diagram (see its own docstring), so it must decide this
    condition itself rather than lean on a validity precondition it never asserts.

    Otherwise deliberately kept as close to :func:`_build_clean_diagram` as possible -- one
    concrete dimension shared by every leg of every node, fully concrete phases -- so a
    match that *is* found still reaches :func:`~qufzx.semantics.check.compare` with an empty
    assignment, and this arm tests condition 5 rather than re-testing dimension resolution.
    """
    dim = Dim.concrete(rng.choice(_CLEAN_DIM_VALUES))
    diagram = Diagram()
    ports: list[PortRef] = []
    for node_index in range(rng.randint(2, 3)):
        n_in, n_out = rng.randint(0, 2), rng.randint(0, 2)
        phase = _random_clean_phase(rng, dim, node_index)
        if n_in == 0 and n_out == 0 and phase is None:
            phase = PhaseVector(dim, {})
        node_id = diagram.add_node(
            rng.choice(_COLORS),
            input_dims=[dim] * n_in,
            output_dims=[dim] * n_out,
            phase=phase,
        )
        ports.extend(PortRef(node_id, Direction.INPUT, i) for i in range(n_in))
        ports.extend(PortRef(node_id, Direction.OUTPUT, i) for i in range(n_out))
    if len(ports) < 2:
        return diagram

    # With replacement across wires: a port already used by an earlier wire can be picked
    # again, which is precisely the PORT_WIRED_TWICE shape condition 5 must refuse.
    # Diagram._wires is a set and Wire equality is endpoint-set equality, so a repeat of the
    # *same* pair collapses harmlessly rather than producing a duplicate wire.
    for _ in range(rng.randint(1, 3)):
        a, b = rng.sample(ports, 2)
        diagram.add_wire(a, b)

    wired = {ref for wire in diagram.wires for ref in (wire.a, wire.b)}
    boundary_inputs: list[PortRef] = []
    boundary_outputs: list[PortRef] = []
    for ref in ports:
        # An unwired port always goes on its boundary (as every other generator here does);
        # a wired one goes on it too, sometimes -- the PORT_WIRED_AND_BOUNDARY shape.
        if ref in wired and rng.random() >= 0.4:
            continue
        target = boundary_inputs if ref.direction is Direction.INPUT else boundary_outputs
        target.append(ref)
    diagram.set_boundary_inputs(boundary_inputs)
    diagram.set_boundary_outputs(boundary_outputs)
    return diagram


_CONDITION_5_NAME = "consumed_ports_singly_claimed"


def _condition5_outcome(
    diagram: Diagram, a_id: NodeId, b_id: NodeId, wire: Wire
) -> SideConditionOutcome | None:
    """``consumed_ports_singly_claimed``'s outcome, or ``None`` if it was never evaluated.

    :func:`~qufzx.rewrite.match.resolve_fusion_match` short-circuits: once a condition fails,
    every later one is still *reported* (``outcomes`` always has exactly seven entries) but
    with ``passed=False`` and a "not evaluated: ... failed first" detail. Cross-checking
    condition 5's verdict against diagram contention is only meaningful when it was actually
    decided, so this returns ``None`` whenever an earlier condition failed first -- reading
    that off the preceding outcomes rather than re-deciding the earlier conditions here.
    """
    resolution = match_module.resolve_fusion_match(diagram, a_id, b_id, wire)
    index = next(
        i for i, entry in enumerate(resolution.outcomes) if entry.name == _CONDITION_5_NAME
    )
    if not all(entry.passed for entry in resolution.outcomes[:index]):
        return None
    return resolution.outcomes[index]


def _consumed_refs(a_id: NodeId, b_id: NodeId, wire: Wire) -> tuple[PortRef, PortRef]:
    """``wire``'s two endpoints, ordered ``(on a_id, on b_id)``."""
    if wire.a.node_id == a_id:
        return wire.a, wire.b
    return wire.b, wire.a


def _port_is_contended(diagram: Diagram, ref: PortRef, consuming_wire: Wire) -> bool:
    """Independently: is ``ref`` claimed by a second wire, or listed on a boundary?

    Re-derived here rather than by importing
    :func:`~qufzx.rewrite.match._consumed_port_claim_conflict`, so this arm cross-checks the
    matcher's verdict against a separate computation instead of restating it.
    """
    other_wires = sum(
        1 for wire in diagram.wires if wire != consuming_wire and ref in (wire.a, wire.b)
    )
    on_boundary = ref in diagram.boundary_inputs or ref in diagram.boundary_outputs
    return bool(other_wires) or on_boundary


_MIN_CONDITION5_REJECTIONS = 2000
"""Floor for how many candidates ``consumed_ports_singly_claimed`` actually *rejects*.

Without a floor this arm would pass vacuously the moment a generator change stopped
producing contended ports at all -- the exact failure mode
:data:`_MIN_CLEAN_ORACLE_COMPARISONS` and :data:`_MIN_MALFORMED_BOUNDARY_HITS` exist to
prevent for their own paths. Counted only over candidates where condition 5 was genuinely
*evaluated* (every earlier condition passed -- see :func:`_condition5_outcome`), never over
the short-circuited "not evaluated" reports, which would inflate this number with candidates
that failed on colour or direction instead. Measured at 6,663 such rejections over
``_CONTENDED_SEEDS``' 20,000 seeds; 2,000 leaves wide headroom for incidental generator
tuning while still failing loudly if this arm's whole reason for existing evaporated."""

_MIN_CONDITION5_ACCEPTANCES = 400
"""Floor for how many candidates ``consumed_ports_singly_claimed`` actually *passes*.

The other half of the same anti-vacuity guard: an arm in which condition 5 rejected
*everything* would prove only that the resolver can say no, never that it still says yes for
a legitimately single-claimed consumed port sitting in a diagram that is contended
elsewhere. Measured at 1,208 acceptances over the same seed range."""

_MIN_CONDITION5_ORACLE_COMPARISONS = 250
"""Floor for oracle comparisons on this arm specifically.

Deliberately modest, and *not* this arm's point -- oracle equality at scale is
``test_clean_diagrams_fuse_soundly``'s job (thousands of comparisons). Most diagrams this
generator builds are hard-invalid *somewhere*, so :func:`~qufzx.semantics.check.compare`
refuses them outright (``ContractValidationError``) and the rewrite simply cannot be scored.
Measured at 807 comparisons against 401 such skips; a floor of 250 pins that the seam
between "contended enough to exercise condition 5" and "still contractible end to end" has
not closed entirely, which is what would happen if this generator drifted toward producing
only unscoreable diagrams."""


class TestSpiderFusionProperties:
    def test_random_diagrams_fuse_soundly(self) -> None:
        # Phase 5 post-closing audit round 18, Defect 3: prove _verify_fixpoint_closure's
        # own "this is unreachable" claim actually holds, over this property harness's own
        # (deliberately messy, mixed-dimension) diagram population -- not only over the
        # exhaustive module's fully-concrete finite space (see
        # test_phase5_exhaustive_oracle.py's identical instrumentation). This population
        # exercises the deferred/binding fixpoint path the exhaustive sweep structurally
        # cannot (see that module's own "Scope boundary" docstring section), so this is a
        # genuinely different slice of coverage for the same claim, not a repeat of it.
        closure_results: list[bool] = []
        real_closure = match_module._verify_fixpoint_closure

        def _wrapped_closure(*args: object, **kwargs: object) -> bool:
            result = real_closure(*args, **kwargs)  # type: ignore[arg-type]
            closure_results.append(result)
            return result

        # Phase 5 post-closing audit round 23, Task 4: _merge_bindings' contradiction guard
        # is claimed (in its own docstring) to be currently unreachable, for a structural
        # reason (every operand is pre-resolved through _resolve_with_bindings before it
        # ever reaches Dim.unify, so an already-bound symbol can never come back as a fresh
        # binding key). Pinned here the same way _verify_fixpoint_closure's own
        # unreachability claim is pinned just above: wrap the real function, assert it is
        # actually exercised, and assert it never returns False across this sweep. A Phase
        # 10 change to Dim.unify's contract that makes this reachable should fail this
        # assertion rather than pass silently. _merge_bindings returns None on a clean
        # merge, (name, existing, new) on a contradictory rebind (Task 3) -- "no hit" is
        # `result is None`, not `result is truthy`.
        merge_bindings_results: list[object] = []
        real_merge_bindings = match_module._merge_bindings

        def _wrapped_merge_bindings(*args: object, **kwargs: object) -> object:
            result = real_merge_bindings(*args, **kwargs)  # type: ignore[arg-type]
            merge_bindings_results.append(result)
            return result

        with (
            patch.object(match_module, "_verify_fixpoint_closure", _wrapped_closure),
            patch.object(match_module, "_merge_bindings", _wrapped_merge_bindings),
        ):
            self._run_random_diagrams_fuse_soundly()

        assert closure_results, (
            "_verify_fixpoint_closure was never called at all -- this instrumentation "
            "would then be vacuously proving nothing"
        )
        assert all(closure_results), (
            f"_verify_fixpoint_closure returned False {closure_results.count(False)} "
            f"time(s) out of {len(closure_results)} calls across the random property "
            "harness -- its own docstring's unreachability claim does not actually hold"
        )
        assert merge_bindings_results, (
            "_merge_bindings was never called at all -- this instrumentation would then be "
            "vacuously proving nothing"
        )
        conflicts = [r for r in merge_bindings_results if r is not None]
        assert not conflicts, (
            f"_merge_bindings reported a contradictory rebind {len(conflicts)} time(s) out "
            f"of {len(merge_bindings_results)} calls across the random property harness "
            f"({conflicts!r}) -- its own docstring's unreachability claim does not actually "
            "hold"
        )

    def _run_random_diagrams_fuse_soundly(self) -> None:
        checked_any_match = False
        total_oracle_comparisons = 0
        malformed_boundary_hits = 0
        for seed in _SEEDS:
            rng = random.Random(seed)
            diagram = _build_random_diagram(rng)
            matches = _find_matches_tolerating_malformed_boundary(diagram, seed)
            if matches is None:
                # Defect 2 (Phase 5 post-closing audit round 18): _build_random_diagram
                # sometimes corrupts a boundary entry into a malformed PortRef (see
                # _maybe_corrupt_a_boundary_ref); find_matches must reject the whole
                # diagram outright, before ever returning a match, exactly as it already
                # does for a malformed wire endpoint. This is the property-harness-scale
                # counterpart of test_match.py's TestOutOfRangeBoundaryRefRaises.
                malformed_boundary_hits += 1
                continue
            for match in matches:
                checked_any_match = True
                # The match-implies-applicable invariant, asserted directly (Phase 5
                # post-closing audit round 18, Defect 2's acceptance test): for every match
                # find_matches returns, apply() must either succeed outright or raise only
                # the step-8 relative-postcondition marker -- never any other exception,
                # and never a bare crash. _check_one_match already enforces exactly this
                # (its `except RewriteError` branch turns anything else into a hard
                # AssertionError, and its `except RewriteDomainError` branch requires the
                # marker), so this call *is* that direct assertion, not a separate one
                # layered on top of it.
                total_oracle_comparisons += _check_one_match(diagram, match, seed)
        assert checked_any_match, "the generator never produced a single fusion match"
        assert total_oracle_comparisons >= _MIN_ORACLE_COMPARISONS, (
            f"only {total_oracle_comparisons} oracle comparisons actually executed "
            f"(floor is {_MIN_ORACLE_COMPARISONS}); the oracle arm may be silently "
            "skipping every substitution instead of exercising compare()"
        )
        assert malformed_boundary_hits >= _MIN_MALFORMED_BOUNDARY_HITS, (
            f"only {malformed_boundary_hits} seed(s) produced a malformed boundary ref "
            f"correctly rejected by find_matches (floor is {_MIN_MALFORMED_BOUNDARY_HITS}); "
            "the corruption path may have silently stopped firing"
        )

    def test_clean_diagrams_fuse_soundly(self) -> None:
        """Task 4 (Phase 5 closing round): the oracle-equality arm, at real scale.

        Uses :func:`_build_clean_diagram` (cleanly contractible by construction, unlike
        ``_build_random_diagram``'s deliberately mixed-dimension population) so that
        essentially every match reaches :func:`~qufzx.semantics.check.compare`, not just the
        ~57-out-of-2,500-seeds' worth the other arm's own floor documents. Also asserts the
        six colour/direction shapes ``consumed_wire_direction_permitted_for_color`` actually
        permits (see
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

    def test_contended_ports_exercise_consumed_ports_singly_claimed(self) -> None:
        """Round 24: property-scale coverage for the seventh side condition.

        Every other generator in this module wires ports by popping them out of a pool, so
        no port is ever claimed twice and the boundary lists are exactly the leftovers --
        which means ``consumed_ports_singly_claimed`` could not fail for any candidate they
        produce, and round 23's headline change was pinned only by hand-picked unit tests.
        :func:`_build_contended_diagram` produces the shapes that *do* exercise it.

        Four properties, all on the same sweep:

        1. The condition genuinely rejects, at scale (:data:`_MIN_CONDITION5_REJECTIONS`)
           and genuinely accepts, at scale (:data:`_MIN_CONDITION5_ACCEPTANCES`) -- so the
           arm cannot pass vacuously from either direction.
        2. Its verdict is *correct*, per candidate, against an independently re-derived
           notion of contention (:func:`_port_is_contended`) rather than against the
           matcher's own helper restated.
        3. Match-implies-applicable still holds on exactly these diagrams: every match
           :func:`~qufzx.rewrite.match.find_matches` returns applies without raising
           anything but the step-8 relative postcondition -- in particular never
           :func:`~qufzx.rewrite.engine._remap_endpoint`'s "absent from the builder's
           port_mapping" ``RewriteDomainError``, which is the failure condition 5 exists to
           prevent and which a contended consumed port would otherwise trigger.
        4. The rewrite is still oracle-exact on the diagrams that survive.
        """
        rejections = 0
        acceptances = 0
        oracle_comparisons = 0
        oracle_skips = 0
        for seed in _CONTENDED_SEEDS:
            rng = random.Random(seed)
            diagram = _build_contended_diagram(rng)

            # Property 1/2: walk every candidate pair the matcher itself would consider,
            # not only the ones it accepted, so a rejected candidate is inspected too.
            for wire in diagram.wires:
                if wire.a.node_id == wire.b.node_id:
                    continue
                a_id, b_id = match_module._ordered_pair(wire)
                outcome = _condition5_outcome(diagram, a_id, b_id, wire)
                if outcome is None:
                    continue  # an earlier condition failed first; 5 was never decided
                ref_a, ref_b = _consumed_refs(a_id, b_id, wire)
                contended = _port_is_contended(diagram, ref_a, wire) or _port_is_contended(
                    diagram, ref_b, wire
                )
                assert outcome.passed is not contended, (
                    f"seed {seed}: consumed_ports_singly_claimed reported "
                    f"passed={outcome.passed} for wire {wire!r}, but an independent check "
                    f"says contended={contended} (detail: {outcome.detail!r})"
                )
                if outcome.passed:
                    acceptances += 1
                else:
                    rejections += 1

            # Properties 3/4: every *returned* match must still apply cleanly and be
            # oracle-exact, on these deliberately ill-formed diagrams.
            for match in find_matches(diagram):
                try:
                    result = apply(diagram, SPIDER_FUSION, match)
                except RewriteDomainError as exc:
                    assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
                        f"seed {seed}: apply() raised a RewriteDomainError that is not the "
                        f"step-8 relative postcondition: {exc}"
                    )
                    continue
                post = result.diagram
                introduced = _hard_error_kinds(post) - _hard_error_kinds(diagram)
                assert not introduced, (
                    f"seed {seed}: rewrite introduced hard-error issue kind(s) "
                    f"{sorted(k.value for k in introduced)} not present in the input diagram"
                )
                try:
                    comparison = compare(diagram, post, {})
                except ContractSizeError:
                    oracle_skips += 1
                    continue
                except ContractValidationError:
                    # Expected, and the reason this arm's oracle floor is modest rather
                    # than in the thousands: a diagram with a contended port *elsewhere* is
                    # not contractible at all
                    # (``contract`` refuses a hard-invalid input), so the oracle simply has
                    # nothing to say about it. That is a property of the input, not a defect
                    # in the rewrite -- properties 1-3 above still applied to it, and the
                    # oracle-equality property itself is carried at scale by
                    # ``test_clean_diagrams_fuse_soundly``. Counted, never silently dropped.
                    oracle_skips += 1
                    continue
                oracle_comparisons += 1
                assert comparison.matched, f"seed {seed}: oracle mismatch: {comparison.reason}"

        assert rejections >= _MIN_CONDITION5_REJECTIONS, (
            f"consumed_ports_singly_claimed rejected only {rejections} candidate(s) "
            f"(floor is {_MIN_CONDITION5_REJECTIONS}); this arm's whole purpose is to "
            "exercise that rejection at property scale, so a low count means the generator "
            "has stopped producing contended ports and the arm is passing vacuously"
        )
        assert acceptances >= _MIN_CONDITION5_ACCEPTANCES, (
            f"consumed_ports_singly_claimed accepted only {acceptances} candidate(s) "
            f"(floor is {_MIN_CONDITION5_ACCEPTANCES}); an arm that rejects everything "
            "proves only that the resolver can say no, never that it still says yes for a "
            "legitimately single-claimed consumed port"
        )
        assert oracle_comparisons >= _MIN_CONDITION5_ORACLE_COMPARISONS, (
            f"only {oracle_comparisons} oracle comparison(s) actually executed on this arm "
            f"(floor is {_MIN_CONDITION5_ORACLE_COMPARISONS}, {oracle_skips} skipped as "
            "un-contractible or oversized); the generator may have drifted toward producing "
            "only diagrams the oracle refuses outright"
        )

    def test_phase_index_out_of_range_under_binding_is_not_a_match(self) -> None:
        """Regression test for the Task 1 defect: see match.py's module docstring, condition 7.

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


_FOREIGN_SEEDS: tuple[int, ...] = tuple(range(400))
"""Seeds for :class:`TestForeignFusionMatchArm` (B4, Phase 5 round-12 audit).

Reuses :func:`_build_clean_diagram` (fully concrete, cleanly contractible by construction --
see that function's docstring) so every surviving leg of every match is guaranteed to be
either wired or on a boundary, which in turn guarantees the "port_mapping entry removed"
corruption below always has something to break -- a leg that were neither wired nor on a
boundary would make an unmapped surviving port simply never get looked at, silently not
raising anything. A smaller seed range than the other arms' (400, not thousands): this arm
is one ``pytest.raises`` per match per corruption kind, not an oracle comparison, so it does
not need anywhere near the same sample size to give real coverage across shapes/colours/
directions -- and unlike the other arms, false-negative risk here is symmetric (a defect
would show up on the very first match it can reach), not something that needs volume to
surface rarely.
"""


def _all_claimed_passing(match: FusionMatch) -> tuple[SideConditionOutcome, ...]:
    """``match``'s own outcome names, all claimed ``passed=True`` -- a fabricated-passing
    ``side_condition_outcomes`` tuple (B4/A1/A2): this is what lets a corrupted ``shared_dim``,
    ``bindings``, or diagram slip past ``check_side_condition_coverage`` (which only checks
    outcome *names* and passedness, never re-evaluates a predicate) and reach the builder's
    own re-verification via :func:`~qufzx.rewrite.match.resolve_fusion_match`.
    """
    return tuple(
        dataclasses.replace(outcome, passed=True, detail="fabricated: claims to pass")
        for outcome in match.side_condition_outcomes
    )


def _diagram_with_swapped_color(diagram: Diagram, node_id: NodeId) -> Diagram:
    """A copy of ``diagram`` with ``node_id``'s ``generator_type`` flipped to the other colour.

    ``Node`` is immutable (see ``graph.py``) and ``Diagram`` exposes no public API to change
    a node's colour in place -- deliberately, since no legitimate rewrite ever does this.
    Reaching into ``Diagram``'s private ``_nodes`` dict here is the same posture every other
    "hand-built or foreign" test fixture in this suite takes: constructing a diagram no real
    builder could ever produce, specifically to prove the untrusted-input path rejects it.
    """
    copied = diagram.copy()
    node = copied.nodes[node_id]
    swapped_color = X_SPIDER if node.generator_type is Z_SPIDER else Z_SPIDER
    swapped_node = dataclasses.replace(node, generator_type=swapped_color)
    copied._nodes[node_id] = swapped_node
    return copied


class TestForeignFusionMatchArm:
    """B4 (Phase 5 round-12 audit): for every match a legitimate diagram produces, construct
    corrupted variants of the match/diagram/BuildResult reaching ``spider_fusion_builder`` or
    ``apply``, and assert every one raises a :class:`RewriteError` subclass rather than
    silently producing a wrong diagram. A1-A3 existed because nothing before this ever
    exercised this untrusted-input path -- the trusted (``find_matches``-produced) path has
    been fuzzed to death elsewhere in this file.
    """

    def _matches(self) -> list[tuple[Diagram, FusionMatch]]:
        pairs: list[tuple[Diagram, FusionMatch]] = []
        for seed in _FOREIGN_SEEDS:
            rng = random.Random(seed)
            diagram = _build_clean_diagram(rng)
            for match in find_matches(diagram):
                pairs.append((diagram, match))
        return pairs

    def test_wrong_shared_dim_is_rejected(self) -> None:
        pairs = self._matches()
        assert pairs, "the clean generator never produced a single fusion match"
        for diagram, match in pairs:
            bogus_dim = Dim.concrete(match.shared_dim.to_int() + 1)
            corrupted = dataclasses.replace(
                match, shared_dim=bogus_dim, side_condition_outcomes=_all_claimed_passing(match)
            )
            with pytest.raises(RewriteError):
                spider_fusion_builder(diagram, corrupted)

    def test_wrong_bindings_is_rejected(self) -> None:
        pairs = self._matches()
        assert pairs, "the clean generator never produced a single fusion match"
        checked = 0
        for diagram, match in pairs:
            # Only meaningful when the fusion actually could carry a binding (a symbolic
            # dimension participates) -- _build_clean_diagram is entirely concrete, so
            # instead a bogus binding is *introduced* where none was assumed, which the
            # builder must still catch since it disagrees with what resolve_fusion_match
            # independently derives (empty, for an all-concrete diagram).
            corrupted = dataclasses.replace(
                match,
                bindings={"d": Dim.concrete(999)},
                side_condition_outcomes=_all_claimed_passing(match),
            )
            with pytest.raises(RewriteError):
                spider_fusion_builder(diagram, corrupted)
            checked += 1
        assert checked > 0

    def test_differing_generator_type_is_rejected(self) -> None:
        pairs = self._matches()
        assert pairs, "the clean generator never produced a single fusion match"
        for diagram, match in pairs:
            corrupted_diagram = _diagram_with_swapped_color(diagram, match.b_id)
            corrupted_match = dataclasses.replace(
                match, side_condition_outcomes=_all_claimed_passing(match)
            )
            with pytest.raises(RewriteError):
                spider_fusion_builder(corrupted_diagram, corrupted_match)

    def test_duplicated_consumed_node_ids_is_rejected(self) -> None:
        pairs = self._matches()
        assert pairs, "the clean generator never produced a single fusion match"
        for diagram, match in pairs:
            working = diagram.copy()
            legitimate = spider_fusion_builder(working, match)
            duplicated = dataclasses.replace(
                legitimate, consumed_node_ids=(legitimate.consumed_node_ids[0],) * 2
            )

            def _builder(
                _working: Diagram, _match: Match, result: BuildResult = duplicated
            ) -> BuildResult:
                return result

            rule = dataclasses.replace(SPIDER_FUSION, builder=_builder)
            with pytest.raises(RewriteError):
                apply(diagram, rule, match)

    def test_removed_port_mapping_entry_is_rejected(self) -> None:
        pairs = self._matches()
        checked = 0
        for diagram, match in pairs:
            working = diagram.copy()
            legitimate = spider_fusion_builder(working, match)
            if not legitimate.port_mapping:
                continue
            removed_key = next(iter(legitimate.port_mapping))
            shrunk_mapping = {
                k: v for k, v in legitimate.port_mapping.items() if k != removed_key
            }
            corrupted = dataclasses.replace(legitimate, port_mapping=shrunk_mapping)

            def _builder(
                _working: Diagram, _match: Match, result: BuildResult = corrupted
            ) -> BuildResult:
                return result

            rule = dataclasses.replace(SPIDER_FUSION, builder=_builder)
            with pytest.raises(RewriteError):
                apply(diagram, rule, match)
            checked += 1
        assert checked > 0, (
            "no match in the clean generator's sample had a non-empty port_mapping -- "
            "this arm never actually exercised the removed-entry corruption"
        )

    def test_fabricated_dimension_constraints_is_rejected(self) -> None:
        """Defect 2 (Phase 5 post-closing audit): a match's own ``dimension_constraints``
        is never trusted for the certificate -- it must agree exactly with what
        ``resolve_fusion_match`` derives fresh, or the builder refuses outright.
        """
        pairs = self._matches()
        assert pairs, "the clean generator never produced a single fusion match"
        for diagram, match in pairs:
            fabricated = DimensionConstraint(
                assumed=Dim.concrete(2),
                equal_to=Dim.concrete(3),
                source=ConstraintSource.connecting_pair(),
                outcome=ConstraintOutcome.BOUND,
                # Round 20, Task 6: DimensionConstraint.__post_init__ now structurally
                # requires a non-empty bound_here for a BOUND outcome (this fixture
                # previously relied on the unenforced invariant, constructing a BOUND entry
                # with none). This binding is itself nonsensical (2 == 3 binds nothing real)
                # -- fine, since the whole point of this fixture is that it is fabricated and
                # must be rejected regardless of its content.
                bound_here=(("d", Dim.concrete(3)),),
            )
            corrupted = dataclasses.replace(
                match,
                dimension_constraints=match.dimension_constraints + (fabricated,),
                side_condition_outcomes=_all_claimed_passing(match),
            )
            with pytest.raises(RewriteError):
                spider_fusion_builder(diagram, corrupted)


class TestStructuralSatisfiabilityAtScale:
    """N1's required structural-satisfiability arm, run broadly (cheap, no oracle call) over
    the deliberately messy ``_build_random_diagram`` population -- this is what should have
    caught D1 (Phase 5 audit round 15): every match ``find_matches`` returns must have every
    surviving leg, the connecting pair, and every present phase resolve, under
    ``match.bindings``, to something that unifies with ``match.shared_dim`` -- and the same
    must hold of the *applied* ``RewriteStep.dimension_constraints``, not only the match's
    own pre-apply fields, so a divergence introduced between matching and building would
    still be caught.
    """

    def test_every_match_and_every_applied_step_is_structurally_satisfiable(self) -> None:
        checked_any_match = False
        for seed in _SEEDS:
            rng = random.Random(seed)
            diagram = _build_random_diagram(rng)
            matches = _find_matches_tolerating_malformed_boundary(diagram, seed)
            if matches is None:
                # See _find_matches_tolerating_malformed_boundary's docstring: Defect 2,
                # Phase 5 post-closing audit round 18.
                continue
            for match in matches:
                checked_any_match = True
                _assert_match_structurally_satisfiable(diagram, match)
                try:
                    result = apply(diagram, SPIDER_FUSION, match)
                except RewriteDomainError as exc:
                    assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
                        f"seed {seed}: unexpected RewriteDomainError: {exc}"
                    )
                    continue
                _assert_constraints_satisfiable(
                    result.step.dimension_constraints, dict(match.bindings)
                )
        assert checked_any_match, "the generator never produced a single fusion match"


_MIN_BINDINGS_ORACLE_COMPARISONS = 55
"""Floor for the total oracle comparisons run by substituting ``match.bindings`` itself.

N1's required non-skippable invariant arm exists precisely because the fixed
``_ORACLE_DIM_PAIRS`` palette used by :func:`_check_one_match` never tries the substitution
that actually matters -- ``match.bindings`` itself -- so a D1-shaped defect (a match whose
own recorded assumptions are jointly unsatisfiable) was invisible to it by construction: a
diagram exhibiting D1 fails ``_is_cleanly_contractible`` at its own bindings and would, pre
this arm, only ever be silently ``continue``d past. Measured over ``_SEEDS``'s 2,500 seeds
(most matches either have all-symbolic bindings that don't fully concretize the diagram --
counted as ``unconstrained_skips``, a legitimate skip per this arm's own contract -- or trip
``ContractSizeError``). Set at roughly 70% of that measurement, leaving headroom against
incidental generator tuning while still failing hard if this arm's oracle calls silently
stopped running or its skip reasons silently widened to swallow everything.
"""


class TestBindingsSubstitutionIsCleanAndOracleEqual:
    """N1's required non-skippable invariant arm: for every match ``find_matches`` returns,
    substituting ``match.bindings`` into the input diagram must yield a cleanly contractible
    diagram, and pre/post must be exactly oracle-equal there. A failure here is an ``assert``,
    never a ``continue`` -- the only legitimate skips are a genuine oracle-scale limit
    (``ContractSizeError``) or free symbols ``match.bindings`` does not constrain (both
    counted, with a coverage floor on the oracle-comparison count so this arm cannot silently
    degenerate to only ever skipping).
    """

    def test_every_match_bindings_substitution_is_clean_and_oracle_equal(self) -> None:
        checked_any_match = False
        oracle_runs = 0
        unconstrained_skips = 0
        size_skips = 0
        for seed in _SEEDS:
            rng = random.Random(seed)
            diagram = _build_random_diagram(rng)
            matches = _find_matches_tolerating_malformed_boundary(diagram, seed)
            if matches is None:
                # See _find_matches_tolerating_malformed_boundary's docstring: Defect 2,
                # Phase 5 post-closing audit round 18.
                continue
            for match in matches:
                checked_any_match = True

                # Isolate the pair before doing anything else: see _isolate_match_pair for
                # why the full (possibly multi-node) generated diagram is not used directly.
                isolated = _isolate_match_pair(diagram, match)
                isolated_matches = find_matches(isolated)
                assert len(isolated_matches) == 1, (
                    f"seed {seed}: isolating the matched pair changed match count to "
                    f"{len(isolated_matches)}"
                )
                isolated_match = isolated_matches[0]
                assert isolated_match.shared_dim == match.shared_dim
                assert dict(isolated_match.bindings) == dict(match.bindings)

                try:
                    result = apply(isolated, SPIDER_FUSION, isolated_match)
                except RewriteDomainError as exc:
                    assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
                        f"seed {seed}: unexpected RewriteDomainError: {exc}"
                    )
                    continue
                post = result.diagram

                bindings_subs = _bindings_as_int_subs(isolated_match.bindings)
                remaining = (_free_symbol_names(isolated) | _free_symbol_names(post)) - set(
                    bindings_subs
                )
                if remaining:
                    # A legitimate skip (per this arm's own declared contract, not a
                    # silently-widened escape hatch): match.bindings alone did not fully
                    # concretize the isolated pair, so there is no single substitution this
                    # arm can both derive purely from the match and pass to compare()
                    # unambiguously.
                    unconstrained_skips += 1
                    continue

                try:
                    pre_concrete = isolated.substitute(bindings_subs)
                    post_concrete = post.substitute(bindings_subs)
                except PhaseDomainError as exc:
                    raise AssertionError(
                        f"seed {seed}: match.bindings {bindings_subs!r} substituted into "
                        f"the isolated pair itself produces an invalid phase entry: {exc}"
                    ) from exc

                assert _is_cleanly_contractible(pre_concrete), (
                    f"seed {seed}: substituting match.bindings {bindings_subs!r} into the "
                    "isolated pre-fusion pair is not cleanly contractible -- the match's own "
                    "recorded assumptions do not actually make it well-formed (D1)"
                )
                assert _is_cleanly_contractible(post_concrete), (
                    f"seed {seed}: substituting match.bindings {bindings_subs!r} into the "
                    "isolated post-fusion pair is not cleanly contractible"
                )

                try:
                    comparison = compare(isolated, post, bindings_subs)
                except ContractSizeError:
                    size_skips += 1
                    continue
                assert comparison.matched, (
                    f"seed {seed}: oracle mismatch at match.bindings {bindings_subs!r}: "
                    f"{comparison.reason}"
                )
                oracle_runs += 1
        assert checked_any_match, "the generator never produced a single fusion match"
        assert oracle_runs >= _MIN_BINDINGS_ORACLE_COMPARISONS, (
            f"only {oracle_runs} bindings-substitution oracle comparisons ran (floor is "
            f"{_MIN_BINDINGS_ORACLE_COMPARISONS}, {unconstrained_skips} skipped for "
            f"unconstrained symbols, {size_skips} for ContractSizeError); this arm may be "
            "silently degenerating to only ever skipping instead of exercising compare()"
        )


# Phase 5 post-closing audit round 23, Task 8: the two sweeps that verified round 23 (a
# structural-invariant fuzz and a fresh-seed oracle differential) were written ad hoc and
# thrown away. Promoted into permanent tests here so every future round has to clear them
# too, not just the ones that happened to motivate this round's fixes.

_STRUCTURAL_SEEDS: tuple[int, ...] = tuple(range(40000, 42000))
"""Disjoint from every other seed pool in this module (``_SEEDS`` 0-2499, ``_CLEAN_SEEDS``
0-19999, ``_MIXED_SEEDS`` 0-39999, ``_FOREIGN_SEEDS`` 0-399): starts at 40000, past the
highest end of any of them, so this sweep is not merely re-confirming shapes the matcher and
builder have already been tuned against."""

_MIN_STRUCTURAL_APPLICATIONS = 100
"""Floor for the number of applications actually checked by
:class:`TestStructuralInvariants`, across all three generators over ``_STRUCTURAL_SEEDS`` --
the same "a sweep that silently stops exercising the path is worse than no sweep" discipline
as ``_MIN_ORACLE_COMPARISONS`` above."""


def _check_structural_invariants(diagram: Diagram, match: FusionMatch, seed: int) -> int:
    """Apply ``match`` and assert the structural postconditions Task 8a exists to pin.

    Returns 1 if the application went through and was checked, 0 if it was legitimately
    blocked by the step-8 relative postcondition (see ``_RELATIVE_POSTCONDITION_MARKER`` --
    that carve-out is real and documented, not something this sweep should treat as a
    failure; see ``TestSurvivingLegOverwriteIntroducesDeferral`` in ``test_match.py``).
    """
    node_a = diagram.nodes[match.a_id]
    node_b = diagram.nodes[match.b_id]
    ref_a = match.wire.a if match.wire.a.node_id == match.a_id else match.wire.b
    ref_b = match.wire.b if match.wire.a.node_id == match.a_id else match.wire.a
    expected_inputs = (
        node_a.num_inputs
        + node_b.num_inputs
        - (1 if ref_a.direction is Direction.INPUT else 0)
        - (1 if ref_b.direction is Direction.INPUT else 0)
    )
    expected_outputs = (
        node_a.num_outputs
        + node_b.num_outputs
        - (1 if ref_a.direction is Direction.OUTPUT else 0)
        - (1 if ref_b.direction is Direction.OUTPUT else 0)
    )
    before_node_count = len(diagram.nodes)
    before_boundary_inputs = len(diagram.boundary_inputs)
    before_boundary_outputs = len(diagram.boundary_outputs)
    before_scalar = diagram.scalar

    try:
        result = apply(diagram, SPIDER_FUSION, match)
    except RewriteDomainError as exc:
        assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
            f"seed {seed}: apply() raised a RewriteDomainError that is not the relative "
            f"post-condition, for a match find_matches() itself returned: {exc}"
        )
        return 0

    post = result.diagram
    assert len(post.nodes) == before_node_count - 1, (
        f"seed {seed}: expected exactly one node removed, got "
        f"{before_node_count} -> {len(post.nodes)}"
    )
    assert len(post.boundary_inputs) == before_boundary_inputs, f"seed {seed}: boundary_inputs"
    assert len(post.boundary_outputs) == before_boundary_outputs, f"seed {seed}: boundary_outputs"
    assert post.scalar == before_scalar, (
        f"seed {seed}: scalar changed by a rewrite that never claims to introduce one"
    )

    assert len(result.new_node_ids) == 1
    merged = post.nodes[result.new_node_ids[0]]
    assert merged.num_inputs == expected_inputs, (
        f"seed {seed}: merged node has {merged.num_inputs} inputs, expected {expected_inputs}"
    )
    assert merged.num_outputs == expected_outputs, (
        f"seed {seed}: merged node has {merged.num_outputs} outputs, expected {expected_outputs}"
    )
    for leg in (*merged.inputs, *merged.outputs):
        assert leg.dim == match.shared_dim, (
            f"seed {seed}: merged leg dim {leg.dim} != match.shared_dim {match.shared_dim}"
        )

    for node_id, consumed_ref in ((match.a_id, ref_a), (match.b_id, ref_b)):
        node = diagram.nodes[node_id]
        for direction in (Direction.INPUT, Direction.OUTPUT):
            for index in range(len(node.legs(direction))):
                ref = PortRef(node_id, direction, index)
                if ref == consumed_ref:
                    continue
                assert ref in result.step.port_mapping, (
                    f"seed {seed}: surviving port {ref!r} of a consumed node is missing "
                    "from step.port_mapping"
                )

    # Must neither raise nor crash on the result -- a corrupted merged node (e.g. a leg
    # count/dim mismatch this function's own asserts above did not already catch) could
    # still trip an internal assertion inside find_matches itself.
    find_matches(post)
    return 1


class TestStructuralInvariants:
    """Task 8a: every match this module returns, once applied, must leave the diagram in a
    structurally coherent state -- not merely "oracle-equal at some substitution" (already
    covered by the oracle-differential arms above), but the graph-shape invariants a
    certificate consumer (Phase 6) and a future strategy layer (Phase 11) both need to be
    able to rely on without re-deriving them by hand each time.
    """

    def test_structural_invariants_hold_across_generators(self) -> None:
        total_checked = 0
        for build_diagram in (_build_random_diagram, _build_clean_diagram, _build_mixed_diagram):
            for seed in _STRUCTURAL_SEEDS:
                rng = random.Random(seed)
                diagram = build_diagram(rng)
                try:
                    matches = find_matches(diagram)
                except RewriteGrammarError:
                    # _build_random_diagram sometimes deliberately corrupts a boundary ref
                    # (see its own docstring); find_matches rejects the whole diagram
                    # outright before returning any match, so there is nothing to check.
                    continue
                for match in matches:
                    total_checked += _check_structural_invariants(diagram, match, seed)
        assert total_checked >= _MIN_STRUCTURAL_APPLICATIONS, (
            f"only {total_checked} application(s) were actually checked (floor is "
            f"{_MIN_STRUCTURAL_APPLICATIONS}); the structural sweep may be silently "
            "exercising almost nothing"
        )


_ORACLE_DIFF_SEEDS: tuple[int, ...] = tuple(range(42000, 44000))
"""Disjoint from ``_STRUCTURAL_SEEDS`` too, and from every pinned pool this module already
uses (see that constant's own docstring) -- a genuinely fresh range for Task 8b, not merely
re-confirming seeds the matcher and builder have already been tuned against."""

_MIN_ORACLE_DIFF_COMPARISONS = 500
_MIN_ORACLE_DIFF_EXACT_MATCHES = 500
"""Floors for :class:`TestFreshSeedOracleDifferential`: a comparison-count floor (the arm
actually ran ``compare()``, not silently skipping every match) and an exact-match floor (the
arm actually found agreement, not merely running comparisons that all happened to mismatch
or get skipped for size) -- the same two-floor discipline
``TestOracleTiesBackToRecordedConstraints`` in ``test_phase5_certificate_sweep.py`` uses."""


class TestFreshSeedOracleDifferential:
    """Task 8b: a fresh-seed oracle differential over a range disjoint from every existing
    pinned pool in this module, so the suite is not merely re-confirming the seeds it was
    tuned against. Uses :func:`_build_clean_diagram` (fully concrete by construction) so
    every match reaches :func:`~qufzx.semantics.check.compare` directly, via
    :func:`_check_one_clean_match` -- the same mechanics
    ``TestSpiderFusionProperties::test_clean_diagrams_fuse_soundly_at_real_scale`` already
    uses, over a disjoint seed range instead of a shared one.
    """

    def test_fresh_seeds_agree_with_the_oracle(self) -> None:
        total_comparisons = 0
        total_size_skips = 0
        for seed in _ORACLE_DIFF_SEEDS:
            rng = random.Random(seed)
            diagram = _build_clean_diagram(rng)
            for match in find_matches(diagram):
                comparisons, size_skips = _check_one_clean_match(diagram, match, seed)
                total_comparisons += comparisons
                total_size_skips += size_skips
        assert total_comparisons >= _MIN_ORACLE_DIFF_EXACT_MATCHES, (
            f"only {total_comparisons} exact oracle match(es) over the fresh seed range "
            f"(floor is {_MIN_ORACLE_DIFF_EXACT_MATCHES}, {total_size_skips} skipped for "
            "ContractSizeError); this arm may be silently exercising almost nothing"
        )
        assert total_comparisons >= _MIN_ORACLE_DIFF_COMPARISONS
