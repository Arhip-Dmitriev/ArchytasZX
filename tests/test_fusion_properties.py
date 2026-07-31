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
from collections.abc import Iterable

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase, PhaseDomainError, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import IssueKind, validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import RewriteDomainError, RewriteError
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import compare
from qufzx.semantics.contract_numeric import ContractSizeError

_SEEDS: tuple[int, ...] = tuple(range(2500))
_DIM_PALETTE = (Dim.concrete(2), Dim.concrete(3), Dim.symbol("d"), Dim.symbol("e"))
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
    """Sometimes absent, sometimes concrete, sometimes symbolic; always over ``dim`` --
    or, when ``dim`` is symbolic, sometimes over a concrete value it could unify to."""
    choice = rng.random()
    if choice < 0.3:
        return None
    phase_dim = dim if dim.is_concrete else rng.choice((dim, Dim.concrete(2), Dim.concrete(3)))
    index = _random_phase_index(rng, phase_dim)
    if choice < 0.65:
        return PhaseVector(phase_dim, {index: Phase.turns(rng.choice(_CONCRETE_TURNS))})
    return PhaseVector(phase_dim, {index: Phase.symbol(f"theta_{node_index}")})


def _build_random_diagram(rng: random.Random) -> Diagram:
    """2-4 nodes, 0-3 legs per side, colour Z/X, one dim per node, a random wiring."""
    diagram = Diagram()
    all_ports: list[PortRef] = []
    for node_index in range(rng.randint(2, 4)):
        color = rng.choice(_COLORS)
        dim = rng.choice(_DIM_PALETTE)
        n_in = rng.randint(0, 3)
        n_out = rng.randint(0, 3)
        phase = _random_phase(rng, dim, node_index)
        if n_in == 0 and n_out == 0 and phase is None:
            # A node with no legs and no phase has no port or slot left to carry its
            # dimension at all, so semantics/denote.py cannot resolve it -- an explicit
            # all-zero PhaseVector gives it somewhere to live, mirroring
            # rules_library.py's own "no legs survive" corner case for a merged node.
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


def _hard_error_kinds(diagram: Diagram) -> frozenset[IssueKind]:
    return frozenset(issue.kind for issue in validate(diagram).errors)


def _is_cleanly_contractible(diagram: Diagram) -> bool:
    report = validate(diagram)
    return report.is_valid and not report.deferred


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
        return 0
    except RewriteError as exc:  # pragma: no cover - documents the intended failure mode
        raise AssertionError(f"seed {seed}: unexpected {type(exc).__name__}: {exc}") from exc

    post = result.diagram

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


_MIN_ORACLE_COMPARISONS = 150
"""Floor for the total oracle comparisons summed across every checked match.

Without this, an always-skipped oracle arm (e.g. every substitution failing
``_is_cleanly_contractible`` or raising ``PhaseDomainError``) would let the test pass
while never actually calling :func:`~qufzx.semantics.check.compare` -- see the module
docstring and this file's Task 2 history. The actual count on the current seed list and
generator is around 260; 150 leaves headroom against incidental generator changes while
still failing hard if the oracle arm silently stops running.
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
