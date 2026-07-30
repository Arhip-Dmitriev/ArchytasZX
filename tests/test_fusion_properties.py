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
from qufzx.algebra.phase import Phase, PhaseVector
from qufzx.diagram.generators import X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.diagram.validate import IssueKind, validate
from qufzx.rewrite.engine import apply
from qufzx.rewrite.match import FusionMatch, find_matches
from qufzx.rewrite.rule import RewriteDomainError, RewriteError
from qufzx.rewrite.rules_library import SPIDER_FUSION
from qufzx.semantics.check import compare

_SEEDS: tuple[int, ...] = tuple(range(150))
_DIM_PALETTE = (Dim.concrete(2), Dim.concrete(3), Dim.symbol("d"), Dim.symbol("e"))
_DIM_SYMBOL_NAMES = frozenset({"d", "e"})
_CONCRETE_TURNS = (sp.Integer(0), sp.Rational(1, 3), sp.Rational(2, 5))
_ORACLE_D0_VALUES = (2, 3, 5)
_PHASE_SUBSTITUTE_TURNS = sp.Rational(1, 3)
_COLORS = (Z_SPIDER, X_SPIDER)

# apply()'s own message for the step-8 relative post-condition (see engine.py) -- the one
# RewriteDomainError a matcher-approved match may legitimately raise, since it reflects a
# genuine downstream conflict (e.g. a symbol forced two different concrete values by two
# independent fusions), not a builder bug. Anything else escaping apply() for a match
# find_matches() itself returned is exactly the class of defect this harness exists to
# catch (see defects 1 and 2 in the Phase 5 audit), so it must not be swallowed.
_RELATIVE_POSTCONDITION_MARKER = "rewrite introduced hard-error issue kind"


def _random_phase(rng: random.Random, dim: Dim, node_index: int) -> PhaseVector | None:
    """Sometimes absent, sometimes concrete, sometimes symbolic; always over ``dim`` --
    or, when ``dim`` is symbolic, sometimes over a concrete value it could unify to."""
    choice = rng.random()
    if choice < 0.3:
        return None
    phase_dim = dim if dim.is_concrete else rng.choice((dim, Dim.concrete(2), Dim.concrete(3)))
    if choice < 0.65:
        return PhaseVector(phase_dim, {1: Phase.turns(rng.choice(_CONCRETE_TURNS))})
    return PhaseVector(phase_dim, {1: Phase.symbol(f"theta_{node_index}")})


def _build_random_diagram(rng: random.Random) -> Diagram:
    """2-4 nodes, 0-3 legs per side, colour Z/X, one dim per node, a random wiring."""
    diagram = Diagram()
    all_ports: list[PortRef] = []
    for node_index in range(rng.randint(2, 4)):
        color = rng.choice(_COLORS)
        dim = rng.choice(_DIM_PALETTE)
        n_in = rng.randint(0, 3)
        n_out = rng.randint(0, 3)
        node_id = diagram.add_node(
            color,
            input_dims=[dim] * n_in,
            output_dims=[dim] * n_out,
            phase=_random_phase(rng, dim, node_index),
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


def _substitution_for(names: Iterable[str], d0: int) -> dict[str, int | sp.Rational]:
    """Every dim symbol -> the same ``d0``; every other (phase) symbol -> a fixed rational."""
    return {name: (d0 if name in _DIM_SYMBOL_NAMES else _PHASE_SUBSTITUTE_TURNS) for name in names}


def _hard_error_kinds(diagram: Diagram) -> frozenset[IssueKind]:
    return frozenset(issue.kind for issue in validate(diagram).errors)


def _is_cleanly_contractible(diagram: Diagram) -> bool:
    report = validate(diagram)
    return report.is_valid and not report.deferred


def _check_one_match(diagram: Diagram, match: FusionMatch, seed: int) -> None:
    try:
        result = apply(diagram, SPIDER_FUSION, match)
    except RewriteDomainError as exc:
        assert _RELATIVE_POSTCONDITION_MARKER in str(exc), (
            f"seed {seed}: apply() raised a RewriteDomainError that is not the relative "
            f"post-condition, for a match find_matches() itself returned: {exc}"
        )
        return
    except RewriteError as exc:  # pragma: no cover - documents the intended failure mode
        raise AssertionError(f"seed {seed}: unexpected {type(exc).__name__}: {exc}") from exc

    post = result.diagram

    introduced = _hard_error_kinds(post) - _hard_error_kinds(diagram)
    assert not introduced, (
        f"seed {seed}: rewrite introduced hard-error issue kind(s) "
        f"{sorted(k.value for k in introduced)} not present in the input diagram"
    )

    all_names = _free_symbol_names(diagram) | _free_symbol_names(post)
    for d0 in _ORACLE_D0_VALUES:
        subs = _substitution_for(all_names, d0)
        pre_concrete = diagram.substitute(subs)
        if not _is_cleanly_contractible(pre_concrete):
            continue
        comparison = compare(diagram, post, subs)
        assert comparison.matched, f"seed {seed}, d0={d0}: oracle mismatch: {comparison.reason}"


class TestSpiderFusionProperties:
    def test_random_diagrams_fuse_soundly(self) -> None:
        checked_any_match = False
        for seed in _SEEDS:
            rng = random.Random(seed)
            diagram = _build_random_diagram(rng)
            for match in find_matches(diagram):
                checked_any_match = True
                _check_one_match(diagram, match, seed)
        assert checked_any_match, "the generator never produced a single fusion match"
