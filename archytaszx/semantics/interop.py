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

"""Oracle checks for the scalable notation: round trips, sheet steps, and mixed workflows.

:func:`default_samples` lists concrete assignments: each multiplicity symbol over
``{0, 1, 2}``, each dimension symbol over ``{2, 3}``, each phase or scalar symbol over two
pool values, every value meeting its symbol's sympy assumptions. Diagonals come first, then
the remaining index tuples by (largest index, index sum, tuple), capped at ``limit``; a
final assignment omits the symbols a diagram binds, which then take their bound values.

:func:`check_round_trip` translates a diagram to scalable notation and back, and checks at
each sample that both diagrams instantiate to isomorphic diagrams, that
:func:`~archytaszx.diagram.scalable.strip` is isomorphic to bang-box instantiation, and that
the oracle finds both diagrams exactly equal, scalar included.
:func:`check_sheet_step` replays a :class:`~archytaszx.rewrite.sheet.SheetStep`, validates
its result, and oracle-compares its two sides. :func:`via_scalable` runs sheet operations on
a bang-box diagram and checks the round trip and every step.

A sample outside a side's domain (a domain error, or a value breaking an assumption) is
skipped; any other error fails. Where the reference (the original diagram, or the step's
``before``) does not instantiate, every check skips; where it does not contract, the oracle
skips. Where only the other side is outside its domain, the sample is a one-sided skip;
when every sample the reference evaluates at is one-sided, the first is the
counterexample. Skips are counted in ``reason``. A report is ``ok`` only when at least
one sample was evaluated.
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import DimensionDomainError, DimensionError
from archytaszx.algebra.phase import PhaseDomainError, PhaseError
from archytaszx.algebra.scalar import ScalarDomainError, ScalarError
from archytaszx.diagram.bangbox import (
    BangBoxDomainError,
    BangBoxError,
    expand_concrete_boxes,
    free_mult_symbols,
    instantiate_symbol,
)
from archytaszx.diagram.compare import compare_structure, isomorphic
from archytaszx.diagram.generators import GeneratorDomainError, GeneratorError
from archytaszx.diagram.graph import Diagram, GraphDomainError, GraphError
from archytaszx.diagram.scalable import (
    ScalableDiagram,
    ScalableError,
    from_scalable,
    strip,
    to_scalable,
    validate_scalable,
)
from archytaszx.diagram.validate import ValidateError, validate
from archytaszx.rewrite.rule import RewriteError
from archytaszx.rewrite.sheet import SHEET_OPERATIONS, SheetStep, replay_sheet_step
from archytaszx.semantics.check import (
    DEFAULT_TOLERANCE,
    CheckAssignmentValue,
    CheckDomainError,
    CheckError,
    CheckGrammarError,
    EqualityMode,
    compare,
    score,
)
from archytaszx.semantics.contract_numeric import ContractDomainError, ContractError
from archytaszx.semantics.decide import INTEGER_SAMPLE_POOL, RATIONAL_SAMPLE_POOL
from archytaszx.semantics.denote import DenoteDomainError, DenoteError

Sample = Mapping[str, CheckAssignmentValue]

MULTIPLICITY_VALUES: tuple[int, ...] = (0, 1, 2)
DIMENSION_VALUES: tuple[int, ...] = (2, 3)
OTHER_VALUE_COUNT = 2
DEFAULT_SAMPLE_LIMIT = 12
_MAX_GRID_PRODUCT = 4096

_ORACLE_ERRORS: tuple[type[Exception], ...] = (
    BangBoxError,
    CheckError,
    ContractError,
    DenoteError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    ScalableError,
    ScalarError,
    ValidateError,
)
_DOMAIN_ERRORS: tuple[type[Exception], ...] = (
    BangBoxDomainError,
    CheckDomainError,
    ContractDomainError,
    DenoteDomainError,
    DimensionDomainError,
    GeneratorDomainError,
    GraphDomainError,
    PhaseDomainError,
    ScalarDomainError,
)


class InteropError(Exception):
    """Base class for all errors raised by this module."""


class InteropGrammarError(InteropError):
    """A request is malformed: a wrong argument type, a bad limit, or an unknown operation."""


# -- sampling -----------------------------------------------------------------------------


def _as_diagram(value: object, what: str) -> Diagram:
    if isinstance(value, ScalableDiagram):
        return from_scalable(value)
    if isinstance(value, Diagram):
        return value
    raise InteropGrammarError(f"{what} must be a Diagram or ScalableDiagram, got {value!r}")


def _symbol_roles(diagrams: Sequence[Diagram]) -> tuple[set[str], set[str], set[str]]:
    """(multiplicity, dimension, other) free symbol names of ``diagrams``."""
    mults: set[str] = set()
    dims: set[str] = set()
    others: set[str] = set()
    for diagram in diagrams:
        mults |= free_mult_symbols(diagram)
        others |= diagram.scalar.free_symbols
        for node in diagram.nodes.values():
            for port in (*node.inputs, *node.outputs):
                dims |= port.dim.free_symbols
            if node.phase is not None:
                dims |= node.phase.dim.free_symbols
                others |= node.phase.free_symbols
    dims -= mults
    others -= mults | dims
    return mults, dims, others


def _sympy_symbols(diagrams: Sequence[Diagram]) -> dict[str, list[sp.Symbol]]:
    """Every free sympy symbol of ``diagrams``, grouped by name."""
    exprs: list[sp.Expr] = []
    for diagram in diagrams:
        exprs.append(diagram.scalar.to_sympy())
        exprs.extend(box.multiplicity.to_sympy() for box in diagram.bang_boxes.values())
        for node in diagram.nodes.values():
            exprs.extend(port.dim.to_sympy() for port in (*node.inputs, *node.outputs))
            if node.phase is not None:
                exprs.append(node.phase.dim.to_sympy())
                exprs.extend(p.to_sympy_turns() for p in node.phase.entries().values())
    table: dict[str, list[sp.Symbol]] = {}
    for expr in exprs:
        for symbol in sorted(expr.free_symbols, key=str):
            table.setdefault(str(symbol.name), []).append(symbol)
    return table


def _satisfies(value: CheckAssignmentValue, symbols: Sequence[sp.Symbol]) -> bool:
    expr = sp.sympify(value)
    for symbol in symbols:
        for key, expected in sorted(symbol.assumptions0.items()):
            actual = getattr(expr, f"is_{key}", None)
            if actual is not None and bool(actual) is not bool(expected):
                return False
    return True


def _admissible(sample: Sample, table: Mapping[str, Sequence[sp.Symbol]]) -> bool:
    """Whether every value of ``sample`` meets its symbol's assumptions in ``table``."""
    try:
        return all(_satisfies(value, table.get(name, [])) for name, value in sample.items())
    except (sp.SympifyError, TypeError):
        return False


def _grid(widths: Sequence[int]) -> list[tuple[int, ...]]:
    """Index tuples: diagonals first, then the rest by (largest index, index sum, tuple)."""
    diagonals = [tuple(min(i, w - 1) for w in widths) for i in range(max(widths, default=1))]
    total = math.prod(widths)
    if total <= _MAX_GRID_PRODUCT:
        rest = sorted(
            itertools.product(*(range(w) for w in widths)), key=lambda t: (max(t), sum(t), t)
        )
    else:
        rest = [
            tuple(i if j == position else min(base, w - 1) for j, w in enumerate(widths))
            for base in range(max(widths))
            for position, width in enumerate(widths)
            for i in range(width)
        ]
    ordered: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for index_tuple in (*diagonals, *rest):
        if index_tuple not in seen:
            seen.add(index_tuple)
            ordered.append(index_tuple)
    return ordered


def default_samples(
    *diagrams: Diagram | ScalableDiagram, limit: int = DEFAULT_SAMPLE_LIMIT
) -> tuple[Mapping[str, CheckAssignmentValue], ...]:
    """Deterministic assignments over every free symbol of ``diagrams``, at most ``limit``.

    One empty assignment when there is no free symbol; none when some symbol has no value
    meeting its assumptions. When some diagram binds a sampled symbol in its parameters, the
    last assignment is the second grid row without the bound symbols.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise InteropGrammarError(f"limit must be a positive int, got {limit!r}")
    plain = [_as_diagram(d, "default_samples argument") for d in diagrams]
    mults, dims, others = _symbol_roles(plain)
    table = _sympy_symbols(plain)
    values: dict[str, tuple[CheckAssignmentValue, ...]] = {}
    for name in sorted(mults | dims | others):
        symbols = table.get(name, [])
        if name in mults:
            pool: Sequence[CheckAssignmentValue] = MULTIPLICITY_VALUES
        elif name in dims:
            pool = DIMENSION_VALUES
        elif any(symbol.is_integer for symbol in symbols):
            pool = INTEGER_SAMPLE_POOL
        else:
            pool = RATIONAL_SAMPLE_POOL
        chosen = [v for v in pool if _satisfies(v, symbols)]
        values[name] = tuple(chosen if name in mults | dims else chosen[:OTHER_VALUE_COUNT])
    if not values:
        return (MappingProxyType({}),)
    if not all(values.values()):
        return ()
    names = tuple(values)
    rows = [
        {name: values[name][i] for name, i in zip(names, t, strict=True)}
        for t in _grid([len(values[name]) for name in names])
    ]
    bound = {name for diagram in plain for name in diagram.parameters} & set(names)
    if bound and limit > 1:
        base = rows[min(1, len(rows) - 1)]
        rows = [*rows[: limit - 1], {k: v for k, v in base.items() if k not in bound}]
    return tuple(MappingProxyType(row) for row in rows[:limit])


def _check_samples(samples: object) -> tuple[Sample, ...] | None:
    if samples is None:
        return None
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise InteropGrammarError("samples must be None or a Sequence of Mapping")
    if not all(isinstance(sample, Mapping) for sample in samples):
        raise InteropGrammarError("samples must be None or a Sequence of Mapping")
    return tuple(MappingProxyType(dict(sample)) for sample in samples)


# -- oracle -------------------------------------------------------------------------------


def _expand(diagram: Diagram, sample: Sample) -> Diagram:
    """``diagram`` with every bang box instantiated at ``sample`` over its parameters."""
    env = {**diagram.parameters, **sample}
    working = diagram
    for name in sorted(free_mult_symbols(diagram)):
        if name not in env:
            raise CheckGrammarError(f"multiplicity symbol {name!r} has no value")
        value = env[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise BangBoxDomainError(f"multiplicity symbol {name!r} needs an int, got {value!r}")
        if name in free_mult_symbols(working):
            working = instantiate_symbol(working, name, value)
    return expand_concrete_boxes(working)


def _instantiable(diagram: Diagram, sample: Sample) -> None:
    _expand(diagram, sample)


def _evaluable(diagram: Diagram, sample: Sample) -> None:
    score(diagram, sample)


def _magnitude(diagram: Diagram, sample: Sample) -> float:
    tensor = score(diagram, sample).tensor
    return float(abs(tensor).max()) if tensor.size else 0.0


class _OneSided(Exception):
    """The reference evaluates at a sample and the other side is outside its domain there."""


def _oracle_mismatch(reference: Diagram, other: Diagram, sample: Sample) -> str | None:
    """Why the oracle finds ``reference`` and ``other`` unequal at ``sample``, or None.

    A deviation within the tolerance scaled by the largest entry modulus is a match. Raises
    :class:`_OneSided` on a domain error of ``other``; any other error is a mismatch.
    """
    try:
        result = compare(reference, other, sample, mode=EqualityMode.EXACT)
    except _DOMAIN_ERRORS as exc:
        raise _OneSided(str(exc)) from exc
    except _ORACLE_ERRORS as exc:
        return f"the other side cannot be evaluated: {type(exc).__name__}: {exc}"
    if result.matched:
        return None
    if math.isfinite(result.max_abs_deviation):
        scale = max(1.0, _magnitude(reference, sample))
        if result.max_abs_deviation <= DEFAULT_TOLERANCE * scale:
            return None
    return result.reason


def _strip_problem(diagram: Diagram, scalable: ScalableDiagram, sample: Sample) -> str | None:
    """Why :func:`strip` of ``scalable`` differs from instantiating ``diagram`` at ``sample``."""
    mults = scalable.free_mult_symbols()
    cut = {k: v for k, v in sample.items() if k in mults and isinstance(v, int)}
    try:
        expected = _expand(diagram, sample)
        stripped = strip(scalable, cut)
    except _ORACLE_ERRORS as exc:
        return f"strip or instantiation failed: {exc}"
    if not isomorphic(stripped, expected):
        return "strip is not isomorphic to bang-box instantiation"
    return None


def _family_problem(diagram: Diagram, back: Diagram, sample: Sample) -> str | None:
    """Why instantiating ``back`` and ``diagram`` at ``sample`` gives non-isomorphic results."""
    try:
        expected = _expand(diagram, sample)
        actual = _expand(back, sample)
    except _ORACLE_ERRORS as exc:
        return f"instantiation failed: {exc}"
    if not isomorphic(actual, expected):
        return "instances are not isomorphic"
    return None


def _sample_text(sample: Sample) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in sorted(sample.items())) + "}"


@dataclass(frozen=True, slots=True)
class _Sweep:
    """Evaluated, gated-out, and one-sided sample counts, and the first failure."""

    evaluated: int
    gated: int
    one_sided: int
    counterexample: Sample | None
    failure: str


def _sweep(
    reference: Diagram,
    samples: Sequence[Sample],
    gate: Callable[[Diagram, Sample], None],
    checks: Sequence[Callable[[Sample], str | None]],
) -> tuple[_Sweep, ...]:
    """Run each check at every admissible sample where ``gate`` accepts ``reference``,
    keeping each first failure.

    A domain error of ``gate`` skips the sample and any other error fails every check there.
    A one-sided sample is skipped; a check that skips every such sample fails at the first.
    """
    table = _sympy_symbols([reference])
    evaluated = [0] * len(checks)
    gated = 0
    skipped: list[list[tuple[Sample, str]]] = [[] for _ in checks]
    failures: dict[int, tuple[Sample, str]] = {}
    for sample in samples:
        if len(failures) == len(checks):
            break
        if not _admissible(sample, table):
            gated += 1
            continue
        try:
            gate(reference, sample)
        except _DOMAIN_ERRORS:
            gated += 1
            continue
        except _ORACLE_ERRORS as exc:
            why = f"the reference cannot be evaluated: {type(exc).__name__}: {exc}"
            failures.update({p: (sample, why) for p in range(len(checks)) if p not in failures})
            continue
        for position, check in enumerate(checks):
            if position in failures:
                continue
            try:
                problem = check(sample)
            except _OneSided as exc:
                skipped[position].append((sample, str(exc)))
                continue
            evaluated[position] += 1
            if problem is not None:
                failures[position] = (sample, problem)
    for position, misses in enumerate(skipped):
        if position not in failures and misses and not evaluated[position]:
            sample, why = misses[0]
            failures[position] = (sample, f"the other side cannot be evaluated: {why}")
    return tuple(
        _Sweep(evaluated[position], gated, len(skipped[position]), *failures[position])
        if position in failures
        else _Sweep(evaluated[position], gated, len(skipped[position]), None, "")
        for position in range(len(checks))
    )


def _notes(sweep: _Sweep, label: str, offered: int) -> list[str]:
    notes: list[str] = []
    if sweep.counterexample is not None:
        notes.append(f"{label} fails at {_sample_text(sweep.counterexample)}: {sweep.failure}")
    if sweep.gated:
        notes.append(f"{label}: {sweep.gated} sample(s) skipped: outside the reference's domain")
    if sweep.one_sided:
        notes.append(f"{label}: {sweep.one_sided} sample(s) skipped: only the reference evaluates")
    if sweep.evaluated == 0:
        notes.append(f"{label}: no sample evaluated ({offered} offered)")
    return notes


# -- round trip ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class RoundTripReport:
    """The outcome of :func:`check_round_trip`."""

    scalable: ScalableDiagram
    back: Diagram
    identical: bool
    family_equal: bool
    stripped_ok: bool
    oracle_ok: bool
    samples_evaluated: int
    counterexample: Mapping[str, CheckAssignmentValue] | None
    reason: str

    @property
    def ok(self) -> bool:
        """True when some sample was evaluated and every check held at each evaluated one."""
        return (
            self.samples_evaluated > 0 and self.family_equal and self.stripped_ok and self.oracle_ok
        )


def check_round_trip(
    diagram: Diagram, samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None
) -> RoundTripReport:
    """Check ``from_scalable(to_scalable(diagram))`` against ``diagram``.

    ``identical`` is id-for-id structural equality. At every sample (default
    :func:`default_samples`) where ``diagram`` instantiates, ``family_equal`` requires the
    two instances isomorphic and ``stripped_ok`` requires :func:`strip` isomorphic to the
    instance; where it contracts, ``oracle_ok`` requires exact oracle equality. Raises
    ScalableGrammarError when ``diagram`` has no scalable form.
    """
    if not isinstance(diagram, Diagram):
        raise InteropGrammarError(f"check_round_trip requires a Diagram, got {diagram!r}")
    chosen = _check_samples(samples)
    scalable = to_scalable(diagram)
    back = from_scalable(scalable)
    structure = compare_structure(back, diagram)
    if chosen is None:
        chosen = default_samples(diagram)
    family, stripped = _sweep(
        diagram,
        chosen,
        _instantiable,
        (
            functools.partial(_family_problem, diagram, back),
            functools.partial(_strip_problem, diagram, scalable),
        ),
    )
    (oracle,) = _sweep(
        diagram, chosen, _evaluable, (functools.partial(_oracle_mismatch, diagram, back),)
    )
    parts: list[str] = []
    for label, sweep in (("family check", family), ("strip", stripped)):
        if sweep.counterexample is not None:
            parts.append(f"{label} fails at {_sample_text(sweep.counterexample)}: {sweep.failure}")
    parts.extend(_notes(oracle, "oracle", len(chosen)))
    if not structure.identical:
        parts.append(f"not identical: {structure.reason}")
    counterexample = next(
        (s.counterexample for s in (family, stripped, oracle) if s.counterexample is not None),
        None,
    )
    return RoundTripReport(
        scalable=scalable,
        back=back,
        identical=structure.identical,
        family_equal=family.counterexample is None,
        stripped_ok=stripped.counterexample is None,
        oracle_ok=oracle.counterexample is None,
        samples_evaluated=oracle.evaluated,
        counterexample=counterexample,
        reason="; ".join(parts) or f"round trip agrees at {oracle.evaluated} sample(s)",
    )


# -- sheet steps --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class SheetCheck:
    """The outcome of :func:`check_sheet_step`."""

    step: SheetStep
    replayed: bool
    valid: bool
    oracle_ok: bool
    samples_evaluated: int
    counterexample: Mapping[str, CheckAssignmentValue] | None
    reason: str

    @property
    def ok(self) -> bool:
        """True when the step replays, both sides are valid, and the oracle agrees at one or
        more samples."""
        return self.replayed and self.valid and self.oracle_ok and self.samples_evaluated > 0


def _bang_box_form(s: ScalableDiagram, side: str, issues: list[str]) -> Diagram | None:
    """The bang-box form of ``s``, or None with its problems appended to ``issues``."""
    try:
        found = list(validate_scalable(s))
        back = None if found else from_scalable(s)
    except (ScalableError, BangBoxError, GraphError, ValidateError) as exc:
        found, back = [f"{type(exc).__name__}: {exc}"], None
    if back is not None:
        found.extend(issue.message for issue in validate(back).errors)
    if found:
        issues.append(f"{side} is invalid: " + "; ".join(found))
        return None
    return back


def check_sheet_step(
    step: SheetStep, samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None
) -> SheetCheck:
    """Replay ``step``, validate both sides, and oracle-compare their bang-box forms at
    every sample (default :func:`default_samples`)."""
    if not isinstance(step, SheetStep):
        raise InteropGrammarError(f"check_sheet_step requires a SheetStep, got {step!r}")
    chosen = _check_samples(samples)
    parts: list[str] = []
    try:
        replay_sheet_step(step)
        replayed = True
    except (RewriteError, ScalableError, BangBoxError, TypeError) as exc:
        replayed = False
        parts.append(f"replay failed: {exc}")
    issues: list[str] = []
    before = _bang_box_form(step.before, "before", issues)
    after = _bang_box_form(step.after, "after", issues)
    parts.extend(issues)
    oracle = _Sweep(0, 0, 0, None, "")
    if before is not None and after is not None:
        if chosen is None:
            chosen = default_samples(before, after)
        (oracle,) = _sweep(
            before, chosen, _evaluable, (functools.partial(_oracle_mismatch, before, after),)
        )
        parts.extend(_notes(oracle, "oracle", len(chosen)))
    return SheetCheck(
        step=step,
        replayed=replayed,
        valid=not issues,
        oracle_ok=oracle.counterexample is None,
        samples_evaluated=oracle.evaluated,
        counterexample=oracle.counterexample,
        reason="; ".join(parts) or f"step checks at {oracle.evaluated} sample(s)",
    )


# -- mixed workflows ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class InteropOutcome:
    """The outcome of :func:`via_scalable`: both diagrams, the steps, and their checks."""

    start: Diagram
    result: Diagram
    steps: tuple[SheetStep, ...]
    checks: tuple[SheetCheck, ...]
    round_trip: RoundTripReport

    @property
    def ok(self) -> bool:
        """True when the round trip of ``start`` and every step check pass."""
        return self.round_trip.ok and all(check.ok for check in self.checks)


def via_scalable(
    diagram: Diagram,
    operations: Sequence[tuple[str, Mapping[str, object]]],
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
) -> InteropOutcome:
    """Run named sheet operations on the scalable form of ``diagram``, translate back, and
    check the round trip and every step.

    Node and scale ids in the arguments are ids of the current scalable state; the first
    operation sees the node and box ids of ``diagram``. A refused operation raises its
    SheetError.
    """
    if not isinstance(diagram, Diagram):
        raise InteropGrammarError(f"via_scalable requires a Diagram, got {diagram!r}")
    if isinstance(operations, (str, bytes)) or not isinstance(operations, Sequence):
        raise InteropGrammarError("operations must be a Sequence of (name, arguments) pairs")
    chosen = _check_samples(samples)
    round_trip = check_round_trip(diagram, chosen)
    state = round_trip.scalable
    steps: list[SheetStep] = []
    for entry in operations:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise InteropGrammarError(f"operation must be a (name, arguments) pair: {entry!r}")
        name, arguments = entry
        if not isinstance(name, str) or name not in SHEET_OPERATIONS:
            raise InteropGrammarError(
                f"unknown sheet operation {name!r}; known: {sorted(SHEET_OPERATIONS)}"
            )
        if not isinstance(arguments, Mapping) or not all(isinstance(k, str) for k in arguments):
            raise InteropGrammarError(f"arguments of {name!r} must be a str-keyed Mapping")
        step = SHEET_OPERATIONS[name](state, **arguments)
        steps.append(step)
        state = step.after
    checks = tuple(check_sheet_step(step, chosen) for step in steps)
    return InteropOutcome(
        start=diagram,
        result=from_scalable(state),
        steps=tuple(steps),
        checks=checks,
        round_trip=round_trip,
    )
