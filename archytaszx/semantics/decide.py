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

"""Sound, incomplete decision of diagram equality as families over every free symbol.

:func:`decide_equal` climbs a ladder and stops at the first rung that settles:

1. ``INTERFACE``: boundary arity differs, or a boundary dimension pair has no solution.
2. ``NORMAL_FORM``: both :func:`~archytaszx.rewrite.normal_form.normal_form` results agree.
3. ``ORACLE_COUNTEREXAMPLE``: the numeric oracle, on the original diagrams, mismatches at a
   small deterministic assignment.
4. ``SATURATION``: an :class:`~archytaszx.rewrite.egraph.EGraph` holding both original
   diagrams, saturated within ``saturation_limits``, merges their e-classes.
5. ``SYMBOLIC_CONTRACTION``: both symbolic contractions agree with every symbol formal.
6. ``INDUCTION``: a recursively decided base case plus a proved step case on one
   multiplicity index.

``EQUAL`` comes only from rungs 2, 4, 5 and 6; ``UNEQUAL`` only from rung 1 or an oracle
counterexample; every other outcome is ``UNKNOWN``. A normal-form match needs both
certificates to replay; one whose derivations assumed ``DEFERRED`` dimension constraints runs
rung 3 before it is reported, and the constraints are carried in
:attr:`Decision.assumptions`. A saturation merge yields one certificate per edge of
:meth:`~archytaszx.rewrite.egraph.EGraph.explain`'s path, each from its edge's parent to its
child; each must replay onto a diagram whose comparison view is isomorphic to its child's, and
their ``DEFERRED`` constraints become the assumptions; sides sharing one e-node need none.
An oracle mismatch counts only above ``tolerance`` times the larger entry magnitude (at least
1), at an assignment satisfying every symbol's sympy assumptions. Induction proves ``EQUAL``
only from base 0. The parameter environment never enters a verdict;
:attr:`Decision.parameters_agree` reports it. :func:`interface_reason` and
:func:`refute_by_oracle` run rungs 1 and 3 alone.
"""

from __future__ import annotations

import enum
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionError, UnifyStatus
from archytaszx.algebra.phase import PhaseError
from archytaszx.algebra.scalar import ScalarError
from archytaszx.diagram.bangbox import BangBoxError, free_mult_symbols, instantiate_symbol
from archytaszx.diagram.compare import isomorphic
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, GraphError, PortRef
from archytaszx.diagram.validate import ValidateError, validate_or_raise
from archytaszx.rewrite.egraph import EGraph, SaturationLimits, SaturationReport
from archytaszx.rewrite.engine import DEFAULT_GUARD, TerminationGuard
from archytaszx.rewrite.normal_form import (
    NormalForm,
    comparison_view,
    normal_form,
    same_normal_form,
)
from archytaszx.rewrite.rule import ConstraintOutcome, DimensionConstraint, RewriteError, Rule
from archytaszx.semantics.certificate import Certificate, CertificateError, certify, replay
from archytaszx.semantics.check import (
    DEFAULT_TOLERANCE,
    CheckAssignmentValue,
    CheckError,
    ComparisonResult,
    EqualityMode,
    compare,
    compare_symbolic,
    score,
)
from archytaszx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractError
from archytaszx.semantics.contract_symbolic import SymbolicContractionError, contract_symbolic
from archytaszx.semantics.denote import DenoteError
from archytaszx.semantics.induction import (
    InductionError,
    InductionResult,
    StepDischarge,
    Verdict,
    choose_index,
    prove_by_induction,
)


class DecideError(Exception):
    """Base class for all errors raised by this module."""


class DecideGrammarError(DecideError):
    """A request is malformed: a wrong argument type or an out-of-range bound."""


class EqualityVerdict(enum.Enum):
    """What :func:`decide_equal` established."""

    EQUAL = "equal"
    UNEQUAL = "unequal"
    UNKNOWN = "unknown"


class DecisionMethod(enum.Enum):
    """The ladder rung that settled a :class:`Decision`, or ``NONE``."""

    INTERFACE = "interface"
    NORMAL_FORM = "normal_form"
    SATURATION = "saturation"
    SYMBOLIC_CONTRACTION = "symbolic_contraction"
    INDUCTION = "induction"
    ORACLE_COUNTEREXAMPLE = "oracle_counterexample"
    NONE = "none"


@dataclass(frozen=True, slots=True, eq=False)
class Decision:
    """A verdict, the rung behind it, and the evidence each rung produced.

    ``induction_reversed`` is True when :attr:`induction` was run with ``right`` as its left
    side. ``saturation`` is the saturation rung's report whenever that rung's saturation
    finished.
    """

    verdict: EqualityVerdict
    method: DecisionMethod
    reason: str
    left_nf: NormalForm | None = None
    right_nf: NormalForm | None = None
    certificates: tuple[Certificate, ...] = ()
    counterexample: Mapping[str, CheckAssignmentValue] | None = None
    comparison: ComparisonResult | None = None
    induction: InductionResult | None = None
    samples_checked: int = 0
    assumptions: tuple[DimensionConstraint, ...] = ()
    parameters_agree: bool = True
    induction_reversed: bool = False
    saturation: SaturationReport | None = None

    def __post_init__(self) -> None:
        """Validate the enum and text fields' types."""
        if not isinstance(self.verdict, EqualityVerdict):
            raise DecideGrammarError(f"verdict must be an EqualityVerdict, got {self.verdict!r}")
        if not isinstance(self.method, DecisionMethod):
            raise DecideGrammarError(f"method must be a DecisionMethod, got {self.method!r}")
        if not isinstance(self.reason, str):
            raise DecideGrammarError(f"reason must be a str, got {type(self.reason).__name__}")

    @property
    def decided(self) -> bool:
        """True when the verdict is EQUAL or UNEQUAL."""
        return self.verdict is not EqualityVerdict.UNKNOWN


DIMENSION_SAMPLE_VALUES: tuple[int, ...] = (2, 3, 4, 1)
MULTIPLICITY_SAMPLE_VALUES: tuple[int, ...] = (0, 1, 2, 3)
PHASE_SAMPLE_VALUES: tuple[CheckAssignmentValue, ...] = (
    sp.Rational(1, 3),
    sp.Rational(1, 4),
    0,
    sp.Rational(1, 2),
)
INTEGER_SAMPLE_POOL: tuple[int, ...] = (1, 2, 0, 3, -1, 4, 5, -2, 6, 7, -3, 8)
RATIONAL_SAMPLE_POOL: tuple[CheckAssignmentValue, ...] = (
    *PHASE_SAMPLE_VALUES,
    1,
    2,
    sp.Rational(-1, 3),
    sp.Rational(-1, 4),
    -1,
    sp.Rational(-1, 2),
    -2,
)
DECIDE_SATURATION_LIMITS = SaturationLimits(
    max_iterations=3, max_enodes=128, max_applications=1024, node_margin=4
)
_MAX_GRID_PRODUCT = 4096
_STEP_LADDER: tuple[StepDischarge, ...] = (
    StepDischarge.UNIFORM_REWRITE,
    StepDischarge.INDUCTION_REWRITE,
    StepDischarge.SYMBOLIC_CONTRACTION,
)
_DOMAIN_ERRORS: tuple[type[Exception], ...] = (
    BangBoxError,
    DimensionError,
    GeneratorError,
    GraphError,
    PhaseError,
    ScalarError,
    ValidateError,
)
_ORACLE_ERRORS: tuple[type[Exception], ...] = (
    *_DOMAIN_ERRORS,
    CheckError,
    ContractError,
    DenoteError,
)
_NORMAL_FORM_ERRORS: tuple[type[Exception], ...] = (
    *_DOMAIN_ERRORS,
    CertificateError,
    RewriteError,
)
_SYMBOLIC_ERRORS: tuple[type[Exception], ...] = (
    *_DOMAIN_ERRORS,
    CheckError,
    DenoteError,
    SymbolicContractionError,
)
_INDUCTION_ERRORS: tuple[type[Exception], ...] = (
    *_ORACLE_ERRORS,
    InductionError,
    RewriteError,
    SymbolicContractionError,
)


def _free_symbols(diagram: Diagram) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """``diagram``'s free symbols split into (dimension, multiplicity, phase-or-scalar)."""
    mults = set(free_mult_symbols(diagram))
    dims: set[str] = set()
    others: set[str] = set(diagram.scalar.free_symbols)
    for node in diagram.nodes.values():
        for port in (*node.outputs, *node.inputs):
            dims |= port.dim.free_symbols
        if node.phase is not None:
            dims |= node.phase.dim.free_symbols
            others |= node.phase.free_symbols
    dims -= mults
    others -= mults | dims
    return frozenset(dims), frozenset(mults), frozenset(others)


def _symbol_table(*diagrams: Diagram) -> dict[str, frozenset[sp.Symbol]]:
    """Every free sympy symbol of ``diagrams``, grouped by name."""
    exprs: list[sp.Expr] = []
    for diagram in diagrams:
        exprs.append(diagram.scalar.to_sympy())
        exprs.extend(box.multiplicity.to_sympy() for box in diagram.bang_boxes.values())
        for node in diagram.nodes.values():
            exprs.extend(port.dim.to_sympy() for port in (*node.outputs, *node.inputs))
            if node.phase is not None:
                exprs.append(node.phase.dim.to_sympy())
                exprs.extend(phase.to_sympy_turns() for phase in node.phase.entries().values())
    table: dict[str, set[sp.Symbol]] = {}
    for expr in exprs:
        for symbol in expr.free_symbols:
            table.setdefault(str(symbol.name), set()).add(symbol)
    return {name: frozenset(symbols) for name, symbols in table.items()}


def _satisfies(value: CheckAssignmentValue, symbols: frozenset[sp.Symbol]) -> bool:
    """Whether ``value`` meets every stated sympy assumption of every symbol in ``symbols``."""
    expr = sp.sympify(value)
    for symbol in symbols:
        for key, expected in symbol.assumptions0.items():
            actual = getattr(expr, f"is_{key}", None)
            if actual is not None and bool(actual) is not bool(expected):
                return False
    return True


def _admissible(
    assignment: Mapping[str, CheckAssignmentValue], table: Mapping[str, frozenset[sp.Symbol]]
) -> bool:
    """Whether every value of ``assignment`` meets its symbol's assumptions in ``table``."""
    try:
        return all(
            _satisfies(value, table.get(name, frozenset())) for name, value in assignment.items()
        )
    except (sp.SympifyError, TypeError):
        return False


def _assumption_samples(symbols: frozenset[sp.Symbol]) -> tuple[CheckAssignmentValue, ...]:
    """Up to four pool values meeting ``symbols``' assumptions, integers for an integer symbol,
    cycled to four."""
    integer = any(symbol.is_integer for symbol in symbols)
    pool = INTEGER_SAMPLE_POOL if integer else RATIONAL_SAMPLE_POOL
    chosen = [value for value in pool if _satisfies(value, symbols)][: len(PHASE_SAMPLE_VALUES)]
    return tuple(itertools.islice(itertools.cycle(chosen), len(PHASE_SAMPLE_VALUES)))


def _sample_values(left: Diagram, right: Diagram) -> dict[str, tuple[CheckAssignmentValue, ...]]:
    """Every samplable free symbol of either diagram, mapped to the sample values of its role
    and assumptions."""
    dims_l, mults_l, others_l = _free_symbols(left)
    dims_r, mults_r, others_r = _free_symbols(right)
    mults = mults_l | mults_r
    dims = (dims_l | dims_r) - mults
    others = (others_l | others_r) - mults - dims
    table = _symbol_table(left, right)
    values: dict[str, tuple[CheckAssignmentValue, ...]] = {}
    for name in sorted(mults | dims | others):
        if name in mults:
            values[name] = MULTIPLICITY_SAMPLE_VALUES
        elif name in dims:
            values[name] = DIMENSION_SAMPLE_VALUES
        else:
            chosen = _assumption_samples(table.get(name, frozenset()))
            if chosen:
                values[name] = chosen
    return values


def sample_grid(left: Diagram, right: Diagram) -> tuple[Mapping[str, CheckAssignmentValue], ...]:
    """The default assignments over both diagrams' free symbols: diagonals first, then the
    remaining index tuples by (largest index, index sum, tuple)."""
    values = _sample_values(left, right)
    names = tuple(values)
    width = len(DIMENSION_SAMPLE_VALUES)
    if not names:
        return (MappingProxyType({}),)
    diagonals = [tuple([i] * len(names)) for i in range(width)]
    if width ** len(names) <= _MAX_GRID_PRODUCT:
        rest = sorted(
            (t for t in itertools.product(range(width), repeat=len(names)) if t not in diagonals),
            key=lambda t: (max(t), sum(t), t),
        )
    else:
        rest = [
            tuple(i if j == position else base for j in range(len(names)))
            for base in range(width)
            for position in range(len(names))
            for i in range(width)
            if i != base
        ]
    ordered: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for index_tuple in (*diagonals, *rest):
        if index_tuple not in seen:
            seen.add(index_tuple)
            ordered.append(index_tuple)
    return tuple(
        MappingProxyType({name: values[name][i] for name, i in zip(names, t, strict=True)})
        for t in ordered
    )


def _family(diagram: Diagram) -> Diagram:
    """A copy of ``diagram`` with an empty parameter environment."""
    view = diagram.copy()
    view.set_parameters({})
    return view


def _in_box_scope(diagram: Diagram) -> bool:
    """Whether any boundary port lies on a bang-boxed node or in a box's port scope."""
    for box in diagram.bang_boxes.values():
        for ref in (*diagram.boundary_inputs, *diagram.boundary_outputs):
            if ref.node_id in box.node_scope or ref in box.port_scope:
                return True
    return False


def _boundary_dims(diagram: Diagram, refs: Sequence[PortRef]) -> tuple[Dim, ...] | None:
    """The dimension of each boundary port in ``refs``, or None when one does not resolve."""
    dims: list[Dim] = []
    for ref in refs:
        node = diagram.nodes.get(ref.node_id)
        if node is None:
            return None
        legs = node.legs(ref.direction)
        if not 0 <= ref.index < len(legs):
            return None
        dims.append(legs[ref.index].dim)
    return tuple(dims)


def _interface_reason(left: Diagram, right: Diagram) -> str | None:
    """Why the two interfaces can never agree, or None when they might."""
    if _in_box_scope(left) or _in_box_scope(right):
        return None
    for label, refs_l, refs_r in (
        ("input", left.boundary_inputs, right.boundary_inputs),
        ("output", left.boundary_outputs, right.boundary_outputs),
    ):
        if len(refs_l) != len(refs_r):
            return f"boundary {label} count {len(refs_l)} != {len(refs_r)}"
        dims_l = _boundary_dims(left, refs_l)
        dims_r = _boundary_dims(right, refs_r)
        if dims_l is None or dims_r is None:
            return None
        for position, (dim_l, dim_r) in enumerate(zip(dims_l, dims_r, strict=True)):
            if dim_l.unify(dim_r).status is UnifyStatus.FAILURE:
                return f"boundary {label} {position} has dimension {dim_l} vs {dim_r}, never equal"
    return None


def _symbolic_interface_reason(left: Diagram, right: Diagram) -> str | None:
    """Why the symbolic rung cannot trust a match: a boxed boundary port, or a boundary
    input/output count or per-position dimension that differs; None when none applies."""
    if _in_box_scope(left) or _in_box_scope(right):
        return "a boundary port lies in a bang box's scope"
    for label, refs_l, refs_r in (
        ("input", left.boundary_inputs, right.boundary_inputs),
        ("output", left.boundary_outputs, right.boundary_outputs),
    ):
        dims_l = _boundary_dims(left, refs_l)
        dims_r = _boundary_dims(right, refs_r)
        if dims_l is None or dims_r is None:
            return f"a boundary {label} does not resolve"
        if dims_l != dims_r:
            return f"boundary {label} dimensions {dims_l} vs {dims_r}"
    return None


def _check_args(
    left: object,
    right: object,
    rules: object,
    guard: object,
    samples: object,
    max_samples: object,
    use_symbolic: object,
    use_induction: object,
    max_depth: object,
    tolerance: object,
    max_elements: object,
    use_saturation: object,
    saturation_limits: object,
) -> None:
    """Raise DecideGrammarError on any malformed :func:`decide_equal` argument."""
    for name, value in (("left", left), ("right", right)):
        if not isinstance(value, Diagram):
            raise DecideGrammarError(f"{name} must be a Diagram, got {type(value).__name__}")
    if rules is not None and (
        isinstance(rules, (str, bytes))
        or not isinstance(rules, Sequence)
        or not all(isinstance(rule, Rule) for rule in rules)
    ):
        raise DecideGrammarError("rules must be None or a Sequence of Rule")
    if not isinstance(guard, TerminationGuard):
        raise DecideGrammarError(f"guard must be a TerminationGuard, got {type(guard).__name__}")
    if samples is not None and (
        isinstance(samples, (str, bytes))
        or not isinstance(samples, Sequence)
        or not all(isinstance(sample, Mapping) for sample in samples)
    ):
        raise DecideGrammarError("samples must be None or a Sequence of Mapping")
    for name, value, low in (
        ("max_samples", max_samples, 0),
        ("max_depth", max_depth, 0),
        ("max_elements", max_elements, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise DecideGrammarError(f"{name} must be an int >= {low}, got {value!r}")
    for name, value in (
        ("use_symbolic", use_symbolic),
        ("use_induction", use_induction),
        ("use_saturation", use_saturation),
    ):
        if not isinstance(value, bool):
            raise DecideGrammarError(f"{name} must be a bool, got {value!r}")
    if not isinstance(saturation_limits, SaturationLimits):
        raise DecideGrammarError(
            f"saturation_limits must be a SaturationLimits, got {type(saturation_limits).__name__}"
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not tolerance >= 0:
        raise DecideGrammarError(f"tolerance must be a non-negative float, got {tolerance!r}")


@dataclass(frozen=True, slots=True)
class _OracleRun:
    """The oracle rung's outcome: evaluated samples, and the first mismatch if any."""

    evaluated: tuple[Mapping[str, CheckAssignmentValue], ...]
    mismatch: Mapping[str, CheckAssignmentValue] | None
    comparison: ComparisonResult | None


def _magnitude(
    diagram: Diagram, assignment: Mapping[str, CheckAssignmentValue], max_elements: int
) -> float:
    """The largest entry modulus of ``diagram`` contracted at ``assignment``; 0 when empty."""
    tensor = score(diagram, assignment, max_elements=max_elements).tensor
    return float(np.max(np.abs(tensor))) if tensor.size else 0.0


def _oracle_compare(
    left: Diagram,
    right: Diagram,
    assignment: Mapping[str, CheckAssignmentValue],
    table: Mapping[str, frozenset[sp.Symbol]],
    tolerance: float,
    max_elements: int,
) -> tuple[bool, ComparisonResult | None]:
    """(evaluated, mismatch): the oracle's EXACT comparison at ``assignment``, a mismatch kept
    only when its deviation exceeds ``tolerance * max(1, largest entry modulus)``."""
    if not _admissible(assignment, table):
        return False, None
    try:
        result = compare(
            left,
            right,
            assignment,
            mode=EqualityMode.EXACT,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    except _ORACLE_ERRORS:
        return False, None
    if result.matched:
        return True, None
    if math.isfinite(result.max_abs_deviation):
        try:
            scale = max(
                1.0,
                _magnitude(left, assignment, max_elements),
                _magnitude(right, assignment, max_elements),
            )
        except _ORACLE_ERRORS:
            return True, None
        if result.max_abs_deviation <= tolerance * scale:
            return True, None
    return True, result


def _run_oracle(
    left: Diagram,
    right: Diagram,
    candidates: Sequence[Mapping[str, CheckAssignmentValue]],
    max_samples: int,
    tolerance: float,
    max_elements: int,
) -> _OracleRun:
    """Compare ``left`` and ``right`` at each candidate until ``max_samples`` evaluate or one
    mismatches; a candidate the oracle refuses or that breaks an assumption is skipped."""
    table = _symbol_table(left, right)
    evaluated: list[Mapping[str, CheckAssignmentValue]] = []
    for candidate in candidates:
        if len(evaluated) >= max_samples:
            break
        assignment = MappingProxyType(dict(candidate))
        done, mismatch = _oracle_compare(left, right, assignment, table, tolerance, max_elements)
        if not done:
            continue
        evaluated.append(assignment)
        if mismatch is not None:
            return _OracleRun(tuple(evaluated), assignment, mismatch)
    return _OracleRun(tuple(evaluated), None, None)


def interface_reason(left: Diagram, right: Diagram) -> str | None:
    """Why the interfaces of ``left`` and ``right``, as families, can never agree, or None."""
    for name, value in (("left", left), ("right", right)):
        if not isinstance(value, Diagram):
            raise DecideGrammarError(f"{name} must be a Diagram, got {type(value).__name__}")
    return _interface_reason(_family(left), _family(right))


@dataclass(frozen=True, slots=True, eq=False)
class OracleRefutation:
    """The samples the oracle evaluated, and the first counterexample with its comparison."""

    evaluated: tuple[Mapping[str, CheckAssignmentValue], ...]
    counterexample: Mapping[str, CheckAssignmentValue] | None
    comparison: ComparisonResult | None

    @property
    def refuted(self) -> bool:
        """True when a counterexample was found."""
        return self.counterexample is not None


def refute_by_oracle(
    left: Diagram,
    right: Diagram,
    *,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> OracleRefutation:
    """Rung 3 alone: compare both families at ``samples`` (default :func:`sample_grid`) until
    ``max_samples`` evaluate or one mismatches."""
    _check_args(
        left,
        right,
        None,
        DEFAULT_GUARD,
        samples,
        max_samples,
        True,
        True,
        0,
        tolerance,
        max_elements,
        True,
        DECIDE_SATURATION_LIMITS,
    )
    family_l = _family(left)
    family_r = _family(right)
    candidates = tuple(samples) if samples is not None else sample_grid(family_l, family_r)
    run = _run_oracle(family_l, family_r, candidates, max_samples, tolerance, max_elements)
    return OracleRefutation(run.evaluated, run.mismatch, run.comparison)


def _confirm_mismatch(
    left: Diagram,
    right: Diagram,
    assignment: Mapping[str, CheckAssignmentValue],
    tolerance: float,
    max_elements: int,
) -> ComparisonResult | None:
    """The oracle's mismatching comparison at ``assignment`` as :func:`_oracle_compare` judges
    it, or None."""
    table = _symbol_table(left, right)
    return _oracle_compare(left, right, assignment, table, tolerance, max_elements)[1]


def _symbolic_match(left: Diagram, right: Diagram) -> tuple[bool, str]:
    """Whether both symbolic contractions agree exactly over equal boundaries, and a reason."""
    interface = _symbolic_interface_reason(left, right)
    if interface is not None:
        return False, f"symbolic contraction skipped: {interface}"
    try:
        tensor_l = contract_symbolic(left)
        tensor_r = contract_symbolic(right)
        result = compare_symbolic(tensor_l, tensor_r)
    except _SYMBOLIC_ERRORS as exc:
        return False, f"symbolic contraction declined: {type(exc).__name__}: {exc}"
    return result.matched, result.reason


@dataclass(frozen=True, slots=True)
class _SaturationRun:
    """The saturation rung's outcome: its report, the edge certificates and their ``DEFERRED``
    constraints when the sides merged, and a reason."""

    report: SaturationReport | None
    certificates: tuple[Certificate, ...] | None
    assumptions: tuple[DimensionConstraint, ...]
    reason: str


def _run_saturation(
    left: Diagram, right: Diagram, rules: Sequence[Rule] | None, limits: SaturationLimits
) -> _SaturationRun:
    """Saturate an e-graph rooted at ``left`` and ``right``, then certify and replay every edge
    of the path between them."""
    graph = EGraph()
    report: SaturationReport | None = None
    try:
        first = graph.add(left)
        second = graph.add(right)
        report = graph.saturate(rules, limits=limits)
        edges = graph.explain(first, second)
        if edges is None:
            return _SaturationRun(
                report,
                None,
                (),
                f"no merge after {report.iterations} round(s) and {len(graph)} e-node(s), "
                f"stopped by {report.stop_reason.value}",
            )
        certificates: list[Certificate] = []
        for position, edge in enumerate(edges):
            certificate = certify(
                graph.diagram(edge.parent),
                [edge.result],
                label=f"saturation edge {position}: {edge.rule_name}",
            )
            replayed = replay(certificate, rediscover=False)
            if not replayed.reproduced:
                return _SaturationRun(
                    report, None, (), f"edge {position} did not replay: {replayed.reason}"
                )
            child = comparison_view(graph.diagram(edge.child))
            if not isomorphic(comparison_view(certificate.final), child):
                return _SaturationRun(
                    report, None, (), f"edge {position} does not reach its child e-node"
                )
            certificates.append(certificate)
    except _NORMAL_FORM_ERRORS as exc:
        return _SaturationRun(report, None, (), f"failed: {type(exc).__name__}: {exc}")
    deferred = _dedupe(
        [
            constraint
            for certificate in certificates
            for step in certificate.steps
            for constraint in step.dimension_constraints
            if constraint.outcome is ConstraintOutcome.DEFERRED
        ]
    )
    reason = f"{len(edges)} edge(s)" if edges else "0 edge(s), both sides one e-node"
    return _SaturationRun(report, tuple(certificates), deferred, reason)


def _dedupe(constraints: Sequence[DimensionConstraint]) -> tuple[DimensionConstraint, ...]:
    """``constraints`` without repeats, first occurrence order kept."""
    kept: list[DimensionConstraint] = []
    for constraint in constraints:
        if constraint not in kept:
            kept.append(constraint)
    return tuple(kept)


def _induction_assumptions(
    base: Decision, result: InductionResult
) -> tuple[DimensionConstraint, ...]:
    """``base``'s assumptions, then every ``DEFERRED`` constraint of ``result``'s tier steps."""
    deferred = (
        constraint
        for tier in result.tiers
        for step in tier.steps
        for constraint in step.dimension_constraints
        if constraint.outcome is ConstraintOutcome.DEFERRED
    )
    return _dedupe((*base.assumptions, *deferred))


def _instantiate_index(diagram: Diagram, index: str, value: int) -> Diagram:
    """``diagram`` with multiplicity ``index`` set to ``value``, validated."""
    if index in free_mult_symbols(diagram):
        diagram = instantiate_symbol(diagram, index, value)
    else:
        diagram = diagram.copy()
    validate_or_raise(diagram)
    return diagram


def _restrict_samples(
    samples: Sequence[Mapping[str, CheckAssignmentValue]],
    index: str,
    left: Diagram,
    right: Diagram,
) -> tuple[Mapping[str, CheckAssignmentValue], ...]:
    """``samples`` without ``index`` and cut to the free symbols of ``left`` and ``right``,
    deduplicated in order."""
    names = set(_sample_values(left, right)) - {index}
    restricted: list[Mapping[str, CheckAssignmentValue]] = []
    for sample in samples:
        cut = MappingProxyType({k: v for k, v in sample.items() if k in names})
        if cut not in restricted:
            restricted.append(cut)
    return tuple(restricted)


def _base_witness(
    left: Diagram,
    right: Diagram,
    index: str,
    base: int,
    evaluated: Sequence[Mapping[str, CheckAssignmentValue]],
    tolerance: float,
    max_elements: int,
) -> dict[str, CheckAssignmentValue]:
    """The first evaluated sample, ``index`` dropped, at which the oracle evaluates both sides
    with ``index`` set to ``base``; empty when none does."""
    for sample in evaluated:
        witness = {name: value for name, value in sample.items() if name != index}
        try:
            compare(
                left,
                right,
                {**witness, index: base},
                tolerance=tolerance,
                max_elements=max_elements,
            )
        except _ORACLE_ERRORS:
            continue
        return witness
    return {}


def decide_equal(
    left: Diagram,
    right: Diagram,
    *,
    rules: Sequence[Rule] | None = None,
    guard: TerminationGuard = DEFAULT_GUARD,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    max_samples: int = 24,
    use_symbolic: bool = True,
    use_induction: bool = True,
    max_depth: int = 2,
    tolerance: float = DEFAULT_TOLERANCE,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    use_saturation: bool = True,
    saturation_limits: SaturationLimits = DECIDE_SATURATION_LIMITS,
) -> Decision:
    """Decide whether ``left`` and ``right`` are equal for every value of their free symbols.

    Runs the module docstring's ladder; ``samples`` replaces :func:`sample_grid` for the
    oracle rung and, cut to its symbols, for the induction base case; ``max_depth`` bounds
    nested induction; ``rules`` also replaces the saturation rung's default rules. The
    induction step is tried left-to-right, then right-to-left.
    """
    _check_args(
        left,
        right,
        rules,
        guard,
        samples,
        max_samples,
        use_symbolic,
        use_induction,
        max_depth,
        tolerance,
        max_elements,
        use_saturation,
        saturation_limits,
    )
    parameters_agree = dict(left.parameters) == dict(right.parameters)
    family_l = _family(left)
    family_r = _family(right)

    interface = _interface_reason(family_l, family_r)
    if interface is not None:
        return Decision(
            EqualityVerdict.UNEQUAL,
            DecisionMethod.INTERFACE,
            interface,
            parameters_agree=parameters_agree,
        )

    nf_l: NormalForm | None = None
    nf_r: NormalForm | None = None
    certificates: tuple[Certificate, ...] = ()
    nf_assumptions: tuple[DimensionConstraint, ...] = ()
    nf_reason = "normal forms differ"
    nf_equal = False
    try:
        nf_l = normal_form(left, rules=rules, guard=guard)
        nf_r = normal_form(right, rules=rules, guard=guard)
        certificates = (
            certify(nf_l.source, nf_l.results, label="left normal form"),
            certify(nf_r.source, nf_r.results, label="right normal form"),
        )
        nf_assumptions = _dedupe((*nf_l.assumed_constraints, *nf_r.assumed_constraints))
        nf_equal = same_normal_form(nf_l, nf_r)
        if nf_equal:
            for certificate in certificates:
                replayed = replay(certificate, rediscover=False)
                if not replayed.reproduced:
                    nf_equal = False
                    nf_reason = (
                        f"normal forms agree but the {certificate.derivation.label} certificate "
                        f"did not replay: {replayed.reason}"
                    )
                    break
    except _NORMAL_FORM_ERRORS as exc:
        nf_l = nf_r = None
        certificates = ()
        nf_assumptions = ()
        nf_equal = False
        nf_reason = f"normal form failed: {type(exc).__name__}: {exc}"

    saturation: SaturationReport | None = None

    def finish(
        verdict: EqualityVerdict,
        method: DecisionMethod,
        reason: str,
        *,
        assumptions: tuple[DimensionConstraint, ...] = nf_assumptions,
        proofs: tuple[Certificate, ...] = certificates,
        **evidence: object,
    ) -> Decision:
        return Decision(
            verdict,
            method,
            reason,
            left_nf=nf_l,
            right_nf=nf_r,
            certificates=proofs,
            assumptions=assumptions,
            parameters_agree=parameters_agree,
            saturation=saturation,
            **evidence,  # type: ignore[arg-type]
        )

    if nf_equal and not nf_assumptions:
        return finish(EqualityVerdict.EQUAL, DecisionMethod.NORMAL_FORM, "normal forms agree")

    candidates = tuple(samples) if samples is not None else sample_grid(family_l, family_r)
    oracle = _run_oracle(family_l, family_r, candidates, max_samples, tolerance, max_elements)
    checked = len(oracle.evaluated)
    if oracle.mismatch is not None:
        return finish(
            EqualityVerdict.UNEQUAL,
            DecisionMethod.ORACLE_COUNTEREXAMPLE,
            f"oracle mismatch at {dict(oracle.mismatch)!r}: "
            f"{oracle.comparison.reason if oracle.comparison else ''}",
            assumptions=(),
            counterexample=oracle.mismatch,
            comparison=oracle.comparison,
            samples_checked=checked,
        )
    sampled = (
        "no oracle sample evaluated"
        if checked == 0
        else f"no oracle mismatch in {checked} sample(s)"
    )
    if nf_equal:
        return finish(
            EqualityVerdict.EQUAL,
            DecisionMethod.NORMAL_FORM,
            f"normal forms agree, conditional on the {len(nf_assumptions)} deferred dimension "
            f"constraint(s) in assumptions; {sampled}",
            samples_checked=checked,
        )

    notes = [nf_reason, f"no oracle mismatch in {checked} sample(s)"]
    if use_saturation:
        run = _run_saturation(left, right, rules, saturation_limits)
        saturation = run.report
        if run.certificates is not None:
            conditional = (
                f", conditional on the {len(run.assumptions)} deferred dimension constraint(s) "
                f"in assumptions; {sampled}"
                if run.assumptions
                else ""
            )
            return finish(
                EqualityVerdict.EQUAL,
                DecisionMethod.SATURATION,
                f"saturation merged both sides along {run.reason}{conditional}",
                assumptions=run.assumptions,
                proofs=run.certificates,
                samples_checked=checked,
            )
        notes.append(f"saturation: {run.reason}")
    else:
        notes.append("saturation disabled")
    if use_symbolic:
        matched, symbolic_reason = _symbolic_match(family_l, family_r)
        if matched:
            return finish(
                EqualityVerdict.EQUAL,
                DecisionMethod.SYMBOLIC_CONTRACTION,
                f"symbolic contractions agree: {symbolic_reason}",
                assumptions=(),
                samples_checked=checked,
            )
        notes.append(f"symbolic: {symbolic_reason}")
    else:
        notes.append("symbolic contraction disabled")

    live = free_mult_symbols(family_l) | free_mult_symbols(family_r)
    if use_induction and max_depth > 0 and live:
        settled = _decide_by_induction(
            family_l,
            family_r,
            evaluated=oracle.evaluated,
            samples=samples,
            rules=rules,
            guard=guard,
            max_samples=max_samples,
            use_symbolic=use_symbolic,
            max_depth=max_depth,
            tolerance=tolerance,
            max_elements=max_elements,
            use_saturation=use_saturation,
            saturation_limits=saturation_limits,
        )
        if isinstance(settled, str):
            notes.append(f"induction: {settled}")
        else:
            verdict, method, reason, evidence = settled
            if verdict is EqualityVerdict.UNKNOWN:
                reason = "; ".join([*notes, f"induction: {reason}"])
            return finish(verdict, method, reason, samples_checked=checked, **evidence)  # type: ignore[arg-type]
    elif not use_induction:
        notes.append("induction disabled")
    elif not live:
        notes.append("no multiplicity symbol to induct on")
    else:
        notes.append("induction depth exhausted")

    return finish(
        EqualityVerdict.UNKNOWN,
        DecisionMethod.NONE,
        "; ".join(notes),
        samples_checked=checked,
    )


def _decide_by_induction(
    left: Diagram,
    right: Diagram,
    *,
    evaluated: Sequence[Mapping[str, CheckAssignmentValue]],
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    rules: Sequence[Rule] | None,
    guard: TerminationGuard,
    max_samples: int,
    use_symbolic: bool,
    max_depth: int,
    tolerance: float,
    max_elements: int,
    use_saturation: bool,
    saturation_limits: SaturationLimits,
) -> tuple[EqualityVerdict, DecisionMethod, str, dict[str, object]] | str:
    """The induction rung: a settled (verdict, method, reason, evidence), or why it did not
    settle."""
    try:
        index = choose_index(left, right)
    except InductionError as exc:
        return str(exc)
    base: int | None = None
    pair: tuple[Diagram, Diagram] | None = None
    for candidate in (0, 1):
        try:
            pair = (
                _instantiate_index(left, index, candidate),
                _instantiate_index(right, index, candidate),
            )
        except _DOMAIN_ERRORS:
            continue
        base = candidate
        break
    if base is None or pair is None:
        return f"neither {index}=0 nor {index}=1 instantiates both sides"
    base_decision = decide_equal(
        pair[0],
        pair[1],
        rules=rules,
        guard=guard,
        samples=None if samples is None else _restrict_samples(samples, index, *pair),
        max_samples=max_samples,
        use_symbolic=use_symbolic,
        use_induction=True,
        max_depth=max_depth - 1,
        tolerance=tolerance,
        max_elements=max_elements,
        use_saturation=use_saturation,
        saturation_limits=saturation_limits,
    )
    if (
        base_decision.verdict is EqualityVerdict.UNEQUAL
        and base_decision.counterexample is not None
    ):
        lifted = MappingProxyType({**dict(base_decision.counterexample), index: base})
        confirmed = _confirm_mismatch(left, right, lifted, tolerance, max_elements)
        if confirmed is not None:
            return (
                EqualityVerdict.UNEQUAL,
                DecisionMethod.ORACLE_COUNTEREXAMPLE,
                f"base case {index}={base} refuted at {dict(lifted)!r}",
                {"counterexample": lifted, "comparison": confirmed, "assumptions": ()},
            )
    if base_decision.verdict is not EqualityVerdict.EQUAL:
        return f"base case {index}={base} not proved: {base_decision.reason}"

    witness = _base_witness(left, right, index, base, evaluated, tolerance, max_elements)
    reasons: list[str] = []
    for reversed_ in (False, True):
        first, second = (right, left) if reversed_ else (left, right)
        try:
            result = prove_by_induction(
                first,
                second,
                index=index,
                base=base,
                witness=witness,
                ladder=_STEP_LADDER,
                rules=rules,
                tolerance=tolerance,
                max_elements=max_elements,
            )
        except _INDUCTION_ERRORS as exc:
            reasons.append(f"step case declined: {type(exc).__name__}: {exc}")
            continue
        if result.verdict is Verdict.REFUTED and result.counterexample is not None:
            lifted = MappingProxyType({**witness, index: result.counterexample})
            confirmed = _confirm_mismatch(left, right, lifted, tolerance, max_elements)
            if confirmed is not None:
                return (
                    EqualityVerdict.UNEQUAL,
                    DecisionMethod.ORACLE_COUNTEREXAMPLE,
                    f"induction refuted the claim at {dict(lifted)!r}",
                    {
                        "counterexample": lifted,
                        "comparison": confirmed,
                        "induction": result,
                        "induction_reversed": reversed_,
                        "assumptions": (),
                    },
                )
        if not result.proved:
            reasons.append(f"step case over {index} not proved: {result.reason}")
            continue
        proof = (
            f"base case {index}={base} proved by {base_decision.method.value}; step case "
            f"proved by {result.discharge.value if result.discharge else 'none'}"
            f"{' with sides reversed' if reversed_ else ''}"
        )
        if base != 0:
            return (
                EqualityVerdict.UNKNOWN,
                DecisionMethod.NONE,
                f"{proof}; proved for {index} >= {base} only, {index}=0 does not instantiate",
                {"induction": result, "induction_reversed": reversed_},
            )
        return (
            EqualityVerdict.EQUAL,
            DecisionMethod.INDUCTION,
            f"{proof}; holds for {index} >= 0",
            {
                "induction": result,
                "induction_reversed": reversed_,
                "assumptions": _induction_assumptions(base_decision, result),
            },
        )
    return "; reversed: ".join(reasons)
