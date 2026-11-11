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

1. ``INTERFACE``: boundary arity differs as a polynomial in the free multiplicities, or a
   boundary dimension pair has no solution.
2. ``NORMAL_FORM``: both :func:`~archytaszx.rewrite.normal_form.normal_form` results agree.
3. ``ORACLE_COUNTEREXAMPLE``: the numeric oracle, on the original diagrams, mismatches at a
   small deterministic assignment.
4. ``SATURATION``: an :class:`~archytaszx.rewrite.egraph.EGraph` holding both original
   diagrams, saturated within ``saturation_limits``, merges their e-classes.
5. ``SYMBOLIC_CONTRACTION``: both symbolic contractions agree with every symbol formal;
   ``SYMBOLIC_WITNESS``: otherwise, one entry of the two contractions, evaluated at a sample
   assignment and an index tuple the difference's deltas pin, differs.
6. ``INDUCTION``: a recursively decided base case plus a proved step case on one
   multiplicity index.

Mixed operands (a discard on either side, or ``mixed=True``) are ``DOUBLED``: the ladder runs
unchanged on both doubled diagrams, the settling rung recorded as ``inner_method``.

``EQUAL`` comes only from rungs 2, 4, 5 and 6; ``UNEQUAL`` only from rung 1, an oracle
counterexample or a symbolic witness; every other outcome is ``UNKNOWN``. :func:`decide_equal`
reports an ``UNEQUAL`` not settled by rung 1 only with its Lean refutation
(:attr:`Decision.refutation`, :func:`lean_refutation`), else ``UNEQUAL_UNCERTIFIED``, and an
``EQUAL`` only with its proof (:attr:`Decision.proof`), else ``EQUAL_UNCERTIFIED``. A
normal-form match needs rung 3 to find no mismatch before it is reported; ``DEFERRED``
dimension constraints its derivations assumed are carried in :attr:`Decision.assumptions`. A
saturation merge yields one certificate per edge of
:meth:`~archytaszx.rewrite.egraph.EGraph.explain`'s path, each from its edge's parent to its
child; each must replay onto a diagram whose comparison view is isomorphic to its child's, and
their ``DEFERRED`` constraints become the assumptions; sides sharing one e-node need none. An
oracle mismatch counts only above ``tolerance`` times the larger entry magnitude (at least 1),
at an assignment satisfying every symbol's sympy assumptions. Induction proves ``EQUAL`` only
from base 0. The parameter environment never enters a verdict;
:attr:`Decision.parameters_agree` reports it. :func:`interface_reason`,
:func:`refute_by_oracle` and :func:`refute_by_symbolic_witness` run rung 1, rung 3 and the
witness search alone.
"""

from __future__ import annotations

import contextvars
import enum
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx import trace as _trace
from archytaszx.algebra.dimension import Dim, DimensionError, UnifyStatus
from archytaszx.algebra.phase import INDEX, PhaseError
from archytaszx.algebra.scalar import ScalarError
from archytaszx.diagram.bangbox import (
    BangBoxError,
    boundary_arity,
    free_mult_symbols,
    instantiate_symbol,
    substitute_index_phases,
)
from archytaszx.diagram.generators import GeneratorError
from archytaszx.diagram.graph import Diagram, GraphError, PortRef
from archytaszx.diagram.mixed import double, is_pure
from archytaszx.diagram.validate import ValidateError, validate_or_raise
from archytaszx.rewrite.egraph import EGraph, SaturationLimits, SaturationReport
from archytaszx.rewrite.engine import DEFAULT_GUARD, TerminationGuard
from archytaszx.rewrite.normal_form import (
    NormalForm,
    comparison_view,
    normal_form,
    same_normal_form,
    views_isomorphic,
)
from archytaszx.rewrite.rule import DimensionConstraint, RewriteError, Rule, assumed
from archytaszx.semantics.certificate import Certificate, CertificateError, certify, replay
from archytaszx.semantics.check import (
    DEFAULT_TOLERANCE,
    CheckAssignmentValue,
    CheckError,
    ComparisonResult,
    EntryWitness,
    EqualityMode,
    compare,
    compare_symbolic,
    find_entry_witness,
    score,
)
from archytaszx.semantics.contract_numeric import DEFAULT_MAX_ELEMENTS, ContractError
from archytaszx.semantics.contract_symbolic import (
    SymbolicContractionError,
    SymbolicTensor,
    contract_symbolic,
)
from archytaszx.semantics.denote import DenoteError
from archytaszx.semantics.induction import (
    InductionError,
    InductionResult,
    StepDischarge,
    Verdict,
    choose_index,
    prove_by_induction,
)

if TYPE_CHECKING:
    from archytaszx.semantics.kernel import Theorem
    from archytaszx.semantics.restrict import Restriction


class DecideError(Exception):
    """Base class for all errors raised by this module."""


class DecideGrammarError(DecideError):
    """A request is malformed: a wrong argument type or an out-of-range bound."""


class EqualityVerdict(enum.Enum):
    """What :func:`decide_equal` established; the uncertified verdicts carry no proof or Lean
    refutation."""

    EQUAL = "equal"
    UNEQUAL = "unequal"
    EQUAL_UNCERTIFIED = "equal_uncertified"
    UNEQUAL_UNCERTIFIED = "unequal_uncertified"
    UNKNOWN = "unknown"


class DecisionMethod(enum.Enum):
    """The ladder rung that settled a :class:`Decision`, or ``NONE``."""

    INTERFACE = "interface"
    NORMAL_FORM = "normal_form"
    SATURATION = "saturation"
    SYMBOLIC_CONTRACTION = "symbolic_contraction"
    INDUCTION = "induction"
    ORACLE_COUNTEREXAMPLE = "oracle_counterexample"
    SYMBOLIC_WITNESS = "symbolic_witness"
    DOUBLED = "doubled"
    NONE = "none"


@dataclass(frozen=True, slots=True, eq=False)
class Decision:
    """A verdict, the rung behind it, and the evidence each rung produced.

    ``induction_reversed`` is True when :attr:`induction` was run with ``right`` as its left
    side. ``saturation`` is the saturation rung's report whenever that rung's saturation
    finished. ``witness`` is the differing entry behind a ``SYMBOLIC_WITNESS`` verdict, whose
    assignment is also :attr:`counterexample`. A restricted decision carries its
    ``restriction``, the unrestricted ``family`` decision, ``family_holds`` when that decision
    was EQUAL, the ``restricted_pair`` it decided otherwise, the counterexample in the original
    symbols as ``counterexample_original``, and the dimension floors a symbolic contraction
    needed as ``floors_used``. ``refutation`` is the Lean refutation source of an UNEQUAL
    verdict not settled by the interface; ``proof`` is the proof of an EQUAL verdict;
    ``induction_proof`` is the kernel theorem of an INDUCTION verdict whose step case was
    settled by rewriting.
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
    witness: EntryWitness | None = None
    inner_method: DecisionMethod | None = None
    doubled: tuple[Diagram, Diagram] | None = None
    restriction: Restriction | None = None
    family: Decision | None = None
    family_holds: bool = False
    counterexample_original: Mapping[str, CheckAssignmentValue] | None = None
    restricted_pair: tuple[Diagram, Diagram] | None = None
    floors_used: Mapping[str, int] | None = None
    refutation: str | None = None
    proof: object | None = None
    induction_proof: Theorem | None = None

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
        return self.verdict in (EqualityVerdict.EQUAL, EqualityVerdict.UNEQUAL)

    @property
    def holds_uncertified(self) -> bool:
        """True when the verdict is EQUAL_UNCERTIFIED or UNEQUAL_UNCERTIFIED."""
        return self.verdict in (
            EqualityVerdict.EQUAL_UNCERTIFIED,
            EqualityVerdict.UNEQUAL_UNCERTIFIED,
        )

    @property
    def claims_equal(self) -> bool:
        """True when the verdict is EQUAL or EQUAL_UNCERTIFIED."""
        return self.verdict in (EqualityVerdict.EQUAL, EqualityVerdict.EQUAL_UNCERTIFIED)

    @property
    def claims_unequal(self) -> bool:
        """True when the verdict is UNEQUAL or UNEQUAL_UNCERTIFIED."""
        return self.verdict in (EqualityVerdict.UNEQUAL, EqualityVerdict.UNEQUAL_UNCERTIFIED)


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
_NESTED: contextvars.ContextVar[bool] = contextvars.ContextVar("decide_nested", default=False)
"""True inside an induction base case's :func:`decide_equal`, which is not certified."""
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
                if node.phase.rule is not None:
                    exprs.append(node.phase.rule.to_sympy_turns().xreplace({INDEX: 1}))
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


def _floor_samples(floor: int) -> tuple[int, ...]:
    """The first four values at least ``floor`` of 2, 3, 4, 1, 5, 6, ..."""
    order = itertools.chain(DIMENSION_SAMPLE_VALUES, itertools.count(5))
    return tuple(itertools.islice((v for v in order if v >= floor), len(DIMENSION_SAMPLE_VALUES)))


def _sample_values(
    left: Diagram, right: Diagram, floors: Mapping[str, int] | None = None
) -> dict[str, tuple[CheckAssignmentValue, ...]]:
    """Every samplable free symbol of either diagram, mapped to the sample values of its role
    and assumptions; a dimension with a floor in ``floors`` samples from that floor and its
    phase-vector floor."""
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
        elif name in dims and floors and name in floors:
            low = max(floors[name], dimension_floors(left, right).get(name, 1))
            values[name] = _floor_samples(low)
        elif name in dims:
            values[name] = DIMENSION_SAMPLE_VALUES
        else:
            chosen = _assumption_samples(table.get(name, frozenset()))
            if chosen:
                values[name] = chosen
    return values


def sample_grid(
    left: Diagram, right: Diagram, floors: Mapping[str, int] | None = None
) -> tuple[Mapping[str, CheckAssignmentValue], ...]:
    """The default assignments over both diagrams' free symbols: diagonals first, then the
    remaining index tuples by (largest index, index sum, tuple); ``floors`` raises the samples
    of the dimensions it names."""
    values = _sample_values(left, right, floors)
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


def _instantiate_at(diagram: Diagram, point: Mapping[str, int]) -> Diagram:
    """``diagram`` with every multiplicity symbol of ``point`` it carries instantiated."""
    for name, value in point.items():
        if name in free_mult_symbols(diagram):
            diagram = instantiate_symbol(diagram, name, value)
    return diagram


def _arity_witness(
    left: Diagram,
    right: Diagram,
    arity_l: tuple[sp.Expr, sp.Expr],
    arity_r: tuple[sp.Expr, sp.Expr],
    counts: tuple[sp.Expr, sp.Expr],
) -> dict[str, int] | None:
    """The first point of the degree grid from 0, over the symbols of ``counts``, where their
    difference is nonzero and both sides instantiate to the substituted arities, or None."""
    difference = sp.expand(counts[0] - counts[1])
    names = sorted(counts[0].free_symbols | counts[1].free_symbols, key=lambda symbol: symbol.name)
    ranges = [range(int(sp.degree(difference, symbol)) + 1) for symbol in names]
    for values in itertools.product(*ranges):
        point = {symbol.name: value for symbol, value in zip(names, values, strict=True)}
        mapping = dict(zip(names, values, strict=True))
        if difference.subs(mapping) == 0:
            continue
        try:
            instances = (_instantiate_at(left, point), _instantiate_at(right, point))
        except _DOMAIN_ERRORS:
            continue
        for instance, arity in zip(instances, (arity_l, arity_r), strict=True):
            expected = tuple(sp.expand(count.subs(mapping)) for count in arity)
            if boundary_arity(instance) != expected:
                return None
        return point
    return None


def _arity_reason(left: Diagram, right: Diagram) -> str | None:
    """Why the boundary counts, as multiplicity polynomials, differ at an admissible point."""
    arity_l = boundary_arity(left)
    arity_r = boundary_arity(right)
    if arity_l is None or arity_r is None:
        return None
    for label, count_l, count_r in (
        ("input", arity_l[0], arity_r[0]),
        ("output", arity_l[1], arity_r[1]),
    ):
        difference = sp.expand(count_l - count_r)
        if difference == 0:
            continue
        if not (count_l.free_symbols or count_r.free_symbols):
            return f"boundary {label} count {count_l} != {count_r}"
        point = _arity_witness(left, right, arity_l, arity_r, (count_l, count_r))
        if point is None:
            return None
        where = ", ".join(f"{name} = {value}" for name, value in point.items())
        return f"boundary {label} count {count_l} != {count_r}, differing at {where}"
    return None


@_trace.traced("decide.interface")
def _interface_reason(left: Diagram, right: Diagram) -> str | None:
    """Why the two interfaces can never agree, or None when they might."""
    if _in_box_scope(left) or _in_box_scope(right):
        return _arity_reason(left, right)
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
    """Why the symbolic rung cannot trust a match: a boundary input/output count or
    per-position dimension that differs; None when none applies."""
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
    """The oracle rung's outcome: evaluated samples, the first mismatch if any, and why each
    refused sample was refused."""

    evaluated: tuple[Mapping[str, CheckAssignmentValue], ...]
    mismatch: Mapping[str, CheckAssignmentValue] | None
    comparison: ComparisonResult | None
    refusals: tuple[str, ...] = ()


def _magnitude(
    diagram: Diagram, assignment: Mapping[str, CheckAssignmentValue], max_elements: int
) -> float:
    """The largest entry modulus of ``diagram`` contracted at ``assignment``; 0 when empty."""
    tensor = score(diagram, assignment, max_elements=max_elements).tensor
    return float(np.max(np.abs(tensor))) if tensor.size else 0.0


@_trace.traced("decide.oracle.sample")
def _oracle_compare(
    left: Diagram,
    right: Diagram,
    assignment: Mapping[str, CheckAssignmentValue],
    table: Mapping[str, frozenset[sp.Symbol]],
    tolerance: float,
    max_elements: int,
) -> tuple[str | None, ComparisonResult | None]:
    """(refusal, mismatch): why the oracle could not evaluate ``assignment`` (None when it did),
    and its EXACT comparison's mismatch, kept only when the deviation exceeds
    ``tolerance * max(1, largest entry modulus)``."""
    if not _admissible(assignment, table):
        return "breaks a symbol's assumptions", None
    try:
        result = compare(
            left,
            right,
            assignment,
            mode=EqualityMode.EXACT,
            tolerance=tolerance,
            max_elements=max_elements,
        )
    except _ORACLE_ERRORS as exc:
        return f"{type(exc).__name__}: {exc}", None
    if result.matched:
        return None, None
    if math.isfinite(result.max_abs_deviation):
        try:
            scale = max(
                1.0,
                _magnitude(left, assignment, max_elements),
                _magnitude(right, assignment, max_elements),
            )
        except _ORACLE_ERRORS:
            return None, None
        if result.max_abs_deviation <= tolerance * scale:
            return None, None
    return None, result


@_trace.traced("decide.oracle")
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
    refusals: list[str] = []
    seen = [dict(c) for c in candidates]
    fallback = _smallest_admissible(left, right, table)
    queue = list(candidates)
    while queue:
        if len(evaluated) >= max_samples:
            break
        assignment = MappingProxyType(dict(queue.pop(0)))
        refusal, mismatch = _oracle_compare(left, right, assignment, table, tolerance, max_elements)
        if refusal is not None:
            refusals.append(f"{dict(assignment)!r}: {refusal}")
            if fallback is not None and dict(fallback) not in seen:
                seen.append(dict(fallback))
                queue.insert(0, fallback)
            continue
        evaluated.append(assignment)
        if mismatch is not None:
            return _OracleRun(tuple(evaluated), assignment, mismatch, tuple(refusals))
    return _OracleRun(tuple(evaluated), None, None, tuple(refusals))


_ADMISSIBLE_SEARCH = 64


def _smallest_admissible(
    left: Diagram, right: Diagram, table: Mapping[str, frozenset[sp.Symbol]]
) -> Mapping[str, CheckAssignmentValue] | None:
    """The point giving each free symbol its least value meeting its assumptions: a
    multiplicity from 0, a dimension from its floor, any other symbol its first sample; None
    when a symbol has none."""
    dims_l, mults_l, others_l = _free_symbols(left)
    dims_r, mults_r, others_r = _free_symbols(right)
    mults = mults_l | mults_r
    dims = (dims_l | dims_r) - mults
    others = (others_l | others_r) - mults - dims
    floors = dimension_floors(left, right)
    point: dict[str, CheckAssignmentValue] = {}
    for name in sorted(mults | dims | others):
        symbols = table.get(name, frozenset())
        chosen: CheckAssignmentValue | None
        if name in others:
            pool = _assumption_samples(symbols)
            chosen = pool[0] if pool else None
        else:
            low = 0 if name in mults else floors.get(name, 1)
            chosen = next(
                (v for v in range(low, low + _ADMISSIBLE_SEARCH) if _satisfies(v, symbols)), None
            )
        if chosen is None:
            return None
        point[name] = chosen
    return MappingProxyType(point) if point else None


def oracle_summary(checked: int, refusals: Sequence[str]) -> str:
    """One line on an oracle run that found no mismatch: how many samples evaluated, and how
    many were refused, with the first refusal."""
    head = (
        "no oracle sample evaluated"
        if checked == 0
        else (f"no oracle mismatch in {checked} sample(s)")
    )
    if not refusals:
        return head
    return f"{head}; {len(refusals)} sample(s) refused, first at {refusals[0]}"


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
    refusals: tuple[str, ...] = ()

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
    return OracleRefutation(run.evaluated, run.mismatch, run.comparison, run.refusals)


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


def dimension_floors(*diagrams: Diagram) -> dict[str, int]:
    """The least value each bare dimension symbol can take for every phase vector over it in
    ``diagrams`` to be defined: one above its largest stored index."""
    floors: dict[str, int] = {}
    for diagram in diagrams:
        for node in diagram.nodes.values():
            if node.phase is None:
                continue
            indices = [*node.phase.entries(), *node.phase.weights()]
            if not indices:
                continue
            dim = node.phase.dim.to_sympy()
            if dim.is_Symbol:
                name = str(dim.name)
                floors[name] = max(floors.get(name, 1), max(indices) + 1)
    return floors


@dataclass(frozen=True, slots=True)
class _SymbolicRun:
    """The symbolic rung's outcome: whether the contractions matched, a reason, and both
    contractions when they were built over equal boundaries."""

    matched: bool
    reason: str
    tensors: tuple[SymbolicTensor, SymbolicTensor] | None = None
    floors_used: Mapping[str, int] | None = None


def _merged_floors(
    left: Diagram, right: Diagram, floors: Mapping[str, int] | None
) -> dict[str, int]:
    """The phase-vector floors of both diagrams combined with ``floors`` by maximum."""
    merged = dimension_floors(left, right)
    for name, low in (floors or {}).items():
        merged[name] = max(merged.get(name, 1), low)
    return merged


@_trace.traced("decide.symbolic_contraction")
def _symbolic_match(
    left: Diagram, right: Diagram, floors: Mapping[str, int] | None = None
) -> _SymbolicRun:
    """Whether both symbolic contractions agree exactly over equal boundaries, and a reason;
    with ``floors``, compared again under them when they disagree without."""
    interface = _symbolic_interface_reason(left, right)
    if interface is not None:
        return _SymbolicRun(False, f"symbolic contraction skipped: {interface}")
    try:
        tensor_l = contract_symbolic(left)
        tensor_r = contract_symbolic(right)
        result = compare_symbolic(
            tensor_l, tensor_r, dimension_floors=dimension_floors(left, right)
        )
        if not result.matched and floors:
            floored = compare_symbolic(
                tensor_l, tensor_r, dimension_floors=_merged_floors(left, right, floors)
            )
            if floored.matched:
                used = MappingProxyType(dict(floors))
                return _SymbolicRun(True, floored.reason, (tensor_l, tensor_r), used)
    except _SYMBOLIC_ERRORS as exc:
        return _SymbolicRun(False, f"symbolic contraction declined: {type(exc).__name__}: {exc}")
    return _SymbolicRun(result.matched, result.reason, (tensor_l, tensor_r))


def _witness_assignments(
    left: Diagram,
    right: Diagram,
    candidates: Sequence[Mapping[str, CheckAssignmentValue]],
    restricted: Mapping[str, int] | None = None,
) -> tuple[Mapping[str, CheckAssignmentValue], ...]:
    """``candidates`` that meet every symbol's assumptions, every dimension floor and every
    floor in ``restricted``."""
    table = _symbol_table(left, right)
    floors = _merged_floors(left, right, restricted)
    return tuple(
        candidate
        for candidate in candidates
        if _admissible(candidate, table)
        and all(
            not isinstance(candidate.get(name), int) or candidate[name] >= low
            for name, low in floors.items()
        )
    )


@_trace.traced("decide.counterexample_search")
def _symbolic_witness(
    left: Diagram,
    right: Diagram,
    tensors: tuple[SymbolicTensor, SymbolicTensor],
    candidates: Sequence[Mapping[str, CheckAssignmentValue]],
    tolerance: float,
    floors: Mapping[str, int] | None = None,
) -> EntryWitness | None:
    """The first entry where the two contractions differ at an admissible candidate."""
    try:
        return find_entry_witness(
            *tensors, _witness_assignments(left, right, candidates, floors), tolerance=tolerance
        )
    except _SYMBOLIC_ERRORS:
        return None


def refute_by_symbolic_witness(
    left: Diagram,
    right: Diagram,
    *,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> EntryWitness | None:
    """The symbolic witness search alone, at ``samples`` (default :func:`sample_grid`); None
    when the contractions are unavailable, agree, or no differing entry is found."""
    _check_args(
        left,
        right,
        None,
        DEFAULT_GUARD,
        samples,
        0,
        True,
        True,
        0,
        tolerance,
        1,
        True,
        DECIDE_SATURATION_LIMITS,
    )
    family_l = _family(left)
    family_r = _family(right)
    run = _symbolic_match(family_l, family_r)
    if run.matched or run.tensors is None:
        return None
    candidates = tuple(samples) if samples is not None else sample_grid(family_l, family_r)
    return _symbolic_witness(family_l, family_r, run.tensors, candidates, tolerance)


@dataclass(frozen=True, slots=True)
class _SaturationRun:
    """The saturation rung's outcome: its report, the edge certificates and their ``DEFERRED``
    constraints when the sides merged, and a reason."""

    report: SaturationReport | None
    certificates: tuple[Certificate, ...] | None
    assumptions: tuple[DimensionConstraint, ...]
    reason: str


@_trace.traced("decide.saturation")
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
            if not views_isomorphic(comparison_view(certificate.final), child):
                return _SaturationRun(
                    report, None, (), f"edge {position} does not reach its child e-node"
                )
            certificates.append(certificate)
    except _NORMAL_FORM_ERRORS as exc:
        return _SaturationRun(report, None, (), f"failed: {type(exc).__name__}: {exc}")
    deferred = assumed(
        c
        for certificate in certificates
        for step in certificate.steps
        for c in step.dimension_constraints
    )
    reason = f"{len(edges)} edge(s)" if edges else "0 edge(s), both sides one e-node"
    return _SaturationRun(report, tuple(certificates), deferred, reason)


def _induction_assumptions(
    base: Decision, result: InductionResult
) -> tuple[DimensionConstraint, ...]:
    """``base``'s assumptions, then every assumed dimension equation of ``result``'s tier
    steps, by :func:`assumed`."""
    steps = (c for tier in result.tiers for step in tier.steps for c in step.dimension_constraints)
    return assumed((*base.assumptions, *steps))


def _instantiate_index(diagram: Diagram, index: str, value: int) -> Diagram:
    """``diagram`` with multiplicity ``index`` set to ``value`` in boxes, scalar and phases."""
    if index in free_mult_symbols(diagram):
        diagram = instantiate_symbol(diagram, index, value)
    else:
        diagram = diagram.copy()
    if index in diagram.scalar.free_symbols:
        diagram.set_scalar(diagram.scalar.substitute({index: value}))
    diagram = substitute_index_phases(diagram, index, value)
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
    mixed: bool = False,
    restrict: Restriction | str | Mapping[str, str] | None = None,
    describe: Callable[[Diagram], str] | None = None,
    prove_equal: bool = True,
) -> Decision:
    """Decide whether ``left`` and ``right`` are equal for every value of their free symbols,
    or for the values ``restrict`` covers; ``describe`` gives the Lean refutation's and the
    proof's diagram comments.

    Runs the module docstring's ladder; ``samples`` replaces :func:`sample_grid` for the
    oracle rung and, cut to its symbols, for the induction base case; ``max_depth`` bounds
    nested induction; ``rules`` also replaces the saturation rung's default rules. The
    induction step is tried left-to-right, then right-to-left. When either side has a
    discard, or ``mixed`` is set, the ladder runs on both doubled diagrams and the decision
    is ``DOUBLED``, the rung that settled it in ``inner_method``. A restriction with clauses
    runs :func:`decide_restricted`. ``prove_equal=False`` leaves EQUAL unproved and unchanged.
    """
    if not isinstance(prove_equal, bool):
        raise DecideGrammarError(f"prove_equal must be a bool, got {prove_equal!r}")
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
    options: dict[str, object] = {
        "rules": rules,
        "guard": guard,
        "samples": samples,
        "max_samples": max_samples,
        "use_symbolic": use_symbolic,
        "use_induction": use_induction,
        "max_depth": max_depth,
        "tolerance": tolerance,
        "max_elements": max_elements,
        "use_saturation": use_saturation,
        "saturation_limits": saturation_limits,
        "mixed": mixed,
    }
    if restrict is not None:
        from archytaszx.semantics.restrict import parse_restriction, validate

        restriction = validate(parse_restriction(restrict), left, right)
        if not restriction.empty:
            decision = decide_restricted(left, right, restriction, **options)
            return _certify(decision, left, right, describe, prove_equal)
    decision = _decide(left, right, floors=None, **options)  # type: ignore[arg-type]
    return _certify(decision, left, right, describe, prove_equal)


def lean_refutation(
    left: Diagram,
    right: Diagram,
    point: Mapping[str, CheckAssignmentValue] | None,
    describe: Callable[[Diagram], str] | None = None,
    index: Sequence[int] | None = None,
) -> str:
    """The Lean refutation of ``left = right`` at ``point`` (or a default sample point), over
    the doubled diagrams when either side has a discard, the boundary values ``index`` tried
    first; raises :class:`~archytaszx.semantics.lean_net.NetUnsupported` when there is none."""
    from archytaszx.semantics.lean_net import NetUnsupported
    from archytaszx.semantics.lean_refute import export_refutation

    if not (is_pure(left) and is_pure(right)):
        left, right = double(left), double(right)
        index = None
    try:
        return export_refutation(left, right, dict(point or {}), describe=describe, index=index)
    except NetUnsupported:
        raise
    except Exception as exc:
        raise NetUnsupported(f"{type(exc).__name__}: {exc}") from exc


def _certify(
    decision: Decision,
    left: Diagram,
    right: Diagram,
    describe: Callable[[Diagram], str] | None,
    prove_equal: bool,
) -> Decision:
    """``decision`` with its Lean refutation when UNEQUAL past rung 1 and with its proof when
    EQUAL and ``prove_equal``; the uncertified verdict, the missing one named in its reason,
    when there is none."""
    from archytaszx.semantics.lean_net import NetUnsupported

    if _NESTED.get():
        return decision
    if decision.verdict is EqualityVerdict.EQUAL:
        if not prove_equal:
            return decision
        from archytaszx.semantics.certify_decision import decision_proof

        with _trace.span("decide.proof"):
            proof, why = decision_proof(decision, left, right, describe)
        if proof is None:
            return replace(
                decision,
                verdict=EqualityVerdict.EQUAL_UNCERTIFIED,
                reason=f"{decision.reason}; uncertified: no proof the kernel re-checks ({why})",
            )
        return replace(decision, proof=proof)
    if decision.verdict is not EqualityVerdict.UNEQUAL:
        return decision
    if decision.method is DecisionMethod.INTERFACE or (
        decision.method is DecisionMethod.DOUBLED
        and decision.inner_method is DecisionMethod.INTERFACE
    ):
        return decision
    point = decision.counterexample
    if decision.restriction is not None and decision.counterexample_original is not None:
        point = decision.counterexample_original
    induced = decision.induction
    if point is None and induced is not None and induced.counterexample is not None:
        point = {**dict(induced.witness), induced.index: induced.counterexample}
    witness = decision.witness
    index = witness.index if witness is not None and point is decision.counterexample else None
    try:
        with _trace.span("decide.refutation"):
            source = lean_refutation(left, right, point, describe, index)
    except NetUnsupported as exc:
        return replace(
            decision,
            verdict=EqualityVerdict.UNEQUAL_UNCERTIFIED,
            reason=f"{decision.reason}; uncertified: no Lean refutation ({exc})",
        )
    return replace(decision, refutation=source)


def decide_restricted(
    left: Diagram, right: Diagram, restriction: Restriction, **options: object
) -> Decision:
    """The restricted decision: the family decided first, its EQUAL kept and an in-range
    counterexample kept, otherwise the ladder on the restricted pair under the restriction's
    floors."""
    from archytaszx.semantics.restrict import in_domain, restricted_samples, subst_pair, to_original

    with _trace.span("decide.restriction.family"):
        family = _decide(left, right, floors=None, **options)  # type: ignore[arg-type]
    if family.verdict is EqualityVerdict.EQUAL:
        return replace(family, restriction=restriction, family=family, family_holds=True)
    with _trace.span("decide.restriction.substitute"):
        pair = subst_pair(left, right, restriction)
    if family.verdict is EqualityVerdict.UNEQUAL:
        point = family.counterexample
        induced = family.induction
        if point is None and induced is not None and induced.counterexample is not None:
            point = {**dict(induced.witness), induced.index: induced.counterexample}
        if family.method is DecisionMethod.INTERFACE:
            reason = _interface_reason(_family(pair[0]), _family(pair[1]))
            if reason is not None:
                return replace(
                    family,
                    reason=reason,
                    restriction=restriction,
                    family=family,
                    restricted_pair=pair,
                )
        elif point is not None and in_domain(restriction, point):
            return replace(
                family,
                restriction=restriction,
                family=family,
                counterexample_original=MappingProxyType(dict(point)),
            )
    samples = options.get("samples")
    if samples is not None:
        options = {**options, "samples": restricted_samples(restriction, samples)}  # type: ignore[arg-type]
    with _trace.span("decide.restriction.pair"):
        inner = _decide(*pair, floors=restriction.floors, **options)  # type: ignore[arg-type]
    original = None
    reason = inner.reason
    if inner.counterexample is not None:
        original = MappingProxyType(to_original(restriction, inner.counterexample))
        where = ", ".join(f"{k} = {v}" for k, v in original.items())
        reason = f"{reason}; in the original symbols at {where}"
    return replace(
        inner,
        reason=reason,
        restriction=restriction,
        family=family,
        family_holds=False,
        restricted_pair=pair,
        counterexample_original=original,
        samples_checked=family.samples_checked + inner.samples_checked,
    )


def _decide(
    left: Diagram,
    right: Diagram,
    *,
    rules: Sequence[Rule] | None,
    guard: TerminationGuard,
    samples: Sequence[Mapping[str, CheckAssignmentValue]] | None,
    max_samples: int,
    use_symbolic: bool,
    use_induction: bool,
    max_depth: int,
    tolerance: float,
    max_elements: int,
    use_saturation: bool,
    saturation_limits: SaturationLimits,
    mixed: bool,
    floors: Mapping[str, int] | None,
) -> Decision:
    """The ladder of :func:`decide_equal`, sampling and contracting under ``floors``."""
    if mixed or not (is_pure(left) and is_pure(right)):
        doubled = (double(left), double(right))
        inner = _decide(
            *doubled,
            mixed=False,
            floors=floors,
            rules=rules,
            guard=guard,
            samples=samples,
            max_samples=max_samples,
            use_symbolic=use_symbolic,
            use_induction=use_induction,
            max_depth=max_depth,
            tolerance=tolerance,
            max_elements=max_elements,
            use_saturation=use_saturation,
            saturation_limits=saturation_limits,
        )
        return replace(
            inner,
            method=DecisionMethod.DOUBLED,
            inner_method=inner.method,
            doubled=doubled,
            reason=f"doubled, {inner.method.value}: {inner.reason}",
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
        with _trace.span("decide.normal_form"):
            nf_l = normal_form(family_l, rules=rules, guard=guard)
            nf_r = normal_form(family_r, rules=rules, guard=guard)
        certificates = (
            certify(nf_l.source, nf_l.results, label="left normal form"),
            certify(nf_r.source, nf_r.results, label="right normal form"),
        )
        nf_assumptions = assumed((*nf_l.assumed_constraints, *nf_r.assumed_constraints))
        nf_equal = same_normal_form(nf_l, nf_r)
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

    candidates = tuple(samples) if samples is not None else sample_grid(family_l, family_r, floors)
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
    sampled = oracle_summary(checked, oracle.refusals)
    if nf_equal:
        conditional = (
            f", conditional on the assumed dimension equation(s) "
            f"{', '.join(c.equation for c in nf_assumptions)}"
            if nf_assumptions
            else ""
        )
        return finish(
            EqualityVerdict.EQUAL,
            DecisionMethod.NORMAL_FORM,
            f"normal forms agree{conditional}; {sampled}",
            samples_checked=checked,
        )

    notes = [nf_reason, sampled]
    if use_saturation:
        run = _run_saturation(family_l, family_r, rules, saturation_limits)
        saturation = run.report
        if run.certificates is not None:
            conditional = (
                ", conditional on the assumed dimension equation(s) "
                f"{', '.join(c.equation for c in run.assumptions)}; {sampled}"
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
        symbolic = _symbolic_match(family_l, family_r, floors)
        if symbolic.matched:
            return finish(
                EqualityVerdict.EQUAL,
                DecisionMethod.SYMBOLIC_CONTRACTION,
                f"symbolic contractions agree: {symbolic.reason}",
                assumptions=(),
                samples_checked=checked,
                floors_used=symbolic.floors_used,
            )
        witness = (
            None
            if symbolic.tensors is None
            else _symbolic_witness(
                family_l, family_r, symbolic.tensors, candidates, tolerance, floors
            )
        )
        if witness is not None:
            return finish(
                EqualityVerdict.UNEQUAL,
                DecisionMethod.SYMBOLIC_WITNESS,
                f"symbolic contractions differ at {dict(witness.assignment)!r}, boundary index "
                f"{witness.index}: {witness.left} vs {witness.right}",
                assumptions=(),
                counterexample=witness.assignment,
                comparison=ComparisonResult(
                    EqualityMode.EXACT,
                    False,
                    f"entry {witness.index} differs by {witness.deviation}",
                    witness.deviation,
                ),
                samples_checked=checked,
                witness=witness,
            )
        notes.append(f"symbolic: {symbolic.reason}")
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
            floors=floors,
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


@_trace.traced("decide.induction")
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
    floors: Mapping[str, int] | None = None,
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
    base_options: dict[str, object] = {
        "rules": rules,
        "guard": guard,
        "samples": None if samples is None else _restrict_samples(samples, index, *pair),
        "max_samples": max_samples,
        "use_symbolic": use_symbolic,
        "use_induction": True,
        "max_depth": max_depth - 1,
        "tolerance": tolerance,
        "max_elements": max_elements,
        "use_saturation": use_saturation,
        "saturation_limits": saturation_limits,
    }
    with _trace.span("decide.induction.base"):
        token = _NESTED.set(True)
        try:
            base_decision = (
                decide_equal(pair[0], pair[1], **base_options)  # type: ignore[arg-type]
                if floors is None
                else _decide(pair[0], pair[1], mixed=False, floors=floors, **base_options)  # type: ignore[arg-type]
            )
        finally:
            _NESTED.reset(token)
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
            with _trace.span("decide.induction.step", "reversed" if reversed_ else "forward"):
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
        from archytaszx.semantics.induction_kernel import induction_theorem

        with _trace.span("decide.induction.kernel"):
            theorem, why = induction_theorem(
                left, right, result, reversed_, pair, base_decision, rules
            )
        built = "kernel proof built" if theorem is not None else f"no kernel proof: {why}"
        return (
            EqualityVerdict.EQUAL,
            DecisionMethod.INDUCTION,
            f"{proof}; holds for {index} >= 0; {built}",
            {
                "induction": result,
                "induction_reversed": reversed_,
                "assumptions": _induction_assumptions(base_decision, result),
                "induction_proof": theorem,
            },
        )
    return "; reversed: ".join(reasons)
