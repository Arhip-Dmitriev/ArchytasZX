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

"""Symbolic dimension expressions: concrete integers, symbols, and products/powers thereof.

A :class:`Dim` is normalized and compared through a single canonical
``sympy`` expression (see :meth:`Dim.to_sympy`). Dimension symbols carry the
sympy assumptions ``positive=True, integer=True`` and exponent symbols carry
``integer=True, nonnegative=True``, so sympy's own normalization (e.g.
folding ``d**n * d**m`` to ``d**(n+m)``) is sound over this domain. This
supports the spec invariant that dimension is stored per port, never
as a single global parameter: this module only supplies dimension *values*,
it holds no ambient "current dimension" state of its own.

Phase 10 note: :meth:`Dim.unify` here is a deliberate placeholder. It never
guesses, factors, or partially solves -- it only recognizes the cases it can
decide outright (both concrete, syntactically equal, or one side a bare
symbol) and defers everything else as a residual constraint. The real
unifier, which must actually solve constraints such as ``d = d1 * d2``,
arrives in Phase 10 and should replace the body of this method, not its
contract: callers rely on it never reporting success for something it did
not verify.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Union, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker


class DimensionError(Exception):
    """Base class for all errors raised by this module."""


class DimensionDomainError(DimensionError):
    """A value is outside the mathematical domain of dimensions or exponents.

    Raised for concrete dimensions that are not positive integers, exponents
    that are not non-negative integers, and substitution values that violate
    those same domains.
    """


class DimensionGrammarError(DimensionError):
    """An expression falls outside the dimension grammar.

    The grammar is: concrete positive integers, positive-integer symbols,
    products of these, and powers of these with a non-negative-integer
    exponent. Sums, quotients, roots, floats, and anything else are rejected.
    """


def _dimension_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, positive=True, integer=True)


def _exponent_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, integer=True, nonnegative=True)


def _check_exponent_domain(expr: sp.Expr) -> None:
    if expr.is_Integer:
        if int(expr) < 0:
            raise DimensionDomainError(f"exponent must be a non-negative integer, got {expr}")
        return
    if expr.is_Symbol:
        assumptions = expr.assumptions0
        if not (assumptions.get("integer") and assumptions.get("nonnegative")):
            raise DimensionGrammarError(
                f"symbol {expr} is not a valid exponent symbol "
                "(exponent symbols must be non-negative integers)"
            )
        return
    if expr.is_Add or expr.is_Mul:
        for arg in expr.args:
            _check_exponent_domain(arg)
        return
    raise DimensionGrammarError(f"exponent {expr} is outside the exponent grammar")


def _check_dimension_domain(expr: sp.Expr) -> None:
    if expr.is_Integer:
        if int(expr) < 1:
            raise DimensionDomainError(f"concrete dimension must be >= 1, got {expr}")
        return
    if expr.is_Symbol:
        assumptions = expr.assumptions0
        if not (assumptions.get("integer") and assumptions.get("positive")):
            raise DimensionGrammarError(
                f"symbol {expr} is not a valid dimension symbol "
                "(dimension symbols must be positive integers)"
            )
        return
    if expr.is_Mul:
        for arg in expr.args:
            _check_dimension_domain(arg)
        return
    if expr.is_Pow:
        base, exponent = expr.args
        _check_dimension_domain(base)
        _check_exponent_domain(exponent)
        return
    raise DimensionGrammarError(
        f"expression {expr} is outside the dimension grammar "
        "(sums, quotients, roots, and floats are not allowed)"
    )


DimSymbolKey = Union[str, "Dim"]
DimSubstituteValue = Union[int, "Dim"]


class Dim:
    """An immutable, hashable dimension expression.

    A dimension is a positive integer (>= 1), possibly symbolic. Valid forms
    are a concrete integer, a symbol, a product of dimensions, and a power of
    a dimension by a non-negative integer exponent (itself concrete or
    symbolic). This is the per-port dimension value referenced by the
    "dimension is stored per port" invariant in the spec; ``Dim`` itself
    holds no global or default dimension.

    Two ``Dim`` values are equal iff their normalized expressions are
    mathematically identical, e.g. ``d1 * d2 == d2 * d1`` and
    ``d * d == d ** 2``.
    """

    __slots__ = ("_expr",)
    _expr: sp.Expr

    def __init__(self, value: int | str | Dim) -> None:
        """Build a Dim from a concrete positive int, a symbol name, or a copy of another Dim.

        Raises DimensionDomainError for non-positive integers and
        DimensionGrammarError for any other invalid input.
        """
        if isinstance(value, bool):
            raise DimensionGrammarError(f"Dim does not accept bool, got {value!r}")
        if isinstance(value, Dim):
            self._expr = value._expr
        elif isinstance(value, int):
            if value < 1:
                raise DimensionDomainError(f"concrete dimension must be >= 1, got {value}")
            self._expr = sp.Integer(value)
        elif isinstance(value, str):
            expr = _dimension_symbol(value)
            _check_dimension_domain(expr)
            self._expr = expr
        else:
            raise TypeError(f"Dim() accepts int, str, or Dim, got {type(value).__name__}")

    @classmethod
    def concrete(cls, value: int) -> Dim:
        """Build a Dim from a concrete positive integer. Alias for Dim(value)."""
        return cls(value)

    @classmethod
    def symbol(cls, name: str) -> Dim:
        """Build a Dim from a fresh positive-integer symbol name. Alias for Dim(name)."""
        return cls(name)

    @classmethod
    def _from_expr(cls, expr: sp.Expr) -> Dim:
        normalized = sp.powsimp(expr, force=True)
        _check_dimension_domain(normalized)
        obj = cls.__new__(cls)
        obj._expr = normalized
        return obj

    def to_sympy(self) -> sp.Expr:
        """Escape hatch returning the underlying canonical sympy expression.

        This is the only sanctioned way to reach the sympy backend; every
        other public method keeps sympy as an implementation detail so later
        phases can work entirely in terms of Dim.
        """
        return self._expr

    @property
    def is_concrete(self) -> bool:
        """True iff this expression has no free symbols, i.e. it denotes a known integer."""
        return not self._expr.free_symbols

    @property
    def free_symbols(self) -> frozenset[str]:
        """The names of all symbols (dimension or exponent) appearing in this expression."""
        return frozenset(str(s.name) for s in self._expr.free_symbols)

    def to_int(self) -> int:
        """Return the concrete int value of this dimension.

        This is the single gate through which a symbolic dimension becomes a
        Python int. Everywhere downstream that needs to build a numpy array
        or dense tensor must go through this method, and it is exactly the
        enforcement point of the invariant "never construct a matrix or
        dense tensor while any dimension or count in scope is symbolic":
        calling it on a symbolic Dim raises rather than silently proceeding.
        """
        if not self.is_concrete:
            raise DimensionDomainError(
                f"cannot convert symbolic dimension {self} to int; "
                f"free symbols: {sorted(self.free_symbols)}"
            )
        return int(self._expr)

    @staticmethod
    def _key_symbol_name(key: DimSymbolKey) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, Dim):
            if not key._expr.is_Symbol:
                raise DimensionGrammarError(f"substitution key must be a bare symbol, got {key}")
            return str(key._expr.name)
        raise TypeError(f"substitution key must be str or Dim, got {type(key).__name__}")

    def substitute(self, mapping: Mapping[DimSymbolKey, DimSubstituteValue]) -> Dim:
        """Return a new Dim with symbols replaced by concrete integers.

        Keys may be symbol names (str) or bare-symbol Dims; values may be
        int or a concrete Dim. The mapping need not be total: symbols not
        mentioned are left symbolic (e.g. substituting only d in d ** n
        leaves 2 ** n). Substitution reaches into exponents as well as
        bases. Substituted values are validated against the domain of the
        symbol they replace (positive integers for dimension symbols,
        non-negative integers for exponent symbols). This Dim is never
        mutated; a new Dim is always returned.
        """
        resolved: dict[str, int] = {}
        for key, value in mapping.items():
            name = self._key_symbol_name(key)
            if isinstance(value, Dim):
                if not value.is_concrete:
                    raise DimensionDomainError(
                        f"substitution value for {name!r} must be concrete, got {value}"
                    )
                resolved[name] = value.to_int()
            elif isinstance(value, int) and not isinstance(value, bool):
                resolved[name] = value
            else:
                raise TypeError(
                    f"substitution value for {name!r} must be int or Dim, "
                    f"got {type(value).__name__}"
                )

        subs_dict: dict[sp.Symbol, sp.Integer] = {}
        # Sorted by name: sympy's free_symbols is a plain set whose Symbol hashing bottoms
        # out in the PYTHONHASHSEED-dependent string hash of the name, so with several
        # out-of-domain values in one call the error below would name a different symbol
        # first across processes.
        for sym in sorted(self._expr.free_symbols, key=lambda s: str(s.name)):
            sym_name = str(sym.name)
            if sym_name not in resolved:
                continue
            value = resolved[sym_name]
            assumptions = sym.assumptions0
            if assumptions.get("positive"):
                if value < 1:
                    raise DimensionDomainError(
                        f"dimension symbol {sym_name!r} requires a positive integer, got {value}"
                    )
            else:
                if value < 0:
                    raise DimensionDomainError(
                        f"exponent symbol {sym_name!r} requires a non-negative integer, "
                        f"got {value}"
                    )
            subs_dict[sym] = sp.Integer(value)

        new_expr = self._expr.subs(subs_dict)
        return Dim._from_expr(new_expr)

    def __mul__(self, other: Dim | int) -> Dim:
        """Multiply two dimensions, returning a normalized product."""
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            other = Dim(other)
        if not isinstance(other, Dim):
            return NotImplemented
        return Dim._from_expr(self._expr * other._expr)

    def __rmul__(self, other: Dim | int) -> Dim:
        """Support ``2 * d`` by delegating to the commutative __mul__."""
        return self.__mul__(other)

    @staticmethod
    def _coerce_exponent(exponent: Dim | int) -> sp.Expr:
        if isinstance(exponent, bool):
            raise DimensionGrammarError(f"exponent does not accept bool, got {exponent!r}")
        if isinstance(exponent, int):
            if exponent < 0:
                raise DimensionDomainError(f"exponent must be >= 0, got {exponent}")
            return sp.Integer(exponent)
        if isinstance(exponent, Dim):
            expr = exponent._expr
            if expr.is_Integer:
                if int(expr) < 0:
                    raise DimensionDomainError(f"exponent must be >= 0, got {expr}")
                return expr
            if expr.is_Symbol:
                return _exponent_symbol(str(expr.name))
            raise DimensionGrammarError(
                f"exponent {expr} must be a concrete non-negative integer or a single symbol"
            )
        raise TypeError(f"exponent must be int or Dim, got {type(exponent).__name__}")

    def __pow__(self, exponent: Dim | int) -> Dim:
        """Raise this dimension to a non-negative integer (possibly symbolic) power."""
        exponent_expr = self._coerce_exponent(exponent)
        return Dim._from_expr(self._expr**exponent_expr)

    def __eq__(self, other: object) -> bool:
        """Exact mathematical equality after normalization.

        Only Dim-to-Dim comparison is supported: comparing against a
        non-Dim (e.g. ``dim == 2``) returns NotImplemented rather than
        silently coercing, so such a comparison evaluates to False instead
        of implying an int is ever "the same thing" as a Dim.
        """
        if not isinstance(other, Dim):
            return NotImplemented
        return bool(self._expr == other._expr)

    def __hash__(self) -> int:
        """Hash agrees with __eq__: equal Dims (post-normalization) hash equal."""
        return hash(self._expr)

    def __repr__(self) -> str:
        return f"Dim({self._expr})"

    def __str__(self) -> str:
        return str(self._expr)

    def unify(self, other: Dim) -> UnifyResult:
        """Placeholder unifier (Phase 10 will replace this with a real solver).

        Resolves only the cases that can be decided without guessing:
        concrete-vs-concrete (definite success or definite failure),
        syntactically-equal-after-normalization (success, no constraints),
        and a bare symbol against anything that does not contain it (a
        binding). If a symbol occurs in the other side as a proper subterm
        (e.g. d against d*e, or d against d**n), it is deferred rather than
        bound: such an equation is satisfiable only in degenerate cases
        (d = d*e holds at e == 1; d = d**n holds at n == 1, or at d == 1 for
        any n), so neither binding it as success nor reporting failure would
        be honest. Everything else is reported as deferred with the pair
        recorded as a residual constraint -- it is never guessed at,
        partially factored, or silently reported as equal.
        """
        if not isinstance(other, Dim):
            raise TypeError(f"unify() requires a Dim, got {type(other).__name__}")
        if self.is_concrete and other.is_concrete:
            status = UnifyStatus.SUCCESS if self.to_int() == other.to_int() else UnifyStatus.FAILURE
            return UnifyResult(status=status)
        if self._expr == other._expr:
            return UnifyResult(status=UnifyStatus.SUCCESS)
        if self._expr.is_Symbol:
            if str(self._expr.name) in other.free_symbols:
                return UnifyResult(status=UnifyStatus.DEFERRED, constraints=((self, other),))
            return UnifyResult(status=UnifyStatus.SUCCESS, bindings={str(self._expr.name): other})
        if other._expr.is_Symbol:
            if str(other._expr.name) in self.free_symbols:
                return UnifyResult(status=UnifyStatus.DEFERRED, constraints=((self, other),))
            return UnifyResult(status=UnifyStatus.SUCCESS, bindings={str(other._expr.name): self})
        return UnifyResult(status=UnifyStatus.DEFERRED, constraints=((self, other),))


class UnifyStatus(enum.Enum):
    """The three outcomes the Phase 1 unify placeholder can report."""

    SUCCESS = "success"
    FAILURE = "failure"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class UnifyResult:
    """The result of Dim.unify: a status plus any bindings or residual constraints found.

    ``bindings`` maps a symbol name to the Dim it was unified with.
    ``constraints`` lists pairs of Dims asserted equal that this placeholder
    declined to solve; a real solver (Phase 10) would resolve these further.
    """

    status: UnifyStatus
    bindings: Mapping[str, Dim] = field(default_factory=dict)
    constraints: tuple[tuple[Dim, Dim], ...] = ()

    @property
    def is_success(self) -> bool:
        return self.status is UnifyStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.status is UnifyStatus.FAILURE

    @property
    def is_deferred(self) -> bool:
        return self.status is UnifyStatus.DEFERRED


_MAX_UNIFY_ALL_PASSES = 32
"""Iteration budget for :func:`unify_all`'s bindings fixpoint. Module-level so a test can
patch it low, mirroring :mod:`qufzx.rewrite.match`'s ``_MAX_FIXPOINT_PASSES``."""


@dataclass(frozen=True)
class UnifyAllResult:
    """The result of :func:`unify_all`: a status, accumulated bindings, and residual pairs.

    ``residual_pairs`` holds every pair left unresolved once the loop stopped -- either
    because the fixpoint genuinely stabilised with real residual constraints left (the
    ordinary ``DEFERRED`` case), or because :data:`_MAX_UNIFY_ALL_PASSES` was exhausted
    before it could stabilise, in which case ``residual_pairs`` holds whatever was still
    unresolved on that final, non-converged pass -- a snapshot of an interrupted
    computation, not a decided answer.

    ``exhausted`` is the discriminator between those two cases: ``False`` for an ordinary,
    converged ``DEFERRED`` (or ``SUCCESS``/``FAILURE``), ``True`` only when the pass budget
    ran out first. Both are reported as ``status=DEFERRED`` (in neither case has this
    function decided FAILURE or SUCCESS), but they are not the same finding and a caller
    that treats them identically is silently trusting an interrupted computation as much as
    a completed one -- see :mod:`qufzx.diagram.validate`'s ``_check_generator_policy`` for a
    caller that fails closed on ``exhausted`` (a hard error) rather than folding it into the
    ordinary deferred-constraint bookkeeping.

    General rule for the next bounded fixpoint this codebase grows (this one mirrors
    :mod:`qufzx.rewrite.match`'s ``_MAX_FIXPOINT_PASSES``, which already fails closed by
    rejecting the match outright on exhaustion): a bounded fixpoint's exhaustion path must
    be distinguishable from its success path at every call site, not just internally. A
    ``return`` statement that produces the same shape of result on both paths -- as this
    function's exhaustion return used to, before this field existed, always with
    ``residual_pairs=()`` regardless of what was actually still outstanding -- lets an
    undecided node read as decided-and-fine three calls away from where the budget was
    actually exhausted.
    """

    status: UnifyStatus
    bindings: Mapping[str, Dim] = field(default_factory=dict)
    residual_pairs: tuple[tuple[Dim, Dim], ...] = ()
    exhausted: bool = False
    declined_bindings: Mapping[str, Dim] = field(default_factory=dict)
    """Every binding to a non-concrete ``Dim`` (e.g. ``d := e``) that a pairwise
    :meth:`Dim.unify` call produced but this function declined to fold into ``bindings`` --
    see :func:`unify_all`'s own inline comment for why only concrete bindings are ever
    accumulated for resolution.

    It exists so a caller can see that an assumption was made even on a ``SUCCESS``: a node
    whose legs unify only via such a binding (legs ``d`` and ``e``) would otherwise report
    ``SUCCESS`` with nothing to show for it. Populating it changes no verdict and no
    caller's behavior.

    :mod:`qufzx.diagram.validate`'s ``_check_generator_policy`` does not yet surface this as
    a reported issue -- see that function for why, and
    ``tests/test_symbolic_dimension_sweep.py`` for the pin of the current silent behavior.
    Doing so would ripple into :mod:`qufzx.rewrite.engine`'s step-8 deferred-issue
    bookkeeping, which is Phase 6/10 certificate territory.
    """

    @property
    def is_success(self) -> bool:
        return self.status is UnifyStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.status is UnifyStatus.FAILURE

    @property
    def is_deferred(self) -> bool:
        return self.status is UnifyStatus.DEFERRED


def unify_all(dims: Sequence[Dim]) -> UnifyAllResult:
    """Resolve a multiset of Dims that must all be pairwise equal to one shared value.

    Sorts ``dims`` by a canonical key before doing anything else, so the result depends
    only on the multiset of dims given, never their input order. Runs :meth:`Dim.unify`
    to a bounded fixpoint, accumulating concrete bindings monotonically (a name rebound to
    a different concrete value is FAILURE); a binding to a non-concrete Dim is never folded
    into ``bindings`` or substituted through; it is recorded on
    :attr:`UnifyAllResult.declined_bindings` rather than dropped. Returns
    FAILURE on any non-unifiable pair, DEFERRED with every pair still unresolved once the
    fixpoint stabilises, or SUCCESS. A DEFERRED returned because
    :data:`_MAX_UNIFY_ALL_PASSES` ran out first carries ``exhausted=True`` and holds only
    what the final, non-converged pass left unresolved -- see :attr:`UnifyAllResult.exhausted`.
    Raises only :class:`DimensionError` subclasses.
    """
    ordered = sorted(dims, key=lambda d: sp.srepr(d.to_sympy()))
    if len(ordered) < 2:
        return UnifyAllResult(status=UnifyStatus.SUCCESS)

    bindings: dict[str, Dim] = {}
    declined: dict[str, Dim] = {}
    residual: dict[str, tuple[Dim, Dim]] = {}

    for _pass_index in range(_MAX_UNIFY_ALL_PASSES):
        pass_start_bindings = dict(bindings)
        residual = {}
        concrete_bindings = cast(Mapping[DimSymbolKey, DimSubstituteValue], bindings)
        base = ordered[0].substitute(concrete_bindings) if bindings else ordered[0]
        for other in ordered[1:]:
            concrete_bindings = cast(Mapping[DimSymbolKey, DimSubstituteValue], bindings)
            resolved_other = other.substitute(concrete_bindings) if bindings else other
            result = base.unify(resolved_other)
            if result.is_failure:
                return UnifyAllResult(status=UnifyStatus.FAILURE)
            if result.is_deferred:
                key = f"{base!r}|{resolved_other!r}"
                residual[key] = (base, resolved_other)
                continue
            new_concrete = {
                name: value for name, value in result.bindings.items() if value.is_concrete
            }
            declined.update(
                {name: value for name, value in result.bindings.items() if not value.is_concrete}
            )
            for name, value in new_concrete.items():
                existing = bindings.get(name)
                if existing is not None and existing != value:
                    return UnifyAllResult(status=UnifyStatus.FAILURE)
                bindings[name] = value
            if new_concrete:
                new_concrete_typed = cast(Mapping[DimSymbolKey, DimSubstituteValue], new_concrete)
                base = base.substitute(new_concrete_typed)
        if bindings == pass_start_bindings:
            if residual:
                return UnifyAllResult(
                    status=UnifyStatus.DEFERRED,
                    bindings=dict(bindings),
                    declined_bindings=dict(declined),
                    residual_pairs=tuple(residual.values()),
                )
            return UnifyAllResult(
                status=UnifyStatus.SUCCESS,
                bindings=dict(bindings),
                declined_bindings=dict(declined),
            )

    # Budget exhausted without stabilising: report whatever the final, non-converged pass
    # actually left unresolved, and mark it ``exhausted`` -- see UnifyAllResult's own
    # docstring for why this must not be a bare, contentless DEFERRED indistinguishable from
    # a genuinely converged one.
    return UnifyAllResult(
        status=UnifyStatus.DEFERRED,
        bindings=dict(bindings),
        declined_bindings=dict(declined),
        residual_pairs=tuple(residual.values()),
        exhausted=True,
    )
