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

:meth:`Dim.unify` decides one asserted equality: it cancels common factors,
then either decides the reduced equation outright or reports it as a residual
constraint. It never reports SUCCESS for an equation it did not verify and
never FAILURE for one that has a solution over the positive integers.

:func:`solve` runs :meth:`Dim.unify` over a whole system of asserted
equalities to a monotone fixpoint bounded by :data:`_MAX_SOLVE_PASSES`,
resolving symbolic bindings against each other. :func:`unify_all` is the
special case of one shared value.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
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


def _check_symbol_name(name: str) -> None:
    """Reject a symbol name that is not a bare identifier."""
    if not isinstance(name, str) or not name.isidentifier():
        raise DimensionGrammarError(
            f"symbol name must be a bare identifier, got {name!r}; a name like 'd*e' "
            "builds a symbol that renders identically to the product d*e but is a "
            "different Dim"
        )


def _dimension_symbol(name: str) -> sp.Symbol:
    _check_symbol_name(name)
    return sp.Symbol(name, positive=True, integer=True)


def _exponent_symbol(name: str) -> sp.Symbol:
    _check_symbol_name(name)
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
        _check_symbol_name(str(expr.name))
        return
    if expr.is_Add or expr.is_Mul:
        for arg in expr.args:
            _check_exponent_domain(arg)
        return
    if expr.is_Pow:
        # Closes the grammar under substitution: rewriting a dimension symbol by a power
        # multiplies exponents, and sympy folds m*m to m**2.
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
        _check_symbol_name(str(expr.name))
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
    def from_sympy(cls, expr: sp.Expr) -> Dim:
        """Build a Dim from a sympy expression, rejecting anything outside the dimension grammar."""
        return cls._from_expr(sp.sympify(expr))

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
        int or a concrete Dim. The mapping need not be total: symbols not mentioned are left
        symbolic (e.g. substituting only d in d ** n leaves 2 ** n).
        Substitution reaches into exponents as well as bases. Values are
        validated against the domain of the symbol they replace (positive
        integers for dimension symbols, non-negative integers for exponent
        symbols). This Dim is never mutated; a new Dim is always returned.
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
                        f"exponent symbol {sym_name!r} requires a non-negative integer, got {value}"
                    )
            subs_dict[sym] = sp.Integer(value)

        new_expr = self._expr.subs(subs_dict)
        return Dim._from_expr(new_expr)

    def abstract(
        self, *, avoid: Iterable[str] = (), stem: str = "d"
    ) -> tuple[Dim, Mapping[str, int]]:
        """Return a fresh symbol standing for this concrete dimension, and its binding.

        The inverse direction of :meth:`substitute`: ``symbol.substitute(binding)`` returns
        this Dim again. The name is ``stem``, or ``stem`` suffixed with the lowest positive
        integer not in ``avoid``. Raises DimensionDomainError on a symbolic Dim.
        """
        if not self.is_concrete:
            raise DimensionDomainError(
                f"cannot abstract symbolic dimension {self}; abstract() takes a concrete "
                f"value, free symbols: {sorted(self.free_symbols)}"
            )
        _check_symbol_name(stem)
        taken = set(avoid)
        name = stem
        suffix = 0
        while name in taken:
            suffix += 1
            name = f"{stem}{suffix}"
        return Dim.symbol(name), MappingProxyType({name: self.to_int()})

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

    def rewrite(self, mapping: Mapping[DimSymbolKey, Dim]) -> Dim:
        """Return a new Dim with symbols replaced by arbitrary (possibly symbolic) Dims.

        Keys are symbol names or bare-symbol Dims. A value replacing a dimension symbol may
        be any Dim; a value replacing an exponent symbol must be concrete or a bare symbol.
        The result is validated against the dimension grammar.
        """
        resolved: dict[str, Dim] = {}
        for key, value in mapping.items():
            name = self._key_symbol_name(key)
            if not isinstance(value, Dim):
                raise TypeError(
                    f"rewrite value for {name!r} must be a Dim, got {type(value).__name__}"
                )
            resolved[name] = value
        if not resolved:
            return self

        subs_dict: dict[sp.Symbol, sp.Expr] = {}
        for sym in sorted(self._expr.free_symbols, key=lambda s: str(s.name)):
            sym_name = str(sym.name)
            if sym_name not in resolved:
                continue
            value_expr = resolved[sym_name]._expr
            if sym.assumptions0.get("positive"):
                subs_dict[sym] = value_expr
            else:
                subs_dict[sym] = Dim._coerce_exponent(resolved[sym_name])
        # Simultaneous: a sequential subs would feed one value into another, so
        # {d: e, e: 2} would send d*e to 4 rather than to 2*e.
        return Dim._from_expr(self._expr.subs(subs_dict, simultaneous=True))

    def unify(self, other: Dim) -> UnifyResult:
        """Decide the equation ``self == other`` over the positive integers.

        Cancels the common factors of both sides base by base (a base with a symbolic or
        undetermined residual exponent is left uncancelled), then decides the reduced
        equation: both sides concrete, one side ``1``, one side a lone symbol, or a lone
        symbol at a concrete power against a concrete integer. Anything else is DEFERRED
        with the reduced pair as its single residual constraint. SUCCESS is reported only
        for a verified equation, FAILURE only for one with no solution.
        """
        if not isinstance(other, Dim):
            raise TypeError(f"unify() requires a Dim, got {type(other).__name__}")
        if self.is_concrete and other.is_concrete:
            status = UnifyStatus.SUCCESS if self.to_int() == other.to_int() else UnifyStatus.FAILURE
            return UnifyResult(status=status)
        if self._expr == other._expr:
            return UnifyResult(status=UnifyStatus.SUCCESS)

        left, right = _cancel_common_factors(self._expr, other._expr)
        return _decide_reduced(Dim._from_expr(left), Dim._from_expr(right))


class UnifyStatus(enum.Enum):
    """The three outcomes a dimension equation can be decided into."""

    SUCCESS = "success"
    FAILURE = "failure"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class UnifyResult:
    """The result of Dim.unify: a status plus any bindings or residual constraints found.

    ``bindings`` maps a symbol name to the Dim it was unified with.
    ``constraints`` lists the reduced pairs of Dims asserted equal that a single
    :meth:`Dim.unify` call could not decide; :func:`solve` carries them further.
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


def _cancel_common_factors(left: sp.Expr, right: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Split ``left == right`` into an equivalent pair sharing no base.

    Subtracts exponents base by base. A base whose residual exponent is not a concrete
    integer stays on the side or sides it started on.
    """
    left_powers = left.as_powers_dict()
    right_powers = right.as_powers_dict()
    reduced_left: sp.Expr = sp.Integer(1)
    reduced_right: sp.Expr = sp.Integer(1)
    for base in sorted(set(left_powers) | set(right_powers), key=sp.srepr):
        left_exponent = sp.sympify(left_powers.get(base, 0))
        right_exponent = sp.sympify(right_powers.get(base, 0))
        net = sp.expand(left_exponent - right_exponent)
        if net.is_Integer:
            if net > 0:
                reduced_left *= base**net
            elif net < 0:
                reduced_right *= base ** (-net)
            continue
        if left_exponent != 0:
            reduced_left *= base**left_exponent
        if right_exponent != 0:
            reduced_right *= base**right_exponent
    return reduced_left, reduced_right


def _lone_symbol_power(dim: Dim) -> tuple[str, sp.Expr] | None:
    """The (symbol name, exponent) of a Dim that is one symbol at unit coefficient."""
    powers = dim.to_sympy().as_powers_dict()
    if len(powers) != 1:
        return None
    ((base, exponent),) = powers.items()
    if not base.is_Symbol:
        return None
    return str(base.name), sp.sympify(exponent)


def _decide_against_one(other: Dim) -> UnifyResult:
    """Decide ``other == 1``: every factor of ``other`` must itself be 1."""
    bindings: dict[str, Dim] = {}
    for base, exponent in sorted(
        other.to_sympy().as_powers_dict().items(), key=lambda kv: sp.srepr(kv[0])
    ):
        exponent_expr = sp.sympify(exponent)
        if not (exponent_expr.is_Integer and exponent_expr > 0):
            return UnifyResult(status=UnifyStatus.DEFERRED, constraints=((other, Dim.concrete(1)),))
        if base.is_Integer:
            if int(base) != 1:
                return UnifyResult(status=UnifyStatus.FAILURE)
            continue
        if not base.is_Symbol:
            return UnifyResult(status=UnifyStatus.DEFERRED, constraints=((other, Dim.concrete(1)),))
        bindings[str(base.name)] = Dim.concrete(1)
    return UnifyResult(status=UnifyStatus.SUCCESS, bindings=bindings)


def _divide_out_coefficient(left: Dim, right: Dim) -> UnifyResult | None:
    """Decide ``k * rest == c`` by divisibility, or return None when that shape is absent.

    Every symbolic factor is a positive integer, so ``k * rest`` is a multiple of ``k``.
    A ``c`` that ``k`` does not divide is FAILURE; otherwise the equation becomes
    ``rest == c // k``.
    """
    for side, counterpart in ((left, right), (right, left)):
        if side.is_concrete or not counterpart.is_concrete:
            continue
        coefficient, rest = side.to_sympy().as_coeff_Mul()
        if not (coefficient.is_Integer and int(coefficient) > 1):
            continue
        factor = int(coefficient)
        value = counterpart.to_int()
        if value % factor != 0:
            return UnifyResult(status=UnifyStatus.FAILURE)
        return _decide_reduced(Dim._from_expr(rest), Dim.concrete(value // factor))
    return None


def _decide_reduced(left: Dim, right: Dim) -> UnifyResult:
    """Decide a reduced equation whose two sides share no base."""
    if left.is_concrete and right.is_concrete:
        status = UnifyStatus.SUCCESS if left == right else UnifyStatus.FAILURE
        return UnifyResult(status=status)
    if left == right:
        return UnifyResult(status=UnifyStatus.SUCCESS)

    one = Dim.concrete(1)
    if left == one:
        return _decide_against_one(right)
    if right == one:
        return _decide_against_one(left)

    divided = _divide_out_coefficient(left, right)
    if divided is not None:
        return divided

    for side, counterpart in ((left, right), (right, left)):
        lone = _lone_symbol_power(side)
        if lone is None:
            continue
        name, exponent = lone
        if name in counterpart.free_symbols:
            continue
        if exponent == 1:
            return UnifyResult(status=UnifyStatus.SUCCESS, bindings={name: counterpart})
        if exponent.is_Integer and exponent >= 2 and counterpart.is_concrete:
            root, exact = sp.integer_nthroot(counterpart.to_int(), int(exponent))
            if not exact:
                return UnifyResult(status=UnifyStatus.FAILURE)
            return UnifyResult(status=UnifyStatus.SUCCESS, bindings={name: Dim.concrete(int(root))})
    return UnifyResult(status=UnifyStatus.DEFERRED, constraints=((left, right),))


_MAX_SOLVE_PASSES = 32
"""Iteration budget for :func:`solve`'s fixpoint. Module-level so a test can patch it low."""

_MAX_BINDING_MERGES = 64
"""Per-pair cap on the binding merges one :func:`solve` pass will chase."""


@dataclass(frozen=True)
class SolveResult:
    """The result of :func:`solve`: a status, resolved bindings, and residual pairs.

    ``bindings`` holds concrete and symbolic bindings, resolved against each other.
    ``residual_pairs`` holds every pair the fixpoint left undecided. ``exhausted`` is True
    only when :data:`_MAX_SOLVE_PASSES` ran out before the fixpoint converged, in which case
    ``residual_pairs`` is a snapshot of the final, non-converged pass.
    """

    status: UnifyStatus
    bindings: Mapping[str, Dim] = field(default_factory=dict)
    residual_pairs: tuple[tuple[Dim, Dim], ...] = ()
    exhausted: bool = False

    @property
    def is_success(self) -> bool:
        return self.status is UnifyStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.status is UnifyStatus.FAILURE

    @property
    def is_deferred(self) -> bool:
        return self.status is UnifyStatus.DEFERRED


def _pair_sort_key(pair: tuple[Dim, Dim]) -> tuple[str, str]:
    return (sp.srepr(pair[0].to_sympy()), sp.srepr(pair[1].to_sympy()))


def _dedupe_pairs(pairs: Sequence[tuple[Dim, Dim]]) -> tuple[tuple[Dim, Dim], ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[Dim, Dim]] = []
    for pair in pairs:
        key = _pair_sort_key(pair)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return tuple(unique)


def solve(constraints: Sequence[tuple[Dim, Dim]], *, max_passes: int | None = None) -> SolveResult:
    """Resolve a system of asserted dimension equalities to a monotone fixpoint.

    Sorts ``constraints`` first, so the result depends only on the set of constraints. Each
    pass resolves both members of every pair through the running bindings and calls
    :meth:`Dim.unify`. A symbol rebound to a Dim that does not unify with its current value
    is FAILURE; a binding whose value contains the symbol being bound is deferred as a
    residual pair. Converged with residuals is DEFERRED with ``exhausted=False``; a budget
    of :data:`_MAX_SOLVE_PASSES` passes (or ``max_passes``) running out is DEFERRED with
    ``exhausted=True``.
    """
    pairs = tuple(sorted(constraints, key=_pair_sort_key))
    if not pairs:
        return SolveResult(status=UnifyStatus.SUCCESS)
    budget = _MAX_SOLVE_PASSES if max_passes is None else max_passes

    bindings: dict[str, Dim] = {}
    residual: list[tuple[Dim, Dim]] = []
    for _pass_index in range(budget):
        changed = False
        residual = []
        for left, right in pairs:
            mapping = cast(Mapping[DimSymbolKey, Dim], bindings)
            resolved_left = left.rewrite(mapping)
            resolved_right = right.rewrite(mapping)
            result = resolved_left.unify(resolved_right)
            if result.is_failure:
                return SolveResult(status=UnifyStatus.FAILURE, bindings=dict(bindings))
            if result.is_deferred:
                residual.extend(result.constraints or ((resolved_left, resolved_right),))
                continue
            queue = sorted(result.bindings.items())
            merges = 0
            while queue:
                merges += 1
                if merges > _MAX_BINDING_MERGES:
                    residual.extend((Dim.symbol(name), value) for name, value in queue)
                    break
                name, value = queue.pop(0)
                value = value.rewrite(cast(Mapping[DimSymbolKey, Dim], bindings))
                existing = bindings.get(name)
                if existing is not None:
                    if existing == value:
                        continue
                    merged = existing.unify(value)
                    if merged.is_failure:
                        return SolveResult(status=UnifyStatus.FAILURE, bindings=dict(bindings))
                    if merged.is_deferred:
                        residual.append((existing, value))
                        continue
                    queue.extend(sorted(merged.bindings.items()))
                    continue
                if name in value.free_symbols:
                    residual.append((Dim.symbol(name), value))
                    continue
                shift = cast(Mapping[DimSymbolKey, Dim], {name: value})
                bindings = {key: held.rewrite(shift) for key, held in bindings.items()}
                bindings[name] = value
                changed = True
        if not changed:
            status = UnifyStatus.DEFERRED if residual else UnifyStatus.SUCCESS
            return SolveResult(
                status=status,
                bindings=dict(bindings),
                residual_pairs=_dedupe_pairs(residual),
            )

    # A final pass that added a binding and left nothing unresolved has in fact solved the
    # system; only an outstanding residual makes the budget's end an undecided answer.
    if not residual:
        return SolveResult(status=UnifyStatus.SUCCESS, bindings=dict(bindings))
    return SolveResult(
        status=UnifyStatus.DEFERRED,
        bindings=dict(bindings),
        residual_pairs=_dedupe_pairs(residual),
        exhausted=True,
    )


_MAX_UNIFY_ALL_PASSES = 32
"""Iteration budget for :func:`unify_all`'s bindings fixpoint. Module-level so a test can
patch it low, mirroring :mod:`archytaszx.rewrite.match`'s ``_MAX_FIXPOINT_PASSES``."""


@dataclass(frozen=True)
class UnifyAllResult:
    """The result of :func:`unify_all`: a status, accumulated bindings, and residual pairs.

    ``residual_pairs`` holds every pair left unresolved once the loop stopped -- either
    with the fixpoint genuinely stabilised on real residual constraints (the ordinary
    ``DEFERRED`` case), or with :data:`_MAX_UNIFY_ALL_PASSES` exhausted
    before it could stabilise, in which case ``residual_pairs`` holds whatever was still
    unresolved on that final, non-converged pass -- a snapshot of an interrupted
    computation, not a decided answer.

    ``exhausted`` is the discriminator: ``False`` for an ordinary converged result, ``True``
    only when the pass budget ran out first. Both report ``status=DEFERRED``, and every
    caller must keep them apart -- :mod:`archytaszx.diagram.validate`'s
    ``_check_generator_policy`` fails closed on ``exhausted`` with a hard error rather than
    folding it into the ordinary deferred-constraint bookkeeping. Every bounded fixpoint
    here keeps that distinction visible at the call site, not only internally.
    """

    status: UnifyStatus
    bindings: Mapping[str, Dim] = field(default_factory=dict)
    residual_pairs: tuple[tuple[Dim, Dim], ...] = ()
    exhausted: bool = False
    declined_bindings: Mapping[str, Dim] = field(default_factory=dict)
    """Every solved binding to a non-concrete ``Dim`` (e.g. ``d := e``), split out of
    ``bindings`` so that ``bindings`` stays usable by :meth:`Dim.substitute`, which takes
    concrete values only.

    :mod:`archytaszx.diagram.validate`'s ``_check_generator_policy`` reports it, together with
    ``bindings``, as a deferred
    :class:`~archytaszx.diagram.validate.IssueKind.DIMENSION_BOUND` issue.
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

    Sorts ``dims`` by a canonical key, chains them into consecutive pairs, and hands the
    system to :func:`solve` with a budget of :data:`_MAX_UNIFY_ALL_PASSES` passes. The final
    binding map is split: concrete values into ``bindings``, non-concrete ones into
    :attr:`UnifyAllResult.declined_bindings`. ``exhausted`` propagates from :func:`solve`.
    Raises only :class:`DimensionError` subclasses.
    """
    ordered = sorted(dims, key=lambda d: sp.srepr(d.to_sympy()))
    if len(ordered) < 2:
        return UnifyAllResult(status=UnifyStatus.SUCCESS)

    result = solve(
        tuple(itertools.pairwise(ordered)),
        max_passes=_MAX_UNIFY_ALL_PASSES,
    )
    return UnifyAllResult(
        status=result.status,
        bindings={name: value for name, value in result.bindings.items() if value.is_concrete},
        residual_pairs=result.residual_pairs,
        exhausted=result.exhausted,
        declined_bindings={
            name: value for name, value in result.bindings.items() if not value.is_concrete
        },
    )
