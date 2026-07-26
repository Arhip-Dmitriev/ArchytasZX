"""Exact scalar representation: roots of unity, free symbolic scalars, and their products and sums.

A :class:`Scalar` is an exact complex number built from exact (Gaussian) rationals, roots
of unity ``omega_d^j = e^{2*pi*i*j/d}`` where ``d`` is a (possibly symbolic)
:class:`~qufzx.algebra.dimension.Dim` and ``j`` may itself be symbolic, dimension factors
(contraction produces bare factors of ``d``, so :meth:`Scalar.from_dim` exists for that),
and free symbolic scalars. Scalars are closed under product, sum, integer powers, and
complex conjugation.

Like :class:`~qufzx.algebra.dimension.Dim`, a Scalar is backed by a single canonical
sympy expression (see :meth:`Scalar.to_sympy`), built with ``sympy.exp`` and
``sympy.I`` so that root-of-unity factors combine algebraically under sympy's own
exact arithmetic. Normalization performed here is limited to cheap, always-sound
steps: collecting like terms, folding concrete rational arithmetic, and reducing a
root-of-unity index modulo ``d`` only when ``d`` is concrete. Nothing here attempts to
decide whether a symbolic ``d`` divides some symbolic index -- that requires exactly
the character-sum reasoning (Sum_{k=0}^{d-1} omega_d^{jk} = d * [j == 0 mod d]) that is
explicitly deferred to Phase 9's simplifier. As with :class:`~qufzx.algebra.phase.Phase`,
equality here is sound but incomplete: two Scalars that compare equal are exactly equal,
but two exactly-equal Scalars may compare unequal if this module's cheap normalization
cannot see it.

CLAUDE.md invariant, restated precisely for this module: scalars are tracked exactly and
nothing here ever quotients a global factor, normalizes to a leading coefficient, or
drops a unit-modulus factor. There is no such method, public or private, anywhere in
this module -- not even as an option. ``s`` and ``2*s`` are unequal; ``s`` and
``omega_d * s`` are unequal; there is no API that makes them equal. If a later phase
wants an up-to-global-phase comparison, that flag belongs at the semantics/check.py
layer (Phase 4), never here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase


class ScalarError(Exception):
    """Base class for all errors raised by this module."""


class ScalarDomainError(ScalarError):
    """A value is outside the mathematical domain required by an operation.

    Raised for numeric evaluation attempts on a scalar that still carries free symbols.
    """


class ScalarGrammarError(ScalarError):
    """A value falls outside the grammar this module accepts.

    Raised for non-exact numeric input (e.g. floats where an exact rational is
    required) and other malformed constructor arguments.
    """


ScalarSymbolKey = Union[str, "Scalar"]
ScalarSubstituteValue = Union[int, "sp.Rational", "Scalar"]


def _scalar_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, complex=True)


def _normalize(expr: sp.Expr) -> sp.Expr:
    """Cheap, always-sound normalization: expand/collect and fold same-base exponentials.

    Two steps, in this order: ``sympy.expand`` distributes products over sums (folding
    concrete rational and Gaussian-rational arithmetic along the way), then
    ``sympy.powsimp(..., force=True)`` recombines products of same-base exponentials
    (e.g. ``exp(a) * exp(b) -> exp(a + b)``) into one canonical exponential term. Both
    steps are exact identities for complex exponentials (``e^a * e^b = e^{a+b}`` holds
    unconditionally, with no branch-cut subtlety, since ``exp`` is entire) -- this is
    not the kind of number-theoretic reasoning about a symbolic ``d`` that Phase 9's
    character-sum simplifier is reserved for.
    """
    return sp.powsimp(sp.expand(expr), force=True)


class Scalar:
    """An immutable, hashable exact complex scalar, backed by a canonical sympy expression.

    See the module docstring for the normalization and equality contract, and for the
    standing prohibition on any global-factor quotienting anywhere in this class.
    """

    __slots__ = ("_expr",)
    _expr: sp.Expr

    def __init__(self, expr: sp.Expr) -> None:
        """Build a Scalar directly from a sympy expression. Prefer the named constructors."""
        self._expr = _normalize(sp.sympify(expr))

    @classmethod
    def zero(cls) -> Scalar:
        """The exact scalar 0."""
        return cls(sp.Integer(0))

    @classmethod
    def one(cls) -> Scalar:
        """The exact scalar 1."""
        return cls(sp.Integer(1))

    @classmethod
    def rational(cls, p: int, q: int = 1) -> Scalar:
        """Build an exact rational scalar p/q. Raises ScalarDomainError if q == 0."""
        if isinstance(p, bool) or isinstance(q, bool):
            raise ScalarGrammarError("rational() does not accept bool")
        if not isinstance(p, int) or not isinstance(q, int):
            raise ScalarGrammarError("rational() requires int numerator and denominator")
        if q == 0:
            raise ScalarDomainError("rational() denominator must be nonzero")
        return cls(sp.Rational(p, q))

    @classmethod
    def gaussian_rational(cls, real: sp.Rational | int, imag: sp.Rational | int) -> Scalar:
        """Build an exact Gaussian rational real + imag*i, both parts exact."""
        return cls(sp.sympify(real) + sp.sympify(imag) * sp.I)

    @classmethod
    def omega(cls, dim: Dim, index: int | sp.Expr = 1) -> Scalar:
        """Build omega_d^index = e^{2*pi*i*index/d}, with dim possibly symbolic.

        Reaches the sympy layer only through ``dim.to_sympy()``, per the per-port
        dimension invariant: this module holds no ambient "current d".
        """
        if isinstance(index, bool):
            raise ScalarGrammarError(f"omega() index does not accept bool, got {index!r}")
        if isinstance(index, int):
            index_expr: sp.Expr = sp.Integer(index)
        elif isinstance(index, sp.Expr):
            index_expr = index
        else:
            raise ScalarGrammarError(
                f"omega() index must be int or sympy expression, got {type(index).__name__}"
            )
        d_expr = dim.to_sympy()
        if not d_expr.free_symbols and index_expr.is_Integer:
            index_expr = index_expr % d_expr
        return cls(sp.exp(2 * sp.pi * sp.I * index_expr / d_expr))

    @classmethod
    def from_dim(cls, dim: Dim) -> Scalar:
        """Build the scalar factor equal to the bare dimension value d (possibly symbolic)."""
        return cls(dim.to_sympy())

    @classmethod
    def symbol(cls, name: str) -> Scalar:
        """Build a free symbolic scalar parameter."""
        return cls(_scalar_symbol(name))

    @classmethod
    def from_phase(cls, phase: Phase) -> Scalar:
        """The unit scalar e^{i*angle} corresponding to a :class:`~qufzx.algebra.phase.Phase`.

        This is the bridge Phase 5 needs when spider fusion introduces a phase-derived
        scalar factor.
        """
        return cls(sp.exp(sp.I * phase.to_radians()))

    def to_sympy(self) -> sp.Expr:
        """Escape hatch returning the underlying canonical sympy expression."""
        return self._expr

    @property
    def is_concrete(self) -> bool:
        """True iff this expression has no free symbols."""
        return not self._expr.free_symbols

    @property
    def free_symbols(self) -> frozenset[str]:
        """The names of all free symbols appearing in this expression."""
        return frozenset(str(s.name) for s in self._expr.free_symbols)

    @property
    def is_zero(self) -> bool:
        """True iff this scalar is provably zero. Conservative: False if unverified."""
        return bool(self._expr == 0)

    @property
    def is_one(self) -> bool:
        """True iff this scalar is provably one. Conservative: False if unverified."""
        return bool(self._expr == 1)

    def __add__(self, other: Scalar) -> Scalar:
        """Exact sum of two scalars."""
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self._expr + other._expr)

    def __sub__(self, other: Scalar) -> Scalar:
        """Exact difference of two scalars."""
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self._expr - other._expr)

    def __neg__(self) -> Scalar:
        """Negate this scalar."""
        return Scalar(-self._expr)

    def __mul__(self, other: Scalar) -> Scalar:
        """Exact product of two scalars."""
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self._expr * other._expr)

    def __rmul__(self, other: Scalar) -> Scalar:
        return self.__mul__(other)

    def __pow__(self, exponent: int) -> Scalar:
        """Raise this scalar to an exact integer power (positive, negative, or zero)."""
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise ScalarGrammarError(f"__pow__ requires an int exponent, got {exponent!r}")
        if exponent < 0 and self.is_zero:
            raise ScalarDomainError("cannot raise the zero scalar to a negative power")
        return Scalar(self._expr**exponent)

    def conjugate(self) -> Scalar:
        """Exact complex conjugate (e.g. omega_d^j -> omega_d^{-j})."""
        return Scalar(sp.conjugate(self._expr))

    def substitute(self, mapping: Mapping[ScalarSymbolKey, ScalarSubstituteValue]) -> Scalar:
        """Return a new Scalar with symbols replaced by concrete values.

        Keys may be symbol names (str) or bare-symbol Scalars; values may be an int, a
        sympy Rational, or a concrete Scalar. Both dimension symbols (appearing inside
        an omega or from_dim factor) and free scalar symbols live in the same underlying
        sympy expression and are substituted through the same mapping. The mapping need
        not be total, and this Scalar is never mutated.
        """
        subs_dict: dict[sp.Symbol, sp.Expr] = {}
        by_name = {str(s.name): s for s in self._expr.free_symbols}
        for key, value in mapping.items():
            name = key if isinstance(key, str) else self._scalar_symbol_name(key)
            if name not in by_name:
                continue
            if isinstance(value, bool):
                raise ScalarGrammarError(f"substitution value for {name!r} does not accept bool")
            if isinstance(value, int):
                subs_dict[by_name[name]] = sp.Integer(value)
            elif isinstance(value, sp.Rational):
                subs_dict[by_name[name]] = value
            elif isinstance(value, Scalar):
                subs_dict[by_name[name]] = value._expr
            else:
                raise ScalarGrammarError(
                    f"substitution value for {name!r} must be int, sympy Rational, or Scalar, "
                    f"got {type(value).__name__}"
                )
        return Scalar(self._expr.subs(subs_dict))

    @staticmethod
    def _scalar_symbol_name(key: Scalar) -> str:
        if not key._expr.is_Symbol:
            raise ScalarGrammarError(f"substitution key must be a bare symbol, got {key}")
        return str(key._expr.name)

    def to_complex(self) -> complex:
        """The single gated numeric evaluator: this scalar as a Python complex.

        Raises ScalarDomainError if any free symbol remains. This is the only
        sanctioned path from a Scalar to a concrete number.
        """
        if not self.is_concrete:
            raise ScalarDomainError(
                f"cannot convert symbolic scalar {self} to a numeric value; "
                f"free symbols: {sorted(self.free_symbols)}"
            )
        return complex(self._expr)

    def __eq__(self, other: object) -> bool:
        """Exact equality after normalization; see the module docstring for soundness."""
        if not isinstance(other, Scalar):
            return NotImplemented
        return bool(self._expr == other._expr)

    def __hash__(self) -> int:
        """Hash agrees with __eq__: equal Scalars (post-normalization) hash equal."""
        return hash(self._expr)

    def __repr__(self) -> str:
        return f"Scalar({self._expr})"

    def __str__(self) -> str:
        return str(self._expr)
