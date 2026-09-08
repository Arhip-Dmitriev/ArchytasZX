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
root-of-unity index modulo ``d`` only when ``d`` is concrete. Deciding whether a symbolic
``d`` divides a symbolic index is :meth:`Scalar.simplify`'s job, not this normalization's:
it closes an index sum through the character-sum identity
Sum_{k=0}^{d-1} omega_d^{jk} = d * [j == 0 mod d]. As with :class:`~qufzx.algebra.phase.Phase`,
equality here is sound but incomplete: two Scalars that compare equal are exactly equal,
but two exactly-equal Scalars may compare unequal if this module's cheap normalization
cannot see it.

The spec invariant, restated precisely for this module: scalars are tracked exactly and
nothing here ever quotients a global factor, normalizes to a leading coefficient, or
drops a unit-modulus factor. There is no such method, public or private, anywhere in
this module -- not even as an option. ``s`` and ``2*s`` are unequal; ``s`` and
``omega_d * s`` are unequal; there is no API that makes them equal. If a later phase
wants an up-to-global-phase comparison, that flag belongs at the semantics/check.py
layer, where :class:`~qufzx.semantics.check.EqualityMode` now carries it, never here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Union, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim, DimensionError
from qufzx.algebra.phase import Phase


class ScalarError(Exception):
    """Base class for all errors raised by this module."""


class ScalarDomainError(ScalarError):
    """A value is outside the mathematical domain required by an operation.

    Raised for numeric evaluation of a scalar that still carries free symbols, for a zero
    denominator in :meth:`Scalar.rational`, and for raising the zero scalar to a negative
    power.
    """


class ScalarGrammarError(ScalarError):
    """A value falls outside the grammar this module accepts.

    Raised for non-exact numeric input (e.g. floats where an exact rational is
    required) and other malformed constructor arguments.
    """


class ScalarSumError(ScalarError):
    """An index sum is malformed, or a bound index is used where a free symbol is required."""


class ScalarBudgetError(ScalarError):
    """Simplification exceeded its step budget."""


DEFAULT_MAX_SIMPLIFY_STEPS = 1024

_RESERVED_INDEX = re.compile(r"^_[ik][0-9]+$")

ScalarSymbolKey = Union[str, "Scalar"]
ScalarSubstituteValue = Union[int, "sp.Rational", "Scalar"]


def _check_symbol_name(name: str, *, allow_reserved: bool = False) -> None:
    """Reject a symbol name that is not a bare identifier, or that is a reserved index name."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ScalarGrammarError(
            f"symbol name must be a bare identifier, got {name!r}; a name carrying "
            "operator characters builds a symbol that renders identically to an "
            "expression but is a different Scalar"
        )
    if not allow_reserved and _RESERVED_INDEX.match(name):
        raise ScalarGrammarError(f"symbol name {name!r} is reserved for engine-generated indices")


def _scalar_symbol(name: str) -> sp.Symbol:
    _check_symbol_name(name)
    return sp.Symbol(name, complex=True)


def _index_symbol(name: str) -> sp.Symbol:
    """An engine-generated index symbol: a nonnegative integer symbol."""
    _check_symbol_name(name, allow_reserved=True)
    return sp.Symbol(name, integer=True, nonnegative=True)


def _check_sum_structure(expr: sp.Sum) -> None:
    """Enforce the one-limit, 0-to-d-1, integer-index shape an admitted Sum must carry."""
    if not expr.limits:
        raise ScalarSumError(f"index sum {expr} carries no limit")
    for var, lower, upper in expr.limits:
        if not var.is_Symbol:
            raise ScalarSumError(f"index sum variable {var} must be a nonnegative integer symbol")
        assumptions = var.assumptions0
        if not (assumptions.get("integer") and assumptions.get("nonnegative")):
            raise ScalarSumError(f"index sum variable {var} must be a nonnegative integer symbol")
        _check_symbol_name(str(var.name), allow_reserved=True)
        if lower != sp.Integer(0):
            raise ScalarSumError(
                f"index sum upper bound {upper} is not of the form d - 1 for a dimension d"
            )
        try:
            Dim.from_sympy(sp.expand(upper + 1))
        except DimensionError as exc:
            raise ScalarSumError(
                f"index sum upper bound {upper} is not of the form d - 1 for a dimension d"
            ) from exc


def _check_scalar_domain(expr: sp.Expr) -> None:
    """Reject an expression outside the scalar grammar.

    The grammar is: exact numbers, ``I``, sympy's named constants, symbols, and sums,
    products, powers and ``exp``/``conjugate`` applications of these. A transcendental
    function (``log``, ``sin``) or any other head is rejected.
    """
    if expr.is_Number or expr is sp.I or expr.is_NumberSymbol:
        return
    if expr.is_Symbol:
        _check_symbol_name(str(expr.name), allow_reserved=True)
        return
    if isinstance(expr, sp.Sum):
        _check_sum_structure(expr)
        _check_scalar_domain(expr.function)
        return
    if expr.is_Add or expr.is_Mul or expr.is_Pow or isinstance(expr, (sp.exp, sp.conjugate)):
        for arg in expr.args:
            _check_scalar_domain(arg)
        return
    raise ScalarGrammarError(
        f"expression {expr} is outside the scalar grammar (head {type(expr).__name__})"
    )


def _normalize(expr: sp.Expr) -> sp.Expr:
    """Cheap, always-sound normalization: expand/collect and fold same-base exponentials.

    Two steps, in this order: ``sympy.expand`` distributes products over sums (folding
    concrete rational and Gaussian-rational arithmetic along the way), then
    ``sympy.powsimp(..., force=True)`` recombines products of same-base exponentials
    (e.g. ``exp(a) * exp(b) -> exp(a + b)``) into one canonical exponential term. Both
    steps are exact identities for complex exponentials (``e^a * e^b = e^{a+b}`` holds
    unconditionally, with no branch-cut subtlety, since ``exp`` is entire) -- this is
    not the kind of number-theoretic reasoning about a symbolic ``d`` that
    :meth:`Scalar.simplify` performs.

    Before any of that: an expression containing an ``sp.Float`` atom anywhere in its
    tree (not just at the top level) is rejected outright, since this module tracks
    scalars exactly, never approximately.
    """
    if expr.atoms(sp.Float):
        raise ScalarGrammarError(f"Scalar requires an exact expression, got a float in {expr!r}")
    normalized = sp.powsimp(sp.expand(expr), force=True)
    _check_scalar_domain(normalized)
    return cast(sp.Expr, normalized)


def _is_zero_mod(j_expr: sp.Expr, d_expr: sp.Expr) -> str:
    """Decide whether j == 0 (mod d), returning "YES", "NO", or "UNDECIDABLE".

    Only a fully concrete pair can answer "NO": a free dimension symbol admits d = 1, where
    every integer is 0 mod d, so no universal negative is available.
    """
    if j_expr.is_Integer and d_expr.is_Integer:
        return "YES" if int(j_expr) % int(d_expr) == 0 else "NO"
    if j_expr == sp.Integer(0):
        return "YES"
    ratio = sp.cancel(j_expr / d_expr)
    if sp.expand(d_expr * ratio - j_expr) == 0 and ratio.is_integer is True:
        return "YES"
    return "UNDECIDABLE"


def _rename_indices(expr: sp.Expr) -> sp.Expr:
    """Alpha-rename every bound summation index to _k0, _k1, ... in pre-order."""

    counter = [0]

    def stage_a(node: sp.Expr) -> sp.Expr:
        if isinstance(node, sp.Sum):
            body = node.function
            limits = []
            for var, lower, upper in node.limits:
                temp = sp.Symbol(f"_tmp{counter[0]}", integer=True, nonnegative=True)
                counter[0] += 1
                body = body.xreplace({var: temp})
                limits.append((temp, lower, upper))
            return sp.Sum(stage_a(body), *limits)
        if not node.args:
            return node
        new_args = list(node.args)
        for position in sorted(range(len(node.args)), key=lambda i: sp.srepr(node.args[i])):
            new_args[position] = stage_a(node.args[position])
        return cast(sp.Expr, node.func(*new_args))

    staged = stage_a(expr)
    mapping = {
        symbol: sp.Symbol(f"_k{symbol.name[4:]}", integer=True, nonnegative=True)
        for symbol in staged.atoms(sp.Symbol)
        if str(symbol.name).startswith("_tmp")
    }
    return cast(sp.Expr, staged.xreplace(mapping) if mapping else staged)


def _split_term(
    term: sp.Expr, var: sp.Symbol, d_expr: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr] | None:
    """Split a summand into (index-free factor, index j, offset c) with the var-dependence
    read as omega_d^{j*var + c}, or None when the dependence is not of that shape.

    The exponential factors are combined by adding their arguments rather than by
    ``powsimp``, which ``expand`` would undo.
    """
    constant = sp.Integer(1)
    exponent: sp.Expr = sp.Integer(0)
    for factor in sp.Mul.make_args(term):
        if var not in factor.free_symbols:
            constant *= factor
            continue
        if not isinstance(factor, sp.exp):
            return None
        exponent = exponent + factor.args[0]
    ratio = sp.expand(sp.cancel(exponent * d_expr / (2 * sp.pi * sp.I)))
    try:
        polynomial = sp.Poly(ratio, var)
    except sp.PolynomialError:
        return None
    if polynomial.degree() > 1:
        return None
    index = sp.expand(polynomial.coeff_monomial(var))
    if var in index.free_symbols:
        return None
    return constant, index, sp.expand(polynomial.coeff_monomial(1))


def _close_term(term: sp.Expr, var: sp.Symbol, d_expr: sp.Expr) -> sp.Expr | None:
    """Close one product term of a summand against the character sum, or return None."""
    split = _split_term(term, var, d_expr)
    if split is None:
        return None
    constant, index, offset = split
    if index == sp.Integer(0) and offset == sp.Integer(0):
        return cast(sp.Expr, constant * d_expr)
    verdict = _is_zero_mod(index, d_expr)
    if verdict == "NO":
        return sp.Integer(0)
    if verdict == "UNDECIDABLE":
        return None
    shift = sp.exp(2 * sp.pi * sp.I * offset / d_expr)
    return cast(sp.Expr, constant * shift * d_expr)


def _close_sum(node: sp.Sum) -> sp.Expr | None:
    """Evaluate one Sum node through the character-sum identity, or return None to leave it."""
    outer = list(node.limits[:-1])
    var, _, upper = node.limits[-1]
    d_expr = sp.expand(upper + 1)

    def rewrap(expr: sp.Expr) -> sp.Expr:
        return cast(sp.Expr, sp.Sum(expr, *outer) if outer else expr)

    body = sp.expand(sp.powsimp(sp.expand(node.function), force=True))
    terms = sp.Add.make_args(body)
    closed: list[sp.Expr] = []
    changed = False
    for term in terms:
        result = _close_term(term, var, d_expr)
        if result is None:
            closed.append(sp.Sum(term, (var, 0, upper)))
        else:
            closed.append(result)
            changed = True
    if changed:
        return rewrap(sp.Add(*closed))
    if len(terms) == 1 and outer:
        eliminated = _eliminate_via_delta(terms[0], var, d_expr, outer)
        if eliminated is not None:
            return eliminated
    if d_expr.is_Integer:
        return rewrap(
            sp.Add(*(node.function.xreplace({var: sp.Integer(v)}) for v in range(int(d_expr))))
        )
    return None


def _is_periodic_in(expr: sp.Expr, var: sp.Symbol, d_expr: sp.Expr) -> bool:
    """Whether every var-dependent factor of expr is an omega_d power of integer index."""
    for factor in sp.Mul.make_args(expr):
        if var not in factor.free_symbols:
            continue
        if not isinstance(factor, sp.exp):
            return False
        ratio = sp.expand(sp.cancel(factor.args[0] * d_expr / (2 * sp.pi * sp.I)))
        try:
            polynomial = sp.Poly(ratio, var)
        except sp.PolynomialError:
            return False
        if polynomial.degree() > 1:
            return False
        if not polynomial.coeff_monomial(var).is_integer:
            return False
    return True


def _eliminate_via_delta(
    term: sp.Expr,
    var: sp.Symbol,
    d_expr: sp.Expr,
    outer: list[tuple[sp.Symbol, sp.Expr, sp.Expr]],
) -> sp.Expr | None:
    """Close a sum whose index is an enclosing bound variable, collapsing that outer sum.

    Sum_u Sum_k omega_d^{k*(u + rest)} f(u) closes to d * f(-rest), fired only when the
    outer index runs over a full residue system mod d and f is d-periodic in it.
    """
    split = _split_term(term, var, d_expr)
    if split is None:
        return None
    constant, index, offset = split
    constant = constant * sp.exp(2 * sp.pi * sp.I * offset / d_expr)
    for position, (candidate, _lower, upper) in enumerate(outer):
        if candidate not in index.free_symbols:
            continue
        if sp.expand(upper + 1) != d_expr:
            continue
        try:
            polynomial = sp.Poly(index, candidate)
        except sp.PolynomialError:
            continue
        if polynomial.degree() != 1:
            continue
        coefficient = polynomial.coeff_monomial(candidate)
        if coefficient not in (sp.Integer(1), sp.Integer(-1)):
            continue
        rest = sp.expand(index - coefficient * candidate)
        if candidate in rest.free_symbols:
            continue
        if not _is_periodic_in(constant, candidate, d_expr):
            continue
        remaining = [limit for i, limit in enumerate(outer) if i != position]
        collapsed = d_expr * constant.xreplace({candidate: sp.expand(-coefficient * rest)})
        return cast(sp.Expr, sp.Sum(collapsed, *remaining) if remaining else collapsed)
    return None


def _pull_constants(expr: sp.Expr) -> sp.Expr:
    """Move every factor independent of a sum's bound indices outside that sum.

    Canonical placement: an index-free factor always sits outside, so two expressions
    differing only in whether a constant was carried into the summand compare equal.
    """
    if expr.args:
        new_args = list(expr.args)
        for position, arg in enumerate(expr.args):
            if isinstance(expr, sp.Sum) and position != 0:
                continue
            new_args[position] = _pull_constants(cast(sp.Expr, arg))
        expr = cast(sp.Expr, expr.func(*new_args))
    if isinstance(expr, sp.Sum):
        bound = {limit[0] for limit in expr.limits}
        inside: sp.Expr = sp.Integer(1)
        outside: sp.Expr = sp.Integer(1)
        for factor in sp.Mul.make_args(expr.function):
            if factor.free_symbols & bound:
                inside = inside * factor
            else:
                outside = outside * factor
        if outside != sp.Integer(1):
            return cast(sp.Expr, outside * sp.Sum(inside, *expr.limits))
    return expr


def _close_all(expr: sp.Expr, budget: list[int]) -> tuple[sp.Expr, bool]:
    """Close every closable Sum in one bottom-up pass, reporting whether anything changed."""
    changed = False
    if expr.args:
        new_args = list(expr.args)
        for position, arg in enumerate(expr.args):
            if isinstance(expr, sp.Sum) and position != 0:
                continue  # limit tuples carry bound symbols, never sub-expressions to close
            replaced, sub_changed = _close_all(cast(sp.Expr, arg), budget)
            new_args[position] = replaced
            changed = changed or sub_changed
        expr = cast(sp.Expr, expr.func(*new_args))
    if isinstance(expr, sp.Sum):
        budget[0] -= 1
        if budget[0] < 0:
            raise ScalarBudgetError(
                "simplification exceeded its step budget; the expression is left unsimplified"
            )
        result = _close_sum(expr)
        if result is not None:
            return result, True
    return expr, changed


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
        """Build an exact Gaussian rational real + imag*i, both parts exact.

        Both ``real`` and ``imag`` must be an int or sympy Rational; floats are
        rejected: this module tracks scalars exactly, never approximately.
        """
        if isinstance(real, bool) or isinstance(imag, bool):
            raise ScalarGrammarError("gaussian_rational() does not accept bool")
        if not isinstance(real, (int, sp.Rational)) or not isinstance(imag, (int, sp.Rational)):
            raise ScalarGrammarError(
                "gaussian_rational() requires int or sympy Rational for both real and imag parts, "
                f"got {type(real).__name__} and {type(imag).__name__}"
            )
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

        The Phase-to-Scalar bridge, for a rule whose introduced scalar is phase-derived.
        Spider fusion is not one: it introduces :meth:`Scalar.one`.
        """
        return cls(sp.exp(sp.I * phase.to_radians()))

    @classmethod
    def index_sum(cls, dim: Dim, body: Callable[[Scalar], Scalar]) -> Scalar:
        """A sum of body(k) over k from 0 to dim - 1, left unevaluated."""
        placeholder = sp.Symbol("_kbound", integer=True, nonnegative=True)
        body_scalar = body(cls(placeholder))
        if not isinstance(body_scalar, Scalar):
            raise ScalarSumError(
                f"index_sum() body must return a Scalar, got {type(body_scalar).__name__}"
            )
        body_expr = body_scalar.to_sympy()
        used = {
            int(str(symbol.name)[2:])
            for symbol in body_expr.atoms(sp.Symbol)
            if _RESERVED_INDEX.match(str(symbol.name)) and str(symbol.name).startswith("_k")
        }
        index = _index_symbol(f"_k{max(used) + 1 if used else 0}")
        return cls(sp.Sum(body_expr.xreplace({placeholder: index}), (index, 0, dim.to_sympy() - 1)))

    @classmethod
    def dim_power(cls, dim: Dim, numerator: int, denominator: int = 1) -> Scalar:
        """A rational power of a dimension, e.g. d^(-1/2)."""
        if isinstance(numerator, bool) or isinstance(denominator, bool):
            raise ScalarGrammarError("dim_power() does not accept bool")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise ScalarGrammarError("dim_power() requires int numerator and denominator")
        if denominator == 0:
            raise ScalarDomainError("dim_power() denominator must be nonzero")
        return cls(sp.Pow(dim.to_sympy(), sp.Rational(numerator, denominator)))

    def simplify(self, *, max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS) -> Scalar:
        """Close every index sum this scalar carries whose character-sum verdict is decidable."""
        budget = [max_steps]
        expr = _rename_indices(self._expr)
        while True:
            budget[0] -= 1
            if budget[0] < 0:
                raise ScalarBudgetError(
                    "simplification exceeded its step budget; the expression is left unsimplified"
                )
            expr, changed = _close_all(expr, budget)
            expr = _pull_constants(sp.powsimp(sp.expand(expr), force=True))
            if not changed:
                break
        return Scalar(_rename_indices(expr))

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
        bound = {str(node.limits[0][0].name) for node in self._expr.atoms(sp.Sum)}
        for key, value in mapping.items():
            name = key if isinstance(key, str) else self._scalar_symbol_name(key)
            if name in bound:
                raise ScalarSumError(
                    f"cannot substitute for {name!r}: it is a bound summation index"
                )
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
