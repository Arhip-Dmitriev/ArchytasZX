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
:class:`~archytaszx.algebra.dimension.Dim` and ``j`` may itself be symbolic, dimension factors
(contraction produces bare factors of ``d``, so :meth:`Scalar.from_dim` exists for that),
and free symbolic scalars. Scalars are closed under product, sum, integer powers, and
complex conjugation.

Like :class:`~archytaszx.algebra.dimension.Dim`, a Scalar is backed by a single canonical
sympy expression (see :meth:`Scalar.to_sympy`), built with ``sympy.exp`` and
``sympy.I`` so that root-of-unity factors combine algebraically under sympy's own
exact arithmetic. Normalization performed here is limited to cheap, always-sound
steps: collecting like terms, folding concrete rational arithmetic, and reducing a
root-of-unity index modulo ``d`` only when ``d`` is concrete. Deciding whether a symbolic
``d`` divides a symbolic index is :meth:`Scalar.simplify`'s job, not this normalization's:
it closes an index sum through the character-sum identity
Sum_{k=0}^{d-1} omega_d^{jk} = d * [j == 0 mod d], the bracket being the exact atom
:class:`ModDelta`, then removes the bound indices those deltas pin. As with
:class:`~archytaszx.algebra.phase.Phase`,
equality here is sound but incomplete: two Scalars that compare equal are exactly equal,
but two exactly-equal Scalars may compare unequal if this module's cheap normalization
cannot see it.

The spec invariant, restated precisely for this module: scalars are tracked exactly and
nothing here ever quotients a global factor, normalizes to a leading coefficient, or
drops a unit-modulus factor. There is no such method, public or private, anywhere in
this module -- not even as an option. ``s`` and ``2*s`` are unequal; ``s`` and
``omega_d * s`` are unequal; there is no API that makes them equal. If a later phase
wants an up-to-global-phase comparison, that flag belongs at the semantics/check.py
layer, where :class:`~archytaszx.semantics.check.EqualityMode` now carries it, never here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Union, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.algebra.dimension import Dim, DimensionError
from archytaszx.algebra.phase import Phase


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
_CASE_SYMBOL = re.compile(r"^_q[0-9]+$")

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
    if not allow_reserved and (_RESERVED_INDEX.match(name) or _CASE_SYMBOL.match(name)):
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
    if (
        expr.is_Add
        or expr.is_Mul
        or expr.is_Pow
        or isinstance(expr, (sp.exp, sp.conjugate, ModDelta, ModGcd))
    ):
        for arg in expr.args:
            _check_scalar_domain(arg)
        return
    raise ScalarGrammarError(
        f"expression {expr} is outside the scalar grammar (head {type(expr).__name__})"
    )


def _normalize(expr: sp.Expr) -> sp.Expr:
    """Cheap, always-sound normalization: expand/collect, fold same-base exponentials, and
    alpha-rename bound indices.

    Three steps, in this order: ``sympy.expand`` distributes products over sums (folding
    concrete rational and Gaussian-rational arithmetic along the way), then
    ``sympy.powsimp(..., force=True)`` recombines products of same-base exponentials
    (e.g. ``exp(a) * exp(b) -> exp(a + b)``) into one canonical exponential term. Both
    steps are exact identities for complex exponentials (``e^a * e^b = e^{a+b}`` holds
    unconditionally, with no branch-cut subtlety, since ``exp`` is entire) -- this is
    not the kind of number-theoretic reasoning about a symbolic ``d`` that
    :meth:`Scalar.simplify` performs. Finally :func:`_rename_indices` gives every bound
    summation index a name fixed by its binder depth, so two scalars differing only in a
    bound name are the same object here.

    Before any of that: an expression containing an ``sp.Float`` atom anywhere in its
    tree (not just at the top level) is rejected outright, since this module tracks
    scalars exactly, never approximately.
    """
    if expr.atoms(sp.Float):
        raise ScalarGrammarError(f"Scalar requires an exact expression, got a float in {expr!r}")
    normalized = _rename_indices(sp.powsimp(sp.expand(expr), force=True))
    _check_scalar_domain(normalized)
    return cast(sp.Expr, normalized)


_Limit = tuple[sp.Symbol, sp.Expr, sp.Expr]


def _reduce_mod(expr: sp.Expr, modulus: int) -> sp.Expr:
    """``expr`` with every integer coefficient taken into ``(-modulus/2, modulus/2]``."""
    symbols = sorted(expr.free_symbols, key=lambda symbol: str(symbol.name))
    if not symbols:
        if not expr.is_Integer:
            return expr
        residue = int(expr) % modulus
        return sp.Integer(residue - modulus if 2 * residue > modulus else residue)
    try:
        poly = sp.Poly(expr, *symbols)
    except sp.PolynomialError:
        return expr
    if not all(coefficient.is_Integer for coefficient in poly.coeffs()):
        return expr
    total: sp.Expr = sp.Integer(0)
    for monomial, coefficient in poly.terms():
        residue = int(coefficient) % modulus
        if 2 * residue > modulus:
            residue -= modulus
        total += residue * sp.Mul(*(s**e for s, e in zip(symbols, monomial)))
    return cast(sp.Expr, sp.expand(total))


def _canonical_residue(expr: sp.Expr, modulus: sp.Expr) -> sp.Expr:
    """The representative of ``+-expr`` (reduced when ``modulus`` is concrete) a ModDelta keeps."""
    candidates = [sp.expand(expr), sp.expand(-expr)]
    if modulus.is_Integer:
        candidates = [_reduce_mod(candidate, int(modulus)) for candidate in candidates]
    preferred = [c for c in candidates if not c.could_extract_minus_sign()] or candidates
    return cast(sp.Expr, min(preferred, key=sp.default_sort_key))


class ModDelta(sp.Function):  # type: ignore[misc]  # sympy is untyped
    """``[j == 0 mod m]`` for an integer-valued ``j`` and a positive integer modulus ``m``."""

    nargs = 2
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, j: sp.Expr, m: sp.Expr) -> sp.Expr | None:
        """The value when decidable, else ``j`` moved to its canonical residue."""
        args_j = j
        j = sp.expand(j)
        if j == 0 or m == 1:
            return sp.Integer(1)
        if m.is_Integer and j.is_Integer:
            return sp.Integer(1 if int(j) % int(m) == 0 else 0)
        if not m.is_Integer and sp.cancel(j / m).is_integer is True:
            return sp.Integer(1)
        if not m.is_Integer and j.is_Add:
            kept = [t for t in j.args if sp.cancel(t / m).is_integer is not True]
            if len(kept) < len(j.args):
                return cast(sp.Expr, cls(sp.Add(*kept), m))
        if isinstance(m, ModGcd) and m.args[0].is_Integer:
            j = _canonical_residue(_reduce_mod(j, int(m.args[0])), m)
            if j == 0:
                return sp.Integer(1)
            return cast(sp.Expr, cls(j, m, evaluate=False)) if j != args_j else None
        canonical = _canonical_residue(j, m)
        if canonical != j:
            return cast(sp.Expr, cls(canonical, m))
        return None


class ModGcd(sp.Function):  # type: ignore[misc]  # sympy is untyped
    """``gcd(c, m)`` for an integer ``c`` and a positive integer modulus ``m``."""

    nargs = 2
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, c: sp.Expr, m: sp.Expr) -> sp.Expr | None:
        """The value when decidable, else ``c`` made nonnegative."""
        if c.is_Integer and m.is_Integer:
            return sp.Integer(math.gcd(int(c), int(m)))
        if c in (sp.Integer(1), sp.Integer(-1)) or m == 1:
            return sp.Integer(1)
        if c == 0:
            return m
        if c.is_Integer and c < 0:
            return cast(sp.Expr, cls(-c, m))
        if c.is_Integer and isinstance(m, ModGcd) and m.args[0].is_Integer:
            return cast(sp.Expr, cls(math.gcd(int(c), int(m.args[0])), m.args[1]))
        if c.is_Integer and m.is_integer is True:
            residue = _reduce_mod(sp.expand(m), int(c))
            if residue.is_Integer:
                return sp.Integer(math.gcd(int(c), int(residue)))
        return None


def _rename_indices(expr: sp.Expr) -> sp.Expr:
    """Alpha-rename every bound summation index by how many binders enclose it.

    Two alpha-equivalent subexpressions are named identically wherever they sit, so sibling
    sums that differ only in their bound name cancel; a counter running over the whole
    expression would give them different names and leave the difference standing.
    """

    def stage_a(node: sp.Expr, level: int) -> sp.Expr:
        if isinstance(node, sp.Sum):
            body = node.function
            limits = []
            for position, (var, lower, upper) in enumerate(node.limits):
                temp = sp.Symbol(f"_tmp{level + position}", integer=True, nonnegative=True)
                body = body.xreplace({var: temp})
                limits.append((temp, lower, upper))
            return sp.Sum(stage_a(body, level + len(node.limits)), *limits)
        if not node.args:
            return node
        new_args = list(node.args)
        for position in sorted(range(len(node.args)), key=lambda i: sp.srepr(node.args[i])):
            new_args[position] = stage_a(node.args[position], level)
        return cast(sp.Expr, node.func(*new_args))

    staged = stage_a(expr, 0)
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


def _gauss_term(term: sp.Expr, var: sp.Symbol, d_expr: sp.Expr) -> sp.Expr | None:
    """Close Sum_var omega_d^{s*var^2 + b*var + c} * constant for s = +-1 and integer b, or None."""
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
    if polynomial.degree() != 2:
        return None
    sign = polynomial.coeff_monomial(var**2)
    if sign not in (sp.Integer(1), sp.Integer(-1)):
        return None
    linear = sp.expand(polynomial.coeff_monomial(var))
    if linear.is_integer is not True:
        return None
    half_linear = sp.expand(linear / 2)
    offset = polynomial.coeff_monomial(1)
    # s*k^2 + 2*h*k + c = s*(k + s*h)^2 - s*h^2 + c, using s^2 = 1.
    shift = sp.exp(2 * sp.pi * sp.I * sp.expand(offset - sign * half_linear**2) / d_expr)
    if half_linear.is_integer is True:
        gauss = sp.sqrt(d_expr) * (1 + sign * sp.I) * (1 + (sign * sp.I) ** (-d_expr)) / 2
        return cast(sp.Expr, constant * shift * gauss)
    # sqrt(d) (1 + s*i)/2 omega_d^{-s*b^2/4} (1 + (-s*i)^d (-1)^b)
    parity = sp.exp(sp.pi * sp.I * linear)
    gauss = sp.sqrt(d_expr) * (1 + sign * sp.I) * (1 + (sign * sp.I) ** (-d_expr) * parity) / 2
    return cast(sp.Expr, constant * shift * gauss)


def _unit_sign(coefficient: sp.Expr, modulus: sp.Expr) -> int | None:
    """``+1`` or ``-1`` when ``coefficient`` is congruent to it mod ``modulus``, else None."""
    if coefficient == 1:
        return 1
    if coefficient == -1:
        return -1
    if coefficient.is_Integer and modulus.is_Integer and int(modulus) > 1:
        residue = int(coefficient) % int(modulus)
        if residue == 1:
            return 1
        if residue == int(modulus) - 1:
            return -1
    return None


def _linear_in(expr: sp.Expr, var: sp.Symbol) -> tuple[int, sp.Expr] | None:
    """``(c, rest)`` with ``expr == c*var + rest``, ``c`` a nonzero integer, or None."""
    try:
        polynomial = sp.Poly(expr, var)
    except sp.PolynomialError:
        return None
    if polynomial.degree() != 1:
        return None
    coefficient = polynomial.coeff_monomial(var)
    if not coefficient.is_Integer:
        return None
    rest = sp.expand(expr - coefficient * var)
    if var in rest.free_symbols:
        return None
    return int(coefficient), rest


def _integer_slope(expr: sp.Expr, var: sp.Symbol) -> bool:
    """Whether ``expr`` is a polynomial in ``var`` whose var-dependent coefficients are integers."""
    try:
        polynomial = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError:
        return False
    return all(
        coefficient.is_integer is True
        for monomial, coefficient in polynomial.terms()
        if monomial[0] > 0
    )


def _divides(modulus: sp.Expr, d_expr: sp.Expr) -> bool:
    """Whether ``modulus`` provably divides ``d_expr``."""
    if sp.expand(modulus - d_expr) == 0:
        return True
    if isinstance(modulus, ModGcd) and sp.expand(modulus.args[1] - d_expr) == 0:
        return True
    return sp.cancel(d_expr / modulus).is_integer is True


def _is_periodic_in(expr: sp.Expr, var: sp.Symbol, d_expr: sp.Expr) -> bool:
    """Whether expr, read as a function of the integer var, provably has period d.

    An omega_d power is periodic when its exponent is a polynomial in var whose
    var-dependent coefficients are integers; a ModDelta likewise, when its modulus divides d.
    Sums, products, integer-free powers, conjugates and nested index sums of periodic
    pieces are periodic. Anything else answers False, which only ever withholds a closure.
    """
    if var not in expr.free_symbols:
        return True
    if isinstance(expr, sp.exp):
        return _integer_slope(expr.args[0] * d_expr / (2 * sp.pi * sp.I), var)
    if isinstance(expr, ModDelta):
        frequency, modulus = expr.args
        if var in modulus.free_symbols:
            return False
        if _divides(modulus, d_expr):
            return _integer_slope(frequency, var)
        linear = _linear_in(sp.expand(frequency), var) if frequency.is_integer else None
        if linear is None:
            try:
                polynomial = sp.Poly(sp.expand(frequency), var)
            except sp.PolynomialError:
                return False
            if polynomial.degree() != 1:
                return False
            slope = polynomial.coeff_monomial(var)
        else:
            slope = sp.Integer(linear[0])
        return bool(sp.cancel(slope * d_expr / modulus).is_integer)
    if expr.is_Mul or expr.is_Add:
        return all(_is_periodic_in(cast(sp.Expr, arg), var, d_expr) for arg in expr.args)
    if expr.is_Pow:
        base, exponent = expr.args
        return var not in exponent.free_symbols and _is_periodic_in(base, var, d_expr)
    if isinstance(expr, sp.conjugate):
        return _is_periodic_in(cast(sp.Expr, expr.args[0]), var, d_expr)
    if isinstance(expr, sp.Sum):
        for bound, _lower, upper in expr.limits:
            if bound == var or var in upper.free_symbols:
                return False
        return _is_periodic_in(cast(sp.Expr, expr.function), var, d_expr)
    return False


def _bound_symbols(expr: sp.Expr) -> set[sp.Symbol]:
    """Every index bound by a Sum anywhere inside expr."""
    return {limit[0] for node in expr.atoms(sp.Sum) for limit in node.limits}


def _delta_base(factor: sp.Expr) -> ModDelta | None:
    """The ModDelta ``factor`` is, or is a positive integer power of, else None."""
    if isinstance(factor, ModDelta):
        return factor
    if factor.is_Pow and isinstance(factor.base, ModDelta) and factor.exp.is_Integer:
        return factor.base if factor.exp > 0 else None
    return None


def _without(limits: list[_Limit], position: int) -> list[_Limit]:
    return [limit for i, limit in enumerate(limits) if i != position]


def _solve_congruence(
    weight: sp.Expr, var: sp.Symbol, coefficient: int, rest: sp.Expr, size: int
) -> sp.Expr:
    """Sum_{var in Z_size} [coefficient*var + rest == 0 mod size] * weight, over its gcd roots."""
    coefficient %= size
    g = math.gcd(coefficient, size)
    step = size // g
    inverse = pow(coefficient // g, -1, step) if step > 1 else 0
    if rest.is_Integer:
        if int(rest) % g:
            return sp.Integer(0)
        base = (-(int(rest) // g) * inverse) % step if step > 1 else 0
        return cast(
            sp.Expr,
            sp.Add(*(weight.xreplace({var: sp.Integer(base + t * step)}) for t in range(g))),
        )
    root = sp.expand(-rest * sp.Rational(inverse, g))
    roots = sp.Add(*(weight.xreplace({var: sp.expand(root + t * step)}) for t in range(g)))
    return cast(sp.Expr, ModDelta(rest, sp.Integer(g)) * roots)


def _residue_roots(
    coefficient: int, rest: sp.Expr, modulus: sp.Expr
) -> tuple[sp.Expr, list[sp.Expr]] | None:
    """``(guard, roots)`` solving ``c*u + r == 0 mod M`` when ``M mod |c|`` is a known residue, else
    None."""
    c = abs(coefficient)
    if c < 2 or modulus.is_integer is not True:
        return None
    residue = _reduce_mod(sp.expand(modulus), c)
    if not residue.is_Integer:
        return None
    e = math.gcd(c, int(residue))
    reduced_c = c // e
    reduced_modulus = sp.expand(modulus / e)
    if reduced_c == 1:
        inverse: sp.Expr = sp.Integer(1)
    else:
        y0 = (-pow((int(residue) // e) % reduced_c, -1, reduced_c)) % reduced_c
        inverse = sp.expand((1 + reduced_modulus * y0) / reduced_c)
    target = -rest if coefficient > 0 else rest
    root = sp.expand(target * inverse / e)
    roots = [sp.expand(root + t * reduced_modulus) for t in range(e)]
    return ModDelta(rest, sp.Integer(e)), roots


def _expand_delta(
    term: sp.Expr, limits: list[_Limit], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, list[_Limit]] | None:
    """Close an index pinned by a concrete-modulus ModDelta by expanding the delta into characters,
    else None."""
    factors = sp.Mul.make_args(term)
    for position, factor in enumerate(factors):
        delta = _delta_base(factor)
        if delta is None:
            continue
        frequency, modulus = delta.args
        if not modulus.is_Integer or int(modulus) < 2:
            continue
        weight = cast(sp.Expr, sp.Mul(*(f for i, f in enumerate(factors) if i != position)))
        for index, (var, _lower, upper) in enumerate(limits):
            size = sp.expand(upper + 1)
            if var not in frequency.free_symbols or not _divides(modulus, size):
                continue
            if not _integer_slope(frequency, var) or not _is_periodic_in(weight, var, size):
                continue
            m = int(modulus)
            characters = sp.Add(
                *(sp.exp(2 * sp.pi * sp.I * j * sp.expand(frequency) / m) for j in range(m))
            )
            expanded = sp.expand(sp.powsimp(sp.expand(weight * characters / m), force=True))
            closed = sp.Add(
                *(
                    _reduce_term(cast(sp.Expr, piece), limits, ranges)[0]
                    for piece in sp.Add.make_args(expanded)
                )
            )
            if var in closed.free_symbols or var in _bound_symbols(closed):
                continue
            return cast(sp.Expr, closed), []
    return None


def _fresh_index(expr: sp.Expr) -> sp.Symbol:
    """An engine index ``_k<n>`` named after every reserved index in ``expr``."""
    taken = [
        int(str(s.name)[2:]) for s in expr.atoms(sp.Symbol) if _RESERVED_INDEX.match(str(s.name))
    ]
    return sp.Symbol(f"_k{max(taken, default=-1) + 1}", integer=True, nonnegative=True)


def _radix_step(term: sp.Expr, limits: list[_Limit]) -> tuple[sp.Expr, list[_Limit]] | None:
    """Merge two bound indices ``a``, ``b`` read only as ``a*R_b + b`` into one index over
    ``R_a*R_b``, else None."""
    whole = cast(sp.Expr, sp.Sum(term, *limits))
    for i, (a, _lo_a, upper_a) in enumerate(limits):
        for j, (b, _lo_b, upper_b) in enumerate(limits):
            if i == j or a not in term.free_symbols or b not in term.free_symbols:
                continue
            size_a, size_b = sp.expand(upper_a + 1), sp.expand(upper_b + 1)
            if size_b.is_Integer or a in size_b.free_symbols or b in size_a.free_symbols:
                continue
            u = _fresh_index(whole)
            merged = sp.powsimp(sp.expand(term.xreplace({b: u - a * size_b})), force=True)
            if a in merged.free_symbols:
                continue
            rest = [limit for k, limit in enumerate(limits) if k not in (i, j)]
            return cast(sp.Expr, merged), [
                *rest,
                (u, sp.Integer(0), sp.expand(size_a * size_b - 1)),
            ]
    return None


def _bounds(expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> tuple[sp.Expr, sp.Expr] | None:
    """``(least, greatest)`` of an integer-linear ``expr`` over indices sized in ``ranges``, else
    None."""
    form = _linear_form(expr)
    if form is None:
        return None
    least = greatest = sp.expand(expr - sum((c * v for v, c in form.items()), sp.Integer(0)))
    for var, coefficient in form.items():
        size = ranges.get(var)
        if size is None:
            return None
        if coefficient > 0:
            greatest += coefficient * (size - 1)
        else:
            least += coefficient * (size - 1)
    return sp.expand(least), sp.expand(greatest)


def _radix_split(delta: ModDelta, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    """``[t*A + B == 0 mod P*t]`` as ``[B == 0 mod t] * [A == 0 mod P]`` when ``ranges`` keep ``|B|
    < t``, else None."""
    frequency, modulus = delta.args
    if modulus.is_Integer or len(sp.Mul.make_args(modulus)) < 2:
        return None
    for t in (f for f in sp.Mul.make_args(modulus) if not f.is_Integer):
        carry: sp.Expr = sp.Integer(0)
        digit: sp.Expr = sp.Integer(0)
        for piece in sp.Add.make_args(sp.expand(frequency)):
            if sp.denom(sp.cancel(piece / t)) == 1:
                carry += sp.cancel(piece / t)
            else:
                digit += piece
        if digit != 0:
            bounds = _bounds(digit, ranges)
            if bounds is None:
                continue
            least, greatest = bounds
            if not (
                sp.expand(t - greatest).is_positive is True
                and sp.expand(t + least).is_positive is True
            ):
                continue
        rest = sp.cancel(modulus / t)
        return cast(sp.Expr, ModDelta(sp.expand(digit), t) * ModDelta(sp.expand(carry), rest))
    return None


def _split_radix_deltas(expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """:func:`_radix_split` on every ModDelta, bound indices sized by their limits."""
    if isinstance(expr, sp.Sum):
        inner = dict(ranges)
        for var, _lower, upper in expr.limits:
            inner[var] = sp.expand(upper + 1)
        return cast(sp.Expr, sp.Sum(_split_radix_deltas(expr.function, inner), *expr.limits))
    if isinstance(expr, ModDelta):
        return _radix_split(expr, ranges) or expr
    if not expr.args:
        return expr
    return cast(sp.Expr, expr.func(*(_split_radix_deltas(arg, ranges) for arg in expr.args)))


def _case_candidates(term: sp.Expr, limits: list[_Limit]) -> list[tuple[sp.Symbol, int]]:
    """``(d, L)`` pairs whose residue cases ``d = L*q + rho`` can close a stuck delta in
    ``term``."""
    bound = {limit[0] for limit in limits}
    found: list[tuple[sp.Symbol, int]] = []
    for factor in sp.Mul.make_args(term):
        delta = _delta_base(factor)
        if delta is None:
            continue
        frequency, modulus = delta.args
        for var in bound & frequency.free_symbols:
            linear = _linear_in(frequency, var)
            if linear is None:
                continue
            if isinstance(modulus, ModGcd) and modulus.args[0].is_Integer:
                period, carrier = int(modulus.args[0]), modulus.args[1]
            elif not modulus.is_Integer and abs(linear[0]) >= 2:
                period, carrier = abs(linear[0]), modulus
            else:
                continue
            symbols = [
                s
                for s in carrier.free_symbols
                if s.is_integer is True
                and not _RESERVED_INDEX.match(str(s.name))
                and not _CASE_SYMBOL.match(str(s.name))
            ]
            single = len(symbols) == 1 and len(carrier.free_symbols) == 1
            if period >= 2 and single and (symbols[0], period) not in found:
                found.append((symbols[0], period))
    return found


def _fresh_case_symbol(expr: sp.Expr) -> sp.Symbol:
    taken = {str(s.name) for s in expr.atoms(sp.Symbol) if _CASE_SYMBOL.match(str(s.name))}
    n = 0
    while f"_q{n}" in taken:
        n += 1
    return sp.Symbol(f"_q{n}", integer=True, nonnegative=True)


def _case_step(
    term: sp.Expr, limits: list[_Limit], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, list[_Limit]] | None:
    """Close a stuck delta by splitting a dimension into its residue cases, else None."""
    whole = cast(sp.Expr, sp.Sum(term, *limits))
    for symbol, period in _case_candidates(term, limits):
        q = _fresh_case_symbol(whole)
        cases: list[sp.Expr] = []
        progressed = False
        for rho in range(period):
            value = sp.expand(period * q + rho)
            case_term = cast(sp.Expr, term.xreplace({symbol: value}))
            case_limits = [(v, lo, sp.expand(up.xreplace({symbol: value}))) for v, lo, up in limits]
            case_ranges = {v: sp.expand(r.xreplace({symbol: value})) for v, r in ranges.items()}
            body = sp.expand(sp.powsimp(sp.expand(case_term), force=True))
            reduced = []
            for piece in sp.Add.make_args(body):
                closed, changed = _reduce_term(cast(sp.Expr, piece), case_limits, case_ranges)
                reduced.append(closed)
                progressed = progressed or changed
            back = sp.Add(*reduced).xreplace({q: (symbol - rho) / sp.Integer(period)})
            cases.append(ModDelta(symbol - rho, sp.Integer(period)) * back)
        if progressed:
            return cast(sp.Expr, sp.Add(*cases)), []
    return None


def _delta_step(
    term: sp.Expr, limits: list[_Limit], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, list[_Limit]] | None:
    """Remove one bound index pinned by a ModDelta factor over that index's own range.

    Sum_u [c*u + r == 0 mod m] g(u), m dividing u's range d and g m-periodic in u, is
    (d/m) times the sum over Z_m: c = +-1 substitutes u = -c*r; any other integer c gives
    gcd(c, m) * [gcd(c, m) | r] g when g is free of u, or the gcd(c, m) roots one by one when
    m is concrete or its residue mod c is known (:func:`_residue_roots`).
    """
    factors = sp.Mul.make_args(term)
    for position, factor in enumerate(factors):
        delta = _delta_base(factor)
        if delta is None:
            continue
        frequency, modulus = delta.args
        weight = cast(sp.Expr, sp.Mul(*(f for i, f in enumerate(factors) if i != position)))
        for index, (var, _lower, upper) in enumerate(limits):
            size = sp.expand(upper + 1)
            if var not in frequency.free_symbols or not _divides(modulus, size):
                continue
            linear = _linear_in(frequency, var)
            if linear is None:
                continue
            coefficient, rest = linear
            if rest.is_integer is not True or _bound_symbols(weight) & rest.free_symbols:
                continue
            sign = _unit_sign(sp.Integer(coefficient), modulus)
            if not _is_periodic_in(weight, var, modulus):
                if sign is None or sp.expand(modulus - size) != 0:
                    continue
                known = {**ranges, **{v: sp.expand(u + 1) for v, _l, u in limits if v != var}}
                bounds = _bounds(sp.expand(-sign * rest), known)
                if bounds is None or not (
                    bounds[0].is_nonnegative is True
                    and sp.expand(size - 1 - bounds[1]).is_nonnegative is True
                ):
                    continue
                return weight.xreplace({var: sp.expand(-sign * rest)}), _without(limits, index)
            scaled = weight * sp.cancel(size / modulus)
            if sign is not None:
                return scaled.xreplace({var: sp.expand(-sign * rest)}), _without(limits, index)
            if modulus.is_Integer:
                closed = _solve_congruence(scaled, var, coefficient, rest, int(modulus))
                return closed, _without(limits, index)
            if var not in weight.free_symbols:
                g = ModGcd(coefficient, modulus)
                return scaled * g * ModDelta(rest, g), _without(limits, index)
            roots = _residue_roots(coefficient, rest, modulus)
            if roots is not None:
                guard, values = roots
                pinned = sp.Add(*(scaled.xreplace({var: value}) for value in values))
                return guard * pinned, _without(limits, index)
    return None


def _character_step(term: sp.Expr, limits: list[_Limit]) -> tuple[sp.Expr, list[_Limit]] | None:
    """Close one index read only through omega powers:
    Sum_u omega_d^{u*j + o} = omega_d^o * d * [d | j]."""
    for index, (var, _lower, upper) in enumerate(limits):
        size = sp.expand(upper + 1)
        split = _split_term(term, var, size)
        if split is None:
            continue
        constant, frequency, offset = split
        if frequency.is_integer is not True:
            continue
        shift = sp.exp(2 * sp.pi * sp.I * offset / size)
        return constant * shift * size * ModDelta(frequency, size), _without(limits, index)
    return None


def _gauss_step(term: sp.Expr, limits: list[_Limit]) -> tuple[sp.Expr, list[_Limit]] | None:
    """Close one index through :func:`_gauss_term`."""
    for index, (var, _lower, upper) in enumerate(limits):
        closed = _gauss_term(term, var, sp.expand(upper + 1))
        if closed is not None:
            return closed, _without(limits, index)
    return None


def _reduce_term(
    term: sp.Expr, limits: list[_Limit], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, bool]:
    """``Sum(term, *limits)`` with every closable index closed, and whether any was."""
    if not limits:
        return term, False
    unused = [limit for limit in limits if limit[0] not in term.free_symbols]
    if unused:
        size = sp.Mul(*(sp.expand(upper + 1) for _var, _lower, upper in unused))
        used = [limit for limit in limits if limit[0] in term.free_symbols]
        step: tuple[sp.Expr, list[_Limit]] | None = (term * size, used)
    else:
        step = _delta_step(term, limits, ranges) or _character_step(term, limits)
        step = step or _gauss_step(term, limits) or _radix_step(term, limits)
        step = step or _expand_delta(term, limits, ranges) or _case_step(term, limits, ranges)
    if step is None:
        return cast(sp.Expr, sp.Sum(term, *limits)), False
    closed, remaining = step
    body = sp.expand(sp.powsimp(sp.expand(closed), force=True))
    pieces = [
        _reduce_term(cast(sp.Expr, piece), remaining, ranges)[0] for piece in sp.Add.make_args(body)
    ]
    return cast(sp.Expr, sp.Add(*pieces)), True


def _close_sum(node: sp.Sum, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    """Close what the identities reach of one Sum node, else enumerate one concrete index."""
    limits: list[_Limit] = list(node.limits)
    body = sp.expand(sp.powsimp(sp.expand(node.function), force=True))
    pieces: list[sp.Expr] = []
    changed = False
    for term in sp.Add.make_args(body):
        reduced, term_changed = _reduce_term(cast(sp.Expr, term), limits, ranges)
        pieces.append(reduced)
        changed = changed or term_changed
    if changed:
        return cast(sp.Expr, sp.Add(*pieces))
    concrete = [i for i, limit in enumerate(limits) if sp.expand(limit[2] + 1).is_Integer]
    if not concrete:
        return None
    position = min(concrete, key=lambda i: (int(sp.expand(limits[i][2] + 1)), -i))
    var, _lower, upper = limits[position]
    enumerated = sp.Add(
        *(node.function.xreplace({var: sp.Integer(v)}) for v in range(int(upper) + 1))
    )
    others = _without(limits, position)
    return cast(sp.Expr, sp.Sum(enumerated, *others) if others else enumerated)


def _linear_form(expr: sp.Expr) -> dict[sp.Symbol, int] | None:
    """The integer coefficient of each symbol in an integer-linear ``expr``, or None."""
    symbols = sorted(expr.free_symbols, key=lambda symbol: str(symbol.name))
    if not all(symbol.is_integer is True for symbol in symbols):
        return None
    if not symbols:
        return {} if expr.is_Integer else None
    try:
        polynomial = sp.Poly(expr, *symbols)
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() > 1 or not all(c.is_Integer for c in polynomial.coeffs()):
        return None
    return {symbol: int(polynomial.coeff_monomial(symbol)) for symbol in symbols}


def _pivot(row: sp.Expr, modulus: sp.Expr) -> tuple[sp.Symbol, sp.Expr] | None:
    """``(var, value)`` with ``[row == 0] == [var == value]`` mod ``modulus``, var the
    greatest-named engine index of unit coefficient, or None."""
    form = _linear_form(row)
    if form is None:
        return None
    candidates = [
        symbol
        for symbol, coefficient in form.items()
        if _RESERVED_INDEX.match(str(symbol.name))
        and _unit_sign(sp.Integer(coefficient), modulus) is not None
    ]
    if not candidates:
        return None
    var = max(candidates, key=lambda symbol: str(symbol.name))
    sign = _unit_sign(sp.Integer(form[var]), modulus)
    assert sign is not None
    return var, sp.expand(-sign * (row - form[var] * var))


def _echelon(
    rows: list[sp.Expr], modulus: sp.Expr
) -> tuple[dict[sp.Symbol, sp.Expr], list[sp.Expr]]:
    """Gauss-Jordan reduction of ``rows`` mod ``modulus`` over unit pivots, as (pivots,
    residual rows)."""
    pivots: dict[sp.Symbol, sp.Expr] = {}
    pending = list(rows)
    while True:
        residual: list[sp.Expr] = []
        progressed = False
        for row in pending:
            row = sp.expand(row.xreplace(pivots))
            found = _pivot(row, modulus)
            if found is None:
                residual.append(row)
                continue
            var, value = found
            pivots = {s: sp.expand(v.xreplace({var: value})) for s, v in pivots.items()}
            pivots[var] = value
            progressed = True
        pending = residual
        if not progressed:
            return pivots, pending


def _canonical_deltas(term: sp.Expr) -> sp.Expr:
    """A product with its linear ModDelta factors in echelon form, each pivot substituted
    into the rest wherever that rest is periodic in it."""
    by_modulus: dict[sp.Expr, list[sp.Expr]] = {}
    others: list[sp.Expr] = []
    for factor in sp.Mul.make_args(term):
        delta = _delta_base(factor)
        if delta is not None and _linear_form(delta.args[0]) is not None:
            by_modulus.setdefault(delta.args[1], []).append(delta.args[0])
        else:
            others.append(factor)
    if not by_modulus:
        return term
    rest = cast(sp.Expr, sp.Mul(*others))
    emitted: list[sp.Expr] = []
    for modulus in sorted(by_modulus, key=sp.default_sort_key):
        pivots, residual = _echelon(by_modulus[modulus], modulus)
        for var in sorted(pivots, key=lambda symbol: str(symbol.name)):
            value = pivots[var]
            if (
                var in rest.free_symbols
                and var not in _bound_symbols(rest)
                and not _bound_symbols(rest) & value.free_symbols
                and _is_periodic_in(rest, var, modulus)
            ):
                rest = rest.xreplace({var: value})
            emitted.append(ModDelta(var - value, modulus))
        emitted.extend(ModDelta(row, modulus) for row in residual)
    return cast(sp.Expr, rest * sp.Mul(*emitted))


def _canonical_top(expr: sp.Expr) -> sp.Expr:
    """:func:`_canonical_deltas` over every top-level term of ``expr``."""
    collapsed = expr.replace(
        lambda node: _delta_base(node) is not None and node.is_Pow,
        lambda node: node.base,
    )
    terms = sp.Add.make_args(sp.expand(collapsed))
    return cast(sp.Expr, sp.expand(sp.Add(*(_canonical_deltas(t) for t in terms))))


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


def _reduce_integer_phase(arg: sp.Expr) -> sp.Expr:
    """``2*pi*i*q`` with each rational coefficient of ``q``'s integer-symbol monomials taken mod 1.

    Returns ``arg`` unchanged unless ``q`` is a polynomial over the rationals in symbols all
    assumed integer.
    """
    q = sp.expand(arg / (2 * sp.pi * sp.I))
    symbols = sorted(q.free_symbols, key=lambda symbol: str(symbol.name))
    if not all(symbol.is_integer is True for symbol in symbols):
        return arg
    try:
        poly = sp.Poly(q, *symbols) if symbols else None
    except sp.PolynomialError:
        return arg
    if poly is None:
        return 2 * sp.pi * sp.I * (q % 1) if q.is_Rational else arg
    if not all(coefficient.is_Rational for coefficient in poly.coeffs()):
        return arg
    reduced = sum(
        (
            (coefficient % 1) * sp.Mul(*(s**e for s, e in zip(symbols, monomial)))
            for monomial, coefficient in poly.terms()
        ),
        sp.Integer(0),
    )
    return 2 * sp.pi * sp.I * reduced


def _reduce_integer_phases(expr: sp.Expr) -> sp.Expr:
    """Apply :func:`_reduce_integer_phase` to every exponential's argument in ``expr``."""
    return cast(
        sp.Expr,
        expr.replace(
            lambda node: isinstance(node, sp.exp),
            lambda node: sp.exp(_reduce_integer_phase(node.args[0])),
        ),
    )


def _close_all(
    expr: sp.Expr, budget: list[int], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, bool]:
    """Close every closable Sum bottom-up, sizing free indices by ``ranges``; reports whether
    anything changed."""
    changed = False
    inner = ranges
    if isinstance(expr, sp.Sum):
        inner = {**ranges, **{var: sp.expand(upper + 1) for var, _lower, upper in expr.limits}}
    if expr.args:
        new_args = list(expr.args)
        for position, arg in enumerate(expr.args):
            if isinstance(expr, sp.Sum) and position != 0:
                continue  # limit tuples carry bound symbols, never sub-expressions to close
            replaced, sub_changed = _close_all(cast(sp.Expr, arg), budget, inner)
            new_args[position] = replaced
            changed = changed or sub_changed
        expr = cast(sp.Expr, expr.func(*new_args))
    if isinstance(expr, sp.Sum):
        budget[0] -= 1
        if budget[0] < 0:
            raise ScalarBudgetError(
                "simplification exceeded its step budget; the expression is left unsimplified"
            )
        result = _close_sum(expr, ranges)
        if result is not None:
            return result, True
    return expr, changed


def _case_delta_symbol(node: sp.Expr) -> tuple[sp.Symbol, int] | None:
    """``(d, L)`` when ``node`` is ``[p(d) == 0 mod L]`` for a concrete ``L`` and integer polynomial
    ``p``, else None."""
    if not isinstance(node, ModDelta):
        return None
    frequency, modulus = node.args
    if not modulus.is_Integer or int(modulus) < 2 or len(frequency.free_symbols) != 1:
        return None
    (symbol,) = frequency.free_symbols
    name = str(symbol.name)
    if symbol.is_integer is not True or _RESERVED_INDEX.match(name) or _CASE_SYMBOL.match(name):
        return None
    try:
        polynomial = sp.Poly(frequency, symbol)
    except sp.PolynomialError:
        return None
    if not all(coefficient.is_Integer for coefficient in polynomial.coeffs()):
        return None
    return symbol, int(modulus)


def _split_cases(expr: sp.Expr, budget: list[int], ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """``expr`` as ``Sum_rho [d == rho mod L] * expr|_{d = L*q + rho}``, each case simplified."""
    moduli: dict[sp.Symbol, int] = {}
    for node in expr.atoms(ModDelta):
        found = _case_delta_symbol(node)
        if found is not None:
            symbol, modulus = found
            moduli[symbol] = math.lcm(moduli.get(symbol, 1), modulus)
    if not moduli:
        return expr
    symbol = min(moduli, key=lambda s: str(s.name))
    period = moduli[symbol]
    q = _fresh_case_symbol(expr)
    cases: list[sp.Expr] = []
    for rho in range(period):
        value = sp.expand(period * q + rho)
        case = expr.xreplace({symbol: value})
        case_ranges = {
            var: sp.expand(size.xreplace({symbol: value})) for var, size in ranges.items()
        }
        case = _simplify_expr(_rename_indices(cast(sp.Expr, case)), budget, case_ranges)
        if case != 0:
            back = case.xreplace({q: (symbol - rho) / sp.Integer(period)})
            cases.append(ModDelta(symbol - rho, sp.Integer(period)) * back)
    return cast(sp.Expr, sp.expand(sp.Add(*cases)))


def _simplify_expr(
    expr: sp.Expr, budget: list[int], ranges: Mapping[sp.Symbol, sp.Expr]
) -> sp.Expr:
    """The fixpoint of radix splitting, :func:`_close_all` and canonicalisation, then residue cases
    split."""
    while True:
        budget[0] -= 1
        if budget[0] < 0:
            raise ScalarBudgetError(
                "simplification exceeded its step budget; the expression is left unsimplified"
            )
        split = _split_radix_deltas(expr, ranges)
        radix_changed = split != expr
        expr, changed = _close_all(split, budget, ranges)
        changed = changed or radix_changed
        expr = _canonical_top(_reduce_integer_phases(expr))
        expr = _pull_constants(sp.powsimp(sp.expand(expr), force=True))
        if not changed:
            break
    return _split_cases(expr, budget, ranges)


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
        """The unit scalar e^{i*angle} corresponding to a :class:`~archytaszx.algebra.phase.Phase`.

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

    def simplify(
        self,
        *,
        max_steps: int = DEFAULT_MAX_SIMPLIFY_STEPS,
        ranges: Mapping[str, Dim] | None = None,
    ) -> Scalar:
        """Close every index sum the identities reach and put the remaining deltas in echelon form;
        ``ranges`` sizes free indices."""
        known = {_index_symbol(name): dim.to_sympy() for name, dim in (ranges or {}).items()}
        return Scalar(
            _rename_indices(_simplify_expr(_rename_indices(self._expr), [max_steps], known))
        )

    def with_dimension_floors(self, floors: Mapping[str, int]) -> Scalar:
        """This scalar with every ``[s | c]`` set to zero, ``c`` a nonzero integer and ``s`` a
        dimension symbol whose floor in ``floors`` exceeds ``|c|``."""

        def vanishes(node: sp.Expr) -> bool:
            if not isinstance(node, ModDelta):
                return False
            frequency, modulus = node.args
            return bool(
                frequency.is_Integer
                and frequency != 0
                and modulus.is_Symbol
                and abs(int(frequency)) < floors.get(str(modulus.name), 0)
            )

        return Scalar(self._expr.replace(vanishes, lambda _node: sp.Integer(0)))

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
