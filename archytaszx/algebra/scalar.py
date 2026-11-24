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

import hashlib
import itertools
import math
import re
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Union, cast

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker
from sympy.core.add import _unevaluated_Add  # type: ignore[import-untyped]
from sympy.core.mul import _unevaluated_Mul  # type: ignore[import-untyped]
from sympy.polys.constructor import construct_domain  # type: ignore[import-untyped]
from sympy.polys.polyoptions import build_options  # type: ignore[import-untyped]
from sympy.polys.polyutils import (  # type: ignore[import-untyped]
    _dict_from_expr,
    dict_from_expr,
)

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

    The grammar is: exact numbers, ``I``, sympy's named constants, algebraic numbers as
    ``CRootOf``, symbols, and sums,
    products, powers and ``exp``/``conjugate`` applications of these. A transcendental
    function (``log``, ``sin``) or any other head is rejected.
    """
    if expr.is_Number or expr is sp.I or expr.is_NumberSymbol or isinstance(expr, sp.CRootOf):
        return
    if expr.is_Symbol:
        _check_symbol_name(str(expr.name), allow_reserved=True)
        return
    if isinstance(expr, sp.Sum):
        _check_sum_structure(expr)
        _check_scalar_domain(expr.function)
        return
    if isinstance(expr, ModDelta):
        if expr not in _DOMAIN_OK:
            for arg in expr.args:
                _check_scalar_domain(arg)
            if len(_DOMAIN_OK) >= _MEMO_CAP:
                _DOMAIN_OK.clear()
            _DOMAIN_OK.add(expr)
        return
    if (
        expr.is_Add
        or expr.is_Mul
        or expr.is_Pow
        or isinstance(expr, (sp.exp, sp.conjugate, ModDelta, ModGcd, ModResidue))
    ):
        for arg in expr.args:
            _check_scalar_domain(arg)
        return
    raise ScalarGrammarError(
        f"expression {expr} is outside the scalar grammar (head {type(expr).__name__})"
    )


def _is_pi_rational(arg: sp.Expr) -> bool:
    """Whether ``arg`` is a rational multiple of pi."""
    return bool((arg / sp.pi).is_Rational)


def cyclotomic_trig(expr: sp.Expr) -> sp.Expr:
    """``expr`` with every cos(q pi) and sin(q pi), q rational, as (z + 1/z)/2 and
    (z - 1/z)/(2 I) for z = exp(I q pi)."""
    if not _has_outside_flat(expr, sp.cos, sp.sin):
        return expr

    def exact(e: sp.Expr) -> sp.Expr:
        z = sp.exp(sp.I * e.args[0])
        if isinstance(e, sp.cos):
            return (z + 1 / z) / 2
        return (z - 1 / z) / (2 * sp.I)

    return cast(
        sp.Expr,
        expr.replace(
            lambda e: isinstance(e, (sp.cos, sp.sin)) and _is_pi_rational(e.args[0]), exact
        ),
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

    cos(q pi) and sin(q pi) with q rational enter as their cyclotomic values
    (:func:`cyclotomic_trig`) before the expansion.

    Before any of that: an expression containing an ``sp.Float`` atom anywhere in its
    tree (not just at the top level) is rejected outright, since this module tracks
    scalars exactly, never approximately.
    """
    hit = _NORMALIZED.get(expr)
    if hit is not None:
        return hit
    if _has_outside_flat(expr, sp.Float):
        raise ScalarGrammarError(f"Scalar requires an exact expression, got a float in {expr!r}")
    normalized = _rename_indices(sp.powsimp(_expand(cyclotomic_trig(expr)), force=True))
    _check_scalar_domain(normalized)
    if len(_NORMALIZED) >= _MEMO_CAP:
        _NORMALIZED.clear()
    _NORMALIZED[expr] = normalized
    return cast(sp.Expr, normalized)


_NORMALIZED: dict[sp.Expr, sp.Expr] = {}
"""Results of :func:`_normalize` keyed by its argument."""

_SIMPLIFIED: dict[tuple[object, ...], object] = {}
"""Results of :meth:`Scalar.simplify` (a Scalar, or the budget error's message) keyed by the
expression, step budget, ranges and ``enumerate_concrete``."""

_DOMAIN_OK: set[sp.Expr] = set()
"""ModDelta nodes :func:`_check_scalar_domain` accepted."""

_MEMO_CAP = 1 << 13
"""The most entries :data:`_NORMALIZED` or :data:`_SIMPLIFIED` holds; a full table is emptied
before the next entry."""


_Limit = tuple[sp.Symbol, sp.Expr, sp.Expr]


def _freeze(*exprs: sp.Expr) -> tuple[list[sp.Expr], dict[sp.Symbol, sp.Expr]]:
    """``exprs`` with every :class:`ModResidue` atom replaced by an integer symbol
    ``_frozen<i>``, and the map back."""
    atoms: set[sp.Expr] = set()
    for expr in exprs:
        atoms |= expr.atoms(ModResidue)
    if not atoms:
        return list(exprs), {}
    forward = {
        atom: sp.Symbol(f"_frozen{i}", integer=True, nonnegative=True)
        for i, atom in enumerate(sorted(atoms, key=sp.default_sort_key))
    }
    return [expr.xreplace(forward) for expr in exprs], {v: k for k, v in forward.items()}


_Terms = dict[frozenset[tuple[sp.Symbol, int]], int]

_TERMS: dict[sp.Expr, _Terms] = {}
"""Results of :func:`_integer_terms` on sums, keyed by the sum; never mutated."""


def _remember_terms(expr: sp.Expr, terms: _Terms) -> None:
    """Store ``terms`` as :func:`_integer_terms` of the sum ``expr`` in :data:`_TERMS`."""
    if len(_TERMS) >= _MEMO_CAP:
        _TERMS.clear()
    _TERMS[expr] = terms


_FLAT: dict[sp.Expr, bool] = {}
"""Results of :func:`_flat_polynomial` on sums, keyed by the sum."""


def _flat_monomial(expr: sp.Expr) -> bool:
    """Whether ``expr`` is an integer, a symbol, a positive integer power of a symbol, or a product
    of these."""
    if expr.is_Symbol or expr.is_Integer:
        return True
    if expr.is_Pow:
        return bool(expr.base.is_Symbol and expr.exp.is_Integer and int(expr.exp) > 0)
    if expr.is_Mul:
        return all(
            a.is_Symbol
            or a.is_Integer
            or (a.is_Pow and a.base.is_Symbol and a.exp.is_Integer and int(a.exp) > 0)
            for a in expr.args
        )
    return False


def _flat_polynomial(expr: sp.Expr) -> bool:
    """Whether ``expr`` is a :func:`_flat_monomial` or a sum of them; memoised in :data:`_FLAT`."""
    if not expr.is_Add:
        return _flat_monomial(expr)
    hit = _FLAT.get(expr)
    if hit is None:
        hit = all(_flat_monomial(a) for a in expr.args)
        if len(_FLAT) >= _MEMO_CAP:
            _FLAT.clear()
        _FLAT[expr] = hit
    return hit


def _flat_delta(expr: sp.Expr) -> bool:
    """Whether ``expr`` is a ModDelta with an integer modulus and a :func:`_flat_polynomial`
    frequency."""
    return isinstance(expr, ModDelta) and expr.args[1].is_Integer and _flat_polynomial(expr.args[0])


def _fixed_factor(expr: sp.Expr) -> bool:
    """Whether ``expr`` is an integer, a symbol, a :func:`_flat_delta`, or a positive integer
    power of a symbol or of a flat delta."""
    if expr.is_Symbol or expr.is_Integer:
        return True
    if expr.is_Pow:
        return bool(
            expr.exp.is_Integer
            and int(expr.exp) > 0
            and (expr.base.is_Symbol or _flat_delta(expr.base))
        )
    return _flat_delta(expr)


def _expand_fixed(expr: sp.Expr) -> bool:
    """Whether ``expr`` is a :func:`_fixed_factor`, a product of them, or a sum of such products
    (sympy's expand returns it unchanged)."""
    terms = expr.args if expr.is_Add else (expr,)
    return all(
        all(_fixed_factor(f) for f in t.args) if t.is_Mul else _fixed_factor(t) for t in terms
    )


def _expand(expr: sp.Expr) -> sp.Expr:
    """``sympy.expand(expr)``, returning ``expr`` itself when :func:`_expand_fixed`."""
    if _expand_fixed(expr):
        return expr
    return cast(sp.Expr, sp.expand(expr))


def _outside_flat(expr: sp.Expr, test: Callable[[sp.Expr], bool]) -> bool:
    """Whether a node of ``expr`` passes ``test``, reading no node inside the frequency of a
    :func:`_flat_delta`."""
    stack = [expr]
    while stack:
        node = stack.pop()
        if test(node):
            return True
        if not _flat_delta(node):
            stack.extend(node.args)
    return False


def _delta_atoms(expr: sp.Expr) -> list[sp.Expr]:
    """The distinct ModDelta nodes of ``expr`` (``expr.atoms(ModDelta)``) in first-visit order,
    reading no node inside the frequency of a :func:`_flat_delta`."""
    found: dict[sp.Expr, None] = {}
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ModDelta):
            found[node] = None
        if not _flat_delta(node):
            stack.extend(node.args)
    return list(found)


def _has_outside_flat(expr: sp.Expr, *types: type) -> bool:
    """``expr.has(*types)`` for classes ``types`` that no flat polynomial holds."""
    return _outside_flat(expr, lambda node: isinstance(node, types))


def poly_terms(
    expr: sp.Expr, gens: tuple[sp.Expr, ...]
) -> list[tuple[tuple[int, ...], sp.Expr]] | None:
    """The terms of ``sp.Poly(expr, *gens)``, read from its sparse dict without the dense
    representation; None when it is not a polynomial over ``gens``."""
    try:
        found, _gens = dict_from_expr(expr, gens=gens)
    except sp.PolynomialError:
        return None
    terms = sorted(((m, c) for m, c in found.items() if c != 0), reverse=True)
    return terms or [((0,) * len(gens), sp.Integer(0))]


def poly_coeffs(expr: sp.Expr, gens: tuple[sp.Expr, ...]) -> list[sp.Expr]:
    """``sp.Poly(expr, *gens).coeffs()`` from the sparse dict and the domain ``sp.Poly`` builds,
    without the dense representation; raises as ``sp.Poly`` does."""
    opt = build_options(tuple(gens), {})
    rep, opt = _dict_from_expr(sp.sympify(expr), opt)
    domain, rep = construct_domain(rep, opt=opt)
    terms = sorted(((m, c) for m, c in rep.items() if not domain.is_zero(c)), reverse=True)
    return [domain.to_sympy(c) for _m, c in terms] or [sp.Integer(0)]


def _integer_terms(expr: sp.Expr) -> _Terms | None:
    """The nonzero integer coefficient of each monomial (a set of ``(symbol, exponent)``) of a
    polynomial ``expr`` built from symbols and integers by sums, products and nonnegative
    integer powers, or None for any other ``expr``."""
    if expr.is_Symbol:
        return {frozenset({(expr, 1)}): 1}
    if expr.is_Integer:
        return {frozenset(): int(expr)} if expr != 0 else {}
    if expr.is_Add:
        hit = _TERMS.get(expr)
        if hit is not None:
            return hit
        out: _Terms = {}
        for arg in expr.args:
            part = _integer_terms(arg)
            if part is None:
                return None
            for mono, c in part.items():
                total = out.get(mono, 0) + c
                if total:
                    out[mono] = total
                else:
                    out.pop(mono, None)
        _remember_terms(expr, out)
        return out
    if expr.is_Mul:
        product: _Terms = {frozenset(): 1}
        for arg in expr.args:
            part = _integer_terms(arg)
            if part is None:
                return None
            product = _times(product, part)
        return product
    if expr.is_Pow and expr.exp.is_Integer and int(expr.exp) >= 0:
        if expr.base.is_Symbol and int(expr.exp) > 0:
            return {frozenset({(expr.base, int(expr.exp))}): 1}
        base = _integer_terms(expr.base)
        if base is None:
            return None
        power: _Terms = {frozenset(): 1}
        for _ in range(int(expr.exp)):
            power = _times(power, base)
        return power
    return None


def _times(left: _Terms, right: _Terms) -> _Terms:
    """The product of two :func:`_integer_terms` polynomials."""
    out: _Terms = {}
    for m1, c1 in left.items():
        powers1 = dict(m1)
        for m2, c2 in right.items():
            if not m1:
                mono = m2
            elif not m2:
                mono = m1
            else:
                powers = dict(powers1)
                for s, e in m2:
                    powers[s] = powers.get(s, 0) + e
                mono = frozenset(powers.items())
            total = out.get(mono, 0) + c1 * c2
            if total:
                out[mono] = total
            else:
                out.pop(mono, None)
    return out


def _residue_terms(terms: _Terms, modulus: int) -> _Terms:
    """``terms`` with every coefficient taken into ``(-modulus/2, modulus/2]``, zeros dropped."""
    out: _Terms = {}
    for mono, c in terms.items():
        residue = c % modulus
        if 2 * residue > modulus:
            residue -= modulus
        if residue:
            out[mono] = residue
    return out


def _terms_expr(terms: _Terms) -> sp.Expr:
    """The sympy sum of :func:`_integer_terms` polynomial ``terms``, each monomial's symbols in
    name order, built from its canonically sorted arguments (sympy's unevaluated Add and Mul)
    without flattening."""
    args: list[sp.Expr] = []
    constant = 0
    for mono, c in terms.items():
        if not mono:
            constant += c
            continue
        factors = [
            s if e == 1 else s**e for s, e in sorted(mono, key=lambda item: str(item[0].name))
        ]
        args.append(_unevaluated_Mul(sp.Integer(c), *factors))
    out = cast(sp.Expr, _unevaluated_Add(sp.Integer(constant), *args))
    if out.is_Add and all(terms.values()):
        _remember_terms(out, terms)
    return out


def _fast_reduce_mod(expr: sp.Expr, modulus: int) -> sp.Expr | None:
    """:func:`_reduce_mod` read from :func:`_integer_terms`, or None when ``expr`` is not such a
    polynomial in symbols."""
    if not expr.free_symbols:
        return None
    terms = _integer_terms(expr)
    if terms is None:
        return None
    return _terms_expr(_residue_terms(terms, modulus))


def _reduce_mod(expr: sp.Expr, modulus: int) -> sp.Expr:
    """``expr`` with every integer coefficient taken into ``(-modulus/2, modulus/2]``."""
    fast = _fast_reduce_mod(expr, modulus)
    if fast is not None:
        return fast
    if expr.has(ModResidue):
        (frozen,), back = _freeze(expr)
        return cast(sp.Expr, _reduce_mod(frozen, modulus).xreplace(back))
    symbols = sorted(expr.free_symbols, key=lambda symbol: str(symbol.name))
    if not symbols:
        if not expr.is_Integer:
            return expr
        residue = int(expr) % modulus
        return sp.Integer(residue - modulus if 2 * residue > modulus else residue)
    found = poly_terms(expr, tuple(symbols))
    if found is None or not all(coefficient.is_Integer for _m, coefficient in found):
        return expr
    terms: list[sp.Expr] = []
    for monomial, coefficient in found:
        residue = int(coefficient) % modulus
        if 2 * residue > modulus:
            residue -= modulus
        if residue:
            terms.append(residue * sp.Mul(*(s**e for s, e in zip(symbols, monomial) if e)))
    return cast(sp.Expr, sp.expand(sp.Add(*terms)))


def _plain_monomial(expr: sp.Expr) -> bool:
    """Whether ``expr`` is a rational number, a symbol, an integer power of a symbol, or a
    product of these."""
    return all(
        f.is_Rational or f.is_Symbol or (f.is_Pow and f.base.is_Symbol and f.exp.is_Integer)
        for f in sp.Mul.make_args(expr)
    )


def _ratio(a: sp.Expr, m: sp.Expr) -> sp.Expr:
    """``sp.cancel(a / m)``; for two :func:`_plain_monomial` arguments the quotient as sympy
    builds it."""
    if _plain_monomial(a) and _plain_monomial(m):
        return cast(sp.Expr, a / m)
    return cast(sp.Expr, sp.cancel(a / m))


def _reduce_symbolic_mod(expr: sp.Expr, modulus: sp.Expr) -> sp.Expr:
    """``expr`` with each term ``q * modulus`` (``q`` a rational number) moved to
    ``(q mod 1) * modulus``."""
    out: list[sp.Expr] = []
    for term in sp.Add.make_args(expr):
        ratio = _ratio(term, modulus)
        if ratio.is_Rational and not ratio.is_Integer:
            term = sp.expand((ratio - sp.floor(ratio)) * modulus)
        out.append(cast(sp.Expr, term))
    return cast(sp.Expr, sp.expand(sp.Add(*out)))


def _printable(number: sp.Rational) -> bool:
    """Whether ``str(number)`` stays within the interpreter's integer-to-string digit limit."""
    try:
        str(number)
    except ValueError:
        return False
    return True


def _int_digest(value: int) -> str:
    """A sha256 of ``value``'s signed big-endian bytes."""
    data = value.to_bytes(value.bit_length() // 8 + 1, "big", signed=True)
    return hashlib.sha256(data).hexdigest()[:32]


def _digit_free(expr: sp.Expr) -> sp.Expr:
    """``expr`` with each rational ``p/q`` too long to print replaced by the symbol
    ``_int_<digest p>_<digest q>``."""
    huge = {
        a: sp.Symbol(f"_int_{_int_digest(int(a.p))}_{_int_digest(int(a.q))}")
        for a in expr.atoms(sp.Rational)
        if not _printable(a)
    }
    return cast(sp.Expr, expr.xreplace(huge))


def _canonical_residue(expr: sp.Expr, modulus: sp.Expr) -> sp.Expr:
    """The representative of ``+-expr`` (reduced when ``modulus`` is concrete) a ModDelta keeps."""
    if modulus.is_Integer and expr.free_symbols:
        terms = _integer_terms(expr)
        if terms is not None:
            return _canonical_terms(terms, int(modulus))[1]
    candidates = [sp.expand(expr), sp.expand(-expr)]
    if modulus.is_Integer:
        candidates = [_reduce_mod(candidate, int(modulus)) for candidate in candidates]
    else:
        candidates = [_reduce_symbolic_mod(candidate, modulus) for candidate in candidates]
    preferred = [c for c in candidates if not c.could_extract_minus_sign()] or candidates
    return cast(sp.Expr, min(preferred, key=sp.default_sort_key))


def _canonical_terms(terms: _Terms, modulus: int) -> tuple[_Terms, sp.Expr]:
    """:func:`_canonical_residue` of the polynomial ``terms`` at a concrete ``modulus``, as its
    terms and its expression."""
    plus, minus, chosen = _canonical_choice(terms, modulus)
    if chosen is not None:
        return chosen, _terms_expr(chosen)
    candidates = [(plus, _terms_expr(plus)), (minus, _terms_expr(minus))]
    preferred = [c for c in candidates if not c[1].could_extract_minus_sign()] or candidates
    return min(preferred, key=lambda c: sp.default_sort_key(c[1]))


def _canonical_choice(terms: _Terms, modulus: int) -> tuple[_Terms, _Terms, _Terms | None]:
    """The residues of ``+-terms`` and the one :func:`_canonical_terms` keeps when the signs of
    the coefficients decide it (sympy's ``could_extract_minus_sign`` of a sum counts its negative
    terms), else None."""
    plus = _residue_terms(terms, modulus)
    minus = _residue_terms({mono: -c for mono, c in terms.items()}, modulus)
    if plus == minus:
        return plus, minus, plus
    signs = [_extracts_minus(plus), _extracts_minus(minus)]
    if None in signs or signs[0] == signs[1]:
        return plus, minus, None
    return plus, minus, minus if signs[0] else plus


def _extracts_minus(terms: _Terms) -> bool | None:
    """``could_extract_minus_sign`` of :func:`_terms_expr` of nonempty ``terms``, or None when
    sympy breaks a tie by sort keys."""
    negative = sum(1 for c in terms.values() if c < 0)
    if len(terms) == 1:
        return negative == 1
    if 2 * negative == len(terms):
        return None
    return 2 * negative > len(terms)


def _off_factor(j: sp.Expr, m: sp.Expr) -> bool:
    """True when a divisor ``g >= 2`` of the integer factor of the symbolic modulus ``m`` divides
    every term of ``j`` but its integer constant, and not that constant."""
    factor = sp.Mul(*(f for f in sp.Mul.make_args(m) if f.is_Integer))
    constant, rest = j.as_coeff_Add()
    if not (factor.is_Integer and constant.is_Integer):
        return False
    for g in range(2, min(int(factor), _OFF_FACTOR_MAX) + 1):
        if int(factor) % g or int(constant) % g == 0:
            continue
        if rest == 0 or sp.cancel(rest / g).is_integer is True:
            return True
    return False


_OFF_FACTOR_MAX = 64
"""The largest divisor :func:`_off_factor` tries."""


def _delta_of_terms(terms: _Terms, m: sp.Expr) -> sp.Expr | None:
    """:meth:`ModDelta.eval` of a frequency whose :func:`_integer_terms` are ``terms`` at an
    integer modulus ``m != 1``: the value, the canonical delta, or None for the frequency kept."""
    if not terms:
        return sp.Integer(1)
    if set(terms) == {frozenset()}:
        return sp.Integer(1 if terms[frozenset()] % int(m) == 0 else 0)
    chosen = _canonical_choice(terms, int(m))[2]
    if chosen is None:
        chosen, canonical = _canonical_terms(terms, int(m))
        return cast(sp.Expr, ModDelta(canonical, m)) if chosen != terms else None
    if chosen == terms:
        return None
    canonical = _terms_expr(chosen)
    if any(chosen):
        return cast(sp.Expr, ModDelta(canonical, m, evaluate=False))
    return cast(sp.Expr, ModDelta(canonical, m))


def _substitute_terms(terms: _Terms, var: sp.Symbol, value: _Terms) -> _Terms:
    """The :func:`_integer_terms` polynomial ``terms`` with ``var`` replaced by the polynomial
    ``value``."""
    out: _Terms = {}
    powers: dict[int, _Terms] = {1: value}
    for mono, c in terms.items():
        e = next((e for s, e in mono if s == var), 0)
        if not e:
            part: _Terms = {mono: c}
        else:
            if e not in powers:
                power: _Terms = {frozenset(): 1}
                for _ in range(e):
                    power = _times(power, value)
                powers[e] = power
            part = _times({frozenset(p for p in mono if p[0] != var): c}, powers[e])
        for m2, c2 in part.items():
            total = out.get(m2, 0) + c2
            if total:
                out[m2] = total
            else:
                out.pop(m2, None)
    return out


_SYMBOLIC_DELTAS: dict[tuple[sp.Expr, sp.Expr], sp.Expr | None] = {}
"""Results of :meth:`ModDelta.eval` at a symbolic modulus, keyed by its arguments."""


class ModDelta(sp.Function):  # type: ignore[misc]  # sympy is untyped
    """``[j == 0 mod m]`` for an integer-valued ``j`` and a positive integer modulus ``m``."""

    nargs = 2
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, j: sp.Expr, m: sp.Expr) -> sp.Expr | None:
        """The value when decidable, else ``j`` moved to its canonical residue; at a symbolic
        ``m`` memoised in :data:`_SYMBOLIC_DELTAS`."""
        if m.is_Integer:
            return cls._eval(j, m)
        key = (j, m)
        if key in _SYMBOLIC_DELTAS:
            return _SYMBOLIC_DELTAS[key]
        out = cls._eval(j, m)
        if len(_SYMBOLIC_DELTAS) >= _MEMO_CAP:
            _SYMBOLIC_DELTAS.clear()
        _SYMBOLIC_DELTAS[key] = out
        return out

    @classmethod
    def _eval(cls, j: sp.Expr, m: sp.Expr) -> sp.Expr | None:
        """:meth:`eval` without the memo."""
        if m.is_Integer and m != 1 and j.free_symbols:
            terms = _integer_terms(j)
            if terms is not None:
                return _delta_of_terms(terms, m)
        args_j = j
        j = sp.expand(j)
        if j == 0 or m == 1:
            return sp.Integer(1)
        if m.is_Integer and j.is_Integer:
            return sp.Integer(1 if int(j) % int(m) == 0 else 0)
        if not m.is_Integer and _ratio(j, m).is_integer is True:
            return sp.Integer(1)
        if not m.is_Integer and _off_factor(j, m):
            return sp.Integer(0)
        if not m.is_Integer and j.is_Add:
            kept = [t for t in j.args if _ratio(t, m).is_integer is not True]
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


class ModResidue(sp.Function):  # type: ignore[misc]  # sympy is untyped
    """``x mod m``: the residue in ``[0, m)`` of an integer ``x`` for a concrete modulus ``m``."""

    nargs = 2
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, x: sp.Expr, m: sp.Expr) -> sp.Expr | None:
        """The value when ``x`` is an integer, else ``x`` with its coefficients taken into
        ``[0, m)`` and each term ``c*(y mod m')`` with ``m | c*m'`` read as ``c*y``."""
        if not m.is_Integer or int(m) < 1:
            return None
        modulus = int(m)
        if modulus == 1:
            return sp.Integer(0)
        x = sp.expand(x)
        if x.is_Integer:
            return sp.Integer(int(x) % modulus)
        reduced: sp.Expr = sp.Integer(0)
        for term in sp.Add.make_args(x):
            coefficient, rest = term.as_coeff_Mul()
            if not coefficient.is_Integer or rest.is_integer is not True:
                return None
            inner = rest.args[1] if isinstance(rest, ModResidue) else None
            if inner is not None and inner.is_Integer and int(coefficient * inner) % modulus == 0:
                rest = rest.args[0]
            reduced += (int(coefficient) % modulus) * rest
        reduced = sp.expand(reduced)
        if reduced.is_Integer:
            return sp.Integer(int(reduced) % modulus)
        return cast(sp.Expr, cls(reduced, m)) if reduced != x else None


def _multiple_of(expr: sp.Expr, modulus: int) -> bool:
    """Whether every term of ``expr`` is an integer multiple of ``modulus`` times an integer."""
    for term in sp.Add.make_args(sp.expand(expr)):
        coefficient, rest = term.as_coeff_Mul()
        if not coefficient.is_Integer or int(coefficient) % modulus:
            return False
        if rest != 1 and rest.is_integer is not True:
            return False
    return True


def _residue_coefficient(expr: sp.Expr, atom: sp.Expr) -> sp.Expr | None:
    """``a`` with ``expr == a*atom + rest``, neither ``a`` nor ``rest`` reading ``atom``, else
    None."""
    expanded = sp.expand(expr)
    coefficient = expanded.coeff(atom)
    if coefficient.has(atom) or sp.expand(expanded - coefficient * atom).has(atom):
        return None
    return cast(sp.Expr, coefficient)


def _collapse_residues(expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """``expr`` with each ``x mod m`` read as ``x`` where only ``x mod m`` matters: an exponent
    ``2 pi i a (x mod m)`` with ``a*m`` an integer, a delta ``[a (x mod m) + r == 0 mod M]`` with
    ``a*m/M`` an integer, or ``x`` bounded in ``[0, m)`` by ``ranges``."""
    if not _has_outside_flat(expr, ModResidue):
        return expr

    def phase(node: sp.Expr) -> sp.Expr:
        q = sp.expand(node.args[0] / (2 * sp.pi * sp.I))
        for atom in sorted(q.atoms(ModResidue), key=sp.default_sort_key):
            a = _residue_coefficient(q, atom)
            if a is not None and sp.expand(a * atom.args[1]).is_integer is True:
                q = sp.expand(q.xreplace({atom: atom.args[0]}))
        return cast(sp.Expr, sp.exp(2 * sp.pi * sp.I * q))

    def delta(node: sp.Expr) -> sp.Expr:
        frequency, modulus = node.args
        for atom in sorted(frequency.atoms(ModResidue), key=sp.default_sort_key):
            a = _residue_coefficient(frequency, atom)
            if a is not None and sp.cancel(a * atom.args[1] / modulus).is_integer is True:
                frequency = sp.expand(frequency.xreplace({atom: atom.args[0]}))
        return cast(sp.Expr, ModDelta(frequency, modulus))

    def bounded(node: sp.Expr) -> sp.Expr:
        x, m = node.args
        found = _bounds(x, ranges) if x.free_symbols <= set(ranges) else None
        if found is None:
            return node
        least, greatest = found
        if least.is_nonnegative is True and sp.expand(m - 1 - greatest).is_nonnegative is True:
            return cast(sp.Expr, x)
        wrap = _wrap(x, m, ranges)
        return node if wrap is None else cast(sp.Expr, sp.expand(x - m * wrap))

    expr = expr.replace(lambda n: isinstance(n, ModResidue), bounded)
    expr = expr.replace(lambda n: isinstance(n, sp.exp) and n.has(ModResidue), phase)
    return cast(
        sp.Expr, expr.replace(lambda n: isinstance(n, ModDelta) and n.has(ModResidue), delta)
    )


def _wrap(x: sp.Expr, m: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    """``w`` with ``x mod m == x - m*w``, for an integer ``m`` and ``x`` integer-linear over bits
    of ``ranges`` in ``[0, 2m)`` or ``[-m, m)`` whose wrapping corners form one face of the cube
    (the bits ``S`` whose flip from the extreme corner unwraps it, fixed there): ``w`` the bit
    product of ``S`` at that corner, negated for ``[-m, m)``; else None."""
    form = _linear_over(x, ranges) if m.is_Integer else None
    if not form or any(sp.expand(ranges[v] - 2) != 0 or c == 0 for v, c in form.items()):
        return None
    found = _bounds(x, ranges)
    if found is None or not (found[0].is_Integer and found[1].is_Integer):
        return None
    least, greatest, modulus = int(found[0]), int(found[1]), int(m)
    if 0 <= least and modulus <= greatest < 2 * modulus:
        top, edge, sign = True, greatest - modulus, 1
    elif -modulus <= least < 0 and greatest < modulus:
        top, edge, sign = False, -1 - least, -1
    else:
        return None
    face = _face(form, edge, top)
    if face is not None:
        return cast(sp.Expr, sign * face)
    other = (modulus - 1 - least) if top else (greatest - 0)
    face = _face(form, other, not top)
    return None if face is None else cast(sp.Expr, sign * (1 - face))


def _face(form: dict[sp.Symbol, int], edge: int, top: bool) -> sp.Expr | None:
    """The bit product of the face of corners within ``edge`` of the greatest (``top``) or least
    corner of ``sum c*v`` over bits, when those corners form one face; else None."""
    face = [v for v, c in form.items() if abs(c) > edge]
    if sum(abs(c) for v, c in form.items() if v not in face) > edge:
        return None
    return cast(
        sp.Expr, sp.Mul(*(v if (c > 0) == top else 1 - v for v, c in form.items() if v in face))
    )


def _pin_residues(term: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """``term`` with each free ``x mod m`` replaced by an index ``v`` of range ``m`` that a delta
    factor ``[c*v + L == 0 mod M]`` of the term, ``m | M``, ``c`` a unit, pins to ``x mod m``."""
    if not _has_outside_flat(term, ModResidue):
        return term
    bound = _bound_symbols(term)
    atoms = [
        a for a in term.atoms(ModResidue) if a.args[1].is_Integer and not a.free_symbols & bound
    ]
    if not atoms:
        return term
    deltas = [d for d in map(_delta_base, sp.Mul.make_args(term)) if d is not None]
    mapping: dict[sp.Expr, sp.Expr] = {}
    for atom in atoms:
        x, m = atom.args
        modulus = int(m)
        for d in deltas:
            frequency, big = d.args
            if not big.is_Integer or int(big) % modulus:
                continue
            form = _linear_form(frequency)
            if form is None:
                continue
            for var, coefficient in sorted(form.items(), key=lambda item: str(item[0].name)):
                size = ranges.get(var)
                if var in bound or size is None or sp.expand(size - modulus) != 0:
                    continue
                sign = _unit_sign(sp.Integer(coefficient), m)
                if sign is None:
                    continue
                rest = sp.expand(frequency - coefficient * var)
                if _multiple_of(sp.expand(-sign * rest - x), modulus):
                    mapping[atom] = var
                    break
            if atom in mapping:
                break
    return cast(sp.Expr, term.xreplace(mapping)) if mapping else term


def _rename_indices(expr: sp.Expr) -> sp.Expr:
    """Alpha-rename every bound summation index by how many binders enclose it, numbered past
    every free ``_k`` index.

    Two alpha-equivalent subexpressions are named identically wherever they sit, so sibling
    sums that differ only in their bound name cancel; a counter running over the whole
    expression would give them different names and leave the difference standing.
    """
    if not _has_outside_flat(expr, sp.Sum):
        return expr
    if _named_by_depth(expr):
        return expr

    def stage_a(node: sp.Expr, level: int) -> sp.Expr:
        if isinstance(node, sp.Sum):
            body = node.function
            limits = []
            for position, (var, lower, upper) in enumerate(node.limits):
                temp = sp.Symbol(f"_tmp{level + position}", integer=True, nonnegative=True)
                body = body.xreplace({var: temp})
                limits.append((temp, lower, upper))
            return sp.Sum(stage_a(body, level + len(node.limits)), *limits)
        if not node.args or not node.has(sp.Sum):
            return node
        return cast(sp.Expr, node.func(*(stage_a(arg, level) for arg in node.args)))

    staged = stage_a(expr, 0)
    free = [
        int(str(s.name)[2:])
        for s in staged.free_symbols
        if isinstance(s, sp.Symbol) and re.match(r"^_k[0-9]+$", str(s.name))
    ]
    offset = max(free) + 1 if free else 0
    mapping = {
        symbol: sp.Symbol(f"_k{int(symbol.name[4:]) + offset}", integer=True, nonnegative=True)
        for symbol in staged.atoms(sp.Symbol)
        if str(symbol.name).startswith("_tmp")
    }
    return cast(sp.Expr, staged.xreplace(mapping) if mapping else staged)


def _named_by_depth(expr: sp.Expr) -> bool:
    """Whether every bound index of ``expr`` already has the name :func:`_rename_indices` gives
    it."""
    free = [
        int(str(s.name)[2:])
        for s in expr.free_symbols
        if isinstance(s, sp.Symbol) and re.match(r"^_k[0-9]+$", str(s.name))
    ]
    offset = max(free) + 1 if free else 0

    def named(node: sp.Expr, level: int) -> bool:
        if isinstance(node, sp.Sum):
            for position, (var, _lower, _upper) in enumerate(node.limits):
                wanted = sp.Symbol(f"_k{level + position + offset}", integer=True, nonnegative=True)
                if var != wanted:
                    return False
            return named(node.function, level + len(node.limits))
        if not node.args or not node.has(sp.Sum):
            return True
        return all(named(arg, level) for arg in node.args)

    return named(expr, 0)


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
    """Close Sum_var omega_d^{a*var^2 + b*var + c} * constant for nonzero integer a and
    integer b, or None."""
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
    linear = sp.expand(polynomial.coeff_monomial(var))
    if linear.is_integer is not True:
        return None
    if sign not in (sp.Integer(1), sp.Integer(-1)):
        if not sign.is_Integer or sign == 0:
            return None
        return cast(
            sp.Expr,
            constant
            * _reciprocal_gauss(int(sign), linear, d_expr)
            * sp.exp(2 * sp.pi * sp.I * polynomial.coeff_monomial(1) / d_expr),
        )
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


def _reciprocal_gauss(a: int, b: sp.Expr, d_expr: sp.Expr) -> sp.Expr:
    """Sum_{k<d} omega_d^{a k^2 + b k} by Landsberg-Schaar reciprocity: sqrt(d/(2|a|))
    e^{i pi sgn(a)/4} omega_d^{-b^2/(4a)} Sum_{n<2|a|} e^{-2 pi i (d n^2 + 2 b n)/(4a)}."""
    size = 2 * abs(a)
    dual = sp.Add(
        *(
            sp.exp(-2 * sp.pi * sp.I * sp.expand(d_expr * n * n + 2 * b * n) / (4 * a))
            for n in range(size)
        )
    )
    eighth = (1 + sp.I) / sp.sqrt(2) if a > 0 else (1 - sp.I) / sp.sqrt(2)
    shift = sp.exp(-2 * sp.pi * sp.I * sp.expand(b**2) / (4 * a * d_expr))
    return cast(sp.Expr, sp.sqrt(d_expr) / sp.sqrt(size) * eighth * shift * dual)


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
    terms = _integer_terms(expr) if expr.free_symbols else None
    if terms is not None and all(
        s.is_integer is True
        for mono in terms
        if any(symbol == var for symbol, _ in mono)
        for s, _e in mono
    ):
        return True
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
    if isinstance(expr, ModResidue):
        found = _linear_in(sp.expand(expr.args[0]), var)
        return found is not None and sp.cancel(found[0] * d_expr / expr.args[1]).is_integer is True
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
    None; an integer ``expr`` free of those indices is its own bounds."""
    if not expr.free_symbols & set(ranges) and expr.is_integer is True:
        return sp.expand(expr), sp.expand(expr)
    form: Mapping[sp.Symbol, sp.Expr] | None = _linear_form(expr)
    if form is None:
        form = _linear_over(expr, ranges)
    if form is None:
        form = _signed_linear_over(expr, ranges)
    if form is None:
        return None
    least = greatest = sp.expand(expr - sum((c * v for v, c in form.items()), sp.Integer(0)))
    for var, coefficient in form.items():
        size = ranges.get(var)
        if size is None:
            return None
        if sp.sympify(coefficient).is_nonnegative is True:
            greatest += coefficient * (size - 1)
        else:
            least += coefficient * (size - 1)
    return sp.expand(least), sp.expand(greatest)


def _signed_linear_over(
    expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]
) -> dict[sp.Symbol, sp.Expr] | None:
    """The coefficient of each index of ``ranges`` in ``expr``, each an integer expression free
    of them and of known sign (nonnegative or nonpositive), the rest an integer free of them;
    else None."""
    indices = sorted(expr.free_symbols & set(ranges), key=lambda symbol: str(symbol.name))
    if not indices:
        return None
    form: dict[sp.Symbol, sp.Expr] = {}
    rest = sp.expand(expr)
    for var in indices:
        coefficient = sp.expand(rest).coeff(var, 1)
        if (
            coefficient.free_symbols & set(ranges)
            or coefficient.is_integer is not True
            or (coefficient.is_nonnegative is not True and coefficient.is_nonpositive is not True)
        ):
            return None
        form[var] = coefficient
        rest = sp.expand(rest - coefficient * var)
    if rest.free_symbols & set(ranges) or rest.is_integer is not True:
        return None
    return form


def _linear_over(expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> dict[sp.Symbol, int] | None:
    """The integer coefficient of each index of ``ranges`` in ``expr``, the rest an integer free
    of them; else None."""
    indices = sorted(expr.free_symbols & set(ranges), key=lambda symbol: str(symbol.name))
    if not indices:
        return None
    form: dict[sp.Symbol, int] = {}
    rest = sp.expand(expr)
    for var in indices:
        coefficient = sp.expand(rest).coeff(var, 1)
        if not coefficient.is_Integer:
            return None
        form[var] = int(coefficient)
        rest = sp.expand(rest - coefficient * var)
    if rest.free_symbols & set(ranges) or rest.is_integer is not True:
        return None
    return form


def _extreme(expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr], top: bool) -> sp.Expr | None:
    """The product of index deltas holding the integer-linear ``expr`` at its greatest (``top``)
    or least value over the indices sized in ``ranges``, else None."""
    form = _linear_form(expr)
    if not form:
        return None
    out: sp.Expr = sp.Integer(1)
    for var, coefficient in sorted(form.items(), key=lambda item: str(item[0].name)):
        size = ranges.get(var)
        if size is None:
            return None
        value = sp.expand(size - 1) if (coefficient > 0) == top else sp.Integer(0)
        out *= ModDelta(sp.expand(var - value), size)
    return out


def _radix_moduli(modulus: sp.Expr) -> list[sp.Expr]:
    """The radix candidates of a symbolic ``modulus = a * S`` (``a`` its integer factor, ``S``
    the rest): each non-integer factor, then ``c * S`` for each divisor ``c > 1`` of ``a`` below
    ``a``, then each divisor ``c > 1`` of ``a`` itself, largest first."""
    factors = sp.Mul.make_args(modulus)
    out = [f for f in factors if not f.is_Integer]
    whole = sp.Mul(*out)
    integer = sp.Mul(*(f for f in factors if f.is_Integer))
    if integer.is_Integer and int(integer) > 2:
        out += [sp.Integer(c) * whole for c in sp.divisors(int(integer))[1:-1]]
    if integer.is_Integer and int(integer) > 1:
        out += [sp.Integer(c) for c in reversed(sp.divisors(int(integer))[1:])]
    return out


def _radix_split(delta: ModDelta, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    """``[t*A + B == 0 mod P*t]`` (``t`` from :func:`_radix_moduli`, a symbolic factor or a
    concrete low digit) as ``[B == 0 mod t] * [A == 0 mod P]`` when ``ranges`` keep ``|B| < t``;
    when ``B`` reaches ``t`` (or ``-t``) only at one corner ``E`` of the ranges, as ``([B == 0
    mod t] - E) * [A == 0 mod P] + E * [A +- 1 == 0 mod P]``; when ``B`` reaches ``t`` only at
    a corner ``H`` and ``-t`` only at a corner ``L``, as
    ``([B == 0 mod t] - H - L) * [A == 0 mod P] + H * [A + 1 == 0 mod P] + L * [A - 1 == 0 mod
    P]``; else None."""
    frequency, modulus = delta.args
    if modulus.is_Integer:
        return _radix_split_concrete(frequency, int(modulus), ranges)
    if len(sp.Mul.make_args(modulus)) < 2:
        return None
    for t in _radix_moduli(modulus):
        carry: sp.Expr = sp.Integer(0)
        digit: sp.Expr = sp.Integer(0)
        for piece in sp.Add.make_args(sp.expand(frequency)):
            if sp.denom(sp.cancel(piece / t)) == 1:
                carry += sp.cancel(piece / t)
            else:
                digit += piece
        rest = sp.cancel(modulus / t)
        if digit != 0:
            bounds = _bounds(digit, ranges)
            if bounds is None:
                continue
            least, greatest = bounds
            above = sp.expand(t - greatest)
            below = sp.expand(t + least)
            if above == 0 and below == 0:
                high, low = _extreme(digit, ranges, True), _extreme(digit, ranges, False)
                if high is None or low is None:
                    continue
                return cast(
                    sp.Expr,
                    (ModDelta(sp.expand(digit), t) - high - low) * ModDelta(sp.expand(carry), rest)
                    + high * ModDelta(sp.expand(carry + 1), rest)
                    + low * ModDelta(sp.expand(carry - 1), rest),
                )
            if above.is_positive is not True or below.is_positive is not True:
                top = above == 0 and below.is_positive is True
                if not (top or (below == 0 and above.is_positive is True)):
                    continue
                corner = _extreme(digit, ranges, top)
                if corner is None:
                    continue
                return cast(
                    sp.Expr,
                    (ModDelta(sp.expand(digit), t) - corner) * ModDelta(sp.expand(carry), rest)
                    + corner * ModDelta(sp.expand(carry + (1 if top else -1)), rest),
                )
        return cast(sp.Expr, ModDelta(sp.expand(digit), t) * ModDelta(sp.expand(carry), rest))
    return None


def _radix_split_concrete(
    frequency: sp.Expr, modulus: int, ranges: Mapping[sp.Symbol, sp.Expr]
) -> sp.Expr | None:
    """``[t*A + B == 0 mod P*t]`` as ``[B == 0 mod t] * [A == 0 mod P]`` for a divisor ``t`` of the
    concrete ``modulus`` splitting ``frequency``'s integer terms, when ``ranges`` keep
    ``|B| < t``, else None."""
    if modulus < 4 or not any(modulus % t == 0 for t in range(2, math.isqrt(modulus) + 1)):
        return None
    pieces = sp.Add.make_args(_expand(frequency))
    for t in sp.divisors(modulus)[1:-1]:
        carry: sp.Expr = sp.Integer(0)
        digit: sp.Expr = sp.Integer(0)
        for piece in pieces:
            coefficient, _rest = piece.as_coeff_Mul()
            if coefficient.is_Integer and int(coefficient) % t == 0:
                carry += piece / t
            else:
                digit += piece
        if digit == 0 or carry == 0:
            continue
        bounds = _bounds(digit, ranges)
        if bounds is None:
            continue
        least, greatest = bounds
        if not (sp.expand(t - greatest).is_positive and sp.expand(t + least).is_positive):
            continue
        return cast(
            sp.Expr, ModDelta(sp.expand(digit), t) * ModDelta(sp.expand(carry), modulus // t)
        )
    return None


def _carry_split(delta: ModDelta, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    """``[p*A + B == 0 mod p*q]`` (``p*q`` a power of the prime ``p``, ``B`` the terms of
    coefficient prime to ``p`` and the constant mod ``p``) whose frequency ``ranges`` do not
    bound in ``(-p*q, p*q)``: over bits at ``p = q = 2`` as ``[B == 0 mod 2] * [A + B(B-1)/2 ==
    0 mod 2]``; else as ``Sum_c [B == p*c mod p*q] * [A + c == 0 mod q]`` over the carries
    ``c`` of ``B``, each ``B - p*c`` bounded in ``(-p*q, p*q)``; else None."""
    frequency, m = delta.args
    if not m.is_Integer or int(m) < 4:
        return None
    modulus = int(m)
    pieces = sp.Add.make_args(_expand(frequency))
    whole = _carry_bounds(cast(sp.Expr, sp.Add(*pieces)), ranges)
    if (
        whole is None
        or not (whole[0].is_Integer and whole[1].is_Integer)
        or (whole[0] > -modulus and whole[1] < modulus)
    ):
        return None
    primes = sp.factorint(modulus)
    if len(primes) != 1:
        return None
    p = min(primes)
    q = modulus // p
    carry: sp.Expr = sp.Integer(0)
    digit: sp.Expr = sp.Integer(0)
    for piece in pieces:
        coefficient, rest = piece.as_coeff_Mul()
        if not coefficient.is_Integer:
            return None
        low = int(coefficient) % p
        if low > p // 2:
            low -= p
        digit += low * rest
        carry += (int(coefficient) - low) // p * rest
    if digit == 0 or carry == 0:
        return None
    bounds = _carry_bounds(digit, ranges)
    if bounds is None or not (bounds[0].is_Integer and bounds[1].is_Integer):
        return None
    least, greatest = int(bounds[0]), int(bounds[1])
    symbols = digit.free_symbols
    if p == 2 and all(sp.expand(ranges.get(s, 0) - 2) == 0 for s in symbols):
        half = _bit_half(digit, q)
        if half is not None:
            return cast(sp.Expr, ModDelta(digit, 2) * ModDelta(sp.expand(carry + half), q))
    out: sp.Expr = sp.Integer(0)
    for c in range(-((-least) // p), greatest // p + 1):
        if not (least - p * c > -modulus and greatest - p * c < modulus):
            return None
        out += ModDelta(sp.expand(digit - p * c), modulus) * ModDelta(sp.expand(carry + c), q)
    return out


def _carry_bounds(
    expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, sp.Expr] | None:
    """:func:`_bounds`, or for an integer polynomial over bits of ``ranges`` the sums of its
    negative and of its positive coefficients plus its constant."""
    found = _bounds(expr, ranges)
    if found is not None:
        return found
    symbols = sorted(expr.free_symbols, key=lambda s: str(s.name))
    if not symbols or any(sp.expand(ranges.get(s, 0) - 2) != 0 for s in symbols):
        return None
    terms = poly_terms(expr, tuple(symbols))
    if terms is None or not all(c.is_Integer for _m, c in terms):
        return None
    least = greatest = 0
    for monomial, c in terms:
        if not any(monomial):
            least, greatest = least + int(c), greatest + int(c)
        elif c > 0:
            greatest += int(c)
        else:
            least += int(c)
    return sp.Integer(least), sp.Integer(greatest)


def _bit_power_free(expr: sp.Expr) -> sp.Expr:
    """``expr`` expanded with every power of a symbol read as the symbol."""
    return cast(
        sp.Expr,
        sp.expand(expr).replace(
            lambda node: node.is_Pow and node.base.is_Symbol and node.exp.is_Integer,
            lambda node: node.base,
        ),
    )


def _bit_half(digit: sp.Expr, q: int) -> sp.Expr | None:
    """A bit polynomial equal to ``digit / 2`` mod ``q`` wherever ``digit`` is even: at ``q = 2``
    ``digit (digit - 1) / 2``; else ``(R + par(R)) / 2`` for ``digit = x + R``, ``x`` the
    greatest-named bit read only in its own term and ``par(R) = (1 - (-1)^R) / 2`` over bits; else
    None."""
    if q == 2:
        return _bit_power_free(digit * (digit - 1) / 2)
    terms = sp.Add.make_args(digit)
    singles = [
        t
        for t in terms
        if t.is_Symbol and not any(t in o.free_symbols for o in terms if o is not t)
    ]
    if not singles:
        return None
    x = max(singles, key=lambda s: str(s.name))
    rest = sp.expand(digit - x)
    sign: sp.Expr = sp.Integer(1)
    for term in sp.Add.make_args(rest):
        coefficient, monomial = term.as_coeff_Mul()
        if not coefficient.is_Integer:
            return None
        if int(coefficient) % 2:
            sign = sign * (-1 if monomial == 1 else 1 - 2 * monomial)
    parity = _bit_power_free((1 - sign) / 2)
    return _bit_power_free((rest + parity) / 2)


_RADIX: dict[tuple[object, ...], sp.Expr] = {}
"""Results of :func:`_split_radix_deltas` on one ModDelta, keyed by the delta and the ranges of
its symbols."""


_CARRIES: ContextVar[bool] = ContextVar("split_carries", default=False)


def _split_radix_deltas(
    expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr], top: bool = True
) -> sp.Expr:
    """:func:`_radix_split` on every ModDelta, bound indices sized by their limits, then
    :func:`_carry_split` on each outside every sum; memoised in :data:`_RADIX`."""
    if isinstance(expr, sp.Sum):
        inner = dict(ranges)
        for var, _lower, upper in expr.limits:
            inner[var] = sp.expand(upper + 1)
        return cast(sp.Expr, sp.Sum(_split_radix_deltas(expr.function, inner, False), *expr.limits))
    if isinstance(expr, ModDelta):
        key = (
            expr,
            frozenset((s, r) for s, r in ranges.items() if s in expr.free_symbols),
            top and _CARRIES.get(),
        )
        if key not in _RADIX:
            if len(_RADIX) >= _MEMO_CAP:
                _RADIX.clear()
            found = _radix_split(expr, ranges)
            if found is None and key[2]:
                found = _carry_split(expr, ranges)
            _RADIX[key] = found or expr
        return _RADIX[key]
    if not expr.args or not expr.has(ModDelta):
        return expr
    return cast(sp.Expr, expr.func(*(_split_radix_deltas(arg, ranges, top) for arg in expr.args)))


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
            if var not in frequency.free_symbols:
                continue
            linear = _linear_in(frequency, var)
            if linear is None:
                continue
            coefficient, rest = linear
            if rest.is_integer is not True or _bound_symbols(weight) & rest.free_symbols:
                continue
            sign = _unit_sign(sp.Integer(coefficient), modulus)
            if not _divides(modulus, size):
                wrapped = _wrap_pin(weight, var, coefficient, rest, modulus, size)
                if wrapped is None and sign is not None:
                    known = {**ranges, **{v: sp.expand(u + 1) for v, _l, u in limits if v != var}}
                    wrapped = _window_pin(
                        weight, var, sp.expand(-sign * rest), size, modulus, known
                    )
                if wrapped is not None:
                    return wrapped, _without(limits, index)
                continue
            if not _is_periodic_in(weight, var, modulus):
                if sign is None or sp.expand(modulus - size) != 0:
                    continue
                known = {**ranges, **{v: sp.expand(u + 1) for v, _l, u in limits if v != var}}
                bounds = _bounds(sp.expand(-sign * rest), known)
                if bounds is None or not (
                    bounds[0].is_nonnegative is True
                    and sp.expand(size - 1 - bounds[1]).is_nonnegative is True
                ):
                    wrapped = _wrap_pin(weight, var, coefficient, rest, modulus, size)
                    if wrapped is None and bounds is not None:
                        wrapped = _corner_pin(
                            weight, var, sp.expand(-sign * rest), bounds, size, known
                        )
                    if wrapped is None:
                        continue
                    return wrapped, _without(limits, index)
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


def _wrap_pin(
    weight: sp.Expr,
    var: sp.Symbol,
    coefficient: int,
    rest: sp.Expr,
    modulus: sp.Expr,
    size: sp.Expr,
) -> sp.Expr | None:
    """``Sum_{var < m} [c*var + rest == 0 mod M] * weight`` for concrete ``m | M`` and ``c`` a unit
    mod ``M``: ``weight`` at ``var = (-c*rest) mod m`` times the delta at that value (dropped when
    ``m == M``), else None."""
    if not (modulus.is_Integer and size.is_Integer) or int(modulus) % int(size):
        return None
    sign = _unit_sign(sp.Integer(coefficient), modulus)
    if sign is None:
        return None
    value = ModResidue(sp.expand(-sign * rest), size)
    pinned = weight.xreplace({var: value})
    if int(modulus) == int(size):
        return cast(sp.Expr, pinned)
    return cast(sp.Expr, pinned * ModDelta(sp.expand(coefficient * value + rest), modulus))


def _nonnegative(expr: sp.Expr) -> bool:
    """True when sympy finds ``expr`` nonnegative, or ``expr`` with each power ``b^e`` (``b >= 2``
    an integer, ``e`` symbolic and nonnegative) read as ``1 + y``, ``y >= 0``, is a polynomial
    in those ``y`` with nonnegative coefficients."""
    expr = sp.expand(expr)
    if expr.is_nonnegative is True:
        return True
    powers = [
        p
        for p in expr.atoms(sp.Pow)
        if p.base.is_Integer and int(p.base) >= 2 and p.exp.free_symbols
        if p.exp.is_nonnegative is True
    ]
    if not powers:
        return False
    taken = {str(s.name) for s in expr.free_symbols}
    names = (f"_y{n}" for n in itertools.count() if f"_y{n}" not in taken)
    ys = {p: sp.Symbol(next(names), nonnegative=True) for p in powers}
    relaxed = sp.expand(expr.xreplace({p: 1 + y for p, y in ys.items()}))
    if relaxed.free_symbols - set(ys.values()):
        return relaxed.is_nonnegative is True
    if relaxed.is_Number:
        return bool(relaxed >= 0)
    return all(c >= 0 for c in sp.Poly(relaxed, *ys.values()).coeffs())


def _window_pin(
    weight: sp.Expr,
    var: sp.Symbol,
    value: sp.Expr,
    size: sp.Expr,
    modulus: sp.Expr,
    ranges: Mapping[sp.Symbol, sp.Expr],
) -> sp.Expr | None:
    """``Sum_{var < size} [var == value mod M] * weight`` for ``size <= M`` when the ranges hold
    ``value`` inside one window: ``[0, size)`` gives ``weight`` at ``value``, ``[-M, size - M)``
    gives it at ``value + M``, ``[size - M, 0)`` or ``[size, M)`` gives zero; else None."""
    if sp.expand(modulus - size).is_nonnegative is not True:
        return None
    bounds = _bounds(value, ranges)
    if bounds is None:
        return None
    least, greatest = bounds

    def inside(lo: sp.Expr, hi: sp.Expr) -> bool:
        return _nonnegative(least - lo) and _nonnegative(hi - 1 - greatest)

    if inside(sp.Integer(0), size):
        return cast(sp.Expr, weight.xreplace({var: value}))
    if inside(sp.expand(-modulus), sp.expand(size - modulus)):
        return cast(sp.Expr, weight.xreplace({var: sp.expand(value + modulus)}))
    if inside(sp.expand(size - modulus), sp.Integer(0)) or inside(size, modulus):
        return sp.Integer(0)
    return None


def _corner_pin(
    weight: sp.Expr,
    var: sp.Symbol,
    value: sp.Expr,
    bounds: tuple[sp.Expr, sp.Expr],
    size: sp.Expr,
    ranges: Mapping[sp.Symbol, sp.Expr],
) -> sp.Expr | None:
    """``Sum_{var < size} [var == value mod size] * weight`` for ``value`` in ``[-1, size)`` or
    ``[0, size]``, the out-of-range end one corner ``E`` of the ranges: ``(1 - E) * weight`` at
    ``value`` plus ``E * weight`` at ``value`` moved by ``size``; else None."""
    least, greatest = bounds
    if sp.expand(least + 1) == 0 and sp.expand(size - 1 - greatest).is_nonnegative is True:
        top, shift = False, size
    elif least.is_nonnegative is True and sp.expand(greatest - size) == 0:
        top, shift = True, -size
    else:
        return None
    if not value.free_symbols:
        return cast(sp.Expr, weight.xreplace({var: sp.expand(value + shift)}))
    corner = _extreme(value, ranges, top)
    if corner is None:
        return None
    return cast(
        sp.Expr,
        (1 - corner) * weight.xreplace({var: value})
        + corner * weight.xreplace({var: sp.expand(value + shift)}),
    )


def _carried_frequency(term: sp.Expr, frequency: sp.Expr) -> tuple[sp.Expr, int] | None:
    """``(X, t)`` with ``frequency == X/t``, ``X`` an integer polynomial in integer symbols,
    when a factor ``[Y == 0 mod t']`` of ``term`` with ``t | t'`` makes ``X == l*Y mod t`` for a
    unit ``l`` (so ``X/t`` is an integer wherever the term is nonzero), else None."""
    if frequency.has(ModResidue) or term.has(ModResidue):
        (frozen_term, frozen_frequency), back = _freeze(term, frequency)
        found = _carried_frequency(frozen_term, frozen_frequency)
        if found is None:
            return None
        return cast(sp.Expr, found[0].xreplace(back)), found[1]
    numerator, denominator = sp.fraction(sp.together(sp.expand(frequency)))
    if not denominator.is_Integer or int(denominator) < 2:
        return None
    t = int(denominator)
    numerator = sp.expand(numerator)
    symbols = sorted(numerator.free_symbols, key=lambda x: str(x.name))
    if not symbols or any(x.is_integer is not True for x in symbols):
        return None
    numerator_terms = poly_terms(numerator, tuple(symbols))
    if numerator_terms is None or not all(c.is_Integer for _m, c in numerator_terms):
        return None
    for factor in sp.Mul.make_args(term):
        delta = _delta_base(factor)
        if delta is None:
            continue
        carry, modulus = delta.args
        if not modulus.is_Integer or int(modulus) % t:
            continue
        solved = _unit_multiple(numerator, carry, tuple(symbols), t)
        if solved is not None:
            if solved:
                return numerator, t
            continue
        for unit in range(1, t):
            if math.gcd(unit, t) != 1:
                continue
            rest = poly_terms(sp.expand(numerator - unit * carry), tuple(symbols))
            if rest is None:
                break
            if all(c.is_Integer and int(c) % t == 0 for _m, c in rest):
                return numerator, t
    return None


def _unit_multiple(
    x: sp.Expr, y: sp.Expr, symbols: tuple[sp.Symbol, ...], modulus: int
) -> bool | None:
    """Whether ``x == l*y`` coefficientwise mod ``modulus`` for a unit ``l``, the congruences
    of the monomials solved for ``l`` one after another; None when ``x`` or ``y`` has a
    coefficient over ``symbols`` that is not an integer."""
    if symbols:
        x_terms, y_terms = poly_terms(x, symbols), poly_terms(y, symbols)
    else:
        x_terms, y_terms = [((), x)], [((), y)]
    if x_terms is None or y_terms is None:
        return None
    if not all(c.is_Integer for _m, c in (*x_terms, *y_terms)):
        return None
    xs = {m: int(c) for m, c in x_terms}
    ys = {m: int(c) for m, c in y_terms}
    residue, step = 0, 1
    for m in xs.keys() | ys.keys():
        a = ys.get(m, 0) * step % modulus
        b = (xs.get(m, 0) - ys.get(m, 0) * residue) % modulus
        g = math.gcd(a, modulus)
        if b % g:
            return False
        n = modulus // g
        if n > 1:
            residue += step * ((b // g) * pow(a // g, -1, n) % n)
            step *= n
    return math.gcd(residue % step, step) == 1


def _character_step(term: sp.Expr, limits: list[_Limit]) -> tuple[sp.Expr, list[_Limit]] | None:
    """Close one index read only through omega powers:
    Sum_u omega_d^{u*j + o} = omega_d^o * d * [d | j]; a fractional ``j = X/t`` made integral by a
    delta factor of the term gives ``omega_d^o * d * [t*d | X]``."""
    for index, (var, _lower, upper) in enumerate(limits):
        size = sp.expand(upper + 1)
        split = _split_term(term, var, size)
        if split is None:
            continue
        constant, frequency, offset = split
        shift = sp.exp(2 * sp.pi * sp.I * offset / size)
        if frequency.is_integer is not True:
            carried = _carried_frequency(term, frequency)
            if carried is None or not size.is_Integer:
                continue
            numerator, t = carried
            delta = ModDelta(numerator, sp.Integer(t) * size)
            return constant * shift * size * delta, _without(limits, index)
        return constant * shift * size * ModDelta(frequency, size), _without(limits, index)
    return None


def _gauss_step(term: sp.Expr, limits: list[_Limit]) -> tuple[sp.Expr, list[_Limit]] | None:
    """Close one index through :func:`_gauss_term`."""
    for index, (var, _lower, upper) in enumerate(limits):
        closed = _gauss_term(term, var, sp.expand(upper + 1))
        if closed is not None:
            return closed, _without(limits, index)
    return None


_REDUCED: dict[tuple[object, ...], tuple[sp.Expr, bool]] = {}
"""Results of :func:`_reduce_term` keyed by (term, limits, ranges in order)."""

_REDUCED_CAP = 1 << 12
"""The most entries :data:`_REDUCED` holds; a full table is emptied before the next entry."""


def _reduce_term(
    term: sp.Expr, limits: list[_Limit], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, bool]:
    """``Sum(term, *limits)`` with every closable index closed, and whether any was; memoised
    in :data:`_REDUCED`."""
    key = (term, tuple(limits), tuple(ranges.items()))
    hit = _REDUCED.get(key)
    if hit is None:
        hit = _reduce_term_once(term, limits, ranges)
        if len(_REDUCED) >= _REDUCED_CAP:
            _REDUCED.clear()
        _REDUCED[key] = hit
    return hit


def _reduce_term_once(
    term: sp.Expr, limits: list[_Limit], ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, bool]:
    """:func:`_reduce_term` without the table."""
    if not limits:
        return term, False
    free = term.free_symbols
    unused = [limit for limit in limits if limit[0] not in free]
    if unused:
        size = sp.Mul(*(sp.expand(upper + 1) for _var, _lower, upper in unused))
        used = [limit for limit in limits if limit[0] in free]
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


_ENUMERATE: ContextVar[bool] = ContextVar("enumerate_concrete", default=True)


def _close_sum(node: sp.Sum, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    """Close what the identities reach of one Sum node, else enumerate one concrete index when
    enumeration is on."""
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
    if not concrete or not _ENUMERATE.get():
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
    terms = _integer_terms(expr)
    if terms is not None:
        if any(sum(e for _s, e in mono) > 1 for mono in terms):
            return None
        return {symbol: terms.get(frozenset({(symbol, 1)}), 0) for symbol in symbols}
    found = poly_terms(expr, tuple(symbols))
    if found is None:
        return None
    if max(sum(m) for m, _c in found) > 1 or not all(c.is_Integer for _m, c in found):
        return None
    linear = {m.index(1): int(c) for m, c in found if sum(m) == 1}
    return {symbol: linear.get(i, 0) for i, symbol in enumerate(symbols)}


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


def _trim_mod(expr: sp.Expr, modulus: sp.Expr) -> sp.Expr:
    """``expr`` with each integer coefficient ``c`` of ``|c| >= modulus`` replaced by
    ``sign(c) (|c| mod modulus)`` at an integer ``modulus``; ``expr`` itself otherwise."""
    if not modulus.is_Integer or not expr.free_symbols:
        return expr
    m = int(modulus)
    terms = _integer_terms(expr)
    if terms is None or all(abs(c) < m for c in terms.values()):
        return expr
    parts = []
    for mono, c in terms.items():
        kept = (abs(c) % m) * (1 if c > 0 else -1)
        if kept:
            parts.append(kept * sp.Mul(*(s**e for s, e in mono)))
    return cast(sp.Expr, sp.Add(*parts))


def _echelon(
    rows: list[sp.Expr], modulus: sp.Expr
) -> tuple[dict[sp.Symbol, sp.Expr], list[sp.Expr]]:
    """Gauss-Jordan reduction of ``rows`` mod ``modulus`` over unit pivots, as (pivots,
    residual rows); at an integer modulus every substituted row and pivot value is taken
    through :func:`_trim_mod`."""
    pivots: dict[sp.Symbol, sp.Expr] = {}
    pending = list(rows)
    while True:
        residual: list[sp.Expr] = []
        progressed = False
        for row in pending:
            row = _trim_mod(sp.expand(row.xreplace(pivots)), modulus)
            found = _pivot(row, modulus)
            if found is None:
                residual.append(row)
                continue
            var, value = found
            pivots = {
                s: _trim_mod(sp.expand(v.xreplace({var: value})), modulus)
                for s, v in pivots.items()
            }
            pivots[var] = value
            progressed = True
        pending = residual
        if not progressed:
            return pivots, pending


def _implies(big: ModDelta, small: ModDelta) -> bool:
    """Whether ``[X == 0 mod M]`` forces ``[Y == 0 mod t]``: integer moduli with ``t | M`` and
    ``X == l*Y mod t`` coefficientwise for a unit ``l``."""
    (x, m), (y, t) = big.args, small.args
    if not (m.is_Integer and t.is_Integer) or int(t) < 2 or int(m) % int(t) or m == t:
        return False
    if x.has(ModResidue) or y.has(ModResidue):
        (x, y), _back = _freeze(x, y)
    symbols = sorted(x.free_symbols | y.free_symbols, key=lambda v: str(v.name))
    if any(v.is_integer is not True for v in symbols):
        return False
    modulus = int(t)
    solved = _unit_multiple(x, y, tuple(symbols), modulus)
    if solved is not None:
        return solved
    for unit in range(1, modulus):
        if math.gcd(unit, modulus) != 1:
            continue
        difference = sp.expand(x - unit * y)
        found = poly_terms(difference, tuple(symbols)) if symbols else [((), difference)]
        if found is None:
            return False
        coefficients = [c for _m, c in found]
        if all(c.is_Integer and int(c) % modulus == 0 for c in coefficients):
            return True
    return False


def _drop_implied(term: sp.Expr) -> sp.Expr:
    """``term`` without each ModDelta factor another of its ModDelta factors implies."""
    factors = list(sp.Mul.make_args(term))
    deltas = [(i, _delta_base(f)) for i, f in enumerate(factors)]
    present = [(i, d) for i, d in deltas if d is not None]
    dropped: set[int] = set()
    for i, small in present:
        if any(j != i and j not in dropped and _implies(big, small) for j, big in present):
            dropped.add(i)
    if not dropped:
        return term
    return cast(sp.Expr, sp.Mul(*(f for i, f in enumerate(factors) if i not in dropped)))


def _exact_pivot(
    var: sp.Symbol, value: sp.Expr, modulus: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]
) -> bool:
    """Whether ``var == value mod modulus`` forces ``var == value``: ``ranges`` bound
    ``var - value`` in ``(-modulus, modulus)``, or ``var`` ranges over ``modulus`` and ``ranges``
    bound ``value`` in ``[0, modulus)``."""
    size = ranges.get(var)
    if size is None:
        return False
    span = _bounds(sp.expand(var - value), ranges)
    if (
        span is not None
        and sp.expand(modulus + span[0]).is_positive is True
        and sp.expand(modulus - span[1]).is_positive is True
    ):
        return True
    if sp.expand(size - modulus) != 0:
        return False
    found = _bounds(value, ranges)
    return (
        found is not None
        and found[0].is_nonnegative is True
        and sp.expand(modulus - 1 - found[1]).is_nonnegative is True
    )


def _never(delta: ModDelta, ranges: Mapping[sp.Symbol, sp.Expr]) -> bool:
    """Whether ``ranges`` hold the linear frequency of ``delta`` strictly between two consecutive
    multiples of its modulus (0 and the modulus, or minus the modulus and 0)."""
    frequency, modulus = delta.args
    if not ranges or frequency.free_symbols - set(ranges):
        return False
    bounds = _bounds(frequency, ranges)
    if bounds is None:
        return False
    least, greatest = bounds
    if least.is_positive is True and sp.expand(modulus - greatest).is_positive is True:
        return True
    return greatest.is_negative is True and sp.expand(modulus + least).is_positive is True


class _FreeSymbols:
    """``expr.free_symbols`` remembered per expression object."""

    def __init__(self) -> None:
        self._seen: dict[int, tuple[sp.Expr, set[sp.Symbol]]] = {}

    def __call__(self, expr: sp.Expr) -> set[sp.Symbol]:
        hit = self._seen.get(id(expr))
        if hit is None or hit[0] is not expr:
            hit = (expr, expr.free_symbols)
            self._seen[id(expr)] = hit
        return hit[1]


def _xreplace_symbol(expr: sp.Expr, var: sp.Symbol, value: sp.Expr, frees: _FreeSymbols) -> sp.Expr:
    """``expr.xreplace({var: value})``, each ModDelta factor with an integer modulus and an integer
    polynomial frequency rebuilt from its substituted :func:`_integer_terms`."""
    if not expr.is_Mul:
        return _xreplace_factor(expr, var, value, frees)
    if not any(var in frees(f) for f in expr.args):
        return expr
    return cast(sp.Expr, expr.func(*(_xreplace_factor(f, var, value, frees) for f in expr.args)))


def _xreplace_factor(
    factor: sp.Expr, var: sp.Symbol, value: sp.Expr, frees: _FreeSymbols
) -> sp.Expr:
    """``factor.xreplace({var: value})`` for one factor of :func:`_xreplace_symbol`."""
    if var not in frees(factor):
        return factor
    if isinstance(factor, ModDelta) and factor.args[1].is_Integer and factor.args[1] != 1:
        terms = _integer_terms(factor.args[0])
        lift = _integer_terms(value)
        if terms is not None and lift is not None:
            out = _delta_of_terms(_substitute_terms(terms, var, lift), factor.args[1])
            if out is not None:
                return out
    return cast(sp.Expr, factor.xreplace({var: value}))


def _canonical_deltas(term: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr] | None = None) -> sp.Expr:
    """A product with its implied ModDelta factors dropped and its linear ModDelta factors in
    echelon form, each pivot substituted into the rest wherever that rest is periodic in it, or
    everywhere when ``ranges`` force the pivot exactly."""
    ranges = ranges or {}
    if any(_never(d, ranges) for d in map(_delta_base, sp.Mul.make_args(term)) if d is not None):
        return sp.Integer(0)
    term = _drop_implied(term)
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
    frees = _FreeSymbols()
    summed = bool(_bound_symbols(rest))
    emitted: list[sp.Expr] = []
    substitutions: list[tuple[sp.Symbol, sp.Expr, sp.Expr, bool, sp.Expr]] = []
    for modulus in sorted(by_modulus, key=sp.default_sort_key):
        pivots, residual = _echelon(by_modulus[modulus], modulus)
        for var in sorted(pivots, key=lambda symbol: str(symbol.name)):
            value = pivots[var]
            exact = _exact_pivot(var, value, modulus, ranges)
            factors = sp.Mul.make_args(rest)
            bound = _bound_symbols(rest) if summed else set()
            if (
                any(var in frees(f) for f in factors)
                and var not in bound
                and not bound & value.free_symbols
                and (
                    exact
                    or all(var not in frees(f) or _is_periodic_in(f, var, modulus) for f in factors)
                )
            ):
                rest = _xreplace_symbol(rest, var, value, frees)
            defining = ModDelta(var - value, modulus)
            emitted.append(defining)
            substitutions.append((var, value, modulus, exact, defining))
        emitted.extend(ModDelta(row, modulus) for row in residual)
    if len(by_modulus) > 1:
        for var, value, modulus, exact, defining in substitutions:
            if defining not in emitted:
                continue
            emitted = [
                _xreplace_symbol(delta, var, value, frees)
                if isinstance(delta, ModDelta)
                and delta.args[1] != modulus
                and var in frees(delta)
                and (exact or _is_periodic_in(delta, var, modulus))
                else delta
                for delta in emitted
            ]
        return _drop_implied(cast(sp.Expr, rest * sp.Mul(*emitted)))
    return cast(sp.Expr, rest * sp.Mul(*emitted))


def _residue_pivots(term: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """``term`` with each free ``x mod m`` replaced by a residue-free ``v`` in the other factors
    where a delta factor ``[x mod m - v == 0 mod M]`` holds exactly: ``ranges``, with each
    ``x mod m`` in ``[0, m)``, bound ``x mod m - v`` in ``(-M, M)``."""
    if not ranges or not _has_outside_flat(term, ModResidue):
        return term
    bound = _bound_symbols(term)
    factors = list(sp.Mul.make_args(term))
    for position, factor in enumerate(factors):
        delta = _delta_base(factor)
        if delta is None or not delta.has(ModResidue):
            continue
        frequency, modulus = delta.args
        atoms = sorted(frequency.atoms(ModResidue), key=sp.default_sort_key)
        if any(not a.args[1].is_Integer or a.free_symbols & bound for a in atoms):
            continue
        frozen: dict[sp.Expr, sp.Expr] = {}
        for a in atoms:
            frozen[a] = _fresh_index(cast(sp.Expr, sp.Mul(term, *frozen.values())))
        local = {**ranges, **{s: a.args[1] for a, s in frozen.items()}}
        flat = sp.expand(frequency.xreplace(frozen))
        span = _bounds(flat, local) if flat.free_symbols <= set(local) else None
        if span is None or not (
            sp.expand(modulus + span[0]).is_positive is True
            and sp.expand(modulus - span[1]).is_positive is True
        ):
            continue
        for atom in atoms:
            coefficient = flat.coeff(frozen[atom], 1)
            if coefficient not in (1, -1):
                continue
            value = sp.expand(-coefficient * (flat - coefficient * frozen[atom]))
            value = value.xreplace({s: a for a, s in frozen.items()})
            if value.has(ModResidue):
                continue
            others = [
                f.xreplace({atom: value}) if i != position else f for i, f in enumerate(factors)
            ]
            if others != factors:
                return _residue_pivots(cast(sp.Expr, sp.Mul(*others)), ranges)
    return term


def _canonical_top(expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """:func:`_pin_residues` and :func:`_canonical_deltas` over every top-level term of ``expr``."""
    collapsed = (
        expr.replace(_is_delta_power, lambda node: node.base)
        if _outside_flat(expr, _is_delta_power)
        else expr
    )
    terms = sp.Add.make_args(_expand(collapsed))
    pinned = (
        _residue_pivots(_collapse_residues(_pin_residues(t, ranges), ranges), ranges) for t in terms
    )
    if _CARRIES.get():
        pinned = (_symbolic_pivots(_parity_echelon(t, ranges), ranges) for t in pinned)
    return _expand(cast(sp.Expr, sp.Add(*(_canonical_deltas(t, ranges) for t in pinned))))


def _signed_bounds(
    expr: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]
) -> tuple[sp.Expr, sp.Expr] | None:
    """``(least, greatest)`` of ``expr`` linear over the indices of ``ranges`` with coefficients
    free of them of known sign, the rest free of them; else None."""
    rest = sp.expand(expr)
    least = greatest = sp.Integer(0)
    for var in sorted(rest.free_symbols & set(ranges), key=lambda symbol: str(symbol.name)):
        coefficient = rest.coeff(var, 1)
        if coefficient.free_symbols & set(ranges):
            return None
        rest = sp.expand(rest - coefficient * var)
        if var in rest.free_symbols:
            return None
        top = coefficient * (ranges[var] - 1)
        if coefficient.is_nonnegative is True:
            greatest += top
        elif coefficient.is_nonpositive is True:
            least += top
        else:
            return None
    if rest.is_integer is not True:
        return None
    return sp.expand(least + rest), sp.expand(greatest + rest)


def _symbolic_pivots(term: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """``term`` with, per delta factor ``[c*v + r == 0 mod M]`` (``c = +-1``, ``r`` free of
    ``v``, a coefficient not an integer) that ``ranges`` force exactly (:func:`_signed_bounds` of
    ``c*v + r`` inside ``(-M, M)``), the first such ``v`` by name replaced by ``-c*r`` in the
    other factors."""
    if not ranges:
        return term
    bound = _bound_symbols(term)
    factors = list(sp.Mul.make_args(term))
    for position, factor in enumerate(factors):
        delta = _delta_base(factor)
        if delta is None or _linear_form(delta.args[0]) is not None:
            continue
        frequency, modulus = delta.args
        if frequency.free_symbols & bound or not frequency.free_symbols & set(ranges):
            continue
        span = _signed_bounds(frequency, ranges)
        if span is None or not (
            sp.expand(modulus + span[0]).is_positive is True
            and sp.expand(modulus - span[1]).is_positive is True
        ):
            continue
        flat = sp.expand(frequency)
        for var in sorted(flat.free_symbols & set(ranges), key=lambda symbol: str(symbol.name)):
            coefficient = flat.coeff(var, 1)
            if coefficient not in (1, -1):
                continue
            value = sp.expand(-coefficient * (flat - coefficient * var))
            others = [
                f.xreplace({var: value}) if i != position else f for i, f in enumerate(factors)
            ]
            if others != factors:
                return _symbolic_pivots(cast(sp.Expr, sp.Mul(*others)), ranges)
            break
    return term


def _parity_echelon(term: sp.Expr, ranges: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
    """``term`` with its parity deltas over bits of ``ranges``, at least one non-linear, in
    Gauss-Jordan form: each pivot the greatest-named bit read only in its own linear term,
    substituted (powers of bits read as the bits) into every other such delta."""
    rows: list[sp.Expr] = []
    others: list[sp.Expr] = []
    for factor in sp.Mul.make_args(term):
        if (
            isinstance(factor, ModDelta)
            and factor.args[1] == 2
            and factor.free_symbols
            and all(sp.expand(ranges.get(s, 0) - 2) == 0 for s in factor.free_symbols)
        ):
            rows.append(factor.args[0])
        else:
            others.append(factor)
    if len(rows) < 2 or all(_linear_form(r) is not None for r in rows):
        return term

    def reduced(expr: sp.Expr, values: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
        return _trim_mod(_bit_power_free(expr.xreplace(dict(values))), sp.Integer(2))

    pivots: dict[sp.Symbol, sp.Expr] = {}
    residual: list[sp.Expr] = []
    for row in sorted(rows, key=sp.default_sort_key):
        row = reduced(row, pivots)
        terms = sp.Add.make_args(row)
        singles = [
            t
            for t in terms
            if t.is_Symbol and not any(t in o.free_symbols for o in terms if o is not t)
        ]
        if not singles:
            residual.append(row)
            continue
        x = max(singles, key=lambda s: str(s.name))
        value = sp.expand(row - x)
        pivots = {s: reduced(v, {x: value}) for s, v in pivots.items()}
        pivots[x] = value
    deltas = [ModDelta(x + v, 2) for x, v in pivots.items()]
    deltas += [ModDelta(reduced(r, pivots), 2) for r in residual]
    return cast(sp.Expr, sp.Mul(*others, *deltas))


def _is_delta_power(node: sp.Expr) -> bool:
    """Whether ``node`` is a positive integer power of a ModDelta."""
    return _delta_base(node) is not None and bool(node.is_Pow)


def _pull_constants(expr: sp.Expr) -> sp.Expr:
    """Move every factor independent of a sum's bound indices outside that sum.

    Canonical placement: an index-free factor always sits outside, so two expressions
    differing only in whether a constant was carried into the summand compare equal.
    """
    if expr.args and not _flat_delta(expr):
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
    if not symbols:
        return 2 * sp.pi * sp.I * (q % 1) if q.is_Rational else arg
    found = poly_terms(q, tuple(symbols))
    if found is None or not all(coefficient.is_Rational for _m, coefficient in found):
        return arg
    reduced = sp.Add(
        *(
            (coefficient % 1) * sp.Mul(*(s**e for s, e in zip(symbols, monomial) if e))
            for monomial, coefficient in found
        )
    )
    return 2 * sp.pi * sp.I * reduced


def _reduce_integer_phases(expr: sp.Expr) -> sp.Expr:
    """Apply :func:`_reduce_integer_phase` to every exponential's argument in ``expr``."""
    if not _has_outside_flat(expr, sp.exp):
        return expr
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
    if not _has_outside_flat(expr, sp.Sum):
        return expr, False
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
    for node in _delta_atoms(expr):
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
    """The fixpoint of residue collapsing (:func:`_collapse_residues`, inside sums too), radix
    splitting, :func:`_close_all` and canonicalisation, then residue cases split."""
    while True:
        budget[0] -= 1
        if budget[0] < 0:
            raise ScalarBudgetError(
                "simplification exceeded its step budget; the expression is left unsimplified"
            )
        expr = _collapse_residues(expr, ranges)
        split = _split_radix_deltas(expr, ranges)
        radix_changed = split != expr
        expr, changed = _close_all(split, budget, ranges)
        changed = changed or radix_changed
        expr = _canonical_top(_reduce_integer_phases(expr), ranges)
        expr = _pull_constants(sp.powsimp(_expand(expr), force=True))
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
        enumerate_concrete: bool = True,
        split_carries: bool = False,
    ) -> Scalar:
        """Close every index sum the identities reach and put the remaining deltas in echelon form;
        ``ranges`` sizes free indices; ``enumerate_concrete`` False leaves a stuck concrete sum
        unexpanded; ``split_carries`` applies :func:`_carry_split` outside every sum; memoised
        in :data:`_SIMPLIFIED`."""
        known = {_index_symbol(name): dim.to_sympy() for name, dim in (ranges or {}).items()}
        key = (self._expr, max_steps, tuple(known.items()), enumerate_concrete, split_carries)
        hit = _SIMPLIFIED.get(key)
        if isinstance(hit, Scalar):
            return hit
        if isinstance(hit, str):
            raise ScalarBudgetError(hit)
        token = _ENUMERATE.set(enumerate_concrete)
        carries = _CARRIES.set(split_carries)
        try:
            simplified = _simplify_expr(_rename_indices(self._expr), [max_steps], known)
        except ScalarBudgetError as exc:
            self._remember(key, str(exc))
            raise
        finally:
            _CARRIES.reset(carries)
            _ENUMERATE.reset(token)
        result = Scalar(_rename_indices(simplified))
        self._remember(key, result)
        return result

    @staticmethod
    def _remember(key: tuple[object, ...], value: object) -> None:
        """Store ``value`` in :data:`_SIMPLIFIED` under ``key``."""
        if len(_SIMPLIFIED) >= _MEMO_CAP:
            _SIMPLIFIED.clear()
        _SIMPLIFIED[key] = value

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
    def is_provably_nonzero(self) -> bool:
        """True iff this concrete scalar evaluates to a nonzero number at 30 significant digits
        under ``evalf(strict=True)``; False when symbolic or when that accuracy is not reached."""
        if not self.is_concrete:
            return False
        try:
            value = sp.N(self._expr, 30, strict=True)
        except (sp.core.evalf.PrecisionExhausted, TypeError, ValueError):
            return False
        return bool(value != 0)

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
        """``Scalar(expr)``; past the interpreter's integer-to-string digit limit, each rational
        too long to print written as ``_int_<p digest>_<q digest>`` (:func:`_digit_free`)."""
        try:
            return f"Scalar({self._expr})"
        except ValueError:
            return f"Scalar({_digit_free(self._expr)})"

    def __str__(self) -> str:
        return str(self._expr)
