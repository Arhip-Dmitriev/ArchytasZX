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

"""Symbolic phase values and phase vectors tied to a wire dimension, with spider-fusion addition.

Mathematical model. A qudit Z spider with phase vector (alpha_1, ..., alpha_{d-1})
denotes Sum_{k=0}^{d-1} e^{i alpha_k} |k>^{tensor n} <k|^{tensor m}, with alpha_0 == 0
fixed as gauge (index 0 is not a free phase slot; see :class:`PhaseVector`). Fusing two
same-color spiders adds their phase vectors componentwise -- this is why :class:`Phase`
and :class:`PhaseVector` both define ``+`` and why that operator is exactly spider-fusion
semantics, not a generic convenience.

A :class:`Phase` is stored internally as a sympy expression in *turns*, i.e. fractions of
a full 2*pi rotation, so that ``angle = 2*pi*expr``. This makes a root-of-unity index
natural (``omega_d^j`` corresponds to ``expr = j/d``) and makes "reduce mod 2*pi" into
"reduce mod 1", a purely rational operation when the expression is a concrete rational.

Normalization is deliberately narrow: a turns-expression is reduced modulo one turn only
when it is a concrete sympy Rational. An expression such as ``j/d`` with symbolic ``d``,
or a bare free symbol ``theta``, is left untouched -- there is no sound way to reduce it
mod 1 without knowing its value, and this module never calls a general simplifier and
hopes. Consequently equality is sound but incomplete: two phases that compare equal after
normalization are mathematically equal, but two mathematically equal phases may compare
unequal if this module cannot verify it (e.g. two symbolic expressions that happen to
differ by an integer number of turns). Reporting "not equal" for something unverified is
always preferred here to reporting "equal" for something unverified -- the same contract
:meth:`qufzx.algebra.dimension.Dim.unify` follows.

Phase 9 note: the character-sum identity Sum_{k=0}^{d-1} omega_d^{jk} = d * [j == 0 mod d]
is explicitly out of scope here. It belongs to the scalar simplifier arriving in Phase 9
and must not be anticipated by this module or by :mod:`qufzx.algebra.scalar`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim


class PhaseError(Exception):
    """Base class for all errors raised by this module."""


class PhaseDomainError(PhaseError):
    """A value is outside the mathematical domain of phases or phase-vector indices.

    Raised for non-concrete numeric evaluation attempts, out-of-range indices at a
    concrete dimension, and gauge violations at index 0.
    """


class PhaseGrammarError(PhaseError):
    """A value falls outside the grammar this module accepts.

    Raised for non-exact numeric input (e.g. floats where an exact turns value is
    required) and other malformed constructor arguments.
    """


PhaseSymbolKey = Union[str, "Phase"]
PhaseSubstituteValue = Union[int, "sp.Rational", "Phase"]


def _turns_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def _normalize_turns(expr: sp.Expr) -> sp.Expr:
    """Reduce a turns-expression mod one turn, but only when it is a concrete Rational.

    This is the one and only normalization step this module performs on a Phase's
    internal expression. Anything that is not literally a sympy Rational (or Integer)
    after simplification of numeric structure -- in particular any expression carrying
    a free symbol, even something as simple as ``j/d`` with symbolic ``d`` -- is
    returned unchanged.

    Before any of that: an expression containing an ``sp.Float`` atom anywhere in its
    tree (not just at the top level) is rejected outright, since this module tracks
    phases exactly, never approximately.
    """
    if expr.atoms(sp.Float):
        raise PhaseGrammarError(
            f"Phase requires an exact turns-expression, got a float in {expr!r}"
        )
    if expr.free_symbols:
        return expr
    if expr.is_Rational:
        return expr - sp.floor(expr)
    return expr


class Phase:
    """An immutable, hashable angle, stored as an exact fraction of a full turn.

    The angle is ``2*pi * turns``, where ``turns`` is a sympy expression that may be a
    concrete Rational, a free symbol, or a compound expression such as ``j/d`` built
    from a root-of-unity index and a (possibly symbolic) :class:`~qufzx.algebra.dimension.Dim`.
    See the module docstring for the normalization and equality contract.
    """

    __slots__ = ("_turns",)
    _turns: sp.Expr

    def __init__(self, turns: sp.Expr) -> None:
        """Build a Phase directly from a sympy turns-expression. Prefer the named constructors."""
        self._turns = _normalize_turns(sp.sympify(turns))

    @classmethod
    def zero(cls) -> Phase:
        """The gauge-fixed zero phase, angle 0."""
        return cls(sp.Integer(0))

    @classmethod
    def turns(cls, value: sp.Rational | int) -> Phase:
        """Build a phase from an exact fraction of a full turn (angle = 2*pi*value).

        ``value`` must be an exact int or sympy Rational; floats are rejected because
        this module tracks phases exactly, never approximately.
        """
        if isinstance(value, bool):
            raise PhaseGrammarError(f"turns() does not accept bool, got {value!r}")
        if isinstance(value, int):
            return cls(sp.Integer(value))
        if isinstance(value, sp.Rational):
            return cls(value)
        raise PhaseGrammarError(
            f"turns() requires an int or sympy Rational, got {type(value).__name__}"
        )

    @classmethod
    def root_of_unity(cls, index: int | sp.Expr, dim: Dim) -> Phase:
        """Build the phase of omega_d^index, i.e. angle 2*pi*index/d.

        ``dim`` is a :class:`~qufzx.algebra.dimension.Dim`, possibly symbolic; ``index``
        may be a concrete int or a symbolic sympy expression. The sympy layer is reached
        only through ``dim.to_sympy()``, per the per-port-dimension invariant: this
        module holds no ambient notion of "the current d".
        """
        if isinstance(index, bool):
            raise PhaseGrammarError(f"root_of_unity() index does not accept bool, got {index!r}")
        if isinstance(index, int):
            index_expr: sp.Expr = sp.Integer(index)
        elif isinstance(index, sp.Expr):
            index_expr = index
        else:
            raise PhaseGrammarError(
                f"root_of_unity() index must be int or sympy expression, got {type(index).__name__}"
            )
        return cls(index_expr / dim.to_sympy())

    @classmethod
    def symbol(cls, name: str) -> Phase:
        """Build a free symbolic phase parameter, given in turns (not radians)."""
        return cls(_turns_symbol(name))

    def to_sympy_turns(self) -> sp.Expr:
        """Escape hatch returning the underlying canonical turns-expression."""
        return self._turns

    @property
    def is_zero(self) -> bool:
        """True iff this phase is provably zero (angle == 0 mod one turn).

        Sound but incomplete: a symbolic expression that is mathematically zero but
        not syntactically so after normalization returns False here.
        """
        return bool(self._turns == 0)

    @property
    def is_concrete(self) -> bool:
        """True iff this phase has no free symbols, i.e. it denotes a known angle."""
        return not self._turns.free_symbols

    @property
    def free_symbols(self) -> frozenset[str]:
        """The names of all free symbols appearing in this phase's turns-expression."""
        return frozenset(str(s.name) for s in self._turns.free_symbols)

    def __add__(self, other: Phase) -> Phase:
        """Add two phases (the spider-fusion operation on a single phase slot)."""
        if not isinstance(other, Phase):
            return NotImplemented
        return Phase(self._turns + other._turns)

    def __sub__(self, other: Phase) -> Phase:
        """Subtract two phases."""
        if not isinstance(other, Phase):
            return NotImplemented
        return Phase(self._turns - other._turns)

    def __neg__(self) -> Phase:
        """Negate this phase."""
        return Phase(-self._turns)

    def scale(self, factor: int) -> Phase:
        """Scale this phase by an exact integer (e.g. doubling for a squared spider)."""
        if isinstance(factor, bool) or not isinstance(factor, int):
            raise PhaseGrammarError(f"scale() requires an int, got {type(factor).__name__}")
        return Phase(self._turns * factor)

    def substitute(self, mapping: Mapping[PhaseSymbolKey, PhaseSubstituteValue]) -> Phase:
        """Return a new Phase with symbols replaced by concrete values.

        Keys may be symbol names (str) or bare-symbol Phases; values may be an int, a
        sympy Rational, or a concrete Phase (interpreted as its turns value). The mapping
        need not be total: symbols not mentioned are left symbolic. This Phase is never
        mutated; a new Phase is always returned. Note that mapping keys here are phase
        symbol names, distinct from dimension symbol names -- a Phase built via
        :meth:`root_of_unity` at a symbolic Dim carries the *dimension's* symbol in its
        expression, and that symbol is substituted through the same mapping, since both
        live in one shared sympy expression.
        """
        subs_dict: dict[sp.Symbol, sp.Expr] = {}
        by_name = {str(s.name): s for s in self._turns.free_symbols}
        for key, value in mapping.items():
            name = key if isinstance(key, str) else self._phase_symbol_name(key)
            if name not in by_name:
                continue
            if isinstance(value, bool):
                raise PhaseGrammarError(f"substitution value for {name!r} does not accept bool")
            if isinstance(value, int):
                subs_dict[by_name[name]] = sp.Integer(value)
            elif isinstance(value, sp.Rational):
                subs_dict[by_name[name]] = value
            elif isinstance(value, Phase):
                subs_dict[by_name[name]] = value._turns
            else:
                raise PhaseGrammarError(
                    f"substitution value for {name!r} must be int, sympy Rational, or Phase, "
                    f"got {type(value).__name__}"
                )
        return Phase(self._turns.subs(subs_dict))

    @staticmethod
    def _phase_symbol_name(key: Phase) -> str:
        if not key._turns.is_Symbol:
            raise PhaseGrammarError(f"substitution key must be a bare symbol, got {key}")
        return str(key._turns.name)

    def to_radians(self) -> sp.Expr:
        """Return the exact angle in radians, as a sympy expression (2*pi*turns).

        Does not require concreteness; this is a sympy-level view, not the numeric gate.
        """
        return 2 * sp.pi * self._turns

    def to_complex(self) -> complex:
        """The single gated numeric evaluator: e^{i*angle} as a Python complex.

        Raises PhaseDomainError if any free symbol remains. This is the only sanctioned
        path from a Phase to a concrete number.
        """
        if not self.is_concrete:
            raise PhaseDomainError(
                f"cannot convert symbolic phase {self} to a numeric value; "
                f"free symbols: {sorted(self.free_symbols)}"
            )
        return complex(sp.exp(2 * sp.pi * sp.I * self._turns))

    def __eq__(self, other: object) -> bool:
        """Exact equality after normalization; see the module docstring for soundness."""
        if not isinstance(other, Phase):
            return NotImplemented
        return bool(self._turns == other._turns)

    def __hash__(self) -> int:
        """Hash agrees with __eq__: equal Phases (post-normalization) hash equal."""
        return hash(self._turns)

    def __repr__(self) -> str:
        return f"Phase(turns={self._turns})"

    def __str__(self) -> str:
        return f"{self._turns} turns"


class PhaseVector:
    """The phase data of one spider: a mapping from leg index to :class:`Phase`.

    Entries are stored sparsely as ``index -> Phase`` with an implicit ``Phase.zero()``
    everywhere else. Sparse storage is the only representation under which a phase
    vector at a symbolic dimension can exist at all: a dense representation would need
    an enumerable list of length ``d - 1``, and ``d`` may not be concrete.

    Index 0 is gauge-fixed to zero (see the module docstring's denotation formula):
    this class rejects an explicit nonzero entry at index 0 outright, rather than
    silently absorbing or renormalizing it, so that a caller's mistake is never masked.

    :class:`~qufzx.algebra.dimension.Dim` has no subtraction, so the natural vector
    length ``d - 1`` is not itself representable as a Dim. This class therefore exposes
    :attr:`length` as a derived Python int (only when ``dim`` is concrete) rather than
    trying to construct a ``Dim`` for it.
    """

    __slots__ = ("_dim", "_entries")
    _dim: Dim
    _entries: Mapping[int, Phase]

    def __init__(self, dim: Dim, entries: Mapping[int, Phase] | None = None) -> None:
        """Build a PhaseVector over ``dim`` from a sparse index -> Phase mapping.

        Every index must be >= 1 and, when ``dim`` is concrete, <= dim - 1; violations
        raise PhaseDomainError. Explicit zero entries are dropped (they are equivalent
        to absence). An explicit nonzero entry at index 0 raises PhaseDomainError.
        """
        if not isinstance(dim, Dim):
            raise TypeError(f"PhaseVector requires a Dim, got {type(dim).__name__}")
        self._dim = dim
        cleaned: dict[int, Phase] = {}
        for index, phase in (entries or {}).items():
            if not isinstance(index, int) or isinstance(index, bool):
                raise PhaseGrammarError(f"phase-vector index must be an int, got {index!r}")
            if not isinstance(phase, Phase):
                raise PhaseGrammarError(f"phase-vector entry must be a Phase, got {phase!r}")
            if index == 0:
                if not phase.is_zero:
                    raise PhaseDomainError(
                        "index 0 is gauge-fixed to zero; explicit nonzero entry not allowed"
                    )
                continue
            if index < 0:
                raise PhaseDomainError(f"phase-vector index must be >= 1, got {index}")
            if dim.is_concrete and index > dim.to_int() - 1:
                raise PhaseDomainError(
                    f"phase-vector index {index} out of range for dim={dim} "
                    f"(valid range is 1..{dim.to_int() - 1})"
                )
            if phase.is_zero:
                continue
            cleaned[index] = phase
        self._entries = cleaned

    @property
    def dim(self) -> Dim:
        """The wire dimension this phase vector is tied to."""
        return self._dim

    @property
    def length(self) -> int | None:
        """The number of free phase slots (d - 1) as a Python int, or None if dim is symbolic.

        There is no Dim for d - 1 (Dim has no subtraction), so this is a derived
        property rather than something stored as a Dim.
        """
        if not self._dim.is_concrete:
            return None
        return self._dim.to_int() - 1

    def entries(self) -> Mapping[int, Phase]:
        """The sparse nonzero entries of this vector, index -> Phase."""
        return dict(self._entries)

    def get(self, index: int) -> Phase:
        """The phase at ``index`` (Phase.zero() if absent or index == 0)."""
        if index == 0:
            return Phase.zero()
        return self._entries.get(index, Phase.zero())

    @property
    def is_zero(self) -> bool:
        """True iff every entry is provably zero (i.e. there are no stored entries)."""
        return not self._entries

    @property
    def is_concrete(self) -> bool:
        """True iff the dimension and every stored phase are concrete."""
        return self._dim.is_concrete and all(p.is_concrete for p in self._entries.values())

    @property
    def free_symbols(self) -> frozenset[str]:
        """All free symbols appearing in the dimension or in any stored phase."""
        symbols = set(self._dim.free_symbols)
        for phase in self._entries.values():
            symbols |= phase.free_symbols
        return frozenset(symbols)

    def __add__(self, other: PhaseVector) -> PhaseVector:
        """Componentwise addition: the spider-fusion operation on phase vectors.

        Requires both vectors to share the same Dim (mixed-dimension fusion is
        undefined and must be rejected, per the wire well-formedness rule); raises
        PhaseDomainError otherwise.
        """
        if not isinstance(other, PhaseVector):
            return NotImplemented
        if self._dim != other._dim:
            raise PhaseDomainError(
                f"cannot add phase vectors over mismatched dimensions {self._dim} and {other._dim}"
            )
        indices = set(self._entries) | set(other._entries)
        merged = {i: self.get(i) + other.get(i) for i in indices}
        return PhaseVector(self._dim, merged)

    def substitute(self, mapping: Mapping[PhaseSymbolKey, PhaseSubstituteValue]) -> PhaseVector:
        """Return a new PhaseVector with symbols substituted into both dim and entries.

        Dimension symbols and phase symbols are substituted through the same mapping
        (a key, str or bare-symbol Phase, is recognized as targeting the dimension by
        name whenever that name is one of ``self.dim``'s free symbols). Substituting the
        dimension can turn a symbolic-length vector into a concrete-length one; any
        index bound that could not previously be checked (because dim was symbolic) is
        checked now, against the newly concrete dim, and raises PhaseDomainError if
        violated.

        A value substituted for a dimension symbol must be an exact integral quantity:
        a Python int, an ``sp.Integer``, or an ``sp.Rational`` whose denominator is 1
        (all reach :meth:`Dim.substitute`, which further enforces positivity). A
        non-integral ``sp.Rational`` (e.g. ``3/2``) or a ``Phase`` value for a
        dimension symbol is mathematically incoherent -- a dimension is a positive
        integer, not an angle -- and raises PhaseDomainError rather than silently
        substituting into the entries while leaving ``self.dim`` desynced from them.

        Symbols that are not among the dimension's free symbols are treated purely as
        phase symbols: substitution reaches only the entries, unmentioned symbols are
        left symbolic, and this PhaseVector is never mutated.
        """
        dim_symbols = self._dim.free_symbols
        dim_mapping: dict[str | Dim, int | Dim] = {}
        for key, value in mapping.items():
            name = key if isinstance(key, str) else Phase._phase_symbol_name(key)
            if name not in dim_symbols:
                continue
            if isinstance(value, bool):
                raise PhaseGrammarError(f"substitution value for {name!r} does not accept bool")
            if isinstance(value, Phase):
                raise PhaseDomainError(
                    f"dimension symbol {name!r} cannot be substituted with a Phase "
                    "value (a dimension is a positive integer, not an angle)"
                )
            if isinstance(value, int):
                dim_mapping[name] = value
            elif isinstance(value, sp.Rational):
                if value.q != 1:
                    raise PhaseDomainError(
                        f"dimension symbol {name!r} requires an exact integer value, "
                        f"got non-integral rational {value}"
                    )
                dim_mapping[name] = int(value)
            else:
                raise PhaseGrammarError(
                    f"substitution value for dimension symbol {name!r} must be int or "
                    f"sympy Rational, got {type(value).__name__}"
                )
        new_dim = self._dim.substitute(dim_mapping) if dim_mapping else self._dim
        new_entries = {index: phase.substitute(mapping) for index, phase in self._entries.items()}
        return PhaseVector(new_dim, new_entries)

    def __eq__(self, other: object) -> bool:
        """Equal iff same Dim and same entries after dropping implicit zeros."""
        if not isinstance(other, PhaseVector):
            return NotImplemented
        return self._dim == other._dim and self._entries == other._entries

    def __hash__(self) -> int:
        """Hash agrees with __eq__: equal PhaseVectors hash equal."""
        return hash((self._dim, tuple(sorted(self._entries.items(), key=lambda kv: kv[0]))))

    def __repr__(self) -> str:
        entries_repr = ", ".join(f"{i}: {p!r}" for i, p in sorted(self._entries.items()))
        return f"PhaseVector(dim={self._dim!r}, entries={{{entries_repr}}})"

    def __str__(self) -> str:
        entries_str = ", ".join(f"{i}: {p}" for i, p in sorted(self._entries.items()))
        return f"PhaseVector[{self._dim}]({{{entries_str}}})"
