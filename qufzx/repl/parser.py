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

"""Input DSL parser, Phase 5's Dirac slice only.

Supplies the Dirac-to-graph end of Phase 5's completion condition:

* :func:`parse_dirac_source` accepts one restricted form, a summed ket family
  ``sum_{k=0}^{D-1} |k,k,...>`` (or the ``|k>^{n}`` tensor-power shorthand), optionally
  followed by ``; copy`` to feed the state into a fixed two-output copy spider.
* ``D`` may be a concrete positive integer or a bare identifier (a symbolic
  :class:`~qufzx.algebra.dimension.Dim`); ``n`` must be concrete. The bound summation index
  is rejected in a dimension slot.
* The emitted diagram never builds a matrix or dense tensor; it allocates nodes, wires and a
  boundary order.
* The tensor-power leg count is bounded by :data:`_MAX_KET_LEG_COUNT`.
* No foreign exception hierarchy escapes this module; see :class:`DiracError`.

Out of scope, and belonging to later phases: a general spider/wire/bang-box declaration
syntax (Phase 18), bang boxes (Phase 7), multi-index families, and the diagram-to-Dirac
printer (Phase 17). ``copy`` is a single keyword standing in for the one copy spider the
worked example needs; Phase 18 must keep this grammar as a strict subset of its own.
"""

from __future__ import annotations

import re

from qufzx.algebra.dimension import Dim, DimensionDomainError, DimensionError
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef


class DiracError(Exception):
    """Base of every error :func:`parse_dirac_source` can raise.

    No foreign exception class from a package this module calls into reaches a caller.
    ``Dim.concrete`` and ``Dim.symbol`` are both wrapped in :func:`_parse_dim`, whose
    identifier shape is Unicode-aware and so wider than ``Dim``'s own; ``Diagram.add_node``,
    ``add_wire`` and ``set_boundary_outputs`` are only ever called with a ``Z_SPIDER``, a list
    of ``Dim``, no phase, and two distinct fresh ports on different nodes; every ``int()``
    call site first confirms its token is a run of :data:`_ASCII_DIGITS`.
    """


class DiracGrammarError(DiracError):
    """The source text does not match this module's restricted Dirac grammar at all."""


class DiracDomainError(DiracError):
    """The source text parses, but names a value outside this slice: a zero leg count, a
    dimension outside ``Dim``'s domain, the bound summation index used as a dimension symbol,
    or a leg count above :data:`_MAX_KET_LEG_COUNT`."""


_SUMMATION_INDEX = "k"
"""The only bound summation variable this slice's grammar recognizes. Shared by
:data:`_KET_SUM_RE` and :func:`_parse_dim`'s dimension-symbol exclusion."""

_ASCII_DIGITS = "[0-9]+"
"""The one decimal-literal shape this module's grammar admits, as a regex fragment.

ASCII-only, unlike ``\\d`` (Unicode category ``Nd``) and ``str.isdigit()`` (also ``No``).
Leading zeros are accepted: ``07`` is 7. Shared with :data:`_KET_SUM_RE`'s numeric groups
and :data:`_ASCII_DIGITS_RE`."""

_ASCII_DIGITS_RE = re.compile(rf"^{_ASCII_DIGITS}$")
"""Whole-token form of :data:`_ASCII_DIGITS`, for :func:`_parse_dim`'s guard."""

_IDENTIFIER = r"[A-Za-z_]\w*"
"""The one bare-identifier shape this module's grammar admits, as a regex fragment.
Unicode-aware, unlike the numeric branch. Shared by :data:`_KET_SUM_RE`'s ``dim`` group and
:data:`_IDENTIFIER_RE`."""

_IDENTIFIER_RE = re.compile(rf"^{_IDENTIFIER}$")
"""Whole-token form of :data:`_IDENTIFIER`, for :func:`_parse_dim`'s guard."""

_MAX_KET_LEG_COUNT = 1024
"""Parser sanity bound on the ``^{n}`` tensor-power leg count, not a semantic limit. Same
role as ``_MAX_FIXPOINT_PASSES`` in :mod:`qufzx.rewrite.match`."""

_KET_SUM_RE = re.compile(
    rf"^sum_\{{{_SUMMATION_INDEX}=0\}}\^\{{(?P<dim>{_IDENTIFIER}|{_ASCII_DIGITS})-1\}}\s*"
    rf"\|(?P<body>[^>]*)>"
    rf"(?:\^\{{(?:\\otimes|⊗)?\s*(?P<power>{_ASCII_DIGITS})\s*\}})?$"
)
"""Matches ``sum_{k=0}^{D-1} |BODY>`` with an optional ``^{n}`` (or ``^{\\otimes n}``/
``^{⊗n}``) tensor-power suffix on the ket. ``BODY`` is read by :func:`_leg_count_from_body`."""


def _leg_count_from_body(body: str, power: str | None) -> int:
    """How many output legs the ket family declares.

    ``body`` must, once commas and whitespace are stripped, consist only of repeated ``k``
    characters (``"k"``, ``"kk"``, ``"k,k"``, ``"k, k, k"``, ...) -- one leg per ``k``.
    Commas and spaces are separators only, so ``|k,,k>`` and ``|kk>`` are the same two-leg
    body. A ``power`` suffix (``|k>^{n}``) is the tensor-power shorthand instead: it
    requires ``body`` to be the single index ``"k"`` and supplies the leg count directly.
    """
    stripped = body.replace(",", "").replace(" ", "")
    if not stripped or any(ch != "k" for ch in stripped):
        raise DiracGrammarError(
            f"unsupported ket body {body!r}: this slice only accepts repeated 'k' indices, "
            "e.g. '|k,k>' or '|kk>'"
        )
    if power is not None:
        if stripped != "k":
            raise DiracGrammarError(
                f"ket body {body!r} combined with a tensor-power suffix ^{{{power}}} is "
                f"ambiguous: use either '|k>^{{n}}' or '|k,k,...>', not both"
            )
        leg_count = int(power)
    else:
        leg_count = len(stripped)
    if leg_count < 1:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs; at least 1 is required"
        )
    if leg_count > _MAX_KET_LEG_COUNT:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs, above this parser's sanity bound "
            f"of {_MAX_KET_LEG_COUNT} (see _MAX_KET_LEG_COUNT's docstring)"
        )
    return leg_count


def _parse_dim(token: str) -> Dim:
    """A concrete positive integer, or a bare identifier naming a symbolic ``Dim``.

    Raises :class:`DiracDomainError` for a concrete value outside ``Dim``'s domain and for
    the bound summation index (:data:`_SUMMATION_INDEX`) in a dimension slot, and
    :class:`DiracGrammarError` for a token matching neither shape, or matching
    :data:`_IDENTIFIER` but not ``Dim``'s own narrower name rule.
    """
    if _ASCII_DIGITS_RE.match(token):
        try:
            return Dim.concrete(int(token))
        except DimensionDomainError as exc:
            raise DiracDomainError(
                f"dimension token {token!r} is outside Dim's domain: {exc}"
            ) from exc
    if token == _SUMMATION_INDEX:
        raise DiracDomainError(
            f"dimension token {token!r} is the bound summation index; a dimension symbol "
            "must be free, not the variable the enclosing sum binds"
        )
    # Total on its own terms, not merely on the tokens _KET_SUM_RE currently hands it.
    if not _IDENTIFIER_RE.match(token):
        raise DiracGrammarError(
            f"dimension token {token!r} is neither a decimal literal ({_ASCII_DIGITS}) nor "
            f"a bare identifier ({_IDENTIFIER})"
        )
    try:
        return Dim.symbol(token)
    except DimensionError as exc:
        raise DiracGrammarError(
            f"dimension token {token!r} matches this module's identifier shape "
            f"({_IDENTIFIER}, Unicode-aware) but is not a name Dim accepts: {exc}"
        ) from exc


def _parse_ket_sum(text: str) -> tuple[Dim, int]:
    """Parse ``sum_{k=0}^{D-1} |...>`` into ``(dim, leg_count)``."""
    match = _KET_SUM_RE.match(text.strip())
    if match is None:
        raise DiracGrammarError(
            f"{text!r} does not match this slice's grammar: expected "
            "'sum_{k=0}^{D-1} |k,k,...>' (or the '|k>^{n}' shorthand)"
        )
    dim = _parse_dim(match.group("dim"))
    leg_count = _leg_count_from_body(match.group("body"), match.group("power"))
    return dim, leg_count


def parse_dirac_source(source: str) -> Diagram:
    """Parse a restricted Dirac-ket source string into a :class:`~qufzx.diagram.graph.Diagram`.

    Two forms, both detailed in the module docstring:

    * ``"sum_{k=0}^{D-1} |k,k,...>"``: one state-prep spider, 0 inputs, one output per ket
      index, boundary outputs in ket order.
    * ``"sum_{k=0}^{D-1} |k,k,...>; copy"``: the same state with its first output wired into
      a fixed 1-input/2-output copy spider over ``D``; the boundary is the state's remaining
      outputs in order, then the copy spider's two. Port-for-port identical to
      :func:`tests.helpers.build_ghz_with_copy` (see ``tests/test_phase5_dirac_oracle.py``).

    Raises :class:`DiracGrammarError` for source outside this grammar and
    :class:`DiracDomainError` for a value this slice does not accept.
    """
    terms = [term.strip() for term in source.split(";")]
    if len(terms) not in (1, 2) or any(not term for term in terms):
        raise DiracGrammarError(
            f"{source!r}: expected a single ket-sum term, optionally followed by one "
            "';' and the keyword 'copy'"
        )
    dim, leg_count = _parse_ket_sum(terms[0])

    diagram = Diagram()
    state_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim] * leg_count)

    if len(terms) == 1:
        diagram.set_boundary_outputs(
            [PortRef(state_id, Direction.OUTPUT, i) for i in range(leg_count)]
        )
        return diagram

    if terms[1] != "copy":
        raise DiracGrammarError(
            f"{terms[1]!r}: this slice recognizes only the 'copy' keyword after ';' "
            "(a general spider-declaration syntax is Phase 18's, not this slice's)"
        )
    copy_id = diagram.add_node(Z_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    diagram.add_wire(PortRef(state_id, Direction.OUTPUT, 0), PortRef(copy_id, Direction.INPUT, 0))
    boundary_outputs = [PortRef(state_id, Direction.OUTPUT, i) for i in range(1, leg_count)] + [
        PortRef(copy_id, Direction.OUTPUT, 0),
        PortRef(copy_id, Direction.OUTPUT, 1),
    ]
    diagram.set_boundary_outputs(boundary_outputs)
    return diagram
