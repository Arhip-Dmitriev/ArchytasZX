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

"""Input DSL parser: spiders, wires, symbolic phases, bang boxes, dimensions, and Dirac kets.

Phase 5 slice only, supplying the Dirac-to-graph end of that phase's completion condition:

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

from qufzx.algebra.dimension import Dim, DimensionDomainError
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef


class DiracError(Exception):
    """Base of every error :func:`parse_dirac_source` can raise.

    Every exception reaching a caller is a :class:`DiracError`, never a foreign class from a
    package this module calls into. A change here should re-check that every outbound call is
    still contained. The current calls:

    * ``Dim.concrete`` -- raises :class:`~qufzx.algebra.dimension.DimensionDomainError` for
      a non-positive integer; :func:`_parse_dim` wraps the call.
    * ``Dim.symbol`` -- gated in :func:`_parse_dim` itself, not only in :data:`_KET_SUM_RE`,
      on :data:`_IDENTIFIER_RE` (``[A-Za-z_]\\w*``), so every name reaching it is a shape
      ``sympy.Symbol`` accepts and ``Dim`` builds with the ``positive=True, integer=True``
      pair ``_check_dimension_domain`` requires.
    * ``Diagram.add_node`` -- forwards to ``Node``/``Port``, which raise
      ``GraphGrammarError`` for a non-``GeneratorType``, a non-``Port`` leg, or a
      non-``PhaseVector`` phase. Every call here passes ``Z_SPIDER``, a list of ``Dim``, and
      no phase.
    * ``Diagram.add_wire`` -- raises ``GraphDomainError`` only for ``a == b``, and every wire
      built here joins two distinct fresh ports on different nodes.
    * ``Diagram.set_boundary_outputs`` -- never raises.
    * ``int(token)`` -- every call site first confirms the token is a non-empty run of ASCII
      ``0-9`` (:data:`_ASCII_DIGITS_RE`, or the regex's own ``(?P<power>[0-9]+)`` group).
      ``str.isdigit()`` is not that predicate: it is true for Unicode category ``No``, so
      ``'²'`` is ``isdigit()``, matches neither ``\\d+`` nor ``[0-9]+``, and makes ``int()``
      raise ``ValueError``.
    """


class DiracGrammarError(DiracError):
    """The source text does not match this module's restricted Dirac grammar at all."""


class DiracDomainError(DiracError):
    """The source text parses, but names a value outside this slice: a zero leg count, a
    dimension outside ``Dim``'s domain, the bound summation index used as a dimension symbol,
    or a leg count above :data:`_MAX_KET_LEG_COUNT`."""


_SUMMATION_INDEX = "k"
"""The only bound summation variable this slice's grammar recognizes.

One constant, so :data:`_KET_SUM_RE` and :func:`_parse_dim`'s dimension-symbol exclusion
cannot drift apart."""

_ASCII_DIGITS = "[0-9]+"
"""The one decimal-literal shape this module's grammar admits, as a regex fragment.

``[0-9]+``, not ``\\d+`` and not ``str.isdigit()``: Python's ``\\d`` is Unicode-aware
(category ``Nd``) and admits e.g. ``'\u0663'``, for which ``int()`` returns 3 -- a non-ASCII
digit silently accepted as a concrete dimension in an otherwise all-ASCII DSL.
``str.isdigit()`` is broader still, admitting category ``No`` (``'\u00b2'``), for which
``int()`` raises ``ValueError``. Leading zeros are accepted: ``07`` is 7.

One constant, shared with :data:`_KET_SUM_RE`'s numeric groups and
:data:`_ASCII_DIGITS_RE`."""

_ASCII_DIGITS_RE = re.compile(rf"^{_ASCII_DIGITS}$")
"""Whole-token form of :data:`_ASCII_DIGITS`, for :func:`_parse_dim`'s guard."""

_IDENTIFIER = r"[A-Za-z_]\w*"
"""The one bare-identifier shape this module's grammar admits, as a regex fragment.

``\\w`` stays Unicode-aware, unlike the numeric branch: ``sympy.Symbol`` accepts such a name
and nothing downstream cares. Shared by :data:`_KET_SUM_RE`'s ``dim`` group and
:data:`_IDENTIFIER_RE`."""

_IDENTIFIER_RE = re.compile(rf"^{_IDENTIFIER}$")
"""Whole-token form of :data:`_IDENTIFIER`, for :func:`_parse_dim`'s guard."""

_MAX_KET_LEG_COUNT = 1024
"""Parser sanity bound on the ``^{n}`` tensor-power leg count, not a semantic limit.

Nothing about ZX-calculus caps a spider's leg count; this only stops one adversarial source
string forcing unbounded eager port allocation. Same role as ``_MAX_FIXPOINT_PASSES`` in
:mod:`qufzx.rewrite.match`."""

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

    Raises :class:`DiracDomainError` for a concrete value outside ``Dim``'s domain, and for
    the bound summation index (:data:`_SUMMATION_INDEX`) in a dimension slot -- a later
    ``substitute`` would bind it independently of the sum it lexically came from. Raises
    :class:`DiracGrammarError` for a token matching neither shape, which
    :data:`_KET_SUM_RE` rejects before this function is reached.
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
    # Total on its own terms, not merely on the tokens _KET_SUM_RE currently hands it:
    # without this branch a token matching neither shape becomes a symbol named after
    # itself.
    if not _IDENTIFIER_RE.match(token):
        raise DiracGrammarError(
            f"dimension token {token!r} is neither a decimal literal ({_ASCII_DIGITS}) nor "
            f"a bare identifier ({_IDENTIFIER})"
        )
    return Dim.symbol(token)


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
