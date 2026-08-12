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

Phase 5 slice only (Phase 5 post-closing audit round 19, Task 3). ``FULL_PLAN.md``'s Phase 5
completion condition names a specific chain: "the full path Dirac to graph to fuse to graph
runs and the oracle confirms exact equality". Before this module, that chain did not exist --
``tests/test_phase5_oracle.py`` built its "A into B" worked example by hand
(``tests/helpers.py::build_ghz_with_copy``) and stated in its own docstring that the Dirac
half was deferred to Phases 17/18. This module closes exactly that gap, and no more:

* :func:`parse_dirac_source` accepts a single, restricted form -- a summed ket family
  ``sum_{k=0}^{D-1} |k,k,...>`` (or the ``|k>^{n}`` tensor-power shorthand), optionally
  followed by ``; copy`` to feed the resulting state into a fixed two-output "copy" spider.
  This is exactly, and only, the shape Phase 5's worked example (``build_ghz_with_copy``)
  needs: a state-prep spider with 0 inputs and ``n`` outputs, optionally wired into a
  1-input/2-output copy spider.
* ``D`` may be a concrete positive integer or a bare identifier (a symbolic
  :class:`~qufzx.algebra.dimension.Dim`); ``n`` (the leg count) must be concrete, per the
  round-19 task's own scope -- a symbolic leg count is a
  :class:`~qufzx.diagram.generators.LegPolicy` question this slice does not touch.
* The emitted :class:`~qufzx.diagram.graph.Diagram` never builds a matrix or dense tensor
  (the ket-sum itself is never evaluated numerically) -- it only ever allocates nodes,
  wires, and a boundary order, exactly the same graph-construction calls
  ``build_ghz_with_copy`` makes by hand.

What this module deliberately does **not** do, because it belongs to a later phase: a
general spider/wire/bang-box declaration syntax (Phase 18's own DSL), bang boxes, families
indexed by more than the one implicit ``k``, or a printer (the ``Diagram`` to Dirac
direction, Phase 17). ``copy`` is a single recognized keyword standing in for the one
specific copy spider the worked example needs, not a general "declare an arbitrary
generator" mechanism -- Phase 18 is expected to replace it with real declaration syntax,
at which point this module's grammar is a strict subset of that one.
"""

from __future__ import annotations

import re

from qufzx.algebra.dimension import Dim
from qufzx.diagram.generators import Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef


class DiracError(Exception):
    """Base of every error :func:`parse_dirac_source` can raise."""


class DiracGrammarError(DiracError):
    """The source text does not match this module's restricted Dirac grammar at all."""


class DiracDomainError(DiracError):
    """The source text parses, but names a value outside what this slice accepts (e.g. a
    zero leg count)."""


_KET_SUM_RE = re.compile(
    r"^sum_\{k=0\}\^\{(?P<dim>[A-Za-z_]\w*|\d+)-1\}\s*"
    r"\|(?P<body>[^>]*)>"
    r"(?:\^\{(?:\\otimes|⊗)?\s*(?P<power>\d+)\s*\})?$"
)
"""Matches ``sum_{k=0}^{D-1} |BODY>`` with an optional ``^{n}`` (or ``^{\\otimes n}``/
``^{⊗n}``) tensor-power suffix on the ket. ``BODY`` is read by :func:`_leg_count_from_body`."""


def _leg_count_from_body(body: str, power: str | None) -> int:
    """How many output legs the ket family declares.

    ``body`` must, once commas and whitespace are stripped, consist only of repeated ``k``
    characters (``"k"``, ``"kk"``, ``"k,k"``, ``"k, k, k"``, ...) -- one leg per ``k``. A
    ``power`` suffix (``|k>^{n}``) is the tensor-power shorthand instead: it requires
    ``body`` to be the single index ``"k"`` and supplies the leg count directly.
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
                "ambiguous: use either '|k>^{{n}}' or '|k,k,...>', not both"
            )
        leg_count = int(power)
    else:
        leg_count = len(stripped)
    if leg_count < 1:
        raise DiracDomainError(
            f"ket family declares {leg_count} output legs; at least 1 is required"
        )
    return leg_count


def _parse_dim(token: str) -> Dim:
    """A concrete positive integer, or a bare identifier naming a symbolic ``Dim``."""
    if token.isdigit():
        return Dim.concrete(int(token))
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

    See the module docstring for exactly what grammar this accepts. Two forms:

    * ``"sum_{k=0}^{D-1} |k,k,...>"`` alone: one state-prep spider, 0 inputs, one output
      per ket index, boundary outputs in ket order. This is ``A`` in
      :func:`tests.helpers.build_ghz_with_copy`.
    * ``"sum_{k=0}^{D-1} |k,k,...>; copy"``: the same state, with its first output wired
      into a fixed 1-input/2-output copy spider (also over ``D``); the boundary is the
      state's own remaining outputs (in order) followed by the copy spider's two outputs.
      This is the full "A into B" shape ``build_ghz_with_copy`` builds by hand, structurally
      identical port-for-port (see ``tests/test_phase5_dirac_oracle.py``).

    Raises :class:`DiracGrammarError` for source text outside this slice's restricted
    grammar, and :class:`DiracDomainError` for grammatically valid source naming a value
    (e.g. a zero leg count) this slice does not accept.
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
    diagram.add_wire(
        PortRef(state_id, Direction.OUTPUT, 0), PortRef(copy_id, Direction.INPUT, 0)
    )
    boundary_outputs = [
        PortRef(state_id, Direction.OUTPUT, i) for i in range(1, leg_count)
    ] + [PortRef(copy_id, Direction.OUTPUT, 0), PortRef(copy_id, Direction.OUTPUT, 1)]
    diagram.set_boundary_outputs(boundary_outputs)
    return diagram
