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

"""Enforces the project's docstring policy across ``qufzx/``: budgets, and no "because".

A docstring states what a thing is or how it works. Reasoning belongs in FULL_PLAN.md, where
re-reading it produces plan revisions rather than fresh docstring-drift findings. The budgets
below are the mechanical half of that rule; the ban on "because" is the one WHY marker
specific enough to check by hand-free means.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = _REPO_ROOT / "qufzx"

_MAX_MODULE_DOCSTRING_LINES = 65
"""Ceiling on a module docstring. ``qufzx/rewrite/match.py`` sets the high-water mark at 64:
its numbered side-condition list is machine-checked contract, not prose."""

_MAX_DOCSTRING_LINES = 30
"""Ceiling on every other docstring, module-attribute docstrings included.
``qufzx.rewrite.engine.apply``'s enumeration of its raise conditions sets the mark at 30."""

_MAX_PROSE_RATIO = 0.50
"""Ceiling on docstring+comment lines as a fraction of a file's non-blank lines."""

_LICENSE_HEADER_LINES = 12

_BANNED = re.compile(r"\bbecause\b", re.IGNORECASE)
"""The one rationale marker checked lexically. A sentence needing it is answering *why*."""


def _python_files() -> list[Path]:
    return sorted(p for p in _PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_nodes(tree: ast.Module) -> list[ast.Expr]:
    """Every bare string expression: real docstrings and attribute docstrings alike."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
class TestDocstringPolicy:
    """One parametrized case per module in ``qufzx/``."""

    def test_module_docstring_is_within_budget(self, path: Path) -> None:
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        lines = len(docstring.splitlines()) if docstring else 0
        assert lines <= _MAX_MODULE_DOCSTRING_LINES, (
            f"{path.relative_to(_REPO_ROOT)}: module docstring is {lines} lines, over the "
            f"{_MAX_MODULE_DOCSTRING_LINES}-line budget. State what the module is and the "
            "contracts it keeps; put the reasoning in FULL_PLAN.md."
        )

    def test_no_docstring_exceeds_its_budget(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_docstring_line = tree.body[0].lineno if ast.get_docstring(tree) else None
        oversized = [
            (node.lineno, (node.end_lineno or node.lineno) - node.lineno + 1)
            for node in _docstring_nodes(tree)
            if node.lineno != module_docstring_line
            and (node.end_lineno or node.lineno) - node.lineno + 1 > _MAX_DOCSTRING_LINES
        ]
        assert not oversized, (
            f"{path.relative_to(_REPO_ROOT)}: docstring(s) over the "
            f"{_MAX_DOCSTRING_LINES}-line budget at line(s) "
            f"{[line for line, _ in oversized]} ({[n for _, n in oversized]} lines)"
        )

    def test_prose_does_not_outweigh_code(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if not any(not isinstance(node, ast.Expr) for node in tree.body):
            # A Phase 0.2 skeleton: a docstring and nothing else, so the ratio is 1 by
            # construction and says nothing about how it is written.
            return
        prose_lines: set[int] = set()
        for node in _docstring_nodes(tree):
            prose_lines |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        for number, line in enumerate(source.splitlines(), 1):
            if number > _LICENSE_HEADER_LINES and line.lstrip().startswith("#"):
                prose_lines.add(number)
        non_blank = {
            number
            for number, line in enumerate(source.splitlines(), 1)
            if line.strip() and number > _LICENSE_HEADER_LINES
        }
        prose_lines &= non_blank
        if not non_blank:
            return
        ratio = len(prose_lines) / len(non_blank)
        assert ratio <= _MAX_PROSE_RATIO, (
            f"{path.relative_to(_REPO_ROOT)}: {ratio:.0%} of non-blank lines are docstring "
            f"or comment, over the {_MAX_PROSE_RATIO:.0%} budget "
            f"({len(prose_lines)} of {len(non_blank)})"
        )

    def test_no_docstring_or_comment_answers_why(self, path: Path) -> None:
        offenders = [
            number
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if _BANNED.search(line)
        ]
        assert not offenders, (
            f"{path.relative_to(_REPO_ROOT)}: 'because' at line(s) {offenders}. A docstring "
            "states what a thing is or how it works; a sentence needing 'because' is "
            "answering why, which belongs in FULL_PLAN.md."
        )
