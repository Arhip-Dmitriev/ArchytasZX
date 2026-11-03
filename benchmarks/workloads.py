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

"""Workloads shared by the suites: the families (scaling.py's seven plus GHZ), the example and
lemma command files, the breadth files, and splitting a command file into export units."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks import scaling

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
WORKLOADS = Path(__file__).resolve().parent / "workloads"
PAPERS = ROOT / "archytaszx" / "papers"
PAPER_KEYS = {"P1": "poor2023zxw", "P2": "wang2023qufinitezxw"}

GHZ = (
    "n",
    """
@rhs =
    dim d{D}
    count n{N}
    g = Z(0 -> 2)
    repeat n: g.out1
    scalar d^(-1/2)

load
    dim d{D}
    count n{N}
    X(0 -> 1) -> F -> z = Z(1 -> 2)
    repeat n: z.out1 -> t = X(2 -> 1), X(0 -> 1) -> t
    scalar d^(-1/2)

decide against @rhs
""",
)
FAMILIES: dict[str, tuple[str, str]] = {**scaling.FAMILIES, "GHZ": GHZ}
_VERDICT = re.compile(r"^(decide|prove)\b")


def _bind(value: object) -> str:
    """`` = VALUE``, or nothing for ``"free"``."""
    return "" if value == "free" else f" = {value}"


def family_script(family: str, n: object = "free", d: object = "free") -> str:
    """The family's command file with ``n`` and ``d`` bound, or free when ``"free"``."""
    _, text = FAMILIES[family]
    return text.format(N=_bind(n), D=_bind(d))


def example_files() -> list[Path]:
    """Every command file under examples/, sorted."""
    return sorted(EXAMPLES.rglob("*.azx"))


def breadth_files() -> list[Path]:
    """Every command file under workloads/breadth/, sorted."""
    return sorted((WORKLOADS / "breadth").glob("*.azx"))


def rel(path: Path) -> str:
    """``path`` relative to the repository root."""
    return str(path.resolve().relative_to(ROOT))


def _blocks(script: str) -> list[list[str]]:
    """The script's physical lines grouped into logical commands (a command and its indented
    continuation lines); comment and blank lines are dropped."""
    out: list[list[str]] = []
    for raw in script.splitlines():
        code = raw.split("#", 1)[0].rstrip()
        if not code.strip():
            continue
        if raw[:1] in (" ", "\t") and out:
            out[-1].append(raw)
        else:
            out.append([raw])
    return out


def units(script: str) -> list[str]:
    """One script per ``decide``/``prove`` line: every non-verdict, non-certificate line before
    it, the verdict, and the ``certificate emit``/``kernel`` lines right after it."""
    blocks = _blocks(script)
    out: list[str] = []
    for k, block in enumerate(blocks):
        head = block[0].strip()
        if not _VERDICT.match(head):
            continue
        before = [
            b
            for b in blocks[:k]
            if not _VERDICT.match(b[0].strip()) and not b[0].strip().startswith("certificate")
        ]
        after: list[list[str]] = []
        for b in blocks[k + 1 :]:
            word = b[0].strip()
            if not word.startswith("certificate"):
                break
            if word.startswith(("certificate emit", "certificate kernel")):
                after.append(b)
        out.append("\n".join(line for b in [*before, block, *after] for line in b) + "\n")
    return out


def with_lean(script: str, path: Path, kernel: bool = True) -> str:
    """``script`` followed by ``certificate kernel`` (unless present or ``kernel`` is False)
    and ``certificate lean PATH``."""
    lines = [script.rstrip("\n")]
    if kernel and "certificate kernel" not in script:
        lines.append("certificate kernel")
    lines.append(f"certificate lean {path}")
    return "\n".join(lines) + "\n"


def lemma_entries() -> list[dict[str, object]]:
    """Every lemma of both LEMMAS.txt files: ``paper`` (P1, P2), ``lemma`` (its number as
    printed), ``kind`` (Lemma, Proposition, Corollary) and ``file`` (the encoded command file
    relative to the repository root, or None)."""
    out: list[dict[str, object]] = []
    head = re.compile(r"^(Lemma|Proposition|Corollary|Theorem)\s+([0-9A-Z][0-9A-Za-z.]*)\s*(.*)$")
    for paper, key in PAPER_KEYS.items():
        text = (PAPERS / key / "LEMMAS.txt").read_text(encoding="utf-8")
        entry: dict[str, object] | None = None
        for line in text.splitlines():
            m = head.match(line)
            if m:
                entry = {"paper": paper, "kind": m.group(1), "lemma": m.group(2), "file": None}
                out.append(entry)
                continue
            if entry is not None and line.strip().startswith("encoded:"):
                f = re.search(r"(examples/\S+\.azx|benchmarks/\S+\.azx)", line)
                if f and (ROOT / f.group(1)).exists():
                    entry["file"] = f.group(1)
    known = {e["file"] for e in out if e["file"]}
    files = [*sorted((EXAMPLES / "completeness" / "lemmas").glob("*.azx"))]
    files += sorted((WORKLOADS / "completeness").glob("*.azx"))
    for path in files:
        m = re.match(r"(p[12])_lemma(A?)0*([0-9]+)([a-z0-9]*)\.azx$", path.name)
        if not m or rel(path) in known:
            continue
        paper = m.group(1).upper()
        lemma = f"{m.group(2) + '.' if m.group(2) else ''}{m.group(3)}"
        entry = next((e for e in out if e["paper"] == paper and e["lemma"] == lemma), None)
        if entry is not None and entry["file"] is None and not m.group(4):
            entry["file"] = rel(path)
        else:
            out.append(
                {"paper": paper, "kind": "Lemma", "lemma": lemma + m.group(4), "file": rel(path)}
            )
    return out


def unit_scripts() -> list[tuple[str, str, str, str]]:
    """Every proof workload as (group, name, script, base directory): every verdict of every
    example (the lemma files counted once, as lemmas), every family free, every breadth file
    and every encoded lemma."""
    out: list[tuple[str, str, str, str]] = []
    lemma_dir = EXAMPLES / "completeness" / "lemmas"
    for path in example_files():
        if path.parent == lemma_dir:
            continue
        for k, script in enumerate(units(path.read_text(encoding="utf-8"))):
            out.append(("examples", f"{rel(path)} #{k + 1}", script, str(path.parent)))
    for family in FAMILIES:
        out.append(("families", family, family_script(family), str(ROOT)))
    for path in breadth_files():
        for k, script in enumerate(units(path.read_text(encoding="utf-8"))):
            out.append(("breadth", f"{path.stem} #{k + 1}", script, str(path.parent)))
    for entry in lemma_entries():
        if entry["file"] is None:
            continue
        path = ROOT / str(entry["file"])
        for k, script in enumerate(units(path.read_text(encoding="utf-8"))):
            name = f"[{entry['paper']}] {entry['kind']} {entry['lemma']} #{k + 1}"
            out.append(("completeness", name, script, str(path.parent)))
    return out


def split_verdict(script: str) -> tuple[str, str]:
    """The script before its last ``decide``/``prove`` line, and that line."""
    blocks = _blocks(script)
    k = max(i for i, b in enumerate(blocks) if _VERDICT.match(b[0].strip()))
    prefix = "\n".join(line for b in blocks[:k] for line in b) + "\n"
    return prefix, " ".join(line.strip() for line in blocks[k])
