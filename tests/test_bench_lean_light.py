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

"""Light-import attempt of benchmarks.lean.check: the import rewrite, the fallback to the file as
written and the attempts recorded."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks import lean

SOURCE = (
    "import ArchytasZX\n\nnamespace ArchytasZX.Export\n\n"
    "theorem main : True := trivial\n\nend ArchytasZX.Export\n"
)


def test_light_source() -> None:
    light = lean.light_source(SOURCE)
    assert light == SOURCE.replace("import ArchytasZX\n", "import ArchytasZX.Light\n", 1)
    assert lean.light_source("import ArchytasZX.PathSum\n\ntheorem t : True := trivial\n") is None
    assert lean.light_source("theorem t : True := trivial\n-- import ArchytasZX\n") is None
    two = "import ArchytasZX.PathSum\nimport ArchytasZX\n\ntheorem t : True := trivial\n"
    assert lean.light_source(two) == two.replace(
        "\nimport ArchytasZX\n", "\nimport ArchytasZX.Light\n"
    )


def _fake(results: dict[str, dict[str, Any]], seen: list[str]) -> Any:
    def attempt(
        path: Path, source: str, name: str | None, timeout: float | None, fraction: float
    ) -> dict[str, Any]:
        kind = "light" if lean.LIGHT_IMPORT in source else "full"
        seen.append(kind)
        base = {"lean_cpu": 1.0, "lean_wall": 1.0, "lean_peak_mb": 10.0, "decl_ms": []}
        return base | results[kind]

    return attempt


@pytest.mark.parametrize(
    ("light", "used", "runs"),
    [
        ({"lean_status": "accepted", "axioms": ["propext"]}, "light", ["light"]),
        (
            {"lean_status": "rejected", "axioms": None, "lean_detail": "unknown"},
            "full",
            ["light", "full"],
        ),
        ({"lean_status": "oom", "axioms": None, "lean_kill": "budget"}, "full", ["light", "full"]),
        ({"lean_status": "timeout", "axioms": None}, "full", ["light", "full"]),
        ({"lean_status": "accepted", "axioms": None}, "full", ["light", "full"]),
    ],
)
def test_check_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    light: dict[str, Any],
    used: str,
    runs: list[str],
) -> None:
    path = tmp_path / "export.lean"
    path.write_text(SOURCE, encoding="utf-8")
    full = {"lean_status": "rejected", "axioms": None, "lean_detail": "full says no"}
    seen: list[str] = []
    monkeypatch.setattr(lean, "_attempt", _fake({"light": light, "full": full}, seen))
    out = lean.check(path)
    assert seen == runs
    assert out["lean_import"] == used
    assert [a["import"] for a in out["lean_attempts"]] == runs
    final = light if used == "light" else full
    assert out["lean_status"] == final["lean_status"]
    assert out["axiom_class"] == ("clean" if used == "light" else "rejected")
    assert out["lean_attempts"][0]["lean_status"] == light["lean_status"]


def test_check_without_aggregate_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "export.lean"
    path.write_text(
        SOURCE.replace("import ArchytasZX\n", "import ArchytasZX.PathSum\n"), encoding="utf-8"
    )
    seen: list[str] = []
    full = {"lean_status": "accepted", "axioms": []}
    monkeypatch.setattr(lean, "_attempt", _fake({"full": full}, seen))
    out = lean.check(path)
    assert seen == ["full"]
    assert out["lean_import"] == "full"
    assert out["axiom_class"] == "clean"
