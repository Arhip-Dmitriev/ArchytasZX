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

"""QuiZX adapter: as PyZX (drivers/zx.py), with QuiZX's QASM reader and full_simp."""

from __future__ import annotations

from typing import Any

from benchmarks.tools import pyzx as _pyzx
from benchmarks.tools import rival

ENV = "quizx"
PACKAGE = "quizx"


def supports(inst: Any, point: Any) -> str | None:
    """As PyZX."""
    why = rival.installed(ENV)
    if why:
        return why
    if not (inst.subject.general_d or int(inst.subject.dimension) == 2):
        return "qubits only"
    return None


def run(inst: Any, job: dict[str, Any]) -> dict[str, Any]:
    """One QuiZX run."""
    return _pyzx.zx_run("quizx", ENV, PACKAGE, inst, job)
