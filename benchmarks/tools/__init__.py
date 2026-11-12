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

"""Tool adapters: supports(instance, point) (None, or the reason the tool cannot run the
instance) and run(instance, job) (the run's fields)."""

from __future__ import annotations

import importlib
from typing import Any

NAMES = (
    "archytaszx",
    "pyzx",
    "quizx",
    "qcec",
    "feynman",
    "dizx",
    "qiskit",
    "cirq",
    "autoq_para",
    "tket",
    "quantomatic",
    "published_proofs",
)


def adapter(name: str) -> Any:
    """The adapter module of tool name."""
    if name not in NAMES:
        raise KeyError(f"unknown tool {name!r}")
    return importlib.import_module(f"benchmarks.tools.{name}")


HARNESS_IMPORTS = ("numpy", "sympy", "qiskit", "qiskit.qasm2", "qiskit.circuit.library")


def preflight(names: list[str]) -> list[str]:
    """The problems found importing the harness modules in-process and each enabled tool's
    PROBE modules in its environment (an environment not installed is not a problem)."""
    import subprocess

    from benchmarks.tools import rival

    problems = []
    for mod in HARNESS_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"harness: import {mod}: {type(exc).__name__}: {exc}")
    for name in names:
        a = adapter(name)
        env, probe = getattr(a, "ENV", None), getattr(a, "PROBE", ())
        if not probe or env is None or not rival.python(env).exists():
            continue
        try:
            r = subprocess.run(
                [str(rival.python(env)), "-I", "-c", "import " + ", ".join(probe)],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if r.returncode != 0:
                last = (r.stderr.strip().splitlines() or ["exit " + str(r.returncode)])[-1]
                problems.append(f"{name}: benchmarks/.venvs/{env}: {last}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"{name}: benchmarks/.venvs/{env}: {type(exc).__name__}: {exc}")
    return problems
