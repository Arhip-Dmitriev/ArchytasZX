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
