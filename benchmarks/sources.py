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

"""Source digests of benchmark instances: what a run's key and an aggregate's filter compare."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.subject import ROOT, Instance


@functools.cache
def _shared_digest(directory: str) -> bytes:
    """Hash of the catalogue's helper modules (``_*.py``), its data subdirectories and the IR,
    translator and subject modules."""
    base = Path(directory)
    base = base if base.is_absolute() else ROOT / base
    h = hashlib.sha256()
    data = [p for p in base.rglob("*") if p.is_file() and p.parent != base]
    files = sorted(base.glob("_*.py")) + sorted(p for p in data if "__pycache__" not in p.parts)
    here = Path(__file__).parent
    files += [here / "ir.py", here / "translate.py", here / "subject.py"]
    for p in files:
        h.update(str(p.relative_to(base) if p.is_relative_to(base) else p.name).encode())
        h.update(p.read_bytes())
    return h.digest()


def source_digest(
    inst: Instance, false: dict[str, Any] | None = None, directory: str = "benchmarks/subjects"
) -> str:
    """Hash of everything an instance's runs are built from: its subject file and the shared
    sources of :func:`_shared_digest`; for an F instance also breaks.py and its F record."""
    base = Path(directory)
    base = base if base.is_absolute() else ROOT / base
    h = hashlib.sha256(_shared_digest(directory))
    h.update((base / f"{inst.subject.id}.py").read_bytes())
    if false is not None:
        h.update((Path(__file__).parent / "breaks.py").read_bytes())
        h.update(json.dumps(false, sort_keys=True, default=str).encode())
    return h.hexdigest()[:10]
