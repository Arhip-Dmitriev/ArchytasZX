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

"""The benchmark configuration: benchmarks/config.toml with ``--set key=value`` overrides,
resolved and hashed."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Path(__file__).resolve().parent / "config.toml"
UNHASHED = (("platform", "jobs"), ("output", "results"), ("output", "plots"), ("tools", "enabled"))


def _value(text: str) -> Any:
    """A TOML literal, else the text as a string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def override(cfg: dict[str, Any], assignment: str) -> None:
    """Applies one ``a.b.c=value`` in place."""
    key, sep, text = assignment.partition("=")
    if not sep:
        raise ValueError(f"--set needs key=value, got {assignment!r}")
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown config section {key!r}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key {key!r}")
    node[parts[-1]] = _value(text.strip())


def load(
    path: Path | None = None, sets: list[str] | None = None, profile: str | None = None
) -> dict[str, Any]:
    """The resolved config: the file, then ``sets``, then ``profile``; jobs resolved."""
    with open(path or DEFAULT, "rb") as fh:
        cfg = tomllib.load(fh)
    for s in sets or []:
        override(cfg, s)
    if profile is not None:
        cfg["platform"]["profile"] = profile
    if cfg["platform"]["jobs"] == "auto":
        cfg["platform"]["jobs"] = max(1, (os.cpu_count() or 2) // 2)
    return cfg


def repeats(cfg: dict[str, Any]) -> dict[str, Any]:
    """The repeat section of the active profile."""
    return cfg["repeats"][cfg["platform"]["profile"]]


def digest(cfg: dict[str, Any]) -> str:
    """The config hash keying runs: every value but :data:`UNHASHED`."""
    c = copy.deepcopy(cfg)
    for sec, key in UNHASHED:
        c.get(sec, {}).pop(key, None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def path(cfg: dict[str, Any], section: str) -> Path:
    """An output directory of the config, absolute."""
    p = Path(cfg["output"][section])
    return p if p.is_absolute() else ROOT / p
