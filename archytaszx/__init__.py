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

"""ArchytasZX: a qufinite ZX-calculus engine with symbolic qudit count and dimension."""

from typing import Any


def __getattr__(name: str) -> Any:
    """``Program`` and ``azx`` (a shared default Program) from :mod:`archytaszx.embed`."""
    if name not in ("Program", "Value", "azx"):
        raise AttributeError(f"module 'archytaszx' has no attribute {name!r}")
    from archytaszx import embed

    if name == "azx":
        if "azx" not in globals():
            globals()["azx"] = embed.Program()
        return globals()["azx"]
    return getattr(embed, name)
