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

"""Named, nested timed sections (spans) of engine work.

Tracing is on when the environment variable ``ARCHYTASZX_TRACE`` is ``1`` at import. Off,
:func:`traced` returns the function itself and :func:`span` returns a shared null context
after one boolean check. On, calls with the same name and label under the same parent are
merged into one node holding CPU seconds, wall seconds and a call count.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, TypeVar

ON: bool = os.environ.get("ARCHYTASZX_TRACE") == "1"
_NULL: AbstractContextManager[None] = nullcontext()
F = TypeVar("F", bound=Callable[..., Any])


class Node:
    """One merged span: name, label, totals and children keyed by (name, label)."""

    __slots__ = ("calls", "children", "cpu", "label", "name", "wall")

    def __init__(self, name: str, label: str | None) -> None:
        self.name, self.label = name, label
        self.cpu = 0.0
        self.wall = 0.0
        self.calls = 0
        self.children: dict[tuple[str, str | None], Node] = {}

    def child(self, name: str, label: str | None) -> Node:
        key = (name, label)
        node = self.children.get(key)
        if node is None:
            node = self.children[key] = Node(name, label)
        return node

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "cpu": round(self.cpu, 6),
            "wall": round(self.wall, 6),
            "calls": self.calls,
        }
        if self.label is not None:
            out["label"] = self.label
        if self.children:
            out["children"] = [c.to_dict() for c in self.children.values()]
        return out


_root = Node("root", None)
_gen = 0
_lock = threading.RLock()
_local = threading.local()
_starts: dict[int, list[tuple[Node, float, float]]] = {}


def _stack() -> list[Node]:
    st = getattr(_local, "stack", None)
    if st is None or getattr(_local, "gen", -1) != _gen:
        st = _local.stack = [_root]
        _local.gen = _gen
        with _lock:
            _starts[threading.get_ident()] = []
    return st


@contextmanager
def _open(name: str, label: str | None) -> Iterator[None]:
    st = _stack()
    with _lock:
        node = st[-1].child(name, label)
        opened = _starts.setdefault(threading.get_ident(), [])
    st.append(node)
    c0, w0 = time.process_time(), time.perf_counter()
    opened.append((node, c0, w0))
    try:
        yield
    finally:
        c1, w1 = time.process_time(), time.perf_counter()
        st.pop()
        with _lock:
            if opened and opened[-1][0] is node:
                opened.pop()
            node.cpu += c1 - c0
            node.wall += w1 - w0
            node.calls += 1


def span(name: str, label: Any = None) -> AbstractContextManager[None]:
    """A timed section ``name`` (with an optional label) nested under the open one."""
    if not ON:
        return _NULL
    return _open(name, None if label is None else str(label))


def traced(name: str, label: Callable[..., Any] | None = None) -> Callable[[F], F]:
    """A decorator timing every call as span ``name``; ``label(*args, **kwargs)`` names it."""

    def wrap(fn: F) -> F:
        if not ON:
            return fn

        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            lab = label(*args, **kwargs) if label is not None else None
            with _open(name, None if lab is None else str(lab)):
                return fn(*args, **kwargs)

        return inner  # type: ignore[return-value]

    return wrap


def reset() -> None:
    """Drops every recorded span."""
    global _root, _gen
    with _lock:
        _root = Node("root", None)
        _gen += 1


def tree() -> list[dict[str, Any]]:
    """The recorded top-level spans with their children."""
    with _lock:
        return [c.to_dict() for c in _root.children.values()]


def snapshot() -> list[dict[str, Any]]:
    """:func:`tree` with every span still open given its partial CPU (process CPU, every
    thread) and wall seconds so far and ``"open": True``."""
    c1, w1 = time.process_time(), time.perf_counter()
    with _lock:
        partial = {
            id(node): (c1 - c0, w1 - w0) for opened in _starts.values() for node, c0, w0 in opened
        }

        def walk(node: Node) -> dict[str, Any]:
            out = node.to_dict()
            if id(node) in partial:
                dc, dw = partial[id(node)]
                out["cpu"] = round(node.cpu + dc, 6)
                out["wall"] = round(node.wall + dw, 6)
                out["open"] = True
            if node.children:
                out["children"] = [walk(c) for c in node.children.values()]
            return out

        return [walk(c) for c in _root.children.values()]


def open_path(spans: list[dict[str, Any]]) -> list[str]:
    """The names (with labels) of the deepest chain of open spans in a :func:`snapshot`."""
    for s in spans:
        if s.get("open"):
            name = s["name"] + (f"[{s['label']}]" if "label" in s else "")
            return [name, *open_path(s.get("children", []))]
    return []
