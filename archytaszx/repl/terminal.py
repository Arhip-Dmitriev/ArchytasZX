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

"""Terminal images and display settings: image protocol and theme detection, the
configuration file, PNG rendering of mathtext lines and their inline escape sequences."""

from __future__ import annotations

import base64
import io
import itertools
import math
import os
import re
import select
import shutil
import sys
import threading
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass

PROTOCOLS = ("kitty", "iterm", "sixel", "none")
"""Terminal image protocols; ``ARCHYTASZX_IMAGES`` selects one explicitly."""

_SIXEL_LEVELS = 16


def protocol(environ: Mapping[str, str] | None = None, tty: bool | None = None) -> str:
    """The image protocol of the terminal on standard output, ``none`` when unsupported."""
    env = os.environ if environ is None else environ
    forced = env.get("ARCHYTASZX_IMAGES", "").lower()
    if forced in PROTOCOLS:
        return forced
    if not (sys.stdout.isatty() if tty is None else tty):
        return "none"
    term, program = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    if "KITTY_WINDOW_ID" in env or term in ("xterm-kitty", "xterm-ghostty") or program == "ghostty":
        return "kitty"
    if program in ("iTerm.app", "WezTerm", "vscode") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    if "sixel" in term or term.startswith(("foot", "mlterm")) or program == "mlterm":
        return "sixel"
    return "none"


def dark(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the terminal background is dark: ``ARCHYTASZX_IMAGE_THEME``, then ``COLORFGBG``."""
    env = os.environ if environ is None else environ
    theme = env.get("ARCHYTASZX_IMAGE_THEME", "").lower()
    if theme in ("dark", "light"):
        return theme == "dark"
    bg = env.get("COLORFGBG", "").rpartition(";")[2]
    return not (bg.isdigit() and int(bg) in (7, 15))


@dataclass
class Terminal:
    """What the terminal reported: foreground and background as ``#rrggbb`` and the character
    cell height and width in pixels, each None when unknown."""

    fg: str | None = None
    bg: str | None = None
    cell: int | None = None
    cell_width: int | None = None


TERMINAL = Terminal()
"""The terminal facts :func:`prepare` found."""

SAVE_DIR: str | None = None
"""The folder every rendered image is also written to, or None."""
SAVE_WHITE = False
"""Whether saved images are black on white instead of the terminal's colours."""
WHITE = ("#000000", "#ffffff")
"""Black ink on a white page, as ``(fg, bg)``."""
_SAVED: list[int] = [0]


def keep(png: bytes, kind: str) -> str | None:
    """Writes ``png`` to :data:`SAVE_DIR` as ``NNNN-kind.png``; returns the path, None without
    a folder."""
    if SAVE_DIR is None:
        return None
    os.makedirs(SAVE_DIR, exist_ok=True)
    _SAVED[0] += 1
    path = os.path.join(SAVE_DIR, f"{_SAVED[0]:04d}-{kind}.png")
    with open(path, "wb") as handle:
        handle.write(png)
    return path


_DPI = 100
_DENSITY = 2
"""Pixels rendered per displayed pixel for a mathtext image."""
_MAX_PIXELS = 12000
"""The longest side, in pixels, of a rendered image."""
_WARM: list[threading.Thread] = []


@dataclass
class Config:
    """Image settings: ``size`` scales the text (1 = the terminal's cell height) and ``color``
    is ``auto`` (the terminal's text colour) or a colour such as ``#ff8800``."""

    size: float = 1.0
    color: str = "auto"


CONFIG = Config()
"""The settings :func:`render_png` uses."""

_HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})")


def config_path(environ: Mapping[str, str] | None = None) -> str:
    """``$XDG_CONFIG_HOME/archytaszx/config.toml``, ``~/.config`` without XDG_CONFIG_HOME."""
    env = os.environ if environ is None else environ
    base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "archytaszx", "config.toml")


def set_config(config: Config, key: str, value: object) -> None:
    """Sets one setting; ValueError on an unknown key or a bad value."""
    if key == "size":
        try:
            size = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(f"size must be a number, got {value!r}") from None
        if not 0 < size <= 10:
            raise ValueError("size must be in (0, 10]")
        config.size = size
    elif key == "color":
        text = str(value).strip()
        if text != "auto" and not _HEX.fullmatch(text) and not _named_colour(text):
            raise ValueError(f"color must be auto, #rgb, #rrggbb or a colour name, got {text!r}")
        config.color = text
    else:
        raise ValueError(f"unknown setting {key!r}; settings: size, color")


def _named_colour(text: str) -> bool:
    try:
        from matplotlib.colors import is_color_like
    except ImportError:
        return False
    return bool(is_color_like(text))


def load_config(path: str | None = None) -> Config:
    """The settings in the TOML file at ``path`` (default :func:`config_path`); defaults when it
    is missing; ValueError when it is unreadable or holds a bad setting."""
    path = config_path() if path is None else path
    config = Config()
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        return config
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    for key, value in data.items():
        set_config(config, key, value)
    return config


def save_config(config: Config, path: str | None = None) -> str:
    """Writes ``config`` as TOML to ``path`` (default :func:`config_path`); returns the path."""
    path = config_path() if path is None else path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f'size = {config.size!r}\ncolor = "{config.color}"\n')
    return path


def _query(fd: int, request: str, pattern: str) -> re.Match[str] | None:
    """The terminal's reply to ``request`` matching ``pattern``, read for at most 0.2 s."""
    os.write(fd, request.encode())
    reply, deadline = "", time.monotonic() + 0.2
    while (left := deadline - time.monotonic()) > 0:
        if not select.select([fd], [], [], left)[0]:
            break
        reply += os.read(fd, 256).decode(errors="replace")
        found = re.search(pattern, reply)
        if found:
            return found
    return None


def _rgb(found: re.Match[str] | None) -> str | None:
    if found is None:
        return None
    return "#" + "".join(f"{int(h, 16) * 255 // (16 ** len(h) - 1):02x}" for h in found.groups())


def query_terminal() -> Terminal:
    """Asks the controlling terminal for its colours (OSC 10/11) and cell size (CSI 16 t)."""
    try:
        import termios
        import tty

        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except (ImportError, OSError):
        return Terminal()
    try:
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            rgb = r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)"
            fg = _rgb(_query(fd, "\x1b]10;?\x1b\\", "\x1b]10;" + rgb))
            bg = _rgb(_query(fd, "\x1b]11;?\x1b\\", "\x1b]11;" + rgb))
            cell = _query(fd, "\x1b[16t", r"\x1b\[6;(\d+);(\d+)t")
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, saved)
    except (termios.error, OSError):
        return Terminal()
    finally:
        os.close(fd)
    height = int(cell.group(1)) if cell and int(cell.group(1)) > 0 else None
    width = int(cell.group(2)) if cell and int(cell.group(2)) > 0 else None
    return Terminal(fg, bg, height, width)


def _warm() -> None:
    render_png([r"\sum_{x=0}^{q-1} |x\rangle"], dark_theme=True)


def prepare(environ: Mapping[str, str] | None = None) -> None:
    """Reads :data:`TERMINAL` and loads matplotlib in the background when the terminal shows
    images."""
    global TERMINAL
    if protocol(environ) == "none":
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        TERMINAL = query_terminal()
    if not _WARM:
        thread = threading.Thread(target=_warm, daemon=True)
        _WARM.append(thread)
        thread.start()


def colours(dark_theme: bool, environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """``(fg, bg)``: ``ARCHYTASZX_IMAGE_FG``, then the terminal's own, then a theme default."""
    env = os.environ if environ is None else environ
    fg, bg = ("#e6e6e6", "#1e1e1e") if dark_theme else ("#111111", "#ffffff")
    return env.get("ARCHYTASZX_IMAGE_FG") or TERMINAL.fg or fg, TERMINAL.bg or bg


def render_png(
    lines: list[str],
    *,
    dark_theme: bool,
    opaque: bool = False,
    density: float = 1,
    palette: tuple[str, str] | None = None,
) -> bytes:
    """Off-screen PNG of mathtext lines, left aligned, each row as tall as its ink plus a gap
    of a third of the font size; the font matches the terminal cell height when known, times
    :data:`CONFIG` size, in :data:`CONFIG` color unless that is ``auto``, or in ``palette``
    ``(fg, bg)`` when given; ``density`` pixels per displayed pixel."""
    if _WARM and _WARM[0] is not threading.current_thread():
        _WARM[0].join()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fg, bg = colours(dark_theme)
    if CONFIG.color != "auto":
        fg = CONFIG.color
    if palette is not None:
        fg, bg = palette
    px = (0.75 * TERMINAL.cell if TERMINAL.cell else 14.0) * CONFIG.size
    size = px * 72 / _DPI
    fig = Figure(dpi=_DPI)
    canvas = FigureCanvasAgg(fig)
    renderer = canvas.get_renderer()  # type: ignore[no-untyped-call]
    texts = [
        fig.text(0, 0, f"${line.strip()}$", fontsize=size, color=fg, va="bottom") for line in lines
    ]
    boxes = [t.get_window_extent(renderer) for t in texts]
    pad, gap = px / 3, px / 3
    width = max(b.width for b in boxes) + 2 * pad
    height = sum(b.height for b in boxes) + gap * (len(boxes) - 1) + 2 * pad
    fig.set_size_inches(width / _DPI, height / _DPI)
    y = height - pad
    for t, b in zip(texts, boxes, strict=True):
        y -= b.height
        t.set_position((pad / width, y / height))
        y -= gap
    dpi = _DPI * min(density, _MAX_PIXELS / max(width, height))
    out = io.BytesIO()
    fig.savefig(
        out, format="png", dpi=dpi, transparent=not opaque, facecolor=bg if opaque else "none"
    )
    return out.getvalue()


def cells(png: bytes, density: float) -> tuple[int, int] | None:
    """The terminal ``(columns, rows)`` that show ``png`` at 1/``density`` of its pixel size
    when the cell size is known, capped at the terminal width keeping the aspect ratio."""
    if not (TERMINAL.cell and TERMINAL.cell_width) or len(png) < 24:
        return None
    width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    cols = width / density / TERMINAL.cell_width
    rows = height / density / TERMINAL.cell
    room = max(10, shutil.get_terminal_size().columns - 1)
    if cols > room:
        cols, rows = room, rows * room / cols
    return max(1, math.ceil(cols)), max(1, math.ceil(rows))


def _kitty(png: bytes, size: tuple[int, int] | None = None) -> str:
    data = base64.standard_b64encode(png).decode("ascii")
    chunks = [data[k : k + 4096] for k in range(0, len(data), 4096)] or [""]
    fit = f",c={size[0]},r={size[1]}" if size else ""
    out = []
    for k, chunk in enumerate(chunks):
        more = 1 if k < len(chunks) - 1 else 0
        head = f"a=T,f=100{fit},m={more}" if k == 0 else f"m={more}"
        out.append(f"\x1b_G{head};{chunk}\x1b\\")
    return "".join(out)


def _iterm(png: bytes, size: tuple[int, int] | None = None, density: float = 1) -> str:
    data = base64.standard_b64encode(png).decode("ascii")
    if size:
        fit = f";width={size[0]};height={size[1]}"
    elif density != 1 and len(png) >= 24:
        width, height = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
        fit = f";width={round(width / density)}px;height={round(height / density)}px"
    else:
        fit = ""
    return f"\x1b]1337;File=inline=1;size={len(png)}{fit};preserveAspectRatio=1:{data}\x07"


def _sixel(png: bytes, size: tuple[int, int] | None = None) -> str:
    """Grey-quantised sixel image."""
    import matplotlib.image as mpimg
    import numpy as np

    rgba = mpimg.imread(io.BytesIO(png), format="png")
    grey = rgba[..., :3].mean(axis=2)
    idx = np.rint(grey * (_SIXEL_LEVELS - 1)).astype(int)
    height, width = idx.shape
    palette = "".join(
        f"#{c};2;{round(100 * c / (_SIXEL_LEVELS - 1))};"
        f"{round(100 * c / (_SIXEL_LEVELS - 1))};{round(100 * c / (_SIXEL_LEVELS - 1))}"
        for c in range(_SIXEL_LEVELS)
    )
    weights = (1 << np.arange(6))[:, None]
    bands = []
    for y0 in range(0, height, 6):
        band = idx[y0 : y0 + 6]
        rows = []
        for c in np.unique(band):
            mask = band == c
            codes = (mask * weights[: len(band)]).sum(axis=0)
            text = []
            for ch, run in itertools.groupby(chr(63 + int(v)) for v in codes):
                n = len(list(run))
                text.append(f"!{n}{ch}" if n > 3 else ch * n)
            rows.append(f"#{c}" + "".join(text))
        bands.append("$".join(rows))
    return f'\x1bPq"1;1;{width};{height}{palette}' + "-".join(bands) + "\x1b\\"


def encode_image(png: bytes, proto: str | None = None, density: float = 1) -> str | None:
    """``png`` as inline-image escape text for ``proto`` (default :func:`protocol`), shown at
    1/``density`` of its pixel size where the protocol can scale; None for ``none``."""
    proto = protocol() if proto is None else proto
    if proto == "none":
        return None
    size = cells(png, density) if proto != "sixel" else None
    if proto == "iterm":
        return _iterm(png, size, density)
    return {"kitty": _kitty, "sixel": _sixel}[proto](png, size)


def scalable(proto: str) -> bool:
    """Whether images on ``proto`` can be shown smaller than their pixel size."""
    return proto == "iterm" or (proto == "kitty" and bool(TERMINAL.cell_width))
