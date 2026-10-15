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

"""The Python embedding of the Dirac language: one :class:`Program` per interpreter.

::

    from archytaszx import azx               # or azx = archytaszx.Program()
    azx("var n")                             # run statements; returns what they printed
    azx.psi = azx.H ** azx.n | azx.ket(0) ** azx.n     # |psi> = H^@n |0>^@n
    azx["|phi>"] = azx["|psi> - 2 |0><0|psi>"]         # expressions in the language syntax
    y = azx.measure | azx.psi                # or azx.measure(azx.psi)
    azx.A | azx.psi                          # A applied to |psi>; <a| | |b> is <a|b>

A :class:`Value` wraps a state, operator or scalar. ``|`` is juxtaposition (application,
inner and outer products, tensor of kets), ``@`` the tensor product, ``**`` the tensor power
of states and operators (the power of scalars), ``+ - * /`` the arithmetic and ``.dag()`` the
conjugate transpose. Python numbers become exact sympy numbers. Attribute and item
assignment name a state, operator or variable in the program. Measurement needs a named
state and collapses it.
"""

from __future__ import annotations

from collections.abc import Iterable

import sympy  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from archytaszx.repl import display as D
from archytaszx.repl import lang as L
from archytaszx.semantics import pathsum as P

_OWN = frozenset({"_interp", "_count"})


class _BraOp:
    """``<a| U``, waiting for the ket it is closed with."""

    def __init__(self, bra: P.State, op: P.Operator) -> None:
        self.bra = bra
        self.op = op


class Value:
    """A value of a :class:`Program`; ``name`` is set for a named state."""

    __slots__ = ("name", "program", "value")

    def __init__(self, program: Program, value: object, name: str | None = None) -> None:
        self.program = program
        self.value = value
        self.name = name

    def _wrap(self, value: object) -> Value:
        return Value(self.program, value)

    def _binary(self, op: str, other: object, swap: bool = False) -> Value:
        a, b = self.value, self.program._raw(other)
        if swap:
            a, b = b, a
        interp = self.program._interp
        try:
            if op == "|" and isinstance(a, _BraOp):
                if isinstance(b, P.Operator):
                    return self._wrap(_BraOp(a.bra, P.Compose((a.op, b))))
                return self._wrap(interp._juxt(a.bra, interp._juxt(a.op, b)))
            if op == "|" and isinstance(a, P.State) and a.bra and isinstance(b, P.Operator):
                return self._wrap(_BraOp(a, b))
            if op == "|":
                return self._wrap(interp._juxt(a, b))
            return self._wrap(interp._binary(op, a, b))
        except P.PathSumError as exc:
            raise L.LangError(str(exc)) from exc

    def __add__(self, other: object) -> Value:
        return self._binary("+", other)

    def __radd__(self, other: object) -> Value:
        return self._binary("+", other, True)

    def __sub__(self, other: object) -> Value:
        return self._binary("-", other)

    def __rsub__(self, other: object) -> Value:
        return self._binary("-", other, True)

    def __mul__(self, other: object) -> Value:
        return self._binary("*", other)

    def __rmul__(self, other: object) -> Value:
        return self._binary("*", other, True)

    def __truediv__(self, other: object) -> Value:
        return self._binary("/", other)

    def __rtruediv__(self, other: object) -> Value:
        return self._binary("/", other, True)

    def __matmul__(self, other: object) -> Value:
        return self._binary("@", other)

    def __rmatmul__(self, other: object) -> Value:
        return self._binary("@", other, True)

    def __or__(self, other: object) -> Value:
        return self._binary("|", other)

    def __ror__(self, other: object) -> Value:
        return self._binary("|", other, True)

    def __pow__(self, other: object) -> Value:
        op = "^" if isinstance(self.value, sympy.Basic) else "^@"
        return self._binary(op, other)

    def __rpow__(self, other: object) -> Value:
        return self._binary("^", other, True)

    def __neg__(self) -> Value:
        return self._binary("*", -1)

    def __pos__(self) -> Value:
        return self

    def dag(self) -> Value:
        """The conjugate transpose of a state."""
        if not isinstance(self.value, P.State):
            raise L.LangTypeError("dag() takes a state")
        return self._wrap(P.dagger(self.value))

    def text(self, style: str = "unicode") -> str:
        """The value in a display style: unicode, ascii or latex."""
        return self.program.render(self, style=style)

    def latex(self) -> str:
        return self.text("latex")

    def unicode(self) -> str:
        return self.text("unicode")

    def ascii(self) -> str:
        return self.text("ascii")

    def _repr_latex_(self) -> str:
        return self.latex()

    def __str__(self) -> str:
        return self.unicode()

    def __repr__(self) -> str:
        if isinstance(self.value, _BraOp):
            return "<azx bra-operator>"
        return f"<azx {self.program._interp._type(self.value)}: {self.ascii()}>"

    def sympy(self) -> sympy.Basic:
        """The sympy expression of a scalar."""
        if not isinstance(self.value, sympy.Basic):
            raise L.LangTypeError("sympy() takes a scalar")
        return self.value

    def __complex__(self) -> complex:
        return complex(self.sympy())

    def __float__(self) -> float:
        return float(self.sympy())

    def __int__(self) -> int:
        return int(self.sympy())

    def __bool__(self) -> bool:
        v = self.value
        if isinstance(v, (bool, sympy.logic.boolalg.BooleanAtom)):
            return bool(v)
        if isinstance(v, sympy.Basic) and v.is_number:
            return bool(v != 0)
        return True

    def vector(self) -> object:
        """The amplitudes of a concrete state as a complex numpy vector."""
        import numpy as np

        s = self.value
        if not isinstance(s, P.State) or not P.is_concrete(s):
            raise L.LangTypeError("vector() takes a concrete state")
        keys, amps = P.enumerate_state(s)
        out = np.zeros(int(np.prod(P.concrete_levels(s))), dtype=complex)
        np.add.at(out, keys.astype(np.int64), amps)
        return out


class _Measure:
    """``program.measure(psi)`` or ``program.measure | psi``."""

    def __init__(self, program: Program) -> None:
        self.program = program

    def __call__(self, state: Value | str) -> object:
        return self.program._measure(state)

    def __or__(self, state: Value | str) -> object:
        return self.program._measure(state)


class Program:
    """A Dirac-language interpreter driven from Python (see the module docstring)."""

    def __init__(self, *, seed: int | None = None, style: str = "unicode") -> None:
        object.__setattr__(self, "_interp", L.Interpreter(style=style, seed=seed))
        object.__setattr__(self, "_count", 0)
        self._interp: L.Interpreter
        self._count: int

    @property
    def interpreter(self) -> L.Interpreter:
        return self._interp

    @property
    def measure(self) -> _Measure:
        return _Measure(self)

    def __call__(self, source: str) -> str:
        """Runs statements; returns what ``show`` and ``print`` printed."""
        return self._interp.execute(source)

    run = __call__

    def __getitem__(self, source: str) -> Value:
        """The value of an expression in the language syntax."""
        stripped = source.strip()
        if L._IDENT.fullmatch(stripped):
            named: Value = getattr(self, stripped)
            return named
        p = self._interp._parser(L.tokenize(stripped))
        node = p.expr()
        if not p.done():
            raise L.LangSyntaxError(f"cannot read {source!r} as one expression")
        name = None
        if isinstance(node, L.Braket) and not node.bra:
            content = node.content.strip()
            name = content if content in self._interp.states else None
        return Value(self, self._interp._eval(node, L.Ctx()), name)

    def __setitem__(self, target: str, value: object) -> None:
        """``|name>``, ``<name|`` or a bare name assigned a value."""
        text = target.strip()
        if text.startswith("|") and text.endswith(">"):
            self._state(text[1:-1].strip(), value, bra=False)
        elif text.startswith("<") and text.endswith("|"):
            self._state(text[1:-1].strip(), value, bra=True)
        else:
            setattr(self, text, value)

    def __getattr__(self, name: str) -> Value:
        if name.startswith("_"):
            raise AttributeError(name)
        interp = self._interp
        if name in interp.states:
            return Value(self, interp.states[name], name)
        if name in interp.ops:
            return Value(self, interp.ops[name])
        if name in interp.values:
            return Value(self, interp._name(name, L.Ctx()))
        if name in L._CONSTANTS:
            return Value(self, L._CONSTANTS[name])
        if name in interp.funcs:
            fn = interp.funcs[name]
            return Value(self, fn)
        raise AttributeError(f"{name!r} is not defined in this program")

    def __setattr__(self, name: str, value: object) -> None:
        if name in _OWN:
            object.__setattr__(self, name, value)
            return
        raw = self._raw(value)
        interp = self._interp
        if isinstance(raw, P.State):
            self._state(name, raw, bra=raw.bra)
        elif isinstance(raw, P.Operator):
            interp._check_name(name, "op")
            interp.ops[name] = raw
        elif isinstance(raw, sympy.Basic):
            interp._check_name(name, "var")
            interp._forget(name)
            interp.values[name] = raw
            if raw.free_symbols and not raw.is_Symbol:
                interp.sugar.append((name, raw))
        else:
            raise L.LangTypeError(f"cannot name a {type(raw).__name__}")

    def _state(self, name: str, value: object, *, bra: bool) -> None:
        raw = self._raw(value)
        if not isinstance(raw, P.State):
            raise L.LangTypeError(f"|{name}> must be assigned a state")
        if raw.bra != bra:
            raise L.LangTypeError(f"cannot assign a {'bra' if raw.bra else 'ket'} to {name}")
        interp = self._interp
        interp._check_name(name, "state")
        if any(w is None for w in raw.widths):
            raise L.LangTypeError(f"cannot infer every register width of |{name}>")
        state = L._flex(raw, ())
        interp.states[name] = P.simplify(P.dagger(state) if bra else state)

    def _raw(self, value: object) -> object:
        """The interpreter value of a Value, a Python number or a sympy expression."""
        if isinstance(value, Value):
            return value.value
        if isinstance(value, (P.State, P.Operator)):
            return value
        if isinstance(value, bool):
            return sympy.true if value else sympy.false
        if isinstance(value, (int, float, complex, sympy.Basic)):
            return sympy.nsimplify(value) if isinstance(value, float) else sympy.sympify(value)
        if isinstance(value, str):
            return self[value].value
        raise L.LangTypeError(f"cannot use a {type(value).__name__} as a value")

    def ket(self, value: object, dim: object = None) -> Value:
        """``|value>``, ``|value>_dim`` with a dimension."""
        suffix = "" if dim is None else f"_{dim}"
        return self[f"|{value}>{suffix}"]

    def bra(self, value: object, dim: object = None) -> Value:
        """``<value|``, ``<value|_dim`` with a dimension."""
        suffix = "" if dim is None else f"_{dim}"
        return self[f"<{value}|{suffix}"]

    def var(self, *names: str, type: str = "nat") -> Value | tuple[Value, ...]:
        """Declares symbolic variables; returns them."""
        self._interp.execute(f"var {', '.join(f'{n}: {type}' for n in names)}")
        out = tuple(getattr(self, n) for n in names)
        return out[0] if len(out) == 1 else out

    def seed(self, value: int) -> None:
        """Seeds measurement and ``rand``."""
        self._interp.execute(f"seed {int(value)}")

    def _measure(self, state: Value | str) -> object:
        """Measures a named state in place; returns the outcome (bool, Bits or a symbol)."""
        if isinstance(state, str):
            state = self[state if state.startswith("|") else f"|{state}>"]
        if not isinstance(state, Value) or not isinstance(state.value, P.State):
            raise L.LangTypeError("measure takes a state")
        name = state.name
        if name is None or self._interp.states.get(name) is not state.value:
            self._count += 1
            name = f"_m{self._count}"
            self._state(name, state, bra=False)
        p = self._interp._parser(L.tokenize(f"|{name}>"))
        out = self._interp._measure(p, None)
        if isinstance(out, sympy.Basic):
            return Value(self, out)
        return out

    def render(self, *values: object, style: str = "unicode") -> str:
        """The values in a display style, one line each."""
        if style not in D.STYLES or style == "plot":
            raise ValueError(f"unknown text style {style!r}")
        entries: list[tuple[D.Label | None, object]] = []
        for v in values:
            label = None
            if isinstance(v, Value) and v.name is not None:
                label = self._interp._label("ket", v.name)
            entries.append((label, self._raw(v)))
        return self._interp.render(entries, style)

    def show(self, *values: object, style: str = "unicode") -> None:
        """Prints the values in a display style."""
        print(self.render(*values, style=style))

    def names(self) -> Iterable[str]:
        """Every user-defined name."""
        interp = self._interp
        own = set(interp.states) | set(interp.values) | set(interp.funcs) | set(interp.lists)
        return sorted(own | (set(interp.ops) - interp.builtins))
