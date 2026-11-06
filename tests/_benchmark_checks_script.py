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

"""Correctness checks of the benchmarks themselves (``python -m tests._benchmark_checks_script``):

- every true family and breadth workload: its target (``@rhs``) evaluated by the engine's
  numeric oracle against the target computed here from the algorithm's definition, at one
  small instance; and the left side against the same target;
- every buggy variant of the refutations suite: false at one small instance (oracle);
- every PyZX and Qiskit translation of the families suite at n = 1, 2: the tool's tensor
  against the oracle's evaluation of the ArchytasZX workload at d = 2 (PyZX exactly, Qiskit
  up to normalisation); the DiZX gate lists at n = 1, 2: their matrix under DiZX's gate
  definitions (HAD the Fourier matrix, CX |a, b> -> |a, a + b>, Z |k> -> w^k |k>) against the
  oracle's evaluation of the circuit's first half at d = 3.

Axis order everywhere: boundary outputs, then boundary inputs (SEMANTICS.txt, 2.1).
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from benchmarks import breadth, engine, families, qudits, refutations, size, workloads

TOL = 1e-9


def ket(m: int, d: int) -> Any:
    """The basis state |m mod d>."""
    v = np.zeros(d, dtype=complex)
    v[m % d] = 1
    return v


def fourier(d: int) -> Any:
    """F[j][k] = w^(j k) / sqrt(d)."""
    w = np.exp(2j * np.pi / d)
    return np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)


def kron(*vs: np.ndarray) -> Any:
    """The tensor product as an array with one axis per factor."""
    out = np.array(1, dtype=complex)
    for v in vs:
        out = np.multiply.outer(out, v)
    return out


def uniform(d: int) -> Any:
    """sum_k |k>."""
    return np.ones(d, dtype=complex)


def ghz(d: int, legs: int) -> Any:
    """sum_k |k>^legs."""
    t = np.zeros((d,) * legs, dtype=complex)
    for k in range(d):
        t[(k,) * legs] = 1
    return t


def identity(d: int) -> Any:
    """The identity as (output, input)."""
    return np.eye(d, dtype=complex)


def family_target(family: str, n: int, d: int) -> Any:
    """The family's output from the algorithm's definition (examples' headers)."""
    F = fourier(d)
    if family == "state copy":
        return np.sqrt(d) * kron(*[ket(0, d)] * n)
    if family == "spider fusion":
        return ghz(d, n)
    if family == "Fourier order 4":
        out = np.array(1, dtype=complex)
        for _ in range(n):
            out = np.multiply.outer(out, identity(d))
        perm = [2 * i for i in range(n)] + [2 * i + 1 for i in range(n)]
        return out.transpose(perm) if n else out
    if family == "Fourier superposition":
        return d ** (-n / 2) * kron(*[uniform(d)] * n)
    if family == "Bernstein-Vazirani":
        return kron(*[ket(1, d)] * n)
    if family == "Deutsch-Jozsa constant":
        w = np.exp(2j * np.pi / d)
        target = F @ F @ F @ ket(1, d)
        out = np.zeros((d,) * (n + 2), dtype=complex)
        for c in range(d):
            out[(..., c)] = w**c * kron(*[ket(0, d)] * n, target)
        return out
    if family == "Deutsch-Jozsa balanced":
        return kron(*[ket(0, d)] * n, ket(1, d), F @ F @ F @ ket(1, d))
    if family == "GHZ":
        return d ** (-0.5) * ghz(d, n + 1)
    raise ValueError(family)


def breadth_target(name: str, v: dict[str, int]) -> Any:
    """The breadth workload's target from its definition, at the values ``v``."""
    d = v.get("d", 3)
    if name == "teleportation":
        return identity(d) / d
    if name == "entanglement_swapping":
        return d ** (-1.5) * ghz(d, 2)
    if name == "superdense_coding":
        return kron(ket(v["b"], d), ket(v["a"], d))
    if name == "w_state_zero":
        return kron(*[ket(0, d)] * v["n"])
    if name == "w_state_weight":
        return np.array(v["n"] * (d - 1 + 1j))
    if name in ("bernstein_vazirani_secret", "fourier_sampling"):
        m = v["s"] if name == "bernstein_vazirani_secret" else v["t"]
        return kron(*[ket(m, d)] * v["n"])
    if name == "phase_estimation":
        return kron(ket(v["t"], d), ket(v["t"], d))
    if name == "repetition_code":
        t = np.multiply.outer(kron(*[ket(0, d)] * v["n"]), identity(d))
        return np.moveaxis(t, -2, 0)
    if name == "grover_two_qubits":
        return kron(ket(1, 2), ket(1, 2))
    if name == "qufinite_round_trip":
        return identity(v["s"] * v["t"])
    if name == "qufinite_pair_state":
        return uniform(v["d"] * v["e"])
    if name == "qufinite_two_dims":
        return kron(uniform(d), uniform(v["e"])) / np.sqrt(d * v["e"])
    if name == "two_counts_two_dims":
        e = v["e"]
        return kron(*[uniform(d) / np.sqrt(d)] * v["n"], *[uniform(e) / np.sqrt(e)] * v["m"])
    if name.startswith("nested_depth"):
        names = ["n", "m", "p"][: int(name[-1])]
        count = sum(int(np.prod([v[x] for x in names[: i + 1]])) for i in range(len(names)))
        return kron(*[uniform(d) / np.sqrt(d)] * count)
    raise ValueError(name)


BREADTH_POINTS: dict[str, dict[str, int]] = {
    "teleportation": {"d": 3, "a": 1, "b": 2},
    "entanglement_swapping": {"d": 3},
    "superdense_coding": {"d": 3, "a": 1, "b": 2},
    "w_state_zero": {"d": 3, "n": 2},
    "w_state_weight": {"d": 3, "n": 2},
    "bernstein_vazirani_secret": {"d": 3, "n": 2, "s": 2},
    "fourier_sampling": {"d": 3, "n": 2, "t": 2},
    "phase_estimation": {"d": 3, "t": 2},
    "repetition_code": {"d": 3, "n": 2},
    "grover_two_qubits": {},
    "qufinite_round_trip": {"s": 2, "t": 3},
    "qufinite_pair_state": {"d": 2, "e": 3},
    "qufinite_two_dims": {"d": 2, "e": 3},
    "two_counts_two_dims": {"d": 2, "e": 3, "n": 2, "m": 1},
    "nested_depth1": {"d": 2, "n": 2},
    "nested_depth2": {"d": 2, "n": 2, "m": 1},
    "nested_depth3": {"d": 2, "n": 1, "m": 2, "p": 1},
}


def _tensor(diagram: Any, values: dict[str, int]) -> Any:
    from archytaszx.semantics.check import score

    return np.asarray(score(diagram, values).tensor)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, atol=TOL))


def _pair(script: str, base: str) -> tuple[Any, Any]:
    from archytaszx.repl.commands import _pair_args

    text, line = workloads.split_verdict(script)
    session = engine.run_script(text, base=base).session
    args = line.split(None, 1)[1] if " " in line else ""
    return _pair_args(session, args.removeprefix("exhaustive").strip())


def check_targets(report: Callable[[str, bool, str], None]) -> None:
    """Family and breadth targets against their definitions."""
    for family in workloads.FAMILIES:
        n, d = 2, 3
        left, right = _pair(workloads.family_script(family), str(workloads.ROOT))
        symbol = workloads.FAMILIES[family][0]
        values = {symbol: n, "d": d}
        want = family_target(family, n, d)
        report(f"target families/{family}", _same(_tensor(right, values), want), str(values))
        report(f"claim families/{family}", _same(_tensor(left, values), want), str(values))
    for name, path in breadth.files().items():
        values = BREADTH_POINTS[name]
        left, right = _pair(workloads.units(path.read_text(encoding="utf-8"))[0], str(path.parent))
        want = breadth_target(name, values)
        report(f"target breadth/{name}", _same(_tensor(right, values), want), str(values))
        report(f"claim breadth/{name}", _same(_tensor(left, values), want), str(values))


def check_variants(report: Callable[[str, bool, str], None]) -> None:
    """Every buggy variant differs at one small instance."""
    points = [
        {"d": 3, "n": 2, "m": 2, "e": 2},
        {"d": 2, "n": 1, "m": 1, "e": 3},
        {"d": 5, "n": 1, "m": 1, "e": 2},
    ]
    for family, variant, _ in refutations.variants():
        name = variant if family == "refutations.azx" else variant.split(": ", 1)[1]
        left, right = _pair(refutations.variant_script(family, name), str(workloads.EXAMPLES))
        hit = ""
        for p in points:
            try:
                a, b = _tensor(left, p), _tensor(right, p)
            except Exception:  # noqa: BLE001, S112
                continue
            if not _same(a, b):
                hit = str(p)
                break
        report(f"false refutations/{variant}", bool(hit), hit or "equal at every point")


def check_translations(report: Callable[[str, bool, str], None]) -> None:
    """PyZX, Qiskit and DiZX translations against the oracle."""
    for family in workloads.FAMILIES:
        for n in (1, 2):
            left, right = _pair(workloads.family_script(family), str(workloads.ROOT))
            symbol = workloads.FAMILIES[family][0]
            values = {symbol: n, "d": 2}
            lnet, rnet = families.nets(family, n)
            for side, net, diagram in (("lhs", lnet, left), ("rhs", rnet, right)):
                tool = np.asarray(families.to_pyzx(net).to_tensor(preserve_scalar=True))
                engine_t = _tensor(diagram, values)
                report(
                    f"pyzx {family} n={n} {side}", _same(tool.reshape(engine_t.shape), engine_t), ""
                )
            if family in families.QISKIT_UNSUPPORTED:
                continue
            qc, _target = families.qiskit_circuit(family, n)
            from qiskit.quantum_info import Operator, Statevector

            engine_t = _tensor(left, values)
            if family == "Fourier order 4":
                m = np.asarray(Operator(qc).data)
                k = engine_t.reshape(2**n, 2**n)
                rev = list(reversed(range(n)))
                m = m.reshape([2] * 2 * n).transpose(rev + [n + r for r in rev]).reshape(2**n, 2**n)
                ok = _proportional(m.ravel(), k.ravel())
            else:
                q = np.asarray(Statevector(qc).data).reshape([2] * qc.num_qubits)
                q = q.transpose(list(reversed(range(qc.num_qubits))))
                ok = _proportional(q.ravel(), engine_t.ravel())
            report(f"qiskit {family} n={n}", ok, "")
    for n in (1, 2):
        gates = qudits.clifford(n)
        half = gates[: len(gates) // 2]
        d = 3
        script = size.azx_script(half, n, False).replace("dim d = 2", f"dim d = {d}")
        script = script.split("@rhs =")[0] + "load" + script.split("load", 1)[1].split("decide")[0]
        session = engine.run_script(script).session
        engine_t = _tensor(session.current, {"d": d}).reshape(d**n, d**n)
        report(
            f"dizx gate list n={n}",
            _proportional(_dizx_matrix(half, n, d).ravel(), engine_t.ravel()),
            "",
        )


def _proportional(a: np.ndarray, b: np.ndarray) -> bool:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < TOL or nb < TOL:
        return bool(na < TOL and nb < TOL)
    return bool(np.allclose(a / na, b / nb, atol=1e-8))


def _dizx_matrix(gates: list[tuple[Any, ...]], n: int, d: int) -> Any:
    """The circuit's matrix (outputs, inputs; qudit 0 the first tensor factor)."""
    F = fourier(d)
    w = np.exp(2j * np.pi / d)
    dim = d**n
    total = np.eye(dim, dtype=complex)
    basis = list(itertools.product(range(d), repeat=n))
    index = {b: i for i, b in enumerate(basis)}
    for g in gates:
        u = np.zeros((dim, dim), dtype=complex)
        for b in basis:
            if g[0] == "P":
                u[index[b], index[b]] = w ** (g[2] * b[g[1]])
            elif g[0] == "C":
                c, t = g[1], g[2]
                out = list(b)
                out[t] = (b[t] + (-1 if g[3] else 1) * b[c]) % d
                u[index[tuple(out)], index[b]] = 1
        if g[0] == "F":
            m = F if g[2] == 1 else F.conj().T
            u = np.array(1, dtype=complex)
            for q in range(n):
                u = np.kron(u, m if q == g[1] else np.eye(d))
        total = u @ total
    return total


def main() -> int:
    """Runs every check; prints one line each; exit status 1 when one fails."""
    failed = 0

    def report(name: str, ok: bool, note: str) -> None:
        nonlocal failed
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{'  ' + note if note else ''}", flush=True)

    check_targets(report)
    check_variants(report)
    check_translations(report)
    print(f"{failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
