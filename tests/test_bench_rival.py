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

"""Circuit conversions of benchmarks.tools.rival against Qiskit operators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _qram_circuit() -> Any:
    """The Qiskit circuit of qram/main at n = 2 (7 qubits, idle wires included)."""
    from benchmarks import subject
    from benchmarks.tools import rival

    inst = next(i for i in subject.instances(list(subject.catalogue())) if i.id == "qram/main")
    return rival.qiskit_circuit(rival.qubit_ir(inst, 2))[0]


def test_qasm_keeps_the_map() -> None:
    """qasm_and_phase and transpiled give the operator of the circuit on every input."""
    pytest.importorskip("qiskit")
    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    from benchmarks.tools import rival

    qc = _qram_circuit()
    want = Operator(qc).data
    text, phase = rival.qasm_and_phase(qc)
    back = qasm2.loads(text)
    back.global_phase = phase
    assert np.abs(Operator(back).data - want).max() < 1e-8
    t = rival.transpiled(qc, rival.QELIB_BASIS)
    assert np.abs(Operator(t).data - want).max() < 1e-8


def test_phase_functions_keep_exact_angles() -> None:
    """A PhaseFn on 3 wires is its diagonal, and its rz angles are exact multiples of pi down to
    a turn of 2^-70."""
    pytest.importorskip("qiskit")
    from fractions import Fraction

    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    from benchmarks import ir
    from benchmarks.tools import rival

    vals = [Fraction(0), Fraction(1, 3), Fraction(2, 7), Fraction(1, 2**70), Fraction(5, 8)]

    def f(x: tuple[int, ...]) -> Fraction:
        return vals[(x[0] + 2 * x[1] + 3 * x[2]) % len(vals)] + Fraction(x[0] * x[2], 2**70)

    c = ir.Circuit(("a", "b", "e"), (ir.PhaseFn(("a", "b", "e"), f),))
    qc = rival.qiskit_circuit(c)[0]
    diag = []
    for k in range(8):
        x = (k >> 2 & 1, k >> 1 & 1, k & 1)
        diag.append(np.exp(2j * np.pi * float(f(x))))
    want = rival._little(np.diag(diag), 3)
    assert np.abs(Operator(qc).data - want).max() < 1e-12
    text, phase = rival.qasm_and_phase(qc)
    angles = [line.split("(")[1].split(")")[0] for line in text.splitlines() if "rz(" in line]
    assert angles and all("pi" in a for a in angles)
    assert any("/" in a and int(a.split("/")[1]) % 2**69 == 0 for a in angles)
    back = qasm2.loads(text)
    back.global_phase = phase
    assert np.abs(Operator(back).data - want).max() < 1e-12
