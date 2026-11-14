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
