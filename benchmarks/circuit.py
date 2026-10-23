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

"""Qiskit to PyZX circuit conversion."""

from __future__ import annotations

from fractions import Fraction


class Unsupported(Exception):
    """A tool cannot express the case."""


def qiskit_to_pyzx(qc):  # type: ignore[no-untyped-def]
    """Transpile to {h, rz, cx, x, z, s, sdg, t, tdg} and copy gate by gate."""
    import math

    import pyzx as zx
    from qiskit import transpile

    basis = ["h", "rz", "cx", "x", "z", "s", "sdg", "t", "tdg"]
    tq = transpile(qc, basis_gates=basis, optimization_level=0)
    out = zx.Circuit(tq.num_qubits)
    fixed = {"s": Fraction(1, 2), "sdg": Fraction(3, 2), "t": Fraction(1, 4), "tdg": Fraction(7, 4)}
    for inst in tq.data:
        name = inst.operation.name
        qs = [tq.find_bit(q).index for q in inst.qubits]
        if name in fixed:
            out.add_gate("ZPhase", qs[0], phase=fixed[name])
        elif name == "rz":
            turns = Fraction(float(inst.operation.params[0]) / math.pi).limit_denominator(1 << 20)
            out.add_gate("ZPhase", qs[0], phase=turns % 2)
        else:
            out.add_gate({"h": "HAD", "x": "NOT", "z": "Z", "cx": "CNOT"}[name], *qs)
    return out
