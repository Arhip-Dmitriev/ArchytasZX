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

"""Pauli propagation through the controlled add: X on the control before CX equals CX then X on
both; Z on the target before CX equals CX then Z on the target and Z^-1 on the control."""

from benchmarks.subjects._common import *


def _x(n, d):
    """X(c), then CX."""
    return circuit(["c", "t"], [X("c"), CX("c", "t")])


def _x_spec(c):
    """CX followed by shifts on both wires."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"sc = X(1 -> 1, {c.D}) phase{{k: -k/{c.d}}}\nst = X(1 -> 1, {c.D}) phase{{k: -k/{c.d}}}\n"
        f"wire z.out0 -- sc.in0, x.out0 -- st.in0\ninputs z.in0, x.in0\noutputs sc.out0, st.out0\n"
        f"scalar {c.d}^(1/2)"
    )


def _z(n, d):
    """Z(t), then CX."""
    return circuit(["c", "t"], [Z("t"), CX("c", "t")])


def _z_spec(c):
    """CX followed by Z^-1 on the control and Z on the target."""
    return (
        f"z = Z(1 -> 2, {c.D})\nx = X(2 -> 1, {c.D})\nwire z.out1 -- x.in1\n"
        f"pc = Z(1 -> 1, {c.D}) phase{{k: -k/{c.d}}}\npt = Z(1 -> 1, {c.D}) phase{{k: k/{c.d}}}\n"
        f"wire z.out0 -- pc.in0, x.out0 -- pt.in0\ninputs z.in0, x.in0\noutputs pc.out0, pt.out0\n"
        f"scalar {c.d}^(1/2)"
    )


def _x_ref(n, d):
    """CX, X, X."""
    return circuit(["c", "t"], [CX("c", "t"), X("c"), X("t")])


def _z_ref(n, d):
    """CX, Z^-1, Z."""
    return circuit(["c", "t"], [CX("c", "t"), Z("c", -1), Z("t")])


SUBJECT = Subject(
    id="pauli_propagation",
    name="Pauli propagation through CX",
    category="circuit",
    source="gottesman1999fault",
    size_parameter=None,
    dimension="general",
    domain=Domain(d_min=1),
    equations=(
        Equation("x", _x, _x_spec, form="fixed", reference=_x_ref),
        Equation("z", _z, _z_spec, form="fixed", reference=_z_ref),
    ),
)
