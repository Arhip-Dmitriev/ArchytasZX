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


"""Standalone driver for ``TestCrossProcessDeterminism`` in ``test_contract_symbolic.py``.

Not a pytest module: run as a plain script, once per ``PYTHONHASHSEED``, via ``subprocess``.
Contracts a diagram whose entry keeps a residual unevaluated index sum inside a larger
product, then prints the ``repr`` of the tensor and of a separately simplified scalar. Prints
no ``hash()``, which legitimately varies across processes.
"""

from __future__ import annotations

import sympy as sp  # type: ignore[import-untyped]  # sympy ships no py.typed marker

from qufzx.algebra.dimension import Dim
from qufzx.algebra.scalar import Scalar
from qufzx.diagram.generators import FOURIER_BOX, X_SPIDER, Z_SPIDER
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.semantics.contract_symbolic import contract_symbolic


def _build_diagram() -> Diagram:
    """A Z spider, an X spider and a Fourier box wired into one three-boundary diagram."""
    dim = Dim.symbol("d")
    diagram = Diagram()
    z = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[dim, dim, dim])
    x = diagram.add_node(X_SPIDER, input_dims=[dim], output_dims=[dim, dim])
    f = diagram.add_node(FOURIER_BOX, input_dims=[dim], output_dims=[dim])
    diagram.add_wire(PortRef(z, Direction.OUTPUT, 2), PortRef(x, Direction.INPUT, 0))
    diagram.add_wire(PortRef(x, Direction.OUTPUT, 1), PortRef(f, Direction.INPUT, 0))
    diagram.set_boundary_outputs(
        [
            PortRef(z, Direction.OUTPUT, 0),
            PortRef(z, Direction.OUTPUT, 1),
            PortRef(x, Direction.OUTPUT, 0),
            PortRef(f, Direction.OUTPUT, 0),
        ]
    )
    diagram.set_boundary_inputs([])
    return diagram


def _residual_product() -> Scalar:
    """A product of a closable sum and one the simplifier must leave standing."""
    dim = Dim.symbol("d")
    a = sp.Symbol("a", integer=True, nonnegative=True)
    closable = Scalar.index_sum(dim, lambda _k: Scalar.one())
    residual = Scalar.index_sum(dim, lambda k: Scalar.omega(dim, 3 * k.to_sympy()))
    other = Scalar.index_sum(dim, lambda k: Scalar.omega(dim, a * k.to_sympy()))
    return (closable * residual * other * Scalar.symbol("s")).simplify()


def main() -> None:
    """Print the two reprs the determinism test compares across processes."""
    tensor = contract_symbolic(_build_diagram())
    print(repr(tensor.axes))
    print(repr(tensor.entry))
    print(repr(_residual_product()))


if __name__ == "__main__":
    main()
