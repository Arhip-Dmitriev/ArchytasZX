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

"""Generator denotations: the tensor formula for each generator type at a concrete dimension.

The one leaf the rest of the semantics layer calls for "what does a single node mean".
:mod:`archytaszx.semantics.contract_numeric` is the only other module allowed to build on it.

Axis convention (shared with :mod:`archytaszx.semantics.contract_numeric` and
:mod:`archytaszx.semantics.check`; stated once here). A node with ``m`` inputs and ``n`` outputs
denotes a rank-``m + n`` tensor whose axes are outputs first in ``Node.outputs`` order,
then inputs in ``Node.inputs`` order; axis ``i`` has length that port's ``Dim.to_int()``. A
diagram's boundary follows the same rule. Every consumer must agree with it.

Pairing convention (stated once here, used by ``"B"`` and ``"S"`` alike). A pair ``(a, b)``
with ``a`` in ``[0, s)`` and ``b`` in ``[0, t)`` corresponds to the single index
``k = a * t + b`` in ``[0, s*t)``. ``B . S = id`` and ``S . B = id`` hold exactly.

Z spider. With phase vector ``(alpha_1, ..., alpha_{d-1})`` and ``alpha_0 == 0`` as gauge::

    Z_{m -> n} = sum_{k=0}^{d-1} e^{i * angle(alpha_k)} |k>^{ox n} <k|^{ox m}

The tensor is zero off the "all axes equal k" diagonal and holds ``Phase.to_complex()`` of
``phase.get(k)`` on it; a missing entry is ``Phase.zero()``, so a ``phase=None`` node is
the all-ones diagonal. ``m = n = 0`` collapses to the scalar ``sum_k e^{i alpha_k}``.

X spider, the Fourier conjugate of Z::

    X_{m -> n} = (F^{ox n}) . Z_{m -> n} . ((F^dagger)^{ox m})

with the unitary DFT matrix ``F[j][k] = omega_d^{j*k} / sqrt(d)``, ``omega_d =
e^{2*pi*i/d}``. The exponent sign and the ``1/sqrt(d)`` normalization are fixed here and
nowhere else. ``F`` is symmetric, so ``F^dagger = conj(F)``.

``T`` is ``1`` iff ``out == 0 or out == in``; ``Ti`` is ``1`` iff ``out == in``, ``-1`` iff
``out == 0 and in >= 1``, built in closed integer form. ``W`` has ``W[0][0][0] = 1`` and,
for ``i`` in ``1..d-1``, ``W[0][i][i] = W[i][0][i] = 1``, everything else ``0``.

Guards. :func:`denote` raises a typed error, allocating nothing, when any port dimension or
the phase vector is not concrete, the phase vector's dimension disagrees with the leg
dimension, an ``ALL_LEGS_EQUAL`` generator's legs differ, the generator name is unregistered,
or a fixed-arity box carries the wrong leg counts.

A zero-leg node has no port to read a dimension from, so its dimension comes from its phase
vector's own ``Dim``. A zero-leg node with no phase is rejected as malformed, not defaulted.
"""

from __future__ import annotations

import numpy as np

from archytaszx.algebra.dimension import Dim
from archytaszx.algebra.phase import Phase
from archytaszx.diagram.generators import (
    DIM_BINDER,
    DIM_SPLITTER,
    FOURIER_BOX,
    REGISTRY,
    TRIANGLE,
    TRIANGLE_INVERSE,
    W_NODE,
    X_SPIDER,
    Z_SPIDER,
    DimensionPolicy,
)
from archytaszx.diagram.graph import Node


class DenoteError(Exception):
    """Base class for all errors raised by this module."""


class DenoteDomainError(DenoteError):
    """A value is outside the mathematical domain this module requires.

    Raised for a non-concrete port dimension or phase vector, and for a node whose legs
    or phase vector disagree on dimension.
    """


class DenoteGrammarError(DenoteError):
    """A request is malformed: an unknown generator name, or a dimension-less node.

    Raised for a generator type this module does not know how to denote, for a node whose
    dimension policy :func:`resolve_dimension` does not support, for a fixed-arity box with
    the wrong leg counts, and for a zero-leg node with no phase vector to supply its
    dimension.
    """


def _leg_dim(node: Node, *, require_concrete: bool = True) -> Dim | None:
    """The single dimension shared by every leg of ``node``, or None if it has no legs.

    Raises DenoteDomainError if legs disagree, and, when ``require_concrete``, if any leg
    dimension is non-concrete.
    """
    all_ports = (*node.outputs, *node.inputs)
    if not all_ports:
        return None
    shared = all_ports[0].dim
    for port in all_ports:
        if require_concrete and not port.dim.is_concrete:
            raise DenoteDomainError(
                f"node {node.id!r} ({node.generator_type.name}) has a non-concrete port "
                f"dimension {port.dim}; denote() requires every dimension in scope to be "
                "concrete"
            )
        if port.dim != shared:
            raise DenoteDomainError(
                f"node {node.id!r} ({node.generator_type.name}) has dimension policy "
                f"ALL_LEGS_EQUAL but its legs disagree: {shared} vs {port.dim}"
            )
    return shared


def resolve_dim(node: Node) -> Dim:
    """The single leg dimension of ``node`` as a Dim, concrete or symbolic.

    :func:`resolve_dimension` without its concreteness gate.
    """
    if node.generator_type.dimension_policy is not DimensionPolicy.ALL_LEGS_EQUAL:
        raise DenoteGrammarError(
            f"node {node.id!r} ({node.generator_type.name}) has dimension policy "
            f"{node.generator_type.dimension_policy!r}, which denote() does not yet support"
        )
    leg_dim = _leg_dim(node, require_concrete=False)
    if leg_dim is not None:
        dim = leg_dim
    elif node.phase is not None:
        dim = node.phase.dim
    else:
        raise DenoteGrammarError(
            f"node {node.id!r} ({node.generator_type.name}) has no legs and no phase "
            "vector; its dimension cannot be determined"
        )
    if node.phase is not None and node.phase.dim != dim:
        raise DenoteDomainError(
            f"node {node.id!r} ({node.generator_type.name}) phase vector is over "
            f"{node.phase.dim}, but its legs share dimension {dim}"
        )
    return dim


def resolve_dimension(node: Node) -> int:
    """The single concrete leg dimension of ``node``, as a Python int.

    The public dimension-resolution gate: a node's dimension comes from its legs when it
    has any, otherwise from its phase vector; it is checked concrete and cross-checked
    against a present phase vector's dimension.

    Raises DenoteDomainError for non-concrete or disagreeing dimensions, and
    DenoteGrammarError for a generator whose dimension policy is not ``ALL_LEGS_EQUAL`` and
    for a zero-leg node with no phase vector to fall back on.
    """
    if node.generator_type.dimension_policy is not DimensionPolicy.ALL_LEGS_EQUAL:
        raise DenoteGrammarError(
            f"node {node.id!r} ({node.generator_type.name}) has dimension policy "
            f"{node.generator_type.dimension_policy!r}, which denote() does not yet support"
        )

    leg_dim = _leg_dim(node)
    if leg_dim is not None:
        dim = leg_dim
    elif node.phase is not None:
        dim = node.phase.dim
    else:
        raise DenoteGrammarError(
            f"node {node.id!r} ({node.generator_type.name}) has no legs and no phase "
            "vector; its dimension cannot be determined"
        )

    if not dim.is_concrete:
        raise DenoteDomainError(
            f"node {node.id!r} ({node.generator_type.name}) has non-concrete dimension "
            f"{dim}; denote() requires every dimension in scope to be concrete"
        )

    if node.phase is not None:
        if not node.phase.is_concrete:
            raise DenoteDomainError(
                f"node {node.id!r} ({node.generator_type.name}) has a non-concrete phase "
                f"vector {node.phase}; denote() requires every phase in scope to be concrete"
            )
        if node.phase.dim != dim:
            raise DenoteDomainError(
                f"node {node.id!r} ({node.generator_type.name}) phase vector is over "
                f"{node.phase.dim}, but its legs share dimension {dim}"
            )

    return dim.to_int()


def _z_tensor(node: Node, d: int) -> np.ndarray:
    """The Z spider tensor at concrete dimension ``d``. See the module docstring."""
    rank = node.num_outputs + node.num_inputs
    tensor = np.zeros((d,) * rank, dtype=np.complex128)
    get_phase = node.phase.get if node.phase is not None else (lambda _k: Phase.zero())
    for k in range(d):
        tensor[(k,) * rank] += get_phase(k).to_complex()
    return tensor


def _fourier_matrix(d: int) -> np.ndarray:
    """The unitary DFT matrix F[j][k] = omega_d^{j*k} / sqrt(d). See the module docstring."""
    indices = np.arange(d)
    exponent = np.outer(indices, indices)
    omega = np.exp(2j * np.pi / d)
    return np.asarray(omega**exponent / np.sqrt(d), dtype=np.complex128)


def _apply_matrix_to_axis(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    """Replace ``tensor``'s ``axis`` by ``matrix @ (that axis)``, leaving other axes in place."""
    contracted = np.tensordot(matrix, tensor, axes=([1], [axis]))
    return np.asarray(np.moveaxis(contracted, 0, axis))


def _x_tensor(node: Node, d: int) -> np.ndarray:
    """The X spider tensor at concrete dimension ``d``: the Fourier conjugate of Z. See above."""
    tensor = _z_tensor(node, d)
    fourier = _fourier_matrix(d)
    conj_fourier = np.conjugate(fourier)
    n = node.num_outputs
    m = node.num_inputs
    for axis in range(n):
        tensor = _apply_matrix_to_axis(tensor, fourier, axis)
    for axis in range(n, n + m):
        tensor = _apply_matrix_to_axis(tensor, conj_fourier, axis)
    return tensor


def leg_dims(node: Node) -> tuple[Dim, ...]:
    """Every leg dimension of ``node`` as a Dim, outputs then inputs, with no agreement check."""
    return tuple(port.dim for port in (*node.outputs, *node.inputs))


def leg_dimensions(node: Node) -> tuple[int, ...]:
    """Every leg dimension of ``node`` as an int, outputs then inputs.

    Raises DenoteDomainError on any non-concrete port dim. A zero-leg ``ALL_LEGS_EQUAL``
    node falls back to its phase dim exactly as :func:`resolve_dimension` does.
    """
    dims = leg_dims(node)
    if not dims:
        if node.generator_type.dimension_policy is DimensionPolicy.ALL_LEGS_EQUAL:
            # Called for its raise on a zero-leg node whose phase dim is missing or
            # symbolic. The int it returns is not an axis extent; a rank-0 node has none.
            resolve_dimension(node)
        return ()
    out: list[int] = []
    for dim in dims:
        if not dim.is_concrete:
            raise DenoteDomainError(
                f"node {node.id!r} ({node.generator_type.name}) has a non-concrete port "
                f"dimension {dim}; denote() requires every dimension in scope to be concrete"
            )
        out.append(dim.to_int())
    return tuple(out)


def _check_arity(node: Node, num_inputs: int, num_outputs: int) -> None:
    """Raise DenoteGrammarError unless ``node`` has exactly the given leg counts."""
    if node.num_inputs != num_inputs or node.num_outputs != num_outputs:
        raise DenoteGrammarError(
            f"node {node.id!r} ({node.generator_type.name}) has {node.num_inputs} input(s) "
            f"and {node.num_outputs} output(s); exactly {num_inputs} and {num_outputs} "
            "are required"
        )


def _triangle_tensor(d: int) -> np.ndarray:
    """T at concrete dimension ``d``: tensor[out][in] = 1 iff out == 0 or out == in."""
    tensor = np.zeros((d, d), dtype=np.complex128)
    for out in range(d):
        for inp in range(d):
            if out == 0 or out == inp:
                tensor[out][inp] = 1
    return tensor


def _triangle_inverse_tensor(d: int) -> np.ndarray:
    """Ti at concrete dimension ``d``, in closed integer form: the exact inverse of T."""
    tensor = np.zeros((d, d), dtype=np.complex128)
    for out in range(d):
        for inp in range(d):
            if out == inp:
                tensor[out][inp] = 1
            elif out == 0 and inp >= 1:
                tensor[out][inp] = -1
    return tensor


def _w_tensor(d: int) -> np.ndarray:
    """W at concrete dimension ``d``: shape (d, d, d), tensor[out0][out1][in]."""
    tensor = np.zeros((d, d, d), dtype=np.complex128)
    tensor[0][0][0] = 1
    for i in range(1, d):
        tensor[0][i][i] = 1
        tensor[i][0][i] = 1
    return tensor


def _connective_tensor(node: Node) -> np.ndarray:
    """B or S at concrete per-port dims, under the module docstring's pairing convention."""
    name = node.generator_type.name
    if name == DIM_BINDER.name:
        _check_arity(node, 2, 1)
    else:
        _check_arity(node, 1, 2)
    dims = leg_dimensions(node)
    if name == DIM_BINDER.name:
        product, s, t = dims
    else:
        s, t, product = dims
    if s * t != product:
        raise DenoteGrammarError(
            f"node {node.id!r} ({name}) has leg dims {dims}; the product of its input dims "
            "must equal the product of its output dims"
        )
    if name == DIM_BINDER.name:
        tensor = np.zeros((product, s, t), dtype=np.complex128)
        for a in range(s):
            for b in range(t):
                tensor[a * t + b][a][b] = 1
        return tensor
    tensor = np.zeros((s, t, product), dtype=np.complex128)
    for a in range(s):
        for b in range(t):
            tensor[a][b][a * t + b] = 1
    return tensor


def denote(node: Node) -> np.ndarray:
    """The denotation of ``node`` as a ``complex128`` numpy tensor. See the module docstring.

    The single point in this module that may allocate a dense array for one node, reached
    only after every dimension and phase in scope is confirmed concrete. Dispatches on
    ``node.generator_type`` against the registered generator constants.
    """
    if not REGISTRY.is_registered(node.generator_type):
        raise DenoteGrammarError(
            f"node {node.id!r} carries generator type {node.generator_type.name!r}, which is "
            "not the type registered under that name; denote() dispatches on the registry, "
            "never on a name alone"
        )
    name = node.generator_type.name
    if name in (DIM_BINDER.name, DIM_SPLITTER.name):
        return _connective_tensor(node)
    d = resolve_dimension(node)
    if name == Z_SPIDER.name:
        return _z_tensor(node, d)
    if name == X_SPIDER.name:
        return _x_tensor(node, d)
    if name == FOURIER_BOX.name:
        _check_arity(node, 1, 1)
        return _fourier_matrix(d)
    if name == TRIANGLE.name:
        _check_arity(node, 1, 1)
        return _triangle_tensor(d)
    if name == TRIANGLE_INVERSE.name:
        _check_arity(node, 1, 1)
        return _triangle_inverse_tensor(d)
    if name == W_NODE.name:
        _check_arity(node, 1, 2)
        return _w_tensor(d)
    raise DenoteGrammarError(
        f"node {node.id!r} has generator type {name!r}, which denote() does not know how to denote"
    )
