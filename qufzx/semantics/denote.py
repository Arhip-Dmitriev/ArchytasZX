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
:mod:`qufzx.semantics.contract_numeric` is the only other module allowed to build on it,
and it is independent of :mod:`qufzx.rewrite` -- rewriting never contracts, per the spec.

Axis convention (shared with :mod:`qufzx.semantics.contract_numeric` and
:mod:`qufzx.semantics.check`; stated once here). A node with ``m`` inputs and ``n`` outputs
denotes a rank-``m + n`` tensor whose axes are outputs first in ``Node.outputs`` order,
then inputs in ``Node.inputs`` order; axis ``i`` has length that port's ``Dim.to_int()``. A
diagram's boundary follows the same rule. This is a convention, not a derivation --
outputs-before-inputs matches the "ket before bra" reading of ``|k>^{ox n} <k|^{ox m}`` --
and every consumer must agree with it rather than re-deriving its own order.

Z spider. With phase vector ``(alpha_1, ..., alpha_{d-1})`` and ``alpha_0 == 0`` as gauge::

    Z_{m -> n} = sum_{k=0}^{d-1} e^{i * angle(alpha_k)} |k>^{ox n} <k|^{ox m}

The tensor is zero off the "all axes equal k" diagonal and holds ``Phase.to_complex()`` of
``phase.get(k)`` on it; a missing entry is ``Phase.zero()``, so a ``phase=None`` node is
the all-ones diagonal. Degenerate cases fall out of the formula read literally: ``m = n =
0`` collapses to the scalar ``sum_k e^{i alpha_k}``, since every ``k`` lands on the sole
rank-0 position; ``m = 0, n = 2`` denotes ``sum_k |kk>``. The implementation accumulates
into position ``(k,) * rank`` rather than special-casing rank 0.

X spider, the Fourier conjugate of Z::

    X_{m -> n} = (F^{ox n}) . Z_{m -> n} . ((F^dagger)^{ox m})

with the unitary DFT matrix ``F[j][k] = omega_d^{j*k} / sqrt(d)``, ``omega_d =
e^{2*pi*i/d}``. The exponent sign and the ``1/sqrt(d)`` normalization are chosen here and
nowhere else; every later consumer, in particular Phase 9's Hadamard/Fourier generator,
must reuse this exact ``F``. At ``d = 2`` it reduces to the self-inverse Hadamard, so the
asymmetry between ``F`` and ``F^dagger`` is invisible; at ``d > 2`` it is real. ``F`` is
symmetric, so ``F^dagger = conj(F)`` and the implementation applies a plain elementwise
conjugate to input axes with no transpose.

Guards. :func:`denote` raises a typed error, allocating nothing, when any port dimension or
the phase vector is not concrete, the phase vector's dimension disagrees with the leg
dimension, an ``ALL_LEGS_EQUAL`` generator's legs differ, the generator carries some other
dimension policy, or the generator name is neither ``"Z"`` nor ``"X"``. There is exactly one
dispatch point for Phase 9's Hadamard/Fourier generator and Phase 10's triangle, W, and
connective generators to extend; they are not stubbed in with placeholder tensors.

A zero-leg node has no port to read a dimension from, so its dimension comes from its phase
vector's own ``Dim``. A zero-leg node with no phase has no source at all and is rejected as
malformed, not defaulted.
"""

from __future__ import annotations

import numpy as np

from qufzx.algebra.dimension import Dim
from qufzx.algebra.phase import Phase
from qufzx.diagram.generators import REGISTRY, X_SPIDER, Z_SPIDER, DimensionPolicy
from qufzx.diagram.graph import Node


class DenoteError(Exception):
    """Base class for all errors raised by this module."""


class DenoteDomainError(DenoteError):
    """A value is outside the mathematical domain this module requires.

    Raised for a non-concrete port dimension or phase vector, and for a node whose legs
    or phase vector disagree on dimension.
    """


class DenoteGrammarError(DenoteError):
    """A request is malformed: an unknown generator name, or a dimension-less node.

    Raised for a generator type this module does not know how to denote, for one whose
    dimension policy is not ``ALL_LEGS_EQUAL``, and for a zero-leg node with no phase vector
    to supply its dimension.
    """


def _leg_dim(node: Node) -> Dim | None:
    """The single dimension shared by every leg of ``node``, or None if it has no legs.

    Raises DenoteDomainError if any leg dimension is non-concrete or if legs disagree.
    """
    all_ports = (*node.outputs, *node.inputs)
    if not all_ports:
        return None
    shared = all_ports[0].dim
    for port in all_ports:
        if not port.dim.is_concrete:
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


def resolve_dimension(node: Node) -> int:
    """The single concrete leg dimension of ``node``, as a Python int.

    This is the public dimension-resolution gate: it decides where a node's dimension
    comes from (its legs when it has any, otherwise its phase vector), checks it is
    concrete, and cross-checks a present phase vector's dimension against it. Exposed
    separately from :func:`denote` so :mod:`qufzx.semantics.contract_numeric` can size a
    node's tensor before allocating it (the size-cap guard), without duplicating this
    resolution logic.

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


def denote(node: Node) -> np.ndarray:
    """The denotation of ``node`` as a ``complex128`` numpy tensor. See the module docstring.

    This is the single point in this module (and, transitively, in the whole oracle)
    that may allocate a dense array for one node; it does so only after
    :func:`resolve_dimension` has confirmed every dimension and phase in scope is
    concrete. Dispatches on ``node.generator_type`` against the registered
    ``Z_SPIDER``/``X_SPIDER`` constants -- the one dispatch point later generators
    (Phase 9's Hadamard/Fourier, Phase 10's triangle, W, and connectives) must extend.
    """
    d = resolve_dimension(node)
    if not REGISTRY.is_registered(node.generator_type):
        raise DenoteGrammarError(
            f"node {node.id!r} carries generator type {node.generator_type.name!r}, which is "
            "not the type registered under that name; denote() dispatches on the registry, "
            "never on a name alone"
        )
    if node.generator_type.name == Z_SPIDER.name:
        return _z_tensor(node, d)
    if node.generator_type.name == X_SPIDER.name:
        return _x_tensor(node, d)
    raise DenoteGrammarError(
        f"node {node.id!r} has generator type {node.generator_type.name!r}, which "
        "denote() does not know how to denote (only 'Z' and 'X' are registered)"
    )
