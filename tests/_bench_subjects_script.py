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

"""Test, ``python -m tests._bench_subjects_script [SUBJECT_GLOB ...] [--jobs J]`` (default 2
workers,
each capped at 1.5 GiB): for every S
equation at every point with n <= 3 and d <= 4 in its domain, the naive circuit equals the
specification under the numeric oracle, the equation's reference circuit (when it has one)
equals the specification, and the naive circuit's tensor equals the dense
reference evaluation of its IR (benchmarks/tools/dense.py). Prints one line per failure and a
summary; exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

N_MAX = 3
D_MAX = 4
TOL = 1e-8
DENSE_CAP = 1 << 22
WORKER_MEMORY = 3 << 29


def small_points(inst: Any) -> list[tuple[int | None, int]]:
    """Every (n, d) with n <= 3, d <= 4 in the instance's domain."""
    has_n, has_d = inst.subject.size_parameter is not None, inst.subject.general_d
    ns = [n for n in range(N_MAX + 1) if inst.domain.has_n(n)] if has_n else [None]
    ds = (
        [d for d in range(1, D_MAX + 1) if inst.domain.has_d(d)]
        if has_d
        else [int(inst.subject.dimension)]
    )
    return [(n, d) for n in ns for d in ds]


def engine_tensor(diagram: Any) -> np.ndarray:
    """The oracle's tensor of a concrete diagram: pure axes (outs, ins), or the CPM."""
    from archytaszx.semantics import check, denote

    inst = check.instantiate(diagram, {})
    if inst.is_pure():
        return np.asarray(check.contract(inst).tensor)
    return np.asarray(denote.denote_mixed(inst))


def _limit_memory() -> None:
    """Caps a pool worker's address space at WORKER_MEMORY bytes."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY, WORKER_MEMORY))


def _too_large(c: Any, diagram: Any, d: int) -> bool:
    """Whether the dense checks would exceed DENSE_CAP entries: the boundary (squared when
    mixed) or the reference evaluation (every wire plus every discarded or measured leg)."""
    from archytaszx.semantics import check
    from benchmarks import ir

    inst = check.instantiate(diagram, {})
    legs = len(inst.boundary_outputs) + len(inst.boundary_inputs)
    if not inst.is_pure():
        legs *= 2
    env = sum(1 for g in c.gates if isinstance(g, (ir.Measure, ir.Discard)))
    wires = len(c.wires)
    return max(d, 2) ** legs > DENSE_CAP or max(d, 2) ** (2 * (wires + env)) > DENSE_CAP * DENSE_CAP


def check_point(job: tuple[str, int, int | None, int]) -> list[str]:
    """The failures of one instance at one point."""
    from archytaszx.semantics import check
    from benchmarks import subject
    from benchmarks.tools import dense as bench_reference

    sid, k, n, d = job
    insts = [i for i in subject.instances(list(subject.catalogue())) if i.subject.id == sid]
    inst = insts[k]
    tag = f"{inst.id} n={n} d={d}"
    try:
        sc = subject.script(inst, n, d)
        left, right = subject_diagrams(sc.text)
        from benchmarks.tools import common

        if _too_large(common.expand(subject.circuit_at(inst, n, d), n or 0), left, d):
            return [f"SKIP {tag}: too large for the dense oracle"]
        cmp = check.compare(left, right, {})
        out = [] if cmp.matched else [f"{tag}: circuit != specification: {cmp.reason}"]
        c = subject.circuit_at(inst, n, d)
        ref = bench_reference.evaluate(c, n if n is not None else 0, d)
        eng = engine_tensor(left)
        if ref.size != eng.size or not np.allclose(ref.reshape(-1), eng.reshape(-1), atol=TOL):
            dev = (
                np.max(np.abs(ref.reshape(-1) - eng.reshape(-1)))
                if ref.size == eng.size
                else "shape"
            )
            out.append(f"{tag}: translation != IR reference (deviation {dev})")
        if inst.equation.reference is not None and right.is_pure():
            from benchmarks.tools import common

            refc = common.expand(inst.equation.reference(n, d), n or 0)
            ref_t = bench_reference.evaluate(refc, n if n is not None else 0, d)
            spec_t = engine_tensor(right)
            if ref_t.size != spec_t.size or not np.allclose(
                ref_t.reshape(-1), spec_t.reshape(-1), atol=TOL
            ):
                out.append(f"{tag}: reference circuit != specification")
        return out
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("ContractSizeError", "MemoryError"):
            return [f"SKIP {tag}: too large for the dense oracle"]
        return [f"{tag}: error {type(exc).__name__}: {str(exc)[:300]}"]


def subject_diagrams(text: str) -> tuple[Any, Any]:
    """Both diagrams of an equation script."""
    from benchmarks import stages

    return stages.diagrams(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", default=["*"])
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    from benchmarks import subject

    jobs = []
    insts = list(subject.instances(list(subject.catalogue())))
    for sid in sorted({i.subject.id for i in insts}):
        if not any(fnmatch.fnmatch(sid, g) for g in args.subjects):
            continue
        mine = [i for i in insts if i.subject.id == sid]
        for k, inst in enumerate(mine):
            jobs += [(sid, k, n, d) for n, d in small_points(inst)]
    failures: list[str] = []
    with ProcessPoolExecutor(args.jobs, initializer=_limit_memory) as pool:
        for res in pool.map(check_point, jobs, chunksize=1):
            for line in res:
                print(line, flush=True)
            failures += [ln for ln in res if not ln.startswith("SKIP ")]
    print(f"subjects test: {len(jobs)} points, {len(failures)} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
