"""Benchmark fairness checks: rival tensors, scalars and timers (F62)."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

DRIVERS = Path(__file__).resolve().parent.parent / "benchmarks" / "tools" / "drivers"
TOOL_MODULES = {"pyzx", "quizx", "qiskit", "cirq", "mqt", "dizx", "pytket"}


def _driver(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"_driver_{name}", DRIVERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dense_phase_gate_reads_its_k0_entry() -> None:
    from benchmarks import ir
    from benchmarks.tools import dense

    z = dense._single(ir.ZPhase("a", {0: Fraction(1, 4), 1: Fraction(1, 2)}), 2, 1, 2)
    assert np.allclose(np.diag(z), [1j, -1])
    r = dense._single(ir.ZPhase("a", rule="k/4 + 1/8"), 2, 1, 2)
    assert np.allclose(np.diag(r), [np.exp(2j * np.pi / 8), np.exp(2j * np.pi * 3 / 8)])


def test_zx_scalar_with_a_zero_sum_of_phases_is_zero() -> None:
    zx = _driver("zx")
    s = types.SimpleNamespace(
        is_zero=False,
        power2=0,
        phase=Fraction(0),
        floatfactor=1.0,
        phasenodes=[],
        sum_of_phases={Fraction(0): 1, Fraction(1): 1},
    )
    assert zx.scalar_parts(s)[0]


def _timer_line(tree: ast.Module) -> dict[str, int]:
    """Line of each function's first time.process_time() assignment."""
    out: dict[str, int] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "process_time"
            ):
                out[fn.name] = min(out.get(fn.name, node.lineno), node.lineno)
    return out


def test_driver_timers_start_after_the_tool_imports() -> None:
    for path in sorted(DRIVERS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        starts = _timer_line(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name not in starts:
                continue
            called = {
                n.func.id
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            helpers = [
                f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name in called
            ]
            early = {t for t, line in _tool_imports(fn) if line < starts[fn.name]}
            for t, line in _tool_imports(fn):
                assert line < starts[fn.name], (path.name, fn.name, t)
            for helper in helpers:
                for t, _ in _tool_imports(helper):
                    assert t in early, (path.name, fn.name, helper.name, t)


def _tool_imports(fn: ast.FunctionDef) -> list[tuple[str, int]]:
    """(top-level tool module, line) of every tool import inside ``fn``."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        out += [(n.split(".")[0], node.lineno) for n in names if n.split(".")[0] in TOOL_MODULES]
    return out


def test_cirq_driver_keeps_an_idle_wire() -> None:
    sim = _driver("sim")
    calls: dict[str, Any] = {}

    class Circuit:
        def __init__(self, ops: list[Any]) -> None:
            self.ops = ops

        def unitary(self, qubit_order: Any = None, **_: Any) -> np.ndarray:
            calls["order"] = qubit_order
            return np.eye(2 ** len(qubit_order))

    class Gate:
        def __init__(self, mat: Any, qid_shape: Any) -> None:
            self.mat = mat

        def on(self, *qs: Any) -> tuple[Any, ...]:
            return qs

    stub = types.SimpleNamespace(
        LineQid=types.SimpleNamespace(range=lambda m, dimension: list(range(m))),
        MatrixGate=Gate,
        Circuit=Circuit,
        unitary=lambda c: np.eye(2 ** len({q for op in c.ops for q in op})),
    )
    sys.modules["cirq"] = stub  # type: ignore[assignment]
    try:
        p = {"wires": 2, "d": 2, "gates": [[[0], [[0, 1], [1, 0]], [[0, 0], [0, 0]]]]}
        p |= {"scalar": [1, 0], "inits": {}}
        t, is_state = sim.cirq_tensor(p)
    finally:
        del sys.modules["cirq"]
    assert not is_state and t.shape == (2, 2, 2, 2)


class _Patch:
    """Sets module attributes for the duration of a with block."""

    def __init__(self, mod: Any, **values: Any) -> None:
        self.mod, self.values = mod, values
        self.old: dict[str, Any] = {}

    def __enter__(self) -> None:
        for k, v in self.values.items():
            self.old[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)

    def __exit__(self, *_: object) -> None:
        for k, v in self.old.items():
            setattr(self.mod, k, v)


def _burn(cmd: Any, limit: float) -> dict[str, Any]:
    import time

    t = time.process_time()
    while time.process_time() - t < 0.05:
        pass
    return {"status": "ok", "returncode": 0, "cpu": 50.0}


def test_published_proofs_prep_is_parent_cpu_and_path_is_restored(tmp_path: Path) -> None:
    import os

    from benchmarks.tools import published_proofs as pp
    from benchmarks.tools import rival

    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "_CoqProject").write_text("")
    (tmp_path / "P" / "a.v").write_text("x\n")
    seen: dict[str, Any] = {}
    inst = types.SimpleNamespace(subject=types.SimpleNamespace(id="s"))
    job = {"point": None, "config": {"stages": {"timeout_seconds": 10}}}
    path0 = os.environ.get("PATH", "")

    def finish(inst: Any, res: Any, prep: float, versions: Any) -> dict[str, Any]:
        seen["prep"] = prep
        return dict(res)

    with (
        _Patch(pp, supports=lambda i, p: None, SRC=tmp_path, PROOFS={"s": [("P", "a.v")]}),
        _Patch(
            rival,
            run_binary=_burn,
            finish=finish,
        ),
    ):
        pp.run(inst, job)
    assert os.environ.get("PATH", "") == path0
    assert 0.01 < seen["prep"] < 50.0


def test_rival_versions_are_commits_or_package_versions() -> None:
    tools = DRIVERS.parent
    for path in sorted(tools.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "finish"
                and len(node.args) == 4
                and isinstance(node.args[3], ast.Dict)
            ):
                for v in node.args[3].values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        assert "/" not in v.value, (path.name, v.value)


def _quanto_job() -> tuple[Any, dict[str, Any]]:
    inst = types.SimpleNamespace(subject=types.SimpleNamespace(id="s"))
    return inst, {"point": [1, 2], "config": {"stages": {"timeout_seconds": 10}}}


def test_quantomatic_spec_refusal_is_a_spec_bound() -> None:
    from benchmarks import stages
    from benchmarks.tools import quantomatic as q
    from benchmarks.tools import rival

    def refuse(text: str) -> Any:
        raise RuntimeError("load: too many nodes")

    inst, job = _quanto_job()
    script = types.SimpleNamespace(script=lambda i, n, d: types.SimpleNamespace(text="x"))
    raised = None
    with _Patch(q, supports=lambda i, p: None, subject=script), _Patch(stages, diagrams=refuse):
        try:
            q.run(inst, job)
        except rival.SpecBound as exc:
            raised = exc
    assert raised is not None


def test_quantomatic_unparsable_output_keeps_its_times() -> None:
    from benchmarks import stages
    from benchmarks.tools import quantomatic as q
    from benchmarks.tools import rival

    inst, job = _quanto_job()
    script = types.SimpleNamespace(script=lambda i, n, d: types.SimpleNamespace(text="x"))
    seen: dict[str, Any] = {}

    def finish(inst: Any, res: Any, prep: float, versions: Any) -> dict[str, Any]:
        seen.update(res)
        return dict(res)

    with (
        _Patch(q, supports=lambda i, p: None, subject=script, qgraph=lambda d: {}),
        _Patch(q, java=lambda: "java", jar=lambda: "x.jar", rules=lambda: "r"),
        _Patch(stages, diagrams=lambda t: (None, None)),
        _Patch(
            rival,
            run_binary=lambda cmd, limit, **k: {
                "status": "ok",
                "stdout": "no json",
                "cpu": 1.5,
                "wall": 2.5,
            },
            finish=finish,
        ),
    ):
        q.run(inst, job)
    assert seen["status"] == "error" and seen["cpu"] == 1.5 and seen["wall"] == 2.5
