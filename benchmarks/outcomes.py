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

"""Run outcomes, stages and cause codes (BENCHMARKS.txt, section 10.6), and the classifier that
gives every run record its ``outcome``, ``stage``, ``cause`` and ``cause_message``.

:func:`classify` reads only a record's fields, so it applies to new records (the runner) and to
logged ones (benchmarks/reclassify.py) alike.
"""

from __future__ import annotations

import re
from typing import Any

OUTCOMES = (
    "ok",
    "uncertified",
    "wrong",
    "wrong (uncertified)",
    "unknown",
    "unsupported",
    "timeout",
    "oom",
    "error",
    "harness_error",
    "invalid_point",
)
STAGES = ("generate", "decide", "certificate", "lean_export", "lean_check", "prepare", "rival")
NOT_SCORED = ("harness_error", "invalid_point")
VERDICT_OUTCOMES = ("ok", "uncertified", "wrong", "wrong (uncertified)", "unknown")
ENGINE_STAGES = ("generate", "decide", "certificate", "lean_export", "lean_check")

UNSUPPORTED: dict[str, str] = {
    "register_count_mismatch": "gate across registers of different counts",
    "phase_table_on_register": "a phase table on a register",
    "phase_table_dims": "a phase table needs literal concrete dimensions",
    "phase_table_irrational": "a phase table entry is not a rational number of turns",
    "controlled_mixed_dims": "a controlled gate between wires of different dimensions",
    "multi_controlled_dims": "a multi-controlled gate needs literal concrete dimensions",
    "oracle_dims": "an oracle needs literal concrete dimensions",
    "dense_gate_no_zxw": "a dense reference gate has no ZXW translation",
    "permutation_on_register": "a permutation on a register",
    "permutation_dims": "a permutation needs literal concrete dimensions",
    "no_general_form": "the circuit has no general form in n",
    "needs_concrete_d": "the equation needs a concrete dimension",
    "symbolic_point": "no symbolic n or d",
    "qubits_only": "qubits only",
    "odd_prime_only": "odd prime d only",
    "fixed_dimension": "fixed dimension",
    "channel": "measurement or discard (a channel, not a pure map)",
    "wire_dim_not_d": "a wire of dimension other than d",
    "mixed_wire_dims": "wires of mixed dimensions",
    "gate_unsupported": "gate ",
    "scalar_modulus": "a circuit scalar of modulus other than 1",
    "circuit_scalar": "a circuit scalar",
    "spec_mixed": "a mixed specification",
    "spec_axes": "the specification has more than",
    "spec_too_large": "the specification tensor is too large",
    "spec_generator": "generator ",
    "spec_weighted": "a weighted spider",
    "spec_no_wires": "the specification has no wires",
    "spec_zero_state": "the specification is the zero state",
    "spec_isometry": "the specification is an isometry, not a unitary",
    "spec_not_unitary": "the specification is not unitary",
    "no_reference": "no gate-level reference circuit for this equation",
    "gate_set": "not expressible in the gate set",
    "tensor_axes": " tensor axes: above numpy's",
    "too_large_to_simulate": ": too large to simulate",
    "post_selected": "post-selected wires",
    "phase_not_pi4": "a phase that is not a multiple of pi/4",
    "not_clifford_t": " outside Clifford+T",
    "not_dizx_clifford": " is not a DiZX Clifford gate",
    "width_mismatch": "circuit and reference differ in width",
    "autoq_amplitude": "an amplitude outside sqrt2 powers and eighth turns",
    "states_only": "only states (every wire initialised, none post-selected)",
    "zero_state": "the zero state",
    "no_public_release": "symbolic n needs AutoQ-Para, which has no public release",
    "quantomatic_bang_boxes": "symbolic n through Quantomatic's own !-boxes is not built",
    "mixed_diagram": "a mixed diagram",
    "hadamard_arity": "a hadamard vertex of arity other than 2",
    "published_once": "published proofs are checked once, at their stated size",
    "proof_file_missing": " proof file ",
    "not_installed": "not installed",
}

CAUSES: dict[str, str] = {
    "verdict": "verdict.<verdict>: the run's verdict (ok, uncertified, wrong)",
    "pair_equal_at_point": "the F pair is equal at this point (benchmarks.differs)",
    "rival.no_verdict": "the rival returned no verdict",
    "kill.stage_timer": "the runner's wall timer killed the worker",
    "kill.sigkill": "the worker died of SIGKILL or exit 137",
    "kill.cgroup_oom_kill": "the scope's memory.events oom_kill rose",
    "kill.exit": "the worker died with another exit code",
    "limit.stage_cpu": "a stage used up its CPU limit",
    "limit.prepare_cpu": "a rival's preparation in the worker used up its CPU limit",
    "limit.rival_cpu": "the rival used up its CPU limit",
    "limit.rival_wall": "the rival used up its wall limit",
    "memory.error": "MemoryError in the worker",
    "memory.rival": "the rival was killed for memory or raised MemoryError",
    "memory.lean": "Lean was killed for memory",
    "rival.exception": "an exception raised in the rival's library",
    "rival.exit": "the rival's binary or driver exited nonzero",
    "engine.load_refused": "a load line of the script failed in the engine",
    "engine.exception": "an exception in an ArchytasZX stage",
    "lean.rejected": "Lean did not accept the exported file",
    "harness.prepare": "an exception in the worker while preparing a rival's input",
    "harness.driver": "an exception raised in the harness's own driver code",
    "harness.generate": "an exception while generating the ArchytasZX script",
    "unsupported.<code>": "a code of UNSUPPORTED",
    "unknown.<rung code>": "the last rung tried and its stop code (RUNGS)",
}

RUNGS: tuple[tuple[str, str], ...] = (
    ("normal_form.differ", r"normal forms differ"),
    ("normal_form.agree", r"normal forms agree"),
    ("oracle.refused", r"\d+ sample\(s\) refused"),
    ("oracle.no_mismatch", r"no oracle mismatch in \d+ sample"),
    ("oracle.none_evaluated", r"no oracle sample evaluated"),
    ("oracle.mismatch", r"oracle mismatch at"),
    ("saturation.iteration_limit", r"saturation: .*stopped by iteration_limit"),
    ("saturation.enode_limit", r"saturation: .*stopped by enode_limit"),
    ("saturation.saturated", r"saturation: .*stopped by saturated"),
    (
        "symbolic.declined.box_meets_wire",
        (
            r"symbolic: symbolic contraction declined: SymbolicContractionUnsupportedError: "
            r"bang box \d+ has symbolic multiplicity \w+ and meets the rest"
        ),
    ),
    ("symbolic.declined.scalar_budget", r"symbolic: symbolic contraction declined: ScalarBudget"),
    ("symbolic.declined.other", r"symbolic: symbolic contraction declined"),
    ("symbolic.indeterminate.replicated_factors", r"symbolic: indeterminate: the replicated fact"),
    ("symbolic.indeterminate", r"symbolic: indeterminate"),
    ("symbolic.entries_differ", r"symbolic: entries differ"),
    ("symbolic.differ", r"symbolic contractions differ"),
    ("induction.base_not_proved", r"induction: base case \S+ not proved"),
    ("induction.step_not_discharged", r"induction: step case over \w+ not proved: no discharge"),
    ("induction_reversed.base_not_proved", r"reversed: base case"),
    ("induction_reversed.step_not_discharged", r"reversed: step c"),
    ("refutation.no_lean", r"(?:not reported unequal|uncertified): no Lean refutation"),
)
_RUNG_RE = [(code, re.compile(pat)) for code, pat in RUNGS]
_RUNG_HEAD = re.compile(r"(?:doubled, \w+: )?([a-z_]+):")
_DOUBLED = re.compile(r"doubled, \w+: ")
_AFTER_INDUCTION = ("induction", "induction_reversed", "refutation", "reversed")
_FRAME = re.compile(r'(?:File ")?([^"\n]*)", line \d+, in ([^\n]+)\n\s*([^\n]*)')
DRIVER_OWN = ("number", "_build", "_identity", "add")
RIVAL_PACKAGES: dict[str, tuple[str, ...]] = {
    "pyzx": ("pyzx",),
    "quizx": ("quizx",),
    "qcec": ("mqt", "QuantumComputation", "verify("),
    "cirq": ("cirq",),
    "qiskit": ("qiskit", "qasm2", "Operator"),
    "tket": ("pytket", "FullPeepholeOptimise", "circuit_from_qasm_str"),
    "dizx": ("dizx", "clifford_simp"),
}


def rung_codes(reason: str) -> list[str]:
    """The stop code of every rung note (``; ``-separated) in a decide reason, in order; a note
    no pattern matches is ``<rung>.other``, the last note of a reason cut at 400 characters is
    ``<rung>.truncated``; after an induction note only induction, reversed and refutation notes
    are read (the rest is the base case's own decision)."""
    out: list[str] = []
    segs = [s.strip() for s in reason.split("; ") if s.strip()]
    nested = False
    for i, seg in enumerate(segs):
        body = _DOUBLED.sub("", seg, count=1)
        prefix = "doubled." if body != seg else ""
        code = next((c for c, rx in _RUNG_RE if rx.match(body)), None)
        if code is None:
            head = _RUNG_HEAD.match(seg)
            if head is None:
                continue
            code = f"{head.group(1)}.other"
        rung = code.split(".", 1)[0]
        if nested and rung not in _AFTER_INDUCTION:
            continue
        if i == len(segs) - 1 and len(reason) >= 399:
            code = f"{rung}.truncated"
        nested = nested or rung in ("induction", "induction_reversed")
        code = prefix + code
        if not out or out[-1] != code:
            out.append(code)
    return out


def unsupported_code(message: str) -> str:
    """The UNSUPPORTED code whose message the text contains, else ``other``."""
    text = re.sub(r"^[\w.]*Unsupported: ", "", message.strip())
    best = ""
    for code, frag in UNSUPPORTED.items():
        if frag in text and len(frag) > len(UNSUPPORTED.get(best, "")):
            best = code
    return best or "other"


def _frame_cause(tool: str, detail: str) -> str:
    """harness.driver or rival.exception from a driver traceback: harness.driver when the
    deepest driver frame is a DRIVER_OWN helper (the harness's own use of the rival's objects),
    or is the deepest frame and its line calls none of the rival's names; rival.exception
    otherwise; rival.exit without a traceback."""
    frames = _FRAME.findall(detail or "")
    if not frames:
        return "rival.exit"
    ours = [f for f in frames if "/tools/drivers/" in f[0]]
    if ours and ours[-1][1].strip() in DRIVER_OWN:
        return "harness.driver"
    path, _, code = frames[-1]
    if "/tools/drivers/" in path:
        names = RIVAL_PACKAGES.get(tool, (tool,))
        return "rival.exception" if any(nm in code for nm in names) else "harness.driver"
    return "rival.exception"


def _first_failed(stages: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    for s in ENGINE_STAGES:
        st = stages.get(s)
        if st and st.get("status") not in ("ok", "not_run"):
            return s, st
    return None, {}


def _killed(rec: dict[str, Any], tool: str) -> tuple[str, str | None, str]:
    """(outcome, stage, cause) of a record the runner wrote for a worker without a result."""
    oc = rec.get("outcome")
    stage = rec.get("killed_in")
    stage = None if stage in (None, "start") else stage
    kill = rec.get("kill")
    code = rec.get("exit")
    if rec.get("limit") == "prepare_cpu":
        return "timeout", "prepare", "limit.prepare_cpu"
    if kill == "stage timer" or (kill is None and oc == "timeout"):
        return "timeout", stage, "kill.stage_timer"
    if kill == "cgroup oom_kill" or rec.get("journal_oom_kill"):
        return "oom", stage, "kill.cgroup_oom_kill"
    if kill == "SIGKILL" or code in (-9, 137):
        return "oom", stage, "kill.sigkill"
    if oc == "oom":
        return "oom", stage, "memory.error"
    return "error", stage, "kill.exit"


def _archytaszx(rec: dict[str, Any]) -> tuple[str, str | None, str, str]:
    st = rec.get("stages") or {}
    oc = rec.get("outcome") or "error"
    dec = st.get("decide") or {}
    if oc in ("ok", "uncertified", "wrong", "wrong (uncertified)", "unknown"):
        reason = str(dec.get("reason") or "")
        if oc == "unknown":
            rungs = dec.get("rungs") or rung_codes(reason)
            return oc, "decide", "unknown." + (rungs[-1] if rungs else "none"), reason
        return oc, "decide", f"verdict.{rec.get('verdict')}", str(dec.get("method") or "")
    s, failed = _first_failed(st)
    detail = str(failed.get("detail") or rec.get("detail") or "")
    if s is None:
        return "error", None, "engine.exception", detail
    status = failed.get("status")
    if status == "timeout":
        return "timeout", s, "limit.stage_cpu", detail
    if status == "oom":
        return "oom", s, "memory.lean" if s == "lean_check" else "memory.error", detail
    if s == "generate":
        return "harness_error", s, "harness.generate", detail
    if s == "lean_check":
        return "error", s, "lean.rejected", detail
    if s == "decide" and re.match(r"RuntimeError: [^\n]*?(?:load|@\w+ =): error:", detail):
        return "error", s, "engine.load_refused", detail
    return "error", s, "engine.exception", detail


def _rival(rec: dict[str, Any], tool: str) -> tuple[str, str | None, str, str]:
    oc = rec.get("outcome") or "error"
    st = rec.get("stages") or {}
    check = st.get("check") or {}
    if not st:
        detail = str(rec.get("detail") or "")
        if oc == "oom":
            return "oom", "prepare", "memory.error", detail
        if oc in ("error", "harness_error"):
            if "MemoryError" in detail.split(":", 1)[0]:
                return "oom", "prepare", "memory.error", detail
            return "harness_error", "prepare", "harness.prepare", detail
        return oc, "prepare", "unknown.none", detail
    detail = str(check.get("detail") or "")
    status = check.get("status")
    if oc in ("ok", "uncertified", "wrong", "wrong (uncertified)"):
        return oc, "rival", f"verdict.{rec.get('verdict')}", ""
    if oc == "unknown":
        return oc, "rival", "rival.no_verdict", str(check.get("equivalence") or "")
    if status == "timeout":
        cause = check.get("limit") or ("wall" if "detail" not in check else "cpu")
        return "timeout", "rival", f"limit.rival_{cause}", detail
    if status == "oom":
        return "oom", "rival", "memory.rival", detail
    cause = _frame_cause(tool, detail)
    oc = "harness_error" if cause.startswith("harness.") else "error"
    return oc, "rival", cause, detail


FREE_RANGE = range(65)


def free_axes(inst: Any, rec: dict[str, Any]) -> tuple[str, ...]:
    """The axes a run decides over beyond its own point: none for a rival; for ArchytasZX n
    unless it is fixed (an unrolled circuit or no size parameter) and d unless it is literal
    (no general d, or a literal script)."""
    if rec.get("tool") != "archytaszx":
        return ()
    n_fixed = (
        rec.get("n") is None
        or inst.equation.form == "unrolled"
        or inst.subject.size_parameter is None
    )
    gen = (rec.get("stages") or {}).get("generate") or {}
    d_fixed = not inst.subject.general_d or bool(gen.get("literal"))
    return tuple(a for a, fixed in (("n", n_fixed), ("d", d_fixed)) if not fixed)


def equal_region(false: dict[str, Any], n: int | None, d: int, free: tuple[str, ...]) -> bool:
    """Whether the F pair is provably equal (benchmarks.differs.equal_at True) at (n, d) and at
    every value in FREE_RANGE of each free axis."""
    from benchmarks.differs import equal_at

    ns = list(FREE_RANGE) if "n" in free else [n]
    ds = [v for v in FREE_RANGE if v >= 1] if "d" in free else [d]
    return equal_at(false, n, d) is True and all(
        equal_at(false, a, b) is True for a in ns for b in ds
    )


def instances(catalogue: str = "benchmarks/subjects") -> dict[str, Any]:
    """Every S instance of the catalogue by id."""
    from benchmarks import subject

    return {i.id: i for i in subject.instances(list(subject.catalogue(catalogue)))}


def reclassify_record(
    rec: dict[str, Any], insts: dict[str, Any], falses: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """:func:`classify` of a logged run record, its F record and instance looked up by equation
    id (an F id ``EQ!`` uses EQ's instance); a run of an unknown instance has no free axis."""
    eq = str(rec.get("equation") or "")
    inst = insts.get(eq.removesuffix("!"))
    free = () if inst is None else free_axes(inst, rec)
    return classify(rec, falses.get(eq), free)


def classify(
    rec: dict[str, Any], false: dict[str, Any] | None = None, free: tuple[str, ...] = ()
) -> dict[str, Any]:
    """``rec`` with ``outcome``, ``stage``, ``cause`` and ``cause_message`` set from its fields
    (``raw_outcome`` keeps a replaced outcome); ``false`` is the F record of an F run. When its
    pair is equal on the run's region (its point and its ``free`` axes, :func:`free_axes`), a
    run with a verdict becomes invalid_point and any other run is flagged ``point_invalid``."""
    tool = str(rec.get("tool") or "")
    oc = rec.get("outcome")
    if "killed_in" in rec and not rec.get("stages"):
        oc2, stage, cause = _killed(rec, tool)
        msg = str(rec.get("detail") or "")
    elif oc == "unsupported":
        oc2, msg = "unsupported", str(rec.get("detail") or "")
        stage = "generate" if tool == "archytaszx" else "prepare"
        cause = "unsupported." + (rec.get("unsupported_code") or unsupported_code(msg))
    elif tool == "archytaszx":
        oc2, stage, cause, msg = _archytaszx(rec)
    else:
        oc2, stage, cause, msg = _rival(rec, tool)
    d = rec.get("d")
    invalid = (
        false is not None and d is not None and equal_region(false, rec.get("n"), int(d), free)
    )
    if invalid and oc2 in VERDICT_OUTCOMES:
        oc2, cause = "invalid_point", "pair_equal_at_point"
    out = {**rec, "outcome": oc2, "stage": stage, "cause": cause, "cause_message": msg[:300]}
    if invalid:
        out["point_invalid"] = True
    if oc2 != oc and "raw_outcome" not in rec:
        out["raw_outcome"] = oc
    if oc2 in ("wrong", "wrong (uncertified)"):
        out["wrong"] = oc2 == "wrong"
    elif "wrong" in rec:
        out["wrong"] = False
    return out


def counted(rec: dict[str, Any]) -> bool:
    """Whether the run counts as the tool's success or failure (not a harness_error, an
    invalid_point or a run flagged ``point_invalid``)."""
    return rec.get("outcome") not in NOT_SCORED and not rec.get("point_invalid")


__all__ = [
    "CAUSES",
    "NOT_SCORED",
    "OUTCOMES",
    "RUNGS",
    "STAGES",
    "UNSUPPORTED",
    "VERDICT_OUTCOMES",
    "classify",
    "counted",
    "equal_region",
    "free_axes",
    "instances",
    "reclassify_record",
    "rung_codes",
    "unsupported_code",
]
