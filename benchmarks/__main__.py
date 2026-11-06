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

"""``python -m benchmarks run|plot [SUITE ...]``: suites run one after another;
``python -m benchmarks anc OUT_DIR``: the ancillary-file export."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from benchmarks import harness

SUITES = (
    "scaling",
    "size",
    "families",
    "refutations",
    "lean",
    "completeness",
    "breadth",
    "certificates",
    "exhaustive",
    "limits",
    "ablation",
    "qudits",
    "mixed",
    "sweep",
    "uniformity",
    "restricted",
)
DEFAULT_REPEAT = 10
DEFAULT_TIMEOUT = 600.0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "anc":
        from benchmarks import anc

        anc.main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    parser.add_argument("action", choices=("run", "plot"))
    parser.add_argument("suites", nargs="*", choices=SUITES, default=list(SUITES[:2]))
    parser.add_argument("--quick", action="store_true", help="a few cases per suite")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=float, default=None, help="CPU seconds per run")
    parser.add_argument("--repeat", type=int, default=None, help="runs per case")
    parser.add_argument("--resume", action="store_true", help="keep finished cases")
    parser.add_argument("--results", default=None, help="results directory")
    parser.add_argument("--plots", default=None, help="plots directory")
    args = parser.parse_args()
    if args.results:
        os.environ[harness.RESULTS_ENV] = os.path.abspath(args.results)
    if args.plots:
        os.environ["BENCH_PLOTS"] = os.path.abspath(args.plots)
    if args.action == "run":
        for suite in args.suites:
            module = importlib.import_module(f"benchmarks.{suite}")
            harness.run_suite(
                suite,
                quick=args.quick,
                jobs=args.jobs,
                timeout=args.timeout or getattr(module, "TIMEOUT", DEFAULT_TIMEOUT),
                repeat=args.repeat or getattr(module, "REPEAT", DEFAULT_REPEAT),
                resume=args.resume,
                case_defaults=args.timeout is None and args.repeat is None,
            )
    from benchmarks import plots

    for suite in args.suites:
        plots.PLOTS[suite]()


if __name__ == "__main__":
    main()
