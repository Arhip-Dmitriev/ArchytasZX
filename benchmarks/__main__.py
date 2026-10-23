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

"""``python -m benchmarks {run,plot} ...`` (see the package docstring)."""

from __future__ import annotations

import argparse
import importlib
import os

from benchmarks import harness

SUITES = ("algorithms",)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmarks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run suites, then plot them")
    r.add_argument("suites", nargs="*", default=["all"], choices=[*SUITES, "all"])
    r.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    r.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="CPU seconds per run (wall-clock safety net: 4x)",
    )
    r.add_argument(
        "--repeat", type=int, default=10, help="separate cold runs per case (median CPU kept)"
    )
    r.add_argument(
        "--mem-gb",
        default="auto",
        help="memory cap per run in GB, 0 = none, auto = 85%% of available memory / jobs",
    )
    r.add_argument("--quick", action="store_true", help="the smaller half of every sweep")
    r.add_argument("--tool", action="append", help="only these tools (repeatable)")
    r.add_argument("--family", action="append", help="only these families (repeatable)")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--resume", action="store_true", help="keep finished rows of an earlier run")
    p = sub.add_parser("plot", help="plot saved results")
    p.add_argument("suites", nargs="*", default=["all"], choices=[*SUITES, "all"])
    args = ap.parse_args(argv)
    suites = SUITES if "all" in args.suites else tuple(args.suites)
    from benchmarks import plots

    for suite in suites:
        if args.cmd == "run":
            module = importlib.import_module(f"benchmarks.{suite}")
            cases = module.cases(args.quick)
            if args.tool:
                cases = [c for c in cases if c["tool"] in args.tool]
            if args.family:
                cases = [c for c in cases if c["family"] in args.family]
            mem = harness.safe_memory_gb(args.jobs) if args.mem_gb == "auto" else float(args.mem_gb)
            print(
                f"== {suite}: {len(cases)} cases x {args.repeat} runs, {args.jobs} jobs, "
                f"{args.timeout:g} s cpu cap, {mem:.2f} GB memory cap"
            )
            rows = harness.run_cases(
                cases,
                jobs=args.jobs,
                timeout=args.timeout,
                repeat=args.repeat,
                mem_gb=mem,
                resume=args.resume,
            )
            print(f"saved {harness.save(suite, rows)}")
            if args.no_plot:
                continue
        for path in plots.plot(suite):
            print(f"plot  {path}")


if __name__ == "__main__":
    main()
