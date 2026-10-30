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

"""``python -m benchmarks run|plot [SUITE ...]``: suites run one after another."""

from __future__ import annotations

import argparse
import os

from benchmarks import harness

SUITES = ("scaling", "size")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    parser.add_argument("action", choices=("run", "plot"))
    parser.add_argument("suites", nargs="*", choices=SUITES, default=list(SUITES))
    parser.add_argument("--quick", action="store_true", help="a few cases per suite")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--timeout", type=float, default=600.0, help="CPU seconds per run")
    parser.add_argument("--repeat", type=int, default=10, help="runs per case")
    parser.add_argument("--resume", action="store_true", help="keep finished cases")
    args = parser.parse_args()
    if args.action == "run":
        for suite in args.suites:
            harness.run_suite(
                suite,
                quick=args.quick,
                jobs=args.jobs,
                timeout=args.timeout,
                repeat=args.repeat,
                resume=args.resume,
            )
    from benchmarks import plots

    for suite in args.suites:
        plots.PLOTS[suite]()


if __name__ == "__main__":
    main()
