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

"""python -m benchmarks run [--profile laptop|cluster] [--set k=v ...] [--shard I/N]
                           [--only GLOB ...]
python -m benchmarks false [--set k=v ...]          generate the false set
python -m benchmarks aggregate [--set k=v ...]      aggregates from the logs
python -m benchmarks plot [--set k=v ...]           plots from the aggregates
python -m benchmarks anc OUT_DIR                    the ancillary-file export
python -m benchmarks list [--set k=v ...]           the S equations and units
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmarks")
    ap.add_argument("action", choices=("run", "false", "aggregate", "plot", "anc", "list", "slurm"))
    ap.add_argument("target", nargs="?", default=None, help="OUT_DIR for anc")
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", choices=("laptop", "cluster"), default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", action="append", default=None, help="unit id globs (EQUATION|TOOL)")
    args = ap.parse_args()
    from pathlib import Path

    from benchmarks import config

    cfg = config.load(Path(args.config) if args.config else None, args.sets, args.profile)
    if args.action == "run":
        from benchmarks import runner

        i, n = (int(v) for v in args.shard.split("/"))
        runner.run(cfg, (i, n), args.only)
    elif args.action == "false":
        from benchmarks import breaks

        recs = breaks.generate(
            cfg, config.path(cfg, "results") / "false_set.json", int(cfg["platform"]["jobs"])
        )
        for r in recs:
            print(
                r["equation"],
                r.get("kind"),
                r.get("label"),
                "skipped:",
                ",".join(r.get("skipped", [])),
            )
    elif args.action == "aggregate":
        from benchmarks import aggregate

        aggregate.main(cfg)
    elif args.action == "plot":
        from benchmarks import plots

        plots.main(cfg)
    elif args.action == "anc":
        from benchmarks import anc

        if not args.target:
            sys.exit("anc needs OUT_DIR")
        anc.main(cfg, Path(args.target))
    elif args.action == "slurm":
        from benchmarks import slurm

        slurm.main(cfg, args)
    else:
        from benchmarks import subject

        sc = cfg["subjects"]
        subs = subject.select(
            subject.catalogue(sc["catalogue"]), sc["include"], sc["exclude"], sc["categories"]
        )
        for s in subs:
            for i in subject.instances([s]):
                print(f"{i.id:50s} {s.category:10s} sym={subject.has_symbolic(i)}")


if __name__ == "__main__":
    main()
