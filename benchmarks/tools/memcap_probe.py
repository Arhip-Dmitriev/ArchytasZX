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

"""Memory-cap probe: ``python -m benchmarks.tools.memcap_probe [--cap-mb M]`` prints one JSON line
per mechanism of benchmarks.memcap (usable, and for a usable one whether a child touching 2 M
MiB was killed as oom and one touching M / 2 MiB finished ok), then a summary line with the
mechanism the runner would choose; exit 0 when every usable mechanism enforced the cap."""

from __future__ import annotations

import argparse
import json
import sys
import time

from benchmarks import memcap


def child(mb: float, hold: float = 4.0) -> list[str]:
    """A Python child touching ``mb`` MiB of anonymous memory and holding it ``hold`` seconds."""
    code = f"import time; b = b'\\x01' * {int(mb * 2**20)}; time.sleep({hold}); print('done')"
    return [sys.executable, "-c", code]


def trial(mech: str, cap_mb: float, mb: float) -> dict[str, object]:
    """One child of ``mb`` MiB under a ``cap_mb`` cap by ``mech``: exit code, oom, wall."""
    cap = memcap.Cap(cap_mb, mech, tag="probe")
    w0 = time.perf_counter()
    proc, oom = memcap.run(child(mb), cap, 120)
    return {
        "mech": cap.mech,
        "exit": proc.returncode,
        "oom": oom or proc.returncode in (-9, 137),
        "wall": round(time.perf_counter() - w0, 2),
        "stderr": (proc.stderr or "")[-200:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-mb", type=float, default=256.0)
    args = ap.parse_args()
    good = True
    for mech in memcap.ALL:
        row: dict[str, object] = {"mechanism": mech, "usable": memcap.usable(mech)}
        if row["usable"]:
            over = trial(mech, args.cap_mb, 2 * args.cap_mb)
            under = trial(mech, args.cap_mb, args.cap_mb / 2)
            row |= {"over": over, "under": under}
            row["enforced"] = bool(over["oom"]) and under["exit"] == 0 and not under["oom"]
            good = good and bool(row["enforced"])
        print(json.dumps(row), flush=True)
    print(json.dumps({"summary": memcap.describe(), "all_enforced": good}), flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
