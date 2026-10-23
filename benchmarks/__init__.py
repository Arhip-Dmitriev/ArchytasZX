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

"""Benchmarks of ArchytasZX against Qiskit and PyZX on algorithm outputs over (n, d).

::

    python -m benchmarks run                # run, then plot
    python -m benchmarks run -j 8 --timeout 60 --quick
    python -m benchmarks plot               # replot from benchmarks/results/algorithms.json

Each run is a fresh worker process under a CPU-time and resident-memory cap, in interleaved
passes over every case; a case's status is one of :data:`harness.STATUSES`. Results go to
``benchmarks/results/``, plots to ``benchmarks/plots/``.
"""
