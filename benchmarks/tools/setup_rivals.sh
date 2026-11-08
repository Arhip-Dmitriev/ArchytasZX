#!/usr/bin/env bash
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
#
# Builds every rival tool at its pinned version under benchmarks/.venvs/ (docs/BENCHMARKS.txt,
# section 11). Usage: bash benchmarks/tools/setup_rivals.sh [python|haskell|autoq|quantomatic|coq|all]

set -euo pipefail
V="$(cd "$(dirname "$0")/.." && pwd)/.venvs"
UV="${UV:-uv}"
mkdir -p "$V/src" "$V/logs"

clone() {  # clone URL DIR COMMIT
    [ -d "$V/src/$2/.git" ] || git clone -q "$1" "$V/src/$2"
    git -C "$V/src/$2" fetch -q --depth 200 origin || true
    git -C "$V/src/$2" checkout -q "$3"
}

python_envs() {
    for spec in "pyzx pyzx==0.10.8" "quizx quizx==0.4.0" "qcec mqt.qcec==3.11.0" \
                "qiskit qiskit==2.6.2" "cirq cirq-core==1.7.0" "tket pytket==2.18.5"; do
        set -- $spec
        [ -x "$V/$1/bin/python" ] || "$UV" venv -q -p 3.12 "$V/$1"
        VIRTUAL_ENV="$V/$1" "$UV" pip install -q "$2" numpy
    done
}

haskell() {  # Feynman (feynver) at its pinned commit, GHC 9.6.7 through ghcup
    export GHCUP_INSTALL_BASE_PREFIX="$V/haskell" BOOTSTRAP_HASKELL_NONINTERACTIVE=1
    export BOOTSTRAP_HASKELL_GHC_VERSION=9.4.8 BOOTSTRAP_HASKELL_CABAL_VERSION=3.12.1.0
    export BOOTSTRAP_HASKELL_INSTALL_NO_STACK=1 BOOTSTRAP_HASKELL_ADJUST_BASHRC=0
    [ -x "$V/haskell/.ghcup/bin/ghcup" ] || curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
    export PATH="$V/haskell/.ghcup/bin:$PATH"
    ghcup install ghc 9.4.8 && ghcup set ghc 9.4.8
    clone https://github.com/meamy/feynman feynman d2c382a2ab43a40a87f12f4255645bbb55f704f8
    (cd "$V/src/feynman" && cabal update && cabal build -w ghc-9.4.8 --constraint="mtl<2.3" exe:feynver \
        && mkdir -p "$V/feynman/bin" && cp "$(cabal list-bin feynver)" "$V/feynman/bin/feynver")
}

autoq() {  # AutoQ 2.0 at its pinned commit (AutoQ-Para has no public release)
    clone https://github.com/fmlab-iis/AutoQ AutoQ 3d92eb355a8850ab19dd3e7d7f620f138deda955
    if [ ! -d /usr/include/boost ] || [ ! -d /usr/include/antlr4-runtime ]; then
        echo "AutoQ needs boost and the ANTLR4 C++ runtime; on Arch: sudo pacman -S --needed boost antlr4-runtime" >&2
        echo "(on Ubuntu or Debian: sudo $V/src/AutoQ/install-dependencies.sh)" >&2
        return 1
    fi
    (cd "$V/src/AutoQ" && make \
        && mkdir -p "$V/autoq/bin" && cp build/cli/autoq "$V/autoq/bin/autoq")
}

quantomatic() {  # Quantomatic at its pinned commit with the headless driver, sbt on Temurin JDK 8
    clone https://github.com/Quantomatic/quantomatic quantomatic 0762b5fad411f80ea0bba14b4ad5c801efd7f019
    [ -x "$V/sbt/bin/sbt" ] || curl -sSL https://github.com/sbt/sbt/releases/download/v1.10.7/sbt-1.10.7.tgz | tar xz -C "$V"
    if [ ! -x "$V/jdk8/bin/java" ]; then
        mkdir -p "$V/jdk8"
        curl -sSL "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jdk/hotspot/normal/eclipse" \
            | tar xz -C "$V/jdk8" --strip-components=1
    fi
    export JAVA_HOME="$V/jdk8" PATH="$V/jdk8/bin:$PATH"
    local q="$V/src/quantomatic/scala"
    git -C "$q" checkout -q -- project/plugins.sbt build.sbt
    printf 'addSbtPlugin("com.eed3si9n" %% "sbt-assembly" %% "0.14.5")\n' > "$q/project/plugins.sbt"
    sed -i -e '/appbundle/d' -e 's|http://|https://|g' "$q/build.sbt"
    mkdir -p "$q/src/main/scala/quanto/bench"
    cp "$(dirname "$0")/drivers/QuantoBench.scala" "$q/src/main/scala/quanto/bench/QuantoBench.scala"
    (cd "$q" && "$V/sbt/bin/sbt" -java-home "$V/jdk8" -batch assembly \
        && mkdir -p "$V/quantomatic" && cp target/scala-2.12/Quantomatic.jar "$V/quantomatic/quanto.jar" \
        && cp "$V/src/quantomatic/rulesets/red_green/default.rules" "$V/quantomatic/red_green.rules")
}

coq() {  # opam root with Coq 8.20.1 and why3; SQIR/VOQC, VyZX, CoqQ, QBricks at pinned commits,
         # all under an ASCII path (Coq's dune build rejects non-ASCII paths)
    local C="${BENCH_COQ_HOME:-$HOME/.cache/archytaszx-bench/coq}"
    mkdir -p "$C/src"
    export OPAMROOT="$C/opam" OPAMYES=1 OPAMCONFIRMLEVEL=unsafe-yes
    [ -d "$OPAMROOT" ] || opam init --bare -n --disable-sandboxing
    [ -d "$OPAMROOT/bench" ] || opam switch create bench ocaml-base-compiler.4.14.2
    eval "$(opam env --switch=bench)"
    opam repo add coq-released https://coq.inria.fr/opam/released || true
    opam install dune coq.8.20.1
    opam install coq-mathcomp-ssreflect coq-interval || true
    opam pin add -n coq-quantumlib https://github.com/inQWIRE/QuantumLib.git#0b72c0fa552986c8a6b82f4cb56384c99b9de963 || true
    opam install coq-quantumlib || true
    coqclone() {  # coqclone URL DIR COMMIT, under the ASCII source directory
        [ -d "$C/src/$2/.git" ] || git clone -q "$1" "$C/src/$2"
        git -C "$C/src/$2" fetch -q --depth 200 origin || true
        git -C "$C/src/$2" checkout -q "$3"
    }
    coqclone https://github.com/inQWIRE/SQIR SQIR 134fb942265091baa3e3972f78f72da41827351f
    coqclone https://github.com/inQWIRE/VyZX VyZX 10f8c16741952dbaa2bb7a73e13767945e696b8f
    coqclone https://github.com/coq-quantum/CoqQ CoqQ b169ea462b13d20b05e03615ad1d56cc8d8f5784
    coqclone https://github.com/Qbricks/qbricks.github.io qbricks e00fc97f14fb9fcbc44bbfbc5e5ec3e483f87367
    opam pin add -y coq-sqir "$C/src/SQIR" || true
    opam pin add -y coq-voqc "$C/src/SQIR" || true
    (cd "$C/src/SQIR" && dune build) || true
    (cd "$C/src/VyZX" && dune build) || true
    CC="gcc -std=gnu17 -Wno-error=incompatible-pointer-types" \
        opam install why3.1.8.0 alt-ergo.2.6.2 || true  # QBricks' pins; why3's C predates GCC 15
    # CoqQ pins Coq 8.18.0 and mathcomp 2.2 / analysis 1.3.1: a switch of its own
    [ -d "$OPAMROOT/coqq" ] || opam switch create coqq ocaml-base-compiler.4.14.2
    opam repo add coq-released https://coq.inria.fr/opam/released --switch=coqq || true
    opam install --switch=coqq --deps-only "$C/src/CoqQ/coq-mathcomp-quantum.opam"
    (eval "$(opam env --switch=coqq)" && cd "$C/src/CoqQ" && make -j6) || true
}

case "${1:-all}" in
    python) python_envs ;;
    haskell) haskell ;;
    autoq) autoq ;;
    quantomatic) quantomatic ;;
    coq) coq ;;
    all) python_envs; haskell; autoq; quantomatic; coq ;;
    *) echo "unknown target $1" >&2; exit 2 ;;
esac
