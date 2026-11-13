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
# Builds every rival tool at its pinned version under benchmarks/.venvs/ (or $BENCH_VENVS)
# (docs/BENCHMARKS.txt, section 11), and the harness extra (.[benchmarks]) into the active
# Python. Usage:
# bash benchmarks/tools/setup_rivals.sh [harness|python|dizx|haskell|autoq|quantomatic|coq|all|fetch [TARGET...]]
# fetch downloads every source, wheel, toolchain and dependency the build targets need and builds
# nothing; with BENCH_OFFLINE=1 the build targets make no network calls (wheels from
# $BENCH_WHEELS, ghcup --offline, sbt offline, opam from its download cache).

set -euo pipefail
V="${BENCH_VENVS:-$(cd "$(dirname "$0")/.." && pwd)/.venvs}"
UV="${UV:-uv}"
OFFLINE="${BENCH_OFFLINE:-0}"
FETCH=0
mkdir -p "$V/src" "$V/logs"
UVFLAGS=()
if [ -n "${BENCH_WHEELS:-}" ]; then UVFLAGS=(--no-index --find-links "$BENCH_WHEELS"); fi
if [ "$OFFLINE" = 1 ]; then UVFLAGS+=(--offline); export UV_PYTHON_DOWNLOADS=never; fi

PYENVS=("pyzx pyzx==0.10.8" "quizx quizx==0.4.0" "qcec mqt.qcec==3.11.0"
        "qiskit qiskit==2.6.2" "cirq cirq-core==1.7.0" "tket pytket==2.18.5")
DIZX_DEPS=(numpy matplotlib ipython ipywidgets typing_extensions sympy qiskit==2.6.2)

need_net() {  # need_net WHAT: fails under BENCH_OFFLINE=1
    if [ "$OFFLINE" = 1 ]; then echo "offline: $1 missing; run the fetch target first" >&2; return 1; fi
}

clone() {  # clone URL DIR COMMIT
    [ -d "$V/src/$2/.git" ] || { need_net "$V/src/$2" && git clone -q "$1" "$V/src/$2"; }
    [ "$OFFLINE" = 1 ] || git -C "$V/src/$2" fetch -q --depth 200 origin || true
    git -C "$V/src/$2" checkout -q "$3"
}

uvpip() {  # uvpip VENV PACKAGE...
    VIRTUAL_ENV="$1" "$UV" pip install -q ${UVFLAGS[@]+"${UVFLAGS[@]}"} "${@:2}"
}

harness() {  # the in-process imports of the adapters, into the Python running the harness
    local py="${HARNESS_PYTHON:-$(command -v python)}"
    "$UV" pip install -q --python "$py" -e "$(cd "$(dirname "$0")/../.." && pwd)[benchmarks]"
}

python_envs() {
    if [ "$FETCH" = 1 ]; then  # BENCH_PIP: a pip command of the Python the venvs use
        local pkgs=(numpy "${DIZX_DEPS[@]}")
        for spec in "${PYENVS[@]}"; do set -- $spec; pkgs+=("$2"); done
        ${BENCH_PIP:-python3 -m pip} download -q --only-binary=:all: -d "${BENCH_WHEELS:?}" "${pkgs[@]}"
        return
    fi
    for spec in "${PYENVS[@]}"; do
        set -- $spec
        [ -x "$V/$1/bin/python" ] || "$UV" venv -q -p 3.12 "$V/$1"
        uvpip "$V/$1" "$2" numpy
    done
}

dizx() {  # DiZX at its pinned commit (no package metadata: the source tree on a .pth)
    clone https://github.com/jvdwetering/dizx dizx e387226c99e2fca04c4972c0757687cf43cdac33
    [ "$FETCH" = 0 ] || return 0
    [ -x "$V/dizx/bin/python" ] || "$UV" venv -q -p 3.12 "$V/dizx"
    uvpip "$V/dizx" "${DIZX_DEPS[@]}"
    local site
    site="$("$V/dizx/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "$V/src/dizx" > "$site/dizx-src.pth"
}

haskell() {  # Feynman (feynver) at its pinned commit, GHC 9.4.8 and cabal 3.12.1.0 through ghcup
    export GHCUP_INSTALL_BASE_PREFIX="$V/haskell" BOOTSTRAP_HASKELL_NONINTERACTIVE=1
    export BOOTSTRAP_HASKELL_GHC_VERSION=9.4.8 BOOTSTRAP_HASKELL_CABAL_VERSION=3.12.1.0
    export BOOTSTRAP_HASKELL_INSTALL_NO_STACK=1 BOOTSTRAP_HASKELL_ADJUST_BASHRC=0
    local g="$V/haskell/.ghcup/bin/ghcup" off=()
    [ "$OFFLINE" = 1 ] && off=(--offline --cache)
    export PATH="$V/haskell/.ghcup/bin:$PATH"
    if [ "$FETCH" = 1 ]; then
        if [ ! -x "$g" ]; then
            mkdir -p "$(dirname "$g")"
            curl -sSfL https://downloads.haskell.org/~ghcup/x86_64-linux-ghcup -o "$g" && chmod +x "$g"
        fi
        ghcup prefetch metadata && ghcup prefetch ghc 9.4.8 && ghcup prefetch cabal 3.12.1.0
        clone https://github.com/meamy/feynman feynman d2c382a2ab43a40a87f12f4255645bbb55f704f8
        return
    fi
    [ -x "$g" ] || { need_net ghcup && curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh; }
    ghcup ${off[@]+"${off[@]}"} install ghc 9.4.8 && ghcup set ghc 9.4.8
    ghcup ${off[@]+"${off[@]}"} install cabal 3.12.1.0 && ghcup set cabal 3.12.1.0
    clone https://github.com/meamy/feynman feynman d2c382a2ab43a40a87f12f4255645bbb55f704f8
    (cd "$V/src/feynman" && { [ "$OFFLINE" = 1 ] || cabal update; } \
        && cabal build -w ghc-9.4.8 --constraint="mtl<2.3" exe:feynver \
        && mkdir -p "$V/feynman/bin" && cp "$(cabal list-bin feynver)" "$V/feynman/bin/feynver")
}

autoq() {  # AutoQ 2.0 at its pinned commit (AutoQ-Para has no public release); boost from the
           # system or CMAKE_PREFIX_PATH, the ANTLR4 C++ runtime 4.13.2 from the system or built
           # static under $V/antlr4
    clone https://github.com/fmlab-iis/AutoQ AutoQ 3d92eb355a8850ab19dd3e7d7f620f138deda955
    local a="$V/src/antlr4-4.13.2" zw="$V/src/z3wheel" oldc=0
    if [ ! -d /usr/include/antlr4-runtime ] && [ ! -d "$a" ]; then
        need_net "$a" && curl -sSfL https://github.com/antlr/antlr4/archive/refs/tags/4.13.2.tar.gz \
            | tar xz -C "$V/src"
    fi
    # the bundled libz3.so.4.12 needs glibc 2.34; older hosts use the z3-solver 4.12.5.0 wheel's libz3
    [ "$(printf '%s\n' "$(ldd --version | sed -n '1s/.* //p')" 2.34 | sort -V | head -1)" = 2.34 ] || oldc=1
    if [ "$oldc" = 1 ] && [ "$FETCH" = 1 ] && ! ls "$zw"/z3_solver-4.12.5.0-*.whl >/dev/null 2>&1; then
        ${BENCH_PIP:-python3 -m pip} download -q --only-binary=:all: --no-deps -d "$zw" z3-solver==4.12.5.0
    fi
    [ "$FETCH" = 0 ] || return 0
    local z3=()
    if [ "$oldc" = 1 ]; then
        if [ ! -d "$V/z3/z3/lib" ]; then
            ls "$zw"/z3_solver-4.12.5.0-*.whl >/dev/null 2>&1 || { need_net z3-solver; return 1; }
            mkdir -p "$V/z3" && python3 -m zipfile -e "$(ls "$zw"/z3_solver-4.12.5.0-*.whl | head -1)" "$V/z3"
        fi
        z3=(-DUSE_SYSTEM_Z3=ON -DCMAKE_LIBRARY_PATH="$V/z3/z3/lib")
    fi
    if [ ! -d /usr/include/boost ] && [ -z "${CMAKE_PREFIX_PATH:-}" ]; then
        echo "AutoQ needs boost: on Arch sudo pacman -S --needed boost; on a cluster module load boost" >&2
        return 1
    fi
    if [ ! -d /usr/include/antlr4-runtime ]; then
        if [ ! -f "$V/antlr4/lib/libantlr4-runtime.a" ]; then
            cmake -S "$a/runtime/Cpp" -B "$V/src/antlr4-build" -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$V/antlr4" -DCMAKE_INSTALL_LIBDIR=lib \
                -DANTLR_BUILD_CPP_TESTS=OFF -DANTLR_BUILD_SHARED=OFF -DWITH_DEMO=OFF
            cmake --build "$V/src/antlr4-build" -j "$(nproc)" && cmake --install "$V/src/antlr4-build"
        fi
        export LIBRARY_PATH="$V/antlr4/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi
    (cd "$V/src/AutoQ" && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ${z3[@]+"${z3[@]}"} \
        && cmake --build build -j "$(nproc)" --target autoq \
        && mkdir -p "$V/autoq/bin" && cp build/cli/autoq "$V/autoq/bin/autoq")
}

quantomatic() {  # Quantomatic at its pinned commit with the headless driver, sbt on Temurin JDK 8
    clone https://github.com/Quantomatic/quantomatic quantomatic 0762b5fad411f80ea0bba14b4ad5c801efd7f019
    if [ ! -x "$V/sbt/bin/sbt" ]; then
        need_net sbt && curl -sSL https://github.com/sbt/sbt/releases/download/v1.10.7/sbt-1.10.7.tgz | tar xz -C "$V"
    fi
    if [ ! -x "$V/jdk8/bin/java" ]; then
        need_net jdk8 && mkdir -p "$V/jdk8" && curl -sSL \
            "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jdk/hotspot/normal/eclipse" \
            | tar xz -C "$V/jdk8" --strip-components=1
    fi
    export JAVA_HOME="$V/jdk8" PATH="$V/jdk8/bin:$PATH"
    local q="$V/src/quantomatic/scala" h="$V/sbt-home"
    local sbt=("$V/sbt/bin/sbt" -java-home "$V/jdk8" -sbt-dir "$h/sbt" -sbt-boot "$h/boot" -ivy "$h/ivy2" -batch)
    git -C "$q" checkout -q -- project/plugins.sbt build.sbt
    printf 'addSbtPlugin("com.eed3si9n" %% "sbt-assembly" %% "0.14.5")\n' > "$q/project/plugins.sbt"
    sed -i -e '/appbundle/d' -e 's|http://|https://|g' "$q/build.sbt"
    mkdir -p "$q/src/main/scala/quanto/bench"
    cp "$(dirname "$0")/drivers/QuantoBench.scala" "$q/src/main/scala/quanto/bench/QuantoBench.scala"
    if [ "$FETCH" = 1 ]; then  # the build's dependencies, and the compiler interface by a one-file build
        [ ! -f "$h/fetched" ] || return 0
        (cd "$q" && "${sbt[@]}" update)
        local w="$V/src/sbt-warm"
        mkdir -p "$w/project" "$w/src/main/scala"
        cp "$q/project/build.properties" "$w/project/build.properties"
        printf 'scalaVersion := "2.12.6"\n' > "$w/build.sbt"
        printf 'object Warm\n' > "$w/src/main/scala/Warm.scala"
        (cd "$w" && "${sbt[@]}" compile) && touch "$h/fetched"
        return
    fi
    local off=()
    [ "$OFFLINE" = 1 ] && off=("set every offline := true")
    (cd "$q" && "${sbt[@]}" ${off[@]+"${off[@]}"} assembly \
        && mkdir -p "$V/quantomatic" && cp target/scala-2.12/Quantomatic.jar "$V/quantomatic/quanto.jar" \
        && cp "$V/src/quantomatic/rulesets/red_green/default.rules" "$V/quantomatic/red_green.rules")
}

coq() {  # opam root with switches bench (Coq 8.20.1, why3), sqir23 (Coq 8.16.1) and coqq (Coq 8.18.0);
         # SQIR/VOQC, VyZX, CoqQ, QBricks at pinned commits, all under an ASCII path (Coq's dune
         # build rejects non-ASCII paths). Every opam command names its switch (OPAMSWITCH).
    local C="${BENCH_COQ_HOME:-$HOME/.cache/archytaszx-bench/coq}"
    mkdir -p "$C/src"
    export OPAMROOT="$C/opam" OPAMYES=1 OPAMCONFIRMLEVEL=unsafe-yes OPAMNODEPEXTS=1  # system libraries
                                                                                 # (gmp) from the host
    [ -d "$OPAMROOT" ] || { need_net "$OPAMROOT" && opam init --bare -n --disable-sandboxing; }
    local dl=()
    [ "$FETCH" = 1 ] && dl=(--download-only)
    newswitch() {
        [ -d "$OPAMROOT/$1" ] || { need_net "switch $1" && opam switch create --no-switch "$1" ocaml-base-compiler.4.14.2; }
    }
    coqclone() {  # coqclone URL DIR COMMIT, under the ASCII source directory
        [ -d "$C/src/$2/.git" ] || { need_net "$C/src/$2" && git clone -q "$1" "$C/src/$2"; }
        [ "$OFFLINE" = 1 ] || git -C "$C/src/$2" fetch -q --depth 200 origin || true
        git -C "$C/src/$2" checkout -q "$3"
    }
    repoadd() { [ "$OFFLINE" = 1 ] || opam repo add "$@" || true; }
    have() { opam list -i --short "$1" 2>/dev/null | grep -qx "$1"; }
    oinst() { opam install ${dl[@]+"${dl[@]}"} "$@"; }
    build() { [ "$FETCH" = 1 ] || "$@"; }
    coqclone https://github.com/inQWIRE/SQIR SQIR 134fb942265091baa3e3972f78f72da41827351f
    coqclone https://github.com/inQWIRE/SQIR SQIR-2023 b1a3ef452c95f13a9a3f79fb26064a21df1570ab
    coqclone https://github.com/inQWIRE/VyZX VyZX 10f8c16741952dbaa2bb7a73e13767945e696b8f
    coqclone https://github.com/coq-quantum/CoqQ CoqQ b169ea462b13d20b05e03615ad1d56cc8d8f5784
    coqclone https://github.com/Qbricks/qbricks.github.io qbricks e00fc97f14fb9fcbc44bbfbc5e5ec3e483f87367
    (   # bench: SQIR 134fb94 and VyZX on QuantumLib 1.8.0; why3 and alt-ergo for QBricks
        newswitch bench
        export OPAMSWITCH=bench
        eval "$(opam env)"
        repoadd coq-released https://coq.inria.fr/opam/released
        oinst dune coq.8.20.1
        oinst coq-mathcomp-ssreflect coq-interval || true
        have coq-quantumlib || opam pin add -n coq-quantumlib https://github.com/inQWIRE/QuantumLib.git#0b72c0fa552986c8a6b82f4cb56384c99b9de963 || true
        oinst coq-quantumlib || true
        have coq-sqir || build opam pin add -y coq-sqir "$C/src/SQIR" || true
        have coq-voqc || build opam pin add -y coq-voqc "$C/src/SQIR" || true
        (cd "$C/src/SQIR" && build dune build) || true
        (cd "$C/src/VyZX" && build dune build) || true
        CC="gcc -std=gnu17 -Wno-error=incompatible-pointer-types" \
            oinst why3.1.8.0 alt-ergo.2.6.2 || true  # QBricks' pins; why3's C predates GCC 15
    )
    (   # sqir23: SQIR's Bernstein-Vazirani, Deutsch-Jozsa, Grover, QPE and GHZ proofs, which last
        # build at SQIR b1a3ef4 (2023-04-08), on Coq 8.16.1 and QuantumLib 1.1.0
        newswitch sqir23
        export OPAMSWITCH=sqir23
        eval "$(opam env)"
        repoadd coq-released https://coq.inria.fr/opam/released
        opam pin remove -n coq-quantumlib || true
        oinst dune coq.8.16.1 coq-quantumlib.1.1.0
        (cd "$C/src/SQIR-2023" && build dune build SQIR examples/examples examples/ghz) || true
    )
    (   # coqq: CoqQ's own pins (Coq 8.18.0, mathcomp 2.2, analysis 1.3.1); its dune-configurator
        # 3.12.1 is in the opam archive repository
        newswitch coqq
        export OPAMSWITCH=coqq
        eval "$(opam env)"
        repoadd coq-released https://coq.inria.fr/opam/released
        repoadd archive git+https://github.com/ocaml/opam-repository-archive
        oinst --deps-only "$C/src/CoqQ/coq-mathcomp-quantum.opam" || true
        (cd "$C/src/CoqQ" && build make -j"$(nproc)") || true
    )
}

case "${1:-all}" in
    harness) harness ;;
    python) python_envs ;;
    dizx) dizx ;;
    haskell) haskell ;;
    autoq) autoq ;;
    quantomatic) quantomatic ;;
    coq) coq ;;
    all) harness; python_envs; dizx; haskell; autoq; quantomatic; coq ;;
    fetch)
        FETCH=1
        shift
        for t in "${@:-python dizx haskell autoq quantomatic coq}"; do
            for u in $t; do case $u in python) python_envs ;; dizx|haskell|autoq|quantomatic|coq) "$u" ;;
                *) echo "unknown fetch target $u" >&2; exit 2 ;; esac; done
        done ;;
    *) echo "unknown target $1" >&2; exit 2 ;;
esac
