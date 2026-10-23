# ArchytasZX

**Develop quantum algorithms without fixing the size or dimensionality of the machine.** Write a circuit once,
for `n` qudits of dimension `d`, and get its states, measurement statistics and proofs of its
identities as closed forms that hold for every `n` and every `d` at once.

[![PyPI](https://img.shields.io/pypi/v/archytaszx.svg)](https://pypi.org/project/archytaszx/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/LICENSE)
[![Proofs checked in Lean 4](https://img.shields.io/badge/proofs-Lean%204-purple.svg)](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/LEAN.md)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](https://github.com/Arhip-Dmitriev/ArchytasZX#status)

ArchytasZX is a symbolic engine built on the qufinite ZX-calculus, with integration into Dirac notation. Qudit count and dimension are symbols, so nothing is ever enumerated, sampled or
simulated: every result comes from exact rewriting and has the same shape for every size.

## Example: Deutsch–Jozsa for every `n`

[`examples/deutsch_jozsa.azx`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/examples/deutsch_jozsa.azx), a balanced oracle on an `n`-qubit register,
with `n` never given a value:

```text
var n
gate U_f = |x>|y> -> |x>|y xor bit(x, 0)>
|psi> = |0>^@n @ |1>
(H^@n @ H)|psi>
U_f|psi>
(H^@n @ I)|psi>
var z = floor(measure|psi>/2)
```

```console
$ archytaszx --unicode examples/deutsch_jozsa.azx
|ψ⟩ = |0⟩^⊗n ⊗ |1⟩
|ψ⟩ = 2^(-n/2 - 1/2) Σ_{y=0}^{1} Σ_{x=0}^{2^n - 1} (-1)^y |x⟩ ⊗ |y⟩
|ψ⟩ = 2^(-n/2 - 1/2) Σ_{y=0}^{1} Σ_{x=0}^{2^n - 1} (-1)^y |x⟩ ⊗ |(y ⊕ bit(x, 0))⟩
|ψ⟩ = 2^(-n - 1/2) Σ_{y=0}^{1} Σ_{x=0}^{2^n - 1} Σ_{x'=0}^{2^n - 1} (-1)^(y + x·x') |x'⟩ ⊗ |(y ⊕ bit(x, 0))⟩
z = [(1, 1)]
|ψ⟩⟨ψ| = 1/2 Σ_{k1=0}^{1} |1⟩⟨1| ⊗ |k1⟩⟨k1|
```

The measurement is a Born distribution derived symbolically: outcome `1` with probability `1`,
so the oracle is balanced, for every register size. The post-measurement state is the mixed
state on the last line. The same states can be drawn as ZX diagrams (`--zx`), where the
register is a bang box of multiplicity `n - 1` rather than `n` copies:

<p align="center">
  <img src="https://raw.githubusercontent.com/Arhip-Dmitriev/ArchytasZX/main/docs/images/deutsch_jozsa_zx.png" alt="ZX diagram of the Deutsch–Jozsa state before measurement, with the register as a bang box of multiplicity n - 1" width="640">
</p>

The constant-oracle case over qudits,
[`examples/deutsch_jozsa_proof.azx`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/examples/deutsch_jozsa_proof.azx),
proves that `F†^⊗n F^⊗n |0⟩^⊗n = |0⟩^⊗n` for every `n` and every `d` and writes the proof
out as a Lean 4 file.

## What it does

- **Symbolic size.** Qudit count, dimension and multiplicities stay symbols. `|0>^@30` is one
  leg under a bang box, not thirty, and a concrete value is substituted into the closed form at
  the end at no cost in its size.
- **Exact arithmetic.** Scalars are tracked exactly (roots of unity, character sums, Gauss sums);
  no global factor is ever dropped, so results are equalities, not proportionalities.
- **Proofs for all `n`.** Diagram rewriting, a normal-form procedure, equality saturation and
  proof search decide equalities; induction over bang-box multiplicities proves whole families.
- **Independent checking.** Every proof emits a replayable certificate, is re-checked by a small
  proof kernel, and can be exported to Lean 4 and Mathlib, where it is checked again from scratch.
- **A language for algorithms.** Dirac notation with gates, sums, measurement, control flow and
  f-strings, run as scripts or interactively, and embeddable in Python and Jupyter.

## Install

Python ≥ 3.11. `numpy` and `sympy` are the only runtime dependencies. With `matplotlib`
installed, output is typeset and ZX diagrams are drawn as images in terminals that show them;
without it, both fall back to Unicode text.

```bash
pip install archytaszx           # or 'archytaszx[viz]' to add matplotlib
archytaszx                       # interactive shell
archytaszx script.azx            # run a script
```

## Quickstart

### The `.azx` language

```text
var n
gate F = |x>_d -> 1/sqrt(d) sum(y in 0..d){ e^(2*pi*i*x*y/d) |y>_d }
|psi> = H^@n |0>^@n
var z = measure|psi>
show(z, |psi>)
print(f"z = {z}")
```

Output styles are `--render` (default, typeset images), `--unicode`, `--ascii` and `--latex`;
several style flags print every output in each style, and `--order` groups the output by style
at the end. `--zx`, `--both` and `--raw` switch between Dirac notation and ZX diagrams, and
`--images DIR` saves every rendered image as a PNG. In the shell, commands starting with `:`
drive the engine directly (`:help` lists them).

[`examples/learn_azx.azx`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/examples/learn_azx.azx)
is a guided tour of the whole language, ending in Deutsch–Jozsa for every `n`; the full syntax is
in [`SYNTAX.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/SYNTAX.txt).

### From Python

```python
from archytaszx import azx               # a shared Program; archytaszx.Program() for a fresh one

azx("var n")
azx.psi = azx.H ** azx.n | azx.ket(0) ** azx.n      # |psi> = H^@n |0>^@n
azx["|phi>"] = azx["|psi> - 2 |0><0|psi>"]
azx.phi       # |φ⟩ = 2^(-n/2) Σ_{x=0}^{2^n - 1} |x⟩ - 2^(1 - n/2) |0⟩^⊗n
azx.phi.latex()                          # also rendered by Jupyter

y = azx.measure | azx.psi
y             # [(k, 2^(-n)) : k ∈ {0,…,2^n - 1}]
```

[`examples/learn_azx.py`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/examples/learn_azx.py)
is the same tour from Python.

### The diagram engine

Build two Z spiders over a symbolic dimension `d`, fuse them, check the rewrite, and prove it for
every count at once.

```python
from archytaszx.algebra.dimension import Dim
from archytaszx.diagram.generators import Z_SPIDER
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import SPIDER_FUSION

d = Dim.symbol("d")

g = Diagram()
a = g.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
b = g.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
g.add_wire(PortRef(a, Direction.OUTPUT, 0), PortRef(b, Direction.INPUT, 0))
g.set_boundary_outputs([
    PortRef(a, Direction.OUTPUT, 1),
    PortRef(b, Direction.OUTPUT, 0),
    PortRef(b, Direction.OUTPUT, 1),
])

result = apply(g, SPIDER_FUSION, find_matches(g)[0])

len(result.diagram.nodes)             # 1         -- two spiders became one
len(result.diagram.boundary_outputs)  # 3         -- the boundary is unchanged
result.diagram.scalar                 # Scalar(1) -- exact, never discarded
```

A certificate records the derivation; `verify` replays it step by step, and tampering with any
recorded field (scalar, rule, consumed wire, side condition, multiplicity) fails the replay.

```python
from archytaszx.semantics.check import compare
from archytaszx.semantics.certificate import certify, verify

compare(g, result.diagram, {"d": 5}).matched   # True -- numeric oracle at d = 5

certificate = certify(g, [result], label="fuse A into B")
verify(certificate, {"d": 3}).verified         # True
```

Wrap the graph in a bang box of multiplicity `m` and prove the fusion for every `m`, with `d`
still symbolic:

```python
from archytaszx.diagram.bangbox import abstract_subgraph_count
from archytaszx.semantics.induction import prove_by_induction

boxed, box_id, mult = abstract_subgraph_count(g, frozenset({a, b}), 1, stem="m")
fused = apply(boxed, SPIDER_FUSION, find_matches(boxed)[0]).diagram

proof = prove_by_induction(boxed, fused, witness={"d": 2})

proof.proved          # True
proof.verdict         # Verdict.PROVED_UNIFORM -- for all m, not a finite schema check
proof.index           # 'm'
proof.discharge       # StepDischarge.UNIFORM_REWRITE -- the tier that settled the step
proof.held_symbolic   # frozenset({'d'}) -- what stayed universally quantified
```

Concrete numbers typed by the user are abstracted back into symbols, with the values kept for
output; prefix a numeral with `literal` to keep it concrete:

```python
from archytaszx.repl.parser import parse_dirac_source

p = parse_dirac_source("sum_{k=0}^{3-1} |k>^{30}; copy")
dict(p.parameters)   # {'d': 3, 'n': 29}
```

## Formal proofs in Lean

Proofs found by the engine are exported to Lean 4 (`:certificate lean proof.lean`) against a
Mathlib library bundled in `archytaszx/lean`, in which a diagram is a tensor network over ℂ and
every rewrite rule is a lemma proved once. Families over multiplicities become Lean
`induction`s, equalities the rewriter cannot reach are closed by a verified normaliser of
symbolic sums, and verdicts that two diagrams differ export a Lean refutation. `#print axioms`
lists only Lean's standard axioms; any step without a lemma is a named `unproved_*` axiom, and
the REPL reports how many there are.

```sh
cd archytaszx/lean && lake exe cache get && lake build      # once
lake env lean /path/to/proof.lean
```

[`docs/LEAN.md`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/LEAN.md) has the
semantics, the trust boundary, the normaliser and the refutation method.

## Architecture

Four engine layers, depending strictly downward, with the language on top:

| Layer | Package | Contents |
|---|---|---|
| A | [`algebra`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/algebra) | Dimension expressions with one canonical form and a unifier; phase vectors tied to `d`; exact scalars over roots of unity with a character-sum simplifier |
| B | [`diagram`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/diagram) | Qufinite ZXW generators (Z, X, Fourier box, triangle, W, dimension binder/splitter) with per-port dimensions; bang boxes and scalable sheet wires; validation |
| C | [`rewrite`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/rewrite) | Rule library, matcher and engine with provenance; normal form; diagram-level e-graph; tactics and bidirectional proof search |
| D | [`semantics`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/semantics) | Symbolic and numeric contraction; path sums and Born distributions; the `decide_equal` ladder; induction; certificates, proof kernel and Lean export |
| — | [`repl`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/repl) | The `.azx` language, parser and printer, ZX view and drawing, shell |

Sub-package `__init__.py` files are empty; `archytaszx` itself exports only `Program`, `Value`
and `azx`, so engine objects are imported by full module path. Each module raises only its own
`<Area>Error` hierarchy: `<Area>GrammarError` for a wrong call and `<Area>DomainError` for a case
not implemented yet.

## Status

A **research prototype under active development**: the API is unstable and changes between
releases. Equality checking is sound but not complete, so `decide` and `prove` can answer
"unknown". Known limits:

- Symbolic contraction slows sharply as the number of phased spiders grows.
- Symbolic contraction refuses a bang box wired to a node outside it, and a port box over two
  W nodes.
- The dimension unifier reports `DEFERRED` where an equation has several solution branches
  (`d1*d2 == 6`, `d**n == d**m`); deferred dimensions are recorded as assumptions and appear in
  certificates and Lean theorems as hypotheses.
- A rule restricted to prime `d` refuses to fire at a symbolic `d`.
- `compare_symbolic` has three outcomes (equal, unequal, indeterminate); read `.reason` when it
  does not match.
- Steps the Lean export cannot prove yet are listed in
  [`LEAN_GAPS.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/LEAN_GAPS.txt).

## Development

```bash
git clone https://github.com/Arhip-Dmitriev/ArchytasZX.git
cd ArchytasZX
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

python -m pytest -n 8     # default tier
python -m pytest -m slow  # fuzz sweeps and oracle differentials
ruff format . && ruff check . && mypy archytaszx
```

The default tier is 10,072 tests and the slow tier 3,898 (counted 2026-10-22). `ruff format` is
authoritative for layout; `mypy` runs in strict mode.

## Citation

If you use ArchytasZX in academic work, please cite it via
[`CITATION.cff`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/CITATION.cff), or:

```bibtex
@software{dmitriev_archytaszx,
  author  = {Dmitriev, Arkhip Alekseyevich},
  title   = {ArchytasZX: size-agnostic quantum algorithm development with the symbolic qufinite ZX-calculus},
  year    = {2026},
  url     = {https://github.com/Arhip-Dmitriev/ArchytasZX},
  license = {Apache-2.0}
}
```

## Contributing

Bug reports, counterexamples and questions about the mathematics are the most useful
contributions; a soundness bug (a rewrite that changes the denoted map, scalar included) is the
most valuable report of all. See
[`CONTRIBUTING.md`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/CONTRIBUTING.md).

## References

ArchytasZX builds chiefly on the qufinite ZX-calculus of Wang
([arXiv:2104.06429](https://arxiv.org/abs/2104.06429)), the ZXW completeness results of Poór,
Wang et al. ([arXiv:2302.12135](https://arxiv.org/abs/2302.12135),
[arXiv:2309.13014](https://arxiv.org/abs/2309.13014)), and bang boxes and scalable notation
([arXiv:2204.11702](https://arxiv.org/abs/2204.11702)). Full BibTeX is in
[`REFERENCES.bib`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/REFERENCES.bib).

<details>
<summary>Full reference list</summary>

**Dirac notation and quantum foundations**
- Dirac, *A new notation for quantum mechanics*, Math. Proc. Camb. Phil. Soc. 35 (1939) —
  [doi:10.1017/S0305004100021162](https://doi.org/10.1017/S0305004100021162)
- Born, *Zur Quantenmechanik der Stoßvorgänge*, Z. Phys. 37 (1926), the Born rule —
  [doi:10.1007/BF01397477](https://doi.org/10.1007/BF01397477)
- Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010), density matrices,
  measurement, the quantum Fourier transform —
  [doi:10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)
- Gottesman, *Fault-tolerant quantum computation with higher-dimensional systems* (1999), qudits —
  [arXiv:quant-ph/9802007](https://arxiv.org/abs/quant-ph/9802007)

**ZX-calculus, qudit and qufinite ZX/ZXW**
- Penrose, *Applications of negative dimensional tensors* (1971), tensor-network diagrams
- Coecke & Duncan, *Interacting quantum observables*, ICALP 2008 / NJP 13 (2011) —
  [arXiv:0906.4725](https://arxiv.org/abs/0906.4725)
- Coecke & Kissinger, *Picturing Quantum Processes* (2017) —
  [doi:10.1017/9781316219317](https://doi.org/10.1017/9781316219317)
- van de Wetering, *ZX-calculus for the working quantum computer scientist* —
  [arXiv:2012.13966](https://arxiv.org/abs/2012.13966)
- Ranchin, *Depicting qudit quantum mechanics and mutually unbiased qudit theories*, EPTCS 172 (2014) —
  [arXiv:1404.1288](https://arxiv.org/abs/1404.1288)
- Hadzihasanovic, *A diagrammatic axiomatisation for qubit entanglement*, LICS 2015, the W node —
  [arXiv:1501.07082](https://arxiv.org/abs/1501.07082)
- Wang, *Qufinite ZX-calculus: a unified framework of qudit ZX-calculi*, the generator set,
  triangle, Fourier box and dimension connectives — [arXiv:2104.06429](https://arxiv.org/abs/2104.06429)
- Poór, Wang, Shaikh, Yeh, Yeung & Coecke, *Completeness for arbitrary finite dimensions of
  ZXW-calculus*, LICS 2023 — [arXiv:2302.12135](https://arxiv.org/abs/2302.12135)
- Wang, Poór & Shaikh, *Completeness of qufinite ZXW calculus* —
  [arXiv:2309.13014](https://arxiv.org/abs/2309.13014)
- Shaikh, Wang & Yeung, *How to sum and exponentiate Hamiltonians in ZXW calculus*, EPTCS 394 (2023) —
  [arXiv:2212.04462](https://arxiv.org/abs/2212.04462)

**Bang boxes and scalable notation**
- Dixon & Duncan, *Graphical reasoning in compact closed categories for quantum computation*,
  AMAI 56 (2009) — [doi:10.1007/s10472-009-9141-x](https://doi.org/10.1007/s10472-009-9141-x)
- Dixon, Duncan & Kissinger, *Open graphs and computational reasoning*, EPTCS 26 (2010) —
  [arXiv:1007.3794](https://arxiv.org/abs/1007.3794)
- Kissinger, Merry & Soloviev, *Pattern graph rewrite systems*, EPTCS 143 (2014) —
  [arXiv:1204.6695](https://arxiv.org/abs/1204.6695)
- Kissinger & Zamdzhiev, *Equational reasoning with context-free families of string diagrams*,
  ICGT 2015 — [arXiv:1504.02716](https://arxiv.org/abs/1504.02716)
- Carette, Horsman & Perdrix, *SZX-calculus: scalable graphical quantum reasoning*, MFCS 2019 —
  [arXiv:1905.00041](https://arxiv.org/abs/1905.00041)
- Carette & Lemonnier, *Large-scale quantum diagrammatic reasoning tools, !-boxes vs. scalable
  notations* — [arXiv:2204.11702](https://arxiv.org/abs/2204.11702)

**Path sums and exact symbolic sums**
- Amy, *Towards large-scale functional verification of universal quantum circuits*, EPTCS 287
  (2019) — [arXiv:1805.06908](https://arxiv.org/abs/1805.06908)
- Vilmart, *The structure of sum-over-paths, its consequences, and completeness for Clifford*,
  FoSSaCS 2021 — [arXiv:2003.05678](https://arxiv.org/abs/2003.05678)
- Berndt, Evans & Williams, *Gauss and Jacobi Sums* (1998), quadratic Gauss sums and their sign
- Landsberg, *Zur Theorie der Gaussschen Summen…*, J. reine angew. Math. 111 (1893),
  Landsberg–Schaar reciprocity — [doi:10.1515/crll.1893.111.234](https://doi.org/10.1515/crll.1893.111.234)
- Dechter, *Bucket elimination*, Artif. Intell. 113 (1999), contraction by variable elimination —
  [doi:10.1016/S0004-3702(99)00059-4](https://doi.org/10.1016/S0004-3702(99)00059-4)
- Richardson, *Some undecidable problems involving elementary functions of a real variable*,
  J. Symb. Logic 33 (1968) — [doi:10.2307/2271358](https://doi.org/10.2307/2271358)

**Rewriting and equality saturation**
- Tate, Stepp, Tatlock & Lerner, *Equality saturation*, LMCS 7 (2011) —
  [arXiv:1012.1802](https://arxiv.org/abs/1012.1802)
- Willsey et al., *egg: fast and extensible equality saturation*, POPL 2021 —
  [arXiv:2004.03082](https://arxiv.org/abs/2004.03082)
- Nelson & Oppen, *Fast decision procedures based on congruence closure*, JACM 27 (1980) —
  [doi:10.1145/322186.322198](https://doi.org/10.1145/322186.322198)
- Tarjan, *Efficiency of a good but not linear set union algorithm*, JACM 22 (1975) —
  [doi:10.1145/321879.321884](https://doi.org/10.1145/321879.321884)

**Lean formalisation**
- de Moura & Ullrich, *The Lean 4 theorem prover and programming language*, CADE-28 (2021) —
  [doi:10.1007/978-3-030-79876-5_37](https://doi.org/10.1007/978-3-030-79876-5_37)
- The mathlib Community, *The Lean mathematical library*, CPP 2020 —
  [arXiv:1910.09336](https://arxiv.org/abs/1910.09336)
- Lehmann, Caldwell, Shah, Spencer & Rand, *VyZX: formal verification of a graphical quantum
  language* — [arXiv:2311.11571](https://arxiv.org/abs/2311.11571)

**Algorithms in the examples**
- Deutsch & Jozsa, *Rapid solution of problems by quantum computation*, Proc. R. Soc. A 439 (1992) —
  [doi:10.1098/rspa.1992.0167](https://doi.org/10.1098/rspa.1992.0167)
- Grover, *A fast quantum mechanical algorithm for database search*, STOC 1996 —
  [arXiv:quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043)
- Shor, *Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum
  computer*, SIAM J. Comput. 26 (1997) — [arXiv:quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027)
- Coppersmith, *An approximate Fourier transform useful in quantum factoring* (1994) —
  [arXiv:quant-ph/0201067](https://arxiv.org/abs/quant-ph/0201067)

**Software**
- Kissinger & van de Wetering, *PyZX*, EPTCS 318 (2020) — [arXiv:1904.04735](https://arxiv.org/abs/1904.04735)
- Kissinger & Zamdzhiev, *Quantomatic*, CADE-25 (2015) — [arXiv:1503.01034](https://arxiv.org/abs/1503.01034)
- de Felice, Toumi & Coecke, *DisCoPy*, EPTCS 333 (2021) — [arXiv:2005.02975](https://arxiv.org/abs/2005.02975)
- Harris et al., *Array programming with NumPy*, Nature 585 (2020) —
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- Meurer et al., *SymPy*, PeerJ Comput. Sci. 3 (2017) — [doi:10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103)

</details>

## License

Apache License 2.0 — see
[`LICENSE`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/LICENSE).

<!--
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
