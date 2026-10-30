# ArchytasZX

**Prove identities of quantum algorithms without fixing the size or dimensionality of the
machine.** Write a circuit once as a ZX diagram, for `n` qudits of dimension `d`, and get
rewrites, decisions of equality and proofs that hold for every `n` and every `d` at once.

[![PyPI](https://img.shields.io/pypi/v/archytaszx.svg)](https://pypi.org/project/archytaszx/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/LICENSE)
[![Proofs checked in Lean 4](https://img.shields.io/badge/proofs-Lean%204-purple.svg)](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/LEAN.txt)
[![Status: beta](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/Arhip-Dmitriev/ArchytasZX#status)

ArchytasZX is a symbolic engine for the qufinite ZX(W)-calculus. Qudit count, dimension and
multiplicities are symbols, so nothing is ever enumerated, sampled or simulated: every
rewrite, decision and proof holds for every size at once, and every proof can be re-checked in
Lean 4. To write algorithms in Dirac notation instead of diagrams, see its sister project
[DiracULA](https://github.com/Arhip-Dmitriev/DiracULA).

## Example: Deutsch–Jozsa for every `n` and every `d`

[`examples/deutsch_jozsa_proof.azx`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/examples/deutsch_jozsa_proof.azx)
writes the qudit Deutsch–Jozsa circuit as ZXW diagrams, with the register a bang box of
symbolic multiplicity, and proves that it returns `e^{2πi c/d} |0⟩^⊗n ⊗ F†|1⟩` for the constant
oracle and `|0⟩^⊗(n-1) ⊗ |1⟩ ⊗ F†|1⟩` for the balanced one:

```console
$ archytaszx --unicode examples/deutsch_jozsa_proof.azx
...
decide: equal by symbolic_contraction: symbolic contractions agree: entries are exactly equal with d formal
proof: re-checkable; certificate kernel / certificate lean
lean: tensor-network semantics, the normal form computed by the Lean kernel
lean: wrote deutsch_jozsa_constant.lean
...
decide: equal by symbolic_contraction: symbolic contractions agree: entries are exactly equal with d formal
...
lean: wrote deutsch_jozsa_balanced.lean
```

Both equalities hold for every `n` and every `d`, and both are written out as Lean 4 files.

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

## Install

Python ≥ 3.12. `numpy` and `sympy` are the only runtime dependencies. With `matplotlib`
installed, ZX diagrams are drawn as images in terminals that show them; without it, they fall
back to Unicode text.

```bash
pip install archytaszx           # or 'archytaszx[viz]' to add matplotlib
archytaszx                       # interactive shell
archytaszx commands.azx          # run a command file
```

## Quickstart

### The shell

```console
$ archytaszx --unicode -c 'load sum_{k=0}^{3-1} |k>^{30}; copy'
loaded dirac input
Σ_{k0,k1=0}^{d-1} [k0 = k1] |k0⟩^{n} |k1, k1⟩
at d = 3, n = 29: Σ_{k0,k1=0}^{3-1} [k0 = k1] |k0⟩^{29} |k1, k1⟩
```

The DSL reads as a circuit: declare nodes, then connect them with arrows.

```console
$ archytaszx --unicode -c 'a = Z(0 -> 2) phase{1: alpha}; b = Z(1 -> 2); a -> b' \
                       -c 'apply spider_fusion'
loaded dsl input
Σ_{k0,k1=0}^{d-1} e^{2 π i (alpha [k0=1])} [k0 = k1] |k0, k1, k1⟩
step: spider_fusion (consumed n0, n1, new n2, scalar 1)
result:
  Z_{0->3}(1: alpha) = Σ_{k=0}^{d-1} e^{2 π i (alpha [k=1])} |k, k, k⟩
```

A command file spans a diagram over indented lines;
[`examples/deutsch_jozsa_proof.azx`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/examples/deutsch_jozsa_proof.azx)
writes n data qudits as `repeat n: X(0 -> 1) -> F^4`.

Diagrams are loaded in a DSL, in Dirac notation or as a graph listing, then rewritten,
normalised, decided and proved with commands such as `apply`, `normalize`, `decide`, `prove`
and `induct`; `help` lists them all. `--zx`, `--both` and `--raw` switch between Dirac notation
and ZX drawings. The reference is
[`docs/ENGINE_COMMANDS.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/ENGINE_COMMANDS.txt).

### From Python

`azx` builds diagrams with Python operators and runs the engine on them. The qudit
Deutsch–Jozsa circuit with a constant oracle, proved for every `n` and `d`:

```python
from archytaszx import azx

n, d = azx.count("n"), azx.d
data = (azx.F ^ 4 | azx.X(0, 1)) ** n                          # a !-box of n data qudits
target = azx.X(2, 1) | azx.F ^ 3 | azx.X(0, 1, phase="-k/d")   # the target through U_f
circuit = data @ target * d ** (-n / 2)
goal = (azx.X(0, 1) ** n @ azx.Z(0, 1, phase="-k/d") @ azx.Z(1, 0, phase="k/d")) \
    * d ** (-(n + 1) / 2)

azx.decide(circuit, goal).verdict      # 'equal'
azx.lean("deutsch_jozsa.lean")         # the Lean 4 theorem
```

`a | b` feeds `b` into `a`, `@` is the tensor product, `** n` repeats a piece in a !-box, `^ k`
puts `k` copies in series and `*` scales. `azx("...")` runs any engine command.

The engine objects can also be used directly. Build two Z spiders over a symbolic dimension `d`, fuse them, check the rewrite, and prove it for
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

## Documentation

The full reference documentation is in [`docs/`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/docs):

| Document | Contents |
|---|---|
| [`INDEX.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/INDEX.txt) | Index of the documentation, conventions and reading order |
| [`OVERVIEW.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/OVERVIEW.txt) | What the engine is, its guarantees and its limits |
| [`INSTALL.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/INSTALL.txt) | Installation, the Lean library, configuration and environment variables |
| [`SHELL.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/SHELL.txt) | The `archytaszx` command: flags, views, shell commands, exit codes |
| [`ENGINE_COMMANDS.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/ENGINE_COMMANDS.txt) | The `:` engine commands and the diagram input forms |
| [`PYTHON_API.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/PYTHON_API.txt) | The engine import map and a worked example |
| [`ALGEBRA.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/ALGEBRA.txt) | Layer A: dimensions, phases, exact scalars |
| [`DIAGRAM.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/DIAGRAM.txt) | Layer B: generators, diagrams, bang boxes, scalable notation, validation |
| [`REWRITE.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/REWRITE.txt) | Layer C: rules, matching, strategies, normal form, e-graph, tactics |
| [`RULES.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/RULES.txt) | Every rewrite rule with its scalar and side conditions |
| [`SEMANTICS.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/SEMANTICS.txt) | Layer D: denotation, contraction, the oracle |
| [`PROOFS.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/PROOFS.txt) | Deciding equality, proof search, induction, certificates, the kernel |
| [`LEAN.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/LEAN.txt) | The Lean 4 library and the export of proofs and refutations |
| [`ERRORS.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/ERRORS.txt) | The exception hierarchy of every module |
| [`GLOSSARY.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/GLOSSARY.txt) | Definitions of the terms used throughout |

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

[`docs/LEAN.txt`](https://github.com/Arhip-Dmitriev/ArchytasZX/blob/main/docs/LEAN.txt) has the
semantics, the trust boundary, the normaliser and the refutation method.

## Architecture

Four engine layers, depending strictly downward, with the shell on top:

| Layer | Package | Contents |
|---|---|---|
| A | [`algebra`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/algebra) | Dimension expressions with one canonical form and a unifier; phase vectors tied to `d`; exact scalars over roots of unity with a character-sum simplifier |
| B | [`diagram`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/diagram) | Qufinite ZXW generators (Z, X, Fourier box, triangle, W, dimension binder/splitter) with per-port dimensions; bang boxes and scalable sheet wires; validation |
| C | [`rewrite`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/rewrite) | Rule library, matcher and engine with provenance; normal form; diagram-level e-graph; tactics and bidirectional proof search |
| D | [`semantics`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/semantics) | Symbolic and numeric contraction; the `decide_equal` ladder; induction; certificates, proof kernel and Lean export |
| — | [`repl`](https://github.com/Arhip-Dmitriev/ArchytasZX/tree/main/archytaszx/repl) | Diagram parser and printer, ZX drawing, engine commands, shell |

The package and sub-package `__init__.py` files export nothing, so engine objects are imported
by full module path. Each module raises only its own
`<Area>Error` hierarchy: `<Area>GrammarError` for a wrong call and `<Area>DomainError` for a case
not implemented yet.

## Status

In **beta and under active development**: the API is unstable and changes between
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
  title   = {ArchytasZX: Size- and Dimension-Agnostic Engine for Quantum Algorithm Development Based on {ZX}-Calculus},
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

**ZX-calculus, W node, qufinite ZX/ZXW**
- Coecke & Duncan, *Interacting quantum observables: categorical algebra and diagrammatics*, NJP 13 (2011) —
  [arXiv:0906.4725](https://arxiv.org/abs/0906.4725)
- van de Wetering, *ZX-calculus for the working quantum computer scientist* —
  [arXiv:2012.13966](https://arxiv.org/abs/2012.13966)
- Hadzihasanovic, *A diagrammatic axiomatisation for qubit entanglement*, LICS 2015, the W node —
  [arXiv:1501.07082](https://arxiv.org/abs/1501.07082)
- Wang, *Qufinite ZX-calculus: a unified framework of qudit ZX-calculi*, the generator set,
  triangle, Fourier box and dimension connectives — [arXiv:2104.06429](https://arxiv.org/abs/2104.06429)
- Poór, Wang, Shaikh, Yeh, Yeung & Coecke, *Completeness for arbitrary finite dimensions of
  ZXW-calculus*, LICS 2023 — [arXiv:2302.12135](https://arxiv.org/abs/2302.12135)
- Wang, Poór & Shaikh, *Completeness of qufinite ZXW calculus* —
  [arXiv:2309.13014](https://arxiv.org/abs/2309.13014)

**Bang boxes and scalable notation**
- Dixon & Duncan, *Graphical reasoning in compact closed categories for quantum computation*,
  AMAI 56 (2009) — [doi:10.1007/s10472-009-9141-x](https://doi.org/10.1007/s10472-009-9141-x)
- Kissinger, Merry & Soloviev, *Pattern graph rewrite systems*, EPTCS 143 (2014) —
  [arXiv:1204.6695](https://arxiv.org/abs/1204.6695)
- Kissinger & Quick, *A first-order logic for string diagrams*, CALCO 2015, !-box induction —
  [arXiv:1505.00343](https://arxiv.org/abs/1505.00343)
- Carette, Horsman & Perdrix, *SZX-calculus: scalable graphical quantum reasoning*, MFCS 2019 —
  [arXiv:1905.00041](https://arxiv.org/abs/1905.00041)
- Carette & Lemonnier, *Large-scale quantum diagrammatic reasoning tools, !-boxes vs. scalable
  notations* — [arXiv:2204.11702](https://arxiv.org/abs/2204.11702)

**Exact symbolic sums and contraction**
- Berndt, Evans & Williams, *Gauss and Jacobi Sums* (1998), quadratic Gauss sums and their sign
- Landsberg, *Zur Theorie der Gaussschen Summen…*, J. reine angew. Math. 111 (1893),
  Landsberg–Schaar reciprocity — [doi:10.1515/crll.1893.111.234](https://doi.org/10.1515/crll.1893.111.234)
- Dechter, *Bucket elimination*, Artif. Intell. 113 (1999), contraction by variable elimination —
  [doi:10.1016/S0004-3702(99)00059-4](https://doi.org/10.1016/S0004-3702(99)00059-4)

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

**Algorithms in the examples**
- Deutsch & Jozsa, *Rapid solution of problems by quantum computation*, Proc. R. Soc. A 439 (1992) —
  [doi:10.1098/rspa.1992.0167](https://doi.org/10.1098/rspa.1992.0167)
- Fan, *A generalization of the Deutsch–Jozsa algorithm to multi-valued quantum logic*, ISMVL 2007 —
  [doi:10.1109/ISMVL.2007.3](https://doi.org/10.1109/ISMVL.2007.3)
- Bernstein & Vazirani, *Quantum complexity theory*, SIAM J. Comput. 26 (1997) —
  [doi:10.1137/S0097539796300921](https://doi.org/10.1137/S0097539796300921)
- Krishna, Makwana & Suresh, *A generalization of Bernstein–Vazirani algorithm to qudit systems* (2016) —
  [arXiv:1609.03185](https://arxiv.org/abs/1609.03185)

**Software**
- Kissinger & van de Wetering, *PyZX*, EPTCS 318 (2020) — [arXiv:1904.04735](https://arxiv.org/abs/1904.04735)
- *ZXLive* — [github.com/zxcalc/zxlive](https://github.com/zxcalc/zxlive)
- Harris et al., *Array programming with NumPy*, Nature 585 (2020) —
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- Meurer et al., *SymPy*, PeerJ Comput. Sci. 3 (2017) — [doi:10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103)

**Diagrammatic proof assistants and their formalisation**
- Kissinger & Zamdzhiev, *Quantomatic: a proof assistant for diagrammatic reasoning*, CADE-25
  (2015) — [arXiv:1503.01034](https://arxiv.org/abs/1503.01034)
- Kissinger, *Chyp: composing hypergraphs, proving theorems*, ACT 2023 —
  [github.com/akissinger/chyp](https://github.com/akissinger/chyp)
- Lehmann, Caldwell, Shah, Spencer & Rand, *VyZX: formal verification of a graphical quantum
  language* — [arXiv:2311.11571](https://arxiv.org/abs/2311.11571)
- Shah, Spencer, Zielinski, Caldwell, Lehmann & Rand, *ViCAR: visualizing categories with
  automated rewriting in Coq*, EPTCS 429 (2025) — [arXiv:2404.08163](https://arxiv.org/abs/2404.08163)
- Caldwell, Spencer, Kissinger & Rand, *TensorRocq: enabling diagrammatic reasoning in Rocq* (2026) —
  [arXiv:2604.17592](https://arxiv.org/abs/2604.17592)
- Poór, Booth, Carette, van de Wetering & Yeh, *The qupit stabiliser ZX-travaganza*, EPTCS 384
  (2023), with the DiZX library — [arXiv:2306.05204](https://arxiv.org/abs/2306.05204),
  [github.com/jvdwetering/dizx](https://github.com/jvdwetering/dizx)
- Sutcliffe & Kissinger, *Fast classical simulation of quantum circuits via parametric rewriting
  in the ZX-calculus* (ParamZX), EPTCS 426 (2025) — [arXiv:2403.06777](https://arxiv.org/abs/2403.06777)

**Verification of quantum programs and parameterized families**
- Hietala, Rand, Hung, Li & Hicks, *Proving quantum programs correct* (SQIR), ITP 2021 —
  [arXiv:2010.01240](https://arxiv.org/abs/2010.01240)
- Zhou, Barthe, Strub, Liu & Ying, *CoqQ: foundational verification of quantum programs*, POPL 2023 —
  [arXiv:2207.11350](https://arxiv.org/abs/2207.11350)
- Chareton, Bardin, Bobot, Perrelle & Valiron, *An automated deductive verification framework for
  circuit-building quantum programs* (QBricks), ESOP 2021 — [arXiv:2003.05841](https://arxiv.org/abs/2003.05841)
- Abdulla et al., *Parameterized verification of quantum circuits*, POPL 2026 —
  [arXiv:2511.19897](https://arxiv.org/abs/2511.19897)
- Ross & Wesley, *Cutoff theorems for the equivalence of parameterized quantum circuits*, MFCS 2025 —
  [arXiv:2506.20985](https://arxiv.org/abs/2506.20985)
- Miller-Bakewell, *Finite verification of infinite families of diagram equations*, EPTCS 318
  (2020) — [arXiv:1904.00706](https://arxiv.org/abs/1904.00706)

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
