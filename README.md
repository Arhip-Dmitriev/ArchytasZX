# ArchytasZX

**A qufinite ZX-calculus engine for reasoning and interacting with quantum states that
leverages symbolic algebra to remove qudit-count and dimensionality limits.**

[![PyPI](https://img.shields.io/pypi/v/archytaszx.svg)](https://pypi.org/project/archytaszx/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#project-status)

Example:

```python
prove_by_induction(boxed, fused, witness={"d": 2})
# proved=True  verdict=proved_uniform  index='m'  held_symbolic={'d'}
```

One call, one proof, an entire doubly indexed family of identities for every qudit count and
every qudit dimension at once.

---

## Why

*Coming soon.*

## Install

Python ≥ 3.11. `numpy` and `sympy` are the only runtime dependencies.

```bash
pip install archytaszx
```

To work on the engine itself, install from a clone instead:

```bash
git clone https://github.com/Arhip-Dmitriev/ArchytasZX.git
cd ArchytasZX
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest
```

The distribution is named `ArchytasZX` and installs from PyPI as `archytaszx` (the index
normalizes the two to the same name); the importable package is `archytaszx`. There is
**no REPL and no command-line entry point yet** — `archytaszx/repl/shell.py` is a skeleton
awaiting implementation. The engine can currently only be used as a Python library, and has
generally limited use.

### Imports

The sub-package `__init__.py` files are empty. There are **no re-exports**, so
`from archytaszx import Diagram` will fail — import by full module path. It is verbose during the
development phase:

```python
# Layer A -- symbolic algebra
from archytaszx.algebra.dimension import Dim, unify_all
from archytaszx.algebra.phase import Phase, PhaseVector
from archytaszx.algebra.scalar import Scalar

# Layer B -- diagram data model
from archytaszx.diagram.graph import Diagram, Direction, PortRef
from archytaszx.diagram.generators import Z_SPIDER, X_SPIDER, FOURIER_BOX
from archytaszx.diagram.bangbox import abstract_subgraph_count, abstract_port_count, peel_one
from archytaszx.diagram.validate import validate, validate_or_raise

# Layer C -- rewriting
from archytaszx.rewrite.match import find_matches
from archytaszx.rewrite.engine import apply
from archytaszx.rewrite.rules_library import SPIDER_FUSION, FOURIER_CANCELLATION, ZX_CAP

# Layer D -- semantics, proof, certificates
from archytaszx.semantics.check import compare, score, compare_symbolic
from archytaszx.semantics.contract_symbolic import contract_symbolic
from archytaszx.semantics.induction import prove_by_induction
from archytaszx.semantics.certificate import certify, verify

# The restricted Dirac front end
from archytaszx.repl.parser import parse_dirac_source
```

### Errors

Every module raises only its own hierarchy, under a base `<Area>Error`:

| Exception | Meaning |
|---|---|
| `<Area>GrammarError` | Wrong call |
| `<Area>DomainError` | Unimplemented as of now |

## Quickstart

### 1. Fuse two spiders over a symbolic dimension

Build a state-prep Z spider whose output feeds a copy spider, with the dimension left as the
symbol `d`, and fuse them.

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

`d` was never given a value. The exact scalar the rule introduces is tracked rather than
dropped, which is what makes the result an equality rather than a proportionality.

### 2. Check it, and get a replayable certificate

```python
from archytaszx.semantics.check import compare
from archytaszx.semantics.certificate import certify, verify

compare(g, result.diagram, {"d": 5}).matched   # True -- numeric oracle at d = 5

certificate = certify(g, [result], label="fuse A into B")
verify(certificate, {"d": 3}).verified         # True
```

`verify` **replays** the derivation from the recorded steps and oracle-checks the input
against the re-derived output. Tampering with any recorded field (the scalar, the rule name,
a consumed wire, a side-condition outcome, a multiplicity) fails the replay.

### 3. Prove it for every count at once

Wrap the same graph in a bang box with symbolic multiplicity `m`, and discharge the identity
for every value of `m` with `d` still symbolic.

```python
from archytaszx.diagram.bangbox import abstract_subgraph_count
from archytaszx.semantics.induction import prove_by_induction

boxed, box_id, mult = abstract_subgraph_count(g, frozenset({a, b}), 1, stem="m")
fused = apply(boxed, SPIDER_FUSION, find_matches(boxed)[0]).diagram

proof = prove_by_induction(boxed, fused, witness={"d": 2})

proof.proved          # True
proof.verdict         # Verdict.PROVED_UNIFORM
proof.index           # 'm': the multiplicity inducted on
proof.discharge       # StepDischarge.UNIFORM_REWRITE -- which tier settled it
proof.held_symbolic   # frozenset({'d'}) -- what stayed universally quantified
```

The verdict distinguishes a proof for all `n` from a finite schema check, and `discharge`
names the tier that did the work. `SCHEMA_CHECKED` is reported separately from the two
`PROVED_*` verdicts.

### 4. Dirac in, with concrete numbers

```python
from archytaszx.repl.parser import parse_dirac_source

p = parse_dirac_source("sum_{k=0}^{3-1} |k>^{30}; copy")

len(p.nodes)         # 2
len(p.bang_boxes)    # 1
dict(p.parameters)   # {'d': 3, 'n': 29}
```

The user typed `3` and `30`. What came back is symbolic in **both** axes, with the values
recorded in the parameter environment for substitution back on output. The tensor power costs
*one* leg under a bang box, not thirty, and stays open to induction. Prefix a numeral with
`literal` to suppress abstraction.

### The full tour

Every layer end to end — validation, matching, certificates, the numeric and symbolic
oracles, the Dirac front end, and a proof by induction:

```bash
python examples/api_tour.py
```

[`docs/TUTORIAL.md`](docs/TUTORIAL.md) is the matching API reference, section for section.
[`examples/demo_visual.py`](examples/demo_visual.py) is a wide-terminal walkthrough that
prints the full unfiltered trace next to the diagram pictures (working, but outdated).

## Architecture

Four layers, with the dependency direction running downward:
`algebra` ← `diagram` ← `rewrite` ← `semantics`.

**A — Symbolic algebra substrate** ([`archytaszx/algebra/`](archytaszx/algebra/))
Dimension expressions — a concrete integer, a symbol such as `d`, or arithmetic such as `d^n`
or `d1·d2` — normalised through one canonical form with a unifier that decides or constrains
when two must agree, and `abstract`/`substitute` as inverse directions. Phases as concrete
values, root-of-unity indices, or free symbolic parameters, carried in vectors whose length is
tied to `d`. Exact scalars built from roots of unity `ω_d = e^{2πi/d}` and free symbols, with
a character-sum simplifier that knows `Σ_{k=0}^{d-1} ω_d^{jk} = d·[j ≡ 0 mod d]`.

**B — Diagram data model** ([`archytaszx/diagram/`](archytaszx/diagram/))
Ports carrying their own dimension label; nodes carrying a generator type, ordered input and
output ports, and a symbolic phase slot; diagrams holding nodes, wires, ordered boundary
lists, a parameter environment, and an exact scalar accumulator. Bang boxes annotate a scoped
subgraph with a multiplicity symbol and support instantiate, copy, kill, merge and peel.

**C — Rewrite engine** ([`archytaszx/rewrite/`](archytaszx/rewrite/))
A rule bundles a left-hand pattern, a right-hand builder, side conditions, quantifiers over
counts and dimensions, and the exact scalar it introduces. The matcher finds occurrences and
checks every side condition before a rule may fire. The engine applies a rule at a match,
returns a new diagram, and records structured provenance from which a certificate is emitted.

**D — Semantics oracle and proof** ([`archytaszx/semantics/`](archytaszx/semantics/))
Three rungs in order of preference: rewriting first; symbolic contraction with `d` kept formal
as the general fallback, and the path by which a supplied concrete value of any size is
evaluated; numeric contraction at small concrete instantiations as the verification oracle of
last resort. Alongside them, induction over bang-box multiplicities and replayable
certificates.

Numeric contraction is the oracle. A spider's denotation is `d^rank` entries, so it saturates
within single digits of legs. Large concrete values are answered by substituting the parameter
environment into the closed symbolic form, which costs nothing in the size of the value.

## Project status

This is a **research prototype under active development**. The API is unstable, the rule
library is very limited, and nothing here should be treated as a useful, finished tool at this
moment.

### Working today

| Area | State |
|---|---|
| Dimension algebra | Integers, symbols, products, powers; one canonical form; `abstract`/`substitute` both directions |
| Phase and scalar algebra | Symbolic phase vectors tied to `d`; exact scalars with no silent global factors |
| Diagram model | Per-port dimensions, ordered boundaries, parameter environment, deep copy |
| Validation | Joint (leg-order independent) dimension resolution, boundary and port checks, symbol-role separation; typed finding kinds |
| Generators | Z spider, X spider, Fourier box |
| Rewrite core | Rule/pattern/builder abstraction, matcher, engine, structured provenance |
| Rule library | `spider_fusion`, `fourier_cancellation`, `zx_cap` |
| Numeric oracle | Denotation, contraction, exact comparison, opt-in up-to-global-phase mode |
| Symbolic contraction | Arbitrary diagram closed with `d` formal, through the character-sum simplifier |
| Induction | Base + step over a bang-box multiplicity, four-tier step ladder, five distinct verdicts |
| Certificates | Per-step provenance, independent replay, tamper detection |
| Dirac front end | One restricted grammar: a summed ket family |

### Not yet implemented

Mixed dimensions across a diagram and the full qufinite generator set (triangle, W, dimension
connectives) · the broader rule library and strategy layer · the normal-form decision
procedure · equality saturation and the e-graph · tactics and proof search · match and
denotation caching · scalable sheet-wire notation · the diagram-to-Dirac printer · the general
declaration syntax and the interactive REPL · diagram-wide dimension-constraint propagation.

### Known limits inside what is implemented

- The dimension unifier is still a placeholder: it decides simple cases, and reports
  `DEFERRED` otherwise. Deferred and bound dimension findings are recorded as
  assumptions, and both reach the certificate.
- Symbolic contraction handles a symbolic multiplicity only where the bang box is **closed off**
  from the rest of the diagram; a box meeting a wire or a boundary slot has a rank that varies
  with the count.
- `compare_symbolic` has **three** outcomes, not two — equal, unequal, and *indeterminate*
  (the residual still carries an undecided index sum). Read `.reason` whenever a comparison
  comes back unmatched.
- The induction step case settles a family whose peeled copies the scalar layer can close; the
  rest falls through to rewriting at symbolic `n` or to a finite schema check, which reports
  itself as one.

## Development

```bash
python -m pytest          # default tier: 1408 tests, ~45s
python -m pytest -m slow  # 28 multi-thousand-seed fuzz sweeps and oracle differentials, ~15m
python -m pytest -m ""    # both tiers together
ruff format . && ruff check . && mypy archytaszx
```

`ruff format` is authoritative for layout; `mypy` runs in strict mode. Counts and timings
measured 2026-09-08.

## References

- Wang, *Qufinite ZX-calculus: a unified framework of qudit ZX-calculi* —
  [arXiv:2104.06429](https://arxiv.org/abs/2104.06429)
- Kissinger et al., on bang boxes and scalable notation —
  [arXiv:2204.11702](https://arxiv.org/abs/2204.11702)
- van de Wetering, *ZX-calculus for the working quantum computer scientist* —
  [arXiv:2012.13966](https://arxiv.org/abs/2012.13966)

## Citation

If you use ArchytasZX in academic work, please cite it via [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{dmitriev_archytaszx,
  author  = {Dmitriev, Arkhip Alekseyevich},
  title   = {ArchytasZX: a qufinite ZX-calculus engine with symbolic qudit count and dimension},
  year    = {2026},
  url     = {https://github.com/Arhip-Dmitriev/ArchytasZX},
  license = {Apache-2.0}
}
```

## Contributing

Issues — bug reports, counterexamples, questions about the mathematics — are a vital part of
the development process. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, the required
checks, and how to report a soundness bug.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

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
