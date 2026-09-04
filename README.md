# ArchytasZX

**A qufinite ZX-calculus engine and REPL for reasoning about quantum states and linear maps when both the number of subsystems and the dimension of each subsystem are symbolic.**

> **This is a work in progress.** The project is being built phase by phase against a
> written build plan, and large parts of the intended system are not implemented yet.
> The API is unstable, the rule library is deliberately tiny, and nothing here should be
> treated as a finished tool. See [Current state](#current-state) for what actually runs
> today.

## The problem

A linear map on `n` qudits of dimension `d` is a `d^n × d^n` matrix, and a state is a
vector of length `d^n`. When `n` or `d` is a symbol rather than a number, that array does
not exist — there is nothing for a tensor or matrix library to allocate. So the regime
this project cares about, symbolic qudit count *and* symbolic qudit dimension, is
precisely the regime where conventional numerical tooling has nothing to offer.

A concrete integer does not rescue that. `d^n` is astronomically large long before it is
symbolic — a 20-leg spider at `d = 30` is `30^20` entries — so the gap between typing `d`
and typing `30` is not the gap between symbolic and computable. This engine treats them
the same way, and for the same reason.

ZX-calculus with bang boxes does operate there. Its rewrite rules are equalities of
diagrams that preserve the denoted linear map, and they are *schematic*: stated once, they
hold for every leg count and every dimension. A single graph rewrite is therefore a proof
of an entire doubly indexed family of Dirac-level identities — one for every value of `n`
and every value of `d` at once. That is the capability this project exists to provide.

## The two symbolic axes

- **Symbolic qudit count, `n`.** Handled by bang boxes: a bang box marks a subgraph
  repeated an unspecified number of times, so one diagram denotes a whole family indexed
  by `n`. Bang boxes may nest, and several independent count symbols may coexist in a
  single diagram.
- **Symbolic qudit dimension, `d`.** Handled by carrying dimension as data on the diagram
  rather than as a fixed global integer. The target is the full *qufinite* setting, where
  one diagram may carry wires of different dimensions — so dimension is stored **per
  port**, never as one global parameter.

Symbolic input is permitted; **concrete input is the expected case**. A user who writes
`3` and `30` rather than `d` and `n` gets the same engine: the values are abstracted to
symbols on entry, recorded in the diagram's *parameter environment*, and substituted back
on output. There is one algebra, always symbolic, and a concrete-input derivation proves
the whole family as well as the user's own instance of it.

## Intended workflow

The round trip being optimised for is **Dirac in, ZX manipulation, Dirac out**:

1. Describe a state or map, typically in Dirac notation, possibly with free symbolic
   phases — e.g. `sum_{k=0}^{d-1} |k>^{⊗n}`, or the same thing with `d = 3` and `n = 30`
   written as literal numbers.
2. The engine represents it as a ZX diagram with the appropriate bang boxes and per-port
   dimensions. Any concrete dimension or count is **abstracted here** — to a fresh symbol
   whose value is recorded in the parameter environment — so everything downstream is
   symbolic algebra in `d` and `n` regardless of what was typed.
3. Rewrite rules are applied, by hand or by an automated strategy. This is pure graph
   surgery: pattern match, splice, merge, add phases, track the exact scalar. Nothing is
   contracted.
4. When a proof is wanted rather than a spot-check, symbolic-`n` equalities are discharged
   by induction, and equality of two diagrams is decided via normal form — returning a
   machine-checkable certificate.
5. The result is read back out as a diagram and, where it lands in a recognisable form, as
   Dirac notation.
6. Where concrete values were supplied, the recorded environment is substituted back in —
   into the finished diagram and into the closed symbolic form of its value. That
   substitution costs nothing in the size of the value, so it answers `d = 30` as readily
   as `d = 3`.
7. Optionally the oracle instantiates `n` and `d` to **small** concrete numbers, contracts
   numerically, and confirms the rewrite preserved the map exactly, scalar included. This
   is the verification oracle, not the route to a large concrete answer: a spider's
   denotation is `d^rank` entries, so it saturates within single digits of legs.

## Architecture

Four layers, with the dependency direction running strictly downward. The REPL depends on
the engine; the engine never depends on the REPL.

**A — Symbolic algebra substrate** (`algebra/`)
Dimension expressions (a concrete integer, a symbol such as `d`, or arithmetic such as
`d^n` or `d1·d2`) with normalisation, equality, a unifier that decides or constrains when
two dimension expressions must agree, and `abstract`/`substitute` as inverse directions —
a supplied integer becomes a fresh symbol, and the recorded binding brings it back. Phases as concrete values, root-of-unity
indices, or free symbolic parameters, carried in vectors whose length is tied to `d`.
Scalars built from roots of unity `ω_d = e^{2πi/d}` and free symbols, tracked exactly, with
a character-sum simplifier that knows `sum_{k=0}^{d-1} ω_d^{jk} = d·[j ≡ 0 mod d]`.

**B — Diagram data model** (`diagram/`)
Ports carrying their own dimension label; nodes carrying a generator type, ordered input
and output ports, and a symbolic phase slot; diagrams holding nodes, wires, ordered
boundary lists, and an exact scalar accumulator. Generators are a small fixed set, each
with a denotation defined once as a formula in `n` and `d` — Z and X spiders to start, with
Hadamard/Fourier, the triangle, the W generator, and the qufinite dimension connectives to
follow. Bang boxes annotate a scoped subgraph with a multiplicity symbol and support
instantiate, copy, kill, and merge. Scalable sheet-wire notation translates losslessly to
and from bang boxes.

**C — Rewrite engine** (`rewrite/`)
A rule bundles a left-hand pattern, a right-hand builder, side conditions, quantifiers over
counts and dimensions, and the exact scalar it introduces. The matcher finds occurrences
and checks every side condition — dimension side conditions included — before a rule may
fire. The engine applies a rule at a match, returns a new diagram, and records structured
provenance from which a certificate is emitted. Above that sit a normal-form driver that
decides diagram equality, an equality-saturation e-graph for non-destructive rewriting with
cost-based extraction, and a tactic and proof-search layer.

**D — Semantics oracle and proof** (`semantics/`)
Three rungs in order of preference: rewriting first; symbolic contraction with `d` kept
formal as the general fallback, and the path by which a supplied concrete value of any
size is evaluated; numeric contraction at small concrete instantiations as the
verification oracle of last resort. Alongside them sits the proof machinery — induction
over bang-box multiplicities for symbolic-`n` equalities, and replayable certificates.

**The REPL** (`repl/`)
A parser for a small input DSL (spiders, wires, symbolic phases, nested and multi-index
bang boxes, dimensions, and simple Dirac kets), a printer that renders diagrams textually
and back to Dirac where possible, and an interactive command loop.

## Current state

Implemented and under test:

- **Dimension algebra** — concrete integers, symbols, products and powers, normalised and
  compared through a single canonical form, with substitution and a placeholder unifier.
  Substitution currently runs one way only, symbol → integer; `abstract` and the parameter
  environment are specified but not yet built, so concrete input is at present carried
  through the algebra as a literal rather than abstracted on entry.
- **Phase and scalar algebra** — symbolic phases in vectors tied to the wire dimension,
  spider-fusion phase addition, and exact scalars (roots of unity, free symbols, products
  and sums) with no silent discarding of global factors.
- **Diagram data model** — ports, nodes, wires, ordered boundaries, exact scalar
  accumulator, deep copy and controlled mutation, and a generator registry for the Z and X
  spiders.
- **Validation** — per-port dimension agreement, boundary consistency, port usage, and
  generator policy. A node's legs must be *jointly* unifiable to one shared dimension, not
  merely pairwise unifiable against the first leg, and a phase vector tied to its node's
  leg dimension is checked against that same jointly-resolved value; both are therefore
  independent of leg order. Every node's dimension must be determinable at all, so a node
  with no legs and no phase is rejected, matching the numeric oracle's own refusal. A name
  may not serve as more than one of `qufzx.algebra`'s four symbol roles (a dimension, a
  dimension's exponent, a phase parameter, or a scalar) within one diagram, since
  substitution is name-keyed and would otherwise conflate them. Checks are local to each
  node; diagram-wide dimension-constraint propagation is deferred to a later phase. A pair
  of dimensions that agree only *under a binding* (a symbol against a concrete value, or
  against another symbol) is recorded as `DIMENSION_BOUND`, alongside the `DIMENSION_DEFERRED`
  case where the unifier could not decide at all. Both are assumptions rather than failures,
  so neither fails validation, and both reach the rewrite engine's before/after compare.
- **Numeric oracle** — generator denotations at concrete `d`, contraction of a fully
  concrete diagram into a tensor carrying the exact scalar, and an equality check that
  instantiates symbols, contracts both sides, and compares exactly, with an opt-in
  up-to-global-phase mode.
- **Rewrite core** — the rule/pattern/builder abstraction, a matcher for same-colour
  spider fusion, the fusion rule itself with its exact scalar, and an engine that applies a
  rule at a match and records step provenance. Seven side conditions are declared: five are
  decisions, and two (`distinct_nodes`, `parallel_wires_become_self_loops`) are structural
  facts recorded for the certificate that no candidate can fail. The builder re-derives
  every one fresh, from the same function the matcher uses, and rejects a match whose own
  claims disagree rather than quietly correcting them. Dimension assumptions the matcher
  could not verify as a syntactic identity are recorded as source-keyed
  `DimensionConstraint`s — one entry per connecting pair, surviving leg, or node phase,
  replaced in place across a fixpoint pass rather than re-appended, and adequate on their
  own: the finished record implies every equality any pass ever asserted, with no appeal to
  the resolver's binding accumulator. A rewrite's effect on pre-existing deferred
  assumptions is recorded in both directions, `removed_deferred_issues` and
  `introduced_deferred_issues`. Every field a builder hands back is checked before it is
  spliced or recorded — including that no surviving port is remapped onto a node the rewrite
  consumes — and the builder's own effect on the working diagram is checked against the
  pre-builder state: it adds the replacement node(s) and reports every other change through
  its result, so an edit to the wire set or to either boundary list is rejected rather than
  adopted as ground truth. Graph-to-fuse-to-graph is oracle-checked exactly, including at
  substitutions where a recorded constraint holds only by assumption.
- **Dirac parsing (Phase 5 slice only)** — one restricted form: a summed ket family
  `sum_{k=0}^{D-1} |k,k,...>` (or the `|k>^{n}` tensor-power shorthand), optionally
  followed by `; copy` to feed the state into a fixed copy spider. That is exactly the
  shape Phase 5's worked example needs, oracle-checked end to end (source → diagram →
  fusion match → post-diagram) at several concrete `d`. Every exception it raises is a
  `DiracError`, the bound summation index cannot be captured as a dimension symbol, every
  numeric token is ASCII by construction (identifier tokens stay Unicode-aware, since a
  symbol name has no numeric domain to be silently misread into), and a tensor-power leg
  count is bounded. A general spider/wire/bang-box declaration syntax, bang boxes,
  multi-index families, and the Dirac printer belong to Phases 18, 7, and 17.

Not yet implemented:

`abstract` and the parameter environment (concrete input abstracted on entry and
substituted back on output) · certificates and replayable derivations · bang boxes and
free `n` · induction over
multiplicities · symbolic contraction with `d` formal · the character-sum simplifier ·
mixed dimensions and the full qufinite generator set · the broader rule library and
strategy layer · match/denotation caching · the normal-form decision procedure · equality
saturation · tactics and proof search · scalable notation · a Dirac printer and the general
REPL declaration syntax/bang-box grammar.

## References (Highly Incomplete List)

Full texts of these are under [`references/`](references/).

- Wang, *Qufinite ZX-calculus: a unified framework of qudit ZX-calculi* —
  [arXiv:2104.06429](https://arxiv.org/abs/2104.06429)
- Kissinger et al., on bang boxes and scalable notation —
  [arXiv:2204.11702](https://arxiv.org/abs/2204.11702)
- van de Wetering, *ZX-calculus for the working quantum computer scientist* —
  [arXiv:2012.13966](https://arxiv.org/abs/2012.13966)

## License

Apache 2.0.

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
