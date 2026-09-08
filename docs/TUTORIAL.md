# qufzx: a tutorial and API reference

A working guide to the engine as it exists today. Every code fragment here is drawn from
[`examples/api_tour.py`](../examples/api_tour.py), which runs end to end:

```
python examples/api_tour.py
```

**There is no REPL yet.** `qufzx/repl/shell.py`, `commands.py` and `printer.py` are
docstring skeletons awaiting Phase 17/18. The engine is used as a Python library. The one
live file in that package is `parser.py`, the restricted Dirac front end ([§6](#6-the-dirac-front-end)).

## 0. Setup and import conventions

Python ≥ 3.11; `numpy` and `sympy` are the only runtime dependencies.

```
pip install -e '.[dev]'
python -m pytest        # 1396 tests
```

**The sub-package `__init__.py` files are empty.** There are no re-exports, so
`from qufzx import Diagram` fails. Import by full module path throughout:

```python
from qufzx.algebra.dimension import Dim
from qufzx.diagram.graph import Diagram, Direction, PortRef
from qufzx.rewrite.engine import apply
```

Every module raises only its own exception hierarchy. Each has a base `<Area>Error` split
into `<Area>GrammarError` (a caller passed the wrong shape or type) and `<Area>DomainError`
(the value is well-formed but outside what this phase accepts). No foreign exception
escapes a module boundary — useful when you want to distinguish "I called it wrong" from
"the engine declines this input".

The four layers depend strictly downward: `algebra` ← `diagram` ← `rewrite` ← `semantics`.

---

## Layer A — `qufzx.algebra`

### `Dim` — dimension expressions

A dimension is a concrete integer, a symbol, or arithmetic over them. All three are the
same type, normalised through one canonical form, so equality is decidable by `==`.

```python
d = Dim.symbol("d")          # or Dim("d")
three = Dim.concrete(3)      # or Dim(3)
d ** 2                       # powers
d * Dim("e")                 # products, mixed dimensions
d.is_concrete                # False
d.free_symbols               # frozenset({'d'})
```

The two directions that matter for the concrete-input story:

```python
Dim("d").substitute({"d": 3})            # symbol -> integer
Dim(3).abstract(avoid=..., stem="d")     # integer -> (fresh symbol, binding)
```

`abstract` is what makes a user's literal `3` behave like `d`: it mints a fresh
non-colliding symbol and hands back the binding that recovers the value.

`unify(other)` reports one of three statuses — `SUCCESS`, `FAILURE`, `DEFERRED` —
plus any bindings and residual pairs. It is deliberately a placeholder; full
diagram-wide constraint propagation is a later phase.

### `Phase` and `PhaseVector`

A `Phase` is an angle in *turns* (not radians), normalised into `[0, 1)`:

```python
Phase.zero()
Phase.turns(sp.Rational(1, 4))
Phase.root_of_unity(1, d)     # 1/d of a turn
Phase.symbol("alpha")         # free symbolic phase
```

A qudit spider carries a whole *vector* of phases, one per level, and the vector's length is
tied to the wire dimension — so at symbolic `d` the length is unknown and entries are given
sparsely by index:

```python
PhaseVector(d, {1: Phase.symbol("alpha")})   # level 1 carries alpha, the rest zero
```

`PhaseVector.__add__` is spider-fusion phase addition. Validation checks the vector's `dim`
against the node's jointly-resolved leg dimension.

### `Scalar` — the exact scalar accumulator

Global factors are never silently dropped. `Scalar` tracks roots of unity, free symbols,
rational and Gaussian-rational values, dimension powers (including *rational* powers, so
`sqrt(d)` from the Fourier box is exact), and index sums:

```python
Scalar.one()
Scalar.omega(d, 1)                 # omega_d = exp(2*pi*i/d)
Scalar.from_dim(d)                 # d itself as a scalar
Scalar.dim_power(d, -1, 2)         # d^(-1/2)
Scalar.index_sum(d, lambda k: ...) # sum_{k=0}^{d-1} ...
```

`simplify()` runs the **character-sum simplifier**, which knows
`sum_{k=0}^{d-1} omega_d^{jk} = d·[j ≡ 0 mod d]`. This is the piece that closes index sums
without ever instantiating `d`. It is budgeted (`max_steps`) and raises `ScalarBudgetError`
rather than returning a partial result.

---

## Layer B — `qufzx.diagram`

### Ports, nodes, wires

Dimension lives **per port**, never as a global parameter — that is the qufinite
requirement, and it is why one diagram may carry wires of different dimensions.

A `PortRef(node_id, direction, index)` addresses one port. `Direction` is `INPUT` or
`OUTPUT`. A `Wire` joins two `PortRef`s.

### `Diagram`

```python
diagram = Diagram()
a_id = diagram.add_node(Z_SPIDER, input_dims=[], output_dims=[d, d])
b_id = diagram.add_node(Z_SPIDER, input_dims=[d], output_dims=[d, d])
diagram.add_wire(PortRef(a_id, Direction.OUTPUT, 0), PortRef(b_id, Direction.INPUT, 0))
diagram.set_boundary_outputs([...])   # the boundary is ORDERED
```

Read-only views: `.nodes`, `.wires`, `.boundary_inputs`, `.boundary_outputs`, `.scalar`,
`.parameters`, `.bang_boxes`. Mutators: `add_node`, `remove_node`, `set_phase`, `add_wire`,
`remove_wire`, `set_boundary_inputs/outputs`, `multiply_scalar`, plus the bang-box family.
`.copy()` deep-copies.

`.parameters` is the **parameter environment**: symbol name → the concrete value the user
supplied for it. It survives copying and rewriting, and it is what lets a diagram parsed
from concrete source be scored with no assignment passed at all.

### Generators

Three are registered today, in `REGISTRY`:

| Constant | Name | Legs | Phase |
|---|---|---|---|
| `Z_SPIDER` | `"Z"` | any | vector tied to leg dim |
| `X_SPIDER` | `"X"` | any | vector tied to leg dim |
| `FOURIER_BOX` | `"F"` | exactly 1 in, 1 out | none |

(The `generators.py` module docstring still says "only Z and X" — stale; `F` was added
with the Fourier rule.) The triangle, W, and the qufinite dimension connectives are not
implemented.

### Validation

```python
report = validate(diagram)
report.errors    # fatal
report.issues    # everything, including non-fatal findings
validate_or_raise(diagram)
```

Findings are typed by `IssueKind` (27 kinds). Two deserve attention because they are
**assumptions, not failures**, and both pass validation and reach the rewrite engine:

- `DIMENSION_BOUND` — the pair agrees *under a binding* (a symbol against a value, or
  against another symbol).
- `DIMENSION_DEFERRED` — the unifier could not decide at all.

These are independent, not alternatives: a leg set that ends up deferred may still have
bound a symbol on the way, and that binding is reported rather than dropped.

Checks worth knowing: a node's legs must be *jointly* unifiable to one shared dimension
(not merely pairwise against the first leg), so the verdict is leg-order independent; a
node whose dimension is undeterminable at all is rejected; and a name may not serve two of
the four symbol roles (dimension, exponent, phase parameter, scalar) in one diagram, since
substitution is name-keyed.

Dimension checks are **local to each node**. Diagram-wide constraint propagation is deferred.

### Bang boxes — `qufzx.diagram.bangbox`

A bang box marks a scoped subgraph repeated an unspecified number of times, so one diagram
denotes a family indexed by a count. Multiplicities are `Mult` (a non-negative integer or a
symbol). Boxes may nest and several independent count symbols may coexist.

Two scope flavours:

- **node scope** — the box repeats a set of nodes.
- **port scope** — the box repeats a set of boundary ports on one node.

The count is abstracted exactly as a dimension is:

```python
boxed, box_id, mult = abstract_subgraph_count(diagram, frozenset({a_id, b_id}), 1, stem="m")
boxed, box_id, mult = abstract_port_count(diagram, port_ref, 30, stem="n")
```

Operations: `instantiate_symbol(diagram, name, value)`, `peel_one`, `kill`, `copy_box`,
`merge`, `expand_concrete_boxes`, and the queries `free_mult_symbols` and `scope_is_closed`.

`peel_one` is the one the induction machinery leans on: it splits one copy off the front of
a box, turning multiplicity `k+1` into one explicit copy plus a box at `k`.

---

## Layer C — `qufzx.rewrite`

### The rule abstraction

A `Rule` bundles a left-hand `Pattern`, a right-hand builder, side conditions, quantifiers
over counts and dimensions, and the exact `Scalar` it introduces. Three rules exist, in
`RULES` (look one up by name with `lookup_rule`):

| Name | Constant |
|---|---|
| `spider_fusion` | `SPIDER_FUSION` |
| `fourier_cancellation` | `FOURIER_CANCELLATION` |
| `zx_cap` | `ZX_CAP` |

`zx_cap` is the rule whose exact scalar is a power of the dimension — the one that would
be quietly wrong in a tool that discards global factors.

### Matching and applying

```python
matches = find_matches(diagram)                 # fusion matches, deterministically ordered
result  = apply(diagram, SPIDER_FUSION, matches[0])
result.diagram                                   # the new diagram
result.step                                      # RewriteStep provenance
```

Match finders: `find_matches` (fusion), `find_fourier_matches`, `find_cap_matches`.

Spider fusion declares **eight side conditions**, each reported with a human-readable
detail string whether it passed or failed:

```
distinct_nodes · same_generator_type · parallel_wires_become_self_loops ·
consumed_wire_direction_permitted_for_color · consumed_ports_singly_claimed ·
bang_box_scope_agreement · dimension_agreement · phase_dimension_agreement
```

Two of them (`distinct_nodes`, `parallel_wires_become_self_loops`) are structural facts no
candidate can fail; they are recorded for the certificate rather than used as gates.

The builder **re-derives every side condition fresh**, from the same function the matcher
used, and rejects a match whose own claims disagree rather than quietly correcting them.
The engine also checks the builder's own effect against the pre-builder state: a builder
adds its replacement node(s) and reports every other change through its result, so an edit
to the wire set or a boundary list is rejected rather than adopted as ground truth.

### `RewriteStep` — the provenance record

```python
step.rule_name              step.consumed_node_ids     step.consumed_wires
step.side_condition_outcomes                           step.dimension_constraints
step.scalar_introduced      step.port_mapping          step.new_node_ids
step.removed_deferred_issues  step.introduced_deferred_issues
```

`dimension_constraints` is every equality the rewrite **assumed** rather than verified as a
syntactic identity, source-keyed, one entry per connecting pair / surviving leg / node
phase. The record is self-contained: it implies every equality any pass asserted, with no
appeal to the resolver's internal binding accumulator.

Read a `DEFERRED` constraint as an assumption, not a satisfiability claim. A surviving leg
of `d**2` forced onto a shared `d` records `d**2 == d`, which is true only at `d = 1`.
Discharging such constraints is a later phase.

The two `*_deferred_issues` fields record the rewrite's effect on pre-existing assumptions
in both directions, as a multiset difference. Neither is a gate; both are certificate facts.

`normal_form.py`, `egraph.py`, `tactics.py` and `cache.py` are **empty skeletons**. The
normal-form decision procedure, equality saturation, tactics and caching do not exist.

---

## Layer D — `qufzx.semantics`

Three rungs, in order of preference: rewrite first, then symbolic contraction with `d`
formal, then numeric contraction at small concrete values as the oracle of last resort.

### Numeric oracle

```python
instantiate(diagram, {"d": 3})           # substitute + expand concrete bang boxes
contract(diagram)                        # -> ContractionResult (fully concrete only)
score(diagram, {"d": 3})                 # instantiate then contract
compare(before, after, {"d": 3})         # -> ComparisonResult
```

`compare` defaults to `EqualityMode.EXACT`; `UP_TO_GLOBAL_PHASE` is opt-in only, by design.
Before tensors are compared the two interfaces are checked to correspond (same
input/output split, same per-axis dimensions); a mismatch is reported as its own non-match
result rather than surfacing as a numeric deviation.

A symbol you do not supply falls back to its parameter-environment binding, and a supplied
value overrides that binding — so a diagram parsed from concrete source scores with no
assignment, or can be spot-checked elsewhere.

Contraction is capped at `DEFAULT_MAX_ELEMENTS = 10_000_000` and raises `ContractSizeError`
past it. **This is the verification oracle, not a route to large concrete answers**: a
spider's denotation is `d^rank` entries and saturates within single digits of legs.

### Symbolic contraction

```python
tensor = contract_symbolic(diagram)   # -> SymbolicTensor
tensor.entry                          # one closed Scalar entry, d formal
compare_symbolic(left, right)
```

Two limits to state rather than let a reader discover:

1. **`compare_symbolic` has three outcomes, not two.** Equal, definitely unequal, and
   *indeterminate* — the difference still carries an index sum whose character-sum verdict
   was undecidable. An indeterminate result reports `matched=False` with a reason naming the
   residual sum, so treating "not matched" as "unequal" is a bug. Read `.reason`.
2. **Only closed node-scope bang boxes contract symbolically.** A box meeting the rest of
   the diagram at a wire or a boundary slot has a rank that varies with the count, and
   raises `SymbolicContractionUnsupportedError`. A concrete multiplicity is expanded
   outright; a closed symbolic one contributes its own scalar raised to that multiplicity.

`compare_symbolic` also refuses `UP_TO_GLOBAL_PHASE` — that needs concrete entries.

### Induction over a multiplicity

The headline capability: one proof discharging an entire family.

```python
result = prove_by_induction(left, right, witness={"d": 2})
result.proved          # True
result.verdict         # Verdict.PROVED_UNIFORM
result.index           # 'm' -- the multiplicity it inducted on
result.discharge       # StepDischarge.UNIFORM_REWRITE -- the tier that settled it
result.held_symbolic   # frozenset({'d'}) -- what stayed universally quantified
result.counterexample  # None
```

`Verdict` has five values, and the distinctions are the point:

| Verdict | Meaning |
|---|---|
| `PROVED_UNIFORM` | one rewrite works at every value of the index |
| `PROVED_INDUCTION` | genuine base + step, using the induction hypothesis |
| `SCHEMA_CHECKED` | the shapes line up, but this is **not** a proof |
| `REFUTED` | the base case or a window instance fails; see `.counterexample` |
| `INCONCLUSIVE` | no tier settled it |

Only the first two make `.proved` true. `SCHEMA_CHECKED` is reported separately precisely
so it cannot be mistaken for a discharge.

The step case is attempted through a **ladder** of tiers, in this default order, stopping at
the first that settles:

1. `UNIFORM_REWRITE` — the same rewrite fires at every index value.
2. `INDUCTION_REWRITE` — peel one copy, then use the hypothesis as a rewrite rule.
3. `SYMBOLIC_CONTRACTION` — close both sides with `d` formal.
4. `ORACLE_WINDOW` — check a window of consecutive concrete values (width 4 by default).

Note the ordering: the oracle window is **last**, and settling there is the weakest outcome
since it is sampling over a finite window. `result.discharge` tells you which tier actually
did the work — worth reporting alongside any claimed result.

The `witness` binds symbols the base case needs concretely (here `d = 2`); what the settling
tier left universally quantified comes back in `.held_symbolic`.

`InductionObligation` exposes the six diagrams the proof ran against (`left_at_base`,
`right_at_base`, `left_at_k`, `right_at_k`, `left_at_successor`, `right_at_successor`),
which is the natural thing to display in a paper.

### Certificates

```python
certificate = certify(initial, [result1, result2], label="fuse A into B")
report = verify(certificate, {"d": 3})
report.verified                # bool
```

`certify_induction` builds the induction-flavoured certificate instead, carrying an
`InductionClaim` plus base and step derivations. `CheckMethod` names how the claim is
discharged (`NUMERIC_ORACLE`, `INDUCTION`, …).

`verify` **replays** the derivation from the recorded steps and then oracle-checks the
initial against the *re-derived* final. With `rediscover=True` (the default) it re-runs the
matcher rather than trusting the stored match, so a certificate that only reproduces
because it memorised its answer will fail. `replay()` alone gives you the replay without
the oracle check, and `compare_structure` gives a structural diff of two diagrams.

---

## 6. The Dirac front end

One restricted grammar, `qufzx.repl.parser.parse_dirac_source`:

```
sum_{k=0}^{D-1} |k,k,...>            # a summed ket family
sum_{k=0}^{D-1} |k>^{n}              # tensor-power shorthand
sum_{k=0}^{D-1} |k,k>; copy          # ...fed into a fixed copy spider
```

- `D` is a positive integer literal or a bare identifier (a symbolic `Dim`); `n` must be concrete.
- The bound summation index cannot be captured as a dimension symbol.
- **A concrete dimension is abstracted on entry**: the numeral becomes a fresh symbol and
  the parameter environment records its value. Prefix it with `literal` to suppress this.
- **A tensor power is abstracted into a port-scope bang box**, so `^{30}` costs *one leg*,
  not thirty, and stays open to induction:

```python
parse_dirac_source("sum_{k=0}^{3-1} |k>^{30}; copy")
# 2 nodes, 1 bang box, parameters {'d': 3, 'n': 29}
```

That line is the concrete-input claim in miniature: the user typed `3` and `30`, and what
came back is symbolic in both axes with the values recorded for substitution back.

Everything raises `DiracError` or a subclass. A general spider/wire/bang-box declaration
syntax, multi-index families, and the diagram-to-Dirac printer are later phases.

---

## What does not exist yet

Do not describe these as available: the interactive REPL and command loop · the Dirac
*printer* (diagram → Dirac) · the general declaration syntax · the normal-form decision
procedure · equality saturation and the e-graph · tactics and proof search · match and
denotation caching · scalable sheet-wire notation · mixed dimensions across a diagram and
the full qufinite generator set (triangle, W, dimension connectives) · the broader rule
library and strategy layer · diagram-wide dimension-constraint propagation.

Note also that the README's **"Not yet implemented"** list is stale: it still disclaims
certificates, bang boxes, free `n`, induction, symbolic contraction and the character-sum
simplifier, all of which are implemented and tested.

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
