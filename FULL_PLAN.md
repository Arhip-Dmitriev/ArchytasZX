===============================================================
BUILD PLAN

PART 0: PROJECT SETUP

Phase 0.1: Repository and environment
i. pyproject.toml
- declare the package qufzx, Python version, and dependencies numpy and sympy
- declare dev dependencies pytest, ruff, mypy
- make the package installable in editable mode
ii. .gitignore, README.md, LICENSE (apache 2.0)
- ignore venv, caches, build artifacts
- README states the one-line purpose and how to run tests
iii. qufzx/__init__.py and tests/__init__.py
- import qufzx succeeds from a clean environment
- test and debug: run pytest on an empty suite and confirm it collects zero tests
- done when: editable install works, ruff and mypy run clean on empty package, git is initialized

Phase 0.2: Empty module skeleton
i. create the full tree as empty stubs with docstrings only:
qufzx/algebra/dimension.py, phase.py, scalar.py
qufzx/diagram/graph.py, generators.py, bangbox.py, scalable.py, validate.py
qufzx/rewrite/rule.py, match.py, rules_library.py, engine.py, cache.py, normal_form.py, egraph.py, tactics.py
qufzx/semantics/denote.py, contract_numeric.py, contract_symbolic.py, induction.py, certificate.py, check.py
qufzx/repl/parser.py, printer.py, commands.py, shell.py
- every file has a top docstring stating its job
- test and debug: import every module; confirm no import errors and no circular imports
- done when: the tree matches the plan and the whole package imports cleanly

PART 1: THE ENGINE, CORE REPRESENTATION AND ORACLE

Phase 1: Dimension algebra, minimal
i. algebra/dimension.py
- defines a Dim type covering a concrete integer, a symbol d, and product and power expressions like d^n or d1*d2
- normalizes and compares dim expressions so that equal dimensions test equal
- exposes is_concrete, substitute (symbol to integer), and a placeholder unify that returns constraints
- test and debug: unit tests for equality, substitution, and product normalization
- done when: d, 2, and d^n can be represented and compared symbolically

Phase 2: Phase and scalar algebra, exact
i. algebra/phase.py
- a Phase can be a concrete value, a root-of-unity index, or a free symbolic parameter
- phases live in vectors whose length is tied to the wire dimension d, symbolic length permitted
- phase addition (spider fusion semantics) is defined and normalizes
ii. algebra/scalar.py, representation only
- a Scalar represents roots of unity w_d = e^(2*pi*i/d), free symbolic scalars, and products and sums of these
- defer the character-sum simplifier to Phase 9!!!!
- test and debug: unit tests for symbolic phase addition, phase-vector length tied to d, and exact scalar equality without any global-factor quotient
- done when: phases and scalars are first-class symbolic objects and nothing silently discards a global factor

Phase 3: Diagram data model, no bang box and connectives
i. diagram/graph.py
- a Port carries its own dimension label, per port
- a Node carries a generator type, ordered input ports, ordered output ports, and a phase-data slot that accepts symbolic phases from Phase 2
- a Diagram holds nodes, wires (each joining two ports), ordered boundary input and output lists, and an exact scalar accumulator
- there is a deep copy and controlled mutation
ii. diagram/generators.py
- registers generator types, starting with Zspider and Xspider
- each type records its leg policy (any number of legs), its phase schema (vector length tied to d, symbolic entries allowed), and its dimension policy (for Z and X, all legs share one dimension)
iii. diagram/validate.py
- rejects any wire whose two ports carry unequal dimensions
- checks boundary consistency
- test and debug: build a two-leg Z spider, build the A-into-B graph from the worked example, confirm validation passes, then confirm a deliberately mismatched wire fails
- done when: the GHZ-with-copy graph, carrying a symbolic phase on at least one node, can be constructed and validated

Phase 4: Numeric oracle (rung 3) with exact scalar tracking
i. semantics/denote.py
- returns the denotation of a Z spider m to n at a concrete d as a numpy tensor, and likewise for X, honoring any concrete or instantiated phase
- the Z spider with zero phase gives sum_{k=0}^{d-1} |k>^{(x)n} <k|^{(x)m}

ii. semantics/contract_numeric.py [testing]
- contracts a fully concrete diagram (all dimensions concrete, no bang box) into a numpy tensor by contracting along wires, and carries the exact scalar accumulator through
iii. semantics/check.py
- given two diagrams claimed equal, instantiates all symbols to supplied concrete values, contracts both, and compares exactly including the overall scalar, up to floating-point tolerance
- an explicit up-to-global-phase mode exists as an OPT IN!
- test and debug: confirm the Z spider 0 to 2 at d equal to 2 gives the vector for |00> + |11>; confirm A-into-B contracts to the GHZ vector for d equal to 2 and 3; confirm that a deliberately scalar-shifted copy fails exact comparison but passes in up-to-global-phase mode
- done when: the oracle can score any concrete diagram and compare any two diagrams, exactly by default

Phase 5: Rewrite core and spider fusion [CURRENT]
i. rewrite/rule.py
- a Rule bundles a left-hand pattern, a right-hand builder, side conditions, quantifiers over n and over dimensions, and the exact scalar it introduces
ii. rewrite/match.py
- finds occurrences of the fusion pattern ONLY, that is two same-color spiders joined by a wire and sharing a dimension
iii. rewrite/rules_library.py
- spider_fusion merges the two spiders, unions their legs minus the consumed wire, adds their phases, and records any scalar factor it introduces
iv. rewrite/engine.py
- applies a chosen rule at a found match, returns a new diagram, and records structured provenance of the step in a form a certificate can consume
v. repl/parser.py, Dirac slice ONLY
- parses one restricted form, the summed ket family sum_{k=0}^{D-1} |k,k,...> with the |k>^{n} tensor-power shorthand, optionally followed by "; copy", which is the Dirac-to-graph end from the done-when clause below
- this grammar must remain a strict subset of the one in Phase 18
- test and debug: fuse A-into-B into a single spider; oracle-check that the pre and post diagrams are exactly equal at several concrete d; validate that the post diagram is well formed
- done when: the full path Dirac to graph to fuse to graph runs and the oracle confirms exact equality

Phase 6: Proof certificates
i. semantics/certificate.py
- every rewrite step emits a machine-checkable record: the rule fired, the match location, the side conditions checked, the dimension constraints assumed, and the scalar introduced
- a full derivation is a sequence of such records that can be independently replayed and verified against the oracle
- test and debug: emit a certificate for the Phase 5 fusion, replay it on a fresh copy of the input, and confirm the replay reproduces the output and passes the oracle
- done when: any rewrite sequence carries evidence sufficient to re-derive and re-check it

PART 2: FREE n AND FREE d ---------

Phase 7: Bang boxes, free n, with nesting and multiple indices
i. diagram/bangbox.py
- a bang box records a scope over a subgraph or set of ports, a multiplicity symbol, and its overlap and boundary edges
- supports instantiate (a multiplicity to a concrete k, expanding to k copies), copy, kill (multiplicity 0), and merge
- bang boxes may nest, and several independent count symbols may coexist in one diagram, with an arithmetic of multiplicities (sums, products, and substitution)
ii. extend diagram/validate.py
- checks that bang-box scopes, nesting, and boundary edges are well formed, and that distinct count symbols are tracked without collision
iii. extend semantics/check.py
- instantiates every count symbol as well as every dimension symbol before contracting
iv. extend rewrite/match.py
- fusion can fire on a bang-boxed spider, at minimum matching with the box left intact
- test and debug: confirm the bang-boxed GHZ instantiates to the correct concrete GHZ for k equal to 0, 1, 2, 3; confirm a nested two-index family instantiates correctly on both indices; confirm fusion under a bang box preserves exact oracle equality across several tuples of counts and d
- done when: free n works end to end, including nesting and multiple indices

Phase 8: Symbolic verification over n by induction
i. semantics/induction.py
- an equality carrying a bang box can be discharged for all values of its multiplicity by induction: a base case at multiplicity 0 or 1 and a step case relating multiplicity k to k+1, each discharged by the oracle or by symbolic contraction
- multi-index families induct on one index at a time with the others held symbolic
- test and debug: prove the fusion GHZ identity for all n by induction rather than by sampling, and confirm the induction fails cleanly on false near-identity
- done when: symbolic-n equalities can be proved, not merely spot-checked

Phase 9: Character-sum simplifier and full symbolic contraction (rung 2)
i. complete algebra/scalar.py
- the character-sum simplifier knows sum_{k=0}^{d-1} w_d^(jk) = d*[j = 0 mod d] and the standard consequences, all with d symbolic
ii. semantics/contract_symbolic.py
- contracts an arbitrary diagram while d stays symbolic, producing a closed symbolic tensor expression simplified through the scalar layer
- is the general fallback below rewriting whenever a value is needed with d formal
iii. extend denote.py and rules_library.py
- a Hadamard or Fourier generator is added, and one rule that needs a root-of-unity scalar (color change or bialgebra) is added with its exact scalar output
- test and debug: unit-test the character-sum identity; verify the new rule two independent ways, once by full symbolic contraction in d and once by the numeric oracle at concrete d; contract a small nontrivial diagram symbolically and confirm it matches numeric instantiation at several d
- done when: the symbolic contractor evaluates arbitrary diagrams with d formal, and at least one root-of-unity rule is verified symbolically in d

Phase 10: Mixed dimensions and the full generator set
i. algebra/dimension.py
- the placeholder unify becomes a real dimension checker and unifier over expressions with constraints such as d = d1*d2

ii. diagram/generators.py
- adds the triangle node, the W generator, and the qufinite dimension-connective generators as defined by Wang
iii. diagram/validate.py
- per-port dimensions are enforced across good mixed-dimension diagrams
iv. rewrite/rule.py and match.py
- rules carry dimension side conditions and the matcher checks them before firing
- test and debug: a diagram mixing two dimensions validates; a rule restricted to prime d refuses to fire at composite d; a connective relating d1 and d2 to d1*d2 checks out on the oracle at small concretes
- done when: heterogeneous-dimension diagrams are first class and dimension-guarded rules behave

PART 3: REWRITING POWER

Phase 11: Rule library breadth and strategies
i. rewrite/rules_library.py
- adds identity removal, copy and Hopf, bialgebra, and the qufinite normal-form-directed rules, each recording its exact scalar
ii. rewrite/engine.py
- adds a strategy layer with apply-until-fixpoint and toward-normal-form, plus a termination guard against nonterminating loops
- test and debug: oracle-check every new rule at several tuples of counts and d; drive a known diagram to its expected form; confirm the termination guard trips on a deliberately looping strategy
- done when: the rule set is a usable rewriting toolkit

Phase 12: Rewrite caching and incrementality
i. rewrite/cache.py
- matches and denotations are memoized
- a local edit triggers incremental re-matching over the affected region rather than a full rescan
- test and debug: confirm cached and uncached runs produce identical results and certificates; confirm an incremental re-match after a small edit matches a full re-match; measure that repeated matching on a stable region hits the cache
- done when: interactive sessions stay responsive as diagrams grow, with no change in results and caching/memoization is implemented.

-- at this point do a full optimization sweep!

Phase 13: Normal-form driver and decision procedure
i. rewrite/normal_form.py
- a driver reduces a diagram to the qufinite normal form
- equality of two diagrams is decided by reducing both and comparing normal forms
- test and debug: reduce several diagrams known to be equal and confirm identical normal forms; reduce two known-unequal diagrams and confirm distinct normal forms; cross-check every decision against the oracle
- done when: the engine can decide diagram equality

Phase 14: Equality saturation
i. rewrite/egraph.py
- an e-graph applies rules non-destructively, growing a congruence closure of equal diagrams
- an extraction step selects an optimal representative under a stated cost
- test and debug: saturate a diagram where greedy rewriting gets stuck and confirm extraction finds the shorter form; confirm every equivalence class member is oracle-equal to the input
- done when: saturation reaches simplifications a greedy strategy cannot

Phase 15: Proof search and tactics
i. rewrite/tactics.py
- a tactic language composes rules into larger moves
- a search finds a derivation between two stated diagrams and reports the path as a certificate
- test and debug: ask the searcher to prove a target identity from a start diagram and confirm the returned path replays and verifies; confirm it reports failure cleanly when no derivation is found within bounds
- done when: the engine can be asked to prove a goal and returns a checkable derivation

Phase 16: Scalable notation interoperability
i. diagram/scalable.py
- the scalable sheet-wire notation is representable
- translation runs both ways between bang boxes and scalable notation (arXiv:2204.11702)
- a rewrite that is awkward in one notation may be performed after translating to the other and translating back
- test and debug: round-trip several families through both notations and confirm the diagram and its denotation are preserved; confirm a rule easier in scalable form yields the same result as the bang-box path
- done when: users may work in whichever notation suits a construction, losslessly

PART 4: THE REPL

Phase 17: Printer and Dirac output
i. repl/printer.py
- pretty-prints a graph textually, showing nodes, symbolic phases, exact scalars, port dimensions, and bang boxes including nesting and multiple indices
- renders every diagram back to Dirac: by its recognized name when the diagram is in a known normal form, otherwise as the structural index-sum form, one bound index per spider, valid for symbolic n and d and requiring no contraction
- implements the session notation mode of the Notation modes section: Dirac mode emits no graph output, ZX mode emits both
- test and debug: confirm the printer round-trips the fusion example and that the Dirac output of a single Z spider matches the expected sum; confirm every state of a multi-step derivation has a Dirac rendering in Dirac mode
- done when: any engine state can be shown as text, always as Dirac and always as a graph

Phase 18: Parser and input DSL
i. repl/parser.py
- the file already exists, carrying Phase 5's Dirac slice (see Phase 5, item v); this phase widens it, and the existing ket-sum grammar and its "copy" keyword must survive as a strict subset of the DSL below
- a small DSL can declare spiders, wires, symbolic phases, bang boxes (nested and multi-index), and dimensions
- the parser is mode-independent: Dirac expressions, the spider and wire DSL, and step-level rule application are all accepted in every session mode
- test and debug: parse the GHZ-with-copy input and confirm it yields the same graph the Phase 3 tests build by hand
- done when: text input produces valid diagrams, and every input form is accepted regardless of the session notation mode

Phase 19: Commands and shell
i. repl/commands.py
- implements load, show, list-rules, match, apply (rule, optionally at a chosen match), check (with supplied counts and d), normalize, decide (equality via normal form), saturate (equality saturation), prove (tactic search), induct (symbolic-n proof), certificate (emit and verify), and translate (bang box to and from scalable)
ii. repl/shell.py
- wires commands to the engine in an interactive loop with history and clean error handling
- holds the session notation mode, Dirac by default, with a command to set it; the mode selects output rendering only and gates nothing
- errors, proofs, derivations, certificates, and engine detail are available, but not visible by default, in full in both modes, with no flag
- detail payloads, that is error engine-detail, proof traces, derivation steps, and certificate contents, are shown in both notations side by side in either mode
- failure reports state the Dirac-level outcome, an oracle counterexample at concrete n and d where one exists, otherwise the furthest-simplified state and the bound that was hit
- test and debug: a scripted session builds the example, applies fusion, checks it, proves it for all n by induction, and emits and re-verifies a certificate, asserting the final state; malformed input yields a clean error; confirm the same scripted session runs identically in both notation modes and that a ZX command issued in Dirac mode returns Dirac output
- done when: the entire worked example, including a symbolic-n proof and a certificate, runs interactively

PART 5: HARDENING AND FINALIZATION

Phase 20: Test breadth and continuous integration
i. tests/
- property-based tests generate random small diagrams and confirm that every rule preserves the oracle at randomized concrete counts and d, exactly
- certificates for random derivations replay and verify
ii. CI configuration
- pytest, ruff, and mypy run on every push
- done when: the suite is broad and CI is green

Phase 21: Performance pass, only if profiling demands it
i. profile match.py and contract_numeric.py
- the matcher and the numeric kernel sit behind narrow interfaces so a later port touches only them
- done when: performance is acceptable at the working diagram sizes

Phase 22: Documentation and examples
i. README and an examples script
- there is a quickstart and a runnable script reproducing the Dirac to ZX to rewrite to Dirac example, including a symbolic-n proof and a certificate
- there is a rule reference listing each rule, its side conditions, and its exact scalar
- done when: a new reader can reproduce the example unaided
- actually we want several examples

===============================================================
CONTEXT

PROJECT CONTEXT: Qufinite ZX Engine and REPL

This document states what the system is, why it is built this way, and the invariants that must never be violated. Where a design decision is ambiguous, "Non-negotiable design rules" is the tie-breaker.

What we are building

We are building an engine and an interactive REPL for symbolically representing and manipulating quantum states and linear maps, where both the number of subsystems and the dimension of each subsystem may be symbolic rather than fixed numbers.

The mathematical substrate is the qufinite ZX-calculus with bang boxes. Concretely, the system represents a state or map as a ZX diagram, which is a typed graph, and manipulates it by graph rewriting according to the ZX rewrite rules. The user thinks in Dirac notation; the engine ingests or constructs the corresponding diagram, rewrites it, and reports results, ideally back in Dirac notation.

The two symbolic axes:

Symbolic qudit count, written n. Handled by bang boxes. A bang box marks a subgraph that is repeated an unspecified number of times, so a single diagram denotes a whole family indexed by n. Bang boxes may nest and several independent count symbols may coexist in one diagram.
Symbolic qudit dimension, written d. Handled by carrying dimension as data on the diagram rather than as a fixed integer. We are targeting the full qufinite setting, meaning a single diagram may contain wires of different dimensions, so dimension is stored per port, not as one global constant.

The deliverable has two parts: an Engine (data model, rewrite system, semantics oracle, proof machinery) and a REPL (parser, printer, interactive command loop) that sits on top of it for algorithm development.

Why we are building it this way

At symbolic n or symbolic d there is no matrix to compute with. A map on n qudits of dimension d is a d^n x d^n matrix and a state is a vector of length d^n; when n or d is a symbol, that size is a symbol, so the array does not exist.

ZX rewrite rules are equalities of diagrams that preserve the denoted linear map, and they are schematic: stated once, they hold for every leg count and every dimension. One graph rewrite is therefore a proof of a doubly indexed family of Dirac-level identities, one for every value of n and d.

The diagram, not the matrix, is the primary object. Matrices appear only as a verification oracle at concrete instantiations, never as the working representation. The general symbolic fallback below rewriting is a full symbolic contractor that keeps d formal, not a matrix.

The intended user workflow

The round trip we optimize for is: Dirac in, ZX manipulation, Dirac out.

The user describes a state or map, often in Dirac notation, for example sum_{k=0}^{d-1} |k>^{(x)n}, possibly with free symbolic phases.
The engine represents it as a ZX diagram with the appropriate bang boxes and per-port dimensions.
The user, or an automated strategy, applies rewrite rules. The engine performs pure graph manipulation: pattern match, splice, merge, adjust phases, track the exact scalar. It does not contract anything.
When a genuine proof is wanted rather than a spot-check, the engine discharges symbolic-n equalities by induction and can decide equality of two diagrams via normal form, returning a machine-checkable certificate.
The resulting diagram is read back out, as a diagram and as Dirac notation: by its recognized name when it lands in a known form, otherwise as the structural index-sum form.
Optionally, the user asks for a sanity check. The engine instantiates n and d to small concrete numbers, contracts numerically, and confirms the rewrite preserved the map exactly.

Notation modes

The REPL has one session-level notation mode, Dirac or ZX, defaulting to Dirac. It selects output rendering only.

Input is universal. Every command is accepted in every mode: Dirac expressions, the spider and wire DSL, and step-level rule application. No command, rule, or capability is gated by the mode.

Engine behaviour is identical in both modes. The same commands are accepted, the same results computed, the same state reached.

Every reachable engine state must have both renderings. Dirac mode never emits graph output; ZX mode never withholds a Dirac form. A state with no recognized Dirac name renders as its structural index-sum form.

Errors, proofs, derivations, certificates, and engine detail are available in full in both modes, without a flag. The mode is a rendering choice, not an access boundary.

Detail payloads are shown in both notations side by side: error engine-detail, proof traces, derivation steps, and certificate contents.

Architecture

Four layers. The dependency direction runs downward; the REPL depends on the engine, the engine does not depend on the REPL.

Layer A, symbolic algebra substrate.

Dimension expressions: a concrete integer, a symbol such as d, or arithmetic such as d^n or a product d1*d2, with normalization, equality, and a checker and unifier that can decide or constrain when two dimension expressions must be equal.
Phases: concrete values, root-of-unity indices, or free symbolic parameters, carried as vectors whose length is tied to d.
Scalars: roots of unity w_d = e^(2*pi*i/d) and free symbolic scalars, tracked exactly, with a character-sum simplifier that knows sum_{k=0}^{d-1} w_d^(jk) = d*[j = 0 mod d].

Layer B, diagram data model.

Ports carry their own dimension label. Nodes carry a generator type, ordered input and output ports, and a symbolic phase slot. A diagram holds nodes, wires joining port to port, ordered boundary lists, and an exact scalar accumulator.
Generators are a small fixed set of types, each with a denotation defined once as a formula in n and d. The starting types are the Z spider and the X spider; the qufinite set later adds Hadamard or Fourier, the triangle node, the W generator, and the dimension-connective generators.
Bang boxes annotate a scoped subgraph with a multiplicity symbol and support instantiate, copy, kill, and merge, may nest, and admit several independent count symbols with an arithmetic of multiplicities.
Scalable sheet-wire notation is representable and translates losslessly to and from bang boxes.

Layer C, rewrite engine. This is the main manipulation layer.

A rule bundles a left-hand pattern, a right-hand builder, side conditions, quantifiers over n and over dimension variables, and the exact scalar it introduces.
The matcher finds occurrences of a pattern and checks side conditions, including dimension side conditions, before a rule may fire. Matches and denotations are cached, and edits trigger incremental re-matching.
The engine applies a rule at a match, returns a new diagram, and records structured provenance from which a proof certificate is emitted.
Above the basic strategy layer sit a normal-form driver that decides diagram equality, an equality-saturation engine for non-destructive rewriting with optimal extraction, and a tactic and proof-search layer that finds a derivation between two stated diagrams.

Layer D, semantics oracle and proof. Three rungs, in order of preference, plus proof machinery.

Stay diagrammatic. Rewrite only. Exact, cheapest, and valid while n and d are symbolic. This is the normal mode of operation.
Symbolic contraction. A full contractor that evaluates an arbitrary diagram while d stays formal, producing a closed symbolic tensor expression simplified through the character-sum layer. This is the general fallback below rewriting.
Numeric contraction. When n and d are concrete, instantiate and contract with real arrays, carrying the exact scalar. Used as the verification oracle and for concrete output.
Beyond the rungs: induction discharges symbolic-n equalities for all n, and certificates record any derivation in a form that can be independently replayed and checked.

The matrix rung is the concrete-instantiation path, not a general fallback beneath rewriting. The general fallback below rewriting is rung 2.

Non-negotiable design rules

These invariants hold across all phases.

The diagram is the single source of truth. Do not introduce a parallel operator or matrix object as a primary representation.
Never construct a matrix or dense tensor while any dimension or count in scope is symbolic.
Rewriting never contracts. Contraction lives only in the semantics oracle. A rewrite is graph to graph.
Generator denotations are leaves of the evaluator, defined once per generator type as a formula in n and d. They are never the working object and are never enumerated per instance.
Dimension is stored per port from the first version. Do not implement dimension as a single global parameter.
Every wire is well formed only when its two ports carry equal dimensions. Composition is otherwise undefined and must be rejected by validation.
Every rule is a quantified equation, "for all n, for all d satisfying its constraints." The matcher enforces those constraints before firing. Some rules are restricted, for example to prime d, and must refuse to fire outside their domain.
Scalars are tracked exactly. Comparison is exact up to floating-point tolerance by default, and quotienting out a global factor is opt-in, never the default. Every rule records the exact scalar it introduces.
Phases are first-class symbolic objects. A phase slot must accept a free symbolic parameter, not only a concrete or root-of-unity value.
Every rewrite emits a certificate step, and provenance is structured so a derivation can be replayed and re-checked independently.
A symbolic-n equality that is claimed as proved, rather than spot-checked, is discharged by induction on the relevant multiplicity, not by sampling alone.
Every build phase ends with a numeric oracle check and a stated completion condition. No phase proceeds until the previous one verifies.
The hot paths, the matcher and the numeric contraction kernel, sit behind narrow interfaces so a later port of only those pieces to a faster language touches nothing else.
Language and tooling

Python. Reference code to learn from: PyZX and DisCoPy. Dependencies: numpy and sympy for computation, pytest, ruff, and mypy for quality.

What to reuse and what is original
PyZX: study for the graph data model, concrete rule implementations, and its numeric tensor evaluator. It is qubit-only, so borrow the architecture and discard the fixed-dimension assumptions.
DisCoPy: study for the pattern of a diagram plus a functor that evaluates it into tensors.
Quantomatic: unmaintained, but the reference for genuine bang-box rewriting semantics.
Original to this project: symbolic dimension carried per port, a full symbolic contractor with d kept formal, dimension-guarded rewrite rules over mixed dimensions, exact scalar and symbolic-phase tracking, symbolic-n proof by induction, proof certificates, a normal-form decision procedure, equality saturation over these diagrams, a tactic layer, and lossless bang-box to scalable translation.

The precise qufinite generator set, the triangle and W generators, the dimension connectives, and which rules carry which dimension constraints, must be taken directly from Wang, "Qufinite ZX-calculus: a unified framework of qudit ZX-calculi," arXiv:2104.06429, and the qufinite ZXW completeness work, at specification time. These must not be reconstructed from memory.

Scope boundaries for the first version

In scope: Z and X spiders through the full qufinite generator set; spider fusion and a broad rule library with strategies; bang boxes with instantiate, nesting, and multiple indices; symbolic phases and exact scalar tracking; the numeric oracle; a full symbolic contractor with d formal; symbolic-n proof by induction; proof certificates; a normal-form decision procedure; equality saturation; a tactic and proof-search layer; lossless scalable-notation interoperability; rewrite caching and incrementality; and a REPL that performs the full Dirac to ZX to Dirac round trip, including a symbolic-n proof and a re-verifiable certificate, interactively.

Out of scope for the first version unless promoted deliberately: a graphical user interface and visual editor, circuit extraction, format bridges to external toolchains, TikZ figure output, mechanical formal verification of the engine itself, differentiation and integration of diagrams, domain application libraries, alternative calculi as pluggable theories, and any performance port of the hot-path kernels. These are deferred, not designed out; the interfaces leave room for them.

Build sequence, in one line

Setup, then dimension algebra, then symbolic phase and exact scalar algebra, then the graph model, then the numeric oracle with exact scalars, then the first vertical slice which is spider fusion verified end to end, then proof certificates, then bang boxes with nesting and multiple indices, then symbolic-n proof by induction, then the character-sum simplifier and full symbolic contraction, then mixed dimensions and the full generator set, then rule breadth and strategies, then caching, then the normal-form decision procedure, then equality saturation, then tactics and proof search, then scalable-notation interoperability, then the REPL, then hardening. The first slice that proves the whole architecture is spider fusion checked by the oracle.

Future improvements, updates, and features

The items below are deferred, not designed out; the first-version architecture leaves room for each, and the invariants above continue to hold for all of them. They are grouped by theme and ordered roughly from most foundational to most speculative.

10.1 Interoperability and import/export

Circuit extraction. Convert a rewritten diagram back into an executable qudit circuit where one exists.
Format bridges. Import and export against QASM, PyZX, DisCoPy, and Quantomatic project files.
TikZ and figure output. Render diagrams to TikZ for direct inclusion in papers.

10.2 Frontend and user experience

Graphical diagram editor and visualization. A visual surface for building, viewing, and interactively rewriting diagrams, with animation of each rewrite step.
Notebook integration. First-class rendering and interaction inside Jupyter, with diagrams displayed inline.
Step explanation mode. For each rewrite, report which rule fired, at which match, and under which dimension conditions, in human-readable form, from the certificate record.

10.3 Performance and the kernel port

Kernel port. Move only the two hot paths, the subgraph matcher and the numeric contraction kernel, to a faster language such as C, Rust, or Java, keeping the Python API identical and differentially testing the port against the original through the oracle.
Matcher acceleration. Index structures and pruning for subgraph isomorphism, and parallel match enumeration.

10.4 Theory breadth

Pluggable calculi. Generalize the generator table and rule library so that related graphical calculi, ZH, ZW, and the full ZXW, can be loaded as alternative theories over the same graph engine. ZXW may be promoted earlier than the others, since the qufinite completeness results are stated in that setting.
Mixed classical and quantum wires. Support diagrams that carry both classical and quantum information.

10.5 Correctness of the engine itself

Property-based and fuzz testing at scale. Generate large random families of diagrams and confirm that every rule preserves the oracle across randomized concrete d and n, continuously. The first version includes a baseline; the deferred item is scaling it up.
Formal verification of the core. Following the direction of VyZX (arXiv:2311.11571), mechanically verify the rewrite core against the linear-algebraic semantics, so the engine itself, not only individual results, carries a correctness guarantee.

10.6 Domain applications

Differentiation and integration of diagrams. Implement diagram differentiation and integration in the sense of Wang, Yeung, and Koch (Quantum 8:1491, 2024), enabling gradient-based work and quantum machine learning directly at the diagrammatic level. The prerequisite is the first version's symbolic-phase support.
Application libraries. Prebuilt constructions for condensed-matter tensor networks, error-correcting codes, and other areas where reasoning over diagram families at symbolic n and d is the natural mode.

10.7 Standard library and ergonomics

Named catalog of states and gates. A library of standard qudit constructions, GHZ and W states, the qudit Fourier transform, Clifford and Clifford-plus-T generators, and common gadgets, each defined once as a diagram parameterized by n and d.
Session provenance and replay. Persist a derivation, reload it, and replay or edit it, using the certificate format as the persistence unit.

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
