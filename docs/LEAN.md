# Lean export

Proofs found by the engine can be re-checked by the proof kernel
(`archytaszx.semantics.kernel`, `:certificate kernel`) and exported to Lean 4
(`:certificate lean proof.lean`).

```sh
cd archytaszx/lean && lake exe cache get && lake build      # once
lake env lean /path/to/proof.lean
```

From Python, `archytaszx.semantics.lean_net.check(source, path)` does both steps. What has no
Lean proof path is listed in [`LEAN_GAPS.txt`](../LEAN_GAPS.txt).

## Semantics

The export targets the Lean library bundled in `archytaszx/lean` (Lean 4.35, Mathlib). There a
diagram is a tensor network over ℂ: the sum, over its wire labels, of the product of its
generators' tensors. Every rewrite rule is a lemma proved there once. An exported proof states
`Equal E s N s' N'` (`s·⟦N⟧ = s'·⟦N'⟧`) between the start and goal diagrams. Each step is an
instance of a rule lemma, applied inside the rest of the diagram by a frame lemma, plus a
relabelling checked by `decide`. The theorem states the diagrams' own scalars; their identity
is a `linear_combination` whose coefficients the engine computes and Lean's `ring` checks.

Families over bang-box multiplicities are networks whose bundled wires carry a chain of
multiplicities and a value below `d^(k₁⋯kₙ)`, read as nested base-`d` digits. Spiders may bundle
any number of legs; a node in node boxes is copies of copies (`Bun`); the environment holds every
multiplicity as a list (`F.env (ksOf [k, j])`). A rule applied inside node boxes is lifted to
every multiplicity by induction, one box at a time, peeling one copy per step. An induction proof,
nested ones included, becomes a Lean `induction` over the same peeling. Dimension equalities a
rewrite assumed are hypotheses of the theorem (outside bang boxes); a scalar outside
the exact model (π, irrational powers, conjugates, sums) is an opaque complex variable, named in
the header, over which the theorem is stated.

## Trust

Lean re-checks the whole proof independently of the engine; `#print axioms main` should list
only `propext`, `Classical.choice` and `Quot.sound`. What remains trusted is how a diagram is
written as a network. A step with no lemma becomes an axiom named `unproved_*`, listed in the
header; the REPL reports their number.

## Verdicts

`:decide` verdicts take the same route. A normal-form or saturation verdict becomes a kernel
theorem from the rewrites that reached it. Other rungs, symbolic contraction among them, are
proved by the normaliser of symbolic sums, else by a proof search. `:induct` works the same way.
When no proof is found, the REPL prints `proof: none the kernel re-checks` and names the method
the verdict rests on.

## The symbolic-sum normaliser

The normaliser (`ArchytasZX.Sym`, `SymNorm`, `SymCanon`, `SymNet`) proves equalities for every
value of the dimensions and phases. A network's entry is translated, once and generically
(`eval_tr`), into a sum of terms: a coefficient times sums over bound indices of Kronecker deltas
`[M ∣ p]` and roots of unity `ω_D^p`, with `p` an integer polynomial in the indices and `M`, `D`
monomials in the dimension symbols. Each move is proved once: a bound index unused, pinned by a
delta whose modulus divides its range, summed by the character sum `Σ_k ω_D^(kA+B) = D [D ∣ A]
ω_D^B`, or counted by `gcd(a, D)`; each delta's pivot substituted into the factors it
determines; roots of unity merged; constant factors evaluated; bound indices renamed and factors
sorted. `nfEx` composes them, `nfEx_den` proves it keeps the value, and `equal_sym` turns an
empty normal form of `s·⟦N⟧ - s'·⟦N'⟧` into `Equal E s N s' N'`. The Lean kernel computes the
normal form (`decide +kernel`); `archytaszx.semantics.lean_sym` mirrors it to decide whether to
export. Phase parameters are atoms `ex g`, whole turns of an integer phase symbol dropped; phases
stored at index `j` need `d > j`, stated as a floor hypothesis. A phase rule `p(k)/D`, `p` an
integer polynomial and `D` a dimension monomial, is a root-of-unity factor at the spider's bound
index (`rule_ex`), so quadratic phases are reached when their sums cancel.

## Induction over multiplicities

A family is proved for every value of every multiplicity by induction on one multiplicity at a
time, outermost box first; a box nested in a box of the same symbol, and a compound multiplicity
such as `k + 1`, get a variable of their own that the theorem then instantiates. A node box meeting
the rest only at the boundary peels one copy per step: the hypothesis is plugged into the peeled
networks and the peeled copies are proved the same way, scaled by the factor a scalar geometric in
the multiplicity gains per copy. A port box on Z and X spiders or a W node's outputs steps by
`Equal.of_port` (`ArchytasZX.Port`): at `k ≥ 1` the split network is the network at an assignment
that advances each X or W bundle's lowest digit by its peeled leg, times a factor of the boundary
alone, so equal networks have equal splits. The zero networks, and for a port box the one-copy
networks, go to the normaliser.

Scalars that differ between the sides by an exact factor are proved with their common part,
outside the exact model or depending on a multiplicity, multiplied on afterwards. A node whose legs
differ in dimension, such as `d^p` and `d^q`, is proved under the hypothesis `d^p = d^q`, which the
theorem states and the REPL reports. A rewrite proof resting on a deferred dimension constraint is
proved over the merged network and transported to the diagrams' own dimensions
(`Equal.of_mapDim'`), the constraint a hypothesis of the theorem. A rewrite proof with a step
lacking a lemma, or one that does not translate, is replaced by the normaliser's proof when that
one exists.

## Refutations

Verdicts that two diagrams differ (`:decide`, `:prove`, `:check`, `:induct`) export a Lean
refutation: `¬ ∀ params, Equal …`, proved at the counterexample or a default sample point. There
`eval_cyc` evaluates both networks exactly as polynomials in `ζ = om M`, computed by the Lean
kernel (`decide +kernel`). At one boundary assignment the scaled difference is rewritten, by
`linear_combination` over the relations of `√x`, `ζ^M = 1` and the prime-gon sums, into a nonzero
integer vector of length at most `φ(M)`. `cyc_ne` shows that vector does not vanish, by the degree
of the cyclotomic minimal polynomial. `√p` for an odd prime is a quadratic Gauss sum, checked by
its square and the sign of its real part. A point involving `π` or an irrational phase is refuted
by ball arithmetic (`ArchytasZX.Ball`): every subterm gets an enclosure from `π`'s 20-digit bounds,
square roots and a Taylor bound on `exp`, and a ball away from zero contradicts the equality.
Diagrams whose interfaces differ are refuted on their leg dimensions; families whose boundaries are
written differently are refuted after instantiating their multiplicities.
