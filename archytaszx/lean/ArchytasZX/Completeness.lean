/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net
import ArchytasZX.Rules
import ArchytasZX.WRules

/-!
# Lemmas of the complete axiomatisation

Rule lemmas for template rules of `archytaszx.rewrite.rules_axioms` that no earlier lemma states.
-/

namespace ArchytasZX

open Complex Finset

/-- [P1] (Sym), [P2] (2.10): the outputs of a W node may be permuted. -/
theorem wPerm (E : Env) (a b dz i : ℕ) (os os' iface : List ℕ) (h : os.Perm os') :
    LocalEq E [⟨.W, os, [i], dz, a⟩] [] [⟨.W, os', [i], dz, b⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val]
  exact tW_perm _ _ _ _ (h.map τ)

/-- [P2] (2.9): the legs of a Z spider may be permuted and bent between inputs and outputs. -/
theorem zPerm (E : Env) (a dz : ℕ) (os is os' is' iface : List ℕ)
    (h : (os ++ is).Perm (os' ++ is')) :
    LocalEq E [⟨.Z, os, is, dz, a⟩] [] [⟨.Z, os', is', dz, a⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val]
  rw [← List.map_append, ← List.map_append]
  exact tZ_perm _ _ _ _ (h.map τ)

/-- [P1] (Aso): two ways of nesting W nodes, both the W node with every output (by wFusion). -/
theorem wAso (E : Env) (dz w w' i : ℕ) (o₁ o₂ p₁ p₂ o₃ iface : List ℕ)
    (hw : labIdx E.labs w = dz) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₂ : w ∉ o₂)
    (hwi : w ≠ i) (hi : i ∈ iface ∧ labIdx E.labs i = dz) (hd : 0 < E.dimv dz)
    (ho : o₃.Perm (o₁.erase w ++ o₂))
    (gw : labIdx E.labs w' = dz) (g₁ : w' ∈ p₁) (g₁' : w' ∉ p₁.erase w') (g₂ : w' ∉ p₂)
    (gwi : w' ≠ i) (go : o₃.Perm (p₁.erase w' ++ p₂)) :
    LocalEq E [⟨.W, o₁, [i], dz, 0⟩, ⟨.W, o₂, [w], dz, 0⟩] [w]
      [⟨.W, p₁, [i], dz, 0⟩, ⟨.W, p₂, [w'], dz, 0⟩] [w'] iface 1 1 := by
  intro τ hτ
  rw [wFusion E dz w i o₁ o₂ o₃ iface hw h₁ h₁' h₂ hwi hi hd ho τ hτ,
    wFusion E dz w' i p₁ p₂ o₃ iface gw g₁ g₁' g₂ gwi hi hd go τ hτ]

end ArchytasZX
