/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumDigits

/-!
# Legless and weighted spiders as fragments

A legless Z spider has the legless X spider's entry (`val_z_legless`), so `x_frag_dig` gives its
fragment. A weighted Z or X spider whose coefficient at `v` is `C` times `ex` of its phase at
`v` is `C` times the unweighted spider (`val_zw`, `val_xw`), so its fragment is the unweighted
one scaled by `C` (`w_frag`).
-/

namespace ArchytasZX

/-- The node with another kind. -/
def Node.withKind (n : Node) (k : Kind) : Node := { n with kind := k }

namespace PSem

open Complex Finset

/-- A fragment with its constant multiplied by `C`. -/
def Frag.scale (C : ℂ) (f : Frag) : Frag := { f with c := C * f.c }

theorem fval_scale (M : ℕ) (C : ℂ) (f : Frag) (τ : ℕ → ZMod 2) :
    fval M (f.scale C) τ = C * fval M f τ := by
  unfold fval Frag.scale
  exact val_smul M C f.c _ τ

theorem val_z_legless (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .Z) (ho : n.outs = [])
    (hi : n.ins = []) : ArchytasZX.val E σ n = ArchytasZX.val E σ (n.withKind .X) := by
  unfold ArchytasZX.val Node.withKind
  rw [hk, ho, hi]
  simp [tZ, tX]

theorem val_zw (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .ZW) (C : ℂ)
    (hc : ∀ v, E.coef n.ph v = C * ex (E.phase n.ph v)) :
    ArchytasZX.val E σ n = C * ArchytasZX.val E σ (n.withKind .Z) := by
  unfold ArchytasZX.val Node.withKind
  rw [hk]
  simp only [tZw, tZ, Finset.mul_sum, hc]
  apply Finset.sum_congr rfl
  intro k _
  split_ifs <;> simp

theorem val_xw (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .XW) (C : ℂ)
    (hc : ∀ v, E.coef n.ph v = C * ex (E.phase n.ph v)) :
    ArchytasZX.val E σ n = C * ArchytasZX.val E σ (n.withKind .X) := by
  unfold ArchytasZX.val Node.withKind
  rw [hk]
  simp only [tXw, tX, Finset.mul_sum, hc]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- A node `C` times another whose entry is a fragment's value has the fragment scaled by
`C`. -/
theorem w_frag (E : Env) (σ : ℕ → ℕ) (n n' : Node) (M : ℕ) (f : Frag) (τ : ℕ → ZMod 2) (C : ℂ)
    (hw : ArchytasZX.val E σ n = C * ArchytasZX.val E σ n')
    (hf : ArchytasZX.val E σ n' = fval M f τ) : ArchytasZX.val E σ n = fval M (f.scale C) τ := by
  rw [hw, hf, fval_scale]

end PSem
end ArchytasZX
