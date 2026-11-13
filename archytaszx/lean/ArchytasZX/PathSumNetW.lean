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

/-- Whether a constraint of the sum is the constant 1. -/
def zeroOK (s : Data) : Bool := s.cons.any (· = [[]])

theorem bsum_zero : ∀ (ys : List ℕ) (σ : ℕ → ZMod 2), bsum ys (fun _ => (0 : ℂ)) σ = 0
  | [], _ => rfl
  | _ :: ys, σ => by simp only [bsum, bsum_zero ys, add_zero]

/-- A sum with the constraint 1 = 0 is zero. -/
theorem val_of_zeroOK (M : ℕ) (c : ℂ) (s : Data) (σ : ℕ → ZMod 2) (h : zeroOK s = true) :
    val M c s σ = 0 := by
  obtain ⟨q, hq, hq1⟩ := List.any_eq_true.1 h
  have hq1' : q = [[]] := of_decide_eq_true hq1
  subst hq1'
  have hs : ∀ τ, summand M c s τ = 0 := by
    intro τ
    unfold summand
    split_ifs with H
    · have h1 := H [[]] (List.mem_append_left _ hq)
      simp [fz, mz] at h1
    · rfl
  unfold val
  rw [bsum_congr _ _ hs, bsum_zero]

end PSem
end ArchytasZX
