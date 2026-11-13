/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainInduct

/-!
# Periodic families

`Equal.periodic` gives two families equal at every index from the pairs at `0 .. p - 1` and,
for each side, its network at `k + p` equal to `a` times its network at `k` read through a
relabelling `f k` that both sides share (`kernel.periodic`'s scaling of both sides by one
constant).
-/

namespace ArchytasZX

/-- Two families equal at every index from the pairs below the period `p` and each side's
step from `k` to `k + p`: the side at `k + p` with scalar `s (k + p)` equal to the side at `k`
relabelled by `f k` with scalar `a * s k`. -/
theorem Equal.periodic (E : Env) (p : ℕ) (hp : 0 < p) (S G : ℕ → Net) (s g : ℕ → ℂ) (a : ℂ)
    (f : ℕ → ℕ → ℕ) (hf : ∀ k, Function.Injective (f k))
    (hd : ∀ k x, labDim E (f k x) = labDim E x)
    (base : ∀ r < p, Equal E (s r) (S r) (g r) (G r))
    (stepS : ∀ k, Equal E (s (k + p)) (S (k + p)) (a * s k) ((S k).mapL (f k)))
    (stepG : ∀ k, Equal E (g (k + p)) (G (k + p)) (a * g k) ((G k).mapL (f k))) :
    ∀ k, Equal E (s k) (S k) (g k) (G k) := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    by_cases hk : k < p
    · exact base k hk
    · obtain ⟨j, rfl⟩ : ∃ j, k = j + p := ⟨k - p, by omega⟩
      have hj := (ih j (by omega)).mapL E (f j) (hf j) (hd j)
      have hm : Equal E (a * s j) ((S j).mapL (f j)) (a * g j) ((G j).mapL (f j)) :=
        ⟨hj.1, fun σ hσ => by
          have := hj.2 σ hσ
          simp only [mul_assoc]
          rw [this]⟩
      exact (stepS j).trans (hm.trans (stepG j).symm)

end ArchytasZX
