/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainBox
import ArchytasZX.ChainInduct

/-!
# Expanded chain boxes and `chainNet`

`Equal.dropIdentity` removes an identity node between a summed label `a` and a boundary label
`b`, renaming `a` to `b` in the rest. For the one-spider chain of `ChainBoxExample`,
`expand_two_chainNet` gives its expansion at `k = 2` (`expandChain` then `toNet`), the identity
at the chain's open end removed and the labels renamed onto the chain scheme, equal to
`chainNet` at `k = 2`.
-/

namespace ArchytasZX

open Finset

/-- `a` read as `b`. -/
def swapTo (a b : ℕ) (x : ℕ) : ℕ := if x = a then b else x

theorem val_ident (E : Env) (σ : ℕ → ℕ) (a b d : ℕ) :
    val E σ ⟨.I, [b], [a], d, 0⟩ = tI (E.dimv d) (σ b) (σ a) := by
  simp [val]

/-- An identity node from the summed label `a` to the boundary label `b` dropped, `a` read as
`b` in the rest. -/
theorem Equal.dropIdentity (E : Env) (C : List Node) (iC bnd : List ℕ) (a b d : ℕ)
    (hd : labDim E b ≤ E.dimv d) (hda : labDim E a = E.dimv d) (hab : a ≠ b) (hb : b ∈ bnd)
    (hbC : b ∉ iC) :
    Equal E 1 ⟨C ++ [⟨.I, [b], [a], d, 0⟩], iC ++ [a], bnd⟩ 1
      ⟨C.map (Node.rename (swapTo a b)), iC, bnd⟩ := by
  refine ⟨List.Perm.refl _, fun σ hσ => ?_⟩
  simp only [one_mul, eval]
  rw [sumOver_append]
  apply sumOver_congr
  intro τ hτ _
  have hτb : τ b < E.dimv d := by
    rw [hτ b hbC]
    exact lt_of_lt_of_le (hσ b hb) hd
  simp only [sumOver, hda, prodVal, List.map_append, List.map_cons, List.map_nil,
    List.prod_append, List.prod_cons, List.prod_nil, mul_one, val_ident]
  rw [Finset.sum_eq_single (τ b)]
  · have hf : Function.update τ a (τ b) = τ ∘ swapTo a b := by
      funext x
      by_cases hx : x = a
      · subst hx; simp [swapTo]
      · simp [swapTo, hx, Function.update_of_ne hx]
    have hupd : Function.update τ a (τ b) b = τ b := Function.update_of_ne (Ne.symm hab) _ _
    rw [hupd, Function.update_self, hf, List.map_map]
    simp only [tI, hτb, and_self, ↓reduceIte, mul_one]
    congr 1
    refine List.map_congr_left fun n _ => ?_
    simp [val_rename]
  · intro j _ hj
    have hupd : Function.update τ a j b = τ b := Function.update_of_ne (Ne.symm hab) _ _
    simp only [hupd, Function.update_self, tI]
    rw [if_neg (fun h => hj h.1.symm), mul_zero]
  · intro h
    exact absurd (Finset.mem_range.2 hτb) h

namespace ChainBoxExample

/-- Every label of dimension 2. -/
noncomputable def E : Env := { dimv := fun _ => 2, phase := fun _ _ => 0, labs := [] }

/-- The chain at `k = 0`: the head identity from the input `3` to the cut label `5`. -/
def base : Net := ⟨[⟨.I, [5], [3], 0, 0⟩], [], [3, 5]⟩

/-- Each copy: the spider from the cut label `5` to the next cut `105` and its own output `150`. -/
def copy (_ : ℕ) : Net := ⟨[⟨.Z, [105, 150], [5], 0, 0⟩], [], [5, 105, 150]⟩

/-- The relabelling of the expansion's labels onto the chain scheme. -/
def π : List (ℕ × ℕ) :=
  [(0, 150), (1, 250), (2, 205), (10, 5), (11, 105), (150, 0), (250, 1), (205, 2), (5, 10),
    (105, 11)]

theorem net_two : (D.expandChain c 2).toNet 10 =
    ⟨[⟨.I, [10], [3], 0, 0⟩, ⟨.Z, [11, 0], [10], 0, 0⟩, ⟨.Z, [12, 1], [11], 0, 0⟩] ++
      [⟨.I, [2], [12], 0, 0⟩], [10, 11] ++ [12], [0, 1, 2, 3]⟩ := by
  decide +kernel

/-- The expansion at `k = 2`, renamed onto the chain scheme, is `chainNet` at `k = 2`. -/
theorem expand_two_chainNet :
    Equal E 1 (((D.expandChain c 2).toNet 10).rename π) 1 (chainNet 100 [5] base copy 2) := by
  have h0 := Equal.dropIdentity E
    [⟨.I, [10], [3], 0, 0⟩, ⟨.Z, [11, 0], [10], 0, 0⟩, ⟨.Z, [12, 1], [11], 0, 0⟩] [10, 11]
    [0, 1, 2, 3] 12 2 0 (by simp [labDim, E]) (by simp [labDim, E]) (by decide) (by decide)
    (by decide)
  have h1 := Equal.rename E π _ _ (by decide +kernel) (by decide +kernel) h0
  rw [net_two]
  refine h1.reorder E ?_ ?_ ?_ <;> decide +kernel

end ChainBoxExample

end ArchytasZX
