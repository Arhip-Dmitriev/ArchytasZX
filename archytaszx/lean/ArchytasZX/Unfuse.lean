/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bundle

/-!
# Port-box unfusion

A spider whose last leg bundles `k` legs equals a plain spider with one new leg, wired to a
phaseless spider holding the bundle.
-/

namespace ArchytasZX

open Complex Finset

/-- A Z spider split into two along one shared value. -/
theorem tZ_split (d : ℕ) (φ : ℕ → ℝ) (A B : List ℕ) :
    tZ d φ (A ++ B) = ∑ j ∈ range d, tZ d φ (A ++ [j]) * tZ d (fun _ => 0) (j :: B) := by
  have hl : ∀ j ∈ range d, tZ d φ (A ++ [j]) * tZ d (fun _ => 0) (j :: B) =
      if ∀ v ∈ A ++ B, v = j then ex (φ j) else 0 := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    rw [tZ_at _ _ _ j (by simp), tZ_at _ _ _ j (by simp)]
    have c1 : (j < d ∧ ∀ v ∈ A ++ [j], v = j) ↔ ∀ v ∈ A, v = j := by
      simp only [hj', true_and, List.mem_append, List.mem_singleton, or_imp, forall_and,
        forall_eq, and_true]
    have c2 : (j < d ∧ ∀ v ∈ j :: B, v = j) ↔ ∀ v ∈ B, v = j := by simp [hj']
    have c3 : (∀ v ∈ A ++ B, v = j) ↔ (∀ v ∈ A, v = j) ∧ (∀ v ∈ B, v = j) := by
      simp [or_imp, forall_and]
    simp only [c1, c2, c3, ex_zero]
    by_cases hA : ∀ v ∈ A, v = j <;> by_cases hB : ∀ v ∈ B, v = j
    · simp only [eq_true hA, eq_true hB, and_self, if_true, mul_one]
    · simp only [eq_true hA, eq_false hB, and_false, if_true, if_false, mul_zero]
    · simp only [eq_false hA, eq_true hB, false_and, if_true, if_false, zero_mul]
    · simp only [eq_false hA, eq_false hB, false_and, if_false, zero_mul]
  rw [Finset.sum_congr rfl hl]
  rfl

theorem tZ_phaseless (d : ℕ) (φ : ℕ → ℝ) (h : ∀ k < d, ex (φ k) = 1) (L : List ℕ) :
    tZ d φ L = tZ d (fun _ => 0) L := by
  unfold tZ
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [h k (Finset.mem_range.1 hk), ex_zero]

theorem bundled_split (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (hd tl : List ℕ) (cs : List (List ℕ))
    (h : tl.length = cs.length) :
    bundled mult d ((hd ++ tl).map σ) cs = hd.map σ ++ tailVals mult d (tl.map σ) cs := by
  rw [List.map_append, bundled_app _ _ _ _ _ (by simp [h])]

theorem bundled_one (mult : ℕ → ℕ) (d : ℕ) (v : ℕ) (c : List ℕ) :
    bundled mult d [v] [c] = digsC mult d c v := by
  simp [bundled, tailVals]

theorem tailVals_absent (mult : ℕ → ℕ) (d : ℕ) (τ : ℕ → ℕ) (y j : ℕ) (tl : List ℕ)
    (cs : List (List ℕ)) (h : y ∉ tl) :
    tailVals mult d (tl.map (Function.update τ y j)) cs = tailVals mult d (tl.map τ) cs := by
  rw [map_update_absent τ y j tl h]

theorem unfuseZB (E : Env) (cs : List (List ℕ)) (c : List ℕ) (a cph dz y b : ℕ)
    (os hd tl o₂ i₂ iface : List ℕ) (htl : tl.length = cs.length)
    (hy : labIdx E.labs y = dz) (hyl : y ∉ os ++ hd ++ tl ++ [b])
    (hm : (o₂ ++ i₂).Perm (os ++ hd ++ [y]))
    (hc : ∀ k < E.dimv dz, ex (E.phase cph k) = 1) :
    LocalEq E [⟨.ZB (cs ++ [c]), os, hd ++ tl ++ [b], dz, a⟩] []
      [⟨zk cs, o₂, i₂ ++ tl, dz, a⟩, ⟨.ZB [c], [], [y, b], dz, cph⟩] [y] iface 1 1 := by
  intro τ _
  simp only [List.mem_append, List.mem_singleton, not_or] at hyl
  obtain ⟨⟨⟨hyo, hyh⟩, hyt⟩, hyb⟩ := hyl
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E y dz hy, val_zk]
  simp only [val, List.map_nil, List.nil_append]
  rw [List.append_assoc hd, bundled_split _ _ _ _ _ _ (by simp [htl]),
    List.map_append, tailVals_append _ _ _ _ _ _ (by simp [htl])]
  simp only [List.map_cons, List.map_nil, Function.update_self, Function.update_of_ne (Ne.symm hyb)]
  simp only [show ∀ j, bundled E.mult (E.dimv dz) [j, τ b] [c] = j :: digsC E.mult (E.dimv dz) c (τ b)
    from fun j => by simp [bundled, tailVals]]
  rw [show tailVals E.mult (E.dimv dz) [τ b] [c] = digsC E.mult (E.dimv dz) c (τ b) by
    simp [tailVals]]
  rw [← List.append_assoc, ← List.append_assoc, tZ_split]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [tZ_phaseless _ _ hc, bundled_split _ _ _ _ _ _ htl,
    tailVals_absent _ _ τ y j tl cs hyt]
  congr 1
  refine tZ_perm _ _ _ _ ?_
  have hp := hm.map (Function.update τ y j)
  simp only [List.map_append, List.map_cons, List.map_nil, Function.update_self,
    map_update_absent τ y j os hyo, map_update_absent τ y j hd hyh] at hp
  rw [← List.append_assoc]
  refine (?_ : List.Perm _ _).trans (List.Perm.append_right _ hp).symm
  simp only [List.append_assoc]
  exact List.Perm.append_left _ (List.Perm.append_left _ List.perm_append_comm)

theorem tX_phaseless (d : ℕ) (φ : ℕ → ℝ) (h : ∀ k < d, ex (φ k) = 1) (O I : List ℕ) :
    tX d φ O I = tX d (fun _ => 0) O I := by
  unfold tX
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [h k (Finset.mem_range.1 hk), ex_zero]

theorem fou_unitary' (d : ℕ) (hd : 0 < d) (k k' : ℕ) (hk : k < d) (hk' : k' < d) :
    ∑ j ∈ range d, (starRingEnd ℂ) (fou d j k) * fou d j k' = if k = k' then 1 else 0 := by
  have := fou_unitary d hd k' k hk' hk
  rw [show (if k = k' then (1 : ℂ) else 0) = if k' = k then 1 else 0 by simp [eq_comm], ← this]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- An X spider split into two along one output. -/
theorem tX_split_out (d : ℕ) (hd : 0 < d) (φ : ℕ → ℝ) (O G I : List ℕ) :
    tX d φ (O ++ G) I = ∑ j ∈ range d, tX d φ (O ++ [j]) I * tX d (fun _ => 0) G [j] := by
  unfold tX
  rw [eq_comm]
  simp only [Finset.sum_mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [Finset.sum_comm]
  have : ∀ m ∈ range d, ∑ j ∈ range d, ex (φ k) * ((O ++ [j]).map fun o => fou d o k).prod *
      (I.map fun i => (starRingEnd ℂ) (fou d i k)).prod * (ex 0 * (G.map fun o => fou d o m).prod *
        ([j].map fun i => (starRingEnd ℂ) (fou d i m)).prod) =
      (if k = m then 1 else 0) * (ex (φ k) * (O.map fun o => fou d o k).prod *
        (I.map fun i => (starRingEnd ℂ) (fou d i k)).prod * (G.map fun o => fou d o m).prod) := by
    intro m hm
    rw [← fou_unitary d hd k m (Finset.mem_range.1 hk) (Finset.mem_range.1 hm), Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append, List.prod_cons,
      List.prod_nil, mul_one, ex_zero]
    ring
  rw [Finset.sum_congr rfl this]
  simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_range.1 hk, if_true,
    Finset.mem_range, List.map_append, List.prod_append]
  ring

/-- An X spider split into two along one input. -/
theorem tX_split_in (d : ℕ) (hd : 0 < d) (φ : ℕ → ℝ) (O I G : List ℕ) :
    tX d φ O (I ++ G) = ∑ j ∈ range d, tX d φ O (I ++ [j]) * tX d (fun _ => 0) [j] G := by
  unfold tX
  rw [eq_comm]
  simp only [Finset.sum_mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [Finset.sum_comm]
  have : ∀ m ∈ range d, ∑ j ∈ range d, ex (φ k) * (O.map fun o => fou d o k).prod *
      ((I ++ [j]).map fun i => (starRingEnd ℂ) (fou d i k)).prod * (ex 0 *
        ([j].map fun o => fou d o m).prod * (G.map fun i => (starRingEnd ℂ) (fou d i m)).prod) =
      (if k = m then 1 else 0) * (ex (φ k) * (O.map fun o => fou d o k).prod *
        (I.map fun i => (starRingEnd ℂ) (fou d i k)).prod *
        (G.map fun i => (starRingEnd ℂ) (fou d i m)).prod) := by
    intro m hm
    rw [← fou_unitary' d hd k m (Finset.mem_range.1 hk) (Finset.mem_range.1 hm), Finset.sum_mul]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append, List.prod_cons,
      List.prod_nil, mul_one, ex_zero]
    ring
  rw [Finset.sum_congr rfl this]
  simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_range.1 hk, if_true,
    Finset.mem_range, List.map_append, List.prod_append]
  ring


theorem unfuseXB (E : Env) (co ci : List (List ℕ)) (c : List ℕ) (a cph dz y b : ℕ)
    (hd tl is o₂ iface : List ℕ) (htl : tl.length = co.length) (hd0 : 0 < E.dimv dz)
    (hy : labIdx E.labs y = dz) (hyl : y ∉ hd ++ tl ++ [b] ++ is)
    (ho : o₂.Perm (hd ++ [y]))
    (hc : ∀ k < E.dimv dz, ex (E.phase cph k) = 1) :
    LocalEq E [⟨.XB (co ++ [c]) ci, hd ++ tl ++ [b], is, dz, a⟩] []
      [⟨xk co ci, o₂ ++ tl, is, dz, a⟩, ⟨.XB [c] [], [b], [y], dz, cph⟩] [y] iface 1 1 := by
  intro τ _
  simp only [List.mem_append, List.mem_singleton, not_or] at hyl
  obtain ⟨⟨⟨hyh, hyt⟩, hyb⟩, hyi⟩ := hyl
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E y dz hy, val_xk]
  simp only [val]
  rw [List.append_assoc hd, bundled_split _ _ _ _ _ _ (by simp [htl]),
    List.map_append, tailVals_append _ _ _ _ _ _ (by simp [htl])]
  try simp only [List.map_cons, List.map_nil, Function.update_self]
  rw [show tailVals E.mult (E.dimv dz) [τ b] [c] = digsC E.mult (E.dimv dz) c (τ b) by
    simp [tailVals], ← List.append_assoc, tX_split_out _ hd0]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [bundled_split _ _ _ _ _ _ htl, tailVals_absent _ _ τ y j tl co hyt,
    map_update_absent τ y j is hyi, Function.update_of_ne (Ne.symm hyb), bundled_one,
    show bundled E.mult (E.dimv dz) [j] [] = [j] by simp [bundled, tailVals], tX_phaseless _ _ hc]
  congr 1
  refine tX_perm _ _ _ _ _ _ ?_ (List.Perm.refl _)
  have hp := ho.map (Function.update τ y j)
  simp only [List.map_append, List.map_cons, List.map_nil, Function.update_self,
    map_update_absent τ y j hd hyh] at hp
  refine (?_ : List.Perm _ _).trans (List.Perm.append_right _ hp).symm
  simp only [List.append_assoc]
  exact List.Perm.append_left _ List.perm_append_comm

theorem unfuseXBi (E : Env) (co ci : List (List ℕ)) (c : List ℕ) (a cph dz y b : ℕ)
    (os hd tl i₂ iface : List ℕ) (htl : tl.length = ci.length) (hd0 : 0 < E.dimv dz)
    (hy : labIdx E.labs y = dz) (hyl : y ∉ os ++ hd ++ tl ++ [b])
    (hi : i₂.Perm (hd ++ [y]))
    (hc : ∀ k < E.dimv dz, ex (E.phase cph k) = 1) :
    LocalEq E [⟨.XB co (ci ++ [c]), os, hd ++ tl ++ [b], dz, a⟩] []
      [⟨xk co ci, os, i₂ ++ tl, dz, a⟩, ⟨.XB [] [c], [y], [b], dz, cph⟩] [y] iface 1 1 := by
  intro τ _
  simp only [List.mem_append, List.mem_singleton, not_or] at hyl
  obtain ⟨⟨⟨hyo, hyh⟩, hyt⟩, hyb⟩ := hyl
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E y dz hy, val_xk]
  simp only [val]
  rw [List.append_assoc hd, bundled_split _ _ _ _ _ _ (by simp [htl]),
    List.map_append, tailVals_append _ _ _ _ _ _ (by simp [htl])]
  try simp only [List.map_cons, List.map_nil, Function.update_self]
  rw [show tailVals E.mult (E.dimv dz) [τ b] [c] = digsC E.mult (E.dimv dz) c (τ b) by
    simp [tailVals], ← List.append_assoc, tX_split_in _ hd0]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [bundled_split _ _ _ _ _ _ htl, tailVals_absent _ _ τ y j tl ci hyt,
    map_update_absent τ y j os hyo, Function.update_of_ne (Ne.symm hyb), bundled_one,
    show bundled E.mult (E.dimv dz) [j] [] = [j] by simp [bundled, tailVals], tX_phaseless _ _ hc]
  congr 1
  refine tX_perm _ _ _ _ _ _ (List.Perm.refl _) ?_
  have hp := hi.map (Function.update τ y j)
  simp only [List.map_append, List.map_cons, List.map_nil, Function.update_self,
    map_update_absent τ y j hd hyh] at hp
  refine (?_ : List.Perm _ _).trans (List.Perm.append_right _ hp).symm
  simp only [List.append_assoc]
  exact List.Perm.append_left _ List.perm_append_comm

end ArchytasZX
