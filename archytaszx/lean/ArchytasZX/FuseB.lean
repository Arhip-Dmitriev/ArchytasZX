/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Unfuse

/-!
# Fusion with a bundled leg

Spider fusion along a plain wire when one spider's last leg bundles `k` legs.
-/

namespace ArchytasZX

open Complex Finset

/-- The leg values of `l` with `w` set to `j`, `w` occurring once, as `j` and the rest. -/
theorem perm_update_once (τ : ℕ → ℕ) (w j : ℕ) (l : List ℕ) (h : w ∈ l) (hn : w ∉ l.erase w) :
    (l.map (Function.update τ w j)).Perm (j :: (l.erase w).map τ) := by
  have hp := (List.perm_cons_erase h).map (Function.update τ w j)
  rw [List.map_cons, Function.update_self, map_update_absent τ w j (l.erase w) hn] at hp
  exact hp

/-- Two Z spiders summed over one shared value. -/
theorem tZ_fuse (d : ℕ) (φa φb φc : ℕ → ℝ)
    (hph : ∀ k < d, ex (φc k) = ex (φa k) * ex (φb k)) (P Q : List ℕ) :
    ∑ j ∈ range d, tZ d φa (j :: P) * tZ d φb (j :: Q) = tZ d φc (P ++ Q) := by
  unfold tZ
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' := Finset.mem_range.1 hj
  rw [← tZ, ← tZ, tZ_at _ _ _ j (by simp), tZ_at _ _ _ j (by simp)]
  have c : ∀ L : List ℕ, (j < d ∧ ∀ v ∈ j :: L, v = j) ↔ ∀ v ∈ L, v = j := by simp [hj']
  have c2 : (∀ v ∈ P ++ Q, v = j) ↔ (∀ v ∈ P, v = j) ∧ ∀ v ∈ Q, v = j := by
    simp [or_imp, forall_and]
  simp only [c, c2]
  by_cases hP : ∀ v ∈ P, v = j <;> by_cases hQ : ∀ v ∈ Q, v = j
  · simp only [eq_true hP, eq_true hQ, and_self, ite_true, hph j hj']
  · simp only [eq_true hP, eq_false hQ, and_false, ite_true, ite_false, mul_zero]
  · simp only [eq_false hP, eq_true hQ, false_and, ite_true, ite_false, zero_mul]
  · simp only [eq_false hP, eq_false hQ, false_and, ite_false, zero_mul]

/-- Two X spiders summed over one value, an output of the first and an input of the second. -/
theorem tX_fuse (d : ℕ) (hd : 0 < d) (φa φb φc : ℕ → ℝ)
    (hph : ∀ k < d, ex (φc k) = ex (φa k) * ex (φb k)) (P₁ P₂ Q₁ Q₂ : List ℕ) :
    ∑ j ∈ range d, tX d φa (j :: P₁) P₂ * tX d φb Q₁ (j :: Q₂) =
      tX d φc (P₁ ++ Q₁) (P₂ ++ Q₂) := by
  set F : ℕ → ℕ → ℂ := fun x k => fou d x k
  set G : ℕ → ℕ → ℂ := fun x k => (starRingEnd ℂ) (fou d x k)
  set A : ℕ → ℂ := fun k => ex (φa k) * (P₁.map (F · k)).prod * (P₂.map (G · k)).prod
  set B : ℕ → ℂ := fun k => ex (φb k) * (Q₁.map (F · k)).prod * (Q₂.map (G · k)).prod
  have hl : ∀ j, tX d φa (j :: P₁) P₂ = ∑ k ∈ range d, F j k * A k := by
    intro j; unfold tX
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [List.map_cons, List.prod_cons, A, F]; ring
  have hr : ∀ j, tX d φb Q₁ (j :: Q₂) = ∑ k ∈ range d, G j k * B k := by
    intro j; unfold tX
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [List.map_cons, List.prod_cons, B, G]; ring
  simp only [hl, hr, Finset.sum_mul_sum]
  rw [Finset.sum_comm]
  have inner : ∀ k ∈ range d, ∑ j ∈ range d, ∑ k' ∈ range d, F j k * A k * (G j k' * B k') =
      A k * B k := by
    intro k hk
    rw [Finset.sum_comm]
    have : ∀ k' ∈ range d, ∑ j ∈ range d, F j k * A k * (G j k' * B k') =
        (if k = k' then 1 else 0) * (A k * B k') := by
      intro k' hk'
      rw [← fou_unitary d hd k k' (Finset.mem_range.1 hk) (Finset.mem_range.1 hk'),
        Finset.sum_mul]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [F, G]; ring
    rw [Finset.sum_congr rfl this]
    simp [Finset.sum_ite_eq, Finset.mem_range.1 hk]
  rw [Finset.sum_congr rfl inner]
  unfold tX
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hph k (Finset.mem_range.1 hk)]
  simp only [A, B, List.map_append, List.prod_append, F, G]
  ring

theorem labDim_w (E : Env) (w dz : ℕ) (hw : labIdx E.labs w = dz) : labDim E w = E.dimv dz := by
  simp only [labDim]; simp only [labIdx] at hw; rw [hw]

theorem tailVals_append_map (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (a b : List ℕ)
    (ca cb : List (List ℕ)) (h : a.length = ca.length) :
    tailVals mult d ((a ++ b).map σ) (ca ++ cb) =
      tailVals mult d (a.map σ) ca ++ tailVals mult d (b.map σ) cb := by
  rw [List.map_append, tailVals_append _ _ _ _ _ _ (by simp [h])]

/-- A Z fusion whose spiders carry extra leg values `D₁`, `D₂`. -/
theorem fusionZ_gen (d : ℕ) (φa φb φc : ℕ → ℝ)
    (hph : ∀ k < d, ex (φc k) = ex (φa k) * ex (φb k)) (τ : ℕ → ℕ) (w : ℕ)
    (P Q R : List ℕ) (h₁ : w ∈ P) (h₁' : w ∉ P.erase w) (h₂ : w ∈ Q) (h₂' : w ∉ Q.erase w)
    (hm : R.Perm (P.erase w ++ Q.erase w)) (D₁ D₂ : List ℕ) :
    ∑ j ∈ range d, tZ d φa (P.map (Function.update τ w j) ++ D₁) *
      tZ d φb (Q.map (Function.update τ w j) ++ D₂) = tZ d φc (R.map τ ++ (D₁ ++ D₂)) := by
  have e : ∀ j, tZ d φa (P.map (Function.update τ w j) ++ D₁) *
      tZ d φb (Q.map (Function.update τ w j) ++ D₂) =
      tZ d φa (j :: ((P.erase w).map τ ++ D₁)) * tZ d φb (j :: ((Q.erase w).map τ ++ D₂)) :=
    fun j => by
      congr 1
      · exact tZ_perm _ _ _ _ ((perm_update_once τ w j _ h₁ h₁').append_right D₁)
      · exact tZ_perm _ _ _ _ ((perm_update_once τ w j _ h₂ h₂').append_right D₂)
  simp only [e]
  rw [tZ_fuse _ _ _ _ hph]
  refine tZ_perm _ _ _ _ ?_
  have := (hm.map τ).append_right (D₁ ++ D₂)
  refine List.Perm.trans ?_ this.symm
  simp only [List.map_append, List.append_assoc]
  exact List.Perm.append_left _ (by
    rw [← List.append_assoc, ← List.append_assoc]
    exact (List.perm_append_comm.append_right D₂).trans (by simp [List.perm_append_comm]))

theorem fusionZG (E : Env) (cs₁ cs₂ : List (List ℕ)) (a b c dz w : ℕ)
    (o₁ i₁ t₁ o₂ i₂ t₂ o₃ i₃ iface : List ℕ) (ht₁ : t₁.length = cs₁.length)
    (ht₂ : t₂.length = cs₂.length)
    (h₁ : w ∈ o₁ ++ i₁) (h₁' : w ∉ (o₁ ++ i₁).erase w)
    (h₂ : w ∈ o₂ ++ i₂) (h₂' : w ∉ (o₂ ++ i₂).erase w) (hwt : w ∉ t₁ ++ t₂)
    (hm : (o₃ ++ i₃).Perm ((o₁ ++ i₁).erase w ++ (o₂ ++ i₂).erase w))
    (hw : labIdx E.labs w = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase c k) = ex (E.phase a k) * ex (E.phase b k)) :
    LocalEq E [⟨zk cs₁, o₁, i₁ ++ t₁, dz, a⟩, ⟨zk cs₂, o₂, i₂ ++ t₂, dz, b⟩] [w]
      [⟨zk (cs₁ ++ cs₂), o₃, i₃ ++ (t₁ ++ t₂), dz, c⟩] [] iface 1 1 := by
  intro τ _
  simp only [List.mem_append, not_or] at hwt
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val_zk, labDim_w E w dz hw]
  rw [bundled_split _ _ _ _ _ _ (by simp [ht₁, ht₂]), tailVals_append_map _ _ _ _ _ _ _ ht₁]
  simp only [bundled_split _ _ _ _ _ _ ht₁, bundled_split _ _ _ _ _ _ ht₂,
    tailVals_absent _ _ τ w _ t₁ cs₁ hwt.1, tailVals_absent _ _ τ w _ t₂ cs₂ hwt.2,
    ← List.append_assoc, ← List.map_append]
  rw [fusionZ_gen _ _ _ _ hph τ w _ _ _ h₁ h₁' h₂ h₂' hm]
  simp only [List.append_assoc]

/-- Assembles an X fusion from its value form. -/
theorem fusionX_core (d : ℕ) (hd : 0 < d) (φa φb φc : ℕ → ℝ)
    (hph : ∀ k < d, ex (φc k) = ex (φa k) * ex (φb k)) (τ : ℕ → ℕ) (w : ℕ)
    (o₁ i₁ o₂ i₂ : List ℕ) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₁i : w ∉ i₁)
    (h₂ : w ∈ i₂) (h₂' : w ∉ i₂.erase w) (h₂o : w ∉ o₂) (D₁ D₂ D₃ D₄ : List ℕ) :
    ∑ j ∈ range d, tX d φa (o₁.map (Function.update τ w j) ++ D₁)
        (i₁.map (Function.update τ w j) ++ D₂) *
      tX d φb (o₂.map (Function.update τ w j) ++ D₃) (i₂.map (Function.update τ w j) ++ D₄) =
    tX d φc ((o₁.erase w).map τ ++ D₁ ++ (o₂.map τ ++ D₃))
      (i₁.map τ ++ D₂ ++ ((i₂.erase w).map τ ++ D₄)) := by
  have e : ∀ j, tX d φa (o₁.map (Function.update τ w j) ++ D₁)
        (i₁.map (Function.update τ w j) ++ D₂) *
      tX d φb (o₂.map (Function.update τ w j) ++ D₃) (i₂.map (Function.update τ w j) ++ D₄) =
      tX d φa (j :: ((o₁.erase w).map τ ++ D₁)) (i₁.map τ ++ D₂) *
      tX d φb (o₂.map τ ++ D₃) (j :: ((i₂.erase w).map τ ++ D₄)) := by
    intro j
    rw [map_update_absent τ w j i₁ h₁i, map_update_absent τ w j o₂ h₂o]
    congr 1
    · exact tX_perm _ _ _ _ _ _ ((perm_update_once τ w j _ h₁ h₁').append_right D₁)
        (List.Perm.refl _)
    · exact tX_perm _ _ _ _ _ _ (List.Perm.refl _)
        ((perm_update_once τ w j _ h₂ h₂').append_right D₄)
  simp only [e]
  exact tX_fuse d hd φa φb φc hph _ _ _ _

/-- An X fusion whose spiders and result carry extra leg values `D₁ … D₄`. -/
theorem fusionX_gen (E : Env) (a b c dz w : ℕ) (o₁ i₁ o₂ i₂ o₃ i₃ : List ℕ)
    (hd : 0 < E.dimv dz) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₁i : w ∉ i₁)
    (h₂ : w ∈ i₂) (h₂' : w ∉ i₂.erase w) (h₂o : w ∉ o₂)
    (ho : o₃.Perm (o₁.erase w ++ o₂)) (hi : i₃.Perm (i₁ ++ i₂.erase w))
    (hw : labIdx E.labs w = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase c k) = ex (E.phase a k) * ex (E.phase b k))
    (τ : ℕ → ℕ) (D₁ D₂ D₃ D₄ : List ℕ) :
    ∑ j ∈ range (labDim E w), tX (E.dimv dz) (E.phase a)
        (o₁.map (Function.update τ w j) ++ D₁) (i₁.map (Function.update τ w j) ++ D₂) *
      tX (E.dimv dz) (E.phase b) (o₂.map (Function.update τ w j) ++ D₃)
        (i₂.map (Function.update τ w j) ++ D₄) =
    tX (E.dimv dz) (E.phase c) (o₃.map τ ++ (D₁ ++ D₃)) (i₃.map τ ++ (D₂ ++ D₄)) := by
  rw [labDim_w E w dz hw, fusionX_core _ hd _ _ _ hph τ w o₁ i₁ o₂ i₂ h₁ h₁' h₁i h₂ h₂' h₂o]
  refine tX_perm _ _ _ _ _ _ ?_ ?_
  · have := (ho.map τ).append_right (D₁ ++ D₃)
    refine List.Perm.trans ?_ this.symm
    simp only [List.map_append, List.append_assoc]
    exact List.Perm.append_left _ (by
      rw [← List.append_assoc, ← List.append_assoc]
      exact (List.perm_append_comm.append_right D₃).trans (by simp [List.perm_append_comm]))
  · have := (hi.map τ).append_right (D₂ ++ D₄)
    refine List.Perm.trans ?_ this.symm
    simp only [List.map_append, List.append_assoc]
    exact List.Perm.append_left _ (by
      rw [← List.append_assoc, ← List.append_assoc]
      exact (List.perm_append_comm.append_right D₄).trans (by simp [List.perm_append_comm]))

theorem fusionXG (E : Env) (co₁ ci₁ co₂ ci₂ : List (List ℕ)) (a b c dz w : ℕ)
    (o₁ to₁ i₁ ti₁ o₂ to₂ i₂ ti₂ o₃ i₃ iface : List ℕ)
    (l₁ : to₁.length = co₁.length) (l₂ : ti₁.length = ci₁.length)
    (l₃ : to₂.length = co₂.length) (l₄ : ti₂.length = ci₂.length)
    (hd : 0 < E.dimv dz) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₁i : w ∉ i₁)
    (h₂ : w ∈ i₂) (h₂' : w ∉ i₂.erase w) (h₂o : w ∉ o₂) (hwt : w ∉ to₁ ++ ti₁ ++ to₂ ++ ti₂)
    (ho : o₃.Perm (o₁.erase w ++ o₂)) (hi : i₃.Perm (i₁ ++ i₂.erase w))
    (hw : labIdx E.labs w = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase c k) = ex (E.phase a k) * ex (E.phase b k)) :
    LocalEq E [⟨xk co₁ ci₁, o₁ ++ to₁, i₁ ++ ti₁, dz, a⟩, ⟨xk co₂ ci₂, o₂ ++ to₂, i₂ ++ ti₂, dz, b⟩]
      [w] [⟨xk (co₁ ++ co₂) (ci₁ ++ ci₂), o₃ ++ (to₁ ++ to₂), i₃ ++ (ti₁ ++ ti₂), dz, c⟩] []
      iface 1 1 := by
  intro τ _
  simp only [List.mem_append, not_or] at hwt
  obtain ⟨⟨⟨w1, w2⟩, w3⟩, w4⟩ := hwt
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val_xk]
  rw [bundled_split _ _ _ _ _ _ (by simp [l₁, l₃]), tailVals_append_map _ _ _ _ _ _ _ l₁,
    bundled_split _ _ _ _ _ _ (by simp [l₂, l₄]), tailVals_append_map _ _ _ _ _ _ _ l₂]
  simp only [bundled_split _ _ _ _ _ _ l₁, bundled_split _ _ _ _ _ _ l₂,
    bundled_split _ _ _ _ _ _ l₃, bundled_split _ _ _ _ _ _ l₄,
    tailVals_absent _ _ τ w _ _ _ w1, tailVals_absent _ _ τ w _ _ _ w2,
    tailVals_absent _ _ τ w _ _ _ w3, tailVals_absent _ _ τ w _ _ _ w4]
  exact fusionX_gen E a b c dz w o₁ i₁ o₂ i₂ o₃ i₃ hd h₁ h₁' h₁i h₂ h₂' h₂o ho hi hw hph τ _ _ _ _

/-! ## Copied bare wires -/

theorem digits_eq (d k x w : ℕ) (hx : x < d ^ k) (hw : w < d ^ k)
    (h : ∀ i < k, digit d i x = digit d i w) : x = w := by
  have : ∀ j ≤ k, x % d ^ j = w % d ^ j := by
    intro j hj
    induction j with
    | zero => simp [Nat.mod_one]
    | succ j ih =>
      rw [Nat.mod_pow_succ, Nat.mod_pow_succ, ih (by omega)]
      have := h j (by omega)
      simp only [digit] at this
      rw [this]
  have := this k le_rfl
  rwa [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hw] at this

/-- Copies of copies of a bare wire along `ms`, outermost first. -/
def Kind.tower : List ℕ → Kind
  | [] => .I
  | m :: ms => .Bun m (Kind.tower ms)

theorem Kind.tower_cs (ms : List ℕ) : (Kind.tower ms).outCs 1 = [ms] ∧ (Kind.tower ms).inCs 1 = [ms] := by
  induction ms with
  | nil => simp [Kind.tower, Kind.outCs, Kind.inCs]
  | cons m ms ih => simp [Kind.tower, Kind.outCs, Kind.inCs, ih.1, ih.2]

theorem tensor_tower (E : Env) (d : ℕ) (hd : 0 < d) (ms : List ℕ) (vx vw : ℕ)
    (hx : vx < d ^ (ms.map E.mult).prod) (hw : vw < d ^ (ms.map E.mult).prod) :
    tensor E (Kind.tower ms) d 0 [vx] [vw] = tI (d ^ (ms.map E.mult).prod) vx vw := by
  induction ms generalizing vx vw with
  | nil => simp [Kind.tower, tensor]
  | cons m ms ih =>
    set D := d ^ (ms.map E.mult).prod
    have hD : 0 < D := pow_pos hd _
    have e : d ^ ((m :: ms).map E.mult).prod = D ^ E.mult m := by
      simp only [List.map_cons, List.prod_cons, D, ← pow_mul, mul_comm]
    rw [e] at hx hw ⊢
    simp only [Kind.tower, tensor, copyOuts, copyIns, List.length_singleton,
      (Kind.tower_cs ms).1, (Kind.tower_cs ms).2, copyLegs, List.zipWith_cons_cons,
      List.zipWith_nil_left]
    rw [List.map_congr_left fun i _ => ih _ _ (digit_lt _ _ _ hD) (digit_lt _ _ _ hD)]
    simp only [tI]
    by_cases hxw : vx = vw
    · rw [if_pos ⟨hxw, hxw ▸ hx⟩]
      refine List.prod_eq_one fun v hv => ?_
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 hv
      rw [if_pos ⟨by rw [hxw], digit_lt _ _ _ hD⟩]
    · rw [if_neg (fun h => hxw h.1)]
      obtain ⟨i, hi, hne⟩ : ∃ i < E.mult m, digit D i vx ≠ digit D i vw := by
        by_contra hc
        push_neg at hc
        exact hxw (digits_eq _ _ _ _ hx hw hc)
      refine List.prod_eq_zero (List.mem_map.2 ⟨i, List.mem_range.2 hi, ?_⟩)
      rw [if_neg (fun h => hne h.1)]

/-- Copies of a bare wire along `ms` are one bare wire on the bundled dimension. -/
theorem bunI (E : Env) (ms : List ℕ) (dz bz x w : ℕ) (iface : List ℕ)
    (hb : E.dimv bz = E.dimv dz ^ (ms.map E.mult).prod) (hd : 0 < E.dimv dz)
    (hx : labIdx E.labs x = bz) (hw : labIdx E.labs w = bz) (hxi : x ∈ iface) (hwi : w ∈ iface) :
    LocalEq E [⟨Kind.tower ms, [x], [w], dz, 0⟩] [] [⟨.I, [x], [w], bz, 0⟩] [] iface 1 1 := by
  intro τ hτ
  have rx := hτ x hxi
  have rw' := hτ w hwi
  rw [labDim_w E x bz hx, hb] at rx
  rw [labDim_w E w bz hw, hb] at rw'
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  rw [val_tensor E τ _ (by cases ms <;> simp [Kind.tower]), List.map_singleton,
    List.map_singleton, tensor_tower E _ hd ms _ _ rx rw']
  simp only [val, hb, List.map_cons, List.map_nil]

end ArchytasZX
