/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Hopf

/-!
# W and triangle rule lemmas
-/

namespace ArchytasZX

open Complex Finset

/-- The nonzero leg values. -/
abbrev nz (l : List ℕ) : List ℕ := l.filter (· ≠ 0)

theorem nz_nil_iff (l : List ℕ) : nz l = [] ↔ ∀ o ∈ l, o = 0 := by
  simp [nz, List.filter_eq_nil_iff]

theorem tW_eq (d : ℕ) (os : List ℕ) (i : ℕ) :
    tW d os i = if i = 0 then (if nz os = [] then 1 else 0)
      else (if i < d ∧ nz os = [i] then 1 else 0) := by
  unfold tW
  by_cases h : i = 0
  · simp only [h, if_true]; simp only [nz_nil_iff]
  · simp only [h, if_false]

theorem tW_perm (d : ℕ) (os os' : List ℕ) (i : ℕ) (h : os.Perm os') : tW d os i = tW d os' i := by
  rw [tW_eq, tW_eq]
  have hp : (nz os).Perm (nz os') := h.filter _
  have e1 : nz os = [] ↔ nz os' = [] := by rw [nz_nil_iff, nz_nil_iff]; exact
    ⟨fun h' o ho => h' o (h.mem_iff.2 ho), fun h' o ho => h' o (h.mem_iff.1 ho)⟩
  have e2 : nz os = [i] ↔ nz os' = [i] :=
    ⟨fun h' => List.perm_singleton.1 (h' ▸ hp.symm), fun h' => List.perm_singleton.1 (h' ▸ hp)⟩
  simp only [e1, e2]

theorem nz_cons (j : ℕ) (l : List ℕ) : nz (j :: l) = if j = 0 then nz l else j :: nz l := by
  by_cases h : j = 0 <;> simp [nz, List.filter_cons, h]

theorem tW_cons_zero (d : ℕ) (os : List ℕ) (i : ℕ) : tW d (0 :: os) i = tW d os i := by
  rw [tW_eq, tW_eq, nz_cons]; simp

/-- The W node with one output value `j` split off. -/
theorem tW_cons (d : ℕ) (A : List ℕ) (j I : ℕ) (hI : I ≠ 0) :
    tW d (j :: A) I = (if j = 0 then tW d A I else 0) +
      (if j = I then (if nz A = [] ∧ I < d then 1 else 0) else 0) := by
  rw [tW_eq, tW_eq, nz_cons]
  by_cases hj : j = 0
  · subst hj; simp [hI, Ne.symm hI]
  · by_cases hjI : j = I
    · subst hjI; simp [hj]; by_cases h : nz A = [] <;> simp [h, and_comm]
    · simp [hj, hjI, hI]

theorem tW_zero_in (d : ℕ) (A : List ℕ) (j : ℕ) :
    tW d (j :: A) 0 = if j = 0 then tW d A 0 else 0 := by
  rw [tW_eq, tW_eq, nz_cons]; by_cases hj : j = 0 <;> simp [hj]

theorem sum_tW_cons (d : ℕ) (hd : 0 < d) (A : List ℕ) (I : ℕ) (g : ℕ → ℂ) (hI : I < d) :
    ∑ j ∈ range d, tW d (j :: A) I * g j =
      tW d A I * g 0 + (if I = 0 then 0 else (if nz A = [] then 1 else 0) * g I) := by
  by_cases h0 : I = 0
  · subst h0
    simp only [tW_zero_in, ite_mul, zero_mul, if_true]
    rw [Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hd)]; simp
  · simp only [tW_cons d A _ I h0, add_mul, ite_mul, zero_mul, Finset.sum_add_distrib]
    rw [Finset.sum_ite_eq', Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hd),
      if_pos (Finset.mem_range.2 hI)]
    simp [h0, hI]

theorem tW_append (d : ℕ) (A B : List ℕ) (I : ℕ) (hI : I < d) :
    tW d (A ++ B) I = tW d A I * tW d B 0 +
      (if I = 0 then 0 else (if nz A = [] then 1 else 0) * tW d B I) := by
  have hn : nz (A ++ B) = nz A ++ nz B := by simp [nz, List.filter_append]
  simp only [tW_eq, hn, if_true]
  by_cases h0 : I = 0
  · simp only [h0, if_true, List.append_eq_nil_iff]
    by_cases ha : nz A = [] <;> by_cases hb : nz B = [] <;> simp [ha, hb]
  · simp only [h0, if_false, hI, true_and, List.append_eq_singleton_iff]
    by_cases ha : nz A = [] <;> by_cases hb : nz B = [] <;> simp [ha, hb] <;>
      by_cases ha' : nz A = [I] <;> by_cases hb' : nz B = [I] <;> simp_all

theorem wFusion (E : Env) (dz w i : ℕ) (o₁ o₂ o₃ iface : List ℕ)
    (hw : labIdx E.labs w = dz) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₂ : w ∉ o₂)
    (hwi : w ≠ i) (hi : i ∈ iface ∧ labIdx E.labs i = dz) (hd : 0 < E.dimv dz)
    (ho : o₃.Perm (o₁.erase w ++ o₂)) :
    LocalEq E [⟨.W, o₁, [i], dz, 0⟩, ⟨.W, o₂, [w], dz, 0⟩] [w] [⟨.W, o₃, [i], dz, 0⟩] [] iface
      1 1 := by
  intro τ hτ
  have ri := hτ i hi.1
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E w dz hw, Function.update_self,
    Function.update_of_ne hwi.symm, map_update_absent τ w _ o₂ h₂]
  have hp : ∀ j, o₁.map (Function.update τ w j) |>.Perm (j :: (o₁.erase w).map τ) := by
    intro j
    have := (List.perm_cons_erase h₁).map (Function.update τ w j)
    rw [List.map_cons, Function.update_self, map_update_absent τ w j _ h₁'] at this
    exact this
  rw [Finset.sum_congr rfl fun j _ => by rw [tW_perm _ _ _ _ (hp j)],
    sum_tW_cons _ hd _ _ _ ri, tW_perm _ _ _ _ (ho.map τ), List.map_append, tW_append _ _ _ _ ri]

theorem wIdentity (E : Env) (dz p q w x : ℕ) (iface : List ℕ)
    (hpq : (p = w ∧ q = x) ∨ (p = x ∧ q = w)) (hface : p ∈ iface ∧ q ∈ iface)
    (hp : labIdx E.labs p = dz) (hq : labIdx E.labs q = dz) :
    LocalEq E [⟨.W, [p], [q], dz, 0⟩] [] [⟨.I, [x], [w], dz, 0⟩] [] iface 1 1 := by
  intro τ hτ
  have rp := hτ p hface.1
  have rq := hτ q hface.2
  rw [labDim_of E p dz hp] at rp
  rw [labDim_of E q dz hq] at rq
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, tW_eq, nz, tI]
  rcases hpq with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
  · by_cases h0 : τ q = 0 <;> by_cases hp0 : τ p = 0 <;> by_cases he : τ p = τ q <;>
      simp [List.filter_cons, h0, hp0, he, rp, rq] <;> omega

theorem wZeroCopy (E : Env) (a c dz w : ℕ) (os iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwo : w ∉ os)
    (hos : ∀ o ∈ os, o ∈ iface ∧ labIdx E.labs o = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (hc : ∀ k < E.dimv dz, ex (E.phase c k) = 1)
    (hk : k * rt (E.dimv dz) ^ os.length = rt (E.dimv dz)) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.W, os, [w], dz, 0⟩] [w]
      (os.map fun o => ⟨.X, [o], [], dz, c⟩) [] iface 1 k := by
  intro τ hτ
  have hr : ∀ o ∈ os, τ o < E.dimv dz := fun o ho => by
    have := hτ o (hos o ho).1; rwa [labDim_of E o dz (hos o ho).2] at this
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    map_update_absent τ w _ os hwo, List.map_map]
  have hx : ∀ o ∈ os, (Function.comp (val E τ) fun o => (⟨.X, [o], [], dz, c⟩ : Node)) o =
      if τ o = 0 then rt (E.dimv dz) else 0 := by
    intro o ho
    simp only [Function.comp, val, List.map_cons, List.map_nil]
    rw [tX_flat _ hd _ hc]
    simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
      zero_add, pow_one, Nat.zero_mod, Nat.mod_eq_of_lt (hr o ho), mul_ite, mul_zero,
      rt_inv_mul _ hd]
  rw [List.map_congr_left hx, prod_ite_zero]
  generalize E.dimv dz = d at *
  have step : ∀ j ∈ range d, tX d (E.phase a) [j] [] * tW d (os.map τ) j =
      if j % d = 0 then (rt d)⁻¹ * d * tW d (os.map τ) j else 0 := by
    intro j _
    rw [tX_flat _ hd _ ha]
    by_cases h : j % d = 0 <;> simp [h]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hd fun j => (rt d)⁻¹ * d * tW d (os.map τ) j,
    rt_inv_mul _ hd, tW_eq]
  simp only [if_true, nz_nil_iff, List.mem_map, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂]
  by_cases h : ∀ o ∈ os, τ o = 0
  · rw [if_pos h, if_pos h, mul_one, hk]
  · rw [if_neg h, if_neg h]; simp

theorem triZeroState (E : Env) (a dz w o : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwo : w ≠ o) (ho : o ∈ iface ∧ labIdx E.labs o = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.T, [o], [w], dz, 0⟩] [w] [⟨.X, [o], [], dz, a⟩] []
      iface 1 1 := by
  intro τ hτ
  have ro := hτ o ho.1
  rw [labDim_of E o dz ho.2] at ro
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwo.symm]
  generalize E.dimv dz = d at *
  have step : ∀ j ∈ range d, tX d (E.phase a) [j] [] * tT (τ o) j =
      if j % d = 0 then (rt d)⁻¹ * d * tT (τ o) j else 0 := by
    intro j _
    rw [tX_flat _ hd _ ha]
    by_cases h : j % d = 0 <;> simp [h]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hd fun j => (rt d)⁻¹ * d * tT (τ o) j,
    tX_flat _ hd _ ha]
  simp [tT, Nat.mod_eq_of_lt ro]

theorem triInvZeroState (E : Env) (a dz w o : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwo : w ≠ o) (ho : o ∈ iface ∧ labIdx E.labs o = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.Ti, [o], [w], dz, 0⟩] [w] [⟨.X, [o], [], dz, a⟩] []
      iface 1 1 := by
  intro τ hτ
  have ro := hτ o ho.1
  rw [labDim_of E o dz ho.2] at ro
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwo.symm]
  generalize E.dimv dz = d at *
  have step : ∀ j ∈ range d, tX d (E.phase a) [j] [] * tTi (τ o) j =
      if j % d = 0 then (rt d)⁻¹ * d * tTi (τ o) j else 0 := by
    intro j _
    rw [tX_flat _ hd _ ha]
    by_cases h : j % d = 0 <;> simp [h]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hd fun j => (rt d)⁻¹ * d * tTi (τ o) j,
    tX_flat _ hd _ ha]
  simp [tTi, Nat.mod_eq_of_lt ro]

theorem triZeroEffect (E : Env) (a c dz w i : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwi : w ≠ i) (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (hc : ∀ k < E.dimv dz, ex (E.phase c k) = 1) :
    LocalEq E [⟨.T, [w], [i], dz, 0⟩, ⟨.X, [], [w], dz, a⟩] [w] [⟨.Z, [], [i], dz, c⟩] []
      iface 1 (rt (E.dimv dz)) := by
  intro τ hτ
  have ri := hτ i hi.1
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwi.symm, List.nil_append]
  generalize E.dimv dz = d at *
  have step : ∀ j ∈ range d, tT j (τ i) * tX d (E.phase a) [] [j] =
      if j % d = 0 then (rt d)⁻¹ * d * tT j (τ i) else 0 := by
    intro j _
    rw [tX_flat _ hd _ ha]
    by_cases h : j % d = 0 <;> simp [h, eq_comm, mul_comm]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hd fun j => (rt d)⁻¹ * d * tT j (τ i),
    tZ_one _ _ hc, rt_inv_mul _ hd]
  simp [tT, ri]

theorem wZEffect (E : Env) (a dz w i v : ℕ) (os os' iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hv : labIdx E.labs v = dz) (h₁ : w ∈ os)
    (h₁' : w ∉ os.erase w) (hwi : w ≠ i) (hvi : v ≠ i) (hvo : v ∉ os')
    (hi : i ∈ iface ∧ labIdx E.labs i = dz) (ho : os'.Perm (os.erase w))
    (hrest : w ∉ os') (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) :
    LocalEq E [⟨.W, os, [i], dz, 0⟩, ⟨.Z, [], [w], dz, a⟩] [w]
      [⟨.T, [v], [i], dz, 0⟩, ⟨.W, os', [v], dz, 0⟩] [v] iface 1 1 := by
  intro τ hτ
  have ri := hτ i hi.1
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E w dz hw, labDim_of E v dz hv, Function.update_self,
    Function.update_of_ne hwi.symm, Function.update_of_ne hvi.symm, List.nil_append,
    map_update_absent τ v _ os' hvo]
  have hp : ∀ j, os.map (Function.update τ w j) |>.Perm (j :: (os.erase w).map τ) := by
    intro j
    have := (List.perm_cons_erase h₁).map (Function.update τ w j)
    rw [List.map_cons, Function.update_self, map_update_absent τ w j _ h₁'] at this
    exact this
  rw [Finset.sum_congr rfl fun j hj => by rw [tW_perm _ _ _ _ (hp j), tZ_one _ _ ha,
    if_pos (Finset.mem_range.1 hj)]]
  have hR : ∀ j, tW (E.dimv dz) (os'.map τ) j = tW (E.dimv dz) ((os.erase w).map τ) j :=
    fun j => tW_perm _ _ _ _ (ho.map τ)
  simp only [hR]
  generalize E.dimv dz = d at *
  set A := (os.erase w).map τ
  have e1 := sum_tW_cons d hd A (τ i) (fun _ => 1) ri
  simp only [mul_one] at e1 ⊢
  rw [e1]
  have e2 : ∀ j ∈ range d, tT j (τ i) * tW d A j =
      (if j = 0 then tW d A 0 else 0) + (if j = τ i then (if τ i = 0 then 0 else tW d A (τ i))
        else 0) := by
    intro j _
    unfold tT
    by_cases h0 : j = 0 <;> by_cases hj : j = τ i <;> by_cases hi0 : τ i = 0 <;> simp_all
  rw [Finset.sum_congr rfl e2, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.sum_ite_eq',
    if_pos (Finset.mem_range.2 hd), if_pos (Finset.mem_range.2 ri)]
  by_cases hi0 : τ i = 0
  · simp [hi0]
  · simp only [hi0, if_false, add_zero]
    rw [tW_eq d A 0]
    simp only [if_true]
    rw [tW_eq _ A (τ i), if_neg hi0]
    by_cases hA : nz A = [] <;> by_cases hA' : nz A = [τ i] <;> simp_all [add_comm]

theorem wZeroEffect (E : Env) (a dz w i : ℕ) (os os' iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (h₁ : w ∈ os) (h₁' : w ∉ os.erase w) (hwi : w ≠ i)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz) (ho : os'.Perm (os.erase w))
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) :
    LocalEq E [⟨.W, os, [i], dz, 0⟩, ⟨.X, [], [w], dz, a⟩] [w] [⟨.W, os', [i], dz, 0⟩] []
      iface 1 (rt (E.dimv dz)) := by
  intro τ hτ
  have ri := hτ i hi.1
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E w dz hw, Function.update_self,
    Function.update_of_ne hwi.symm, List.nil_append]
  have hp : ∀ j, os.map (Function.update τ w j) |>.Perm (j :: (os.erase w).map τ) := by
    intro j
    have := (List.perm_cons_erase h₁).map (Function.update τ w j)
    rw [List.map_cons, Function.update_self, map_update_absent τ w j _ h₁'] at this
    exact this
  rw [Finset.sum_congr rfl fun j _ => by rw [tW_perm _ _ _ _ (hp j)], tW_perm _ _ _ _ (ho.map τ)]
  generalize E.dimv dz = d at *
  have step : ∀ j ∈ range d, tW d (j :: (os.erase w).map τ) (τ i) * tX d (E.phase a) [] [j] =
      if j % d = 0 then (rt d)⁻¹ * d * tW d (j :: (os.erase w).map τ) (τ i) else 0 := by
    intro j _
    rw [tX_flat _ hd _ ha]
    by_cases h : j % d = 0 <;> simp [h, eq_comm, mul_comm]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hd fun j => (rt d)⁻¹ * d *
    tW d (j :: (os.erase w).map τ) (τ i), rt_inv_mul _ hd, tW_cons_zero]

end ArchytasZX
