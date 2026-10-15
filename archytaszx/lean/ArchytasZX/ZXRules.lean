/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Closed

/-!
# ZX rule lemmas

Cap, Fourier colour change, state copy, triangle cancellation and Fourier cancellation.
-/

namespace ArchytasZX

open Complex Finset

theorem labDim_of (E : Env) (l dz : ℕ) (h : labIdx E.labs l = dz) : labDim E l = E.dimv dz := by
  simp only [labDim]; simp only [labIdx] at h; rw [h]

theorem rt_inv_mul (d : ℕ) (hd : 0 < d) : (rt d)⁻¹ * d = rt d := by
  rw [← rt_sq d]; field_simp [rt_ne d hd]

theorem sum_range_ite_mod (d : ℕ) (hd : 0 < d) (s : ℕ) (f : ℕ → ℂ) :
    ∑ j ∈ range d, (if s % d = j % d then f j else 0) = f (s % d) := by
  rw [sum_mod_eq]; simp [hd]

theorem sum_range_ite_mod' (d : ℕ) (hd : 0 < d) (s : ℕ) (f : ℕ → ℂ) :
    ∑ j ∈ range d, (if j % d = s % d then f j else 0) = f (s % d) := by
  rw [← sum_range_ite_mod d hd s f]
  exact Finset.sum_congr rfl fun j _ => by simp only [eq_comm]

theorem sum_mod0 (d : ℕ) (hd : 0 < d) (f : ℕ → ℂ) :
    ∑ j ∈ range d, (if j % d = 0 then f j else 0) = f 0 := by
  have : ∀ j ∈ range d, (if j % d = 0 then f j else 0) = if j = 0 then f j else 0 := by
    intro j hj; rw [Nat.mod_eq_of_lt (Finset.mem_range.1 hj)]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq']; simp [hd]

theorem sum_mod0' (d : ℕ) (hd : 0 < d) (f : ℕ → ℂ) :
    ∑ j ∈ range d, (if 0 = j % d then f j else 0) = f 0 := by
  rw [← sum_mod0 d hd f]; exact Finset.sum_congr rfl fun j _ => by simp only [eq_comm]

theorem prod_ite_zero (l : List ℕ) (τ : ℕ → ℕ) (c : ℂ) :
    (l.map fun o => if τ o = 0 then c else 0).prod =
      if ∀ o ∈ l, τ o = 0 then c ^ l.length else 0 := by
  induction l with
  | nil => simp
  | cons o l ih =>
    simp only [List.map_cons, List.prod_cons, ih, List.mem_cons, forall_eq_or_imp,
      List.length_cons, pow_succ]
    by_cases h : τ o = 0 <;> by_cases h' : ∀ x ∈ l, τ x = 0 <;> simp [h, h', mul_comm]

/-! ## Cap -/

theorem cap (E : Env) (a b dz w : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1)
    (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.Z, [w], [], dz, a⟩, ⟨.X, [], [w], dz, b⟩] [w] [] [] iface 1 (rt (E.dimv dz)) := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.append_nil, Function.update_self, labDim_of E w dz hw]
  rw [Finset.sum_congr rfl fun j hj => by
    rw [tZ_one _ _ ha, tX_flat _ hd _ hb, if_pos (Finset.mem_range.1 hj)]]
  simp only [List.sum_nil, List.sum_cons, List.length_nil, List.length_cons, add_zero, zero_add,
    one_mul, pow_one, Nat.zero_mod, mul_ite, mul_zero]
  rw [sum_mod0' _ hd fun _ => (rt (E.dimv dz))⁻¹ * (E.dimv dz)]
  exact rt_inv_mul _ hd

theorem capSwapped (E : Env) (a b dz w : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1)
    (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.Z, [], [w], dz, b⟩] [w] [] [] iface 1 (rt (E.dimv dz)) := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.nil_append, Function.update_self, labDim_of E w dz hw]
  rw [Finset.sum_congr rfl fun j hj => by
    rw [tZ_one _ _ hb, tX_flat _ hd _ ha, if_pos (Finset.mem_range.1 hj)]]
  simp only [List.sum_nil, List.sum_cons, List.length_nil, List.length_cons, add_zero, zero_add,
    pow_one, Nat.zero_mod, mul_ite, mul_zero, mul_one]
  rw [sum_mod0 _ hd fun _ => (rt (E.dimv dz))⁻¹ * (E.dimv dz)]
  exact rt_inv_mul _ hd

/-! ## Fourier colour change -/

theorem char_sum0 (d : ℕ) (hd : 0 < d) (a : ℕ) :
    ∑ k ∈ range d, om d ^ (k * a) = if a % d = 0 then (d : ℂ) else 0 := by
  have h := char_sum d hd a 0
  simp only [mul_zero, pow_zero, mul_one] at h
  rw [h]; by_cases hc : a % d = 0 <;> simp [hc]

theorem sum_fou_left (d : ℕ) (hd : 0 < d) (x : ℕ) :
    ∑ j ∈ range d, fou d x j = (rt d)⁻¹ * (if x % d = 0 then (d : ℂ) else 0) := by
  rw [← char_sum0 d hd x, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by simp only [fou, rt, mul_comm x j]; ring

theorem sum_fou_right (d : ℕ) (hd : 0 < d) (x : ℕ) :
    ∑ j ∈ range d, fou d j x = (rt d)⁻¹ * (if x % d = 0 then (d : ℂ) else 0) := by
  rw [← sum_fou_left d hd x]; exact Finset.sum_congr rfl fun j _ => fou_symm d j x

theorem fou_zero (d x : ℕ) : fou d x 0 = (rt d)⁻¹ := by simp [fou, rt]

theorem fou_zero' (d x : ℕ) : fou d 0 x = (rt d)⁻¹ := by simp [fou, rt]

theorem colorState (E : Env) (a b dz w x : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwx : w ≠ x) (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1)
    (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.Z, [w], [], dz, a⟩, ⟨.F, [x], [w], dz, 0⟩] [w] [⟨.X, [x], [], dz, b⟩] [] iface
      1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.append_nil, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwx.symm]
  rw [Finset.sum_congr rfl fun j hj => by rw [tZ_one _ _ ha, if_pos (Finset.mem_range.1 hj),
    one_mul], sum_fou_left _ hd, tX_flat _ hd _ hb]
  simp

theorem colorEffect (E : Env) (a b dz w x : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwx : w ≠ x) (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1)
    (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.Z, [], [w], dz, a⟩, ⟨.F, [w], [x], dz, 0⟩] [w] [⟨.X, [], [x], dz, b⟩] [] iface
      1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.nil_append, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwx.symm]
  rw [Finset.sum_congr rfl fun j hj => by rw [tZ_one _ _ ha, if_pos (Finset.mem_range.1 hj),
    one_mul], sum_fou_right _ hd, tX_flat _ hd _ hb]
  simp [eq_comm]

theorem colorStateX (E : Env) (a b dz w x : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwx : w ≠ x) (hx : x ∈ iface) (hxd : labIdx E.labs x = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.F, [x], [w], dz, 0⟩] [w] [⟨.Z, [x], [], dz, b⟩] [] iface
      1 1 := by
  intro τ hτ
  have rx := hτ x hx
  rw [labDim_of E x dz hxd] at rx
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.append_nil, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwx.symm]
  rw [Finset.sum_congr rfl fun j _ => by rw [tX_flat _ hd _ ha], tZ_one _ _ hb, if_pos rx]
  simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
    Nat.zero_mod, ite_mul, zero_mul, mul_ite, mul_zero]
  rw [sum_mod0 _ hd fun j => (rt (E.dimv dz))⁻¹ ^ (0 + 1) * (E.dimv dz) * fou (E.dimv dz) (τ x) j,
    fou_zero, zero_add, pow_one, rt_inv_mul _ hd, mul_inv_cancel₀ (rt_ne _ hd)]

theorem colorEffectX (E : Env) (a b dz w x : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwx : w ≠ x) (hx : x ∈ iface) (hxd : labIdx E.labs x = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.X, [], [w], dz, a⟩, ⟨.F, [w], [x], dz, 0⟩] [w] [⟨.Z, [], [x], dz, b⟩] [] iface
      1 1 := by
  intro τ hτ
  have rx := hτ x hx
  rw [labDim_of E x dz hxd] at rx
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.nil_append, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwx.symm]
  rw [Finset.sum_congr rfl fun j _ => by rw [tX_flat _ hd _ ha], tZ_one _ _ hb, if_pos rx]
  simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
    Nat.zero_mod, ite_mul, zero_mul, mul_ite, mul_zero, zero_add]
  rw [sum_mod0' _ hd fun j => (rt (E.dimv dz))⁻¹ ^ 1 * (E.dimv dz) * fou (E.dimv dz) j (τ x),
    fou_zero', pow_one, rt_inv_mul _ hd, mul_inv_cancel₀ (rt_ne _ hd)]

/-! ## State copy -/

theorem stateCopy (E : Env) (a b c dz w : ℕ) (os iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwo : w ∉ os)
    (hos : ∀ o ∈ os, o ∈ iface ∧ labIdx E.labs o = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1)
    (hc : ∀ k < E.dimv dz, ex (E.phase c k) = 1)
    (hk : k * rt (E.dimv dz) ^ os.length = rt (E.dimv dz)) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.Z, os, [w], dz, b⟩] [w]
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
  have step : ∀ j ∈ range d, tX d (E.phase a) [j] [] * tZ d (E.phase b) (os.map τ ++ [j]) =
      if j % d = 0 then (rt d)⁻¹ * d * tZ d (E.phase b) (os.map τ ++ [j]) else 0 := by
    intro j _
    rw [tX_flat _ hd _ ha]
    by_cases h : j % d = 0 <;> simp [h]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hd fun j => (rt d)⁻¹ * d * tZ d (E.phase b)
    (os.map τ ++ [j]), rt_inv_mul _ hd, tZ_at _ _ _ 0 (by simp), hb 0 hd]
  have hiff : (0 < d ∧ ∀ v ∈ os.map τ ++ [0], v = 0) ↔ ∀ o ∈ os, τ o = 0 := by
    constructor
    · rintro ⟨-, h⟩ o ho
      exact h _ (List.mem_append_left _ (List.mem_map_of_mem ho))
    · intro h
      refine ⟨hd, fun v hv => ?_⟩
      rcases List.mem_append.1 hv with hv | hv
      · obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hv
        exact h o ho
      · simpa using hv
  by_cases h : ∀ o ∈ os, τ o = 0
  · rw [if_pos (hiff.2 h), if_pos h, mul_one, hk]
  · rw [if_neg (fun h' => h (hiff.1 h')), if_neg h]; simp

theorem stateCopySwapped (E : Env) (a b c dz w : ℕ) (os iface : List ℕ) (k : ℂ)
    (hd : 0 < E.dimv dz) (hw : labIdx E.labs w = dz) (hwo : w ∉ os)
    (hos : ∀ o ∈ os, o ∈ iface ∧ labIdx E.labs o = dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1)
    (hc : ∀ k < E.dimv dz, ex (E.phase c k) = 1)
    (hk : k * rt (E.dimv dz) ^ os.length = rt (E.dimv dz)) :
    LocalEq E [⟨.Z, [w], [], dz, a⟩, ⟨.X, os, [w], dz, b⟩] [w]
      (os.map fun o => ⟨.Z, [o], [], dz, c⟩) [] iface 1 k := by
  intro τ hτ
  have hr : ∀ o ∈ os, τ o < E.dimv dz := fun o ho => by
    have := hτ o (hos o ho).1; rwa [labDim_of E o dz (hos o ho).2] at this
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.append_nil, Function.update_self, labDim_of E w dz hw,
    map_update_absent τ w _ os hwo, List.map_map]
  have hz : ∀ o ∈ os, (Function.comp (val E τ) fun o => (⟨.Z, [o], [], dz, c⟩ : Node)) o = 1 := by
    intro o ho
    simp only [Function.comp, val, List.map_cons, List.map_nil, List.append_nil]
    rw [tZ_one _ _ hc, if_pos (hr o ho)]
  have hp : (os.map (val E τ ∘ fun o => (⟨.Z, [o], [], dz, c⟩ : Node))).prod = 1 :=
    List.prod_eq_one fun x hx => by
      obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hx
      exact hz o ho
  rw [hp, mul_one]
  generalize E.dimv dz = d at *
  have step : ∀ j ∈ range d, tZ d (E.phase a) [j] * tX d (E.phase b) (os.map τ) [j] =
      if (os.map τ).sum % d = j % d then (rt d)⁻¹ ^ (os.length + 1) * d else 0 := by
    intro j hj
    rw [tZ_one _ _ ha, if_pos (Finset.mem_range.1 hj), one_mul, tX_flat _ hd _ hb]
    simp [List.length_map]
  rw [Finset.sum_congr rfl step, sum_range_ite_mod _ hd _ fun _ => (rt d)⁻¹ ^ (os.length + 1) * d]
  have hrt := rt_ne d hd
  have h2 := rt_sq d
  have hk' : k = rt d * ((rt d)⁻¹) ^ os.length := by
    calc k = k * rt d ^ os.length * (rt d)⁻¹ ^ os.length := by
          rw [mul_assoc, ← mul_pow, mul_inv_cancel₀ hrt, one_pow, mul_one]
      _ = rt d * (rt d)⁻¹ ^ os.length := by rw [hk]
  rw [hk', ← h2, pow_succ]
  field_simp

/-! ## Triangle cancellation -/

theorem tri_sum (d o i : ℕ) (ho : o < d) (hi : i < d) :
    ∑ j ∈ range d, tT j i * tTi o j = if o = i then 1 else 0 := by
  have hT : ∀ j, tT j i = (if j = 0 then 1 else 0) + (if j = i then (if i = 0 then 0 else 1)
      else 0) := by
    intro j; unfold tT
    split_ifs <;> first | (exfalso; omega) | (simp; done) | (simp_all; done)
  simp only [hT, add_mul, Finset.sum_add_distrib, ite_mul, one_mul, zero_mul]
  rw [Finset.sum_ite_eq', Finset.sum_ite_eq']
  simp only [Finset.mem_range, show 0 < d by omega, hi, if_true]
  unfold tTi
  rcases Nat.eq_zero_or_pos i with rfl | hi0
  · by_cases ho0 : o = 0 <;> simp [ho0]
  · have hi0' : i ≠ 0 := by omega
    by_cases ho0 : o = 0
    · subst ho0; simp [hi0', Ne.symm hi0', show 1 ≤ i by omega]
    · by_cases hoi : o = i
      · subst hoi; simp [ho0]
      · simp [ho0, hoi]

theorem tri_sum' (d o i : ℕ) (ho : o < d) (hi : i < d) :
    ∑ j ∈ range d, tTi j i * tT o j = if o = i then 1 else 0 := by
  have hT : ∀ j, tTi j i = (if j = i then 1 else 0) - (if j = 0 then (if 1 ≤ i then 1 else 0)
      else 0) := by
    intro j; unfold tTi
    split_ifs <;> first | (exfalso; omega) | (simp; done) | (simp_all; done)
  simp only [hT, sub_mul, Finset.sum_sub_distrib, ite_mul, one_mul, zero_mul]
  rw [Finset.sum_ite_eq', Finset.sum_ite_eq']
  simp only [Finset.mem_range, show 0 < d by omega, hi, if_true]
  unfold tT
  rcases Nat.eq_zero_or_pos i with rfl | hi0
  · by_cases ho0 : o = 0 <;> simp [ho0]
  · have hi0' : i ≠ 0 := by omega
    have hi1 : 1 ≤ i := by omega
    by_cases ho0 : o = 0
    · subst ho0; simp [hi0', Ne.symm hi0', hi1]
    · by_cases hoi : o = i
      · subst hoi; simp [ho0, hi1]
      · simp [ho0, hoi, hi1]

theorem triangle (E : Env) (dz w o i x y : ℕ) (iface : List ℕ)
    (hxy : (x = o ∧ y = i) ∨ (x = i ∧ y = o)) (hw : labIdx E.labs w = dz) (hwo : w ≠ o)
    (hwi : w ≠ i) (ho : o ∈ iface ∧ labIdx E.labs o = dz) (hi : i ∈ iface ∧ labIdx E.labs i = dz) :
    LocalEq E [⟨.T, [w], [i], dz, 0⟩, ⟨.Ti, [o], [w], dz, 0⟩] [w] [⟨.I, [x], [y], dz, 0⟩] []
      iface 1 1 := by
  intro τ hτ
  have ro := hτ o ho.1
  have ri := hτ i hi.1
  rw [labDim_of E o dz ho.2] at ro
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwo.symm, Function.update_of_ne hwi.symm]
  rw [tri_sum _ _ _ ro ri]
  simp only [tI]
  rcases hxy with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> split_ifs <;> first | rfl | (exfalso; omega)

theorem triangle' (E : Env) (dz w o i x y : ℕ) (iface : List ℕ)
    (hxy : (x = o ∧ y = i) ∨ (x = i ∧ y = o)) (hw : labIdx E.labs w = dz) (hwo : w ≠ o)
    (hwi : w ≠ i) (ho : o ∈ iface ∧ labIdx E.labs o = dz) (hi : i ∈ iface ∧ labIdx E.labs i = dz) :
    LocalEq E [⟨.Ti, [w], [i], dz, 0⟩, ⟨.T, [o], [w], dz, 0⟩] [w] [⟨.I, [x], [y], dz, 0⟩] []
      iface 1 1 := by
  intro τ hτ
  have ro := hτ o ho.1
  have ri := hτ i hi.1
  rw [labDim_of E o dz ho.2] at ro
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    Function.update_of_ne hwo.symm, Function.update_of_ne hwi.symm]
  rw [tri_sum' _ _ _ ro ri]
  simp only [tI]
  rcases hxy with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> split_ifs <;> first | rfl | (exfalso; omega)

/-! ## Fourier cancellation -/

theorem antipode_iff (d a j : ℕ) (ha : a < d) (hj : j < d) :
    (j + a) % d = 0 ↔ j = (d - a) % d := by
  rcases Nat.eq_zero_or_pos a with rfl | ha0
  · simp [Nat.mod_eq_of_lt hj]
  · have h1 : (d - a) % d = d - a := Nat.mod_eq_of_lt (by omega)
    rw [h1]
    by_cases hlt : j + a < d
    · rw [Nat.mod_eq_of_lt hlt]; omega
    · have : (j + a) % d = j + a - d := by
        rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
      rw [this]; omega

theorem antipode_inj (d a b : ℕ) (ha : a < d) (hb : b < d) :
    (d - a) % d = (d - b) % d ↔ a = b := by
  constructor
  · intro h
    rcases Nat.eq_zero_or_pos a with rfl | ha0 <;> rcases Nat.eq_zero_or_pos b with rfl | hb0
    · rfl
    · rw [Nat.sub_zero, Nat.mod_self, Nat.mod_eq_of_lt (by omega)] at h; omega
    · rw [Nat.sub_zero, Nat.mod_self, Nat.mod_eq_of_lt (by omega)] at h; omega
    · rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)] at h; omega
  · rintro rfl; rfl

theorem antipode_sq (d a b : ℕ) (ha : a < d) (hb : b < d) :
    ∑ j ∈ range d, (if (j + a) % d = 0 then (1 : ℂ) else 0) *
      (if (b + j) % d = 0 then 1 else 0) = if a = b then 1 else 0 := by
  have : ∀ j ∈ range d, (if (j + a) % d = 0 then (1 : ℂ) else 0) *
      (if (b + j) % d = 0 then 1 else 0) =
      if j = (d - a) % d then (if (j + b) % d = 0 then 1 else 0) else 0 := by
    intro j hj
    simp only [antipode_iff d a j ha (Finset.mem_range.1 hj), add_comm b j]
    split_ifs <;> simp
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq']
  have hlt : (d - a) % d < d := Nat.mod_lt _ (by omega)
  rw [if_pos (Finset.mem_range.2 hlt)]
  simp only [antipode_iff d b _ hb hlt, antipode_inj d a b ha hb]

theorem fourier4 (E : Env) (a b c dz w₁ w₂ w₃ : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (h₁ : labIdx E.labs w₁ = dz) (h₂ : labIdx E.labs w₂ = dz) (h₃ : labIdx E.labs w₃ = dz)
    (h12 : w₁ ≠ w₂) (h13 : w₁ ≠ w₃) (h23 : w₂ ≠ w₃) (ha1 : a ≠ w₁) (ha2 : a ≠ w₂)
    (ha3 : a ≠ w₃) (hb1 : b ≠ w₁) (hb2 : b ≠ w₂) (hb3 : b ≠ w₃)
    (ha : a ∈ iface ∧ labIdx E.labs a = dz) (hb : b ∈ iface ∧ labIdx E.labs b = dz)
    (hc : ∀ k < E.dimv dz, ex (E.phase c k) = 1) :
    LocalEq E [⟨.F, [w₁], [a], dz, 0⟩, ⟨.F, [w₂], [w₁], dz, 0⟩, ⟨.F, [w₃], [w₂], dz, 0⟩,
      ⟨.F, [b], [w₃], dz, 0⟩] [w₁, w₂, w₃] [⟨.Z, [b], [a], dz, c⟩] [] iface 1 1 := by
  intro τ hτ
  have ra := hτ a ha.1
  have rb := hτ b hb.1
  rw [labDim_of E a dz ha.2] at ra
  rw [labDim_of E b dz hb.2] at rb
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E _ dz h₁, labDim_of E _ dz h₂,
    labDim_of E _ dz h₃, Function.update_apply, ha1, ha2, ha3, hb1, hb2, hb3, h12, h13, h23,
    h12.symm, h13.symm, h23.symm, if_true, if_false, List.cons_append, List.nil_append]
  generalize E.dimv dz = d at *
  rw [tZ_at _ _ _ (τ b) (by simp), hc _ rb]
  have inner : ∀ j₁ ∈ range d, ∀ j₂ ∈ range d, ∑ j₃ ∈ range d,
      fou d j₁ (τ a) * (fou d j₂ j₁ * (fou d j₃ j₂ * fou d (τ b) j₃)) =
      (fou d j₂ j₁ * fou d j₁ (τ a)) * ∑ j₃ ∈ range d, fou d (τ b) j₃ * fou d j₃ j₂ := by
    intro j₁ _ j₂ _
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j₃ _ => by ring
  rw [Finset.sum_congr rfl fun j₁ hj₁ => Finset.sum_congr rfl fun j₂ hj₂ => inner j₁ hj₁ j₂ hj₂,
    Finset.sum_comm]
  simp only [← Finset.sum_mul]
  rw [Finset.sum_congr rfl fun j₂ _ => by rw [fou_sq d hd, fou_sq d hd]]
  rw [antipode_sq d _ _ ra rb]
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, true_and]
  by_cases h : τ a = τ b
  · simp [h, rb]
  · simp [h, Ne.symm h]

end ArchytasZX
