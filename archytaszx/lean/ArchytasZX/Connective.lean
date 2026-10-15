/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.WRules

/-!
# Dimension binder and splitter lemmas
-/

namespace ArchytasZX

open Complex Finset

theorem pair_iff (t a b K : ℕ) (hb : b < t) : K = a * t + b ↔ K / t = a ∧ K % t = b := by
  constructor
  · rintro rfl
    have ht : 0 < t := by omega
    refine ⟨?_, ?_⟩
    · rw [Nat.add_comm, Nat.add_mul_div_right _ _ ht, Nat.div_eq_of_lt hb, zero_add]
    · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  · rintro ⟨rfl, rfl⟩
    rw [mul_comm]; exact (Nat.div_add_mod K t).symm

theorem tB_sum (s t K : ℕ) (ht : 0 < t) :
    ∑ a ∈ range s, ∑ b ∈ range t, tB s t K a b = if K < s * t then 1 else 0 := by
  have inner : ∀ a ∈ range s, ∑ b ∈ range t, tB s t K a b = if K / t = a then 1 else 0 := by
    intro a ha
    have : ∀ b ∈ range t, tB s t K a b = if b = K % t then (if K / t = a then 1 else 0) else 0 := by
      intro b hb
      have hb' := Finset.mem_range.1 hb
      unfold tB
      rw [if_congr (show (a < s ∧ b < t ∧ K = a * t + b) ↔ (K / t = a ∧ K % t = b) by
        rw [pair_iff t a b K hb']; exact ⟨fun h => h.2.2, fun h => ⟨Finset.mem_range.1 ha, hb', h⟩⟩)
        rfl rfl]
      by_cases h1 : K / t = a <;> by_cases h2 : b = K % t <;> simp [h1, h2, eq_comm]
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ ht))]
  rw [Finset.sum_congr rfl inner, Finset.sum_ite_eq]
  simp only [Finset.mem_range, Nat.div_lt_iff_lt_mul ht]

theorem connStates (E : Env) (ds dt dk p₁ p₂ c a b k : ℕ) (iface : List ℕ)
    (hs : 0 < E.dimv ds) (ht : 0 < E.dimv dt) (hdk : E.dimv dk = E.dimv ds * E.dimv dt)
    (ha : labIdx E.labs a = ds) (hb : labIdx E.labs b = dt) (hab : a ≠ b) (hak : a ≠ k)
    (hbk : b ≠ k) (hk : k ∈ iface ∧ labIdx E.labs k = dk)
    (h₁ : ∀ k < E.dimv ds, ex (E.phase p₁ k) = 1) (h₂ : ∀ k < E.dimv dt, ex (E.phase p₂ k) = 1)
    (hc : ∀ k < E.dimv dk, ex (E.phase c k) = 1) :
    LocalEq E [⟨.B, [k], [a, b], ds, dt⟩, ⟨.Z, [a], [], ds, p₁⟩, ⟨.Z, [b], [], dt, p₂⟩] [a, b]
      [⟨.Z, [k], [], dk, c⟩] [] iface 1 1 := by
  intro τ hτ
  have rk := hτ k hk.1
  rw [labDim_of E k dk hk.2] at rk
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E a ds ha, labDim_of E b dt hb, Function.update_apply,
    hab, hak, hbk, Ne.symm hab, Ne.symm hak, Ne.symm hbk, if_true, if_false, List.append_nil]
  rw [tZ_one _ _ hc, if_pos rk]
  rw [Finset.sum_congr rfl fun ja hja => Finset.sum_congr rfl fun jb hjb => by
    rw [tZ_one _ _ h₁, tZ_one _ _ h₂, if_pos (Finset.mem_range.1 hja),
      if_pos (Finset.mem_range.1 hjb), mul_one, mul_one], tB_sum _ _ _ ht, ← hdk, if_pos rk]

theorem connEffects (E : Env) (ds dt dk p₁ p₂ c a b k : ℕ) (iface : List ℕ)
    (hs : 0 < E.dimv ds) (ht : 0 < E.dimv dt) (hdk : E.dimv dk = E.dimv ds * E.dimv dt)
    (ha : labIdx E.labs a = ds) (hb : labIdx E.labs b = dt) (hab : a ≠ b) (hak : a ≠ k)
    (hbk : b ≠ k) (hk : k ∈ iface ∧ labIdx E.labs k = dk)
    (h₁ : ∀ k < E.dimv ds, ex (E.phase p₁ k) = 1) (h₂ : ∀ k < E.dimv dt, ex (E.phase p₂ k) = 1)
    (hc : ∀ k < E.dimv dk, ex (E.phase c k) = 1) :
    LocalEq E [⟨.S, [a, b], [k], ds, dt⟩, ⟨.Z, [], [a], ds, p₁⟩, ⟨.Z, [], [b], dt, p₂⟩] [a, b]
      [⟨.Z, [], [k], dk, c⟩] [] iface 1 1 := by
  intro τ hτ
  have rk := hτ k hk.1
  rw [labDim_of E k dk hk.2] at rk
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E a ds ha, labDim_of E b dt hb, Function.update_apply,
    hab, hak, hbk, Ne.symm hab, Ne.symm hak, Ne.symm hbk, if_true, if_false, List.nil_append]
  rw [tZ_one _ _ hc, if_pos rk]
  rw [Finset.sum_congr rfl fun ja hja => Finset.sum_congr rfl fun jb hjb => by
    rw [tZ_one _ _ h₁, tZ_one _ _ h₂, if_pos (Finset.mem_range.1 hja),
      if_pos (Finset.mem_range.1 hjb), mul_one, mul_one], tB_sum _ _ _ ht, ← hdk, if_pos rk]

theorem rt_mul (s t : ℕ) : rt (s * t) = rt s * rt t := by
  simp only [rt, Nat.cast_mul, Real.sqrt_mul (Nat.cast_nonneg s), Complex.ofReal_mul]

theorem tB_zero (s t K : ℕ) (hs : 0 < s) (ht : 0 < t) : tB s t K 0 0 = if K = 0 then 1 else 0 := by
  unfold tB; simp [hs, ht]

theorem connStatesX (E : Env) (ds dt dk p₁ p₂ c a b k : ℕ) (iface : List ℕ)
    (hs : 0 < E.dimv ds) (ht : 0 < E.dimv dt) (hdk : E.dimv dk = E.dimv ds * E.dimv dt)
    (ha : labIdx E.labs a = ds) (hb : labIdx E.labs b = dt) (hab : a ≠ b) (hak : a ≠ k)
    (hbk : b ≠ k) (hk : k ∈ iface ∧ labIdx E.labs k = dk)
    (h₁ : ∀ k < E.dimv ds, ex (E.phase p₁ k) = 1) (h₂ : ∀ k < E.dimv dt, ex (E.phase p₂ k) = 1)
    (hc : ∀ k < E.dimv dk, ex (E.phase c k) = 1) :
    LocalEq E [⟨.B, [k], [a, b], ds, dt⟩, ⟨.X, [a], [], ds, p₁⟩, ⟨.X, [b], [], dt, p₂⟩] [a, b]
      [⟨.X, [k], [], dk, c⟩] [] iface 1 1 := by
  intro τ hτ
  have rk := hτ k hk.1
  rw [labDim_of E k dk hk.2] at rk
  have hkd : 0 < E.dimv dk := by omega
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E a ds ha, labDim_of E b dt hb, Function.update_apply,
    hab, hak, hbk, Ne.symm hab, Ne.symm hak, Ne.symm hbk, if_true, if_false]
  rw [tX_flat _ hkd _ hc]
  simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
    zero_add, pow_one, Nat.zero_mod, Nat.mod_eq_of_lt rk, mul_ite, mul_zero]
  have step : ∀ ja ∈ range (E.dimv ds), ∑ jb ∈ range (E.dimv dt),
      tB (E.dimv ds) (E.dimv dt) (τ k) ja jb * (tX (E.dimv ds) (E.phase p₁) [ja] [] *
        tX (E.dimv dt) (E.phase p₂) [jb] []) =
      if ja % E.dimv ds = 0 then (∑ jb ∈ range (E.dimv dt), if jb % E.dimv dt = 0 then
        rt (E.dimv ds) * rt (E.dimv dt) * tB (E.dimv ds) (E.dimv dt) (τ k) ja jb else 0)
        else 0 := by
    intro ja _
    split_ifs with h
    · refine Finset.sum_congr rfl fun jb _ => ?_
      rw [tX_flat _ hs _ h₁, tX_flat _ ht _ h₂]
      simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
        zero_add, pow_one, Nat.zero_mod, h, if_true, rt_inv_mul _ hs]
      by_cases h' : jb % E.dimv dt = 0
      · simp only [h', if_true, rt_inv_mul _ ht]; ring
      · simp [h']
    · refine Finset.sum_eq_zero fun jb _ => ?_
      rw [tX_flat _ hs _ h₁]; simp [h]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hs, sum_mod0 _ ht, tB_zero _ _ _ hs ht]
  have e : (rt (E.dimv dk))⁻¹ * (E.dimv dk : ℂ) = rt (E.dimv ds) * rt (E.dimv dt) := by
    rw [rt_inv_mul _ hkd, hdk, rt_mul]
  rw [e]
  by_cases h0 : τ k = 0 <;> simp [h0]

theorem connEffectsX (E : Env) (ds dt dk p₁ p₂ c a b k : ℕ) (iface : List ℕ)
    (hs : 0 < E.dimv ds) (ht : 0 < E.dimv dt) (hdk : E.dimv dk = E.dimv ds * E.dimv dt)
    (ha : labIdx E.labs a = ds) (hb : labIdx E.labs b = dt) (hab : a ≠ b) (hak : a ≠ k)
    (hbk : b ≠ k) (hk : k ∈ iface ∧ labIdx E.labs k = dk)
    (h₁ : ∀ k < E.dimv ds, ex (E.phase p₁ k) = 1) (h₂ : ∀ k < E.dimv dt, ex (E.phase p₂ k) = 1)
    (hc : ∀ k < E.dimv dk, ex (E.phase c k) = 1) :
    LocalEq E [⟨.S, [a, b], [k], ds, dt⟩, ⟨.X, [], [a], ds, p₁⟩, ⟨.X, [], [b], dt, p₂⟩] [a, b]
      [⟨.X, [], [k], dk, c⟩] [] iface 1 1 := by
  intro τ hτ
  have rk := hτ k hk.1
  rw [labDim_of E k dk hk.2] at rk
  have hkd : 0 < E.dimv dk := by omega
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E a ds ha, labDim_of E b dt hb, Function.update_apply,
    hab, hak, hbk, Ne.symm hab, Ne.symm hak, Ne.symm hbk, if_true, if_false]
  rw [tX_flat _ hkd _ hc]
  simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
    zero_add, pow_one, Nat.zero_mod, Nat.mod_eq_of_lt rk, mul_ite, mul_zero]
  have step : ∀ ja ∈ range (E.dimv ds), ∑ jb ∈ range (E.dimv dt),
      tB (E.dimv ds) (E.dimv dt) (τ k) ja jb * (tX (E.dimv ds) (E.phase p₁) [] [ja] *
        tX (E.dimv dt) (E.phase p₂) [] [jb]) =
      if ja % E.dimv ds = 0 then (∑ jb ∈ range (E.dimv dt), if jb % E.dimv dt = 0 then
        rt (E.dimv ds) * rt (E.dimv dt) * tB (E.dimv ds) (E.dimv dt) (τ k) ja jb else 0)
        else 0 := by
    intro ja _
    split_ifs with h
    · refine Finset.sum_congr rfl fun jb _ => ?_
      rw [tX_flat _ hs _ h₁, tX_flat _ ht _ h₂]
      simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
        zero_add, pow_one, Nat.zero_mod, @eq_comm ℕ 0, h, if_true]
      by_cases h' : jb % E.dimv dt = 0
      · simp only [h', if_true, rt_inv_mul _ ht, rt_inv_mul _ hs]; ring
      · simp [h']
    · refine Finset.sum_eq_zero fun jb _ => ?_
      rw [tX_flat _ hs _ h₁]; simp [@eq_comm ℕ 0, h]
  rw [Finset.sum_congr rfl step, sum_mod0 _ hs, sum_mod0 _ ht, tB_zero _ _ _ hs ht]
  have e : (rt (E.dimv dk))⁻¹ * (E.dimv dk : ℂ) = rt (E.dimv ds) * rt (E.dimv dt) := by
    rw [rt_inv_mul _ hkd, hdk, rt_mul]
  rw [e]
  by_cases h0 : τ k = 0 <;> simp [h0, @eq_comm ℕ 0]

theorem connInverse (E : Env) (ds dt dk a b k₁ k₂ x y : ℕ) (iface : List ℕ)
    (ht : 0 < E.dimv dt) (hdk : E.dimv dk = E.dimv ds * E.dimv dt)
    (hxy : (x = k₁ ∧ y = k₂) ∨ (x = k₂ ∧ y = k₁)) (ha : labIdx E.labs a = ds)
    (hb : labIdx E.labs b = dt) (hd : [a, b, k₁, k₂].Nodup)
    (h₁ : k₁ ∈ iface ∧ labIdx E.labs k₁ = dk) (h₂ : k₂ ∈ iface ∧ labIdx E.labs k₂ = dk) :
    LocalEq E [⟨.S, [a, b], [k₁], ds, dt⟩, ⟨.B, [k₂], [a, b], ds, dt⟩] [a, b]
      [⟨.I, [x], [y], dk, 0⟩] [] iface 1 1 := by
  intro τ hτ
  have r₁ := hτ k₁ h₁.1
  have r₂ := hτ k₂ h₂.1
  rw [labDim_of E _ dk h₁.2] at r₁
  rw [labDim_of E _ dk h₂.2] at r₂
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hd
  obtain ⟨⟨hab, ha1, ha2⟩, ⟨hb1, hb2⟩, -, -⟩ := hd
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E a ds ha, labDim_of E b dt hb, Function.update_apply,
    hab, Ne.symm hab, Ne.symm ha1, Ne.symm ha2, Ne.symm hb1, Ne.symm hb2, if_true, if_false]
  have pt : ∀ ja jb, tB (E.dimv ds) (E.dimv dt) (τ k₁) ja jb *
      tB (E.dimv ds) (E.dimv dt) (τ k₂) ja jb =
      if τ k₁ = τ k₂ then tB (E.dimv ds) (E.dimv dt) (τ k₁) ja jb else 0 := by
    intro ja jb
    unfold tB
    by_cases e : τ k₁ = τ k₂
    · rw [if_pos e, e]; split_ifs <;> simp
    · rw [if_neg e]
      split_ifs with h1 h2
      · exact absurd (h1.2.2.trans h2.2.2.symm) e
      all_goals simp
  simp only [pt]
  by_cases e : τ k₁ = τ k₂
  · simp only [e, if_true]
    rw [tB_sum _ _ _ ht, ← hdk, if_pos r₂]
    rcases hxy with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp [tI, e, r₂]
  · simp only [e, if_false, Finset.sum_const_zero]
    rcases hxy with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> simp [tI, e, Ne.symm e]

theorem connInverseBind (E : Env) (ds dt dk k a₁ b₁ a₂ b₂ x₁ y₁ x₂ y₂ : ℕ) (iface : List ℕ)
    (hdk : E.dimv dk = E.dimv ds * E.dimv dt) (hk : labIdx E.labs k = dk)
    (hk' : k ∉ [a₁, b₁, a₂, b₂])
    (hxa : (x₁ = a₁ ∧ y₁ = a₂) ∨ (x₁ = a₂ ∧ y₁ = a₁))
    (hxb : (x₂ = b₁ ∧ y₂ = b₂) ∨ (x₂ = b₂ ∧ y₂ = b₁))
    (ha₁ : a₁ ∈ iface ∧ labIdx E.labs a₁ = ds) (ha₂ : a₂ ∈ iface ∧ labIdx E.labs a₂ = ds)
    (hb₁ : b₁ ∈ iface ∧ labIdx E.labs b₁ = dt) (hb₂ : b₂ ∈ iface ∧ labIdx E.labs b₂ = dt) :
    LocalEq E [⟨.B, [k], [a₁, b₁], ds, dt⟩, ⟨.S, [a₂, b₂], [k], ds, dt⟩] [k]
      [⟨.I, [x₁], [y₁], ds, 0⟩, ⟨.I, [x₂], [y₂], dt, 0⟩] [] iface 1 1 := by
  intro τ hτ
  have r₁ := hτ a₁ ha₁.1
  have r₂ := hτ a₂ ha₂.1
  have r₃ := hτ b₁ hb₁.1
  have r₄ := hτ b₂ hb₂.1
  rw [labDim_of E _ ds ha₁.2] at r₁
  rw [labDim_of E _ ds ha₂.2] at r₂
  rw [labDim_of E _ dt hb₁.2] at r₃
  rw [labDim_of E _ dt hb₂.2] at r₄
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hk'
  obtain ⟨n1, n2, n3, n4⟩ := hk'
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E k dk hk, Function.update_apply, Function.update_self,
    Ne.symm n1, Ne.symm n2, Ne.symm n3, Ne.symm n4, if_false, if_true, eq_self_iff_true]
  set s := E.dimv ds
  set t := E.dimv dt
  have ht : 0 < t := by omega
  have pt : ∀ K ∈ range (E.dimv dk), tB s t K (τ a₁) (τ b₁) * tB s t K (τ a₂) (τ b₂) =
      if K = τ a₁ * t + τ b₁ then (if τ a₁ = τ a₂ ∧ τ b₁ = τ b₂ then 1 else 0) else 0 := by
    intro K _
    unfold tB
    simp only [r₁, r₂, r₃, r₄, true_and]
    by_cases e1 : K = τ a₁ * t + τ b₁
    · rw [if_pos e1, if_pos e1]
      have := (pair_iff t (τ a₁) (τ b₁) K r₃).1 e1
      by_cases e2 : K = τ a₂ * t + τ b₂
      · have := (pair_iff t (τ a₂) (τ b₂) K r₄).1 e2
        rw [if_pos e2, if_pos (by omega)]; simp
      · rw [if_neg e2]
        have : ¬ (τ a₁ = τ a₂ ∧ τ b₁ = τ b₂) := by
          rintro ⟨h1, h2⟩; exact e2 (h1 ▸ h2 ▸ e1)
        rw [if_neg this]; simp
    · rw [if_neg e1, if_neg e1]; simp
  rw [Finset.sum_congr rfl pt, Finset.sum_ite_eq']
  have hlt : τ a₁ * t + τ b₁ < E.dimv dk := by
    rw [hdk]
    calc τ a₁ * t + τ b₁ < τ a₁ * t + t := by omega
      _ = (τ a₁ + 1) * t := by ring
      _ ≤ s * t := Nat.mul_le_mul_right _ r₁
  rw [if_pos (Finset.mem_range.2 hlt)]
  rcases hxa with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases hxb with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
    by_cases e1 : τ x₁ = τ y₁ <;> by_cases e2 : τ x₂ = τ y₂ <;> simp_all [tI, eq_comm]

end ArchytasZX
