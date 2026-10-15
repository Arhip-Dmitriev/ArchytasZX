/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ZXRules

/-!
# Bialgebra
-/

namespace ArchytasZX

open Complex Finset

theorem tZ_two (d : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = 1) (x y z : ℕ) (hz : z < d) :
    tZ d φ [x, y, z] = if x = z ∧ y = z then 1 else 0 := by
  rw [tZ_at d φ _ z (by simp)]
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, and_true]
  by_cases h : x = z ∧ y = z
  · rw [if_pos ⟨hz, h⟩, if_pos h, hφ z hz]
  · rw [if_neg (fun h' => h h'.2), if_neg h]

theorem tZ_first (d : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = 1) (x y z : ℕ) (hz : z < d) :
    tZ d φ [z, x, y] = if x = z ∧ y = z then 1 else 0 := by
  rw [tZ_at d φ _ z (by simp)]
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, true_and]
  by_cases h : x = z ∧ y = z
  · rw [if_pos ⟨hz, h⟩, if_pos h, hφ z hz]
  · rw [if_neg (fun h' => h h'.2), if_neg h]

theorem bialgebra (E : Env) (px pz q₁ q₂ q₃ q₄ dz w a b c e n₀₀ n₀₁ n₁₀ n₁₁ : ℕ)
    (iface : List ℕ) (hd : 0 < E.dimv dz) (hw : labIdx E.labs w = dz)
    (hn : ∀ n ∈ [n₀₀, n₀₁, n₁₀, n₁₁], labIdx E.labs n = dz)
    (hnd : [n₀₀, n₀₁, n₁₀, n₁₁, a, b, c, e].Nodup) (hwd : [w, a, b, c, e].Nodup)
    (hf : ∀ x ∈ [a, b, c, e], x ∈ iface ∧ labIdx E.labs x = dz)
    (hpx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) (hpz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (h₁ : ∀ k < E.dimv dz, ex (E.phase q₁ k) = 1) (h₂ : ∀ k < E.dimv dz, ex (E.phase q₂ k) = 1)
    (h₃ : ∀ k < E.dimv dz, ex (E.phase q₃ k) = 1) (h₄ : ∀ k < E.dimv dz, ex (E.phase q₄ k) = 1) :
    LocalEq E [⟨.X, [w], [a, b], dz, px⟩, ⟨.Z, [c, e], [w], dz, pz⟩] [w]
      [⟨.Z, [n₀₀, n₀₁], [a], dz, q₁⟩, ⟨.Z, [n₁₀, n₁₁], [b], dz, q₂⟩,
        ⟨.X, [c], [n₀₀, n₁₀], dz, q₃⟩, ⟨.X, [e], [n₀₁, n₁₁], dz, q₄⟩]
      [n₀₀, n₀₁, n₁₀, n₁₁] iface 1 (rt (E.dimv dz)) := by
  intro τ hτ
  have hr : ∀ x ∈ [a, b, c, e], τ x < E.dimv dz := fun x hx => by
    have := hτ x (hf x hx).1; rwa [labDim_of E x dz (hf x hx).2] at this
  have ra := hr a (by simp)
  have rb := hr b (by simp)
  have rc := hr c (by simp)
  have re := hr e (by simp)
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd hwd
  obtain ⟨⟨h01, h02, h03, h0a, h0b, h0c, h0e⟩, ⟨h12, h13, h1a, h1b, h1c, h1e⟩,
    ⟨h23, h2a, h2b, h2c, h2e⟩, ⟨h3a, h3b, h3c, h3e⟩, -⟩ := hnd
  obtain ⟨⟨hwa, hwb, hwc, hwe⟩, -⟩ := hwd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hn
  obtain ⟨hn0, hn1, hn2, hn3⟩ := hn
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E _ dz hw, labDim_of E _ dz hn0,
    labDim_of E _ dz hn1, labDim_of E _ dz hn2, labDim_of E _ dz hn3, Function.update_apply,
    if_true, if_false, List.cons_append, List.nil_append, Ne.symm hwa, Ne.symm hwb,
    Ne.symm hwc, Ne.symm hwe, Ne.symm h0a, Ne.symm h0b, Ne.symm h0c, Ne.symm h0e,
    Ne.symm h1a, Ne.symm h1b, Ne.symm h1c, Ne.symm h1e, Ne.symm h2a, Ne.symm h2b,
    Ne.symm h2c, Ne.symm h2e, Ne.symm h3a, Ne.symm h3b, Ne.symm h3c, Ne.symm h3e, h01, h02,
    h03, h12, h13, h23, Ne.symm h01, Ne.symm h02, Ne.symm h03, Ne.symm h12, Ne.symm h13,
    Ne.symm h23]
  generalize E.dimv dz = d at *
  set A := τ a
  set B := τ b
  set C := τ c
  set Ee := τ e
  -- left side
  have hl : ∀ j ∈ range d, tX d (E.phase px) [j] [A, B] * tZ d (E.phase pz) [C, Ee, j] =
      if j = C then (if Ee = C then tX d (E.phase px) [C] [A, B] else 0) else 0 := by
    intro j hj
    rw [tZ_two d _ hpz C Ee j (Finset.mem_range.1 hj)]
    by_cases h1 : j = C
    · rw [h1]; by_cases h2 : Ee = C <;> simp [h2]
    · have : ¬ (C = j ∧ Ee = j) := fun h => h1 h.1.symm
      simp [h1, this]
  rw [Finset.sum_congr rfl hl, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 rc)]
  -- right side
  have hrt : ∀ j₀₀ ∈ range d, ∀ j₀₁ ∈ range d, ∀ j₁₀ ∈ range d, ∀ j₁₁ ∈ range d,
      tZ d (E.phase q₁) [j₀₀, j₀₁, A] * (tZ d (E.phase q₂) [j₁₀, j₁₁, B] *
        (tX d (E.phase q₃) [C] [j₀₀, j₁₀] * tX d (E.phase q₄) [Ee] [j₀₁, j₁₁])) =
      if j₁₁ = B then (if j₁₀ = B then (if j₀₁ = A then (if j₀₀ = A then
        tX d (E.phase q₃) [C] [A, B] * tX d (E.phase q₄) [Ee] [A, B] else 0) else 0)
        else 0) else 0 := by
    intro j₀₀ _ j₀₁ _ j₁₀ _ j₁₁ _
    rw [tZ_two d _ h₁ _ _ _ ra, tZ_two d _ h₂ _ _ _ rb]
    by_cases g₀₀ : j₀₀ = A <;> by_cases g₀₁ : j₀₁ = A <;> by_cases g₁₀ : j₁₀ = B <;>
      by_cases g₁₁ : j₁₁ = B <;> simp [g₀₀, g₀₁, g₁₀, g₁₁]
  rw [Finset.sum_congr rfl fun j₀₀ h₀₀ => Finset.sum_congr rfl fun j₀₁ h₀₁ =>
    Finset.sum_congr rfl fun j₁₀ h₁₀ => Finset.sum_congr rfl fun j₁₁ h₁₁ =>
      hrt j₀₀ h₀₀ j₀₁ h₀₁ j₁₀ h₁₀ j₁₁ h₁₁]
  simp only [Finset.sum_ite_eq', Finset.mem_range, rb, ra, if_true]
  rw [tX_flat _ hd _ hpx, tX_flat _ hd _ h₃, tX_flat _ hd _ h₄]
  simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
    Nat.mod_eq_of_lt rc, Nat.mod_eq_of_lt re]
  have hrt0 := rt_ne d hd
  have h2 := rt_sq d
  by_cases hce : Ee = C
  · rw [if_pos hce, hce]
    by_cases hs : C = (A + B) % d
    · simp only [if_pos hs]; rw [← h2]; field_simp; rw [← mul_pow, mul_one_div_cancel hrt0, one_pow]
    · simp [hs]
  · rw [if_neg hce]
    by_cases hs : C = (A + B) % d
    · have : ¬ Ee = (A + B) % d := fun h => hce (h.trans hs.symm)
      simp [this]
    · simp [hs]

theorem bialgebraSwapped (E : Env) (px pz q₁ q₂ q₃ q₄ dz w a b c e n₀₀ n₀₁ n₁₀ n₁₁ : ℕ)
    (iface : List ℕ) (hd : 0 < E.dimv dz) (hw : labIdx E.labs w = dz)
    (hn : ∀ n ∈ [n₀₀, n₀₁, n₁₀, n₁₁], labIdx E.labs n = dz)
    (hnd : [n₀₀, n₀₁, n₁₀, n₁₁, a, b, c, e].Nodup) (hwd : [w, a, b, c, e].Nodup)
    (hf : ∀ x ∈ [a, b, c, e], x ∈ iface ∧ labIdx E.labs x = dz)
    (hpz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hpx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (h₁ : ∀ k < E.dimv dz, ex (E.phase q₁ k) = 1) (h₂ : ∀ k < E.dimv dz, ex (E.phase q₂ k) = 1)
    (h₃ : ∀ k < E.dimv dz, ex (E.phase q₃ k) = 1) (h₄ : ∀ k < E.dimv dz, ex (E.phase q₄ k) = 1) :
    LocalEq E [⟨.Z, [w], [a, b], dz, pz⟩, ⟨.X, [c, e], [w], dz, px⟩] [w]
      [⟨.X, [n₀₀, n₀₁], [a], dz, q₁⟩, ⟨.X, [n₁₀, n₁₁], [b], dz, q₂⟩,
        ⟨.Z, [c], [n₀₀, n₁₀], dz, q₃⟩, ⟨.Z, [e], [n₀₁, n₁₁], dz, q₄⟩]
      [n₀₀, n₀₁, n₁₀, n₁₁] iface 1 (rt (E.dimv dz)) := by
  intro τ hτ
  have hr : ∀ x ∈ [a, b, c, e], τ x < E.dimv dz := fun x hx => by
    have := hτ x (hf x hx).1; rwa [labDim_of E x dz (hf x hx).2] at this
  have ra := hr a (by simp)
  have rb := hr b (by simp)
  have rc := hr c (by simp)
  have re := hr e (by simp)
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd hwd
  obtain ⟨⟨h01, h02, h03, h0a, h0b, h0c, h0e⟩, ⟨h12, h13, h1a, h1b, h1c, h1e⟩,
    ⟨h23, h2a, h2b, h2c, h2e⟩, ⟨h3a, h3b, h3c, h3e⟩, -⟩ := hnd
  obtain ⟨⟨hwa, hwb, hwc, hwe⟩, -⟩ := hwd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hn
  obtain ⟨hn0, hn1, hn2, hn3⟩ := hn
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E _ dz hw, labDim_of E _ dz hn0,
    labDim_of E _ dz hn1, labDim_of E _ dz hn2, labDim_of E _ dz hn3, Function.update_apply,
    if_true, if_false, List.cons_append, List.nil_append, Ne.symm hwa, Ne.symm hwb,
    Ne.symm hwc, Ne.symm hwe, Ne.symm h0a, Ne.symm h0b, Ne.symm h0c, Ne.symm h0e,
    Ne.symm h1a, Ne.symm h1b, Ne.symm h1c, Ne.symm h1e, Ne.symm h2a, Ne.symm h2b,
    Ne.symm h2c, Ne.symm h2e, Ne.symm h3a, Ne.symm h3b, Ne.symm h3c, Ne.symm h3e, h01, h02,
    h03, h12, h13, h23, Ne.symm h01, Ne.symm h02, Ne.symm h03, Ne.symm h12, Ne.symm h13,
    Ne.symm h23]
  generalize E.dimv dz = d at *
  set A := τ a
  set B := τ b
  set C := τ c
  set Ee := τ e
  have hl : ∀ j ∈ range d, tZ d (E.phase pz) [j, A, B] * tX d (E.phase px) [C, Ee] [j] =
      if j = A then (if B = A then tX d (E.phase px) [C, Ee] [A] else 0) else 0 := by
    intro j hj
    rw [tZ_first d _ hpz A B j (Finset.mem_range.1 hj)]
    by_cases h1 : j = A
    · rw [h1]; by_cases h2 : B = A <;> simp [h2]
    · have : ¬ (A = j ∧ B = j) := fun h => h1 h.1.symm
      simp [h1, this]
  rw [Finset.sum_congr rfl hl, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ra)]
  have hrt : ∀ j₀₀ ∈ range d, ∀ j₀₁ ∈ range d, ∀ j₁₀ ∈ range d, ∀ j₁₁ ∈ range d,
      tX d (E.phase q₁) [j₀₀, j₀₁] [A] * (tX d (E.phase q₂) [j₁₀, j₁₁] [B] *
        (tZ d (E.phase q₃) [C, j₀₀, j₁₀] * tZ d (E.phase q₄) [Ee, j₀₁, j₁₁])) =
      if j₁₁ = Ee then (if j₁₀ = C then (if j₀₁ = Ee then (if j₀₀ = C then
        tX d (E.phase q₁) [C, Ee] [A] * tX d (E.phase q₂) [C, Ee] [B] else 0) else 0)
        else 0) else 0 := by
    intro j₀₀ _ j₀₁ _ j₁₀ _ j₁₁ _
    rw [tZ_first d _ h₃ _ _ _ rc, tZ_first d _ h₄ _ _ _ re]
    by_cases g₀₀ : j₀₀ = C <;> by_cases g₀₁ : j₀₁ = Ee <;> by_cases g₁₀ : j₁₀ = C <;>
      by_cases g₁₁ : j₁₁ = Ee <;> simp [g₀₀, g₀₁, g₁₀, g₁₁]
  rw [Finset.sum_congr rfl fun j₀₀ h₀₀ => Finset.sum_congr rfl fun j₀₁ h₀₁ =>
    Finset.sum_congr rfl fun j₁₀ h₁₀ => Finset.sum_congr rfl fun j₁₁ h₁₁ =>
      hrt j₀₀ h₀₀ j₀₁ h₀₁ j₁₀ h₁₀ j₁₁ h₁₁]
  simp only [Finset.sum_ite_eq', Finset.mem_range, rc, re, if_true]
  rw [tX_flat _ hd _ hpx, tX_flat _ hd _ h₁, tX_flat _ hd _ h₂]
  simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
    Nat.mod_eq_of_lt ra, Nat.mod_eq_of_lt rb]
  have hrt0 := rt_ne d hd
  have h2 := rt_sq d
  by_cases hab : B = A
  · rw [if_pos hab, hab]
    by_cases hs : (C + Ee) % d = A
    · simp only [if_pos hs]; rw [← h2]; field_simp; rw [← mul_pow, mul_one_div_cancel hrt0, one_pow]
    · simp [hs]
  · rw [if_neg hab]
    by_cases hs : (C + Ee) % d = A
    · have : ¬ (C + Ee) % d = B := fun h => hab (h.symm.trans hs)
      simp [this]
    · simp [hs]

end ArchytasZX
