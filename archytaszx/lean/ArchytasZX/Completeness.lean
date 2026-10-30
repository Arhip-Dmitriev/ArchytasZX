/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net
import ArchytasZX.Rules
import ArchytasZX.WRules
import ArchytasZX.Closed
import ArchytasZX.ZXRules

/-!
# Lemmas of the complete axiomatisation

Rule lemmas for template rules of `archytaszx.rewrite.rules_axioms` that no earlier lemma states.
-/

namespace ArchytasZX

open Complex Finset

/-- [P1] (Sym), [P2] (2.10): the outputs of a W node may be permuted. -/
theorem wPerm (E : Env) (a b dz i : ℕ) (os os' iface : List ℕ) (h : os.Perm os') :
    LocalEq E [⟨.W, os, [i], dz, a⟩] [] [⟨.W, os', [i], dz, b⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val]
  exact tW_perm _ _ _ _ (h.map τ)

/-- [P2] (2.9): the legs of a Z spider may be permuted and bent between inputs and outputs. -/
theorem zPerm (E : Env) (a dz : ℕ) (os is os' is' iface : List ℕ)
    (h : (os ++ is).Perm (os' ++ is')) :
    LocalEq E [⟨.Z, os, is, dz, a⟩] [] [⟨.Z, os', is', dz, a⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val]
  rw [← List.map_append, ← List.map_append]
  exact tZ_perm _ _ _ _ (h.map τ)

/-- [P1] (Aso): two ways of nesting W nodes, both the W node with every output (by wFusion). -/
theorem wAso (E : Env) (dz w w' i : ℕ) (o₁ o₂ p₁ p₂ o₃ iface : List ℕ)
    (hw : labIdx E.labs w = dz) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₂ : w ∉ o₂)
    (hwi : w ≠ i) (hi : i ∈ iface ∧ labIdx E.labs i = dz) (hd : 0 < E.dimv dz)
    (ho : o₃.Perm (o₁.erase w ++ o₂))
    (gw : labIdx E.labs w' = dz) (g₁ : w' ∈ p₁) (g₁' : w' ∉ p₁.erase w') (g₂ : w' ∉ p₂)
    (gwi : w' ≠ i) (go : o₃.Perm (p₁.erase w' ++ p₂)) :
    LocalEq E [⟨.W, o₁, [i], dz, 0⟩, ⟨.W, o₂, [w], dz, 0⟩] [w]
      [⟨.W, p₁, [i], dz, 0⟩, ⟨.W, p₂, [w'], dz, 0⟩] [w'] iface 1 1 := by
  intro τ hτ
  rw [wFusion E dz w i o₁ o₂ o₃ iface hw h₁ h₁' h₂ hwi hi hd ho τ hτ,
    wFusion E dz w' i p₁ p₂ o₃ iface gw g₁ g₁' g₂ gwi hi hd go τ hτ]

/-- The X spider with phases `K_j` (`e^{2πi jk/d}`) in closed form. -/
theorem tX_kj (d j : ℕ) (hd : 0 < d) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = om d ^ (k * j))
    (os is : List ℕ) :
    tX d φ os is = (rt d)⁻¹ ^ (os.length + is.length) *
      (if (os.sum + j) % d = is.sum % d then (d : ℂ) else 0) := by
  unfold tX
  rw [← char_sum d hd, Finset.mul_sum]
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hφ k (Finset.mem_range.1 hk), prod_fou, prod_conj_fou, pow_add,
    show k * (os.sum + j) = k * j + k * os.sum by ring, pow_add]
  ring

theorem prod_ite_val (l : List ℕ) (τ : ℕ → ℕ) (v : ℕ) (c : ℂ) :
    (l.map fun o => if τ o = v then c else 0).prod =
      if ∀ o ∈ l, τ o = v then c ^ l.length else 0 := by
  induction l with
  | nil => simp
  | cons o l ih =>
    simp only [List.map_cons, List.prod_cons, ih, List.mem_cons, forall_eq_or_imp,
      List.length_cons, pow_succ]
    by_cases h : τ o = v <;> by_cases h' : ∀ x ∈ l, τ x = v <;> simp [h, h', mul_comm]

/-- The unique value below `d` cancelling `j` modulo `d`. -/
theorem cancel_iff (d j w : ℕ) (hd : 0 < d) (hw : w < d) :
    (w + j) % d = 0 ↔ w = (d - j % d) % d := by
  constructor
  · intro h
    have h1 : (w + j % d) % d = 0 := by rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]; exact h
    have hj : j % d < d := Nat.mod_lt _ hd
    rcases Nat.eq_zero_or_pos (j % d) with h0 | hpos
    · rw [h0, Nat.add_zero, Nat.mod_eq_of_lt hw] at h1
      rw [h0, Nat.sub_zero, Nat.mod_self]; exact h1
    · have hlt : d - j % d < d := Nat.sub_lt hd hpos
      rw [Nat.mod_eq_of_lt hlt]
      have hsum : w + j % d < 2 * d := by omega
      have := Nat.dvd_of_mod_eq_zero h1
      obtain ⟨c, hc⟩ := this
      have hc1 : c = 1 := by
        rcases c with _ | _ | c
        · omega
        · rfl
        · nlinarith
      subst hc1; omega
  · intro h
    rw [h]
    have hj : j % d < d := Nat.mod_lt _ hd
    rw [Nat.add_mod, Nat.mod_mod_of_dvd _ (dvd_refl d)]
    rcases Nat.eq_zero_or_pos (j % d) with h0 | hpos
    · rw [h0]; simp
    · rw [Nat.mod_eq_of_lt (Nat.sub_lt hd hpos), Nat.sub_add_cancel hj.le, Nat.mod_self]

/-- [P1] (K0) for `n` outputs: the pink `K_j` state copied by a plain green dot. -/
theorem kCopy (E : Env) (a b c dz w j : ℕ) (os iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwo : w ∉ os)
    (hos : ∀ o ∈ os, o ∈ iface ∧ labIdx E.labs o = dz)
    (ha : ∀ q < E.dimv dz, ex (E.phase a q) = om (E.dimv dz) ^ (q * j))
    (hb : ∀ q < E.dimv dz, ex (E.phase b q) = 1)
    (hc : ∀ q < E.dimv dz, ex (E.phase c q) = om (E.dimv dz) ^ (q * j))
    (hk : k * rt (E.dimv dz) ^ os.length = rt (E.dimv dz)) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.Z, os, [w], dz, b⟩] [w]
      (os.map fun o => ⟨.X, [o], [], dz, c⟩) [] iface 1 k := by
  intro τ hτ
  have hr : ∀ o ∈ os, τ o < E.dimv dz := fun o ho => by
    have := hτ o (hos o ho).1; rwa [labDim_of E o dz (hos o ho).2] at this
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    map_update_absent τ w _ os hwo, List.map_map]
  set d := E.dimv dz with hdd
  set v := (d - j % d) % d with hv
  have hvd : v < d := Nat.mod_lt _ hd
  have hx : ∀ o ∈ os, (Function.comp (val E τ) fun o => (⟨.X, [o], [], dz, c⟩ : Node)) o =
      if τ o = v then rt d else 0 := by
    intro o ho
    simp only [Function.comp, val, List.map_cons, List.map_nil]
    rw [tX_kj _ j hd _ hc]
    simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
      zero_add, pow_one, Nat.zero_mod]
    by_cases h : τ o = v
    · rw [if_pos ((cancel_iff d j (τ o) hd (hr o ho)).2 h), if_pos h, rt_inv_mul _ hd]
    · rw [if_neg (fun h' => h ((cancel_iff d j (τ o) hd (hr o ho)).1 h')), if_neg h, mul_zero]
  rw [List.map_congr_left hx, prod_ite_val]
  have step : ∀ x ∈ range d, tX d (E.phase a) [x] [] * tZ d (E.phase b) (os.map τ ++ [x]) =
      if x = v then (rt d)⁻¹ * d * tZ d (E.phase b) (os.map τ ++ [x]) else 0 := by
    intro x hx'
    rw [tX_kj _ j hd _ ha]
    simp only [List.sum_cons, List.sum_nil, add_zero, List.length_cons, List.length_nil,
      zero_add, pow_one, Nat.zero_mod]
    by_cases h : x = v
    · rw [if_pos ((cancel_iff d j x hd (Finset.mem_range.1 hx')).2 h), if_pos h]
    · rw [if_neg (fun h' => h ((cancel_iff d j x hd (Finset.mem_range.1 hx')).1 h')),
        if_neg h, mul_zero, zero_mul]
  rw [Finset.sum_congr rfl step, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hvd),
    rt_inv_mul _ hd, tZ_at _ _ _ v (by simp), hb v hvd]
  have hiff : (v < d ∧ ∀ u ∈ os.map τ ++ [v], u = v) ↔ ∀ o ∈ os, τ o = v := by
    constructor
    · rintro ⟨-, h⟩ o ho
      exact h _ (List.mem_append_left _ (List.mem_map_of_mem ho))
    · intro h
      refine ⟨hvd, fun u hu => ?_⟩
      rcases List.mem_append.1 hu with hu | hu
      · obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hu
        exact h o ho
      · simpa using hu
  by_cases h : ∀ o ∈ os, τ o = v
  · rw [if_pos (hiff.2 h), if_pos h, mul_one, hk]
  · rw [if_neg (fun h' => h (hiff.1 h')), if_neg h]; simp

end ArchytasZX
