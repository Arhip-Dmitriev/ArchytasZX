/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bialgebra

/-!
# Hopf law
-/

namespace ArchytasZX

open Complex Finset

theorem tX_perm (d : ℕ) (φ : ℕ → ℝ) (os os' is is' : List ℕ) (ho : os.Perm os')
    (hi : is.Perm is') : tX d φ os is = tX d φ os' is' := by
  unfold tX
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [(ho.map _).prod_eq, (hi.map _).prod_eq]

theorem om_pow_zero_mod (d : ℕ) (hd : 0 < d) (s m : ℕ) (h : s % d = 0) : om d ^ (s * m) = 1 := by
  rw [pow_mul, om_pow_mod d hd s, h, pow_zero, one_pow]

/-- Two inputs carrying opposite values drop out of an X spider as `1/d`. -/
theorem tX_antipode_in (d : ℕ) (hd : 0 < d) (φ : ℕ → ℝ) (os is : List ℕ) (j j' : ℕ)
    (h : (j + j') % d = 0) : tX d φ os (j :: j' :: is) = (d : ℂ)⁻¹ * tX d φ os is := by
  unfold tX
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun m _ => ?_
  have hp : (starRingEnd ℂ) (fou d j m) * (starRingEnd ℂ) (fou d j' m) = (d : ℂ)⁻¹ := by
    rw [← map_mul]
    simp only [fou]
    rw [div_mul_div_comm, ← pow_add, ← add_mul, om_pow_zero_mod d hd _ _ h, sqrt_mul_sqrt]
    simp
  simp only [List.map_cons, List.prod_cons]
  rw [← mul_assoc ((starRingEnd ℂ) (fou d j m)), hp]
  ring

/-- Two outputs carrying opposite values drop out of an X spider as `1/d`. -/
theorem tX_antipode_out (d : ℕ) (hd : 0 < d) (φ : ℕ → ℝ) (os is : List ℕ) (j j' : ℕ)
    (h : (j + j') % d = 0) : tX d φ (j :: j' :: os) is = (d : ℂ)⁻¹ * tX d φ os is := by
  unfold tX
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun m _ => ?_
  have hp : fou d j m * fou d j' m = (d : ℂ)⁻¹ := by
    simp only [fou]
    rw [div_mul_div_comm, ← pow_add, ← add_mul, om_pow_zero_mod d hd _ _ h, sqrt_mul_sqrt]
    simp
  simp only [List.map_cons, List.prod_cons]
  rw [← mul_assoc (fou d j m), hp]
  ring

theorem tZ_sum_cons (d : ℕ) (φ : ℕ → ℝ) (rz : List ℕ) :
    ∑ j ∈ range d, tZ d φ (j :: rz) = tZ d φ rz := by
  unfold tZ
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  have : ∀ j ∈ range d, (if ∀ v ∈ j :: rz, v = k then ex (φ k) else 0) =
      if j = k then (if ∀ v ∈ rz, v = k then ex (φ k) else 0) else 0 := by
    intro j _
    by_cases h : j = k
    · subst h; simp
    · simp [h]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos hk]

theorem tZ_cons_cons (d : ℕ) (φ : ℕ → ℝ) (rz : List ℕ) (j j' : ℕ) :
    tZ d φ (j :: j' :: rz) = if j' = j then tZ d φ (j :: rz) else 0 := by
  unfold tZ
  split_ifs with h
  · subst h
    refine Finset.sum_congr rfl fun k _ => ?_
    simp
  · refine Finset.sum_eq_zero fun k _ => ?_
    have : ¬ ∀ v ∈ j :: j' :: rz, v = k := fun h' =>
      h ((h' j' (by simp)).trans (h' j (by simp)).symm)
    simp only [this, if_false]

theorem antipode_collapse (d : ℕ) (hd : 0 < d) (j : ℕ) (hj : j < d) (g : ℕ → ℂ) :
    ∑ m ∈ range d, (if (m + j) % d = 0 then g m else 0) = g ((d - j) % d) := by
  have : ∀ m ∈ range d, (if (m + j) % d = 0 then g m else 0) =
      if m = (d - j) % d then g m else 0 := by
    intro m hm
    simp only [antipode_iff d j m hj (Finset.mem_range.1 hm)]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd))]

theorem antipode_spec (d j : ℕ) (hd : 0 < d) (hj : j < d) : (j + (d - j) % d) % d = 0 := by
  rw [add_comm, antipode_iff d j _ hj (Nat.mod_lt _ hd)]

theorem hopf (E : Env) (pz px dz u v₁ v₂ v₃ : ℕ) (zo zi xo xi zo' zi' xo' xi' iface : List ℕ)
    (k : ℂ) (hd : 0 < E.dimv dz) (hz : (zo ++ zi).Perm (u :: v₁ :: (zo' ++ zi')))
    (hxo : xo.Perm xo') (hxi : xi.Perm (u :: v₃ :: xi')) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.Z, zo, zi, dz, pz⟩, ⟨.X, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.Z, zo', zi', dz, pz⟩, ⟨.X, xo', xi', dz, px⟩]
      [] iface 1 k := by
  intro τ _
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd
  obtain ⟨⟨h01, h02, h03⟩, ⟨h12, h13⟩, h23, -⟩ := hnd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hdim
  obtain ⟨du, d1, d2, d3⟩ := hdim
  set d := E.dimv dz
  set R := (zo' ++ zi').map τ
  set XO := xo'.map τ
  set XI := xi'.map τ
  have hprod : ∀ ρ : ℕ → ℕ, (∀ x ∈ zo' ++ zi' ++ xo' ++ xi', ρ x = τ x) →
      prodVal E [⟨.Z, zo, zi, dz, pz⟩, ⟨.X, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tZ d (E.phase pz) (ρ u :: ρ v₁ :: R) *
        (tX d (E.phase px) XO (ρ u :: ρ v₃ :: XI) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZ_perm _ _ _ _ (hz.map ρ), tX_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
    have e1 : (zo' ++ zi').map ρ = R := List.map_congr_left fun x hx =>
      hρ x (by simp only [List.mem_append] at hx ⊢; tauto)
    have e2 : xo'.map ρ = XO := List.map_congr_left fun x hx =>
      hρ x (by simp only [List.mem_append] at hx ⊢; tauto)
    have e3 : xi'.map ρ = XI := List.map_congr_left fun x hx =>
      hρ x (by simp only [List.mem_append] at hx ⊢; tauto)
    simp only [List.map_cons, e1, e2, e3]
    rfl
  rw [one_mul]
  simp only [sumOver, labDim_of E _ dz du, labDim_of E _ dz d1, labDim_of E _ dz d2,
    labDim_of E _ dz d3]
  rw [Finset.sum_congr rfl fun ju _ => Finset.sum_congr rfl fun j₁ _ =>
    Finset.sum_congr rfl fun j₂ _ => Finset.sum_congr rfl fun j₃ _ => hprod _ fun x hx => by
      have := hfree x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at this
      simp [Function.update_apply, this.1, this.2.1, this.2.2.1, this.2.2.2]]
  simp only [Function.update_apply, h01, h02, h03, h12, h13, h23, Ne.symm h01, Ne.symm h02,
    Ne.symm h03, Ne.symm h12, Ne.symm h13, Ne.symm h23, if_true, if_false]
  have hA : ∀ ju ∈ range d, ∀ j₁ ∈ range d, ∑ j₂ ∈ range d, ∑ j₃ ∈ range d,
      tZ d (E.phase pz) (ju :: j₁ :: R) * (tX d (E.phase px) XO (ju :: j₃ :: XI) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if j₁ = ju then tZ d (E.phase pz) (ju :: R) * ∑ j₃ ∈ range d,
        (if (j₃ + ju) % d = 0 then tX d (E.phase px) XO (ju :: j₃ :: XI) else 0) else 0 := by
    intro ju _ j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tZ d (E.phase pz) (ju :: j₁ :: R) *
        (tX d (E.phase px) XO (ju :: j₃ :: XI) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        tZ d (E.phase pz) (ju :: j₁ :: R) * tX d (E.phase px) XO (ju :: j₃ :: XI) *
          (if (j₃ + j₁) % d = 0 then 1 else 0) := by
      intro j₃ _
      rw [← fou_sq d hd, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j₂ _ => by ring
    rw [Finset.sum_congr rfl this, tZ_cons_cons]
    split_ifs with h
    · subst h; rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j₃ _ => by split_ifs <;> simp
    · simp
  rw [Finset.sum_congr rfl fun ju hju => (Finset.sum_congr rfl (hA ju hju)).trans
    (by rw [Finset.sum_ite_eq', if_pos hju])]
  have hC : ∀ ju ∈ range d, tZ d (E.phase pz) (ju :: R) * ∑ j₃ ∈ range d,
      (if (j₃ + ju) % d = 0 then tX d (E.phase px) XO (ju :: j₃ :: XI) else 0) =
      tZ d (E.phase pz) (ju :: R) * ((d : ℂ)⁻¹ * tX d (E.phase px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tX_antipode_in d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZ_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring

theorem hopfSwapped (E : Env) (px pz dz u v₁ v₂ v₃ : ℕ)
    (xo xi zo zi xo' xi' zo' zi' iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hxo : xo.Perm (u :: v₁ :: xo')) (hxi : xi.Perm xi')
    (hz : (zo ++ zi).Perm (u :: v₃ :: (zo' ++ zi'))) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.X, xo, xi, dz, px⟩, ⟨.Z, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.X, xo', xi', dz, px⟩, ⟨.Z, zo', zi', dz, pz⟩]
      [] iface 1 k := by
  intro τ _
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd
  obtain ⟨⟨h01, h02, h03⟩, ⟨h12, h13⟩, h23, -⟩ := hnd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hdim
  obtain ⟨du, d1, d2, d3⟩ := hdim
  set d := E.dimv dz
  set R := (zo' ++ zi').map τ
  set XO := xo'.map τ
  set XI := xi'.map τ
  have hprod : ∀ ρ : ℕ → ℕ, (∀ x ∈ zo' ++ zi' ++ xo' ++ xi', ρ x = τ x) →
      prodVal E [⟨.X, xo, xi, dz, px⟩, ⟨.Z, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tX d (E.phase px) (ρ u :: ρ v₁ :: XO) XI *
        (tZ d (E.phase pz) (ρ u :: ρ v₃ :: R) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZ_perm _ _ _ _ (hz.map ρ), tX_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
    have e1 : (zo' ++ zi').map ρ = R := List.map_congr_left fun x hx =>
      hρ x (by simp only [List.mem_append] at hx ⊢; tauto)
    have e2 : xo'.map ρ = XO := List.map_congr_left fun x hx =>
      hρ x (by simp only [List.mem_append] at hx ⊢; tauto)
    have e3 : xi'.map ρ = XI := List.map_congr_left fun x hx =>
      hρ x (by simp only [List.mem_append] at hx ⊢; tauto)
    simp only [List.map_cons, e1, e2, e3]
    rfl
  rw [one_mul]
  simp only [sumOver, labDim_of E _ dz du, labDim_of E _ dz d1, labDim_of E _ dz d2,
    labDim_of E _ dz d3]
  rw [Finset.sum_congr rfl fun ju _ => Finset.sum_congr rfl fun j₁ _ =>
    Finset.sum_congr rfl fun j₂ _ => Finset.sum_congr rfl fun j₃ _ => hprod _ fun x hx => by
      have := hfree x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at this
      simp [Function.update_apply, this.1, this.2.1, this.2.2.1, this.2.2.2]]
  simp only [Function.update_apply, h01, h02, h03, h12, h13, h23, Ne.symm h01, Ne.symm h02,
    Ne.symm h03, Ne.symm h12, Ne.symm h13, Ne.symm h23, if_true, if_false]
  have hA : ∀ ju ∈ range d, ∀ j₁ ∈ range d, ∑ j₂ ∈ range d, ∑ j₃ ∈ range d,
      tX d (E.phase px) (ju :: j₁ :: XO) XI * (tZ d (E.phase pz) (ju :: j₃ :: R) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if (j₁ + ju) % d = 0 then tX d (E.phase px) (ju :: j₁ :: XO) XI *
        tZ d (E.phase pz) (ju :: R) else 0 := by
    intro ju hju j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tX d (E.phase px) (ju :: j₁ :: XO) XI *
        (tZ d (E.phase pz) (ju :: j₃ :: R) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        if j₃ = ju then tX d (E.phase px) (ju :: j₁ :: XO) XI * tZ d (E.phase pz) (ju :: R) *
          (if (j₃ + j₁) % d = 0 then 1 else 0) else 0 := by
      intro j₃ _
      rw [← fou_sq d hd, Finset.mul_sum, tZ_cons_cons]
      split_ifs with h
      · exact Finset.sum_congr rfl fun j₂ _ => by ring
      · simp
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos hju, add_comm ju j₁]
    split_ifs <;> simp
  rw [Finset.sum_congr rfl fun ju hju => Finset.sum_congr rfl (hA ju hju)]
  have hC : ∀ ju ∈ range d, ∑ j₁ ∈ range d, (if (j₁ + ju) % d = 0 then
      tX d (E.phase px) (ju :: j₁ :: XO) XI * tZ d (E.phase pz) (ju :: R) else 0) =
      tZ d (E.phase pz) (ju :: R) * ((d : ℂ)⁻¹ * tX d (E.phase px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tX_antipode_out d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
    ring
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZ_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring

end ArchytasZX
