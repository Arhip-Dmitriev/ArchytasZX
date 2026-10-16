/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Connective

/-!
# Z into W bialgebra
-/

namespace ArchytasZX

open Complex Finset

theorem sumOver_zero (D : ℕ → ℕ) (ls : List ℕ) (σ : ℕ → ℕ) :
    sumOver D ls (fun _ => 0) σ = 0 := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l rest ih => simp only [sumOver, ih, Finset.sum_const_zero]

/-- Summing labels pinned to labels outside them sets each to its pin. -/
theorem sumOver_pin (D : ℕ → ℕ) (ls : List ℕ) (hnd : ls.Nodup) (f : ℕ → ℕ)
    (hf : ∀ l ∈ ls, f l ∉ ls) (g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ)
    (hr : ∀ l ∈ ls, σ (f l) < D l) :
    sumOver D ls (fun τ => if ∀ l ∈ ls, τ l = τ (f l) then g τ else 0) σ =
      g (fun x => if x ∈ ls then σ (f x) else σ x) := by
  induction ls generalizing σ with
  | nil => simp [sumOver]
  | cons l rest ih =>
    simp only [List.nodup_cons] at hnd
    simp only [sumOver]
    have hfl : f l ∉ l :: rest := hf l List.mem_cons_self
    have hfl' : f l ≠ l := fun h => hfl (by rw [h]; exact List.mem_cons_self)
    have hflr : f l ∉ rest := fun h => hfl (List.mem_cons_of_mem _ h)
    have inner : ∀ j ∈ range (D l), sumOver D rest (fun τ => if ∀ x ∈ l :: rest,
        τ x = τ (f x) then g τ else 0) (Function.update σ l j) =
        if j = σ (f l) then sumOver D rest (fun τ => if ∀ x ∈ rest, τ x = τ (f x) then g τ
          else 0) (Function.update σ l j) else 0 := by
      intro j _
      have key : ∀ τ : ℕ → ℕ, (∀ x ∉ rest, τ x = Function.update σ l j x) →
          τ l = j ∧ τ (f l) = σ (f l) := fun τ hτ =>
        ⟨by rw [hτ l hnd.1, Function.update_self],
          by rw [hτ (f l) hflr, Function.update_of_ne hfl']⟩
      split_ifs with hj
      · refine sumOver_congr _ _ _ _ _ fun τ hτ _ => ?_
        obtain ⟨h1, h2⟩ := key τ hτ
        simp only [List.mem_cons, forall_eq_or_imp, h1, h2, hj, true_and]
      · refine (sumOver_congr D rest _ (fun _ => 0) _ fun τ hτ _ => ?_).trans
          (sumOver_zero D rest _)
        obtain ⟨h1, h2⟩ := key τ hτ
        have : ¬ ∀ x ∈ l :: rest, τ x = τ (f x) := fun h => hj (by
          have := h l List.mem_cons_self; rwa [h1, h2] at this)
        rw [if_neg this]
    rw [Finset.sum_congr rfl inner, Finset.sum_ite_eq', if_pos (Finset.mem_range.2
      (hr l List.mem_cons_self))]
    have hfx : ∀ x ∈ rest, f x ≠ l := fun x hx e =>
      hf x (List.mem_cons_of_mem _ hx) (by rw [e]; exact List.mem_cons_self)
    rw [ih hnd.2 (fun x hx h => hf x (List.mem_cons_of_mem _ hx) (List.mem_cons_of_mem _ h))
      _ (fun x hx => by rw [Function.update_of_ne (hfx x hx)]; exact hr x (List.mem_cons_of_mem _ hx))]
    congr 1
    funext x
    by_cases hx : x = l
    · subst hx; simp [hnd.1]
    · by_cases hm : x ∈ rest
      · simp [hm, hx, Function.update_of_ne (hfx x hm)]
      · simp [hm, hx]

theorem tW_mul (d : ℕ) (os : List ℕ) (a b : ℕ) :
    tW d os a * tW d os b = if a = b then tW d os a else 0 := by
  simp only [tW_eq]
  split_ifs <;> simp_all

theorem tW_prod (d : ℕ) (os vs : List ℕ) (v₀ : ℕ) :
    tW d os v₀ * (vs.map (tW d os)).prod = if ∀ v ∈ vs, v = v₀ then tW d os v₀ else 0 := by
  induction vs with
  | nil => simp
  | cons v vs ih =>
    rw [List.map_cons, List.prod_cons, ← mul_assoc, tW_mul]
    by_cases h : v₀ = v
    · subst h
      rw [if_pos rfl, ih]
      simp
    · rw [if_neg h, zero_mul]
      have : ¬ ∀ x ∈ v :: vs, x = v₀ := fun h' => h (h' v List.mem_cons_self).symm
      rw [if_neg this]

theorem zip_map_fst {α β γ : Type*} (f : α → γ) (l₁ : List α) (l₂ : List β)
    (h : l₁.length = l₂.length) : (l₁.zip l₂).map (fun p => f p.1) = l₁.map f := by
  induction l₁ generalizing l₂ with
  | nil => simp
  | cons a l₁ ih =>
    cases l₂ with
    | nil => simp at h
    | cons b l₂ => simp [ih l₂ (by simpa using h)]

theorem wzBialgebra (E : Env) (pz q dz w : ℕ) (zi wo iface : List ℕ)
    (rows cols : List (List ℕ)) (pin : List (ℕ × ℕ)) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hzi : zi ≠ []) (hwz : w ∉ zi) (hwo : w ∉ wo)
    (hrange : ∀ x ∈ zi ++ wo, x ∈ iface ∧ labIdx E.labs x = dz)
    (hpin : rows.flatten = pin.map Prod.fst) (hnd : rows.flatten.Nodup)
    (hto : ∀ p ∈ pin, p.2 ∈ wo) (hdisj : ∀ x ∈ zi ++ wo ++ [w], x ∉ rows.flatten)
    (hcol : ((wo.zip cols).flatMap fun p => p.2.map fun e => (e, p.1)).Perm pin)
    (hrow : ∀ r ∈ rows, r.map (ren pin) = wo) (hlen : zi.length = rows.length)
    (hcl : wo.length = cols.length) (hdims : ∀ e ∈ rows.flatten, labIdx E.labs e = dz)
    (hpz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hq : ∀ k < E.dimv dz, ex (E.phase q k) = 1) :
    LocalEq E [⟨.Z, [w], zi, dz, pz⟩, ⟨.W, wo, [w], dz, 0⟩] [w]
      ((zi.zip rows).map (fun p => ⟨.W, p.2, [p.1], dz, 0⟩) ++
        (wo.zip cols).map (fun p => ⟨.Z, [p.1], p.2, dz, q⟩)) rows.flatten iface 1 1 := by
  intro σ hσ
  have hr : ∀ x ∈ zi ++ wo, σ x < E.dimv dz := fun x hx => by
    have := hσ x (hrange x hx).1; rwa [labDim_of E x dz (hrange x hx).2] at this
  set d := E.dimv dz
  obtain ⟨h₀, zt, rfl⟩ : ∃ h₀ zt, zi = h₀ :: zt := List.exists_cons_of_ne_nil hzi
  -- the pin is a function on the summed labels
  have hkeys : (pin.map Prod.fst).Nodup := hpin ▸ hnd
  have hpinv : ∀ p ∈ pin, ren pin p.1 = p.2 := by
    intro p hp
    clear hcol hrow hpin hto
    induction pin with
    | nil => simp at hp
    | cons a pin ih =>
      obtain ⟨a₁, a₂⟩ := a
      simp only [List.map_cons, List.nodup_cons, List.mem_map] at hkeys
      rcases List.mem_cons.1 hp with rfl | hp'
      · simp [ren]
      · have : p.1 ≠ a₁ := fun e => hkeys.1 ⟨p, hp', e⟩
        simp only [ren, this, if_false]; exact ih hkeys.2 hp'
  -- right side
  have hZ : ∀ τ : ℕ → ℕ, (∀ x ∈ wo, τ x < d) →
      prodVal E ((wo.zip cols).map (fun p => ⟨.Z, [p.1], p.2, dz, q⟩)) τ =
      if ∀ l ∈ rows.flatten, τ l = τ (ren pin l) then 1 else 0 := by
    intro τ hτ
    have e1 : prodVal E ((wo.zip cols).map (fun p => ⟨.Z, [p.1], p.2, dz, q⟩)) τ =
        ((wo.zip cols).map fun p => if ∀ e ∈ p.2, τ e = τ p.1 then (1 : ℂ) else 0).prod := by
      simp only [prodVal, List.map_map]
      congr 1
      refine List.map_congr_left fun p hp => ?_
      have ho : p.1 ∈ wo := (List.of_mem_zip hp).1
      simp only [Function.comp, val, List.map_cons, List.map_nil, List.cons_append,
        List.nil_append]
      rw [tZ_at _ _ _ (τ p.1) List.mem_cons_self, hq _ (hτ _ ho)]
      simp only [List.mem_cons, List.mem_map, forall_eq_or_imp, true_and, forall_exists_index,
        and_imp, forall_apply_eq_imp_iff₂, show τ p.1 < E.dimv dz from hτ _ ho]
    rw [e1]
    have e2 : ∀ (l : List (ℕ × List ℕ)), (l.map fun p => if ∀ e ∈ p.2, τ e = τ p.1 then
        (1 : ℂ) else 0).prod = if ∀ p ∈ l, ∀ e ∈ p.2, τ e = τ p.1 then 1 else 0 := by
      intro l
      induction l with
      | nil => simp
      | cons p l ih =>
        simp only [List.map_cons, List.prod_cons, ih, List.mem_cons, forall_eq_or_imp]
        split_ifs <;> first | rfl | (exfalso; tauto) | simp
    rw [e2]
    congr 1
    apply propext
    have hmem : ∀ e o, (e, o) ∈ pin ↔ ∃ p ∈ wo.zip cols, e ∈ p.2 ∧ o = p.1 := by
      intro e o
      rw [← hcol.mem_iff]
      simp only [List.mem_flatMap, List.mem_map, Prod.mk.injEq]
      constructor
      · rintro ⟨p, hp, e', he, rfl, rfl⟩; exact ⟨p, hp, he, rfl⟩
      · rintro ⟨p, hp, he, rfl⟩; exact ⟨p, hp, e, he, rfl, rfl⟩
    constructor
    · intro h l hl
      rw [hpin] at hl
      obtain ⟨⟨e, o⟩, hp, rfl⟩ := List.mem_map.1 hl
      rw [hpinv _ hp]
      obtain ⟨p, hpz, he, rfl⟩ := (hmem e o).1 hp
      exact h p hpz e he
    · intro h p hp e he
      have hin : (e, p.1) ∈ pin := (hmem e p.1).2 ⟨p, hp, he, rfl⟩
      have := h e (hpin ▸ List.mem_map_of_mem hin)
      rwa [hpinv _ hin] at this
  have hW : ∀ τ : ℕ → ℕ, prodVal E (List.map (fun p => (⟨.W, p.2, [p.1], dz, 0⟩ : Node)) ((h₀ :: zt).zip rows)) τ =
      (((h₀ :: zt).zip rows).map fun p => tW d (p.2.map τ) (τ p.1)).prod := by
    intro τ
    simp only [prodVal, List.map_map]
    rfl
  simp only [one_mul]
  have hfree : ∀ l ∈ rows.flatten, ren pin l ∉ rows.flatten := by
    intro l hl
    rw [hpin] at hl
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hl
    rw [hpinv _ hp]
    exact hdisj _ (List.mem_append_left _ (List.mem_append_right _ (hto p hp)))
  have hpos : ∀ l ∈ rows.flatten, σ (ren pin l) < labDim E l := by
    intro l hl
    rw [labDim_of E l dz (hdims l hl)]
    rw [hpin] at hl
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hl
    rw [hpinv _ hp]
    exact hr _ (List.mem_append_right _ (hto p hp))
  have rhs : sumOver (labDim E) rows.flatten (prodVal E (List.map (fun p => (⟨.W, p.2, [p.1], dz, 0⟩ : Node))
      ((h₀ :: zt).zip rows) ++ (wo.zip cols).map (fun p => ⟨.Z, [p.1], p.2, dz, q⟩))) σ =
      (((h₀ :: zt).zip rows).map fun p => tW d (wo.map σ) (σ p.1)).prod := by
    have step : sumOver (labDim E) rows.flatten (prodVal E (List.map (fun p => (⟨.W, p.2, [p.1], dz, 0⟩ : Node))
        ((h₀ :: zt).zip rows) ++ (wo.zip cols).map
          (fun p => ⟨.Z, [p.1], p.2, dz, q⟩))) σ =
        sumOver (labDim E) rows.flatten (fun τ => if ∀ l ∈ rows.flatten, τ l = τ (ren pin l)
          then (((h₀ :: zt).zip rows).map fun p => tW d (p.2.map τ) (τ p.1)).prod else 0) σ := by
      refine sumOver_congr _ _ _ _ _ fun τ hτ _ => ?_
      have hwo : ∀ x ∈ wo, τ x < d := fun x hx => by
        rw [hτ x (fun h => hdisj x (by simp [hx]) h)]
        exact hr x (List.mem_append_right _ hx)
      rw [prodVal_append, hZ τ hwo, hW τ]
      split_ifs <;> simp
    rw [step, sumOver_pin _ _ hnd _ hfree _ _ hpos]
    congr 1
    refine List.map_congr_left fun p hp => ?_
    have hrow' := hrow p.2 (List.of_mem_zip hp).2
    have hp1 : p.1 ∉ rows.flatten := hdisj p.1
      (List.mem_append_left _ (List.mem_append_left _ (List.of_mem_zip hp).1))
    simp only [hp1, if_false]
    congr 1
    rw [← hrow', List.map_map]
    refine List.map_congr_left fun e he => ?_
    have : e ∈ rows.flatten := List.mem_flatten.2 ⟨p.2, (List.of_mem_zip hp).2, he⟩
    simp [this]
  rw [rhs, zip_map_fst (fun i => tW d (wo.map σ) (σ i)) _ _ hlen]
  -- left side
  have hwz' : w ∉ h₀ :: zt := hwz
  simp only [sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, val, List.cons_append, List.nil_append, labDim_of E w dz hw, Function.update_self,
    map_update_absent σ w _ _ hwz', map_update_absent σ w _ wo hwo]
  have hh₀ : σ h₀ < d := hr h₀ (by simp)
  have lhs : ∀ j ∈ range d, tZ d (E.phase pz) (j :: σ h₀ :: zt.map σ) * tW d (wo.map σ) j =
      if j = σ h₀ then (if ∀ x ∈ zt, σ x = σ h₀ then tW d (wo.map σ) (σ h₀) else 0) else 0 := by
    intro j hj
    rw [tZ_at _ _ _ j List.mem_cons_self, if_congr (show (j < d ∧ ∀ v ∈ j :: σ h₀ :: zt.map σ,
      v = j) ↔ (j = σ h₀ ∧ ∀ x ∈ zt, σ x = σ h₀) by
        simp only [List.mem_cons, List.mem_map, forall_eq_or_imp, true_and,
          forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, Finset.mem_range.1 hj]
        constructor
        · rintro ⟨h1, h2⟩; exact ⟨h1.symm, fun x hx => (h2 x hx).trans h1.symm⟩
        · rintro ⟨h1, h2⟩; exact ⟨h1.symm, fun x hx => (h2 x hx).trans h1.symm⟩) rfl rfl]
    split_ifs with h1 h2 <;> simp_all [hpz _ hh₀]
  rw [Finset.sum_congr rfl lhs, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hh₀)]
  rw [show (zt.map fun i => tW d (wo.map σ) (σ i)) = (zt.map σ).map (tW d (wo.map σ)) by
    simp [List.map_map, Function.comp_def], tW_prod]
  simp

end ArchytasZX
