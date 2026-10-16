/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Iso

/-!
# Rule lemmas

Each rule of the ArchytasZX library as a `LocalEq` between its two sides, for every leg
labelling the engine can produce, plus `wire_elim` removing a bare wire.
-/

namespace ArchytasZX

open Complex Finset

/-! ## Facts about the generators -/

theorem ex_zero : ex 0 = 1 := by simp [ex]

theorem ex_add (a b : ℝ) : ex (a + b) = ex a * ex b := by
  simp only [ex, ← Complex.exp_add]; congr 1; push_cast; ring

theorem ex_int (a : ℝ) (z : ℤ) : ex (a + z) = ex a := by
  rw [ex_add]
  have : ex z = 1 := by
    simp only [ex]
    rw [Complex.exp_eq_one_iff]
    exact ⟨z, by push_cast; ring⟩
  rw [this, mul_one]

theorem ex_congr_int {x y : ℝ} (h : ∃ z : ℤ, x = y + z) : ex x = ex y := by
  obtain ⟨z, rfl⟩ := h; exact ex_int y z

theorem om_prim (d : ℕ) (hd : 0 < d) : IsPrimitiveRoot (om d) d := by
  have := Complex.isPrimitiveRoot_exp d hd.ne'
  simpa [om] using this

theorem conj_om (d : ℕ) : (starRingEnd ℂ) (om d) = (om d)⁻¹ := by
  simp only [om, ← Complex.exp_conj, ← Complex.exp_neg]
  congr 1
  simp [map_div₀, Complex.conj_ofReal, map_ofNat]
  ring

theorem conj_sqrt (d : ℕ) : (starRingEnd ℂ) (Real.sqrt d : ℂ) = Real.sqrt d :=
  Complex.conj_ofReal _

theorem sqrt_mul_sqrt (d : ℕ) : (Real.sqrt d : ℂ) * Real.sqrt d = d := by
  rw [← Complex.ofReal_mul, Real.mul_self_sqrt (Nat.cast_nonneg d)]; simp

/-- The DFT is unitary. -/
theorem fou_unitary (d : ℕ) (hd : 0 < d) (k k' : ℕ) (hk : k < d) (hk' : k' < d) :
    ∑ j ∈ range d, fou d j k * (starRingEnd ℂ) (fou d j k') = if k = k' then 1 else 0 := by
  have hp := om_prim d hd
  have hz : om d ≠ 0 := hp.ne_zero hd.ne'
  have hs : (Real.sqrt d : ℂ) ≠ 0 := by
    have : (0 : ℝ) < Real.sqrt d := Real.sqrt_pos.2 (by exact_mod_cast hd)
    exact_mod_cast this.ne'
  set u := om d ^ k * (om d ^ k')⁻¹
  have hterm : ∀ j, fou d j k * (starRingEnd ℂ) (fou d j k') = u ^ j / d := by
    intro j
    simp only [fou, map_div₀, map_pow, conj_om, conj_sqrt, u, mul_pow, inv_pow, ← pow_mul]
    rw [← sqrt_mul_sqrt d]
    field_simp
    ring
  simp only [hterm, ← Finset.sum_div]
  have hdc : (d : ℂ) ≠ 0 := by exact_mod_cast hd.ne'
  split_ifs with h
  · subst h
    have : u = 1 := by simp [u, hz]
    simp [this, hdc]
  · have hu : u ≠ 1 := by
      intro h1
      apply h
      have : om d ^ k = om d ^ k' := by
        have hk0 : om d ^ k' ≠ 0 := pow_ne_zero _ hz
        exact (mul_inv_eq_one₀ hk0).1 h1
      exact hp.pow_inj hk hk' this
    have hud : u ^ d = 1 := by
      simp only [u, mul_pow, inv_pow, ← pow_mul]
      rw [mul_comm k d, mul_comm k' d, pow_mul, pow_mul, hp.pow_eq_one]; simp
    rw [geom_sum_eq hu, hud]; simp

/-- The DFT is symmetric. -/
theorem fou_symm (d j k : ℕ) : fou d j k = fou d k j := by simp [fou, mul_comm]

/-- The Z spider at a value one of its legs takes. -/
theorem tZ_at (d : ℕ) (φ : ℕ → ℝ) (vs : List ℕ) (j : ℕ) (hj : j ∈ vs) :
    tZ d φ vs = if j < d ∧ ∀ v ∈ vs, v = j then ex (φ j) else 0 := by
  unfold tZ
  have : ∀ k ∈ range d, (if ∀ v ∈ vs, v = k then ex (φ k) else 0) =
      if k = j then (if ∀ v ∈ vs, v = j then ex (φ j) else 0) else 0 := by
    intro k _
    by_cases hk : k = j
    · subst hk; simp
    · have : ¬ ∀ v ∈ vs, v = k := fun h => hk (h j hj).symm
      simp [this, hk]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq']
  simp only [Finset.mem_range]
  by_cases h : j < d <;> simp [h]

theorem tZ_perm (d : ℕ) (φ : ℕ → ℝ) (a b : List ℕ) (h : a.Perm b) : tZ d φ a = tZ d φ b := by
  unfold tZ
  refine Finset.sum_congr rfl fun k _ => ?_
  have : (∀ v ∈ a, v = k) ↔ (∀ v ∈ b, v = k) :=
    ⟨fun h' v hv => h' v (h.mem_iff.2 hv), fun h' v hv => h' v (h.mem_iff.1 hv)⟩
  simp only [this]

/-! ## Bare wires -/

theorem wire_elim (E : Env) (s : ℂ) (ns₁ ns₂ : List Node) (int bnd : List ℕ) (a b dz : ℕ)
    (hb : b ∈ int) (hab : a ≠ b) (ha : a ∈ int ∨ a ∈ bnd)
    (hda : labIdx E.labs a = dz) (hdb : labIdx E.labs b = dz) :
    Equal E s ⟨ns₁ ++ ⟨.I, [a], [b], dz, 0⟩ :: ns₂, int, bnd⟩
      s ⟨(ns₁ ++ ns₂).map (Node.rename (ren [(b, a)])), int.erase b, bnd⟩ := by
  refine ⟨List.Perm.refl _, fun σ hσ => ?_⟩
  congr 1
  simp only [eval]
  have hD : ∀ l, labIdx E.labs l = dz → labDim E l = E.dimv dz := by
    intro l h; simp only [labDim]; simp only [labIdx] at h; rw [h]
  have hperm : int.Perm (int.erase b ++ [b]) :=
    (List.perm_cons_erase hb).trans (List.perm_append_comm (l₁ := [b]))
  rw [sumOver_perm _ hperm, sumOver_append]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  have hmid : prodVal E (ns₁ ++ ⟨.I, [a], [b], dz, 0⟩ :: ns₂) =
      fun ρ => tI (E.dimv dz) (ρ a) (ρ b) * prodVal E (ns₁ ++ ns₂) ρ := by
    funext ρ
    have := (List.perm_middle (a := (⟨.I, [a], [b], dz, 0⟩ : Node)) (l₁ := ns₁)
      (l₂ := ns₂)).map (val E ρ)
    simp only [prodVal, this.prod_eq, List.map_cons, List.prod_cons]
    rfl
  rw [hmid]
  simp only [sumOver, hD b hdb, Function.update_of_ne hab, Function.update_self]
  have hta : τ a < E.dimv dz := by
    rw [← hD a hda]
    by_cases hm : a ∈ int.erase b
    · exact hr a hm
    · rw [hτ a hm]
      rcases ha with ha | ha
      · exact absurd ((List.mem_erase_of_ne hab).2 ha) hm
      · exact hσ a ha
  simp only [tI]
  have : ∀ j ∈ range (E.dimv dz), (if τ a = j ∧ τ a < E.dimv dz then (1 : ℂ) else 0) *
      prodVal E (ns₁ ++ ns₂) (Function.update τ b j) =
      if τ a = j then prodVal E (ns₁ ++ ns₂) (Function.update τ b (τ a)) else 0 := by
    intro j _
    by_cases h : τ a = j
    · subst h; simp [hta]
    · simp [h]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq]
  simp only [Finset.mem_range, hta, ite_true]
  rw [prodVal_rename]
  congr 1
  funext x
  simp only [Function.comp, ren, Function.update]
  by_cases hx : x = b
  · subst hx; simp
  · simp [hx]

/-! ## Spider fusion -/

theorem fusionZ (E : Env) (a b c dz w : ℕ) (o₁ i₁ o₂ i₂ o₃ i₃ iface : List ℕ)
    (h₁ : w ∈ o₁ ++ i₁) (h₂ : w ∈ o₂ ++ i₂)
    (hm : (o₃ ++ i₃).Perm ((o₁ ++ i₁ ++ (o₂ ++ i₂)).filter (· ≠ w)))
    (hw : labIdx E.labs w = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase c k) = ex (E.phase a k) * ex (E.phase b k)) :
    LocalEq E [⟨.Z, o₁, i₁, dz, a⟩, ⟨.Z, o₂, i₂, dz, b⟩] [w] [⟨.Z, o₃, i₃, dz, c⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, ← List.map_append]
  have hD : labDim E w = E.dimv dz := by
    simp only [labDim]; simp only [labIdx] at hw; rw [hw]
  rw [hD, tZ_perm _ _ _ _ (hm.map τ)]
  set d := E.dimv dz
  set rest := (o₁ ++ i₁ ++ (o₂ ++ i₂)).filter (· ≠ w)
  have key : ∀ j < d, ∀ (l : List ℕ), w ∈ l → ((∀ v ∈ l.map (Function.update τ w j), v = j) ↔
      ∀ x ∈ l, ¬x = w → τ x = j) := by
    intro j _ l _
    simp only [List.mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    refine ⟨fun h x hx hxw => ?_, fun h x hx => ?_⟩
    · have := h x hx; rwa [Function.update_of_ne hxw] at this
    · by_cases hxw : x = w
      · subst hxw; simp
      · rw [Function.update_of_ne hxw]; exact h x hx hxw
  have step : ∀ j ∈ range d, tZ d (E.phase a) ((o₁ ++ i₁).map (Function.update τ w j)) *
      tZ d (E.phase b) ((o₂ ++ i₂).map (Function.update τ w j)) =
      if ∀ v ∈ rest.map τ, v = j then ex (E.phase c j) else 0 := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    rw [tZ_at _ _ _ j (List.mem_map.2 ⟨w, h₁, by simp⟩),
      tZ_at _ _ _ j (List.mem_map.2 ⟨w, h₂, by simp⟩)]
    simp only [key j hj' _ h₁, key j hj' _ h₂]
    have hr : (∀ v ∈ rest.map τ, v = j) ↔
        (∀ x ∈ o₁ ++ i₁, ¬x = w → τ x = j) ∧ (∀ x ∈ o₂ ++ i₂, ¬x = w → τ x = j) := by
      constructor
      · intro h
        refine ⟨fun x hx hxw => h (τ x) (List.mem_map_of_mem (List.mem_filter.2
          ⟨List.mem_append_left _ hx, by simpa using hxw⟩)), fun x hx hxw => h (τ x)
          (List.mem_map_of_mem (List.mem_filter.2 ⟨List.mem_append_right _ hx,
            by simpa using hxw⟩))⟩
      · rintro ⟨h1, h2⟩ v hv
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv
        obtain ⟨hx, hxw⟩ := List.mem_filter.1 hx
        have hxw : x ≠ w := by simpa using hxw
        rcases List.mem_append.1 hx with hx | hx
        · exact h1 x hx hxw
        · exact h2 x hx hxw
    simp only [hr, hph j hj']
    split_ifs <;> first | rfl | (exfalso; tauto) | simp
  rw [Finset.sum_congr rfl step]
  rfl

/-- Setting an absent label changes no leg value. -/
theorem map_update_absent (τ : ℕ → ℕ) (w j : ℕ) (l : List ℕ) (h : w ∉ l) :
    l.map (Function.update τ w j) = l.map τ :=
  List.map_congr_left fun x hx => Function.update_of_ne (by rintro rfl; exact h hx) _ _

/-- The product over the legs of `l` under `τ` with `w` set to `j`, `w` occurring once. -/
theorem prod_update_once {M : Type*} [CommMonoid M] (g : ℕ → M) (τ : ℕ → ℕ) (w j : ℕ)
    (l : List ℕ) (h : w ∈ l) (hn : w ∉ l.erase w) :
    ((l.map (Function.update τ w j)).map g).prod = g j * (((l.erase w).map τ).map g).prod := by
  have hp := (List.perm_cons_erase h).map (Function.update τ w j)
  rw [(hp.map g).prod_eq, List.map_cons, List.map_cons, List.prod_cons, Function.update_self]
  rw [map_update_absent τ w j (l.erase w) hn]

theorem fusionX (E : Env) (a b c dz w : ℕ) (o₁ i₁ o₂ i₂ o₃ i₃ iface : List ℕ)
    (hd : 0 < E.dimv dz) (h₁ : w ∈ o₁) (h₁' : w ∉ o₁.erase w) (h₁i : w ∉ i₁)
    (h₂ : w ∈ i₂) (h₂' : w ∉ i₂.erase w) (h₂o : w ∉ o₂)
    (ho : o₃.Perm (o₁.erase w ++ o₂)) (hi : i₃.Perm (i₁ ++ i₂.erase w))
    (hw : labIdx E.labs w = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase c k) = ex (E.phase a k) * ex (E.phase b k)) :
    LocalEq E [⟨.X, o₁, i₁, dz, a⟩, ⟨.X, o₂, i₂, dz, b⟩] [w] [⟨.X, o₃, i₃, dz, c⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val]
  have hD : labDim E w = E.dimv dz := by
    simp only [labDim]; simp only [labIdx] at hw; rw [hw]
  rw [hD]
  set d := E.dimv dz
  set F : ℕ → ℕ → ℂ := fun x k => fou d x k
  set G : ℕ → ℕ → ℂ := fun x k => (starRingEnd ℂ) (fou d x k)
  set A : ℕ → ℂ := fun k => ex (E.phase a k) * (((o₁.erase w).map τ).map (F · k)).prod *
    ((i₁.map τ).map (G · k)).prod
  set B : ℕ → ℂ := fun k => ex (E.phase b k) * ((o₂.map τ).map (F · k)).prod *
    (((i₂.erase w).map τ).map (G · k)).prod
  have hl : ∀ j, tX d (E.phase a) (o₁.map (Function.update τ w j))
      (i₁.map (Function.update τ w j)) = ∑ k ∈ range d, F j k * A k := by
    intro j
    unfold tX
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [map_update_absent τ w j i₁ h₁i, prod_update_once (F · k) τ w j o₁ h₁ h₁']
    simp only [A]; ring
  have hr : ∀ j, tX d (E.phase b) (o₂.map (Function.update τ w j))
      (i₂.map (Function.update τ w j)) = ∑ k ∈ range d, G j k * B k := by
    intro j
    unfold tX
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [map_update_absent τ w j o₂ h₂o, prod_update_once (G · k) τ w j i₂ h₂ h₂']
    simp only [B]; ring
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
  rw [((ho.map τ).map (F · k)).prod_eq, ((hi.map τ).map (G · k)).prod_eq,
    hph k (Finset.mem_range.1 hk)]
  simp only [A, B, List.map_append, List.prod_append, F, G]
  ring

/-! ## Identity removal -/

theorem identityZ (E : Env) (a dz p q w x : ℕ) (iface : List ℕ)
    (hpq : (p = w ∧ q = x) ∨ (p = x ∧ q = w))
    (hph : ∀ k < E.dimv dz, ex (E.phase a k) = 1) :
    LocalEq E [⟨.Z, [p], [q], dz, a⟩] [] [⟨.I, [x], [w], dz, 0⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, List.cons_append, List.nil_append]
  rw [tZ_at _ _ _ (τ p) (by simp)]
  simp only [tI, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
    true_and]
  rcases hpq with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> split_ifs <;>
    first | rfl | (exfalso; omega) | exact hph _ (by omega)

theorem identityX (E : Env) (a dz p q w x : ℕ) (iface : List ℕ)
    (hd : 0 < E.dimv dz) (hpq : (p = w ∧ q = x) ∨ (p = x ∧ q = w))
    (hface : p ∈ iface ∧ q ∈ iface) (hp : labIdx E.labs p = dz) (hq : labIdx E.labs q = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase a k) = 1) :
    LocalEq E [⟨.X, [p], [q], dz, a⟩] [] [⟨.I, [x], [w], dz, 0⟩] [] iface 1 1 := by
  intro τ hτ
  have hD : ∀ l, labIdx E.labs l = dz → labDim E l = E.dimv dz := by
    intro l h; simp only [labDim]; simp only [labIdx] at h; rw [h]
  have rp := hτ p hface.1
  have rq := hτ q hface.2
  rw [hD p hp] at rp
  rw [hD q hq] at rq
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, tX]
  have : ∑ k ∈ range (E.dimv dz), ex (E.phase a k) * fou (E.dimv dz) (τ p) k *
      (starRingEnd ℂ) (fou (E.dimv dz) (τ q) k) =
      ∑ k ∈ range (E.dimv dz), fou (E.dimv dz) k (τ p) *
      (starRingEnd ℂ) (fou (E.dimv dz) k (τ q)) := by
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [hph k (Finset.mem_range.1 hk), fou_symm _ (τ p), fou_symm _ (τ q), one_mul]
  rw [this, fou_unitary _ hd _ _ rp rq]
  simp only [tI]
  rcases hpq with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> split_ifs <;> first | rfl | (exfalso; omega)

end ArchytasZX
