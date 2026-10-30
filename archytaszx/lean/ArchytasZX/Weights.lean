/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net
import ArchytasZX.Rules
import ArchytasZX.Closed
import ArchytasZX.ZXRules
import ArchytasZX.Completeness

/-!
# Weighted Z spiders

Rule lemmas stated for Z and weighted Z spiders alike, through their coefficient vectors.
-/

namespace ArchytasZX

open Complex Finset

/-- A Z or weighted Z spider's coefficient at `k`. -/
noncomputable def zc (E : Env) (n : Node) (k : ℕ) : ℂ :=
  match n.kind with
  | .Z => 1 * ex (E.phase n.ph k)
  | .ZW => E.coef n.ph k
  | _ => 0

/-- A Z or weighted Z spider. -/
def isZc (n : Node) : Bool := n.kind == .Z || n.kind == .ZW

theorem val_zc (E : Env) (σ : ℕ → ℕ) (n : Node) (h : isZc n = true) :
    val E σ n = tZw (E.dimv n.dim) (zc E n) ((n.outs ++ n.ins).map σ) := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only [isZc, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with rfl | rfl <;> simp only [val, zc, tZ, tZw, one_mul] <;> rw [List.map_append]

theorem tZw_at (d : ℕ) (c : ℕ → ℂ) (vs : List ℕ) (j : ℕ) (hj : j ∈ vs) :
    tZw d c vs = if j < d ∧ ∀ v ∈ vs, v = j then c j else 0 := by
  unfold tZw
  have : ∀ k ∈ range d, (if ∀ v ∈ vs, v = k then c k else 0) =
      if k = j then (if ∀ v ∈ vs, v = j then c j else 0) else 0 := by
    intro k _
    by_cases hk : k = j
    · subst hk; simp
    · have : ¬ ∀ v ∈ vs, v = k := fun h => hk (h j hj).symm
      simp [this, hk]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq']
  simp only [Finset.mem_range]
  by_cases h : j < d <;> simp [h]

theorem tZw_perm (d : ℕ) (c : ℕ → ℂ) (a b : List ℕ) (h : a.Perm b) : tZw d c a = tZw d c b := by
  unfold tZw
  refine Finset.sum_congr rfl fun k _ => ?_
  have : (∀ v ∈ a, v = k) ↔ (∀ v ∈ b, v = k) :=
    ⟨fun h' v hv => h' v (h.mem_iff.2 hv), fun h' v hv => h' v (h.mem_iff.1 hv)⟩
  simp only [this]

/-- [P1] green circle spider: a spider is any spider of the same legs and coefficients. -/
theorem zcCongr (E : Env) (n m : Node) (iface : List ℕ) (hn : isZc n = true)
    (hm : isZc m = true) (hd : m.dim = n.dim) (hl : (n.outs ++ n.ins).Perm (m.outs ++ m.ins))
    (hc : ∀ k < E.dimv n.dim, zc E m k = zc E n k) :
    LocalEq E [n] [] [m] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  rw [val_zc E τ n hn, val_zc E τ m hm, hd, tZw_perm _ _ _ _ (hl.map τ)]
  unfold tZw
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hc k (Finset.mem_range.1 hk)]

/-- [P1] (S1): two spiders joined by `w` fuse, their coefficients multiplying. -/
theorem zcFusion (E : Env) (na nb m : Node) (dz w : ℕ) (iface : List ℕ) (ha : isZc na = true)
    (hb : isZc nb = true) (hm : isZc m = true) (hda : na.dim = dz) (hdb : nb.dim = dz)
    (hdm : m.dim = dz) (h₁ : w ∈ na.outs ++ na.ins) (h₂ : w ∈ nb.outs ++ nb.ins)
    (hl : (m.outs ++ m.ins).Perm ((na.outs ++ na.ins ++ (nb.outs ++ nb.ins)).filter (· ≠ w)))
    (hw : labIdx E.labs w = dz)
    (hc : ∀ k < E.dimv dz, zc E m k = zc E na k * zc E nb k) :
    LocalEq E [na, nb] [w] [m] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  have hD : labDim E w = E.dimv dz := by
    simp only [labDim]; simp only [labIdx] at hw; rw [hw]
  rw [hD, val_zc E τ m hm, hdm, tZw_perm _ _ _ _ (hl.map τ)]
  set d := E.dimv dz
  set rest := (na.outs ++ na.ins ++ (nb.outs ++ nb.ins)).filter (· ≠ w)
  have key : ∀ j < d, ∀ (l : List ℕ), w ∈ l → ((∀ v ∈ l.map (Function.update τ w j), v = j) ↔
      ∀ x ∈ l, ¬x = w → τ x = j) := by
    intro j _ l _
    simp only [List.mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    refine ⟨fun h x hx hxw => ?_, fun h x hx => ?_⟩
    · have := h x hx; rwa [Function.update_of_ne hxw] at this
    · by_cases hxw : x = w
      · subst hxw; simp
      · rw [Function.update_of_ne hxw]; exact h x hx hxw
  have step : ∀ j ∈ range d, val E (Function.update τ w j) na * val E (Function.update τ w j) nb =
      if ∀ v ∈ rest.map τ, v = j then zc E m j else 0 := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    rw [val_zc _ _ na ha, val_zc _ _ nb hb, hda, hdb,
      tZw_at _ _ _ j (List.mem_map.2 ⟨w, h₁, by simp⟩),
      tZw_at _ _ _ j (List.mem_map.2 ⟨w, h₂, by simp⟩)]
    simp only [key j hj' _ h₁, key j hj' _ h₂]
    have hr : (∀ v ∈ rest.map τ, v = j) ↔
        (∀ x ∈ na.outs ++ na.ins, ¬x = w → τ x = j) ∧
          (∀ x ∈ nb.outs ++ nb.ins, ¬x = w → τ x = j) := by
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
    simp only [hr, hc j hj']
    split_ifs <;> first | rfl | (exfalso; tauto) | simp
  rw [Finset.sum_congr rfl step]
  rfl

/-- A spider with no legs is the sum of its coefficients. -/
theorem zcScalar (E : Env) (n : Node) (iface : List ℕ) (s : ℂ) (hn : isZc n = true)
    (ho : n.outs = []) (hi : n.ins = []) (hs : s = ∑ k ∈ range (E.dimv n.dim), zc E n k) :
    LocalEq E [n] [] [] [] iface 1 s := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  rw [val_zc E τ n hn, ho, hi, hs]
  simp [tZw]

/-- The X state of phase zero at `j < d`: `√d` at `0`, else `0`. -/
theorem tX_zero_state (d j : ℕ) (hd : 0 < d) (hj : j < d) (φ : ℕ → ℝ)
    (hφ : ∀ k < d, ex (φ k) = 1) :
    tX d φ [j] [] = if j = 0 then rt d else 0 := by
  rw [tX_kj d 0 hd φ (fun k hk => by rw [hφ k hk]; simp) [j] []]
  simp only [List.length_cons, List.length_nil, List.sum_cons, List.sum_nil, add_zero,
    Nat.zero_mod, zero_add, pow_one, Nat.mod_eq_of_lt hj]
  by_cases h : j = 0
  · simp only [h, ite_true]; exact rt_inv_mul d hd
  · simp [h]

/-- [P1] (Ept): the pink state into a spider effect of coefficient `1` at `0` is `√d`. -/
theorem zcEpt (E : Env) (a dz w : ℕ) (m : Node) (iface : List ℕ) (hm : isZc m = true)
    (hdm : m.dim = dz) (hl : m.outs ++ m.ins = [w]) (hw : labIdx E.labs w = dz)
    (hd : 0 < E.dimv dz) (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1) (h0 : zc E m 0 = 1) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, m] [w] [] [] iface 1 (rt (E.dimv dz)) := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  have hD : labDim E w = E.dimv dz := by
    simp only [labDim]; simp only [labIdx] at hw; rw [hw]
  rw [hD]
  have step : ∀ j ∈ range (E.dimv dz), val E (Function.update τ w j) ⟨.X, [w], [], dz, a⟩ *
      val E (Function.update τ w j) m = if j = 0 then rt (E.dimv dz) else 0 := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    rw [val_zc _ _ m hm, hl, hdm]
    simp only [val, List.map_cons, List.map_nil, Function.update_self]
    rw [tX_zero_state _ j hd hj' _ ha, tZw_at _ _ _ j (by simp)]
    by_cases h : j = 0
    · subst h; simp [hj', h0]
    · simp [h]
  rw [Finset.sum_congr rfl step, Finset.sum_ite_eq']
  simp [hd]

/-- [P1] (Zer): the spider state of coefficients `[k = 0]` is the pink state. -/
theorem zcZer (E : Env) (a dz o : ℕ) (m : Node) (iface : List ℕ) (hm : isZc m = true)
    (hdm : m.dim = dz) (hl : m.outs ++ m.ins = [o]) (ho : o ∈ iface)
    (hdo : labIdx E.labs o = dz) (hd : 0 < E.dimv dz)
    (ha : ∀ k < E.dimv dz, ex (E.phase a k) = 1)
    (hc : ∀ k < E.dimv dz, zc E m k = if k = 0 then 1 else 0) :
    LocalEq E [m] [] [⟨.X, [o], [], dz, a⟩] [] iface 1 (rt (E.dimv dz))⁻¹ := by
  intro τ hτ
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  have hlt : τ o < E.dimv dz := by
    have := hτ o ho; simp only [labDim] at this; simp only [labIdx] at hdo; rwa [hdo] at this
  rw [val_zc _ _ m hm, hl, hdm]
  simp only [val, List.map_cons, List.map_nil]
  rw [tX_zero_state _ _ hd hlt _ ha, tZw_at _ _ _ (τ o) (by simp), hc _ hlt]
  by_cases h : τ o = 0 <;> simp [h, hlt, hd.ne', inv_mul_cancel₀ (rt_ne _ hd)]

/-- A two-output W node at `I < d` as its two nonzero entries. -/
theorem tW_pair (d j₁ j₂ I : ℕ) (hI : I < d) :
    tW d [j₁, j₂] I = (if j₁ = I ∧ j₂ = 0 then 1 else 0) +
      (if j₁ = 0 ∧ j₂ = I ∧ I ≠ 0 then 1 else 0) := by
  unfold tW
  by_cases h0 : I = 0
  · subst h0; by_cases h1 : j₁ = 0 <;> by_cases h2 : j₂ = 0 <;> simp [h1, h2]
  · by_cases h1 : j₁ = 0 <;> by_cases h2 : j₂ = 0 <;>
      simp [h0, Ne.symm h0, h1, h2, hI]

theorem dsum_pick (d a b : ℕ) (ha : a < d) (hb : b < d) (f : ℕ → ℕ → ℂ) :
    ∑ j₁ ∈ range d, ∑ j₂ ∈ range d, (if j₁ = a ∧ j₂ = b then f j₁ j₂ else 0) = f a b := by
  have : ∀ j₁, (∑ j₂ ∈ range d, if j₁ = a ∧ j₂ = b then f j₁ j₂ else 0) =
      if j₁ = a then f j₁ b else 0 := by
    intro j₁
    by_cases h : j₁ = a <;> simp [h, hb, Finset.sum_ite_eq']
  rw [Finset.sum_congr rfl fun j _ => this j, Finset.sum_ite_eq']
  simp [ha]

/-- [P1] (AD): spider effects `a` and `b` on the outputs of a W node are the spider effect
of coefficients `a_k b_0 + a_0 b_k` (`k ≠ 0`) and `a_0 b_0`. -/
theorem zcAd (E : Env) (dz x y i : ℕ) (na nb m : Node) (iface : List ℕ) (ha : isZc na = true)
    (hb : isZc nb = true) (hm : isZc m = true) (hda : na.dim = dz) (hdb : nb.dim = dz)
    (hdm : m.dim = dz) (hla : na.outs ++ na.ins = [x]) (hlb : nb.outs ++ nb.ins = [y])
    (hlm : m.outs ++ m.ins = [i]) (hxy : x ≠ y) (hxi : x ≠ i) (hyi : y ≠ i)
    (hx : labIdx E.labs x = dz) (hy : labIdx E.labs y = dz)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (h0 : zc E m 0 = zc E na 0 * zc E nb 0)
    (hc : ∀ k < E.dimv dz, k ≠ 0 → zc E m k = zc E na k * zc E nb 0 + zc E na 0 * zc E nb k) :
    LocalEq E [⟨.W, [x, y], [i], dz, 0⟩, na, nb] [x, y] [m] [] iface 1 1 := by
  intro τ hτ
  have ri := hτ i hi.1
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E x dz hx, labDim_of E y dz hy]
  rw [val_zc _ _ m hm, hlm, hdm]
  simp only [List.map_cons, List.map_nil]
  rw [tZw_at _ _ _ (τ i) (by simp)]
  simp only [ri, List.mem_singleton, forall_eq, true_and, ite_true]
  have hv : ∀ j₁ ∈ range (E.dimv dz), ∀ j₂ ∈ range (E.dimv dz),
      val E (Function.update (Function.update τ x j₁) y j₂) ⟨.W, [x, y], [i], dz, 0⟩ *
        (val E (Function.update (Function.update τ x j₁) y j₂) na *
          val E (Function.update (Function.update τ x j₁) y j₂) nb) =
      (if j₁ = τ i ∧ j₂ = 0 then zc E na j₁ * zc E nb j₂ else 0) +
        (if τ i ≠ 0 then (if j₁ = 0 ∧ j₂ = τ i then zc E na j₁ * zc E nb j₂ else 0) else 0) := by
    intro j₁ hj₁ j₂ hj₂
    have h1 := Finset.mem_range.1 hj₁
    have h2 := Finset.mem_range.1 hj₂
    rw [val_zc _ _ na ha, val_zc _ _ nb hb, hla, hlb, hda, hdb]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne hxy, Function.update_of_ne hxi.symm, Function.update_of_ne hyi.symm]
    rw [tW_pair _ _ _ _ ri, tZw_at _ _ _ j₁ (by simp), tZw_at _ _ _ j₂ (by simp)]
    simp only [h1, h2, List.mem_singleton, forall_eq, true_and, ite_true]
    by_cases e1 : j₁ = τ i <;> by_cases e2 : j₂ = 0 <;> by_cases e3 : j₁ = 0 <;>
      by_cases e4 : j₂ = τ i <;> by_cases e5 : τ i = 0 <;> simp_all
  rw [Finset.sum_congr rfl fun j₁ hj₁ => Finset.sum_congr rfl fun j₂ hj₂ => hv j₁ hj₁ j₂ hj₂]
  have hd : 0 < E.dimv dz := lt_of_le_of_lt (Nat.zero_le _) ri
  simp only [Finset.sum_add_distrib]
  rw [dsum_pick _ _ _ ri hd (fun a b => zc E na a * zc E nb b)]
  by_cases hi0 : τ i = 0
  · simp only [hi0, ne_eq, not_true_eq_false, ite_false, Finset.sum_const_zero, add_zero]
    rw [h0]
  · simp only [ne_eq, hi0, not_false_eq_true, ite_true]
    rw [dsum_pick _ _ _ hd ri (fun a b => zc E na a * zc E nb b), hc _ ri hi0]

/-- The X spider of phases `K_j` with one output `a` and one input `b < d`: `[a + j ≡ b]`. -/
theorem tX_shift (d j a b : ℕ) (hd : 0 < d) (hb : b < d) (φ : ℕ → ℝ)
    (hφ : ∀ k < d, ex (φ k) = om d ^ (k * j)) :
    tX d φ [a] [b] = if (a + j) % d = b then 1 else 0 := by
  rw [tX_kj d j hd φ hφ [a] [b]]
  simp only [List.length_singleton, List.sum_singleton, Nat.mod_eq_of_lt hb]
  split_ifs
  · rw [← rt_sq' d]
    norm_num only
    rw [← mul_pow, inv_mul_cancel₀ (rt_ne d hd), one_pow]
  · simp

/-- [P1] (K2): the pink `K_j` into a spider `a` is the spider `b` into the pink `K_j`, times
`s`, when `a_k = s b_{k + j}`. -/
theorem zcK2 (E : Env) (dz p j w w' o i : ℕ) (na nb : Node) (s : ℂ) (iface : List ℕ)
    (ha : isZc na = true) (hb : isZc nb = true) (hda : na.dim = dz) (hdb : nb.dim = dz)
    (hla : na.outs ++ na.ins = [o, w]) (hlb : nb.outs ++ nb.ins = [w', i])
    (hw : labIdx E.labs w = dz) (hw' : labIdx E.labs w' = dz)
    (ho : o ∈ iface ∧ labIdx E.labs o = dz) (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (hwo : w ≠ o) (hwi : w ≠ i) (hw'o : w' ≠ o) (hw'i : w' ≠ i) (hd : 0 < E.dimv dz)
    (hp : ∀ q < E.dimv dz, ex (E.phase p q) = om (E.dimv dz) ^ (q * j))
    (hc : ∀ k < E.dimv dz, zc E na k = s * zc E nb ((k + j) % E.dimv dz)) :
    LocalEq E [⟨.X, [w], [i], dz, p⟩, na] [w] [nb, ⟨.X, [o], [w'], dz, p⟩] [w'] iface 1 s := by
  intro τ hτ
  have ro := hτ o ho.1
  have ri := hτ i hi.1
  rw [labDim_of E o dz ho.2] at ro
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E w dz hw, labDim_of E w' dz hw']
  have L : ∀ v ∈ range (E.dimv dz), val E (Function.update τ w v) ⟨.X, [w], [i], dz, p⟩ *
      val E (Function.update τ w v) na =
      if v = τ o then (if (τ o + j) % E.dimv dz = τ i then zc E na (τ o) else 0) else 0 := by
    intro v hv
    have hv' := Finset.mem_range.1 hv
    rw [val_zc _ _ na ha, hla, hda]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne hwo.symm, Function.update_of_ne hwi.symm]
    rw [tX_shift _ _ _ _ hd ri _ hp, tZw_at _ _ _ v (by simp)]
    by_cases h : v = τ o
    · subst h; simp [ro]
    · have : ¬ (v < E.dimv dz ∧ ∀ x ∈ [τ o, v], x = v) := fun h' =>
        h (h'.2 (τ o) (by simp)).symm
      simp only [this, h, ite_false, mul_zero]
  have R : ∀ v ∈ range (E.dimv dz),
      val E (Function.update τ w' v) nb *
        val E (Function.update τ w' v) ⟨.X, [o], [w'], dz, p⟩ =
      if v = τ i then (if (τ o + j) % E.dimv dz = τ i then zc E nb (τ i) else 0) else 0 := by
    intro v hv
    have hv' := Finset.mem_range.1 hv
    rw [val_zc _ _ nb hb, hlb, hdb]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne hw'o.symm, Function.update_of_ne hw'i.symm]
    rw [tX_shift _ _ _ _ hd hv' _ hp, tZw_at _ _ _ v (by simp)]
    by_cases h : v = τ i
    · subst h; simp [ri]
    · have : ¬ (v < E.dimv dz ∧ ∀ x ∈ [v, τ i], x = v) := fun h' =>
        h (h'.2 (τ i) (by simp)).symm
      simp only [this, h, ite_false, zero_mul]
  rw [Finset.sum_congr rfl L, Finset.sum_congr rfl R, Finset.sum_ite_eq', Finset.sum_ite_eq']
  simp only [Finset.mem_range, ro, ri, ite_true]
  split_ifs with h
  · rw [hc _ ro, h]
  · simp

/-- [P1] (YT): a W node with a spider effect of coefficients `1, -1, …, -1` on its first
output is the inverse triangle. -/
theorem zcTi (E : Env) (dz e o i : ℕ) (m : Node) (iface : List ℕ) (hm : isZc m = true)
    (hdm : m.dim = dz) (hl : m.outs ++ m.ins = [e]) (he : labIdx E.labs e = dz)
    (ho : o ∈ iface ∧ labIdx E.labs o = dz) (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (heo : e ≠ o) (hei : e ≠ i) (h0 : zc E m 0 = 1)
    (hc : ∀ k < E.dimv dz, k ≠ 0 → zc E m k = -1) :
    LocalEq E [⟨.W, [e, o], [i], dz, 0⟩, m] [e] [⟨.Ti, [o], [i], dz, 0⟩] [] iface 1 1 := by
  intro τ hτ
  have ro := hτ o ho.1
  have ri := hτ i hi.1
  rw [labDim_of E o dz ho.2] at ro
  rw [labDim_of E i dz hi.2] at ri
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E e dz he]
  have L : ∀ v ∈ range (E.dimv dz), val E (Function.update τ e v) ⟨.W, [e, o], [i], dz, 0⟩ *
      val E (Function.update τ e v) m =
      (if v = τ i then (if τ o = 0 then zc E m v else 0) else 0) +
        (if v = 0 then (if τ o = τ i ∧ τ i ≠ 0 then zc E m v else 0) else 0) := by
    intro v hv
    have hv' := Finset.mem_range.1 hv
    rw [val_zc _ _ m hm, hl, hdm]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne heo.symm, Function.update_of_ne hei.symm]
    rw [tW_pair _ _ _ _ ri, tZw_at _ _ _ v (by simp)]
    simp only [hv', List.mem_singleton, forall_eq, true_and, ite_true]
    by_cases e1 : v = τ i <;> by_cases e2 : τ o = 0 <;> by_cases e3 : v = 0 <;>
      by_cases e4 : τ o = τ i <;> by_cases e5 : τ i = 0 <;> simp_all
  rw [Finset.sum_congr rfl L, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.sum_ite_eq']
  have hd : 0 < E.dimv dz := lt_of_le_of_lt (Nat.zero_le _) ri
  simp only [Finset.mem_range, ri, hd, ite_true, val, tTi]
  by_cases e1 : τ o = τ i <;> by_cases e2 : τ o = 0 <;> by_cases e5 : τ i = 0
  all_goals simp_all
  all_goals first | (rw [hc _ ri (by omega)]) | omega

end ArchytasZX
