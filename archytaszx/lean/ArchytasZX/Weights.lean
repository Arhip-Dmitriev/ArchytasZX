/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net
import ArchytasZX.Rules
import ArchytasZX.Closed
import ArchytasZX.ZXRules
import ArchytasZX.Completeness
import ArchytasZX.WZ
import ArchytasZX.Hopf
import ArchytasZX.Connective

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

/-- [P1] (YT): a W node with a spider effect of coefficients `1, …, 1` on its first output is
the triangle. -/
theorem zcT (E : Env) (dz e o i : ℕ) (m : Node) (iface : List ℕ) (hm : isZc m = true)
    (hdm : m.dim = dz) (hl : m.outs ++ m.ins = [e]) (he : labIdx E.labs e = dz)
    (ho : o ∈ iface ∧ labIdx E.labs o = dz) (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (heo : e ≠ o) (hei : e ≠ i) (h0 : zc E m 0 = 1)
    (hc : ∀ k < E.dimv dz, k ≠ 0 → zc E m k = 1) :
    LocalEq E [⟨.W, [e, o], [i], dz, 0⟩, m] [e] [⟨.T, [o], [i], dz, 0⟩] [] iface 1 1 := by
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
  simp only [Finset.mem_range, ri, hd, ite_true, val, tT]
  by_cases e1 : τ o = τ i <;> by_cases e2 : τ o = 0 <;> by_cases e5 : τ i = 0
  all_goals simp_all
  all_goals first | (rw [hc _ ri (by omega)]) | omega

/-- [P1] (Pcy): a spider `n` on the input of a W node is spiders of its coefficients on both
outputs, when its coefficient at `0` is `1`. -/
theorem zcPcy (E : Env) (dz u x y i p q : ℕ) (n m₁ m₂ : Node) (iface : List ℕ)
    (hn : isZc n = true) (h₁ : isZc m₁ = true) (h₂ : isZc m₂ = true)
    (hdn : n.dim = dz) (hd₁ : m₁.dim = dz) (hd₂ : m₂.dim = dz)
    (hln : n.outs ++ n.ins = [u, i]) (hl₁ : m₁.outs ++ m₁.ins = [x, p])
    (hl₂ : m₂.outs ++ m₂.ins = [y, q])
    (hu : labIdx E.labs u = dz) (hp : labIdx E.labs p = dz) (hq : labIdx E.labs q = dz)
    (hx : x ∈ iface ∧ labIdx E.labs x = dz) (hy : y ∈ iface ∧ labIdx E.labs y = dz)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (hne : [u, p, q, x, y, i].Nodup) (h0 : zc E n 0 = 1)
    (hc₁ : ∀ k < E.dimv dz, zc E m₁ k = zc E n k)
    (hc₂ : ∀ k < E.dimv dz, zc E m₂ k = zc E n k) :
    LocalEq E [n, ⟨.W, [x, y], [u], dz, 0⟩] [u]
      [⟨.W, [p, q], [i], dz, 0⟩, m₁, m₂] [p, q] iface 1 1 := by
  intro τ hτ
  have rx := hτ x hx.1
  have ry := hτ y hy.1
  have ri := hτ i hi.1
  rw [labDim_of E x dz hx.2] at rx
  rw [labDim_of E y dz hy.2] at ry
  rw [labDim_of E i dz hi.2] at ri
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false,
    not_or, List.nodup_nil, and_true] at hne
  obtain ⟨⟨hup, huq, hux, huy, hui⟩, ⟨hpq, hpx, hpy, hpi⟩, ⟨hqx, hqy, hqi⟩, ⟨hxy, hxi⟩, hyi⟩ :=
    hne
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E u dz hu, labDim_of E p dz hp, labDim_of E q dz hq]
  have L : ∀ v ∈ range (E.dimv dz), val E (Function.update τ u v) n *
      val E (Function.update τ u v) ⟨.W, [x, y], [u], dz, 0⟩ =
      if v = τ i then zc E n v * tW (E.dimv dz) [τ x, τ y] v else 0 := by
    intro v hv
    rw [val_zc _ _ n hn, hln, hdn]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne (Ne.symm hui), Function.update_of_ne (Ne.symm hux),
      Function.update_of_ne (Ne.symm huy)]
    rw [tZw_at _ _ _ v (by simp)]
    have hv' := Finset.mem_range.1 hv
    by_cases e : v = τ i
    · subst e; simp [hv']
    · have : ¬ (v < E.dimv dz ∧ ∀ w ∈ [v, τ i], w = v) := fun h => e (h.2 (τ i) (by simp)).symm
      simp only [this, e, ite_false, zero_mul]
  have R : ∀ j₁ ∈ range (E.dimv dz), ∀ j₂ ∈ range (E.dimv dz),
      val E (Function.update (Function.update τ p j₁) q j₂) ⟨.W, [p, q], [i], dz, 0⟩ *
        (val E (Function.update (Function.update τ p j₁) q j₂) m₁ *
          val E (Function.update (Function.update τ p j₁) q j₂) m₂) =
      if j₁ = τ x ∧ j₂ = τ y then tW (E.dimv dz) [j₁, j₂] (τ i) * (zc E n j₁ * zc E n j₂)
        else 0 := by
    intro j₁ hj₁ j₂ hj₂
    have h1 := Finset.mem_range.1 hj₁
    have h2 := Finset.mem_range.1 hj₂
    rw [val_zc _ _ m₁ h₁, val_zc _ _ m₂ h₂, hl₁, hl₂, hd₁, hd₂]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne hpq, Function.update_of_ne (Ne.symm hqi),
      Function.update_of_ne (Ne.symm hpi),
      Function.update_of_ne (Ne.symm hqx), Function.update_of_ne (Ne.symm hpx),
      Function.update_of_ne (Ne.symm hqy), Function.update_of_ne (Ne.symm hpy)]
    rw [tZw_at _ _ _ j₁ (by simp), tZw_at _ _ _ j₂ (by simp)]
    by_cases e1 : j₁ = τ x <;> by_cases e2 : j₂ = τ y
    · subst e1; subst e2; simp [h1, h2, hc₁ _ h1, hc₂ _ h2]
    · have e2' : ¬ τ y = j₂ := fun h => e2 h.symm
      simp [e2, e2']
    all_goals
      have e1' : ¬ τ x = j₁ := fun h => e1 h.symm
      simp [e1, e1']
  rw [Finset.sum_congr rfl L, Finset.sum_ite_eq']
  rw [Finset.sum_congr rfl fun j₁ hj₁ => Finset.sum_congr rfl fun j₂ hj₂ => R j₁ hj₁ j₂ hj₂]
  rw [dsum_pick _ _ _ rx ry (fun a b => tW (E.dimv dz) [a, b] (τ i) * (zc E n a * zc E n b))]
  simp only [Finset.mem_range, ri, ite_true]
  rw [tW_pair _ _ _ _ ri]
  by_cases e1 : τ x = τ i <;> by_cases e2 : τ y = 0 <;> by_cases e3 : τ x = 0 <;>
    by_cases e4 : τ y = τ i <;> by_cases e5 : τ i = 0 <;> simp_all

/-- [P1] (P1): two Fourier boxes in series are the phaseless X state with two outputs, the
first bent onto the input by a phaseless Z spider. -/
theorem fourierSq (E : Env) (px pz dz a b w v : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hv : labIdx E.labs v = dz)
    (hwa : w ≠ a) (hwb : w ≠ b) (hva : v ≠ a) (hvb : v ≠ b)
    (ha : a ∈ iface ∧ labIdx E.labs a = dz) (hb : b ∈ iface ∧ labIdx E.labs b = dz)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) :
    LocalEq E [⟨.F, [w], [a], dz, 0⟩, ⟨.F, [b], [w], dz, 0⟩] [w]
      [⟨.X, [v, b], [], dz, px⟩, ⟨.Z, [v], [a], dz, pz⟩] [v] iface 1 1 := by
  intro τ hτ
  have ra := hτ a ha.1
  have rb := hτ b hb.1
  rw [labDim_of E a dz ha.2] at ra
  rw [labDim_of E b dz hb.2] at rb
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E _ dz hw, labDim_of E _ dz hv,
    Function.update_apply, hwa, hwb, hva, hvb, Ne.symm hwa, Ne.symm hwb, Ne.symm hva,
    Ne.symm hvb, if_true, if_false,
    List.cons_append, List.nil_append]
  have R : ∀ j ∈ range (E.dimv dz), tX (E.dimv dz) (E.phase px) [j, τ b] [] *
      tZ (E.dimv dz) (E.phase pz) [j, τ a] =
      if j = τ a then tX (E.dimv dz) (E.phase px) [τ a, τ b] [] else 0 := by
    intro j hj
    rw [tZ_at _ _ _ j (by simp)]
    have hj' := Finset.mem_range.1 hj
    by_cases e : j = τ a
    · subst e; simp [hj', hz _ hj']
    · have : ¬ (j < E.dimv dz ∧ ∀ x ∈ [j, τ a], x = j) := fun h => e (h.2 (τ a) (by simp)).symm
      have e' : ¬ τ a = j := fun h => e h.symm
      simp [this, e, e']
  rw [Finset.sum_congr rfl R, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ra)]
  simp only [tX, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
  have hx' : ∀ k ∈ range (E.dimv dz), ex (E.phase px k) * (fou (E.dimv dz) (τ a) k *
      fou (E.dimv dz) (τ b) k) = fou (E.dimv dz) (τ a) k * fou (E.dimv dz) k (τ b) := by
    intro k hk
    rw [hx _ (Finset.mem_range.1 hk), one_mul, fou_symm _ (τ b) k]
  rw [Finset.sum_congr rfl hx', fou_sq _ hd]
  have hl : ∀ j ∈ range (E.dimv dz), fou (E.dimv dz) j (τ a) * fou (E.dimv dz) (τ b) j =
      fou (E.dimv dz) (τ b) j * fou (E.dimv dz) j (τ a) := fun _ _ => mul_comm _ _
  rw [Finset.sum_congr rfl hl, fou_sq _ hd, add_comm]

/-- [P1] (P1): two Fourier boxes in series are the phaseless X state on their two outer legs. -/
theorem fourierSq' (E : Env) (px dz a b w : ℕ) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwa : w ≠ a) (hwb : w ≠ b)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    LocalEq E [⟨.F, [w], [a], dz, 0⟩, ⟨.F, [b], [w], dz, 0⟩] [w]
      [⟨.X, [a, b], [], dz, px⟩] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E _ dz hw, Function.update_apply, Ne.symm hwa,
    Ne.symm hwb, if_true, if_false, List.cons_append, List.nil_append, tX]
  have hx' : ∀ k ∈ range (E.dimv dz), ex (E.phase px k) * (fou (E.dimv dz) (τ a) k *
      fou (E.dimv dz) (τ b) k) = fou (E.dimv dz) (τ a) k * fou (E.dimv dz) k (τ b) := by
    intro k hk
    rw [hx _ (Finset.mem_range.1 hk), one_mul, fou_symm _ (τ b) k]
  rw [Finset.sum_congr rfl hx', fou_sq _ hd]
  have hl : ∀ j ∈ range (E.dimv dz), fou (E.dimv dz) j (τ a) * fou (E.dimv dz) (τ b) j =
      fou (E.dimv dz) (τ b) j * fou (E.dimv dz) j (τ a) := fun _ _ => mul_comm _ _
  rw [Finset.sum_congr rfl hl, fou_sq _ hd, add_comm]

/-- [P1] (S2): the phaseless X spider with one input and one output is the phaseless Z spider
with the same legs. -/
theorem identityXZ (E : Env) (a b dz p q w x : ℕ) (iface : List ℕ)
    (hd : 0 < E.dimv dz) (hpq : (p = x ∧ q = w) ∨ (p = w ∧ q = x))
    (hface : p ∈ iface ∧ q ∈ iface) (hp : labIdx E.labs p = dz) (hq : labIdx E.labs q = dz)
    (hph : ∀ k < E.dimv dz, ex (E.phase a k) = 1)
    (hb : ∀ k < E.dimv dz, ex (E.phase b k) = 1) :
    LocalEq E [⟨.X, [p], [q], dz, a⟩] [] [⟨.Z, [x], [w], dz, b⟩] [] iface 1 1 := by
  intro τ hτ
  have rp := hτ p hface.1
  have rq := hτ q hface.2
  rw [labDim_of E p dz hp] at rp
  rw [labDim_of E q dz hq] at rq
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, tX, List.cons_append, List.nil_append]
  have : ∑ k ∈ range (E.dimv dz), ex (E.phase a k) * fou (E.dimv dz) (τ p) k *
      (starRingEnd ℂ) (fou (E.dimv dz) (τ q) k) =
      ∑ k ∈ range (E.dimv dz), fou (E.dimv dz) k (τ p) *
      (starRingEnd ℂ) (fou (E.dimv dz) k (τ q)) := by
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [hph k (Finset.mem_range.1 hk), fou_symm _ (τ p), fou_symm _ (τ q), one_mul]
  rw [this, fou_unitary _ hd _ _ rp rq]
  rcases hpq with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · rw [tZ_at _ _ _ (τ p) (by simp)]
    by_cases h : τ p = τ q
    · simp [h, rq, hb _ rq]
    · simp [h, Ne.symm h]
  · rw [tZ_at _ _ _ (τ q) (by simp)]
    by_cases h : τ p = τ q
    · simp [h, rq, hb _ rq]
    · simp [h, Ne.symm h]

theorem sumOver_fsum (D : ℕ → ℕ) (ls : List ℕ) (s : Finset ℕ) (g : ℕ → (ℕ → ℕ) → ℂ)
    (σ : ℕ → ℕ) : sumOver D ls (fun τ => ∑ k ∈ s, g k τ) σ = ∑ k ∈ s, sumOver D ls (g k) σ := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih => simp only [sumOver, ih]; exact Finset.sum_comm

/-- Summing an indicator that every label of `L` is `k` over labels `ws ⊆ L`. -/
theorem sumOver_allEq (D : ℕ → ℕ) (ws L : List ℕ) (hnd : ws.Nodup) (hsub : ∀ w ∈ ws, w ∈ L)
    (k : ℕ) (hk : ∀ w ∈ ws, k < D w) (c : ℂ) (σ : ℕ → ℕ) :
    sumOver D ws (fun τ => if ∀ v ∈ L, τ v = k then c else 0) σ =
      if ∀ v ∈ L, v ∉ ws → σ v = k then c else 0 := by
  induction ws generalizing σ with
  | nil => simp [sumOver]
  | cons w ws ih =>
    simp only [sumOver]
    obtain ⟨hw, hnd'⟩ := List.nodup_cons.1 hnd
    rw [Finset.sum_congr rfl fun j _ => ih hnd' (fun x hx => hsub x (by simp [hx]))
      (fun x hx => hk x (by simp [hx])) (Function.update σ w j)]
    have hwL := hsub w (by simp)
    have : ∀ j ∈ range (D w), (if ∀ v ∈ L, v ∉ ws → Function.update σ w j v = k then c else 0) =
        if j = k then (if ∀ v ∈ L, v ∉ w :: ws → σ v = k then c else 0) else 0 := by
      intro j _
      by_cases hj : j = k
      · subst hj
        have iff : (∀ v ∈ L, v ∉ ws → Function.update σ w j v = j) ↔
            (∀ v ∈ L, v ∉ w :: ws → σ v = j) := by
          constructor
          · intro h v hv hvn
            simp only [List.mem_cons, not_or] at hvn
            have := h v hv hvn.2
            rwa [Function.update_of_ne hvn.1] at this
          · intro h v hv hvn
            by_cases hvw : v = w
            · subst hvw; simp
            · rw [Function.update_of_ne hvw]; exact h v hv (by simp [hvw, hvn])
        simp only [iff, ite_true]
      · have : ¬ ∀ v ∈ L, v ∉ ws → Function.update σ w j v = k := fun h =>
          hj (by simpa using h w hwL hw)
        simp [this, hj]
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq']
    simp [hk w (by simp)]

/-- [P1] (S1): two spiders joined by the wires `ws` fuse, their coefficients multiplying. -/
theorem zcFusionN (E : Env) (na nb m : Node) (dz : ℕ) (ws iface : List ℕ)
    (ha : isZc na = true) (hb : isZc nb = true) (hm : isZc m = true) (hda : na.dim = dz)
    (hdb : nb.dim = dz) (hdm : m.dim = dz) (hnd : ws.Nodup) (w₀ : ℕ) (h₀ : w₀ ∈ ws)
    (h₁ : ∀ w ∈ ws, w ∈ na.outs ++ na.ins) (h₂ : ∀ w ∈ ws, w ∈ nb.outs ++ nb.ins)
    (hl : (m.outs ++ m.ins).Perm
      ((na.outs ++ na.ins ++ (nb.outs ++ nb.ins)).filter (fun x => !ws.contains x)))
    (hw : ∀ w ∈ ws, labIdx E.labs w = dz)
    (hc : ∀ k < E.dimv dz, zc E m k = zc E na k * zc E nb k) :
    LocalEq E [na, nb] ws [m] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, sumOver]
  set A := na.outs ++ na.ins
  set B := nb.outs ++ nb.ins
  have prod : ∀ ρ : ℕ → ℕ, val E ρ na * val E ρ nb =
      ∑ k ∈ range (E.dimv dz), if ∀ v ∈ A ++ B, ρ v = k then zc E na k * zc E nb k else 0 := by
    intro ρ
    rw [val_zc _ _ na ha, val_zc _ _ nb hb, hda, hdb,
      tZw_at _ _ _ (ρ w₀) (List.mem_map_of_mem (h₁ w₀ h₀)),
      tZw_at _ _ _ (ρ w₀) (List.mem_map_of_mem (h₂ w₀ h₀))]
    have hs : (∑ k ∈ range (E.dimv dz), if ∀ v ∈ A ++ B, ρ v = k then zc E na k * zc E nb k else 0) =
        ∑ k ∈ range (E.dimv dz), if k = ρ w₀ then
          (if ∀ v ∈ A ++ B, ρ v = ρ w₀ then zc E na k * zc E nb k else 0) else 0 := by
      refine Finset.sum_congr rfl fun k _ => ?_
      by_cases e : k = ρ w₀
      · subst e; simp
      · have : ¬ ∀ v ∈ A ++ B, ρ v = k := fun h =>
          e (h w₀ (List.mem_append_left _ (h₁ w₀ h₀))).symm
        rw [if_neg this, if_neg e]
    rw [hs, Finset.sum_ite_eq']
    have eA : (∀ v ∈ A.map ρ, v = ρ w₀) ↔ ∀ v ∈ A, ρ v = ρ w₀ := by simp
    have eB : (∀ v ∈ B.map ρ, v = ρ w₀) ↔ ∀ v ∈ B, ρ v = ρ w₀ := by simp
    have eAB : (∀ v ∈ A ++ B, ρ v = ρ w₀) ↔ (∀ v ∈ A, ρ v = ρ w₀) ∧ ∀ v ∈ B, ρ v = ρ w₀ := by
      simp only [List.mem_append, or_imp, forall_and]
    simp only [eA, eB, eAB, Finset.mem_range]
    by_cases hr : ρ w₀ < E.dimv dz <;> by_cases hA : ∀ v ∈ A, ρ v = ρ w₀ <;>
      by_cases hB : ∀ v ∈ B, ρ v = ρ w₀ <;>
      first | (simp [hr, hA, hB]; done) | (simp [hr]; rw [if_pos hB, if_pos hA, if_pos ⟨hA, hB⟩])
  rw [show prodVal E [na, nb] = fun τ =>
      ∑ k ∈ range (E.dimv dz), if ∀ v ∈ A ++ B, τ v = k then zc E na k * zc E nb k else 0 from
      funext fun ρ => by
        simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
        exact prod ρ, sumOver_fsum]
  have hD : ∀ w ∈ ws, labDim E w = E.dimv dz := fun w hw' => labDim_of E w dz (hw w hw')
  rw [Finset.sum_congr rfl fun k hk => sumOver_allEq _ ws (A ++ B) hnd
    (fun w hw' => List.mem_append_left _ (h₁ w hw')) k
    (fun w hw' => by rw [hD w hw']; exact Finset.mem_range.1 hk) _ τ]
  rw [val_zc E τ m hm, hdm, tZw_perm _ _ _ _ (hl.map τ)]
  unfold tZw
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [hc k (Finset.mem_range.1 hk)]
  have iff : (∀ v ∈ ((A ++ B).filter (fun x => !ws.contains x)).map τ, v = k) ↔
      ∀ v ∈ A ++ B, v ∉ ws → τ v = k := by
    constructor
    · intro h v hv hvn
      exact h (τ v) (List.mem_map_of_mem (List.mem_filter.2 ⟨hv, by simpa using hvn⟩))
    · intro h v hv
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv
      obtain ⟨hx1, hx2⟩ := List.mem_filter.1 hx
      exact h x hx1 (by simpa using hx2)
  simp only [iff]

/-- Summing four labels each read by one of two phaseless Z spiders `(c | x₁ y₁)`, `(e | x₂ y₂)`
sets them to `c` and `e`. -/
theorem collapse4 (E : Env) (dz c e x₁ x₂ y₁ y₂ p₁ p₂ : ℕ) (g : (ℕ → ℕ) → ℂ) (τ : ℕ → ℕ)
    (hnd : [x₁, x₂, y₁, y₂, c, e].Nodup) (hx₁ : labIdx E.labs x₁ = dz)
    (hx₂ : labIdx E.labs x₂ = dz) (hy₁ : labIdx E.labs y₁ = dz) (hy₂ : labIdx E.labs y₂ = dz)
    (rc : τ c < E.dimv dz) (re : τ e < E.dimv dz)
    (h₁ : ∀ k < E.dimv dz, ex (E.phase p₁ k) = 1) (h₂ : ∀ k < E.dimv dz, ex (E.phase p₂ k) = 1) :
    sumOver (labDim E) [x₁, x₂, y₁, y₂] (fun ρ => g ρ *
      (tZ (E.dimv dz) (E.phase p₁) [ρ c, ρ x₁, ρ y₁] *
        tZ (E.dimv dz) (E.phase p₂) [ρ e, ρ x₂, ρ y₂])) τ =
    g (fun x => if x ∈ [x₁, x₂, y₁, y₂] then
      τ (if x = x₁ ∨ x = y₁ then c else e) else τ x) := by
  simp only [List.nodup_cons, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false,
    not_or, List.nodup_nil, and_true] at hnd
  obtain ⟨⟨h12, h13, h14, h1c, h1e⟩, ⟨h23, h24, h2c, h2e⟩, ⟨h34, h3c, h3e⟩, ⟨h4c, h4e⟩, hce⟩ :=
    hnd
  set f : ℕ → ℕ := fun l => if l = x₁ ∨ l = y₁ then c else e
  have hf : ∀ l ∈ [x₁, x₂, y₁, y₂], f l ∉ [x₁, x₂, y₁, y₂] := by
    intro l _
    simp only [f]
    split_ifs <;> simp [Ne.symm h1c, Ne.symm h2c, Ne.symm h3c, Ne.symm h4c, Ne.symm h1e,
      Ne.symm h2e, Ne.symm h3e, Ne.symm h4e]
  have hD : ∀ l ∈ [x₁, x₂, y₁, y₂], labDim E l = E.dimv dz := by
    intro l hl
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hl
    rcases hl with rfl | rfl | rfl | rfl <;> exact labDim_of E _ dz (by assumption)
  rw [← sumOver_pin (labDim E) [x₁, x₂, y₁, y₂] (by simp [*]) f hf g τ (by
    intro l hl; rw [hD l hl]; simp only [f]; split_ifs <;> assumption)]
  refine sumOver_congr _ _ _ _ _ fun ρ hρ _ => ?_
  have ec : ρ c = τ c := hρ c (by simp [Ne.symm h1c, Ne.symm h2c, Ne.symm h3c, Ne.symm h4c])
  have ee : ρ e = τ e := hρ e (by simp [Ne.symm h1e, Ne.symm h2e, Ne.symm h3e, Ne.symm h4e])
  rw [tZ_at _ _ _ (ρ c) (by simp), tZ_at _ _ _ (ρ e) (by simp), ec, ee, h₁ _ rc, h₂ _ re]
  have fx₁ : f x₁ = c := by simp [f]
  have fy₁ : f y₁ = c := by simp [f]
  have fx₂ : f x₂ = e := by simp [f, Ne.symm h12, h23]
  have fy₂ : f y₂ = e := by simp [f, Ne.symm h14, Ne.symm h34]
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false, forall_eq_or_imp,
    forall_eq, fx₁, fy₁, fx₂, fy₂, ec, ee, rc, re, true_and]
  by_cases a1 : ρ x₁ = τ c <;> by_cases a2 : ρ x₂ = τ e <;> by_cases a3 : ρ y₁ = τ c <;>
    by_cases a4 : ρ y₂ = τ e <;> simp [a1, a2, a3, a4]

theorem ta_arith (d a b c e : ℕ) (hd : 0 < d) (ra : a < d) (rb : b < d) (rc : c < d)
    (re : e < d) :
    ((if c = a ∧ e = 0 then (1 : ℂ) else 0) + (if c = 0 ∧ e = a ∧ a ≠ 0 then 1 else 0)) *
      ((rt d)⁻¹ ^ (2 + 1) * if (c + e + 0) % d = b % d then (d : ℂ) else 0) =
    (rt d)⁻¹ * (((if c = a ∧ e = 0 then (1 : ℂ) else 0) +
      (if c = 0 ∧ e = a ∧ a ≠ 0 then 1 else 0)) *
      ((if c = b ∧ e = 0 then (1 : ℂ) else 0) + (if c = 0 ∧ e = b ∧ b ≠ 0 then 1 else 0))) := by
  have hr := rt_ne d hd
  have key : (rt d ^ 3)⁻¹ * (d : ℂ) = (rt d)⁻¹ := by
    rw [← rt_sq d]; field_simp
  rw [Nat.mod_eq_of_lt rb]
  by_cases h1 : c = a ∧ e = 0
  · obtain ⟨rfl, rfl⟩ := h1
    simp only [Nat.add_zero, Nat.mod_eq_of_lt rc, and_self, ite_true, ne_eq]
    by_cases hcb : c = b
    · subst hcb; by_cases c0 : c = 0 <;> simp [c0, key, inv_pow]
    · have : ¬ (c = 0 ∧ 0 = b ∧ ¬ b = 0) := fun h => h.2.2 h.2.1.symm
      simp [hcb, this]
  · by_cases h2 : c = 0 ∧ e = a ∧ a ≠ 0
    · obtain ⟨rfl, rfl, ha⟩ := h2
      simp only [h1, ite_false, zero_add, Nat.add_zero, Nat.mod_eq_of_lt re, ne_eq, ha,
        not_false_eq_true, and_self, ite_true, true_and, one_mul]
      by_cases heb : e = b
      · subst heb; simp [ha, key, inv_pow]
      · simp [heb, Ne.symm heb, ha]
    · simp [h1, h2]

/-- [P1] (TA): the phaseless X spider (1 -> 2) under a W node, crossed into two phaseless
Z spiders, is a W node in its place, times `d^(-1/2)`. -/
theorem zxwTA (E : Env) (dz a b c e x₁ x₂ y₁ y₂ u₁ u₂ v₁ v₂ px p₁ p₂ q₁ q₂ : ℕ)
    (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hnL : [x₁, x₂, y₁, y₂, c, e].Nodup) (hnR : [u₁, u₂, v₁, v₂, c, e].Nodup)
    (hxa : ∀ l ∈ [x₁, x₂, y₁, y₂, u₁, u₂, v₁, v₂], l ≠ a ∧ l ≠ b)
    (hx : ∀ l ∈ [x₁, x₂, y₁, y₂, u₁, u₂, v₁, v₂], labIdx E.labs l = dz)
    (ha : a ∈ iface ∧ labIdx E.labs a = dz) (hb : b ∈ iface ∧ labIdx E.labs b = dz)
    (hc : c ∈ iface ∧ labIdx E.labs c = dz) (he : e ∈ iface ∧ labIdx E.labs e = dz)
    (hpx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hp₁ : ∀ k < E.dimv dz, ex (E.phase p₁ k) = 1) (hp₂ : ∀ k < E.dimv dz, ex (E.phase p₂ k) = 1)
    (hq₁ : ∀ k < E.dimv dz, ex (E.phase q₁ k) = 1) (hq₂ : ∀ k < E.dimv dz, ex (E.phase q₂ k) = 1) :
    LocalEq E [⟨.W, [x₁, x₂], [a], dz, 0⟩, ⟨.X, [y₁, y₂], [b], dz, px⟩,
        ⟨.Z, [c], [x₁, y₁], dz, p₁⟩, ⟨.Z, [e], [x₂, y₂], dz, p₂⟩] [x₁, x₂, y₁, y₂]
      [⟨.W, [u₁, u₂], [a], dz, 0⟩, ⟨.W, [v₁, v₂], [b], dz, 0⟩,
        ⟨.Z, [c], [u₁, v₁], dz, q₁⟩, ⟨.Z, [e], [u₂, v₂], dz, q₂⟩] [u₁, u₂, v₁, v₂]
      iface 1 (rt (E.dimv dz))⁻¹ := by
  intro τ hτ
  have ra := hτ a ha.1
  have rb := hτ b hb.1
  have rc := hτ c hc.1
  have re := hτ e he.1
  rw [labDim_of E a dz ha.2] at ra
  rw [labDim_of E b dz hb.2] at rb
  rw [labDim_of E c dz hc.2] at rc
  rw [labDim_of E e dz he.2] at re
  have hL : prodVal E [⟨.W, [x₁, x₂], [a], dz, 0⟩, ⟨.X, [y₁, y₂], [b], dz, px⟩,
      ⟨.Z, [c], [x₁, y₁], dz, p₁⟩, ⟨.Z, [e], [x₂, y₂], dz, p₂⟩] = fun ρ =>
      (tW (E.dimv dz) [ρ x₁, ρ x₂] (ρ a) * tX (E.dimv dz) (E.phase px) [ρ y₁, ρ y₂] [ρ b]) *
      (tZ (E.dimv dz) (E.phase p₁) [ρ c, ρ x₁, ρ y₁] *
        tZ (E.dimv dz) (E.phase p₂) [ρ e, ρ x₂, ρ y₂]) := by
    funext ρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, val,
      List.cons_append, List.nil_append]
    ring
  have hR : prodVal E [⟨.W, [u₁, u₂], [a], dz, 0⟩, ⟨.W, [v₁, v₂], [b], dz, 0⟩,
      ⟨.Z, [c], [u₁, v₁], dz, q₁⟩, ⟨.Z, [e], [u₂, v₂], dz, q₂⟩] = fun ρ =>
      (tW (E.dimv dz) [ρ u₁, ρ u₂] (ρ a) * tW (E.dimv dz) [ρ v₁, ρ v₂] (ρ b)) *
      (tZ (E.dimv dz) (E.phase q₁) [ρ c, ρ u₁, ρ v₁] *
        tZ (E.dimv dz) (E.phase q₂) [ρ e, ρ u₂, ρ v₂]) := by
    funext ρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, val,
      List.cons_append, List.nil_append]
    ring
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false,
    forall_eq_or_imp, forall_eq] at hxa hx
  rw [hL, hR, collapse4 E dz c e x₁ x₂ y₁ y₂ p₁ p₂ _ τ hnL hx.1 hx.2.1 hx.2.2.1 hx.2.2.2.1 rc re
    hp₁ hp₂, collapse4 E dz c e u₁ u₂ v₁ v₂ q₁ q₂ _ τ hnR hx.2.2.2.2.1 hx.2.2.2.2.2.1
    hx.2.2.2.2.2.2.1 hx.2.2.2.2.2.2.2 rc re hq₁ hq₂]
  clear hL hR
  simp only [List.nodup_cons, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false,
    not_or, List.nodup_nil, and_true] at hnL hnR
  obtain ⟨⟨a1, a2, a3, a4, a5⟩, ⟨a6, a7, a8, a9⟩, ⟨a10, a11, a12⟩, ⟨a13, a14⟩, a15⟩ := hnL
  obtain ⟨⟨b1, b2, b3, b4, b5⟩, ⟨b6, b7, b8, b9⟩, ⟨b10, b11, b12⟩, ⟨b13, b14⟩, b15⟩ := hnR
  obtain ⟨⟨c1, c2⟩, ⟨c3, c4⟩, ⟨c5, c6⟩, ⟨c7, c8⟩, ⟨c9, c10⟩, ⟨c11, c12⟩, ⟨c13, c14⟩,
    ⟨c15, c16⟩⟩ := hxa
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false, true_or, or_true,
    ite_true, Ne.symm a1, Ne.symm a2, Ne.symm a3, Ne.symm a6, Ne.symm a7, Ne.symm a10,
    Ne.symm b1, Ne.symm b2, Ne.symm b3, Ne.symm b6, Ne.symm b7, Ne.symm b10, a2, a6, b2, b6,
    Ne.symm c1, Ne.symm c2, Ne.symm c3, Ne.symm c4, Ne.symm c5, Ne.symm c6, Ne.symm c7,
    Ne.symm c8, Ne.symm c9, Ne.symm c10, Ne.symm c11, Ne.symm c12, Ne.symm c13, Ne.symm c14,
    Ne.symm c15, Ne.symm c16, false_or, or_false, ite_false]
  rw [tX_kj _ 0 hd _ (fun k hk => by rw [hpx k hk]; simp), tW_pair _ _ _ _ ra,
    tW_pair _ _ _ _ rb]
  simp only [List.length_cons, List.length_nil, List.length_singleton, List.sum_cons,
    List.sum_nil, List.sum_singleton, Nat.add_zero, one_mul]
  exact ta_arith _ _ _ _ _ hd ra rb rc re

/-- [P1] (Bsj): the pink `K_j` state into a W node with its first output on a phaseless Z
effect is the spider state of coefficients `1` at `0` and at `-j`, and `0` elsewhere, times
`√d`. -/
theorem zxwBsj (E : Env) (a pz dz j w e o : ℕ) (m : Node) (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hj : j < E.dimv dz) (hm : isZc m = true) (hdm : m.dim = dz) (hl : m.outs ++ m.ins = [o])
    (hw : labIdx E.labs w = dz) (he : labIdx E.labs e = dz)
    (ho : o ∈ iface ∧ labIdx E.labs o = dz) (hwe : w ≠ e) (hwo : w ≠ o) (heo : e ≠ o)
    (hφ : ∀ k < E.dimv dz, ex (E.phase a k) = om (E.dimv dz) ^ (k * j))
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (hc : ∀ k < E.dimv dz,
      zc E m k = if k = 0 ∨ k = (E.dimv dz - j) % E.dimv dz then 1 else 0) :
    LocalEq E [⟨.X, [w], [], dz, a⟩, ⟨.W, [e, o], [w], dz, 0⟩, ⟨.Z, [], [e], dz, pz⟩] [w, e]
      [m] [] iface 1 (rt (E.dimv dz)) := by
  intro τ hτ
  have ro := hτ o ho.1
  rw [labDim_of E o dz ho.2] at ro
  have hvm : val E τ m = tZw (E.dimv dz) (zc E m) [τ o] := by
    rw [val_zc _ _ m hm, hl, hdm]; rfl
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one]
  rw [hvm]
  simp only [val, labDim_of E _ dz hw, labDim_of E _ dz he, Function.update_apply,
    hwe, hwo, heo, Ne.symm hwe, Ne.symm hwo, Ne.symm heo, if_true, if_false, List.nil_append,
    List.map_cons, List.map_nil, Function.update_self]
  set d := E.dimv dz
  set i := (d - j) % d
  have ri : i < d := Nat.mod_lt _ hd
  have hX : ∀ v ∈ range d, tX d (E.phase a) [v] [] = if v = i then rt d else 0 := by
    intro v hv
    rw [tX_kj d j hd _ hφ]
    simp only [List.length_singleton, List.length_nil, List.sum_singleton, List.sum_nil,
      Nat.zero_mod, pow_one, Nat.add_zero]
    simp only [antipode_iff d j v hj (Finset.mem_range.1 hv)]
    split_ifs
    · rw [← rt_sq d]; field_simp [rt_ne d hd]
    · simp
  have hZ : ∀ u ∈ range d, tZ d (E.phase pz) [u] = 1 := by
    intro u hu
    rw [tZ_at _ _ _ u (by simp)]
    simp [Finset.mem_range.1 hu, hz u (Finset.mem_range.1 hu)]
  have inner : ∀ v ∈ range d, ∑ u ∈ range d, tX d (E.phase a) [v] [] *
      (tW d [u, τ o] v * tZ d (E.phase pz) [u]) =
      if v = i then rt d * ∑ u ∈ range d, tW d [u, τ o] i else 0 := by
    intro v hv
    rw [hX v hv]
    split_ifs with h
    · subst h
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun u hu => by rw [hZ u hu, mul_one]
    · simp
  rw [Finset.sum_congr rfl inner, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ri)]
  rw [tZw_at _ _ _ (τ o) (by simp)]
  simp only [ro, List.mem_singleton, forall_eq, true_and, ite_true, hc _ ro]
  rw [Finset.sum_congr rfl fun u _ => tW_pair d u (τ o) i ri, Finset.sum_add_distrib]
  by_cases h0 : τ o = 0
  · by_cases hi0 : i = 0
    · simp [h0, hi0, Finset.sum_ite_eq', hd]
    · simp [h0, Ne.symm hi0, hi0, Finset.sum_ite_eq', ri]
  · by_cases hoi : τ o = i
    · have : i ≠ 0 := fun h => h0 (hoi.trans h)
      simp [h0, hoi, this, Finset.sum_ite_eq', hd]
    · simp [h0, hoi]

/-- The unique value below `d` that `j` shifts to `b`. -/
theorem shift_iff (d j w b : ℕ) (hd : 0 < d) (hw : w < d) (hb : b < d) :
    (w + j) % d = b ↔ w = (b + (d - j % d)) % d := by
  have hj : j % d < d := Nat.mod_lt _ hd
  have key : (w + j) % d = (w + j % d) % d := by rw [Nat.add_mod, Nat.mod_eq_of_lt hw]
  rw [key]
  generalize j % d = r at hj ⊢
  have e1 : (w + r) % d = if w + r < d then w + r else w + r - d := by
    split_ifs with h
    · exact Nat.mod_eq_of_lt h
    · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
  have e2 : (b + (d - r)) % d = if b + (d - r) < d then b + (d - r) else b + (d - r) - d := by
    split_ifs with h
    · exact Nat.mod_eq_of_lt h
    · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [e1, e2]
  split_ifs <;> omega

/-- The product over paired labels with every first label set to `v`. -/
theorem prod_zip_pin (us os : List ℕ) (hl : us.length = os.length) (ρ : ℕ → ℕ) (v : ℕ)
    (hρ : ∀ u ∈ us, ρ u = v) (f : ℕ → ℕ → ℂ) :
    ((us.zip os).map fun p => f (ρ p.2) (ρ p.1)).prod = (os.map fun o => f (ρ o) v).prod := by
  induction us generalizing os with
  | nil => cases os <;> simp_all
  | cons u us ih =>
    cases os with
    | nil => simp at hl
    | cons o os =>
      simp only [List.zip_cons_cons, List.map_cons, List.prod_cons]
      rw [hρ u (by simp), ih os (by simpa using hl) (fun x hx => hρ x (by simp [hx]))]

/-- [P1] (K1): the pink `K_j` spider (1 -> 1) before a plain green dot with outputs `os` is the
green dot followed by one pink `K_j` spider on each output. -/
theorem kOne (E : Env) (a b b' c dz w i j : ℕ) (os us iface : List ℕ) (hd : 0 < E.dimv dz)
    (hw : labIdx E.labs w = dz) (hwo : w ∉ os) (hwi : w ≠ i)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz)
    (hos : ∀ o ∈ os, o ∈ iface ∧ labIdx E.labs o = dz)
    (hus : ∀ u ∈ us, labIdx E.labs u = dz) (hnd : us.Nodup) (hui : i ∉ us)
    (huo : ∀ u ∈ us, u ∉ os) (hl : us.length = os.length)
    (ha : ∀ q < E.dimv dz, ex (E.phase a q) = om (E.dimv dz) ^ (q * j))
    (hc : ∀ q < E.dimv dz, ex (E.phase c q) = om (E.dimv dz) ^ (q * j))
    (hb : ∀ q < E.dimv dz, ex (E.phase b q) = 1) (hb' : ∀ q < E.dimv dz, ex (E.phase b' q) = 1) :
    LocalEq E [⟨.X, [w], [i], dz, a⟩, ⟨.Z, os, [w], dz, b⟩] [w]
      (⟨.Z, us, [i], dz, b'⟩ :: (us.zip os).map fun p => ⟨.X, [p.2], [p.1], dz, c⟩) us
      iface 1 1 := by
  intro τ hτ
  have ri := hτ i hi.1
  rw [labDim_of E i dz hi.2] at ri
  have hr : ∀ o ∈ os, τ o < E.dimv dz := fun o ho => by
    have := hτ o (hos o ho).1; rwa [labDim_of E o dz (hos o ho).2] at this
  set d := E.dimv dz
  set v := (τ i + (d - j % d)) % d
  have hv : v < d := Nat.mod_lt _ hd
  have hiff : ∀ x < d, (x + j) % d = τ i ↔ x = v := fun x hx => shift_iff d j x (τ i) hd hx ri
  -- the right side: pin every `u` to `i`
  have hR : sumOver (labDim E) us (prodVal E (⟨.Z, us, [i], dz, b'⟩ ::
      (us.zip os).map fun p => ⟨.X, [p.2], [p.1], dz, c⟩)) τ =
      (os.map fun o => tX d (E.phase c) [τ o] [τ i]).prod := by
    have hD : ∀ u ∈ us, labDim E u = d := fun u hu => labDim_of E u dz (hus u hu)
    set G : (ℕ → ℕ) → ℂ := fun ρ =>
      ((us.zip os).map fun p => tX d (E.phase c) [ρ p.2] [ρ p.1]).prod
    have h1 : sumOver (labDim E) us (prodVal E (⟨.Z, us, [i], dz, b'⟩ ::
        (us.zip os).map fun p => ⟨.X, [p.2], [p.1], dz, c⟩)) τ =
        sumOver (labDim E) us (fun ρ => if ∀ u ∈ us, ρ u = ρ ((fun _ => i) u) then G ρ else 0)
          τ := by
      refine sumOver_congr _ _ _ _ _ fun ρ hρ _ => ?_
      have eρ : ρ i = τ i := hρ i hui
      simp only [prodVal, List.map_cons, List.prod_cons, List.map_map, val, List.cons_append,
        List.nil_append]
      have rρ : ρ i < d := eρ ▸ ri
      rw [tZ_at _ _ _ (ρ i) (by simp), hb' _ rρ]
      have hz : (ρ i < E.dimv dz ∧ ∀ x ∈ us.map ρ ++ ρ i :: List.map ρ [], x = ρ i) ↔
          ∀ u ∈ us, ρ u = ρ i := by
        constructor
        · rintro ⟨-, h'⟩ u hu
          exact h' _ (List.mem_append_left _ (List.mem_map_of_mem hu))
        · intro h'
          refine ⟨rρ, fun y hy => ?_⟩
          rcases List.mem_append.1 hy with hy | hy
          · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hy
            exact h' u hu
          · simpa using hy
      by_cases hc' : ∀ u ∈ us, ρ u = ρ i
      · rw [if_pos (hz.2 hc'), if_pos hc', one_mul]
        simp only [G, Function.comp_def, val, List.map_cons, List.map_nil]; rfl
      · rw [if_neg (fun h' => hc' (hz.1 h')), if_neg hc', zero_mul]
    rw [h1, sumOver_pin (labDim E) us hnd (fun _ => i) (fun _ _ => hui) G τ
      (fun u hu => by rw [hD u hu]; exact ri)]
    have := prod_zip_pin us os hl (fun x => if x ∈ us then τ ((fun _ => i) x) else τ x) (τ i)
      (fun u hu => by simp [hu]) (fun x y => tX d (E.phase c) [x] [y])
    simp only at this
    simp only [G, this]
    refine congrArg _ (List.map_congr_left fun o ho => ?_)
    have : o ∉ us := fun h => huo o h ho
    simp [this]
  rw [hR]
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, Function.update_self, labDim_of E w dz hw,
    map_update_absent τ w _ os hwo, Function.update_of_ne (Ne.symm hwi), List.cons_append,
    List.nil_append]
  have step : ∀ x ∈ range d, tX d (E.phase a) [x] [τ i] * tZ d (E.phase b) (os.map τ ++ [x]) =
      if x = v then (if ∀ o ∈ os, τ o = v then 1 else 0) else 0 := by
    intro x hx'
    have hx := Finset.mem_range.1 hx'
    rw [tX_shift _ _ _ _ hd ri _ ha, tZ_at _ _ _ x (by simp)]
    by_cases h : x = v
    · rw [if_pos ((hiff _ hx).2 h), if_pos h, one_mul, hb _ hx]
      have hz : (x < d ∧ ∀ y ∈ os.map τ ++ [x], y = x) ↔ ∀ o ∈ os, τ o = v := by
        rw [← h]
        constructor
        · rintro ⟨-, h'⟩ o ho
          exact h' _ (List.mem_append_left _ (List.mem_map_of_mem ho))
        · intro h'
          refine ⟨hx, fun y hy => ?_⟩
          rcases List.mem_append.1 hy with hy | hy
          · obtain ⟨o, ho, rfl⟩ := List.mem_map.1 hy
            exact h' o ho
          · simpa using hy
      simp only [hz]
    · rw [if_neg (fun h' => h ((hiff _ hx).1 h')), if_neg h, zero_mul]
  rw [Finset.sum_congr rfl step, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hv)]
  have hx : ∀ o ∈ os, tX d (E.phase c) [τ o] [τ i] = if τ o = v then 1 else 0 := by
    intro o ho
    rw [tX_shift _ _ _ _ hd ri _ hc]
    by_cases h : τ o = v
    · rw [if_pos ((hiff _ (hr o ho)).2 h), if_pos h]
    · rw [if_neg (fun h' => h ((hiff _ (hr o ho)).1 h')), if_neg h]
  rw [List.map_congr_left hx, prod_ite_val]
  split_ifs <;> simp

/-- Summing labels all pinned to one value `k` sets each to `k`. -/
theorem sumOver_pinK {D : ℕ → ℕ} (ws : List ℕ) (hnd : ws.Nodup) (k : ℕ) (hk : ∀ w ∈ ws, k < D w)
    (g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D ws (fun τ => if ∀ v ∈ ws, τ v = k then g τ else 0) σ =
      g (fun x => if x ∈ ws then k else σ x) := by
  induction ws generalizing σ with
  | nil => simp [sumOver]
  | cons w ws ih =>
    obtain ⟨hw, hnd'⟩ := List.nodup_cons.1 hnd
    simp only [sumOver]
    have inner : ∀ j ∈ range (D w), sumOver D ws (fun τ => if ∀ v ∈ w :: ws, τ v = k then g τ
        else 0) (Function.update σ w j) =
        if j = k then sumOver D ws (fun τ => if ∀ v ∈ ws, τ v = k then g τ else 0)
          (Function.update σ w j) else 0 := by
      intro j _
      split_ifs with hj
      · refine sumOver_congr _ _ _ _ _ fun τ hτ _ => ?_
        have h1 : τ w = j := by rw [hτ w hw, Function.update_self]
        simp only [List.mem_cons, forall_eq_or_imp, h1, hj, true_and]
      · refine (sumOver_congr D ws _ (fun _ => 0) _ fun τ hτ _ => ?_).trans
          (sumOver_zero D ws _)
        have h1 : τ w = j := by rw [hτ w hw, Function.update_self]
        have : ¬ ∀ v ∈ w :: ws, τ v = k := fun h => hj (h1 ▸ h w List.mem_cons_self)
        rw [if_neg this]
    rw [Finset.sum_congr rfl inner, Finset.sum_ite_eq',
      if_pos (Finset.mem_range.2 (hk w List.mem_cons_self)),
      ih hnd' (fun x hx => hk x (List.mem_cons_of_mem _ hx))]
    congr 1
    funext x
    by_cases hx : x = w
    · subst hx; simp [hw]
    · by_cases hx' : x ∈ ws
      · simp [hx, hx']
      · simp [hx, hx', Function.update_of_ne hx]

/-- A sum over two label lists of a product whose factors each read labels outside the
other list. -/
theorem sumOver_mul_split {D : ℕ → ℕ} (a b Sf Sg : List ℕ) (f g : (ℕ → ℕ) → ℂ) (hf : Reads Sf f)
    (hg : Reads Sg g) (hfb : ∀ x ∈ Sf, x ∉ b) (hga : ∀ x ∈ Sg, x ∉ a) (σ : ℕ → ℕ) :
    sumOver D (a ++ b) (fun τ => f τ * g τ) σ = sumOver D a f σ * sumOver D b g σ := by
  rw [sumOver_append]
  have h1 : ∀ τ : ℕ → ℕ, sumOver D b (fun τ => f τ * g τ) τ = f τ * sumOver D b g τ :=
    fun τ => sumOver_const_mul D b Sf f g hf hfb τ
  rw [show sumOver D b (fun τ => f τ * g τ) = fun τ => f τ * sumOver D b g τ from funext h1]
  have h2 : ∀ τ : ℕ → ℕ, (∀ x ∉ a, τ x = σ x) → sumOver D b g τ = sumOver D b g σ := by
    intro τ hτ
    exact sumOver_ext D b Sg g hg τ σ fun x hx _ => hτ x (hga x hx)
  rw [show sumOver D a (fun τ => f τ * sumOver D b g τ) σ =
      sumOver D a (fun τ => f τ * sumOver D b g σ) σ from
    sumOver_congr _ _ _ _ _ fun τ hτ _ => by rw [h2 τ hτ]]
  rw [show (fun τ => f τ * sumOver D b g σ) = fun τ => sumOver D b g σ * f τ from
    funext fun τ => mul_comm _ _]
  rw [sumOver_const_mul D a [] (fun _ => sumOver D b g σ) f (fun _ _ _ => rfl) (by simp) σ,
    mul_comm]

/-- `F ; F ; F` is the conjugate of `F`. -/
theorem fou_cube (d : ℕ) (hd : 0 < d) (i k : ℕ) (hi : i < d) (hk : k < d) :
    ∑ a ∈ range d, ∑ b ∈ range d, fou d a i * (fou d b a * fou d k b) =
      (starRingEnd ℂ) (fou d i k) := by
  have : ∀ a ∈ range d, ∑ b ∈ range d, fou d a i * (fou d b a * fou d k b) =
      fou d a i * (if (k + a) % d = 0 then 1 else 0) := by
    intro a _
    rw [← Finset.mul_sum]
    congr 1
    rw [Finset.sum_congr rfl fun b _ => mul_comm (fou d b a) (fou d k b), fou_sq d hd]
  rw [Finset.sum_congr rfl this]
  have hsel : ∀ a ∈ range d, fou d a i * (if (k + a) % d = 0 then (1 : ℂ) else 0) =
      if a = (d - k) % d then fou d a i else 0 := by
    intro a ha
    have e : (k + a) % d = 0 ↔ a = (d - k) % d := by
      rw [add_comm]; exact antipode_iff d k a hk (Finset.mem_range.1 ha)
    simp only [e]
    split_ifs <;> simp
  rw [Finset.sum_congr rfl hsel, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd))]
  simp only [fou, map_div₀, map_pow, conj_om, conj_sqrt]
  congr 1
  have h1 : om d ^ ((d - k) % d * i) * om d ^ (i * k) = 1 := by
    rw [← pow_add, show (d - k) % d * i + i * k = ((d - k) % d + k) * i by ring]
    exact om_pow_zero_mod d hd _ _ (by rw [add_comm]; exact antipode_spec d k hd hk)
  rw [inv_pow]; exact eq_inv_of_mul_eq_one_left h1

/-- The labels of an input chain: its input, its two inner wires, its end on the Z spider. -/
def chainLabs (c : ℕ × (ℕ × ℕ) × ℕ) : List ℕ := [c.1, c.2.1.1, c.2.1.2, c.2.2]

/-- The entry of an input chain `F ; F ; F`. -/
noncomputable def chainVal (d : ℕ) (τ : ℕ → ℕ) (c : ℕ × (ℕ × ℕ) × ℕ) : ℂ :=
  fou d (τ c.2.1.1) (τ c.1) * (fou d (τ c.2.1.2) (τ c.2.1.1) * fou d (τ c.2.2) (τ c.2.1.2))

/-- Summing the inner wires of input chains ending on labels set to `k`. -/
theorem sumOver_chains {D : ℕ → ℕ} (d : ℕ) (hd : 0 < d) (k : ℕ) (hk : k < d) (cs : List (ℕ × (ℕ × ℕ) × ℕ))
    (hD : ∀ c ∈ cs, D c.2.1.1 = d ∧ D c.2.1.2 = d)
    (hnd : (cs.flatMap fun c => [c.2.1.1, c.2.1.2]).Nodup)
    (hio : ∀ c ∈ cs, ∀ c' ∈ cs, c.1 ∉ [c'.2.1.1, c'.2.1.2] ∧ c.2.2 ∉ [c'.2.1.1, c'.2.1.2])
    (ρ : ℕ → ℕ) (hz : ∀ c ∈ cs, ρ c.2.2 = k) (hi : ∀ c ∈ cs, ρ c.1 < d) :
    sumOver D (cs.flatMap fun c => [c.2.1.1, c.2.1.2])
        (fun τ => (cs.map (chainVal d τ)).prod) ρ =
      (cs.map fun c => (starRingEnd ℂ) (fou d (ρ c.1) k)).prod := by
  induction cs with
  | nil => simp [sumOver]
  | cons c rest ih =>
    simp only [List.flatMap_cons, List.map_cons, List.prod_cons]
    have hnd' := hnd
    simp only [List.flatMap_cons, List.cons_append, List.nil_append, List.nodup_cons,
      List.mem_cons, List.mem_flatMap, not_or, not_exists, not_and] at hnd'
    obtain ⟨⟨hab, ha⟩, ⟨hb, hrest⟩⟩ := hnd'
    have hc := hio c List.mem_cons_self c List.mem_cons_self
    rw [sumOver_mul_split (D := D) [c.2.1.1, c.2.1.2] _ (chainLabs c)
      (rest.flatMap chainLabs) (fun τ => chainVal d τ c) (fun τ => (rest.map (chainVal d τ)).prod)
      (fun τ τ' h => by
        simp only [chainVal]
        rw [h c.1 (by simp [chainLabs]), h c.2.1.1 (by simp [chainLabs]),
          h c.2.1.2 (by simp [chainLabs]), h c.2.2 (by simp [chainLabs])])
      (fun τ τ' h => by
        refine congrArg List.prod (List.map_congr_left fun c' hc' => ?_)
        have m : ∀ x ∈ chainLabs c', x ∈ rest.flatMap chainLabs := fun x hx =>
          List.mem_flatMap.2 ⟨c', hc', hx⟩
        simp only [chainVal]
        rw [h c'.1 (m _ (by simp [chainLabs])), h c'.2.1.1 (m _ (by simp [chainLabs])),
          h c'.2.1.2 (m _ (by simp [chainLabs])), h c'.2.2 (m _ (by simp [chainLabs]))])
      (fun x hx hxr => by
        obtain ⟨c', hc', hx'⟩ := List.mem_flatMap.1 hxr
        have h' := hio c List.mem_cons_self c' (List.mem_cons_of_mem _ hc')
        have hA := ha c' hc'
        have hB := hb c' hc'
        simp only [chainLabs, List.mem_cons, List.not_mem_nil, or_false] at hx hx'
        rcases hx with rfl | rfl | rfl | rfl <;> rcases hx' with h | h <;>
          first | exact hA.1 h | exact hA.2.1 h | exact hB.1 h | exact hB.2.1 h |
            (simp [h] at h'))
      (fun x hx hxa => by
        obtain ⟨c', hc', hx'⟩ := List.mem_flatMap.1 hx
        have h' := hio c' (List.mem_cons_of_mem _ hc') c List.mem_cons_self
        have hA := ha c' hc'
        have hB := hb c' hc'
        simp only [chainLabs, List.mem_cons, List.not_mem_nil, or_false] at hx' hxa
        rcases hx' with rfl | rfl | rfl | rfl <;> rcases hxa with h | h <;>
          first | exact hA.1 h.symm | exact hA.2.1 h.symm | exact hB.1 h.symm |
            exact hB.2.1 h.symm | (simp [h] at h'))]
    rw [ih (fun c' hc' => hD c' (List.mem_cons_of_mem _ hc')) hrest
      (fun c₁ h₁ c₂ h₂ => hio c₁ (List.mem_cons_of_mem _ h₁) c₂ (List.mem_cons_of_mem _ h₂))
      (fun c' hc' => hz c' (List.mem_cons_of_mem _ hc'))
      (fun c' hc' => hi c' (List.mem_cons_of_mem _ hc'))]
    congr 1
    obtain ⟨hDa, hDb⟩ := hD c List.mem_cons_self
    have hzc := hz c List.mem_cons_self
    have hic := hi c List.mem_cons_self
    simp only [sumOver, chainVal, hDa, hDb, Function.update_self, Function.update_of_ne hab,
      Function.update_of_ne (Ne.symm hab)]
    have e1 : c.1 ≠ c.2.1.1 := fun h => hc.1 (by simp [h])
    have e2 : c.1 ≠ c.2.1.2 := fun h => hc.1 (by simp [h])
    have e3 : c.2.2 ≠ c.2.1.1 := fun h => hc.2 (by simp [h])
    have e4 : c.2.2 ≠ c.2.1.2 := fun h => hc.2 (by simp [h])
    simp only [Function.update_of_ne e1, Function.update_of_ne e2, Function.update_of_ne e3,
      Function.update_of_ne e4, hzc]
    exact fou_cube d hd _ _ hic hk

theorem prod_flatMap' {α M : Type*} [CommMonoid M] (cs : List α) (g : α → List Node) (f : Node → M) :
    ((cs.flatMap g).map f).prod = (cs.map fun c => ((g c).map f).prod).prod := by
  induction cs with
  | nil => simp
  | cons c rest ih => simp [List.flatMap_cons, List.prod_append, ih]

/-- The input chain `F ; F ; F` of `c` as nodes. -/
def chainNodes (dz : ℕ) (c : ℕ × (ℕ × ℕ) × ℕ) : List Node :=
  [⟨.F, [c.2.1.1], [c.1], dz, 0⟩, ⟨.F, [c.2.1.2], [c.2.1.1], dz, 0⟩, ⟨.F, [c.2.2], [c.2.1.2], dz, 0⟩]

/-- The X spider is the Z spider with `F` on each output (pairs `(z, o)` of `po`) and
`F ; F ; F` on each input (chains `(i, (a, b), z)` of `cs`). -/
theorem xDef (E : Env) (a b dz : ℕ) (po : List (ℕ × ℕ)) (cs : List (ℕ × (ℕ × ℕ) × ℕ))
    (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hnd : (po.map Prod.fst ++ cs.map (fun c => c.2.2) ++
      cs.flatMap fun c => [c.2.1.1, c.2.1.2]).Nodup)
    (hdis : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, x ∉ po.map Prod.fst ++
      cs.map (fun c => c.2.2) ++ cs.flatMap fun c => [c.2.1.1, c.2.1.2])
    (hface : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, x ∈ iface ∧ labIdx E.labs x = dz)
    (hint : ∀ x ∈ po.map Prod.fst ++ cs.map (fun c => c.2.2) ++
      cs.flatMap (fun c => [c.2.1.1, c.2.1.2]), labIdx E.labs x = dz)
    (hb : ∀ k < E.dimv dz, ex (E.phase b k) = ex (E.phase a k)) :
    LocalEq E [⟨.X, po.map Prod.snd, cs.map Prod.fst, dz, a⟩] []
      (⟨.Z, po.map Prod.fst, cs.map (fun c => c.2.2), dz, b⟩ ::
        ((po.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz)))
      (po.map Prod.fst ++ cs.map (fun c => c.2.2) ++ cs.flatMap fun c => [c.2.1.1, c.2.1.2])
      iface 1 1 := by
  intro τ hτ
  have hr : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, τ x < E.dimv dz := fun x hx => by
    have := hτ x (hface x hx).1; rwa [labDim_of E x dz (hface x hx).2] at this
  set zs := po.map Prod.fst
  set zi := cs.map fun c => c.2.2
  set ws := cs.flatMap fun c => [c.2.1.1, c.2.1.2]
  -- the right side's entry
  have hR : ∀ ρ : ℕ → ℕ, prodVal E (⟨.Z, zs, zi, dz, b⟩ ::
      ((po.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz))) ρ =
      ∑ k ∈ range (E.dimv dz), (if ∀ v ∈ zs ++ zi, ρ v = k then ex (E.phase b k) *
        (po.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod else 0) *
        (cs.map (chainVal (E.dimv dz) ρ)).prod := by
    intro ρ
    simp only [prodVal, List.map_cons, List.prod_cons, List.map_append, List.prod_append,
      prod_flatMap', List.map_map]
    simp only [val, tZ, Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => ?_
    have e : (∀ v ∈ zs.map ρ ++ zi.map ρ, v = k) ↔ ∀ v ∈ zs ++ zi, ρ v = k := by
      constructor
      · intro h v hv
        rcases List.mem_append.1 hv with hv | hv
        · exact h _ (List.mem_append_left _ (List.mem_map_of_mem hv))
        · exact h _ (List.mem_append_right _ (List.mem_map_of_mem hv))
      · intro h v hv
        rcases List.mem_append.1 hv with hv | hv <;> obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv
        · exact h x (List.mem_append_left _ hx)
        · exact h x (List.mem_append_right _ hx)
    simp only [e]
    split_ifs
    · rw [mul_assoc]
      congr 2
      refine congrArg List.prod (List.map_congr_left fun c _ => ?_)
      simp [val, chainNodes, chainVal]
    · simp
  have hnd' := hnd
  rw [List.nodup_append, List.nodup_append] at hnd'
  obtain ⟨⟨hzs, hzi, hzz⟩, hws, hzw⟩ := hnd'
  have hk_in : ∀ k ∈ range (E.dimv dz), ∀ x ∈ zs ++ zi, k < labDim E x := fun k hk x hx => by
    rw [labDim_of E x dz (hint x (List.mem_append_left _ hx))]; exact Finset.mem_range.1 hk
  simp only [one_mul]
  rw [show prodVal E (⟨.Z, zs, zi, dz, b⟩ ::
      ((po.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz))) = _ from
    funext hR, sumOver_fsum]
  simp only [sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, val, tX]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.1 hk
  rw [sumOver_append]
  set G : (ℕ → ℕ) → ℂ := fun ρ => if ∀ v ∈ zs ++ zi, ρ v = k then ex (E.phase b k) *
    (po.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod else 0
  set I : (ℕ → ℕ) → ℂ := fun ρ => (cs.map (chainVal (E.dimv dz) ρ)).prod
  have hG : Reads (zs ++ zi ++ po.map Prod.snd) G := by
    intro ρ ρ' h
    simp only [G]
    have e1 : (∀ v ∈ zs ++ zi, ρ v = k) ↔ ∀ v ∈ zs ++ zi, ρ' v = k :=
      ⟨fun h' v hv => h v (List.mem_append_left _ hv) ▸ h' v hv,
        fun h' v hv => (h v (List.mem_append_left _ hv)).symm ▸ h' v hv⟩
    simp only [e1]
    congr 2
    refine congrArg List.prod (List.map_congr_left fun p hp => ?_)
    rw [h p.2 (List.mem_append_right _ (List.mem_map_of_mem hp)),
      h p.1 (List.mem_append_left _ (List.mem_append_left _ (List.mem_map_of_mem hp)))]
  rw [show sumOver (labDim E) ws (fun ρ => G ρ * I ρ) = fun ρ => G ρ * sumOver (labDim E) ws I ρ
    from funext fun ρ => sumOver_const_mul _ ws _ G I hG (fun x hx hxw => by
      rcases List.mem_append.1 hx with hx | hx
      · exact hzw x hx x hxw rfl
      · exact hdis x (List.mem_append_left _ hx) (List.mem_append_right _ hxw)) ρ]
  rw [show (fun ρ => G ρ * sumOver (labDim E) ws I ρ) = fun ρ => if ∀ v ∈ zs ++ zi, ρ v = k then
      ex (E.phase b k) * (po.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod *
        sumOver (labDim E) ws I ρ else 0 from funext fun ρ => by
      simp only [G]; split_ifs <;> simp]
  rw [sumOver_pinK (D := labDim E) (zs ++ zi) (List.nodup_append.2 ⟨hzs, hzi, hzz⟩) k
    (hk_in k hk)]
  set ρk : ℕ → ℕ := fun x => if x ∈ zs ++ zi then k else τ x
  have hρz : ∀ x ∈ zs ++ zi, ρk x = k := fun x hx => by
    show (if x ∈ zs ++ zi then k else τ x) = k; rw [if_pos hx]
  have hρo : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, ρk x = τ x := fun x hx => by
    have : x ∉ zs ++ zi := fun h => hdis x hx (List.mem_append_left _ h)
    show (if x ∈ zs ++ zi then k else τ x) = τ x; rw [if_neg this]
  have hO : (po.map fun p => fou (E.dimv dz) (ρk p.2) (ρk p.1)) =
      ((po.map Prod.snd).map τ).map fun o => fou (E.dimv dz) o k := by
    simp only [List.map_map]
    refine List.map_congr_left fun p hp => ?_
    simp only [Function.comp]
    rw [hρo p.2 (List.mem_append_left _ (List.mem_map_of_mem hp)),
      hρz p.1 (List.mem_append_left _ (List.mem_map_of_mem hp))]
  have hS : sumOver (labDim E) ws I ρk =
      ((cs.map Prod.fst).map τ |>.map fun i => (starRingEnd ℂ) (fou (E.dimv dz) i k)).prod := by
    rw [sumOver_chains (D := labDim E) (E.dimv dz) hd k hk' cs
      (fun c hc => ⟨labDim_of E _ dz (hint _ (List.mem_append_right _ (List.mem_flatMap.2
          ⟨c, hc, by simp⟩))), labDim_of E _ dz (hint _ (List.mem_append_right _
          (List.mem_flatMap.2 ⟨c, hc, by simp⟩)))⟩)
      hws
      (fun c hc c' hc' => ⟨fun h => hdis c.1 (List.mem_append_right _ (List.mem_map_of_mem hc))
          (List.mem_append_right _ (List.mem_flatMap.2 ⟨c', hc', h⟩)),
        fun h => hzw c.2.2 (List.mem_append_right _ (List.mem_map_of_mem hc)) c.2.2
          (List.mem_flatMap.2 ⟨c', hc', h⟩) rfl⟩)
      ρk (fun c hc => hρz _ (List.mem_append_right _ (List.mem_map_of_mem hc)))
      (fun c hc => by
        rw [hρo c.1 (List.mem_append_right _ (List.mem_map_of_mem hc))]
        exact hr c.1 (List.mem_append_right _ (List.mem_map_of_mem hc)))]
    simp only [List.map_map]
    refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    simp only [Function.comp]
    rw [hρo c.1 (List.mem_append_right _ (List.mem_map_of_mem hc))]
  rw [hO, hS, hb k hk']

/-- Summing the inner wires of chains `F ; F ; F`. -/
theorem sumOver_chains' {D : ℕ → ℕ} (d : ℕ) (hd : 0 < d) (cs : List (ℕ × (ℕ × ℕ) × ℕ))
    (hD : ∀ c ∈ cs, D c.2.1.1 = d ∧ D c.2.1.2 = d)
    (hnd : (cs.flatMap fun c => [c.2.1.1, c.2.1.2]).Nodup)
    (hio : ∀ c ∈ cs, ∀ c' ∈ cs, c.1 ∉ [c'.2.1.1, c'.2.1.2] ∧ c.2.2 ∉ [c'.2.1.1, c'.2.1.2])
    (ρ : ℕ → ℕ) (hz : ∀ c ∈ cs, ρ c.2.2 < d) (hi : ∀ c ∈ cs, ρ c.1 < d) :
    sumOver D (cs.flatMap fun c => [c.2.1.1, c.2.1.2])
        (fun τ => (cs.map (chainVal d τ)).prod) ρ =
      (cs.map fun c => (starRingEnd ℂ) (fou d (ρ c.1) (ρ c.2.2))).prod := by
  induction cs with
  | nil => simp [sumOver]
  | cons c rest ih =>
    simp only [List.flatMap_cons, List.map_cons, List.prod_cons]
    have hnd' := hnd
    simp only [List.flatMap_cons, List.cons_append, List.nil_append, List.nodup_cons,
      List.mem_cons, List.mem_flatMap, not_or, not_exists, not_and] at hnd'
    obtain ⟨⟨hab, ha⟩, ⟨hb, hrest⟩⟩ := hnd'
    have hc := hio c List.mem_cons_self c List.mem_cons_self
    rw [sumOver_mul_split (D := D) [c.2.1.1, c.2.1.2] _ (chainLabs c)
      (rest.flatMap chainLabs) (fun τ => chainVal d τ c) (fun τ => (rest.map (chainVal d τ)).prod)
      (fun τ τ' h => by
        simp only [chainVal]
        rw [h c.1 (by simp [chainLabs]), h c.2.1.1 (by simp [chainLabs]),
          h c.2.1.2 (by simp [chainLabs]), h c.2.2 (by simp [chainLabs])])
      (fun τ τ' h => by
        refine congrArg List.prod (List.map_congr_left fun c' hc' => ?_)
        have m : ∀ x ∈ chainLabs c', x ∈ rest.flatMap chainLabs := fun x hx =>
          List.mem_flatMap.2 ⟨c', hc', hx⟩
        simp only [chainVal]
        rw [h c'.1 (m _ (by simp [chainLabs])), h c'.2.1.1 (m _ (by simp [chainLabs])),
          h c'.2.1.2 (m _ (by simp [chainLabs])), h c'.2.2 (m _ (by simp [chainLabs]))])
      (fun x hx hxr => by
        obtain ⟨c', hc', hx'⟩ := List.mem_flatMap.1 hxr
        have h' := hio c List.mem_cons_self c' (List.mem_cons_of_mem _ hc')
        have hA := ha c' hc'
        have hB := hb c' hc'
        simp only [chainLabs, List.mem_cons, List.not_mem_nil, or_false] at hx hx'
        rcases hx with rfl | rfl | rfl | rfl <;> rcases hx' with h | h <;>
          first | exact hA.1 h | exact hA.2.1 h | exact hB.1 h | exact hB.2.1 h |
            (simp [h] at h'))
      (fun x hx hxa => by
        obtain ⟨c', hc', hx'⟩ := List.mem_flatMap.1 hx
        have h' := hio c' (List.mem_cons_of_mem _ hc') c List.mem_cons_self
        have hA := ha c' hc'
        have hB := hb c' hc'
        simp only [chainLabs, List.mem_cons, List.not_mem_nil, or_false] at hx' hxa
        rcases hx' with rfl | rfl | rfl | rfl <;> rcases hxa with h | h <;>
          first | exact hA.1 h.symm | exact hA.2.1 h.symm | exact hB.1 h.symm |
            exact hB.2.1 h.symm | (simp [h] at h'))]
    rw [ih (fun c' hc' => hD c' (List.mem_cons_of_mem _ hc')) hrest
      (fun c₁ h₁ c₂ h₂ => hio c₁ (List.mem_cons_of_mem _ h₁) c₂ (List.mem_cons_of_mem _ h₂))
      (fun c' hc' => hz c' (List.mem_cons_of_mem _ hc'))
      (fun c' hc' => hi c' (List.mem_cons_of_mem _ hc'))]
    congr 1
    obtain ⟨hDa, hDb⟩ := hD c List.mem_cons_self
    have hzc := hz c List.mem_cons_self
    have hic := hi c List.mem_cons_self
    simp only [sumOver, chainVal, hDa, hDb, Function.update_self, Function.update_of_ne hab,
      Function.update_of_ne (Ne.symm hab)]
    have e1 : c.1 ≠ c.2.1.1 := fun h => hc.1 (by simp [h])
    have e2 : c.1 ≠ c.2.1.2 := fun h => hc.1 (by simp [h])
    have e3 : c.2.2 ≠ c.2.1.1 := fun h => hc.2 (by simp [h])
    have e4 : c.2.2 ≠ c.2.1.2 := fun h => hc.2 (by simp [h])
    simp only [Function.update_of_ne e1, Function.update_of_ne e2, Function.update_of_ne e3,
      Function.update_of_ne e4]
    exact fou_cube d hd _ _ hic hzc

theorem neg_neg_mod (d k : ℕ) (hd : 0 < d) (hk : k < d) : (d - (d - k) % d) % d = k := by
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · simp
  · rw [Nat.mod_eq_of_lt (by omega : d - k < d), Nat.mod_eq_of_lt (by omega : d - (d - k) < d)]
    omega

theorem fou_neg (d k x : ℕ) (hd : 0 < d) (hk : k < d) :
    fou d ((d - k) % d) x = (starRingEnd ℂ) (fou d k x) := by
  simp only [fou, map_div₀, map_pow, conj_om, conj_sqrt]
  congr 1
  have h1 : om d ^ ((d - k) % d * x) * om d ^ (k * x) = 1 := by
    rw [← pow_add, show (d - k) % d * x + k * x = ((d - k) % d + k) * x by ring]
    exact om_pow_zero_mod d hd _ _ (by rw [add_comm]; exact antipode_spec d k hd hk)
  rw [inv_pow]; exact eq_inv_of_mul_eq_one_left h1

/-- The X spider of phase `a` is the Z spider of the negated phase `b` with `F` into each
input (pairs `(i, z)` of `pi`) and `F ; F ; F` out of each output (chains `(z, (a, b), o)` of
`cs`). -/
theorem xDefT (E : Env) (a b dz : ℕ) (pi : List (ℕ × ℕ)) (cs : List (ℕ × (ℕ × ℕ) × ℕ))
    (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hnd : (cs.map Prod.fst ++ pi.map Prod.snd ++
      cs.flatMap fun c => [c.2.1.1, c.2.1.2]).Nodup)
    (hdis : ∀ x ∈ cs.map (fun c => c.2.2) ++ pi.map Prod.fst, x ∉ cs.map Prod.fst ++
      pi.map Prod.snd ++ cs.flatMap fun c => [c.2.1.1, c.2.1.2])
    (hface : ∀ x ∈ cs.map (fun c => c.2.2) ++ pi.map Prod.fst,
      x ∈ iface ∧ labIdx E.labs x = dz)
    (hint : ∀ x ∈ cs.map Prod.fst ++ pi.map Prod.snd ++
      cs.flatMap (fun c => [c.2.1.1, c.2.1.2]), labIdx E.labs x = dz)
    (hb : ∀ k < E.dimv dz,
      ex (E.phase b k) = ex (E.phase a ((E.dimv dz - k) % E.dimv dz))) :
    LocalEq E [⟨.X, cs.map (fun c => c.2.2), pi.map Prod.fst, dz, a⟩] []
      (⟨.Z, cs.map Prod.fst, pi.map Prod.snd, dz, b⟩ ::
        ((pi.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz)))
      (cs.map Prod.fst ++ pi.map Prod.snd ++ cs.flatMap fun c => [c.2.1.1, c.2.1.2])
      iface 1 1 := by
  intro τ hτ
  have hr : ∀ x ∈ cs.map (fun c => c.2.2) ++ pi.map Prod.fst, τ x < E.dimv dz := fun x hx => by
    have := hτ x (hface x hx).1; rwa [labDim_of E x dz (hface x hx).2] at this
  set zs := cs.map Prod.fst
  set zi := pi.map Prod.snd
  set ws := cs.flatMap fun c => [c.2.1.1, c.2.1.2]
  have hR : ∀ ρ : ℕ → ℕ, prodVal E (⟨.Z, zs, zi, dz, b⟩ ::
      ((pi.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz))) ρ =
      ∑ k ∈ range (E.dimv dz), (if ∀ v ∈ zs ++ zi, ρ v = k then ex (E.phase b k) *
        (pi.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod else 0) *
        (cs.map (chainVal (E.dimv dz) ρ)).prod := by
    intro ρ
    simp only [prodVal, List.map_cons, List.prod_cons, List.map_append, List.prod_append,
      prod_flatMap', List.map_map]
    simp only [val, tZ, Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => ?_
    have e : (∀ v ∈ zs.map ρ ++ zi.map ρ, v = k) ↔ ∀ v ∈ zs ++ zi, ρ v = k := by
      constructor
      · intro h v hv
        rcases List.mem_append.1 hv with hv | hv
        · exact h _ (List.mem_append_left _ (List.mem_map_of_mem hv))
        · exact h _ (List.mem_append_right _ (List.mem_map_of_mem hv))
      · intro h v hv
        rcases List.mem_append.1 hv with hv | hv <;> obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv
        · exact h x (List.mem_append_left _ hx)
        · exact h x (List.mem_append_right _ hx)
    simp only [e]
    split_ifs
    · rw [mul_assoc]
      congr 2
      refine congrArg List.prod (List.map_congr_left fun c _ => ?_)
      simp [val, chainNodes, chainVal]
    · simp
  have hnd' := hnd
  rw [List.nodup_append, List.nodup_append] at hnd'
  obtain ⟨⟨hzs, hzi, hzz⟩, hws, hzw⟩ := hnd'
  have hk_in : ∀ k ∈ range (E.dimv dz), ∀ x ∈ zs ++ zi, k < labDim E x := fun k hk x hx => by
    rw [labDim_of E x dz (hint x (List.mem_append_left _ hx))]; exact Finset.mem_range.1 hk
  simp only [one_mul]
  rw [show prodVal E (⟨.Z, zs, zi, dz, b⟩ ::
      ((pi.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz))) = _ from
    funext hR, sumOver_fsum]
  simp only [sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, val, tX]
  -- reindex the left side by k ↦ -k
  rw [← Finset.sum_nbij' (fun k => (E.dimv dz - k) % E.dimv dz)
    (fun k => (E.dimv dz - k) % E.dimv dz)
    (fun k _ => Finset.mem_range.2 (Nat.mod_lt _ hd)) (fun k _ => Finset.mem_range.2 (Nat.mod_lt _ hd))
    (fun k hk => neg_neg_mod _ _ hd (Finset.mem_range.1 hk))
    (fun k hk => neg_neg_mod _ _ hd (Finset.mem_range.1 hk)) (fun _ _ => rfl)]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.1 hk
  rw [sumOver_append]
  set G : (ℕ → ℕ) → ℂ := fun ρ => if ∀ v ∈ zs ++ zi, ρ v = k then ex (E.phase b k) *
    (pi.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod else 0
  set I : (ℕ → ℕ) → ℂ := fun ρ => (cs.map (chainVal (E.dimv dz) ρ)).prod
  have hG : Reads (zs ++ zi ++ pi.map Prod.fst) G := by
    intro ρ ρ' h
    simp only [G]
    have e1 : (∀ v ∈ zs ++ zi, ρ v = k) ↔ ∀ v ∈ zs ++ zi, ρ' v = k :=
      ⟨fun h' v hv => h v (List.mem_append_left _ hv) ▸ h' v hv,
        fun h' v hv => (h v (List.mem_append_left _ hv)).symm ▸ h' v hv⟩
    simp only [e1]
    congr 2
    refine congrArg List.prod (List.map_congr_left fun p hp => ?_)
    rw [h p.1 (List.mem_append_right _ (List.mem_map_of_mem hp)),
      h p.2 (List.mem_append_left _ (List.mem_append_right _ (List.mem_map_of_mem hp)))]
  rw [show sumOver (labDim E) ws (fun ρ => G ρ * I ρ) = fun ρ => G ρ * sumOver (labDim E) ws I ρ
    from funext fun ρ => sumOver_const_mul _ ws _ G I hG (fun x hx hxw => by
      rcases List.mem_append.1 hx with hx | hx
      · exact hzw x hx x hxw rfl
      · exact hdis x (List.mem_append_right _ hx) (List.mem_append_right _ hxw)) ρ]
  rw [show (fun ρ => G ρ * sumOver (labDim E) ws I ρ) = fun ρ => if ∀ v ∈ zs ++ zi, ρ v = k then
      ex (E.phase b k) * (pi.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod *
        sumOver (labDim E) ws I ρ else 0 from funext fun ρ => by
      simp only [G]; split_ifs <;> simp]
  rw [sumOver_pinK (D := labDim E) (zs ++ zi) (List.nodup_append.2 ⟨hzs, hzi, hzz⟩) k
    (hk_in k hk)]
  set ρk : ℕ → ℕ := fun x => if x ∈ zs ++ zi then k else τ x
  have hρz : ∀ x ∈ zs ++ zi, ρk x = k := fun x hx => by
    show (if x ∈ zs ++ zi then k else τ x) = k; rw [if_pos hx]
  have hρo : ∀ x ∈ cs.map (fun c => c.2.2) ++ pi.map Prod.fst, ρk x = τ x := fun x hx => by
    have : x ∉ zs ++ zi := fun h => hdis x hx (List.mem_append_left _ h)
    show (if x ∈ zs ++ zi then k else τ x) = τ x; rw [if_neg this]
  have hO : (pi.map fun p => fou (E.dimv dz) (ρk p.2) (ρk p.1)) =
      ((pi.map Prod.fst).map τ).map fun i => (starRingEnd ℂ)
        (fou (E.dimv dz) i ((E.dimv dz - k) % E.dimv dz)) := by
    simp only [List.map_map]
    refine List.map_congr_left fun p hp => ?_
    simp only [Function.comp]
    rw [hρo p.1 (List.mem_append_right _ (List.mem_map_of_mem hp)),
      hρz p.2 (List.mem_append_right _ (List.mem_map_of_mem hp)), fou_symm _ (τ p.1),
      fou_neg _ _ _ hd hk', Complex.conj_conj]
  have hS : sumOver (labDim E) ws I ρk = (((cs.map fun c => c.2.2).map τ).map
      fun o => fou (E.dimv dz) o ((E.dimv dz - k) % E.dimv dz)).prod := by
    rw [sumOver_chains' (D := labDim E) (E.dimv dz) hd cs
      (fun c hc => ⟨labDim_of E _ dz (hint _ (List.mem_append_right _ (List.mem_flatMap.2
          ⟨c, hc, by simp⟩))), labDim_of E _ dz (hint _ (List.mem_append_right _
          (List.mem_flatMap.2 ⟨c, hc, by simp⟩)))⟩)
      hws
      (fun c hc c' hc' => ⟨fun h => hzw c.1 (List.mem_append_left _ (List.mem_map_of_mem hc))
          c.1 (List.mem_flatMap.2 ⟨c', hc', h⟩) rfl,
        fun h => hdis c.2.2 (List.mem_append_left _ (List.mem_map_of_mem hc))
          (List.mem_append_right _ (List.mem_flatMap.2 ⟨c', hc', h⟩))⟩)
      ρk
      (fun c hc => by
        rw [hρo c.2.2 (List.mem_append_left _ (List.mem_map_of_mem hc))]
        exact hr c.2.2 (List.mem_append_left _ (List.mem_map_of_mem hc)))
      (fun c hc => by
        rw [hρz c.1 (List.mem_append_left _ (List.mem_map_of_mem hc))]; exact hk')]
    simp only [List.map_map]
    refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    simp only [Function.comp]
    rw [hρz c.1 (List.mem_append_left _ (List.mem_map_of_mem hc)),
      hρo c.2.2 (List.mem_append_left _ (List.mem_map_of_mem hc)), ← fou_neg _ _ _ hd hk',
      fou_symm]
  rw [hO, hS, hb k hk']
  ring

theorem LocalEq.trans' {E : Env} {L M R : List Node} {iL iM iR iface : List ℕ}
    (h₁ : LocalEq E L iL M iM iface 1 1) (h₂ : LocalEq E M iM R iR iface 1 1) :
    LocalEq E L iL R iR iface 1 1 := fun τ hτ => (h₁ τ hτ).trans (h₂ τ hτ)

theorem LocalEq.symm' {E : Env} {L R : List Node} {iL iR iface : List ℕ} {kL kR : ℂ}
    (h : LocalEq E L iL R iR iface kL kR) : LocalEq E R iR L iL iface kR kL :=
  fun τ hτ => (h τ hτ).symm

/-- Phases `K_{j'}` with `j' + j ≡ 0` at `k` are the phases `K_j` at `-k`. -/
theorem kjNeg (d j j' : ℕ) (hd : 0 < d) (hjj : (j' + j) % d = 0) (φp φg : ℕ → ℝ)
    (hp : ∀ q < d, ex (φp q) = om d ^ (q * j))
    (hg : ∀ q < d, ex (φg q) = om d ^ (q * j')) :
    ∀ k < d, ex (φg k) = ex (φp ((d - k) % d)) := by
  intro k hk
  rw [hg k hk, hp _ (Nat.mod_lt _ hd)]
  have hz : om d ≠ 0 := om_ne_zero d hd
  have h1 : om d ^ (k * j') * om d ^ (k * j) = 1 := by
    rw [← pow_add, ← mul_add, mul_comm]
    exact om_pow_zero_mod d hd _ _ hjj
  have h2 : om d ^ ((d - k) % d * j) * om d ^ (k * j) = 1 := by
    rw [← pow_add, show (d - k) % d * j + k * j = ((d - k) % d + k) * j by ring]
    exact om_pow_zero_mod d hd _ _ (by rw [add_comm]; exact antipode_spec d k hd hk)
  have hne : om d ^ (k * j) ≠ 0 := pow_ne_zero _ hz
  exact mul_right_cancel₀ hne (h1.trans h2.symm)


/-- The multiplier `e`, a plain green dot (`a` -> `ws`) into the phaseless X spider
(`ws` -> `b`), at an input value `v`: `[b ≡ e v]` times `d^((1 - e)/2)`. -/
theorem multiplier_val (E : Env) (pz px dz a b : ℕ) (ws : List ℕ) (τ : ℕ → ℕ)
    (hd : 0 < E.dimv dz) (hnd : ws.Nodup) (ha : a ∉ ws) (hb : b ∉ ws)
    (hws : ∀ w ∈ ws, labIdx E.labs w = dz) (ra : τ a < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    sumOver (labDim E) ws (prodVal E [⟨.Z, ws, [a], dz, pz⟩, ⟨.X, [b], ws, dz, px⟩]) τ =
      (rt (E.dimv dz))⁻¹ ^ (1 + ws.length) *
        (if τ b % E.dimv dz = (ws.length * τ a) % E.dimv dz then (E.dimv dz : ℂ) else 0) := by
  have hD : ∀ w ∈ ws, labDim E w = E.dimv dz := fun w hw => labDim_of E w dz (hws w hw)
  have key : sumOver (labDim E) ws (prodVal E [⟨.Z, ws, [a], dz, pz⟩, ⟨.X, [b], ws, dz, px⟩]) τ =
      sumOver (labDim E) ws (fun ρ => if ∀ l ∈ ws, ρ l = ρ ((fun _ => a) l) then
        tX (E.dimv dz) (E.phase px) [ρ b] (ws.map ρ) else 0) τ := by
    refine sumOver_congr _ _ _ _ _ fun ρ hρ _ => ?_
    have ea : ρ a = τ a := hρ a ha
    have ra' : ρ a < E.dimv dz := ea ▸ ra
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      val, List.cons_append, List.nil_append]
    rw [tZ_at _ _ _ (ρ a) (by simp), hz _ ra']
    have hz' : (ρ a < E.dimv dz ∧ ∀ x ∈ ws.map ρ ++ [ρ a], x = ρ a) ↔ ∀ w ∈ ws, ρ w = ρ a := by
      constructor
      · rintro ⟨-, h⟩ w hw
        exact h _ (List.mem_append_left _ (List.mem_map_of_mem hw))
      · intro h
        refine ⟨ra', fun x hx => ?_⟩
        rcases List.mem_append.1 hx with hx | hx
        · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hx
          exact h w hw
        · simpa using hx
    by_cases hc : ∀ w ∈ ws, ρ w = ρ a
    · rw [if_pos (hz'.2 hc), if_pos hc, one_mul]
    · rw [if_neg (fun h => hc (hz'.1 h)), if_neg hc, zero_mul]
  rw [key, sumOver_pin (labDim E) ws hnd (fun _ => a) (fun _ _ => ha) _ τ
    (fun w hw => by rw [hD w hw]; exact ra)]
  simp only
  have eb : (if b ∈ ws then τ a else τ b) = τ b := by rw [if_neg hb]
  rw [eb, tX_kj _ 0 hd _ (fun k hk => by rw [hx k hk]; simp)]
  have hm : ws.map (fun x => if x ∈ ws then τ a else τ x) = List.replicate ws.length (τ a) := by
    rw [List.eq_replicate_iff]
    refine ⟨by simp, fun x hx => ?_⟩
    obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hx
    simp [hw]
  rw [hm]
  simp [List.sum_replicate]

theorem mult_neg_iff (d e a b : ℕ) (he : (e + 1) % d = 0) (hb : b < d) :
    b % d = (e * a) % d ↔ (a + b) % d = 0 := by
  have h0 : (e * a + a) % d = 0 := by
    rw [show e * a + a = (e + 1) * a by ring, Nat.mul_mod, he, zero_mul, Nat.zero_mod]
  constructor
  · intro h
    have : (a + b) % d = (a + e * a) % d := by rw [Nat.add_mod, h, ← Nat.add_mod]
    rw [this, add_comm, h0]
  · intro h
    have h1 : (b + a) % d = (e * a + a) % d := by rw [add_comm, h, h0]
    exact Nat.ModEq.add_right_cancel' a h1

/-- [P1] (P1): `F ; F` is the multiplier `e` when `e + 1 ≡ 0`, times `d^((e - 1)/2)`. -/
theorem fourierMult (E : Env) (pz px dz a b w : ℕ) (ws iface : List ℕ) (hd : 0 < E.dimv dz)
    (hnd : ws.Nodup) (ha : a ∉ ws) (hb : b ∉ ws) (hw : w ≠ a ∧ w ≠ b)
    (hab : a ∈ iface ∧ labIdx E.labs a = dz) (hbb : b ∈ iface ∧ labIdx E.labs b = dz)
    (hwl : labIdx E.labs w = dz) (hws : ∀ x ∈ ws, labIdx E.labs x = dz) (hne : ws ≠ [])
    (he : (ws.length + 1) % E.dimv dz = 0)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (s : ℂ) (hs : s = rt (E.dimv dz) ^ (ws.length - 1)) :
    LocalEq E [⟨.F, [w], [a], dz, 0⟩, ⟨.F, [b], [w], dz, 0⟩] [w]
      [⟨.Z, ws, [a], dz, pz⟩, ⟨.X, [b], ws, dz, px⟩] ws iface 1 s := by
  subst hs
  intro τ hτ
  have ra := hτ a hab.1
  have rb := hτ b hbb.1
  rw [labDim_of E a dz hab.2] at ra
  rw [labDim_of E b dz hbb.2] at rb
  rw [multiplier_val E pz px dz a b ws τ hd hnd ha hb hws ra hz hx]
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E w dz hwl, Function.update_self,
    Function.update_of_ne (Ne.symm hw.1), Function.update_of_ne (Ne.symm hw.2)]
  rw [Finset.sum_congr rfl fun j _ => mul_comm (fou (E.dimv dz) j (τ a)) _, fou_sq _ hd]
  simp only [mult_neg_iff _ _ _ _ he rb, add_comm (τ b) (τ a)]
  have hr := rt_ne (E.dimv dz) hd
  have hlen : 1 ≤ ws.length := List.length_pos_iff.2 hne
  split_ifs
  · rw [← rt_sq (E.dimv dz)]
    obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hlen
    rw [hn, show 1 + n - 1 = n by omega]
    have h2 : rt (E.dimv dz) ^ n ≠ 0 := pow_ne_zero _ hr
    have h3 : rt (E.dimv dz) ^ 2 ≠ 0 := pow_ne_zero _ hr
    field_simp
    rw [div_pow, one_pow, show 1 + (1 + n) = n + 2 by ring, pow_add]
    field_simp
  · simp

/-- The multiplier as a function of its input and output values, for `sumOver_const_mul`. -/
theorem multiplier_reads (E : Env) (pz px dz a b : ℕ) (ws : List ℕ) :
    Reads (a :: b :: ws) (prodVal E [⟨.Z, ws, [a], dz, pz⟩, ⟨.X, [b], ws, dz, px⟩]) := by
  intro ρ ρ' h
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, val]
  have hw : ws.map ρ = ws.map ρ' := List.map_congr_left fun w hw => h w (by simp [hw])
  rw [hw, h a (by simp), h b (by simp)]

/-- [P2] (MW): the box `1` and the multiplier `e` before a W node is the box `1`, the W node and
the multiplier `e` on each output, when `e ≢ 0`. -/
theorem qfMW (E : Env) (dz i u v x y u' p q z0 x0 z1 x1 z2 x2 : ℕ) (m m' : Node)
    (ws ws1 ws2 iface : List ℕ) (hd : 0 < E.dimv dz)
    (hm : isZc m = true) (hm' : isZc m' = true) (hdm : m.dim = dz) (hdm' : m'.dim = dz)
    (hlm : m.outs ++ m.ins = [u, i]) (hlm' : m'.outs ++ m'.ins = [u', i])
    (hndL : (u :: v :: ws).Nodup) (hndR : (u' :: p :: q :: (ws1 ++ ws2)).Nodup)
    (hoL : ∀ l ∈ u :: v :: ws, l ∉ [i, x, y]) (hoR : ∀ l ∈ u' :: p :: q :: (ws1 ++ ws2),
      l ∉ [i, x, y])
    (hxy : x ≠ y ∧ x ≠ i ∧ y ≠ i)
    (hl : ∀ l ∈ u :: v :: u' :: p :: q :: (ws ++ ws1 ++ ws2), labIdx E.labs l = dz)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz) (hx : x ∈ iface ∧ labIdx E.labs x = dz)
    (hy : y ∈ iface ∧ labIdx E.labs y = dz)
    (h1 : ws1.length = ws.length) (h2 : ws2.length = ws.length)
    (he : ws.length % E.dimv dz ≠ 0)
    (hbox : ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E m k = 0)
    (hc : ∀ k < E.dimv dz, zc E m' k = zc E m k)
    (hph : ∀ ph ∈ [z0, x0, z1, x1, z2, x2], ∀ k < E.dimv dz, ex (E.phase ph k) = 1)
    (s : ℂ) (hs : s = rt (E.dimv dz) ^ (ws.length - 1)) :
    LocalEq E [m, ⟨.Z, ws, [u], dz, z0⟩, ⟨.X, [v], ws, dz, x0⟩, ⟨.W, [x, y], [v], dz, 0⟩]
      (u :: v :: ws)
      [m', ⟨.W, [p, q], [u'], dz, 0⟩, ⟨.Z, ws1, [p], dz, z1⟩, ⟨.X, [x], ws1, dz, x1⟩,
        ⟨.Z, ws2, [q], dz, z2⟩, ⟨.X, [y], ws2, dz, x2⟩] (u' :: p :: q :: (ws1 ++ ws2))
      iface 1 s := by
  subst hs
  intro τ hτ
  have ri := hτ i hi.1
  have rx := hτ x hx.1
  have ry := hτ y hy.1
  rw [labDim_of E i dz hi.2] at ri
  rw [labDim_of E x dz hx.2] at rx
  rw [labDim_of E y dz hy.2] at ry
  set d := E.dimv dz with hdd
  set e := ws.length
  set R := (rt d)⁻¹ ^ (1 + e) * (d : ℂ)
  have hlab : ∀ l, l ∈ u :: v :: u' :: p :: q :: (ws ++ ws1 ++ ws2) → labDim E l = d :=
    fun l hl' => labDim_of E l dz (hl l hl')
  simp only [List.nodup_cons, List.mem_cons, not_or] at hndL hndR
  obtain ⟨⟨huv, huws⟩, hvws, hws⟩ := hndL
  obtain ⟨⟨hup, huq, hu12⟩, ⟨hpq, hp12⟩, hq12, h12⟩ := hndR
  -- the multiplier values
  have mult : ∀ (a b pz px : ℕ) (vs : List ℕ) (ρ : ℕ → ℕ), vs.Nodup → a ∉ vs → b ∉ vs →
      (∀ w ∈ vs, labIdx E.labs w = dz) → ρ a < d → vs.length = e →
      (∀ k < d, ex (E.phase pz k) = 1) → (∀ k < d, ex (E.phase px k) = 1) →
      sumOver (labDim E) vs (prodVal E [⟨.Z, vs, [a], dz, pz⟩, ⟨.X, [b], vs, dz, px⟩]) ρ =
        (rt d)⁻¹ ^ (1 + e) * (if ρ b % d = (e * ρ a) % d then (d : ℂ) else 0) := by
    intro a b pz px vs ρ hnd ha hb hvs ra hlen hz hx'
    rw [multiplier_val E pz px dz a b vs ρ hd hnd ha hb hvs ra hz hx', hlen]
  have hph' : ∀ ph ∈ [z0, x0, z1, x1, z2, x2], ∀ k < d, ex (E.phase ph k) = 1 := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨hz0, hx0, hz1, hx1, hz2, hx2⟩ := hph'
  simp only [List.mem_cons, List.mem_append, List.not_mem_nil, or_false] at hoL hoR
  have hm_val : ∀ ρ : ℕ → ℕ, ρ i = τ i → ρ u < d →
      val E ρ m = if ρ u = τ i then zc E m (τ i) else 0 := by
    intro ρ hρi hρu
    rw [val_zc _ _ m hm, hlm, hdm]
    simp only [List.map_cons, List.map_nil]
    rw [tZw_at _ _ _ (ρ i) (by simp), hρi]
    by_cases h : ρ u = τ i
    · rw [if_pos ⟨ri, by simp [h]⟩, if_pos h]
    · have : ¬ (τ i < d ∧ ∀ w ∈ [ρ u, τ i], w = τ i) := fun h' => h (h'.2 _ (by simp))
      rw [if_neg this, if_neg h]
  have hm_val' : ∀ ρ : ℕ → ℕ, ρ i = τ i → ρ u' < d →
      val E ρ m' = if ρ u' = τ i then zc E m (τ i) else 0 := by
    intro ρ hρi hρu
    rw [val_zc _ _ m' hm', hlm', hdm']
    simp only [List.map_cons, List.map_nil]
    rw [tZw_at _ _ _ (ρ i) (by simp), hρi, hc _ ri]
    by_cases h : ρ u' = τ i
    · rw [if_pos ⟨ri, by simp [h]⟩, if_pos h]
    · have : ¬ (τ i < d ∧ ∀ w ∈ [ρ u', τ i], w = τ i) := fun h' => h (h'.2 _ (by simp))
      rw [if_neg this, if_neg h]
  have hL : sumOver (labDim E) (u :: v :: ws) (prodVal E [m, ⟨.Z, ws, [u], dz, z0⟩,
      ⟨.X, [v], ws, dz, x0⟩, ⟨.W, [x, y], [v], dz, 0⟩]) τ =
      zc E m (τ i) * ((rt d)⁻¹ ^ (1 + e) * d) * tW d [τ x, τ y] ((e * τ i) % d) := by
    have split : prodVal E [m, ⟨.Z, ws, [u], dz, z0⟩, ⟨.X, [v], ws, dz, x0⟩,
        ⟨.W, [x, y], [v], dz, 0⟩] = fun ρ => (val E ρ m * val E ρ ⟨.W, [x, y], [v], dz, 0⟩) *
          prodVal E [⟨.Z, ws, [u], dz, z0⟩, ⟨.X, [v], ws, dz, x0⟩] ρ := by
      funext ρ; simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons,
        List.prod_nil]; ring
    have hG : Reads [u, i, x, y, v] (fun ρ => val E ρ m * val E ρ ⟨.W, [x, y], [v], dz, 0⟩) := by
      intro ρ ρ' h
      simp only
      rw [val_zc _ _ m hm, val_zc _ _ m hm, hlm]
      simp only [val, List.map_cons, List.map_nil]
      rw [h u (by simp), h i (by simp), h x (by simp), h y (by simp), h v (by simp)]
    simp only [sumOver, hlab u (by simp), hlab v (by simp)]
    rw [split]
    have inner : ∀ j ∈ range d, ∑ l ∈ range d, sumOver (labDim E) ws
        (fun ρ => (val E ρ m * val E ρ ⟨.W, [x, y], [v], dz, 0⟩) *
          prodVal E [⟨.Z, ws, [u], dz, z0⟩, ⟨.X, [v], ws, dz, x0⟩] ρ)
        (Function.update (Function.update τ u j) v l) =
        if j = τ i then zc E m (τ i) * ((rt d)⁻¹ ^ (1 + e) * d) *
          tW d [τ x, τ y] ((e * τ i) % d) else 0 := by
      intro j hj
      have hj' := Finset.mem_range.1 hj
      have step : ∀ l ∈ range d, sumOver (labDim E) ws
          (fun ρ => (val E ρ m * val E ρ ⟨.W, [x, y], [v], dz, 0⟩) *
            prodVal E [⟨.Z, ws, [u], dz, z0⟩, ⟨.X, [v], ws, dz, x0⟩] ρ)
          (Function.update (Function.update τ u j) v l) =
          (if j = τ i then zc E m (τ i) else 0) * tW d [τ x, τ y] l *
            ((rt d)⁻¹ ^ (1 + e) * (if l % d = (e * j) % d then (d : ℂ) else 0)) := by
        intro l hl'
        have hl'' := Finset.mem_range.1 hl'
        rw [sumOver_const_mul _ ws [u, i, x, y, v] _ _ hG (by
          intro z hz hzw
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hz
          rcases hz with rfl | rfl | rfl | rfl | rfl
          · exact huws hzw
          · exact hoL _ (Or.inr (Or.inr hzw)) (Or.inl rfl)
          · exact hoL _ (Or.inr (Or.inr hzw)) (Or.inr (Or.inl rfl))
          · exact hoL _ (Or.inr (Or.inr hzw)) (Or.inr (Or.inr rfl))
          · exact hvws hzw)]
        have nvi : v ≠ i := fun h => hoL v (Or.inr (Or.inl rfl)) (Or.inl h)
        have nvx : v ≠ x := fun h => hoL v (Or.inr (Or.inl rfl)) (Or.inr (Or.inl h))
        have nvy : v ≠ y := fun h => hoL v (Or.inr (Or.inl rfl)) (Or.inr (Or.inr h))
        have nui : u ≠ i := fun h => hoL u (Or.inl rfl) (Or.inl h)
        have nux : u ≠ x := fun h => hoL u (Or.inl rfl) (Or.inr (Or.inl h))
        have nuy : u ≠ y := fun h => hoL u (Or.inl rfl) (Or.inr (Or.inr h))
        rw [mult u v z0 x0 ws _ hws huws hvws (fun w hw => hl w (by simp [hw]))
          (by rw [Function.update_of_ne huv, Function.update_self]; exact hj') rfl hz0 hx0,
          hm_val _ (by rw [Function.update_of_ne (Ne.symm nvi), Function.update_of_ne (Ne.symm nui)])
            (by rw [Function.update_of_ne huv, Function.update_self]; exact hj')]
        simp only [val, List.map_cons, List.map_nil, Function.update_self,
          Function.update_of_ne huv, Function.update_of_ne (Ne.symm nvx),
          Function.update_of_ne (Ne.symm nvy), Function.update_of_ne (Ne.symm nux),
          Function.update_of_ne (Ne.symm nuy)]
        rfl
      rw [Finset.sum_congr rfl step]
      by_cases hji : j = τ i
      · have pick : ∀ l ∈ range d, (if j = τ i then zc E m (τ i) else 0) * tW d [τ x, τ y] l *
            ((rt d)⁻¹ ^ (1 + e) * (if l % d = (e * j) % d then (d : ℂ) else 0)) =
            if l = (e * τ i) % d then zc E m (τ i) * ((rt d)⁻¹ ^ (1 + e) * d) *
              tW d [τ x, τ y] ((e * τ i) % d) else 0 := by
          intro l hl'
          rw [Nat.mod_eq_of_lt (Finset.mem_range.1 hl'), if_pos hji, hji]
          by_cases h : l = (e * τ i) % d
          · rw [if_pos h, if_pos h, h]; ring
          · rw [if_neg h, if_neg h]; ring
        rw [Finset.sum_congr rfl pick, Finset.sum_ite_eq', if_pos (Finset.mem_range.2
          (Nat.mod_lt _ hd)), if_pos hji]
      · simp only [if_neg hji, zero_mul, Finset.sum_const_zero]
    rw [Finset.sum_congr rfl inner, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ri)]
  have nu'i : u' ≠ i := fun h => hoR u' (Or.inl rfl) (Or.inl h)
  have npi : p ≠ i := fun h => hoR p (Or.inr (Or.inl rfl)) (Or.inl h)
  have nqi : q ≠ i := fun h => hoR q (Or.inr (Or.inr (Or.inl rfl))) (Or.inl h)
  have npx : p ≠ x := fun h => hoR p (Or.inr (Or.inl rfl)) (Or.inr (Or.inl h))
  have nqy : q ≠ y := fun h => hoR q (Or.inr (Or.inr (Or.inl rfl))) (Or.inr (Or.inr h))
  have npy : p ≠ y := fun h => hoR p (Or.inr (Or.inl rfl)) (Or.inr (Or.inr h))
  have nqx : q ≠ x := fun h => hoR q (Or.inr (Or.inr (Or.inl rfl))) (Or.inr (Or.inl h))
  have nu'x : u' ≠ x := fun h => hoR u' (Or.inl rfl) (Or.inr (Or.inl h))
  have nu'y : u' ≠ y := fun h => hoR u' (Or.inl rfl) (Or.inr (Or.inr h))
  have hR : sumOver (labDim E) (u' :: p :: q :: (ws1 ++ ws2)) (prodVal E [m',
      ⟨.W, [p, q], [u'], dz, 0⟩, ⟨.Z, ws1, [p], dz, z1⟩, ⟨.X, [x], ws1, dz, x1⟩,
      ⟨.Z, ws2, [q], dz, z2⟩, ⟨.X, [y], ws2, dz, x2⟩]) τ =
      zc E m (τ i) * ∑ a ∈ range d, ∑ b ∈ range d, tW d [a, b] (τ i) *
        (((rt d)⁻¹ ^ (1 + e) * (if τ x % d = (e * a) % d then (d : ℂ) else 0)) *
          ((rt d)⁻¹ ^ (1 + e) * (if τ y % d = (e * b) % d then (d : ℂ) else 0))) := by
    have split : prodVal E [m', ⟨.W, [p, q], [u'], dz, 0⟩, ⟨.Z, ws1, [p], dz, z1⟩,
        ⟨.X, [x], ws1, dz, x1⟩, ⟨.Z, ws2, [q], dz, z2⟩, ⟨.X, [y], ws2, dz, x2⟩] =
        fun ρ => (val E ρ m' * val E ρ ⟨.W, [p, q], [u'], dz, 0⟩) *
          (prodVal E [⟨.Z, ws1, [p], dz, z1⟩, ⟨.X, [x], ws1, dz, x1⟩] ρ *
            prodVal E [⟨.Z, ws2, [q], dz, z2⟩, ⟨.X, [y], ws2, dz, x2⟩] ρ) := by
      funext ρ; simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons,
        List.prod_nil]; ring
    have hG : Reads [u', i, p, q] (fun ρ => val E ρ m' * val E ρ ⟨.W, [p, q], [u'], dz, 0⟩) := by
      intro ρ ρ' h
      simp only
      rw [val_zc _ _ m' hm', val_zc _ _ m' hm', hlm']
      simp only [val, List.map_cons, List.map_nil]
      rw [h u' (by simp), h i (by simp), h p (by simp), h q (by simp)]
    simp only [sumOver, hlab u' (by simp), hlab p (by simp), hlab q (by simp)]
    rw [split]
    have hdisj1 : ∀ z ∈ p :: x :: ws1, z ∉ ws2 := by
      intro z hz hz2
      simp only [List.mem_cons] at hz
      rcases hz with rfl | rfl | hz
      · exact hp12 (List.mem_append_right _ hz2)
      · exact hoR _ (Or.inr (Or.inr (Or.inr (Or.inr hz2)))) (Or.inr (Or.inl rfl))
      · exact List.disjoint_of_nodup_append h12 hz hz2
    have hdisj2 : ∀ z ∈ q :: y :: ws2, z ∉ ws1 := by
      intro z hz hz1
      simp only [List.mem_cons] at hz
      rcases hz with rfl | rfl | hz
      · exact hq12 (List.mem_append_left _ hz1)
      · exact hoR _ (Or.inr (Or.inr (Or.inr (Or.inl hz1)))) (Or.inr (Or.inr rfl))
      · exact List.disjoint_of_nodup_append h12 hz1 hz
    have inner : ∀ j ∈ range d, ∀ a ∈ range d, ∀ b ∈ range d, sumOver (labDim E) (ws1 ++ ws2)
        (fun ρ => (val E ρ m' * val E ρ ⟨.W, [p, q], [u'], dz, 0⟩) *
          (prodVal E [⟨.Z, ws1, [p], dz, z1⟩, ⟨.X, [x], ws1, dz, x1⟩] ρ *
            prodVal E [⟨.Z, ws2, [q], dz, z2⟩, ⟨.X, [y], ws2, dz, x2⟩] ρ))
        (Function.update (Function.update (Function.update τ u' j) p a) q b) =
        (if j = τ i then zc E m (τ i) else 0) * (tW d [a, b] j *
          (((rt d)⁻¹ ^ (1 + e) * (if τ x % d = (e * a) % d then (d : ℂ) else 0)) *
            ((rt d)⁻¹ ^ (1 + e) * (if τ y % d = (e * b) % d then (d : ℂ) else 0)))) := by
      intro j hj a ha b hb
      have hj' := Finset.mem_range.1 hj
      have ha' := Finset.mem_range.1 ha
      have hb' := Finset.mem_range.1 hb
      rw [sumOver_const_mul _ (ws1 ++ ws2) [u', i, p, q] _ _ hG (by
        intro z hz hzw
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hz
        rcases hz with rfl | rfl | rfl | rfl
        · exact hu12 hzw
        · exact hoR _ (Or.inr (Or.inr (Or.inr (List.mem_append.1 hzw)))) (Or.inl rfl)
        · exact hp12 hzw
        · exact hq12 hzw)]
      rw [sumOver_mul_split (D := labDim E) ws1 ws2 (p :: x :: ws1) (q :: y :: ws2) _ _
        (multiplier_reads E z1 x1 dz p x ws1) (multiplier_reads E z2 x2 dz q y ws2) hdisj1
        hdisj2]
      have n1 := (List.nodup_append.1 h12).1
      have n2 := (List.nodup_append.1 h12).2.1
      have hpx1 : p ∉ ws1 := fun h => hp12 (List.mem_append_left _ h)
      have hqy2 : q ∉ ws2 := fun h => hq12 (List.mem_append_right _ h)
      have hxw1 : x ∉ ws1 := fun h => hoR x (Or.inr (Or.inr (Or.inr (Or.inl h))))
        (Or.inr (Or.inl rfl))
      have hyw2 : y ∉ ws2 := fun h => hoR y (Or.inr (Or.inr (Or.inr (Or.inr h))))
        (Or.inr (Or.inr rfl))
      rw [mult p x z1 x1 ws1 (Function.update (Function.update (Function.update τ u' j) p a) q b) n1 hpx1 hxw1 (fun w hw => hl w (by simp [hw]))
          (by rw [Function.update_of_ne hpq, Function.update_self]; exact ha') h1
          hz1 hx1,
        mult q y z2 x2 ws2 (Function.update (Function.update (Function.update τ u' j) p a) q b) n2 hqy2 hyw2 (fun w hw => hl w (by simp [hw]))
          (by rw [Function.update_self]; exact hb') h2 hz2 hx2,
        hm_val' (Function.update (Function.update (Function.update τ u' j) p a) q b) (by rw [Function.update_of_ne (Ne.symm nqi), Function.update_of_ne (Ne.symm npi),
            Function.update_of_ne (Ne.symm nu'i)])
          (by rw [Function.update_of_ne huq, Function.update_of_ne hup, Function.update_self]
              exact hj')]
      simp only [val, List.map_cons, List.map_nil, Function.update_self,
        Function.update_of_ne hpq, Function.update_of_ne (Ne.symm hpq), Function.update_of_ne hup,
        Function.update_of_ne huq, Function.update_of_ne (Ne.symm nqx),
        Function.update_of_ne (Ne.symm npx), Function.update_of_ne (Ne.symm nu'x),
        Function.update_of_ne (Ne.symm nqy), Function.update_of_ne (Ne.symm npy),
        Function.update_of_ne (Ne.symm nu'y)]
      ring
    rw [Finset.sum_congr rfl fun j hj => Finset.sum_congr rfl fun a ha =>
      Finset.sum_congr rfl fun b hb => inner j hj a ha b hb]
    have pick : ∀ j ∈ range d, (∑ a ∈ range d, ∑ b ∈ range d,
        (if j = τ i then zc E m (τ i) else 0) * (tW d [a, b] j *
          (((rt d)⁻¹ ^ (1 + e) * (if τ x % d = (e * a) % d then (d : ℂ) else 0)) *
            ((rt d)⁻¹ ^ (1 + e) * (if τ y % d = (e * b) % d then (d : ℂ) else 0))))) =
        if j = τ i then zc E m (τ i) * ∑ a ∈ range d, ∑ b ∈ range d, tW d [a, b] (τ i) *
          (((rt d)⁻¹ ^ (1 + e) * (if τ x % d = (e * a) % d then (d : ℂ) else 0)) *
            ((rt d)⁻¹ ^ (1 + e) * (if τ y % d = (e * b) % d then (d : ℂ) else 0))) else 0 := by
      intro j _
      by_cases h : j = τ i
      · subst h; simp only [if_true, Finset.mul_sum]
      · simp only [if_neg h, zero_mul, Finset.sum_const_zero]
    rw [Finset.sum_congr rfl pick, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ri)]
  have wsum : ∀ I < d, ∀ F : ℕ → ℕ → ℂ, ∑ a ∈ range d, ∑ b ∈ range d, tW d [a, b] I * F a b =
      F I 0 + (if I ≠ 0 then F 0 I else 0) := by
    intro I hI F
    have e1 : ∀ a ∈ range d, ∀ b ∈ range d, tW d [a, b] I * F a b =
        (if a = I ∧ b = 0 then F a b else 0) + (if a = 0 ∧ b = I then
          (if I ≠ 0 then F a b else 0) else 0) := by
      intro a _ b _
      rw [tW_pair d a b I hI]
      by_cases h1 : a = I ∧ b = 0 <;> by_cases h2 : a = 0 ∧ b = I <;> by_cases h3 : I = 0 <;>
        simp [h1, h2, h3]
    rw [Finset.sum_congr rfl fun a ha => Finset.sum_congr rfl fun b hb => e1 a ha b hb]
    simp only [Finset.sum_add_distrib]
    rw [dsum_pick d I 0 hI hd F, dsum_pick d 0 I hd hI (fun a b => if I ≠ 0 then F a b else 0)]
  rw [hL, hR, one_mul, wsum _ ri]
  have hne : e ≠ 0 := fun h => he (by rw [h, Nat.zero_mod])
  have hr := rt_ne d hd
  set A : ℂ := (rt d)⁻¹ ^ (1 + e) * d
  set K : ℂ := rt d ^ (e - 1)
  have hsc : K * (A * A) = A := by
    obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le (Nat.one_le_iff_ne_zero.2 hne)
    have hA' : A = (rt d)⁻¹ ^ n := by
      simp only [A]
      rw [hn, ← rt_sq' d, show 1 + (1 + n) = n + 2 by ring, pow_add, mul_assoc, ← mul_pow,
        inv_mul_cancel₀ hr, one_pow, mul_one]
    have hK : K = rt d ^ n := by simp only [K]; rw [hn, show 1 + n - 1 = n by omega]
    rw [hA', hK, ← mul_assoc, ← mul_pow, mul_inv_cancel₀ hr, one_pow, one_mul]
  have hA : ∀ (P : Prop) [Decidable P], (rt d)⁻¹ ^ (1 + e) * (if P then (d : ℂ) else 0) =
      if P then A else 0 := by
    intro P _; split_ifs <;> simp [A]
  simp only [hA]
  rw [Nat.mod_eq_of_lt rx, Nat.mod_eq_of_lt ry, tW_pair _ _ _ _ (Nat.mod_lt _ hd)]
  set c := zc E m (τ i)
  by_cases i0 : τ i = 0
  · have hc0 : (e * τ i) % d = 0 := by rw [i0, mul_zero, Nat.zero_mod]
    rw [hc0]
    simp only [mul_zero, Nat.zero_mod, i0, ne_eq, not_true_eq_false, ite_false, add_zero]
    by_cases a1 : τ x = 0 <;> by_cases a2 : τ y = 0 <;>
      simp only [a1, a2, and_true, true_and, and_false, false_and, and_self, if_true, if_false,
        add_zero, mul_one, mul_zero, zero_mul] <;>
      first | linear_combination c * hsc | linear_combination (-c) * hsc | ring
  · by_cases i1 : τ i = d - 1
    · have ht : (e * τ i) % d ≠ 0 := by
        rw [i1]
        intro h
        apply he
        have h2 : (e * (d - 1) + e) % d = 0 := by
          rw [show e * (d - 1) + e = e * d by
            rw [← Nat.mul_succ]; congr 1; omega]
          simp
        rw [Nat.add_mod, h, zero_add, Nat.mod_mod] at h2
        exact h2
      generalize (e * τ i) % d = t at ht ⊢
      simp only [mul_zero, Nat.zero_mod, ne_eq, i0, not_false_eq_true, if_true, ht]
      by_cases a1 : τ x = t <;> by_cases a2 : τ y = 0 <;> by_cases a3 : τ x = 0 <;>
        by_cases a4 : τ y = t <;>
        first
        | (exfalso; exact ht (a1.symm.trans a3))
        | (exfalso; exact ht (a4.symm.trans a2))
        | (simp only [a1, a2, a3, a4, ht, Ne.symm ht, and_true, true_and, and_false, false_and,
            and_self, if_true, if_false, add_zero, zero_add, mul_one, mul_zero, zero_mul] <;>
           first | linear_combination c * hsc | linear_combination (-c) * hsc | ring)
    · have : c = 0 := hbox _ ri i0 i1
      rw [this]; simp

/-- A phaseless Z spider with two legs at a value `v` of one leg. -/
theorem tZ_pair (d : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = 1) (a v : ℕ) (ha : a < d) :
    tZ d φ [v, a] = if v = a then 1 else 0 := by
  rw [tZ_at _ _ _ a (by simp)]
  by_cases h : v = a
  · subst h; simp [ha, hφ _ ha]
  · have : ¬ (a < d ∧ ∀ w ∈ [v, a], w = a) := fun h' => h (h'.2 v (by simp))
    rw [if_neg this, if_neg h]

/-- [P2] (2.8): the binder is the splitter with its three legs bent by phaseless Z spiders. -/
theorem bindDef (E : Env) (ds dt dk a b c x y z p₁ p₂ p₃ : ℕ) (iface : List ℕ)
    (hdk : E.dimv dk = E.dimv ds * E.dimv dt)
    (hnd : [x, y, z, a, b, c].Nodup) (hx : labIdx E.labs x = ds) (hy : labIdx E.labs y = dt)
    (hz : labIdx E.labs z = dk) (ha : a ∈ iface ∧ labIdx E.labs a = ds)
    (hb : b ∈ iface ∧ labIdx E.labs b = dt) (hc : c ∈ iface ∧ labIdx E.labs c = dk)
    (h₁ : ∀ k < E.dimv ds, ex (E.phase p₁ k) = 1) (h₂ : ∀ k < E.dimv dt, ex (E.phase p₂ k) = 1)
    (h₃ : ∀ k < E.dimv dk, ex (E.phase p₃ k) = 1) :
    LocalEq E [⟨.B, [c], [a, b], ds, dt⟩] []
      [⟨.S, [x, y], [z], ds, dt⟩, ⟨.Z, [x], [a], ds, p₁⟩, ⟨.Z, [y], [b], dt, p₂⟩,
        ⟨.Z, [c], [z], dk, p₃⟩] [x, y, z] iface 1 1 := by
  intro τ hτ
  have ra := hτ a ha.1
  have rb := hτ b hb.1
  have rc := hτ c hc.1
  rw [labDim_of E a ds ha.2] at ra
  rw [labDim_of E b dt hb.2] at rb
  rw [labDim_of E c dk hc.2] at rc
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd
  obtain ⟨⟨hxy, hxz, hxa, hxb, hxc⟩, ⟨hyz, hya, hyb, hyc⟩, ⟨hza, hzb, hzc⟩, -, -, -⟩ := hnd
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val, labDim_of E x ds hx, labDim_of E y dt hy, labDim_of E z dk hz,
    Function.update_apply, hxy, hxz, hyz, Ne.symm hxy, Ne.symm hxz, Ne.symm hyz, hxa, hxb, hxc,
    hya, hyb, hyc, hza, hzb, hzc, Ne.symm hxa, Ne.symm hxb, Ne.symm hxc, Ne.symm hya,
    Ne.symm hyb, Ne.symm hyc, Ne.symm hza, Ne.symm hzb, Ne.symm hzc, if_true, if_false,
    List.cons_append, List.nil_append]
  simp only [tZ_pair _ _ h₁ _ _ ra, tZ_pair _ _ h₂ _ _ rb]
  have hzc' : ∀ k ∈ range (E.dimv dk), tZ (E.dimv dk) (E.phase p₃) [τ c, k] =
      if k = τ c then 1 else 0 := by
    intro k hk
    rw [tZ_at _ _ _ k (by simp)]
    by_cases h : k = τ c
    · subst h; simp [Finset.mem_range.1 hk, h₃ _ (Finset.mem_range.1 hk)]
    · have : ¬ (k < E.dimv dk ∧ ∀ w ∈ [τ c, k], w = k) := fun h' => h (h'.2 _ (by simp)).symm
      rw [if_neg this, if_neg h]
  rw [Finset.sum_congr rfl fun xv _ => Finset.sum_congr rfl fun yv _ =>
    Finset.sum_congr rfl fun zv hzv => by rw [hzc' zv hzv]]
  have inner : ∀ xv ∈ range (E.dimv ds), ∀ yv ∈ range (E.dimv dt),
      ∑ zv ∈ range (E.dimv dk), tB (E.dimv ds) (E.dimv dt) zv xv yv *
        ((if xv = τ a then 1 else 0) * ((if yv = τ b then 1 else 0) *
          (if zv = τ c then 1 else 0))) =
      if xv = τ a ∧ yv = τ b then tB (E.dimv ds) (E.dimv dt) (τ c) xv yv else 0 := by
    intro xv _ yv _
    have : ∀ zv ∈ range (E.dimv dk), tB (E.dimv ds) (E.dimv dt) zv xv yv *
        ((if xv = τ a then 1 else 0) * ((if yv = τ b then 1 else 0) *
          (if zv = τ c then 1 else 0))) =
        if zv = τ c then (if xv = τ a ∧ yv = τ b then tB (E.dimv ds) (E.dimv dt) (τ c) xv yv
          else 0) else 0 := by
      intro zv _
      by_cases h1 : zv = τ c <;> by_cases h2 : xv = τ a <;> by_cases h3 : yv = τ b <;>
        simp [h1, h2, h3]
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 rc)]
  rw [Finset.sum_congr rfl fun xv hx => Finset.sum_congr rfl fun yv hy => inner xv hx yv hy]
  simp only [ite_and]
  rw [Finset.sum_congr rfl fun xv _ => by rw [Finset.sum_ite_irrel]]
  simp only [Finset.sum_const_zero]
  rw [Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ra), Finset.sum_ite_eq',
    if_pos (Finset.mem_range.2 rb)]

/-- [P2] (2.8): the binder is the splitter read the other way. -/
theorem bindDef0 (E : Env) (ds dt a b c : ℕ) (iface : List ℕ) :
    LocalEq E [⟨.B, [c], [a, b], ds, dt⟩] [] [⟨.S, [a, b], [c], ds, dt⟩] [] iface 1 1 := by
  intro τ _
  simp [sumOver, prodVal, val]

/-- A two-leg spider at a value of its second leg. -/
theorem val_zc_two (E : Env) (σ : ℕ → ℕ) (n : Node) (hn : isZc n = true) (p q : ℕ)
    (hl : n.outs ++ n.ins = [p, q]) (hq : σ q < E.dimv n.dim) :
    val E σ n = if σ p = σ q then zc E n (σ q) else 0 := by
  rw [val_zc _ _ n hn, hl]
  simp only [List.map_cons, List.map_nil]
  rw [tZw_at _ _ _ (σ q) (by simp)]
  by_cases h : σ p = σ q
  · rw [if_pos ⟨hq, by simp [h]⟩, if_pos h]
  · have : ¬ (σ q < E.dimv n.dim ∧ ∀ w ∈ [σ p, σ q], w = σ q) := fun h' => h (h'.2 _ (by simp))
    rw [if_neg this, if_neg h]

/-- [P1] (D1): the spider `m` followed by `F ; F` is `F ; F` followed by the spider `m'` of
reversed coefficients. -/
theorem zxwD1 (E : Env) (dz i o w v v' w' : ℕ) (m m' : Node) (iface : List ℕ)
    (hd : 0 < E.dimv dz) (hm : isZc m = true) (hm' : isZc m' = true) (hdm : m.dim = dz)
    (hdm' : m'.dim = dz) (hl : m.outs ++ m.ins = [w, i]) (hl' : m'.outs ++ m'.ins = [o, w'])
    (hnd : [w, v, v', w', i, o].Nodup) (hw : ∀ x ∈ [w, v, v', w'], labIdx E.labs x = dz)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz) (ho : o ∈ iface ∧ labIdx E.labs o = dz)
    (hc : ∀ k < E.dimv dz, zc E m' k = zc E m ((E.dimv dz - k) % E.dimv dz)) :
    LocalEq E [m, ⟨.F, [v], [w], dz, 0⟩, ⟨.F, [o], [v], dz, 0⟩] [w, v]
      [⟨.F, [v'], [i], dz, 0⟩, ⟨.F, [w'], [v'], dz, 0⟩, m'] [v', w'] iface 1 1 := by
  intro τ hτ
  have ri := hτ i hi.1
  have ro := hτ o ho.1
  rw [labDim_of E i dz hi.2] at ri
  rw [labDim_of E o dz ho.2] at ro
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd
  obtain ⟨⟨hwv, hwv', hww', hwi, hwo⟩, ⟨hvv', hvw', hvi, hvo⟩, ⟨hv'w', hv'i, hv'o⟩,
    ⟨hw'i, hw'o⟩, hio, -⟩ := hnd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hw
  obtain ⟨lw, lv, lv', lw'⟩ := hw
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, labDim_of E w dz lw, labDim_of E v dz lv, labDim_of E v' dz lv',
    labDim_of E w' dz lw']
  have hL : ∀ a ∈ range (E.dimv dz), ∑ b ∈ range (E.dimv dz),
      val E (Function.update (Function.update τ w a) v b) m *
        (val E (Function.update (Function.update τ w a) v b) ⟨.F, [v], [w], dz, 0⟩ *
          val E (Function.update (Function.update τ w a) v b) ⟨.F, [o], [v], dz, 0⟩) =
      if a = τ i then zc E m (τ i) * (if (τ o + τ i) % E.dimv dz = 0 then 1 else 0) else 0 := by
    intro a ha
    have ha' := Finset.mem_range.1 ha
    have e1 : ∀ b, val E (Function.update (Function.update τ w a) v b) m =
        if a = τ i then zc E m (τ i) else 0 := by
      intro b
      rw [val_zc_two E _ m hm w i hl (by
        rw [Function.update_of_ne (Ne.symm hvi), Function.update_of_ne (Ne.symm hwi), hdm]
        exact ri)]
      simp only [Function.update_of_ne hwv, Function.update_self,
        Function.update_of_ne (Ne.symm hvi), Function.update_of_ne (Ne.symm hwi)]
    simp only [e1]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne hwv, Function.update_of_ne (Ne.symm hvo),
      Function.update_of_ne (Ne.symm hwo)]
    by_cases h : a = τ i
    · subst h
      rw [if_pos rfl, if_pos rfl, ← Finset.mul_sum]
      congr 1
      rw [Finset.sum_congr rfl fun b _ => mul_comm (fou (E.dimv dz) b (τ i)) _, fou_sq _ hd]
    · simp [h]
  rw [Finset.sum_congr rfl hL, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ri)]
  have hR : ∀ a ∈ range (E.dimv dz), ∑ b ∈ range (E.dimv dz),
      val E (Function.update (Function.update τ v' a) w' b) ⟨.F, [v'], [i], dz, 0⟩ *
        (val E (Function.update (Function.update τ v' a) w' b) ⟨.F, [w'], [v'], dz, 0⟩ *
          val E (Function.update (Function.update τ v' a) w' b) m') =
      fou (E.dimv dz) a (τ i) * (fou (E.dimv dz) (τ o) a * zc E m' (τ o)) := by
    intro a _
    have e1 : ∀ b ∈ range (E.dimv dz), val E (Function.update (Function.update τ v' a) w' b) m' =
        if b = τ o then zc E m' (τ o) else 0 := by
      intro b hb
      rw [val_zc_two E _ m' hm' o w' hl' (by
        rw [Function.update_self, hdm']; exact Finset.mem_range.1 hb)]
      simp only [Function.update_self, Function.update_of_ne (Ne.symm hw'o),
        Function.update_of_ne (Ne.symm hv'o)]
      by_cases h : b = τ o
      · subst h; simp
      · rw [if_neg (Ne.symm h), if_neg h]
    rw [Finset.sum_congr rfl fun b hb => by rw [e1 b hb]]
    simp only [val, List.map_cons, List.map_nil, Function.update_self,
      Function.update_of_ne hv'w', Function.update_of_ne (Ne.symm hv'w'),
      Function.update_of_ne (Ne.symm hv'i), Function.update_of_ne (Ne.symm hw'i)]
    rw [Finset.sum_congr rfl fun b _ => by rw [mul_ite, mul_zero, mul_ite, mul_zero]]
    rw [Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ro)]
  rw [Finset.sum_congr rfl hR]
  rw [show (∑ a ∈ range (E.dimv dz), fou (E.dimv dz) a (τ i) *
      (fou (E.dimv dz) (τ o) a * zc E m' (τ o))) =
      (∑ a ∈ range (E.dimv dz), fou (E.dimv dz) (τ o) a * fou (E.dimv dz) a (τ i)) *
        zc E m' (τ o) by rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun a _ => by ring,
    fou_sq _ hd, hc _ ro]
  by_cases h : (τ o + τ i) % E.dimv dz = 0
  · have : (E.dimv dz - τ o) % E.dimv dz = τ i := by
      have := (antipode_iff (E.dimv dz) (τ o) (τ i) ro ri).1 (by rw [add_comm]; exact h)
      exact this.symm
    rw [if_pos h, this]; ring
  · rw [if_neg h]; ring


/-- The labels of a chain `F ; F`: its start, inner wire, end. -/
def chain2Labs (c : ℕ × ℕ × ℕ) : List ℕ := [c.1, c.2.1, c.2.2]

/-- The entry of a chain `F ; F` from `c.1` through `c.2.1` to `c.2.2`. -/
noncomputable def chain2Val (d : ℕ) (τ : ℕ → ℕ) (c : ℕ × ℕ × ℕ) : ℂ :=
  fou d (τ c.2.1) (τ c.1) * fou d (τ c.2.2) (τ c.2.1)

/-- The chain `F ; F` as nodes. -/
def chain2Nodes (dz : ℕ) (c : ℕ × ℕ × ℕ) : List Node :=
  [⟨.F, [c.2.1], [c.1], dz, 0⟩, ⟨.F, [c.2.2], [c.2.1], dz, 0⟩]

theorem prod_ind (l : List ℕ) (P : ℕ → Prop) [DecidablePred P] :
    (l.map fun x => if P x then (1 : ℂ) else 0).prod = if ∀ x ∈ l, P x then 1 else 0 := by
  induction l with
  | nil => simp
  | cons x l ih =>
    simp only [List.map_cons, List.prod_cons, ih, List.mem_cons, forall_eq_or_imp]
    by_cases h : P x <;> by_cases h' : ∀ y ∈ l, P y <;> simp [h, h']

/-- Summing the inner wires of chains `F ; F`. -/
theorem sumOver_chains2 {D : ℕ → ℕ} (d : ℕ) (hd : 0 < d) (cs : List (ℕ × ℕ × ℕ))
    (hD : ∀ c ∈ cs, D c.2.1 = d) (hnd : (cs.map fun c => c.2.1).Nodup)
    (hio : ∀ c ∈ cs, ∀ c' ∈ cs, c.1 ≠ c'.2.1 ∧ c.2.2 ≠ c'.2.1) (ρ : ℕ → ℕ) :
    sumOver D (cs.map fun c => c.2.1) (fun τ => (cs.map (chain2Val d τ)).prod) ρ =
      (cs.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then (1 : ℂ) else 0).prod := by
  induction cs generalizing ρ with
  | nil => simp [sumOver]
  | cons c rest ih =>
    simp only [List.map_cons, List.prod_cons, sumOver]
    obtain ⟨ha, hrest⟩ := List.nodup_cons.1 hnd
    have hc := hio c List.mem_cons_self c List.mem_cons_self
    rw [hD c List.mem_cons_self]
    have inner : ∀ j ∈ range d, sumOver D (rest.map fun c => c.2.1)
        (fun τ => chain2Val d τ c * (rest.map (chain2Val d τ)).prod) (Function.update ρ c.2.1 j) =
        chain2Val d (Function.update ρ c.2.1 j) c *
          (rest.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then (1 : ℂ) else 0).prod := by
      intro j _
      rw [sumOver_const_mul D _ (chain2Labs c) (fun τ => chain2Val d τ c) _ (fun τ τ' h => by
          simp only [chain2Val]
          rw [h c.1 (by simp [chain2Labs]), h c.2.1 (by simp [chain2Labs]),
            h c.2.2 (by simp [chain2Labs])])
        (fun x hx hxr => by
          obtain ⟨c', hc', rfl⟩ := List.mem_map.1 hxr
          simp only [chain2Labs, List.mem_cons, List.not_mem_nil, or_false] at hx
          rcases hx with h | h | h
          · exact (hio c List.mem_cons_self c' (List.mem_cons_of_mem _ hc')).1 h.symm
          · exact ha (List.mem_map.2 ⟨c', hc', h⟩)
          · exact (hio c List.mem_cons_self c' (List.mem_cons_of_mem _ hc')).2 h.symm)]
      rw [ih (fun c' hc' => hD c' (List.mem_cons_of_mem _ hc')) hrest
        (fun c₁ h₁ c₂ h₂ => hio c₁ (List.mem_cons_of_mem _ h₁) c₂ (List.mem_cons_of_mem _ h₂))
        (Function.update ρ c.2.1 j)]
      congr 1
      refine congrArg List.prod (List.map_congr_left fun c' hc' => ?_)
      have h1 := (hio c' (List.mem_cons_of_mem _ hc') c List.mem_cons_self)
      rw [Function.update_of_ne h1.1, Function.update_of_ne h1.2]
    rw [Finset.sum_congr rfl inner, ← Finset.sum_mul]
    congr 1
    simp only [chain2Val, Function.update_self, Function.update_of_ne hc.1,
      Function.update_of_ne hc.2]
    rw [Finset.sum_congr rfl fun j _ => mul_comm (fou d j (ρ c.1)) _, fou_sq d hd]

/-- A spider whose legs are `B` and the pinned ends of chains `F ; F`, summed over the pinned
ends and inner wires, as a sum over the spider's index. -/
theorem spiderF2 (E : Env) (dz : ℕ) (m : Node) (B : List ℕ) (cs : List (ℕ × ℕ × ℕ))
    (pin free : ℕ × ℕ × ℕ → ℕ) (τ : ℕ → ℕ) (hd : 0 < E.dimv dz) (hm : isZc m = true)
    (hdm : m.dim = dz) (hl : (m.outs ++ m.ins).Perm (B ++ cs.map pin))
    (hpf : ∀ c ∈ cs, (pin c = c.1 ∧ free c = c.2.2) ∨ (pin c = c.2.2 ∧ free c = c.1))
    (hnd : (cs.map pin ++ cs.map fun c => c.2.1).Nodup)
    (hB : ∀ x ∈ B ++ cs.map free, x ∉ cs.map pin ++ cs.map fun c => c.2.1)
    (hio : ∀ c ∈ cs, ∀ c' ∈ cs, c.1 ≠ c'.2.1 ∧ c.2.2 ≠ c'.2.1)
    (hlab : ∀ x ∈ cs.map pin ++ cs.map fun c => c.2.1, labIdx E.labs x = dz) :
    sumOver (labDim E) (cs.map pin ++ cs.map fun c => c.2.1)
        (prodVal E (m :: cs.flatMap (chain2Nodes dz))) τ =
      ∑ k ∈ range (E.dimv dz), (if ∀ b ∈ B, τ b = k then zc E m k else 0) *
        (cs.map fun c => if (k + τ (free c)) % E.dimv dz = 0 then (1 : ℂ) else 0).prod := by
  set d := E.dimv dz
  set zs := cs.map pin
  set as := cs.map fun c => c.2.1
  have hR : ∀ ρ : ℕ → ℕ, prodVal E (m :: cs.flatMap (chain2Nodes dz)) ρ =
      ∑ k ∈ range d, (if ∀ v ∈ B ++ zs, ρ v = k then zc E m k else 0) *
        (cs.map (chain2Val d ρ)).prod := by
    intro ρ
    simp only [prodVal, List.map_cons, List.prod_cons, prod_flatMap']
    rw [val_zc _ _ m hm, hdm, tZw_perm _ _ _ _ (hl.map ρ)]
    unfold tZw
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => ?_
    have e : (∀ v ∈ (B ++ zs).map ρ, v = k) ↔ ∀ v ∈ B ++ zs, ρ v = k :=
      ⟨fun h v hv => h _ (List.mem_map_of_mem hv), fun h v hv => by
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv; exact h x hx⟩
    simp only [e]
    congr 1
    refine congrArg List.prod (List.map_congr_left fun c _ => ?_)
    simp only [chain2Nodes, chain2Val, val, List.map_cons, List.map_nil, List.prod_cons,
      List.prod_nil, mul_one]
    rfl
  have hnd' := hnd
  rw [List.nodup_append] at hnd'
  obtain ⟨hzs, has, hza⟩ := hnd'
  rw [show prodVal E (m :: cs.flatMap (chain2Nodes dz)) = _ from funext hR, sumOver_fsum]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.1 hk
  rw [sumOver_append]
  set G : (ℕ → ℕ) → ℂ := fun ρ => if ∀ v ∈ B ++ zs, ρ v = k then zc E m k else 0
  have hG : Reads (B ++ zs) G := by
    intro ρ ρ' h
    simp only [G]
    have e1 : (∀ v ∈ B ++ zs, ρ v = k) ↔ ∀ v ∈ B ++ zs, ρ' v = k :=
      ⟨fun h' v hv => h v hv ▸ h' v hv, fun h' v hv => (h v hv).symm ▸ h' v hv⟩
    simp only [e1]
  have hcs : ∀ ρ : ℕ → ℕ, sumOver (labDim E) as (fun τ => (cs.map (chain2Val d τ)).prod) ρ =
      (cs.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then (1 : ℂ) else 0).prod := fun ρ =>
    sumOver_chains2 (D := labDim E) d hd cs (fun c hc => labDim_of E _ dz (hlab _
      (List.mem_append_right _ (List.mem_map_of_mem hc)))) has hio ρ
  rw [show sumOver (labDim E) as (fun ρ => G ρ * (cs.map (chain2Val d ρ)).prod) =
      fun ρ => G ρ * (cs.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then (1 : ℂ) else 0).prod
    from funext fun ρ => by
      rw [sumOver_const_mul _ as (B ++ zs) G _ hG (fun x hx hxa => by
        rcases List.mem_append.1 hx with hx | hx
        · exact hB x (List.mem_append_left _ hx) (List.mem_append_right _ hxa)
        · exact hza x hx x hxa rfl) ρ, hcs]]
  have hsplit : ∀ ρ : ℕ → ℕ, G ρ * (cs.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then
      (1 : ℂ) else 0).prod = if ∀ v ∈ zs, ρ v = k then (if ∀ b ∈ B, ρ b = k then zc E m k
        else 0) * (cs.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then (1 : ℂ) else 0).prod
      else 0 := by
    intro ρ
    simp only [G]
    by_cases h1 : ∀ v ∈ zs, ρ v = k <;> by_cases h2 : ∀ b ∈ B, ρ b = k
    · rw [if_pos (fun v hv => (List.mem_append.1 hv).elim (h2 v) (h1 v)), if_pos h1, if_pos h2]
    · rw [if_neg (fun h => h2 fun b hb => h b (List.mem_append_left _ hb)), if_pos h1, if_neg h2]
    · rw [if_neg (fun h => h1 fun v hv => h v (List.mem_append_right _ hv)), if_neg h1]; ring
    · rw [if_neg (fun h => h1 fun v hv => h v (List.mem_append_right _ hv)), if_neg h1]; ring
  rw [show (fun ρ => G ρ * (cs.map fun c => if (ρ c.2.2 + ρ c.1) % d = 0 then (1 : ℂ)
      else 0).prod) = _ from funext hsplit]
  rw [sumOver_pinK (D := labDim E) zs hzs k (fun x hx => by
    rw [labDim_of E x dz (hlab x (List.mem_append_left _ hx))]; exact hk')]
  have hρB : ∀ x ∈ B ++ cs.map free, (if x ∈ zs then k else τ x) = τ x := fun x hx => by
    rw [if_neg fun h => hB x hx (List.mem_append_left _ h)]
  congr 1
  · simp only [List.mem_append] at hρB
    have e : (∀ b ∈ B, (if b ∈ zs then k else τ b) = k) ↔ ∀ b ∈ B, τ b = k :=
      ⟨fun h b hb => (hρB b (Or.inl hb)) ▸ h b hb, fun h b hb => (hρB b (Or.inl hb)).symm ▸ h b hb⟩
    simp only [e]
  · refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    have hpin : pin c ∈ zs := List.mem_map_of_mem hc
    have hfree : free c ∈ B ++ cs.map free := List.mem_append_right _ (List.mem_map_of_mem hc)
    dsimp only
    rcases hpf c hc with ⟨hp, hf⟩ | ⟨hp, hf⟩
    · rw [← hp, ← hf]
      simp only [if_pos hpin, hρB _ hfree, Nat.add_comm (τ (free c)) k]
    · rw [← hp, ← hf]
      simp only [if_pos hpin, hρB _ hfree]

/-- [P2] (D1): `F ; F` on every input of a spider `m` is `F ; F` on every output of the spider
`m'` of reversed coefficients. -/
theorem qfD1 (E : Env) (dz : ℕ) (m m' : Node) (os is : List ℕ) (cl cr : List (ℕ × ℕ × ℕ))
    (iface : List ℕ) (hd : 0 < E.dimv dz) (hm : isZc m = true) (hm' : isZc m' = true)
    (hdm : m.dim = dz) (hdm' : m'.dim = dz)
    (hl : (m.outs ++ m.ins).Perm (os ++ cl.map fun c => c.2.2)) (hli : cl.map Prod.fst = is)
    (hl' : (m'.outs ++ m'.ins).Perm (is ++ cr.map Prod.fst)) (hro : cr.map (fun c => c.2.2) = os)
    (hndL : ((cl.map fun c => c.2.2) ++ cl.map fun c => c.2.1).Nodup)
    (hBL : ∀ x ∈ os ++ cl.map Prod.fst, x ∉ (cl.map fun c => c.2.2) ++ cl.map fun c => c.2.1)
    (hioL : ∀ c ∈ cl, ∀ c' ∈ cl, c.1 ≠ c'.2.1 ∧ c.2.2 ≠ c'.2.1)
    (hlabL : ∀ x ∈ (cl.map fun c => c.2.2) ++ cl.map fun c => c.2.1, labIdx E.labs x = dz)
    (hndR : (cr.map Prod.fst ++ cr.map fun c => c.2.1).Nodup)
    (hBR : ∀ x ∈ is ++ cr.map (fun c => c.2.2), x ∉ cr.map Prod.fst ++ cr.map fun c => c.2.1)
    (hioR : ∀ c ∈ cr, ∀ c' ∈ cr, c.1 ≠ c'.2.1 ∧ c.2.2 ≠ c'.2.1)
    (hlabR : ∀ x ∈ cr.map Prod.fst ++ cr.map fun c => c.2.1, labIdx E.labs x = dz)
    (hface : ∀ x ∈ os ++ is, x ∈ iface ∧ labIdx E.labs x = dz)
    (hc : ∀ k < E.dimv dz, zc E m' k = zc E m ((E.dimv dz - k) % E.dimv dz)) :
    LocalEq E (m :: cl.flatMap (chain2Nodes dz))
      ((cl.map fun c => c.2.2) ++ cl.map fun c => c.2.1)
      (m' :: cr.flatMap (chain2Nodes dz)) (cr.map Prod.fst ++ cr.map fun c => c.2.1)
      iface 1 1 := by
  intro τ hτ
  have hr : ∀ x ∈ os ++ is, τ x < E.dimv dz := fun x hx => by
    have := hτ x (hface x hx).1; rwa [labDim_of E x dz (hface x hx).2] at this
  rw [one_mul, one_mul,
    spiderF2 E dz m os cl (fun c => c.2.2) Prod.fst τ hd hm hdm hl
      (fun c _ => Or.inr ⟨rfl, rfl⟩) hndL hBL hioL hlabL,
    spiderF2 E dz m' is cr Prod.fst (fun c => c.2.2) τ hd hm' hdm' hl'
      (fun c _ => Or.inl ⟨rfl, rfl⟩) hndR hBR hioR hlabR]
  rw [← Finset.sum_nbij' (fun k => (E.dimv dz - k) % E.dimv dz)
    (fun k => (E.dimv dz - k) % E.dimv dz)
    (fun k _ => Finset.mem_range.2 (Nat.mod_lt _ hd))
    (fun k _ => Finset.mem_range.2 (Nat.mod_lt _ hd))
    (fun k hk => neg_neg_mod _ _ hd (Finset.mem_range.1 hk))
    (fun k hk => neg_neg_mod _ _ hd (Finset.mem_range.1 hk)) (fun _ _ => rfl)]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.1 hk
  have hk2 : (E.dimv dz - k) % E.dimv dz < E.dimv dz := Nat.mod_lt _ hd
  rw [hc _ hk']
  rw [show (cl.map fun c => if ((E.dimv dz - k) % E.dimv dz + τ c.1) % E.dimv dz = 0 then
      (1 : ℂ) else 0) = (is.map fun i => if τ i = k then (1 : ℂ) else 0) by
    rw [← hli, List.map_map]
    refine List.map_congr_left fun c hc' => ?_
    have hi : τ c.1 < E.dimv dz := hr _ (List.mem_append_right _ (hli ▸ List.mem_map_of_mem hc'))
    simp only [Function.comp]
    simp only [show ((E.dimv dz - k) % E.dimv dz + τ c.1) % E.dimv dz = 0 ↔ τ c.1 = k by
      rw [add_comm, antipode_iff _ _ _ hk2 hi, neg_neg_mod _ _ hd hk']]]
  rw [show (cr.map fun c => if (k + τ c.2.2) % E.dimv dz = 0 then (1 : ℂ) else 0) =
      (os.map fun o => if τ o = (E.dimv dz - k) % E.dimv dz then (1 : ℂ) else 0) by
    rw [← hro, List.map_map]
    refine List.map_congr_left fun c hc' => ?_
    have ho : τ c.2.2 < E.dimv dz := hr _ (List.mem_append_left _ (hro ▸ List.mem_map_of_mem hc'))
    simp only [Function.comp]
    simp only [show (k + τ c.2.2) % E.dimv dz = 0 ↔ τ c.2.2 = (E.dimv dz - k) % E.dimv dz by
      rw [add_comm]; exact antipode_iff _ _ _ hk' ho]]
  rw [prod_ind, prod_ind]
  by_cases h1 : ∀ o ∈ os, τ o = (E.dimv dz - k) % E.dimv dz <;> by_cases h2 : ∀ i ∈ is, τ i = k
  · simp only [eq_true h1, eq_true h2, if_true, mul_one]
  · simp only [eq_true h1, eq_false h2, if_true, if_false, mul_zero, zero_mul]
  · simp only [eq_false h1, eq_true h2, if_true, if_false, mul_zero, zero_mul, mul_one]
  · simp only [eq_false h1, eq_false h2, if_false, mul_zero, zero_mul]


/-- Summing labels each pinned to a value set by the summed-over function: each takes its value. -/
theorem sumOver_pinV {D : ℕ → ℕ} (ws : List ℕ) (hnd : ws.Nodup) (v : ℕ → ℕ)
    (hv : ∀ w ∈ ws, v w < D w) (g : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D ws (fun τ => if ∀ l ∈ ws, τ l = v l then g τ else 0) σ =
      g (fun x => if x ∈ ws then v x else σ x) := by
  induction ws generalizing σ with
  | nil => simp [sumOver]
  | cons w ws ih =>
    obtain ⟨hw, hnd'⟩ := List.nodup_cons.1 hnd
    simp only [sumOver]
    have inner : ∀ j ∈ range (D w), sumOver D ws (fun τ => if ∀ l ∈ w :: ws, τ l = v l then g τ
        else 0) (Function.update σ w j) =
        if j = v w then sumOver D ws (fun τ => if ∀ l ∈ ws, τ l = v l then g τ else 0)
          (Function.update σ w j) else 0 := by
      intro j _
      split_ifs with hj
      · refine sumOver_congr _ _ _ _ _ fun τ hτ _ => ?_
        have h1 : τ w = j := by rw [hτ w hw, Function.update_self]
        simp only [List.mem_cons, forall_eq_or_imp, h1, hj, true_and]
      · refine (sumOver_congr D ws _ (fun _ => 0) _ fun τ hτ _ => ?_).trans
          (sumOver_zero D ws _)
        have h1 : τ w = j := by rw [hτ w hw, Function.update_self]
        have : ¬ ∀ l ∈ w :: ws, τ l = v l := fun h => hj (h1 ▸ h w List.mem_cons_self)
        rw [if_neg this]
    rw [Finset.sum_congr rfl inner, Finset.sum_ite_eq',
      if_pos (Finset.mem_range.2 (hv w List.mem_cons_self)),
      ih hnd' (fun x hx => hv x (List.mem_cons_of_mem _ hx))]
    congr 1
    funext x
    by_cases hx : x = w
    · subst hx; simp [hw]
    · by_cases hx' : x ∈ ws
      · simp [hx, hx']
      · simp [hx, hx', Function.update_of_ne hx]

/-- A sum over disjoint groups of labels of a product of functions, each reading its own group
and labels outside every group, is the product of the sums. -/
theorem sumOver_groups {D : ℕ → ℕ} (gs : List (List ℕ × ((ℕ → ℕ) → ℂ))) (S : List ℕ)
    (hnd : (gs.flatMap Prod.fst).Nodup) (hS : ∀ x ∈ S, x ∉ gs.flatMap Prod.fst)
    (hf : ∀ g ∈ gs, Reads (g.1 ++ S) g.2) (σ : ℕ → ℕ) :
    sumOver D (gs.flatMap Prod.fst) (fun τ => (gs.map fun g => g.2 τ).prod) σ =
      (gs.map fun g => sumOver D g.1 g.2 σ).prod := by
  induction gs with
  | nil => simp [sumOver]
  | cons g rest ih =>
    simp only [List.flatMap_cons, List.map_cons, List.prod_cons]
    have hnd' := List.nodup_append.1 (by simpa using hnd)
    obtain ⟨hg, hr, hgr⟩ := hnd'
    have hS' : ∀ x ∈ S, x ∉ rest.flatMap Prod.fst := fun x hx h =>
      hS x hx (by simp only [List.flatMap_cons]; exact List.mem_append_right _ h)
    have hSg : ∀ x ∈ S, x ∉ g.1 := fun x hx h =>
      hS x hx (by simp only [List.flatMap_cons]; exact List.mem_append_left _ h)
    rw [sumOver_mul_split (D := D) g.1 (rest.flatMap Prod.fst) (g.1 ++ S)
      (rest.flatMap Prod.fst ++ S) g.2 (fun τ => (rest.map fun g => g.2 τ).prod)
      (hf g List.mem_cons_self)
      (fun τ τ' h => by
        refine congrArg List.prod (List.map_congr_left fun g' hg' => ?_)
        exact hf g' (List.mem_cons_of_mem _ hg') τ τ' fun x hx => by
          rcases List.mem_append.1 hx with hx | hx
          · exact h x (List.mem_append_left _ (List.mem_flatMap.2 ⟨g', hg', hx⟩))
          · exact h x (List.mem_append_right _ hx))
      (fun x hx hxr => by
        rcases List.mem_append.1 hx with hx | hx
        · exact hgr x hx x hxr rfl
        · exact hS' x hx hxr)
      (fun x hx hxa => by
        rcases List.mem_append.1 hx with hx | hx
        · exact hgr x hxa x hx rfl
        · exact hSg x hx hxa),
      ih hr hS' (fun g' hg' => hf g' (List.mem_cons_of_mem _ hg'))]

theorem tW_cons' (d j u : ℕ) (os : List ℕ) (hu : u < d) :
    tW d (j :: os) u = if j = 0 then tW d os u else if j = u then tW d os 0 else 0 := by
  unfold tW
  by_cases hj : j = 0
  · subst hj; by_cases hu0 : u = 0 <;> simp [hu0, List.filter_cons]
  · by_cases hu0 : u = 0
    · subst hu0; simp [hj]
    · by_cases hju : j = u
      · subst hju
        simp only [hj, if_false, hu, true_and, List.filter_cons, ne_eq, not_false_eq_true,
          decide_true, if_true, List.cons.injEq, true_and, if_true]
        simp [List.filter_eq_nil_iff]
      · simp only [hj, hu0, hju, if_false, List.filter_cons, ne_eq, not_false_eq_true,
          decide_true, if_true, List.cons.injEq, false_and, and_false]

/-- Summing the outputs `ws` of a W node at input `u` against `G`: all zero when `u = 0`, else
one output at `u`. -/
theorem sumOver_W {D : ℕ → ℕ} (d : ℕ) (ws : List ℕ) (hnd : ws.Nodup) (hD : ∀ w ∈ ws, D w = d)
    (u : ℕ) (hu : u < d) (hd : 0 < d) (G : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D ws (fun τ => tW d (ws.map τ) u * G τ) σ =
      if u = 0 then G (fun x => if x ∈ ws then 0 else σ x)
      else (ws.map fun w => G (Function.update (fun x => if x ∈ ws then 0 else σ x) w u)).sum := by
  induction ws generalizing σ u with
  | nil => by_cases h : u = 0 <;> simp [sumOver, tW, h]
  | cons w ws ih =>
    obtain ⟨hw, hnd'⟩ := List.nodup_cons.1 hnd
    have hDw := hD w List.mem_cons_self
    have hD' : ∀ x ∈ ws, D x = d := fun x hx => hD x (List.mem_cons_of_mem _ hx)
    simp only [sumOver, hDw, List.map_cons]
    have hmap : ∀ j, ∀ τ : ℕ → ℕ, (∀ x ∉ ws, τ x = Function.update σ w j x) →
        ws.map τ = ws.map τ := fun _ _ _ => rfl
    have step : ∀ j ∈ range d, sumOver D ws (fun τ => tW d (τ w :: ws.map τ) u * G τ)
        (Function.update σ w j) =
        if j = 0 then sumOver D ws (fun τ => tW d (ws.map τ) u * G τ) (Function.update σ w 0)
        else if j = u then sumOver D ws (fun τ => tW d (ws.map τ) 0 * G τ) (Function.update σ w u)
        else 0 := by
      intro j _
      have hwj : ∀ τ : ℕ → ℕ, (∀ x ∉ ws, τ x = Function.update σ w j x) → τ w = j :=
        fun τ hτ => by rw [hτ w hw, Function.update_self]
      split_ifs with h0 h1
      · subst h0
        exact sumOver_congr _ _ _ _ _ fun τ hτ _ => by rw [tW_cons' _ _ _ _ hu, if_pos (hwj τ hτ)]
      · subst h1
        exact sumOver_congr _ _ _ _ _ fun τ hτ _ => by
          rw [tW_cons' _ _ _ _ hu, if_neg (by rw [hwj τ hτ]; exact h0), if_pos (hwj τ hτ)]
      · refine (sumOver_congr D ws _ (fun _ => 0) _ fun τ hτ _ => ?_).trans (sumOver_zero D ws _)
        rw [tW_cons' _ _ _ _ hu, if_neg (by rw [hwj τ hτ]; exact h0),
          if_neg (by rw [hwj τ hτ]; exact h1), zero_mul]
    rw [Finset.sum_congr rfl step]
    have e1 : ∀ j, (fun x => if x ∈ ws then 0 else Function.update σ w j x) =
        Function.update (fun x => if x ∈ w :: ws then 0 else σ x) w j := by
      intro j; funext x
      by_cases hx : x = w
      · subst hx; simp [hw]
      · by_cases hx' : x ∈ ws <;> simp [hx, hx', Function.update_of_ne hx]
    by_cases hu0 : u = 0
    · subst hu0
      simp only [if_true]
      rw [Finset.sum_congr rfl fun j _ => by rw [show (if j = 0 then sumOver D ws
          (fun τ => tW d (ws.map τ) 0 * G τ) (Function.update σ w 0) else if j = 0 then
          sumOver D ws (fun τ => tW d (ws.map τ) 0 * G τ) (Function.update σ w 0) else 0) =
          if j = 0 then sumOver D ws (fun τ => tW d (ws.map τ) 0 * G τ)
            (Function.update σ w 0) else 0 by split_ifs <;> rfl],
        Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hd), ih hnd' hD' 0 hd, if_pos rfl, e1]
      congr 1
      funext x
      by_cases hx : x = w
      · subst hx; simp
      · simp [hx, Function.update_of_ne hx]
    · simp only [hu0, if_false]
      have split : ∀ j ∈ range d, (if j = 0 then sumOver D ws (fun τ => tW d (ws.map τ) u * G τ)
          (Function.update σ w 0) else if j = u then sumOver D ws
          (fun τ => tW d (ws.map τ) 0 * G τ) (Function.update σ w u) else 0) =
          (if j = 0 then sumOver D ws (fun τ => tW d (ws.map τ) u * G τ)
            (Function.update σ w 0) else 0) + (if j = u then sumOver D ws
            (fun τ => tW d (ws.map τ) 0 * G τ) (Function.update σ w u) else 0) := by
        intro j _
        by_cases h0 : j = 0
        · have : j ≠ u := fun h => hu0 (h ▸ h0)
          simp [h0, this, Ne.symm hu0]
        · simp [h0]
      rw [Finset.sum_congr rfl split, Finset.sum_add_distrib, Finset.sum_ite_eq',
        Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hd), if_pos (Finset.mem_range.2 hu),
        ih hnd' hD' u hu, ih hnd' hD' 0 hd, if_neg hu0, if_pos rfl, e1, e1, List.sum_cons,
        add_comm]
      congr 1
      refine congrArg List.sum (List.map_congr_left fun x hx => ?_)
      congr 1
      funext y
      by_cases hy : y = x
      · subst hy; simp
      · by_cases hyw : y = w
        · subst hyw; simp [hy, Ne.symm hy]
        · simp [hy, hyw, Function.update_of_ne hy, Function.update_of_ne hyw]

/-- A multiplier from `c.1` through the wires `c.2.1` to `c.2.2`, as nodes. -/
def multNodes (dz pz px : ℕ) (c : ℕ × List ℕ × ℕ) : List Node :=
  [⟨.Z, c.2.1, [c.1], dz, pz⟩, ⟨.X, [c.2.2], c.2.1, dz, px⟩]

/-- The value of a multiplier given its input and output values. -/
noncomputable def multVal (d : ℕ) (c : ℕ × List ℕ × ℕ) (ρ : ℕ → ℕ) : ℂ :=
  (rt d)⁻¹ ^ (1 + c.2.1.length) * (if ρ c.2.2 % d = (c.2.1.length * ρ c.1) % d then (d : ℂ)
    else 0)

/-- Summing the wires of several multipliers. -/
theorem sumOver_mults (E : Env) (dz pz px : ℕ) (ms : List (ℕ × List ℕ × ℕ)) (ρ : ℕ → ℕ)
    (hd : 0 < E.dimv dz) (hnd : (ms.flatMap fun c => c.2.1).Nodup)
    (hends : ∀ c ∈ ms, ∀ c' ∈ ms, c.1 ∉ c'.2.1 ∧ c.2.2 ∉ c'.2.1)
    (hlab : ∀ x ∈ ms.flatMap (fun c => c.2.1), labIdx E.labs x = dz)
    (hr : ∀ c ∈ ms, ρ c.1 < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    sumOver (labDim E) (ms.flatMap fun c => c.2.1)
        (fun τ => (ms.map fun c => prodVal E (multNodes dz pz px c) τ).prod) ρ =
      (ms.map fun c => multVal (E.dimv dz) c ρ).prod := by
  have hg := sumOver_groups (D := labDim E) (ms.map fun c => (c.2.1,
      fun τ => prodVal E (multNodes dz pz px c) τ))
    (ms.flatMap fun c => [c.1, c.2.2])
  simp only [List.flatMap_map, List.map_map, Function.comp_def] at hg
  rw [hg hnd (fun x hx hx' => by
      obtain ⟨c, hc, hxc⟩ := List.mem_flatMap.1 hx
      obtain ⟨c', hc', hxc'⟩ := List.mem_flatMap.1 hx'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hxc
      rcases hxc with rfl | rfl
      · exact (hends c hc c' hc').1 hxc'
      · exact (hends c hc c' hc').2 hxc')
    (fun g hg' => by
      obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hg'
      intro τ τ' h
      simp only [multNodes, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
        val]
      have hw : c.2.1.map τ = c.2.1.map τ' :=
        List.map_congr_left fun w hw => h w (List.mem_append_left _ hw)
      rw [hw, h c.1 (List.mem_append_right _ (List.mem_flatMap.2 ⟨c, hc, by simp⟩)),
        h c.2.2 (List.mem_append_right _ (List.mem_flatMap.2 ⟨c, hc, by simp⟩))]) ρ]
  refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
  have hnd_c : c.2.1.Nodup := (List.nodup_flatMap.1 hnd).1 c hc
  rw [show prodVal E (multNodes dz pz px c) = prodVal E [⟨.Z, c.2.1, [c.1], dz, pz⟩,
    ⟨.X, [c.2.2], c.2.1, dz, px⟩] from rfl,
    multiplier_val E pz px dz c.1 c.2.2 c.2.1 ρ hd hnd_c (hends c hc c hc).1 (hends c hc c hc).2
      (fun w hw => hlab w (List.mem_flatMap.2 ⟨c, hc, hw⟩)) (hr c hc) hz hx]
  rfl

theorem flatMap_single_snd {β : Type*} (cs : List β) (fv : β → ℕ) :
    cs.flatMap (fun c => [fv c]) = cs.map fv := by
  induction cs with
  | nil => rfl
  | cons c cs ih => simp only [List.flatMap_cons, List.map_cons, ih]; rfl

/-- A W node whose outputs `fw c` pass through gadgets `g c` to the inputs `fv c` of a phaseless
pink dot with output `o`, summed over the outputs and inputs, as a character sum. -/
theorem wPinkSum {D : ℕ → ℕ} {β : Type*} (d : ℕ) (hd : 0 < d) (u o : ℕ) (cs : List β)
    (fw fv : β → ℕ) (g : β → ℕ → ℕ → ℂ) (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = 1) (σ : ℕ → ℕ)
    (hnd : (cs.map fw ++ cs.map fv).Nodup)
    (hD : ∀ x ∈ cs.map fw ++ cs.map fv, D x = d)
    (hu : u ∉ cs.map fw ++ cs.map fv) (ho : o ∉ cs.map fw ++ cs.map fv)
    (ru : σ u < d) :
    sumOver D (cs.map fw ++ cs.map fv)
        (fun ρ => tW d (cs.map fun c => ρ (fw c)) (ρ u) * (tX d φ [ρ o] (cs.map fun c => ρ (fv c)) *
          (cs.map fun c => g c (ρ (fw c)) (ρ (fv c))).prod)) σ =
      ∑ k ∈ range d, fou d (σ o) k *
        (if σ u = 0 then (cs.map fun c => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
            g c 0 y).prod
          else ((cs.map fw).map fun w => (cs.map fun c => ∑ y ∈ range d,
            (starRingEnd ℂ) (fou d y k) * g c (if fw c = w then σ u else 0) y).prod).sum) := by
  set ws := cs.map fw
  set vs := cs.map fv
  rw [List.nodup_append] at hnd
  obtain ⟨hws, hvs, hwv⟩ := hnd
  -- the pink dot as a character sum
  have hX : ∀ ρ : ℕ → ℕ, tX d φ [ρ o] (cs.map fun c => ρ (fv c)) =
      ∑ k ∈ range d, fou d (ρ o) k * (cs.map fun c => (starRingEnd ℂ) (fou d (ρ (fv c)) k)).prod := by
    intro ρ
    unfold tX
    refine Finset.sum_congr rfl fun k hk => ?_
    rw [hφ k (Finset.mem_range.1 hk), one_mul]
    simp [List.map_map, Function.comp_def]
  have hI : ∀ ρ : ℕ → ℕ, tW d (cs.map fun c => ρ (fw c)) (ρ u) * (tX d φ [ρ o]
      (cs.map fun c => ρ (fv c)) * (cs.map fun c => g c (ρ (fw c)) (ρ (fv c))).prod) =
      ∑ k ∈ range d, (fou d (ρ o) k * tW d (cs.map fun c => ρ (fw c)) (ρ u)) *
        (cs.map fun c => (starRingEnd ℂ) (fou d (ρ (fv c)) k) * g c (ρ (fw c)) (ρ (fv c))).prod := by
    intro ρ
    rw [hX, Finset.sum_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [List.prod_map_mul]
    ring
  simp only [hI]
  rw [sumOver_fsum]
  refine Finset.sum_congr rfl fun k hk => ?_
  set Gk : β → ℕ → ℂ := fun c x => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) * g c x y
  have hvsg : cs.flatMap (fun c => [fv c]) = vs := flatMap_single_snd cs fv
  have inner : ∀ ρ : ℕ → ℕ, sumOver D vs (fun τ => (fou d (τ o) k * tW d (cs.map fun c => τ (fw c))
      (τ u)) * (cs.map fun c => (starRingEnd ℂ) (fou d (τ (fv c)) k) * g c (τ (fw c)) (τ (fv c))).prod) ρ =
      (fou d (ρ o) k * tW d (cs.map fun c => ρ (fw c)) (ρ u)) * (cs.map fun c => Gk c (ρ (fw c))).prod := by
    intro ρ
    rw [sumOver_const_mul D vs (o :: u :: ws) _ _ (fun τ τ' h => by
        have hw : (cs.map fun c => τ (fw c)) = cs.map fun c => τ' (fw c) :=
          List.map_congr_left fun c hc => h (fw c) (by simp [ws, List.mem_map_of_mem hc])
        rw [hw, h o (by simp), h u (by simp)])
      (fun x hx hxv => by
        simp only [List.mem_cons] at hx
        rcases hx with rfl | rfl | hx
        · exact ho (List.mem_append_right _ hxv)
        · exact hu (List.mem_append_right _ hxv)
        · exact hwv x hx x hxv rfl) ρ]
    congr 1
    have hg := sumOver_groups (D := D) (cs.map fun c => ([fv c],
      fun τ => (starRingEnd ℂ) (fou d (τ (fv c)) k) * g c (τ (fw c)) (τ (fv c)))) ws
    simp only [List.flatMap_map, List.map_map, Function.comp_def] at hg
    rw [hvsg] at hg
    rw [hg hvs (fun x hx hxv => hwv x hx x hxv rfl) (fun gg hgg => by
        obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hgg
        intro τ τ' h
        simp only
        rw [h (fv c) (by simp), h (fw c) (List.mem_append_right _ (List.mem_map_of_mem hc))]) ρ]
    refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    have hcv : fv c ∈ vs := List.mem_map_of_mem hc
    have hcw : fw c ≠ fv c := fun h => hwv (fw c) (List.mem_map_of_mem hc) (fv c) hcv h
    simp only [sumOver, hD (fv c) (List.mem_append_right _ hcv), Function.update_self,
      Function.update_of_ne hcw, Gk]
  rw [sumOver_append]
  rw [show sumOver D ws (sumOver D vs fun τ => (fou d (τ o) k * tW d (cs.map fun c => τ (fw c))
      (τ u)) * (cs.map fun c => (starRingEnd ℂ) (fou d (τ (fv c)) k) * g c (τ (fw c)) (τ (fv c))).prod) σ =
      sumOver D ws (fun ρ => (fou d (ρ o) k * tW d (cs.map fun c => ρ (fw c)) (ρ u)) *
        (cs.map fun c => Gk c (ρ (fw c))).prod) σ from congrArg (fun F => sumOver D ws F σ)
      (funext inner)]
  rw [show sumOver D ws (fun ρ => (fou d (ρ o) k * tW d (cs.map fun c => ρ (fw c)) (ρ u)) *
      (cs.map fun c => Gk c (ρ (fw c))).prod) σ = fou d (σ o) k * sumOver D ws (fun ρ =>
        tW d (ws.map ρ) (σ u) * (cs.map fun c => Gk c (ρ (fw c))).prod) σ by
    rw [← sumOver_const_mul D ws [o] (fun ρ => fou d (ρ o) k) _ (fun τ τ' h => by
      simp only; rw [h o (by simp)]) (fun x hx hxw => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        subst hx; exact ho (List.mem_append_left _ hxw)) σ]
    refine sumOver_congr _ _ _ _ _ fun ρ hρ _ => ?_
    have eu : ρ u = σ u := hρ u (fun h => hu (List.mem_append_left _ h))
    simp only [eu, ws, List.map_map, Function.comp_def]
    ring]
  rw [sumOver_W d ws hws (fun x hx => hD x (List.mem_append_left _ hx)) (σ u) ru hd]
  congr 1
  split_ifs
  · refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    simp [Gk, ws, List.mem_map_of_mem hc]
  · refine congrArg List.sum (List.map_congr_left fun w hw => ?_)
    refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    by_cases h : fw c = w
    · simp [Gk, h]
    · have hm : ∃ a ∈ cs, fw a = fw c := ⟨c, hc, rfl⟩
      simp [Gk, h, Function.update_of_ne h, ws, hm]

theorem prod_one_hot {α : Type*} [DecidableEq α] (l : List α) (key : α → ℕ)
    (hnd : (l.map key).Nodup) (a : α) (ha : a ∈ l) (f : α → ℂ) (z : ℂ)
    (hf : ∀ b ∈ l, key b ≠ key a → f b = z) : (l.map f).prod = f a * z ^ (l.length - 1) := by
  induction l with
  | nil => simp at ha
  | cons b l ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map, not_exists, not_and] at hnd
    simp only [List.map_cons, List.prod_cons, List.length_cons, Nat.add_sub_cancel]
    rcases List.mem_cons.1 ha with rfl | ha'
    · have : (l.map f) = l.map fun _ => z := List.map_congr_left fun c hc =>
        hf c (List.mem_cons_of_mem _ hc) (fun h => hnd.1 c hc h)
      rw [this, List.map_const', List.prod_replicate]
    · have hb : key b ≠ key a := fun h => hnd.1 a ha' h.symm
      rw [hf b List.mem_cons_self hb, ih hnd.2 ha' (fun c hc => hf c (List.mem_cons_of_mem _ hc))]
      have hl : 1 ≤ l.length := List.length_pos_of_mem ha'
      obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hl
      rw [hn, show 1 + n - 1 = n by omega]
      ring

theorem sum_list_comm {α : Type*} (s : Finset ℕ) (l : List α) (f : ℕ → α → ℂ) :
    ∑ k ∈ s, (l.map (f k)).sum = (l.map fun c => ∑ k ∈ s, f k c).sum := by
  induction l with
  | nil => simp
  | cons c l ih => simp [Finset.sum_add_distrib, ih]

/-- A W node with input `u` into multipliers `ms` (from `c.1` through `c.2.1` to `c.2.2`) into
a phaseless pink dot with output `o`, summed over everything inside. -/
theorem wMultPinkVal (E : Env) (dz u o pq pz px : ℕ) (ms : List (ℕ × List ℕ × ℕ))
    (σ : ℕ → ℕ) (hd : 0 < E.dimv dz)
    (hnd : (ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1)).Nodup)
    (hu : u ∉ ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1))
    (ho : o ∉ ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1))
    (hlab : ∀ x ∈ ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1),
      labIdx E.labs x = dz)
    (ru : σ u < E.dimv dz) (ro : σ o < E.dimv dz)
    (hq : ∀ k < E.dimv dz, ex (E.phase pq k) = 1) (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    sumOver (labDim E) (ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1))
        (prodVal E (⟨.W, ms.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], ms.map (fun c => c.2.2), dz, pq⟩ ::
          ms.flatMap (multNodes dz pz px))) σ =
      (ms.map fun c => (rt (E.dimv dz))⁻¹ ^ (1 + c.2.1.length) * E.dimv dz).prod *
        ((rt (E.dimv dz))⁻¹ ^ ms.length * rt (E.dimv dz)) *
        (if σ u = 0 then (if σ o = 0 then 1 else 0)
          else (ms.map fun c => if σ o = (c.2.1.length * σ u) % E.dimv dz then (1 : ℂ)
            else 0).sum) := by
  set d := E.dimv dz with hdd
  set ws := ms.map Prod.fst
  set vs := ms.map fun c => c.2.2
  set wr := ms.flatMap fun c => c.2.1
  have hnd' := hnd
  rw [List.nodup_append] at hnd'
  obtain ⟨hwv, hwr, hdis⟩ := hnd'
  have hD : ∀ x ∈ ws ++ vs ++ wr, labDim E x = d := fun x hx => labDim_of E x dz (hlab x hx)
  rw [sumOver_append]
  set g : ℕ × List ℕ × ℕ → ℕ → ℕ → ℂ := fun c x y =>
    (rt d)⁻¹ ^ (1 + c.2.1.length) * (if y % d = (c.2.1.length * x) % d then (d : ℂ) else 0)
  have hin : ∀ ρ : ℕ → ℕ, (∀ x ∉ ws ++ vs, ρ x = σ x) → (∀ x ∈ ws ++ vs, ρ x < labDim E x) →
      sumOver (labDim E) wr (prodVal E (⟨.W, ws, [u], dz, 0⟩ :: ⟨.X, [o], vs, dz, pq⟩ ::
        ms.flatMap (multNodes dz pz px))) ρ =
      tW d (ms.map fun c => ρ c.1) (ρ u) * (tX d (E.phase pq) [ρ o] (ms.map fun c => ρ c.2.2) *
        (ms.map fun c => g c (ρ c.1) (ρ c.2.2)).prod) := by
    intro ρ hρ hr
    have split : prodVal E (⟨.W, ws, [u], dz, 0⟩ :: ⟨.X, [o], vs, dz, pq⟩ ::
        ms.flatMap (multNodes dz pz px)) = fun τ => (tW d (ws.map τ) (τ u) *
          tX d (E.phase pq) [τ o] (vs.map τ)) *
          (ms.map fun c => prodVal E (multNodes dz pz px c) τ).prod := by
      funext τ
      simp only [prodVal, List.map_cons, List.prod_cons, prod_flatMap', val, List.map_nil]
      ring
    rw [split, sumOver_const_mul _ wr (u :: o :: (ws ++ vs)) _ _ (fun τ τ' h => by
        have hw : ws.map τ = ws.map τ' := List.map_congr_left fun x hx =>
          h x (by simp [hx])
        have hv : vs.map τ = vs.map τ' := List.map_congr_left fun x hx =>
          h x (by simp [hx])
        rw [hw, hv, h u (by simp), h o (by simp)])
      (fun x hx hxw => by
        simp only [List.mem_cons] at hx
        rcases hx with rfl | rfl | hx
        · exact hu (List.mem_append_right _ hxw)
        · exact ho (List.mem_append_right _ hxw)
        · exact hdis x hx x hxw rfl) ρ]
    rw [sumOver_mults E dz pz px ms ρ hd hwr
      (fun c hc c' hc' => ⟨fun h => hdis c.1 (List.mem_append_left _ (List.mem_map_of_mem hc)) c.1
          (List.mem_flatMap.2 ⟨c', hc', h⟩) rfl,
        fun h => hdis c.2.2 (List.mem_append_right _ (List.mem_map_of_mem hc)) c.2.2
          (List.mem_flatMap.2 ⟨c', hc', h⟩) rfl⟩)
      (fun x hx => hlab x (List.mem_append_right _ hx))
      (fun c hc => by
        have := hr c.1 (List.mem_append_left _ (List.mem_map_of_mem hc))
        rwa [hD c.1 (List.mem_append_left _ (List.mem_append_left _ (List.mem_map_of_mem hc)))]
          at this) hz hx]
    simp only [ws, vs, List.map_map, Function.comp_def, multVal, g]
    ring
  rw [sumOver_congr _ _ _ _ _ fun ρ hρ hr => hin ρ hρ hr]
  rw [wPinkSum (D := labDim E) d hd u o ms Prod.fst (fun c => c.2.2) g (E.phase pq) hq σ hwv
    (fun x hx => hD x (List.mem_append_left _ hx)) (fun h => hu (List.mem_append_left _ h))
    (fun h => ho (List.mem_append_left _ h)) ru]
  have hGk : ∀ (c : ℕ × List ℕ × ℕ) (x k : ℕ), (∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
      g c x y) = ((rt d)⁻¹ ^ (1 + c.2.1.length) * d) *
        (starRingEnd ℂ) (fou d ((c.2.1.length * x) % d) k) := by
    intro c x k
    have e : ∀ y ∈ range d, (starRingEnd ℂ) (fou d y k) * g c x y =
        if y = (c.2.1.length * x) % d then ((rt d)⁻¹ ^ (1 + c.2.1.length) * d) *
          (starRingEnd ℂ) (fou d y k) else 0 := by
      intro y hy
      simp only [g, Nat.mod_eq_of_lt (Finset.mem_range.1 hy)]
      split_ifs <;> ring
    rw [Finset.sum_congr rfl e, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd))]
  simp only [hGk]
  have hc0 : ∀ k, (starRingEnd ℂ) (fou d 0 k) = (rt d)⁻¹ := by
    intro k; rw [fou_zero']; simp [rt, conj_sqrt]
  set A := (ms.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod
  have hsplit : ∀ (t : ℕ × List ℕ × ℕ → ℕ) (k : ℕ),
      (ms.map fun c => ((rt d)⁻¹ ^ (1 + c.2.1.length) * d) *
        (starRingEnd ℂ) (fou d (t c) k)).prod =
      A * (ms.map fun c => (starRingEnd ℂ) (fou d (t c) k)).prod := by
    intro t k; rw [List.prod_map_mul]
  by_cases hu0 : σ u = 0
  · simp only [hu0, if_true, mul_zero, Nat.zero_mod]
    simp only [hc0]
    have h0 : (ms.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ) * (rt d)⁻¹).prod =
        A * (rt d)⁻¹ ^ ms.length := by
      rw [List.prod_map_mul, List.map_const', List.prod_replicate]
    rw [h0, ← Finset.sum_mul, sum_fou_left d hd, Nat.mod_eq_of_lt ro]
    by_cases ho0 : σ o = 0
    · simp only [ho0, if_true, mul_one]
      rw [rt_inv_mul d hd]
      ring
    · simp [ho0]
  · simp only [hu0, if_false]
    have hws : ws.Nodup := (List.nodup_append.1 hwv).1
    have hone : ∀ k, ∀ c' ∈ ms, (ms.map fun c => ((rt d)⁻¹ ^ (1 + c.2.1.length) * d) *
        (starRingEnd ℂ) (fou d ((c.2.1.length * if c.1 = c'.1 then σ u else 0) % d) k)).prod =
        A * ((starRingEnd ℂ) (fou d ((c'.2.1.length * σ u) % d) k) *
          (rt d)⁻¹ ^ (ms.length - 1)) := by
      intro k c' hc'
      rw [List.prod_map_mul]
      congr 1
      rw [prod_one_hot ms Prod.fst hws c' hc' _ ((rt d)⁻¹) (fun b _ hb => by
        simp only [if_neg hb, mul_zero, Nat.zero_mod, hc0])]
      simp
    simp only [ws, List.map_map, Function.comp_def]
    have hk : ∀ k ∈ range d, fou d (σ o) k * (ms.map fun c' => (ms.map fun c =>
        ((rt d)⁻¹ ^ (1 + c.2.1.length) * d) * (starRingEnd ℂ) (fou d ((c.2.1.length *
          if c.1 = c'.1 then σ u else 0) % d) k)).prod).sum =
        (ms.map fun c' => A * (rt d)⁻¹ ^ (ms.length - 1) * (fou d (σ o) k *
          (starRingEnd ℂ) (fou d ((c'.2.1.length * σ u) % d) k))).sum := by
      intro k _
      rw [← List.sum_map_mul_left]
      refine congrArg List.sum (List.map_congr_left fun c' hc' => ?_)
      rw [hone k c' hc']
      ring
    rw [Finset.sum_congr rfl hk, sum_list_comm]
    have hu' : ∀ c' ∈ ms, (∑ k ∈ range d, A * (rt d)⁻¹ ^ (ms.length - 1) * (fou d (σ o) k *
        (starRingEnd ℂ) (fou d ((c'.2.1.length * σ u) % d) k))) =
        A * (rt d)⁻¹ ^ (ms.length - 1) *
          (if σ o = (c'.2.1.length * σ u) % d then 1 else 0) := by
      intro c' _
      rw [← Finset.mul_sum]
      congr 1
      rw [Finset.sum_congr rfl fun k _ => by rw [fou_symm d (σ o) k,
        fou_symm d ((c'.2.1.length * σ u) % d) k],
        fou_unitary d hd _ _ ro (Nat.mod_lt _ hd)]
    rw [List.map_congr_left hu', List.sum_map_mul_left]
    by_cases h : ms = []
    · subst h; simp
    · have hl : 1 ≤ ms.length := List.length_pos_iff.2 h
      obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hl
      have hr := rt_ne d hd
      rw [hn, show 1 + n - 1 = n by omega, pow_add, pow_one]
      field_simp

/-- The labels summed inside a V box with front label `u` and multipliers `ms`. -/
def vLabs (u : ℕ) (ms : List (ℕ × List ℕ × ℕ)) : List ℕ :=
  u :: (ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1))

/-- The nodes of a V box: the box `m` (from `i` to `u`), the W node, the pink dot (to `o`) and
the multipliers. -/
def vNodes (dz u o pq pz px : ℕ) (m : Node) (ms : List (ℕ × List ℕ × ℕ)) : List Node :=
  m :: ⟨.W, ms.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], ms.map (fun c => c.2.2), dz, pq⟩ ::
    ms.flatMap (multNodes dz pz px)

/-- The constant of a V box. -/
noncomputable def vConst (d : ℕ) (ms : List (ℕ × List ℕ × ℕ)) : ℂ :=
  (ms.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * d).prod * ((rt d)⁻¹ ^ ms.length * rt d)

theorem vBoxVal (E : Env) (dz i u o pq pz px : ℕ) (m : Node) (ms : List (ℕ × List ℕ × ℕ))
    (σ : ℕ → ℕ) (hd : 0 < E.dimv dz) (hm : isZc m = true) (hdm : m.dim = dz)
    (hlm : m.outs ++ m.ins = [u, i]) (hnd : (vLabs u ms).Nodup)
    (hi : i ∉ vLabs u ms) (ho : o ∉ vLabs u ms)
    (hlab : ∀ x ∈ vLabs u ms, labIdx E.labs x = dz)
    (ri : σ i < E.dimv dz) (ro : σ o < E.dimv dz)
    (hq : ∀ k < E.dimv dz, ex (E.phase pq k) = 1) (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    sumOver (labDim E) (vLabs u ms) (prodVal E (vNodes dz u o pq pz px m ms)) σ =
      zc E m (σ i) * vConst (E.dimv dz) ms *
        (if σ i = 0 then (if σ o = 0 then 1 else 0)
          else (ms.map fun c => if σ o = (c.2.1.length * σ i) % E.dimv dz then (1 : ℂ)
            else 0).sum) := by
  simp only [vLabs, List.nodup_cons, List.mem_cons, not_or] at hnd hi ho
  obtain ⟨hu, hnd'⟩ := hnd
  simp only [vLabs, sumOver, labDim_of E u dz (hlab u (by simp [vLabs]))]
  have step : ∀ j ∈ range (E.dimv dz), sumOver (labDim E)
      (ms.map Prod.fst ++ ms.map (fun c => c.2.2) ++ ms.flatMap (fun c => c.2.1))
      (prodVal E (vNodes dz u o pq pz px m ms)) (Function.update σ u j) =
      if j = σ i then zc E m (σ i) * vConst (E.dimv dz) ms *
        (if σ i = 0 then (if σ o = 0 then 1 else 0)
          else (ms.map fun c => if σ o = (c.2.1.length * σ i) % E.dimv dz then (1 : ℂ)
            else 0).sum) else 0 := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    have split : prodVal E (vNodes dz u o pq pz px m ms) = fun τ => val E τ m *
        prodVal E (⟨.W, ms.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], ms.map (fun c => c.2.2), dz, pq⟩ ::
          ms.flatMap (multNodes dz pz px)) τ := by
      funext τ; simp only [vNodes, prodVal, List.map_cons, List.prod_cons]
    rw [split, sumOver_const_mul _ _ [u, i] (fun τ => val E τ m) _ (fun τ τ' h => by
        simp only
        rw [val_zc _ _ m hm, val_zc _ _ m hm, hlm]
        simp only [List.map_cons, List.map_nil]
        rw [h u (by simp), h i (by simp)])
      (fun x hx hx' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases hx with rfl | rfl
        · exact hu hx'
        · exact hi.2 hx')]
    rw [wMultPinkVal E dz u o pq pz px ms _ hd hnd' hu (fun h => ho.2 (by
        rcases List.mem_append.1 h with h | h <;> simp_all))
      (fun x hx => hlab x (List.mem_cons_of_mem _ hx))
      (by rw [Function.update_self]; exact hj')
      (by rw [Function.update_of_ne ho.1]; exact ro) hq hz hx]
    rw [val_zc_two E _ m hm u i hlm (by rw [Function.update_of_ne hi.1, hdm]; exact ri)]
    simp only [Function.update_self, Function.update_of_ne hi.1, Function.update_of_ne ho.1]
    by_cases h : j = σ i
    · subst h; simp only [if_true, vConst]; ring
    · simp [h]
  rw [Finset.sum_congr rfl step, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ri)]

theorem sum_ind_count {α : Type*} (l : List α) (P : α → Prop) [DecidablePred P] :
    (l.map fun c => if P c then (1 : ℂ) else 0).sum = ((l.filter fun c => decide (P c)).length : ℂ) := by
  induction l with
  | nil => simp
  | cons c l ih =>
    by_cases h : P c <;> simp [h, ih, List.filter_cons] <;> ring

theorem vNodes_labels (dz u o pq pz px i : ℕ) (m : Node) (ms : List (ℕ × List ℕ × ℕ))
    (hlm : m.outs ++ m.ins = [u, i]) :
    ∀ x ∈ labelsOf (vNodes dz u o pq pz px m ms), x ∈ i :: o :: vLabs u ms := by
  intro x hx
  simp only [labelsOf, vNodes, List.flatMap_cons, List.mem_append, Node.labels, hlm,
    List.mem_flatMap, multNodes, List.mem_cons, List.mem_map, List.not_mem_nil, or_false,
    vLabs] at hx ⊢
  aesop

theorem w_pair_nonzero (d a b j : ℕ) (hj : j < d) :
    tW d [a, b] j * (if j = 0 then (0 : ℂ) else 1) =
      (if j = a then (if b = 0 ∧ a ≠ 0 then 1 else 0) else 0) +
        (if j = b then (if a = 0 ∧ b ≠ 0 then 1 else 0) else 0) := by
  rw [tW_pair d a b j hj]
  simp only [@eq_comm _ j a, @eq_comm _ j b]
  by_cases h1 : a = j <;> by_cases h2 : b = j <;> by_cases h3 : j = 0 <;> simp_all

theorem wsum2 (d I : ℕ) (hd : 0 < d) (hI : I < d) (F : ℕ → ℕ → ℂ) :
    ∑ a ∈ range d, ∑ b ∈ range d, tW d [a, b] I * F a b =
      F I 0 + (if I ≠ 0 then F 0 I else 0) := by
  have e1 : ∀ a ∈ range d, ∀ b ∈ range d, tW d [a, b] I * F a b =
      (if a = I ∧ b = 0 then F a b else 0) + (if a = 0 ∧ b = I then
        (if I ≠ 0 then F a b else 0) else 0) := by
    intro a _ b _
    rw [tW_pair d a b I hI]
    by_cases h1 : a = I ∧ b = 0 <;> by_cases h2 : a = 0 ∧ b = I <;> by_cases h3 : I = 0 <;>
      simp [h1, h2, h3]
  rw [Finset.sum_congr rfl fun a ha => Finset.sum_congr rfl fun b hb => e1 a ha b hb]
  simp only [Finset.sum_add_distrib]
  rw [dsum_pick d I 0 hI hd F, dsum_pick d 0 I hd hI (fun a b => if I ≠ 0 then F a b else 0)]

/-- The value of a V box over its constant: output `x`, input `y`. -/
noncomputable def vF (E : Env) (m : Node) (ms : List (ℕ × List ℕ × ℕ)) (d x y : ℕ) : ℂ :=
  zc E m y * (if y = 0 then (if x = 0 then 1 else 0)
    else (ms.map fun c => if x = (c.2.1.length * y) % d then (1 : ℂ) else 0).sum)

theorem vConst_congr (d : ℕ) (ms ms' : List (ℕ × List ℕ × ℕ))
    (h : ms'.map (fun c => c.2.1.length) = ms.map (fun c => c.2.1.length)) :
    vConst d ms' = vConst d ms := by
  have hl : ms'.length = ms.length := by
    rw [← List.length_map (f := fun c : ℕ × List ℕ × ℕ => c.2.1.length), h, List.length_map]
  have hp : (ms'.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod =
      (ms.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod := by
    have e1 : ∀ l : List (ℕ × List ℕ × ℕ), (l.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) *
        (d : ℂ)) = (l.map fun c => c.2.1.length).map fun n => (rt d)⁻¹ ^ (1 + n) * (d : ℂ) :=
      fun l => by rw [List.map_map]; rfl
    rw [e1, e1, h]
  simp only [vConst, hp, hl]

theorem vF_congr (E : Env) (m m' : Node) (ms ms' : List (ℕ × List ℕ × ℕ)) (d x y : ℕ)
    (hc : zc E m' y = zc E m y)
    (h : ms'.map (fun c => c.2.1.length) = ms.map (fun c => c.2.1.length)) :
    vF E m' ms' d x y = vF E m ms d x y := by
  have e1 : ∀ l : List (ℕ × List ℕ × ℕ), (l.map fun c => if x = (c.2.1.length * y) % d then
      (1 : ℂ) else 0) = (l.map fun c => c.2.1.length).map fun n => if x = (n * y) % d then
        (1 : ℂ) else 0 := fun l => by rw [List.map_map]; rfl
  simp only [vF, hc, e1, h]

/-- [P1] (VW) with the V box constant on the left. -/
theorem zxwVW' (E : Env) (dz i a b mo u₁ pq₁ pz₁ px₁ p q u₂ pq₂ pz₂ px₂ u₃ pq₃ pz₃ px₃ : ℕ)
    (m₁ m₂ m₃ : Node) (ms₁ ms₂ ms₃ : List (ℕ × List ℕ × ℕ)) (iface : List ℕ)
    (hd : 0 < E.dimv dz) (hm₁ : isZc m₁ = true) (hm₂ : isZc m₂ = true) (hm₃ : isZc m₃ = true)
    (hdm₁ : m₁.dim = dz) (hdm₂ : m₂.dim = dz) (hdm₃ : m₃.dim = dz)
    (hl₁ : m₁.outs ++ m₁.ins = [u₁, i]) (hl₂ : m₂.outs ++ m₂.ins = [u₂, p])
    (hl₃ : m₃.outs ++ m₃.ins = [u₃, q])
    (hndL : (mo :: vLabs u₁ ms₁).Nodup) (hoL : ∀ x ∈ [i, a, b], x ∉ mo :: vLabs u₁ ms₁)
    (hndR : (p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃)).Nodup)
    (hoR : ∀ x ∈ [i, a, b], x ∉ p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃))
    (hlab : ∀ x ∈ mo :: p :: q :: (vLabs u₁ ms₁ ++ vLabs u₂ ms₂ ++ vLabs u₃ ms₃),
      labIdx E.labs x = dz)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz) (ha : a ∈ iface ∧ labIdx E.labs a = dz)
    (hb : b ∈ iface ∧ labIdx E.labs b = dz)
    (hph : ∀ ph ∈ [pq₁, pz₁, px₁, pq₂, pz₂, px₂, pq₃, pz₃, px₃], ∀ k < E.dimv dz,
      ex (E.phase ph k) = 1)
    (hbox : ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E m₁ k = 0)
    (h0 : zc E m₁ 0 = 1)
    (hc₂ : ∀ k < E.dimv dz, zc E m₂ k = zc E m₁ k) (hc₃ : ∀ k < E.dimv dz, zc E m₃ k = zc E m₁ k)
    (hlen₂ : ms₂.map (fun c => c.2.1.length) = ms₁.map (fun c => c.2.1.length))
    (hlen₃ : ms₃.map (fun c => c.2.1.length) = ms₁.map (fun c => c.2.1.length))
    (hcount : ∀ x < E.dimv dz, (ms₁.filter fun c => decide (x = (c.2.1.length *
      (E.dimv dz - 1)) % E.dimv dz)).length = if x = 0 then 0 else 1)
    (s : ℂ) (hs : s = vConst (E.dimv dz) ms₁) :
    LocalEq E (vNodes dz u₁ mo pq₁ pz₁ px₁ m₁ ms₁ ++ [⟨.W, [a, b], [mo], dz, 0⟩])
      (mo :: vLabs u₁ ms₁)
      (⟨.W, [p, q], [i], dz, 0⟩ :: (vNodes dz u₂ a pq₂ pz₂ px₂ m₂ ms₂ ++
        vNodes dz u₃ b pq₃ pz₃ px₃ m₃ ms₃)) (p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃))
      iface s 1 := by
  subst hs
  intro τ hτ
  set d := E.dimv dz with hdd
  have ri := hτ i hi.1
  have ra := hτ a ha.1
  have rb := hτ b hb.1
  rw [labDim_of E i dz hi.2] at ri
  rw [labDim_of E a dz ha.2] at ra
  rw [labDim_of E b dz hb.2] at rb
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨hq₁, hz₁, hx₁, hq₂, hz₂, hx₂, hq₃, hz₃, hx₃⟩ := hph'
  have hiL := hoL i (by simp)
  have haL := hoL a (by simp)
  have hbL := hoL b (by simp)
  have hiR := hoR i (by simp)
  have haR := hoR a (by simp)
  have hbR := hoR b (by simp)
  have nim : i ≠ mo := fun h => hiL (by rw [h]; exact List.mem_cons_self)
  have nam : a ≠ mo := fun h => haL (by rw [h]; exact List.mem_cons_self)
  have nbm : b ≠ mo := fun h => hbL (by rw [h]; exact List.mem_cons_self)
  simp only [List.nodup_cons, List.mem_cons, not_or] at hndL
  obtain ⟨hmo, hnd₁⟩ := hndL
  -- the left side
  have hL : sumOver (labDim E) (mo :: vLabs u₁ ms₁)
      (prodVal E (vNodes dz u₁ mo pq₁ pz₁ px₁ m₁ ms₁ ++ [⟨.W, [a, b], [mo], dz, 0⟩])) τ =
      vConst d ms₁ * ∑ j ∈ range d, tW d [τ a, τ b] j * vF E m₁ ms₁ d j (τ i) := by
    simp only [sumOver, labDim_of E mo dz (hlab mo (by simp))]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hj' := Finset.mem_range.1 hj
    have split : prodVal E (vNodes dz u₁ mo pq₁ pz₁ px₁ m₁ ms₁ ++ [⟨.W, [a, b], [mo], dz, 0⟩]) =
        fun ρ => tW d [ρ a, ρ b] (ρ mo) * prodVal E (vNodes dz u₁ mo pq₁ pz₁ px₁ m₁ ms₁) ρ := by
      funext ρ
      simp only [prodVal, List.map_append, List.prod_append, List.map_cons, List.map_nil,
        List.prod_cons, List.prod_nil, mul_one, val]
      ring
    rw [split, sumOver_const_mul _ _ [a, b, mo] (fun ρ => tW d [ρ a, ρ b] (ρ mo)) _
      (fun ρ ρ' h => by simp only; rw [h a (by simp), h b (by simp), h mo (by simp)])
      (fun x hx hx' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases hx with rfl | rfl | rfl
        · exact haL (List.mem_cons_of_mem _ hx')
        · exact hbL (List.mem_cons_of_mem _ hx')
        · exact hmo hx')]
    rw [vBoxVal E dz i u₁ mo pq₁ pz₁ px₁ m₁ ms₁ _ hd hm₁ hdm₁ hl₁ hnd₁
      (fun h => hiL (List.mem_cons_of_mem _ h)) hmo
      (fun x hx => hlab x (by simp [hx]))
      (by rw [Function.update_of_ne nim]; exact ri)
      (by rw [Function.update_self]; exact hj') hq₁ hz₁ hx₁]
    simp only [Function.update_self, Function.update_of_ne nim, Function.update_of_ne nam,
      Function.update_of_ne nbm, vF]
    ring
  have npq : p ≠ q := fun h => (List.nodup_cons.1 hndR).1 (by rw [h]; exact List.mem_cons_self)
  have nip : i ≠ p := fun h => hiR (by rw [h]; exact List.mem_cons_self)
  have niq : i ≠ q := fun h => hiR (by rw [h]; simp)
  have nap : a ≠ p := fun h => haR (by rw [h]; exact List.mem_cons_self)
  have naq : a ≠ q := fun h => haR (by rw [h]; simp)
  have nbp : b ≠ p := fun h => hbR (by rw [h]; exact List.mem_cons_self)
  have nbq : b ≠ q := fun h => hbR (by rw [h]; simp)
  simp only [List.nodup_cons, List.mem_cons, List.mem_append, not_or] at hndR
  obtain ⟨⟨-, hp2, hp3⟩, ⟨hq2, hq3⟩, hnd23⟩ := hndR
  obtain ⟨hnd₂, hnd₃, h23⟩ := List.nodup_append.1 hnd23
  have hR : sumOver (labDim E) (p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃))
      (prodVal E (⟨.W, [p, q], [i], dz, 0⟩ :: (vNodes dz u₂ a pq₂ pz₂ px₂ m₂ ms₂ ++
        vNodes dz u₃ b pq₃ pz₃ px₃ m₃ ms₃))) τ =
      vConst d ms₂ * vConst d ms₃ * ∑ jp ∈ range d, ∑ jq ∈ range d, tW d [jp, jq] (τ i) *
        (vF E m₂ ms₂ d (τ a) jp * vF E m₃ ms₃ d (τ b) jq) := by
    simp only [sumOver, labDim_of E p dz (hlab p (by simp)), labDim_of E q dz (hlab q (by simp))]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun jp hjp => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun jq hjq => ?_
    have hjp' := Finset.mem_range.1 hjp
    have hjq' := Finset.mem_range.1 hjq
    set ρ := Function.update (Function.update τ p jp) q jq
    have split : prodVal E (⟨.W, [p, q], [i], dz, 0⟩ :: (vNodes dz u₂ a pq₂ pz₂ px₂ m₂ ms₂ ++
        vNodes dz u₃ b pq₃ pz₃ px₃ m₃ ms₃)) = fun τ' => tW d [τ' p, τ' q] (τ' i) *
          (prodVal E (vNodes dz u₂ a pq₂ pz₂ px₂ m₂ ms₂) τ' *
            prodVal E (vNodes dz u₃ b pq₃ pz₃ px₃ m₃ ms₃) τ') := by
      funext τ'
      simp only [prodVal, List.map_cons, List.prod_cons, List.map_append, List.prod_append, val,
        List.map_nil]
      rfl
    rw [split, sumOver_const_mul _ _ [p, q, i] (fun τ' => tW d [τ' p, τ' q] (τ' i)) _
      (fun x y h => by simp only; rw [h p (by simp), h q (by simp), h i (by simp)])
      (fun x hx hx' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases List.mem_append.1 hx' with h | h <;> rcases hx with rfl | rfl | rfl <;>
          first | exact hp2 h | exact hp3 h | exact hq2 h | exact hq3 h |
            exact hiR (by simp [h]))]
    have l2 := vNodes_labels dz u₂ a pq₂ pz₂ px₂ p m₂ ms₂ hl₂
    have l3 := vNodes_labels dz u₃ b pq₃ pz₃ px₃ q m₃ ms₃ hl₃
    rw [sumOver_mul_split (D := labDim E) (vLabs u₂ ms₂) (vLabs u₃ ms₃) _ _ _ _
      (prodVal_reads E _) (prodVal_reads E _)
      (fun x hx hx3 => by
        rcases List.mem_cons.1 (l2 x hx) with rfl | h
        · exact hp3 hx3
        rcases List.mem_cons.1 h with rfl | h
        · exact haR (by simp [hx3])
        · exact h23 x h x hx3 rfl)
      (fun x hx hx2 => by
        rcases List.mem_cons.1 (l3 x hx) with rfl | h
        · exact hq2 hx2
        rcases List.mem_cons.1 h with rfl | h
        · exact hbR (by simp [hx2])
        · exact h23 x hx2 x h rfl)]
    have eρp : ρ p = jp := by simp only [ρ]; rw [Function.update_of_ne npq, Function.update_self]
    have eρq : ρ q = jq := by simp only [ρ]; rw [Function.update_self]
    have eρi : ρ i = τ i := by
      simp only [ρ]; rw [Function.update_of_ne niq, Function.update_of_ne nip]
    have eρa : ρ a = τ a := by
      simp only [ρ]; rw [Function.update_of_ne naq, Function.update_of_ne nap]
    have eρb : ρ b = τ b := by
      simp only [ρ]; rw [Function.update_of_ne nbq, Function.update_of_ne nbp]
    rw [vBoxVal E dz p u₂ a pq₂ pz₂ px₂ m₂ ms₂ ρ hd hm₂ hdm₂ hl₂ hnd₂ hp2
        (fun h => haR (by simp [h])) (fun x hx => hlab x (by simp [hx]))
        (by rw [eρp]; exact hjp') (by rw [eρa]; exact ra) hq₂ hz₂ hx₂,
      vBoxVal E dz q u₃ b pq₃ pz₃ px₃ m₃ ms₃ ρ hd hm₃ hdm₃ hl₃ hnd₃ hq3
        (fun h => hbR (by simp [h])) (fun x hx => hlab x (by simp [hx]))
        (by rw [eρq]; exact hjq') (by rw [eρb]; exact rb) hq₃ hz₃ hx₃,
      eρp, eρq, eρi, eρa, eρb]
    simp only [vF]
    ring
  rw [hL, hR, one_mul, vConst_congr d ms₁ ms₂ hlen₂, vConst_congr d ms₁ ms₃ hlen₃]
  have hF2 : ∀ x y, y < d → vF E m₂ ms₂ d x y = vF E m₁ ms₁ d x y :=
    fun x y hy => vF_congr E m₁ m₂ ms₁ ms₂ d x y (hc₂ y hy) hlen₂
  have hF3 : ∀ x y, y < d → vF E m₃ ms₃ d x y = vF E m₁ ms₁ d x y :=
    fun x y hy => vF_congr E m₁ m₃ ms₁ ms₃ d x y (hc₃ y hy) hlen₃
  rw [Finset.sum_congr rfl fun jp hjp => Finset.sum_congr rfl fun jq hjq => by
    rw [hF2 _ _ (Finset.mem_range.1 hjp), hF3 _ _ (Finset.mem_range.1 hjq)]]
  rw [wsum2 d (τ i) hd ri (fun jp jq => vF E m₁ ms₁ d (τ a) jp * vF E m₁ ms₁ d (τ b) jq)]
  have hv0 : ∀ x, vF E m₁ ms₁ d x 0 = if x = 0 then 1 else 0 := fun x => by simp [vF, h0]
  simp only [hv0]
  by_cases i0 : τ i = 0
  · simp only [i0, hv0, ne_eq, not_true_eq_false, if_false, add_zero]
    have : ∀ j ∈ range d, tW d [τ a, τ b] j * (if j = 0 then (1 : ℂ) else 0) =
        if j = 0 then tW d [τ a, τ b] 0 else 0 := fun j _ => by
      split_ifs with h <;> simp [h]
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hd),
      tW_pair d _ _ 0 hd]
    by_cases ha0 : τ a = 0 <;> by_cases hb0 : τ b = 0 <;> simp [ha0, hb0, h0]
    all_goals ring
  · by_cases i1 : τ i = d - 1
    · have hcnt : ∀ x < d, vF E m₁ ms₁ d x (τ i) = zc E m₁ (τ i) * (if x = 0 then 0 else 1) := by
        intro x hx
        simp only [vF, if_neg i0]
        rw [sum_ind_count, i1, hcount x hx]
        split_ifs <;> simp
      rw [Finset.sum_congr rfl fun j hj => by rw [hcnt j (Finset.mem_range.1 hj)], hcnt _ ra,
        hcnt _ rb, if_pos i0]
      have hsum : ∑ j ∈ range d, tW d [τ a, τ b] j * (zc E m₁ (τ i) *
          (if j = 0 then 0 else 1)) = zc E m₁ (τ i) * ((if τ a ≠ 0 ∧ τ b = 0 then 1 else 0) +
            (if τ a = 0 ∧ τ b ≠ 0 then 1 else 0)) := by
        have e : ∀ j ∈ range d, tW d [τ a, τ b] j * (zc E m₁ (τ i) * (if j = 0 then 0 else 1)) =
            zc E m₁ (τ i) * ((if j = τ a then (if τ b = 0 ∧ τ a ≠ 0 then 1 else 0) else 0) +
              (if j = τ b then (if τ a = 0 ∧ τ b ≠ 0 then 1 else 0) else 0)) := by
          intro j hj
          rw [mul_left_comm, w_pair_nonzero d _ _ j (Finset.mem_range.1 hj)]
        rw [Finset.sum_congr rfl e, ← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_ite_eq',
          Finset.sum_ite_eq', if_pos (Finset.mem_range.2 ra), if_pos (Finset.mem_range.2 rb)]
        congr 1
        by_cases h4 : τ a = 0 <;> by_cases h5 : τ b = 0 <;> simp [h4, h5]
      rw [hsum]
      by_cases h4 : τ a = 0 <;> by_cases h5 : τ b = 0 <;> simp [h4, h5]
      all_goals ring
    · have hz0 : zc E m₁ (τ i) = 0 := hbox _ ri i0 i1
      have : ∀ x, vF E m₁ ms₁ d x (τ i) = 0 := fun x => by simp [vF, hz0]
      simp [this]

theorem vConst_pow (d : ℕ) (hd : 0 < d) (ms : List (ℕ × List ℕ × ℕ)) :
    vConst d ms * rt d ^ ((ms.map fun c => c.2.1.length).sum) = rt d := by
  have hr := rt_ne d hd
  have key : ∀ l : List (ℕ × List ℕ × ℕ),
      (l.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod *
        rt d ^ ((l.map fun c => c.2.1.length).sum) * (rt d)⁻¹ ^ l.length = 1 := by
    intro l
    induction l with
    | nil => simp
    | cons c l ih =>
      simp only [List.map_cons, List.prod_cons, List.sum_cons, List.length_cons]
      have hf : (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ) * rt d ^ c.2.1.length * (rt d)⁻¹ = 1 := by
        rw [← rt_sq' d, show (rt d)⁻¹ ^ (1 + c.2.1.length) * rt d ^ 2 * rt d ^ c.2.1.length *
          (rt d)⁻¹ = ((rt d)⁻¹ * rt d) ^ (1 + c.2.1.length) * (rt d * (rt d)⁻¹) by ring,
          inv_mul_cancel₀ hr, mul_inv_cancel₀ hr, one_pow, one_mul]
      calc (rt d)⁻¹ ^ (1 + c.2.1.length) * ↑d *
            (l.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod *
            rt d ^ (c.2.1.length + (l.map fun c => c.2.1.length).sum) * (rt d)⁻¹ ^ (l.length + 1)
          = ((rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ) * rt d ^ c.2.1.length * (rt d)⁻¹) *
            ((l.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod *
              rt d ^ ((l.map fun c => c.2.1.length).sum) * (rt d)⁻¹ ^ l.length) := by ring
        _ = 1 := by rw [hf, ih, one_mul]
  calc vConst d ms * rt d ^ ((ms.map fun c => c.2.1.length).sum)
      = ((ms.map fun c => (rt d)⁻¹ ^ (1 + c.2.1.length) * (d : ℂ)).prod *
          rt d ^ ((ms.map fun c => c.2.1.length).sum) * (rt d)⁻¹ ^ ms.length) * rt d := by
        simp only [vConst]; ring
    _ = rt d := by rw [key, one_mul]

/-- [P1] (VW): a V box into a W node is the W node with a V box on each output. -/
theorem zxwVW (E : Env) (dz i a b mo u₁ pq₁ pz₁ px₁ p q u₂ pq₂ pz₂ px₂ u₃ pq₃ pz₃ px₃ : ℕ)
    (m₁ m₂ m₃ : Node) (ms₁ ms₂ ms₃ : List (ℕ × List ℕ × ℕ)) (iface : List ℕ)
    (hd : 0 < E.dimv dz) (hm₁ : isZc m₁ = true) (hm₂ : isZc m₂ = true) (hm₃ : isZc m₃ = true)
    (hdm₁ : m₁.dim = dz) (hdm₂ : m₂.dim = dz) (hdm₃ : m₃.dim = dz)
    (hl₁ : m₁.outs ++ m₁.ins = [u₁, i]) (hl₂ : m₂.outs ++ m₂.ins = [u₂, p])
    (hl₃ : m₃.outs ++ m₃.ins = [u₃, q])
    (hndL : (mo :: vLabs u₁ ms₁).Nodup) (hoL : ∀ x ∈ [i, a, b], x ∉ mo :: vLabs u₁ ms₁)
    (hndR : (p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃)).Nodup)
    (hoR : ∀ x ∈ [i, a, b], x ∉ p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃))
    (hlab : ∀ x ∈ mo :: p :: q :: (vLabs u₁ ms₁ ++ vLabs u₂ ms₂ ++ vLabs u₃ ms₃),
      labIdx E.labs x = dz)
    (hi : i ∈ iface ∧ labIdx E.labs i = dz) (ha : a ∈ iface ∧ labIdx E.labs a = dz)
    (hb : b ∈ iface ∧ labIdx E.labs b = dz)
    (hph : ∀ ph ∈ [pq₁, pz₁, px₁, pq₂, pz₂, px₂, pq₃, pz₃, px₃], ∀ k < E.dimv dz,
      ex (E.phase ph k) = 1)
    (hbox : ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E m₁ k = 0)
    (h0 : zc E m₁ 0 = 1)
    (hc₂ : ∀ k < E.dimv dz, zc E m₂ k = zc E m₁ k) (hc₃ : ∀ k < E.dimv dz, zc E m₃ k = zc E m₁ k)
    (hlen₂ : ms₂.map (fun c => c.2.1.length) = ms₁.map (fun c => c.2.1.length))
    (hlen₃ : ms₃.map (fun c => c.2.1.length) = ms₁.map (fun c => c.2.1.length))
    (hcount : ∀ x < E.dimv dz, (ms₁.filter fun c => decide (x = (c.2.1.length *
      (E.dimv dz - 1)) % E.dimv dz)).length = if x = 0 then 0 else 1)
    (hsum : 1 ≤ (ms₁.map fun c => c.2.1.length).sum)
    (s : ℂ) (hs : s = rt (E.dimv dz) ^ ((ms₁.map fun c => c.2.1.length).sum - 1)) :
    LocalEq E (vNodes dz u₁ mo pq₁ pz₁ px₁ m₁ ms₁ ++ [⟨.W, [a, b], [mo], dz, 0⟩])
      (mo :: vLabs u₁ ms₁)
      (⟨.W, [p, q], [i], dz, 0⟩ :: (vNodes dz u₂ a pq₂ pz₂ px₂ m₂ ms₂ ++
        vNodes dz u₃ b pq₃ pz₃ px₃ m₃ ms₃)) (p :: q :: (vLabs u₂ ms₂ ++ vLabs u₃ ms₃))
      iface 1 s := by
  intro τ hτ
  have h := zxwVW' E dz i a b mo u₁ pq₁ pz₁ px₁ p q u₂ pq₂ pz₂ px₂ u₃ pq₃ pz₃ px₃ m₁ m₂ m₃ ms₁
    ms₂ ms₃ iface hd hm₁ hm₂ hm₃ hdm₁ hdm₂ hdm₃ hl₁ hl₂ hl₃ hndL hoL hndR hoR hlab hi ha hb hph
    hbox h0 hc₂ hc₃ hlen₂ hlen₃ hcount _ rfl τ hτ
  have hc := vConst_pow (E.dimv dz) hd ms₁
  have hr := rt_ne (E.dimv dz) hd
  obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hsum
  rw [hn] at hc
  have this : vConst (E.dimv dz) ms₁ * rt (E.dimv dz) ^ n = 1 := by
    have h2 : (vConst (E.dimv dz) ms₁ * rt (E.dimv dz) ^ n) * rt (E.dimv dz) =
        1 * rt (E.dimv dz) := by
      calc (vConst (E.dimv dz) ms₁ * rt (E.dimv dz) ^ n) * rt (E.dimv dz)
          = vConst (E.dimv dz) ms₁ * rt (E.dimv dz) ^ (1 + n) := by ring
        _ = rt (E.dimv dz) := hc
        _ = 1 * rt (E.dimv dz) := (one_mul _).symm
    exact mul_right_cancel₀ hr h2
  rw [hs, hn, show 1 + n - 1 = n by omega]
  rw [one_mul] at h ⊢
  rw [← h]
  linear_combination (-(sumOver (labDim E) (mo :: vLabs u₁ ms₁)
    (prodVal E (vNodes dz u₁ mo pq₁ pz₁ px₁ m₁ ms₁ ++ [⟨.W, [a, b], [mo], dz, 0⟩])) τ)) * this


theorem sum_two_points (d : ℕ) (hd : 1 < d) (f : ℕ → ℂ)
    (hf : ∀ j < d, j ≠ 0 → j ≠ d - 1 → f j = 0) :
    ∑ j ∈ range d, f j = f 0 + f (d - 1) := by
  have e : ∀ j ∈ range d, f j = (if j = 0 then f 0 else 0) + (if j = d - 1 then f (d - 1) else 0) := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    by_cases h0 : j = 0
    · subst h0; simp [show (0 : ℕ) ≠ d - 1 by omega]
    · by_cases h1 : j = d - 1
      · subst h1; simp [h0]
      · simp [h0, h1, hf j hj' h0 h1]
  rw [Finset.sum_congr rfl e, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.sum_ite_eq',
    if_pos (Finset.mem_range.2 (by omega)), if_pos (Finset.mem_range.2 (by omega))]

/-- [P1] (ZV): a spider state `a` into a V box is the spider state of coefficients `1` at `0`
and `a_{d-1}` elsewhere. -/
theorem zxwZV (E : Env) (dz x u o pq pz px : ℕ) (mz m mr : Node) (ms : List (ℕ × List ℕ × ℕ))
    (iface : List ℕ) (hd : 1 < E.dimv dz) (hmz : isZc mz = true) (hm : isZc m = true)
    (hmr : isZc mr = true) (hdz : mz.dim = dz) (hdm : m.dim = dz) (hdr : mr.dim = dz)
    (hlz : mz.outs ++ mz.ins = [x]) (hl : m.outs ++ m.ins = [u, x]) (hlr : mr.outs ++ mr.ins = [o])
    (hnd : (x :: vLabs u ms).Nodup) (ho : o ∉ x :: vLabs u ms)
    (hlab : ∀ y ∈ x :: vLabs u ms, labIdx E.labs y = dz) (hol : o ∈ iface ∧ labIdx E.labs o = dz)
    (hph : ∀ ph ∈ [pq, pz, px], ∀ k < E.dimv dz, ex (E.phase ph k) = 1)
    (hbox : ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E m k = 0)
    (h0 : zc E m 0 = 1) (hz0 : zc E mz 0 = 1)
    (hcount : ∀ y < E.dimv dz, (ms.filter fun c => decide (y = (c.2.1.length *
      (E.dimv dz - 1)) % E.dimv dz)).length = if y = 0 then 0 else 1)
    (hr : ∀ k < E.dimv dz, zc E mr k = if k = 0 then 1 else zc E mz (E.dimv dz - 1))
    (hd1 : zc E m (E.dimv dz - 1) = 1)
    (hsum : 1 ≤ (ms.map fun c => c.2.1.length).sum)
    (s : ℂ) (hs : s = (rt (E.dimv dz))⁻¹ ^ ((ms.map fun c => c.2.1.length).sum - 1)) :
    LocalEq E (mz :: vNodes dz u o pq pz px m ms) (x :: vLabs u ms) [mr] [] iface 1 s := by
  have hvc : vConst (E.dimv dz) ms = (rt (E.dimv dz))⁻¹ ^
      ((ms.map fun c => c.2.1.length).sum - 1) := by
    have hc := vConst_pow (E.dimv dz) (by omega) ms
    have hr := rt_ne (E.dimv dz) (by omega)
    obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hsum
    rw [hn] at hc
    rw [hn, show 1 + n - 1 = n by omega, inv_pow]
    apply eq_inv_of_mul_eq_one_left
    apply mul_right_cancel₀ hr
    calc vConst (E.dimv dz) ms * rt (E.dimv dz) ^ n * rt (E.dimv dz)
        = vConst (E.dimv dz) ms * rt (E.dimv dz) ^ (1 + n) := by ring
      _ = rt (E.dimv dz) := hc
      _ = 1 * rt (E.dimv dz) := (one_mul _).symm
  rw [← hvc] at hs
  subst hs
  intro τ hτ
  have rol := hτ o hol.1
  rw [labDim_of E o dz hol.2] at rol
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨hq, hz, hx⟩ := hph'
  simp only [List.nodup_cons] at hnd
  obtain ⟨hxv, hnd'⟩ := hnd
  have hox : o ≠ x := fun h => ho (by rw [h]; exact List.mem_cons_self)
  have hov : o ∉ vLabs u ms := fun h => ho (List.mem_cons_of_mem _ h)
  simp only [one_mul, sumOver, labDim_of E x dz (hlab x List.mem_cons_self)]
  have step : ∀ j ∈ range (E.dimv dz), sumOver (labDim E) (vLabs u ms)
      (prodVal E (mz :: vNodes dz u o pq pz px m ms)) (Function.update τ x j) =
      zc E mz j * (zc E m j * vConst (E.dimv dz) ms *
        (if j = 0 then (if τ o = 0 then 1 else 0)
          else (ms.map fun c => if τ o = (c.2.1.length * j) % E.dimv dz then (1 : ℂ)
            else 0).sum)) := by
    intro j hj
    have hj' := Finset.mem_range.1 hj
    have split : prodVal E (mz :: vNodes dz u o pq pz px m ms) = fun ρ => val E ρ mz *
        prodVal E (vNodes dz u o pq pz px m ms) ρ := by
      funext ρ; simp only [prodVal, List.map_cons, List.prod_cons]
    rw [split, sumOver_const_mul _ _ [x] (fun ρ => val E ρ mz) _ (fun ρ ρ' h => by
        simp only; rw [val_zc _ _ mz hmz, val_zc _ _ mz hmz, hlz]
        simp only [List.map_cons, List.map_nil]; rw [h x (by simp)])
      (fun y hy hy' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
        subst hy; exact hxv hy')]
    rw [vBoxVal E dz x u o pq pz px m ms _ (by omega) hm hdm hl hnd' hxv hov
      (fun y hy => hlab y (List.mem_cons_of_mem _ hy))
      (by rw [Function.update_self]; exact hj') (by rw [Function.update_of_ne hox]; exact rol)
      hq hz hx]
    rw [val_zc _ _ mz hmz, hlz]
    simp only [List.map_cons, List.map_nil, Function.update_self, Function.update_of_ne hox]
    rw [tZw_at _ _ _ j (by simp), hdz]
    simp only [hj', List.mem_singleton, forall_eq, true_and, if_true]
  rw [Finset.sum_congr rfl step, sum_two_points _ hd _ (fun j hj h0' h1' => by
    rw [hbox j hj h0' h1']; ring)]
  have hne : E.dimv dz - 1 ≠ 0 := by omega
  rw [if_pos rfl, if_neg hne, sum_ind_count, hcount _ rol, hz0, h0, hd1]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
    sumOver]
  rw [val_zc _ _ mr hmr, hlr]
  simp only [List.map_cons, List.map_nil]
  rw [tZw_at _ _ _ (τ o) (by simp), hdr]
  simp only [rol, List.mem_singleton, forall_eq, true_and, if_true, hr _ rol]
  by_cases h : τ o = 0 <;> simp [h] <;> ring


/-- The nodes of a VA chain: the box `c.2.1` (from `c.1`) and the multiplier `c.2.2`. -/
def vaNodes (dz pz px : ℕ) (c : ℕ × Node × (ℕ × List ℕ × ℕ)) : List Node :=
  c.2.1 :: multNodes dz pz px c.2.2

theorem vaNodes_labels (dz pz px : ℕ) (c : ℕ × Node × (ℕ × List ℕ × ℕ))
    (hbl : c.2.1.outs ++ c.2.1.ins = [c.2.2.1, c.1]) :
    ∀ x ∈ labelsOf (vaNodes dz pz px c),
      x = c.2.2.1 ∨ x ∈ c.2.2.2.1 ∨ x = c.1 ∨ x = c.2.2.2.2 := by
  intro x hx
  simp only [labelsOf, vaNodes, multNodes, List.flatMap_cons, List.flatMap_nil, Node.labels, hbl,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.append_nil] at hx
  aesop

theorem perm_map_flatMap {α : Type*} (l : List α) (f : α → ℕ) (g : α → List ℕ) :
    (l.map f ++ l.flatMap g).Perm (l.flatMap fun a => f a :: g a) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.flatMap_cons, List.cons_append]
    refine List.Perm.cons _ ?_
    exact (List.perm_append_comm_assoc _ _ _).trans (List.Perm.append_left _ ih)

/-- Boxes on the outputs of a W node, then multipliers, then a pink dot, summed inside. -/
theorem vaVal (E : Env) (dz u o pq pz px : ℕ) (cs : List (ℕ × Node × (ℕ × List ℕ × ℕ)))
    (σ : ℕ → ℕ) (hd : 0 < E.dimv dz)
    (hbx : ∀ c ∈ cs, isZc c.2.1 = true ∧ c.2.1.dim = dz ∧ c.2.1.outs ++ c.2.1.ins = [c.2.2.1, c.1])
    (hnd : (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1)).Nodup)
    (hu : u ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (ho : o ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (hlab : ∀ x ∈ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1), labIdx E.labs x = dz)
    (ru : σ u < E.dimv dz) (ro : σ o < E.dimv dz)
    (hq : ∀ k < E.dimv dz, ex (E.phase pq k) = 1) (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    sumOver (labDim E) (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++
        cs.map (fun c => c.2.2.1) ++ cs.flatMap (fun c => c.2.2.2.1))
      (prodVal E (⟨.W, cs.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], cs.map (fun c => c.2.2.2.2), dz, pq⟩ ::
        cs.flatMap (vaNodes dz pz px))) σ =
      ∑ k ∈ range (E.dimv dz), fou (E.dimv dz) (σ o) k *
        (if σ u = 0 then (cs.map fun c => zc E c.2.1 0 * ((rt (E.dimv dz))⁻¹ ^
            (1 + c.2.2.2.1.length) * E.dimv dz) * (starRingEnd ℂ) (fou (E.dimv dz) 0 k)).prod
          else ((cs.map Prod.fst).map fun w => (cs.map fun c =>
            zc E c.2.1 (if c.1 = w then σ u else 0) * ((rt (E.dimv dz))⁻¹ ^
              (1 + c.2.2.2.1.length) * E.dimv dz) * (starRingEnd ℂ) (fou (E.dimv dz)
              ((c.2.2.2.1.length * if c.1 = w then σ u else 0) % E.dimv dz) k)).prod).sum) := by
  set d := E.dimv dz with hdd
  set ws := cs.map Prod.fst
  set vs := cs.map fun c => c.2.2.2.2
  set xs := cs.map fun c => c.2.2.1
  set wr := cs.flatMap fun c => c.2.2.2.1
  have hnd' := hnd
  rw [List.nodup_append, List.nodup_append] at hnd'
  obtain ⟨⟨hwv, hxsnd, hwvx⟩, hwr, hdis⟩ := hnd'
  have hD : ∀ x ∈ ws ++ vs ++ xs ++ wr, labDim E x = d := fun x hx => labDim_of E x dz (hlab x hx)
  rw [List.append_assoc (ws ++ vs) xs wr, sumOver_append]
  set g : ℕ × Node × (ℕ × List ℕ × ℕ) → ℕ → ℕ → ℂ := fun c x y =>
    zc E c.2.1 x * ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) *
      (if y % d = (c.2.2.2.1.length * x) % d then (d : ℂ) else 0))
  have hgrp : ∀ c ∈ cs, ∀ ρ : ℕ → ℕ, ρ c.1 < d → sumOver (labDim E) (c.2.2.1 :: c.2.2.2.1)
      (prodVal E (vaNodes dz pz px c)) ρ = g c (ρ c.1) (ρ c.2.2.2.2) := by
    intro c hc ρ hw
    obtain ⟨hbz, hbd, hbl⟩ := hbx c hc
    have hxw : c.2.2.1 ≠ c.1 := fun h => hwvx c.1 (List.mem_append_left _ (List.mem_map_of_mem hc))
      c.2.2.1 (List.mem_map_of_mem hc) h.symm
    have hxv : c.2.2.1 ≠ c.2.2.2.2 := fun h => hwvx c.2.2.2.2
      (List.mem_append_right _ (List.mem_map_of_mem hc)) c.2.2.1 (List.mem_map_of_mem hc) h.symm
    have hxn : c.2.2.1 ∉ c.2.2.2.1 := fun h => hdis c.2.2.1 (List.mem_append_right _
      (List.mem_map_of_mem hc)) c.2.2.1 (List.mem_flatMap.2 ⟨c, hc, h⟩) rfl
    have hwn : c.1 ∉ c.2.2.2.1 := fun h => hdis c.1 (List.mem_append_left _ (List.mem_append_left _
      (List.mem_map_of_mem hc))) c.1 (List.mem_flatMap.2 ⟨c, hc, h⟩) rfl
    have hvn : c.2.2.2.2 ∉ c.2.2.2.1 := fun h => hdis c.2.2.2.2 (List.mem_append_left _
      (List.mem_append_right _ (List.mem_map_of_mem hc))) c.2.2.2.2
      (List.mem_flatMap.2 ⟨c, hc, h⟩) rfl
    have hwr_c : c.2.2.2.1.Nodup := (List.nodup_flatMap.1 hwr).1 c hc
    simp only [sumOver, hD c.2.2.1 (by simp [xs, List.mem_map_of_mem hc])]
    have e : ∀ j ∈ range d, sumOver (labDim E) c.2.2.2.1 (prodVal E (vaNodes dz pz px c))
        (Function.update ρ c.2.2.1 j) = if j = ρ c.1 then g c (ρ c.1) (ρ c.2.2.2.2) else 0 := by
      intro j hj
      have hj' := Finset.mem_range.1 hj
      have split : prodVal E (vaNodes dz pz px c) = fun τ => val E τ c.2.1 *
          prodVal E [⟨.Z, c.2.2.2.1, [c.2.2.1], dz, pz⟩, ⟨.X, [c.2.2.2.2], c.2.2.2.1, dz, px⟩] τ := by
        funext τ; simp only [vaNodes, multNodes, prodVal, List.map_cons, List.prod_cons]
      rw [split, sumOver_const_mul _ _ [c.2.2.1, c.1] (fun τ => val E τ c.2.1) _ (fun τ τ' h => by
          simp only; rw [val_zc _ _ _ hbz, val_zc _ _ _ hbz, hbl]
          simp only [List.map_cons, List.map_nil]; rw [h c.2.2.1 (by simp), h c.1 (by simp)])
        (fun y hy hy' => by
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
          rcases hy with rfl | rfl
          · exact hxn hy'
          · exact hwn hy')]
      rw [multiplier_val E pz px dz _ _ _ (Function.update ρ c.2.2.1 j) hd hwr_c hxn hvn
        (fun y hy => hlab y (List.mem_append_right _ (List.mem_flatMap.2 ⟨c, hc, hy⟩)))
        (by rw [Function.update_self]; exact hj') hz hx]
      rw [val_zc_two E _ _ hbz c.2.2.1 c.1 hbl (by rw [Function.update_of_ne (Ne.symm hxw), hbd]; exact hw)]
      simp only [Function.update_self, Function.update_of_ne (Ne.symm hxw),
        Function.update_of_ne (Ne.symm hxv)]
      by_cases h : j = ρ c.1
      · subst h; simp only [if_true, g]; rfl
      · simp [h]
    rw [Finset.sum_congr rfl e, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hw)]
  have hperm := perm_map_flatMap cs (fun c => c.2.2.1) (fun c => c.2.2.2.1)
  have hnd2 : (cs.flatMap fun c => c.2.2.1 :: c.2.2.2.1).Nodup :=
    hperm.nodup_iff.1 (List.nodup_append.2 ⟨hxsnd, hwr, fun a ha b hb => hdis a
      (List.mem_append_right _ ha) b hb⟩)
  have hin : ∀ ρ : ℕ → ℕ, (∀ x ∉ ws ++ vs, ρ x = σ x) → (∀ x ∈ ws ++ vs, ρ x < labDim E x) →
      sumOver (labDim E) (xs ++ wr) (prodVal E (⟨.W, ws, [u], dz, 0⟩ :: ⟨.X, [o], vs, dz, pq⟩ ::
        cs.flatMap (vaNodes dz pz px))) ρ =
      tW d (cs.map fun c => ρ c.1) (ρ u) * (tX d (E.phase pq) [ρ o] (cs.map fun c => ρ c.2.2.2.2) *
        (cs.map fun c => g c (ρ c.1) (ρ c.2.2.2.2)).prod) := by
    intro ρ hρ hr
    have split : prodVal E (⟨.W, ws, [u], dz, 0⟩ :: ⟨.X, [o], vs, dz, pq⟩ ::
        cs.flatMap (vaNodes dz pz px)) = fun τ => (tW d (ws.map τ) (τ u) *
          tX d (E.phase pq) [τ o] (vs.map τ)) *
          (cs.map fun c => prodVal E (vaNodes dz pz px c) τ).prod := by
      funext τ
      simp only [prodVal, List.map_cons, List.prod_cons, prod_flatMap', val, List.map_nil]
      ring
    rw [split, sumOver_const_mul _ (xs ++ wr) (u :: o :: (ws ++ vs)) _ _ (fun τ τ' h => by
        have hw : ws.map τ = ws.map τ' := List.map_congr_left fun x hx => h x (by simp [hx])
        have hv : vs.map τ = vs.map τ' := List.map_congr_left fun x hx => h x (by simp [hx])
        rw [hw, hv, h u (by simp), h o (by simp)])
      (fun x hx hxw => by
        simp only [List.mem_cons] at hx
        rcases hx with rfl | rfl | hx
        · exact hu (by rcases List.mem_append.1 hxw with h | h <;> simp [h])
        · exact ho (by rcases List.mem_append.1 hxw with h | h <;> simp [h])
        · rcases List.mem_append.1 hxw with h | h
          · exact hwvx x hx x h rfl
          · exact hdis x (List.mem_append_left _ hx) x h rfl) ρ]
    rw [sumOver_perm _ hperm]
    have hg := sumOver_groups (D := labDim E) (cs.map fun c => (c.2.2.1 :: c.2.2.2.1,
      fun τ => prodVal E (vaNodes dz pz px c) τ)) (ws ++ vs)
    simp only [List.flatMap_map, List.map_map, Function.comp_def] at hg
    rw [hg hnd2 (fun x hx hx' => by
        obtain ⟨c, hc, hxc⟩ := List.mem_flatMap.1 hx'
        rcases List.mem_cons.1 hxc with h | h
        · exact hwvx x hx x (h ▸ List.mem_map_of_mem hc) rfl
        · exact hdis x (List.mem_append_left _ hx) x (List.mem_flatMap.2 ⟨c, hc, h⟩) rfl)
      (fun gg hgg => by
        obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hgg
        intro τ τ' h
        refine prodVal_reads E (vaNodes dz pz px c) τ τ' fun x hx => h x ?_
        obtain ⟨-, -, hbl⟩ := hbx c hc
        rcases vaNodes_labels dz pz px c hbl x hx with h1 | h1 | h1 | h1
        · exact List.mem_append_left _ (h1 ▸ List.mem_cons_self)
        · exact List.mem_append_left _ (List.mem_cons_of_mem _ h1)
        · exact List.mem_append_right _ (List.mem_append_left _ (h1 ▸ List.mem_map_of_mem hc))
        · exact List.mem_append_right _ (List.mem_append_right _
            (h1 ▸ List.mem_map_of_mem (f := fun c => c.2.2.2.2) hc))) ρ]
    rw [List.map_congr_left fun c hc => hgrp c hc ρ (by
      have := hr c.1 (List.mem_append_left _ (List.mem_map_of_mem hc))
      rwa [hD c.1 (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
        (List.mem_map_of_mem hc))))] at this)]
    simp only [ws, vs, List.map_map, Function.comp_def]
    ring
  rw [sumOver_congr _ _ _ _ _ fun ρ hρ hr => hin ρ hρ hr]
  rw [wPinkSum (D := labDim E) d hd u o cs Prod.fst (fun c => c.2.2.2.2) g (E.phase pq) hq σ hwv
    (fun x hx => hD x (List.mem_append_left _ (List.mem_append_left _ hx)))
    (fun h => hu (List.mem_append_left _ (List.mem_append_left _ h)))
    (fun h => ho (List.mem_append_left _ (List.mem_append_left _ h))) ru]
  have hGk : ∀ (c : ℕ × Node × (ℕ × List ℕ × ℕ)) (x k : ℕ), (∑ y ∈ range d,
      (starRingEnd ℂ) (fou d y k) * g c x y) = zc E c.2.1 x * ((rt d)⁻¹ ^
        (1 + c.2.2.2.1.length) * d) * (starRingEnd ℂ) (fou d ((c.2.2.2.1.length * x) % d) k) := by
    intro c x k
    have e : ∀ y ∈ range d, (starRingEnd ℂ) (fou d y k) * g c x y =
        if y = (c.2.2.2.1.length * x) % d then zc E c.2.1 x * ((rt d)⁻¹ ^
          (1 + c.2.2.2.1.length) * d) * (starRingEnd ℂ) (fou d y k) else 0 := by
      intro y hy
      simp only [g, Nat.mod_eq_of_lt (Finset.mem_range.1 hy)]
      split_ifs <;> ring
    rw [Finset.sum_congr rfl e, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd))]
  simp only [hGk, mul_zero, Nat.zero_mod]
  rfl

theorem vaClosed (E : Env) (dz u o pq pz px : ℕ) (cs : List (ℕ × Node × (ℕ × List ℕ × ℕ)))
    (σ : ℕ → ℕ) (hd : 0 < E.dimv dz)
    (hbx : ∀ c ∈ cs, isZc c.2.1 = true ∧ c.2.1.dim = dz ∧ c.2.1.outs ++ c.2.1.ins = [c.2.2.1, c.1])
    (hnd : (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1)).Nodup)
    (hu : u ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (ho : o ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (hlab : ∀ x ∈ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1), labIdx E.labs x = dz)
    (ru : σ u < E.dimv dz) (ro : σ o < E.dimv dz)
    (hq : ∀ k < E.dimv dz, ex (E.phase pq k) = 1) (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) (hb0 : ∀ c ∈ cs, zc E c.2.1 0 = 1) :
    sumOver (labDim E) (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++
        cs.map (fun c => c.2.2.1) ++ cs.flatMap (fun c => c.2.2.2.1))
      (prodVal E (⟨.W, cs.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], cs.map (fun c => c.2.2.2.2), dz, pq⟩ ::
        cs.flatMap (vaNodes dz pz px))) σ =
      vConst (E.dimv dz) (cs.map fun c => c.2.2) *
        (if σ u = 0 then (if σ o = 0 then 1 else 0)
          else (cs.map fun c => zc E c.2.1 (σ u) * (if σ o = (c.2.2.2.1.length * σ u) %
            E.dimv dz then (1 : ℂ) else 0)).sum) := by
  rw [vaVal E dz u o pq pz px cs σ hd hbx hnd hu ho hlab ru ro hq hz hx]
  set d := E.dimv dz with hdd
  have hc0 : ∀ k, (starRingEnd ℂ) (fou d 0 k) = (rt d)⁻¹ := by
    intro k; rw [fou_zero']; simp [rt]
  set A := (cs.map fun c => (rt d)⁻¹ ^ (1 + c.2.2.2.1.length) * (d : ℂ)).prod
  have hvc : vConst d (cs.map fun c => c.2.2) = A * ((rt d)⁻¹ ^ cs.length * rt d) := by
    simp only [vConst, List.map_map, Function.comp_def, List.length_map, A]
  rw [hvc]
  have hwv := (List.nodup_append.1 (List.nodup_append.1 (List.nodup_append.1 hnd).1).1).1
  by_cases hu0 : σ u = 0
  · simp only [hu0, if_true]
    have h0 : (cs.map fun c => zc E c.2.1 0 * ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) * (d : ℂ)) *
        (starRingEnd ℂ) (fou d 0 0)).prod = A * (rt d)⁻¹ ^ cs.length := by
      rw [List.map_congr_left fun c hc => by rw [hb0 c hc, one_mul, hc0], List.prod_map_mul,
        List.map_const', List.prod_replicate]
    have h0' : ∀ k, (cs.map fun c => zc E c.2.1 0 * ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) *
        (d : ℂ)) * (starRingEnd ℂ) (fou d 0 k)).prod = A * (rt d)⁻¹ ^ cs.length := by
      intro k
      rw [List.map_congr_left fun c hc => by rw [hb0 c hc, one_mul, hc0], List.prod_map_mul,
        List.map_const', List.prod_replicate]
    simp only [h0']
    rw [← Finset.sum_mul, sum_fou_left d hd, Nat.mod_eq_of_lt ro]
    by_cases ho0 : σ o = 0
    · simp only [ho0, if_true, mul_one]; rw [rt_inv_mul d hd]; ring
    · simp [ho0]
  · simp only [hu0, if_false]
    have hws : (cs.map Prod.fst).Nodup := hwv
    have hone : ∀ k, ∀ c' ∈ cs, (cs.map fun c => zc E c.2.1 (if c.1 = c'.1 then σ u else 0) *
        ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) * d) * (starRingEnd ℂ) (fou d
          ((c.2.2.2.1.length * if c.1 = c'.1 then σ u else 0) % d) k)).prod =
        A * ((zc E c'.2.1 (σ u) * (starRingEnd ℂ) (fou d ((c'.2.2.2.1.length * σ u) % d) k)) *
          (rt d)⁻¹ ^ (cs.length - 1)) := by
      intro k c' hc'
      rw [show (cs.map fun c => zc E c.2.1 (if c.1 = c'.1 then σ u else 0) *
          ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) * d) * (starRingEnd ℂ) (fou d
            ((c.2.2.2.1.length * if c.1 = c'.1 then σ u else 0) % d) k)) =
          cs.map fun c => ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) * d) *
            (zc E c.2.1 (if c.1 = c'.1 then σ u else 0) * (starRingEnd ℂ) (fou d
              ((c.2.2.2.1.length * if c.1 = c'.1 then σ u else 0) % d) k)) from
        List.map_congr_left fun c _ => by ring, List.prod_map_mul]
      congr 1
      rw [prod_one_hot cs Prod.fst hws c' hc' _ ((rt d)⁻¹) (fun b hb hbn => by
        simp only [if_neg hbn, mul_zero, Nat.zero_mod, hc0, hb0 b hb, one_mul])]
      simp
    simp only [List.map_map, Function.comp_def]
    have hk : ∀ k ∈ range d, fou d (σ o) k * (cs.map fun c' => (cs.map fun c =>
        zc E c.2.1 (if c.1 = c'.1 then σ u else 0) * ((rt d)⁻¹ ^ (1 + c.2.2.2.1.length) * d) *
          (starRingEnd ℂ) (fou d ((c.2.2.2.1.length * if c.1 = c'.1 then σ u else 0) % d)
            k)).prod).sum =
        (cs.map fun c' => A * (rt d)⁻¹ ^ (cs.length - 1) * zc E c'.2.1 (σ u) *
          (fou d (σ o) k * (starRingEnd ℂ) (fou d ((c'.2.2.2.1.length * σ u) % d) k))).sum := by
      intro k _
      rw [← List.sum_map_mul_left]
      refine congrArg List.sum (List.map_congr_left fun c' hc' => ?_)
      rw [hone k c' hc']
      ring
    rw [Finset.sum_congr rfl hk, sum_list_comm]
    have hu' : ∀ c' ∈ cs, (∑ k ∈ range d, A * (rt d)⁻¹ ^ (cs.length - 1) * zc E c'.2.1 (σ u) *
        (fou d (σ o) k * (starRingEnd ℂ) (fou d ((c'.2.2.2.1.length * σ u) % d) k))) =
        A * (rt d)⁻¹ ^ (cs.length - 1) * (zc E c'.2.1 (σ u) *
          (if σ o = (c'.2.2.2.1.length * σ u) % d then 1 else 0)) := by
      intro c' _
      rw [← Finset.mul_sum, mul_assoc]
      congr 1
      congr 1
      rw [Finset.sum_congr rfl fun k _ => by rw [fou_symm d (σ o) k,
        fou_symm d ((c'.2.2.2.1.length * σ u) % d) k],
        fou_unitary d hd _ _ ro (Nat.mod_lt _ hd)]
    rw [List.map_congr_left hu', List.sum_map_mul_left]
    by_cases h : cs = []
    · subst h; simp
    · have hl : 1 ≤ cs.length := List.length_pos_iff.2 h
      obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hl
      have hr := rt_ne d hd
      rw [hn, show 1 + n - 1 = n by omega, pow_add, pow_one]
      field_simp

/-- [P1] (VA): chain boxes `b_c` on a W node through multipliers into a pink dot are a V box
followed by the Z box `A`. -/
theorem zxwVA (E : Env) (dz u o pq pz px u₂ m₂ pq₂ pz₂ px₂ : ℕ)
    (cs : List (ℕ × Node × (ℕ × List ℕ × ℕ))) (box A : Node) (ms : List (ℕ × List ℕ × ℕ))
    (iface : List ℕ) (hd : 1 < E.dimv dz)
    (hbx : ∀ c ∈ cs, isZc c.2.1 = true ∧ c.2.1.dim = dz ∧ c.2.1.outs ++ c.2.1.ins = [c.2.2.1, c.1])
    (hnd : (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1)).Nodup)
    (hu : u ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (ho : o ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (hlab : ∀ x ∈ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1), labIdx E.labs x = dz)
    (hui : u ∈ iface ∧ labIdx E.labs u = dz) (hoi : o ∈ iface ∧ labIdx E.labs o = dz)
    (hm : isZc box = true) (hdm : box.dim = dz) (hlm : box.outs ++ box.ins = [u₂, u])
    (hA : isZc A = true) (hdA : A.dim = dz) (hlA : A.outs ++ A.ins = [o, m₂])
    (hndR : (m₂ :: vLabs u₂ ms).Nodup) (huR : u ∉ m₂ :: vLabs u₂ ms) (hoR : o ∉ m₂ :: vLabs u₂ ms)
    (hlabR : ∀ x ∈ m₂ :: vLabs u₂ ms, labIdx E.labs x = dz)
    (hph : ∀ ph ∈ [pq, pz, px, pq₂, pz₂, px₂], ∀ k < E.dimv dz, ex (E.phase ph k) = 1)
    (hbox : ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E box k = 0)
    (h0 : zc E box 0 = 1) (h1 : zc E box (E.dimv dz - 1) = 1)
    (hb0 : ∀ c ∈ cs, zc E c.2.1 0 = 1)
    (hbc : ∀ c ∈ cs, ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E c.2.1 k = 0)
    (hA0 : zc E A 0 = 1)
    (hAc : ∀ c ∈ cs, zc E c.2.1 (E.dimv dz - 1) =
      zc E A ((c.2.2.2.1.length * (E.dimv dz - 1)) % E.dimv dz))
    (hlen : ms.map (fun c => c.2.1.length) = cs.map (fun c => c.2.2.2.1.length)) :
    LocalEq E (⟨.W, cs.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], cs.map (fun c => c.2.2.2.2), dz, pq⟩ ::
        cs.flatMap (vaNodes dz pz px))
      (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
        cs.flatMap (fun c => c.2.2.2.1))
      (vNodes dz u₂ m₂ pq₂ pz₂ px₂ box ms ++ [A]) (m₂ :: vLabs u₂ ms) iface 1 1 := by
  intro τ hτ
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  have ru := hτ u hui.1
  have ro := hτ o hoi.1
  rw [labDim_of E u dz hui.2] at ru
  rw [labDim_of E o dz hoi.2] at ro
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨hq, hz, hx, hq₂, hz₂, hx₂⟩ := hph'
  rw [one_mul, one_mul, vaClosed E dz u o pq pz px cs τ hd' (fun c hc => hbx c hc) hnd hu ho hlab
    ru ro hq hz hx hb0, ← hdd]
  have nu : u ≠ m₂ := fun h => huR (by rw [h]; exact List.mem_cons_self)
  have no : o ≠ m₂ := fun h => hoR (by rw [h]; exact List.mem_cons_self)
  simp only [List.nodup_cons] at hndR
  obtain ⟨hm₂, hnd₂⟩ := hndR
  have hR : sumOver (labDim E) (m₂ :: vLabs u₂ ms) (prodVal E (vNodes dz u₂ m₂ pq₂ pz₂ px₂ box ms ++
      [A])) τ = ∑ j ∈ range d, (if τ o = j then zc E A j else 0) * (zc E box (τ u) * vConst d ms *
        (if τ u = 0 then (if j = 0 then 1 else 0)
          else (ms.map fun c => if j = (c.2.1.length * τ u) % d then (1 : ℂ) else 0).sum)) := by
    simp only [sumOver, labDim_of E m₂ dz (hlabR m₂ (by simp))]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hj' := Finset.mem_range.1 hj
    have split : prodVal E (vNodes dz u₂ m₂ pq₂ pz₂ px₂ box ms ++ [A]) =
        fun ρ => val E ρ A * prodVal E (vNodes dz u₂ m₂ pq₂ pz₂ px₂ box ms) ρ := by
      funext ρ
      simp only [prodVal, List.map_append, List.prod_append, List.map_cons, List.map_nil,
        List.prod_cons, List.prod_nil, mul_one]
      ring
    rw [split, sumOver_const_mul _ _ [o, m₂] (fun ρ => val E ρ A) _
      (fun ρ ρ' h => by
        simp only
        rw [val_zc _ _ A hA, val_zc _ _ A hA, hlA]
        simp only [List.map_cons, List.map_nil]
        rw [h o (by simp), h m₂ (by simp)])
      (fun x hx hx' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases hx with rfl | rfl
        · exact hoR (List.mem_cons_of_mem _ hx')
        · exact hm₂ hx')]
    rw [vBoxVal E dz u u₂ m₂ pq₂ pz₂ px₂ box ms _ hd' hm hdm hlm hnd₂
      (fun h => huR (List.mem_cons_of_mem _ h)) hm₂
      (fun x hx => hlabR x (List.mem_cons_of_mem _ hx))
      (by rw [Function.update_of_ne nu]; exact ru)
      (by rw [Function.update_self]; exact hj') hq₂ hz₂ hx₂]
    rw [val_zc_two E _ A hA o m₂ hlA (by rw [Function.update_self, hdA]; exact hj')]
    simp only [Function.update_self, Function.update_of_ne nu, Function.update_of_ne no]
    rfl
  rw [hR]
  simp only [ite_mul, zero_mul]
  rw [Finset.sum_ite_eq (range d) (τ o), if_pos (Finset.mem_range.2 ro)]
  have hvc : vConst d ms = vConst d (cs.map fun c => c.2.2) :=
    vConst_congr d _ _ (by rw [hlen, List.map_map]; rfl)
  rw [hvc]
  by_cases hu0 : τ u = 0
  · simp only [hu0, if_true, h0]
    by_cases ho0 : τ o = 0
    · simp [ho0, hA0]
    · simp [ho0]
  · simp only [hu0, if_false]
    have e1 : ∀ (l : List (ℕ × List ℕ × ℕ)) (g : ℕ → ℂ), (l.map fun c => g c.2.1.length) =
        (l.map fun c => c.2.1.length).map g := fun l g => by rw [List.map_map]; rfl
    by_cases hu1 : τ u = d - 1
    · rw [hu1, h1, one_mul]
      have hl : (cs.map fun c => zc E c.2.1 (d - 1) * (if τ o = (c.2.2.2.1.length * (d - 1)) % d
          then (1 : ℂ) else 0)) = cs.map fun c => zc E A (τ o) *
            (if τ o = (c.2.2.2.1.length * (d - 1)) % d then (1 : ℂ) else 0) :=
        List.map_congr_left fun c hc => by
          rw [hAc c hc]
          split_ifs with h
          · rw [h]
          · simp
      have hm : (ms.map fun c => if τ o = (c.2.1.length * (d - 1)) % d then (1 : ℂ) else 0) =
          cs.map fun c => if τ o = (c.2.2.2.1.length * (d - 1)) % d then (1 : ℂ) else 0 := by
        rw [e1 ms (fun n => if τ o = (n * (d - 1)) % d then (1 : ℂ) else 0), hlen, List.map_map]
        rfl
      rw [hl, List.sum_map_mul_left, hm]
      ring
    · have hz0 : zc E box (τ u) = 0 := hbox _ ru hu0 hu1
      rw [hz0]
      simp only [zero_mul, mul_zero]
      rw [show (cs.map fun c => zc E c.2.1 (τ u) * (if τ o = (c.2.2.2.1.length * τ u) % d
          then (1 : ℂ) else 0)) = cs.map fun _ => (0 : ℂ) from
        List.map_congr_left fun c hc => by rw [hbc c hc _ ru hu0 hu1, zero_mul]]
      simp


theorem wGreen (d a b : ℕ) (hb : b < d) :
    ∑ g ∈ range d, tW d [a, g] b = if a = 0 then 1 else if a = b then 1 else 0 := by
  by_cases h3 : b = 0
  · subst h3
    by_cases ha : a = 0
    · subst ha; simp [tW]; omega
    · simp [tW, ha]
  · by_cases ha : a = 0
    · subst ha
      have e : ∀ x : ℕ, ((if x = 0 then [] else [x]) = [b]) ↔ x = b := by
        intro x
        split_ifs with h
        · subst h; simp; omega
        · simp
      simp [tW, h3, hb, List.filter_cons, e]
    · simp [tW, h3, hb, ha, List.filter_cons]
      by_cases hab : a = b
      · subst hab; simp [show 0 < d by omega]
      · simp [hab]

theorem sum_pin_shift (d c b : ℕ) (hd : 0 < d) (hb : b < d) (f : ℕ → ℂ) :
    ∑ j ∈ range d, (if (j + c) % d = b then f j else 0) = f ((b + (d - c % d)) % d) := by
  rw [Finset.sum_congr rfl fun j hj =>
    (show _ = (if j = (b + (d - c % d)) % d then f j else 0) from
      if_congr (shift_iff d c j b hd (Finset.mem_range.1 hj) hb) rfl rfl)]
  rw [Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd))]

/-- The box `b`, pink `K_jm`, a W transpose on a plain green state, pink `K_1` and the box `A`
in series. -/
theorem qfVaR (E : Env) (dz u o x₁ x₂ x₃ x₄ g pm pg pp jm : ℕ) (b A : Node) (τ : ℕ → ℕ)
    (hd : 1 < E.dimv dz) (hb : isZc b = true) (hlb : b.outs ++ b.ins = [x₁, u])
    (hA : isZc A = true) (hdA : A.dim = dz) (hlA : A.outs ++ A.ins = [o, x₄])
    (hdb : b.dim = dz)
    (hnd : [x₁, x₂, x₃, x₄, g].Nodup) (hu : u ∉ [x₁, x₂, x₃, x₄, g]) (ho : o ∉ [x₁, x₂, x₃, x₄, g])
    (hlab : ∀ x ∈ [x₁, x₂, x₃, x₄, g], labIdx E.labs x = dz)
    (ru : τ u < E.dimv dz) (ro : τ o < E.dimv dz)
    (hpm : ∀ q < E.dimv dz, ex (E.phase pm q) = om (E.dimv dz) ^ (q * jm))
    (hjm : (jm + 1) % E.dimv dz = 0)
    (hpp : ∀ q < E.dimv dz, ex (E.phase pp q) = om (E.dimv dz) ^ (q * 1))
    (hpg : ∀ q < E.dimv dz, ex (E.phase pg q) = 1) :
    sumOver (labDim E) [x₁, x₂, x₃, x₄, g] (prodVal E [b, ⟨.X, [x₂], [x₁], dz, pm⟩,
        ⟨.W, [x₂, g], [x₃], dz, 0⟩, ⟨.Z, [g], [], dz, pg⟩, ⟨.X, [x₄], [x₃], dz, pp⟩, A]) τ =
      zc E b (τ u) * zc E A (τ o) * (if (τ u + 1) % E.dimv dz = 0 then 1
        else if (τ u + 1) % E.dimv dz = (τ o + 1) % E.dimv dz then 1 else 0) := by
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or,
    List.nodup_nil, and_true] at hnd hu ho
  obtain ⟨⟨n12, n13, n14, n1g⟩, ⟨n23, n24, n2g⟩, ⟨n34, n3g⟩, n4g, -⟩ := hnd
  obtain ⟨u1, u2, u3, u4, ug⟩ := hu
  obtain ⟨o1, o2, o3, o4, og⟩ := ho
  have hcan : sumOver (labDim E) [x₁, x₂, x₃, x₄, g] (prodVal E [b, ⟨.X, [x₂], [x₁], dz, pm⟩,
      ⟨.W, [x₂, g], [x₃], dz, 0⟩, ⟨.Z, [g], [], dz, pg⟩, ⟨.X, [x₄], [x₃], dz, pp⟩, A]) τ =
      ∑ j₁ ∈ range d, ∑ j₂ ∈ range d, ∑ j₃ ∈ range d, ∑ j₄ ∈ range d, ∑ jg ∈ range d,
        (if j₁ = τ u then zc E b (τ u) else 0) * ((if (j₂ + jm) % d = j₁ then 1 else 0) *
          (tW d [j₂, jg] j₃ * ((if (j₄ + 1) % d = j₃ then 1 else 0) *
            (if τ o = j₄ then zc E A j₄ else 0)))) := by
    simp only [sumOver, labDim_of E x₁ dz (hlab x₁ (by simp)), labDim_of E x₂ dz (hlab x₂ (by simp)),
      labDim_of E x₃ dz (hlab x₃ (by simp)), labDim_of E x₄ dz (hlab x₄ (by simp)),
      labDim_of E g dz (hlab g (by simp))]
    refine Finset.sum_congr rfl fun j₁ h₁ => Finset.sum_congr rfl fun j₂ h₂ =>
      Finset.sum_congr rfl fun j₃ h₃ => Finset.sum_congr rfl fun j₄ h₄ =>
        Finset.sum_congr rfl fun jg hg => ?_
    have r₁ := Finset.mem_range.1 h₁
    have r₄ := Finset.mem_range.1 h₄
    have r₃ := Finset.mem_range.1 h₃
    have rg := Finset.mem_range.1 hg
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      val_zc E _ b hb, val_zc E _ A hA, hlb, hlA]
    simp only [val, List.map_cons, List.map_nil, Function.update_apply,
      n12, n13, n14, n1g, n23, n24, n2g, n34, n3g, n4g, Ne.symm n12, Ne.symm n13, Ne.symm n14,
      Ne.symm n1g, Ne.symm n23, Ne.symm n24, Ne.symm n2g, Ne.symm n34, Ne.symm n3g, Ne.symm n4g,
      u1, u2, u3, u4, ug, o1, o2, o3, o4, og, if_true, if_false, hdb, hdA, List.append_nil]
    rw [← hdd]
    rw [tX_shift d jm _ _ hd' r₁ _ hpm, tX_shift d 1 _ _ hd' r₃ _ hpp,
      tZ_at d _ [jg] jg (by simp), if_pos (show jg < d ∧ ∀ v ∈ [jg], v = jg from ⟨rg, by simp⟩),
      hpg _ rg, one_mul, tZw_at _ _ _ (τ u) (by simp), tZw_at _ _ _ j₄ (by simp)]
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, and_true,
      ru, r₄, true_and]
  rw [hcan]
  simp only [← Finset.mul_sum, ← Finset.sum_mul]
  have pinA : ∀ j₃, ∑ j₄ ∈ range d, (if (j₄ + 1) % d = j₃ then (1 : ℂ) else 0) *
      (if τ o = j₄ then zc E A j₄ else 0) = if (τ o + 1) % d = j₃ then zc E A (τ o) else 0 := by
    intro j₃
    simp only [mul_ite, mul_zero]
    rw [Finset.sum_ite_eq (range d) (τ o), if_pos (Finset.mem_range.2 ro), ite_mul, one_mul,
      zero_mul]
  simp only [pinA]
  have pin3 : ∀ j₂, ∑ j₃ ∈ range d, (∑ jg ∈ range d, tW d [j₂, jg] j₃) *
      (if (τ o + 1) % d = j₃ then zc E A (τ o) else 0) =
      (if j₂ = 0 then 1 else if j₂ = (τ o + 1) % d then 1 else 0) * zc E A (τ o) := by
    intro j₂
    simp only [mul_ite, mul_zero]
    rw [Finset.sum_ite_eq (range d) ((τ o + 1) % d), if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd')),
      wGreen d j₂ _ (Nat.mod_lt _ hd')]
  simp only [pin3]
  simp only [ite_mul, zero_mul, one_mul]
  rw [Finset.sum_ite_eq' (range d) (τ u), if_pos (Finset.mem_range.2 ru),
    sum_pin_shift d jm (τ u) hd' ru]
  have hjm' : d - jm % d = 1 := by
    have h1 := Nat.mod_lt jm hd'
    have h2 : (jm % d + 1) % d = 0 := by
      rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]; exact hjm
    by_cases h3 : jm % d + 1 < d
    · rw [Nat.mod_eq_of_lt h3] at h2; omega
    · omega
  rw [hjm']
  split_ifs <;> ring


/-- [P2] (VA): boxes on a W node through multipliers into a pink dot are the box `b`, pink
`K_{-1}`, a W transpose on a plain green state, pink `K_1` and the box `A`. -/
theorem qfVA (E : Env) (dz u o pq pz px x₁ x₂ x₃ x₄ g pm pg pp jm : ℕ)
    (cs : List (ℕ × Node × (ℕ × List ℕ × ℕ))) (b A : Node) (iface : List ℕ)
    (hd : 1 < E.dimv dz)
    (hbx : ∀ c ∈ cs, isZc c.2.1 = true ∧ c.2.1.dim = dz ∧ c.2.1.outs ++ c.2.1.ins = [c.2.2.1, c.1])
    (hnd : (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1)).Nodup)
    (hu : u ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (ho : o ∉ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1))
    (hlab : ∀ x ∈ cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
      cs.flatMap (fun c => c.2.2.2.1), labIdx E.labs x = dz)
    (hui : u ∈ iface ∧ labIdx E.labs u = dz) (hoi : o ∈ iface ∧ labIdx E.labs o = dz)
    (hb : isZc b = true) (hdb : b.dim = dz) (hlb : b.outs ++ b.ins = [x₁, u])
    (hA : isZc A = true) (hdA : A.dim = dz) (hlA : A.outs ++ A.ins = [o, x₄])
    (hndR : [x₁, x₂, x₃, x₄, g].Nodup) (huR : u ∉ [x₁, x₂, x₃, x₄, g])
    (hoR : o ∉ [x₁, x₂, x₃, x₄, g]) (hlabR : ∀ x ∈ [x₁, x₂, x₃, x₄, g], labIdx E.labs x = dz)
    (hph : ∀ ph ∈ [pq, pz, px, pg], ∀ k < E.dimv dz, ex (E.phase ph k) = 1)
    (hpm : ∀ q < E.dimv dz, ex (E.phase pm q) = om (E.dimv dz) ^ (q * jm))
    (hjm : (jm + 1) % E.dimv dz = 0)
    (hpp : ∀ q < E.dimv dz, ex (E.phase pp q) = om (E.dimv dz) ^ (q * 1))
    (hbox : ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E b k = 0)
    (h0 : zc E b 0 = 1) (h1 : zc E b (E.dimv dz - 1) = 1)
    (hb0 : ∀ c ∈ cs, zc E c.2.1 0 = 1)
    (hbc : ∀ c ∈ cs, ∀ k < E.dimv dz, k ≠ 0 → k ≠ E.dimv dz - 1 → zc E c.2.1 k = 0)
    (hA0 : zc E A 0 = 1)
    (hAc : ∀ c ∈ cs, zc E c.2.1 (E.dimv dz - 1) =
      zc E A ((c.2.2.2.1.length * (E.dimv dz - 1)) % E.dimv dz))
    (hcount : ∀ x < E.dimv dz, (cs.filter fun c => decide (x = (c.2.2.2.1.length *
      (E.dimv dz - 1)) % E.dimv dz)).length = 1)
    (hsum : 1 ≤ (cs.map fun c => c.2.2.2.1.length).sum)
    (s : ℂ) (hs : s = (rt (E.dimv dz))⁻¹ ^ ((cs.map fun c => c.2.2.2.1.length).sum - 1)) :
    LocalEq E (⟨.W, cs.map Prod.fst, [u], dz, 0⟩ :: ⟨.X, [o], cs.map (fun c => c.2.2.2.2), dz, pq⟩ ::
        cs.flatMap (vaNodes dz pz px))
      (cs.map Prod.fst ++ cs.map (fun c => c.2.2.2.2) ++ cs.map (fun c => c.2.2.1) ++
        cs.flatMap (fun c => c.2.2.2.1))
      [b, ⟨.X, [x₂], [x₁], dz, pm⟩, ⟨.W, [x₂, g], [x₃], dz, 0⟩, ⟨.Z, [g], [], dz, pg⟩,
        ⟨.X, [x₄], [x₃], dz, pp⟩, A] [x₁, x₂, x₃, x₄, g] iface 1 s := by
  intro τ hτ
  have ru := hτ u hui.1
  have ro := hτ o hoi.1
  rw [labDim_of E u dz hui.2] at ru
  rw [labDim_of E o dz hoi.2] at ro
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨hq, hz, hx, hg⟩ := hph'
  rw [one_mul, vaClosed E dz u o pq pz px cs τ (by omega) (fun c hc => hbx c hc) hnd hu ho hlab
    ru ro hq hz hx hb0, qfVaR E dz u o x₁ x₂ x₃ x₄ g pm pg pp jm b A τ hd hb hlb hA hdA hlA hdb
    hndR huR hoR hlabR ru ro hpm hjm hpp hg]
  set d := E.dimv dz with hdd
  have hvc : vConst d (cs.map fun c => c.2.2) = s := by
    have hc := vConst_pow d (by omega) (cs.map fun c => c.2.2)
    have hr := rt_ne d (by omega)
    rw [List.map_map] at hc
    have e : (cs.map ((fun c : ℕ × List ℕ × ℕ => c.2.1.length) ∘ fun c => c.2.2)) =
        cs.map fun c => c.2.2.2.1.length := rfl
    rw [e] at hc
    obtain ⟨n, hn⟩ := Nat.exists_eq_add_of_le hsum
    rw [hn] at hc
    rw [hs, hn, show 1 + n - 1 = n by omega, inv_pow]
    apply eq_inv_of_mul_eq_one_left
    apply mul_right_cancel₀ hr
    calc vConst d (cs.map fun c => c.2.2) * rt d ^ n * rt d
        = vConst d (cs.map fun c => c.2.2) * rt d ^ (1 + n) := by ring
      _ = rt d := hc
      _ = 1 * rt d := (one_mul _).symm
  rw [hvc]
  have h1d : 1 % d = 1 := Nat.mod_eq_of_lt hd
  by_cases hu0 : τ u = 0
  · simp only [hu0, if_true, h0, zero_add, h1d, one_ne_zero, if_false, one_mul]
    by_cases ho0 : τ o = 0
    · simp [ho0, hA0, h1d]
    · have : ¬ (1 = (τ o + 1) % d) := by
        intro h
        by_cases h' : τ o + 1 < d
        · rw [Nat.mod_eq_of_lt h'] at h; omega
        · have : τ o + 1 = d := by omega
          rw [this, Nat.mod_self] at h; omega
      simp [ho0, this]
  · simp only [hu0, if_false]
    by_cases hu1 : τ u = d - 1
    · rw [hu1, h1, one_mul, show d - 1 + 1 = d by omega, Nat.mod_self, if_pos rfl, mul_one]
      have hl : (cs.map fun c => zc E c.2.1 (d - 1) * (if τ o = (c.2.2.2.1.length * (d - 1)) % d
          then (1 : ℂ) else 0)) = cs.map fun c => zc E A (τ o) *
            (if τ o = (c.2.2.2.1.length * (d - 1)) % d then (1 : ℂ) else 0) :=
        List.map_congr_left fun c hc => by
          rw [hAc c hc]
          split_ifs with h
          · rw [h]
          · simp
      rw [hl, List.sum_map_mul_left, sum_ind_count, hcount _ ro]
      simp
    · have hz0 : zc E b (τ u) = 0 := hbox _ ru hu0 hu1
      rw [hz0]
      simp only [zero_mul, mul_zero]
      rw [show (cs.map fun c => zc E c.2.1 (τ u) * (if τ o = (c.2.2.2.1.length * τ u) % d
          then (1 : ℂ) else 0)) = cs.map fun _ => (0 : ℂ) from
        List.map_congr_left fun c hc => by rw [hbc c hc _ ru hu0 hu1, zero_mul]]
      simp

theorem LocalEq.flip' {E : Env} {L R : List Node} {iL iR iface : List ℕ} {s : ℂ}
    (h : LocalEq E L iL R iR iface 1 s) (hs : s ≠ 0) (t : ℂ) (ht : t = s⁻¹) :
    LocalEq E R iR L iL iface 1 t := by
  intro τ hτ
  have e := h τ hτ
  rw [ht, one_mul] at *
  rw [e, ← mul_assoc, inv_mul_cancel₀ hs, one_mul]


/-- Two labels per element, each pinned to a value, under a function of their value lists. -/
theorem sumOver_pin2 {D : ℕ → ℕ} {β : Type*} (cs : List β) (a b va vb : β → ℕ) (k : β → ℂ)
    (G : List ℕ → List ℕ → ℂ) (hnd : (cs.flatMap fun c => [a c, b c]).Nodup)
    (hva : ∀ c ∈ cs, va c < D (a c)) (hvb : ∀ c ∈ cs, vb c < D (b c)) (σ : ℕ → ℕ) :
    sumOver D (cs.flatMap fun c => [a c, b c]) (fun ρ =>
      (cs.map fun c => if ρ (a c) = va c ∧ ρ (b c) = vb c then k c else 0).prod *
        G (cs.map fun c => ρ (a c)) (cs.map fun c => ρ (b c))) σ =
      (cs.map k).prod * G (cs.map va) (cs.map vb) := by
  induction cs generalizing G σ with
  | nil => simp [sumOver]
  | cons c cs ih =>
    simp only [List.flatMap_cons, List.cons_append, List.nil_append] at hnd ⊢
    obtain ⟨hna, hnb⟩ := List.nodup_cons.1 hnd
    obtain ⟨hnb', hnr⟩ := List.nodup_cons.1 hnb
    have hab : a c ≠ b c := fun h => hna (h ▸ List.mem_cons_self)
    have har : a c ∉ cs.flatMap fun c => [a c, b c] := fun h => hna (List.mem_cons_of_mem _ h)
    simp only [sumOver]
    have step : ∀ j j', sumOver D (cs.flatMap fun c => [a c, b c]) (fun ρ =>
        (List.map (fun c => if ρ (a c) = va c ∧ ρ (b c) = vb c then k c else 0) (c :: cs)).prod *
          G (List.map (fun c => ρ (a c)) (c :: cs)) (List.map (fun c => ρ (b c)) (c :: cs)))
        (Function.update (Function.update σ (a c) j) (b c) j') =
        (if j = va c ∧ j' = vb c then k c else 0) *
          ((cs.map k).prod * G (j :: cs.map va) (j' :: cs.map vb)) := by
      intro j j'
      rw [sumOver_congr _ _ _ (fun ρ => (if j = va c ∧ j' = vb c then k c else 0) *
        ((cs.map fun c => if ρ (a c) = va c ∧ ρ (b c) = vb c then k c else 0).prod *
          (fun xs ys => G (j :: xs) (j' :: ys)) (cs.map fun c => ρ (a c))
            (cs.map fun c => ρ (b c)))) _ fun ρ hρ _ => by
        have e1 : ρ (a c) = j := by
          rw [hρ _ har, Function.update_of_ne hab, Function.update_self]
        have e2 : ρ (b c) = j' := by rw [hρ _ hnb', Function.update_self]
        simp only [List.map_cons, List.prod_cons, e1, e2]
        ring]
      rw [← mul_sumOver, ih (fun xs ys => G (j :: xs) (j' :: ys)) hnr (fun c hc => hva c (List.mem_cons_of_mem _ hc))
        (fun c hc => hvb c (List.mem_cons_of_mem _ hc))]
    simp only [step]
    rw [Finset.sum_eq_single (va c) (fun j _ hj => Finset.sum_eq_zero fun j' _ => by
        rw [if_neg (fun h => hj h.1), zero_mul])
      (fun h => absurd (Finset.mem_range.2 (hva c List.mem_cons_self)) h),
      Finset.sum_eq_single (vb c) (fun j' _ hj => by rw [if_neg (fun h => hj h.2), zero_mul])
      (fun h => absurd (Finset.mem_range.2 (hvb c List.mem_cons_self)) h)]
    simp only [and_self, if_true, List.map_cons, List.prod_cons]
    ring

theorem filter_single_sum {β : Type*} (d x : ℕ) (cs : List β) (fo e : β → ℕ)
    (h : (cs.map fo).filter (· ≠ 0) = [x]) :
    ∃ c ∈ cs, (cs.map fun c => (e c * fo c) % d).sum = (e c * x) % d := by
  induction cs with
  | nil => simp at h
  | cons c cs ih =>
    simp only [List.map_cons, List.filter_cons] at h
    by_cases h0 : fo c = 0
    · simp only [h0, ne_eq, not_true_eq_false, decide_false, Bool.false_eq_true,
        if_false] at h
      obtain ⟨c', hc', e'⟩ := ih h
      exact ⟨c', List.mem_cons_of_mem _ hc', by simp [h0, e']⟩
    · simp only [ne_eq, h0, not_false_eq_true, decide_true, if_true, List.cons.injEq] at h
      obtain ⟨hx, hr⟩ := h
      have hz : ∀ c' ∈ cs, fo c' = 0 := by
        intro c' hc'
        by_contra hne
        have : fo c' ∈ (cs.map fo).filter (· ≠ 0) := by simp [hne]; exact ⟨c', hc', rfl⟩
        rw [hr] at this
        simp at this
      refine ⟨c, List.mem_cons_self, ?_⟩
      simp only [List.map_cons, List.sum_cons, hx]
      rw [List.map_congr_left fun c' hc' => by rw [hz c' hc', mul_zero, Nat.zero_mod]]
      simp

/-- A W node at `d - 1` with a pink dot on multiples `e c` of its other outputs. -/
theorem wKz {β : Type*} (d : ℕ) (hd : 1 < d) (o₀ : ℕ) (cs : List β) (fo e : β → ℕ)
    (he : ∀ c ∈ cs, e c % d ≠ 0) :
    tW d (o₀ :: cs.map fo) (d - 1) *
        (if (cs.map fun c => (e c * fo c) % d).sum % d = 0 then 1 else 0) =
      if o₀ = d - 1 ∧ ∀ c ∈ cs, fo c = 0 then 1 else 0 := by
  have hd1 : d - 1 ≠ 0 := by omega
  simp only [tW, hd1, if_false, show d - 1 < d by omega, true_and, List.filter_cons]
  by_cases h0 : o₀ = 0
  · have hn : ¬ (o₀ = d - 1 ∧ ∀ c ∈ cs, fo c = 0) := fun h => hd1 (h.1 ▸ h0)
    rw [if_neg hn]
    simp only [h0, ne_eq, not_true_eq_false, decide_false, Bool.false_eq_true, if_false]
    split_ifs with hf hs
    · exfalso
      obtain ⟨c, hc, hsum⟩ := filter_single_sum d (d - 1) cs fo e hf
      rw [hsum, Nat.mod_mod] at hs
      apply he c hc
      have : (e c * (d - 1) + e c) % d = 0 := by
        rw [show e c * (d - 1) + e c = e c * d by
          rw [Nat.mul_sub, mul_one]; have := Nat.le_mul_of_pos_right (e c) (by omega : 0 < d)
          omega]
        exact Nat.mul_mod_left _ _
      rw [Nat.add_mod, hs, zero_add, Nat.mod_mod] at this
      exact this
    · simp
    · simp
    · simp
  · simp only [ne_eq, h0, not_false_eq_true, decide_true, if_true, List.cons.injEq]
    have hall : (cs.map fo).filter (· ≠ 0) = [] ↔ ∀ c ∈ cs, fo c = 0 := by
      simp [List.filter_eq_nil_iff]
    by_cases hc : o₀ = d - 1 ∧ ∀ c ∈ cs, fo c = 0
    · rw [if_pos ⟨hc.1, hall.2 hc.2⟩, if_pos hc]
      have : (cs.map fun c => (e c * fo c) % d) = cs.map fun _ => 0 :=
        List.map_congr_left fun c h => by rw [hc.2 c h, mul_zero, Nat.zero_mod]
      simp [this]
    · rw [if_neg (fun h => hc ⟨h.1, hall.1 h.2⟩), if_neg hc, zero_mul]

/-- The transposed multiplier, pink dot (`q` -> `ws`) into the plain green dot (`ws` -> `m`), at
an output value. -/
theorem multT_val (E : Env) (pz px dz q m : ℕ) (ws : List ℕ) (τ : ℕ → ℕ)
    (hd : 0 < E.dimv dz) (hnd : ws.Nodup) (hq : q ∉ ws) (hm : m ∉ ws)
    (hws : ∀ w ∈ ws, labIdx E.labs w = dz) (rm : τ m < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) :
    sumOver (labDim E) ws (prodVal E [⟨.X, ws, [q], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩]) τ =
      (rt (E.dimv dz))⁻¹ ^ (ws.length + 1) *
        (if (ws.length * τ m) % E.dimv dz = τ q % E.dimv dz then (E.dimv dz : ℂ) else 0) := by
  have hD : ∀ w ∈ ws, labDim E w = E.dimv dz := fun w hw => labDim_of E w dz (hws w hw)
  have key : sumOver (labDim E) ws (prodVal E [⟨.X, ws, [q], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩]) τ =
      sumOver (labDim E) ws (fun ρ => if ∀ l ∈ ws, ρ l = ρ ((fun _ => m) l) then
        tX (E.dimv dz) (E.phase px) (ws.map ρ) [ρ q] else 0) τ := by
    refine sumOver_congr _ _ _ _ _ fun ρ hρ _ => ?_
    have em : ρ m = τ m := hρ m hm
    have rm' : ρ m < E.dimv dz := em ▸ rm
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      val, List.cons_append, List.nil_append]
    rw [tZ_at _ _ _ (ρ m) (by simp), hz _ rm']
    have hz' : (ρ m < E.dimv dz ∧ ∀ x ∈ ρ m :: ws.map ρ, x = ρ m) ↔ ∀ w ∈ ws, ρ w = ρ m := by
      constructor
      · rintro ⟨-, h⟩ w hw
        exact h _ (List.mem_cons_of_mem _ (List.mem_map_of_mem hw))
      · intro h
        refine ⟨rm', fun x hx => ?_⟩
        rcases List.mem_cons.1 hx with hx | hx
        · exact hx
        · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hx
          exact h w hw
    by_cases hc : ∀ w ∈ ws, ρ w = ρ m
    · rw [if_pos (hz'.2 hc), if_pos hc, mul_one]
    · rw [if_neg (fun h => hc (hz'.1 h)), if_neg hc, mul_zero]
  rw [key, sumOver_pin (labDim E) ws hnd (fun _ => m) (fun _ _ => hm) _ τ
    (fun w hw => by rw [hD w hw]; exact rm)]
  simp only
  have eq : (if q ∈ ws then τ m else τ q) = τ q := by rw [if_neg hq]
  rw [eq, tX_kj _ 0 hd _ (fun k hk => by rw [hx k hk]; simp)]
  have hmap : ws.map (fun x => if x ∈ ws then τ m else τ x) = List.replicate ws.length (τ m) := by
    rw [List.eq_replicate_iff]
    refine ⟨by simp, fun x hx => ?_⟩
    obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hx
    simp [hw]
  rw [hmap]
  simp [List.sum_replicate]

/-- The nodes of a KZ chain `(w, q, ws, m, o)`: the transposed multiplier and the green dot. -/
def kzNodes (dz px pz pg : ℕ) (c : ℕ × ℕ × List ℕ × ℕ × ℕ) : List Node :=
  [⟨.X, c.2.2.1, [c.2.1], dz, px⟩, ⟨.Z, [c.2.2.2.1], c.2.2.1, dz, pz⟩,
    ⟨.Z, [c.2.2.2.2], [c.1, c.2.2.2.1], dz, pg⟩]

theorem kzChainVal (E : Env) (dz px pz pg : ℕ) (c : ℕ × ℕ × List ℕ × ℕ × ℕ) (ρ : ℕ → ℕ)
    (hd : 0 < E.dimv dz) (hnd : (c.2.2.2.1 :: c.2.2.1).Nodup)
    (hout : ∀ x ∈ [c.1, c.2.1, c.2.2.2.2], x ∉ c.2.2.2.1 :: c.2.2.1)
    (hlab : ∀ x ∈ c.2.2.2.1 :: c.2.2.1, labIdx E.labs x = dz) (ro : ρ c.2.2.2.2 < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) :
    sumOver (labDim E) (c.2.2.2.1 :: c.2.2.1) (prodVal E (kzNodes dz px pz pg c)) ρ =
      (if ρ c.1 = ρ c.2.2.2.2 then 1 else 0) * ((rt (E.dimv dz))⁻¹ ^ (c.2.2.1.length + 1) *
        (if (c.2.2.1.length * ρ c.2.2.2.2) % E.dimv dz = ρ c.2.1 % E.dimv dz then
          (E.dimv dz : ℂ) else 0)) := by
  obtain ⟨w, q, ws, m, o⟩ := c
  simp only at hnd hout hlab ro ⊢
  obtain ⟨hm, hws⟩ := List.nodup_cons.1 hnd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
    not_or] at hout
  obtain ⟨⟨nwm, nww⟩, ⟨nqm, nqw⟩, ⟨nom, now⟩⟩ := hout
  simp only [sumOver, labDim_of E m dz (hlab m List.mem_cons_self)]
  have split : prodVal E (kzNodes dz px pz pg (w, q, ws, m, o)) = fun τ =>
      tZ (E.dimv dz) (E.phase pg) [τ o, τ w, τ m] *
        prodVal E [⟨.X, ws, [q], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩] τ := by
    funext τ
    simp only [kzNodes, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      mul_one, val, List.cons_append, List.nil_append]
    ring
  rw [split]
  have inner : ∀ j ∈ range (E.dimv dz), sumOver (labDim E) ws (fun τ =>
      tZ (E.dimv dz) (E.phase pg) [τ o, τ w, τ m] *
        prodVal E [⟨.X, ws, [q], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩] τ) (Function.update ρ m j) =
      (if j = ρ o ∧ ρ w = ρ o then 1 else 0) * ((rt (E.dimv dz))⁻¹ ^ (ws.length + 1) *
        (if (ws.length * j) % E.dimv dz = ρ q % E.dimv dz then (E.dimv dz : ℂ) else 0)) := by
    intro j hj
    have rj := Finset.mem_range.1 hj
    rw [sumOver_const_mul _ _ [o, w, m] (fun τ => tZ (E.dimv dz) (E.phase pg) [τ o, τ w, τ m]) _
      (fun τ τ' h => by simp only; rw [h o (by simp), h w (by simp), h m (by simp)])
      (fun x hx hx' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases hx with rfl | rfl | rfl
        · exact now hx'
        · exact nww hx'
        · exact hm hx'),
      multT_val E pz px dz q m ws _ hd hws (fun h => nqw h) hm
        (fun x hx => hlab x (List.mem_cons_of_mem _ hx)) (by rw [Function.update_self]; exact rj)
        hz hx]
    simp only [Function.update_self, Function.update_of_ne nom, Function.update_of_ne nwm,
      Function.update_of_ne nqm]
    rw [tZ_at _ _ _ (ρ o) (by simp), hg _ ro]
    congr 1
    by_cases h1 : j = ρ o <;> by_cases h2 : ρ w = ρ o <;> simp [h1, h2, ro, eq_comm]
  rw [Finset.sum_congr rfl inner]
  simp only [ite_and, ite_mul, zero_mul, one_mul]
  rw [Finset.sum_ite_eq' (range (E.dimv dz)) (ρ o), if_pos (Finset.mem_range.2 ro)]

theorem kzNodes_labels (dz px pz pg : ℕ) (c : ℕ × ℕ × List ℕ × ℕ × ℕ) :
    ∀ x ∈ labelsOf (kzNodes dz px pz pg c),
      x ∈ c.2.2.2.1 :: c.2.2.1 ∨ x = c.1 ∨ x = c.2.1 ∨ x = c.2.2.2.2 := by
  intro x hx
  simp only [labelsOf, kzNodes, List.flatMap_cons, List.flatMap_nil, Node.labels,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.append_nil] at hx ⊢
  aesop

/-- The KZ left side: a pink `K_1` state on a W node whose other outputs meet a pink dot through
transposed multipliers. -/
theorem kzVal (E : Env) (dz x₀ o₀ pk pq px pz pg : ℕ) (cs : List (ℕ × ℕ × List ℕ × ℕ × ℕ))
    (τ : ℕ → ℕ) (hd : 1 < E.dimv dz)
    (hP : (cs.flatMap fun c => [c.1, c.2.1]).Nodup)
    (hQ : (cs.flatMap fun c => c.2.2.2.1 :: c.2.2.1).Nodup)
    (hPQ : ∀ x ∈ cs.flatMap (fun c => [c.1, c.2.1]), x ∉ cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1))
    (hx₀ : x₀ ∉ cs.flatMap (fun c => [c.1, c.2.1]) ++ cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1))
    (houts : ∀ x ∈ o₀ :: cs.map (fun c => c.2.2.2.2), x ∉ x₀ :: (cs.flatMap (fun c => [c.1, c.2.1]) ++
      cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1)))
    (hlab : ∀ x ∈ x₀ :: (cs.flatMap (fun c => [c.1, c.2.1]) ++
      cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1)), labIdx E.labs x = dz)
    (ro₀ : τ o₀ < E.dimv dz) (ros : ∀ c ∈ cs, τ c.2.2.2.2 < E.dimv dz)
    (hk : ∀ k < E.dimv dz, ex (E.phase pk k) = om (E.dimv dz) ^ (k * 1))
    (hq : ∀ k < E.dimv dz, ex (E.phase pq k) = 1) (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1)
    (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1) (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1)
    (he : ∀ c ∈ cs, c.2.2.1.length % E.dimv dz ≠ 0) :
    sumOver (labDim E) (x₀ :: (cs.flatMap (fun c => [c.1, c.2.1]) ++
        cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1)))
      (prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ ::
        ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ :: cs.flatMap (kzNodes dz px pz pg))) τ =
      ((rt (E.dimv dz))⁻¹ * E.dimv dz) *
        (cs.map fun c => (rt (E.dimv dz))⁻¹ ^ (c.2.2.1.length + 1) * E.dimv dz).prod *
        ((rt (E.dimv dz))⁻¹ ^ cs.length * E.dimv dz) *
        (if τ o₀ = E.dimv dz - 1 ∧ ∀ c ∈ cs, τ c.2.2.2.2 = 0 then 1 else 0) := by
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  set P := cs.flatMap fun c => [c.1, c.2.1]
  set Q := cs.flatMap fun c => c.2.2.2.1 :: c.2.2.1
  have hD : ∀ x ∈ x₀ :: (P ++ Q), labDim E x = d := fun x hx => labDim_of E x dz (hlab x hx)
  have hwP : ∀ c ∈ cs, c.1 ∈ P := fun c hc => List.mem_flatMap.2 ⟨c, hc, by simp⟩
  have hqP : ∀ c ∈ cs, c.2.1 ∈ P := fun c hc => List.mem_flatMap.2 ⟨c, hc, by simp⟩
  have hQc : ∀ c ∈ cs, ∀ x ∈ c.2.2.2.1 :: c.2.2.1, x ∈ Q := fun c hc x hx =>
    List.mem_flatMap.2 ⟨c, hc, hx⟩
  have ho₀ := houts o₀ List.mem_cons_self
  have hoc : ∀ c ∈ cs, c.2.2.2.2 ∉ x₀ :: (P ++ Q) := fun c hc =>
    houts _ (List.mem_cons_of_mem _ (List.mem_map_of_mem hc))
  -- the chains
  have hgrp : ∀ c ∈ cs, ∀ ρ : ℕ → ℕ, ρ c.2.2.2.2 < d →
      sumOver (labDim E) (c.2.2.2.1 :: c.2.2.1) (prodVal E (kzNodes dz px pz pg c)) ρ =
      (if ρ c.1 = ρ c.2.2.2.2 then 1 else 0) * ((rt d)⁻¹ ^ (c.2.2.1.length + 1) *
        (if (c.2.2.1.length * ρ c.2.2.2.2) % d = ρ c.2.1 % d then (d : ℂ) else 0)) := by
    intro c hc ρ hr
    refine kzChainVal E dz px pz pg c ρ hd' ((List.nodup_flatMap.1 hQ).1 c hc) ?_
      (fun x hx => hlab x (List.mem_cons_of_mem _ (List.mem_append_right _ (hQc c hc x hx))))
      hr hz hx hg
    intro x hx hx'
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl
    · exact hPQ _ (hwP c hc) (hQc c hc _ hx')
    · exact hPQ _ (hqP c hc) (hQc c hc _ hx')
    · exact hoc c hc (List.mem_cons_of_mem _ (List.mem_append_right _ (hQc c hc _ hx')))
  set S := P ++ cs.map (fun c => c.2.2.2.2)
  have hin : ∀ ρ : ℕ → ℕ, (∀ c ∈ cs, ρ c.2.2.2.2 < d) →
      sumOver (labDim E) Q (prodVal E (⟨.X, [x₀], [], dz, pk⟩ ::
        ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ :: ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ ::
          cs.flatMap (kzNodes dz px pz pg))) ρ =
      (tX d (E.phase pk) [ρ x₀] [] * (tW d (ρ o₀ :: cs.map fun c => ρ c.1) (ρ x₀) *
        tX d (E.phase pq) (cs.map fun c => ρ c.2.1) [])) *
        (cs.map fun c => (if ρ c.1 = ρ c.2.2.2.2 then 1 else 0) * ((rt d)⁻¹ ^ (c.2.2.1.length + 1) *
          (if (c.2.2.1.length * ρ c.2.2.2.2) % d = ρ c.2.1 % d then (d : ℂ) else 0))).prod := by
    intro ρ hr
    have split : prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ ::
        ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ :: cs.flatMap (kzNodes dz px pz pg)) =
        fun ρ => (tX d (E.phase pk) [ρ x₀] [] * (tW d (ρ o₀ :: cs.map fun c => ρ c.1) (ρ x₀) *
          tX d (E.phase pq) (cs.map fun c => ρ c.2.1) [])) *
          (cs.map fun c => prodVal E (kzNodes dz px pz pg c) ρ).prod := by
      funext ρ
      simp only [prodVal, List.map_cons, List.prod_cons, prod_flatMap', val, List.map_nil,
        List.map_map, Function.comp_def]
      ring
    rw [split, sumOver_const_mul _ Q (x₀ :: o₀ :: P) _ _ (fun ρ ρ' h => by
        have hw : (cs.map fun c => ρ c.1) = cs.map fun c => ρ' c.1 :=
          List.map_congr_left fun c hc => h _ (by simp [hwP c hc])
        have hq' : (cs.map fun c => ρ c.2.1) = cs.map fun c => ρ' c.2.1 :=
          List.map_congr_left fun c hc => h _ (by simp [hqP c hc])
        rw [hw, hq', h x₀ (by simp), h o₀ (by simp)])
      (fun x hx hxq => by
        simp only [List.mem_cons] at hx
        rcases hx with rfl | rfl | hx
        · exact hx₀ (List.mem_append_right _ hxq)
        · exact ho₀ (List.mem_cons_of_mem _ (List.mem_append_right _ hxq))
        · exact hPQ x hx hxq) ρ]
    have hg := sumOver_groups (D := labDim E) (cs.map fun c => (c.2.2.2.1 :: c.2.2.1,
      fun τ => prodVal E (kzNodes dz px pz pg c) τ)) S
    simp only [List.flatMap_map, List.map_map, Function.comp_def] at hg
    rw [hg hQ (fun x hx hx' => by
        rcases List.mem_append.1 hx with h | h
        · exact hPQ x h hx'
        · obtain ⟨c, hc, rfl⟩ := List.mem_map.1 h
          exact hoc c hc (List.mem_cons_of_mem _ (List.mem_append_right _ hx')))
      (fun gg hgg => by
        obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hgg
        intro τ τ' h
        refine prodVal_reads E (kzNodes dz px pz pg c) τ τ' fun x hx => h x ?_
        rcases kzNodes_labels dz px pz pg c x hx with h1 | h1 | h1 | h1
        · exact List.mem_append_left _ h1
        · exact List.mem_append_right _ (List.mem_append_left _ (h1 ▸ hwP c hc))
        · exact List.mem_append_right _ (List.mem_append_left _ (h1 ▸ hqP c hc))
        · exact List.mem_append_right _ (List.mem_append_right _
            (h1 ▸ List.mem_map_of_mem (f := fun c => c.2.2.2.2) hc))) ρ]
    rw [List.map_congr_left fun c hc => hgrp c hc ρ (hr c hc)]
  set K := (cs.map fun c => (rt d)⁻¹ ^ (c.2.2.1.length + 1) * (d : ℂ)).prod
  have hj : ∀ j ∈ range d, sumOver (labDim E) (P ++ Q) (prodVal E (⟨.X, [x₀], [], dz, pk⟩ ::
      ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ :: ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ ::
        cs.flatMap (kzNodes dz px pz pg))) (Function.update τ x₀ j) =
      K * (tX d (E.phase pk) [j] [] * (tW d (τ o₀ :: cs.map fun c => τ c.2.2.2.2) j *
        tX d (E.phase pq) (cs.map fun c => (c.2.2.1.length * τ c.2.2.2.2) % d) [])) := by
    intro j _
    rw [sumOver_append]
    have e_out : ∀ ρ : ℕ → ℕ, (∀ x ∉ P, ρ x = Function.update τ x₀ j x) →
        ρ x₀ = j ∧ ρ o₀ = τ o₀ ∧ ∀ c ∈ cs, ρ c.2.2.2.2 = τ c.2.2.2.2 := by
      intro ρ hρ
      refine ⟨?_, ?_, fun c hc => ?_⟩
      · rw [hρ x₀ (fun h => hx₀ (List.mem_append_left _ h)), Function.update_self]
      · rw [hρ o₀ (fun h => ho₀ (List.mem_cons_of_mem _ (List.mem_append_left _ h))),
          Function.update_of_ne (fun h => ho₀ (by rw [h]; exact List.mem_cons_self))]
      · have h1 := hoc c hc
        rw [hρ _ (fun h => h1 (List.mem_cons_of_mem _ (List.mem_append_left _ h))),
          Function.update_of_ne (fun h => h1 (by rw [h]; exact List.mem_cons_self))]
    rw [sumOver_congr _ _ _ (fun ρ => (cs.map fun c => if ρ c.1 = τ c.2.2.2.2 ∧
        ρ c.2.1 = (c.2.2.1.length * τ c.2.2.2.2) % d then (rt d)⁻¹ ^ (c.2.2.1.length + 1) * (d : ℂ)
          else 0).prod * (fun xs ys => tX d (E.phase pk) [j] [] * (tW d (τ o₀ :: xs) j *
            tX d (E.phase pq) ys [])) (cs.map fun c => ρ c.1) (cs.map fun c => ρ c.2.1)) _
      fun ρ hρ hr => by
        obtain ⟨e0, eo, ec⟩ := e_out ρ hρ
        rw [hin ρ (fun c hc => by rw [ec c hc]; exact ros c hc), e0, eo, mul_comm]
        congr 1
        refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
        have rq : ρ c.2.1 < d := by
          have := hr _ (hqP c hc); rwa [hD _ (List.mem_cons_of_mem _
            (List.mem_append_left _ (hqP c hc)))] at this
        rw [ec c hc, Nat.mod_eq_of_lt rq]
        simp only [@eq_comm _ (ρ c.2.1)]
        by_cases h1 : ρ c.1 = τ c.2.2.2.2 <;>
          by_cases h2 : (c.2.2.1.length * τ c.2.2.2.2) % d = ρ c.2.1 <;>
          simp only [h1, h2, and_true, and_false, true_and, false_and, ite_true, ite_false,
            one_mul, zero_mul, mul_zero, not_false_eq_true]]
    refine (sumOver_pin2 (D := labDim E) cs Prod.fst (fun c => c.2.1) (fun c => τ c.2.2.2.2)
      (fun c => (c.2.2.1.length * τ c.2.2.2.2) % d)
      (fun c => (rt d)⁻¹ ^ (c.2.2.1.length + 1) * (d : ℂ))
      (fun xs ys => tX d (E.phase pk) [j] [] * (tW d (τ o₀ :: xs) j * tX d (E.phase pq) ys [])) hP
      (fun c hc => by
        rw [hD _ (List.mem_cons_of_mem _ (List.mem_append_left _ (hwP c hc)))]; exact ros c hc)
      (fun c hc => by
        rw [hD _ (List.mem_cons_of_mem _ (List.mem_append_left _ (hqP c hc)))]
        exact Nat.mod_lt _ hd') _).trans ?_
    rfl
  rw [show sumOver (labDim E) (x₀ :: (P ++ Q)) (prodVal E (⟨.X, [x₀], [], dz, pk⟩ ::
      ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ :: ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ ::
        cs.flatMap (kzNodes dz px pz pg))) τ = ∑ j ∈ range d, sumOver (labDim E) (P ++ Q)
          (prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ ::
            ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ :: cs.flatMap (kzNodes dz px pz pg)))
          (Function.update τ x₀ j) by
    simp only [sumOver, hD x₀ List.mem_cons_self]]
  rw [Finset.sum_congr rfl hj]
  have hq0 : ∀ k < d, ex (E.phase pq k) = om d ^ (k * 0) := fun k hk => by rw [hq k hk]; simp
  simp only [tX_kj d 1 hd' _ hk, tX_kj d 0 hd' _ hq0, List.length_singleton, List.length_nil,
    List.sum_singleton, List.sum_nil, Nat.zero_mod, add_zero, List.length_map]
  set Y := (cs.map fun c => (c.2.2.1.length * τ c.2.2.2.2) % d).sum % d = 0
  set F : ℕ → ℂ := fun x => K * ((rt d)⁻¹ * d * (tW d (τ o₀ :: cs.map fun c => τ c.2.2.2.2) x *
    ((rt d)⁻¹ ^ cs.length * (if Y then (d : ℂ) else 0))))
  rw [Finset.sum_congr rfl fun x _ => (show _ = if (x + 1) % d = 0 then F x else 0 by
    simp only [F]; split_ifs <;> ring), sum_pin_shift d 1 0 hd' hd' F, Nat.mod_eq_of_lt hd,
    zero_add, Nat.mod_eq_of_lt (show d - 1 < d by omega)]
  have hY : (if Y then (d : ℂ) else 0) = d * (if Y then 1 else 0) := by split_ifs <;> simp
  have hw := wKz d hd (τ o₀) cs (fun c => τ c.2.2.2.2) (fun c => c.2.2.1.length) he
  simp only [F, hY]
  rw [← hw]
  ring

theorem prod_ite_all {β : Type*} (cs : List β) (P : β → Prop) [DecidablePred P] (a b : ℂ) :
    (cs.map fun c => a * if P c then b else 0).prod =
      if ∀ c ∈ cs, P c then (a * b) ^ cs.length else 0 := by
  induction cs with
  | nil => simp
  | cons c cs ih =>
    simp only [List.map_cons, List.prod_cons, ih, List.forall_mem_cons, List.length_cons]
    by_cases h : P c <;> by_cases h' : ∀ x ∈ cs, P x <;> simp [h, h', pow_succ] <;> ring

theorem kz_const {β : Type*} (r : ℂ) (hr : r ≠ 0) (cs : List β) (len : β → ℕ) :
    (cs.map fun c => r⁻¹ ^ (len c + 1) * r ^ 2).prod * r ^ (cs.map len).sum = r ^ cs.length := by
  induction cs with
  | nil => simp
  | cons c cs ih =>
    simp only [List.map_cons, List.prod_cons, List.sum_cons, List.length_cons]
    have h1 : r⁻¹ ^ (len c + 1) * r ^ 2 * r ^ len c = r := by
      rw [inv_pow]; field_simp; ring
    calc r⁻¹ ^ (len c + 1) * r ^ 2 * (cs.map fun c => r⁻¹ ^ (len c + 1) * r ^ 2).prod *
          r ^ (len c + (cs.map len).sum)
        = (r⁻¹ ^ (len c + 1) * r ^ 2 * r ^ len c) *
          ((cs.map fun c => r⁻¹ ^ (len c + 1) * r ^ 2).prod * r ^ (cs.map len).sum) := by ring
      _ = r ^ (cs.length + 1) := by rw [h1, ih]; ring

/-- [P1] (KZ): a pink `K_1` state on a W node whose other outputs meet a pink dot through
transposed multipliers `e_c` is a pink `K_1` state and pink dot states, times `s`. -/
theorem zxwKZ (E : Env) (dz x₀ o₀ pk pq px pz pg pk' pq' : ℕ)
    (cs : List (ℕ × ℕ × List ℕ × ℕ × ℕ)) (iface : List ℕ) (hd : 1 < E.dimv dz)
    (hP : (cs.flatMap fun c => [c.1, c.2.1]).Nodup)
    (hQ : (cs.flatMap fun c => c.2.2.2.1 :: c.2.2.1).Nodup)
    (hPQ : ∀ x ∈ cs.flatMap (fun c => [c.1, c.2.1]), x ∉ cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1))
    (hx₀ : x₀ ∉ cs.flatMap (fun c => [c.1, c.2.1]) ++ cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1))
    (houts : ∀ x ∈ o₀ :: cs.map (fun c => c.2.2.2.2), x ∉ x₀ :: (cs.flatMap (fun c => [c.1, c.2.1]) ++
      cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1)))
    (hlab : ∀ x ∈ x₀ :: (cs.flatMap (fun c => [c.1, c.2.1]) ++
      cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1)), labIdx E.labs x = dz)
    (hio : ∀ x ∈ o₀ :: cs.map (fun c => c.2.2.2.2), x ∈ iface ∧ labIdx E.labs x = dz)
    (hk : ∀ k < E.dimv dz, ex (E.phase pk k) = om (E.dimv dz) ^ (k * 1))
    (hk' : ∀ k < E.dimv dz, ex (E.phase pk' k) = om (E.dimv dz) ^ (k * 1))
    (hph : ∀ ph ∈ [pq, pz, px, pg, pq'], ∀ k < E.dimv dz, ex (E.phase ph k) = 1)
    (he : ∀ c ∈ cs, c.2.2.1.length % E.dimv dz ≠ 0)
    (hge : 2 ≤ cs.length + (cs.map fun c => c.2.2.1.length).sum)
    (s : ℂ) (hs : s = (rt (E.dimv dz))⁻¹ ^ (cs.length + (cs.map fun c => c.2.2.1.length).sum - 2)) :
    LocalEq E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, o₀ :: cs.map Prod.fst, [x₀], dz, 0⟩ ::
        ⟨.X, cs.map (fun c => c.2.1), [], dz, pq⟩ :: cs.flatMap (kzNodes dz px pz pg))
      (x₀ :: (cs.flatMap (fun c => [c.1, c.2.1]) ++ cs.flatMap (fun c => c.2.2.2.1 :: c.2.2.1)))
      (⟨.X, [o₀], [], dz, pk'⟩ :: cs.map fun c => ⟨.X, [c.2.2.2.2], [], dz, pq'⟩) [] iface 1 s := by
  intro τ hτ
  have hr : ∀ x ∈ o₀ :: cs.map (fun c => c.2.2.2.2), τ x < E.dimv dz := fun x hx => by
    have := hτ x (hio x hx).1; rwa [labDim_of E x dz (hio x hx).2] at this
  have ro₀ := hr o₀ List.mem_cons_self
  have ros : ∀ c ∈ cs, τ c.2.2.2.2 < E.dimv dz := fun c hc =>
    hr _ (List.mem_cons_of_mem _ (List.mem_map_of_mem hc))
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨hq, hz, hx, hg, hq'⟩ := hph'
  rw [one_mul, kzVal E dz x₀ o₀ pk pq px pz pg cs τ hd hP hQ hPQ hx₀ houts hlab ro₀ ros hk hq hz hx
    hg he]
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  have hq0 : ∀ k < d, ex (E.phase pq' k) = om d ^ (k * 0) := fun k hk => by rw [hq' k hk]; simp
  simp only [sumOver, prodVal, List.map_cons, List.map_map, Function.comp_def, List.prod_cons, val,
    List.map_nil]
  simp only [← hdd]
  simp only [tX_kj d 1 hd' _ hk', tX_kj d 0 hd' _ hq0, List.length_singleton,
    List.length_nil, List.sum_singleton, List.sum_nil, Nat.zero_mod, add_zero, zero_add]
  rw [prod_ite_all]
  have e0 : ((τ o₀ + 1) % d = 0) ↔ τ o₀ = d - 1 := by
    constructor
    · intro h
      by_cases h' : τ o₀ + 1 < d
      · rw [Nat.mod_eq_of_lt h'] at h; omega
      · omega
    · intro h; rw [h, show d - 1 + 1 = d by omega, Nat.mod_self]
  have ec : (∀ c ∈ cs, τ c.2.2.2.2 % d = 0) ↔ ∀ c ∈ cs, τ c.2.2.2.2 = 0 :=
    ⟨fun h c hc => by have := h c hc; rwa [Nat.mod_eq_of_lt (ros c hc)] at this,
      fun h c hc => by rw [h c hc, Nat.zero_mod]⟩
  have hrr := rt_ne d hd'
  have hdr : (d : ℂ) = rt d ^ 2 := (rt_sq' d).symm
  by_cases hc : τ o₀ = d - 1 ∧ ∀ c ∈ cs, τ c.2.2.2.2 = 0
  · rw [if_pos hc, if_pos (e0.2 hc.1), if_pos (ec.2 hc.2), mul_one, hs]
    have hA := kz_const (rt d) hrr cs fun c => c.2.2.1.length
    obtain ⟨N, hN⟩ := Nat.exists_eq_add_of_le hge
    rw [hN, show 2 + N - 2 = N by omega, hdr]
    have hrr2 : (rt d)⁻¹ * rt d ^ 2 = rt d := by field_simp
    apply mul_right_cancel₀ (pow_ne_zero (cs.map fun c => c.2.2.1.length).sum hrr)
    simp only [pow_one, hrr2]
    have hpow : rt d ^ cs.length * rt d ^ (cs.map fun c => c.2.2.1.length).sum =
        rt d ^ 2 * rt d ^ N := by rw [← pow_add, ← pow_add, hN]
    calc rt d * (cs.map fun c => (rt d)⁻¹ ^ (c.2.2.1.length + 1) * rt d ^ 2).prod *
          ((rt d)⁻¹ ^ cs.length * rt d ^ 2) * rt d ^ (cs.map fun c => c.2.2.1.length).sum
        = rt d * ((cs.map fun c => (rt d)⁻¹ ^ (c.2.2.1.length + 1) * rt d ^ 2).prod *
          rt d ^ (cs.map fun c => c.2.2.1.length).sum) * ((rt d)⁻¹ ^ cs.length * rt d ^ 2) := by
          ring
      _ = rt d * rt d ^ cs.length * ((rt d)⁻¹ ^ cs.length * rt d ^ 2) := by rw [hA]
      _ = (rt d)⁻¹ ^ N * rt d * (rt d ^ 2 * rt d ^ N) := by
          rw [inv_pow, inv_pow]; field_simp
      _ = (rt d)⁻¹ ^ N * (rt d * rt d ^ cs.length) *
          rt d ^ (cs.map fun c => c.2.2.1.length).sum := by rw [← hpow]; ring
  · rw [if_neg hc]
    by_cases h0 : τ o₀ = d - 1
    · rw [if_pos (e0.2 h0), if_neg (fun h => hc ⟨h0, ec.1 h⟩)]; simp
    · rw [if_neg (fun h => h0 (e0.1 h))]; simp


/-- A qf_hd chain: top value `t`, multiplier wires `ws`, the green dot `m, w -> y`, `K_{-1}`
(`y -> z`), the W transpose (`x -> s, z`) on the green state `s`, and `K_c` (`x -> v`). -/
structure HdC where
  t : ℕ
  ws : List ℕ
  m : ℕ
  w : ℕ
  y : ℕ
  z : ℕ
  s : ℕ
  x : ℕ
  v : ℕ
  pk : ℕ
  deriving DecidableEq

/-- The nodes of a qf_hd chain. -/
def hdNodes (dz px pz pg pm pr : ℕ) (c : HdC) : List Node :=
  [⟨.X, c.ws, [c.t], dz, px⟩, ⟨.Z, [c.m], c.ws, dz, pz⟩, ⟨.Z, [c.y], [c.m, c.w], dz, pg⟩,
    ⟨.X, [c.z], [c.y], dz, pm⟩, ⟨.W, [c.s, c.z], [c.x], dz, 0⟩, ⟨.Z, [c.s], [], dz, pr⟩,
    ⟨.Z, [c.v], [c.x], dz, c.pk⟩]

/-- The summed labels of a qf_hd chain. -/
def hdInt (c : HdC) : List ℕ := c.m :: c.y :: c.z :: c.s :: c.x :: c.ws

theorem wGreen' (d a b : ℕ) (hb : b < d) :
    ∑ g ∈ range d, tW d [g, a] b = if a = 0 then 1 else if a = b then 1 else 0 := by
  rw [← wGreen d a b hb]
  refine Finset.sum_congr rfl fun g _ => ?_
  unfold tW
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
    List.filter_cons]
  by_cases h1 : a = 0 <;> by_cases h2 : g = 0 <;> simp [h1, h2, and_comm]

theorem hdChainVal (E : Env) (dz px pz pg pm pr jm : ℕ) (c : HdC) (ρ : ℕ → ℕ)
    (hd : 1 < E.dimv dz) (hnd : (hdInt c).Nodup) (hout : ∀ x ∈ [c.t, c.w, c.v], x ∉ hdInt c)
    (hlab : ∀ x ∈ hdInt c, labIdx E.labs x = dz) (rw : ρ c.w < E.dimv dz)
    (rv : ρ c.v < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) (hr : ∀ k < E.dimv dz, ex (E.phase pr k) = 1)
    (hpm : ∀ q < E.dimv dz, ex (E.phase pm q) = om (E.dimv dz) ^ (q * jm))
    (hjm : (jm + 1) % E.dimv dz = 0) :
    sumOver (labDim E) (hdInt c) (prodVal E (hdNodes dz px pz pg pm pr c)) ρ =
      ((rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) *
        (if (c.ws.length * ρ c.w) % E.dimv dz = ρ c.t % E.dimv dz then (E.dimv dz : ℂ) else 0)) *
      (if (ρ c.w + 1) % E.dimv dz = 0 then ex (E.phase c.pk (ρ c.v))
        else if (ρ c.w + 1) % E.dimv dz = ρ c.v then ex (E.phase c.pk (ρ c.v)) else 0) := by
  obtain ⟨t, ws, m, w, y, z, s, x, v, pk⟩ := c
  simp only [hdInt, hdNodes] at hnd hout hlab rw rv ⊢
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  simp only [List.nodup_cons, List.mem_cons, not_or] at hnd
  obtain ⟨⟨nmy, nmz, nms, nmx, nmw⟩, ⟨nyz, nys, nyx, nyw⟩, ⟨nzs, nzx, nzw⟩, ⟨nsx, nsw⟩, nxw, hws⟩ := hnd
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
    not_or] at hout
  obtain ⟨⟨ntm, nty, ntz, nts, ntx, ntw⟩, ⟨nwm, nwy, nwz, nws, nwx, nww⟩,
    ⟨nvm, nvy, nvz, nvs, nvx, nvw⟩⟩ := hout
  have hl : ∀ q ∈ [m, y, z, s, x], labDim E q = d := by
    intro q hq
    refine labDim_of E q dz (hlab q ?_)
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq ⊢
    rcases hq with h | h | h | h | h <;> simp [h]
  have hcan : sumOver (labDim E) (m :: y :: z :: s :: x :: ws) (prodVal E
      [⟨.X, ws, [t], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩, ⟨.Z, [y], [m, w], dz, pg⟩,
        ⟨.X, [z], [y], dz, pm⟩, ⟨.W, [s, z], [x], dz, 0⟩, ⟨.Z, [s], [], dz, pr⟩,
        ⟨.Z, [v], [x], dz, pk⟩]) ρ =
      ∑ j₁ ∈ range d, ∑ j₂ ∈ range d, ∑ j₃ ∈ range d, ∑ j₄ ∈ range d, ∑ j₅ ∈ range d,
        (if j₂ = ρ w ∧ j₁ = ρ w then 1 else 0) * ((if (j₃ + jm) % d = j₂ then 1 else 0) *
          (tW d [j₄, j₃] j₅ * ((if j₅ = ρ v then ex (E.phase pk (ρ v)) else 0) *
            ((rt d)⁻¹ ^ (ws.length + 1) *
              (if (ws.length * j₁) % d = ρ t % d then (d : ℂ) else 0))))) := by
    simp only [sumOver, hl m (by simp), hl y (by simp), hl z (by simp), hl s (by simp),
      hl x (by simp)]
    refine Finset.sum_congr rfl fun j₁ h₁ => Finset.sum_congr rfl fun j₂ h₂ =>
      Finset.sum_congr rfl fun j₃ h₃ => Finset.sum_congr rfl fun j₄ h₄ =>
        Finset.sum_congr rfl fun j₅ h₅ => ?_
    have r₁ := Finset.mem_range.1 h₁
    have r₂ := Finset.mem_range.1 h₂
    have r₄ := Finset.mem_range.1 h₄
    have r₅ := Finset.mem_range.1 h₅
    set σ := Function.update (Function.update (Function.update (Function.update
      (Function.update ρ m j₁) y j₂) z j₃) s j₄) x j₅ with hσ
    have split : prodVal E [⟨.X, ws, [t], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩, ⟨.Z, [y], [m, w], dz, pg⟩,
        ⟨.X, [z], [y], dz, pm⟩, ⟨.W, [s, z], [x], dz, 0⟩, ⟨.Z, [s], [], dz, pr⟩,
        ⟨.Z, [v], [x], dz, pk⟩] = fun τ => (tZ d (E.phase pg) [τ y, τ m, τ w] *
          (tX d (E.phase pm) [τ z] [τ y] * (tW d [τ s, τ z] (τ x) *
            (tZ d (E.phase pr) [τ s] * tZ d (E.phase pk) [τ v, τ x])))) *
          prodVal E [⟨.X, ws, [t], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩] τ := by
      funext τ
      simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
        List.cons_append, List.nil_append, List.append_nil]
      ring
    rw [split, sumOver_const_mul _ _ [y, m, w, z, s, x, v] _ _ (fun τ τ' h => by
        rw [h y (by simp), h m (by simp), h w (by simp), h z (by simp), h s (by simp),
          h x (by simp), h v (by simp)])
      (fun q hq hq' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
        rcases hq with rfl | rfl | rfl | rfl | rfl | rfl | rfl
        · exact nyw hq'
        · exact nmw hq'
        · exact nww hq'
        · exact nzw hq'
        · exact nsw hq'
        · exact nxw hq'
        · exact nvw hq')]
    rw [multT_val E pz px dz t m ws σ hd' hws ntw nmw
      (fun q hq => hlab q (by simp [hq])) (by
        rw [hσ, Function.update_of_ne nmx, Function.update_of_ne nms, Function.update_of_ne nmz,
          Function.update_of_ne nmy, Function.update_self]; exact r₁) hz hx]
    simp only [hσ, Function.update_apply, nmy, nmz, nms, nmx, nyz, nys, nyx, nzs, nzx, nsx,
      Ne.symm nmy, Ne.symm nmz, Ne.symm nms, Ne.symm nmx, Ne.symm nyz, Ne.symm nys, Ne.symm nyx,
      Ne.symm nzs, Ne.symm nzx, Ne.symm nsx, ntm, nty, ntz, nts, ntx, nwm, nwy, nwz, nws, nwx,
      nvm, nvy, nvz, nvs, nvx, if_true, if_false]
    rw [← hdd, tZ_at d _ _ (ρ w) (by simp), hg _ rw, tX_shift d jm j₃ j₂ hd' r₂ _ hpm,
      tZ_at d _ [j₄] j₄ (by simp), if_pos (show j₄ < d ∧ ∀ q ∈ [j₄], q = j₄ from ⟨r₄, by simp⟩),
      hr _ r₄, tZ_at d _ _ (ρ v) (by simp)]
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, rw, rv,
      true_and, and_true]
    split_ifs <;> ring
  rw [hcan]
  rw [Finset.sum_eq_single (ρ w) (fun j₁ _ hj => Finset.sum_eq_zero fun _ _ =>
      Finset.sum_eq_zero fun _ _ => Finset.sum_eq_zero fun _ _ => Finset.sum_eq_zero fun _ _ => by
        rw [if_neg (fun h => hj h.2), zero_mul]) (fun h => absurd (Finset.mem_range.2 rw) h)]
  rw [Finset.sum_eq_single (ρ w) (fun j₂ _ hj => Finset.sum_eq_zero fun _ _ =>
      Finset.sum_eq_zero fun _ _ => Finset.sum_eq_zero fun _ _ => by
        rw [if_neg (fun h => hj h.1), zero_mul]) (fun h => absurd (Finset.mem_range.2 rw) h)]
  simp only [and_self, if_true, one_mul]
  simp only [← Finset.mul_sum]
  simp only [ite_mul, one_mul, zero_mul]
  rw [sum_pin_shift d jm (ρ w) hd' rw]
  have hjm' : d - jm % d = 1 := by
    have h1 := Nat.mod_lt jm hd'
    have h2 : (jm % d + 1) % d = 0 := by
      rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]; exact hjm
    by_cases h3 : jm % d + 1 < d
    · rw [Nat.mod_eq_of_lt h3] at h2; omega
    · omega
  rw [hjm', Finset.sum_comm]
  simp only [← Finset.sum_mul]
  rw [Finset.sum_congr rfl fun j₅ hj₅ => by rw [wGreen' d _ j₅ (Finset.mem_range.1 hj₅)]]
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq' (range d) (ρ v), if_pos (Finset.mem_range.2 rv)]
  split_ifs <;> ring

/-- A qf_hd chain with its top value summed against the character `k`. -/
noncomputable def hdG (E : Env) (dz : ℕ) (c : HdC) (k w v : ℕ) : ℂ :=
  (rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * (E.dimv dz : ℂ) *
    fou (E.dimv dz) ((c.ws.length * w) % E.dimv dz) k *
    (if (w + 1) % E.dimv dz = 0 then ex (E.phase c.pk v)
      else if (w + 1) % E.dimv dz = v then ex (E.phase c.pk v) else 0)

theorem hdGroup (E : Env) (dz px pz pg pm pr jm : ℕ) (c : HdC) (k : ℕ) (ρ : ℕ → ℕ)
    (hd : 1 < E.dimv dz) (hnd : (c.t :: hdInt c).Nodup) (hout : ∀ x ∈ [c.w, c.v], x ∉ c.t :: hdInt c)
    (hlab : ∀ x ∈ c.t :: hdInt c, labIdx E.labs x = dz) (rw : ρ c.w < E.dimv dz)
    (rv : ρ c.v < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) (hr : ∀ k < E.dimv dz, ex (E.phase pr k) = 1)
    (hpm : ∀ q < E.dimv dz, ex (E.phase pm q) = om (E.dimv dz) ^ (q * jm))
    (hjm : (jm + 1) % E.dimv dz = 0) :
    sumOver (labDim E) (c.t :: hdInt c)
      (fun τ => fou (E.dimv dz) (τ c.t) k * prodVal E (hdNodes dz px pz pg pm pr c) τ) ρ =
      hdG E dz c k (ρ c.w) (ρ c.v) := by
  have hd' : 0 < E.dimv dz := by omega
  obtain ⟨ht, hint⟩ := List.nodup_cons.1 hnd
  have hw := hout c.w (by simp)
  have hv := hout c.v (by simp)
  have ntw : c.t ≠ c.w := fun h => hw (h ▸ List.mem_cons_self)
  have ntv : c.t ≠ c.v := fun h => hv (h ▸ List.mem_cons_self)
  simp only [sumOver, labDim_of E c.t dz (hlab c.t List.mem_cons_self)]
  have e : ∀ j ∈ range (E.dimv dz), sumOver (labDim E) (hdInt c) (fun τ => fou (E.dimv dz) (τ c.t) k *
      prodVal E (hdNodes dz px pz pg pm pr c) τ) (Function.update ρ c.t j) =
      if j = (c.ws.length * ρ c.w) % E.dimv dz then hdG E dz c k (ρ c.w) (ρ c.v) else 0 := by
    intro j hj
    have rj := Finset.mem_range.1 hj
    rw [sumOver_const_mul _ _ [c.t] (fun τ => fou (E.dimv dz) (τ c.t) k) _
      (fun τ τ' h => by simp only; rw [h c.t (by simp)])
      (fun x hx hx' => by simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; subst hx; exact ht hx'),
      hdChainVal E dz px pz pg pm pr jm c _ hd hint (fun x hx hx' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
        rcases hx with rfl | rfl | rfl
        · exact ht hx'
        · exact hw (List.mem_cons_of_mem _ hx')
        · exact hv (List.mem_cons_of_mem _ hx'))
      (fun x hx => hlab x (List.mem_cons_of_mem _ hx))
      (by rw [Function.update_of_ne (Ne.symm ntw)]; exact rw)
      (by rw [Function.update_of_ne (Ne.symm ntv)]; exact rv) hz hx hg hr hpm hjm]
    simp only [Function.update_self, Function.update_of_ne (Ne.symm ntw),
      Function.update_of_ne (Ne.symm ntv), Nat.mod_eq_of_lt rj]
    by_cases h : j = (c.ws.length * ρ c.w) % E.dimv dz
    · rw [if_pos h.symm, if_pos h, hdG, ← h]; ring
    · rw [if_neg (Ne.symm h), if_neg h]; ring
  rw [Finset.sum_congr rfl e, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd'))]

theorem hdNodes_labels (dz px pz pg pm pr : ℕ) (c : HdC) :
    ∀ x ∈ labelsOf (hdNodes dz px pz pg pm pr c), x ∈ c.t :: hdInt c ∨ x = c.w ∨ x = c.v := by
  intro x hx
  simp only [labelsOf, hdNodes, hdInt, List.flatMap_cons, List.flatMap_nil, Node.labels,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false, List.append_nil] at hx ⊢
  aesop

/-- Chains between a pink `K_1` state on a W node, a top pink dot and a bottom pink dot, after
their summed labels: characters `k'` of the top dot and `k` of the bottom one, one chain at
`d - 1`. -/
theorem chainsVal {β : Type*} (E : Env) (dz x₀ i o pk pt pb : ℕ) (cs : List β)
    (ft fw fv : β → ℕ) (I : β → List ℕ) (N : β → List Node) (G : β → ℕ → ℕ → ℕ → ℂ)
    (τ : ℕ → ℕ) (hd : 1 < E.dimv dz)
    (hW : (cs.map fw ++ cs.map fv).Nodup)
    (hQ : (cs.flatMap fun c => ft c :: I c).Nodup)
    (hWQ : ∀ x ∈ cs.map fw ++ cs.map fv, x ∉ cs.flatMap fun c => ft c :: I c)
    (hx₀ : x₀ ∉ (cs.map fw ++ cs.map fv) ++ cs.flatMap fun c => ft c :: I c)
    (hio : ∀ x ∈ [i, o], x ∉ x₀ :: ((cs.map fw ++ cs.map fv) ++
      cs.flatMap fun c => ft c :: I c))
    (hlab : ∀ x ∈ x₀ :: ((cs.map fw ++ cs.map fv) ++ cs.flatMap fun c => ft c :: I c),
      labIdx E.labs x = dz)
    (hk : ∀ q < E.dimv dz, ex (E.phase pk q) = om (E.dimv dz) ^ (q * 1))
    (ht : ∀ q < E.dimv dz, ex (E.phase pt q) = 1) (hb : ∀ q < E.dimv dz, ex (E.phase pb q) = 1)
    (hgrp : ∀ c ∈ cs, ∀ k (ρ : ℕ → ℕ), ρ (fw c) < E.dimv dz → ρ (fv c) < E.dimv dz →
      sumOver (labDim E) (ft c :: I c) (fun τ => fou (E.dimv dz) (τ (ft c)) k * prodVal E (N c) τ) ρ =
        G c k (ρ (fw c)) (ρ (fv c)))
    (hlabels : ∀ c ∈ cs, ∀ x ∈ labelsOf (N c), x ∈ ft c :: I c ∨ x = fw c ∨ x = fv c) :
    sumOver (labDim E) (x₀ :: ((cs.map fw ++ cs.map fv) ++ cs.flatMap fun c => ft c :: I c))
      (prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, cs.map fw, [x₀], dz, 0⟩ ::
        ⟨.X, [o], cs.map fv, dz, pb⟩ :: ⟨.X, cs.map ft, [i], dz, pt⟩ ::
          cs.flatMap N)) τ =
      (rt (E.dimv dz))⁻¹ * E.dimv dz * ∑ k' ∈ range (E.dimv dz),
        (starRingEnd ℂ) (fou (E.dimv dz) (τ i) k') * ∑ k ∈ range (E.dimv dz), fou (E.dimv dz) (τ o) k *
          ((cs.map fw).map fun w => (cs.map fun c => ∑ y ∈ range (E.dimv dz),
            (starRingEnd ℂ) (fou (E.dimv dz) y k) *
              G c k' (if fw c = w then E.dimv dz - 1 else 0) y).prod).sum := by
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  set W := cs.map fw
  set V := cs.map fv
  set Q := cs.flatMap fun c => ft c :: I c
  have hD : ∀ x ∈ x₀ :: ((W ++ V) ++ Q), labDim E x = d := fun x hx => labDim_of E x dz (hlab x hx)
  have hi := hio i (by simp)
  have ho := hio o (by simp)
  have hwW : ∀ c ∈ cs, fw c ∈ W ++ V := fun c hc => List.mem_append_left _ (List.mem_map_of_mem (f := _) hc)
  have hvV : ∀ c ∈ cs, fv c ∈ W ++ V := fun c hc => List.mem_append_right _ (List.mem_map_of_mem (f := _) hc)
  have hQc : ∀ c ∈ cs, ∀ x ∈ ft c :: I c, x ∈ Q := fun c hc x hx => List.mem_flatMap.2 ⟨c, hc, hx⟩
  have hin : ∀ ρ : ℕ → ℕ, (∀ x ∈ W ++ V, ρ x < labDim E x) →
      sumOver (labDim E) Q (prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, W, [x₀], dz, 0⟩ ::
        ⟨.X, [o], V, dz, pb⟩ :: ⟨.X, cs.map ft, [i], dz, pt⟩ ::
          cs.flatMap N)) ρ =
      tX d (E.phase pk) [ρ x₀] [] * (tW d (cs.map fun c => ρ (fw c)) (ρ x₀) *
        (tX d (E.phase pb) [ρ o] (cs.map fun c => ρ (fv c)) *
          ∑ k' ∈ range d, (starRingEnd ℂ) (fou d (ρ i) k') *
            (cs.map fun c => G c k' (ρ (fw c)) (ρ (fv c))).prod)) := by
    intro ρ hrg
    have split : prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, W, [x₀], dz, 0⟩ ::
        ⟨.X, [o], V, dz, pb⟩ :: ⟨.X, cs.map ft, [i], dz, pt⟩ ::
          cs.flatMap N) = fun τ =>
        (tX d (E.phase pk) [τ x₀] [] * (tW d (cs.map fun c => τ (fw c)) (τ x₀) *
          tX d (E.phase pb) [τ o] (cs.map fun c => τ (fv c)))) *
          ∑ k' ∈ range d, (starRingEnd ℂ) (fou d (τ i) k') *
            (cs.map fun c => fou d (τ (ft c)) k' * prodVal E (N c) τ).prod := by
      funext τ
      have htop : tX d (E.phase pt) (cs.map fun c => τ (ft c)) [τ i] = ∑ k' ∈ range d,
          (starRingEnd ℂ) (fou d (τ i) k') * (cs.map fun c => fou d (τ (ft c)) k').prod := by
        unfold tX
        refine Finset.sum_congr rfl fun k hk => ?_
        simp only [List.map_map, Function.comp_def, List.map_cons, List.map_nil, List.prod_cons,
          List.prod_nil, mul_one, ht k (Finset.mem_range.1 hk), one_mul]
        ring
      simp only [prodVal, List.map_cons, List.prod_cons, prod_flatMap', List.map_nil,
        List.map_map, Function.comp_def, W, V]
      simp only [val, List.map_cons, List.map_nil, List.map_map, Function.comp_def]
      rw [← hdd, htop]
      simp only [List.prod_map_mul, Finset.sum_mul, Finset.mul_sum]
      refine Finset.sum_congr rfl fun _ _ => by ring
    rw [split, sumOver_const_mul _ Q (x₀ :: o :: (W ++ V)) _ _ (fun τ τ' h => by
        have hw : (cs.map fun c => τ (fw c)) = cs.map fun c => τ' (fw c) :=
          List.map_congr_left fun c hc => h _ (by simp [hwW c hc])
        have hv : (cs.map fun c => τ (fv c)) = cs.map fun c => τ' (fv c) :=
          List.map_congr_left fun c hc => h _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
            (hvV c hc)))
        rw [hw, hv, h x₀ (by simp), h o (by simp)])
      (fun x hx hxq => by
        simp only [List.mem_cons] at hx
        rcases hx with rfl | rfl | hx
        · exact hx₀ (List.mem_append_right _ hxq)
        · exact ho (List.mem_cons_of_mem _ (List.mem_append_right _ hxq))
        · exact hWQ x hx hxq) ρ, sumOver_fsum]
    have hk' : ∀ k' ∈ range d, sumOver (labDim E) Q (fun τ => (starRingEnd ℂ) (fou d (τ i) k') *
        (cs.map fun c => fou d (τ (ft c)) k' * prodVal E (N c) τ).prod) ρ =
        (starRingEnd ℂ) (fou d (ρ i) k') * (cs.map fun c => G c k' (ρ (fw c)) (ρ (fv c))).prod := by
      intro k' _
      rw [sumOver_const_mul _ Q [i] (fun τ => (starRingEnd ℂ) (fou d (τ i) k')) _
        (fun τ τ' h => by simp only; rw [h i (by simp)])
        (fun x hx hxq => by
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; subst hx
          exact hi (List.mem_cons_of_mem _ (List.mem_append_right _ hxq))) ρ]
      congr 1
      have hgr := sumOver_groups (D := labDim E) (cs.map fun c => (ft c :: I c,
        fun τ => fou d (τ (ft c)) k' * prodVal E (N c) τ)) (W ++ V)
      simp only [List.flatMap_map, List.map_map, Function.comp_def] at hgr
      rw [hgr hQ (fun x hx hx' => hWQ x hx hx')
        (fun gg hgg => by
          obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hgg
          intro τ τ' h
          simp only
          rw [prodVal_reads E (N c) τ τ' fun x hx => h x (by
            rcases hlabels c hc x hx with h1 | h1 | h1
            · exact List.mem_append_left _ h1
            · exact List.mem_append_right _ (h1 ▸ hwW c hc)
            · exact List.mem_append_right _ (h1 ▸ hvV c hc)),
            h (ft c) (List.mem_append_left _ List.mem_cons_self)]) ρ]
      refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
      have rw' : ρ (fw c) < d := by have := hrg _ (hwW c hc); rwa [hD _ (List.mem_cons_of_mem _
        (List.mem_append_left _ (hwW c hc)))] at this
      have rv' : ρ (fv c) < d := by have := hrg _ (hvV c hc); rwa [hD _ (List.mem_cons_of_mem _
        (List.mem_append_left _ (hvV c hc)))] at this
      exact hgrp c hc k' ρ rw' rv'
    rw [Finset.sum_congr rfl hk']
    ring
  have hj : ∀ j ∈ range d, sumOver (labDim E) ((W ++ V) ++ Q) (prodVal E (⟨.X, [x₀], [], dz, pk⟩ ::
      ⟨.W, W, [x₀], dz, 0⟩ :: ⟨.X, [o], V, dz, pb⟩ :: ⟨.X, cs.map ft, [i], dz, pt⟩ ::
        cs.flatMap N)) (Function.update τ x₀ j) =
      tX d (E.phase pk) [j] [] * ∑ k' ∈ range d, (starRingEnd ℂ) (fou d (τ i) k') *
        ∑ k ∈ range d, fou d (τ o) k * (if j = 0 then (cs.map fun c => ∑ y ∈ range d,
          (starRingEnd ℂ) (fou d y k) * G c k' 0 y).prod
        else (W.map fun w => (cs.map fun c => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
          G c k' (if fw c = w then j else 0) y).prod).sum) := by
    intro j hj
    have rj := Finset.mem_range.1 hj
    have ei : Function.update τ x₀ j i = τ i :=
      Function.update_of_ne (fun (h : i = x₀) => hi (by rw [h]; exact List.mem_cons_self)) _ _
    have eo : Function.update τ x₀ j o = τ o :=
      Function.update_of_ne (fun (h : o = x₀) => ho (by rw [h]; exact List.mem_cons_self)) _ _
    rw [sumOver_append, sumOver_congr _ _ _ _ _ fun ρ _ hr => hin ρ hr]
    rw [sumOver_const_mul _ (W ++ V) [x₀] (fun ρ => tX d (E.phase pk) [ρ x₀] []) _
      (fun τ τ' h => by simp only; rw [h x₀ (by simp)])
      (fun x hx hxq => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; subst hx
        exact hx₀ (List.mem_append_left _ hxq))]
    simp only [Function.update_self]
    congr 1
    rw [show (fun ρ : ℕ → ℕ => tW d (cs.map fun c => ρ (fw c)) (ρ x₀) *
        (tX d (E.phase pb) [ρ o] (cs.map fun c => ρ (fv c)) * ∑ k' ∈ range d,
          (starRingEnd ℂ) (fou d (ρ i) k') * (cs.map fun c => G c k' (ρ (fw c)) (ρ (fv c))).prod)) =
        fun ρ => ∑ k' ∈ range d, (starRingEnd ℂ) (fou d (ρ i) k') *
          (tW d (cs.map fun c => ρ (fw c)) (ρ x₀) * (tX d (E.phase pb) [ρ o]
            (cs.map fun c => ρ (fv c)) * (cs.map fun c => G c k' (ρ (fw c)) (ρ (fv c))).prod)) from
      funext fun ρ => by rw [Finset.mul_sum, Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring]
    rw [sumOver_fsum]
    refine Finset.sum_congr rfl fun k' _ => ?_
    rw [sumOver_const_mul _ (W ++ V) [i] (fun ρ => (starRingEnd ℂ) (fou d (ρ i) k')) _
      (fun τ τ' h => by simp only; rw [h i (by simp)])
      (fun x hx hxq => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hx; subst hx
        exact hi (List.mem_cons_of_mem _ (List.mem_append_left _ hxq))), ei]
    congr 1
    rw [wPinkSum (D := labDim E) d hd' x₀ o cs fw fv (fun c w v => G c k' w v)
      (E.phase pb) hb _ hW (fun x hx => hD x (List.mem_cons_of_mem _ (List.mem_append_left _ hx)))
      (fun h => hx₀ (List.mem_append_left _ h))
      (fun h => ho (List.mem_cons_of_mem _ (List.mem_append_left _ h)))
      (by rw [Function.update_self]; exact rj), eo, Function.update_self]
  rw [show sumOver (labDim E) (x₀ :: ((W ++ V) ++ Q)) (prodVal E (⟨.X, [x₀], [], dz, pk⟩ ::
      ⟨.W, W, [x₀], dz, 0⟩ :: ⟨.X, [o], V, dz, pb⟩ :: ⟨.X, cs.map ft, [i], dz, pt⟩ ::
        cs.flatMap N)) τ = ∑ j ∈ range d, sumOver (labDim E)
          ((W ++ V) ++ Q) (prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, W, [x₀], dz, 0⟩ ::
            ⟨.X, [o], V, dz, pb⟩ :: ⟨.X, cs.map ft, [i], dz, pt⟩ ::
              cs.flatMap N)) (Function.update τ x₀ j) by
    simp only [sumOver, hD x₀ List.mem_cons_self]]
  rw [Finset.sum_congr rfl hj]
  simp only [tX_kj d 1 hd' _ hk, List.length_singleton, List.length_nil, List.sum_singleton,
    List.sum_nil, Nat.zero_mod, zero_add, pow_one]
  rw [Finset.sum_congr rfl fun j _ => (show _ = if (j + 1) % d = 0 then (rt d)⁻¹ * d *
      ∑ k' ∈ range d, (starRingEnd ℂ) (fou d (τ i) k') * ∑ k ∈ range d, fou d (τ o) k *
        (if j = 0 then (cs.map fun c => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
          G c k' 0 y).prod
        else (W.map fun w => (cs.map fun c => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
          G c k' (if fw c = w then j else 0) y).prod).sum) else 0 by split_ifs <;> ring),
    sum_pin_shift d 1 0 hd' hd' _, Nat.mod_eq_of_lt hd, zero_add,
    Nat.mod_eq_of_lt (show d - 1 < d by omega)]
  have h1 : d - 1 ≠ 0 := by omega
  simp only [h1, if_false]


/-- The character sums of `chainsVal` when each chain at `0` is `A c * h k` and at `d - 1` picks
the character `J c`. -/
theorem chainsClosed {β : Type*} [DecidableEq β] (d : ℕ) (cs : List β) (fw J : β → ℕ)
    (G : β → ℕ → ℕ → ℕ → ℂ) (A : β → ℂ) (h φ : ℕ → ℂ) (a b : ℕ) (hd : 1 < d) (ha : a < d)
    (hb : b < d) (hwnd : (cs.map fw).Nodup) (hJ : ∀ c ∈ cs, J c < d)
    (hH : ∀ c ∈ cs, ∀ k < d, ∀ k', ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) * G c k' 0 y =
      A c * h k)
    (hF : ∀ c ∈ cs, ∀ k < d, ∀ k', ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) * G c k' (d - 1) y =
      A c * (fou d (J c) k' * rt d * (if k = J c then 1 else 0) * φ (J c)))
    (hcount : ∀ x < d, (cs.filter fun c => decide (x = J c)).length = 1) :
    ∑ k' ∈ range d, (starRingEnd ℂ) (fou d a k') * ∑ k ∈ range d, fou d b k *
        ((cs.map fw).map fun w => (cs.map fun c => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
          G c k' (if fw c = w then d - 1 else 0) y).prod).sum =
      (cs.map A).prod * rt d * φ a * h a ^ (cs.length - 1) * fou d b a := by
  have hd' : 0 < d := by omega
  set f : β → ℕ → ℕ → ℂ := fun c k k' => fou d (J c) k' * rt d *
    (if k = J c then 1 else 0) * φ (J c) with hf
  have hprod : ∀ k < d, ∀ k', ∀ c' ∈ cs, (cs.map fun c => ∑ y ∈ range d,
      (starRingEnd ℂ) (fou d y k) * G c k' (if fw c = fw c' then d - 1 else 0) y).prod =
      (cs.map A).prod * (f c' k k' * h k ^ (cs.length - 1)) := by
    intro k hk k' c' hc'
    rw [show (cs.map fun c => ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
        G c k' (if fw c = fw c' then d - 1 else 0) y) =
        cs.map fun c => A c * (if fw c = fw c' then f c k k' else h k) from
      List.map_congr_left fun c hc => by
        split_ifs
        · exact hF c hc k hk k'
        · exact hH c hc k hk k', List.prod_map_mul]
    congr 1
    rw [prod_one_hot cs fw hwnd c' hc' _ (h k) (fun b _ hb => if_neg hb), if_pos rfl]
  have e1 : ∀ k', ∀ k ∈ range d, fou d b k * ((cs.map fw).map fun w => (cs.map fun c =>
      ∑ y ∈ range d, (starRingEnd ℂ) (fou d y k) *
        G c k' (if fw c = w then d - 1 else 0) y).prod).sum =
      (cs.map fun c' => fou d b k * ((cs.map A).prod * (f c' k k' * h k ^ (cs.length - 1)))).sum := by
    intro k' k hk
    rw [List.map_map, ← List.sum_map_mul_left]
    exact congrArg List.sum (List.map_congr_left fun c' hc' => by
      simp only [Function.comp_apply]; rw [hprod k (Finset.mem_range.1 hk) k' c' hc'])
  rw [Finset.sum_congr rfl fun k' _ => congrArg (fun z => (starRingEnd ℂ) (fou d a k') * z)
    (Finset.sum_congr rfl (e1 k'))]
  simp only [sum_list_comm]
  set Gs : ℕ → ℂ := fun x => fou d b x * (cs.map A).prod * rt d * φ x *
    h x ^ (cs.length - 1) with hG
  have e2 : ∀ k', ∀ c' ∈ cs, ∑ k ∈ range d, fou d b k * ((cs.map A).prod *
      (f c' k k' * h k ^ (cs.length - 1))) = fou d (J c') k' * Gs (J c') := by
    intro k' c' hc'
    have e : ∀ k, fou d b k * ((cs.map A).prod * (f c' k k' * h k ^ (cs.length - 1))) =
        if k = J c' then fou d (J c') k' * Gs (J c') else 0 := by
      intro k
      simp only [hf, hG]
      split_ifs with hk
      · rw [hk]; ring
      · ring
    rw [Finset.sum_congr rfl fun k _ => e k, Finset.sum_ite_eq',
      if_pos (Finset.mem_range.2 (hJ c' hc'))]
  rw [Finset.sum_congr rfl fun k' _ => congrArg (fun z => (starRingEnd ℂ) (fou d a k') * z)
    (congrArg List.sum (List.map_congr_left (e2 k')))]
  simp only [← List.sum_map_mul_left]
  rw [sum_list_comm]
  have e3 : ∀ c ∈ cs, ∑ k ∈ range d, (starRingEnd ℂ) (fou d a k) * (fou d (J c) k * Gs (J c)) =
      (if a = J c then 1 else 0) * Gs a := by
    intro c hc
    rw [Finset.sum_congr rfl fun k _ => (show (starRingEnd ℂ) (fou d a k) * (fou d (J c) k *
        Gs (J c)) = fou d k (J c) * (starRingEnd ℂ) (fou d k a) * Gs (J c) by
      rw [fou_symm d a k, fou_symm d (J c) k]; ring), ← Finset.sum_mul,
      fou_unitary d hd' (J c) a (hJ c hc) ha]
    by_cases h' : a = J c
    · rw [if_pos h'.symm, if_pos h', h']
    · rw [if_neg (Ne.symm h'), if_neg h', zero_mul, zero_mul]
  rw [List.map_congr_left e3, List.sum_map_mul_right, sum_ind_count, hcount a ha]
  simp only [Nat.cast_one, one_mul, hG]
  ring

/-- The qf_hd left side after its summed labels: characters `k'` of the top pink dot and `k` of
the bottom one, and the chain at `d - 1`. -/
theorem hdVal (E : Env) (dz x₀ i o pk pt pb px pz pg pm pr jm : ℕ) (cs : List HdC) (τ : ℕ → ℕ)
    (hd : 1 < E.dimv dz)
    (hW : (cs.map HdC.w ++ cs.map HdC.v).Nodup)
    (hQ : (cs.flatMap fun c => c.t :: hdInt c).Nodup)
    (hWQ : ∀ x ∈ cs.map HdC.w ++ cs.map HdC.v, x ∉ cs.flatMap fun c => c.t :: hdInt c)
    (hx₀ : x₀ ∉ (cs.map HdC.w ++ cs.map HdC.v) ++ cs.flatMap fun c => c.t :: hdInt c)
    (hio : ∀ x ∈ [i, o], x ∉ x₀ :: ((cs.map HdC.w ++ cs.map HdC.v) ++
      cs.flatMap fun c => c.t :: hdInt c))
    (hlab : ∀ x ∈ x₀ :: ((cs.map HdC.w ++ cs.map HdC.v) ++ cs.flatMap fun c => c.t :: hdInt c),
      labIdx E.labs x = dz)
    (hk : ∀ q < E.dimv dz, ex (E.phase pk q) = om (E.dimv dz) ^ (q * 1))
    (ht : ∀ q < E.dimv dz, ex (E.phase pt q) = 1) (hb : ∀ q < E.dimv dz, ex (E.phase pb q) = 1)
    (hz : ∀ q < E.dimv dz, ex (E.phase pz q) = 1) (hx : ∀ q < E.dimv dz, ex (E.phase px q) = 1)
    (hg : ∀ q < E.dimv dz, ex (E.phase pg q) = 1) (hr : ∀ q < E.dimv dz, ex (E.phase pr q) = 1)
    (hpm : ∀ q < E.dimv dz, ex (E.phase pm q) = om (E.dimv dz) ^ (q * jm))
    (hjm : (jm + 1) % E.dimv dz = 0) :
    sumOver (labDim E) (x₀ :: ((cs.map HdC.w ++ cs.map HdC.v) ++ cs.flatMap fun c => c.t :: hdInt c))
      (prodVal E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, cs.map HdC.w, [x₀], dz, 0⟩ ::
        ⟨.X, [o], cs.map HdC.v, dz, pb⟩ :: ⟨.X, cs.map HdC.t, [i], dz, pt⟩ ::
          cs.flatMap (hdNodes dz px pz pg pm pr))) τ =
      (rt (E.dimv dz))⁻¹ * E.dimv dz * ∑ k' ∈ range (E.dimv dz),
        (starRingEnd ℂ) (fou (E.dimv dz) (τ i) k') * ∑ k ∈ range (E.dimv dz), fou (E.dimv dz) (τ o) k *
          ((cs.map HdC.w).map fun w => (cs.map fun c => ∑ y ∈ range (E.dimv dz),
            (starRingEnd ℂ) (fou (E.dimv dz) y k) *
              hdG E dz c k' (if c.w = w then E.dimv dz - 1 else 0) y).prod).sum := by
  have hQc : ∀ c ∈ cs, ∀ y ∈ c.t :: hdInt c, y ∈ cs.flatMap fun c => c.t :: hdInt c :=
    fun c hc y hy => List.mem_flatMap.2 ⟨c, hc, hy⟩
  exact chainsVal E dz x₀ i o pk pt pb cs HdC.t HdC.w HdC.v hdInt (hdNodes dz px pz pg pm pr)
    (hdG E dz) τ hd hW hQ hWQ hx₀ hio hlab hk ht hb
    (fun c hc k ρ rw rv => hdGroup E dz px pz pg pm pr jm c k ρ hd
      ((List.nodup_flatMap.1 hQ).1 c hc)
      (fun y hy hy' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
        rcases hy with rfl | rfl
        · exact hWQ _ (List.mem_append_left _ (List.mem_map_of_mem hc)) (hQc c hc _ hy')
        · exact hWQ _ (List.mem_append_right _ (List.mem_map_of_mem hc)) (hQc c hc _ hy'))
      (fun y hy => hlab y (List.mem_cons_of_mem _ (List.mem_append_right _ (hQc c hc y hy))))
      rw rv hz hx hg hr hpm hjm)
    (fun c _ => hdNodes_labels dz px pz pg pm pr c)

theorem prod_om_pow {β : Type*} (d : ℕ) (cs : List β) (j : β → ℕ) :
    (cs.map fun c => om d ^ j c).prod = om d ^ (cs.map j).sum := by
  induction cs with
  | nil => simp
  | cons c cs ih => simp [ih, pow_add]

theorem hdAt0 (E : Env) (dz : ℕ) (c : HdC) (k k' : ℕ) (hd : 1 < E.dimv dz)
    (hK : ∀ q < E.dimv dz, ex (E.phase c.pk q) = om (E.dimv dz) ^
      (q * ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz))) :
    ∑ y ∈ range (E.dimv dz), (starRingEnd ℂ) (fou (E.dimv dz) y k) * hdG E dz c k' 0 y =
      ((rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * E.dimv dz *
        om (E.dimv dz) ^ ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz)) *
        ((rt (E.dimv dz))⁻¹ * (starRingEnd ℂ) (fou (E.dimv dz) 1 k)) := by
  set d := E.dimv dz with hdd
  have h1 : 1 % d = 1 := Nat.mod_eq_of_lt hd
  simp only [hdG, ← hdd, mul_zero, Nat.zero_mod, zero_add, h1, one_ne_zero, if_false, fou_zero']
  rw [Finset.sum_eq_single 1 (fun y _ hy => by rw [if_neg (Ne.symm hy), mul_zero, mul_zero])
    (fun h => absurd (Finset.mem_range.2 hd) h), if_pos rfl, hK 1 hd, one_mul]
  ring

theorem hdAtTop (E : Env) (dz : ℕ) (c : HdC) (k k' : ℕ) (hd : 1 < E.dimv dz) (hk : k < E.dimv dz)
    (hK : ∀ q < E.dimv dz, ex (E.phase c.pk q) = om (E.dimv dz) ^
      (q * ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz))) :
    ∑ y ∈ range (E.dimv dz), (starRingEnd ℂ) (fou (E.dimv dz) y k) *
        hdG E dz c k' (E.dimv dz - 1) y =
      ((rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * E.dimv dz *
        om (E.dimv dz) ^ ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz)) *
        (fou (E.dimv dz) ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz) k' * rt (E.dimv dz) *
          (if k = (c.ws.length * (E.dimv dz - 1)) % E.dimv dz then 1 else 0) *
          (om (E.dimv dz) ^ ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz))⁻¹) := by
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  set j := (c.ws.length * (d - 1)) % d with hj
  have hjd : j < d := Nat.mod_lt _ hd'
  have hz : (d - 1 + 1) % d = 0 := by rw [show d - 1 + 1 = d by omega, Nat.mod_self]
  simp only [hdG, ← hdd, ← hj, hz, if_true]
  have e : ∀ y ∈ range d, (starRingEnd ℂ) (fou d y k) * ((rt d)⁻¹ ^ (c.ws.length + 1) * d *
      fou d j k' * ex (E.phase c.pk y)) = ((rt d)⁻¹ ^ (c.ws.length + 1) * d * fou d j k' * rt d) *
        (fou d y j * (starRingEnd ℂ) (fou d y k)) := by
    intro y hy
    rw [hK y (Finset.mem_range.1 hy)]
    have hr := rt_ne d hd'
    simp only [fou, rt]
    field_simp
  rw [Finset.sum_congr rfl e, ← Finset.mul_sum, fou_unitary d hd' j k hjd hk]
  have hom : om d ^ j ≠ 0 := pow_ne_zero _ (om_ne_zero d hd')
  by_cases h : k = j
  · rw [if_pos h.symm, if_pos h]; field_simp
  · rw [if_neg (Ne.symm h), if_neg h]; ring

/-- The qf_hd character sums: one chain at `d - 1`, the others at `0`, as the Fourier entry. -/
theorem hdClosed (E : Env) (dz : ℕ) (cs : List HdC) (a b : ℕ) (hd : 1 < E.dimv dz)
    (ha : a < E.dimv dz) (hb : b < E.dimv dz) (hwnd : (cs.map HdC.w).Nodup)
    (hK : ∀ c ∈ cs, ∀ q < E.dimv dz, ex (E.phase c.pk q) = om (E.dimv dz) ^
      (q * ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz)))
    (hcount : ∀ x < E.dimv dz, (cs.filter fun c => decide (x = (c.ws.length * (E.dimv dz - 1)) %
      E.dimv dz)).length = 1)
    (hn : cs.length = E.dimv dz) :
    ∑ k' ∈ range (E.dimv dz), (starRingEnd ℂ) (fou (E.dimv dz) a k') *
      ∑ k ∈ range (E.dimv dz), fou (E.dimv dz) b k *
        ((cs.map HdC.w).map fun w => (cs.map fun c => ∑ y ∈ range (E.dimv dz),
          (starRingEnd ℂ) (fou (E.dimv dz) y k) *
            hdG E dz c k' (if c.w = w then E.dimv dz - 1 else 0) y).prod).sum =
      (cs.map fun c => (rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * E.dimv dz).prod *
        om (E.dimv dz) ^ (cs.map fun c => (c.ws.length * (E.dimv dz - 1)) % E.dimv dz).sum *
        rt (E.dimv dz) * (rt (E.dimv dz))⁻¹ ^ (2 * (E.dimv dz - 1)) *
        fou (E.dimv dz) b a := by
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  rw [chainsClosed d cs HdC.w (fun c => (c.ws.length * (d - 1)) % d) (hdG E dz)
    (fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * d * om d ^ ((c.ws.length * (d - 1)) % d))
    (fun k => (rt d)⁻¹ * (starRingEnd ℂ) (fou d 1 k)) (fun j => (om d ^ j)⁻¹) a b hd ha hb hwnd
    (fun c _ => Nat.mod_lt _ hd')
    (fun c hc k _ k' => hdAt0 E dz c k k' hd (hK c hc))
    (fun c hc k hk k' => hdAtTop E dz c k k' hd hk (hK c hc)) hcount]
  simp only [List.prod_map_mul, prod_om_pow]
  have hr := rt_ne d hd'
  have hom : om d ^ a ≠ 0 := pow_ne_zero _ (om_ne_zero d hd')
  have hc : (starRingEnd ℂ) (fou d 1 a) = (om d ^ a)⁻¹ * (rt d)⁻¹ := by
    simp only [fou, one_mul, map_div₀, map_pow, conj_om, Complex.conj_ofReal, inv_pow]
    simp only [rt, div_eq_mul_inv]
  rw [hc, hn]
  obtain ⟨m, hm⟩ : ∃ m, d = m + 1 := ⟨d - 1, by omega⟩
  have hd1 : om d ^ a * ((om d ^ a) ^ m) = 1 := by
    rw [← pow_succ', ← pow_mul, mul_comm, pow_mul, ← hm, om_pow_self d hd', one_pow]
  rw [show d - 1 = m by omega]
  have e4 : ((rt d)⁻¹ * ((om d ^ a)⁻¹ * (rt d)⁻¹)) ^ m = (rt d)⁻¹ ^ (2 * m) * ((om d ^ a) ^ m)⁻¹ := by
    rw [mul_pow, mul_pow, inv_pow (om d ^ a), pow_mul, sq]; ring
  rw [e4]
  have hm' : (om d ^ a) ^ m = (om d ^ a)⁻¹ := eq_inv_of_mul_eq_one_left (by rw [mul_comm]; exact hd1)
  rw [hm', inv_inv]
  field_simp

theorem om_tri (d : ℕ) (hd : 0 < d) : om d ^ (d * (d - 1) / 2) = (-1) ^ (d + 1) := by
  obtain ⟨m, hm | hm⟩ := Nat.even_or_odd' d
  · subst hm
    have hm0 : 0 < m := by omega
    have e : 2 * m * (2 * m - 1) / 2 = m * (2 * m - 1) := by
      rw [mul_assoc, Nat.mul_div_cancel_left _ (by norm_num)]
    have hhalf : om (2 * m) ^ m = -1 := by
      rw [om, ← Complex.exp_nat_mul]
      have hm' : ((m : ℕ) : ℂ) ≠ 0 := by exact_mod_cast hm0.ne'
      rw [show (m : ℂ) * (2 * Real.pi * Complex.I / ((2 * m : ℕ) : ℂ)) = Real.pi * Complex.I by
        push_cast; field_simp]
      exact Complex.exp_pi_mul_I
    have h2 : om (2 * m) ^ (m * (2 * m - 1)) * om (2 * m) ^ m = 1 := by
      rw [← pow_add, show m * (2 * m - 1) + m = (2 * m) * m by
        rw [Nat.mul_sub, mul_one]; have : m ≤ m * (2 * m) := Nat.le_mul_of_pos_right m (by omega)
        rw [mul_comm m (2 * m)] at this ⊢; omega]
      exact om_pow_zero_mod (2 * m) hd _ _ (Nat.mod_self _)
    rw [e, hhalf] at *
    rw [show (-1 : ℂ) ^ (2 * m + 1) = -1 by rw [pow_succ, pow_mul]; norm_num]
    linear_combination -h2
  · subst hm
    have e : (2 * m + 1) * (2 * m + 1 - 1) / 2 = (2 * m + 1) * m := by
      rw [show 2 * m + 1 - 1 = 2 * m by omega, mul_comm 2 m, ← mul_assoc,
        Nat.mul_div_cancel _ (by norm_num)]
    rw [e, om_pow_zero_mod (2 * m + 1) hd _ _ (Nat.mod_self _),
      show 2 * m + 1 + 1 = 2 * (m + 1) by ring, pow_mul]
    norm_num

/-- [P2] (HD): the Hadamard box decomposition, the Fourier box times `s`. -/
theorem qfHD (E : Env) (dz x₀ i o pk pt pb px pz pg pm pr jm : ℕ) (cs : List HdC) (iface : List ℕ)
    (hd : 1 < E.dimv dz)
    (hW : (cs.map HdC.w ++ cs.map HdC.v).Nodup)
    (hQ : (cs.flatMap fun c => c.t :: hdInt c).Nodup)
    (hWQ : ∀ x ∈ cs.map HdC.w ++ cs.map HdC.v, x ∉ cs.flatMap fun c => c.t :: hdInt c)
    (hx₀ : x₀ ∉ (cs.map HdC.w ++ cs.map HdC.v) ++ cs.flatMap fun c => c.t :: hdInt c)
    (hio : ∀ x ∈ [i, o], x ∉ x₀ :: ((cs.map HdC.w ++ cs.map HdC.v) ++
      cs.flatMap fun c => c.t :: hdInt c))
    (hlab : ∀ x ∈ x₀ :: ((cs.map HdC.w ++ cs.map HdC.v) ++ cs.flatMap fun c => c.t :: hdInt c),
      labIdx E.labs x = dz)
    (hif : ∀ x ∈ [i, o], x ∈ iface ∧ labIdx E.labs x = dz)
    (hk : ∀ q < E.dimv dz, ex (E.phase pk q) = om (E.dimv dz) ^ (q * 1))
    (hph : ∀ ph ∈ [pt, pb, px, pz, pg, pr], ∀ q < E.dimv dz, ex (E.phase ph q) = 1)
    (hpm : ∀ q < E.dimv dz, ex (E.phase pm q) = om (E.dimv dz) ^ (q * jm))
    (hjm : (jm + 1) % E.dimv dz = 0)
    (hK : ∀ c ∈ cs, ∀ q < E.dimv dz, ex (E.phase c.pk q) = om (E.dimv dz) ^
      (q * ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz)))
    (hcount : ∀ x < E.dimv dz, (cs.filter fun c => decide (x = (c.ws.length * (E.dimv dz - 1)) %
      E.dimv dz)).length = 1)
    (hn : cs.length = E.dimv dz)
    (hS : (cs.map fun c => (c.ws.length * (E.dimv dz - 1)) % E.dimv dz).sum =
      E.dimv dz * (E.dimv dz - 1) / 2)
    (s : ℂ) (hs : s * rt (E.dimv dz) ^ (E.dimv dz + (cs.map fun c => c.ws.length).sum) =
      (-1) ^ (E.dimv dz + 1) * rt (E.dimv dz) ^ 4) :
    LocalEq E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, cs.map HdC.w, [x₀], dz, 0⟩ ::
        ⟨.X, [o], cs.map HdC.v, dz, pb⟩ :: ⟨.X, cs.map HdC.t, [i], dz, pt⟩ ::
          cs.flatMap (hdNodes dz px pz pg pm pr))
      (x₀ :: ((cs.map HdC.w ++ cs.map HdC.v) ++ cs.flatMap fun c => c.t :: hdInt c))
      [⟨.F, [o], [i], dz, 0⟩] [] iface 1 s := by
  intro τ hτ
  have ri := hτ i (hif i (by simp)).1
  have ro := hτ o (hif o (by simp)).1
  rw [labDim_of E i dz (hif i (by simp)).2] at ri
  rw [labDim_of E o dz (hif o (by simp)).2] at ro
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨ht, hb, hx, hz, hg, hr⟩ := hph'
  rw [one_mul, hdVal E dz x₀ i o pk pt pb px pz pg pm pr jm cs τ hd hW hQ hWQ hx₀ hio hlab hk ht hb
    hz hx hg hr hpm hjm, hdClosed E dz cs (τ i) (τ o) hd ri ro (List.nodup_append.1 hW).1 hK
    hcount hn, hS, om_tri _ (by omega)]
  simp only [sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, val]
  set d := E.dimv dz with hdd
  have hr := rt_ne d (by omega)
  have hA := kz_const (rt d) hr cs fun c => c.ws.length
  rw [hn] at hA
  have hdr : (d : ℂ) = rt d ^ 2 := (rt_sq' d).symm
  suffices hsc : (rt d)⁻¹ * d * (cs.map fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * d).prod *
      (-1) ^ (d + 1) * rt d * (rt d)⁻¹ ^ (2 * (d - 1)) = s by rw [← hsc]; ring
  apply mul_right_cancel₀ (pow_ne_zero (d + (cs.map fun c => c.ws.length).sum) hr)
  rw [hs, hdr, pow_add (rt d) d]
  obtain ⟨m, hm⟩ : ∃ m, d = m + 1 := ⟨d - 1, by omega⟩
  calc (rt d)⁻¹ * rt d ^ 2 * (cs.map fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * rt d ^ 2).prod *
        (-1) ^ (d + 1) * rt d * (rt d)⁻¹ ^ (2 * (d - 1)) *
        (rt d ^ d * rt d ^ (cs.map fun c => c.ws.length).sum)
      = (rt d)⁻¹ * rt d ^ 2 * ((cs.map fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * rt d ^ 2).prod *
        rt d ^ (cs.map fun c => c.ws.length).sum) * (-1) ^ (d + 1) * rt d *
        (rt d)⁻¹ ^ (2 * (d - 1)) * rt d ^ d := by ring
    _ = (-1) ^ (d + 1) * rt d ^ 4 := by
      rw [hA, show d - 1 = m by omega, hm, inv_pow]
      field_simp
      ring


/-- A zxw_hd chain: top value `t`, multiplier wires `ws`, the green dot on `m, w, a`, the M box
(`box` from `a` to `b`, multipliers `ms`, output `x`) and `K_c` (`x -> v`). -/
structure ZhdC where
  t : ℕ
  ws : List ℕ
  m : ℕ
  w : ℕ
  a : ℕ
  box : Node
  b : ℕ
  ms : List (ℕ × List ℕ × ℕ)
  x : ℕ
  v : ℕ
  pk : ℕ
  deriving DecidableEq

/-- The nodes of a zxw_hd chain. -/
def zhdNodes (dz px pz pg pq₂ pz₂ px₂ : ℕ) (c : ZhdC) : List Node :=
  [⟨.X, c.ws, [c.t], dz, px⟩, ⟨.Z, [c.m], c.ws, dz, pz⟩, ⟨.Z, [], [c.m, c.w, c.a], dz, pg⟩] ++
    vNodes dz c.b c.x pq₂ pz₂ px₂ c.box c.ms ++ [⟨.Z, [c.v], [c.x], dz, c.pk⟩]

/-- The summed labels of a zxw_hd chain. -/
def zhdInt (c : ZhdC) : List ℕ := c.m :: c.a :: c.x :: (c.ws ++ vLabs c.b c.ms)

theorem zhdChainVal (E : Env) (dz px pz pg pq₂ pz₂ px₂ : ℕ) (c : ZhdC) (ρ : ℕ → ℕ)
    (hd : 1 < E.dimv dz) (hnd : (zhdInt c).Nodup)
    (hout : ∀ y ∈ [c.t, c.w, c.v], y ∉ zhdInt c)
    (hlab : ∀ y ∈ zhdInt c, labIdx E.labs y = dz) (hbz : isZc c.box = true)
    (hbd : c.box.dim = dz) (hbl : c.box.outs ++ c.box.ins = [c.b, c.a])
    (rw : ρ c.w < E.dimv dz) (rv : ρ c.v < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) (hq₂ : ∀ k < E.dimv dz, ex (E.phase pq₂ k) = 1)
    (hz₂ : ∀ k < E.dimv dz, ex (E.phase pz₂ k) = 1)
    (hx₂ : ∀ k < E.dimv dz, ex (E.phase px₂ k) = 1) :
    sumOver (labDim E) (zhdInt c) (prodVal E (zhdNodes dz px pz pg pq₂ pz₂ px₂ c)) ρ =
      ((rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) *
        (if (c.ws.length * ρ c.w) % E.dimv dz = ρ c.t % E.dimv dz then (E.dimv dz : ℂ) else 0)) *
      (zc E c.box (ρ c.w) * vConst (E.dimv dz) c.ms *
        (if ρ c.w = 0 then (if ρ c.v = 0 then 1 else 0)
          else (c.ms.map fun p => if ρ c.v = (p.2.1.length * ρ c.w) % E.dimv dz then (1 : ℂ)
            else 0).sum)) * ex (E.phase c.pk (ρ c.v)) := by
  obtain ⟨t, ws, m, w, a, box, b, ms, x, v, pk⟩ := c
  simp only [zhdInt, zhdNodes] at hnd hout hlab hbz hbd hbl rw rv ⊢
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  simp only [List.nodup_cons, List.mem_cons, List.mem_append, not_or] at hnd
  obtain ⟨⟨nma, nmx, nmws, nmv⟩, ⟨nax, naws, nav⟩, ⟨nxws, nxv⟩, hwv⟩ := hnd
  obtain ⟨hws, hvl, hwvl⟩ := List.nodup_append.1 hwv
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
    List.mem_append, not_or] at hout
  obtain ⟨⟨ntm, nta, ntx, ntws, ntv⟩, ⟨nwm, nwa, nwx, nwws, nwv⟩,
    ⟨nvm, nva, nvx, nvws, nvv⟩⟩ := hout
  have hl : ∀ q ∈ [m, a, x], labDim E q = d := by
    intro q hq
    refine labDim_of E q dz (hlab q ?_)
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq ⊢
    rcases hq with h | h | h <;> simp [h]
  have hcan : sumOver (labDim E) (m :: a :: x :: (ws ++ vLabs b ms)) (prodVal E
      ([⟨.X, ws, [t], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩, ⟨.Z, [], [m, w, a], dz, pg⟩] ++
        vNodes dz b x pq₂ pz₂ px₂ box ms ++ [⟨.Z, [v], [x], dz, pk⟩])) ρ =
      ∑ j₁ ∈ range d, ∑ j₂ ∈ range d, ∑ j₃ ∈ range d,
        (if j₁ = ρ w ∧ j₂ = ρ w then 1 else 0) * ((if j₃ = ρ v then ex (E.phase pk (ρ v)) else 0) *
          (((rt d)⁻¹ ^ (ws.length + 1) * (if (ws.length * j₁) % d = ρ t % d then (d : ℂ) else 0)) *
            (zc E box j₂ * vConst d ms * (if j₂ = 0 then (if j₃ = 0 then 1 else 0)
              else (ms.map fun p => if j₃ = (p.2.1.length * j₂) % d then (1 : ℂ) else 0).sum)))) := by
    simp only [sumOver, hl m (by simp), hl a (by simp), hl x (by simp)]
    refine Finset.sum_congr rfl fun j₁ h₁ => Finset.sum_congr rfl fun j₂ h₂ =>
      Finset.sum_congr rfl fun j₃ h₃ => ?_
    have r₁ := Finset.mem_range.1 h₁
    have r₂ := Finset.mem_range.1 h₂
    have r₃ := Finset.mem_range.1 h₃
    set σ := Function.update (Function.update (Function.update ρ m j₁) a j₂) x j₃ with hσ
    have split : prodVal E ([⟨.X, ws, [t], dz, px⟩, ⟨.Z, [m], ws, dz, pz⟩,
        ⟨.Z, [], [m, w, a], dz, pg⟩] ++ vNodes dz b x pq₂ pz₂ px₂ box ms ++
          [⟨.Z, [v], [x], dz, pk⟩]) = fun τ => (tZ d (E.phase pg) [τ m, τ w, τ a] *
            tZ d (E.phase pk) [τ v, τ x]) * (prodVal E [⟨.X, ws, [t], dz, px⟩,
              ⟨.Z, [m], ws, dz, pz⟩] τ * prodVal E (vNodes dz b x pq₂ pz₂ px₂ box ms) τ) := by
      funext τ
      simp only [prodVal_append]
      simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
        List.cons_append, List.nil_append, List.append_nil]
      ring
    have hvl' := vNodes_labels dz b x pq₂ pz₂ px₂ a box ms hbl
    rw [split, sumOver_const_mul _ _ [m, w, a, v, x] _ _ (fun τ τ' h => by
        rw [h m (by simp), h w (by simp), h a (by simp), h v (by simp), h x (by simp)])
      (fun q hq hq' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
        rcases List.mem_append.1 hq' with hq' | hq' <;>
        rcases hq with rfl | rfl | rfl | rfl | rfl
        all_goals first
          | exact nmws hq' | exact nwws hq' | exact naws hq' | exact nvws hq' | exact nxws hq'
          | exact nmv hq' | exact nwv hq' | exact nav hq' | exact nvv hq' | exact nxv hq'),
      sumOver_mul_split ws (vLabs b ms) (t :: m :: ws) (a :: x :: vLabs b ms) _ _
        (fun τ τ' h => prodVal_reads E _ τ τ' fun q hq => h q (by
          simp only [labelsOf, List.flatMap_cons, List.flatMap_nil, Node.labels, List.mem_append,
            List.mem_cons, List.not_mem_nil, or_false, List.append_nil] at hq ⊢
          aesop))
        (fun τ τ' h => prodVal_reads E _ τ τ' fun q hq => h q (hvl' q hq))
        (fun q hq hq' => by
          simp only [List.mem_cons] at hq
          rcases hq with rfl | rfl | hq
          · exact ntv hq'
          · exact nmv hq'
          · exact hwvl q hq q hq' rfl)
        (fun q hq hq' => by
          simp only [List.mem_cons] at hq
          rcases hq with rfl | rfl | hq
          · exact naws hq'
          · exact nxws hq'
          · exact hwvl q hq' q hq rfl)]
    rw [multT_val E pz px dz t m ws σ hd' hws ntws nmws
      (fun q hq => hlab q (by simp [hq])) (by
        rw [hσ, Function.update_of_ne nmx, Function.update_of_ne nma, Function.update_self]
        exact r₁) hz hx]
    rw [vBoxVal E dz a b x pq₂ pz₂ px₂ box ms σ hd' hbz hbd hbl hvl nav nxv
      (fun q hq => hlab q (by simp [hq])) (by
        rw [hσ, Function.update_of_ne nax, Function.update_self]; exact r₂)
      (by rw [hσ, Function.update_self]; exact r₃) hq₂ hz₂ hx₂]
    simp only [hσ, Function.update_apply, nma, nmx, nax, Ne.symm nma, Ne.symm nmx, Ne.symm nax,
      ntm, nta, ntx, nwm, nwa, nwx, nvm, nva, nvx, if_true, if_false]
    rw [← hdd, tZ_at d _ _ (ρ w) (by simp), hg _ rw, tZ_at d _ _ (ρ v) (by simp)]
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, rw, rv,
      true_and, and_true]
    split_ifs <;> ring
  rw [hcan]
  rw [Finset.sum_eq_single (ρ w) (fun j₁ _ hj => Finset.sum_eq_zero fun _ _ =>
      Finset.sum_eq_zero fun _ _ => by rw [if_neg (fun h => hj h.1), zero_mul])
    (fun h => absurd (Finset.mem_range.2 rw) h)]
  rw [Finset.sum_eq_single (ρ w) (fun j₂ _ hj => Finset.sum_eq_zero fun _ _ => by
      rw [if_neg (fun h => hj h.2), zero_mul]) (fun h => absurd (Finset.mem_range.2 rw) h)]
  simp only [and_self, if_true, one_mul, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq' (range d) (ρ v), if_pos (Finset.mem_range.2 rv)]
  ring

/-- A zxw_hd chain with its top value summed against the character `k`. -/
noncomputable def zhdG (E : Env) (dz : ℕ) (c : ZhdC) (k w v : ℕ) : ℂ :=
  (rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * (E.dimv dz : ℂ) *
    fou (E.dimv dz) ((c.ws.length * w) % E.dimv dz) k *
    (zc E c.box w * vConst (E.dimv dz) c.ms *
      (if w = 0 then (if v = 0 then 1 else 0)
        else (c.ms.map fun p => if v = (p.2.1.length * w) % E.dimv dz then (1 : ℂ) else 0).sum)) *
    ex (E.phase c.pk v)

theorem zhdGroup (E : Env) (dz px pz pg pq₂ pz₂ px₂ : ℕ) (c : ZhdC) (k : ℕ) (ρ : ℕ → ℕ)
    (hd : 1 < E.dimv dz) (hnd : (c.t :: zhdInt c).Nodup)
    (hout : ∀ y ∈ [c.w, c.v], y ∉ c.t :: zhdInt c)
    (hlab : ∀ y ∈ c.t :: zhdInt c, labIdx E.labs y = dz) (hbz : isZc c.box = true)
    (hbd : c.box.dim = dz) (hbl : c.box.outs ++ c.box.ins = [c.b, c.a])
    (rw : ρ c.w < E.dimv dz) (rv : ρ c.v < E.dimv dz)
    (hz : ∀ k < E.dimv dz, ex (E.phase pz k) = 1) (hx : ∀ k < E.dimv dz, ex (E.phase px k) = 1)
    (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) (hq₂ : ∀ k < E.dimv dz, ex (E.phase pq₂ k) = 1)
    (hz₂ : ∀ k < E.dimv dz, ex (E.phase pz₂ k) = 1)
    (hx₂ : ∀ k < E.dimv dz, ex (E.phase px₂ k) = 1) :
    sumOver (labDim E) (c.t :: zhdInt c)
      (fun τ => fou (E.dimv dz) (τ c.t) k * prodVal E (zhdNodes dz px pz pg pq₂ pz₂ px₂ c) τ) ρ =
      zhdG E dz c k (ρ c.w) (ρ c.v) := by
  have hd' : 0 < E.dimv dz := by omega
  obtain ⟨ht, hint⟩ := List.nodup_cons.1 hnd
  have hw := hout c.w (by simp)
  have hv := hout c.v (by simp)
  have ntw : c.t ≠ c.w := fun h => hw (h ▸ List.mem_cons_self)
  have ntv : c.t ≠ c.v := fun h => hv (h ▸ List.mem_cons_self)
  simp only [sumOver, labDim_of E c.t dz (hlab c.t List.mem_cons_self)]
  have e : ∀ j ∈ range (E.dimv dz), sumOver (labDim E) (zhdInt c) (fun τ =>
      fou (E.dimv dz) (τ c.t) k * prodVal E (zhdNodes dz px pz pg pq₂ pz₂ px₂ c) τ)
        (Function.update ρ c.t j) =
      if j = (c.ws.length * ρ c.w) % E.dimv dz then zhdG E dz c k (ρ c.w) (ρ c.v) else 0 := by
    intro j hj
    have rj := Finset.mem_range.1 hj
    rw [sumOver_const_mul _ _ [c.t] (fun τ => fou (E.dimv dz) (τ c.t) k) _
      (fun τ τ' h => by simp only; rw [h c.t (by simp)])
      (fun y hy hy' => by simp only [List.mem_cons, List.not_mem_nil, or_false] at hy; subst hy; exact ht hy'),
      zhdChainVal E dz px pz pg pq₂ pz₂ px₂ c _ hd hint (fun y hy hy' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
        rcases hy with rfl | rfl | rfl
        · exact ht hy'
        · exact hw (List.mem_cons_of_mem _ hy')
        · exact hv (List.mem_cons_of_mem _ hy'))
      (fun y hy => hlab y (List.mem_cons_of_mem _ hy)) hbz hbd hbl
      (by rw [Function.update_of_ne (Ne.symm ntw)]; exact rw)
      (by rw [Function.update_of_ne (Ne.symm ntv)]; exact rv) hz hx hg hq₂ hz₂ hx₂]
    simp only [Function.update_self, Function.update_of_ne (Ne.symm ntw),
      Function.update_of_ne (Ne.symm ntv), Nat.mod_eq_of_lt rj]
    by_cases h : j = (c.ws.length * ρ c.w) % E.dimv dz
    · rw [if_pos h.symm, if_pos h, zhdG, ← h]; ring
    · rw [if_neg (Ne.symm h), if_neg h]; ring
  rw [Finset.sum_congr rfl e, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 (Nat.mod_lt _ hd'))]

theorem zhdNodes_labels (dz px pz pg pq₂ pz₂ px₂ : ℕ) (c : ZhdC)
    (hbl : c.box.outs ++ c.box.ins = [c.b, c.a]) :
    ∀ y ∈ labelsOf (zhdNodes dz px pz pg pq₂ pz₂ px₂ c), y ∈ c.t :: zhdInt c ∨ y = c.w ∨ y = c.v := by
  intro y hy
  have hv := vNodes_labels dz c.b c.x pq₂ pz₂ px₂ c.a c.box c.ms hbl
  simp only [zhdNodes, labelsOf, List.flatMap_append, List.mem_append] at hy
  rcases hy with (hy | hy) | hy
  · simp only [List.flatMap_cons, List.flatMap_nil, Node.labels, List.mem_append, List.mem_cons,
      List.not_mem_nil, or_false, List.append_nil] at hy
    simp only [zhdInt, List.mem_cons, List.mem_append]
    aesop
  · have := hv y hy
    simp only [zhdInt, List.mem_cons, List.mem_append] at this ⊢
    aesop
  · simp only [List.flatMap_cons, List.flatMap_nil, Node.labels, List.mem_append, List.mem_cons,
      List.not_mem_nil, or_false, List.append_nil] at hy
    simp only [zhdInt, List.mem_cons, List.mem_append]
    aesop

theorem zhdAt0 (E : Env) (dz : ℕ) (c : ZhdC) (k k' : ℕ) (hd : 1 < E.dimv dz)
    (h0 : zc E c.box 0 = 1) (hK0 : ex (E.phase c.pk 0) = 1) :
    ∑ y ∈ range (E.dimv dz), (starRingEnd ℂ) (fou (E.dimv dz) y k) * zhdG E dz c k' 0 y =
      ((rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * E.dimv dz * vConst (E.dimv dz) c.ms) *
        ((rt (E.dimv dz))⁻¹ * (rt (E.dimv dz))⁻¹) := by
  have hd' : 0 < E.dimv dz := by omega
  simp only [zhdG, mul_zero, Nat.zero_mod, if_true, h0, fou_zero']
  rw [Finset.sum_eq_single 0 (fun y _ hy => by rw [if_neg hy]; ring)
    (fun h => absurd (Finset.mem_range.2 hd') h), if_pos rfl, hK0, fou_zero']
  simp only [rt, map_inv₀, Complex.conj_ofReal]
  ring

theorem zhdAtTop (E : Env) (dz : ℕ) (c : ZhdC) (k k' : ℕ) (hd : 1 < E.dimv dz) (hk : k < E.dimv dz)
    (h1 : zc E c.box (E.dimv dz - 1) = 1)
    (hms : ∀ y < E.dimv dz, (c.ms.map fun p => if y = (p.2.1.length * (E.dimv dz - 1)) %
      E.dimv dz then (1 : ℂ) else 0).sum = 1)
    (hK : ∀ q < E.dimv dz, ex (E.phase c.pk q) = om (E.dimv dz) ^
      (q * ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz))) :
    ∑ y ∈ range (E.dimv dz), (starRingEnd ℂ) (fou (E.dimv dz) y k) *
        zhdG E dz c k' (E.dimv dz - 1) y =
      ((rt (E.dimv dz))⁻¹ ^ (c.ws.length + 1) * E.dimv dz * vConst (E.dimv dz) c.ms) *
        (fou (E.dimv dz) ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz) k' * rt (E.dimv dz) *
          (if k = (c.ws.length * (E.dimv dz - 1)) % E.dimv dz then 1 else 0) * 1) := by
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  set j := (c.ws.length * (d - 1)) % d with hj
  have hjd : j < d := Nat.mod_lt _ hd'
  have hne : d - 1 ≠ 0 := by omega
  simp only [zhdG, ← hdd, ← hj, h1, hne, if_false, one_mul]
  have e : ∀ y ∈ range d, (starRingEnd ℂ) (fou d y k) * ((rt d)⁻¹ ^ (c.ws.length + 1) * d *
      fou d j k' * (vConst d c.ms * (c.ms.map fun p => if y = (p.2.1.length * (d - 1)) % d
        then (1 : ℂ) else 0).sum) * ex (E.phase c.pk y)) =
      ((rt d)⁻¹ ^ (c.ws.length + 1) * d * vConst d c.ms * fou d j k' * rt d) *
        (fou d y j * (starRingEnd ℂ) (fou d y k)) := by
    intro y hy
    rw [hms y (Finset.mem_range.1 hy), hK y (Finset.mem_range.1 hy)]
    have hr := rt_ne d hd'
    simp only [fou, rt]
    field_simp
  rw [Finset.sum_congr rfl e, ← Finset.mul_sum, fou_unitary d hd' j k hjd hk]
  by_cases h : k = j
  · rw [if_pos h.symm, if_pos h]; ring
  · rw [if_neg (Ne.symm h), if_neg h]; ring

theorem prod_vConst (d : ℕ) (hd : 0 < d) (cs : List ZhdC) :
    (cs.map fun c => vConst d c.ms).prod *
      rt d ^ (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum = rt d ^ cs.length := by
  induction cs with
  | nil => simp
  | cons c cs ih =>
    simp only [List.map_cons, List.prod_cons, List.sum_cons, List.length_cons, pow_add, pow_one]
    have h := vConst_pow d hd c.ms
    calc vConst d c.ms * (cs.map fun c => vConst d c.ms).prod *
          (rt d ^ (c.ms.map fun p => p.2.1.length).sum *
            rt d ^ (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum)
        = (vConst d c.ms * rt d ^ (c.ms.map fun p => p.2.1.length).sum) *
          ((cs.map fun c => vConst d c.ms).prod *
            rt d ^ (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum) := by ring
      _ = rt d ^ cs.length * rt d := by rw [h, ih]; ring

/-- [P1] (HD): the Hadamard box decomposition through M boxes, the Fourier box times `s`. -/
theorem zxwHD (E : Env) (dz x₀ i o pk pt pb px pz pg pq₂ pz₂ px₂ : ℕ) (cs : List ZhdC)
    (iface : List ℕ) (hd : 1 < E.dimv dz)
    (hW : (cs.map ZhdC.w ++ cs.map ZhdC.v).Nodup)
    (hQ : (cs.flatMap fun c => c.t :: zhdInt c).Nodup)
    (hWQ : ∀ x ∈ cs.map ZhdC.w ++ cs.map ZhdC.v, x ∉ cs.flatMap fun c => c.t :: zhdInt c)
    (hx₀ : x₀ ∉ (cs.map ZhdC.w ++ cs.map ZhdC.v) ++ cs.flatMap fun c => c.t :: zhdInt c)
    (hio : ∀ x ∈ [i, o], x ∉ x₀ :: ((cs.map ZhdC.w ++ cs.map ZhdC.v) ++
      cs.flatMap fun c => c.t :: zhdInt c))
    (hlab : ∀ x ∈ x₀ :: ((cs.map ZhdC.w ++ cs.map ZhdC.v) ++ cs.flatMap fun c => c.t :: zhdInt c),
      labIdx E.labs x = dz)
    (hif : ∀ x ∈ [i, o], x ∈ iface ∧ labIdx E.labs x = dz)
    (hk : ∀ q < E.dimv dz, ex (E.phase pk q) = om (E.dimv dz) ^ (q * 1))
    (hph : ∀ ph ∈ [pt, pb, px, pz, pg, pq₂, pz₂, px₂], ∀ q < E.dimv dz, ex (E.phase ph q) = 1)
    (hbx : ∀ c ∈ cs, isZc c.box = true ∧ c.box.dim = dz ∧ c.box.outs ++ c.box.ins = [c.b, c.a])
    (hb0 : ∀ c ∈ cs, zc E c.box 0 = 1) (hb1 : ∀ c ∈ cs, zc E c.box (E.dimv dz - 1) = 1)
    (hms : ∀ c ∈ cs, ∀ y < E.dimv dz, (c.ms.filter fun p => decide (y =
      (p.2.1.length * (E.dimv dz - 1)) % E.dimv dz)).length = 1)
    (hK : ∀ c ∈ cs, ∀ q < E.dimv dz, ex (E.phase c.pk q) = om (E.dimv dz) ^
      (q * ((c.ws.length * (E.dimv dz - 1)) % E.dimv dz)))
    (hcount : ∀ x < E.dimv dz, (cs.filter fun c => decide (x = (c.ws.length * (E.dimv dz - 1)) %
      E.dimv dz)).length = 1)
    (s : ℂ) (hs : s * rt (E.dimv dz) ^ ((cs.map fun c => c.ws.length).sum +
      (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum) = rt (E.dimv dz) ^ 4) :
    LocalEq E (⟨.X, [x₀], [], dz, pk⟩ :: ⟨.W, cs.map ZhdC.w, [x₀], dz, 0⟩ ::
        ⟨.X, [o], cs.map ZhdC.v, dz, pb⟩ :: ⟨.X, cs.map ZhdC.t, [i], dz, pt⟩ ::
          cs.flatMap (zhdNodes dz px pz pg pq₂ pz₂ px₂))
      (x₀ :: ((cs.map ZhdC.w ++ cs.map ZhdC.v) ++ cs.flatMap fun c => c.t :: zhdInt c))
      [⟨.F, [o], [i], dz, 0⟩] [] iface 1 s := by
  intro τ hτ
  have ri := hτ i (hif i (by simp)).1
  have ro := hτ o (hif o (by simp)).1
  rw [labDim_of E i dz (hif i (by simp)).2] at ri
  rw [labDim_of E o dz (hif o (by simp)).2] at ro
  have hph' := hph
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq] at hph'
  obtain ⟨ht, hb, hx, hz, hg, hq₂, hz₂, hx₂⟩ := hph'
  have hQc : ∀ c ∈ cs, ∀ y ∈ c.t :: zhdInt c, y ∈ cs.flatMap fun c => c.t :: zhdInt c :=
    fun c hc y hy => List.mem_flatMap.2 ⟨c, hc, hy⟩
  rw [one_mul, chainsVal E dz x₀ i o pk pt pb cs ZhdC.t ZhdC.w ZhdC.v zhdInt
    (zhdNodes dz px pz pg pq₂ pz₂ px₂) (zhdG E dz) τ hd hW hQ hWQ hx₀ hio hlab hk ht hb
    (fun c hc k ρ rw rv => zhdGroup E dz px pz pg pq₂ pz₂ px₂ c k ρ hd
      ((List.nodup_flatMap.1 hQ).1 c hc)
      (fun y hy hy' => by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hy
        rcases hy with rfl | rfl
        · exact hWQ _ (List.mem_append_left _ (List.mem_map_of_mem hc)) (hQc c hc _ hy')
        · exact hWQ _ (List.mem_append_right _ (List.mem_map_of_mem hc)) (hQc c hc _ hy'))
      (fun y hy => hlab y (List.mem_cons_of_mem _ (List.mem_append_right _ (hQc c hc y hy))))
      (hbx c hc).1 (hbx c hc).2.1 (hbx c hc).2.2 rw rv hz hx hg hq₂ hz₂ hx₂)
    (fun c hc => zhdNodes_labels dz px pz pg pq₂ pz₂ px₂ c (hbx c hc).2.2)]
  set d := E.dimv dz with hdd
  have hd' : 0 < d := by omega
  rw [chainsClosed d cs ZhdC.w (fun c => (c.ws.length * (d - 1)) % d) (zhdG E dz)
    (fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * d * vConst d c.ms) (fun _ => (rt d)⁻¹ * (rt d)⁻¹)
    (fun _ => 1) (τ i) (τ o) hd ri ro (List.nodup_append.1 hW).1
    (fun c _ => Nat.mod_lt _ hd')
    (fun c hc k _ k' => zhdAt0 E dz c k k' hd (hb0 c hc) (by
      rw [hK c hc 0 hd', zero_mul, pow_zero]))
    (fun c hc k hk k' => zhdAtTop E dz c k k' hd hk (hb1 c hc) (fun y hy => by
      rw [sum_ind_count, hms c hc y hy, Nat.cast_one]) (hK c hc)) hcount]
  simp only [sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, val]
  have hr := rt_ne d hd'
  have hA := kz_const (rt d) hr cs fun c => c.ws.length
  have hV := prod_vConst d hd' cs
  have hdr : (d : ℂ) = rt d ^ 2 := (rt_sq' d).symm
  suffices hsc : (rt d)⁻¹ * d * ((cs.map fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * d *
      vConst d c.ms).prod * rt d * 1 * ((rt d)⁻¹ * (rt d)⁻¹) ^ (cs.length - 1)) = s by
    rw [← hsc]; ring
  apply mul_right_cancel₀ (pow_ne_zero ((cs.map fun c => c.ws.length).sum +
    (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum) hr)
  rw [hs, List.prod_map_mul, hdr, pow_add (rt d)]
  by_cases hcs : cs = []
  · subst hcs
    have := hcount 0 hd'
    simp at this
  · obtain ⟨m, hm⟩ : ∃ m, cs.length = m + 1 := ⟨cs.length - 1, by
      have := List.length_pos_iff.2 hcs; omega⟩
    calc (rt d)⁻¹ * rt d ^ 2 * ((cs.map fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * rt d ^ 2).prod *
          (cs.map fun c => vConst d c.ms).prod * rt d * 1 * ((rt d)⁻¹ * (rt d)⁻¹) ^
            (cs.length - 1)) * (rt d ^ (cs.map fun c => c.ws.length).sum *
              rt d ^ (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum)
        = (rt d)⁻¹ * rt d ^ 2 * (((cs.map fun c => (rt d)⁻¹ ^ (c.ws.length + 1) * rt d ^ 2).prod *
            rt d ^ (cs.map fun c => c.ws.length).sum) * ((cs.map fun c => vConst d c.ms).prod *
            rt d ^ (cs.map fun c => (c.ms.map fun p => p.2.1.length).sum).sum)) * rt d *
            ((rt d)⁻¹ * (rt d)⁻¹) ^ (cs.length - 1) := by ring
      _ = rt d ^ 4 := by
        rw [hA, hV, hm, show m + 1 - 1 = m by omega]
        field_simp
        rw [one_div, inv_pow, ← pow_mul, ← pow_mul, show (m + 1) * 2 = 2 * m + 2 by ring, pow_add,
          mul_comm 2 m]
        field_simp


theorem tZw_sum_cons (d : ℕ) (c : ℕ → ℂ) (rz : List ℕ) :
    ∑ j ∈ range d, tZw d c (j :: rz) = tZw d c rz := by
  unfold tZw
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  have : ∀ j ∈ range d, (if ∀ v ∈ j :: rz, v = k then c k else 0) =
      if j = k then (if ∀ v ∈ rz, v = k then c k else 0) else 0 := by
    intro j _
    by_cases h : j = k
    · subst h; simp
    · simp [h]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos hk]


theorem tZw_cons_cons (d : ℕ) (c : ℕ → ℂ) (rz : List ℕ) (j j' : ℕ) :
    tZw d c (j :: j' :: rz) = if j' = j then tZw d c (j :: rz) else 0 := by
  unfold tZw
  split_ifs with h
  · subst h
    refine Finset.sum_congr rfl fun k _ => ?_
    simp
  · refine Finset.sum_eq_zero fun k _ => ?_
    have : ¬ ∀ v ∈ j :: j' :: rz, v = k := fun h' =>
      h ((h' j' (by simp)).trans (h' j (by simp)).symm)
    simp only [this, if_false]


/-- (Hopf) with a weighted Z spider. -/
theorem hopfW (E : Env) (pz px dz u v₁ v₂ v₃ : ℕ) (zo zi xo xi zo' zi' xo' xi' iface : List ℕ)
    (k : ℂ) (hd : 0 < E.dimv dz) (hz : (zo ++ zi).Perm (u :: v₁ :: (zo' ++ zi')))
    (hxo : xo.Perm xo') (hxi : xi.Perm (u :: v₃ :: xi')) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.ZW, zo, zi, dz, pz⟩, ⟨.X, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.ZW, zo', zi', dz, pz⟩, ⟨.X, xo', xi', dz, px⟩]
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
      prodVal E [⟨.ZW, zo, zi, dz, pz⟩, ⟨.X, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tZw d (E.coef pz) (ρ u :: ρ v₁ :: R) *
        (tX d (E.phase px) XO (ρ u :: ρ v₃ :: XI) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZw_perm _ _ _ _ (hz.map ρ), tX_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
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
      tZw d (E.coef pz) (ju :: j₁ :: R) * (tX d (E.phase px) XO (ju :: j₃ :: XI) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if j₁ = ju then tZw d (E.coef pz) (ju :: R) * ∑ j₃ ∈ range d,
        (if (j₃ + ju) % d = 0 then tX d (E.phase px) XO (ju :: j₃ :: XI) else 0) else 0 := by
    intro ju _ j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tZw d (E.coef pz) (ju :: j₁ :: R) *
        (tX d (E.phase px) XO (ju :: j₃ :: XI) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        tZw d (E.coef pz) (ju :: j₁ :: R) * tX d (E.phase px) XO (ju :: j₃ :: XI) *
          (if (j₃ + j₁) % d = 0 then 1 else 0) := by
      intro j₃ _
      rw [← fou_sq d hd, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j₂ _ => by ring
    rw [Finset.sum_congr rfl this, tZw_cons_cons]
    split_ifs with h
    · subst h; rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j₃ _ => by split_ifs <;> simp
    · simp
  rw [Finset.sum_congr rfl fun ju hju => (Finset.sum_congr rfl (hA ju hju)).trans
    (by rw [Finset.sum_ite_eq', if_pos hju])]
  have hC : ∀ ju ∈ range d, tZw d (E.coef pz) (ju :: R) * ∑ j₃ ∈ range d,
      (if (j₃ + ju) % d = 0 then tX d (E.phase px) XO (ju :: j₃ :: XI) else 0) =
      tZw d (E.coef pz) (ju :: R) * ((d : ℂ)⁻¹ * tX d (E.phase px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tX_antipode_in d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZw_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring


/-- (Hopf) with the X spider first and a weighted Z spider. -/
theorem hopfSwappedW (E : Env) (px pz dz u v₁ v₂ v₃ : ℕ)
    (xo xi zo zi xo' xi' zo' zi' iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hxo : xo.Perm (u :: v₁ :: xo')) (hxi : xi.Perm xi')
    (hz : (zo ++ zi).Perm (u :: v₃ :: (zo' ++ zi'))) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.X, xo, xi, dz, px⟩, ⟨.ZW, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.X, xo', xi', dz, px⟩, ⟨.ZW, zo', zi', dz, pz⟩]
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
      prodVal E [⟨.X, xo, xi, dz, px⟩, ⟨.ZW, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tX d (E.phase px) (ρ u :: ρ v₁ :: XO) XI *
        (tZw d (E.coef pz) (ρ u :: ρ v₃ :: R) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZw_perm _ _ _ _ (hz.map ρ), tX_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
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
      tX d (E.phase px) (ju :: j₁ :: XO) XI * (tZw d (E.coef pz) (ju :: j₃ :: R) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if (j₁ + ju) % d = 0 then tX d (E.phase px) (ju :: j₁ :: XO) XI *
        tZw d (E.coef pz) (ju :: R) else 0 := by
    intro ju hju j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tX d (E.phase px) (ju :: j₁ :: XO) XI *
        (tZw d (E.coef pz) (ju :: j₃ :: R) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        if j₃ = ju then tX d (E.phase px) (ju :: j₁ :: XO) XI * tZw d (E.coef pz) (ju :: R) *
          (if (j₃ + j₁) % d = 0 then 1 else 0) else 0 := by
      intro j₃ _
      rw [← fou_sq d hd, Finset.mul_sum, tZw_cons_cons]
      split_ifs with h
      · exact Finset.sum_congr rfl fun j₂ _ => by ring
      · simp
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos hju, add_comm ju j₁]
    split_ifs <;> simp
  rw [Finset.sum_congr rfl fun ju hju => Finset.sum_congr rfl (hA ju hju)]
  have hC : ∀ ju ∈ range d, ∑ j₁ ∈ range d, (if (j₁ + ju) % d = 0 then
      tX d (E.phase px) (ju :: j₁ :: XO) XI * tZw d (E.coef pz) (ju :: R) else 0) =
      tZw d (E.coef pz) (ju :: R) * ((d : ℂ)⁻¹ * tX d (E.phase px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tX_antipode_out d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
    ring
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZw_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring



/-- (X definition) for a weighted X spider. -/
theorem xDefW (E : Env) (a b dz : ℕ) (po : List (ℕ × ℕ)) (cs : List (ℕ × (ℕ × ℕ) × ℕ))
    (iface : List ℕ) (hd : 0 < E.dimv dz)
    (hnd : (po.map Prod.fst ++ cs.map (fun c => c.2.2) ++
      cs.flatMap fun c => [c.2.1.1, c.2.1.2]).Nodup)
    (hdis : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, x ∉ po.map Prod.fst ++
      cs.map (fun c => c.2.2) ++ cs.flatMap fun c => [c.2.1.1, c.2.1.2])
    (hface : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, x ∈ iface ∧ labIdx E.labs x = dz)
    (hint : ∀ x ∈ po.map Prod.fst ++ cs.map (fun c => c.2.2) ++
      cs.flatMap (fun c => [c.2.1.1, c.2.1.2]), labIdx E.labs x = dz)
    (hb : ∀ k < E.dimv dz, E.coef b k = E.coef a k) :
    LocalEq E [⟨.XW, po.map Prod.snd, cs.map Prod.fst, dz, a⟩] []
      (⟨.ZW, po.map Prod.fst, cs.map (fun c => c.2.2), dz, b⟩ ::
        ((po.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz)))
      (po.map Prod.fst ++ cs.map (fun c => c.2.2) ++ cs.flatMap fun c => [c.2.1.1, c.2.1.2])
      iface 1 1 := by
  intro τ hτ
  have hr : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, τ x < E.dimv dz := fun x hx => by
    have := hτ x (hface x hx).1; rwa [labDim_of E x dz (hface x hx).2] at this
  set zs := po.map Prod.fst
  set zi := cs.map fun c => c.2.2
  set ws := cs.flatMap fun c => [c.2.1.1, c.2.1.2]
  -- the right side's entry
  have hR : ∀ ρ : ℕ → ℕ, prodVal E (⟨.ZW, zs, zi, dz, b⟩ ::
      ((po.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz))) ρ =
      ∑ k ∈ range (E.dimv dz), (if ∀ v ∈ zs ++ zi, ρ v = k then E.coef b k *
        (po.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod else 0) *
        (cs.map (chainVal (E.dimv dz) ρ)).prod := by
    intro ρ
    simp only [prodVal, List.map_cons, List.prod_cons, List.map_append, List.prod_append,
      prod_flatMap', List.map_map]
    simp only [val, tZw, Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => ?_
    have e : (∀ v ∈ zs.map ρ ++ zi.map ρ, v = k) ↔ ∀ v ∈ zs ++ zi, ρ v = k := by
      constructor
      · intro h v hv
        rcases List.mem_append.1 hv with hv | hv
        · exact h _ (List.mem_append_left _ (List.mem_map_of_mem hv))
        · exact h _ (List.mem_append_right _ (List.mem_map_of_mem hv))
      · intro h v hv
        rcases List.mem_append.1 hv with hv | hv <;> obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hv
        · exact h x (List.mem_append_left _ hx)
        · exact h x (List.mem_append_right _ hx)
    simp only [e]
    split_ifs
    · rw [mul_assoc]
      congr 2
      refine congrArg List.prod (List.map_congr_left fun c _ => ?_)
      simp [val, chainNodes, chainVal]
    · simp
  have hnd' := hnd
  rw [List.nodup_append, List.nodup_append] at hnd'
  obtain ⟨⟨hzs, hzi, hzz⟩, hws, hzw⟩ := hnd'
  have hk_in : ∀ k ∈ range (E.dimv dz), ∀ x ∈ zs ++ zi, k < labDim E x := fun k hk x hx => by
    rw [labDim_of E x dz (hint x (List.mem_append_left _ hx))]; exact Finset.mem_range.1 hk
  simp only [one_mul]
  rw [show prodVal E (⟨.ZW, zs, zi, dz, b⟩ ::
      ((po.map fun p => ⟨.F, [p.2], [p.1], dz, 0⟩) ++ cs.flatMap (chainNodes dz))) = _ from
    funext hR, sumOver_fsum]
  simp only [sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one, val, tXw]
  refine Finset.sum_congr rfl fun k hk => ?_
  have hk' := Finset.mem_range.1 hk
  rw [sumOver_append]
  set G : (ℕ → ℕ) → ℂ := fun ρ => if ∀ v ∈ zs ++ zi, ρ v = k then E.coef b k *
    (po.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod else 0
  set I : (ℕ → ℕ) → ℂ := fun ρ => (cs.map (chainVal (E.dimv dz) ρ)).prod
  have hG : Reads (zs ++ zi ++ po.map Prod.snd) G := by
    intro ρ ρ' h
    simp only [G]
    have e1 : (∀ v ∈ zs ++ zi, ρ v = k) ↔ ∀ v ∈ zs ++ zi, ρ' v = k :=
      ⟨fun h' v hv => h v (List.mem_append_left _ hv) ▸ h' v hv,
        fun h' v hv => (h v (List.mem_append_left _ hv)).symm ▸ h' v hv⟩
    simp only [e1]
    congr 2
    refine congrArg List.prod (List.map_congr_left fun p hp => ?_)
    rw [h p.2 (List.mem_append_right _ (List.mem_map_of_mem hp)),
      h p.1 (List.mem_append_left _ (List.mem_append_left _ (List.mem_map_of_mem hp)))]
  rw [show sumOver (labDim E) ws (fun ρ => G ρ * I ρ) = fun ρ => G ρ * sumOver (labDim E) ws I ρ
    from funext fun ρ => sumOver_const_mul _ ws _ G I hG (fun x hx hxw => by
      rcases List.mem_append.1 hx with hx | hx
      · exact hzw x hx x hxw rfl
      · exact hdis x (List.mem_append_left _ hx) (List.mem_append_right _ hxw)) ρ]
  rw [show (fun ρ => G ρ * sumOver (labDim E) ws I ρ) = fun ρ => if ∀ v ∈ zs ++ zi, ρ v = k then
      E.coef b k * (po.map fun p => fou (E.dimv dz) (ρ p.2) (ρ p.1)).prod *
        sumOver (labDim E) ws I ρ else 0 from funext fun ρ => by
      simp only [G]; split_ifs <;> simp]
  rw [sumOver_pinK (D := labDim E) (zs ++ zi) (List.nodup_append.2 ⟨hzs, hzi, hzz⟩) k
    (hk_in k hk)]
  set ρk : ℕ → ℕ := fun x => if x ∈ zs ++ zi then k else τ x
  have hρz : ∀ x ∈ zs ++ zi, ρk x = k := fun x hx => by
    show (if x ∈ zs ++ zi then k else τ x) = k; rw [if_pos hx]
  have hρo : ∀ x ∈ po.map Prod.snd ++ cs.map Prod.fst, ρk x = τ x := fun x hx => by
    have : x ∉ zs ++ zi := fun h => hdis x hx (List.mem_append_left _ h)
    show (if x ∈ zs ++ zi then k else τ x) = τ x; rw [if_neg this]
  have hO : (po.map fun p => fou (E.dimv dz) (ρk p.2) (ρk p.1)) =
      ((po.map Prod.snd).map τ).map fun o => fou (E.dimv dz) o k := by
    simp only [List.map_map]
    refine List.map_congr_left fun p hp => ?_
    simp only [Function.comp]
    rw [hρo p.2 (List.mem_append_left _ (List.mem_map_of_mem hp)),
      hρz p.1 (List.mem_append_left _ (List.mem_map_of_mem hp))]
  have hS : sumOver (labDim E) ws I ρk =
      ((cs.map Prod.fst).map τ |>.map fun i => (starRingEnd ℂ) (fou (E.dimv dz) i k)).prod := by
    rw [sumOver_chains (D := labDim E) (E.dimv dz) hd k hk' cs
      (fun c hc => ⟨labDim_of E _ dz (hint _ (List.mem_append_right _ (List.mem_flatMap.2
          ⟨c, hc, by simp⟩))), labDim_of E _ dz (hint _ (List.mem_append_right _
          (List.mem_flatMap.2 ⟨c, hc, by simp⟩)))⟩)
      hws
      (fun c hc c' hc' => ⟨fun h => hdis c.1 (List.mem_append_right _ (List.mem_map_of_mem hc))
          (List.mem_append_right _ (List.mem_flatMap.2 ⟨c', hc', h⟩)),
        fun h => hzw c.2.2 (List.mem_append_right _ (List.mem_map_of_mem hc)) c.2.2
          (List.mem_flatMap.2 ⟨c', hc', h⟩) rfl⟩)
      ρk (fun c hc => hρz _ (List.mem_append_right _ (List.mem_map_of_mem hc)))
      (fun c hc => by
        rw [hρo c.1 (List.mem_append_right _ (List.mem_map_of_mem hc))]
        exact hr c.1 (List.mem_append_right _ (List.mem_map_of_mem hc)))]
    simp only [List.map_map]
    refine congrArg List.prod (List.map_congr_left fun c hc => ?_)
    simp only [Function.comp]
    rw [hρo c.1 (List.mem_append_right _ (List.mem_map_of_mem hc))]
  rw [hO, hS, hb k hk']



theorem tXw_perm (d : ℕ) (c : ℕ → ℂ) (os os' is is' : List ℕ) (ho : os.Perm os')
    (hi : is.Perm is') : tXw d c os is = tXw d c os' is' := by
  unfold tXw
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [(ho.map _).prod_eq, (hi.map _).prod_eq]


theorem tXw_antipode_in (d : ℕ) (hd : 0 < d) (c : ℕ → ℂ) (os is : List ℕ) (j j' : ℕ)
    (h : (j + j') % d = 0) : tXw d c os (j :: j' :: is) = (d : ℂ)⁻¹ * tXw d c os is := by
  unfold tXw
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


theorem tXw_antipode_out (d : ℕ) (hd : 0 < d) (c : ℕ → ℂ) (os is : List ℕ) (j j' : ℕ)
    (h : (j + j') % d = 0) : tXw d c (j :: j' :: os) is = (d : ℂ)⁻¹ * tXw d c os is := by
  unfold tXw
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun m _ => ?_
  have hp : fou d j m * fou d j' m = (d : ℂ)⁻¹ := by
    simp only [fou]
    rw [div_mul_div_comm, ← pow_add, ← add_mul, om_pow_zero_mod d hd _ _ h, sqrt_mul_sqrt]
    simp
  simp only [List.map_cons, List.prod_cons]
  rw [← mul_assoc (fou d j m), hp]
  ring


/-- (Hopf) with a weighted X spider. -/
theorem hopfXW (E : Env) (pz px dz u v₁ v₂ v₃ : ℕ) (zo zi xo xi zo' zi' xo' xi' iface : List ℕ)
    (k : ℂ) (hd : 0 < E.dimv dz) (hz : (zo ++ zi).Perm (u :: v₁ :: (zo' ++ zi')))
    (hxo : xo.Perm xo') (hxi : xi.Perm (u :: v₃ :: xi')) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.Z, zo, zi, dz, pz⟩, ⟨.XW, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.Z, zo', zi', dz, pz⟩, ⟨.XW, xo', xi', dz, px⟩]
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
      prodVal E [⟨.Z, zo, zi, dz, pz⟩, ⟨.XW, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tZ d (E.phase pz) (ρ u :: ρ v₁ :: R) *
        (tXw d (E.coef px) XO (ρ u :: ρ v₃ :: XI) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZ_perm _ _ _ _ (hz.map ρ), tXw_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
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
      tZ d (E.phase pz) (ju :: j₁ :: R) * (tXw d (E.coef px) XO (ju :: j₃ :: XI) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if j₁ = ju then tZ d (E.phase pz) (ju :: R) * ∑ j₃ ∈ range d,
        (if (j₃ + ju) % d = 0 then tXw d (E.coef px) XO (ju :: j₃ :: XI) else 0) else 0 := by
    intro ju _ j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tZ d (E.phase pz) (ju :: j₁ :: R) *
        (tXw d (E.coef px) XO (ju :: j₃ :: XI) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        tZ d (E.phase pz) (ju :: j₁ :: R) * tXw d (E.coef px) XO (ju :: j₃ :: XI) *
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
      (if (j₃ + ju) % d = 0 then tXw d (E.coef px) XO (ju :: j₃ :: XI) else 0) =
      tZ d (E.phase pz) (ju :: R) * ((d : ℂ)⁻¹ * tXw d (E.coef px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tXw_antipode_in d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZ_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring


/-- (Hopf) with a weighted X spider. -/
theorem hopfSwappedXW (E : Env) (px pz dz u v₁ v₂ v₃ : ℕ)
    (xo xi zo zi xo' xi' zo' zi' iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hxo : xo.Perm (u :: v₁ :: xo')) (hxi : xi.Perm xi')
    (hz : (zo ++ zi).Perm (u :: v₃ :: (zo' ++ zi'))) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.XW, xo, xi, dz, px⟩, ⟨.Z, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.XW, xo', xi', dz, px⟩, ⟨.Z, zo', zi', dz, pz⟩]
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
      prodVal E [⟨.XW, xo, xi, dz, px⟩, ⟨.Z, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tXw d (E.coef px) (ρ u :: ρ v₁ :: XO) XI *
        (tZ d (E.phase pz) (ρ u :: ρ v₃ :: R) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZ_perm _ _ _ _ (hz.map ρ), tXw_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
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
      tXw d (E.coef px) (ju :: j₁ :: XO) XI * (tZ d (E.phase pz) (ju :: j₃ :: R) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if (j₁ + ju) % d = 0 then tXw d (E.coef px) (ju :: j₁ :: XO) XI *
        tZ d (E.phase pz) (ju :: R) else 0 := by
    intro ju hju j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tXw d (E.coef px) (ju :: j₁ :: XO) XI *
        (tZ d (E.phase pz) (ju :: j₃ :: R) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        if j₃ = ju then tXw d (E.coef px) (ju :: j₁ :: XO) XI * tZ d (E.phase pz) (ju :: R) *
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
      tXw d (E.coef px) (ju :: j₁ :: XO) XI * tZ d (E.phase pz) (ju :: R) else 0) =
      tZ d (E.phase pz) (ju :: R) * ((d : ℂ)⁻¹ * tXw d (E.coef px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tXw_antipode_out d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
    ring
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZ_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring


/-- (Hopf) with weighted Z and X spiders. -/
theorem hopfWXW (E : Env) (pz px dz u v₁ v₂ v₃ : ℕ) (zo zi xo xi zo' zi' xo' xi' iface : List ℕ)
    (k : ℂ) (hd : 0 < E.dimv dz) (hz : (zo ++ zi).Perm (u :: v₁ :: (zo' ++ zi')))
    (hxo : xo.Perm xo') (hxi : xi.Perm (u :: v₃ :: xi')) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.ZW, zo, zi, dz, pz⟩, ⟨.XW, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.ZW, zo', zi', dz, pz⟩, ⟨.XW, xo', xi', dz, px⟩]
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
      prodVal E [⟨.ZW, zo, zi, dz, pz⟩, ⟨.XW, xo, xi, dz, px⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tZw d (E.coef pz) (ρ u :: ρ v₁ :: R) *
        (tXw d (E.coef px) XO (ρ u :: ρ v₃ :: XI) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZw_perm _ _ _ _ (hz.map ρ), tXw_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
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
      tZw d (E.coef pz) (ju :: j₁ :: R) * (tXw d (E.coef px) XO (ju :: j₃ :: XI) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if j₁ = ju then tZw d (E.coef pz) (ju :: R) * ∑ j₃ ∈ range d,
        (if (j₃ + ju) % d = 0 then tXw d (E.coef px) XO (ju :: j₃ :: XI) else 0) else 0 := by
    intro ju _ j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tZw d (E.coef pz) (ju :: j₁ :: R) *
        (tXw d (E.coef px) XO (ju :: j₃ :: XI) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        tZw d (E.coef pz) (ju :: j₁ :: R) * tXw d (E.coef px) XO (ju :: j₃ :: XI) *
          (if (j₃ + j₁) % d = 0 then 1 else 0) := by
      intro j₃ _
      rw [← fou_sq d hd, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j₂ _ => by ring
    rw [Finset.sum_congr rfl this, tZw_cons_cons]
    split_ifs with h
    · subst h; rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun j₃ _ => by split_ifs <;> simp
    · simp
  rw [Finset.sum_congr rfl fun ju hju => (Finset.sum_congr rfl (hA ju hju)).trans
    (by rw [Finset.sum_ite_eq', if_pos hju])]
  have hC : ∀ ju ∈ range d, tZw d (E.coef pz) (ju :: R) * ∑ j₃ ∈ range d,
      (if (j₃ + ju) % d = 0 then tXw d (E.coef px) XO (ju :: j₃ :: XI) else 0) =
      tZw d (E.coef pz) (ju :: R) * ((d : ℂ)⁻¹ * tXw d (E.coef px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tXw_antipode_in d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZw_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring



/-- (Hopf) with weighted Z and X spiders. -/
theorem hopfSwappedWXW (E : Env) (px pz dz u v₁ v₂ v₃ : ℕ)
    (xo xi zo zi xo' xi' zo' zi' iface : List ℕ) (k : ℂ) (hd : 0 < E.dimv dz)
    (hxo : xo.Perm (u :: v₁ :: xo')) (hxi : xi.Perm xi')
    (hz : (zo ++ zi).Perm (u :: v₃ :: (zo' ++ zi'))) (hnd : [u, v₁, v₂, v₃].Nodup)
    (hfree : ∀ x ∈ zo' ++ zi' ++ xo' ++ xi', x ∉ [u, v₁, v₂, v₃])
    (hdim : ∀ x ∈ [u, v₁, v₂, v₃], labIdx E.labs x = dz) (hk : k * E.dimv dz = 1) :
    LocalEq E [⟨.XW, xo, xi, dz, px⟩, ⟨.ZW, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
      ⟨.F, [v₃], [v₂], dz, 0⟩] [u, v₁, v₂, v₃] [⟨.XW, xo', xi', dz, px⟩, ⟨.ZW, zo', zi', dz, pz⟩]
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
      prodVal E [⟨.XW, xo, xi, dz, px⟩, ⟨.ZW, zo, zi, dz, pz⟩, ⟨.F, [v₂], [v₁], dz, 0⟩,
        ⟨.F, [v₃], [v₂], dz, 0⟩] ρ = tXw d (E.coef px) (ρ u :: ρ v₁ :: XO) XI *
        (tZw d (E.coef pz) (ρ u :: ρ v₃ :: R) * (fou d (ρ v₂) (ρ v₁) *
          fou d (ρ v₃) (ρ v₂))) := by
    intro ρ hρ
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val]
    rw [← List.map_append, tZw_perm _ _ _ _ (hz.map ρ), tXw_perm _ _ _ _ _ _ (hxo.map ρ) (hxi.map ρ)]
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
      tXw d (E.coef px) (ju :: j₁ :: XO) XI * (tZw d (E.coef pz) (ju :: j₃ :: R) *
        (fou d j₂ j₁ * fou d j₃ j₂)) =
      if (j₁ + ju) % d = 0 then tXw d (E.coef px) (ju :: j₁ :: XO) XI *
        tZw d (E.coef pz) (ju :: R) else 0 := by
    intro ju hju j₁ _
    rw [Finset.sum_comm]
    have : ∀ j₃ ∈ range d, ∑ j₂ ∈ range d, tXw d (E.coef px) (ju :: j₁ :: XO) XI *
        (tZw d (E.coef pz) (ju :: j₃ :: R) * (fou d j₂ j₁ * fou d j₃ j₂)) =
        if j₃ = ju then tXw d (E.coef px) (ju :: j₁ :: XO) XI * tZw d (E.coef pz) (ju :: R) *
          (if (j₃ + j₁) % d = 0 then 1 else 0) else 0 := by
      intro j₃ _
      rw [← fou_sq d hd, Finset.mul_sum, tZw_cons_cons]
      split_ifs with h
      · exact Finset.sum_congr rfl fun j₂ _ => by ring
      · simp
    rw [Finset.sum_congr rfl this, Finset.sum_ite_eq', if_pos hju, add_comm ju j₁]
    split_ifs <;> simp
  rw [Finset.sum_congr rfl fun ju hju => Finset.sum_congr rfl (hA ju hju)]
  have hC : ∀ ju ∈ range d, ∑ j₁ ∈ range d, (if (j₁ + ju) % d = 0 then
      tXw d (E.coef px) (ju :: j₁ :: XO) XI * tZw d (E.coef pz) (ju :: R) else 0) =
      tZw d (E.coef pz) (ju :: R) * ((d : ℂ)⁻¹ * tXw d (E.coef px) XO XI) := by
    intro ju hju
    have hj := Finset.mem_range.1 hju
    rw [antipode_collapse d hd ju hj, tXw_antipode_out d hd _ _ _ _ _ (antipode_spec d ju hd hj)]
    ring
  rw [Finset.sum_congr rfl hC, ← Finset.sum_mul, tZw_sum_cons]
  have : k = (d : ℂ)⁻¹ := by
    exact eq_inv_of_mul_eq_one_left hk
  rw [this]
  simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, val,
    ← List.map_append]
  ring





/-- An X or weighted X spider's coefficient at `k`. -/
noncomputable def xc (E : Env) (n : Node) (k : ℕ) : ℂ :=
  match n.kind with
  | .X => ex (E.phase n.ph k)
  | .XW => E.coef n.ph k
  | _ => 0

/-- An X or weighted X spider. -/
def isXc (n : Node) : Bool := n.kind == .X || n.kind == .XW

theorem val_xc (E : Env) (σ : ℕ → ℕ) (n : Node) (h : isXc n = true) :
    val E σ n = tXw (E.dimv n.dim) (xc E n) (n.outs.map σ) (n.ins.map σ) := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only [isXc, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with rfl | rfl <;> simp only [val, xc, tX, tXw]

/-- Fusion of two X or weighted X spiders along `w`: the coefficients multiply. -/
theorem xcFusion (E : Env) (na nb m : Node) (dz w : ℕ) (iface : List ℕ) (ha : isXc na = true)
    (hb : isXc nb = true) (hm : isXc m = true) (hda : na.dim = dz) (hdb : nb.dim = dz)
    (hdm : m.dim = dz) (hd : 0 < E.dimv dz) (h₁ : w ∈ na.outs) (h₁' : w ∉ na.outs.erase w)
    (h₁i : w ∉ na.ins) (h₂ : w ∈ nb.ins) (h₂' : w ∉ nb.ins.erase w) (h₂o : w ∉ nb.outs)
    (ho : m.outs.Perm (na.outs.erase w ++ nb.outs)) (hi : m.ins.Perm (na.ins ++ nb.ins.erase w))
    (hw : labIdx E.labs w = dz)
    (hc : ∀ k < E.dimv dz, xc E m k = xc E na k * xc E nb k) :
    LocalEq E [na, nb] [w] [m] [] iface 1 1 := by
  intro τ _
  simp only [one_mul, sumOver, prodVal, List.map_cons, List.map_nil, List.prod_cons,
    List.prod_nil, mul_one, val_xc E _ na ha, val_xc E _ nb hb, val_xc E _ m hm, hda, hdb, hdm]
  have hD : labDim E w = E.dimv dz := by
    simp only [labDim]; simp only [labIdx] at hw; rw [hw]
  rw [hD]
  set d := E.dimv dz
  set F : ℕ → ℕ → ℂ := fun x k => fou d x k
  set G : ℕ → ℕ → ℂ := fun x k => (starRingEnd ℂ) (fou d x k)
  set A : ℕ → ℂ := fun k => xc E na k * (((na.outs.erase w).map τ).map (F · k)).prod *
    ((na.ins.map τ).map (G · k)).prod
  set B : ℕ → ℂ := fun k => xc E nb k * ((nb.outs.map τ).map (F · k)).prod *
    (((nb.ins.erase w).map τ).map (G · k)).prod
  have hl : ∀ j, tXw d (xc E na) (na.outs.map (Function.update τ w j))
      (na.ins.map (Function.update τ w j)) = ∑ k ∈ range d, F j k * A k := by
    intro j
    unfold tXw
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [map_update_absent τ w j na.ins h₁i, prod_update_once (F · k) τ w j na.outs h₁ h₁']
    simp only [A]; ring
  have hr : ∀ j, tXw d (xc E nb) (nb.outs.map (Function.update τ w j))
      (nb.ins.map (Function.update τ w j)) = ∑ k ∈ range d, G j k * B k := by
    intro j
    unfold tXw
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [map_update_absent τ w j nb.outs h₂o, prod_update_once (G · k) τ w j nb.ins h₂ h₂']
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
  unfold tXw
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [((ho.map τ).map (F · k)).prod_eq, ((hi.map τ).map (G · k)).prod_eq,
    hc k (Finset.mem_range.1 hk)]
  simp only [A, B, List.map_append, List.prod_append, F, G]
  ring



/-- A W node at `u` is nonzero only on all-zero outputs (`u = 0`) or on one output `u`. -/
theorem tW_support (d u : ℕ) (ys : List ℕ) (h : tW d ys u ≠ 0) :
    (u = 0 ∧ ∀ y ∈ ys, y = 0) ∨ (u ≠ 0 ∧ u < d ∧ ys.filter (· ≠ 0) = [u]) := by
  unfold tW at h
  by_cases hu : u = 0
  · subst hu
    simp only [if_true] at h
    split_ifs at h with hy
    · exact Or.inl ⟨rfl, hy⟩
    · exact absurd rfl h
  · simp only [hu, if_false] at h
    split_ifs at h with hy
    · exact Or.inr ⟨hu, hy.1, hy.2⟩
    · exact absurd rfl h

/-- A W node at `j` summed against a W node at `j` over a shared input: the zero term plus the
one-hot terms. -/
theorem wPairSum (d : ℕ) (xs ys : List ℕ) :
    ∑ z ∈ range d, tW d xs z * tW d ys z =
      (if (∀ x ∈ xs, x = 0) ∧ ∀ y ∈ ys, y = 0 then (if 0 < d then 1 else 0) else 0) +
        ∑ z ∈ (range d).erase 0, (if xs.filter (· ≠ 0) = [z] ∧ ys.filter (· ≠ 0) = [z] then 1
          else 0) := by
  by_cases hd : 0 < d
  · rw [← Finset.add_sum_erase _ _ (Finset.mem_range.2 hd), if_pos hd]
    congr 1
    · simp only [tW, if_true]
      split_ifs <;> first | rfl | (exfalso; tauto) | simp
    · refine Finset.sum_congr rfl fun z hz => ?_
      have hz0 := Finset.ne_of_mem_erase hz
      have hzd := Finset.mem_range.1 (Finset.mem_of_mem_erase hz)
      simp only [tW, hz0, if_false, hzd, true_and]
      split_ifs <;> first | rfl | (exfalso; tauto) | simp
  · have : d = 0 := by omega
    subst this; simp

/-- A W node whose outputs are all zero. -/
theorem tW_replicateZero (d m x : ℕ) : tW d (List.replicate m 0) x = if x = 0 then 1 else 0 := by
  unfold tW
  by_cases hx : x = 0
  · simp [hx]
  · have : (List.replicate m 0).filter (· ≠ 0) = [] := by
      simp [List.filter_eq_nil_iff]
    simp [hx, this]

/-- A W node whose outputs are zero except one entry `v ≠ 0`. -/
theorem tW_single (d x v : ℕ) (a b : List ℕ) (ha : ∀ y ∈ a, y = 0) (hb : ∀ y ∈ b, y = 0)
    (hv : v ≠ 0) : tW d (a ++ v :: b) x = if x = v ∧ v < d then 1 else 0 := by
  have hfa : a.filter (· ≠ 0) = [] := List.filter_eq_nil_iff.2 fun y hy => by simp [ha y hy]
  have hfb : b.filter (· ≠ 0) = [] := List.filter_eq_nil_iff.2 fun y hy => by simp [hb y hy]
  have hf : (a ++ v :: b).filter (· ≠ 0) = [v] := by
    rw [List.filter_append, List.filter_cons, hfa, hfb]
    simp [hv]
  unfold tW
  by_cases hx : x = 0
  · subst hx
    have : ¬ ∀ y ∈ a ++ v :: b, y = 0 := fun h => hv (h v (by simp))
    simp only [if_true, this, if_false]
    split_ifs with h
    · exact absurd h.1 (Ne.symm hv)
    · rfl
  · simp only [hx, if_false, hf, List.cons.injEq, and_true]
    by_cases h : x = v
    · subst h; simp
    · simp [h, Ne.symm h]

/-- W nodes at the values `xs`, each with all-zero outputs. -/
theorem prod_tW_zeros (d m : ℕ) (xs : List ℕ) :
    (xs.map fun x => tW d (List.replicate m 0) x).prod = if ∀ x ∈ xs, x = 0 then 1 else 0 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.prod_cons]
    rw [ih, tW_replicateZero]
    simp only [List.forall_mem_cons]
    split_ifs <;> first | rfl | (exfalso; tauto) | simp

theorem prod_tW_zero_in {α : Type*} (d : ℕ) (cols : List α) (f : α → List ℕ) :
    (cols.map fun c => tW d (f c) 0).prod =
      if ∀ c ∈ cols, ∀ y ∈ f c, y = 0 then 1 else 0 := by
  induction cols with
  | nil => simp
  | cons c cs ih =>
    simp only [List.map_cons, List.prod_cons]
    rw [ih]
    simp only [tW, if_true, List.forall_mem_cons]
    split_ifs <;> first | rfl | (exfalso; tauto) | simp

/-- Columns of W outputs, each W at `0`, summed: every column label is pinned to `0`. -/
theorem sumOver_cols_zero {D : ℕ → ℕ} (d : ℕ) (cols : List (List ℕ)) (hnd : cols.flatten.Nodup)
    (hD : ∀ l ∈ cols.flatten, 0 < D l) (G : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D cols.flatten (fun τ => (cols.map fun c => tW d (c.map τ) 0).prod * G τ) σ =
      G (fun x => if x ∈ cols.flatten then 0 else σ x) := by
  have e : ∀ τ : ℕ → ℕ, (cols.map fun c => tW d (c.map τ) 0).prod * G τ =
      if ∀ l ∈ cols.flatten, τ l = (fun _ => 0) l then G τ else 0 := by
    intro τ
    rw [prod_tW_zero_in d cols (fun c => c.map τ)]
    have hiff : (∀ c ∈ cols, ∀ y ∈ c.map τ, y = 0) ↔ ∀ l ∈ cols.flatten, τ l = 0 := by
      constructor
      · intro h l hl
        obtain ⟨c, hc, hlc⟩ := List.mem_flatten.1 hl
        exact h c hc _ (List.mem_map_of_mem hlc)
      · intro h c hc y hy
        obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hy
        exact h l (List.mem_flatten.2 ⟨c, hc, hl⟩)
    by_cases h : ∀ l ∈ cols.flatten, τ l = 0
    · rw [if_pos (hiff.2 h), if_pos h, one_mul]
    · rw [if_neg (fun h' => h (hiff.1 h')), if_neg h, zero_mul]
  simp only [e]
  exact sumOver_pinV cols.flatten hnd (fun _ => 0) hD G σ

/-- Columns of W outputs at `0` and one column at `v`, summed. -/
theorem sumOver_cols_hot {D : ℕ → ℕ} (d : ℕ) (hd : 0 < d) (A : List (List ℕ)) (col : List ℕ)
    (v : ℕ) (hv : v < d) (hnd : (A.flatten ++ col).Nodup)
    (hD : ∀ l ∈ A.flatten ++ col, D l = d) (G : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ) :
    sumOver D (A.flatten ++ col) (fun τ => (A.map fun c => tW d (c.map τ) 0).prod *
        (tW d (col.map τ) v * G τ)) σ =
      if v = 0 then G (fun x => if x ∈ col then 0 else
          (fun x => if x ∈ A.flatten then 0 else σ x) x)
      else (col.map fun l => G (Function.update (fun x => if x ∈ col then 0 else
          (fun x => if x ∈ A.flatten then 0 else σ x) x) l v)).sum := by
  obtain ⟨hA, hc, hAc⟩ := List.nodup_append.1 hnd
  rw [sumOver_append]
  have inner : ∀ τ : ℕ → ℕ, sumOver D col (fun τ => (A.map fun c => tW d (c.map τ) 0).prod *
      (tW d (col.map τ) v * G τ)) τ = (A.map fun c => tW d (c.map τ) 0).prod *
      (if v = 0 then G (fun x => if x ∈ col then 0 else τ x)
        else (col.map fun l => G (Function.update (fun x => if x ∈ col then 0 else τ x) l v)).sum) := by
    intro τ
    rw [sumOver_const_mul D col A.flatten (fun τ => (A.map fun c => tW d (c.map τ) 0).prod) _
      (fun τ τ' h => by
        simp only
        congr 1
        refine List.map_congr_left fun c hc => ?_
        congr 1
        exact List.map_congr_left fun l hl => h l (List.mem_flatten.2 ⟨c, hc, hl⟩))
      (fun x hx hx' => hAc x hx x hx' rfl),
      sumOver_W d col hc (fun l hl => hD l (List.mem_append_right _ hl)) v hv hd G τ]
  rw [show sumOver D col (fun τ => (A.map fun c => tW d (c.map τ) 0).prod *
      (tW d (col.map τ) v * G τ)) = _ from funext inner]
  exact sumOver_cols_zero d A hA (fun l hl => by rw [hD l (List.mem_append_left _ hl)]; exact hd)
    _ σ

/-- A plain Z spider `y, p, w` at `y < d`: `p` and `w` pinned to `y`. -/
theorem tZ_pin3 (d : ℕ) (φ : ℕ → ℝ) (y p w : ℕ) (hy : y < d) (hφ : ex (φ y) = 1) :
    tZ d φ [y, p, w] = if p = y ∧ w = y then 1 else 0 := by
  rw [tZ_at d φ _ y (by simp)]
  by_cases h1 : p = y <;> by_cases h2 : w = y <;> simp [h1, h2, hy, hφ]

/-- The nodes of the WW left side: W transpose `xs ← z`, W `ws ← z`, W `ps ← u` and the green
dots `y_k ← p_k, w_k` of `cs = (p_k, w_k, y_k)`. -/
def wwLeft (dz z u pg : ℕ) (xs : List ℕ) (cs : List (ℕ × ℕ × ℕ)) : List Node :=
  ⟨.W, xs, [z], dz, 0⟩ :: ⟨.W, cs.map (·.2.1), [z], dz, 0⟩ :: ⟨.W, cs.map Prod.fst, [u], dz, 0⟩ ::
    cs.map fun c => ⟨.Z, [c.2.2], [c.1, c.2.1], dz, pg⟩

/-- The WW left side: a W node at `u` on `ys` times the shared-input W pair. -/
theorem wwLeftVal (E : Env) (dz z u pg : ℕ) (xs : List ℕ) (cs : List (ℕ × ℕ × ℕ)) (τ : ℕ → ℕ)
    (hnd : (z :: cs.flatMap fun c => [c.1, c.2.1]).Nodup)
    (hout : ∀ x ∈ u :: xs ++ cs.map (·.2.2), x ∉ z :: cs.flatMap fun c => [c.1, c.2.1])
    (hlab : ∀ x ∈ z :: cs.flatMap fun c => [c.1, c.2.1], labIdx E.labs x = dz)
    (ry : ∀ c ∈ cs, τ c.2.2 < E.dimv dz) (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) :
    sumOver (labDim E) (z :: cs.flatMap fun c => [c.1, c.2.1]) (prodVal E (wwLeft dz z u pg xs cs))
        τ =
      tW (E.dimv dz) (cs.map fun c => τ c.2.2) (τ u) *
        ∑ j ∈ range (E.dimv dz), tW (E.dimv dz) (xs.map τ) j *
          tW (E.dimv dz) (cs.map fun c => τ c.2.2) j := by
  set d := E.dimv dz with hdd
  set P := cs.flatMap fun c => [c.1, c.2.1]
  obtain ⟨hzP, hP⟩ := List.nodup_cons.1 hnd
  have hD : ∀ x ∈ z :: P, labDim E x = d := fun x hx => labDim_of E x dz (hlab x hx)
  have hu : u ∉ z :: P := hout u (by simp)
  have hx : ∀ x ∈ xs, x ∉ z :: P := fun x h => hout x (by simp [h])
  have hy : ∀ c ∈ cs, c.2.2 ∉ z :: P := fun c h => hout _
    (List.mem_cons_of_mem _ (List.mem_append_right _ (List.mem_map_of_mem (f := (·.2.2)) h)))
  simp only [sumOver, hD z List.mem_cons_self, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [sumOver_congr _ P _ (fun ρ => (cs.map fun c => if ρ c.1 = τ c.2.2 ∧ ρ c.2.1 = τ c.2.2
      then (1 : ℂ) else 0).prod * (fun ps ws => tW d (xs.map τ) j * tW d ws j * tW d ps (τ u))
        (cs.map fun c => ρ c.1) (cs.map fun c => ρ c.2.1)) _ fun ρ hρ _ => by
    have e : ∀ x, x ∉ z :: P → ρ x = τ x := fun x h => by
      rw [hρ x (fun h' => h (List.mem_cons_of_mem _ h')),
        Function.update_of_ne (fun h' => h (by rw [h']; exact List.mem_cons_self))]
    have ez : ρ z = j := by rw [hρ z hzP, Function.update_self]
    have hz : ((cs.map fun c => (⟨.Z, [c.2.2], [c.1, c.2.1], dz, pg⟩ : Node)).map (val E ρ)) =
        cs.map fun c => if ρ c.1 = τ c.2.2 ∧ ρ c.2.1 = τ c.2.2 then (1 : ℂ) else 0 := by
      rw [List.map_map]
      refine List.map_congr_left fun c hc => ?_
      simp only [Function.comp_apply, val, List.map_cons, List.map_nil, List.cons_append,
        List.nil_append]
      rw [e _ (hy c hc), tZ_pin3 d _ _ _ _ (ry c hc) (hg _ (ry c hc))]
    simp only [wwLeft, prodVal, List.map_cons, List.prod_cons, hz]
    simp only [val, List.map_cons, List.map_nil, List.map_map, Function.comp_def]
    rw [List.map_congr_left (l := xs) fun x h => e x (hx x h), ez, e u hu]
    ring]
  have hv : ∀ c ∈ cs, τ c.2.2 < labDim E c.1 := fun c hc => by
    rw [hD _ (List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨c, hc, by simp⟩))]; exact ry c hc
  have hv' : ∀ c ∈ cs, τ c.2.2 < labDim E c.2.1 := fun c hc => by
    rw [hD _ (List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨c, hc, by simp⟩))]; exact ry c hc
  refine (sumOver_pin2 (D := labDim E) cs Prod.fst (fun c => c.2.1) (fun c => τ c.2.2)
    (fun c => τ c.2.2) (fun _ => 1) (fun ps ws => tW d (xs.map τ) j * tW d ws j * tW d ps (τ u))
    hP hv hv' _).trans ?_
  simp only [List.map_const', List.prod_replicate, one_pow, one_mul]
  ring

/-- The nodes of the WW right side: W `ps ← u`, the row W nodes `row_k ← a_k` of
`rows = (a_k, row_k)`, the column W nodes `col_c ← s_c` and green dots `y_c ← p_c, s_c` of
`cols = (p_c, s_c, y_c, col_c)`. -/
def wwRight (dz u pg : ℕ) (rows : List (ℕ × List ℕ)) (cols : List (ℕ × ℕ × ℕ × List ℕ)) :
    List Node :=
  ⟨.W, cols.map Prod.fst, [u], dz, 0⟩ :: ((rows.map fun r => ⟨.W, r.2, [r.1], dz, 0⟩) ++
    (cols.map fun c => ⟨.W, c.2.2.2, [c.2.1], dz, 0⟩) ++
    cols.map fun c => ⟨.Z, [c.2.2.1], [c.1, c.2.1], dz, pg⟩)

/-- The WW right side with `p_c, s_c` summed: both pinned to `y_c`. -/
theorem wwRightPin (E : Env) (dz u pg : ℕ) (rows : List (ℕ × List ℕ))
    (cols : List (ℕ × ℕ × ℕ × List ℕ)) (τ : ℕ → ℕ)
    (hnd : (cols.flatMap fun c => [c.1, c.2.1]).Nodup)
    (hout : ∀ x ∈ u :: (rows.flatMap fun r => r.1 :: r.2) ++
      cols.flatMap (fun c => c.2.2.1 :: c.2.2.2), x ∉ cols.flatMap fun c => [c.1, c.2.1])
    (hlab : ∀ x ∈ cols.flatMap fun c => [c.1, c.2.1], labIdx E.labs x = dz)
    (ry : ∀ c ∈ cols, τ c.2.2.1 < E.dimv dz) (hg : ∀ k < E.dimv dz, ex (E.phase pg k) = 1) :
    sumOver (labDim E) (cols.flatMap fun c => [c.1, c.2.1]) (prodVal E (wwRight dz u pg rows cols))
        τ =
      tW (E.dimv dz) (cols.map fun c => τ c.2.2.1) (τ u) *
        ((rows.map fun r => tW (E.dimv dz) (r.2.map τ) (τ r.1)).prod *
          (cols.map fun c => tW (E.dimv dz) (c.2.2.2.map τ) (τ c.2.2.1)).prod) := by
  set d := E.dimv dz with hdd
  set P := cols.flatMap fun c => [c.1, c.2.1]
  have hD : ∀ x ∈ P, labDim E x = d := fun x hx => labDim_of E x dz (hlab x hx)
  have hu : u ∉ P := hout u (by simp)
  have hr : ∀ r ∈ rows, ∀ x ∈ r.1 :: r.2, x ∉ P := fun r h x hx => hout x
    (List.mem_cons_of_mem _ (List.mem_append_left _ (List.mem_flatMap.2 ⟨r, h, hx⟩)))
  have hc : ∀ c ∈ cols, ∀ x ∈ c.2.2.1 :: c.2.2.2, x ∉ P := fun c h x hx => hout x
    (List.mem_cons_of_mem _ (List.mem_append_right _ (List.mem_flatMap.2 ⟨c, h, hx⟩)))
  set R : ℂ := (rows.map fun r => tW d (r.2.map τ) (τ r.1)).prod
  rw [sumOver_congr _ P _ (fun ρ => (cols.map fun c => if ρ c.1 = τ c.2.2.1 ∧
      ρ c.2.1 = τ c.2.2.1 then (1 : ℂ) else 0).prod * (fun ps ss => tW d ps (τ u) * (R *
        (List.zipWith (fun c s => tW d (c.2.2.2.map τ) s) cols ss).prod))
        (cols.map fun c => ρ c.1) (cols.map fun c => ρ c.2.1)) _ fun ρ hρ _ => by
    have e : ∀ x, x ∉ P → ρ x = τ x := hρ
    have hz : ((cols.map fun c => (⟨.Z, [c.2.2.1], [c.1, c.2.1], dz, pg⟩ : Node)).map (val E ρ)) =
        cols.map fun c => if ρ c.1 = τ c.2.2.1 ∧ ρ c.2.1 = τ c.2.2.1 then (1 : ℂ) else 0 := by
      rw [List.map_map]
      refine List.map_congr_left fun c h => ?_
      simp only [Function.comp_apply, val, List.map_cons, List.map_nil, List.cons_append,
        List.nil_append]
      rw [e _ (hc c h _ List.mem_cons_self), tZ_pin3 d _ _ _ _ (ry c h) (hg _ (ry c h))]
    have hR : ((rows.map fun r => (⟨.W, r.2, [r.1], dz, 0⟩ : Node)).map (val E ρ)).prod = R := by
      rw [List.map_map]
      refine congrArg List.prod (List.map_congr_left fun r h => ?_)
      simp only [Function.comp_apply, val, List.map_cons, List.map_nil]
      rw [e _ (hr r h _ List.mem_cons_self),
        List.map_congr_left fun x hx => e x (hr r h x (List.mem_cons_of_mem _ hx))]
    have hC : ((cols.map fun c => (⟨.W, c.2.2.2, [c.2.1], dz, 0⟩ : Node)).map (val E ρ)).prod =
        (List.zipWith (fun c s => tW d (c.2.2.2.map τ) s) cols (cols.map fun c => ρ c.2.1)).prod := by
      rw [List.map_map, List.zipWith_map_right, List.zipWith_self]
      refine congrArg List.prod (List.map_congr_left fun c h => ?_)
      simp only [Function.comp_apply, val, List.map_cons, List.map_nil]
      rw [List.map_congr_left fun x hx => e x (hc c h x (List.mem_cons_of_mem _ hx))]
    simp only [wwRight, prodVal, List.map_cons, List.map_append, List.prod_cons, List.prod_append,
      hz, hR, hC]
    simp only [val, List.map_cons, List.map_nil, List.map_map, Function.comp_def]
    rw [e u hu]
    ring]
  have hv : ∀ c ∈ cols, τ c.2.2.1 < labDim E c.1 := fun c h => by
    rw [hD _ (List.mem_flatMap.2 ⟨c, h, by simp⟩)]; exact ry c h
  have hv' : ∀ c ∈ cols, τ c.2.2.1 < labDim E c.2.1 := fun c h => by
    rw [hD _ (List.mem_flatMap.2 ⟨c, h, by simp⟩)]; exact ry c h
  refine (sumOver_pin2 (D := labDim E) cols Prod.fst (fun c => c.2.1) (fun c => τ c.2.2.1)
    (fun c => τ c.2.2.1) (fun _ => 1) (fun ps ss => tW d ps (τ u) * (R *
      (List.zipWith (fun c s => tW d (c.2.2.2.map τ) s) cols ss).prod)) hnd hv hv' _).trans ?_
  simp only [List.map_const', List.prod_replicate, one_pow, one_mul, List.zipWith_map_right,
    List.zipWith_self]


end ArchytasZX
