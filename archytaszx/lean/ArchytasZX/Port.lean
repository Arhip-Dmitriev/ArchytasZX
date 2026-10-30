/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bundle

/-!
# Port boxes

A port box along `m` bundles boundary legs of Z and X spiders. At `m = k ≥ 1` the split
network, one more leg per bundle, is the network itself read at an assignment that advances
each X bundle's lowest digit by its peeled leg, times a factor of the boundary values alone:
`[peeled = lowest digit]` per Z bundle and `1 / √d` per X bundle. Two equal networks with the
same port legs therefore have equal splits.
-/

namespace ArchytasZX

open Complex Finset

/-! ## Digits -/

/-- `v` with its lowest base-`d` digit advanced by `e`. -/
def adj (d v e : ℕ) : ℕ := v - v % d + (v % d + e) % d

theorem adj_eq (d v e : ℕ) : adj d v e = d * (v / d) + (v % d + e) % d := by
  unfold adj
  have := Nat.div_add_mod v d
  omega

theorem adj_div (d v e : ℕ) (hd : 0 < d) : adj d v e / d = v / d := by
  rw [adj_eq, Nat.mul_add_div hd, Nat.div_eq_of_lt (Nat.mod_lt _ hd), add_zero]

theorem adj_mod (d v e : ℕ) : adj d v e % d = (v % d + e) % d := by
  rw [adj_eq, Nat.mul_add_mod, Nat.mod_mod]

theorem adj_lt (d k v e : ℕ) (hd : 0 < d) (hk : 1 ≤ k) (hv : v < d ^ k) : adj d v e < d ^ k := by
  rw [adj_eq]
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  have h1 : v / d < d ^ j := by
    rw [Nat.div_lt_iff_lt_mul hd]; rw [pow_succ] at hv; exact hv
  have h2 := Nat.mod_lt (v % d + e) hd
  have : d * (v / d) + d ≤ d ^ (j + 1) := by
    rw [pow_succ, mul_comm (d ^ j)]
    nlinarith
  omega

theorem digit_adj_succ (d i v e : ℕ) (hd : 0 < d) :
    digit d (i + 1) (adj d v e) = digit d (i + 1) v := by
  simp only [digit, pow_succ, mul_comm (d ^ i), ← Nat.div_div_eq_div_mul, adj_div d v e hd]

theorem digit_adj_zero (d v e : ℕ) :
    digit d 0 (adj d v e) = (v % d + e) % d := by
  simp only [digit, pow_zero, Nat.div_one, adj_mod d v e]

theorem digs_cons (d k v : ℕ) :
    digs d (k + 1) v = digit d 0 v :: (List.range k).map fun i => digit d (i + 1) v := by
  simp [digs, List.range_succ_eq_map, List.map_map, Function.comp_def]

theorem digs_adj (d k v e : ℕ) (hd : 0 < d) :
    digs d (k + 1) (adj d v e) =
      (v % d + e) % d :: (List.range k).map fun i => digit d (i + 1) v := by
  rw [digs_cons, digit_adj_zero d v e]
  simp only [digit_adj_succ d _ v e hd]

theorem digit_zero (d v : ℕ) : digit d 0 v = v % d := by simp [digit]

theorem fou_add_mod (d a e kk : ℕ) (hd : 0 < d) :
    fou d ((a + e) % d) kk = rt d * fou d a kk * fou d e kk := by
  have hr : rt d ≠ 0 := rt_ne d hd
  unfold fou
  have : om d ^ ((a + e) % d * kk) = om d ^ (a * kk) * om d ^ (e * kk) := by
    rw [← pow_add, om_pow_mod d hd, om_pow_mod d hd (a * kk + e * kk)]
    congr 1
    rw [← add_mul, Nat.mul_mod, Nat.mod_mod, ← Nat.mul_mod]
  rw [this]
  unfold rt
  field_simp

theorem conj_rt (d : ℕ) : (starRingEnd ℂ) (rt d) = rt d := by
  unfold rt; exact Complex.conj_ofReal _

theorem conj_fou_add_mod (d a e kk : ℕ) (hd : 0 < d) :
    (starRingEnd ℂ) (fou d ((a + e) % d) kk) =
      rt d * (starRingEnd ℂ) (fou d a kk) * (starRingEnd ℂ) (fou d e kk) := by
  rw [fou_add_mod d a e kk hd, map_mul, map_mul, conj_rt]

/-! ## Port legs -/

/-- The legs `tl` whose chains `cs` start at `m`, each with the colour `c`. -/
def portLegs (m : ℕ) (c : ℕ) : List ℕ → List (List ℕ) → List (ℕ × ℕ)
  | x :: tl, ch :: cs => if ch.head? = some m then (x, c) :: portLegs m c tl cs
      else portLegs m c tl cs
  | _, _ => []

/-- A node's legs bundled along `m` outermost: colour `0` on a Z spider, `1` on an X spider,
`2` on a W node. -/
def Node.ports (m : ℕ) (n : Node) : List (ℕ × ℕ) :=
  match n.kind with
  | .ZB cs => portLegs m 0 (tailOf n.ins cs) cs
  | .XB co ci => portLegs m 1 (tailOf n.outs co) co ++ portLegs m 1 (tailOf n.ins ci) ci
  | .WB cs => portLegs m 2 (tailOf n.outs cs) cs
  | _ => []

/-- A network's legs bundled along `m` outermost, with their colours. -/
def Net.ports (m : ℕ) (N : Net) : List (ℕ × ℕ) := N.nodes.flatMap (Node.ports m)

/-- A Z or X spider or W node whose chains starting at `m` are `[m]`, or a node with no bundle
along `m`. -/
def portNode (S : Spec) (n : Node) : Bool :=
  match n.kind with
  | .ZB cs => cs.all fun c => c.head? != some S.m || c == [S.m]
  | .XB co ci => (co ++ ci).all fun c => c.head? != some S.m || c == [S.m]
  | .Bun m _ => m != S.m && n.labels.all fun l => !S.isB l
  | .WB cs => cs.all fun c => c.head? != some S.m || c == [S.m]
  | _ => n.labels.all fun l => !S.isB l

/-- Every node fit, the port labels distinct, no summed label a bundle along `m`. -/
def Spec.portNet (S : Spec) (N : Net) : Bool :=
  N.nodes.all (portNode S) && ((N.ports S.m).map Prod.fst).Nodup &&
    N.int.all fun l => !S.isB l

/-- Both networks fit, with the same port legs. -/
def Spec.portOK (S : Spec) (X Y : Net) : Bool :=
  S.portNet X && S.portNet Y && decide ((X.ports S.m).Perm (Y.ports S.m))

/-- The factor of the peeled legs. -/
noncomputable def portC (E : Env) (H : ℕ) (P : List (ℕ × ℕ)) (σ : ℕ → ℕ) : ℂ :=
  (P.map fun p => if p.2 = 1 then (rt (labDim E (p.1 + H)))⁻¹
    else if p.2 = 2 then (if σ p.1 = 0 ∨ σ (p.1 + H) = 0 then 1 else 0)
    else if σ (p.1 + H) = σ p.1 % labDim E (p.1 + H) then 1 else 0).prod

/-- `σ` with each X or W bundle's lowest digit advanced by its peeled leg. -/
noncomputable def portG (S : Spec) (E : Env) (H : ℕ) (P : List (ℕ × ℕ)) (σ : ℕ → ℕ)
    (x : ℕ) : ℕ :=
  if S.isB x = true ∧ ((x, 1) ∈ P ∨ (x, 2) ∈ P) then adj (labDim E (x + H)) (σ x) (σ (x + H)) else σ x

theorem peel_fst (m H : ℕ) (c : ℕ) (tl : List ℕ) (cs : List (List ℕ)) :
    (peel m H tl cs).1 = (portLegs m c tl cs).map (·.1 + H) := by
  induction tl generalizing cs with
  | nil => simp [peel, portLegs]
  | cons x tl ih =>
    cases cs with
    | nil => simp [peel, portLegs]
    | cons ch cs =>
      have := ih cs
      simp only [peel] at this ⊢
      simp only [List.zip_cons_cons, List.filter_cons, portLegs]
      by_cases h : ch.head? = some m
      · simp [h, this]
      · simp [h, this]

theorem peel_snd (m H : ℕ) (tl : List ℕ) (cs : List (List ℕ))
    (h : ∀ c ∈ cs, c.head? = some m → c = [m]) : ∀ c ∈ (peel m H tl cs).2, c = [] := by
  intro c hc
  simp only [peel, List.mem_map, List.mem_filter, beq_iff_eq] at hc
  obtain ⟨⟨x, ch⟩, ⟨hz, hh⟩, rfl⟩ := hc
  rw [h ch (List.of_mem_zip hz).2 hh]
  rfl

theorem tailVals_nil (mult : ℕ → ℕ) (d : ℕ) (vs : List ℕ) (cs : List (List ℕ))
    (h : ∀ c ∈ cs, c = []) (hl : vs.length = cs.length) : tailVals mult d vs cs = vs := by
  induction vs generalizing cs with
  | nil => simp [tailVals]
  | cons v vs ih =>
    cases cs with
    | nil => simp at hl
    | cons c cs =>
      have hc : c = [] := h c List.mem_cons_self
      subst hc
      have := ih cs (fun c hc => h c (List.mem_cons_of_mem _ hc)) (by simpa using hl)
      simp only [tailVals, List.zipWith_cons_cons, List.flatten_cons] at this ⊢
      rw [this]
      simp [digsC]

theorem digsC_one (mult : ℕ → ℕ) (d m v : ℕ) : digsC mult d [m] v = digs d (mult m) v := by
  simp [digsC]

theorem portLegs_sub (m : ℕ) (c : ℕ) (tl : List ℕ) (cs : List (List ℕ)) :
    ∀ p ∈ portLegs m c tl cs, p.1 ∈ tl ∧ p.2 = c := by
  induction tl generalizing cs with
  | nil => simp [portLegs]
  | cons x tl ih =>
    cases cs with
    | nil => simp [portLegs]
    | cons ch cs =>
      intro p hp
      simp only [portLegs] at hp
      split_ifs at hp
      · rcases List.mem_cons.1 hp with rfl | hp
        · simp
        · exact ⟨List.mem_cons_of_mem _ (ih cs p hp).1, (ih cs p hp).2⟩
      · exact ⟨List.mem_cons_of_mem _ (ih cs p hp).1, (ih cs p hp).2⟩

/-! ## One spider's legs -/

/-- The Z spider with one more leg pinned to a value already on it. -/
theorem tZ_cons_of_mem (d : ℕ) (φ : ℕ → ℝ) (y z : ℕ) (L : List ℕ) (hz : z ∈ L) :
    tZ d φ (y :: L) = (if y = z then 1 else 0) * tZ d φ L := by
  unfold tZ
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases hy : y = z
  · subst hy
    by_cases hall : ∀ v ∈ L, v = k
    · have h1 : ∀ v ∈ y :: L, v = k := fun v hv => by
        rcases List.mem_cons.1 hv with rfl | hv
        exacts [hall v hz, hall v hv]
      rw [if_pos h1, if_pos hall, if_pos rfl, one_mul]
    · have h1 : ¬ ∀ v ∈ y :: L, v = k := fun h => hall fun v hv => h v (List.mem_cons_of_mem _ hv)
      rw [if_neg h1, if_neg hall, mul_zero]
  · have h1 : ¬ ∀ v ∈ y :: L, v = k := fun h => hy ((h y List.mem_cons_self).trans
      (h z (List.mem_cons_of_mem _ hz)).symm)
    rw [if_neg h1, if_neg hy, zero_mul]

theorem tZ_append_pinned (d : ℕ) (φ : ℕ → ℝ) (L : List ℕ) (xs : List ℕ) (f g : ℕ → ℕ)
    (hg : ∀ x ∈ xs, g x ∈ L) :
    tZ d φ (L ++ xs.map f) = (xs.map fun x => if f x = g x then (1 : ℂ) else 0).prod *
      tZ d φ L := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hp : (L ++ (x :: xs).map f).Perm (f x :: (L ++ xs.map f)) := by
      simp only [List.map_cons]; exact List.perm_middle
    rw [tZ_perm _ _ _ _ hp, tZ_cons_of_mem d φ (f x) (g x) _
      (List.mem_append_left _ (hg x List.mem_cons_self)),
      ih fun y hy => hg y (List.mem_cons_of_mem _ hy)]
    simp

/-- The leg facts the port lemmas read: chain, root and peeled dimension of each leg. -/
structure LegFacts (S : Spec) (H k dim : ℕ) (tl : List ℕ) (cs : List (List ℕ)) : Prop where
  len : tl.length = cs.length
  ok : S.legsOK dim tl cs = true
  chains : S.chainsOK cs = true
  port : ∀ c ∈ cs, c.head? = some S.m → c = [S.m]
  good : ∀ x ∈ tl, S.good (labIdx S.F.labs x) = true
  hat : ∀ x ∈ tl, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x

theorem LegFacts.tail {S : Spec} {H k dim x : ℕ} {ch : List ℕ} {tl : List ℕ} {cs : List (List ℕ)} :
    LegFacts S H k dim (x :: tl) (ch :: cs) → LegFacts S H k dim tl cs := by
  intro h
  refine ⟨by simpa using h.len, ?_, ?_, fun c hc => h.port c (List.mem_cons_of_mem _ hc),
    fun y hy => h.good y (List.mem_cons_of_mem _ hy),
    fun y hy => h.hat y (List.mem_cons_of_mem _ hy)⟩
  · have := h.ok; rw [S.legsOK_cons] at this; simp only [Bool.and_eq_true] at this; exact this.2
  · have := h.chains; simp only [Spec.chainsOK, List.all_cons, Bool.and_eq_true] at this ⊢
    exact this.2

theorem LegFacts.head {S : Spec} {H k dim x : ℕ} {ch : List ℕ} {tl : List ℕ} {cs : List (List ℕ)}
    (D : SplitData S H) (h : LegFacts S H k dim (x :: tl) (ch :: cs)) :
    (ch.head? = some S.m → ch = [S.m] ∧ S.isB x = true ∧
      labDim (S.at k) (x + H) = S.F.dims dim) ∧
    (ch.head? ≠ some S.m → S.isB x = false) := by
  have hok := h.ok
  rw [S.legsOK_cons] at hok
  simp only [Bool.and_eq_true, beq_iff_eq] at hok
  obtain ⟨⟨hc, hr⟩, -⟩ := hok
  refine ⟨fun hh => ?_, fun hh => ?_⟩
  · have hch := h.port ch List.mem_cons_self hh
    subst hch
    obtain ⟨hb, -⟩ := S.isB_of_chain D.wf x [] hc
    refine ⟨rfl, hb, ?_⟩
    rw [S.labDim_hat H k x D (h.good x List.mem_cons_self) hb (h.hat x List.mem_cons_self hb),
      S.base_chain D.wf x dim [] hc hr]
    simp
  · refine S.not_isB_of_chain D.wf x ?_
    rw [hc]
    have hcs := h.chains
    simp only [Spec.chainsOK, List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hcs
    intro hm
    cases ch with
    | nil => simp at hm
    | cons a t =>
      rcases List.mem_cons.1 hm with he | ht
      · exact hh (by simp [he])
      · exact hcs.1 ht

/-- Each Z bundle's lowest digit is among the bundled values. -/
theorem port_mem (S : Spec) (H k dim : ℕ) (D : SplitData S H) (hk : 1 ≤ k) (c : ℕ)
    (tl : List ℕ) (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs) (τ : ℕ → ℕ) :
    ∀ p ∈ portLegs S.m c tl cs,
      τ p.1 % S.F.dims dim ∈ tailVals (S.at k).mult (S.F.dims dim) (tl.map τ) cs := by
  induction tl generalizing cs with
  | nil => simp [portLegs]
  | cons x tl ih =>
    cases cs with
    | nil => simp [portLegs]
    | cons ch cs =>
      intro p hp
      have hrest := ih cs F.tail
      simp only [tailVals, List.map_cons, List.zipWith_cons_cons, List.flatten_cons] at hrest ⊢
      simp only [portLegs] at hp
      split_ifs at hp with hh
      · rcases List.mem_cons.1 hp with rfl | hp
        · refine List.mem_append_left _ ?_
          obtain ⟨hch, -, -⟩ := (F.head (k := k) D).1 hh
          subst hch
          rw [digsC_one, S.mult_self]
          obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
          rw [digs_cons, digit_zero]
          exact List.mem_cons_self
        · exact List.mem_append_right _ (hrest p hp)
      · exact List.mem_append_right _ (hrest p hp)

/-- The values of X bundles advanced by their peeled legs, through a character-like `f`. -/
theorem port_prod (S : Spec) (H k dim : ℕ) (D : SplitData S H) (hk : 1 ≤ k)
    (hd : 0 < S.F.dims dim) (c : ℕ)
    (tl : List ℕ) (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs) (τ g : ℕ → ℕ)
    (hg1 : ∀ x ∈ tl, S.isB x = true → g x = adj (S.F.dims dim) (τ x) (τ (x + H)))
    (hg0 : ∀ x ∈ tl, S.isB x = false → g x = τ x) (f : ℕ → ℂ) (r : ℂ)
    (hf : ∀ a e, f ((a + e) % S.F.dims dim) = r * f a * f e) :
    ((tailVals (S.at k).mult (S.F.dims dim) (tl.map g) cs).map f).prod =
      ((portLegs S.m c tl cs).map fun p => r * f (τ (p.1 + H))).prod *
        ((tailVals (S.at k).mult (S.F.dims dim) (tl.map τ) cs).map f).prod := by
  induction tl generalizing cs with
  | nil => simp [tailVals, portLegs]
  | cons x tl ih =>
    cases cs with
    | nil => simp [tailVals, portLegs]
    | cons ch cs =>
      have hrest := ih cs F.tail (fun y hy => hg1 y (List.mem_cons_of_mem _ hy))
        (fun y hy => hg0 y (List.mem_cons_of_mem _ hy))
      simp only [tailVals, List.map_cons, List.zipWith_cons_cons, List.flatten_cons,
        List.map_append, List.prod_append] at hrest ⊢
      rw [hrest]
      simp only [portLegs]
      by_cases hh : ch.head? = some S.m
      · obtain ⟨hch, hb, -⟩ := (F.head (k := k) D).1 hh
        subst hch
        rw [if_pos hh, hg1 x List.mem_cons_self hb, digsC_one, digsC_one, S.mult_self]
        obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
        rw [digs_adj _ _ _ _ hd, digs_cons, digit_zero]
        simp only [List.map_cons, List.prod_cons, List.map_cons, hf]
        ring
      · rw [if_neg hh, hg0 x List.mem_cons_self ((F.head (k := k) D).2 hh)]
        ring

theorem port_of_isB (S : Spec) (H k dim : ℕ) (D : SplitData S H) (c : ℕ) (tl : List ℕ)
    (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs) (x : ℕ) (hx : x ∈ tl)
    (hb : S.isB x = true) :
    (x, c) ∈ portLegs S.m c tl cs ∧ labDim (S.at k) (x + H) = S.F.dims dim := by
  induction tl generalizing cs with
  | nil => simp at hx
  | cons y tl ih =>
    cases cs with
    | nil => have := F.len; simp at this
    | cons ch cs =>
      simp only [portLegs]
      by_cases hh : ch.head? = some S.m
      · rw [if_pos hh]
        rcases List.mem_cons.1 hx with rfl | hx'
        · exact ⟨List.mem_cons_self, ((F.head (k := k) D).1 hh).2.2⟩
        · obtain ⟨a, b⟩ := ih cs F.tail hx'
          exact ⟨List.mem_cons_of_mem _ a, b⟩
      · rw [if_neg hh]
        rcases List.mem_cons.1 hx with rfl | hx'
        · rw [(F.head (k := k) D).2 hh] at hb; simp at hb
        · exact ih cs F.tail hx'

theorem portC_legs (S : Spec) (H k dim : ℕ) (D : SplitData S H) (c : ℕ) (tl : List ℕ)
    (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs) (τ : ℕ → ℕ) :
    portC (S.at k) H (portLegs S.m c tl cs) τ = ((portLegs S.m c tl cs).map fun p =>
      if c = 1 then (rt (S.F.dims dim))⁻¹
      else if c = 2 then (if τ p.1 = 0 ∨ τ (p.1 + H) = 0 then (1 : ℂ) else 0)
      else if τ (p.1 + H) = τ p.1 % S.F.dims dim then (1 : ℂ) else 0).prod := by
  unfold portC
  congr 1
  refine List.map_congr_left fun p hp => ?_
  obtain ⟨h1, h2⟩ := portLegs_sub _ _ _ _ p hp
  have hb : S.isB p.1 = true := by
    induction tl generalizing cs with
    | nil => simp [portLegs] at hp
    | cons y tl ih =>
      cases cs with
      | nil => simp [portLegs] at hp
      | cons ch cs =>
        simp only [portLegs] at hp
        split_ifs at hp with hh
        · rcases List.mem_cons.1 hp with rfl | hp
          · exact ((F.head (k := k) D).1 hh).2.1
          · exact ih cs F.tail hp (portLegs_sub _ _ _ _ p hp).1
        · exact ih cs F.tail hp (portLegs_sub _ _ _ _ p hp).1
  rw [(port_of_isB S H k dim D c tl cs F p.1 h1 hb).2, h2]

/-- The legs of a spider kind: plain head legs and the facts of its bundled tail. -/
theorem legFacts_of (S : Spec) (H k dim : ℕ) (ls : List ℕ) (cs : List (List ℕ))
    (hl : S.legsOK dim ls (List.replicate (ls.length - cs.length) [] ++ cs) = true)
    (hc : S.chainsOK cs = true) (hp : ∀ c ∈ cs, c.head? = some S.m → c = [S.m])
    (hg : ∀ x ∈ ls, S.good (labIdx S.F.labs x) = true)
    (hh : ∀ x ∈ ls, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x)
    (D : SplitData S H) :
    LegFacts S H k dim (tailOf ls cs) cs ∧ ∀ x ∈ headOf ls cs, S.isB x = false := by
  obtain ⟨hle, h1, h2⟩ := S.legsOK_split dim ls cs hl
  have hsub : ∀ x ∈ tailOf ls cs, x ∈ ls := fun x hx => List.mem_of_mem_drop hx
  refine ⟨⟨by simp [tailOf]; omega, h2, hc, hp, fun x hx => hg x (hsub x hx),
    fun x hx => hh x (hsub x hx)⟩, ?_⟩
  exact S.legs_plain D.wf dim _ _ h1

/-! ## W bundles -/

theorem adj_zero (d v : ℕ) : adj d v 0 = v := by
  unfold adj; rw [Nat.add_zero, Nat.mod_mod]; have := Nat.mod_le v d; omega

theorem digit_zero_val (d i : ℕ) : digit d i 0 = 0 := by simp [digit]

theorem digs_zero (d k : ℕ) : digs d k 0 = List.replicate k 0 := by
  simp [digs, digit_zero_val, List.map_const']

theorem eq_zero_of_digs (d : ℕ) (hd : 0 < d) : ∀ (k v : ℕ), v < d ^ k →
    (∀ x ∈ digs d k v, x = 0) → v = 0 := by
  intro k
  induction k with
  | zero => intro v hv _; simpa using hv
  | succ k ih =>
    intro v hv h
    rw [digs_cons] at h
    have h0 : v % d = 0 := by simpa [digit_zero] using h _ List.mem_cons_self
    have hq : v / d = 0 := by
      refine ih (v / d) ?_ fun x hx => ?_
      · rw [Nat.div_lt_iff_lt_mul hd]; rw [pow_succ] at hv; exact hv
      · simp only [digs, List.mem_map, List.mem_range] at hx
        obtain ⟨i, hi, rfl⟩ := hx
        have := h (digit d (i + 1) v) (List.mem_cons_of_mem _ (List.mem_map.2 ⟨i,
          List.mem_range.2 hi, rfl⟩))
        simpa [digit, pow_succ, mul_comm (d ^ i), ← Nat.div_div_eq_div_mul] using this
    have := Nat.div_add_mod v d
    rw [hq, h0] at this; omega

/-- A W node with nonzero values in two parts of its outputs is zero. -/
theorem tW_two (d : ℕ) (A B : List ℕ) (i : ℕ) (ha : ∃ a ∈ A, a ≠ 0) (hb : ∃ b ∈ B, b ≠ 0) :
    tW d (A ++ B) i = 0 := by
  rw [tW_eq]
  obtain ⟨a, haA, ha0⟩ := ha
  obtain ⟨b, hbB, hb0⟩ := hb
  have h1 : nz (A ++ B) ≠ [] := by
    rw [ne_eq, nz_nil_iff]; intro h; exact ha0 (h a (List.mem_append_left _ haA))
  have h2 : 2 ≤ (nz (A ++ B)).length := by
    simp only [nz, List.filter_append, List.length_append]
    have p1 : 0 < (A.filter (· ≠ 0)).length :=
      List.length_pos_of_mem (List.mem_filter.2 ⟨haA, by simpa using ha0⟩)
    have p2 : 0 < (B.filter (· ≠ 0)).length :=
      List.length_pos_of_mem (List.mem_filter.2 ⟨hbB, by simpa using hb0⟩)
    omega
  by_cases hi : i = 0
  · rw [if_pos hi, if_neg h1]
  · rw [if_neg hi, if_neg]
    rintro ⟨-, h4⟩
    rw [h4] at h2; simp at h2

theorem tW_zeros (d : ℕ) (A : List ℕ) (r i : ℕ) : tW d (A ++ List.replicate r 0) i = tW d A i := by
  rw [tW_eq, tW_eq]
  have : nz (A ++ List.replicate r 0) = nz A := by simp [nz, List.filter_append]
  rw [this]

theorem perm_four (a T E a' T' Z : List ℕ) (e z : ℕ) (h1 : (a ++ [e]).Perm (a' ++ [z]))
    (h2 : (T ++ E).Perm (T' ++ Z)) : ((a ++ T) ++ (e :: E)).Perm ((a' ++ T') ++ (z :: Z)) := by
  have l : ∀ (a T E : List ℕ) (e : ℕ), ((a ++ T) ++ (e :: E)).Perm ((a ++ [e]) ++ (T ++ E)) := by
    intro a T E e
    simp only [List.append_assoc, List.singleton_append]
    exact List.Perm.append_left _ List.perm_middle
  exact (l a T E e).trans ((h1.append h2).trans (l a' T' Z z).symm)

theorem digs_peel_zero (d j e : ℕ) (hd : 0 < d) (he : e < d) :
    (digs d (j + 1) 0 ++ [e]).Perm (digs d (j + 1) (adj d 0 e) ++ [0]) := by
  rw [digs_zero, digs_adj _ _ _ _ hd, Nat.zero_mod, Nat.zero_add, Nat.mod_eq_of_lt he]
  have : (List.range j).map (fun i => digit d (i + 1) 0) = List.replicate j 0 := by
    simp [digit_zero_val, List.map_const']
  rw [this, List.cons_append, ← List.replicate_succ']
  exact List.perm_append_singleton _ _

/-- A W bundle's digits and its peeled leg against the advanced digits and a zero. -/
theorem port_wperm (S : Spec) (H k dim : ℕ) (D : SplitData S H) (hk : 1 ≤ k)
    (hd : 0 < S.F.dims dim) (tl : List ℕ) (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs)
    (τ g : ℕ → ℕ)
    (hg1 : ∀ x ∈ tl, S.isB x = true → g x = adj (S.F.dims dim) (τ x) (τ (x + H)))
    (hg0 : ∀ x ∈ tl, S.isB x = false → g x = τ x)
    (hz : ∀ p ∈ portLegs S.m 2 tl cs, τ p.1 = 0 ∨ τ (p.1 + H) = 0)
    (hr : ∀ p ∈ portLegs S.m 2 tl cs, τ (p.1 + H) < S.F.dims dim) :
    (tailVals (S.at k).mult (S.F.dims dim) (tl.map τ) cs ++
        (portLegs S.m 2 tl cs).map (fun p => τ (p.1 + H))).Perm
      (tailVals (S.at k).mult (S.F.dims dim) (tl.map g) cs ++
        List.replicate (portLegs S.m 2 tl cs).length 0) := by
  induction tl generalizing cs with
  | nil => simp [tailVals, portLegs]
  | cons x tl ih =>
    cases cs with
    | nil => simp [tailVals, portLegs]
    | cons ch cs =>
      have hrest := ih cs F.tail (fun y hy => hg1 y (List.mem_cons_of_mem _ hy))
        (fun y hy => hg0 y (List.mem_cons_of_mem _ hy))
      simp only [tailVals, List.map_cons, List.zipWith_cons_cons, List.flatten_cons] at hrest ⊢
      simp only [portLegs] at hz hr ⊢
      by_cases hh : ch.head? = some S.m
      · obtain ⟨hch, hb, -⟩ := (F.head (k := k) D).1 hh
        subst hch
        simp only [if_pos hh, List.map_cons, List.length_cons, List.replicate_succ] at hz hr ⊢
        have hrest' := hrest (fun p hp => hz p (List.mem_cons_of_mem _ hp))
          (fun p hp => hr p (List.mem_cons_of_mem _ hp))
        rw [hg1 x List.mem_cons_self hb, digsC_one, digsC_one, S.mult_self]
        have hx : (digs (S.F.dims dim) k (τ x) ++ [τ (x + H)]).Perm
            (digs (S.F.dims dim) k (adj (S.F.dims dim) (τ x) (τ (x + H))) ++ [0]) := by
          rcases hz _ List.mem_cons_self with h0 | h0
          · obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
            have he : τ (x + H) < S.F.dims dim := hr _ List.mem_cons_self
            simp only at h0
            rw [h0]
            exact digs_peel_zero _ j _ hd he
          · rw [h0, adj_zero]
        exact perm_four _ _ _ _ _ _ _ _ hx hrest'
      · simp only [if_neg hh] at hz hr ⊢
        rw [hg0 x List.mem_cons_self ((F.head (k := k) D).2 hh)]
        simp only [List.append_assoc]
        exact List.Perm.append_left _ (hrest hz hr)

/-- A nonzero W bundle beside a nonzero peeled leg. -/
theorem port_wtwo (S : Spec) (H k dim : ℕ) (D : SplitData S H) (hk : 1 ≤ k)
    (hd : 0 < S.F.dims dim) (tl : List ℕ) (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs)
    (τ : ℕ → ℕ) (p : ℕ × ℕ) (hp : p ∈ portLegs S.m 2 tl cs) (hx : τ p.1 ≠ 0)
    (hxr : τ p.1 < S.F.dims dim ^ k) :
    ∃ a ∈ tailVals (S.at k).mult (S.F.dims dim) (tl.map τ) cs, a ≠ 0 := by
  induction tl generalizing cs with
  | nil => simp [portLegs] at hp
  | cons x tl ih =>
    cases cs with
    | nil => simp [portLegs] at hp
    | cons ch cs =>
      simp only [tailVals, List.map_cons, List.zipWith_cons_cons, List.flatten_cons]
      simp only [portLegs] at hp
      split_ifs at hp with hh
      · rcases List.mem_cons.1 hp with rfl | hp'
        · obtain ⟨hch, -, -⟩ := (F.head (k := k) D).1 hh
          subst hch
          rw [digsC_one, S.mult_self]
          by_contra hn
          push Not at hn
          exact hx (eq_zero_of_digs _ hd k _ hxr fun a ha => hn a (List.mem_append_left _ ha))
        · obtain ⟨a, ha, h0⟩ := ih cs F.tail hp'
          exact ⟨a, List.mem_append_right _ ha, h0⟩
      · obtain ⟨a, ha, h0⟩ := ih cs F.tail hp
        exact ⟨a, List.mem_append_right _ ha, h0⟩

/-! ## One node -/

theorem bundled_tail (mult : ℕ → ℕ) (d : ℕ) (σ : ℕ → ℕ) (ls : List ℕ) (cs : List (List ℕ)) :
    bundled mult d (ls.map σ) cs = (headOf ls cs).map σ ++
      tailVals mult d ((tailOf ls cs).map σ) cs := by
  simp [bundled, headOf, tailOf, List.map_take, List.map_drop]

theorem bundled_append_tail (mult : ℕ → ℕ) (d : ℕ) (vs ws : List ℕ) (cs bs : List (List ℕ))
    (hv : cs.length ≤ vs.length) (hw : ws.length = bs.length) :
    bundled mult d (vs ++ ws) (cs ++ bs) = bundled mult d vs cs ++ tailVals mult d ws bs := by
  unfold bundled
  have e : (vs ++ ws).length - (cs ++ bs).length = vs.length - cs.length := by
    simp [List.length_append, hw]; omega
  rw [e, List.take_append_of_le_length (by omega), List.drop_append_of_le_length (by omega),
    tailVals_append _ _ _ _ _ _ (by simp; omega), List.append_assoc]

theorem peel_empty (m H : ℕ) (tl : List ℕ) (cs : List (List ℕ))
    (h : ∀ c ∈ cs, c.head? ≠ some m) : peel m H tl cs = ([], []) := by
  have : (List.zip tl cs).filter (fun p => p.2.head? == some m) = [] := by
    rw [List.filter_eq_nil_iff]
    intro p hp
    simpa using h p.2 (List.of_mem_zip hp).2
  simp [peel, this]

theorem nodup_fst {P : List (ℕ × ℕ)} (h : (P.map Prod.fst).Nodup) {x : ℕ} {a b : ℕ}
    (ha : (x, a) ∈ P) (hb : (x, b) ∈ P) : a = b := by
  have := List.inj_on_of_nodup_map h ha hb rfl
  simpa using this

theorem prod_map_const_mul {α : Type*} (l : List α) (r : ℂ) (f : α → ℂ) :
    (l.map fun p => r * f p).prod = r ^ l.length * (l.map f).prod := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.prod_cons, ih, List.length_cons]; ring

theorem prod_map_const {α : Type*} (l : List α) (r : ℂ) : (l.map fun _ => r).prod = r ^ l.length := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.prod_cons, ih, List.length_cons]; ring

theorem portC_append (E : Env) (H : ℕ) (a b : List (ℕ × ℕ)) (σ : ℕ → ℕ) :
    portC E H (a ++ b) σ = portC E H a σ * portC E H b σ := by
  simp [portC, List.prod_append]

theorem portG_still (S : Spec) (E : Env) (H : ℕ) (P : List (ℕ × ℕ)) (τ : ℕ → ℕ) (x : ℕ)
    (h : S.isB x = false) : portG S E H P τ x = τ x := by
  unfold portG; rw [if_neg]; simp [h]

theorem portLegs_isB (S : Spec) (H k dim : ℕ) (D : SplitData S H) (c : ℕ) (tl : List ℕ)
    (cs : List (List ℕ)) (F : LegFacts S H k dim tl cs) :
    ∀ p ∈ portLegs S.m c tl cs, S.isB p.1 = true := by
  induction tl generalizing cs with
  | nil => simp [portLegs]
  | cons y tl ih =>
    cases cs with
    | nil => simp [portLegs]
    | cons ch cs =>
      intro p hp
      simp only [portLegs] at hp
      split_ifs at hp with hh
      · rcases List.mem_cons.1 hp with rfl | hp
        · exact ((F.head (k := k) D).1 hh).2.1
        · exact ih cs F.tail p hp
      · exact ih cs F.tail p hp

/-- A node's split entry: the port factor times its entry at the advanced assignment. -/
theorem port_node (S : Spec) (H k : ℕ) (D : SplitData S H) (hk : 1 ≤ k) (P : List (ℕ × ℕ))
    (hnd : (P.map Prod.fst).Nodup) (n : Node) (hn : S.nodeOK n = true)
    (hp : portNode S n = true) (hsub : ∀ q ∈ n.ports S.m, q ∈ P)
    (hg : ∀ x ∈ n.labels, S.good (labIdx S.F.labs x) = true)
    (hh : ∀ x ∈ n.labels, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x)
    (hpos : 0 < S.F.dims n.dim) (τ : ℕ → ℕ)
    (hr : ∀ q ∈ n.ports S.m, τ q.1 < labDim (S.at k) q.1 ∧ τ (q.1 + H) < labDim (S.at k) (q.1 + H)) :
    prodVal (S.at k) (Node.split S.m H n) τ =
      portC (S.at k) H (n.ports S.m) τ * val (S.at k) (portG S (S.at k) H P τ) n := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  have hn' := hn
  simp only [Spec.nodeOK, Bool.and_eq_true] at hn
  obtain ⟨hdim, hn⟩ := hn
  have hdk : (S.at k).dimv dim = S.F.dims dim := S.dimv_plain _ _ hdim
  have hgo : ∀ x ∈ os, S.good (labIdx S.F.labs x) = true := fun x hx =>
    hg x (by simp [Node.labels, hx])
  have hgi : ∀ x ∈ is, S.good (labIdx S.F.labs x) = true := fun x hx =>
    hg x (by simp [Node.labels, hx])
  have hho : ∀ x ∈ os, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x := fun x hx =>
    hh x (by simp [Node.labels, hx])
  have hhi : ∀ x ∈ is, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x := fun x hx =>
    hh x (by simp [Node.labels, hx])
  have plain : (∀ x ∈ os ++ is, S.isB x = false) →
      Node.split S.m H ⟨kind, os, is, dim, ph⟩ = [⟨kind, os, is, dim, ph⟩] →
      Node.ports S.m ⟨kind, os, is, dim, ph⟩ = [] →
      prodVal (S.at k) (Node.split S.m H ⟨kind, os, is, dim, ph⟩) τ =
        portC (S.at k) H (Node.ports S.m ⟨kind, os, is, dim, ph⟩) τ *
          val (S.at k) (portG S (S.at k) H P τ) ⟨kind, os, is, dim, ph⟩ := by
    intro hl hs hpo
    rw [hs, hpo]
    simp only [prodVal, portC, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      mul_one, one_mul]
    exact val_reads _ _ _ _ fun x hx => (portG_still S _ H P τ x (hl x hx)).symm
  have labs_plain : ((Node.labels ⟨kind, os, is, dim, ph⟩).all fun l => !S.isB l) = true →
      ∀ l ∈ os ++ is, S.isB l = false := fun h l hl => by
    simpa using List.all_eq_true.1 h l (by simpa [Node.labels] using hl)
  cases kind with
  | ZB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, hli⟩ := hn
    have hpc : ∀ c ∈ cs, c.head? = some S.m → c = [S.m] := by
      intro c hc hm
      have := List.all_eq_true.1 hp c hc
      simpa [hm] using this
    obtain ⟨F, hhd⟩ := legFacts_of S H k dim is cs hli hch hpc hgi hhi D
    have hos := S.legs_plain D.wf dim os _ hlo
    obtain ⟨hle, -, -⟩ := S.legsOK_split dim is cs hli
    have hgis : is.map (portG S (S.at k) H P τ) = is.map τ := by
      refine List.map_congr_left fun x hx => ?_
      cases hb : S.isB x
      · exact portG_still S _ H P τ x hb
      · unfold portG
        rw [if_neg]
        rintro ⟨-, ht⟩
        have hxt : x ∈ tailOf is cs := by
          rw [← headOf_tailOf is cs] at hx
          rcases List.mem_append.1 hx with h | h
          · rw [hhd x h] at hb; simp at hb
          · exact h
        have hf := (port_of_isB S H k dim D 0 _ cs F x hxt hb).1
        rcases ht with ht | ht <;>
          · have := nodup_fst hnd (hsub _ (by simpa [Node.ports] using hf)) ht
            simp at this
    have hgos : os.map (portG S (S.at k) H P τ) = os.map τ :=
      List.map_congr_left fun x hx => portG_still S _ H P τ x (hos x hx)
    have hlen := peel_len S.m H (tailOf is cs) cs
    simp only [Node.split, Node.ports, prodVal, List.map_cons, List.map_nil, List.prod_cons,
      List.prod_nil, mul_one, val, hdk, hgis, hgos]
    rw [List.map_append, bundled_append_tail _ _ _ _ _ _ (by simp; omega) (by simp [hlen]),
      tailVals_nil _ _ _ _ (peel_snd _ _ _ _ hpc) (by simp [hlen]), peel_fst S.m H 0,
      ← List.append_assoc, List.map_map, portC_legs S H k dim D 0 _ cs F τ]
    have hfst : ((portLegs S.m 0 (tailOf is cs) cs).map (τ ∘ (·.1 + H))) =
        ((portLegs S.m 0 (tailOf is cs) cs).map Prod.fst).map (fun x => τ (x + H)) := by
      simp [List.map_map, Function.comp_def]
    rw [hfst, tZ_append_pinned _ _ _ _ _ (fun x => τ x % S.F.dims dim)]
    · simp [List.map_map, Function.comp_def]
    · intro x hx
      obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hx
      refine List.mem_append_right _ ?_
      rw [bundled_tail]
      exact List.mem_append_right _ (port_mem S H k dim D hk 0 _ cs F τ q hq)
  | XB co ci =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨⟨hco, hci⟩, hlo⟩, hli⟩ := hn
    have hpc : ∀ c ∈ co ++ ci, c.head? = some S.m → c = [S.m] := by
      intro c hc hm
      have := List.all_eq_true.1 hp c hc
      simpa [hm] using this
    obtain ⟨Fo, hhdo⟩ := legFacts_of S H k dim os co hlo hco
      (fun c hc => hpc c (List.mem_append_left _ hc)) hgo hho D
    obtain ⟨Fi, hhdi⟩ := legFacts_of S H k dim is ci hli hci
      (fun c hc => hpc c (List.mem_append_right _ hc)) hgi hhi D
    obtain ⟨hleo, -, -⟩ := S.legsOK_split dim os co hlo
    obtain ⟨hlei, -, -⟩ := S.legsOK_split dim is ci hli
    have hlo' := peel_len S.m H (tailOf os co) co
    have hli' := peel_len S.m H (tailOf is ci) ci
    set g := portG S (S.at k) H P τ with hgdef
    have hX : ∀ (ls : List ℕ) (cs : List (List ℕ)), LegFacts S H k dim (tailOf ls cs) cs →
        (∀ q ∈ portLegs S.m 1 (tailOf ls cs) cs, q ∈ P) →
        (∀ x ∈ tailOf ls cs, S.isB x = true → g x = adj (S.F.dims dim) (τ x) (τ (x + H))) ∧
        (∀ x ∈ tailOf ls cs, S.isB x = false → g x = τ x) := by
      intro ls cs F hs
      refine ⟨fun x hx hb => ?_, fun x _ hb => portG_still S _ H P τ x hb⟩
      obtain ⟨hm, hlab⟩ := port_of_isB S H k dim D 1 _ cs F x hx hb
      rw [hgdef, portG, if_pos ⟨hb, Or.inl (hs _ hm)⟩, hlab]
    have hso : ∀ q ∈ portLegs S.m 1 (tailOf os co) co, q ∈ P := fun q hq =>
      hsub q (by simp [Node.ports, hq])
    have hsi : ∀ q ∈ portLegs S.m 1 (tailOf is ci) ci, q ∈ P := fun q hq =>
      hsub q (by simp [Node.ports, hq])
    obtain ⟨ho1, ho0⟩ := hX os co Fo hso
    obtain ⟨hi1, hi0⟩ := hX is ci Fi hsi
    have hdo : (headOf os co).map g = (headOf os co).map τ :=
      List.map_congr_left fun x hx => portG_still S _ H P τ x (hhdo x hx)
    have hdi : (headOf is ci).map g = (headOf is ci).map τ :=
      List.map_congr_left fun x hx => portG_still S _ H P τ x (hhdi x hx)
    simp only [Node.split, Node.ports, prodVal, List.map_cons, List.map_nil, List.prod_cons,
      List.prod_nil, mul_one, val, hdk]
    rw [List.map_append, List.map_append,
      bundled_append_tail _ _ _ _ _ _ (by simp; omega) (by simp [hlo']),
      bundled_append_tail _ _ _ _ _ _ (by simp; omega) (by simp [hli']),
      tailVals_nil _ _ _ _ (peel_snd _ _ _ _ fun c hc => hpc c (List.mem_append_left _ hc))
        (by simp [hlo']),
      tailVals_nil _ _ _ _ (peel_snd _ _ _ _ fun c hc => hpc c (List.mem_append_right _ hc))
        (by simp [hli']),
      peel_fst S.m H 1, peel_fst S.m H 1, portC_append,
      portC_legs S H k dim D 1 _ co Fo τ, portC_legs S H k dim D 1 _ ci Fi τ,
      bundled_tail, bundled_tail, bundled_tail _ _ g, bundled_tail _ _ g, hdo, hdi]
    simp only [if_true, prod_map_const]
    unfold tX
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun kk _ => ?_
    have hr : rt (S.F.dims dim) ≠ 0 := rt_ne _ hpos
    simp only [List.map_append, List.prod_append]
    rw [port_prod S H k dim D hk hpos 1 _ co Fo τ g ho1 ho0 (fun v => fou (S.F.dims dim) v kk)
        (rt (S.F.dims dim)) (fun a e => fou_add_mod _ a e kk hpos),
      port_prod S H k dim D hk hpos 1 _ ci Fi τ g hi1 hi0
        (fun v => (starRingEnd ℂ) (fou (S.F.dims dim) v kk)) (rt (S.F.dims dim))
        (fun a e => conj_fou_add_mod _ a e kk hpos),
      prod_map_const_mul, prod_map_const_mul]
    simp only [List.map_map, Function.comp_def]
    generalize rt (S.F.dims dim) = r at hr ⊢
    generalize (portLegs S.m 1 (tailOf os co) co).length = a
    generalize (portLegs S.m 1 (tailOf is ci) ci).length = b
    field_simp
    have e : ∀ Y : ℂ, Y * (1 / r) ^ a * (1 / r) ^ b * r ^ a * r ^ b = Y := by
      intro Y
      have ea : (1 / r) ^ a * r ^ a = 1 := by rw [← mul_pow, one_div, inv_mul_cancel₀ hr, one_pow]
      have eb : (1 / r) ^ b * r ^ b = 1 := by rw [← mul_pow, one_div, inv_mul_cancel₀ hr, one_pow]
      calc Y * (1 / r) ^ a * (1 / r) ^ b * r ^ a * r ^ b
          = Y * ((1 / r) ^ a * r ^ a) * ((1 / r) ^ b * r ^ b) := by ring
        _ = Y := by rw [ea, eb]; ring
    rw [e]
  | Bun m b =>
    have hp' := hp
    simp only [portNode, Bool.and_eq_true, bne_iff_ne, ne_eq] at hp'
    refine plain (labs_plain hp'.2) ?_ rfl
    simp [Node.split, hp'.1]
  | WB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, hli⟩ := hn
    have hpc : ∀ c ∈ cs, c.head? = some S.m → c = [S.m] := by
      intro c hc hm
      have := List.all_eq_true.1 hp c hc
      simpa [hm] using this
    obtain ⟨F, hhd⟩ := legFacts_of S H k dim os cs hlo hch.1 hpc hgo hho D
    have his := S.legs_plain D.wf dim is _ hli
    obtain ⟨hle, -, -⟩ := S.legsOK_split dim os cs hlo
    have hlen := peel_len S.m H (tailOf os cs) cs
    set g := portG S (S.at k) H P τ with hgdef
    have hsw : ∀ q ∈ portLegs S.m 2 (tailOf os cs) cs, q ∈ P := fun q hq =>
      hsub q (by simpa [Node.ports] using hq)
    have hg1 : ∀ x ∈ tailOf os cs, S.isB x = true →
        g x = adj (S.F.dims dim) (τ x) (τ (x + H)) := by
      intro x hx hb
      obtain ⟨hm, hlab⟩ := port_of_isB S H k dim D 2 _ cs F x hx hb
      rw [hgdef, portG, if_pos ⟨hb, Or.inr (hsw _ hm)⟩, hlab]
    have hg0 : ∀ x ∈ tailOf os cs, S.isB x = false → g x = τ x :=
      fun x _ hb => portG_still S _ H P τ x hb
    have hdo : (headOf os cs).map g = (headOf os cs).map τ :=
      List.map_congr_left fun x hx => portG_still S _ H P τ x (hhd x hx)
    have hrng : ∀ q ∈ portLegs S.m 2 (tailOf os cs) cs,
        τ q.1 < S.F.dims dim ^ k ∧ τ (q.1 + H) < S.F.dims dim := by
      intro q hq
      obtain ⟨hq1, hq2⟩ := portLegs_sub _ _ _ _ q hq
      have hb := portLegs_isB S H k dim D 2 _ cs F q hq
      obtain ⟨-, hlab⟩ := port_of_isB S H k dim D 2 _ cs F q.1 hq1 hb
      have hgq := hgo q.1 (List.mem_of_mem_drop hq1)
      have hhq := hho q.1 (List.mem_of_mem_drop hq1) hb
      obtain ⟨r1, r2⟩ := hr q (by simpa [Node.ports] using hq)
      rw [S.labDim_bundle H k q.1 D hgq hb] at r1
      rw [hlab] at r2
      rw [← S.labDim_hat H k q.1 D hgq hb hhq, hlab] at r1
      exact ⟨r1, r2⟩
    rcases is with _ | ⟨w, _ | ⟨w2, rest⟩⟩
    · simp [Node.split, prodVal, val]
    swap
    · simp [Node.split, prodVal, val]
    have hw : g w = τ w := portG_still S _ H P τ w (his w List.mem_cons_self)
    simp only [Node.split, Node.ports, prodVal, List.map_cons, List.map_nil, List.prod_cons,
      List.prod_nil, mul_one, val, hdk, hw]
    rw [List.map_append, bundled_append_tail _ _ _ _ _ _ (by simp; omega) (by simp [hlen]),
      tailVals_nil _ _ _ _ (peel_snd _ _ _ _ hpc) (by simp [hlen]), peel_fst S.m H 2,
      portC_legs S H k dim D 2 _ cs F τ, bundled_tail, bundled_tail _ _ g, hdo, List.map_map]
    simp only [show ((2 : ℕ) = 1) = False by decide, show ((2 : ℕ) = 2) = True by decide,
      ite_false, ite_true]
    by_cases hall : ∀ p ∈ portLegs S.m 2 (tailOf os cs) cs, τ p.1 = 0 ∨ τ (p.1 + H) = 0
    · rw [List.prod_eq_one (fun x hx => by
        obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hx
        simp [hall p hp]), one_mul]
      have hperm := port_wperm S H k dim D hk hpos _ cs F τ g hg1 hg0 hall
        (fun p hp => (hrng p hp).2)
      rw [tW_perm _ _ ((List.map τ (headOf os cs) ++
          tailVals (S.at k).mult (S.F.dims dim) (List.map g (tailOf os cs)) cs) ++
          List.replicate (portLegs S.m 2 (tailOf os cs) cs).length 0) _ ?_, tW_zeros]
      simp only [List.append_assoc, Function.comp_def]
      exact List.Perm.append_left _ hperm
    · push Not at hall
      obtain ⟨p, hp, h1, h2⟩ := hall
      rw [List.prod_eq_zero (List.mem_map.2 ⟨p, hp, by simp [h1, h2]⟩), zero_mul]
      obtain ⟨a, ha, ha0⟩ := port_wtwo S H k dim D hk hpos _ cs F τ p hp h1 (hrng p hp).1
      exact tW_two _ _ _ _ ⟨a, List.mem_append_right _ ha, ha0⟩
        ⟨τ (p.1 + H), List.mem_map.2 ⟨p, hp, rfl⟩, h2⟩
  | Z => exact plain (labs_plain hp) rfl rfl
  | X => exact plain (labs_plain hp) rfl rfl
  | F => exact plain (labs_plain hp) rfl rfl
  | T => exact plain (labs_plain hp) rfl rfl
  | Ti => exact plain (labs_plain hp) rfl rfl
  | W => exact plain (labs_plain hp) rfl rfl
  | B => exact plain (labs_plain hp) rfl rfl
  | S => exact plain (labs_plain hp) rfl rfl
  | I => exact plain (labs_plain hp) rfl rfl
  | ZW => exact plain (labs_plain hp) rfl rfl

/-! ## Networks -/

/-- A port label is a bundle along `m` and a label of its node. -/
theorem Node.ports_isB (S : Spec) (H : ℕ) (D : SplitData S H) (n : Node) (hn : S.nodeOK n = true)
    (hp : portNode S n = true) (hg : ∀ x ∈ n.labels, S.good (labIdx S.F.labs x) = true)
    (hh : ∀ x ∈ n.labels, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x) :
    ∀ q ∈ n.ports S.m, S.isB q.1 = true ∧ q.1 ∈ n.labels := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  simp only [Spec.nodeOK, Bool.and_eq_true] at hn
  obtain ⟨-, hn⟩ := hn
  have hgo : ∀ x ∈ os, S.good (labIdx S.F.labs x) = true := fun x hx =>
    hg x (by simp [Node.labels, hx])
  have hgi : ∀ x ∈ is, S.good (labIdx S.F.labs x) = true := fun x hx =>
    hg x (by simp [Node.labels, hx])
  have hho : ∀ x ∈ os, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x := fun x hx =>
    hh x (by simp [Node.labels, hx])
  have hhi : ∀ x ∈ is, S.isB x = true → labIdx S.F.labs (x + H) = S.baseIdx x := fun x hx =>
    hh x (by simp [Node.labels, hx])
  cases kind with
  | ZB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, -⟩, hli⟩ := hn
    have hpc : ∀ c ∈ cs, c.head? = some S.m → c = [S.m] := by
      intro c hc hm
      have := List.all_eq_true.1 hp c hc
      simpa [hm] using this
    obtain ⟨F, -⟩ := legFacts_of S H 0 dim is cs hli hch hpc hgi hhi D
    intro q hq
    simp only [Node.ports] at hq
    exact ⟨portLegs_isB S H 0 dim D 0 _ cs F q hq, by
      simp only [Node.labels, List.mem_append]
      exact Or.inr (List.mem_of_mem_drop (portLegs_sub _ _ _ _ q hq).1)⟩
  | XB co ci =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨⟨hco, hci⟩, hlo⟩, hli⟩ := hn
    have hpc : ∀ c ∈ co ++ ci, c.head? = some S.m → c = [S.m] := by
      intro c hc hm
      have := List.all_eq_true.1 hp c hc
      simpa [hm] using this
    obtain ⟨Fo, -⟩ := legFacts_of S H 0 dim os co hlo hco
      (fun c hc => hpc c (List.mem_append_left _ hc)) hgo hho D
    obtain ⟨Fi, -⟩ := legFacts_of S H 0 dim is ci hli hci
      (fun c hc => hpc c (List.mem_append_right _ hc)) hgi hhi D
    intro q hq
    simp only [Node.ports, List.mem_append] at hq
    rcases hq with hq | hq
    · exact ⟨portLegs_isB S H 0 dim D 1 _ co Fo q hq, by
        simp only [Node.labels, List.mem_append]
        exact Or.inl (List.mem_of_mem_drop (portLegs_sub _ _ _ _ q hq).1)⟩
    · exact ⟨portLegs_isB S H 0 dim D 1 _ ci Fi q hq, by
        simp only [Node.labels, List.mem_append]
        exact Or.inr (List.mem_of_mem_drop (portLegs_sub _ _ _ _ q hq).1)⟩
  | WB cs =>
    simp only [Kind.outCs, Kind.inCs, S.chainsOK_append, S.chainsOK_replicate, Bool.true_and,
      Bool.and_eq_true] at hn
    obtain ⟨⟨hch, hlo⟩, -⟩ := hn
    have hpc : ∀ c ∈ cs, c.head? = some S.m → c = [S.m] := by
      intro c hc hm
      have := List.all_eq_true.1 hp c hc
      simpa [hm] using this
    obtain ⟨F, -⟩ := legFacts_of S H 0 dim os cs hlo hch.1 hpc hgo hho D
    intro q hq
    simp only [Node.ports] at hq
    exact ⟨portLegs_isB S H 0 dim D 2 _ cs F q hq, by
      simp only [Node.labels, List.mem_append]
      exact Or.inl (List.mem_of_mem_drop (portLegs_sub _ _ _ _ q hq).1)⟩
  | _ => intro q hq; simp [Node.ports] at hq

theorem sumOver_comp (D : ℕ → ℕ) (ls : List ℕ) (f : (ℕ → ℕ) → ℂ) (g : (ℕ → ℕ) → ℕ → ℕ)
    (hg : ∀ τ l j, l ∈ ls → g (Function.update τ l j) = Function.update (g τ) l j)
    (σ : ℕ → ℕ) : sumOver D ls (fun τ => f (g τ)) σ = sumOver D ls f (g σ) := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih =>
    simp only [sumOver]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [ih (fun τ l' j' hl' => hg τ l' j' (List.mem_cons_of_mem _ hl')),
      hg σ l j List.mem_cons_self]

theorem portC_flatMap (E : Env) (H : ℕ) (ns : List Node) (f : Node → List (ℕ × ℕ))
    (σ : ℕ → ℕ) : portC E H (ns.flatMap f) σ = (ns.map fun n => portC E H (f n) σ).prod := by
  induction ns with
  | nil => simp [portC]
  | cons n ns ih => rw [List.flatMap_cons, portC_append, ih, List.map_cons, List.prod_cons]

theorem portC_perm (E : Env) (H : ℕ) {P Q : List (ℕ × ℕ)} (h : P.Perm Q) (σ : ℕ → ℕ) :
    portC E H P σ = portC E H Q σ := (h.map _).prod_eq

theorem portC_reads (E : Env) (H : ℕ) (P : List (ℕ × ℕ)) (σ τ : ℕ → ℕ)
    (h : ∀ q ∈ P, σ q.1 = τ q.1 ∧ σ (q.1 + H) = τ (q.1 + H)) : portC E H P σ = portC E H P τ := by
  unfold portC
  congr 1
  refine List.map_congr_left fun q hq => ?_
  rw [(h q hq).1, (h q hq).2]

/-- The split network's entry: the port factor times the network's at the advanced assignment. -/
theorem eval_port (S : Spec) (H k : ℕ) (hk : 1 ≤ k) (hd : ∀ i, 0 < S.F.dims i) (N : Net)
    (hN : S.splitOK H N = true) (hp : S.portNet N = true) (P : List (ℕ × ℕ))
    (hnd : (P.map Prod.fst).Nodup) (hsub : ∀ q ∈ N.ports S.m, q ∈ P) (σ : ℕ → ℕ)
    (hσ : InRange (S.at k) (S.expand H N.bnd) σ) :
    eval (S.at k) (N.split S H) σ =
      portC (S.at k) H (N.ports S.m) σ * eval (S.at k) N (portG S (S.at k) H P σ) := by
  have F := S.splitOK_facts H N hN
  simp only [Spec.portNet, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hp
  obtain ⟨⟨hpn, -⟩, hint⟩ := hp
  have hint' : ∀ l ∈ N.int, S.isB l = false := fun l hl => by simpa using hint l hl
  have hexp : ∀ ls : List ℕ, (∀ l ∈ ls, S.isB l = false) → S.expand H ls = ls := by
    intro ls h
    induction ls with
    | nil => rfl
    | cons l ls ih =>
      rw [S.expand_plain H l ls (h l List.mem_cons_self),
        ih fun x hx => h x (List.mem_cons_of_mem _ hx)]
  have hlab : ∀ n ∈ N.nodes, ∀ x ∈ n.labels, x ∈ labelsOf N.nodes ++ N.int ++ N.bnd :=
    fun n hn x hx => List.mem_append_left _ (List.mem_append_left _ (List.mem_flatMap.2 ⟨n, hn, hx⟩))
  have hports : ∀ q ∈ N.ports S.m, S.isB q.1 = true ∧ q.1 ∈ labelsOf N.nodes := by
    intro q hq
    obtain ⟨n, hn, hq⟩ := List.mem_flatMap.1 hq
    obtain ⟨a, b⟩ := Node.ports_isB S H F.data n (F.nodes n hn) (hpn n hn)
      (fun x hx => F.good x (hlab n hn x hx)) (fun x hx => F.hat x (hlab n hn x hx)) q hq
    exact ⟨a, List.mem_flatMap.2 ⟨n, hn, b⟩⟩
  have hagree : ∀ τ : ℕ → ℕ, (∀ x ∉ N.int, τ x = σ x) → ∀ q ∈ N.ports S.m,
      τ q.1 = σ q.1 ∧ τ (q.1 + H) = σ (q.1 + H) := by
    intro τ hτ q hq
    obtain ⟨hb, -⟩ := hports q hq
    have hqi : q.1 ∉ N.int := fun h => by rw [hint' _ h] at hb; simp at hb
    have hqH : q.1 + H ∉ N.int := fun h => by
      have := F.lt (q.1 + H) (by simp [h]); omega
    exact ⟨hτ _ hqi, hτ _ hqH⟩
  have hrange : ∀ q ∈ N.ports S.m, σ q.1 < labDim (S.at k) q.1 ∧
      σ (q.1 + H) < labDim (S.at k) (q.1 + H) := by
    intro q hq
    obtain ⟨hb, hm⟩ := hports q hq
    have hqi : q.1 ∉ N.int := fun h => by rw [hint' _ h] at hb; simp at hb
    have hqb : q.1 ∈ N.bnd := (F.closed q.1 hm).resolve_left hqi
    exact ⟨hσ _ ((S.mem_expand H _ N.bnd).2 (Or.inl hqb)),
      hσ _ ((S.mem_expand H _ N.bnd).2 (Or.inr ⟨q.1, hqb, hb, rfl⟩))⟩
  have hnode : ∀ τ : ℕ → ℕ, (∀ x ∉ N.int, τ x = σ x) →
      prodVal (S.at k) (N.nodes.flatMap (Node.split S.m H)) τ =
      portC (S.at k) H (N.ports S.m) τ * prodVal (S.at k) N.nodes (portG S (S.at k) H P τ) := by
    intro τ hτ
    rw [prodVal_flatMap, Net.ports, portC_flatMap]
    unfold prodVal
    rw [← List.prod_map_mul]
    congr 1
    refine List.map_congr_left fun n hn => ?_
    have hq : ∀ q ∈ n.ports S.m, q ∈ N.ports S.m := fun q hq => List.mem_flatMap.2 ⟨n, hn, hq⟩
    exact port_node S H k F.data hk P hnd n (F.nodes n hn) (hpn n hn)
      (fun q h => hsub q (hq q h))
      (fun x hx => F.good x (hlab n hn x hx)) (fun x hx => F.hat x (hlab n hn x hx))
      (hd _) τ fun q h => by
        obtain ⟨e1, e2⟩ := hagree τ hτ q (hq q h)
        rw [e1, e2]; exact hrange q (hq q h)
  simp only [eval, Net.split]
  rw [hexp N.int hint']
  rw [sumOver_congr _ _ _ (fun τ => portC (S.at k) H (N.ports S.m) σ *
    prodVal (S.at k) N.nodes (portG S (S.at k) H P τ)) σ ?_]
  · rw [← mul_sumOver, sumOver_comp _ _ (prodVal (S.at k) N.nodes) (portG S (S.at k) H P) ?_ σ]
    intro τ l j hl
    funext x
    have hlH : l < H := F.lt l (by simp [hl])
    unfold portG
    by_cases hx : x = l
    · subst hx; simp [hint' x hl]
    · have hxH : x + H ≠ l := by omega
      simp only [Function.update_of_ne hx, Function.update_of_ne hxH]
  · intro τ hτ _
    rw [hnode τ hτ, portC_reads _ _ _ τ σ]
    exact hagree τ hτ

/-- Equal networks with the same port legs have equal splits at `m = k ≥ 1`. -/
theorem Equal.of_port (S : Spec) (H k : ℕ) (hk : 1 ≤ k) (hd : ∀ i, 0 < S.F.dims i) (X Y : Net)
    (hX : S.splitOK H X = true) (hY : S.splitOK H Y = true) (hp : S.portOK X Y = true)
    {s s' : ℂ} (h : Equal (S.at k) s X s' Y) :
    Equal (S.at k) s (X.split S H) s' (Y.split S H) := by
  have pX : S.portNet X = true := by
    simp only [Spec.portOK, Bool.and_eq_true] at hp; exact hp.1.1
  have pY : S.portNet Y = true := by
    simp only [Spec.portOK, Bool.and_eq_true] at hp; exact hp.1.2
  have hperm : (X.ports S.m).Perm (Y.ports S.m) := by
    simp only [Spec.portOK, Bool.and_eq_true, decide_eq_true_eq] at hp; exact hp.2
  have hnd : ((X.ports S.m).map Prod.fst).Nodup := by
    have := pX
    simp only [Spec.portNet, Bool.and_eq_true, decide_eq_true_eq] at this; exact this.1.2
  have FX := S.splitOK_facts H X hX
  refine ⟨h.1.flatMap_right _, fun σ hσ => ?_⟩
  simp only [Net.split] at hσ
  set g := portG S (S.at k) H (X.ports S.m) σ
  have hr : InRange (S.at k) X.bnd g := by
    intro x hx
    have h1 := hσ x ((S.mem_expand H x X.bnd).2 (Or.inl hx))
    show portG S (S.at k) H (X.ports S.m) σ x < _
    unfold portG
    split_ifs with hc
    · obtain ⟨hb, -⟩ := hc
      have hg := FX.good x (by simp [hx])
      rw [S.labDim_bundle H k x FX.data hg hb] at h1 ⊢
      rw [S.labDim_hat H k x FX.data hg hb (FX.hat x (by simp [hx]) hb)]
      exact adj_lt _ k _ _ (S.base_pos (fun p _ _ => Fam.dimv_pos _ _ _ (hd _)) x hb) hk h1
    · exact h1
  rw [eval_port S H k hk hd X hX pX (X.ports S.m) hnd (fun q hq => hq) σ hσ,
    eval_port S H k hk hd Y hY pY (X.ports S.m) hnd (fun q hq => hperm.symm.subset hq) σ
      (hσ.perm (h.1.flatMap_right _)),
    portC_perm _ _ hperm σ]
  have := h.2 g hr
  calc s * (portC (S.at k) H (Y.ports S.m) σ * eval (S.at k) X g)
      = portC (S.at k) H (Y.ports S.m) σ * (s * eval (S.at k) X g) := by ring
    _ = portC (S.at k) H (Y.ports S.m) σ * (s' * eval (S.at k) Y g) := by rw [this]
    _ = s' * (portC (S.at k) H (Y.ports S.m) σ * eval (S.at k) Y g) := by ring

/-- `Equal.of_port` with the multiplicities written as a list. -/
theorem Equal.of_port_at (F : Fam) (m H k : ℕ) (l : List ℕ) (hm : m < l.length) (hk : 1 ≤ k)
    (hd : ∀ i, 0 < F.dims i) (N M : Net) (hN : Spec.splitOK ⟨F, m, ksOf l⟩ H N = true)
    (hM : Spec.splitOK ⟨F, m, ksOf l⟩ H M = true) (hp : Spec.portOK ⟨F, m, ksOf l⟩ N M = true)
    {s s' : ℂ} (h : Equal (F.env (ksOf (l.set m k))) s N s' M) :
    Equal (F.env (ksOf (l.set m k))) s (N.split ⟨F, m, ksOf l⟩ H) s'
      (M.split ⟨F, m, ksOf l⟩ H) := by
  rw [← Spec.at_ksOf F m l _ hm] at h ⊢
  exact Equal.of_port _ H k hk hd N M hN hM hp h

/-! ## Structure-only checks -/

/-- A family's bundles and labels alone. -/
def Fam.struct (F : Fam) : Fam :=
  { dims := fun _ => 1, bund := F.bund, phase := fun _ _ => 0, labs := F.labs }

theorem Fam.chainF_struct (F : Fam) : F.struct.chainF = F.chainF := by
  funext n i
  induction n generalizing i with
  | zero => rfl
  | succ n ih => simp only [Fam.chainF, ih]; rfl

theorem Fam.rootF_struct (F : Fam) : F.struct.rootF = F.rootF := by
  funext n i
  induction n generalizing i with
  | zero => rfl
  | succ n ih => simp only [Fam.rootF, ih]; rfl

theorem Spec.nodeOK_struct (F : Fam) (m : ℕ) (ks ks' : ℕ → ℕ) :
    Spec.nodeOK ⟨F.struct, m, ks'⟩ = Spec.nodeOK ⟨F, m, ks⟩ := by
  funext n
  simp only [Spec.nodeOK, Spec.legsOK, Spec.chainsOK, Fam.chain, Fam.root, Fam.chainF_struct,
    Fam.rootF_struct]
  rfl

/-- `splitOK` read off the family's structure. -/
theorem Spec.splitOK_struct (F : Fam) (m : ℕ) (ks ks' : ℕ → ℕ) (H : ℕ) (N : Net)
    (h : Spec.splitOK ⟨F.struct, m, ks'⟩ H N = true) : Spec.splitOK ⟨F, m, ks⟩ H N = true := by
  have e : Spec.splitOK ⟨F.struct, m, ks'⟩ H N = Spec.splitOK ⟨F, m, ks⟩ H N := by
    simp only [Spec.splitOK, Spec.good, Spec.nodeOK_struct F m ks ks', Fam.chain,
      Fam.chainF_struct]
    rfl
  rw [← e]; exact h

/-- `portOK` read off the family's structure. -/
theorem Spec.portOK_struct (F : Fam) (m : ℕ) (ks ks' : ℕ → ℕ) (X Y : Net)
    (h : Spec.portOK ⟨F.struct, m, ks'⟩ X Y = true) : Spec.portOK ⟨F, m, ks⟩ X Y = true := h

/-! ## Scalars -/

/-- The same equation stated with equal scalars. -/
theorem Equal.congr_s {E : Env} {a b a' b' : ℂ} {X Y : Net} (h : Equal E a X b Y) (ha : a = a')
    (hb : b = b') : Equal E a' X b' Y := ha ▸ hb ▸ h

/-- Both scalars times `c`. -/
theorem Equal.smul {E : Env} {a b : ℂ} {X Y : Net} (c : ℂ) (h : Equal E a X b Y) :
    Equal E (c * a) X (c * b) Y :=
  ⟨h.1, fun σ hσ => by rw [mul_assoc, h.2 σ hσ, ← mul_assoc]⟩

end ArchytasZX
