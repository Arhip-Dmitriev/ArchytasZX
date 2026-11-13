/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.WAssoc
import ArchytasZX.FuseB
import ArchytasZX.NodeBox
import ArchytasZX.Weights

/-!
# W fusion over a bundled leg

A W node whose one output `t` bundles `M` legs along the multiplicity `m`, leg `j` the input of
copy `j` of a node `b` (`Bun m b`) whose outputs `ys` bundle along `m` and whose entry at input
`x` is `w x` times the W node from `x` to its outputs: this is the weighted Z spider `w` before
the W node on the outputs `ys`, each bundled along `m` (`wFuseB`). The bundled values are read
as their digits (`sum_digs`: summing a bundled value is `sumW` over its digits) and the copies'
product as `wTerm` (`wTerm_map`), so `wAssoc_nil` applies; its joined groups are the outputs'
digits copy by copy, a permutation of the W node's bundled outputs digit by digit. `wFuseZ`: the
copy a Z or weighted Z spider with one output.
-/

namespace ArchytasZX

open Complex Finset

theorem digs_cons_low (d M x y : ℕ) (hd : 0 < d) (hx : x < d) :
    digs d (M + 1) (x + d * y) = x :: digs d M y := by
  unfold digs
  rw [List.range_succ_eq_map, List.map_cons, List.map_map]
  congr 1
  · simp [digit, Nat.mod_eq_of_lt hx]
  · refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp, digit, pow_succ']
    rw [← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ hd, Nat.div_eq_of_lt hx, zero_add]

/-- Summing a value below `d ^ M` read as its `M` digits is `sumW`. -/
theorem sum_digs (d : ℕ) (hd : 0 < d) : ∀ (M : ℕ) (f : List ℕ → ℂ),
    ∑ v ∈ range (d ^ M), f (digs d M v) = sumW d M f
  | 0, f => by simp [digs, sumW]
  | M + 1, f => by
    rw [pow_succ', sum_range_mul]
    simp only [sumW]
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [← sum_digs d hd M]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [digs_cons_low d M x y hd (Finset.mem_range.1 hx)]

theorem wTerm_map (d : ℕ) (w : ℕ → ℂ) (f : ℕ → ℕ) (g : ℕ → List ℕ) : ∀ l : List ℕ,
    wTerm d w (l.map f) (l.map g) = (l.map fun i => w (f i) * tW d (g i) (f i)).prod
  | [] => rfl
  | i :: l => by
    simp only [List.map_cons, wTerm, List.prod_cons, wTerm_map d w f g l]

theorem tailVals_reps (mult : ℕ → ℕ) (d m : ℕ) : ∀ vs : List ℕ,
    tailVals mult d vs (vs.map fun _ => [m]) = vs.flatMap (digs d (mult m))
  | [] => rfl
  | v :: vs => by
    have ih := tailVals_reps mult d m vs
    simp only [tailVals, List.map_cons, List.zipWith_cons_cons, List.flatten_cons,
      List.flatMap_cons] at ih ⊢
    rw [ih]
    congr 1
    simp [digsC]

theorem bundled_reps (mult : ℕ → ℕ) (d m : ℕ) (vs : List ℕ) :
    bundled mult d vs (vs.map fun _ => [m]) = vs.flatMap (digs d (mult m)) := by
  simp only [bundled, List.length_map, Nat.sub_self, List.take_zero, List.drop_zero,
    List.nil_append, tailVals_reps]

theorem copyLegs_plain (mult : ℕ → ℕ) (d i : ℕ) : ∀ vs : List ℕ,
    copyLegs mult d i (List.replicate vs.length []) vs = vs.map (digit d i)
  | [] => rfl
  | v :: vs => by
    have ih := copyLegs_plain mult d i vs
    simp only [copyLegs, List.length_cons, List.replicate_succ, List.zipWith_cons_cons,
      List.map_cons, List.map_nil, List.prod_nil, pow_one] at ih ⊢
    rw [ih]

theorem tZw_pair (d : ℕ) (c : ℕ → ℂ) (v x : ℕ) (hx : x < d) :
    tZw d c [v, x] = if v = x then c x else 0 := by
  unfold tZw
  by_cases h : v = x
  · subst h
    simp [hx]
  · simp only [h, ite_false]
    refine Finset.sum_eq_zero fun k _ => ?_
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    refine ite_eq_right_iff.2 fun h' => absurd (h'.1.trans h'.2.symm) h

theorem tW_one (d v x : ℕ) (hx : x < d) : tW d [v] x = if v = x then 1 else 0 := by
  rw [tW_eq]
  by_cases h0 : x = 0
  · subst h0
    by_cases hv : v = 0 <;> simp [nz, hv]
  · by_cases hv : v = 0
    · subst hv
      simp [nz, h0, Ne.symm h0]
    · simp [nz, h0, hv, hx]

/-- W fusion over a bundled leg: `W(a → t)` with `t` fanned into `M` copies of `b` (each
`w x` times `W(x → ys_j)`) is `Zw(a → a')` then `W(a' → ys)`, every `ys` bundled along `m`. -/
theorem wFuseB (E : Env) (b : Kind) (hb : b = .Z ∨ b = .ZW ∨ b = .W) (w : ℕ → ℂ)
    (dz m bz pw pb pz pw' t a a' : ℕ) (ys : List ℕ)
    (hd : 0 < E.dimv dz) (hbz : E.dimv bz = E.dimv dz ^ ([m].map E.mult).prod)
    (ht : labIdx E.labs t = bz)
    (ha : labIdx E.labs a = dz) (ha' : labIdx E.labs a' = dz)
    (hta : t ≠ a) (hty : t ∉ ys) (haa : a' ≠ a) (hay : a' ∉ ys)
    (hcopy : ∀ x < E.dimv dz, ∀ vs : List ℕ, vs.length = ys.length →
      tensor E b (E.dimv dz) pb vs [x] = w x * tW (E.dimv dz) vs x)
    (hw0 : w 0 = 1) (hz : ∀ k < E.dimv dz, E.coef pz k = w k) :
    LocalEq E [⟨.WB [[m]], [t], [a], dz, pw⟩, ⟨.Bun m b, ys, [t], dz, pb⟩] [t]
      [⟨.ZW, [a'], [a], dz, pz⟩, ⟨.WB (ys.map fun _ => [m]), ys, [a'], dz, pw'⟩] [a']
      (a :: ys) 1 1 := by
  intro τ hτ
  set d := E.dimv dz with hdd
  set M := E.mult m with hM
  have hbz' : E.dimv bz = d ^ M := by simpa using hbz
  have hia : τ a < d := by
    have := hτ a List.mem_cons_self
    rwa [labDim_w E a dz ha] at this
  have hbo : ∀ n, b.outCs n = List.replicate n [] := by
    rcases hb with rfl | rfl | rfl <;> intro n <;> rfl
  have hbi : ∀ n, b.inCs n = List.replicate n [] := by
    rcases hb with rfl | rfl | rfl <;> intro n <;> rfl
  set Bs : List (List ℕ) := (List.range M).map fun i => (ys.map τ).map (digit d i) with hBs
  have hlen : Bs.length = M := by simp [hBs]
  have hL : sumOver (labDim E) [t]
      (prodVal E [⟨.WB [[m]], [t], [a], dz, pw⟩, ⟨.Bun m b, ys, [t], dz, pb⟩]) τ =
      ∑ v ∈ range (d ^ M), tW d (digs d M v) (τ a) * wTerm d w (digs d M v) Bs := by
    simp only [sumOver, labDim_w E t bz ht, hbz']
    refine Finset.sum_congr rfl fun v _ => ?_
    have hys : ys.map (Function.update τ t v) = ys.map τ := map_update_absent τ t v ys hty
    simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
    rw [show val E (Function.update τ t v) ⟨.Bun m b, ys, [t], dz, pb⟩ =
        tensor E (.Bun m b) d pb (ys.map τ) [v] by
      simp only [val, List.map_cons, List.map_nil, Function.update_self, hys]
      rfl]
    rw [show val E (Function.update τ t v) ⟨.WB [[m]], [t], [a], dz, pw⟩ =
        tW d (digs d M v) (τ a) by
      simp only [val, List.map_cons, List.map_nil, Function.update_self,
        Function.update_of_ne hta.symm]
      congr 1
      simp [bundled, tailVals, digsC, hM, hdd]]
    congr 1
    simp only [tensor]
    rw [show digs d M v = (List.range M).map fun i => digit d i v from rfl, hBs, wTerm_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    have hov : copyOuts E b d i (ys.map τ) = (ys.map τ).map (digit d i) := by
      unfold copyOuts
      rw [hbo, List.length_map]
      simpa using copyLegs_plain E.mult d i (ys.map τ)
    have hiv : copyIns E b d i [v] = [digit d i v] := by
      unfold copyIns
      rw [hbi]
      simpa using copyLegs_plain E.mult d i [v]
    rw [hov, hiv, hcopy _ (digit_lt d i v hd) _ (by simp)]
  have hR : sumOver (labDim E) [a']
      (prodVal E [⟨.ZW, [a'], [a], dz, pz⟩, ⟨.WB (ys.map fun _ => [m]), ys, [a'], dz, pw'⟩]) τ =
      w (τ a) * tW d ((ys.map τ).flatMap (digs d M)) (τ a) := by
    simp only [sumOver, labDim_w E a' dz ha']
    have hys : ∀ v, ys.map (Function.update τ a' v) = ys.map τ :=
      fun v => map_update_absent τ a' v ys hay
    have e : ∀ v ∈ range d, prodVal E [⟨.ZW, [a'], [a], dz, pz⟩,
        ⟨.WB (ys.map fun _ => [m]), ys, [a'], dz, pw'⟩] (Function.update τ a' v) =
        tZw d (E.coef pz) [v, τ a] * tW d ((ys.map τ).flatMap (digs d M)) v := by
      intro v _
      simp only [prodVal, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one]
      congr 1
      · simp only [val, List.map_cons, List.map_nil, Function.update_self,
          Function.update_of_ne haa.symm]
        rfl
      · simp only [val, List.map_cons, List.map_nil, Function.update_self, hys]
        rw [show (ys.map fun _ => [m]) = (ys.map τ).map (fun _ => [m]) by
          rw [List.map_map]; rfl, bundled_reps]
    rw [Finset.sum_congr rfl e]
    simp only [tZw_pair d _ _ _ hia, ite_mul, zero_mul]
    rw [Finset.sum_ite_eq' (range d) (τ a)]
    simp only [Finset.mem_range.2 hia, ↓reduceIte, hz _ hia]
  simp only [one_mul]
  rw [hL, hR, sum_digs d hd M (fun xs => tW d xs (τ a) * wTerm d w xs Bs), ← hlen,
    wAssoc_nil d hd w hw0 (τ a) hia Bs]
  congr 1
  apply tW_perm
  have e1 : Bs.flatten = (List.range M).flatMap (fun i => (ys.map τ).map (digit d i)) := by
    rw [hBs]
    rfl
  have e2 : (ys.map τ).flatMap (digs d M) =
      (ys.map τ).flatMap (fun v => (List.range M).map fun i => digit d i v) := rfl
  rw [hlen, e1, e2]
  exact Diagram.flatMap_swap_perm (List.range M) (ys.map τ) (fun i v => digit d i v)

/-- `wFuseB` for a copy that is a Z or weighted Z spider with one output, its coefficients
those of the weighted Z spider. -/
theorem wFuseZ (E : Env) (b : Kind) (hb : b = .Z ∨ b = .ZW)
    (dz m bz pw pb pz pw' t a a' y : ℕ)
    (hd : 0 < E.dimv dz) (hbz : E.dimv bz = E.dimv dz ^ ([m].map E.mult).prod)
    (ht : labIdx E.labs t = bz)
    (ha : labIdx E.labs a = dz) (ha' : labIdx E.labs a' = dz)
    (hta : t ≠ a) (hty : t ≠ y) (haa : a' ≠ a) (hay : a' ≠ y)
    (hc : ∀ k < E.dimv dz, zc E ⟨b, [y], [t], dz, pb⟩ k = zc E ⟨.ZW, [a'], [a], dz, pz⟩ k)
    (hw0 : zc E ⟨.ZW, [a'], [a], dz, pz⟩ 0 = 1) :
    LocalEq E [⟨.WB [[m]], [t], [a], dz, pw⟩, ⟨.Bun m b, [y], [t], dz, pb⟩] [t]
      [⟨.ZW, [a'], [a], dz, pz⟩, ⟨.WB [[m]], [y], [a'], dz, pw'⟩] [a'] [a, y] 1 1 := by
  have h := wFuseB E b (hb.imp_right Or.inl) (fun k => E.coef pz k) dz m bz pw pb pz pw' t a a'
    [y] hd hbz ht ha ha' hta (by simpa using hty) haa (by simpa using hay)
    ?_ (by simpa [zc] using hw0) (fun _ _ => rfl)
  · simpa using h
  intro x hx vs hvs
  obtain ⟨v, rfl⟩ : ∃ v, vs = [v] := by
    match vs, hvs with
    | [v], _ => exact ⟨v, rfl⟩
  have hcx := hc x hx
  simp only [zc] at hcx
  rw [tW_one _ _ _ hx]
  rcases hb with rfl | rfl
  · rw [show tensor E .Z (E.dimv dz) pb [v] [x] = tZw (E.dimv dz) (fun k => ex (E.phase pb k))
        [v, x] from rfl, tZw_pair _ _ _ _ hx]
    split_ifs <;> simp [← hcx]
  · rw [show tensor E .ZW (E.dimv dz) pb [v] [x] = tZw (E.dimv dz) (E.coef pb) [v, x] from rfl,
      tZw_pair _ _ _ _ hx]
    split_ifs <;> simp [← hcx]

end ArchytasZX
