/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymHint
import ArchytasZX.PathSumSem

/-!
# Sum-free residuals through their characters

A term with no sum, constant moduli dividing `N` and a constant `q ex r` (`r N` an integer)
expands into characters `w ω_N^P`: each root of unity `ω_M^p` as `ω_N^((N/M) p)`, each delta
`[M ∣ p]` as `(1/M) Σ_{k<M} ω_N^(k (N/M) p)` (`expTerm`). Each character linear over the
certificate's variables gets a slot, its coefficients mod `N` read as digits in base `N`
(`slotOf`, `slot_eq`). `fzOK` sorts the characters and the certificate's cycles
`-λ Σ_{i<p} ω_N^(K + a + i N/p)` (`1 < p ∣ N`, each worth `0` by `om_geom`) by slot and checks
that every run of one slot sums to zero (`runsGo`); `fz_zero` gives `exDen e = 0` at every
valuation.
`symCheckF` and `equal_symF` are `symCheck` and `equal_sym` with this check beside the empty
normal form.
-/

namespace ArchytasZX.Sym

open Complex Finset

theorem om_zpow_congr (N : ℕ) (hN : 0 < N) (a b : ℤ) (h : (N : ℤ) ∣ a - b) :
    om N ^ a = om N ^ b := by
  obtain ⟨k, hk⟩ := h
  have hz := om_ne_zero N hN
  rw [show a = b + N * k by omega, zpow_add₀ hz, zpow_mul, zpow_natCast, om_pow_self N hN,
    one_zpow, mul_one]

/-- `ω_M^z = ω_N^((N/M) z)` for `M ∣ N`. -/
theorem om_up (N M : ℕ) (hN0 : 0 < N) (hMN : N % M = 0) (z : ℤ) :
    om M ^ z = om N ^ (((N / M : ℕ) : ℤ) * z) := by
  have hN : N = M * (N / M) := (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hMN)).symm
  rw [om_sub N M (N / M) hN0 hN, ← zpow_natCast, ← zpow_mul]

theorem lsum_range (g : ℕ → ℂ) (n : ℕ) : ((List.range n).map g).sum = ∑ i ∈ range n, g i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

/-! ## Expansions -/

/-- The value of characters `Σ w ω_N^P`. -/
noncomputable def expV (N : ℕ) (ν : ℕ → ℤ) (L : List (ℚ × IPoly)) : ℂ :=
  (L.map fun s => (s.1 : ℂ) * om N ^ ipDen ν s.2).sum

theorem expV_append (N : ℕ) (ν : ℕ → ℤ) (a b : List (ℚ × IPoly)) :
    expV N ν (a ++ b) = expV N ν a + expV N ν b := by
  simp [expV, List.map_append, List.sum_append]

/-- Whether a factor's modulus is a positive constant dividing `N`. -/
def facOK (N : ℕ) (f : Fac) : Bool := f.1.isConst && decide (0 < f.1.c) && N % f.1.c == 0

/-- Characters times `ω_M^p`. -/
def mulPh (N : ℕ) (f : Fac) (L : List (ℚ × IPoly)) : List (ℚ × IPoly) :=
  L.map fun s => (s.1, ipAdd s.2 (ipSmul ((N / f.1.c : ℕ) : ℤ) f.2))

/-- Characters times `[M ∣ p] = (1/M) Σ_{k<M} ω_M^(k p)`. -/
def mulDel (N : ℕ) (f : Fac) (L : List (ℚ × IPoly)) : List (ℚ × IPoly) :=
  L.flatMap fun s => (List.range f.1.c).map fun k =>
    (s.1 / f.1.c, ipAdd s.2 (ipSmul ((k * (N / f.1.c) : ℕ) : ℤ) f.2))

theorem expV_mulPh (ρ : Val) (N : ℕ) (hN : 0 < N) (f : Fac) (hf : facOK N f = true)
    (ν : ℕ → ℤ) (L : List (ℚ × IPoly)) :
    expV N ν (mulPh N f L) = phV ρ.dv ν f * expV N ν L := by
  simp only [facOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hf
  obtain ⟨⟨hc, _⟩, hdiv⟩ := hf
  unfold phV
  rw [den_const_c ρ.dv f.1 hc, om_up N f.1.c hN hdiv]
  induction L with
  | nil => simp [mulPh, expV]
  | cons s L ih =>
    simp only [mulPh, expV, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih, ipDen_add, ipDen_smul, zpow_add₀ (om_ne_zero N hN)]
    ring

theorem delV_avg (ρ : Val) (N : ℕ) (hN : 0 < N) (f : Fac) (hf : facOK N f = true) (ν : ℕ → ℤ) :
    delV ρ.dv ν f = ((1 : ℂ) / f.1.c) *
      ∑ k ∈ range f.1.c, om N ^ ((((k * (N / f.1.c) : ℕ) : ℤ)) * ipDen ν f.2) := by
  simp only [facOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hf
  obtain ⟨⟨hc, hpos⟩, hdiv⟩ := hf
  have hs := sum_om_lin f.1.c hpos (ipDen ν f.2) 0
  have he : ∀ k : ℕ, om N ^ ((((k * (N / f.1.c) : ℕ) : ℤ)) * ipDen ν f.2) =
      om f.1.c ^ ((k : ℤ) * ipDen ν f.2 + 0) := by
    intro k
    rw [add_zero, om_up N f.1.c hN hdiv]
    congr 1
    push_cast
    ring
  simp only [he, hs, zpow_zero, one_mul]
  unfold delV
  rw [den_const_c ρ.dv f.1 hc]
  have hc0 : (f.1.c : ℂ) ≠ 0 := by exact_mod_cast hpos.ne'
  split_ifs <;> field_simp <;> simp

theorem expV_mulDel (ρ : Val) (N : ℕ) (hN : 0 < N) (f : Fac) (hf : facOK N f = true)
    (ν : ℕ → ℤ) (L : List (ℚ × IPoly)) :
    expV N ν (mulDel N f L) = delV ρ.dv ν f * expV N ν L := by
  have hav := delV_avg ρ N hN f hf ν
  rw [hav]
  induction L with
  | nil => simp [mulDel, expV]
  | cons s L ih =>
    simp only [mulDel, List.flatMap_cons, expV_append] at ih ⊢
    rw [ih]
    have hk : expV N ν ((List.range f.1.c).map fun k =>
        (s.1 / f.1.c, ipAdd s.2 (ipSmul ((k * (N / f.1.c) : ℕ) : ℤ) f.2))) =
        ((s.1 : ℂ) / f.1.c) * ∑ k ∈ range f.1.c,
          om N ^ ipDen ν s.2 * om N ^ ((((k * (N / f.1.c) : ℕ) : ℤ)) * ipDen ν f.2) := by
      unfold expV
      rw [List.map_map, Finset.mul_sum, lsum_range]
      refine Finset.sum_congr rfl fun k _ => ?_
      simp only [Function.comp_apply, ipDen_add, ipDen_smul]
      rw [zpow_add₀ (om_ne_zero N hN)]; push_cast; ring
    rw [hk]
    simp only [expV, List.map_cons, List.sum_cons, ← Finset.sum_mul, ← Finset.mul_sum]
    push_cast
    ring

/-- The constant `q ex r` as one character, when `r N` is an integer and nothing else. -/
def cExp (N : ℕ) (c : CMon) : Option (List (ℚ × IPoly)) :=
  if c.sq = [] ∧ c.rc = [] ∧ c.am = [] ∧ c.gc = [] ∧ (c.r * N).den = 1 then
    some [(c.q, ipC (c.r * N).num)]
  else none

theorem expV_cExp (ρ : Val) (N : ℕ) (hN : 0 < N) (c : CMon) (L : List (ℚ × IPoly))
    (h : cExp N c = some L) (ν : ℕ → ℤ) : expV N ν L = c.den ρ := by
  unfold cExp at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  obtain ⟨hsq, hrc, ham, hgc, hden⟩ := hc
  obtain ⟨q, r, sq, rc, am, gc⟩ := c
  simp only at hsq hrc ham hgc hden ⊢
  subst hsq hrc ham hgc
  simp only [expV, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, ipDen_C,
    CMon.den, zpw, List.map_nil, List.prod_nil, mul_one]
  rw [om_zpow_ex]
  congr 2
  have hnum : ((r * N).num : ℚ) = r * N := by
    have := Rat.num_div_den (r * N); rw [hden] at this; simpa using this
  have hN' : (N : ℚ) ≠ 0 := by exact_mod_cast hN.ne'
  rw [hnum]; field_simp

/-- A sum-free term as characters. -/
def expTerm (N : ℕ) (t : STm) : Option (List (ℚ × IPoly)) :=
  if t.sums = [] ∧ (t.dels ++ t.phs).all (facOK N) = true then
    (cExp N t.c).map fun L => t.phs.foldr (mulPh N) (t.dels.foldr (mulDel N) L)
  else none

theorem expTerm_ok (ρ : Val) (N : ℕ) (hN : 0 < N) (t : STm) (L : List (ℚ × IPoly))
    (h : expTerm N t = some L) (ν : ℕ → ℤ) : expV N ν L = t.den ρ ν := by
  unfold expTerm at h
  split_ifs at h with hc
  obtain ⟨hs, hok⟩ := hc
  simp only [List.all_append, Bool.and_eq_true, List.all_eq_true] at hok
  obtain ⟨hd, hp⟩ := hok
  cases hce : cExp N t.c with
  | none => rw [hce] at h; simp at h
  | some L0 =>
    rw [hce] at h
    simp only [Option.map_some, Option.some.injEq] at h
    subst h
    obtain ⟨c, sums, dels, phs⟩ := t
    simp only at hs hd hp hce ⊢
    subst hs
    rw [den_zero]
    have h0 := expV_cExp ρ N hN c L0 hce ν
    have hds : ∀ L, expV N ν (dels.foldr (mulDel N) L) =
        (dels.map (delV ρ.dv ν)).prod * expV N ν L := by
      intro L
      induction dels with
      | nil => simp
      | cons f ds ih =>
        simp only [List.foldr_cons, List.map_cons, List.prod_cons]
        rw [expV_mulDel ρ N hN f (hd f List.mem_cons_self) ν,
          ih (fun g hg => hd g (List.mem_cons_of_mem _ hg))]
        ring
    have hps : ∀ L, expV N ν (phs.foldr (mulPh N) L) =
        (phs.map (phV ρ.dv ν)).prod * expV N ν L := by
      intro L
      induction phs with
      | nil => simp
      | cons f ps ih =>
        simp only [List.foldr_cons, List.map_cons, List.prod_cons]
        rw [expV_mulPh ρ N hN f (hp f List.mem_cons_self) ν,
          ih (fun g hg => hp g (List.mem_cons_of_mem _ hg))]
        ring
    rw [hps, hds, h0, bodyOf]
    ring

/-- A sum-free expression as characters. -/
def expEx (N : ℕ) : List STm → Option (List (ℚ × IPoly))
  | [] => some []
  | t :: e =>
    match expTerm N t, expEx N e with
    | some a, some b => some (a ++ b)
    | _, _ => none

theorem expEx_ok (ρ : Val) (N : ℕ) (hN : 0 < N) :
    ∀ (e : List STm) (L : List (ℚ × IPoly)), expEx N e = some L → ∀ ν,
      expV N ν L = exDen ρ ν e
  | [], L, h, ν => by simp only [expEx, Option.some.injEq] at h; subst h; rfl
  | t :: e, L, h, ν => by
    simp only [expEx] at h
    split at h
    · next a b ha hb =>
      simp only [Option.some.injEq] at h
      subst h
      rw [expV_append, expTerm_ok ρ N hN t a ha ν, expEx_ok ρ N hN e b hb ν, exDen_cons]
    · simp at h

/-! ## Slots: characters packed into one number -/


/-- Whether a monomial is `[]` or `[x]` with `x` in `vs`. -/
def monOK (vs : List ℕ) : IMon → Bool
  | [] => true
  | [x] => vs.contains x
  | _ => false

/-- Whether every monomial is `[]` or `[x]` with `x` in `vs`. -/
def linOK (vs : List ℕ) (P : IPoly) : Bool := P.all fun t => monOK vs t.1

/-- The digit position of a monomial: `0` for `[]`, `1 +` the index of `x` for `[x]`. -/
def posOf (vs : List ℕ) (m : IMon) : ℕ :=
  match m with
  | [x] => vs.idxOf x + 1
  | _ => 0

/-- The coefficient at position `i`. -/
def cAt (vs : List ℕ) (P : IPoly) (i : ℕ) : ℤ :=
  ((P.filter fun t => posOf vs t.1 == i).map (·.2)).sum

/-- The digits mod `N`, position `0` first. -/
def digitsOf (vs : List ℕ) (N : ℕ) (P : IPoly) : List ℕ :=
  (List.range (vs.length + 1)).map fun i => (cAt vs P i % (N : ℤ)).toNat

/-- The slot: the digits read in base `N`. -/
def slotOf (vs : List ℕ) (N : ℕ) (P : IPoly) : ℕ := Nat.ofDigits N (digitsOf vs N P)

/-- The value at position `i`: `1`, then `ν` of each variable. -/
def valAt (vs : List ℕ) (ν : ℕ → ℤ) (i : ℕ) : ℤ := if i = 0 then 1 else ν (vs.getD (i - 1) 0)

theorem pos_lt (vs : List ℕ) (ν : ℕ → ℤ) (m : IMon) (h : monOK vs m = true) :
    posOf vs m < vs.length + 1 ∧ imDen ν m = valAt vs ν (posOf vs m) := by
  match m, h with
  | [], _ => exact ⟨by simp [posOf], by simp [posOf, valAt, imDen]⟩
  | _ :: _ :: _, h => simp [monOK] at h
  | [x], h =>
    simp only [monOK, List.contains_iff_mem] at h
    refine ⟨by simp only [posOf]; have := List.idxOf_lt_length_iff.2 h; omega, ?_⟩
    simp only [posOf, valAt, imDen, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      mul_one, Nat.add_sub_cancel, Nat.add_one_ne_zero, ite_false]
    rw [List.getD_eq_getElem _ _ (List.idxOf_lt_length_iff.2 h), List.getElem_idxOf]

theorem ipDen_pos (vs : List ℕ) (ν : ℕ → ℤ) : ∀ P : IPoly, linOK vs P = true →
    ipDen ν P = ∑ i ∈ range (vs.length + 1), cAt vs P i * valAt vs ν i
  | [], _ => by simp [ipDen, cAt]
  | t :: P, h => by
    simp only [linOK, List.all_cons, Bool.and_eq_true] at h
    have ih := ipDen_pos vs ν P h.2
    obtain ⟨hlt, hv⟩ := pos_lt vs ν t.1 h.1
    have e : ∀ i, cAt vs (t :: P) i * valAt vs ν i =
        cAt vs P i * valAt vs ν i + (if posOf vs t.1 = i then t.2 * valAt vs ν i else 0) := by
      intro i
      unfold cAt
      by_cases hi : posOf vs t.1 = i
      · simp [List.filter_cons, hi]; ring
      · simp [List.filter_cons, hi]
    rw [Finset.sum_congr rfl fun i _ => e i, Finset.sum_add_distrib, ← ih, Finset.sum_ite_eq,
      if_pos (Finset.mem_range.2 hlt), ipDen_cons, hv]
    ring

theorem slot_eq (vs : List ℕ) (N : ℕ) (hN : 1 < N) (P P' : IPoly) (h : linOK vs P = true)
    (h' : linOK vs P' = true) (hs : slotOf vs N P = slotOf vs N P') (ν : ℕ → ℤ) :
    (N : ℤ) ∣ ipDen ν P - ipDen ν P' := by
  have hN0 : (0 : ℤ) < N := by exact_mod_cast (by omega : 0 < N)
  have hlt : ∀ Q : IPoly, ∀ l ∈ digitsOf vs N Q, l < N := by
    intro Q l hl
    simp only [digitsOf, List.mem_map, List.mem_range] at hl
    obtain ⟨i, _, rfl⟩ := hl
    have h1 := Int.emod_lt_of_pos (cAt vs Q i) hN0
    have h2 := Int.emod_nonneg (cAt vs Q i) hN0.ne'
    have : ((cAt vs Q i % N).toNat : ℤ) = cAt vs Q i % N := Int.toNat_of_nonneg h2
    omega
  have hd := Nat.ofDigits_inj_of_len_eq hN (by simp [digitsOf]) (hlt P) (hlt P') hs
  rw [ipDen_pos vs ν P h, ipDen_pos vs ν P' h', ← Finset.sum_sub_distrib]
  refine Finset.dvd_sum fun i hi => ?_
  have hi' : i < vs.length + 1 := Finset.mem_range.1 hi
  have hdi : (cAt vs P i % N).toNat = (cAt vs P' i % N).toNat := by
    have := congrArg (fun l => l.getD i 0) hd
    simpa [digitsOf, hi'] using this
  have h1 := Int.emod_nonneg (cAt vs P i) hN0.ne'
  have h2 := Int.emod_nonneg (cAt vs P' i) hN0.ne'
  have heq : cAt vs P i % N = cAt vs P' i % N := by
    rw [← Int.toNat_of_nonneg h1, ← Int.toNat_of_nonneg h2, hdi]
  rw [← sub_mul]
  exact Dvd.dvd.mul_right (Int.ModEq.dvd heq.symm) _

/-! ## Runs of one slot -/

/-- Each run of equal slots in a list sums to zero, `r` the current run's slot and `q` its
running sum. -/
def runsGo {α : Type} : Option ℕ → ℚ → List (ℕ × α × ℚ) → Bool
  | _, q, [] => q == 0
  | none, _, a :: l => runsGo (some a.1) a.2.2 l
  | some r, q, a :: l => if a.1 = r then runsGo (some r) (q + a.2.2) l
      else q == 0 && runsGo (some a.1) a.2.2 l

theorem runsGo_ok {α : Type} (g : ℕ → ℂ) :
    ∀ (r : ℕ) (q : ℚ) (l : List (ℕ × α × ℚ)), runsGo (some r) q l = true →
      (q : ℂ) * g r + (l.map fun a => (a.2.2 : ℂ) * g a.1).sum = 0
  | r, q, [], h => by simp only [runsGo, beq_iff_eq] at h; simp [h]
  | r, q, a :: l, h => by
    simp only [runsGo] at h
    split_ifs at h with he
    · have ih := runsGo_ok g r (q + a.2.2) l h
      simp only [List.map_cons, List.sum_cons, he]
      push_cast at ih
      linear_combination ih
    · simp only [Bool.and_eq_true, beq_iff_eq] at h
      have ih := runsGo_ok g a.1 a.2.2 l h.2
      simp only [List.map_cons, List.sum_cons, h.1]
      push_cast
      linear_combination ih

theorem runs_ok {α : Type} (g : ℕ → ℂ) (l : List (ℕ × α × ℚ)) (h : runsGo none 0 l = true) :
    (l.map fun a => (a.2.2 : ℂ) * g a.1).sum = 0 := by
  match l, h with
  | [], _ => simp
  | a :: l, h =>
    simp only [runsGo] at h
    have := runsGo_ok g a.1 a.2.2 l h
    simpa using this

/-! ## Residues of pinned labels -/

/-- `t` with `x` set to `v`. -/
def subTm (x v : ℕ) (t : STm) : STm :=
  ⟨t.c, t.sums, t.dels.map (facSub x (ipC v)), t.phs.map (facSub x (ipC v))⟩

/-- No sum, and every modulus a divisor of `D`. -/
def modOK (D : ℕ) (t : STm) : Bool :=
  t.sums.isEmpty && (t.dels ++ t.phs).all fun f => f.1.dvd ⟨D, []⟩

theorem modOK_sub (D x v : ℕ) (t : STm) (h : modOK D t = true) : modOK D (subTm x v t) = true := by
  simpa [modOK, subTm, facSub] using h

theorem den_const_D (dv : ℕ → ℕ) (D : ℕ) : DMon.den dv ⟨D, []⟩ = D := by
  show D * npow dv 0 [] = D
  rw [show npow dv 0 [] = 1 from rfl, mul_one]

theorem lsum_flatMap {α : Type} (l : List α) (f : α → List ℂ) :
    (l.flatMap f).sum = (l.map fun a => (f a).sum).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.flatMap_cons, List.sum_append, ih]

theorem subTm_den (ρ : Val) (hρ : ρ.OK) (x D : ℕ) (t : STm) (h : modOK D t = true)
    (ν : ℕ → ℤ) (v : ℕ) (hv : (D : ℤ) ∣ ν x - v) : (subTm x v t).den ρ ν = t.den ρ ν := by
  obtain ⟨c, sums, ds, ps⟩ := t
  simp only [modOK, Bool.and_eq_true, List.isEmpty_iff, List.all_eq_true] at h
  obtain ⟨hs, hf⟩ := h
  subst hs
  simp only [subTm, den_zero]
  rw [bodyOf_sub, ipDen_C]
  congr 1
  have hupd : Function.update ν x (ν x) = ν := Function.update_eq_self x ν
  have key : ∀ f ∈ ds ++ ps, delV ρ.dv (Function.update ν x (v : ℤ)) f = delV ρ.dv ν f ∧
      phV ρ.dv (Function.update ν x (v : ℤ)) f = phV ρ.dv ν f := by
    intro f hfm
    have := fac_congr ρ.dv hρ.dpos x ⟨D, []⟩ f (hf f hfm) ν (v : ℤ) (ν x)
      (by rw [den_const_D]; exact (dvd_neg.2 hv).trans (by rw [neg_sub]))
    rwa [hupd] at this
  unfold bodyOf
  congr 1
  · congr 1; exact List.map_congr_left fun f hf' => (key f (List.mem_append_left _ hf')).1
  · congr 1; exact List.map_congr_left fun f hf' => (key f (List.mem_append_right _ hf')).2

theorem sum_ind_one (D : ℕ) (hD : 0 < D) (a : ℤ) :
    ∑ v ∈ range D, (if (D : ℤ) ∣ a - v then (1 : ℂ) else 0) = 1 := by
  have e : ∀ v ∈ range D, (if (D : ℤ) ∣ a - v then (1 : ℂ) else 0) =
      if v = (a % D).toNat then 1 else 0 := by
    intro v hv
    have := dvd_sub_iff_eq D hD v (Finset.mem_range.1 hv) a
    have h2 : (D : ℤ) ∣ a - v ↔ (D : ℤ) ∣ (v : ℤ) - a := by
      rw [← dvd_neg, neg_sub]
    simp only [h2, this]
  rw [Finset.sum_congr rfl e, Finset.sum_ite_eq']
  rw [if_pos (Finset.mem_range.2 ?_)]
  have h0 := Int.emod_nonneg a (by exact_mod_cast hD.ne' : (D : ℤ) ≠ 0)
  have h1 := Int.emod_lt_of_pos a (by exact_mod_cast hD : (0 : ℤ) < D)
  omega

/-- The indicator `[D ∣ x - v]`. -/
noncomputable def ind (D : ℕ) (ν : ℕ → ℤ) (x v : ℕ) : ℂ := if (D : ℤ) ∣ ν x - v then 1 else 0

theorem den_split (ρ : Val) (hρ : ρ.OK) (x D : ℕ) (hD : 0 < D) (t : STm) (h : modOK D t = true)
    (ν : ℕ → ℤ) : t.den ρ ν = ∑ v ∈ range D, ind D ν x v * (subTm x v t).den ρ ν := by
  have e : ∀ v ∈ range D, ind D ν x v * (subTm x v t).den ρ ν = ind D ν x v * t.den ρ ν := by
    intro v _
    unfold ind
    split_ifs with hv
    · rw [subTm_den ρ hρ x D t h ν v hv]
    · simp
  rw [Finset.sum_congr rfl e, ← Finset.sum_mul]
  unfold ind
  rw [sum_ind_one D hD, one_mul]

/-- `some b` for a delta with a constant argument over a positive constant modulus, `b` whether
it holds. -/
def cdel (f : Fac) : Option Bool :=
  match ipConst f.2 with
  | some c => if f.1.isConst && decide (0 < f.1.c) then some (c % (f.1.c : ℤ) == 0) else none
  | none => none

/-- Constant deltas evaluated: none when one fails, else the term without them. -/
def pruneTm (t : STm) : Option STm :=
  if t.dels.any (fun f => cdel f == some false) then none
  else some ⟨t.c, t.sums, t.dels.filter (fun f => cdel f != some true), t.phs⟩

theorem cdel_val (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) (b : Bool) (h : cdel f = some b) :
    delV dv ν f = if b then 1 else 0 := by
  unfold cdel at h
  split at h
  · next c hc =>
    split_ifs at h with hk
    simp only [Option.some.injEq] at h
    subst h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hk
    unfold delV
    rw [ipConst_spec _ c hc ν, den_const_c dv f.1 hk.1]
    by_cases hd : c % (f.1.c : ℤ) = 0
    · simp [hd, Int.dvd_of_emod_eq_zero hd]
    · have : ¬ ((f.1.c : ℤ) ∣ c) := fun h => hd (Int.emod_eq_zero_of_dvd h)
      simp [hd, this]
  · simp at h


theorem list_fsum_comm {α : Type} (l : List α) (D : ℕ) (f : α → ℕ → ℂ) :
    (l.map fun a => ∑ v ∈ range D, f a v).sum = ∑ v ∈ range D, (l.map fun a => f a v).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih, Finset.sum_add_distrib]

theorem prod_prune (dv : ℕ → ℕ) (ν : ℕ → ℤ) : ∀ ds : List Fac, (ds.map (delV dv ν)).prod =
    ((ds.filter fun f => cdel f != some true).map (delV dv ν)).prod *
      (if ds.any (fun f => cdel f == some false) then 0 else 1)
  | [] => by simp
  | f :: ds => by
    have ih := prod_prune dv ν ds
    rw [List.map_cons, List.prod_cons, ih, List.any_cons, List.filter_cons]
    cases hf : cdel f with
    | none =>
      simp only [reduceCtorEq, bne_iff_ne, ne_eq, not_false_eq_true, ite_true, beq_iff_eq,
        Bool.false_or, List.map_cons, List.prod_cons]
      rw [show ((none : Option Bool) == some false) = false from rfl, Bool.false_or]
      ring
    | some b =>
      rw [cdel_val dv ν f b hf]
      cases b
      · simp only [Option.some.injEq, bne_iff_ne, ne_eq, Bool.false_eq_true, not_false_eq_true,
          ite_true, beq_self_eq_true, Bool.true_or, ite_false, List.map_cons, List.prod_cons,
          cdel_val dv ν f false hf]
        simp
      · simp only [Option.some.injEq, bne_self_eq_false, Bool.false_eq_true, ite_false,
          if_true, one_mul]
        rw [show ((some true : Option Bool) == some false) = false from rfl, Bool.false_or]

theorem pruneTm_den (ρ : Val) (t : STm) (hs : t.sums = []) (ν : ℕ → ℤ) :
    (match pruneTm t with | some t' => t'.den ρ ν | none => 0) = t.den ρ ν := by
  obtain ⟨c, sums, ds, ps⟩ := t
  simp only at hs
  subst hs
  unfold pruneTm
  simp only
  rw [den_zero, bodyOf, prod_prune ρ.dv ν ds]
  split_ifs with h
  · simp
  · simp only [den_zero, bodyOf]; ring

theorem modOK_prune (D : ℕ) (t t' : STm) (h : modOK D t = true) (hp : pruneTm t = some t') :
    modOK D t' = true := by
  unfold pruneTm at hp
  split_ifs at hp
  simp only [Option.some.injEq] at hp
  subst hp
  simp only [modOK, Bool.and_eq_true, List.isEmpty_iff, List.all_eq_true, List.mem_append,
    List.mem_filter] at h ⊢
  exact ⟨h.1, fun f hf => h.2 f (hf.imp_left (·.1))⟩

theorem exDen_prune (ρ : Val) (e : List STm) (hs : ∀ t ∈ e, t.sums = []) (ν : ℕ → ℤ) :
    exDen ρ ν (e.filterMap pruneTm) = exDen ρ ν e := by
  induction e with
  | nil => rfl
  | cons t e ih =>
    have ih' := ih fun u hu => hs u (List.mem_cons_of_mem _ hu)
    have ht := pruneTm_den ρ t (hs t List.mem_cons_self) ν
    rw [List.filterMap_cons, exDen_cons, ← ih']
    revert ht
    cases pruneTm t with
    | none => intro ht; simp only at ht; rw [← ht, zero_add]
    | some t' => intro ht; simp only at ht; rw [exDen_cons, ht]

/-- Every pinned label split into its residues mod `D`, constant deltas evaluated. -/
def explode (D : ℕ) : List ℕ → List ℕ → List STm → List (List ℕ × STm)
  | [], vv, e => e.map fun t => (vv, t)
  | x :: xs, vv, e => (List.range D).flatMap fun v =>
      explode D xs (vv ++ [v]) ((e.map (subTm x v)).filterMap pruneTm)

/-- The indicators of the residues `vv` of the labels `pv`. -/
noncomputable def indV (D : ℕ) (ν : ℕ → ℤ) (pv vv : List ℕ) : ℂ :=
  ((pv.zip vv).map fun p => ind D ν p.1 p.2).prod

theorem indV_snoc (D : ℕ) (ν : ℕ → ℤ) (pv vv : List ℕ) (h : pv.length = vv.length) (x v : ℕ) :
    indV D ν (pv ++ [x]) (vv ++ [v]) = indV D ν pv vv * ind D ν x v := by
  unfold indV
  rw [List.zip_append h]
  simp

theorem explode_ok (ρ : Val) (hρ : ρ.OK) (D : ℕ) (hD : 0 < D) (ν : ℕ → ℤ) :
    ∀ (xs done vv : List ℕ) (e : List STm), done.length = vv.length →
      (∀ t ∈ e, modOK D t = true) →
      ((explode D xs vv e).map fun s => indV D ν (done ++ xs) s.1 * s.2.den ρ ν).sum =
        indV D ν done vv * exDen ρ ν e
  | [], done, vv, e, _, _ => by
    simp only [explode, List.map_map, List.append_nil]
    unfold exDen
    rw [← List.sum_map_mul_left]
    rfl
  | x :: xs, done, vv, e, hl, hm => by
    simp only [explode, List.map_flatMap]
    rw [lsum_flatMap]
    have hsub : ∀ v, ∀ t ∈ (e.map (subTm x v)).filterMap pruneTm, modOK D t = true := by
      intro v t ht
      simp only [List.mem_filterMap, List.mem_map] at ht
      obtain ⟨u, ⟨w, hw, rfl⟩, hu⟩ := ht
      exact modOK_prune D _ t (modOK_sub D x v w (hm w hw)) hu
    have step : ∀ v, ((explode D xs (vv ++ [v]) ((e.map (subTm x v)).filterMap pruneTm)).map
        fun s => indV D ν (done ++ x :: xs) s.1 * s.2.den ρ ν).sum =
        indV D ν done vv * (ind D ν x v * exDen ρ ν ((e.map (subTm x v)).filterMap pruneTm)) := by
      intro v
      have ih := explode_ok ρ hρ D hD ν xs (done ++ [x]) (vv ++ [v])
        ((e.map (subTm x v)).filterMap pruneTm) (by simp [hl]) (hsub v)
      rw [List.append_assoc, List.singleton_append] at ih
      rw [ih, indV_snoc D ν done vv hl x v]
      ring
    simp only [step]
    rw [lsum_range, ← Finset.mul_sum]
    congr 1
    have hs : ∀ v, ∀ t ∈ e.map (subTm x v), t.sums = [] := by
      intro v t ht
      simp only [List.mem_map] at ht
      obtain ⟨w, hw, rfl⟩ := ht
      have := hm w hw
      simp only [modOK, Bool.and_eq_true, List.isEmpty_iff] at this
      exact this.1
    simp only [exDen_prune ρ _ (hs _) ν]
    have hd : ∀ t ∈ e, t.den ρ ν = ∑ v ∈ range D, ind D ν x v * (subTm x v t).den ρ ν :=
      fun t ht => den_split ρ hρ x D hD t (hm t ht) ν
    have he : exDen ρ ν e = (e.map fun t => ∑ v ∈ range D, ind D ν x v * (subTm x v t).den ρ ν).sum := by
      unfold exDen; congr 1; exact List.map_congr_left hd
    rw [he, list_fsum_comm]
    refine Finset.sum_congr rfl fun v _ => ?_
    unfold exDen
    rw [List.map_map, ← List.sum_map_mul_left]
    rfl

/-! ## Slots with residues -/

/-- The slot of residues `vv` and a character `P`: `vv` then `P`'s digits, read in base `N`. -/
def slot2 (vs : List ℕ) (N : ℕ) (vv : List ℕ) (P : IPoly) : ℕ :=
  Nat.ofDigits N (vv ++ digitsOf vs N P)

theorem slot2_eq (vs : List ℕ) (N : ℕ) (hN : 1 < N) (vv vv' : List ℕ) (P P' : IPoly)
    (hl : vv.length = vv'.length) (hv : ∀ a ∈ vv, a < N) (hv' : ∀ a ∈ vv', a < N)
    (h : linOK vs P = true) (h' : linOK vs P' = true) (hs : slot2 vs N vv P = slot2 vs N vv' P')
    (ν : ℕ → ℤ) : vv = vv' ∧ (N : ℤ) ∣ ipDen ν P - ipDen ν P' := by
  have hN0 : (0 : ℤ) < N := by exact_mod_cast (by omega : 0 < N)
  have hlt : ∀ Q : IPoly, ∀ l ∈ digitsOf vs N Q, l < N := by
    intro Q l hl
    simp only [digitsOf, List.mem_map, List.mem_range] at hl
    obtain ⟨i, _, rfl⟩ := hl
    have h1 := Int.emod_lt_of_pos (cAt vs Q i) hN0
    have h2 := Int.emod_nonneg (cAt vs Q i) hN0.ne'
    have : ((cAt vs Q i % N).toNat : ℤ) = cAt vs Q i % N := Int.toNat_of_nonneg h2
    omega
  have hd := Nat.ofDigits_inj_of_len_eq hN (by simp [digitsOf, hl])
    (fun a ha => (List.mem_append.1 ha).elim (hv a) (hlt P a))
    (fun a ha => (List.mem_append.1 ha).elim (hv' a) (hlt P' a)) hs
  obtain ⟨h1, h2⟩ := List.append_inj hd hl
  refine ⟨h1, ?_⟩
  have := slot_eq vs N hN P P' h h' (by unfold slotOf; rw [h2]) ν
  exact this

/-! ## The certified check -/

/-- The cycles' characters, each weight negated, with their residues. -/
def certItems (N : ℕ) (rows : List (List ℕ × IPoly × ℕ × ℕ × ℚ)) : List (List ℕ × ℚ × IPoly) :=
  rows.flatMap fun r => (List.range r.2.2.1).map fun i =>
    (r.1, -r.2.2.2.2, ipAdd r.2.1 (ipC ((r.2.2.2.1 + i * (N / r.2.2.1) : ℕ) : ℤ)))

/-- Every cycle has `1 < p ∣ N`. -/
def rowsOK (N : ℕ) (rows : List (List ℕ × IPoly × ℕ × ℕ × ℚ)) : Bool :=
  rows.all fun r => decide (1 < r.2.2.1) && N % r.2.2.1 == 0

/-- The value of items `Σ w [residues] ω_N^P`. -/
noncomputable def itV (N : ℕ) (ν : ℕ → ℤ) (pv : List ℕ) (l : List (List ℕ × ℚ × IPoly)) : ℂ :=
  (l.map fun s => (s.2.1 : ℂ) * (indV N ν pv s.1 * om N ^ ipDen ν s.2.2)).sum

theorem itV_append (N : ℕ) (ν : ℕ → ℤ) (pv : List ℕ) (a b : List (List ℕ × ℚ × IPoly)) :
    itV N ν pv (a ++ b) = itV N ν pv a + itV N ν pv b := by
  simp [itV, List.map_append, List.sum_append]

theorem itV_cert (N : ℕ) (hN : 0 < N) (ν : ℕ → ℤ) (pv : List ℕ)
    (rows : List (List ℕ × IPoly × ℕ × ℕ × ℚ)) (h : rowsOK N rows = true) :
    itV N ν pv (certItems N rows) = 0 := by
  induction rows with
  | nil => simp [certItems, itV]
  | cons r rows ih =>
    simp only [rowsOK, List.all_cons, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h ih
    have ih' := ih h.2
    unfold certItems at ih' ⊢
    rw [List.flatMap_cons, itV_append, ih', add_zero]
    unfold itV
    rw [List.map_map, lsum_range]
    have hg := om_geom N r.2.2.1 hN h.1.1 (Nat.dvd_of_mod_eq_zero h.1.2)
    have hz := om_ne_zero N hN
    have e : ∀ i : ℕ, ((fun s : List ℕ × ℚ × IPoly => (s.2.1 : ℂ) *
        (indV N ν pv s.1 * om N ^ ipDen ν s.2.2)) ∘ fun i => (r.1, -r.2.2.2.2,
          ipAdd r.2.1 (ipC ((r.2.2.2.1 + i * (N / r.2.2.1) : ℕ) : ℤ)))) i =
        (-(r.2.2.2.2 : ℂ)) * indV N ν pv r.1 * om N ^ ipDen ν r.2.1 * om N ^ r.2.2.2.1 *
          om N ^ (i * (N / r.2.2.1)) := by
      intro i
      simp only [Function.comp_apply, ipDen_add, ipDen_C, zpow_add₀ hz, zpow_natCast, pow_add]
      push_cast
      ring
    simp only [e, ← Finset.mul_sum, hg, mul_zero]

/-- The characters of each split term, with its residues. -/
def charsOf (N : ℕ) : List (List ℕ × STm) → Option (List (List ℕ × ℚ × IPoly))
  | [] => some []
  | s :: l =>
    match expTerm N s.2, charsOf N l with
    | some L, some a => some (L.map (fun c => (s.1, c.1, c.2)) ++ a)
    | _, _ => none

theorem charsOf_ok (ρ : Val) (N : ℕ) (hN : 0 < N) (ν : ℕ → ℤ) (pv : List ℕ) :
    ∀ (l : List (List ℕ × STm)) (a : List (List ℕ × ℚ × IPoly)), charsOf N l = some a →
      itV N ν pv a = (l.map fun s => indV N ν pv s.1 * s.2.den ρ ν).sum
  | [], a, h => by simp only [charsOf, Option.some.injEq] at h; subst h; simp [itV]
  | s :: l, a, h => by
    simp only [charsOf] at h
    split at h
    · next L b hL hb =>
      simp only [Option.some.injEq] at h
      subst h
      rw [itV_append, charsOf_ok ρ N hN ν pv l b hb, List.map_cons, List.sum_cons,
        ← expTerm_ok ρ N hN s.2 L hL ν]
      congr 1
      unfold itV expV
      rw [List.map_map, ← List.sum_map_mul_left]
      congr 1
      refine List.map_congr_left fun c _ => ?_
      simp only [Function.comp_apply]
      ring
    · simp at h

/-- A certificate: the pinned labels, the slot variables and the cycles `(vv, K, p, a, λ)`,
each `λ [residues vv] Σ_{i<p} ω_N^(K + a + i N/p)`. -/
abbrev FzCert := List ℕ × List ℕ × List (List ℕ × IPoly × ℕ × ℕ × ℚ)

/-- Slot order. -/
def slotLe (a b : ℕ × (List ℕ × IPoly) × ℚ) : Bool := decide (a.1 ≤ b.1)

/-- The items of a sum-free `e` (every modulus dividing `N`, the pinned labels split into
residues) and the certificate's negated cycles cancel slot by slot. -/
def fzOK (N : ℕ) (cert : FzCert) (e : List STm) : Bool :=
  decide (1 < N) && rowsOK N cert.2.2 && e.all (modOK N) &&
    match charsOf N (explode N cert.1 [] e) with
    | none => false
    | some a =>
      (a ++ certItems N cert.2.2).all (fun s => s.1.length == cert.1.length &&
        s.1.all (fun v => decide (v < N)) && linOK cert.2.1 s.2.2) &&
      runsGo none 0 (PSem.sortB slotLe ((a ++ certItems N cert.2.2).map fun s =>
        (slot2 cert.2.1 N s.1 s.2.2, (s.1, s.2.2), s.2.1)))

theorem fz_zero (ρ : Val) (hρ : ρ.OK) (N : ℕ) (cert : FzCert) (e : List STm)
    (h : fzOK N cert e = true) (ν : ℕ → ℤ) : exDen ρ ν e = 0 := by
  unfold fzOK at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨hN, hrows⟩, hmod⟩, h⟩ := h
  have hN0 : 0 < N := by omega
  split at h
  · simp at h
  · next a ha =>
    simp only [Bool.and_eq_true, List.all_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    obtain ⟨hall, hruns⟩ := h
    set A := a ++ certItems N cert.2.2 with hA
    have hval : itV N ν cert.1 A = exDen ρ ν e := by
      rw [hA, itV_append, itV_cert N hN0 ν cert.1 _ hrows, add_zero,
        charsOf_ok ρ N hN0 ν cert.1 _ a ha]
      have := explode_ok ρ hρ N hN0 ν cert.1 [] [] e rfl hmod
      simpa [indV] using this
    set S := A.map fun s => (slot2 cert.2.1 N s.1 s.2.2, (s.1, s.2.2), s.2.1) with hS
    let χ : List ℕ × IPoly → ℂ := fun q => indV N ν cert.1 q.1 * om N ^ ipDen ν q.2
    let g : ℕ → ℂ := fun s => match S.find? (fun b => b.1 == s) with
      | some b => χ b.2.1
      | none => 0
    have hg : ∀ b ∈ S, χ b.2.1 = g b.1 := by
      intro b hb
      have hsome : (S.find? fun c => c.1 == b.1).isSome := by
        rw [List.find?_isSome]; exact ⟨b, hb, by simp⟩
      obtain ⟨c, hc⟩ := Option.isSome_iff_exists.1 hsome
      have hcm := List.mem_of_find?_eq_some hc
      have hck : c.1 = b.1 := by simpa using List.find?_some hc
      simp only [g, hc]
      simp only [hS, List.mem_map] at hb hcm
      obtain ⟨s, hs, rfl⟩ := hb
      obtain ⟨u, hu, rfl⟩ := hcm
      obtain ⟨⟨hl1, hv1⟩, hn1⟩ := hall s hs
      obtain ⟨⟨hl2, hv2⟩, hn2⟩ := hall u hu
      obtain ⟨hvv, hdv⟩ := slot2_eq cert.2.1 N hN s.1 u.1 s.2.2 u.2.2 (by rw [hl1, hl2])
        hv1 hv2 hn1 hn2 hck.symm ν
      simp only [χ, hvv, om_zpow_congr N hN0 _ _ hdv]
    have hS1 : itV N ν cert.1 A = (S.map fun b => (b.2.2 : ℂ) * χ b.2.1).sum := by
      simp only [hS, itV, List.map_map]
      rfl
    have hperm := PSem.sortB_perm slotLe S
    rw [← hval, hS1, ← (hperm.map _).sum_eq]
    have hc : ((PSem.sortB slotLe S).map fun b => (b.2.2 : ℂ) * χ b.2.1) =
        (PSem.sortB slotLe S).map fun b => (b.2.2 : ℂ) * g b.1 := by
      refine List.map_congr_left fun b hb => ?_
      rw [hg b (hperm.mem_iff.1 hb)]
    rw [hc]
    exact runs_ok g _ hruns

/-- `nf e`, or nothing when its characters are certified to vanish. -/
def fzNf (N : ℕ) (cert : FzCert) (nf : List STm → List STm) (e : List STm) : List STm :=
  if fzOK N cert (nf e) then [] else nf e

theorem fzNf_den (ρ : Val) (hρ : ρ.OK) (N : ℕ) (cert : FzCert)
    (nf : List STm → List STm) (hnf : ∀ e ν, exDen ρ ν (nf e) = exDen ρ ν e) :
    ∀ e ν, exDen ρ ν (fzNf N cert nf e) = exDen ρ ν e := by
  intro e ν
  unfold fzNf
  split_ifs with h
  · rw [← hnf e ν, fz_zero ρ hρ N cert _ h ν]; rfl
  · exact hnf e ν

/-- A phase table's entry off its listed indices `R` is `0`, so its turn is `1`. -/
theorem tab_zero (l R : List ℕ) (c : ℝ) (j : ℕ)
    (hl : (List.range l.length).all (fun i => l.getD i 0 == 0 || R.contains i) = true)
    (hj : j ∉ R) : ex ((((l[j]?.getD 0 : ℕ) : ℝ)) / c) = 1 := by
  have h0 : l[j]?.getD 0 = 0 := by
    by_cases hlt : j < l.length
    · have := (List.all_eq_true.1 hl) j (List.mem_range.2 hlt)
      simp only [Bool.or_eq_true, beq_iff_eq, List.contains_iff_mem] at this
      rcases this with h | h
      · rw [List.getD_eq_getElem?_getD] at h; exact h
      · exact absurd h hj
    · rw [List.getElem?_eq_none (by omega)]; rfl
  rw [h0]; simp [ex_zero]

/-- `tab_zero` on an integer table. -/
theorem tab_zeroZ (l : List ℤ) (R : List ℕ) (c : ℝ) (j : ℕ)
    (hl : (List.range l.length).all (fun i => l.getD i 0 == 0 || R.contains i) = true)
    (hj : j ∉ R) : ex ((((l[j]?.getD 0 : ℤ) : ℝ)) / c) = 1 := by
  have h0 : l[j]?.getD 0 = 0 := by
    by_cases hlt : j < l.length
    · have := (List.all_eq_true.1 hl) j (List.mem_range.2 hlt)
      simp only [Bool.or_eq_true, beq_iff_eq, List.contains_iff_mem] at this
      rcases this with h | h
      · rw [List.getD_eq_getElem?_getD] at h; exact h
      · exact absurd h hj
    · rw [List.getElem?_eq_none (by omega)]; rfl
  rw [h0]; simp [ex_zero]

theorem phC_den0 (ρ : Val) (q : ℚ) : (phC q []).den ρ = ex (q : ℝ) := by
  have hr := rnorm_den 1 q
  simp only [phC, CMon.den, zpw, ztrim, List.map_nil, List.prod_nil, mul_one]
  rw [hr]; push_cast; ring

/-- Distinct keys, each entry `phC (l_i / c) []`, and every `l_i` off the keys a multiple of `c`. -/
def ptOK (l : List ℤ) (c : ℕ) (pt : List (ℕ × CMon)) : Bool :=
  decide (0 < c) && decide (pt.map Prod.fst).Nodup &&
    pt.all (fun p => p.2 == phC ((l.getD p.1 0 : ℚ) / c) []) &&
    (List.range l.length).all (fun i => l.getD i 0 % c == 0 || (pt.map Prod.fst).contains i)

theorem sum_key (pt : List (ℕ × CMon)) (f : CMon → ℂ) (hnd : (pt.map Prod.fst).Nodup) (j : ℕ)
    (p : ℕ × CMon) (hp : p ∈ pt) (hj : p.1 = j) :
    (pt.map fun p => if j = p.1 then f p.2 else 0).sum = f p.2 := by
  induction pt with
  | nil => simp at hp
  | cons a rest ih =>
    simp only [List.map_cons, List.nodup_cons] at hnd
    simp only [List.map_cons, List.sum_cons]
    have hz : ∀ k, k ∉ rest.map Prod.fst →
        (rest.map fun p => if k = p.1 then f p.2 else 0).sum = 0 := by
      intro k hk
      apply List.sum_eq_zero
      intro x hx
      obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hx
      have : k ≠ b.1 := fun h => hk (h ▸ List.mem_map_of_mem hb)
      simp [this]
    rcases List.mem_cons.1 hp with rfl | hp'
    · rw [ite_cond_eq_true _ _ (eq_true hj.symm), hz _ (hj ▸ hnd.1), add_zero]
    · have hne : j ≠ a.1 := by
        intro h; apply hnd.1; rw [← h, ← hj]; exact List.mem_map_of_mem hp'
      rw [ite_cond_eq_false _ _ (eq_false hne), ih hnd.2 hp', zero_add]

theorem sum_nokey (pt : List (ℕ × CMon)) (f : CMon → ℂ) (j : ℕ) (hj : j ∉ pt.map Prod.fst) :
    (pt.map fun p => if j = p.1 then f p.2 else 0).sum = 0 := by
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hx
  have : j ≠ b.1 := fun h => hj (h ▸ List.mem_map_of_mem hb)
  simp [this]

/-- A rational phase table `l / c` against its stored phases. -/
theorem tab_ph (ρ : Val) (l : List ℤ) (c : ℕ) (cr : ℝ) (hc : cr = c) (pt : List (ℕ × CMon))
    (h : ptOK l c pt = true) (j : ℕ) : ex (((l.getD j 0 : ℤ) : ℝ) / cr) = phF ρ pt j := by
  simp only [ptOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨hc0, hnd⟩, hall⟩, hcov⟩ := h
  subst hc
  unfold phF
  by_cases hk : j ∈ pt.map Prod.fst
  · obtain ⟨p, hp, hpj⟩ := List.mem_map.1 hk
    rw [sum_key pt (fun m => m.den ρ - 1) hnd j p hp hpj]
    have he := hall p hp
    simp only [beq_iff_eq] at he
    rw [he, phC_den0, hpj]; push_cast; ring
  · rw [sum_nokey pt (fun m => m.den ρ - 1) j hk, add_zero]
    have hm : (c : ℤ) ∣ l.getD j 0 := by
      by_cases hlt : j < l.length
      · have := hcov j (List.mem_range.2 hlt)
        simp only [Bool.or_eq_true, beq_iff_eq, List.contains_iff_mem] at this
        rcases this with h | h
        · exact Int.dvd_of_emod_eq_zero h
        · exact absurd h hk
      · rw [List.getD_eq_default _ _ (by omega)]; exact dvd_zero _
    obtain ⟨z, hz⟩ := hm
    rw [hz]
    have hc' : (c : ℝ) ≠ 0 := by exact_mod_cast hc0.ne'
    have : (((c : ℤ) * z : ℤ) : ℝ) / (c : ℝ) = 0 + (z : ℝ) := by
      push_cast; field_simp; ring
    rw [this, ex_int, ex_zero]

/-! ## The check -/

/-- `checkWith` with the certified characters beside the empty normal form. -/
def checkWithF (N : ℕ) (cert : FzCert) (tr : Net → Option (List STm))
    (fl : List ℕ) (N0 N1 : Net) (S0 S1 : List STm) : Bool :=
  match tr N0, tr N1 with
  | some e0, some e1 => exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
      ((nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty ||
        fzOK N cert (nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)))
  | _, _ => false

/-- `symCheck` with `checkWithF`. -/
def symCheckF (N : ℕ) (cert : FzCert) (fl : List ℕ) (dt : ℕ → DMon)
    (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (N0 N1 : Net) (S0 S1 : List STm) : Bool :=
  N0.bnd.isPerm N1.bnd && netChk fl dt pt rl labs IX N0 && netChk fl dt pt rl labs IX N1 &&
    exConst S0 && exConst S1 && (checkWithF N cert (trNet fl dt pt rl labs) fl N0 N1 S0 S1 ||
      checkWithF N cert (trNetF dt pt rl labs) fl N0 N1 S0 S1)

theorem equal_of_checkF (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (N : ℕ) (cert : FzCert)
    (tr : Net → Option (List STm)) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (htr0 : ∀ e, tr N0 = some e → ∀ σ, InRange E N0.bnd σ → eval E N0 σ = exDen ρ (fun x => (σ x : ℤ)) e)
    (htr1 : ∀ e, tr N1 = some e → ∀ σ, InRange E N1.bnd σ → eval E N1 σ = exDen ρ (fun x => (σ x : ℤ)) e)
    (hp : N0.bnd.Perm N1.bnd) (hk0 : exConst S0 = true) (hk1 : exConst S1 = true)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : checkWithF N cert tr fl N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold checkWithF at h
  split at h
  · next e0 e1 ht0 ht1 =>
    simp only [Bool.and_eq_true, Bool.or_eq_true, List.isEmpty_iff] at h
    obtain ⟨⟨hm0, hm1⟩, hnf⟩ := h
    refine ⟨hp, fun σ hσ => ?_⟩
    set ν : ℕ → ℤ := fun x => (σ x : ℤ)
    have hz : exDen ρ ν (exMul S0 e0 ++ exMul (exNeg S1) e1) = 0 := by
      rcases hnf with hnf | hnf
      · exact nfEx_zero ρ hρ fl hfl _ hnf ν
      · rw [← nfEx_den ρ hρ fl hfl _ ν]; exact fz_zero ρ hρ N cert _ hnf ν
    rw [exDen_append, exDen_mul ρ hρ _ _ hm0, exDen_mul ρ hρ _ _ hm1, exDen_neg,
      ← htr0 e0 ht0 σ hσ, ← htr1 e1 ht1 σ (hσ.perm hp),
      exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
      ← hs1] at hz
    linear_combination hz
  · simp at h

theorem equal_symF (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (N : ℕ) (cert : FzCert)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : symCheckF N cert fl dt pt rl labs IX N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold symCheckF at h
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hperm, hc0⟩, hc1⟩, hk0⟩, hk1⟩, hrest⟩ := h
  have hp : N0.bnd.Perm N1.bnd := List.isPerm_iff.1 hperm
  rcases hrest with hb | hf
  · exact equal_of_checkF E ρ hρ fl hfl N cert _ N0 N1 S0 S1 s0 s1
      (fun e ht σ hσ => eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e hc0 ht σ hσ)
      (fun e ht σ hσ => eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e hc1 ht σ hσ)
      hp hk0 hk1 hs0 hs1 hb
  · exact equal_of_checkF E ρ hρ fl hfl N cert _ N0 N1 S0 S1 s0 s1
      (fun e ht σ hσ => eval_trF E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e hc0 ht σ hσ)
      (fun e ht σ hσ => eval_trF E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e hc1 ht σ hσ)
      hp hk0 hk1 hs0 hs1 hf

end ArchytasZX.Sym
