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

/-- A certificate: the variable order of the slots and the cycles `(K, p, a, λ)`, each
`λ Σ_{i<p} ω_N^(K + a + i N/p)`. -/
abbrev FzCert := List ℕ × List (IPoly × ℕ × ℕ × ℚ)

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
def runsGo : Option ℕ → ℚ → List (ℕ × IPoly × ℚ) → Bool
  | _, q, [] => q == 0
  | none, _, a :: l => runsGo (some a.1) a.2.2 l
  | some r, q, a :: l => if a.1 = r then runsGo (some r) (q + a.2.2) l
      else q == 0 && runsGo (some a.1) a.2.2 l

theorem runsGo_ok (g : ℕ → ℂ) :
    ∀ (r : ℕ) (q : ℚ) (l : List (ℕ × IPoly × ℚ)), runsGo (some r) q l = true →
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

theorem runs_ok (g : ℕ → ℂ) (l : List (ℕ × IPoly × ℚ)) (h : runsGo none 0 l = true) :
    (l.map fun a => (a.2.2 : ℂ) * g a.1).sum = 0 := by
  match l, h with
  | [], _ => simp
  | a :: l, h =>
    simp only [runsGo] at h
    have := runsGo_ok g a.1 a.2.2 l h
    simpa using this

/-! ## The certified check -/

/-- The cycles' characters, each weight negated. -/
def certItems (N : ℕ) (rows : List (IPoly × ℕ × ℕ × ℚ)) : List (ℚ × IPoly) :=
  rows.flatMap fun r => (List.range r.2.1).map fun i =>
    (-r.2.2.2, ipAdd r.1 (ipC ((r.2.2.1 + i * (N / r.2.1) : ℕ) : ℤ)))

/-- Every cycle has `1 < p ∣ N`. -/
def rowsOK (N : ℕ) (rows : List (IPoly × ℕ × ℕ × ℚ)) : Bool :=
  rows.all fun r => decide (1 < r.2.1) && N % r.2.1 == 0

theorem expV_cert (N : ℕ) (hN : 0 < N) (ν : ℕ → ℤ) (rows : List (IPoly × ℕ × ℕ × ℚ))
    (h : rowsOK N rows = true) : expV N ν (certItems N rows) = 0 := by
  induction rows with
  | nil => simp [certItems, expV]
  | cons r rows ih =>
    simp only [rowsOK, List.all_cons, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h ih
    have ih' := ih h.2
    unfold certItems at ih' ⊢
    rw [List.flatMap_cons, expV_append, ih', add_zero]
    unfold expV
    rw [List.map_map, lsum_range]
    have hg := om_geom N r.2.1 hN h.1.1 (Nat.dvd_of_mod_eq_zero h.1.2)
    have hz := om_ne_zero N hN
    have e : ∀ i : ℕ, ((fun s : ℚ × IPoly => (s.1 : ℂ) * om N ^ ipDen ν s.2) ∘
        fun i => (-r.2.2.2, ipAdd r.1 (ipC ((r.2.2.1 + i * (N / r.2.1) : ℕ) : ℤ)))) i =
        (-(r.2.2.2 : ℂ)) * om N ^ ipDen ν r.1 * om N ^ r.2.2.1 * om N ^ (i * (N / r.2.1)) := by
      intro i
      simp only [Function.comp_apply, ipDen_add, ipDen_C, zpow_add₀ hz]
      rw [show (((r.2.2.1 + i * (N / r.2.1) : ℕ) : ℤ)) = ((r.2.2.1 + i * (N / r.2.1) : ℕ) : ℤ)
        from rfl, zpow_natCast, pow_add]
      push_cast
      ring
    simp only [e, ← Finset.mul_sum, hg, mul_zero]

/-- The slot items of characters. -/
def slotItems (vs : List ℕ) (N : ℕ) (L : List (ℚ × IPoly)) : List (ℕ × IPoly × ℚ) :=
  L.map fun s => (slotOf vs N s.2, s.2, s.1)

/-- Slot order. -/
def slotLe (a b : ℕ × IPoly × ℚ) : Bool := decide (a.1 ≤ b.1)

/-- The characters of `e` and the certificate's cycles, negated, cancel slot by slot. -/
def fzOK (N : ℕ) (cert : FzCert) (e : List STm) : Bool :=
  decide (1 < N) && rowsOK N cert.2 &&
    match expEx N e with
    | none => false
    | some L =>
      (L ++ certItems N cert.2).all (fun s => linOK cert.1 s.2) &&
        runsGo none 0 (PSem.sortB slotLe (slotItems cert.1 N (L ++ certItems N cert.2)))

theorem fz_zero (ρ : Val) (N : ℕ) (cert : FzCert) (e : List STm) (h : fzOK N cert e = true)
    (ν : ℕ → ℤ) : exDen ρ ν e = 0 := by
  unfold fzOK at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hN, hrows⟩, h⟩ := h
  have hN0 : 0 < N := by omega
  split at h
  · simp at h
  · next L hL =>
    simp only [Bool.and_eq_true, List.all_eq_true] at h
    obtain ⟨hlin, hruns⟩ := h
    set A := L ++ certItems N cert.2 with hA
    set S := slotItems cert.1 N A with hS
    have hval : expV N ν A = exDen ρ ν e := by
      rw [hA, expV_append, expV_cert N hN0 ν cert.2 hrows, add_zero, expEx_ok ρ N hN0 e L hL ν]
    -- the character of a slot, read off the first item in that slot
    let g : ℕ → ℂ := fun s => match S.find? (fun a => a.1 == s) with
      | some a => om N ^ ipDen ν a.2.1
      | none => 0
    have hg : ∀ a ∈ S, om N ^ ipDen ν a.2.1 = g a.1 := by
      intro a ha
      have hsome : (S.find? fun b => b.1 == a.1).isSome := by
        rw [List.find?_isSome]; exact ⟨a, ha, by simp⟩
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hsome
      have hbm := List.mem_of_find?_eq_some hb
      have hbk : b.1 = a.1 := by simpa using List.find?_some hb
      simp only [g, hb]
      simp only [hS, slotItems, List.mem_map] at ha hbm
      obtain ⟨s, hs, rfl⟩ := ha
      obtain ⟨u, hu, rfl⟩ := hbm
      exact om_zpow_congr N hN0 _ _
        (slot_eq cert.1 N hN s.2 u.2 (hlin s hs) (hlin u hu) hbk.symm ν)
    have hS1 : expV N ν A = (S.map fun a => (a.2.2 : ℂ) * om N ^ ipDen ν a.2.1).sum := by
      simp only [hS, slotItems, expV, List.map_map]
      rfl
    have hperm := PSem.sortB_perm slotLe S
    rw [← hval, hS1, ← (hperm.map _).sum_eq]
    have hc : ((PSem.sortB slotLe S).map fun a => (a.2.2 : ℂ) * om N ^ ipDen ν a.2.1) =
        (PSem.sortB slotLe S).map fun a => (a.2.2 : ℂ) * g a.1 := by
      refine List.map_congr_left fun a ha => ?_
      rw [hg a (hperm.mem_iff.1 ha)]
    rw [hc]
    exact runs_ok g _ hruns

/-- `nf e`, or nothing when its characters are certified to vanish. -/
def fzNf (N : ℕ) (cert : FzCert) (nf : List STm → List STm)
    (e : List STm) : List STm :=
  if fzOK N cert (nf e) then [] else nf e

theorem fzNf_den (ρ : Val) (N : ℕ) (cert : FzCert)
    (nf : List STm → List STm) (hnf : ∀ e ν, exDen ρ ν (nf e) = exDen ρ ν e) :
    ∀ e ν, exDen ρ ν (fzNf N cert nf e) = exDen ρ ν e := by
  intro e ν
  unfold fzNf
  split_ifs with h
  · rw [← hnf e ν, fz_zero ρ N cert _ h ν]; rfl
  · exact hnf e ν

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
      · rw [← nfEx_den ρ hρ fl hfl _ ν]; exact fz_zero ρ N cert _ hnf ν
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
