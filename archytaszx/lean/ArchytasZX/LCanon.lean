/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LDigit

/-!
# The normaliser of symbolic sums with atoms

`normA` alternates eliminations of bound variables (`elimPlain`, `elimPin`, `elimDigit`) with
canonicalisation: atom rules (zero digit products, digit conditions read as deltas, unused
atoms), the factor canonicalisation of `Sym`, renaming of bound and atom variables by first
occurrence and sorting of the sums and the table. `nfA` normalises every term of an expression
and merges terms of equal shape; `nfA_den` proves it keeps the value.
-/

namespace ArchytasZX.LSym

open ArchytasZX.Sym Complex Finset

/-- `t'` has the value of `t`, or `t` vanishes. -/
def ARes (V : LV) (t : ATm) : Option ATm → Prop
  | none => ∀ ν, t.den V ν = 0
  | some t' => ∀ ν, t'.den V ν = t.den V ν

theorem den_of_pw (V : LV) (t t' : ATm) (hs : t'.sums = t.sums)
    (h : ∀ μ, t'.c.den V.ρ * abody V t' μ = t.c.den V.ρ * abody V t μ) (ν : ℕ → ℤ) :
    t'.den V ν = t.den V ν := by
  unfold ATm.den
  rw [hs, ← sumsV_mul_left, ← sumsV_mul_left]
  exact sumsV_congr _ _ _ _ h ν

theorem den_zero_of_pw (V : LV) (t : ATm) (h : ∀ μ, abody V t μ = 0) (ν : ℕ → ℤ) :
    t.den V ν = 0 := by
  unfold ATm.den
  rw [sumsV_congr _ _ _ (fun _ => 0) h, sumsV_zero, mul_zero]

/-! ## Eliminations -/

/-- The first elimination of `x` that applies. -/
def aElimVar (PW : List (DMon × DMon)) (x : ℕ) (D : DMon) (ats : List Atom) (ds ps : List Fac) :
    Option AElim :=
  match elimPlain x D ats ds ps with
  | some e => some e
  | none =>
    match elimPin x D ats ds ps with
    | some e => some e
    | none => elimDigit PW x D ats ds ps

theorem aElimVar_ok (V : LV) (hρ : V.ρ.OK) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ)
    (x : ℕ) (D : DMon) (ats : List Atom) (ds ps : List Fac) (e : AElim)
    (h : aElimVar PW x D ats ds ps = some e) : AElimOK V x D ats ds ps e := by
  unfold aElimVar at h
  split at h
  · next e' he => simp only [Option.some.injEq] at h; subst h; exact elimPlain_ok V hρ _ _ _ _ _ _ he
  · split at h
    · next e' he => simp only [Option.some.injEq] at h; subst h; exact elimPin_ok V hρ _ _ _ _ _ _ he
    · exact elimDigit_ok V hρ PW hPW _ _ _ _ _ _ h

/-- One elimination of a bound variable, the first that applies. -/
def aElimStep (PW : List (DMon × DMon)) (t : ATm) : Option ATm :=
  firstSome (fun p => if (p.1 ++ p.2.2).map (·.1) |>.contains p.2.1.1 then none else
    match aElimVar PW p.2.1.1 p.2.1.2 t.ats t.dels t.phs with
    | some e => some ⟨CMon.mul t.c e.k, p.1 ++ p.2.2, e.ats, e.ds, e.ps⟩
    | none => none) (splits t.sums)

theorem aElimStep_ok (V : LV) (hρ : V.ρ.OK) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ)
    (t t' : ATm) (h : aElimStep PW t = some t') (ν : ℕ → ℤ) : t'.den V ν = t.den V ν := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  obtain ⟨a, ⟨x, D⟩, b⟩ := p
  simp only at he
  split_ifs at he with hx
  split at he
  · next e hev =>
    simp only [Option.some.injEq] at he
    subst he
    have hok := aElimVar_ok V hρ PW hPW x D t.ats t.dels t.phs e hev
    have hs := splits_spec t.sums _ hp
    simp only at hs
    unfold ATm.den
    rw [hs, sumsV_pull _ a b x D (by simpa using hx)]
    show (CMon.mul t.c e.k).den V.ρ * sumsV V.ρ.dv (a ++ b)
        (fun μ => bodyOf V.ρ.dv e.ds e.ps (aext V e.ats μ)) ν =
      t.c.den V.ρ * sumsV V.ρ.dv (a ++ b) (fun ν' => ∑ j ∈ range (D.den V.ρ.dv),
        bodyOf V.ρ.dv t.dels t.phs (aext V t.ats (Function.update ν' x j))) ν
    rw [sumsV_congr _ _ _ (fun ν' => e.k.den V.ρ * bodyOf V.ρ.dv e.ds e.ps (aext V e.ats ν'))
      hok, sumsV_mul_left, CMon.den_mul V.ρ hρ]
    ring
  · simp at he


/-! ## Factors -/

/-- The factor canonicalisation of `Sym`: the coefficient gained and the factors, or `none`
when they vanish. -/
def canonF (fl : List ℕ) (ds ps : List Fac) : Option (CMon × List Fac × List Fac) :=
  let pp := pivPass ds.length [] ds ps
  match normDels fl pp.1 with
  | none => none
  | some ds' =>
    let np := normPhs (phMerge pp.2)
    some (np.1, ds'.foldr delIns [], np.2.foldr facIns [])

theorem canonF_none (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (ds ps : List Fac)
    (h : canonF fl ds ps = none) (μ : ℕ → ℤ) : bodyOf ρ.dv ds ps μ = 0 := by
  unfold canonF at h
  simp only at h
  split at h
  · next hn =>
    have := normDels_ok ρ fl hfl (pivPass ds.length [] ds ps).1 μ
    rw [hn] at this
    rw [← show bodyOf ρ.dv (pivPass ds.length [] ds ps).1 (pivPass ds.length [] ds ps).2 μ =
      bodyOf ρ.dv ds ps μ from by rw [pivPass_ok ρ.dv hρ.dpos]; rfl]
    simp only [bodyOf]; rw [this, zero_mul]
  · simp at h

theorem canonF_some (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (ds ps : List Fac)
    (k : CMon) (ds' ps' : List Fac) (h : canonF fl ds ps = some (k, ds', ps')) (μ : ℕ → ℤ) :
    k.den ρ * bodyOf ρ.dv ds' ps' μ = bodyOf ρ.dv ds ps μ := by
  unfold canonF at h
  simp only at h
  split at h
  · simp at h
  · next dsn hn =>
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    have hd := normDels_ok ρ fl hfl (pivPass ds.length [] ds ps).1 μ
    rw [hn] at hd
    rw [← show bodyOf ρ.dv (pivPass ds.length [] ds ps).1 (pivPass ds.length [] ds ps).2 μ =
      bodyOf ρ.dv ds ps μ from by rw [pivPass_ok ρ.dv hρ.dpos]; rfl]
    simp only [bodyOf, delSort_ok, facSort_ok]
    simp only [DelsRes] at hd
    rw [hd, ← phMerge_ok ρ.dv μ (pivPass ds.length [] ds ps).2,
      normPhs_ok ρ hρ (phMerge (pivPass ds.length [] ds ps).2) μ]
    ring

/-! ## Digits of equal values -/

theorem nat_digits_eq (d : ℕ) (hd : 0 < d) (n v w : ℕ) (hv : v < d ^ n) (hw : w < d ^ n) :
    (∀ i < n, v / d ^ i % d = w / d ^ i % d) ↔ v = w := by
  constructor
  · intro h
    induction n generalizing v w with
    | zero => simp at hv hw; omega
    | succ n ih =>
      have h0 := h 0 (by omega)
      simp only [pow_zero, Nat.div_one] at h0
      have hvd : v / d < d ^ n := by rw [Nat.div_lt_iff_lt_mul hd, ← pow_succ]; exact hv
      have hwd : w / d < d ^ n := by rw [Nat.div_lt_iff_lt_mul hd, ← pow_succ]; exact hw
      have e := ih (v / d) (w / d) hvd hwd fun i hi => by
        have := h (i + 1) (by omega)
        rwa [pow_succ', ← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul] at this
      rw [← Nat.mod_add_div v d, ← Nat.mod_add_div w d, h0, e]
  · rintro rfl _ _; rfl

theorem dig_lt (d i : ℕ) (hd : 0 < d) (a : ℤ) : 0 ≤ dig d i a ∧ dig d i a < d :=
  ⟨Int.emod_nonneg _ (by exact_mod_cast hd.ne'), Int.emod_lt_of_pos _ (by exact_mod_cast hd)⟩

/-- Two values reduced modulo `d^n` agree when every digit does. -/
theorem digits_eq_int (d n : ℕ) (hd : 0 < d) (a b : ℤ) :
    (∀ i < n, dig d i (a % ((d ^ n : ℕ) : ℤ)) = dig d i (b % ((d ^ n : ℕ) : ℤ))) ↔
      ((d ^ n : ℕ) : ℤ) ∣ a - b := by
  have hN : (0 : ℤ) < ((d ^ n : ℕ) : ℤ) := by positivity
  obtain ⟨v, hv⟩ : ∃ v : ℕ, a % ((d ^ n : ℕ) : ℤ) = v :=
    ⟨(a % ((d ^ n : ℕ) : ℤ)).toNat, (Int.toNat_of_nonneg (Int.emod_nonneg _ hN.ne')).symm⟩
  obtain ⟨w, hw⟩ : ∃ w : ℕ, b % ((d ^ n : ℕ) : ℤ) = w :=
    ⟨(b % ((d ^ n : ℕ) : ℤ)).toNat, (Int.toNat_of_nonneg (Int.emod_nonneg _ hN.ne')).symm⟩
  have hvl : v < d ^ n := by have := Int.emod_lt_of_pos a hN; omega
  have hwl : w < d ^ n := by have := Int.emod_lt_of_pos b hN; omega
  have e1 : (∀ i < n, dig d i (v : ℤ) = dig d i (w : ℤ)) ↔ v = w := by
    simp only [dig_nat, Nat.cast_inj]; exact nat_digits_eq d hd n v w hvl hwl
  rw [hv, hw, e1, ← Int.modEq_iff_dvd]
  unfold Int.ModEq
  rw [hv, hw]
  constructor <;> intro h <;> omega

theorem small_dvd (d u r : ℤ) (hu : 0 ≤ u ∧ u < d) (hr : 0 ≤ r ∧ r < d) (h : d ∣ u - r) :
    u = r := by
  obtain ⟨k, hk⟩ := h
  rcases lt_trichotomy k 0 with hk0 | rfl | hk0
  · nlinarith
  · simpa [sub_eq_zero] using hk
  · nlinarith

/-! ## Digit conditions read as deltas -/

/-- The digit `-c b` reduced: modulo `d` for a constant `d`. -/
def rOf (c b : ℤ) (d : DMon) : ℤ := if d.isConst then (-c * b) % (d.c : ℤ) else -c * b

/-- The polynomial `P` and modulus `N` of a digit condition with `[N ∣ z] = [N ∣ P]`. -/
def dzPoly (PW : List (DMon × DMon)) (ats : List Atom) (a : Atom) : Option (DMon × IPoly) :=
  if a.f = 3 then
    match a.dp, a.ks, a.args with
    | [d, N], [c], [(A, M1), (Bp, M2)] =>
      match ipConst Bp with
      | some b =>
        if (d, N) ∈ PW ∧ M1 = N ∧ M2 = d ∧ (c = 1 ∨ c = -1) ∧ 0 ≤ rOf c b d ∧
            (rOf c b d ≤ 1 ∨ d.isConst = true) ∧
            plainP ats A = true then some (N, ipSub A (ipC (rOf c b d)))
        else if (d, N) ∈ PW ∧ M1 = N ∧ M2 = d ∧ A = [] ∧ (b = 1 ∨ b = -1) then
          some (N, ipC 1)
        else none
      | none => none
    | [d, N], [c, c'], [(A1, M1), (A2, M2), (Bp, M3)] =>
      if (d, N) ∈ PW ∧ M1 = N ∧ M2 = N ∧ M3 = d ∧ (c = 1 ∨ c = -1) ∧ c' = -c ∧
          ipConst Bp = some 0 ∧ plainP ats A1 = true ∧ plainP ats A2 = true then
        some (N, ipSub A1 A2) else none
    | _, _, _ => none
  else none

theorem dz_digit (d : ℕ) (hd : 2 ≤ d) (c u v : ℤ) (hc : c = 1 ∨ c = -1)
    (hu : 0 ≤ u ∧ u < d) (hv : 0 ≤ v ∧ v < d) (e : ℤ) (he : (d : ℤ) ∣ e + c * v) :
    (c * u + e) % d = 0 ↔ u = v := by
  rw [← Int.dvd_iff_emod_eq_zero]
  constructor
  · intro h
    have h2 : (d : ℤ) ∣ c * (u - v) := by
      have := dvd_sub h he
      rwa [show c * u + e - (e + c * v) = c * (u - v) by ring] at this
    have h3 : (d : ℤ) ∣ u - v := by
      rcases hc with rfl | rfl
      · simpa using h2
      · have : (d : ℤ) ∣ -(-1 * (u - v)) := dvd_neg.2 h2
        simpa using this
    exact small_dvd d u v hu hv h3
  · rintro rfl
    rwa [add_comm]

theorem dig_small (d i N : ℕ) (hd : 0 < d) (r : ℤ) (hr : 0 ≤ r ∧ r < d) (hN : d ≤ N) :
    dig d i (r % (N : ℤ)) = if i = 0 then r else 0 := by
  have e1 : r % (N : ℤ) = r := Int.emod_eq_of_lt hr.1 (by omega)
  rw [e1]
  unfold dig
  split_ifs with hi
  · subst hi; simp [Int.emod_eq_of_lt hr.1 hr.2]
  · have : (d : ℤ) ≤ (d : ℤ) ^ i := by
      have := Nat.le_self_pow hi d
      exact_mod_cast this
    rw [Int.ediv_eq_zero_of_lt hr.1 (by omega)]; simp

theorem bfn3_one (c b' : ℤ) (dd N : ℕ) (aN : ℤ) :
    bfn 3 [c] [dd, N] [aN, b'] =
      ∑ i ∈ pos dd N, ((c * dig dd i aN + if i = 0 then b' else 0) % dd) * (dd : ℤ) ^ i := by
  simp [bfn, lsum]

theorem bfn3_two (c c' b' : ℤ) (dd N : ℕ) (a1 a2 : ℤ) :
    bfn 3 [c, c'] [dd, N] [a1, a2, b'] = ∑ i ∈ pos dd N,
      ((c * dig dd i a1 + (c' * dig dd i a2 + if i = 0 then b' else 0)) % dd) * (dd : ℤ) ^ i := by
  simp only [bfn, lsum, List.zipWith_cons_cons, List.sum_cons, List.getD_cons_zero,
    List.getD_cons_succ, List.length_cons, List.length_nil, List.zipWith_nil_left, List.sum_nil]
  refine Finset.sum_congr rfl fun i _ => by congr 2; ring

/-- The digit conditions, digit by digit. -/
theorem dz_core (dd n : ℕ) (hd2 : 2 ≤ dd) (u v e : ℕ → ℤ) (c : ℤ) (hc : c = 1 ∨ c = -1)
    (hu : ∀ i, 0 ≤ u i ∧ u i < dd) (hv : ∀ i, 0 ≤ v i ∧ v i < dd)
    (he : ∀ i < n, (dd : ℤ) ∣ e i + c * v i) :
    (∀ i ∈ range n, (c * u i + e i) % dd = 0) ↔ (∀ i < n, u i = v i) := by
  constructor
  · intro h i hi
    exact (dz_digit dd hd2 c (u i) (v i) hc (hu i) (hv i) (e i) (he i hi)).1
      (h i (Finset.mem_range.2 hi))
  · intro h i hi
    have hi' := Finset.mem_range.1 hi
    exact (dz_digit dd hd2 c (u i) (v i) hc (hu i) (hv i) (e i) (he i hi')).2 (h i hi')

theorem dzPoly_ok (V : LV) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (ats : List Atom)
    (a : Atom) (N : DMon) (P : IPoly) (h : dzPoly PW ats a = some (N, P)) (μ : ℕ → ℤ) :
    (((N.den V.ρ.dv : ℕ) : ℤ) ∣ aval V μ a) ↔ (((N.den V.ρ.dv : ℕ) : ℤ) ∣ ipDen μ P) := by
  unfold dzPoly at h
  split_ifs at h with hf
  split at h
  · next d N' c A M1 Bp M2 hdp hks hargs =>
    split at h
    · next b hb =>
      split_ifs at h with hc hc'
      swap
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h1, h2⟩ := h
        subst N; subst P
        obtain ⟨hpw, h3, h4, hA, hb1⟩ := hc'
        subst M1; subst M2; subst A
        obtain ⟨hd2, n, hNn⟩ := hPW _ hpw
        have hd2' : 2 ≤ d.den V.ρ.dv := hd2
        have hNn' : N'.den V.ρ.dv = d.den V.ρ.dv ^ n := hNn
        have hdd : 0 < d.den V.ρ.dv := by omega
        simp only [aval, hf, show (3 : ℕ) < 100 by norm_num, if_true, hdp, hks, hargs,
          List.map_cons, List.map_nil, argV, ipConst_spec Bp b hb, ipDen_C, ipDen_nil]
        rw [bfn3_one, hNn', pos_pow _ _ hd2', digits_zero _ n hdd _ fun i => ⟨Int.emod_nonneg _
          (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩]
        have hbd : ¬ (b % (d.den V.ρ.dv : ℤ)) % (d.den V.ρ.dv : ℤ) = 0 := by
          rw [Int.emod_emod_of_dvd _ (dvd_refl _)]
          intro h0
          have := Int.dvd_of_emod_eq_zero h0
          rcases hb1 with rfl | rfl
          · have := Int.le_of_dvd (by norm_num) this; omega
          · have := Int.le_of_dvd (by norm_num) (dvd_neg.2 this); omega
        constructor
        · intro h
          rcases Nat.eq_zero_or_pos n with rfl | hn
          · simp
          · exfalso
            have := h 0 (Finset.mem_range.2 hn)
            simp only [dig, Int.zero_emod, Int.zero_ediv, mul_zero, zero_add, ite_true] at this
            exact hbd this
        · intro h i hi
          have hn : n = 0 := by
            by_contra hn
            have h1 : (1 : ℤ) < ((d.den V.ρ.dv ^ n : ℕ) : ℤ) := by
              have : 1 < d.den V.ρ.dv ^ n := Nat.one_lt_pow hn (by omega)
              exact_mod_cast this
            have := Int.le_of_dvd (by norm_num) h
            omega
          subst hn; simp at hi
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h1, h2⟩ := h
        subst N; subst P
        obtain ⟨hpw, h3, h4, hc1, hr0, hr1, -⟩ := hc
        subst M1; subst M2
        obtain ⟨hd2, n, hNn⟩ := hPW _ hpw
        have hd2' : 2 ≤ d.den V.ρ.dv := hd2
        have hNn' : N'.den V.ρ.dv = d.den V.ρ.dv ^ n := hNn
        have hdd : 0 < d.den V.ρ.dv := by omega
        simp only [aval, hf, show (3 : ℕ) < 100 by norm_num, if_true, hdp, hks, hargs,
          List.map_cons, List.map_nil, argV, ipConst_spec Bp b hb, ipDen_sub, ipDen_C]
        rw [bfn3_one, hNn', pos_pow _ _ hd2', digits_zero _ n hdd _ fun i => ⟨Int.emod_nonneg _
          (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩, ← digits_eq_int _ n hdd]
        refine dz_core _ n hd2' _ _ _ c hc1 (fun i => dig_lt _ i hdd _) (fun i => dig_lt _ i hdd _)
          fun i hi => ?_
        have hn : d.den V.ρ.dv ≤ d.den V.ρ.dv ^ n := Nat.le_self_pow (by omega) _
        have hrl : rOf c b d < (d.den V.ρ.dv : ℤ) := by
          rcases hr1 with hr1 | hr1
          · omega
          · unfold rOf; rw [if_pos hr1, ← DMon.den_isConst V.ρ.dv d hr1]
            exact Int.emod_lt_of_pos _ (by omega)
        rw [dig_small _ i _ hdd (rOf c b d) ⟨hr0, hrl⟩ (by exact_mod_cast hn)]
        split_ifs
        · rw [Int.emod_def]
          unfold rOf
          split_ifs with hk
          · rw [← DMon.den_isConst V.ρ.dv d hk, Int.emod_def]
            rcases hc1 with rfl | rfl
            · exact ⟨-(b / (d.den V.ρ.dv : ℤ)) - (-1 * b / (d.den V.ρ.dv : ℤ)), by ring⟩
            · exact ⟨-(b / (d.den V.ρ.dv : ℤ)) + (-(-1) * b / (d.den V.ρ.dv : ℤ)), by ring⟩
          · rcases hc1 with rfl | rfl
            · exact ⟨-(b / (d.den V.ρ.dv : ℤ)), by ring⟩
            · exact ⟨-(b / (d.den V.ρ.dv : ℤ)), by ring⟩
        · simp
    · simp at h
  · next d N' c c' A1 M1 A2 M2 Bp M3 hdp hks hargs =>
    split_ifs at h with hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨h1, h2⟩ := h
    subst N; subst P
    obtain ⟨hpw, h3, h4, h5, hc1, h6, hb, -, -⟩ := hc
    subst M1; subst M2; subst M3; subst c'
    obtain ⟨hd2, n, hNn⟩ := hPW _ hpw
    have hd2' : 2 ≤ d.den V.ρ.dv := hd2
    have hNn' : N'.den V.ρ.dv = d.den V.ρ.dv ^ n := hNn
    have hdd : 0 < d.den V.ρ.dv := by omega
    simp only [aval, hf, show (3 : ℕ) < 100 by norm_num, if_true, hdp, hks, hargs,
      List.map_cons, List.map_nil, argV, ipConst_spec Bp 0 hb, ipDen_sub, Int.zero_emod]
    rw [bfn3_two, hNn', pos_pow _ _ hd2', digits_zero _ n hdd _ fun i => ⟨Int.emod_nonneg _
      (by positivity), Int.emod_lt_of_pos _ (by positivity)⟩, ← digits_eq_int _ n hdd]
    refine dz_core _ n hd2' _ _ _ c hc1 (fun i => dig_lt _ i hdd _) (fun i => dig_lt _ i hdd _)
      fun i hi => ⟨0, by split_ifs <;> ring⟩
  · simp at h

/-! ## Atom rules -/

/-- Same coefficient and sums; the summands agree pointwise. -/
def PWEq (V : LV) (t t' : ATm) : Prop :=
  t'.c = t.c ∧ t'.sums = t.sums ∧ ∀ μ, abody V t' μ = abody V t μ

theorem PWEq.den {V : LV} {t t' : ATm} (h : PWEq V t t') (ν : ℕ → ℤ) : t'.den V ν = t.den V ν :=
  den_of_pw V t t' h.2.1 (fun μ => by rw [h.1, h.2.2]) ν

theorem PWEq.trans {V : LV} {a b c : ATm} (h1 : PWEq V a b) (h2 : PWEq V b c) : PWEq V a c :=
  ⟨h2.1.trans h1.1, h2.2.1.trans h1.2.1, fun μ => (h2.2.2 μ).trans (h1.2.2 μ)⟩

theorem aext_find (V : LV) (ats : List Atom) (z : ℕ) (a : Atom) (hnd : (ids ats).Nodup)
    (h : ats.find? (fun b => b.id == z) = some a) (μ : ℕ → ℤ) :
    ∃ μ', aext V ats μ z = aval V μ' a ∧ ∀ y, y ∉ ids ats → μ' y = μ y := by
  induction ats generalizing μ with
  | nil => simp at h
  | cons b l ih =>
    simp only [ids, List.map_cons, List.nodup_cons] at hnd
    simp only [List.find?_cons] at h
    by_cases hb : (b.id == z) = true
    · rw [hb] at h
      simp only [Option.some.injEq] at h
      subst h
      simp only [beq_iff_eq] at hb
      subst hb
      refine ⟨μ, ?_, fun _ _ => rfl⟩
      simp only [aext]; rw [aext_off V l _ _ hnd.1, Function.update_self]
    · simp only [Bool.not_eq_true] at hb
      rw [hb] at h
      obtain ⟨μ', h1, h2⟩ := ih hnd.2 h (Function.update μ b.id (aval V μ b))
      refine ⟨μ', h1, fun y hy => ?_⟩
      simp only [ids, List.map_cons, List.mem_cons, not_or] at hy
      rw [h2 y hy.2, Function.update_of_ne hy.1]

/-- A delta reading a digit condition, read as the delta it is equivalent to. -/
def dzFind (PW : List (DMon × DMon)) (ats : List Atom) (f : Fac) : Option Fac :=
  match f.2 with
  | [([z], 1)] =>
    match ats.find? (fun b => b.id == z) with
    | some a =>
      match dzPoly PW ats a with
      | some (N, P) => if f.1 = N ∧ plainP ats P = true then some (N, P) else none
      | none => none
    | none => none
  | _ => none

/-- Every delta reading a digit condition rewritten. -/
def dzRule (PW : List (DMon × DMon)) (ats : List Atom) (ds : List Fac) : List Fac :=
  ds.map fun f => (dzFind PW ats f).getD f

theorem dzRule_ok (V : LV) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (ats : List Atom)
    (hnd : (ids ats).Nodup) (ds : List Fac) (μ : ℕ → ℤ) :
    ((dzRule PW ats ds).map (delV V.ρ.dv (aext V ats μ))).prod =
      (ds.map (delV V.ρ.dv (aext V ats μ))).prod := by
  unfold dzRule
  rw [List.map_map]
  congr 1
  refine List.map_congr_left fun f _ => ?_
  simp only [Function.comp_apply]
  unfold dzFind
  split
  · next z hz =>
    split
    · next a ha =>
      split
      · next N P hP =>
        split_ifs with hc
        · simp only [Option.getD_some]
          obtain ⟨μ', h1, h2⟩ := aext_find V ats z a hnd ha μ
          have hPμ : ipDen μ' P = ipDen (aext V ats μ) P := by
            rw [ipDen_aext V ats P hc.2]
            refine ipDen_reads P _ _ fun y hy => h2 y ?_
            have := hc.2
            simp only [plainP, List.all_eq_true, Bool.not_eq_true', List.contains_eq_mem,
              decide_eq_false_iff_not] at this
            exact this y hy
          unfold delV
          simp only [hz, hc.1]
          have : ipDen (aext V ats μ) [([z], 1)] = aext V ats μ z := by
            simp [ipDen, imDen]
          rw [this, h1, ← hPμ]
          by_cases hd : (((N.den V.ρ.dv : ℕ) : ℤ) ∣ ipDen μ' P)
          · rw [if_pos hd, if_pos ((dzPoly_ok V PW hPW ats a N P hP μ').2 hd)]
          · rw [if_neg hd, if_neg (fun h' => hd ((dzPoly_ok V PW hPW ats a N P hP μ').1 h'))]
        · rfl
      · rfl
    · rfl
  · rfl

/-- A digit product with a zero argument. -/
def isZeroDig (a : Atom) : Bool :=
  a.f == 0 && match a.args with
    | [(p, _), (q, _)] => p.isEmpty || q.isEmpty
    | _ => false

theorem aval_zeroDig (V : LV) (μ : ℕ → ℤ) (a : Atom) (h : isZeroDig a = true) :
    aval V μ a = 0 := by
  unfold isZeroDig at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨hf, h⟩ := h
  split at h
  · next p M1 q M2 hargs =>
    simp only [aval, hf, show (0 : ℕ) < 100 by norm_num, if_true, bfn, hargs, List.map_cons,
      List.map_nil, argV, List.getD_cons_zero, List.getD_cons_succ]
    simp only [Bool.or_eq_true, List.isEmpty_iff] at h
    rcases h with rfl | rfl <;> simp [ipDen, dig]
  · simp at h

/-- A zero digit product at the split `p`, substituted by `0`. -/
def zeroAt (t : ATm) (p : List Atom × Atom × List Atom) : Option ATm :=
  if isZeroDig p.2.1 && !(ids p.2.2).contains p.2.1.id then
    some ⟨t.c, t.sums, p.1 ++ p.2.2.map (atSub p.2.1.id []), t.dels.map (facSub p.2.1.id []),
      t.phs.map (facSub p.2.1.id [])⟩
  else none

theorem zeroAt_ok (V : LV) (t t' : ATm) (p : List Atom × Atom × List Atom)
    (hp : t.ats = p.1 ++ p.2.1 :: p.2.2) (h : zeroAt t p = some t') : PWEq V t t' := by
  unfold zeroAt at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem,
    decide_eq_false_iff_not] at hc
  refine ⟨rfl, rfl, fun μ => ?_⟩
  unfold abody
  simp only
  rw [hp, aext_append, aext_append]
  simp only [aext]
  rw [aval_zeroDig V _ _ hc.1]
  have e := aext_sub V p.2.2 p.2.1.id hc.2 [] (by simp [plainP, ipVs]) rfl (aext V p.1 μ)
  simp only [ipDen_nil] at e
  rw [e, bodyOf_sub]
  rfl

/-- An atom nothing reads, at the split `p`, removed. -/
def dropAt (t : ATm) (p : List Atom × Atom × List Atom) : Option ATm :=
  if !(fvs t).contains p.2.1.id && !(argVs p.2.2).contains p.2.1.id &&
      !(ids p.2.2).contains p.2.1.id then
    some ⟨t.c, t.sums, p.1 ++ p.2.2, t.dels, t.phs⟩
  else none

theorem dropAt_ok (V : LV) (t t' : ATm) (p : List Atom × Atom × List Atom)
    (hp : t.ats = p.1 ++ p.2.1 :: p.2.2) (h : dropAt t p = some t') : PWEq V t t' := by
  unfold dropAt at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem,
    decide_eq_false_iff_not] at hc
  obtain ⟨⟨h1, h2⟩, h3⟩ := hc
  refine ⟨rfl, rfl, fun μ => ?_⟩
  unfold abody
  simp only
  rw [hp, aext_append, aext_append]
  simp only [aext]
  rw [aext_upd V p.2.2 p.2.1.id (by simp only [List.mem_append, not_or]; exact ⟨h2, h3⟩)]
  refine bodyOf_reads _ _ _ _ _ fun y hy => ?_
  have hy' : y ≠ p.2.1.id := fun e => h1 (e ▸ (by simpa [fvs] using hy))
  rw [Function.update_of_ne hy']

theorem find_id (l : List Atom) (hnd : (ids l).Nodup) (b : Atom) (hb : b ∈ l) :
    l.find? (fun a => a.id == b.id) = some b := by
  induction l with
  | nil => simp at hb
  | cons a l ih =>
    simp only [ids, List.map_cons, List.nodup_cons] at hnd
    rw [List.find?_cons]
    rcases List.mem_cons.1 hb with rfl | hb'
    · simp
    · have hne : a.id ≠ b.id := fun e => hnd.1 (e ▸ List.mem_map_of_mem hb')
      simp only [show (a.id == b.id) = false from by simpa using hne]
      exact ih hnd.2 hb'

/-- Whether two atoms compute the same function of the same arguments. -/
def sameAt (a b : Atom) : Bool := a.f == b.f && a.ks == b.ks && a.dp == b.dp && a.args == b.args

/-- The atom at the split `sp` read as an earlier atom computing the same value. -/
def dedupAt (t : ATm) (sp : List Atom × Atom × List Atom) : Option ATm :=
  match sp.1.find? (fun a => a.id != sp.2.1.id && sameAt a sp.2.1) with
  | some a =>
    if (ids t.ats).Nodup && sp.2.1.args.all (fun p => plainP t.ats p.1) &&
        !(argVs t.ats).contains sp.2.1.id then
      some ⟨t.c, t.sums, t.ats, t.dels.map (facSub sp.2.1.id (ipV a.id)),
        t.phs.map (facSub sp.2.1.id (ipV a.id))⟩
    else none
  | none => none

theorem dedupAt_ok (V : LV) (t t' : ATm) (sp : List Atom × Atom × List Atom)
    (hsp : t.ats = sp.1 ++ sp.2.1 :: sp.2.2) (h : dedupAt t sp = some t') : PWEq V t t' := by
  unfold dedupAt at h
  split at h
  · next a ha =>
    split_ifs at h with hc
    simp only [Option.some.injEq] at h
    subst h
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not, List.all_eq_true] at hc
    obtain ⟨⟨hnd, hpl⟩, hna⟩ := hc
    have hnd : (ids t.ats).Nodup := of_decide_eq_true hnd
    obtain ⟨ha1, ha2⟩ := List.find?_some ha |> fun h' => (⟨h', List.mem_of_find?_eq_some ha⟩ :
      _ ∧ a ∈ sp.1)
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, sameAt, beq_iff_eq] at ha1
    obtain ⟨hid, ⟨⟨⟨hf, hks⟩, hdp⟩, hargs⟩⟩ := ha1
    refine ⟨rfl, rfl, fun μ => ?_⟩
    unfold abody
    simp only
    set E := aext V t.ats μ
    have hbm : sp.2.1 ∈ t.ats := by rw [hsp]; simp
    have ham : a ∈ t.ats := by rw [hsp]; exact List.mem_append_left _ ha2
    obtain ⟨μb, hb1, hb2⟩ := aext_find V t.ats sp.2.1.id sp.2.1 hnd (find_id _ hnd _ hbm) μ
    obtain ⟨μa, ha3, ha4⟩ := aext_find V t.ats a.id a hnd (find_id _ hnd _ ham) μ
    have hval : E sp.2.1.id = E a.id := by
      show aext V t.ats μ _ = aext V t.ats μ _
      rw [hb1, ha3]
      have : aval V μb sp.2.1 = aval V μa sp.2.1 := by
        refine aval_reads V _ _ _ fun y hy => ?_
        simp only [atVs, List.mem_flatMap] at hy
        obtain ⟨p, hp, hy⟩ := hy
        have := hpl p hp
        simp only [plainP, List.all_eq_true, Bool.not_eq_true', List.contains_eq_mem,
          decide_eq_false_iff_not] at this
        rw [hb2 y (this y hy), ha4 y (this y hy)]
      rw [this]
      unfold aval
      rw [hf, hks, hdp, hargs]
    rw [bodyOf_sub, ipDen_V, ← hval, Function.update_eq_self]
  · simp at h

/-- The pairs of coefficients and arguments of a digit condition without the zero arguments. -/
def nzPairs (ks : List ℤ) (args : List (IPoly × DMon)) : List (ℤ × (IPoly × DMon)) :=
  (List.zip ks args).filter fun p => !p.2.1.isEmpty

theorem lsum_nz (ks : List ℤ) (args : List (IPoly × DMon)) (g : ℤ → ℤ) (h0 : g 0 = 0)
    (f : IPoly × DMon → ℤ) (hf : ∀ M, f ([], M) = 0) :
    lsum ((nzPairs ks args).map (·.1)) ((nzPairs ks args).map (f ·.2)) g =
      lsum ks (args.map f) g := by
  induction ks generalizing args with
  | nil => simp [nzPairs, lsum]
  | cons c ks ih =>
    cases args with
    | nil => simp [nzPairs, lsum]
    | cons p args =>
      have ih' := ih args
      simp only [nzPairs, List.zip_cons_cons, List.filter_cons] at ih' ⊢
      by_cases hp : p.1.isEmpty = true
      · obtain ⟨q, M⟩ := p
        simp only at hp
        rw [List.isEmpty_iff] at hp
        subst hp
        simp only [List.isEmpty_nil, Bool.not_true, Bool.false_eq_true, ite_false, ih']
        simp [lsum, hf, h0]
      · simp only [Bool.not_eq_true] at hp
        simp only [hp, Bool.not_false, ite_true, List.map_cons]
        simp only [lsum, List.zipWith_cons_cons, List.sum_cons] at ih' ⊢
        rw [ih']

/-- A digit condition without its zero arguments. -/
def dzDrop (a : Atom) : Option Atom :=
  if a.f = 3 ∧ a.args.length = a.ks.length + 1 ∧
      ((List.zip a.ks a.args).any fun p => p.2.1.isEmpty) = true ∧ nzPairs a.ks a.args ≠ [] then
    some { a with ks := (nzPairs a.ks a.args).map (·.1),
                  args := (nzPairs a.ks a.args).map (·.2) ++ a.args.drop a.ks.length }
  else none

theorem dzDrop_ok (V : LV) (a a' : Atom) (h : dzDrop a = some a') (μ : ℕ → ℤ) :
    aval V μ a' = aval V μ a := by
  unfold dzDrop at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  obtain ⟨hf, hlen, -, -⟩ := hc
  obtain ⟨id, f, ks, dp, args⟩ := a
  simp only at hf hlen ⊢
  subst hf
  have hl : (nzPairs ks args).length ≤ ks.length := by
    unfold nzPairs; exact (List.length_filter_le _ _).trans (by simp)
  simp only [aval, show (3 : ℕ) < 100 by norm_num, if_true, bfn]
  refine Finset.sum_congr rfl fun i _ => ?_
  congr 2
  rw [List.map_append, lsum_append _ _ _ _ (by simp), List.map_map]
  have e1 := lsum_nz ks args (dig (dp.map (·.den V.ρ.dv) |>.getD 0 2) i) (by simp [dig])
    (argV V.ρ.dv μ) (fun M => by simp [argV, ipDen])
  simp only [Function.comp_def] at e1 ⊢
  rw [e1]
  have e2 : (args.map (argV V.ρ.dv μ)).getD ks.length 0 =
      (((nzPairs ks args).map fun p => argV V.ρ.dv μ p.2) ++
        (args.drop ks.length).map (argV V.ρ.dv μ)).getD (nzPairs ks args).length 0 := by
    rw [List.getD_append_right _ _ _ _ (by simp), List.length_map, Nat.sub_self]
    obtain ⟨B, hB⟩ : ∃ B, args.drop ks.length = [B] := by
      have : (args.drop ks.length).length = 1 := by simp [hlen]
      match h' : args.drop ks.length, this with
      | [B], _ => exact ⟨B, rfl⟩
    rw [hB]
    simp only [List.map_cons, List.map_nil, List.getD_cons_zero]
    rw [List.getD_eq_getElem?_getD, List.getElem?_map]
    rw [show args[ks.length]? = some B by rw [← List.head?_drop, hB]; rfl]
    rfl
  rw [e2]
  simp only [List.length_map]

/-- The digit condition at the split `sp` without its zero arguments. -/
def dzDropAt (t : ATm) (sp : List Atom × Atom × List Atom) : Option ATm :=
  match dzDrop sp.2.1 with
  | some a' => some ⟨t.c, t.sums, sp.1 ++ a' :: sp.2.2, t.dels, t.phs⟩
  | none => none

theorem dzDropAt_ok (V : LV) (t t' : ATm) (sp : List Atom × Atom × List Atom)
    (hsp : t.ats = sp.1 ++ sp.2.1 :: sp.2.2) (h : dzDropAt t sp = some t') : PWEq V t t' := by
  unfold dzDropAt at h
  split at h
  · next a' ha =>
    simp only [Option.some.injEq] at h
    subst h
    refine ⟨rfl, rfl, fun μ => ?_⟩
    unfold abody
    simp only
    rw [hsp, aext_append, aext_append]
    simp only [aext]
    have hid : a'.id = sp.2.1.id := by
      unfold dzDrop at ha; split_ifs at ha; simp only [Option.some.injEq] at ha; rw [← ha]
    rw [hid, dzDrop_ok V _ _ ha]
  · simp at h

/-- One atom rule, the first that applies. -/
def aRule1 (t : ATm) : Option ATm :=
  match firstSome (zeroAt t) (splits t.ats) with
  | some t' => some t'
  | none =>
    match firstSome (dropAt t) (splits t.ats) with
    | some t' => some t'
    | none =>
      match firstSome (dedupAt t) (splits t.ats) with
      | some t' => some t'
      | none => firstSome (dzDropAt t) (splits t.ats)

theorem aRule1_ok (V : LV) (t t' : ATm) (h : aRule1 t = some t') : PWEq V t t' := by
  unfold aRule1 at h
  split at h
  · next t'' h' =>
    simp only [Option.some.injEq] at h
    subst h
    obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h'
    exact zeroAt_ok V t _ p (splits_spec _ p hp) he
  · split at h
    · next t'' h' =>
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h'
      exact dropAt_ok V t _ p (splits_spec _ p hp) he
    · split at h
      · next t'' h' =>
        simp only [Option.some.injEq] at h
        subst h
        obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h'
        exact dedupAt_ok V t _ p (splits_spec _ p hp) he
      · obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
        exact dzDropAt_ok V t _ p (splits_spec _ p hp) he

/-- Atom rules repeated, at most `n` of them. -/
def aRules : ℕ → ATm → ATm
  | 0, t => t
  | n + 1, t => match aRule1 t with
    | some t' => aRules n t'
    | none => t

theorem aRules_ok (V : LV) (n : ℕ) (t : ATm) : PWEq V t (aRules n t) := by
  induction n generalizing t with
  | zero => exact ⟨rfl, rfl, fun _ => rfl⟩
  | succ n ih =>
    simp only [aRules]
    split
    · next t' h => exact (aRule1_ok V t t' h).trans (ih t')
    · exact ⟨rfl, rfl, fun _ => rfl⟩

/-! ## Free variables -/

/-- Each atom's arguments read no atom of its own or a later position. -/
def wfA : List Atom → Bool
  | [] => true
  | a :: l => (atVs a).all (fun v => !(ids (a :: l)).contains v) && wfA l

theorem aext_agreeT (V : LV) (l : List Atom) (hl : wfA l = true) (T : List ℕ)
    (hT : ∀ v ∈ argVs l, v ∈ T ∨ v ∈ ids l) (μ μ' : ℕ → ℤ) (h : ∀ v ∈ T, μ v = μ' v) :
    ∀ v, v ∈ T ∨ v ∈ ids l → aext V l μ v = aext V l μ' v := by
  induction l generalizing T μ μ' with
  | nil => intro v hv; simp [ids] at hv; exact h v hv
  | cons a l ih =>
    simp only [wfA, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_true',
      List.contains_eq_mem, decide_eq_false_iff_not] at hl
    have ha : ∀ v ∈ atVs a, v ∈ T := fun v hv => by
      rcases hT v (by simp only [argVs, List.flatMap_cons, List.mem_append]; exact Or.inl hv)
        with h' | h'
      · exact h'
      · exact absurd h' (hl.1 v hv)
    have hv : aval V μ a = aval V μ' a := aval_reads V a _ _ fun v hv' => h v (ha v hv')
    simp only [aext]
    intro v hvT
    refine ih hl.2 (a.id :: T) (fun w hw => ?_) _ _ (fun w hw => ?_) v ?_
    · rcases hT w (by simp only [argVs, List.flatMap_cons, List.mem_append]; exact Or.inr hw)
        with h' | h'
      · exact Or.inl (List.mem_cons_of_mem _ h')
      · simp only [ids, List.map_cons, List.mem_cons] at h'
        rcases h' with h' | h'
        · exact Or.inl (h' ▸ List.mem_cons_self)
        · exact Or.inr h'
    · rcases List.mem_cons.1 hw with rfl | hw
      · simp [hv]
      · by_cases e : w = a.id
        · subst e; simp [hv]
        · rw [Function.update_of_ne e, Function.update_of_ne e]; exact h w hw
    · simp only [ids, List.map_cons, List.mem_cons] at hvT
      rcases hvT with hvT | hvT | hvT
      · exact Or.inl (List.mem_cons_of_mem _ hvT)
      · exact Or.inl (hvT ▸ List.mem_cons_self)
      · exact Or.inr hvT

/-- The variables a term reads that it neither binds nor computes. -/
def freeV (t : ATm) : List ℕ :=
  (fvs t ++ argVs t.ats).filter fun v => !(ids t.ats).contains v && !(abvs t).contains v

theorem sumsV_reads_free (dv : ℕ → ℕ) (S : List ℕ) (f : (ℕ → ℤ) → ℂ) (hf : RD S f)
    (s : List (ℕ × DMon)) (ν ν' : ℕ → ℤ)
    (h : ∀ v ∈ S, v ∉ s.map (·.1) → ν v = ν' v) : sumsV dv s f ν = sumsV dv s f ν' := by
  induction s generalizing ν ν' with
  | nil => exact hf _ _ fun v hv => h v hv (by simp)
  | cons p s ih =>
    simp only [sumsV]
    refine Finset.sum_congr rfl fun j _ => ih _ _ fun v hv hvs => ?_
    by_cases e : v = p.1
    · subst e; simp
    · rw [Function.update_of_ne e, Function.update_of_ne e]
      exact h v hv (by simp only [List.map_cons, List.mem_cons, not_or]; exact ⟨e, hvs⟩)

theorem den_reads_free (V : LV) (t : ATm) (hw : wfA t.ats = true) (ν ν' : ℕ → ℤ)
    (h : ∀ v ∈ freeV t, ν v = ν' v) : t.den V ν = t.den V ν' := by
  unfold ATm.den
  congr 1
  set T := (fvs t ++ argVs t.ats).filter fun v => !(ids t.ats).contains v
  have hbody : RD T (abody V t) := by
    intro μ μ' hμ
    have ha := aext_agreeT V t.ats hw T (fun v hv => by
      by_cases hi : v ∈ ids t.ats
      · exact Or.inr hi
      · exact Or.inl (List.mem_filter.2 ⟨by simp [hv], by simpa using hi⟩)) μ μ' hμ
    refine bodyOf_reads _ _ _ _ _ fun v hv => ha v ?_
    by_cases hi : v ∈ ids t.ats
    · exact Or.inr hi
    · exact Or.inl (List.mem_filter.2 ⟨List.mem_append_left _ hv, by simpa using hi⟩)
  refine sumsV_reads_free _ T _ hbody _ _ _ fun v hv hb => h v ?_
  simp only [T, List.mem_filter] at hv
  refine List.mem_filter.2 ⟨hv.1, ?_⟩
  simp only [Bool.and_eq_true, Bool.not_eq_true'] at hv ⊢
  exact ⟨hv.2, by simpa [abvs] using hb⟩

/-! ## Canonical names -/

/-- The first occurrences of a list's elements. -/
def uniq (l : List ℕ) : List ℕ := l.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []

/-- Bound variables renamed `RBASE + k` and atoms `ABASE + k` in order of first occurrence. -/
def cmap (bl al : List ℕ) (v : ℕ) : ℕ :=
  if bl.contains v then RBASE + bl.idxOf v else if al.contains v then ABASE + al.idxOf v else v

/-- The term renamed canonically, when the renaming is injective. -/
def canonRen (t : ATm) : ATm :=
  let occ := uniq (fvs t ++ argVs t.ats ++ ids t.ats ++ abvs t)
  let bl := occ.filter fun v => (abvs t).contains v
  let al := occ.filter fun v => (ids t.ats).contains v && !(abvs t).contains v
  if injOn (cmap bl al) (allV t) && wfA t.ats then tRen (cmap bl al) t else t

theorem canonRen_ok (V : LV) (t : ATm) (ν : ℕ → ℤ) : (canonRen t).den V ν = t.den V ν := by
  unfold canonRen
  simp only
  split_ifs with hc
  · simp only [Bool.and_eq_true] at hc
    rw [tRen_den V _ t hc.1]
    refine den_reads_free V t hc.2 _ _ fun v hv => ?_
    simp only [freeV, List.mem_filter, Bool.and_eq_true, Bool.not_eq_true',
      List.contains_eq_mem, decide_eq_false_iff_not] at hv
    simp only [Function.comp_apply, cmap, List.contains_eq_mem, List.mem_filter,
      decide_eq_true_eq, Bool.and_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
    rw [if_neg (fun h' => hv.2.2 h'.2), if_neg (fun h' => hv.2.1 h'.2.1)]
  · rfl

/-! ## Sorting the sums -/

/-- Sorted insertion of a sum by its variable. -/
def sIns (p : ℕ × DMon) : List (ℕ × DMon) → List (ℕ × DMon)
  | [] => [p]
  | q :: l => if p.1 ≤ q.1 then p :: q :: l else q :: sIns p l

/-- The sums sorted by variable. -/
def sSort (l : List (ℕ × DMon)) : List (ℕ × DMon) := l.foldr sIns []

theorem sIns_map (p : ℕ × DMon) (l : List (ℕ × DMon)) :
    ∀ v, v ∈ (sIns p l).map (·.1) ↔ v = p.1 ∨ v ∈ l.map (·.1) := by
  induction l with
  | nil => simp [sIns]
  | cons q l ih =>
    intro v
    simp only [sIns]
    split_ifs
    · simp
    · simp only [List.map_cons, List.mem_cons, ih]; tauto

theorem sumsV_sIns (dv : ℕ → ℕ) (p : ℕ × DMon) (l : List (ℕ × DMon)) (hp : p.1 ∉ l.map (·.1))
    (f : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) : sumsV dv (sIns p l) f ν = sumsV dv (p :: l) f ν := by
  induction l generalizing ν with
  | nil => rfl
  | cons q l ih =>
    simp only [List.map_cons, List.mem_cons, not_or] at hp
    simp only [sIns]
    split_ifs
    · rfl
    · simp only [sumsV]
      rw [Finset.sum_congr rfl fun j _ => ih hp.2 _]
      simp only [sumsV]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      rw [Function.update_comm hp.1]

theorem sSort_map (l : List (ℕ × DMon)) : ∀ v, v ∈ (sSort l).map (·.1) ↔ v ∈ l.map (·.1) := by
  induction l with
  | nil => simp [sSort]
  | cons q l ih =>
    intro v
    simp only [sSort, List.foldr_cons] at ih ⊢
    rw [sIns_map, ih]; simp

theorem sumsV_sSort (dv : ℕ → ℕ) (l : List (ℕ × DMon)) (hl : (l.map (·.1)).Nodup)
    (f : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) : sumsV dv (sSort l) f ν = sumsV dv l f ν := by
  induction l generalizing ν with
  | nil => rfl
  | cons p l ih =>
    simp only [List.map_cons, List.nodup_cons] at hl
    show sumsV dv (sIns p (sSort l)) f ν = _
    rw [sumsV_sIns _ _ _ (fun h => hl.1 ((sSort_map l _).1 h))]
    simp only [sumsV]
    exact Finset.sum_congr rfl fun j _ => ih hl.2 _

/-- The sums sorted, when their variables are distinct. -/
def sortSums (t : ATm) : ATm :=
  if (abvs t).Nodup then ⟨t.c, sSort t.sums, t.ats, t.dels, t.phs⟩ else t

theorem sortSums_ok (V : LV) (t : ATm) (ν : ℕ → ℤ) : (sortSums t).den V ν = t.den V ν := by
  unfold sortSums
  split_ifs with h
  · unfold ATm.den; simp only
    rw [show abody V ⟨t.c, sSort t.sums, t.ats, t.dels, t.phs⟩ = abody V t from rfl,
      sumsV_sSort _ _ h]
  · rfl

/-! ## The normaliser -/

/-- Eliminations repeated, at most `n` of them, the atom rules applied before each. -/
@[irreducible] def aElimAll (PW : List (DMon × DMon)) : ℕ → ATm → ATm
  | 0, t => t
  | n + 1, t => match aElimStep PW (aRules (t.ats.length + 1) t) with
    | some t' => aElimAll PW n t'
    | none => t

theorem aElimAll_ok (V : LV) (hρ : V.ρ.OK) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ)
    (n : ℕ) (t : ATm) (ν : ℕ → ℤ) : (aElimAll PW n t).den V ν = t.den V ν := by
  induction n generalizing t with
  | zero => simp only [aElimAll]
  | succ n ih =>
    simp only [aElimAll]
    split
    · next t' h =>
      rw [ih, aElimStep_ok V hρ PW hPW _ t' h, (aRules_ok V (t.ats.length + 1) t).den]
    · rfl

/-- Deltas reading digit conditions rewritten, when the table's variables are distinct. -/
def dzStep (PW : List (DMon × DMon)) (t : ATm) : ATm :=
  if (ids t.ats).Nodup then ⟨t.c, t.sums, t.ats, dzRule PW t.ats t.dels, t.phs⟩ else t

theorem dzStep_ok (V : LV) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (t : ATm) :
    PWEq V t (dzStep PW t) := by
  unfold dzStep
  split_ifs with hn
  · refine ⟨rfl, rfl, fun μ => ?_⟩
    unfold abody bodyOf
    simp only
    rw [dzRule_ok V PW hPW t.ats hn]
  · exact ⟨rfl, rfl, fun _ => rfl⟩

/-! ## Reduced arguments read under a divisor -/

/-- `K ∣ M` syntactically, or `M = K^n` with a floor of at least `2`. -/
def dvdP (fl : List ℕ) (PW : List (DMon × DMon)) (K M : DMon) : Bool :=
  K.dvd M || (PW.contains (K, M) && decide (2 ≤ M.den (flv fl)))

theorem dvdP_ok (ρ : Val) (fl : List ℕ) (hfl : FloorOK fl ρ) (PW : List (DMon × DMon))
    (hPW : PowOK PW ρ) (K M : DMon) (h : dvdP fl PW K M = true) :
    K.den ρ.dv ∣ M.den ρ.dv := by
  unfold dvdP at h
  simp only [Bool.or_eq_true, Bool.and_eq_true, List.contains_iff_mem, decide_eq_true_eq] at h
  rcases h with h | ⟨hm, h2⟩
  · exact DMon.dvd_den _ _ _ h
  · obtain ⟨-, n, hn⟩ := hPW _ hm
    have h3 : M.den (flv fl) ≤ M.den ρ.dv := DMon.den_mono _ _ hfl M
    simp only at hn
    rcases n with _ | n
    · rw [pow_zero] at hn; omega
    · rw [hn]; exact dvd_pow_self _ (Nat.succ_ne_zero n)

theorem fac_congr' (dv : ℕ → ℕ) (x : ℕ) (f : Fac) (ν : ℕ → ℤ) (a b : ℤ)
    (hab : ((f.1.den dv : ℕ) : ℤ) ∣ a - b) (hpos : 0 < f.1.den dv) :
    delV dv (Function.update ν x a) f = delV dv (Function.update ν x b) f ∧
      phV dv (Function.update ν x a) f = phV dv (Function.update ν x b) f := by
  have hN := hab.trans (ipDen_upd_dvd ν x a b f.2)
  constructor
  · unfold delV
    have : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen (Function.update ν x a) f.2 ↔
        ((f.1.den dv : ℕ) : ℤ) ∣ ipDen (Function.update ν x b) f.2 := by
      constructor
      · intro h; have := dvd_sub h hN; simpa using this
      · intro h; have := dvd_add h hN; simpa using this
    simp only [this]
  · unfold phV
    obtain ⟨k, hk⟩ := hN
    have hz := om_ne_zero _ hpos
    rw [show ipDen (Function.update ν x a) f.2 = ipDen (Function.update ν x b) f.2 +
      (f.1.den dv : ℤ) * k by omega, zpow_add₀ hz, zpow_mul, zpow_natCast, om_pow_self _ hpos,
      one_zpow, mul_one]

/-- A reduced argument `mod(q, M)` at the split `sp`, read only by factors whose modulus divides
`M`, replaced by `q` there. -/
def modDropAt (fl : List ℕ) (PW : List (DMon × DMon)) (t : ATm)
    (sp : List Atom × Atom × List Atom) : Option ATm :=
  match sp.2.1.f, sp.2.1.args with
  | 2, [(q, M)] =>
    if (ipVs q).all (fun y => y != sp.2.1.id && !(ids sp.2.2).contains y) &&
        !(argVs t.ats).contains sp.2.1.id && !(ids sp.2.2).contains sp.2.1.id &&
        (t.dels ++ t.phs).all (fun g => !facHas sp.2.1.id g || (g.1.c != 0 && dvdP fl PW g.1 M)) then
      some ⟨t.c, t.sums, t.ats, t.dels.map (facSub sp.2.1.id q), t.phs.map (facSub sp.2.1.id q)⟩
    else none
  | _, _ => none

theorem modDropAt_ok (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (t t' : ATm)
    (sp : List Atom × Atom × List Atom) (hsp : t.ats = sp.1 ++ sp.2.1 :: sp.2.2)
    (h : modDropAt fl PW t sp = some t') : PWEq V t t' := by
  unfold modDropAt at h
  split at h
  · next q M hf hargs =>
    split_ifs at h with hc
    simp only [Option.some.injEq] at h
    subst h
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not, List.all_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq] at hc
    obtain ⟨⟨⟨hq, hna⟩, hnp⟩, hfac⟩ := hc
    refine ⟨rfl, rfl, fun μ => ?_⟩
    unfold abody
    simp only
    set E := aext V t.ats μ
    set P := aext V sp.1 μ
    have hE : E = aext V sp.2.2 (Function.update P sp.2.1.id (aval V P sp.2.1)) := by
      show aext V t.ats μ = _
      rw [hsp, aext_append]; rfl
    have hqE : ipDen E q = ipDen P q := by
      rw [hE]
      refine ipDen_reads q _ _ fun y hy => ?_
      rw [aext_off V sp.2.2 _ y (hq y hy).2, Function.update_of_ne (hq y hy).1]
    have hEa : E sp.2.1.id = ipDen P q % ((M.den V.ρ.dv : ℕ) : ℤ) := by
      rw [hE, aext_off V sp.2.2 _ _ hnp, Function.update_self]
      simp only [aval, hf, show (2 : ℕ) < 100 by norm_num, if_true, bfn, hargs, List.map_cons,
        List.map_nil, argV, List.getD_cons_zero]
    rw [bodyOf_sub, hqE]
    have hE2 : E = Function.update E sp.2.1.id (ipDen P q % ((M.den V.ρ.dv : ℕ) : ℤ)) := by
      rw [← hEa, Function.update_eq_self]
    conv_rhs => rw [hE2]
    have hcong : ∀ g ∈ t.dels ++ t.phs,
        delV V.ρ.dv (Function.update E sp.2.1.id (ipDen P q)) g =
          delV V.ρ.dv (Function.update E sp.2.1.id (ipDen P q % ((M.den V.ρ.dv : ℕ) : ℤ))) g ∧
        phV V.ρ.dv (Function.update E sp.2.1.id (ipDen P q)) g =
          phV V.ρ.dv (Function.update E sp.2.1.id (ipDen P q % ((M.den V.ρ.dv : ℕ) : ℤ))) g := by
      intro g hg
      rcases hfac g hg with hn | ⟨hc0, hd⟩
      · have hn' : ipHas sp.2.1.id g.2 = false := by simpa [facHas] using hn
        unfold delV phV
        rw [ipDen_upd_of _ _ _ _ hn', ipDen_upd_of _ _ _ _ hn']
        exact ⟨rfl, rfl⟩
      · have hK := dvdP_ok V.ρ fl hfl PW hPW _ _ hd
        refine fac_congr' _ _ g E _ _ ?_ (DMon.den_pos _ hρ.dpos _ (Nat.pos_of_ne_zero hc0))
        refine (Int.natCast_dvd_natCast.2 hK).trans ?_
        rw [Int.emod_def]; exact ⟨ipDen P q / ((M.den V.ρ.dv : ℕ) : ℤ), by ring⟩
    simp only [bodyOf]
    congr 1
    · congr 1; exact List.map_congr_left fun g hg => (hcong g (List.mem_append_left _ hg)).1
    · congr 1; exact List.map_congr_left fun g hg => (hcong g (List.mem_append_right _ hg)).2
  · simp at h

/-- Reduced arguments read under divisors replaced, at most `n` times. -/
def modRules (fl : List ℕ) (PW : List (DMon × DMon)) : ℕ → ATm → ATm
  | 0, t => t
  | n + 1, t => match firstSome (modDropAt fl PW t) (splits t.ats) with
    | some t' => modRules fl PW n t'
    | none => t

theorem modRules_ok (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (n : ℕ) (t : ATm) :
    PWEq V t (modRules fl PW n t) := by
  induction n generalizing t with
  | zero => exact ⟨rfl, rfl, fun _ => rfl⟩
  | succ n ih =>
    simp only [modRules]
    split
    · next t' h =>
      obtain ⟨sp, hsp, he⟩ := firstSome_spec _ _ _ h
      exact (modDropAt_ok V hρ fl hfl PW hPW t t' sp (splits_spec _ sp hsp) he).trans (ih t')
    · exact ⟨rfl, rfl, fun _ => rfl⟩

/-- Atom rules around the reduced-argument and digit-condition steps. -/
@[irreducible] def preA (fl : List ℕ) (PW : List (DMon × DMon)) (t : ATm) : ATm :=
  aRules ((dzStep PW (modRules fl PW (t.ats.length + 1) (aRules (t.ats.length + 1) t))).ats.length + 1)
    (dzStep PW (modRules fl PW (t.ats.length + 1) (aRules (t.ats.length + 1) t)))

theorem preA_ok (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (t : ATm) : PWEq V t (preA fl PW t) := by
  unfold preA
  exact (((aRules_ok V _ t).trans (modRules_ok V hρ fl hfl PW hPW _ _)).trans
    (dzStep_ok V PW hPW _)).trans (aRules_ok V _ _)

/-- Atom rules, digit conditions read as deltas, factor canonicalisation, canonical names and
sorted sums; `none` when the term vanishes. -/
def canonCore (fl : List ℕ) (t : ATm) : Option ATm :=
  match canonF fl t.dels t.phs with
  | none => none
  | some (k, ds, ps) => some (sortSums (canonRen ⟨CMon.mul t.c k, t.sums, t.ats, ds, ps⟩))

/-- Atom rules, digit conditions read as deltas, factor canonicalisation, canonical names and
sorted sums; `none` when the term vanishes. -/
@[irreducible] def canonA (fl : List ℕ) (PW : List (DMon × DMon)) (t : ATm) : Option ATm :=
  canonCore fl (preA fl PW t)

theorem canonA_ok (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (t : ATm) : ARes V t (canonA fl PW t) := by
  have h13 := preA_ok V hρ fl hfl PW hPW t
  unfold canonA
  generalize preA fl PW t = t3 at h13 ⊢
  unfold canonCore
  split
  · next hn =>
    intro ν
    rw [← h13.den ν]
    exact den_zero_of_pw V t3 (fun μ => canonF_none V.ρ hρ fl hfl _ _ hn _) ν
  · next k ds ps hk =>
    intro ν
    rw [sortSums_ok, canonRen_ok, ← h13.den ν]
    refine den_of_pw V t3 ⟨CMon.mul t3.c k, t3.sums, t3.ats, ds, ps⟩ rfl (fun μ => ?_) ν
    unfold abody
    simp only
    rw [CMon.den_mul V.ρ hρ, mul_assoc, canonF_some V.ρ hρ fl hfl _ _ k ds ps hk]

/-- Eliminations and canonicalisation, alternated `r` times. -/
def normA (fl : List ℕ) (PW : List (DMon × DMon)) : ℕ → ATm → Option ATm
  | 0, t => some t
  | r + 1, t =>
    match canonA fl PW (aElimAll PW (t.sums.length + 1) t) with
    | none => none
    | some t' => normA fl PW r t'

theorem normA_ok (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (r : ℕ) (t : ATm) :
    ARes V t (normA fl PW r t) := by
  induction r generalizing t with
  | zero => intro ν; rfl
  | succ r ih =>
    have hc := canonA_ok V hρ fl hfl PW hPW (aElimAll PW (t.sums.length + 1) t)
    have he := aElimAll_ok V hρ PW hPW (t.sums.length + 1) t
    simp only [normA]
    split
    · next h => rw [h] at hc; intro ν; rw [← he ν, hc ν]
    · next t' h =>
      rw [h] at hc
      have := ih t'
      revert this
      cases normA fl PW r t' with
      | none => intro h' ν; rw [← he ν, ← hc ν, h' ν]
      | some t'' => intro h' ν; rw [h' ν, hc ν, he ν]

/-- The term without its rational coefficient. -/
def ashape (t : ATm) : ATm := { t with c := { t.c with q := 0 } }

/-- A term with rational coefficient `q`. -/
def awithQ (t : ATm) (q : ℚ) : ATm := { t with c := { t.c with q := q } }

/-- A term added to an expression, merged with one of the same shape. -/
def taAdd (t : ATm) : List ATm → List ATm
  | [] => if t.c.q = 0 then [] else [t]
  | u :: l =>
    if ashape u = ashape t then
      (if qAdd u.c.q t.c.q = 0 then l else awithQ u (qAdd u.c.q t.c.q) :: l)
    else u :: taAdd t l

theorem aden_withQ (V : LV) (ν : ℕ → ℤ) (t : ATm) (q : ℚ) :
    (awithQ t q).den V ν = (q : ℂ) * (awithQ t 1).den V ν := by
  unfold ATm.den
  rw [show abody V (awithQ t q) = abody V t from rfl,
    show abody V (awithQ t 1) = abody V t from rfl]
  simp only [awithQ, CMon.den]; push_cast; ring

theorem aden_self (V : LV) (ν : ℕ → ℤ) (t : ATm) :
    t.den V ν = (t.c.q : ℂ) * (awithQ t 1).den V ν := by
  rw [← aden_withQ]; rfl

theorem awithQ_shape (t u : ATm) (h : ashape u = ashape t) (q : ℚ) : awithQ u q = awithQ t q := by
  obtain ⟨⟨_, r, sq, rc, am, gc⟩, s, at', d, p⟩ := u
  obtain ⟨⟨_, r', sq', rc', am', gc'⟩, s', at'', d', p'⟩ := t
  simp only [ashape, ATm.mk.injEq, CMon.mk.injEq] at h
  obtain ⟨⟨-, rfl, rfl, rfl, rfl, rfl⟩, rfl, rfl, rfl, rfl⟩ := h
  rfl

theorem taAdd_ok (V : LV) (ν : ℕ → ℤ) (t : ATm) (l : List ATm) :
    axDen V ν (taAdd t l) = t.den V ν + axDen V ν l := by
  induction l with
  | nil =>
    simp only [taAdd]
    split_ifs with h
    · simp [axDen, aden_self V ν t, h]
    · simp [axDen]
  | cons u l ih =>
    simp only [taAdd, qAdd_eq]
    split_ifs with h1 h2
    · have e := awithQ_shape t u h1 1
      have hq : ((u.c.q + t.c.q : ℚ) : ℂ) = 0 := by rw [h2]; simp
      simp only [axDen, List.map_cons, List.sum_cons]
      rw [aden_self V ν t, aden_self V ν u, e]
      push_cast at hq
      linear_combination -(awithQ t 1).den V ν * hq
    · have e := awithQ_shape t u h1 1
      simp only [axDen, List.map_cons, List.sum_cons]
      rw [aden_withQ, aden_self V ν t, aden_self V ν u, e]
      push_cast; ring
    · simp only [axDen, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]; ring

/-- Rounds of eliminations per term. -/
def AROUNDS : ℕ := 8

/-- Every term normalised, terms of one shape merged. -/
def nfA (fl : List ℕ) (PW : List (DMon × DMon)) (e : List ATm) : List ATm :=
  (e.filterMap (normA fl PW AROUNDS)).foldl (fun acc t => taAdd t acc) []

theorem nfA_den (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (e : List ATm) (ν : ℕ → ℤ) :
    axDen V ν (nfA fl PW e) = axDen V ν e := by
  have h1 : ∀ (l : List ATm) (acc : List ATm),
      axDen V ν (l.foldl (fun acc t => taAdd t acc) acc) = axDen V ν acc + axDen V ν l := by
    intro l
    induction l with
    | nil => intro acc; simp [axDen]
    | cons t l ih =>
      intro acc
      rw [List.foldl_cons, ih, taAdd_ok]; simp only [axDen, List.map_cons, List.sum_cons]; ring
  have h2 : ∀ l : List ATm, axDen V ν (l.filterMap (normA fl PW AROUNDS)) = axDen V ν l := by
    intro l
    induction l with
    | nil => rfl
    | cons t l ih =>
      have := normA_ok V hρ fl hfl PW hPW AROUNDS t
      simp only [List.filterMap_cons]
      revert this
      cases normA fl PW AROUNDS t with
      | none => intro h; simp only [axDen, List.map_cons, List.sum_cons] at ih ⊢; rw [ih, h]; ring
      | some t' =>
        intro h
        simp only [axDen, List.map_cons, List.sum_cons] at ih ⊢; rw [ih, h]
  unfold nfA
  rw [h1, h2]; simp [axDen]

theorem nfA_zero (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (e : List ATm) (h : nfA fl PW e = [])
    (ν : ℕ → ℤ) : axDen V ν e = 0 := by
  rw [← nfA_den V hρ fl hfl PW hPW e ν, h]; rfl

/-- Every term negated. -/
def axNeg (e : List ATm) : List ATm := e.map fun t => awithQ t (-t.c.q)

theorem axDen_neg (V : LV) (ν : ℕ → ℤ) (e : List ATm) : axDen V ν (axNeg e) = -axDen V ν e := by
  induction e with
  | nil => simp [axDen, axNeg]
  | cons t e ih =>
    simp only [axNeg, List.map_cons, axDen, List.sum_cons] at ih ⊢
    rw [ih, aden_withQ, aden_self V ν t]; push_cast; ring

end ArchytasZX.LSym