/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymRange

/-!
# Carry steps over the labels' ranges

`elimStepC` is `elimStepX` with steps that keep a term's value wherever the labels of `R` lie in
range, each at the delta `f = [M ∣ p]` with index `j`: `7` `f` as `[M ∣ x - ((-s b) mod M)]`
for `p = s x + b`, `x` the sum at `i`; `8`, `9` `p` plus or minus the delta at `i` of the same
modulus, coefficients mod `M`; `10` `[D ∣ p mod D]` beside `f`, `D ∣ M` the range of the sum at
`i`; `11` `f` dropped beside a delta at `i` that implies it; `12` `f` split by `splitDel` over
the range at `i` of the sums and labels; `13` the pivot of `f` substituted into the root of
unity at `i`, its modulus dividing `M` times each coefficient of a monomial reading the pivot;
`14` `p` negated; `15` the square roots at `i` and `j` of the coefficient merged when their
product is a square; `16` the variable at `i` of the sums and labels, held by `f` at one value
by the ranges, substituted in every other factor. `rxNfC` is `rxNf` with `elimStepC`; `rxNfC_den` holds wherever the labels
of `R` lie in range.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Values of deltas -/

theorem dvd_iff_of_dvd_sub {k a b : ℤ} (h : k ∣ a - b) : k ∣ a ↔ k ∣ b :=
  ⟨fun ha => by have := dvd_sub ha h; simpa using this,
    fun hb => by have := dvd_add hb h; simpa using this⟩

theorem delV_congr (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f f' : Fac) (hm : f'.1 = f.1)
    (h : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f.2 ↔ ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f'.2) :
    delV dv ν f' = delV dv ν f := by
  unfold delV
  rw [hm]
  by_cases hc : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f.2
  · rw [if_pos hc, if_pos (h.1 hc)]
  · rw [if_neg hc, if_neg (fun h' => hc (h.2 h'))]

theorem delV_one (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) (h : delV dv ν f = 1) :
    ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f.2 := by
  unfold delV at h
  split_ifs at h with h'
  · exact h'
  · simp at h

theorem bodyOf_zero (dv : ℕ → ℕ) (ds ps : List Fac) (ν : ℕ → ℤ) (g : Fac) (hg : g ∈ ds)
    (h : delV dv ν g = 0) : bodyOf dv ds ps ν = 0 := by
  unfold bodyOf
  rw [List.prod_eq_zero (List.mem_map.2 ⟨g, hg, h⟩), zero_mul]

theorem om_congr (M : ℕ) (hM : 0 < M) (A B : ℤ) (h : (M : ℤ) ∣ A - B) : om M ^ A = om M ^ B := by
  obtain ⟨k, hk⟩ := h
  have hz := om_ne_zero _ hM
  rw [show A = B + (M : ℤ) * k by omega, zpow_add₀ hz, zpow_mul, zpow_natCast, om_pow_self _ hM,
    one_zpow, mul_one]

/-- `x` moving by a multiple of `m` moves `p` by a multiple of `M` when `M ∣ c m` for each
coefficient `c` of a monomial reading `x`. -/
theorem ipDen_upd_dvdM (ν : ℕ → ℤ) (x : ℕ) (m M a b : ℤ) (hab : m ∣ a - b) (p : IPoly)
    (h : p.all (fun t => !t.1.contains x || t.2 * m % M == 0) = true) :
    M ∣ ipDen (Function.update ν x a) p - ipDen (Function.update ν x b) p := by
  induction p with
  | nil => simp [ipDen]
  | cons t p ih =>
    simp only [List.all_cons, Bool.and_eq_true] at h
    rw [ipDen_cons, ipDen_cons, imDen_upd, imDen_upd]
    have e : t.2 * (a ^ t.1.count x * imDen ν (t.1.filter (· != x))) +
        ipDen (Function.update ν x a) p - (t.2 * (b ^ t.1.count x *
        imDen ν (t.1.filter (· != x))) + ipDen (Function.update ν x b) p) =
        (a ^ t.1.count x - b ^ t.1.count x) * (t.2 * imDen ν (t.1.filter (· != x))) +
        (ipDen (Function.update ν x a) p - ipDen (Function.update ν x b) p) := by ring
    rw [e]
    refine dvd_add ?_ (ih h.2)
    by_cases hc : t.1.contains x = true
    · have h1 := h.1
      simp only [hc, Bool.not_true, Bool.false_or, beq_iff_eq] at h1
      have hcm : M ∣ t.2 * m := Int.dvd_of_emod_eq_zero h1
      obtain ⟨u, hu⟩ := hab
      obtain ⟨w, hw⟩ := sub_dvd_pow_sub_pow a b (t.1.count x)
      rw [hw, hu]
      have e2 : m * u * w * (t.2 * imDen ν (t.1.filter (· != x))) =
          (t.2 * m) * (u * w * imDen ν (t.1.filter (· != x))) := by ring
      rw [e2]
      exact Dvd.dvd.mul_right hcm _
    · have hc0 : t.1.count x = 0 := List.count_eq_zero.2 (by simpa using hc)
      simp [hc0]

/-! ## Steps on the factors -/

/-- Rule `7`: `f = [M ∣ s x + b]` as `[M ∣ x - ((-s b) mod M)]`, `x` the sum at `i` of `S`. -/
def carry7 (S : List (ℕ × DMon)) (i : ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option (List Fac × List Fac) :=
  match S[i]? with
  | some p =>
    if f.1.isConst = true ∧ 0 < f.1.c then
      match ipLin p.1 f.2 with
      | some (a, b) =>
        match ipConst a with
        | some s =>
          if s = 1 ∨ s = -1 then
            some (pre ++ (f.1, ipSub (ipV p.1) (ipModC f.1.c (pinVal s b))) :: post, ps)
          else none
        | none => none
      | none => none
    else none
  | none => none

theorem carry7_ok (dv : ℕ → ℕ) (S : List (ℕ × DMon)) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps ds' ps' : List Fac) (h : carry7 S i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry7 at h
  split at h
  · next p _ =>
    split_ifs at h with hc
    obtain ⟨hk, _⟩ := hc
    split at h
    · next a b hlin =>
      split at h
      · next s hs =>
        split_ifs at h with hs1
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨_, _, hp⟩ := ipLin_spec p.1 f.2 a b hlin
        rw [bodyOf_cons_mid, bodyOf_cons_mid]
        congr 1
        refine delV_congr dv ν f _ rfl ?_
        rw [DMon.den_isConst dv f.1 hk, hp, ipConst_spec a s hs, ipDen_sub, ipDen_V,
          pin_dvd s hs1]
        have hv : ipDen ν (pinVal s b) = -s * ipDen ν b := by simp [pinVal, ipDen_smul]
        rw [← hv]
        refine dvd_iff_of_dvd_sub ?_
        have hm := ipDen_modC ν f.1.c (pinVal s b)
        rw [show ν p.1 - ipDen ν (pinVal s b) - (ν p.1 - ipDen ν (ipModC f.1.c (pinVal s b))) =
          -(ipDen ν (pinVal s b) - ipDen ν (ipModC f.1.c (pinVal s b))) by ring]
        exact dvd_neg.2 hm
      · simp at h
    · simp at h
  · simp at h

/-- Rules `8`, `9`: `f = [M ∣ p]` as `[M ∣ (p + σ q) mod M]` beside the delta `[M ∣ q]` at `i`
of `pre ++ post`. -/
def carry89 (σ : ℤ) (i : ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option (List Fac × List Fac) :=
  match (pre ++ post)[i]? with
  | some g =>
    if g.1 = f.1 ∧ f.1.isConst = true ∧ 0 < f.1.c then
      some (pre ++ (f.1, ipModC f.1.c (ipAdd f.2 (ipSmul σ g.2))) :: post, ps)
    else none
  | none => none

theorem carry89_ok (dv : ℕ → ℕ) (σ : ℤ) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps ds' ps' : List Fac) (h : carry89 σ i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry89 at h
  split at h
  · next g hg =>
    split_ifs at h with hc
    obtain ⟨hgf, hk, _⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hmem : g ∈ pre ++ post := List.mem_of_getElem? hg
    rw [bodyOf_cons_mid, bodyOf_cons_mid]
    rcases delV_cases dv ν g with h0 | h1
    · rw [bodyOf_zero dv _ ps ν g hmem h0]; ring
    · congr 1
      refine delV_congr dv ν f _ rfl ?_
      have hgd := delV_one dv ν g h1
      rw [hgf, DMon.den_isConst dv f.1 hk] at hgd
      rw [DMon.den_isConst dv f.1 hk]
      refine (dvd_iff_of_dvd_sub ?_).trans (dvd_iff_of_dvd_sub (ipDen_modC ν f.1.c _))
      rw [ipDen_add, ipDen_smul, show ipDen ν f.2 - (ipDen ν f.2 + σ * ipDen ν g.2) =
        -σ * ipDen ν g.2 by ring]
      exact Dvd.dvd.mul_left hgd _
  · simp at h

/-- Rule `10`: `[D ∣ p mod D]` after `f = [M ∣ p]`, `D ∣ M` the constant range of the sum at
`i` of `S`. -/
def carry10 (S : List (ℕ × DMon)) (i : ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option (List Fac × List Fac) :=
  match S[i]? with
  | some p =>
    if p.2.isConst = true ∧ 0 < p.2.c ∧ f.1.isConst = true ∧ f.1.c % p.2.c = 0 then
      some (pre ++ f :: (p.2, ipModC p.2.c f.2) :: post, ps)
    else none
  | none => none

theorem carry10_ok (dv : ℕ → ℕ) (S : List (ℕ × DMon)) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps ds' ps' : List Fac) (h : carry10 S i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry10 at h
  split at h
  · next p _ =>
    split_ifs at h with hc
    obtain ⟨hD, _, hk, hmod⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    rw [bodyOf_cons_mid dv pre f, bodyOf_cons_mid dv pre (p.2, ipModC p.2.c f.2),
      bodyOf_cons_mid dv pre f]
    rcases delV_cases dv ν f with h0 | h1
    · rw [h0]; ring
    · have hfd := delV_one dv ν f h1
      rw [DMon.den_isConst dv f.1 hk] at hfd
      have hDf : (p.2.c : ℤ) ∣ (f.1.c : ℤ) :=
        Int.natCast_dvd_natCast.2 (Nat.dvd_of_mod_eq_zero hmod)
      have hw : delV dv ν (p.2, ipModC p.2.c f.2) = 1 := by
        unfold delV
        rw [if_pos]
        show ((p.2.den dv : ℕ) : ℤ) ∣ ipDen ν (ipModC p.2.c f.2)
        rw [DMon.den_isConst dv p.2 hD]
        exact (dvd_iff_of_dvd_sub (ipDen_modC ν p.2.c f.2)).1 (hDf.trans hfd)
      rw [hw]; ring
  · simp at h

/-- Rule `11`: `f = [K ∣ p]` dropped beside the delta `[M ∣ q]` at `i` of `pre ++ post`,
`K ∣ M`, `q ≡ ±p` mod `K`. -/
def carry11 (i : ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option (List Fac × List Fac) :=
  match (pre ++ post)[i]? with
  | some g =>
    if g.1.isConst = true ∧ 0 < g.1.c ∧ f.1.isConst = true ∧ 0 < f.1.c ∧ g.1.c % f.1.c = 0 ∧
        (ipModC f.1.c g.2 == ipModC f.1.c f.2 ||
          ipModC f.1.c (ipSmul (-1) g.2) == ipModC f.1.c f.2) = true then
      some (pre ++ post, ps)
    else none
  | none => none

theorem carry11_ok (dv : ℕ → ℕ) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps ds' ps' : List Fac) (h : carry11 i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry11 at h
  split at h
  · next g hg =>
    split_ifs at h with hc
    obtain ⟨hgk, _, hk, _, hmod, hm⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hmem : g ∈ pre ++ post := List.mem_of_getElem? hg
    rw [bodyOf_cons_mid]
    rcases delV_cases dv ν g with h0 | h1
    · rw [bodyOf_zero dv _ ps ν g hmem h0]; ring
    · have hgd := delV_one dv ν g h1
      rw [DMon.den_isConst dv g.1 hgk] at hgd
      have hKg : (f.1.c : ℤ) ∣ (g.1.c : ℤ) :=
        Int.natCast_dvd_natCast.2 (Nat.dvd_of_mod_eq_zero hmod)
      have hf : delV dv ν f = 1 := by
        unfold delV
        rw [if_pos]
        rw [DMon.den_isConst dv f.1 hk]
        exact (dvd_iff_of_modC ν f.1.c g.2 f.2 hm).1 (hKg.trans hgd)
      rw [hf]; ring
  · simp at h

/-- Rule `12`: `f` split by `splitDel` over the range at `i` of `rgOf R S`. -/
def carry12 (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps : List Fac) : Option (List Fac × List Fac) :=
  match (rgOf R S)[i]? with
  | some p =>
    if f.1 = p.2 then none else
      match splitDel (rgOf R S) p.1 p.2 f with
      | some (f1, f2, _) => some (pre ++ f1 :: f2 :: post, ps)
      | none => none
  | none => none

theorem carry12_ok (dv : ℕ → ℕ) (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) (i : ℕ)
    (pre : List Fac) (f : Fac) (post ps ds' ps' : List Fac)
    (h : carry12 R S i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ)
    (hν : InR dv (rgOf R S) ν) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry12 at h
  split at h
  · next p _ =>
    split_ifs at h
    split at h
    · next f1 f2 U hs =>
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hU, hval⟩ := splitDel_ok dv (rgOf R S) p.1 p.2 f f1 f2 U hs
      have hv := hval ν (fun q hq => hν q (hU q hq).1)
      rw [bodyOf_cons_mid dv pre f1, bodyOf_cons_mid dv pre f2, bodyOf_cons_mid dv pre f, hv]
      ring
    · simp at h
  · simp at h

/-- Rule `13`: the pivot `x = -s b` of `f = [m ∣ s x + b]` substituted into the root of unity
`ω_M^q` at `i`, `M ∣ c m` for each coefficient `c` of a monomial of `q` reading `x`. -/
def carry13 (i : ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option (List Fac × List Fac) :=
  match ps.drop i, pivotOf f.2 with
  | g :: ppost, some q =>
    if f.1.isConst = true ∧ 0 < f.1.c ∧ g.1.isConst = true ∧ 0 < g.1.c ∧
        g.2.all (fun t => !t.1.contains q.1 || t.2 * (f.1.c : ℤ) % (g.1.c : ℤ) == 0) = true then
      some (pre ++ f :: post, ps.take i ++ facSub q.1 (pinVal q.2.1 q.2.2) g :: ppost)
    else none
  | _, _ => none

theorem carry13_ok (dv : ℕ → ℕ) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps ds' ps' : List Fac) (h : carry13 i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry13 at h
  split at h
  · next g ppost q hd hq =>
    split_ifs at h with hc
    obtain ⟨hk, _, hgk, hgpos, hall⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hps : ps = ps.take i ++ g :: ppost := by rw [← hd, List.take_append_drop]
    obtain ⟨hs1, _, hp⟩ := pivotOf_spec f.2 q hq
    conv_rhs => rw [hps]
    rw [bodyOf_cons_mid, bodyOf_cons_mid]
    rcases delV_cases dv ν f with h0 | h1
    · rw [h0]; ring
    · congr 1
      have hfd := delV_one dv ν f h1
      rw [DMon.den_isConst dv f.1 hk, hp, pin_dvd q.2.1 hs1] at hfd
      have hv : ipDen ν (pinVal q.2.1 q.2.2) = -q.2.1 * ipDen ν q.2.2 := by
        simp [pinVal, ipDen_smul]
      rw [← hv] at hfd
      have hph : phV dv ν (facSub q.1 (pinVal q.2.1 q.2.2) g) = phV dv ν g := by
        unfold phV facSub
        simp only
        rw [ipDen_subst, DMon.den_isConst dv g.1 hgk]
        refine om_congr g.1.c hgpos _ _ ?_
        have := ipDen_upd_dvdM ν q.1 (f.1.c : ℤ) (g.1.c : ℤ) (ipDen ν (pinVal q.2.1 q.2.2))
          (ν q.1) (by rw [← neg_sub]; exact dvd_neg.2 hfd) g.2 hall
        rwa [Function.update_eq_self] at this
      unfold bodyOf
      simp only [List.map_append, List.map_cons, List.prod_append, List.prod_cons, hph]
  · simp at h

/-- Rule `14`: `f = [M ∣ p]` as `[M ∣ -p]`. -/
def carry14 (pre : List Fac) (f : Fac) (post ps : List Fac) : List Fac × List Fac :=
  (pre ++ (f.1, ipSmul (-1) f.2) :: post, ps)

theorem carry14_ok (dv : ℕ → ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) (ν : ℕ → ℤ) :
    bodyOf dv (carry14 pre f post ps).1 (carry14 pre f post ps).2 ν =
      bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry14
  rw [bodyOf_cons_mid, bodyOf_cons_mid]
  congr 1
  refine delV_congr dv ν f _ rfl ?_
  rw [ipDen_smul, neg_one_mul, dvd_neg]

/-- Rule `16`: `x`, the variable at `i` of `rgOf R S`, of constant range `D`, held by
`f = [D ∣ s x + b]`, `s = ±1`, with `V = -s b` in `[k D, k D + D)` by the ranges: `x = V - k D`
substituted in every other factor. -/
def carry16 (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps : List Fac) : Option (List Fac × List Fac) :=
  match (rgOf R S)[i]? with
  | some p =>
    if f.1 = p.2 ∧ p.2.isConst = true ∧ 0 < p.2.c then
      match ipLin p.1 f.2 with
      | some (a, b) =>
        match ipConst a with
        | some s =>
          if s = 1 ∨ s = -1 then
            match ivl (rgOf R S) (pinVal s b) with
            | some (lo, hi) =>
              if hi < lo / (p.2.c : ℤ) * p.2.c + p.2.c then
                some (pre.map (facSub p.1 (ipAdd (pinVal s b) (ipC (-(lo / (p.2.c : ℤ) * p.2.c)))))
                  ++ f :: post.map (facSub p.1 (ipAdd (pinVal s b)
                    (ipC (-(lo / (p.2.c : ℤ) * p.2.c))))),
                  ps.map (facSub p.1 (ipAdd (pinVal s b) (ipC (-(lo / (p.2.c : ℤ) * p.2.c))))))
              else none
            | none => none
          else none
        | none => none
      | none => none
    else none
  | none => none

theorem carry16_ok (dv : ℕ → ℕ) (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) (i : ℕ)
    (pre : List Fac) (f : Fac) (post ps ds' ps' : List Fac)
    (h : carry16 R S i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ)
    (hν : InR dv (rgOf R S) ν) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry16 at h
  split at h
  · next p hpi =>
    split_ifs at h with hc
    obtain ⟨hf, hk, hpos⟩ := hc
    split at h
    · next a b hlin =>
      split at h
      · next s hs =>
        split_ifs at h with hs1
        split at h
        · next lo hi hiv =>
          split_ifs at h with hlt
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          set v := ipAdd (pinVal s b) (ipC (-(lo / (p.2.c : ℤ) * p.2.c))) with hvdef
          obtain ⟨_, _, hp⟩ := ipLin_spec p.1 f.2 a b hlin
          rw [bodyOf_cons_mid, ← List.map_append, bodyOf_cons_mid, bodyOf_sub]
          rcases delV_cases dv ν f with h0 | h1
          · rw [h0]; ring
          · congr 1
            have hfd := delV_one dv ν f h1
            rw [hf, DMon.den_isConst dv p.2 hk, hp, ipConst_spec a s hs, pin_dvd s hs1] at hfd
            have hV : -s * ipDen ν b = ipDen ν (pinVal s b) := by simp [pinVal, ipDen_smul]
            rw [hV] at hfd
            obtain ⟨hlo, hhi⟩ := ivl_spec dv (rgOf R S) ν hν _ lo hi hiv
            have hr := hν p (List.mem_of_getElem? hpi)
            rw [DMon.den_isConst dv p.2 hk] at hr
            obtain ⟨hr0, hr1⟩ := hr
            set D : ℤ := (p.2.c : ℤ)
            set k := lo / D
            have hDz : (0 : ℤ) < D := Int.natCast_pos.2 hpos
            have hkl : k * D ≤ lo := Int.ediv_mul_le lo hDz.ne'
            have hval : ipDen ν v = ipDen ν (pinVal s b) - k * D := by
              rw [hvdef, ipDen_add, ipDen_C]; ring
            have hx : ν p.1 = ipDen ν v := by
              have hd : D ∣ ν p.1 - ipDen ν v := by
                rw [hval, show ν p.1 - (ipDen ν (pinVal s b) - k * D) =
                  (ν p.1 - ipDen ν (pinVal s b)) + D * k by ring]
                exact dvd_add hfd (dvd_mul_right D k)
              have := Int.eq_zero_of_abs_lt_dvd hd (abs_lt.2 ⟨by omega, by omega⟩)
              omega
            rw [← hx, Function.update_eq_self]
        · simp at h
      · simp at h
    · simp at h
  · simp at h

/-- The factors after step `(i, r, j)`, `7 ≤ r ≤ 16`, at the delta `f` between `pre` and
`post`, beside the sums `S`. -/
def carryFac (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) (i r : ℕ) (pre : List Fac) (f : Fac)
    (post ps : List Fac) : Option (List Fac × List Fac) :=
  if r = 7 then carry7 S i pre f post ps
  else if r = 8 then carry89 1 i pre f post ps
  else if r = 9 then carry89 (-1) i pre f post ps
  else if r = 10 then carry10 S i pre f post ps
  else if r = 11 then carry11 i pre f post ps
  else if r = 12 then carry12 R S i pre f post ps
  else if r = 13 then carry13 i pre f post ps
  else if r = 14 then some (carry14 pre f post ps)
  else if r = 16 then carry16 R S i pre f post ps
  else none

theorem carryFac_ok (dv : ℕ → ℕ) (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) (i r : ℕ)
    (pre : List Fac) (f : Fac) (post ps ds' ps' : List Fac)
    (h : carryFac R S i r pre f post ps = some (ds', ps')) (ν : ℕ → ℤ)
    (hν : r = 12 ∨ r = 16 → InR dv (rgOf R S) ν) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carryFac at h
  split_ifs at h
  · exact carry7_ok dv S i pre f post ps ds' ps' h ν
  · exact carry89_ok dv 1 i pre f post ps ds' ps' h ν
  · exact carry89_ok dv (-1) i pre f post ps ds' ps' h ν
  · exact carry10_ok dv S i pre f post ps ds' ps' h ν
  · exact carry11_ok dv i pre f post ps ds' ps' h ν
  · exact carry12_ok dv R S i pre f post ps ds' ps' h ν (hν (Or.inl (by assumption)))
  · exact carry13_ok dv i pre f post ps ds' ps' h ν
  · have h' := Option.some.inj h
    have := carry14_ok dv pre f post ps ν
    rw [h'] at this
    exact this
  · exact carry16_ok dv R S i pre f post ps ds' ps' h ν (hν (Or.inr (by assumption)))

/-! ## Square roots of the coefficient -/

/-- Rule `15`: the square roots `√a`, `√b` at `i` and `j` of the coefficient as the integer
`√(a b)`. -/
def rcStep (t : STm) (i j : ℕ) : Option STm :=
  match t.c.rc[i]?, t.c.rc[j]? with
  | some a, some b =>
    if (t.c.rc.erase a).contains b = true ∧ CMon.sqrtS (a * b) * CMon.sqrtS (a * b) = a * b then
      some ⟨{ t.c with q := t.c.q * (CMon.sqrtS (a * b) : ℚ), rc := (t.c.rc.erase a).erase b },
        t.sums, t.dels, t.phs⟩
    else none
  | _, _ => none

theorem rcStep_ok (ρ : Val) (t t' : STm) (i j : ℕ) (h : rcStep t i j = some t') (ν : ℕ → ℤ) :
    t'.den ρ ν = t.den ρ ν := by
  unfold rcStep at h
  split at h
  · next a b ha _ =>
    split_ifs at h with hc
    obtain ⟨hcon, hsq⟩ := hc
    simp only [Option.some.injEq] at h
    subst h
    have ham : a ∈ t.c.rc := List.mem_of_getElem? ha
    have hbm : b ∈ t.c.rc.erase a := by simpa using hcon
    have hp := (List.perm_cons_erase ham).trans ((List.perm_cons_erase hbm).cons a)
    have hprod : (t.c.rc.map rt).prod =
        rt a * rt b * (((t.c.rc.erase a).erase b).map rt).prod := by
      rw [(hp.map rt).prod_eq]
      simp only [List.map_cons, List.prod_cons]
      ring
    set s := CMon.sqrtS (a * b) with hs
    have hrt : rt a * rt b = (s : ℂ) := by rw [← rt_mul, ← hsq, rt_mul, rt_sq]
    unfold STm.den CMon.den
    simp only [hprod, body_eq]
    push_cast
    rw [← hrt]
    ring
  · simp at h

/-! ## Steps and normal forms -/

/-- Rule `15` `rcStep`; rules `7` to `14` and `16` `carryFac` at the delta at `j`, the sums'
names distinct for `12` and `16`; else `elimStepX`. -/
def elimStepC (R : List (ℕ × ℕ)) (t : STm) (i r j : ℕ) : Option STm :=
  if r = 15 then rcStep t i j
  else if 7 ≤ r ∧ r ≤ 16 then
    if (r = 12 ∨ r = 16) ∧ ¬(t.sums.map (·.1)).Nodup then none
    else
      match t.dels.drop j with
      | f :: post =>
        match carryFac R t.sums i r (t.dels.take j) f post t.phs with
        | some (ds, ps) => some ⟨t.c, t.sums, ds, ps⟩
        | none => none
      | [] => none
  else elimStepX R t i r j

theorem elimStepC_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (t t' : STm) (i r j : ℕ)
    (h : elimStepC R t i r j = some t') (ν : ℕ → ℤ) (hν : InLab R ν) :
    t'.den ρ ν = t.den ρ ν := by
  unfold elimStepC at h
  split_ifs at h with h15 hr hnd
  · exact rcStep_ok ρ t t' i j h ν
  · split at h
    · next f post hd =>
      split at h
      · next ds ps hc =>
        simp only [Option.some.injEq] at h
        subst h
        have hdl : t.dels.take j ++ f :: post = t.dels := by rw [← hd, List.take_append_drop]
        unfold STm.den
        congr 1
        simp only [body_eq]
        by_cases hr2 : r = 12 ∨ r = 16
        · have hnd' : (t.sums.map (·.1)).Nodup := by
            by_contra hc'
            exact hnd ⟨hr2, hc'⟩
          obtain ⟨hA, hB⟩ := rgOf_in ρ.dv R t.sums hnd' ν hν
          refine sumsV_congr_rng ρ.dv t.sums (rgOf R t.sums) _ _ ν hA hB (fun ν' hν' => ?_)
          rw [carryFac_ok ρ.dv R t.sums i r _ f post t.phs ds ps hc ν' (fun _ => hν'), hdl]
        · refine sumsV_congr ρ.dv t.sums _ _ (fun ν' => ?_) ν
          rw [carryFac_ok ρ.dv R t.sums i r _ f post t.phs ds ps hc ν'
            (fun h2 => absurd h2 hr2), hdl]
      · simp at h
    · simp at h
  · exact elimStepX_ok ρ hρ R t t' i r j h ν hν

/-- `elimStepC` along the hints, each result forced; the term at the first that fails. -/
def elimAllC (R : List (ℕ × ℕ)) : List (ℕ × ℕ × ℕ) → STm → STm
  | [], t => t
  | (i, r, j) :: hs, t =>
    match elimStepC R t i r j with
    | some t' => fTm t' (elimAllC R hs)
    | none => t

theorem elimAllC_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) :
    ∀ (hs : List (ℕ × ℕ × ℕ)) (t : STm) (ν : ℕ → ℤ), InLab R ν →
      (elimAllC R hs t).den ρ ν = t.den ρ ν := by
  intro hs
  induction hs with
  | nil => intro t ν _; rfl
  | cons h hs ih =>
    obtain ⟨i, r, j⟩ := h
    intro t ν hν
    simp only [elimAllC]
    cases he : elimStepC R t i r j with
    | none => rfl
    | some t' => dsimp only; rw [fTm_eq, ih _ ν hν, elimStepC_ok ρ hρ R t t' i r j he ν hν]

/-- `elimStepC` along the hints, each result forced; `none` at the first that fails. -/
def elimAllCS (R : List (ℕ × ℕ)) : List (ℕ × ℕ × ℕ) → STm → Option STm
  | [], t => some t
  | (i, r, j) :: hs, t =>
    match elimStepC R t i r j with
    | some t' => fTm t' (elimAllCS R hs)
    | none => none

theorem elimAllC_append (R : List (ℕ × ℕ)) :
    ∀ (es1 es2 : List (ℕ × ℕ × ℕ)) (t u : STm), elimAllCS R es1 t = some u →
      elimAllC R (es1 ++ es2) t = elimAllC R es2 u := by
  intro es1
  induction es1 with
  | nil => intro es2 t u h; simp only [elimAllCS, Option.some.injEq] at h; subst h; rfl
  | cons s es1 ih =>
    obtain ⟨i, r, j⟩ := s
    intro es2 t u h
    simp only [elimAllCS] at h
    simp only [List.cons_append, elimAllC]
    cases he : elimStepC R t i r j with
    | none => simp [he] at h
    | some t' =>
      simp only [he, fTm_eq] at h ⊢
      exact ih es2 t' u h

theorem elimAllC_of_S (R : List (ℕ × ℕ)) (es : List (ℕ × ℕ × ℕ)) (t u : STm)
    (h : elimAllCS R es t = some u) : elimAllC R es t = u := by
  have := elimAllC_append R es [] t u h
  rw [List.append_nil] at this
  rw [this]; rfl

/-- `normTmX` with `elimAllC`. -/
def normTmC (R : List (ℕ × ℕ)) (fl : List ℕ) : List RHint → STm → Option STm
  | [], t => some t
  | (c, es) :: cs, t =>
    match canonTmH c fl (fTm t (elimAllC R es)) with
    | none => none
    | some t' => fTm t' (normTmC R fl cs)

theorem normTmC_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ) :
    ∀ (cs : List RHint) (t : STm), TmResR ρ R t (normTmC R fl cs t) := by
  intro cs
  induction cs with
  | nil => intro t ν _; rfl
  | cons h cs ih =>
    obtain ⟨c, es⟩ := h
    intro t
    have hc := canonTmH_ok ρ hρ fl hfl c (elimAllC R es t)
    have he := elimAllC_ok ρ hρ R es t
    simp only [normTmC, fTm_eq]
    revert hc
    cases canonTmH c fl (elimAllC R es t) with
    | none => intro hc ν hν; rw [← he ν hν, hc ν]
    | some t' =>
      intro hc
      simp only [fTm_eq]
      have := ih t'
      revert this
      cases normTmC R fl cs t' with
      | none => intro h' ν hν; rw [← he ν hν, ← hc ν, h' ν hν]
      | some t'' => intro h' ν hν; rw [h' ν hν, hc ν, he ν hν]

/-- The terms normalised with their rounds, in order, summed into `acc`. -/
def nfGoC (R : List (ℕ × ℕ)) (fl : List ℕ) : List (List RHint) → List STm → List STm → List STm
  | _, acc, [] => acc
  | hs, acc, t :: ts =>
    match normTmC R fl (hs.headD []) t with
    | none => nfGoC R fl hs.tail acc ts
    | some t' => nfGoC R fl hs.tail (tmAdd t' acc) ts

theorem nfGoC_den (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (ν : ℕ → ℤ) (hν : InLab R ν) : ∀ (e : List STm) (hs : List (List RHint)) (acc : List STm),
    exDen ρ ν (nfGoC R fl hs acc e) = exDen ρ ν acc + exDen ρ ν e := by
  intro e
  induction e with
  | nil => intro hs acc; simp [nfGoC, exDen]
  | cons t ts ih =>
    intro hs acc
    have hn := normTmC_ok ρ hρ R fl hfl (hs.headD []) t
    simp only [nfGoC]
    revert hn
    cases normTmC R fl (hs.headD []) t with
    | none =>
      intro hn
      dsimp only
      rw [ih]
      simp only [exDen, List.map_cons, List.sum_cons]
      rw [hn ν hν]; ring
    | some t' =>
      intro hn
      dsimp only
      rw [ih, tmAdd_ok]
      simp only [exDen, List.map_cons, List.sum_cons]
      rw [hn ν hν]; ring

theorem normTmC_one (R : List (ℕ × ℕ)) (fl : List ℕ) (c : ℕ) (es : List (ℕ × ℕ × ℕ))
    (t u : STm) (h : elimAllC R es t = u) : normTmC R fl [(c, es)] t = canonTmH c fl u := by
  simp only [normTmC, fTm_eq, h]
  cases canonTmH c fl u <;> simp [fTm_eq, normTmC]

theorem normTmC_step (R : List (ℕ × ℕ)) (fl : List ℕ) (h : RHint) (cs : List RHint) (t t' : STm)
    (hh : normTmC R fl [h] t = some t') : normTmC R fl (h :: cs) t = normTmC R fl cs t' := by
  obtain ⟨c, es⟩ := h
  simp only [normTmC, fTm_eq] at hh ⊢
  cases hc : canonTmH c fl (elimAllC R es t) with
  | none => rw [hc] at hh; simp at hh
  | some u =>
    rw [hc] at hh
    simp only [fTm_eq, normTmC, Option.some.injEq] at hh ⊢
    rw [hh]

theorem normTmC_stop (R : List (ℕ × ℕ)) (fl : List ℕ) (h : RHint) (cs : List RHint) (t : STm)
    (hh : normTmC R fl [h] t = none) : normTmC R fl (h :: cs) t = none := by
  obtain ⟨c, es⟩ := h
  simp only [normTmC, fTm_eq] at hh ⊢
  cases hc : canonTmH c fl (elimAllC R es t) with
  | none => rfl
  | some u => rw [hc] at hh; simp [fTm_eq, normTmC] at hh

theorem nfGoC_some (R : List (ℕ × ℕ)) (fl : List ℕ) (h : List RHint) (hs : List (List RHint))
    (acc : List STm) (t u : STm) (ts : List STm) (hh : normTmC R fl h t = some u) :
    nfGoC R fl (h :: hs) acc (t :: ts) = nfGoC R fl hs (tmAdd u acc) ts := by
  simp [nfGoC, hh]

theorem nfGoC_none (R : List (ℕ × ℕ)) (fl : List ℕ) (h : List RHint) (hs : List (List RHint))
    (acc : List STm) (t : STm) (ts : List STm) (hh : normTmC R fl h t = none) :
    nfGoC R fl (h :: hs) acc (t :: ts) = nfGoC R fl hs acc ts := by
  simp [nfGoC, hh]

/-- The residual normalised term by term with the labels' ranges `R`, the rounds `H` and the
carry steps. -/
def rxNfC (R : List (ℕ × ℕ)) (fl : List ℕ) (H : List (List RHint)) (e : List STm) : List STm :=
  nfGoC R fl H [] e

theorem rxNfC_den (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (H : List (List RHint)) (nf : List STm → List STm)
    (hnf : ∀ e ν, exDen ρ ν (nf e) = exDen ρ ν e) (e : List STm) (ν : ℕ → ℤ)
    (hν : InLab R ν) : exDen ρ ν (rxNfC R fl H (nf e)) = exDen ρ ν e := by
  unfold rxNfC
  rw [nfGoC_den ρ hρ R fl hfl ν hν, hnf]
  simp [exDen]

end ArchytasZX.Sym
