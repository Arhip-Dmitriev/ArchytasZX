/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymFourier

/-!
# Residuals that vanish on the labels' ranges

A residual of `nfEx` can vanish only where every boundary label lies in its range. `elimPin1`
removes a bound variable `x` of constant range `D` held by a delta `[D ∣ s x + c]`, `s = ±1`,
`c` constant: on `[0, D)` the delta holds at `x = (-s c) mod D` alone, so every factor takes
that value, whatever its modulus. `elimAllR` runs these pins before the moves of `elimStep`.
`rsplit` substitutes each label of `R` by every value of its range; `rgOK` checks that the
pinned residual is sum-free and that each case, pruned, is empty or passes `fzOK`.
`rgOK_zero` gives the residual's value `0` at every valuation with the labels of `R` in range,
and `labOK` reads those ranges off the boundary labels' dimensions, so `equal_symR` and
`equal_symSR` conclude `Equal`.
`rxNf` normalises a residual term by term with hinted steps (`elimStepX`): `elimGeo1` sums a
geometric series `Σ_{x<d} ω_{Kd}^(A x)` beside `[K ∣ A]` to `d [Kd ∣ A]`, `elimXpin1` pins a
bound variable by a delta of its range whose other side lies in one block by the ranges of the
other sums and of the labels; `rxNf_den` holds wherever the labels of `R` lie in range.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Pins by range -/

/-- `x` of constant range `D` held by the delta `f = [D ∣ s x + c]`, `s = ±1`, `c` constant. -/
def elimPin1 (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option Elim :=
  if f.1 = D ∧ D.isConst = true ∧ 0 < D.c then
    match ipLin x f.2 with
    | some (a, b) =>
      match ipConst a, ipConst b with
      | some s, some c =>
        if s = 1 ∨ s = -1 then
          some ⟨CMon.ofQ 1, (pre ++ post).map (facSub x (ipC ((-s * c) % (D.c : ℤ)))),
            ps.map (facSub x (ipC ((-s * c) % (D.c : ℤ))))⟩
        else none
      | _, _ => none
    | none => none
  else none

/-- `x` held by the first delta that pins it by range. -/
def elimPin (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  firstSome (fun p => elimPin1 x D p.1 p.2.1 p.2.2 ps) (splits ds)

theorem bodyOf_cons_mid (dv : ℕ → ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac)
    (ν : ℕ → ℤ) : bodyOf dv (pre ++ f :: post) ps ν = delV dv ν f * bodyOf dv (pre ++ post) ps ν := by
  simp only [bodyOf, List.map_append, List.map_cons, List.prod_append, List.prod_cons]; ring

theorem elimPin1_ok (ρ : Val) (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac)
    (post ps : List Fac) (e : Elim) (h : elimPin1 x D pre f post ps = some e) :
    ElimOK ρ x D (pre ++ f :: post) ps e := by
  unfold elimPin1 at h
  split_ifs at h with hc
  obtain ⟨hf, hk, hpos⟩ := hc
  split at h
  · next a b hlin =>
    split at h
    · next s c hs hcst =>
      split_ifs at h with hs1
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨_, _, hp⟩ := ipLin_spec x f.2 a b hlin
      intro ν
      have hDd : D.den ρ.dv = D.c := DMon.den_isConst ρ.dv D hk
      set w : ℤ := (-s * c) % (D.c : ℤ) with hwdef
      have hw0 : 0 ≤ w := Int.emod_nonneg _ (by exact_mod_cast hpos.ne')
      have hw1 : w < D.c := Int.emod_lt_of_pos _ (by exact_mod_cast hpos)
      have hdel : ∀ j : ℕ, j < D.c →
          delV ρ.dv (Function.update ν x (j : ℤ)) f = if j = w.toNat then 1 else 0 := by
        intro j hj
        unfold delV
        rw [hp, ipConst_spec a s hs, ipConst_spec b c hcst, Function.update_self, hf, hDd]
        have e1 := pin_dvd s hs1 (D.c : ℤ) (j : ℤ) c
        have e2 := dvd_sub_iff_eq D.c hpos j hj (-s * c)
        by_cases hj' : j = w.toNat
        · rw [if_pos (e1.2 (e2.2 hj')), if_pos hj']
        · rw [if_neg (fun hd => hj' (e2.1 (e1.1 hd))), if_neg hj']
      have hwlt : w.toNat < D.c := by omega
      rw [hDd, Finset.sum_congr rfl fun j hj => by
        rw [bodyOf_cons_mid, hdel j (Finset.mem_range.1 hj), ite_mul, one_mul, zero_mul]]
      rw [Finset.sum_ite_eq' (range D.c) w.toNat, if_pos (Finset.mem_range.2 hwlt), ofQ_den,
        bodyOf_sub, ipDen_C, Int.toNat_of_nonneg hw0]
      push_cast; ring
    · simp at h
  · simp at h

theorem elimPin_ok (ρ : Val) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimPin x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  rw [splits_spec ds p hp]
  exact elimPin1_ok ρ x D _ _ _ _ e he

/-- A pin by range first, else `elimVar`. -/
def elimVarR (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  match elimPin x D ds ps with
  | some e => some e
  | none => elimVar x D ds ps

theorem elimVarR_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimVarR x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  unfold elimVarR at h
  split at h
  · next e' he => simp only [Option.some.injEq] at h; subst h; exact elimPin_ok ρ x D ds ps _ he
  · exact elimVar_ok ρ hρ x D ds ps e h

/-- `elimFirst` with `elimVarR`. -/
def elimFirstR (t : STm) : Option STm :=
  firstSome (fun p => if (p.1 ++ p.2.2).map (·.1) |>.contains p.2.1.1 then none else
    match elimVarR p.2.1.1 p.2.1.2 t.dels t.phs with
    | some e => some ⟨CMon.mul t.c e.k, p.1 ++ p.2.2, e.ds, e.ps⟩
    | none => none) (splits t.sums)

theorem elimFirstR_ok (ρ : Val) (hρ : ρ.OK) (t t' : STm) (h : elimFirstR t = some t')
    (ν : ℕ → ℤ) : t'.den ρ ν = t.den ρ ν := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  obtain ⟨a, ⟨x, D⟩, b⟩ := p
  simp only at he
  split_ifs at he with hx
  split at he
  · next e hev =>
    simp only [Option.some.injEq] at he
    subst he
    have hok := elimVarR_ok ρ hρ x D t.dels t.phs e hev
    have hs := splits_spec t.sums _ hp
    simp only at hs
    unfold STm.den
    rw [hs, sumsV_pull ρ.dv a b x D (by simpa using hx)]
    simp only [body_eq] at *
    rw [sumsV_congr _ _ _ _ hok, sumsV_mul_left, CMon.den_mul ρ hρ]
    ring
  · simp at he

/-- `splitStep` without splits of a delta whose modulus is the variable's range. -/
def splitStepR (t : STm) : Option STm :=
  firstSome (fun p => firstSome (fun q => if p.2.1.1 = q.2 then none else
    splitApply t q.1 q.2 p.1 p.2.1 p.2.2) t.sums) (splits t.dels)

theorem splitStepR_ok (ρ : Val) (t t' : STm) (h : splitStepR t = some t') (ν : ℕ → ℤ) :
    t'.den ρ ν = t.den ρ ν := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  obtain ⟨q, _, he'⟩ := firstSome_spec _ _ _ he
  split_ifs at he'
  exact splitApply_ok ρ t t' q.1 q.2 p.1 p.2.1 p.2.2 (splits_spec t.dels p hp) he' ν

/-- `elimStep` with `elimFirstR` and `splitStepR`. -/
def elimStepR (t : STm) : Option STm :=
  match elimFirstR t with
  | some t' => some t'
  | none => splitStepR t

theorem elimStepR_ok (ρ : Val) (hρ : ρ.OK) (t t' : STm) (h : elimStepR t = some t')
    (ν : ℕ → ℤ) : t'.den ρ ν = t.den ρ ν := by
  unfold elimStepR at h
  split at h
  · next t'' h' =>
    simp only [Option.some.injEq] at h
    subst h
    exact elimFirstR_ok ρ hρ t t'' h' ν
  · exact splitStepR_ok ρ t t' h ν

/-- `elimStepR` repeated, at most `n` times. -/
def elimAllR : ℕ → STm → STm
  | 0, t => t
  | n + 1, t => match elimStepR t with
    | some t' => elimAllR n t'
    | none => t

theorem elimAllR_ok (ρ : Val) (hρ : ρ.OK) (n : ℕ) (t : STm) (ν : ℕ → ℤ) :
    (elimAllR n t).den ρ ν = t.den ρ ν := by
  induction n generalizing t with
  | zero => rfl
  | succ n ih =>
    simp only [elimAllR]
    split
    · next t' h => rw [ih, elimStepR_ok ρ hρ t t' h]
    · rfl

theorem exDen_elimAllR (ρ : Val) (hρ : ρ.OK) (n : ℕ) (e : List STm) (ν : ℕ → ℤ) :
    exDen ρ ν (e.map (elimAllR n)) = exDen ρ ν e := by
  unfold exDen
  rw [List.map_map]
  congr 1
  exact List.map_congr_left fun t _ => elimAllR_ok ρ hρ n t ν

/-! ## Cases of the labels -/

/-- Every pair `(x, D)` of `R` has `0 ≤ ν x < D`. -/
def InLab (R : List (ℕ × ℕ)) (ν : ℕ → ℤ) : Prop := ∀ p ∈ R, 0 ≤ ν p.1 ∧ ν p.1 < p.2

/-- The expression with each label of `R` substituted by each value of its range. -/
def rsplit : List (ℕ × ℕ) → List STm → List (List STm)
  | [], e => [e]
  | p :: R, e => (List.range p.2).flatMap fun v => rsplit R (e.map (subTm p.1 v))

theorem subTm_den0 (ρ : Val) (x v : ℕ) (t : STm) (hs : t.sums = []) (ν : ℕ → ℤ)
    (hv : ν x = v) : (subTm x v t).den ρ ν = t.den ρ ν := by
  obtain ⟨c, sums, ds, ps⟩ := t
  simp only at hs
  subst hs
  simp only [STm.den, subTm, sumsV, body_eq]
  rw [bodyOf_sub, ipDen_C, ← hv, Function.update_eq_self]

theorem rsplit_zero (ρ : Val) : ∀ (R : List (ℕ × ℕ)) (e : List STm), (∀ t ∈ e, t.sums = []) →
    (∀ c ∈ rsplit R e, ∀ ν, exDen ρ ν c = 0) → ∀ ν, InLab R ν → exDen ρ ν e = 0
  | [], e, _, h, ν, _ => h e (by simp [rsplit]) ν
  | p :: R, e, hs, h, ν, hν => by
    obtain ⟨h0, h1⟩ := hν p List.mem_cons_self
    have hv : ν p.1 = ((ν p.1).toNat : ℕ) := (Int.toNat_of_nonneg h0).symm
    have hvlt : (ν p.1).toNat < p.2 := by omega
    have he : exDen ρ ν e = exDen ρ ν (e.map (subTm p.1 (ν p.1).toNat)) := by
      unfold exDen
      rw [List.map_map]
      congr 1
      exact List.map_congr_left fun t ht => (subTm_den0 ρ p.1 _ t (hs t ht) ν hv).symm
    rw [he]
    refine rsplit_zero ρ R _ (fun t ht => ?_) (fun c hc => h c ?_) ν
      (fun q hq => hν q (List.mem_cons_of_mem _ hq))
    · obtain ⟨t', ht', rfl⟩ := List.mem_map.1 ht
      exact hs t' ht'
    · simp only [rsplit]
      exact List.mem_flatMap.2 ⟨_, List.mem_range.2 hvlt, hc⟩

/-! ## The check -/

/-- A pruned case: nothing, or characters vanishing by `fzOK`. -/
def caseR (nc : ℕ × FzCert) : List STm → Bool
  | [] => true
  | r@(_ :: _) => fzOK nc.1 nc.2 r

theorem caseR_zero (ρ : Val) (hρ : ρ.OK) (nc : ℕ × FzCert) (r : List STm)
    (h : caseR nc r = true) (ν : ℕ → ℤ) : exDen ρ ν r = 0 := by
  cases r with
  | nil => rfl
  | cons t r => exact fz_zero ρ hρ nc.1 nc.2 _ h ν

/-- Each sum-free case, its deltas with constant arguments evaluated (`pruneTm`), against its
modulus and certificate. -/
def allCases : List (List STm) → List (ℕ × FzCert) → Bool
  | [], [] => true
  | c :: cs, n :: ns =>
    c.all (fun t => t.sums.isEmpty) && caseR n (c.filterMap pruneTm) && allCases cs ns
  | _, _ => false

theorem allCases_ok (ρ : Val) (hρ : ρ.OK) :
    ∀ (cs : List (List STm)) (ns : List (ℕ × FzCert)), allCases cs ns = true →
      ∀ c ∈ cs, ∀ ν, exDen ρ ν c = 0
  | [], _, _, c, hc, _ => by simp at hc
  | c :: cs, [], h, _, _, _ => by simp [allCases] at h
  | c :: cs, n :: ns, h, c', hc, ν => by
    simp only [allCases, Bool.and_eq_true, List.all_eq_true, List.isEmpty_iff] at h
    rcases List.mem_cons.1 hc with rfl | hc
    · rw [← exDen_prune ρ c' h.1.1 ν]
      exact caseR_zero ρ hρ n _ h.1.2 ν
    · exact allCases_ok ρ hρ cs ns h.2 c' hc ν

/-- Fuel of the pins, the labels with their ranges, and each case's modulus and certificate. -/
abbrev RgCert := ℕ × List (ℕ × ℕ) × List (ℕ × FzCert)

/-- The pinned residual is sum-free and every case of the labels vanishes. -/
def rgGo (rc : RgCert) (e : List STm) : Bool :=
  e.all (fun t => t.sums.isEmpty) && allCases (rsplit rc.2.1 e) rc.2.2

/-- `rgGo` after the pins. -/
def rgOK (rc : RgCert) (e : List STm) : Bool := rgGo rc (e.map (elimAllR rc.1))

theorem rgOK_zero (ρ : Val) (hρ : ρ.OK) (rc : RgCert) (e : List STm) (h : rgOK rc e = true)
    (ν : ℕ → ℤ) (hν : InLab rc.2.1 ν) : exDen ρ ν e = 0 := by
  unfold rgOK rgGo at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.isEmpty_iff] at h
  rw [← exDen_elimAllR ρ hρ rc.1 e ν]
  exact rsplit_zero ρ rc.2.1 _ h.1 (allCases_ok ρ hρ _ _ h.2) ν hν

/-- `nf`, or nothing when `rgOK` holds of its result. -/
def rgNf (rc : RgCert) (nf : List STm → List STm) (e : List STm) : List STm :=
  if rgOK rc (nf e) then [] else nf e

theorem rgNf_den (ρ : Val) (hρ : ρ.OK) (rc : RgCert)
    (nf : List STm → List STm) (hnf : ∀ e ν, exDen ρ ν (nf e) = exDen ρ ν e) (e : List STm)
    (ν : ℕ → ℤ) (hν : InLab rc.2.1 ν) : exDen ρ ν (rgNf rc nf e) = exDen ρ ν e := by
  unfold rgNf
  split_ifs with h
  · rw [← hnf e ν, rgOK_zero ρ hρ rc _ h ν hν]; rfl
  · exact hnf e ν

/-! ## Ranges of boundary labels -/

/-- Each pair `(l, D)` of `R`: `l` a boundary label whose dimension is the constant `D`. -/
def labOK (labs : List (ℕ × ℕ)) (IX : List ℕ) (dt : ℕ → DMon) (bnd : List ℕ)
    (R : List (ℕ × ℕ)) : Bool :=
  R.all fun p => bnd.contains p.1 && IX.contains ((labs.lookup p.1).getD 0) &&
    (dt ((labs.lookup p.1).getD 0)).isConst && (dt ((labs.lookup p.1).getD 0)).c == p.2

theorem labOK_in (E : Env) (ρ : Val) (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (IX : List ℕ)
    (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv) (hL : E.labs = labs) (bnd : List ℕ)
    (R : List (ℕ × ℕ)) (h : labOK labs IX dt bnd R = true) (σ : ℕ → ℕ)
    (hσ : InRange E bnd σ) : InLab R (fun x => (σ x : ℤ)) := by
  intro p hp
  unfold labOK at h
  have hq := List.all_eq_true.1 h p hp
  simp only [Bool.and_eq_true, List.contains_iff_mem, beq_iff_eq] at hq
  obtain ⟨⟨⟨hb, hi⟩, hk⟩, hc⟩ := hq
  refine ⟨by positivity, ?_⟩
  have hlt := hσ p.1 hb
  unfold labDim at hlt
  rw [hL, hE _ hi, DMon.den_isConst _ _ hk, hc] at hlt
  show (σ p.1 : ℤ) < p.2
  exact_mod_cast hlt

/-! ## Equality -/

/-- `checkWith` with `rgNf` around the normaliser. -/
def checkWithR (rc : RgCert) (tr : Net → Option (List STm)) (fl : List ℕ) (N0 N1 : Net)
    (S0 S1 : List STm) : Bool :=
  match tr N0, tr N1 with
  | some e0, some e1 => exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
      (rgNf rc (nfEx fl) (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty
  | _, _ => false

/-- `symCheck` with `checkWithR` and the labels' ranges by `labOK`. -/
def symCheckR (rc : RgCert) (fl : List ℕ) (dt : ℕ → DMon)
    (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (N0 N1 : Net) (S0 S1 : List STm) : Bool :=
  N0.bnd.isPerm N1.bnd && labOK labs IX dt N0.bnd rc.2.1 && netChk fl dt pt rl labs IX N0 &&
    netChk fl dt pt rl labs IX N1 && exConst S0 && exConst S1 &&
    (checkWithR rc (trNet fl dt pt rl labs) fl N0 N1 S0 S1 ||
      checkWithR rc (trNetF dt pt rl labs) fl N0 N1 S0 S1)

theorem equal_of_checkR (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (rc : RgCert) (tr : Net → Option (List STm)) (N0 N1 : Net)
    (hin : ∀ σ, InRange E N0.bnd σ → InLab rc.2.1 (fun x => (σ x : ℤ))) (S0 S1 : List STm) (s0 s1 : ℂ)
    (htr0 : ∀ e, tr N0 = some e → ∀ σ, InRange E N0.bnd σ → eval E N0 σ = exDen ρ (fun x => (σ x : ℤ)) e)
    (htr1 : ∀ e, tr N1 = some e → ∀ σ, InRange E N1.bnd σ → eval E N1 σ = exDen ρ (fun x => (σ x : ℤ)) e)
    (hp : N0.bnd.Perm N1.bnd) (hk0 : exConst S0 = true) (hk1 : exConst S1 = true)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : checkWithR rc tr fl N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold checkWithR at h
  split at h
  · next e0 e1 ht0 ht1 =>
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    obtain ⟨⟨hm0, hm1⟩, hnf⟩ := h
    refine ⟨hp, fun σ hσ => ?_⟩
    set ν : ℕ → ℤ := fun x => (σ x : ℤ)
    have hz : exDen ρ ν (exMul S0 e0 ++ exMul (exNeg S1) e1) = 0 := by
      rw [← rgNf_den ρ hρ rc (nfEx fl) (nfEx_den ρ hρ fl hfl) _ ν (hin σ hσ), hnf]; rfl
    rw [exDen_append, exDen_mul ρ hρ _ _ hm0, exDen_mul ρ hρ _ _ hm1, exDen_neg,
      ← htr0 e0 ht0 σ hσ, ← htr1 e1 ht1 σ (hσ.perm hp),
      exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
      ← hs1] at hz
    linear_combination hz
  · simp at h

theorem equal_symR (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (rc : RgCert)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : symCheckR rc fl dt pt rl labs IX N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold symCheckR at h
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨hperm, hlab⟩, hc0⟩, hc1⟩, hk0⟩, hk1⟩, hrest⟩ := h
  have hp : N0.bnd.Perm N1.bnd := List.isPerm_iff.1 hperm
  have hin := labOK_in E ρ dt labs IX hE hL N0.bnd rc.2.1 hlab
  rcases hrest with hb | hf
  · exact equal_of_checkR E ρ hρ fl hfl rc _ N0 N1 hin S0 S1 s0 s1
      (fun e ht σ hσ => eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e hc0 ht σ hσ)
      (fun e ht σ hσ => eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e hc1 ht σ hσ)
      hp hk0 hk1 hs0 hs1 hb
  · exact equal_of_checkR E ρ hρ fl hfl rc _ N0 N1 hin S0 S1 s0 s1
      (fun e ht σ hσ => eval_trF E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e hc0 ht σ hσ)
      (fun e ht σ hσ => eval_trF E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e hc1 ht σ hσ)
      hp hk0 hk1 hs0 hs1 hf

/-- `equal_symS` with a normaliser of the difference exact where `labOK` gives the ranges. -/
theorem equal_symSR (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (nf0 nf1 : ℕ → List STm → List STm) (hnf0 : ∀ i e ν, exDen ρ ν (nf0 i e) = exDen ρ ν e)
    (hnf1 : ∀ i e ν, exDen ρ ν (nf1 i e) = exDen ρ ν e) (e0 e1 : List STm)
    (h0 : trNetG nf0 dt pt rl labs N0 = some e0) (h1 : trNetG nf1 dt pt rl labs N1 = some e1)
    (hp : N0.bnd.isPerm N1.bnd = true) (hc0 : netChk fl dt pt rl labs IX N0 = true)
    (hc1 : netChk fl dt pt rl labs IX N1 = true) (hk : (exConst S0 && exConst S1) = true)
    (hm : (exMulOK S0 e0 && exMulOK (exNeg S1) e1) = true) (R : List (ℕ × ℕ))
    (nfz : List STm → List STm) (hnfz : ∀ e ν, InLab R ν → exDen ρ ν (nfz e) = exDen ρ ν e)
    (hz : (nfz (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty = true)
    (hR : labOK labs IX dt N0.bnd R = true) :
    Equal E s0 N0 s1 N1 := by
  simp only [Bool.and_eq_true] at hk hm
  obtain ⟨hk0, hk1⟩ := hk
  obtain ⟨hm0, hm1⟩ := hm
  have hp' : N0.bnd.Perm N1.bnd := List.isPerm_iff.1 hp
  refine ⟨hp', fun σ hσ => ?_⟩
  set ν : ℕ → ℤ := fun x => (σ x : ℤ)
  have hz' : exDen ρ ν (exMul S0 e0 ++ exMul (exNeg S1) e1) = 0 := by
    rw [← hnfz _ ν (labOK_in E ρ dt labs IX hE hL N0.bnd R hR σ hσ), List.isEmpty_iff.1 hz]
    simp [exDen]
  rw [exDen_append, exDen_mul ρ hρ _ _ hm0, exDen_mul ρ hρ _ _ hm1, exDen_neg,
    ← eval_trG E ρ hρ fl hfl nf0 hnf0 dt pt rl labs IX hE hP hL N0 e0 hc0 h0 σ hσ,
    ← eval_trG E ρ hρ fl hfl nf1 hnf1 dt pt rl labs IX hE hP hL N1 e1 hc1 h1 σ (hσ.perm hp'),
    exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
    ← hs1] at hz'
  linear_combination hz'

/-! ## Merging by fingerprint -/

/-- One integer folded into a fingerprint. -/
def fpZ (h : ℕ) (z : ℤ) : ℕ :=
  (h * 1000003 + (2 * z.natAbs + if z < 0 then 1 else 0)) % 2305843009213693951

/-- A dimension monomial folded into a fingerprint. -/
def fpDM (h : ℕ) (m : DMon) : ℕ := m.e.foldl (fun h n => fpZ h n) (fpZ h m.c)

/-- A factor folded into a fingerprint. -/
def fpFac (h : ℕ) (f : Fac) : ℕ :=
  f.2.foldl (fun h t => fpZ (t.1.foldl (fun h n => fpZ h n) h) t.2) (fpDM h f.1)

/-- A fingerprint of a term's sums and factors. -/
def fpTm (t : STm) : ℕ :=
  t.phs.foldl fpFac (t.dels.foldl fpFac (t.sums.foldl (fun h p => fpDM (fpZ h p.1) p.2) 0))

/-- `tmAdd` over terms stored with their fingerprints, `k` that of `t`. -/
def tmAddP (k : ℕ) (t : STm) : List (ℕ × STm) → List (ℕ × STm)
  | [] => if t.c.q = 0 then [] else [(k, t)]
  | u :: l =>
    if u.1 = k ∧ shape u.2 = shape t then
      (if qAdd u.2.c.q t.c.q = 0 then l else (u.1, withQ u.2 (qAdd u.2.c.q t.c.q)) :: l)
    else u :: tmAddP k t l

/-- Every stored fingerprint is its term's. -/
def FpOK (acc : List (ℕ × STm)) : Prop := ∀ p ∈ acc, p.1 = fpTm p.2

theorem fpTm_shape (u t : STm) (h : shape u = shape t) : fpTm u = fpTm t := by
  have h1 := congrArg STm.sums h
  have h2 := congrArg STm.dels h
  have h3 := congrArg STm.phs h
  simp only [shape] at h1 h2 h3
  simp only [fpTm, h1, h2, h3]

theorem tmAddP_eq (t : STm) : ∀ acc : List (ℕ × STm), FpOK acc →
    (tmAddP (fpTm t) t acc).map (·.2) = tmAdd t (acc.map (·.2)) ∧ FpOK (tmAddP (fpTm t) t acc)
  | [], _ => by
    unfold tmAddP tmAdd
    split_ifs
    · exact ⟨rfl, fun p hp => by simp at hp⟩
    · refine ⟨rfl, fun p hp => ?_⟩
      simp only [List.mem_singleton] at hp
      subst hp; rfl
  | u :: l, hok => by
    have hu : u.1 = fpTm u.2 := hok u List.mem_cons_self
    have hl : FpOK l := fun p hp => hok p (List.mem_cons_of_mem _ hp)
    have hiff : (u.1 = fpTm t ∧ shape u.2 = shape t) ↔ shape u.2 = shape t :=
      ⟨fun h => h.2, fun h => ⟨hu.trans (fpTm_shape _ _ h), h⟩⟩
    simp only [tmAddP, List.map_cons, tmAdd]
    by_cases hs : shape u.2 = shape t
    · rw [if_pos (hiff.2 hs), if_pos hs]
      split_ifs
      · exact ⟨rfl, hl⟩
      · refine ⟨rfl, fun p hp => ?_⟩
        rcases List.mem_cons.1 hp with rfl | hp
        · exact hu
        · exact hl p hp
    · rw [if_neg (fun h => hs (hiff.1 h)), if_neg hs]
      obtain ⟨h1, h2⟩ := tmAddP_eq t l hl
      refine ⟨by simp only [List.map_cons, h1], fun p hp => ?_⟩
      rcases List.mem_cons.1 hp with rfl | hp
      · exact hu
      · exact h2 p hp

/-- `nfGoH` with `tmAddP`. -/
def nfGoP (fl : List ℕ) : List THint → List (ℕ × STm) → List STm → List (ℕ × STm)
  | _, acc, [] => acc
  | hs, acc, t :: ts =>
    match normTmK fl (hs.headD (0, 0, [])).1 (hs.headD (0, 0, [])).2.1
      (hs.headD (0, 0, [])).2.2 t with
    | none => nfGoP fl hs.tail acc ts
    | some t' => nfGoP fl hs.tail (tmAddP (fpTm t') t' acc) ts

theorem nfGoP_eq (fl : List ℕ) : ∀ (e : List STm) (hs : List THint) (acc : List (ℕ × STm)),
    FpOK acc → (nfGoP fl hs acc e).map (·.2) = nfGoH fl hs (acc.map (·.2)) e
  | [], _, _, _ => rfl
  | t :: ts, hs, acc, hok => by
    simp only [nfGoP, nfGoH]
    cases normTmK fl (hs.headD (0, 0, [])).1 (hs.headD (0, 0, [])).2.1
        (hs.headD (0, 0, [])).2.2 t with
    | none => exact nfGoP_eq fl ts hs.tail acc hok
    | some t' =>
      obtain ⟨h1, h2⟩ := tmAddP_eq t' acc hok
      rw [nfGoP_eq fl ts hs.tail _ h2, h1]

/-- `nfH` merging by fingerprint. -/
def nfHP (fl : List ℕ) (H : ℕ → List THint) (i : ℕ) (e : List STm) : List STm :=
  (nfGoP fl (H i) [] e).map (·.2)

theorem nfHP_eq (fl : List ℕ) (H : ℕ → List THint) (i : ℕ) (e : List STm) :
    nfHP fl H i e = nfH fl H i e :=
  nfGoP_eq fl e (H i) [] (fun p hp => by simp at hp)

theorem nfHP_den (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (H : ℕ → List THint) (i : ℕ) (e : List STm) (ν : ℕ → ℤ) :
    exDen ρ ν (nfHP fl H i e) = exDen ρ ν e := by
  rw [nfHP_eq]; exact nfH_den ρ hρ fl hfl H i e ν

/-! ## Geometric sums -/

theorem ipHas_smul (x : ℕ) (k : ℤ) (p : IPoly) (h : ipHas x p = false) :
    ipHas x (ipSmul k p) = false := by
  unfold ipSmul
  split_ifs
  · rfl
  · simpa [ipHas, List.any_map, Function.comp_def] using h

/-- The roots of unity reading `x`, each `ω_m^(a x + b)` with `m ∣ L`, over `ω_L`: the sums of
`(L / m) a` and of `(L / m) b`. -/
def geoData (x L : ℕ) : List Fac → Option (IPoly × IPoly)
  | [] => some ([], [])
  | f :: l =>
    match geoData x L l with
    | none => none
    | some (A, B) =>
      if facHas x f then
        if f.1.isConst && decide (0 < f.1.c) && L % f.1.c == 0 then
          match ipLin x f.2 with
          | some (a, b) => some (ipAdd A (ipSmul ((L / f.1.c : ℕ) : ℤ) a),
              ipAdd B (ipSmul ((L / f.1.c : ℕ) : ℤ) b))
          | none => none
        else none
      else some (A, B)

theorem geoData_spec (dv : ℕ → ℕ) (x L : ℕ) (hL : 0 < L) (l : List Fac) (A B : IPoly)
    (h : geoData x L l = some (A, B)) :
    ipHas x A = false ∧ ipHas x B = false ∧ ∀ (ν : ℕ → ℤ) (j : ℤ),
      (l.map (phV dv (Function.update ν x j))).prod =
        ((l.filter (fun f => !facHas x f)).map (phV dv ν)).prod *
          om L ^ (j * ipDen ν A + ipDen ν B) := by
  have hz : om L ≠ 0 := om_ne_zero _ hL
  induction l generalizing A B with
  | nil =>
    simp only [geoData, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [ipHas, ipDen]
  | cons f l ih =>
    simp only [geoData] at h
    split at h
    · simp at h
    · next A' B' hl =>
      obtain ⟨hA', hB', hrec⟩ := ih A' B' hl
      split_ifs at h with hx hfc
      · split at h
        · next a b hlin =>
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hfc
          obtain ⟨⟨hk, hpos⟩, hdiv⟩ := hfc
          obtain ⟨ha, hb, hp⟩ := ipLin_spec x f.2 a b hlin
          refine ⟨ipHas_add _ _ _ hA' (ipHas_smul _ _ _ ha),
            ipHas_add _ _ _ hB' (ipHas_smul _ _ _ hb), fun ν j => ?_⟩
          have hm : f.1.den dv = f.1.c := DMon.den_isConst _ _ hk
          simp only [List.map_cons, List.prod_cons, List.filter_cons, hx, Bool.not_true,
            Bool.false_eq_true, ite_false, hrec, ipDen_add, ipDen_smul]
          unfold phV
          rw [hp, ipDen_upd_of _ _ _ _ ha, ipDen_upd_of _ _ _ _ hb, Function.update_self, hm,
            om_up L f.1.c hL hdiv, mul_left_comm, ← zpow_add₀ hz]
          congr 2
          ring
        · simp at h
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨hA', hB', fun ν j => ?_⟩
        simp only [Bool.not_eq_true] at hx
        simp only [List.map_cons, List.prod_cons, List.filter_cons, hx, Bool.not_false,
          ite_true, hrec]
        unfold phV
        rw [ipDen_upd_of _ _ _ _ hx]; ring

/-- `x` of constant range `d`, read by no delta and by roots of unity over moduli dividing
`L = K d` with exponent `A x + B` over `L` (`geoData`), beside the delta `f = [K ∣ P]` with
`P ≡ ±A mod K`: `d [L ∣ A] ω_L^B`. -/
def elimGeo1 (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option Elim :=
  if D.isConst && decide (0 < D.c) && f.1.isConst && decide (0 < f.1.c) &&
      (pre ++ f :: post).all (fun g => !facHas x g) then
    match geoData x (f.1.c * D.c) ps with
    | some (A, B) =>
      if ipModC f.1.c f.2 == ipModC f.1.c A ||
          ipModC f.1.c (ipSmul (-1) f.2) == ipModC f.1.c A then
        some ⟨CMon.rtD D 2, pre ++ (⟨f.1.c * D.c, []⟩, A) :: post,
          ps.filter (fun g => !facHas x g) ++ [(⟨f.1.c * D.c, []⟩, B)]⟩
      else none
    | none => none
  else none

theorem dvd_iff_of_modC (ν : ℕ → ℤ) (K : ℕ) (p a : IPoly)
    (h : (ipModC K p == ipModC K a || ipModC K (ipSmul (-1) p) == ipModC K a) = true) :
    ((K : ℤ) ∣ ipDen ν p ↔ (K : ℤ) ∣ ipDen ν a) := by
  have h2 := ipDen_modC ν K a
  simp only [Bool.or_eq_true, beq_iff_eq] at h
  have key : ∀ q : IPoly, ipModC K q = ipModC K a → (K : ℤ) ∣ ipDen ν q - ipDen ν a := by
    intro q hq
    have h1 := ipDen_modC ν K q
    rw [hq] at h1
    have := dvd_sub h1 h2
    rwa [show ipDen ν q - ipDen ν (ipModC K a) - (ipDen ν a - ipDen ν (ipModC K a)) =
      ipDen ν q - ipDen ν a by ring] at this
  rcases h with hm | hm
  · have hPA := key _ hm
    exact ⟨fun hp => by have := dvd_sub hp hPA; rwa [sub_sub_cancel] at this,
      fun ha => by have := dvd_add hPA ha; rwa [sub_add_cancel] at this⟩
  · have hPA := key _ hm
    rw [ipDen_smul] at hPA
    constructor
    · intro hp
      have := dvd_sub (dvd_neg.2 hp) hPA
      rwa [show -ipDen ν p - (-1 * ipDen ν p - ipDen ν a) = ipDen ν a by ring] at this
    · intro ha
      have := dvd_add hPA ha
      rw [show -1 * ipDen ν p - ipDen ν a + ipDen ν a = -ipDen ν p by ring] at this
      exact dvd_neg.1 this

theorem elimGeo1_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac)
    (post ps : List Fac) (e : Elim) (h : elimGeo1 x D pre f post ps = some e) :
    ElimOK ρ x D (pre ++ f :: post) ps e := by
  unfold elimGeo1 at h
  split_ifs at h with hc
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨hD, hd⟩, hK⟩, hKpos⟩, hfree⟩ := hc
  split at h
  · next A B hg =>
    split_ifs at h with hm
    simp only [Option.some.injEq] at h
    subst h
    have hL : 0 < f.1.c * D.c := Nat.mul_pos hKpos hd
    obtain ⟨hA, hB, hrec⟩ := geoData_spec ρ.dv x (f.1.c * D.c) hL ps A B hg
    have hDd : D.den ρ.dv = D.c := DMon.den_isConst _ _ hD
    have hKd : f.1.den ρ.dv = f.1.c := DMon.den_isConst _ _ hK
    intro ν
    have hdel : ∀ j : ℤ, ((pre ++ f :: post).map (delV ρ.dv (Function.update ν x j))).prod =
        ((pre ++ f :: post).map (delV ρ.dv ν)).prod := by
      intro j
      have := bodyOf_upd ρ.dv x (pre ++ f :: post) [] (by simpa using hfree) ν j
      simpa [bodyOf] using this
    have hmod := dvd_iff_of_modC ν f.1.c f.2 A hm
    have hsum : ∑ j ∈ range (D.den ρ.dv), bodyOf ρ.dv (pre ++ f :: post) ps
        (Function.update ν x j) = ((pre ++ f :: post).map (delV ρ.dv ν)).prod *
          ((ps.filter (fun f => !facHas x f)).map (phV ρ.dv ν)).prod *
          ∑ j ∈ range D.c, om (f.1.c * D.c) ^ ((j : ℤ) * ipDen ν A + ipDen ν B) := by
      rw [hDd, Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [bodyOf, hdel, hrec]
      ring
    rw [hsum, rtD_two ρ D hd, hDd]
    have hcm := bodyOf_cons_mid ρ.dv pre f post [] ν
    have hcm2 := bodyOf_cons_mid ρ.dv pre (⟨f.1.c * D.c, []⟩, A) post [] ν
    simp only [bodyOf, List.map_nil, List.prod_nil, mul_one] at hcm hcm2
    have hLd : DMon.den ρ.dv ⟨f.1.c * D.c, []⟩ = f.1.c * D.c := by simp [DMon.den, npow]
    simp only [bodyOf]
    rw [hcm, hcm2]
    simp only [List.map_append, List.prod_append, List.map_cons, List.map_nil, List.prod_cons,
      List.prod_nil, mul_one, phV, hLd]
    by_cases hP : (f.1.c : ℤ) ∣ ipDen ν f.2
    · obtain ⟨t, ht⟩ := hmod.1 hP
      have h1 : delV ρ.dv ν f = 1 := by simp [delV, hKd, hP]
      have h2 : delV ρ.dv ν (⟨f.1.c * D.c, []⟩, A) = if (D.c : ℤ) ∣ t then 1 else 0 := by
        have hdvd : ((f.1.c : ℤ) * D.c ∣ f.1.c * t) ↔ ((D.c : ℤ) ∣ t) :=
          mul_dvd_mul_iff_left (by exact_mod_cast hKpos.ne')
        simp only [delV, hLd, ht]
        push_cast
        simp only [hdvd]
      have hz : om (f.1.c * D.c) ≠ 0 := om_ne_zero _ hL
      have h3 : ∀ j : ℕ, om (f.1.c * D.c) ^ ((j : ℤ) * ipDen ν A + ipDen ν B) =
          om (f.1.c * D.c) ^ ipDen ν B * om D.c ^ ((j : ℤ) * t + 0) := by
        intro j
        rw [om_up (f.1.c * D.c) D.c hL (by simp), Nat.mul_div_cancel _ hd, ht, ← zpow_add₀ hz]
        congr 1
        ring
      rw [Finset.sum_congr rfl fun j _ => h3 j, ← Finset.mul_sum, sum_om_lin _ hd, h1, h2]
      simp only [hLd, zpow_zero, one_mul]
      split_ifs <;> ring
    · have h1 : delV ρ.dv ν f = 0 := by simp [delV, hKd, hP]
      have h2 : delV ρ.dv ν (⟨f.1.c * D.c, []⟩, A) = 0 := by
        simp only [delV, hLd]
        rw [if_neg]
        intro hdv
        exact hP (hmod.2 (dvd_trans (by push_cast; exact dvd_mul_right _ _) hdv))
      rw [h1, h2]
      ring
  · simp at h

/-! ## Pins by the ranges of other variables -/

/-- The interval of a linear index polynomial whose variables have positive constant ranges in
`rg`. -/
def ivl (rg : List (ℕ × DMon)) : IPoly → Option (ℤ × ℤ)
  | [] => some (0, 0)
  | (m, c) :: p =>
    match ivl rg p with
    | none => none
    | some (lo, hi) =>
      match m with
      | [] => some (lo + c, hi + c)
      | [u] =>
        match rg.lookup u with
        | some Dm =>
          if Dm.isConst && decide (0 < Dm.c) then
            some (lo + min 0 (c * ((Dm.c - 1 : ℕ) : ℤ)), hi + max 0 (c * ((Dm.c - 1 : ℕ) : ℤ)))
          else none
        | none => none
      | _ => none

theorem lookup_mem (rg : List (ℕ × DMon)) (u : ℕ) (Dm : DMon) (h : rg.lookup u = some Dm) :
    (u, Dm) ∈ rg := by
  induction rg with
  | nil => simp at h
  | cons p rg ih =>
    obtain ⟨a, b⟩ := p
    simp only [List.lookup_cons] at h
    split at h
    · next he =>
      simp only [Option.some.injEq] at h
      subst h
      have : u = a := by simpa using he
      subst this
      exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (ih h)

theorem ivl_spec (dv : ℕ → ℕ) (rg : List (ℕ × DMon)) (ν : ℕ → ℤ) (hν : InR dv rg ν) :
    ∀ (p : IPoly) (lo hi : ℤ), ivl rg p = some (lo, hi) → lo ≤ ipDen ν p ∧ ipDen ν p ≤ hi
  | [], lo, hi, h => by
    simp only [ivl, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [ipDen]
  | (m, c) :: p, lo, hi, h => by
    simp only [ivl] at h
    split at h
    · simp at h
    · next lo' hi' hp =>
      obtain ⟨h1, h2⟩ := ivl_spec dv rg ν hν p lo' hi' hp
      rw [ipDen_cons]
      split at h
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp only [imDen, List.map_nil, List.prod_nil, mul_one]
        omega
      · next u =>
        split at h
        · next Dm hl =>
          split_ifs at h with hc
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
          obtain ⟨hk, hpos⟩ := hc
          have hr : 0 ≤ ν u ∧ ν u < (Dm.c : ℤ) := by
            have := hν _ (lookup_mem rg u Dm hl)
            simpa [DMon.den_isConst _ _ hk] using this
          simp only [imDen, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
            mul_one]
          obtain ⟨hr0, hr1⟩ := hr
          have hdc : ((Dm.c - 1 : ℕ) : ℤ) = (Dm.c : ℤ) - 1 := by omega
          rw [hdc]
          have hb : ν u ≤ (Dm.c : ℤ) - 1 := by omega
          rcases le_or_gt 0 c with hc0 | hc0
          · have e1 : c * ν u ≤ c * ((Dm.c : ℤ) - 1) := mul_le_mul_of_nonneg_left hb hc0
            have e2 : 0 ≤ c * ν u := mul_nonneg hc0 hr0
            have e3 : 0 ≤ c * ((Dm.c : ℤ) - 1) := mul_nonneg hc0 (by omega)
            rw [min_eq_left e3, max_eq_right e3]
            constructor <;> linarith
          · have e1 : c * ((Dm.c : ℤ) - 1) ≤ c * ν u :=
              mul_le_mul_of_nonpos_left hb hc0.le
            have e2 : c * ν u ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hc0.le hr0
            have e3 : c * ((Dm.c : ℤ) - 1) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hc0.le (by omega)
            rw [min_eq_right e3, max_eq_left e3]
            constructor <;> linarith
        · simp at h
      · simp at h

/-- Summing `x` over `D` out of `ds`, `ps` gives `e` wherever the ranges `rg` hold. -/
def ElimOKR (ρ : Val) (rg : List (ℕ × DMon)) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim) :
    Prop :=
  ∀ ν, InR ρ.dv rg ν → ∑ j ∈ range (D.den ρ.dv), bodyOf ρ.dv ds ps (Function.update ν x j) =
    e.k.den ρ * bodyOf ρ.dv e.ds e.ps ν

/-- `x` of constant range `D` held by `f = [D ∣ s x + b]`, `s = ±1`, with `V = -s b` in
`[k D, k D + D)` by the ranges `rg`: `x = V - k D`. -/
def elimXpin1 (rg : List (ℕ × DMon)) (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac)
    (post ps : List Fac) : Option Elim :=
  if f.1 = D ∧ D.isConst = true ∧ 0 < D.c then
    match ipLin x f.2 with
    | some (a, b) =>
      match ipConst a with
      | some s =>
        if s = 1 ∨ s = -1 then
          match ivl rg (pinVal s b) with
          | some (lo, hi) =>
            if hi < lo / (D.c : ℤ) * D.c + D.c then
              some ⟨CMon.ofQ 1,
                (pre ++ post).map (facSub x (ipAdd (pinVal s b) (ipC (-(lo / (D.c : ℤ) * D.c))))),
                ps.map (facSub x (ipAdd (pinVal s b) (ipC (-(lo / (D.c : ℤ) * D.c)))))⟩
            else none
          | none => none
        else none
      | none => none
    | none => none
  else none

theorem elimXpin1_ok (ρ : Val) (rg : List (ℕ × DMon)) (x : ℕ) (D : DMon) (pre : List Fac)
    (f : Fac) (post ps : List Fac) (e : Elim) (h : elimXpin1 rg x D pre f post ps = some e) :
    ElimOKR ρ rg x D (pre ++ f :: post) ps e := by
  unfold elimXpin1 at h
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
        simp only [Option.some.injEq] at h
        subst h
        obtain ⟨_, hb, hp⟩ := ipLin_spec x f.2 a b hlin
        intro ν hν
        obtain ⟨hlo, hhi⟩ := ivl_spec ρ.dv rg ν hν _ lo hi hiv
        have hDd : D.den ρ.dv = D.c := DMon.den_isConst ρ.dv D hk
        set V := ipDen ν (pinVal s b) with hVdef
        set k := lo / (D.c : ℤ)
        have hDz : (0 : ℤ) < D.c := by exact_mod_cast hpos
        have hkl : k * D.c ≤ lo := Int.ediv_mul_le lo hDz.ne'
        set w : ℤ := V - k * D.c with hwdef
        have hw0 : 0 ≤ w := by omega
        have hw1 : w < D.c := by omega
        have hwm : V % (D.c : ℤ) = w := by
          rw [show V = w + k * D.c by omega, Int.add_mul_emod_self_right]
          exact Int.emod_eq_of_lt hw0 hw1
        have hdel : ∀ j : ℕ, j < D.c →
            delV ρ.dv (Function.update ν x (j : ℤ)) f = if j = w.toNat then 1 else 0 := by
          intro j hj
          unfold delV
          rw [hp, ipConst_spec a s hs, ipDen_upd_of _ _ _ _ hb, Function.update_self, hf, hDd]
          have e1 := pin_dvd s hs1 (D.c : ℤ) (j : ℤ) (ipDen ν b)
          have e2 := dvd_sub_iff_eq D.c hpos j hj V
          have hV : -s * ipDen ν b = V := by simp [hVdef, pinVal, ipDen_smul]
          rw [hV] at e1
          rw [hwm] at e2
          by_cases hj' : j = w.toNat
          · rw [if_pos (e1.2 (e2.2 hj')), if_pos hj']
          · rw [if_neg (fun hd => hj' (e2.1 (e1.1 hd))), if_neg hj']
        have hwlt : w.toNat < D.c := by omega
        rw [hDd, Finset.sum_congr rfl fun j hj => by
          rw [bodyOf_cons_mid, hdel j (Finset.mem_range.1 hj), ite_mul, one_mul, zero_mul]]
        rw [Finset.sum_ite_eq' (range D.c) w.toNat, if_pos (Finset.mem_range.2 hwlt), ofQ_den,
          bodyOf_sub]
        have hval : ipDen ν (ipAdd (pinVal s b) (ipC (-(lo / (D.c : ℤ) * D.c)))) = w.toNat := by
          rw [ipDen_add, ipDen_C, Int.toNat_of_nonneg hw0]
          simp only [hwdef, k, ← hVdef]
          ring
        rw [hval]
        push_cast; ring
      · simp at h
    · simp at h
  · simp at h

/-! ## Hinted steps with the labels' ranges -/

/-- The ranges a pin may read beside the sums `S`: `S`, then the labels of `R` no sum of `S`
binds. -/
def rgOf (R : List (ℕ × ℕ)) (S : List (ℕ × DMon)) : List (ℕ × DMon) :=
  S ++ (R.filter fun p => !(S.map (·.1)).contains p.1).map fun p => (p.1, ⟨p.2, []⟩)

/-- Rule `5`: `elimGeo1`; else `elimXpin1` with the ranges `rgOf R S`. -/
def elimXR (R : List (ℕ × ℕ)) (r : ℕ) (S : List (ℕ × DMon)) (x : ℕ) (D : DMon)
    (pre : List Fac) (f : Fac) (post ps : List Fac) : Option Elim :=
  if r = 5 then elimGeo1 x D pre f post ps else elimXpin1 (rgOf R S) x D pre f post ps

theorem elimXR_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (r : ℕ) (S : List (ℕ × DMon))
    (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac) (post ps : List Fac) (e : Elim)
    (h : elimXR R r S x D pre f post ps = some e) :
    ElimOKR ρ (rgOf R S) x D (pre ++ f :: post) ps e := by
  unfold elimXR at h
  split_ifs at h
  · intro ν _; exact elimGeo1_ok ρ hρ x D pre f post ps e h ν
  · exact elimXpin1_ok ρ _ x D pre f post ps e h

/-- The step `(i, r, j)`: `5` the geometric sum of the sum at `i` beside the delta at `j`, `6`
the pin by ranges of the sum at `i` by the delta at `j`, else `elimStepH`. -/
def elimStepX (R : List (ℕ × ℕ)) (t : STm) (i r j : ℕ) : Option STm :=
  if r = 5 ∨ r = 6 then
    match t.sums.drop i, t.dels.drop j with
    | (x, D) :: post, f :: dpost =>
      if ((t.sums.take i ++ post).map (·.1)).Nodup ∧
          x ∉ (t.sums.take i ++ post).map (·.1) then
        match elimXR R r (t.sums.take i ++ post) x D (t.dels.take j) f dpost t.phs with
        | some e => some ⟨CMon.mul t.c e.k, t.sums.take i ++ post, e.ds, e.ps⟩
        | none => none
      else none
    | _, _ => none
  else elimStepH t i r j

theorem rgOf_in (dv : ℕ → ℕ) (R : List (ℕ × ℕ)) (S : List (ℕ × DMon))
    (hnd : (S.map (·.1)).Nodup) (ν : ℕ → ℤ) (hν : InLab R ν) :
    (∀ p ∈ rgOf R S, p.1 ∈ S.map (·.1) → p ∈ S ∧ (S.map (·.1)).count p.1 = 1) ∧
    (∀ p ∈ rgOf R S, p.1 ∉ S.map (·.1) → 0 ≤ ν p.1 ∧ ν p.1 < ((p.2.den dv : ℕ) : ℤ)) := by
  constructor
  · intro p hp hpS
    simp only [rgOf, List.mem_append, List.mem_map, List.mem_filter] at hp
    rcases hp with hp | ⟨q, ⟨_, hq⟩, rfl⟩
    · exact ⟨hp, List.count_eq_one_of_mem hnd hpS⟩
    · have : q.1 ∉ S.map (·.1) := by simpa using hq
      exact absurd hpS this
  · intro p hp hpS
    simp only [rgOf, List.mem_append, List.mem_map, List.mem_filter] at hp
    rcases hp with hp | ⟨q, ⟨hq, _⟩, rfl⟩
    · exact absurd (List.mem_map_of_mem hp) hpS
    · have := hν q hq
      simpa [DMon.den, npow] using this

theorem elimStepX_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (t t' : STm) (i r j : ℕ)
    (h : elimStepX R t i r j = some t') (ν : ℕ → ℤ) (hν : InLab R ν) :
    t'.den ρ ν = t.den ρ ν := by
  unfold elimStepX at h
  split_ifs at h with hr
  · split at h
    · next x D post f dpost hs hd =>
      split_ifs at h with hc
      obtain ⟨hnd, hx⟩ := hc
      split at h
      · next e hev =>
        simp only [Option.some.injEq] at h
        subst h
        have hsum : t.sums = t.sums.take i ++ (x, D) :: post := by
          rw [← hs, List.take_append_drop]
        have hdl : t.dels = t.dels.take j ++ f :: dpost := by rw [← hd, List.take_append_drop]
        have hin : ∀ ν', InR ρ.dv (rgOf R (t.sums.take i ++ post)) ν' →
            ∑ k ∈ range (D.den ρ.dv), bodyOf ρ.dv t.dels t.phs (Function.update ν' x k) =
              e.k.den ρ * bodyOf ρ.dv e.ds e.ps ν' := by
          intro ν' hν'
          rw [hdl]
          exact elimXR_ok ρ hρ R r _ x D _ f dpost t.phs e hev ν' hν'
        obtain ⟨hA, hB⟩ := rgOf_in ρ.dv R (t.sums.take i ++ post) hnd ν hν
        set A := t.sums.take i with hAdef
        unfold STm.den
        rw [hsum, sumsV_pull ρ.dv _ post x D hx]
        simp only [body_eq] at *
        rw [sumsV_congr_rng ρ.dv _ _ _ (fun ν' => e.k.den ρ * bodyOf ρ.dv e.ds e.ps ν') ν hA hB hin,
          sumsV_mul_left, CMon.den_mul ρ hρ]
        ring
      · simp at h
    · simp at h
  · exact elimStepH_ok ρ hρ t t' i r j h ν

/-- `elimStepX` along the hints, each result forced; the term at the first that fails. -/
def elimAllX (R : List (ℕ × ℕ)) : List (ℕ × ℕ × ℕ) → STm → STm
  | [], t => t
  | (i, r, j) :: hs, t =>
    match elimStepX R t i r j with
    | some t' => fTm t' (elimAllX R hs)
    | none => t

theorem elimAllX_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) :
    ∀ (hs : List (ℕ × ℕ × ℕ)) (t : STm) (ν : ℕ → ℤ), InLab R ν →
      (elimAllX R hs t).den ρ ν = t.den ρ ν := by
  intro hs
  induction hs with
  | nil => intro t ν _; rfl
  | cons h hs ih =>
    obtain ⟨i, r, j⟩ := h
    intro t ν hν
    simp only [elimAllX]
    cases he : elimStepX R t i r j with
    | none => rfl
    | some t' => dsimp only; rw [fTm_eq, ih _ ν hν, elimStepX_ok ρ hρ R t t' i r j he ν hν]

/-- `t'` has the value of `t`, or `t` vanishes, wherever the labels of `R` lie in range. -/
def TmResR (ρ : Val) (R : List (ℕ × ℕ)) (t : STm) : Option STm → Prop
  | none => ∀ ν, InLab R ν → t.den ρ ν = 0
  | some t' => ∀ ν, InLab R ν → t'.den ρ ν = t.den ρ ν

/-- `normTmH` with `elimAllX`. -/
def normTmX (R : List (ℕ × ℕ)) (fl : List ℕ) : List RHint → STm → Option STm
  | [], t => some t
  | (c, es) :: cs, t =>
    match canonTmH c fl (fTm t (elimAllX R es)) with
    | none => none
    | some t' => fTm t' (normTmX R fl cs)

theorem normTmX_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ) :
    ∀ (cs : List RHint) (t : STm), TmResR ρ R t (normTmX R fl cs t) := by
  intro cs
  induction cs with
  | nil => intro t ν _; rfl
  | cons h cs ih =>
    obtain ⟨c, es⟩ := h
    intro t
    have hc := canonTmH_ok ρ hρ fl hfl c (elimAllX R es t)
    have he := elimAllX_ok ρ hρ R es t
    simp only [normTmX, fTm_eq]
    revert hc
    cases canonTmH c fl (elimAllX R es t) with
    | none => intro hc ν hν; rw [← he ν hν, hc ν]
    | some t' =>
      intro hc
      simp only [fTm_eq]
      have := ih t'
      revert this
      cases normTmX R fl cs t' with
      | none => intro h' ν hν; rw [← he ν hν, ← hc ν, h' ν hν]
      | some t'' => intro h' ν hν; rw [h' ν hν, hc ν, he ν hν]

/-- The terms normalised with their rounds, in order, summed into `acc`. -/
def nfGoX (R : List (ℕ × ℕ)) (fl : List ℕ) : List (List RHint) → List STm → List STm → List STm
  | _, acc, [] => acc
  | hs, acc, t :: ts =>
    match normTmX R fl (hs.headD []) t with
    | none => nfGoX R fl hs.tail acc ts
    | some t' => nfGoX R fl hs.tail (tmAdd t' acc) ts

theorem nfGoX_den (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (ν : ℕ → ℤ) (hν : InLab R ν) : ∀ (e : List STm) (hs : List (List RHint)) (acc : List STm),
    exDen ρ ν (nfGoX R fl hs acc e) = exDen ρ ν acc + exDen ρ ν e := by
  intro e
  induction e with
  | nil => intro hs acc; simp [nfGoX, exDen]
  | cons t ts ih =>
    intro hs acc
    have hn := normTmX_ok ρ hρ R fl hfl (hs.headD []) t
    simp only [nfGoX]
    revert hn
    cases normTmX R fl (hs.headD []) t with
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

/-- The residual normalised term by term with the labels' ranges `R` and the rounds `H`. -/
def rxNf (R : List (ℕ × ℕ)) (fl : List ℕ) (H : List (List RHint)) (e : List STm) : List STm :=
  nfGoX R fl H [] e

theorem rxNf_den (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (H : List (List RHint)) (nf : List STm → List STm)
    (hnf : ∀ e ν, exDen ρ ν (nf e) = exDen ρ ν e) (e : List STm) (ν : ℕ → ℤ)
    (hν : InLab R ν) : exDen ρ ν (rxNf R fl H (nf e)) = exDen ρ ν e := by
  unfold rxNf
  rw [nfGoX_den ρ hρ R fl hfl ν hν, hnf]
  simp [exDen]

end ArchytasZX.Sym
