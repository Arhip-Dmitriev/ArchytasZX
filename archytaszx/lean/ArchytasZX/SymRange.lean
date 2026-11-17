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

end ArchytasZX.Sym
