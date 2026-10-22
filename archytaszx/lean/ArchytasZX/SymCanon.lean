/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymNorm

/-!
# Canonical terms and expressions

`canonTm` substitutes each delta's pivot into the factors whose modulus divides the delta's,
merges roots of unity of one order, evaluates constant factors and sorts the factors.
`normTm` alternates eliminations and canonicalisation; `nfEx` normalises every term of an
expression and merges terms of equal shape. Each keeps the value: `nfEx_den`.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Pivots -/

/-- `p = s x + b` with `s = ±1` and `b` free of `x`: `x`, `s`, `b`. -/
def pivTry (p : IPoly) (x : ℕ) : Option (ℕ × ℤ × IPoly) :=
  match ipLin x p with
  | some (a, b) =>
    match ipConst a with
    | some s => if s = 1 ∨ s = -1 then some (x, s, b) else none
    | none => none
  | none => none

/-- The pivot with the greater variable. -/
def pivPick (p : IPoly) (acc : Option (ℕ × ℤ × IPoly)) (x : ℕ) : Option (ℕ × ℤ × IPoly) :=
  match pivTry p x with
  | some q =>
    match acc with
    | some r => if r.1 < x then some q else acc
    | none => some q
  | none => acc

/-- The pivot on the greatest variable. -/
def pivotOf (p : IPoly) : Option (ℕ × ℤ × IPoly) := (p.flatMap (·.1)).foldl (pivPick p) none

/-- What a pivot guarantees. -/
def PivOK (p : IPoly) (q : ℕ × ℤ × IPoly) : Prop :=
  (q.2.1 = 1 ∨ q.2.1 = -1) ∧ ipHas q.1 q.2.2 = false ∧
    ∀ ν, ipDen ν p = q.2.1 * ν q.1 + ipDen ν q.2.2

theorem pivTry_spec (p : IPoly) (x : ℕ) (q : ℕ × ℤ × IPoly) (h : pivTry p x = some q) :
    PivOK p q := by
  unfold pivTry at h
  split at h
  · next a b hlin =>
    split at h
    · next s hs =>
      split_ifs at h with h1
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨_, hb, hp⟩ := ipLin_spec x p a b hlin
      exact ⟨h1, hb, fun ν => by rw [hp, ipConst_spec a s hs]⟩
    · simp at h
  · simp at h

theorem pivotOf_spec (p : IPoly) (q : ℕ × ℤ × IPoly) (h : pivotOf p = some q) : PivOK p q := by
  unfold pivotOf at h
  suffices H : ∀ (l : List ℕ) (acc : Option (ℕ × ℤ × IPoly)), (∀ r, acc = some r → PivOK p r) →
      ∀ r, l.foldl (pivPick p) acc = some r → PivOK p r from H _ none (by simp) q h
  intro l
  induction l with
  | nil => intro acc hacc r hr; exact hacc r hr
  | cons x l ih =>
    intro acc hacc r hr
    simp only [List.foldl_cons] at hr
    refine ih _ ?_ r hr
    intro r' hr'
    unfold pivPick at hr'
    split at hr'
    · next q' hq =>
      split at hr'
      · split_ifs at hr'
        · simp only [Option.some.injEq] at hr'; subst hr'; exact pivTry_spec p x _ hq
        · exact hacc r' hr'
      · simp only [Option.some.injEq] at hr'; subst hr'; exact pivTry_spec p x _ hq
    · exact hacc r' hr'

/-- `x` substituted by `v` in a factor whose modulus divides `M`. -/
def pivSub (M : DMon) (x : ℕ) (v : IPoly) (g : Fac) : Fac := if g.1.dvd M then facSub x v g else g

theorem pivSub_val (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (M : DMon) (x : ℕ) (v : IPoly) (g : Fac)
    (ν : ℕ → ℤ) (h : ((M.den dv : ℕ) : ℤ) ∣ ν x - ipDen ν v) :
    delV dv ν (pivSub M x v g) = delV dv ν g ∧ phV dv ν (pivSub M x v g) = phV dv ν g := by
  unfold pivSub
  split_ifs with hg
  · have hc := fac_congr dv hdv x M g hg ν (ipDen ν v) (ν x) (by
      rw [← neg_sub]; exact (dvd_neg).2 h)
    rw [Function.update_eq_self] at hc
    constructor
    · rw [← hc.1]; simp [delV, facSub, ipDen_subst]
    · rw [← hc.2]; simp [phV, facSub, ipDen_subst]
  · exact ⟨rfl, rfl⟩

/-- Coefficients divided by `c`. -/
def ipDivC (c : ℤ) (p : IPoly) : IPoly := p.foldr (fun t acc => ipIns t.1 (t.2 / c) acc) []

theorem ipDen_divC (ν : ℕ → ℤ) (c : ℤ) (p : IPoly) (h : ipDivBy c p = true) :
    c * ipDen ν (ipDivC c p) = ipDen ν p := by
  induction p with
  | nil => simp [ipDivC, ipDen]
  | cons t p ih =>
    simp only [ipDivBy, List.all_cons, Bool.and_eq_true, beq_iff_eq] at h ih
    simp only [ipDivC, List.foldr_cons] at ih ⊢
    obtain ⟨m, k⟩ := t
    rw [ipDen_ins, ipDen_cons, mul_add, ← ih h.2, ← mul_assoc, Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero h.1)]

/-- `p = c x + b` with a constant `c ≠ 0`, `x` the greatest such variable: `x`, `c`, `b`. -/
def divTry (p : IPoly) (x : ℕ) : Option (ℕ × ℤ × IPoly) :=
  match ipLin x p with
  | some (a, b) =>
    match ipConst a with
    | some c => if c ≠ 0 then some (x, c, b) else none
    | none => none
  | none => none

/-- The division variable with the greater index. -/
def divPick (p : IPoly) (acc : Option (ℕ × ℤ × IPoly)) (x : ℕ) : Option (ℕ × ℤ × IPoly) :=
  match divTry p x with
  | some q =>
    match acc with
    | some r => if r.1 < x then some q else acc
    | none => some q
  | none => acc

/-- The division on the greatest variable. -/
def divOf (p : IPoly) : Option (ℕ × ℤ × IPoly) := (p.flatMap (·.1)).foldl (divPick p) none

/-- What a division guarantees. -/
def DivOK (p : IPoly) (q : ℕ × ℤ × IPoly) : Prop :=
  ipHas q.1 q.2.2 = false ∧ ∀ ν, ipDen ν p = q.2.1 * ν q.1 + ipDen ν q.2.2

theorem divOf_spec (p : IPoly) (q : ℕ × ℤ × IPoly) (h : divOf p = some q) : DivOK p q := by
  have base : ∀ x q', divTry p x = some q' → DivOK p q' := by
    intro x q' hq
    unfold divTry at hq
    split at hq
    · next a b hlin =>
      split at hq
      · next c hc =>
        split_ifs at hq
        simp only [Option.some.injEq] at hq
        subst hq
        obtain ⟨_, hb, hp⟩ := ipLin_spec x p a b hlin
        exact ⟨hb, fun ν => by rw [hp, ipConst_spec a c hc]⟩
      · simp at hq
    · simp at hq
  unfold divOf at h
  suffices H : ∀ (l : List ℕ) (acc : Option (ℕ × ℤ × IPoly)), (∀ r, acc = some r → DivOK p r) →
      ∀ r, l.foldl (divPick p) acc = some r → DivOK p r from H _ none (by simp) q h
  intro l
  induction l with
  | nil => intro acc hacc r hr; exact hacc r hr
  | cons x l ih =>
    intro acc hacc r hr
    simp only [List.foldl_cons] at hr
    refine ih _ ?_ r hr
    intro r' hr'
    unfold divPick at hr'
    split at hr'
    · next q' hq =>
      split at hr'
      · split_ifs at hr'
        · simp only [Option.some.injEq] at hr'; subst hr'; exact base x _ hq
        · exact hacc r' hr'
      · simp only [Option.some.injEq] at hr'; subst hr'; exact base x _ hq
    · exact hacc r' hr'

/-- `c x` replaced by `-b` in a factor whose modulus divides `M` and whose `x` coefficients
`c` divides. -/
def divSub (M : DMon) (x : ℕ) (c : ℤ) (b : IPoly) (g : Fac) : Fac :=
  if g.1.dvd M then
    match ipLin x g.2 with
    | some (ag, bg) => if ipDivBy c ag then (g.1, ipSub bg (ipMul (ipDivC c ag) b)) else g
    | none => g
  else g

/-- The substitution a delta allows in the other factors. -/
def stepOf (f : Fac) : Option (Fac → Fac) :=
  match pivotOf f.2 with
  | some q => some (pivSub f.1 q.1 (pinVal q.2.1 q.2.2))
  | none =>
    match divOf f.2 with
    | some q => some (divSub f.1 q.1 q.2.1 q.2.2)
    | none => none

/-- Where the delta holds, the substitution keeps every factor's value. -/
def StepOK (dv : ℕ → ℕ) (f : Fac) (σ : Fac → Fac) : Prop :=
  ∀ ν, delV dv ν f = 1 → ∀ g, delV dv ν (σ g) = delV dv ν g ∧ phV dv ν (σ g) = phV dv ν g

theorem fac_mod (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (M : DMon) (g : Fac) (hg : g.1.dvd M = true)
    (ν : ℕ → ℤ) (q : IPoly) (h : ((M.den dv : ℕ) : ℤ) ∣ ipDen ν g.2 - ipDen ν q) :
    delV dv ν (g.1, q) = delV dv ν g ∧ phV dv ν (g.1, q) = phV dv ν g := by
  have hN : ((g.1.den dv : ℕ) : ℤ) ∣ ipDen ν g.2 - ipDen ν q :=
    (Int.natCast_dvd_natCast.2 (DMon.dvd_den dv _ _ hg)).trans h
  constructor
  · unfold delV
    simp only
    have : ((g.1.den dv : ℕ) : ℤ) ∣ ipDen ν q ↔ ((g.1.den dv : ℕ) : ℤ) ∣ ipDen ν g.2 := by
      constructor
      · intro h'; have := dvd_add hN h'; rwa [sub_add_cancel] at this
      · intro h'; have := dvd_sub h' hN; rwa [sub_sub_cancel] at this
    simp only [this]
  · unfold phV
    simp only
    obtain ⟨k, hk⟩ := hN
    have hpos : 0 < g.1.den dv := by
      unfold DMon.dvd at hg
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hg
      exact DMon.den_pos dv hdv _ (Nat.pos_of_ne_zero hg.1.1)
    rw [show ipDen ν g.2 = ipDen ν q + (g.1.den dv : ℤ) * k by omega,
      zpow_add₀ (om_ne_zero _ hpos), zpow_mul, zpow_natCast, om_pow_self _ hpos, one_zpow, mul_one]

theorem stepOf_ok (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (f : Fac) (σ : Fac → Fac)
    (h : stepOf f = some σ) : StepOK dv f σ := by
  intro ν h1 g
  have hM : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f.2 := by
    by_contra hn; unfold delV at h1; rw [if_neg hn] at h1; simp at h1
  unfold stepOf at h
  split at h
  · next q hq =>
    simp only [Option.some.injEq] at h
    subst h
    obtain ⟨hs, _, hp⟩ := pivotOf_spec _ _ hq
    rw [hp] at hM
    have := (pin_dvd q.2.1 hs _ (ν q.1) (ipDen ν q.2.2)).1 hM
    exact pivSub_val dv hdv f.1 q.1 (pinVal q.2.1 q.2.2) g ν (by simpa [pinVal, ipDen_smul] using this)
  · split at h
    · next q hq =>
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨_, hp⟩ := divOf_spec _ _ hq
      unfold divSub
      split_ifs with hg
      · split
        · next ag bg hlin =>
          split_ifs with hdiv
          · obtain ⟨_, _, hgp⟩ := ipLin_spec q.1 g.2 ag bg hlin
            refine fac_mod dv hdv f.1 g hg ν _ ?_
            rw [ipDen_sub, ipDen_mul, hgp, ← ipDen_divC ν q.2.1 ag hdiv]
            have e : q.2.1 * ipDen ν (ipDivC q.2.1 ag) * ν q.1 + ipDen ν bg -
                (ipDen ν bg - ipDen ν (ipDivC q.2.1 ag) * ipDen ν q.2.2) =
                ipDen ν (ipDivC q.2.1 ag) * (q.2.1 * ν q.1 + ipDen ν q.2.2) := by ring
            rw [e, ← hp]
            exact Dvd.dvd.mul_left hM _
          · exact ⟨rfl, rfl⟩
        · exact ⟨rfl, rfl⟩
      · exact ⟨rfl, rfl⟩
    · simp at h

theorem step_body (dv : ℕ → ℕ) (A B ps : List Fac) (f : Fac) (σ : Fac → Fac)
    (hσ : StepOK dv f σ) (ν : ℕ → ℤ) :
    bodyOf dv (A.map σ ++ f :: B.map σ) (ps.map σ) ν = bodyOf dv (A ++ f :: B) ps ν := by
  simp only [bodyOf, List.map_append, List.map_cons, List.prod_append, List.prod_cons,
    List.map_map]
  rcases delV_cases dv ν f with h0 | h1
  · rw [h0]; ring
  · have hv := hσ ν h1
    have e1 : ∀ l : List Fac, (l.map (delV dv ν ∘ σ)).prod = (l.map (delV dv ν)).prod :=
      fun l => congrArg List.prod (List.map_congr_left fun g _ => (hv g).1)
    have e2 : ∀ l : List Fac, (l.map (phV dv ν ∘ σ)).prod = (l.map (phV dv ν)).prod :=
      fun l => congrArg List.prod (List.map_congr_left fun g _ => (hv g).2)
    rw [e1, e1, e2]

/-- Each delta in turn substitutes into the factors it determines. -/
def pivPass : ℕ → List Fac → List Fac → List Fac → List Fac × List Fac
  | 0, A, B, ps => (A ++ B, ps)
  | _ + 1, A, [], ps => (A, ps)
  | n + 1, A, f :: B, ps =>
    match stepOf f with
    | some σ => pivPass n ((A.map σ) ++ [f]) (B.map σ) (ps.map σ)
    | none => pivPass n (A ++ [f]) B ps

theorem pivPass_ok (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (n : ℕ) (A B ps : List Fac) (ν : ℕ → ℤ) :
    bodyOf dv (pivPass n A B ps).1 (pivPass n A B ps).2 ν = bodyOf dv (A ++ B) ps ν := by
  induction n generalizing A B ps with
  | zero => rfl
  | succ n ih =>
    cases B with
    | nil => simp [pivPass]
    | cons f B =>
      simp only [pivPass]
      split
      · next σ hq =>
        rw [ih, List.append_assoc, List.singleton_append]
        exact step_body dv A B ps f σ (stepOf_ok dv hdv f σ hq) ν
      · rw [ih, List.append_assoc, List.singleton_append]

/-! ## Factors -/

/-- Floors of the dimension symbols. -/
def flv (fl : List ℕ) (i : ℕ) : ℕ := fl.getD i 0

/-- The valuation meets the floors. -/
def FloorOK (fl : List ℕ) (ρ : Val) : Prop := ∀ i, flv fl i ≤ ρ.dv i

theorem DMon.den_isConst (dv : ℕ → ℕ) (m : DMon) (h : m.isConst = true) : m.den dv = m.c := by
  rw [DMon.den_const dv (fun _ => 1) m h]; simp [DMon.den, npow_one]

/-- `c` reduced modulo `k` into `(-k/2, k/2]`. -/
def symMod (c : ℤ) (k : ℕ) : ℤ := if 2 * (c % (k : ℤ)) > k then c % (k : ℤ) - k else c % (k : ℤ)

/-- Coefficients reduced modulo `k` into `(-k/2, k/2]`. -/
def ipModS (k : ℕ) (p : IPoly) : IPoly := p.foldr (fun t acc => ipIns t.1 (symMod t.2 k) acc) []

theorem ipDen_modS (ν : ℕ → ℤ) (k : ℕ) (p : IPoly) :
    (k : ℤ) ∣ ipDen ν p - ipDen ν (ipModS k p) := by
  induction p with
  | nil => simp [ipModS, ipDen]
  | cons t p ih =>
    simp only [ipModS, List.foldr_cons] at ih ⊢
    rw [ipDen_ins, ipDen_cons]
    have : t.2 * imDen ν t.1 + ipDen ν p - (symMod t.2 k * imDen ν t.1 +
        ipDen ν (List.foldr (fun t acc => ipIns t.1 (symMod t.2 k) acc) [] p)) =
        (t.2 - symMod t.2 k) * imDen ν t.1 + (ipDen ν p -
        ipDen ν (List.foldr (fun t acc => ipIns t.1 (symMod t.2 k) acc) [] p)) := by ring
    rw [this]
    refine dvd_add (Dvd.dvd.mul_right ?_ _) ih
    unfold symMod
    split_ifs
    · rw [show t.2 - (t.2 % (k : ℤ) - k) = (t.2 - t.2 % (k : ℤ)) + k by ring, Int.emod_def]
      simp
    · rw [Int.emod_def]; simp

/-- A delta's polynomial reduced and signed. -/
def delPoly (f : Fac) : IPoly := if f.1.isConst then ipSign (ipModS f.1.c f.2) else ipSign f.2

/-- A delta normalised: `none` when it vanishes, `some none` when it is one. -/
def normDel (fl : List ℕ) (f : Fac) : Option (Option Fac) :=
  let p := delPoly f
  match ipConst p with
  | some c =>
    if c = 0 then some none
    else if f.1.isConst then (if c % (f.1.c : ℤ) = 0 then some none else none)
    else if c.natAbs < f.1.den (flv fl) then none else some (some (f.1, p))
  | none => some (some (f.1, p))

theorem delV_mod (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) :
    delV dv ν (f.1, delPoly f) = delV dv ν f := by
  have hs : ∀ q : IPoly, (((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν (ipSign q) ↔
      ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν q) := by
    intro q
    rcases ipDen_sign ν q with e | e <;> rw [e]; exact dvd_neg
  have key : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν (delPoly f) ↔
      ((f.1.den dv : ℕ) : ℤ) ∣ ipDen ν f.2 := by
    unfold delPoly
    split_ifs with h1
    · rw [hs, DMon.den_isConst dv _ h1]
      have hm := ipDen_modS ν f.1.c f.2
      constructor
      · intro h; have := dvd_add hm h; rwa [sub_add_cancel] at this
      · intro h; have := dvd_sub h hm; rwa [sub_sub_cancel] at this
    · exact hs _
  unfold delV
  simp only [key]

/-- What a normalised delta says about the delta. -/
def DelRes (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) : Option (Option Fac) → Prop
  | none => delV dv ν f = 0
  | some none => delV dv ν f = 1
  | some (some g) => delV dv ν g = delV dv ν f

theorem normDel_ok (ρ : Val) (fl : List ℕ) (hfl : FloorOK fl ρ) (f : Fac) (ν : ℕ → ℤ) :
    DelRes ρ.dv ν f (normDel fl f) := by
  have hm := delV_mod ρ.dv ν f
  unfold normDel
  simp only
  split
  · next c hc =>
    have hv := ipConst_spec _ c hc ν
    split_ifs with h1 h2 h3 h4
    · simp only [DelRes]; rw [← hm]; simp [delV, hv, h1]
    · simp only [DelRes]; rw [← hm]; simp only [delV, hv, DMon.den_isConst _ _ h2]
      rw [if_pos (Int.dvd_of_emod_eq_zero h3)]
    · simp only [DelRes]; rw [← hm]; simp only [delV, hv, DMon.den_isConst _ _ h2]
      rw [if_neg (fun h => h3 (Int.emod_eq_zero_of_dvd h))]
    · simp only [DelRes]; rw [← hm]; simp only [delV, hv]
      rw [if_neg]
      intro hd
      have hle := DMon.den_mono _ _ hfl f.1
      have := Int.natAbs_dvd_natAbs.2 hd
      simp only [Int.natAbs_natCast] at this
      have := Nat.le_of_dvd (Int.natAbs_pos.2 h1) this
      omega
    · exact hm
  · exact hm

/-- The deltas normalised; `none` when one vanishes. -/
def normDels (fl : List ℕ) : List Fac → Option (List Fac)
  | [] => some []
  | f :: l =>
    match normDel fl f, normDels fl l with
    | none, _ => none
    | _, none => none
    | some none, some l' => some l'
    | some (some g), some l' => some (g :: l')

/-- What the normalised deltas say about the deltas. -/
def DelsRes (dv : ℕ → ℕ) (ν : ℕ → ℤ) (l : List Fac) : Option (List Fac) → Prop
  | none => (l.map (delV dv ν)).prod = 0
  | some l' => (l'.map (delV dv ν)).prod = (l.map (delV dv ν)).prod

theorem normDels_ok (ρ : Val) (fl : List ℕ) (hfl : FloorOK fl ρ) (l : List Fac) (ν : ℕ → ℤ) :
    DelsRes ρ.dv ν l (normDels fl l) := by
  induction l with
  | nil => simp [normDels, DelsRes]
  | cons f l ih =>
    have hf := normDel_ok ρ fl hfl f ν
    simp only [normDels]
    cases h1 : normDel fl f with
    | none => rw [h1] at hf; simp [DelsRes, DelRes] at hf ⊢; simp [hf]
    | some o =>
      cases h2 : normDels fl l with
      | none => rw [h2] at ih; simp [DelsRes] at ih ⊢; simp [ih]
      | some l' =>
        rw [h2] at ih; simp only [DelsRes] at ih
        cases o with
        | none => rw [h1] at hf; simp only [DelRes] at hf; simp [DelsRes, ih, hf]
        | some g => rw [h1] at hf; simp only [DelRes] at hf; simp [DelsRes, ih, hf]

/-- The value of an optional root of unity. -/
noncomputable def optPh (dv : ℕ → ℕ) (ν : ℕ → ℤ) : Option Fac → ℂ
  | none => 1
  | some f => phV dv ν f

/-- A root of unity normalised: a constant one becomes a coefficient. -/
def normPh (f : Fac) : CMon × Option Fac :=
  if f.1.isConst ∧ f.1.c ≠ 0 then
    let p := ipModC f.1.c f.2
    match ipConst p with
    | some c => (CMon.exQ ((c : ℚ) / f.1.c), none)
    | none => (CMon.ofQ 1, some (f.1, p))
  else if f.2 = [] then (CMon.ofQ 1, none) else (CMon.ofQ 1, some f)

theorem om_zpow_ex (c : ℕ) (k : ℤ) : om c ^ k = ex (((k : ℚ) / c : ℚ) : ℝ) := by
  unfold om ex
  rw [← Complex.exp_int_mul]
  congr 1
  push_cast
  ring

theorem normPh_ok (ρ : Val) (f : Fac) (ν : ℕ → ℤ) :
    phV ρ.dv ν f = (normPh f).1.den ρ * optPh ρ.dv ν (normPh f).2 := by
  unfold normPh
  split_ifs with h1 h2
  · have hc := DMon.den_isConst ρ.dv _ h1.1
    have hpos : 0 < f.1.c := Nat.pos_of_ne_zero h1.2
    have hmod : phV ρ.dv ν f = phV ρ.dv ν (f.1, ipModC f.1.c f.2) := by
      unfold phV
      simp only [hc]
      obtain ⟨k, hk⟩ := ipDen_modC ν f.1.c f.2
      rw [show ipDen ν f.2 = ipDen ν (ipModC f.1.c f.2) + f.1.c * k by omega,
        zpow_add₀ (om_ne_zero _ hpos), zpow_mul, zpow_natCast, om_pow_self _ hpos, one_zpow,
        mul_one]
    simp only
    split
    · next c hcst =>
      rw [hmod]
      simp only [optPh, mul_one, exQ_den, phV, hc, ipConst_spec _ c hcst, om_zpow_ex]
    · simp only [optPh, ofQ_den, Rat.cast_one, one_mul]; exact hmod
  · simp only [optPh, ofQ_den, Rat.cast_one, one_mul, phV, h2, ipDen_nil, zpow_zero]
  · simp [optPh, ofQ_den]

/-- Every root of unity normalised: the coefficient gained and the factors kept. -/
def normPhs : List Fac → CMon × List Fac
  | [] => (CMon.ofQ 1, [])
  | f :: l =>
    let r := normPhs l
    let n := normPh f
    (CMon.mul n.1 r.1, match n.2 with
      | none => r.2
      | some g => g :: r.2)

theorem normPhs_ok (ρ : Val) (hρ : ρ.OK) (l : List Fac) (ν : ℕ → ℤ) :
    (l.map (phV ρ.dv ν)).prod = (normPhs l).1.den ρ * ((normPhs l).2.map (phV ρ.dv ν)).prod := by
  induction l with
  | nil => simp [normPhs, ofQ_den]
  | cons f l ih =>
    simp only [normPhs, List.map_cons, List.prod_cons, CMon.den_mul ρ hρ, ih, normPh_ok ρ f ν]
    split
    · next h => simp only [h, optPh]; ring
    · next g h => simp only [h, optPh, List.map_cons, List.prod_cons]; ring

/-- A root of unity merged into one of its order. -/
def phIns (f : Fac) : List Fac → List Fac
  | [] => [f]
  | g :: l => if g.1 = f.1 then (g.1, ipAdd g.2 f.2) :: l else g :: phIns f l

/-- Roots of unity of one order merged. -/
def phMerge (ps : List Fac) : List Fac := ps.foldl (fun acc f => phIns f acc) []

theorem phIns_ok (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) (l : List Fac) :
    ((phIns f l).map (phV dv ν)).prod = phV dv ν f * (l.map (phV dv ν)).prod := by
  induction l with
  | nil => simp [phIns]
  | cons g l ih =>
    simp only [phIns]
    split_ifs with h
    · simp only [List.map_cons, List.prod_cons, phV, ipDen_add, h]
      rw [zpow_add₀ (show om (f.1.den dv) ≠ 0 from Complex.exp_ne_zero _)]; ring
    · simp only [List.map_cons, List.prod_cons, ih]; ring

theorem phMerge_ok (dv : ℕ → ℕ) (ν : ℕ → ℤ) (ps : List Fac) :
    ((phMerge ps).map (phV dv ν)).prod = (ps.map (phV dv ν)).prod := by
  unfold phMerge
  suffices H : ∀ acc : List Fac, ((ps.foldl (fun acc f => phIns f acc) acc).map (phV dv ν)).prod =
      (acc.map (phV dv ν)).prod * (ps.map (phV dv ν)).prod by simpa using H []
  induction ps with
  | nil => intro acc; simp
  | cons f ps ih => intro acc; simp only [List.foldl_cons, ih, phIns_ok, List.map_cons,
      List.prod_cons]; ring

/-! ## Sorting -/

/-- Lexicographic order on polynomials. -/
def ipLt : IPoly → IPoly → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | (m, c) :: p, (m', c') :: q =>
    lexLt m m' || (m == m' && (decide (c < c') || (c == c' && ipLt p q)))

/-- Order on factors. -/
def facLt (f g : Fac) : Bool :=
  decide (f.1.c < g.1.c) || (f.1.c == g.1.c &&
    (lexLt f.1.e g.1.e || (f.1.e == g.1.e && ipLt f.2 g.2)))

/-- Sorted insertion. -/
def facIns (f : Fac) : List Fac → List Fac
  | [] => [f]
  | g :: l => if facLt g f then g :: facIns f l else f :: g :: l

/-- Sorted insertion dropping a repeated delta. -/
def delIns (f : Fac) : List Fac → List Fac
  | [] => [f]
  | g :: l => if g = f then g :: l else if facLt g f then g :: delIns f l else f :: g :: l

theorem facIns_perm (f : Fac) (l : List Fac) : (facIns f l).Perm (f :: l) := by
  induction l with
  | nil => simp [facIns]
  | cons g l ih =>
    simp only [facIns]
    split_ifs
    · exact (ih.cons g).trans (List.Perm.swap f g l)
    · exact List.Perm.refl _

theorem facSort_ok (dv : ℕ → ℕ) (ν : ℕ → ℤ) (l : List Fac) :
    ((l.foldr facIns []).map (phV dv ν)).prod = (l.map (phV dv ν)).prod := by
  induction l with
  | nil => rfl
  | cons f l ih =>
    simp only [List.foldr_cons]
    rw [((facIns_perm f _).map _).prod_eq]
    simp [ih]

theorem delIns_ok (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) (l : List Fac) :
    ((delIns f l).map (delV dv ν)).prod = delV dv ν f * (l.map (delV dv ν)).prod := by
  induction l with
  | nil => simp [delIns]
  | cons g l ih =>
    simp only [delIns]
    split_ifs with h1 h2
    · subst h1
      simp only [List.map_cons, List.prod_cons]
      rcases delV_cases dv ν g with h | h <;> rw [h] <;> ring
    · simp only [List.map_cons, List.prod_cons, ih]; ring
    · simp

theorem delSort_ok (dv : ℕ → ℕ) (ν : ℕ → ℤ) (l : List Fac) :
    ((l.foldr delIns []).map (delV dv ν)).prod = (l.map (delV dv ν)).prod := by
  induction l with
  | nil => rfl
  | cons f l ih => simp [delIns_ok, ih]

/-! ## Terms -/

/-! ## Renaming bound variables -/

/-- The first canonical bound variable. -/
def RBASE : ℕ := 2000000

/-- The bound variable `x` renamed `y`, when `x` is bound once and `y` is fresh. -/
def renOne (t : STm) (x y : ℕ) : Option STm :=
  if (bvs t).count x = 1 ∧ y ∉ bvs t ∧ y ∉ tmVars t then
    some ⟨t.c, t.sums.map (fun p => if p.1 = x then (y, p.2) else p),
      t.dels.map (facSub x (ipV y)), t.phs.map (facSub x (ipV y))⟩
  else none

theorem renOne_ok (ρ : Val) (t t' : STm) (x y : ℕ) (h : renOne t x y = some t') (ν : ℕ → ℤ) :
    t'.den ρ ν = t.den ρ ν := by
  unfold renOne at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  obtain ⟨h1, h2, h3⟩ := hc
  have hx : x ∈ bvs t := List.count_pos_iff.1 (by omega)
  obtain ⟨p, hp, hpx⟩ : ∃ p ∈ t.sums, p.1 = x := by simpa [bvs] using hx
  obtain ⟨a, b, hab⟩ := List.append_of_mem hp
  have hcount := h1
  rw [bvs, hab] at hcount
  simp only [List.map_append, List.map_cons, List.count_append, List.count_cons, hpx,
    beq_self_eq_true, ite_true] at hcount
  have hna : x ∉ a.map (·.1) := fun h' => by
    have := List.count_pos_iff.2 h'; omega
  have hnb : x ∉ b.map (·.1) := fun h' => by
    have := List.count_pos_iff.2 h'; omega
  have hmap : ∀ l : List (ℕ × DMon), x ∉ l.map (·.1) →
      l.map (fun p => if p.1 = x then (y, p.2) else p) = l := by
    intro l hl
    conv_rhs => rw [← List.map_id l]
    refine List.map_congr_left fun q hq => ?_
    have : q.1 ≠ x := fun e => hl (List.mem_map.2 ⟨q, hq, e⟩)
    simp [this]
  have hy : y ∉ (a ++ b).map (·.1) := by
    intro h'
    apply h2
    rw [bvs, hab]
    simp only [List.map_append, List.map_cons, List.mem_append, List.mem_cons] at h' ⊢
    tauto
  have hx' : x ∉ (a ++ b).map (·.1) := by
    simp only [List.map_append, List.mem_append, not_or]; exact ⟨hna, hnb⟩
  obtain ⟨px, D⟩ := p
  simp only at hpx
  subst hpx
  unfold STm.den
  simp only [hab, List.map_append, List.map_cons, ite_true, hmap a hna, hmap b hnb]
  rw [sumsV_pull _ _ _ _ _ hy, sumsV_pull _ _ _ _ _ hx']
  congr 1
  refine sumsV_congr _ _ _ _ (fun ν' => Finset.sum_congr rfl fun j _ => ?_) ν
  rw [body_eq, body_eq]
  simp only
  rw [bodyOf_sub, ipDen_V, Function.update_self]
  refine body_reads ρ.dv t _ _ fun z hz => ?_
  have hzy : z ≠ y := fun e => h3 (e ▸ hz)
  by_cases hzx : z = px
  · subst hzx; simp
  · rw [Function.update_of_ne hzx, Function.update_of_ne hzy, Function.update_of_ne hzx]

/-- The bound variables `xs` renamed `RBASE + k`, `RBASE + k + 1`, … where fresh. -/
def renList (t : STm) (k : ℕ) : List ℕ → STm
  | [] => t
  | x :: xs => renList ((renOne t x (RBASE + k)).getD t) (k + 1) xs

theorem renList_ok (ρ : Val) (t : STm) (k : ℕ) (xs : List ℕ) (ν : ℕ → ℤ) :
    (renList t k xs).den ρ ν = t.den ρ ν := by
  induction xs generalizing t k with
  | nil => rfl
  | cons x xs ih =>
    simp only [renList]
    rw [ih]
    cases h : renOne t x (RBASE + k) with
    | none => rfl
    | some t' => exact renOne_ok ρ t t' x _ h ν

/-- The term with its factors canonical; `none` when it vanishes. -/
def canonTm' (fl : List ℕ) (t : STm) : Option STm :=
  let pp := pivPass t.dels.length [] t.dels t.phs
  match normDels fl pp.1 with
  | none => none
  | some ds =>
    let np := normPhs (phMerge pp.2)
    some ⟨CMon.mul t.c np.1, t.sums, ds.foldr delIns [], np.2.foldr facIns []⟩

/-- The term renamed and canonical; `none` when it vanishes. -/
def canonTm (fl : List ℕ) (t : STm) : Option STm :=
  canonTm' fl (renList t 0 (bvs t))

/-- `t'` has the value of `t`, or `t` vanishes. -/
def TmRes (ρ : Val) (t : STm) : Option STm → Prop
  | none => ∀ ν, t.den ρ ν = 0
  | some t' => ∀ ν, t'.den ρ ν = t.den ρ ν

theorem canonTm'_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (t : STm) :
    TmRes ρ t (canonTm' fl t) := by
  have hpp : ∀ ν, body ρ.dv t ν = bodyOf ρ.dv (pivPass t.dels.length [] t.dels t.phs).1
      (pivPass t.dels.length [] t.dels t.phs).2 ν := fun ν => by
    rw [pivPass_ok ρ.dv hρ.dpos]; rfl
  unfold canonTm'
  simp only
  split
  · next h =>
    have hn : ∀ ν, ((pivPass t.dels.length [] t.dels t.phs).1.map (delV ρ.dv ν)).prod = 0 :=
      fun ν => by have := normDels_ok ρ fl hfl (pivPass t.dels.length [] t.dels t.phs).1 ν
                  rw [h] at this; exact this
    intro ν
    unfold STm.den
    rw [sumsV_congr _ _ _ (fun _ => 0) (fun ν' => by rw [hpp ν']; simp [bodyOf, hn ν']),
      sumsV_zero, mul_zero]
  · next ds h =>
    have hn : ∀ ν, (ds.map (delV ρ.dv ν)).prod =
        ((pivPass t.dels.length [] t.dels t.phs).1.map (delV ρ.dv ν)).prod :=
      fun ν => by have := normDels_ok ρ fl hfl (pivPass t.dels.length [] t.dels t.phs).1 ν
                  rw [h] at this; exact this
    intro ν
    unfold STm.den
    simp only [CMon.den_mul ρ hρ]
    have key : ∀ ν', bodyOf ρ.dv (ds.foldr delIns [])
        ((normPhs (phMerge (pivPass t.dels.length [] t.dels t.phs).2)).2.foldr facIns []) ν' *
        (normPhs (phMerge (pivPass t.dels.length [] t.dels t.phs).2)).1.den ρ = body ρ.dv t ν' := by
      intro ν'
      rw [hpp ν']
      simp only [bodyOf, delSort_ok, facSort_ok, hn ν']
      rw [← phMerge_ok ρ.dv ν' (pivPass t.dels.length [] t.dels t.phs).2,
        normPhs_ok ρ hρ (phMerge (pivPass t.dels.length [] t.dels t.phs).2) ν']
      ring
    rw [← sumsV_congr _ _ _ _ key, sumsV_congr _ _ _ _ (fun _ => mul_comm _ _),
      sumsV_mul_left]
    simp only [body_eq]
    ring

theorem canonTm_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (t : STm) :
    TmRes ρ t (canonTm fl t) := by
  have h := canonTm'_ok ρ hρ fl hfl (renList t 0 (bvs t))
  unfold canonTm
  revert h
  cases canonTm' fl (renList t 0 (bvs t)) with
  | none => intro h ν; rw [← renList_ok ρ t 0 (bvs t) ν]; exact h ν
  | some t' => intro h ν; rw [h ν, renList_ok]

/-- Eliminations and canonicalisation, alternated `r` times. -/
def normTm (fl : List ℕ) : ℕ → STm → Option STm
  | 0, t => some t
  | r + 1, t =>
    match canonTm fl (elimAll (t.sums.length + 1) t) with
    | none => none
    | some t' => normTm fl r t'

theorem normTm_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (r : ℕ) (t : STm) :
    TmRes ρ t (normTm fl r t) := by
  induction r generalizing t with
  | zero => intro ν; rfl
  | succ r ih =>
    have hc := canonTm_ok ρ hρ fl hfl (elimAll (t.sums.length + 1) t)
    have he := elimAll_ok ρ hρ (t.sums.length + 1) t
    simp only [normTm]
    split
    · next h => rw [h] at hc; intro ν; rw [← he ν, hc ν]
    · next t' h =>
      rw [h] at hc
      have := ih t'
      revert this
      cases normTm fl r t' with
      | none => intro h' ν; rw [← he ν, ← hc ν, h' ν]
      | some t'' => intro h' ν; rw [h' ν, hc ν, he ν]

/-! ## Expressions -/

/-- The term without its rational coefficient. -/
def shape (t : STm) : STm := { t with c := { t.c with q := 0 } }

/-- A term with rational coefficient `q`. -/
def withQ (t : STm) (q : ℚ) : STm := { t with c := { t.c with q := q } }

/-- `a + b` through `mkRat`. -/
def qAdd (a b : ℚ) : ℚ := mkRat (a.num * b.den + b.num * a.den) (a.den * b.den)

theorem qAdd_eq (a b : ℚ) : qAdd a b = a + b := (Rat.add_def' a b).symm

/-- A term added to an expression, merged with one of the same shape. -/
def tmAdd (t : STm) : List STm → List STm
  | [] => if t.c.q = 0 then [] else [t]
  | u :: l =>
    if shape u = shape t then
      (if qAdd u.c.q t.c.q = 0 then l else withQ u (qAdd u.c.q t.c.q) :: l)
    else u :: tmAdd t l

theorem den_withQ (ρ : Val) (ν : ℕ → ℤ) (t : STm) (q : ℚ) :
    (withQ t q).den ρ ν = (q : ℂ) * (withQ t 1).den ρ ν := by
  unfold STm.den
  rw [show body ρ.dv (withQ t q) = body ρ.dv t from rfl,
    show body ρ.dv (withQ t 1) = body ρ.dv t from rfl]
  simp only [withQ, CMon.den]; push_cast; ring

theorem den_self (ρ : Val) (ν : ℕ → ℤ) (t : STm) : t.den ρ ν = (t.c.q : ℂ) * (withQ t 1).den ρ ν := by
  rw [← den_withQ]; rfl

theorem withQ_shape (t u : STm) (h : shape u = shape t) (q : ℚ) : withQ u q = withQ t q := by
  obtain ⟨⟨_, r, sq, rc, am, gc⟩, s, d, p⟩ := u
  obtain ⟨⟨_, r', sq', rc', am', gc'⟩, s', d', p'⟩ := t
  simp only [shape, STm.mk.injEq, CMon.mk.injEq] at h
  obtain ⟨⟨-, rfl, rfl, rfl, rfl, rfl⟩, rfl, rfl, rfl⟩ := h
  rfl

theorem tmAdd_ok (ρ : Val) (ν : ℕ → ℤ) (t : STm) (l : List STm) :
    exDen ρ ν (tmAdd t l) = t.den ρ ν + exDen ρ ν l := by
  induction l with
  | nil =>
    simp only [tmAdd]
    split_ifs with h
    · simp [exDen, den_self ρ ν t, h]
    · simp [exDen]
  | cons u l ih =>
    simp only [tmAdd, qAdd_eq]
    split_ifs with h1 h2
    · have e := withQ_shape t u h1 1
      have hq : ((u.c.q + t.c.q : ℚ) : ℂ) = 0 := by rw [h2]; simp
      simp only [exDen, List.map_cons, List.sum_cons]
      rw [den_self ρ ν t, den_self ρ ν u, e]
      push_cast at hq
      linear_combination -(withQ t 1).den ρ ν * hq
    · have e := withQ_shape t u h1 1
      simp only [exDen, List.map_cons, List.sum_cons]
      rw [den_withQ, den_self ρ ν t, den_self ρ ν u, e]
      push_cast; ring
    · simp only [exDen, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]; ring

/-- Rounds of eliminations per term. -/
def ROUNDS : ℕ := 3

/-- Every term normalised, terms of one shape merged. -/
def nfEx (fl : List ℕ) (e : List STm) : List STm :=
  (e.filterMap (normTm fl ROUNDS)).foldl (fun acc t => tmAdd t acc) []

theorem nfEx_den (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (e : List STm)
    (ν : ℕ → ℤ) : exDen ρ ν (nfEx fl e) = exDen ρ ν e := by
  have h1 : ∀ (l : List STm) (acc : List STm), exDen ρ ν (l.foldl (fun acc t => tmAdd t acc) acc)
      = exDen ρ ν acc + exDen ρ ν l := by
    intro l
    induction l with
    | nil => intro acc; simp [exDen]
    | cons t l ih =>
      intro acc
      rw [List.foldl_cons, ih, tmAdd_ok]; simp only [exDen, List.map_cons, List.sum_cons]; ring
  have h2 : ∀ l : List STm, exDen ρ ν (l.filterMap (normTm fl ROUNDS)) = exDen ρ ν l := by
    intro l
    induction l with
    | nil => rfl
    | cons t l ih =>
      have := normTm_ok ρ hρ fl hfl ROUNDS t
      simp only [List.filterMap_cons]
      revert this
      cases normTm fl ROUNDS t with
      | none => intro h; simp only [exDen, List.map_cons, List.sum_cons] at ih ⊢; rw [ih, h]; ring
      | some t' =>
        intro h
        simp only [exDen, List.map_cons, List.sum_cons] at ih ⊢; rw [ih, h]
  unfold nfEx
  rw [h1, h2]; simp [exDen]

theorem nfEx_zero (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (e : List STm)
    (h : nfEx fl e = []) (ν : ℕ → ℤ) : exDen ρ ν e = 0 := by
  rw [← nfEx_den ρ hρ fl hfl e ν, h]; rfl

end ArchytasZX.Sym
