/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymCarry
import ArchytasZX.Cyclo

/-!
# Dimension variables and symbolic-modulus steps

The index variable `dvar i` holds the value of dimension symbol `i`, so a polynomial may carry
dimension-monomial coefficients. Each step of `dgRun` keeps an expression's value wherever the
dimension variables hold their symbols and the labels of `R` lie in range: `pin` sums a bound
variable out at the delta `[D ∣ ±x + b]` with `∓b` a variable of range `D`; `comb` adds a
multiple of one delta to another of the same modulus; `merge` writes `ω_M^p` as
`ω_{M d_i}^(d_i p)`; `pchar` sums `x < M` out of roots of unity `ω_{M d_i}^(x A + B)` when a
delta `[d_i ∣ R]` has `d_i ∣ A - R` (`Σ_x ω_{M d_i}^(x A) = M [M d_i ∣ A]`); `dsplit` writes
`[M d_i ∣ d_i A + B]` with `|B| < d_i` as `[d_i ∣ B] [M ∣ A]`; `impl` adds `[d_i ∣ B]` beside
`[M d_i ∣ d_i A + B]`; `nf` is `nfEx`.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Dimension variables -/

/-- The first dimension variable. -/
def DV0 : ℕ := 3000000

/-- The index variable of dimension symbol `i`. -/
def dvar (i : ℕ) : ℕ := DV0 + i

/-- The monomial `d_i`. -/
def unitD (i : ℕ) : DMon := ⟨1, List.replicate i 0 ++ [1]⟩

theorem npow_unit (f : ℕ → ℕ) (k i : ℕ) : npow f k (List.replicate i 0 ++ [1]) = f (k + i) := by
  induction i generalizing k with
  | zero => simp [npow]
  | succ i ih =>
    rw [List.replicate_succ, List.cons_append, npow, ih]
    simp only [pow_zero, one_mul]
    congr 1; omega

theorem unitD_den (dv : ℕ → ℕ) (i : ℕ) : (unitD i).den dv = dv i := by
  simp [DMon.den, unitD, npow_unit]

/-- Each dimension variable holds its symbol. -/
def DimsOK (dv : ℕ → ℕ) (ν : ℕ → ℤ) : Prop := ∀ i, ν (dvar i) = dv i

/-- The valuation `ν` below `DV0`, the symbols from `DV0` on. -/
def withDims (dv : ℕ → ℕ) (ν : ℕ → ℤ) : ℕ → ℤ := fun x => if x < DV0 then ν x else dv (x - DV0)

theorem withDims_ok (dv : ℕ → ℕ) (ν : ℕ → ℤ) : DimsOK dv (withDims dv ν) := by
  intro i
  simp [withDims, dvar]

/-! ## Products over lists -/

theorem prodSet_mul (l : List ℂ) (j : ℕ) (hj : j < l.length) (a w : ℂ) (h : l[j] = a * w) :
    (l.set j a).prod * w = l.prod := by
  have h1 := List.prod_set l j a
  have h2 := List.prod_set l j l[j]
  rw [List.set_getElem_self] at h2
  rw [h1, if_pos hj]
  conv_rhs => rw [h2, if_pos hj, h]
  ring

theorem map_set_prod (v : Fac → ℂ) (l : List Fac) (j : ℕ) (hj : j < l.length) (a : Fac) (w : ℂ)
    (h : v l[j] = v a * w) : ((l.set j a).map v).prod * w = (l.map v).prod := by
  rw [List.map_set]
  exact prodSet_mul _ j (by simpa using hj) _ _ (by simpa using h)

/-! ## Sums under an invariant -/

theorem sumsV_congr_P (dv : ℕ → ℕ) (P : (ℕ → ℤ) → Prop) : ∀ (s R : List (ℕ × DMon))
    (f g : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ),
    (∀ p ∈ R, p.1 ∈ s.map (·.1) → p ∈ s ∧ (s.map (·.1)).count p.1 = 1) →
    (∀ p ∈ R, p.1 ∉ s.map (·.1) → 0 ≤ ν p.1 ∧ ν p.1 < ((p.2.den dv : ℕ) : ℤ)) →
    (∀ q ∈ s, ∀ ν' (j : ℤ), P ν' → P (Function.update ν' q.1 j)) → P ν →
    (∀ ν', P ν' → InR dv R ν' → f ν' = g ν') → sumsV dv s f ν = sumsV dv s g ν
  | [], R, f, g, ν, _, hout, _, hν, hfg => hfg ν hν fun p hp => hout p hp (by simp)
  | q :: s, R, f, g, ν, hin, hout, hP, hν, hfg => by
    simp only [sumsV]
    refine Finset.sum_congr rfl fun j hj => sumsV_congr_P dv P s R f g _ ?_ ?_
      (fun q' hq' => hP q' (List.mem_cons_of_mem _ hq')) (hP q List.mem_cons_self _ _ hν) hfg
    · intro p hp hps
      obtain ⟨hmem, hc⟩ := hin p hp (by simp only [List.map_cons, List.mem_cons]; exact Or.inr hps)
      have hpos : 0 < (s.map (·.1)).count p.1 := List.count_pos_iff.2 hps
      have hne : p.1 ≠ q.1 := by
        intro e
        rw [List.map_cons, List.count_cons, e] at hc
        simp only [beq_self_eq_true, ite_true] at hc
        rw [e] at hpos
        omega
      refine ⟨?_, ?_⟩
      · rcases List.mem_cons.1 hmem with e | h
        · exact absurd (by rw [e]) hne
        · exact h
      · rw [List.map_cons, List.count_cons] at hc
        simpa [Ne.symm hne] using hc
    · intro p hp hps
      by_cases e : p.1 = q.1
      · have hq : p.1 ∈ (q :: s).map (·.1) := by simp [e]
        obtain ⟨hmem, hc⟩ := hin p hp hq
        have hpq : p = q := by
          rcases List.mem_cons.1 hmem with h | h
          · exact h
          · exact absurd (List.mem_map_of_mem (f := (·.1)) h) hps
        subst hpq
        rw [Function.update_self]
        have := List.mem_range.1 hj
        exact ⟨by positivity, by exact_mod_cast this⟩
      · rw [Function.update_of_ne e]
        exact hout p hp (by simp only [List.map_cons, List.mem_cons, not_or]; exact ⟨e, hps⟩)

/-- The bound variables are distinct, below `DV0` and apart from the labels of `R`. -/
def sumsOK (R s : List (ℕ × DMon)) : Bool :=
  decide (s.map (·.1)).Nodup && s.all (fun p => decide (p.1 < DV0)) &&
    R.all (fun p => !(s.map (·.1)).contains p.1)

theorem sumsOK_spec (R s : List (ℕ × DMon)) (h : sumsOK R s = true) :
    (s.map (·.1)).Nodup ∧ (∀ p ∈ s, p.1 < DV0) ∧ ∀ p ∈ R, p.1 ∉ s.map (·.1) := by
  simp only [sumsOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, Bool.not_eq_true',
    List.contains_eq_mem, decide_eq_false_iff_not] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- `sumsV_congr_P` at `DimsOK` with the ranges of `R` and of the sums. -/
theorem sumsV_congr_D (dv : ℕ → ℕ) (R s : List (ℕ × DMon)) (h : sumsOK R s = true)
    (f g : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) (hν : DimsOK dv ν) (hR : InR dv R ν)
    (hfg : ∀ ν', DimsOK dv ν' → InR dv (R ++ s) ν' → f ν' = g ν') :
    sumsV dv s f ν = sumsV dv s g ν := by
  obtain ⟨hnd, hlt, hap⟩ := sumsOK_spec R s h
  refine sumsV_congr_P dv (DimsOK dv) s (R ++ s) f g ν ?_ ?_ ?_ hν hfg
  · intro p hp hps
    rcases List.mem_append.1 hp with hp | hp
    · exact absurd hps (hap p hp)
    · exact ⟨hp, List.count_eq_one_of_mem hnd hps⟩
  · intro p hp hps
    rcases List.mem_append.1 hp with hp | hp
    · exact hR p hp
    · exact absurd (List.mem_map_of_mem (f := (·.1)) hp) hps
  · intro q hq ν' j hν' i
    have : dvar i ≠ q.1 := by have := hlt q hq; simp only [dvar]; omega
    rw [Function.update_of_ne this]; exact hν' i

/-! ## Polynomials divisible by a variable -/

/-- The monomials without `v`. -/
def ipDropV (v : ℕ) (p : IPoly) : IPoly := p.filter fun t => !t.1.contains v

/-- The monomials with `v`, `v` taken out once. -/
def ipTakeV (v : ℕ) (p : IPoly) : IPoly :=
  (p.filter fun t => t.1.contains v).map fun t => (t.1.erase v, t.2)

theorem ipDen_takeV (ν : ℕ → ℤ) (v : ℕ) (p : IPoly) :
    ipDen ν (p.filter fun t => t.1.contains v) = ν v * ipDen ν (ipTakeV v p) := by
  induction p with
  | nil => simp [ipTakeV, ipDen]
  | cons t p ih =>
    by_cases h : t.1.contains v = true
    · simp only [ipTakeV, List.filter_cons, h, ite_true, List.map_cons, ipDen, List.sum_cons] at ih ⊢
      rw [ih, imDen_erase ν v t.1 h]; ring
    · simp only [Bool.not_eq_true] at h
      simp only [ipTakeV, List.filter_cons, h, Bool.false_eq_true, ite_false] at ih ⊢
      exact ih

theorem ipDen_splitV (ν : ℕ → ℤ) (v : ℕ) (p : IPoly) :
    ipDen ν p = ν v * ipDen ν (ipTakeV v p) + ipDen ν (ipDropV v p) := by
  rw [ipDen_filter ν (fun t => t.1.contains v) p, ipDen_takeV]; rfl

theorem dvd_of_allV (ν : ℕ → ℤ) (v : ℕ) (p : IPoly) (h : p.all (fun t => t.1.contains v) = true) :
    ν v ∣ ipDen ν p := by
  have : ipDropV v p = [] := by
    simp only [ipDropV, List.filter_eq_nil_iff, Bool.not_eq_true']
    intro t ht; simpa using List.all_eq_true.1 h t ht
  rw [ipDen_splitV ν v p, this]
  simp [ipDen]

/-! ## Steps on a term -/

/-- A step on one term. -/
inductive DgOp where
  | pin (p j : ℕ)
  | comb (j k : ℕ) (c : ℤ)
  | merge (j i : ℕ)
  | pchar (p i k : ℕ)
  | dsplit (j i : ℕ)
  | impl (j i : ℕ)
  deriving DecidableEq, Repr

/-- The pin of the sum at `p` by the delta at `j`. -/
def opPin (R : List (ℕ × DMon)) (p j : ℕ) (t : STm) : Option STm :=
  match t.sums[p]?, t.dels[j]? with
  | some (x, D), some f =>
    match ipLin x f.2 with
    | some (a, b) =>
      match ipConst a with
      | some s =>
        match pinVal s b with
        | [([y], 1)] =>
          if (s == 1 || s == -1) && f.1 == D && y != x && D.c != 0 &&
              (R ++ t.sums).lookup y == some D then
            some ⟨t.c, t.sums.eraseIdx p, t.dels.map (facSub x (ipV y)),
              t.phs.map (facSub x (ipV y))⟩
          else none
        | _ => none
      | none => none
    | none => none
  | _, _ => none

/-- The delta at `j` plus `c` times the delta at `k`. -/
def opComb (j k : ℕ) (c : ℤ) (t : STm) : Option STm :=
  match t.dels[j]?, t.dels[k]? with
  | some f, some g =>
    if j != k && f.1 == g.1 then
      some ⟨t.c, t.sums, t.dels.set j (f.1, ipAdd f.2 (ipSmul c g.2)), t.phs⟩
    else none
  | _, _ => none

/-- The root of unity at `j` over the modulus times `d_i`. -/
def opMerge (j i : ℕ) (t : STm) : Option STm :=
  match t.phs[j]? with
  | some f =>
    if f.1.c != 0 then
      some ⟨t.c, t.sums, t.dels, t.phs.set j (f.1.mul (unitD i), ipMul (ipV (dvar i)) f.2)⟩
    else none
  | none => none

/-- The partial character sum of the sum at `p` against the delta at `k`. -/
def opPchar (p i k : ℕ) (t : STm) : Option STm :=
  match t.sums[p]?, t.dels[k]? with
  | some (x, M), some g =>
    if M.c != 0 && g.1 == unitD i && t.dels.all (fun f => !facHas x f) then
      match charData x (M.mul (unitD i)) t.phs with
      | some (A, B) =>
        if (ipSub A g.2).all (fun u => u.1.contains (dvar i)) then
          some ⟨CMon.mul t.c (CMon.rtD M 2), t.sums.eraseIdx p,
            t.dels ++ [(M.mul (unitD i), A)],
            t.phs.filter (fun f => !facHas x f) ++ [(M.mul (unitD i), B)]⟩
        else none
      | none => none
    else none
  | _, _ => none

/-- Whether `p` is `0`, `±a` or `±(a - b)` with `a`, `b` of range `Q` in `rg`. -/
def smallB (rg : List (ℕ × DMon)) (Q : DMon) : IPoly → Bool
  | [] => true
  | [([a], c)] => (c == 1 || c == -1) && rg.lookup a == some Q
  | [([a], c), ([b], e)] =>
    (c == 1 || c == -1) && c + e == 0 && rg.lookup a == some Q && rg.lookup b == some Q
  | _ => false

/-- The delta at `j`, `[M d_i ∣ d_i A + B]`, as `[d_i ∣ B] [M ∣ A]`. -/
def opSplit (R : List (ℕ × DMon)) (j i : ℕ) (t : STm) : Option STm :=
  match t.dels[j]? with
  | some f =>
    if f.1.c != 0 && (unitD i).dvd f.1 && smallB (R ++ t.sums) (unitD i) (ipDropV (dvar i) f.2) then
      some ⟨t.c, t.sums, t.dels.set j (unitD i, ipDropV (dvar i) f.2) ++
        [(f.1.div (unitD i), ipTakeV (dvar i) f.2)], t.phs⟩
    else none
  | none => none

/-- `[d_i ∣ B]` added beside the delta at `j`, `[M ∣ d_i A + B]` with `d_i ∣ M`. -/
def opImpl (j i : ℕ) (t : STm) : Option STm :=
  match t.dels[j]? with
  | some f =>
    if (unitD i).dvd f.1 then
      some ⟨t.c, t.sums, t.dels ++ [(unitD i, ipDropV (dvar i) f.2)], t.phs⟩
    else none
  | none => none

/-- One step on a term. -/
def dgOp (R : List (ℕ × DMon)) : DgOp → STm → Option STm
  | .pin p j => opPin R p j
  | .comb j k c => opComb j k c
  | .merge j i => opMerge j i
  | .pchar p i k => opPchar p i k
  | .dsplit j i => opSplit R j i
  | .impl j i => opImpl j i

/-- A term's value is kept wherever the dimension variables hold and `R` lies in range. -/
def KeepsD (ρ : Val) (R : List (ℕ × DMon)) (t t' : STm) : Prop :=
  ∀ ν, DimsOK ρ.dv ν → InR ρ.dv R ν → t'.den ρ ν = t.den ρ ν

/-- A step that changes only the summand. -/
theorem keeps_body (ρ : Val) (R : List (ℕ × DMon)) (t t' : STm) (hc : t'.c = t.c)
    (hs : t'.sums = t.sums) (hok : sumsOK R t.sums = true)
    (hb : ∀ ν', DimsOK ρ.dv ν' → InR ρ.dv (R ++ t.sums) ν' → body ρ.dv t' ν' = body ρ.dv t ν') :
    KeepsD ρ R t t' := by
  intro ν hν hR
  unfold STm.den
  rw [hc, hs, sumsV_congr_D ρ.dv R t.sums hok _ _ ν hν hR hb]

theorem lookupD_mem (rg : List (ℕ × DMon)) (a : ℕ) (Q : DMon) (h : rg.lookup a = some Q) :
    (a, Q) ∈ rg := by
  induction rg with
  | nil => simp at h
  | cons q rg ih =>
    obtain ⟨b, D⟩ := q
    by_cases e : a = b
    · subst e; simp [List.lookup] at h; simp [h]
    · simp only [List.lookup, show (a == b) = false from by simpa using e] at h
      exact List.mem_cons_of_mem _ (ih h)

theorem inR_lookup (dv : ℕ → ℕ) (rg : List (ℕ × DMon)) (ν : ℕ → ℤ) (h : InR dv rg ν) (a : ℕ)
    (Q : DMon) (hl : rg.lookup a = some Q) : 0 ≤ ν a ∧ ν a < ((Q.den dv : ℕ) : ℤ) :=
  h _ (lookupD_mem rg a Q hl)

/-! ### comb -/

theorem opComb_ok (ρ : Val) (R : List (ℕ × DMon)) (j k : ℕ) (c : ℤ) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : opComb j k c t = some t') : KeepsD ρ R t t' := by
  unfold opComb at h
  split at h
  · next f g hf hg =>
    split_ifs at h with hc
    simp only [Option.some.injEq] at h
    subst h
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at hc
    obtain ⟨hjk, hm⟩ := hc
    obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 hf
    obtain ⟨hk, rfl⟩ := List.getElem?_eq_some_iff.1 hg
    refine keeps_body ρ R _ _ rfl rfl hok fun ν _ _ => ?_
    simp only [body_eq, bodyOf]
    congr 1
    have hmem : t.dels[k] ∈ t.dels.set j (t.dels[j].1, ipAdd t.dels[j].2 (ipSmul c t.dels[k].2)) := by
      have hk' : k < (t.dels.set j (t.dels[j].1, ipAdd t.dels[j].2 (ipSmul c t.dels[k].2))).length := by
        simpa using hk
      have := List.getElem_mem hk'
      rwa [List.getElem_set_ne hjk] at this
    rcases delV_cases ρ.dv ν t.dels[k] with h0 | h1
    · rw [List.prod_eq_zero (List.mem_map.2 ⟨_, hmem, h0⟩),
        List.prod_eq_zero (List.mem_map.2 ⟨_, List.getElem_mem hk, h0⟩)]
    · have hd := delV_one ρ.dv ν _ h1
      have e := map_set_prod (delV ρ.dv ν) t.dels j hj
        (t.dels[j].1, ipAdd t.dels[j].2 (ipSmul c t.dels[k].2)) 1 (by
          rw [mul_one]
          refine (delV_congr ρ.dv ν (t.dels[j].1, ipAdd t.dels[j].2 (ipSmul c t.dels[k].2))
            t.dels[j] rfl ?_)
          simp only
          rw [ipDen_add, ipDen_smul]
          rw [← hm] at hd
          constructor
          · intro h'; have := dvd_sub h' (Dvd.dvd.mul_left hd c); simpa using this
          · intro h'; exact dvd_add h' (Dvd.dvd.mul_left hd _))
      rw [mul_one] at e
      exact e
  · simp at h

/-! ### impl -/

theorem opImpl_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × DMon)) (j i : ℕ) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : opImpl j i t = some t') : KeepsD ρ R t t' := by
  unfold opImpl at h
  split at h
  · next f hf =>
    split_ifs at h with hd
    simp only [Option.some.injEq] at h
    subst h
    obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 hf
    refine keeps_body ρ R _ _ rfl rfl hok fun ν hν _ => ?_
    simp only [body_eq]
    rw [bodyOf, bodyOf, List.map_append, List.prod_append]
    rcases delV_cases ρ.dv ν t.dels[j] with h0 | h1
    · rw [List.prod_eq_zero (List.mem_map.2 ⟨_, List.getElem_mem hj, h0⟩)]; ring
    · have hdv := delV_one ρ.dv ν _ h1
      have hq : ((ρ.dv i : ℕ) : ℤ) ∣ ipDen ν t.dels[j].2 := by
        have := (Int.natCast_dvd_natCast.2 (DMon.dvd_den ρ.dv _ _ hd)).trans hdv
        rwa [unitD_den] at this
      have hnew : delV ρ.dv ν (unitD i, ipDropV (dvar i) t.dels[j].2) = 1 := by
        unfold delV
        rw [if_pos]
        simp only [unitD_den]
        rw [ipDen_splitV ν (dvar i), hν i] at hq
        have := (Int.dvd_add_right (Dvd.intro _ rfl)).1 hq
        exact this
      simp [hnew]
  · simp at h

/-! ### merge -/

theorem opMerge_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × DMon)) (j i : ℕ) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : opMerge j i t = some t') : KeepsD ρ R t t' := by
  unfold opMerge at h
  split at h
  · next f hf =>
    split_ifs at h with hc
    simp only [Option.some.injEq] at h
    subst h
    simp only [bne_iff_ne, ne_eq] at hc
    obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 hf
    refine keeps_body ρ R _ _ rfl rfl hok fun ν hν _ => ?_
    simp only [body_eq, bodyOf]
    congr 1
    have e := map_set_prod (phV ρ.dv ν) t.phs j hj
      (t.phs[j].1.mul (unitD i), ipMul (ipV (dvar i)) t.phs[j].2) 1 (by
        rw [mul_one]
        unfold phV
        simp only
        rw [DMon.den_mul, unitD_den, ipDen_mul, ipDen_V, hν i]
        have hM := DMon.den_pos ρ.dv hρ.dpos _ (Nat.pos_of_ne_zero hc)
        have hd := hρ.dpos i
        rw [om_sub (t.phs[j].1.den ρ.dv * ρ.dv i) (t.phs[j].1.den ρ.dv) (ρ.dv i)
          (Nat.mul_pos hM hd) rfl, ← zpow_natCast, ← zpow_mul])
    rw [mul_one] at e
    exact e
  · simp at h

/-! ### dsplit -/

theorem dvd_small (q m A B : ℤ) (hq : 0 < q) (hm : m ≠ 0) (hB : -q < B ∧ B < q) :
    q * m ∣ q * A + B ↔ q ∣ B ∧ m ∣ A := by
  have key : q ∣ B → B = 0 := by
    intro h
    obtain ⟨k, rfl⟩ := h
    have : -1 < k ∧ k < 1 := by
      constructor
      · by_contra hk; push_neg at hk; nlinarith
      · by_contra hk; push_neg at hk; nlinarith
    have : k = 0 := by omega
    simp [this]
  constructor
  · intro h
    have hqB : q ∣ B := (Int.dvd_add_right (Dvd.intro A rfl)).1 ((Dvd.intro m rfl).trans h)
    have hB0 := key hqB
    rw [hB0, add_zero] at h
    exact ⟨hqB, (mul_dvd_mul_iff_left hq.ne').1 h⟩
  · rintro ⟨hqB, hmA⟩
    rw [key hqB, add_zero]
    exact mul_dvd_mul_left q hmA

theorem smallB_spec (dv : ℕ → ℕ) (rg : List (ℕ × DMon)) (Q : DMon) (p : IPoly) (ν : ℕ → ℤ)
    (hr : InR dv rg ν) (hQ : 0 < Q.den dv) (h : smallB rg Q p = true) :
    -((Q.den dv : ℕ) : ℤ) < ipDen ν p ∧ ipDen ν p < ((Q.den dv : ℕ) : ℤ) := by
  have hq : 0 < ((Q.den dv : ℕ) : ℤ) := by exact_mod_cast hQ
  match p, h with
  | [], _ => simp only [ipDen, List.map_nil, List.sum_nil]; constructor <;> omega
  | [([a], c)], h =>
    simp only [smallB, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
    obtain ⟨hc, hl⟩ := h
    have ha := inR_lookup dv rg ν hr a Q hl
    simp only [ipDen, imDen, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      mul_one, List.sum_cons, List.sum_nil, add_zero]
    rcases hc with rfl | rfl <;> constructor <;> omega
  | [([a], c), ([b], e)], h =>
    simp only [smallB, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨hc, hce⟩, hla⟩, hlb⟩ := h
    have ha := inR_lookup dv rg ν hr a Q hla
    have hb := inR_lookup dv rg ν hr b Q hlb
    simp only [ipDen, imDen, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
      mul_one, List.sum_cons, List.sum_nil, add_zero]
    have he : e = -c := by omega
    subst he
    rcases hc with rfl | rfl <;> constructor <;> omega

theorem opSplit_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × DMon)) (j i : ℕ) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : opSplit R j i t = some t') : KeepsD ρ R t t' := by
  unfold opSplit at h
  split at h
  · next f hf =>
    split_ifs at h with hd
    simp only [Option.some.injEq] at h
    subst h
    simp only [Bool.and_eq_true] at hd
    obtain ⟨⟨hc0, hdv⟩, hsm⟩ := hd
    simp only [bne_iff_ne, ne_eq] at hc0
    obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 hf
    refine keeps_body ρ R _ _ rfl rfl hok fun ν hν hR => ?_
    simp only [body_eq]
    rw [bodyOf, bodyOf, List.map_append, List.prod_append, List.map_cons, List.map_nil,
      List.prod_cons, List.prod_nil, mul_one]
    have hsm' := smallB_spec ρ.dv _ _ _ ν hR (by rw [unitD_den]; exact hρ.dpos i) hsm
    rw [unitD_den] at hsm'
    have hm : 0 < (t.dels[j].1.div (unitD i)).den ρ.dv :=
      DMon.den_pos ρ.dv hρ.dpos _ (by simp [DMon.div, unitD]; omega)
    have hiff : ((t.dels[j].1.den ρ.dv : ℕ) : ℤ) ∣ ipDen ν t.dels[j].2 ↔
        (((unitD i).den ρ.dv : ℕ) : ℤ) ∣ ipDen ν (ipDropV (dvar i) t.dels[j].2) ∧
          (((t.dels[j].1.div (unitD i)).den ρ.dv : ℕ) : ℤ) ∣
            ipDen ν (ipTakeV (dvar i) t.dels[j].2) := by
      rw [DMon.den_div ρ.dv _ _ hdv, unitD_den, ipDen_splitV ν (dvar i), hν i]
      push_cast
      exact dvd_small _ _ _ _ (by exact_mod_cast hρ.dpos i) (by exact_mod_cast hm.ne') hsm'
    have e := map_set_prod (delV ρ.dv ν) t.dels j hj (unitD i, ipDropV (dvar i) t.dels[j].2)
      (delV ρ.dv ν (t.dels[j].1.div (unitD i), ipTakeV (dvar i) t.dels[j].2)) (by
        unfold delV
        simp only [hiff]
        by_cases hA : (((unitD i).den ρ.dv : ℕ) : ℤ) ∣ ipDen ν (ipDropV (dvar i) t.dels[j].2) <;>
          by_cases hB : (((t.dels[j].1.div (unitD i)).den ρ.dv : ℕ) : ℤ) ∣
            ipDen ν (ipTakeV (dvar i) t.dels[j].2) <;> simp [hA, hB])
    rw [← e]
  · simp at h


/-! ### pin -/

theorem eq_zero_of_dvd_small (q B : ℤ) (hB : -q < B ∧ B < q) (h : q ∣ B) : B = 0 := by
  obtain ⟨k, rfl⟩ := h
  have hq : 0 < q := by linarith
  have : -1 < k ∧ k < 1 := by
    constructor
    · by_contra hk; push_neg at hk; nlinarith
    · by_contra hk; push_neg at hk; nlinarith
  have : k = 0 := by omega
  simp [this]

theorem sumsOK_sub (R s s' : List (ℕ × DMon)) (h : sumsOK R s = true) (hs : s'.Sublist s) :
    sumsOK R s' = true := by
  obtain ⟨hnd, hlt, hap⟩ := sumsOK_spec R s h
  simp only [sumsOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, Bool.not_eq_true',
    List.contains_eq_mem, decide_eq_false_iff_not]
  refine ⟨⟨hnd.sublist (hs.map _), fun p hp => hlt p (hs.subset hp)⟩, fun p hp hm => hap p hp ?_⟩
  exact (hs.map _).subset hm

theorem sums_split (s : List (ℕ × DMon)) (p : ℕ) (hp : p < s.length) :
    s = s.take p ++ s[p] :: s.drop (p + 1) ∧ s.eraseIdx p = s.take p ++ s.drop (p + 1) := by
  refine ⟨?_, List.eraseIdx_eq_take_drop_succ _ _⟩
  conv_lhs => rw [← List.take_append_drop p s, List.drop_eq_getElem_cons hp]

theorem not_mem_split (s : List (ℕ × DMon)) (p : ℕ) (hp : p < s.length)
    (hnd : (s.map (·.1)).Nodup) : s[p].1 ∉ (s.take p ++ s.drop (p + 1)).map (·.1) := by
  have hs := (sums_split s p hp).1
  rw [hs, List.map_append, List.map_cons] at hnd
  have := List.nodup_middle.1 hnd
  rw [List.nodup_cons] at this
  simpa using this.1

theorem opPin_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × DMon)) (p j : ℕ) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : opPin R p j t = some t') : KeepsD ρ R t t' := by
  unfold opPin at h
  split at h
  · next x D f hx hf =>
    split at h
    · next a b hlin =>
      split at h
      · next s hs =>
        split at h
        · next y hy =>
          split_ifs at h with hc
          simp only [Option.some.injEq] at h
          subst h
          simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hc
          obtain ⟨⟨⟨⟨hs1, hfD⟩, hyx⟩, hDc⟩, hly⟩ := hc
          obtain ⟨hp, hpx⟩ := List.getElem?_eq_some_iff.1 hx
          obtain ⟨hj, rfl⟩ := List.getElem?_eq_some_iff.1 hf
          obtain ⟨hnd, _, _⟩ := sumsOK_spec R t.sums hok
          obtain ⟨hsp, her⟩ := sums_split t.sums p hp
          have hxn := not_mem_split t.sums p hp hnd
          rw [hpx] at hxn
          have hok' : sumsOK R (t.sums.eraseIdx p) = true :=
            sumsOK_sub R _ _ hok (List.eraseIdx_sublist _ _)
          obtain ⟨ha, hb, hpe⟩ := ipLin_spec x _ a b hlin
          intro ν hν hR
          unfold STm.den
          simp only
          conv_rhs => rw [hsp, hpx, sumsV_pull ρ.dv _ _ x D hxn]
          rw [← her]
          congr 1
          refine (sumsV_congr_D ρ.dv R _ hok' _ _ ν hν hR fun ν' _ hR' => ?_).symm
          have hyr : (y, D) ∈ R ++ t.sums.eraseIdx p := by
            have hm := lookupD_mem _ y D hly
            rw [her]
            rw [hsp, hpx] at hm
            simp only [List.mem_append, List.mem_cons, Prod.mk.injEq] at hm ⊢
            rcases hm with hm | hm | ⟨hm, _⟩ | hm
            · exact Or.inl hm
            · exact Or.inr (Or.inl hm)
            · exact absurd hm hyx
            · exact Or.inr (Or.inr hm)
          obtain ⟨hw0, hw1⟩ := hR' _ hyr
          simp only at hw0 hw1
          have hD := DMon.den_pos ρ.dv hρ.dpos D (Nat.pos_of_ne_zero hDc)
          simp only [body_eq]
          rw [bodyOf_sub, ipDen_V]
          have hwn : (ν' y).toNat < D.den ρ.dv := by omega
          rw [Finset.sum_eq_single_of_mem (ν' y).toNat (Finset.mem_range.2 hwn)]
          · rw [Int.toNat_of_nonneg hw0]
          · intro k hk hne
            apply bodyOf_zero ρ.dv _ _ _ _ (List.getElem_mem hj)
            unfold delV
            rw [if_neg]
            rw [hpe, ipConst_spec a s hs, ipDen_upd_of _ _ _ _ hb, Function.update_self, hfD]
            intro hdv
            have hs' : s = 1 ∨ s = -1 := hs1
            rw [pin_dvd s hs' _ _ _] at hdv
            have hv : -s * ipDen ν' b = ν' y := by
              have := congrArg (ipDen ν') hy
              simp only [pinVal, ipDen_smul] at this
              rw [this]; simp [ipDen, imDen]
            rw [hv] at hdv
            have hk' := Finset.mem_range.1 hk
            have := eq_zero_of_dvd_small _ _ (by constructor <;> omega) hdv
            omega
        · simp at h
      · simp at h
    · simp at h
  · simp at h

/-! ### pchar -/

theorem sum_om_part (M q : ℕ) (hM : 0 < M) (hq : 0 < q) (A B : ℤ) (hA : (q : ℤ) ∣ A) :
    ∑ j ∈ range M, om (M * q) ^ ((j : ℤ) * A + B) =
      om (M * q) ^ B * (if (((M * q : ℕ) : ℕ) : ℤ) ∣ A then (M : ℂ) else 0) := by
  obtain ⟨m, rfl⟩ := hA
  have hMq : 0 < M * q := Nat.mul_pos hM hq
  have hz := om_ne_zero _ hMq
  have hterm : ∀ j : ℕ, om (M * q) ^ ((j : ℤ) * (q * m) + B) =
      om (M * q) ^ B * om M ^ ((j : ℤ) * m + 0) := by
    intro j
    rw [om_sub (M * q) M q hMq rfl, ← zpow_natCast (om (M * q)) q, ← zpow_mul, ← zpow_add₀ hz]
    congr 1; ring
  rw [Finset.sum_congr rfl fun j _ => hterm j, ← Finset.mul_sum, sum_om_lin M hM m 0]
  congr 1
  have : (((M * q : ℕ) : ℕ) : ℤ) ∣ (q : ℤ) * m ↔ (M : ℤ) ∣ m := by
    push_cast
    rw [mul_comm (q : ℤ) m]
    exact mul_dvd_mul_iff_right (by exact_mod_cast hq.ne')
  simp only [this, zpow_zero, one_mul]

theorem opPchar_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × DMon)) (p i k : ℕ) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : opPchar p i k t = some t') : KeepsD ρ R t t' := by
  unfold opPchar at h
  split at h
  · next x M g hx hg =>
    split_ifs at h with hc
    split at h
    · next A B hcd =>
      split_ifs at h with hdiv
      simp only [Option.some.injEq] at h
      subst h
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at hc
      obtain ⟨⟨hMc, hgQ⟩, hfree⟩ := hc
      obtain ⟨hp, hpx⟩ := List.getElem?_eq_some_iff.1 hx
      obtain ⟨hk, rfl⟩ := List.getElem?_eq_some_iff.1 hg
      obtain ⟨hnd, _, _⟩ := sumsOK_spec R t.sums hok
      obtain ⟨hsp, her⟩ := sums_split t.sums p hp
      have hxn := not_mem_split t.sums p hp hnd
      rw [hpx] at hxn
      have hok' : sumsOK R (t.sums.eraseIdx p) = true :=
        sumsOK_sub R _ _ hok (List.eraseIdx_sublist _ _)
      have hMp := DMon.den_pos ρ.dv hρ.dpos M (Nat.pos_of_ne_zero hMc)
      have hMQc : 0 < (M.mul (unitD i)).c := by simp [DMon.mul, unitD]; omega
      obtain ⟨hA, hB, hrec⟩ := charData_spec ρ.dv hρ.dpos x _ hMQc t.phs A B hcd
      intro ν hν hR
      unfold STm.den
      simp only
      conv_rhs => rw [hsp, hpx, sumsV_pull ρ.dv _ _ x M hxn]
      rw [← her, CMon.den_mul ρ hρ, rtD_two ρ M (Nat.pos_of_ne_zero hMc), mul_assoc,
        ← sumsV_mul_left]
      congr 1
      refine (sumsV_congr_D ρ.dv R _ hok' _ _ ν hν hR fun ν' hν' _ => ?_).symm
      simp only [body_eq, bodyOf]
      have hds : ∀ j : ℤ, (t.dels.map (delV ρ.dv (Function.update ν' x j))).prod =
          (t.dels.map (delV ρ.dv ν')).prod := by
        intro j
        have := bodyOf_upd ρ.dv x t.dels [] (by simpa using hfree) ν' j
        simpa [bodyOf] using this
      simp only [hds, hrec]
      rw [← Finset.mul_sum, ← Finset.mul_sum]
      rcases delV_cases ρ.dv ν' t.dels[k] with h0 | h1
      · have hz : (t.dels.map (delV ρ.dv ν')).prod = 0 :=
          List.prod_eq_zero (List.mem_map.2 ⟨_, List.getElem_mem hk, h0⟩)
        simp only [List.map_append, List.prod_append, hz, zero_mul, mul_zero]
      · have hdk := delV_one ρ.dv ν' _ h1
        rw [hgQ, unitD_den] at hdk
        have hdA : ((ρ.dv i : ℕ) : ℤ) ∣ ipDen ν' A := by
          have h2 := dvd_of_allV ν' (dvar i) _ hdiv
          rw [hν' i, ipDen_sub] at h2
          have := dvd_add h2 hdk
          simpa using this
        rw [DMon.den_mul, unitD_den, sum_om_part _ _ hMp (hρ.dpos i) _ _ hdA]
        simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append,
          List.prod_cons, List.prod_nil, mul_one, delV, phV, DMon.den_mul, unitD_den]
        split_ifs <;> ring
    · simp at h
  · simp at h

/-! ## Runs -/

theorem dgOp_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × DMon)) (o : DgOp) (t t' : STm)
    (hok : sumsOK R t.sums = true) (h : dgOp R o t = some t') : KeepsD ρ R t t' := by
  cases o with
  | pin p j => exact opPin_ok ρ hρ R p j t t' hok h
  | comb j k c => exact opComb_ok ρ R j k c t t' hok h
  | merge j i => exact opMerge_ok ρ hρ R j i t t' hok h
  | pchar p i k => exact opPchar_ok ρ hρ R p i k t t' hok h
  | dsplit j i => exact opSplit_ok ρ hρ R j i t t' hok h
  | impl j i => exact opImpl_ok ρ hρ R j i t t' hok h

/-- A step on the expression: `nfEx`, or a step on one term. -/
inductive DgStep where
  | nf
  | op (k : ℕ) (o : DgOp)
  deriving DecidableEq, Repr

/-- The steps in turn. -/
def dgRun (fl : List ℕ) (R : List (ℕ × DMon)) : List DgStep → List STm → Option (List STm)
  | [], e => some e
  | .nf :: s, e => dgRun fl R s (nfEx fl e)
  | .op k o :: s, e =>
    match e[k]? with
    | some t =>
      if sumsOK R t.sums then
        match dgOp R o t with
        | some t' => dgRun fl R s (e.set k t')
        | none => none
      else none
    | none => none

theorem exDen_set (ρ : Val) (ν : ℕ → ℤ) (e : List STm) (k : ℕ) (hk : k < e.length) (t' : STm)
    (h : t'.den ρ ν = e[k].den ρ ν) : exDen ρ ν (e.set k t') = exDen ρ ν e := by
  unfold exDen
  rw [List.map_set]
  congr 1
  have : STm.den ρ ν t' = (e.map (STm.den ρ ν))[k]'(by simpa using hk) := by simpa using h
  rw [this, List.set_getElem_self]

theorem dgRun_ok (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (R : List (ℕ × DMon)) :
    ∀ (s : List DgStep) (e e' : List STm), dgRun fl R s e = some e' → ∀ ν, DimsOK ρ.dv ν →
      InR ρ.dv R ν → exDen ρ ν e' = exDen ρ ν e
  | [], e, e', h, ν, _, _ => by simp only [dgRun, Option.some.injEq] at h; rw [h]
  | .nf :: s, e, e', h, ν, hν, hR => by
    rw [dgRun_ok ρ hρ fl hfl R s _ e' h ν hν hR, nfEx_den ρ hρ fl hfl]
  | .op k o :: s, e, e', h, ν, hν, hR => by
    simp only [dgRun] at h
    split at h
    · next t ht =>
      split_ifs at h with hok
      split at h
      · next t' ho =>
        obtain ⟨hk, rfl⟩ := List.getElem?_eq_some_iff.1 ht
        rw [dgRun_ok ρ hρ fl hfl R s _ e' h ν hν hR,
          exDen_set ρ ν e k hk t' (dgOp_ok ρ hρ R o _ t' hok ho ν hν hR)]
      · simp at h
    · simp at h

/-! ## Equality -/

/-- The labels of `R` below `DV0`, boundary labels of dimension `p.2`; the steps of the
certificate. -/
abbrev DgCert := List (ℕ × DMon) × List DgStep

/-- Every label below `DV0`, and the steps leave nothing. -/
def dgOK (fl : List ℕ) (dc : DgCert) (e : List STm) : Bool :=
  e.all (fun t => (tmVars t).all (· < DV0)) &&
    match dgRun fl dc.1 dc.2 e with
    | some e' => e'.isEmpty
    | none => false

theorem exDen_reads (ρ : Val) (e : List STm) (ν ν' : ℕ → ℤ)
    (h : ∀ t ∈ e, ∀ x ∈ tmVars t, ν x = ν' x) : exDen ρ ν e = exDen ρ ν' e := by
  induction e with
  | nil => rfl
  | cons t e ih =>
    rw [exDen_cons, exDen_cons, ih fun u hu => h u (List.mem_cons_of_mem _ hu)]
    congr 1
    unfold STm.den
    rw [sumsV_reads ρ.dv (tmVars t) (body ρ.dv t) (body_reads ρ.dv t) t.sums ν ν'
      (h t List.mem_cons_self)]

theorem dgOK_zero (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ) (dc : DgCert)
    (e : List STm) (h : dgOK fl dc e = true) (ν : ℕ → ℤ) (hR : InR ρ.dv dc.1 ν)
    (hRl : ∀ p ∈ dc.1, p.1 < DV0) : exDen ρ ν e = 0 := by
  unfold dgOK at h
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  obtain ⟨hv, hr⟩ := h
  split at hr
  · next e' he =>
    have hR' : InR ρ.dv dc.1 (withDims ρ.dv ν) := by
      intro p hp
      have := hR p hp
      simp only [withDims, if_pos (hRl p hp)]
      exact this
    have := dgRun_ok ρ hρ fl hfl dc.1 dc.2 e e' he _ (withDims_ok ρ.dv ν) hR'
    rw [List.isEmpty_iff.1 hr] at this
    rw [exDen_reads ρ e ν (withDims ρ.dv ν) fun t ht x hx => by
      simp only [withDims, if_pos (hv t ht x hx)], ← this]
    rfl
  · simp at hr

/-- Each pair `(l, D)` of `R`: `l` below `DV0`, a boundary label of dimension `D`. -/
def labOKD (labs : List (ℕ × ℕ)) (IX : List ℕ) (dt : ℕ → DMon) (bnd : List ℕ)
    (R : List (ℕ × DMon)) : Bool :=
  R.all fun p => decide (p.1 < DV0) && bnd.contains p.1 &&
    IX.contains ((labs.lookup p.1).getD 0) && dt ((labs.lookup p.1).getD 0) == p.2

theorem labOKD_in (E : Env) (ρ : Val) (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (IX : List ℕ)
    (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv) (hL : E.labs = labs) (bnd : List ℕ)
    (R : List (ℕ × DMon)) (h : labOKD labs IX dt bnd R = true) (σ : ℕ → ℕ)
    (hσ : InRange E bnd σ) : InR ρ.dv R (fun x => (σ x : ℤ)) ∧ ∀ p ∈ R, p.1 < DV0 := by
  unfold labOKD at h
  refine ⟨fun p hp => ?_, fun p hp => ?_⟩
  · have hq := List.all_eq_true.1 h p hp
    simp only [Bool.and_eq_true, List.contains_iff_mem, beq_iff_eq, decide_eq_true_eq] at hq
    obtain ⟨⟨⟨_, hb⟩, hi⟩, hk⟩ := hq
    refine ⟨by positivity, ?_⟩
    have hlt := hσ p.1 hb
    unfold labDim at hlt
    rw [hL, hE _ hi, hk] at hlt
    show (σ p.1 : ℤ) < _
    exact_mod_cast hlt
  · have hq := List.all_eq_true.1 h p hp
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hq
    exact hq.1.1.1

/-- `symCheck` with the steps of `dc` on the difference and the labels' ranges by `labOKD`. -/
def symCheckD (dc : DgCert) (fl : List ℕ) (dt : ℕ → DMon)
    (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (N0 N1 : Net) (S0 S1 : List STm) : Bool :=
  N0.bnd.isPerm N1.bnd && labOKD labs IX dt N0.bnd dc.1 && netChk fl dt pt rl labs IX N0 &&
    netChk fl dt pt rl labs IX N1 && exConst S0 && exConst S1 &&
    match trNet fl dt pt rl labs N0, trNet fl dt pt rl labs N1 with
    | some e0, some e1 => exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
        dgOK fl dc (exMul S0 e0 ++ exMul (exNeg S1) e1)
    | _, _ => false

theorem equal_symD (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dc : DgCert)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : symCheckD dc fl dt pt rl labs IX N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold symCheckD at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨hperm, hlab⟩, hc0⟩, hc1⟩, hk0⟩, hk1⟩, hrest⟩ := h
  have hp : N0.bnd.Perm N1.bnd := List.isPerm_iff.1 hperm
  split at hrest
  · next e0 e1 ht0 ht1 =>
    simp only [Bool.and_eq_true] at hrest
    obtain ⟨⟨hm0, hm1⟩, hdg⟩ := hrest
    refine ⟨hp, fun σ hσ => ?_⟩
    set ν : ℕ → ℤ := fun x => (σ x : ℤ)
    obtain ⟨hin, hlt⟩ := labOKD_in E ρ dt labs IX hE hL N0.bnd dc.1 hlab σ hσ
    have hz := dgOK_zero ρ hρ fl hfl dc _ hdg ν hin hlt
    rw [exDen_append, exDen_mul ρ hρ _ _ hm0, exDen_mul ρ hρ _ _ hm1, exDen_neg,
      ← eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e0 hc0 ht0 σ hσ,
      ← eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e1 hc1 ht1 σ (hσ.perm hp),
      exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
      ← hs1] at hz
    linear_combination hz
  · simp at hrest

end ArchytasZX.Sym
