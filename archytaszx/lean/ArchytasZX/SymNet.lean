/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymCanon

/-!
# Networks as symbolic sums

`trNode` writes a generator's entry as an expression over its leg labels and one bound
variable, `trNet` a network's entry as an expression over its boundary labels. `equal_sym`
proves `Equal` from the normal form of the difference of the two scaled networks being empty,
for every value of the dimension symbols and coefficient atoms.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-- The first node-bound variable; labels lie below it. -/
def OFF : ℕ := 1000000

/-- The sum of the variables `ls`. -/
def sumL (ls : List ℕ) : IPoly := ls.foldr (fun l acc => ipAdd acc (ipV l)) []

/-- Every leg equal to `s`. -/
def zLegs (D : DMon) (s : ℕ) (ls : List ℕ) : List Fac := ls.map fun l => (D, ipSub (ipV l) (ipV s))

/-- The phase corrections of a spider: per stored phase `(j, A)`, `(A - 1) [s = j]`. -/
def phTerms (k : CMon) (D : DMon) (s : ℕ) (pt : List (ℕ × CMon)) (ds ps : List Fac) : List STm :=
  pt.flatMap fun p =>
    [⟨CMon.mul k p.2, [(s, D)], (D, ipSub (ipV s) (ipC p.1)) :: ds, ps⟩,
     ⟨CMon.mul k (CMon.ofQ (-1)), [(s, D)], (D, ipSub (ipV s) (ipC p.1)) :: ds, ps⟩]

/-- `k Σ_s ex(φ s) Π ds Π ps`. -/
def spider (k : CMon) (D : DMon) (s : ℕ) (pt : List (ℕ × CMon)) (ds ps : List Fac) : List STm :=
  ⟨k, [(s, D)], ds, ps⟩ :: phTerms k D s pt ds ps

/-- The value of a phase vector entry from its stored phases. -/
noncomputable def phF (ρ : Val) (pt : List (ℕ × CMon)) (j : ℕ) : ℂ :=
  1 + (pt.map fun p => if j = p.1 then p.2.den ρ - 1 else 0).sum

/-- `Σ_i c_i x^i` by Horner's rule. -/
def rEv (x : ℤ) : List ℤ → ℤ
  | [] => 0
  | c :: cs => c + x * rEv x cs

/-- `Σ_i c_i s^i` as an index polynomial. -/
def rPoly (s : ℕ) : List ℤ → IPoly
  | [] => []
  | c :: cs => ipAdd (ipC c) (ipMul (ipV s) (rPoly s cs))

theorem ipDen_rPoly (ν : ℕ → ℤ) (s : ℕ) (cs : List ℤ) : ipDen ν (rPoly s cs) = rEv (ν s) cs := by
  induction cs with
  | nil => rfl
  | cons c cs ih => simp only [rPoly, rEv, ipDen_add, ipDen_C, ipDen_mul, ipDen_V, ih]

/-- A phase rule's value `ω_D^(Σ_i c_i j^i)` at index `j`. -/
noncomputable def rV (ρ : Val) (r : DMon × List ℤ) (j : ℕ) : ℂ :=
  om (r.1.den ρ.dv) ^ rEv (j : ℤ) r.2

/-- A phase rule as root-of-unity factors at the bound variable `s`. -/
def rF (s : ℕ) (r : DMon × List ℤ) : List Fac := if r.2 = [] then [] else [(r.1, rPoly s r.2)]

theorem rF_body (dv : ℕ → ℕ) (ρ : Val) (hdv : ρ.dv = dv) (s : ℕ) (r : DMon × List ℤ)
    (ν : ℕ → ℤ) (j : ℕ) :
    ((rF s r).map (phV dv (Function.update ν s j))).prod = rV ρ r j := by
  unfold rF rV
  split_ifs with h
  · simp [h, rEv]
  · simp [phV, ipDen_rPoly, hdv]

theorem dvd_iff_eq_lt (d v k : ℕ) (hv : v < d) (hk : k < d) :
    ((d : ℕ) : ℤ) ∣ (v : ℤ) - k ↔ v = k := by
  constructor
  · intro h
    have := Int.eq_zero_of_dvd_of_natAbs_lt_natAbs h (by simp; omega)
    omega
  · intro h; subst h; simp

theorem exDen_cons (ρ : Val) (ν : ℕ → ℤ) (t : STm) (e : List STm) :
    exDen ρ ν (t :: e) = t.den ρ ν + exDen ρ ν e := by simp [exDen]

theorem exDen_append (ρ : Val) (ν : ℕ → ℤ) (a b : List STm) :
    exDen ρ ν (a ++ b) = exDen ρ ν a + exDen ρ ν b := by simp [exDen]

theorem exDen_nil (ρ : Val) (ν : ℕ → ℤ) : exDen ρ ν [] = 0 := rfl

theorem den_one (ρ : Val) (ν : ℕ → ℤ) (c : CMon) (s : ℕ) (D : DMon) (ds ps : List Fac) :
    STm.den ρ ν ⟨c, [(s, D)], ds, ps⟩ =
      c.den ρ * ∑ j ∈ range (D.den ρ.dv), bodyOf ρ.dv ds ps (Function.update ν s j) := by
  unfold STm.den; simp only [sumsV]; rfl

theorem den_zero (ρ : Val) (ν : ℕ → ℤ) (c : CMon) (ds ps : List Fac) :
    STm.den ρ ν ⟨c, [], ds, ps⟩ = c.den ρ * bodyOf ρ.dv ds ps ν := rfl

theorem phTerms_den (ρ : Val) (hρ : ρ.OK) (k : CMon) (D : DMon) (s : ℕ) (pt : List (ℕ × CMon))
    (ds ps : List Fac) (ν : ℕ → ℤ) (hsupp : ∀ p ∈ pt, p.1 < D.den ρ.dv) :
    exDen ρ ν (phTerms k D s pt ds ps) = k.den ρ *
      (pt.map fun p => (p.2.den ρ - 1) * bodyOf ρ.dv ds ps (Function.update ν s p.1)).sum := by
  induction pt with
  | nil => simp [phTerms, exDen]
  | cons p pt ih =>
    have hp := hsupp p List.mem_cons_self
    have ih' := ih fun q hq => hsupp q (List.mem_cons_of_mem _ hq)
    have hδ : ∀ j ∈ range (D.den ρ.dv), bodyOf ρ.dv ((D, ipSub (ipV s) (ipC p.1)) :: ds) ps
        (Function.update ν s j) = if j = p.1 then bodyOf ρ.dv ds ps (Function.update ν s j)
        else 0 := by
      intro j hj
      have hj' := Finset.mem_range.1 hj
      simp only [bodyOf, List.map_cons, List.prod_cons, delV, ipDen_sub, ipDen_V, ipDen_C,
        Function.update_self, dvd_iff_eq_lt _ _ _ hj' hp]
      split_ifs <;> ring
    have e : phTerms k D s (p :: pt) ds ps =
        ⟨CMon.mul k p.2, [(s, D)], (D, ipSub (ipV s) (ipC p.1)) :: ds, ps⟩ ::
        ⟨CMon.mul k (CMon.ofQ (-1)), [(s, D)], (D, ipSub (ipV s) (ipC p.1)) :: ds, ps⟩ ::
        phTerms k D s pt ds ps := rfl
    rw [e, exDen_cons, exDen_cons, ih', den_one, den_one, CMon.den_mul ρ hρ, CMon.den_mul ρ hρ,
      ofQ_den, Finset.sum_congr rfl hδ, Finset.sum_ite_eq', if_pos (Finset.mem_range.2 hp)]
    simp only [List.map_cons, List.sum_cons]
    push_cast; ring

theorem sum_ite_list (d : ℕ) (pt : List (ℕ × CMon)) (a : ℕ × CMon → ℂ) (B : ℕ → ℂ)
    (hsupp : ∀ p ∈ pt, p.1 < d) :
    ∑ j ∈ range d, (pt.map fun p => if j = p.1 then a p else 0).sum * B j =
      (pt.map fun p => a p * B p.1).sum := by
  induction pt with
  | nil => simp
  | cons p pt ih =>
    simp only [List.map_cons, List.sum_cons, add_mul, Finset.sum_add_distrib]
    rw [ih fun q hq => hsupp q (List.mem_cons_of_mem _ hq)]
    congr 1
    rw [Finset.sum_congr rfl fun j _ => by rw [ite_mul, zero_mul], Finset.sum_ite_eq',
      if_pos (Finset.mem_range.2 (hsupp p List.mem_cons_self))]

theorem spider_denA (ρ : Val) (hρ : ρ.OK) (k : CMon) (D : DMon) (s : ℕ) (pt : List (ℕ × CMon))
    (ds ps : List Fac) (ν : ℕ → ℤ) (hsupp : ∀ p ∈ pt, p.1 < D.den ρ.dv) (A : ℕ → ℂ)
    (hA : ∀ j, A j = phF ρ pt j) :
    exDen ρ ν (spider k D s pt ds ps) = k.den ρ * ∑ j ∈ range (D.den ρ.dv),
      A j * bodyOf ρ.dv ds ps (Function.update ν s j) := by
  rw [spider, exDen_cons, phTerms_den ρ hρ _ _ _ _ _ _ _ hsupp, den_one]
  have := sum_ite_list (D.den ρ.dv) pt (fun p => p.2.den ρ - 1)
    (fun j => bodyOf ρ.dv ds ps (Function.update ν s j)) hsupp
  simp only [hA, phF, add_mul, one_mul, Finset.sum_add_distrib, this]
  ring

theorem spider_den (ρ : Val) (hρ : ρ.OK) (k : CMon) (D : DMon) (s : ℕ) (pt : List (ℕ × CMon))
    (ds ps : List Fac) (ν : ℕ → ℤ) (hsupp : ∀ p ∈ pt, p.1 < D.den ρ.dv) (φ : ℕ → ℝ)
    (hφ : ∀ j, ex (φ j) = phF ρ pt j) :
    exDen ρ ν (spider k D s pt ds ps) = k.den ρ * ∑ j ∈ range (D.den ρ.dv),
      ex (φ j) * bodyOf ρ.dv ds ps (Function.update ν s j) :=
  spider_denA ρ hρ k D s pt ds ps ν hsupp _ hφ

/-- A spider whose phases carry a rule: the rule's factors at the bound variable. -/
theorem spider_den_rule (ρ : Val) (hρ : ρ.OK) (k : CMon) (D : DMon) (s : ℕ)
    (pt : List (ℕ × CMon)) (ds ps : List Fac) (r : DMon × List ℤ) (ν : ℕ → ℤ)
    (hsupp : ∀ p ∈ pt, p.1 < D.den ρ.dv) (φ : ℕ → ℝ) (hφ : ∀ j, ex (φ j) = phF ρ pt j * rV ρ r j) :
    exDen ρ ν (spider k D s pt ds (ps ++ rF s r)) = k.den ρ * ∑ j ∈ range (D.den ρ.dv),
      ex (φ j) * bodyOf ρ.dv ds ps (Function.update ν s j) := by
  rw [spider_denA ρ hρ k D s pt ds _ ν hsupp (phF ρ pt) fun _ => rfl]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [bodyOf, List.map_append, List.prod_append, rF_body ρ.dv ρ rfl, hφ]
  ring

theorem ipDen_sumL (ν : ℕ → ℤ) (ls : List ℕ) : ipDen ν (sumL ls) = (ls.map ν).sum := by
  induction ls with
  | nil => rfl
  | cons l ls ih => simp only [sumL, List.foldr_cons] at ih ⊢; rw [ipDen_add, ih, ipDen_V]; simp; ring

/-- Each leg of `ls` has a value below `d` under `σ` and the same value under `ν`. -/
def LegsAt (d : ℕ) (σ : ℕ → ℕ) (ν : ℕ → ℤ) (ls : List ℕ) : Prop := ∀ l ∈ ls, σ l < d ∧ ν l = σ l

theorem zLegs_body (dv : ℕ → ℕ) (D : DMon) (s : ℕ) (ls : List ℕ) (σ : ℕ → ℕ) (ν : ℕ → ℤ)
    (hs : s ∉ ls) (h : LegsAt (D.den dv) σ ν ls) (j : ℕ) (hj : j < D.den dv) :
    bodyOf dv (zLegs D s ls) [] (Function.update ν s j) =
      if ∀ v ∈ ls.map σ, v = j then 1 else 0 := by
  induction ls with
  | nil => simp [bodyOf, zLegs]
  | cons l ls ih =>
    have hl := h l List.mem_cons_self
    have hsl : l ≠ s := fun e => hs (e ▸ List.mem_cons_self)
    have ih' := ih (fun e => hs (List.mem_cons_of_mem _ e)) (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp only [bodyOf, zLegs, List.map_cons, List.prod_cons, List.map_nil, List.prod_nil,
      mul_one] at ih' ⊢
    rw [ih']
    simp only [delV, ipDen_sub, ipDen_V, Function.update_self, Function.update_of_ne hsl, hl.2,
      dvd_iff_eq_lt _ _ _ hl.1 hj, List.mem_cons, forall_eq_or_imp]
    by_cases h1 : σ l = j <;> simp [h1]

/-! ## Generators -/

/-- `[v = 0]` for each leg. -/
def wZeros (D : DMon) (os : List ℕ) : List Fac := os.map fun o => (D, ipV o)

/-- `Π acc · Σ_p [o_p = i] Π_{q ≠ p} [o_q = 0]`. -/
def wS (D : DMon) (i : ℕ) : List Fac → List ℕ → List STm
  | _, [] => []
  | acc, o :: os => ⟨CMon.ofQ 1, [], acc ++ (D, ipSub (ipV o) (ipV i)) :: wZeros D os, []⟩ ::
      wS D i (acc ++ [(D, ipV o)]) os

/-- A generator's entry over its legs, `s` its bound variable. -/
def trNode (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (s : ℕ) (n : Node) : Option (List STm) :=
  let D := dt n.dim
  match n.kind, n.outs, n.ins with
  | .Z, os, is => some (spider (CMon.ofQ 1) D s (pt n.ph) (zLegs D s (os ++ is)) (rF s (rl n.ph)))
  | .X, os, is => some (spider (CMon.rtD D (-((os.length + is.length : ℕ) : ℤ))) D s (pt n.ph)
      [] ((D, ipMul (ipV s) (ipSub (sumL os) (sumL is))) :: rF s (rl n.ph)))
  | .F, [o], [i] => some [⟨CMon.rtD D (-1), [], [], [(D, ipMul (ipV o) (ipV i))]⟩]
  | .T, [o], [i] => some [⟨CMon.ofQ 1, [], [(D, ipV o)], []⟩,
      ⟨CMon.ofQ 1, [], [(D, ipSub (ipV o) (ipV i))], []⟩,
      ⟨CMon.ofQ (-1), [], [(D, ipV o), (D, ipV i)], []⟩]
  | .Ti, [o], [i] => some [⟨CMon.ofQ 1, [], [(D, ipSub (ipV o) (ipV i))], []⟩,
      ⟨CMon.ofQ (-1), [], [(D, ipV o)], []⟩,
      ⟨CMon.ofQ 1, [], [(D, ipV o), (D, ipV i)], []⟩]
  | .W, os, [i] => some (wS D i [] os ++
      [⟨CMon.ofQ (1 - os.length), [], (D, ipV i) :: wZeros D os, []⟩])
  | .I, [a], [b] => some [⟨CMon.ofQ 1, [], [(D, ipSub (ipV a) (ipV b))], []⟩]
  | _, _, _ => none

/-- A node's dimension and phases agree with the tables. -/
structure NodeOK (E : Env) (ρ : Val) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (n : Node) :
    Prop where
  dim : E.dimv n.dim = (dt n.dim).den ρ.dv
  pos : 0 < (dt n.dim).c
  ph : ∀ j, ex (E.phase n.ph j) = phF ρ (pt n.ph) j * rV ρ (rl n.ph) j
  supp : ∀ p ∈ pt n.ph, p.1 < (dt n.dim).den ρ.dv

theorem delV_lt (dv : ℕ → ℕ) (ν : ℕ → ℤ) (D : DMon) (σ : ℕ → ℕ) (a b : ℕ)
    (ha : σ a < D.den dv ∧ ν a = σ a) (hb : σ b < D.den dv ∧ ν b = σ b) :
    delV dv ν (D, ipSub (ipV a) (ipV b)) = if σ a = σ b then 1 else 0 := by
  simp only [delV, ipDen_sub, ipDen_V, ha.2, hb.2, dvd_iff_eq_lt _ _ _ ha.1 hb.1]

theorem delV_zero (dv : ℕ → ℕ) (ν : ℕ → ℤ) (D : DMon) (σ : ℕ → ℕ) (a : ℕ)
    (ha : σ a < D.den dv ∧ ν a = σ a) : delV dv ν (D, ipV a) = if σ a = 0 then 1 else 0 := by
  have h0 : 0 < D.den dv := by omega
  have := dvd_iff_eq_lt _ _ _ ha.1 h0
  simp only [Nat.cast_zero, sub_zero] at this
  simp only [delV, ipDen_V, ha.2, this]

/-- All legs zero. -/
def zF (vs : List ℕ) : ℂ := if ∀ v ∈ vs, v = 0 then 1 else 0

/-- One leg equal to `I`, the others zero. -/
def sF : List ℕ → ℕ → ℂ
  | [], _ => 0
  | o :: os, I => (if o = I then zF os else 0) + (if o = 0 then sF os I else 0)

theorem tW_sF (d : ℕ) (vs : List ℕ) (I : ℕ) (hI : I < d) :
    tW d vs I = sF vs I - ((vs.length : ℂ) - 1) * (if I = 0 then zF vs else 0) := by
  by_cases h0 : I = 0
  · subst h0
    have : sF vs 0 = vs.length * zF vs := by
      induction vs with
      | nil => simp [sF]
      | cons o os ih =>
        simp only [sF, ih, zF, List.mem_cons, forall_eq_or_imp, List.length_cons]
        by_cases ho : o = 0
        · simp only [ho, if_true, true_and]; push_cast; split_ifs <;> ring
        · simp [ho]
    simp only [tW, if_true, this, zF]; ring
  · have : sF vs I = if vs.filter (· ≠ 0) = [I] then 1 else 0 := by
      induction vs with
      | nil => simp [sF]
      | cons o os ih =>
        simp only [sF, ih, List.filter_cons]
        by_cases ho : o = 0
        · subst ho; simp [Ne.symm h0]
        · have hf : (os.filter (· ≠ 0) = []) ↔ ∀ v ∈ os, v = 0 := by
            rw [List.filter_eq_nil_iff]; simp
          simp only [ho, ne_eq, not_false_eq_true, decide_true, ite_true, ite_false, add_zero,
            List.cons.injEq, zF, ← hf]
          by_cases hoI : o = I <;> simp [hoI]
    simp only [tW, h0, if_false, hI, true_and, this, mul_zero, sub_zero]

theorem wZeros_den (ρ : Val) (ν : ℕ → ℤ) (D : DMon) (σ : ℕ → ℕ) (os : List ℕ)
    (ho : ∀ o ∈ os, σ o < D.den ρ.dv ∧ ν o = σ o) :
    ((wZeros D os).map (delV ρ.dv ν)).prod = zF (os.map σ) := by
  induction os with
  | nil => simp [wZeros, zF]
  | cons o os ih =>
    have ih2 := ih fun x hx => ho x (List.mem_cons_of_mem _ hx)
    simp only [wZeros, List.map_cons, List.prod_cons] at ih2 ⊢
    rw [ih2, delV_zero _ _ _ _ _ (ho o List.mem_cons_self)]
    by_cases h : σ o = 0 <;> simp [zF, h]

theorem wS_den (ρ : Val) (ν : ℕ → ℤ) (D : DMon) (σ : ℕ → ℕ) (i : ℕ) (acc : List Fac)
    (os : List ℕ) (hi : σ i < D.den ρ.dv ∧ ν i = σ i)
    (ho : ∀ o ∈ os, σ o < D.den ρ.dv ∧ ν o = σ o) :
    exDen ρ ν (wS D i acc os) = (acc.map (delV ρ.dv ν)).prod * sF (os.map σ) (σ i) := by
  induction os generalizing acc with
  | nil => simp [wS, exDen, sF]
  | cons o os ih =>
    have hoo := ho o List.mem_cons_self
    have hos : ∀ x ∈ os, σ x < D.den ρ.dv ∧ ν x = σ x := fun x hx => ho x (List.mem_cons_of_mem _ hx)
    rw [wS, exDen_cons, ih _ hos, den_zero, ofQ_den]
    simp only [bodyOf, List.map_append, List.map_cons, List.prod_append, List.prod_cons,
      List.map_nil, List.prod_nil, mul_one, wZeros_den ρ ν D σ os hos, delV_lt _ _ _ _ _ _ hoo hi,
      delV_zero _ _ _ _ _ hoo, sF]
    push_cast
    split_ifs <;> ring

theorem legs_cons {d : ℕ} {σ : ℕ → ℕ} {ν : ℕ → ℤ} {l : ℕ} {ls : List ℕ}
    (h : LegsAt d σ ν (l :: ls)) : (σ l < d ∧ ν l = σ l) ∧ LegsAt d σ ν ls :=
  ⟨h l List.mem_cons_self, fun x hx => h x (List.mem_cons_of_mem _ hx)⟩

theorem val_tr (E : Env) (ρ : Val) (hρ : ρ.OK) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (s : ℕ) (n : Node) (e : List STm) (hn : NodeOK E ρ dt pt rl n) (σ : ℕ → ℕ) (ν : ℕ → ℤ)
    (hs : s ∉ n.outs ++ n.ins) (hl : LegsAt (E.dimv n.dim) σ ν (n.outs ++ n.ins))
    (h : trNode dt pt rl s n = some e) : val E σ n = exDen ρ ν e := by
  obtain ⟨kind, os, is, dim, ph⟩ := n
  obtain ⟨hdim, hpos, hph, hsupp⟩ := hn
  simp only at hdim hpos hph hsupp hs hl ⊢
  rw [hdim] at hl
  have hD : 0 < (dt dim).den ρ.dv := DMon.den_pos _ hρ.dpos _ hpos
  cases kind with
  | Z =>
    simp only [trNode, Option.some.injEq] at h
    subst h
    have hsp := spider_den_rule ρ hρ (CMon.ofQ 1) (dt dim) (s : ℕ) (pt ph)
      (zLegs (dt dim) s (os ++ is)) [] (rl ph) ν hsupp (E.phase ph) hph
    simp only [List.nil_append] at hsp
    rw [hsp, ofQ_den]
    simp only [val, tZ, hdim, Rat.cast_one, one_mul]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [zLegs_body _ _ _ _ σ ν hs hl j (Finset.mem_range.1 hj), List.map_append]
    split_ifs <;> simp
  | X =>
    simp only [trNode, Option.some.injEq] at h
    subst h
    have hsp := spider_den_rule ρ hρ (CMon.rtD (dt dim) (-((os.length + is.length : ℕ) : ℤ)))
      (dt dim) s (pt ph) [] [((dt dim), ipMul (ipV s) (ipSub (sumL os) (sumL is)))] (rl ph) ν
      hsupp (E.phase ph) hph
    simp only [List.singleton_append] at hsp
    rw [hsp, rtD_den ρ _ hpos]
    simp only [val, tX, hdim, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hso : ∀ x ∈ os, ν x = σ x := fun x hx => (hl x (List.mem_append_left _ hx)).2
    have hsi : ∀ x ∈ is, ν x = σ x := fun x hx => (hl x (List.mem_append_right _ hx)).2
    have hsno : s ∉ os := fun h' => hs (List.mem_append_left _ h')
    have hsni : s ∉ is := fun h' => hs (List.mem_append_right _ h')
    have e1 : ipDen (Function.update ν s j) (sumL os) = ((os.map σ).sum : ℕ) := by
      rw [ipDen_sumL]; push_cast
      rw [List.map_map]; congr 1; refine List.map_congr_left fun x hx => ?_
      simp only [Function.comp]
      rw [Function.update_of_ne (fun (e : x = s) => hsno (e ▸ hx)), hso x hx]
    have e2 : ipDen (Function.update ν s j) (sumL is) = ((is.map σ).sum : ℕ) := by
      rw [ipDen_sumL]; push_cast
      rw [List.map_map]; congr 1; refine List.map_congr_left fun x hx => ?_
      simp only [Function.comp]
      rw [Function.update_of_ne (fun (e : x = s) => hsni (e ▸ hx)), hsi x hx]
    simp only [bodyOf, List.map_nil, List.prod_nil, List.map_cons, List.prod_cons, one_mul,
      mul_one, phV, ipDen_mul, ipDen_V, ipDen_sub, Function.update_self, e1, e2]
    rw [prod_fou, prod_conj_fou, conj_om]
    have hz : om ((dt dim).den ρ.dv) ≠ 0 := om_ne_zero _ hD
    have hr : rt ((dt dim).den ρ.dv) ≠ 0 := rt_ne _ hD
    simp only [List.length_map, ← zpow_natCast]
    rw [show ((j : ℤ) * (((os.map σ).sum : ℕ) - ((is.map σ).sum : ℕ))) =
      (((j * (os.map σ).sum : ℕ) : ℤ) + -(((j * (is.map σ).sum : ℕ) : ℤ))) by push_cast; ring,
      zpow_add₀ hz]
    simp only [zpow_neg, zpow_natCast, pow_add, mul_inv]
    ring
  | F =>
    rcases os with _ | ⟨o, _ | _⟩ <;> rcases is with _ | ⟨i, _ | _⟩ <;>
      simp only [trNode, reduceCtorEq, Option.some.injEq] at h
    subst h
    obtain ⟨ho, hl'⟩ := legs_cons hl
    obtain ⟨hi, -⟩ := legs_cons hl'
    rw [exDen_cons, exDen_nil, add_zero, den_zero, rtD_den ρ _ hpos]
    simp only [val, List.map_cons, List.map_nil, fou, hdim, bodyOf, List.prod_nil, phV,
      List.prod_cons, mul_one, one_mul, ipDen_mul, ipDen_V, ho.2, hi.2, zpow_neg, zpow_one]
    rw [← zpow_natCast]; push_cast; ring
  | T =>
    rcases os with _ | ⟨o, _ | _⟩ <;> rcases is with _ | ⟨i, _ | _⟩ <;>
      simp only [trNode, reduceCtorEq, Option.some.injEq] at h
    subst h
    obtain ⟨ho, hl'⟩ := legs_cons hl
    obtain ⟨hi, -⟩ := legs_cons hl'
    simp only [exDen_cons, exDen_nil, add_zero, den_zero, ofQ_den, bodyOf, List.map_cons,
      List.map_nil, List.prod_cons, List.prod_nil, mul_one, delV_zero _ _ _ _ _ ho,
      delV_zero _ _ _ _ _ hi, delV_lt _ _ _ _ _ _ ho hi, val, tT]
    by_cases h1 : σ o = 0 <;> by_cases h2 : σ o = σ i <;> by_cases h3 : σ i = 0 <;>
      simp [h1, h2, h3] <;> omega
  | Ti =>
    rcases os with _ | ⟨o, _ | _⟩ <;> rcases is with _ | ⟨i, _ | _⟩ <;>
      simp only [trNode, reduceCtorEq, Option.some.injEq] at h
    subst h
    obtain ⟨ho, hl'⟩ := legs_cons hl
    obtain ⟨hi, -⟩ := legs_cons hl'
    simp only [exDen_cons, exDen_nil, add_zero, den_zero, ofQ_den, bodyOf, List.map_cons,
      List.map_nil, List.prod_cons, List.prod_nil, mul_one, delV_zero _ _ _ _ _ ho,
      delV_zero _ _ _ _ _ hi, delV_lt _ _ _ _ _ _ ho hi, val, tTi]
    by_cases h1 : σ o = 0 <;> by_cases h2 : σ o = σ i <;> by_cases h3 : σ i = 0 <;>
      simp [h1, h2, h3] <;> first | omega | (exfalso; omega) | simp_all
  | W =>
    rcases is with _ | ⟨i, _ | _⟩ <;> simp only [trNode, reduceCtorEq, Option.some.injEq] at h
    subst h
    have hi := hl i (List.mem_append_right _ List.mem_cons_self)
    have ho : ∀ o ∈ os, σ o < (dt dim).den ρ.dv ∧ ν o = σ o :=
      fun o h' => hl o (List.mem_append_left _ h')
    rw [exDen_append, wS_den ρ ν _ σ i [] os hi ho, exDen_cons, exDen_nil, den_zero, ofQ_den]
    simp only [val, List.map_cons, List.map_nil, bodyOf, List.prod_cons, List.prod_nil, mul_one,
      wZeros_den ρ ν _ σ os ho, delV_zero _ _ _ _ _ hi, hdim, tW_sF _ _ _ hi.1, List.length_map]
    push_cast
    split_ifs <;> ring
  | I =>
    rcases os with _ | ⟨o, _ | _⟩ <;> rcases is with _ | ⟨i, _ | _⟩ <;>
      simp only [trNode, reduceCtorEq, Option.some.injEq] at h
    subst h
    obtain ⟨ho, hl'⟩ := legs_cons hl
    obtain ⟨hi, -⟩ := legs_cons hl'
    simp only [exDen_cons, exDen_nil, add_zero, den_zero, ofQ_den, bodyOf, List.map_cons,
      List.map_nil, List.prod_cons, List.prod_nil, mul_one, delV_lt _ _ _ _ _ _ ho hi, val, tI,
      hdim]
    by_cases h2 : σ o = σ i <;> simp [h2, ho.1, hi.1]
  | _ => simp [trNode] at h

/-! ## Networks -/

/-- The product of two expressions. -/
def exMul (a b : List STm) : List STm := a.flatMap fun s => b.map fun t => STm.mul s t

/-- Every pair of terms multiplies without capture. -/
def exMulOK (a b : List STm) : Bool := a.all fun s => b.all fun t => mulOK s t

theorem exDen_mul (ρ : Val) (hρ : ρ.OK) (a b : List STm) (h : exMulOK a b = true) (ν : ℕ → ℤ) :
    exDen ρ ν (exMul a b) = exDen ρ ν a * exDen ρ ν b := by
  induction a with
  | nil => simp [exMul, exDen]
  | cons s a ih =>
    simp only [exMulOK, List.all_cons, Bool.and_eq_true] at h
    simp only [exMul, List.flatMap_cons] at ih ⊢
    rw [exDen_append, ih h.2, exDen_cons, add_mul]
    congr 1
    have hb := h.1
    clear ih h
    induction b with
    | nil => simp [exDen]
    | cons t b ih =>
      simp only [List.all_cons, Bool.and_eq_true] at hb
      rw [List.map_cons, exDen_cons, exDen_cons, ih hb.2, STm.den_mul ρ hρ s t hb.1 ν]; ring

/-- A label's dimension monomial. -/
def labDM (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (l : ℕ) : DMon := dt ((labs.lookup l).getD 0)

/-- The sums over the labels `ls`. -/
def labSums (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (ls : List ℕ) : List (ℕ × DMon) :=
  ls.map fun l => (l, labDM dt labs l)

/-- Every term summed over `S` first. -/
def exSums (S : List (ℕ × DMon)) (e : List STm) : List STm :=
  e.map fun t => { t with sums := S ++ t.sums }

theorem exDen_sums (ρ : Val) (S : List (ℕ × DMon)) (e : List STm) (ν : ℕ → ℤ) :
    exDen ρ ν (exSums S e) = sumsV ρ.dv S (fun ν' => exDen ρ ν' e) ν := by
  have h1 : ∀ t : STm, STm.den ρ ν { t with sums := S ++ t.sums } =
      sumsV ρ.dv S (fun ν' => t.den ρ ν') ν := by
    intro t
    unfold STm.den
    rw [show body ρ.dv { t with sums := S ++ t.sums } = body ρ.dv t from rfl, sumsV_append,
      ← sumsV_mul_left]
  unfold exDen exSums
  rw [List.map_map]
  simp only [Function.comp_def, h1]
  have := sumsV_lsum ρ.dv S (e.map fun t => fun ν' => t.den ρ ν') ν
  simp only [List.map_map, Function.comp_def] at this
  rw [← this]

theorem sumOver_sumsV (ρ : Val) (D : ℕ → ℕ) (dm : ℕ → DMon) (ls : List ℕ)
    (hD : ∀ l ∈ ls, D l = (dm l).den ρ.dv) (f : (ℕ → ℕ) → ℂ) (g : (ℕ → ℤ) → ℂ) (σ : ℕ → ℕ)
    (hfg : ∀ τ : ℕ → ℕ, (∀ x ∉ ls, τ x = σ x) → (∀ x ∈ ls, τ x < D x) →
      f τ = g (fun x => (τ x : ℤ))) :
    sumOver D ls f σ = sumsV ρ.dv (ls.map fun l => (l, dm l)) g (fun x => (σ x : ℤ)) := by
  induction ls generalizing σ with
  | nil => exact hfg σ (fun _ _ => rfl) (by simp)
  | cons l ls ih =>
    simp only [sumOver, List.map_cons, sumsV]
    rw [← hD l List.mem_cons_self]
    refine Finset.sum_congr rfl fun j hj => ?_
    have hcast : Function.update (fun x => (σ x : ℤ)) l (j : ℤ) =
        fun x => ((Function.update σ l j x : ℕ) : ℤ) := by
      funext x; by_cases hx : x = l
      · subst hx; simp
      · simp [Function.update_of_ne hx]
    rw [hcast]
    refine ih (fun x hx => hD x (List.mem_cons_of_mem _ hx)) _ fun τ hτ hr => hfg τ ?_ ?_
    · intro x hx
      rw [hτ x (fun h' => hx (List.mem_cons_of_mem _ h'))]
      exact Function.update_of_ne (fun (h' : x = l) => hx (h' ▸ List.mem_cons_self)) _ _
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx'
      · by_cases hm : x ∈ ls
        · exact hr x hm
        · rw [hτ x hm, Function.update_self]; exact Finset.mem_range.1 hj
      · exact hr x hx'

/-- A node's tables are usable: positive constant, stored phases below the floors. -/
def nodeChk (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (n : Node) : Bool :=
  (dt n.dim).c != 0 && (pt n.ph).all fun p => decide (p.1 < (dt n.dim).den (flv fl))

/-- The network's labels fit the translation. -/
def netChk (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (N : Net) : Bool :=
  (N.nodes.all fun n => nodeChk fl dt pt rl n && IX.contains n.dim && (n.outs ++ n.ins).all fun l =>
    decide (l < OFF) && (N.bnd.contains l || N.int.contains l) && labDM dt labs l == dt n.dim &&
      IX.contains ((labs.lookup l).getD 0)) &&
  N.int.all fun l => IX.contains ((labs.lookup l).getD 0)

/-- The internal labels `n` reads first, after the labels `pre`. -/
def firstAt (int pre : List ℕ) (n : Node) : List ℕ :=
  int.filter fun l => (n.outs ++ n.ins).contains l && !pre.contains l

/-- The internal labels `ns` reads, after the labels `pre`. -/
def firstOf (int pre : List ℕ) (ns : List Node) : List ℕ :=
  int.filter fun l => (labelsOf ns).contains l && !pre.contains l

/-- The nodes' product, each internal label summed, and the terms normalised, at the first node
reading it; node `i`'s bound variable `OFF + i`. -/
def trNodesB (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (int : List ℕ) : List ℕ → ℕ → List Node → Option (List STm)
  | _, _, [] => some [⟨CMon.ofQ 1, [], [], []⟩]
  | pre, i, n :: ns =>
    match trNode dt pt rl (OFF + i) n,
      trNodesB fl dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns with
    | some a, some b =>
      if exMulOK a b then
        some (nfEx fl (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)))
      else none
    | _, _ => none

theorem firstOf_perm (int pre : List ℕ) (n : Node) (ns : List Node) :
    (firstOf int pre (n :: ns)).Perm
      (firstAt int pre n ++ firstOf int (pre ++ (n.outs ++ n.ins)) ns) := by
  unfold firstOf firstAt
  have h := List.filter_append_perm (fun l => (n.outs ++ n.ins).contains l && !pre.contains l)
    (int.filter fun l => (labelsOf (n :: ns)).contains l && !pre.contains l)
  rw [List.filter_filter, List.filter_filter] at h
  have hc : ∀ l, (labelsOf (n :: ns)).contains l =
      ((n.outs ++ n.ins).contains l || (labelsOf ns).contains l) := fun l => by
    simp only [labelsOf, List.flatMap_cons, Node.labels]; exact List.contains_append
  refine h.symm.trans (List.Perm.append (List.Perm.of_eq (List.filter_congr fun l _ => ?_))
    (List.Perm.of_eq (List.filter_congr fun l _ => ?_)))
  · rw [hc l]
    generalize (n.outs ++ n.ins).contains l = A
    generalize pre.contains l = B
    generalize (labelsOf ns).contains l = C
    cases A <;> cases B <;> cases C <;> rfl
  · have h2 : (pre ++ (n.outs ++ n.ins)).contains l =
        (pre.contains l || (n.outs ++ n.ins).contains l) := List.contains_append
    rw [hc l, h2]
    generalize (n.outs ++ n.ins).contains l = A
    generalize pre.contains l = B
    generalize (labelsOf ns).contains l = C
    cases A <;> cases B <;> cases C <;> rfl

theorem trNodesB_ok (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ)
    (hD : ∀ l ∈ int, labDim E l = (labDM dt labs l).den ρ.dv) (ns : List Node)
    (hn : ∀ n ∈ ns, NodeOK E ρ dt pt rl n ∧ ∀ l ∈ n.outs ++ n.ins, l < OFF ∧
      labDim E l = E.dimv n.dim) (pre : List ℕ) (i : ℕ) (e : List STm)
    (h : trNodesB fl dt pt rl labs int pre i ns = some e) (σ : ℕ → ℕ)
    (hσ : ∀ l ∈ labelsOf ns, l ∉ firstOf int pre ns → σ l < labDim E l) :
    sumOver (labDim E) (firstOf int pre ns) (prodVal E ns) σ = exDen ρ (fun x => (σ x : ℤ)) e := by
  induction ns generalizing pre i e σ with
  | nil =>
    simp only [trNodesB, Option.some.injEq] at h
    subst h
    simp [firstOf, labelsOf, sumOver, prodVal, exDen_cons, exDen_nil, den_zero, ofQ_den, bodyOf]
  | cons n ns ih =>
    simp only [trNodesB] at h
    split at h
    · next a b ha hb =>
      split_ifs at h with hok
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨hnode, hlt⟩ := hn n List.mem_cons_self
      have hperm := firstOf_perm int pre n ns
      have hmem : ∀ l, l ∈ firstOf int pre (n :: ns) →
          l ∈ firstAt int pre n ∨ l ∈ firstOf int (pre ++ (n.outs ++ n.ins)) ns := by
        intro l hl; rw [← List.mem_append]; exact hperm.subset hl
      have hS : ∀ x ∈ n.outs ++ n.ins, x ∉ firstOf int (pre ++ (n.outs ++ n.ins)) ns := by
        intro x hx hx'
        have h1 := (List.mem_filter.1 hx').2
        have h2 : (pre ++ (n.outs ++ n.ins)).contains x = true := by simp [hx]
        rw [h2] at h1; simp at h1
      rw [sumOver_perm _ hperm, sumOver_append, nfEx_den ρ hρ fl hfl, exDen_sums]
      unfold labSums
      refine sumOver_sumsV ρ _ _ _ (fun l hl => hD l (List.mem_filter.1 hl).1) _ _ σ
        fun τ hτ hr => ?_
      have hlab : ∀ l ∈ labelsOf (n :: ns), l ∉ firstOf int pre (n :: ns) →
          l ∉ firstAt int pre n → τ l < labDim E l := by
        intro l hl hn' hF
        rw [hτ l hF]; exact hσ l hl hn'
      have hprod : ∀ τ', prodVal E (n :: ns) τ' = val E τ' n * prodVal E ns τ' := fun τ' => by
        simp [prodVal]
      rw [sumOver_congr _ _ _ (fun τ' => val E τ' n * prodVal E ns τ') τ (fun τ' _ _ => hprod τ'),
        sumOver_const_mul _ _ (n.outs ++ n.ins) _ _ (val_reads E n) hS τ]
      rw [ih (fun m hm => hn m (List.mem_cons_of_mem _ hm)) _ _ _ hb τ (fun l hl hl' => by
        by_cases hF : l ∈ firstAt int pre n
        · exact hr l hF
        · refine hlab l (by simp [labelsOf] at hl ⊢; tauto) (fun h' => ?_) hF
          rcases hmem l h' with h1 | h1
          · exact hF h1
          · exact hl' h1)]
      rw [exDen_mul ρ hρ a b hok, ← val_tr E ρ hρ dt pt rl (OFF + i) n a hnode τ _ (fun h' => by
        have := (hlt _ h').1; omega) (fun l hl => ⟨?_, rfl⟩) ha]
      rw [← (hlt l hl).2]
      by_cases hF : l ∈ firstAt int pre n
      · exact hr l hF
      · refine hlab l (by simp [labelsOf]; tauto) (fun h' => ?_) hF
        rcases hmem l h' with h1 | h1
        · exact hF h1
        · exact hS l hl h1
    · simp at h

/-- The network's entry over its boundary labels. -/
def trNet (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (N : Net) : Option (List STm) :=
  (trNodesB fl dt pt rl labs N.int [] 0 N.nodes).map
    (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)))

theorem eval_tr (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N : Net) (e : List STm) (hc : netChk fl dt pt rl labs IX N = true)
    (ht : trNet fl dt pt rl labs N = some e) (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = exDen ρ (fun x => (σ x : ℤ)) e := by
  unfold trNet at ht
  cases h0 : trNodesB fl dt pt rl labs N.int [] 0 N.nodes with
  | none => rw [h0] at ht; simp at ht
  | some e0 =>
    rw [h0] at ht
    simp only [Option.map_some, Option.some.injEq] at ht
    subst ht
    rw [exDen_sums]
    have hlab : ∀ l, (labs.lookup l).getD 0 ∈ IX → labDim E l = (labDM dt labs l).den ρ.dv :=
      fun l hl => by simp only [labDim, labDM, hL]; exact hE _ hl
    simp only [netChk, List.all_eq_true, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq,
      beq_iff_eq, List.contains_iff_mem] at hc
    obtain ⟨hc, hci⟩ := hc
    set rest := N.int.filter fun l => !(labelsOf N.nodes).contains l
    have hfirst : firstOf N.int [] N.nodes = N.int.filter fun l => (labelsOf N.nodes).contains l := by
      unfold firstOf; exact List.filter_congr fun l _ => by simp
    have hperm : N.int.Perm (rest ++ firstOf N.int [] N.nodes) := by
      rw [hfirst]
      exact (List.filter_append_perm _ N.int).symm.trans List.perm_append_comm
    unfold eval labSums
    rw [sumOver_perm _ hperm, sumOver_append]
    refine sumOver_sumsV ρ _ _ _ (fun l hl => hlab l (hci l (List.mem_filter.1 hl).1)) _ _ σ
      fun τ hτ hr => ?_
    have hn : ∀ n ∈ N.nodes, NodeOK E ρ dt pt rl n ∧ ∀ l ∈ n.outs ++ n.ins, l < OFF ∧
        labDim E l = E.dimv n.dim := by
      intro n hn
      obtain ⟨⟨hnc, hnix⟩, hlegs⟩ := hc n hn
      simp only [nodeChk, Bool.and_eq_true, bne_iff_ne, ne_eq, List.all_eq_true,
        decide_eq_true_eq] at hnc
      refine ⟨⟨hE _ hnix, Nat.pos_of_ne_zero hnc.1, hP _, fun p hp => ?_⟩, fun l hl => ?_⟩
      · exact lt_of_lt_of_le (hnc.2 p hp) (DMon.den_mono _ _ hfl _)
      · obtain ⟨⟨⟨hlt, -⟩, hd⟩, hix⟩ := hlegs l hl
        exact ⟨hlt, by rw [hlab l hix, hd, hE _ hnix]⟩
    refine trNodesB_ok E ρ hρ fl hfl dt pt rl labs N.int (fun l hl => hlab l (hci l hl)) N.nodes hn
      [] 0 e0 h0 τ fun l hl hl' => ?_
    obtain ⟨n, hnm, hln⟩ := List.mem_flatMap.1 hl
    obtain ⟨⟨⟨-, hbi⟩, -⟩, -⟩ := (hc n hnm).2 l hln
    have hni : l ∉ N.int := fun hi => hl' (by
      rw [hfirst]; exact List.mem_filter.2 ⟨hi, by simpa using hl⟩)
    have hnr : l ∉ rest := fun h' => hni (List.mem_filter.1 h').1
    rcases hbi with hb | hi
    · rw [hτ l hnr]; exact hσ l hb
    · exact absurd hi hni

/-! ## The equality -/

/-- Every term negated. -/
def exNeg (e : List STm) : List STm := e.map fun t => withQ t (-t.c.q)

theorem exDen_neg (ρ : Val) (ν : ℕ → ℤ) (e : List STm) : exDen ρ ν (exNeg e) = -exDen ρ ν e := by
  induction e with
  | nil => simp [exNeg, exDen]
  | cons t e ih =>
    simp only [exNeg, List.map_cons] at ih ⊢
    rw [exDen_cons, exDen_cons, ih, den_withQ, den_self ρ ν t]; push_cast; ring

/-- Every term a constant. -/
def exConst (e : List STm) : Bool := e.all fun t => t.sums.isEmpty && t.dels.isEmpty && t.phs.isEmpty

theorem exDen_const (ρ : Val) (e : List STm) (h : exConst e = true) (ν ν' : ℕ → ℤ) :
    exDen ρ ν e = exDen ρ ν' e := by
  induction e with
  | nil => rfl
  | cons t e ih =>
    simp only [exConst, List.all_cons, Bool.and_eq_true, List.isEmpty_iff] at h ih
    rw [exDen_cons, exDen_cons, ih h.2]
    obtain ⟨c, s, d, p⟩ := t
    obtain ⟨⟨h1, h2⟩, h3⟩ := h.1
    simp only at h1 h2 h3
    subst h1 h2 h3
    rfl

/-- The product of the nodes' entries, node `i`'s bound variable `OFF + i`, unnormalised. -/
def trNodesF (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) : ℕ → List Node → Option (List STm)
  | _, [] => some [⟨CMon.ofQ 1, [], [], []⟩]
  | i, n :: ns =>
    match trNode dt pt rl (OFF + i) n, trNodesF dt pt rl (i + 1) ns with
    | some a, some b => if exMulOK a b then some (exMul a b) else none
    | _, _ => none

theorem trNodesF_ok (E : Env) (ρ : Val) (hρ : ρ.OK) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (σ : ℕ → ℕ) (ν : ℕ → ℤ) (ns : List Node) (i : ℕ) (e : List STm)
    (hn : ∀ n ∈ ns, NodeOK E ρ dt pt rl n ∧ (∀ l ∈ n.outs ++ n.ins, l < OFF) ∧
      LegsAt (E.dimv n.dim) σ ν (n.outs ++ n.ins))
    (h : trNodesF dt pt rl i ns = some e) : prodVal E ns σ = exDen ρ ν e := by
  induction ns generalizing i e with
  | nil =>
    simp only [trNodesF, Option.some.injEq] at h
    subst h
    simp [prodVal, exDen_cons, exDen_nil, den_zero, ofQ_den, bodyOf]
  | cons n ns ih =>
    simp only [trNodesF] at h
    split at h
    · next a b ha hb =>
      split_ifs at h with hok
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨hnode, hlt, hlegs⟩ := hn n List.mem_cons_self
      rw [exDen_mul ρ hρ a b hok, ← ih (i + 1) b (fun m hm => hn m (List.mem_cons_of_mem _ hm)) hb,
        ← val_tr E ρ hρ dt pt rl (OFF + i) n a hnode σ ν (fun h' => by
          have := hlt _ h'; omega) hlegs ha]
      simp [prodVal]
    · simp at h

/-- The network's entry over its boundary labels, every label summed last. -/
def trNetF (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (N : Net) :
    Option (List STm) :=
  (trNodesF dt pt rl 0 N.nodes).map (exSums (labSums dt labs N.int))

theorem eval_trF (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N : Net) (e : List STm) (hc : netChk fl dt pt rl labs IX N = true)
    (ht : trNetF dt pt rl labs N = some e) (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = exDen ρ (fun x => (σ x : ℤ)) e := by
  unfold trNetF at ht
  cases h0 : trNodesF dt pt rl 0 N.nodes with
  | none => rw [h0] at ht; simp at ht
  | some e0 =>
    rw [h0] at ht
    simp only [Option.map_some, Option.some.injEq] at ht
    subst ht
    rw [exDen_sums]
    have hlab : ∀ l, (labs.lookup l).getD 0 ∈ IX → labDim E l = (labDM dt labs l).den ρ.dv :=
      fun l hl => by simp only [labDim, labDM, hL]; exact hE _ hl
    simp only [netChk, List.all_eq_true, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq,
      beq_iff_eq, List.contains_iff_mem] at hc
    obtain ⟨hc, hci⟩ := hc
    unfold eval labSums
    refine sumOver_sumsV ρ _ _ _ (fun l hl => hlab l (hci l hl)) _ _ σ fun τ hτ hr => ?_
    refine trNodesF_ok E ρ hρ dt pt rl τ _ N.nodes 0 e0 (fun n hn => ?_) h0
    obtain ⟨⟨hnc, hnix⟩, hlegs⟩ := hc n hn
    simp only [nodeChk, Bool.and_eq_true, bne_iff_ne, ne_eq, List.all_eq_true,
      decide_eq_true_eq] at hnc
    refine ⟨⟨hE _ hnix, Nat.pos_of_ne_zero hnc.1, hP _, fun p hp => ?_⟩,
      fun l hl => (hlegs l hl).1.1.1, fun l hl => ⟨?_, rfl⟩⟩
    · exact lt_of_lt_of_le (hnc.2 p hp) (DMon.den_mono _ _ hfl _)
    · obtain ⟨⟨⟨-, hbi⟩, hd⟩, hix⟩ := hlegs l hl
      have hdim : labDim E l = E.dimv n.dim := by rw [hlab l hix, hd, hE _ hnix]
      rw [← hdim]
      rcases hbi with hb | hi
      · by_cases hm : l ∈ N.int
        · exact hr l hm
        · rw [hτ l hm]; exact hσ l hb
      · exact hr l hi

/-- Both networks translate by `tr` and the scaled difference normalises to nothing. -/
def checkWith (tr : Net → Option (List STm)) (fl : List ℕ) (N0 N1 : Net) (S0 S1 : List STm) :
    Bool :=
  match tr N0, tr N1 with
  | some e0, some e1 => exMulOK S0 e0 && exMulOK (exNeg S1) e1 &&
      (nfEx fl (exMul S0 e0 ++ exMul (exNeg S1) e1)).isEmpty
  | _, _ => false

/-- The check `equal_sym` reads: the labels fit, and the normal form of the scaled difference,
labels summed where first read or else summed last, is empty. -/
def symCheck (fl : List ℕ) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (N0 N1 : Net) (S0 S1 : List STm) : Bool :=
  N0.bnd.isPerm N1.bnd && netChk fl dt pt rl labs IX N0 && netChk fl dt pt rl labs IX N1 && exConst S0 &&
    exConst S1 && (checkWith (trNet fl dt pt rl labs) fl N0 N1 S0 S1 ||
      checkWith (trNetF dt pt rl labs) fl N0 N1 S0 S1)

theorem equal_of_check (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (tr : Net → Option (List STm)) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (htr0 : ∀ e, tr N0 = some e → ∀ σ, InRange E N0.bnd σ → eval E N0 σ = exDen ρ (fun x => (σ x : ℤ)) e)
    (htr1 : ∀ e, tr N1 = some e → ∀ σ, InRange E N1.bnd σ → eval E N1 σ = exDen ρ (fun x => (σ x : ℤ)) e)
    (hp : N0.bnd.Perm N1.bnd) (hk0 : exConst S0 = true) (hk1 : exConst S1 = true)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : checkWith tr fl N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold checkWith at h
  split at h
  · next e0 e1 ht0 ht1 =>
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    obtain ⟨⟨hm0, hm1⟩, hnf⟩ := h
    refine ⟨hp, fun σ hσ => ?_⟩
    set ν : ℕ → ℤ := fun x => (σ x : ℤ)
    have hz := nfEx_zero ρ hρ fl hfl _ hnf ν
    rw [exDen_append, exDen_mul ρ hρ _ _ hm0, exDen_mul ρ hρ _ _ hm1, exDen_neg,
      ← htr0 e0 ht0 σ hσ, ← htr1 e1 ht1 σ (hσ.perm hp),
      exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
      ← hs1] at hz
    linear_combination hz
  · simp at h

theorem equal_sym (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1)
    (h : symCheck fl dt pt rl labs IX N0 N1 S0 S1 = true) : Equal E s0 N0 s1 N1 := by
  unfold symCheck at h
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hperm, hc0⟩, hc1⟩, hk0⟩, hk1⟩, hrest⟩ := h
  have hp : N0.bnd.Perm N1.bnd := List.isPerm_iff.1 hperm
  rcases hrest with hb | hf
  · exact equal_of_check E ρ hρ fl hfl _ N0 N1 S0 S1 s0 s1
      (fun e ht σ hσ => eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e hc0 ht σ hσ)
      (fun e ht σ hσ => eval_tr E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e hc1 ht σ hσ)
      hp hk0 hk1 hs0 hs1 hb
  · exact equal_of_check E ρ hρ fl hfl _ N0 N1 S0 S1 s0 s1
      (fun e ht σ hσ => eval_trF E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e hc0 ht σ hσ)
      (fun e ht σ hσ => eval_trF E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e hc1 ht σ hσ)
      hp hk0 hk1 hs0 hs1 hf

/-! ## Phases and atoms -/

/-- `ex (q + Σ_j ks_j g_j)` over the atoms `ex g_j`. -/
def phC (q : ℚ) (ks : List ℤ) : CMon := ⟨(CMon.rnorm 1 q).1, (CMon.rnorm 1 q).2, [], [], ztrim ks, []⟩

theorem ex_zpow' (g : ℝ) (z : ℤ) : ex g ^ z = ex (z * g) := by
  unfold ex; rw [← Complex.exp_int_mul]; congr 1; push_cast; ring

/-- `Σ_j ks_j g_j`. -/
noncomputable def lin (ks : List ℤ) (gs : List ℝ) : ℝ :=
  (List.zipWith (fun (z : ℤ) (g : ℝ) => (z : ℝ) * g) ks gs).sum

theorem zpw_ex (gs : List ℝ) (ks : List ℤ) (k : ℕ) :
    zpw (fun j => ex (gs.getD j 0)) k ks = ex (lin ks (gs.drop k)) := by
  induction ks generalizing k with
  | nil => simp [zpw, ex_zero, lin]
  | cons z ks ih =>
    simp only [zpw, ih]
    cases h : gs.drop k with
    | nil =>
      have : gs.length ≤ k := by simpa using h
      have h2 : gs.drop (k + 1) = [] := by simp; omega
      rw [h2, List.getD_eq_default _ _ (by omega)]
      simp [ex_zero, lin]
    | cons g rest =>
      have hg : gs.getD k 0 = g := by
        rw [List.getD_eq_getElem?_getD, ← List.head?_drop, h]; rfl
      have h2 : gs.drop (k + 1) = rest := by
        rw [← List.drop_drop, h]; rfl
      rw [hg, h2, ex_zpow', ← ex_add]
      simp [lin]

theorem phC_den (ρ : Val) (gs : List ℝ) (hg : ∀ j, ρ.atom j = ex (gs.getD j 0)) (q : ℚ)
    (ks : List ℤ) (θ : ℝ) (hθ : θ = q + lin ks gs) :
    (phC q ks).den ρ = ex θ := by
  have hr := rnorm_den 1 q
  have hz : zpw ρ.atom 0 (ztrim ks) = zpw (fun j => ex (gs.getD j 0)) 0 ks := by
    rw [zpw_ztrim]; congr 1; funext j; exact hg j
  simp only [phC, CMon.den, zpw, List.map_nil, List.prod_nil, mul_one, hz, zpw_ex, List.drop_zero]
  rw [hr, hθ, ex_add]; push_cast; ring

theorem phC_den_int (ρ : Val) (gs : List ℝ) (hg : ∀ j, ρ.atom j = ex (gs.getD j 0)) (q : ℚ)
    (ks : List ℤ) (θ : ℝ) (hθ : ∃ z : ℤ, θ = q + lin ks gs + z) :
    (phC q ks).den ρ = ex θ := by
  obtain ⟨z, hz⟩ := hθ
  rw [phC_den ρ gs hg q ks (q + lin ks gs) rfl, hz, ex_int]

/-- A phase rule `θ k = p(k) / D`, zero at index `0`, as root-of-unity factors. -/
theorem rule_ex (ρ : Val) (D : DMon) (cs : List ℤ) (θ : ℕ → ℝ)
    (hθ : ∀ k : ℕ, θ k = (rEv (k : ℤ) cs : ℝ) / (D.den ρ.dv : ℝ)) (h0 : rEv 0 cs = 0) (j : ℕ) :
    ex (if j = 0 then 0 else θ j) = rV ρ (D, cs) j := by
  unfold rV
  split_ifs with hj
  · subst hj; simp [h0, ex_zero]
  · rw [hθ, om_zpow_ex]; congr 1; push_cast; ring

/-- `D ^ k`. -/
def dP (D : DMon) (k : ℤ) : CMon := CMon.rtD D (2 * k)

theorem dP_den (ρ : Val) (D : DMon) (hc : 0 < D.c) (k : ℤ) :
    (dP D k).den ρ = ((D.den ρ.dv : ℕ) : ℂ) ^ k := by
  rw [dP, rtD_den ρ D hc, zpow_mul, zpow_ofNat, rt_sq']

end ArchytasZX.Sym
