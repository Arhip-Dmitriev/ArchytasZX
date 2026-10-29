/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LCanon

/-!
# The meaning of .azx programs

A program value is a complex function of the index variables (`Fn`): output register `i` is
variable `i`, input register `i` of an operator is variable `IN + i`, and every other variable
is a parameter, a summation index or an atom. `Prog` is a program's structure over leaf
expressions; `Prog.sem` is its meaning as sums and products over ℂ, `Prog.sym` its translation
into one symbolic sum with atoms, and `Prog.sym_ok` proves the two agree. `certify` proves a
program equal to an expression when the normal form of their difference is empty.
-/

namespace ArchytasZX.Lang

open ArchytasZX ArchytasZX.Sym ArchytasZX.LSym Complex Finset

/-- Complex functions of the index variables. -/
abbrev Fn := (ℕ → ℤ) → ℂ

/-- The variable of input register `0`. -/
def IN : ℕ := 1000

/-- `ν` with the variables `b, b + 1, …` set to `js`. -/
def setL (ν : ℕ → ℤ) (b : ℕ) : List ℤ → ℕ → ℤ
  | [] => ν
  | j :: js => setL (Function.update ν b j) (b + 1) js

/-- The sum over every tuple below the levels `L`. -/
noncomputable def sumT : List ℕ → (List ℤ → ℂ) → ℂ
  | [], f => f []
  | N :: L, f => ∑ j ∈ range N, sumT L (fun js => f ((j : ℤ) :: js))

/-- Output and input registers moved up by `k`. -/
def shiftN (k v : ℕ) : ℕ := if v < 2 * IN then v + k else v

/-- A program's structure. -/
inductive Prog where
  /-- An expression: a state, a gate's kernel or a scalar. -/
  | leaf (e : List ATm)
  /-- `a` applied to `b` over registers of levels `L`: `Σ_y a(o, y) b(y, x)`. -/
  | app (L : List DMon) (a b : Prog)
  /-- `a ⊗ b`, `a` on the first `k` registers. -/
  | ten (k : ℕ) (a b : Prog)
  /-- The pointwise product. -/
  | mul (a b : Prog)
  /-- The sum. -/
  | add (a b : Prog)
  /-- The inner product `⟨a|b⟩` over registers of levels `L`. -/
  | inn (L : List DMon) (a b : Prog)
  /-- The complex conjugate. -/
  | cnj (a : Prog)
  /-- `step` applied `cnt` times to `init` over registers of levels `L`, iteration `i` reading
  `J = lo + i`; `inv` is the closed form after the iterations before `J`. -/
  | loop (L : List DMon) (J : ℕ) (lo : ℤ) (cnt : IPoly) (step init : Prog) (inv : List ATm)

/-- `k` iterations of `u` from `s`, iteration `i` reading `J = lo + i`. -/
noncomputable def loopSem (dv : ℕ → ℕ) (L : List DMon) (J : ℕ) (lo : ℤ) (u s : Fn) : ℕ → Fn
  | 0 => s
  | i + 1 => fun ν => sumT (L.map (·.den dv)) fun js =>
      u (Function.update (setL ν IN js) J (lo + i)) * loopSem dv L J lo u s i (setL ν 0 js)

/-- The meaning of a program. -/
noncomputable def Prog.sem (V : LV) : Prog → Fn
  | .leaf e => fun ν => axDen V ν e
  | .app L a b => fun ν => sumT (L.map (·.den V.ρ.dv)) fun js =>
      a.sem V (setL ν IN js) * b.sem V (setL ν 0 js)
  | .ten k a b => fun ν => a.sem V ν * b.sem V (ν ∘ shiftN k)
  | .mul a b => fun ν => a.sem V ν * b.sem V ν
  | .add a b => fun ν => a.sem V ν + b.sem V ν
  | .inn L a b => fun ν => sumT (L.map (·.den V.ρ.dv)) fun js =>
      starRingEnd ℂ (a.sem V (setL ν 0 js)) * b.sem V (setL ν 0 js)
  | .cnj a => fun ν => starRingEnd ℂ (a.sem V ν)
  | .loop L J lo cnt step init _ => fun ν =>
      loopSem V.ρ.dv L J lo (step.sem V) (init.sem V) (ipDen ν cnt).toNat ν

/-- The iteration counts of a program's loops. -/
def Prog.counts : Prog → List IPoly
  | .leaf _ => []
  | .app _ a b => a.counts ++ b.counts
  | .ten _ a b => a.counts ++ b.counts
  | .mul a b => a.counts ++ b.counts
  | .add a b => a.counts ++ b.counts
  | .inn _ a b => a.counts ++ b.counts
  | .cnj a => a.counts
  | .loop _ _ _ c s i _ => c :: (s.counts ++ i.counts)

/-! ## Tuples -/

theorem setL_apply (ν : ℕ → ℤ) (b : ℕ) (js : List ℤ) (v : ℕ) :
    setL ν b js v = if b ≤ v ∧ v < b + js.length then js.getD (v - b) 0 else ν v := by
  induction js generalizing ν b with
  | nil => simp [setL]
  | cons j js ih =>
    simp only [setL, ih, List.length_cons]
    by_cases h1 : v = b
    · subst h1; simp
    · rw [Function.update_of_ne h1]
      by_cases h2 : b + 1 ≤ v ∧ v < b + 1 + js.length
      · have e : v - b = (v - (b + 1)) + 1 := by omega
        rw [ite_eq_left_of_eq_true _ _ (eq_true h2),
          ite_eq_left_of_eq_true _ _ (eq_true (by omega)), e, List.getD_cons_succ]
      · rw [ite_eq_right_of_eq_false _ _ (eq_false h2),
          ite_eq_right_of_eq_false _ _ (eq_false (by omega))]

theorem sumT_congr (L : List ℕ) (f g : List ℤ → ℂ)
    (h : ∀ js : List ℤ, js.length = L.length → f js = g js) : sumT L f = sumT L g := by
  induction L generalizing f g with
  | nil => exact h [] rfl
  | cons N L ih =>
    simp only [sumT]
    exact Finset.sum_congr rfl fun j _ =>
      ih _ _ fun js hjs => h _ (by simp [hjs])

/-- The variables `T, T + 1, …` over the levels `L`. -/
def blk (T : ℕ) : List DMon → List (ℕ × DMon)
  | [] => []
  | D :: L => (T, D) :: blk (T + 1) L

theorem sumsV_blk (dv : ℕ → ℕ) (T : ℕ) (L : List DMon) (f : (ℕ → ℤ) → ℂ) (ν : ℕ → ℤ) :
    sumsV dv (blk T L) f ν = sumT (L.map (·.den dv)) fun js => f (setL ν T js) := by
  induction L generalizing T ν with
  | nil => rfl
  | cons D L ih => simp only [blk, sumsV, List.map_cons, sumT, ih, setL]

/-! ## Expressions -/

/-- An expression renamed. -/
def axRen (m : ℕ → ℕ) (e : List ATm) : List ATm := e.map (tRen m)

/-- `m` is injective on every term's variables. -/
def renOKA (m : ℕ → ℕ) (e : List ATm) : Bool := e.all fun t => injOn m (allV t)

theorem axDen_ren (V : LV) (m : ℕ → ℕ) (e : List ATm) (h : renOKA m e = true) (ν : ℕ → ℤ) :
    axDen V ν (axRen m e) = axDen V (ν ∘ m) e := by
  simp only [renOKA, List.all_eq_true] at h
  unfold axDen axRen
  rw [List.map_map]
  congr 1
  exact List.map_congr_left fun t ht => tRen_den V m t (h t ht) ν

/-- Every term summed over `S` first. -/
def axSums (S : List (ℕ × DMon)) (e : List ATm) : List ATm :=
  e.map fun t => { t with sums := S ++ t.sums }

theorem axDen_sums (V : LV) (S : List (ℕ × DMon)) (e : List ATm) (ν : ℕ → ℤ) :
    axDen V ν (axSums S e) = sumsV V.ρ.dv S (fun ν' => axDen V ν' e) ν := by
  have h1 : ∀ t : ATm, ATm.den V ν { t with sums := S ++ t.sums } =
      sumsV V.ρ.dv S (fun ν' => t.den V ν') ν := by
    intro t
    unfold ATm.den
    rw [show abody V { t with sums := S ++ t.sums } = abody V t from rfl, sumsV_append,
      ← sumsV_mul_left]
  unfold axDen axSums
  rw [List.map_map]
  simp only [Function.comp_def, h1]
  have := sumsV_lsum V.ρ.dv S (e.map fun t => fun ν' => t.den V ν') ν
  simp only [List.map_map, Function.comp_def] at this
  rw [← this]

/-- The greatest variable an expression names. -/
def maxVar (e : List ATm) : ℕ := (e.flatMap allV).foldr max 0

/-- The first variable of a block free in `a` and `b`. -/
def fresh (a b : List ATm) : ℕ := max (3 * IN) (max (maxVar a) (maxVar b) + 1)

/-- Bound variables renamed `B, B + 1, …` and then the atoms. -/
def bmap (B : ℕ) (bl al : List ℕ) (v : ℕ) : ℕ :=
  if bl.contains v then B + bl.idxOf v
  else if al.contains v then B + bl.length + al.idxOf v else v

/-- A term's bound variables and atoms renamed from `B`, when the renaming is injective. -/
def tApart (B : ℕ) (t : ATm) : ATm :=
  if injOn (bmap B (abvs t) ((ids t.ats).filter fun v => !(abvs t).contains v)) (allV t) &&
      wfA t.ats then
    tRen (bmap B (abvs t) ((ids t.ats).filter fun v => !(abvs t).contains v)) t
  else t

theorem tApart_ok (V : LV) (B : ℕ) (t : ATm) (ν : ℕ → ℤ) : (tApart B t).den V ν = t.den V ν := by
  unfold tApart
  split_ifs with hc
  · simp only [Bool.and_eq_true] at hc
    rw [tRen_den V _ t hc.1]
    refine den_reads_free V t hc.2 _ _ fun v hv => ?_
    simp only [freeV, List.mem_filter, Bool.and_eq_true, Bool.not_eq_true',
      List.contains_eq_mem, decide_eq_false_iff_not] at hv
    simp only [Function.comp_apply, bmap, List.contains_eq_mem, List.mem_filter,
      decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not]
    rw [if_neg hv.2.2, if_neg (fun h' => hv.2.1 h'.1)]
  · rfl

/-- Every term's bound variables and atoms renamed from `B`. -/
def apart (B : ℕ) (e : List ATm) : List ATm := e.map (tApart B)

theorem axDen_apart (V : LV) (B : ℕ) (e : List ATm) (ν : ℕ → ℤ) :
    axDen V ν (apart B e) = axDen V ν e := by
  unfold axDen apart
  rw [List.map_map]
  congr 1
  exact List.map_congr_left fun t _ => tApart_ok V B t ν

/-! ## Application -/

/-- Input register `i` renamed `T + i`, for `i < k`. -/
def inMap (T k v : ℕ) : ℕ := if IN ≤ v ∧ v < IN + k then T + (v - IN) else v

/-- Output register `i` renamed `T + i`, for `i < k`. -/
def outMap (T k v : ℕ) : ℕ := if v < k then T + v else v

/-- No variable of `e` in the block `T, …, T + k - 1`. -/
def avoids (T k : ℕ) (e : List ATm) : Bool := (axVs e).all fun v => !(decide (T ≤ v ∧ v < T + k))

/-- `Σ_y a(o, y) b(y, x)` as one expression, the intermediate registers summed over the block
from `fresh a b`. -/
def appSym (L : List DMon) (a b : List ATm) : Option (List ATm) :=
  if decide (L.length < IN) && decide (3 * IN ≤ fresh a b) &&
      avoids (fresh a b) L.length a && avoids (fresh a b) L.length b &&
      renOKA (inMap (fresh a b) L.length) a && renOKA (outMap (fresh a b) L.length) b &&
      axMulOK (axRen (inMap (fresh a b) L.length) a) (axRen (outMap (fresh a b) L.length) b) then
    some (axSums (blk (fresh a b) L)
      (axMul (axRen (inMap (fresh a b) L.length) a) (axRen (outMap (fresh a b) L.length) b)))
  else none

theorem appSym_ok (V : LV) (hρ : V.ρ.OK) (L : List DMon) (a b e : List ATm)
    (h : appSym L a b = some e) (ν : ℕ → ℤ) :
    axDen V ν e = sumT (L.map (·.den V.ρ.dv)) fun js =>
      axDen V (setL ν IN js) a * axDen V (setL ν 0 js) b := by
  unfold appSym at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨⟨⟨⟨⟨⟨hk, hT⟩, haA⟩, haB⟩, hrA⟩, hrB⟩, hm⟩ := hc
  generalize fresh a b = T at *
  rw [axDen_sums, sumsV_blk]
  refine sumT_congr _ _ _ fun js hjs => ?_
  simp only [List.length_map] at hjs
  rw [axDen_mul V hρ _ _ hm, axDen_ren V _ _ hrA, axDen_ren V _ _ hrB]
  simp only [avoids, List.all_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at haA haB
  congr 1
  · refine axDen_reads V a _ _ fun v hv => ?_
    have hv' := haA v hv
    simp only [Function.comp_apply, inMap, setL_apply, hjs]
    by_cases hc : IN ≤ v ∧ v < IN + L.length
    · have h1 : T ≤ T + (v - IN) ∧ T + (v - IN) < T + L.length := by omega
      simp only [hc, h1, and_self, ite_true]
      rw [Nat.add_sub_cancel_left]
    · simp only [hc, hv', ite_false]
  · refine axDen_reads V b _ _ fun v hv => ?_
    have hv' := haB v hv
    simp only [Function.comp_apply, outMap, setL_apply, hjs]
    by_cases hc : v < L.length
    · have h1 : T ≤ T + v ∧ T + v < T + L.length := by omega
      have h2 : 0 ≤ v ∧ v < 0 + L.length := by omega
      simp only [hc, h1, h2, and_self, ite_true]
      rw [Nat.add_sub_cancel_left, Nat.sub_zero]
    · have h2 : ¬(0 ≤ v ∧ v < 0 + L.length) := by omega
      simp only [hc, hv', h2, ite_false]

/-! ## Tensor products -/

/-- `a ⊗ b` as one expression, `a` on the first `k` registers. -/
def tenSym (k : ℕ) (a b : List ATm) : Option (List ATm) :=
  if renOKA (shiftN k) b && axMulOK a (axRen (shiftN k) b) then
    some (axMul a (axRen (shiftN k) b)) else none

theorem tenSym_ok (V : LV) (hρ : V.ρ.OK) (k : ℕ) (a b e : List ATm) (h : tenSym k a b = some e)
    (ν : ℕ → ℤ) : axDen V ν e = axDen V ν a * axDen V (ν ∘ shiftN k) b := by
  unfold tenSym at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  simp only [Bool.and_eq_true] at hc
  rw [axDen_mul V hρ _ _ hc.2, axDen_ren V _ _ hc.1]

/-! ## Conjugates -/

def cConj (c : CMon) : CMon :=
  ⟨(CMon.rnorm c.q (-c.r)).1, (CMon.rnorm c.q (-c.r)).2, c.sq, c.rc, c.am, c.gc⟩

theorem conj_ex (r : ℝ) : starRingEnd ℂ (ex r) = ex (-r) := by
  simp only [ex, ← Complex.exp_conj]
  congr 1
  simp only [map_mul, map_ofNat, Complex.conj_ofReal, Complex.conj_I]
  push_cast; ring

theorem conj_zpw (f : ℕ → ℂ) (hf : ∀ i, starRingEnd ℂ (f i) = f i) (k : ℕ) (l : List ℤ) :
    starRingEnd ℂ (zpw f k l) = zpw f k l := by
  induction l generalizing k with
  | nil => simp [zpw]
  | cons a l ih => simp only [zpw, map_mul, map_zpow₀, hf, ih]

theorem conj_cden (ρ : Val) (c : CMon) (h : c.am = []) :
    starRingEnd ℂ (c.den ρ) = (cConj c).den ρ := by
  have hr := rnorm_den c.q (-c.r)
  have hrt : ∀ x : ℕ, starRingEnd ℂ (rt x) = rt x := fun x => Complex.conj_ofReal _
  have e1 : starRingEnd ℂ (c.rc.map rt).prod = (c.rc.map rt).prod := by
    rw [map_list_prod]; simp [List.map_map, Function.comp_def, hrt]
  have e2 : starRingEnd ℂ (c.gc.map (gcdV ρ.dv)).prod = (c.gc.map (gcdV ρ.dv)).prod := by
    rw [map_list_prod, List.map_map]
    exact congrArg _ (List.map_congr_left fun g _ => by simp [gcdV])
  have e3 : starRingEnd ℂ (zpw (rtv ρ.dv) 0 c.sq) = zpw (rtv ρ.dv) 0 c.sq :=
    conj_zpw _ (fun i => hrt _) _ _
  simp only [CMon.den, cConj, h, map_mul, map_ratCast, conj_ex, e1, e2, e3]
  have hr' : ((CMon.rnorm c.q (-c.r)).1 : ℂ) * ex ((CMon.rnorm c.q (-c.r)).2 : ℝ) =
      (c.q : ℂ) * ex (-(c.r : ℝ)) := by rw [hr]; push_cast; rfl
  rw [← hr']
  simp [zpw]

/-- The conjugate term. -/
def tConj (t : ATm) : ATm :=
  ⟨cConj t.c, t.sums, t.ats, t.dels, t.phs.map fun f => (f.1, ipSmul (-1) f.2)⟩

/-- No term has coefficient atoms. -/
def conjOK (e : List ATm) : Bool := e.all fun t => t.c.am.isEmpty

theorem conj_sumsV (dv : ℕ → ℕ) (s : List (ℕ × DMon)) (f g : (ℕ → ℤ) → ℂ)
    (h : ∀ μ, starRingEnd ℂ (f μ) = g μ) (ν : ℕ → ℤ) :
    starRingEnd ℂ (sumsV dv s f ν) = sumsV dv s g ν := by
  induction s generalizing ν with
  | nil => exact h ν
  | cons p s ih => simp only [sumsV, map_sum, ih]

theorem conj_body (V : LV) (t : ATm) (μ : ℕ → ℤ) :
    starRingEnd ℂ (abody V t μ) = abody V (tConj t) μ := by
  unfold abody bodyOf
  simp only [tConj, map_mul, map_list_prod, List.map_map]
  congr 1
  · congr 1; refine List.map_congr_left fun f _ => ?_
    simp only [Function.comp_apply, delV]; split_ifs <;> simp
  · congr 1; refine List.map_congr_left fun f _ => ?_
    simp only [Function.comp_apply, phV, map_zpow₀, conj_om, ipDen_smul, inv_zpow', neg_one_mul]

theorem conj_den (V : LV) (t : ATm) (h : t.c.am = []) (ν : ℕ → ℤ) :
    starRingEnd ℂ (t.den V ν) = (tConj t).den V ν := by
  unfold ATm.den
  rw [map_mul, conj_cden V.ρ t.c h, conj_sumsV _ _ _ _ (conj_body V t)]
  rfl

theorem axDen_conj (V : LV) (e : List ATm) (h : conjOK e = true) (ν : ℕ → ℤ) :
    starRingEnd ℂ (axDen V ν e) = axDen V ν (e.map tConj) := by
  simp only [conjOK, List.all_eq_true, List.isEmpty_iff] at h
  unfold axDen
  rw [map_list_sum, List.map_map, List.map_map]
  congr 1
  exact List.map_congr_left fun t ht => conj_den V t (h t ht) ν

/-! ## Inner products -/

/-- Output register `i` renamed input register `i`, for `i < k`. -/
def toIn (k v : ℕ) : ℕ := if v < k then IN + v else v

/-- `⟨a|b⟩` as one expression: the conjugate of `a` read on input registers, applied to `b`. -/
def innSym (L : List DMon) (a b : List ATm) : Option (List ATm) :=
  if conjOK a && renOKA (toIn L.length) (a.map tConj) && avoids IN L.length a then
    appSym L (axRen (toIn L.length) (a.map tConj)) b
  else none

theorem innSym_ok (V : LV) (hρ : V.ρ.OK) (L : List DMon) (a b e : List ATm)
    (h : innSym L a b = some e) (ν : ℕ → ℤ) :
    axDen V ν e = sumT (L.map (·.den V.ρ.dv)) fun js =>
      starRingEnd ℂ (axDen V (setL ν 0 js) a) * axDen V (setL ν 0 js) b := by
  unfold innSym at h
  split_ifs at h with hc
  simp only [Bool.and_eq_true] at hc
  obtain ⟨⟨hc1, hc2⟩, hc3⟩ := hc
  rw [appSym_ok V hρ L _ b e h]
  refine sumT_congr _ _ _ fun js hjs => ?_
  simp only [List.length_map] at hjs
  rw [axDen_ren V _ _ hc2, axDen_conj V a hc1]
  congr 1
  have hv : ∀ v ∈ axVs (a.map tConj), (setL ν IN js ∘ toIn L.length) v = setL ν 0 js v := by
    intro v hv
    have hva : v ∈ axVs a := by
      simp only [axVs, List.mem_flatMap, List.mem_map] at hv ⊢
      obtain ⟨_, ⟨t, ht, rfl⟩, hv⟩ := hv
      exact ⟨t, ht, by simpa [rvs, fvs, tConj, facVars, ipSmul, ipVs] using hv⟩
    simp only [avoids, List.all_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at hc3
    have := hc3 v hva
    simp only [Function.comp_apply, toIn, setL_apply, hjs]
    by_cases hc : v < L.length
    · have h1 : IN ≤ IN + v ∧ IN + v < IN + L.length := by omega
      have h2 : 0 ≤ v ∧ v < 0 + L.length := by omega
      simp only [hc, h1, h2, and_self, ite_true]
      rw [Nat.add_sub_cancel_left, Nat.sub_zero]
    · have h2 : ¬(0 ≤ v ∧ v < 0 + L.length) := by omega
      simp only [hc, this, h2, ite_false]
  exact axDen_reads V _ _ _ hv

/-! ## Substituting a free variable -/

theorem aext_sub' (V : LV) (l : List Atom) (x : ℕ) (hx : x ∉ ids l) (v : IPoly)
    (hv : plainP l v = true) (μ : ℕ → ℤ) :
    aext V l (Function.update μ x (ipDen μ v)) =
      Function.update (aext V (l.map (atSub x v)) μ) x (ipDen μ v) := by
  induction l generalizing μ with
  | nil => simp [aext]
  | cons t l ih =>
    simp only [ids, List.map_cons, List.mem_cons, not_or] at hx
    simp only [plainP, ids, List.all_eq_true, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not] at hv
    have hv' : plainP l v = true := by
      simp only [plainP, ids, List.all_eq_true, Bool.not_eq_true', List.contains_eq_mem,
        decide_eq_false_iff_not]
      exact fun y hy h' => hv y hy (by simp only [List.map_cons]; exact List.mem_cons_of_mem _ h')
    have ht : t.id ∉ ipVs v := fun hy => hv _ hy (by simp)
    simp only [aext, List.map_cons]
    have hval : aval V (Function.update μ x (ipDen μ v)) t = aval V μ (atSub x v t) := by
      have : (atSub x v t).args.map (argV V.ρ.dv μ) =
          t.args.map (argV V.ρ.dv (Function.update μ x (ipDen μ v))) := by
        simp only [atSub, List.map_map]
        exact List.map_congr_left fun p _ => by simp [argV, ipDen_subst]
      unfold aval; rw [this]; rfl
    have hden : ipDen (Function.update μ t.id (aval V μ (atSub x v t))) v = ipDen μ v :=
      ipDen_upd_of _ _ _ _ (ipHas_false_of _ _ ht)
    rw [hval, Function.update_comm hx.1, ← hden, ih hx.2 hv']; rfl

/-- `x` substituted by `v` in a term's factors and atom arguments. -/
def tSub (x : ℕ) (v : IPoly) (t : ATm) : ATm :=
  ⟨t.c, t.sums, t.ats.map (atSub x v), t.dels.map (facSub x v), t.phs.map (facSub x v)⟩

/-- `x` is free in `t` and `v` reads no bound variable or atom of `t`. -/
def subOK (x : ℕ) (v : IPoly) (t : ATm) : Bool :=
  !(abvs t).contains x && !(ids t.ats).contains x && plainP t.ats v &&
    (ipVs v).all fun y => !(abvs t).contains y

theorem tSub_den (V : LV) (x : ℕ) (v : IPoly) (t : ATm) (h : subOK x v t = true) (ν : ℕ → ℤ) :
    (tSub x v t).den V ν = t.den V (Function.update ν x (ipDen ν v)) := by
  simp only [subOK, Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem,
    decide_eq_false_iff_not, List.all_eq_true] at h
  obtain ⟨⟨⟨hb, hi⟩, hp⟩, hvb⟩ := h
  have hbody : ∀ μ, abody V (tSub x v t) μ = abody V t (Function.update μ x (ipDen μ v)) := by
    intro μ
    unfold abody
    simp only [tSub]
    rw [aext_sub' V t.ats x hi v hp μ, bodyOf_sub, ipDen_aext V _ v (by
      simpa [plainP, ids, atSub] using hp)]
  unfold ATm.den
  show t.c.den V.ρ * sumsV V.ρ.dv t.sums (abody V (tSub x v t)) ν = _
  congr 1
  rw [sumsV_congr _ _ _ _ hbody]
  have key : ∀ (s : List (ℕ × DMon)), (∀ p ∈ s, p.1 ∈ abvs t) → ∀ μ : ℕ → ℤ,
      sumsV V.ρ.dv s (fun μ => abody V t (Function.update μ x (ipDen μ v))) μ =
        sumsV V.ρ.dv s (abody V t) (Function.update μ x (ipDen μ v)) := by
    intro s
    induction s with
    | nil => intro _ μ; rfl
    | cons p s ih =>
      intro hs μ
      simp only [sumsV]
      refine Finset.sum_congr rfl fun j _ => ?_
      have hp := hs p List.mem_cons_self
      have hpx : p.1 ≠ x := fun e => hb (e ▸ hp)
      rw [ih (fun q hq => hs q (List.mem_cons_of_mem _ hq)),
        ipDen_upd_of _ _ _ _ (ipHas_false_of _ _ fun hy => hvb _ hy hp),
        Function.update_comm hpx]
  exact key t.sums (fun p hp => List.mem_map_of_mem hp) ν

theorem axDen_tSub (V : LV) (x : ℕ) (v : IPoly) (e : List ATm) (h : e.all (subOK x v) = true)
    (ν : ℕ → ℤ) : axDen V ν (e.map (tSub x v)) = axDen V (Function.update ν x (ipDen ν v)) e := by
  simp only [List.all_eq_true] at h
  unfold axDen
  rw [List.map_map]
  congr 1
  exact List.map_congr_left fun t ht => tSub_den V x v t (h t ht) ν

/-- The checks of a loop: the invariant at `J = lo` is the initial state and one step takes
the invariant at `J` to the invariant at `J + 1`. -/
def loopChk (fl : List ℕ) (PW : List (DMon × DMon)) (L : List DMon) (J : ℕ) (lo : ℤ)
    (s i0 inv : List ATm) : Bool :=
  decide (2 * IN ≤ J) && inv.all (subOK J (ipC lo)) &&
    inv.all (subOK J (ipAdd (ipV J) (ipC 1))) &&
    (nfA fl PW (i0 ++ axNeg (inv.map (tSub J (ipC lo))))).isEmpty &&
    (match appSym L (apart (fresh s inv) s) inv with
     | some e1 => (nfA fl PW (e1 ++ axNeg (inv.map (tSub J (ipAdd (ipV J) (ipC 1)))))).isEmpty
     | none => false)

theorem setL_update (ν : ℕ → ℤ) (b : ℕ) (js : List ℤ) (J : ℕ) (hJ : b + js.length ≤ J)
    (v : ℤ) : setL (Function.update ν J v) b js = Function.update (setL ν b js) J v := by
  funext y
  rw [setL_apply]
  by_cases hy : y = J
  · subst hy; simp only [Function.update_self]; rw [if_neg (by omega)]
  · rw [Function.update_of_ne hy, Function.update_of_ne hy, setL_apply]

theorem loop_inv (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (L : List DMon) (J : ℕ) (lo : ℤ)
    (s i0 inv : List ATm) (u w : Fn) (hu : ∀ μ, u μ = axDen V μ s) (hw : ∀ μ, w μ = axDen V μ i0)
    (hL : L.length < IN) (h : loopChk fl PW L J lo s i0 inv = true) (k : ℕ) (ν : ℕ → ℤ) :
    loopSem V.ρ.dv L J lo u w k ν = axDen V (Function.update ν J (lo + k)) inv := by
  unfold loopChk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.isEmpty_iff] at h
  obtain ⟨⟨⟨⟨hJ, hs0⟩, hs1⟩, h0⟩, h1⟩ := h
  induction k generalizing ν with
  | zero =>
    have hz := nfA_zero V hρ fl hfl PW hPW _ h0 ν
    rw [axDen_append, axDen_neg, axDen_tSub V J _ inv hs0] at hz
    simp only [loopSem, hw, Nat.cast_zero, add_zero]
    rw [show (lo : ℤ) = ipDen ν (ipC lo) from (ipDen_C ν lo).symm]
    linear_combination hz
  | succ k ih =>
    split at h1
    · next e1 he1 =>
      set μ := Function.update ν J (lo + k)
      have hz := nfA_zero V hρ fl hfl PW hPW _ (List.isEmpty_iff.1 h1) μ
      rw [axDen_append, axDen_neg, axDen_tSub V J _ inv hs1, appSym_ok V hρ L _ inv e1 he1] at hz
      have e2 : ipDen μ (ipAdd (ipV J) (ipC 1)) = lo + (k + 1 : ℕ) := by
        simp only [ipDen_add, ipDen_V, ipDen_C, μ, Function.update_self]; push_cast; ring
      rw [e2, Function.update_idem] at hz
      simp only [loopSem]
      rw [← sub_eq_zero.1 hz]
      refine sumT_congr _ _ _ fun js hjs => ?_
      simp only [List.length_map] at hjs
      rw [hu, axDen_apart, ih (setL ν 0 js),
        ← setL_update ν IN js J (by simp only [IN] at hL hJ ⊢; omega),
        ← setL_update ν 0 js J (by simp only [IN] at hL hJ ⊢; omega)]
    · simp at h1

/-! ## Programs -/

/-- A program as one expression. -/
def Prog.sym (fl : List ℕ) (PW : List (DMon × DMon)) : Prog → Option (List ATm)
  | .leaf e => some e
  | .app L a b =>
    match a.sym fl PW, b.sym fl PW with
    | some x, some y => appSym L (apart (fresh x y) x) y
    | _, _ => none
  | .ten k a b =>
    match a.sym fl PW, b.sym fl PW with
    | some x, some y => tenSym k x (apart (fresh x y) y)
    | _, _ => none
  | .mul a b =>
    match a.sym fl PW, b.sym fl PW with
    | some x, some y =>
      if axMulOK x (apart (fresh x y) y) then some (axMul x (apart (fresh x y) y)) else none
    | _, _ => none
  | .add a b =>
    match a.sym fl PW, b.sym fl PW with
    | some x, some y => some (x ++ y)
    | _, _ => none
  | .inn L a b =>
    match a.sym fl PW, b.sym fl PW with
    | some x, some y => innSym L (apart (fresh x y) x) y
    | _, _ => none
  | .cnj a =>
    match a.sym fl PW with
    | some x => if conjOK x then some (x.map tConj) else none
    | none => none
  | .loop L J lo cnt step init inv =>
    match step.sym fl PW, init.sym fl PW with
    | some x, some y =>
      if step.counts = [] ∧ init.counts = [] ∧ L.length < IN ∧
          (ipVs cnt).all (fun v => decide (2 * IN ≤ v)) = true ∧
          inv.all (subOK J (ipAdd (ipC lo) cnt)) = true ∧
          loopChk fl PW L J lo x y inv = true then
        some (inv.map (tSub J (ipAdd (ipC lo) cnt)))
      else none
    | _, _ => none

/-- Every loop count of `p` is nonnegative at each valuation agreeing with `ν` from `2 IN` on. -/
def HiOK (p : Prog) (ν : ℕ → ℤ) : Prop :=
  ∀ c ∈ p.counts, ∀ μ : ℕ → ℤ, (∀ v, 2 * IN ≤ v → μ v = ν v) → 0 ≤ ipDen μ c

theorem HiOK.sub {p q : Prog} {ν : ℕ → ℤ} (h : HiOK p ν) (hsub : ∀ c ∈ q.counts, c ∈ p.counts)
    (μ0 : ℕ → ℤ) (hμ0 : ∀ v, 2 * IN ≤ v → μ0 v = ν v) : HiOK q μ0 :=
  fun c hc μ hμ => h c (hsub c hc) μ fun v hv => (hμ v hv).trans (hμ0 v hv)

theorem setL_hi (ν : ℕ → ℤ) (b : ℕ) (js : List ℤ) (hb : b + js.length ≤ 2 * IN) :
    ∀ v, 2 * IN ≤ v → setL ν b js v = ν v := by
  intro v hv; rw [setL_apply, if_neg (by omega)]

theorem shift_hi (ν : ℕ → ℤ) (k : ℕ) : ∀ v, 2 * IN ≤ v → (ν ∘ shiftN k) v = ν v := by
  intro v hv; simp [shiftN, show ¬ v < 2 * IN by omega]

theorem Prog.sym_ok (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (p : Prog) (e : List ATm)
    (h : p.sym fl PW = some e) (ν : ℕ → ℤ) (hc : HiOK p ν) : p.sem V ν = axDen V ν e := by
  induction p generalizing e ν with
  | leaf e' => simp only [Prog.sym, Option.some.injEq] at h; subst h; rfl
  | app L a b iha ihb =>
    simp only [Prog.sym] at h
    split at h
    · next x y hx hy =>
      have hL : L.length < IN := by
        unfold appSym at h; split_ifs at h with hcc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hcc; exact hcc.1.1.1.1.1.1
      rw [appSym_ok V hρ L _ y e h]
      simp only [Prog.sem]
      refine sumT_congr _ _ _ fun js hjs => ?_
      simp only [List.length_map] at hjs
      rw [axDen_apart, iha x hx _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        (setL_hi ν IN js (by simp only [IN] at hL ⊢; omega))),
        ihb y hy _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        (setL_hi ν 0 js (by simp only [IN] at hL ⊢; omega)))]
    · simp at h
  | ten k a b iha ihb =>
    simp only [Prog.sym] at h
    split at h
    · next x y hx hy =>
      rw [tenSym_ok V hρ k x _ e h]
      simp only [Prog.sem]
      rw [axDen_apart, iha x hx _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        fun _ _ => rfl), ihb y hy _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        (shift_hi ν k))]
    · simp at h
  | mul a b iha ihb =>
    simp only [Prog.sym] at h
    split at h
    · next x y hx hy =>
      split_ifs at h with hm
      simp only [Option.some.injEq] at h
      subst h
      simp only [Prog.sem]
      rw [axDen_mul V hρ _ _ hm, axDen_apart,
        iha x hx _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _ fun _ _ => rfl),
        ihb y hy _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _ fun _ _ => rfl)]
    · simp at h
  | add a b iha ihb =>
    simp only [Prog.sym] at h
    split at h
    · next x y hx hy =>
      simp only [Option.some.injEq] at h
      subst h
      simp only [Prog.sem]
      rw [axDen_append, iha x hx _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        fun _ _ => rfl), ihb y hy _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        fun _ _ => rfl)]
    · simp at h
  | inn L a b iha ihb =>
    simp only [Prog.sym] at h
    split at h
    · next x y hx hy =>
      have hL : L.length < IN := by
        unfold innSym at h; split_ifs at h
        unfold appSym at h; split_ifs at h with hcc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hcc; exact hcc.1.1.1.1.1.1
      rw [innSym_ok V hρ L _ y e h]
      simp only [Prog.sem]
      refine sumT_congr _ _ _ fun js hjs => ?_
      simp only [List.length_map] at hjs
      rw [axDen_apart, iha x hx _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        (setL_hi ν 0 js (by simp only [IN] at hL ⊢; omega))),
        ihb y hy _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _
        (setL_hi ν 0 js (by simp only [IN] at hL ⊢; omega)))]
    · simp at h
  | cnj a iha =>
    simp only [Prog.sym] at h
    split at h
    · next x hx =>
      split_ifs at h with hcj
      simp only [Option.some.injEq] at h
      subst h
      simp only [Prog.sem]
      rw [iha x hx _ (hc.sub (fun c h' => by simp [Prog.counts, h']) _ fun _ _ => rfl),
        axDen_conj V x hcj]
    · simp at h
  | loop L J lo cnt step init inv ihs ihi =>
    simp only [Prog.sym] at h
    split at h
    · next x y hx hy =>
      split_ifs at h with hcc
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨hsc, hic, hL, hcv, hsub, hchk⟩ := hcc
      have hu : ∀ μ, step.sem V μ = axDen V μ x := fun μ =>
        ihs x hx μ fun c h' => by simp [hsc] at h'
      have hw : ∀ μ, init.sem V μ = axDen V μ y := fun μ =>
        ihi y hy μ fun c h' => by simp [hic] at h'
      have h0 : 0 ≤ ipDen ν cnt := hc cnt (by simp [Prog.counts]) ν fun _ _ => rfl
      simp only [Prog.sem]
      rw [loop_inv V hρ fl hfl PW hPW L J lo x y inv _ _ hu hw hL hchk, axDen_tSub V J _ inv hsub,
        ipDen_add, ipDen_C, Int.toNat_of_nonneg h0]
    · simp at h

/-! ## Certificates -/

theorem pow_floor (a b n : ℕ) (h : a = b ^ n) (hb : 2 ≤ b) (hn : 1 ≤ n) : 2 ≤ a := by
  subst h
  calc 2 ≤ b := hb
    _ = b ^ 1 := (pow_one b).symm
    _ ≤ b ^ n := Nat.pow_le_pow_right (by omega) hn

end ArchytasZX.Lang

namespace ArchytasZX.Lang

open ArchytasZX ArchytasZX.Sym ArchytasZX.LSym

theorem nfA_of_lit (fl : List ℕ) (PW : List (DMon × DMon)) (e : List ATm) (ns : List (Option ATm))
    (hn : e.map (normA fl PW AROUNDS) = ns) :
    nfA fl PW e = (ns.filterMap id).foldl (fun acc t => taAdd t acc) [] := by
  unfold nfA
  rw [← hn, List.filterMap_map]
  rfl

theorem hi_of (p : Prog) (hv : ∀ c ∈ p.counts, ∀ v ∈ ipVs c, 2 * IN ≤ v) (ν : ℕ → ℤ)
    (h : ∀ c ∈ p.counts, 0 ≤ ipDen ν c) : HiOK p ν := by
  intro c hc μ hμ
  rw [ipDen_reads c μ ν fun v hv' => hμ v (hv c hc v hv')]
  exact h c hc

/-- `p` equals `r` at every valuation where its loop counts are nonnegative, given the
translation and every term's normal form. -/
theorem certify_lit_c (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (p : Prog) (r e : List ATm)
    (ns : List (Option ATm)) (he : p.sym fl PW = some e)
    (hn : (e ++ axNeg r).map (normA fl PW AROUNDS) = ns)
    (hz : (ns.filterMap id).foldl (fun acc t => taAdd t acc) [] = [])
    (hv : ∀ c ∈ p.counts, ∀ v ∈ ipVs c, 2 * IN ≤ v) :
    ∀ ν, (∀ c ∈ p.counts, 0 ≤ ipDen ν c) → p.sem V ν = axDen V ν r := by
  intro ν hc
  have hz' := nfA_zero V hρ fl hfl PW hPW _ (by rw [nfA_of_lit fl PW _ ns hn, hz]) ν
  rw [axDen_append, axDen_neg] at hz'
  rw [Prog.sym_ok V hρ fl hfl PW hPW p e he ν (hi_of p hv ν hc)]
  linear_combination hz'

/-- `certify_lit_c` for a program without loops. -/
theorem certify_lit (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (p : Prog) (r e : List ATm)
    (ns : List (Option ATm)) (hc : p.counts = []) (he : p.sym fl PW = some e)
    (hn : (e ++ axNeg r).map (normA fl PW AROUNDS) = ns)
    (hz : (ns.filterMap id).foldl (fun acc t => taAdd t acc) [] = []) :
    p.sem V = fun ν => axDen V ν r := by
  funext ν
  exact certify_lit_c V hρ fl hfl PW hPW p r e ns he hn hz (by simp [hc]) ν (by simp [hc])

/-- Two programs equal at every valuation where their loop counts are nonnegative. -/
theorem certify_eq_lit_c (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (p q : Prog) (e r : List ATm)
    (ns : List (Option ATm)) (he : p.sym fl PW = some e) (hr : q.sym fl PW = some r)
    (hn : (e ++ axNeg r).map (normA fl PW AROUNDS) = ns)
    (hz : (ns.filterMap id).foldl (fun acc t => taAdd t acc) [] = [])
    (hvp : ∀ c ∈ p.counts, ∀ v ∈ ipVs c, 2 * IN ≤ v)
    (hvq : ∀ c ∈ q.counts, ∀ v ∈ ipVs c, 2 * IN ≤ v) :
    ∀ ν, (∀ c ∈ p.counts ++ q.counts, 0 ≤ ipDen ν c) → p.sem V ν = q.sem V ν := by
  intro ν hc
  rw [certify_lit_c V hρ fl hfl PW hPW p r e ns he hn hz hvp ν
    (fun c h' => hc c (List.mem_append_left _ h')),
    Prog.sym_ok V hρ fl hfl PW hPW q r hr ν (hi_of q hvq ν fun c h' => hc c
      (List.mem_append_right _ h'))]

/-- `certify_eq_lit_c` for programs without loops. -/
theorem certify_eq_lit (V : LV) (hρ : V.ρ.OK) (fl : List ℕ) (hfl : FloorOK fl V.ρ)
    (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ) (p q : Prog) (e r : List ATm)
    (ns : List (Option ATm)) (hcp : p.counts = []) (hcq : q.counts = [])
    (he : p.sym fl PW = some e) (hr : q.sym fl PW = some r)
    (hn : (e ++ axNeg r).map (normA fl PW AROUNDS) = ns)
    (hz : (ns.filterMap id).foldl (fun acc t => taAdd t acc) [] = []) :
    p.sem V = q.sem V := by
  funext ν
  exact certify_eq_lit_c V hρ fl hfl PW hPW p q e r ns he hr hn hz (by simp [hcp])
    (by simp [hcq]) ν (by simp [hcp, hcq])

end ArchytasZX.Lang
