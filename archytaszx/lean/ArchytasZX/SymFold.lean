/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymCarry

/-!
# Folded controlled phases

`foldBZS` folds a binder `B (a, b) → k`, a Z spider `k → k'` whose phases are a compact table of
kind `2` on `d²` (`c (j / d) (j % d) / M`) and a splitter `k' → (a', b')` into the one term
`[d ∣ a' - a] [d ∣ b' - b] ω_M^(c a b)`, the labels `k`, `k'` summed out. `trNodesK` is
`trNodesG` with the identity normaliser that takes such a block of three nodes as one step when
`k`, `k'` are internal labels read nowhere else (`foldOK`); `eval_trK` and `equal_symSK` prove
`Equal` through it. `elimStepF` adds two steps to `elimStepC`, each keeping a term's value for
every valuation: `17` the delta `f = [M ∣ p]` at `j` plus `-(c s)` times the delta
`g = [m ∣ s x + b]` at `i` of the others (`x` the pivot of `g`, `c` the coefficient of `x` in
`p`, `M ∣ c s m`), coefficients mod `M`; `18` the root of unity `[M ∣ p]` at `i` split into
`[M ∣ p₀]` and `[d ∣ p₁ / (M / d)]`, `p₁` the monomials of `p` reading the sum `x` at `j` of
constant range `d ∣ M`, each coefficient a multiple of `M / d`. `rxNfF` is `rxNfC` with
`elimStepF`. `elimAllFS`, `elimAllF_append`, `elimAllF_of_S` and `normTmF_one` check a round's
steps in chunks; `trNodesKM` is `trNodesK` with the labels before each step as a bit mask
(`trNetKM_eq`).
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Rule 17 -/

/-- Rule `17`: `f = [M ∣ p]` as `[M ∣ (p - c s q) mod M]` beside the delta `[m ∣ q]` at `i` of
`pre ++ post`, `q = s x + b` at its pivot `x`, `c` the coefficient of `x` in `p`, `M ∣ c s m`. -/
def carry17 (i : ℕ) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option (List Fac × List Fac) :=
  match (pre ++ post)[i]? with
  | some g =>
    if g.1.isConst = true ∧ f.1.isConst = true ∧ 0 < f.1.c then
      match pivotOf g.2 with
      | some q =>
        match ipLin q.1 f.2 with
        | some (a, _) =>
          match ipConst a with
          | some c =>
            if (c * q.2.1 * (g.1.c : ℤ)) % (f.1.c : ℤ) = 0 then
              some (pre ++ (f.1, ipModC f.1.c (ipAdd f.2 (ipSmul (-(c * q.2.1)) g.2))) :: post, ps)
            else none
          | none => none
        | none => none
      | none => none
    else none
  | none => none

theorem carry17_ok (dv : ℕ → ℕ) (i : ℕ) (pre : List Fac) (f : Fac)
    (post ps ds' ps' : List Fac) (h : carry17 i pre f post ps = some (ds', ps')) (ν : ℕ → ℤ) :
    bodyOf dv ds' ps' ν = bodyOf dv (pre ++ f :: post) ps ν := by
  unfold carry17 at h
  split at h
  · next g hg =>
    split_ifs at h with hc
    obtain ⟨hgk, hk, _⟩ := hc
    split at h
    · next q _ =>
      split at h
      · next a _ _ =>
        split at h
        · next c _ =>
          split_ifs at h with hdv
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          have hmem : g ∈ pre ++ post := List.mem_of_getElem? hg
          rw [bodyOf_cons_mid, bodyOf_cons_mid]
          rcases delV_cases dv ν g with h0 | h1
          · rw [bodyOf_zero dv _ ps ν g hmem h0]; ring
          · congr 1
            refine delV_congr dv ν f _ rfl ?_
            have hgd := delV_one dv ν g h1
            rw [DMon.den_isConst dv g.1 hgk] at hgd
            rw [DMon.den_isConst dv f.1 hk]
            refine (dvd_iff_of_dvd_sub ?_).trans (dvd_iff_of_dvd_sub (ipDen_modC ν f.1.c _))
            rw [ipDen_add, ipDen_smul, show ipDen ν f.2 - (ipDen ν f.2 + -(c * q.2.1) *
              ipDen ν g.2) = (c * q.2.1) * ipDen ν g.2 by ring]
            obtain ⟨u, hu⟩ := hgd
            rw [hu, show c * q.2.1 * ((g.1.c : ℤ) * u) = (c * q.2.1 * (g.1.c : ℤ)) * u by ring]
            exact Dvd.dvd.mul_right (Int.dvd_of_emod_eq_zero hdv) _
        · simp at h
      · simp at h
    · simp at h
  · simp at h

/-! ## Rule 18 -/

/-- The monomials of `p` reading `x`. -/
def ipWith (x : ℕ) (p : IPoly) : IPoly := p.filter fun m => m.1.contains x

/-- The monomials of `p` not reading `x`. -/
def ipWithout (x : ℕ) (p : IPoly) : IPoly := p.filter fun m => !m.1.contains x

theorem ipDen_split (ν : ℕ → ℤ) (P : IMon × ℤ → Bool) (p : IPoly) :
    ipDen ν p = ipDen ν (p.filter P) + ipDen ν (p.filter fun m => !P m) := by
  induction p with
  | nil => simp [ipDen]
  | cons t p ih =>
    unfold ipDen at ih ⊢
    by_cases hP : P t = true <;> simp [List.filter_cons, hP, ih] <;> ring

theorem ipDen_divc (ν : ℕ → ℤ) (k : ℤ) (p : IPoly) (h : p.all (fun m => m.2 % k == 0) = true) :
    ipDen ν p = k * ipDen ν (p.map fun m => (m.1, m.2 / k)) := by
  induction p with
  | nil => simp [ipDen]
  | cons t p ih =>
    simp only [List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    unfold ipDen at ih ⊢
    simp only [List.map_cons, List.sum_cons] at ih ⊢
    rw [ih h.2, mul_add, ← mul_assoc, Int.mul_ediv_cancel' (Int.dvd_of_emod_eq_zero h.1)]

/-- Rule `18`: the root of unity `g = [M ∣ p]` at `i` as `[M ∣ p₀]` in place and `[d ∣ p₁ / k]`
appended, `x` the sum at `j` of constant range `d`, `M = k d`, `p₁` the monomials reading `x`, each
coefficient a multiple of `k`, `p₀` the others. -/
def phSplit (t : STm) (i j : ℕ) : Option STm :=
  match t.phs[i]?, t.sums[j]? with
  | some g, some p =>
    if p.2.isConst = true ∧ 0 < p.2.c ∧ g.1.isConst = true ∧ 0 < g.1.c ∧ g.1.c % p.2.c = 0 ∧
        (ipWith p.1 g.2).all (fun m => m.2 % ((g.1.c / p.2.c : ℕ) : ℤ) == 0) = true then
      some ⟨t.c, t.sums, t.dels, t.phs.set i (g.1, ipWithout p.1 g.2) ++
        [(p.2, (ipWith p.1 g.2).map fun m => (m.1, m.2 / ((g.1.c / p.2.c : ℕ) : ℤ)))]⟩
    else none
  | _, _ => none

theorem prod_set_mul (f : Fac → ℂ) (x : ℂ) (g g' : Fac) (hx : f g = f g' * x) :
    ∀ (l : List Fac) (i : ℕ), l[i]? = some g → ((l.set i g').map f).prod * x = (l.map f).prod
  | [], i, h => by simp at h
  | a :: l, 0, h => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp only [List.set_cons_zero, List.map_cons, List.prod_cons, hx]
    ring
  | a :: l, i + 1, h => by
    simp only [List.getElem?_cons_succ] at h
    simp only [List.set_cons_succ, List.map_cons, List.prod_cons]
    rw [mul_assoc, prod_set_mul f x g g' hx l i h]

theorem phSplit_ok (ρ : Val) (t t' : STm) (i j : ℕ) (h : phSplit t i j = some t') (ν : ℕ → ℤ) :
    t'.den ρ ν = t.den ρ ν := by
  unfold phSplit at h
  split at h
  · next g p hg hp =>
    split_ifs at h with hc
    obtain ⟨hpk, hpos, hgk, hgpos, hdvd, hall⟩ := hc
    simp only [Option.some.injEq] at h
    subst h
    have hb : body ρ.dv ⟨t.c, t.sums, t.dels, t.phs.set i (g.1, ipWithout p.1 g.2) ++
        [(p.2, (ipWith p.1 g.2).map fun m => (m.1, m.2 / ((g.1.c / p.2.c : ℕ) : ℤ)))]⟩ =
        body ρ.dv t := by
      funext μ
      simp only [body, List.map_append, List.map_cons, List.map_nil, List.prod_append,
        List.prod_cons, List.prod_nil, mul_one]
      congr 1
      refine prod_set_mul (phV ρ.dv μ) _ g _ ?_ t.phs i hg
      simp only [phV, DMon.den_isConst ρ.dv g.1 hgk, DMon.den_isConst ρ.dv p.2 hpk]
      have hkd : g.1.c = p.2.c * (g.1.c / p.2.c) :=
        (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdvd)).symm
      rw [ipDen_split μ (fun m => m.1.contains p.1) g.2, zpow_add₀ (om_ne_zero _ hgpos),
        mul_comm (om g.1.c ^ ipDen μ (List.filter (fun m => m.1.contains p.1) g.2))]
      congr 1
      rw [om_up g.1.c p.2.c hgpos hdvd, ← ipDen_divc μ _ _ hall]
      rfl
    unfold STm.den
    rw [hb]
  · simp at h

/-! ## Steps and normal forms -/

/-- Rule `18` `phSplit`; rule `17` `carry17` at the delta at `j`; else `elimStepC`. -/
def elimStepF (R : List (ℕ × ℕ)) (t : STm) (i r j : ℕ) : Option STm :=
  if r = 18 then phSplit t i j
  else if r = 17 then
    match t.dels.drop j with
    | f :: post =>
      match carry17 i (t.dels.take j) f post t.phs with
      | some (ds, ps) => some ⟨t.c, t.sums, ds, ps⟩
      | none => none
    | [] => none
  else elimStepC R t i r j

theorem elimStepF_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (t t' : STm) (i r j : ℕ)
    (h : elimStepF R t i r j = some t') (ν : ℕ → ℤ) (hν : InLab R ν) :
    t'.den ρ ν = t.den ρ ν := by
  unfold elimStepF at h
  split_ifs at h with h18 h17
  · exact phSplit_ok ρ t t' i j h ν
  · split at h
    · next f post hd =>
      split at h
      · next ds ps hc =>
        simp only [Option.some.injEq] at h
        subst h
        have hdl : t.dels.take j ++ f :: post = t.dels := by rw [← hd, List.take_append_drop]
        have hb : body ρ.dv ⟨t.c, t.sums, ds, ps⟩ = body ρ.dv t := by
          funext μ
          rw [body_eq, body_eq]
          simp only
          rw [carry17_ok ρ.dv i _ f post t.phs ds ps hc μ, hdl]
        unfold STm.den
        rw [hb]
      · simp at h
    · simp at h
  · exact elimStepC_ok ρ hρ R t t' i r j h ν hν

/-- `elimStepF` along the hints, each result forced; the term at the first that fails. -/
def elimAllF (R : List (ℕ × ℕ)) : List (ℕ × ℕ × ℕ) → STm → STm
  | [], t => t
  | (i, r, j) :: hs, t =>
    match elimStepF R t i r j with
    | some t' => fTm t' (elimAllF R hs)
    | none => t

theorem elimAllF_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) :
    ∀ (hs : List (ℕ × ℕ × ℕ)) (t : STm) (ν : ℕ → ℤ), InLab R ν →
      (elimAllF R hs t).den ρ ν = t.den ρ ν := by
  intro hs
  induction hs with
  | nil => intro t ν _; rfl
  | cons h hs ih =>
    obtain ⟨i, r, j⟩ := h
    intro t ν hν
    simp only [elimAllF]
    cases he : elimStepF R t i r j with
    | none => rfl
    | some t' => dsimp only; rw [fTm_eq, ih _ ν hν, elimStepF_ok ρ hρ R t t' i r j he ν hν]

/-- `normTmC` with `elimAllF`. -/
def normTmF (R : List (ℕ × ℕ)) (fl : List ℕ) : List RHint → STm → Option STm
  | [], t => some t
  | (c, es) :: cs, t =>
    match canonTmH c fl (fTm t (elimAllF R es)) with
    | none => none
    | some t' => fTm t' (normTmF R fl cs)

theorem normTmF_ok (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ) :
    ∀ (cs : List RHint) (t : STm), TmResR ρ R t (normTmF R fl cs t) := by
  intro cs
  induction cs with
  | nil => intro t ν _; rfl
  | cons h cs ih =>
    obtain ⟨c, es⟩ := h
    intro t
    have hc := canonTmH_ok ρ hρ fl hfl c (elimAllF R es t)
    have he := elimAllF_ok ρ hρ R es t
    simp only [normTmF, fTm_eq]
    revert hc
    cases canonTmH c fl (elimAllF R es t) with
    | none => intro hc ν hν; rw [← he ν hν, hc ν]
    | some t' =>
      intro hc
      simp only [fTm_eq]
      have := ih t'
      revert this
      cases normTmF R fl cs t' with
      | none => intro h' ν hν; rw [← he ν hν, ← hc ν, h' ν hν]
      | some t'' => intro h' ν hν; rw [h' ν hν, hc ν, he ν hν]

/-- The terms normalised with their rounds, in order, summed into `acc`. -/
def nfGoF (R : List (ℕ × ℕ)) (fl : List ℕ) : List (List RHint) → List STm → List STm → List STm
  | _, acc, [] => acc
  | hs, acc, t :: ts =>
    match normTmF R fl (hs.headD []) t with
    | none => nfGoF R fl hs.tail acc ts
    | some t' => nfGoF R fl hs.tail (tmAdd t' acc) ts

theorem nfGoF_den (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (ν : ℕ → ℤ) (hν : InLab R ν) : ∀ (e : List STm) (hs : List (List RHint)) (acc : List STm),
    exDen ρ ν (nfGoF R fl hs acc e) = exDen ρ ν acc + exDen ρ ν e := by
  intro e
  induction e with
  | nil => intro hs acc; simp [nfGoF, exDen]
  | cons t ts ih =>
    intro hs acc
    have hn := normTmF_ok ρ hρ R fl hfl (hs.headD []) t
    simp only [nfGoF]
    revert hn
    cases normTmF R fl (hs.headD []) t with
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

theorem normTmF_step (R : List (ℕ × ℕ)) (fl : List ℕ) (h : RHint) (cs : List RHint) (t t' : STm)
    (hh : normTmF R fl [h] t = some t') : normTmF R fl (h :: cs) t = normTmF R fl cs t' := by
  obtain ⟨c, es⟩ := h
  simp only [normTmF, fTm_eq] at hh ⊢
  cases hc : canonTmH c fl (elimAllF R es t) with
  | none => rw [hc] at hh; simp at hh
  | some u =>
    rw [hc] at hh
    simp only [fTm_eq, normTmF, Option.some.injEq] at hh ⊢
    rw [hh]

theorem normTmF_stop (R : List (ℕ × ℕ)) (fl : List ℕ) (h : RHint) (cs : List RHint) (t : STm)
    (hh : normTmF R fl [h] t = none) : normTmF R fl (h :: cs) t = none := by
  obtain ⟨c, es⟩ := h
  simp only [normTmF, fTm_eq] at hh ⊢
  cases hc : canonTmH c fl (elimAllF R es t) with
  | none => rfl
  | some u => rw [hc] at hh; simp [fTm_eq, normTmF] at hh

theorem nfGoF_some (R : List (ℕ × ℕ)) (fl : List ℕ) (h : List RHint) (hs : List (List RHint))
    (acc : List STm) (t u : STm) (ts : List STm) (hh : normTmF R fl h t = some u) :
    nfGoF R fl (h :: hs) acc (t :: ts) = nfGoF R fl hs (tmAdd u acc) ts := by
  simp [nfGoF, hh]

theorem nfGoF_none (R : List (ℕ × ℕ)) (fl : List ℕ) (h : List RHint) (hs : List (List RHint))
    (acc : List STm) (t : STm) (ts : List STm) (hh : normTmF R fl h t = none) :
    nfGoF R fl (h :: hs) acc (t :: ts) = nfGoF R fl hs acc ts := by
  simp [nfGoF, hh]

/-- The residual normalised term by term with the labels' ranges `R`, the rounds `H`, the carry
steps and rules `17`, `18`. -/
def rxNfF (R : List (ℕ × ℕ)) (fl : List ℕ) (H : List (List RHint)) (e : List STm) : List STm :=
  nfGoF R fl H [] e

theorem rxNfF_den (ρ : Val) (hρ : ρ.OK) (R : List (ℕ × ℕ)) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (H : List (List RHint)) (nf : List STm → List STm)
    (hnf : ∀ e ν, exDen ρ ν (nf e) = exDen ρ ν e) (e : List STm) (ν : ℕ → ℤ)
    (hν : InLab R ν) : exDen ρ ν (rxNfF R fl H (nf e)) = exDen ρ ν e := by
  unfold rxNfF
  rw [nfGoF_den ρ hρ R fl hfl ν hν, hnf]
  simp [exDen]

/-! ## Folding a binder, compact spider and splitter -/

/-- The folded term of `B (a, b) → k`, `Z k → k'` with a compact table of kind `2` on `d²`, and
`S k' → (a', b')`, with `k`, `k'`. -/
def foldBZS (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (n1 n2 n3 : Node) : Option (STm × ℕ × ℕ) :=
  match n1.outs, n1.ins, n2.outs, n2.ins, n3.outs, n3.ins, cpData (rl n2.ph) with
  | [k], [a, b], [k'], [k2], [a', b'], [k3], some q =>
    if n1.kind = .B ∧ n2.kind = .Z ∧ n3.kind = .S ∧ k2 = k ∧ k3 = k' ∧
        [k, k', a, b, a', b'].Nodup ∧ dt n1.ph = dt n1.dim ∧ dt n3.dim = dt n1.dim ∧
        dt n3.ph = dt n1.dim ∧ (dt n1.dim).isConst = true ∧ (dt n2.dim).isConst = true ∧
        (dt n2.dim).c = (dt n1.dim).c * (dt n1.dim).c ∧ q.1 = 2 ∧ q.2.2.2 = (dt n1.dim).c ∧
        ptOK ((List.range (dt n2.dim).c).map (cpNum q)) q.2.1 (pt n2.ph) = true then
      some (⟨CMon.ofQ 1, [], [(dt n1.dim, ipSub (ipV a') (ipV a)),
        (dt n1.dim, ipSub (ipV b') (ipV b))],
        [(⟨q.2.1, []⟩, ipSmul q.2.2.1 (ipMul (ipV a) (ipV b)))]⟩, k, k')
    else none
  | _, _, _, _, _, _, _ => none

theorem digitsF (d X Y : ℕ) (hY : Y < d) : (X * d + Y) / d = X ∧ (X * d + Y) % d = Y := by
  have hd : 0 < d := by omega
  refine ⟨?_, ?_⟩
  · rw [Nat.mul_comm, Nat.mul_add_div hd, Nat.div_eq_of_lt hY, add_zero]
  · rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hY]

theorem digits_eqF (d A B A' B' : ℕ) (hB : B < d) (hB' : B' < d) :
    A' * d + B' = A * d + B ↔ A' = A ∧ B' = B := by
  constructor
  · intro h
    have h1 := digitsF d A' B' hB'
    have h2 := digitsF d A B hB
    rw [h] at h1
    exact ⟨h1.1.symm.trans h2.1, h1.2.symm.trans h2.2⟩
  · rintro ⟨rfl, rfl⟩; rfl

/-- The block's sum over its inner labels. -/
theorem bzs_sum (d M : ℕ) (φ : ℕ → ℝ) (c : ℤ)
    (hφ : ∀ j, j < d * d → ex (φ j) = om M ^ (c * (((j / d : ℕ) : ℤ) * ((j % d : ℕ) : ℤ))))
    (A B A' B' : ℕ) (hA : A < d) (hB : B < d) (hA' : A' < d) (hB' : B' < d) :
    ∑ j ∈ range (d * d), ∑ j' ∈ range (d * d),
      tB d d j A B * (tZ (d * d) φ ([j'] ++ [j]) * tB d d j' A' B') =
      (if A' = A ∧ B' = B then 1 else 0) * om M ^ (c * ((A : ℤ) * B)) := by
  have hX : A * d + B < d * d := by nlinarith
  have hY : A' * d + B' < d * d := by nlinarith
  simp only [tB, hA, hB, hA', hB', true_and, ite_mul, one_mul, zero_mul, mul_ite, mul_one,
    mul_zero, Finset.sum_ite_eq', Finset.mem_range, hX, hY, if_true]
  unfold tZ
  simp only [List.singleton_append, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp,
    forall_eq]
  by_cases hYX : A' * d + B' = A * d + B
  · have hdig := (digits_eqF d A B A' B' hB hB').1 hYX
    rw [if_pos hdig, hYX]
    simp only [and_self, Finset.sum_ite_eq, Finset.mem_range, hX, if_true, one_mul]
    rw [hφ _ hX]
    rw [(digitsF d A B hB).1, (digitsF d A B hB).2]
  · rw [if_neg (fun h => hYX ((digits_eqF d A B A' B' hB hB').2 h))]
    try rw [zero_mul]
    refine Finset.sum_eq_zero fun u _ => ?_
    rw [if_neg]
    rintro ⟨h1, h2⟩
    exact hYX (h1.trans h2.symm)

/-- What `foldBZS` reads. -/
theorem foldBZS_spec (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (n1 n2 n3 : Node) (t : STm) (k k' : ℕ) (h : foldBZS dt pt rl n1 n2 n3 = some (t, k, k')) :
    ∃ a b a' b' q, n1.outs = [k] ∧ n1.ins = [a, b] ∧ n2.outs = [k'] ∧ n2.ins = [k] ∧
      n3.outs = [a', b'] ∧ n3.ins = [k'] ∧ cpData (rl n2.ph) = some q ∧ n1.kind = .B ∧
      n2.kind = .Z ∧ n3.kind = .S ∧ [k, k', a, b, a', b'].Nodup ∧ dt n1.ph = dt n1.dim ∧
      dt n3.dim = dt n1.dim ∧ dt n3.ph = dt n1.dim ∧ (dt n1.dim).isConst = true ∧
      (dt n2.dim).isConst = true ∧ (dt n2.dim).c = (dt n1.dim).c * (dt n1.dim).c ∧ q.1 = 2 ∧
      q.2.2.2 = (dt n1.dim).c ∧
      ptOK ((List.range (dt n2.dim).c).map (cpNum q)) q.2.1 (pt n2.ph) = true ∧
      t = ⟨CMon.ofQ 1, [], [(dt n1.dim, ipSub (ipV a') (ipV a)),
        (dt n1.dim, ipSub (ipV b') (ipV b))],
        [(⟨q.2.1, []⟩, ipSmul q.2.2.1 (ipMul (ipV a) (ipV b)))]⟩ := by
  unfold foldBZS at h
  split at h
  · next k0 a b k0' k2 a' b' k3 q h1 h2 h3 h4 h5 h6 h7 =>
    split_ifs at h with hc
    obtain ⟨hB, hZ, hS, rfl, rfl, hnd, hph1, hd3, hph3, hc1, hc2, hc22, hq1, hq2, hpt⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨a, b, a', b', q, h1, h2, h3, h4, h5, h6, h7, hB, hZ, hS, hnd, hph1, hd3, hph3, hc1, hc2,
      hc22, hq1, hq2, hpt, rfl⟩
  · simp at h

/-- The folded term's value is the block's sum over `k`, `k'`. -/
theorem foldBZS_val (E : Env) (ρ : Val) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (n1 n2 n3 : Node) (t : STm) (k k' : ℕ)
    (h : foldBZS dt pt rl n1 n2 n3 = some (t, k, k')) (h1 : NodeOK E ρ dt pt rl n1)
    (h1p : E.dimv n1.ph = (dt n1.ph).den ρ.dv) (h2 : NodeOK E ρ dt pt rl n2)
    (h3 : NodeOK E ρ dt pt rl n3) (h3p : E.dimv n3.ph = (dt n3.ph).den ρ.dv)
    (hk : labDim E k = (dt n1.dim).c * (dt n1.dim).c)
    (hk' : labDim E k' = (dt n1.dim).c * (dt n1.dim).c) (τ : ℕ → ℕ)
    (hr : ∀ l ∈ n1.ins ++ n3.outs, τ l < (dt n1.dim).c) :
    sumOver (labDim E) [k, k'] (fun μ => val E μ n1 * (val E μ n2 * val E μ n3)) τ =
      t.den ρ (fun x => (τ x : ℤ)) := by
  obtain ⟨a, b, a', b', q, ho1, hi1, ho2, hi2, ho3, hi3, hq, hB, hZ, hS, hnd, hph1, hd3, hph3,
    hc1, hc2, hc22, hq1, hq2, hpt, rfl⟩ := foldBZS_spec dt pt rl n1 n2 n3 t k k' h
  set d := (dt n1.dim).c with hd
  have hdim1 : E.dimv n1.dim = d := by rw [h1.dim, DMon.den_isConst _ _ hc1]
  have hdph1 : E.dimv n1.ph = d := by rw [h1p, hph1, DMon.den_isConst _ _ hc1]
  have hdim3 : E.dimv n3.dim = d := by rw [h3.dim, hd3, DMon.den_isConst _ _ hc1]
  have hdph3 : E.dimv n3.ph = d := by rw [h3p, hph3, DMon.den_isConst _ _ hc1]
  have hdim2 : E.dimv n2.dim = d * d := by rw [h2.dim, DMon.den_isConst _ _ hc2, hc22]
  have hφ : ∀ j, j < d * d → ex (E.phase n2.ph j) =
      om q.2.1 ^ (q.2.2.1 * (((j / d : ℕ) : ℤ) * ((j % d : ℕ) : ℤ))) := by
    intro j hj
    rw [cp_ph ρ (dt n2.dim) (pt n2.ph) (rl n2.ph) q hq hpt (E.phase n2.ph) h2.ph j
      (by rw [hc22]; exact hj)]
    congr 1
    simp only [cpNum, hq1, hq2]
    norm_num
    ring
  simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd
  obtain ⟨⟨hkk', hka, hkb, hka', hkb'⟩, ⟨hk'a, hk'b, hk'a', hk'b'⟩, -⟩ := hnd
  have hA := hr a (by simp [hi1])
  have hB' := hr b (by simp [hi1])
  have hA' := hr a' (by simp [ho3])
  have hB'' := hr b' (by simp [ho3])
  simp only [sumOver, hk, hk']
  have hv : ∀ j j', val E (Function.update (Function.update τ k j) k' j') n1 *
      (val E (Function.update (Function.update τ k j) k' j') n2 *
        val E (Function.update (Function.update τ k j) k' j') n3) =
      tB d d j (τ a) (τ b) * (tZ (d * d) (E.phase n2.ph) ([j'] ++ [j]) *
        tB d d j' (τ a') (τ b')) := by
    intro j j'
    obtain ⟨kind1, outs1, ins1, dim1, ph1⟩ := n1
    obtain ⟨kind2, outs2, ins2, dim2, ph2⟩ := n2
    obtain ⟨kind3, outs3, ins3, dim3, ph3⟩ := n3
    simp only at ho1 hi1 ho2 hi2 ho3 hi3 hB hZ hS hdim1 hdph1 hdim2 hdim3 hdph3 ⊢
    subst ho1 hi1 ho2 hi2 ho3 hi3 hB hZ hS
    simp only [val, List.map_cons, List.map_nil, hdim1, hdph1, hdim2, hdim3, hdph3,
      Function.update_self, Function.update_of_ne hkk', Function.update_of_ne (Ne.symm hkk'),
      Function.update_of_ne (Ne.symm hka), Function.update_of_ne (Ne.symm hkb),
      Function.update_of_ne (Ne.symm hka'), Function.update_of_ne (Ne.symm hkb'),
      Function.update_of_ne (Ne.symm hk'a), Function.update_of_ne (Ne.symm hk'b),
      Function.update_of_ne (Ne.symm hk'a'), Function.update_of_ne (Ne.symm hk'b')]
  simp only [hv]
  rw [bzs_sum d q.2.1 (E.phase n2.ph) q.2.2.1 hφ _ _ _ _ hA hB' hA' hB'']
  rw [den_zero, ofQ_den]
  simp only [bodyOf, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, delV,
    phV, ipDen_sub, ipDen_smul, ipDen_mul, ipDen_V, DMon.den_isConst _ _ hc1, Rat.cast_one,
    one_mul]
  have e1 : ((((dt n1.dim).c : ℕ) : ℤ) ∣ ((τ a' : ℕ) : ℤ) - ((τ a : ℕ) : ℤ)) ↔ τ a' = τ a :=
    dvd_iff_eq_lt (dt n1.dim).c _ _ hA' hA
  have e2 : ((((dt n1.dim).c : ℕ) : ℤ) ∣ ((τ b' : ℕ) : ℤ) - ((τ b : ℕ) : ℤ)) ↔ τ b' = τ b :=
    dvd_iff_eq_lt (dt n1.dim).c _ _ hB'' hB'
  simp only [e1, e2, DMon.den, npow, mul_one]
  by_cases h1' : τ a' = τ a <;> by_cases h2' : τ b' = τ b <;> simp [h1', h2']

/-! ## Translation with folded blocks -/

/-- The labels of three nodes. -/
def blkL (n1 n2 n3 : Node) : List ℕ := n1.outs ++ n1.ins ++ (n2.outs ++ n2.ins) ++ (n3.outs ++ n3.ins)

/-- The inner labels `k`, `k'` internal, once each, and read neither before nor after. -/
def foldOK (int pre : List ℕ) (k k' : ℕ) (ns : List Node) : Bool :=
  !pre.contains k && !pre.contains k' && !(labelsOf ns).contains k &&
    !(labelsOf ns).contains k' &&
    ((int.filter fun l => l == k || l == k') == [k, k'] ||
      (int.filter fun l => l == k || l == k') == [k', k])

/-- The internal labels the block `L` reads first, after `pre`, but `k`, `k'`. -/
def firstBlk (int pre L : List ℕ) (k k' : ℕ) : List ℕ :=
  int.filter fun l => L.contains l && !pre.contains l && !(l == k || l == k')

/-- One node's step on the tail's expression. -/
def kSingle (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node) (rest : Option (List STm)) :
    Option (List STm) :=
  match trNode dt pt rl (OFF + i) n, rest with
  | some a, some b =>
    if exMulOK a b then some (exSums (labSums dt labs (firstAt int pre n)) (exMul a b)) else none
  | _, _ => none

/-- A folded block's step on the tail's expression. -/
def kBlock (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (int pre L : List ℕ) (t : STm) (k k' : ℕ)
    (rest : Option (List STm)) : Option (List STm) :=
  match rest with
  | some b =>
    if exMulOK [t] b then some (exSums (labSums dt labs (firstBlk int pre L k k')) (exMul [t] b))
    else none
  | none => none

/-- `trNodesG` with the identity normaliser, each `foldBZS` block whose inner labels pass
`foldOK` one step. -/
def trNodesK (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int : List ℕ) : List ℕ → ℕ → List Node → Option (List STm)
  | _, _, [] => some [⟨CMon.ofQ 1, [], [], []⟩]
  | pre, i, n :: ns =>
    match ns with
    | n2 :: n3 :: ns' =>
      match foldBZS dt pt rl n n2 n3 with
      | some (t, k, k') =>
        if foldOK int pre k k' ns' = true then
          kBlock dt labs int pre (blkL n n2 n3) t k k'
            (trNodesK dt pt rl labs int (pre ++ blkL n n2 n3) (i + 3) ns')
        else kSingle dt pt rl labs int pre i n
          (trNodesK dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) (n2 :: n3 :: ns'))
      | none => kSingle dt pt rl labs int pre i n
          (trNodesK dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) (n2 :: n3 :: ns'))
    | ns => kSingle dt pt rl labs int pre i n
        (trNodesK dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns)
termination_by structural _ _ ns => ns

/-- The conditions on each node that `trNodesG_ok` takes. -/
def NodesOKF (E : Env) (ρ : Val) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (ns : List Node) : Prop :=
  ∀ n ∈ ns, NodeOK E ρ dt pt rl n ∧
    (bsKind n.kind = true → E.dimv n.ph = (dt n.ph).den ρ.dv) ∧ (∀ l ∈ n.outs ++ n.ins, l < OFF) ∧
    ∀ p ∈ (n.outs ++ n.ins).zip (legDMs dt n), labDim E p.1 = p.2.den ρ.dv

/-- The value through the tail's expression. -/
def TailOK (E : Env) (ρ : Val) (int pre : List ℕ) (ns : List Node) (rest : Option (List STm)) :
    Prop :=
  ∀ b, rest = some b → ∀ σ : ℕ → ℕ, (∀ l ∈ labelsOf ns, l ∉ firstOf int pre ns → σ l < labDim E l) →
    sumOver (labDim E) (firstOf int pre ns) (prodVal E ns) σ = exDen ρ (fun x => (σ x : ℤ)) b

theorem kSingle_ok (E : Env) (ρ : Val) (hρ : ρ.OK) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ)
    (hD : ∀ l ∈ int, labDim E l = (labDM dt labs l).den ρ.dv) (n : Node) (ns : List Node)
    (hn : NodesOKF E ρ dt pt rl (n :: ns)) (pre : List ℕ) (i : ℕ) (rest : Option (List STm))
    (hr : TailOK E ρ int (pre ++ (n.outs ++ n.ins)) ns rest) (e : List STm)
    (h : kSingle dt pt rl labs int pre i n rest = some e) :
    TailOK E ρ int pre (n :: ns) (some e) := by
  intro e' he' σ hσ
  simp only [Option.some.injEq] at he'
  subst he'
  unfold kSingle at h
  cases ha : trNode dt pt rl (OFF + i) n with
  | none => rw [ha] at h; simp at h
  | some a =>
  cases hb : rest with
  | none => rw [ha, hb] at h; simp at h
  | some b =>
    rw [ha, hb] at h
    simp only at h
    split_ifs at h with hok
    simp only [Option.some.injEq] at h
    subst h
    obtain ⟨hnode, hph, hlt, hdl⟩ := hn n List.mem_cons_self
    have hperm := firstOf_perm int pre n ns
    have hmem : ∀ l, l ∈ firstOf int pre (n :: ns) →
        l ∈ firstAt int pre n ∨ l ∈ firstOf int (pre ++ (n.outs ++ n.ins)) ns := by
      intro l hl; rw [← List.mem_append]; exact hperm.subset hl
    have hS : ∀ x ∈ n.outs ++ n.ins, x ∉ firstOf int (pre ++ (n.outs ++ n.ins)) ns := by
      intro x hx hx'
      have h1 := (List.mem_filter.1 hx').2
      have h2 : (pre ++ (n.outs ++ n.ins)).contains x = true := by simp [hx]
      rw [h2] at h1; simp at h1
    rw [sumOver_perm _ hperm, sumOver_append, exDen_sums]
    unfold labSums
    refine sumOver_sumsV ρ _ _ _ (fun l hl => hD l (List.mem_filter.1 hl).1) _ _ σ
      fun τ hτ hr' => ?_
    have hlab : ∀ l ∈ labelsOf (n :: ns), l ∉ firstOf int pre (n :: ns) →
        l ∉ firstAt int pre n → τ l < labDim E l := by
      intro l hl hn' hF
      rw [hτ l hF]; exact hσ l hl hn'
    have hprod : ∀ τ', prodVal E (n :: ns) τ' = val E τ' n * prodVal E ns τ' := fun τ' => by
      simp [prodVal]
    rw [sumOver_congr _ _ _ (fun τ' => val E τ' n * prodVal E ns τ') τ (fun τ' _ _ => hprod τ'),
      sumOver_const_mul _ _ (n.outs ++ n.ins) _ _ (val_reads E n) hS τ]
    rw [hr b hb τ (fun l hl hl' => by
      by_cases hF : l ∈ firstAt int pre n
      · exact hr' l hF
      · refine hlab l (by simp [labelsOf] at hl ⊢; tauto) (fun h' => ?_) hF
        rcases hmem l h' with h1 | h1
        · exact hF h1
        · exact hl' h1)]
    rw [exDen_mul ρ hρ a b hok, ← val_trG E ρ hρ dt pt rl (OFF + i) n a hnode hph τ _
      (fun h' => by have := hlt _ h'; omega) (fun p hp => ⟨?_, rfl⟩) ha]
    have hl : p.1 ∈ n.outs ++ n.ins := (List.of_mem_zip hp).1
    rw [← hdl p hp]
    by_cases hF : p.1 ∈ firstAt int pre n
    · exact hr' _ hF
    · refine hlab _ (by simp [labelsOf]; tauto) (fun h' => ?_) hF
      rcases hmem _ h' with h1 | h1
      · exact hF h1
      · exact hS _ hl h1

theorem labelsOf_blk (n1 n2 n3 : Node) (ns : List Node) (l : ℕ) :
    (labelsOf (n1 :: n2 :: n3 :: ns)).contains l = ((blkL n1 n2 n3).contains l ||
      (labelsOf ns).contains l) := by
  simp only [labelsOf, List.flatMap_cons, Node.labels, blkL, List.contains_append, Bool.or_assoc]

theorem firstOf_blk (int pre : List ℕ) (n1 n2 n3 : Node) (ns : List Node) (k k' : ℕ) :
    (firstOf int pre (n1 :: n2 :: n3 :: ns)).Perm
      (firstBlk int pre (blkL n1 n2 n3) k k' ++
        int.filter (fun l => (blkL n1 n2 n3).contains l && !pre.contains l && (l == k || l == k')) ++
        firstOf int (pre ++ blkL n1 n2 n3) ns) := by
  set L := blkL n1 n2 n3
  have hF : (int.filter fun l => L.contains l && !pre.contains l).Perm
      (firstBlk int pre L k k' ++
        int.filter (fun l => L.contains l && !pre.contains l && (l == k || l == k'))) := by
    unfold firstBlk
    have h := List.filter_append_perm (fun l => !(l == k || l == k'))
      (int.filter fun l => L.contains l && !pre.contains l)
    rw [List.filter_filter, List.filter_filter] at h
    refine h.symm.trans (List.Perm.append (List.Perm.of_eq (List.filter_congr fun l _ => ?_))
      (List.Perm.of_eq (List.filter_congr fun l _ => ?_)))
    · generalize L.contains l = A
      generalize pre.contains l = B
      generalize (l == k || l == k') = C
      cases A <;> cases B <;> cases C <;> rfl
    · generalize L.contains l = A
      generalize pre.contains l = B
      generalize (l == k || l == k') = C
      cases A <;> cases B <;> cases C <;> rfl
  have hG : (firstOf int pre (n1 :: n2 :: n3 :: ns)).Perm
      ((int.filter fun l => L.contains l && !pre.contains l) ++ firstOf int (pre ++ L) ns) := by
    unfold firstOf
    have h := List.filter_append_perm (fun l => L.contains l && !pre.contains l)
      (int.filter fun l => (labelsOf (n1 :: n2 :: n3 :: ns)).contains l && !pre.contains l)
    rw [List.filter_filter, List.filter_filter] at h
    refine h.symm.trans (List.Perm.append (List.Perm.of_eq (List.filter_congr fun l _ => ?_))
      (List.Perm.of_eq (List.filter_congr fun l _ => ?_)))
    · rw [labelsOf_blk]
      generalize L.contains l = A
      generalize pre.contains l = B
      generalize (labelsOf ns).contains l = C
      cases A <;> cases B <;> cases C <;> rfl
    · have h2 : (pre ++ L).contains l = (pre.contains l || L.contains l) := List.contains_append
      rw [labelsOf_blk, h2]
      generalize L.contains l = A
      generalize pre.contains l = B
      generalize (labelsOf ns).contains l = C
      cases A <;> cases B <;> cases C <;> rfl
  exact hG.trans (List.Perm.append_right _ hF)

set_option maxHeartbeats 1000000 in
theorem kBlock_ok (E : Env) (ρ : Val) (hρ : ρ.OK) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ)
    (hD : ∀ l ∈ int, labDim E l = (labDM dt labs l).den ρ.dv) (n1 n2 n3 : Node) (ns : List Node)
    (hn : NodesOKF E ρ dt pt rl (n1 :: n2 :: n3 :: ns)) (pre : List ℕ) (t : STm) (k k' : ℕ)
    (hf : foldBZS dt pt rl n1 n2 n3 = some (t, k, k')) (hok : foldOK int pre k k' ns = true)
    (rest : Option (List STm)) (hr : TailOK E ρ int (pre ++ blkL n1 n2 n3) ns rest)
    (e : List STm) (h : kBlock dt labs int pre (blkL n1 n2 n3) t k k' rest = some e) :
    TailOK E ρ int pre (n1 :: n2 :: n3 :: ns) (some e) := by
  intro e' he' σ hσ
  simp only [Option.some.injEq] at he'
  subst he'
  unfold kBlock at h
  cases hb : rest with
  | none => rw [hb] at h; simp at h
  | some b =>
    rw [hb] at h
    simp only at h
    split_ifs at h with hm
    simp only [Option.some.injEq] at h
    subst h
    set L := blkL n1 n2 n3 with hL
    obtain ⟨a, b', a', b'', q, ho1, hi1, ho2, hi2, ho3, hi3, -, hB, hZ, hS, hnd, hph1, hd3, hph3,
      hc1, -, -, -, -, -, -⟩ := foldBZS_spec dt pt rl n1 n2 n3 t k k' hf
    simp only [foldOK, Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_true, beq_iff_eq,
      List.contains_iff_mem] at hok
    obtain ⟨⟨⟨⟨hpk0, hpk0'⟩, hnk0⟩, hnk0'⟩, hKK⟩ := hok
    have hpk : k ∉ pre := by simpa using hpk0
    have hpk' : k' ∉ pre := by simpa using hpk0'
    have hnk : k ∉ labelsOf ns := by simpa using hnk0
    have hnk' : k' ∉ labelsOf ns := by simpa using hnk0'
    have hkL : k ∈ L := by simp [hL, blkL, ho1]
    have hk'L : k' ∈ L := by simp [hL, blkL, ho2]
    set KF := int.filter (fun l => L.contains l && !pre.contains l && (l == k || l == k'))
    have hKF : KF = int.filter (fun l => l == k || l == k') := by
      refine List.filter_congr fun l _ => ?_
      by_cases h1 : l = k
      · subst h1; simp [hkL, hpk]
      · by_cases h2 : l = k'
        · subst h2; simp [hk'L, hpk']
        · simp [h1, h2]
    have hKp : KF.Perm [k, k'] := by
      rw [hKF]
      rcases hKK with h' | h'
      · rw [h']
      · rw [h']; exact List.Perm.swap k k' []
    set R' := firstOf int (pre ++ L) ns
    set F1 := firstBlk int pre L k k'
    have hperm := (firstOf_blk int pre n1 n2 n3 ns k k').trans
      (List.Perm.append_right R' (List.Perm.append_left F1 hKp))
    rw [sumOver_perm _ hperm, List.append_assoc, sumOver_append, exDen_sums]
    unfold labSums
    refine sumOver_sumsV ρ _ _ _ (fun l hl => hD l (List.mem_filter.1 hl).1) _ _ σ
      fun τ hτ hrF => ?_
    -- labels of the block are not summed in the tail
    have hLR : ∀ x ∈ L, x ∉ R' := by
      intro x hx hx'
      have h1 := (List.mem_filter.1 hx').2
      have h2 : (pre ++ L).contains x = true := by simp [hx]
      rw [h2] at h1; simp at h1
    have hnd' := hnd
    simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd'
    obtain ⟨⟨hkk', -, -, -, -⟩, -⟩ := hnd'
    have hKKns : ∀ x ∈ labelsOf ns, x ∉ [k, k'] := by
      intro x hx hx'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx'
      rcases hx' with rfl | rfl
      · exact hnk hx
      · exact hnk' hx
    -- the block's value
    set A : (ℕ → ℕ) → ℂ := sumOver (labDim E) [k, k']
      (fun μ => val E μ n1 * (val E μ n2 * val E μ n3)) with hA
    have hAr : Reads L A := by
      intro μ μ' hμ
      rw [hA]
      refine sumOver_ext _ [k, k'] L _ ?_ μ μ' fun x hx _ => hμ x hx
      intro ν1 ν2 hν
      dsimp only
      have e1 : val E ν1 n1 = val E ν2 n1 :=
        val_reads E n1 ν1 ν2 fun x hx => hν x (by simp [hL, blkL, Node.labels] at hx ⊢; tauto)
      have e2 : val E ν1 n2 = val E ν2 n2 :=
        val_reads E n2 ν1 ν2 fun x hx => hν x (by simp [hL, blkL, Node.labels] at hx ⊢; tauto)
      have e3 : val E ν1 n3 = val E ν2 n3 :=
        val_reads E n3 ν1 ν2 fun x hx => hν x (by simp [hL, blkL, Node.labels] at hx ⊢; tauto)
      rw [e1, e2, e3]
    have hinner : ∀ μ, sumOver (labDim E) [k, k'] (prodVal E (n1 :: n2 :: n3 :: ns)) μ =
        prodVal E ns μ * A μ := by
      intro μ
      rw [hA, ← sumOver_const_mul _ _ (labelsOf ns) _ _ (prodVal_reads E ns) hKKns μ]
      refine sumOver_congr _ _ _ _ μ fun μ' _ _ => ?_
      simp only [prodVal, List.map_cons, List.prod_cons]
      ring
    rw [sumOver_perm _ (List.perm_append_comm : ([k, k'] ++ R').Perm (R' ++ [k, k'])),
      sumOver_append]
    rw [sumOver_congr _ R' _ (fun μ => A μ * prodVal E ns μ) τ (fun μ _ _ => by
      rw [hinner μ, mul_comm])]
    rw [sumOver_const_mul _ R' L _ _ hAr hLR τ]
    -- the ranges
    have hlab : ∀ l ∈ labelsOf (n1 :: n2 :: n3 :: ns), l ∉ F1 →
        l ∉ firstOf int pre (n1 :: n2 :: n3 :: ns) → τ l < labDim E l := by
      intro l hl hF hn'
      rw [hτ l hF]; exact hσ l hl hn'
    have hnin : ∀ l, l ∈ firstOf int pre (n1 :: n2 :: n3 :: ns) → l ∉ F1 → l ∉ R' →
        l ∈ [k, k'] := by
      intro l hl hF hR
      have hm := hperm.subset hl
      simp only [List.mem_append] at hm
      rcases hm with (h1 | h1) | h1
      · exact absurd h1 hF
      · exact h1
      · exact absurd h1 hR
    have hrt : ∀ l ∈ labelsOf ns, l ∉ R' → τ l < labDim E l := by
      intro l hl hR
      by_cases hF : l ∈ F1
      · exact hrF l hF
      · refine hlab l ?_ hF fun h' => hKKns l hl (hnin l h' hF hR)
        simp only [labelsOf, List.flatMap_cons, List.mem_append] at hl ⊢
        tauto
    rw [hr b hb τ hrt, exDen_mul ρ hρ _ _ hm, exDen_cons, exDen_nil, add_zero]
    congr 1
    -- the folded value at τ
    obtain ⟨hn1, hp1, -, hl1⟩ := hn n1 List.mem_cons_self
    obtain ⟨hn2, -, -, -⟩ := hn n2 (by simp)
    obtain ⟨hn3, hp3, -, hl3⟩ := hn n3 (by simp)
    have hdl1 : legDMs dt n1 = [DMon.mul (dt n1.dim) (dt n1.ph), dt n1.dim, dt n1.ph] := by
      obtain ⟨kind1, outs1, ins1, dim1, ph1⟩ := n1
      simp only at ho1 hi1 hB
      subst ho1 hi1 hB
      rfl
    have hdl3 : legDMs dt n3 = [dt n3.dim, dt n3.ph, DMon.mul (dt n3.dim) (dt n3.ph)] := by
      obtain ⟨kind3, outs3, ins3, dim3, ph3⟩ := n3
      simp only at ho3 hi3 hS
      subst ho3 hi3 hS
      rfl
    have hz1 : (n1.outs ++ n1.ins).zip (legDMs dt n1) = [(k, DMon.mul (dt n1.dim) (dt n1.ph)),
        (a, dt n1.dim), (b', dt n1.ph)] := by rw [hdl1, ho1, hi1]; rfl
    have hz3 : (n3.outs ++ n3.ins).zip (legDMs dt n3) = [(a', dt n3.dim), (b'', dt n3.ph),
        (k', DMon.mul (dt n3.dim) (dt n3.ph))] := by rw [hdl3, ho3, hi3]; rfl
    have hcd := DMon.den_isConst ρ.dv _ hc1
    have hlk : labDim E k = (dt n1.dim).c * (dt n1.dim).c := by
      rw [hl1 (k, DMon.mul (dt n1.dim) (dt n1.ph)) (by rw [hz1]; simp), DMon.den_mul, hph1, hcd]
    have hlk' : labDim E k' = (dt n1.dim).c * (dt n1.dim).c := by
      rw [hl3 (k', DMon.mul (dt n3.dim) (dt n3.ph)) (by rw [hz3]; simp), DMon.den_mul, hd3, hph3,
        hcd]
    have hla : labDim E a = (dt n1.dim).c := by
      rw [hl1 (a, dt n1.dim) (by rw [hz1]; simp), hcd]
    have hlb : labDim E b' = (dt n1.dim).c := by
      rw [hl1 (b', dt n1.ph) (by rw [hz1]; simp), hph1, hcd]
    have hla' : labDim E a' = (dt n1.dim).c := by
      rw [hl3 (a', dt n3.dim) (by rw [hz3]; simp), hd3, hcd]
    have hlb' : labDim E b'' = (dt n1.dim).c := by
      rw [hl3 (b'', dt n3.ph) (by rw [hz3]; simp), hph3, hcd]
    have hrange : ∀ l ∈ L, l ≠ k → l ≠ k' → τ l < labDim E l := by
      intro l hl hk hk'
      by_cases hF : l ∈ F1
      · exact hrF l hF
      · refine hlab l ?_ hF fun h' => ?_
        · have hc : (labelsOf (n1 :: n2 :: n3 :: ns)).contains l = true := by
            rw [labelsOf_blk]; simp only [Bool.or_eq_true, List.contains_iff_mem]; exact Or.inl hl
          simpa using hc
        · have hR : l ∉ R' := hLR l hl
          have := hnin l h' hF hR
          simp only [List.mem_cons, List.not_mem_nil, or_false] at this
          rcases this with h | h
          · exact hk h
          · exact hk' h
    have hnd2 := hnd
    simp only [List.nodup_cons, List.mem_cons, List.not_mem_nil, or_false, not_or] at hnd2
    obtain ⟨⟨_, hka, hkb, hka', hkb'⟩, ⟨hk'a, hk'b, hk'a', hk'b'⟩, -⟩ := hnd2
    refine foldBZS_val E ρ dt pt rl n1 n2 n3 t k k' hf hn1 (hp1 (by rw [hB]; rfl)) hn2 hn3
      (hp3 (by rw [hS]; rfl)) hlk hlk' τ fun l hl => ?_
    rw [hi1, ho3] at hl
    simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hl
    rcases hl with (rfl | rfl) | (rfl | rfl)
    · rw [← hla]; exact hrange _ (by simp [hL, blkL, hi1]) (Ne.symm hka) (Ne.symm hk'a)
    · rw [← hlb]; exact hrange _ (by simp [hL, blkL, hi1]) (Ne.symm hkb) (Ne.symm hk'b)
    · rw [← hla']; exact hrange _ (by simp [hL, blkL, ho3]) (Ne.symm hka') (Ne.symm hk'a')
    · rw [← hlb']; exact hrange _ (by simp [hL, blkL, ho3]) (Ne.symm hkb') (Ne.symm hk'b')

theorem trNodesK_ok (E : Env) (ρ : Val) (hρ : ρ.OK) (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon))
    (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ)) (int : List ℕ)
    (hD : ∀ l ∈ int, labDim E l = (labDM dt labs l).den ρ.dv) :
    ∀ (m : ℕ) (ns : List Node), ns.length ≤ m → NodesOKF E ρ dt pt rl ns → ∀ (pre : List ℕ) (i : ℕ),
      TailOK E ρ int pre ns (trNodesK dt pt rl labs int pre i ns) := by
  intro m
  induction m with
  | zero =>
    intro ns hl _ pre i
    rcases ns with _ | ⟨n, ns⟩
    · intro b hb σ _
      simp only [trNodesK, Option.some.injEq] at hb
      subst hb
      simp [firstOf, labelsOf, sumOver, prodVal, exDen_cons, exDen_nil, den_zero, ofQ_den, bodyOf]
    · simp at hl
  | succ m ih =>
    intro ns hl hn pre i
    rcases ns with _ | ⟨n, ns⟩
    · intro b hb σ _
      simp only [trNodesK, Option.some.injEq] at hb
      subst hb
      simp [firstOf, labelsOf, sumOver, prodVal, exDen_cons, exDen_nil, den_zero, ofQ_den, bodyOf]
    · have hn' : NodesOKF E ρ dt pt rl ns := fun x hx => hn x (List.mem_cons_of_mem _ hx)
      have hls : ns.length ≤ m := by simp at hl; omega
      have single : ∀ e, kSingle dt pt rl labs int pre i n
          (trNodesK dt pt rl labs int (pre ++ (n.outs ++ n.ins)) (i + 1) ns) = some e →
          TailOK E ρ int pre (n :: ns) (some e) := fun e he =>
        kSingle_ok E ρ hρ dt pt rl labs int hD n ns hn pre i _
          (ih ns hls hn' (pre ++ (n.outs ++ n.ins)) (i + 1)) e he
      intro e he
      rcases ns with _ | ⟨n2, _ | ⟨n3, ns'⟩⟩
      · exact single e (by simpa [trNodesK] using he) e rfl
      · exact single e (by simpa [trNodesK] using he) e rfl
      · rw [trNodesK] at he
        split at he
        · next t k k' hf =>
          split_ifs at he with hok
          · have hl3 : ns'.length ≤ m := by simp at hls; omega
            have hn3 : NodesOKF E ρ dt pt rl ns' :=
              fun x hx => hn x (by simp [hx])
            exact kBlock_ok E ρ hρ dt pt rl labs int hD n n2 n3 ns' hn pre t k k' hf hok _
              (ih ns' hl3 hn3 (pre ++ blkL n n2 n3) (i + 3)) e he e rfl
          · exact single e he e rfl
        · exact single e he e rfl

/-- The network's entry over its boundary labels through `trNodesK`. -/
def trNetK (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (N : Net) : Option (List STm) :=
  (trNodesK dt pt rl labs N.int [] 0 N.nodes).map
    (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)))

theorem eval_trK (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N : Net) (e : List STm) (hc : netChk fl dt pt rl labs IX N = true)
    (ht : trNetK dt pt rl labs N = some e) (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = exDen ρ (fun x => (σ x : ℤ)) e := by
  unfold trNetK at ht
  cases h0 : trNodesK dt pt rl labs N.int [] 0 N.nodes with
  | none => rw [h0] at ht; simp at ht
  | some e0 =>
    rw [h0] at ht
    simp only [Option.map_some, Option.some.injEq] at ht
    subst ht
    rw [exDen_sums]
    obtain ⟨hc, hci⟩ := netChk_ok E ρ fl hfl dt pt rl labs IX hE hP hL N hc
    set rest := N.int.filter fun l => !(labelsOf N.nodes).contains l
    have hfirst : firstOf N.int [] N.nodes = N.int.filter fun l => (labelsOf N.nodes).contains l := by
      unfold firstOf; exact List.filter_congr fun l _ => by simp
    have hperm : N.int.Perm (rest ++ firstOf N.int [] N.nodes) := by
      rw [hfirst]
      exact (List.filter_append_perm _ N.int).symm.trans List.perm_append_comm
    unfold eval labSums
    rw [sumOver_perm _ hperm, sumOver_append]
    refine sumOver_sumsV ρ _ _ _ (fun l hl => hci l (List.mem_filter.1 hl).1) _ _ σ
      fun τ hτ hr => ?_
    refine trNodesK_ok E ρ hρ dt pt rl labs N.int hci N.nodes.length N.nodes le_rfl
      (fun n hn => ⟨(hc n hn).1, (hc n hn).2.1, fun l hl => ((hc n hn).2.2.1 l hl).1,
        (hc n hn).2.2.2⟩) [] 0 e0 h0 τ fun l hl hl' => ?_
    obtain ⟨n, hnm, hln⟩ := List.mem_flatMap.1 hl
    have hbi := ((hc n hnm).2.2.1 l hln).2
    have hni : l ∉ N.int := fun hi => hl' (by
      rw [hfirst]; exact List.mem_filter.2 ⟨hi, by simpa using hl⟩)
    have hnr : l ∉ rest := fun h' => hni (List.mem_filter.1 h').1
    rcases hbi with hb | hi
    · rw [hτ l hnr]; exact hσ l hb
    · exact absurd hi hni

/-- `equal_symSR` through `trNetK`. -/
theorem equal_symSK (E : Env) (ρ : Val) (hρ : ρ.OK) (fl : List ℕ) (hfl : FloorOK fl ρ)
    (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ) (labs : List (ℕ × ℕ))
    (IX : List ℕ) (hE : ∀ i ∈ IX, E.dimv i = (dt i).den ρ.dv)
    (hP : ∀ ph j, ex (E.phase ph j) = phF ρ (pt ph) j * rV ρ (rl ph) j)
    (hL : E.labs = labs) (N0 N1 : Net) (S0 S1 : List STm) (s0 s1 : ℂ)
    (hs0 : s0 = exDen ρ (fun _ => 0) S0) (hs1 : s1 = exDen ρ (fun _ => 0) S1) (e0 e1 : List STm)
    (h0 : trNetK dt pt rl labs N0 = some e0) (h1 : trNetK dt pt rl labs N1 = some e1)
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
    ← eval_trK E ρ hρ fl hfl dt pt rl labs IX hE hP hL N0 e0 hc0 h0 σ hσ,
    ← eval_trK E ρ hρ fl hfl dt pt rl labs IX hE hP hL N1 e1 hc1 h1 σ (hσ.perm hp'),
    exDen_const ρ S0 hk0 ν (fun _ => 0), exDen_const ρ S1 hk1 ν (fun _ => 0), ← hs0,
    ← hs1] at hz'
  linear_combination hz'

/-! ## Rounds in chunks -/

/-- `elimStepF` along the hints, each result forced; `none` at the first that fails. -/
def elimAllFS (R : List (ℕ × ℕ)) : List (ℕ × ℕ × ℕ) → STm → Option STm
  | [], t => some t
  | (i, r, j) :: hs, t =>
    match elimStepF R t i r j with
    | some t' => fTm t' (elimAllFS R hs)
    | none => none

theorem elimAllF_append (R : List (ℕ × ℕ)) :
    ∀ (es1 es2 : List (ℕ × ℕ × ℕ)) (t u : STm), elimAllFS R es1 t = some u →
      elimAllF R (es1 ++ es2) t = elimAllF R es2 u := by
  intro es1
  induction es1 with
  | nil => intro es2 t u h; simp only [elimAllFS, Option.some.injEq] at h; subst h; rfl
  | cons s es1 ih =>
    obtain ⟨i, r, j⟩ := s
    intro es2 t u h
    simp only [elimAllFS] at h
    simp only [List.cons_append, elimAllF]
    cases he : elimStepF R t i r j with
    | none => simp [he] at h
    | some t' =>
      simp only [he, fTm_eq] at h ⊢
      exact ih es2 t' u h

theorem elimAllF_of_S (R : List (ℕ × ℕ)) (es : List (ℕ × ℕ × ℕ)) (t u : STm)
    (h : elimAllFS R es t = some u) : elimAllF R es t = u := by
  have := elimAllF_append R es [] t u h
  rw [List.append_nil] at this
  rw [this]; rfl

theorem normTmF_one (R : List (ℕ × ℕ)) (fl : List ℕ) (c : ℕ) (es : List (ℕ × ℕ × ℕ))
    (t u : STm) (h : elimAllF R es t = u) : normTmF R fl [(c, es)] t = canonTmH c fl u := by
  simp only [normTmF, fTm_eq, h]
  cases canonTmH c fl u <;> simp

/-! ## Translation with folded blocks on label masks -/

/-- `foldOK` with the labels before the block as the bit mask `pm`. -/
def foldOKM (int : List ℕ) (pm k k' : ℕ) (ns : List Node) : Bool :=
  !pm.testBit k && !pm.testBit k' && !(labelsOf ns).contains k &&
    !(labelsOf ns).contains k' &&
    ((int.filter fun l => l == k || l == k') == [k, k'] ||
      (int.filter fun l => l == k || l == k') == [k', k])

/-- `kSingle` with the labels before the node as the bit mask `pm`. -/
def kSingleM (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int : List ℕ) (pm i : ℕ) (n : Node) (rest : Option (List STm)) :
    Option (List STm) :=
  match trNode dt pt rl (OFF + i) n, rest with
  | some a, some b =>
    if exMulOK a b then
      some (exSums (labSums dt labs
        (int.filter fun l => (n.outs ++ n.ins).contains l && !pm.testBit l)) (exMul a b))
    else none
  | _, _ => none

/-- `kBlock` with the labels before the block as the bit mask `pm`. -/
def kBlockM (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (int L : List ℕ) (pm : ℕ) (t : STm) (k k' : ℕ)
    (rest : Option (List STm)) : Option (List STm) :=
  match rest with
  | some b =>
    if exMulOK [t] b then
      some (exSums (labSums dt labs
        (int.filter fun l => L.contains l && !pm.testBit l && !(l == k || l == k'))) (exMul [t] b))
    else none
  | none => none

/-- `trNodesK` with the labels before the nodes as the bit mask `pm`. -/
def trNodesKM (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int : List ℕ) : ℕ → ℕ → List Node → Option (List STm)
  | _, _, [] => some [⟨CMon.ofQ 1, [], [], []⟩]
  | pm, i, n :: ns =>
    match ns with
    | n2 :: n3 :: ns' =>
      match foldBZS dt pt rl n n2 n3 with
      | some (t, k, k') =>
        if foldOKM int pm k k' ns' = true then
          kBlockM dt labs int (blkL n n2 n3) pm t k k'
            (trNodesKM dt pt rl labs int (pm ||| maskOf (blkL n n2 n3)) (i + 3) ns')
        else kSingleM dt pt rl labs int pm i n
          (trNodesKM dt pt rl labs int (pm ||| maskOf (n.outs ++ n.ins)) (i + 1) (n2 :: n3 :: ns'))
      | none => kSingleM dt pt rl labs int pm i n
          (trNodesKM dt pt rl labs int (pm ||| maskOf (n.outs ++ n.ins)) (i + 1) (n2 :: n3 :: ns'))
    | ns => kSingleM dt pt rl labs int pm i n
        (trNodesKM dt pt rl labs int (pm ||| maskOf (n.outs ++ n.ins)) (i + 1) ns)
termination_by structural _ _ ns => ns

theorem foldOKM_eq (int pre : List ℕ) (k k' : ℕ) (ns : List Node) :
    foldOKM int (maskOf pre) k k' ns = foldOK int pre k k' ns := by
  simp only [foldOKM, foldOK, testBit_maskOf]

theorem kSingleM_eq (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int pre : List ℕ) (i : ℕ) (n : Node) (rest : Option (List STm)) :
    kSingleM dt pt rl labs int (maskOf pre) i n rest = kSingle dt pt rl labs int pre i n rest := by
  unfold kSingleM kSingle firstAt
  simp only [testBit_maskOf]

theorem kBlockM_eq (dt : ℕ → DMon) (labs : List (ℕ × ℕ)) (int pre L : List ℕ) (t : STm)
    (k k' : ℕ) (rest : Option (List STm)) :
    kBlockM dt labs int L (maskOf pre) t k k' rest = kBlock dt labs int pre L t k k' rest := by
  unfold kBlockM kBlock firstBlk
  simp only [testBit_maskOf]

theorem trNodesKM_eq (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (int : List ℕ) : ∀ (m : ℕ) (ns : List Node), ns.length ≤ m →
    ∀ (pre : List ℕ) (i : ℕ),
      trNodesKM dt pt rl labs int (maskOf pre) i ns = trNodesK dt pt rl labs int pre i ns := by
  intro m
  induction m with
  | zero =>
    intro ns hl pre i
    match ns, hl with
    | [], _ => rw [trNodesKM.eq_1, trNodesK.eq_1]
    | _ :: _, hl => simp at hl
  | succ m ih =>
    intro ns hl pre i
    match ns, hl with
    | [], _ => rw [trNodesKM.eq_1, trNodesK.eq_1]
    | [n], _ => simp only [trNodesKM, trNodesK, kSingleM_eq]
    | [n, n2], _ => simp only [trNodesKM, trNodesK, ← maskOf_append, kSingleM_eq]
    | n :: n2 :: n3 :: ns', hl =>
      rw [trNodesKM.eq_2, trNodesK.eq_2]
      rw [← maskOf_append, ← maskOf_append, ih (n2 :: n3 :: ns') (by simp at hl ⊢; omega),
        ih ns' (by simp at hl ⊢; omega)]
      cases foldBZS dt pt rl n n2 n3 with
      | none => simp only [kSingleM_eq]
      | some v =>
        obtain ⟨t, k, k'⟩ := v
        simp only [foldOKM_eq, kSingleM_eq, kBlockM_eq]

/-- `trNetK` through `trNodesKM`. -/
def trNetKM (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (N : Net) : Option (List STm) :=
  (trNodesKM dt pt rl labs N.int 0 0 N.nodes).map
    (exSums (labSums dt labs (N.int.filter fun l => !(labelsOf N.nodes).contains l)))

theorem trNetKM_eq (dt : ℕ → DMon) (pt : ℕ → List (ℕ × CMon)) (rl : ℕ → DMon × List ℤ)
    (labs : List (ℕ × ℕ)) (N : Net) : trNetKM dt pt rl labs N = trNetK dt pt rl labs N := by
  unfold trNetKM trNetK
  rw [← trNodesKM_eq dt pt rl labs N.int N.nodes.length N.nodes le_rfl [] 0]
  rfl

end ArchytasZX.Sym
