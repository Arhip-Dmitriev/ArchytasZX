/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Sym
import ArchytasZX.Bundle

/-!
# The normaliser of symbolic sums

Each move removes one bound variable of a term: an unused one, one pinned by a delta whose
modulus divides its range, one read only through roots of unity of its range (the character
sum), or one read only through one delta of its range (a gcd). Canonicalisation substitutes
the pivot of each delta into the factors it determines, reduces and merges factors and sorts
them. `nfEx` normalises every term and merges terms of equal shape; `nfEx_den` proves it
keeps the value.
-/

namespace ArchytasZX.Sym

open Complex Finset

/-! ## Sums over a range -/

theorem dvd_sub_iff_eq (M : ℕ) (hM : 0 < M) (b : ℕ) (hb : b < M) (v : ℤ) :
    (M : ℤ) ∣ (b : ℤ) - v ↔ b = (v % M).toNat := by
  have h0 : 0 ≤ v % M := Int.emod_nonneg _ (by exact_mod_cast hM.ne')
  have h1 : v % M < M := Int.emod_lt_of_pos _ (by exact_mod_cast hM)
  rw [← Int.modEq_iff_dvd]
  unfold Int.ModEq
  rw [Int.emod_eq_of_lt (a := (b : ℤ)) (by positivity) (by exact_mod_cast hb)]
  omega

/-- A delta pinning a periodic summand. -/
theorem sum_delta_periodic (D M : ℕ) (hM : 0 < M) (hMD : M ∣ D) (v : ℤ) (g : ℤ → ℂ)
    (hg : ∀ a b : ℤ, (M : ℤ) ∣ a - b → g a = g b) :
    ∑ j ∈ range D, (if (M : ℤ) ∣ (j : ℤ) - v then g j else 0) = ((D / M : ℕ) : ℂ) * g v := by
  obtain ⟨q, rfl⟩ := hMD
  rw [sum_range_mul M q, Nat.mul_div_cancel_left _ hM]
  have : ∀ b ∈ range M, ∑ y ∈ range q,
      (if (M : ℤ) ∣ ((b + M * y : ℕ) : ℤ) - v then g ((b + M * y : ℕ) : ℤ) else 0) =
      if b = (v % M).toNat then (q : ℂ) * g v else 0 := by
    intro b hb
    have hb' := Finset.mem_range.1 hb
    have key : ∀ y : ℕ, ((M : ℤ) ∣ ((b + M * y : ℕ) : ℤ) - v ↔ (M : ℤ) ∣ (b : ℤ) - v) := by
      intro y
      have : ((b + M * y : ℕ) : ℤ) - v = ((b : ℤ) - v) + M * y := by push_cast; ring
      rw [this, dvd_add_left (dvd_mul_right _ _)]
    split_ifs with h
    · have hd : (M : ℤ) ∣ (b : ℤ) - v := (dvd_sub_iff_eq M hM b hb' v).2 h
      rw [Finset.sum_congr rfl fun y _ => by rw [ite_cond_eq_true _ _ (eq_true ((key y).2 hd))]]
      rw [Finset.sum_congr rfl fun y _ => hg _ v (by
        have : ((b + M * y : ℕ) : ℤ) - v = ((b : ℤ) - v) + M * y := by push_cast; ring
        rw [this]; exact dvd_add hd (dvd_mul_right _ _))]
      simp
    · have hd : ¬ (M : ℤ) ∣ (b : ℤ) - v := fun h' => h ((dvd_sub_iff_eq M hM b hb' v).1 h')
      exact Finset.sum_eq_zero fun y _ => by
        rw [ite_cond_eq_false _ _ (eq_false (fun h' => hd ((key y).1 h')))]
  rw [Finset.sum_congr rfl this, Finset.sum_ite_eq']
  have hlt : (v % M).toNat < M := by
    have h0 : 0 ≤ v % M := Int.emod_nonneg _ (by exact_mod_cast hM.ne')
    have h1 : v % M < M := Int.emod_lt_of_pos _ (by exact_mod_cast hM)
    omega
  simp [hlt]

/-- The character sum with integer exponents. -/
theorem sum_om_lin (D : ℕ) (hD : 0 < D) (A B : ℤ) :
    ∑ j ∈ range D, om D ^ ((j : ℤ) * A + B) = om D ^ B * (if (D : ℤ) ∣ A then (D : ℂ) else 0) := by
  have hp := om_prim D hD
  have hz : om D ≠ 0 := hp.ne_zero hD.ne'
  set u := om D ^ A
  have hterm : ∀ j : ℕ, om D ^ ((j : ℤ) * A + B) = om D ^ B * u ^ j := by
    intro j
    rw [zpow_add₀ hz, mul_comm, ← zpow_natCast, ← zpow_mul, mul_comm A]
  rw [Finset.sum_congr rfl fun j _ => hterm j, ← Finset.mul_sum]
  congr 1
  split_ifs with h
  · have : u = 1 := (hp.zpow_eq_one_iff_dvd A).2 h
    simp [this]
  · have hu : u ≠ 1 := fun h1 => h ((hp.zpow_eq_one_iff_dvd A).1 h1)
    have hud : u ^ D = 1 := by
      simp only [u, ← zpow_natCast, ← zpow_mul]
      rw [mul_comm, zpow_mul, zpow_natCast, hp.pow_eq_one, one_zpow]
    rw [geom_sum_eq hu, hud]; simp

/-- The number of solutions of `a j + r ≡ 0 mod D` with `a ∣ r`. -/
theorem sum_dvd_lin (D : ℕ) (hD : 0 < D) (a : ℤ) (r : ℤ) (hr : a ∣ r) :
    ∑ j ∈ range D, (if (D : ℤ) ∣ a * j + r then (1 : ℂ) else 0) = (Nat.gcd a.natAbs D : ℂ) := by
  obtain ⟨r', rfl⟩ := hr
  set g := Nat.gcd a.natAbs D
  have hg : 0 < g := Nat.gcd_pos_of_pos_right _ hD
  have hgD : g ∣ D := Nat.gcd_dvd_right _ _
  obtain ⟨D', hD'⟩ := hgD
  have hga : (g : ℤ) ∣ a := Int.natCast_dvd.2 (Nat.gcd_dvd_left _ _)
  obtain ⟨a', ha'⟩ := hga
  have hD'pos : 0 < D' := by
    rcases Nat.eq_zero_or_pos D' with h | h
    · rw [h, mul_zero] at hD'; omega
    · exact h
  have hcop : IsCoprime a' (D' : ℤ) := by
    rw [Int.isCoprime_iff_gcd_eq_one]
    have h1 : Int.gcd a D = g := by simp [Int.gcd, g]
    have h2 := Int.gcd_div_gcd_div_gcd (i := a) (j := D) (by rw [h1]; exact_mod_cast hg)
    rw [h1] at h2
    have e1 : a / (g : ℤ) = a' := by rw [ha']; exact Int.mul_ediv_cancel_left _ (by exact_mod_cast hg.ne')
    have e2 : (D : ℤ) / (g : ℤ) = D' := by
      rw [hD']; push_cast; exact Int.mul_ediv_cancel_left _ (by exact_mod_cast hg.ne')
    rwa [e1, e2] at h2
  have key : ∀ j : ℕ, ((D : ℤ) ∣ a * j + a * r' ↔ (D' : ℤ) ∣ (j : ℤ) - (-r')) := by
    intro j
    have e : a * j + a * r' = (g : ℤ) * (a' * ((j : ℤ) - (-r'))) := by rw [ha']; ring
    rw [e, hD']; push_cast
    rw [mul_dvd_mul_iff_left (by exact_mod_cast hg.ne')]
    exact ⟨fun h => hcop.symm.dvd_of_dvd_mul_left h, fun h => dvd_mul_of_dvd_right h _⟩
  rw [Finset.sum_congr rfl fun j _ => by rw [if_congr (key j) rfl rfl]]
  have := sum_delta_periodic D D' hD'pos ⟨g, by rw [hD', mul_comm]⟩ (-r') (fun _ => 1)
    (fun _ _ _ => rfl)
  rw [this, hD', Nat.mul_div_cancel _ hD'pos, mul_one]

theorem sum_const_range (D : ℕ) (c : ℂ) : ∑ _j ∈ range D, c = (D : ℂ) * c := by simp

/-! ## Bodies -/

/-- The product of deltas and roots of unity. -/
noncomputable def bodyOf (dv : ℕ → ℕ) (ds ps : List Fac) (ν : ℕ → ℤ) : ℂ :=
  (ds.map (delV dv ν)).prod * (ps.map (phV dv ν)).prod

theorem body_eq (dv : ℕ → ℕ) (t : STm) : body dv t = bodyOf dv t.dels t.phs := rfl

/-- Every way to cut a list at one element. -/
def splits {α : Type} : List α → List (List α × α × List α)
  | [] => []
  | x :: l => ([], x, l) :: (splits l).map fun p => (x :: p.1, p.2.1, p.2.2)

theorem splits_spec {α : Type} (l : List α) (p : List α × α × List α) (h : p ∈ splits l) :
    l = p.1 ++ p.2.1 :: p.2.2 := by
  induction l generalizing p with
  | nil => simp [splits] at h
  | cons x l ih =>
    simp only [splits, List.mem_cons, List.mem_map] at h
    rcases h with rfl | ⟨q, hq, rfl⟩
    · rfl
    · simp [ih q hq]

/-- The first value `f` gives on the list, if any. -/
def firstSome {α β : Type} (f : α → Option β) : List α → Option β
  | [] => none
  | x :: l => match f x with
    | some b => some b
    | none => firstSome f l

theorem firstSome_spec {α β : Type} (f : α → Option β) (l : List α) (b : β)
    (h : firstSome f l = some b) : ∃ x ∈ l, f x = some b := by
  induction l with
  | nil => simp [firstSome] at h
  | cons x l ih =>
    simp only [firstSome] at h
    split at h
    · next b' hb => exact ⟨x, List.mem_cons_self, by rw [hb, h]⟩
    · obtain ⟨y, hy, hfy⟩ := ih h
      exact ⟨y, List.mem_cons_of_mem _ hy, hfy⟩

/-- `x` substituted by `v` in a factor. -/
def facSub (x : ℕ) (v : IPoly) (f : Fac) : Fac := (f.1, ipSubst x v f.2)

/-- Whether the factor reads `x`. -/
def facHas (x : ℕ) (f : Fac) : Bool := ipHas x f.2

theorem facVars_has (x : ℕ) (f : Fac) (h : facHas x f = false) : x ∉ facVars f := by
  simp only [facHas, ipHas, List.any_eq_false, Bool.not_eq_true] at h
  simp only [facVars, List.mem_flatMap, not_exists, not_and]
  intro t ht hx
  have := h t ht
  simp at this
  exact this hx

/-- The value a sign `s = ±1` makes `s x + b ≡ 0` pin `x` to. -/
def pinVal (s : ℤ) (b : IPoly) : IPoly := ipSmul (-s) b

theorem pin_dvd (s : ℤ) (hs : s = 1 ∨ s = -1) (M : ℤ) (x b : ℤ) :
    M ∣ s * x + b ↔ M ∣ x - (-s * b) := by
  have e : x - (-s * b) = s * (s * x + b) := by rcases hs with rfl | rfl <;> ring
  rw [e]
  constructor
  · exact fun h => dvd_mul_of_dvd_right h _
  · intro h
    have e2 : s * x + b = s * (s * (s * x + b)) := by rcases hs with rfl | rfl <;> ring
    rw [e2]; exact dvd_mul_of_dvd_right h _

/-- A delta factor is `0` or `1`. -/
theorem delV_cases (dv : ℕ → ℕ) (ν : ℕ → ℤ) (f : Fac) : delV dv ν f = 0 ∨ delV dv ν f = 1 := by
  unfold delV; split_ifs <;> simp

/-- A factor whose modulus divides `M`, at two values of `x` congruent modulo `M`. -/
theorem fac_congr (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (x : ℕ) (M : DMon) (f : Fac)
    (hf : f.1.dvd M = true) (ν : ℕ → ℤ) (a b : ℤ) (hab : ((M.den dv : ℕ) : ℤ) ∣ a - b) :
    delV dv (Function.update ν x a) f = delV dv (Function.update ν x b) f ∧
      phV dv (Function.update ν x a) f = phV dv (Function.update ν x b) f := by
  have hN : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen (Function.update ν x a) f.2 -
      ipDen (Function.update ν x b) f.2 :=
    (Int.natCast_dvd_natCast.2 (DMon.dvd_den dv _ _ hf)).trans (hab.trans (ipDen_upd_dvd ν x a b f.2))
  constructor
  · unfold delV
    have : ((f.1.den dv : ℕ) : ℤ) ∣ ipDen (Function.update ν x a) f.2 ↔
        ((f.1.den dv : ℕ) : ℤ) ∣ ipDen (Function.update ν x b) f.2 := by
      constructor
      · intro h; have := dvd_sub h hN; simpa using this
      · intro h; have := dvd_add h hN; simpa using this
    simp only [this]
  · unfold phV
    have hpos : 0 < f.1.den dv := by
      unfold DMon.dvd at hf
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hf
      exact DMon.den_pos dv hdv _ (Nat.pos_of_ne_zero hf.1.1)
    obtain ⟨k, hk⟩ := hN
    have hz := om_ne_zero _ hpos
    rw [show ipDen (Function.update ν x a) f.2 = ipDen (Function.update ν x b) f.2 +
      (f.1.den dv : ℤ) * k by omega, zpow_add₀ hz, zpow_mul, zpow_natCast, om_pow_self _ hpos,
      one_zpow, mul_one]

theorem bodyOf_upd (dv : ℕ → ℕ) (x : ℕ) (ds ps : List Fac)
    (h : (ds ++ ps).all (fun f => !facHas x f) = true) (ν : ℕ → ℤ) (v : ℤ) :
    bodyOf dv ds ps (Function.update ν x v) = bodyOf dv ds ps ν := by
  simp only [List.all_eq_true, Bool.not_eq_true', List.mem_append] at h
  unfold bodyOf delV phV
  congr 1
  · congr 1; refine List.map_congr_left fun f hf => ?_
    rw [ipDen_upd_of _ _ _ _ (h f (Or.inl hf))]
  · congr 1; refine List.map_congr_left fun f hf => ?_
    rw [ipDen_upd_of _ _ _ _ (h f (Or.inr hf))]

theorem bodyOf_append (dv : ℕ → ℕ) (a b ps : List Fac) (ν : ℕ → ℤ) :
    bodyOf dv (a ++ b) ps ν = (a.map (delV dv ν)).prod * bodyOf dv b ps ν := by
  simp only [bodyOf, List.map_append, List.prod_append]; ring

theorem bodyOf_sub (dv : ℕ → ℕ) (x : ℕ) (v : IPoly) (ds ps : List Fac) (ν : ℕ → ℤ) :
    bodyOf dv (ds.map (facSub x v)) (ps.map (facSub x v)) ν =
      bodyOf dv ds ps (Function.update ν x (ipDen ν v)) := by
  simp only [bodyOf, List.map_map]
  congr 1
  · congr 1; refine List.map_congr_left fun f _ => ?_
    simp [delV, facSub, ipDen_subst]
  · congr 1; refine List.map_congr_left fun f _ => ?_
    simp [phV, facSub, ipDen_subst]

/-! ## Eliminating a bound variable -/

/-- The coefficient an elimination gains and the factors it leaves. -/
structure Elim where
  k : CMon
  ds : List Fac
  ps : List Fac

/-- Summing `x` over `D` out of the factors `ds`, `ps` gives `e`. -/
def ElimOK (ρ : Val) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim) : Prop :=
  ∀ ν, ∑ j ∈ range (D.den ρ.dv), bodyOf ρ.dv ds ps (Function.update ν x j) =
    e.k.den ρ * bodyOf ρ.dv e.ds e.ps ν

/-- A variable no factor reads. -/
def elimUnused (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  if D.c != 0 && (ds ++ ps).all (fun f => !facHas x f) then some ⟨CMon.rtD D 2, ds, ps⟩
  else none

/-- `x` pinned by the delta `f = [M ∣ s x + b]`, `s = ±1`, `M ∣ D`, the other factors reading
`x` periodic modulo `M`. -/
def elimDelta1 (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option Elim :=
  match ipLin x f.2 with
  | some (a, b) =>
    match ipConst a with
    | some s =>
      if (s == 1 || s == -1) && D.c != 0 && f.1.dvd D &&
          (pre ++ post ++ ps).all (fun g => !facHas x g || g.1.dvd f.1) then
        some ⟨CMon.rtD (D.div f.1) 2, (pre ++ post).map (facSub x (pinVal s b)),
          ps.map (facSub x (pinVal s b))⟩
      else none
    | none => none
  | none => none

/-- `x` pinned by the first delta that pins it. -/
def elimDelta (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  firstSome (fun p => elimDelta1 x D p.1 p.2.1 p.2.2 ps) (splits ds)

/-- The roots of unity reading `x`, each `ω_D^(a x + b)`: the sums of the `a` and of the `b`. -/
def charData (x : ℕ) (D : DMon) : List Fac → Option (IPoly × IPoly)
  | [] => some ([], [])
  | f :: l =>
    match charData x D l with
    | none => none
    | some (A, B) =>
      if facHas x f then
        if f.1 = D then
          match ipLin x f.2 with
          | some (a, b) => some (ipAdd A a, ipAdd B b)
          | none => none
        else none
      else some (A, B)

/-- `x` read only by roots of unity of its range, linearly. -/
def elimChar (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  if D.c != 0 && ds.all (fun f => !facHas x f) then
    match charData x D ps with
    | some (A, B) =>
      some ⟨CMon.rtD D 2, ds ++ [(D, A)], ps.filter (fun f => !facHas x f) ++ [(D, B)]⟩
    | none => none
  else none

/-- `x` read only by the delta `f = [D ∣ α x + b]` with `α ∣ b`. -/
def elimGcd1 (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac) (post ps : List Fac) :
    Option Elim :=
  if f.1 = D ∧ D.c ≠ 0 ∧ (pre ++ post ++ ps).all (fun g => !facHas x g) = true then
    match ipLin x f.2 with
    | some (a, b) =>
      match ipConst a with
      | some α => if ipDivBy α b then some ⟨CMon.gcdA α D, pre ++ post, ps⟩ else none
      | none => none
    | none => none
  else none

/-- `x` read only by one delta of its range. -/
def elimGcd (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  firstSome (fun p => elimGcd1 x D p.1 p.2.1 p.2.2 ps) (splits ds)

/-- The first elimination of `x` that applies. -/
def elimVar (x : ℕ) (D : DMon) (ds ps : List Fac) : Option Elim :=
  match elimUnused x D ds ps with
  | some e => some e
  | none =>
    match elimDelta x D ds ps with
    | some e => some e
    | none =>
      match elimChar x D ds ps with
      | some e => some e
      | none => elimGcd x D ds ps

theorem rtD_two (ρ : Val) (D : DMon) (hc : 0 < D.c) :
    (CMon.rtD D 2).den ρ = ((D.den ρ.dv : ℕ) : ℂ) := by
  rw [rtD_den ρ D hc, zpow_ofNat, rt_sq']

theorem elimUnused_ok (ρ : Val) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimUnused x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  unfold elimUnused at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
  intro ν
  simp only [bodyOf_upd ρ.dv x ds ps hc.2, sum_const_range]
  rw [rtD_two ρ D (Nat.pos_of_ne_zero hc.1)]

theorem ipDivBy_spec (k : ℤ) (p : IPoly) (h : ipDivBy k p = true) (ν : ℕ → ℤ) : k ∣ ipDen ν p := by
  induction p with
  | nil => simp [ipDen]
  | cons t p ih =>
    simp only [ipDivBy, List.all_cons, Bool.and_eq_true, beq_iff_eq] at h ih
    obtain ⟨m, c⟩ := t
    rw [ipDen_cons]
    exact dvd_add (Dvd.dvd.mul_right (Int.dvd_of_emod_eq_zero h.1) _) (ih h.2)

theorem elimDelta1_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac)
    (post ps : List Fac) (e : Elim) (h : elimDelta1 x D pre f post ps = some e) :
    ElimOK ρ x D (pre ++ f :: post) ps e := by
  unfold elimDelta1 at h
  split at h
  · next a b hlin =>
    split at h
    · next s hs =>
      split_ifs at h with hc
      simp only [Option.some.injEq] at h
      subst h
      simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, bne_iff_ne, ne_eq,
        List.all_eq_true, Bool.not_eq_true'] at hc
      obtain ⟨⟨⟨hs1, hD⟩, hdvd⟩, hper⟩ := hc
      obtain ⟨ha, hb, hp⟩ := ipLin_spec x f.2 a b hlin
      have hsv : ∀ ν, ipDen ν a = s := fun ν => ipConst_spec a s hs ν
      have hMpos : 0 < f.1.den ρ.dv := by
        unfold DMon.dvd at hdvd
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hdvd
        exact DMon.den_pos _ hρ.dpos _ (Nat.pos_of_ne_zero hdvd.1.1)
      have hdD := DMon.den_div ρ.dv _ _ hdvd
      intro ν
      set v := ipDen ν (pinVal s b)
      have hv : v = -s * ipDen ν b := by simp [v, pinVal, ipDen_smul]
      set g : ℤ → ℂ := fun j => bodyOf ρ.dv (pre ++ post) ps (Function.update ν x j)
      have hsplit : ∀ j : ℤ, bodyOf ρ.dv (pre ++ f :: post) ps (Function.update ν x j) =
          (if ((f.1.den ρ.dv : ℕ) : ℤ) ∣ j - v then g j else 0) := by
        intro j
        have hd : delV ρ.dv (Function.update ν x j) f =
            if ((f.1.den ρ.dv : ℕ) : ℤ) ∣ j - v then 1 else 0 := by
          unfold delV
          rw [hp, hsv, ipDen_upd_of _ _ _ _ hb, Function.update_self, hv]
          simp only [pin_dvd s hs1]
        simp only [g, bodyOf, List.map_append, List.map_cons, List.prod_append, List.prod_cons, hd]
        split_ifs <;> ring
      have hper' : ∀ c d : ℤ, ((f.1.den ρ.dv : ℕ) : ℤ) ∣ c - d → g c = g d := by
        intro c d hcd
        simp only [g, bodyOf]
        congr 1
        · congr 1; refine List.map_congr_left fun g' hg' => ?_
          by_cases hx : facHas x g' = true
          · have := hper g' (by simp only [List.mem_append] at hg' ⊢; tauto)
            rw [hx] at this
            exact (fac_congr ρ.dv hρ.dpos x f.1 g' (by simpa using this) ν c d hcd).1
          · simp only [Bool.not_eq_true] at hx
            unfold delV; rw [ipDen_upd_of _ _ _ _ hx, ipDen_upd_of _ _ _ _ hx]
        · congr 1; refine List.map_congr_left fun g' hg' => ?_
          by_cases hx : facHas x g' = true
          · have := hper g' (by simp only [List.mem_append]; tauto)
            rw [hx] at this
            exact (fac_congr ρ.dv hρ.dpos x f.1 g' (by simpa using this) ν c d hcd).2
          · simp only [Bool.not_eq_true] at hx
            unfold phV; rw [ipDen_upd_of _ _ _ _ hx, ipDen_upd_of _ _ _ _ hx]
      rw [Finset.sum_congr rfl fun (j : ℕ) _ => hsplit (j : ℤ),
        sum_delta_periodic _ _ hMpos ⟨_, hdD⟩ v g hper']
      have hq : D.den ρ.dv / f.1.den ρ.dv = (D.div f.1).den ρ.dv := by
        rw [hdD, Nat.mul_div_cancel_left _ hMpos]
      have hqc : 0 < (D.div f.1).c := by
        have hpos := DMon.den_pos ρ.dv hρ.dpos D (Nat.pos_of_ne_zero hD)
        rw [hdD] at hpos
        have := Nat.pos_of_mul_pos_left hpos
        by_contra hz
        simp only [not_lt, Nat.le_zero] at hz
        simp [DMon.den, hz] at this
      rw [hq, rtD_two ρ _ hqc, bodyOf_sub]
    · simp at h
  · simp at h

theorem elimDelta_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimDelta x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  rw [splits_spec ds p hp]
  exact elimDelta1_ok ρ hρ x D _ _ _ _ e he

theorem charData_spec (dv : ℕ → ℕ) (hdv : ∀ i, 0 < dv i) (x : ℕ) (D : DMon) (hD : 0 < D.c)
    (l : List Fac) (A B : IPoly) (h : charData x D l = some (A, B)) :
    ipHas x A = false ∧ ipHas x B = false ∧ ∀ (ν : ℕ → ℤ) (j : ℤ),
      (l.map (phV dv (Function.update ν x j))).prod =
        ((l.filter (fun f => !facHas x f)).map (phV dv ν)).prod *
          om (D.den dv) ^ (j * ipDen ν A + ipDen ν B) := by
  have hz : om (D.den dv) ≠ 0 := om_ne_zero _ (DMon.den_pos dv hdv D hD)
  induction l generalizing A B with
  | nil =>
    simp only [charData, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [ipHas, ipDen]
  | cons f l ih =>
    simp only [charData] at h
    split at h
    · simp at h
    · next A' B' hl =>
      obtain ⟨hA', hB', hrec⟩ := ih A' B' hl
      split_ifs at h with hx hfD
      · split at h
        · next a b hlin =>
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          obtain ⟨ha, hb, hp⟩ := ipLin_spec x f.2 a b hlin
          refine ⟨ipHas_add _ _ _ hA' ha, ipHas_add _ _ _ hB' hb, fun ν j => ?_⟩
          simp only [List.map_cons, List.prod_cons, List.filter_cons, hx, Bool.not_true,
            Bool.false_eq_true, ite_false, hrec, ipDen_add]
          unfold phV
          rw [hp, ipDen_upd_of _ _ _ _ ha, ipDen_upd_of _ _ _ _ hb, Function.update_self, hfD,
            show j * (ipDen ν A' + ipDen ν a) + (ipDen ν B' + ipDen ν b) =
              (ipDen ν a * j + ipDen ν b) + (j * ipDen ν A' + ipDen ν B') by ring,
            zpow_add₀ hz (ipDen ν a * j + ipDen ν b) (j * ipDen ν A' + ipDen ν B')]
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

theorem elimChar_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimChar x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  unfold elimChar at h
  split_ifs at h with hc
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
  split at h
  · next A B hcd =>
    simp only [Option.some.injEq] at h
    subst h
    have hD := Nat.pos_of_ne_zero hc.1
    obtain ⟨hA, hB, hrec⟩ := charData_spec ρ.dv hρ.dpos x D hD ps A B hcd
    intro ν
    have hds : ∀ j : ℤ, (ds.map (delV ρ.dv (Function.update ν x j))).prod =
        (ds.map (delV ρ.dv ν)).prod := by
      intro j
      have := bodyOf_upd ρ.dv x ds [] (by simpa using hc.2) ν j
      simpa [bodyOf] using this
    simp only [bodyOf, hds, hrec]
    rw [← Finset.mul_sum, ← Finset.mul_sum]
    rw [sum_om_lin _ (DMon.den_pos _ hρ.dpos D hD), rtD_two ρ D hD]
    simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append, List.prod_cons,
      List.prod_nil, mul_one, delV, phV]
    split_ifs <;> ring
  · simp at h

theorem elimGcd1_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (pre : List Fac) (f : Fac)
    (post ps : List Fac) (e : Elim) (h : elimGcd1 x D pre f post ps = some e) :
    ElimOK ρ x D (pre ++ f :: post) ps e := by
  unfold elimGcd1 at h
  split_ifs at h with hc
  obtain ⟨hfD, hD, hfree⟩ := hc
  split at h
  · next a b hlin =>
    split at h
    · next α hα =>
      split_ifs at h with hdiv
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨ha, hb, hp⟩ := ipLin_spec x f.2 a b hlin
      intro ν
      have hrest : ∀ j : ℤ, bodyOf ρ.dv (pre ++ post) ps (Function.update ν x j) =
          bodyOf ρ.dv (pre ++ post) ps ν := fun j => bodyOf_upd ρ.dv x _ ps hfree ν j
      have hsplit : ∀ j : ℤ, bodyOf ρ.dv (pre ++ f :: post) ps (Function.update ν x j) =
          (if ((D.den ρ.dv : ℕ) : ℤ) ∣ α * j + ipDen ν b then 1 else 0) *
            bodyOf ρ.dv (pre ++ post) ps ν := by
        intro j
        rw [← hrest j]
        have hd : delV ρ.dv (Function.update ν x j) f =
            if ((D.den ρ.dv : ℕ) : ℤ) ∣ α * j + ipDen ν b then 1 else 0 := by
          unfold delV
          rw [hp, ipConst_spec a α hα, ipDen_upd_of _ _ _ _ hb, Function.update_self, hfD,
            mul_comm]
        simp only [bodyOf, List.map_append, List.map_cons, List.prod_append, List.prod_cons, hd]
        ring
      rw [Finset.sum_congr rfl fun (j : ℕ) _ => hsplit (j : ℤ), ← Finset.sum_mul,
        sum_dvd_lin _ (DMon.den_pos _ hρ.dpos D (Nat.pos_of_ne_zero hD)) α _
          (ipDivBy_spec α b hdiv ν), gcdA_den]
    · simp at h
  · simp at h

theorem elimGcd_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimGcd x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  rw [splits_spec ds p hp]
  exact elimGcd1_ok ρ hρ x D _ _ _ _ e he

theorem elimVar_ok (ρ : Val) (hρ : ρ.OK) (x : ℕ) (D : DMon) (ds ps : List Fac) (e : Elim)
    (h : elimVar x D ds ps = some e) : ElimOK ρ x D ds ps e := by
  unfold elimVar at h
  split at h
  · next e' he => simp only [Option.some.injEq] at h; subst h; exact elimUnused_ok ρ x D ds ps _ he
  · split at h
    · next e' he => simp only [Option.some.injEq] at h; subst h; exact elimDelta_ok ρ hρ x D ds ps _ he
    · split at h
      · next e' he =>
        simp only [Option.some.injEq] at h; subst h; exact elimChar_ok ρ hρ x D ds ps _ he
      · exact elimGcd_ok ρ hρ x D ds ps e h

/-- One elimination of a bound variable, the first that applies. -/
def elimStep (t : STm) : Option STm :=
  firstSome (fun p => if (p.1 ++ p.2.2).map (·.1) |>.contains p.2.1.1 then none else
    match elimVar p.2.1.1 p.2.1.2 t.dels t.phs with
    | some e => some ⟨CMon.mul t.c e.k, p.1 ++ p.2.2, e.ds, e.ps⟩
    | none => none) (splits t.sums)

theorem elimStep_ok (ρ : Val) (hρ : ρ.OK) (t t' : STm) (h : elimStep t = some t') (ν : ℕ → ℤ) :
    t'.den ρ ν = t.den ρ ν := by
  obtain ⟨p, hp, he⟩ := firstSome_spec _ _ _ h
  obtain ⟨a, ⟨x, D⟩, b⟩ := p
  simp only at he
  split_ifs at he with hx
  split at he
  · next e hev =>
    simp only [Option.some.injEq] at he
    subst he
    have hok := elimVar_ok ρ hρ x D t.dels t.phs e hev
    have hs := splits_spec t.sums _ hp
    simp only at hs
    unfold STm.den
    rw [hs, sumsV_pull ρ.dv a b x D (by simpa using hx)]
    simp only [body_eq] at *
    rw [sumsV_congr _ _ _ _ hok, sumsV_mul_left, CMon.den_mul ρ hρ]
    ring
  · simp at he

/-- Eliminations repeated, at most `n` of them. -/
def elimAll : ℕ → STm → STm
  | 0, t => t
  | n + 1, t => match elimStep t with
    | some t' => elimAll n t'
    | none => t

theorem elimAll_ok (ρ : Val) (hρ : ρ.OK) (n : ℕ) (t : STm) (ν : ℕ → ℤ) :
    (elimAll n t).den ρ ν = t.den ρ ν := by
  induction n generalizing t with
  | zero => rfl
  | succ n ih =>
    simp only [elimAll]
    split
    · next t' h => rw [ih, elimStep_ok ρ hρ t t' h]
    · rfl

end ArchytasZX.Sym
