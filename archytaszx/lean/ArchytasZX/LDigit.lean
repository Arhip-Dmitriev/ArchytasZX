/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LSym

/-!
# Digit-wise character sums

For `N = d^n`, `Σ_{x<N} ω_d^(Σ_{i<n} x_i e_i) = Π_{i<n} d [d ∣ e_i]`, `x_i` the base-`d`
digits of `x` (`sum_dig`). `elimDigit` removes a variable read only through digit products
`ddot(x, A_k)` and linearly through roots of unity of order `d`; it leaves the digit condition
atom, which vanishes exactly when every digit of `Σ_k c_k A_k + B` (digit-wise) is divisible
by `d`.
-/

namespace ArchytasZX.LSym

open ArchytasZX.Sym Complex Finset

/-! ## Digits -/

theorem dig_nat (d i m : ℕ) : dig d i (m : ℤ) = ((m / d ^ i % d : ℕ) : ℤ) := by
  simp [dig, Int.natCast_div, Int.natCast_mod]

theorem nat_dig_low (d n i b y : ℕ) (hd : 0 < d) (hi : i < n) :
    (b + d ^ n * y) / d ^ i % d = b / d ^ i % d := by
  have hpos : 0 < d ^ i := pow_pos hd i
  have e0 : d ^ n = d ^ i * d * d ^ (n - i - 1) := by
    rw [← pow_succ, ← pow_add]; congr 1; omega
  have e : d ^ n * y = d ^ i * (d * (d ^ (n - i - 1) * y)) := by rw [e0]; ring
  rw [e, Nat.add_mul_div_left _ _ hpos, Nat.add_mul_mod_self_left]

theorem nat_dig_top (d n b y : ℕ) (hd : 0 < d) (hb : b < d ^ n) (hy : y < d) :
    (b + d ^ n * y) / d ^ n % d = y := by
  rw [Nat.add_mul_div_left _ _ (pow_pos hd n), Nat.div_eq_of_lt hb, zero_add,
    Nat.mod_eq_of_lt hy]

theorem om_zpow_congr (d : ℕ) (hd : 0 < d) (a b : ℤ) (h : (d : ℤ) ∣ a - b) :
    om d ^ a = om d ^ b := by
  obtain ⟨k, hk⟩ := h
  have hz := om_ne_zero d hd
  rw [show a = b + d * k by omega, zpow_add₀ hz, zpow_mul, zpow_natCast, om_pow_self d hd,
    one_zpow, mul_one]

/-- The digit-wise character sum. -/
theorem sum_dig (d : ℕ) (hd : 0 < d) (n : ℕ) (e : ℕ → ℤ) :
    ∑ x ∈ range (d ^ n), om d ^ (∑ i ∈ range n, dig d i x * e i) =
      ∏ i ∈ range n, (if (d : ℤ) ∣ e i then (d : ℂ) else 0) := by
  have hz := om_ne_zero d hd
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ, sum_range_mul, Finset.prod_range_succ, ← ih, Finset.sum_mul]
    refine Finset.sum_congr rfl fun b hb => ?_
    have hb' := Finset.mem_range.1 hb
    have hsplit : ∀ y ∈ range d, om d ^ (∑ i ∈ range (n + 1), dig d i ((b + d ^ n * y : ℕ) : ℤ) * e i) =
        om d ^ (∑ i ∈ range n, dig d i (b : ℤ) * e i) * om d ^ ((y : ℤ) * e n + 0) := by
      intro y hy
      rw [← zpow_add₀ hz, Finset.sum_range_succ]
      congr 1
      rw [dig_nat, nat_dig_top d n b y hd hb' (Finset.mem_range.1 hy)]
      simp only [add_zero]
      congr 1
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [dig_nat, dig_nat, nat_dig_low d n i b y hd (Finset.mem_range.1 hi)]
    rw [Finset.sum_congr rfl hsplit, ← Finset.mul_sum, sum_om_lin d hd]
    simp [zpow_zero]

/-- With a linear part read through the lowest digit. -/
theorem sum_dig_lin (d : ℕ) (hd : 0 < d) (n : ℕ) (e : ℕ → ℤ) (B C : ℤ) :
    ∑ x ∈ range (d ^ n), om d ^ ((x : ℤ) * B + (∑ i ∈ range n, dig d i x * e i) + C) =
      om d ^ C * ∏ i ∈ range n,
        (if (d : ℤ) ∣ e i + (if i = 0 then B else 0) then (d : ℂ) else 0) := by
  have hz := om_ne_zero d hd
  rw [← sum_dig d hd n, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [← zpow_add₀ hz]
  apply om_zpow_congr d hd
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    have : x = 0 := by simpa using hx
    subst this; simp
  · have h0 : 0 ∈ range n := Finset.mem_range.2 hn
    have e1 : ∑ i ∈ range n, dig d i x * (e i + if i = 0 then B else 0) =
        ∑ i ∈ range n, dig d i x * e i + dig d 0 x * B := by
      simp only [mul_add, Finset.sum_add_distrib, mul_ite, mul_zero]
      simp [Finset.sum_ite_eq', h0]
    rw [e1]
    have e2 : dig d 0 (x : ℤ) = (x : ℤ) % d := by simp [dig]
    rw [e2, Int.emod_def]
    exact ⟨(x : ℤ) / d * B, by ring⟩

theorem pos_pow (d n : ℕ) (hd : 2 ≤ d) : pos d (d ^ n) = range n := by
  ext i
  simp only [pos, Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨_, h⟩; exact (Nat.pow_lt_pow_iff_right (by omega)).1 h
  · intro h
    exact ⟨lt_of_lt_of_le (Nat.lt_pow_self (by omega)) (Nat.pow_le_pow_right (by omega) h.le),
      (Nat.pow_lt_pow_iff_right (by omega)).2 h⟩

/-- A value read as digits below `d`, each in `[0, d)`: zero exactly when every digit is. -/
theorem digits_zero (d n : ℕ) (hd : 0 < d) (r : ℕ → ℤ) (hr : ∀ i, 0 ≤ r i ∧ r i < d) :
    (((d ^ n : ℕ) : ℤ) ∣ ∑ i ∈ range n, r i * (d : ℤ) ^ i) ↔ ∀ i ∈ range n, r i = 0 := by
  have hb : ∀ m, 0 ≤ ∑ i ∈ range m, r i * (d : ℤ) ^ i ∧
      ∑ i ∈ range m, r i * (d : ℤ) ^ i < (d : ℤ) ^ m := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [Finset.sum_range_succ, pow_succ]
      have h1 := hr m
      have h2 : (0 : ℤ) < (d : ℤ) ^ m := by positivity
      constructor
      · nlinarith
      · nlinarith
  obtain ⟨h0, h1⟩ := hb n
  constructor
  · intro h
    have hz : ∑ i ∈ range n, r i * (d : ℤ) ^ i = 0 := by
      have := Int.emod_eq_of_lt h0 (by exact_mod_cast h1)
      rw [Int.emod_eq_zero_of_dvd (by exact_mod_cast h)] at this
      exact this.symm
    intro i hi
    have hnn : ∀ j ∈ range n, 0 ≤ r j * (d : ℤ) ^ j := fun j _ =>
      mul_nonneg (hr j).1 (by positivity)
    have := (Finset.sum_eq_zero_iff_of_nonneg hnn).1 hz i hi
    exact (mul_eq_zero.1 this).resolve_right (by positivity)
  · intro h
    rw [Finset.sum_eq_zero fun i hi => by rw [h i hi, zero_mul]]
    exact dvd_zero _

/-! ## The digit elimination -/

/-- The pairs `(d, N)` with `2 ≤ d` and `N = d^n`. -/
def PowOK (PW : List (DMon × DMon)) (ρ : Val) : Prop :=
  ∀ p ∈ PW, 2 ≤ p.1.den ρ.dv ∧ ∃ n, p.2.den ρ.dv = p.1.den ρ.dv ^ n

/-- Whether an atom's arguments read `x`. -/
def atReads (x : ℕ) (a : Atom) : Bool := (atVs a).contains x

/-- `ddot_{d,N}(x, A)`, both arguments reduced modulo `N`: `A`. -/
def digAt (x : ℕ) (d N : DMon) (a : Atom) : Option IPoly :=
  if a.f = 0 ∧ a.ks = [] ∧ a.dp = [d, N] then
    match a.args with
    | [(p, M1), (A, M2)] =>
      if M1 = N ∧ M2 = N ∧ p = ipV x ∧ ipHas x A = false then some A
      else if M1 = N ∧ M2 = N ∧ A = ipV x ∧ ipHas x p = false then some p else none
    | _ => none
  else none

/-- The linear coefficients of the variables `zs`, all constant: the coefficients and the
rest. -/
def coefs : List ℕ → IPoly → Option (List ℤ × IPoly)
  | [], R => some ([], R)
  | z :: zs, R =>
    match ipLin z R with
    | some (c, R') =>
      match ipConst c, coefs zs R' with
      | some k, some (ks, R'') => some (k :: ks, R'')
      | _, _ => none
    | none => none

/-- Whether a factor reads any of `zs`. -/
def facAny (zs : List ℕ) (f : Fac) : Bool := zs.any fun z => facHas z f

/-- The polynomial `Σ` of the roots of unity. -/
def psum (ps : List Fac) : IPoly := ps.foldr (fun f acc => ipAdd f.2 acc) []

/-- `f` on every element, when all are defined. -/
def omap {α β : Type} (f : α → Option β) : List α → Option (List β)
  | [] => some []
  | a :: l =>
    match f a, omap f l with
    | some b, some bs => some (b :: bs)
    | _, _ => none

/-- The digit product of two values, both reduced modulo `N`. -/
def ddv (V : LV) (d N : DMon) (u a : ℤ) : ℤ :=
  bfn 0 [] [d.den V.ρ.dv, N.den V.ρ.dv]
    [u % ((N.den V.ρ.dv : ℕ) : ℤ), a % ((N.den V.ρ.dv : ℕ) : ℤ)]

theorem digAt_spec (x : ℕ) (d N : DMon) (a : Atom) (A : IPoly) (h : digAt x d N a = some A) :
    a.f = 0 ∧ a.ks = [] ∧ a.dp = [d, N] ∧
      (a.args = [(ipV x, N), (A, N)] ∨ a.args = [(A, N), (ipV x, N)]) ∧
      ipHas x A = false := by
  unfold digAt at h
  split_ifs at h with h1
  split at h
  · next p M1 A' M2 hargs =>
    split_ifs at h with h2 h3
    · simp only [Option.some.injEq] at h
      subst h
      obtain ⟨e1, e2, e3, e4⟩ := h2
      subst e1 e2 e3
      exact ⟨h1.1, h1.2.1, h1.2.2, Or.inl hargs, e4⟩
    · simp only [Option.some.injEq] at h
      subst h
      obtain ⟨e1, e2, e3, e4⟩ := h3
      subst e1 e2 e3
      exact ⟨h1.1, h1.2.1, h1.2.2, Or.inr hargs, e4⟩
  · simp at h

theorem aval_dig (V : LV) (x : ℕ) (d N : DMon) (a : Atom) (A : IPoly)
    (h : digAt x d N a = some A) (μ : ℕ → ℤ) : aval V μ a = ddv V d N (μ x) (ipDen μ A) := by
  obtain ⟨h1, h2, h3, h4, -⟩ := digAt_spec x d N a A h
  rcases h4 with h4 | h4
  · simp [aval, h1, h2, h3, h4, argV, ddv, ipDen_V]
  · simp only [aval, h1, h2, h3, h4, argV, ddv, ipDen_V, show (0 : ℕ) < 100 by norm_num,
      if_true, bfn, List.map_cons, List.map_nil, List.getD_cons_zero, List.getD_cons_succ]
    exact Finset.sum_congr rfl fun i _ => mul_comm _ _

theorem zipWith_congr' {α β γ : Type} (f g : α → β → γ) (l1 : List α) (l2 : List β)
    (h : ∀ a b, b ∈ l2 → f a b = g a b) : List.zipWith f l1 l2 = List.zipWith g l1 l2 := by
  induction l1 generalizing l2 with
  | nil => simp
  | cons a l1 ih =>
    cases l2 with
    | nil => simp
    | cons b l2 =>
      simp only [List.zipWith_cons_cons]
      rw [h a b List.mem_cons_self, ih l2 fun a' b' hb => h a' b' (List.mem_cons_of_mem _ hb)]

theorem digs_sum (V : LV) (x : ℕ) (d N : DMon) (dd : List Atom) (As : List IPoly)
    (h : omap (digAt x d N) dd = some As) (cs : List ℤ) (hnd : (ids dd).Nodup)
    (hx : x ∉ ids dd) (hA : ∀ A ∈ As, ∀ z ∈ ids dd, z ∉ ipVs A) (μ : ℕ → ℤ) :
    (List.zipWith (fun c z => c * aext V dd μ z) cs (ids dd)).sum =
      (List.zipWith (fun c A => c * ddv V d N (μ x) (ipDen μ A)) cs As).sum := by
  induction dd generalizing As cs μ with
  | nil => simp only [omap, Option.some.injEq] at h; subst h; simp [ids]
  | cons a dd ih =>
    simp only [omap] at h
    split at h
    · next A As' ha hl =>
      simp only [Option.some.injEq] at h
      subst h
      cases cs with
      | nil => simp
      | cons c cs =>
        simp only [ids, List.map_cons, List.nodup_cons, List.mem_cons, not_or] at hnd hx
        simp only [ids, List.map_cons, List.zipWith_cons_cons, List.sum_cons, aext]
        rw [aext_off V dd _ a.id hnd.1, Function.update_self, aval_dig V x d N a A ha]
        have ih' := ih As' hl cs hnd.2 hx.2 (fun A' hA' z hz => hA A' (List.mem_cons_of_mem _ hA') z
          (by simp only [ids, List.map_cons, List.mem_cons]; exact Or.inr hz))
          (Function.update μ a.id (ddv V d N (μ x) (ipDen μ A)))
        simp only [ids] at ih'
        rw [ih']
        congr 2
        refine zipWith_congr' _ _ _ _ fun c' A' hA' => ?_
        rw [Function.update_of_ne hx.1, ipDen_upd_of _ _ _ _ (ipHas_false_of _ _
          (hA A' (List.mem_cons_of_mem _ hA') a.id (by simp [ids])))]
    · simp at h

theorem coefs_spec (zs : List ℕ) (R0 R : IPoly) (cs : List ℤ) (h : coefs zs R0 = some (cs, R))
    (ν : ℕ → ℤ) : ipDen ν R0 = (List.zipWith (fun c z => c * ν z) cs zs).sum + ipDen ν R := by
  induction zs generalizing R0 cs with
  | nil => simp only [coefs, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h; simp
  | cons z zs ih =>
    simp only [coefs] at h
    split at h
    · next c R' hlin =>
      split at h
      · next k ks R'' hk hc =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨-, -, hp⟩ := ipLin_spec z R0 c R' hlin
        rw [hp, ipConst_spec c k hk, ih R' ks hc]
        simp only [List.zipWith_cons_cons, List.sum_cons]; ring
      · simp at h
    · simp at h

theorem psum_om (dv : ℕ → ℕ) (d : DMon) (hd : 0 < d.den dv) (rd : List Fac)
    (h : ∀ f ∈ rd, f.1 = d) (μ : ℕ → ℤ) :
    (rd.map (phV dv μ)).prod = om (d.den dv) ^ ipDen μ (psum rd) := by
  induction rd with
  | nil => simp [psum, ipDen]
  | cons f rd ih =>
    simp only [List.map_cons, List.prod_cons, psum, List.foldr_cons] at ih ⊢
    rw [ih fun g hg => h g (List.mem_cons_of_mem _ hg), ipDen_add, phV,
      h f List.mem_cons_self, zpow_add₀ (om_ne_zero _ hd)]

theorem prod_filter_split {α : Type} (l : List α) (p : α → Bool) (g : α → ℂ) :
    (l.map g).prod = ((l.filter fun a => !p a).map g).prod * ((l.filter p).map g).prod := by
  induction l with
  | nil => simp
  | cons a l ih =>
    by_cases ha : p a = true
    · simp [List.filter_cons, ha, ih]; ring
    · simp only [Bool.not_eq_true] at ha
      simp [List.filter_cons, ha, ih]; ring

/-- The digit elimination, its data found. -/
def digitCore (PW : List (DMon × DMon)) (x : ℕ) (D : DMon) (ats others dd : List Atom)
    (a0 : Atom) (d N : DMon) (As : List IPoly) (ds rest : List Fac) (B R : IPoly)
    (cs : List ℤ) : Option AElim :=
  if ats = others ++ dd ∧ a0 ∈ dd ∧ D = N ∧ (d, N) ∈ PW ∧ D.c ≠ 0 ∧ (ids dd).Nodup ∧
      x ∉ ids ats ∧ x ∉ argVs others ∧
      (∀ z ∈ ids dd, z ∉ argVs ats ∧ z ∉ ipVs B ∧ z ∉ ipVs R) ∧
      (∀ A ∈ As, plainP ats A = true) ∧
      (∀ f ∈ ds, facHas x f = false ∧ facAny (ids dd) f = false) ∧
      (∀ f ∈ rest, facHas x f = false ∧ facAny (ids dd) f = false) ∧ ipHas x R = false ∧
      ipHas x B = false then
    some ⟨CMon.rtD N 2, others ++ [⟨a0.id, 3, cs, [d, N], As.map (·, N) ++ [(B, d)]⟩],
      ds ++ [(N, ipV a0.id)], rest ++ [(d, R)]⟩
  else none

theorem omap_spec {α β : Type} (f : α → Option β) (l : List α) (bs : List β)
    (h : omap f l = some bs) : bs.length = l.length ∧ ∀ b ∈ bs, ∃ a ∈ l, f a = some b := by
  induction l generalizing bs with
  | nil => simp only [omap, Option.some.injEq] at h; subst h; simp
  | cons a l ih =>
    simp only [omap] at h
    split at h
    · next b bs' hb hl =>
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨h1, h2⟩ := ih bs' hl
      refine ⟨by simp [h1], fun b' hb' => ?_⟩
      rcases List.mem_cons.1 hb' with rfl | hb'
      · exact ⟨a, List.mem_cons_self, hb⟩
      · obtain ⟨a', ha', e⟩ := h2 b' hb'
        exact ⟨a', List.mem_cons_of_mem _ ha', e⟩
    · simp at h

theorem not_mem_ipVs (x : ℕ) (p : IPoly) (h : ipHas x p = false) : x ∉ ipVs p := by
  simp only [ipHas, List.any_eq_false, Bool.not_eq_true, List.contains_eq_mem,
    decide_eq_false_iff_not] at h
  simp only [ipVs, List.mem_flatMap, not_exists, not_and]
  exact fun t ht hx => h t ht hx

theorem fac_free (x : ℕ) (zs : List ℕ) (f : Fac) (h1 : facHas x f = false)
    (h2 : facAny zs f = false) : ∀ y ∈ facVars f, y ≠ x ∧ y ∉ zs := by
  intro y hy
  refine ⟨fun e => not_mem_ipVs x f.2 h1 (e ▸ hy), fun hz => ?_⟩
  simp only [facAny, List.any_eq_false, Bool.not_eq_true] at h2
  exact not_mem_ipVs y f.2 (h2 y hz) hy

theorem zipWith_sum_swap (s : Finset ℕ) (f : ℕ → ℤ) (g : ℤ → ℕ → ℤ) (cs as : List ℤ) :
    (List.zipWith (fun c a => c * ∑ i ∈ s, f i * g a i) cs as).sum =
      ∑ i ∈ s, f i * (List.zipWith (fun c a => c * g a i) cs as).sum := by
  induction cs generalizing as with
  | nil => simp
  | cons c cs ih =>
    cases as with
    | nil => simp
    | cons a as =>
      simp only [List.zipWith_cons_cons, List.sum_cons]
      rw [ih, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring

theorem lsum_append (ks xs ys : List ℤ) (g : ℤ → ℤ) (h : ks.length ≤ xs.length) :
    lsum ks (xs ++ ys) g = lsum ks xs g := by
  induction ks generalizing xs with
  | nil => simp [lsum]
  | cons k ks ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      simp only [lsum, List.cons_append, List.zipWith_cons_cons, List.sum_cons] at ih ⊢
      rw [ih xs (by simpa using h)]

theorem ddv_eq (V : LV) (d N : DMon) (n : ℕ) (hd2 : 2 ≤ d.den V.ρ.dv)
    (hN : N.den V.ρ.dv = d.den V.ρ.dv ^ n) (u a : ℤ) :
    ddv V d N u a = ∑ i ∈ range n, dig (d.den V.ρ.dv) i (u % ((N.den V.ρ.dv : ℕ) : ℤ)) *
      dig (d.den V.ρ.dv) i (a % ((N.den V.ρ.dv : ℕ) : ℤ)) := by
  simp only [ddv, bfn, List.getD_cons_zero, List.getD_cons_succ]
  rw [hN, pos_pow _ _ hd2]

theorem digitCore_ok (V : LV) (hρ : V.ρ.OK) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ)
    (x : ℕ) (D : DMon) (ats others dd : List Atom) (a0 : Atom) (d N : DMon) (As : List IPoly)
    (ds ps rd rest : List Fac) (B R0 R : IPoly) (cs : List ℤ) (e : AElim)
    (hAs : omap (digAt x d N) dd = some As)
    (hps : ∀ μ, (ps.map (phV V.ρ.dv μ)).prod =
      (rest.map (phV V.ρ.dv μ)).prod * (rd.map (phV V.ρ.dv μ)).prod)
    (hrd : ∀ f ∈ rd, f.1 = d) (hlin : ipLin x (psum rd) = some (B, R0))
    (hco : coefs (ids dd) R0 = some (cs, R))
    (h : digitCore PW x D ats others dd a0 d N As ds rest B R cs = some e) :
    AElimOK V x D ats ds ps e := by
  unfold digitCore at h
  split_ifs at h with hc
  simp only [Option.some.injEq] at h
  subst h
  obtain ⟨hats, ha0, hDN, hpw, hDc, hnd, hxi, hxo, hz, hA, hds, hrest, hRx, hBx⟩ := hc
  subst hDN
  obtain ⟨hd2, n, hNn⟩ := hPW _ hpw
  have hd2' : 2 ≤ d.den V.ρ.dv := hd2
  have hNn' : D.den V.ρ.dv = d.den V.ρ.dv ^ n := hNn
  clear hd2 hNn
  have hdpos : 0 < d.den V.ρ.dv := by omega
  obtain ⟨hcl, -⟩ := omap_spec _ _ _ hAs
  have hAx : ∀ A ∈ As, ipHas x A = false := fun A hA' => by
    obtain ⟨a, _, ha⟩ := (omap_spec _ _ _ hAs).2 A hA'
    exact (digAt_spec x d D a A ha).2.2.2.2
  have hAp : ∀ A ∈ As, ∀ y ∈ ipVs A, y ∉ ids ats := fun A hA' y hy => by
    have := hA A hA'
    simp only [plainP, List.all_eq_true, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not] at this
    exact this y hy
  have hidsA : ids ats = ids others ++ ids dd := by rw [hats]; simp [ids]
  obtain ⟨-, -, hP⟩ := ipLin_spec x _ B R0 hlin
  have hcs : cs.length = As.length := by
    have : ∀ (zs : List ℕ) (R0 R : IPoly) (cs : List ℤ), coefs zs R0 = some (cs, R) →
        cs.length = zs.length := by
      intro zs
      induction zs with
      | nil => intro R0 R cs h; simp only [coefs, Option.some.injEq, Prod.mk.injEq] at h; rw [← h.1]; rfl
      | cons z zs ih =>
        intro R0 R cs h
        simp only [coefs] at h
        split at h
        · split at h
          · next k ks R'' _ hc' =>
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            rw [← h.1]; simp [ih _ _ _ hc']
          · simp at h
        · simp at h
    rw [this _ _ _ _ hco, hcl]; simp [ids]
  intro ν
  set dv := V.ρ.dv
  set E := aext V others ν
  have hxo' : x ∉ argVs others ++ ids others := by
    simp only [List.mem_append, not_or]
    exact ⟨hxo, fun h' => hxi (by rw [hidsA]; exact List.mem_append_left _ h')⟩
  have hxd : x ∉ ids dd := fun h' => hxi (by rw [hidsA]; exact List.mem_append_right _ h')
  have hAν : ∀ A ∈ As, ∀ μ : ℕ → ℤ, (∀ y, y ∉ ids ats → y ≠ x → μ y = ν y) →
      ipDen μ A = ipDen ν A := fun A hA' μ hμ =>
    ipDen_reads A _ _ fun y hy => hμ y (hAp A hA' y hy) (fun e' =>
      not_mem_ipVs x A (hAx A hA') (e' ▸ hy))
  have hEν : ∀ y, y ∉ ids others → E y = ν y := fun y hy => aext_off V others ν y hy
  -- the summand at `x = j`
  set a : List ℤ := As.map fun A => ipDen ν A % ((D.den dv : ℕ) : ℤ)
  set ee : ℕ → ℤ := fun i => lsum cs a (dig (d.den dv) i)
  have hsum : ∀ j : ℕ, j < D.den dv →
      bodyOf dv ds ps (aext V ats (Function.update ν x j)) =
        (ds.map (delV dv E)).prod * (rest.map (phV dv E)).prod *
          om (d.den dv) ^ ((j : ℤ) * ipDen E B +
            (∑ i ∈ range n, dig (d.den dv) i j * ee i) + ipDen E R) := by
    intro j hj
    rw [hats, aext_append, aext_upd V others x hxo']
    set G := aext V dd (Function.update E x j)
    have hG : ∀ y, y ∉ ids dd → y ≠ x → G y = E y := fun y h1 h2 => by
      show aext V dd (Function.update E x j) y = E y
      rw [aext_off V dd _ y h1, Function.update_of_ne h2]
    have hGx : G x = j := by
      show aext V dd (Function.update E x j) x = j
      rw [aext_off V dd _ x hxd, Function.update_self]
    have hfac : ∀ f : Fac, facHas x f = false → facAny (ids dd) f = false →
        ipDen G f.2 = ipDen E f.2 := fun f h1 h2 =>
      ipDen_reads f.2 _ _ fun y hy =>
        hG y (fac_free x _ f h1 h2 y hy).2 (fac_free x _ f h1 h2 y hy).1
    unfold bodyOf
    rw [hps, psum_om dv d hdpos rd hrd]
    have e1 : (ds.map (delV dv G)).prod = (ds.map (delV dv E)).prod :=
      congrArg _ (List.map_congr_left fun f hf => by
        unfold delV; rw [hfac f (hds f hf).1 (hds f hf).2])
    have e2 : (rest.map (phV dv G)).prod = (rest.map (phV dv E)).prod :=
      congrArg _ (List.map_congr_left fun f hf => by
        unfold phV; rw [hfac f (hrest f hf).1 (hrest f hf).2])
    rw [e1, e2, mul_assoc]
    congr 2
    have hB : ipDen G B = ipDen E B := ipDen_reads B _ _ fun y hy =>
      hG y (fun hy' => (hz y hy').2.1 hy) (fun e' => not_mem_ipVs x B hBx (e' ▸ hy))
    have hR : ipDen G R = ipDen E R := ipDen_reads R _ _ fun y hy =>
      hG y (fun hy' => (hz y hy').2.2 hy) (fun e' => not_mem_ipVs x R hRx (e' ▸ hy))
    rw [hP, coefs_spec _ _ _ _ hco, hB, hR, hGx]
    have hd := digs_sum V x d D dd As hAs cs hnd hxd (fun A hA' z hz' hzA =>
      hAp A hA' z hzA (by rw [hidsA]; exact List.mem_append_right _ hz'))
      (Function.update E x j)
    simp only [ids] at hd ⊢
    rw [hd]
    have hjN : ((j : ℤ) % ((D.den dv : ℕ) : ℤ)) = j := Int.emod_eq_of_lt (by positivity)
      (by exact_mod_cast hj)
    have hA' : ∀ A ∈ As, ipDen (Function.update E x j) A = ipDen ν A := fun A hA'' =>
      hAν A hA'' _ fun y hy hyx => by
        rw [Function.update_of_ne hyx]
        exact hEν y fun h' => hy (by rw [hidsA]; exact List.mem_append_left _ h')
    have hz2 : List.zipWith (fun c A => c * ddv V d D (Function.update E x (j : ℤ) x)
        (ipDen (Function.update E x j) A)) cs As = List.zipWith (fun c a' => c *
          ∑ i ∈ range n, dig (d.den dv) i j * dig (d.den dv) i a') cs a := by
      rw [show a = As.map (fun A => ipDen ν A % ((D.den dv : ℕ) : ℤ)) from rfl,
        List.zipWith_map_right]
      refine zipWith_congr' _ _ _ _ fun c A hA'' => ?_
      rw [Function.update_self, hA' A hA'', ddv_eq V d D n hd2' hNn', hjN]
    rw [hz2, zipWith_sum_swap]
    simp only [ee, lsum]
    ring
  rw [Finset.sum_congr rfl fun j hj => hsum j (Finset.mem_range.1 hj), ← Finset.mul_sum, hNn',
    sum_dig_lin _ hdpos n ee (ipDen E B) (ipDen E R), Finset.prod_ite_zero,
    rtD_two V.ρ D (Nat.pos_of_ne_zero hDc), aext_append]
  simp only [aext]
  have hz0 : a0.id ∈ ids dd := List.mem_map_of_mem ha0
  set dz : Atom := ⟨a0.id, 3, cs, [d, D], As.map (·, D) ++ [(B, d)]⟩
  set dzv := aval V E dz
  have hnz : ∀ f : Fac, facAny (ids dd) f = false →
      ipDen (Function.update E a0.id dzv) f.2 = ipDen E f.2 := fun f hf => by
    simp only [facAny, List.any_eq_false, Bool.not_eq_true] at hf
    exact ipDen_upd_of _ _ _ _ (hf _ hz0)
  simp only [bodyOf, List.map_append, List.map_cons, List.map_nil, List.prod_append,
    List.prod_cons, List.prod_nil, mul_one]
  have e1 : (ds.map (delV dv (Function.update E a0.id dzv))).prod = (ds.map (delV dv E)).prod :=
    congrArg _ (List.map_congr_left fun f hf => by unfold delV; rw [hnz f (hds f hf).2])
  have e2 : (rest.map (phV dv (Function.update E a0.id dzv))).prod =
      (rest.map (phV dv E)).prod :=
    congrArg _ (List.map_congr_left fun f hf => by unfold phV; rw [hnz f (hrest f hf).2])
  have e3 : phV dv (Function.update E a0.id dzv) (d, R) = om (d.den dv) ^ ipDen E R := by
    unfold phV; rw [ipDen_upd_of _ _ _ _ (ipHas_false_of _ _ (hz _ hz0).2.2)]
  have e4 : delV dv (Function.update E a0.id dzv) (D, ipV a0.id) =
      if ((D.den dv : ℕ) : ℤ) ∣ dzv then 1 else 0 := by
    unfold delV; rw [ipDen_V, Function.update_self]
  rw [e1, e2, e3, e4]
  -- the digit condition
  have ha : a.length = cs.length := by simp [a, hcs]
  have hargs : dz.args.map (argV dv E) = a ++ [ipDen E B % ((d.den dv : ℕ) : ℤ)] := by
    simp only [dz, List.map_append, List.map_map, List.map_cons, List.map_nil]
    congr 1
    refine List.map_congr_left fun A hA'' => ?_
    simp only [Function.comp_apply, argV]
    rw [hAν A hA'' E fun y hy _ => hEν y fun h' => hy (by
      rw [hidsA]; exact List.mem_append_left _ h')]
  have hdzv : dzv = ∑ i ∈ range n, ((ee i + if i = 0 then ipDen E B % ((d.den dv : ℕ) : ℤ)
      else 0) % (d.den dv : ℤ)) * (d.den dv : ℤ) ^ i := by
    show aval V E dz = _
    simp only [aval, dz, show (3 : ℕ) < 100 by norm_num, if_true, bfn]
    rw [hargs]
    simp only [List.map_cons, List.map_nil, List.getD_cons_zero, List.getD_cons_succ]
    rw [hNn', pos_pow _ _ hd2']
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [lsum_append _ _ _ _ ha.ge, List.getD_append_right _ _ _ _ (by omega)]
    (simp [ee, ha]) <;> rfl
  have hcond : (((D.den dv : ℕ) : ℤ) ∣ dzv) ↔
      ∀ i ∈ range n, ((d.den dv : ℕ) : ℤ) ∣ ee i + if i = 0 then ipDen E B else 0 := by
    rw [hdzv, hNn']
    rw [digits_zero (d.den dv) n hdpos _ fun i => ⟨Int.emod_nonneg _ (by positivity),
      Int.emod_lt_of_pos _ (by positivity)⟩]
    refine forall₂_congr fun i _ => ?_
    by_cases h0 : i = 0
    · simp only [h0, if_true, Int.add_emod_emod]; exact (Int.dvd_iff_emod_eq_zero ..).symm
    · simp only [h0, if_false, add_zero]; exact (Int.dvd_iff_emod_eq_zero ..).symm
  rw [Finset.prod_const, Finset.card_range]
  by_cases hc : ∀ i ∈ range n, ((d.den dv : ℕ) : ℤ) ∣ ee i + if i = 0 then ipDen E B else 0
  · rw [if_pos hc, if_pos (hcond.2 hc), show D.den V.ρ.dv = d.den V.ρ.dv ^ n from hNn']
    push_cast; ring
  · rw [if_neg hc, if_neg (fun h' => hc (hcond.1 h'))]; ring

/-- `x` read only through digit products `ddot_{d,N}(x, A_k)` at the end of the table and
linearly through roots of unity of order `d`. -/
def elimDigit (PW : List (DMon × DMon)) (x : ℕ) (D : DMon) (ats : List Atom)
    (ds ps : List Fac) : Option AElim :=
  let others := ats.filter fun a => !atReads x a
  let dd := ats.filter fun a => atReads x a
  match dd with
  | [] => none
  | a0 :: _ =>
    match a0.dp with
    | [d, N] =>
      match omap (digAt x d N) dd with
      | none => none
      | some As =>
        let sel := fun f : Fac => decide (f.1 = d) && (facHas x f || facAny (ids dd) f)
        match ipLin x (psum (ps.filter sel)) with
        | none => none
        | some (B, R0) =>
          match coefs (ids dd) R0 with
          | none => none
          | some (cs, R) =>
            digitCore PW x D ats others dd a0 d N As ds (ps.filter fun f => !sel f) B R cs
    | _ => none

theorem elimDigit_ok (V : LV) (hρ : V.ρ.OK) (PW : List (DMon × DMon)) (hPW : PowOK PW V.ρ)
    (x : ℕ) (D : DMon) (ats : List Atom) (ds ps : List Fac) (e : AElim)
    (h : elimDigit PW x D ats ds ps = some e) : AElimOK V x D ats ds ps e := by
  unfold elimDigit at h
  simp only at h
  split at h
  · simp at h
  · next a0 rest' hdd =>
    split at h
    · next d N hdp =>
      split at h
      · simp at h
      · next As hAs =>
        split at h
        · simp at h
        · next B R0 hlin =>
          split at h
          · simp at h
          · next cs R hco =>
            refine digitCore_ok V hρ PW hPW x D ats _ _ a0 d N As ds ps _ _ B R0 R cs e hAs
              (fun μ => prod_filter_split ps _ (phV V.ρ.dv μ)) (fun f hf => ?_) hlin hco h
            simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq] at hf
            exact hf.2.1
    · simp at h

end ArchytasZX.LSym
