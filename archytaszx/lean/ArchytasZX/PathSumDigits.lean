/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumNet
import ArchytasZX.Positional

/-!
# Labels of dimension 2^k as digit variables

A label of dimension `2^k` is `k` Boolean digit variables `dig l`, low digit first; `own` lists
each digit variable with its label and index (`ownOK`), and `dbits own σ` reads the digit
variables from label values. `sumOver_dbits` turns the network's sum over such labels into a
sum over their digits, and `z_frag_dig` gives a Z spider on `2^k` as a fragment: digit `j` of
every leg equal to an anchor leg's digit `j`, the phase table interpolated over the anchor
digits (checked by `tableOK`).
-/

namespace ArchytasZX
namespace PSem

open Complex Finset

/-! ## Labels of dimension 2^k as digit variables -/

/-- `τ` with the variables `ds` set to the binary digits of `j`, low digit first. -/
def setBits : List ℕ → ℕ → (ℕ → ZMod 2) → ℕ → ZMod 2
  | [], _, τ => τ
  | x :: ds, j, τ => setBits ds (j / 2) (Function.update τ x ((j % 2 : ℕ) : ZMod 2))

theorem sum_setBits (G : (ℕ → ZMod 2) → ℂ) :
    ∀ (ds : List ℕ) (τ : ℕ → ZMod 2),
      ∑ j ∈ range (2 ^ ds.length), G (setBits ds j τ) = bsum ds G τ
  | [], τ => by simp [setBits, bsum]
  | x :: ds, τ => by
    rw [List.length_cons, pow_succ', sum_range_mul 2 (2 ^ ds.length)]
    rw [Finset.sum_range_succ, Finset.sum_range_one]
    simp only [setBits, bsum]
    have h0 : ∀ y, (0 + 2 * y) / 2 = y ∧ (0 + 2 * y) % 2 = 0 := by intro y; omega
    have h1 : ∀ y, (1 + 2 * y) / 2 = y ∧ (1 + 2 * y) % 2 = 1 := by intro y; omega
    simp only [fun y => (h0 y).1, fun y => (h0 y).2, fun y => (h1 y).1, fun y => (h1 y).2]
    rw [sum_setBits G ds, sum_setBits G ds]
    simp

theorem setBits_not (x : ℕ) : ∀ (ds : List ℕ) (j : ℕ) (τ : ℕ → ZMod 2), x ∉ ds →
    setBits ds j τ x = τ x
  | [], _, _, _ => rfl
  | y :: ds, j, τ, h => by
    simp only [setBits]
    rw [setBits_not x ds _ _ (fun e => h (List.mem_cons_of_mem _ e))]
    exact Function.update_of_ne (fun e : x = y => h (by rw [e]; exact List.mem_cons_self)) _ _

theorem setBits_idx : ∀ (ds : List ℕ) (j : ℕ) (τ : ℕ → ZMod 2), ds.Nodup →
    ∀ i (hi : i < ds.length), setBits ds j τ (ds.get ⟨i, hi⟩) = ((digit 2 i j : ℕ) : ZMod 2)
  | [], _, _, _, i, hi => by simp at hi
  | y :: ds, j, τ, hnd, i, hi => by
    obtain ⟨hy, hnd'⟩ := List.nodup_cons.1 hnd
    simp only [setBits]
    rcases i with _ | i
    · simp only [List.get_eq_getElem, List.getElem_cons_zero]
      rw [setBits_not y ds _ _ hy, Function.update_self]
      simp [digit]
    · simp only [List.get_eq_getElem, List.getElem_cons_succ]
      have hi' : i < ds.length := by simpa using hi
      have := setBits_idx ds (j / 2) (Function.update τ y ((j % 2 : ℕ) : ZMod 2)) hnd' i hi'
      simp only [List.get_eq_getElem] at this
      rw [this]
      have hd : digit 2 (i + 1) j = digit 2 i (j / 2) := by
        unfold digit
        rw [pow_succ', ← Nat.div_div_eq_div_mul]
      rw [hd]

/-- A digit owner table: each digit variable with its label and digit index. -/
abbrev Own := List (ℕ × ℕ × ℕ)

/-- The Boolean assignment of digit variables read from label values. -/
def dbits (own : Own) (σ : ℕ → ℕ) (x : ℕ) : ZMod 2 :=
  match List.lookup x own with
  | some p => ((digit 2 p.2 (σ p.1) : ℕ) : ZMod 2)
  | none => 0

/-- Whether `own` lists exactly the digit variables `dig l` of the labels `labs`, each once. -/
def ownOK (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ) : Bool :=
  decide ((own.map Prod.fst).Nodup) && decide (labs.Nodup) &&
    own.all (fun e => decide (e.2.1 ∈ labs) && decide ((dig e.2.1)[e.2.2]? = some e.1)) &&
    labs.all (fun l => decide ((dig l).Nodup) &&
      (List.range (dig l).length).all fun i => decide ((((dig l)[i]?).bind fun x => List.lookup x own) =
        some (l, i)))

theorem mem_of_lookup {x : ℕ} {v : ℕ × ℕ} : ∀ {own : Own}, List.lookup x own = some v → (x, v) ∈ own
  | [], h => by simp at h
  | e :: own, h => by
    by_cases hx : x = e.1
    · subst hx
      simp only [List.lookup, beq_self_eq_true, Option.some.injEq] at h
      subst h
      exact List.mem_cons_self
    · have : List.lookup x (e :: own) = List.lookup x own := by
        have hb : (x == e.1) = false := by simpa using hx
        simp only [List.lookup, hb]
      rw [this] at h
      exact List.mem_cons_of_mem _ (mem_of_lookup h)

theorem ownOK_lookup {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : ownOK own labs dig = true) {x l i : ℕ} (hx : List.lookup x own = some (l, i)) :
    l ∈ labs ∧ (dig l)[i]? = some x := by
  simp only [ownOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨_, _⟩, hall⟩, _⟩ := h
  have hmem : (x, (l, i)) ∈ own := mem_of_lookup hx
  exact hall _ hmem

theorem ownOK_get {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : ownOK own labs dig = true) {l : ℕ} (hl : l ∈ labs) {i : ℕ} (hi : i < (dig l).length) :
    List.lookup ((dig l)[i]) own = some (l, i) := by
  simp only [ownOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨_, hlabs⟩ := h
  have := (hlabs l hl).2 i (List.mem_range.2 hi)
  rw [List.getElem?_eq_getElem hi] at this
  simpa using this

theorem ownOK_nodup {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : ownOK own labs dig = true) {l : ℕ} (hl : l ∈ labs) : (dig l).Nodup := by
  simp only [ownOK, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  exact (h.2 l hl).1

theorem dbits_get {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : ownOK own labs dig = true) (σ : ℕ → ℕ) {l : ℕ} (hl : l ∈ labs) {i : ℕ}
    (hi : i < (dig l).length) : dbits own σ ((dig l)[i]) = ((digit 2 i (σ l) : ℕ) : ZMod 2) := by
  unfold dbits
  rw [ownOK_get h hl hi]

theorem dbits_update {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : ownOK own labs dig = true) (σ : ℕ → ℕ) {l : ℕ} (hl : l ∈ labs) (j : ℕ) :
    dbits own (Function.update σ l j) = setBits (dig l) j (dbits own σ) := by
  funext x
  have hnd := ownOK_nodup h hl
  by_cases hx : x ∈ dig l
  · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
    have := setBits_idx (dig l) j (dbits own σ) hnd i hi
    simp only [List.get_eq_getElem] at this
    rw [this, dbits_get h _ hl hi, Function.update_self]
  · rw [setBits_not x _ _ _ hx]
    unfold dbits
    rcases hlk : List.lookup x own with _ | ⟨l', i⟩
    · rfl
    · obtain ⟨_, hget⟩ := ownOK_lookup h hlk
      have hne : l' ≠ l := by
        rintro rfl
        exact hx (List.mem_of_getElem? hget)
      simp only
      rw [Function.update_of_ne hne]

theorem sumOver_dbits (D : ℕ → ℕ) {own : Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : ownOK own labs dig = true) (g : (ℕ → ZMod 2) → ℂ) :
    ∀ L : List ℕ, (∀ l ∈ L, l ∈ labs) → (∀ l ∈ L, D l = 2 ^ (dig l).length) → ∀ σ,
      sumOver D L (fun σ' => g (dbits own σ')) σ = bsum (L.flatMap dig) g (dbits own σ)
  | [], _, _, σ => rfl
  | l :: L, hL, hD, σ => by
    simp only [sumOver, List.flatMap_cons]
    rw [hD l List.mem_cons_self, bsum_append]
    have ih := sumOver_dbits D h g L (fun x hx => hL x (List.mem_cons_of_mem _ hx))
      (fun x hx => hD x (List.mem_cons_of_mem _ hx))
    simp only [ih, dbits_update h _ (hL l List.mem_cons_self)]
    exact sum_setBits _ _ _

/-- Values below `2^k` with equal first `k` digits are equal. -/
theorem eq_of_digits : ∀ (k v w : ℕ), v < 2 ^ k → w < 2 ^ k →
    (∀ i < k, digit 2 i v = digit 2 i w) → v = w
  | 0, v, w, hv, hw, _ => by simp at hv hw; omega
  | k + 1, v, w, hv, hw, h => by
    have h0 := h 0 (by omega)
    have hs : ∀ i < k, digit 2 i (v / 2) = digit 2 i (w / 2) := by
      intro i hi
      have := h (i + 1) (by omega)
      unfold digit at this ⊢
      rwa [pow_succ', ← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul] at this
    have := eq_of_digits k (v / 2) (w / 2) (by rw [pow_succ] at hv; omega)
      (by rw [pow_succ] at hw; omega) hs
    simp only [digit, pow_zero, Nat.div_one] at h0
    omega

theorem digit_bit (i v : ℕ) : ((digit 2 i v : ℕ) : ZMod 2) = 0 ↔ digit 2 i v = 0 := by
  have := digit_lt 2 i v (by norm_num)
  rcases bit_cases _ this with e | e <;> rw [e] <;> decide

theorem bits_eq' (a b : ℕ) (ha : a < 2) (hb : b < 2) :
    ((a : ZMod 2) + (b : ZMod 2) = 0) ↔ a = b := by
  rw [bits_eq a b ha hb]
  exact eq_comm

/-! ## Fragments over digits -/

/-- The digit-variable constraints `x + y = 0` of two digit lists. -/
def eqCons (xs ys : List ℕ) : List F2 := List.zipWith (fun x y => [[x], [y]]) xs ys

/-- Whether `ls`'s digit lists all have length `k`. -/
def lensOK (dig : ℕ → List ℕ) (k : ℕ) (ls : List ℕ) : Bool := ls.all fun l => decide ((dig l).length = k)

theorem tZ_val (d : ℕ) (φ : ℕ → ℝ) (v : ℕ) (vs : List ℕ) (hv : v < d) :
    tZ d φ (v :: vs) = if ∀ w ∈ vs, w = v then ex (φ v) else 0 := by
  unfold tZ
  rw [Finset.sum_eq_single v]
  · simp
  · intro b _ hb
    rw [if_neg]
    simp only [List.mem_cons, forall_eq_or_imp, not_and]
    intro h
    exact absurd h.symm hb
  · intro h
    exact absurd (Finset.mem_range.2 hv) h

theorem pn_congr (τ τ' : ℕ → ZMod 2) :
    ∀ P : Poly, (∀ t ∈ P, ∀ x ∈ t.1, τ x = τ' x) → pn τ P = pn τ' P
  | [] => fun _ => rfl
  | t :: P => fun h => by
    rw [pn_cons, pn_cons, pn_congr τ τ' P (fun u hu => h u (List.mem_cons_of_mem _ hu))]
    have : mz τ t.1 = mz τ' t.1 := by
      unfold mz
      congr 1
      exact List.map_congr_left fun x hx => h t List.mem_cons_self x hx
    rw [this]

/-- The anchor digit variables: digit `j` of label `as[j]`. -/
def anchorD (dig : ℕ → List ℕ) (as : List ℕ) : List ℕ := as.mapIdx fun j a => (dig a).getD j 0

/-- Whether `P` takes the value `ts[v]` (mod `M`) at the digits of every `v < 2^|ds|` on `ds`. -/
def tableOK (M : ℕ) (ds : List ℕ) (P : Poly) (ts : List ℕ) : Bool :=
  (List.range (2 ^ ds.length)).all fun v =>
    decide ((pn (setBits ds v fun _ => 0) P - (ts.getD v 0 : ℤ)) % M = 0)

/-- The Z spider's fragment over digits: digit `j` of each leg equal to the anchor's digit `j`,
the phase `P` over the anchor digits. -/
def zFragD (dig : ℕ → List ℕ) (as ls : List ℕ) (P : Poly) : Frag :=
  ⟨[], P, ls.flatMap fun l => eqCons (anchorD dig as) (dig l), 1⟩

theorem fz_pair (τ : ℕ → ZMod 2) (x y : ℕ) : fz τ [[x], [y]] = τ x + τ y := by simp [fz, mz]

theorem mem_eqCons_iff (τ : ℕ → ZMod 2) (xs ys : List ℕ) (h : xs.length = ys.length) :
    (∀ q ∈ eqCons xs ys, fz τ q = 0) ↔ ∀ i (hi : i < xs.length),
      τ (xs[i]) = τ (ys[i]'(h ▸ hi)) := by
  unfold eqCons
  constructor
  · intro H i hi
    have hmem : [[xs[i]], [ys[i]'(h ▸ hi)]] ∈ List.zipWith (fun x y => [[x], [y]]) xs ys := by
      rw [List.mem_iff_getElem]
      refine ⟨i, by rw [List.length_zipWith]; exact lt_min hi (h ▸ hi), ?_⟩
      simp
    have := H _ hmem
    rw [fz_pair] at this
    exact z2_eq_of_add _ _ this
  · intro H q hq
    rw [List.mem_iff_getElem] at hq
    obtain ⟨i, hi, rfl⟩ := hq
    simp only [List.length_zipWith, lt_min_iff] at hi
    simp only [List.getElem_zipWith, fz_pair]
    rw [H i hi.1]
    exact z2_add_self _

theorem z_frag_dig (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hk : n.kind = .Z) (k : ℕ) (hd : E.dimv n.dim = 2 ^ k)
    (hlabs : ∀ l ∈ n.outs ++ n.ins, l ∈ labs) (hlen : lensOK dig k (n.outs ++ n.ins) = true)
    (as : List ℕ) (has : as.length = k) (hasl : ∀ a ∈ as, a ∈ n.outs ++ n.ins)
    (l0 : ℕ) (hl0 : l0 ∈ n.outs ++ n.ins)
    (M : ℕ) (hM : 0 < M) (P : Poly) (ts : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / M)
    (htab : tableOK M (anchorD dig as) P ts = true)
    (hnd : (anchorD dig as).Nodup)
    (hPv : P.all (fun t => t.1.all fun x => decide (x ∈ anchorD dig as)) = true) :
    ArchytasZX.val E σ n = fval M (zFragD dig as (n.outs ++ n.ins) P) (dbits own σ) := by
  set ls := n.outs ++ n.ins with hls
  have hlen' : ∀ l ∈ ls, (dig l).length = k := by
    intro l hl
    simpa [lensOK] using (List.all_eq_true.1 hlen) l hl
  have hσk : ∀ l ∈ ls, σ l < 2 ^ k := fun l hl => (hlen' l hl) ▸ hσ l (hlabs l hl)
  have hv : ArchytasZX.val E σ n = tZ (2 ^ k) (E.phase n.ph) (σ l0 :: ls.map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd, List.map_append]
    apply tZ_congr
    intro x
    simp only [List.mem_cons, ← List.map_append]
    constructor
    · exact Or.inr
    · rintro (rfl | hx)
      · exact List.mem_map.2 ⟨l0, hl0, rfl⟩
      · exact hx
  rw [hv, tZ_val _ _ _ _ (hσk l0 hl0)]
  have hal : (anchorD dig as).length = k := by simp [anchorD, has]
  have hanc : ∀ j (hj : j < k), (anchorD dig as)[j]'(hal ▸ hj) =
      (dig (as[j]'(has ▸ hj)))[j]'(by
        rw [hlen' _ (hasl _ (List.getElem_mem (has ▸ hj)))]; exact hj) := by
    intro j hj
    have hlj : j < (dig (as[j]'(has ▸ hj))).length := by
      rw [hlen' _ (hasl _ (List.getElem_mem (has ▸ hj)))]; exact hj
    simp only [anchorD, List.getElem_mapIdx, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem hlj, Option.getD_some]
  have hdb : ∀ l (hl : l ∈ ls) j (hj : j < k),
      dbits own σ ((dig l)[j]'(by rw [hlen' l hl]; exact hj)) =
      ((digit 2 j (σ l) : ℕ) : ZMod 2) := by
    intro l hl j hj
    exact dbits_get hown σ (hlabs l hl) _
  have hdba : ∀ j (hj : j < k), dbits own σ ((anchorD dig as)[j]'(hal ▸ hj)) =
      ((digit 2 j (σ (as[j]'(has ▸ hj))) : ℕ) : ZMod 2) := by
    intro j hj
    rw [hanc j hj]
    exact hdb _ (hasl _ (List.getElem_mem (has ▸ hj))) j hj
  have hcond : (∀ q ∈ (zFragD dig as ls P).cons, fz (dbits own σ) q = 0) ↔
      ∀ w ∈ ls.map σ, w = σ l0 := by
    simp only [zFragD, List.mem_flatMap, forall_exists_index, and_imp]
    have hpair : ∀ l ∈ ls, (∀ q ∈ eqCons (anchorD dig as) (dig l), fz (dbits own σ) q = 0) ↔
        ∀ j < k, digit 2 j (σ (as.getD j 0)) = digit 2 j (σ l) := by
      intro l hl
      rw [mem_eqCons_iff _ _ _ (by rw [hal, hlen' l hl])]
      constructor
      · intro H j hj
        have := H j (hal ▸ hj)
        rw [hdba j hj, hdb l hl j hj] at this
        have e1 := digit_lt 2 j (σ (as[j]'(has ▸ hj))) (by norm_num)
        have e2 := digit_lt 2 j (σ l) (by norm_num)
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (has ▸ hj), Option.getD_some]
        rcases bit_cases _ e1 with a | a <;> rcases bit_cases _ e2 with b | b <;>
          rw [a, b] at this ⊢ <;> first | rfl | exact absurd this (by decide)
      · intro H j hj
        have hj' : j < k := hal ▸ hj
        rw [hdba j hj', hdb l hl j hj']
        have := H j hj'
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (has ▸ hj'), Option.getD_some]
          at this
        rw [this]
    constructor
    · intro H w hw
      obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hw
      have h1 := (hpair l hl).1 fun q hq => H q l hl hq
      have h2 := (hpair l0 hl0).1 fun q hq => H q l0 hl0 hq
      exact eq_of_digits k _ _ (hσk l hl) (hσk l0 hl0) fun j hj => (h1 j hj).symm.trans (h2 j hj)
    · intro H q l hl hq
      refine (hpair l hl).2 (fun j hj => ?_) q hq
      have ha : as.getD j 0 ∈ ls := by
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (has ▸ hj), Option.getD_some]
        exact hasl _ (List.getElem_mem (has ▸ hj))
      rw [H _ (List.mem_map.2 ⟨_, ha, rfl⟩), H _ (List.mem_map.2 ⟨l, hl, rfl⟩)]
  unfold fval val
  simp only [bsum, summand, List.append_nil, zFragD]
  have hcond' := hcond
  simp only [zFragD] at hcond'
  rw [if_congr hcond' rfl rfl]
  split_ifs with hc
  · rw [one_mul]
    -- the phase at the anchor digits of σ l0
    have hset : ∀ x ∈ anchorD dig as, dbits own σ x = setBits (anchorD dig as) (σ l0)
        (fun _ => 0) x := by
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
      have hj' : j < k := hal ▸ hj
      have := setBits_idx (anchorD dig as) (σ l0) (fun _ => 0) hnd j hj
      simp only [List.get_eq_getElem] at this
      rw [this, hdba j hj']
      have ha : as[j]'(has ▸ hj') ∈ ls := hasl _ (List.getElem_mem (has ▸ hj'))
      rw [hc _ (List.mem_map.2 ⟨_, ha, rfl⟩)]
    have hpn : pn (dbits own σ) P = pn (setBits (anchorD dig as) (σ l0) fun _ => 0) P := by
      apply pn_congr
      intro t ht x hx
      apply hset
      have := List.all_eq_true.1 hPv t ht
      simpa using List.all_eq_true.1 this x hx
    rw [hpn, hph]
    have hv0 : σ l0 < 2 ^ (anchorD dig as).length := by rw [hal]; exact hσk l0 hl0
    have := List.all_eq_true.1 htab (σ l0) (List.mem_range.2 hv0)
    simp only [decide_eq_true_eq] at this
    obtain ⟨q, hq⟩ := Int.dvd_of_emod_eq_zero this
    have e : (((ts.getD (σ l0) 0 : ℕ) : ℝ)) = (((ts.getD (σ l0) 0 : ℕ) : ℤ) : ℝ) := by
      push_cast; rfl
    rw [e]
    first
      | exact ex_peq hM ⟨q, by linarith⟩
      | exact ex_peq hM ⟨-q, by linarith⟩
      | exact (ex_peq hM ⟨q, by linarith⟩).symm
      | exact (ex_peq hM ⟨-q, by linarith⟩).symm
  · rfl

end PSem
end ArchytasZX
