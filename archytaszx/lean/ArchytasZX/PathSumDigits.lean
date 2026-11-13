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
digits (checked by `tableOK`). `b_frag` and `s_frag` give the binder and splitter as digit
constraints, `f_frag_dig` and `x_frag_dig` give F and X on `2^k` with the digit products
`prodTerms` (`ex_prod_digits`: `ex (o j / 2^k)` from the terms with `a + b < k`), and `NetPSD`,
`NetPSD.eval_eq`, `equal_psd` and `equal_psd2` (one owner table per side, the final sums compared
by `val_agree` on the boundary digits) give a network's value as its fragments' path sum over digits.
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


theorem digit_hi (q j a b : ℕ) (hb : b < 2 ^ q) : digit 2 (q + j) (b + 2 ^ q * a) = digit 2 j a := by
  unfold digit
  rw [pow_add, ← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ (by positivity),
    Nat.div_eq_of_lt hb, zero_add]

/-- The binder's or splitter's fragment: the whole label's digits equal the low part's digits
then the high part's. -/
def bFrag (dig : ℕ → List ℕ) (k a b : ℕ) : Frag := ⟨[], [], eqCons (dig k) (dig b ++ dig a), 1⟩

theorem b_cond (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ) (hown : ownOK own labs dig = true)
    (σ : ℕ → ℕ) (p q k a b : ℕ) (hk : k ∈ labs) (ha : a ∈ labs) (hb : b ∈ labs)
    (hka : (dig k).length = q + p) (hla : (dig a).length = p) (hlb : (dig b).length = q)
    (hσk : σ k < 2 ^ (q + p)) (hσa : σ a < 2 ^ p) (hσb : σ b < 2 ^ q) :
    (∀ c ∈ eqCons (dig k) (dig b ++ dig a), fz (dbits own σ) c = 0) ↔ σ k = σ a * 2 ^ q + σ b := by
  rw [mem_eqCons_iff _ _ _ (by rw [hka, List.length_append, hla, hlb])]
  have hw : σ a * 2 ^ q + σ b < 2 ^ (q + p) := by
    rw [pow_add]
    have : σ a * 2 ^ q + σ b < (σ a + 1) * 2 ^ q := by nlinarith
    calc σ a * 2 ^ q + σ b < (σ a + 1) * 2 ^ q := this
      _ ≤ 2 ^ p * 2 ^ q := Nat.mul_le_mul_right _ hσa
      _ = 2 ^ q * 2 ^ p := by ring
  have hdig : ∀ i (hi : i < q + p),
      ((digit 2 i (σ k) : ℕ) : ZMod 2) = dbits own σ ((dig b ++ dig a)[i]'(by
        rw [List.length_append, hla, hlb]; exact hi)) ↔
      digit 2 i (σ k) = digit 2 i (σ a * 2 ^ q + σ b) := by
    intro i hi
    have hrhs : dbits own σ ((dig b ++ dig a)[i]'(by rw [List.length_append, hla, hlb]; exact hi)) =
        ((digit 2 i (σ a * 2 ^ q + σ b) : ℕ) : ZMod 2) := by
      by_cases hiq : i < q
      · rw [List.getElem_append_left (by rw [hlb]; exact hiq), dbits_get hown σ hb]
        rw [show σ a * 2 ^ q + σ b = σ b + 2 ^ q * σ a by ring,
          digit_low 2 q i (σ b) (σ a) (by norm_num) hiq hσb]
      · have hj : i - q < (dig a).length := by rw [hla]; omega
        rw [List.getElem_append_right (by rw [hlb]; omega)]
        simp only [hlb]
        rw [dbits_get hown σ ha hj]
        rw [show σ a * 2 ^ q + σ b = σ b + 2 ^ q * σ a by ring,
          show i = q + (i - q) by omega, digit_hi q (i - q) (σ a) (σ b) hσb]
        simp
    rw [hrhs]
    have e1 := digit_lt 2 i (σ k) (by norm_num)
    have e2 := digit_lt 2 i (σ a * 2 ^ q + σ b) (by norm_num)
    rcases bit_cases _ e1 with x | x <;> rcases bit_cases _ e2 with y | y <;> rw [x, y] <;> decide
  constructor
  · intro H
    apply eq_of_digits (q + p) _ _ hσk hw
    intro i hi
    have := H i (by rw [hka]; exact hi)
    rw [dbits_get hown σ hk] at this
    exact (hdig i hi).1 this
  · intro H i hi
    have hi' : i < q + p := by rw [← hka]; exact hi
    rw [dbits_get hown σ hk]
    exact (hdig i hi').2 (by rw [H])

theorem b_frag (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hkind : n.kind = .B) (p q : ℕ) (hd : E.dimv n.dim = 2 ^ p)
    (ht : E.dimv n.ph = 2 ^ q) (k a b : ℕ) (ho : n.outs = [k]) (hi : n.ins = [a, b])
    (hk : k ∈ labs) (ha : a ∈ labs) (hb : b ∈ labs)
    (hka : (dig k).length = q + p) (hla : (dig a).length = p) (hlb : (dig b).length = q)
    (M : ℕ) : ArchytasZX.val E σ n = fval M (bFrag dig k a b) (dbits own σ) := by
  have hσk : σ k < 2 ^ (q + p) := hka ▸ hσ k hk
  have hσa : σ a < 2 ^ p := hla ▸ hσ a ha
  have hσb : σ b < 2 ^ q := hlb ▸ hσ b hb
  have hv : ArchytasZX.val E σ n = tB (2 ^ p) (2 ^ q) (σ k) (σ a) (σ b) := by
    unfold ArchytasZX.val
    rw [hkind, ho, hi]
    simp only [hd, ht, List.map_cons, List.map_nil]
  rw [hv]
  unfold fval val bFrag tB
  simp only [bsum, summand, List.append_nil, pn_nil, Int.cast_zero, zero_div, ex_zero, mul_one]
  rw [if_congr (b_cond own labs dig hown σ p q k a b hk ha hb hka hla hlb hσk hσa hσb) rfl rfl]
  simp [hσa, hσb]

theorem s_frag (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hkind : n.kind = .S) (p q : ℕ) (hd : E.dimv n.dim = 2 ^ p)
    (ht : E.dimv n.ph = 2 ^ q) (k a b : ℕ) (ho : n.outs = [a, b]) (hi : n.ins = [k])
    (hk : k ∈ labs) (ha : a ∈ labs) (hb : b ∈ labs)
    (hka : (dig k).length = q + p) (hla : (dig a).length = p) (hlb : (dig b).length = q)
    (M : ℕ) : ArchytasZX.val E σ n = fval M (bFrag dig k a b) (dbits own σ) := by
  have hσk : σ k < 2 ^ (q + p) := hka ▸ hσ k hk
  have hσa : σ a < 2 ^ p := hla ▸ hσ a ha
  have hσb : σ b < 2 ^ q := hlb ▸ hσ b hb
  have hv : ArchytasZX.val E σ n = tB (2 ^ p) (2 ^ q) (σ k) (σ a) (σ b) := by
    unfold ArchytasZX.val
    rw [hkind, ho, hi]
    simp only [hd, ht, List.map_cons, List.map_nil]
  rw [hv]
  unfold fval val bFrag tB
  simp only [bsum, summand, List.append_nil, pn_nil, Int.cast_zero, zero_div, ex_zero, mul_one]
  rw [if_congr (b_cond own labs dig hown σ p q k a b hk ha hb hka hla hlb hσk hσa hσb) rfl rfl]
  simp [hσa, hσb]

theorem val_digits : ∀ (k v : ℕ), v < 2 ^ k →
    (v : ℤ) = ∑ a ∈ range k, (2 : ℤ) ^ a * (digit 2 a v : ℤ)
  | 0, v, hv => by simp at hv; simp [hv]
  | k + 1, v, hv => by
    rw [Finset.sum_range_succ']
    have ih := val_digits k (v / 2) (by rw [pow_succ] at hv; omega)
    have hs : ∀ a, digit 2 (a + 1) v = digit 2 a (v / 2) := by
      intro a; unfold digit; rw [pow_succ', ← Nat.div_div_eq_div_mul]
    simp only [hs, pow_succ]
    have h0 : digit 2 0 v = v % 2 := by simp [digit]
    rw [h0]
    have : ∑ a ∈ range k, (2 : ℤ) ^ a * 2 * (digit 2 a (v / 2) : ℤ) =
        2 * ∑ a ∈ range k, (2 : ℤ) ^ a * (digit 2 a (v / 2) : ℤ) := by
      rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun a _ => by ring
    rw [this, ← ih]
    have := Nat.mod_add_div v 2
    simp only [pow_zero, one_mul]
    omega

/-- The terms `2^(a+b) x_a y_b / 2^k` with `a + b < k`, in units of `1 / M`. -/
def prodTerms (M K : ℕ) (xs ys : List ℕ) : Poly :=
  (List.range xs.length).flatMap fun a => (List.range (xs.length - a)).map fun b =>
    ([xs.getD a 0, ys.getD b 0], 2 ^ (a + b) * (M / K))

theorem pn_flatMap (τ : ℕ → ZMod 2) {α : Type} (f : α → Poly) :
    ∀ L : List α, pn τ (L.flatMap f) = (L.map fun x => pn τ (f x)).sum
  | [] => rfl
  | x :: L => by rw [List.flatMap_cons, pn_append, pn_flatMap τ f L]; simp

theorem list_sum_range {β : Type} [AddCommMonoid β] (f : ℕ → β) :
    ∀ n, ((List.range n).map f).sum = ∑ i ∈ range n, f i
  | 0 => by simp
  | n + 1 => by rw [List.range_succ, List.map_append, List.sum_append, list_sum_range f n,
      Finset.sum_range_succ]; simp

theorem pn_prodTerms (τ : ℕ → ZMod 2) (M K : ℕ) (xs ys : List ℕ) :
    pn τ (prodTerms M K xs ys) = ∑ a ∈ range xs.length, ∑ b ∈ range (xs.length - a),
      ((2 ^ (a + b) * (M / K) : ℕ) : ℤ) * (zq (τ (xs.getD a 0)) * zq (τ (ys.getD b 0))) := by
  unfold prodTerms
  rw [pn_flatMap, list_sum_range]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [← list_sum_range]
  simp only [pn, List.map_map, Function.comp_def]
  congr 1
  apply List.map_congr_left
  intro b _
  simp [mz, zq_mul]

theorem zq_digit (i v : ℕ) : zq ((digit 2 i v : ℕ) : ZMod 2) = (digit 2 i v : ℤ) := by
  have := digit_lt 2 i v (by norm_num)
  rcases bit_cases _ this with e | e <;> rw [e] <;> decide

/-- `ex (o j / 2^k)` as the terms of `prodTerms` at the digits of `o` and `j`. -/
theorem ex_prod_digits (M K q k : ℕ) (hK : K = 2 ^ k) (hq : M = K * q) (hM : 0 < M)
    (xs ys : List ℕ) (hx : xs.length = k) (hy : ys.length = k) (τ : ℕ → ZMod 2) (o j : ℕ)
    (ho : o < 2 ^ k) (hj : j < 2 ^ k)
    (hxo : ∀ a < k, τ (xs.getD a 0) = ((digit 2 a o : ℕ) : ZMod 2))
    (hyj : ∀ b < k, τ (ys.getD b 0) = ((digit 2 b j : ℕ) : ZMod 2)) :
    ex ((pn τ (prodTerms M K xs ys) : ℝ) / M) = ex (((o * j : ℕ) : ℝ) / K) := by
  have hK0 : 0 < K := by rw [hK]; positivity
  have hMK : M / K = q := by rw [hq, Nat.mul_div_cancel_left _ hK0]
  rw [pn_prodTerms, hx, hMK]
  set low : ℤ := ∑ a ∈ range k, ∑ b ∈ range (k - a),
    (2 : ℤ) ^ (a + b) * ((digit 2 a o : ℤ) * (digit 2 b j : ℤ)) with hlow
  have hpn : ∑ a ∈ range k, ∑ b ∈ range (k - a),
      ((2 ^ (a + b) * q : ℕ) : ℤ) * (zq (τ (xs.getD a 0)) * zq (τ (ys.getD b 0))) =
      q * low := by
    rw [hlow, Finset.mul_sum]
    refine Finset.sum_congr rfl fun a ha => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun b hb => ?_
    have ha' : a < k := Finset.mem_range.1 ha
    have hb' : b < k := by have := Finset.mem_range.1 hb; omega
    rw [hxo a ha', hyj b hb', zq_digit, zq_digit]
    push_cast
    ring
  rw [hpn]
  -- o j = low + K R
  obtain ⟨R, hR⟩ : ∃ R : ℤ, ((o * j : ℕ) : ℤ) = low + K * R := by
    have eo := val_digits k o ho
    have ej := val_digits k j hj
    refine ⟨∑ a ∈ range k, ∑ b ∈ Ico (k - a) k,
      (2 : ℤ) ^ (a + b - k) * ((digit 2 a o : ℤ) * (digit 2 b j : ℤ)), ?_⟩
    push_cast
    rw [eo, ej, Finset.sum_mul_sum, hlow, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun a ha => ?_
    have ha' : a < k := Finset.mem_range.1 ha
    rw [← Finset.sum_range_add_sum_Ico _ (show k - a ≤ k by omega), Finset.mul_sum]
    congr 1
    · refine Finset.sum_congr rfl fun b _ => by ring
    · refine Finset.sum_congr rfl fun b hb => ?_
      have hb' := Finset.mem_Ico.1 hb
      rw [hK]
      have : (2 : ℤ) ^ a * (digit 2 a o : ℤ) * ((2 : ℤ) ^ b * (digit 2 b j : ℤ)) =
          (2 : ℤ) ^ (a + b) * ((digit 2 a o : ℤ) * (digit 2 b j : ℤ)) := by ring
      rw [this, show a + b = k + (a + b - k) by omega, pow_add, Nat.add_sub_cancel_left]
      push_cast
      ring_nf
  have hK' : (K : ℝ) ≠ 0 := by exact_mod_cast hK0.ne'
  have hq0 : (q : ℝ) ≠ 0 := by
    intro e
    have : (M : ℝ) = 0 := by rw [hq]; push_cast; rw [e, mul_zero]
    exact absurd this (by exact_mod_cast hM.ne')
  have e1 : ((q * low : ℤ) : ℝ) / M = (low : ℝ) / K := by
    rw [hq]; push_cast; field_simp
  have e2 : (((o * j : ℕ) : ℝ)) / K = (low : ℝ) / K + ((R : ℤ) : ℝ) := by
    have : (((o * j : ℕ) : ℝ)) = (((o * j : ℕ) : ℤ) : ℝ) := by push_cast; rfl
    rw [this, hR]
    push_cast
    field_simp
  rw [e1, e2, ex_int]


/-- F on `2^k`: the phase `o i / 2^k` over the digits, constant `sqrt 2 ^ (-k)`. -/
noncomputable def fFragD (M K k : ℕ) (dig : ℕ → List ℕ) (o i : ℕ) : Frag :=
  ⟨[], prodTerms M K (dig o) (dig i), [], (rt 2)⁻¹ ^ k⟩

theorem f_frag_dig (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hkind : n.kind = .F) (k : ℕ) (hd : E.dimv n.dim = 2 ^ k) (o i : ℕ)
    (ho : n.outs = [o]) (hi : n.ins = [i]) (hol : o ∈ labs) (hil : i ∈ labs)
    (hlo : (dig o).length = k) (hli : (dig i).length = k) (M K q : ℕ) (hK : K = 2 ^ k)
    (hq : M = K * q) (hM : 0 < M) :
    ArchytasZX.val E σ n = fval M (fFragD M K k dig o i) (dbits own σ) := by
  have hσo : σ o < 2 ^ k := hlo ▸ hσ o hol
  have hσi : σ i < 2 ^ k := hli ▸ hσ i hil
  have hv : ArchytasZX.val E σ n = fou (2 ^ k) (σ o) (σ i) := by
    unfold ArchytasZX.val
    rw [hkind, ho, hi]
    simp only [hd, List.map_cons, List.map_nil]
  rw [hv]
  unfold fval val fFragD
  simp only [bsum, summand, List.append_nil, List.not_mem_nil, IsEmpty.forall_iff,
    forall_const, if_true]
  rw [ex_prod_digits M K q k hK hq hM (dig o) (dig i) hlo hli _ (σ o) (σ i) hσo hσi
    (fun a ha => by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by rw [hlo]; exact ha),
        Option.getD_some, dbits_get hown σ hol])
    (fun b hb => by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by rw [hli]; exact hb),
        Option.getD_some, dbits_get hown σ hil])]
  unfold fou
  rw [om_eq_ex, hK]
  have hr : ((Real.sqrt ((2 ^ k : ℕ) : ℝ) : ℝ) : ℂ) = rt 2 ^ k := by
    rw [← ArchytasZX.Sym.rt_pow]
  rw [hr, div_eq_mul_inv, inv_pow, mul_comm]


theorem ex_conj (θ : ℝ) : (starRingEnd ℂ) (ex θ) = ex (-θ) := by
  unfold ex
  rw [← Complex.exp_conj]
  congr 1
  simp only [map_mul, Complex.conj_ofReal, Complex.conj_I, map_ofNat]
  push_cast
  ring

theorem ex_neg_of (a b : ℝ) (h : ex a = ex b) : ex (-a) = ex (-b) := by
  have ha : ex (-a) * ex a = 1 := by rw [← ex_add]; simp [ex_zero]
  have hb : ex (-b) * ex b = 1 := by rw [← ex_add]; simp [ex_zero]
  have hne : ex b ≠ 0 := Complex.exp_ne_zero _
  rw [h] at ha
  exact mul_right_cancel₀ hne (ha.trans hb.symm)

/-- X on `2^k` with the fresh digit variables `ys`: the phase table `P` over `ys`, `y o / 2^k`
per output and `-y i / 2^k` per input over the digits, constant `sqrt 2 ^ (-k legs)`. -/
noncomputable def xFragD (M K k : ℕ) (dig : ℕ → List ℕ) (P : Poly) (ys os is : List ℕ) : Frag :=
  ⟨ys, P ++ (os.flatMap fun o => prodTerms M K (dig o) ys) ++
    (is.flatMap fun i => negP M (prodTerms M K (dig i) ys)), [],
    (rt 2)⁻¹ ^ (k * (os.length + is.length))⟩

theorem x_frag_dig (E : Env) (σ : ℕ → ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < 2 ^ (dig l).length)
    (n : Node) (hkind : n.kind = .X) (k : ℕ) (hd : E.dimv n.dim = 2 ^ k)
    (hlabs : ∀ l ∈ n.outs ++ n.ins, l ∈ labs) (hlen : lensOK dig k (n.outs ++ n.ins) = true)
    (ys : List ℕ) (hys : ys.length = k) (hnd : ys.Nodup)
    (hfresh : ys.all (fun y => (List.lookup y own).isNone) = true)
    (M K q : ℕ) (hK : K = 2 ^ k) (hq : M = K * q) (hM : 0 < M) (P : Poly) (ts : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / M)
    (htab : tableOK M ys P ts = true)
    (hPv : P.all (fun t => t.1.all fun x => decide (x ∈ ys)) = true) :
    ArchytasZX.val E σ n = fval M (xFragD M K k dig P ys n.outs n.ins) (dbits own σ) := by
  subst hK
  have hrt : ((Real.sqrt ((2 ^ k : ℕ) : ℝ) : ℝ) : ℂ) = rt 2 ^ k := by
    rw [← ArchytasZX.Sym.rt_pow]
  have hlen' : ∀ l ∈ n.outs ++ n.ins, (dig l).length = k := by
    intro l hl
    simpa [lensOK] using (List.all_eq_true.1 hlen) l hl
  have hσk : ∀ l ∈ n.outs ++ n.ins, σ l < 2 ^ k := fun l hl =>
    (hlen' l hl) ▸ hσ l (hlabs l hl)
  have hnone : ∀ y ∈ ys, List.lookup y own = none := by
    intro y hy
    simpa using (List.all_eq_true.1 hfresh) y hy
  have hdisj : ∀ l ∈ n.outs ++ n.ins, ∀ x ∈ dig l, x ∉ ys := by
    intro l hl x hx hxy
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
    have := ownOK_get hown (hlabs l hl) hi
    rw [hnone _ hxy] at this
    exact absurd this (by simp)
  have hv : ArchytasZX.val E σ n = tX (2 ^ k) (E.phase n.ph) (n.outs.map σ) (n.ins.map σ) := by
    unfold ArchytasZX.val
    rw [hkind]
    simp only [hd]
  rw [hv]
  unfold fval val xFragD
  dsimp only
  rw [← sum_setBits _ ys (dbits own σ), hys]
  unfold tX
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' : j < 2 ^ k := Finset.mem_range.1 hj
  set τ := setBits ys j (dbits own σ) with hτ
  -- digits under τ
  have hleg : ∀ l ∈ n.outs ++ n.ins, ∀ a < k, τ ((dig l).getD a 0) = ((digit 2 a (σ l) : ℕ) : ZMod 2) := by
    intro l hl a ha
    have hal : a < (dig l).length := by rw [hlen' l hl]; exact ha
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hal, Option.getD_some, hτ,
      setBits_not _ _ _ _ (hdisj l hl _ (List.getElem_mem hal)), dbits_get hown σ (hlabs l hl)]
  have hyj : ∀ b < k, τ (ys.getD b 0) = ((digit 2 b j : ℕ) : ZMod 2) := by
    intro b hb
    have hbl : b < ys.length := by rw [hys]; exact hb
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hbl, Option.getD_some, hτ]
    have := setBits_idx ys j (dbits own σ) hnd b hbl
    simpa using this
  have hG : ∀ o ∈ n.outs, fou (2 ^ k) (σ o) j =
      (rt 2)⁻¹ ^ k * ex ((pn τ (prodTerms M (2 ^ k) (dig o) ys) : ℝ) / M) := by
    intro o ho
    have hol : o ∈ n.outs ++ n.ins := List.mem_append_left _ ho
    rw [ex_prod_digits M _ q k rfl hq hM (dig o) ys (hlen' o hol) hys τ (σ o) j (hσk o hol) hj'
      (hleg o hol) hyj]
    unfold fou
    rw [om_eq_ex, hrt, div_eq_mul_inv, inv_pow, mul_comm]
  have hG' : ∀ i ∈ n.ins, (starRingEnd ℂ) (fou (2 ^ k) (σ i) j) =
      (rt 2)⁻¹ ^ k * ex ((pn τ (negP M (prodTerms M (2 ^ k) (dig i) ys)) : ℝ) / M) := by
    intro i hi
    have hil : i ∈ n.outs ++ n.ins := List.mem_append_right _ hi
    have e := ex_prod_digits M _ q k rfl hq hM (dig i) ys (hlen' i hil) hys τ (σ i) j
      (hσk i hil) hj' (hleg i hil) hyj
    have en : ex ((pn τ (negP M (prodTerms M (2 ^ k) (dig i) ys)) : ℝ) / M) =
        ex (-((pn τ (prodTerms M (2 ^ k) (dig i) ys) : ℝ) / M)) := by
      rw [ex_peq hM (pn_negP τ M hM _)]
      push_cast
      ring_nf
    rw [en, ex_neg_of _ _ e]
    unfold fou
    rw [om_eq_ex, map_div₀, ex_conj, Complex.conj_ofReal, hrt, div_eq_mul_inv, inv_pow, mul_comm]
  -- the summand
  simp only [summand, List.append_nil, List.not_mem_nil, IsEmpty.forall_iff, forall_const,
    if_true, pn_append, pn_flatMap]
  have hPj : ex ((pn τ P : ℝ) / M) = ex (E.phase n.ph j) := by
    rw [hph]
    have hpn : pn τ P = pn (setBits ys j fun _ => 0) P := by
      apply pn_congr
      intro t ht x hx
      have hxy : x ∈ ys := by
        have := List.all_eq_true.1 hPv t ht
        simpa using List.all_eq_true.1 this x hx
      obtain ⟨b, hb, rfl⟩ := List.getElem_of_mem hxy
      rw [hτ, show ys[b] = ys.get ⟨b, hb⟩ from rfl, setBits_idx ys j _ hnd b hb,
        setBits_idx ys j _ hnd b hb]
    have := List.all_eq_true.1 htab j (List.mem_range.2 (by rw [hys]; exact hj'))
    simp only [decide_eq_true_eq] at this
    obtain ⟨r, hr⟩ := Int.dvd_of_emod_eq_zero this
    rw [hpn]
    have e : (((ts.getD j 0 : ℕ) : ℝ)) = (((ts.getD j 0 : ℕ) : ℤ) : ℝ) := by push_cast; rfl
    rw [e]
    first
      | exact ex_peq hM ⟨r, by linarith⟩
      | exact ex_peq hM ⟨-r, by linarith⟩
  have hO : ((n.outs.map σ).map fun o => fou (2 ^ k) o j).prod =
      ((rt 2)⁻¹ ^ k) ^ n.outs.length *
        ex ((((n.outs.map fun o => pn τ (prodTerms M (2 ^ k) (dig o) ys)).sum : ℤ) : ℝ) / M) := by
    rw [List.map_map, ← prod_ex]
    congr 1
    apply List.map_congr_left
    intro o ho
    simpa using hG o ho
  have hI : ((n.ins.map σ).map fun i => (starRingEnd ℂ) (fou (2 ^ k) i j)).prod =
      ((rt 2)⁻¹ ^ k) ^ n.ins.length *
        ex ((((n.ins.map fun i => pn τ (negP M (prodTerms M (2 ^ k) (dig i) ys))).sum : ℤ) : ℝ) /
          M) := by
    rw [List.map_map, ← prod_ex]
    congr 1
    apply List.map_congr_left
    intro i hi
    simpa using hG' i hi
  rw [hO, hI]
  push_cast
  rw [add_div, add_div, ex_add, ex_add, hPj]
  ring


/-! ## Networks over digit variables -/

/-- The network's path sum over digits: the summed labels' digits then the fragments' paths. -/
def netDataD (dig : ℕ → List ℕ) (N : Net) (fs : List Frag) : Data :=
  ⟨N.int.flatMap dig ++ (joinAll fs).path, (joinAll fs).phase, (joinAll fs).cons, []⟩

/-- `NetPS` over digit variables: the owner table, each node's entry as its fragment's value at
the digits of in-range label values, disjoint fragment paths, every label of dimension
`2 ^ |dig l|`, and every label summed or on the boundary. -/
structure NetPSD (E : Env) (N : Net) (M : ℕ) (own : Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (ps : List (Node × Frag)) : Prop where
  ownok : ownOK own labs dig = true
  nodes : N.nodes = ps.map Prod.fst
  node : ∀ p ∈ ps, ∀ σ, (∀ l ∈ labs, σ l < 2 ^ (dig l).length) →
    ArchytasZX.val E σ p.1 = fval M p.2 (dbits own σ)
  disj : (ps.map Prod.snd).Pairwise fun a b => Disj a b ∧ Disj b a
  dims : ∀ l ∈ labs, labDim E l = 2 ^ (dig l).length
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd
  sub : ∀ l ∈ N.int ++ N.bnd, l ∈ labs

theorem NetPSD.eval_eq {E : Env} {N : Net} {M : ℕ} {own : Own} {labs : List ℕ}
    {dig : ℕ → List ℕ} {ps : List (Node × Frag)} (h : NetPSD E N M own labs dig ps)
    (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = val M (joinAll (ps.map Prod.snd)).c (netDataD dig N (ps.map Prod.snd))
      (dbits own (clamp N.bnd σ)) := by
  have he : eval E N σ = eval E N (clamp N.bnd σ) := by
    unfold eval
    apply sumOver_ext (labDim E) N.int (labelsOf N.nodes) _ (prodVal_reads E N.nodes)
    intro x hx hn
    rcases h.cover x hx with hi | hb
    · exact absurd hi hn
    · simp [clamp, hb]
  rw [he]
  unfold eval
  have hint : ∀ l ∈ N.int, l ∈ labs := fun l hl => h.sub l (List.mem_append_left _ hl)
  set F := joinAll (ps.map Prod.snd) with hF
  have hcongr : sumOver (labDim E) N.int (prodVal E N.nodes) (clamp N.bnd σ) =
      sumOver (labDim E) N.int (fun σ' => fval M F (dbits own σ')) (clamp N.bnd σ) := by
    apply sumOver_congr
    intro τ hτ hr
    have hrange : ∀ l ∈ labs, τ l < 2 ^ (dig l).length := by
      intro l hl
      by_cases hli : l ∈ N.int
      · have := hr l hli
        rwa [h.dims l hl] at this
      · rw [hτ l hli]
        unfold clamp
        split_ifs with hlb
        · have := hσ l hlb
          rwa [h.dims l hl] at this
        · positivity
    rw [hF, fval_joinAll M _ _ h.disj, prodVal, h.nodes, List.map_map, List.map_map]
    congr 1
    apply List.map_congr_left
    intro p hp
    exact h.node p hp τ hrange
  rw [hcongr, sumOver_dbits (labDim E) h.ownok (fval M F) N.int hint
    (fun l hl => h.dims l (hint l hl))]
  unfold fval val netDataD
  rw [bsum_append]
  rfl

/-- Two networks with one boundary whose digit path sums, scaled by `s₁` and `s₂`, agree at
every Boolean assignment. -/
theorem equal_psd {E : Env} {N₁ N₂ : Net} {M : ℕ} {own : Own} {labs : List ℕ}
    {dig : ℕ → List ℕ} {ps₁ ps₂ : List (Node × Frag)} (s₁ s₂ : ℂ)
    (h₁ : NetPSD E N₁ M own labs dig ps₁) (h₂ : NetPSD E N₂ M own labs dig ps₂)
    (hb : N₁.bnd = N₂.bnd)
    (hv : ∀ τ, s₁ * val M (joinAll (ps₁.map Prod.snd)).c (netDataD dig N₁ (ps₁.map Prod.snd)) τ =
      s₂ * val M (joinAll (ps₂.map Prod.snd)).c (netDataD dig N₂ (ps₂.map Prod.snd)) τ) :
    Equal E s₁ N₁ s₂ N₂ := by
  refine ⟨hb ▸ List.Perm.refl _, fun σ hσ => ?_⟩
  have hσ₂ : InRange E N₂.bnd σ := hb ▸ hσ
  rw [h₁.eval_eq σ hσ, h₂.eval_eq σ hσ₂, ← hb]
  exact hv _


/-- Whether every variable of the phase, constraints and rows of `s` is in `vs`. -/
def varsIn (vs : List ℕ) (s : Data) : Bool :=
  (pvars s.phase ++ fvars (s.cons ++ s.rows)).all fun x => decide (x ∈ vs)

theorem mz_agree (τ τ' : ℕ → ZMod 2) (m : Mono) (h : ∀ x ∈ m, τ x = τ' x) : mz τ m = mz τ' m := by
  unfold mz
  congr 1
  exact List.map_congr_left h

theorem fz_agree (τ τ' : ℕ → ZMod 2) (q : F2) (h : ∀ m ∈ q, ∀ x ∈ m, τ x = τ' x) :
    fz τ q = fz τ' q := by
  unfold fz
  congr 1
  exact List.map_congr_left fun m hm => mz_agree τ τ' m (h m hm)

/-- A path-free sum over variables in `vs`, at two assignments agreeing on `vs`. -/
theorem val_agree (M : ℕ) (c : ℂ) (s : Data) (vs : List ℕ) (τ τ' : ℕ → ZMod 2)
    (hp : s.path = []) (hv : varsIn vs s = true) (h : ∀ x ∈ vs, τ x = τ' x) :
    val M c s τ = val M c s τ' := by
  unfold val
  rw [hp]
  simp only [bsum]
  have hall := List.all_eq_true.1 hv
  have hx : ∀ x ∈ pvars s.phase ++ fvars (s.cons ++ s.rows), τ x = τ' x := fun x hx =>
    h x (by simpa using hall x hx)
  unfold summand
  have hpn : pn τ s.phase = pn τ' s.phase := by
    apply pn_congr
    intro t ht x hxt
    exact hx x (List.mem_append_left _ (List.mem_flatMap.2 ⟨t, ht, hxt⟩))
  have hcond : (∀ q ∈ s.cons ++ s.rows, fz τ q = 0) ↔ (∀ q ∈ s.cons ++ s.rows, fz τ' q = 0) := by
    have hq : ∀ q ∈ s.cons ++ s.rows, fz τ q = fz τ' q := fun q hq =>
      fz_agree τ τ' q fun m hm x hxm => hx x (List.mem_append_right _
        (List.mem_flatMap.2 ⟨q, hq, List.mem_flatMap.2 ⟨m, hm, hxm⟩⟩))
    exact ⟨fun H q h' => (hq q h') ▸ H q h', fun H q h' => (hq q h').symm ▸ H q h'⟩
  rw [hpn, if_congr hcond rfl rfl]

theorem dbits_agree (own₁ own₂ : Own) (σ : ℕ → ℕ) (vs : List ℕ)
    (h : vs.all (fun x => decide (List.lookup x own₁ = List.lookup x own₂)) = true) :
    ∀ x ∈ vs, dbits own₁ σ x = dbits own₂ σ x := by
  intro x hx
  have := List.all_eq_true.1 h x hx
  simp only [decide_eq_true_eq] at this
  unfold dbits
  rw [this]

/-- `equal_psd` with an owner table, labels and digits per side. -/
theorem equal_psd2 {E : Env} {N₁ N₂ : Net} {M : ℕ} {own₁ own₂ : Own} {labs₁ labs₂ : List ℕ}
    {dig₁ dig₂ : ℕ → List ℕ} {ps₁ ps₂ : List (Node × Frag)} (s₁ s₂ : ℂ)
    (h₁ : NetPSD E N₁ M own₁ labs₁ dig₁ ps₁) (h₂ : NetPSD E N₂ M own₂ labs₂ dig₂ ps₂)
    (hb : N₁.bnd = N₂.bnd)
    (hv : ∀ σ, s₁ * val M (joinAll (ps₁.map Prod.snd)).c (netDataD dig₁ N₁ (ps₁.map Prod.snd))
        (dbits own₁ (clamp N₁.bnd σ)) =
      s₂ * val M (joinAll (ps₂.map Prod.snd)).c (netDataD dig₂ N₂ (ps₂.map Prod.snd))
        (dbits own₂ (clamp N₂.bnd σ))) :
    Equal E s₁ N₁ s₂ N₂ := by
  refine ⟨hb ▸ List.Perm.refl _, fun σ hσ => ?_⟩
  have hσ₂ : InRange E N₂.bnd σ := hb ▸ hσ
  rw [h₁.eval_eq σ hσ, h₂.eval_eq σ hσ₂]
  exact hv σ

end PSem
end ArchytasZX
