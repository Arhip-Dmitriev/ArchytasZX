/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumQuditNet

/-!
# Qudit labels of dimension p^k as digit variables over Z_p

A label of dimension `p^k` is `k` digit variables over `ZMod p`, low digit first; `own` lists
each digit variable with its label and index (`PSem.ownOK`), and `qdig p own σ` reads the digit
variables from label values. `sumOver_qdig` turns the network's sum over such labels into a sum
over their digits. `z_frag_qd` gives a Z spider on `p^k` as a fragment (digit `j` of every leg
equal to an anchor leg's digit `j`, the phase table a polynomial in the anchor digits checked by
`tabQD`), `b_frag_q` and `s_frag_q` the binder and splitter as digit constraints, and `lift_q`
moves a fragment of a node on dimension `p` (`z_frag_q`, `x_frag_q`, `f_frag_q`) to digit
variables. `QNetPSD.eval_eq` gives a network's value as its fragments' path sum over digits, and
`equal_qpsd2` gives `Equal` from the two sums agreeing at every assignment, the second side read
on boundary digits whose owners agree.
-/

namespace ArchytasZX

theorem PSem.ownOK_get? {own : PSem.Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : PSem.ownOK own labs dig = true) {l : ℕ} (hl : l ∈ labs) {i x : ℕ}
    (hx : (dig l)[i]? = some x) : List.lookup x own = some (l, i) := by
  obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.1 hx
  exact PSem.ownOK_get h hl hi

namespace QSem

open Complex Finset

section Digits

variable {p : ℕ} [NeZero p]

/-! ## Digit assignments -/

/-- `τ` with the variables `ds` set to the base-`p` digits of `j`, low digit first. -/
def setDigs (p : ℕ) : List ℕ → ℕ → (ℕ → ZMod p) → ℕ → ZMod p
  | [], _, τ => τ
  | x :: ds, j, τ => setDigs p ds (j / p) (Function.update τ x ((j % p : ℕ) : ZMod p))

theorem qsum_setDigs (G : (ℕ → ZMod p) → ℂ) :
    ∀ (ds : List ℕ) (τ : ℕ → ZMod p),
      ∑ j ∈ range (p ^ ds.length), G (setDigs p ds j τ) = qsum ds G τ
  | [], τ => by simp [setDigs, qsum]
  | x :: ds, τ => by
    rw [List.length_cons, pow_succ', sum_range_mul p (p ^ ds.length)]
    simp only [setDigs, qsum]
    have hR : ∑ b : ZMod p, qsum ds G (Function.update τ x b) =
        ∑ a ∈ range p, qsum ds G (Function.update τ x ((a : ℕ) : ZMod p)) := by
      rw [← zsum_range (fun a => qsum ds G (Function.update τ x ((a : ℕ) : ZMod p)))]
      simp only [ZMod.natCast_zmod_val]
    rw [hR]
    apply Finset.sum_congr rfl
    intro a ha
    have hp : 0 < p := NeZero.pos p
    have ha' : a < p := Finset.mem_range.1 ha
    have h1 : ∀ y, (a + p * y) / p = y := fun y => by
      rw [Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt ha', zero_add]
    have h2 : ∀ y, (a + p * y) % p = a := fun y => by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ha']
    simp only [h1, h2]
    exact qsum_setDigs G ds _

omit [NeZero p] in
theorem setDigs_not (x : ℕ) : ∀ (ds : List ℕ) (j : ℕ) (τ : ℕ → ZMod p), x ∉ ds →
    setDigs p ds j τ x = τ x
  | [], _, _, _ => rfl
  | y :: ds, j, τ, h => by
    simp only [setDigs]
    rw [setDigs_not x ds _ _ (fun e => h (List.mem_cons_of_mem _ e))]
    exact Function.update_of_ne (fun e : x = y => h (by rw [e]; exact List.mem_cons_self)) _ _

omit [NeZero p] in
theorem setDigs_idx : ∀ (ds : List ℕ) (j : ℕ) (τ : ℕ → ZMod p), ds.Nodup →
    ∀ i (hi : i < ds.length), setDigs p ds j τ (ds[i]) = ((digit p i j : ℕ) : ZMod p)
  | [], _, _, _, i, hi => by simp at hi
  | y :: ds, j, τ, hnd, i, hi => by
    obtain ⟨hy, hnd'⟩ := List.nodup_cons.1 hnd
    simp only [setDigs]
    rcases i with _ | i
    · simp only [List.getElem_cons_zero]
      rw [setDigs_not y ds _ _ hy, Function.update_self]
      simp [digit]
    · simp only [List.getElem_cons_succ]
      have hi' : i < ds.length := by simpa using hi
      rw [setDigs_idx ds (j / p) _ hnd' i hi']
      have hd : digit p (i + 1) j = digit p i (j / p) := by
        unfold digit
        rw [pow_succ', ← Nat.div_div_eq_div_mul]
      rw [hd]

/-- The assignment of digit variables read from label values. -/
def qdig (p : ℕ) (own : PSem.Own) (σ : ℕ → ℕ) (x : ℕ) : ZMod p :=
  match List.lookup x own with
  | some e => ((digit p e.2 (σ e.1) : ℕ) : ZMod p)
  | none => 0

omit [NeZero p] in
theorem qdig_of_lookup (own : PSem.Own) (σ : ℕ → ℕ) {x l i : ℕ}
    (h : List.lookup x own = some (l, i)) : qdig p own σ x = ((digit p i (σ l) : ℕ) : ZMod p) := by
  unfold qdig
  rw [h]

omit [NeZero p] in
theorem qdig_get {own : PSem.Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : PSem.ownOK own labs dig = true) (σ : ℕ → ℕ) {l : ℕ} (hl : l ∈ labs) {i : ℕ}
    (hi : i < (dig l).length) :
    qdig p own σ ((dig l)[i]) = ((digit p i (σ l) : ℕ) : ZMod p) :=
  qdig_of_lookup own σ (PSem.ownOK_get h hl hi)

omit [NeZero p] in
theorem qdig_update {own : PSem.Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : PSem.ownOK own labs dig = true) (σ : ℕ → ℕ) {l : ℕ} (hl : l ∈ labs) (j : ℕ) :
    qdig p own (Function.update σ l j) = setDigs p (dig l) j (qdig p own σ) := by
  funext x
  have hnd := PSem.ownOK_nodup h hl
  by_cases hx : x ∈ dig l
  · obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
    rw [setDigs_idx (dig l) j (qdig p own σ) hnd i hi, qdig_get h _ hl hi, Function.update_self]
  · rw [setDigs_not x _ _ _ hx]
    unfold qdig
    rcases hlk : List.lookup x own with _ | ⟨l', i⟩
    · rfl
    · obtain ⟨_, hget⟩ := PSem.ownOK_lookup h hlk
      have hne : l' ≠ l := by
        rintro rfl
        exact hx (List.mem_of_getElem? hget)
      simp only
      rw [Function.update_of_ne hne]

theorem sumOver_qdig (D : ℕ → ℕ) {own : PSem.Own} {labs : List ℕ} {dig : ℕ → List ℕ}
    (h : PSem.ownOK own labs dig = true) (g : (ℕ → ZMod p) → ℂ) :
    ∀ L : List ℕ, (∀ l ∈ L, l ∈ labs) → (∀ l ∈ L, D l = p ^ (dig l).length) → ∀ σ,
      sumOver D L (fun σ' => g (qdig p own σ')) σ = qsum (L.flatMap dig) g (qdig p own σ)
  | [], _, _, σ => rfl
  | l :: L, hL, hD, σ => by
    simp only [sumOver, List.flatMap_cons]
    rw [hD l List.mem_cons_self, qsum_append]
    have ih := sumOver_qdig D h g L (fun x hx => hL x (List.mem_cons_of_mem _ hx))
      (fun x hx => hD x (List.mem_cons_of_mem _ hx))
    simp only [ih, qdig_update h _ (hL l List.mem_cons_self)]
    exact qsum_setDigs _ _ _

theorem eq_of_digitsP : ∀ (k v w : ℕ), v < p ^ k → w < p ^ k →
    (∀ i < k, digit p i v = digit p i w) → v = w
  | 0, v, w, hv, hw, _ => by simp at hv hw; omega
  | k + 1, v, w, hv, hw, h => by
    have hp : 0 < p := NeZero.pos p
    have h0 := h 0 (by omega)
    have hs : ∀ i < k, digit p i (v / p) = digit p i (w / p) := by
      intro i hi
      have := h (i + 1) (by omega)
      unfold digit at this ⊢
      rwa [pow_succ', ← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul] at this
    have hv' : v / p < p ^ k := by
      rw [Nat.div_lt_iff_lt_mul hp, ← pow_succ]; exact hv
    have hw' : w / p < p ^ k := by
      rw [Nat.div_lt_iff_lt_mul hp, ← pow_succ]; exact hw
    have := eq_of_digitsP k (v / p) (w / p) hv' hw' hs
    simp only [digit, pow_zero, Nat.div_one] at h0
    rw [← Nat.mod_add_div v p, ← Nat.mod_add_div w p, h0, this]

omit [NeZero p] in
theorem natCast_inj_lt {a b : ℕ} (ha : a < p) (hb : b < p) :
    ((a : ZMod p) = (b : ZMod p)) ↔ a = b := by
  rw [ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb]

theorem digit_hiP (q j a b : ℕ) (hb : b < p ^ q) :
    digit p (q + j) (b + p ^ q * a) = digit p j a := by
  unfold digit
  rw [pow_add, ← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ (pow_pos (NeZero.pos p) q),
    Nat.div_eq_of_lt hb, zero_add]

/-! ## Agreement -/

theorem qsum_agree (f : (ℕ → ZMod p) → ℂ) (vs : List ℕ)
    (hf : ∀ τ τ', (∀ x ∈ vs, τ x = τ' x) → f τ = f τ') :
    ∀ (ys : List ℕ) (τ τ' : ℕ → ZMod p), (∀ x ∈ vs, x ∉ ys → τ x = τ' x) →
      qsum ys f τ = qsum ys f τ'
  | [], τ, τ', h => hf τ τ' fun x hx => h x hx (by simp)
  | y :: ys, τ, τ', h => by
    simp only [qsum]
    apply Finset.sum_congr rfl
    intro b _
    apply qsum_agree f vs hf ys
    intro x hx hxs
    by_cases hxy : x = y
    · subst hxy; simp
    · rw [Function.update_of_ne hxy, Function.update_of_ne hxy]
      exact h x hx (by simp [hxy, hxs])

theorem summand_agree (c : ℂ) (s : Data) (τ τ' : ℕ → ZMod p)
    (h : ∀ x ∈ pvars s.phase ++ cvars (s.cons ++ s.rows), τ x = τ' x) :
    summand p c s τ = summand p c s τ' := by
  unfold summand
  have hP : qv τ s.phase = qv τ' s.phase :=
    qv_agree τ τ' _ fun x hx => h x (List.mem_append_left _ hx)
  have hq : ∀ q ∈ s.cons ++ s.rows, qv τ q = qv τ' q := fun q hq =>
    qv_agree τ τ' q fun x hx => h x (List.mem_append_right _ (List.mem_flatMap.2 ⟨q, hq, hx⟩))
  have hC : (∀ q ∈ s.cons ++ s.rows, qv τ q = 0) ↔ (∀ q ∈ s.cons ++ s.rows, qv τ' q = 0) :=
    ⟨fun H q h' => by rw [← hq q h']; exact H q h', fun H q h' => by rw [hq q h']; exact H q h'⟩
  rw [hP, if_congr hC rfl rfl]

theorem val_agreeQ (c : ℂ) (s : Data) (τ τ' : ℕ → ZMod p)
    (h : ∀ x ∈ pvars s.phase ++ cvars (s.cons ++ s.rows), x ∉ s.path → τ x = τ' x) :
    val p c s τ = val p c s τ' := by
  unfold val
  exact qsum_agree _ _ (fun σ σ' hσ => summand_agree c s σ σ' hσ) s.path τ τ' h

/-- The variables a fragment reads off its path. -/
def fvarsQ (f : QFrag) : List ℕ :=
  (pvars f.phase ++ cvars f.cons).filter fun x => !decide (x ∈ f.path)

theorem fval_agree (f : QFrag) (τ τ' : ℕ → ZMod p) (h : ∀ x ∈ fvarsQ f, τ x = τ' x) :
    fval p f τ = fval p f τ' := by
  unfold fval
  apply val_agreeQ
  intro x hx hp
  apply h
  rw [fvarsQ, List.mem_filter]
  exact ⟨by simpa using hx, by simpa using hp⟩

/-- Whether every variable `s` reads off its path is in `vs`. -/
def freeIn (vs : List ℕ) (s : Data) : Bool :=
  (pvars s.phase ++ cvars (s.cons ++ s.rows)).all fun x => decide (x ∈ s.path ∨ x ∈ vs)

theorem val_agree_free (c : ℂ) (s : Data) (vs : List ℕ) (τ τ' : ℕ → ZMod p)
    (hv : freeIn vs s = true) (h : ∀ x ∈ vs, τ x = τ' x) : val p c s τ = val p c s τ' := by
  apply val_agreeQ
  intro x hx hp
  have := List.all_eq_true.1 hv x hx
  simp only [decide_eq_true_eq] at this
  rcases this with h1 | h1
  · exact absurd h1 hp
  · exact h x h1

omit [NeZero p] in
theorem qdig_agree (own₁ own₂ : PSem.Own) (σ : ℕ → ℕ) (vs : List ℕ)
    (h : vs.all (fun x => decide (List.lookup x own₁ = List.lookup x own₂)) = true) :
    ∀ x ∈ vs, qdig p own₁ σ x = qdig p own₂ σ x := by
  intro x hx
  have := List.all_eq_true.1 h x hx
  simp only [decide_eq_true_eq] at this
  unfold qdig
  rw [this]

/-! ## Fragments over digits -/

/-- The constraint `x - y = 0`. -/
def eqQ (p x y : ℕ) : Poly := [([(x, 1)], 1), ([(y, 1)], p - 1)]

theorem qv_eqQ (τ : ℕ → ZMod p) (x y : ℕ) : qv τ (eqQ p x y) = τ x - τ y := by
  unfold eqQ
  rw [qv_cons, qv_cons, qv_nil, qm_cons, qm_nil, qm_cons, qm_nil, cast_pred_one]
  push_cast
  ring

/-- The constraints `x - y = 0` of two digit lists. -/
def eqConsQ (p : ℕ) (xs ys : List ℕ) : List Poly := List.zipWith (eqQ p) xs ys

theorem mem_eqConsQ_iff (τ : ℕ → ZMod p) (xs ys : List ℕ) (h : xs.length = ys.length) :
    (∀ q ∈ eqConsQ p xs ys, qv τ q = 0) ↔ ∀ i (hi : i < xs.length),
      τ (xs[i]) = τ (ys[i]'(h ▸ hi)) := by
  unfold eqConsQ
  constructor
  · intro H i hi
    have hmem : eqQ p (xs[i]) (ys[i]'(h ▸ hi)) ∈ List.zipWith (eqQ p) xs ys := by
      rw [List.mem_iff_getElem]
      refine ⟨i, by rw [List.length_zipWith]; exact lt_min hi (h ▸ hi), ?_⟩
      simp
    have := H _ hmem
    rw [qv_eqQ] at this
    exact sub_eq_zero.1 this
  · intro H q hq
    rw [List.mem_iff_getElem] at hq
    obtain ⟨i, hi, rfl⟩ := hq
    simp only [List.length_zipWith, lt_min_iff] at hi
    simp only [List.getElem_zipWith, qv_eqQ]
    rw [H i hi.1, sub_self]

/-- Whether `P` takes the value `ts[i]` at the base-`p` digits of `v + i` on `ds`, for every
entry of `ts`. -/
def tabGo (p : ℕ) (ds : List ℕ) (P : Poly) : ℕ → List ℕ → Bool
  | _, [] => true
  | v, t :: ts => decide (qv (setDigs p ds v fun _ => 0) P = ((t : ℕ) : ZMod p)) &&
      tabGo p ds P (v + 1) ts

omit [NeZero p] in
theorem tabGo_spec (ds : List ℕ) (P : Poly) : ∀ (ts : List ℕ) (v : ℕ),
    tabGo p ds P v ts = true → ∀ i < ts.length,
      qv (setDigs p ds (v + i) fun _ => 0) P = ((ts.getD i 0 : ℕ) : ZMod p)
  | [], _, _, i, hi => absurd hi (by simp)
  | t :: ts, v, h, i, hi => by
    simp only [tabGo, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases i with _ | i
    · simpa using h.1
    · have := tabGo_spec ds P ts (v + 1) h.2 i (by simpa using hi)
      rw [show v + (i + 1) = v + 1 + i by omega]
      simpa using this

/-- Whether `P` takes the value `ts[v]` (0 past the end of `ts`) at the base-`p` digits of every
`v < p^|ds|` on `ds`. -/
def tabQD (p : ℕ) (ds : List ℕ) (P : Poly) (ts : List ℕ) : Bool :=
  tabGo p ds P 0 ts && (List.range' ts.length (p ^ ds.length - ts.length)).all fun v =>
    decide (qv (setDigs p ds v fun _ => 0) P = 0)

omit [NeZero p] in
theorem tabQD_spec (ds : List ℕ) (P : Poly) (ts : List ℕ) (h : tabQD p ds P ts = true)
    (v : ℕ) (hv : v < p ^ ds.length) :
    qv (setDigs p ds v fun _ => 0) P = ((ts.getD v 0 : ℕ) : ZMod p) := by
  simp only [tabQD, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  by_cases hl : v < ts.length
  · simpa using tabGo_spec ds P ts 0 h.1 v hl
  · rw [List.getD_eq_default _ _ (by omega), h.2 v (List.mem_range'_1.2 ⟨by omega, by omega⟩)]
    simp

/-- The Z spider's fragment over digits: digit `j` of each leg equal to the anchor's digit `j`,
the phase `P` over the anchor digits. -/
def zFragQD (p : ℕ) (dig : ℕ → List ℕ) (as ls : List ℕ) (P : Poly) : QFrag :=
  ⟨[], P, ls.flatMap fun l => eqConsQ p (PSem.anchorD dig as) (dig l), 1⟩

theorem z_frag_qd (E : Env) (σ : ℕ → ℕ) (own : PSem.Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : PSem.ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < p ^ (dig l).length)
    (n : Node) (hk : n.kind = .Z) (k : ℕ) (hd : E.dimv n.dim = p ^ k)
    (hlabs : ∀ l ∈ n.outs ++ n.ins, l ∈ labs) (hlen : PSem.lensOK dig k (n.outs ++ n.ins) = true)
    (as : List ℕ) (has : as.length = k) (hasl : ∀ a ∈ as, a ∈ n.outs ++ n.ins)
    (l0 : ℕ) (hl0 : l0 ∈ n.outs ++ n.ins) (P : Poly) (ts : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / p)
    (htab : tabQD p (PSem.anchorD dig as) P ts = true)
    (hnd : (PSem.anchorD dig as).Nodup)
    (hPv : (pvars P).all (fun x => decide (x ∈ PSem.anchorD dig as)) = true) :
    ArchytasZX.val E σ n = fval p (zFragQD p dig as (n.outs ++ n.ins) P) (qdig p own σ) := by
  set ls := n.outs ++ n.ins with hls
  have hp : 0 < p := NeZero.pos p
  have hlen' : ∀ l ∈ ls, (dig l).length = k := by
    intro l hl
    simpa [PSem.lensOK] using (List.all_eq_true.1 hlen) l hl
  have hσk : ∀ l ∈ ls, σ l < p ^ k := fun l hl => (hlen' l hl) ▸ hσ l (hlabs l hl)
  have hv : ArchytasZX.val E σ n = tZ (p ^ k) (E.phase n.ph) (σ l0 :: ls.map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd]
    apply PSem.tZ_congr
    intro x
    simp only [List.mem_cons, ← List.map_append]
    constructor
    · exact Or.inr
    · rintro (rfl | hx)
      · exact List.mem_map.2 ⟨l0, hl0, rfl⟩
      · exact hx
  rw [hv, PSem.tZ_val _ _ _ _ (hσk l0 hl0)]
  have hal : (PSem.anchorD dig as).length = k := by simp [PSem.anchorD, has]
  have hanc : ∀ j (hj : j < k), (PSem.anchorD dig as)[j]'(hal ▸ hj) =
      (dig (as[j]'(has ▸ hj)))[j]'(by
        rw [hlen' _ (hasl _ (List.getElem_mem (has ▸ hj)))]; exact hj) := by
    intro j hj
    have hlj : j < (dig (as[j]'(has ▸ hj))).length := by
      rw [hlen' _ (hasl _ (List.getElem_mem (has ▸ hj)))]; exact hj
    simp only [PSem.anchorD, List.getElem_mapIdx, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem hlj, Option.getD_some]
  have hdb : ∀ l (hl : l ∈ ls) j (hj : j < k),
      qdig p own σ ((dig l)[j]'(by rw [hlen' l hl]; exact hj)) =
      ((digit p j (σ l) : ℕ) : ZMod p) := by
    intro l hl j hj
    exact qdig_get hown σ (hlabs l hl) _
  have hdba : ∀ j (hj : j < k), qdig p own σ ((PSem.anchorD dig as)[j]'(hal ▸ hj)) =
      ((digit p j (σ (as[j]'(has ▸ hj))) : ℕ) : ZMod p) := by
    intro j hj
    rw [hanc j hj]
    exact hdb _ (hasl _ (List.getElem_mem (has ▸ hj))) j hj
  have hcond : (∀ q ∈ (zFragQD p dig as ls P).cons, qv (qdig p own σ) q = 0) ↔
      ∀ w ∈ ls.map σ, w = σ l0 := by
    simp only [zFragQD, List.mem_flatMap, forall_exists_index, and_imp]
    have hpair : ∀ l ∈ ls, (∀ q ∈ eqConsQ p (PSem.anchorD dig as) (dig l),
        qv (qdig p own σ) q = 0) ↔
        ∀ j < k, digit p j (σ (as.getD j 0)) = digit p j (σ l) := by
      intro l hl
      rw [mem_eqConsQ_iff _ _ _ (by rw [hal, hlen' l hl])]
      constructor
      · intro H j hj
        have := H j (hal ▸ hj)
        rw [hdba j hj, hdb l hl j hj,
          natCast_inj_lt (digit_lt p j _ hp) (digit_lt p j _ hp)] at this
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (has ▸ hj), Option.getD_some]
        exact this
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
      exact eq_of_digitsP k _ _ (hσk l hl) (hσk l0 hl0) fun j hj =>
        (h1 j hj).symm.trans (h2 j hj)
    · intro H q l hl hq
      refine (hpair l hl).2 (fun j hj => ?_) q hq
      have ha : as.getD j 0 ∈ ls := by
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (has ▸ hj), Option.getD_some]
        exact hasl _ (List.getElem_mem (has ▸ hj))
      rw [H _ (List.mem_map.2 ⟨_, ha, rfl⟩), H _ (List.mem_map.2 ⟨l, hl, rfl⟩)]
  unfold fval val
  simp only [qsum, summand, List.append_nil, zFragQD]
  have hcond' := hcond
  simp only [zFragQD] at hcond'
  rw [if_congr hcond' rfl rfl]
  split_ifs with hc
  · rw [one_mul]
    have hset : ∀ x ∈ PSem.anchorD dig as, qdig p own σ x =
        setDigs p (PSem.anchorD dig as) (σ l0) (fun _ => 0) x := by
      intro x hx
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
      have hj' : j < k := hal ▸ hj
      rw [setDigs_idx (PSem.anchorD dig as) (σ l0) (fun _ => 0) hnd j hj, hdba j hj']
      have ha : as[j]'(has ▸ hj') ∈ ls := hasl _ (List.getElem_mem (has ▸ hj'))
      rw [hc _ (List.mem_map.2 ⟨_, ha, rfl⟩)]
    have hqv : qv (qdig p own σ) P =
        qv (setDigs p (PSem.anchorD dig as) (σ l0) fun _ => 0) P := by
      apply qv_agree
      intro x hx
      apply hset
      simpa using List.all_eq_true.1 hPv x hx
    have hv0 : σ l0 < p ^ (PSem.anchorD dig as).length := by rw [hal]; exact hσk l0 hl0
    rw [hqv, tabQD_spec _ P ts htab (σ l0) hv0, ψ_nat, om_eq_ex, hph]
  · rfl

theorem b_cond_q (own : PSem.Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : PSem.ownOK own labs dig = true) (σ : ℕ → ℕ) (r q k a b : ℕ)
    (hk : k ∈ labs) (ha : a ∈ labs) (hb : b ∈ labs)
    (hka : (dig k).length = q + r) (hla : (dig a).length = r) (hlb : (dig b).length = q)
    (hσk : σ k < p ^ (q + r)) (hσa : σ a < p ^ r) (hσb : σ b < p ^ q) :
    (∀ c ∈ eqConsQ p (dig k) (dig b ++ dig a), qv (qdig p own σ) c = 0) ↔
      σ k = σ a * p ^ q + σ b := by
  have hp : 0 < p := NeZero.pos p
  rw [mem_eqConsQ_iff _ _ _ (by rw [hka, List.length_append, hla, hlb])]
  have hw : σ a * p ^ q + σ b < p ^ (q + r) := by
    rw [pow_add]
    have hpq : 0 < p ^ q := pow_pos hp q
    calc σ a * p ^ q + σ b < (σ a + 1) * p ^ q := by nlinarith
      _ ≤ p ^ r * p ^ q := Nat.mul_le_mul_right _ hσa
      _ = p ^ q * p ^ r := by ring
  have hrhs : ∀ i (hi : i < q + r), qdig p own σ ((dig b ++ dig a)[i]'(by
        rw [List.length_append, hla, hlb]; exact hi)) =
      ((digit p i (σ a * p ^ q + σ b) : ℕ) : ZMod p) := by
    intro i hi
    by_cases hiq : i < q
    · rw [List.getElem_append_left (by rw [hlb]; exact hiq), qdig_get hown σ hb]
      rw [show σ a * p ^ q + σ b = σ b + p ^ q * σ a by ring,
        digit_low p q i (σ b) (σ a) hp hiq hσb]
    · have hj : i - q < (dig a).length := by rw [hla]; omega
      rw [List.getElem_append_right (by rw [hlb]; omega)]
      simp only [hlb]
      rw [qdig_get hown σ ha hj]
      rw [show σ a * p ^ q + σ b = σ b + p ^ q * σ a by ring,
        show i = q + (i - q) by omega, digit_hiP q (i - q) (σ a) (σ b) hσb]
      simp
  constructor
  · intro H
    apply eq_of_digitsP (q + r) _ _ hσk hw
    intro i hi
    have := H i (by rw [hka]; exact hi)
    rw [qdig_get hown σ hk, hrhs i hi,
      natCast_inj_lt (digit_lt p i _ hp) (digit_lt p i _ hp)] at this
    exact this
  · intro H i hi
    have hi' : i < q + r := by rw [← hka]; exact hi
    rw [qdig_get hown σ hk, hrhs i hi', H]

/-- The binder's or splitter's fragment: the whole label's digits equal the low part's digits
then the high part's. -/
def bFragQ (p : ℕ) (dig : ℕ → List ℕ) (k a b : ℕ) : QFrag :=
  ⟨[], [], eqConsQ p (dig k) (dig b ++ dig a), 1⟩

theorem b_frag_q (E : Env) (σ : ℕ → ℕ) (own : PSem.Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : PSem.ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < p ^ (dig l).length)
    (n : Node) (hkind : n.kind = .B) (r q : ℕ) (hd : E.dimv n.dim = p ^ r)
    (ht : E.dimv n.ph = p ^ q) (k a b : ℕ) (ho : n.outs = [k]) (hi : n.ins = [a, b])
    (hk : k ∈ labs) (ha : a ∈ labs) (hb : b ∈ labs)
    (hka : (dig k).length = q + r) (hla : (dig a).length = r) (hlb : (dig b).length = q) :
    ArchytasZX.val E σ n = fval p (bFragQ p dig k a b) (qdig p own σ) := by
  have hσk : σ k < p ^ (q + r) := hka ▸ hσ k hk
  have hσa : σ a < p ^ r := hla ▸ hσ a ha
  have hσb : σ b < p ^ q := hlb ▸ hσ b hb
  have hv : ArchytasZX.val E σ n = tB (p ^ r) (p ^ q) (σ k) (σ a) (σ b) := by
    unfold ArchytasZX.val
    rw [hkind, ho, hi]
    simp only [hd, ht, List.map_cons, List.map_nil]
  rw [hv]
  unfold fval val bFragQ tB
  simp only [qsum, summand, List.append_nil, qv_nil, ψ_zero, mul_one]
  rw [if_congr (b_cond_q own labs dig hown σ r q k a b hk ha hb hka hla hlb hσk hσa hσb) rfl rfl]
  simp [hσa, hσb]

theorem s_frag_q (E : Env) (σ : ℕ → ℕ) (own : PSem.Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : PSem.ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < p ^ (dig l).length)
    (n : Node) (hkind : n.kind = .S) (r q : ℕ) (hd : E.dimv n.dim = p ^ r)
    (ht : E.dimv n.ph = p ^ q) (k a b : ℕ) (ho : n.outs = [a, b]) (hi : n.ins = [k])
    (hk : k ∈ labs) (ha : a ∈ labs) (hb : b ∈ labs)
    (hka : (dig k).length = q + r) (hla : (dig a).length = r) (hlb : (dig b).length = q) :
    ArchytasZX.val E σ n = fval p (bFragQ p dig k a b) (qdig p own σ) := by
  have hσk : σ k < p ^ (q + r) := hka ▸ hσ k hk
  have hσa : σ a < p ^ r := hla ▸ hσ a ha
  have hσb : σ b < p ^ q := hlb ▸ hσ b hb
  have hv : ArchytasZX.val E σ n = tB (p ^ r) (p ^ q) (σ k) (σ a) (σ b) := by
    unfold ArchytasZX.val
    rw [hkind, ho, hi]
    simp only [hd, ht, List.map_cons, List.map_nil]
  rw [hv]
  unfold fval val bFragQ tB
  simp only [qsum, summand, List.append_nil, qv_nil, ψ_zero, mul_one]
  rw [if_congr (b_cond_q own labs dig hown σ r q k a b hk ha hb hka hla hlb hσk hσa hσb) rfl rfl]
  simp [hσa, hσb]

/-- A fragment of a node on dimension `p` read at digit variables: every leg a label of one
digit, itself, and the fragment reading only the legs off its path. -/
theorem lift_q (E : Env) (σ : ℕ → ℕ) (own : PSem.Own) (labs : List ℕ) (dig : ℕ → List ℕ)
    (hown : PSem.ownOK own labs dig = true) (hσ : ∀ l ∈ labs, σ l < p ^ (dig l).length)
    (n : Node) (f : QFrag)
    (hone : (n.outs ++ n.ins).all (fun l => decide (l ∈ labs) && decide (dig l = [l])) = true)
    (hfv : (fvarsQ f).all (fun x => decide (x ∈ n.outs ++ n.ins)) = true)
    (h : ∀ σ' : ℕ → ℕ, (∀ l, σ' l < p) → ArchytasZX.val E σ' n = fval p f (zm p σ')) :
    ArchytasZX.val E σ n = fval p f (qdig p own σ) := by
  have hp : 0 < p := NeZero.pos p
  have hl : ∀ l ∈ n.outs ++ n.ins, l ∈ labs ∧ dig l = [l] := by
    intro l hl'
    simpa using List.all_eq_true.1 hone l hl'
  have hlt : ∀ l ∈ n.outs ++ n.ins, σ l < p := by
    intro l hl'
    have := hσ l (hl l hl').1
    rwa [(hl l hl').2, List.length_singleton, pow_one] at this
  have hr : ArchytasZX.val E σ n = ArchytasZX.val E (fun l => σ l % p) n := by
    apply val_reads E n
    intro x hx
    exact (Nat.mod_eq_of_lt (hlt x hx)).symm
  rw [hr, h _ (fun l => Nat.mod_lt _ hp)]
  apply fval_agree
  intro x hx
  have hxl : x ∈ n.outs ++ n.ins := by simpa using List.all_eq_true.1 hfv x hx
  obtain ⟨hlab, hdig⟩ := hl x hxl
  have hlk : List.lookup x own = some (x, 0) :=
    PSem.ownOK_get? hown hlab (by rw [hdig]; rfl)
  rw [qdig_of_lookup own σ hlk]
  simp [zm, digit]

/-! ## Networks over digit variables -/

/-- The network's path sum over digits: the summed labels' digits then the fragments' paths. -/
def qnetDataD (dig : ℕ → List ℕ) (N : Net) (fs : List QFrag) : Data :=
  ⟨N.int.flatMap dig ++ (qjoinAll fs).path, (qjoinAll fs).phase, (qjoinAll fs).cons, []⟩

/-- `QNetPS` over digit variables: the owner table, each node's entry as its fragment's value at
the digits of in-range label values, disjoint fragment paths, every label of dimension
`p ^ |dig l|`, and every label summed or on the boundary. -/
structure QNetPSD (E : Env) (N : Net) (p : ℕ) [NeZero p] (own : PSem.Own) (labs : List ℕ)
    (dig : ℕ → List ℕ) (ps : List (Node × QFrag)) : Prop where
  ownok : PSem.ownOK own labs dig = true
  nodes : N.nodes = ps.map Prod.fst
  node : ∀ q ∈ ps, ∀ σ, (∀ l ∈ labs, σ l < p ^ (dig l).length) →
    ArchytasZX.val E σ q.1 = fval p q.2 (qdig p own σ)
  disj : (ps.map Prod.snd).Pairwise fun a b => QDisj a b ∧ QDisj b a
  dims : ∀ l ∈ labs, labDim E l = p ^ (dig l).length
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd
  sub : ∀ l ∈ N.int ++ N.bnd, l ∈ labs

theorem QNetPSD.eval_eq {E : Env} {N : Net} {own : PSem.Own} {labs : List ℕ}
    {dig : ℕ → List ℕ} {ps : List (Node × QFrag)} (h : QNetPSD E N p own labs dig ps)
    (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = val p (qjoinAll (ps.map Prod.snd)).c (qnetDataD dig N (ps.map Prod.snd))
      (qdig p own (PSem.clamp N.bnd σ)) := by
  have he : eval E N σ = eval E N (PSem.clamp N.bnd σ) := by
    unfold eval
    apply sumOver_ext (labDim E) N.int (labelsOf N.nodes) _ (prodVal_reads E N.nodes)
    intro x hx hn
    rcases h.cover x hx with hi | hb
    · exact absurd hi hn
    · simp [PSem.clamp, hb]
  rw [he]
  unfold eval
  have hint : ∀ l ∈ N.int, l ∈ labs := fun l hl => h.sub l (List.mem_append_left _ hl)
  set F := qjoinAll (ps.map Prod.snd) with hF
  have hcongr : sumOver (labDim E) N.int (prodVal E N.nodes) (PSem.clamp N.bnd σ) =
      sumOver (labDim E) N.int (fun σ' => fval p F (qdig p own σ')) (PSem.clamp N.bnd σ) := by
    apply sumOver_congr
    intro τ hτ hr
    have hrange : ∀ l ∈ labs, τ l < p ^ (dig l).length := by
      intro l hl
      by_cases hli : l ∈ N.int
      · have := hr l hli
        rwa [h.dims l hl] at this
      · rw [hτ l hli]
        unfold PSem.clamp
        split_ifs with hlb
        · have := hσ l hlb
          rwa [h.dims l hl] at this
        · exact pow_pos (NeZero.pos p) _
    rw [hF, fval_joinAll _ _ h.disj, prodVal, h.nodes, List.map_map, List.map_map]
    congr 1
    apply List.map_congr_left
    intro q hq
    exact h.node q hq τ hrange
  rw [hcongr, sumOver_qdig (labDim E) h.ownok (fval p F) N.int hint
    (fun l hl => h.dims l (hint l hl))]
  unfold fval val qnetDataD
  rw [qsum_append]
  rfl

/-- Two networks with one boundary whose digit path sums, scaled by `s₁` and `s₂`, agree at
every assignment, the second reading off its path only variables in `vs`, owned alike by both
tables. -/
theorem equal_qpsd2 {E : Env} {N₁ N₂ : Net} {own₁ own₂ : PSem.Own} {labs₁ labs₂ : List ℕ}
    {dig₁ dig₂ : ℕ → List ℕ} {ps₁ ps₂ : List (Node × QFrag)} (s₁ s₂ : ℂ)
    (h₁ : QNetPSD E N₁ p own₁ labs₁ dig₁ ps₁) (h₂ : QNetPSD E N₂ p own₂ labs₂ dig₂ ps₂)
    (vs : List ℕ) (hfree : freeIn vs (qnetDataD dig₂ N₂ (ps₂.map Prod.snd)) = true)
    (hown : vs.all (fun x => decide (List.lookup x own₁ = List.lookup x own₂)) = true)
    (hb : N₁.bnd = N₂.bnd)
    (hv : ∀ τ, s₁ * val p (qjoinAll (ps₁.map Prod.snd)).c (qnetDataD dig₁ N₁ (ps₁.map Prod.snd)) τ =
      s₂ * val p (qjoinAll (ps₂.map Prod.snd)).c (qnetDataD dig₂ N₂ (ps₂.map Prod.snd)) τ) :
    Equal E s₁ N₁ s₂ N₂ := by
  refine ⟨hb ▸ List.Perm.refl _, fun σ hσ => ?_⟩
  have hσ₂ : InRange E N₂.bnd σ := hb ▸ hσ
  rw [h₁.eval_eq σ hσ, h₂.eval_eq σ hσ₂, hv, ← hb]
  congr 1
  exact val_agree_free _ _ vs _ _ hfree fun x hx => qdig_agree own₁ own₂ _ vs hown x hx

end Digits

end QSem
end ArchytasZX
