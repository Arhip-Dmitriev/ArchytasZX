/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFamFca

/-!
# The Draper adder as a network family

The boxed `draper_adder` of the engine at `n` copies, copy `i` of weight `d ^ i`, as networks over
`envAt (E0 d) L n`. Copy `n` (the new most significant digit) owns the label block
`[B n, B (n + 1))`: `a` input `0`, `b` input `1`, `a` output `2`, `b` output `3`, the wire chain
`4 … 11`, the spec's chain values `12 … 16`, and per earlier copy `i` the slots
`20 + 10 i … 28 + 10 i`. The pair phases `± x y / d ^ (n - i + 1)` of the pair `(n, i)` are `ZP`
nodes between copy `n`'s Fourier value and copy `i`'s pending `b` input, `a` input and `b` output
legs (`pend`, `pendA`, `pendO`); `cut n` are the pending legs before copy `n`.

* `B`, `sl`, `E0`, `L`, `cut`, `cp`, `lhs`, `rhs`: the family.
* `labsMono`, `copyAt`, `fitsL`, `fitsR`: the side conditions of `netIfaceAtE`.
* `eval_rhsN`, `eval_cp`, `eval_glue`: the spec, one copy, and one copy glued to the spec in closed
  form.
* `arith`, `close`, `fam`, `fam_8`, `fam_24`: the family theorem.
-/

namespace ArchytasZX

namespace Draper

open List

/-- Start of the label block of level `n`. -/
def B (n : ℕ) : ℕ := 5 * n * n + 19 * n + 5

theorem B_succ (n : ℕ) : B (n + 1) = B n + (10 * n + 24) := by unfold B; ring

theorem B_le {a b : ℕ} (h : a ≤ b) : B a ≤ B b := by
  unfold B; nlinarith [Nat.mul_le_mul h h]

theorem B_lt {a b : ℕ} (h : a < b) : B a + (10 * a + 24) ≤ B b := by
  rw [← B_succ]; exact B_le h

/-- Slot `t` of level `n`. -/
def sl (n t : ℕ) : ℕ := B n + t

theorem five_le_sl (n t : ℕ) : 5 ≤ sl n t := by unfold sl B; omega

theorem five_le_B (n : ℕ) : 5 ≤ B n := by unfold B; omega

theorem sl_inj {a b x y : ℕ} (hx : x < 10 * a + 24) (hy : y < 10 * b + 24)
    (h : sl a x = sl b y) : a = b ∧ x = y := by
  unfold sl at h
  rcases lt_trichotomy a b with hab | rfl | hab
  · have := B_lt hab; omega
  · omega
  · have := B_lt hab; omega

theorem sl_lt {n i t : ℕ} (hi : i < n) (ht : t < 10 * i + 24) : sl i t < B n := by
  unfold sl; have := B_lt hi; omega

theorem sl_injective (n : ℕ) : Function.Injective (sl n) := fun a b h => by unfold sl at h; omega

/-- The environment with no listed labels and no phases. -/
noncomputable def E0 (d : ℕ) : Env := { dimv := Fca.dimv d, phase := fun _ _ => 0, labs := [] }

/-- The labels of level `n` with a dimension other than `d`: the spec values `d ^ (n + 1)`. -/
def blockL (n : ℕ) : List (ℕ × ℕ) := [12, 13, 14, 15, 16].map fun s => (sl n s, n + 2)

/-- The listed labels below `B n`. -/
def L (n : ℕ) : List (ℕ × ℕ) := [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)] ++ (range n).flatMap blockL

/-- The pending `b` input leg of copy `i` before copy `n`. -/
def pend (n i : ℕ) : ℕ := sl (n - 1) (20 + 10 * i)

/-- The pending `b` output leg of copy `i` before copy `n`. -/
def pendO (n i : ℕ) : ℕ := sl (n - 1) (21 + 10 * i)

/-- The pending `a` input leg of copy `i` before copy `n`. -/
def pendA (n i : ℕ) : ℕ := sl (n - 1) (22 + 10 * i)

/-- The pending legs before copy `n`. -/
def cut (n : ℕ) : List ℕ := (range n).flatMap fun i => [pend n i, pendO n i, pendA n i]

/-- A plain Z spider of dimension `d`. -/
def zN (os is : List ℕ) : Node := ⟨.Z, os, is, 0, 0⟩

/-- A Fourier box of dimension `d`. -/
def fN (o i : ℕ) : Node := ⟨.F, [o], [i], 0, 0⟩

/-- A pair phase `e^{2πi c y x / D}` over the legs `y`, `x`, `D` of dimension index `D`. -/
def pN (c : ℤ) (D y x : ℕ) : Node := ⟨.ZP 0 0 c, [y], [x], D, 0⟩

/-- The pair nodes of copy `n` with copy `i`: per pending leg of copy `i`, a spider moving it on
and a pair phase to copy `n`'s Fourier value. -/
def pairN (n i : ℕ) : List Node :=
  [zN [sl n (23 + 10 * i), sl n (20 + 10 * i)] [pend n i],
   pN 1 (n - i + 2) (sl n (23 + 10 * i)) (sl n (24 + 10 * i)),
   zN [sl n (25 + 10 * i), sl n (22 + 10 * i)] [pendA n i],
   pN 1 (n - i + 2) (sl n (25 + 10 * i)) (sl n (26 + 10 * i)),
   zN [sl n (27 + 10 * i), sl n (21 + 10 * i)] [pendO n i],
   pN (-1) (n - i + 2) (sl n (27 + 10 * i)) (sl n (28 + 10 * i))]

/-- The summed labels of copy `n` no node pins: the Fourier value and the `F³` legs. -/
def freeC (n : ℕ) : List ℕ := [sl n 5, sl n 9, sl n 10]

/-- The summed labels of copy `n` a node pins. -/
def pinC (n : ℕ) : List ℕ :=
  [sl n 4, sl n 6, sl n 7, sl n 8, sl n 11] ++ (range n).flatMap fun i =>
    [sl n (23 + 10 * i), sl n (24 + 10 * i), sl n (25 + 10 * i), sl n (26 + 10 * i),
      sl n (27 + 10 * i), sl n (28 + 10 * i)]

/-- The pair legs on copy `n`'s merge spider. -/
def legs (n : ℕ) : List ℕ :=
  (range n).flatMap fun i => [sl n (24 + 10 * i), sl n (26 + 10 * i), sl n (28 + 10 * i)]

/-- The fixed nodes of copy `n`. -/
def fixN (n : ℕ) : List Node :=
  [zN [sl n 4, sl n (20 + 10 * n)] [sl n 1],
   fN (sl n 5) (sl n 4),
   zN [sl n 6, sl n 2, sl n (22 + 10 * n)] [sl n 0],
   fN (sl n 7) (sl n 6),
   zN (sl n 8 :: legs n) [sl n 5, sl n 7],
   fN (sl n 9) (sl n 8),
   fN (sl n 10) (sl n 9),
   fN (sl n 11) (sl n 10),
   zN [sl n 3, sl n (21 + 10 * n)] [sl n 11]]

/-- Copy `n` of the boxed side. -/
def cp (n : ℕ) : Net where
  nodes := fixN n ++ (range n).flatMap (pairN n)
  int := freeC n ++ pinC n
  bnd := [sl n 0, sl n 1, sl n 2, sl n 3, sl n (20 + 10 * n), sl n (21 + 10 * n),
    sl n (22 + 10 * n)] ++
    (range n).flatMap fun i => [sl n (20 + 10 * i), sl n (21 + 10 * i), sl n (22 + 10 * i)]

/-- The boundary after `n` copies. -/
def bndL : ℕ → List ℕ
  | 0 => []
  | n + 1 => (bndL n).filter (· ∉ cut n) ++ (cp n).bnd.filter (· ∉ cut n)

/-- The boxed side: copies `0 … n - 1` glued in order. -/
def lhsN : ℕ → Net
  | 0 => ⟨[], [], []⟩
  | n + 1 => glueAt (cut n) (cp n) (lhsN n)

/-- The boxed side with its scalar. -/
def lhs (n : ℕ) : ℂ × Net := (1, lhsN n)

/-- The spec's chain value `c` of the top `t` digits: `0`, `1` the `a`, `b` binders, `2`, `3` the
`a`, `b` splitters, `4` the copy leg of the adder. -/
def cv (c : ℕ) : ℕ → ℕ
  | 0 => c
  | t + 1 => sl t (12 + c)

/-- A binder node of chain `c` at step `t` of `n`. -/
def bN (n c t : ℕ) : Node := ⟨.B, [cv c (t + 1)], [cv c t, sl (n - 1 - t) c], t + 1, 0⟩

/-- A splitter node of chain `c` at step `t` of `n`. -/
def sN (n c t : ℕ) : Node := ⟨.S, [cv c t, sl (n - 1 - t) c], [cv c (t + 1)], t + 1, 0⟩

/-- The spec at `n` digits: the binder chains of `a` and `b`, the copy of `a` and the sum
`a + b` mod `d ^ n`, the splitter chains, and the pending legs as spiders on the inputs and
outputs. -/
def rhsN (n : ℕ) : Net where
  nodes :=
    [⟨.Z, [cv 0 0], [], 1, 0⟩, ⟨.Z, [cv 1 0], [], 1, 0⟩] ++
    (range n).map (bN n 0) ++ (range n).map (bN n 1) ++
    [⟨.Z, [cv 2 n, cv 4 n], [cv 0 n], n + 1, 0⟩, ⟨.X, [cv 3 n], [cv 1 n, cv 4 n], n + 1, 0⟩] ++
    (range n).map (sN n 2) ++ (range n).map (sN n 3) ++
    [⟨.Z, [], [cv 2 0], 1, 0⟩, ⟨.Z, [], [cv 3 0], 1, 0⟩] ++
    (range n).flatMap fun i =>
      [zN [pend n i] [sl i 1], zN [pendO n i] [sl i 3], zN [pendA n i] [sl i 0]]
  int := [cv 0 0, cv 1 0, cv 2 0, cv 3 0, cv 4 n] ++
    (range n).flatMap fun t => [cv 0 (t + 1), cv 1 (t + 1), cv 2 (t + 1), cv 3 (t + 1)]
  bnd := bndL n

/-! ## Side conditions -/

theorem blockL_ge (n : ℕ) : ∀ p ∈ blockL n, B n ≤ p.1 := by
  intro p hp
  simp only [blockL, map_cons, map_nil, mem_cons, not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl <;> simp [sl]

theorem labsMono (d : ℕ) : LabsMono (E0 d) L B := by
  intro n x hx
  have h : L (n + 1) = L n ++ blockL n := by
    simp [L, range_succ, flatMap_append]
  simp only [labDim, envAt, h]
  rw [Fca.lookup_append_of_not x _ _ fun p hp e => by have := blockL_ge n p hp; omega]

theorem mem_cut {n x : ℕ} : x ∈ cut n ↔ ∃ i < n, x = pend n i ∨ x = pendO n i ∨ x = pendA n i := by
  simp only [cut, mem_flatMap, mem_range, mem_cons, not_mem_nil, or_false]

/-- Labels owned by level `n` or pending before it. -/
def Own (n x : ℕ) : Prop := (∃ t < 10 * n + 24, x = sl n t) ∨ x ∈ cut n

theorem cp_own (n : ℕ) : ∀ x ∈ labelsOf (cp n).nodes ++ (cp n).int ++ (cp n).bnd, Own n x := by
  simp only [cp, fixN, legs, freeC, pinC, labelsOf, pairN, zN, fN, pN, Node.labels,
    mem_append, mem_flatMap, mem_cons, mem_range, not_mem_nil, or_false, flatMap_cons,
    cons_append, nil_append, or_imp, forall_and, forall_eq, forall_exists_index, and_imp, Own,
    mem_cut]
  and_intros
  all_goals (intros; subst_vars)
  all_goals (try simp only [mem_cons, not_mem_nil, or_false] at *)
  all_goals (try casesm* _ ∨ _)
  all_goals (try subst_vars)
  all_goals first
    | exact Or.inl ⟨_, by omega, rfl⟩
    | exact Or.inr ⟨_, by assumption, Or.inl rfl⟩
    | exact Or.inr ⟨_, by assumption, Or.inr (Or.inl rfl)⟩
    | exact Or.inr ⟨_, by assumption, Or.inr (Or.inr rfl)⟩

theorem cut_lt {n x : ℕ} (h : x ∈ cut n) : x < B n := by
  obtain ⟨i, hi, h | h | h⟩ := mem_cut.1 h <;> subst h <;>
    exact sl_lt (by omega : n - 1 < n) (by omega)

theorem cp_int {n x : ℕ} (h : x ∈ (cp n).int) :
    ∃ t, ((4 ≤ t ∧ t < 12) ∨ ∃ i < n, ∃ r < 6, t = 23 + 10 * i + r) ∧ x = sl n t := by
  simp only [cp, freeC, pinC, mem_append, mem_cons, not_mem_nil, or_false, mem_range,
    mem_flatMap] at h
  rcases h with (rfl | rfl | rfl) |
    ((rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, rfl | rfl | rfl | rfl | rfl | rfl⟩)
  all_goals first
    | exact ⟨_, Or.inl (by omega), rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 0 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 1 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 2 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 3 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 4 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 5 < 6), by omega⟩, rfl⟩

theorem copyAt : CopyAt B cut cp where
  mono n := by rw [B_succ]; omega
  cutLt _ _ h := cut_lt h
  own n x hx := by
    rcases cp_own n x hx with ⟨t, ht, rfl⟩ | h
    · right; rw [B_succ]; unfold sl; omega
    · exact Or.inl h
  intOff n m hnm x hx hc := by
    obtain ⟨t, ht, rfl⟩ := cp_int hx
    obtain ⟨j, hj, h | h | h⟩ := mem_cut.1 hc <;>
    · obtain ⟨h1, h2⟩ := sl_inj (by omega) (by omega) h
      omega
  cutOff n m hnm x hx hc := by
    obtain ⟨i, hi, h | h | h⟩ := mem_cut.1 hx <;> obtain ⟨j, hj, h' | h' | h'⟩ := mem_cut.1 hc <;>
    · rw [h] at h'
      have := (sl_inj (by omega) (by omega) h').1
      omega

theorem fitsL (n : ℕ) : FitsAt B cut n (lhsN n) := by
  induction n with
  | zero => exact ⟨by simp [labelsOf, lhsN], fun _ _ _ h => by simp [lhsN] at h⟩
  | succ n ih => exact glueAt_fits copyAt n ih

theorem lhsN_bnd (n : ℕ) : (lhsN n).bnd = bndL n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [lhsN, glueAt, bndL, ih]

theorem bndL_lt {n x : ℕ} (h : x ∈ bndL n) : x < B n :=
  (fitsL n).1 x (by rw [← lhsN_bnd] at h; exact mem_all_of_bnd h)

theorem cv_lt {n c t : ℕ} (hc : c < 5) (h : t ≤ n) : cv c t < B n := by
  cases t with
  | zero => have := five_le_B n; simp only [cv]; omega
  | succ t => exact sl_lt (by omega) (by omega)

theorem pend_mem {n i : ℕ} (h : i < n) : pend n i ∈ cut n := mem_cut.2 ⟨i, h, Or.inl rfl⟩

theorem pendO_mem {n i : ℕ} (h : i < n) : pendO n i ∈ cut n :=
  mem_cut.2 ⟨i, h, Or.inr (Or.inl rfl)⟩

theorem pendA_mem {n i : ℕ} (h : i < n) : pendA n i ∈ cut n :=
  mem_cut.2 ⟨i, h, Or.inr (Or.inr rfl)⟩

theorem cv_off (c t a s : ℕ) (hc : c < 5) (hs : 20 ≤ s) (hs' : s < 10 * a + 24) :
    cv c t ≠ sl a s := by
  cases t with
  | zero => have := five_le_sl a s; simp only [cv]; omega
  | succ t =>
    intro h
    have := (sl_inj (by omega) hs' h).2
    omega

theorem rhsInt_mem {n x : ℕ} (h : x ∈ (rhsN n).int) : ∃ c < 5, ∃ t ≤ n, x = cv c t := by
  simp only [rhsN, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range] at h
  rcases h with (rfl | rfl | rfl | rfl | rfl) | ⟨t, ht, rfl | rfl | rfl | rfl⟩
  · exact ⟨0, by omega, 0, by omega, rfl⟩
  · exact ⟨1, by omega, 0, by omega, rfl⟩
  · exact ⟨2, by omega, 0, by omega, rfl⟩
  · exact ⟨3, by omega, 0, by omega, rfl⟩
  · exact ⟨4, by omega, n, le_rfl, rfl⟩
  · exact ⟨0, by omega, t + 1, by omega, rfl⟩
  · exact ⟨1, by omega, t + 1, by omega, rfl⟩
  · exact ⟨2, by omega, t + 1, by omega, rfl⟩
  · exact ⟨3, by omega, t + 1, by omega, rfl⟩

theorem fitsR (n : ℕ) : FitsAt B cut n (rhsN n) := by
  refine ⟨?_, fun m hm x hx hc => ?_⟩
  · simp only [rhsN, bN, sN, labelsOf, zN, Node.labels, flatMap_append, mem_append, mem_flatMap,
      mem_cons, mem_range, mem_map, not_mem_nil, or_false, flatMap_cons, cons_append,
      nil_append, or_imp, forall_and, forall_eq, forall_exists_index, and_imp]
    and_intros
    all_goals (intros; subst_vars)
    all_goals (try simp only [mem_cons, not_mem_nil, or_false] at *)
    all_goals (try casesm* _ ∨ _)
    all_goals (try subst_vars)
    all_goals first
      | exact cv_lt (by omega) (by omega)
      | exact cut_lt (pend_mem (by omega))
      | exact cut_lt (pendO_mem (by omega))
      | exact cut_lt (pendA_mem (by omega))
      | exact sl_lt (by omega) (by omega)
      | exact bndL_lt (by assumption)
  · obtain ⟨c, hc5, t, _, rfl⟩ := rhsInt_mem hx
    obtain ⟨j, hj, h | h | h⟩ := mem_cut.1 hc <;>
    · exact cv_off c t (m - 1) _ hc5 (by omega) (by omega) h

/-! ## Label dimensions -/

theorem L_keys (t : ℕ) : ∀ p ∈ L t, p.1 < B t := by
  intro p hp
  simp only [L, blockL, map_cons, map_nil, mem_append, mem_cons, not_mem_nil, or_false,
    mem_flatMap, mem_range] at hp
  have := five_le_B t
  rcases hp with (rfl | rfl | rfl | rfl | rfl) | ⟨s, hs, rfl | rfl | rfl | rfl | rfl⟩
  all_goals first | (dsimp only; omega) | exact sl_lt hs (by omega)

theorem labDim_mono (d : ℕ) {k x : ℕ} (hx : x < B k) :
    ∀ j, labDim (envAt (E0 d) L (k + j)) x = labDim (envAt (E0 d) L k) x
  | 0 => rfl
  | j + 1 => by
    rw [← labDim_mono d hx j, ← add_assoc]
    exact labsMono d (k + j) x (lt_of_lt_of_le hx (B_le (by omega)))

theorem labDim_blk (d : ℕ) {t m x : ℕ} (htm : t < m) (hx : B t ≤ x) (hx' : x < B (t + 1)) :
    labDim (envAt (E0 d) L m) x = Fca.dimv d (((blockL t).lookup x).getD 0) := by
  obtain ⟨j, rfl⟩ : ∃ j, m = t + 1 + j := ⟨m - (t + 1), by omega⟩
  rw [labDim_mono d hx' j]
  have hL : L (t + 1) = L t ++ blockL t := by simp [L, range_succ, flatMap_append]
  simp only [labDim, envAt, E0, hL]
  rw [Fca.lookup_append_none x _ _ fun p hp e => by have := L_keys t p hp; omega]

theorem blockL_fst (n : ℕ) : (blockL n).map Prod.fst = [12, 13, 14, 15, 16].map (sl n) := by
  simp [blockL]

theorem labDim_cv (d : ℕ) {c t m : ℕ} (hc : c < 5) (h : t ≤ m) :
    labDim (envAt (E0 d) L m) (cv c t) = d ^ t := by
  cases t with
  | zero =>
    interval_cases c <;> simp [labDim, envAt, E0, L, cv, List.lookup, Fca.dimv]
  | succ t =>
    rw [cv, labDim_blk d (t := t) (by omega) (by simp [sl]) (by rw [B_succ]; unfold sl; omega)]
    have : (blockL t).lookup (sl t (12 + c)) = some (t + 2) := by
      interval_cases c <;> simp only [blockL, map_cons, map_nil, List.lookup] <;>
        (repeat' split) <;> first | rfl | simp_all [sl]
    rw [this]; rfl

theorem labDim_plain (d : ℕ) {n m t : ℕ} (hnm : n < m) (ht : t < 10 * n + 24)
    (hb : t < 12 ∨ 16 < t) : labDim (envAt (E0 d) L m) (sl n t) = d := by
  rw [labDim_blk d hnm (by simp [sl]) (by rw [B_succ]; unfold sl; omega),
    Fca.lookup_none _ _ (by
      rw [blockL_fst]; intro h
      obtain ⟨u, hu, e⟩ := mem_map.1 h
      have e := sl_injective n e
      subst e
      simp only [mem_cons, not_mem_nil, or_false] at hu
      omega)]
  rfl

/-! ## The spec in closed form -/

/-- The value of the top `t` of `n` digits at slot `s`, most significant first. -/
def topv (d : ℕ) (σ : ℕ → ℕ) (n s : ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => topv d σ n s t * d + σ (sl (n - 1 - t) s)

theorem topv_lt {d : ℕ} {σ : ℕ → ℕ} {n s : ℕ} (h : ∀ i < n, σ (sl i s) < d) :
    ∀ t ≤ n, topv d σ n s t < d ^ t
  | 0, _ => by simp [topv]
  | t + 1, ht => by
    have ih := topv_lt h t (by omega)
    have hdig := h (n - 1 - t) (by omega)
    simp only [topv, pow_succ]
    nlinarith

theorem topv_congr {d : ℕ} {σ τ : ℕ → ℕ} {n s : ℕ} (h : ∀ i < n, τ (sl i s) = σ (sl i s)) :
    ∀ t ≤ n, topv d τ n s t = topv d σ n s t
  | 0, _ => rfl
  | t + 1, ht => by
    simp only [topv, topv_congr h t (by omega), h (n - 1 - t) (by omega)]

theorem eD (d m e : ℕ) : (envAt (E0 d) L m).dimv e = Fca.dimv d e := rfl

theorem eP (d m p k : ℕ) : (envAt (E0 d) L m).phase p k = 0 := rfl

theorem mem_rhs {n : ℕ} {x : Node} (h : x ∈ [⟨.Z, [cv 0 0], [], 1, 0⟩, ⟨.Z, [cv 1 0], [], 1, 0⟩] ∨
    (∃ t < n, x = bN n 0 t ∨ x = bN n 1 t ∨ x = sN n 2 t ∨ x = sN n 3 t) ∨
    x ∈ [⟨.Z, [cv 2 n, cv 4 n], [cv 0 n], n + 1, 0⟩, ⟨.X, [cv 3 n], [cv 1 n, cv 4 n], n + 1, 0⟩,
      ⟨.Z, [], [cv 2 0], 1, 0⟩, ⟨.Z, [], [cv 3 0], 1, 0⟩]) : x ∈ (rhsN n).nodes := by
  simp only [mem_cons, not_mem_nil, or_false] at h
  simp only [rhsN, mem_append, mem_map, mem_range, mem_cons, not_mem_nil, or_false]
  rcases h with (rfl | rfl) | ⟨t, ht, rfl | rfl | rfl | rfl⟩ | (rfl | rfl | rfl | rfl)
  · exact .inl (.inl (.inl (.inl (.inl (.inl (.inl (.inl rfl)))))))
  · exact .inl (.inl (.inl (.inl (.inl (.inl (.inl (.inr rfl)))))))
  · exact .inl (.inl (.inl (.inl (.inl (.inl (.inr ⟨t, ht, rfl⟩))))))
  · exact .inl (.inl (.inl (.inl (.inl (.inr ⟨t, ht, rfl⟩)))))
  · exact .inl (.inl (.inl (.inr ⟨t, ht, rfl⟩)))
  · exact .inl (.inl (.inr ⟨t, ht, rfl⟩))
  · exact .inl (.inl (.inl (.inl (.inr (.inl rfl)))))
  · exact .inl (.inl (.inl (.inl (.inr (.inr rfl)))))
  · exact .inl (.inr (.inl rfl))
  · exact .inl (.inr (.inr rfl))

theorem dimv_one (d : ℕ) : Fca.dimv d 1 = 1 := by simp [Fca.dimv]

theorem rhs_support (d : ℕ) {n m : ℕ} {τ : ℕ → ℕ}
    (h : prodVal (envAt (E0 d) L m) (rhsN n).nodes τ ≠ 0) :
    ∀ c < 4, ∀ t ≤ n, τ (cv c t) = topv d τ n c t := by
  have hv := Fca.prodVal_ne h
  intro c hc t ht
  induction t with
  | zero =>
    have hz : ∀ x : Node, x.kind = .Z → x.dim = 1 → x.outs ++ x.ins = [cv c 0] →
        x ∈ (rhsN n).nodes → τ (cv c 0) = 0 := by
      intro x hk hd hl hm
      have h1 := hv x hm
      obtain ⟨k, os, is, dm, ph⟩ := x
      simp only at hk hd hl
      subst hk hd
      simp only [val, eD, dimv_one] at h1
      rw [show os.map τ ++ is.map τ = [τ (cv c 0)] by rw [← map_append, hl]; rfl,
        Fca.tZ_one _ _ (eP d m ph)] at h1
      by_contra hc'; exact h1 (if_neg hc')
    simp only [topv]
    interval_cases c
    · exact hz ⟨.Z, [cv 0 0], [], 1, 0⟩ rfl rfl rfl (mem_rhs (Or.inl (by simp)))
    · exact hz ⟨.Z, [cv 1 0], [], 1, 0⟩ rfl rfl rfl (mem_rhs (Or.inl (by simp)))
    · exact hz ⟨.Z, [], [cv 2 0], 1, 0⟩ rfl rfl rfl (mem_rhs (Or.inr (Or.inr (by simp))))
    · exact hz ⟨.Z, [], [cv 3 0], 1, 0⟩ rfl rfl rfl (mem_rhs (Or.inr (Or.inr (by simp))))
  | succ t ih =>
    have ih := ih (by omega)
    have hB : ∀ x : Node, x ∈ (rhsN n).nodes → val (envAt (E0 d) L m) τ x ≠ 0 → (x = bN n c t ∨
        x = sN n c t) → τ (cv c (t + 1)) = τ (cv c t) * d + τ (sl (n - 1 - t) c) := by
      intro x _ h1 hx
      rcases hx with rfl | rfl <;>
      · simp only [val, bN, sN, map_cons, map_nil] at h1
        exact Fca.tB_eq h1
    have e : τ (cv c (t + 1)) = τ (cv c t) * d + τ (sl (n - 1 - t) c) := by
      interval_cases c
      · exact hB _ (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inl rfl⟩))) (hv _
          (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inl rfl⟩)))) (Or.inl rfl)
      · exact hB _ (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inr (Or.inl rfl)⟩))) (hv _
          (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inr (Or.inl rfl)⟩)))) (Or.inl rfl)
      · exact hB _ (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inr (Or.inr (Or.inl rfl))⟩)))
          (hv _ (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inr (Or.inr (Or.inl rfl))⟩))))
          (Or.inr rfl)
      · exact hB _ (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inr (Or.inr (Or.inr rfl))⟩)))
          (hv _ (mem_rhs (Or.inr (Or.inl ⟨t, by omega, Or.inr (Or.inr (Or.inr rfl))⟩))))
          (Or.inr rfl)
    rw [e, ih]; rfl

theorem rhs_cz (d : ℕ) {n m : ℕ} {τ : ℕ → ℕ}
    (h : prodVal (envAt (E0 d) L m) (rhsN n).nodes τ ≠ 0) : τ (cv 4 n) = τ (cv 0 n) :=
  Z_eq _ τ ⟨.Z, [cv 2 n, cv 4 n], [cv 0 n], n + 1, 0⟩ rfl
    (Fca.prodVal_ne h _ (mem_rhs (Or.inr (Or.inr (by simp)))))
    (by simp [Node.labels]) (by simp [Node.labels])

theorem cv_inj {c c' t t' : ℕ} (hc : c < 5) (hc' : c' < 5) (h : cv c t = cv c' t') :
    c = c' ∧ t = t' := by
  cases t with
  | zero =>
    cases t' with
    | zero => exact ⟨h, rfl⟩
    | succ t' => have := five_le_sl t' (12 + c'); simp only [cv] at h; omega
  | succ t =>
    cases t' with
    | zero => have := five_le_sl t (12 + c); simp only [cv] at h; omega
    | succ t' =>
      have := sl_inj (by omega) (by omega) h
      exact ⟨by omega, by omega⟩

theorem rhsInt_nodup (n : ℕ) : (rhsN n).int.Nodup := by
  simp only [rhsN]
  rw [nodup_append]
  refine ⟨?_, ?_, ?_⟩
  · simp only [nodup_cons, mem_cons, not_mem_nil, or_false, not_or, nodup_nil, and_true,
      not_false_eq_true]
    and_intros
    all_goals first | trivial | (intro e; have := (cv_inj (by omega) (by omega) e).1; omega)
  · rw [nodup_flatMap]
    refine ⟨fun t _ => ?_, ?_⟩
    · simp only [nodup_cons, mem_cons, not_mem_nil, or_false, not_or, nodup_nil, and_true,
        not_false_eq_true]
      and_intros
      all_goals (intro e; have := (cv_inj (by omega) (by omega) e).1; omega)
    · refine (pairwise_lt_range (n := n)).imp fun {a b} hab => ?_
      simp only [Function.onFun, List.disjoint_left, mem_cons, not_mem_nil, or_false]
      intro x hx hy
      rcases hx with rfl | rfl | rfl | rfl <;> rcases hy with e | e | e | e <;>
      · have := (cv_inj (by omega) (by omega) e).2; omega
  · intro a ha b hb e
    subst e
    simp only [mem_cons, not_mem_nil, or_false] at ha
    simp only [mem_flatMap, mem_range, mem_cons, not_mem_nil, or_false] at hb
    obtain ⟨t, _, hb⟩ := hb
    rcases ha with rfl | rfl | rfl | rfl | rfl <;> rcases hb with e | e | e | e <;>
    · have := cv_inj (by omega) (by omega) e; omega

/-- The pinned values of the spec's summed labels. -/
def pinsR (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : List (ℕ × ℕ) :=
  [(cv 0 0, 0), (cv 1 0, 0), (cv 2 0, 0), (cv 3 0, 0), (cv 4 n, topv d σ n 0 n)] ++
    (range n).flatMap fun t => [(cv 0 (t + 1), topv d σ n 0 (t + 1)),
      (cv 1 (t + 1), topv d σ n 1 (t + 1)), (cv 2 (t + 1), topv d σ n 2 (t + 1)),
      (cv 3 (t + 1), topv d σ n 3 (t + 1))]

/-- `σ` with the spec's summed labels at their pinned values. -/
def tauR (d : ℕ) (σ : ℕ → ℕ) (n x : ℕ) : ℕ := ((pinsR d σ n).lookup x).getD (σ x)

theorem pinsR_fst (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : (pinsR d σ n).map Prod.fst = (rhsN n).int := by
  simp [pinsR, rhsN, map_flatMap]

theorem tauR_off {d : ℕ} {σ : ℕ → ℕ} {n x : ℕ} (h : x ∉ (rhsN n).int) : tauR d σ n x = σ x := by
  rw [tauR, Fca.lookup_none _ _ (by rw [pinsR_fst]; exact h)]; rfl

theorem pinsR_nd (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : ((pinsR d σ n).map Prod.fst).Nodup := by
  rw [pinsR_fst]; exact rhsInt_nodup n

theorem tauR_cv {d : ℕ} {σ : ℕ → ℕ} {n c t : ℕ} (hc : c < 4) (ht : t ≤ n) :
    tauR d σ n (cv c t) = topv d σ n c t := by
  rw [tauR, Fca.lookup_of_mem _ (pinsR_nd d σ n) (v := topv d σ n c t) ?_]
  · rfl
  cases t with
  | zero => interval_cases c <;> simp [pinsR, topv]
  | succ t =>
    refine mem_append_right _ (mem_flatMap.2 ⟨t, mem_range.2 (by omega), ?_⟩)
    interval_cases c <;> simp

theorem tauR_cz {d : ℕ} {σ : ℕ → ℕ} {n : ℕ} : tauR d σ n (cv 4 n) = topv d σ n 0 n := by
  rw [tauR, Fca.lookup_of_mem _ (pinsR_nd d σ n) (v := topv d σ n 0 n) (by simp [pinsR])]; rfl

theorem sl_notin_rhsInt {n i s : ℕ} (hs : s < 12) : sl i s ∉ (rhsN n).int := by
  intro h
  obtain ⟨c, hc, t, _, e⟩ := rhsInt_mem h
  cases t with
  | zero => have := five_le_sl i s; simp only [cv] at e; omega
  | succ t => have := (sl_inj (by omega) (by omega) e).2; omega

/-- The X spider of dimension `D` with zero phase, one output and two inputs is the sum mod `D`. -/
theorem tX_add (D : ℕ) (hD : 0 < D) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (o i j : ℕ) :
    tX D φ [o] [i, j] = (rt D)⁻¹ * if o % D = (i + j) % D then 1 else 0 := by
  have hr : (starRingEnd ℂ) (rt D) = rt D := Complex.conj_ofReal _
  have hterm : ∀ k : ℕ, ex (φ k) * ([o].map fun o => fou D o k).prod *
      ([i, j].map fun i => (starRingEnd ℂ) (fou D i k)).prod =
      (rt D * rt D * rt D)⁻¹ * (om D ^ (k * o) * (starRingEnd ℂ) (om D) ^ (k * (i + j))) := by
    intro k
    simp only [hφ, ex_zero, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one,
      fou, map_div₀, map_pow, one_mul]
    rw [hr, mul_add, pow_add, mul_comm o k, mul_comm i k, mul_comm j k]
    field_simp
  unfold tX
  simp only [hterm, ← Finset.mul_sum, char_sum D hD]
  have hne := rt_ne D hD
  split_ifs
  · rw [mul_one, show rt D * rt D * rt D = rt D * (D : ℂ) by rw [mul_assoc, rt_sq]]
    have hD' : (D : ℂ) ≠ 0 := by exact_mod_cast hD.ne'
    field_simp
  · simp

/-- A Z spider of dimension `D` with zero phase on legs `[b, a, a]`. -/
theorem tZ_baa (D a b : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (ha : a < D) :
    tZ D φ [b, a, a] = if b = a then 1 else 0 := by
  unfold tZ
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, hφ, ex_zero]
  rw [Finset.sum_eq_single a (fun k _ hk => if_neg fun h' => hk h'.2.1.symm)
    (fun h' => absurd (Finset.mem_range.2 ha) h')]
  by_cases h : b = a <;> simp [h]

/-- The spec's sum and copy factor at `n` digits. -/
noncomputable def IndS (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : ℂ :=
  (if topv d σ n 3 n % d ^ n = (topv d σ n 1 n + topv d σ n 0 n) % d ^ n then 1 else 0) *
    (if topv d σ n 2 n = topv d σ n 0 n then 1 else 0)

/-- The spec's pending-leg factors at `n` digits. -/
def pinsN (n : ℕ) (σ : ℕ → ℕ) : ℂ :=
  ((range n).map fun i => (if σ (pend n i) = σ (sl i 1) then (1 : ℂ) else 0) *
    (if σ (pendO n i) = σ (sl i 3) then 1 else 0) *
    (if σ (pendA n i) = σ (sl i 0) then 1 else 0)).prod

theorem cut_off_int {n x : ℕ} (hc : x ∈ cut n) (h : x ∈ (rhsN n).int) : False :=
  (fitsR n).2 n le_rfl x h hc

/-- The spec in closed form: `1 / √(d ^ n)`, the sum and copy factor of the digit values, the
pending legs on the inputs and outputs. -/
theorem eval_rhsN (d : ℕ) (hd : 0 < d) {n m : ℕ} (hm : n ≤ m) (σ : ℕ → ℕ)
    (hs : ∀ s < 4, ∀ i < n, σ (sl i s) < d) :
    eval (envAt (E0 d) L m) (rhsN n) σ = (rt (d ^ n))⁻¹ * IndS d σ n * pinsN n σ := by
  have hdig : ∀ s < 12, ∀ i, tauR d σ n (sl i s) = σ (sl i s) :=
    fun s hs i => tauR_off (sl_notin_rhsInt hs)
  unfold eval
  rw [Fca.sumOver_single _ _ (rhsInt_nodup n) _ σ (tauR d σ n) (fun x hx => tauR_off hx) ?hr ?hs]
  case hr =>
    intro l hl
    obtain ⟨c, hc, t, ht, rfl⟩ := rhsInt_mem hl
    rw [labDim_cv d hc (by omega)]
    rcases Nat.lt_or_ge c 4 with hc4 | hc4
    · rw [tauR_cv hc4 ht]; exact topv_lt (hs c hc4) t ht
    · obtain rfl : c = 4 := by omega
      have : t = n := by
        simp only [rhsN, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap,
          mem_range] at hl
        rcases hl with (e | e | e | e | e) | ⟨t', _, e | e | e | e⟩ <;>
        · have := cv_inj (by omega) (by omega) e; omega
      subst this
      rw [tauR_cz]; exact topv_lt (hs 0 (by omega)) t le_rfl
  case hs =>
    intro τ h0 _ hne l hl
    have hsup := rhs_support d hne
    have hT : ∀ c < 4, ∀ t ≤ n, topv d τ n c t = topv d σ n c t := fun c hc t ht =>
      topv_congr (fun i _ => h0 _ (sl_notin_rhsInt (by omega))) t ht
    obtain ⟨c, hc, t, ht, rfl⟩ := rhsInt_mem hl
    rcases Nat.lt_or_ge c 4 with hc4 | hc4
    · rw [hsup c hc4 t ht, tauR_cv hc4 ht, hT c hc4 t ht]
    · obtain rfl : c = 4 := by omega
      have : t = n := by
        simp only [rhsN, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap,
          mem_range] at hl
        rcases hl with (e | e | e | e | e) | ⟨t', _, e | e | e | e⟩ <;>
        · have := cv_inj (by omega) (by omega) e; omega
      subst this
      rw [rhs_cz d hne, hsup 0 (by omega) t le_rfl, tauR_cz, hT 0 (by omega) t le_rfl]
  have hH : ∀ c < 2, val (envAt (E0 d) L m) (tauR d σ n) ⟨.Z, [cv c 0], [], 1, 0⟩ = 1 := fun c hc => by
    simp only [val, map_cons, map_nil, append_nil, eD, dimv_one]
    rw [Fca.tZ_one _ _ (eP d m 0), tauR_cv (by omega) (Nat.zero_le _)]; rfl
  have hT : ∀ c < 4, 2 ≤ c → val (envAt (E0 d) L m) (tauR d σ n) ⟨.Z, [], [cv c 0], 1, 0⟩ = 1 := fun c hc _ => by
    simp only [val, map_cons, map_nil, nil_append, eD, dimv_one]
    rw [Fca.tZ_one _ _ (eP d m 0), tauR_cv hc (Nat.zero_le _)]; rfl
  have hZ : val (envAt (E0 d) L m) (tauR d σ n) ⟨.Z, [cv 2 n, cv 4 n], [cv 0 n], n + 1, 0⟩ =
      if topv d σ n 2 n = topv d σ n 0 n then 1 else 0 := by
    simp only [val, map_cons, map_nil, cons_append, nil_append, eD]
    rw [tauR_cz, tauR_cv (c := 2) (by omega) le_rfl, tauR_cv (c := 0) (by omega) le_rfl]
    exact tZ_baa _ _ _ _ (eP d m 0) (topv_lt (hs 0 (by omega)) n le_rfl)
  have hX : val (envAt (E0 d) L m) (tauR d σ n) ⟨.X, [cv 3 n], [cv 1 n, cv 4 n], n + 1, 0⟩ =
      (rt (d ^ n))⁻¹ *
        if topv d σ n 3 n % d ^ n = (topv d σ n 1 n + topv d σ n 0 n) % d ^ n then 1 else 0 := by
    simp only [val, map_cons, map_nil, eD]
    rw [tauR_cz, tauR_cv (c := 3) (by omega) le_rfl, tauR_cv (c := 1) (by omega) le_rfl]
    exact tX_add _ (pow_pos hd n) _ (eP d m 0) _ _ _
  have hB : ∀ c < 2, ((range n).map fun t => val (envAt (E0 d) L m) (tauR d σ n) (bN n c t)).prod = 1 :=
    fun c hc => by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hx
    have ht := mem_range.1 ht
    simp only [val, bN, map_cons, map_nil, eD, Fca.dimv]
    rw [tauR_cv (by omega) (by omega), tauR_cv (by omega) (by omega), hdig c (by omega)]
    unfold tB
    rw [if_pos ⟨topv_lt (hs c (by omega)) t (by omega), hs c (by omega) _ (by omega), rfl⟩]
  have hS : ∀ c < 4, 2 ≤ c → ((range n).map fun t => val (envAt (E0 d) L m) (tauR d σ n) (sN n c t)).prod = 1 :=
    fun c hc _ => by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hx
    have ht := mem_range.1 ht
    simp only [val, sN, map_cons, map_nil, eD, Fca.dimv]
    rw [tauR_cv hc (by omega), tauR_cv hc (by omega), hdig c (by omega)]
    unfold tB
    rw [if_pos ⟨topv_lt (hs c hc) t (by omega), hs c hc _ (by omega), rfl⟩]
  have hP : (((range n).flatMap fun i => [zN [pend n i] [sl i 1], zN [pendO n i] [sl i 3],
      zN [pendA n i] [sl i 0]]).map (val (envAt (E0 d) L m) (tauR d σ n))).prod = pinsN n σ := by
    rw [map_flatMap, prod_flatMapC]
    unfold pinsN
    congr 1
    refine List.map_congr_left fun i hi => ?_
    have hi := mem_range.1 hi
    simp only [map_cons, map_nil, prod_cons, prod_nil, mul_one, zN, val, cons_append, nil_append,
      eD, Fca.dimv]
    rw [tauR_off (fun h => cut_off_int (pend_mem hi) h),
      tauR_off (fun h => cut_off_int (pendO_mem hi) h),
      tauR_off (fun h => cut_off_int (pendA_mem hi) h),
      hdig 0 (by omega), hdig 1 (by omega), hdig 3 (by omega),
      Fca.tZ_two _ _ _ _ (eP d m 0) (hs 1 (by omega) i hi),
      Fca.tZ_two _ _ _ _ (eP d m 0) (hs 3 (by omega) i hi),
      Fca.tZ_two _ _ _ _ (eP d m 0) (hs 0 (by omega) i hi)]
    ring
  simp only [rhsN, prodVal, map_append, prod_append, map_cons, map_nil, prod_cons, prod_nil,
    mul_one, map_map]
  try simp only [Function.comp_def]
  rw [hH 0 (by omega), hH 1 (by omega), hB 0 (by omega), hB 1 (by omega), hZ, hX,
    hS 2 (by omega) (by omega), hS 3 (by omega) (by omega), hT 2 (by omega) (by omega),
    hT 3 (by omega) (by omega), hP]
  unfold IndS
  ring

/-! ## Copy `n` in closed form -/

theorem nodup_blocks (n c : ℕ) (A R : List ℕ) (hA : A.Nodup) (hR : R.Nodup)
    (hR8 : ∀ r ∈ R, r < 10) (hAc : ∀ a ∈ A, a < c) :
    (A ++ (range n).flatMap fun i => R.map fun r => c + 10 * i + r).Nodup := by
  rw [nodup_append]
  refine ⟨hA, ?_, ?_⟩
  · rw [nodup_flatMap]
    refine ⟨fun i _ => hR.map fun a b h => by omega, ?_⟩
    refine (pairwise_lt_range (n := n)).imp fun {i j} hij => ?_
    simp only [Function.onFun, List.disjoint_left, mem_map]
    rintro _ ⟨r, hr, rfl⟩ ⟨r', hr', e⟩
    have := hR8 r hr; have := hR8 r' hr'; omega
  · intro a ha b hb e
    simp only [mem_flatMap, mem_range, mem_map] at hb
    obtain ⟨i, _, r, _, rfl⟩ := hb
    have := hAc a ha; omega

theorem pinC_eq (n : ℕ) : pinC n = (([4, 6, 7, 8, 11] ++ (range n).flatMap fun i =>
    [0, 1, 2, 3, 4, 5].map fun r => 23 + 10 * i + r)).map (sl n) := by
  simp only [pinC, map_append, map_flatMap, map_cons, map_nil, sl]
  ring_nf

theorem pinC_nodup (n : ℕ) : (pinC n).Nodup := by
  rw [pinC_eq]
  exact (nodup_blocks n 23 _ _ (by decide) (by decide) (by decide) (by decide)).map
    (sl_injective n)

/-- The dimension of the pair `(n, i)`. -/
def pD (d n i : ℕ) : ℕ := d ^ (n - i + 1)

/-- The values the nodes of copy `n` pin its pinned labels to, read from `ρ`. -/
def valsC (n : ℕ) (ρ : ℕ → ℕ) : List (ℕ × ℕ) :=
  [(sl n 4, ρ (sl n 1)), (sl n 6, ρ (sl n 0)), (sl n 7, ρ (sl n 5)), (sl n 8, ρ (sl n 5)),
    (sl n 11, ρ (sl n 3))] ++
  (range n).flatMap fun i => [(sl n (23 + 10 * i), ρ (pend n i)), (sl n (24 + 10 * i), ρ (sl n 5)),
    (sl n (25 + 10 * i), ρ (pendA n i)), (sl n (26 + 10 * i), ρ (sl n 5)),
    (sl n (27 + 10 * i), ρ (pendO n i)), (sl n (28 + 10 * i), ρ (sl n 5))]

/-- `ρ` with the pinned labels of copy `n` at their pins. -/
def tauC (n : ℕ) (ρ : ℕ → ℕ) (x : ℕ) : ℕ := ((valsC n ρ).lookup x).getD (ρ x)

theorem valsC_fst (n : ℕ) (ρ : ℕ → ℕ) : (valsC n ρ).map Prod.fst = pinC n := by
  simp [valsC, pinC, map_flatMap]

theorem tauC_of_mem {n x y : ℕ} {ρ : ℕ → ℕ} (h : (x, y) ∈ valsC n ρ) : tauC n ρ x = y := by
  rw [tauC, Fca.lookup_of_mem _ (by rw [valsC_fst]; exact pinC_nodup n) h]; rfl

theorem tauC_off {n x : ℕ} {ρ : ℕ → ℕ} (h : x ∉ pinC n) : tauC n ρ x = ρ x := by
  rw [tauC, Fca.lookup_none _ _ (by rw [valsC_fst]; exact h)]; rfl

theorem pend_ne {n i t : ℕ} (hi : i < n) (ht : t < 10 * n + 24) : pend n i ≠ sl n t := fun h =>
  absurd (sl_inj (by omega) ht h).1 (by omega)

theorem pendO_ne {n i t : ℕ} (hi : i < n) (ht : t < 10 * n + 24) : pendO n i ≠ sl n t := fun h =>
  absurd (sl_inj (by omega) ht h).1 (by omega)

theorem pendA_ne {n i t : ℕ} (hi : i < n) (ht : t < 10 * n + 24) : pendA n i ≠ sl n t := fun h =>
  absurd (sl_inj (by omega) ht h).1 (by omega)

theorem notin_pinC {n t : ℕ} (h4 : t ≠ 4) (h6 : t ≠ 6) (h7 : t ≠ 7) (h8 : t ≠ 8) (h11 : t ≠ 11)
    (hb : ∀ i < n, ∀ r < 6, t ≠ 23 + 10 * i + r) : sl n t ∉ pinC n := by
  rw [pinC_eq]
  intro h
  obtain ⟨u, hu, e⟩ := mem_map.1 h
  have e := sl_injective n e
  subst e
  simp only [mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range, mem_map] at hu
  rcases hu with (rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, r, hr, rfl⟩
  all_goals first | omega | exact hb i hi r (by omega) rfl

theorem notin_low {n t : ℕ} (ht : t < 23) (h4 : t ≠ 4) (h6 : t ≠ 6) (h7 : t ≠ 7) (h8 : t ≠ 8)
    (h11 : t ≠ 11) : sl n t ∉ pinC n :=
  notin_pinC h4 h6 h7 h8 h11 fun _ _ _ _ => by omega

theorem pinC_lt {n x : ℕ} (h : x ∈ pinC n) : ∃ u < 10 * n + 24, x = sl n u := by
  rw [pinC_eq] at h
  obtain ⟨u, hu, rfl⟩ := mem_map.1 h
  refine ⟨u, ?_, rfl⟩
  simp only [mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range, mem_map] at hu
  rcases hu with (rfl | rfl | rfl | rfl | rfl) | ⟨j, hj, r, hr, rfl⟩
  all_goals omega

theorem pend_notin {n i : ℕ} (hi : i < n) : pend n i ∉ pinC n := fun h => by
  obtain ⟨u, hu, e⟩ := pinC_lt h; exact pend_ne hi hu e

theorem pendO_notin {n i : ℕ} (hi : i < n) : pendO n i ∉ pinC n := fun h => by
  obtain ⟨u, hu, e⟩ := pinC_lt h; exact pendO_ne hi hu e

theorem pendA_notin {n i : ℕ} (hi : i < n) : pendA n i ∉ pinC n := fun h => by
  obtain ⟨u, hu, e⟩ := pinC_lt h; exact pendA_ne hi hu e

theorem valsC_congr {n : ℕ} {τ ρ : ℕ → ℕ} (h : ∀ x ∉ pinC n, τ x = ρ x) :
    valsC n τ = valsC n ρ := by
  have h0 := h _ (notin_low (n := n) (t := 0) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega))
  have h1 := h _ (notin_low (n := n) (t := 1) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega))
  have h3 := h _ (notin_low (n := n) (t := 3) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega))
  have h5 := h _ (notin_low (n := n) (t := 5) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega))
  simp only [valsC, h0, h1, h3, h5]
  congr 1
  refine flatMap_congr fun i hi => ?_
  rw [h _ (pend_notin (mem_range.1 hi)), h _ (pendO_notin (mem_range.1 hi)),
    h _ (pendA_notin (mem_range.1 hi))]

theorem mem_fix {n : ℕ} {x : Node} (h : x ∈ fixN n) : x ∈ (cp n).nodes := mem_append_left _ h

theorem mem_cpP {n i : ℕ} (hi : i < n) {x : Node} (h : x ∈ pairN n i) : x ∈ (cp n).nodes :=
  mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi, h⟩)

theorem zeq {E : Env} {τ : ℕ → ℕ} {os is : List ℕ} (h : val E τ (zN os is) ≠ 0)
    {a b : ℕ} (ha : a ∈ os ++ is) (hb : b ∈ os ++ is) : τ a = τ b :=
  Z_eq E τ _ rfl h ha hb

theorem mem_legs {n i : ℕ} (hi : i < n) :
    sl n (24 + 10 * i) ∈ legs n ∧ sl n (26 + 10 * i) ∈ legs n ∧ sl n (28 + 10 * i) ∈ legs n := by
  simp only [legs, mem_flatMap, mem_range, mem_cons, not_mem_nil, or_false]
  exact ⟨⟨i, hi, Or.inl rfl⟩, ⟨i, hi, Or.inr (Or.inl rfl)⟩, ⟨i, hi, Or.inr (Or.inr rfl)⟩⟩

theorem cp_support {d m n : ℕ} {τ : ℕ → ℕ} (h : prodVal (envAt (E0 d) L m) (cp n).nodes τ ≠ 0) :
    ∀ l ∈ pinC n, τ l = tauC n τ l := by
  have hv := Fca.prodVal_ne h
  have hm : val (envAt (E0 d) L m) τ (zN (sl n 8 :: legs n) [sl n 5, sl n 7]) ≠ 0 :=
    hv _ (mem_fix (by simp [fixN]))
  have e4 : τ (sl n 4) = τ (sl n 1) := zeq (hv (zN [sl n 4, sl n (20 + 10 * n)] [sl n 1])
    (mem_fix (by simp [fixN]))) (by simp) (by simp)
  have e6 : τ (sl n 6) = τ (sl n 0) := zeq (hv (zN [sl n 6, sl n 2, sl n (22 + 10 * n)] [sl n 0])
    (mem_fix (by simp [fixN]))) (by simp) (by simp)
  have e7 : τ (sl n 7) = τ (sl n 5) := zeq hm (by simp) (by simp)
  have e8 : τ (sl n 8) = τ (sl n 5) := zeq hm (by simp) (by simp)
  have e11 : τ (sl n 11) = τ (sl n 3) := zeq (hv (zN [sl n 3, sl n (21 + 10 * n)] [sl n 11])
    (mem_fix (by simp [fixN]))) (by simp) (by simp)
  have el : ∀ i < n, τ (sl n (24 + 10 * i)) = τ (sl n 5) ∧ τ (sl n (26 + 10 * i)) = τ (sl n 5) ∧
      τ (sl n (28 + 10 * i)) = τ (sl n 5) := fun i hi => by
    obtain ⟨l1, l2, l3⟩ := mem_legs hi
    exact ⟨zeq hm (mem_append_left _ (mem_cons_of_mem _ l1)) (by simp),
      zeq hm (mem_append_left _ (mem_cons_of_mem _ l2)) (by simp),
      zeq hm (mem_append_left _ (mem_cons_of_mem _ l3)) (by simp)⟩
  have ey : ∀ i < n, τ (sl n (23 + 10 * i)) = τ (pend n i) := fun i hi =>
    zeq (hv (zN [sl n (23 + 10 * i), sl n (20 + 10 * i)] [pend n i])
      (mem_cpP hi (by simp [pairN]))) (by simp) (by simp)
  have ya : ∀ i < n, τ (sl n (25 + 10 * i)) = τ (pendA n i) := fun i hi =>
    zeq (hv (zN [sl n (25 + 10 * i), sl n (22 + 10 * i)] [pendA n i])
      (mem_cpP hi (by simp [pairN]))) (by simp) (by simp)
  have yo : ∀ i < n, τ (sl n (27 + 10 * i)) = τ (pendO n i) := fun i hi =>
    zeq (hv (zN [sl n (27 + 10 * i), sl n (21 + 10 * i)] [pendO n i])
      (mem_cpP hi (by simp [pairN]))) (by simp) (by simp)
  intro l hl
  simp only [pinC, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range] at hl
  have mem : ∀ {i : ℕ} {k : ℕ × ℕ}, i < n → k ∈ [(sl n (23 + 10 * i), τ (pend n i)),
      (sl n (24 + 10 * i), τ (sl n 5)), (sl n (25 + 10 * i), τ (pendA n i)),
      (sl n (26 + 10 * i), τ (sl n 5)), (sl n (27 + 10 * i), τ (pendO n i)),
      (sl n (28 + 10 * i), τ (sl n 5))] → k ∈ valsC n τ := fun hi hk =>
    mem_append_right _ (mem_flatMap.2 ⟨_, mem_range.2 hi, hk⟩)
  rcases hl with (rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, rfl | rfl | rfl | rfl | rfl | rfl⟩
  · rw [tauC_of_mem (y := τ (sl n 1)) (by simp [valsC, sl]), e4]
  · rw [tauC_of_mem (y := τ (sl n 0)) (by simp [valsC, sl]), e6]
  · rw [tauC_of_mem (y := τ (sl n 5)) (by simp [valsC, sl]), e7]
  · rw [tauC_of_mem (y := τ (sl n 5)) (by simp [valsC, sl]), e8]
  · rw [tauC_of_mem (y := τ (sl n 3)) (by simp [valsC, sl]), e11]
  · rw [tauC_of_mem (y := τ (pend n i)) (mem hi (by simp)), ey i hi]
  · rw [tauC_of_mem (y := τ (sl n 5)) (mem hi (by simp)), (el i hi).1]
  · rw [tauC_of_mem (y := τ (pendA n i)) (mem hi (by simp)), ya i hi]
  · rw [tauC_of_mem (y := τ (sl n 5)) (mem hi (by simp)), (el i hi).2.1]
  · rw [tauC_of_mem (y := τ (pendO n i)) (mem hi (by simp)), yo i hi]
  · rw [tauC_of_mem (y := τ (sl n 5)) (mem hi (by simp)), (el i hi).2.2]

theorem tauC_congr {n : ℕ} {τ ρ : ℕ → ℕ} (h : ∀ x ∉ pinC n, τ x = ρ x) (x : ℕ) :
    tauC n τ x = tauC n ρ x := by
  by_cases hx : x ∈ pinC n
  · obtain ⟨v, hv⟩ := Fca.lookup_some_of_mem (valsC n ρ) (by rw [valsC_fst]; exact hx)
    simp only [tauC, valsC_congr h, hv, Option.getD_some]
  · rw [tauC_off hx, tauC_off hx, h x hx]

/-- The pending-leg factors of copy `n`. -/
def pinsCp (n : ℕ) (σ : ℕ → ℕ) : ℂ :=
  (if σ (sl n (20 + 10 * n)) = σ (sl n 1) then 1 else 0) *
    (if σ (sl n (21 + 10 * n)) = σ (sl n 3) then 1 else 0) *
    (if σ (sl n (22 + 10 * n)) = σ (sl n 0) then 1 else 0) *
    ((range n).map fun i => (if σ (sl n (20 + 10 * i)) = σ (pend n i) then (1 : ℂ) else 0) *
      (if σ (sl n (21 + 10 * i)) = σ (pendO n i) then 1 else 0) *
      (if σ (sl n (22 + 10 * i)) = σ (pendA n i) then 1 else 0)).prod

/-- The copy factor of copy `n`: its `a` output equals its `a` input. -/
def aeqC (n : ℕ) (σ : ℕ → ℕ) : ℂ := if σ (sl n 2) = σ (sl n 0) then 1 else 0

/-- The pair factors of copy `n` at the Fourier value `x`. -/
noncomputable def pairsF (d n : ℕ) (σ : ℕ → ℕ) (x : ℕ) : ℂ :=
  ((range n).map fun i => tP (pD d n i) 1 [σ (pend n i), x] * tP (pD d n i) 1 [σ (pendA n i), x] *
    tP (pD d n i) (-1) [σ (pendO n i), x]).prod

/-- A Z spider of dimension `D` with zero phase on legs `[a, b, c, a]`. -/
theorem tZ4 (D a b c : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (ha : a < D) :
    tZ D φ [a, b, c, a] = (if b = a then 1 else 0) * (if c = a then 1 else 0) := by
  unfold tZ
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, hφ, ex_zero]
  rw [Finset.sum_eq_single a (fun k _ hk => if_neg fun h' => hk h'.1.symm)
    (fun h' => absurd (Finset.mem_range.2 ha) h')]
  by_cases h : b = a <;> by_cases h' : c = a <;> simp [h, h']

/-- Copy `n`'s nodes at the pinned labels. -/
theorem prod_cp (d : ℕ) (n m : ℕ) (ρ : ℕ → ℕ) (h0 : ρ (sl n 0) < d) (h1 : ρ (sl n 1) < d)
    (h3 : ρ (sl n 3) < d) (h5 : ρ (sl n 5) < d) (hA : ∀ i < n, ρ (pend n i) < d)
    (hB : ∀ i < n, ρ (pendO n i) < d) (hC : ∀ i < n, ρ (pendA n i) < d) :
    prodVal (envAt (E0 d) L m) (cp n).nodes (tauC n ρ) = pinsCp n ρ * (aeqC n ρ *
      (fou d (ρ (sl n 5)) (ρ (sl n 1)) * fou d (ρ (sl n 5)) (ρ (sl n 0)) *
        (fou d (ρ (sl n 9)) (ρ (sl n 5)) * fou d (ρ (sl n 10)) (ρ (sl n 9)) *
          fou d (ρ (sl n 3)) (ρ (sl n 10))) * pairsF d n ρ (ρ (sl n 5)))) := by
  have t4 : tauC n ρ (sl n 4) = ρ (sl n 1) := tauC_of_mem (by simp [valsC, sl])
  have t6 : tauC n ρ (sl n 6) = ρ (sl n 0) := tauC_of_mem (by simp [valsC, sl])
  have t7 : tauC n ρ (sl n 7) = ρ (sl n 5) := tauC_of_mem (by simp [valsC, sl])
  have t8 : tauC n ρ (sl n 8) = ρ (sl n 5) := tauC_of_mem (by simp [valsC, sl])
  have t11 : tauC n ρ (sl n 11) = ρ (sl n 3) := tauC_of_mem (by simp [valsC, sl])
  have o : ∀ t, t < 23 → t ≠ 4 → t ≠ 6 → t ≠ 7 → t ≠ 8 → t ≠ 11 →
      tauC n ρ (sl n t) = ρ (sl n t) := fun t ht h4 h6 h7 h8 h11 =>
    tauC_off (notin_low ht h4 h6 h7 h8 h11)
  have oP : ∀ r < 3, tauC n ρ (sl n (20 + 10 * n + r)) = ρ (sl n (20 + 10 * n + r)) :=
    fun r hr => tauC_off (notin_pinC (by omega) (by omega) (by omega) (by omega) (by omega)
      fun _ _ _ _ => by omega)
  have tl : ∀ l ∈ legs n, tauC n ρ l = ρ (sl n 5) := by
    intro l hl
    simp only [legs, mem_flatMap, mem_range, mem_cons, not_mem_nil, or_false] at hl
    obtain ⟨i, hi, rfl | rfl | rfl⟩ := hl
    all_goals refine tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi, ?_⟩))
    all_goals simp
  have hfix : ((fixN n).map (val (envAt (E0 d) L m) (tauC n ρ))).prod =
      (if ρ (sl n (20 + 10 * n)) = ρ (sl n 1) then 1 else 0) *
      (if ρ (sl n (21 + 10 * n)) = ρ (sl n 3) then 1 else 0) *
      (if ρ (sl n (22 + 10 * n)) = ρ (sl n 0) then 1 else 0) * aeqC n ρ *
      (fou d (ρ (sl n 5)) (ρ (sl n 1)) * fou d (ρ (sl n 5)) (ρ (sl n 0)) *
        (fou d (ρ (sl n 9)) (ρ (sl n 5)) * fou d (ρ (sl n 10)) (ρ (sl n 9)) *
          fou d (ρ (sl n 3)) (ρ (sl n 10)))) := by
    have hm : val (envAt (E0 d) L m) (tauC n ρ) (zN (sl n 8 :: legs n) [sl n 5, sl n 7]) = 1 := by
      simp only [val, zN, eD, Fca.dimv]
      rw [Fca.tZ_all d (ρ (sl n 5)) _ _ (by simp) (fun v hv => by
        simp only [map_cons, map_nil, mem_append, mem_cons, mem_map, not_mem_nil,
          or_false] at hv
        rcases hv with (rfl | ⟨l, hl, rfl⟩) | rfl | rfl
        · exact t8
        · exact tl l hl
        · exact o 5 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
        · exact t7) h5, eP, ex_zero]
    have e20 := oP 0 (by omega)
    have e21 := oP 1 (by omega)
    have e22 := oP 2 (by omega)
    simp only [add_zero] at e20
    simp only [fixN, map_cons, map_nil, prod_cons, prod_nil, mul_one]
    rw [hm]
    simp only [val, zN, fN, cons_append, nil_append, map_cons, map_nil, eD, Fca.dimv]
    rw [t4, t6, t11, e20, show 20 + 10 * n + 1 = 21 + 10 * n by omega,
      show 20 + 10 * n + 2 = 22 + 10 * n by omega] at *
    rw [e21, e22, o 0 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega),
      o 1 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega),
      o 2 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega),
      o 3 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega),
      o 5 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega),
      o 9 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega),
      o 10 (by omega) (by omega) (by omega) (by omega) (by omega) (by omega), t7, t8,
      Fca.tZ3 _ _ _ _ (eP d m 0) h1, Fca.tZ3 _ _ _ _ (eP d m 0) h3,
      tZ4 _ _ _ _ _ (eP d m 0) h0]
    unfold aeqC
    ring
  have hpair : ∀ i ∈ range n, (((pairN n i).map (val (envAt (E0 d) L m) (tauC n ρ))).prod) =
      ((if ρ (sl n (20 + 10 * i)) = ρ (pend n i) then (1 : ℂ) else 0) *
        (if ρ (sl n (21 + 10 * i)) = ρ (pendO n i) then 1 else 0) *
        (if ρ (sl n (22 + 10 * i)) = ρ (pendA n i) then 1 else 0)) *
      (tP (pD d n i) 1 [ρ (pend n i), ρ (sl n 5)] * tP (pD d n i) 1 [ρ (pendA n i), ρ (sl n 5)] *
        tP (pD d n i) (-1) [ρ (pendO n i), ρ (sl n 5)]) := by
    intro i hi'
    have hi := mem_range.1 hi'
    have mem : ∀ {k : ℕ × ℕ}, k ∈ [(sl n (23 + 10 * i), ρ (pend n i)),
        (sl n (24 + 10 * i), ρ (sl n 5)), (sl n (25 + 10 * i), ρ (pendA n i)),
        (sl n (26 + 10 * i), ρ (sl n 5)), (sl n (27 + 10 * i), ρ (pendO n i)),
        (sl n (28 + 10 * i), ρ (sl n 5))] → k ∈ valsC n ρ :=
      fun hk => mem_append_right _ (mem_flatMap.2 ⟨i, hi', hk⟩)
    have t23 : tauC n ρ (sl n (23 + 10 * i)) = ρ (pend n i) := tauC_of_mem (mem (by simp))
    have t24 : tauC n ρ (sl n (24 + 10 * i)) = ρ (sl n 5) := tauC_of_mem (mem (by simp))
    have t25 : tauC n ρ (sl n (25 + 10 * i)) = ρ (pendA n i) := tauC_of_mem (mem (by simp))
    have t26 : tauC n ρ (sl n (26 + 10 * i)) = ρ (sl n 5) := tauC_of_mem (mem (by simp))
    have t27 : tauC n ρ (sl n (27 + 10 * i)) = ρ (pendO n i) := tauC_of_mem (mem (by simp))
    have t28 : tauC n ρ (sl n (28 + 10 * i)) = ρ (sl n 5) := tauC_of_mem (mem (by simp))
    have ob : ∀ r < 3, tauC n ρ (sl n (20 + 10 * i + r)) = ρ (sl n (20 + 10 * i + r)) :=
      fun r hr => tauC_off (notin_pinC (by omega) (by omega) (by omega) (by omega) (by omega)
        fun j _ s hs => by omega)
    have o20 := ob 0 (by omega)
    have o21 := ob 1 (by omega)
    have o22 := ob 2 (by omega)
    simp only [add_zero, show 20 + 10 * i + 1 = 21 + 10 * i by omega,
      show 20 + 10 * i + 2 = 22 + 10 * i by omega] at o20 o21 o22
    simp only [pairN, pN, map_cons, map_nil, prod_cons, prod_nil, mul_one, val, zN, cons_append,
      nil_append, eD, Fca.dimv]
    rw [t23, t24, t25, t26, t27, t28, o20, o21, o22, tauC_off (pend_notin hi),
      tauC_off (pendO_notin hi), tauC_off (pendA_notin hi),
      Fca.tZ3 _ _ _ _ (eP d m 0) (hA i hi), Fca.tZ3 _ _ _ _ (eP d m 0) (hB i hi),
      Fca.tZ3 _ _ _ _ (eP d m 0) (hC i hi)]
    simp only [pD]
    ring
  simp only [cp, prodVal, map_append, prod_append, map_flatMap, prod_flatMapC]
  rw [hfix, List.map_congr_left hpair, List.prod_map_mul]
  simp only [pinsCp, pairsF]
  ring

theorem notin_freeC {n t : ℕ} (h5 : t ≠ 5) (h9 : t ≠ 9) (h10 : t ≠ 10) : sl n t ∉ freeC n := by
  simp only [freeC, mem_cons, not_mem_nil, or_false, not_or]
  exact ⟨fun e => h5 (sl_injective n e), fun e => h9 (sl_injective n e),
    fun e => h10 (sl_injective n e)⟩

theorem pend_notinF {n i : ℕ} (hi : i < n) : pend n i ∉ freeC n := by
  simp only [freeC, mem_cons, not_mem_nil, or_false, not_or]
  exact ⟨pend_ne hi (by omega), pend_ne hi (by omega), pend_ne hi (by omega)⟩

theorem pendO_notinF {n i : ℕ} (hi : i < n) : pendO n i ∉ freeC n := by
  simp only [freeC, mem_cons, not_mem_nil, or_false, not_or]
  exact ⟨pendO_ne hi (by omega), pendO_ne hi (by omega), pendO_ne hi (by omega)⟩

theorem pendA_notinF {n i : ℕ} (hi : i < n) : pendA n i ∉ freeC n := by
  simp only [freeC, mem_cons, not_mem_nil, or_false, not_or]
  exact ⟨pendA_ne hi (by omega), pendA_ne hi (by omega), pendA_ne hi (by omega)⟩

theorem pinsCp_congr {n : ℕ} {ρ σ : ℕ → ℕ} (h : ∀ x ∉ freeC n, ρ x = σ x) :
    pinsCp n ρ = pinsCp n σ := by
  have f : ∀ t, t ≠ 5 → t ≠ 9 → t ≠ 10 → ρ (sl n t) = σ (sl n t) := fun t h5 h9 h10 =>
    h _ (notin_freeC h5 h9 h10)
  simp only [pinsCp]
  rw [f 0 (by omega) (by omega) (by omega), f 1 (by omega) (by omega) (by omega),
    f 3 (by omega) (by omega) (by omega), f (20 + 10 * n) (by omega) (by omega) (by omega),
    f (21 + 10 * n) (by omega) (by omega) (by omega),
    f (22 + 10 * n) (by omega) (by omega) (by omega)]
  congr 1
  refine congrArg _ (List.map_congr_left fun i hi => ?_)
  have hi := mem_range.1 hi
  rw [f (20 + 10 * i) (by omega) (by omega) (by omega),
    f (21 + 10 * i) (by omega) (by omega) (by omega),
    f (22 + 10 * i) (by omega) (by omega) (by omega), h _ (pend_notinF hi),
    h _ (pendO_notinF hi), h _ (pendA_notinF hi)]

theorem aeqC_congr {n : ℕ} {ρ σ : ℕ → ℕ} (h : ∀ x ∉ freeC n, ρ x = σ x) :
    aeqC n ρ = aeqC n σ := by
  simp only [aeqC]
  rw [h _ (notin_freeC (t := 2) (by omega) (by omega) (by omega)),
    h _ (notin_freeC (t := 0) (by omega) (by omega) (by omega))]

theorem pairsF_congr {d n x : ℕ} {ρ σ : ℕ → ℕ} (h : ∀ x ∉ freeC n, ρ x = σ x) :
    pairsF d n ρ x = pairsF d n σ x := by
  simp only [pairsF]
  refine congrArg _ (List.map_congr_left fun i hi => ?_)
  have hi := mem_range.1 hi
  rw [h _ (pend_notinF hi), h _ (pendO_notinF hi), h _ (pendA_notinF hi)]

/-- The sum over copy `n`'s Fourier value. -/
noncomputable def SumC (d n : ℕ) (σ : ℕ → ℕ) : ℂ :=
  ∑ x ∈ Finset.range d, fou d x (σ (sl n 1)) * fou d x (σ (sl n 0)) *
    (starRingEnd ℂ) (fou d (σ (sl n 3)) x) * pairsF d n σ x

/-- Copy `n` in closed form: its pending-leg and copy factors times the sum over its Fourier
value. -/
theorem eval_cp (d : ℕ) (hd : 0 < d) (n : ℕ) (σ : ℕ → ℕ) (h0 : σ (sl n 0) < d)
    (h1 : σ (sl n 1) < d) (h3 : σ (sl n 3) < d) (hA : ∀ i < n, σ (pend n i) < d)
    (hB : ∀ i < n, σ (pendO n i) < d) (hC : ∀ i < n, σ (pendA n i) < d) :
    sumOver (labDim (envAt (E0 d) L (n + 1))) (cp n).int
      (prodVal (envAt (E0 d) L (n + 1)) (cp n).nodes) σ =
      pinsCp n σ * (aeqC n σ * SumC d n σ) := by
  have hn : n < n + 1 := Nat.lt_succ_self n
  have lp : ∀ t, t < 12 → labDim (envAt (E0 d) L (n + 1)) (sl n t) = d := fun t ht =>
    labDim_plain d hn (by omega) (Or.inl ht)
  have lpi : ∀ i < n, ∀ r < 6, labDim (envAt (E0 d) L (n + 1)) (sl n (23 + 10 * i + r)) = d :=
    fun i hi r hr => labDim_plain d hn (by omega) (Or.inr (by omega))
  show sumOver _ (freeC n ++ pinC n) _ σ = _
  rw [sumOver_append]
  rw [sumOver_congr _ (freeC n) _ (fun ρ => pinsCp n σ * (aeqC n σ *
      (fou d (ρ (sl n 5)) (σ (sl n 1)) * fou d (ρ (sl n 5)) (σ (sl n 0)) *
        (fou d (ρ (sl n 9)) (ρ (sl n 5)) * fou d (ρ (sl n 10)) (ρ (sl n 9)) *
          fou d (σ (sl n 3)) (ρ (sl n 10))) * pairsF d n σ (ρ (sl n 5))))) σ
      (fun ρ hρ hr => ?_)]
  · have n59 : sl n 5 ≠ sl n 9 := by simp [sl]
    have n510 : sl n 5 ≠ sl n 10 := by simp [sl]
    have n910 : sl n 9 ≠ sl n 10 := by simp [sl]
    simp only [freeC, sumOver]
    rw [lp 5 (by omega), lp 9 (by omega), lp 10 (by omega)]
    simp only [Function.update_apply, n59, n510, n910, if_false, if_true, Ne.symm n59,
      Ne.symm n510, Ne.symm n910]
    rw [SumC, Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun x _ => ?_
    have key : ∑ p ∈ Finset.range d, ∑ q ∈ Finset.range d,
        fou d p x * fou d q p * fou d (σ (sl n 3)) q = (starRingEnd ℂ) (fou d (σ (sl n 3)) x) := by
      rw [Finset.sum_comm, ← fou_three d (σ (sl n 3)) x hd]
      refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun p _ => ?_
      ring
    calc ∑ p ∈ Finset.range d, ∑ q ∈ Finset.range d, pinsCp n σ * (aeqC n σ *
          (fou d x (σ (sl n 1)) * fou d x (σ (sl n 0)) *
            (fou d p x * fou d q p * fou d (σ (sl n 3)) q) * pairsF d n σ x))
        = ∑ p ∈ Finset.range d, ∑ q ∈ Finset.range d, (pinsCp n σ * aeqC n σ *
          (fou d x (σ (sl n 1)) * fou d x (σ (sl n 0))) * pairsF d n σ x) *
            (fou d p x * fou d q p * fou d (σ (sl n 3)) q) :=
          Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => by ring
      _ = _ := by
          simp only [← Finset.mul_sum]
          rw [key]
          ring
  · have r0 : ρ (sl n 0) = σ (sl n 0) := hρ _ (notin_freeC (by omega) (by omega) (by omega))
    have r1 : ρ (sl n 1) = σ (sl n 1) := hρ _ (notin_freeC (by omega) (by omega) (by omega))
    have r3 : ρ (sl n 3) = σ (sl n 3) := hρ _ (notin_freeC (by omega) (by omega) (by omega))
    have r5 : ρ (sl n 5) < d := by
      have := hr (sl n 5) (by simp [freeC]); rwa [lp 5 (by omega)] at this
    have rA : ∀ i < n, ρ (pend n i) < d := fun i hi => by rw [hρ _ (pend_notinF hi)]; exact hA i hi
    have rB : ∀ i < n, ρ (pendO n i) < d := fun i hi => by
      rw [hρ _ (pendO_notinF hi)]; exact hB i hi
    have rC : ∀ i < n, ρ (pendA n i) < d := fun i hi => by
      rw [hρ _ (pendA_notinF hi)]; exact hC i hi
    rw [Fca.sumOver_single _ (pinC n) (pinC_nodup n) _ ρ (tauC n ρ) (fun x hx => tauC_off hx)
      ?hr ?hs, prod_cp d n (n + 1) ρ (r0 ▸ h0) (r1 ▸ h1) (r3 ▸ h3) r5 rA rB rC,
      pinsCp_congr hρ, aeqC_congr hρ, pairsF_congr hρ, r0, r1, r3]
    case hr =>
      intro l hl
      have mem : ∀ {i : ℕ} {k : ℕ × ℕ}, i < n → k ∈ [(sl n (23 + 10 * i), ρ (pend n i)),
          (sl n (24 + 10 * i), ρ (sl n 5)), (sl n (25 + 10 * i), ρ (pendA n i)),
          (sl n (26 + 10 * i), ρ (sl n 5)), (sl n (27 + 10 * i), ρ (pendO n i)),
          (sl n (28 + 10 * i), ρ (sl n 5))] → k ∈ valsC n ρ := fun hi hk =>
        mem_append_right _ (mem_flatMap.2 ⟨_, mem_range.2 hi, hk⟩)
      simp only [pinC, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range] at hl
      rcases hl with (rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, rfl | rfl | rfl | rfl | rfl | rfl⟩
      · rw [tauC_of_mem (y := ρ (sl n 1)) (by simp [valsC, sl]), lp 4 (by omega), r1]; exact h1
      · rw [tauC_of_mem (y := ρ (sl n 0)) (by simp [valsC, sl]), lp 6 (by omega), r0]; exact h0
      · rw [tauC_of_mem (y := ρ (sl n 5)) (by simp [valsC, sl]), lp 7 (by omega)]; exact r5
      · rw [tauC_of_mem (y := ρ (sl n 5)) (by simp [valsC, sl]), lp 8 (by omega)]; exact r5
      · rw [tauC_of_mem (y := ρ (sl n 3)) (by simp [valsC, sl]), lp 11 (by omega), r3]; exact h3
      · rw [tauC_of_mem (y := ρ (pend n i)) (mem hi (by simp)),
          show 23 + 10 * i = 23 + 10 * i + 0 from rfl, lpi i hi 0 (by omega)]; exact rA i hi
      · rw [tauC_of_mem (y := ρ (sl n 5)) (mem hi (by simp)),
          show 24 + 10 * i = 23 + 10 * i + 1 by omega, lpi i hi 1 (by omega)]; exact r5
      · rw [tauC_of_mem (y := ρ (pendA n i)) (mem hi (by simp)),
          show 25 + 10 * i = 23 + 10 * i + 2 by omega, lpi i hi 2 (by omega)]; exact rC i hi
      · rw [tauC_of_mem (y := ρ (sl n 5)) (mem hi (by simp)),
          show 26 + 10 * i = 23 + 10 * i + 3 by omega, lpi i hi 3 (by omega)]; exact r5
      · rw [tauC_of_mem (y := ρ (pendO n i)) (mem hi (by simp)),
          show 27 + 10 * i = 23 + 10 * i + 4 by omega, lpi i hi 4 (by omega)]; exact rB i hi
      · rw [tauC_of_mem (y := ρ (sl n 5)) (mem hi (by simp)),
          show 28 + 10 * i = 23 + 10 * i + 5 by omega, lpi i hi 5 (by omega)]; exact r5
    case hs =>
      intro τ h0' _ hne l hl
      rw [cp_support hne l hl, tauC_congr h0' l]

/-! ## Copy `n` glued to the spec -/

theorem cp_off_rhs (n : ℕ) : ∀ x ∈ labelsOf (cp n).nodes, x ∉ (rhsN n).int := by
  intro x hx hi
  rcases cp_own n x (mem_all_of_nodes hx) with ⟨t, _, rfl⟩ | hc
  · have := (fitsR n).1 _ (mem_all_of_int hi); unfold sl at this; omega
  · exact (fitsR n).2 n le_rfl x hi hc

theorem rhs_off_cp (n : ℕ) : ∀ x ∈ labelsOf (rhsN n).nodes, x ∉ (cp n).int := by
  intro x hx hi
  obtain ⟨t, _, rfl⟩ := cp_int hi
  have := (fitsR n).1 _ (mem_all_of_nodes hx); unfold sl at this; omega

theorem cut_nodup (n : ℕ) : (cut n).Nodup := by
  have : cut n = (([] : List ℕ) ++ (range n).flatMap fun i =>
      [0, 1, 2].map fun r => 20 + 10 * i + r).map (sl (n - 1)) := by
    simp only [cut, pend, pendO, pendA, nil_append, map_flatMap, map_cons, map_nil, sl]
    ring_nf
  rw [this]
  exact (nodup_blocks n 20 _ _ (by decide) (by decide) (by decide) (by simp)).map
    (sl_injective _)

/-- The pending legs before copy `n` at the digits they read in the spec. -/
def pinsG (n : ℕ) (σ : ℕ → ℕ) : List (ℕ × ℕ) :=
  (range n).flatMap fun i => [(pend n i, σ (sl i 1)), (pendO n i, σ (sl i 3)), (pendA n i, σ (sl i 0))]

/-- `σ` with the pending legs before copy `n` at their digits. -/
def tauG (n : ℕ) (σ : ℕ → ℕ) (x : ℕ) : ℕ := ((pinsG n σ).lookup x).getD (σ x)

theorem pinsG_fst (n : ℕ) (σ : ℕ → ℕ) : (pinsG n σ).map Prod.fst = cut n := by
  simp [pinsG, cut, map_flatMap]

theorem tauG_off {n x : ℕ} {σ : ℕ → ℕ} (h : x ∉ cut n) : tauG n σ x = σ x := by
  rw [tauG, Fca.lookup_none _ _ (by rw [pinsG_fst]; exact h)]; rfl

theorem tauG_mem {n x y : ℕ} {σ : ℕ → ℕ} (h : (x, y) ∈ pinsG n σ) : tauG n σ x = y := by
  rw [tauG, Fca.lookup_of_mem (pinsG n σ) (by rw [pinsG_fst]; exact cut_nodup n) h]; rfl

theorem tauG_pend {n i : ℕ} {σ : ℕ → ℕ} (hi : i < n) : tauG n σ (pend n i) = σ (sl i 1) :=
  tauG_mem (mem_flatMap.2 ⟨i, mem_range.2 hi, by simp⟩)

theorem tauG_pendO {n i : ℕ} {σ : ℕ → ℕ} (hi : i < n) : tauG n σ (pendO n i) = σ (sl i 3) :=
  tauG_mem (mem_flatMap.2 ⟨i, mem_range.2 hi, by simp⟩)

theorem tauG_pendA {n i : ℕ} {σ : ℕ → ℕ} (hi : i < n) : tauG n σ (pendA n i) = σ (sl i 0) :=
  tauG_mem (mem_flatMap.2 ⟨i, mem_range.2 hi, by simp⟩)

theorem dig_notin_cut {n i s : ℕ} (hs : s < 4) : sl i s ∉ cut n := by
  intro h
  obtain ⟨j, hj, e | e | e⟩ := mem_cut.1 h <;>
  · have := (sl_inj (by omega) (by omega) e).2; omega

theorem labDim_pend (d : ℕ) {n i r : ℕ} (hi : i < n) (hr : r < 3) :
    labDim (envAt (E0 d) L (n + 1)) (sl (n - 1) (20 + 10 * i + r)) = d :=
  labDim_plain d (by omega) (by omega) (Or.inr (by omega))

/-- Copy `n` glued to the spec at `n` digits: the spec's factors, copy `n`'s pending and copy
factors and Fourier sum with the old pending legs at the spec's digits. -/
theorem eval_glue (d : ℕ) (hd : 0 < d) (n : ℕ) (σ : ℕ → ℕ)
    (hs : ∀ s < 4, ∀ i < n + 1, σ (sl i s) < d) :
    eval (envAt (E0 d) L (n + 1)) (glueAt (cut n) (cp n) (rhsN n)) σ =
      (rt (d ^ n))⁻¹ * IndS d σ n *
        (pinsCp n (tauG n σ) * (aeqC n (tauG n σ) * SumC d n (tauG n σ))) := by
  have hperm : ((cp n).int ++ cut n ++ (rhsN n).int).Perm
      (cut n ++ ((cp n).int ++ (rhsN n).int)) := by
    simpa [List.append_assoc] using
      (List.perm_append_comm (l₁ := (cp n).int) (l₂ := cut n)).append_right (rhsN n).int
  have lpd : ∀ i < n, labDim (envAt (E0 d) L (n + 1)) (pend n i) = d := fun i hi => by
    have := labDim_pend d (r := 0) hi (by omega); simpa [pend] using this
  have lpo : ∀ i < n, labDim (envAt (E0 d) L (n + 1)) (pendO n i) = d := fun i hi => by
    have := labDim_pend d (r := 1) hi (by omega)
    simpa [pendO, show 20 + 10 * i + 1 = 21 + 10 * i by omega] using this
  have lpa : ∀ i < n, labDim (envAt (E0 d) L (n + 1)) (pendA n i) = d := fun i hi => by
    have := labDim_pend d (r := 2) hi (by omega)
    simpa [pendA, show 20 + 10 * i + 2 = 22 + 10 * i by omega] using this
  unfold eval
  simp only [glueAt]
  rw [sumOver_perm _ hperm, sumOver_append]
  rw [sumOver_congr _ (cut n) _ (fun τ => (rt (d ^ n))⁻¹ * IndS d σ n * pinsN n τ *
      (pinsCp n τ * (aeqC n τ * SumC d n τ))) σ (fun τ hτ hr => ?_)]
  · rw [Fca.sumOver_single _ (cut n) (cut_nodup n) _ σ (tauG n σ) (fun x hx => tauG_off hx)
      ?hr ?hs]
    case hr =>
      intro l hl
      obtain ⟨i, hi, rfl | rfl | rfl⟩ := mem_cut.1 hl
      · rw [tauG_pend hi, lpd i hi]; exact hs 1 (by omega) i (by omega)
      · rw [tauG_pendO hi, lpo i hi]; exact hs 3 (by omega) i (by omega)
      · rw [tauG_pendA hi, lpa i hi]; exact hs 0 (by omega) i (by omega)
    case hs =>
      intro τ h0 _ hne l hl
      have hp : pinsN n τ ≠ 0 := fun h => hne (by rw [h]; ring)
      have hall : ∀ x ∈ ((range n).map fun i =>
          (if τ (pend n i) = τ (sl i 1) then (1 : ℂ) else 0) *
            (if τ (pendO n i) = τ (sl i 3) then 1 else 0) *
            (if τ (pendA n i) = τ (sl i 0) then 1 else 0)), x ≠ 0 :=
        fun x hx h0 => hp (by unfold pinsN; exact List.prod_eq_zero (h0 ▸ hx))
      obtain ⟨i, hi, rfl | rfl | rfl⟩ := mem_cut.1 hl
      · have := hall _ (mem_map.2 ⟨i, mem_range.2 hi, rfl⟩)
        rw [tauG_pend hi, ← h0 _ (dig_notin_cut (by omega))]
        by_contra hc; exact this (by rw [if_neg hc]; ring)
      · have := hall _ (mem_map.2 ⟨i, mem_range.2 hi, rfl⟩)
        rw [tauG_pendO hi, ← h0 _ (dig_notin_cut (by omega))]
        by_contra hc; exact this (by rw [if_neg hc]; ring)
      · have := hall _ (mem_map.2 ⟨i, mem_range.2 hi, rfl⟩)
        rw [tauG_pendA hi, ← h0 _ (dig_notin_cut (by omega))]
        by_contra hc; exact this (by rw [if_neg hc]; ring)
    have h1 : pinsN n (tauG n σ) = 1 := by
      refine List.prod_eq_one fun x hx => ?_
      obtain ⟨i, hi, rfl⟩ := mem_map.1 hx
      have hi := mem_range.1 hi
      rw [tauG_pend hi, tauG_pendO hi, tauG_pendA hi, tauG_off (dig_notin_cut (by omega)),
        tauG_off (dig_notin_cut (by omega)), tauG_off (dig_notin_cut (by omega))]
      simp
    rw [h1, mul_one]
  · have hdg : ∀ s < 4, ∀ i, τ (sl i s) = σ (sl i s) := fun s hs i =>
      hτ _ (dig_notin_cut hs)
    rw [sumOver_append]
    have step : ∀ τ', sumOver (labDim (envAt (E0 d) L (n + 1))) (rhsN n).int
        (prodVal (envAt (E0 d) L (n + 1)) ((cp n).nodes ++ (rhsN n).nodes)) τ' =
        eval (envAt (E0 d) L (n + 1)) (rhsN n) τ' *
          prodVal (envAt (E0 d) L (n + 1)) (cp n).nodes τ' := fun τ' => by
      have e : prodVal (envAt (E0 d) L (n + 1)) ((cp n).nodes ++ (rhsN n).nodes) =
          fun ρ => prodVal (envAt (E0 d) L (n + 1)) (cp n).nodes ρ *
            prodVal (envAt (E0 d) L (n + 1)) (rhsN n).nodes ρ :=
        funext fun ρ => prodVal_append _ _ _ ρ
      rw [e, sumOver_const_mul _ _ _ _ _ (prodVal_reads _ _) (cp_off_rhs n) τ', mul_comm]; rfl
    rw [show sumOver (labDim (envAt (E0 d) L (n + 1))) (rhsN n).int
        (prodVal (envAt (E0 d) L (n + 1)) ((cp n).nodes ++ (rhsN n).nodes)) =
        fun τ' => eval (envAt (E0 d) L (n + 1)) (rhsN n) τ' *
          prodVal (envAt (E0 d) L (n + 1)) (cp n).nodes τ' from funext step,
      sumOver_const_mul _ _ _ _ _ (Fca.eval_reads _ _) (rhs_off_cp n) τ,
      eval_cp d hd n τ (by rw [hdg 0 (by omega)]; exact hs 0 (by omega) n (by omega))
        (by rw [hdg 1 (by omega)]; exact hs 1 (by omega) n (by omega))
        (by rw [hdg 3 (by omega)]; exact hs 3 (by omega) n (by omega))
        (fun i hi => by have := hr _ (pend_mem hi); rwa [lpd i hi] at this)
        (fun i hi => by have := hr _ (pendO_mem hi); rwa [lpo i hi] at this)
        (fun i hi => by have := hr _ (pendA_mem hi); rwa [lpa i hi] at this),
      eval_rhsN d hd (by omega) τ (fun s hs' i hi => by
        rw [hdg s hs']; exact hs s hs' i (by omega))]
    have hI : IndS d τ n = IndS d σ n := by
      simp only [IndS, topv_congr (fun i _ => hdg 0 (by omega) i) n le_rfl,
        topv_congr (fun i _ => hdg 1 (by omega) i) n le_rfl,
        topv_congr (fun i _ => hdg 2 (by omega) i) n le_rfl,
        topv_congr (fun i _ => hdg 3 (by omega) i) n le_rfl]
    rw [hI]

/-! ## The arithmetic of copy `n`'s Fourier sum -/

/-- The value of the first `m` digits at slot `s`, least significant first. -/
def dv (d : ℕ) (σ : ℕ → ℕ) (s m : ℕ) : ℕ := ∑ i ∈ Finset.range m, σ (sl i s) * d ^ i

theorem topv_sum (d : ℕ) (σ : ℕ → ℕ) (n s : ℕ) :
    ∀ t ≤ n, topv d σ n s t = ∑ j ∈ Finset.range t, σ (sl (n - t + j) s) * d ^ j
  | 0, _ => by simp [topv]
  | t + 1, ht => by
    rw [topv, topv_sum d σ n s t (by omega), Finset.sum_range_succ', Finset.sum_mul,
      show n - 1 - t = n - (t + 1) + 0 by omega, pow_zero, mul_one]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [show n - (t + 1) + (j + 1) = n - t + j by omega, pow_succ, mul_assoc]

theorem topv_dv (d : ℕ) (σ : ℕ → ℕ) (n s : ℕ) : topv d σ n s n = dv d σ s n := by
  rw [topv_sum d σ n s n le_rfl, dv]
  simp only [Nat.sub_self, zero_add]

theorem dv_succ (d : ℕ) (σ : ℕ → ℕ) (s n : ℕ) :
    dv d σ s (n + 1) = dv d σ s n + σ (sl n s) * d ^ n :=
  Finset.sum_range_succ _ _

theorem dv_lt {d : ℕ} {σ : ℕ → ℕ} {s : ℕ} :
    ∀ n, (∀ i < n, σ (sl i s) < d) → dv d σ s n < d ^ n
  | 0, _ => by simp [dv]
  | n + 1, h => by
    rw [dv_succ, pow_succ]
    have := dv_lt n fun i hi => h i (by omega)
    have := h n (by omega)
    nlinarith

theorem tP_lift (d : ℕ) (hd : 0 < d) {n i : ℕ} (hi : i ≤ n) (y x : ℕ) :
    tP (pD d n i) 1 [y, x] = om (d ^ (n + 1)) ^ (x * (y * d ^ i)) := by
  have hd' : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have hp : (d : ℝ) ^ (n + 1) = (d : ℝ) ^ (n - i + 1) * (d : ℝ) ^ i := by
    rw [← pow_add]; congr 1; omega
  unfold tP pD
  rw [om_eq_ex]
  push_cast
  rw [hp]
  field_simp

theorem tP_lift_neg (d : ℕ) (hd : 0 < d) {n i : ℕ} (hi : i ≤ n) (y x : ℕ) :
    tP (pD d n i) (-1) [y, x] = (om (d ^ (n + 1)))⁻¹ ^ (x * (y * d ^ i)) := by
  have hd' : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have hp : (d : ℝ) ^ (n + 1) = (d : ℝ) ^ (n - i + 1) * (d : ℝ) ^ i := by
    rw [← pow_add]; congr 1; omega
  unfold tP pD
  rw [inv_pow, om_eq_ex, ex_inv]
  push_cast
  rw [hp]
  field_simp

theorem pairsF_lift (d : ℕ) (hd : 0 < d) (n : ℕ) (σ : ℕ → ℕ) (x : ℕ) :
    pairsF d n (tauG n σ) x = om (d ^ (n + 1)) ^ (x * dv d σ 1 n) *
      om (d ^ (n + 1)) ^ (x * dv d σ 0 n) * (om (d ^ (n + 1)))⁻¹ ^ (x * dv d σ 3 n) := by
  have hf : ∀ i ∈ Finset.range n, tP (pD d n i) 1 [tauG n σ (pend n i), x] *
      tP (pD d n i) 1 [tauG n σ (pendA n i), x] * tP (pD d n i) (-1) [tauG n σ (pendO n i), x] =
      om (d ^ (n + 1)) ^ (x * (σ (sl i 1) * d ^ i)) * om (d ^ (n + 1)) ^ (x * (σ (sl i 0) * d ^ i)) *
        (om (d ^ (n + 1)))⁻¹ ^ (x * (σ (sl i 3) * d ^ i)) := by
    intro i hi
    have hi := Finset.mem_range.1 hi
    rw [tauG_pend hi, tauG_pendO hi, tauG_pendA hi, tP_lift d hd hi.le, tP_lift d hd hi.le,
      tP_lift_neg d hd hi.le]
  rw [pairsF, Fca.lprod_range, Finset.prod_congr rfl hf, Finset.prod_mul_distrib,
    Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum,
    Finset.prod_pow_eq_pow_sum, dv, dv, dv, Finset.mul_sum, Finset.mul_sum, Finset.mul_sum]

theorem sumC_lift (d : ℕ) (hd : 0 < d) (n : ℕ) (σ : ℕ → ℕ) :
    SumC d n (tauG n σ) = ((rt d)⁻¹ ^ 3) * ∑ x ∈ Finset.range d,
      (om (d ^ (n + 1)) ^ (dv d σ 1 (n + 1) + dv d σ 0 (n + 1)) *
        (om (d ^ (n + 1)))⁻¹ ^ dv d σ 3 (n + 1)) ^ x := by
  rw [SumC, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  have o : ∀ s < 4, tauG n σ (sl n s) = σ (sl n s) := fun s hs => tauG_off (dig_notin_cut hs)
  have hdn : d = d ^ (n - n + 1) := by simp
  have hc : (starRingEnd ℂ) (fou d (σ (sl n 3)) x) =
      (om (d ^ (n + 1)))⁻¹ ^ (x * (σ (sl n 3) * d ^ n)) / rt d := by
    unfold fou
    rw [map_div₀, map_pow, conj_om, Complex.conj_ofReal, Fca.om_lift d hd le_rfl hdn, ← inv_pow,
      ← pow_mul, show d ^ n * (σ (sl n 3) * x) = x * (σ (sl n 3) * d ^ n) by ring]
  rw [o 0 (by omega), o 1 (by omega), o 3 (by omega), hc, Fca.fou_lift d hd le_rfl hdn,
    Fca.fou_lift d hd le_rfl hdn, pairsF_lift d hd, dv_succ, dv_succ, dv_succ]
  ring

theorem rt_succ (d n : ℕ) : rt (d ^ (n + 1)) = rt (d ^ n) * rt d := by
  simp only [rt]
  rw [pow_succ, Nat.cast_mul, Real.sqrt_mul (Nat.cast_nonneg _), Complex.ofReal_mul]

theorem copy_iff {d : ℕ} {σ : ℕ → ℕ} {n : ℕ} (h0 : ∀ i < n, σ (sl i 0) < d)
    (h2 : ∀ i < n, σ (sl i 2) < d) : dv d σ 2 (n + 1) = dv d σ 0 (n + 1) ↔
      dv d σ 2 n = dv d σ 0 n ∧ σ (sl n 2) = σ (sl n 0) := by
  rw [dv_succ, dv_succ]
  have a2 := dv_lt n h2
  have a0 := dv_lt n h0
  constructor
  · intro e
    have hm := congrArg (· % d ^ n) e
    have hq := congrArg (· / d ^ n) e
    have hpos : 0 < d ^ n := by
      rcases Nat.eq_zero_or_pos (d ^ n) with h | h
      · omega
      · exact h
    rw [Nat.add_mul_mod_self_right, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt a2,
      Nat.mod_eq_of_lt a0] at hm
    rw [Nat.add_mul_div_right _ _ hpos, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt a2,
      Nat.div_eq_of_lt a0] at hq
    exact ⟨hm, by omega⟩
  · rintro ⟨e1, e2⟩; rw [e1, e2]

/-- The arithmetic of copy `n`'s Fourier sum against the spec's factors. -/
theorem arith (d : ℕ) (hd : 0 < d) (n : ℕ) (σ : ℕ → ℕ)
    (hs : ∀ s < 4, ∀ i < n + 1, σ (sl i s) < d) :
    IndS d σ n * (aeqC n σ * SumC d n (tauG n σ)) = (rt d)⁻¹ * IndS d σ (n + 1) := by
  rw [sumC_lift d hd]
  simp only [IndS, topv_dv, aeqC]
  have hcp := copy_iff (n := n) (fun i hi => hs 0 (by omega) i (by omega))
    (fun i hi => hs 2 (by omega) i (by omega))
  have hP : (dv d σ 1 (n + 1) + dv d σ 0 (n + 1)) % d ^ n = (dv d σ 1 n + dv d σ 0 n) % d ^ n := by
    rw [dv_succ, dv_succ, show dv d σ 1 n + σ (sl n 1) * d ^ n + (dv d σ 0 n + σ (sl n 0) * d ^ n)
      = dv d σ 1 n + dv d σ 0 n + (σ (sl n 1) + σ (sl n 0)) * d ^ n by ring,
      Nat.add_mul_mod_self_right]
  have hQ : dv d σ 3 (n + 1) % d ^ n = dv d σ 3 n % d ^ n := by
    rw [dv_succ, Nat.add_mul_mod_self_right]
  have hdv : d ^ n ∣ d ^ (n + 1) := pow_dvd_pow d (Nat.le_succ n)
  have hrt := rt_ne d hd
  have hsc : (rt d)⁻¹ ^ 3 * (d : ℂ) = (rt d)⁻¹ := by
    rw [← rt_sq d]; field_simp
  have hcpC : (if dv d σ 2 (n + 1) = dv d σ 0 (n + 1) then (1 : ℂ) else 0) =
      (if dv d σ 2 n = dv d σ 0 n then 1 else 0) * (if σ (sl n 2) = σ (sl n 0) then 1 else 0) := by
    by_cases e1 : dv d σ 2 n = dv d σ 0 n <;> by_cases e2 : σ (sl n 2) = σ (sl n 0) <;>
      simp [e1, e2, hcp]
  rw [hcpC]
  by_cases h : dv d σ 3 n % d ^ n = (dv d σ 1 n + dv d σ 0 n) % d ^ n
  · rw [if_pos h, Fca.geom_lift d hd n _ _ (by rw [hP, hQ, h])]
    by_cases h2 : (dv d σ 1 (n + 1) + dv d σ 0 (n + 1)) % d ^ (n + 1) =
        dv d σ 3 (n + 1) % d ^ (n + 1)
    · rw [if_pos h2, if_pos h2.symm]
      linear_combination (if dv d σ 2 n = dv d σ 0 n then (1 : ℂ) else 0) *
        (if σ (sl n 2) = σ (sl n 0) then (1 : ℂ) else 0) * hsc
    · have h4 : ¬dv d σ 3 (n + 1) % d ^ (n + 1) =
          (dv d σ 1 (n + 1) + dv d σ 0 (n + 1)) % d ^ (n + 1) := fun e => h2 e.symm
      rw [if_neg h2, if_neg h4]; simp
  · have h3 : ¬dv d σ 3 (n + 1) % d ^ (n + 1) =
        (dv d σ 1 (n + 1) + dv d σ 0 (n + 1)) % d ^ (n + 1) :=
      fun e => h (by rw [← hP, ← hQ, ← Nat.mod_mod_of_dvd _ hdv, e, Nat.mod_mod_of_dvd _ hdv])
    rw [if_neg h, if_neg h3]
    simp

/-! ## The family -/

/-- The spec with its scalar. -/
def rhs (n : ℕ) : ℂ × Net := (1, rhsN n)

theorem base (d : ℕ) (hd : 0 < d) :
    Equal (envAt (E0 d) L 0) (lhs 0).1 (lhs 0).2 (rhs 0).1 (rhs 0).2 := by
  refine ⟨by simp [lhs, rhs, lhsN, rhsN, bndL], fun σ _ => ?_⟩
  show 1 * eval _ (lhsN 0) σ = 1 * eval _ (rhsN 0) σ
  rw [eval_rhsN d hd le_rfl σ (fun _ _ i h => absurd h (Nat.not_lt_zero i))]
  simp [eval, lhsN, sumOver, prodVal, IndS, pinsN, topv, rt]

theorem peel (d n : ℕ) : Equal (envAt (E0 d) L (n + 1)) (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
    (glueAt (cut n) (cp n) (lhs n).2) := Equal.refl _ _ _

theorem pinsCp_tauG (n : ℕ) (σ : ℕ → ℕ) : pinsCp n (tauG n σ) = pinsN (n + 1) σ := by
  have o : ∀ t, t < 4 → tauG n σ (sl n t) = σ (sl n t) := fun t ht =>
    tauG_off (dig_notin_cut ht)
  have oP : ∀ t, 20 ≤ t → t < 10 * n + 24 → tauG n σ (sl n t) = σ (sl n t) := fun t h1 h2 =>
    tauG_off fun h => by
      obtain ⟨j, hj, e | e | e⟩ := mem_cut.1 h <;>
      · have := (sl_inj (by omega) (by omega) e).1; omega
  simp only [pinsCp, pinsN, range_succ, map_append, prod_append, map_cons, map_nil, prod_cons,
    prod_nil, mul_one]
  rw [o 0 (by omega), o 1 (by omega), o 3 (by omega), oP _ (by omega) (by omega),
    oP _ (by omega) (by omega), oP _ (by omega) (by omega)]
  have hl : ∀ i ∈ range n,
      (if tauG n σ (sl n (20 + 10 * i)) = tauG n σ (pend n i) then (1 : ℂ) else 0) *
        (if tauG n σ (sl n (21 + 10 * i)) = tauG n σ (pendO n i) then 1 else 0) *
        (if tauG n σ (sl n (22 + 10 * i)) = tauG n σ (pendA n i) then 1 else 0) =
      (if σ (pend (n + 1) i) = σ (sl i 1) then (1 : ℂ) else 0) *
        (if σ (pendO (n + 1) i) = σ (sl i 3) then 1 else 0) *
        (if σ (pendA (n + 1) i) = σ (sl i 0) then 1 else 0) := by
    intro i hi
    have hi := mem_range.1 hi
    rw [tauG_pend hi, tauG_pendO hi, tauG_pendA hi, oP _ (by omega) (by omega),
      oP _ (by omega) (by omega), oP _ (by omega) (by omega)]
    simp only [pend, pendO, pendA, Nat.add_sub_cancel]
  rw [List.map_congr_left hl]
  simp only [pend, pendO, pendA, Nat.add_sub_cancel]
  ring

theorem aeqC_tauG (n : ℕ) (σ : ℕ → ℕ) : aeqC n (tauG n σ) = aeqC n σ := by
  simp only [aeqC, tauG_off (dig_notin_cut (by omega : 2 < 4)),
    tauG_off (dig_notin_cut (by omega : 0 < 4))]

theorem dig_mem_bndL : ∀ m, ∀ i < m, ∀ s < 4, sl i s ∈ bndL m
  | 0, _, h, _, _ => absurd h (Nat.not_lt_zero _)
  | m + 1, i, hi, s, hs => by
    simp only [bndL, mem_append, mem_filter, decide_eq_true_eq]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
    · exact Or.inl ⟨dig_mem_bndL m i hi s hs, dig_notin_cut hs⟩
    · refine Or.inr ⟨?_, dig_notin_cut hs⟩
      simp only [cp, mem_append, mem_cons]
      interval_cases s <;> simp

/-- The close step: copy `n` glued to the spec at `n` digits is the spec at `n + 1`. -/
theorem close (d : ℕ) (hd : 0 < d) (n : ℕ) :
    Equal (envAt (E0 d) L (n + 1)) (rhs n).1 (glueAt (cut n) (cp n) (rhs n).2)
      (rhs (n + 1)).1 (rhs (n + 1)).2 := by
  refine ⟨List.Perm.refl _, fun σ hσ => ?_⟩
  have hs : ∀ s < 4, ∀ i < n + 1, σ (sl i s) < d := fun s hs i hi => by
    have := hσ _ (dig_mem_bndL (n + 1) i hi s hs)
    rwa [labDim_plain d hi (by omega) (Or.inl (by omega))] at this
  show 1 * eval _ (glueAt (cut n) (cp n) (rhsN n)) σ = 1 * eval _ (rhsN (n + 1)) σ
  rw [eval_glue d hd n σ hs, eval_rhsN d hd le_rfl σ hs, pinsCp_tauG, aeqC_tauG, rt_succ,
    mul_inv]
  have h := arith d hd n σ hs
  linear_combination (rt (d ^ n))⁻¹ * pinsN (n + 1) σ * h

/-- The boxed adder equals the spec at every `n`. -/
theorem fam (d : ℕ) (hd : 0 < d) (n : ℕ) :
    Equal (envAt (E0 d) L n) (lhs n).1 (lhs n).2 (rhs n).1 (rhs n).2 :=
  netIfaceAtE_fam (E0 d) L B cut cp lhs rhs (labsMono d) copyAt (fun n => fitsL n)
    (fun n => fitsR n) (base d hd) (peel d) (close d hd) n

theorem fam_8 (d : ℕ) (hd : 0 < d) :
    Equal (envAt (E0 d) L 8) (lhs 8).1 (lhs 8).2 (rhs 8).1 (rhs 8).2 := fam d hd 8

theorem fam_24 (d : ℕ) (hd : 0 < d) :
    Equal (envAt (E0 d) L 24) (lhs 24).1 (lhs 24).2 (rhs 24).1 (rhs 24).2 := fam d hd 24

end Draper

end ArchytasZX
