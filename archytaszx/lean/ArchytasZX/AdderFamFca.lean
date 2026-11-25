/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFamEnv
import ArchytasZX.Ball
import ArchytasZX.Weights
import ArchytasZX.PairEval

/-!
# The Fourier constant adder as a network family

The boxed `fourier_constant_adder` of the engine at `n` copies, copy `i` of weight `d ^ i`, as
networks over `envAt (E0 d) L n`. Copy `n` (the new most significant digit) owns the label block
`[B n, B (n + 1))`: input `0`, output `1`, the wire chain `2 … 9`, the spec's binder and splitter
values `12`, `13`, and per earlier copy `i` the slots `14 + 8 i … 21 + 8 i`. The pair phase
`x y / d ^ (n - i + 1)` of the pair `(n, i)` is an `F` node of dimension `d ^ (n - i + 1)`, its
negative that `F` through the antipode cup (an X spider with two outputs). Copy `i`'s input and output spiders keep one pending leg (`pend`) for the pairs of
later copies; `cut n` are the pending legs before copy `n`.

* `B`, `sl`, `dimv`, `phase`, `E0`, `L`, `cut`, `cp`, `lhs`, `rhs`: the family.
* `labsMono`, `copyAt`, `fitsL`, `fitsR`: the side conditions of `netIfaceAtE`.
-/

namespace ArchytasZX

namespace Fca

open List

/-- Start of the label block of level `n`. -/
def B (n : ℕ) : ℕ := 4 * n * n + 18 * n + 2

theorem B_succ (n : ℕ) : B (n + 1) = B n + (8 * n + 22) := by unfold B; ring

theorem B_le {a b : ℕ} (h : a ≤ b) : B a ≤ B b := by
  unfold B; nlinarith [Nat.mul_le_mul h h]

theorem B_lt {a b : ℕ} (h : a < b) : B a + (8 * a + 22) ≤ B b := by
  rw [← B_succ]; exact B_le h

/-- Slot `t` of level `n`. -/
def sl (n t : ℕ) : ℕ := B n + t

theorem two_le_sl (n t : ℕ) : 2 ≤ sl n t := by unfold sl B; omega

theorem sl_inj {a b x y : ℕ} (hx : x < 8 * a + 22) (hy : y < 8 * b + 22) (h : sl a x = sl b y) :
    a = b ∧ x = y := by
  unfold sl at h
  rcases lt_trichotomy a b with hab | rfl | hab
  · have := B_lt hab; omega
  · omega
  · have := B_lt hab; omega

theorem sl_lt {n i t : ℕ} (hi : i < n) (ht : t < 8 * i + 22) : sl i t < B n := by
  unfold sl; have := B_lt hi; omega

/-- Dimension values: index `0` is `d`, index `e + 1` is `d ^ e`. -/
def dimv (d : ℕ) : ℕ → ℕ
  | 0 => d
  | e + 1 => d ^ e

/-- Phases in turns: `0` none, `2 m + 1` is `k ↦ 2 k / d ^ (m + 1)`, `2 m + 2` is
`k ↦ -2 k / d ^ m`. -/
noncomputable def phase (d p k : ℕ) : ℝ :=
  if p = 0 then 0
  else if p % 2 = 1 then 2 * k / (d : ℝ) ^ (p / 2 + 1)
  else -(2 * k) / (d : ℝ) ^ (p / 2 - 1)

/-- The environment with no listed labels. -/
noncomputable def E0 (d : ℕ) : Env := { dimv := dimv d, phase := phase d, labs := [] }

/-- The labels of level `n` with a dimension other than `d`: the spec values `d ^ (n + 1)`, the
`F³` legs of the pair `(n, i)` at `d ^ (n - i + 1)`. -/
def blockL (n : ℕ) : List (ℕ × ℕ) :=
  [(sl n 12, n + 2), (sl n 13, n + 2)] ++
    (range n).flatMap fun i => [(sl n (20 + 8 * i), n - i + 2), (sl n (21 + 8 * i), n - i + 2)]

/-- The listed labels below `B n`. -/
def L (n : ℕ) : List (ℕ × ℕ) := [(0, 1), (1, 1)] ++ (range n).flatMap blockL

/-- The pending input leg of copy `i` before copy `n`. -/
def pend (n i : ℕ) : ℕ := sl (n - 1) (14 + 8 * i)

/-- The pending output leg of copy `i` before copy `n`. -/
def pendO (n i : ℕ) : ℕ := sl (n - 1) (15 + 8 * i)

/-- The pending legs before copy `n`. -/
def cut (n : ℕ) : List ℕ := (range n).flatMap fun i => [pend n i, pendO n i]

/-- A plain Z spider of dimension `d`. -/
def zN (os is : List ℕ) (ph : ℕ := 0) : Node := ⟨.Z, os, is, 0, ph⟩

/-- A Fourier box of dimension index `D`. -/
def fN (D o i : ℕ) : Node := ⟨.F, [o], [i], D, 0⟩

/-- The pair nodes of copy `n` with copy `i`: `F` from copy `i`'s input, `F` through the
antipode cup from its output, and the spiders moving its pending legs. -/
def pairN (n i : ℕ) : List Node :=
  [fN (n - i + 2) (sl n (18 + 8 * i)) (sl n (16 + 8 * i)),
   zN [sl n (16 + 8 * i), sl n (14 + 8 * i)] [pend n i],
   fN (n - i + 2) (sl n (19 + 8 * i)) (sl n (20 + 8 * i)),
   ⟨.X, [sl n (20 + 8 * i), sl n (17 + 8 * i)], [], n - i + 2, 0⟩,
   zN [sl n (17 + 8 * i), sl n (15 + 8 * i)] [pendO n i]]

/-- The summed labels of copy `n` no node pins: the Fourier value and the `F³` legs. -/
def freeC (n : ℕ) : List ℕ := [sl n 3, sl n 7, sl n 8]

/-- The summed labels of copy `n` a node pins. -/
def pinC (n : ℕ) : List ℕ :=
  [sl n 2, sl n 4, sl n 5, sl n 6, sl n 9] ++ (range n).flatMap fun i =>
    [sl n (16 + 8 * i), sl n (17 + 8 * i), sl n (18 + 8 * i), sl n (19 + 8 * i),
      sl n (20 + 8 * i)]

/-- Copy `n` of the boxed side. -/
def cp (n : ℕ) : Net where
  nodes :=
    [zN [sl n 2, sl n (14 + 8 * n)] [sl n 0],
     fN 0 (sl n 3) (sl n 2),
     zN (sl n 4 :: (range n).map fun i => sl n (18 + 8 * i)) [sl n 3],
     zN [sl n 5] [sl n 4] (2 * n + 1),
     zN (sl n 6 :: (range n).map fun i => sl n (19 + 8 * i)) [sl n 5],
     fN 0 (sl n 7) (sl n 6),
     fN 0 (sl n 8) (sl n 7),
     fN 0 (sl n 9) (sl n 8),
     zN [sl n 1, sl n (15 + 8 * n)] [sl n 9]] ++ (range n).flatMap (pairN n)
  int := freeC n ++ pinC n
  bnd := [sl n 0, sl n 1, sl n (14 + 8 * n), sl n (15 + 8 * n)] ++
    (range n).flatMap fun i => [sl n (14 + 8 * i), sl n (15 + 8 * i)]

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

/-- The spec's binder value of the top `t` digits. -/
def cv : ℕ → ℕ
  | 0 => 0
  | t + 1 => sl t 12

/-- The spec's splitter value of the top `t` digits. -/
def cw : ℕ → ℕ
  | 0 => 1
  | t + 1 => sl t 13

/-- The spec at `n` digits: the binder chain, the shift by `2` mod `d ^ n`, the splitter chain,
and the pending legs as spiders on the inputs and outputs. -/
def rhsN (n : ℕ) : Net where
  nodes :=
    [⟨.Z, [cv 0], [], 1, 0⟩] ++
    (range n).map (fun t => ⟨.B, [cv (t + 1)], [cv t, sl (n - 1 - t) 0], t + 1, 0⟩) ++
    [⟨.X, [cw n], [cv n], n + 1, 2 * n + 2⟩] ++
    (range n).map (fun t => ⟨.S, [cw t, sl (n - 1 - t) 1], [cw (t + 1)], t + 1, 0⟩) ++
    [⟨.Z, [], [cw 0], 1, 0⟩] ++
    (range n).flatMap fun i => [zN [pend n i] [sl i 0], zN [pendO n i] [sl i 1]]
  int := [cv 0, cw 0] ++ (range n).flatMap fun t => [cv (t + 1), cw (t + 1)]
  bnd := bndL n

/-! ## Side conditions -/

theorem lookup_append_of_not (x : ℕ) (a b : List (ℕ × ℕ)) (h : ∀ p ∈ b, p.1 ≠ x) :
    (a ++ b).lookup x = a.lookup x := by
  induction a with
  | nil =>
    simp only [nil_append, List.lookup]
    induction b with
    | nil => rfl
    | cons p b ih =>
      obtain ⟨k, v⟩ := p
      have hk : (x == k) = false := by simpa using Ne.symm (h _ mem_cons_self)
      simp only [List.lookup, hk]
      exact ih fun q hq => h q (mem_cons_of_mem _ hq)
  | cons p a ih =>
    obtain ⟨k, v⟩ := p
    simp only [cons_append, List.lookup]
    split <;> simp_all

theorem blockL_ge (n : ℕ) : ∀ p ∈ blockL n, B n ≤ p.1 := by
  intro p hp
  simp only [blockL, mem_append, mem_cons, mem_flatMap, mem_range, not_mem_nil, or_false] at hp
  rcases hp with (rfl | rfl) | ⟨i, _, rfl | rfl⟩ <;> simp [sl]

theorem labsMono (d : ℕ) : LabsMono (E0 d) L B := by
  intro n x hx
  have h : L (n + 1) = L n ++ blockL n := by
    simp [L, range_succ, flatMap_append]
  simp only [labDim, envAt, h]
  rw [lookup_append_of_not x _ _ fun p hp e => by have := blockL_ge n p hp; omega]

theorem mem_cut {n x : ℕ} : x ∈ cut n ↔ ∃ i < n, x = pend n i ∨ x = pendO n i := by
  simp only [cut, mem_flatMap, mem_range, mem_cons, not_mem_nil, or_false]

/-- Labels owned by level `n` or pending before it. -/
def Own (n x : ℕ) : Prop := (∃ t < 8 * n + 22, x = sl n t) ∨ x ∈ cut n

theorem cp_own (n : ℕ) : ∀ x ∈ labelsOf (cp n).nodes ++ (cp n).int ++ (cp n).bnd, Own n x := by
  simp only [cp, freeC, pinC, labelsOf, pairN, zN, fN, Node.labels, flatMap_append, mem_append, mem_flatMap,
    mem_cons, mem_range, mem_map, not_mem_nil, or_false, flatMap_cons, flatMap_nil,
    append_nil, cons_append, nil_append, or_imp, forall_and, forall_eq, forall_exists_index,
    and_imp, Own, mem_cut]
  and_intros
  all_goals (intros; subst_vars)
  all_goals (try simp only [mem_cons, not_mem_nil, or_false, mem_singleton] at *)
  all_goals (try casesm* _ ∨ _)
  all_goals (try subst_vars)
  all_goals first
    | exact Or.inl ⟨_, by omega, rfl⟩
    | exact Or.inr ⟨_, by assumption, Or.inl rfl⟩
    | exact Or.inr ⟨_, by assumption, Or.inr rfl⟩

theorem cut_lt {n x : ℕ} (h : x ∈ cut n) : x < B n := by
  obtain ⟨i, hi, h | h⟩ := mem_cut.1 h <;> subst h <;>
    exact sl_lt (by omega : n - 1 < n) (by omega)

theorem cp_int {n x : ℕ} (h : x ∈ (cp n).int) :
    ∃ t, ((2 ≤ t ∧ t < 10) ∨ ∃ i < n, ∃ r < 6, t = 16 + 8 * i + r) ∧ x = sl n t := by
  simp only [cp, freeC, pinC, mem_append, mem_cons, not_mem_nil, or_false, mem_range,
    mem_flatMap] at h
  rcases h with (rfl | rfl | rfl) |
    ((rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, rfl | rfl | rfl | rfl | rfl⟩)
  all_goals first
    | exact ⟨_, Or.inl (by omega), rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 4 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 0 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 1 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 2 < 6), by omega⟩, rfl⟩
    | exact ⟨_, Or.inr ⟨i, hi, _, (by omega : 3 < 6), by omega⟩, rfl⟩

theorem copyAt : CopyAt B cut cp where
  mono n := by rw [B_succ]; omega
  cutLt _ _ h := cut_lt h
  own n x hx := by
    rcases cp_own n x hx with ⟨t, ht, rfl⟩ | h
    · right; rw [B_succ]; unfold sl; omega
    · exact Or.inl h
  intOff n m hnm x hx hc := by
    obtain ⟨t, ht, rfl⟩ := cp_int hx
    obtain ⟨j, hj, h | h⟩ := mem_cut.1 hc <;>
    · obtain ⟨h1, h2⟩ := sl_inj (by omega) (by omega) h
      omega
  cutOff n m hnm x hx hc := by
    obtain ⟨i, hi, h | h⟩ := mem_cut.1 hx <;> obtain ⟨j, hj, h' | h'⟩ := mem_cut.1 hc <;>
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

theorem two_le_B (n : ℕ) : 2 ≤ B n := by unfold B; omega

theorem cv_lt {n t : ℕ} (h : t ≤ n) : cv t < B n := by
  cases t with
  | zero => have := two_le_B n; simp only [cv]; omega
  | succ t => exact sl_lt (by omega) (by omega)

theorem cw_lt {n t : ℕ} (h : t ≤ n) : cw t < B n := by
  cases t with
  | zero => have := two_le_B n; simp only [cw]; omega
  | succ t => exact sl_lt (by omega) (by omega)

theorem pend_mem {n i : ℕ} (h : i < n) : pend n i ∈ cut n := mem_cut.2 ⟨i, h, Or.inl rfl⟩

theorem pendO_mem {n i : ℕ} (h : i < n) : pendO n i ∈ cut n := mem_cut.2 ⟨i, h, Or.inr rfl⟩

theorem cv_off (t a s : ℕ) (hs : 14 ≤ s) (hs' : s < 8 * a + 22) : cv t ≠ sl a s := by
  cases t with
  | zero => have := two_le_sl a s; simp only [cv]; omega
  | succ t =>
    intro h
    have := (sl_inj (by omega) hs' h).2
    omega

theorem cw_off (t a s : ℕ) (hs : 14 ≤ s) (hs' : s < 8 * a + 22) : cw t ≠ sl a s := by
  cases t with
  | zero => have := two_le_sl a s; simp only [cw]; omega
  | succ t =>
    intro h
    have := (sl_inj (by omega) hs' h).2
    omega

theorem fitsR (n : ℕ) : FitsAt B cut n (rhsN n) := by
  refine ⟨?_, fun m hm x hx hc => ?_⟩
  · simp only [rhsN, labelsOf, zN, Node.labels, flatMap_append, mem_append, mem_flatMap,
      mem_cons, mem_range, mem_map, not_mem_nil, or_false, flatMap_cons, cons_append,
      nil_append, or_imp, forall_and, forall_eq, forall_exists_index, and_imp]
    and_intros
    all_goals (intros; subst_vars)
    all_goals (try simp only [mem_cons, not_mem_nil, or_false] at *)
    all_goals (try casesm* _ ∨ _)
    all_goals (try subst_vars)
    all_goals first
      | exact cv_lt (by omega)
      | exact cw_lt (by omega)
      | exact cut_lt (pend_mem (by omega))
      | exact cut_lt (pendO_mem (by omega))
      | exact sl_lt (by omega) (by omega)
      | exact bndL_lt (by assumption)
  · obtain ⟨j, hj, h | h⟩ := mem_cut.1 hc <;> subst h <;>
    · simp only [rhsN, mem_append, mem_flatMap, mem_cons, mem_range, not_mem_nil,
        or_false] at hx
      rcases hx with (h | h) | ⟨t, _, h | h⟩
      all_goals first
        | exact cv_off _ (m - 1) (14 + 8 * j) (by omega) (by omega) (by simpa [pend] using h.symm)
        | exact cv_off _ (m - 1) (15 + 8 * j) (by omega) (by omega) (by simpa [pendO] using h.symm)
        | exact cw_off _ (m - 1) (14 + 8 * j) (by omega) (by omega) (by simpa [pend] using h.symm)
        | exact cw_off _ (m - 1) (15 + 8 * j) (by omega) (by omega) (by simpa [pendO] using h.symm)

/-! ## Generator entries -/

/-- The X spider of dimension `D` with phase `k ↦ -2 k / D` is the shift by `2` mod `D`. -/
theorem tX_shift (D : ℕ) (hD : 0 < D) (o i : ℕ) :
    tX D (fun k => -(2 * (k : ℝ)) / (D : ℝ)) [o] [i] = if o % D = (i + 2) % D then 1 else 0 := by
  have hr : (starRingEnd ℂ) (rt D) = rt D := Complex.conj_ofReal _
  have hterm : ∀ k : ℕ, ex (-(2 * (k : ℝ)) / (D : ℝ)) * ([o].map fun o => fou D o k).prod *
      ([i].map fun i => (starRingEnd ℂ) (fou D i k)).prod =
      (rt D * rt D)⁻¹ * (om D ^ (k * o) * (starRingEnd ℂ) (om D) ^ (k * (i + 2))) := by
    intro k
    have he : ex (-(2 * (k : ℝ)) / (D : ℝ)) = (starRingEnd ℂ) (om D) ^ (k * 2) := by
      rw [conj_om, inv_pow, om_eq_ex, ex_inv]; congr 1; push_cast; ring
    simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, fou,
      map_div₀, map_pow, he]
    rw [hr, mul_add, pow_add, mul_comm o k, mul_comm i k]
    field_simp
  unfold tX
  simp only [hterm, ← Finset.mul_sum, char_sum D hD, rt_sq]
  split_ifs
  · exact nat_inv_mul D hD
  · simp

/-- A Z spider of dimension `D` with zero phase on two legs. -/
theorem tZ_two (D a b : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (hb : b < D) :
    tZ D φ [a, b] = if a = b then 1 else 0 := by
  unfold tZ
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, hφ, ex_zero]
  split_ifs with h
  · subst h
    rw [Finset.sum_eq_single a (fun k _ hk => if_neg fun h' => hk h'.1.symm)
      (fun h' => absurd (Finset.mem_range.2 hb) h')]
    simp
  · exact Finset.sum_eq_zero fun k _ => if_neg fun h' => h (h'.1.trans h'.2.symm)

/-- A Z spider of dimension `1` with zero phase on one leg. -/
theorem tZ_one (a : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) :
    tZ 1 φ [a] = if a = 0 then 1 else 0 := by
  unfold tZ
  simp [hφ, ex_zero]

/-! ## Label dimensions -/

theorem lookup_append_none (x : ℕ) (a b : List (ℕ × ℕ)) (h : ∀ p ∈ a, p.1 ≠ x) :
    (a ++ b).lookup x = b.lookup x := by
  induction a with
  | nil => rfl
  | cons p a ih =>
    obtain ⟨k, v⟩ := p
    have hk : (x == k) = false := by simpa using Ne.symm (h _ mem_cons_self)
    simp only [cons_append, List.lookup, hk]
    exact ih fun q hq => h q (mem_cons_of_mem _ hq)

theorem lookup_of_mem (l : List (ℕ × ℕ)) (hnd : (l.map Prod.fst).Nodup) {k v : ℕ}
    (h : (k, v) ∈ l) : l.lookup k = some v := by
  induction l with
  | nil => simp at h
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    simp only [map_cons, nodup_cons, mem_map] at hnd
    rcases mem_cons.1 h with e | h
    · simp only [Prod.mk.injEq] at e; obtain ⟨rfl, rfl⟩ := e; simp [List.lookup]
    · have hk : k ≠ k' := fun e => hnd.1 ⟨(k, v), h, e⟩
      have : (k == k') = false := by simpa using hk
      simp only [List.lookup, this]
      exact ih hnd.2 h

theorem L_keys (t : ℕ) : ∀ p ∈ L t, p.1 < B t := by
  intro p hp
  simp only [L, blockL, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap,
    mem_range] at hp
  have := two_le_B t
  rcases hp with (rfl | rfl) | ⟨s, hs, (rfl | rfl) | ⟨i, hi, rfl | rfl⟩⟩
  all_goals first | (dsimp only; omega) | exact sl_lt hs (by omega)

theorem labDim_mono (d : ℕ) {k x : ℕ} (hx : x < B k) :
    ∀ j, labDim (envAt (E0 d) L (k + j)) x = labDim (envAt (E0 d) L k) x
  | 0 => rfl
  | j + 1 => by
    rw [← labDim_mono d hx j, ← add_assoc]
    exact labsMono d (k + j) x (lt_of_lt_of_le hx (B_le (by omega)))

theorem labDim_block (d : ℕ) {t m x e : ℕ} (htm : t < m) (hx : B t ≤ x) (hx' : x < B (t + 1))
    (h : (blockL t).lookup x = some e) : labDim (envAt (E0 d) L m) x = dimv d e := by
  obtain ⟨j, rfl⟩ : ∃ j, m = t + 1 + j := ⟨m - (t + 1), by omega⟩
  rw [labDim_mono d hx' j]
  have hL : L (t + 1) = L t ++ blockL t := by simp [L, range_succ, flatMap_append]
  simp only [labDim, envAt, E0, hL]
  rw [lookup_append_none x _ _ fun p hp e => by have := L_keys t p hp; omega, h]
  rfl

theorem labDim_cv (d : ℕ) {t m : ℕ} (h : t ≤ m) : labDim (envAt (E0 d) L m) (cv t) = d ^ t := by
  cases t with
  | zero => simp [labDim, envAt, E0, L, cv, List.lookup, dimv]
  | succ t =>
    rw [cv, labDim_block d (t := t) (by omega) (by simp [sl]) (by rw [B_succ]; simp [sl]) (e := t + 2)]
    · rfl
    · simp [blockL, List.lookup]

theorem labDim_cw (d : ℕ) {t m : ℕ} (h : t ≤ m) : labDim (envAt (E0 d) L m) (cw t) = d ^ t := by
  cases t with
  | zero => simp [labDim, envAt, E0, L, cw, List.lookup, dimv]
  | succ t =>
    rw [cw, labDim_block d (t := t) (by omega) (by simp [sl]) (by rw [B_succ]; simp [sl]) (e := t + 2)]
    · rfl
    · have : (sl t 13 == sl t 12) = false := by simp [sl]
      simp [blockL, List.lookup, this]

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

theorem prodVal_ne {E : Env} {ns : List Node} {τ : ℕ → ℕ} (h : prodVal E ns τ ≠ 0) :
    ∀ x ∈ ns, val E τ x ≠ 0 := fun x hx h0 =>
  h (List.prod_eq_zero (List.mem_map.2 ⟨x, hx, h0⟩))

theorem tB_eq {s t k a b : ℕ} (h : tB s t k a b ≠ 0) : k = a * t + b := by
  unfold tB at h; split_ifs at h with h'; exact h'.2.2; exact absurd rfl h

theorem mem_hR (n : ℕ) : (⟨.Z, [cv 0], [], 1, 0⟩ : Node) ∈ (rhsN n).nodes :=
  mem_append_left _ (mem_append_left _ (mem_append_left _ (mem_append_left _
    (mem_append_left _ (mem_singleton_self _)))))

theorem mem_bR {n t : ℕ} (ht : t < n) :
    (⟨.B, [cv (t + 1)], [cv t, sl (n - 1 - t) 0], t + 1, 0⟩ : Node) ∈ (rhsN n).nodes :=
  mem_append_left _ (mem_append_left _ (mem_append_left _ (mem_append_left _
    (mem_append_right _ (mem_map.2 ⟨t, mem_range.2 ht, rfl⟩)))))

theorem mem_xR (n : ℕ) : (⟨.X, [cw n], [cv n], n + 1, 2 * n + 2⟩ : Node) ∈ (rhsN n).nodes :=
  mem_append_left _ (mem_append_left _ (mem_append_left _ (mem_append_right _
    (mem_singleton_self _))))

theorem mem_sR {n t : ℕ} (ht : t < n) :
    (⟨.S, [cw t, sl (n - 1 - t) 1], [cw (t + 1)], t + 1, 0⟩ : Node) ∈ (rhsN n).nodes :=
  mem_append_left _ (mem_append_left _ (mem_append_right _
    (mem_map.2 ⟨t, mem_range.2 ht, rfl⟩)))

theorem mem_eR (n : ℕ) : (⟨.Z, [], [cw 0], 1, 0⟩ : Node) ∈ (rhsN n).nodes :=
  mem_append_left _ (mem_append_right _ (mem_singleton_self _))

theorem rhs_support (d : ℕ) {n m : ℕ} {τ : ℕ → ℕ}
    (h : prodVal (envAt (E0 d) L m) (rhsN n).nodes τ ≠ 0) :
    ∀ t ≤ n, τ (cv t) = topv d τ n 0 t ∧ τ (cw t) = topv d τ n 1 t := by
  have hv := prodVal_ne h
  intro t ht
  induction t with
  | zero =>
    have h1 := hv _ (mem_hR n)
    have h2 := hv _ (mem_eR n)
    simp only [val, map_cons, map_nil, nil_append, cons_append] at h1 h2
    have e1 : (envAt (E0 d) L m).dimv 1 = 1 := by simp [envAt, E0, dimv]
    rw [e1, tZ_one _ _ (fun k => by simp [envAt, E0, phase])] at h1 h2
    exact ⟨by simpa [topv] using h1, by simpa [topv] using h2⟩
  | succ t ih =>
    obtain ⟨ih1, ih2⟩ := ih (by omega)
    have h1 := hv _ (mem_bR (n := n) (t := t) (by omega))
    have h2 := hv _ (mem_sR (n := n) (t := t) (by omega))
    simp only [val, map_cons, map_nil] at h1 h2
    have e1 := tB_eq h1
    have e2 := tB_eq h2
    simp only [envAt, E0, dimv] at e1 e2
    exact ⟨by rw [e1, ih1]; rfl, by rw [e2, ih2]; rfl⟩

/-- A sum over `ws` whose summand vanishes away from `τ0` is the summand at `τ0`. -/
theorem sumOver_single (D : ℕ → ℕ) (ws : List ℕ) (hnd : ws.Nodup) (f : (ℕ → ℕ) → ℂ)
    (σ τ0 : ℕ → ℕ) (h0 : ∀ x ∉ ws, τ0 x = σ x) (hr : ∀ l ∈ ws, τ0 l < D l)
    (hs : ∀ τ, (∀ x ∉ ws, τ x = σ x) → (∀ x ∈ ws, τ x < D x) → f τ ≠ 0 →
      ∀ l ∈ ws, τ l = τ0 l) :
    sumOver D ws f σ = f τ0 := by
  rw [sumOver_congr D ws f (fun τ => if ∀ l ∈ ws, τ l = τ0 l then f τ else 0) σ
    (fun τ h1 h2 => by
      split_ifs with h
      · rfl
      · by_contra hne; exact h (hs τ h1 h2 hne)), sumOver_pinV ws hnd τ0 hr]
  congr 1
  funext x
  split_ifs with h
  · rfl
  · exact (h0 x h).symm

theorem lookup_none (l : List (ℕ × ℕ)) (x : ℕ) (h : x ∉ l.map Prod.fst) : l.lookup x = none := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    simp only [map_cons, mem_cons, not_or] at h
    have : (x == k) = false := by simpa using h.1
    simp only [List.lookup, this]
    exact ih h.2

theorem rhsInt_mem {n x : ℕ} (h : x ∈ (rhsN n).int) : ∃ t ≤ n, x = cv t ∨ x = cw t := by
  simp only [rhsN, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range] at h
  rcases h with (rfl | rfl) | ⟨t, ht, rfl | rfl⟩
  · exact ⟨0, by omega, Or.inl rfl⟩
  · exact ⟨0, by omega, Or.inr rfl⟩
  · exact ⟨t + 1, by omega, Or.inl rfl⟩
  · exact ⟨t + 1, by omega, Or.inr rfl⟩

theorem sl_notin_rhsInt {n i s : ℕ} (hs : s < 12) : sl i s ∉ (rhsN n).int := by
  intro h
  obtain ⟨t, _, e | e⟩ := rhsInt_mem h <;> cases t with
  | zero => have := two_le_sl i s; simp [cv, cw] at e; omega
  | succ t => have := (sl_inj (by omega) (by omega) e).2; omega

theorem rhsInt_nodup (n : ℕ) : (rhsN n).int.Nodup := by
  simp only [rhsN]
  rw [nodup_append]
  refine ⟨by simp [cv, cw], ?_, ?_⟩
  · rw [nodup_flatMap]
    refine ⟨fun t _ => by simp [cv, cw, sl], ?_⟩
    refine (pairwise_lt_range (n := n)).imp fun {a b} hab => ?_
    simp only [Function.onFun, List.disjoint_cons_left, List.disjoint_cons_right, mem_cons,
      not_mem_nil, or_false, not_or, List.disjoint_nil_left, cv, cw]
    and_intros
    all_goals first
      | trivial
      | (intro e; have := (sl_inj (by omega) (by omega) e).1; omega)
      | simp
  · intro a ha b hb e
    simp only [mem_cons, not_mem_nil, or_false, cv, cw] at ha
    simp only [mem_flatMap, mem_range, mem_cons, not_mem_nil, or_false, cv, cw] at hb
    obtain ⟨t, _, rfl | rfl⟩ := hb <;> rcases ha with rfl | rfl <;>
      have := two_le_sl t 12 <;> have := two_le_sl t 13 <;> omega

/-- The pinned values of the spec's summed labels. -/
def pinsR (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : List (ℕ × ℕ) :=
  [(cv 0, 0), (cw 0, 0)] ++ (range n).flatMap fun t =>
    [(cv (t + 1), topv d σ n 0 (t + 1)), (cw (t + 1), topv d σ n 1 (t + 1))]

/-- `σ` with the spec's summed labels at their pinned values. -/
def tauR (d : ℕ) (σ : ℕ → ℕ) (n x : ℕ) : ℕ := ((pinsR d σ n).lookup x).getD (σ x)

theorem pinsR_fst (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : (pinsR d σ n).map Prod.fst = (rhsN n).int := by
  simp [pinsR, rhsN, map_flatMap]

theorem tauR_off {d : ℕ} {σ : ℕ → ℕ} {n x : ℕ} (h : x ∉ (rhsN n).int) : tauR d σ n x = σ x := by
  rw [tauR, lookup_none _ _ (by rw [pinsR_fst]; exact h)]; rfl

theorem pinsR_nd (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : ((pinsR d σ n).map Prod.fst).Nodup := by
  rw [pinsR_fst]; exact rhsInt_nodup n

theorem tauR_cv {d : ℕ} {σ : ℕ → ℕ} {n t : ℕ} (ht : t ≤ n) :
    tauR d σ n (cv t) = topv d σ n 0 t := by
  cases t with
  | zero => rw [tauR, lookup_of_mem _ (pinsR_nd d σ n) (mem_append_left _ mem_cons_self)]; rfl
  | succ t =>
    rw [tauR, lookup_of_mem _ (pinsR_nd d σ n) (mem_append_right _
      (mem_flatMap.2 ⟨t, mem_range.2 (by omega), mem_cons_self⟩))]
    rfl

theorem tauR_cw {d : ℕ} {σ : ℕ → ℕ} {n t : ℕ} (ht : t ≤ n) :
    tauR d σ n (cw t) = topv d σ n 1 t := by
  cases t with
  | zero =>
    rw [tauR, lookup_of_mem _ (pinsR_nd d σ n) (mem_append_left _
      (mem_cons_of_mem _ mem_cons_self))]
    rfl
  | succ t =>
    rw [tauR, lookup_of_mem _ (pinsR_nd d σ n) (mem_append_right _
      (mem_flatMap.2 ⟨t, mem_range.2 (by omega), mem_cons_of_mem _ mem_cons_self⟩))]
    rfl

theorem phase_shift (d n k : ℕ) : phase d (2 * n + 2) k = -(2 * (k : ℝ)) / ((d ^ n : ℕ) : ℝ) := by
  have h1 : (2 * n + 2) % 2 = 0 := by omega
  have h2 : (2 * n + 2) / 2 - 1 = n := by omega
  simp only [phase, h1, h2]
  push_cast
  simp

theorem phase_zero (d k : ℕ) : phase d 0 k = 0 := by simp [phase]

theorem cv_off_int {n x : ℕ} (hc : x ∈ cut n) (h : x ∈ (rhsN n).int) : False :=
  (fitsR n).2 n le_rfl x h hc

/-- The spec in closed form: the shift by `2` mod `d ^ n` of the digit values, the pending legs
on the inputs and outputs. -/
theorem eval_rhsN (d : ℕ) (hd : 0 < d) {n m : ℕ} (hm : n ≤ m) (σ : ℕ → ℕ)
    (ha : ∀ i < n, σ (sl i 0) < d) (hb : ∀ i < n, σ (sl i 1) < d) :
    eval (envAt (E0 d) L m) (rhsN n) σ =
      (if topv d σ n 1 n % d ^ n = (topv d σ n 0 n + 2) % d ^ n then 1 else 0) *
        ((range n).map fun i => (if σ (pend n i) = σ (sl i 0) then (1 : ℂ) else 0) *
          (if σ (pendO n i) = σ (sl i 1) then 1 else 0)).prod := by
  have hdig : ∀ s < 12, ∀ i, tauR d σ n (sl i s) = σ (sl i s) :=
    fun s hs i => tauR_off (sl_notin_rhsInt hs)
  unfold eval
  rw [sumOver_single _ _ (rhsInt_nodup n) _ σ (tauR d σ n) (fun x hx => tauR_off hx) ?hr ?hs]
  case hr =>
    intro l hl
    obtain ⟨t, ht, rfl | rfl⟩ := rhsInt_mem hl
    · rw [tauR_cv ht, labDim_cv d (by omega)]; exact topv_lt ha t ht
    · rw [tauR_cw ht, labDim_cw d (by omega)]; exact topv_lt hb t ht
  case hs =>
    intro τ h0 _ hne l hl
    have hsup := rhs_support d hne
    obtain ⟨t, ht, rfl | rfl⟩ := rhsInt_mem hl
    · rw [(hsup t ht).1, tauR_cv ht, topv_congr (fun i _ => h0 _ (sl_notin_rhsInt (by omega))) t ht]
    · rw [(hsup t ht).2, tauR_cw ht, topv_congr (fun i _ => h0 _ (sl_notin_rhsInt (by omega))) t ht]
  have eD : ∀ e, (envAt (E0 d) L m).dimv e = dimv d e := fun _ => rfl
  have eP : ∀ p, (envAt (E0 d) L m).phase p = phase d p := fun _ => rfl
  have hH : val (envAt (E0 d) L m) (tauR d σ n) ⟨.Z, [cv 0], [], 1, 0⟩ = 1 := by
    simp only [val, map_cons, map_nil, append_nil, eD, eP, dimv, pow_zero]
    rw [tZ_one _ _ (phase_zero d), tauR_cv (Nat.zero_le _)]; rfl
  have hE : val (envAt (E0 d) L m) (tauR d σ n) ⟨.Z, [], [cw 0], 1, 0⟩ = 1 := by
    simp only [val, map_cons, map_nil, nil_append, eD, eP, dimv, pow_zero]
    rw [tZ_one _ _ (phase_zero d), tauR_cw (Nat.zero_le _)]; rfl
  have hX : val (envAt (E0 d) L m) (tauR d σ n) ⟨.X, [cw n], [cv n], n + 1, 2 * n + 2⟩ =
      if topv d σ n 1 n % d ^ n = (topv d σ n 0 n + 2) % d ^ n then 1 else 0 := by
    simp only [val, map_cons, map_nil, eD, eP, dimv]
    rw [tauR_cv le_rfl, tauR_cw le_rfl, show phase d (2 * n + 2) =
      fun k : ℕ => -(2 * (k : ℝ)) / ((d ^ n : ℕ) : ℝ) from funext (phase_shift d n),
      tX_shift _ (pow_pos hd n)]
  have hBs : ((range n).map fun t => val (envAt (E0 d) L m) (tauR d σ n)
      ⟨.B, [cv (t + 1)], [cv t, sl (n - 1 - t) 0], t + 1, 0⟩).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hx
    have ht := mem_range.1 ht
    simp only [val, map_cons, map_nil, eD, dimv]
    rw [tauR_cv (by omega), tauR_cv (by omega), hdig 0 (by omega)]
    unfold tB
    rw [if_pos ⟨topv_lt ha t (by omega), ha _ (by omega), rfl⟩]
  have hSs : ((range n).map fun t => val (envAt (E0 d) L m) (tauR d σ n)
      ⟨.S, [cw t, sl (n - 1 - t) 1], [cw (t + 1)], t + 1, 0⟩).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hx
    have ht := mem_range.1 ht
    simp only [val, map_cons, map_nil, eD, dimv]
    rw [tauR_cw (by omega), tauR_cw (by omega), hdig 1 (by omega)]
    unfold tB
    rw [if_pos ⟨topv_lt hb t (by omega), hb _ (by omega), rfl⟩]
  have hPs : (((range n).flatMap fun i => [zN [pend n i] [sl i 0], zN [pendO n i] [sl i 1]]).map
      (val (envAt (E0 d) L m) (tauR d σ n))).prod =
      ((range n).map fun i => (if σ (pend n i) = σ (sl i 0) then (1 : ℂ) else 0) *
        (if σ (pendO n i) = σ (sl i 1) then 1 else 0)).prod := by
    rw [map_flatMap, prod_flatMapC]
    congr 1
    refine List.map_congr_left fun i hi => ?_
    have hi := mem_range.1 hi
    simp only [map_cons, map_nil, prod_cons, prod_nil, mul_one, zN, val, cons_append, nil_append,
      eD, eP, dimv]
    rw [tauR_off (fun h => cv_off_int (pend_mem hi) h), tauR_off (fun h => cv_off_int (pendO_mem hi) h),
      hdig 0 (by omega), hdig 1 (by omega), tZ_two _ _ _ _ (phase_zero d) (ha i hi),
      tZ_two _ _ _ _ (phase_zero d) (hb i hi)]
  simp only [rhsN, prodVal, map_append, prod_append, map_cons, map_nil, prod_cons, prod_nil,
    mul_one, map_map]
  try simp only [Function.comp_def]
  rw [hH, hBs, hX, hSs, hE, hPs]
  ring

/-! ## Copy `n` in closed form -/

theorem nodup_blocks (n c : ℕ) (A R : List ℕ) (hA : A.Nodup) (hR : R.Nodup)
    (hR8 : ∀ r ∈ R, r < 8) (hAc : ∀ a ∈ A, a < c) :
    (A ++ (range n).flatMap fun i => R.map fun r => c + 8 * i + r).Nodup := by
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

theorem sl_injective (n : ℕ) : Function.Injective (sl n) := fun a b h => by unfold sl at h; omega

theorem pinC_eq (n : ℕ) : pinC n = (([2, 4, 5, 6, 9] ++ (range n).flatMap fun i =>
    [0, 1, 2, 3, 4].map fun r => 16 + 8 * i + r)).map (sl n) := by
  simp only [pinC, map_append, map_flatMap, map_cons, map_nil, sl]
  ring_nf

theorem pinC_nodup (n : ℕ) : (pinC n).Nodup := by
  rw [pinC_eq]
  exact (nodup_blocks n 16 _ _ (by decide) (by decide) (by decide) (by decide)).map
    (sl_injective n)

/-- The dimension of the pair `(n, i)`. -/
def pD (d n i : ℕ) : ℕ := d ^ (n - i + 1)

/-- The values the nodes of copy `n` pin its pinned labels to, read from `ρ`. -/
def valsC (d n : ℕ) (ρ : ℕ → ℕ) : List (ℕ × ℕ) :=
  [(sl n 2, ρ (sl n 0)), (sl n 4, ρ (sl n 3)), (sl n 5, ρ (sl n 3)), (sl n 6, ρ (sl n 3)),
    (sl n 9, ρ (sl n 1))] ++
  (range n).flatMap fun i => [(sl n (16 + 8 * i), ρ (pend n i)), (sl n (17 + 8 * i), ρ (pendO n i)),
    (sl n (18 + 8 * i), ρ (sl n 3)), (sl n (19 + 8 * i), ρ (sl n 3)),
    (sl n (20 + 8 * i), (pD d n i - ρ (pendO n i) % pD d n i) % pD d n i)]

/-- `ρ` with the pinned labels of copy `n` at their pins. -/
def tauC (d n : ℕ) (ρ : ℕ → ℕ) (x : ℕ) : ℕ := ((valsC d n ρ).lookup x).getD (ρ x)

theorem valsC_fst (d n : ℕ) (ρ : ℕ → ℕ) : (valsC d n ρ).map Prod.fst = pinC n := by
  simp [valsC, pinC, map_flatMap]

theorem tauC_of_mem {d n x y : ℕ} {ρ : ℕ → ℕ} (h : (x, y) ∈ valsC d n ρ) : tauC d n ρ x = y := by
  rw [tauC, lookup_of_mem _ (by rw [valsC_fst]; exact pinC_nodup n) h]; rfl

theorem tauC_off {d n x : ℕ} {ρ : ℕ → ℕ} (h : x ∉ pinC n) : tauC d n ρ x = ρ x := by
  rw [tauC, lookup_none _ _ (by rw [valsC_fst]; exact h)]; rfl

theorem pend_ne {n i t : ℕ} (hi : i < n) (ht : t < 8 * n + 22) : pend n i ≠ sl n t := fun h =>
  absurd (sl_inj (by omega) ht h).1 (by omega)

theorem pendO_ne {n i t : ℕ} (hi : i < n) (ht : t < 8 * n + 22) : pendO n i ≠ sl n t := fun h =>
  absurd (sl_inj (by omega) ht h).1 (by omega)

theorem notin_pinC {n t : ℕ} (ht : t < 8 * n + 22) (h2 : t ≠ 2) (h4 : t ≠ 4) (h5 : t ≠ 5)
    (h6 : t ≠ 6) (h9 : t ≠ 9) (hb : ∀ i < n, ∀ r < 5, t ≠ 16 + 8 * i + r) : sl n t ∉ pinC n := by
  rw [pinC_eq]
  intro h
  obtain ⟨u, hu, e⟩ := mem_map.1 h
  have e := sl_injective n e
  subst e
  simp only [mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range, mem_map] at hu
  rcases hu with (rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, r, hr, rfl⟩
  all_goals first | omega | exact hb i hi r (by omega) rfl

theorem pend_notin {n i : ℕ} (hi : i < n) : pend n i ∉ pinC n := by
  rw [pinC_eq]; intro h
  obtain ⟨u, hu, e⟩ := mem_map.1 h
  have hu8 : u < 8 * n + 22 := by
    simp only [mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range, mem_map] at hu
    rcases hu with (rfl | rfl | rfl | rfl | rfl) | ⟨j, hj, r, hr, rfl⟩
    all_goals first | omega | (simp at hr; omega)
  exact pend_ne hi hu8 e.symm

theorem pendO_notin {n i : ℕ} (hi : i < n) : pendO n i ∉ pinC n := by
  rw [pinC_eq]; intro h
  obtain ⟨u, hu, e⟩ := mem_map.1 h
  have hu8 : u < 8 * n + 22 := by
    simp only [mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range, mem_map] at hu
    rcases hu with (rfl | rfl | rfl | rfl | rfl) | ⟨j, hj, r, hr, rfl⟩
    all_goals first | omega | (simp at hr; omega)
  exact pendO_ne hi hu8 e.symm

theorem valsC_congr {d n : ℕ} {τ ρ : ℕ → ℕ} (h : ∀ x ∉ pinC n, τ x = ρ x) :
    valsC d n τ = valsC d n ρ := by
  have h0 := h _ (notin_pinC (n := n) (t := 0) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega) (fun _ _ _ _ => by omega))
  have h1 := h _ (notin_pinC (n := n) (t := 1) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega) (fun _ _ _ _ => by omega))
  have h3 := h _ (notin_pinC (n := n) (t := 3) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega) (fun _ _ _ _ => by omega))
  simp only [valsC, h0, h1, h3]
  congr 1
  refine flatMap_congr fun i hi => ?_
  rw [h _ (pend_notin (mem_range.1 hi)), h _ (pendO_notin (mem_range.1 hi))]

theorem mem_cpN {n : ℕ} {x : Node} (h : x ∈ [zN [sl n 2, sl n (14 + 8 * n)] [sl n 0],
     fN 0 (sl n 3) (sl n 2),
     zN (sl n 4 :: (range n).map fun i => sl n (18 + 8 * i)) [sl n 3],
     zN [sl n 5] [sl n 4] (2 * n + 1),
     zN (sl n 6 :: (range n).map fun i => sl n (19 + 8 * i)) [sl n 5],
     fN 0 (sl n 7) (sl n 6),
     fN 0 (sl n 8) (sl n 7),
     fN 0 (sl n 9) (sl n 8),
     zN [sl n 1, sl n (15 + 8 * n)] [sl n 9]]) : x ∈ (cp n).nodes := mem_append_left _ h

theorem mem_cpP {n i : ℕ} (hi : i < n) {x : Node} (h : x ∈ pairN n i) : x ∈ (cp n).nodes :=
  mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi, h⟩)

theorem zeq {E : Env} {τ : ℕ → ℕ} {os is : List ℕ} {ph : ℕ} (h : val E τ (zN os is ph) ≠ 0)
    {a b : ℕ} (ha : a ∈ os ++ is) (hb : b ∈ os ++ is) : τ a = τ b :=
  Z_eq E τ _ rfl h ha hb

/-- The antipode cup: the X spider of dimension `D` with two outputs. -/
theorem tX_cup (D : ℕ) (hD : 0 < D) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (a b : ℕ) :
    tX D φ [a, b] [] = if (a + b) % D = 0 then 1 else 0 := by
  rw [← fou_sq D hD a b]
  unfold tX
  refine Finset.sum_congr rfl fun k _ => ?_
  simp only [hφ, ex_zero, one_mul, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
    mul_one]
  congr 1
  unfold fou; rw [mul_comm k b]

theorem cup_val {D a b : ℕ} (hD : 0 < D) (ha : a < D) (h : (a + b) % D = 0) :
    a = (D - b % D) % D := by
  have hb := Nat.mod_lt b hD
  have h' : (a + b % D) % D = 0 := by rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]; exact h
  obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero h'
  rcases k with _ | _ | k
  · have : a = 0 := by omega
    have : b % D = 0 := by omega
    subst_vars; simp_all
  · have : a = D - b % D := by omega
    rcases Nat.eq_zero_or_pos (b % D) with h0 | h0
    · rw [h0] at this ⊢; omega
    · rw [Nat.mod_eq_of_lt (by omega)]; exact this
  · nlinarith

theorem cp_support {d m n : ℕ} (hd : 0 < d) {τ : ℕ → ℕ}
    (hz : ∀ i < n, τ (sl n (20 + 8 * i)) < pD d n i)
    (h : prodVal (envAt (E0 d) L m) (cp n).nodes τ ≠ 0) :
    ∀ l ∈ pinC n, τ l = tauC d n τ l := by
  have hv := prodVal_ne h
  have e2 : τ (sl n 2) = τ (sl n 0) := zeq (hv (zN [sl n 2, sl n (14 + 8 * n)] [sl n 0])
    (mem_cpN (by simp))) (by simp) (by simp)
  have e4 : τ (sl n 4) = τ (sl n 3) := zeq (hv (zN (sl n 4 :: (range n).map fun i =>
    sl n (18 + 8 * i)) [sl n 3]) (mem_cpN (by simp))) (by simp) (by simp)
  have e5 : τ (sl n 5) = τ (sl n 4) := zeq (hv (zN [sl n 5] [sl n 4] (2 * n + 1))
    (mem_cpN (by simp))) (by simp) (by simp)
  have e6 : τ (sl n 6) = τ (sl n 5) := zeq (hv (zN (sl n 6 :: (range n).map fun i =>
    sl n (19 + 8 * i)) [sl n 5]) (mem_cpN (by simp))) (by simp) (by simp)
  have e9 : τ (sl n 9) = τ (sl n 1) := zeq (hv (zN [sl n 1, sl n (15 + 8 * n)] [sl n 9])
    (mem_cpN (by simp))) (by simp) (by simp)
  have el : ∀ i < n, τ (sl n (18 + 8 * i)) = τ (sl n 3) := fun i hi =>
    zeq (hv (zN (sl n 4 :: (range n).map fun i => sl n (18 + 8 * i)) [sl n 3])
      (mem_cpN (by simp))) (mem_append_left _ (mem_cons_of_mem _
        (mem_map.2 ⟨i, mem_range.2 hi, rfl⟩))) (by simp)
  have el' : ∀ i < n, τ (sl n (19 + 8 * i)) = τ (sl n 5) := fun i hi =>
    zeq (hv (zN (sl n 6 :: (range n).map fun i => sl n (19 + 8 * i)) [sl n 5])
      (mem_cpN (by simp))) (mem_append_left _ (mem_cons_of_mem _
        (mem_map.2 ⟨i, mem_range.2 hi, rfl⟩))) (by simp)
  have ey : ∀ i < n, τ (sl n (16 + 8 * i)) = τ (pend n i) := fun i hi =>
    zeq (hv (zN [sl n (16 + 8 * i), sl n (14 + 8 * i)] [pend n i])
      (mem_cpP hi (by simp [pairN]))) (by simp) (by simp)
  have ey' : ∀ i < n, τ (sl n (17 + 8 * i)) = τ (pendO n i) := fun i hi =>
    zeq (hv (zN [sl n (17 + 8 * i), sl n (15 + 8 * i)] [pendO n i])
      (mem_cpP hi (by simp [pairN]))) (by simp) (by simp)
  have ez : ∀ i < n, τ (sl n (20 + 8 * i)) =
      (pD d n i - τ (pendO n i) % pD d n i) % pD d n i := fun i hi => by
    have hx := hv ⟨.X, [sl n (20 + 8 * i), sl n (17 + 8 * i)], [], n - i + 2, 0⟩
      (mem_cpP hi (by simp [pairN]))
    have hpD : 0 < pD d n i := pow_pos hd _
    simp only [val, map_cons, map_nil] at hx
    rw [show (envAt (E0 d) L m).dimv (n - i + 2) = pD d n i from rfl,
      tX_cup _ hpD ((envAt (E0 d) L m).phase 0) (fun k => phase_zero d k)] at hx
    rw [← ey' i hi]
    exact cup_val hpD (hz i hi) (by by_contra hc; exact hx (if_neg hc))
  intro l hl
  simp only [pinC, mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range] at hl
  rcases hl with (rfl | rfl | rfl | rfl | rfl) | ⟨i, hi, rfl | rfl | rfl | rfl | rfl⟩
  · rw [tauC_of_mem (y := τ (sl n 0)) (by simp [valsC, sl]), e2]
  · rw [tauC_of_mem (y := τ (sl n 3)) (by simp [valsC, sl]), e4]
  · rw [tauC_of_mem (y := τ (sl n 3)) (by simp [valsC, sl]), e5, e4]
  · rw [tauC_of_mem (y := τ (sl n 3)) (by simp [valsC, sl]), e6, e5, e4]
  · rw [tauC_of_mem (y := τ (sl n 1)) (by simp [valsC, sl]), e9]
  · rw [tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi,
      mem_cons_self⟩)), ey i hi]
  · rw [tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi,
      mem_cons_of_mem _ mem_cons_self⟩)), ey' i hi]
  · rw [tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi,
      mem_cons_of_mem _ (mem_cons_of_mem _ mem_cons_self)⟩)), el i hi]
  · rw [tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi,
      mem_cons_of_mem _ (mem_cons_of_mem _ (mem_cons_of_mem _ mem_cons_self))⟩)), el' i hi, e5, e4]
  · rw [tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi,
      mem_cons_of_mem _ (mem_cons_of_mem _ (mem_cons_of_mem _ (mem_cons_of_mem _
        mem_cons_self)))⟩)), ez i hi]

theorem lookup_some_of_mem (l : List (ℕ × ℕ)) {x : ℕ} (h : x ∈ l.map Prod.fst) :
    ∃ v, l.lookup x = some v := by
  induction l with
  | nil => simp at h
  | cons p l ih =>
    obtain ⟨k, v⟩ := p
    by_cases hk : x = k
    · subst hk; exact ⟨v, by simp [List.lookup]⟩
    · have : (x == k) = false := by simpa using hk
      simp only [List.lookup, this]
      exact ih (by simpa [hk] using h)

theorem tauC_congr {d n : ℕ} {τ ρ : ℕ → ℕ} (h : ∀ x ∉ pinC n, τ x = ρ x) (x : ℕ) :
    tauC d n τ x = tauC d n ρ x := by
  by_cases hx : x ∈ pinC n
  · obtain ⟨v, hv⟩ := lookup_some_of_mem (valsC d n ρ) (by rw [valsC_fst]; exact hx)
    simp only [tauC, valsC_congr h, hv, Option.getD_some]
  · rw [tauC_off hx, tauC_off hx, h x hx]

theorem labDim_blk (d : ℕ) {t m x : ℕ} (htm : t < m) (hx : B t ≤ x) (hx' : x < B (t + 1)) :
    labDim (envAt (E0 d) L m) x = dimv d (((blockL t).lookup x).getD 0) := by
  obtain ⟨j, rfl⟩ : ∃ j, m = t + 1 + j := ⟨m - (t + 1), by omega⟩
  rw [labDim_mono d hx' j]
  have hL : L (t + 1) = L t ++ blockL t := by simp [L, range_succ, flatMap_append]
  simp only [labDim, envAt, E0, hL]
  rw [lookup_append_none x _ _ fun p hp e => by have := L_keys t p hp; omega]

theorem blockL_fst (n : ℕ) : (blockL n).map Prod.fst = (([12, 13] ++ (range n).flatMap fun i =>
    [0, 1].map fun r => 20 + 8 * i + r)).map (sl n) := by
  simp only [blockL, map_append, map_flatMap, map_cons, map_nil, sl]
  ring_nf

theorem labDim_plain (d : ℕ) {n m t : ℕ} (hnm : n < m) (ht : t < 8 * n + 22) (h12 : t ≠ 12)
    (h13 : t ≠ 13) (hb : ∀ i < n, ∀ r < 2, t ≠ 20 + 8 * i + r) :
    labDim (envAt (E0 d) L m) (sl n t) = d := by
  rw [labDim_blk d hnm (by simp [sl]) (by rw [B_succ]; unfold sl; omega),
    lookup_none _ _ (by
      rw [blockL_fst]; intro h
      obtain ⟨u, hu, e⟩ := mem_map.1 h
      have e := sl_injective n e
      subst e
      simp only [mem_append, mem_cons, not_mem_nil, or_false, mem_flatMap, mem_range,
        mem_map] at hu
      rcases hu with (rfl | rfl) | ⟨i, hi, r, hr, rfl⟩
      all_goals first | omega | exact hb i hi r (by omega) rfl)]
  rfl

theorem labDim_z (d : ℕ) {n m i : ℕ} (hnm : n < m) (hi : i < n) :
    labDim (envAt (E0 d) L m) (sl n (20 + 8 * i)) = pD d n i := by
  rw [labDim_blk d hnm (by simp [sl]) (by rw [B_succ]; unfold sl; omega),
    lookup_of_mem (blockL n) (by
      rw [blockL_fst]
      exact (nodup_blocks n 20 _ _ (by decide) (by decide) (by decide) (by decide)).map
        (sl_injective n)) (k := sl n (20 + 8 * i)) (v := n - i + 2)
      (mem_append_right _ (mem_flatMap.2 ⟨i, mem_range.2 hi, mem_cons_self⟩))]
  show d ^ (n - i + 1) = _; rfl

/-- The pending-leg factors of copy `n`. -/
def pinsCp (n : ℕ) (σ : ℕ → ℕ) : ℂ :=
  (if σ (sl n (14 + 8 * n)) = σ (sl n 0) then 1 else 0) *
    (if σ (sl n (15 + 8 * n)) = σ (sl n 1) then 1 else 0) *
    ((range n).map fun i => (if σ (sl n (14 + 8 * i)) = σ (pend n i) then (1 : ℂ) else 0) *
      (if σ (sl n (15 + 8 * i)) = σ (pendO n i) then 1 else 0)).prod

/-- The pair factors of copy `n` at the Fourier value `x`. -/
noncomputable def pairsF (d n : ℕ) (σ : ℕ → ℕ) (x : ℕ) : ℂ :=
  ((range n).map fun i => fou (pD d n i) x (σ (pend n i)) *
    fou (pD d n i) x ((pD d n i - σ (pendO n i) % pD d n i) % pD d n i)).prod

theorem tZ3 (D a b : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (ha : a < D) :
    tZ D φ [a, b, a] = if b = a then 1 else 0 := by
  unfold tZ
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq, hφ, ex_zero]
  rw [Finset.sum_eq_single a (fun k _ hk => if_neg fun h' => hk h'.1.symm)
    (fun h' => absurd (Finset.mem_range.2 ha) h')]
  by_cases h : b = a <;> simp [h]

theorem tZ_all (D k : ℕ) (φ : ℕ → ℝ) (vs : List ℕ) (hne : vs ≠ []) (hv : ∀ v ∈ vs, v = k)
    (hk : k < D) : tZ D φ vs = ex (φ k) := by
  obtain ⟨v, hv0⟩ := List.exists_mem_of_ne_nil vs hne
  unfold tZ
  rw [Finset.sum_eq_single k (fun j _ hj => if_neg fun h' => hj ((h' v hv0).symm.trans (hv v hv0)))
    (fun h' => absurd (Finset.mem_range.2 hk) h'), if_pos hv]

theorem notin_low {n t : ℕ} (ht : t < 16) (h2 : t ≠ 2) (h4 : t ≠ 4) (h5 : t ≠ 5) (h6 : t ≠ 6)
    (h9 : t ≠ 9) : sl n t ∉ pinC n :=
  notin_pinC (by omega) h2 h4 h5 h6 h9 fun _ _ _ _ => by omega

/-- Copy `n`'s nodes at the pinned labels. -/
theorem prod_cp (d : ℕ) (hd : 0 < d) (n m : ℕ) (ρ : ℕ → ℕ) (h0 : ρ (sl n 0) < d) (h1 : ρ (sl n 1) < d)
    (h3 : ρ (sl n 3) < d) (hA : ∀ i < n, ρ (pend n i) < d) (hB : ∀ i < n, ρ (pendO n i) < d) :
    prodVal (envAt (E0 d) L m) (cp n).nodes (tauC d n ρ) = pinsCp n ρ *
      (fou d (ρ (sl n 3)) (ρ (sl n 0)) * ex (phase d (2 * n + 1) (ρ (sl n 3))) *
        (fou d (ρ (sl n 7)) (ρ (sl n 3)) * fou d (ρ (sl n 8)) (ρ (sl n 7)) *
          fou d (ρ (sl n 1)) (ρ (sl n 8))) * pairsF d n ρ (ρ (sl n 3))) := by
  have eD : ∀ e, (envAt (E0 d) L m).dimv e = dimv d e := fun _ => rfl
  have eP : ∀ p, (envAt (E0 d) L m).phase p = phase d p := fun _ => rfl
  have t2 : tauC d n ρ (sl n 2) = ρ (sl n 0) := tauC_of_mem (by simp [valsC, sl])
  have t4 : tauC d n ρ (sl n 4) = ρ (sl n 3) := tauC_of_mem (by simp [valsC, sl])
  have t5 : tauC d n ρ (sl n 5) = ρ (sl n 3) := tauC_of_mem (by simp [valsC, sl])
  have t6 : tauC d n ρ (sl n 6) = ρ (sl n 3) := tauC_of_mem (by simp [valsC, sl])
  have t9 : tauC d n ρ (sl n 9) = ρ (sl n 1) := tauC_of_mem (by simp [valsC, sl])
  have o0 : tauC d n ρ (sl n 0) = ρ (sl n 0) := tauC_off (notin_low (by omega) (by omega)
    (by omega) (by omega) (by omega) (by omega))
  have o1 : tauC d n ρ (sl n 1) = ρ (sl n 1) := tauC_off (notin_low (by omega) (by omega)
    (by omega) (by omega) (by omega) (by omega))
  have o3 : tauC d n ρ (sl n 3) = ρ (sl n 3) := tauC_off (notin_low (by omega) (by omega)
    (by omega) (by omega) (by omega) (by omega))
  have o7 : tauC d n ρ (sl n 7) = ρ (sl n 7) := tauC_off (notin_low (by omega) (by omega)
    (by omega) (by omega) (by omega) (by omega))
  have o8 : tauC d n ρ (sl n 8) = ρ (sl n 8) := tauC_off (notin_low (by omega) (by omega)
    (by omega) (by omega) (by omega) (by omega))
  have oP : tauC d n ρ (sl n (14 + 8 * n)) = ρ (sl n (14 + 8 * n)) := tauC_off (notin_pinC
    (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) fun _ _ _ _ => by omega)
  have oQ : tauC d n ρ (sl n (15 + 8 * n)) = ρ (sl n (15 + 8 * n)) := tauC_off (notin_pinC
    (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) fun _ _ _ _ => by omega)
  have hfix : (([zN [sl n 2, sl n (14 + 8 * n)] [sl n 0],
     fN 0 (sl n 3) (sl n 2),
     zN (sl n 4 :: (range n).map fun i => sl n (18 + 8 * i)) [sl n 3],
     zN [sl n 5] [sl n 4] (2 * n + 1),
     zN (sl n 6 :: (range n).map fun i => sl n (19 + 8 * i)) [sl n 5],
     fN 0 (sl n 7) (sl n 6),
     fN 0 (sl n 8) (sl n 7),
     fN 0 (sl n 9) (sl n 8),
     zN [sl n 1, sl n (15 + 8 * n)] [sl n 9]] : List Node).map
       (val (envAt (E0 d) L m) (tauC d n ρ))).prod =
      (if ρ (sl n (14 + 8 * n)) = ρ (sl n 0) then 1 else 0) *
      (if ρ (sl n (15 + 8 * n)) = ρ (sl n 1) then 1 else 0) *
      (fou d (ρ (sl n 3)) (ρ (sl n 0)) * ex (phase d (2 * n + 1) (ρ (sl n 3))) *
        (fou d (ρ (sl n 7)) (ρ (sl n 3)) * fou d (ρ (sl n 8)) (ρ (sl n 7)) *
          fou d (ρ (sl n 1)) (ρ (sl n 8)))) := by
    have tl : ∀ i ∈ range n, tauC d n ρ (sl n (18 + 8 * i)) = ρ (sl n 3) := fun i hi =>
      tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, hi,
        mem_cons_of_mem _ (mem_cons_of_mem _ mem_cons_self)⟩))
    have tl' : ∀ i ∈ range n, tauC d n ρ (sl n (19 + 8 * i)) = ρ (sl n 3) := fun i hi =>
      tauC_of_mem (mem_append_right _ (mem_flatMap.2 ⟨i, hi,
        mem_cons_of_mem _ (mem_cons_of_mem _ (mem_cons_of_mem _ mem_cons_self))⟩))
    simp only [map_cons, map_nil, prod_cons, prod_nil, mul_one, val, zN, fN, cons_append,
      nil_append, eD, eP, dimv, map_map]
    rw [t2, oP, o0, o3, t4, t5, t6, o7, o8, t9, o1, oQ,
      tZ3 _ _ _ _ (phase_zero d) h0, tZ3 _ _ _ _ (phase_zero d) h1,
      tZ_all d (ρ (sl n 3)) _ _ (by simp) (fun v hv => by
        simp only [mem_cons, mem_append, mem_map, Function.comp, mem_singleton, not_mem_nil,
          or_false] at hv
        rcases hv with rfl | ⟨i, hi, rfl⟩ | rfl <;> simp_all) h3,
      tZ_all d (ρ (sl n 3)) _ _ (by simp) (fun v hv => by simp at hv; omega) h3,
      tZ_all d (ρ (sl n 3)) _ _ (by simp) (fun v hv => by
        simp only [mem_cons, mem_append, mem_map, Function.comp, mem_singleton, not_mem_nil,
          or_false] at hv
        rcases hv with rfl | ⟨i, hi, rfl⟩ | rfl <;> simp_all) h3,
      phase_zero, ex_zero]
    ring
  have hpair : ∀ i ∈ range n, (((pairN n i).map (val (envAt (E0 d) L m) (tauC d n ρ))).prod) =
      ((if ρ (sl n (14 + 8 * i)) = ρ (pend n i) then (1 : ℂ) else 0) *
        (if ρ (sl n (15 + 8 * i)) = ρ (pendO n i) then 1 else 0)) *
      (fou (pD d n i) (ρ (sl n 3)) (ρ (pend n i)) *
        fou (pD d n i) (ρ (sl n 3)) ((pD d n i - ρ (pendO n i) % pD d n i) % pD d n i)) := by
    intro i hi'
    have hi := mem_range.1 hi'
    have hpD : 0 < pD d n i := pow_pos hd _
    have mem : ∀ {k : ℕ × ℕ}, k ∈ [(sl n (16 + 8 * i), ρ (pend n i)),
        (sl n (17 + 8 * i), ρ (pendO n i)), (sl n (18 + 8 * i), ρ (sl n 3)),
        (sl n (19 + 8 * i), ρ (sl n 3)),
        (sl n (20 + 8 * i), (pD d n i - ρ (pendO n i) % pD d n i) % pD d n i)] →
        k ∈ valsC d n ρ := fun hk => mem_append_right _ (mem_flatMap.2 ⟨i, hi', hk⟩)
    have t16 : tauC d n ρ (sl n (16 + 8 * i)) = ρ (pend n i) := tauC_of_mem (mem mem_cons_self)
    have t17 : tauC d n ρ (sl n (17 + 8 * i)) = ρ (pendO n i) :=
      tauC_of_mem (mem (mem_cons_of_mem _ mem_cons_self))
    have t18 : tauC d n ρ (sl n (18 + 8 * i)) = ρ (sl n 3) :=
      tauC_of_mem (mem (mem_cons_of_mem _ (mem_cons_of_mem _ mem_cons_self)))
    have t19 : tauC d n ρ (sl n (19 + 8 * i)) = ρ (sl n 3) :=
      tauC_of_mem (mem (mem_cons_of_mem _ (mem_cons_of_mem _ (mem_cons_of_mem _ mem_cons_self))))
    have t20 : tauC d n ρ (sl n (20 + 8 * i)) = (pD d n i - ρ (pendO n i) % pD d n i) % pD d n i :=
      tauC_of_mem (mem (mem_cons_of_mem _ (mem_cons_of_mem _ (mem_cons_of_mem _
        (mem_cons_of_mem _ mem_cons_self)))))
    have o14 : tauC d n ρ (sl n (14 + 8 * i)) = ρ (sl n (14 + 8 * i)) := tauC_off (notin_pinC
      (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) fun _ _ _ _ => by omega)
    have o15 : tauC d n ρ (sl n (15 + 8 * i)) = ρ (sl n (15 + 8 * i)) := tauC_off (notin_pinC
      (by omega) (by omega) (by omega) (by omega) (by omega) (by omega) fun _ _ _ _ => by omega)
    have op : tauC d n ρ (pend n i) = ρ (pend n i) := tauC_off (pend_notin hi)
    have op' : tauC d n ρ (pendO n i) = ρ (pendO n i) := tauC_off (pendO_notin hi)
    have hcup : ((pD d n i - ρ (pendO n i) % pD d n i) % pD d n i + ρ (pendO n i)) % pD d n i
        = 0 := by
      have hr := Nat.mod_lt (ρ (pendO n i)) hpD
      rw [Nat.add_mod, Nat.mod_mod]
      rcases Nat.eq_zero_or_pos (ρ (pendO n i) % pD d n i) with h0 | h0
      · rw [h0]; simp
      · rw [Nat.mod_eq_of_lt (by omega : pD d n i - ρ (pendO n i) % pD d n i < pD d n i),
          Nat.sub_add_cancel hr.le, Nat.mod_self]
    simp only [pairN, map_cons, map_nil, prod_cons, prod_nil, mul_one, val, zN, fN, cons_append,
      nil_append, eD, eP, dimv]
    rw [t16, t17, t18, t19, t20, o14, o15, op, op',
      tZ3 _ _ _ _ (phase_zero d) (hA i hi), tZ3 _ _ _ _ (phase_zero d) (hB i hi),
      show d ^ (n - i + 1) = pD d n i from rfl,
      tX_cup _ hpD (phase d 0) (phase_zero d), if_pos hcup]
    ring
  simp only [cp, prodVal, map_append, prod_append, map_flatMap, prod_flatMapC]
  rw [hfix, List.map_congr_left hpair, List.prod_map_mul]
  simp only [pinsCp, pairsF]
  ring

/-! ## The family -/

/-- The scalar copy `n` adds: `∏_{i<n} d ^ -(n - i + 1)`. -/
noncomputable def cs (d n : ℕ) : ℂ := ((range n).map fun i => ((d : ℂ) ^ (n - i + 1))⁻¹).prod

/-- The spec's scalar after `n` copies. -/
noncomputable def rs (d : ℕ) : ℕ → ℂ
  | 0 => 1
  | n + 1 => rs d n * cs d n

/-- The spec with its scalar. -/
noncomputable def rhs (d n : ℕ) : ℂ × Net := (rs d n, rhsN n)

theorem base (d : ℕ) (hd : 0 < d) :
    Equal (envAt (E0 d) L 0) (lhs 0).1 (lhs 0).2 (rhs d 0).1 (rhs d 0).2 := by
  refine ⟨by simp [lhs, rhs, lhsN, rhsN, bndL], fun σ _ => ?_⟩
  show 1 * eval _ (lhsN 0) σ = 1 * eval _ (rhsN 0) σ
  rw [eval_rhsN d hd le_rfl σ (fun i h => absurd h (Nat.not_lt_zero i))
    (fun i h => absurd h (Nat.not_lt_zero i))]
  simp [eval, lhsN, sumOver, prodVal, Nat.mod_one]

theorem peel (d n : ℕ) : Equal (envAt (E0 d) L (n + 1)) (lhs (n + 1)).1 (lhs (n + 1)).2 (lhs n).1
    (glueAt (cut n) (cp n) (lhs n).2) := Equal.refl _ _ _

/-- The family theorem from the close step. -/
theorem fam_of_close (d : ℕ) (hd : 0 < d)
    (close : ∀ n, Equal (envAt (E0 d) L (n + 1)) (rhs d n).1
      (glueAt (cut n) (cp n) (rhs d n).2) (rhs d (n + 1)).1 (rhs d (n + 1)).2) (n : ℕ) :
    Equal (envAt (E0 d) L n) (lhs n).1 (lhs n).2 (rhs d n).1 (rhs d n).2 :=
  netIfaceAtE_fam (E0 d) L B cut cp lhs (rhs d) (labsMono d) copyAt (fun n => fitsL n)
    (fun n => fitsR n) (base d hd) (peel d) close n

end Fca

end ArchytasZX
