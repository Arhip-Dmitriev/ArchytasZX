/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFamEnv

/-!
# The Fourier constant adder as a network family

The boxed `fourier_constant_adder` of the engine at `n` copies, copy `i` of weight `d ^ i`, as
networks over `envAt (E0 d) L n`. Copy `n` (the new most significant digit) owns the label block
`[B n, B (n + 1))`: input `0`, output `1`, the wire chain `2 … 9`, the spec's binder and splitter
values `12`, `13`, and per earlier copy `i` the slots `14 + 8 i … 21 + 8 i`. The pair phase
`x y / d ^ (n - i + 1)` of the pair `(n, i)` is an `F` node of dimension `d ^ (n - i + 1)`, its
negative `F³`. Copy `i`'s input and output spiders keep one pending leg (`pend`) for the pairs of
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

/-- The pair nodes of copy `n` with copy `i`: `F` from copy `i`'s input, `F³` from its
output, and the spiders moving its pending legs. -/
def pairN (n i : ℕ) : List Node :=
  [fN (n - i + 2) (sl n (18 + 8 * i)) (sl n (16 + 8 * i)),
   zN [sl n (16 + 8 * i), sl n (14 + 8 * i)] [pend n i],
   fN (n - i + 2) (sl n (19 + 8 * i)) (sl n (20 + 8 * i)),
   fN (n - i + 2) (sl n (20 + 8 * i)) (sl n (21 + 8 * i)),
   fN (n - i + 2) (sl n (21 + 8 * i)) (sl n (17 + 8 * i)),
   zN [sl n (17 + 8 * i), sl n (15 + 8 * i)] [pendO n i]]

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
  int := (range 8).map (fun t => sl n (2 + t)) ++
    (range n).flatMap fun i => (range 6).map fun t => sl n (16 + 8 * i + t)
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
  simp only [cp, labelsOf, pairN, zN, fN, Node.labels, flatMap_append, mem_append, mem_flatMap,
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
  simp only [cp, mem_append, mem_map, mem_range, mem_flatMap] at h
  rcases h with ⟨t, ht, rfl⟩ | ⟨i, hi, t, ht, rfl⟩
  · exact ⟨2 + t, Or.inl (by omega), rfl⟩
  · exact ⟨_, Or.inr ⟨i, hi, t, ht, rfl⟩, rfl⟩

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

end Fca

end ArchytasZX
