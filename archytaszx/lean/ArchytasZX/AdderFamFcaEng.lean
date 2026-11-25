/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFamFca

/-!
# The engine's Fourier constant adder networks

The networks the engine exports for the flat `fourier_constant_adder` at `n` copies and its
binder / shift / splitter spec, with their environment.

* `engL n`: outputs `0 … n - 1`, inputs `n … 2n - 1`, copy `j`'s chain at `2n + 8j`, the pair
  legs from `10n`, the pair phases as `ZP` nodes.
* `engR n`: the spec from `2n² + 8n`.
* `engE d n`: dimension `i < n` is `d ^ (i + 1)`, dimension `n` is `1`; phase `p ∈ [1, n]` is
  `k ↦ 2k / d ^ (n - p + 1)`, phase `n + 1` is `k ↦ -2k / d ^ n`.
* `engL_lit`, `engR_lit`, `engLabs_lit`: equal to the engine's literal exports at `n = 1 … 5`.
* `eval_engR`: the spec as the shift by `2` mod `d ^ n` of the input digit value.
* `eval_engL`: the boxed side as the product over copies of each copy's Fourier sum with its
  `ZP` pair phases, by pinning every chain and pair label.
* `prod_Ce`: that product, copy `k` read as `Fca`'s copy `n - 1 - k`, is `Fca.IndS` (from
  `Fca.arith`).
* `engFam`: `engL n = engR n` over `engE d n` for every `n ≥ 1` and every `d`.
-/

namespace ArchytasZX

namespace FcaEng

open List

/-- `j (j - 1) / 2`. -/
def tri : ℕ → ℕ
  | 0 => 0
  | j + 1 => tri j + j

/-- The number of pairs `(m', j)` with `m' < m`, `m' < j < n`. -/
def off (n : ℕ) : ℕ → ℕ
  | 0 => 0
  | m + 1 => off n m + (n - 1 - m)

/-- Copy `j`'s chain base. -/
def cb (n j : ℕ) : ℕ := 2 * n + 8 * j

/-- Position of the pair `(m, j)`, `m < j`, ordered by `m` then `j`. -/
def i2 (n m j : ℕ) : ℕ := off n m + (j - m - 1)

/-- The `g0` leg of copy `j` to copy `m < j`. -/
def a0 (n j m : ℕ) : ℕ := 10 * n + tri j + m

/-- The `g2` leg of copy `m` to copy `j > m`. -/
def a2 (n m j : ℕ) : ℕ := 10 * n + tri n + i2 n m j

/-- The `g5` leg of copy `m` to copy `j > m`. -/
def a5 (n m j : ℕ) : ℕ := 10 * n + 2 * tri n + i2 n m j

/-- The `g9` leg of copy `j` to copy `m < j`. -/
def a9 (n j m : ℕ) : ℕ := 10 * n + 3 * tri n + tri j + m

/-- A node of dimension and phase index `0`. -/
def nd (k : Kind) (os is : List ℕ) : Node := ⟨k, os, is, 0, 0⟩

/-- Copy `j`'s chain nodes `g1, g4, g6, g7, g8`. -/
def chainN (n j : ℕ) : List Node :=
  [nd .F [cb n j + 1] [cb n j], ⟨.Z, [cb n j + 3], [cb n j + 2], 0, j + 1⟩,
   nd .F [cb n j + 5] [cb n j + 4], nd .F [cb n j + 6] [cb n j + 5],
   nd .F [cb n j + 7] [cb n j + 6]]

/-- The copies `j > m`. -/
def later (n m : ℕ) : List ℕ := (range (n - m - 1)).map (m + 1 + ·)

/-- The engine's boxed side at `n` copies. -/
def engL (n : ℕ) : Net where
  nodes :=
    (range n).flatMap (chainN n) ++
    (range n).map (fun j => nd .Z [cb n j] ((n + j) :: (range j).map (a0 n j))) ++
    (range n).map (fun j => nd .Z ((cb n j + 2) :: (later n j).map (a2 n j)) [cb n j + 1]) ++
    (range n).map (fun j => nd .Z ((cb n j + 4) :: (later n j).map (a5 n j)) [cb n j + 3]) ++
    (range n).map (fun j => nd .Z [j] ((cb n j + 7) :: (range j).map (a9 n j))) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 1, [a0 n j m], [a2 n m j], j - m, 0⟩) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 (-1), [a9 n j m], [a5 n m j], j - m, 0⟩)
  int := (range (8 * n + 4 * tri n)).map (2 * n + ·)
  bnd := range (2 * n)

/-- Start of the spec's labels. -/
def rb (n : ℕ) : ℕ := 2 * n * n + 8 * n

/-- The dimension index of `d ^ t`. -/
def dI (n t : ℕ) : ℕ := if t = 0 then n else t - 1

/-- The output of binder `t`. -/
def bout (n t : ℕ) : ℕ := if t + 1 = n then rb n + 1 else rb n + 4 + t

/-- The input of binder `t`. -/
def bin (n t : ℕ) : ℕ := if t = 0 then rb n else rb n + 3 + t

/-- The splitter value of the top `t` digits. -/
def sv (n t : ℕ) : ℕ := if t = 0 then rb n + 3 else if t = n then rb n + 2 else rb n + n + 2 + t

/-- The binder chain. -/
def bN (n : ℕ) : List Node := (range n).map fun t => ⟨.B, [bout n t], [bin n t, n + t], dI n t, 0⟩

/-- The splitter chain. -/
def sN (n : ℕ) : List Node := (range n).map fun t => ⟨.S, [sv n t, t], [sv n (t + 1)], dI n t, 0⟩

/-- The binder head. -/
def hN (n : ℕ) : Node := ⟨.Z, [rb n], [], n, 0⟩

/-- The shift by `2` mod `d ^ n`. -/
def xN (n : ℕ) : Node := ⟨.X, [rb n + 2], [rb n + 1], n - 1, n + 1⟩

/-- The splitter end. -/
def eN (n : ℕ) : Node := ⟨.Z, [], [rb n + 3], n, 0⟩

/-- The engine's spec at `n` digits. -/
def engR (n : ℕ) : Net where
  nodes := if n = 1 then [hN n] ++ bN n ++ [xN n] ++ sN n ++ [eN n]
    else [hN n, xN n, eN n] ++ bN n ++ sN n
  int := (range (2 * n + 2)).map (rb n + ·)
  bnd := range (2 * n)

/-- The exponent of `d` in the dimension of the spec's label `rb n + k`. -/
def edim (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = 3 then 0 else if k ≤ 2 then n else if k < n + 3 then k - 3 else k - n - 2

/-- The spec's label dimensions. -/
def engLabs (n : ℕ) : List (ℕ × ℕ) := (range (2 * n + 2)).map fun k => (rb n + k, dI n (edim n k))

/-- The engine's dimensions. -/
def engD (d n i : ℕ) : ℕ := if i < n then d ^ (i + 1) else 1

/-- The engine's phases. -/
noncomputable def engP (d n p k : ℕ) : ℝ :=
  if p = 0 then 0
  else if p ≤ n then 2 * k / (d : ℝ) ^ (n - p + 1)
  else -(2 * k) / (d : ℝ) ^ n

/-- The engine's environment at `n`. -/
noncomputable def engE (d n : ℕ) : Env := { dimv := engD d n, phase := engP d n, labs := engLabs n }

/-! ## The boxed side's labels -/

theorem tri_mono {a b : ℕ} (h : a ≤ b) : tri a ≤ tri b := by
  induction b with
  | zero => rw [Nat.le_zero.1 h]
  | succ b ih =>
    rcases Nat.lt_or_eq_of_le h with h | rfl
    · have := ih (by omega); simp only [tri]; omega
    · rfl

theorem tri_lt {j m : ℕ} (hm : m < j) : tri j + m < tri (j + 1) := by simp only [tri]; omega

/-- The copy `j` of the pair position `s`, ordered by `j` then `m`. -/
def jOf (s : ℕ) : ℕ := Nat.find (p := fun j => s < tri (j + 1)) ⟨s + 1, by
  have := tri_mono (Nat.le_succ (s + 1)); simp only [tri] at this ⊢; omega⟩

theorem jOf_spec (s : ℕ) : tri (jOf s) ≤ s ∧ s < tri (jOf s + 1) := by
  refine ⟨?_, Nat.find_spec (p := fun j => s < tri (j + 1)) _⟩
  rcases Nat.eq_zero_or_pos (jOf s) with h | h
  · rw [h]; exact Nat.zero_le _
  · have := Nat.find_min (p := fun j => s < tri (j + 1)) _ (show jOf s - 1 < jOf s by omega)
    rw [show jOf s - 1 + 1 = jOf s by omega] at this; omega

theorem jOf_tri {j m : ℕ} (hm : m < j) : jOf (tri j + m) = j := by
  have ⟨h1, h2⟩ := jOf_spec (tri j + m)
  rcases lt_trichotomy (jOf (tri j + m)) j with h | h | h
  · have := tri_mono (show jOf (tri j + m) + 1 ≤ j by omega); omega
  · exact h
  · have := tri_mono (show j + 1 ≤ jOf (tri j + m) by omega); have := tri_lt hm; omega

theorem jOf_lt {n s : ℕ} (hs : s < tri n) : jOf s < n := by
  by_contra h
  have := tri_mono (show n ≤ jOf s by omega); have := (jOf_spec s).1; omega

theorem off_mono (n : ℕ) {a b : ℕ} (h : a ≤ b) : off n a ≤ off n b := by
  induction b with
  | zero => rw [Nat.le_zero.1 h]
  | succ b ih =>
    rcases Nat.lt_or_eq_of_le h with h | rfl
    · have := ih (by omega); simp only [off]; omega
    · rfl

theorem off_tri (n : ℕ) : ∀ m ≤ n, off n m + tri (n - m) = tri n
  | 0, _ => by simp [off]
  | m + 1, h => by
    have := off_tri n m (by omega)
    rw [show n - m = n - (m + 1) + 1 by omega] at this
    simp only [off, tri] at this ⊢; omega

/-- The copy `m` of the pair position `s`, ordered by `m` then `j`. -/
def mOf (n s : ℕ) : ℕ := Nat.find (p := fun m => s < off n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩

theorem mOf_spec {n s : ℕ} (hs : s < tri n) :
    mOf n s < n ∧ off n (mOf n s) ≤ s ∧ s < off n (mOf n s + 1) := by
  have hP := Nat.find_spec (p := fun m => s < off n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩
  have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp [tri] at hs)
  have hlt : mOf n s < n := by
    by_contra h
    have := Nat.find_min (p := fun m => s < off n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩
      (show n - 1 < mOf n s by omega)
    have e := off_tri n n le_rfl
    rw [Nat.sub_self, show tri 0 = 0 from rfl, add_zero] at e
    rw [show n - 1 + 1 = n by omega, e] at this
    exact this (Or.inl hs)
  refine ⟨hlt, ?_, hP.resolve_right (by change ¬mOf n s = n; omega)⟩
  rcases Nat.eq_zero_or_pos (mOf n s) with h | h
  · rw [h]; exact Nat.zero_le _
  · have := Nat.find_min (p := fun m => s < off n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩
      (show mOf n s - 1 < mOf n s by omega)
    change ¬(s < off n (mOf n s - 1 + 1) ∨ mOf n s - 1 = n) at this
    rw [show mOf n s - 1 + 1 = mOf n s by omega] at this; omega

theorem mOf_off {n m j : ℕ} (hm : m < j) (hj : j < n) : mOf n (i2 n m j) = m := by
  have hs : i2 n m j < tri n := by
    have := off_mono n (show m + 1 ≤ n by omega); have := off_tri n n le_rfl
    simp only [i2, off, Nat.sub_self, tri, add_zero] at *; omega
  obtain ⟨_, h1, h2⟩ := mOf_spec hs
  have hb : i2 n m j < off n (m + 1) := by simp only [i2, off]; omega
  rcases lt_trichotomy (mOf n (i2 n m j)) m with h | h | h
  · have := off_mono n (show mOf n (i2 n m j) + 1 ≤ m by omega); simp only [i2] at *; omega
  · exact h
  · have := off_mono n (show m + 1 ≤ mOf n (i2 n m j) by omega); omega

/-- The free labels of the boxed side: each copy's Fourier value and `F³` legs. -/
def frees (n : ℕ) : List ℕ := (range n).flatMap fun m => [1, 5, 6].map fun r => 2 * n + 8 * m + r

/-- The pinned chain labels of the boxed side. -/
def chainP (n : ℕ) : List ℕ := (range n).flatMap fun m =>
  [0, 2, 3, 4, 7].map fun r => 2 * n + 8 * m + r

/-- The pair labels of the boxed side. -/
def pairP (n : ℕ) : List ℕ := (range (4 * tri n)).map (10 * n + ·)

theorem frees_nodup (n : ℕ) : (frees n).Nodup := by
  have := Fca.nodup_blocks n (2 * n) [] [1, 5, 6] (by simp) (by decide) (by simp) (by simp)
  simpa [frees] using this

theorem pins_nodup (n : ℕ) : (chainP n ++ pairP n).Nodup := by
  have h1 := Fca.nodup_blocks n (2 * n) [] [0, 2, 3, 4, 7] (by simp) (by decide) (by simp)
    (by simp)
  have h2 : (pairP n).Nodup := (nodup_range).map fun a b h => by simpa using h
  rw [nodup_append]
  refine ⟨by simpa [chainP] using h1, h2, ?_⟩
  intro a ha b hb e
  simp only [chainP, pairP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil,
    or_false] at ha hb
  obtain ⟨m, hm, r, hr, rfl⟩ := ha
  obtain ⟨k, _, rfl⟩ := hb
  rcases hr with rfl | rfl | rfl | rfl | rfl <;> omega

theorem int_perm (n : ℕ) : (engL n).int.Perm (frees n ++ (chainP n ++ pairP n)) := by
  have hnd : (frees n ++ (chainP n ++ pairP n)).Nodup := by
    rw [nodup_append]
    refine ⟨frees_nodup n, pins_nodup n, ?_⟩
    intro a ha b hb e
    simp only [frees, chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons,
      not_mem_nil, or_false] at ha hb
    obtain ⟨m, hm, r, hr, rfl⟩ := ha
    rcases hb with ⟨k, hk, s, hs, rfl⟩ | ⟨k, _, rfl⟩
    · rcases hr with rfl | rfl | rfl <;> rcases hs with rfl | rfl | rfl | rfl | rfl <;> omega
    · rcases hr with rfl | rfl | rfl <;> omega
  refine (List.perm_ext_iff_of_nodup ((nodup_range).map fun a b h => by simpa using h) hnd).2
    fun x => ?_
  simp only [frees, chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons,
    not_mem_nil, or_false]
  constructor
  · rintro ⟨k, hk, rfl⟩
    by_cases h : k < 8 * n
    · have h8 : k % 8 < 8 := Nat.mod_lt _ (by norm_num)
      have e : 2 * n + k = 2 * n + 8 * (k / 8) + k % 8 := by omega
      have hm : k / 8 < n := by omega
      interval_cases hr : k % 8
      · exact Or.inr (Or.inl ⟨k / 8, hm, 0, Or.inl rfl, e.symm⟩)
      · exact Or.inl ⟨k / 8, hm, 1, Or.inl rfl, e.symm⟩
      · exact Or.inr (Or.inl ⟨k / 8, hm, 2, Or.inr (Or.inl rfl), e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 8, hm, 3, Or.inr (Or.inr (Or.inl rfl)), e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 8, hm, 4, Or.inr (Or.inr (Or.inr (Or.inl rfl))), e.symm⟩)
      · exact Or.inl ⟨k / 8, hm, 5, Or.inr (Or.inl rfl), e.symm⟩
      · exact Or.inl ⟨k / 8, hm, 6, Or.inr (Or.inr rfl), e.symm⟩
      · exact Or.inr (Or.inl ⟨k / 8, hm, 7, Or.inr (Or.inr (Or.inr (Or.inr rfl))), e.symm⟩)
    · exact Or.inr (Or.inr ⟨k - 8 * n, by omega, by omega⟩)
  · rintro (⟨m, hm, r, hr, rfl⟩ | ⟨m, hm, r, hr, rfl⟩ | ⟨k, hk, rfl⟩)
    · exact ⟨8 * m + r, by rcases hr with rfl | rfl | rfl <;> omega, by omega⟩
    · exact ⟨8 * m + r, by rcases hr with rfl | rfl | rfl | rfl | rfl <;> omega, by omega⟩
    · exact ⟨8 * n + k, by omega, by omega⟩


/-! ## The boxed side's pinned labels -/

theorem a0_lt {n j m : ℕ} (hm : m < j) (hj : j < n) : tri j + m < tri n :=
  lt_of_lt_of_le (tri_lt hm) (tri_mono hj)

theorem i2_lt {n m j : ℕ} (hm : m < j) (hj : j < n) : i2 n m j < tri n := by
  have := off_mono n (show m + 1 ≤ n by omega); have := off_tri n n le_rfl
  simp only [i2, off, Nat.sub_self, tri, add_zero] at *; omega

theorem blk {T g s : ℕ} (hs : s < T) : (T * g + s) / T = g ∧ (T * g + s) % T = s := by
  have hT : 0 < T := by omega
  rw [Nat.mul_add_div hT, Nat.mul_add_mod, Nat.div_eq_of_lt hs, Nat.mod_eq_of_lt hs, add_zero]
  exact ⟨rfl, rfl⟩

/-- The pinned value of each pinned label of the boxed side, the free and boundary labels from
`τ`. -/
def tz (n : ℕ) (τ : ℕ → ℕ) (x : ℕ) : ℕ :=
  if 2 * n ≤ x ∧ x < 10 * n then
    if (x - 2 * n) % 8 = 0 then τ (n + (x - 2 * n) / 8)
    else if (x - 2 * n) % 8 = 7 then τ ((x - 2 * n) / 8)
    else if (x - 2 * n) % 8 = 2 ∨ (x - 2 * n) % 8 = 3 ∨ (x - 2 * n) % 8 = 4 then
      τ (2 * n + 8 * ((x - 2 * n) / 8) + 1)
    else τ x
  else if 10 * n ≤ x ∧ x < 10 * n + 4 * tri n then
    if (x - 10 * n) / tri n = 0 then τ (n + jOf ((x - 10 * n) % tri n))
    else if (x - 10 * n) / tri n = 3 then τ (jOf ((x - 10 * n) % tri n))
    else τ (2 * n + 8 * mOf n ((x - 10 * n) % tri n) + 1)
  else τ x

theorem tz_chain (n : ℕ) (τ : ℕ → ℕ) {m r : ℕ} (hm : m < n) (hr : r < 8) :
    tz n τ (2 * n + 8 * m + r) = if r = 0 then τ (n + m) else if r = 7 then τ m
      else if r = 2 ∨ r = 3 ∨ r = 4 then τ (2 * n + 8 * m + 1) else τ (2 * n + 8 * m + r) := by
  have e1 : (2 * n + 8 * m + r - 2 * n) % 8 = r := by omega
  have e2 : (2 * n + 8 * m + r - 2 * n) / 8 = m := by omega
  unfold tz
  rw [if_pos (by omega), e1, e2]

theorem tz_pair (n : ℕ) (τ : ℕ → ℕ) {g s : ℕ} (hg : g < 4) (hs : s < tri n) :
    tz n τ (10 * n + (tri n * g + s)) = if g = 0 then τ (n + jOf s) else if g = 3 then τ (jOf s)
      else τ (2 * n + 8 * mOf n s + 1) := by
  have hT : tri n * g + s < 4 * tri n := by nlinarith
  obtain ⟨e1, e2⟩ := blk (g := g) hs
  unfold tz
  rw [if_neg (by omega), if_pos ⟨by omega, by omega⟩, show 10 * n + (tri n * g + s) - 10 * n =
    tri n * g + s by omega, e1, e2]

theorem mem_g0 {n j : ℕ} (hj : j < n) :
    nd .Z [cb n j] ((n + j) :: (range j).map (a0 n j)) ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩))))))

theorem mem_g2 {n j : ℕ} (hj : j < n) :
    nd .Z ((cb n j + 2) :: (later n j).map (a2 n j)) [cb n j + 1] ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩)))))

theorem mem_g5 {n j : ℕ} (hj : j < n) :
    nd .Z ((cb n j + 4) :: (later n j).map (a5 n j)) [cb n j + 3] ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inr (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩))))

theorem mem_g9 {n j : ℕ} (hj : j < n) :
    nd .Z [j] ((cb n j + 7) :: (range j).map (a9 n j)) ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inr (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩)))

theorem mem_chain {n j : ℕ} (hj : j < n) {x : Node} (hx : x ∈ chainN n j) : x ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (mem_flatMap.2 ⟨j, mem_range.2 hj, hx⟩))))))

theorem mem_later {n m j : ℕ} : j ∈ later n m ↔ m < j ∧ j < n := by
  simp only [later, mem_map, mem_range]
  constructor
  · rintro ⟨t, ht, rfl⟩; omega
  · intro h; exact ⟨j - m - 1, by omega, by omega⟩

theorem free_notin {n m r : ℕ} (hm : m < n) (hr : r = 1 ∨ r = 5 ∨ r = 6) :
    2 * n + 8 * m + r ∉ chainP n ++ pairP n := by
  simp only [chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil,
    or_false, not_or, not_exists, not_and]
  refine ⟨fun k _ s hs => ?_, fun k _ => by omega⟩
  rcases hs with rfl | rfl | rfl | rfl | rfl <;> omega

/-- The boxed side's nodes are nonzero only at the pinned values. -/
theorem engL_support (d : ℕ) {n : ℕ} {τ τ' : ℕ → ℕ}
    (h : prodVal (engE d n) (engL n).nodes τ' ≠ 0) (hτ : ∀ x ∉ chainP n ++ pairP n, τ' x = τ x) :
    ∀ l ∈ chainP n ++ pairP n, τ' l = tz n τ l := by
  have hv := Fca.prodVal_ne h
  have zq : ∀ {os is : List ℕ}, nd .Z os is ∈ (engL n).nodes → ∀ {a b : ℕ}, a ∈ os ++ is →
      b ∈ os ++ is → τ' a = τ' b := fun {os is} hn {a b} ha hb =>
    Z_eq _ _ _ rfl (hv _ hn) (a := a) (b := b) ha hb
  have bnd : ∀ x, x < 2 * n → τ' x = τ x := fun x hx => hτ x fun h => by
    simp only [chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons,
      not_mem_nil, or_false] at h
    rcases h with ⟨k, _, s, hs, rfl⟩ | ⟨k, _, rfl⟩
    · rcases hs with rfl | rfl | rfl | rfl | rfl <;> omega
    · omega
  have c1 : ∀ m < n, τ' (cb n m + 1) = τ (2 * n + 8 * m + 1) := fun m hm =>
    hτ _ (free_notin hm (Or.inl rfl))
  have c2 : ∀ m < n, τ' (cb n m + 2) = τ (2 * n + 8 * m + 1) := fun m hm => by
    rw [zq (mem_g2 hm) (a := cb n m + 2) (b := cb n m + 1) (mem_append_left _ mem_cons_self)
      (mem_append_right _ (mem_singleton_self _)), c1 m hm]
  have c3 : ∀ m < n, τ' (cb n m + 3) = τ (2 * n + 8 * m + 1) := fun m hm => by
    have hz : (⟨.Z, [cb n m + 3], [cb n m + 2], 0, m + 1⟩ : Node) ∈ (engL n).nodes :=
      mem_chain hm (by simp [chainN])
    rw [Z_eq _ _ _ rfl (hv _ hz) (a := cb n m + 3) (b := cb n m + 2) (by simp [Node.labels])
      (by simp [Node.labels]), c2 m hm]
  have c4 : ∀ m < n, τ' (cb n m + 4) = τ (2 * n + 8 * m + 1) := fun m hm => by
    rw [zq (mem_g5 hm) (a := cb n m + 4) (b := cb n m + 3) (mem_append_left _ mem_cons_self)
      (mem_append_right _ (mem_singleton_self _)), c3 m hm]
  intro l hl
  rcases mem_append.1 hl with hc | hp
  · simp only [chainP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil, or_false] at hc
    obtain ⟨m, hm, r, hr, rfl⟩ := hc
    rw [tz_chain n τ hm (by rcases hr with rfl | rfl | rfl | rfl | rfl <;> omega)]
    rcases hr with rfl | rfl | rfl | rfl | rfl
    · simp only [if_true]
      rw [zq (mem_g0 hm) (a := 2 * n + 8 * m + 0) (b := n + m) (by simp [cb]) (by simp),
        bnd _ (by omega)]
    · simpa [cb] using c2 m hm
    · simpa [cb] using c3 m hm
    · simpa [cb] using c4 m hm
    · simp only [show (7 : ℕ) ≠ 0 by omega, if_false, if_true]
      rw [zq (mem_g9 hm) (a := 2 * n + 8 * m + 7) (b := m) (by simp [cb]) (by simp),
        bnd _ (by omega)]
  · simp only [pairP, mem_map, mem_range] at hp
    obtain ⟨k, hk, rfl⟩ := hp
    have hT : 0 < tri n := by
      rcases Nat.eq_zero_or_pos (tri n) with h0 | h0
      · rw [h0] at hk; omega
      · exact h0
    set s := k % tri n with hs
    have hs' : s < tri n := Nat.mod_lt _ hT
    have hg : k / tri n < 4 := by
      rw [Nat.div_lt_iff_lt_mul hT]; linarith
    have ek : k = tri n * (k / tri n) + s := (Nat.div_add_mod k (tri n)).symm
    rw [ek, tz_pair n τ hg hs']
    obtain ⟨hj1, hj2⟩ := jOf_spec s
    have hjn := jOf_lt hs'
    obtain ⟨hm1, hm2, hm3⟩ := mOf_spec hs'
    have hm3' : s < off n (mOf n s) + (n - 1 - mOf n s) := hm3
    set j := jOf s
    set m := mOf n s
    have e0 : s = tri j + (s - tri j) := by omega
    have e2 : s = i2 n m (m + 1 + (s - off n m)) := by simp only [i2]; omega
    have hjl : m + 1 + (s - off n m) ∈ later n m := mem_later.2 ⟨by omega, by omega⟩
    have hlt : s - tri j < j := by simp only [tri] at hj2; omega
    interval_cases hgv : k / tri n
    · simp only [if_true]
      have hl : 10 * n + (tri n * 0 + s) = a0 n j (s - tri j) := by
        simp only [a0]; omega
      rw [hl, zq (mem_g0 hjn) (a := a0 n j (s - tri j)) (b := n + j)
        (mem_append_right _ (mem_cons_of_mem _ (mem_map.2 ⟨_, mem_range.2 hlt, rfl⟩)))
        (mem_append_right _ mem_cons_self), bnd _ (by omega)]
    · simp only [show (1 : ℕ) ≠ 0 by omega, show (1 : ℕ) ≠ 3 by omega, if_false]
      have hl : 10 * n + (tri n * 1 + s) = a2 n m (m + 1 + (s - off n m)) := by
        simp only [a2]; omega
      rw [hl, zq (mem_g2 hm1) (a := a2 n m (m + 1 + (s - off n m))) (b := cb n m + 1)
        (mem_append_left _ (mem_cons_of_mem _ (mem_map.2 ⟨_, hjl, rfl⟩)))
        (mem_append_right _ (mem_singleton_self _)), c1 m hm1]
    · simp only [show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 3 by omega, if_false]
      have hl : 10 * n + (tri n * 2 + s) = a5 n m (m + 1 + (s - off n m)) := by
        simp only [a5]; omega
      rw [hl, zq (mem_g5 hm1) (a := a5 n m (m + 1 + (s - off n m))) (b := cb n m + 3)
        (mem_append_left _ (mem_cons_of_mem _ (mem_map.2 ⟨_, hjl, rfl⟩)))
        (mem_append_right _ (mem_singleton_self _)), c3 m hm1]
    · simp only [show (3 : ℕ) ≠ 0 by omega, if_false, if_true]
      have hl : 10 * n + (tri n * 3 + s) = a9 n j (s - tri j) := by
        simp only [a9]; omega
      rw [hl, zq (mem_g9 hjn) (a := a9 n j (s - tri j)) (b := j)
        (mem_append_right _ (mem_cons_of_mem _ (mem_map.2 ⟨_, mem_range.2 hlt, rfl⟩)))
        (mem_append_left _ (mem_singleton_self _)), bnd _ (by omega)]


/-! ## The spec in closed form -/

theorem engD_dI (d : ℕ) {n t : ℕ} (h : t ≤ n) : engD d n (dI n t) = d ^ t := by
  unfold engD dI
  split_ifs with h0 h1 h1
  · omega
  · subst h0; rfl
  · congr 1; omega
  · omega

theorem edim_le {n k : ℕ} (hk : k < 2 * n + 2) : edim n k ≤ n := by
  unfold edim; split_ifs <;> omega

theorem lookup_shift (g : ℕ → ℕ) (N k b : ℕ) (hk : k < N) :
    ((range N).map fun j => (b + j, g j)).lookup (b + k) = some (g k) := by
  refine Fca.lookup_of_mem _ ?_ (mem_map.2 ⟨k, mem_range.2 hk, rfl⟩)
  rw [map_map]
  exact (nodup_range).map fun x y h => by simpa using h

theorem labDim_eng (d : ℕ) {n k : ℕ} (hk : k < 2 * n + 2) :
    labDim (engE d n) (rb n + k) = d ^ edim n k := by
  unfold labDim
  rw [show (engE d n).labs = engLabs n from rfl, engLabs,
    lookup_shift (fun k => dI n (edim n k)) _ _ _ hk]
  exact engD_dI d (edim_le hk)

/-- The value of the first `t` digits from label `c`, the first most significant. -/
def etv (d : ℕ) (σ : ℕ → ℕ) (c : ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => etv d σ c t * d + σ (c + t)

theorem etv_lt {d : ℕ} {σ : ℕ → ℕ} {c n : ℕ} (h : ∀ t < n, σ (c + t) < d) :
    ∀ t ≤ n, etv d σ c t < d ^ t
  | 0, _ => by simp [etv]
  | t + 1, ht => by
    have ih := etv_lt h t (by omega)
    have hdig := h t (by omega)
    simp only [etv, pow_succ]
    nlinarith

/-- The pinned value of the spec's label `rb n + k`. -/
def pv (d : ℕ) (σ : ℕ → ℕ) (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = 3 then 0 else if k = 1 then etv d σ n n else if k = 2 then etv d σ 0 n
  else if k < n + 3 then etv d σ n (k - 3) else etv d σ 0 (k - n - 2)

/-- `σ` with the spec's summed labels at their pinned values. -/
def tauE (d : ℕ) (σ : ℕ → ℕ) (n x : ℕ) : ℕ :=
  if rb n ≤ x ∧ x < rb n + (2 * n + 2) then pv d σ n (x - rb n) else σ x

theorem mem_engR_int {n x : ℕ} : x ∈ (engR n).int ↔ rb n ≤ x ∧ x < rb n + (2 * n + 2) := by
  simp only [engR, mem_map, mem_range]
  constructor
  · rintro ⟨k, hk, rfl⟩; omega
  · intro h; exact ⟨x - rb n, by omega, by omega⟩

theorem engR_int_nodup (n : ℕ) : (engR n).int.Nodup :=
  (nodup_range).map fun x y h => by simpa using h

theorem tauE_rb (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : tauE d σ n (rb n) = 0 := by
  simp [tauE, pv]

theorem tauE_bin (d : ℕ) (σ : ℕ → ℕ) {n t : ℕ} (ht : t < n) :
    tauE d σ n (bin n t) = etv d σ n t := by
  unfold tauE bin pv
  split_ifs <;> first | omega | (subst_vars; rfl) | (congr 1; omega)

theorem tauE_bout (d : ℕ) (σ : ℕ → ℕ) {n t : ℕ} (ht : t < n) :
    tauE d σ n (bout n t) = etv d σ n (t + 1) := by
  unfold tauE bout pv
  split_ifs <;> first | omega | (congr 1; omega)

theorem tauE_sv (d : ℕ) (σ : ℕ → ℕ) {n t : ℕ} (hn : 0 < n) (ht : t ≤ n) :
    tauE d σ n (sv n t) = etv d σ 0 t := by
  unfold tauE sv pv
  split_ifs <;> first | omega | (subst_vars; rfl) | (congr 1; omega)

theorem pv_lt {d : ℕ} {σ : ℕ → ℕ} {n k : ℕ} (hk : k < 2 * n + 2)
    (ha : ∀ t < n, σ (n + t) < d) (hb : ∀ t < n, σ (0 + t) < d) :
    pv d σ n k < d ^ edim n k := by
  unfold pv edim
  split_ifs <;> first
    | simp
    | exact etv_lt ha _ (by omega)
    | exact etv_lt hb _ (by omega)
    | omega

theorem mem_engR {n : ℕ} {x : Node} :
    x ∈ (engR n).nodes ↔ x = hN n ∨ x ∈ bN n ∨ x = xN n ∨ x ∈ sN n ∨ x = eN n := by
  unfold engR
  split_ifs <;> simp only [mem_append, mem_cons, not_mem_nil, or_false] <;> tauto

theorem prodVal_engR (E : Env) (n : ℕ) (τ : ℕ → ℕ) :
    prodVal E (engR n).nodes τ = val E τ (hN n) * ((bN n).map (val E τ)).prod * val E τ (xN n) *
      ((sN n).map (val E τ)).prod * val E τ (eN n) := by
  unfold engR prodVal
  split_ifs <;> simp only [map_append, prod_append, map_cons, map_nil, prod_cons, prod_nil] <;>
    ring

theorem engP_zero (d n : ℕ) : engP d n 0 = fun _ => 0 := by funext k; simp [engP]

theorem engD_zero (d : ℕ) {n : ℕ} (hn : 0 < n) : engD d n 0 = d := by simp [engD, hn]

theorem engR_support (d : ℕ) {n : ℕ} (hn : 0 < n) {τ : ℕ → ℕ}
    (h : prodVal (engE d n) (engR n).nodes τ ≠ 0) :
    τ (rb n) = 0 ∧ (∀ t < n, τ (bout n t) = etv d τ n (t + 1)) ∧
      ∀ t ≤ n, τ (sv n t) = etv d τ 0 t := by
  have hv := Fca.prodVal_ne h
  have h0 : τ (rb n) = 0 := by
    have h1 := hv _ (mem_engR.2 (Or.inl rfl))
    simp only [hN, val, map_cons, map_nil, append_nil] at h1
    rw [show (engE d n).dimv n = 1 by simp [engE, engD], show (engE d n).phase 0 = fun _ => 0
      from engP_zero d n, Fca.tZ_one _ _ (fun _ => rfl)] at h1
    simpa using h1
  have hB : ∀ t < n, τ (bout n t) = τ (bin n t) * d + τ (n + t) := by
    intro t ht
    have h1 := hv _ (mem_engR.2 (Or.inr (Or.inl (mem_map.2 ⟨t, mem_range.2 ht, rfl⟩))))
    simp only [val, map_cons, map_nil] at h1
    have e := Fca.tB_eq h1
    rwa [show (engE d n).dimv 0 = d from engD_zero d hn] at e
  have hS : ∀ t < n, τ (sv n (t + 1)) = τ (sv n t) * d + τ t := by
    intro t ht
    have h1 := hv _ (mem_engR.2 (Or.inr (Or.inr (Or.inr (Or.inl
      (mem_map.2 ⟨t, mem_range.2 ht, rfl⟩))))))
    simp only [val, map_cons, map_nil] at h1
    have e := Fca.tB_eq h1
    rwa [show (engE d n).dimv 0 = d from engD_zero d hn] at e
  have hE : τ (sv n 0) = 0 := by
    have h1 := hv _ (mem_engR.2 (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))
    simp only [eN, val, map_cons, map_nil, nil_append] at h1
    rw [show (engE d n).dimv n = 1 by simp [engE, engD], show (engE d n).phase 0 = fun _ => 0
      from engP_zero d n, Fca.tZ_one _ _ (fun _ => rfl)] at h1
    simpa [sv] using h1
  refine ⟨h0, fun t ht => ?_, fun t ht => ?_⟩
  · induction t with
    | zero => rw [hB 0 ht]; simp [etv, bin, h0]
    | succ t ih =>
      have e : bin n (t + 1) = bout n t := by
        simp only [bin, bout]; rw [if_neg (by omega), if_neg (by omega)]; omega
      rw [hB _ ht, e, ih (by omega)]
      rfl
  · induction t with
    | zero => simpa [etv] using hE
    | succ t ih => rw [hS t (by omega), ih (by omega)]; simp [etv]

theorem engP_shift (d n : ℕ) :
    engP d n (n + 1) = fun k : ℕ => -(2 * (k : ℝ)) / ((d ^ n : ℕ) : ℝ) := by
  funext k; simp [engP]


/-- The spec at `n ≥ 1` digits: the shift by `2` mod `d ^ n` of the input digits. -/
theorem eval_engR (d : ℕ) (hd : 0 < d) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ)
    (ha : ∀ t < n, σ (n + t) < d) (hb : ∀ t < n, σ t < d) :
    eval (engE d n) (engR n) σ =
      if etv d σ 0 n % d ^ n = (etv d σ n n + 2) % d ^ n then 1 else 0 := by
  have hb' : ∀ t < n, σ (0 + t) < d := fun t ht => by simpa using hb t ht
  have hrb : 2 * n ≤ rb n := by unfold rb; nlinarith
  have hoff : ∀ x, x < 2 * n → tauE d σ n x = σ x := fun x hx => by
    unfold tauE; rw [if_neg (by omega)]
  unfold eval
  rw [Fca.sumOver_single (labDim (engE d n)) _ (engR_int_nodup n) _ σ (tauE d σ n) ?h0 ?hr ?hs]
  case h0 =>
    intro x hx
    unfold tauE
    rw [if_neg fun h => hx (mem_engR_int.2 h)]
  case hr =>
    intro l hl
    obtain ⟨h1, h2⟩ := mem_engR_int.1 hl
    obtain ⟨k, rfl⟩ : ∃ k, l = rb n + k := ⟨l - rb n, by omega⟩
    rw [labDim_eng d (by omega)]
    unfold tauE
    rw [if_pos (by omega), show rb n + k - rb n = k by omega]
    exact pv_lt (by omega) ha hb'
  case hs =>
    intro τ hτ _ hne l hl
    obtain ⟨s0, s1, s2⟩ := engR_support d hn hne
    have dig : ∀ t, t < 2 * n → τ t = σ t := fun t ht =>
      hτ t fun h => by have := (mem_engR_int.1 h).1; omega
    have eA : ∀ t ≤ n, etv d τ n t = etv d σ n t := by
      intro t ht
      induction t with
      | zero => rfl
      | succ t ih => simp only [etv, ih (by omega), dig (n + t) (by omega)]
    have eB : ∀ t ≤ n, etv d τ 0 t = etv d σ 0 t := by
      intro t ht
      induction t with
      | zero => rfl
      | succ t ih => simp only [etv, ih (by omega), dig (0 + t) (by omega)]
    obtain ⟨h1, h2⟩ := mem_engR_int.1 hl
    obtain ⟨k, rfl⟩ : ∃ k, l = rb n + k := ⟨l - rb n, by omega⟩
    rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ (4 ≤ k ∧ k < n + 3) ∨ n + 3 ≤ k by omega) with
      rfl | rfl | rfl | rfl | hk | hk
    · rw [Nat.add_zero, s0, tauE_rb]
    · have e : rb n + 1 = bout n (n - 1) := by unfold bout; rw [if_pos (by omega)]
      rw [e, s1 _ (by omega), tauE_bout d σ (by omega), eA _ (by omega)]
    · have e : rb n + 2 = sv n n := by unfold sv; split_ifs <;> omega
      rw [e, s2 _ le_rfl, tauE_sv d σ hn le_rfl, eB _ le_rfl]
    · have e : rb n + 3 = sv n 0 := by unfold sv; split_ifs <;> omega
      rw [e, s2 _ (Nat.zero_le _), tauE_sv d σ hn (Nat.zero_le _), eB _ (Nat.zero_le _)]
    · have e : rb n + k = bout n (k - 4) := by unfold bout; split_ifs <;> omega
      rw [e, s1 _ (by omega), tauE_bout d σ (by omega), eA _ (by omega)]
    · have e : rb n + k = sv n (k - n - 2) := by unfold sv; split_ifs <;> omega
      rw [e, s2 _ (by omega), tauE_sv d σ hn (by omega), eB _ (by omega)]
  have eD1 : (engE d n).dimv n = 1 := by simp [engE, engD]
  have eP0 : (engE d n).phase 0 = fun _ => 0 := engP_zero d n
  have eD0 : (engE d n).dimv 0 = d := engD_zero d hn
  have hH : val (engE d n) (tauE d σ n) (hN n) = 1 := by
    simp only [hN, val, map_cons, map_nil, append_nil]
    rw [eD1, eP0, Fca.tZ_one _ _ (fun _ => rfl), tauE_rb]; simp
  have hE : val (engE d n) (tauE d σ n) (eN n) = 1 := by
    simp only [eN, val, map_cons, map_nil, nil_append]
    rw [eD1, eP0, Fca.tZ_one _ _ (fun _ => rfl)]
    have e : rb n + 3 = sv n 0 := by unfold sv; split_ifs <;> omega
    rw [e, tauE_sv d σ hn (Nat.zero_le _)]; simp [etv]
  have hX : val (engE d n) (tauE d σ n) (xN n) =
      if etv d σ 0 n % d ^ n = (etv d σ n n + 2) % d ^ n then 1 else 0 := by
    simp only [xN, val, map_cons, map_nil]
    have e1 : rb n + 1 = bout n (n - 1) := by unfold bout; rw [if_pos (by omega)]
    have e2 : rb n + 2 = sv n n := by unfold sv; split_ifs <;> omega
    have eX : (engE d n).dimv (n - 1) = d ^ n := by
      have := engD_dI d (le_refl n); rwa [dI, if_neg hn.ne'] at this
    rw [e1, e2, tauE_bout d σ (by omega), tauE_sv d σ hn le_rfl, eX,
      show (engE d n).phase (n + 1) = _ from engP_shift d n, Fca.tX_shift _ (pow_pos hd n),
      show n - 1 + 1 = n by omega]
  have hBs : ((bN n).map (val (engE d n) (tauE d σ n))).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨y, hy, rfl⟩ := mem_map.1 hx
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hy
    have ht := mem_range.1 ht
    simp only [val, map_cons, map_nil]
    rw [show (engE d n).dimv (dI n t) = d ^ t from engD_dI d ht.le, eD0, tauE_bout d σ ht,
      tauE_bin d σ ht, hoff (n + t) (by omega)]
    unfold tB
    rw [if_pos ⟨etv_lt ha t ht.le, ha t ht, rfl⟩]
  have hSs : ((sN n).map (val (engE d n) (tauE d σ n))).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨y, hy, rfl⟩ := mem_map.1 hx
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hy
    have ht := mem_range.1 ht
    simp only [val, map_cons, map_nil]
    rw [show (engE d n).dimv (dI n t) = d ^ t from engD_dI d ht.le, eD0, tauE_sv d σ hn ht.le,
      tauE_sv d σ hn (by omega), hoff t (by omega)]
    unfold tB
    rw [if_pos ⟨etv_lt hb' t ht.le, hb t ht, by simp [etv]⟩]
  rw [prodVal_engR, hH, hBs, hX, hSs, hE]
  ring

/-! ## The boxed side at its pinned labels -/

theorem tz_a0 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a0 n j m) = τ (n + j) := by
  rw [show a0 n j m = 10 * n + (tri n * 0 + (tri j + m)) by simp only [a0]; ring,
    tz_pair n τ (by omega) (a0_lt hm hj), if_pos rfl, jOf_tri hm]

theorem tz_a9 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a9 n j m) = τ j := by
  rw [show a9 n j m = 10 * n + (tri n * 3 + (tri j + m)) by simp only [a9]; ring,
    tz_pair n τ (by omega) (a0_lt hm hj), if_neg (by omega), if_pos rfl, jOf_tri hm]

theorem tz_a2 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a2 n m j) = τ (cb n m + 1) := by
  rw [show a2 n m j = 10 * n + (tri n * 1 + i2 n m j) by simp only [a2]; ring,
    tz_pair n τ (by omega) (i2_lt hm hj), if_neg (by omega), if_neg (by omega), mOf_off hm hj]
  rfl

theorem tz_a5 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a5 n m j) = τ (cb n m + 1) := by
  rw [show a5 n m j = 10 * n + (tri n * 2 + i2 n m j) by simp only [a5]; ring,
    tz_pair n τ (by omega) (i2_lt hm hj), if_neg (by omega), if_neg (by omega), mOf_off hm hj]
  rfl

theorem tz_cb (n : ℕ) (τ : ℕ → ℕ) {m r : ℕ} (hm : m < n) (hr : r < 8) :
    tz n τ (cb n m + r) = if r = 0 then τ (n + m) else if r = 7 then τ m
      else if r = 2 ∨ r = 3 ∨ r = 4 then τ (cb n m + 1) else τ (cb n m + r) :=
  tz_chain n τ hm hr

theorem tz_low (n : ℕ) (τ : ℕ → ℕ) {x : ℕ} (hx : x < 2 * n) : tz n τ x = τ x := by
  unfold tz; rw [if_neg (by omega), if_neg (by omega)]

theorem tz_off (n : ℕ) (τ : ℕ → ℕ) {x : ℕ} (hx : x ∉ chainP n ++ pairP n) : tz n τ x = τ x := by
  have hc : ¬(2 * n ≤ x ∧ x < 10 * n ∧ ((x - 2 * n) % 8 = 0 ∨ (x - 2 * n) % 8 = 2 ∨
      (x - 2 * n) % 8 = 3 ∨ (x - 2 * n) % 8 = 4 ∨ (x - 2 * n) % 8 = 7)) := fun ⟨h1, h2, h3⟩ =>
    hx (mem_append_left _ (mem_flatMap.2 ⟨(x - 2 * n) / 8, mem_range.2 (by omega),
      mem_map.2 ⟨(x - 2 * n) % 8, by simp only [mem_cons, not_mem_nil, or_false]; omega,
        by omega⟩⟩))
  have hp : ¬(10 * n ≤ x ∧ x < 10 * n + 4 * tri n) := fun ⟨h1, h2⟩ =>
    hx (mem_append_right _ (mem_map.2 ⟨x - 10 * n, mem_range.2 (by omega), by omega⟩))
  unfold tz
  split_ifs <;> first | rfl | (exfalso; omega)

theorem labDim_low (d : ℕ) {n x : ℕ} (hn : 0 < n) (hx : x < rb n) : labDim (engE d n) x = d := by
  unfold labDim
  rw [show (engE d n).labs = engLabs n from rfl, Fca.lookup_none _ _ (by
    simp only [engLabs, map_map, mem_map, mem_range, Function.comp_def, not_exists, not_and]
    intro k _ e; omega)]
  exact engD_zero d hn

theorem tz_lt (d : ℕ) {n : ℕ} {τ : ℕ → ℕ} (hf : ∀ m < n, τ (cb n m + 1) < d)
    (hb : ∀ x < 2 * n, τ x < d) : ∀ l ∈ chainP n ++ pairP n, tz n τ l < d := by
  intro l hl
  rcases mem_append.1 hl with hc | hp
  · simp only [chainP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil, or_false] at hc
    obtain ⟨m, hm, r, hr, rfl⟩ := hc
    rw [show 2 * n + 8 * m + r = cb n m + r from rfl,
      tz_cb n τ hm (by rcases hr with rfl | rfl | rfl | rfl | rfl <;> omega)]
    rcases hr with rfl | rfl | rfl | rfl | rfl <;> simp <;> first
      | exact hb _ (by omega) | exact hf m hm
  · simp only [pairP, mem_map, mem_range] at hp
    obtain ⟨k, hk, rfl⟩ := hp
    have hT : 0 < tri n := by
      rcases Nat.eq_zero_or_pos (tri n) with h0 | h0
      · rw [h0] at hk; omega
      · exact h0
    have hs' : k % tri n < tri n := Nat.mod_lt _ hT
    have hg : k / tri n < 4 := by rw [Nat.div_lt_iff_lt_mul hT]; linarith
    rw [(Nat.div_add_mod k (tri n)).symm, tz_pair n τ hg hs']
    have := jOf_lt hs'
    have := (mOf_spec hs').1
    split_ifs
    · exact hb _ (by omega)
    · exact hb _ (by omega)
    · exact hf _ (by omega)

/-- Copy `m`'s nodes at the pinned labels, as a function of its free labels' values. -/
noncomputable def Tm (d n : ℕ) (σ : ℕ → ℕ) (m x u v : ℕ) : ℂ :=
  fou d x (σ (n + m)) * ex (engP d n (m + 1) x) * fou d u x * fou d v u * fou d (σ m) v *
    ((later n m).map fun j => tP (engD d n (j - m)) 1 [σ (n + j), x] *
      tP (engD d n (j - m)) (-1) [σ j, x]).prod

theorem prodVal_tz (d : ℕ) {n : ℕ} (hn : 0 < n) {τ : ℕ → ℕ}
    (hf : ∀ m < n, τ (cb n m + 1) < d) (hb : ∀ x < 2 * n, τ x < d) :
    prodVal (engE d n) (engL n).nodes (tz n τ) = ((range n).map fun m =>
      Tm d n τ m (τ (cb n m + 1)) (τ (cb n m + 5)) (τ (cb n m + 6))).prod := by
  set E := engE d n
  have eD0 : E.dimv 0 = d := engD_zero d hn
  have eP0 : E.phase 0 = fun _ => 0 := engP_zero d n
  have hz1 : ∀ {os is : List ℕ} {k : ℕ}, (∀ v ∈ (os ++ is).map (tz n τ), v = k) → os ++ is ≠ [] →
      k < d → val E (tz n τ) (nd .Z os is) = 1 := fun {os is k} hv hne hk => by
    simp only [nd, val]
    rw [eD0, eP0, ← map_append, Fca.tZ_all _ k _ _ (by simpa using hne) hv hk, ex_zero]
  have hcb : ∀ m < n, ∀ r < 8, tz n τ (cb n m + r) = if r = 0 then τ (n + m) else if r = 7 then τ m
      else if r = 2 ∨ r = 3 ∨ r = 4 then τ (cb n m + 1) else τ (cb n m + r) :=
    fun m hm r hr => tz_cb n τ hm hr
  have hG0 : ((range n).map fun j => val E (tz n τ)
      (nd .Z [cb n j] ((n + j) :: (range j).map (a0 n j)))).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (n + j)) ?_ (by simp) (hb _ (by omega))
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map, mem_range,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | ⟨k, hk, rfl⟩
    · rw [show cb n j = cb n j + 0 from rfl, hcb j hj 0 (by omega)]; rfl
    · exact tz_low n τ (by omega)
    · exact tz_a0 n τ hk hj
  have hG2 : ((range n).map fun j => val E (tz n τ)
      (nd .Z ((cb n j + 2) :: (later n j).map (a2 n j)) [cb n j + 1])).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (cb n j + 1)) ?_ (by simp) (hf j hj)
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with (rfl | ⟨k, hk, rfl⟩) | rfl
    · rw [hcb j hj 2 (by omega)]; simp
    · obtain ⟨h1, h2⟩ := mem_later.1 hk; exact tz_a2 n τ h1 h2
    · rw [hcb j hj 1 (by omega)]; simp
  have hG5 : ((range n).map fun j => val E (tz n τ)
      (nd .Z ((cb n j + 4) :: (later n j).map (a5 n j)) [cb n j + 3])).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (cb n j + 1)) ?_ (by simp) (hf j hj)
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with (rfl | ⟨k, hk, rfl⟩) | rfl
    · rw [hcb j hj 4 (by omega)]; simp
    · obtain ⟨h1, h2⟩ := mem_later.1 hk; exact tz_a5 n τ h1 h2
    · rw [hcb j hj 3 (by omega)]; simp
  have hG9 : ((range n).map fun j => val E (tz n τ)
      (nd .Z [j] ((cb n j + 7) :: (range j).map (a9 n j)))).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ j) ?_ (by simp) (hb _ (by omega))
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map, mem_range,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | ⟨k, hk, rfl⟩
    · exact tz_low n τ (by omega)
    · rw [hcb j hj 7 (by omega)]; simp
    · exact tz_a9 n τ hk hj
  have hC : ∀ m ∈ range n, ((chainN n m).map (val E (tz n τ))).prod =
      fou d (τ (cb n m + 1)) (τ (n + m)) * ex (engP d n (m + 1) (τ (cb n m + 1))) *
        fou d (τ (cb n m + 5)) (τ (cb n m + 1)) * fou d (τ (cb n m + 6)) (τ (cb n m + 5)) *
        fou d (τ m) (τ (cb n m + 6)) := by
    intro m hm
    have hm := mem_range.1 hm
    simp only [chainN, nd, map_cons, map_nil, prod_cons, prod_nil, mul_one, val]
    rw [show cb n m = cb n m + 0 from rfl, hcb m hm 0 (by omega), hcb m hm 1 (by omega),
      hcb m hm 2 (by omega), hcb m hm 3 (by omega), hcb m hm 4 (by omega),
      hcb m hm 5 (by omega), hcb m hm 6 (by omega), hcb m hm 7 (by omega), eD0]
    simp only [show (1 : ℕ) ≠ 0 by omega, show (1 : ℕ) ≠ 7 by omega, show (2 : ℕ) ≠ 0 by omega,
      show (2 : ℕ) ≠ 7 by omega, show (3 : ℕ) ≠ 0 by omega, show (3 : ℕ) ≠ 7 by omega,
      show (5 : ℕ) ≠ 0 by omega, show (5 : ℕ) ≠ 7 by omega, show (6 : ℕ) ≠ 0 by omega,
      show (6 : ℕ) ≠ 7 by omega, show (7 : ℕ) ≠ 0 by omega, show (4 : ℕ) ≠ 0 by omega,
      show (4 : ℕ) ≠ 7 by omega, if_true, if_false, true_or, or_true,
      show ¬((1 : ℕ) = 2 ∨ (1 : ℕ) = 3 ∨ (1 : ℕ) = 4) by omega,
      show ¬((5 : ℕ) = 2 ∨ (5 : ℕ) = 3 ∨ (5 : ℕ) = 4) by omega,
      show ¬((6 : ℕ) = 2 ∨ (6 : ℕ) = 3 ∨ (6 : ℕ) = 4) by omega]
    rw [show [τ (cb n m + 1)] ++ [τ (cb n m + 1)] = [τ (cb n m + 1), τ (cb n m + 1)] from rfl,
      Fca.tZ_all _ (τ (cb n m + 1)) _ _ (by simp) (by simp) (hf m hm)]
    rw [show E.phase (m + 1) = engP d n (m + 1) from rfl]
    ring
  have hP0 : ∀ m ∈ range n, ((later n m).map fun j => val E (tz n τ)
      ⟨.ZP 0 0 1, [a0 n j m], [a2 n m j], j - m, 0⟩).prod =
      ((later n m).map fun j => tP (engD d n (j - m)) 1 [τ (n + j), τ (cb n m + 1)]).prod := by
    intro m _
    congr 1
    refine List.map_congr_left fun j hj => ?_
    obtain ⟨h1, h2⟩ := mem_later.1 hj
    simp only [val, map_cons, map_nil, cons_append, nil_append]
    rw [tz_a0 n τ h1 h2, tz_a2 n τ h1 h2]; rfl
  have hP3 : ∀ m ∈ range n, ((later n m).map fun j => val E (tz n τ)
      ⟨.ZP 0 0 (-1), [a9 n j m], [a5 n m j], j - m, 0⟩).prod =
      ((later n m).map fun j => tP (engD d n (j - m)) (-1) [τ j, τ (cb n m + 1)]).prod := by
    intro m _
    congr 1
    refine List.map_congr_left fun j hj => ?_
    obtain ⟨h1, h2⟩ := mem_later.1 hj
    simp only [val, map_cons, map_nil, cons_append, nil_append]
    rw [tz_a9 n τ h1 h2, tz_a5 n τ h1 h2]; rfl
  simp only [engL, prodVal, map_append, prod_append, map_flatMap, prod_flatMapC, map_map,
    Function.comp_def]
  rw [hG0, hG2, hG5, hG9, List.map_congr_left hC, List.map_congr_left hP0,
    List.map_congr_left hP3]
  simp only [Fca.lprod_range, mul_one]
  rw [← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
  refine Finset.prod_congr rfl fun m _ => ?_
  simp only [Tm, List.prod_map_mul]
  ring


/-! ## The adder arithmetic over all copies -/

section arith

open Fca

/-- Copy `i`'s Fourier sum in the form of `Fca.sumC_lift`, without its pair scalar. -/
noncomputable def Gl (d : ℕ) (σ : ℕ → ℕ) (i : ℕ) : ℂ :=
  (d : ℂ)⁻¹ * ∑ x ∈ Finset.range d, (om (d ^ (i + 1)) ^ (dv d σ 0 (i + 1) + 2) *
    (om (d ^ (i + 1)))⁻¹ ^ dv d σ 1 (i + 1)) ^ x

theorem cs_ne (d n : ℕ) (hd : 0 < d) : cs d n ≠ 0 := by
  unfold cs
  rw [lprod_range]
  exact Finset.prod_ne_zero_iff.2 fun i _ => inv_ne_zero (pow_ne_zero _ (by exact_mod_cast hd.ne'))

theorem prod_Gl (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) : ∀ n, (∀ i < n, σ (sl i 0) < d) →
    (∀ i < n, σ (sl i 1) < d) → ∏ i ∈ Finset.range n, Gl d σ i = IndS d σ n
  | 0, _, _ => by simp [IndS, topv]
  | n + 1, ha, hb => by
    rw [Finset.prod_range_succ, prod_Gl d hd σ n (fun i hi => ha i (by omega))
      (fun i hi => hb i (by omega))]
    have h := arith d hd n σ ha hb
    rw [sumC_lift d hd] at h
    apply mul_left_cancel₀ (cs_ne d n hd)
    rw [← h, Gl]
    ring

theorem tP_pos (D a x : ℕ) : tP D 1 [a, x] = om D ^ (a * x) := by
  rw [om_eq_ex]; simp only [tP]; congr 1; push_cast; ring

theorem tP_neg (D o x : ℕ) : tP D (-1) [o, x] = (om D ^ (o * x))⁻¹ := by
  rw [om_eq_ex, ex_inv]; simp only [tP]; congr 1; push_cast; ring

/-- Copy `i`'s summand at the Fourier value `x`, its pairs as pair phases. -/
noncomputable def Ce (d : ℕ) (σ : ℕ → ℕ) (i x : ℕ) : ℂ :=
  fou d x (σ (sl i 0)) * ex (phase d (2 * i + 1) x) * (starRingEnd ℂ) (fou d (σ (sl i 1)) x) *
    ((range i).map fun k => tP (pD d i k) 1 [σ (sl k 0), x] *
      tP (pD d i k) (-1) [σ (sl k 1), x]).prod

theorem Ce_sum (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) (i : ℕ) :
    ∑ x ∈ Finset.range d, Ce d σ i x = Gl d σ i := by
  rw [Gl, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  have hp : ((range i).map fun k => tP (pD d i k) 1 [σ (sl k 0), x] *
      tP (pD d i k) (-1) [σ (sl k 1), x]).prod =
      om (d ^ (i + 1)) ^ (x * dv d σ 0 i) * (om (d ^ (i + 1)))⁻¹ ^ (x * dv d σ 1 i) := by
    have hf : ∀ k ∈ Finset.range i, tP (pD d i k) 1 [σ (sl k 0), x] *
        tP (pD d i k) (-1) [σ (sl k 1), x] =
        om (d ^ (i + 1)) ^ (x * (σ (sl k 0) * d ^ k)) *
          (om (d ^ (i + 1)))⁻¹ ^ (x * (σ (sl k 1) * d ^ k)) := by
      intro k hk
      have hk := Finset.mem_range.1 hk
      rw [tP_pos, tP_neg, om_lift d hd (n := i) (D := pD d i k) hk.le rfl, ← pow_mul, ← pow_mul,
        ← inv_pow,
        show d ^ k * (σ (sl k 0) * x) = x * (σ (sl k 0) * d ^ k) by ring,
        show d ^ k * (σ (sl k 1) * x) = x * (σ (sl k 1) * d ^ k) by ring]
    rw [lprod_range, Finset.prod_congr rfl hf, Finset.prod_mul_distrib,
      Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum, dv, dv, Finset.mul_sum,
      Finset.mul_sum]
  have hdn : d = d ^ (i - i + 1) := by simp
  have hc : (starRingEnd ℂ) (fou d (σ (sl i 1)) x) =
      (om (d ^ (i + 1)))⁻¹ ^ (x * (σ (sl i 1) * d ^ i)) / rt d := by
    unfold fou
    rw [map_div₀, map_pow, conj_om, Complex.conj_ofReal, om_lift d hd le_rfl hdn, ← inv_pow,
      ← pow_mul, show d ^ i * (σ (sl i 1) * x) = x * (σ (sl i 1) * d ^ i) by ring]
  rw [Ce, hp, hc, fou_lift d hd le_rfl hdn, ex_lift, dv_succ, dv_succ, mul_pow, ← pow_mul,
    ← pow_mul,
    show (dv d σ 0 i + σ (sl i 0) * d ^ i + 2) * x =
      x * (σ (sl i 0) * d ^ i) + 2 * x + x * dv d σ 0 i by ring,
    show (dv d σ 1 i + σ (sl i 1) * d ^ i) * x =
      x * (σ (sl i 1) * d ^ i) + x * dv d σ 1 i by ring,
    pow_add, pow_add, pow_add, ← rt_inv_sq d]
  ring

/-- Every copy's Fourier sum with its pair phases, multiplied: the shift factor. -/
theorem prod_Ce (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) (n : ℕ) (ha : ∀ i < n, σ (sl i 0) < d)
    (hb : ∀ i < n, σ (sl i 1) < d) :
    ∏ i ∈ Finset.range n, ∑ x ∈ Finset.range d, Ce d σ i x = IndS d σ n := by
  simp only [Ce_sum d hd]
  exact prod_Gl d hd σ n ha hb

end arith


/-! ## The boxed side in closed form -/

section closed

open Fca

theorem sumOver_blocks (D : ℕ → ℕ) (c : ℕ) (G : ℕ → ℕ → ℕ → ℕ → ℂ) :
    ∀ (ms : List ℕ), ms.Nodup → ∀ σ : ℕ → ℕ,
      sumOver D (ms.flatMap fun m => [c + 8 * m + 1, c + 8 * m + 5, c + 8 * m + 6])
        (fun τ => (ms.map fun m => G m (τ (c + 8 * m + 1)) (τ (c + 8 * m + 5))
          (τ (c + 8 * m + 6))).prod) σ =
      (ms.map fun m => ∑ x ∈ Finset.range (D (c + 8 * m + 1)),
        ∑ u ∈ Finset.range (D (c + 8 * m + 5)), ∑ v ∈ Finset.range (D (c + 8 * m + 6)),
          G m x u v).prod
  | [], _, σ => by simp [sumOver]
  | m :: ms, h, σ => by
    have hm : m ∉ ms := (nodup_cons.1 h).1
    rw [flatMap_cons]
    simp only [map_cons, prod_cons]
    rw [sumOver_mul_split (Sf := [c + 8 * m + 1, c + 8 * m + 5, c + 8 * m + 6])
      (Sg := ms.flatMap fun m => [c + 8 * m + 1, c + 8 * m + 5, c + 8 * m + 6])]
    · rw [sumOver_blocks D c G ms (nodup_cons.1 h).2 σ]
      congr 1
      simp only [sumOver]
      refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun u _ =>
        Finset.sum_congr rfl fun v _ => ?_
      simp
    · intro τ τ' e
      dsimp only
      rw [e (c + 8 * m + 1) (by simp), e (c + 8 * m + 5) (by simp), e (c + 8 * m + 6) (by simp)]
    · intro τ τ' e
      dsimp only
      congr 1
      refine List.map_congr_left fun k hk => ?_
      rw [e (c + 8 * k + 1) (mem_flatMap.2 ⟨k, hk, by simp⟩),
        e (c + 8 * k + 5) (mem_flatMap.2 ⟨k, hk, by simp⟩),
        e (c + 8 * k + 6) (mem_flatMap.2 ⟨k, hk, by simp⟩)]
    · intro x hx hx'
      obtain ⟨k, hk, hk'⟩ := mem_flatMap.1 hx'
      have hne : k ≠ m := fun e => hm (e ▸ hk)
      simp only [mem_cons, not_mem_nil, or_false] at hx hk'
      rcases hx with rfl | rfl | rfl <;> rcases hk' with e | e | e <;> omega
    · intro x hx hx'
      obtain ⟨k, hk, hk'⟩ := mem_flatMap.1 hx
      have hne : k ≠ m := fun e => hm (e ▸ hk)
      simp only [mem_cons, not_mem_nil, or_false] at hx' hk'
      rcases hx' with rfl | rfl | rfl <;> rcases hk' with e | e | e <;> omega

theorem tri_sq (n : ℕ) : 2 * tri n + n = n * n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [tri]; nlinarith

theorem rb_eq (n : ℕ) : rb n = 10 * n + 4 * tri n := by
  have := tri_sq n; unfold rb; nlinarith

theorem Tm_congr (d n : ℕ) {σ τ : ℕ → ℕ} (h : ∀ x < 2 * n, τ x = σ x) {m : ℕ} (hm : m < n)
    (x u v : ℕ) : Tm d n τ m x u v = Tm d n σ m x u v := by
  unfold Tm
  rw [h (n + m) (by omega), h m (by omega)]
  congr 2
  refine List.map_congr_left fun j hj => ?_
  obtain ⟨_, h2⟩ := mem_later.1 hj
  rw [h (n + j) (by omega), h j (by omega)]

/-- Copy `m`'s summand at the Fourier value `x`, the `F³` chain summed. -/
noncomputable def Ec (d n : ℕ) (σ : ℕ → ℕ) (m x : ℕ) : ℂ :=
  fou d x (σ (n + m)) * ex (engP d n (m + 1) x) * (starRingEnd ℂ) (fou d (σ m) x) *
    ((later n m).map fun j => tP (engD d n (j - m)) 1 [σ (n + j), x] *
      tP (engD d n (j - m)) (-1) [σ j, x]).prod

theorem frees_eq (n : ℕ) :
    frees n = (range n).flatMap fun m =>
      [2 * n + 8 * m + 1, 2 * n + 8 * m + 5, 2 * n + 8 * m + 6] := by
  simp [frees]

/-- The boxed side at `n ≥ 1` copies: the product over copies of the Fourier sums. -/
theorem eval_engL (d : ℕ) (hd : 0 < d) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ)
    (hb : ∀ x < 2 * n, σ x < d) :
    eval (engE d n) (engL n) σ = ∏ m ∈ Finset.range n, ∑ x ∈ Finset.range d, Ec d n σ m x := by
  have hrb : ∀ x, x < 10 * n + 4 * tri n → labDim (engE d n) x = d := fun x hx =>
    labDim_low d hn (by rw [rb_eq]; exact hx)
  unfold eval
  rw [sumOver_perm _ (int_perm n), sumOver_append]
  rw [sumOver_congr _ (frees n) _ (fun τ => ((range n).map fun m =>
    Tm d n σ m (τ (cb n m + 1)) (τ (cb n m + 5)) (τ (cb n m + 6))).prod) σ ?_]
  · rw [frees_eq]
    refine (sumOver_blocks (labDim (engE d n)) (2 * n) (Tm d n σ) _ nodup_range σ).trans ?_
    rw [lprod_range]
    refine Finset.prod_congr rfl fun m hm => ?_
    have hm := Finset.mem_range.1 hm
    rw [hrb (2 * n + 8 * m + 1) (by omega), hrb (2 * n + 8 * m + 5) (by omega),
      hrb (2 * n + 8 * m + 6) (by omega)]
    refine Finset.sum_congr rfl fun x _ => ?_
    have h3 := fou_three d (σ m) x hd
    rw [Finset.sum_comm, Ec, ← h3]
    simp only [Tm, Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun u _ => ?_
    ring
  · intro τ hτ hr
    have hlow : ∀ x < 2 * n, τ x = σ x := fun x hx => hτ x fun h => by
      rw [frees_eq] at h
      obtain ⟨k, _, hk⟩ := mem_flatMap.1 h
      simp only [mem_cons, not_mem_nil, or_false] at hk
      rcases hk with e | e | e <;> omega
    have hf : ∀ m < n, τ (cb n m + 1) < d := fun m hm => by
      have := hr (cb n m + 1)
        (by rw [frees_eq]; exact mem_flatMap.2 ⟨m, mem_range.2 hm, by simp [cb]⟩)
      rwa [hrb (cb n m + 1) (by unfold cb; omega)] at this
    have hbτ : ∀ x < 2 * n, τ x < d := fun x hx => by rw [hlow x hx]; exact hb x hx
    rw [Fca.sumOver_single _ _ (pins_nodup n) _ τ (tz n τ) (fun x hx => tz_off n τ hx) ?_ ?_,
      prodVal_tz d hn hf hbτ]
    · congr 1
      refine List.map_congr_left fun m hm => ?_
      exact Tm_congr d n hlow (mem_range.1 hm) _ _ _
    · intro l hl
      rw [hrb l ?_]
      · exact tz_lt d hf hbτ l hl
      · rcases mem_append.1 hl with hc | hp
        · simp only [chainP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil,
            or_false] at hc
          obtain ⟨m, hm, r, hr, rfl⟩ := hc
          rcases hr with rfl | rfl | rfl | rfl | rfl <;> omega
        · simp only [pairP, mem_map, mem_range] at hp
          obtain ⟨k, hk, rfl⟩ := hp
          omega
    · intro τ' h0 _ hne l hl
      exact engL_support d hne h0 l hl

/-- The digits of the engine's boundary at Fca's digit labels: copy `k` is the engine's copy
`n - 1 - k`. -/
def psiL (n : ℕ) : List (ℕ × ℕ) :=
  (range n).flatMap fun k => [(sl k 0, n + (n - 1 - k)), (sl k 1, n - 1 - k)]

/-- `σ` read at Fca's digit labels. -/
def sigF (n : ℕ) (σ : ℕ → ℕ) (y : ℕ) : ℕ := σ (((psiL n).lookup y).getD 0)

theorem psiL_nodup (n : ℕ) : ((psiL n).map Prod.fst).Nodup := by
  have e : (psiL n).map Prod.fst = (range n).flatMap fun k => [sl k 0, sl k 1] := by
    simp [psiL, map_flatMap]
  rw [e, nodup_flatMap]
  refine ⟨fun k _ => by simp [sl], ?_⟩
  refine (pairwise_lt_range (n := n)).imp fun {a b} hab => List.disjoint_left.2 fun x hx hx' => ?_
  simp only [mem_cons, not_mem_nil, or_false] at hx hx'
  rcases hx with rfl | rfl <;> rcases hx' with e | e <;>
    have := (sl_inj (by omega) (by omega) e).1 <;> omega

theorem sigF_0 (n : ℕ) (σ : ℕ → ℕ) {k : ℕ} (hk : k < n) :
    sigF n σ (sl k 0) = σ (n + (n - 1 - k)) := by
  rw [sigF, Fca.lookup_of_mem _ (psiL_nodup n) (k := sl k 0) (v := n + (n - 1 - k))
    (mem_flatMap.2 ⟨k, mem_range.2 hk, mem_cons_self⟩)]; rfl

theorem sigF_1 (n : ℕ) (σ : ℕ → ℕ) {k : ℕ} (hk : k < n) :
    sigF n σ (sl k 1) = σ (n - 1 - k) := by
  rw [sigF, Fca.lookup_of_mem _ (psiL_nodup n) (k := sl k 1) (v := n - 1 - k)
    (mem_flatMap.2 ⟨k, mem_range.2 hk, mem_cons_of_mem _ mem_cons_self⟩)]; rfl

theorem engP_phase (d : ℕ) {n m : ℕ} (hm : m < n) (x : ℕ) :
    engP d n (m + 1) x = phase d (2 * (n - 1 - m) + 1) x := by
  unfold engP phase
  rw [if_neg (by omega), if_pos (by omega), if_neg (by omega), if_pos (by omega),
    show n - (m + 1) + 1 = (2 * (n - 1 - m) + 1) / 2 + 1 by omega]

theorem Ec_Ce (d : ℕ) {n m : ℕ} (hm : m < n) (σ : ℕ → ℕ) (x : ℕ) :
    Ec d n σ m x = Ce d (sigF n σ) (n - 1 - m) x := by
  unfold Ec Ce
  rw [sigF_0 n σ (by omega), sigF_1 n σ (by omega), engP_phase d hm,
    show n + (n - 1 - (n - 1 - m)) = n + m by omega, show n - 1 - (n - 1 - m) = m by omega]
  congr 1
  rw [later, map_map, lprod_range, lprod_range]
  rw [show n - m - 1 = n - 1 - m by omega,
    ← Finset.prod_range_reflect _ (n - 1 - m)]
  refine Finset.prod_congr rfl fun t ht => ?_
  have ht := Finset.mem_range.1 ht
  simp only [Function.comp_def]
  rw [sigF_0 n σ (show t < n by omega), sigF_1 n σ (show t < n by omega),
    show m + 1 + (n - 1 - m - 1 - t) = n - 1 - t by omega,
    show engD d n (n - 1 - t - m) = pD d (n - 1 - m) t by
      unfold engD pD; rw [if_pos (by omega)]; congr 1; omega]

theorem topv_sigF (d n s : ℕ) (σ : ℕ → ℕ) (hs : s < 2) :
    ∀ t ≤ n, topv d (sigF n σ) n s t = etv d σ (if s = 0 then n else 0) t
  | 0, _ => rfl
  | t + 1, ht => by
    simp only [topv, etv, topv_sigF d n s σ hs t (by omega)]
    congr 1
    interval_cases s
    · rw [sigF_0 n σ (by omega), if_pos rfl]; congr 1; omega
    · rw [sigF_1 n σ (by omega), if_neg (by omega), Nat.zero_add]; congr 1; omega

end closed

/-! ## The engine's fca family -/

/-- The engine's boxed Fourier constant adder equals its spec at every `n ≥ 1` copies, for
every dimension `d`. -/
theorem engFam (d : ℕ) (hd : 0 < d) (n : ℕ) (hn : 1 ≤ n) :
    Equal (engE d n) 1 (engL n) 1 (engR n) := by
  refine ⟨List.Perm.refl _, fun σ hσ => ?_⟩
  have hrb : 2 * n ≤ rb n := by unfold rb; nlinarith
  have hb : ∀ x < 2 * n, σ x < d := fun x hx => by
    have := hσ x (mem_range.2 hx)
    rwa [labDim_low d hn (by omega)] at this
  rw [one_mul, one_mul, eval_engL d hd hn σ hb, eval_engR d hd hn σ
    (fun t ht => hb _ (by omega)) (fun t ht => hb _ (by omega))]
  have key : ∏ m ∈ Finset.range n, ∑ x ∈ Finset.range d, Ec d n σ m x =
      ∏ i ∈ Finset.range n, ∑ x ∈ Finset.range d, Ce d (sigF n σ) i x := by
    rw [← Finset.prod_range_reflect _ n]
    refine Finset.prod_congr rfl fun i hi => Finset.sum_congr rfl fun x _ => ?_
    have hi := Finset.mem_range.1 hi
    rw [Ec_Ce d (by omega), show n - 1 - (n - 1 - i) = i by omega]
  rw [key, prod_Ce d hd (sigF n σ) n (fun i hi => by rw [sigF_0 n σ hi]; exact hb _ (by omega))
    (fun i hi => by rw [sigF_1 n σ hi]; exact hb _ (by omega)), Fca.IndS,
    topv_sigF d n 0 σ (by omega) n le_rfl, topv_sigF d n 1 σ (by omega) n le_rfl]
  simp

theorem engFam_8 (d : ℕ) (hd : 0 < d) : Equal (engE d 8) 1 (engL 8) 1 (engR 8) :=
  engFam d hd 8 (by norm_num)

theorem engFam_2 : ∀ n, 1 ≤ n → Equal (engE 2 n) 1 (engL n) 1 (engR n) :=
  engFam 2 (by norm_num)


/-! ## The engine's literal exports -/

/-- The engine export of the boxed side at n = 1. -/
def litL1 : Net := ⟨([⟨.F, [3], [2], 0, 0⟩, ⟨.Z, [5], [4], 0, 1⟩, ⟨.F, [7], [6], 0, 0⟩, ⟨.F, [8], [7], 0, 0⟩, ⟨.F, [9], [8], 0, 0⟩, ⟨.Z, [2], [1], 0, 0⟩, ⟨.Z, [4], [3], 0, 0⟩, ⟨.Z, [6], [5], 0, 0⟩, ⟨.Z, [0], [9], 0, 0⟩] : List Node), [2, 3, 4, 5, 6, 7, 8, 9], [0, 1]⟩

/-- The engine export of the spec at n = 1. -/
def litR1 : Net := ⟨([⟨.Z, [10], [], 1, 0⟩, ⟨.B, [11], [10, 1], 1, 0⟩, ⟨.X, [12], [11], 0, 2⟩, ⟨.S, [13, 0], [12], 1, 0⟩, ⟨.Z, [], [13], 1, 0⟩] : List Node), [10, 11, 12, 13], [0, 1]⟩

/-- The engine export of the boxed side at n = 2. -/
def litL2 : Net := ⟨([⟨.F, [5], [4], 0, 0⟩, ⟨.Z, [7], [6], 0, 1⟩, ⟨.F, [9], [8], 0, 0⟩, ⟨.F, [10], [9], 0, 0⟩, ⟨.F, [11], [10], 0, 0⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.Z, [15], [14], 0, 2⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.F, [18], [17], 0, 0⟩, ⟨.F, [19], [18], 0, 0⟩, ⟨.Z, [4], [2], 0, 0⟩, ⟨.Z, [12], [3, 20], 0, 0⟩, ⟨.Z, [6, 21], [5], 0, 0⟩, ⟨.Z, [14], [13], 0, 0⟩, ⟨.Z, [8, 22], [7], 0, 0⟩, ⟨.Z, [16], [15], 0, 0⟩, ⟨.Z, [0], [11], 0, 0⟩, ⟨.Z, [1], [19, 23], 0, 0⟩, ⟨.ZP 0 0 (1), [20], [21], 1, 0⟩, ⟨.ZP 0 0 (-1), [23], [22], 1, 0⟩] : List Node), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [0, 1, 2, 3]⟩

/-- The engine export of the spec at n = 2. -/
def litR2 : Net := ⟨([⟨.Z, [24], [], 2, 0⟩, ⟨.X, [26], [25], 1, 3⟩, ⟨.Z, [], [27], 2, 0⟩, ⟨.B, [28], [24, 2], 2, 0⟩, ⟨.B, [25], [28, 3], 0, 0⟩, ⟨.S, [27, 0], [29], 2, 0⟩, ⟨.S, [29, 1], [26], 0, 0⟩] : List Node), [24, 25, 26, 27, 28, 29], [0, 1, 2, 3]⟩

/-- The engine export of the boxed side at n = 3. -/
def litL3 : Net := ⟨([⟨.F, [7], [6], 0, 0⟩, ⟨.Z, [9], [8], 0, 1⟩, ⟨.F, [11], [10], 0, 0⟩, ⟨.F, [12], [11], 0, 0⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.Z, [17], [16], 0, 2⟩, ⟨.F, [19], [18], 0, 0⟩, ⟨.F, [20], [19], 0, 0⟩, ⟨.F, [21], [20], 0, 0⟩, ⟨.F, [23], [22], 0, 0⟩, ⟨.Z, [25], [24], 0, 3⟩, ⟨.F, [27], [26], 0, 0⟩, ⟨.F, [28], [27], 0, 0⟩, ⟨.F, [29], [28], 0, 0⟩, ⟨.Z, [6], [3], 0, 0⟩, ⟨.Z, [14], [4, 30], 0, 0⟩, ⟨.Z, [22], [5, 31, 32], 0, 0⟩, ⟨.Z, [8, 33, 34], [7], 0, 0⟩, ⟨.Z, [16, 35], [15], 0, 0⟩, ⟨.Z, [24], [23], 0, 0⟩, ⟨.Z, [10, 36, 37], [9], 0, 0⟩, ⟨.Z, [18, 38], [17], 0, 0⟩, ⟨.Z, [26], [25], 0, 0⟩, ⟨.Z, [0], [13], 0, 0⟩, ⟨.Z, [1], [21, 39], 0, 0⟩, ⟨.Z, [2], [29, 40, 41], 0, 0⟩, ⟨.ZP 0 0 (1), [30], [33], 1, 0⟩, ⟨.ZP 0 0 (1), [31], [34], 2, 0⟩, ⟨.ZP 0 0 (1), [32], [35], 1, 0⟩, ⟨.ZP 0 0 (-1), [39], [36], 1, 0⟩, ⟨.ZP 0 0 (-1), [40], [37], 2, 0⟩, ⟨.ZP 0 0 (-1), [41], [38], 1, 0⟩] : List Node), [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], [0, 1, 2, 3, 4, 5]⟩

/-- The engine export of the spec at n = 3. -/
def litR3 : Net := ⟨([⟨.Z, [42], [], 3, 0⟩, ⟨.X, [44], [43], 2, 4⟩, ⟨.Z, [], [45], 3, 0⟩, ⟨.B, [46], [42, 3], 3, 0⟩, ⟨.B, [47], [46, 4], 0, 0⟩, ⟨.B, [43], [47, 5], 1, 0⟩, ⟨.S, [45, 0], [48], 3, 0⟩, ⟨.S, [48, 1], [49], 0, 0⟩, ⟨.S, [49, 2], [44], 1, 0⟩] : List Node), [42, 43, 44, 45, 46, 47, 48, 49], [0, 1, 2, 3, 4, 5]⟩

/-- The engine export of the boxed side at n = 4. -/
def litL4 : Net := ⟨([⟨.F, [9], [8], 0, 0⟩, ⟨.Z, [11], [10], 0, 1⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.F, [14], [13], 0, 0⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.Z, [19], [18], 0, 2⟩, ⟨.F, [21], [20], 0, 0⟩, ⟨.F, [22], [21], 0, 0⟩, ⟨.F, [23], [22], 0, 0⟩, ⟨.F, [25], [24], 0, 0⟩, ⟨.Z, [27], [26], 0, 3⟩, ⟨.F, [29], [28], 0, 0⟩, ⟨.F, [30], [29], 0, 0⟩, ⟨.F, [31], [30], 0, 0⟩, ⟨.F, [33], [32], 0, 0⟩, ⟨.Z, [35], [34], 0, 4⟩, ⟨.F, [37], [36], 0, 0⟩, ⟨.F, [38], [37], 0, 0⟩, ⟨.F, [39], [38], 0, 0⟩, ⟨.Z, [8], [4], 0, 0⟩, ⟨.Z, [16], [5, 40], 0, 0⟩, ⟨.Z, [24], [6, 41, 42], 0, 0⟩, ⟨.Z, [32], [7, 43, 44, 45], 0, 0⟩, ⟨.Z, [10, 46, 47, 48], [9], 0, 0⟩, ⟨.Z, [18, 49, 50], [17], 0, 0⟩, ⟨.Z, [26, 51], [25], 0, 0⟩, ⟨.Z, [34], [33], 0, 0⟩, ⟨.Z, [12, 52, 53, 54], [11], 0, 0⟩, ⟨.Z, [20, 55, 56], [19], 0, 0⟩, ⟨.Z, [28, 57], [27], 0, 0⟩, ⟨.Z, [36], [35], 0, 0⟩, ⟨.Z, [0], [15], 0, 0⟩, ⟨.Z, [1], [23, 58], 0, 0⟩, ⟨.Z, [2], [31, 59, 60], 0, 0⟩, ⟨.Z, [3], [39, 61, 62, 63], 0, 0⟩, ⟨.ZP 0 0 (1), [40], [46], 1, 0⟩, ⟨.ZP 0 0 (1), [41], [47], 2, 0⟩, ⟨.ZP 0 0 (1), [43], [48], 3, 0⟩, ⟨.ZP 0 0 (1), [42], [49], 1, 0⟩, ⟨.ZP 0 0 (1), [44], [50], 2, 0⟩, ⟨.ZP 0 0 (1), [45], [51], 1, 0⟩, ⟨.ZP 0 0 (-1), [58], [52], 1, 0⟩, ⟨.ZP 0 0 (-1), [59], [53], 2, 0⟩, ⟨.ZP 0 0 (-1), [61], [54], 3, 0⟩, ⟨.ZP 0 0 (-1), [60], [55], 1, 0⟩, ⟨.ZP 0 0 (-1), [62], [56], 2, 0⟩, ⟨.ZP 0 0 (-1), [63], [57], 1, 0⟩] : List Node), [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], [0, 1, 2, 3, 4, 5, 6, 7]⟩

/-- The engine export of the spec at n = 4. -/
def litR4 : Net := ⟨([⟨.Z, [64], [], 4, 0⟩, ⟨.X, [66], [65], 3, 5⟩, ⟨.Z, [], [67], 4, 0⟩, ⟨.B, [68], [64, 4], 4, 0⟩, ⟨.B, [69], [68, 5], 0, 0⟩, ⟨.B, [70], [69, 6], 1, 0⟩, ⟨.B, [65], [70, 7], 2, 0⟩, ⟨.S, [67, 0], [71], 4, 0⟩, ⟨.S, [71, 1], [72], 0, 0⟩, ⟨.S, [72, 2], [73], 1, 0⟩, ⟨.S, [73, 3], [66], 2, 0⟩] : List Node), [64, 65, 66, 67, 68, 69, 70, 71, 72, 73], [0, 1, 2, 3, 4, 5, 6, 7]⟩

/-- The engine export of the boxed side at n = 5. -/
def litL5 : Net := ⟨([⟨.F, [11], [10], 0, 0⟩, ⟨.Z, [13], [12], 0, 1⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.F, [16], [15], 0, 0⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.F, [19], [18], 0, 0⟩, ⟨.Z, [21], [20], 0, 2⟩, ⟨.F, [23], [22], 0, 0⟩, ⟨.F, [24], [23], 0, 0⟩, ⟨.F, [25], [24], 0, 0⟩, ⟨.F, [27], [26], 0, 0⟩, ⟨.Z, [29], [28], 0, 3⟩, ⟨.F, [31], [30], 0, 0⟩, ⟨.F, [32], [31], 0, 0⟩, ⟨.F, [33], [32], 0, 0⟩, ⟨.F, [35], [34], 0, 0⟩, ⟨.Z, [37], [36], 0, 4⟩, ⟨.F, [39], [38], 0, 0⟩, ⟨.F, [40], [39], 0, 0⟩, ⟨.F, [41], [40], 0, 0⟩, ⟨.F, [43], [42], 0, 0⟩, ⟨.Z, [45], [44], 0, 5⟩, ⟨.F, [47], [46], 0, 0⟩, ⟨.F, [48], [47], 0, 0⟩, ⟨.F, [49], [48], 0, 0⟩, ⟨.Z, [10], [5], 0, 0⟩, ⟨.Z, [18], [6, 50], 0, 0⟩, ⟨.Z, [26], [7, 51, 52], 0, 0⟩, ⟨.Z, [34], [8, 53, 54, 55], 0, 0⟩, ⟨.Z, [42], [9, 56, 57, 58, 59], 0, 0⟩, ⟨.Z, [12, 60, 61, 62, 63], [11], 0, 0⟩, ⟨.Z, [20, 64, 65, 66], [19], 0, 0⟩, ⟨.Z, [28, 67, 68], [27], 0, 0⟩, ⟨.Z, [36, 69], [35], 0, 0⟩, ⟨.Z, [44], [43], 0, 0⟩, ⟨.Z, [14, 70, 71, 72, 73], [13], 0, 0⟩, ⟨.Z, [22, 74, 75, 76], [21], 0, 0⟩, ⟨.Z, [30, 77, 78], [29], 0, 0⟩, ⟨.Z, [38, 79], [37], 0, 0⟩, ⟨.Z, [46], [45], 0, 0⟩, ⟨.Z, [0], [17], 0, 0⟩, ⟨.Z, [1], [25, 80], 0, 0⟩, ⟨.Z, [2], [33, 81, 82], 0, 0⟩, ⟨.Z, [3], [41, 83, 84, 85], 0, 0⟩, ⟨.Z, [4], [49, 86, 87, 88, 89], 0, 0⟩, ⟨.ZP 0 0 (1), [50], [60], 1, 0⟩, ⟨.ZP 0 0 (1), [51], [61], 2, 0⟩, ⟨.ZP 0 0 (1), [53], [62], 3, 0⟩, ⟨.ZP 0 0 (1), [56], [63], 4, 0⟩, ⟨.ZP 0 0 (1), [52], [64], 1, 0⟩, ⟨.ZP 0 0 (1), [54], [65], 2, 0⟩, ⟨.ZP 0 0 (1), [57], [66], 3, 0⟩, ⟨.ZP 0 0 (1), [55], [67], 1, 0⟩, ⟨.ZP 0 0 (1), [58], [68], 2, 0⟩, ⟨.ZP 0 0 (1), [59], [69], 1, 0⟩, ⟨.ZP 0 0 (-1), [80], [70], 1, 0⟩, ⟨.ZP 0 0 (-1), [81], [71], 2, 0⟩, ⟨.ZP 0 0 (-1), [83], [72], 3, 0⟩, ⟨.ZP 0 0 (-1), [86], [73], 4, 0⟩, ⟨.ZP 0 0 (-1), [82], [74], 1, 0⟩, ⟨.ZP 0 0 (-1), [84], [75], 2, 0⟩, ⟨.ZP 0 0 (-1), [87], [76], 3, 0⟩, ⟨.ZP 0 0 (-1), [85], [77], 1, 0⟩, ⟨.ZP 0 0 (-1), [88], [78], 2, 0⟩, ⟨.ZP 0 0 (-1), [89], [79], 1, 0⟩] : List Node), [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]⟩

/-- The engine export of the spec at n = 5. -/
def litR5 : Net := ⟨([⟨.Z, [90], [], 5, 0⟩, ⟨.X, [92], [91], 4, 6⟩, ⟨.Z, [], [93], 5, 0⟩, ⟨.B, [94], [90, 5], 5, 0⟩, ⟨.B, [95], [94, 6], 0, 0⟩, ⟨.B, [96], [95, 7], 1, 0⟩, ⟨.B, [97], [96, 8], 2, 0⟩, ⟨.B, [91], [97, 9], 3, 0⟩, ⟨.S, [93, 0], [98], 5, 0⟩, ⟨.S, [98, 1], [99], 0, 0⟩, ⟨.S, [99, 2], [100], 1, 0⟩, ⟨.S, [100, 3], [101], 2, 0⟩, ⟨.S, [101, 4], [92], 3, 0⟩] : List Node), [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]⟩

theorem engL_lit1 : engL 1 = litL1 := by decide +kernel

theorem engR_lit1 : engR 1 = litR1 := by decide +kernel

theorem engL_lit2 : engL 2 = litL2 := by decide +kernel

theorem engR_lit2 : engR 2 = litR2 := by decide +kernel

theorem engL_lit3 : engL 3 = litL3 := by decide +kernel

theorem engR_lit3 : engR 3 = litR3 := by decide +kernel

theorem engL_lit4 : engL 4 = litL4 := by decide +kernel

theorem engR_lit4 : engR 4 = litR4 := by decide +kernel

theorem engL_lit5 : engL 5 = litL5 := by decide +kernel

theorem engR_lit5 : engR 5 = litR5 := by decide +kernel

theorem engLabs_lit1 : (engLabs 1).filter (fun p => p.2 ≠ 0) = [(10, 1), (13, 1)] := by
  decide +kernel

theorem engLabs_lit2 : (engLabs 2).filter (fun p => p.2 ≠ 0) = [(24, 2), (25, 1), (26, 1), (27, 2)] := by
  decide +kernel

theorem engLabs_lit3 : (engLabs 3).filter (fun p => p.2 ≠ 0) = [(42, 3), (43, 2), (44, 2), (45, 3), (47, 1), (49, 1)] := by
  decide +kernel

theorem engLabs_lit4 : (engLabs 4).filter (fun p => p.2 ≠ 0) = [(64, 4), (65, 3), (66, 3), (67, 4), (69, 1), (70, 2), (72, 1), (73, 2)] := by
  decide +kernel

theorem engLabs_lit5 : (engLabs 5).filter (fun p => p.2 ≠ 0) = [(90, 5), (91, 4), (92, 4), (93, 5), (95, 1), (96, 2), (97, 3), (99, 1), (100, 2), (101, 3)] := by
  decide +kernel

end FcaEng

end ArchytasZX
