/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.AdderFamDraper
import ArchytasZX.AdderFamFcaEng

/-!
# The engine's Draper adder networks

The networks the engine exports for the flat `draper_adder` at `n` copies and its binder / copy
and sum / splitter spec, with their environment.

* `engL n`: outputs `2j` (`a`), `2j + 1` (`b`), inputs `2n + 2j`, `2n + 2j + 1`, copy `j`'s chain
  at `4n + 11j`, the pair legs from `15n`, the pair phases as `ZP` nodes.
* `engR n`: the spec from `16n + 6 tri n`.
* `engE d n`: dimension `i < n` is `d ^ (i + 1)`, dimension `n` is `1`, every phase `0`.
* `engL_lit`, `engR_lit`, `engLabs_lit`: equal to the engine's literal exports at `n = 0 … 5`.
* `eval_engR`: the spec as the copy of `a` and the sum `a + b` mod `d ^ n`.
* `eval_engL`: the boxed side as the product over copies of each copy's Fourier sum with its
  `ZP` pair phases, by pinning every chain and pair label.
* `Ec_Ce`, `prod_Ce`: copy `k` read as Draper's copy `n - 1 - k`; the product is `Draper.IndS`
  (from `Draper.arith`).
* `engFam`: `engL n = engR n` over `engE d n` for every `n` and every `d > 0`.
-/

namespace ArchytasZX

namespace DraperEng

open List
open FcaEng (tri off i2 later nd dI engD engD_dI lookup_shift mem_later jOf jOf_tri jOf_spec jOf_lt mOf mOf_off mOf_spec
  off_mono off_tri tri_mono tri_lt a0_lt i2_lt blk tri_sq)

/-- The `a` output of copy `j`. -/
def ao (j : ℕ) : ℕ := 2 * j

/-- The `b` output of copy `j`. -/
def bo (j : ℕ) : ℕ := 2 * j + 1

/-- The `a` input of copy `j`. -/
def ai (n j : ℕ) : ℕ := 2 * n + 2 * j

/-- The `b` input of copy `j`. -/
def bi (n j : ℕ) : ℕ := 2 * n + 2 * j + 1

/-- Copy `j`'s chain base. -/
def cb (n j : ℕ) : ℕ := 4 * n + 11 * j

/-- The `g0` leg of copy `j` to copy `m < j`. -/
def a0 (n j m : ℕ) : ℕ := 15 * n + tri j + m

/-- The `g2` leg of copy `m` to copy `j > m`. -/
def a2 (n m j : ℕ) : ℕ := 15 * n + tri n + i2 n m j

/-- The `g7` leg of copy `j` to copy `m < j`. -/
def a7 (n j m : ℕ) : ℕ := 15 * n + 2 * tri n + tri j + m

/-- The `g8` output of copy `m`. -/
def g8o (n m : ℕ) : ℕ := 15 * n + 3 * tri n + off n m + m

/-- The `g8` leg of copy `m` to copy `j > m`. -/
def a8 (n m j : ℕ) : ℕ := 15 * n + 3 * tri n + i2 n m j + m + 1

/-- The `g10` leg of copy `m` to copy `j > m`. -/
def a10 (n m j : ℕ) : ℕ := 15 * n + 4 * tri n + n + i2 n m j

/-- The `g14` leg of copy `j` to copy `m < j`. -/
def a14 (n j m : ℕ) : ℕ := 15 * n + 5 * tri n + n + tri j + m

/-- Copy `j`'s chain nodes `g1, g4, g5, g6, g11, g12, g13`. -/
def chainN (n j : ℕ) : List Node :=
  [nd .F [cb n j + 1] [cb n j], nd .Z [cb n j + 2, cb n j + 3] [ai n j],
   nd .Z [cb n j + 6] [cb n j + 4, cb n j + 5], nd .F [cb n j + 5] [cb n j + 3],
   nd .F [cb n j + 8] [cb n j + 7], nd .F [cb n j + 9] [cb n j + 8],
   nd .F [cb n j + 10] [cb n j + 9]]

/-- The engine's boxed side at `n` copies. -/
def engL (n : ℕ) : Net where
  nodes :=
    (range n).flatMap (chainN n) ++
    (range n).map (fun j => nd .Z [cb n j] (bi n j :: (range j).map (a0 n j))) ++
    (range n).map (fun j => nd .Z ((cb n j + 4) :: (later n j).map (a2 n j)) [cb n j + 1]) ++
    (range n).map (fun j => nd .Z [ao j] ((cb n j + 2) :: (range j).map (a7 n j))) ++
    (range n).map (fun j => nd .Z (g8o n j :: (later n j).map (a8 n j)) [cb n j + 6]) ++
    (range n).map (fun j => nd .Z ((cb n j + 7) :: (later n j).map (a10 n j)) [g8o n j]) ++
    (range n).map (fun j => nd .Z [bo j] ((cb n j + 10) :: (range j).map (a14 n j))) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 1, [a0 n j m], [a2 n m j], j - m, 0⟩) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 1, [a7 n j m], [a8 n m j], j - m, 0⟩) ++
    (range n).flatMap (fun m => (later n m).map fun j =>
      ⟨.ZP 0 0 (-1), [a14 n j m], [a10 n m j], j - m, 0⟩)
  int := (range (12 * n + 6 * tri n)).map (4 * n + ·)
  bnd := range (4 * n)

/-- Start of the spec's labels. -/
def rb (n : ℕ) : ℕ := 16 * n + 6 * tri n

/-- The spec's offset of role `k`: at `n = 1` the engine's own order. -/
def pofs (n k : ℕ) : ℕ := if n = 1 then [0, 3, 4, 7, 1, 2, 8, 5, 6].getD k k else k

/-- The role of the spec's offset `o`. -/
def rofs (n o : ℕ) : ℕ := if n = 1 then [0, 4, 5, 1, 2, 7, 8, 3, 6].getD o o else o

/-- The spec's label of role `k`: `0 … 3` the heads and ends, `4, 5, 6` the copy spider, `7, 8`
the sum spider, then the binder and the splitter values. -/
def rl (n k : ℕ) : ℕ := rb n + pofs n k

/-- The binder value of chain `c` after step `t`. -/
def bv (n c t : ℕ) : ℕ := if t + 1 = n then rl n (4 + 3 * c) else rl n (9 + 2 * t + c)

/-- The binder value of chain `c` before step `t`. -/
def bw (n c t : ℕ) : ℕ := if t = 0 then rl n (2 * c) else bv n c (t - 1)

/-- The splitter value of chain `c` of the top `t` digits. -/
def sv (n c t : ℕ) : ℕ :=
  if t = 0 then rl n (2 * c + 1) else if t = n then rl n (5 + 3 * c)
  else rl n (9 + 2 * (n - 1) + 2 * (t - 1) + c)

/-- The binder of chain `c` at step `t`. -/
def bN (n c t : ℕ) : Node := ⟨.B, [bv n c t], [bw n c t, 2 * n + 2 * t + c], dI n t, 0⟩

/-- The splitter of chain `c` at step `t`. -/
def sN (n c t : ℕ) : Node := ⟨.S, [sv n c t, 2 * t + c], [sv n c (t + 1)], dI n t, 0⟩

/-- The head of chain `c`. -/
def hN (n c : ℕ) : Node := ⟨.Z, [rl n (2 * c)], [], n, 0⟩

/-- The end of chain `c`. -/
def eN (n c : ℕ) : Node := ⟨.Z, [], [rl n (2 * c + 1)], n, 0⟩

/-- The copy spider. -/
def zN (n : ℕ) : Node := ⟨.Z, [rl n 5, rl n 6], [rl n 4], n - 1, 0⟩

/-- The sum spider. -/
def xN (n : ℕ) : Node := ⟨.X, [rl n 8], [rl n 7, rl n 6], n - 1, 0⟩

/-- The engine's spec at `n` digits. -/
def engR (n : ℕ) : Net where
  nodes := if n = 0 then [⟨.Z, [0], [], 0, 0⟩, ⟨.Z, [], [1], 0, 0⟩, ⟨.Z, [2], [], 0, 0⟩,
      ⟨.Z, [], [3], 0, 0⟩, ⟨.Z, [1, 4], [0], 0, 0⟩, ⟨.X, [3], [2, 4], 0, 0⟩]
    else if n = 1 then [hN n 0, bN n 0 0, sN n 0 0, eN n 0, hN n 1, bN n 1 0, sN n 1 0, eN n 1,
      zN n, xN n]
    else [hN n 0, eN n 0, hN n 1, eN n 1, zN n, xN n] ++
      (range n).flatMap (fun t => [bN n 0 t, bN n 1 t]) ++
      (range n).flatMap (fun t => [sN n 0 t, sN n 1 t])
  int := (range (4 * n + 5)).map (rb n + ·)
  bnd := range (4 * n)

/-- The exponent of `d` in the dimension of the spec's role `k`. -/
def edim (n k : ℕ) : ℕ :=
  if k < 4 then 0 else if k < 9 then n else if k < 9 + 2 * (n - 1) then (k - 9) / 2 + 1
  else (k - 9 - 2 * (n - 1)) / 2 + 1

/-- The spec's label dimensions. -/
def engLabs (n : ℕ) : List (ℕ × ℕ) :=
  (range (4 * n + 5)).map fun o => (rb n + o, dI n (edim n (rofs n o)))

/-- The engine's environment at `n`: the dimensions of `FcaEng`, every phase `0`. -/
noncomputable def engE (d n : ℕ) : Env :=
  { dimv := engD d n, phase := fun _ _ => 0, labs := engLabs n }

/-! ## The boxed side's labels -/

/-- The free labels of the boxed side: each copy's Fourier value and `F³` legs. -/
def frees (n : ℕ) : List ℕ := (range n).flatMap fun m => [1, 8, 9].map fun r => 4 * n + 11 * m + r

/-- The pinned chain labels of the boxed side. -/
def chainP (n : ℕ) : List ℕ := (range n).flatMap fun m =>
  [0, 2, 3, 4, 5, 6, 7, 10].map fun r => 4 * n + 11 * m + r

/-- The pair region of the boxed side: the pair legs and the `g8` outputs. -/
def pairP (n : ℕ) : List ℕ := (range (n + 6 * tri n)).map (15 * n + ·)

theorem nodup_blocks (n c : ℕ) (R : List ℕ) (hR : R.Nodup) (hR8 : ∀ r ∈ R, r < 11) :
    ((range n).flatMap fun i => R.map fun r => c + 11 * i + r).Nodup := by
  rw [nodup_flatMap]
  refine ⟨fun i _ => hR.map fun a b h => by omega, ?_⟩
  refine (pairwise_lt_range (n := n)).imp fun {i j} hij => ?_
  simp only [Function.onFun, List.disjoint_left, mem_map]
  rintro _ ⟨r, hr, rfl⟩ ⟨r', hr', e⟩
  have := hR8 r hr; have := hR8 r' hr'; omega

theorem frees_nodup (n : ℕ) : (frees n).Nodup :=
  nodup_blocks n (4 * n) [1, 8, 9] (by decide) (by simp)

theorem pins_nodup (n : ℕ) : (chainP n ++ pairP n).Nodup := by
  have h1 := nodup_blocks n (4 * n) [0, 2, 3, 4, 5, 6, 7, 10] (by decide) (by simp)
  have h2 : (pairP n).Nodup := (nodup_range).map fun a b h => by simpa using h
  rw [nodup_append]
  refine ⟨h1, h2, ?_⟩
  intro a ha b hb e
  simp only [chainP, pairP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil,
    or_false] at ha hb
  obtain ⟨m, hm, r, hr, rfl⟩ := ha
  obtain ⟨k, _, rfl⟩ := hb
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega

theorem int_perm (n : ℕ) : (engL n).int.Perm (frees n ++ (chainP n ++ pairP n)) := by
  have hnd : (frees n ++ (chainP n ++ pairP n)).Nodup := by
    rw [nodup_append]
    refine ⟨frees_nodup n, pins_nodup n, ?_⟩
    intro a ha b hb e
    simp only [frees, chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons,
      not_mem_nil, or_false] at ha hb
    obtain ⟨m, hm, r, hr, rfl⟩ := ha
    rcases hb with ⟨k, hk, s, hs, rfl⟩ | ⟨k, _, rfl⟩
    · rcases hr with rfl | rfl | rfl <;>
        rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega
    · rcases hr with rfl | rfl | rfl <;> omega
  refine (List.perm_ext_iff_of_nodup ((nodup_range).map fun a b h => by simpa using h) hnd).2
    fun x => ?_
  simp only [frees, chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons,
    not_mem_nil, or_false]
  constructor
  · rintro ⟨k, hk, rfl⟩
    by_cases h : k < 11 * n
    · have h8 : k % 11 < 11 := Nat.mod_lt _ (by norm_num)
      have e : 4 * n + k = 4 * n + 11 * (k / 11) + k % 11 := by omega
      have hm : k / 11 < n := by omega
      interval_cases hr : k % 11
      · exact Or.inr (Or.inl ⟨k / 11, hm, 0, by simp, e.symm⟩)
      · exact Or.inl ⟨k / 11, hm, 1, by simp, e.symm⟩
      · exact Or.inr (Or.inl ⟨k / 11, hm, 2, by simp, e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 11, hm, 3, by simp, e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 11, hm, 4, by simp, e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 11, hm, 5, by simp, e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 11, hm, 6, by simp, e.symm⟩)
      · exact Or.inr (Or.inl ⟨k / 11, hm, 7, by simp, e.symm⟩)
      · exact Or.inl ⟨k / 11, hm, 8, by simp, e.symm⟩
      · exact Or.inl ⟨k / 11, hm, 9, by simp, e.symm⟩
      · exact Or.inr (Or.inl ⟨k / 11, hm, 10, by simp, e.symm⟩)
    · exact Or.inr (Or.inr ⟨k - 11 * n, by omega, by omega⟩)
  · rintro (⟨m, hm, r, hr, rfl⟩ | ⟨m, hm, r, hr, rfl⟩ | ⟨k, hk, rfl⟩)
    · exact ⟨11 * m + r, by rcases hr with rfl | rfl | rfl <;> omega, by omega⟩
    · exact ⟨11 * m + r, by
        rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega, by omega⟩
    · exact ⟨11 * n + k, by omega, by omega⟩

/-! ## The `g8` block -/

/-- The start of copy `m`'s `g8` block. -/
def off2 (n m : ℕ) : ℕ := off n m + m

theorem off2_succ {n m : ℕ} (hm : m < n) : off2 n (m + 1) = off2 n m + (n - m) := by
  simp only [off2, off]; omega

theorem off2_mono (n : ℕ) {a b : ℕ} (h : a ≤ b) : off2 n a ≤ off2 n b := by
  have := off_mono n h; unfold off2; omega

theorem off2_top (n : ℕ) : off2 n n = tri n + n := by
  have := off_tri n n le_rfl
  simp only [Nat.sub_self] at this
  unfold off2; simp only [tri] at this; omega

/-- The copy `m` of the `g8` block position `s`. -/
def mOf2 (n s : ℕ) : ℕ := Nat.find (p := fun m => s < off2 n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩

theorem mOf2_spec {n s : ℕ} (hs : s < tri n + n) :
    mOf2 n s < n ∧ off2 n (mOf2 n s) ≤ s ∧ s < off2 n (mOf2 n s + 1) := by
  have hP := Nat.find_spec (p := fun m => s < off2 n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩
  have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp [tri] at hs)
  have hlt : mOf2 n s < n := by
    by_contra h
    have := Nat.find_min (p := fun m => s < off2 n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩
      (show n - 1 < mOf2 n s by omega)
    rw [show n - 1 + 1 = n by omega, off2_top] at this
    exact this (Or.inl hs)
  refine ⟨hlt, ?_, hP.resolve_right (by change ¬mOf2 n s = n; omega)⟩
  rcases Nat.eq_zero_or_pos (mOf2 n s) with h | h
  · rw [h]; simp [off2, off]
  · have := Nat.find_min (p := fun m => s < off2 n (m + 1) ∨ m = n) ⟨n, Or.inr rfl⟩
      (show mOf2 n s - 1 < mOf2 n s by omega)
    change ¬(s < off2 n (mOf2 n s - 1 + 1) ∨ mOf2 n s - 1 = n) at this
    rw [show mOf2 n s - 1 + 1 = mOf2 n s by omega] at this; omega

theorem mOf2_eq {n m t : ℕ} (hm : m < n) (ht : t < n - m) : mOf2 n (off2 n m + t) = m := by
  have hs : off2 n m + t < tri n + n := by
    have := off2_mono n (show m + 1 ≤ n by omega); rw [off2_succ hm, off2_top] at this; omega
  obtain ⟨h0, h1, h2⟩ := mOf2_spec hs
  have hb : off2 n m + t < off2 n (m + 1) := by rw [off2_succ hm]; omega
  rcases lt_trichotomy (mOf2 n (off2 n m + t)) m with h | h | h
  · have := off2_mono n (show mOf2 n (off2 n m + t) + 1 ≤ m by omega); omega
  · exact h
  · have := off2_mono n (show m + 1 ≤ mOf2 n (off2 n m + t) by omega); omega

theorem g8o_eq (n m : ℕ) : g8o n m = 15 * n + 3 * tri n + (off2 n m + 0) := by
  simp only [g8o, off2]; ring

theorem a8_eq {n m j : ℕ} (hm : m < j) : a8 n m j = 15 * n + 3 * tri n + (off2 n m + (j - m)) := by
  simp only [a8, off2, i2]; omega

/-! ## The boxed side's pinned labels -/

/-- The pinned value of each pinned label of the boxed side, the free and boundary labels from
`τ`. -/
def tz (n : ℕ) (τ : ℕ → ℕ) (x : ℕ) : ℕ :=
  if 4 * n ≤ x ∧ x < 15 * n then
    if (x - 4 * n) % 11 = 0 then τ (bi n ((x - 4 * n) / 11))
    else if (x - 4 * n) % 11 = 2 ∨ (x - 4 * n) % 11 = 3 then τ (ai n ((x - 4 * n) / 11))
    else if 4 ≤ (x - 4 * n) % 11 ∧ (x - 4 * n) % 11 ≤ 7 then τ (cb n ((x - 4 * n) / 11) + 1)
    else if (x - 4 * n) % 11 = 10 then τ (bo ((x - 4 * n) / 11))
    else τ x
  else if 15 * n ≤ x ∧ x < 15 * n + 3 * tri n then
    if (x - 15 * n) / tri n = 0 then τ (bi n (jOf ((x - 15 * n) % tri n)))
    else if (x - 15 * n) / tri n = 1 then τ (cb n (mOf n ((x - 15 * n) % tri n)) + 1)
    else τ (ai n (jOf ((x - 15 * n) % tri n)))
  else if 15 * n + 3 * tri n ≤ x ∧ x < 15 * n + 4 * tri n + n then
    τ (cb n (mOf2 n (x - (15 * n + 3 * tri n))) + 1)
  else if 15 * n + 4 * tri n + n ≤ x ∧ x < 16 * n + 6 * tri n then
    if (x - (15 * n + 4 * tri n + n)) / tri n = 0 then
      τ (cb n (mOf n ((x - (15 * n + 4 * tri n + n)) % tri n)) + 1)
    else τ (bo (jOf ((x - (15 * n + 4 * tri n + n)) % tri n)))
  else τ x

theorem tz_chain (n : ℕ) (τ : ℕ → ℕ) {m r : ℕ} (hm : m < n) (hr : r < 11) :
    tz n τ (cb n m + r) = if r = 0 then τ (bi n m) else if r = 2 ∨ r = 3 then τ (ai n m)
      else if 4 ≤ r ∧ r ≤ 7 then τ (cb n m + 1) else if r = 10 then τ (bo m)
      else τ (cb n m + r) := by
  have e1 : (cb n m + r - 4 * n) % 11 = r := by unfold cb; omega
  have e2 : (cb n m + r - 4 * n) / 11 = m := by unfold cb; omega
  unfold tz
  rw [if_pos (by unfold cb; omega), e1, e2]

theorem tz_pA (n : ℕ) (τ : ℕ → ℕ) {g s : ℕ} (hg : g < 3) (hs : s < tri n) :
    tz n τ (15 * n + (tri n * g + s)) = if g = 0 then τ (bi n (jOf s))
      else if g = 1 then τ (cb n (mOf n s) + 1) else τ (ai n (jOf s)) := by
  have hT : tri n * g + s < 3 * tri n := by nlinarith
  obtain ⟨e1, e2⟩ := blk (g := g) hs
  unfold tz
  rw [if_neg (by omega), if_pos ⟨by omega, by omega⟩, show 15 * n + (tri n * g + s) - 15 * n =
    tri n * g + s by omega, e1, e2]

theorem tz_g8 (n : ℕ) (τ : ℕ → ℕ) {s : ℕ} (hs : s < tri n + n) :
    tz n τ (15 * n + 3 * tri n + s) = τ (cb n (mOf2 n s) + 1) := by
  unfold tz
  rw [if_neg (by omega), if_neg (by omega), if_pos ⟨by omega, by omega⟩,
    show 15 * n + 3 * tri n + s - (15 * n + 3 * tri n) = s by omega]

theorem tz_pB (n : ℕ) (τ : ℕ → ℕ) {g s : ℕ} (hg : g < 2) (hs : s < tri n) :
    tz n τ (15 * n + 4 * tri n + n + (tri n * g + s)) =
      if g = 0 then τ (cb n (mOf n s) + 1) else τ (bo (jOf s)) := by
  have hT : tri n * g + s < 2 * tri n := by nlinarith
  obtain ⟨e1, e2⟩ := blk (g := g) hs
  unfold tz
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos ⟨by omega, by omega⟩,
    show 15 * n + 4 * tri n + n + (tri n * g + s) - (15 * n + 4 * tri n + n) = tri n * g + s by
      omega, e1, e2]

theorem tz_a0 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a0 n j m) = τ (bi n j) := by
  rw [show a0 n j m = 15 * n + (tri n * 0 + (tri j + m)) by simp only [a0]; ring,
    tz_pA n τ (by omega) (a0_lt hm hj), if_pos rfl, jOf_tri hm]

theorem tz_a2 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a2 n m j) = τ (cb n m + 1) := by
  rw [show a2 n m j = 15 * n + (tri n * 1 + i2 n m j) by simp only [a2]; ring,
    tz_pA n τ (by omega) (i2_lt hm hj), if_neg (by omega), if_pos rfl, mOf_off hm hj]

theorem tz_a7 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a7 n j m) = τ (ai n j) := by
  rw [show a7 n j m = 15 * n + (tri n * 2 + (tri j + m)) by simp only [a7]; ring,
    tz_pA n τ (by omega) (a0_lt hm hj), if_neg (by omega), if_neg (by omega), jOf_tri hm]

theorem tz_g8o (n : ℕ) (τ : ℕ → ℕ) {m : ℕ} (hm : m < n) : tz n τ (g8o n m) = τ (cb n m + 1) := by
  have hs : off2 n m + 0 < tri n + n := by
    have := off2_mono n (show m + 1 ≤ n by omega); rw [off2_succ hm, off2_top] at this; omega
  rw [g8o_eq, tz_g8 n τ hs, mOf2_eq hm (by omega)]

theorem tz_a8 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a8 n m j) = τ (cb n m + 1) := by
  have hs : off2 n m + (j - m) < tri n + n := by
    have := off2_mono n (show m + 1 ≤ n by omega)
    rw [off2_succ (by omega), off2_top] at this; omega
  rw [a8_eq hm, tz_g8 n τ hs, mOf2_eq (by omega) (by omega)]

theorem tz_a10 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a10 n m j) = τ (cb n m + 1) := by
  rw [show a10 n m j = 15 * n + 4 * tri n + n + (tri n * 0 + i2 n m j) by simp only [a10]; ring,
    tz_pB n τ (by omega) (i2_lt hm hj), if_pos rfl, mOf_off hm hj]

theorem tz_a14 (n : ℕ) (τ : ℕ → ℕ) {m j : ℕ} (hm : m < j) (hj : j < n) :
    tz n τ (a14 n j m) = τ (bo j) := by
  rw [show a14 n j m = 15 * n + 4 * tri n + n + (tri n * 1 + (tri j + m)) by
      simp only [a14]; ring,
    tz_pB n τ (by omega) (a0_lt hm hj), if_neg (by omega), jOf_tri hm]

theorem tz_low (n : ℕ) (τ : ℕ → ℕ) {x : ℕ} (hx : x < 4 * n) : tz n τ x = τ x := by
  unfold tz; rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega)]

theorem tz_off (n : ℕ) (τ : ℕ → ℕ) {x : ℕ} (hx : x ∉ chainP n ++ pairP n) : tz n τ x = τ x := by
  have hc : ¬(4 * n ≤ x ∧ x < 15 * n ∧ ((x - 4 * n) % 11 = 0 ∨ (x - 4 * n) % 11 = 2 ∨
      (x - 4 * n) % 11 = 3 ∨ (x - 4 * n) % 11 = 4 ∨ (x - 4 * n) % 11 = 5 ∨
      (x - 4 * n) % 11 = 6 ∨ (x - 4 * n) % 11 = 7 ∨ (x - 4 * n) % 11 = 10)) :=
    fun ⟨h1, h2, h3⟩ =>
    hx (mem_append_left _ (mem_flatMap.2 ⟨(x - 4 * n) / 11, mem_range.2 (by omega),
      mem_map.2 ⟨(x - 4 * n) % 11, by simp only [mem_cons, not_mem_nil, or_false]; omega,
        by omega⟩⟩))
  have hp : ¬(15 * n ≤ x ∧ x < 16 * n + 6 * tri n) := fun ⟨h1, h2⟩ =>
    hx (mem_append_right _ (mem_map.2 ⟨x - 15 * n, mem_range.2 (by omega), by omega⟩))
  unfold tz
  split_ifs <;> first | rfl | (exfalso; omega)

theorem mem_later' {n m j : ℕ} (h : j ∈ later n m) : m < j ∧ j < n := mem_later.1 h

/-! ## The boxed side's support -/

theorem mem_chain {n j : ℕ} (hj : j < n) {x : Node} (hx : x ∈ chainN n j) :
    x ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl
    (mem_flatMap.2 ⟨j, mem_range.2 hj, hx⟩)))))))))

theorem mem_g0 {n j : ℕ} (hj : j < n) :
    nd .Z [cb n j] (bi n j :: (range j).map (a0 n j)) ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr
    (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩)))))))))

theorem mem_g2 {n j : ℕ} (hj : j < n) :
    nd .Z ((cb n j + 4) :: (later n j).map (a2 n j)) [cb n j + 1] ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr
    (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩))))))))

theorem mem_g7 {n j : ℕ} (hj : j < n) :
    nd .Z [ao j] ((cb n j + 2) :: (range j).map (a7 n j)) ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr
    (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩)))))))

theorem mem_g8 {n j : ℕ} (hj : j < n) :
    nd .Z (g8o n j :: (later n j).map (a8 n j)) [cb n j + 6] ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inl (Or.inr
    (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩))))))

theorem mem_g10 {n j : ℕ} (hj : j < n) :
    nd .Z ((cb n j + 7) :: (later n j).map (a10 n j)) [g8o n j] ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inl (Or.inr (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩)))))

theorem mem_g14 {n j : ℕ} (hj : j < n) :
    nd .Z [bo j] ((cb n j + 10) :: (range j).map (a14 n j)) ∈ (engL n).nodes := by
  simp only [engL, mem_append]
  exact Or.inl (Or.inl (Or.inl (Or.inr (mem_map.2 ⟨j, mem_range.2 hj, rfl⟩))))

theorem free_notin {n m r : ℕ} (hm : m < n) (hr : r = 1 ∨ r = 8 ∨ r = 9) :
    4 * n + 11 * m + r ∉ chainP n ++ pairP n := by
  simp only [chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil,
    or_false, not_or, not_exists, not_and]
  refine ⟨fun k _ s hs => ?_, fun k _ => by omega⟩
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega

/-- The boxed side's nodes are nonzero only at the pinned values. -/
theorem engL_support (d : ℕ) {n : ℕ} {τ τ' : ℕ → ℕ}
    (h : prodVal (engE d n) (engL n).nodes τ' ≠ 0) (hτ : ∀ x ∉ chainP n ++ pairP n, τ' x = τ x) :
    ∀ l ∈ chainP n ++ pairP n, τ' l = tz n τ l := by
  have hv := Fca.prodVal_ne h
  have zq : ∀ {os is : List ℕ}, nd .Z os is ∈ (engL n).nodes → ∀ {a b : ℕ}, a ∈ os ++ is →
      b ∈ os ++ is → τ' a = τ' b := fun {os is} hn {a b} ha hb =>
    Z_eq _ _ _ rfl (hv _ hn) (a := a) (b := b) ha hb
  have bnd : ∀ x, x < 4 * n → τ' x = τ x := fun x hx => hτ x fun h => by
    simp only [chainP, pairP, mem_append, mem_flatMap, mem_range, mem_map, mem_cons,
      not_mem_nil, or_false] at h
    rcases h with ⟨k, _, s, hs, rfl⟩ | ⟨k, _, rfl⟩
    · rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega
    · omega
  have c1 : ∀ m < n, τ' (cb n m + 1) = τ (cb n m + 1) := fun m hm =>
    hτ _ (free_notin hm (Or.inl rfl))
  have c0 : ∀ m < n, τ' (cb n m) = τ (bi n m) := fun m hm => by
    rw [zq (mem_g0 hm) (a := cb n m) (b := bi n m) (by simp) (by simp),
      bnd _ (by unfold bi; omega)]
  have hg4 : ∀ m < n, nd .Z [cb n m + 2, cb n m + 3] [ai n m] ∈ (engL n).nodes := fun m hm =>
    mem_chain hm (by simp [chainN])
  have c2 : ∀ m < n, τ' (cb n m + 2) = τ (ai n m) := fun m hm => by
    rw [zq (hg4 m hm) (a := cb n m + 2) (b := ai n m) (by simp) (by simp),
      bnd _ (by unfold ai; omega)]
  have c3 : ∀ m < n, τ' (cb n m + 3) = τ (ai n m) := fun m hm => by
    rw [zq (hg4 m hm) (a := cb n m + 3) (b := ai n m) (by simp) (by simp),
      bnd _ (by unfold ai; omega)]
  have c4 : ∀ m < n, τ' (cb n m + 4) = τ (cb n m + 1) := fun m hm => by
    rw [zq (mem_g2 hm) (a := cb n m + 4) (b := cb n m + 1) (mem_append_left _ mem_cons_self)
      (mem_append_right _ (mem_singleton_self _)), c1 m hm]
  have hg5 : ∀ m < n, nd .Z [cb n m + 6] [cb n m + 4, cb n m + 5] ∈ (engL n).nodes :=
    fun m hm => mem_chain hm (by simp [chainN])
  have c5 : ∀ m < n, τ' (cb n m + 5) = τ (cb n m + 1) := fun m hm => by
    rw [zq (hg5 m hm) (a := cb n m + 5) (b := cb n m + 4) (by simp) (by simp), c4 m hm]
  have c6 : ∀ m < n, τ' (cb n m + 6) = τ (cb n m + 1) := fun m hm => by
    rw [zq (hg5 m hm) (a := cb n m + 6) (b := cb n m + 4) (by simp) (by simp), c4 m hm]
  have c8 : ∀ m < n, τ' (g8o n m) = τ (cb n m + 1) := fun m hm => by
    rw [zq (mem_g8 hm) (a := g8o n m) (b := cb n m + 6) (mem_append_left _ mem_cons_self)
      (mem_append_right _ (mem_singleton_self _)), c6 m hm]
  have c7 : ∀ m < n, τ' (cb n m + 7) = τ (cb n m + 1) := fun m hm => by
    rw [zq (mem_g10 hm) (a := cb n m + 7) (b := g8o n m) (mem_append_left _ mem_cons_self)
      (mem_append_right _ (mem_singleton_self _)), c8 m hm]
  have c10 : ∀ m < n, τ' (cb n m + 10) = τ (bo m) := fun m hm => by
    rw [zq (mem_g14 hm) (a := cb n m + 10) (b := bo m) (mem_append_right _ mem_cons_self)
      (mem_append_left _ (mem_singleton_self _)), bnd _ (by unfold bo; omega)]
  intro l hl
  rcases mem_append.1 hl with hc | hp
  · simp only [chainP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil, or_false] at hc
    obtain ⟨m, hm, r, hr, rfl⟩ := hc
    rw [show 4 * n + 11 * m + r = cb n m + r from rfl,
      tz_chain n τ hm (by rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega)]
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · simpa using c0 m hm
    · simpa using c2 m hm
    · simpa using c3 m hm
    · simpa using c4 m hm
    · simpa using c5 m hm
    · simpa using c6 m hm
    · simpa using c7 m hm
    · simpa using c10 m hm
  · simp only [pairP, mem_map, mem_range] at hp
    obtain ⟨k, hk, rfl⟩ := hp
    rcases (show k < 3 * tri n ∨ (3 * tri n ≤ k ∧ k < 4 * tri n + n) ∨ 4 * tri n + n ≤ k by
      omega) with hA | hG | hB
    · have hT : 0 < tri n := by omega
      set s := k % tri n with hs
      have hs' : s < tri n := Nat.mod_lt _ hT
      have hg : k / tri n < 3 := by rw [Nat.div_lt_iff_lt_mul hT]; linarith
      have ek : k = tri n * (k / tri n) + s := (Nat.div_add_mod k (tri n)).symm
      rw [ek, tz_pA n τ hg hs']
      obtain ⟨hj1, hj2⟩ := jOf_spec s
      have hjn := jOf_lt hs'
      obtain ⟨hm1, hm2, hm3⟩ := mOf_spec hs'
      have hm3' : s < off n (mOf n s) + (n - 1 - mOf n s) := hm3
      set j := jOf s
      set m := mOf n s
      have hjl : m + 1 + (s - off n m) ∈ later n m := mem_later.2 ⟨by omega, by omega⟩
      have hlt : s - tri j < j := by simp only [tri] at hj2; omega
      interval_cases hgv : k / tri n
      · simp only [if_true]
        have hl : 15 * n + (tri n * 0 + s) = a0 n j (s - tri j) := by simp only [a0]; omega
        rw [hl, zq (mem_g0 hjn) (a := a0 n j (s - tri j)) (b := bi n j)
          (mem_append_right _ (mem_cons_of_mem _ (mem_map.2 ⟨_, mem_range.2 hlt, rfl⟩)))
          (mem_append_right _ mem_cons_self), bnd _ (by unfold bi; omega)]
      · simp only [show (1 : ℕ) ≠ 0 by omega, if_false, if_true]
        have hl : 15 * n + (tri n * 1 + s) = a2 n m (m + 1 + (s - off n m)) := by
          simp only [a2, i2]; omega
        rw [hl, zq (mem_g2 hm1) (a := a2 n m (m + 1 + (s - off n m))) (b := cb n m + 1)
          (mem_append_left _ (mem_cons_of_mem _ (mem_map.2 ⟨_, hjl, rfl⟩)))
          (mem_append_right _ (mem_singleton_self _)), c1 m hm1]
      · simp only [show (2 : ℕ) ≠ 0 by omega, show (2 : ℕ) ≠ 1 by omega, if_false]
        have hl : 15 * n + (tri n * 2 + s) = a7 n j (s - tri j) := by simp only [a7]; omega
        rw [hl, zq (mem_g7 hjn) (a := a7 n j (s - tri j)) (b := cb n j + 2)
          (mem_append_right _ (mem_cons_of_mem _ (mem_map.2 ⟨_, mem_range.2 hlt, rfl⟩)))
          (mem_append_right _ mem_cons_self), c2 j hjn]
    · have hs : k - 3 * tri n < tri n + n := by omega
      rw [show 15 * n + k = 15 * n + 3 * tri n + (k - 3 * tri n) by omega, tz_g8 n τ hs]
      obtain ⟨hm1, hm2, hm3⟩ := mOf2_spec hs
      set m := mOf2 n (k - 3 * tri n)
      rw [off2_succ hm1] at hm3
      rcases Nat.eq_zero_or_pos (k - 3 * tri n - off2 n m) with h0 | h0
      · rw [show 15 * n + 3 * tri n + (k - 3 * tri n) = g8o n m by rw [g8o_eq]; omega, c8 m hm1]
      · have hjl : m + (k - 3 * tri n - off2 n m) ∈ later n m := mem_later.2 ⟨by omega, by omega⟩
        rw [show 15 * n + 3 * tri n + (k - 3 * tri n) = a8 n m (m + (k - 3 * tri n - off2 n m)) by
            rw [a8_eq (by omega)]; omega,
          zq (mem_g8 hm1) (a := a8 n m (m + (k - 3 * tri n - off2 n m))) (b := cb n m + 6)
            (mem_append_left _ (mem_cons_of_mem _ (mem_map.2 ⟨_, hjl, rfl⟩)))
            (mem_append_right _ (mem_singleton_self _)), c6 m hm1]
    · have hT : 0 < tri n := by omega
      set k' := k - (4 * tri n + n) with hk'
      set s := k' % tri n with hs
      have hs' : s < tri n := Nat.mod_lt _ hT
      have hg : k' / tri n < 2 := by rw [Nat.div_lt_iff_lt_mul hT]; omega
      have ek : 15 * n + k = 15 * n + 4 * tri n + n + (tri n * (k' / tri n) + s) := by
        have := (Nat.div_add_mod k' (tri n)); omega
      rw [ek, tz_pB n τ hg hs']
      obtain ⟨hj1, hj2⟩ := jOf_spec s
      have hjn := jOf_lt hs'
      obtain ⟨hm1, hm2, hm3⟩ := mOf_spec hs'
      have hm3' : s < off n (mOf n s) + (n - 1 - mOf n s) := hm3
      set j := jOf s
      set m := mOf n s
      have hjl : m + 1 + (s - off n m) ∈ later n m := mem_later.2 ⟨by omega, by omega⟩
      have hlt : s - tri j < j := by simp only [tri] at hj2; omega
      interval_cases hgv : k' / tri n
      · simp only [if_true]
        have hl : 15 * n + 4 * tri n + n + (tri n * 0 + s) = a10 n m (m + 1 + (s - off n m)) := by
          simp only [a10, i2]; omega
        rw [hl, zq (mem_g10 hm1) (a := a10 n m (m + 1 + (s - off n m))) (b := g8o n m)
          (mem_append_left _ (mem_cons_of_mem _ (mem_map.2 ⟨_, hjl, rfl⟩)))
          (mem_append_right _ (mem_singleton_self _)), c8 m hm1]
      · simp only [show (1 : ℕ) ≠ 0 by omega, if_false]
        have hl : 15 * n + 4 * tri n + n + (tri n * 1 + s) = a14 n j (s - tri j) := by
          simp only [a14]; omega
        rw [hl, zq (mem_g14 hjn) (a := a14 n j (s - tri j)) (b := bo j)
          (mem_append_right _ (mem_cons_of_mem _ (mem_map.2 ⟨_, mem_range.2 hlt, rfl⟩)))
          (mem_append_left _ (mem_singleton_self _)), bnd _ (by unfold bo; omega)]

/-- Each pinned value of the boxed side is a boundary digit or a copy's Fourier value. -/
theorem tz_cases (n : ℕ) (τ : ℕ → ℕ) : ∀ l ∈ chainP n ++ pairP n, ∃ m < n,
    tz n τ l = τ (bi n m) ∨ tz n τ l = τ (ai n m) ∨ tz n τ l = τ (cb n m + 1) ∨
      tz n τ l = τ (bo m) := by
  intro l hl
  rcases mem_append.1 hl with hc | hp
  · simp only [chainP, mem_flatMap, mem_range, mem_map, mem_cons, not_mem_nil, or_false] at hc
    obtain ⟨m, hm, r, hr, rfl⟩ := hc
    refine ⟨m, hm, ?_⟩
    rw [show 4 * n + 11 * m + r = cb n m + r from rfl,
      tz_chain n τ hm (by rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega)]
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
  · simp only [pairP, mem_map, mem_range] at hp
    obtain ⟨k, hk, rfl⟩ := hp
    rcases (show k < 3 * tri n ∨ (3 * tri n ≤ k ∧ k < 4 * tri n + n) ∨ 4 * tri n + n ≤ k by
      omega) with hA | hG | hB
    · have hT : 0 < tri n := by omega
      have hs' : k % tri n < tri n := Nat.mod_lt _ hT
      have hg : k / tri n < 3 := by rw [Nat.div_lt_iff_lt_mul hT]; linarith
      rw [(Nat.div_add_mod k (tri n)).symm, tz_pA n τ hg hs']
      have := jOf_lt hs'
      have := (mOf_spec hs').1
      split_ifs
      · exact ⟨_, by assumption, Or.inl rfl⟩
      · exact ⟨_, by assumption, Or.inr (Or.inr (Or.inl rfl))⟩
      · exact ⟨_, by assumption, Or.inr (Or.inl rfl)⟩
    · have hs : k - 3 * tri n < tri n + n := by omega
      rw [show 15 * n + k = 15 * n + 3 * tri n + (k - 3 * tri n) by omega, tz_g8 n τ hs]
      exact ⟨_, (mOf2_spec hs).1, Or.inr (Or.inr (Or.inl rfl))⟩
    · have hT : 0 < tri n := by omega
      have hs' : (k - (4 * tri n + n)) % tri n < tri n := Nat.mod_lt _ hT
      have hg : (k - (4 * tri n + n)) / tri n < 2 := by
        rw [Nat.div_lt_iff_lt_mul hT]; omega
      have ek : 15 * n + k = 15 * n + 4 * tri n + n + (tri n * ((k - (4 * tri n + n)) / tri n) +
          (k - (4 * tri n + n)) % tri n) := by
        have := Nat.div_add_mod (k - (4 * tri n + n)) (tri n); omega
      rw [ek, tz_pB n τ hg hs']
      have := jOf_lt hs'
      have := (mOf_spec hs').1
      split_ifs
      · exact ⟨_, by assumption, Or.inr (Or.inr (Or.inl rfl))⟩
      · exact ⟨_, by assumption, Or.inr (Or.inr (Or.inr rfl))⟩

theorem tz_lt (d : ℕ) {n : ℕ} {τ : ℕ → ℕ} (hf : ∀ m < n, τ (cb n m + 1) < d)
    (hb : ∀ x < 4 * n, τ x < d) : ∀ l ∈ chainP n ++ pairP n, tz n τ l < d := by
  intro l hl
  obtain ⟨m, hm, e | e | e | e⟩ := tz_cases n τ l hl <;> rw [e]
  · exact hb _ (by unfold bi; omega)
  · exact hb _ (by unfold ai; omega)
  · exact hf m hm
  · exact hb _ (by unfold bo; omega)

/-! ## The boxed side at its pinned labels -/

/-- A Z spider with zero phase on a head leg `o` and other legs all `a`. -/
theorem tZ_head (D o a : ℕ) (φ : ℕ → ℝ) (hφ : ∀ k, φ k = 0) (vs : List ℕ) (hne : vs ≠ [])
    (hv : ∀ v ∈ vs, v = a) (ha : a < D) : tZ D φ (o :: vs) = if o = a then 1 else 0 := by
  by_cases h : o = a
  · rw [if_pos h, Fca.tZ_all D a φ _ (by simp) (fun v hv' => by
      rcases mem_cons.1 hv' with rfl | hv'
      · exact h
      · exact hv v hv') ha, hφ, ex_zero]
  · rw [if_neg h]
    unfold tZ
    refine Finset.sum_eq_zero fun k _ => if_neg fun hk => h ?_
    obtain ⟨v, hv0⟩ := List.exists_mem_of_ne_nil vs hne
    rw [hk o mem_cons_self, ← hk v (mem_cons_of_mem _ hv0), hv v hv0]

theorem engD_zero (d : ℕ) {n : ℕ} (hn : 0 < n) : engD d n 0 = d := by simp [engD, hn]

/-- Copy `m`'s nodes at the pinned labels, as a function of its free labels' values. -/
noncomputable def Tm (d n : ℕ) (σ : ℕ → ℕ) (m x u v : ℕ) : ℂ :=
  (if σ (ao m) = σ (ai n m) then 1 else 0) * fou d x (σ (bi n m)) * fou d x (σ (ai n m)) *
    fou d u x * fou d v u * fou d (σ (bo m)) v *
    ((later n m).map fun j => tP (engD d n (j - m)) 1 [σ (bi n j), x] *
      tP (engD d n (j - m)) 1 [σ (ai n j), x] * tP (engD d n (j - m)) (-1) [σ (bo j), x]).prod

theorem prodVal_tz (d : ℕ) {n : ℕ} (hn : 0 < n) {τ : ℕ → ℕ}
    (hf : ∀ m < n, τ (cb n m + 1) < d) (hb : ∀ x < 4 * n, τ x < d) :
    prodVal (engE d n) (engL n).nodes (tz n τ) = ((range n).map fun m =>
      Tm d n τ m (τ (cb n m + 1)) (τ (cb n m + 8)) (τ (cb n m + 9))).prod := by
  set E := engE d n
  have eD0 : E.dimv 0 = d := engD_zero d hn
  have eP0 : E.phase 0 = fun _ => 0 := rfl
  have hz1 : ∀ {os is : List ℕ} {k : ℕ}, (∀ v ∈ (os ++ is).map (tz n τ), v = k) → os ++ is ≠ [] →
      k < d → val E (tz n τ) (nd .Z os is) = 1 := fun {os is k} hv hne hk => by
    simp only [nd, val]
    rw [eD0, eP0, ← map_append, Fca.tZ_all _ k _ _ (by simpa using hne) hv hk, ex_zero]
  have hcb : ∀ m < n, ∀ r < 11, tz n τ (cb n m + r) = if r = 0 then τ (bi n m)
      else if r = 2 ∨ r = 3 then τ (ai n m) else if 4 ≤ r ∧ r ≤ 7 then τ (cb n m + 1)
      else if r = 10 then τ (bo m) else τ (cb n m + r) :=
    fun m hm r hr => tz_chain n τ hm hr
  have lbi : ∀ j < n, bi n j < 4 * n := fun j hj => by unfold bi; omega
  have lai : ∀ j < n, ai n j < 4 * n := fun j hj => by unfold ai; omega
  have lbo : ∀ j < n, bo j < 4 * n := fun j hj => by unfold bo; omega
  have lao : ∀ j < n, ao j < 4 * n := fun j hj => by unfold ao; omega
  have hG0 : ((range n).map fun j => val E (tz n τ)
      (nd .Z [cb n j] (bi n j :: (range j).map (a0 n j)))).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (bi n j)) ?_ (by simp) (hb _ (lbi j hj))
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map, mem_range,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | ⟨k, hk, rfl⟩
    · rw [show cb n j = cb n j + 0 from rfl, hcb j hj 0 (by omega)]; rfl
    · exact tz_low n τ (lbi j hj)
    · exact tz_a0 n τ hk hj
  have hG2 : ((range n).map fun j => val E (tz n τ)
      (nd .Z ((cb n j + 4) :: (later n j).map (a2 n j)) [cb n j + 1])).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (cb n j + 1)) ?_ (by simp) (hf j hj)
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with (rfl | ⟨k, hk, rfl⟩) | rfl
    · rw [hcb j hj 4 (by omega)]; simp
    · obtain ⟨h1, h2⟩ := mem_later.1 hk; exact tz_a2 n τ h1 h2
    · rw [hcb j hj 1 (by omega)]; simp
  have hG7 : ((range n).map fun j => val E (tz n τ)
      (nd .Z [ao j] ((cb n j + 2) :: (range j).map (a7 n j)))).prod =
      ((range n).map fun j => if τ (ao j) = τ (ai n j) then (1 : ℂ) else 0).prod := by
    congr 1
    refine List.map_congr_left fun j hj => ?_
    have hj := mem_range.1 hj
    simp only [nd, val, map_cons, cons_append]
    rw [eD0, eP0, tz_low n τ (lao j hj)]
    refine tZ_head _ _ (τ (ai n j)) _ (fun _ => rfl) _ (by simp) ?_ (hb _ (lai j hj))
    intro v hv
    rcases mem_cons.1 hv with rfl | hv
    · rw [hcb j hj 2 (by omega)]; simp
    · obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hv
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hk
      exact tz_a7 n τ (mem_range.1 ht) hj
  have hG8 : ((range n).map fun j => val E (tz n τ)
      (nd .Z (g8o n j :: (later n j).map (a8 n j)) [cb n j + 6])).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (cb n j + 1)) ?_ (by simp) (hf j hj)
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with (rfl | ⟨k, hk, rfl⟩) | rfl
    · exact tz_g8o n τ hj
    · obtain ⟨h1, h2⟩ := mem_later.1 hk; exact tz_a8 n τ h1 h2
    · rw [hcb j hj 6 (by omega)]; simp
  have hG10 : ((range n).map fun j => val E (tz n τ)
      (nd .Z ((cb n j + 7) :: (later n j).map (a10 n j)) [g8o n j])).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (cb n j + 1)) ?_ (by simp) (hf j hj)
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with (rfl | ⟨k, hk, rfl⟩) | rfl
    · rw [hcb j hj 7 (by omega)]; simp
    · obtain ⟨h1, h2⟩ := mem_later.1 hk; exact tz_a10 n τ h1 h2
    · exact tz_g8o n τ hj
  have hG14 : ((range n).map fun j => val E (tz n τ)
      (nd .Z [bo j] ((cb n j + 10) :: (range j).map (a14 n j)))).prod = 1 := by
    refine List.prod_eq_one fun y hy => ?_
    obtain ⟨j, hj, rfl⟩ := mem_map.1 hy
    have hj := mem_range.1 hj
    refine hz1 (k := τ (bo j)) ?_ (by simp) (hb _ (lbo j hj))
    intro v hv
    simp only [map_append, map_cons, map_map, mem_append, mem_cons, mem_map, mem_range,
      Function.comp_def, map_nil, not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | ⟨k, hk, rfl⟩
    · exact tz_low n τ (lbo j hj)
    · rw [hcb j hj 10 (by omega)]; simp
    · exact tz_a14 n τ hk hj
  have hC : ∀ m ∈ range n, ((chainN n m).map (val E (tz n τ))).prod =
      fou d (τ (cb n m + 1)) (τ (bi n m)) * fou d (τ (cb n m + 1)) (τ (ai n m)) *
        fou d (τ (cb n m + 8)) (τ (cb n m + 1)) * fou d (τ (cb n m + 9)) (τ (cb n m + 8)) *
        fou d (τ (bo m)) (τ (cb n m + 9)) := by
    intro m hm
    have hm := mem_range.1 hm
    simp only [chainN, nd, map_cons, map_nil, prod_cons, prod_nil, mul_one, val]
    rw [show cb n m = cb n m + 0 from rfl, hcb m hm 0 (by omega), hcb m hm 1 (by omega),
      hcb m hm 2 (by omega), hcb m hm 3 (by omega), hcb m hm 4 (by omega),
      hcb m hm 5 (by omega), hcb m hm 6 (by omega), hcb m hm 7 (by omega),
      hcb m hm 8 (by omega), hcb m hm 9 (by omega), hcb m hm 10 (by omega), eD0,
      tz_low n τ (lai m hm)]
    simp only [Nat.add_zero, if_true, if_false, OfNat.ofNat_ne_zero, OfNat.zero_ne_ofNat,
      Nat.succ_ne_self, true_or, or_true]
    norm_num
    rw [Fca.tZ_all _ (τ (ai n m)) _ _ (by simp) (by simp) (hb _ (lai m hm)),
      Fca.tZ_all _ (τ (cb n m + 1)) _ _ (by simp) (by simp) (hf m hm), eP0, ex_zero]
    ring
  have hP0 : ∀ m ∈ range n, ((later n m).map fun j => val E (tz n τ)
      ⟨.ZP 0 0 1, [a0 n j m], [a2 n m j], j - m, 0⟩).prod =
      ((later n m).map fun j => tP (engD d n (j - m)) 1 [τ (bi n j), τ (cb n m + 1)]).prod := by
    intro m _
    congr 1
    refine List.map_congr_left fun j hj => ?_
    obtain ⟨h1, h2⟩ := mem_later.1 hj
    simp only [val, map_cons, map_nil, cons_append, nil_append]
    rw [tz_a0 n τ h1 h2, tz_a2 n τ h1 h2]; rfl
  have hP3 : ∀ m ∈ range n, ((later n m).map fun j => val E (tz n τ)
      ⟨.ZP 0 0 1, [a7 n j m], [a8 n m j], j - m, 0⟩).prod =
      ((later n m).map fun j => tP (engD d n (j - m)) 1 [τ (ai n j), τ (cb n m + 1)]).prod := by
    intro m _
    congr 1
    refine List.map_congr_left fun j hj => ?_
    obtain ⟨h1, h2⟩ := mem_later.1 hj
    simp only [val, map_cons, map_nil, cons_append, nil_append]
    rw [tz_a7 n τ h1 h2, tz_a8 n τ h1 h2]; rfl
  have hP6 : ∀ m ∈ range n, ((later n m).map fun j => val E (tz n τ)
      ⟨.ZP 0 0 (-1), [a14 n j m], [a10 n m j], j - m, 0⟩).prod =
      ((later n m).map fun j => tP (engD d n (j - m)) (-1) [τ (bo j), τ (cb n m + 1)]).prod := by
    intro m _
    congr 1
    refine List.map_congr_left fun j hj => ?_
    obtain ⟨h1, h2⟩ := mem_later.1 hj
    simp only [val, map_cons, map_nil, cons_append, nil_append]
    rw [tz_a14 n τ h1 h2, tz_a10 n τ h1 h2]; rfl
  simp only [engL, prodVal, map_append, prod_append, map_flatMap, prod_flatMapC, map_map,
    Function.comp_def]
  rw [hG0, hG2, hG7, hG8, hG10, hG14, List.map_congr_left hC, List.map_congr_left hP0,
    List.map_congr_left hP3, List.map_congr_left hP6]
  simp only [Fca.lprod_range, mul_one]
  simp only [← Finset.prod_mul_distrib]
  refine Finset.prod_congr rfl fun m _ => ?_
  simp only [Tm, List.prod_map_mul]
  ring

/-! ## The boxed side in closed form -/

section closed

open Fca

theorem sumOver_blocks (D : ℕ → ℕ) (c : ℕ) (G : ℕ → ℕ → ℕ → ℕ → ℂ) :
    ∀ (ms : List ℕ), ms.Nodup → ∀ σ : ℕ → ℕ,
      sumOver D (ms.flatMap fun m => [c + 11 * m + 1, c + 11 * m + 8, c + 11 * m + 9])
        (fun τ => (ms.map fun m => G m (τ (c + 11 * m + 1)) (τ (c + 11 * m + 8))
          (τ (c + 11 * m + 9))).prod) σ =
      (ms.map fun m => ∑ x ∈ Finset.range (D (c + 11 * m + 1)),
        ∑ u ∈ Finset.range (D (c + 11 * m + 8)), ∑ v ∈ Finset.range (D (c + 11 * m + 9)),
          G m x u v).prod
  | [], _, σ => by simp [sumOver]
  | m :: ms, h, σ => by
    have hm : m ∉ ms := (nodup_cons.1 h).1
    rw [flatMap_cons]
    simp only [map_cons, prod_cons]
    rw [sumOver_mul_split (Sf := [c + 11 * m + 1, c + 11 * m + 8, c + 11 * m + 9])
      (Sg := ms.flatMap fun m => [c + 11 * m + 1, c + 11 * m + 8, c + 11 * m + 9])]
    · rw [sumOver_blocks D c G ms (nodup_cons.1 h).2 σ]
      congr 1
      simp only [sumOver]
      refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun u _ =>
        Finset.sum_congr rfl fun v _ => ?_
      simp
    · intro τ τ' e
      dsimp only
      rw [e (c + 11 * m + 1) (by simp), e (c + 11 * m + 8) (by simp),
        e (c + 11 * m + 9) (by simp)]
    · intro τ τ' e
      dsimp only
      congr 1
      refine List.map_congr_left fun k hk => ?_
      rw [e (c + 11 * k + 1) (mem_flatMap.2 ⟨k, hk, by simp⟩),
        e (c + 11 * k + 8) (mem_flatMap.2 ⟨k, hk, by simp⟩),
        e (c + 11 * k + 9) (mem_flatMap.2 ⟨k, hk, by simp⟩)]
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

theorem rb_ge (n : ℕ) : 16 * n + 6 * tri n ≤ rb n := le_rfl

theorem labDim_low (d : ℕ) {n x : ℕ} (hn : 0 < n) (hx : x < rb n) : labDim (engE d n) x = d := by
  unfold labDim
  rw [show (engE d n).labs = engLabs n from rfl, Fca.lookup_none _ _ (by
    simp only [engLabs, map_map, mem_map, mem_range, Function.comp_def, not_exists, not_and]
    intro k _ e; omega)]
  exact engD_zero d hn

theorem Tm_congr (d n : ℕ) {σ τ : ℕ → ℕ} (h : ∀ x < 4 * n, τ x = σ x) {m : ℕ} (hm : m < n)
    (x u v : ℕ) : Tm d n τ m x u v = Tm d n σ m x u v := by
  unfold Tm
  rw [h (ao m) (by unfold ao; omega), h (ai n m) (by unfold ai; omega),
    h (bi n m) (by unfold bi; omega), h (bo m) (by unfold bo; omega)]
  congr 1
  refine congrArg _ (List.map_congr_left fun j hj => ?_)
  obtain ⟨_, h2⟩ := mem_later.1 hj
  rw [h (bi n j) (by unfold bi; omega), h (ai n j) (by unfold ai; omega),
    h (bo j) (by unfold bo; omega)]

/-- Copy `m`'s summand at the Fourier value `x`, the `F³` chain summed. -/
noncomputable def Ec (d n : ℕ) (σ : ℕ → ℕ) (m x : ℕ) : ℂ :=
  (if σ (ao m) = σ (ai n m) then 1 else 0) * fou d x (σ (bi n m)) * fou d x (σ (ai n m)) *
    (starRingEnd ℂ) (fou d (σ (bo m)) x) *
    ((later n m).map fun j => tP (engD d n (j - m)) 1 [σ (bi n j), x] *
      tP (engD d n (j - m)) 1 [σ (ai n j), x] * tP (engD d n (j - m)) (-1) [σ (bo j), x]).prod

theorem frees_eq (n : ℕ) :
    frees n = (range n).flatMap fun m =>
      [4 * n + 11 * m + 1, 4 * n + 11 * m + 8, 4 * n + 11 * m + 9] := by
  simp [frees]

/-- The boxed side at `n ≥ 1` copies: the product over copies of the Fourier sums. -/
theorem eval_engL (d : ℕ) (hd : 0 < d) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ)
    (hb : ∀ x < 4 * n, σ x < d) :
    eval (engE d n) (engL n) σ = ∏ m ∈ Finset.range n, ∑ x ∈ Finset.range d, Ec d n σ m x := by
  have hrb : ∀ x, x < 16 * n + 6 * tri n → labDim (engE d n) x = d := fun x hx =>
    labDim_low d hn hx
  unfold eval
  rw [sumOver_perm _ (int_perm n), sumOver_append]
  rw [sumOver_congr _ (frees n) _ (fun τ => ((range n).map fun m =>
    Tm d n σ m (τ (cb n m + 1)) (τ (cb n m + 8)) (τ (cb n m + 9))).prod) σ ?_]
  · rw [frees_eq]
    refine (sumOver_blocks (labDim (engE d n)) (4 * n) (Tm d n σ) _ nodup_range σ).trans ?_
    rw [lprod_range]
    refine Finset.prod_congr rfl fun m hm => ?_
    have hm := Finset.mem_range.1 hm
    rw [hrb (4 * n + 11 * m + 1) (by omega), hrb (4 * n + 11 * m + 8) (by omega),
      hrb (4 * n + 11 * m + 9) (by omega)]
    refine Finset.sum_congr rfl fun x _ => ?_
    have h3 := fou_three d (σ (bo m)) x hd
    rw [Finset.sum_comm, Ec, ← h3]
    simp only [Tm, Finset.mul_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl fun v _ => Finset.sum_congr rfl fun u _ => ?_
    ring
  · intro τ hτ hr
    have hlow : ∀ x < 4 * n, τ x = σ x := fun x hx => hτ x fun h => by
      rw [frees_eq] at h
      obtain ⟨k, _, hk⟩ := mem_flatMap.1 h
      simp only [mem_cons, not_mem_nil, or_false] at hk
      rcases hk with e | e | e <;> omega
    have hf : ∀ m < n, τ (cb n m + 1) < d := fun m hm => by
      have := hr (cb n m + 1)
        (by rw [frees_eq]; exact mem_flatMap.2 ⟨m, mem_range.2 hm, by simp [cb]⟩)
      rwa [hrb (cb n m + 1) (by unfold cb; omega)] at this
    have hbτ : ∀ x < 4 * n, τ x < d := fun x hx => by rw [hlow x hx]; exact hb x hx
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
          rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega
        · simp only [pairP, mem_map, mem_range] at hp
          obtain ⟨k, hk, rfl⟩ := hp
          omega
    · intro τ' h0 _ hne l hl
      exact engL_support d hne h0 l hl

end closed

/-! ## The spec's roles -/

theorem pofs_lt {n k : ℕ} (hk : k < 4 * n + 5) : pofs n k < 4 * n + 5 := by
  unfold pofs
  split_ifs with h
  · subst h; interval_cases k <;> decide
  · exact hk

theorem rofs_pofs {n k : ℕ} (hk : k < 4 * n + 5) : rofs n (pofs n k) = k := by
  unfold pofs rofs
  split_ifs with h
  · subst h; interval_cases k <;> decide
  · rfl

theorem pofs_rofs {n o : ℕ} (ho : o < 4 * n + 5) : pofs n (rofs n o) = o := by
  unfold pofs rofs
  split_ifs with h
  · subst h; interval_cases o <;> decide
  · rfl

theorem rofs_lt {n o : ℕ} (ho : o < 4 * n + 5) : rofs n o < 4 * n + 5 := by
  unfold rofs
  split_ifs with h
  · subst h; interval_cases o <;> decide
  · exact ho

theorem rl_inj {n k k' : ℕ} (hk : k < 4 * n + 5) (hk' : k' < 4 * n + 5) (h : rl n k = rl n k') :
    k = k' := by
  unfold rl at h
  rw [← rofs_pofs hk, ← rofs_pofs hk', show pofs n k = pofs n k' by omega]

theorem mem_engR_int {n x : ℕ} : x ∈ (engR n).int ↔ rb n ≤ x ∧ x < rb n + (4 * n + 5) := by
  simp only [engR, mem_map, mem_range]
  constructor
  · rintro ⟨k, hk, rfl⟩; omega
  · intro h; exact ⟨x - rb n, by omega, by omega⟩

theorem engR_int_nodup (n : ℕ) : (engR n).int.Nodup :=
  (nodup_range).map fun x y h => by simpa using h

theorem rl_int {n k : ℕ} (hk : k < 4 * n + 5) : rl n k ∈ (engR n).int := by
  have := pofs_lt hk
  exact mem_engR_int.2 ⟨by unfold rl; omega, by unfold rl; omega⟩

theorem int_rl {n x : ℕ} (hx : x ∈ (engR n).int) :
    rofs n (x - rb n) < 4 * n + 5 ∧ x = rl n (rofs n (x - rb n)) := by
  obtain ⟨h1, h2⟩ := mem_engR_int.1 hx
  refine ⟨rofs_lt (by omega), ?_⟩
  unfold rl; rw [pofs_rofs (by omega)]; omega

theorem edim_le {n k : ℕ} (hk : k < 4 * n + 5) : edim n k ≤ n := by
  unfold edim; split_ifs <;> omega

theorem labDim_rl (d : ℕ) {n k : ℕ} (hk : k < 4 * n + 5) :
    labDim (engE d n) (rl n k) = d ^ edim n k := by
  unfold labDim
  rw [show (engE d n).labs = engLabs n from rfl, engLabs, rl,
    lookup_shift (fun o => dI n (edim n (rofs n o))) _ _ _ (pofs_lt hk)]
  simp only [Option.getD_some, rofs_pofs hk]
  exact engD_dI d (edim_le hk)

/-! ## The spec in closed form -/

/-- The value of the first `t` digits at the labels `c, c + 2, …`, the first most significant. -/
def evd (d : ℕ) (σ : ℕ → ℕ) (c : ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => evd d σ c t * d + σ (c + 2 * t)

theorem evd_lt {d : ℕ} {σ : ℕ → ℕ} {c n : ℕ} (h : ∀ t < n, σ (c + 2 * t) < d) :
    ∀ t ≤ n, evd d σ c t < d ^ t
  | 0, _ => by simp [evd]
  | t + 1, ht => by
    have ih := evd_lt h t (by omega)
    have hdig := h t (by omega)
    simp only [evd, pow_succ]
    nlinarith

theorem evd_congr {d : ℕ} {σ τ : ℕ → ℕ} {c n : ℕ} (h : ∀ t < n, τ (c + 2 * t) = σ (c + 2 * t)) :
    ∀ t ≤ n, evd d τ c t = evd d σ c t
  | 0, _ => rfl
  | t + 1, ht => by simp only [evd, evd_congr h t (by omega), h t (by omega)]

/-- The pinned value of the spec's role `k`. -/
def pv (d : ℕ) (σ : ℕ → ℕ) (n k : ℕ) : ℕ :=
  if k < 4 then 0 else if k = 4 ∨ k = 6 then evd d σ (2 * n) n
  else if k = 5 then evd d σ 0 n else if k = 7 then evd d σ (2 * n + 1) n
  else if k = 8 then evd d σ 1 n
  else if k < 9 + 2 * (n - 1) then evd d σ (2 * n + (k - 9) % 2) ((k - 9) / 2 + 1)
  else evd d σ ((k - 9 - 2 * (n - 1)) % 2) ((k - 9 - 2 * (n - 1)) / 2 + 1)

/-- `σ` with the spec's summed labels at their pinned values. -/
def tauE (d : ℕ) (σ : ℕ → ℕ) (n x : ℕ) : ℕ :=
  if rb n ≤ x ∧ x < rb n + (4 * n + 5) then pv d σ n (rofs n (x - rb n)) else σ x

theorem tauE_rl (d : ℕ) (σ : ℕ → ℕ) {n k : ℕ} (hk : k < 4 * n + 5) :
    tauE d σ n (rl n k) = pv d σ n k := by
  have := pofs_lt hk
  unfold tauE
  rw [if_pos (by unfold rl; omega), show rl n k - rb n = pofs n k by unfold rl; omega,
    rofs_pofs hk]

/-- The role of the binder value of chain `c` after step `t`. -/
def rbv (n c t : ℕ) : ℕ := if t + 1 = n then 4 + 3 * c else 9 + 2 * t + c

/-- The role of the splitter value of chain `c` of the top `t` digits. -/
def rsv (n c t : ℕ) : ℕ :=
  if t = 0 then 2 * c + 1 else if t = n then 5 + 3 * c else 9 + 2 * (n - 1) + 2 * (t - 1) + c

theorem bv_eq (n c t : ℕ) : bv n c t = rl n (rbv n c t) := by
  unfold bv rbv; split_ifs <;> rfl

theorem sv_eq (n c t : ℕ) : sv n c t = rl n (rsv n c t) := by
  unfold sv rsv; split_ifs <;> rfl

theorem rbv_lt {n c t : ℕ} (hc : c < 2) (ht : t < n) : rbv n c t < 4 * n + 5 := by
  unfold rbv; split_ifs <;> omega

theorem rsv_lt {n c t : ℕ} (hc : c < 2) (ht : t ≤ n) : rsv n c t < 4 * n + 5 := by
  unfold rsv; split_ifs <;> omega

theorem pv_rbv (d : ℕ) (σ : ℕ → ℕ) {n c t : ℕ} (hc : c < 2) (ht : t < n) :
    pv d σ n (rbv n c t) = evd d σ (2 * n + c) (t + 1) := by
  unfold pv rbv
  interval_cases c <;> split_ifs <;> first | omega | (congr 1 <;> omega) | rfl

theorem pv_rsv (d : ℕ) (σ : ℕ → ℕ) {n c t : ℕ} (hc : c < 2) (ht : t ≤ n) :
    pv d σ n (rsv n c t) = evd d σ c t := by
  unfold pv rsv
  interval_cases c <;> split_ifs <;> first | omega | (subst_vars; rfl) | (congr 1 <;> omega) | rfl

theorem pv_six (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : pv d σ n 6 = evd d σ (2 * n) n := by
  simp [pv]

theorem pv_four (d : ℕ) (σ : ℕ → ℕ) (n : ℕ) : pv d σ n 4 = evd d σ (2 * n) n := by
  simp [pv]

/-- Every role is a head, an end, the copy leg, a binder value or a splitter value. -/
theorem role_cases {n k : ℕ} (hn : 0 < n) (hk : k < 4 * n + 5) :
    k = 6 ∨ (∃ c < 2, ∃ t < n, k = rbv n c t) ∨ (∃ c < 2, ∃ t ≤ n, k = rsv n c t) ∨
      (∃ c < 2, k = 2 * c) := by
  rcases (show k < 4 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ (9 ≤ k ∧ k < 9 + 2 * (n - 1)) ∨
      9 + 2 * (n - 1) ≤ k by omega) with h | h | h | h | h | h | h | h
  · rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 by omega) with rfl | rfl | rfl | rfl
    · exact Or.inr (Or.inr (Or.inr ⟨0, by omega, rfl⟩))
    · exact Or.inr (Or.inr (Or.inl ⟨0, by omega, 0, by omega, by simp [rsv]⟩))
    · exact Or.inr (Or.inr (Or.inr ⟨1, by omega, rfl⟩))
    · exact Or.inr (Or.inr (Or.inl ⟨1, by omega, 0, by omega, by simp [rsv]⟩))
  · exact Or.inr (Or.inl ⟨0, by omega, n - 1, by omega, by unfold rbv; rw [if_pos (by omega)]; omega⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨0, by omega, n, le_rfl, by unfold rsv; rw [if_neg (by omega), if_pos rfl]; omega⟩))
  · exact Or.inl h
  · exact Or.inr (Or.inl ⟨1, by omega, n - 1, by omega, by unfold rbv; rw [if_pos (by omega)]; omega⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨1, by omega, n, le_rfl, by unfold rsv; rw [if_neg (by omega), if_pos rfl]; omega⟩))
  · refine Or.inr (Or.inl ⟨(k - 9) % 2, by omega, (k - 9) / 2, by omega, ?_⟩)
    unfold rbv; rw [if_neg (by omega)]; omega
  · refine Or.inr (Or.inr (Or.inl ⟨(k - 9 - 2 * (n - 1)) % 2, by omega,
      (k - 9 - 2 * (n - 1)) / 2 + 1, by omega, ?_⟩))
    unfold rsv; rw [if_neg (by omega), if_neg (by omega)]; omega

theorem mem_engR {n : ℕ} (hn : 0 < n) {x : Node}
    (h : x = hN n 0 ∨ x = hN n 1 ∨ x = eN n 0 ∨ x = eN n 1 ∨ x = zN n ∨ x = xN n ∨
      ∃ t < n, x = bN n 0 t ∨ x = bN n 1 t ∨ x = sN n 0 t ∨ x = sN n 1 t) :
    x ∈ (engR n).nodes := by
  unfold engR
  rw [if_neg hn.ne']
  split_ifs with h1
  · subst h1
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | ⟨t, ht, h⟩
    all_goals try simp
    obtain rfl : t = 0 := by omega
    rcases h with rfl | rfl | rfl | rfl <;> simp
  · simp only [mem_append, mem_cons, mem_flatMap, mem_range, not_mem_nil, or_false]
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | ⟨t, ht, rfl | rfl | rfl | rfl⟩
    all_goals first
      | exact Or.inl (Or.inr ⟨t, ht, Or.inl rfl⟩)
      | exact Or.inl (Or.inr ⟨t, ht, Or.inr rfl⟩)
      | exact Or.inr ⟨t, ht, Or.inl rfl⟩
      | exact Or.inr ⟨t, ht, Or.inr rfl⟩
      | simp

theorem prodVal_engR (E : Env) {n : ℕ} (hn : 0 < n) (τ : ℕ → ℕ) :
    prodVal E (engR n).nodes τ = val E τ (hN n 0) * val E τ (eN n 0) * val E τ (hN n 1) *
      val E τ (eN n 1) * val E τ (zN n) * val E τ (xN n) *
      ((range n).map fun t => val E τ (bN n 0 t) * val E τ (bN n 1 t)).prod *
      ((range n).map fun t => val E τ (sN n 0 t) * val E τ (sN n 1 t)).prod := by
  unfold engR prodVal
  rw [if_neg hn.ne']
  split_ifs with h1
  · subst h1
    simp only [map_cons, map_nil, prod_cons, prod_nil, range_one]
    ring
  · simp only [map_append, prod_append, map_cons, map_nil, prod_cons, prod_nil, map_flatMap,
      prod_flatMapC]
    ring

theorem engR_support (d : ℕ) {n : ℕ} (hn : 0 < n) {τ : ℕ → ℕ}
    (h : prodVal (engE d n) (engR n).nodes τ ≠ 0) :
    (∀ c < 2, τ (rl n (2 * c)) = 0) ∧ (∀ c < 2, ∀ t < n, τ (bv n c t) = evd d τ (2 * n + c) (t + 1)) ∧
      (∀ c < 2, ∀ t ≤ n, τ (sv n c t) = evd d τ c t) ∧ τ (rl n 6) = τ (rl n 4) := by
  have hv := Fca.prodVal_ne h
  have eD1 : (engE d n).dimv n = 1 := by simp [engE, engD]
  have eD0 : (engE d n).dimv 0 = d := engD_zero d hn
  have hH : ∀ c < 2, τ (rl n (2 * c)) = 0 := fun c hc => by
    have h1 := hv _ (mem_engR hn (x := hN n c) (by
      rcases (show c = 0 ∨ c = 1 by omega) with rfl | rfl
      · exact Or.inl rfl
      · exact Or.inr (Or.inl rfl)))
    simp only [hN, val, map_cons, map_nil, append_nil] at h1
    rw [eD1, Fca.tZ_one _ ((engE d n).phase 0) (fun _ => rfl)] at h1
    simpa using h1
  have hE : ∀ c < 2, τ (rl n (2 * c + 1)) = 0 := fun c hc => by
    have h1 := hv _ (mem_engR hn (x := eN n c) (by
      rcases (show c = 0 ∨ c = 1 by omega) with rfl | rfl
      · exact Or.inr (Or.inr (Or.inl rfl))
      · exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
    simp only [eN, val, map_cons, map_nil, nil_append] at h1
    rw [eD1, Fca.tZ_one _ ((engE d n).phase 0) (fun _ => rfl)] at h1
    simpa using h1
  have hB : ∀ c < 2, ∀ t < n, τ (bv n c t) = τ (bw n c t) * d + τ (2 * n + 2 * t + c) := by
    intro c hc t ht
    have h1 := hv _ (mem_engR hn (x := bN n c t) (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
      (Or.inr ⟨t, ht, by
        rcases (show c = 0 ∨ c = 1 by omega) with rfl | rfl
        · exact Or.inl rfl
        · exact Or.inr (Or.inl rfl)⟩)))))))
    simp only [bN, val, map_cons, map_nil] at h1
    have e := Fca.tB_eq h1
    rwa [eD0] at e
  have hS : ∀ c < 2, ∀ t < n, τ (sv n c (t + 1)) = τ (sv n c t) * d + τ (2 * t + c) := by
    intro c hc t ht
    have h1 := hv _ (mem_engR hn (x := sN n c t) (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
      (Or.inr ⟨t, ht, by
        rcases (show c = 0 ∨ c = 1 by omega) with rfl | rfl
        · exact Or.inr (Or.inr (Or.inl rfl))
        · exact Or.inr (Or.inr (Or.inr rfl))⟩)))))))
    simp only [sN, val, map_cons, map_nil] at h1
    have e := Fca.tB_eq h1
    rwa [eD0] at e
  refine ⟨hH, fun c hc t ht => ?_, fun c hc t ht => ?_, ?_⟩
  · induction t with
    | zero =>
      rw [hB c hc 0 ht]
      simp only [bw, if_true, hH c hc, zero_mul, zero_add, evd, Nat.add_zero, mul_zero]
    | succ t ih =>
      have e : bw n c (t + 1) = bv n c t := by simp [bw]
      rw [hB c hc _ ht, e, ih (by omega)]
      simp only [evd]
      congr 2; ring
  · induction t with
    | zero => simp only [sv, if_true, evd]; exact hE c hc
    | succ t ih =>
      rw [hS c hc t (by omega), ih (by omega)]
      simp only [evd]
      congr 2; ring
  · exact (Z_eq _ τ (zN n) rfl (hv _ (mem_engR hn (by simp))) (a := rl n 6) (b := rl n 4)
      (by simp [zN, Node.labels]) (by simp [zN, Node.labels]))

/-- The spec at `n ≥ 1` digits: `1 / √(d ^ n)`, the sum mod `d ^ n` and the copy of the input
digit values. -/
theorem eval_engR (d : ℕ) (hd : 0 < d) {n : ℕ} (hn : 0 < n) (σ : ℕ → ℕ)
    (hb : ∀ x < 4 * n, σ x < d) :
    eval (engE d n) (engR n) σ = (rt (d ^ n))⁻¹ *
      (if evd d σ 1 n % d ^ n = (evd d σ (2 * n + 1) n + evd d σ (2 * n) n) % d ^ n then 1
        else 0) * (if evd d σ 0 n = evd d σ (2 * n) n then 1 else 0) := by
  have hrb : 4 * n ≤ rb n := by unfold rb; omega
  have hoff : ∀ x, x < 4 * n → tauE d σ n x = σ x := fun x hx => by
    unfold tauE; rw [if_neg (by omega)]
  have dg : ∀ c < 2, ∀ c' ≤ 1, ∀ t < n, σ (c' * (2 * n) + c + 2 * t) < d := fun c hc c' hc' t ht =>
    hb _ (by interval_cases c' <;> omega)
  have ev0 : ∀ c < 2, ∀ t ≤ n, evd d σ c t < d ^ t := fun c hc =>
    evd_lt fun t ht => hb _ (by omega)
  have ev1 : ∀ c < 2, ∀ t ≤ n, evd d σ (2 * n + c) t < d ^ t := fun c hc =>
    evd_lt fun t ht => hb _ (by omega)
  unfold eval
  rw [Fca.sumOver_single (labDim (engE d n)) _ (engR_int_nodup n) _ σ (tauE d σ n) ?h0 ?hr ?hs]
  case h0 =>
    intro x hx
    unfold tauE
    rw [if_neg fun h => hx (mem_engR_int.2 h)]
  case hr =>
    intro l hl
    obtain ⟨hk, e⟩ := int_rl hl
    rw [e, labDim_rl d hk, tauE_rl d σ hk]
    set k := rofs n (l - rb n)
    rcases role_cases hn hk with h6 | ⟨c, hc, t, ht, hkt⟩ | ⟨c, hc, t, ht, hkt⟩ | ⟨c, hc, hkt⟩
    · rw [h6, pv_six]; simp only [edim]; norm_num; exact ev1 0 (by omega) n le_rfl
    · rw [hkt, pv_rbv d σ hc ht]
      have : edim n (rbv n c t) = t + 1 := by
        unfold edim rbv; interval_cases c <;> split_ifs <;> omega
      rw [this]; exact ev1 c hc _ (by omega)
    · rw [hkt, pv_rsv d σ hc ht]
      have : edim n (rsv n c t) = t := by
        unfold edim rsv; interval_cases c <;> split_ifs <;> omega
      rw [this]; exact ev0 c hc _ ht
    · rw [hkt]; simp only [pv, edim]; interval_cases c <;> simp
  case hs =>
    intro τ hτ _ hne l hl
    obtain ⟨s0, s1, s2, s3⟩ := engR_support d hn hne
    have dig : ∀ t, t < 4 * n → τ t = σ t := fun t ht =>
      hτ t fun h => by have := (mem_engR_int.1 h).1; omega
    have eA : ∀ c < 2, ∀ t ≤ n, evd d τ (2 * n + c) t = evd d σ (2 * n + c) t := fun c hc =>
      evd_congr fun t ht => dig _ (by omega)
    have eB : ∀ c < 2, ∀ t ≤ n, evd d τ c t = evd d σ c t := fun c hc =>
      evd_congr fun t ht => dig _ (by omega)
    obtain ⟨hk, e⟩ := int_rl hl
    rw [e, tauE_rl d σ hk]
    set k := rofs n (l - rb n)
    rcases role_cases hn hk with h6 | ⟨c, hc, t, ht, hkt⟩ | ⟨c, hc, t, ht, hkt⟩ | ⟨c, hc, hkt⟩
    · rw [h6, s3, pv_six, show rl n 4 = bv n 0 (n - 1) by unfold bv; rw [if_pos (by omega)],
        s1 0 (by omega) _ (by omega), show n - 1 + 1 = n by omega, Nat.add_zero,
        show evd d τ (2 * n) n = evd d σ (2 * n) n by simpa using eA 0 (by omega) n le_rfl]
    · rw [hkt, ← bv_eq, s1 c hc t ht, pv_rbv d σ hc ht, eA c hc _ (by omega)]
    · rw [hkt, ← sv_eq, s2 c hc t ht, pv_rsv d σ hc ht, eB c hc _ ht]
    · rw [hkt, s0 c hc]; simp only [pv]; interval_cases c <;> simp
  have eD1 : (engE d n).dimv n = 1 := by simp [engE, engD]
  have eD0 : (engE d n).dimv 0 = d := engD_zero d hn
  have eDn : (engE d n).dimv (n - 1) = d ^ n := by
    have := engD_dI d (le_refl n); rwa [dI, if_neg hn.ne'] at this
  have eP : ∀ p, (engE d n).phase p = fun _ => 0 := fun p => rfl
  have t4 : tauE d σ n (rl n 4) = evd d σ (2 * n) n := by rw [tauE_rl d σ (by omega), pv_four]
  have t6 : tauE d σ n (rl n 6) = evd d σ (2 * n) n := by rw [tauE_rl d σ (by omega), pv_six]
  have t5 : tauE d σ n (rl n 5) = evd d σ 0 n := by
    rw [tauE_rl d σ (by omega), show 5 = rsv n 0 n by simp [rsv]; omega,
      pv_rsv d σ (by omega) le_rfl]
  have t7 : tauE d σ n (rl n 7) = evd d σ (2 * n + 1) n := by
    rw [tauE_rl d σ (by omega), show 7 = rbv n 1 (n - 1) by simp [rbv]; omega,
      pv_rbv d σ (by omega) (by omega), show n - 1 + 1 = n by omega]
  have t8 : tauE d σ n (rl n 8) = evd d σ 1 n := by
    rw [tauE_rl d σ (by omega), show 8 = rsv n 1 n by simp [rsv]; omega,
      pv_rsv d σ (by omega) le_rfl]
  have hH : ∀ c < 2, val (engE d n) (tauE d σ n) (hN n c) = 1 := fun c hc => by
    simp only [hN, val, map_cons, map_nil, append_nil]
    rw [eD1, eP, Fca.tZ_one _ _ (fun _ => rfl), tauE_rl d σ (by omega)]
    simp only [pv]; interval_cases c <;> simp
  have hE : ∀ c < 2, val (engE d n) (tauE d σ n) (eN n c) = 1 := fun c hc => by
    simp only [eN, val, map_cons, map_nil, nil_append]
    rw [eD1, eP, Fca.tZ_one _ _ (fun _ => rfl), tauE_rl d σ (by omega)]
    simp only [pv]; interval_cases c <;> simp
  have hZ : val (engE d n) (tauE d σ n) (zN n) =
      if evd d σ 0 n = evd d σ (2 * n) n then 1 else 0 := by
    simp only [zN, val, map_cons, map_nil, cons_append, nil_append]
    rw [eDn, eP, t4, t5, t6]
    exact Draper.tZ_baa _ _ _ _ (fun _ => rfl) (ev1 0 (by omega) n le_rfl)
  have hX : val (engE d n) (tauE d σ n) (xN n) = (rt (d ^ n))⁻¹ *
      (if evd d σ 1 n % d ^ n = (evd d σ (2 * n + 1) n + evd d σ (2 * n) n) % d ^ n then 1
        else 0) := by
    simp only [xN, val, map_cons, map_nil]
    rw [eDn, eP, t6, t7, t8]
    exact Draper.tX_add _ (pow_pos hd n) _ (fun _ => rfl) _ _ _
  have hBs : ((range n).map fun t => val (engE d n) (tauE d σ n) (bN n 0 t) *
      val (engE d n) (tauE d σ n) (bN n 1 t)).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hx
    have ht := mem_range.1 ht
    have one : ∀ c < 2, val (engE d n) (tauE d σ n) (bN n c t) = 1 := fun c hc => by
      simp only [bN, val, map_cons, map_nil]
      rw [show (engE d n).dimv (dI n t) = d ^ t from engD_dI d ht.le, eD0, bv_eq,
        tauE_rl d σ (rbv_lt hc ht), pv_rbv d σ hc ht, hoff (2 * n + 2 * t + c) (by omega)]
      have hw : tauE d σ n (bw n c t) = evd d σ (2 * n + c) t := by
        unfold bw
        split_ifs with h0
        · subst h0; rw [tauE_rl d σ (by omega)]; simp only [pv, evd]; interval_cases c <;> simp
        · rw [bv_eq, tauE_rl d σ (rbv_lt hc (by omega)), pv_rbv d σ hc (by omega),
            show t - 1 + 1 = t by omega]
      rw [hw]
      unfold tB
      rw [if_pos ⟨ev1 c hc t ht.le, hb _ (by omega), by simp only [evd]; ring_nf⟩]
    rw [one 0 (by omega), one 1 (by omega), mul_one]
  have hSs : ((range n).map fun t => val (engE d n) (tauE d σ n) (sN n 0 t) *
      val (engE d n) (tauE d σ n) (sN n 1 t)).prod = 1 := by
    refine List.prod_eq_one fun x hx => ?_
    obtain ⟨t, ht, rfl⟩ := mem_map.1 hx
    have ht := mem_range.1 ht
    have one : ∀ c < 2, val (engE d n) (tauE d σ n) (sN n c t) = 1 := fun c hc => by
      simp only [sN, val, map_cons, map_nil]
      rw [show (engE d n).dimv (dI n t) = d ^ t from engD_dI d ht.le, eD0, sv_eq, sv_eq,
        tauE_rl d σ (rsv_lt hc ht.le), tauE_rl d σ (rsv_lt hc (by omega)),
        pv_rsv d σ hc ht.le, pv_rsv d σ hc (by omega), hoff (2 * t + c) (by omega)]
      unfold tB
      rw [if_pos ⟨ev0 c hc t ht.le, hb _ (by omega), by simp only [evd]; ring_nf⟩]
    rw [one 0 (by omega), one 1 (by omega), mul_one]
  rw [prodVal_engR _ hn, hH 0 (by omega), hH 1 (by omega), hE 0 (by omega), hE 1 (by omega), hZ,
    hX, hBs, hSs]
  ring

/-! ## The adder arithmetic over all copies -/

section arith

open Draper

/-- Copy `i`'s summand at the Fourier value `x`, its pairs as pair phases on the digit labels. -/
noncomputable def Ce (d : ℕ) (σ : ℕ → ℕ) (i x : ℕ) : ℂ :=
  fou d x (σ (sl i 1)) * fou d x (σ (sl i 0)) * (starRingEnd ℂ) (fou d (σ (sl i 3)) x) *
    ((range i).map fun k => tP (pD d i k) 1 [σ (sl k 1), x] * tP (pD d i k) 1 [σ (sl k 0), x] *
      tP (pD d i k) (-1) [σ (sl k 3), x]).prod

theorem Ce_sum (d : ℕ) (σ : ℕ → ℕ) (i : ℕ) :
    ∑ x ∈ Finset.range d, Ce d σ i x = SumC d i (tauG i σ) := by
  unfold SumC
  refine Finset.sum_congr rfl fun x _ => ?_
  have o : ∀ s < 4, tauG i σ (sl i s) = σ (sl i s) := fun s hs => tauG_off (dig_notin_cut hs)
  rw [Ce, o 0 (by omega), o 1 (by omega), o 3 (by omega)]
  congr 1
  unfold pairsF
  congr 1
  refine List.map_congr_left fun k hk => ?_
  have hk := mem_range.1 hk
  rw [tauG_pend hk, tauG_pendO hk, tauG_pendA hk]

theorem rt_pow (d : ℕ) : ∀ n, ((rt d)⁻¹) ^ n = (rt (d ^ n))⁻¹
  | 0 => by simp [rt]
  | n + 1 => by rw [pow_succ, rt_pow d n, rt_succ, mul_inv]

/-- Every copy's copy factor and Fourier sum, multiplied: the spec's factor. -/
theorem prod_Ce (d : ℕ) (hd : 0 < d) (σ : ℕ → ℕ) : ∀ n, (∀ s < 4, ∀ i < n, σ (sl i s) < d) →
    ∏ i ∈ Finset.range n, (aeqC i σ * ∑ x ∈ Finset.range d, Ce d σ i x) =
      (rt (d ^ n))⁻¹ * IndS d σ n
  | 0, _ => by simp [IndS, topv, rt]
  | n + 1, hs => by
    rw [Finset.prod_range_succ, prod_Ce d hd σ n fun s hs' i hi => hs s hs' i (by omega),
      Ce_sum, ← rt_pow, ← rt_pow]
    have h := arith d hd n σ hs
    linear_combination ((rt d)⁻¹) ^ n * h

end arith

/-! ## The engine's copies as Draper's copies -/

section bridge

open Draper

/-- The engine's boundary labels at Draper's digit labels: Draper's copy `k` is the engine's
copy `n - 1 - k`. -/
def psiL (n : ℕ) : List (ℕ × ℕ) :=
  (range n).flatMap fun k => [(sl k 0, ai n (n - 1 - k)), (sl k 1, bi n (n - 1 - k)),
    (sl k 2, ao (n - 1 - k)), (sl k 3, bo (n - 1 - k))]

/-- `σ` read at Draper's digit labels. -/
def sigF (n : ℕ) (σ : ℕ → ℕ) (y : ℕ) : ℕ := σ (((psiL n).lookup y).getD 0)

theorem psiL_nodup (n : ℕ) : ((psiL n).map Prod.fst).Nodup := by
  have e : (psiL n).map Prod.fst = (range n).flatMap fun k => [sl k 0, sl k 1, sl k 2, sl k 3] := by
    simp [psiL, map_flatMap]
  rw [e, nodup_flatMap]
  refine ⟨fun k _ => by simp [sl], ?_⟩
  refine (pairwise_lt_range (n := n)).imp fun {a b} hab => List.disjoint_left.2 fun x hx hx' => ?_
  simp only [mem_cons, not_mem_nil, or_false] at hx hx'
  rcases hx with rfl | rfl | rfl | rfl <;> rcases hx' with e | e | e | e <;>
    have := (sl_inj (by omega) (by omega) e).1 <;> omega

theorem sigF_mem (n : ℕ) (σ : ℕ → ℕ) {k s v : ℕ} (h : (sl k s, v) ∈ psiL n) :
    sigF n σ (sl k s) = σ v := by
  rw [sigF, Fca.lookup_of_mem _ (psiL_nodup n) h]; rfl

theorem sigF_0 (n : ℕ) (σ : ℕ → ℕ) {k : ℕ} (hk : k < n) :
    sigF n σ (sl k 0) = σ (ai n (n - 1 - k)) :=
  sigF_mem n σ (mem_flatMap.2 ⟨k, mem_range.2 hk, by simp⟩)

theorem sigF_1 (n : ℕ) (σ : ℕ → ℕ) {k : ℕ} (hk : k < n) :
    sigF n σ (sl k 1) = σ (bi n (n - 1 - k)) :=
  sigF_mem n σ (mem_flatMap.2 ⟨k, mem_range.2 hk, by simp⟩)

theorem sigF_2 (n : ℕ) (σ : ℕ → ℕ) {k : ℕ} (hk : k < n) :
    sigF n σ (sl k 2) = σ (ao (n - 1 - k)) :=
  sigF_mem n σ (mem_flatMap.2 ⟨k, mem_range.2 hk, by simp⟩)

theorem sigF_3 (n : ℕ) (σ : ℕ → ℕ) {k : ℕ} (hk : k < n) :
    sigF n σ (sl k 3) = σ (bo (n - 1 - k)) :=
  sigF_mem n σ (mem_flatMap.2 ⟨k, mem_range.2 hk, by simp⟩)

theorem Ec_Ce (d : ℕ) {n m : ℕ} (hm : m < n) (σ : ℕ → ℕ) (x : ℕ) :
    Ec d n σ m x = aeqC (n - 1 - m) (sigF n σ) * Ce d (sigF n σ) (n - 1 - m) x := by
  have hp : ((later n m).map fun j => tP (engD d n (j - m)) 1 [σ (bi n j), x] *
      tP (engD d n (j - m)) 1 [σ (ai n j), x] * tP (engD d n (j - m)) (-1) [σ (bo j), x]).prod =
      ((range (n - 1 - m)).map fun k => tP (pD d (n - 1 - m) k) 1 [sigF n σ (sl k 1), x] *
        tP (pD d (n - 1 - m) k) 1 [sigF n σ (sl k 0), x] *
        tP (pD d (n - 1 - m) k) (-1) [sigF n σ (sl k 3), x]).prod := by
    unfold later
    rw [map_map, Fca.lprod_range, Fca.lprod_range,
      show n - m - 1 = n - 1 - m by omega, ← Finset.prod_range_reflect _ (n - 1 - m)]
    refine Finset.prod_congr rfl fun t ht => ?_
    have ht := Finset.mem_range.1 ht
    simp only [Function.comp_def]
    rw [sigF_0 n σ (show t < n by omega), sigF_1 n σ (show t < n by omega),
      sigF_3 n σ (show t < n by omega),
      show m + 1 + (n - 1 - m - 1 - t) = n - 1 - t by omega,
      show engD d n (n - 1 - t - m) = pD d (n - 1 - m) t by
        unfold engD pD; rw [if_pos (by omega)]; congr 1; omega]
  unfold Ec Ce aeqC
  rw [hp, sigF_0 n σ (by omega), sigF_1 n σ (by omega), sigF_2 n σ (by omega),
    sigF_3 n σ (by omega), show n - 1 - (n - 1 - m) = m by omega]
  ring

theorem topv_sigF (d n : ℕ) (σ : ℕ → ℕ) :
    ∀ t ≤ n, topv d (sigF n σ) n 0 t = evd d σ (2 * n) t ∧
      topv d (sigF n σ) n 1 t = evd d σ (2 * n + 1) t ∧
      topv d (sigF n σ) n 2 t = evd d σ 0 t ∧ topv d (sigF n σ) n 3 t = evd d σ 1 t
  | 0, _ => ⟨rfl, rfl, rfl, rfl⟩
  | t + 1, ht => by
    obtain ⟨h0, h1, h2, h3⟩ := topv_sigF d n σ t (by omega)
    have e : n - 1 - (n - 1 - t) = t := by omega
    have hk : n - 1 - t < n := by omega
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [topv, evd, h0, sigF_0 n σ hk, e]; rfl
    · rw [topv, evd, h1, sigF_1 n σ hk, e, show bi n t = 2 * n + 1 + 2 * t by unfold bi; omega]
    · rw [topv, evd, h2, sigF_2 n σ hk, e, show ao t = 0 + 2 * t by unfold ao; omega]
    · rw [topv, evd, h3, sigF_3 n σ hk, e, show bo t = 1 + 2 * t by unfold bo; omega]

end bridge

/-! ## The engine's draper family -/

theorem engFam0 (d : ℕ) : Equal (engE d 0) 1 (engL 0) 1 (engR 0) := by
  refine ⟨List.Perm.refl _, fun σ _ => ?_⟩
  have hD : labDim (engE d 0) = fun _ => 1 := funext fun x => by
    simp [labDim, engE, engD]
  have hL : eval (engE d 0) (engL 0) σ = 1 := by
    simp [eval, engL, prodVal, tri, sumOver]
  have hR : eval (engE d 0) (engR 0) σ = 1 := by
    unfold eval
    rw [hD, show (engR 0).int = [0, 1, 2, 3, 4] from rfl]
    simp only [sumOver, Finset.range_one, Finset.sum_singleton]
    simp [prodVal, engR, val, engE, engD, tZ, tX, fou, ex_zero]
  rw [hL, hR]

/-- The engine's boxed Draper adder equals its spec at every `n` copies, for every dimension
`d`. -/
theorem engFam (d : ℕ) (hd : 0 < d) (n : ℕ) : Equal (engE d n) 1 (engL n) 1 (engR n) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact engFam0 d
  refine ⟨List.Perm.refl _, fun σ hσ => ?_⟩
  have hrb : 4 * n ≤ rb n := by unfold rb; omega
  have hb : ∀ x < 4 * n, σ x < d := fun x hx => by
    have := hσ x (mem_range.2 hx)
    rwa [labDim_low d hn (by omega)] at this
  rw [one_mul, one_mul, eval_engL d hd hn σ hb, eval_engR d hd hn σ hb]
  have key : ∏ m ∈ Finset.range n, ∑ x ∈ Finset.range d, Ec d n σ m x =
      ∏ i ∈ Finset.range n, (Draper.aeqC i (sigF n σ) *
        ∑ x ∈ Finset.range d, Ce d (sigF n σ) i x) := by
    rw [← Finset.prod_range_reflect _ n]
    refine Finset.prod_congr rfl fun i hi => ?_
    have hi := Finset.mem_range.1 hi
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Ec_Ce d (by omega), show n - 1 - (n - 1 - i) = i by omega]
  have hs : ∀ s < 4, ∀ i < n, sigF n σ (Draper.sl i s) < d := by
    intro s hs i hi
    interval_cases s
    · rw [sigF_0 n σ hi]; exact hb _ (by unfold ai; omega)
    · rw [sigF_1 n σ hi]; exact hb _ (by unfold bi; omega)
    · rw [sigF_2 n σ hi]; exact hb _ (by unfold ao; omega)
    · rw [sigF_3 n σ hi]; exact hb _ (by unfold bo; omega)
  obtain ⟨h0, h1, h2, h3⟩ := topv_sigF d n σ n le_rfl
  rw [key, prod_Ce d hd (sigF n σ) n hs, Draper.IndS, h0, h1, h2, h3]
  ring

theorem engFam_8 (d : ℕ) (hd : 0 < d) : Equal (engE d 8) 1 (engL 8) 1 (engR 8) :=
  engFam d hd 8

theorem engFam_2 : ∀ n, Equal (engE 2 n) 1 (engL n) 1 (engR n) :=
  engFam 2 (by norm_num)

/-! ## The engine's literal exports -/

/-- The engine export of the boxed side at n = 1. -/
def litL1 : Net := ⟨([⟨.F, [5], [4], 0, 0⟩, ⟨.Z, [6, 7], [2], 0, 0⟩, ⟨.Z, [10], [8, 9], 0, 0⟩, ⟨.F, [9], [7], 0, 0⟩, ⟨.F, [12], [11], 0, 0⟩, ⟨.F, [13], [12], 0, 0⟩, ⟨.F, [14], [13], 0, 0⟩, ⟨.Z, [4], [3], 0, 0⟩, ⟨.Z, [8], [5], 0, 0⟩, ⟨.Z, [0], [6], 0, 0⟩, ⟨.Z, [15], [10], 0, 0⟩, ⟨.Z, [11], [15], 0, 0⟩, ⟨.Z, [1], [14], 0, 0⟩] : List Node), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3]⟩

/-- The engine export of the spec at n = 1. -/
def litR1 : Net := ⟨([⟨.Z, [16], [], 1, 0⟩, ⟨.B, [17], [16, 2], 1, 0⟩, ⟨.S, [19, 0], [18], 1, 0⟩, ⟨.Z, [], [19], 1, 0⟩, ⟨.Z, [20], [], 1, 0⟩, ⟨.B, [21], [20, 3], 1, 0⟩, ⟨.S, [23, 1], [22], 1, 0⟩, ⟨.Z, [], [23], 1, 0⟩, ⟨.Z, [18, 24], [17], 0, 0⟩, ⟨.X, [22], [21, 24], 0, 0⟩] : List Node), [16, 17, 18, 19, 20, 21, 22, 23, 24], [0, 1, 2, 3]⟩

/-- The engine export of the boxed side at n = 2. -/
def litL2 : Net := ⟨([⟨.F, [9], [8], 0, 0⟩, ⟨.Z, [10, 11], [4], 0, 0⟩, ⟨.Z, [14], [12, 13], 0, 0⟩, ⟨.F, [13], [11], 0, 0⟩, ⟨.F, [16], [15], 0, 0⟩, ⟨.F, [17], [16], 0, 0⟩, ⟨.F, [18], [17], 0, 0⟩, ⟨.F, [20], [19], 0, 0⟩, ⟨.Z, [21, 22], [6], 0, 0⟩, ⟨.Z, [25], [23, 24], 0, 0⟩, ⟨.F, [24], [22], 0, 0⟩, ⟨.F, [27], [26], 0, 0⟩, ⟨.F, [28], [27], 0, 0⟩, ⟨.F, [29], [28], 0, 0⟩, ⟨.Z, [8], [5], 0, 0⟩, ⟨.Z, [19], [7, 30], 0, 0⟩, ⟨.Z, [12, 31], [9], 0, 0⟩, ⟨.Z, [23], [20], 0, 0⟩, ⟨.Z, [0], [10], 0, 0⟩, ⟨.Z, [2], [21, 32], 0, 0⟩, ⟨.Z, [33, 34], [14], 0, 0⟩, ⟨.Z, [35], [25], 0, 0⟩, ⟨.Z, [15, 36], [33], 0, 0⟩, ⟨.Z, [26], [35], 0, 0⟩, ⟨.Z, [1], [18], 0, 0⟩, ⟨.Z, [3], [29, 37], 0, 0⟩, ⟨.ZP 0 0 (1), [30], [31], 1, 0⟩, ⟨.ZP 0 0 (1), [32], [34], 1, 0⟩, ⟨.ZP 0 0 (-1), [37], [36], 1, 0⟩] : List Node), [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37], [0, 1, 2, 3, 4, 5, 6, 7]⟩

/-- The engine export of the spec at n = 2. -/
def litR2 : Net := ⟨([⟨.Z, [38], [], 2, 0⟩, ⟨.Z, [], [39], 2, 0⟩, ⟨.Z, [40], [], 2, 0⟩, ⟨.Z, [], [41], 2, 0⟩, ⟨.Z, [43, 44], [42], 1, 0⟩, ⟨.X, [46], [45, 44], 1, 0⟩, ⟨.B, [47], [38, 4], 2, 0⟩, ⟨.B, [48], [40, 5], 2, 0⟩, ⟨.B, [42], [47, 6], 0, 0⟩, ⟨.B, [45], [48, 7], 0, 0⟩, ⟨.S, [39, 0], [49], 2, 0⟩, ⟨.S, [41, 1], [50], 2, 0⟩, ⟨.S, [49, 2], [43], 0, 0⟩, ⟨.S, [50, 3], [46], 0, 0⟩] : List Node), [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], [0, 1, 2, 3, 4, 5, 6, 7]⟩

/-- The engine export of the boxed side at n = 3. -/
def litL3 : Net := ⟨([⟨.F, [13], [12], 0, 0⟩, ⟨.Z, [14, 15], [6], 0, 0⟩, ⟨.Z, [18], [16, 17], 0, 0⟩, ⟨.F, [17], [15], 0, 0⟩, ⟨.F, [20], [19], 0, 0⟩, ⟨.F, [21], [20], 0, 0⟩, ⟨.F, [22], [21], 0, 0⟩, ⟨.F, [24], [23], 0, 0⟩, ⟨.Z, [25, 26], [8], 0, 0⟩, ⟨.Z, [29], [27, 28], 0, 0⟩, ⟨.F, [28], [26], 0, 0⟩, ⟨.F, [31], [30], 0, 0⟩, ⟨.F, [32], [31], 0, 0⟩, ⟨.F, [33], [32], 0, 0⟩, ⟨.F, [35], [34], 0, 0⟩, ⟨.Z, [36, 37], [10], 0, 0⟩, ⟨.Z, [40], [38, 39], 0, 0⟩, ⟨.F, [39], [37], 0, 0⟩, ⟨.F, [42], [41], 0, 0⟩, ⟨.F, [43], [42], 0, 0⟩, ⟨.F, [44], [43], 0, 0⟩, ⟨.Z, [12], [7], 0, 0⟩, ⟨.Z, [23], [9, 45], 0, 0⟩, ⟨.Z, [34], [11, 46, 47], 0, 0⟩, ⟨.Z, [16, 48, 49], [13], 0, 0⟩, ⟨.Z, [27, 50], [24], 0, 0⟩, ⟨.Z, [38], [35], 0, 0⟩, ⟨.Z, [0], [14], 0, 0⟩, ⟨.Z, [2], [25, 51], 0, 0⟩, ⟨.Z, [4], [36, 52, 53], 0, 0⟩, ⟨.Z, [54, 55, 56], [18], 0, 0⟩, ⟨.Z, [57, 58], [29], 0, 0⟩, ⟨.Z, [59], [40], 0, 0⟩, ⟨.Z, [19, 60, 61], [54], 0, 0⟩, ⟨.Z, [30, 62], [57], 0, 0⟩, ⟨.Z, [41], [59], 0, 0⟩, ⟨.Z, [1], [22], 0, 0⟩, ⟨.Z, [3], [33, 63], 0, 0⟩, ⟨.Z, [5], [44, 64, 65], 0, 0⟩, ⟨.ZP 0 0 (1), [45], [48], 1, 0⟩, ⟨.ZP 0 0 (1), [46], [49], 2, 0⟩, ⟨.ZP 0 0 (1), [47], [50], 1, 0⟩, ⟨.ZP 0 0 (1), [51], [55], 1, 0⟩, ⟨.ZP 0 0 (1), [52], [56], 2, 0⟩, ⟨.ZP 0 0 (1), [53], [58], 1, 0⟩, ⟨.ZP 0 0 (-1), [63], [60], 1, 0⟩, ⟨.ZP 0 0 (-1), [64], [61], 2, 0⟩, ⟨.ZP 0 0 (-1), [65], [62], 1, 0⟩] : List Node), [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]⟩

/-- The engine export of the spec at n = 3. -/
def litR3 : Net := ⟨([⟨.Z, [66], [], 3, 0⟩, ⟨.Z, [], [67], 3, 0⟩, ⟨.Z, [68], [], 3, 0⟩, ⟨.Z, [], [69], 3, 0⟩, ⟨.Z, [71, 72], [70], 2, 0⟩, ⟨.X, [74], [73, 72], 2, 0⟩, ⟨.B, [75], [66, 6], 3, 0⟩, ⟨.B, [76], [68, 7], 3, 0⟩, ⟨.B, [77], [75, 8], 0, 0⟩, ⟨.B, [78], [76, 9], 0, 0⟩, ⟨.B, [70], [77, 10], 1, 0⟩, ⟨.B, [73], [78, 11], 1, 0⟩, ⟨.S, [67, 0], [79], 3, 0⟩, ⟨.S, [69, 1], [80], 3, 0⟩, ⟨.S, [79, 2], [81], 0, 0⟩, ⟨.S, [80, 3], [82], 0, 0⟩, ⟨.S, [81, 4], [71], 1, 0⟩, ⟨.S, [82, 5], [74], 1, 0⟩] : List Node), [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]⟩

/-- The engine export of the boxed side at n = 4. -/
def litL4 : Net := ⟨([⟨.F, [17], [16], 0, 0⟩, ⟨.Z, [18, 19], [8], 0, 0⟩, ⟨.Z, [22], [20, 21], 0, 0⟩, ⟨.F, [21], [19], 0, 0⟩, ⟨.F, [24], [23], 0, 0⟩, ⟨.F, [25], [24], 0, 0⟩, ⟨.F, [26], [25], 0, 0⟩, ⟨.F, [28], [27], 0, 0⟩, ⟨.Z, [29, 30], [10], 0, 0⟩, ⟨.Z, [33], [31, 32], 0, 0⟩, ⟨.F, [32], [30], 0, 0⟩, ⟨.F, [35], [34], 0, 0⟩, ⟨.F, [36], [35], 0, 0⟩, ⟨.F, [37], [36], 0, 0⟩, ⟨.F, [39], [38], 0, 0⟩, ⟨.Z, [40, 41], [12], 0, 0⟩, ⟨.Z, [44], [42, 43], 0, 0⟩, ⟨.F, [43], [41], 0, 0⟩, ⟨.F, [46], [45], 0, 0⟩, ⟨.F, [47], [46], 0, 0⟩, ⟨.F, [48], [47], 0, 0⟩, ⟨.F, [50], [49], 0, 0⟩, ⟨.Z, [51, 52], [14], 0, 0⟩, ⟨.Z, [55], [53, 54], 0, 0⟩, ⟨.F, [54], [52], 0, 0⟩, ⟨.F, [57], [56], 0, 0⟩, ⟨.F, [58], [57], 0, 0⟩, ⟨.F, [59], [58], 0, 0⟩, ⟨.Z, [16], [9], 0, 0⟩, ⟨.Z, [27], [11, 60], 0, 0⟩, ⟨.Z, [38], [13, 61, 62], 0, 0⟩, ⟨.Z, [49], [15, 63, 64, 65], 0, 0⟩, ⟨.Z, [20, 66, 67, 68], [17], 0, 0⟩, ⟨.Z, [31, 69, 70], [28], 0, 0⟩, ⟨.Z, [42, 71], [39], 0, 0⟩, ⟨.Z, [53], [50], 0, 0⟩, ⟨.Z, [0], [18], 0, 0⟩, ⟨.Z, [2], [29, 72], 0, 0⟩, ⟨.Z, [4], [40, 73, 74], 0, 0⟩, ⟨.Z, [6], [51, 75, 76, 77], 0, 0⟩, ⟨.Z, [78, 79, 80, 81], [22], 0, 0⟩, ⟨.Z, [82, 83, 84], [33], 0, 0⟩, ⟨.Z, [85, 86], [44], 0, 0⟩, ⟨.Z, [87], [55], 0, 0⟩, ⟨.Z, [23, 88, 89, 90], [78], 0, 0⟩, ⟨.Z, [34, 91, 92], [82], 0, 0⟩, ⟨.Z, [45, 93], [85], 0, 0⟩, ⟨.Z, [56], [87], 0, 0⟩, ⟨.Z, [1], [26], 0, 0⟩, ⟨.Z, [3], [37, 94], 0, 0⟩, ⟨.Z, [5], [48, 95, 96], 0, 0⟩, ⟨.Z, [7], [59, 97, 98, 99], 0, 0⟩, ⟨.ZP 0 0 (1), [60], [66], 1, 0⟩, ⟨.ZP 0 0 (1), [61], [67], 2, 0⟩, ⟨.ZP 0 0 (1), [63], [68], 3, 0⟩, ⟨.ZP 0 0 (1), [62], [69], 1, 0⟩, ⟨.ZP 0 0 (1), [64], [70], 2, 0⟩, ⟨.ZP 0 0 (1), [65], [71], 1, 0⟩, ⟨.ZP 0 0 (1), [72], [79], 1, 0⟩, ⟨.ZP 0 0 (1), [73], [80], 2, 0⟩, ⟨.ZP 0 0 (1), [75], [81], 3, 0⟩, ⟨.ZP 0 0 (1), [74], [83], 1, 0⟩, ⟨.ZP 0 0 (1), [76], [84], 2, 0⟩, ⟨.ZP 0 0 (1), [77], [86], 1, 0⟩, ⟨.ZP 0 0 (-1), [94], [88], 1, 0⟩, ⟨.ZP 0 0 (-1), [95], [89], 2, 0⟩, ⟨.ZP 0 0 (-1), [97], [90], 3, 0⟩, ⟨.ZP 0 0 (-1), [96], [91], 1, 0⟩, ⟨.ZP 0 0 (-1), [98], [92], 2, 0⟩, ⟨.ZP 0 0 (-1), [99], [93], 1, 0⟩] : List Node), [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]⟩

/-- The engine export of the spec at n = 4. -/
def litR4 : Net := ⟨([⟨.Z, [100], [], 4, 0⟩, ⟨.Z, [], [101], 4, 0⟩, ⟨.Z, [102], [], 4, 0⟩, ⟨.Z, [], [103], 4, 0⟩, ⟨.Z, [105, 106], [104], 3, 0⟩, ⟨.X, [108], [107, 106], 3, 0⟩, ⟨.B, [109], [100, 8], 4, 0⟩, ⟨.B, [110], [102, 9], 4, 0⟩, ⟨.B, [111], [109, 10], 0, 0⟩, ⟨.B, [112], [110, 11], 0, 0⟩, ⟨.B, [113], [111, 12], 1, 0⟩, ⟨.B, [114], [112, 13], 1, 0⟩, ⟨.B, [104], [113, 14], 2, 0⟩, ⟨.B, [107], [114, 15], 2, 0⟩, ⟨.S, [101, 0], [115], 4, 0⟩, ⟨.S, [103, 1], [116], 4, 0⟩, ⟨.S, [115, 2], [117], 0, 0⟩, ⟨.S, [116, 3], [118], 0, 0⟩, ⟨.S, [117, 4], [119], 1, 0⟩, ⟨.S, [118, 5], [120], 1, 0⟩, ⟨.S, [119, 6], [105], 2, 0⟩, ⟨.S, [120, 7], [108], 2, 0⟩] : List Node), [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]⟩

/-- The engine export of the boxed side at n = 5. -/
def litL5 : Net := ⟨([⟨.F, [21], [20], 0, 0⟩, ⟨.Z, [22, 23], [10], 0, 0⟩, ⟨.Z, [26], [24, 25], 0, 0⟩, ⟨.F, [25], [23], 0, 0⟩, ⟨.F, [28], [27], 0, 0⟩, ⟨.F, [29], [28], 0, 0⟩, ⟨.F, [30], [29], 0, 0⟩, ⟨.F, [32], [31], 0, 0⟩, ⟨.Z, [33, 34], [12], 0, 0⟩, ⟨.Z, [37], [35, 36], 0, 0⟩, ⟨.F, [36], [34], 0, 0⟩, ⟨.F, [39], [38], 0, 0⟩, ⟨.F, [40], [39], 0, 0⟩, ⟨.F, [41], [40], 0, 0⟩, ⟨.F, [43], [42], 0, 0⟩, ⟨.Z, [44, 45], [14], 0, 0⟩, ⟨.Z, [48], [46, 47], 0, 0⟩, ⟨.F, [47], [45], 0, 0⟩, ⟨.F, [50], [49], 0, 0⟩, ⟨.F, [51], [50], 0, 0⟩, ⟨.F, [52], [51], 0, 0⟩, ⟨.F, [54], [53], 0, 0⟩, ⟨.Z, [55, 56], [16], 0, 0⟩, ⟨.Z, [59], [57, 58], 0, 0⟩, ⟨.F, [58], [56], 0, 0⟩, ⟨.F, [61], [60], 0, 0⟩, ⟨.F, [62], [61], 0, 0⟩, ⟨.F, [63], [62], 0, 0⟩, ⟨.F, [65], [64], 0, 0⟩, ⟨.Z, [66, 67], [18], 0, 0⟩, ⟨.Z, [70], [68, 69], 0, 0⟩, ⟨.F, [69], [67], 0, 0⟩, ⟨.F, [72], [71], 0, 0⟩, ⟨.F, [73], [72], 0, 0⟩, ⟨.F, [74], [73], 0, 0⟩, ⟨.Z, [20], [11], 0, 0⟩, ⟨.Z, [31], [13, 75], 0, 0⟩, ⟨.Z, [42], [15, 76, 77], 0, 0⟩, ⟨.Z, [53], [17, 78, 79, 80], 0, 0⟩, ⟨.Z, [64], [19, 81, 82, 83, 84], 0, 0⟩, ⟨.Z, [24, 85, 86, 87, 88], [21], 0, 0⟩, ⟨.Z, [35, 89, 90, 91], [32], 0, 0⟩, ⟨.Z, [46, 92, 93], [43], 0, 0⟩, ⟨.Z, [57, 94], [54], 0, 0⟩, ⟨.Z, [68], [65], 0, 0⟩, ⟨.Z, [0], [22], 0, 0⟩, ⟨.Z, [2], [33, 95], 0, 0⟩, ⟨.Z, [4], [44, 96, 97], 0, 0⟩, ⟨.Z, [6], [55, 98, 99, 100], 0, 0⟩, ⟨.Z, [8], [66, 101, 102, 103, 104], 0, 0⟩, ⟨.Z, [105, 106, 107, 108, 109], [26], 0, 0⟩, ⟨.Z, [110, 111, 112, 113], [37], 0, 0⟩, ⟨.Z, [114, 115, 116], [48], 0, 0⟩, ⟨.Z, [117, 118], [59], 0, 0⟩, ⟨.Z, [119], [70], 0, 0⟩, ⟨.Z, [27, 120, 121, 122, 123], [105], 0, 0⟩, ⟨.Z, [38, 124, 125, 126], [110], 0, 0⟩, ⟨.Z, [49, 127, 128], [114], 0, 0⟩, ⟨.Z, [60, 129], [117], 0, 0⟩, ⟨.Z, [71], [119], 0, 0⟩, ⟨.Z, [1], [30], 0, 0⟩, ⟨.Z, [3], [41, 130], 0, 0⟩, ⟨.Z, [5], [52, 131, 132], 0, 0⟩, ⟨.Z, [7], [63, 133, 134, 135], 0, 0⟩, ⟨.Z, [9], [74, 136, 137, 138, 139], 0, 0⟩, ⟨.ZP 0 0 (1), [75], [85], 1, 0⟩, ⟨.ZP 0 0 (1), [76], [86], 2, 0⟩, ⟨.ZP 0 0 (1), [78], [87], 3, 0⟩, ⟨.ZP 0 0 (1), [81], [88], 4, 0⟩, ⟨.ZP 0 0 (1), [77], [89], 1, 0⟩, ⟨.ZP 0 0 (1), [79], [90], 2, 0⟩, ⟨.ZP 0 0 (1), [82], [91], 3, 0⟩, ⟨.ZP 0 0 (1), [80], [92], 1, 0⟩, ⟨.ZP 0 0 (1), [83], [93], 2, 0⟩, ⟨.ZP 0 0 (1), [84], [94], 1, 0⟩, ⟨.ZP 0 0 (1), [95], [106], 1, 0⟩, ⟨.ZP 0 0 (1), [96], [107], 2, 0⟩, ⟨.ZP 0 0 (1), [98], [108], 3, 0⟩, ⟨.ZP 0 0 (1), [101], [109], 4, 0⟩, ⟨.ZP 0 0 (1), [97], [111], 1, 0⟩, ⟨.ZP 0 0 (1), [99], [112], 2, 0⟩, ⟨.ZP 0 0 (1), [102], [113], 3, 0⟩, ⟨.ZP 0 0 (1), [100], [115], 1, 0⟩, ⟨.ZP 0 0 (1), [103], [116], 2, 0⟩, ⟨.ZP 0 0 (1), [104], [118], 1, 0⟩, ⟨.ZP 0 0 (-1), [130], [120], 1, 0⟩, ⟨.ZP 0 0 (-1), [131], [121], 2, 0⟩, ⟨.ZP 0 0 (-1), [133], [122], 3, 0⟩, ⟨.ZP 0 0 (-1), [136], [123], 4, 0⟩, ⟨.ZP 0 0 (-1), [132], [124], 1, 0⟩, ⟨.ZP 0 0 (-1), [134], [125], 2, 0⟩, ⟨.ZP 0 0 (-1), [137], [126], 3, 0⟩, ⟨.ZP 0 0 (-1), [135], [127], 1, 0⟩, ⟨.ZP 0 0 (-1), [138], [128], 2, 0⟩, ⟨.ZP 0 0 (-1), [139], [129], 1, 0⟩] : List Node), [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]⟩

/-- The engine export of the spec at n = 5. -/
def litR5 : Net := ⟨([⟨.Z, [140], [], 5, 0⟩, ⟨.Z, [], [141], 5, 0⟩, ⟨.Z, [142], [], 5, 0⟩, ⟨.Z, [], [143], 5, 0⟩, ⟨.Z, [145, 146], [144], 4, 0⟩, ⟨.X, [148], [147, 146], 4, 0⟩, ⟨.B, [149], [140, 10], 5, 0⟩, ⟨.B, [150], [142, 11], 5, 0⟩, ⟨.B, [151], [149, 12], 0, 0⟩, ⟨.B, [152], [150, 13], 0, 0⟩, ⟨.B, [153], [151, 14], 1, 0⟩, ⟨.B, [154], [152, 15], 1, 0⟩, ⟨.B, [155], [153, 16], 2, 0⟩, ⟨.B, [156], [154, 17], 2, 0⟩, ⟨.B, [144], [155, 18], 3, 0⟩, ⟨.B, [147], [156, 19], 3, 0⟩, ⟨.S, [141, 0], [157], 5, 0⟩, ⟨.S, [143, 1], [158], 5, 0⟩, ⟨.S, [157, 2], [159], 0, 0⟩, ⟨.S, [158, 3], [160], 0, 0⟩, ⟨.S, [159, 4], [161], 1, 0⟩, ⟨.S, [160, 5], [162], 1, 0⟩, ⟨.S, [161, 6], [163], 2, 0⟩, ⟨.S, [162, 7], [164], 2, 0⟩, ⟨.S, [163, 8], [145], 3, 0⟩, ⟨.S, [164, 9], [148], 3, 0⟩] : List Node), [140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]⟩

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

theorem engLabs_lit1 : (engLabs 1).filter (fun p => p.2 ≠ 0) =
    [(16, 1), (19, 1), (20, 1), (23, 1)] := by
  decide +kernel

theorem engLabs_lit2 : (engLabs 2).filter (fun p => p.2 ≠ 0) =
    [(38, 2), (39, 2), (40, 2), (41, 2), (42, 1), (43, 1), (44, 1), (45, 1), (46, 1)] := by
  decide +kernel

theorem engLabs_lit3 : (engLabs 3).filter (fun p => p.2 ≠ 0) =
    [(66, 3), (67, 3), (68, 3), (69, 3), (70, 2), (71, 2), (72, 2), (73, 2), (74, 2), (77, 1), (78, 1), (81, 1), (82, 1)] := by
  decide +kernel

theorem engLabs_lit4 : (engLabs 4).filter (fun p => p.2 ≠ 0) =
    [(100, 4), (101, 4), (102, 4), (103, 4), (104, 3), (105, 3), (106, 3), (107, 3), (108, 3), (111, 1), (112, 1), (113, 2), (114, 2), (117, 1), (118, 1), (119, 2), (120, 2)] := by
  decide +kernel

theorem engLabs_lit5 : (engLabs 5).filter (fun p => p.2 ≠ 0) =
    [(140, 5), (141, 5), (142, 5), (143, 5), (144, 4), (145, 4), (146, 4), (147, 4), (148, 4), (151, 1), (152, 1), (153, 2), (154, 2), (155, 3), (156, 3), (159, 1), (160, 1), (161, 2), (162, 2), (163, 3), (164, 3)] := by
  decide +kernel

theorem engL_lit0 : engL 0 = ⟨([] : List Node), [], []⟩ := by decide +kernel

theorem engR_lit0 : engR 0 = ⟨([⟨.Z, [0], [], 0, 0⟩, ⟨.Z, [], [1], 0, 0⟩, ⟨.Z, [2], [], 0, 0⟩, ⟨.Z, [], [3], 0, 0⟩, ⟨.Z, [1, 4], [0], 0, 0⟩, ⟨.X, [3], [2, 4], 0, 0⟩] : List Node), [0, 1, 2, 3, 4], []⟩ := by decide +kernel

end DraperEng

end ArchytasZX
