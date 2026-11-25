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
* `engL_lit`, `engR_lit`: equal to the engine's literal exports at `n = 1 … 5`.
-/

namespace ArchytasZX

namespace FcaEng

open List

/-- `j (j - 1) / 2`. -/
def tri (j : ℕ) : ℕ := j * (j - 1) / 2

/-- Copy `j`'s chain base. -/
def cb (n j : ℕ) : ℕ := 2 * n + 8 * j

/-- Position of the pair `(m, j)`, `m < j`, ordered by `m` then `j`. -/
def i2 (n m j : ℕ) : ℕ := m * (n - 1) - tri m + (j - m - 1)

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

/-- The spec's label dimensions. -/
def engLabs (n : ℕ) : List (ℕ × ℕ) :=
  [(rb n, n), (rb n + 1, n - 1), (rb n + 2, n - 1), (rb n + 3, n)] ++
    (range (n - 1)).map (fun t => (rb n + 4 + t, t)) ++
    (range (n - 1)).map (fun t => (rb n + n + 3 + t, t))

/-- The engine's dimensions. -/
def engD (d n i : ℕ) : ℕ := if i < n then d ^ (i + 1) else 1

/-- The engine's phases. -/
noncomputable def engP (d n p k : ℕ) : ℝ :=
  if p = 0 then 0
  else if p ≤ n then 2 * k / (d : ℝ) ^ (n - p + 1)
  else -(2 * k) / (d : ℝ) ^ n

/-- The engine's environment at `n`. -/
noncomputable def engE (d n : ℕ) : Env := { dimv := engD d n, phase := engP d n, labs := engLabs n }

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

end FcaEng

end ArchytasZX
