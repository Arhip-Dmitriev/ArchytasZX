/-
  ArchytasZX proof export over the tensor-network semantics of the ArchytasZX Lean
  library: every diagram is a Net, every rule step an instance of a lemma proved
  there. `#print axioms` lists any step stated without proof.
-/

import ArchytasZX

namespace ArchytasZX.Export

open ArchytasZX

set_option linter.unusedVariables false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false


set_option hygiene false in
/-- Decides a structural side condition. -/
macro "chk" : tactic => `(tactic| first | decide | (simp only [F, Fam.env, bundL, List.lookup, ksOf, List.getD]; decide))

set_option hygiene false in
/-- A dimension is positive. -/
macro "pos" : tactic => `(tactic| first | exact Fam.env_dimv_pos _ _ _ (Fpos d h_d _) | (simp [F, Fam.env, bundL, List.lookup, ksOf, List.getD, Δ] <;> first | decide | omega | positivity | (apply Nat.mul_pos <;> omega) | (apply pow_pos; omega)))

set_option hygiene false in
/-- A phase vector is zero. -/
macro "phz" : tactic => `(tactic| (intro k _; simp [F, Fam.env, bundL, List.lookup, ksOf, List.getD, Φ, ex_zero]))

set_option hygiene false in
/-- A product of nonzero scalars is nonzero. -/
macro "nzt" : tactic => `(tactic| repeat (first | exact one_ne_zero | apply mul_ne_zero | apply pow_ne_zero | apply inv_ne_zero | exact Complex.I_ne_zero | exact Complex.exp_ne_zero _ | (apply rt_ne; first | omega | norm_num | positivity | (apply pow_pos; omega)) | (apply om_ne_zero; first | omega | norm_num) | (apply Nat.cast_ne_zero.2; first | omega | positivity | norm_num) | norm_num))

set_option hygiene false in
/-- A phase vector is the sum of two. -/
macro "phs" : tactic => `(tactic| (intro k _; simp only [F, Fam.env, bundL, List.lookup, ksOf, List.getD, Φ]; rw [← ex_add]; apply ex_congr_int; (try split_ifs) <;> first | (refine ⟨0, ?_⟩; push_cast; linarith) | (refine ⟨-1, ?_⟩; push_cast; linarith) | (refine ⟨1, ?_⟩; push_cast; linarith) | (refine ⟨-2, ?_⟩; push_cast; linarith) | (refine ⟨2, ?_⟩; push_cast; linarith)))

set_option hygiene false in
/-- A coefficient vector is `[k = 0]`. -/
macro "zcz" : tactic => `(tactic| (intro k hk; simp only [F, Fam.env, bundL, List.lookup, ksOf, List.getD, Δ] at hk; simp only [zc, F, Fam.env, bundL, List.lookup, ksOf, List.getD, Φ]; (try simp only [Ψ]); (try split_ifs) <;> first | (exfalso; omega) | simp [ex_zero]))

/-- Dimension values. -/
def Δ (d : ℕ) : ℕ → ℕ
  | 0 => d
  | 1 => d
  | _ => 1

/-- Every dimension value is positive. -/
theorem Δpos (d : ℕ) (h_d : 0 < d) : ∀ i, 0 < Δ d i := by
  intro i
  unfold Δ
  split <;> first | omega | positivity | (apply pow_pos; omega) | norm_num

/-- Phase vectors in turns. -/
noncomputable def Φ (d : ℕ) : ℕ → ℕ → ℝ
  | 0 => fun _ => 0
  | _ => fun _ => 0

/-- Each label's dimension index. -/
def labs : List (ℕ × ℕ) := [(0, 0), (1, 1), (2, 0), (3, 0), (4, 1), (5, 1), (6, 0), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (65537, 0), (65540, 0), (65541, 0), (65543, 0), (65544, 0), (65545, 0), (65546, 0), (65547, 0)]

/-- The environment. -/
noncomputable def E (d : ℕ) : Env := { dimv := Δ d, phase := Φ d, labs := labs }

/-- Each bundle dimension index with its base index and multiplicity. -/
def bundL : List (ℕ × ℕ × ℕ) := [(1, 0, 0)]

/-- The family, its multiplicities in order: n = n. -/
noncomputable def F (d : ℕ) : Fam := { dims := Δ d, bund := bundL, phase := Φ d, labs := labs }

/-- Every dimension value of the family is positive. -/
theorem Fpos (d : ℕ) (h_d : 0 < d) : ∀ i, 0 < (F d).dims i := fun i => Δpos d h_d i

/-- The family's structure alone. -/
noncomputable def FC : Fam := { dims := fun _ => 1, bund := bundL, phase := fun _ _ => 0, labs := labs }

/-- The family's structure along multiplicity `m`. -/
noncomputable def SpC (m : ℕ) : Spec := ⟨FC, m, ksOf []⟩

/-- (1/sqrt(d)) Σ_{k2,q0=0}^{d-1} Σ_{q3[i0],q4[i0]=0, i0=1..n}^{d-1} Σ_{j0o0=0}^{d-1} Σ_{j3i1[i0],j3o0[i0]=0, i0=1..n}^{d-1} d^{-1/2} ω_{d}^{q0 (j0o0)} d^{-1/2} ω_{d}^{k2 j0o0} ∏_{i0=1}^{n}( d^{-3/2} ω_{d}^{q3[i0] (j3o0[i0] - k2 - j3i1[i0])} d^{-1/2} ω_{d}^{q4[i0] (j3i1[i0])} ) |k2⟩ ⊗_{i0=1}^{n}( |j3o0[i0]⟩ ) -/
def N0 : Net := ⟨([⟨.X, [2], [], 0, 0⟩, ⟨.F, [3], [2], 0, 0⟩, ⟨.ZB [[0]], [0], [3, 4], 0, 0⟩, ⟨.Bun 0 .X, [1], [4, 5], 0, 0⟩, ⟨.Bun 0 .X, [5], [], 0, 0⟩] : List Node), [2, 3, 4, 5], [0, 1]⟩

/-- (1/sqrt(d)) Σ_{k=0}^{d-1} |k⟩ |k⟩^{n} -/
def N1 : Net := ⟨([⟨.ZB [[0]], [0], [1], 0, 0⟩] : List Node), [], [0, 1]⟩

/-- (1/sqrt(d)) Σ_{k2,k5=0}^{d-1} Σ_{q3[i0],q4[i0],j3i1[i0],j3o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-3/2} ω_{d}^{q3[i0] (j3o0[i0] - k2 - j3i1[i0])} d^{-1/2} ω_{d}^{q4[i0] (j3i1[i0])} ) [k2 = k5] |k2⟩ ⊗_{i0=1}^{n}( |j3o0[i0]⟩ ) -/
def N2 : Net := ⟨([⟨.ZB [[0]], [0], [6, 7], 0, 0⟩, ⟨.Bun 0 .X, [1], [7, 8], 0, 0⟩, ⟨.Bun 0 .X, [8], [], 0, 0⟩, ⟨.Z, [6], [], 0, 0⟩] : List Node), [6, 7, 8], [0, 1]⟩

theorem step1 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N0 1 N2 := by
  refine Equal.trans (isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.ZB [[0]], [0], [3, 4], 0, 0⟩, ⟨.Bun 0 .X, [1], [4, 5], 0, 0⟩, ⟨.Bun 0 .X, [5], [], 0, 0⟩, ⟨.X, [2], [], 0, 0⟩, ⟨.F, [3], [2], 0, 0⟩] : List Node), [3, 4, 5, 2], [0, 1]⟩ [] [2, 3, 4, 0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [n])) ([⟨.ZB [[0]], [0], [3, 4], 0, 0⟩, ⟨.Bun 0 .X, [1], [4, 5], 0, 0⟩, ⟨.Bun 0 .X, [5], [], 0, 0⟩] : List Node) ([⟨.X, [2], [], 0, 0⟩, ⟨.F, [3], [2], 0, 0⟩] : List Node) ([⟨.Z, [3], [], 0, 0⟩] : List Node) [3, 4, 5] [2] [] [0, 1] [3] 1 1 1 (colorStateX ((F d).env (ksOf [n])) 0 0 0 2 3 [3] (by pos) (by chk) (by chk) (by chk) (by chk) (by phz) (by phz)) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.ZB [[0]], [0], [6, 7], 0, 0⟩, ⟨.Bun 0 .X, [1], [7, 8], 0, 0⟩, ⟨.Bun 0 .X, [8], [], 0, 0⟩, ⟨.Z, [6], [], 0, 0⟩] : List Node), [6, 7, 8], [0, 1]⟩ [(3, 6), (4, 7), (5, 8), (6, 3), (7, 4), (8, 5)] [0, 1, 2, 3] (by chk)

/-- (1/sqrt(d)) Σ_{k6=0}^{d-1} Σ_{q3[i0],q4[i0],j3i1[i0],j3o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-3/2} ω_{d}^{q3[i0] (j3o0[i0] - k6 - j3i1[i0])} d^{-1/2} ω_{d}^{q4[i0] (j3i1[i0])} ) |k6⟩ ⊗_{i0=1}^{n}( |j3o0[i0]⟩ ) -/
def N3 : Net := ⟨([⟨.Bun 0 .X, [1], [9, 10], 0, 0⟩, ⟨.Bun 0 .X, [10], [], 0, 0⟩, ⟨.ZB [[0]], [0], [9], 0, 0⟩] : List Node), [9, 10], [0, 1]⟩

theorem step2 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N2 1 N3 := by
  refine Equal.trans (isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.Bun 0 .X, [1], [7, 8], 0, 0⟩, ⟨.Bun 0 .X, [8], [], 0, 0⟩, ⟨.ZB [[0]], [0], [6, 7], 0, 0⟩, ⟨.Z, [6], [], 0, 0⟩] : List Node), [7, 8, 6], [0, 1]⟩ [] [1, 2, 0, 3] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [n])) ([⟨.Bun 0 .X, [1], [7, 8], 0, 0⟩, ⟨.Bun 0 .X, [8], [], 0, 0⟩] : List Node) ([⟨.ZB [[0]], [0], [6, 7], 0, 0⟩, ⟨.Z, [6], [], 0, 0⟩] : List Node) ([⟨.ZB [[0]], [0], [7], 0, 0⟩] : List Node) [7, 8] [6] [] [0, 1] [0, 7] 1 1 1 (fusionZG ((F d).env (ksOf [n])) [[0]] [] 0 0 0 0 6 [0] [6] [7] [6] [] [] [0] [] [0, 7] (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by phs)) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.Bun 0 .X, [1], [9, 10], 0, 0⟩, ⟨.Bun 0 .X, [10], [], 0, 0⟩, ⟨.ZB [[0]], [0], [9], 0, 0⟩] : List Node), [9, 10], [0, 1]⟩ [(7, 9), (8, 10), (9, 7), (10, 8)] [0, 1, 2] (by chk)

/-- (1/sqrt(d)) Σ_{k6=0}^{d-1} Σ_{q7[i0],j7o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-1} ω_{d}^{q7[i0] (j7o0[i0] - k6)} ) |k6⟩ ⊗_{i0=1}^{n}( |j7o0[i0]⟩ ) -/
def N4 : Net := ⟨([⟨.ZB [[0]], [0], [11], 0, 0⟩, ⟨.Bun 0 .X, [1], [11], 0, 0⟩] : List Node), [11], [0, 1]⟩

/-- The boxed region before `bun3`. -/
def bun3L : Net := ⟨([⟨.Bun 0 .X, [10], [], 0, 0⟩, ⟨.Bun 0 .X, [1], [9, 10], 0, 0⟩] : List Node), [10], [1, 9]⟩

/-- The boxed region after `bun3`. -/
def bun3R : Net := ⟨([⟨.Bun 0 .X, [1], [9], 0, 0⟩] : List Node), [], [1, 9]⟩

/-- The boxed rewrite at every multiplicity. -/
theorem bun3 (d : ℕ) (j : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [j])) 1 bun3L ((1) ^ j) bun3R := by
  induction j with
  | zero => exact Equal.of_zero_at (F d) 0 65536 [0] (by simp) bun3L bun3R (by decide : Spec.splitOK (SpC 0) 65536 bun3L = true) (by decide : Spec.splitOK (SpC 0) 65536 bun3R = true) (by decide) (Equal.rescale (isoB ((F d).env (ksOf [0])) 1 (Net.zero (SpC 0) bun3L) (Net.zero (SpC 0) bun3R) [] [] (by chk)) one_ne_zero (by simp))
  | succ j ih => exact Equal.of_split_at (F d) 0 65536 j [j] (by simp) bun3L bun3R (by decide : Spec.splitOK (SpC 0) 65536 bun3L = true) (by decide : Spec.splitOK (SpC 0) 65536 bun3R = true) (fun _ _ _ => Fam.dimv_pos _ _ _ (Fpos d h_d _)) (Equal.rescale (Equal.mul_trans (isoB ((F d).env (ksOf [j])) 1 (Net.split (SpC 0) 65536 bun3L) ⟨([⟨.X, [65546], [], 0, 0⟩, ⟨.X, [65537], [65545, 65546], 0, 0⟩] : List Node) ++ bun3L.nodes, [65546] ++ bun3L.int, [1, 65537, 9, 65545]⟩ [] [1, 3, 0, 2] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [j])) ([⟨.X, [65546], [], 0, 0⟩, ⟨.X, [65537], [65545, 65546], 0, 0⟩] : List Node) bun3L.nodes bun3R.nodes [65546] bun3L.int bun3R.int [1, 65537, 9, 65545] bun3L.bnd 1 ((1) ^ j) (Equal.toLocal _ _ _ ih) (by chk) (by chk) (by chk)) (Equal.mul_trans (isoB ((F d).env (ksOf [j])) 1 ⟨([⟨.X, [65546], [], 0, 0⟩, ⟨.X, [65537], [65545, 65546], 0, 0⟩] : List Node) ++ bun3R.nodes, [65546] ++ bun3R.int, [1, 65537, 9, 65545]⟩ ⟨bun3R.nodes ++ ([⟨.X, [65546], [], 0, 0⟩, ⟨.X, [65537], [65545, 65546], 0, 0⟩] : List Node), bun3R.int ++ [65546], [1, 65537, 9, 65545]⟩ [] [2, 0, 1] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [j])) bun3R.nodes ([⟨.X, [65546], [], 0, 0⟩, ⟨.X, [65537], [65545, 65546], 0, 0⟩] : List Node) ([⟨.X, [65537], [65545], 0, 0⟩] : List Node) bun3R.int [65546] [] [1, 65537, 9, 65545] [65537, 65545] 1 (1) (fusionX ((F d).env (ksOf [j])) 0 0 0 0 65546 [65546] [] [65537] [65545, 65546] [65537] [65545] [65537, 65545] (by pos) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by chk) (by phs)) (by chk) (by chk) (by chk)) (isoB ((F d).env (ksOf [j])) 1 ⟨bun3R.nodes ++ ([⟨.X, [65537], [65545], 0, 0⟩] : List Node), bun3R.int ++ [], [1, 65537, 9, 65545]⟩ (Net.split (SpC 0) 65536 bun3R) [] [0, 1] (by chk)))))) (by norm_num) (by ring))

theorem step4 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N3 ((1) ^ n) N4 := by
  refine Equal.trans (isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.ZB [[0]], [0], [9], 0, 0⟩, ⟨.Bun 0 .X, [10], [], 0, 0⟩, ⟨.Bun 0 .X, [1], [9, 10], 0, 0⟩] : List Node), [9, 10], [0, 1]⟩ [] [2, 1, 0] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [n])) ([⟨.ZB [[0]], [0], [9], 0, 0⟩] : List Node) ([⟨.Bun 0 .X, [10], [], 0, 0⟩, ⟨.Bun 0 .X, [1], [9, 10], 0, 0⟩] : List Node) ([⟨.Bun 0 .X, [1], [9], 0, 0⟩] : List Node) [9] [10] [] [0, 1] [1, 9] 1 ((1) ^ n) ((1) ^ n) (Equal.toLocal _ _ _ (bun3 d n h_d)) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [n])) ((1) ^ n) _ ⟨([⟨.ZB [[0]], [0], [11], 0, 0⟩, ⟨.Bun 0 .X, [1], [11], 0, 0⟩] : List Node), [11], [0, 1]⟩ [(9, 11), (11, 9)] [0, 1] (by chk)

/-- (1/sqrt(d)) Σ_{k=0}^{d-1} |k⟩ |k⟩^{n} -/
def N5 : Net := ⟨([⟨.ZB [[0]], [0], [1], 0, 0⟩] : List Node), [], [0, 1]⟩

/-- The boxed region before `bun5`. -/
def bun5L : Net := ⟨([⟨.Bun 0 .X, [1], [11], 0, 0⟩] : List Node), [], [1, 11]⟩

/-- The boxed region after `bun5`. -/
def bun5R : Net := ⟨([⟨.Bun 0 .I, [1], [11], 0, 0⟩] : List Node), [], [1, 11]⟩

/-- The boxed rewrite at every multiplicity. -/
theorem bun5 (d : ℕ) (j : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [j])) 1 bun5L ((1) ^ j) bun5R := by
  induction j with
  | zero => exact Equal.of_zero_at (F d) 0 65536 [0] (by simp) bun5L bun5R (by decide : Spec.splitOK (SpC 0) 65536 bun5L = true) (by decide : Spec.splitOK (SpC 0) 65536 bun5R = true) (by decide) (Equal.rescale (isoB ((F d).env (ksOf [0])) 1 (Net.zero (SpC 0) bun5L) (Net.zero (SpC 0) bun5R) [] [] (by chk)) one_ne_zero (by simp))
  | succ j ih => exact Equal.of_split_at (F d) 0 65536 j [j] (by simp) bun5L bun5R (by decide : Spec.splitOK (SpC 0) 65536 bun5L = true) (by decide : Spec.splitOK (SpC 0) 65536 bun5R = true) (fun _ _ _ => Fam.dimv_pos _ _ _ (Fpos d h_d _)) (Equal.rescale (Equal.mul_trans (isoB ((F d).env (ksOf [j])) 1 (Net.split (SpC 0) 65536 bun5L) ⟨([⟨.X, [65537], [65547], 0, 0⟩] : List Node) ++ bun5L.nodes, [] ++ bun5L.int, [1, 65537, 11, 65547]⟩ [] [1, 0] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [j])) ([⟨.X, [65537], [65547], 0, 0⟩] : List Node) bun5L.nodes bun5R.nodes [] bun5L.int bun5R.int [1, 65537, 11, 65547] bun5L.bnd 1 ((1) ^ j) (Equal.toLocal _ _ _ ih) (by chk) (by chk) (by chk)) (Equal.mul_trans (isoB ((F d).env (ksOf [j])) 1 ⟨([⟨.X, [65537], [65547], 0, 0⟩] : List Node) ++ bun5R.nodes, [] ++ bun5R.int, [1, 65537, 11, 65547]⟩ ⟨bun5R.nodes ++ ([⟨.X, [65537], [65547], 0, 0⟩] : List Node), bun5R.int ++ [], [1, 65537, 11, 65547]⟩ [] [1, 0] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [j])) bun5R.nodes ([⟨.X, [65537], [65547], 0, 0⟩] : List Node) ([⟨.I, [65537], [65547], 0, 0⟩] : List Node) bun5R.int [] [] [1, 65537, 11, 65547] [65537, 65547] 1 (1) (identityX ((F d).env (ksOf [j])) 0 0 65537 65547 65547 65537 [65537, 65547] (by pos) (by chk) (by chk) (by chk) (by chk) (by phz)) (by chk) (by chk) (by chk)) (isoB ((F d).env (ksOf [j])) 1 ⟨bun5R.nodes ++ ([⟨.I, [65537], [65547], 0, 0⟩] : List Node), bun5R.int ++ [], [1, 65537, 11, 65547]⟩ (Net.split (SpC 0) 65536 bun5R) [] [0, 1] (by chk)))))) (by norm_num) (by ring))

theorem step6 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N4 ((1) ^ n) N5 := by
  refine Equal.trans (isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.ZB [[0]], [0], [11], 0, 0⟩, ⟨.Bun 0 .X, [1], [11], 0, 0⟩] : List Node), [11], [0, 1]⟩ [] [0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [n])) ([⟨.ZB [[0]], [0], [11], 0, 0⟩] : List Node) ([⟨.Bun 0 .X, [1], [11], 0, 0⟩] : List Node) ([⟨.Bun 0 .I, [1], [11], 0, 0⟩] : List Node) [11] [] [] [0, 1] [1, 11] 1 ((1) ^ n) ((1) ^ n) (Equal.toLocal _ _ _ (bun5 d n h_d)) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  refine Equal.trans (isoB ((F d).env (ksOf [n])) ((1) ^ n) _ ⟨([⟨.ZB [[0]], [0], [11], 0, 0⟩, ⟨.Bun 0 .I, [1], [11], 0, 0⟩] : List Node), [11], [0, 1]⟩ [] [0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [n])) ([⟨.ZB [[0]], [0], [11], 0, 0⟩] : List Node) ([⟨.Bun 0 .I, [1], [11], 0, 0⟩] : List Node) ([⟨.I, [1], [11], 1, 0⟩] : List Node) [11] [] [] [0, 1] [1, 11] ((1) ^ n) 1 ((1) ^ n) (bunI ((F d).env (ksOf [n])) [0] 0 1 1 11 [1, 11] (Fam.env_dimv_of _ _ _ _ _ rfl rfl ⟨rfl, rfl⟩) (by pos) (by chk) (by chk) (by chk) (by chk)) (by chk) (by chk) (by chk) (mul_one _).symm) ?_
  refine Equal.trans (isoB ((F d).env (ksOf [n])) ((1) ^ n) _ ⟨([⟨.ZB [[0]], [0], [11], 0, 0⟩, ⟨.I, [1], [11], 1, 0⟩] : List Node), [11], [0, 1]⟩ [] [0, 1] (by chk)) ?_
  refine Equal.trans (wire_elim ((F d).env (ksOf [n])) ((1) ^ n) ([⟨.ZB [[0]], [0], [11], 0, 0⟩] : List Node) ([] : List Node) [11] [0, 1] 1 11 1 (by chk) (by chk) (by chk) (by chk) (by chk)) ?_
  exact isoB ((F d).env (ksOf [n])) ((1) ^ n) _ ⟨([⟨.ZB [[0]], [0], [1], 0, 0⟩] : List Node), [], [0, 1]⟩ [] [0] (by chk)

theorem step7 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N5 1 N1 := by
  exact isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.ZB [[0]], [0], [1], 0, 0⟩] : List Node), [], [0, 1]⟩ [] [0] (by chk)

/-- The start and goal diagrams with their own scalars. -/
theorem main (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) (((rt d))⁻¹) N0 (((rt d))⁻¹) N1 :=
  Equal.rescale (Equal.mul_trans (Equal.mul_trans (Equal.refl ((F d).env (ksOf [n])) 1 N0) (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans (step1 d n h_d) (step2 d n h_d)) (step4 d n h_d)) (step6 d n h_d))) (Equal.mul_trans (step7 d n h_d) (Equal.mul_trans ((Equal.refl ((F d).env (ksOf [n])) 1 N1)).symm (Equal.refl ((F d).env (ksOf [n])) 1 N1))))
    (mul_ne_zero (mul_ne_zero one_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero one_ne_zero one_ne_zero) one_ne_zero) one_ne_zero)) (mul_ne_zero one_ne_zero (mul_ne_zero one_ne_zero one_ne_zero)))
    (by ring)

end ArchytasZX.Export
