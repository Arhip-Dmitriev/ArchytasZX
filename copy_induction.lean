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
def labs : List (ℕ × ℕ) := [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (65536, 0)]

/-- The environment. -/
noncomputable def E (d : ℕ) : Env := { dimv := Δ d, phase := Φ d, labs := labs }

/-- Each bundle dimension index with its base index and multiplicity. -/
def bundL : List (ℕ × ℕ × ℕ) := [(1, 0, 0)]

/-- The family, its multiplicities in order: n = n, k = n. -/
noncomputable def F (d : ℕ) : Fam := { dims := Δ d, bund := bundL, phase := Φ d, labs := labs }

/-- Every dimension value of the family is positive. -/
theorem Fpos (d : ℕ) (h_d : 0 < d) : ∀ i, 0 < (F d).dims i := fun i => Δpos d h_d i

/-- The family's structure alone. -/
noncomputable def FC : Fam := { dims := fun _ => 1, bund := bundL, phase := fun _ _ => 0, labs := labs }

/-- The family's structure along multiplicity `m`. -/
noncomputable def SpC (m : ℕ) : Spec := ⟨FC, m, ksOf []⟩

/-- Σ_{k1,q0=0}^{d-1} d^{-1/2} ω_{d}^{q0 (k1)} |k1⟩^{n} -/
def N0 : Net := ⟨([⟨.X, [1], [], 0, 0⟩, ⟨.ZB [[0]], [], [1, 0], 0, 0⟩] : List Node), [1], [0]⟩

/-- (d^(1/2-n/2)) Σ_{q0[i0],j0o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-1/2} ω_{d}^{q0[i0] (j0o0[i0])} ) ⊗_{i0=1}^{n}( |j0o0[i0]⟩ ) -/
def N1 : Net := ⟨([⟨.Bun 0 .X, [0], [], 0, 0⟩] : List Node), [], [0]⟩

/-- The start's scalar. -/
noncomputable def sS (d : ℕ) (n : ℕ) : ℂ := 1

/-- The goal's scalar. -/
noncomputable def sG (d : ℕ) (n : ℕ) : ℂ := ((((rt d))⁻¹ ^ n) * (rt d))

/-- Σ_{k2,q0=0}^{d-1} d^{-1/2} ω_{d}^{q0 (k2)} -/
def N2 : Net := ⟨([⟨.X, [2], [], 0, 0⟩, ⟨.Z, [], [2], 0, 0⟩] : List Node), [2], []⟩

/-- sqrt(d) -/
def N3 : Net := ⟨([] : List Node), [], []⟩

theorem step1 (d : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [0])) 1 N2 (rt d) N3 := by
  refine Equal.trans (isoB ((F d).env (ksOf [0])) 1 _ ⟨([⟨.X, [2], [], 0, 0⟩, ⟨.Z, [], [2], 0, 0⟩] : List Node), [2], []⟩ [] [0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [0])) ([] : List Node) ([⟨.X, [2], [], 0, 0⟩, ⟨.Z, [], [2], 0, 0⟩] : List Node) ([] : List Node) [] [2] [] [] [] 1 (rt d) (rt d) (stateCopy ((F d).env (ksOf [0])) 0 0 0 0 2 [] [] (rt d) (by pos) (by chk) (by chk) (by chk) (by phz) (by phz) (by phz) (show (rt d) * (rt d) ^ 0 = (rt d) from (by ring))) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [0])) (rt d) _ ⟨([] : List Node), [], []⟩ [] [] (by chk)

theorem zcase (d : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [0])) (sS d 0) (Net.zero (SpC 0) N0) (sG d 0) (Net.zero (SpC 0) N1) :=
  Equal.rescale (Equal.mul_trans (isoB ((F d).env (ksOf [0])) 1 (Net.zero (SpC 0) N0) N2 [(1, 2), (2, 1)] [0, 1] (by chk)) (Equal.mul_trans (Equal.mul_trans (step1 d h_d) (Equal.mul_trans (Equal.refl ((F d).env (ksOf [0])) 1 N3) ((Equal.refl ((F d).env (ksOf [0])) 1 N3)).symm)) (isoB ((F d).env (ksOf [0])) 1 N3 (Net.zero (SpC 0) N1) [] [] (by chk))))
    (by (try simp only [sS, sG]); nzt)
    (by simp only [sS, sG]; ring)

/-- Σ_{k2,q0=0}^{d-1} d^{-1/2} ω_{d}^{q0 (k2)} |k2⟩^{k} |k2⟩ -/
def N4 : Net := ⟨([⟨.X, [3], [], 0, 0⟩, ⟨.ZB [[0]], [65536], [3, 0], 0, 0⟩] : List Node), [3], [0, 65536]⟩

/-- Σ_{k3,k4,q0=0}^{d-1} d^{-1/2} ω_{d}^{q0 (k3)} [k3 = k4] |k4⟩^{k} |k3⟩ -/
def N5 : Net := ⟨([⟨.X, [4], [], 0, 0⟩, ⟨.Z, [65536, 5], [4], 0, 0⟩, ⟨.ZB [[0]], [], [5, 0], 0, 0⟩] : List Node), [4, 5], [0, 65536]⟩

theorem step2 (d : ℕ) (k : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [k])) 1 N4 1 N5 := by
  refine Equal.trans (isoB ((F d).env (ksOf [k])) 1 _ ⟨([⟨.X, [3], [], 0, 0⟩, ⟨.ZB [[0]], [65536], [3, 0], 0, 0⟩] : List Node), [3], [0, 65536]⟩ [] [0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [k])) ([⟨.X, [3], [], 0, 0⟩] : List Node) ([⟨.ZB [[0]], [65536], [3, 0], 0, 0⟩] : List Node) ([⟨.Z, [65536, 6], [3], 0, 0⟩, ⟨.ZB [[0]], [], [6, 0], 0, 0⟩] : List Node) [3] [] [6] [0, 65536] [65536, 3, 0] 1 1 1 (unfuseZB ((F d).env (ksOf [k])) [] [0] 0 0 0 6 0 [65536] [3] [] [65536, 6] [3] [65536, 3, 0] (by chk) (by chk) (by chk) (by chk) (by phz)) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [k])) 1 _ ⟨([⟨.X, [4], [], 0, 0⟩, ⟨.Z, [65536, 5], [4], 0, 0⟩, ⟨.ZB [[0]], [], [5, 0], 0, 0⟩] : List Node), [4, 5], [0, 65536]⟩ [(3, 4), (6, 5), (4, 3), (5, 6)] [0, 1, 2] (by chk)

/-- (1/sqrt(d)) Σ_{k4,q5,q6,j5o0=0}^{d-1} d^{-1/2} ω_{d}^{q5 (j5o0)} d^{-1/2} ω_{d}^{q6 (k4)} |k4⟩^{k} |j5o0⟩ -/
def N6 : Net := ⟨([⟨.ZB [[0]], [], [7, 0], 0, 0⟩, ⟨.X, [65536], [], 0, 0⟩, ⟨.X, [7], [], 0, 0⟩] : List Node), [7], [0, 65536]⟩

theorem step3 (d : ℕ) (k : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [k])) 1 N5 (((rt d))⁻¹) N6 := by
  refine Equal.trans (isoB ((F d).env (ksOf [k])) 1 _ ⟨([⟨.ZB [[0]], [], [5, 0], 0, 0⟩, ⟨.X, [4], [], 0, 0⟩, ⟨.Z, [65536, 5], [4], 0, 0⟩] : List Node), [5, 4], [0, 65536]⟩ [] [2, 0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [k])) ([⟨.ZB [[0]], [], [5, 0], 0, 0⟩] : List Node) ([⟨.X, [4], [], 0, 0⟩, ⟨.Z, [65536, 5], [4], 0, 0⟩] : List Node) ([⟨.X, [65536], [], 0, 0⟩, ⟨.X, [5], [], 0, 0⟩] : List Node) [5] [4] [] [0, 65536] [65536, 5] 1 (((rt d))⁻¹) (((rt d))⁻¹) (stateCopy ((F d).env (ksOf [k])) 0 0 0 0 4 [65536, 5] [65536, 5] (((rt d))⁻¹) (by pos) (by chk) (by chk) (by chk) (by phz) (by phz) (by phz) (show ((rt d))⁻¹ * (rt d) ^ 2 = (rt d) from (by have r0 := (mul_inv_cancel₀ (rt_ne d (by first | omega | norm_num | positivity | (apply Nat.mul_pos <;> omega) | exact Nat.pos_pow_of_pos _ (by omega)))); linear_combination ((rt d)) * r0))) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [k])) (((rt d))⁻¹) _ ⟨([⟨.ZB [[0]], [], [7, 0], 0, 0⟩, ⟨.X, [65536], [], 0, 0⟩, ⟨.X, [7], [], 0, 0⟩] : List Node), [7], [0, 65536]⟩ [(5, 7), (7, 5)] [0, 1, 2] (by chk)

/-- (d^(-k/2)) Σ_{q5=0}^{d-1} Σ_{q7[i1]=0, i1=1..k}^{d-1} Σ_{j5o0=0}^{d-1} Σ_{j7o0[i1]=0, i1=1..k}^{d-1} d^{-1/2} ω_{d}^{q5 (j5o0)} ∏_{i1=1}^{k}( d^{-1/2} ω_{d}^{q7[i1] (j7o0[i1])} ) ⊗_{i1=1}^{k}( |j7o0[i1]⟩ ) |j5o0⟩ -/
def N7 : Net := ⟨([⟨.X, [65536], [], 0, 0⟩, ⟨.Bun 0 .X, [0], [], 0, 0⟩] : List Node), [], [0, 65536]⟩

/-- (d^(-k/2)) Σ_{q0[i0]=0, i0=1..k}^{d-1} Σ_{q1=0}^{d-1} Σ_{j0o0[i0]=0, i0=1..k}^{d-1} Σ_{j1o0=0}^{d-1} ∏_{i0=1}^{k}( d^{-1/2} ω_{d}^{q0[i0] (j0o0[i0])} ) d^{-1/2} ω_{d}^{q1 (j1o0)} ⊗_{i0=1}^{k}( |j0o0[i0]⟩ ) |j1o0⟩ -/
def N8 : Net := ⟨([⟨.Bun 0 .X, [0], [], 0, 0⟩, ⟨.X, [65536], [], 0, 0⟩] : List Node), [], [0, 65536]⟩

theorem step4 (d : ℕ) (k : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [k])) 1 N7 1 N8 := by
  exact isoB ((F d).env (ksOf [k])) 1 _ ⟨([⟨.Bun 0 .X, [0], [], 0, 0⟩, ⟨.X, [65536], [], 0, 0⟩] : List Node), [], [0, 65536]⟩ [] [1, 0] (by chk)

theorem scase (d : ℕ) (h_d : 0 < d) (k : ℕ)
    (ih : Equal ((F d).env (ksOf [k])) (sS d k) N0 (sG d k) N1) :
    Equal ((F d).env (ksOf [k])) (sS d (k + 1)) (Net.split (SpC 0) 65536 N0) (sG d (k + 1)) (Net.split (SpC 0) 65536 N1) :=
  Equal.rescale (Equal.mul_trans (isoB ((F d).env (ksOf [k])) 1 (Net.split (SpC 0) 65536 N0) N4 [(1, 3), (3, 1)] [0, 1] (by chk)) (Equal.mul_trans (Equal.mul_trans (step2 d k h_d) (step3 d k h_d)) (Equal.mul_trans (isoB ((F d).env (ksOf [k])) 1 N6 ⟨([⟨.X, [65536], [], 0, 0⟩] : List Node) ++ (Net.rename [(1, 7), (7, 1)] N0).nodes, [] ++ (Net.rename [(1, 7), (7, 1)] N0).int, [0, 65536]⟩ [] [1, 2, 0] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [k])) ([⟨.X, [65536], [], 0, 0⟩] : List Node) (Net.rename [(1, 7), (7, 1)] N0).nodes (Net.rename [(1, 7), (7, 1)] N1).nodes [] (Net.rename [(1, 7), (7, 1)] N0).int (Net.rename [(1, 7), (7, 1)] N1).int [0, 65536] (Net.rename [(1, 7), (7, 1)] N0).bnd (sS d k) (sG d k) (Equal.toLocal _ _ _ (Equal.rename ((F d).env (ksOf [k])) [(1, 7), (7, 1)] N0 N1 (by chk) (by chk) ih)) (by chk) (by chk) (by chk)) (Equal.mul_trans (isoB ((F d).env (ksOf [k])) 1 ⟨([⟨.X, [65536], [], 0, 0⟩] : List Node) ++ (Net.rename [(1, 7), (7, 1)] N1).nodes, [] ++ (Net.rename [(1, 7), (7, 1)] N1).int, [0, 65536]⟩ N7 [] [0, 1] (by chk)) (Equal.mul_trans (Equal.mul_trans (Equal.refl ((F d).env (ksOf [k])) 1 N7) (Equal.mul_trans (step4 d k h_d) ((Equal.refl ((F d).env (ksOf [k])) 1 N8)).symm)) (isoB ((F d).env (ksOf [k])) 1 N8 (Net.split (SpC 0) 65536 N1) [] [0, 1] (by chk))))))))
    (by (try simp only [sS, sG]); nzt)
    (by simp only [sS, sG]; ring)

/-- The start and goal families with their own scalars, for every `n`. -/
theorem main (d : ℕ) (h_d : 0 < d) :
    ∀ n : ℕ, Equal ((F d).env (ksOf [n])) (sS d n) N0 (sG d n) N1 := by
  intro n
  induction n with
  | zero => exact Equal.of_zero_at (F d) 0 65536 [0] (by simp) N0 N1 (by decide : Spec.splitOK (SpC 0) 65536 N0 = true) (by decide : Spec.splitOK (SpC 0) 65536 N1 = true) (by decide) (zcase d h_d)
  | succ k ih => exact Equal.of_split_at (F d) 0 65536 k [k] (by simp) N0 N1 (by decide : Spec.splitOK (SpC 0) 65536 N0 = true) (by decide : Spec.splitOK (SpC 0) 65536 N1 = true) (fun _ _ _ => Fam.dimv_pos _ _ _ (Fpos d h_d _)) (scase d h_d k ih)

end ArchytasZX.Export
