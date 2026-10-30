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
def labs : List (ℕ × ℕ) := [(0, 1), (1, 1), (65536, 0), (65537, 0)]

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

/-- (d^(-n/2)) Σ_{q0[i0],j0o0[i0],j1o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-1/2} ω_{d}^{q0[i0] (j0o0[i0])} d^{-1/2} ω_{d}^{j1o0[i0] j0o0[i0]} ) ⊗_{i0=1}^{n}( |j1o0[i0]⟩ ) -/
def N0 : Net := ⟨([⟨.Bun 0 .X, [1], [], 0, 0⟩, ⟨.Bun 0 .F, [0], [1], 0, 0⟩] : List Node), [1], [0]⟩

/-- (d^(-n/2)) Σ_{k0[i0]=0, i0=1..n}^{d-1} ⊗_{i0=1}^{n}( |k0[i0]⟩ ) -/
def N1 : Net := ⟨([⟨.Bun 0 .Z, [0], [], 0, 0⟩] : List Node), [], [0]⟩

/-- (d^(-n/2)) Σ_{k2[i0]=0, i0=1..n}^{d-1} ⊗_{i0=1}^{n}( |k2[i0]⟩ ) -/
def N2 : Net := ⟨([⟨.Bun 0 .Z, [0], [], 0, 0⟩] : List Node), [], [0]⟩

/-- The boxed region before `bun1`. -/
def bun1L : Net := ⟨([⟨.Bun 0 .X, [1], [], 0, 0⟩, ⟨.Bun 0 .F, [0], [1], 0, 0⟩] : List Node), [1], [0]⟩

/-- The boxed region after `bun1`. -/
def bun1R : Net := ⟨([⟨.Bun 0 .Z, [0], [], 0, 0⟩] : List Node), [], [0]⟩

/-- The boxed rewrite at every multiplicity. -/
theorem bun1 (d : ℕ) (j : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [j])) 1 bun1L ((1) ^ j) bun1R := by
  induction j with
  | zero => exact Equal.of_zero_at (F d) 0 65536 [0] (by simp) bun1L bun1R (by decide : Spec.splitOK (SpC 0) 65536 bun1L = true) (by decide : Spec.splitOK (SpC 0) 65536 bun1R = true) (by decide) (Equal.rescale (isoB ((F d).env (ksOf [0])) 1 (Net.zero (SpC 0) bun1L) (Net.zero (SpC 0) bun1R) [] [] (by chk)) one_ne_zero (by simp))
  | succ j ih => exact Equal.of_split_at (F d) 0 65536 j [j] (by simp) bun1L bun1R (by decide : Spec.splitOK (SpC 0) 65536 bun1L = true) (by decide : Spec.splitOK (SpC 0) 65536 bun1R = true) (fun _ _ _ => Fam.dimv_pos _ _ _ (Fpos d h_d _)) (Equal.rescale (Equal.mul_trans (isoB ((F d).env (ksOf [j])) 1 (Net.split (SpC 0) 65536 bun1L) ⟨([⟨.X, [65537], [], 0, 0⟩, ⟨.F, [65536], [65537], 0, 0⟩] : List Node) ++ bun1L.nodes, [65537] ++ bun1L.int, [0, 65536]⟩ [] [1, 3, 0, 2] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [j])) ([⟨.X, [65537], [], 0, 0⟩, ⟨.F, [65536], [65537], 0, 0⟩] : List Node) bun1L.nodes bun1R.nodes [65537] bun1L.int bun1R.int [0, 65536] bun1L.bnd 1 ((1) ^ j) (Equal.toLocal _ _ _ ih) (by chk) (by chk) (by chk)) (Equal.mul_trans (isoB ((F d).env (ksOf [j])) 1 ⟨([⟨.X, [65537], [], 0, 0⟩, ⟨.F, [65536], [65537], 0, 0⟩] : List Node) ++ bun1R.nodes, [65537] ++ bun1R.int, [0, 65536]⟩ ⟨bun1R.nodes ++ ([⟨.X, [65537], [], 0, 0⟩, ⟨.F, [65536], [65537], 0, 0⟩] : List Node), bun1R.int ++ [65537], [0, 65536]⟩ [] [2, 0, 1] (by chk)) (Equal.mul_trans (plug ((F d).env (ksOf [j])) bun1R.nodes ([⟨.X, [65537], [], 0, 0⟩, ⟨.F, [65536], [65537], 0, 0⟩] : List Node) ([⟨.Z, [65536], [], 0, 0⟩] : List Node) bun1R.int [65537] [] [0, 65536] [65536] 1 (1) (colorStateX ((F d).env (ksOf [j])) 0 0 0 65537 65536 [65536] (by pos) (by chk) (by chk) (by chk) (by chk) (by phz) (by phz)) (by chk) (by chk) (by chk)) (isoB ((F d).env (ksOf [j])) 1 ⟨bun1R.nodes ++ ([⟨.Z, [65536], [], 0, 0⟩] : List Node), bun1R.int ++ [], [0, 65536]⟩ (Net.split (SpC 0) 65536 bun1R) [] [0, 1] (by chk)))))) (by norm_num) (by ring))

theorem step2 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N0 ((1) ^ n) N2 := by
  refine Equal.trans (isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.Bun 0 .X, [1], [], 0, 0⟩, ⟨.Bun 0 .F, [0], [1], 0, 0⟩] : List Node), [1], [0]⟩ [] [0, 1] (by chk)) ?_
  refine Equal.trans (rewrite ((F d).env (ksOf [n])) ([] : List Node) ([⟨.Bun 0 .X, [1], [], 0, 0⟩, ⟨.Bun 0 .F, [0], [1], 0, 0⟩] : List Node) ([⟨.Bun 0 .Z, [0], [], 0, 0⟩] : List Node) [] [1] [] [0] [0] 1 ((1) ^ n) ((1) ^ n) (Equal.toLocal _ _ _ (bun1 d n h_d)) (by chk) (by chk) (by chk) (one_mul _).symm) ?_
  exact isoB ((F d).env (ksOf [n])) ((1) ^ n) _ ⟨([⟨.Bun 0 .Z, [0], [], 0, 0⟩] : List Node), [], [0]⟩ [] [0] (by chk)

theorem step3 (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) 1 N2 1 N1 := by
  exact isoB ((F d).env (ksOf [n])) 1 _ ⟨([⟨.Bun 0 .Z, [0], [], 0, 0⟩] : List Node), [], [0]⟩ [] [0] (by chk)

/-- The start and goal diagrams with their own scalars. -/
theorem main (d : ℕ) (n : ℕ) (h_d : 0 < d) :
    Equal ((F d).env (ksOf [n])) (((rt d))⁻¹ ^ n) N0 (((rt d))⁻¹ ^ n) N1 :=
  Equal.rescale (Equal.mul_trans (Equal.mul_trans (Equal.refl ((F d).env (ksOf [n])) 1 N0) (step2 d n h_d)) (Equal.mul_trans (step3 d n h_d) (Equal.mul_trans ((Equal.refl ((F d).env (ksOf [n])) 1 N1)).symm (Equal.refl ((F d).env (ksOf [n])) 1 N1))))
    (mul_ne_zero (mul_ne_zero one_ne_zero one_ne_zero) (mul_ne_zero one_ne_zero (mul_ne_zero one_ne_zero one_ne_zero)))
    (by ring)

end ArchytasZX.Export
