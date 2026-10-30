/-
  ArchytasZX proof export over the tensor-network semantics of the ArchytasZX Lean
  library: every diagram is a Net, every rule step an instance of a lemma proved
  there. `#print axioms` lists any step stated without proof.
  refutation at d = 3, n = 1, ζ = om 12
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
  | 1 => fun k => (0) + (if k = 0 then 0 else ((k : ℝ) * ((d : ℝ) ^ 1)⁻¹))
  | _ => fun _ => 0

/-- Each label's dimension index. -/
def labs : List (ℕ × ℕ) := [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (65536, 0), (65537, 0), (65538, 0), (65539, 0), (65540, 0), (65541, 0)]

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

/-- (d^(-n/2)) Σ_{k2[i0],q0[i0],j0o0[i0],j3o0[i0],j4o0[i0],j5o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-1/2} ω_{d}^{q0[i0] (j0o0[i0])} d^{-1/2} ω_{d}^{k2[i0] j0o0[i0]} e^{2 π i ((k2[i0]/d))} d^{-1/2} ω_{d}^{j3o0[i0] k2[i0]} d^{-1/2} ω_{d}^{j4o0[i0] j3o0[i0]} d^{-1/2} ω_{d}^{j5o0[i0] j4o0[i0]} ) ⊗_{i0=1}^{n}( |j5o0[i0]⟩ ) -/
def N0 : Net := ⟨([⟨.Bun 0 .X, [1], [], 0, 0⟩, ⟨.Bun 0 .F, [2], [1], 0, 0⟩, ⟨.Bun 0 .Z, [3], [2], 0, 1⟩, ⟨.Bun 0 .F, [4], [3], 0, 0⟩, ⟨.Bun 0 .F, [5], [4], 0, 0⟩, ⟨.Bun 0 .F, [0], [5], 0, 0⟩] : List Node), [1, 2, 3, 4, 5], [0]⟩

/-- (d^(-n/2)) Σ_{q0[i0],j0o0[i0]=0, i0=1..n}^{d-1} ∏_{i0=1}^{n}( d^{-1/2} e^{2 π i ((q0[i0]/d))} ω_{d}^{q0[i0] (j0o0[i0])} ) ⊗_{i0=1}^{n}( |j0o0[i0]⟩ ) -/
def N1 : Net := ⟨([⟨.Bun 0 .X, [0], [], 0, 1⟩] : List Node), [], [0]⟩

/-- Each phase entry's power of `om 12`. -/
def Pt (i k : ℕ) : ℕ × List ℤ := ((([((0, 0), (0, [])), ((0, 1), (0, [])), ((0, 2), (0, [])), ((1, 0), (0, [])), ((1, 1), (4, [])), ((1, 2), (8, []))] : List ((ℕ × ℕ) × (ℕ × List ℤ))).lookup (i, k)).getD (0, []))

/-- The boundary assignment where the diagrams differ. -/
def σ0 (l : ℕ) : ℕ := ((([(0, 1)] : List (ℕ × ℕ)).lookup l).getD 0)

set_option maxHeartbeats 0 in
/-- The diagrams are not equal: they differ at d = 3, n = 1. -/
theorem main :
    ¬ ∀ (d : ℕ) (n : ℕ) (h_d : 0 < d), Equal ((F d).env (ksOf [n])) (((rt d))⁻¹ ^ n) N0 (((rt d))⁻¹ ^ n) N1 := by
  intro h
  have he := h 3 1 (by norm_num)
  refine refute σ0 (inRange_of _ _ _ (by decide +kernel)) ?_ he
  intro h0
  have hP : PhOK ((F 3).env (ksOf [1])) 12 [] Pt [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] := ⟨ex_atoms 12 0 (0) [] [] (by norm_num) (by norm_num [Fam.env, F, Φ]), ex_atoms 12 0 (0) [] [] (by norm_num) (by norm_num [Fam.env, F, Φ]), ex_atoms 12 0 (0) [] [] (by norm_num) (by norm_num [Fam.env, F, Φ]), ex_atoms 12 0 (0) [] [] (by norm_num) (by norm_num [Fam.env, F, Φ]), ex_atoms 12 4 (0) [] [] (by norm_num) (by norm_num [Fam.env, F, Φ]), ex_atoms 12 8 (0) [] [] (by norm_num) (by norm_num [Fam.env, F, Φ]), trivial⟩
  rw [eval_cyc ((F 3).env (ksOf [1])) 12 [] Pt [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] N0 σ0 (by norm_num) (by decide +kernel) hP, eval_cyc ((F 3).env (ksOf [1])) 12 [] Pt [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] N1 σ0 (by norm_num) (by decide +kernel) hP] at h0
  unfold sqK at h0
  rw [show sqL ((F 3).env (ksOf [1])).shape N0 = [(3, 1), (3, 1), (3, 0), (3, 1), (3, 1), (3, 1)] from by decide +kernel, show evalG ((F 3).env (ksOf [1])).shape 12 Pt N0 σ0 = [(([] : List ℤ), [99, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0])] from by decide +kernel, show sqL ((F 3).env (ksOf [1])).shape N1 = [(3, 1)] from by decide +kernel, show evalG ((F 3).env (ksOf [1])).shape 12 Pt N1 σ0 = [(([] : List ℤ), [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])] from by decide +kernel] at h0
  simp only [sqK, gval, mono_nil, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, List.sum_cons, List.sum_nil, one_mul, cvz] at h0
  try push_cast at h0
  have hzM : om 12 ^ 12 = 1 := om_pow_self 12 (by norm_num)
  have hg2 : (1 : ℂ) + om 12 ^ 6 = 0 := by simpa [Finset.sum_range_succ] using om_geom 12 2 (by norm_num) (by norm_num) (by norm_num)
  have hg3 : (1 : ℂ) + om 12 ^ 4 + om 12 ^ 8 = 0 := by simpa [Finset.sum_range_succ] using om_geom 12 3 (by norm_num) (by norm_num) (by norm_num)
  have hsq0 : ((rt 3))⁻¹ ^ 2 = ((1 : ℂ) / 3) := by rw [inv_pow, rt_sq' 3]; ring
  exact cyc_ne 12 (by norm_num) [1] (by decide +kernel) (by decide +kernel)
    (by simp only [cvz]; push_cast; linear_combination (1) * h0 + (((-10 : ℂ) + ((-99 : ℂ) * (((rt 3))⁻¹ ^ 4)) + ((-33 : ℂ) * (((rt 3))⁻¹ ^ 2)) + ((-7 : ℂ) * ((om 12) ^ 4)) + ((-7 : ℂ) * ((om 12) ^ 8)) + ((-72 : ℂ) * (((rt 3))⁻¹ ^ 4) * ((om 12) ^ 4)) + ((-72 : ℂ) * (((rt 3))⁻¹ ^ 4) * ((om 12) ^ 8)) + ((-24 : ℂ) * (((rt 3))⁻¹ ^ 2) * ((om 12) ^ 4)) + ((-24 : ℂ) * (((rt 3))⁻¹ ^ 2) * ((om 12) ^ 8)))) * hsq0 + (((((7 : ℂ) / 3) * ((om 12) ^ 2)) + (((7 : ℂ) / 3) * ((om 12) ^ 4)) + (((7 : ℂ) / 3) * ((om 12) ^ 6)))) * hg2 + ((((-7 : ℂ) / 3) + (((-7 : ℂ) / 3) * ((om 12) ^ 2)) + (((-7 : ℂ) / 3) * ((om 12) ^ 4)))) * hg3)

end ArchytasZX.Export
