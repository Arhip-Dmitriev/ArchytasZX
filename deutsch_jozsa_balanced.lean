/-
  ArchytasZX proof export over the tensor-network semantics of the ArchytasZX Lean
  library: every diagram is a Net, every rule step an instance of a lemma proved
  there. `#print axioms` lists any step stated without proof.
  equality over symbolic dimensions by the verified normaliser of ArchytasZX.SymNet
  by induction on the multiplicity m, one copy peeled per step
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
macro "pos" : tactic => `(tactic| first | exact Fam.env_dimv_pos _ _ _ (Fpos  _) | (simp [F, Fam.env, bundL, List.lookup, ksOf, List.getD, Δ] <;> first | decide | omega | positivity | (apply Nat.mul_pos <;> omega) | (apply pow_pos; omega)))

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
  | 1 => fun k => (0) + (if k = 0 then 0 else ((-1 : ℝ) * (k : ℝ) * ((d : ℝ) ^ 1)⁻¹))
  | _ => fun _ => 0

/-- Each label's dimension index. -/
def labs : List (ℕ × ℕ) := [(0, 1), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 1), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (65536, 0), (65539, 0), (65540, 0), (65541, 0), (65542, 0)]

/-- The environment. -/
noncomputable def E (d : ℕ) : Env := { dimv := Δ d, phase := Φ d, labs := labs }

/-- Each bundle dimension index with its base index and multiplicity. -/
def bundL : List (ℕ × ℕ × ℕ) := [(1, 0, 0)]

/-- The family, its multiplicities in order: m = m. -/
noncomputable def F (d : ℕ) : Fam := { dims := Δ d, bund := bundL, phase := Φ d, labs := labs }

/-- Every dimension value of the family is positive. -/
theorem Fpos (d : ℕ) (h_d : 0 < d) : ∀ i, 0 < (F d).dims i := fun i => Δpos d h_d i

/-- The family's structure alone. -/
noncomputable def FC : Fam := { dims := fun _ => 1, bund := bundL, phase := fun _ _ => 0, labs := labs }

/-- The family's structure along multiplicity `m`. -/
noncomputable def SpC (m : ℕ) : Spec := ⟨FC, m, ksOf []⟩

/-- (d^(-m/2-1/2)) Σ_{k7=0}^{d-1} Σ_{q0[i0]=0, i0=1..m}^{d-1} Σ_{q5,q11,q15=0}^{d-1} Σ_{j0o0[i0],j1o0[i0],j2o0[i0],j3o0[i0],j4o0[i0]=0, i0=1..m}^{d-1} Σ_{j5o0,j8o0,j9o0,j10o0,j11o0,j12o0,j13o0,j14o0,j15o0=0}^{d-1} ∏_{i0=1}^{m}( d^{-1/2} ω_{d}^{q0[i0] (j0o0[i0])} d^{-1/2} ω_{d}^{j1o0[i0] j0o0[i0]} d^{-1/2} ω_{d}^{j2o0[i0] j1o0[i0]} d^{-1/2} ω_{d}^{j3o0[i0] j2o0[i0]} d^{-1/2} ω_{d}^{j4o0[i0] j3o0[i0]} ) d^{-1/2} ω_{d}^{q5 (j5o0)} d^{-1/2} ω_{d}^{k7 j5o0} d^{-1/2} ω_{d}^{j8o0 k7} d^{-1/2} ω_{d}^{j9o0 j8o0} d^{-1/2} ω_{d}^{j10o0 j9o0} d^{-1/2} e^{2 π i ((-q11/d))} ω_{d}^{q11 (j11o0)} d^{-1/2} ω_{d}^{j12o0 j11o0} d^{-1/2} ω_{d}^{j13o0 j12o0} d^{-1/2} ω_{d}^{j14o0 j13o0} d^{-3/2} ω_{d}^{q15 (j15o0 - j14o0 - k7)} ⊗_{i0=1}^{m}( |j4o0[i0]⟩ ) |j10o0, j15o0⟩ -/
def N0 : Net := ⟨([⟨.Bun 0 .X, [3], [], 0, 0⟩, ⟨.Bun 0 .F, [4], [3], 0, 0⟩, ⟨.Bun 0 .F, [5], [4], 0, 0⟩, ⟨.Bun 0 .F, [6], [5], 0, 0⟩, ⟨.Bun 0 .F, [0], [6], 0, 0⟩, ⟨.X, [7], [], 0, 0⟩, ⟨.F, [8], [7], 0, 0⟩, ⟨.Z, [9, 10], [8], 0, 0⟩, ⟨.F, [11], [9], 0, 0⟩, ⟨.F, [12], [11], 0, 0⟩, ⟨.F, [1], [12], 0, 0⟩, ⟨.X, [13], [], 0, 1⟩, ⟨.F, [14], [13], 0, 0⟩, ⟨.F, [15], [14], 0, 0⟩, ⟨.F, [16], [15], 0, 0⟩, ⟨.X, [2], [16, 10], 0, 0⟩] : List Node), [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0, 1, 2]⟩

/-- (d^(-m/2-1)) Σ_{k2=0}^{d-1} Σ_{q0[i0]=0, i0=1..m}^{d-1} Σ_{q1=0}^{d-1} Σ_{j0o0[i0]=0, i0=1..m}^{d-1} Σ_{j1o0=0}^{d-1} ∏_{i0=1}^{m}( d^{-1/2} ω_{d}^{q0[i0] (j0o0[i0])} ) d^{-1/2} e^{2 π i ((-q1/d))} ω_{d}^{q1 (j1o0)} e^{2 π i ((-k2/d))} ⊗_{i0=1}^{m}( |j0o0[i0]⟩ ) |j1o0, k2⟩ -/
def N1 : Net := ⟨([⟨.Bun 0 .X, [0], [], 0, 0⟩, ⟨.X, [1], [], 0, 1⟩, ⟨.Z, [2], [], 0, 1⟩] : List Node), [], [0, 1, 2]⟩

open ArchytasZX.Sym

set_option linter.unusedSimpArgs false

/-- Each dimension index as a monomial in the dimension symbols. -/
def DT (i : ℕ) : DMon := ([⟨1, [1]⟩, ⟨1, [1]⟩] : List DMon).getD i ⟨1, []⟩

/-- Each phase vector's stored entries. -/
def PT (i : ℕ) : List (ℕ × CMon) := ([[], []] : List (List (ℕ × CMon))).getD i []

/-- Each phase vector's rule as a dimension monomial and the coefficients of its numerator. -/
def RL (i : ℕ) : DMon × List ℤ := ([(⟨1, []⟩, []), (⟨1, [1]⟩, [0, (-1 : ℤ)])] : List (DMon × List ℤ)).getD i (⟨1, []⟩, [])

/-- The dimension indices the networks use. -/
def IX : List ℕ := [0]

/-- Floors of the dimension symbols. -/
def FL : List ℕ := []

/-- The atoms' arguments. -/
noncomputable def GS (d : ℕ) : List ℝ := []

/-- Dimension symbol values and atoms. -/
noncomputable def ρ (d : ℕ) : Val := ⟨fun i => ([d] : List ℕ).getD i 1, fun j => ex ((GS d).getD j 0)⟩

theorem rhoOK (d : ℕ) (h_d : 1 ≤ d) : (ρ d).OK :=
  ⟨fun i => by rcases i with _ | _ <;> simp [ρ] <;> first | omega | exact Nat.pos_of_ne_zero (pow_ne_zero _ (by omega)), fun j => Complex.exp_ne_zero _⟩

theorem flOK (d : ℕ) (h_d : 1 ≤ d) : FloorOK FL (ρ d) := fun i => by
  rcases i with _ | _ <;> simp [flv, FL, ρ] <;> omega

theorem envD (d : ℕ) (h_d : 1 ≤ d) (ks : ℕ → ℕ) : ∀ i ∈ IX, ((F d).env ks).dimv i = (DT i).den (ρ d).dv := by
  intro i hi
  simp only [IX, List.mem_cons, List.not_mem_nil, or_false] at hi
  obtain rfl := hi <;> simp [Fam.env, F, bundL, Fam.dimv, List.lookup, Δ, DT, ρ, DMon.den, npow]

theorem envP (d : ℕ) (h_d : 1 ≤ d) (ks : ℕ → ℕ) : ∀ ph j, ex (((F d).env ks).phase ph j) = phF (ρ d) (PT ph) j * rV (ρ d) (RL ph) j := by
  intro ph j
  rcases ph with _ | _ | _
  · rw [show rV (ρ d) (RL 0) j = 1 by simp [rV, rEv, RL], mul_one]
    simp [Fam.env, F, bundL, Fam.dimv, List.lookup, Φ, PT, phF, ex_zero]
  · simp only [Fam.env, F, bundL, Fam.dimv, List.lookup, Φ]
    have hr := rule_ex (ρ d) ⟨1, [1]⟩ [0, (-1 : ℤ)] (fun k : ℕ => ((-1 : ℝ) * (k : ℝ) * ((d : ℝ) ^ 1)⁻¹)) (fun k => by simp [DMon.den, npow, ρ, rEv]; push_cast; ring) (by decide) j
    try simp only at hr
    rw [ex_add, hr, show RL 1 = (⟨1, [1]⟩, [0, (-1 : ℤ)]) from rfl]
    congr 1
    simp [Fam.env, F, bundL, Fam.dimv, List.lookup, Φ, PT, phF, ex_zero]
  · simp [Fam.env, F, bundL, Fam.dimv, List.lookup, Φ, PT, phF, ex_zero, rV, rEv, RL]

/-- A scalar's terms. -/
def SS0 : List STm := [⟨CMon.mul (CMon.mul (CMon.ofQ 1) (dP ⟨1, [1]⟩ (1))) (CMon.rtD ⟨1, [1]⟩ (-1)), [], [], []⟩]

theorem scalS0 (d : ℕ) (h_d : 1 ≤ d) : ((d : ℂ) * ((rt d))⁻¹) = exDen (ρ d) (fun _ => 0) SS0 := by
  have hρ := (rhoOK d h_d)
  have hg : ∀ j, (ρ d).atom j = ex ((GS d).getD j 0) := fun j => rfl
  simp (disch := decide) only [SS0, exDen, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, den_zero, bodyOf, List.prod_nil, mul_one, CMon.den_mul (ρ d) hρ, ofQ_den, exQ_den, rtD_den, dP_den, phC_den (ρ d) _ hg] <;> simp [DMon.den, npow, ρ, zpow_neg, inv_pow] <;> push_cast <;> ring_nf

/-- A scalar's terms. -/
def SS1 : List STm := [⟨CMon.mul (CMon.ofQ 1) (CMon.ofQ (1 : ℚ)), [], [], []⟩]

theorem scalS1 (d : ℕ) (h_d : 1 ≤ d) : 1 = exDen (ρ d) (fun _ => 0) SS1 := by
  have hρ := (rhoOK d h_d)
  have hg : ∀ j, (ρ d).atom j = ex ((GS d).getD j 0) := fun j => rfl
  simp (disch := decide) only [SS1, exDen, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, den_zero, bodyOf, List.prod_nil, mul_one, CMon.den_mul (ρ d) hρ, ofQ_den, exQ_den, rtD_den, dP_den, phC_den (ρ d) _ hg] <;> simp [DMon.den, npow, ρ, zpow_neg, inv_pow] <;> push_cast <;> ring_nf

/-- A peeled copy. -/
def T0a : Net := ⟨([⟨.X, [65539], [], 0, 0⟩, ⟨.F, [65540], [65539], 0, 0⟩, ⟨.F, [65541], [65540], 0, 0⟩, ⟨.F, [65542], [65541], 0, 0⟩, ⟨.F, [65536], [65542], 0, 0⟩] : List Node), [65539, 65540, 65541, 65542], [65536]⟩

/-- A peeled copy. -/
def T0b : Net := ⟨([⟨.X, [65536], [], 0, 0⟩] : List Node), [], [65536]⟩

set_option maxHeartbeats 0 in
/-- The kernel's normalisation. -/
theorem C2 : symCheck FL DT PT RL labs IX T0a T0b SS1 SS1 = true := by decide +kernel

theorem P2 (d : ℕ) (h_d : 1 ≤ d) : ∀ m : ℕ, Equal ((F d).env (ksOf [m])) 1 T0a 1 T0b :=
  fun m => equal_sym ((F d).env (ksOf [m])) (ρ d) (rhoOK d h_d) FL (flOK d h_d) DT PT RL labs IX (envD d h_d (ksOf [m])) (envP d h_d (ksOf [m])) rfl T0a T0b SS1 SS1 _ _ (scalS1 d h_d) (scalS1 d h_d) C2

set_option maxHeartbeats 0 in
/-- The kernel's normalisation. -/
theorem C1 : symCheck FL DT PT RL labs IX (Net.zero (SpC 0) N0) (Net.zero (SpC 0) N1) SS0 SS1 = true := by decide +kernel

theorem P1 (d : ℕ) (h_d : 1 ≤ d) : ∀ m : ℕ, Equal ((F d).env (ksOf [m])) ((d : ℂ) * ((rt d))⁻¹) (Net.zero (SpC 0) N0) 1 (Net.zero (SpC 0) N1) :=
  fun m => equal_sym ((F d).env (ksOf [m])) (ρ d) (rhoOK d h_d) FL (flOK d h_d) DT PT RL labs IX (envD d h_d (ksOf [m])) (envP d h_d (ksOf [m])) rfl (Net.zero (SpC 0) N0) (Net.zero (SpC 0) N1) SS0 SS1 _ _ (scalS0 d h_d) (scalS1 d h_d) C1

theorem Q0 (d : ℕ) (h_d : 1 ≤ d) (k_ : ℕ) (ih : Equal ((F d).env (ksOf [k_])) ((d : ℂ) * ((rt d))⁻¹) N0 1 N1) :
    Equal ((F d).env (ksOf [k_])) ((d : ℂ) * ((rt d))⁻¹) (Net.split (SpC 0) 65536 N0) 1 (Net.split (SpC 0) 65536 N1) :=
  Equal.congr_s (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans (Equal.mul_trans (isoB ((F d).env (ksOf [k_])) 1 (Net.split (SpC 0) 65536 N0) ⟨T0a.nodes ++ N0.nodes, T0a.int ++ N0.int, [0, 65536, 1, 2]⟩ [] [1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] (by chk))
    (plug ((F d).env (ksOf [k_])) T0a.nodes N0.nodes N1.nodes T0a.int N0.int N1.int [0, 65536, 1, 2] N0.bnd ((d : ℂ) * ((rt d))⁻¹) 1 (Equal.toLocal _ _ _ ih) (by chk) (by chk) (by chk)))
    (isoB ((F d).env (ksOf [k_])) 1 ⟨T0a.nodes ++ N1.nodes, T0a.int ++ N1.int, [0, 65536, 1, 2]⟩ ⟨N1.nodes ++ T0a.nodes, N1.int ++ T0a.int, [0, 65536, 1, 2]⟩ [] [5, 6, 7, 0, 1, 2, 3, 4] (by chk)))
    (plug ((F d).env (ksOf [k_])) N1.nodes T0a.nodes T0b.nodes N1.int T0a.int T0b.int [0, 65536, 1, 2] T0a.bnd 1 1 (Equal.toLocal _ _ _ (P2 d h_d k_)) (by chk) (by chk) (by chk)))
    (isoB ((F d).env (ksOf [k_])) 1 ⟨N1.nodes ++ T0b.nodes, N1.int ++ T0b.int, [0, 65536, 1, 2]⟩ (Net.split (SpC 0) 65536 N1) [] [0, 3, 1, 2] (by chk))) (by ring) (by ring)

theorem P0 (d : ℕ) (h_d : 1 ≤ d) : ∀ m : ℕ, Equal ((F d).env (ksOf [m])) ((d : ℂ) * ((rt d))⁻¹) N0 1 N1 := by
  intro m
  induction m with
  | zero => exact Equal.of_zero_at (F d) 0 65536 [0] (by simp) N0 N1 (Spec.splitOK_struct _ 0 _ _ 65536 N0 (by decide +kernel : Spec.splitOK (SpC 0) 65536 N0 = true)) (Spec.splitOK_struct _ 0 _ _ 65536 N1 (by decide +kernel : Spec.splitOK (SpC 0) 65536 N1 = true)) (by decide +kernel) (Equal.congr_s (P1 d h_d 0) (by ring) (by ring))
  | succ k_ ih => exact Equal.of_split_at (F d) 0 65536 k_ [k_] (by simp) N0 N1 (Spec.splitOK_struct _ 0 _ _ 65536 N0 (by decide +kernel : Spec.splitOK (SpC 0) 65536 N0 = true)) (Spec.splitOK_struct _ 0 _ _ 65536 N1 (by decide +kernel : Spec.splitOK (SpC 0) 65536 N1 = true)) (fun _ _ _ => Fam.dimv_pos _ _ _ (Fpos d (by omega : 0 < d) _)) (Q0 d h_d k_ ih)

/-- The families are equal for every multiplicity and every value of their dimensions and phases. -/
theorem main (d : ℕ) (h_d : 1 ≤ d) :
    ∀ m : ℕ, Equal ((F d).env (ksOf [m])) ((((rt d))⁻¹ ^ m) * ((rt d))⁻¹) N0 ((((rt d))⁻¹ ^ m) * ((d : ℂ))⁻¹) N1 :=
  fun m => Equal.congr_s (Equal.smul ((((rt d))⁻¹ ^ m) * ((d : ℂ))⁻¹) ((P0 d h_d m))) (by have r0 := (mul_inv_cancel₀ (Nat.cast_ne_zero.2 (Nat.pos_iff_ne_zero.1 (by first | omega | norm_num | positivity | (apply Nat.mul_pos <;> omega) | exact Nat.pos_pow_of_pos _ (by omega))) : (d : ℂ) ≠ 0)); linear_combination (((((rt d))⁻¹ ^ m) * ((rt d))⁻¹)) * r0) (by ring)

end ArchytasZX.Export
