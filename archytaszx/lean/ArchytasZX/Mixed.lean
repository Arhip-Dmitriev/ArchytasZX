/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Closed

/-!
# Mixed networks and their completely positive maps

An `MNet` is a network with discards, held as its purification: a pure `Net` and the
boundary labels its discards end, `env`. `MNet.cpm` is the doubled network's entry: the sum
over a value on every environment label, each over its dimension, of the network's entry at the
plain boundary values `σ` times the conjugate of its entry at the conjugate boundary values
`σ'`, both reading the same environment values.

The doubling facts: `cpm_pure` (a network without discards doubles to itself beside its
conjugate), `Equal.cpm` (a proved pure equation lifts to every mixed context, its scalar by its
squared modulus), `cpm_phase` (a unit-modulus scalar is invisible), `cpm_conj` (the conjugate
network doubles to the swapped copies), `cpm_causal` (an isometry with every output discarded
is the discard on its inputs). The discard rules of `archytaszx.rewrite.rules_discard` hold as
the tensor identities `discard_fourier`, `discard_phase`, `discard_zero`, `discard_loop` and
`discard_binder`.
-/

namespace ArchytasZX

open Complex Finset

/-- A network with discards: its purification and the labels its discards end. -/
structure MNet where
  net : Net
  env : List ℕ
  deriving DecidableEq, Repr

/-- The sum over a value on each label of `ls`, each over its dimension, written into both
assignments. -/
noncomputable def envSum (D : ℕ → ℕ) :
    List ℕ → ((ℕ → ℕ) → (ℕ → ℕ) → ℂ) → (ℕ → ℕ) → (ℕ → ℕ) → ℂ
  | [], f, σ, σ' => f σ σ'
  | l :: ls, f, σ, σ' =>
      ∑ j ∈ range (D l), envSum D ls f (Function.update σ l j) (Function.update σ' l j)

/-- The CPM entry of a mixed network at plain values `σ` and conjugate values `σ'`. -/
noncomputable def MNet.cpm (E : Env) (M : MNet) (σ σ' : ℕ → ℕ) : ℂ :=
  envSum (labDim E) M.env (fun a b => eval E M.net a * (starRingEnd ℂ) (eval E M.net b)) σ σ'

theorem envSum_congr (D : ℕ → ℕ) (P : (ℕ → ℕ) → Prop) (f g : (ℕ → ℕ) → (ℕ → ℕ) → ℂ)
    (hP : ∀ σ l j, P σ → j < D l → P (Function.update σ l j))
    (hfg : ∀ σ σ', P σ → P σ' → f σ σ' = g σ σ') :
    ∀ (ls : List ℕ) σ σ', P σ → P σ' → envSum D ls f σ σ' = envSum D ls g σ σ'
  | [], σ, σ', hσ, hσ' => hfg σ σ' hσ hσ'
  | l :: ls, σ, σ', hσ, hσ' => by
      refine Finset.sum_congr rfl fun j hj => ?_
      have hj' := Finset.mem_range.1 hj
      exact envSum_congr D P f g hP hfg ls _ _ (hP σ l j hσ hj') (hP σ' l j hσ' hj')

theorem envSum_mul (D : ℕ → ℕ) (c : ℂ) (f : (ℕ → ℕ) → (ℕ → ℕ) → ℂ) :
    ∀ (ls : List ℕ) σ σ', c * envSum D ls f σ σ' = envSum D ls (fun a b => c * f a b) σ σ'
  | [], _, _ => rfl
  | l :: ls, σ, σ' => by
      simp only [envSum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => envSum_mul D c f ls _ _

theorem envSum_swap (D : ℕ → ℕ) (f : (ℕ → ℕ) → (ℕ → ℕ) → ℂ) :
    ∀ (ls : List ℕ) σ σ', envSum D ls (fun a b => f b a) σ σ' = envSum D ls f σ' σ
  | [], _, _ => rfl
  | l :: ls, σ, σ' => by
      simp only [envSum]
      exact Finset.sum_congr rfl fun j _ => envSum_swap D f ls _ _

/-- (F3) Without discards, the doubled network is the network beside its conjugate. -/
theorem cpm_pure (E : Env) (N : Net) (σ σ' : ℕ → ℕ) :
    (MNet.mk N []).cpm E σ σ' = eval E N σ * (starRingEnd ℂ) (eval E N σ') := rfl

theorem InRange.update {E : Env} {ls : List ℕ} {σ : ℕ → ℕ} (h : InRange E ls σ) (l j : ℕ)
    (hj : j < labDim E l) : InRange E ls (Function.update σ l j) := by
  intro m hm
  by_cases hml : m = l
  · subst hml; simpa using hj
  · simpa [Function.update_of_ne hml] using h m hm

/-- (F4) A pure equation `s₁ • N₁ = s₂ • N₂` lifts to the mixed networks with the same
discarded labels: `|s₁|² • cpm₁ = |s₂|² • cpm₂`. -/
theorem Equal.cpm {E : Env} {s₁ s₂ : ℂ} {N₁ N₂ : Net} (h : Equal E s₁ N₁ s₂ N₂)
    (env : List ℕ) (σ σ' : ℕ → ℕ) (hσ : InRange E N₁.bnd σ) (hσ' : InRange E N₁.bnd σ') :
    s₁ * (starRingEnd ℂ) s₁ * (MNet.mk N₁ env).cpm E σ σ' =
      s₂ * (starRingEnd ℂ) s₂ * (MNet.mk N₂ env).cpm E σ σ' := by
  simp only [MNet.cpm, envSum_mul]
  refine envSum_congr (labDim E) (InRange E N₁.bnd) _ _ (fun σ l j hσ hj => hσ.update l j hj)
    (fun a b ha hb => ?_) env σ σ' hσ hσ'
  have ea := h.2 a ha
  have eb := congrArg (starRingEnd ℂ) (h.2 b hb)
  simp only [map_mul] at eb
  calc s₁ * (starRingEnd ℂ) s₁ * (eval E N₁ a * (starRingEnd ℂ) (eval E N₁ b))
      = (s₁ * eval E N₁ a) * ((starRingEnd ℂ) s₁ * (starRingEnd ℂ) (eval E N₁ b)) := by ring
    _ = (s₂ * eval E N₂ a) * ((starRingEnd ℂ) s₂ * (starRingEnd ℂ) (eval E N₂ b)) := by
        rw [ea, eb]
    _ = s₂ * (starRingEnd ℂ) s₂ * (eval E N₂ a * (starRingEnd ℂ) (eval E N₂ b)) := by ring

/-- (F5) Networks equal up to a unit-modulus scalar have the same CPM. -/
theorem cpm_phase {E : Env} {s : ℂ} {N₁ N₂ : Net} (h : Equal E 1 N₁ s N₂)
    (hs : s * (starRingEnd ℂ) s = 1) (env : List ℕ) (σ σ' : ℕ → ℕ)
    (hσ : InRange E N₁.bnd σ) (hσ' : InRange E N₁.bnd σ') :
    (MNet.mk N₁ env).cpm E σ σ' = (MNet.mk N₂ env).cpm E σ σ' := by
  have := h.cpm env σ σ' hσ hσ'
  simpa [hs] using this

/-- (F1) The conjugate network doubles to the two copies swapped. -/
theorem cpm_conj {E : Env} {N₁ N₂ : Net} (h : ∀ a, eval E N₂ a = (starRingEnd ℂ) (eval E N₁ a))
    (env : List ℕ) (σ σ' : ℕ → ℕ) :
    (MNet.mk N₂ env).cpm E σ σ' = (MNet.mk N₁ env).cpm E σ' σ := by
  simp only [MNet.cpm]
  rw [← envSum_swap]
  congr 1
  funext a b
  rw [h a, h b, Complex.conj_conj, mul_comm]

/-- (F6) A network whose discarded labels make it an isometry onto its remaining labels is the
discard on them: its CPM is `1` where the plain and conjugate values agree, else `0`. -/
theorem cpm_causal {E : Env} {N : Net} {env keep : List ℕ}
    (hiso : ∀ σ σ', (MNet.mk N env).cpm E σ σ' =
      if ∀ l ∈ keep, σ l = σ' l then 1 else 0)
    (σ σ' : ℕ → ℕ) (hk : ∀ l ∈ keep, σ l = σ' l) : (MNet.mk N env).cpm E σ σ' = 1 := by
  rw [hiso, if_pos hk]

/-- The discard after `F`: `F` is unitary. -/
theorem discard_fourier (d : ℕ) (hd : 0 < d) (i i' : ℕ) (hi : i < d) (hi' : i' < d) :
    ∑ o ∈ range d, fou d o i * (starRingEnd ℂ) (fou d o i') = if i = i' then 1 else 0 :=
  fou_unitary d hd i i' hi hi'

/-- `e^{2πiθ}` has modulus one. -/
theorem ex_mul_conj (θ : ℝ) : ex θ * (starRingEnd ℂ) (ex θ) = 1 := by
  have h : ‖ex θ‖ = 1 := by
    rw [ex, Complex.norm_exp]
    simp
  rw [Complex.mul_conj, Complex.normSq_eq_norm_sq, h]
  simp

theorem mixed_tZ_leg (d : ℕ) (φ : ℕ → ℝ) (o i : ℕ) (hi : i < d) :
    tZ d φ [o, i] = if o = i then ex (φ i) else 0 := by
  simp only [tZ, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  by_cases ho : o = i
  · subst ho
    rw [Finset.sum_eq_single o (fun k _ hk => by simp [Ne.symm hk])
      (fun h => absurd (Finset.mem_range.2 hi) h)]
    simp
  · rw [if_neg ho]
    refine Finset.sum_eq_zero fun k _ => ?_
    split_ifs with h
    · exact absurd (h.1.trans h.2.symm) ho
    · rfl

/-- The discard after the phase gate `Z (1 → 1)` with unit phases. -/
theorem discard_phase (d : ℕ) (φ : ℕ → ℝ) (i i' : ℕ) (hi : i < d) (hi' : i' < d) :
    ∑ o ∈ range d, tZ d φ [o, i] * (starRingEnd ℂ) (tZ d φ [o, i']) =
      if i = i' then 1 else 0 := by
  simp only [mixed_tZ_leg d φ _ i hi, mixed_tZ_leg d φ _ i' hi']
  rw [Finset.sum_eq_single i (fun o _ ho => by simp [ho])
    (fun h => absurd (Finset.mem_range.2 hi) h)]
  by_cases hii : i = i'
  · subst hii
    simp only [if_true]
    exact ex_mul_conj (φ i)
  · simp [hii]

/-- The discarded state `|0⟩`, the pink dot: its trace is `1`. -/
theorem discard_zero (d : ℕ) (hd : 0 < d) :
    ∑ o ∈ range d, (if o = 0 then (1 : ℂ) else 0) * (starRingEnd ℂ) (if o = 0 then 1 else 0) =
      1 := by
  rw [Finset.sum_eq_single 0 (fun o _ ho => by simp [ho])
    (fun h => absurd (Finset.mem_range.2 hd) h)]
  simp

/-- Two discards joined: the trace of the identity, `d`. -/
theorem discard_loop (d : ℕ) :
    ∑ a ∈ range d, ∑ b ∈ range d, tI d a b * (starRingEnd ℂ) (tI d a b) = d := by
  have : ∀ a ∈ range d, ∑ b ∈ range d, tI d a b * (starRingEnd ℂ) (tI d a b) = 1 := by
    intro a ha
    rw [Finset.sum_eq_single a (fun b _ hb => by simp [tI, Ne.symm hb])
      (fun h => absurd ha h)]
    simp [tI, Finset.mem_range.1 ha]
  rw [Finset.sum_congr rfl this]
  simp

/-- The discard after the binder `B (s, t)`: the binder is unitary. -/
theorem discard_binder (s t a b a' b' : ℕ) (ha : a < s) (hb : b < t) (ha' : a' < s)
    (hb' : b' < t) :
    ∑ k ∈ range (s * t), tB s t k a b * (starRingEnd ℂ) (tB s t k a' b') =
      if a = a' ∧ b = b' then 1 else 0 := by
  have hk : a * t + b < s * t := by nlinarith
  rw [Finset.sum_eq_single (a * t + b) (fun k _ hk' => by simp [tB, hk'])
    (fun h => absurd (Finset.mem_range.2 hk) h)]
  by_cases h : a = a' ∧ b = b'
  · obtain ⟨rfl, rfl⟩ := h
    simp [tB, ha, hb]
  · rw [if_neg h]
    have hne : a * t + b ≠ a' * t + b' := by
      intro he
      apply h
      have ht : 0 < t := by omega
      have h1 : (a * t + b) / t = a := by
        rw [Nat.mul_comm a t, Nat.mul_add_div ht, Nat.div_eq_of_lt hb, Nat.add_zero]
      have h2 : (a' * t + b') / t = a' := by
        rw [Nat.mul_comm a' t, Nat.mul_add_div ht, Nat.div_eq_of_lt hb', Nat.add_zero]
      have ha'' : a = a' := by rw [← h1, ← h2, he]
      subst ha''
      exact ⟨rfl, Nat.add_left_cancel he⟩
    simp [tB, ha, hb, ha', hb', hne]

end ArchytasZX
