/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumRules

/-!
# Qubit networks as path sums

A `Frag` is one generator's path sum: path variables, phase terms, constraints and a constant.
`fval_join` multiplies two fragments with disjoint path variables, `sumOver_bsum` turns the
network's sum over labels of dimension 2 into a sum over Boolean variables, and `eval_frags`
gives a network's value as `val` of its joined fragments with the summed labels as path
variables. `z_frag`, `x_frag` and `f_frag` are the fragments of the Z spider (constraints
`l0 + l = 0`, phase table interpolated), the X spider (a fresh variable `y`, phase
`y l / 2` per leg) and F at dimension 2.
-/

namespace ArchytasZX
namespace PSem

open Complex Finset

/-- A generator's path sum. -/
structure Frag where
  path : List ℕ
  phase : Poly
  cons : List F2
  c : ℂ

/-- A fragment's value. -/
noncomputable def fval (M : ℕ) (f : Frag) (τ : ℕ → ZMod 2) : ℂ :=
  val M f.c ⟨f.path, f.phase, f.cons, []⟩ τ

/-- Two fragments side by side. -/
def Frag.join (a b : Frag) : Frag :=
  ⟨a.path ++ b.path, a.phase ++ b.phase, a.cons ++ b.cons, a.c * b.c⟩

/-- No path variable of `a` in `b`. -/
def Disj (a b : Frag) : Prop :=
  ∀ y ∈ a.path, y ∉ b.path ∧ y ∉ pvars b.phase ∧ y ∉ fvars b.cons

instance (a b : Frag) : Decidable (Disj a b) := by
  unfold Disj
  infer_instance

/-- The empty fragment. -/
def Frag.unit : Frag := ⟨[], [], [], 1⟩

/-- Fragments side by side. -/
def joinAll : List Frag → Frag
  | [] => Frag.unit
  | f :: fs => f.join (joinAll fs)

theorem bsum_inv (g : (ℕ → ZMod 2) → ℂ) (y : ℕ) (b : ZMod 2)
    (hg : ∀ τ, g (Function.update τ y b) = g τ) :
    ∀ (zs : List ℕ) (τ : ℕ → ZMod 2), y ∉ zs →
      bsum zs g (Function.update τ y b) = bsum zs g τ := by
  intro zs τ hy
  rw [bsum_out g y b zs τ hy]
  exact bsum_congr _ _ hg zs τ

theorem summand_join (M : ℕ) (a b : Frag) (τ : ℕ → ZMod 2) :
    summand M (a.join b).c ⟨(a.join b).path, (a.join b).phase, (a.join b).cons, []⟩ τ =
      summand M a.c ⟨a.path, a.phase, a.cons, []⟩ τ *
        summand M b.c ⟨b.path, b.phase, b.cons, []⟩ τ := by
  unfold summand Frag.join
  dsimp only
  simp only [List.append_nil, List.forall_mem_append, pn_append]
  by_cases ha : ∀ q ∈ a.cons, fz τ q = 0
  · by_cases hb : ∀ q ∈ b.cons, fz τ q = 0
    · rw [if_pos ⟨ha, hb⟩, if_pos ha, if_pos hb]
      push_cast
      rw [add_div, ex_add]
      ring
    · rw [if_neg (fun h => hb h.2), if_neg hb, mul_zero]
  · rw [if_neg (fun h => ha h.1), if_neg ha, zero_mul]

theorem summand_frag_inv (M : ℕ) (b : Frag) (y : ℕ) (w : ZMod 2) (hp : y ∉ pvars b.phase)
    (hc : y ∉ fvars b.cons) (τ : ℕ → ZMod 2) :
    summand M b.c ⟨b.path, b.phase, b.cons, []⟩ (Function.update τ y w) =
      summand M b.c ⟨b.path, b.phase, b.cons, []⟩ τ :=
  summand_update_not M b.c _ τ y w hp (by simpa using hc)

theorem fval_join (M : ℕ) (a b : Frag) (hab : Disj a b) (hba : Disj b a) (τ : ℕ → ZMod 2) :
    fval M (a.join b) τ = fval M a τ * fval M b τ := by
  unfold fval val
  show bsum (a.path ++ b.path) (summand M (a.join b).c
    ⟨(a.join b).path, (a.join b).phase, (a.join b).cons, []⟩) τ = _
  rw [bsum_congr _ _ (summand_join M a b), bsum_append]
  have hb : ∀ τ', ∀ y ∈ a.path, ∀ w,
      bsum b.path (summand M b.c ⟨b.path, b.phase, b.cons, []⟩) (Function.update τ' y w) =
        bsum b.path (summand M b.c ⟨b.path, b.phase, b.cons, []⟩) τ' := by
    intro τ' y hy w
    obtain ⟨h1, h2, h3⟩ := hab y hy
    exact bsum_inv _ y w (summand_frag_inv M b y w h2 h3) b.path τ' h1
  have ha : ∀ τ', ∀ y ∈ b.path, ∀ w,
      summand M a.c ⟨a.path, a.phase, a.cons, []⟩ (Function.update τ' y w) =
        summand M a.c ⟨a.path, a.phase, a.cons, []⟩ τ' := by
    intro τ' y hy w
    obtain ⟨_, h2, h3⟩ := hba y hy
    exact summand_frag_inv M a y w h2 h3 τ'
  have inner : ∀ τ', bsum b.path (fun ρ => summand M a.c ⟨a.path, a.phase, a.cons, []⟩ ρ *
      summand M b.c ⟨b.path, b.phase, b.cons, []⟩ ρ) τ' =
      summand M a.c ⟨a.path, a.phase, a.cons, []⟩ τ' *
        bsum b.path (summand M b.c ⟨b.path, b.phase, b.cons, []⟩) τ' :=
    fun τ' => bsum_mul_left _ _ b.path ha τ'
  rw [bsum_congr (bsum b.path fun ρ => summand M a.c ⟨a.path, a.phase, a.cons, []⟩ ρ *
      summand M b.c ⟨b.path, b.phase, b.cons, []⟩ ρ) _ inner]
  have swap : ∀ τ', summand M a.c ⟨a.path, a.phase, a.cons, []⟩ τ' *
      bsum b.path (summand M b.c ⟨b.path, b.phase, b.cons, []⟩) τ' =
      bsum b.path (summand M b.c ⟨b.path, b.phase, b.cons, []⟩) τ' *
        summand M a.c ⟨a.path, a.phase, a.cons, []⟩ τ' := fun τ' => mul_comm _ _
  rw [bsum_congr _ _ swap, bsum_mul_left _ _ a.path hb τ]
  exact mul_comm _ _

theorem pvars_append (P R : Poly) : pvars (P ++ R) = pvars P ++ pvars R := by
  simp [pvars, List.flatMap_append]

theorem fvars_append (C D : List F2) : fvars (C ++ D) = fvars C ++ fvars D := by
  simp [fvars, List.flatMap_append]

theorem disj_join_right (a b c : Frag) (hb : Disj a b) (hc : Disj a c) : Disj a (b.join c) := by
  intro y hy
  obtain ⟨b1, b2, b3⟩ := hb y hy
  obtain ⟨c1, c2, c3⟩ := hc y hy
  refine ⟨?_, ?_, ?_⟩
  · simp only [Frag.join, List.mem_append, not_or]; exact ⟨b1, c1⟩
  · simp only [Frag.join, pvars_append, List.mem_append, not_or]; exact ⟨b2, c2⟩
  · simp only [Frag.join, fvars_append, List.mem_append, not_or]; exact ⟨b3, c3⟩

theorem disj_join_left (a b c : Frag) (hb : Disj b a) (hc : Disj c a) : Disj (b.join c) a := by
  intro y hy
  simp only [Frag.join, List.mem_append] at hy
  rcases hy with hy | hy
  · exact hb y hy
  · exact hc y hy

theorem disj_unit_left (a : Frag) : Disj Frag.unit a := by
  intro y hy; simp [Frag.unit] at hy

theorem disj_unit_right (a : Frag) : Disj a Frag.unit := by
  intro y _; simp [Frag.unit, pvars, fvars]

theorem disj_joinAll_right (f : Frag) :
    ∀ fs : List Frag, (∀ g ∈ fs, Disj f g) → Disj f (joinAll fs)
  | [], _ => disj_unit_right f
  | g :: gs, h => disj_join_right f g _ (h g List.mem_cons_self)
      (disj_joinAll_right f gs fun g' hg' => h g' (List.mem_cons_of_mem _ hg'))

theorem disj_joinAll_left (f : Frag) :
    ∀ fs : List Frag, (∀ g ∈ fs, Disj g f) → Disj (joinAll fs) f
  | [], _ => disj_unit_left f
  | g :: gs, h => disj_join_left f g _ (h g List.mem_cons_self)
      (disj_joinAll_left f gs fun g' hg' => h g' (List.mem_cons_of_mem _ hg'))

theorem fval_unit (M : ℕ) (τ : ℕ → ZMod 2) : fval M Frag.unit τ = 1 := by
  simp [fval, val, Frag.unit, bsum, summand, pn, ex_zero]

theorem fval_joinAll (M : ℕ) (τ : ℕ → ZMod 2) :
    ∀ fs : List Frag, fs.Pairwise (fun a b => Disj a b ∧ Disj b a) →
      fval M (joinAll fs) τ = (fs.map fun f => fval M f τ).prod
  | [], _ => fval_unit M τ
  | f :: fs, h => by
    rw [List.pairwise_cons] at h
    have hd : Disj f (joinAll fs) ∧ Disj (joinAll fs) f :=
      ⟨disj_joinAll_right f fs fun g hg => (h.1 g hg).1,
        disj_joinAll_left f fs fun g hg => (h.1 g hg).2⟩
    rw [joinAll, fval_join M f _ hd.1 hd.2, fval_joinAll M τ fs h.2, List.map_cons,
      List.prod_cons]

/-! ## Labels of dimension 2 -/

/-- The Boolean image of a label assignment. -/
def bits (σ : ℕ → ℕ) : ℕ → ZMod 2 := fun l => (σ l : ZMod 2)

theorem bits_update (σ : ℕ → ℕ) (l j : ℕ) :
    bits (Function.update σ l j) = Function.update (bits σ) l (j : ZMod 2) := by
  funext v
  by_cases h : v = l
  · subst h; simp [bits]
  · simp [bits, Function.update_of_ne h]

theorem sumOver_bsum (D : ℕ → ℕ) (f : (ℕ → ℕ) → ℂ) (g : (ℕ → ZMod 2) → ℂ)
    (hfg : ∀ σ, (∀ l, σ l < 2) → f σ = g (bits σ)) :
    ∀ (ls : List ℕ), (∀ l ∈ ls, D l = 2) →
      ∀ σ, (∀ l, σ l < 2) → sumOver D ls f σ = bsum ls g (bits σ)
  | [], _, σ, hσ => hfg σ hσ
  | l :: ls, hD, σ, hσ => by
    have hD' : ∀ l' ∈ ls, D l' = 2 := fun l' h => hD l' (List.mem_cons_of_mem _ h)
    have hup : ∀ j < 2, ∀ v, Function.update σ l j v < 2 := by
      intro j hj v
      by_cases h : v = l
      · subst h; simpa using hj
      · rw [Function.update_of_ne h]; exact hσ v
    simp only [sumOver, bsum, hD l List.mem_cons_self, Finset.sum_range_succ,
      Finset.sum_range_zero, zero_add]
    rw [sumOver_bsum D f g hfg ls hD' _ (hup 0 (by norm_num)),
      sumOver_bsum D f g hfg ls hD' _ (hup 1 (by norm_num)), bits_update, bits_update]
    norm_num

/-- A network's value as the joined fragments of its nodes, its summed labels as path
variables. -/
theorem eval_frags (E : Env) (N : Net) (M : ℕ) (ps : List (Node × Frag))
    (hN : N.nodes = ps.map Prod.fst)
    (hnode : ∀ p ∈ ps, ∀ σ, (∀ l, σ l < 2) → ArchytasZX.val E σ p.1 = fval M p.2 (bits σ))
    (hdisj : (ps.map Prod.snd).Pairwise fun a b => Disj a b ∧ Disj b a)
    (hint : ∀ l ∈ N.int, labDim E l = 2) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) :
    eval E N σ = val M (joinAll (ps.map Prod.snd)).c
      ⟨N.int ++ (joinAll (ps.map Prod.snd)).path, (joinAll (ps.map Prod.snd)).phase,
        (joinAll (ps.map Prod.snd)).cons, []⟩ (bits σ) := by
  unfold eval
  refine (sumOver_bsum (labDim E) (prodVal E N.nodes) (fval M (joinAll (ps.map Prod.snd))) ?_
    N.int hint σ hσ).trans ?_
  · intro σ' hσ'
    rw [fval_joinAll M _ _ hdisj, prodVal, hN, List.map_map, List.map_map]
    congr 1
    apply List.map_congr_left
    intro p hp
    exact hnode p hp σ' hσ'
  · unfold val
    rw [bsum_append]
    rfl

/-! ## Generators at dimension 2 -/

theorem bit_cases (n : ℕ) (h : n < 2) : n = 0 ∨ n = 1 := by omega

theorem zq_bit (n : ℕ) (h : n < 2) : zq (n : ZMod 2) = n := by
  rcases bit_cases n h with rfl | rfl <;> decide

theorem bits_eq (a b : ℕ) (ha : a < 2) (hb : b < 2) :
    ((a : ZMod 2) + (b : ZMod 2) = 0) ↔ b = a := by
  rcases bit_cases a ha with rfl | rfl <;> rcases bit_cases b hb with rfl | rfl <;> decide

theorem tZ_bits (φ : ℕ → ℝ) (v : ℕ) (vs : List ℕ) (hv : v < 2) :
    tZ 2 φ (v :: vs) = if ∀ w ∈ vs, w = v then ex (φ v) else 0 := by
  unfold tZ
  rw [Finset.sum_range_succ, Finset.sum_range_one]
  rcases bit_cases v hv with rfl | rfl <;> simp

/-- The Z spider's fragment on the legs `l0 :: ls`: `l0 + l = 0` per other leg and the phase
`t0 + (t1 - t0) l0`, in units of `1 / M`. -/
def zFrag (M t0 t1 l0 : ℕ) (ls : List ℕ) : Frag :=
  ⟨[], [([], t0), ([l0], t1 + (M - t0))], ls.map fun l => [[l0], [l]], 1⟩

theorem z_frag (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (n : Node) (hk : n.kind = .Z)
    (hd : E.dimv n.dim = 2) (M t0 t1 : ℕ) (hM : 0 < M) (ht : t0 ≤ M)
    (h0 : E.phase n.ph 0 = (t0 : ℝ) / M) (h1 : E.phase n.ph 1 = (t1 : ℝ) / M) (l0 : ℕ)
    (ls : List ℕ) (hl : n.outs ++ n.ins = l0 :: ls) :
    ArchytasZX.val E σ n = fval M (zFrag M t0 t1 l0 ls) (bits σ) := by
  have hv : ArchytasZX.val E σ n = tZ 2 (E.phase n.ph) ((n.outs ++ n.ins).map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd, List.map_append]
  rw [hv, hl, List.map_cons, tZ_bits _ _ _ (hσ l0)]
  unfold fval val zFrag
  simp only [bsum, summand, List.append_nil, List.forall_mem_map]
  have hc : ∀ l, fz (bits σ) [[l0], [l]] = 0 ↔ σ l = σ l0 := by
    intro l
    have e : fz (bits σ) [[l0], [l]] = (σ l0 : ZMod 2) + (σ l : ZMod 2) := by
      simp [fz, mz, bits]
    rw [e, bits_eq _ _ (hσ l0) (hσ l)]
  simp only [hc]
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  have hp : pn (bits σ) [([], t0), ([l0], t1 + (M - t0))] =
      (t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * (σ l0 : ℤ) := by
    simp [pn, mz, bits, zq_bit _ (hσ l0), zq1]
  rw [hp]
  split_ifs
  · rw [one_mul]
    rcases bit_cases (σ l0) (hσ l0) with e | e
    · rw [e, h0]; push_cast; simp
    · rw [e, h1]
      have : (((t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * ((1 : ℕ) : ℤ) : ℤ) : ℝ) / M =
          (t1 : ℝ) / M + ((1 : ℤ) : ℝ) := by
        rw [Nat.cast_add, Nat.cast_sub ht]
        push_cast
        field_simp
        ring
      rw [this, ex_int]
  · rfl

/-- F's fragment on the legs `o`, `i`: the phase `o i / 2`. -/
noncomputable def fFrag (M o i : ℕ) : Frag := ⟨[], [([o, i], M / 2)], [], (rt 2)⁻¹⟩

theorem fou2 (j k : ℕ) (hj : j < 2) (hk : k < 2) :
    fou 2 j k = (if j * k = 0 then 1 else -1) / rt 2 := by
  have h2 : om 2 = -1 := by
    have := om_eq_ex 2 1
    rw [pow_one] at this
    rw [this, show ((1 : ℕ) : ℝ) / ((2 : ℕ) : ℝ) = 1 / 2 by norm_num, ArchytasZX.Sym.ex_half]
  unfold fou
  rcases bit_cases j hj with rfl | rfl <;> rcases bit_cases k hk with rfl | rfl <;>
    simp [h2, rt]

theorem f_frag (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (n : Node) (hk : n.kind = .F)
    (hd : E.dimv n.dim = 2) (M : ℕ) (hM : 0 < M) (hM2 : M % 2 = 0) (o i : ℕ)
    (ho : n.outs = [o]) (hi : n.ins = [i]) :
    ArchytasZX.val E σ n = fval M (fFrag M o i) (bits σ) := by
  have hv : ArchytasZX.val E σ n = fou 2 (σ o) (σ i) := by
    unfold ArchytasZX.val
    rw [hk, ho, hi]
    simp only [hd, List.map_cons, List.map_nil]
  rw [hv, fou2 _ _ (hσ o) (hσ i)]
  unfold fval val fFrag
  simp only [bsum, summand, List.append_nil, List.nil_append, List.not_mem_nil,
    IsEmpty.forall_iff, forall_const, if_true]
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  obtain ⟨r, hr⟩ : ∃ r, M = 2 * r := ⟨M / 2, by omega⟩
  have hr2 : M / 2 = r := by omega
  rw [hr2]
  have hr0 : (r : ℝ) ≠ 0 := by
    intro e
    have : r = 0 := by exact_mod_cast e
    omega
  have hx : ex ((r : ℝ) / M) = -1 := by
    rw [hr, show ((r : ℝ) / ((2 * r : ℕ) : ℝ)) = 1 / 2 by push_cast; field_simp]
    exact ArchytasZX.Sym.ex_half
  rcases bit_cases (σ o) (hσ o) with e1 | e1 <;> rcases bit_cases (σ i) (hσ i) with e2 | e2 <;>
    simp [pn, mz, bits, e1, e2, zq0, zq1, ex_zero, hx] <;> ring

/-- The X spider's fragment on the legs `ls` with the fresh variable `y`: the phase
`t0 + (t1 - t0) y + Sum_l y l / 2` in units of `1 / M`, the constant `sqrt 2 ^ (-|ls|)`. -/
noncomputable def xFrag (M t0 t1 y : ℕ) (ls : List ℕ) : Frag :=
  ⟨[y], [([], t0), ([y], t1 + (M - t0))] ++ ls.map (fun l => ([y, l], M / 2)), [],
    (rt 2)⁻¹ ^ ls.length⟩

theorem ex_half_r (M r : ℕ) (hM : 0 < M) (hr : M = 2 * r) : ex ((r : ℝ) / M) = -1 := by
  have hr0 : (r : ℝ) ≠ 0 := by
    intro e
    have : r = 0 := by exact_mod_cast e
    omega
  rw [hr, show ((r : ℝ) / ((2 * r : ℕ) : ℝ)) = 1 / 2 by push_cast; field_simp]
  exact ArchytasZX.Sym.ex_half

theorem fou2_ex (M r j k : ℕ) (hM : 0 < M) (hr : M = 2 * r) (hj : j < 2) (hk : k < 2) :
    fou 2 j k = (rt 2)⁻¹ * ex (((r * zq ((k : ZMod 2) * (j : ZMod 2)) : ℤ) : ℝ) / M) := by
  have hx := ex_half_r M r hM hr
  rw [fou2 j k hj hk]
  rcases bit_cases j hj with rfl | rfl <;> rcases bit_cases k hk with rfl | rfl <;>
    simp [zq0, zq1, ex_zero, hx] <;> ring

theorem fou2_conj (j k : ℕ) (hj : j < 2) (hk : k < 2) :
    (starRingEnd ℂ) (fou 2 j k) = fou 2 j k := by
  rw [fou2 j k hj hk]
  split_ifs <;> simp [map_div₀, rt, Complex.conj_ofReal]

theorem prod_ex (a : ℂ) (M : ℕ) (g : ℕ → ℤ) : ∀ L : List ℕ,
    (L.map fun l => a * ex ((g l : ℝ) / M)).prod =
      a ^ L.length * ex ((((L.map g).sum : ℤ) : ℝ) / M)
  | [] => by simp [ex_zero]
  | l :: L => by
    rw [List.map_cons, List.prod_cons, prod_ex a M g L, List.map_cons, List.sum_cons,
      List.length_cons, Int.cast_add, add_div, ex_add]
    ring

theorem x_frag (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (n : Node) (hk : n.kind = .X)
    (hd : E.dimv n.dim = 2) (M r t0 t1 y : ℕ) (hM : 0 < M) (hr : M = 2 * r) (ht : t0 ≤ M)
    (h0 : E.phase n.ph 0 = (t0 : ℝ) / M) (h1 : E.phase n.ph 1 = (t1 : ℝ) / M)
    (hy : y ∉ n.outs ++ n.ins) :
    ArchytasZX.val E σ n = fval M (xFrag M t0 t1 y (n.outs ++ n.ins)) (bits σ) := by
  have hv : ArchytasZX.val E σ n = tX 2 (E.phase n.ph) (n.outs.map σ) (n.ins.map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd]
  set g : ℕ → ℕ → ℤ := fun k l => r * zq ((k : ZMod 2) * (σ l : ZMod 2)) with hg
  have hleg : ∀ k < 2, ∀ L : List ℕ, ((L.map σ).map fun o => fou 2 o k).prod =
      (rt 2)⁻¹ ^ L.length * ex ((((L.map (g k)).sum : ℤ) : ℝ) / M) := by
    intro k hk2 L
    simp only [List.map_map, Function.comp_def]
    rw [List.map_congr_left (fun l _ => fou2_ex M r (σ l) k hM hr (hσ l) hk2)]
    exact prod_ex _ M (g k) L
  have hconj : ∀ k < 2, ∀ L : List ℕ, ((L.map σ).map fun i => (starRingEnd ℂ) (fou 2 i k)) =
      (L.map σ).map fun o => fou 2 o k := by
    intro k hk2 L
    simp only [List.map_map]
    exact List.map_congr_left fun l _ => fou2_conj _ _ (hσ l) hk2
  rw [hv]
  unfold tX
  rw [Finset.sum_range_succ, Finset.sum_range_one, hconj 0 (by norm_num),
    hconj 1 (by norm_num), hleg 0 (by norm_num), hleg 0 (by norm_num), hleg 1 (by norm_num),
    hleg 1 (by norm_num)]
  unfold fval val xFrag
  simp only [bsum, summand, List.append_nil, List.not_mem_nil, IsEmpty.forall_iff,
    forall_const, ite_true]
  have hr2 : M / 2 = r := by omega
  have hpn : ∀ k : ℕ, pn (Function.update (bits σ) y (k : ZMod 2))
      ([([], t0), ([y], t1 + (M - t0))] ++ (n.outs ++ n.ins).map fun l => ([y, l], M / 2)) =
      (t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * zq (k : ZMod 2) +
        ((n.outs.map (g k)).sum + (n.ins.map (g k)).sum) := by
    intro k
    rw [pn_append, pn_cons, pn_cons, pn_nil, ← List.sum_append, ← List.map_append]
    have hm : ∀ l ∈ n.outs ++ n.ins,
        mz (Function.update (bits σ) y (k : ZMod 2)) [y, l] = (k : ZMod 2) * (σ l : ZMod 2) := by
      intro l hl
      have hyl : l ≠ y := fun e => hy (e ▸ hl)
      simp [mz, bits, Function.update_of_ne hyl]
    have hs : pn (Function.update (bits σ) y (k : ZMod 2))
        ((n.outs ++ n.ins).map fun l => ([y, l], M / 2)) = ((n.outs ++ n.ins).map (g k)).sum := by
      simp only [pn, List.map_map, Function.comp_def]
      congr 1
      apply List.map_congr_left
      intro l hl
      rw [hm l hl, hr2]
    rw [hs]
    simp [mz, zq1]
  rw [show (0 : ZMod 2) = ((0 : ℕ) : ZMod 2) from Nat.cast_zero.symm,
    show (1 : ZMod 2) = ((1 : ℕ) : ZMod 2) from Nat.cast_one.symm, hpn 0, hpn 1, h0, h1,
    List.length_append, pow_add]
  have hM' : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  set So0 := (n.outs.map (g 0)).sum
  set Si0 := (n.ins.map (g 0)).sum
  set So1 := (n.outs.map (g 1)).sum
  set Si1 := (n.ins.map (g 1)).sum
  have e0 : ((((t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * zq ((0 : ℕ) : ZMod 2) + (So0 + Si0)) : ℤ) :
      ℝ) / M = (t0 : ℝ) / M + (So0 : ℝ) / M + (Si0 : ℝ) / M := by
    rw [Nat.cast_zero, zq0]
    push_cast
    field_simp
    ring
  have e1 : ((((t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * zq ((1 : ℕ) : ZMod 2) + (So1 + Si1)) : ℤ) :
      ℝ) / M = (t1 : ℝ) / M + (So1 : ℝ) / M + (Si1 : ℝ) / M + ((1 : ℤ) : ℝ) := by
    rw [Nat.cast_one, zq1, Nat.cast_add, Nat.cast_sub ht]
    push_cast
    field_simp
    ring
  rw [e0, e1, ex_int, ex_add, ex_add, ex_add, ex_add]
  ring

/-! ## Networks -/

theorem tZ_congr (d : ℕ) (φ : ℕ → ℝ) (a b : List ℕ) (h : ∀ x, x ∈ a ↔ x ∈ b) :
    tZ d φ a = tZ d φ b := by
  unfold tZ
  apply Finset.sum_congr rfl
  intro k _
  have : (∀ v ∈ a, v = k) ↔ (∀ v ∈ b, v = k) :=
    ⟨fun H v hv => H v ((h v).2 hv), fun H v hv => H v ((h v).1 hv)⟩
  simp only [this]

/-- The Z spider's fragment on the legs `ls` around the leg `l0`: `l0 + l = 0` per leg and the
phase `t0 + (t1 - t0) l0`, in units of `1 / M`. -/
def zFragAt (M t0 t1 l0 : ℕ) (ls : List ℕ) : Frag :=
  ⟨[], [([], t0), ([l0], t1 + (M - t0))], ls.map fun l => [[l0], [l]], 1⟩

theorem z_frag_at (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < 2) (n : Node) (hk : n.kind = .Z)
    (hd : E.dimv n.dim = 2) (M t0 t1 : ℕ) (hM : 0 < M) (ht : t0 ≤ M)
    (h0 : E.phase n.ph 0 = (t0 : ℝ) / M) (h1 : E.phase n.ph 1 = (t1 : ℝ) / M) (l0 : ℕ)
    (hl : l0 ∈ n.outs ++ n.ins) :
    ArchytasZX.val E σ n = fval M (zFragAt M t0 t1 l0 (n.outs ++ n.ins)) (bits σ) := by
  have hv : ArchytasZX.val E σ n = tZ 2 (E.phase n.ph) ((n.outs ++ n.ins).map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd, List.map_append]
  have hc0 : tZ 2 (E.phase n.ph) ((n.outs ++ n.ins).map σ) =
      tZ 2 (E.phase n.ph) (σ l0 :: (n.outs ++ n.ins).map σ) := by
    apply tZ_congr
    intro x
    simp only [List.mem_cons]
    constructor
    · exact Or.inr
    · rintro (rfl | hx)
      · exact List.mem_map.2 ⟨l0, hl, rfl⟩
      · exact hx
  rw [hv, hc0, tZ_bits _ _ _ (hσ l0)]
  unfold fval val zFragAt
  simp only [bsum, summand, List.append_nil, List.forall_mem_map]
  have hc : ∀ l, fz (bits σ) [[l0], [l]] = 0 ↔ σ l = σ l0 := by
    intro l
    have e : fz (bits σ) [[l0], [l]] = (σ l0 : ZMod 2) + (σ l : ZMod 2) := by
      simp [fz, mz, bits]
    rw [e, bits_eq _ _ (hσ l0) (hσ l)]
  simp only [hc]
  have hp : pn (bits σ) [([], t0), ([l0], t1 + (M - t0))] =
      (t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * (σ l0 : ℤ) := by
    simp [pn, mz, bits, zq_bit _ (hσ l0), zq1]
  rw [hp]
  split_ifs
  · rw [one_mul]
    rcases bit_cases (σ l0) (hσ l0) with e | e
    · rw [e, h0]; push_cast; simp
    · rw [e, h1]
      have : (((t0 : ℤ) + ((t1 + (M - t0) : ℕ) : ℤ) * ((1 : ℕ) : ℤ) : ℤ) : ℝ) / M =
          (t1 : ℝ) / M + ((1 : ℤ) : ℝ) := by
        rw [Nat.cast_add, Nat.cast_sub ht]
        push_cast
        field_simp
        ring
      rw [this, ex_int]
  · rfl

/-- The network's path sum over its fragments: the summed labels then the fragments' paths. -/
def netData (N : Net) (fs : List Frag) : Data :=
  ⟨N.int ++ (joinAll fs).path, (joinAll fs).phase, (joinAll fs).cons, []⟩

/-- `σ` on the labels `bnd`, `0` elsewhere. -/
def clamp (bnd : List ℕ) (σ : ℕ → ℕ) : ℕ → ℕ := fun l => if l ∈ bnd then σ l else 0

/-- The nodes of `N` are the first components of `ps`, each node's entry is its fragment's value
at labels of dimension 2, the fragments' paths are disjoint, and every label is summed or on the
boundary, of dimension 2. -/
structure NetPS (E : Env) (N : Net) (M : ℕ) (ps : List (Node × Frag)) : Prop where
  nodes : N.nodes = ps.map Prod.fst
  node : ∀ p ∈ ps, ∀ σ, (∀ l, σ l < 2) → ArchytasZX.val E σ p.1 = fval M p.2 (bits σ)
  disj : (ps.map Prod.snd).Pairwise fun a b => Disj a b ∧ Disj b a
  int : ∀ l ∈ N.int, labDim E l = 2
  bnd : ∀ l ∈ N.bnd, labDim E l = 2
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd

theorem NetPS.eval_eq {E : Env} {N : Net} {M : ℕ} {ps : List (Node × Frag)} (h : NetPS E N M ps)
    (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = val M (joinAll (ps.map Prod.snd)).c (netData N (ps.map Prod.snd))
      (bits (clamp N.bnd σ)) := by
  have he : eval E N σ = eval E N (clamp N.bnd σ) := by
    unfold eval
    apply sumOver_ext (labDim E) N.int (labelsOf N.nodes) _ (prodVal_reads E N.nodes)
    intro x hx hn
    rcases h.cover x hx with hi | hb
    · exact absurd hi hn
    · simp [clamp, hb]
  have hc : ∀ l, clamp N.bnd σ l < 2 := by
    intro l
    unfold clamp
    split_ifs with hl
    · have := hσ l hl
      rwa [h.bnd l hl] at this
    · omega
  rw [he, eval_frags E N M ps h.nodes h.node h.disj h.int _ hc]
  rfl

/-- Two networks with one boundary whose fragments' path sums, scaled by `s₁` and `s₂`, agree
at every Boolean assignment. -/
theorem equal_ps {E : Env} {N₁ N₂ : Net} {M : ℕ} {ps₁ ps₂ : List (Node × Frag)} (s₁ s₂ : ℂ)
    (h₁ : NetPS E N₁ M ps₁) (h₂ : NetPS E N₂ M ps₂) (hb : N₁.bnd = N₂.bnd)
    (hv : ∀ τ, s₁ * val M (joinAll (ps₁.map Prod.snd)).c (netData N₁ (ps₁.map Prod.snd)) τ =
      s₂ * val M (joinAll (ps₂.map Prod.snd)).c (netData N₂ (ps₂.map Prod.snd)) τ) :
    Equal E s₁ N₁ s₂ N₂ := by
  refine ⟨hb ▸ List.Perm.refl _, fun σ hσ => ?_⟩
  have hσ₂ : InRange E N₂.bnd σ := hb ▸ hσ
  rw [h₁.eval_eq σ hσ, h₂.eval_eq σ hσ₂, ← hb]
  exact hv _

end PSem
end ArchytasZX
