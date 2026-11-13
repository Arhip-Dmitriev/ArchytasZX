/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.PathSumQudit
import ArchytasZX.PathSumDigits

/-!
# Qudit networks as path sums over Z_p

A `QFrag` is one generator's path sum over `ZMod p`: path variables, phase terms, constraints
and a constant. `fval_join` multiplies two fragments with disjoint path variables, `sumOver_qsum`
turns the network's sum over labels of dimension `p` into a sum over `ZMod p`, and
`QNetPS.eval_eq` gives a network's value as `val` of its joined fragments with the summed labels
as path variables, read at `zm p σ`. `z_frag_q` (constraints `l0 - l = 0` per leg, the phase
table as a polynomial in `l0`), `x_frag_q` (a fresh variable `y`, `y o` per output and `-y i` per
input, constant `sqrt p ^ (-legs)`) and `f_frag_q` (`o i`, constant `sqrt p ^ (-1)`) are the
fragments of Z, X and F on dimension `p`. `equal_qps` gives `Equal` from the two path sums.
-/

namespace ArchytasZX
namespace QSem

open Complex Finset

/-- A generator's path sum over `ZMod p`. -/
structure QFrag where
  path : List ℕ
  phase : Poly
  cons : List Poly
  c : ℂ

/-- A fragment's value. -/
noncomputable def fval (p : ℕ) [NeZero p] (f : QFrag) (τ : ℕ → ZMod p) : ℂ :=
  val p f.c ⟨f.path, f.phase, f.cons, []⟩ τ

/-- Two fragments side by side. -/
def QFrag.join (a b : QFrag) : QFrag :=
  ⟨a.path ++ b.path, a.phase ++ b.phase, a.cons ++ b.cons, a.c * b.c⟩

/-- No path variable of `a` in `b`. -/
def QDisj (a b : QFrag) : Prop :=
  ∀ y ∈ a.path, y ∉ b.path ∧ y ∉ pvars b.phase ∧ y ∉ cvars b.cons

instance (a b : QFrag) : Decidable (QDisj a b) := by
  unfold QDisj
  infer_instance

/-- The empty fragment. -/
def QFrag.unit : QFrag := ⟨[], [], [], 1⟩

/-- Fragments side by side. -/
def qjoinAll : List QFrag → QFrag
  | [] => QFrag.unit
  | f :: fs => f.join (qjoinAll fs)

/-- Label values read in `ZMod p`. -/
def zm (p : ℕ) (σ : ℕ → ℕ) : ℕ → ZMod p := fun l => (σ l : ZMod p)

/-- Whether `P` takes the value `ts[k]` at `k` on every variable for `k < p`, and has no variable
but `y`. -/
def tabOK (p y : ℕ) (P : Poly) (ts : List ℕ) : Bool :=
  (List.range p).all (fun k => decide (qv (fun _ => (k : ZMod p)) P = ((ts.getD k 0 : ℕ) : ZMod p)))
    && (pvars P).all fun v => decide (v = y)

/-- The Z spider's fragment around the leg `l0`: `l0 - l = 0` per leg `l`, the phase `P`. -/
def zFragQ (p l0 : ℕ) (ls : List ℕ) (P : Poly) : QFrag :=
  ⟨[], P, ls.map fun l => [([(l0, 1)], 1), ([(l, 1)], p - 1)], 1⟩

/-- F's fragment on the legs `o`, `i`: the phase `o i`. -/
noncomputable def fFragQ (p o i : ℕ) : QFrag := ⟨[], [([(o, 1), (i, 1)], 1)], [], (rt p)⁻¹⟩

/-- The X spider's fragment with the fresh variable `y`: the phase `P(y) + Sum_o y o -
Sum_i y i`, constant `sqrt p ^ (-legs)`. -/
noncomputable def xFragQ (p y : ℕ) (P : Poly) (os is : List ℕ) : QFrag :=
  ⟨[y], P ++ os.map (fun o => ([(y, 1), (o, 1)], 1)) ++ is.map (fun i => ([(y, 1), (i, 1)], p - 1)),
    [], (rt p)⁻¹ ^ (os.length + is.length)⟩

/-- The network's path sum over its fragments: the summed labels then the fragments' paths. -/
def qnetData (N : Net) (fs : List QFrag) : Data :=
  ⟨N.int ++ (qjoinAll fs).path, (qjoinAll fs).phase, (qjoinAll fs).cons, []⟩

section Net

variable {p : ℕ} [NeZero p]

theorem qsum_append (f : (ℕ → ZMod p) → ℂ) :
    ∀ (a b : List ℕ) (σ : ℕ → ZMod p), qsum (a ++ b) f σ = qsum a (qsum b f) σ
  | [], _, _ => rfl
  | y :: a, b, σ => by
    simp only [List.cons_append, qsum, qsum_append f a b]

theorem qsum_mul_left (g f : (ℕ → ZMod p) → ℂ) :
    ∀ (ys : List ℕ), (∀ τ, ∀ y ∈ ys, ∀ w, g (Function.update τ y w) = g τ) →
      ∀ σ, qsum ys (fun τ => g τ * f τ) σ = g σ * qsum ys f σ
  | [], _, _ => rfl
  | y :: ys, h, σ => by
    simp only [qsum, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro b _
    rw [qsum_mul_left g f ys (fun τ y' hy' w => h τ y' (List.mem_cons_of_mem _ hy') w),
      h σ y List.mem_cons_self b]

theorem qsum_inv (g : (ℕ → ZMod p) → ℂ) (y : ℕ) (b : ZMod p)
    (hg : ∀ τ, g (Function.update τ y b) = g τ) :
    ∀ (zs : List ℕ) (τ : ℕ → ZMod p), y ∉ zs →
      qsum zs g (Function.update τ y b) = qsum zs g τ := by
  intro zs τ hy
  rw [qsum_out g y b zs τ hy]
  exact qsum_congr _ _ hg zs τ

theorem val_smul (a c : ℂ) (s : Data) (σ : ℕ → ZMod p) :
    val p (a * c) s σ = a * val p c s σ := by
  unfold val
  rw [← qsum_mul_left (fun _ => a) _ s.path (fun _ _ _ _ => rfl)]
  apply qsum_congr
  intro τ
  unfold summand
  split_ifs <;> ring

theorem summand_join (a b : QFrag) (τ : ℕ → ZMod p) :
    summand p (a.join b).c ⟨(a.join b).path, (a.join b).phase, (a.join b).cons, []⟩ τ =
      summand p a.c ⟨a.path, a.phase, a.cons, []⟩ τ *
        summand p b.c ⟨b.path, b.phase, b.cons, []⟩ τ := by
  unfold summand QFrag.join
  dsimp only
  simp only [List.append_nil, List.forall_mem_append, qv_append, ψ_add]
  by_cases ha : ∀ q ∈ a.cons, qv τ q = 0
  · by_cases hb : ∀ q ∈ b.cons, qv τ q = 0
    · rw [if_pos ⟨ha, hb⟩, if_pos ha, if_pos hb]
      ring
    · rw [if_neg (fun h => hb h.2), if_neg hb, mul_zero]
  · rw [if_neg (fun h => ha h.1), if_neg ha, zero_mul]

theorem summand_frag_inv (b : QFrag) (y : ℕ) (w : ZMod p) (hp : y ∉ pvars b.phase)
    (hc : y ∉ cvars b.cons) (τ : ℕ → ZMod p) :
    summand p b.c ⟨b.path, b.phase, b.cons, []⟩ (Function.update τ y w) =
      summand p b.c ⟨b.path, b.phase, b.cons, []⟩ τ :=
  summand_update_not b.c _ τ y w hp (by simpa using hc)

theorem fval_join (a b : QFrag) (hab : QDisj a b) (hba : QDisj b a) (τ : ℕ → ZMod p) :
    fval p (a.join b) τ = fval p a τ * fval p b τ := by
  unfold fval val
  show qsum (a.path ++ b.path) (summand p (a.join b).c
    ⟨(a.join b).path, (a.join b).phase, (a.join b).cons, []⟩) τ = _
  rw [qsum_congr _ _ (summand_join a b), qsum_append]
  have hb : ∀ τ', ∀ y ∈ a.path, ∀ w,
      qsum b.path (summand p b.c ⟨b.path, b.phase, b.cons, []⟩) (Function.update τ' y w) =
        qsum b.path (summand p b.c ⟨b.path, b.phase, b.cons, []⟩) τ' := by
    intro τ' y hy w
    obtain ⟨h1, h2, h3⟩ := hab y hy
    exact qsum_inv _ y w (summand_frag_inv b y w h2 h3) b.path τ' h1
  have ha : ∀ τ', ∀ y ∈ b.path, ∀ w,
      summand p a.c ⟨a.path, a.phase, a.cons, []⟩ (Function.update τ' y w) =
        summand p a.c ⟨a.path, a.phase, a.cons, []⟩ τ' := by
    intro τ' y hy w
    obtain ⟨_, h2, h3⟩ := hba y hy
    exact summand_frag_inv a y w h2 h3 τ'
  have inner : ∀ τ', qsum b.path (fun ρ => summand p a.c ⟨a.path, a.phase, a.cons, []⟩ ρ *
      summand p b.c ⟨b.path, b.phase, b.cons, []⟩ ρ) τ' =
      summand p a.c ⟨a.path, a.phase, a.cons, []⟩ τ' *
        qsum b.path (summand p b.c ⟨b.path, b.phase, b.cons, []⟩) τ' :=
    fun τ' => qsum_mul_left _ _ b.path ha τ'
  rw [qsum_congr (qsum b.path fun ρ => summand p a.c ⟨a.path, a.phase, a.cons, []⟩ ρ *
      summand p b.c ⟨b.path, b.phase, b.cons, []⟩ ρ) _ inner]
  have swap : ∀ τ', summand p a.c ⟨a.path, a.phase, a.cons, []⟩ τ' *
      qsum b.path (summand p b.c ⟨b.path, b.phase, b.cons, []⟩) τ' =
      qsum b.path (summand p b.c ⟨b.path, b.phase, b.cons, []⟩) τ' *
        summand p a.c ⟨a.path, a.phase, a.cons, []⟩ τ' := fun τ' => mul_comm _ _
  rw [qsum_congr _ _ swap, qsum_mul_left _ _ a.path hb τ]
  exact mul_comm _ _

theorem pvars_append (P R : Poly) : pvars (P ++ R) = pvars P ++ pvars R := by
  simp [pvars, List.flatMap_append]

theorem cvars_append (C D : List Poly) : cvars (C ++ D) = cvars C ++ cvars D := by
  simp [cvars, List.flatMap_append]

theorem qdisj_join_right (a b c : QFrag) (hb : QDisj a b) (hc : QDisj a c) :
    QDisj a (b.join c) := by
  intro y hy
  obtain ⟨b1, b2, b3⟩ := hb y hy
  obtain ⟨c1, c2, c3⟩ := hc y hy
  refine ⟨?_, ?_, ?_⟩
  · simp only [QFrag.join, List.mem_append, not_or]; exact ⟨b1, c1⟩
  · simp only [QFrag.join, pvars_append, List.mem_append, not_or]; exact ⟨b2, c2⟩
  · simp only [QFrag.join, cvars_append, List.mem_append, not_or]; exact ⟨b3, c3⟩

theorem qdisj_join_left (a b c : QFrag) (hb : QDisj b a) (hc : QDisj c a) :
    QDisj (b.join c) a := by
  intro y hy
  simp only [QFrag.join, List.mem_append] at hy
  rcases hy with hy | hy
  · exact hb y hy
  · exact hc y hy

theorem qdisj_unit_left (a : QFrag) : QDisj QFrag.unit a := by
  intro y hy; simp [QFrag.unit] at hy

theorem qdisj_unit_right (a : QFrag) : QDisj a QFrag.unit := by
  intro y _; simp [QFrag.unit, pvars, cvars]

theorem qdisj_joinAll_right (f : QFrag) :
    ∀ fs : List QFrag, (∀ g ∈ fs, QDisj f g) → QDisj f (qjoinAll fs)
  | [], _ => qdisj_unit_right f
  | g :: gs, h => qdisj_join_right f g _ (h g List.mem_cons_self)
      (qdisj_joinAll_right f gs fun g' hg' => h g' (List.mem_cons_of_mem _ hg'))

theorem qdisj_joinAll_left (f : QFrag) :
    ∀ fs : List QFrag, (∀ g ∈ fs, QDisj g f) → QDisj (qjoinAll fs) f
  | [], _ => qdisj_unit_left f
  | g :: gs, h => qdisj_join_left f g _ (h g List.mem_cons_self)
      (qdisj_joinAll_left f gs fun g' hg' => h g' (List.mem_cons_of_mem _ hg'))

theorem fval_unit (τ : ℕ → ZMod p) : fval p QFrag.unit τ = 1 := by
  simp [fval, val, QFrag.unit, qsum, summand, qv, ψ_zero]

theorem fval_joinAll (τ : ℕ → ZMod p) :
    ∀ fs : List QFrag, fs.Pairwise (fun a b => QDisj a b ∧ QDisj b a) →
      fval p (qjoinAll fs) τ = (fs.map fun f => fval p f τ).prod
  | [], _ => fval_unit τ
  | f :: fs, h => by
    rw [List.pairwise_cons] at h
    have hd : QDisj f (qjoinAll fs) ∧ QDisj (qjoinAll fs) f :=
      ⟨qdisj_joinAll_right f fs fun g hg => (h.1 g hg).1,
        qdisj_joinAll_left f fs fun g hg => (h.1 g hg).2⟩
    rw [qjoinAll, fval_join f _ hd.1 hd.2, fval_joinAll τ fs h.2, List.map_cons, List.prod_cons]

/-! ## Labels of dimension p -/

theorem zm_update (σ : ℕ → ℕ) (l j : ℕ) :
    zm p (Function.update σ l j) = Function.update (zm p σ) l (j : ZMod p) := by
  funext v
  by_cases h : v = l
  · subst h; simp [zm]
  · simp [zm, Function.update_of_ne h]

theorem sumOver_qsum (D : ℕ → ℕ) (f : (ℕ → ℕ) → ℂ) (g : (ℕ → ZMod p) → ℂ)
    (hfg : ∀ σ, (∀ l, σ l < p) → f σ = g (zm p σ)) :
    ∀ (ls : List ℕ), (∀ l ∈ ls, D l = p) →
      ∀ σ, (∀ l, σ l < p) → sumOver D ls f σ = qsum ls g (zm p σ)
  | [], _, σ, hσ => hfg σ hσ
  | l :: ls, hD, σ, hσ => by
    have hD' : ∀ l' ∈ ls, D l' = p := fun l' h => hD l' (List.mem_cons_of_mem _ h)
    have hup : ∀ j < p, ∀ v, Function.update σ l j v < p := by
      intro j hj v
      by_cases h : v = l
      · subst h; simpa using hj
      · rw [Function.update_of_ne h]; exact hσ v
    simp only [sumOver, qsum, hD l List.mem_cons_self]
    have hR : ∑ b : ZMod p, qsum ls g (Function.update (zm p σ) l b) =
        ∑ k ∈ range p, qsum ls g (Function.update (zm p σ) l (k : ZMod p)) := by
      rw [← zsum_range]
      simp only [ZMod.natCast_zmod_val]
    rw [hR]
    apply Finset.sum_congr rfl
    intro j hj
    rw [sumOver_qsum D f g hfg ls hD' _ (hup j (Finset.mem_range.1 hj)), zm_update]

theorem qv_agree (τ τ' : ℕ → ZMod p) (P : Poly) (h : ∀ x ∈ pvars P, τ x = τ' x) :
    qv τ P = qv τ' P := by
  unfold qv
  congr 1
  apply List.map_congr_left
  intro t ht
  congr 1
  unfold qm
  congr 1
  apply List.map_congr_left
  intro u hu
  rw [h u.1 (List.mem_flatMap.2 ⟨t, ht, List.mem_map.2 ⟨u, hu, rfl⟩⟩)]

theorem tab_val (y : ℕ) (P : Poly) (ts : List ℕ) (htab : tabOK p y P ts = true)
    (τ : ℕ → ZMod p) (k : ℕ) (hk : k < p) (hy : τ y = (k : ZMod p)) :
    ψ p (qv τ P) = ex (((ts.getD k 0 : ℕ) : ℝ) / p) := by
  simp only [tabOK, Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq] at htab
  obtain ⟨hv, hP⟩ := htab
  rw [qv_agree τ (fun _ => (k : ZMod p)) P (fun x hx => by rw [hP x hx]; exact hy), hv k hk,
    ψ_nat, om_eq_ex]

/-! ## Generators on dimension p -/

theorem z_frag_q (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < p) (n : Node) (hk : n.kind = .Z)
    (hd : E.dimv n.dim = p) (l0 : ℕ) (hl : l0 ∈ n.outs ++ n.ins) (P : Poly) (ts : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / p) (htab : tabOK p l0 P ts = true) :
    ArchytasZX.val E σ n = fval p (zFragQ p l0 (n.outs ++ n.ins) P) (zm p σ) := by
  have hv : ArchytasZX.val E σ n = tZ p (E.phase n.ph) ((n.outs ++ n.ins).map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd, List.map_append]
  have hc0 : tZ p (E.phase n.ph) ((n.outs ++ n.ins).map σ) =
      tZ p (E.phase n.ph) (σ l0 :: (n.outs ++ n.ins).map σ) := by
    apply PSem.tZ_congr
    intro x
    simp only [List.mem_cons]
    constructor
    · exact Or.inr
    · rintro (rfl | hx)
      · exact List.mem_map.2 ⟨l0, hl, rfl⟩
      · exact hx
  rw [hv, hc0, PSem.tZ_val _ _ _ _ (hσ l0)]
  unfold fval val zFragQ
  simp only [qsum, summand, List.append_nil, List.forall_mem_map]
  have hc : ∀ l, qv (zm p σ) [([(l0, 1)], 1), ([(l, 1)], p - 1)] = 0 ↔ σ l = σ l0 := by
    intro l
    have e : qv (zm p σ) [([(l0, 1)], 1), ([(l, 1)], p - 1)] =
        (σ l0 : ZMod p) - (σ l : ZMod p) := by
      rw [qv_cons, qv_cons, qv_nil, qm_cons, qm_nil, qm_cons, qm_nil, cast_pred_one]
      simp only [zm]
      push_cast
      ring
    rw [e, sub_eq_zero, ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt (hσ l0),
      Nat.mod_eq_of_lt (hσ l)]
    exact eq_comm
  simp only [hc]
  rw [tab_val l0 P ts htab (zm p σ) (σ l0) (hσ l0) rfl, ← hph]
  split_ifs <;> simp

theorem fou_q (j k : ℕ) :
    fou p j k = (rt p)⁻¹ * ψ p (((1 : ℕ) : ZMod p) * ((k : ZMod p) * (j : ZMod p))) := by
  have e : ((1 : ℕ) : ZMod p) * ((k : ZMod p) * (j : ZMod p)) = ((j * k : ℕ) : ZMod p) := by
    push_cast; ring
  rw [e, ψ_nat]
  unfold fou
  rw [div_eq_mul_inv, mul_comm]

theorem conj_fou_q (j k : ℕ) :
    (starRingEnd ℂ) (fou p j k) =
      (rt p)⁻¹ * ψ p (((p - 1 : ℕ) : ZMod p) * ((k : ZMod p) * (j : ZMod p))) := by
  have e : ((p - 1 : ℕ) : ZMod p) * ((k : ZMod p) * (j : ZMod p)) =
      (((p - 1) * (j * k) : ℕ) : ZMod p) := by
    push_cast; ring
  rw [e, ψ_nat]
  unfold fou
  rw [map_div₀, map_pow, conj_om_pow p (NeZero.pos p), Complex.conj_ofReal, ← pow_mul,
    div_eq_mul_inv, mul_comm]

theorem prod_terms (g : ℕ → ℂ) (c : ℕ) (τ : ℕ → ZMod p) (σ : ℕ → ℕ) (y : ℕ) (x : ZMod p)
    (hg : ∀ j, g j = (rt p)⁻¹ * ψ p ((c : ZMod p) * (x * (j : ZMod p)))) (hy : τ y = x) :
    ∀ L : List ℕ, (∀ l ∈ L, τ l = (σ l : ZMod p)) →
      ((L.map σ).map g).prod =
        (rt p)⁻¹ ^ L.length * ψ p (qv τ (L.map fun l => ([(y, 1), (l, 1)], c)))
  | [], _ => by simp [qv, ψ_zero]
  | l :: L, h => by
    rw [List.map_cons, List.map_cons, List.prod_cons, hg,
      prod_terms g c τ σ y x hg hy L (fun l' hl' => h l' (List.mem_cons_of_mem _ hl')),
      List.map_cons, qv_cons, ψ_add, List.length_cons, pow_succ]
    simp only [qm_cons, qm_nil, pow_one, mul_one, hy, h l List.mem_cons_self]
    ring

theorem x_frag_q (E : Env) (σ : ℕ → ℕ) (hσ : ∀ l, σ l < p) (n : Node) (hk : n.kind = .X)
    (hd : E.dimv n.dim = p) (y : ℕ) (hy : y ∉ n.outs ++ n.ins) (P : Poly) (ts : List ℕ)
    (hph : ∀ v, E.phase n.ph v = ((ts.getD v 0 : ℕ) : ℝ) / p) (htab : tabOK p y P ts = true) :
    ArchytasZX.val E σ n = fval p (xFragQ p y P n.outs n.ins) (zm p σ) := by
  have hv : ArchytasZX.val E σ n = tX p (E.phase n.ph) (n.outs.map σ) (n.ins.map σ) := by
    unfold ArchytasZX.val
    rw [hk]
    simp only [hd]
  rw [hv]
  unfold tX fval val xFragQ
  simp only [qsum]
  rw [← zsum_range (fun k => ex (E.phase n.ph k) * ((n.outs.map σ).map fun o => fou p o k).prod *
    ((n.ins.map σ).map fun i => (starRingEnd ℂ) (fou p i k)).prod)]
  apply Finset.sum_congr rfl
  intro x _
  beta_reduce
  set τ := Function.update (zm p σ) y x with hτ
  have hτy : τ y = x := by rw [hτ, Function.update_self]
  have hτl : ∀ l ∈ n.outs ++ n.ins, τ l = (σ l : ZMod p) := by
    intro l hl
    have : l ≠ y := fun e => hy (e ▸ hl)
    rw [hτ, Function.update_of_ne this]
    rfl
  have hxv : ((x.val : ℕ) : ZMod p) = x := ZMod.natCast_zmod_val x
  have hO := prod_terms (fun j => fou p j x.val) 1 τ σ y x
    (fun j => by rw [fou_q, hxv]) hτy n.outs (fun l hl => hτl l (List.mem_append_left _ hl))
  have hI := prod_terms (fun j => (starRingEnd ℂ) (fou p j x.val)) (p - 1) τ σ y x
    (fun j => by rw [conj_fou_q, hxv]) hτy n.ins (fun l hl => hτl l (List.mem_append_right _ hl))
  rw [hO, hI]
  unfold summand
  rw [if_pos (by simp), qv_append, qv_append, ψ_add, ψ_add,
    tab_val y P ts htab τ x.val (ZMod.val_lt x) (by rw [hτy, hxv]), ← hph, pow_add]
  ring

theorem f_frag_q (E : Env) (σ : ℕ → ℕ) (n : Node) (hk : n.kind = .F) (hd : E.dimv n.dim = p)
    (o i : ℕ) (ho : n.outs = [o]) (hi : n.ins = [i]) :
    ArchytasZX.val E σ n = fval p (fFragQ p o i) (zm p σ) := by
  have hv : ArchytasZX.val E σ n = fou p (σ o) (σ i) := by
    unfold ArchytasZX.val
    rw [hk, ho, hi]
    simp only [hd, List.map_cons, List.map_nil]
  rw [hv, fou_q]
  unfold fval val fFragQ summand
  simp only [qsum]
  rw [if_pos (by simp)]
  congr 2
  simp [qv, qm, zm]
  ring

/-! ## Networks -/

/-- The nodes of `N` are the first components of `ps`, each node's entry is its fragment's value
at label values below `p`, the fragments' paths are disjoint, every summed or boundary label has
dimension `p`, and every label is summed or on the boundary. -/
structure QNetPS (E : Env) (N : Net) (p : ℕ) [NeZero p] (ps : List (Node × QFrag)) : Prop where
  nodes : N.nodes = ps.map Prod.fst
  node : ∀ q ∈ ps, ∀ σ, (∀ l, σ l < p) → ArchytasZX.val E σ q.1 = fval p q.2 (zm p σ)
  disj : (ps.map Prod.snd).Pairwise fun a b => QDisj a b ∧ QDisj b a
  dims : ∀ l ∈ N.int ++ N.bnd, labDim E l = p
  cover : ∀ l ∈ labelsOf N.nodes, l ∈ N.int ∨ l ∈ N.bnd

theorem QNetPS.eval_eq {E : Env} {N : Net} {ps : List (Node × QFrag)} (h : QNetPS E N p ps)
    (σ : ℕ → ℕ) (hσ : InRange E N.bnd σ) :
    eval E N σ = val p (qjoinAll (ps.map Prod.snd)).c (qnetData N (ps.map Prod.snd))
      (zm p (PSem.clamp N.bnd σ)) := by
  have he : eval E N σ = eval E N (PSem.clamp N.bnd σ) := by
    unfold eval
    apply sumOver_ext (labDim E) N.int (labelsOf N.nodes) _ (prodVal_reads E N.nodes)
    intro x hx hn
    rcases h.cover x hx with hi | hb
    · exact absurd hi hn
    · simp [PSem.clamp, hb]
  have hc : ∀ l, PSem.clamp N.bnd σ l < p := by
    intro l
    unfold PSem.clamp
    split_ifs with hl
    · have := hσ l hl
      rwa [h.dims l (List.mem_append_right _ hl)] at this
    · exact NeZero.pos p
  rw [he]
  unfold eval
  set F := qjoinAll (ps.map Prod.snd) with hF
  rw [sumOver_qsum (labDim E) (prodVal E N.nodes) (fval p F) ?_ N.int
    (fun l hl => h.dims l (List.mem_append_left _ hl)) _ hc]
  · unfold fval val qnetData
    rw [qsum_append]
    rfl
  · intro σ' hσ'
    rw [hF, fval_joinAll _ _ h.disj, prodVal, h.nodes, List.map_map, List.map_map]
    congr 1
    apply List.map_congr_left
    intro q hq
    exact h.node q hq σ' hσ'

/-- Two networks with one boundary whose path sums, scaled by `s₁` and `s₂`, agree at every
assignment. -/
theorem equal_qps {E : Env} {N₁ N₂ : Net} {ps₁ ps₂ : List (Node × QFrag)} (s₁ s₂ : ℂ)
    (h₁ : QNetPS E N₁ p ps₁) (h₂ : QNetPS E N₂ p ps₂) (hb : N₁.bnd = N₂.bnd)
    (hv : ∀ τ, s₁ * val p (qjoinAll (ps₁.map Prod.snd)).c (qnetData N₁ (ps₁.map Prod.snd)) τ =
      s₂ * val p (qjoinAll (ps₂.map Prod.snd)).c (qnetData N₂ (ps₂.map Prod.snd)) τ) :
    Equal E s₁ N₁ s₂ N₂ := by
  refine ⟨hb ▸ List.Perm.refl _, fun σ hσ => ?_⟩
  have hσ₂ : InRange E N₂.bnd σ := hb ▸ hσ
  rw [h₁.eval_eq σ hσ, h₂.eval_eq σ hσ₂, ← hb]
  exact hv _

end Net

end QSem
end ArchytasZX
