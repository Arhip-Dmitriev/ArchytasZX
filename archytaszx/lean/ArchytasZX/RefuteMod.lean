/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Contract

/-!
# Refutation by residues

`tensorW` extends `tensorT` to weighted spiders: a coefficient entry is `(Q ph)⁻¹` times a
formal sum `C ph k`. `tmv p w` sends a formal sum to `ZMod p` at `ζ ↦ w`; when `p` is prime
and `w` has order `M`, it vanishes on every formal sum vanishing at `om M` (`tmv_eq_zero`).
A residue factor holds one residue mod `p` per assignment of its scope; `eval_res` proves
that a checked contraction of residue factors gives a network's formal entry mod `p`, and
`refute_res` disproves an equality whose two sides differ mod `p`.
-/

namespace ArchytasZX

open Complex Finset Polynomial

/-! ## Weighted generators -/

/-- The weighted Z spider's entry as a formal sum. -/
def tZWT (C : ℕ → Terms) (d : ℕ) (vs : List ℕ) : Terms :=
  (List.range d).flatMap fun k => if vs.all (· == k) then C k else []

/-- A formal sum times `ζ^e`. -/
def tshift (e : ℕ) (t : Terms) : Terms := t.map fun q => (q.1, q.2.1 + e, q.2.2)

/-- The weighted X spider's entry without its `(√d)⁻¹` factors as a formal sum. -/
def tXWT (M : ℕ) (C : ℕ → Terms) (d : ℕ) (os is : List ℕ) : Terms :=
  (List.range d).flatMap fun k =>
    tshift (M / d * (k * os.sum) + M / d * ((d - 1) * (k * is.sum))) (C k)

/-- A generator's entry without its `(√d)⁻¹` and `(Q ph)⁻¹` factors as a formal sum. -/
def tensorW (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) :
    Kind → ℕ → ℕ → List ℕ → List ℕ → Terms
  | .ZW, d, ph, os, is => tZWT (C ph) d (os ++ is)
  | .XW, d, ph, os, is => tXWT M (C ph) d os is
  | .Bun m b, d, ph, os, is => ((List.range (E.mult m)).map fun i =>
      tensorW E M P C b d ph (copyLegs E.mult d i (b.outCs os.length) os)
        (copyLegs E.mult d i (b.inCs is.length) is)).foldr tmul (tc 1)
  | .Z, d, ph, os, is => tensorT E M P .Z d ph os is
  | .X, d, ph, os, is => tensorT E M P .X d ph os is
  | .F, d, ph, os, is => tensorT E M P .F d ph os is
  | .T, d, ph, os, is => tensorT E M P .T d ph os is
  | .Ti, d, ph, os, is => tensorT E M P .Ti d ph os is
  | .W, d, ph, os, is => tensorT E M P .W d ph os is
  | .B, d, ph, os, is => tensorT E M P .B d ph os is
  | .S, d, ph, os, is => tensorT E M P .S d ph os is
  | .I, d, ph, os, is => tensorT E M P .I d ph os is
  | .ZB cs, d, ph, os, is => tensorT E M P (.ZB cs) d ph os is
  | .XB co ci, d, ph, os, is => tensorT E M P (.XB co ci) d ph os is
  | .WB cs, d, ph, os, is => tensorT E M P (.WB cs) d ph os is

/-- The number of `(√d)⁻¹` factors of a generator with `no` outputs and `ni` inputs. -/
def sqcW (E : Shape) : Kind → ℕ → ℕ → ℕ → ℕ
  | .XW, _, no, ni => no + ni
  | .Bun m b, d, no, ni => E.mult m * sqcW E b d no ni
  | k, d, no, ni => sqc E k d no ni

/-- The number of `(Q ph)⁻¹` factors of a generator. -/
def qcW (E : Shape) : Kind → ℕ
  | .ZW => 1
  | .XW => 1
  | .Bun m b => E.mult m * qcW E b
  | _ => 0

/-- A kind that reads a phase or coefficient vector. -/
def Kind.phased : Kind → Bool
  | .Z => true
  | .X => true
  | .ZB _ => true
  | .XB _ _ => true
  | .ZW => true
  | .XW => true
  | .Bun _ b => b.phased
  | _ => false

theorem Kind.phased_of_noZW : ∀ k : Kind, k.noZW = false → k.phased = true
  | .ZW, _ => rfl
  | .XW, _ => rfl
  | .Bun _ b, h => Kind.phased_of_noZW b h
  | .Z, h | .X, h | .F, h | .T, h | .Ti, h | .W, h | .B, h | .S, h | .I, h | .ZB _, h
  | .XB _ _, h | .WB _, h => by simp [Kind.noZW] at h

theorem tv_tshift (z : ℂ) (xs : List ℂ) (e : ℕ) (t : Terms) :
    tv z xs (tshift e t) = z ^ e * tv z xs t := by
  induction t with
  | nil => simp [tshift, tv]
  | cons q t ih =>
    simp only [tshift, List.map_cons] at ih ⊢
    simp only [tv_cons, ih, pow_add]
    ring

theorem tZw_cyc (M d : ℕ) (xs : List ℂ) (C : ℕ → Terms) (c : ℕ → ℂ) (q : ℂ) (hq : q ≠ 0)
    (hc : ∀ k < d, q * c k = tv (om M) xs (C k)) (vs : List ℕ) :
    tZw d c vs = q⁻¹ * tv (om M) xs (tZWT C d vs) := by
  unfold tZw tZWT
  rw [fsum_list, tv_flatMap, ← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro k hk
  have hk' := List.mem_range.1 hk
  have hck : c k = q⁻¹ * tv (om M) xs (C k) := by
    rw [← hc k hk', ← mul_assoc, inv_mul_cancel₀ hq, one_mul]
  by_cases h : ∀ v ∈ vs, v = k
  · have : vs.all (· == k) = true := List.all_eq_true.2 fun v hv => by simp [h v hv]
    rw [ite_eq_left h, if_pos this, hck]
  · have : vs.all (· == k) = false := by
      rw [Bool.eq_false_iff]; intro h'
      exact h fun v hv => by simpa using List.all_eq_true.1 h' v hv
    rw [ite_eq_right h, if_neg (by simp [this])]; simp [tv]

theorem tXw_cyc (M d : ℕ) (hM : 0 < M) (hdM : d ∣ M) (xs : List ℂ) (C : ℕ → Terms)
    (c : ℕ → ℂ) (q : ℂ) (hq : q ≠ 0) (hc : ∀ k < d, q * c k = tv (om M) xs (C k))
    (os is : List ℕ) :
    tXw d c os is = (rt d)⁻¹ ^ (os.length + is.length) * q⁻¹ * tv (om M) xs (tXWT M C d os is) := by
  have hd : 0 < d := Nat.pos_of_dvd_of_pos hdM hM
  obtain ⟨r, hr⟩ := hdM
  have hMd : M / d = r := by rw [hr, Nat.mul_div_cancel_left _ hd]
  have hom : om d = om M ^ r := om_sub M d r hM hr
  unfold tXw tXWT
  rw [fsum_list, tv_flatMap, ← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro k hk
  have hck : c k = q⁻¹ * tv (om M) xs (C k) := by
    rw [← hc k (List.mem_range.1 hk), ← mul_assoc, inv_mul_cancel₀ hq, one_mul]
  rw [hck, prod_fou, prod_conj_fou, conj_om_pow d hd, hMd, hom, tv_tshift]
  simp only [← pow_mul, pow_add]
  ring

theorem tensorW_cyc (E : Env) (M : ℕ) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0)
    (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (Q : ℕ → ℕ) (hM : 0 < M) (k : Kind)
    (d ph : ℕ) (hdM : d ∣ M)
    (hφ : k.phased = true → k.noZW = true →
      ∀ j < d, ex (E.phase ph j) = om M ^ (P ph j).1 * mono xs (P ph j).2)
    (hC : k.noZW = false →
      (Q ph : ℂ) ≠ 0 ∧ ∀ j < d, (Q ph : ℂ) * E.coef ph j = tv (om M) xs (C ph j))
    (os is : List ℕ) :
    tensor E k d ph os is = (rt d)⁻¹ ^ sqcW E.shape k d os.length is.length *
      ((Q ph : ℂ)⁻¹) ^ qcW E.shape k * tv (om M) xs (tensorW E.shape M P C k d ph os is) := by
  induction k generalizing os is with
  | ZW =>
    obtain ⟨hq, hc⟩ := hC rfl
    show tZw d (E.coef ph) (os ++ is) = (rt d)⁻¹ ^ 0 * ((Q ph : ℂ)⁻¹) ^ 1 *
      tv (om M) xs (tZWT (C ph) d (os ++ is))
    rw [pow_zero, pow_one, one_mul]
    exact tZw_cyc M d xs (C ph) _ _ hq hc _
  | XW =>
    obtain ⟨hq, hc⟩ := hC rfl
    show tXw d (E.coef ph) os is = (rt d)⁻¹ ^ (os.length + is.length) * ((Q ph : ℂ)⁻¹) ^ 1 *
      tv (om M) xs (tXWT M (C ph) d os is)
    rw [pow_one]
    exact tXw_cyc M d hM hdM xs (C ph) _ _ hq hc os is
  | Bun m b ih =>
    simp only [tensor, tensorW, sqcW, qcW, tv_foldr_tmul _ _ hx, List.map_map, copyOuts,
      copyIns, Env.shape]
    have : ∀ i, tensor E b d ph (copyLegs E.mult d i (b.outCs os.length) os)
        (copyLegs E.mult d i (b.inCs is.length) is) =
        (rt d)⁻¹ ^ sqcW E.shape b d os.length is.length *
          (((Q ph : ℂ)⁻¹) ^ qcW E.shape b *
          tv (om M) xs (tensorW E.shape M P C b d ph (copyLegs E.mult d i (b.outCs os.length) os)
            (copyLegs E.mult d i (b.inCs is.length) is))) := by
      intro i
      rw [← mul_assoc, ih hφ hC, show (copyLegs E.mult d i (b.outCs os.length) os).length =
        os.length from copyOuts_length E b d i os, show (copyLegs E.mult d i (b.inCs is.length)
        is).length = is.length from copyIns_length E b d i is]
    simp only [this, prod_pow_mul, List.length_range, mul_assoc]
    all_goals rfl
  | Z =>
    have h := tensor_cyc E M xs hx P hM .Z rfl d ph hdM (hφ rfl rfl) os is
    rw [show qcW E.shape .Z = 0 from rfl, pow_zero, mul_one]
    exact h
  | X =>
    have h := tensor_cyc E M xs hx P hM .X rfl d ph hdM (hφ rfl rfl) os is
    rw [show qcW E.shape .X = 0 from rfl, pow_zero, mul_one]
    exact h
  | F =>
    rw [show qcW E.shape .F = 0 from rfl, pow_zero, mul_one]
    show tensor E .F d ph os is = (rt d)⁻¹ ^ sqc E.shape .F d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .F d ph os is)
    have hd : 0 < d := Nat.pos_of_dvd_of_pos hdM hM
    obtain ⟨q, hq⟩ := hdM
    have hMd : M / d = q := by rw [hq, Nat.mul_div_cancel_left _ hd]
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [fou, hMd, om_sub M d q hM hq, ← pow_mul, rt, div_eq_mul_inv, mul_comm, mono_nil]
  | T =>
    rw [show qcW E.shape .T = 0 from rfl, pow_zero, mul_one]
    show tensor E .T d ph os is = (rt d)⁻¹ ^ sqc E.shape .T d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .T d ph os is)
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tT, mono_nil]
  | Ti =>
    rw [show qcW E.shape .Ti = 0 from rfl, pow_zero, mul_one]
    show tensor E .Ti d ph os is = (rt d)⁻¹ ^ sqc E.shape .Ti d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .Ti d ph os is)
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tTi, mono_nil]
  | W =>
    rw [show qcW E.shape .W = 0 from rfl, pow_zero, mul_one]
    show tensor E .W d ph os is = (rt d)⁻¹ ^ sqc E.shape .W d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .W d ph os is)
    rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tWZ_eq, mono_nil, Env.shape]
  | B =>
    rw [show qcW E.shape .B = 0 from rfl, pow_zero, mul_one]
    show tensor E .B d ph os is = (rt d)⁻¹ ^ sqc E.shape .B d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .B d ph os is)
    simp [tensor, tensorT, tv]
  | S =>
    rw [show qcW E.shape .S = 0 from rfl, pow_zero, mul_one]
    show tensor E .S d ph os is = (rt d)⁻¹ ^ sqc E.shape .S d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .S d ph os is)
    simp [tensor, tensorT, tv]
  | I =>
    rw [show qcW E.shape .I = 0 from rfl, pow_zero, mul_one]
    show tensor E .I d ph os is = (rt d)⁻¹ ^ sqc E.shape .I d os.length is.length *
      tv (om M) xs (tensorT E.shape M P .I d ph os is)
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tI, mono_nil]
  | ZB cs =>
    have h := tensor_cyc E M xs hx P hM (.ZB cs) rfl d ph hdM (hφ rfl rfl) os is
    rw [show qcW E.shape (.ZB cs) = 0 from rfl, pow_zero, mul_one]
    exact h
  | XB co ci =>
    have h := tensor_cyc E M xs hx P hM (.XB co ci) rfl d ph hdM (hφ rfl rfl) os is
    rw [show qcW E.shape (.XB co ci) = 0 from rfl, pow_zero, mul_one]
    exact h
  | WB cs =>
    rw [show qcW E.shape (.WB cs) = 0 from rfl, pow_zero, mul_one]
    show tensor E (.WB cs) d ph os is = (rt d)⁻¹ ^ sqc E.shape (.WB cs) d os.length is.length *
      tv (om M) xs (tensorT E.shape M P (.WB cs) d ph os is)
    rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tWZ_eq, mono_nil, Env.shape]

/-- A node's entry without its `(√d)⁻¹` and `(Q ph)⁻¹` factors as a formal sum. -/
def valW (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (σ : ℕ → ℕ)
    (n : Node) : Terms :=
  if isBS n then valBS E σ n
  else tensorW E M P C n.kind (E.dimv n.dim) n.ph (n.outs.map σ) (n.ins.map σ)

/-- A node's number of `(√d)⁻¹` factors. -/
def sqNW (E : Shape) (n : Node) : ℕ :=
  if isBS n then 0 else sqcW E n.kind (E.dimv n.dim) n.outs.length n.ins.length

/-- A node's number of `(Q ph)⁻¹` factors. -/
def qNW (E : Shape) (n : Node) : ℕ := if isBS n then 0 else qcW E n.kind

/-- `M` is a multiple of the node's dimension; its phase vector is among the vectors `fs`
known at every index, or `keys` lists its phase entries, or `ckeys` its coefficient entries. -/
def nodeOKW (E : Shape) (M : ℕ) (keys ckeys : List (ℕ × ℕ)) (fs : List ℕ) (n : Node) : Bool :=
  isBS n || (M % E.dimv n.dim == 0 && (!n.kind.phased || (n.kind.noZW && fs.contains n.ph) ||
    (List.range (E.dimv n.dim)).all fun j =>
      if n.kind.noZW then keys.contains (n.ph, j) else ckeys.contains (n.ph, j)))

/-- Each listed coefficient entry is `(Q ph)⁻¹` times its formal sum. -/
def CoefOK (E : Env) (M : ℕ) (xs : List ℂ) (C : ℕ → ℕ → Terms) (Q : ℕ → ℕ) :
    List (ℕ × ℕ) → Prop
  | [] => True
  | e :: l => (Q e.1 : ℂ) ≠ 0 ∧ (Q e.1 : ℂ) * E.coef e.1 e.2 = tv (om M) xs (C e.1 e.2) ∧
      CoefOK E M xs C Q l

theorem CoefOK.get {E : Env} {M : ℕ} {xs : List ℂ} {C : ℕ → ℕ → Terms} {Q : ℕ → ℕ}
    {keys : List (ℕ × ℕ)} (h : CoefOK E M xs C Q keys) {e : ℕ × ℕ} (he : e ∈ keys) :
    (Q e.1 : ℂ) ≠ 0 ∧ (Q e.1 : ℂ) * E.coef e.1 e.2 = tv (om M) xs (C e.1 e.2) := by
  induction keys with
  | nil => simp at he
  | cons a l ih =>
    rcases List.mem_cons.1 he with rfl | he
    · exact ⟨h.1, h.2.1⟩
    · exact ih h.2.2 he

theorem val_cycW (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (C : ℕ → ℕ → Terms) (Q : ℕ → ℕ) (keys ckeys : List (ℕ × ℕ)) (fs : List ℕ) (hM : 0 < M)
    (hph : PhOK E M gs P keys) (hco : CoefOK E M (gs.map ex) C Q ckeys)
    (hf : ∀ i ∈ fs, ∀ k, ex (E.phase i k) = om M ^ (P i k).1 * mono (gs.map ex) (P i k).2)
    (σ : ℕ → ℕ) (n : Node) (hok : nodeOKW E.shape M keys ckeys fs n = true) :
    val E σ n = (rt (E.dimv n.dim))⁻¹ ^ sqNW E.shape n * ((Q n.ph : ℂ)⁻¹) ^ qNW E.shape n *
      tv (om M) (gs.map ex) (valW E.shape M P C σ n) := by
  obtain ⟨kind, outs, ins, dim, ph⟩ := n
  by_cases hb : kind = .B ∨ kind = .S
  · have : isBS ⟨kind, outs, ins, dim, ph⟩ = true := by
      rcases hb with rfl | rfl <;> rfl
    simp only [valW, sqNW, qNW, this, ite_true, pow_zero, one_mul]
    dsimp only [Env.shape]
    rcases hb with rfl | rfl <;> simp only [val, valBS, Env.shape] <;>
      generalize outs.map σ = A <;> generalize ins.map σ = B <;>
      rcases A with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;>
      rcases B with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;>
      simp [tv, tc, tB, mono_nil, Int.cast_ite, Env.shape]
  · have hb' : kind ≠ .B ∧ kind ≠ .S := by tauto
    have hn : isBS ⟨kind, outs, ins, dim, ph⟩ = false := by
      simp [isBS, hb'.1, hb'.2]
    simp only [nodeOKW, hn, Bool.false_or, Bool.and_eq_true, beq_iff_eq, Env.shape] at hok
    obtain ⟨hmod, hkeys'⟩ := hok
    have hkeys0 : kind.phased = true → (kind.noZW = true ∧ ph ∈ fs) ∨ ∀ j < E.dimv dim,
        (if kind.noZW = true then keys.contains (ph, j) else ckeys.contains (ph, j)) = true := by
      intro hph'
      simp only [hph', Bool.not_true, Bool.false_or, Bool.or_eq_true, Bool.and_eq_true,
        List.contains_iff_mem, List.all_eq_true, List.mem_range] at hkeys'
      exact hkeys'
    have hd : 0 < E.dimv dim := by
      rcases Nat.eq_zero_or_pos (E.dimv dim) with h0 | h0
      · rw [h0, Nat.mod_zero] at hmod; omega
      · exact h0
    have hp : kind.phased = true → kind.noZW = true → ∀ j < E.dimv dim,
        ex (E.phase ph j) = om M ^ (P ph j).1 * mono (gs.map ex) (P ph j).2 := by
      intro hph' hk j hj
      rcases hkeys0 hph' with ⟨_, hzs⟩ | hall
      · exact hf ph hzs j
      have := hall j hj
      rw [if_pos hk, List.contains_iff_mem] at this
      exact hph.get (e := (ph, j)) this
    have hc : kind.noZW = false → (Q ph : ℂ) ≠ 0 ∧ ∀ j < E.dimv dim,
        (Q ph : ℂ) * E.coef ph j = tv (om M) (gs.map ex) (C ph j) := by
      intro hk
      have key : ∀ j < E.dimv dim, (ph, j) ∈ ckeys := by
        intro j hj
        rcases hkeys0 (Kind.phased_of_noZW kind hk) with ⟨h1, _⟩ | hall
        · rw [hk] at h1; exact absurd h1 (by simp)
        have := hall j hj
        rw [if_neg (by simp [hk]), List.contains_iff_mem] at this
        exact this
      exact ⟨(hco.get (e := (ph, 0)) (key 0 hd)).1, fun j hj => (hco.get (e := (ph, j)) (key j hj)).2⟩
    rw [val_tensor E σ kind hb', tensorW_cyc E M (gs.map ex) (ex_atoms_ne gs) P C Q hM kind _ ph
      (Nat.dvd_of_mod_eq_zero hmod) hp hc]
    simp [valW, sqNW, qNW, hn, Env.shape]

/-! ## Formal entries -/

/-- The product of the node entries as one formal sum. -/
def prodW (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (ns : List Node)
    (σ : ℕ → ℕ) : Terms :=
  ns.foldr (fun n acc => tmul (valW E M P C σ n) acc) (tc 1)

/-- The sum over the labels `ls` of formal sums. -/
def sumLT (D : ℕ → ℕ) : List ℕ → ((ℕ → ℕ) → Terms) → (ℕ → ℕ) → Terms
  | [], f, σ => f σ
  | l :: ls, f, σ => (List.range (D l)).flatMap fun j => sumLT D ls f (Function.update σ l j)

/-- Each node's dimension, number of `(√d)⁻¹` factors, denominator and number of
denominator factors. -/
def sqLW (E : Shape) (Q : ℕ → ℕ) (N : Net) : List (ℕ × ℕ × ℕ × ℕ) :=
  N.nodes.map fun n => (E.dimv n.dim, sqNW E n, Q n.ph, qNW E n)

/-- The product of a network's `(√d)⁻¹` and denominator factors. -/
noncomputable def KW (E : Shape) (Q : ℕ → ℕ) (N : Net) : ℂ :=
  ((sqLW E Q N).map fun t => (rt t.1)⁻¹ ^ t.2.1 * ((t.2.2.1 : ℂ))⁻¹ ^ t.2.2.2).prod

theorem prodVal_W (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (C : ℕ → ℕ → Terms) (Q : ℕ → ℕ) (keys ckeys : List (ℕ × ℕ)) (fs : List ℕ) (hM : 0 < M)
    (hph : PhOK E M gs P keys) (hco : CoefOK E M (gs.map ex) C Q ckeys)
    (hf : ∀ i ∈ fs, ∀ k, ex (E.phase i k) = om M ^ (P i k).1 * mono (gs.map ex) (P i k).2)
    (ns : List Node) (hok : ns.all (nodeOKW E.shape M keys ckeys fs) = true) (σ : ℕ → ℕ) :
    prodVal E ns σ = (ns.map fun n => (rt (E.dimv n.dim))⁻¹ ^ sqNW E.shape n *
      ((Q n.ph : ℂ)⁻¹) ^ qNW E.shape n).prod * tv (om M) (gs.map ex) (prodW E.shape M P C ns σ) := by
  induction ns with
  | nil => simp [prodVal, prodW, tv_tc]
  | cons n ns ih =>
    simp only [List.all_cons, Bool.and_eq_true] at hok
    have := ih hok.2
    simp only [prodVal, List.map_cons, List.prod_cons, prodW, List.foldr_cons] at this ⊢
    rw [tv_mul _ _ (ex_atoms_ne gs), val_cycW E M gs P C Q keys ckeys fs hM hph hco hf σ n hok.1, this]
    ring

theorem sumOver_sumLT (M : ℕ) (xs : List ℂ) (D : ℕ → ℕ) (K : ℂ) (g : (ℕ → ℕ) → ℂ)
    (f : (ℕ → ℕ) → Terms) (hf : ∀ τ, g τ = K * tv (om M) xs (f τ)) (ls : List ℕ)
    (σ : ℕ → ℕ) : sumOver D ls g σ = K * tv (om M) xs (sumLT D ls f σ) := by
  induction ls generalizing σ with
  | nil => exact hf σ
  | cons l ls ih =>
    simp only [sumOver, sumLT, tv_flatMap, fsum_list, ih, List.sum_map_mul_left]

/-- A network's entry as `KW` times a formal sum. -/
theorem eval_W (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (Q : ℕ → ℕ) (keys ckeys : List (ℕ × ℕ)) (fs : List ℕ) (N : Net) (σ : ℕ → ℕ) (hM : 0 < M)
    (hok : N.nodes.all (nodeOKW E.shape M keys ckeys fs) = true) (hph : PhOK E M gs P keys)
    (hco : CoefOK E M (gs.map ex) C Q ckeys)
    (hf : ∀ i ∈ fs, ∀ k, ex (E.phase i k) = om M ^ (P i k).1 * mono (gs.map ex) (P i k).2) :
    eval E N σ = KW E.shape Q N *
      tv (om M) (gs.map ex) (sumLT E.shape.dim N.int (prodW E.shape M P C N.nodes) σ) := by
  unfold eval KW
  have : ((sqLW E.shape Q N).map fun t => (rt t.1)⁻¹ ^ t.2.1 * ((t.2.2.1 : ℂ))⁻¹ ^ t.2.2.2) =
      N.nodes.map fun n => (rt (E.dimv n.dim))⁻¹ ^ sqNW E.shape n *
        ((Q n.ph : ℂ)⁻¹) ^ qNW E.shape n := by simp [sqLW, Env.shape]
  rw [this, show E.shape.dim = labDim E from rfl]
  exact sumOver_sumLT M _ _ _ _ _ (prodVal_W E M gs P C Q keys ckeys fs hM hph hco hf N.nodes hok) _ _

/-- A phase `n / den` in turns as a power of `om M` when `M = den q`. -/
theorem ex_tab (M den q : ℕ) (hM : 0 < M) (hq : M = den * q) (n : ℤ) (dr : ℝ)
    (hdr : dr = den) : ex ((n : ℝ) / dr) = om M ^ ((n * q) % (M : ℤ)).toNat := by
  have hd : (den : ℝ) ≠ 0 := by
    rintro h
    have : den = 0 := by exact_mod_cast h
    subst this; simp at hq; omega
  have hMr : (M : ℝ) = den * q := by exact_mod_cast hq
  have hM0 : (M : ℤ) ≠ 0 := by exact_mod_cast hM.ne'
  have hnn : 0 ≤ n * q % (M : ℤ) := Int.emod_nonneg _ hM0
  have h2 : (n * q : ℤ) = M * (n * q / M) + n * q % M := by
    exact (Int.mul_ediv_add_emod (n * q) M).symm
  have h2r : (n : ℝ) * q = (M : ℝ) * ((n * q / M : ℤ) : ℝ) + ((n * q % M : ℤ) : ℝ) := by
    exact_mod_cast h2
  apply ex_of M ((n * q) % (M : ℤ)).toNat (n * q / M) hM
  rw [show (((n * q % (M : ℤ)).toNat : ℕ) : ℝ) = ((n * q % (M : ℤ) : ℤ) : ℝ) by
    rw [← Int.cast_natCast (R := ℝ), Int.toNat_of_nonneg hnn], hdr]
  have hMne : (M : ℝ) ≠ 0 := by exact_mod_cast hM.ne'
  field_simp
  rw [hMr] at h2r ⊢
  linear_combination (den : ℝ) * h2r

/-! ## Residues -/

/-- A formal sum at `ζ ↦ w` in `ZMod p`, atoms read as `1`. -/
def tmv (p w : ℕ) (t : Terms) : ZMod p :=
  (t.map fun q => (q.1 : ZMod p) * (w : ZMod p) ^ q.2.1).sum

theorem tmv_append (p w : ℕ) (a b : Terms) : tmv p w (a ++ b) = tmv p w a + tmv p w b := by
  simp [tmv]

theorem tmv_flatMap {α : Type*} (p w : ℕ) (l : List α) (f : α → Terms) :
    tmv p w (l.flatMap f) = (l.map fun x => tmv p w (f x)).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatMap_cons, tmv_append, ih, List.map_cons, List.sum_cons]

theorem tmv_tc (p w : ℕ) (c : ℤ) : tmv p w (tc c) = c := by simp [tmv, tc]

theorem tmv_mul (p w : ℕ) (a b : Terms) : tmv p w (tmul a b) = tmv p w a * tmv p w b := by
  induction a with
  | nil => simp [tmul, tmv]
  | cons q a ih =>
    have : tmul (q :: a) b =
        (b.map fun r => (q.1 * r.1, q.2.1 + r.2.1, kadd q.2.2 r.2.2)) ++ tmul a b := by
      simp [tmul, List.flatMap_cons]
    rw [this, tmv_append, ih]
    simp only [tmv, List.map_cons, List.sum_cons, List.map_map, add_mul]
    congr 1
    rw [← List.sum_map_mul_left]
    congr 1
    apply List.map_congr_left
    intro r _
    simp only [Function.comp, Int.cast_mul, pow_add]
    ring

/-- A formal sum as an integer polynomial, atoms read as `1`. -/
noncomputable def tpoly (t : Terms) : ℤ[X] := (t.map fun q => C q.1 * X ^ q.2.1).sum

theorem mono_nil_left (ks : List ℤ) : mono [] ks = 1 := by cases ks <;> rfl

theorem aeval_tpoly (z : ℂ) (t : Terms) : aeval z (tpoly t) = tv z [] t := by
  induction t with
  | nil => simp [tpoly, tv]
  | cons q t ih =>
    simp only [tpoly, List.map_cons, List.sum_cons] at ih ⊢
    rw [map_add, ih, tv_cons, mono_nil_left]
    simp

theorem eval₂_tpoly (p w : ℕ) (t : Terms) :
    (tpoly t).eval₂ (Int.castRingHom (ZMod p)) (w : ZMod p) = tmv p w t := by
  induction t with
  | nil => simp [tpoly, tmv]
  | cons q t ih =>
    simp only [tpoly, tmv, List.map_cons, List.sum_cons] at ih ⊢
    rw [eval₂_add, ih]
    simp

/-- A formal sum vanishing at `om M` vanishes mod `p` at an element of order `M`. -/
theorem tmv_eq_zero (p w M : ℕ) [Fact p.Prime] (hM : 0 < M)
    (hw : IsPrimitiveRoot (w : ZMod p) M) (t : Terms) (h : tv (om M) [] t = 0) :
    tmv p w t = 0 := by
  have hp := om_prim M hM
  have hroot : aeval (om M) (tpoly t) = 0 := by rw [aeval_tpoly]; exact h
  have hdvd := minpoly.isIntegrallyClosed_dvd (hp.isIntegral hM) hroot
  rw [← cyclotomic_eq_minpoly hp hM] at hdvd
  obtain ⟨g, hg⟩ := hdvd
  rw [← eval₂_tpoly, hg, eval₂_mul, eval₂_eq_eval_map, map_cyclotomic,
    (hw.isRoot_cyclotomic hM).eq_zero, zero_mul]

/-- `b ^ e % p` by squaring, `k` bits of `e`. -/
def pm (p : ℕ) : ℕ → ℕ → ℕ → ℕ
  | 0, _, _ => 1 % p
  | k + 1, b, e =>
    if e = 0 then 1 % p
    else if e % 2 = 1 then pm p k (b * b % p) (e / 2) * b % p else pm p k (b * b % p) (e / 2)

theorem pm_eq (p : ℕ) : ∀ (k b e : ℕ), e < 2 ^ k → pm p k b e = b ^ e % p := by
  intro k
  induction k with
  | zero => intro b e h; simp at h; subst h; simp [pm]
  | succ k ih =>
    intro b e h
    simp only [pm]
    split_ifs with h0 h1
    · subst h0; simp
    · have hlt : e / 2 < 2 ^ k := by rw [pow_succ] at h; omega
      rw [ih _ _ hlt, ← Nat.pow_mod, Nat.mod_mul_mod]
      congr 1
      conv_rhs => rw [show e = 2 * (e / 2) + 1 by omega]
      ring
    · have hlt : e / 2 < 2 ^ k := by rw [pow_succ] at h; omega
      rw [ih _ _ hlt, ← Nat.pow_mod]
      congr 1
      conv_rhs => rw [show e = 2 * (e / 2) by omega]
      ring

theorem prime_mem_of_dvd (q : ℕ) (hq : q.Prime) :
    ∀ (ps : List (ℕ × ℕ)), (∀ x ∈ ps, x.1.Prime) → q ∣ (ps.map fun x => x.1 ^ x.2).prod →
      ∃ x ∈ ps, x.1 = q := by
  intro ps
  induction ps with
  | nil => intro _ h; simp at h; exact absurd h hq.ne_one
  | cons a ps ih =>
    intro hps h
    simp only [List.map_cons, List.prod_cons] at h
    rcases (Nat.Prime.dvd_mul hq).1 h with h1 | h1
    · have := (Nat.prime_dvd_prime_iff_eq hq (hps a (by simp))).1 (hq.dvd_of_dvd_pow h1)
      exact ⟨a, by simp, this.symm⟩
    · obtain ⟨x, hx, e⟩ := ih (fun x hx => hps x (by simp [hx])) h1
      exact ⟨x, by simp [hx], e⟩

/-- `w` has order `M` mod the prime `p`, from the prime factors `ps` of `M`. -/
theorem prim_of (p w M : ℕ) [Fact p.Prime] (hM : 0 < M) (hMb : M < 2 ^ 64)
    (ps : List (ℕ × ℕ)) (hfac : (ps.map fun x => x.1 ^ x.2).prod = M)
    (hps : ∀ x ∈ ps, x.1.Prime) (h1 : pm p 64 w M = 1)
    (h2 : ps.all (fun x => pm p 64 w (M / x.1) != 1) = true) :
    IsPrimitiveRoot (w : ZMod p) M := by
  have hp1 : 1 < p := (Fact.out : p.Prime).one_lt
  have key : ∀ l, l < 2 ^ 64 → ((w : ZMod p) ^ l = 1 ↔ pm p 64 w l = 1) := by
    intro l hl
    rw [pm_eq p 64 w l hl, ← Nat.cast_pow, ← Nat.cast_one, ZMod.natCast_eq_natCast_iff',
      Nat.mod_eq_of_lt hp1]
  have ho : orderOf (w : ZMod p) = M := by
    refine orderOf_eq_of_pow_and_pow_div_prime hM ((key M hMb).2 h1) ?_
    intro q hq hqM
    obtain ⟨x, hx, rfl⟩ := prime_mem_of_dvd q hq ps hps (hfac ▸ hqM)
    have := List.all_eq_true.1 h2 x hx
    simp only [bne_iff_ne, ne_eq] at this
    exact fun h => this ((key _ (lt_of_le_of_lt (Nat.div_le_self _ _) hMb)).1 h)
  rw [← ho]
  exact IsPrimitiveRoot.orderOf _

theorem pow_eq_one_of (p w M : ℕ) (h1 : w ^ M % p = 1) : (w : ZMod p) ^ M = 1 := by
  rw [← Nat.cast_pow, ← ZMod.natCast_mod, h1, Nat.cast_one]

/-- A formal sum at `ζ ↦ w` mod `p`, atoms read as `1`. -/
def tmN (p w M : ℕ) (t : Terms) : ℕ :=
  t.foldr (fun q acc => ((q.1 % (p : ℤ)).toNat * pm p 64 w (q.2.1 % M) + acc) % p) 0

theorem tmN_cast (p w M : ℕ) (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M)
    (hMb : M ≤ 2 ^ 64) (t : Terms) : (tmN p w M t : ZMod p) = tmv p w t := by
  induction t with
  | nil => simp [tmN, tmv]
  | cons q t ih =>
    simp only [tmN, List.foldr_cons] at ih ⊢
    rw [pm_eq p 64 w _ (lt_of_lt_of_le (Nat.mod_lt _ hM0) hMb)]
    simp only [ZMod.natCast_mod, Nat.cast_add, Nat.cast_mul, Nat.cast_pow]
    rw [ih]
    have h1 : (((q.1 % (p : ℤ)).toNat : ℕ) : ZMod p) = (q.1 : ZMod p) := by
      have hnn : 0 ≤ q.1 % (p : ℤ) := Int.emod_nonneg _ (by exact_mod_cast hp)
      rw [← Int.cast_natCast (R := ZMod p), Int.toNat_of_nonneg hnn, ZMod.intCast_mod]
    have h2 : (w : ZMod p) ^ (q.2.1 % M) = (w : ZMod p) ^ q.2.1 := by
      conv_rhs => rw [← Nat.div_add_mod q.2.1 M, pow_add, pow_mul, hw, one_pow, one_mul]
    rw [h1, h2]
    simp [tmv]

/-! ## Sums over labels in a commutative ring -/

section sums

variable {R : Type*} [CommRing R] (D : ℕ → ℕ)

/-- The sum over the labels `ls`, each over its dimension, of `f`. -/
def sumR : List ℕ → ((ℕ → ℕ) → R) → (ℕ → ℕ) → R
  | [], f, σ => f σ
  | l :: ls, f, σ => ∑ j ∈ range (D l), sumR ls f (Function.update σ l j)

theorem sumR_congr (ls : List ℕ) (f g : (ℕ → ℕ) → R) (σ : ℕ → ℕ)
    (h : ∀ τ, (∀ x ∉ ls, τ x = σ x) → (∀ x ∈ ls, τ x < D x) → f τ = g τ) :
    sumR D ls f σ = sumR D ls g σ := by
  induction ls generalizing σ with
  | nil => exact h σ (fun _ _ => rfl) (by simp)
  | cons l ls ih =>
    refine Finset.sum_congr rfl fun j hj => ih _ fun τ hτ hr => h τ ?_ ?_
    · intro x hx
      rw [hτ x (fun h' => hx (List.mem_cons_of_mem _ h'))]
      exact Function.update_of_ne (by rintro rfl; exact hx List.mem_cons_self) _ _
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx'
      · by_cases hm : x ∈ ls
        · exact hr x hm
        · rw [hτ x hm, Function.update_self]; exact Finset.mem_range.1 hj
      · exact hr x hx'

theorem sumR_append (a b : List ℕ) (f : (ℕ → ℕ) → R) (σ : ℕ → ℕ) :
    sumR D (a ++ b) f σ = sumR D a (sumR D b f) σ := by
  induction a generalizing σ with
  | nil => rfl
  | cons l a ih => exact Finset.sum_congr rfl fun j _ => ih _

theorem sumR_perm {a b : List ℕ} (h : a.Perm b) (f : (ℕ → ℕ) → R) (σ : ℕ → ℕ) :
    sumR D a f σ = sumR D b f σ := by
  induction h generalizing σ with
  | nil => rfl
  | cons l _ ih => exact Finset.sum_congr rfl fun j _ => ih _
  | swap x y l =>
    simp only [sumR]
    by_cases hxy : x = y
    · subst hxy; rfl
    · rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
      rw [Function.update_comm hxy]
  | trans _ _ ih₁ ih₂ => exact (ih₁ σ).trans (ih₂ σ)

end sums

theorem fsum_listR {R : Type*} [AddCommMonoid R] (f : ℕ → R) (d : ℕ) :
    ∑ k ∈ range d, f k = ((List.range d).map f).sum := by
  induction d with
  | zero => simp
  | succ d ih => rw [sum_range_succ, ih, List.range_succ, List.map_append, List.sum_append]; simp

theorem tmv_sumLT (p w : ℕ) (D : ℕ → ℕ) (f : (ℕ → ℕ) → Terms) (ls : List ℕ) (σ : ℕ → ℕ) :
    tmv p w (sumLT D ls f σ) = sumR D ls (fun τ => tmv p w (f τ)) σ := by
  induction ls generalizing σ with
  | nil => rfl
  | cons l ls ih => simp only [sumLT, sumR, tmv_flatMap, ih, fsum_listR]

theorem tmv_prodW (p w : ℕ) (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (ns : List Node) (σ : ℕ → ℕ) :
    tmv p w (prodW E M P C ns σ) = (ns.map fun n => tmv p w (valW E M P C σ n)).prod := by
  induction ns with
  | nil => simp [prodW, tmv_tc]
  | cons n ns ih =>
    simp only [prodW, List.foldr_cons, List.map_cons, List.prod_cons] at ih ⊢
    rw [tmv_mul, ih]

/-! ## Residue factors -/

/-- A table of residues, nested one level per label. -/
inductive RT where
  | leaf : ℕ → RT
  | nil : RT
  | cons : RT → RT → RT
  deriving DecidableEq, Repr

/-- The level of subtables `cs`. -/
def RT.node (cs : List RT) : RT := cs.foldr RT.cons RT.nil

/-- The `k`-th subtable of a level; a leaf is its own subtable. -/
def RT.child : RT → ℕ → RT
  | .cons h _, 0 => h
  | .cons _ t, k + 1 => RT.child t k
  | .leaf v, _ => .leaf v
  | .nil, _ => .leaf 0

/-- The entry of a table at `σ`, its levels read at the labels `ls` in order. -/
def RT.get (σ : ℕ → ℕ) : List ℕ → RT → ℕ
  | [], .leaf v => v
  | [], _ => 0
  | l :: ls, T => RT.get σ ls (T.child (σ l))

/-- A scope of labels and a residue table with one level per label of it. -/
structure RFac where
  scope : List ℕ
  table : RT
  deriving DecidableEq, Repr

/-- A residue factor's entry at an assignment. -/
def RFac.at (F : RFac) (σ : ℕ → ℕ) : ℕ := F.table.get σ F.scope

/-- The product of factor entries in `ZMod p`. -/
def rprod (p : ℕ) (Fs : List RFac) (σ : ℕ → ℕ) : ZMod p := (Fs.map fun F => (F.at σ : ZMod p)).prod

/-- The product of factor entries mod `p`. -/
def rfold (p : ℕ) (Fs : List RFac) (σ : ℕ → ℕ) : ℕ := Fs.foldr (fun F acc => F.at σ * acc % p) 1

theorem rfold_cast (p : ℕ) (Fs : List RFac) (σ : ℕ → ℕ) : (rfold p Fs σ : ZMod p) = rprod p Fs σ := by
  induction Fs with
  | nil => simp [rfold, rprod]
  | cons F Fs ih =>
    simp only [rfold, rprod, List.foldr_cons, List.map_cons, List.prod_cons] at ih ⊢
    rw [ZMod.natCast_mod, Nat.cast_mul, ih]

theorem RT.get_congr (σ τ : ℕ → ℕ) : ∀ (ls : List ℕ) (T : RT), (∀ l ∈ ls, σ l = τ l) →
    T.get σ ls = T.get τ ls := by
  intro ls
  induction ls with
  | nil => intro T _; cases T <;> rfl
  | cons l ls ih =>
    intro T h
    simp only [RT.get, h l List.mem_cons_self]
    exact ih _ fun x hx => h x (List.mem_cons_of_mem _ hx)

theorem RFac.at_congr (F : RFac) (σ τ : ℕ → ℕ) (h : ∀ l ∈ F.scope, σ l = τ l) :
    F.at σ = F.at τ := RT.get_congr σ τ F.scope F.table h

theorem rprod_congr (p : ℕ) (Fs : List RFac) (σ τ : ℕ → ℕ)
    (h : ∀ F ∈ Fs, ∀ l ∈ F.scope, σ l = τ l) : rprod p Fs σ = rprod p Fs τ := by
  unfold rprod
  congr 1
  exact List.map_congr_left fun F hF => by rw [F.at_congr σ τ (h F hF)]

theorem rprod_filter (p : ℕ) (q : RFac → Bool) (Fs : List RFac) (σ : ℕ → ℕ) :
    rprod p Fs σ = rprod p (Fs.filter fun F => !q F) σ * rprod p (Fs.filter q) σ := by
  induction Fs with
  | nil => simp [rprod]
  | cons F Fs ih =>
    simp only [rprod, List.filter_cons] at ih ⊢
    cases q F <;> simp [ih] <;> ring

/-- A node's factor holds its residues with its labels in `U` read from `σ`, and covers its
other labels. -/
def leafR (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (U : List ℕ) (σ : ℕ → ℕ) (n : Node) (F : RFac) : Bool :=
  (n.outs ++ n.ins).all (fun x => F.scope.contains x || U.contains x) &&
    (assigns S.dim F.scope).all fun τ => F.at τ == tmN p w M (valW S M P C (pin U σ τ) n)

theorem valW_congr (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (n : Node)
    (σ τ : ℕ → ℕ) (h : ∀ l ∈ n.outs ++ n.ins, σ l = τ l) : valW S M P C σ n = valW S M P C τ n := by
  have ho : n.outs.map σ = n.outs.map τ :=
    List.map_congr_left fun l hl => h l (List.mem_append_left _ hl)
  have hi : n.ins.map σ = n.ins.map τ :=
    List.map_congr_left fun l hl => h l (List.mem_append_right _ hl)
  simp only [valW, valBS, ho, hi]

theorem leafR_val (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M) (hMb : M ≤ 2 ^ 64) (U : List ℕ)
    (σ : ℕ → ℕ) (n : Node) (F : RFac)
    (h : leafR S M P C p w U σ n F = true) (τ : ℕ → ℕ) (hτ : ∀ l ∈ F.scope, τ l < S.dim l)
    (hU : ∀ l ∈ U, τ l = σ l) : tmv p w (valW S M P C τ n) = (F.at τ : ZMod p) := by
  simp only [leafR, Bool.and_eq_true, List.all_eq_true, List.contains_iff_mem,
    beq_iff_eq, Bool.or_eq_true] at h
  obtain ⟨ρ, hρ, hρτ⟩ := assigns_cover S.dim F.scope τ hτ
  rw [F.at_congr τ ρ fun l hl => (hρτ l hl).symm, h.2 ρ hρ, tmN_cast p w M hp hw hM0 hMb]
  congr 1
  refine valW_congr S M P C n τ (pin U σ ρ) fun l hl => ?_
  unfold pin
  by_cases hu : l ∈ U
  · simp [hu, hU l hu]
  · have hF : l ∈ F.scope := (h.1 l hl).resolve_right hu
    simp [hu, hρτ l hF]

/-- Summing `l` out of the factors of `Fs` containing it gives `new`. -/
def elimR (p : ℕ) (D : ℕ → ℕ) (l : ℕ) (Fs : List RFac) (new : RFac) : Bool :=
  let ins := Fs.filter fun F => F.scope.contains l
  !new.scope.contains l &&
    (new.scope.all fun x => ins.any fun F => F.scope.contains x) &&
    (ins.all fun F => F.scope.all fun x => x == l || new.scope.contains x) &&
    (assigns D new.scope).all fun τ => new.at τ ==
      ((List.range (D l)).map fun j => rfold p ins (Function.update τ l j)).sum % p

/-- The factors of `Fs` without `l`, then `new`. -/
def elimStepR (l : ℕ) (Fs : List RFac) (new : RFac) : List RFac :=
  Fs.filter (fun F => !F.scope.contains l) ++ [new]

theorem elimR_val (p : ℕ) (D : ℕ → ℕ) (l : ℕ) (Fs : List RFac) (new : RFac)
    (h : elimR p D l Fs new = true) (σ : ℕ → ℕ) (hσ : ∀ x ∈ new.scope, σ x < D x) :
    ∑ j ∈ range (D l), rprod p Fs (Function.update σ l j) = rprod p (elimStepR l Fs new) σ := by
  unfold elimR at h
  simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', List.all_eq_true,
    List.any_eq_true, Bool.or_eq_true, List.contains_iff_mem] at h
  obtain ⟨⟨⟨h2, h3⟩, h4⟩, h5⟩ := h
  set ins := Fs.filter fun F => F.scope.contains l
  set outs := Fs.filter fun F => !F.scope.contains l
  have hnl : l ∉ new.scope := by simpa using h2
  have hout : ∀ j, rprod p outs (Function.update σ l j) = rprod p outs σ := by
    intro j
    refine rprod_congr _ _ _ _ fun F hF x hxF => ?_
    have : l ∉ F.scope := by
      have := (List.mem_filter.1 hF).2; simpa using this
    have hxl : x ≠ l := fun e => this (e ▸ hxF)
    exact Function.update_of_ne hxl _ _
  have split : ∀ τ, rprod p Fs τ = rprod p outs τ * rprod p ins τ := by
    intro τ
    have := rprod_filter p (fun F => F.scope.contains l) Fs τ
    simpa [outs, ins] using this
  simp only [split, hout, ← Finset.mul_sum]
  unfold elimStepR
  simp only [rprod, List.map_append, List.prod_append, List.map_cons, List.map_nil,
    List.prod_cons, List.prod_nil, mul_one]
  congr 1
  obtain ⟨τ, hτ, hτσ⟩ := assigns_cover D new.scope σ hσ
  rw [new.at_congr σ τ fun x hx => (hτσ x hx).symm, h5 τ hτ, ZMod.natCast_mod,
    Nat.cast_list_sum, List.map_map, ← fsum_listR]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [Function.comp, rfold_cast]
  refine rprod_congr _ _ _ _ fun F hF x hxF => ?_
  rcases h4 F hF x hxF with e | e
  · subst e; simp
  · have hxl : x ≠ l := fun e' => hnl (e' ▸ e)
    rw [Function.update_of_ne hxl, Function.update_of_ne hxl]
    exact (hτσ x e).symm

/-- A chain of eliminations from `Fs` ending at `final`, each step its label and new factor. -/
def chainR (p : ℕ) (D : ℕ → ℕ) : List RFac → List (ℕ × RFac) → List RFac → Bool
  | Fs, [], final => Fs == final
  | Fs, (l, new) :: st, final => elimR p D l Fs new && chainR p D (elimStepR l Fs new) st final

theorem chainR_nil (p : ℕ) (D : ℕ → ℕ) (Fs : List RFac) : chainR p D Fs [] Fs = true := by
  simp [chainR]

theorem chainR_step (p : ℕ) (D : ℕ → ℕ) (l : ℕ) (Fs Fs' : List RFac) (new : RFac)
    (st : List (ℕ × RFac)) (final : List RFac) (h1 : elimR p D l Fs new = true)
    (hs : elimStepR l Fs new = Fs') (h2 : chainR p D Fs' st final = true) :
    chainR p D Fs ((l, new) :: st) final = true := by
  subst hs
  simp only [chainR, h1, h2, Bool.and_self]

/-- Every label of every factor lies in `U` or is eliminated later. -/
def scopesR (U : List ℕ) : List RFac → List (ℕ × RFac) → Bool
  | Fs, [] => Fs.all fun F => F.scope.all (U.contains ·)
  | Fs, (l, new) :: st =>
    (Fs.all fun F => F.scope.all fun x => U.contains x || x == l ||
      (st.map Prod.fst).contains x) && scopesR U (elimStepR l Fs new) st

theorem scopesR_leaves (U : List ℕ) (Fs : List RFac) (st : List (ℕ × RFac))
    (h : scopesR U Fs st = true) :
    ∀ F ∈ Fs, ∀ x ∈ F.scope, x ∈ U ∨ x ∈ st.map Prod.fst := by
  cases st with
  | nil =>
    simp only [scopesR, List.all_eq_true, List.contains_iff_mem] at h
    exact fun F hF x hx => Or.inl (h F hF x hx)
  | cons s rest =>
    obtain ⟨l, new⟩ := s
    simp only [scopesR, Bool.and_eq_true, List.all_eq_true, Bool.or_eq_true,
      List.contains_iff_mem, beq_iff_eq] at h
    intro F hF x hx
    rcases h.1 F hF x hx with (h1 | h1) | h1
    · exact Or.inl h1
    · exact Or.inr (by simp [h1])
    · exact Or.inr (by simp [h1])

theorem chainR_val (p : ℕ) (D : ℕ → ℕ) (U : List ℕ) : ∀ (st : List (ℕ × RFac)) (Fs final : List RFac),
    chainR p D Fs st final = true → scopesR U Fs st = true →
    ∀ σ, (∀ x ∈ U, σ x < D x) →
      sumR D (st.map Prod.fst).reverse (rprod p Fs) σ = rprod p final σ := by
  intro st
  induction st with
  | nil =>
    intro Fs final h _ σ _
    simp only [chainR, beq_iff_eq] at h
    subst h; rfl
  | cons s st ih =>
    obtain ⟨l, new⟩ := s
    intro Fs final h hs σ hσ
    simp only [chainR, Bool.and_eq_true] at h
    have hs' := hs
    simp only [scopesR, Bool.and_eq_true] at hs'
    simp only [List.map_cons, List.reverse_cons]
    rw [sumR_append]
    have key := scopesR_leaves U (elimStepR l Fs new) st hs'.2
    have inner : ∀ τ, (∀ x ∈ U, τ x < D x) → (∀ x ∈ st.map Prod.fst, τ x < D x) →
        sumR D [l] (rprod p Fs) τ = rprod p (elimStepR l Fs new) τ := by
      intro τ hU hS
      simp only [sumR]
      apply elimR_val p D l Fs new h.1 τ
      intro x hx
      rcases key new (by simp [elimStepR]) x hx with hu | hs
      · exact hU x hu
      · exact hS x hs
    calc sumR D (st.map Prod.fst).reverse (sumR D [l] (rprod p Fs)) σ
        = sumR D (st.map Prod.fst).reverse (rprod p (elimStepR l Fs new)) σ := by
          apply sumR_congr
          intro τ hτ hr
          apply inner τ
          · intro x hxU
            by_cases hm : x ∈ (st.map Prod.fst).reverse
            · exact hr x hm
            · rw [hτ x hm]; exact hσ x hxU
          · intro x hx; exact hr x (by simpa using hx)
      _ = rprod p final σ := ih _ final h.2 hs'.2 σ hσ

theorem prod_leavesR (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M) (hMb : M ≤ 2 ^ 64) (U : List ℕ)
    (σ : ℕ → ℕ) :
    ∀ (ns : List Node) (leaves : List RFac), leaves.length = ns.length →
      (ns.zip leaves).all (fun q => leafR S M P C p w U σ q.1 q.2) = true →
      ∀ τ, (∀ F ∈ leaves, ∀ x ∈ F.scope, τ x < S.dim x) → (∀ x ∈ U, τ x = σ x) →
        (ns.map fun n => tmv p w (valW S M P C τ n)).prod = rprod p leaves τ := by
  intro ns
  induction ns with
  | nil => intro leaves hlen _ τ _ _; cases leaves <;> simp_all [rprod]
  | cons n ns ih =>
    intro leaves hlen hleaf τ hτ hU
    cases leaves with
    | nil => simp at hlen
    | cons F Fs =>
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at hleaf
      have h1 := ih Fs (by simpa using hlen) hleaf.2 τ (fun F hF => hτ F (by simp [hF])) hU
      simp only [rprod, List.map_cons, List.prod_cons] at h1 ⊢
      rw [h1, leafR_val S M P C p w hp hw hM0 hMb U σ n F hleaf.1 τ (hτ F (by simp)) hU]

/-- A network's formal entry mod `p` at `σ` by a checked contraction of residue factors whose
leaves read the boundary labels from `σ`. -/
theorem eval_res (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M) (hMb : M ≤ 2 ^ 64) (N : Net)
    (σ : ℕ → ℕ) (leaves : List RFac)
    (hlen : leaves.length = N.nodes.length)
    (hleaf : (N.nodes.zip leaves).all (fun q => leafR S M P C p w N.bnd σ q.1 q.2) = true)
    (steps : List (ℕ × RFac)) (final : List RFac)
    (hperm : N.int.Perm (steps.map Prod.fst).reverse)
    (hchain : chainR p S.dim leaves steps final = true)
    (hscope : scopesR N.bnd leaves steps = true)
    (hdis : N.bnd.all (fun x => !(steps.map Prod.fst).contains x) = true)
    (hσ : ∀ x ∈ N.bnd, σ x < S.dim x) :
    tmv p w (sumLT S.dim N.int (prodW S M P C N.nodes) σ) = (rfold p final σ : ZMod p) := by
  rw [tmv_sumLT, sumR_perm _ hperm, rfold_cast]
  simp only [List.all_eq_true, Bool.not_eq_true'] at hdis
  rw [← chainR_val p S.dim N.bnd steps leaves final hchain hscope σ hσ]
  apply sumR_congr
  intro τ hτ hr
  show tmv p w (prodW S M P C N.nodes τ) = rprod p leaves τ
  rw [tmv_prodW]
  refine prod_leavesR S M P C p w hp hw hM0 hMb N.bnd σ N.nodes leaves hlen hleaf τ ?_ ?_
  · intro F hF x hxF
    rcases scopesR_leaves N.bnd leaves steps hscope F hF x hxF with hb | hs
    · by_cases hm : x ∈ (steps.map Prod.fst).reverse
      · exact hr x hm
      · rw [hτ x hm]; exact hσ x hb
    · exact hr x (by simpa using hs)
  · intro x hx
    apply hτ x
    have := hdis x hx
    simpa using this

/-! ## Refutation -/

/-- The formal sum of an integer vector. -/
def vterms (f : List ℤ) : Terms := ofCoeffs [] f 0

theorem tv_vterms (z : ℂ) (f : List ℤ) : tv z [] (vterms f) = cvz z f := by
  simp [vterms, tv_ofCoeffs, mono_nil]

/-- A formal sum negated. -/
def tneg (t : Terms) : Terms := t.map fun q => (-q.1, q.2)

theorem tv_tneg (z : ℂ) (xs : List ℂ) (t : Terms) : tv z xs (tneg t) = -tv z xs t := by
  induction t with
  | nil => simp [tneg, tv]
  | cons q t ih =>
    have e1 : tneg (q :: t) = (-q.1, q.2) :: tneg t := rfl
    rw [e1, tv_cons, tv_cons, ih]
    push_cast
    ring

theorem tmv_tneg (p w : ℕ) (t : Terms) : tmv p w (tneg t) = -tmv p w t := by
  induction t with
  | nil => simp [tneg, tmv]
  | cons q t ih =>
    have e1 : tneg (q :: t) = (-q.1, q.2) :: tneg t := rfl
    have e2 : ∀ (r : ℤ × ℕ × List ℤ) (u : Terms), tmv p w (r :: u) =
        (r.1 : ZMod p) * (w : ZMod p) ^ r.2.1 + tmv p w u := fun r u => by simp [tmv]
    rw [e1, e2, e2, ih]
    push_cast
    ring

/-- Two scaled networks whose entries at `σ` are `K` times formal sums with residues `r₁`,
`r₂` are not equal when `L s K` are the integer vectors `f₁`, `f₂` and
`f₁(w) r₁ ≠ f₂(w) r₂` mod `p`. -/
theorem refute_res {E : Env} {s₁ s₂ : ℂ} {N₁ N₂ : Net} (σ : ℕ → ℕ) (hin : InRange E N₁.bnd σ)
    (M p w : ℕ) [Fact p.Prime] (hM : 0 < M) (hMb : M ≤ 2 ^ 64) (hw : IsPrimitiveRoot (w : ZMod p) M)
    (K₁ K₂ : ℂ) (T₁ T₂ : Terms) (r₁ r₂ : ℕ)
    (h₁ : eval E N₁ σ = K₁ * tv (om M) [] T₁ ∧ tmv p w T₁ = r₁)
    (h₂ : eval E N₂ σ = K₂ * tv (om M) [] T₂ ∧ tmv p w T₂ = r₂)
    (he : Equal E s₁ N₁ s₂ N₂) (L : ℂ) (f₁ f₂ : List ℤ)
    (hf₁ : L * (s₁ * K₁) = cvz (om M) f₁) (hf₂ : L * (s₂ * K₂) = cvz (om M) f₂)
    (hne : tmN p w M (vterms f₁) * r₁ % p ≠ tmN p w M (vterms f₂) * r₂ % p) : False := by
  have hp0 : p ≠ 0 := (Fact.out : p.Prime).ne_zero
  have hwM : (w : ZMod p) ^ M = 1 := hw.pow_eq_one
  have h0 := he.2 σ hin
  simp only [h₁.1, h₂.1] at h0
  have hnil : ∀ x ∈ ([] : List ℂ), x ≠ 0 := by simp
  have hR : tv (om M) [] (tmul (vterms f₁) T₁ ++ tneg (tmul (vterms f₂) T₂)) = 0 := by
    simp only [tv_append, tv_tneg, tv_mul _ _ hnil, tv_vterms]
    linear_combination L * h0 - tv (om M) [] T₁ * hf₁ + tv (om M) [] T₂ * hf₂
  have hz := tmv_eq_zero p w M hM hw _ hR
  simp only [tmv_append, tmv_tneg, tmv_mul] at hz
  simp only [h₁.2, h₂.2] at hz
  simp only [← tmN_cast p w M hp0 hwM hM hMb (vterms f₁), ← tmN_cast p w M hp0 hwM hM hMb (vterms f₂)]
    at hz
  apply hne
  rw [← ZMod.natCast_eq_natCast_iff']
  push_cast
  linear_combination hz

end ArchytasZX
