/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Cyclo

/-!
# Exact evaluation at a point

`Terms` are formal sums `Σ c ζ^e Π x_i^{k_i}` with integer coefficients, `ζ = om M` and atoms
`x_i = ex g_i`. `evalG` computes a network's entry at a concrete environment and boundary
assignment as groups `(k, l)`, one coefficient vector `l` of powers of `ζ` per monomial `k`;
`eval_cyc` proves the entry is that sum times a product of `(√d)⁻¹` factors. `refute` and
`refute_bnd` disprove an equality at one assignment.
-/

namespace ArchytasZX

open Complex Finset Polynomial

/-- Formal sums `Σ c ζ^e Π x_i^{k_i}`. -/
abbrev Terms := List (ℤ × ℕ × List ℤ)

/-- The value of a formal sum at `ζ = z` and atoms `xs`. -/
noncomputable def tv (z : ℂ) (xs : List ℂ) (t : Terms) : ℂ :=
  (t.map fun p => (p.1 : ℂ) * z ^ p.2.1 * mono xs p.2.2).sum

/-- The product of two formal sums. -/
def tmul (a b : Terms) : Terms :=
  a.flatMap fun p => b.map fun q => (p.1 * q.1, p.2.1 + q.2.1, kadd p.2.2 q.2.2)

/-- A constant. -/
def tc (c : ℤ) : Terms := [(c, 0, [])]

theorem tv_cons (z : ℂ) (xs : List ℂ) (p : ℤ × ℕ × List ℤ) (t : Terms) :
    tv z xs (p :: t) = p.1 * z ^ p.2.1 * mono xs p.2.2 + tv z xs t := by
  simp [tv]

theorem tv_append (z : ℂ) (xs : List ℂ) (a b : Terms) :
    tv z xs (a ++ b) = tv z xs a + tv z xs b := by
  simp [tv]

theorem tv_tc (z : ℂ) (xs : List ℂ) (c : ℤ) : tv z xs (tc c) = c := by
  simp [tv, tc, mono_nil]

theorem tv_flatMap {α : Type*} (z : ℂ) (xs : List ℂ) (l : List α) (f : α → Terms) :
    tv z xs (l.flatMap f) = (l.map fun x => tv z xs (f x)).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatMap_cons, tv_append, ih, List.map_cons, List.sum_cons]

theorem tv_mul (z : ℂ) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0) (a b : Terms) :
    tv z xs (tmul a b) = tv z xs a * tv z xs b := by
  induction a with
  | nil => simp [tmul, tv]
  | cons p a ih =>
    have : tmul (p :: a) b =
        (b.map fun q => (p.1 * q.1, p.2.1 + q.2.1, kadd p.2.2 q.2.2)) ++ tmul a b := by
      simp [tmul, List.flatMap_cons]
    rw [this, tv_append, ih, tv_cons, add_mul]
    congr 1
    simp only [tv, List.map_map]
    rw [← List.sum_map_mul_left]
    congr 1
    apply List.map_congr_left
    intro q _
    simp only [Function.comp, Int.cast_mul, pow_add, mono_kadd xs hx]
    ring

theorem tv_foldr_tmul (z : ℂ) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0) (l : List Terms) :
    tv z xs (l.foldr tmul (tc 1)) = (l.map (tv z xs)).prod := by
  induction l with
  | nil => simp [tv_tc]
  | cons a l ih => simp [tv_mul z xs hx, ih]

/-- Groups: one coefficient vector of powers of `ζ` per atom monomial. -/
abbrev Grp := List (List ℤ × List ℤ)

/-- The value of groups. -/
noncomputable def gval (z : ℂ) (xs : List ℂ) (g : Grp) : ℂ :=
  (g.map fun p => mono xs p.1 * cvz z p.2).sum

/-- Lexicographic order of monomials. -/
def klt : List ℤ → List ℤ → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | a :: x, b :: y => decide (a < b) || (decide (a = b) && klt x y)

/-- `c ζ^e x^k` added to groups sorted by monomial, vectors of length `M`. -/
def gadd (M : ℕ) (k : List ℤ) (e : ℕ) (c : ℤ) : Grp → Grp
  | [] => [(k, addAt (List.replicate M 0) (e % M) c)]
  | g :: gs =>
    if g.1 = k then (g.1, addAt g.2 (e % M) c) :: gs
    else if klt k g.1 then (k, addAt (List.replicate M 0) (e % M) c) :: g :: gs
    else g :: gadd M k e c gs

/-- The groups of a formal sum with no zero vector. -/
def gnorm (M : ℕ) (t : Terms) : Grp :=
  (t.foldr (fun p acc => gadd M (ktrim p.2.2) p.2.1 p.1 acc) []).filter
    fun p => p.2.any (· ≠ 0)

theorem gadd_len (M : ℕ) (k : List ℤ) (e : ℕ) (c : ℤ) (g : Grp)
    (hl : ∀ p ∈ g, p.2.length = M) : ∀ p ∈ gadd M k e c g, p.2.length = M := by
  induction g with
  | nil => simp [gadd, addAt_length]
  | cons q g ih =>
    simp only [gadd]
    split_ifs
    · intro p hp
      rcases List.mem_cons.1 hp with rfl | hp
      · simp [addAt_length, hl q (by simp)]
      · exact hl p (by simp [hp])
    · intro p hp
      rcases List.mem_cons.1 hp with rfl | hp
      · simp [addAt_length]
      · exact hl p hp
    · intro p hp
      rcases List.mem_cons.1 hp with rfl | hp
      · exact hl _ (by simp)
      · exact ih (fun p hp => hl p (by simp [hp])) p hp

theorem gval_gadd (z : ℂ) (xs : List ℂ) (M : ℕ) (hM : 0 < M) (hz : z ^ M = 1) (k : List ℤ)
    (e : ℕ) (c : ℤ) (g : Grp) (hl : ∀ p ∈ g, p.2.length = M) :
    gval z xs (gadd M k e c g) = gval z xs g + c * z ^ e * mono xs k := by
  have he : e % M < M := Nat.mod_lt _ hM
  rw [pow_mod_of z M e hz]
  induction g with
  | nil =>
    simp only [gadd, gval, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    rw [cvz_addAt _ _ _ _ (by rw [List.length_replicate]; exact he), cvz_replicate]
    ring
  | cons q g ih =>
    simp only [gadd]
    split_ifs with h1 h2
    · simp only [gval, List.map_cons, List.sum_cons]
      rw [cvz_addAt _ _ _ _ (by rw [hl q (by simp)]; exact he), ← h1]
      ring
    · simp only [gval, List.map_cons, List.sum_cons]
      rw [cvz_addAt _ _ _ _ (by simpa using he), cvz_replicate]
      ring
    · simp only [gval, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih (fun p hp => hl p (by simp [hp]))]
      ring

theorem gval_filter (z : ℂ) (xs : List ℂ) (g : Grp) :
    gval z xs (g.filter fun p => p.2.any (· ≠ 0)) = gval z xs g := by
  induction g with
  | nil => rfl
  | cons q g ih =>
    simp only [List.filter_cons]
    split_ifs with h
    · simp only [gval, List.map_cons, List.sum_cons] at ih ⊢; rw [ih]
    · simp only [gval, List.map_cons, List.sum_cons] at ih ⊢
      have : q.2.all (· == 0) = true := by
        simp only [Bool.not_eq_true, List.any_eq_false] at h
        simpa [List.all_eq_true] using h
      rw [ih, cvz_zeros z _ this]; ring

theorem gval_gnorm (z : ℂ) (xs : List ℂ) (M : ℕ) (hM : 0 < M) (hz : z ^ M = 1) (t : Terms) :
    gval z xs (gnorm M t) = tv z xs t := by
  unfold gnorm
  rw [gval_filter]
  suffices h : gval z xs (t.foldr (fun p acc => gadd M (ktrim p.2.2) p.2.1 p.1 acc) []) =
      tv z xs t ∧ ∀ p ∈ t.foldr (fun p acc => gadd M (ktrim p.2.2) p.2.1 p.1 acc) [],
        p.2.length = M from h.1
  induction t with
  | nil => simp [gval, tv]
  | cons p t ih =>
    simp only [List.foldr_cons]
    refine ⟨?_, gadd_len _ _ _ _ _ ih.2⟩
    rw [gval_gadd z xs M hM hz _ _ _ _ ih.2, ih.1, tv_cons, mono_ktrim]
    ring

/-- The nonzero coefficients of `l` with monomial `k`, the first at exponent `i`. -/
def ofCoeffs (k : List ℤ) : List ℤ → ℕ → Terms
  | [], _ => []
  | c :: l, i => if c = 0 then ofCoeffs k l (i + 1) else (c, i, k) :: ofCoeffs k l (i + 1)

theorem tv_ofCoeffs (z : ℂ) (xs : List ℂ) (k l : List ℤ) (i : ℕ) :
    tv z xs (ofCoeffs k l i) = z ^ i * mono xs k * cvz z l := by
  induction l generalizing i with
  | nil => simp [ofCoeffs, tv, cvz]
  | cons c l ih =>
    by_cases hc : c = 0
    · simp only [ofCoeffs, hc, ite_true, ih, cvz, pow_succ]; push_cast; ring
    · simp only [ofCoeffs, hc, ite_false, tv_cons, ih, cvz, pow_succ]; ring

/-- Groups as a formal sum. -/
def gterms (g : Grp) : Terms := g.flatMap fun p => ofCoeffs p.1 p.2 0

theorem tv_gterms (z : ℂ) (xs : List ℂ) (g : Grp) : tv z xs (gterms g) = gval z xs g := by
  simp only [gterms, tv_flatMap, tv_ofCoeffs, gval, pow_zero, one_mul]

/-- A formal sum with at most one term per monomial and exponent below `M`. -/
def compact (M : ℕ) (t : Terms) : Terms := gterms (gnorm M t)

theorem tv_compact (M : ℕ) (hM : 0 < M) (xs : List ℂ) (t : Terms) :
    tv (om M) xs (compact M t) = tv (om M) xs t := by
  rw [compact, tv_gterms, gval_gnorm _ _ M hM (om_pow_self M hM)]

/-- The Z spider's entry as a formal sum. -/
def tZT (P : ℕ → ℕ × List ℤ) (d : ℕ) (vs : List ℕ) : Terms :=
  (List.range d).flatMap fun k => if vs.all (· == k) then [(1, (P k).1, (P k).2)] else []

/-- The X spider's entry without its `(√d)⁻¹` factors as a formal sum. -/
def tXT (M : ℕ) (P : ℕ → ℕ × List ℤ) (d : ℕ) (os is : List ℕ) : Terms :=
  (List.range d).map fun k =>
    (1, (P k).1 + M / d * (k * os.sum) + M / d * ((d - 1) * (k * is.sum)), (P k).2)

theorem tZ_cyc (M d : ℕ) (xs : List ℂ) (P : ℕ → ℕ × List ℤ) (φ : ℕ → ℝ)
    (hφ : ∀ k < d, ex (φ k) = om M ^ (P k).1 * mono xs (P k).2)
    (vs : List ℕ) : tZ d φ vs = tv (om M) xs (tZT P d vs) := by
  unfold tZ tZT
  rw [fsum_list, tv_flatMap]
  congr 1
  apply List.map_congr_left
  intro k hk
  have hk' := List.mem_range.1 hk
  by_cases h : ∀ v ∈ vs, v = k
  · have : vs.all (· == k) = true := List.all_eq_true.2 fun v hv => by simp [h v hv]
    rw [ite_eq_left h, ite_eq_left this, hφ k hk']; simp [tv]
  · have : vs.all (· == k) = false := by
      rw [Bool.eq_false_iff]; intro h'
      exact h fun v hv => by simpa using List.all_eq_true.1 h' v hv
    rw [ite_eq_right h, ite_eq_right (by simp [this])]; simp [tv]

theorem tX_cyc (M d : ℕ) (hM : 0 < M) (hdM : d ∣ M) (xs : List ℂ) (P : ℕ → ℕ × List ℤ)
    (φ : ℕ → ℝ) (hφ : ∀ k < d, ex (φ k) = om M ^ (P k).1 * mono xs (P k).2) (os is : List ℕ) :
    tX d φ os is = (rt d)⁻¹ ^ (os.length + is.length) * tv (om M) xs (tXT M P d os is) := by
  have hd : 0 < d := Nat.pos_of_dvd_of_pos hdM hM
  obtain ⟨q, hq⟩ := hdM
  have hMd : M / d = q := by rw [hq, Nat.mul_div_cancel_left _ hd]
  have hom : om d = om M ^ q := om_sub M d q hM hq
  unfold tX tXT
  rw [fsum_list, tv, List.map_map, ← List.sum_map_mul_left]
  congr 1
  apply List.map_congr_left
  intro k hk
  rw [hφ k (List.mem_range.1 hk), prod_fou, prod_conj_fou, conj_om_pow d hd, hMd, hom]
  simp only [Function.comp, Int.cast_one, one_mul, ← pow_mul, pow_add]
  ring

/-- An environment's dimensions, labels and multiplicities. -/
structure Shape where
  dimv : ℕ → ℕ
  labs : List (ℕ × ℕ)
  mult : ℕ → ℕ

/-- An environment's shape. -/
def Env.shape (E : Env) : Shape := ⟨E.dimv, E.labs, E.mult⟩

/-- A label's dimension in a shape. -/
def Shape.dim (S : Shape) (l : ℕ) : ℕ := S.dimv ((S.labs.lookup l).getD 0)

/-- A generator's entry without its `(√d)⁻¹` factors as a formal sum. -/
def tensorT (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) :
    Kind → ℕ → ℕ → List ℕ → List ℕ → Terms
  | .Z, d, ph, os, is => tZT (P ph) d (os ++ is)
  | .X, d, ph, os, is => tXT M (P ph) d os is
  | .F, d, _, [o], [i] => [(1, M / d * (o * i), [])]
  | .T, _, _, [o], [i] => tc (if o = 0 ∨ o = i then 1 else 0)
  | .Ti, _, _, [o], [i] => tc (if o = i then 1 else if o = 0 ∧ 1 ≤ i then -1 else 0)
  | .W, d, _, os, [i] => tc (tWZ d os i)
  | .I, d, _, [a], [b] => tc (if a = b ∧ a < d then 1 else 0)
  | .ZB cs, d, ph, os, is => tZT (P ph) d (os ++ bundled E.mult d is cs)
  | .XB co ci, d, ph, os, is =>
      tXT M (P ph) d (bundled E.mult d os co) (bundled E.mult d is ci)
  | .WB cs, d, _, os, [w] => tc (tWZ d (bundled E.mult d os cs) w)
  | .Bun m b, d, ph, os, is => ((List.range (E.mult m)).map fun i =>
      tensorT E M P b d ph (copyLegs E.mult d i (b.outCs os.length) os)
        (copyLegs E.mult d i (b.inCs is.length) is)).foldr tmul (tc 1)
  | _, _, _, _, _ => []

/-- The number of `(√d)⁻¹` factors of a generator with `no` outputs and `ni` inputs. -/
def sqc (E : Shape) : Kind → ℕ → ℕ → ℕ → ℕ
  | .X, _, no, ni => no + ni
  | .F, _, _, _ => 1
  | .XB co ci, d, no, ni =>
      (bundled E.mult d (List.replicate no 0) co).length +
        (bundled E.mult d (List.replicate ni 0) ci).length
  | .Bun m b, d, no, ni => E.mult m * sqc E b d no ni
  | _, _, _, _ => 0

theorem tensor_cyc (E : Env) (M : ℕ) (xs : List ℂ) (hx : ∀ x ∈ xs, x ≠ 0)
    (P : ℕ → ℕ → ℕ × List ℤ) (hM : 0 < M) (k : Kind) (d ph : ℕ) (hdM : d ∣ M)
    (hφ : ∀ j < d, ex (E.phase ph j) = om M ^ (P ph j).1 * mono xs (P ph j).2)
    (os is : List ℕ) :
    tensor E k d ph os is = (rt d)⁻¹ ^ sqc E.shape k d os.length is.length *
      tv (om M) xs (tensorT E.shape M P k d ph os is) := by
  induction k generalizing os is with
  | Z =>
    simp only [tensor, tensorT, sqc, pow_zero, one_mul]; exact tZ_cyc M d xs (P ph) _ hφ _
  | X => simp only [tensor, tensorT, sqc]; exact tX_cyc M d hM hdM xs (P ph) _ hφ os is
  | ZB cs =>
    simp only [tensor, tensorT, sqc, pow_zero, one_mul]; exact tZ_cyc M d xs (P ph) _ hφ _
  | XB co ci =>
    simp only [tensor, tensorT, sqc, Env.shape]
    rw [tX_cyc M d hM hdM xs (P ph) _ hφ, bundled_length, bundled_length E.mult d is]
  | F =>
    have hd : 0 < d := Nat.pos_of_dvd_of_pos hdM hM
    obtain ⟨q, hq⟩ := hdM
    have hMd : M / d = q := by rw [hq, Nat.mul_div_cancel_left _ hd]
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [fou, hMd, om_sub M d q hM hq, ← pow_mul, rt, div_eq_mul_inv, mul_comm, mono_nil]
  | T =>
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tT, mono_nil]
  | Ti =>
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tTi, mono_nil]
  | I =>
    rcases os with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tI, mono_nil]
  | W =>
    rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tWZ_eq, mono_nil, Env.shape]
  | WB cs =>
    rcases is with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
      simp only [tensor, tensorT, sqc, tv, List.map_nil, List.sum_nil, mul_zero]
    simp [tc, tWZ_eq, mono_nil, Env.shape]
  | B => simp [tensor, tensorT, tv]
  | S => simp [tensor, tensorT, tv]
  | Bun m b ih =>
    simp only [tensor, tensorT, sqc, tv_foldr_tmul _ _ hx, List.map_map, copyOuts, copyIns,
      Env.shape]
    have : ∀ i, tensor E b d ph (copyLegs E.mult d i (b.outCs os.length) os)
        (copyLegs E.mult d i (b.inCs is.length) is) =
        (rt d)⁻¹ ^ sqc E.shape b d os.length is.length *
          tv (om M) xs (tensorT E.shape M P b d ph (copyLegs E.mult d i (b.outCs os.length) os)
            (copyLegs E.mult d i (b.inCs is.length) is)) := by
      intro i
      rw [ih, show (copyLegs E.mult d i (b.outCs os.length) os).length = os.length from
        copyOuts_length E b d i os, show (copyLegs E.mult d i (b.inCs is.length) is).length =
        is.length from copyIns_length E b d i is]
    simp only [this, prod_pow_mul, List.length_range]
    rfl

/-- Whether a node is a binder or splitter. -/
def isBS (n : Node) : Bool := n.kind == .B || n.kind == .S

/-- A binder or splitter node's entry as a formal sum. -/
def valBS (E : Shape) (σ : ℕ → ℕ) (n : Node) : Terms :=
  match n.kind, n.outs.map σ, n.ins.map σ with
  | .B, [k], [a, b] => tc (if a < E.dimv n.dim ∧ b < E.dimv n.ph ∧ k = a * E.dimv n.ph + b
      then 1 else 0)
  | .S, [a, b], [k] => tc (if a < E.dimv n.dim ∧ b < E.dimv n.ph ∧ k = a * E.dimv n.ph + b
      then 1 else 0)
  | _, _, _ => []

/-- A node's entry without its `(√d)⁻¹` factors as a formal sum. -/
def valT (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (σ : ℕ → ℕ) (n : Node) : Terms :=
  if isBS n then valBS E σ n
  else tensorT E M P n.kind (E.dimv n.dim) n.ph (n.outs.map σ) (n.ins.map σ)

/-- A node's number of `(√d)⁻¹` factors. -/
def sqN (E : Shape) (n : Node) : ℕ :=
  if isBS n then 0 else sqc E n.kind (E.dimv n.dim) n.outs.length n.ins.length

/-- `M` is a multiple of the node's dimension and `keys` lists its phase entries. -/
def nodeOK (E : Shape) (M : ℕ) (keys : List (ℕ × ℕ)) (n : Node) : Bool :=
  isBS n || (M % E.dimv n.dim == 0 &&
    (List.range (E.dimv n.dim)).all fun j => keys.contains (n.ph, j))

/-- Each listed phase entry is `om M ^ e` times a monomial in the atoms `ex g`. -/
def PhOK (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ) : List (ℕ × ℕ) → Prop
  | [] => True
  | e :: l => ex (E.phase e.1 e.2) = om M ^ (P e.1 e.2).1 * mono (gs.map ex) (P e.1 e.2).2 ∧
      PhOK E M gs P l

theorem PhOK.get {E : Env} {M : ℕ} {gs : List ℝ} {P : ℕ → ℕ → ℕ × List ℤ}
    {keys : List (ℕ × ℕ)} (h : PhOK E M gs P keys) {e : ℕ × ℕ} (he : e ∈ keys) :
    ex (E.phase e.1 e.2) = om M ^ (P e.1 e.2).1 * mono (gs.map ex) (P e.1 e.2).2 := by
  induction keys with
  | nil => simp at he
  | cons a l ih =>
    rcases List.mem_cons.1 he with rfl | he
    · exact h.1
    · exact ih h.2 he

theorem val_cyc (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (keys : List (ℕ × ℕ)) (hM : 0 < M) (hph : PhOK E M gs P keys) (σ : ℕ → ℕ) (n : Node)
    (hok : nodeOK E.shape M keys n = true) :
    val E σ n = (rt (E.dimv n.dim))⁻¹ ^ sqN E.shape n *
      tv (om M) (gs.map ex) (valT E.shape M P σ n) := by
  obtain ⟨kind, outs, ins, dim, ph⟩ := n
  by_cases hb : kind = .B ∨ kind = .S
  · have : isBS ⟨kind, outs, ins, dim, ph⟩ = true := by
      rcases hb with rfl | rfl <;> rfl
    simp only [valT, sqN, this, ite_true, pow_zero, one_mul]
    dsimp only [Env.shape]
    rcases hb with rfl | rfl <;> simp only [val, valBS, Env.shape] <;>
      generalize outs.map σ = A <;> generalize ins.map σ = B <;>
      rcases A with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;>
      rcases B with _ | ⟨_, _ | ⟨_, _ | ⟨_, _⟩⟩⟩ <;>
      simp [tv, tc, tB, mono_nil, Int.cast_ite, Env.shape]
  · have hb' : kind ≠ .B ∧ kind ≠ .S := by tauto
    have hn : isBS ⟨kind, outs, ins, dim, ph⟩ = false := by
      simp [isBS, hb'.1, hb'.2]
    simp only [nodeOK, hn, Bool.false_or, Bool.and_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range, List.contains_iff_mem, Env.shape] at hok
    rw [val_tensor E σ kind hb', tensor_cyc E M (gs.map ex) (ex_atoms_ne gs) P hM kind _ ph
      (Nat.dvd_of_mod_eq_zero hok.1) (fun j hj => hph.get (hok.2 j hj))]
    simp [valT, sqN, hn, Env.shape]

/-- The product of the node entries without their `(√d)⁻¹` factors. -/
def prodT (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (ns : List Node) (σ : ℕ → ℕ) : Terms :=
  ns.foldr (fun n acc => compact M (tmul (valT E M P σ n) acc)) (tc 1)

theorem prodVal_cyc (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (keys : List (ℕ × ℕ)) (hM : 0 < M) (hph : PhOK E M gs P keys) (ns : List Node)
    (hok : ns.all (nodeOK E.shape M keys) = true) (σ : ℕ → ℕ) :
    prodVal E ns σ = (ns.map fun n => (rt (E.dimv n.dim))⁻¹ ^ sqN E.shape n).prod *
      tv (om M) (gs.map ex) (prodT E.shape M P ns σ) := by
  induction ns with
  | nil => simp [prodVal, prodT, tv_tc]
  | cons n ns ih =>
    simp only [List.all_cons, Bool.and_eq_true] at hok
    have := ih hok.2
    simp only [prodVal, List.map_cons, List.prod_cons, prodT, List.foldr_cons] at this ⊢
    rw [tv_compact M hM, tv_mul _ _ (ex_atoms_ne gs), val_cyc E M gs P keys hM hph σ n hok.1,
      this]
    ring

/-- The sum over the labels `ls` of formal sums, compacted after each label. -/
def sumT (M : ℕ) (D : ℕ → ℕ) : List ℕ → ((ℕ → ℕ) → Terms) → (ℕ → ℕ) → Terms
  | [], f, σ => f σ
  | l :: ls, f, σ =>
    compact M ((List.range (D l)).flatMap fun j => sumT M D ls f (Function.update σ l j))

theorem sumOver_cyc (M : ℕ) (hM : 0 < M) (xs : List ℂ) (D : ℕ → ℕ) (K : ℂ)
    (g : (ℕ → ℕ) → ℂ) (f : (ℕ → ℕ) → Terms) (hf : ∀ τ, g τ = K * tv (om M) xs (f τ))
    (ls : List ℕ) (σ : ℕ → ℕ) :
    sumOver D ls g σ = K * tv (om M) xs (sumT M D ls f σ) := by
  induction ls generalizing σ with
  | nil => exact hf σ
  | cons l ls ih =>
    simp only [sumOver, sumT, tv_compact M hM, tv_flatMap, fsum_list, ih,
      List.sum_map_mul_left]

/-- A network's entry without its `(√d)⁻¹` factors, as groups. -/
def evalG (E : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (N : Net) (σ : ℕ → ℕ) : Grp :=
  gnorm M (sumT M E.dim N.int (prodT E M P N.nodes) σ)

/-- Each node's dimension with its number of `(√d)⁻¹` factors. -/
def sqL (E : Shape) (N : Net) : List (ℕ × ℕ) := N.nodes.map fun n => (E.dimv n.dim, sqN E n)

/-- The product of a network's `(√d)⁻¹` factors. -/
noncomputable def sqK (E : Shape) (N : Net) : ℂ :=
  ((sqL E N).map fun p => (rt p.1)⁻¹ ^ p.2).prod

/-- A network's entry at a concrete environment, computed exactly. -/
theorem eval_cyc (E : Env) (M : ℕ) (gs : List ℝ) (P : ℕ → ℕ → ℕ × List ℤ)
    (keys : List (ℕ × ℕ)) (N : Net) (σ : ℕ → ℕ) (hM : 0 < M)
    (hok : N.nodes.all (nodeOK E.shape M keys) = true) (hph : PhOK E M gs P keys) :
    eval E N σ = sqK E.shape N * gval (om M) (gs.map ex) (evalG E.shape M P N σ) := by
  unfold eval evalG sqK
  rw [gval_gnorm _ _ M hM (om_pow_self M hM)]
  have : ((sqL E.shape N).map fun p => (rt p.1)⁻¹ ^ p.2) =
      N.nodes.map fun n => (rt (E.dimv n.dim))⁻¹ ^ sqN E.shape n := by simp [sqL, Env.shape]
  rw [this, show E.shape.dim = labDim E from rfl]
  exact sumOver_cyc M hM _ _ _ _ _ (prodVal_cyc E M gs P keys hM hph N.nodes hok) _ _

theorem inRange_of (E : Env) (ls : List ℕ) (σ : ℕ → ℕ)
    (h : ls.all (fun l => decide (σ l < labDim E l)) = true) : InRange E ls σ := by
  intro l hl
  simpa using List.all_eq_true.1 h l hl

/-- Two scaled networks differing at one in-range boundary assignment are not equal. -/
theorem refute {E : Env} {s₁ s₂ : ℂ} {N₁ N₂ : Net} (σ : ℕ → ℕ) (hin : InRange E N₁.bnd σ)
    (h : s₁ * eval E N₁ σ ≠ s₂ * eval E N₂ σ) : ¬ Equal E s₁ N₁ s₂ N₂ :=
  fun he => h (he.2 σ hin)

/-- Networks with boundaries that are not permutations of each other are not equal. -/
theorem refute_bnd {E : Env} {s₁ s₂ : ℂ} {N₁ N₂ : Net} (h : ¬ N₁.bnd.Perm N₂.bnd) :
    ¬ Equal E s₁ N₁ s₂ N₂ :=
  fun he => h he.1

/-- Every assignment of the labels `ls` below their dimensions, other labels `0`. -/
def assigns (D : ℕ → ℕ) : List ℕ → List (ℕ → ℕ)
  | [] => [fun _ => 0]
  | l :: ls =>
    (assigns D ls).flatMap fun τ => (List.range (D l)).map fun j => Function.update τ l j

theorem assigns_cover (D : ℕ → ℕ) (ls : List ℕ) (σ : ℕ → ℕ) (h : ∀ l ∈ ls, σ l < D l) :
    ∃ τ ∈ assigns D ls, ∀ l ∈ ls, τ l = σ l := by
  induction ls with
  | nil => exact ⟨fun _ => 0, by simp [assigns], by simp⟩
  | cons l ls ih =>
    obtain ⟨τ, hτ, hτσ⟩ := ih (fun l' hl' => h l' (by simp [hl']))
    refine ⟨Function.update τ l (σ l), ?_, ?_⟩
    · simp only [assigns, List.mem_flatMap, List.mem_map, List.mem_range]
      exact ⟨τ, hτ, σ l, h l (by simp), rfl⟩
    · intro l' hl'
      by_cases e : l' = l
      · subst e; simp
      · rw [Function.update_of_ne e]
        exact hτσ l' (by simpa [e] using hl')

end ArchytasZX
