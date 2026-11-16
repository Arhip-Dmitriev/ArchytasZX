/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.RefuteMod

/-!
# Residues by substitution and geometric sums

A substitution state (`SSt`) binds summed labels to label expressions (`LX`) in the remaining
ones, and carries nodes, divisibility constraints and a constant. A delta step sums out a label
that a binder, splitter, Z spider, wire or constraint fixes; a geometric step sums out a label
shared by exactly two Fourier boxes by `Σ_j u^{j a} = N [N ∣ a]` and adds that constraint.
`sym_final` contracts the remaining labels with residue tables.
-/

namespace ArchytasZX

open Finset

/-- Label expressions. -/
inductive LX where
  | v : ℕ → LX
  | c : ℕ → LX
  | add : LX → LX → LX
  | mul : ℕ → LX → LX
  | div : LX → ℕ → LX
  | mod : LX → ℕ → LX
  | neg : LX → ℕ → LX
  deriving DecidableEq, Repr

/-- `(n - a % n) % n`. -/
def negN (n a : ℕ) : ℕ := (n - a % n) % n

/-- An expression's value at an assignment. -/
def LX.eval (τ : ℕ → ℕ) : LX → ℕ
  | .v x => τ x
  | .c k => k
  | .add a b => a.eval τ + b.eval τ
  | .mul k a => k * a.eval τ
  | .div a k => a.eval τ / k
  | .mod a k => a.eval τ % k
  | .neg a n => negN n (a.eval τ)

/-- The labels an expression reads. -/
def LX.vars : LX → List ℕ
  | .v x => [x]
  | .c _ => []
  | .add a b => a.vars ++ b.vars
  | .mul _ a => a.vars
  | .div a _ => a.vars
  | .mod a _ => a.vars
  | .neg a _ => a.vars

/-- `l` replaced by `r`. -/
def LX.subst : LX → ℕ → LX → LX
  | .v x, l, r => if x = l then r else .v x
  | .c k, _, _ => .c k
  | .add a b, l, r => .add (a.subst l r) (b.subst l r)
  | .mul k a, l, r => .mul k (a.subst l r)
  | .div a k, l, r => .div (a.subst l r) k
  | .mod a k, l, r => .mod (a.subst l r) k
  | .neg a n, l, r => .neg (a.subst l r) n

/-- An upper bound of the value when each label `x` is below `D x`. -/
def LX.ub (D : ℕ → ℕ) : LX → ℕ
  | .v x => D x - 1
  | .c k => k
  | .add a b => a.ub D + b.ub D
  | .mul k a => k * a.ub D
  | .div a k => a.ub D / k
  | .mod a k => if k = 0 then a.ub D else k - 1
  | .neg _ n => n - 1

theorem LX.eval_subst (τ : ℕ → ℕ) (l : ℕ) (r : LX) (e : LX) :
    (e.subst l r).eval τ = e.eval (Function.update τ l (r.eval τ)) := by
  induction e with
  | v x =>
    by_cases h : x = l
    · subst h; simp [LX.subst, LX.eval]
    · simp [LX.subst, LX.eval, h, Function.update_apply]
  | c k => rfl
  | add a b iha ihb => simp [LX.subst, LX.eval, iha, ihb]
  | mul k a ih => simp [LX.subst, LX.eval, ih]
  | div a k ih => simp [LX.subst, LX.eval, ih]
  | mod a k ih => simp [LX.subst, LX.eval, ih]
  | neg a n ih => simp [LX.subst, LX.eval, ih]

theorem LX.eval_congr (τ τ' : ℕ → ℕ) (e : LX) (h : ∀ x ∈ e.vars, τ x = τ' x) :
    e.eval τ = e.eval τ' := by
  induction e with
  | v x => exact h x (by simp [LX.vars])
  | c k => rfl
  | add a b iha ihb =>
    simp only [LX.eval]
    rw [iha fun x hx => h x (by simp [LX.vars, hx]), ihb fun x hx => h x (by simp [LX.vars, hx])]
  | mul k a ih => simp only [LX.eval]; rw [ih fun x hx => h x (by simpa [LX.vars] using hx)]
  | div a k ih => simp only [LX.eval]; rw [ih fun x hx => h x (by simpa [LX.vars] using hx)]
  | mod a k ih => simp only [LX.eval]; rw [ih fun x hx => h x (by simpa [LX.vars] using hx)]
  | neg a n ih => simp only [LX.eval]; rw [ih fun x hx => h x (by simpa [LX.vars] using hx)]

theorem LX.eval_update (τ : ℕ → ℕ) (l j : ℕ) (e : LX) (h : l ∉ e.vars) :
    e.eval (Function.update τ l j) = e.eval τ :=
  e.eval_congr _ _ fun x hx => Function.update_of_ne (fun h' => h (by rw [← h']; exact hx)) _ _

theorem LX.eval_le_ub (D : ℕ → ℕ) (τ : ℕ → ℕ) (e : LX) (h : ∀ x ∈ e.vars, τ x < D x) :
    e.eval τ ≤ e.ub D := by
  induction e with
  | v x => have := h x (by simp [LX.vars]); simp only [LX.eval, LX.ub]; omega
  | c k => exact le_refl _
  | add a b iha ihb =>
    exact Nat.add_le_add (iha fun x hx => h x (by simp [LX.vars, hx]))
      (ihb fun x hx => h x (by simp [LX.vars, hx]))
  | mul k a ih => exact Nat.mul_le_mul_left _ (ih fun x hx => h x (by simpa [LX.vars] using hx))
  | div a k ih => exact Nat.div_le_div_right (ih fun x hx => h x (by simpa [LX.vars] using hx))
  | mod a k ih =>
    simp only [LX.eval, LX.ub]
    split_ifs with hk
    · subst hk; rw [Nat.mod_zero]; exact ih fun x hx => h x (by simpa [LX.vars] using hx)
    · have := Nat.mod_lt (a.eval τ) (Nat.pos_of_ne_zero hk); omega
  | neg a n ih =>
    simp only [LX.eval, LX.ub, negN]
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn; simp
    · have := Nat.mod_lt (n - a.eval τ % n) hn; omega

theorem negN_lt (n a : ℕ) (hn : 0 < n) : negN n a < n := Nat.mod_lt _ hn

theorem add_negN (n a : ℕ) (hn : 0 < n) : (a + negN n a) % n = 0 := by
  unfold negN
  have ha := Nat.mod_lt a hn
  rw [Nat.add_mod, Nat.mod_mod]
  rcases Nat.eq_zero_or_pos (a % n) with h0 | h0
  · rw [h0]; simp
  · rw [Nat.mod_eq_of_lt (show n - a % n < n by omega), show a % n + (n - a % n) = n by omega,
      Nat.mod_self]

theorem eq_negN (n a x : ℕ) (hx : x < n) (h : (x + a) % n = 0) : x = negN n a := by
  have hn : 0 < n := by omega
  have h1 : (x + a) % n = (negN n a + a) % n := by
    rw [h, add_comm (negN n a), add_negN n a hn]
  have h2 : x % n = negN n a % n := Nat.ModEq.add_right_cancel' a h1
  rwa [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt (negN_lt n a hn)] at h2

theorem negN_negN (n x : ℕ) (hx : x < n) : negN n (negN n x) = x := by
  have hn : 0 < n := by omega
  symm
  exact eq_negN n _ x hx (add_negN n x hn)

/-! ## Bindings -/

/-- The expression bound to `x`. -/
def blook : List (ℕ × LX) → ℕ → Option LX
  | [], _ => none
  | (a, e) :: t, x => if a = x then some e else blook t x

/-- A label's expression under bindings. -/
def texpr (bs : List (ℕ × LX)) (x : ℕ) : LX := (blook bs x).getD (.v x)

/-- A label's value under bindings. -/
def tval (bs : List (ℕ × LX)) (τ : ℕ → ℕ) (x : ℕ) : ℕ := (texpr bs x).eval τ

/-- Bindings with `l` replaced by `r`. -/
def bsubst (l : ℕ) (r : LX) (bs : List (ℕ × LX)) : List (ℕ × LX) :=
  bs.map fun b => (b.1, b.2.subst l r)

theorem blook_bsubst (l : ℕ) (r : LX) (x : ℕ) :
    ∀ bs : List (ℕ × LX), blook (bsubst l r bs) x = (blook bs x).map (fun e => e.subst l r)
  | [] => rfl
  | (a, e) :: t => by
    simp only [bsubst, List.map_cons, blook]
    split_ifs
    · rfl
    · exact blook_bsubst l r x t

theorem tval_nil (τ : ℕ → ℕ) : tval [] τ = τ := by
  funext x; rfl

theorem tval_step (bs : List (ℕ × LX)) (l : ℕ) (e : LX) (hl : blook bs l = none)
    (τ : ℕ → ℕ) :
    tval ((l, e) :: bsubst l e bs) τ = tval bs (Function.update τ l (e.eval τ)) := by
  funext x
  unfold tval texpr
  simp only [blook]
  by_cases hx : l = x
  · subst hx; simp [hl, LX.eval]
  · rw [if_neg hx, blook_bsubst]
    cases h : blook bs x with
    | none => simp [LX.eval, Function.update_of_ne (Ne.symm hx)]
    | some e' => simp [LX.eval_subst]

/-! ## States -/

/-- Bindings, nodes, divisibility constraints and a constant. -/
structure SSt where
  bs : List (ℕ × LX)
  ns : List Node
  cs : List (ℕ × LX)
  c : ℕ
  deriving DecidableEq, Repr

/-- A constraint's value: `1` when `N` divides the expression. -/
def cval (p : ℕ) (q : ℕ × LX) (τ : ℕ → ℕ) : ZMod p := if q.1 ∣ q.2.eval τ then 1 else 0

/-- A state's value at an assignment of the summed labels. -/
def sval (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ) (st : SSt)
    (τ : ℕ → ℕ) : ZMod p :=
  (st.c : ZMod p) * (st.ns.map fun n => tmv p w (valW S M P C (tval st.bs τ) n)).prod *
    (st.cs.map fun q => cval p q τ).prod

/-- A node's nlegs. -/
def nlegs (n : Node) : List ℕ := n.outs ++ n.ins

/-- Some leg expression of `n` reads `l`. -/
def nmen (bs : List (ℕ × LX)) (l : ℕ) (n : Node) : Bool :=
  (nlegs n).any fun x => (texpr bs x).vars.contains l

theorem node_frame (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (bs : List (ℕ × LX)) (l j : ℕ) (n : Node) (h : nmen bs l n = false) (τ : ℕ → ℕ) :
    valW S M P C (tval bs (Function.update τ l j)) n = valW S M P C (tval bs τ) n := by
  apply valW_congr
  intro x hx
  simp only [nmen, List.any_eq_false, List.contains_iff_mem, Bool.not_eq_true] at h
  have : l ∉ (texpr bs x).vars := by
    have := h x (by simpa [nlegs] using hx); simpa using this
  exact LX.eval_update τ l j _ this

theorem cons_frame (p : ℕ) (q : ℕ × LX) (l j : ℕ) (h : q.2.vars.contains l = false)
    (τ : ℕ → ℕ) : cval p q (Function.update τ l j) = cval p q τ := by
  unfold cval
  rw [LX.eval_update τ l j _ (by simpa using h)]

/-! ## Delta steps -/

/-- Where a delta step reads its relation: a Z spider through two of its nlegs, a binder or
splitter through one leg, or a constraint. -/
inductive Src where
  | zz : Node → ℕ → ℕ → Src
  | bs : Node → ℕ → Src
  | cn : ℕ × LX → Src
  deriving DecidableEq, Repr

/-- A source node's leg expression and the expression it equals when the node is nonzero. -/
def spair (S : Shape) (bs : List (ℕ × LX)) : Src → Option (LX × LX)
  | .zz n x y => if n.kind = .Z ∧ x ∈ nlegs n ∧ y ∈ nlegs n then some (texpr bs x, texpr bs y)
      else none
  | .bs n pos =>
    match n.kind, n.outs, n.ins, pos with
    | .B, [k], [a, b], 0 =>
      some (texpr bs k, .add (.mul (S.dimv n.ph) (texpr bs a)) (texpr bs b))
    | .B, [k], [a, _], 1 => some (texpr bs a, .div (texpr bs k) (S.dimv n.ph))
    | .B, [k], [_, b], 2 => some (texpr bs b, .mod (texpr bs k) (S.dimv n.ph))
    | .S, [a, b], [k], 2 =>
      some (texpr bs k, .add (.mul (S.dimv n.ph) (texpr bs a)) (texpr bs b))
    | .S, [a, _], [k], 0 => some (texpr bs a, .div (texpr bs k) (S.dimv n.ph))
    | .S, [_, b], [k], 1 => some (texpr bs b, .mod (texpr bs k) (S.dimv n.ph))
    | _, _, _, _ => none
  | .cn _ => none

/-- The value of `l` from a leg expression and the expression it equals. -/
def solveP (D : ℕ → ℕ) (l : ℕ) : Option (LX × LX) → Option LX
  | some (.v x, t) => if x = l then some t else none
  | some (.neg (.v x) n, t) => if x = l ∧ n = D l then some (.neg t n) else none
  | _ => none

/-- The value `l` takes where a constraint `N ∣ a + b` holds, `a` reading `l`. -/
def csolve (D : ℕ → ℕ) (l N : ℕ) : LX → LX → Option LX
  | .v x, b => if x = l ∧ N = D l then some (.neg b N) else none
  | .neg (.v x) n, b => if x = l ∧ n = N ∧ N = D l then some (.mod b N) else none
  | _, _ => none

/-- The value `l` takes where the source is nonzero. -/
def solve (S : Shape) (D : ℕ → ℕ) (bs : List (ℕ × LX)) (l : ℕ) : Src → Option LX
  | .cn (N, .add a b) => (csolve D l N a b).orElse fun _ => csolve D l N b a
  | .cn _ => none
  | .zz n x y => solveP D l (spair S bs (.zz n x y))
  | .bs n pos => solveP D l (spair S bs (.bs n pos))

/-- A source's factor in a state's value. -/
def sfac (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (bs : List (ℕ × LX)) (τ : ℕ → ℕ) : Src → ZMod p
  | .zz n _ _ => tmv p w (valW S M P C (tval bs τ) n)
  | .bs n _ => tmv p w (valW S M P C (tval bs τ) n)
  | .cn q => cval p q τ

theorem tmv_tc_zero (p w : ℕ) : tmv p w (tc 0) = 0 := by simp [tmv, tc]

theorem tZT_eq (Pk : ℕ → ℕ × List ℤ) (d : ℕ) (vs : List ℕ) (p w : ℕ)
    (h : tmv p w (tZT Pk d vs) ≠ 0) : ∃ k, ∀ v ∈ vs, v = k := by
  by_contra hne
  push_neg at hne
  apply h
  unfold tZT
  rw [tmv_flatMap]
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨k, _, rfl⟩ := List.mem_map.1 hx
  obtain ⟨v, hv, hvk⟩ := hne k
  have : vs.all (· == k) = false := by
    rw [Bool.eq_false_iff]; intro h'
    exact hvk (by simpa using List.all_eq_true.1 h' v hv)
  simp [this, tmv]

theorem bs_rel (k a b t : ℕ) (hb : b < t) (hk : k = a * t + b) :
    k = t * a + b ∧ a = k / t ∧ b = k % t := by
  have ht : 0 < t := by omega
  refine ⟨by rw [hk]; ring, ?_, ?_⟩
  · rw [hk, add_comm, Nat.add_mul_div_right _ _ ht, Nat.div_eq_of_lt hb, zero_add]
  · rw [hk, add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

theorem valB_ne (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (τ : ℕ → ℕ) (k a b dim ph : ℕ)
    (hv : tmv p w (valW S M P C τ ⟨.B, [k], [a, b], dim, ph⟩) ≠ 0) :
    τ b < S.dimv ph ∧ τ k = τ a * S.dimv ph + τ b := by
  simp only [valW, isBS, valBS, List.map_cons, List.map_nil, beq_self_eq_true, Bool.true_or,
    ite_true] at hv
  split_ifs at hv with hc
  · exact ⟨hc.2.1, hc.2.2⟩
  · exact absurd (tmv_tc_zero p w) hv

theorem valS_ne (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (τ : ℕ → ℕ) (k a b dim ph : ℕ)
    (hv : tmv p w (valW S M P C τ ⟨.S, [a, b], [k], dim, ph⟩) ≠ 0) :
    τ b < S.dimv ph ∧ τ k = τ a * S.dimv ph + τ b := by
  simp only [valW, isBS, valBS, List.map_cons, List.map_nil, beq_self_eq_true, Bool.or_true,
    ite_true] at hv
  split_ifs at hv with hc
  · exact ⟨hc.2.1, hc.2.2⟩
  · exact absurd (tmv_tc_zero p w) hv

theorem spair_sound (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (bs : List (ℕ × LX)) (src : Src) (a t : LX) (h : spair S bs src = some (a, t))
    (τ : ℕ → ℕ) (hv : sfac S M P C p w bs τ src ≠ 0) : a.eval τ = t.eval τ := by
  cases src with
  | cn q => simp [spair] at h
  | zz n x y =>
    simp only [spair] at h
    split_ifs at h with hc
    obtain ⟨hk, hx, hy⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨kind, outs, ins, dim, ph⟩ := n
    simp only at hk
    subst hk
    have hv' : tmv p w (tZT (P ph) (S.dimv dim)
        (outs.map (tval bs τ) ++ ins.map (tval bs τ))) ≠ 0 := hv
    obtain ⟨k, hk⟩ := tZT_eq _ _ _ p w hv'
    have hx' := hk (tval bs τ x)
      (by rw [← List.map_append]; exact List.mem_map_of_mem (by simpa [nlegs] using hx))
    have hy' := hk (tval bs τ y)
      (by rw [← List.map_append]; exact List.mem_map_of_mem (by simpa [nlegs] using hy))
    show tval bs τ x = tval bs τ y
    rw [hx', hy']
  | bs n pos =>
    obtain ⟨kind, outs, ins, dim, ph⟩ := n
    simp only [sfac] at hv
    simp only [spair] at h
    split at h
    all_goals
      try (simp only [reduceCtorEq] at h; done)
    all_goals
      first
        | (simp only [Option.some.injEq, Prod.mk.injEq] at h
           obtain ⟨rfl, rfl⟩ := h
           first
             | (obtain ⟨hb, hk⟩ := valB_ne S M P C p w _ _ _ _ _ _ hv
                have hr := bs_rel _ _ _ _ hb hk
                simp only [tval] at hr
                simp only [LX.eval]
                first | exact hr.1 | exact hr.2.1 | exact hr.2.2)
             | (obtain ⟨hb, hk⟩ := valS_ne S M P C p w _ _ _ _ _ _ hv
                have hr := bs_rel _ _ _ _ hb hk
                simp only [tval] at hr
                simp only [LX.eval]
                first | exact hr.1 | exact hr.2.1 | exact hr.2.2))
        | simp at h

theorem solveP_sound (D : ℕ → ℕ) (l : ℕ) (pr : Option (LX × LX)) (e : LX)
    (h : solveP D l pr = some e) (τ : ℕ → ℕ) (hl : τ l < D l)
    (hpr : ∀ a t, pr = some (a, t) → a.eval τ = t.eval τ) : τ l = e.eval τ := by
  unfold solveP at h
  split at h
  · rename_i x t
    split_ifs at h with hx
    simp only [Option.some.injEq] at h
    subst hx; subst h
    exact hpr _ _ rfl
  · rename_i x n t
    split_ifs at h with hx
    obtain ⟨rfl, rfl⟩ := hx
    simp only [Option.some.injEq] at h
    subst h
    have := hpr _ _ rfl
    simp only [LX.eval] at this ⊢
    rw [← this, negN_negN _ _ hl]
  · simp at h

theorem negN_case (n y b : ℕ) (hl : y < n) (hd : (negN n y + b) % n = 0) : y = b % n := by
  have hpos : 0 < n := by omega
  have h1 := eq_negN _ _ _ (negN_lt n y hpos) hd
  have h2 : negN n b = negN n (b % n) := by simp [negN, Nat.mod_mod]
  rw [← negN_negN n y hl, h1, h2, negN_negN _ _ (Nat.mod_lt _ hpos)]

theorem csolve_sound (D : ℕ → ℕ) (l N : ℕ) (a b e : LX) (h : csolve D l N a b = some e)
    (τ : ℕ → ℕ) (hl : τ l < D l) (hd : N ∣ a.eval τ + b.eval τ) : τ l = e.eval τ := by
  have hd' : (a.eval τ + b.eval τ) % N = 0 := Nat.mod_eq_zero_of_dvd hd
  cases a with
  | v x =>
    simp only [csolve] at h
    split_ifs at h with hc
    obtain ⟨hx, hN⟩ := hc
    simp only [Option.some.injEq] at h
    subst h hx
    simp only [LX.eval] at hd' ⊢
    rw [hN] at hd' ⊢
    exact eq_negN _ _ _ hl hd'
  | neg a' n =>
    cases a' with
    | v x =>
      simp only [csolve] at h
      split_ifs at h with hc
      obtain ⟨hx, hn, hN⟩ := hc
      simp only [Option.some.injEq] at h
      subst h hx hn
      simp only [LX.eval] at hd' ⊢
      rw [hN] at hd' ⊢
      exact negN_case _ _ _ hl hd'
    | _ => simp [csolve] at h
  | _ => simp [csolve] at h

theorem solve_sound (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (D : ℕ → ℕ) (bs : List (ℕ × LX)) (l : ℕ) (src : Src) (e : LX)
    (h : solve S D bs l src = some e) (τ : ℕ → ℕ) (hl : τ l < D l)
    (hv : sfac S M P C p w bs τ src ≠ 0) : τ l = e.eval τ := by
  cases src with
  | zz n x y =>
    exact solveP_sound D l _ e h τ hl fun a t hp =>
      spair_sound S M P C p w bs _ a t hp τ hv
  | bs n pos =>
    exact solveP_sound D l _ e h τ hl fun a t hp =>
      spair_sound S M P C p w bs _ a t hp τ hv
  | cn q =>
    obtain ⟨N, ex⟩ := q
    simp only [sfac, cval] at hv
    split_ifs at hv with hd
    · cases ex with
      | add a b =>
        simp only [LX.eval] at hd
        simp only [solve, Option.orElse] at h
        cases h1 : csolve D l N a b with
        | some e' =>
          rw [h1] at h; simp only [Option.some.injEq] at h; subst h
          exact csolve_sound D l N a b _ h1 τ hl hd
        | none =>
          rw [h1] at h
          exact csolve_sound D l N b a _ h τ hl (by rwa [add_comm])
      | _ => simp [solve] at h
    · exact absurd rfl hv

/-- The source is among the state's factors. -/
def sin (st : SSt) : Src → Bool
  | .zz n _ _ => st.ns.contains n
  | .bs n _ => st.ns.contains n
  | .cn q => st.cs.contains q

theorem sval_zero (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (st : SSt) (src : Src) (h : sin st src = true) (τ : ℕ → ℕ)
    (h0 : sfac S M P C p w st.bs τ src = 0) : sval S M P C p w st τ = 0 := by
  unfold sval
  cases src with
  | zz n x y =>
    simp only [sin, List.contains_iff_mem] at h
    rw [List.prod_eq_zero (List.mem_map.2 ⟨n, h, h0⟩)]; ring
  | bs n pos =>
    simp only [sin, List.contains_iff_mem] at h
    rw [List.prod_eq_zero (List.mem_map.2 ⟨n, h, h0⟩)]; ring
  | cn q =>
    simp only [sin, List.contains_iff_mem] at h
    rw [List.prod_eq_zero (List.mem_map.2 ⟨q, h, h0⟩)]; ring

/-- The state after binding `l` to `e`. -/
def dapply (l : ℕ) (e : LX) (st : SSt) : SSt :=
  ⟨(l, e) :: bsubst l e st.bs, st.ns, st.cs.map fun q => (q.1, q.2.subst l e), st.c⟩

/-- A delta step on `l` from `src` giving `e`, with `R` the summed and `U` the boundary
labels. -/
def dOK (S : Shape) (D : ℕ → ℕ) (U R : List ℕ) (st : SSt) (l : ℕ) (src : Src) (e : LX) :
    Bool :=
  R.contains l && blook st.bs l == none && sin st src && solve S D st.bs l src == some e &&
    !e.vars.contains l && e.vars.all (fun x => R.contains x || U.contains x) &&
    decide (e.ub D < D l)

theorem dstep_pt (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (D : ℕ → ℕ) (U R : List ℕ) (st : SSt) (l : ℕ) (src : Src) (e : LX)
    (h : dOK S D U R st l src e = true) (τ : ℕ → ℕ)
    (hτ : ∀ x, (x ∈ R ∨ x ∈ U) → x ≠ l → τ x < D x) :
    ∑ j ∈ range (D l), sval S M P C p w st (Function.update τ l j) =
      sval S M P C p w (dapply l e st) τ := by
  simp only [dOK, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', List.all_eq_true,
    Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨_, hfree⟩, hin⟩, hsolve⟩, hnl⟩, hvars⟩, hub⟩ := h
  have hnl' : l ∉ e.vars := by simpa using hnl
  have hj0lt : e.eval τ < D l := by
    have := LX.eval_le_ub D τ e fun x hx => by
      have hxl : x ≠ l := fun h' => hnl' (h' ▸ hx)
      exact hτ x (hvars x hx) hxl
    omega
  rw [Finset.sum_eq_single (e.eval τ)]
  · have hT := tval_step st.bs l e hfree τ
    simp only [sval, dapply, List.map_map, Function.comp_def, hT, cval, LX.eval_subst]
  · intro j hj hne
    apply sval_zero S M P C p w st src hin
    by_contra hv
    have := solve_sound S M P C p w D st.bs l src e hsolve (Function.update τ l j)
      (by simpa using Finset.mem_range.1 hj) hv
    rw [Function.update_self, LX.eval_update τ l j e hnl'] at this
    exact hne this
  · intro h; exact absurd (Finset.mem_range.2 hj0lt) h

/-! ## Geometric steps -/

/-- The other leg expression of a Fourier box with one leg reading exactly `l`. -/
def fleg (bs : List (ℕ × LX)) (l : ℕ) (n : Node) : Option LX :=
  match n.kind, n.outs, n.ins with
  | .F, [o], [i] =>
    if texpr bs o = .v l ∧ !(texpr bs i).vars.contains l then some (texpr bs i)
    else if texpr bs i = .v l ∧ !(texpr bs o).vars.contains l then some (texpr bs o)
    else none
  | _, _, _ => none

theorem valF_eq (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (τ : ℕ → ℕ) (o i dim ph : ℕ) :
    tmv p w (valW S M P C τ ⟨.F, [o], [i], dim, ph⟩) =
      (w : ZMod p) ^ (M / S.dimv dim * (τ o * τ i)) := by
  simp [valW, isBS, tensorW, tensorT, tmv]

theorem fleg_val (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (bs : List (ℕ × LX)) (l : ℕ) (n : Node) (X : LX) (h : fleg bs l n = some X)
    (τ : ℕ → ℕ) (j : ℕ) :
    tmv p w (valW S M P C (tval bs (Function.update τ l j)) n) =
      ((w : ZMod p) ^ (M / S.dimv n.dim)) ^ (j * X.eval τ) := by
  obtain ⟨kind, outs, ins, dim, ph⟩ := n
  cases kind <;> try (simp [fleg] at h; done)
  rcases outs with _ | ⟨o, _ | ⟨_, _⟩⟩ <;> rcases ins with _ | ⟨i, _ | ⟨_, _⟩⟩ <;>
    try (simp [fleg] at h; done)
  unfold fleg at h
  dsimp only at h
  rw [valF_eq, ← pow_mul]
  split_ifs at h with h1 h2
  · simp only [Option.some.injEq] at h; subst h
    have ho : tval bs (Function.update τ l j) o = j := by
      simp [tval, h1.1, LX.eval]
    have hi : tval bs (Function.update τ l j) i = (texpr bs i).eval τ :=
      LX.eval_update τ l j _ (by simpa using h1.2)
    rw [ho, hi]
  · simp only [Option.some.injEq] at h; subst h
    have hi : tval bs (Function.update τ l j) i = j := by
      simp [tval, h2.1, LX.eval]
    have ho : tval bs (Function.update τ l j) o = (texpr bs o).eval τ :=
      LX.eval_update τ l j _ (by simpa using h2.2)
    rw [ho, hi, mul_comm ((texpr bs o).eval τ) j]

theorem geom_root (p N a : ℕ) [Fact p.Prime] (u : ZMod p) (hu : IsPrimitiveRoot u N)
    (hN : 0 < N) : ∑ j ∈ range N, u ^ (j * a) = if N ∣ a then (N : ZMod p) else 0 := by
  split_ifs with hd
  · have : u ^ a = 1 := (hu.pow_eq_one_iff_dvd a).2 hd
    simp [pow_mul', this]
  · have hne : u ^ a ≠ 1 := fun h => hd ((hu.pow_eq_one_iff_dvd a).1 h)
    have hgeom := geom_sum_mul (u ^ a) N
    have hN1 : (u ^ a) ^ N = 1 := by rw [← pow_mul, mul_comm, pow_mul, hu.pow_eq_one, one_pow]
    rw [hN1, sub_self] at hgeom
    have hsub : u ^ a - 1 ≠ 0 := sub_ne_zero.2 hne
    have := (mul_eq_zero.1 hgeom).resolve_right hsub
    simpa [pow_mul'] using this

/-- A geometric step on `l` between the Fourier boxes `n₁`, `n₂`, the others `ns'`. -/
def gOK (S : Shape) (M : ℕ) (D : ℕ → ℕ) (R : List ℕ) (st : SSt) (l : ℕ) (n₁ n₂ : Node)
    (ns' : List Node) (X Z : LX) : Bool :=
  R.contains l && blook st.bs l == none && fleg st.bs l n₁ == some X &&
    fleg st.bs l n₂ == some Z && S.dimv n₁.dim == D l && S.dimv n₂.dim == D l &&
    decide (0 < D l) && M % D l == 0 && st.ns.isPerm (n₁ :: n₂ :: ns') &&
    ns'.all (fun n => !nmen st.bs l n) && st.cs.all (fun q => !q.2.vars.contains l)

/-- The state after a geometric step. -/
def gapply (p : ℕ) (D : ℕ → ℕ) (l : ℕ) (ns' : List Node) (X Z : LX) (st : SSt) : SSt :=
  ⟨st.bs, ns', st.cs ++ [(D l, .add X Z)], st.c * D l % p⟩

theorem gstep_pt (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) [Fact p.Prime] (hw : IsPrimitiveRoot (w : ZMod p) M) (hM : 0 < M) (D : ℕ → ℕ)
    (R : List ℕ) (st : SSt) (l : ℕ) (n₁ n₂ : Node) (ns' : List Node) (X Z : LX)
    (h : gOK S M D R st l n₁ n₂ ns' X Z = true) (τ : ℕ → ℕ) :
    ∑ j ∈ range (D l), sval S M P C p w st (Function.update τ l j) =
      sval S M P C p w (gapply p D l ns' X Z st) τ := by
  simp only [gOK, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.isPerm_iff,
    List.all_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨_, _⟩, h1⟩, h2⟩, hd1⟩, hd2⟩, hN⟩, hMN⟩, hperm⟩, hns⟩, hcs⟩ := h
  obtain ⟨r, hr⟩ := Nat.dvd_of_mod_eq_zero hMN
  have hu : IsPrimitiveRoot ((w : ZMod p) ^ (M / D l)) (D l) := by
    have hMr : M / D l = r := by rw [hr, Nat.mul_div_cancel_left _ hN]
    rw [hMr]
    exact hw.pow hM (by rw [hr, mul_comm])
  have e1 := fun j => fleg_val S M P C p w st.bs l n₁ X h1 τ j
  have e2 := fun j => fleg_val S M P C p w st.bs l n₂ Z h2 τ j
  rw [hd1] at e1
  rw [hd2] at e2
  have hA : ∀ j, (ns'.map fun n => tmv p w (valW S M P C (tval st.bs (Function.update τ l j)) n)).prod
      = (ns'.map fun n => tmv p w (valW S M P C (tval st.bs τ) n)).prod := by
    intro j
    congr 1
    apply List.map_congr_left
    intro n hn
    rw [node_frame S M P C st.bs l j n (hns n hn) τ]
  have hB : ∀ j, (st.cs.map fun q => cval p q (Function.update τ l j)).prod =
      (st.cs.map fun q => cval p q τ).prod := by
    intro j
    congr 1
    apply List.map_congr_left
    intro q hq
    exact cons_frame p q l j (hcs q hq) τ
  have hval : ∀ j, sval S M P C p w st (Function.update τ l j) =
      (st.c : ZMod p) * (ns'.map fun n => tmv p w (valW S M P C (tval st.bs τ) n)).prod *
        (st.cs.map fun q => cval p q τ).prod *
        ((w : ZMod p) ^ (M / D l)) ^ (j * (X.eval τ + Z.eval τ)) := by
    intro j
    unfold sval
    rw [(hperm.map _).prod_eq, List.map_cons, List.map_cons, List.prod_cons, List.prod_cons,
      e1, e2, hA, hB, mul_add, pow_add]
    ring
  simp only [hval, ← Finset.mul_sum, geom_root p (D l) _ _ hu hN]
  unfold sval gapply
  simp only [List.map_append, List.map_cons, List.map_nil, List.prod_append, List.prod_cons,
    List.prod_nil, cval, LX.eval, ZMod.natCast_mod, Nat.cast_mul]
  by_cases hd : D l ∣ X.eval τ + Z.eval τ <;>
    simp only [hd, ite_true, ite_false, mul_one, mul_zero] <;> ring

/-! ## Steps over sums -/

theorem sumR_step (D : ℕ → ℕ) (p : ℕ) (f g : (ℕ → ℕ) → ZMod p) (R R' : List ℕ) (l : ℕ)
    (σ : ℕ → ℕ) (hR : R.Perm (R' ++ [l]))
    (hpt : ∀ τ, (∀ x ∉ R', τ x = σ x) → (∀ x ∈ R', τ x < D x) →
      ∑ j ∈ range (D l), f (Function.update τ l j) = g τ) :
    sumR D R f σ = sumR D R' g σ := by
  rw [sumR_perm D hR, sumR_append]
  apply sumR_congr
  intro τ hτ hr
  simp only [sumR]
  exact hpt τ hτ hr

/-- A delta step over sums. -/
theorem sym_dstep (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (U R R' : List ℕ) (st st' : SSt) (l : ℕ) (src : Src) (e : LX) (σ : ℕ → ℕ)
    (hU : ∀ x ∈ U, σ x < S.dim x) (hR : R.Perm (R' ++ [l]))
    (h : dOK S S.dim U R st l src e = true) (hst : dapply l e st = st') :
    sumR S.dim R (sval S M P C p w st) σ = sumR S.dim R' (sval S M P C p w st') σ := by
  subst hst
  apply sumR_step S.dim p _ _ R R' l σ hR
  intro τ hτ hr
  apply dstep_pt S M P C p w S.dim U R st l src e h τ
  intro x hx hxl
  by_cases hm : x ∈ R'
  · exact hr x hm
  · rw [hτ x hm]
    rcases hx with hx | hx
    · have := hR.subset hx
      simp only [List.mem_append, List.mem_singleton] at this
      rcases this with h' | h'
      · exact absurd h' hm
      · exact absurd h' hxl
    · exact hU x hx

/-- A geometric step over sums. -/
theorem sym_gstep (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) [Fact p.Prime] (hw : IsPrimitiveRoot (w : ZMod p) M) (hM : 0 < M)
    (R R' : List ℕ) (st st' : SSt) (l : ℕ) (n₁ n₂ : Node) (ns' : List Node) (X Z : LX)
    (σ : ℕ → ℕ) (hR : R.Perm (R' ++ [l])) (h : gOK S M S.dim R st l n₁ n₂ ns' X Z = true)
    (hst : gapply p S.dim l ns' X Z st = st') :
    sumR S.dim R (sval S M P C p w st) σ = sumR S.dim R' (sval S M P C p w st') σ := by
  subst hst
  exact sumR_step S.dim p _ _ R R' l σ hR fun τ _ _ =>
    gstep_pt S M P C p w hw hM S.dim R st l n₁ n₂ ns' X Z h τ

/-- A network's formal entry mod `p` as the value of its first substitution state. -/
theorem sym_start (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (N : Net) (σ : ℕ → ℕ) :
    tmv p w (sumLT S.dim N.int (prodW S M P C N.nodes) σ) =
      sumR S.dim N.int (sval S M P C p w ⟨[], N.nodes, [], 1⟩) σ := by
  rw [tmv_sumLT]
  apply sumR_congr
  intro τ _ _
  simp only [sval, tval_nil, tmv_prodW, List.map_nil, List.prod_nil, Nat.cast_one, one_mul,
    mul_one]

/-! ## The remaining labels by residue tables -/

/-- The Z spider's entry as `tZT` gives it, read off the first leg. -/
def tZQ (Pk : ℕ → ℕ × List ℤ) (d : ℕ) : List ℕ → Terms
  | [] => tZT Pk d []
  | v :: rest => if rest.all (· == v) && decide (v < d) then [(1, (Pk v).1, (Pk v).2)] else []

theorem tZQ_eq (Pk : ℕ → ℕ × List ℤ) (d : ℕ) (vs : List ℕ) : tZQ Pk d vs = tZT Pk d vs := by
  cases vs with
  | nil => rfl
  | cons v rest =>
    simp only [tZQ, tZT]
    induction d with
    | zero => simp
    | succ d ih =>
      rw [List.range_succ, List.flatMap_append, ← ih]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.all_cons, beq_iff_eq]
      by_cases hv : v = d
      · subst hv
        by_cases hr : rest.all (· == v) = true
        · simp [hr]
        · simp [hr]
      · by_cases hlt : v < d
        · have : ¬ (v < d + 1) = False := by simp; omega
          simp [hv, hlt, show v < d + 1 by omega]
        · simp [hv, hlt, show ¬ v < d + 1 by omega]

/-- A node's entry as `valW` gives it, a Z spider read off its first leg. -/
def valQ (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (τ : ℕ → ℕ)
    (n : Node) : Terms :=
  if n.kind = .Z then tZQ (P n.ph) (S.dimv n.dim) (n.outs.map τ ++ n.ins.map τ)
  else valW S M P C τ n

theorem valQ_eq (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (τ : ℕ → ℕ) (n : Node) : valQ S M P C τ n = valW S M P C τ n := by
  unfold valQ
  split_ifs with h
  · obtain ⟨kind, outs, ins, dim, ph⟩ := n
    simp only at h
    subst h
    rw [tZQ_eq]
    rfl
  · rfl

/-- The residue factor of a group of nodes under bindings, its labels in `U` read from `σ`. -/
def leafS (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (U : List ℕ) (σ : ℕ → ℕ) (bs : List (ℕ × LX)) (g : List Node) (F : RFac) : Bool :=
  g.all (fun n => (nlegs n).all fun x =>
      (texpr bs x).vars.all fun y => F.scope.contains y || U.contains y) &&
    (assigns S.dim F.scope).all fun τ =>
      F.at τ == (g.map fun n => tmN p w M (valQ S M P C (tval bs (pin U σ τ)) n)).foldr
        (fun a b => a * b % p) 1

/-- A constraint's residue factor, its labels in `U` read from `σ`. -/
def leafQ (p : ℕ) (D : ℕ → ℕ) (U : List ℕ) (σ : ℕ → ℕ) (q : ℕ × LX) (F : RFac) : Bool :=
  q.2.vars.all (fun y => F.scope.contains y || U.contains y) &&
    (assigns D F.scope).all fun τ =>
      F.at τ == if q.1 ∣ q.2.eval (pin U σ τ) then 1 else 0

/-- The residue factors of a state: its nodes in the groups `gss`, its constraints and its
constant. -/
def leavesOK (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (U : List ℕ) (σ : ℕ → ℕ) (st : SSt) (gss : List (List Node)) (ln lq : List RFac) : Bool :=
  st.ns.isPerm gss.flatten && ln.length == gss.length &&
    (gss.zip ln).all (fun q => leafS S M P C p w U σ st.bs q.1 q.2) &&
    lq.length == st.cs.length && (st.cs.zip lq).all (fun q => leafQ p S.dim U σ q.1 q.2)

theorem pin_agree (U : List ℕ) (σ τ ρ : ℕ → ℕ) (F : RFac) (e : LX)
    (he : ∀ y ∈ e.vars, (F.scope.contains y || U.contains y) = true)
    (hρ : ∀ y ∈ F.scope, ρ y = τ y) (hU : ∀ y ∈ U, τ y = σ y) :
    e.eval τ = e.eval (pin U σ ρ) := by
  apply LX.eval_congr
  intro y hy
  simp only [Bool.or_eq_true, List.contains_iff_mem] at he
  unfold pin
  by_cases hu : y ∈ U
  · simp [hu, hU y hu]
  · have : y ∈ F.scope := (he y hy).resolve_right hu
    simp [hu, hρ y this]

theorem foldr_mod_cast (p : ℕ) (xs : List ℕ) :
    ((xs.foldr (fun a b => a * b % p) 1 : ℕ) : ZMod p) = (xs.map (Nat.cast : ℕ → ZMod p)).prod := by
  induction xs with
  | nil => simp
  | cons a xs ih =>
    simp only [List.foldr_cons, List.map_cons, List.prod_cons]
    rw [ZMod.natCast_mod, Nat.cast_mul, ih]

theorem leafS_val (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms) (p w : ℕ)
    (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M) (hMb : M ≤ 2 ^ 64) (U : List ℕ)
    (σ : ℕ → ℕ) (bs : List (ℕ × LX)) (g : List Node) (F : RFac)
    (h : leafS S M P C p w U σ bs g F = true) (τ : ℕ → ℕ)
    (hτ : ∀ y ∈ F.scope, τ y < S.dim y) (hU : ∀ y ∈ U, τ y = σ y) :
    (g.map fun n => tmv p w (valW S M P C (tval bs τ) n)).prod = (F.at τ : ZMod p) := by
  simp only [leafS, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨ρ, hρ, hρτ⟩ := assigns_cover S.dim F.scope τ hτ
  rw [F.at_congr τ ρ fun l hl => (hρτ l hl).symm, h.2 ρ hρ, foldr_mod_cast, List.map_map]
  congr 1
  apply List.map_congr_left
  intro n hn
  simp only [Function.comp_apply]
  rw [tmN_cast p w M hp hw hM0 hMb, valQ_eq]
  congr 1
  apply valW_congr
  intro x hx
  exact pin_agree U σ τ ρ F _ (h.1 n hn x (by simpa [nlegs] using hx)) hρτ hU

theorem leafQ_val (p : ℕ) (D : ℕ → ℕ) (U : List ℕ) (σ : ℕ → ℕ) (q : ℕ × LX) (F : RFac)
    (h : leafQ p D U σ q F = true) (τ : ℕ → ℕ) (hτ : ∀ y ∈ F.scope, τ y < D y)
    (hU : ∀ y ∈ U, τ y = σ y) : cval p q τ = (F.at τ : ZMod p) := by
  simp only [leafQ, Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at h
  obtain ⟨ρ, hρ, hρτ⟩ := assigns_cover D F.scope τ hτ
  rw [F.at_congr τ ρ fun l hl => (hρτ l hl).symm, h.2 ρ hρ, cval,
    pin_agree U σ τ ρ F q.2 h.1 hρτ hU]
  split_ifs <;> simp

theorem prod_leavesS (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M) (hMb : M ≤ 2 ^ 64)
    (U : List ℕ) (σ : ℕ → ℕ) (bs : List (ℕ × LX)) :
    ∀ (gss : List (List Node)) (ln : List RFac), ln.length = gss.length →
      (gss.zip ln).all (fun q => leafS S M P C p w U σ bs q.1 q.2) = true →
      ∀ τ, (∀ F ∈ ln, ∀ y ∈ F.scope, τ y < S.dim y) → (∀ y ∈ U, τ y = σ y) →
        (gss.flatten.map fun n => tmv p w (valW S M P C (tval bs τ) n)).prod =
          rprod p ln τ := by
  intro gss
  induction gss with
  | nil => intro ln hlen _ τ _ _; cases ln <;> simp_all [rprod]
  | cons g gss ih =>
    intro ln hlen hl τ hτ hU
    cases ln with
    | nil => simp at hlen
    | cons F Fs =>
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at hl
      have h1 := ih Fs (by simpa using hlen) hl.2 τ (fun F hF => hτ F (by simp [hF])) hU
      simp only [rprod, List.map_cons, List.prod_cons] at h1 ⊢
      rw [List.flatten_cons, List.map_append, List.prod_append, h1,
        leafS_val S M P C p w hp hw hM0 hMb U σ bs g F hl.1 τ (hτ F (by simp)) hU]

theorem prod_leavesQ (p : ℕ) (D : ℕ → ℕ) (U : List ℕ) (σ : ℕ → ℕ) :
    ∀ (cs : List (ℕ × LX)) (lq : List RFac), lq.length = cs.length →
      (cs.zip lq).all (fun q => leafQ p D U σ q.1 q.2) = true →
      ∀ τ, (∀ F ∈ lq, ∀ y ∈ F.scope, τ y < D y) → (∀ y ∈ U, τ y = σ y) →
        (cs.map fun q => cval p q τ).prod = rprod p lq τ := by
  intro cs
  induction cs with
  | nil => intro lq hlen _ τ _ _; cases lq <;> simp_all [rprod]
  | cons q cs ih =>
    intro lq hlen hl τ hτ hU
    cases lq with
    | nil => simp at hlen
    | cons F Fs =>
      simp only [List.zip_cons_cons, List.all_cons, Bool.and_eq_true] at hl
      have h1 := ih Fs (by simpa using hlen) hl.2 τ (fun F hF => hτ F (by simp [hF])) hU
      simp only [rprod, List.map_cons, List.prod_cons] at h1 ⊢
      rw [h1, leafQ_val p D U σ q F hl.1 τ (hτ F (by simp)) hU]

/-- The constant's residue factor. -/
def kleaf (p c : ℕ) : RFac := ⟨[], RT.leaf (c % p)⟩

/-- The remaining sums of a state by a checked contraction of its residue factors. -/
theorem sym_final (S : Shape) (M : ℕ) (P : ℕ → ℕ → ℕ × List ℤ) (C : ℕ → ℕ → Terms)
    (p w : ℕ) (hp : p ≠ 0) (hw : (w : ZMod p) ^ M = 1) (hM0 : 0 < M) (hMb : M ≤ 2 ^ 64)
    (U R : List ℕ) (σ : ℕ → ℕ) (st : SSt) (gss : List (List Node)) (ln lq : List RFac)
    (hns : st.ns.isPerm gss.flatten = true) (hln : ln.length = gss.length)
    (hl1 : (gss.zip ln).all (fun q => leafS S M P C p w U σ st.bs q.1 q.2) = true)
    (hlq : lq.length = st.cs.length)
    (hl2 : (st.cs.zip lq).all (fun q => leafQ p S.dim U σ q.1 q.2) = true)
    (steps : List (ℕ × RFac)) (final : List RFac)
    (hperm : R.Perm (steps.map Prod.fst).reverse)
    (hchain : chainR p S.dim (ln ++ lq ++ [kleaf p st.c]) steps final = true)
    (hscope : scopesR U (ln ++ lq ++ [kleaf p st.c]) steps = true)
    (hdis : U.all (fun x => !(steps.map Prod.fst).contains x) = true)
    (hσ : ∀ x ∈ U, σ x < S.dim x) :
    sumR S.dim R (sval S M P C p w st) σ = (rfold p final σ : ZMod p) := by
  rw [List.isPerm_iff] at hns
  rw [sumR_perm _ hperm, rfold_cast]
  simp only [List.all_eq_true, Bool.not_eq_true'] at hdis
  rw [← chainR_val p S.dim U steps _ final hchain hscope σ hσ]
  apply sumR_congr
  intro τ hτ hr
  have hrange : ∀ F ∈ ln ++ lq ++ [kleaf p st.c], ∀ y ∈ F.scope, τ y < S.dim y := by
    intro F hF y hy
    rcases scopesR_leaves U _ steps hscope F hF y hy with hb | hs
    · by_cases hm : y ∈ (steps.map Prod.fst).reverse
      · exact hr y hm
      · rw [hτ y hm]; exact hσ y hb
    · exact hr y (by simpa using hs)
  have hU : ∀ y ∈ U, τ y = σ y := by
    intro y hy
    apply hτ y
    have := hdis y hy
    simpa using this
  have e1 := prod_leavesS S M P C p w hp hw hM0 hMb U σ st.bs gss ln hln hl1 τ
    (fun F hF => hrange F (by simp [hF])) hU
  have e2 := prod_leavesQ p S.dim U σ st.cs lq hlq hl2 τ
    (fun F hF => hrange F (by simp [hF])) hU
  unfold sval
  rw [(hns.map _).prod_eq, e1, e2]
  simp only [rprod, List.map_append, List.prod_append, List.map_cons, List.map_nil,
    List.prod_cons, List.prod_nil, mul_one, kleaf, RFac.at, RT.get, ZMod.natCast_mod]
  ring

end ArchytasZX
