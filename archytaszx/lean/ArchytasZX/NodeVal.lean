/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Tower

/-!
# Leg values of a copy

`fv C t d β P`: for legs given as (label, tail chain) pairs, every leg's digits over its tail
at copy `t` of the node chain `C`. The bundled generators at block values and the copies of the
digit sum both read permutations of these lists.
-/

namespace ArchytasZX

/-- Each leg's digits at copy `t`: digit `pos (C ++ c) (t ++ s)` of its value, `s` over its tail. -/
def fv (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) (P : List (ℕ × List ℕ)) : List ℕ :=
  P.flatMap fun q => (cidx mult q.2).map fun s => digit d (pos mult (C ++ q.2) (t ++ s)) (β q.1)

theorem fv_append (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) (P Q : List (ℕ × List ℕ)) :
    fv mult C t d β (P ++ Q) = fv mult C t d β P ++ fv mult C t d β Q := by
  simp [fv, List.flatMap_append]

theorem fv_perm (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) {P Q : List (ℕ × List ℕ)}
    (h : P.Perm Q) : (fv mult C t d β P).Perm (fv mult C t d β Q) := h.flatMap_right _

theorem fv_plain (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) (ls : List ℕ) :
    fv mult C t d β (ls.map (·, [])) = ls.map fun l => digit d (pos mult C t) (β l) := by
  induction ls with
  | nil => rfl
  | cons l ls ih =>
    simp only [fv, List.map_cons, List.flatMap_cons, cidx, List.map_cons, List.map_nil,
      List.append_nil, List.cons_append, List.nil_append] at ih ⊢
    rw [ih]

theorem blk_plain (mult : ℕ → ℕ) (d p : ℕ) (ls : List ℕ) (β : ℕ → ℕ) :
    blkC mult d p (List.replicate ls.length []) (ls.map β) = ls.map fun l => digit d p (β l) := by
  induction ls with
  | nil => rfl
  | cons l ls ih => simp [blkC, List.replicate_succ, size] at ih ⊢; exact ih

theorem blk_append (mult : ℕ → ℕ) (d p : ℕ) (a b : List (List ℕ)) (u v : List ℕ)
    (h : a.length = u.length) : blkC mult d p (a ++ b) (u ++ v) = blkC mult d p a u ++ blkC mult d p b v := by
  simp [blkC, List.zipWith_append h]

/-- Tails bundled at block `pos C t` read their digits. -/
theorem tailVals_blk (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (hd : 0 < d) (ht : t ∈ cidx mult C)
    (β : ℕ → ℕ) (P : List (ℕ × List ℕ)) :
    tailVals mult d (blkC mult d (pos mult C t) (P.map (·.2)) (P.map (β ·.1))) (P.map (·.2)) =
      fv mult C t d β P := by
  induction P with
  | nil => simp [tailVals, blkC, fv]
  | cons q P ih =>
    simp only [tailVals, blkC, List.map_cons, List.zipWith_cons_cons, List.flatten_cons] at ih ⊢
    rw [ih]
    simp only [fv, List.flatMap_cons]
    congr 1
    rw [digsC_pos mult d hd]
    split_ifs with hc
    · obtain ⟨l, c⟩ := q
      simp only at hc; subst hc
      simp [cidx, size, pos]
    · refine List.map_congr_left fun s hs => ?_
      rw [pos_append mult C q.2 t s ht, show d ^ size mult q.2 = d ^ size mult q.2 from rfl,
        show (d ^ size mult q.2) = d ^ size mult q.2 from rfl]
      have := digit_digit d (size mult q.2) (pos mult C t) (pos mult q.2 s) (β q.1) hd
        (pos_lt mult q.2 s hs)
      rw [this, Nat.mul_comm]

theorem lookup_isNone_iff {β : Type*} (l : List (ℕ × β)) (a : ℕ) :
    (l.lookup a).isNone = true ↔ ∀ p ∈ l, p.1 ≠ a := by
  induction l with
  | nil => simp
  | cons q l ih =>
    obtain ⟨b, x⟩ := q
    by_cases h : a = b
    · subst h; simp [List.lookup]
    · have hb : (a == b) = false := by simpa using h
      simp only [List.lookup, hb, List.mem_cons, forall_eq_or_imp, ih]
      exact ⟨fun h' => ⟨Ne.symm h, h'⟩, fun h' => h'.2⟩

/-- The legs kept in place and the sorted tails of `bundleNode` are the legs, permuted. -/
theorem keep_tails_perm (n : ℕ) (lO : ℕ → ℕ) (c : ℕ → List ℕ) (s0 : ℕ)
    (f : List ℕ × ℕ × ℕ → List ℕ × ℕ × ℕ → Prop) [DecidableRel f] :
    ((List.range n).map fun j => (lO j, c j)).Perm
      ((((List.range n).filter fun j => ((((List.range n).map fun j => (j, c j)).filter
          fun p => p.2 ≠ []).lookup j).isNone).map fun j => ((List.range n).map lO).getD j 0).map (·, []) ++
        ((((((List.range n).map fun j => (j, c j)).filter fun p => p.2 ≠ []).map
          fun p => (p.2, s0, p.1)).insertionSort f).map
          fun t => (((List.range n).map lO).getD t.2.2 0, t.1))) := by
  set tl := ((List.range n).map fun j => (j, c j)).filter fun p => p.2 ≠ []
  have hlk : ∀ j ∈ List.range n, (tl.lookup j).isNone = true ↔ c j = [] := fun j hj => by
    rw [lookup_isNone_iff]
    simp only [tl, List.mem_filter, List.mem_map, List.mem_range, decide_eq_true_eq]
    constructor
    · intro h; by_contra hc; exact h (j, c j) ⟨⟨j, List.mem_range.1 hj, rfl⟩, hc⟩ rfl
    · rintro hc p ⟨⟨j', hj', rfl⟩, hne⟩ e; simp only at e; subst e; exact hne hc
  have hget : ∀ j ∈ List.range n, ((List.range n).map lO).getD j 0 = lO j := fun j hj => by
    rw [List.getD_eq_getElem _ _ (by simpa using List.mem_range.1 hj)]; simp
  refine List.Perm.trans ?_ (List.Perm.append_left _ ((List.perm_insertionSort f _).map _).symm)
  rw [List.map_map, List.map_map]
  have e1 : ((List.range n).filter fun j => (tl.lookup j).isNone).map
      ((fun x => (x, ([] : List ℕ))) ∘ fun j => ((List.range n).map lO).getD j 0) =
      ((List.range n).filter fun j => decide (c j = [])).map fun j => (lO j, c j) := by
    rw [List.filter_congr (q := fun j => decide (c j = [])) fun j hj => by rw [Bool.eq_iff_iff, hlk j hj]; simp]
    refine List.map_congr_left fun j hj => ?_
    have hj' := (List.mem_filter.1 hj)
    simp only [Function.comp, hget j hj'.1, of_decide_eq_true hj'.2]
  have e2 : tl.map ((fun t => (((List.range n).map lO).getD t.2.2 0, t.1)) ∘ fun p => (p.2, s0, p.1)) =
      ((List.range n).filter fun j => !decide (c j = [])).map fun j => (lO j, c j) := by
    simp only [tl, List.filter_map, List.map_map]
    refine List.map_congr_left fun j hj => ?_
    simp only [Function.comp, hget j (List.mem_filter.1 hj).1]
  rw [e1, e2, ← List.map_append]
  exact ((List.filter_append_perm _ _).map _).symm

theorem bundled_splitC (mult : ℕ → ℕ) (d : ℕ) (A B : List ℕ) (cs : List (List ℕ)) (h : B.length = cs.length) :
    bundled mult d (A ++ B) cs = A ++ tailVals mult d B cs := by
  simp only [bundled, List.length_append, h, Nat.add_sub_cancel, List.take_left', List.drop_left']

section bnd

variable (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (hd : 0 < d) (ht : t ∈ cidx mult C) (β : ℕ → ℕ)
  (p : ℕ) (hp : p = pos mult C t)

/-- The legs given as label and tail chain. -/
def pairs (n : ℕ) (l : ℕ → ℕ) (c : ℕ → List ℕ) : List (ℕ × List ℕ) := (List.range n).map fun j => (l j, c j)

/-- The tails of a side, as `bnode` lists them. -/
def tls (n : ℕ) (c : ℕ → List ℕ) : List (ℕ × List ℕ) :=
  ((List.range n).map fun j => (j, c j)).filter fun q => q.2 ≠ []

/-- The kept legs of a side. -/
def keepL (n : ℕ) (l : ℕ → ℕ) (c : ℕ → List ℕ) : List ℕ :=
  ((List.range n).filter fun j => ((tls n c).lookup j).isNone).map fun j => ((List.range n).map l).getD j 0

/-- The sorted tails of a side, tagged `s0`. -/
def sortT (n : ℕ) (c : ℕ → List ℕ) (s0 : ℕ) : List (List ℕ × ℕ × ℕ) :=
  ((tls n c).map fun q => (q.2, s0, q.1)).insertionSort fun a b => tailLe a b

theorem side_perm (n : ℕ) (l : ℕ → ℕ) (c : ℕ → List ℕ) (s0 : ℕ) :
    (pairs n l c).Perm ((keepL n l c).map (·, []) ++
      (sortT n c s0).map fun t => (((List.range n).map l).getD t.2.2 0, t.1)) :=
  keep_tails_perm n l c s0 _

include hd ht hp in
/-- One side of an `X` or `W` bundle: kept legs and sorted tails at block `p`. -/
theorem side_vals (n : ℕ) (l : ℕ → ℕ) (c : ℕ → List ℕ) (s0 : ℕ) :
    (bundled mult d (blkC mult d p (List.replicate (keepL n l c).length [] ++ (sortT n c s0).map (·.1))
      ((keepL n l c ++ (sortT n c s0).map fun t => ((List.range n).map l).getD t.2.2 0).map β))
      ((sortT n c s0).map (·.1))).Perm (fv mult C t d β (pairs n l c)) := by
  subst hp
  set K := keepL n l c
  set S := sortT n c s0
  set Pb := S.map fun t => (((List.range n).map l).getD t.2.2 0, t.1)
  have hS1 : S.map (·.1) = Pb.map (·.2) := by simp [Pb, List.map_map, Function.comp_def]
  have hS2 : (S.map fun t => ((List.range n).map l).getD t.2.2 0).map β = Pb.map (β ·.1) := by
    simp [Pb, List.map_map, Function.comp_def]
  rw [List.map_append, blk_append _ _ _ _ _ _ _ (by simp), blk_plain, hS1, hS2,
    bundled_splitC _ _ _ _ _ (by simp [blkC, List.length_zipWith]), tailVals_blk mult C t d hd ht,
    ← fv_plain, ← fv_append]
  exact fv_perm _ _ _ _ _ (side_perm n l c s0).symm

theorem sortT_tag (n : ℕ) (c : ℕ → List ℕ) (s0 : ℕ) : ∀ x ∈ sortT n c s0, x.2.1 = s0 := by
  intro x hx
  have := (List.perm_insertionSort _ _).mem_iff.1 hx
  obtain ⟨q, -, rfl⟩ := List.mem_map.1 this
  rfl

include hd ht hp in
/-- A `Z` bundle: both sides' kept legs and all tails sorted together, at block `p`. -/
theorem z_vals (nO : ℕ) (lO : ℕ → ℕ) (cO : ℕ → List ℕ) (nI : ℕ) (lI : ℕ → ℕ) (cI : ℕ → List ℕ) :
    let both := (sortT nO cO 0 ++ sortT nI cI 1).insertionSort fun a b => tailLe a b
    let lab := fun (x : List ℕ × ℕ × ℕ) => if x.2.1 = 0 then ((List.range nO).map lO).getD x.2.2 0
      else ((List.range nI).map lI).getD x.2.2 0
    (blkC mult d p (List.replicate (keepL nO lO cO).length []) ((keepL nO lO cO).map β) ++
      bundled mult d (blkC mult d p (List.replicate (keepL nI lI cI).length [] ++ both.map (·.1))
        ((keepL nI lI cI ++ both.map lab).map β)) (both.map (·.1))).Perm
      (fv mult C t d β (pairs nO lO cO) ++ fv mult C t d β (pairs nI lI cI)) := by
  intro both lab
  subst hp
  set KO := keepL nO lO cO
  set KI := keepL nI lI cI
  set Pb := both.map fun x => (lab x, x.1)
  have hS1 : both.map (·.1) = Pb.map (·.2) := by simp [Pb, List.map_map, Function.comp_def]
  have hS2 : (both.map lab).map β = Pb.map (β ·.1) := by simp [Pb, List.map_map, Function.comp_def]
  rw [blk_plain, List.map_append, blk_append _ _ _ _ _ _ _ (by simp), blk_plain, hS1, hS2,
    bundled_splitC _ _ _ _ _ (by simp [blkC, List.length_zipWith]), tailVals_blk mult C t d hd ht,
    ← fv_plain, ← fv_plain]
  set SO := (sortT nO cO 0).map fun x => (((List.range nO).map lO).getD x.2.2 0, x.1)
  set SI := (sortT nI cI 1).map fun x => (((List.range nI).map lI).getD x.2.2 0, x.1)
  have hPb : Pb.Perm (SO ++ SI) := by
    refine ((List.perm_insertionSort _ _).map _).trans ?_
    rw [List.map_append]
    refine List.Perm.append (List.Perm.of_eq (List.map_congr_left fun x hx => ?_))
      (List.Perm.of_eq (List.map_congr_left fun x hx => ?_))
    · simp [lab, sortT_tag nO cO 0 x hx]
    · simp [lab, sortT_tag nI cI 1 x hx]
  have h1 := fv_perm mult C t d β (side_perm nO lO cO 0)
  have h2 := fv_perm mult C t d β (side_perm nI lI cI 1)
  rw [fv_append] at h1 h2
  refine List.Perm.trans ?_ (h1.symm.append h2.symm)
  refine (List.Perm.append_left _ (List.Perm.append_left _ (fv_perm mult C t d β hPb))).trans ?_
  rw [fv_append]
  simp only [List.append_assoc]
  refine List.Perm.append_left _ ?_
  rw [← List.append_assoc, ← List.append_assoc]
  exact List.perm_append_comm.append_right _

end bnd

/-- `fv` with each tail in `rcidx` order. -/
def fvR (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) (P : List (ℕ × List ℕ)) : List ℕ :=
  P.flatMap fun q => (rcidx mult q.2).map fun s => digit d (pos mult (C ++ q.2) (t ++ s)) (β q.1)

theorem rcidx_perm (mult : ℕ → ℕ) (c : List ℕ) : (rcidx mult c).Perm (cidx mult c) := by
  have hnd : (rcidx mult c).Nodup := by
    unfold rcidx
    exact (cidx_nodup mult _).map fun a b h => List.reverse_injective h
  exact (List.perm_ext_iff_of_nodup hnd (cidx_nodup mult c)).2 fun x => mem_rcidx mult c x

theorem fvR_perm (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) (P : List (ℕ × List ℕ)) :
    (fvR mult C t d β P).Perm (fv mult C t d β P) :=
  List.Perm.flatMap_left _ fun q _ => (rcidx_perm mult q.2).map _

/-- A copy's leg labels on one side through `ν`, when `ν` reads each digit label's digit. -/
theorem dnode_side (D : Diagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) (o : Bool) (len : ℕ)
    (d : ℕ) (lab : Port → ℕ) (β ν : ℕ → ℕ)
    (hν : ∀ i < len, ∀ s ∈ rcidx ks (D.tail ⟨n.id, o, i⟩), ν (dlab ⟨n.id, o, i⟩ (t ++ s)) =
      digit d (pos ks (D.nodeChain n.id ++ D.tail ⟨n.id, o, i⟩) (t ++ s)) (β (lab ⟨n.id, o, i⟩))) :
    ((List.range len).flatMap fun i => (rcidx ks (D.tail ⟨n.id, o, i⟩)).map fun s =>
      dlab ⟨n.id, o, i⟩ (t ++ s)).map ν =
      fvR ks (D.nodeChain n.id) t d β (pairs len (fun j => lab ⟨n.id, o, j⟩) (fun j => D.tail ⟨n.id, o, j⟩)) := by
  simp only [fvR, pairs, List.flatMap_map, List.map_flatMap, List.map_map]
  refine List.flatMap_congr fun i hi => List.map_congr_left fun s hs => ?_
  exact hν i (List.mem_range.1 hi) s hs

/-- A generator of the engine's diagrams: no copies, no bundled legs. -/
def Kind.plain : Kind → Bool
  | .ZB _ | .XB _ _ | .WB _ | .Bun _ _ => false
  | _ => true

theorem plain_outCs (b : Kind) (h : b.plain = true) (n : ℕ) : b.outCs n = List.replicate n [] := by
  cases b <;> simp_all [Kind.plain, Kind.outCs]

theorem plain_inCs (b : Kind) (h : b.plain = true) (n : ℕ) : b.inCs n = List.replicate n [] := by
  cases b <;> simp_all [Kind.plain, Kind.inCs]

theorem tls_nil (n : ℕ) (c : ℕ → List ℕ) : tls n c = [] ↔ ∀ j < n, c j = [] := by
  simp only [tls, List.filter_eq_nil_iff, List.mem_map, List.mem_range, decide_eq_true_eq,
    not_not]
  constructor
  · intro h j hj; exact h (j, c j) ⟨j, hj, rfl⟩
  · rintro h _ ⟨j, hj, rfl⟩; exact h j hj

theorem tower_ne (b : Kind) (C : List ℕ) (h : C ≠ [] ∨ (b ≠ .B ∧ b ≠ .S)) :
    C.foldr Kind.Bun b ≠ .B ∧ C.foldr Kind.Bun b ≠ .S := by
  cases C with
  | nil => simpa using h
  | cons m C => simp

/-- A node copied along `C` is the product over the copies of its generator at block values. -/
theorem val_tower (E : Env) (b : Kind) (C : List ℕ) (hb : C ≠ [] ∨ (b ≠ .B ∧ b ≠ .S))
    (os is : List ℕ) (dim ph : ℕ) (β : ℕ → ℕ) (hd : 0 < E.dimv dim)
    (hos : ∀ p ∈ (b.outCs os.length).zip (os.map β), p.2 < E.dimv dim ^ size E.mult (C ++ p.1))
    (his : ∀ p ∈ (b.inCs is.length).zip (is.map β), p.2 < E.dimv dim ^ size E.mult (C ++ p.1)) :
    val E β ⟨C.foldr Kind.Bun b, os, is, dim, ph⟩ = ((cidx E.mult C).map fun t =>
      tensor E b (E.dimv dim) ph (blkC E.mult (E.dimv dim) (pos E.mult C t) (b.outCs os.length) (os.map β))
        (blkC E.mult (E.dimv dim) (pos E.mult C t) (b.inCs is.length) (is.map β))).prod := by
  rw [val_tensor E β _ (tower_ne b C hb)]
  have := tensor_towerC E b (E.dimv dim) ph hd C (os.map β) (is.map β) (by simpa using hos)
    (by simpa using his)
  simpa using this

/-- A node's legs as label and chain pairs, outputs then inputs. -/
def bpairs (m : Node) : List (ℕ × List ℕ) :=
  m.outs.zip (m.kind.outCs m.outs.length) ++ m.ins.zip (m.kind.inCs m.ins.length)

theorem zip_replicateC {α β : Type*} (l : List α) (b : β) :
    l.zip (List.replicate l.length b) = l.map (·, b) := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.replicate_succ, ih]

theorem pairs_plain (n : ℕ) (l : ℕ → ℕ) (c : ℕ → List ℕ) (h : tls n c = []) :
    pairs n l c = ((List.range n).map l).map (·, []) := by
  rw [tls_nil] at h
  simp only [pairs, List.map_map]
  exact List.map_congr_left fun j hj => by simp [h j (List.mem_range.1 hj)]

section bundle

variable (m : Node) (nO : ℕ) (lO : ℕ → ℕ) (cO : ℕ → List ℕ) (nI : ℕ) (lI : ℕ → ℕ) (cI : ℕ → List ℕ)
  (hmo : m.outs = (List.range nO).map lO) (hmi : m.ins = (List.range nI).map lI)

include hmo hmi in
theorem keepL_eq : keepL nO lO cO = ((List.range m.outs.length).filter fun j =>
    ((tls nO cO).lookup j).isNone).map fun j => m.outs.getD j 0 := by
  simp [keepL, hmo]

include hmo hmi in
theorem bpairs_plain (hk : m.kind.plain = true) (hO : tls nO cO = []) (hI : tls nI cI = []) :
    bpairs m = pairs nO lO cO ++ pairs nI lI cI := by
  rw [bpairs, plain_outCs _ hk, plain_inCs _ hk, zip_replicateC, zip_replicateC, hmo, hmi,
    pairs_plain _ _ _ hO, pairs_plain _ _ _ hI]

/-- The `Z` tails of both sides, sorted together. -/
def bothZ : List (List ℕ × ℕ × ℕ) :=
  (sortT nO cO 0 ++ sortT nI cI 1).insertionSort fun a b => tailLe a b

/-- A sorted tail's label. -/
def labZ (x : List ℕ × ℕ × ℕ) : ℕ :=
  if x.2.1 = 0 then ((List.range nO).map lO).getD x.2.2 0 else ((List.range nI).map lI).getD x.2.2 0

include hmo hmi in
theorem bundle_Z (hz : m.kind = .Z) (hne : ¬(tls nO cO = [] ∧ tls nI cI = [])) :
    bundleNode m (tls nO cO) (tls nI cI) = { m with
      kind := .ZB ((bothZ nO cO nI cI).map (fun x => x.1))
      outs := keepL nO lO cO
      ins := keepL nI lI cI ++ (bothZ nO cO nI cI).map (labZ nO lO nI lI) } := by
  unfold bundleNode
  rw [if_neg hne, hz]
  simp only [keepL, bothZ, labZ, sortT, hmo, hmi, List.length_map, List.length_range]
  rfl

include hmo hmi in
theorem bundle_X (hx : m.kind = .X) (hne : ¬(tls nO cO = [] ∧ tls nI cI = [])) :
    bundleNode m (tls nO cO) (tls nI cI) = { m with
      kind := .XB ((sortT nO cO 0).map (fun x => x.1)) ((sortT nI cI 1).map (fun x => x.1))
      outs := keepL nO lO cO ++ (sortT nO cO 0).map (fun t => ((List.range nO).map lO).getD t.2.2 0)
      ins := keepL nI lI cI ++ (sortT nI cI 1).map (fun t => ((List.range nI).map lI).getD t.2.2 0) } := by
  unfold bundleNode
  rw [if_neg hne, hx]
  simp only [keepL, sortT, hmo, hmi, List.length_map, List.length_range]

include hmo hmi in
theorem bundle_W (hw : m.kind = .W) (hne : ¬(tls nO cO = [] ∧ tls nI cI = [])) :
    bundleNode m (tls nO cO) (tls nI cI) = { m with
      kind := .WB ((sortT nO cO 0).map (fun x => x.1))
      outs := keepL nO lO cO ++ (sortT nO cO 0).map (fun t => ((List.range nO).map lO).getD t.2.2 0) } := by
  unfold bundleNode
  rw [if_neg hne, hw]
  simp only [keepL, sortT, hmo, hmi, List.length_map, List.length_range]

theorem bothZ_perm : ((bothZ nO cO nI cI).map fun x => (labZ nO lO nI lI x, x.1)).Perm
    ((sortT nO cO 0).map (fun x => (((List.range nO).map lO).getD x.2.2 0, x.1)) ++
      (sortT nI cI 1).map (fun x => (((List.range nI).map lI).getD x.2.2 0, x.1))) := by
  refine ((List.perm_insertionSort _ _).map _).trans ?_
  rw [List.map_append]
  refine List.Perm.append (List.Perm.of_eq (List.map_congr_left fun x hx => ?_))
    (List.Perm.of_eq (List.map_congr_left fun x hx => ?_))
  · simp [labZ, sortT_tag nO cO 0 x hx]
  · simp [labZ, sortT_tag nI cI 1 x hx]

theorem side_zip (K : List ℕ) (S : List (List ℕ × ℕ × ℕ)) (l : ℕ → ℕ) (n : ℕ) :
    (K ++ S.map (fun t => ((List.range n).map l).getD t.2.2 0)).zip
      (List.replicate K.length [] ++ S.map (·.1)) =
    K.map (·, []) ++ S.map fun t => (((List.range n).map l).getD t.2.2 0, t.1) := by
  rw [List.zip_append (by simp), zip_replicateC, List.zip_map']

end bundle

namespace Diagram

/-- The legs of `n` on one side, as label and tail chain. -/
def sidePairs (D : Diagram) (off : ℕ) (n : DNode) (o : Bool) : List (ℕ × List ℕ) :=
  pairs (if o then n.outs.length else n.ins.length) (fun j => D.label off ⟨n.id, o, j⟩)
    (fun j => D.tail ⟨n.id, o, j⟩)

theorem bnode_eq (D : Diagram) (off : ℕ) (n : DNode) : D.bnode off n =
    { bundleNode (n.node (D.label off)) (tls n.outs.length fun j => D.tail ⟨n.id, true, j⟩)
        (tls n.ins.length fun j => D.tail ⟨n.id, false, j⟩) with
      kind := (D.nodeChain n.id).foldr Kind.Bun (bundleNode (n.node (D.label off))
        (tls n.outs.length fun j => D.tail ⟨n.id, true, j⟩)
        (tls n.ins.length fun j => D.tail ⟨n.id, false, j⟩)).kind } := rfl

end Diagram

theorem fvR_plain (mult : ℕ → ℕ) (C t : List ℕ) (d : ℕ) (β : ℕ → ℕ) (n : ℕ) (l : ℕ → ℕ)
    (c : ℕ → List ℕ) (h : tls n c = []) :
    fvR mult C t d β (pairs n l c) = (List.range n).map fun j => digit d (pos mult C t) (β (l j)) := by
  rw [tls_nil] at h
  simp only [fvR, pairs, List.flatMap_map]
  rw [List.map_eq_flatMap]
  refine List.flatMap_congr fun j hj => ?_
  simp [h j (List.mem_range.1 hj), Diagram.rcidx_nil]

theorem zip_map_right' {α β γ : Type*} (cs : List α) (ls : List β) (f : β → γ) :
    cs.zip (ls.map f) = (ls.zip cs).map fun q => (q.2, f q.1) := by
  induction cs generalizing ls with
  | nil => simp
  | cons c cs ih => cases ls with
    | nil => simp
    | cons l ls => simp [ih]

theorem range_of_pairs (mult : ℕ → ℕ) (C : List ℕ) (d : ℕ) (β : ℕ → ℕ) (ls : List ℕ)
    (cs : List (List ℕ)) (P : List (ℕ × List ℕ)) (hP : ∀ q ∈ ls.zip cs, q ∈ P)
    (hr : ∀ q ∈ P, β q.1 < d ^ size mult (C ++ q.2)) :
    ∀ p ∈ cs.zip (ls.map β), p.2 < d ^ size mult (C ++ p.1) := by
  intro p hp
  rw [zip_map_right'] at hp
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
  exact hr q (hP q hq)

end ArchytasZX
