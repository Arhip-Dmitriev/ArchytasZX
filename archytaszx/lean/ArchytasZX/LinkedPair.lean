/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedFirst

/-!
# Pair boxes on linked diagrams

An `LPair` is a pair box over the outermost node box `box`: one pair phase per pair of copies
`j < m`, `ex (E.phase i (Nat.pair x y))` at the phase index
`i = Nat.pair ph (Nat.pair (j + rj.1 * ks rj.2) (m + rm.1 * ks rm.2))`, `x` the digit of the leg
`left` at copy `[j]` and `y` that of the leg `right` at copy `[m]` (swapped with `swap`). With `fixL` the left end is
one leg at copy `[]` and the pair index `j` is `0`, one node per copy `m`; with `fixR` the right
end is one leg at copy `[]` and `m` is `0`, one node per copy `j`. `evalP` is `evalW` with the
node product times the pair product `pairF`.

`PairFirst`: the pairs of `R` after a first-copy peel are, per pair of `L`, the residual pair
reading `j + 1` and `m + 1` and the pairs of copy `0`'s left leg with each residual copy
(`fixL`, reading `m + 1`). `PairLast`: after a last-copy peel, per pair of `L`, the pair itself
and the pairs of each kept copy with the last copy's right leg (`fixR`, reading `m := ks c`).
`LPair.val_first` and `LPair.val_last` split the pair product over `j < m ≤ k`; `evalP_first`,
`evalP_peel` and their flat forms `evalP_peel_first_flat`, `evalP_peel_flat` are `evalX_first`,
`evalX_peel` and their flat forms with the pairs.
-/

namespace ArchytasZX

/-- A pair box template over the outermost node box `box`. -/
structure LPair where
  box : ℕ
  left : Port
  right : Port
  ph : ℕ
  swap : Bool := false
  fixL : Bool := false
  fixR : Bool := false
  rj : ℕ × ℕ := (0, 0)
  rm : ℕ × ℕ := (0, 0)
  deriving DecidableEq, Repr

namespace LPair

/-- The phase index of the pair `(j, m)`. -/
def idx (p : LPair) (ks : ℕ → ℕ) (j m : ℕ) : ℕ :=
  Nat.pair p.ph (Nat.pair (j + p.rj.1 * ks p.rj.2) (m + p.rm.1 * ks p.rm.2))

/-- The pair phase of the pair `(j, m)` on the left digits `tl` and right digits `tr`. -/
noncomputable def pv (p : LPair) (E : Env) (ks : ℕ → ℕ) (j m : ℕ) (tl tr : List ℕ)
    (ν : ℕ → ℕ) : ℂ :=
  ex (E.phase (p.idx ks j m) (if p.swap then Nat.pair (ν (dlab p.right tr)) (ν (dlab p.left tl))
    else Nat.pair (ν (dlab p.left tl)) (ν (dlab p.right tr))))

/-- The pair product of `p` at `K` copies. -/
noncomputable def val (p : LPair) (E : Env) (ks : ℕ → ℕ) (K : ℕ) (ν : ℕ → ℕ) : ℂ :=
  if p.fixL then ((List.range K).map fun m => p.pv E ks 0 m [] [m] ν).prod
  else if p.fixR then ((List.range K).map fun j => p.pv E ks j 0 [j] [] ν).prod
  else ((List.range K).map fun m =>
    ((List.range m).map fun j => p.pv E ks j m [j] [m] ν).prod).prod

/-- The residual pair of a first-copy peel. -/
def resid (p : LPair) (m1 : ℕ) : LPair := { p with rj := (1, m1), rm := (1, m1) }

/-- The pairs of copy `0`'s left leg `l0` with the residual copies. -/
def fixLeft (p : LPair) (l0 : Port) (m1 : ℕ) : LPair :=
  { p with left := l0, fixL := true, rm := (1, m1) }

/-- The pairs of the kept copies with the last copy's right leg `r0`. -/
def fixRight (p : LPair) (r0 : Port) (c : ℕ) : LPair :=
  { p with right := r0, fixR := true, rm := (1, c) }

theorem val_node (E : Env) (τ τ' : ℕ → ℕ) (p q : LPair) (ks : ℕ → ℕ) (j m j' m' : ℕ)
    (tl tr tl' tr' : List ℕ) (hs : q.swap = p.swap)
    (hph : q.idx ks j' m' = p.idx ks j m)
    (hl : τ' (dlab p.left tl) = τ (dlab q.left tl'))
    (hr : τ' (dlab p.right tr) = τ (dlab q.right tr')) :
    p.pv E ks j m tl tr τ' = q.pv E ks j' m' tl' tr' τ := by
  simp only [pv, hs, hph, hl, hr]

theorem prod_range_succ_first (k : ℕ) (f : ℕ → ℂ) :
    ((List.range (k + 1)).map f).prod = f 0 * ((List.range k).map fun m => f (m + 1)).prod := by
  rw [List.range_succ_eq_map, List.map_cons, List.prod_cons, List.map_map]
  rfl

theorem prod_range_succ_last (k : ℕ) (f : ℕ → ℂ) :
    ((List.range (k + 1)).map f).prod = ((List.range k).map f).prod * f k := by
  rw [List.range_succ, List.map_append, List.prod_append]
  simp

/-- The pair product at `k + 1` copies is the residual pairs' at `k` (`j + 1`, `m + 1`) times
the pairs of copy `0` with the residual copies. -/
theorem val_first (p : LPair) (hL : p.fixL = false) (hR : p.fixR = false) (hj : p.rj = (0, 0))
    (hm0 : p.rm = (0, 0)) (E : Env) (ks : ℕ → ℕ) (m1 : ℕ) (hm : ks m1 = 1) (k : ℕ)
    (l0 : Port) (τ τ' : ℕ → ℕ)
    (hl0 : τ' (dlab p.left [0]) = τ (dlab l0 []))
    (hl : ∀ j < k, τ' (dlab p.left [j + 1]) = τ (dlab p.left [j]))
    (hr : ∀ j < k, τ' (dlab p.right [j + 1]) = τ (dlab p.right [j])) :
    (p.resid m1).val E ks k τ * (p.fixLeft l0 m1).val E ks k τ = p.val E ks (k + 1) τ' := by
  have rL : (p.resid m1).fixL = false := hL
  have rR : (p.resid m1).fixR = false := hR
  have fL : (p.fixLeft l0 m1).fixL = true := rfl
  simp only [val, rL, rR, fL, hL, hR, Bool.false_eq_true, ite_true, ite_false]
  rw [prod_range_succ_first]
  simp only [List.range_zero, List.map_nil, List.prod_nil, one_mul]
  have inner : ∀ m ∈ List.range k,
      ((List.range (m + 1)).map fun j => p.pv E ks j (m + 1) [j] [m + 1] τ').prod =
        (p.fixLeft l0 m1).pv E ks 0 m [] [m] τ *
          ((List.range m).map fun j => (p.resid m1).pv E ks j m [j] [m] τ).prod := by
    intro m hmk
    have hmk' := List.mem_range.1 hmk
    rw [prod_range_succ_first]
    congr 1
    · refine val_node E τ τ' p _ ks 0 (m + 1) 0 m [0] [m + 1] [] [m] rfl ?_ hl0
        (hr m hmk')
      simp [idx, fixLeft, hj, hm0, hm]
    · refine congrArg List.prod (List.map_congr_left fun j hjm => ?_)
      have hj' := List.mem_range.1 hjm
      refine val_node E τ τ' p _ ks (j + 1) (m + 1) j m [j + 1] [m + 1] [j] [m] rfl
        ?_ (hl j (by omega)) (hr m hmk')
      simp [idx, resid, hj, hm0, hm]
  rw [List.map_congr_left inner, List.prod_map_mul, mul_comm]

/-- The pair product at `k + 1` copies is the pairs among the first `k` copies times the pairs of
those copies with the last copy's right leg. -/
theorem val_last (p : LPair) (hL : p.fixL = false) (hR : p.fixR = false) (hm0 : p.rm = (0, 0))
    (E : Env) (ks : ℕ → ℕ) (c : ℕ) (k : ℕ) (hc : ks c = k) (r0 : Port) (τ τ' : ℕ → ℕ)
    (hrk : τ' (dlab p.right [k]) = τ (dlab r0 []))
    (hl : ∀ j < k, τ' (dlab p.left [j]) = τ (dlab p.left [j]))
    (hr : ∀ j < k, τ' (dlab p.right [j]) = τ (dlab p.right [j])) :
    p.val E ks k τ * (p.fixRight r0 c).val E ks k τ = p.val E ks (k + 1) τ' := by
  have fL : (p.fixRight r0 c).fixL = false := hL
  have fR : (p.fixRight r0 c).fixR = true := rfl
  simp only [val, fL, fR, hL, hR, Bool.false_eq_true, ite_true, ite_false]
  rw [prod_range_succ_last]
  congr 1
  · refine congrArg List.prod (List.map_congr_left fun m hmk => ?_)
    have hm' := List.mem_range.1 hmk
    refine congrArg List.prod (List.map_congr_left fun j hjm => ?_)
    have hj' := List.mem_range.1 hjm
    exact (val_node E τ τ' p p ks j m j m [j] [m] [j] [m] rfl rfl
      (hl j (by omega)) (hr m hm')).symm
  · refine congrArg List.prod (List.map_congr_left fun j hjm => ?_)
    have hj' := List.mem_range.1 hjm
    refine (val_node E τ τ' p (p.fixRight r0 c) ks j k j 0 [j] [k] [j] [] rfl ?_ (hl j hj') hrk).symm
    simp [idx, fixRight, hm0, hc]

end LPair

namespace LDiagram

/-- The pair products of the pairs `ps`, each at its box's copy count. -/
noncomputable def pairF (L : LDiagram) (ps : List LPair) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  (ps.map fun p => p.val E ks (ks (L.multOf p.box)) ν).prod

/-- `evalX` with the pair products of `ps`. -/
noncomputable def evalP (L : LDiagram) (ps : List LPair) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : ℂ :=
  L.evalW E ks (fun ν => L.prodL E ks ν * L.pairF ps E ks ν) σ

theorem evalP_nil (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalP [] E ks σ = L.evalX E ks σ := by
  simp [evalP, pairF, evalX_evalW]

/-- The pairs `ps` of `L` before and `ps'` of `R` after a first-copy peel of the box `bid`. -/
structure PairFirst (L R : LDiagram) (ps ps' : List LPair) (bid mb c m1 : ℕ) (B : List ℕ)
    (cp : ℕ → ℕ) : Prop where
  multL : L.multOf bid = mb
  multR : R.multOf bid = c
  form : ∀ p ∈ ps, p.box = bid ∧ p.fixL = false ∧ p.fixR = false ∧ p.rj = (0, 0) ∧
    p.rm = (0, 0) ∧ p.left ∈ L.base.legs ∧ p.right ∈ L.base.legs ∧ p.left.node ∈ B ∧
    p.right.node ∈ B ∧ L.base.legChain p.left = [mb] ∧ L.base.legChain p.right = [mb]
  eq : ps' = ps.flatMap fun p => [p.resid m1, p.fixLeft (cpP cp p.left) m1]

/-- The pairs `ps` of `L` before and `ps'` of `L'` after a last-copy peel of the box `bid`. -/
structure PairLast (L L' : LDiagram) (ps ps' : List LPair) (bid mb c : ℕ) (B : List ℕ)
    (cp : ℕ → ℕ) : Prop where
  multL : L.multOf bid = mb
  multR : L'.multOf bid = c
  form : ∀ p ∈ ps, p.box = bid ∧ p.fixL = false ∧ p.fixR = false ∧ p.rm = (0, 0) ∧
    p.left ∈ L.base.legs ∧ p.right ∈ L.base.legs ∧ p.left.node ∈ B ∧ p.right.node ∈ B ∧
    L.base.legChain p.left = [mb] ∧ L.base.legChain p.right = [mb]
  eq : ps' = ps.flatMap fun p => [p, p.fixRight (cpP cp p.right) c]

theorem single_mem (ks : ℕ → ℕ) (mb j : ℕ) (hj : j < ks mb) : [j] ∈ cidx ks [mb] := by
  simp [cidx, hj]

theorem dlab_mem (L : LDiagram) (ks : ℕ → ℕ) (q : Port) (mb j : ℕ) (hq : q ∈ L.base.legs)
    (hc : L.base.legChain q = [mb]) (hj : j < ks mb) : dlab q [j] ∈ L.base.dlabs ks :=
  (mem_dlabs _ _ _).2 ⟨q, hq, [j], by rw [hc]; exact single_mem ks mb j hj, rfl⟩

theorem pairF_flatMap (L : LDiagram) (ps : List LPair) (f : LPair → List LPair) (E : Env)
    (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.pairF (ps.flatMap f) E ks ν = (ps.map fun p => L.pairF (f p) E ks ν).prod := by
  induction ps with
  | nil => simp [pairF]
  | cons p ps ih =>
    rw [List.flatMap_cons, List.map_cons, List.prod_cons, ← ih]
    simp [pairF, List.map_append, List.prod_append]

/-- The pair products agree under `firstφ`. -/
theorem PairFirst.pairF {L R : LDiagram} {ps ps' : List LPair} {bid mb c m1 : ℕ} {B : List ℕ}
    {cp : ℕ → ℕ} (h : PairFirst L R ps ps' bid mb c m1 B cp) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c + 1) (hm : ks m1 = 1) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (firstφ B cp) x)) :
    R.pairF ps' E ks τ = L.pairF ps E ks τ' := by
  rw [h.eq, pairF_flatMap]
  unfold LDiagram.pairF
  refine congrArg List.prod (List.map_congr_left fun p hp => ?_)
  obtain ⟨hb, hL, hR, hj, hm0, hl, hr, hlB, hrB, hlc, hrc⟩ := h.form p hp
  have e1 : (p.resid m1).box = bid := hb
  have e2 : (p.fixLeft (cpP cp p.left) m1).box = bid := hb
  simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, e1, e2, hb,
    h.multR, h.multL, hk]
  refine LPair.val_first p hL hR hj hm0 E ks m1 hm (ks c) _ τ τ' ?_ (fun j hjk => ?_)
    (fun j hjk => ?_)
  · rw [hτ _ (dlab_mem L ks _ mb 0 hl hlc (by omega)), relab_dlab, firstφ_B0 _ hlB]
  · rw [hτ _ (dlab_mem L ks _ mb (j + 1) hl hlc (by omega)), relab_dlab, firstφ_Bs _ hlB]
  · rw [hτ _ (dlab_mem L ks _ mb (j + 1) hr hrc (by omega)), relab_dlab, firstφ_Bs _ hrB]

/-- The pair products agree under `peelφ`. -/
theorem PairLast.pairF {L L' : LDiagram} {ps ps' : List LPair} {bid mb c : ℕ} {B : List ℕ}
    {cp : ℕ → ℕ} (h : PairLast L L' ps ps' bid mb c B cp) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c + 1) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (peelφ B cp (ks c)) x)) :
    L'.pairF ps' E ks τ = L.pairF ps E ks τ' := by
  rw [h.eq, pairF_flatMap]
  unfold LDiagram.pairF
  refine congrArg List.prod (List.map_congr_left fun p hp => ?_)
  obtain ⟨hb, hL, hR, hm0, hl, hr, hlB, hrB, hlc, hrc⟩ := h.form p hp
  have e2 : (p.fixRight (cpP cp p.right) c).box = bid := hb
  simp only [List.map_cons, List.map_nil, List.prod_cons, List.prod_nil, mul_one, e2, hb,
    h.multR, h.multL, hk]
  refine LPair.val_last p hL hR hm0 E ks c (ks c) rfl _ τ τ' ?_ (fun j hjk => ?_)
    (fun j hjk => ?_)
  · rw [hτ _ (dlab_mem L ks _ mb (ks c) hr hrc (by omega)), relab_dlab, peelφ_B _ _ hrB]
  · rw [hτ _ (dlab_mem L ks _ mb j hl hlc (by omega)), relab_dlab, peelφ_ne _ _ _ _ (by omega)]
  · rw [hτ _ (dlab_mem L ks _ mb j hr hrc (by omega)), relab_dlab, peelφ_ne _ _ _ _ (by omega)]

/-- `evalP` reads only the slots' digits on boundary legs. -/
theorem evalP_congr (L : LDiagram) (ps : List LPair) (E : Env) (ks : ℕ → ℕ)
    (σ σ' : ℕ → List ℕ → ℕ)
    (h : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t = σ' i t) :
    L.evalP ps E ks σ = L.evalP ps E ks σ' := by
  unfold evalP evalW
  refine sumOver_extP _ _ (fun x => x % 2 = 0 ∨ ∃ i, ∃ hi : i < L.base.bnd.length,
    ∃ t ∈ cidx ks (L.base.legChain L.base.bnd[i]), x = slab i t) _ (fun τ τ' hτ => ?_) _ _
    (fun x hx _ => ?_)
  · have hd : ∀ p t, τ (dlab p t) = τ' (dlab p t) := fun p t => hτ _ (Or.inl (dlab_even p t))
    have hp : L.pinsX ks τ ↔ L.pinsX ks τ' := by
      unfold pinsX
      refine and_congr (forall₂_congr fun y _ => by rw [hd, hd])
        (forall₂_congr fun q hq => forall₂_congr fun t ht => ?_)
      obtain ⟨p, i⟩ := q
      obtain ⟨hi, e⟩ := (mem_zipIdx' _ _ _).1 hq
      subst e
      rw [hd, hτ _ (Or.inr ⟨i, hi, t, ht, rfl⟩)]
    have hP : L.prodL E ks τ = L.prodL E ks τ' := by
      unfold prodL
      refine congrArg List.prod (List.map_congr_left fun n _ =>
        congrArg List.prod (List.map_congr_left fun t _ => ?_))
      apply val_congr
      · exact List.map_congr_left fun x hx => hτ _ (Or.inl ((dnodeL_even L ks n t).1 x hx))
      · exact List.map_congr_left fun x hx => hτ _ (Or.inl ((dnodeL_even L ks n t).2 x hx))
    have hnode : ∀ (p : LPair) j m tl tr, p.pv E ks j m tl tr τ = p.pv E ks j m tl tr τ' :=
      fun p j m tl tr => by simp only [LPair.pv, hd]
    have hQ : L.pairF ps E ks τ = L.pairF ps E ks τ' := by
      unfold pairF LPair.val
      simp only [hnode]
    beta_reduce
    rw [hP, hQ]
    classical
    exact if_congr hp rfl rfl
  · rcases hx with hx | ⟨i, hi, t, ht, rfl⟩
    · simp [slotsD, hx]
    · rw [slotsD_slab, slotsD_slab]
      exact h i hi t ht

section first

variable {L Lr R Rr : LDiagram} {bid mb c m1 : ℕ} {B : List ℕ} {cp : ℕ → ℕ}

theorem evalW_first (h : FirstOK L Lr R Rr bid mb c B cp) (hx : FirstX L R bid m1 B cp)
    (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (hm : ks m1 = 1) (F F' : (ℕ → ℕ) → ℂ)
    (hF : ∀ τ τ' : ℕ → ℕ, (∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (firstφ B cp) x)) →
      F' τ = F τ')
    (σ' : ℕ → List ℕ → ℕ) :
    R.evalW E ks F' σ' = L.evalW E ks F (bndMap L.base R.base (firstφ B cp) σ') := by
  refine evalW_relabelφ L R E ks ks σ' F F' (firstφ B cp) (Diagram.dlabs_nodup _ _ h.rL.legs)
    (Diagram.dlabs_nodup _ _ h.rR.legs) h.rR.bnd ?_ ?_ ?_ ?_ (fun x hx' => ?_) (fun x hx' => ?_)
    (fun y hy => ?_) (fun p hp => h.rL.bl p hp) ?_ ?_ (fun τ τ' hτ => hF τ τ' hτ)
  · intro p hp t ht
    rcases h.spec ks hk p hp t ht with ⟨u, hB, rfl, hu⟩ | ⟨j, u, hB, rfl, hj, hu⟩ | hB
    · rw [firstφ_B0 p hB u]
      exact ⟨h.memC p hp hB, by rw [h.lcR]; exact hu⟩
    · rw [firstφ_Bs p hB j u]
      obtain ⟨-, c1'⟩ := h.chB' p hp hB
      exact ⟨h.mem' p hp, by rw [h.lcR, c1']; exact cons_mem_cidx ks c j _ u hj hu⟩
    · rw [firstφ_O p hB t]
      exact ⟨h.mem' p hp, by rw [h.lcR, h.chO' p hp hB, ← h.lc]; exact ht⟩
  · intro q hq u hu
    rw [h.legs_eq, List.mem_append] at hq
    rcases hq with hq | hq
    · by_cases hB : q.node ∈ B
      · obtain ⟨c1, c1'⟩ := h.chB' q hq hB
        rw [h.lcR, c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c _ _).1 hu
        exact ⟨q, hq, (j + 1) :: v, by rw [h.lc, c1]; exact cons_mem_cidx ks mb _ _ v (by omega) hv,
          firstφ_Bs q hB j v⟩
      · exact ⟨q, hq, u, by rw [h.lc, ← h.chO' q hq hB, ← h.lcR]; exact hu, firstφ_O q hB u⟩
    · obtain ⟨n, hn, hq⟩ := List.mem_flatMap.1 hq
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hq
      obtain ⟨hn, hnB⟩ := List.mem_filter.1 hn
      have hpn := node_of_leg n p hp
      have hpB : p.node ∈ B := by rw [hpn]; simpa using hnB
      have hpL : p ∈ L.base.legs := List.mem_flatMap.2 ⟨n, hn, hp⟩
      obtain ⟨c1, -⟩ := h.chB' p hpL hpB
      refine ⟨p, hpL, 0 :: u, ?_, firstφ_B0 p hpB u⟩
      rw [h.lc, c1]
      exact cons_mem_cidx ks mb _ _ u (by omega) (by rw [h.lcR] at hu; exact hu)
  · intro p hp t ht p' hp' t' ht' he
    rcases h.spec ks hk p hp t ht with ⟨u, hB, rfl, -⟩ | ⟨j, u, hB, rfl, -, -⟩ | hB <;>
    rcases h.spec ks hk p' hp' t' ht' with ⟨u', hB', rfl, -⟩ | ⟨j', u', hB', rfl, -, -⟩ | hB'
    · rw [firstφ_B0 p hB, firstφ_B0 p' hB'] at he
      obtain ⟨e1, rfl⟩ := Prod.mk.inj he
      exact ⟨h.cpP_inj p hp hB p' hp' hB' e1, rfl⟩
    · rw [firstφ_B0 p hB, firstφ_Bs p' hB'] at he
      exact (h.cp_ne p hp hB p' hp' (Prod.mk.inj he).1).elim
    · rw [firstφ_B0 p hB, firstφ_O p' hB'] at he
      exact (h.cp_ne p hp hB p' hp' (Prod.mk.inj he).1).elim
    · rw [firstφ_Bs p hB, firstφ_B0 p' hB'] at he
      exact (h.cp_ne p' hp' hB' p hp (Prod.mk.inj he).1.symm).elim
    · rw [firstφ_Bs p hB, firstφ_Bs p' hB'] at he
      obtain ⟨rfl, e2⟩ := Prod.mk.inj he
      obtain ⟨rfl, rfl⟩ := List.cons.inj e2
      exact ⟨rfl, rfl⟩
    · rw [firstφ_Bs p hB, firstφ_O p' hB'] at he
      exact (hB' ((Prod.mk.inj he).1 ▸ hB)).elim
    · rw [firstφ_O p hB, firstφ_B0 p' hB'] at he
      exact (h.cp_ne p' hp' hB' p hp (Prod.mk.inj he).1.symm).elim
    · rw [firstφ_O p hB, firstφ_Bs p' hB'] at he
      exact (hB ((Prod.mk.inj he).1 ▸ hB')).elim
    · rw [firstφ_O p hB, firstφ_O p' hB'] at he
      exact Prod.mk.inj he
  · intro p hp t ht
    rcases h.spec ks hk p hp t ht with ⟨u, hB, rfl, hu⟩ | ⟨j, u, hB, rfl, -, -⟩ | hB
    · rw [firstφ_B0 p hB u]
      exact congrArg E.dimv (hx.dimAt_first0 h p hp hB ks u ((mem_cidx ks _ u).1 hu).length_eq)
    · rw [firstφ_Bs p hB j u]
      exact congrArg E.dimv (hx.dimAt_shift h p hp hB ks hm j u)
    · rw [firstφ_O p hB t]
      exact congrArg E.dimv (hx.dimAt_keep h p hp hB ks t)
  · rw [dpairsX_nil L hx.al] at hx'
    exact h.rL.pl ks x hx'
  · rw [dpairsX_nil L hx.al] at hx'
    rw [dpairsX_nil R hx.al']
    obtain ⟨a1, a2, a3, a4⟩ := h.rL.pl ks x hx'
    have s3 := sym_step (revφ B (ks c)) (fun y hy => h.rR'.fwd ks y hy) _
      (sym_step (peelφ B cp (ks c)) (fun y hy => h.ok.fwd ks hk y hy) _ (h.rL.fwd ks x hx'))
    have e : pmap (revφ B (ks c)) (pmap (peelφ B cp (ks c)) (pmap (revφ B (ks mb)) x)) =
        pmap (firstφ B cp) x :=
      Prod.ext (h.comp hx ks hk _ a1 _ a2) (h.comp hx ks hk _ a3 _ a4)
    rw [← e]
    exact s3
  · rw [dpairsX_nil R hx.al'] at hy
    rw [dpairsX_nil L hx.al]
    obtain ⟨w, hw, e⟩ := sym_back (cmpφ (peelφ B cp (ks c)) (revφ B (ks c))) (revφ B (ks mb))
      (fun z hz => h.rL'.bwd ks z hz) y
      (sym_back (revφ B (ks c)) (peelφ B cp (ks c)) (fun x hx => h.ok.bwd ks hk x hx) y
        (h.rR.bwd ks y hy))
    obtain ⟨a1, a2, a3, a4⟩ := h.rL.pl ks w hw
    have e' : pmap (cmpφ (peelφ B cp (ks c)) (revφ B (ks c))) (pmap (revφ B (ks mb)) w) =
        pmap (firstφ B cp) w :=
      Prod.ext (h.comp hx ks hk _ a1 _ a2) (h.comp hx ks hk _ a3 _ a4)
    exact ⟨w, hw, by rw [← e']; exact e⟩
  · intro p hp t ht
    obtain ⟨-, h2, h3⟩ := h.bndL p hp
    rcases h.spec ks hk p (h.rL.bl p hp) t ht with ⟨u, hB, rfl, -⟩ | ⟨j, u, hB, rfl, -, -⟩ | hB
    · rw [firstφ_B0 p hB u]; exact h3 hB
    · rw [firstφ_Bs p hB j u]; exact h2
    · rw [firstφ_O p hB t]; exact h2
  · intro q hq u hu
    rcases h.bnd' q hq with hq' | ⟨p, hp, hB, rfl⟩
    · have hql := (h.bndL q hq').1
      by_cases hB : q.node ∈ B
      · obtain ⟨c1, c1'⟩ := h.chB' q hql hB
        rw [h.lcR, c1'] at hu
        obtain ⟨j, v, rfl, hj, hv⟩ := (mem_cidx_cons ks c _ _).1 hu
        exact ⟨q, hq', (j + 1) :: v,
          by rw [h.lc, c1]; exact cons_mem_cidx ks mb _ _ v (by omega) hv, firstφ_Bs q hB j v⟩
      · exact ⟨q, hq', u, by rw [h.lc, ← h.chO' q hql hB, ← h.lcR]; exact hu, firstφ_O q hB u⟩
    · have hpl := (h.bndL p hp).1
      obtain ⟨c1, -⟩ := h.chB' p hpl hB
      refine ⟨p, hp, 0 :: u, ?_, firstφ_B0 p hB u⟩
      rw [h.lc, c1]
      exact cons_mem_cidx ks mb _ _ u (by omega) (by rw [h.lcR] at hu; exact hu)

/-- A first-copy peel of an outermost chain box with pair boxes keeps the digit sum `evalP`, `L`'s
slots read from `R`'s through `firstφ`. -/
theorem evalP_first {ps ps' : List LPair} (h : FirstOK L Lr R Rr bid mb c B cp)
    (hx : FirstX L R bid m1 B cp) (hp : PairFirst L R ps ps' bid mb c m1 B cp) (E : Env)
    (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (hm : ks m1 = 1) (σ' : ℕ → List ℕ → ℕ) :
    R.evalP ps' E ks σ' = L.evalP ps E ks (bndMap L.base R.base (firstφ B cp) σ') :=
  evalW_first h hx E ks hk hm _ _ (fun τ τ' hτ => by
    rw [h.prod hx E ks hk hm τ τ' hτ, hp.pairF E ks hk hm τ τ' hτ]) σ'

/-- `evalP_first` read through the flat positions of both sides. -/
theorem evalP_peel_first_flat {ps ps' : List LPair} {sd sd' : SlotData}
    (h : FirstOK L Lr R Rr bid mb c B cp) (hx : FirstX L R bid m1 B cp)
    (hp : PairFirst L R ps ps' bid mb c m1 B cp) (hf : PeelFirst L R B cp mb c sd sd') (E : Env)
    (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (hm : ks m1 = 1) (y : ℕ → ℕ) :
    R.evalP ps' E ks (flatAt sd' ks y) = L.evalP ps E ks (flatAt sd ks y) := by
  rw [evalP_first h hx hp E ks hk hm]
  refine evalP_congr L ps E ks _ _ fun i hi t ht => ?_
  have hv : (i, t) ∈ slots sd ks := by
    refine (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, ?_⟩
    rw [hf.ch i hi, bndAt, List.getD_eq_getElem _ _ hi]
    exact ht
  rw [bndMap_first hf _ i t hv]
  exact congrArg y (hf.fpos_psi hk hv)

end first

section last

variable {L L' : LDiagram} {bid mb c : ℕ} {B : List ℕ} {cp : ℕ → ℕ}

/-- A last-copy peel of an outermost chain box with pair boxes keeps the digit sum `evalP`,
`L`'s slots read from `L'`'s through `peelφ`. -/
theorem evalP_peel {ps ps' : List LPair} (h : PeelX L L' bid mb c B cp)
    (hp : PairLast L L' ps ps' bid mb c B cp) (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (σ' : ℕ → List ℕ → ℕ) :
    L'.evalP ps' E ks σ' = L.evalP ps E ks (bndMap L.base L'.base (peelφ B cp (ks c)) σ') :=
  evalW_peel h E ks hk _ _ (fun τ τ' hτ => by
    rw [h.prod E ks hk τ τ' hτ, hp.pairF E ks hk τ τ' hτ]) σ'

/-- `evalP_peel` read through the flat positions of both sides. -/
theorem evalP_peel_flat {ps ps' : List LPair} {sd sd' : SlotData} (h : PeelX L L' bid mb c B cp)
    (hp : PairLast L L' ps ps' bid mb c B cp) (hf : PeelFlat L L' B cp mb c sd sd') (E : Env)
    (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (y : ℕ → ℕ) :
    L'.evalP ps' E ks (flatAt sd' ks y) = L.evalP ps E ks (flatAt sd ks y) := by
  rw [evalP_peel h hp E ks hk]
  refine evalP_congr L ps E ks _ _ fun i hi t ht => ?_
  rw [bndMap_psi]
  have hv : (i, t) ∈ slots sd ks := by
    refine (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, ?_⟩
    rw [hf.ch i hi, bndAt, List.getD_eq_getElem _ _ hi]
    exact ht
  exact congrArg y (hf.fpos_psi hk hv)

end last

end LDiagram

end ArchytasZX
