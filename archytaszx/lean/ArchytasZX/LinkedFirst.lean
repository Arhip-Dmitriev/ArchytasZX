/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedSplice

/-!
# Peels of the first copy over `evalX`

`FirstOK`: `L` and `R` without their readings (`strip`), each with the links of the outermost
chain box `bid` flipped (`Lr`, `Rr`, `RevOK` both ways), and `Rr` the peel of the last copy of
`Lr` (`PeelOK`). `FirstX`: the readings of `L` and `R`; a node or leg off `B` reads alike, one on
`B` reads `bid` one copy higher in `R` (`rdS`: coefficient `1` of a multiplicity index `m1` with
`ks m1 = 1`), its copy reads `bid` at `0` (`rdC0`), and inner boxes one level up on the copy.

`evalX_first`: `R.evalX` is `L.evalX` read through `firstφ` (`evalX_relabelφ`): a leg on `B` at
copy `0` goes to its copy, at copy `j + 1` to copy `j`. The digit pairs go through `RevOK.fwd`,
`PeelOK.fwd` and `RevOK.fwd` (`RevOK.bwd`, `PeelOK.bwd`, `RevOK.bwd` back), whose composite is
`firstφ` on every digit (`FirstOK.comp`); the node products split each node of `B` into its
first copy and the rest (`cidx_first`).
`evalX_peel_first`: the same read through the flat positions of both sides (`PeelFirst`).
-/

namespace ArchytasZX

namespace LDiagram

/-- `t` with its first coordinate one higher. -/
def incT : List ℕ → List ℕ
  | [] => []
  | j :: u => (j + 1) :: u

theorem cidx_first (ks : ℕ → ℕ) (mb c : ℕ) (r : List ℕ) (hk : ks mb = ks c + 1) :
    cidx ks (mb :: r) = (cidx ks r).map (0 :: ·) ++ (cidx ks (c :: r)).map incT := by
  simp only [cidx, hk, List.range_succ_eq_map, List.flatMap_cons, List.flatMap_map,
    List.map_flatMap, List.map_map]
  rfl

/-- The peel of the first copy's map of digits: a leg on `B` at the copy `0` goes to its copy, at
the copy `j + 1` to the copy `j`. -/
def firstφ (B : List ℕ) (cp : ℕ → ℕ) (p : Port) (t : List ℕ) : Dig :=
  if p.node ∈ B ∧ t.head? = some 0 then (cpP cp p, t.tail)
  else if p.node ∈ B then (p, decT t) else (p, t)

section maps

variable {B : List ℕ} {cp : ℕ → ℕ}

theorem firstφ_B0 (p : Port) (hp : p.node ∈ B) (u : List ℕ) :
    firstφ B cp p (0 :: u) = (cpP cp p, u) := by
  simp [firstφ, hp]

theorem firstφ_Bs (p : Port) (hp : p.node ∈ B) (j : ℕ) (u : List ℕ) :
    firstφ B cp p ((j + 1) :: u) = (p, j :: u) := by
  simp [firstφ, hp, decT]

theorem firstφ_O (p : Port) (hp : p.node ∉ B) (t : List ℕ) :
    firstφ B cp p t = (p, t) := by
  simp [firstφ, hp]

end maps

/-- `ψ` after `φ`. -/
def cmpφ (φ ψ : Port → List ℕ → Dig) (p : Port) (t : List ℕ) : Dig := ψ (φ p t).1 (φ p t).2

theorem pmap_pswap (φ : Port → List ℕ → Dig) (x : Dig × Dig) :
    pmap φ (pswap x) = pswap (pmap φ x) := rfl

theorem pswap_pswap (x : Dig × Dig) : pswap (pswap x) = x := rfl

/-- A map of digits sending each pair of `S` into `S'` (either way round) sends a pair of `S` or
its swap the same way. -/
theorem sym_step {S S' : List (Dig × Dig)} (φ : Port → List ℕ → Dig)
    (hf : ∀ y ∈ S, pmap φ y ∈ S' ∨ pswap (pmap φ y) ∈ S') (z : Dig × Dig)
    (hz : z ∈ S ∨ pswap z ∈ S) : pmap φ z ∈ S' ∨ pswap (pmap φ z) ∈ S' := by
  rcases hz with hz | hz
  · exact hf z hz
  · rcases hf _ hz with e | e
    · right; exact e
    · left; exact e

/-- A pair reached from `S` by `φ`, each pair of `S` reached from `S'` by `ψ`, is reached from
`S'` by `ψ` then `φ`. -/
theorem sym_back {S S' : List (Dig × Dig)} (φ ψ : Port → List ℕ → Dig)
    (hb : ∀ x ∈ S, ∃ w ∈ S', pmap ψ w = x ∨ pswap (pmap ψ w) = x) (y : Dig × Dig)
    (hy : ∃ x ∈ S, pmap φ x = y ∨ pswap (pmap φ x) = y) :
    ∃ w ∈ S', pmap φ (pmap ψ w) = y ∨ pswap (pmap φ (pmap ψ w)) = y := by
  obtain ⟨x, hx, e⟩ := hy
  obtain ⟨w, hw, e'⟩ := hb x hx
  refine ⟨w, hw, ?_⟩
  rcases e' with rfl | rfl
  · exact e
  · rcases e with e | e
    · right; exact e
    · left; exact e

/-- The reading `o'` in `R` of the node `x` of `B` of its reading `o` in `L`: a reading of `bid`
of coefficient `0` at depth `0` becomes coefficient `1` of `m1` at depth `0`, any other the same at
the same depth, at least `1`. -/
def rdS (L R : LDiagram) (bid m1 x : ℕ) : Option (ℕ × ℕ × ℕ) → Option (ℕ × ℕ × ℕ) → Bool
  | none, none => true
  | some q, some q' =>
    if q.1 = bid then
      decide (L.depth q.1 x = 0) && decide (q.2.1 = 0) && decide (q'.2 = (1, m1)) &&
        decide (R.depth q'.1 x = 0)
    else decide (q'.2 = q.2) && decide (R.depth q'.1 x = L.depth q.1 x) &&
      decide (1 ≤ L.depth q.1 x)
  | _, _ => false

/-- The reading `o'` of the copy `x'` of a reading `o` of `x`: a reading of `bid` of coefficient
`0` at depth `0` becomes one of coefficient `0` at a box off `x'`, a reading of an inner box the
same one level up. -/
def rdC0 (L R : LDiagram) (bid x x' : ℕ) : Option (ℕ × ℕ × ℕ) → Option (ℕ × ℕ × ℕ) → Bool
  | none, none => true
  | some q, some q' =>
    if q.1 = bid then
      decide (L.depth q.1 x = 0) && decide (q.2.1 = 0) && decide (q'.2.1 = 0) &&
        decide (R.depth q'.1 x' = (R.base.nodeChain x').length)
    else decide (q'.2 = q.2) && decide (R.depth q'.1 x' + 1 = L.depth q.1 x)
  | _, _ => false

theorem rdVal_shift (L R : LDiagram) (bid m1 x : ℕ) (ks : ℕ → ℕ) (hm : ks m1 = 1) (k j : ℕ)
    (u : List ℕ) (o o' : Option (ℕ × ℕ × ℕ)) (h : rdS L R bid m1 x o o' = true) :
    R.rdVal ks x k (j :: u) o' = L.rdVal ks x k ((j + 1) :: u) o := by
  cases o with
  | none =>
    cases o' with
    | none => rfl
    | some q' => simp [rdS] at h
  | some q =>
    cases o' with
    | none => simp [rdS] at h
    | some q' =>
      obtain ⟨y, a, m⟩ := q
      obtain ⟨y', a', m'⟩ := q'
      simp only [rdVal]
      by_cases hb : y = bid
      · subst hb
        simp only [rdS, ite_true, Bool.and_eq_true, decide_eq_true_eq, Prod.mk.injEq] at h
        obtain ⟨⟨⟨h1, h2⟩, h3, h4⟩, h5⟩ := h
        rw [h5, h1, h2, h3, h4, hm]
        simp
      · simp only [rdS, if_neg hb, Bool.and_eq_true, decide_eq_true_eq, Prod.mk.injEq] at h
        obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
        obtain ⟨d, hd⟩ : ∃ d, L.depth y x = d + 1 := ⟨_, (Nat.succ_pred_eq_of_pos h4).symm⟩
        rw [h3, h1, h2, hd, List.getD_cons_succ, List.getD_cons_succ]

theorem rdVal_first0 (L R : LDiagram) (bid x x' : ℕ) (ks : ℕ → ℕ) (k : ℕ) (u : List ℕ)
    (hu : u.length = (R.base.nodeChain x').length) (o o' : Option (ℕ × ℕ × ℕ))
    (h : rdC0 L R bid x x' o o' = true) :
    R.rdVal ks x' k u o' = L.rdVal ks x k (0 :: u) o := by
  cases o with
  | none =>
    cases o' with
    | none => rfl
    | some q' => simp [rdC0] at h
  | some q =>
    cases o' with
    | none => simp [rdC0] at h
    | some q' =>
      obtain ⟨y, a, m⟩ := q
      obtain ⟨y', a', m'⟩ := q'
      simp only [rdVal]
      by_cases hb : y = bid
      · subst hb
        simp only [rdC0, ite_true, Bool.and_eq_true, decide_eq_true_eq] at h
        obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
        rw [h4, ← hu, h1, h2, h3, List.getD_eq_default _ _ (le_refl _)]
        simp
      · simp only [rdC0, if_neg hb, Bool.and_eq_true, decide_eq_true_eq, Prod.mk.injEq] at h
        obtain ⟨⟨h1, h2⟩, h3⟩ := h
        rw [← h3, h1, h2, List.getD_cons_succ]

/-- `L` and `R` without their readings, `Lr` and `Rr` the two with the links of the box `bid`
flipped, `Rr` the peel of the last copy of `Lr`. -/
structure FirstOK (L Lr R Rr : LDiagram) (bid mb c : ℕ) (B : List ℕ) (cp : ℕ → ℕ) : Prop where
  rL : RevOK L.strip Lr bid mb B
  rL' : RevOK Lr L.strip bid mb B
  rR : RevOK R.strip Rr bid c B
  rR' : RevOK Rr R.strip bid c B
  ok : PeelOK Lr Rr bid mb c B cp

/-- The readings of `L` and `R` after a peel of the first copy of the box `bid` over the nodes
`B` into the copies `cp`: no aligned wires, the copies off `B`, every node and leg off `B` read
alike, those on `B` by `rdS` and their copies by `rdC0`. -/
structure FirstX (L R : LDiagram) (bid m1 : ℕ) (B : List ℕ) (cp : ℕ → ℕ) : Prop where
  al : L.aligned = []
  al' : R.aligned = []
  cpB : ∀ n ∈ L.base.nodes, n.id ∈ B → cp n.id ∉ B
  phK : ∀ n ∈ L.base.nodes, n.id ∉ B → (R.cph.find? fun q => decide (q.1 = n.id)) =
      (L.cph.find? fun q => decide (q.1 = n.id)) ∧
    ((L.cph.find? fun q => decide (q.1 = n.id)).all fun q =>
      decide (R.depth q.2.1 n.id = L.depth q.2.1 n.id)) = true
  phS : ∀ n ∈ L.base.nodes, n.id ∈ B → rdS L R bid m1 n.id
    ((L.cph.find? fun q => decide (q.1 = n.id)).map (·.2))
    ((R.cph.find? fun q => decide (q.1 = n.id)).map (·.2)) = true
  phC : ∀ n ∈ L.base.nodes, n.id ∈ B → rdC0 L R bid n.id (cp n.id)
    ((L.cph.find? fun q => decide (q.1 = n.id)).map (·.2))
    ((R.cph.find? fun q => decide (q.1 = cp n.id)).map (·.2)) = true
  dmK : ∀ p ∈ L.base.legs, p.node ∉ B → (R.cdim.find? fun q => decide (q.1 = p)) =
      (L.cdim.find? fun q => decide (q.1 = p)) ∧
    ((L.cdim.find? fun q => decide (q.1 = p)).all fun q =>
      decide (R.depth q.2.1 p.node = L.depth q.2.1 p.node)) = true
  dmS : ∀ p ∈ L.base.legs, p.node ∈ B → rdS L R bid m1 p.node
    ((L.cdim.find? fun q => decide (q.1 = p)).map (·.2))
    ((R.cdim.find? fun q => decide (q.1 = p)).map (·.2)) = true
  dmC : ∀ p ∈ L.base.legs, p.node ∈ B → rdC0 L R bid p.node (cp p.node)
    ((L.cdim.find? fun q => decide (q.1 = p)).map (·.2))
    ((R.cdim.find? fun q => decide (q.1 = cpP cp p)).map (·.2)) = true

section first

variable {L Lr R Rr : LDiagram} {bid mb c m1 : ℕ} {B : List ℕ} {cp : ℕ → ℕ}

theorem FirstOK.lb (h : FirstOK L Lr R Rr bid mb c B cp) : Lr.base = L.base := h.rL.base

theorem FirstOK.rb (h : FirstOK L Lr R Rr bid mb c B cp) : Rr.base = R.base := h.rR.base

theorem FirstOK.lc (h : FirstOK L Lr R Rr bid mb c B cp) (q : Port) :
    L.base.legChain q = L.base.nodeChain q.node := h.rL.lc q

theorem FirstOK.lcR (h : FirstOK L Lr R Rr bid mb c B cp) (q : Port) :
    R.base.legChain q = R.base.nodeChain q.node := h.rR.lc q

theorem FirstOK.chB (h : FirstOK L Lr R Rr bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) : L.base.nodeChain n.id = mb :: R.base.nodeChain (cp n.id) ∧
      R.base.nodeChain n.id = c :: R.base.nodeChain (cp n.id) := by
  have := h.ok.chB n (by rw [h.lb]; exact hn) hB
  rwa [h.lb, h.rb] at this

theorem FirstOK.chB' (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) : L.base.nodeChain p.node = mb :: R.base.nodeChain (cp p.node) ∧
      R.base.nodeChain p.node = c :: R.base.nodeChain (cp p.node) := by
  have := h.ok.chB' p (by rw [h.lb]; exact hp) hB
  rwa [h.lb, h.rb] at this

theorem FirstOK.chO (h : FirstOK L Lr R Rr bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∉ B) : R.base.nodeChain n.id = L.base.nodeChain n.id := by
  have := h.ok.chO n (by rw [h.lb]; exact hn) hB
  rwa [h.lb, h.rb] at this

theorem FirstOK.chO' (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∉ B) : R.base.nodeChain p.node = L.base.nodeChain p.node := by
  have := h.ok.chO' p (by rw [h.lb]; exact hp) hB
  rwa [h.lb, h.rb] at this

theorem FirstOK.nodes (h : FirstOK L Lr R Rr bid mb c B cp) :
    R.base.nodes = L.base.nodes ++ (L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp) := by
  have := h.ok.nodes
  rwa [h.lb, h.rb] at this

theorem FirstOK.legs_eq (h : FirstOK L Lr R Rr bid mb c B cp) :
    R.base.legs = L.base.legs ++
      ((L.base.nodes.filter fun n => n.id ∈ B).flatMap fun n => n.legs.map (cpP cp)) := by
  have := h.ok.legs_eq
  rwa [h.lb, h.rb] at this

theorem FirstOK.mem' (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs) :
    p ∈ R.base.legs := by
  have := h.ok.mem' p (by rw [h.lb]; exact hp)
  rwa [h.rb] at this

theorem FirstOK.memC (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) : cpP cp p ∈ R.base.legs := by
  have := h.ok.memC p (by rw [h.lb]; exact hp) hB
  rwa [h.rb] at this

theorem FirstOK.cp_fresh (h : FirstOK L Lr R Rr bid mb c B cp) (n : DNode)
    (hn : n ∈ L.base.nodes) (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) :
    cp n.id ≠ m.id :=
  h.ok.cp_fresh n (by rw [h.lb]; exact hn) hB m (by rw [h.lb]; exact hm)

theorem FirstOK.cp_inj (h : FirstOK L Lr R Rr bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) (m : DNode) (hm : m ∈ L.base.nodes) (hmB : m.id ∈ B)
    (e : cp n.id = cp m.id) : n = m :=
  h.ok.cp_inj n (by rw [h.lb]; exact hn) hB m (by rw [h.lb]; exact hm) hmB e

theorem FirstOK.portDim_keep (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port)
    (hp : p ∈ L.base.legs) : R.base.portDim p = L.base.portDim p := by
  have := h.ok.portDim_keep p (by rw [h.lb]; exact hp)
  rwa [h.lb, h.rb] at this

theorem FirstOK.portDim_cp (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port)
    (hp : p ∈ L.base.legs) (hB : p.node ∈ B) : R.base.portDim (cpP cp p) = L.base.portDim p := by
  have := h.ok.portDim_cp p (by rw [h.lb]; exact hp) hB
  rwa [h.lb, h.rb] at this

theorem FirstOK.bndL (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.bnd) :
    p ∈ L.base.legs ∧ p ∈ R.base.bnd ∧ (p.node ∈ B → cpP cp p ∈ R.base.bnd) := by
  have := h.ok.bndL p (by rw [h.lb]; exact hp)
  rwa [h.lb, h.rb] at this

theorem FirstOK.bnd' (h : FirstOK L Lr R Rr bid mb c B cp) (q : Port) (hq : q ∈ R.base.bnd) :
    q ∈ L.base.bnd ∨ ∃ p ∈ L.base.bnd, p.node ∈ B ∧ q = cpP cp p := by
  have := h.ok.bnd' q (by rw [h.rb]; exact hq)
  rwa [h.lb] at this

theorem FirstOK.cpP_inj (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) (p' : Port) (hp' : p' ∈ L.base.legs) (hB' : p'.node ∈ B)
    (e : cpP cp p = cpP cp p') : p = p' := by
  obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
  obtain ⟨n', hn', en'⟩ := node_of_legs _ p' hp'
  have e1 : cp p.node = cp p'.node := congrArg Port.node e
  have hnn := h.cp_inj n hn (en ▸ hB) n' hn' (en' ▸ hB') (by rw [← en, ← en']; exact e1)
  have e0 : p.node = p'.node := by rw [en, en', hnn]
  have e2 : p.out = p'.out := by have := congrArg Port.out e; exact this
  have e3 : p.pos = p'.pos := by have := congrArg Port.pos e; exact this
  cases p
  cases p'
  simp only [Port.mk.injEq]
  exact ⟨e0, e2, e3⟩

theorem FirstOK.cp_ne (h : FirstOK L Lr R Rr bid mb c B cp) (p : Port) (hp : p ∈ L.base.legs)
    (hB : p.node ∈ B) (p' : Port) (hp' : p' ∈ L.base.legs) (e : cpP cp p = p') : False := by
  obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
  obtain ⟨m, hm, em⟩ := node_of_legs _ p' hp'
  exact h.cp_fresh n hn (en ▸ hB) m hm (by rw [← em, ← e, ← en]; rfl)

/-- A digit of `L`: on `B` at copy `0`, on `B` at copy `j + 1`, or off `B`. -/
theorem FirstOK.spec (h : FirstOK L Lr R Rr bid mb c B cp) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (p : Port) (hp : p ∈ L.base.legs) (t : List ℕ) (ht : t ∈ cidx ks (L.base.legChain p)) :
    (∃ u, p.node ∈ B ∧ t = 0 :: u ∧ u ∈ cidx ks (R.base.nodeChain (cp p.node))) ∨
      (∃ j u, p.node ∈ B ∧ t = (j + 1) :: u ∧ j < ks c ∧
        u ∈ cidx ks (R.base.nodeChain (cp p.node))) ∨ p.node ∉ B := by
  by_cases hB : p.node ∈ B
  · obtain ⟨c1, -⟩ := h.chB' p hp hB
    rw [h.lc, c1] at ht
    obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks mb _ _).1 ht
    rcases j with _ | j
    · exact Or.inl ⟨u, hB, rfl, hu⟩
    · exact Or.inr (Or.inl ⟨j, u, hB, rfl, by omega, hu⟩)
  · exact Or.inr (Or.inr hB)

/-- On every digit of `L` the reversal, the peel of the last copy and the reversal are
`firstφ`. -/
theorem FirstOK.comp (h : FirstOK L Lr R Rr bid mb c B cp) (hx : FirstX L R bid m1 B cp)
    (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (p : Port) (hp : p ∈ L.base.legs) (t : List ℕ)
    (ht : t ∈ cidx ks (L.base.legChain p)) :
    revφ B (ks c) (peelφ B cp (ks c) (revφ B (ks mb) p t).1 (revφ B (ks mb) p t).2).1
      (peelφ B cp (ks c) (revφ B (ks mb) p t).1 (revφ B (ks mb) p t).2).2 = firstφ B cp p t := by
  rcases h.spec ks hk p hp t ht with ⟨u, hB, rfl, -⟩ | ⟨j, u, hB, rfl, hj, -⟩ | hB
  · have e0 : ks mb - 1 - 0 = ks c := by omega
    have e1 : revφ B (ks mb) p (0 :: u) = (p, ks c :: u) := by
      simp only [revφ, if_pos hB, revT, e0]
    obtain ⟨n, hn, en⟩ := node_of_legs _ p hp
    have hc : (cpP cp p).node ∉ B := by
      show cp p.node ∉ B
      rw [en]
      exact hx.cpB n hn (en ▸ hB)
    have e3 : revφ B (ks c) (cpP cp p) u = (cpP cp p, u) := if_neg hc
    simp only [e1, peelφ_B (ks c) p hB u, e3, firstφ_B0 p hB u]
  · have f1 : ks mb - 1 - (j + 1) = ks c - (j + 1) := by omega
    have e1 : revφ B (ks mb) p ((j + 1) :: u) = (p, (ks c - (j + 1)) :: u) := by
      simp only [revφ, if_pos hB, revT, f1]
    have e2 : peelφ B cp (ks c) p ((ks c - (j + 1)) :: u) = (p, (ks c - (j + 1)) :: u) :=
      peelφ_ne (ks c) p _ u (by omega)
    have f3 : ks c - 1 - (ks c - (j + 1)) = j := by omega
    have e3 : revφ B (ks c) p ((ks c - (j + 1)) :: u) = (p, j :: u) := by
      simp only [revφ, if_pos hB, revT, f3]
    simp only [e1, e2, e3, firstφ_Bs p hB j u]
  · have e1 : revφ B (ks mb) p t = (p, t) := if_neg hB
    have e2 : peelφ B cp (ks c) p t = (p, t) := peelφ_O (ks c) p hB t
    have e3 : revφ B (ks c) p t = (p, t) := if_neg hB
    simp only [e1, e2, e3, firstφ_O p hB t]

theorem FirstX.phAt_keep (hx : FirstX L R bid m1 B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∉ B) (ks : ℕ → ℕ) (t : List ℕ) : R.phAt ks n t = L.phAt ks n t := by
  obtain ⟨e1, e2⟩ := hx.phK n hn hB
  rw [phAt_rdVal, phAt_rdVal, e1]
  refine rdVal_keep L R ks _ _ t _ ?_
  cases e : L.cph.find? (fun q => decide (q.1 = n.id)) with
  | none => rfl
  | some q => rw [e] at e2; simpa using e2

theorem FirstX.phAt_shift (hx : FirstX L R bid m1 B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) (ks : ℕ → ℕ) (hm : ks m1 = 1) (j : ℕ) (u : List ℕ) :
    R.phAt ks n (j :: u) = L.phAt ks n ((j + 1) :: u) := by
  rw [phAt_rdVal, phAt_rdVal]
  exact rdVal_shift L R bid m1 n.id ks hm n.ph j u _ _ (hx.phS n hn hB)

theorem FirstX.phAt_first0 (hx : FirstX L R bid m1 B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) (ks : ℕ → ℕ) (u : List ℕ)
    (hu : u.length = (R.base.nodeChain (cp n.id)).length) :
    R.phAt ks (cpNode cp n) u = L.phAt ks n (0 :: u) := by
  rw [phAt_rdVal, phAt_rdVal]
  exact rdVal_first0 L R bid n.id (cp n.id) ks n.ph u hu _ _ (hx.phC n hn hB)

theorem FirstX.dimAt_keep (hx : FirstX L R bid m1 B cp) (h : FirstOK L Lr R Rr bid mb c B cp)
    (p : Port) (hp : p ∈ L.base.legs) (hB : p.node ∉ B) (ks : ℕ → ℕ) (t : List ℕ) :
    R.dimAt ks p t = L.dimAt ks p t := by
  obtain ⟨e1, e2⟩ := hx.dmK p hp hB
  rw [dimAt_rdVal, dimAt_rdVal, e1, h.portDim_keep p hp]
  refine rdVal_keep L R ks _ _ t _ ?_
  cases e : L.cdim.find? (fun q => decide (q.1 = p)) with
  | none => rfl
  | some q => rw [e] at e2; simpa using e2

theorem FirstX.dimAt_shift (hx : FirstX L R bid m1 B cp) (h : FirstOK L Lr R Rr bid mb c B cp)
    (p : Port) (hp : p ∈ L.base.legs) (hB : p.node ∈ B) (ks : ℕ → ℕ) (hm : ks m1 = 1) (j : ℕ)
    (u : List ℕ) : R.dimAt ks p (j :: u) = L.dimAt ks p ((j + 1) :: u) := by
  rw [dimAt_rdVal, dimAt_rdVal, h.portDim_keep p hp]
  exact rdVal_shift L R bid m1 p.node ks hm _ j u _ _ (hx.dmS p hp hB)

theorem FirstX.dimAt_first0 (hx : FirstX L R bid m1 B cp) (h : FirstOK L Lr R Rr bid mb c B cp)
    (p : Port) (hp : p ∈ L.base.legs) (hB : p.node ∈ B) (ks : ℕ → ℕ) (u : List ℕ)
    (hu : u.length = (R.base.nodeChain (cp p.node)).length) :
    R.dimAt ks (cpP cp p) u = L.dimAt ks p (0 :: u) := by
  rw [dimAt_rdVal, dimAt_rdVal, h.portDim_cp p hp hB]
  exact rdVal_first0 L R bid p.node (cp p.node) ks _ u hu _ _ (hx.dmC p hp hB)

/-- The node products agree under `firstφ`, copy-indexed phases read alike. -/
theorem FirstOK.prod (h : FirstOK L Lr R Rr bid mb c B cp) (hx : FirstX L R bid m1 B cp)
    (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (hm : ks m1 = 1) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (firstφ B cp) x)) :
    R.prodL E ks τ = L.prodL E ks τ' := by
  set φ := firstφ B cp
  have hv : ∀ n ∈ L.base.nodes, ∀ t ∈ cidx ks (L.base.nodeChain n.id),
      val E τ' (L.dnodeL ks n t) = val E τ ⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, L.phAt ks n t⟩ := by
    intro n hn t ht
    have hd : L.dnodeL ks n t = ⟨n.kind,
        (List.range n.outs.length).map fun i => dlab ⟨n.id, true, i⟩ t,
        (List.range n.ins.length).map fun i => dlab ⟨n.id, false, i⟩ t, n.dim, L.phAt ks n t⟩ := by
      simp only [dnodeL]
      rw [L.base.dnode_notail h.rL.ports h.rL.wch]
    have e : (⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        n.dim, L.phAt ks n t⟩ : Node) = (L.dnodeL ks n t).rename (relab φ) := by
      rw [hd]
      simp [Node.rename, List.map_map, Function.comp_def, relab_dlab]
    rw [e, val_rename]
    refine val_reads E _ _ _ fun x hy => ?_
    rw [hd] at hy
    simp only [Node.labels, List.mem_append, List.mem_map, List.mem_range] at hy
    have hmem : x ∈ L.base.dlabs ks := by
      rcases hy with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩
      · refine (mem_dlabs _ _ _).2 ⟨_, List.mem_flatMap.2 ⟨n, hn, ?_⟩, t, by rw [h.lc]; exact ht, rfl⟩
        simp [DNode.legs, hi]
      · refine (mem_dlabs _ _ _).2 ⟨_, List.mem_flatMap.2 ⟨n, hn, ?_⟩, t, by rw [h.lc]; exact ht, rfl⟩
        simp [DNode.legs, hi]
    exact hτ x hmem
  have hd' : ∀ m : DNode, ∀ t, R.dnodeL ks m t = ⟨m.kind,
      (List.range m.outs.length).map (fun i => dlab ⟨m.id, true, i⟩ t),
      (List.range m.ins.length).map (fun i => dlab ⟨m.id, false, i⟩ t), m.dim,
      R.phAt ks m t⟩ := by
    intro m t
    simp only [dnodeL]
    rw [R.base.dnode_notail h.rR.ports h.rR.wch]
  set F' : DNode → ℂ := fun m =>
    ((cidx ks (R.base.nodeChain m.id)).map fun t => val E τ (R.dnodeL ks m t)).prod
  have key : ∀ n ∈ L.base.nodes,
      ((cidx ks (L.base.nodeChain n.id)).map fun t => val E τ' (L.dnodeL ks n t)).prod =
        F' n * (if n.id ∈ B then F' (cpNode cp n) else 1) := by
    intro n hn
    by_cases hB : n.id ∈ B
    · obtain ⟨c1, c1'⟩ := h.chB n hn hB
      rw [if_pos hB]
      simp only [F']
      have hcn : (cpNode cp n).id = cp n.id := rfl
      rw [hcn, c1', c1, cidx_first ks mb c _ hk, List.map_append, List.prod_append, List.map_map,
        List.map_map]
      refine (mul_comm _ _).trans ?_
      congr 1
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c _ t).1 ht
        simp only [Function.comp_apply, incT]
        rw [hv n hn _ (by rw [c1]; exact cons_mem_cidx ks mb (j + 1) _ u (by omega) hu), hd',
          hx.phAt_shift n hn hB ks hm j u]
        have hp : ∀ o i, φ ⟨n.id, o, i⟩ ((j + 1) :: u) = (⟨n.id, o, i⟩, j :: u) := fun o i =>
          firstφ_Bs ⟨n.id, o, i⟩ hB j u
        simp [hp]
      · refine congrArg List.prod (List.map_congr_left fun u hu => ?_)
        simp only [Function.comp_apply]
        rw [hv n hn _ (by rw [c1]; exact cons_mem_cidx ks mb 0 _ u (by omega) hu), hd',
          hx.phAt_first0 n hn hB ks u ((mem_cidx ks _ u).1 hu).length_eq]
        have hp : ∀ o i, φ ⟨n.id, o, i⟩ (0 :: u) = (⟨cp n.id, o, i⟩, u) := fun o i =>
          firstφ_B0 ⟨n.id, o, i⟩ hB u
        simp [hp, cpNode]
    · rw [if_neg hB, mul_one]
      simp only [F']
      rw [show R.base.nodeChain n.id = L.base.nodeChain n.id from h.chO n hn hB]
      refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
      rw [hv n hn _ ht, hd', hx.phAt_keep n hn hB ks]
      have hp : ∀ o i, φ ⟨n.id, o, i⟩ t = (⟨n.id, o, i⟩, t) := fun o i =>
        firstφ_O ⟨n.id, o, i⟩ hB t
      simp [hp]
  unfold prodL
  rw [List.map_congr_left key, List.prod_map_mul, prod_ite_filter, h.nodes, List.map_append,
    List.prod_append, List.map_map]
  rfl

/-- A peel of the first copy of an outermost chain box keeps the digit sum `evalX`, `L`'s slots
read from `R`'s through `firstφ`. -/
theorem evalX_first (h : FirstOK L Lr R Rr bid mb c B cp) (hx : FirstX L R bid m1 B cp)
    (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (hm : ks m1 = 1) (σ' : ℕ → List ℕ → ℕ) :
    R.evalX E ks σ' = L.evalX E ks (bndMap L.base R.base (firstφ B cp) σ') := by
  refine evalX_relabelφ L R E ks ks σ' (firstφ B cp) (Diagram.dlabs_nodup _ _ h.rL.legs)
    (Diagram.dlabs_nodup _ _ h.rR.legs) h.rR.bnd ?_ ?_ ?_ ?_ (fun x hx' => ?_) (fun x hx' => ?_)
    (fun y hy => ?_) (fun p hp => h.rL.bl p hp) ?_ ?_ (fun τ τ' hτ => h.prod hx E ks hk hm τ τ' hτ)
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

end first

/-- `firstφ` reads slot digit `(i, t)` of `L` at `psi0`. -/
theorem bndMap_first {L R : LDiagram} {B : List ℕ} {cp : ℕ → ℕ} {mb c : ℕ}
    {sd sd' : SlotData} (hf : PeelFirst L R B cp mb c sd sd') {ks : ℕ → ℕ}
    (σ : ℕ → List ℕ → ℕ) (i : ℕ) (t : List ℕ) (hv : (i, t) ∈ slots sd ks) :
    bndMap L.base R.base (firstφ B cp) σ i t =
      σ (psi0 L R B cp (i, t)).1 (psi0 L R B cp (i, t)).2 := by
  have hp : L.base.bnd.getD i Port.dflt = L.bndAt i := rfl
  unfold bndMap
  simp only [hp]
  by_cases hc : inB L B i = true ∧ t.head? = some 0
  · have hB : (L.bndAt i).node ∈ B := by simpa [inB] using hc.1
    obtain ⟨u, rfl⟩ : ∃ u, t = 0 :: u := by
      cases t with
      | nil => simp at hc
      | cons j u => simp only [List.head?_cons, Option.some.injEq] at hc; exact ⟨u, by rw [hc.2]⟩
    rw [firstφ_B0 _ hB u]
    simp only [psi0, if_pos hc, List.tail_cons, cI]
  · by_cases hb : inB L B i = true
    · obtain ⟨j, u, rfl, hj, -, -⟩ := hf.kept_head hv hb hc
      have hB : (L.bndAt i).node ∈ B := by simpa [inB] using hb
      obtain ⟨j', rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
      rw [firstφ_Bs _ hB j' u]
      simp only [psi0, if_neg hc, if_pos hb, decT, mI, Nat.add_sub_cancel]
    · have hB : (L.bndAt i).node ∉ B := by simpa [inB] using hb
      rw [firstφ_O _ hB t]
      simp only [psi0, if_neg hc, if_neg hb, mI]

/-- A peel of the first copy of an outermost chain box keeps the digit sum `evalX` read through
the flat positions of both sides. -/
theorem evalX_peel_first {L Lr R Rr : LDiagram} {bid mb c m1 : ℕ} {B : List ℕ} {cp : ℕ → ℕ}
    {sd sd' : SlotData} (h : FirstOK L Lr R Rr bid mb c B cp) (hx : FirstX L R bid m1 B cp)
    (hf : PeelFirst L R B cp mb c sd sd') (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (hm : ks m1 = 1) (y : ℕ → ℕ) :
    R.evalX E ks (flatAt sd' ks y) = L.evalX E ks (flatAt sd ks y) := by
  rw [evalX_first h hx E ks hk hm]
  refine evalX_congr L E ks _ _ fun i hi t ht => ?_
  have hv : (i, t) ∈ slots sd ks := by
    refine (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, ?_⟩
    rw [hf.ch i hi, bndAt, List.getD_eq_getElem _ _ hi]
    exact ht
  rw [bndMap_first hf _ i t hv]
  exact congrArg y (hf.fpos_psi hk hv)

end LDiagram

end ArchytasZX
