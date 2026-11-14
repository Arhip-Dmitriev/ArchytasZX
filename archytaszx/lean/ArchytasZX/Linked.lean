/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Relabel

/-!
# Boxed diagrams with chain links and copy-indexed phases

An `LDiagram` is a boxed `Diagram` with chain links and copy-indexed phases. A link `l` of the
box `l.box` joins copy `j`'s port `l.l1` to copy `j + 1`'s port `l.l2`; the last copy's `l.l1`
meets the outside port `l.e1`, the first copy's `l.l2` meets `l.e2`, and at no copies `l.e1`
meets `l.e2`. The link's copy coordinate is the box's place among the node boxes of `l.l1`'s
node; every other coordinate is carried along, so a link of an outer box on ports of inner boxes
runs copy to copy in the inner boxes, and an end port whose chain is the link port's chain
without that coordinate meets it copy to copy (aligned and zipped ends). The template wires at
link ports are not wires of `base`. A copy-indexed phase `(x, y, c, m)` gives node `x` at copy
`t` the phase index `Nat.pair x.ph (t_y + c * ks m)`, `t_y` its copy index in the box `y`.

`evalL` is the digit sum of `Diagram.evalD` with every link pinned digit by digit (`dpairs`) and
the copy-indexed phases read (`dnodeL`); `evalL_plain` gives `evalD` with no links and no
copy-indexed phases. `evalL_relabelφ` is `evalD_relabelφ` for linked diagrams: a map of digits,
one to one and onto, sending digit pairs to digit pairs (`dpairs`) both ways, keeps the sum.
-/

namespace ArchytasZX

/-- A chain link of the box `box`: copy `j`'s `l1` meets copy `j + 1`'s `l2`, the last copy's
`l1` meets `e1`, the first copy's `l2` meets `e2`. -/
structure DLink where
  box : ℕ
  l1 : Port
  l2 : Port
  e1 : Port
  e2 : Port
  deriving DecidableEq, Repr

/-- A boxed diagram with chain links and copy-indexed phases `(node, box, c, m)`. -/
structure LDiagram where
  base : Diagram
  links : List DLink := []
  cph : List (ℕ × ℕ × ℕ × ℕ) := []
  deriving DecidableEq, Repr

/-- A digit: a leg and a copy index. -/
abbrev Dig := Port × List ℕ

namespace LDiagram

variable (L : LDiagram)

/-- The place of the box `y` among the node boxes of the node `x`, outermost `0`. -/
def depth (y x : ℕ) : ℕ := (L.base.nodeBoxes x).findIdx fun b => decide (b.id = y)

/-- The multiplicity index of the box `y`. -/
def multOf (y : ℕ) : ℕ := ((L.base.boxes.find? fun b => decide (b.id = y)).map (·.mult)).getD 0

/-- The digit pairs of the link `l` with copy coordinate `d`, `k` copies and port chain `c`. -/
def lp (ks : ℕ → ℕ) (d k : ℕ) (c : List ℕ) (l : DLink) : List (Dig × Dig) :=
  (cidx ks c).flatMap (fun t =>
    (if t.getD d 0 + 1 < k then [((l.l1, t), (l.l2, t.set d (t.getD d 0 + 1)))]
     else [((l.l1, t), (l.e1, t.eraseIdx d))]) ++
    (if t.getD d 0 = 0 then [((l.e2, t.eraseIdx d), (l.l2, t))] else [])) ++
  (if k = 0 then (cidx ks (c.eraseIdx d)).map fun u => ((l.e2, u), (l.e1, u)) else [])

/-- The digit pairs of the link `l` at the multiplicities `ks`. -/
def linkPairs (ks : ℕ → ℕ) (l : DLink) : List (Dig × Dig) :=
  lp ks (L.depth l.box l.l1.node) (ks (L.multOf l.box)) (L.base.legChain l.l1) l

/-- Every wire's two legs digit by digit, then every link's digit pairs. -/
def dpairs (ks : ℕ → ℕ) : List (Dig × Dig) :=
  L.base.wires.flatMap (fun w => (cidx ks (L.base.legChain w.1)).map fun t => ((w.1, t), (w.2, t))) ++
    L.links.flatMap (L.linkPairs ks)

/-- Every digit pair agreeing, every boundary leg's digits its slot's. -/
def pinsL (ks : ℕ → ℕ) (ν : ℕ → ℕ) : Prop :=
  (∀ x ∈ L.dpairs ks, ν (dlab x.1.1 x.1.2) = ν (dlab x.2.1 x.2.2)) ∧
    (∀ q ∈ L.base.bnd.zipIdx, ∀ t ∈ cidx ks (L.base.legChain q.1), ν (dlab q.1 t) = ν (slab q.2 t))

/-- The phase index of node `n` at copy `t`. -/
def phAt (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : ℕ :=
  match L.cph.find? fun q => decide (q.1 = n.id) with
  | some q => Nat.pair n.ph (t.getD (L.depth q.2.1 n.id) 0 + q.2.2.1 * ks q.2.2.2)
  | none => n.ph

/-- Copy `t` of the node `n` with its copy-indexed phase. -/
def dnodeL (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : Node :=
  { L.base.dnode ks n t with ph := L.phAt ks n t }

/-- The product over the nodes and their copies of the node values at the digits `ν`. -/
noncomputable def prodL (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  (L.base.nodes.map fun n =>
    ((cidx ks (L.base.nodeChain n.id)).map fun t => val E ν (L.dnodeL ks n t)).prod).prod

open Classical in
/-- The linked diagram at the multiplicities `ks`, over digits: slot `i`'s digit `t` is
`σ i t`. -/
noncomputable def evalL (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) : ℂ :=
  sumOver (fun l => E.dimv (L.base.portDim (dport l))) (L.base.dlabs ks)
    (fun ν => if L.pinsL ks ν then L.prodL E ks ν else 0) (slotsD σ)

theorem dnodeL_plain (h : L.cph = []) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    L.dnodeL ks n t = L.base.dnode ks n t := by
  simp [dnodeL, phAt, h, Diagram.dnode]

/-- With no links and no copy-indexed phases the linked digit sum is `evalD`. -/
theorem evalL_plain (hl : L.links = []) (hc : L.cph = []) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : L.evalL E ks σ = L.base.evalD E ks σ := by
  unfold evalL Diagram.evalD
  congr 1
  funext ν
  have hp : L.prodL E ks ν = L.base.prodD E ks ν := by
    simp only [prodL, Diagram.prodD, L.dnodeL_plain hc]
  have hq : L.pinsL ks ν ↔ L.base.pinsD ks ν := by
    simp only [pinsL, Diagram.pinsD, dpairs, hl, List.flatMap_nil, List.append_nil,
      List.mem_flatMap, List.mem_map]
    refine and_congr ⟨fun h w hw t ht => h _ ⟨w, hw, t, ht, rfl⟩, ?_⟩ Iff.rfl
    rintro h _ ⟨w, hw, t, ht, rfl⟩
    exact h w hw t ht
  rw [hp]
  classical
  exact if_congr hq rfl rfl

/-- Two linked digit sums agree when the digits of one are a relabelling `ρ` of the other's,
the slots read alike, and the integrands agree under the relabelling. -/
theorem evalL_relabel (L L' : LDiagram) (E : Env) (ks ks' : ℕ → ℕ) (σ σ' : ℕ → List ℕ → ℕ)
    (ρ : ℕ → ℕ) (hnd : (L.base.dlabs ks).Nodup) (hnd' : (L'.base.dlabs ks').Nodup)
    (hmem : ∀ x, x ∈ L'.base.dlabs ks' ↔ ∃ y ∈ L.base.dlabs ks, ρ y = x)
    (hinj : ∀ x ∈ L.base.dlabs ks, ∀ y ∈ L.base.dlabs ks, ρ x = ρ y → x = y)
    (hdim : ∀ x ∈ L.base.dlabs ks,
      E.dimv (L'.base.portDim (dport (ρ x))) = E.dimv (L.base.portDim (dport x)))
    (hF : ∀ τ τ' : ℕ → ℕ, (∀ x ∈ L.base.dlabs ks, τ' x = τ (ρ x)) →
      (∀ i t, τ (slab i t) = σ' i t) → (∀ i t, τ' (slab i t) = σ i t) →
      (L'.pinsL ks' τ ↔ L.pinsL ks τ') ∧ L'.prodL E ks' τ = L.prodL E ks τ') :
    L'.evalL E ks' σ' = L.evalL E ks σ := by
  have hperm : (L'.base.dlabs ks').Perm ((L.base.dlabs ks).map ρ) := by
    refine (List.perm_ext_iff_of_nodup hnd' (List.Nodup.map_on hinj hnd)).2 fun x => ?_
    rw [hmem x, List.mem_map]
  have hdl : ∀ x ∈ L'.base.dlabs ks', ∃ p t, x = dlab p t := fun x hx => by
    obtain ⟨p, -, t, -, rfl⟩ := (mem_dlabs L'.base ks' x).1 hx
    exact ⟨p, t, rfl⟩
  have hdl0 : ∀ x ∈ L.base.dlabs ks, ∃ p t, x = dlab p t := fun x hx => by
    obtain ⟨p, -, t, -, rfl⟩ := (mem_dlabs L.base ks x).1 hx
    exact ⟨p, t, rfl⟩
  unfold evalL
  rw [sumOver_perm _ hperm]
  refine sumOver_map_on _ _ ρ _ hnd hinj hdim _ _ _ _ (fun τ τ' hτ hτ' hl => ?_) (fun x hx => ?_)
  · have hs : ∀ i t, τ (slab i t) = σ' i t := fun i t => by
      rw [hτ _ fun h => by
        obtain ⟨p, u, e⟩ := hdl _ ((List.Perm.mem_iff hperm).2 h)
        exact dlab_ne_slab p u i t e.symm, slotsD_slab]
    have hs' : ∀ i t, τ' (slab i t) = σ i t := fun i t => by
      rw [hτ' _ fun h => by
        obtain ⟨p, u, e⟩ := hdl0 _ h
        exact dlab_ne_slab p u i t e.symm, slotsD_slab]
    obtain ⟨h1, h2⟩ := hF τ τ' hl hs hs'
    rw [h2]
    classical
    exact if_congr h1 rfl rfl
  · obtain ⟨p, t, rfl⟩ := hdl0 x hx
    obtain ⟨q, u, e⟩ := hdl _ ((hmem _).2 ⟨_, hx, rfl⟩)
    rw [slotsD_dlab, e, slotsD_dlab]

/-- A digit pair through the map of digits `φ`. -/
def pmap (φ : Port → List ℕ → Dig) (x : Dig × Dig) : Dig × Dig :=
  (φ x.1.1 x.1.2, φ x.2.1 x.2.2)

/-- A digit pair with its ends swapped. -/
def pswap (x : Dig × Dig) : Dig × Dig := (x.2, x.1)

/-- `evalD_relabelφ` for linked diagrams: a map `φ` of digits, one to one and onto, keeping
dimensions and boundary legs, sending every digit pair of `L` to a digit pair of `L'` (either
way round) and every digit pair of `L'` back, with agreeing node products, keeps the sum, `L`'s
slot `i` reading `L'`'s slot of the image of its leg. -/
theorem evalL_relabelφ (L L' : LDiagram) (E : Env) (ks ks' : ℕ → ℕ) (σ' : ℕ → List ℕ → ℕ)
    (φ : Port → List ℕ → Dig)
    (hnd : (L.base.dlabs ks).Nodup) (hnd' : (L'.base.dlabs ks').Nodup) (hbnd' : L'.base.bnd.Nodup)
    (hinto : ∀ p ∈ L.base.legs, ∀ t ∈ cidx ks (L.base.legChain p),
      (φ p t).1 ∈ L'.base.legs ∧ (φ p t).2 ∈ cidx ks' (L'.base.legChain (φ p t).1))
    (honto : ∀ q ∈ L'.base.legs, ∀ u ∈ cidx ks' (L'.base.legChain q),
      ∃ p ∈ L.base.legs, ∃ t ∈ cidx ks (L.base.legChain p), φ p t = (q, u))
    (hinj : ∀ p ∈ L.base.legs, ∀ t ∈ cidx ks (L.base.legChain p), ∀ p' ∈ L.base.legs,
      ∀ t' ∈ cidx ks (L.base.legChain p'), φ p t = φ p' t' → p = p' ∧ t = t')
    (hdim : ∀ p ∈ L.base.legs, ∀ t ∈ cidx ks (L.base.legChain p),
      E.dimv (L'.base.portDim (φ p t).1) = E.dimv (L.base.portDim p))
    (hpl : ∀ x ∈ L.dpairs ks, x.1.1 ∈ L.base.legs ∧ x.1.2 ∈ cidx ks (L.base.legChain x.1.1) ∧
      x.2.1 ∈ L.base.legs ∧ x.2.2 ∈ cidx ks (L.base.legChain x.2.1))
    (hD : ∀ x ∈ L.dpairs ks, pmap φ x ∈ L'.dpairs ks' ∨ pswap (pmap φ x) ∈ L'.dpairs ks')
    (hD' : ∀ y ∈ L'.dpairs ks', ∃ x ∈ L.dpairs ks, pmap φ x = y ∨ pswap (pmap φ x) = y)
    (hbl : ∀ p ∈ L.base.bnd, p ∈ L.base.legs)
    (hB : ∀ p ∈ L.base.bnd, ∀ t ∈ cidx ks (L.base.legChain p), (φ p t).1 ∈ L'.base.bnd)
    (hB' : ∀ q ∈ L'.base.bnd, ∀ u ∈ cidx ks' (L'.base.legChain q), ∃ p ∈ L.base.bnd,
      ∃ t ∈ cidx ks (L.base.legChain p), φ p t = (q, u))
    (hP : ∀ τ τ' : ℕ → ℕ, (∀ x ∈ L.base.dlabs ks, τ' x = τ (relab φ x)) →
      L'.prodL E ks' τ = L.prodL E ks τ') :
    L'.evalL E ks' σ' = L.evalL E ks (bndMap L.base L'.base φ σ') := by
  have dl : ∀ p ∈ L.base.legs, ∀ t ∈ cidx ks (L.base.legChain p), dlab p t ∈ L.base.dlabs ks :=
    fun p hp t ht => (mem_dlabs L.base ks _).2 ⟨p, hp, t, ht, rfl⟩
  refine evalL_relabel L L' E ks ks' _ σ' (relab φ) hnd hnd' (fun x => ?_) ?_ ?_ ?_
  · rw [mem_dlabs]
    constructor
    · rintro ⟨q, hq, u, hu, rfl⟩
      obtain ⟨p, hp, t, ht, he⟩ := honto q hq u hu
      exact ⟨_, dl p hp t ht, by rw [relab_dlab, he]⟩
    · rintro ⟨y, hy, rfl⟩
      obtain ⟨p, hp, t, ht, rfl⟩ := (mem_dlabs L.base ks _).1 hy
      obtain ⟨h1, h2⟩ := hinto p hp t ht
      exact ⟨_, h1, _, h2, relab_dlab φ p t⟩
  · intro x hx y hy he
    obtain ⟨p, hp, t, ht, rfl⟩ := (mem_dlabs L.base ks _).1 hx
    obtain ⟨p', hp', t', ht', rfl⟩ := (mem_dlabs L.base ks _).1 hy
    rw [relab_dlab, relab_dlab] at he
    obtain ⟨e1, e2⟩ := dlab_inj he
    obtain ⟨rfl, rfl⟩ := hinj p hp t ht p' hp' t' ht' (Prod.ext e1 e2)
    rfl
  · intro x hx
    obtain ⟨p, hp, t, ht, rfl⟩ := (mem_dlabs L.base ks _).1 hx
    rw [relab_dlab, dport_dlab, dport_dlab]
    exact hdim p hp t ht
  · intro τ τ' hl hs hs'
    refine ⟨?_, hP τ τ' hl⟩
    have hv : ∀ p ∈ L.base.legs, ∀ t ∈ cidx ks (L.base.legChain p),
        τ' (dlab p t) = τ (dlab (φ p t).1 (φ p t).2) := fun p hp t ht => by
      rw [hl _ (dl p hp t ht), relab_dlab]
    have hx : ∀ x ∈ L.dpairs ks, (τ' (dlab x.1.1 x.1.2) = τ' (dlab x.2.1 x.2.2) ↔
        τ (dlab (pmap φ x).1.1 (pmap φ x).1.2) = τ (dlab (pmap φ x).2.1 (pmap φ x).2.2)) :=
      fun x hx => by
        obtain ⟨h1, h2, h3, h4⟩ := hpl x hx
        rw [hv _ h1 _ h2, hv _ h3 _ h4]
        rfl
    unfold pinsL
    refine and_congr ⟨fun h x hx' => ?_, fun h y hy => ?_⟩ ⟨fun h q hq t ht => ?_, ?_⟩
    · rw [hx x hx']
      rcases hD x hx' with hm | hm
      · exact h _ hm
      · exact (h _ hm).symm
    · obtain ⟨x, hx', e | e⟩ := hD' y hy
      · have := (hx x hx').1 (h x hx')
        rw [← e]; exact this
      · have := (hx x hx').1 (h x hx')
        rw [← e]; exact this.symm
    · obtain ⟨p, i⟩ := q
      obtain ⟨hi, rfl⟩ := (mem_zipIdx' _ _ _).1 hq
      have hp : L.base.bnd[i] ∈ L.base.bnd := List.getElem_mem hi
      have hq' := hB _ hp t ht
      set r := φ L.base.bnd[i] t with hr
      have hidx := List.idxOf_lt_length_of_mem hq'
      have hmem : (r.1, L'.base.bnd.idxOf r.1) ∈ L'.base.bnd.zipIdx :=
        (mem_zipIdx' _ _ _).2 ⟨hidx, List.getElem_idxOf hidx⟩
      have hu : r.2 ∈ cidx ks' (L'.base.legChain r.1) := (hinto _ (hbl _ hp) t ht).2
      have := h _ hmem r.2 hu
      simp only at this
      rw [hl _ (dl _ (hbl _ hp) t ht), relab_dlab, hs', bndMap, List.getD_eq_getElem _ _ hi]
      rw [hs] at this
      exact this
    · intro h q hq u hu
      obtain ⟨q, i'⟩ := q
      obtain ⟨hi', rfl⟩ := (mem_zipIdx' _ _ _).1 hq
      obtain ⟨p, hp, t, ht, he⟩ := hB' _ (List.getElem_mem hi') u hu
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      have := h (L.base.bnd[i], i) ((mem_zipIdx' _ _ _).2 ⟨hi, rfl⟩) t ht
      simp only at this
      rw [hl _ (dl _ (hbl _ hp) t ht), relab_dlab, hs', bndMap, List.getD_eq_getElem _ _ hi,
        he] at this
      simp only at this ⊢
      rw [hs]
      rw [List.Nodup.idxOf_getElem hbnd'] at this
      exact this

end LDiagram

end ArchytasZX
