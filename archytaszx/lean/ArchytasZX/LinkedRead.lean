/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedDropX
import ArchytasZX.LinkedSplice

/-!
# Copy-indexed node dimensions and read ghost bounds

`ndimAt L ks n t` is node `n`'s dimension index at copy `t`, read through `L.cnd` as `rdVal` reads
`cph` and `cdim`; `dnodeN` and `prodN` are `dnodeL` and `prodL` with it. A read ghost bound
`(g, box, c, m)` of `gr` gives the ghost `g` the bound `rdVal` of its dimension index (`gbd`).
`evalGXR L gs gr` is `evalGX` with `prodN` and the bounds `gbd`; `evalN L` is it with no ghost
(`evalN_evalW`: `evalW` with `prodN`), and with no node reading they are `evalGX` and `evalX`
(`evalGXR_plain`, `evalN_plain`).

`PeelN`: `PeelX` with the node readings kept on the kept nodes and copied by `rdC` on the copies;
`evalN_peel` and `evalN_peel_flat` are `evalX_peel` and `evalX_peel_flat` over `evalN`.

`GhostN`: `GhostOK` of `L.strip`, no aligned wires, no copy-indexed phase of `x`, and `x`'s legs
read as `x` itself (its `cnd` reading, at the ghost's depth as at `x`'s). `ghostDropN` is
`ghostDropX` without `x`'s node and leg readings; `GhostN.evalGXR_ghost` drops `x` for the ghost
`(g, i, x.dim)` with `x`'s reading as the ghost's read bound (`grAdd`), and `evalGXR_ghostAt`
does it after moving `x` last.

`DropN`: `DropOK` of `L.strip`, no aligned wires, no copy-indexed phase of `x`, `x`'s legs read
as `x`, and the leg `P` it meets read as `x` at `P`'s depth as at `x`'s. `dropIdN` is `dropIdX`
without `x`'s node and leg readings; `DropN.evalN_drop` and `evalN_dropAt` keep `evalN`.
-/

namespace ArchytasZX

namespace LDiagram

/-- The dimension index of node `n` at copy `t`, read through `cnd`. -/
def ndimAt (L : LDiagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : ℕ :=
  L.rdVal ks n.id n.dim t ((L.cnd.find? fun q => decide (q.1 = n.id)).map (·.2))

/-- Copy `t` of the node `n` with its copy-indexed phase and dimension. -/
def dnodeN (L : LDiagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : Node :=
  { L.dnodeL ks n t with dim := L.ndimAt ks n t }

/-- `prodL` over `dnodeN`. -/
noncomputable def prodN (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  (L.base.nodes.map fun n =>
    ((cidx ks (L.base.nodeChain n.id)).map fun t => val E ν (L.dnodeN ks n t)).prod).prod

theorem ndimAt_plain (L : LDiagram) (h : L.cnd = []) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    L.ndimAt ks n t = n.dim := by
  simp [ndimAt, h, rdVal]

theorem dnodeN_nt (L : LDiagram) (hp : ∀ y ∈ L.base.boxes, y.ports = [])
    (hw : ∀ w ∈ L.base.wires, L.base.nodeChain w.1.node = L.base.nodeChain w.2.node)
    (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    L.dnodeN ks n t = ⟨n.kind, (List.range n.outs.length).map fun i => dlab ⟨n.id, true, i⟩ t,
      (List.range n.ins.length).map fun i => dlab ⟨n.id, false, i⟩ t, L.ndimAt ks n t,
      L.phAt ks n t⟩ := by
  simp only [dnodeN]
  rw [L.dnodeL_nt hp hw]

theorem prodN_plain (L : LDiagram) (h : L.cnd = []) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.prodN E ks ν = L.prodL E ks ν := by
  unfold prodN prodL
  refine congrArg List.prod (List.map_congr_left fun n _ =>
    congrArg List.prod (List.map_congr_left fun t _ => ?_))
  have e : L.dnodeN ks n t = L.dnodeL ks n t := by
    rw [dnodeN, ndimAt_plain L h]
    rfl
  rw [e]

/-- Read ghost bounds `(ghost, box, c, m)`. -/
abbrev GReads := List (Port × ℕ × ℕ × ℕ)

/-- The bound index of the ghost `q` at copy `t`, read through `gr`. -/
def gbd (L : LDiagram) (gr : GReads) (ks : ℕ → ℕ) (q : Port × ℕ × ℕ) (t : List ℕ) : ℕ :=
  L.rdVal ks q.1.node q.2.2 t ((gr.find? fun r => decide (r.1 = q.1)).map (·.2))

/-- `pinsG` with the ghost bounds read through `gr`. -/
def pinsGR (L : LDiagram) (gs : Ghosts) (gr : GReads) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    Prop :=
  (∀ x ∈ L.dpairs ks, ν (labG gs x.1.1 x.1.2) = ν (labG gs x.2.1 x.2.2)) ∧
    (∀ q ∈ L.base.bnd.zipIdx, ∀ t ∈ cidx ks (L.base.legChain q.1),
      ν (labG gs q.1 t) = ν (slab q.2 t)) ∧
    ∀ q ∈ gs, ∀ t ∈ cidx ks (L.base.legChain q.1), ν (slab q.2.1 t) < E.dimv (L.gbd gr ks q t)

/-- `pinsGR` over the pairs of `dpairsX`. -/
def pinsGXR (L : LDiagram) (gs : Ghosts) (gr : GReads) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    Prop :=
  (∀ x ∈ L.dpairsX ks, ν (labG gs x.1.1 x.1.2) = ν (labG gs x.2.1 x.2.2)) ∧
    (∀ q ∈ L.base.bnd.zipIdx, ∀ t ∈ cidx ks (L.base.legChain q.1),
      ν (labG gs q.1 t) = ν (slab q.2 t)) ∧
    ∀ q ∈ gs, ∀ t ∈ cidx ks (L.base.legChain q.1), ν (slab q.2.1 t) < E.dimv (L.gbd gr ks q t)

/-- The integrand of `evalGXR`. -/
noncomputable def integGXR (L : LDiagram) (gs : Ghosts) (gr : GReads) (E : Env) (ks : ℕ → ℕ)
    (ν : ℕ → ℕ) : ℂ :=
  open Classical in if L.pinsGXR gs gr E ks ν then L.prodN E ks ν else 0

/-- `evalGX` with the node dimensions of `cnd` and the ghost bounds read through `gr`. -/
noncomputable def evalGXR (L : LDiagram) (gs : Ghosts) (gr : GReads) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : ℂ :=
  sumOver (fun l => E.dimv (L.dimAt ks (dport l) (dtup l))) (L.base.dlabs ks)
    (L.integGXR gs gr E ks) (slotsD σ)

/-- `evalX` with the node dimensions of `cnd`. -/
noncomputable def evalN (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) : ℂ :=
  L.evalGXR [] [] E ks σ

theorem pinsGXR_strip (L : LDiagram) (hal : L.aligned = []) (gs : Ghosts) (gr : GReads)
    (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : L.pinsGXR gs gr E ks ν ↔ L.strip.pinsGR gs gr E ks ν := by
  simp only [pinsGXR, dpairsX, hal, List.flatMap_nil, List.append_nil]
  exact Iff.rfl

theorem integGXR_strip (L : LDiagram) (hal : L.aligned = []) (gs : Ghosts) (gr : GReads)
    (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.integGXR gs gr E ks ν =
      (open Classical in if L.strip.pinsGR gs gr E ks ν then L.prodN E ks ν else 0) := by
  unfold integGXR
  classical
  exact if_congr (pinsGXR_strip L hal gs gr E ks ν) rfl rfl

/-- With no node reading and no read bound `evalGXR` is `evalGX`. -/
theorem evalGXR_plain (L : LDiagram) (h : L.cnd = []) (gs : Ghosts) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : L.evalGXR gs [] E ks σ = L.evalGX gs E ks σ := by
  unfold evalGXR evalGX
  congr 1
  funext ν
  unfold integGXR integGX
  rw [prodN_plain L h]
  classical
  refine if_congr ?_ rfl rfl
  simp [pinsGXR, pinsGX, gbd, rdVal]

/-- `evalN` is `evalW` with `prodN`. -/
theorem evalN_evalW (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalN E ks σ = L.evalW E ks (L.prodN E ks) σ := by
  unfold evalN evalGXR evalW
  congr 1
  funext ν
  unfold integGXR
  classical
  refine if_congr ?_ rfl rfl
  simp only [pinsGXR, pinsX, labG_nil]
  exact ⟨fun h => ⟨h.1, h.2.1⟩, fun h => ⟨h.1, h.2, fun q hq => absurd hq List.not_mem_nil⟩⟩

/-- With no node reading `evalN` is `evalX`. -/
theorem evalN_plain (L : LDiagram) (h : L.cnd = []) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : L.evalN E ks σ = L.evalX E ks σ := by
  rw [evalN, evalGXR_plain L h, evalGX_nil]

/-! ## Peels -/

/-- `PeelX` with the node readings kept on the kept nodes and copied by `rdC` on the copies. -/
structure PeelN (L L' : LDiagram) (bid mb c : ℕ) (B : List ℕ) (cp : ℕ → ℕ) : Prop where
  ok : PeelX L L' bid mb c B cp
  ndK : ∀ n ∈ L.base.nodes, (L'.cnd.find? fun q => decide (q.1 = n.id)) =
      (L.cnd.find? fun q => decide (q.1 = n.id)) ∧
    ((L.cnd.find? fun q => decide (q.1 = n.id)).all fun q =>
      decide (L'.depth q.2.1 n.id = L.depth q.2.1 n.id)) = true
  ndC : ∀ n ∈ L.base.nodes, n.id ∈ B → rdC L L' bid c n.id (cp n.id)
    ((L.cnd.find? fun q => decide (q.1 = n.id)).map (·.2))
    ((L'.cnd.find? fun q => decide (q.1 = cp n.id)).map (·.2)) = true

section peeln

variable {L L' : LDiagram} {bid mb c : ℕ} {B : List ℕ} {cp : ℕ → ℕ}

theorem PeelN.ndimAt_keep (h : PeelN L L' bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (ks : ℕ → ℕ) (t : List ℕ) : L'.ndimAt ks n t = L.ndimAt ks n t := by
  obtain ⟨e1, e2⟩ := h.ndK n hn
  unfold ndimAt
  rw [e1]
  refine rdVal_keep L L' ks _ _ t _ ?_
  cases e : L.cnd.find? (fun q => decide (q.1 = n.id)) with
  | none => rfl
  | some q => rw [e] at e2; simpa using e2

theorem PeelN.ndimAt_copy (h : PeelN L L' bid mb c B cp) (n : DNode) (hn : n ∈ L.base.nodes)
    (hB : n.id ∈ B) (ks : ℕ → ℕ) (u : List ℕ)
    (hu : u.length = (L'.base.nodeChain (cp n.id)).length) :
    L'.ndimAt ks (cpNode cp n) u = L.ndimAt ks n (ks c :: u) := by
  unfold ndimAt
  exact rdVal_copy L L' bid c n.id (cp n.id) (h.ok.d0 n hn hB) ks n.dim u hu _ _ (h.ndC n hn hB)

/-- The node products `prodN` agree under the peel's relabelling. -/
theorem PeelN.prod (h : PeelN L L' bid mb c B cp) (E : Env) (ks : ℕ → ℕ)
    (hk : ks mb = ks c + 1) (τ τ' : ℕ → ℕ)
    (hτ : ∀ x ∈ L.base.dlabs ks, τ' x = τ (relab (peelφ B cp (ks c)) x)) :
    L'.prodN E ks τ = L.prodN E ks τ' := by
  have ho := h.ok.ok
  have lcL : ∀ q, L.base.legChain q = L.base.nodeChain q.node := ho.lc
  set φ := peelφ B cp (ks c)
  have hd : ∀ n t, L.dnodeN ks n t = ⟨n.kind,
      (List.range n.outs.length).map fun i => dlab ⟨n.id, true, i⟩ t,
      (List.range n.ins.length).map fun i => dlab ⟨n.id, false, i⟩ t, L.ndimAt ks n t,
      L.phAt ks n t⟩ := fun n t => L.dnodeN_nt ho.ports ho.wch ks n t
  have hv : ∀ n ∈ L.base.nodes, ∀ t ∈ cidx ks (L.base.nodeChain n.id),
      val E τ' (L.dnodeN ks n t) = val E τ ⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        L.ndimAt ks n t, L.phAt ks n t⟩ := by
    intro n hn t ht
    have e : (⟨n.kind,
        (List.range n.outs.length).map (fun i => dlab (φ ⟨n.id, true, i⟩ t).1 (φ ⟨n.id, true, i⟩ t).2),
        (List.range n.ins.length).map (fun i => dlab (φ ⟨n.id, false, i⟩ t).1 (φ ⟨n.id, false, i⟩ t).2),
        L.ndimAt ks n t, L.phAt ks n t⟩ : Node) = (L.dnodeN ks n t).rename (relab φ) := by
      rw [hd]
      simp [Node.rename, List.map_map, Function.comp_def, relab_dlab]
    rw [e, val_rename]
    refine val_reads E _ _ _ fun x hx => ?_
    rw [hd] at hx
    simp only [Node.labels, List.mem_append, List.mem_map, List.mem_range] at hx
    have hmem : x ∈ L.base.dlabs ks := by
      rcases hx with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩
      · refine (mem_dlabs _ _ _).2 ⟨_, List.mem_flatMap.2 ⟨n, hn, ?_⟩, t, by rw [lcL]; exact ht, rfl⟩
        simp [DNode.legs, hi]
      · refine (mem_dlabs _ _ _).2 ⟨_, List.mem_flatMap.2 ⟨n, hn, ?_⟩, t, by rw [lcL]; exact ht, rfl⟩
        simp [DNode.legs, hi]
    exact hτ x hmem
  have hd' : ∀ m : DNode, ∀ t, L'.dnodeN ks m t = ⟨m.kind,
      (List.range m.outs.length).map (fun i => dlab ⟨m.id, true, i⟩ t),
      (List.range m.ins.length).map (fun i => dlab ⟨m.id, false, i⟩ t), L'.ndimAt ks m t,
      L'.phAt ks m t⟩ := fun m t => L'.dnodeN_nt ho.ports' ho.wch' ks m t
  set F' : DNode → ℂ := fun m =>
    ((cidx ks (L'.base.nodeChain m.id)).map fun t => val E τ (L'.dnodeN ks m t)).prod
  have key : ∀ n ∈ L.base.nodes,
      ((cidx ks (L.base.nodeChain n.id)).map fun t => val E τ' (L.dnodeN ks n t)).prod =
        F' n * (if n.id ∈ B then F' (cpNode cp n) else 1) := by
    intro n hn
    by_cases hB : n.id ∈ B
    · obtain ⟨c1, c1'⟩ := ho.chB n hn hB
      rw [if_pos hB]
      simp only [F']
      rw [show L'.base.nodeChain n.id = c :: L'.base.nodeChain (cp n.id) from c1',
        show L.base.nodeChain n.id = mb :: L'.base.nodeChain (cp n.id) from c1,
        cidx_succ ks mb c _ hk, List.map_append, List.prod_append, List.map_map]
      have hcn : (cpNode cp n).id = cp n.id := rfl
      rw [hcn]
      congr 1
      · refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
        obtain ⟨j, u, rfl, hj, hu⟩ := (mem_cidx_cons ks c _ t).1 ht
        rw [hv n hn _ (by
          rw [show L.base.nodeChain n.id = mb :: L'.base.nodeChain (cp n.id) from c1,
            cidx_succ ks mb c _ hk]; exact List.mem_append_left _ ht), hd',
          h.ok.phAt_keep n hn ks, h.ndimAt_keep n hn ks]
        simp [φ, peelφ_ne (ks c) _ j u (by omega)]
      · refine congrArg List.prod (List.map_congr_left fun u hu => ?_)
        simp only [Function.comp_apply]
        have hm : ks c :: u ∈ cidx ks (L.base.nodeChain n.id) := by
          rw [show L.base.nodeChain n.id = mb :: L'.base.nodeChain (cp n.id) from c1,
            cidx_succ ks mb c _ hk]
          exact List.mem_append_right _ (List.mem_map.2 ⟨u, hu, rfl⟩)
        rw [hv n hn _ hm, hd', h.ok.phAt_copy n hn hB ks u ((mem_cidx ks _ u).1 hu).length_eq,
          h.ndimAt_copy n hn hB ks u ((mem_cidx ks _ u).1 hu).length_eq]
        have hp : ∀ o i, φ ⟨n.id, o, i⟩ (ks c :: u) = (⟨cp n.id, o, i⟩, u) := fun o i =>
          peelφ_B (ks c) ⟨n.id, o, i⟩ hB u
        simp [hp, cpNode]
    · rw [if_neg hB, mul_one]
      simp only [F']
      rw [show L'.base.nodeChain n.id = L.base.nodeChain n.id from ho.chO n hn hB]
      refine congrArg List.prod (List.map_congr_left fun t ht => ?_)
      rw [hv n hn _ ht, hd', h.ok.phAt_keep n hn ks, h.ndimAt_keep n hn ks]
      have hp : ∀ o i, φ ⟨n.id, o, i⟩ t = (⟨n.id, o, i⟩, t) := fun o i =>
        peelφ_O (ks c) ⟨n.id, o, i⟩ hB t
      simp [hp]
  unfold prodN
  rw [List.map_congr_left key, List.prod_map_mul, prod_ite_filter,
    show L'.base.nodes = L.base.nodes ++ (L.base.nodes.filter fun n => n.id ∈ B).map (cpNode cp)
      from ho.nodes, List.map_append, List.prod_append, List.map_map]
  rfl

/-- Peeling the last copy of an outermost chain box keeps `evalN`, `L`'s slots read from `L'`'s
through `peelφ`. -/
theorem evalN_peel (h : PeelN L L' bid mb c B cp) (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1)
    (σ' : ℕ → List ℕ → ℕ) :
    L'.evalN E ks σ' = L.evalN E ks (bndMap L.base L'.base (peelφ B cp (ks c)) σ') := by
  rw [evalN_evalW, evalN_evalW]
  exact evalW_peel h.ok E ks hk _ _ (fun τ τ' hτ => h.prod E ks hk τ τ' hτ) σ'

end peeln

/-- `evalN` reads only the slot digits of its boundary legs' copies. -/
theorem evalN_congr (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ σ' : ℕ → List ℕ → ℕ)
    (h : ∀ i (hi : i < L.base.bnd.length), ∀ t ∈ cidx ks (L.base.legChain L.base.bnd[i]),
      σ i t = σ' i t) :
    L.evalN E ks σ = L.evalN E ks σ' := by
  rw [evalN_evalW, evalN_evalW]
  unfold evalW
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
    have hP : L.prodN E ks τ = L.prodN E ks τ' := by
      unfold prodN
      refine congrArg List.prod (List.map_congr_left fun n _ =>
        congrArg List.prod (List.map_congr_left fun t _ => ?_))
      apply val_congr
      · exact List.map_congr_left fun x hx => hτ _ (Or.inl ((dnodeL_even L ks n t).1 x hx))
      · exact List.map_congr_left fun x hx => hτ _ (Or.inl ((dnodeL_even L ks n t).2 x hx))
    rw [hP]
    classical
    exact if_congr hp rfl rfl
  · rcases hx with hx | ⟨i, hi, t, ht, rfl⟩
    · simp [slotsD, hx]
    · rw [slotsD_slab, slotsD_slab]
      exact h i hi t ht

/-- A peel keeps `evalN` read through the flat positions of both sides. -/
theorem evalN_peel_flat {L L' : LDiagram} {bid mb c : ℕ} {B : List ℕ} {cp : ℕ → ℕ}
    {sd sd' : SlotData} (h : PeelN L L' bid mb c B cp) (hf : PeelFlat L L' B cp mb c sd sd')
    (E : Env) (ks : ℕ → ℕ) (hk : ks mb = ks c + 1) (y : ℕ → ℕ) :
    L'.evalN E ks (flatAt sd' ks y) = L.evalN E ks (flatAt sd ks y) := by
  rw [evalN_peel h E ks hk]
  refine evalN_congr L E ks _ _ fun i hi t ht => ?_
  rw [bndMap_psi]
  have hv : (i, t) ∈ slots sd ks := by
    refine (mem_slots _ _ _ _).2 ⟨hf.n ▸ hi, ?_⟩
    rw [hf.ch i hi, bndAt, List.getD_eq_getElem _ _ hi]
    exact ht
  exact congrArg y (hf.fpos_psi hk hv)

/-! ## Moving nodes -/

theorem ndimAt_wn (L : LDiagram) (ns : List DNode) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (L.withNodes ns).ndimAt ks n t = L.ndimAt ks n t := by
  show (L.withNodes ns).rdVal ks n.id n.dim t ((L.cnd.find? fun q => decide (q.1 = n.id)).map (·.2)) =
    L.rdVal ks n.id n.dim t ((L.cnd.find? fun q => decide (q.1 = n.id)).map (·.2))
  cases (L.cnd.find? fun q => decide (q.1 = n.id)).map (·.2) with
  | none => rfl
  | some q => simp only [rdVal, depth_wn]

theorem prodN_wn (L : LDiagram) (ns : List DNode) (hp : ns.Perm L.base.nodes) (E : Env)
    (ks : ℕ → ℕ) (ν : ℕ → ℕ) : (L.withNodes ns).prodN E ks ν = L.prodN E ks ν := by
  have e : ∀ n t, (L.withNodes ns).dnodeN ks n t = L.dnodeN ks n t := fun n t => by
    simp only [dnodeN, dnodeL_wn, ndimAt_wn]
  unfold prodN
  simp only [e, nodeChain_wn]
  exact (hp.map fun (n : DNode) => ((cidx ks (L.base.nodeChain n.id)).map fun t =>
    val E ν (L.dnodeN ks n t)).prod).prod_eq

theorem gbd_wn (L : LDiagram) (ns : List DNode) (gr : GReads) (ks : ℕ → ℕ) (q : Port × ℕ × ℕ)
    (t : List ℕ) : (L.withNodes ns).gbd gr ks q t = L.gbd gr ks q t := by
  unfold gbd
  cases (gr.find? fun r => decide (r.1 = q.1)).map (·.2) with
  | none => rfl
  | some r => simp only [rdVal, depth_wn]

theorem pinsGXR_wn (L : LDiagram) (ns : List DNode) :
    (L.withNodes ns).pinsGXR = L.pinsGXR := by
  funext gs gr E ks ν
  have e : (L.withNodes ns).gbd = L.gbd := by
    funext gr ks q t
    exact gbd_wn L ns gr ks q t
  unfold pinsGXR dpairsX alignPairs dpairs linkPairs
  rw [legChain_wn, depth_wn, e]
  rfl

/-- A permutation of the nodes keeps `evalGXR`. -/
theorem evalGXR_withNodes (L : LDiagram) (ns : List DNode) (hp : ns.Perm L.base.nodes)
    (hid : (L.base.nodes.map (·.id)).Nodup) (gs : Ghosts) (gr : GReads) (E : Env) (ks : ℕ → ℕ)
    (σ : ℕ → List ℕ → ℕ) : (L.withNodes ns).evalGXR gs gr E ks σ = L.evalGXR gs gr E ks σ := by
  have hl : ((L.withNodes ns).base.dlabs ks).Perm (L.base.dlabs ks) := by
    unfold Diagram.dlabs Diagram.legs
    rw [legChain_wn]
    exact (hp.flatMap_right DNode.legs).flatMap_right
      (fun p => (cidx ks (L.base.legChain p)).map (dlab p))
  have hd : (fun l => E.dimv ((L.withNodes ns).dimAt ks (dport l) (dtup l))) =
      fun l => E.dimv (L.dimAt ks (dport l) (dtup l)) :=
    funext fun l => by rw [dimAt_wn L ns hp hid]
  unfold evalGXR
  rw [sumOver_perm _ hl, hd]
  congr 1
  funext ν
  unfold integGXR
  rw [prodN_wn L ns hp E ks ν, pinsGXR_wn]

/-! ## Ghosts -/

/-- The ghost read bounds `gr` with `x`'s node reading as the bound of the ghost `g`. -/
def grAdd (L : LDiagram) (x : ℕ) (g : Port) (gr : GReads) : GReads :=
  match L.cnd.find? fun q => decide (q.1 = x) with
  | some q => (g, q.2) :: gr
  | none => gr

/-- `ghostDropX` without the node reading of `x` and the leg readings of its legs. -/
def ghostDropN (L : LDiagram) (x : ℕ) (a c g : Port) : LDiagram :=
  { ({ L.ghostDropX x a c g with cdim := L.cdim.filter (fun q => q.1.node ≠ x) } : LDiagram) with
    cnd := L.cnd.filter (fun q => q.1 ≠ x) }

/-- `GhostOK` of `L` without its readings, no aligned wires, no copy-indexed phase of `x`, `x`'s
legs read as `x`, and `x`'s reading at the ghost's depth as at `x`'s. -/
structure GhostN (L : LDiagram) (gs : Ghosts) (gr : GReads) (x : DNode) (a c g : Port) (i : ℕ) :
    Prop where
  ok : GhostOK L.strip gs x a c g i
  al : L.aligned = []
  xph : ∀ q ∈ L.cph, q.1 ≠ x.id
  xo : (L.cdim.find? fun q => decide (q.1 = ⟨x.id, true, 0⟩)).map (·.2) =
    (L.cnd.find? fun q => decide (q.1 = x.id)).map (·.2)
  xi : (L.cdim.find? fun q => decide (q.1 = ⟨x.id, false, 0⟩)).map (·.2) =
    (L.cnd.find? fun q => decide (q.1 = x.id)).map (·.2)
  gd : ((L.cnd.find? fun q => decide (q.1 = x.id)).all fun q =>
    decide (L.depth q.2.1 g.node = L.depth q.2.1 x.id)) = true
  gfr : ∀ r ∈ gr, r.1 ≠ g

theorem find?_filter_ofN {α : Type} (l : List α) (r p : α → Bool)
    (h : ∀ a, p a = true → r a = true) : (l.filter r).find? p = l.find? p := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    by_cases hr : r a = true
    · simp only [List.filter_cons, hr, ite_true, List.find?_cons, ih]
    · have hp : p a = false := by
        cases e : p a
        · rfl
        · exact absurd (h a e) hr
      simp only [List.filter_cons, hr, List.find?_cons, hp, ih, Bool.false_eq_true, ite_false]

theorem find?_filter_ne (l : List (ℕ × ℕ × ℕ × ℕ)) (x y : ℕ) (h : y ≠ x) :
    (l.filter fun q => q.1 ≠ x).find? (fun q => decide (q.1 = y)) =
      l.find? fun q => decide (q.1 = y) :=
  find?_filter_ofN l _ _ fun a ha => by
    simp only [decide_eq_true_eq] at ha ⊢
    rw [ha]
    exact h

theorem find?_filter_neP (l : List (Port × ℕ × ℕ × ℕ)) (x : ℕ) (p : Port) (h : p.node ≠ x) :
    (l.filter fun q => q.1.node ≠ x).find? (fun q => decide (q.1 = p)) =
      l.find? fun q => decide (q.1 = p) :=
  find?_filter_ofN l _ _ fun a ha => by
    simp only [decide_eq_true_eq] at ha ⊢
    rw [ha]
    exact h

theorem pinsL_congr {L1 L2 : LDiagram} (h1 : L1.base = L2.base) (h2 : L1.links = L2.links) :
    L1.pinsL = L2.pinsL := by
  obtain ⟨b1, l1, _, _, _, _⟩ := L1
  obtain ⟨b2, l2, _, _, _, _⟩ := L2
  simp only at h1 h2
  subst h1 h2
  rfl

theorem pinsGR_congr {L1 L2 : LDiagram} (h1 : L1.base = L2.base) (h2 : L1.links = L2.links) :
    L1.pinsGR = L2.pinsGR := by
  obtain ⟨b1, l1, _, _, _, _⟩ := L1
  obtain ⟨b2, l2, _, _, _, _⟩ := L2
  simp only at h1 h2
  subst h1 h2
  rfl

section ghostN

variable {L : LDiagram} {gs : Ghosts} {gr : GReads} {x : DNode} {a c g : Port} {i : ℕ}

local notation "Lx" => L.ghostDropN x.id a c g

local notation "Lg" => L.strip.ghostDrop x.id a c g

theorem depth_gdN (y b z : ℕ) (a c g : Port) : (L.ghostDropN y a c g).depth b z = L.depth b z :=
  depth_gd (L := L) y b z a c g

theorem rdVal_gdN (y : ℕ) (a c g : Port) (ks : ℕ → ℕ) (z k : ℕ) (t : List ℕ)
    (o : Option (ℕ × ℕ × ℕ)) : (L.ghostDropN y a c g).rdVal ks z k t o = L.rdVal ks z k t o := by
  cases o with
  | none => rfl
  | some q => simp only [rdVal, depth_gdN]

theorem phAt_gdN (y : ℕ) (a c g : Port) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (L.ghostDropN y a c g).phAt ks n t = L.phAt ks n t := by
  rw [phAt_rdVal, phAt_rdVal, rdVal_gdN]
  rfl

theorem ndimAt_gdN (y : ℕ) (a c g : Port) (ks : ℕ → ℕ) (n : DNode) (hn : n.id ≠ y)
    (t : List ℕ) : (L.ghostDropN y a c g).ndimAt ks n t = L.ndimAt ks n t := by
  unfold ndimAt
  rw [rdVal_gdN]
  show L.rdVal ks n.id n.dim t
      (((L.cnd.filter fun q => q.1 ≠ y).find? fun q => decide (q.1 = n.id)).map (·.2)) = _
  rw [find?_filter_ne L.cnd y n.id hn]

theorem gbd_gdN (y : ℕ) (a c g : Port) (gr : GReads) (ks : ℕ → ℕ) (q : Port × ℕ × ℕ)
    (t : List ℕ) : (L.ghostDropN y a c g).gbd gr ks q t = L.gbd gr ks q t := by
  unfold gbd
  rw [rdVal_gdN]

theorem GhostN.alX (h : GhostN L gs gr x a c g i) : (Lx).aligned = [] := h.al

theorem GhostN.dimAt_x (h : GhostN L gs gr x a c g i) (ks : ℕ → ℕ) (b : Bool) (t : List ℕ) :
    L.dimAt ks ⟨x.id, b, 0⟩ t = L.ndimAt ks x t := by
  rw [dimAt_rdVal, show L.base.portDim ⟨x.id, b, 0⟩ = x.dim from h.ok.portDim_x b]
  unfold ndimAt
  cases b
  · rw [h.xi]
  · rw [h.xo]

theorem GhostN.dimAt_gd (h : GhostN L gs gr x a c g i) (ks : ℕ → ℕ) (p : Port)
    (hp : p.node ≠ x.id) (t : List ℕ) : (Lx).dimAt ks p t = L.dimAt ks p t := by
  have e : (Lx).base.portDim p = L.base.portDim p := h.ok.portDim_gd p hp
  rw [dimAt_rdVal, dimAt_rdVal, rdVal_gdN, e]
  show L.rdVal ks p.node (L.base.portDim p) t
      (((L.cdim.filter fun q => q.1.node ≠ x.id).find? fun q => decide (q.1 = p)).map (·.2)) = _
  rw [find?_filter_neP L.cdim x.id p hp]

theorem GhostN.prodN_eq (h : GhostN L gs gr x a c g i) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.prodN E ks ν = (Lx).prodN E ks ν * ((cidx ks (L.base.nodeChain x.id)).map fun t =>
      tZ (E.dimv (L.ndimAt ks x t)) (E.phase x.ph)
        [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod := by
  have ho := h.ok
  have hlast : L.base.nodes = (L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x] := ho.last
  have hx : ∀ t, L.phAt ks x t = x.ph := fun t => by
    have hn : L.cph.find? (fun q => decide (q.1 = x.id)) = none :=
      List.find?_eq_none.2 fun q hq e => h.xph q hq (of_decide_eq_true e)
    rw [phAt_rdVal, hn]
    rfl
  unfold prodN
  conv_lhs => rw [hlast]
  rw [List.map_append, List.prod_append, List.map_singleton, List.prod_singleton]
  congr 1
  · refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    have hn' : n.id ≠ x.id := by
      have hn'' : n ∈ L.base.nodes.filter fun n => n.id ≠ x.id := hn
      rw [List.mem_filter, decide_eq_true_eq] at hn''
      exact hn''.2
    rw [show (Lx).base.nodeChain n.id = L.base.nodeChain n.id from
      nodeChain_gd (L := L) x.id n.id a c g]
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeN_nt ho.ports ho.wch, (Lx).dnodeN_nt ho.ports ho.wch', phAt_gdN,
      ndimAt_gdN x.id a c g ks n hn']
  · refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeN_nt ho.ports ho.wch, hx]
    simp [val, ho.kind, ho.outs, ho.ins]

theorem GhostOK.pins_iffR (h : GhostOK L gs x a c g i) (gr gr' : GReads) (R : List ℕ → ℕ)
    (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ)
    (ha : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (dlab a t) = ν (slab i t))
    (hc : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (dlab c t) = ν (slab i t))
    (hd : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (slab i t) < E.dimv (R t))
    (hg : ∀ t, (L.ghostDrop x.id a c g).gbd gr' ks (g, i, x.dim) t = R t)
    (hgq : ∀ q ∈ gs, ∀ t, (L.ghostDrop x.id a c g).gbd gr' ks q t = L.gbd gr ks q t) :
    L.pinsGR gs gr E ks ν ↔ (L.ghostDrop x.id a c g).pinsGR ((g, i, x.dim) :: gs) gr' E ks ν := by
  have hsub : ∀ p t, p ≠ c → p ≠ g → (p = a → t ∈ cidx ks (L.base.nodeChain x.id)) →
      ν (labG ((g, i, x.dim) :: gs) (substG a c g p) t) = ν (labG gs p t) := by
    intro p t hpc hpg hpa
    by_cases hpa' : p = a
    · subst hpa'
      rw [show substG p c g p = g from if_pos (Or.inl rfl), labG_self,
        labG_none (fun q hq => (h.gsf q hq).1), ha t (hpa rfl)]
    · rw [h.substG_of hpa' hpc, labG_ne hpg]
  have hlq : ∀ q : Port, (L.ghostDrop x.id a c g).base.legChain q = L.base.legChain q :=
    fun q => by rw [h.lc', h.lc]
  obtain ⟨hi, hci⟩ := List.getElem?_eq_some_iff.1 h.ci
  unfold pinsGR
  refine ⟨fun hp => ⟨fun y hy => ?_, fun q hq t ht => ?_, ?_⟩,
    fun hp => ⟨fun y hy => ?_, fun q hq t ht => ?_, fun q hq t ht => ?_⟩⟩
  · rw [h.dpairs_gd, List.mem_map] at hy
    obtain ⟨y0, hy0, rfl⟩ := hy
    obtain ⟨⟨p1, p2, p3⟩, ⟨q1, q2, q3⟩⟩ := h.pair_ports ks y0 hy0
    simp only [portsP]
    rw [hsub _ _ p1 p2 p3, hsub _ _ q1 q2 q3]
    exact hp.1 y0 hy0
  · obtain ⟨p, j⟩ := q
    rw [h.bnd_gd] at hq
    obtain ⟨hj, e⟩ := (mem_zipIdx' _ _ _).1 hq
    rw [List.length_map] at hj
    rw [List.getElem_map] at e
    subst e
    by_cases hji : j = i
    · subst hji
      simp only [hci, show substG a c g c = g from if_pos (Or.inr rfl), labG_self]
    · have hpc : L.base.bnd[j] ≠ c := fun e' =>
        hji ((List.Nodup.getElem_inj_iff h.bnd).1 (e'.trans hci.symm))
      have hpa : L.base.bnd[j] ≠ a := fun e' => h.ab (e' ▸ List.getElem_mem hj)
      have hpg : L.base.bnd[j] ≠ g := fun e' => h.gb (e' ▸ List.getElem_mem hj)
      simp only at ht ⊢
      rw [h.substG_of hpa hpc] at ht ⊢
      rw [labG_ne hpg]
      exact hp.2.1 _ ((mem_zipIdx' _ _ _).2 ⟨hj, rfl⟩) t (by rw [h.lc, ← h.lc']; exact ht)
  · intro q hq' t ht
    rcases List.mem_cons.1 hq' with rfl | hq'
    · rw [h.gchain] at ht
      rw [hg]
      exact hd t ht
    · rw [hlq] at ht
      rw [hgq q hq' t]
      exact hp.2.2 q hq' t ht
  · have := hp.1 _ (by rw [h.dpairs_gd]; exact List.mem_map.2 ⟨y, hy, rfl⟩)
    obtain ⟨⟨p1, p2, p3⟩, ⟨q1, q2, q3⟩⟩ := h.pair_ports ks y hy
    simp only [portsP] at this
    rwa [hsub _ _ p1 p2 p3, hsub _ _ q1 q2 q3] at this
  · obtain ⟨p, j⟩ := q
    obtain ⟨hj, e⟩ := (mem_zipIdx' _ _ _).1 hq
    subst e
    by_cases hji : j = i
    · subst hji
      simp only [hci] at ht ⊢
      rw [h.lc, h.cnode] at ht
      rw [labG_none (fun q hq => (h.gsf q hq).2.1), hc t ht]
    · have hpc : L.base.bnd[j] ≠ c := fun e' =>
        hji ((List.Nodup.getElem_inj_iff h.bnd).1 (e'.trans hci.symm))
      have hpa : L.base.bnd[j] ≠ a := fun e' => h.ab (e' ▸ List.getElem_mem hj)
      have hpg : L.base.bnd[j] ≠ g := fun e' => h.gb (e' ▸ List.getElem_mem hj)
      have hq' : (substG a c g L.base.bnd[j], j) ∈ (L.ghostDrop x.id a c g).base.bnd.zipIdx := by
        rw [h.bnd_gd]
        exact (mem_zipIdx' _ _ _).2 ⟨by simpa using hj, by simp⟩
      have := hp.2.1 _ hq' t (by rw [h.lc', ← h.lc, h.substG_of hpa hpc]; exact ht)
      simp only at this ⊢
      rwa [h.substG_of hpa hpc, labG_ne hpg] at this
  · rw [← hlq] at ht
    have := hp.2.2 q (List.mem_cons_of_mem _ hq) t ht
    rwa [hgq q hq t] at this

theorem GhostN.gbd_g (h : GhostN L gs gr x a c g i) (ks : ℕ → ℕ) (t : List ℕ) :
    (Lg).gbd (L.grAdd x.id g gr) ks (g, i, x.dim) t = L.ndimAt ks x t := by
  have hgr : (gr.find? fun r => decide (r.1 = g)) = none :=
    List.find?_eq_none.2 fun r hr e => h.gfr r hr (of_decide_eq_true e)
  have e1 : (Lg).depth = L.depth := funext fun b => funext fun z => depth_gd (L := L.strip) _ b z a c g
  unfold gbd ndimAt grAdd
  have gd := h.gd
  cases e : L.cnd.find? (fun q => decide (q.1 = x.id)) with
  | none => simp [hgr, rdVal]
  | some q =>
    rw [e] at gd
    simp only [Option.all_some, decide_eq_true_eq] at gd
    simp [rdVal, e1, gd]

theorem GhostN.gbd_q (h : GhostN L gs gr x a c g i) (ks : ℕ → ℕ) (q : Port × ℕ × ℕ)
    (hq : q ∈ gs) (t : List ℕ) : (Lg).gbd (L.grAdd x.id g gr) ks q t = L.strip.gbd gr ks q t := by
  have hqg : q.1 ≠ g := (h.ok.gsf q hq).2.2
  have e1 : (Lg).depth = L.strip.depth :=
    funext fun b => funext fun z => depth_gd (L := L.strip) _ b z a c g
  have hf : ((L.grAdd x.id g gr).find? fun r => decide (r.1 = q.1)) =
      gr.find? fun r => decide (r.1 = q.1) := by
    unfold grAdd
    cases L.cnd.find? (fun q => decide (q.1 = x.id)) with
    | none => rfl
    | some q' =>
      have : ¬ g = q.1 := fun e => hqg e.symm
      simp [List.find?_cons, this]
  unfold gbd
  rw [hf]
  cases (gr.find? fun r => decide (r.1 = q.1)).map (·.2) with
  | none => rfl
  | some r => simp only [rdVal, e1]

theorem GhostN.integ_reads (h : GhostN L gs gr x a c g i) (E : Env) (ks : ℕ → ℕ)
    (ν ν' : ℕ → ℕ) (hag : ∀ l ∉ L.xdigs ks x, ν l = ν' l) :
    (Lx).integGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks ν =
      (Lx).integGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks ν' := by
  have ho := h.ok
  have hl : ∀ p t, p ≠ a → p ≠ c → ∀ gs' : Ghosts, ν (labG gs' p t) = ν' (labG gs' p t) := by
    intro p t ha hc gs'
    unfold labG
    split
    · exact hag _ (slab_not_xdigs L ks x _ _)
    · exact hag _ (ho.not_xdigs' ks p ha hc t)
  have hs : ∀ j t, ν (slab j t) = ν' (slab j t) := fun j t => hag _ (slab_not_xdigs L ks x j t)
  have hne : ∀ p, substG a c g p ≠ a ∧ substG a c g p ≠ c := by
    intro p
    unfold substG
    split_ifs with hp
    · exact ⟨ho.ga, ho.gc⟩
    · exact ⟨fun e => hp (Or.inl e), fun e => hp (Or.inr e)⟩
  have hpins : (Lx).pinsGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks ν ↔
      (Lx).pinsGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks ν' := by
    rw [pinsGXR_strip _ h.alX, pinsGXR_strip _ h.alX,
      show (Lx).strip.pinsGR = (Lg).pinsGR from pinsGR_congr rfl rfl]
    unfold pinsGR
    refine and_congr (forall₂_congr fun y hy => ?_) (and_congr (forall₂_congr fun q hq =>
      forall₂_congr fun t _ => ?_) (forall₂_congr fun q _ => forall₂_congr fun t _ => by rw [hs]))
    · rw [ho.dpairs_gd, List.mem_map] at hy
      obtain ⟨y0, -, rfl⟩ := hy
      simp only [portsP]
      rw [hl _ _ (hne _).1 (hne _).2, hl _ _ (hne _).1 (hne _).2]
    · have hq' : q.1 ∈ (Lg).base.bnd :=
        (List.mem_zipIdx hq).2.2 ▸ List.getElem_mem _
      rw [ho.bnd_gd, List.mem_map] at hq'
      obtain ⟨q0, -, e⟩ := hq'
      rw [← e, hl _ _ (hne _).1 (hne _).2, hs]
  have hprod : (Lx).prodN E ks ν = (Lx).prodN E ks ν' := by
    unfold prodN
    refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [(Lx).dnodeN_nt ho.ports ho.wch']
    have hn' : n.id ≠ x.id := by
      have hn'' : n ∈ L.base.nodes.filter fun n => n.id ≠ x.id := hn
      rw [List.mem_filter, decide_eq_true_eq] at hn''
      exact hn''.2
    apply val_congr
    · simp only [List.map_map]
      exact List.map_congr_left fun j _ => hag _ (ho.not_xdigs ks _ hn' _)
    · simp only [List.map_map]
      exact List.map_congr_left fun j _ => hag _ (ho.not_xdigs ks _ hn' _)
  unfold integGXR
  rw [hprod]
  classical
  exact if_congr hpins rfl rfl

theorem GhostN.integ_point (h : GhostN L gs gr x a c g i) (E : Env)
    (hE : ∀ k, E.phase x.ph k = 0) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.integGXR gs gr E ks ν =
      if ∀ l ∈ L.xdigs ks x, ν l = ν (slab i (dtup l)) then
        (Lx).integGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks ν
      else 0 := by
  have ho := h.ok
  have hox : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab ⟨x.id, true, 0⟩ t ∈ L.xdigs ks x :=
    fun t ht => (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inl rfl⟩
  have hix : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab ⟨x.id, false, 0⟩ t ∈ L.xdigs ks x :=
    fun t ht => (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inr rfl⟩
  have hax : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab a t ∈ L.xdigs ks x := fun t ht => by
    rcases ho.ac with ⟨rfl, -⟩ | ⟨rfl, -⟩
    · exact hox t ht
    · exact hix t ht
  have hcx : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab c t ∈ L.xdigs ks x := fun t ht => by
    rcases ho.ac with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact hix t ht
    · exact hox t ht
  rw [integGXR_strip _ h.al, integGXR_strip _ h.alX,
    show (Lx).strip.pinsGR = (Lg).pinsGR from pinsGR_congr rfl rfl]
  by_cases hA : ∀ l ∈ L.xdigs ks x, ν l = ν (slab i (dtup l))
  · rw [if_pos hA]
    have hv : ∀ t p, dlab p t ∈ L.xdigs ks x → ν (dlab p t) = ν (slab i t) :=
      fun t p hp => by rw [hA _ hp, dtup_dlab]
    by_cases hd : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (slab i t) < E.dimv (L.ndimAt ks x t)
    · have hone : ((cidx ks (L.base.nodeChain x.id)).map fun t =>
          tZ (E.dimv (L.ndimAt ks x t)) (E.phase x.ph)
            [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod = 1 := by
        refine List.prod_eq_one fun z hz => ?_
        obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hz
        rw [tZ_id_pair, if_pos ⟨by rw [hv t _ (hox t ht), hv t _ (hix t ht)],
          by rw [hv t _ (hox t ht)]; exact hd t ht⟩, hE]
        simp [ex]
      rw [h.prodN_eq, hone, mul_one]
      classical
      exact if_congr (ho.pins_iffR gr (L.grAdd x.id g gr) (L.ndimAt ks x) E ks ν
        (fun t ht => hv t _ (hax t ht)) (fun t ht => hv t _ (hcx t ht)) hd (h.gbd_g ks)
        (fun q hq t => h.gbd_q ks q hq t)) rfl rfl
    · obtain ⟨t, ht, hlt⟩ : ∃ t ∈ cidx ks (L.base.nodeChain x.id),
          ¬ ν (slab i t) < E.dimv (L.ndimAt ks x t) := by
        by_contra hc
        exact hd fun t ht => by_contra fun hn => hc ⟨t, ht, hn⟩
      have h1 : ¬ (Lg).pinsGR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks ν := fun hp => by
        have := hp.2.2 _ List.mem_cons_self t (by rw [ho.gchain]; exact ht)
        rw [h.gbd_g] at this
        exact hlt this
      rw [if_neg h1]
      split_ifs with hp
      · rw [h.prodN_eq]
        refine mul_eq_zero_of_right _ (List.prod_eq_zero (List.mem_map.2 ⟨t, ht, ?_⟩))
        rw [tZ_id_pair, if_neg]
        rintro ⟨-, hl⟩
        exact hlt (by rw [← hv t _ (hox t ht)]; exact hl)
      · rfl
  · rw [if_neg hA]
    split_ifs with hp
    · rw [h.prodN_eq]
      obtain ⟨l, hl, hne⟩ : ∃ l ∈ L.xdigs ks x, ¬ ν l = ν (slab i (dtup l)) := by
        by_contra hc
        exact hA fun l hl => by_contra fun hn => hc ⟨l, hl, hn⟩
      obtain ⟨t, ht, hlt⟩ := (mem_xdigs _ _ _ _).1 hl
      refine mul_eq_zero_of_right _ (List.prod_eq_zero (List.mem_map.2 ⟨t, ht, ?_⟩))
      rw [tZ_id_pair, if_neg]
      rintro ⟨heq, -⟩
      apply hne
      have hc' : ν (dlab c t) = ν (slab i t) := by
        obtain ⟨hi, e⟩ := List.getElem?_eq_some_iff.1 ho.ci
        have hm : (c, i) ∈ L.strip.base.bnd.zipIdx := (mem_zipIdx' _ _ _).2 ⟨hi, e⟩
        have := hp.2.1 _ hm t (by rw [ho.lc, ho.cnode]; exact ht)
        rwa [labG_none (fun q hq => (ho.gsf q hq).2.1)] at this
      rcases ho.ac with ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> rcases hlt with rfl | rfl <;> rw [dtup_dlab]
      · rw [heq]; exact hc'
      · exact hc'
      · exact hc'
      · rw [← heq]; exact hc'
    · rfl

/-- Dropping `x` for the ghost `(g, i, x.dim)`, `x`'s reading its read bound, keeps `evalGXR`. -/
theorem GhostN.evalGXR_ghost (h : GhostN L gs gr x a c g i) (E : Env)
    (hE : ∀ k, E.phase x.ph k = 0) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalGXR gs gr E ks σ = (Lx).evalGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks σ := by
  have ho := h.ok
  have hX := xdigs_nodup L ks x
  have hdimX : ∀ l ∈ L.xdigs ks x, E.dimv (L.dimAt ks (dport l) (dtup l)) =
      E.dimv (L.ndimAt ks x (dtup l)) := by
    intro l hl
    obtain ⟨t, -, rfl | rfl⟩ := (mem_xdigs L ks x l).1 hl <;>
      rw [dport_dlab, dtup_dlab, h.dimAt_x]
  have hdims : ∀ l ∈ (Lx).base.dlabs ks, E.dimv (L.dimAt ks (dport l) (dtup l)) =
      E.dimv ((Lx).dimAt ks (dport l) (dtup l)) := by
    intro l hl
    obtain ⟨p, hp, t, -, rfl⟩ := (mem_dlabs _ ks l).1 hl
    rw [dport_dlab, dtup_dlab, h.dimAt_gd ks p (ho.node_ne p hp)]
  have hdl : L.base.dlabs ks = (Lx).base.dlabs ks ++ L.xdigs ks x := ho.dlabs_eq ks
  unfold evalGXR
  rw [hdl, sumOver_append, sumOver_dim_congr _ _ _ hdims]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  rw [sumOver_congr _ _ _ (fun ν => if ∀ l ∈ L.xdigs ks x, ν l = τ (slab i (dtup l)) then
      (Lx).integGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks τ else 0) _ fun ν hν hνr => by
    rw [h.integ_point E hE ks ν]
    have he : ∀ l, ν (slab i (dtup l)) = τ (slab i (dtup l)) :=
      fun l => hν _ (slab_not_xdigs L ks x _ _)
    simp only [he]
    exact if_congr Iff.rfl (h.integ_reads E ks ν τ hν) rfl]
  by_cases hd : ∀ t ∈ cidx ks (L.base.nodeChain x.id), τ (slab i t) < E.dimv (L.ndimAt ks x t)
  · rw [sumOver_pinV _ hX _ fun l hl => by
      rw [hdimX l hl]
      obtain ⟨t, ht, e⟩ := (mem_xdigs L ks x l).1 hl
      have ht' : dtup l = t := by rcases e with rfl | rfl <;> exact dtup_dlab _ _
      rw [ht']
      exact hd t ht]
  · have h0 : (Lx).integGXR ((g, i, x.dim) :: gs) (L.grAdd x.id g gr) E ks τ = 0 := by
      rw [integGXR_strip _ h.alX, show (Lx).strip.pinsGR = (Lg).pinsGR from pinsGR_congr rfl rfl,
        if_neg fun hp => hd fun t ht => by
        have := hp.2.2 _ List.mem_cons_self t (by rw [ho.gchain]; exact ht)
        rwa [h.gbd_g] at this]
    rw [h0]
    simp only [ite_self]
    exact sumOver_zero _ _ _

end ghostN

/-- `x` moved last, then dropped for the ghost `g`, its readings dropped. -/
def ghostAtN (L : LDiagram) (x : DNode) (a c g : Port) : LDiagram :=
  (L.moveLast x.id).ghostDropN x.id a c g

/-- Dropping `x` for a ghost after moving it last keeps `evalGXR`. -/
theorem evalGXR_ghostAt {L : LDiagram} {gs : Ghosts} {gr : GReads} {x : DNode} {a c g : Port}
    {i : ℕ} (h : GhostN (L.moveLast x.id) gs gr x a c g i) (E : Env)
    (hE : ∀ k, E.phase x.ph k = 0) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalGXR gs gr E ks σ =
      (L.ghostAtN x a c g).evalGXR ((g, i, x.dim) :: gs) ((L.moveLast x.id).grAdd x.id g gr) E ks
        σ := by
  have hp := moveLast_perm L x.id
  have hid : (L.base.nodes.map (·.id)).Nodup := (hp.map _).nodup_iff.1 h.ok.ids
  rw [← evalGXR_withNodes L _ hp hid]
  exact h.evalGXR_ghost E hE ks σ

/-! ## Identity drops -/

/-- `dropIdX` without the node reading of `x` and the leg readings of its legs. -/
def dropIdN (L : LDiagram) (x : ℕ) (w : Port × Port) (v P : Port) : LDiagram :=
  { ({ L.dropIdX x w v P with cdim := L.cdim.filter (fun q => q.1.node ≠ x) } : LDiagram) with
    cnd := L.cnd.filter (fun q => q.1 ≠ x) }

/-- `DropOK` of `L` without its readings, no aligned wires, no copy-indexed phase of `x`, `x`'s
legs read as `x`, and `P` read as `x` at `P`'s depth as at `x`'s. -/
structure DropN (L : LDiagram) (x : DNode) (w : Port × Port) (u v P : Port) : Prop where
  ok : DropOK L.strip x w u v P
  al : L.aligned = []
  xph : ∀ q ∈ L.cph, q.1 ≠ x.id
  xo : (L.cdim.find? fun q => decide (q.1 = ⟨x.id, true, 0⟩)).map (·.2) =
    (L.cnd.find? fun q => decide (q.1 = x.id)).map (·.2)
  xi : (L.cdim.find? fun q => decide (q.1 = ⟨x.id, false, 0⟩)).map (·.2) =
    (L.cnd.find? fun q => decide (q.1 = x.id)).map (·.2)
  Pr : (L.cdim.find? fun q => decide (q.1 = P)).map (·.2) =
    (L.cnd.find? fun q => decide (q.1 = x.id)).map (·.2)
  Pd : ((L.cnd.find? fun q => decide (q.1 = x.id)).all fun q =>
    decide (L.depth q.2.1 P.node = L.depth q.2.1 x.id)) = true

section dropN

variable {L : LDiagram} {x : DNode} {w : Port × Port} {u v P : Port}

local notation "Lx" => L.dropIdN x.id w v P

local notation "Ls" => L.strip.dropId x.id w v P

theorem depth_dN (y b z : ℕ) : (L.dropIdN y w v P).depth b z = L.depth b z :=
  depth_drop (L := L) y b z

theorem rdVal_dN (y : ℕ) (ks : ℕ → ℕ) (z k : ℕ) (t : List ℕ) (o : Option (ℕ × ℕ × ℕ)) :
    (L.dropIdN y w v P).rdVal ks z k t o = L.rdVal ks z k t o := by
  cases o with
  | none => rfl
  | some q => simp only [rdVal, depth_dN]

theorem phAt_dN (y : ℕ) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (L.dropIdN y w v P).phAt ks n t = L.phAt ks n t := by
  rw [phAt_rdVal, phAt_rdVal, rdVal_dN]
  rfl

theorem ndimAt_dN (y : ℕ) (ks : ℕ → ℕ) (n : DNode) (hn : n.id ≠ y) (t : List ℕ) :
    (L.dropIdN y w v P).ndimAt ks n t = L.ndimAt ks n t := by
  unfold ndimAt
  rw [rdVal_dN]
  show L.rdVal ks n.id n.dim t
      (((L.cnd.filter fun q => q.1 ≠ y).find? fun q => decide (q.1 = n.id)).map (·.2)) = _
  rw [find?_filter_ne L.cnd y n.id hn]

theorem DropN.alX (h : DropN L x w u v P) : (Lx).aligned = [] := h.al

theorem DropN.dimAt_x (h : DropN L x w u v P) (ks : ℕ → ℕ) (b : Bool) (t : List ℕ) :
    L.dimAt ks ⟨x.id, b, 0⟩ t = L.ndimAt ks x t := by
  rw [dimAt_rdVal, show L.base.portDim ⟨x.id, b, 0⟩ = x.dim from h.ok.portDim_x b]
  unfold ndimAt
  cases b
  · rw [h.xi]
  · rw [h.xo]

theorem DropN.dimAt_P (h : DropN L x w u v P) (ks : ℕ → ℕ) (t : List ℕ) :
    L.dimAt ks P t = L.ndimAt ks x t := by
  rw [dimAt_rdVal, show L.base.portDim P = x.dim from h.ok.pdim, h.Pr]
  unfold ndimAt
  have pd := h.Pd
  cases e : L.cnd.find? (fun q => decide (q.1 = x.id)) with
  | none => rfl
  | some q =>
    rw [e] at pd
    simp only [Option.all_some, decide_eq_true_eq] at pd
    simp [rdVal, pd]

theorem DropN.dimAt_dX (h : DropN L x w u v P) (ks : ℕ → ℕ) (p : Port)
    (hp : p.node ≠ x.id) (t : List ℕ) : (Lx).dimAt ks p t = L.dimAt ks p t := by
  have e : (Lx).base.portDim p = L.base.portDim p := h.ok.portDim_drop p hp
  rw [dimAt_rdVal, dimAt_rdVal, rdVal_dN, e]
  show L.rdVal ks p.node (L.base.portDim p) t
      (((L.cdim.filter fun q => q.1.node ≠ x.id).find? fun q => decide (q.1 = p)).map (·.2)) = _
  rw [find?_filter_neP L.cdim x.id p hp]

theorem DropN.prodN_eq (h : DropN L x w u v P) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.prodN E ks ν = (Lx).prodN E ks ν * ((cidx ks (L.base.nodeChain x.id)).map fun t =>
      tZ (E.dimv (L.ndimAt ks x t)) (E.phase x.ph)
        [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod := by
  have ho := h.ok
  have hlast : L.base.nodes = (L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x] := ho.last
  have hx : ∀ t, L.phAt ks x t = x.ph := fun t => by
    have hn : L.cph.find? (fun q => decide (q.1 = x.id)) = none :=
      List.find?_eq_none.2 fun q hq e => h.xph q hq (of_decide_eq_true e)
    rw [phAt_rdVal, hn]
    rfl
  unfold prodN
  conv_lhs => rw [hlast]
  rw [List.map_append, List.prod_append, List.map_singleton, List.prod_singleton]
  congr 1
  · refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    have hn' : n.id ≠ x.id := by
      have hn'' : n ∈ L.base.nodes.filter fun n => n.id ≠ x.id := hn
      rw [List.mem_filter, decide_eq_true_eq] at hn''
      exact hn''.2
    rw [show (Lx).base.nodeChain n.id = L.base.nodeChain n.id from
      nodeChain_drop (L := L) (w := w) (v := v) (P := P) x.id n.id]
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeN_nt ho.ports ho.wch, (Lx).dnodeN_nt ho.ports ho.wch', phAt_dN,
      ndimAt_dN x.id ks n hn']
  · refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeN_nt ho.ports ho.wch, hx]
    simp [val, ho.kind, ho.outs, ho.ins]

/-- The integrand of `evalN`. -/
noncomputable def integN (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  open Classical in if L.pinsX ks ν then L.prodN E ks ν else 0

theorem integN_strip (L : LDiagram) (hal : L.aligned = []) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.integN E ks ν = (open Classical in if L.strip.pinsL ks ν then L.prodN E ks ν else 0) := by
  have h : L.pinsX ks ν ↔ L.strip.pinsL ks ν := by
    simp only [pinsX, dpairsX, hal, List.flatMap_nil, List.append_nil]
    exact Iff.rfl
  unfold integN
  classical
  exact if_congr h rfl rfl

theorem evalN_integN (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalN E ks σ = sumOver (fun l => E.dimv (L.dimAt ks (dport l) (dtup l))) (L.base.dlabs ks)
      (L.integN E ks) (slotsD σ) := by
  rw [evalN_evalW]
  rfl

theorem DropN.integ_reads (h : DropN L x w u v P) (E : Env) (ks : ℕ → ℕ) (ν ν' : ℕ → ℕ)
    (hag : ∀ l ∉ L.xdigs ks x, ν l = ν' l) : (Lx).integN E ks ν = (Lx).integN E ks ν' := by
  have ho := h.ok
  have hd : ∀ p t, p ≠ u ∧ p ≠ v → ν (dlab p t) = ν' (dlab p t) :=
    fun p t hp => hag _ (ho.not_xdigs ks p hp.1 hp.2 t)
  have hs : ∀ j t, ν (slab j t) = ν' (slab j t) := fun j t => hag _ (slab_not_xdigs L ks x j t)
  have hpins : (Ls).pinsL ks ν ↔ (Ls).pinsL ks ν' := by
    unfold pinsL
    refine and_congr (forall₂_congr fun y hy => ?_) (forall₂_congr fun q hq => forall₂_congr
      fun t _ => ?_)
    · obtain ⟨a, b⟩ := ho.pairs_avoid ks y hy
      rw [hd _ _ a, hd _ _ b]
    · have hq' : q.1 ∈ (Ls).base.bnd := (List.mem_zipIdx hq).2.2 ▸ List.getElem_mem _
      rw [ho.bnd_drop, List.mem_map] at hq'
      obtain ⟨q0, hq0, e⟩ := hq'
      rw [hd _ _ (e ▸ ho.subst_ne q0 (fun e' => ho.ub (e' ▸ hq0))), hs]
  have hprod : (Lx).prodN E ks ν = (Lx).prodN E ks ν' := by
    unfold prodN
    refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [(Lx).dnodeN_nt ho.ports ho.wch']
    have hn' : n.id ≠ x.id := by
      have hn'' : n ∈ L.base.nodes.filter fun n => n.id ≠ x.id := hn
      rw [List.mem_filter, decide_eq_true_eq] at hn''
      exact hn''.2
    have hne : ∀ b i, (⟨n.id, b, i⟩ : Port) ≠ u ∧ (⟨n.id, b, i⟩ : Port) ≠ v := fun b i =>
      ⟨fun e => hn' (by rw [← ho.unode, ← e]), fun e => hn' (by rw [← ho.vnode, ← e])⟩
    apply val_congr
    · simp only [List.map_map]
      exact List.map_congr_left fun i _ => hd _ _ (hne _ _)
    · simp only [List.map_map]
      exact List.map_congr_left fun i _ => hd _ _ (hne _ _)
  have hLs : (Lx).strip.pinsL = (Ls).pinsL := pinsL_congr rfl rfl
  rw [integN_strip _ h.alX, integN_strip _ h.alX, hLs, hprod]
  classical
  exact if_congr hpins rfl rfl

theorem DropN.integ_point (h : DropN L x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) (τ ν : ℕ → ℕ)
    (hτ : ∀ y ∉ (Lx).base.dlabs ks, τ y = slotsD σ y)
    (hν : ∀ y ∉ L.xdigs ks x, ν y = τ y)
    (hνr : ∀ y ∈ L.xdigs ks x, ν y < E.dimv (L.ndimAt ks x (dtup y))) :
    L.integN E ks ν =
      if ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l)) then (Lx).integN E ks ν else 0 := by
  have ho := h.ok
  have hPx : ∀ t, dlab P t ∉ L.xdigs ks x := fun t => ho.not_xdigs ks P ho.Pu ho.Pv t
  have hPt : ∀ t, ν (dlab P t) = τ (dlab P t) := fun t => hν _ (hPx t)
  have hux : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab u t ∈ L.xdigs ks x := by
    intro t ht
    rw [mem_xdigs]
    rcases ho.uv with ⟨rfl, -⟩ | ⟨rfl, -⟩
    · exact ⟨t, ht, Or.inl rfl⟩
    · exact ⟨t, ht, Or.inr rfl⟩
  have hvx : ∀ t ∈ cidx ks (L.base.nodeChain x.id), dlab v t ∈ L.xdigs ks x := by
    intro t ht
    rw [mem_xdigs]
    rcases ho.uv with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact ⟨t, ht, Or.inr rfl⟩
    · exact ⟨t, ht, Or.inl rfl⟩
  have hvx' : ∀ t, t ∉ cidx ks (L.base.nodeChain x.id) → dlab v t ∉ L.xdigs ks x := by
    intro t ht hm
    rw [mem_xdigs] at hm
    obtain ⟨t', ht', e | e⟩ := hm <;> exact ht ((dlab_inj e).2 ▸ ht')
  have hwP : L.strip.pinsL ks ν → ∀ t ∈ cidx ks (L.base.nodeChain x.id),
      ν (dlab u t) = ν (dlab P t) := by
    intro hp t ht
    have := hp.1 _ ((ho.mem_dpairs ks _).2 (Or.inl ⟨t, ho.lw1 ▸ ht, rfl⟩))
    rcases ho.wuP with e | e <;> rw [e] at this
    · exact this
    · exact this.symm
  have hLs : (Lx).strip.pinsL = (Ls).pinsL := pinsL_congr rfl rfl
  rw [integN_strip _ h.al, integN_strip _ h.alX, hLs]
  by_cases hA : ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l))
  · rw [if_pos hA]
    have hsub : ∀ a t, ν (dlab (substP v P a) t) = ν (dlab a t) := by
      intro a t
      unfold substP
      split_ifs with ha
      · rw [ha]
        by_cases ht : t ∈ cidx ks (L.base.nodeChain x.id)
        · rw [hA _ (hvx t ht), dtup_dlab, hPt]
        · have h1 : dlab v t ∉ (Ls).base.dlabs ks := by
            rw [mem_dlabs]
            rintro ⟨p, hp, t', -, e⟩
            obtain ⟨rfl, -⟩ := dlab_inj e
            exact ho.node_ne _ hp ho.vnode
          have h2 : dlab P t ∉ (Ls).base.dlabs ks := by
            rw [mem_dlabs]
            rintro ⟨p, -, t', ht', e⟩
            obtain ⟨rfl, rfl⟩ := dlab_inj e
            rw [ho.lc', ho.chP] at ht'
            exact ht ht'
          rw [hPt, hτ _ h2, hν _ (hvx' t ht), hτ _ h1, slotsD_dlab, slotsD_dlab]
      · rfl
    have huP : ∀ t ∈ cidx ks (L.base.nodeChain x.id), ν (dlab u t) = ν (dlab P t) :=
      fun t ht => by rw [hA _ (hux t ht), dtup_dlab, hPt]
    have hone : ((cidx ks (L.base.nodeChain x.id)).map fun t =>
        tZ (E.dimv (L.ndimAt ks x t)) (E.phase x.ph)
          [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod = 1 := by
      refine List.prod_eq_one fun z hz => ?_
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hz
      have hoo : dlab ⟨x.id, true, 0⟩ t ∈ L.xdigs ks x :=
        (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inl rfl⟩
      have hi : dlab ⟨x.id, false, 0⟩ t ∈ L.xdigs ks x :=
        (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inr rfl⟩
      have hr := hνr _ hoo
      rw [dtup_dlab] at hr
      rw [tZ_id_pair, if_pos ⟨by rw [hA _ hoo, hA _ hi, dtup_dlab, dtup_dlab], hr⟩, hE]
      simp [ex]
    rw [h.prodN_eq, hone, mul_one]
    classical
    exact if_congr (ho.pins_iff ks ν hsub huP) rfl rfl
  · rw [if_neg hA]
    split_ifs with hp
    · rw [h.prodN_eq]
      refine mul_eq_zero_of_right _ (List.prod_eq_zero ?_)
      have hu := hwP hp
      obtain ⟨l, hl, hne⟩ : ∃ l ∈ L.xdigs ks x, ¬ ν l = τ (dlab P (dtup l)) := by
        by_contra hc
        exact hA fun l hl => by_contra fun hn => hc ⟨l, hl, hn⟩
      obtain ⟨t, ht, hlt⟩ := (mem_xdigs _ _ _ _).1 hl
      refine List.mem_map.2 ⟨t, ht, ?_⟩
      rw [tZ_id_pair, if_neg]
      rintro ⟨heq, -⟩
      apply hne
      have hu' := hu t ht
      rw [hPt] at hu'
      rcases ho.uv with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases hlt with rfl | rfl <;> rw [dtup_dlab]
      · exact hu'
      · rw [← heq]; exact hu'
      · rw [heq]; exact hu'
      · exact hu'
    · rfl

/-- Dropping `x` keeps `evalN`. -/
theorem DropN.evalN_drop (h : DropN L x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) : L.evalN E ks σ = (Lx).evalN E ks σ := by
  have ho := h.ok
  have hX := xdigs_nodup L ks x
  have hdimX : ∀ l ∈ L.xdigs ks x, E.dimv (L.dimAt ks (dport l) (dtup l)) =
      E.dimv (L.ndimAt ks x (dtup l)) := by
    intro l hl
    obtain ⟨t, -, rfl | rfl⟩ := (mem_xdigs L ks x l).1 hl <;>
      rw [dport_dlab, dtup_dlab, h.dimAt_x]
  have hdims : ∀ l ∈ (Lx).base.dlabs ks, E.dimv (L.dimAt ks (dport l) (dtup l)) =
      E.dimv ((Lx).dimAt ks (dport l) (dtup l)) := by
    intro l hl
    obtain ⟨p, hp, t, -, rfl⟩ := (mem_dlabs _ ks l).1 hl
    rw [dport_dlab, dtup_dlab, h.dimAt_dX ks p (ho.node_ne p hp)]
  have hdl : L.base.dlabs ks = (Lx).base.dlabs ks ++ L.xdigs ks x := ho.dlabs_eq ks
  rw [evalN_integN, evalN_integN]
  rw [hdl, sumOver_append, sumOver_dim_congr _ _ _ hdims]
  refine sumOver_congr _ _ _ _ _ fun τ hτ hr => ?_
  have hvf : ∀ l ∈ L.xdigs ks x, τ (dlab P (dtup l)) < E.dimv (L.dimAt ks (dport l) (dtup l)) := by
    intro l hl
    rw [hdimX l hl]
    obtain ⟨t, ht, e⟩ := (mem_xdigs L ks x l).1 hl
    have ht' : dtup l = t := by rcases e with rfl | rfl <;> exact dtup_dlab _ _
    rw [ht']
    have hm : dlab P t ∈ (Lx).base.dlabs ks := by
      have : dlab P t ∈ (Ls).base.dlabs ks :=
        (mem_dlabs _ ks _).2 ⟨P, ho.Pmem, t, by rw [ho.lc', ho.chP]; exact ht, rfl⟩
      exact this
    have := hr _ hm
    rwa [dport_dlab, dtup_dlab, h.dimAt_dX ks P ho.hP, h.dimAt_P] at this
  rw [sumOver_congr _ _ _ (fun ν => if ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l)) then
      (Lx).integN E ks ν else 0) _ fun ν hν hνr => h.integ_point E hE ks σ τ ν hτ hν
      fun y hy => by rw [← hdimX y hy]; exact hνr y hy]
  rw [sumOver_pinV _ hX _ hvf]
  exact h.integ_reads E ks _ _ fun l hl => if_neg hl

end dropN

/-- `x` moved last, then dropped, its readings dropped. -/
def dropAtN (L : LDiagram) (x : DNode) (w : Port × Port) (v P : Port) : LDiagram :=
  (L.moveLast x.id).dropIdN x.id w v P

/-- Dropping `x` after moving it last keeps `evalN`. -/
theorem evalN_dropAt {L : LDiagram} {x : DNode} {w : Port × Port} {u v P : Port}
    (h : DropN (L.moveLast x.id) x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalN E ks σ = (L.dropAtN x w v P).evalN E ks σ := by
  have hp := moveLast_perm L x.id
  have hid : (L.base.nodes.map (·.id)).Nodup := (hp.map _).nodup_iff.1 h.ok.ids
  unfold evalN
  rw [← evalGXR_withNodes L _ hp hid]
  exact h.evalN_drop E hE ks σ

end LDiagram

end ArchytasZX
