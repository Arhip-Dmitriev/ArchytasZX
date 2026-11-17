/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.LinkedGhost

/-!
# Identity drops over `evalX`

`DropOKX`: `DropOK` of `L` without its readings, no aligned wires, no reading of the dropped
Z(1 -> 1) `x` or its legs, and no leg-dimension reading of the leg `P` it meets. `dropIdX` is
`dropId` keeping the copy-indexed leg dimensions and the aligned wires; `DropOKX.evalX_drop`: the
drop keeps the digit sum `evalX` when `x`'s phase index reads zero (`evalL_drop` with `phAt` and
`dimAt`), and `evalX_dropAt` does it after moving `x` last.
-/

namespace ArchytasZX

namespace LDiagram

/-- `dropId` keeping the copy-indexed leg dimensions and the aligned wires. -/
def dropIdX (L : LDiagram) (x : ℕ) (w : Port × Port) (v P : Port) : LDiagram :=
  { L.dropId x w v P with cdim := L.cdim, aligned := L.aligned }

theorem strip_dX (L : LDiagram) (x : ℕ) (w : Port × Port) (v P : Port) :
    (L.dropIdX x w v P).strip = L.strip.dropId x w v P := rfl

/-- The integrand of `L`'s digit sum `evalX`. -/
noncomputable def integX (L : LDiagram) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) : ℂ :=
  open Classical in if L.pinsX ks ν then L.prodL E ks ν else 0

theorem integX_strip (L : LDiagram) (hal : L.aligned = []) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.integX E ks ν = (open Classical in if L.strip.pinsL ks ν then L.prodL E ks ν else 0) := by
  have h : L.pinsX ks ν ↔ L.strip.pinsL ks ν := by
    simp only [pinsX, dpairsX, hal, List.flatMap_nil, List.append_nil]
    exact Iff.rfl
  unfold integX
  classical
  exact if_congr h rfl rfl

/-- `DropOK` of `L` without its readings, no aligned wires, no reading of `x` or its legs, and no
leg-dimension reading of `P`. -/
structure DropOKX (L : LDiagram) (x : DNode) (w : Port × Port) (u v P : Port) : Prop where
  ok : DropOK L.strip x w u v P
  al : L.aligned = []
  xph : ∀ q ∈ L.cph, q.1 ≠ x.id
  xdm : ∀ q ∈ L.cdim, q.1.node ≠ x.id
  Pdm : ∀ q ∈ L.cdim, q.1 ≠ P

section dropX

variable {L : LDiagram} {x : DNode} {w : Port × Port} {u v P : Port}

local notation "Lx" => L.dropIdX x.id w v P

local notation "Ls" => L.strip.dropId x.id w v P

theorem depth_dX (y b z : ℕ) : (L.dropIdX y w v P).depth b z = L.depth b z :=
  depth_drop (L := L) y b z

theorem rdVal_dX (y : ℕ) (ks : ℕ → ℕ) (z k : ℕ) (t : List ℕ) (o : Option (ℕ × ℕ × ℕ)) :
    (L.dropIdX y w v P).rdVal ks z k t o = L.rdVal ks z k t o := by
  cases o with
  | none => rfl
  | some q => simp only [rdVal, depth_dX]

theorem phAt_dX (y : ℕ) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) :
    (L.dropIdX y w v P).phAt ks n t = L.phAt ks n t := by
  rw [phAt_rdVal, phAt_rdVal, rdVal_dX]
  rfl

theorem DropOKX.alX (h : DropOKX L x w u v P) : (Lx).aligned = [] := h.al

theorem DropOKX.dimAt_x (h : DropOKX L x w u v P) (ks : ℕ → ℕ) (b : Bool) (t : List ℕ) :
    L.dimAt ks ⟨x.id, b, 0⟩ t = x.dim := by
  have hn : L.cdim.find? (fun q => decide (q.1 = ⟨x.id, b, 0⟩)) = none :=
    List.find?_eq_none.2 fun q hq e => h.xdm q hq (by rw [of_decide_eq_true e])
  rw [dimAt_rdVal, hn]
  exact h.ok.portDim_x b

theorem DropOKX.dimAt_P (h : DropOKX L x w u v P) (ks : ℕ → ℕ) (t : List ℕ) :
    L.dimAt ks P t = x.dim := by
  have hn : L.cdim.find? (fun q => decide (q.1 = P)) = none :=
    List.find?_eq_none.2 fun q hq e => h.Pdm q hq (of_decide_eq_true e)
  rw [dimAt_rdVal, hn]
  exact h.ok.pdim

theorem DropOKX.dimAt_dX (h : DropOKX L x w u v P) (ks : ℕ → ℕ) (p : Port)
    (hp : p.node ≠ x.id) (t : List ℕ) : (Lx).dimAt ks p t = L.dimAt ks p t := by
  have e : (Lx).base.portDim p = L.base.portDim p := h.ok.portDim_drop p hp
  rw [dimAt_rdVal, dimAt_rdVal, rdVal_dX, e]
  rfl

theorem DropOKX.prodL_eq (h : DropOKX L x w u v P) (E : Env) (ks : ℕ → ℕ) (ν : ℕ → ℕ) :
    L.prodL E ks ν = (Lx).prodL E ks ν * ((cidx ks (L.base.nodeChain x.id)).map fun t =>
      tZ (E.dimv x.dim) (E.phase x.ph)
        [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod := by
  have ho := h.ok
  have hlast : L.base.nodes = (L.base.nodes.filter fun n => n.id ≠ x.id) ++ [x] := ho.last
  have hx : ∀ t, L.phAt ks x t = x.ph := fun t => by
    have hn : L.cph.find? (fun q => decide (q.1 = x.id)) = none :=
      List.find?_eq_none.2 fun q hq e => h.xph q hq (of_decide_eq_true e)
    rw [phAt_rdVal, hn]
    rfl
  unfold prodL
  conv_lhs => rw [hlast]
  rw [List.map_append, List.prod_append, List.map_singleton, List.prod_singleton]
  congr 1
  · refine congrArg List.prod (List.map_congr_left fun n _ => ?_)
    rw [show (Lx).base.nodeChain n.id = L.base.nodeChain n.id from
      nodeChain_drop (L := L) (w := w) (v := v) (P := P) x.id n.id]
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeL_nt ho.ports ho.wch, (Lx).dnodeL_nt ho.ports ho.wch', phAt_dX]
  · refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [L.dnodeL_nt ho.ports ho.wch, hx]
    simp [val, ho.kind, ho.outs, ho.ins]

theorem DropOKX.integ_reads (h : DropOKX L x w u v P) (E : Env) (ks : ℕ → ℕ) (ν ν' : ℕ → ℕ)
    (hag : ∀ l ∉ L.xdigs ks x, ν l = ν' l) : (Lx).integX E ks ν = (Lx).integX E ks ν' := by
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
  have hprod : (Lx).prodL E ks ν = (Lx).prodL E ks ν' := by
    unfold prodL
    refine congrArg List.prod (List.map_congr_left fun n hn => ?_)
    refine congrArg List.prod (List.map_congr_left fun t _ => ?_)
    rw [(Lx).dnodeL_nt ho.ports ho.wch']
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
  rw [integX_strip _ h.alX, integX_strip _ h.alX, strip_dX, hprod]
  classical
  exact if_congr hpins rfl rfl

theorem DropOKX.integ_point (h : DropOKX L x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) (τ ν : ℕ → ℕ)
    (hτ : ∀ y ∉ (Lx).base.dlabs ks, τ y = slotsD σ y)
    (hν : ∀ y ∉ L.xdigs ks x, ν y = τ y) (hνr : ∀ y ∈ L.xdigs ks x, ν y < E.dimv x.dim) :
    L.integX E ks ν =
      if ∀ l ∈ L.xdigs ks x, ν l = τ (dlab P (dtup l)) then (Lx).integX E ks ν else 0 := by
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
  rw [integX_strip _ h.al, integX_strip _ h.alX, strip_dX]
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
        tZ (E.dimv x.dim) (E.phase x.ph)
          [ν (dlab ⟨x.id, true, 0⟩ t), ν (dlab ⟨x.id, false, 0⟩ t)]).prod = 1 := by
      refine List.prod_eq_one fun z hz => ?_
      obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hz
      have hoo : dlab ⟨x.id, true, 0⟩ t ∈ L.xdigs ks x :=
        (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inl rfl⟩
      have hi : dlab ⟨x.id, false, 0⟩ t ∈ L.xdigs ks x :=
        (mem_xdigs _ _ _ _).2 ⟨t, ht, Or.inr rfl⟩
      rw [tZ_id_pair, if_pos ⟨by rw [hA _ hoo, hA _ hi, dtup_dlab, dtup_dlab], hνr _ hoo⟩, hE]
      simp [ex]
    rw [h.prodL_eq, hone, mul_one]
    classical
    exact if_congr (ho.pins_iff ks ν hsub huP) rfl rfl
  · rw [if_neg hA]
    split_ifs with hp
    · rw [h.prodL_eq]
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

/-- Dropping `x` keeps the digit sum `evalX`. -/
theorem DropOKX.evalX_drop (h : DropOKX L x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) : L.evalX E ks σ = (Lx).evalX E ks σ := by
  have ho := h.ok
  have hX := xdigs_nodup L ks x
  have hdimX : ∀ l ∈ L.xdigs ks x, E.dimv (L.dimAt ks (dport l) (dtup l)) = E.dimv x.dim := by
    intro l hl
    obtain ⟨t, -, rfl | rfl⟩ := (mem_xdigs L ks x l).1 hl <;>
      rw [dport_dlab, dtup_dlab, h.dimAt_x]
  have hdims : ∀ l ∈ (Lx).base.dlabs ks, E.dimv (L.dimAt ks (dport l) (dtup l)) =
      E.dimv ((Lx).dimAt ks (dport l) (dtup l)) := by
    intro l hl
    obtain ⟨p, hp, t, -, rfl⟩ := (mem_dlabs _ ks l).1 hl
    rw [dport_dlab, dtup_dlab, h.dimAt_dX ks p (ho.node_ne p hp)]
  have hdl : L.base.dlabs ks = (Lx).base.dlabs ks ++ L.xdigs ks x := ho.dlabs_eq ks
  change sumOver _ (L.base.dlabs ks) (L.integX E ks) (slotsD σ) =
    sumOver _ ((Lx).base.dlabs ks) ((Lx).integX E ks) (slotsD σ)
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
      (Lx).integX E ks ν else 0) _ fun ν hν hνr => h.integ_point E hE ks σ τ ν hτ hν
      fun y hy => by rw [← hdimX y hy]; exact hνr y hy]
  rw [sumOver_pinV _ hX _ hvf]
  exact h.integ_reads E ks _ _ fun l hl => if_neg hl

end dropX

/-- `x` moved last, then dropped, the readings kept. -/
def dropAtX (L : LDiagram) (x : DNode) (w : Port × Port) (v P : Port) : LDiagram :=
  (L.moveLast x.id).dropIdX x.id w v P

/-- Dropping `x` after moving it last keeps the digit sum `evalX`. -/
theorem evalX_dropAt {L : LDiagram} {x : DNode} {w : Port × Port} {u v P : Port}
    (h : DropOKX (L.moveLast x.id) x w u v P) (E : Env) (hE : ∀ k, E.phase x.ph k = 0)
    (ks : ℕ → ℕ) (σ : ℕ → List ℕ → ℕ) :
    L.evalX E ks σ = (L.dropAtX x w v P).evalX E ks σ := by
  have hp := moveLast_perm L x.id
  have hid : (L.base.nodes.map (·.id)).Nodup := (hp.map _).nodup_iff.1 h.ok.ids
  rw [← evalGX_nil L, ← evalGX_withNodes L _ hp hid, evalGX_nil]
  exact h.evalX_drop E hE ks σ

end LDiagram

end ArchytasZX
