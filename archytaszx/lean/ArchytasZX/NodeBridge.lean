/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.NodeVal

/-!
# A bundled node as its copies

`val_bnode`: the value of the bundled node `D.bnode off n` at a bundled assignment `β` is the
product over its copies `t` of the copy `D.dnode n t` at the digit assignment `ν`, which reads at
each digit label the digit of the leg's bundled value.
-/

namespace ArchytasZX

namespace Diagram

/-- The conditions on a node for `val_bnode`. -/
structure NodeOK (D : Diagram) (E : Env) (off : ℕ) (n : DNode) (β ν : ℕ → ℕ) : Prop where
  plain : n.kind.plain = true
  boxkind : D.nodeChain n.id ≠ [] → n.kind ≠ .B ∧ n.kind ≠ .S
  tailkind : ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = []) →
    n.kind = .Z ∨ n.kind = .X ∨ (n.kind = .W ∧ tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])
  dims : (D.nodeChain n.id ≠ [] ∨ ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])) → ∀ p ∈ n.legs, D.portDim p = n.dim
  posd : ∀ p ∈ n.legs, 0 < E.dimv (D.portDim p)
  posn : 0 < E.dimv n.dim
  range : ∀ p ∈ n.legs, β (D.label off p) < E.dimv (D.portDim p) ^ size E.mult (D.legChain p)
  dig : ∀ p ∈ n.legs, ∀ u ∈ cidx E.mult (D.legChain p),
    ν (dlab p u) = digit (E.dimv (D.portDim p)) (pos E.mult (D.legChain p) u) (β (D.label off p))

theorem leg_side {n : DNode} (o : Bool) (i : ℕ) (hi : i < (if o then n.outs else n.ins).length) :
    (⟨n.id, o, i⟩ : Port) ∈ n.legs := by
  rw [DNode.mem_legs]; cases o <;> simpa using hi

theorem pairs_mem_leg (D : Diagram) (off : ℕ) (n : DNode) (o : Bool) {q : ℕ × List ℕ}
    (hq : q ∈ pairs (if o then n.outs else n.ins).length (fun j => D.label off ⟨n.id, o, j⟩)
      (fun j => D.tail ⟨n.id, o, j⟩)) :
    ∃ p ∈ n.legs, q = (D.label off p, D.tail p) := by
  simp only [pairs, List.mem_map, List.mem_range] at hq
  obtain ⟨j, hj, rfl⟩ := hq
  exact ⟨_, leg_side o j hj, rfl⟩

theorem dnode_eq (D : Diagram) (ks : ℕ → ℕ) (n : DNode) (t : List ℕ) : D.dnode ks n t =
    ⟨n.kind, (List.range n.outs.length).flatMap fun i => (rcidx ks (D.tail ⟨n.id, true, i⟩)).map
      fun s => dlab ⟨n.id, true, i⟩ (t ++ s), (List.range n.ins.length).flatMap fun i =>
      (rcidx ks (D.tail ⟨n.id, false, i⟩)).map fun s => dlab ⟨n.id, false, i⟩ (t ++ s), n.dim, n.ph⟩ :=
  rfl

/-- `ν` reads the digits of a copy's legs, at the node's dimension. -/
theorem dig_side (D : Diagram) (E : Env) (off : ℕ) (n : DNode) (β ν : ℕ → ℕ)
    (h : D.NodeOK E off n β ν) (hd : ∀ p ∈ n.legs, D.portDim p = n.dim) (o : Bool) (t : List ℕ)
    (ht : t ∈ cidx E.mult (D.nodeChain n.id)) :
    ((List.range (if o then n.outs else n.ins).length).flatMap fun i =>
      (rcidx E.mult (D.tail ⟨n.id, o, i⟩)).map fun s => dlab ⟨n.id, o, i⟩ (t ++ s)).map ν =
    ldgR E.mult (D.nodeChain n.id) t (E.dimv n.dim) β
      (pairs (if o then n.outs else n.ins).length (fun j => D.label off ⟨n.id, o, j⟩)
        (fun j => D.tail ⟨n.id, o, j⟩)) := by
  refine dnode_side D E.mult n t o _ _ _ β ν fun i hi s hs => ?_
  have hp := leg_side o i hi
  rw [h.dig _ hp (t ++ s) (mem_cidx_append _ _ _ _ _ ht ((mem_rcidx _ _ _).1 hs)), hd _ hp]
  rfl

end Diagram

end ArchytasZX

namespace ArchytasZX

section gen

variable (E : Env) (C : List ℕ) (m : Node) (nO : ℕ) (lO : ℕ → ℕ) (cO : ℕ → List ℕ) (nI : ℕ)
  (lI : ℕ → ℕ) (cI : ℕ → List ℕ) (hmo : m.outs = (List.range nO).map lO)
  (hmi : m.ins = (List.range nI).map lI) (β : ℕ → ℕ)

theorem keep_mem (s0 : ℕ) {q : ℕ × List ℕ} (hq : q ∈ (keepL nO lO cO).map (·, [])) :
    q ∈ pairs nO lO cO :=
  (side_perm nO lO cO s0).mem_iff.2 (List.mem_append_left _ hq)

theorem sort_mem (s0 : ℕ) {q : ℕ × List ℕ}
    (hq : q ∈ (sortT nO cO s0).map fun t => (((List.range nO).map lO).getD t.2.2 0, t.1)) :
    q ∈ pairs nO lO cO :=
  (side_perm nO lO cO s0).mem_iff.2 (List.mem_append_right _ hq)

include hmo hmi in
theorem val_bundle_Z (hz : m.kind = .Z) (hne : ¬(tls nO cO = [] ∧ tls nI cI = []))
    (hd : 0 < E.dimv m.dim)
    (hr : ∀ q ∈ pairs nO lO cO ++ pairs nI lI cI, β q.1 < E.dimv m.dim ^ size E.mult (C ++ q.2)) :
    val E β { bundleNode m (tls nO cO) (tls nI cI) with
      kind := C.foldr Kind.Bun (bundleNode m (tls nO cO) (tls nI cI)).kind } =
    ((cidx E.mult C).map fun t => tZ (E.dimv m.dim) (E.phase m.ph)
      (ldg E.mult C t (E.dimv m.dim) β (pairs nO lO cO) ++ ldg E.mult C t (E.dimv m.dim) β (pairs nI lI cI))).prod := by
  rw [bundle_Z m nO lO cO nI lI cI hmo hmi hz hne]
  set both := bothZ nO cO nI cI
  set KO := keepL nO lO cO
  set KI := keepL nI lI cI
  show val E β ⟨C.foldr Kind.Bun (.ZB (both.map (·.1))), KO, KI ++ both.map (labZ nO lO nI lI), m.dim, m.ph⟩ = _
  have hlen : (KI ++ both.map (labZ nO lO nI lI)).length - (both.map (·.1)).length = KI.length := by
    simp
  rw [val_tower E _ C (Or.inr ⟨by simp, by simp⟩) _ _ m.dim m.ph β hd
    (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) hr) (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) hr)]
  · congr 1
    refine List.map_congr_left fun t ht => ?_
    simp only [Kind.outCs, Kind.inCs, hlen, tensor]
    refine tZ_perm _ _ _ _ ?_
    exact z_vals E.mult C t (E.dimv m.dim) hd ht β _ rfl nO lO cO nI lI cI
  · simp only [Kind.outCs, zip_replicateC] at hq
    exact List.mem_append_left _ (keep_mem nO lO cO 0 hq)
  · simp only [Kind.inCs, hlen] at hq
    rw [List.zip_append (by simp), zip_replicateC, List.zip_map'] at hq
    rcases List.mem_append.1 hq with hq | hq
    · exact List.mem_append_right _ (keep_mem nI lI cI 1 hq)
    · have := (bothZ_perm nO lO cO nI lI cI).mem_iff.1 (by simpa [both] using hq)
      rcases List.mem_append.1 this with h1 | h1
      · exact List.mem_append_left _ (sort_mem nO lO cO 0 h1)
      · exact List.mem_append_right _ (sort_mem nI lI cI 1 h1)

include hmo hmi in
theorem val_bundle_X (hx : m.kind = .X) (hne : ¬(tls nO cO = [] ∧ tls nI cI = []))
    (hd : 0 < E.dimv m.dim)
    (hr : ∀ q ∈ pairs nO lO cO ++ pairs nI lI cI, β q.1 < E.dimv m.dim ^ size E.mult (C ++ q.2)) :
    val E β { bundleNode m (tls nO cO) (tls nI cI) with
      kind := C.foldr Kind.Bun (bundleNode m (tls nO cO) (tls nI cI)).kind } =
    ((cidx E.mult C).map fun t => tX (E.dimv m.dim) (E.phase m.ph)
      (ldg E.mult C t (E.dimv m.dim) β (pairs nO lO cO)) (ldg E.mult C t (E.dimv m.dim) β (pairs nI lI cI))).prod := by
  rw [bundle_X m nO lO cO nI lI cI hmo hmi hx hne]
  set SO := sortT nO cO 0
  set SI := sortT nI cI 1
  set KO := keepL nO lO cO
  set KI := keepL nI lI cI
  show val E β ⟨C.foldr Kind.Bun (.XB (SO.map (·.1)) (SI.map (·.1))),
    KO ++ SO.map (fun t => ((List.range nO).map lO).getD t.2.2 0),
    KI ++ SI.map (fun t => ((List.range nI).map lI).getD t.2.2 0), m.dim, m.ph⟩ = _
  have hlo : (KO ++ SO.map (fun t => ((List.range nO).map lO).getD t.2.2 0)).length -
      (SO.map (·.1)).length = KO.length := by simp
  have hli : (KI ++ SI.map (fun t => ((List.range nI).map lI).getD t.2.2 0)).length -
      (SI.map (·.1)).length = KI.length := by simp
  rw [val_tower E _ C (Or.inr ⟨by simp, by simp⟩) _ _ m.dim m.ph β hd
    (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) hr) (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) hr)]
  · congr 1
    refine List.map_congr_left fun t ht => ?_
    simp only [Kind.outCs, Kind.inCs, hlo, hli, tensor]
    exact tX_perm _ _ _ _ _ _ (side_vals E.mult C t (E.dimv m.dim) hd ht β _ rfl nO lO cO 0)
      (side_vals E.mult C t (E.dimv m.dim) hd ht β _ rfl nI lI cI 1)
  · simp only [Kind.outCs, hlo] at hq
    rw [side_zip] at hq
    rcases List.mem_append.1 hq with hq | hq
    · exact List.mem_append_left _ (keep_mem nO lO cO 0 hq)
    · exact List.mem_append_left _ (sort_mem nO lO cO 0 hq)
  · simp only [Kind.inCs, hli] at hq
    rw [side_zip] at hq
    rcases List.mem_append.1 hq with hq | hq
    · exact List.mem_append_right _ (keep_mem nI lI cI 1 hq)
    · exact List.mem_append_right _ (sort_mem nI lI cI 1 hq)

theorem tensor_WB (cs : List (List ℕ)) (d ph : ℕ) (os is : List ℕ) :
    tensor E (.WB cs) d ph os is = tensor E .W d ph (bundled E.mult d os cs) is := by
  rcases is with _ | ⟨w, _ | ⟨_, _⟩⟩ <;> rfl

theorem tensor_W_perm (d ph : ℕ) {os os' : List ℕ} (h : os.Perm os') (is : List ℕ) :
    tensor E .W d ph os is = tensor E .W d ph os' is := by
  rcases is with _ | ⟨w, _ | ⟨_, _⟩⟩
  · rfl
  · exact tW_perm _ _ _ _ h
  · rfl

include hmo hmi in
theorem val_bundle_W (hw : m.kind = .W) (hne : ¬(tls nO cO = [] ∧ tls nI cI = []))
    (hI : tls nI cI = []) (hd : 0 < E.dimv m.dim)
    (hr : ∀ q ∈ pairs nO lO cO ++ pairs nI lI cI, β q.1 < E.dimv m.dim ^ size E.mult (C ++ q.2)) :
    val E β { bundleNode m (tls nO cO) (tls nI cI) with
      kind := C.foldr Kind.Bun (bundleNode m (tls nO cO) (tls nI cI)).kind } =
    ((cidx E.mult C).map fun t => tensor E .W (E.dimv m.dim) m.ph
      (ldg E.mult C t (E.dimv m.dim) β (pairs nO lO cO))
      (m.ins.map fun l => digit (E.dimv m.dim) (pos E.mult C t) (β l))).prod := by
  rw [bundle_W m nO lO cO nI lI cI hmo hmi hw hne]
  set SO := sortT nO cO 0
  set KO := keepL nO lO cO
  show val E β ⟨C.foldr Kind.Bun (.WB (SO.map (·.1))),
    KO ++ SO.map (fun t => ((List.range nO).map lO).getD t.2.2 0), m.ins, m.dim, m.ph⟩ = _
  have hlo : (KO ++ SO.map (fun t => ((List.range nO).map lO).getD t.2.2 0)).length -
      (SO.map (·.1)).length = KO.length := by simp
  rw [val_tower E _ C (Or.inr ⟨by simp, by simp⟩) _ _ m.dim m.ph β hd
    (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) hr) (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) hr)]
  · congr 1
    refine List.map_congr_left fun t ht => ?_
    simp only [Kind.outCs, Kind.inCs, hlo, tensor_WB]
    rw [blk_plain]
    exact tensor_W_perm E _ _ (side_vals E.mult C t (E.dimv m.dim) hd ht β _ rfl nO lO cO 0) _
  · simp only [Kind.outCs, hlo] at hq
    rw [side_zip] at hq
    rcases List.mem_append.1 hq with hq | hq
    · exact List.mem_append_left _ (keep_mem nO lO cO 0 hq)
    · exact List.mem_append_left _ (sort_mem nO lO cO 0 hq)
  · simp only [Kind.inCs, zip_replicateC] at hq
    refine List.mem_append_right _ ?_
    rw [pairs_plain _ _ _ hI, ← hmi]
    exact hq

end gen

namespace Diagram

theorem node_congr (n : DNode) (g g' : Port → ℕ) (h : ∀ p ∈ n.legs, g p = g' p) :
    n.node g = n.node g' := by
  simp only [DNode.node, Node.mk.injEq]
  refine ⟨trivial, ?_, ?_, trivial, trivial⟩ <;> refine List.map_congr_left fun j hj => h _ ?_
  · exact leg_side true j (by simpa using hj)
  · exact leg_side false j (by simpa using hj)

theorem NodeOK.range_pairs {D : Diagram} {E : Env} {off : ℕ} {n : DNode} {β ν : ℕ → ℕ}
    (h : D.NodeOK E off n β ν)
    (hdim : D.nodeChain n.id ≠ [] ∨ ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])) (o : Bool) :
    ∀ q ∈ pairs (if o then n.outs else n.ins).length (fun j => D.label off ⟨n.id, o, j⟩)
      (fun j => D.tail ⟨n.id, o, j⟩), β q.1 < E.dimv n.dim ^ size E.mult (D.nodeChain n.id ++ q.2) := by
  intro q hq
  obtain ⟨p, hp, rfl⟩ := pairs_mem_leg D off n o hq
  have := h.range p hp
  rw [h.dims hdim p hp] at this
  have hn := (DNode.mem_legs.1 hp).1
  simpa [legChain, hn] using this

theorem tail_nil_of (D : Diagram) (n : DNode)
    (hT : tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = []) :
    ∀ p ∈ n.legs, D.tail p = [] := by
  intro p hp
  obtain ⟨x, o, i⟩ := p
  rw [DNode.mem_legs] at hp
  obtain ⟨rfl, hi⟩ := hp
  cases o
  · exact (tls_nil _ _).1 hT.2 i (by simpa using hi)
  · exact (tls_nil _ _).1 hT.1 i (by simpa using hi)

/-- A bundled node is the product over its copies of the copy at the digits `ν`. -/
theorem val_bnode (D : Diagram) (E : Env) (off : ℕ) (n : DNode) (β ν : ℕ → ℕ)
    (h : D.NodeOK E off n β ν) :
    val E β (D.bnode off n) =
      ((cidx E.mult (D.nodeChain n.id)).map fun t => val E ν (D.dnode E.mult n t)).prod := by
  by_cases hT : tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = []
  · have hbun : bundleNode (n.node (D.label off)) (tls n.outs.length fun j => D.tail ⟨n.id, true, j⟩)
        (tls n.ins.length fun j => D.tail ⟨n.id, false, j⟩) = n.node (D.label off) := by
      unfold bundleNode; rw [if_pos hT]
    have htl := tail_nil_of D n hT
    rw [bnode_eq, hbun]
    by_cases hC0 : D.nodeChain n.id = []
    · rw [hC0]
      simp only [List.foldr_nil, cidx, List.map_cons, List.map_nil, List.prod_cons, List.prod_nil,
        mul_one]
      have hdn : D.dnode E.mult n [] = n.node fun p => dlab p [] := by
        rw [dnode_eq]
        simp only [DNode.node, Node.mk.injEq]
        refine ⟨trivial, ?_, ?_, trivial, trivial⟩ <;> rw [List.map_eq_flatMap] <;>
          refine List.flatMap_congr fun i hi => ?_
        · rw [htl _ (leg_side true i (by simpa using hi)), rcidx_nil]; rfl
        · rw [htl _ (leg_side false i (by simpa using hi)), rcidx_nil]; rfl
      rw [hdn]
      show val E β (n.node (D.label off)) = _
      rw [val_node E β n, val_node E ν n]
      refine congrArg (val E id) (node_congr n _ _ fun p hp => ?_)
      have hl : D.legChain p = [] := by simp [legChain, (DNode.mem_legs.1 hp).1, hC0, htl p hp]
      have hr := h.range p hp
      rw [hl] at hr
      simp only [Function.comp]
      rw [h.dig p hp [] (by rw [hl]; simp [cidx]), hl]
      simp only [pos, digit, pow_zero, Nat.div_one]
      exact (Nat.mod_eq_of_lt (by simpa [size] using hr)).symm
    · show val E β ⟨(D.nodeChain n.id).foldr Kind.Bun n.kind, (n.node (D.label off)).outs,
        (n.node (D.label off)).ins, n.dim, n.ph⟩ = _
      have hr := h.range_pairs (Or.inl hC0)
      rw [val_tower E n.kind (D.nodeChain n.id) (Or.inl hC0) _ _ n.dim n.ph β h.posn
        (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) (hr true))
        (range_of_pairs _ _ _ _ _ _ _ (fun q hq => ?_) (hr false))]
      · congr 1
        refine List.map_congr_left fun t ht => ?_
        rw [plain_outCs _ h.plain, plain_inCs _ h.plain, blk_plain, blk_plain, dnode_eq,
          val_tensor E ν n.kind (h.boxkind hC0)]
        have e1 := dig_side D E off n β ν h (h.dims (Or.inl hC0)) true t ht
        have e2 := dig_side D E off n β ν h (h.dims (Or.inl hC0)) false t ht
        simp only [Bool.false_eq_true, ↓reduceIte] at e1 e2
        rw [e1, e2, ldgR_plain _ _ _ _ _ _ _ _ hT.1, ldgR_plain _ _ _ _ _ _ _ _ hT.2]
        simp [DNode.node, List.map_map, Function.comp_def]
      · rw [plain_outCs _ h.plain, zip_replicateC] at hq
        show q ∈ pairs n.outs.length _ _
        rw [pairs_plain _ _ _ hT.1]
        simpa [DNode.node] using hq
      · rw [plain_inCs _ h.plain, zip_replicateC] at hq
        show q ∈ pairs n.ins.length _ _
        rw [pairs_plain _ _ _ hT.2]
        simpa [DNode.node] using hq
  · have hdims := h.dims (Or.inr hT)
    have hr0 := h.range_pairs (Or.inr hT)
    have hr : ∀ q ∈ pairs n.outs.length (fun j => D.label off ⟨n.id, true, j⟩)
        (fun j => D.tail ⟨n.id, true, j⟩) ++ pairs n.ins.length (fun j => D.label off ⟨n.id, false, j⟩)
        (fun j => D.tail ⟨n.id, false, j⟩),
        β q.1 < E.dimv n.dim ^ size E.mult (D.nodeChain n.id ++ q.2) := by
      intro q hq
      rcases List.mem_append.1 hq with hq | hq
      · exact hr0 true q hq
      · exact hr0 false q hq
    have e1 := fun t ht => dig_side D E off n β ν h hdims true t ht
    have e2 := fun t ht => dig_side D E off n β ν h hdims false t ht
    simp only [Bool.false_eq_true, ↓reduceIte] at e1 e2
    rw [bnode_eq]
    rcases h.tailkind hT with hz | hx | ⟨hw, hI⟩
    · rw [val_bundle_Z E (D.nodeChain n.id) (n.node (D.label off)) n.outs.length
        (fun j => D.label off ⟨n.id, true, j⟩) (fun j => D.tail ⟨n.id, true, j⟩) n.ins.length
        (fun j => D.label off ⟨n.id, false, j⟩) (fun j => D.tail ⟨n.id, false, j⟩) rfl rfl β hz hT
        h.posn hr]
      congr 1
      refine List.map_congr_left fun t ht => ?_
      rw [dnode_eq, val_tensor E ν n.kind (by rw [hz]; simp), e1 t ht, e2 t ht, hz]
      exact tZ_perm _ _ _ _ ((ldgR_perm _ _ _ _ _ _).append (ldgR_perm _ _ _ _ _ _)).symm
    · rw [val_bundle_X E (D.nodeChain n.id) (n.node (D.label off)) n.outs.length
        (fun j => D.label off ⟨n.id, true, j⟩) (fun j => D.tail ⟨n.id, true, j⟩) n.ins.length
        (fun j => D.label off ⟨n.id, false, j⟩) (fun j => D.tail ⟨n.id, false, j⟩) rfl rfl β hx hT
        h.posn hr]
      congr 1
      refine List.map_congr_left fun t ht => ?_
      rw [dnode_eq, val_tensor E ν n.kind (by rw [hx]; simp), e1 t ht, e2 t ht, hx]
      exact tX_perm _ _ _ _ _ _ (ldgR_perm _ _ _ _ _ _).symm (ldgR_perm _ _ _ _ _ _).symm
    · rw [val_bundle_W E (D.nodeChain n.id) (n.node (D.label off)) n.outs.length
        (fun j => D.label off ⟨n.id, true, j⟩) (fun j => D.tail ⟨n.id, true, j⟩) n.ins.length
        (fun j => D.label off ⟨n.id, false, j⟩) (fun j => D.tail ⟨n.id, false, j⟩) rfl rfl β hw hT
        hI h.posn hr]
      congr 1
      refine List.map_congr_left fun t ht => ?_
      rw [dnode_eq, val_tensor E ν n.kind (by rw [hw]; simp), e1 t ht, e2 t ht, hw,
        ldgR_plain _ _ _ _ _ _ _ _ hI]
      refine (tensor_W_perm E _ _ (ldgR_perm _ _ _ _ _ _).symm _).trans ?_
      simp [DNode.node, List.map_map, Function.comp_def]

end Diagram

end ArchytasZX
