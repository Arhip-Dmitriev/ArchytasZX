/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamBridge

/-!
# Decidable conditions of the family bridge

`WFb`: the box invariant `WF` with bounded quantifiers, decidable. `OrdAgree`: two boxed
diagrams read their slots alike and expand to the same boundary dimensions at `ks`.
-/

namespace ArchytasZX

namespace Diagram

instance (D : Diagram) (pr : DBox → Bool) (start : Option ℕ) (L : List DBox) :
    Decidable (ChainOK D pr start L) := by
  unfold ChainOK; infer_instance

variable (D : Diagram)

/-- `WF` with every quantifier bounded. -/
def WFb : Prop :=
  (D.nodes.map (·.id)).Nodup ∧ (D.boxes.map (·.id)).Nodup ∧
  (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).Perm D.legs ∧
  (∀ y ∈ D.boxes, y.nodes = [] ∨ y.ports = []) ∧
  (∀ y ∈ D.boxes, ∀ x ∈ y.nodes, x ∈ D.nodes.map (·.id)) ∧
  (∀ x ∈ D.nodes.map (·.id), ChainOK D (fun y => decide (x ∈ y.nodes)) none (D.nodeBoxes x)) ∧
  (∀ y ∈ D.boxes, ∀ x ∈ y.nodes, y ∈ D.nodeBoxes x) ∧
  (∀ p ∈ D.bnd, ChainOK D (fun y => decide (p ∈ y.ports))
    ((D.nodeBoxes p.node).getLast?.map (·.id)) (D.portBoxes p)) ∧
  (∀ p ∈ D.bnd, ∀ y ∈ D.boxes, p ∈ y.ports → y ∈ D.portBoxes p) ∧
  (∀ y ∈ D.boxes, ∀ p ∈ y.ports, p ∈ D.bnd) ∧
  (∀ w ∈ D.wires, D.nodeBoxes w.1.node <+: D.nodeBoxes w.2.node ∨
    D.nodeBoxes w.2.node <+: D.nodeBoxes w.1.node) ∧
  (∀ y ∈ D.boxes, y.nodes ≠ [] ∨ y.ports ≠ []) ∧
  (∀ w ∈ D.wires, D.portDim w.1 = D.portDim w.2)

instance : Decidable D.WFb := by unfold WFb; infer_instance

variable {D}

theorem WF_of_WFb (h : D.WFb) : WF D := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h
  exact ⟨h1, h2, h3, h4, h5, h6, fun x y hy hx => h7 y hy x hx, h8, h9, h10, h11, h12, h13⟩

variable (D)

/-- The structural conditions of `NodeS`. -/
def NodeSb (n : DNode) : Prop :=
  n.kind.plain = true ∧ (D.nodeChain n.id ≠ [] → n.kind ≠ .B ∧ n.kind ≠ .S) ∧
  (¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = []) →
    n.kind = .Z ∨ n.kind = .X ∨ (n.kind = .W ∧ tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])) ∧
  ((D.nodeChain n.id ≠ [] ∨ ¬(tls n.outs.length (fun j => D.tail ⟨n.id, true, j⟩) = [] ∧
      tls n.ins.length (fun j => D.tail ⟨n.id, false, j⟩) = [])) → ∀ p ∈ n.legs, D.portDim p = n.dim)

instance (n : DNode) : Decidable (D.NodeSb n) := by unfold NodeSb; infer_instance

/-- The dimension index `dimFor` gives `base` along `c` is a bundle of the plain index `base`
along `c` in the family `F`. -/
def DimOK (F : Fam) (base : ℕ) (c : List ℕ) : Prop :=
  F.bund.lookup base = none ∧
    (c = [] ∨ ((F.bund.lookup (dimFor F.bund base c)).isSome = true ∧
      F.root (dimFor F.bund base c) = base ∧ F.chain (dimFor F.bund base c) = c))

instance (F : Fam) (base : ℕ) (c : List ℕ) : Decidable (DimOK F base c) := by
  unfold DimOK; infer_instance

/-- Everything the family bridge checks of one boxed diagram labelled at `off` in `F`. -/
def FamOK (F : Fam) (off : ℕ) : Prop :=
  D.WFb ∧ D.bnd.length ≤ off ∧ (∀ n ∈ D.nodes, D.NodeSb n) ∧
  (∀ j (hj : j < D.wires.length),
    F.labs.lookup (off + j) = some (dimFor F.bund (D.portDim D.wires[j].1) (D.legChain D.wires[j].1)) ∧
    DimOK F (D.portDim D.wires[j].1) (D.legChain D.wires[j].1)) ∧
  (∀ i (hi : i < D.bnd.length),
    F.labs.lookup i = some (dimFor F.bund (D.portDim D.bnd[i]) (D.legChain D.bnd[i])) ∧
    DimOK F (D.portDim D.bnd[i]) (D.legChain D.bnd[i]))

instance (F : Fam) (off : ℕ) : Decidable (D.FamOK F off) := by unfold FamOK; infer_instance

/-- Two boxed diagrams read their slots alike and expand to the same boundary dimensions at
`ks`. -/
def OrdAgree (D₁ D₂ : Diagram) (ks : ℕ → ℕ) : Prop :=
  (∀ σ'' : ℕ → ℕ, ∀ i (hi : i < D₁.bnd.length), ∀ t ∈ cidx ks (D₁.legChain D₁.bnd[i]),
    D₁.bmap ks (fun j _ => σ'' j) i t = D₂.bmap ks (fun j _ => σ'' j) i t) ∧
  (D₁.expand ks).bdims = (D₂.expand ks).bdims

variable {D}

theorem dimv_dimFor (F : Fam) (ks : ℕ → ℕ) (base : ℕ) (c : List ℕ) (h : DimOK F base c) :
    (F.env ks).dimv (dimFor F.bund base c) = (F.env ks).dimv base ^ size ks c := by
  obtain ⟨hb, hc⟩ := h
  have hbase : (F.env ks).dimv base = F.dims base := by
    simp [Fam.env, Fam.dimv, hb]
  rw [hbase]
  rcases hc with rfl | ⟨hs, hr, hch⟩
  · simp [dimFor, Fam.env, Fam.dimv, hb, size]
  · obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
    simp only [Fam.env, Fam.dimv, hv, hr, hch]
    rfl

theorem labDim_of (F : Fam) (ks : ℕ → ℕ) (l base : ℕ) (c : List ℕ)
    (hl : F.labs.lookup l = some (dimFor F.bund base c)) (h : DimOK F base c) :
    labDim (F.env ks) l = (F.env ks).dimv base ^ size ks c := by
  rw [← dimv_dimFor F ks base c h]
  simp [labDim, Fam.env, hl]

theorem NodeS_of {E : Env} {n : DNode} (h : D.NodeSb n) (hpos : ∀ i, 0 < E.dimv i) : D.NodeS E n :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2, fun _ _ => hpos _, hpos _⟩

theorem Fam.chainF_congr {F F' : Fam} (h : F.bund = F'.bund) (n i : ℕ) : F.chainF n i = F'.chainF n i := by
  induction n generalizing i with
  | zero => rfl
  | succ n ih => simp only [Fam.chainF, h]; split <;> simp [ih]

theorem Fam.rootF_congr {F F' : Fam} (h : F.bund = F'.bund) (n i : ℕ) : F.rootF n i = F'.rootF n i := by
  induction n generalizing i with
  | zero => rfl
  | succ n ih => simp only [Fam.rootF, h]; split <;> simp [ih]

theorem FamOK_congr {F F' : Fam} {off : ℕ} (hb : F.bund = F'.bund) (hl : F.labs = F'.labs)
    (h : D.FamOK F off) : D.FamOK F' off := by
  have hc : ∀ i, F.chain i = F'.chain i := fun i => Fam.chainF_congr hb _ _
  have hr : ∀ i, F.root i = F'.root i := fun i => Fam.rootF_congr hb _ _
  have hd : ∀ base c, DimOK F base c → DimOK F' base c := fun base c h => by
    unfold DimOK at h ⊢
    simpa [hb, hc, hr] using h
  obtain ⟨w, b, n, l, s⟩ := h
  exact ⟨w, b, n, fun j hj => ⟨by rw [← hl, ← hb]; exact (l j hj).1, hd _ _ (l j hj).2⟩,
    fun i hi => ⟨by rw [← hl, ← hb]; exact (s i hi).1, hd _ _ (s i hi).2⟩⟩

/-- Equal family networks of two checked boxed diagrams whose expansions order their boundary
alike are equal expansions at `ks`. -/
theorem DEqual_expand_of_checks {F : Fam} {D₁ D₂ : Diagram} {o₁ o₂ : ℕ} (ks : ℕ → ℕ) (s₁ s₂ : ℂ)
    (hc₁ : D₁.FamOK F o₁) (hc₂ : D₂.FamOK F o₂) (hbd : D₁.bndData = D₂.bndData)
    (hpos : ∀ i, 0 < F.dims i) (hord : OrdAgree D₁ D₂ ks)
    (hEq : Equal (F.env ks) s₁ (D₁.toNetB o₁) s₂ (D₂.toNetB o₂)) :
    DEqual (F.env ks) s₁ (D₁.expand ks) s₂ (D₂.expand ks) := by
  have hp : ∀ i, 0 < (F.env ks).dimv i := fun i => Fam.env_dimv_pos F ks i (hpos _)
  obtain ⟨w₁, b₁, n₁, l₁, s₁'⟩ := hc₁
  obtain ⟨w₂, b₂, n₂, l₂, -⟩ := hc₂
  exact DEqual_expand_of_net (WF_of_WFb w₁) (WF_of_WFb w₂) (F.env ks) o₁ o₂ s₁ s₂ b₁ b₂
    (fun n hn => NodeS_of (n₁ n hn) hp) (fun n hn => NodeS_of (n₂ n hn) hp)
    (fun j hj => labDim_of F ks _ _ _ (l₁ j hj).1 (l₁ j hj).2)
    (fun j hj => labDim_of F ks _ _ _ (l₂ j hj).1 (l₂ j hj).2)
    (fun i hi => labDim_of F ks _ _ _ (s₁' i hi).1 (s₁' i hi).2) hbd hord.1 hord.2 hEq

end Diagram

end ArchytasZX
