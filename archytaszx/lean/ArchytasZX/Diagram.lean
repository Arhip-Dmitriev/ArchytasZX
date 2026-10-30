/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Net

/-!
# Diagrams and their meaning

A `Diagram` is the engine's diagram as data: nodes with the dimension of every leg, wires
joining two ports, and the boundary ports, outputs first. `Diagram.eval` sums over a value on
every wire the product of each node's tensor at the values on its legs; a boundary port reads
the assignment at its slot. `Diagram.toNet` labels the same diagram as a `Net`, and
`DEqual_iff` states that equality of diagrams is equality of their networks.
-/

namespace ArchytasZX

/-- A node's leg: node id, output side or not, position on that side. -/
structure Port where
  node : ℕ
  out : Bool
  pos : ℕ
  deriving DecidableEq, Repr

/-- A node: id, generator, dimension index, phase index (the second dimension's index of a
binder or splitter), and the dimension index of each output and input leg. -/
structure DNode where
  id : ℕ
  kind : Kind
  dim : ℕ
  ph : ℕ
  outs : List ℕ
  ins : List ℕ
  deriving DecidableEq, Repr

/-- Nodes, wires between two ports, boundary outputs and boundary inputs. -/
structure Diagram where
  nodes : List DNode
  wires : List (Port × Port)
  outs : List Port
  ins : List Port
  deriving DecidableEq, Repr

namespace Diagram

variable (D : Diagram)

/-- The boundary ports in slot order. -/
def bnd : List Port := D.outs ++ D.ins

/-- A port's dimension index. -/
def portDim (p : Port) : ℕ :=
  match D.nodes.find? (fun n => n.id = p.node) with
  | some n => if p.out then n.outs.getD p.pos 0 else n.ins.getD p.pos 0
  | none => 0

/-- Wire `j`'s dimension index. -/
def wireDim (j : ℕ) : ℕ := (D.wires[j]?.map fun w => D.portDim w.1).getD 0

/-- The boundary dimension indices in slot order. -/
def bdims : List ℕ := D.bnd.map D.portDim

/-- `inl i` for a port at boundary slot `i`, `inr j` for a port on wire `j`. -/
def locate (p : Port) : Option (ℕ ⊕ ℕ) :=
  match D.bnd.idxOf? p with
  | some i => some (.inl i)
  | none => (D.wires.findIdx? fun w => w.1 = p ∨ w.2 = p).map .inr

/-- A port's value: `σ` at its boundary slot, `τ` at its wire. -/
def value (σ τ : ℕ → ℕ) (p : Port) : ℕ :=
  match D.locate p with
  | some (.inl i) => σ i
  | some (.inr j) => τ j
  | none => 0

/-- A node with each leg labelled by `f` of its port. -/
def _root_.ArchytasZX.DNode.node (n : DNode) (f : Port → ℕ) : Node :=
  ⟨n.kind, (List.range n.outs.length).map fun k => f ⟨n.id, true, k⟩,
    (List.range n.ins.length).map fun k => f ⟨n.id, false, k⟩, n.dim, n.ph⟩

/-- The sum over every wire's value of the product of the node tensors. -/
noncomputable def eval (E : Env) (σ : ℕ → ℕ) : ℂ :=
  sumOver (fun j => E.dimv (D.wireDim j)) (List.range D.wires.length)
    (fun τ => (D.nodes.map fun n => val E id (n.node (D.value σ τ))).prod) (fun _ => 0)

/-- Every boundary slot within its dimension. -/
def InRange (E : Env) (σ : ℕ → ℕ) : Prop :=
  ∀ i < D.bnd.length, σ i < E.dimv (D.bdims.getD i 0)

/-- A port's network label: its slot, or `off + j` on wire `j`. -/
def label (off : ℕ) (p : Port) : ℕ :=
  match D.locate p with
  | some (.inl i) => i
  | some (.inr j) => off + j
  | none => 0

/-- The diagram as a network with wire `j` labelled `off + j`. -/
def toNet (off : ℕ) : Net :=
  ⟨D.nodes.map (·.node (D.label off)), (List.range D.wires.length).map (off + ·),
    List.range D.bnd.length⟩

/-- A port located on the boundary or a wire. -/
def okPort (p : Port) : Bool :=
  match D.locate p with
  | some (.inl i) => decide (i < D.bnd.length)
  | some (.inr j) => decide (j < D.wires.length)
  | none => false

/-- Every leg located, wire ends of one dimension, `off` above the slots, and the labels'
dimensions in `L`. -/
def check (L : List (ℕ × ℕ)) (off : ℕ) : Bool :=
  decide (D.bnd.length ≤ off) &&
  D.nodes.all (fun n => (List.range n.outs.length).all (fun k => D.okPort ⟨n.id, true, k⟩) &&
    (List.range n.ins.length).all (fun k => D.okPort ⟨n.id, false, k⟩)) &&
  D.wires.all (fun w => D.portDim w.1 = D.portDim w.2) &&
  (List.range D.wires.length).all (fun j => L.lookup (off + j) = some (D.wireDim j)) &&
  decide ((List.range D.bnd.length).map (fun i => L.lookup i) = D.bdims.map some)

end Diagram

/-- `s₁ • D₁` and `s₂ • D₂` are the same tensor. -/
def DEqual (E : Env) (s₁ : ℂ) (D₁ : Diagram) (s₂ : ℂ) (D₂ : Diagram) : Prop :=
  D₁.bdims = D₂.bdims ∧ ∀ σ, D₁.InRange E σ → s₁ * D₁.eval E σ = s₂ * D₂.eval E σ

/-- Summing labels `off + j` is summing `j`, for integrands that agree on such assignments. -/
theorem sumOver_shift (D D' : ℕ → ℕ) (off w : ℕ) (g h : (ℕ → ℕ) → ℂ) (σ : ℕ → ℕ)
    (hgh : ∀ a b, (∀ j < w, a (off + j) = b j) → (∀ l < off, a l = σ l) → g a = h b) :
    ∀ ls : List ℕ, (∀ j ∈ ls, D (off + j) = D' j) → ∀ a b,
      (∀ j < w, j ∉ ls → a (off + j) = b j) → (∀ l < off, a l = σ l) →
      sumOver D (ls.map (off + ·)) g a = sumOver D' ls h b := by
  intro ls
  induction ls with
  | nil => intro _ a b hab ha; exact hgh a b (fun j hj => hab j hj (by simp)) ha
  | cons l ls ih =>
    intro hD a b hab ha
    simp only [List.map_cons, sumOver]
    rw [hD l List.mem_cons_self]
    refine Finset.sum_congr rfl fun v _ => ih (fun j hj => hD j (List.mem_cons_of_mem _ hj))
      _ _ (fun j hj hn => ?_) (fun l' hl' => ?_)
    · by_cases hjl : j = l
      · subst hjl; simp
      · rw [Function.update_of_ne (by omega), Function.update_of_ne hjl]
        exact hab j hj (by simp [hjl, hn])
    · rw [Function.update_of_ne (by omega)]; exact ha l' hl'

namespace Diagram

variable (D : Diagram)

theorem node_rename (n : DNode) (f : Port → ℕ) (a : ℕ → ℕ) :
    (n.node f).rename a = n.node (a ∘ f) := by
  simp [DNode.node, Node.rename, List.map_map]

theorem okPort_label {off : ℕ} {σ a b : ℕ → ℕ} {p : Port} (hoff : D.bnd.length ≤ off)
    (hp : D.okPort p = true) (hab : ∀ j < D.wires.length, a (off + j) = b j)
    (ha : ∀ l < off, a l = σ l) : a (D.label off p) = D.value σ b p := by
  unfold okPort at hp
  unfold label value
  split at hp
  · next i h => exact ha i (by simp at hp; omega)
  · next j h => exact hab j (by simpa using hp)
  · exact absurd hp (by simp)

theorem eval_toNet (E : Env) (off : ℕ) (hc : D.check E.labs off = true) (σ : ℕ → ℕ) :
    ArchytasZX.eval E (D.toNet off) σ = D.eval E σ := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range]
    at hc
  obtain ⟨⟨⟨⟨hoff, hports⟩, -⟩, hlabs⟩, -⟩ := hc
  refine sumOver_shift _ _ off D.wires.length _ _ σ (fun a b hab ha => ?_) _ (fun j hj => ?_)
    σ (fun _ => 0) (fun j hj hn => absurd (List.mem_range.2 hj) hn) (fun _ _ => rfl)
  · simp only [prodVal, toNet, List.map_map]
    congr 1
    refine List.map_congr_left fun n hn => ?_
    have hn' := hports n hn
    show val E a (n.node (D.label off)) = val E id (n.node (D.value σ b))
    rw [← Function.id_comp a, ← val_rename, node_rename]
    simp only [DNode.node]
    congr 2
    · exact List.map_congr_left fun k hk =>
        D.okPort_label hoff (hn'.1 k (List.mem_range.1 hk)) hab ha
    · exact List.map_congr_left fun k hk =>
        D.okPort_label hoff (hn'.2 k (List.mem_range.1 hk)) hab ha
  · have := hlabs j (List.mem_range.1 hj)
    simp [labDim, this]

theorem labDim_bnd (E : Env) (off : ℕ) (hc : D.check E.labs off = true) (i : ℕ)
    (hi : i < D.bnd.length) : labDim E i = E.dimv (D.bdims.getD i 0) := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at hc
  have h := congrArg (·[i]?) hc.2
  simp only [List.getElem?_map, List.getElem?_range hi, Option.map_some] at h
  have hl : i < D.bdims.length := by simpa [bdims] using hi
  rw [List.getElem?_eq_getElem hl, Option.map_some, Option.some_inj] at h
  simp [labDim, h, List.getElem?_eq_getElem hl]

end Diagram

/-- Two diagrams are equal exactly when their networks are. -/
theorem DEqual_iff {E : Env} {s₁ s₂ : ℂ} {D₁ D₂ : Diagram} {o₁ o₂ : ℕ} {L : List (ℕ × ℕ)}
    (hL : E.labs = L) (h₁ : D₁.check L o₁ = true) (h₂ : D₂.check L o₂ = true) :
    DEqual E s₁ D₁ s₂ D₂ ↔ Equal E s₁ (D₁.toNet o₁) s₂ (D₂.toNet o₂) := by
  subst hL
  have hlen : D₁.bdims = D₂.bdims → D₁.bnd.length = D₂.bnd.length := fun h => by
    simpa [Diagram.bdims] using congrArg List.length h
  constructor
  · rintro ⟨hb, he⟩
    refine ⟨by simp [Diagram.toNet, hlen hb], fun σ hσ => ?_⟩
    rw [D₁.eval_toNet E o₁ h₁, D₂.eval_toNet E o₂ h₂]
    refine he σ fun i hi => ?_
    rw [← D₁.labDim_bnd E o₁ h₁ i hi]
    exact hσ i (by simpa [Diagram.toNet] using hi)
  · rintro ⟨hp, he⟩
    have hn : D₁.bnd.length = D₂.bnd.length := by
      simpa [Diagram.toNet] using hp.length_eq
    refine ⟨?_, fun σ hσ => ?_⟩
    · have c₁ := h₁; have c₂ := h₂
      simp only [Diagram.check, Bool.and_eq_true, decide_eq_true_eq] at c₁ c₂
      have := c₁.2.symm.trans (hn ▸ c₂.2)
      exact List.map_injective_iff.2 (fun _ _ h => Option.some_injective _ h) this
    · rw [← D₁.eval_toNet E o₁ h₁, ← D₂.eval_toNet E o₂ h₂]
      refine he σ fun l hl => ?_
      have hi : l < D₁.bnd.length := by simpa [Diagram.toNet] using hl
      rw [D₁.labDim_bnd E o₁ h₁ l hi]
      exact hσ l hi

end ArchytasZX
