/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainBox

/-!
# The expansion of a closed chain box in closed form

For a chain box `c` of `D` whose box is `D`'s only box, with no boundary port and no wire across
it other than its link wires (`ClosedChain`), `expandChain D c k` has `D`'s boundary
(`bnd_expand`), each node outside the box once and each node of the box once per copy with the
id `cpId j` (`nodes_expand`), and each wire outside the box once, each wire inside it once per
copy, then the link wires (`wires_expand`). `flatMap_swap` reorders a double list.
-/

namespace ArchytasZX

namespace Diagram

/-- `D` without the wires at the link ports of `c`. -/
def chainBase (D : Diagram) (c : DChain) : Diagram :=
  { D with wires := D.wires.filter fun w => !onLink c w }

/-- `c`'s box is `D`'s only box, has no port scope, holds no boundary port, and every wire off the
link ports has both ends in it or both out of it. -/
structure ClosedChain (D : Diagram) (c : DChain) : Prop where
  boxes : D.boxes = [c.box]
  ports : c.box.ports = []
  bnd : ∀ p ∈ D.bnd, p.node ∉ c.box.nodes
  side : ∀ w ∈ (D.chainBase c).wires, (w.1.node ∈ c.box.nodes ↔ w.2.node ∈ c.box.nodes)

/-- Copy `j` of each node of the box. -/
def cpNodeAt (D : Diagram) (c : DChain) (j : ℕ) (n : DNode) : DNode := { n with id := D.cpId c.box j n.id }

/-- Each wire inside the box once per copy, each outside once. -/
def chainWire (D : Diagram) (c : DChain) (k : ℕ) (w : Port × Port) : List (Port × Port) :=
  if w.1.node ∈ c.box.nodes then (List.range k).map fun j => (D.cpPort c.box j w.1, D.cpPort c.box j w.2)
  else [w]

theorem zip_map_range {α β : Type} (k : ℕ) (f : ℕ → α) (g : ℕ → β) :
    ((List.range k).map f).zip ((List.range k).map g) = (List.range k).map fun j => (f j, g j) := by
  induction k with
  | zero => rfl
  | succ k ih => simp [List.range_succ, List.zip_append, ih]

theorem zipIdx_singles (ds : List ℕ) (s : ℕ) :
    (ds.zipIdx s).flatMap (fun p : ℕ × ℕ => [p.1]) = ds := by
  induction ds generalizing s with
  | nil => rfl
  | cons d ds ih => simp [List.zipIdx_cons, ih]

theorem growSide_off (g : Port → Bool) (k x : ℕ) (out : Bool) (ds : List ℕ)
    (h : ∀ i, g ⟨x, out, i⟩ = false) : growSide g k x out ds = ds := by
  unfold growSide
  have : (ds.zipIdx.flatMap fun (p : ℕ × ℕ) => if g ⟨x, out, p.2⟩ then List.replicate k p.1 else [p.1]) =
      ds.zipIdx.flatMap fun (p : ℕ × ℕ) => [p.1] :=
    List.flatMap_congr fun p _ => by simp [h]
  simpa using this.trans (zipIdx_singles ds 0)

section closed

variable {D : Diagram} {c : DChain}

theorem chainBase_nodes : (D.chainBase c).nodes = D.nodes := rfl

theorem chainBase_bnd : (D.chainBase c).bnd = D.bnd := rfl

theorem cpId_base (j x : ℕ) : (D.chainBase c).cpId c.box j x = D.cpId c.box j x := rfl

theorem ClosedChain.cross (h : ClosedChain D c) (p : Port) (hp : p.node ∉ c.box.nodes) :
    (D.chainBase c).crossP c.box p = false := by
  unfold crossP inS
  simp only [decide_eq_true_eq, hp, decide_false, Bool.not_false, Bool.true_and,
    List.any_eq_false, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, not_or, not_and]
  intro w hw
  have hs := h.side w hw
  refine ⟨fun e h2 => hp (e ▸ hs.2 h2), fun e h1 => hp (e ▸ hs.1 h1)⟩

theorem ClosedChain.nimg_out (h : ClosedChain D c) (k : ℕ) (p : Port) (hp : p.node ∉ c.box.nodes) :
    (D.chainBase c).nimg c.box k p = [p] := by
  have hc : ∀ i, (D.chainBase c).crossP c.box ⟨p.node, p.out, i⟩ = false := fun i =>
    h.cross ⟨p.node, p.out, i⟩ hp
  simp only [nimg, inS, decide_eq_true_eq, hp, decide_false, Bool.false_eq_true, ite_false,
    growImg, h.cross p hp]
  simp [shiftPos, hc]

theorem chNimg_in (k : ℕ) (p : Port) (hp : p.node ∈ c.box.nodes) :
    (D.chainBase c).nimg c.box k p = (List.range k).map fun j => D.cpPort c.box j p := by
  simp [nimg, inS, hp, cpPort, cpId_base]

theorem ClosedChain.bnd_expand (h : ClosedChain D c) (k : ℕ) : (D.expandChain c k).bnd = D.bnd := by
  have hs : ∀ refs : List Port, (∀ p ∈ refs, p.node ∉ c.box.nodes) →
      (D.chainBase c).spliceN c.box k refs = refs := by
    intro refs hr
    have hf : refs.filter (inS c.box) = [] :=
      List.filter_eq_nil_iff.2 fun p hp => by simpa [inS] using hr p hp
    unfold spliceN
    simp only [hf, ite_true]
    rw [List.flatMap_congr (g := fun p => [p]) fun p hp => h.nimg_out k p (hr p hp)]
    simp
  have ho : ∀ p ∈ D.outs, p.node ∉ c.box.nodes := fun p hp => h.bnd p (List.mem_append_left _ hp)
  have hi : ∀ p ∈ D.ins, p.node ∉ c.box.nodes := fun p hp => h.bnd p (List.mem_append_right _ hp)
  show (D.chainBase c).spliceN c.box k (D.chainBase c).outs ++
      (D.chainBase c).spliceN c.box k (D.chainBase c).ins = D.bnd
  rw [hs _ (show ∀ p ∈ (D.chainBase c).outs, p.node ∉ c.box.nodes from ho),
    hs _ (show ∀ p ∈ (D.chainBase c).ins, p.node ∉ c.box.nodes from hi)]
  rfl

theorem ClosedChain.nodes_expand (h : ClosedChain D c) (k : ℕ) :
    (D.expandChain c k).nodes = D.nodes.flatMap fun n =>
      if n.id ∈ c.box.nodes then (List.range k).map fun j => D.cpNodeAt c j n else [n] := by
  show ((D.chainBase c).copyNodes c.box k).nodes = _
  unfold copyNodes
  simp only [chainBase_nodes]
  refine List.flatMap_congr fun n hn => ?_
  split_ifs with hb
  · rfl
  · have hc : ∀ out i, (D.chainBase c).crossP c.box ⟨n.id, out, i⟩ = false := fun out i =>
      h.cross ⟨n.id, out, i⟩ hb
    rw [growSide_off _ k n.id true n.outs (hc true), growSide_off _ k n.id false n.ins (hc false)]

theorem ClosedChain.wires_expand (h : ClosedChain D c) (k : ℕ) :
    (D.expandChain c k).wires = (D.chainBase c).wires.flatMap (D.chainWire c k) ++
      c.links.flatMap (D.linkWires c k) := by
  show ((D.chainBase c).copyNodes c.box k).wires ++ c.links.flatMap (D.linkWires c k) = _
  congr 1
  unfold copyNodes
  refine List.flatMap_congr fun w hw => ?_
  unfold chainWire
  split_ifs with hb
  · rw [chNimg_in k w.1 hb, chNimg_in k w.2 ((h.side w hw).1 hb), zip_map_range]
  · rw [h.nimg_out k w.1 hb, h.nimg_out k w.2 (fun h2 => hb ((h.side w hw).2 h2))]
    rfl

end closed

/-- A double list read the other way round is a permutation of it. -/
theorem flatMap_swap {α β γ : Type} (l₁ : List α) (l₂ : List β) (f : α → β → γ) :
    (l₁.flatMap fun a => l₂.map (f a)).Perm (l₂.flatMap fun b => l₁.map fun a => f a b) := by
  induction l₁ with
  | nil => simp
  | cons a l₁ ih =>
    simp only [List.flatMap_cons, List.map_cons]
    refine (ih.append_left _).trans ?_
    clear ih
    induction l₂ with
    | nil => simp
    | cons b l₂ ih2 =>
      simp only [List.map_cons, List.flatMap_cons, List.cons_append]
      refine List.Perm.cons _ ?_
      rw [← List.append_assoc]
      refine (List.perm_append_comm.append_right _).trans ?_
      rw [List.append_assoc]
      exact ih2.append_left _

end Diagram

end ArchytasZX
