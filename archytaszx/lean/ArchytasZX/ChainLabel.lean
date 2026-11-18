/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Elim

/-!
# A diagram's network under a port labelling

`Diagram.netBy D lab` is `D` as a network with every leg labelled by `lab`, wire `j` summed at the
label of its first end, the boundary the slots `0 .. m - 1`. `equal_netBy`: when `lab` reads
slot `i` at boundary port `i`, both ends of a wire alike, distinct wires apart and above the
slots, and every leg lies on the boundary or a wire, `D.toNet o` equals `D.netBy lab`
(`sumOver_map_on`, wire `j`'s label `o + j` read as `lab` of its first end).
-/

namespace ArchytasZX

namespace Diagram

/-- `D` as a network with every leg labelled by `lab`. -/
def netBy (D : Diagram) (lab : Port → ℕ) : Net :=
  ⟨D.nodes.map (·.node lab), D.wires.map (fun w => lab w.1), List.range D.bnd.length⟩

/-- Every leg of every node on the boundary or a wire. -/
def portsOK (D : Diagram) : Prop :=
  ∀ n ∈ D.nodes, (∀ k < n.outs.length, D.okPort ⟨n.id, true, k⟩ = true) ∧
    ∀ k < n.ins.length, D.okPort ⟨n.id, false, k⟩ = true

theorem lab_of_locate (D : Diagram) (lab : Port → ℕ) (o : ℕ) (τ τ' : ℕ → ℕ)
    (hb : ∀ i (hi : i < D.bnd.length), lab D.bnd[i] = i)
    (hw : ∀ w ∈ D.wires, lab w.1 = lab w.2)
    (hs : ∀ i < D.bnd.length, τ i = τ' i)
    (hj : ∀ j (hj : j < D.wires.length), τ (lab D.wires[j].1) = τ' (o + j))
    (p : Port) (hp : D.okPort p = true) : τ (lab p) = τ' (D.label o p) := by
  unfold okPort at hp
  unfold label
  split at hp
  · next i h =>
    simp only [decide_eq_true_eq] at hp
    unfold locate at h
    split at h
    · next i' h' =>
      simp only [Option.some.injEq, Sum.inl.injEq] at h
      subst h
      obtain ⟨hi, he, -⟩ := List.idxOf?_eq_some_iff.1 h'
      rw [← he, hb i' hi]
      exact hs i' hi
    · simp at h
  · next j h =>
    simp only [decide_eq_true_eq] at hp
    unfold locate at h
    split at h
    · simp at h
    · next h' =>
      obtain ⟨j', hj', e⟩ := Option.map_eq_some_iff.1 h
      simp only [Sum.inr.injEq] at e
      subst e
      obtain ⟨hl, hpw, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hj'
      simp only [decide_eq_true_eq] at hpw
      rcases hpw with e | e
      · rw [← e]; exact hj j' hl
      · rw [← e, ← hw _ (List.getElem_mem hl)]; exact hj j' hl
  · simp at hp

theorem node_lab (n : DNode) (f g : Port → ℕ) (τ τ' : ℕ → ℕ)
    (ho : ∀ k < n.outs.length, τ (f ⟨n.id, true, k⟩) = τ' (g ⟨n.id, true, k⟩))
    (hi : ∀ k < n.ins.length, τ (f ⟨n.id, false, k⟩) = τ' (g ⟨n.id, false, k⟩)) (E : Env) :
    val E τ (n.node f) = val E τ' (n.node g) := by
  rw [← Function.id_comp τ, ← val_rename, node_rename, ← Function.id_comp τ', ← val_rename,
    node_rename]
  simp only [DNode.node]
  congr 2
  · exact List.map_congr_left fun k hk => ho k (List.mem_range.1 hk)
  · exact List.map_congr_left fun k hk => hi k (List.mem_range.1 hk)

/-- `D.toNet o` is `D.netBy lab` when `lab` reads each slot at its boundary port, both ends of
each wire alike, distinct wires at distinct labels above the slots, of the wire's dimension. -/
theorem equal_netBy (D : Diagram) (E : Env) (o : ℕ) (lab : Port → ℕ) (ho : D.bnd.length ≤ o)
    (hb : ∀ i (hi : i < D.bnd.length), lab D.bnd[i] = i)
    (hw : ∀ w ∈ D.wires, lab w.1 = lab w.2) (hok : D.portsOK)
    (hnd : (D.wires.map fun w => lab w.1).Nodup)
    (hlo : ∀ w ∈ D.wires, D.bnd.length ≤ lab w.1)
    (hdim : ∀ j (hj : j < D.wires.length), labDim E (lab D.wires[j].1) = labDim E (o + j)) :
    Equal E 1 (D.toNet o) 1 (D.netBy lab) := by
  refine ⟨List.Perm.refl _, fun σ _ => ?_⟩
  simp only [one_mul]
  set len := D.wires.length
  set m := D.bnd.length
  let g : ℕ → ℕ := fun x => if h : o ≤ x ∧ x - o < len then lab D.wires[x - o].1 else x
  have hg : ∀ j (hj : j < len), g (o + j) = lab D.wires[j].1 := fun j hj => by
    simp only [g, Nat.add_sub_cancel_left, Nat.le_add_right, true_and, hj, dite_true]
  set ls := (List.range len).map (o + ·)
  have hls : ls.map g = D.wires.map fun w => lab w.1 := by
    simp only [ls, List.map_map]
    refine List.ext_getElem (by simp [len]) fun i h1 h2 => ?_
    simp only [List.getElem_map, List.getElem_range, Function.comp]
    exact hg i (by simpa [len] using h2)
  have hgl : ∀ x ∈ ls, ∃ j, ∃ hj : j < len, x = o + j ∧ g x = lab D.wires[j].1 := by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    exact ⟨j, List.mem_range.1 hj, rfl, hg j (List.mem_range.1 hj)⟩
  have himg : ∀ y, y ∈ ls.map g → m ≤ y := by
    intro y hy
    rw [hls] at hy
    obtain ⟨w, hw', rfl⟩ := List.mem_map.1 hy
    exact hlo w hw'
  let σ' : ℕ → ℕ := fun x => if x ∈ ls then σ (g x) else σ x
  have e1 : ArchytasZX.eval E (D.toNet o) σ = ArchytasZX.eval E (D.toNet o) σ' := by
    unfold ArchytasZX.eval
    refine sumOver_ext _ _ _ _ (prodVal_reads E _) _ _ fun x _ hx => ?_
    have : x ∉ ls := hx
    simp [σ', this]
  rw [e1]
  unfold ArchytasZX.eval
  symm
  show sumOver (labDim E) (D.wires.map fun w => lab w.1) (prodVal E (D.netBy lab).nodes) σ =
    sumOver (labDim E) ls (prodVal E (D.toNet o).nodes) σ'
  rw [← hls]
  refine sumOver_map_on (labDim E) (labDim E) g ls ?_ ?_ ?_ _ _ σ σ' (fun τ τ' hτ hτ' hτf => ?_)
    (fun x hx => by simp [σ', hx])
  · exact List.nodup_range.map fun a b (h : o + a = o + b) => Nat.add_left_cancel h
  · intro x hx y hy e
    obtain ⟨a, ha, rfl, ga⟩ := hgl x hx
    obtain ⟨b, hb', rfl, gb⟩ := hgl y hy
    rw [ga, gb] at e
    have hnd' := (List.nodup_iff_injective_get.1 hnd)
    have : a = b := by
      have := @hnd' ⟨a, by simpa [len] using ha⟩ ⟨b, by simpa [len] using hb'⟩
        (by simpa using e)
      exact congrArg Fin.val this
    rw [this]
  · intro x hx
    obtain ⟨j, hj, rfl, gj⟩ := hgl x hx
    rw [gj]
    exact hdim j hj
  · simp only [prodVal, netBy, toNet, List.map_map]
    congr 1
    refine List.map_congr_left fun n hn => ?_
    simp only [Function.comp]
    obtain ⟨h1, h2⟩ := hok n hn
    have hs : ∀ i < m, τ i = τ' i := fun i hi => by
      have a1 : i ∉ ls.map g := fun h => by have := himg i h; omega
      have a2 : i ∉ ls := fun h => by
        obtain ⟨j, -, rfl⟩ := List.mem_map.1 h; omega
      rw [hτ i a1, hτ' i a2]
      simp [σ', a2]
    have hj : ∀ j (hj : j < D.wires.length), τ (lab D.wires[j].1) = τ' (o + j) := fun j hj => by
      have hm : o + j ∈ ls := List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
      rw [hτf _ hm, hg j hj]
    exact node_lab n _ _ τ τ'
      (fun k hk => lab_of_locate D lab o τ τ' hb hw hs hj _ (h1 k hk))
      (fun k hk => lab_of_locate D lab o τ τ' hb hw hs hj _ (h2 k hk)) E

end Diagram

end ArchytasZX
