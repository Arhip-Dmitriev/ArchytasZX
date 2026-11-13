/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainLink

/-!
# The one-spider chain expanded at every `k`

`expand_closed` gives `expandChain` of `ChainBoxExample` at every `k ≥ 1` in closed form: the
head identity `0`, the copies `3 + j` in copy order wired output `0` to input `0`, the tail
identity `2`, and the copies' second outputs on the boundary before the tail's output.
`net_closed` gives its network at any wire offset `o` as `netK o k`. `expand_chainNet` gives,
for every `k` and every offset `o ≥ k + 3` (above the `k + 3` boundary slots), that network
relabelled by `fK o k` equal to `chainNet 100 [5] base copy (k + 1)`: the tail identity dropped
(`Equal.dropIdentity`), then `chainNet`'s nodes, summed labels and boundary in closed form
(`cn_nodes`, `cn_int`, `cn_bnd`).
-/

namespace ArchytasZX

namespace ChainBoxExample

open Diagram

/-- The nodes at `k` copies. -/
def nodesK (k : ℕ) : List DNode :=
  ⟨0, .I, 0, 0, [0], [0]⟩ :: (List.range k).map (fun j => ⟨3 + j, .Z, 0, 0, [0, 0], [0]⟩) ++
    [⟨2, .I, 0, 0, [0], [0]⟩]

/-- The chain wires at `k + 1` copies, head to tail. -/
def wiresK (k : ℕ) : List (Port × Port) :=
  (⟨0, true, 0⟩, ⟨3, false, 0⟩) ::
    ((List.range k).map fun j => ((⟨3 + j, true, 0⟩ : Port), (⟨3 + (j + 1), false, 0⟩ : Port))) ++
    [(⟨3 + k, true, 0⟩, ⟨2, false, 0⟩)]

/-- The diagram at `k + 1` copies. -/
def closedK (k : ℕ) : Diagram :=
  { nodes := nodesK (k + 1),
    wires := wiresK k,
    outs := (List.range (k + 1)).map (fun j => ⟨3 + j, true, 1⟩) ++ [⟨2, true, 0⟩],
    ins := [⟨0, false, 0⟩],
    boxes := [] }

/-- The diagram without its link wires. -/
def D0 : Diagram := { D with wires := D.wires.filter fun w => !onLink c w }

theorem D0_eq : D0 = { D with wires := [] } := by decide

theorem cpId_D0 (j x : ℕ) (hx : x = 1) : D0.cpId c.box j x = 3 + j := by
  subst hx
  simp [cpId, D0_eq, fresh, D, c]

theorem expand_closed (k : ℕ) : D.expandChain c (k + 1) = closedK k := by
  have hX : D.expandChain c (k + 1) =
      { (D0.copyNodes c.box (k + 1)) with
        wires := (D0.copyNodes c.box (k + 1)).wires ++ c.links.flatMap (D.linkWires c (k + 1)) } :=
    rfl
  rw [hX]
  simp [copyNodes, D0_eq, closedK, nodesK, wiresK, linkWires, partner, cpPort, D, c, cpId, fresh,
    spliceN, nimg, inS, growImg, growSide, crossP, shiftPos]
  have hf : List.findIdx (inS ⟨0, 0, none, [1], []⟩)
      [(⟨1, true, 1⟩ : Port), ⟨2, true, 0⟩] = 0 := by decide
  rw [hf]
  simp [nimg, inS, growImg, crossP, shiftPos, List.flatMap_cons, List.flatMap_nil]
  induction (List.range (k + 1)) with
  | nil => rfl
  | cons j l ih => simp [ih]

/-- Boundary slot `i` of `closedK k`. -/
def bK (k i : ℕ) : Port :=
  if i ≤ k then ⟨3 + i, true, 1⟩ else if i = k + 1 then ⟨2, true, 0⟩ else ⟨0, false, 0⟩

/-- Wire `i` of `closedK k`. -/
def wK (k i : ℕ) : Port × Port :=
  (if i = 0 then ⟨0, true, 0⟩ else ⟨3 + (i - 1), true, 0⟩,
    if i = k + 1 then ⟨2, false, 0⟩ else ⟨3 + i, false, 0⟩)

theorem bnd_closed (k : ℕ) : (closedK k).bnd = (List.range (k + 3)).map (bK k) := by
  rw [show k + 3 = k + 1 + 1 + 1 from rfl, List.range_succ, List.range_succ]
  simp only [bnd, closedK, List.map_append, List.map_cons, List.map_nil, List.append_assoc]
  congr 1
  · exact List.map_congr_left fun i hi => by
      simp [bK, show i ≤ k by simpa [Nat.lt_succ_iff] using List.mem_range.1 hi]
  · simp [bK]

theorem wires_closed (k : ℕ) : (closedK k).wires = (List.range (k + 2)).map (wK k) := by
  rw [show k + 2 = k + 1 + 1 from rfl, List.range_succ_eq_map, List.range_succ]
  simp only [closedK, wiresK, List.map_cons, List.map_append, List.map_map, List.map_nil,
    List.cons_append]
  refine congrArg₂ _ (by simp [wK]) (congrArg₂ _ ?_ (by simp [wK]))
  exact List.map_congr_left fun j hj => by
    have := List.mem_range.1 hj
    simp [wK, show j ≠ k by omega]

theorem bK_inj (k i j : ℕ) (hi : i < k + 3) (hj : j < k + 3) (h : bK k i = bK k j) : i = j := by
  unfold bK at h
  split_ifs at h <;> simp only [Port.mk.injEq] at h <;> omega

theorem locate_slot (k i : ℕ) (hi : i < k + 3) : (closedK k).locate (bK k i) = some (.inl i) := by
  unfold locate
  rw [(List.idxOf?_eq_some_iff).2 ⟨by simpa [bnd_closed] using hi, by simp [bnd_closed],
    fun j hj h => by
      simp only [bnd_closed, List.getElem_map, List.getElem_range] at h
      exact absurd (bK_inj k j i (by omega) hi h) (by omega)⟩]

theorem locate_wire (k i : ℕ) (p : Port) (hi : i < k + 2) (hp : (wK k i).1 = p ∨ (wK k i).2 = p)
    (hb : p ∉ (closedK k).bnd) (hprev : ∀ j < i, ¬((wK k j).1 = p ∨ (wK k j).2 = p)) :
    (closedK k).locate p = some (.inr i) := by
  unfold locate
  rw [(List.idxOf?_eq_none_iff).2 hb]
  simp only [Option.map_eq_some_iff]
  refine ⟨i, (List.findIdx?_eq_some_iff_getElem).2 ⟨by simpa [wires_closed] using hi, ?_, ?_⟩,
    rfl⟩
  · simpa [wires_closed] using hp
  · intro j hj
    simpa [wires_closed] using hprev j hj

theorem not_bnd (k : ℕ) (p : Port) (h : ∀ i < k + 3, bK k i ≠ p) : p ∉ (closedK k).bnd := by
  simp only [bnd_closed, List.mem_map, List.mem_range, not_exists, not_and]
  exact h

theorem label_out1 (k j o : ℕ) (hj : j ≤ k) : (closedK k).label o ⟨3 + j, true, 1⟩ = j := by
  have := locate_slot k j (by omega)
  simp only [bK, hj, ↓reduceIte] at this
  simp [label, this]

theorem label_tail_out (k o : ℕ) : (closedK k).label o ⟨2, true, 0⟩ = k + 1 := by
  have := locate_slot k (k + 1) (by omega)
  simp only [bK, show ¬ k + 1 ≤ k by omega, ↓reduceIte] at this
  simp [label, this]

theorem label_head_in (k o : ℕ) : (closedK k).label o ⟨0, false, 0⟩ = k + 2 := by
  have := locate_slot k (k + 2) (by omega)
  simp only [bK, show ¬ k + 2 ≤ k by omega, show k + 2 ≠ k + 1 by omega, ↓reduceIte] at this
  simp [label, this]

theorem label_head_out (k o : ℕ) : (closedK k).label o ⟨0, true, 0⟩ = o := by
  have := locate_wire k 0 ⟨0, true, 0⟩ (by omega) (by simp [wK])
    (not_bnd k _ fun i _ => by unfold bK; split_ifs <;> simp) (fun j hj => by omega)
  simp [label, this]

theorem label_in (k j o : ℕ) (hj : j ≤ k) : (closedK k).label o ⟨3 + j, false, 0⟩ = o + j := by
  have := locate_wire k j ⟨3 + j, false, 0⟩ (by omega) (by simp [wK, show j ≠ k + 1 by omega])
    (not_bnd k _ fun i _ => by unfold bK; split_ifs <;> simp; omega)
    (fun i hi => by unfold wK; split_ifs <;> simp <;> omega)
  simp [label, this]

theorem label_out0 (k j o : ℕ) (hj : j ≤ k) :
    (closedK k).label o ⟨3 + j, true, 0⟩ = o + (j + 1) := by
  have := locate_wire k (j + 1) ⟨3 + j, true, 0⟩ (by omega) (by simp [wK])
    (not_bnd k _ fun i _ => by unfold bK; split_ifs <;> simp; omega)
    (fun i hi => by unfold wK; split_ifs <;> simp <;> omega)
  simp [label, this]

theorem label_tail_in (k o : ℕ) : (closedK k).label o ⟨2, false, 0⟩ = o + (k + 1) := by
  have := locate_wire k (k + 1) ⟨2, false, 0⟩ (by omega) (by simp [wK])
    (not_bnd k _ fun i _ => by unfold bK; split_ifs <;> simp)
    (fun i hi => by unfold wK; split_ifs <;> simp <;> omega)
  simp [label, this]

/-- The network of `closedK k` with wire `j` labelled `o + j`: the head identity from slot
`k + 2` to `o`, copy `j` from `o + j` to `o + (j + 1)` and slot `j`, the tail identity from
`o + (k + 1)` to slot `k + 1`. -/
def netK (o k : ℕ) : Net :=
  ⟨(⟨.I, [o], [k + 2], 0, 0⟩ ::
      (List.range (k + 1)).map fun j => ⟨.Z, [o + (j + 1), j], [o + j], 0, 0⟩) ++
      [⟨.I, [k + 1], [o + (k + 1)], 0, 0⟩],
    (List.range (k + 1)).map (o + ·) ++ [o + (k + 1)], List.range (k + 3)⟩

theorem net_closed (o k : ℕ) : (closedK k).toNet o = netK o k := by
  have hw : (closedK k).wires.length = k + 2 := by simp [wires_closed]
  have hb : (closedK k).bnd.length = k + 3 := by simp [bnd_closed]
  simp only [toNet, hw, hb, netK, Net.mk.injEq]
  refine ⟨?_, by rw [show k + 2 = k + 1 + 1 from rfl, List.range_succ]; simp, trivial⟩
  rw [show (closedK k).nodes = nodesK (k + 1) from rfl]
  simp only [nodesK, List.map_cons, List.map_append, List.map_map, List.map_nil,
    List.cons_append, DNode.node, Function.comp_def]
  simp only [List.length_cons, List.length_nil, zero_add, List.range_one, List.map_cons,
    List.map_nil, show List.range 2 = [0, 1] from rfl, label_head_out, label_head_in,
    label_tail_out, label_tail_in, List.cons.injEq, true_and]
  refine congrArg (· ++ _) (List.map_congr_left fun j hj => ?_)
  have : j ≤ k := by have := List.mem_range.1 hj; omega
  simp [label_out0 k j o this, label_out1 k j o this, label_in k j o this]

/-- The labels of `netK o k` onto the chain scheme: slot `j ≤ k` to `150 + 100 j`, slot `k + 1`
to `105 + 100 k`, slot `k + 2` to `3`, wire label `o + j` to `5 + 100 j`, any other label to
`100 x + 7`. -/
def fK (o k x : ℕ) : ℕ :=
  if x ≤ k then 150 + x * 100
  else if x = k + 1 then 105 + k * 100
  else if x = k + 2 then 3
  else if o ≤ x ∧ x ≤ o + k then 5 + (x - o) * 100
  else x * 100 + 7

theorem fK_inj (o k : ℕ) (ho : k + 3 ≤ o) : Function.Injective (fK o k) := by
  intro x y h
  unfold fK at h
  revert ho
  split_ifs at h <;> omega

theorem fK_slot (o k j : ℕ) (hj : j ≤ k) : fK o k j = 150 + j * 100 := by simp [fK, hj]

theorem fK_wire (o k j : ℕ) (ho : k + 3 ≤ o) (hj : j ≤ k) : fK o k (o + j) = 5 + j * 100 := by
  unfold fK
  split_ifs <;> omega

theorem fK_tail (o k : ℕ) : fK o k (k + 1) = 105 + k * 100 := by simp [fK]

theorem fK_head (o k : ℕ) : fK o k (k + 2) = 3 := by simp [fK]

/-- Copy `i` of the chain at level `i`. -/
def cpN (i : ℕ) : Node := ⟨.Z, [105 + i * 100, 150 + i * 100], [5 + i * 100], 0, 0⟩

theorem cn_nodes (m : ℕ) : (chainNet 100 [5] base copy m).nodes =
    (List.range m).reverse.map cpN ++ [⟨.I, [5], [3], 0, 0⟩] := by
  induction m with
  | zero => rfl
  | succ m ih =>
    simp only [chainNet, ih, copy, Net.mapL, List.range_succ, List.reverse_append,
      List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append, List.map_cons,
      List.map_nil, Node.rename, cpN]

theorem cn_int (m : ℕ) : (chainNet 100 [5] base copy m).int =
    (List.range m).reverse.map (5 + · * 100) := by
  induction m with
  | zero => rfl
  | succ m ih =>
    simp [chainNet, ih, copy, Net.mapL, List.range_succ, Nat.add_comm]

theorem cn_bnd (m : ℕ) : (chainNet 100 [5] base copy m).bnd.Perm
    (3 :: (5 + m * 100) :: (List.range m).map (150 + · * 100)) := by
  induction m with
  | zero => exact List.Perm.refl _
  | succ m ih =>
    simp only [chainNet]
    have hf := ih.filter (fun x => decide (x ∉ [5].map (· + m * 100)))
    have hl : ((3 :: (5 + m * 100) :: (List.range m).map (150 + · * 100)).filter
        fun x => decide (x ∉ [5].map (· + m * 100))) = 3 :: (List.range m).map (150 + · * 100) := by
      rw [List.filter_cons_of_pos (by simp; omega), List.filter_cons_of_neg (by simp)]
      refine congrArg _ (List.filter_eq_self.2 fun x hx => ?_)
      obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
      simp; omega
    rw [hl] at hf
    have hr : (((copy m).mapL (· + m * 100)).bnd.filter
        fun x => decide (x ∉ [5].map (· + m * 100))) = [105 + m * 100, 150 + m * 100] := by
      simp [copy, Net.mapL, Nat.add_comm]; omega
    rw [hr, List.range_succ, List.map_append, List.map_singleton,
      show 5 + (m + 1) * 100 = 105 + m * 100 by ring]
    refine (hf.append_right _).trans ?_
    refine List.Perm.cons 3 ?_
    exact List.perm_middle.trans (List.Perm.cons _ (List.Perm.refl _))

/-- The one-spider chain expanded at `k + 1`, its network at offset `o ≥ k + 3` relabelled by
`fK o k`, equals `chainNet` at `k + 1`. -/
theorem expand_chainNet (k o : ℕ) (ho : k + 3 ≤ o) :
    Equal E 1 (((D.expandChain c (k + 1)).toNet o).mapL (fK o k)) 1
      (chainNet 100 [5] base copy (k + 1)) := by
  rw [expand_closed, net_closed]
  have h0 := Equal.dropIdentity E
    (⟨.I, [o], [k + 2], 0, 0⟩ :: (List.range (k + 1)).map fun j => ⟨.Z, [o + (j + 1), j], [o + j], 0, 0⟩)
    ((List.range (k + 1)).map (o + ·)) (List.range (k + 3)) (o + (k + 1)) (k + 1) 0
    (by simp [labDim, E]) (by simp [labDim, E]) (by omega) (by simp)
    (by simp only [List.mem_map, List.mem_range, not_exists, not_and]; omega)
  have h1 := h0.mapL E (fK o k) (fK_inj o k ho) (fun x => by simp [labDim, E])
  refine h1.reorder E ?_ ?_ ?_
  · rw [cn_nodes]
    have hc : ∀ j ∈ List.range (k + 1), (Node.rename (fK o k) (Node.rename (swapTo (o + (k + 1)) (k + 1))
        ⟨.Z, [o + (j + 1), j], [o + j], 0, 0⟩)) = cpN j := by
      intro j hj
      have hj := List.mem_range.1 hj
      simp only [Node.rename, List.map_cons, List.map_nil, cpN, Node.mk.injEq, List.cons.injEq,
        and_true, true_and]
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · by_cases h : j = k
        · subst h; simp [swapTo, fK_tail]
        · rw [show swapTo (o + (k + 1)) (k + 1) (o + (j + 1)) = o + (j + 1) by
            simp [swapTo]; omega, fK_wire o k (j + 1) ho (by omega)]
          ring
      · rw [show swapTo (o + (k + 1)) (k + 1) j = j by simp [swapTo]; omega,
          fK_slot o k j (by omega)]
      · rw [show swapTo (o + (k + 1)) (k + 1) (o + j) = o + j by simp [swapTo]; omega,
          fK_wire o k j ho (by omega)]
    simp only [Net.mapL, List.map_cons, List.map_map]
    rw [List.map_congr_left (g := cpN) fun j hj => by simpa [Function.comp] using hc j hj]
    have hh : Node.rename (fK o k) (Node.rename (swapTo (o + (k + 1)) (k + 1))
        ⟨.I, [o], [k + 2], 0, 0⟩) = ⟨.I, [5], [3], 0, 0⟩ := by
      simp only [Node.rename, List.map_cons, List.map_nil, Node.mk.injEq, List.cons.injEq,
        and_true, true_and]
      refine ⟨?_, ?_⟩
      · rw [show swapTo (o + (k + 1)) (k + 1) o = o by simp [swapTo],
          show fK o k o = fK o k (o + 0) from rfl, fK_wire o k 0 ho (by omega)]
      · rw [show swapTo (o + (k + 1)) (k + 1) (k + 2) = k + 2 by simp [swapTo]; omega, fK_head]
    rw [hh, List.map_reverse]
    exact (List.perm_append_singleton _ _).symm.trans ((List.reverse_perm _).symm.append_right _)
  · rw [cn_int]
    simp only [Net.mapL, List.map_map, List.map_reverse]
    rw [List.map_congr_left (g := (5 + · * 100)) fun j hj => by
      simpa [Function.comp] using fK_wire o k j ho (by have := List.mem_range.1 hj; omega)]
    exact (List.reverse_perm _).symm
  · refine List.Perm.trans ?_ (cn_bnd (k + 1)).symm
    simp only [Net.mapL]
    rw [show k + 3 = k + 1 + 1 + 1 from rfl, List.range_succ, List.range_succ, List.map_append,
      List.map_append, List.map_singleton, List.map_singleton, fK_tail, fK_head,
      List.map_congr_left (g := (150 + · * 100)) fun j hj =>
        fK_slot o k j (by have := List.mem_range.1 hj; omega),
      show 5 + (k + 1) * 100 = 105 + k * 100 by ring]
    exact (List.perm_append_singleton _ _).trans
      (List.Perm.cons 3 (List.perm_append_singleton _ _))

end ChainBoxExample

end ArchytasZX
