/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainForm
import ArchytasZX.ChainLabel
import ArchytasZX.PeriodicChain

/-!
# Closed chain boxes as chain families

For a closed chain box (`ClosedChain`) whose template meets `ChainFamOK` (decided by the kernel),
the network of `expandChain D c k` at any offset above the slots equals `chFam D c k`, the chain
family `famNet` of the template, at every `k` (`expand_chFam`). The template is labelled once:
the slots `0 .. m - 1`, the wires outside the box `m + i`, link `t` at `W + t` (`chU`), the wires
inside the box `W + L + i` (`chLab0`, `chLabT`); `chTl` holds the nodes outside the box that no
link ends at, `chHdN` those a link ends at (`chHd`), `chCp` the box. Copy `j` reads its template
label raised `j` levels (`sh W j`), the head side its link labels raised `k` levels (`chLabK`).
The proof labels the expansion by `chLabK` (`equal_netBy`) and reorders its nodes and summed
labels into `famNet`'s.
-/

namespace ArchytasZX

namespace Diagram

variable (D : Diagram) (c : DChain) (W : ℕ)

/-- The wires inside the box. -/
def chIn : List (Port × Port) :=
  (D.chainBase c).wires.filter fun w => decide (w.1.node ∈ c.box.nodes)

/-- The wires outside the box, link wires left out. -/
def chOut : List (Port × Port) :=
  (D.chainBase c).wires.filter fun w => !decide (w.1.node ∈ c.box.nodes)

/-- The link labels at level `0`. -/
def chU : List ℕ := (List.range c.links.length).map (W + ·)

/-- A box leg's template label: its inside wire `i` at `W + L + i`, the second port of link `t`
at `W + t`, the first at `2 W + t`. -/
def chLabT (q : Port) : ℕ :=
  match (D.chIn c).findIdx? (fun w => decide (w.1 = q ∨ w.2 = q)) with
  | some i => W + c.links.length + i
  | none =>
    match c.links.findIdx? (fun l => decide (l.2 = q)) with
    | some t => W + t
    | none =>
      match c.links.findIdx? (fun l => decide (l.1 = q)) with
      | some t => 2 * W + t
      | none => 0

/-- An outside leg's label: its slot, its outside wire `i` at `m + i`, a link's end at `W + t`. -/
def chLab0 (p : Port) : ℕ :=
  match D.bnd.idxOf? p with
  | some i => i
  | none =>
    match (D.chOut c).findIdx? (fun w => decide (w.1 = p ∨ w.2 = p)) with
    | some i => D.bnd.length + i
    | none =>
      match c.links.findIdx? (fun l => decide (D.partner l.2 = some p ∨ D.partner l.1 = some p)) with
      | some t => W + t
      | none => 0

/-- Whether a link's first port meets the node `x`. -/
def chHd (x : ℕ) : Bool := c.links.any fun l => (D.partner l.1).any fun P => decide (P.node = x)

/-- The outside nodes no link's first port meets, over the outside wires. -/
def chTl : Net :=
  ⟨(D.nodes.filter fun n => !decide (n.id ∈ c.box.nodes) && !D.chHd c n.id).map
      (·.node (D.chLab0 c W)),
    (List.range (D.chOut c).length).map (D.bnd.length + ·), []⟩

/-- The box over its inside wires. -/
def chCp : Net :=
  ⟨(D.nodes.filter fun n => decide (n.id ∈ c.box.nodes)).map (·.node (D.chLabT c W)),
    (List.range (D.chIn c).length).map (W + c.links.length + ·), []⟩

/-- The outside nodes a link's first port meets, open on their slots. -/
def chHdN : Net :=
  ⟨(D.nodes.filter fun n => !decide (n.id ∈ c.box.nodes) && D.chHd c n.id).map
      (·.node (D.chLab0 c W)), [],
    (labelsOf ((D.nodes.filter fun n => !decide (n.id ∈ c.box.nodes) && D.chHd c n.id).map
      (·.node (D.chLab0 c W)))).filter (· < D.bnd.length)⟩

/-- The chain family of the template at `k` copies. -/
def chFam (k : ℕ) : Net :=
  famNet W (chU c W) (D.chTl c W) (D.chCp c W) (D.chHdN c W) (List.range D.bnd.length) k

/-- The expansion's leg labels: copy `j`'s at its template label raised `j` levels, the head
side's at its label raised `k` levels, the rest at its label. -/
def chLabK (k : ℕ) (p : Port) : ℕ :=
  if D.fresh ≤ p.node then
    sh W ((p.node - D.fresh) / c.box.nodes.length)
      (D.chLabT c W ⟨(c.box.nodes.mergeSort (· ≤ ·)).getD ((p.node - D.fresh) % c.box.nodes.length) 0,
        p.out, p.pos⟩)
  else if D.chHd c p.node then sh W k (D.chLab0 c W p) else D.chLab0 c W p

/-- Link `t`'s ends: its ports on box nodes at `W + t` and `2 W + t`, its first port's partner on
a head node and its second's on a tail node, both at `W + t`. -/
def chLinkOK (t : ℕ) : Bool :=
  match c.links[t]?, D.partner (c.links.getD t (⟨0, true, 0⟩, ⟨0, true, 0⟩)).1,
    D.partner (c.links.getD t (⟨0, true, 0⟩, ⟨0, true, 0⟩)).2 with
  | some l, some P, some Q =>
    decide (l.1.node ∈ c.box.nodes) && decide (l.2.node ∈ c.box.nodes) &&
      !decide (P.node ∈ c.box.nodes) && !decide (Q.node ∈ c.box.nodes) &&
      D.chHd c P.node && !D.chHd c Q.node && decide (P.node < D.fresh) &&
      decide (Q.node < D.fresh) &&
      decide (D.chLab0 c W P = W + t) && decide (D.chLab0 c W Q = W + t) &&
      decide (D.chLabT c W l.2 = W + t) && decide (D.chLabT c W l.1 = 2 * W + t)
  | _, _, _ => false

/-- A leg of a box node lies on an inside wire or is a link port; a leg of an outside node is a
slot, lies on an outside wire or is a link's end. -/
def chPortOK (q : Port) : Bool :=
  if q.node ∈ c.box.nodes then
    (D.chIn c).any (fun w => decide (w.1 = q ∨ w.2 = q)) ||
      c.links.any fun l => decide (l.1 = q ∨ l.2 = q)
  else
    decide (q ∈ D.bnd) || (D.chOut c).any (fun w => decide (w.1 = q ∨ w.2 = q)) ||
      c.links.any fun l => decide (D.partner l.1 = some q ∨ D.partner l.2 = some q)

/-- The template conditions of the bridge, each decided by the kernel. -/
structure ChainFamOK : Prop where
  wOK : D.bnd.length + (D.chOut c).length + c.links.length + (D.chIn c).length < W
  closed : ClosedChain D c
  nb : 0 < c.box.nodes.length
  bndL : ∀ i (hi : i < D.bnd.length), D.chLab0 c W D.bnd[i] = i
  outW : (D.chOut c).map (fun w => D.chLab0 c W w.1) =
    (List.range (D.chOut c).length).map (D.bnd.length + ·)
  outW2 : ∀ w ∈ D.chOut c, D.chLab0 c W w.2 = D.chLab0 c W w.1
  inW : (D.chIn c).map (fun w => D.chLabT c W w.1) =
    (List.range (D.chIn c).length).map (W + c.links.length + ·)
  inW2 : ∀ w ∈ D.chIn c, D.chLabT c W w.2 = D.chLabT c W w.1
  links : ∀ t < c.links.length, D.chLinkOK c W t = true
  bndIds : ∀ p ∈ D.bnd, p.node < D.fresh
  outIds : ∀ w ∈ D.chOut c, w.1.node < D.fresh ∧ w.2.node < D.fresh
  ports : ∀ n ∈ D.nodes, (∀ k < n.outs.length, D.chPortOK c ⟨n.id, true, k⟩ = true) ∧
    ∀ k < n.ins.length, D.chPortOK c ⟨n.id, false, k⟩ = true

section bridge

variable {D c W}

theorem le_foldr_max (l : List ℕ) (x : ℕ) (hx : x ∈ l) : x ≤ l.foldr max 0 := by
  induction l with
  | nil => simp at hx
  | cons a l ih =>
    rcases List.mem_cons.1 hx with rfl | hx
    · exact le_max_left _ _
    · exact (ih hx).trans (le_max_right _ _)

theorem chLt_fresh (n : DNode) (hn : n ∈ D.nodes) : n.id < D.fresh := by
  have := le_foldr_max (D.nodes.map (·.id)) n.id (List.mem_map.2 ⟨n, hn, rfl⟩)
  unfold fresh; omega

theorem ChainFamOK.W_gt (h : ChainFamOK D c W) :
    D.bnd.length + (D.chOut c).length + c.links.length + (D.chIn c).length < W := h.wOK

theorem ChainFamOK.link (h : ChainFamOK D c W) (t : ℕ) (ht : t < c.links.length) :
    ∃ P Q, D.partner c.links[t].1 = some P ∧ D.partner c.links[t].2 = some Q ∧
      c.links[t].1.node ∈ c.box.nodes ∧ c.links[t].2.node ∈ c.box.nodes ∧
      P.node ∉ c.box.nodes ∧ Q.node ∉ c.box.nodes ∧ D.chHd c P.node = true ∧
      D.chHd c Q.node = false ∧ P.node < D.fresh ∧ Q.node < D.fresh ∧
      D.chLab0 c W P = W + t ∧
      D.chLab0 c W Q = W + t ∧ D.chLabT c W c.links[t].2 = W + t ∧
      D.chLabT c W c.links[t].1 = 2 * W + t := by
  have hl := h.links t ht
  unfold chLinkOK at hl
  rw [List.getElem?_eq_getElem ht, List.getD_eq_getElem _ _ ht] at hl
  split at hl
  · next l P Q e1 e2 e3 =>
    simp only [Option.some.injEq] at e1
    subst e1
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not]
      at hl
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h12⟩, h9⟩, h10⟩, h11⟩ := hl
    exact ⟨P, Q, e2, e3, h1, h2, h3, h4, h5, h6, h7, h8, h12, h9, h10, h11⟩
  · simp at hl

/-- Copy `j`'s leg reads its template label raised `j` levels. -/
theorem ChainFamOK.labK_cp (h : ChainFamOK D c W) (k j : ℕ) (q : Port) (hq : q.node ∈ c.box.nodes) :
    D.chLabK c W k (D.cpPort c.box j q) = sh W j (D.chLabT c W q) := by
  set nb := c.box.nodes.length
  set srt := c.box.nodes.mergeSort (· ≤ ·)
  have hlen : srt.length = nb := (List.mergeSort_perm _ _).length_eq
  have hm : q.node ∈ srt := (List.mergeSort_perm _ _).mem_iff.2 hq
  have hix : srt.idxOf q.node < nb := hlen ▸ List.idxOf_lt_length_of_mem hm
  have hid : D.cpPort c.box j q = ⟨D.fresh + j * nb + srt.idxOf q.node, q.out, q.pos⟩ := rfl
  rw [hid]
  unfold chLabK
  simp only [show D.fresh ≤ D.fresh + j * nb + srt.idxOf q.node by omega, ite_true]
  have e1 : (D.fresh + j * nb + srt.idxOf q.node - D.fresh) / nb = j := by
    rw [show D.fresh + j * nb + srt.idxOf q.node - D.fresh = srt.idxOf q.node + j * nb by omega,
      Nat.add_mul_div_right _ _ h.nb, Nat.div_eq_of_lt hix, zero_add]
  have e2 : (D.fresh + j * nb + srt.idxOf q.node - D.fresh) % nb = srt.idxOf q.node := by
    rw [show D.fresh + j * nb + srt.idxOf q.node - D.fresh = srt.idxOf q.node + j * nb by omega,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hix]
  rw [e1, e2, List.getD_eq_getElem _ _ (hlen ▸ hix), List.getElem_idxOf]

/-- An outside leg reads its label, raised `k` levels on the head side. -/
theorem labK_out (k : ℕ) (p : Port) (hp : p.node < D.fresh) :
    D.chLabK c W k p = if D.chHd c p.node then sh W k (D.chLab0 c W p) else D.chLab0 c W p := by
  unfold chLabK
  simp [show ¬ D.fresh ≤ p.node by omega]

theorem labK_low (k : ℕ) (p : Port) (hp : p.node < D.fresh) (hl : D.chLab0 c W p < W) :
    D.chLabK c W k p = D.chLab0 c W p := by
  rw [labK_out k p hp]
  split_ifs <;> simp [sh_lt hl]

theorem ChainFamOK.out_lt (h : ChainFamOK D c W) (w : Port × Port) (hw : w ∈ D.chOut c) :
    D.chLab0 c W w.1 < W := by
  have hm : D.chLab0 c W w.1 ∈ (List.range (D.chOut c).length).map (D.bnd.length + ·) := by
    rw [← h.outW]; exact List.mem_map.2 ⟨w, hw, rfl⟩
  obtain ⟨i, hi, e⟩ := List.mem_map.1 hm
  have := h.W_gt
  rw [← e]; have := List.mem_range.1 hi; omega

theorem ChainFamOK.in_mem (h : ChainFamOK D c W) (w : Port × Port) (hw : w ∈ D.chIn c) :
    D.chLabT c W w.1 ∈ (List.range (D.chIn c).length).map (W + c.links.length + ·) := by
  rw [← h.inW]; exact List.mem_map.2 ⟨w, hw, rfl⟩

theorem chIn_node (w : Port × Port) (hw : w ∈ D.chIn c) : w.1.node ∈ c.box.nodes := by
  simpa [chIn] using (List.mem_filter.1 hw).2

theorem chOut_node (w : Port × Port) (hw : w ∈ D.chOut c) : w.1.node ∉ c.box.nodes := by
  simpa [chOut] using (List.mem_filter.1 hw).2

theorem chIn_base (w : Port × Port) (hw : w ∈ D.chIn c) : w ∈ (D.chainBase c).wires :=
  (List.mem_filter.1 hw).1

theorem chOut_base (w : Port × Port) (hw : w ∈ D.chOut c) : w ∈ (D.chainBase c).wires :=
  (List.mem_filter.1 hw).1

theorem base_split : ((D.chainBase c).wires.filter fun w => decide (w.1.node ∈ c.box.nodes)) ++
    ((D.chainBase c).wires.filter fun w => !decide (w.1.node ∈ c.box.nodes)) = D.chIn c ++ D.chOut c :=
  rfl

end bridge

end Diagram

end ArchytasZX
