/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.ChainBridge

/-!
# The expansion of a closed chain box is its chain family

`expand_chFam`: under `ChainFamOK`, for every environment whose labels share one dimension, every
offset above the slots and every `k`, the network of `expandChain D c k` equals `chFam D c k`.
The expansion labelled by `chLabK` (`equal_netBy`) has the chain family's nodes and summed labels
up to order: the summed labels are the outside wires', each inside wire's per copy and each
link's at the levels `0 .. k` (`intX`, `famInt`), all distinct (`famQ_nodup`).
-/

namespace ArchytasZX

namespace Diagram

variable {D : Diagram} {c : DChain} {W : ℕ}

theorem okPort_of_mem (X : Diagram) (p : Port)
    (h : p ∈ X.bnd ∨ ∃ w ∈ X.wires, w.1 = p ∨ w.2 = p) : X.okPort p = true := by
  unfold okPort locate
  cases hb : X.bnd.idxOf? p with
  | some i =>
    obtain ⟨hi, -, -⟩ := List.idxOf?_eq_some_iff.1 hb
    simp [hi]
  | none =>
    have hn : p ∉ X.bnd := by simpa using hb
    obtain ⟨w, hw, he⟩ := h.resolve_left hn
    cases hf : X.wires.findIdx? (fun w => decide (w.1 = p ∨ w.2 = p)) with
    | none =>
      exfalso
      rw [List.findIdx?_eq_none_iff] at hf
      exact absurd (hf w hw) (by simp [he])
    | some j =>
      obtain ⟨hj, -, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hf
      simp [hj]

theorem links_range (c : DChain) :
    c.links = (List.range c.links.length).map
      (fun t => c.links.getD t (⟨0, true, 0⟩, ⟨0, true, 0⟩)) := by
  refine List.ext_getElem (by simp) fun i h1 h2 => ?_
  simp [List.getElem?_eq_getElem h1]

theorem sh_lvl (W j t : ℕ) (ht : t < W) : sh W j (W + t) = W + t + j * W := sh_ge (by omega)

theorem flatMap_single {α β : Type} (l : List α) (f : α → β) :
    (l.flatMap fun a => [f a]) = l.map f := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [ih]

theorem getElem_eq_of {α : Type} {l l' : List α} (e : l = l') (i : ℕ) (hi : i < l.length) :
    l[i] = l'[i]'(e ▸ hi) := by
  subst e; rfl

/-- The link wires of link `t`: the second partner to copy `0`, copy `j` to copy `j + 1`, the last
copy to the first partner. -/
theorem ChainFamOK.lw_mem (h : ChainFamOK D c W) (k t : ℕ) (ht : t < c.links.length) :
    ∃ P Q, D.partner c.links[t].1 = some P ∧ D.partner c.links[t].2 = some Q ∧
      (k = 0 → (Q, P) ∈ D.linkWires c k c.links[t]) ∧
      (0 < k → (Q, D.cpPort c.box 0 c.links[t].2) ∈ D.linkWires c k c.links[t]) ∧
      (0 < k → (D.cpPort c.box (k - 1) c.links[t].1, P) ∈ D.linkWires c k c.links[t]) ∧
      ∀ j, j + 1 < k → (D.cpPort c.box j c.links[t].1, D.cpPort c.box (j + 1) c.links[t].2) ∈
        D.linkWires c k c.links[t] := by
  obtain ⟨P, Q, e1, e2, -⟩ := h.link t ht
  refine ⟨P, Q, e1, e2, fun hk => ?_, fun hk => ?_, fun hk => ?_, fun j hj => ?_⟩
  · subst hk
    simp [linkWires, e1, e2]
  · simp only [linkWires, e1, e2, if_neg (show k ≠ 0 by omega)]
    exact List.mem_cons_self
  · simp only [linkWires, e1, e2, if_neg (show k ≠ 0 by omega)]
    exact List.mem_cons_of_mem _ (List.mem_append_right _ (List.mem_singleton_self _))
  · simp only [linkWires, e1, e2, if_neg (show k ≠ 0 by omega)]
    exact List.mem_cons_of_mem _ (List.mem_append_left _
      (List.mem_map.2 ⟨j, List.mem_range.2 (by omega), rfl⟩))

/-- Link `t`'s wires carry its label at the levels `0 .. k`, both ends alike. -/
theorem ChainFamOK.link_labels (h : ChainFamOK D c W) (k t : ℕ) (ht : t < c.links.length) :
    (D.linkWires c k c.links[t]).map (fun w => D.chLabK c W k w.1) =
        (List.range (k + 1)).map (fun j => sh W j (W + t)) ∧
      ∀ w ∈ D.linkWires c k c.links[t], D.chLabK c W k w.1 = D.chLabK c W k w.2 := by
  obtain ⟨P, Q, e1, e2, hl1, hl2, hP, hQ, hdP, hdQ, hPf, hQf, l0P, l0Q, lT2, lT1⟩ := h.link t ht
  have hW := h.W_gt
  have hk : t < W := by omega
  have kP : D.chLabK c W k P = sh W k (W + t) := by
    rw [labK_out k P hPf, hdP, l0P]; rfl
  have kQ : D.chLabK c W k Q = W + t := by
    rw [labK_out k Q hQf, hdQ, l0Q]; rfl
  have k1 : ∀ j, D.chLabK c W k (D.cpPort c.box j c.links[t].1) =
      sh W (j + 1) (W + t) := fun j => by
    rw [h.labK_cp k j _ hl1, lT1, sh_ge (show W ≤ 2 * W + t by omega), sh_ge (show W ≤ W + t by omega)]; ring
  have k2 : ∀ j, D.chLabK c W k (D.cpPort c.box j c.links[t].2) = sh W j (W + t) :=
    fun j => by rw [h.labK_cp k j _ hl2, lT2]
  have hlw : D.linkWires c k c.links[t] =
      if k = 0 then [(Q, P)]
      else (Q, D.cpPort c.box 0 c.links[t].2) ::
        ((List.range (k - 1)).map fun j =>
          (D.cpPort c.box j c.links[t].1, D.cpPort c.box (j + 1) c.links[t].2)) ++
        [(D.cpPort c.box (k - 1) c.links[t].1, P)] := by
    simp [linkWires, e1, e2]
  rw [hlw]
  rcases k with _ | k
  · rw [if_pos rfl]
    refine ⟨by simp [kQ, sh_zero], fun w hw => ?_⟩
    rw [List.mem_singleton] at hw
    subst hw
    rw [kQ, kP, sh_zero]
  · rw [if_neg (by omega), Nat.add_sub_cancel]
    constructor
    · rw [List.range_succ, List.range_succ_eq_map]
      simp only [List.map_cons, List.map_append, List.map_map, List.map_nil, List.cons_append,
        Function.comp_def, kQ, k1, sh_zero, Nat.succ_eq_add_one]
    · intro w hw
      simp only [List.mem_cons, List.mem_append, List.mem_map, List.mem_range,
        List.mem_singleton] at hw
      rcases hw with (rfl | ⟨j, -, rfl⟩) | rfl | hn
      · rw [kQ, k2, sh_zero]
      · rw [k1, k2]
      · rw [k1, kP]
      · exact absurd hn List.not_mem_nil

/-- The summed labels of the chain family in the expansion's order. -/
def famQ (D : Diagram) (c : DChain) (W k : ℕ) : List ℕ :=
  (D.chTl c W).int ++ (List.range k).flatMap (fun j => (D.chCp c W).int.map (sh W j)) ++
    (List.range (k + 1)).flatMap (fun j => (chU c W).map (sh W j))

theorem ChainFamOK.intX (h : ChainFamOK D c W) (k : ℕ) :
    ((D.expandChain c k).wires.map fun w => D.chLabK c W k w.1).Perm (famQ D c W k) := by
  rw [h.closed.wires_expand k, List.map_append]
  unfold famQ
  refine List.Perm.append ?_ ?_
  · have hsp : (D.chIn c ++ D.chOut c).Perm (D.chainBase c).wires := List.filter_append_perm _ _
    have hp := (hsp.flatMap_right (D.chainWire c k)).map (fun w => D.chLabK c W k w.1)
    refine hp.symm.trans ?_
    rw [List.flatMap_append, List.map_append]
    refine List.perm_append_comm.trans (List.Perm.append ?_ ?_)
    · have e1 : ((D.chOut c).flatMap (D.chainWire c k)).map (fun w => D.chLabK c W k w.1) =
          (D.chOut c).map fun w => D.chLab0 c W w.1 := by
        rw [List.map_flatMap, ← flatMap_single]
        refine List.flatMap_congr fun w hw => ?_
        simp only [chainWire, chOut_node w hw, if_false, List.map_cons, List.map_nil]
        rw [labK_low k w.1 (h.outIds w hw).1 (h.out_lt w hw)]
      rw [e1, h.outW]
      exact List.Perm.refl _
    · rw [List.map_flatMap]
      have e : ∀ w ∈ D.chIn c, (D.chainWire c k w).map (fun w => D.chLabK c W k w.1) =
          (List.range k).map fun j => sh W j (D.chLabT c W w.1) := fun w hw => by
        simp only [chainWire, chIn_node w hw, ite_true, List.map_map, Function.comp_def]
        exact List.map_congr_left fun j _ => h.labK_cp k j w.1 (chIn_node w hw)
      rw [List.flatMap_congr e]
      refine (flatMap_swap (D.chIn c) (List.range k)
        (fun w j => sh W j (D.chLabT c W w.1))).trans ?_
      refine List.Perm.of_eq (List.flatMap_congr fun j _ => ?_)
      show _ = ((List.range (D.chIn c).length).map (W + c.links.length + ·)).map _
      rw [← h.inW, List.map_map]
      rfl
  · rw [List.map_flatMap]
    conv_lhs => rw [links_range c]
    rw [List.flatMap_map]
    have e : ∀ t ∈ List.range c.links.length,
        (D.linkWires c k (c.links.getD t (⟨0, true, 0⟩, ⟨0, true, 0⟩))).map
          (fun w => D.chLabK c W k w.1) =
        (List.range (k + 1)).map (fun j => sh W j (W + t)) := fun t ht => by
      have ht' := List.mem_range.1 ht
      rw [List.getD_eq_getElem _ _ ht']
      exact (h.link_labels k t ht').1
    rw [List.flatMap_congr e]
    refine (flatMap_swap (List.range c.links.length) (List.range (k + 1))
      (fun t j => sh W j (W + t))).trans (List.Perm.of_eq ?_)
    refine List.flatMap_congr fun j _ => ?_
    simp only [chU, List.map_map, Function.comp_def]

theorem flatMap_append_perm {α β : Type} (l : List α) (f g : α → List β) :
    (l.flatMap fun a => f a ++ g a).Perm (l.flatMap f ++ l.flatMap g) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.flatMap_cons, List.append_assoc]
    refine List.Perm.append_left _ ?_
    refine (ih.append_left _).trans ?_
    rw [← List.append_assoc, ← List.append_assoc]
    exact List.Perm.append_right _ List.perm_append_comm

theorem ChainFamOK.u_ge (h : ChainFamOK D c W) (x : ℕ) (hx : x ∈ chU c W) : W ≤ x := by
  obtain ⟨t, -, rfl⟩ := List.mem_map.1 hx; omega

theorem ChainFamOK.famInt (h : ChainFamOK D c W) (k : ℕ) : (famQ D c W k).Perm (D.chFam c W k).int := by
  have hW := h.W_gt
  have hf : (chU c W).filter (fun x => decide (x ∉ List.range D.bnd.length)) = chU c W :=
    List.filter_eq_self.2 fun x hx => by
      have := h.u_ge x hx
      simp only [List.mem_range, decide_eq_true_eq, not_lt]; omega
  have hsucc : (List.range (k + 1)).flatMap (fun j => (chU c W).map (sh W j)) =
      chU c W ++ (List.range k).flatMap (fun j => (chU c W).map (sh W (j + 1))) := by
    rw [List.range_succ_eq_map, List.flatMap_cons, List.flatMap_map]
    congr 1
    · exact (List.map_congr_left fun x _ => sh_zero _ x).trans (List.map_id _)
    all_goals rfl
  simp only [chFam, famNet, hf]
  unfold famQ
  rw [hsucc, show (D.chHdN c W).int = [] from rfl, List.append_nil]
  have := flatMap_append_perm (List.range k) (fun j => (D.chCp c W).int.map (sh W j))
    (fun j => (chU c W).map (sh W (j + 1)))
  refine List.Perm.trans ?_ ((this.append_left _).symm)
  rw [List.perm_iff_count]
  intro a
  simp only [List.count_append, List.map_id]
  omega

theorem lev_ne {W a b j m : ℕ} (ha : W ≤ a ∧ a < 2 * W) (hb : W ≤ b ∧ b < 2 * W)
    (h : a + j * W = b + m * W) (hab : a ≠ b) : False := hab (lev_eq ha hb h).2

theorem ChainFamOK.famQ_nodup (h : ChainFamOK D c W) (k : ℕ) : (famQ D c W k).Nodup := by
  have hW := h.W_gt
  set W := W
  set nL := c.links.length
  set nI := (D.chIn c).length
  set m := D.bnd.length
  have hcp : ∀ j, ∀ x ∈ (D.chCp c W).int.map (sh W j), ∃ i < nI, x = (W + nL + i) + j * W := by
    intro j x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hy
    exact ⟨i, List.mem_range.1 hi, sh_ge (by omega)⟩
  have hu : ∀ j, ∀ x ∈ (chU c W).map (sh W j), ∃ t < nL, x = (W + t) + j * W := by
    intro j x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hy
    exact ⟨t, List.mem_range.1 ht, sh_ge (by omega)⟩
  have hsh : ∀ j, Function.Injective (sh W j) := sh_inj W
  have hcpN : ∀ j, ((D.chCp c W).int.map (sh W j)).Nodup := fun j =>
    (List.nodup_range.map fun a b (e : W + nL + a = W + nL + b) => by omega).map (hsh j)
  have huN : ∀ j, ((chU c W).map (sh W j)).Nodup := fun j =>
    (List.nodup_range.map fun a b (e : W + a = W + b) => by omega).map (hsh j)
  have hA : (List.range k).flatMap (fun j => (D.chCp c W).int.map (sh W j)) |>.Nodup := by
    refine List.nodup_flatMap.2 ⟨fun j _ => hcpN j, List.nodup_range.pairwise_of_forall_ne
      fun a _ b _ hab => List.disjoint_left.2 fun x hx hx' => ?_⟩
    obtain ⟨i, hi, rfl⟩ := hcp a x hx
    obtain ⟨i', hi', e⟩ := hcp b _ hx'
    exact hab (lev_eq (W := W) ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ e).1
  have hB : (List.range (k + 1)).flatMap (fun j => (chU c W).map (sh W j)) |>.Nodup := by
    refine List.nodup_flatMap.2 ⟨fun j _ => huN j, List.nodup_range.pairwise_of_forall_ne
      fun a _ b _ hab => List.disjoint_left.2 fun x hx hx' => ?_⟩
    obtain ⟨t, ht, rfl⟩ := hu a x hx
    obtain ⟨t', ht', e⟩ := hu b _ hx'
    exact hab (lev_eq (W := W) ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ e).1
  have hT : (D.chTl c W).int.Nodup :=
    List.nodup_range.map fun a b (e : m + a = m + b) => by omega
  unfold famQ
  refine List.nodup_append.2 ⟨List.nodup_append.2 ⟨hT, hA, ?_⟩, hB, ?_⟩
  · intro x hx y hy e
    subst e
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
    obtain ⟨j, -, hj⟩ := List.mem_flatMap.1 hy
    obtain ⟨i', -, e⟩ := hcp j _ hj
    have := List.mem_range.1 hi
    omega
  · intro x hx y hy e
    subst e
    obtain ⟨j', -, hj'⟩ := List.mem_flatMap.1 hy
    obtain ⟨t, ht, e'⟩ := hu j' _ hj'
    rcases List.mem_append.1 hx with hx | hx
    · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
      have := List.mem_range.1 hi
      omega
    · obtain ⟨j, -, hj⟩ := List.mem_flatMap.1 hx
      obtain ⟨i, hi, e⟩ := hcp j _ hj
      rw [e] at e'
      exact lev_ne (W := W) ⟨by omega, by omega⟩ ⟨by omega, by omega⟩ e' (by omega)

theorem ChainFamOK.famQ_ge (h : ChainFamOK D c W) (k : ℕ) (x : ℕ) (hx : x ∈ famQ D c W k) :
    D.bnd.length ≤ x := by
  have hW := h.W_gt
  unfold famQ at hx
  simp only [chTl, chCp, chU, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range] at hx
  rcases hx with (⟨i, -, rfl⟩ | ⟨j, -, y, ⟨i, -, rfl⟩, rfl⟩) | ⟨j, -, y, ⟨t, -, rfl⟩, rfl⟩
  · omega
  · rw [sh_ge (show W ≤ W + c.links.length + i by omega)]; omega
  · rw [sh_ge (show W ≤ W + t by omega)]; omega

theorem chNode_congr (n : DNode) (f g : Port → ℕ)
    (ho : ∀ i < n.outs.length, f ⟨n.id, true, i⟩ = g ⟨n.id, true, i⟩)
    (hi : ∀ i < n.ins.length, f ⟨n.id, false, i⟩ = g ⟨n.id, false, i⟩) : n.node f = n.node g := by
  simp only [DNode.node]
  congr 1
  · exact List.map_congr_left fun i h => ho i (List.mem_range.1 h)
  · exact List.map_congr_left fun i h => hi i (List.mem_range.1 h)

theorem perm_split3 {α : Type} (l : List α) (p q : α → Bool) :
    l.Perm (l.filter p ++ l.filter (fun a => !p a && q a) ++ l.filter (fun a => !p a && !q a)) := by
  have h1 : l.Perm (l.filter p ++ l.filter (fun a => !p a)) := (List.filter_append_perm p l).symm
  have h2 : (l.filter fun a => !p a).Perm ((l.filter fun a => !p a).filter q ++
      (l.filter fun a => !p a).filter (fun a => !q a)) := (List.filter_append_perm q _).symm
  have e1 : (l.filter fun a => !p a).filter q = l.filter fun a => !p a && q a := by
    rw [List.filter_filter]
    exact List.filter_congr fun a _ => by cases hp : p a <;> cases hq : q a <;> simp [hp, hq]
  have e2 : (l.filter fun a => !p a).filter (fun a => !q a) = l.filter fun a => !p a && !q a := by
    rw [List.filter_filter]
    exact List.filter_congr fun a _ => by cases hp : p a <;> cases hq : q a <;> simp [hp, hq]
  rw [e1, e2] at h2
  rw [List.append_assoc]
  exact h1.trans (h2.append_left _)

theorem ChainFamOK.nodesX (h : ChainFamOK D c W) (k : ℕ) :
    ((D.expandChain c k).nodes.map (·.node (D.chLabK c W k))).Perm (D.chFam c W k).nodes := by
  rw [h.closed.nodes_expand k, List.map_flatMap]
  set G : DNode → List Node := fun n =>
    ((if n.id ∈ c.box.nodes then (List.range k).map fun j => D.cpNodeAt c j n else [n]).map
      (·.node (D.chLabK c W k)))
  set lB := D.nodes.filter fun n => decide (n.id ∈ c.box.nodes)
  set lH := D.nodes.filter fun n => !decide (n.id ∈ c.box.nodes) && D.chHd c n.id
  set lT := D.nodes.filter fun n => !decide (n.id ∈ c.box.nodes) && !D.chHd c n.id
  have hsp : D.nodes.Perm (lB ++ lH ++ lT) :=
    perm_split3 D.nodes (fun n => decide (n.id ∈ c.box.nodes)) (fun n => D.chHd c n.id)
  refine (hsp.flatMap_right G).trans ?_
  rw [List.flatMap_append, List.flatMap_append]
  have eB : lB.flatMap G = lB.flatMap fun n =>
      (List.range k).map fun j => (n.node (D.chLabT c W)).rename (sh W j) := by
    refine List.flatMap_congr fun n hn => ?_
    have hb : n.id ∈ c.box.nodes := by simpa using (List.mem_filter.1 hn).2
    simp only [G, hb, if_true, List.map_map]
    refine List.map_congr_left fun j _ => ?_
    simp only [Function.comp, node_rename, DNode.node, cpNodeAt]
    congr 1
    · simp only [List.map_map, Function.comp_def]
      exact List.map_congr_left fun i _ => h.labK_cp k j ⟨n.id, true, i⟩ hb
    · simp only [List.map_map, Function.comp_def]
      exact List.map_congr_left fun i _ => h.labK_cp k j ⟨n.id, false, i⟩ hb
  have eO : ∀ (l : List DNode), (∀ n ∈ l, n ∈ D.nodes ∧ n.id ∉ c.box.nodes) →
      l.flatMap G = l.map (·.node (D.chLabK c W k)) := by
    intro l hl
    rw [← flatMap_single]
    refine List.flatMap_congr fun n hn => ?_
    simp only [G, (hl n hn).2, if_false, List.map_cons, List.map_nil]
  have eH : lH.flatMap G = ((D.chHdN c W).mapL (sh W k)).nodes := by
    rw [eO lH fun n hn => ⟨(List.mem_filter.1 hn).1, by
      simpa using (Eq.mp (Bool.and_eq_true _ _) (List.mem_filter.1 hn).2).1⟩]
    simp only [chHdN, Net.mapL, List.map_map]
    refine List.map_congr_left fun n hn => ?_
    obtain ⟨hn, hq⟩ := List.mem_filter.1 hn
    have hd : D.chHd c n.id = true := (Eq.mp (Bool.and_eq_true _ _) hq).2
    simp only [Function.comp, node_rename]
    refine chNode_congr _ _ _ (fun i _ => ?_) (fun i _ => ?_) <;>
      rw [labK_out k _ (chLt_fresh n hn)] <;> simp [hd]
  have eT : lT.flatMap G = (D.chTl c W).nodes := by
    rw [eO lT fun n hn => ⟨(List.mem_filter.1 hn).1, by
      simpa using (Eq.mp (Bool.and_eq_true _ _) (List.mem_filter.1 hn).2).1⟩]
    simp only [chTl]
    refine List.map_congr_left fun n hn => ?_
    obtain ⟨hn, hq⟩ := List.mem_filter.1 hn
    have hd : D.chHd c n.id = false := by
      simpa using (Eq.mp (Bool.and_eq_true _ _) hq).2
    refine chNode_congr _ _ _ (fun i _ => ?_) (fun i _ => ?_) <;>
      rw [labK_out k _ (chLt_fresh n hn)] <;> simp [hd]
  rw [eB, eH, eT]
  have hS := flatMap_swap lB (List.range k)
    (fun n j => (n.node (D.chLabT c W)).rename (sh W j))
  have ecp : (List.range k).flatMap (fun j => lB.map fun n =>
        (n.node (D.chLabT c W)).rename (sh W j)) =
      (List.range k).flatMap fun j => ((D.chCp c W).mapL (sh W j)).nodes := by
    refine List.flatMap_congr fun j _ => ?_
    simp [chCp, Net.mapL, List.map_map, Function.comp_def, lB]
  rw [ecp] at hS
  refine ((hS.append_right _).append_right _).trans ?_
  simp only [chFam, famNet]
  rw [List.perm_iff_count]
  intro a
  simp only [List.count_append]
  omega

theorem ChainFamOK.hwX (h : ChainFamOK D c W) (k : ℕ) (w : Port × Port)
    (hw : w ∈ (D.expandChain c k).wires) : D.chLabK c W k w.1 = D.chLabK c W k w.2 := by
  rw [h.closed.wires_expand k, List.mem_append, List.mem_flatMap, List.mem_flatMap] at hw
  rcases hw with ⟨w0, hw0, hw⟩ | ⟨l, hl, hw⟩
  · unfold chainWire at hw
    split_ifs at hw with hb
    · obtain ⟨j, -, rfl⟩ := List.mem_map.1 hw
      have hin : w0 ∈ D.chIn c := List.mem_filter.2 ⟨hw0, by simpa using hb⟩
      show D.chLabK c W k (D.cpPort c.box j w0.1) = D.chLabK c W k (D.cpPort c.box j w0.2)
      rw [h.labK_cp k j _ hb, h.labK_cp k j _ ((h.closed.side w0 hw0).1 hb), h.inW2 w0 hin]
    · rw [List.mem_singleton] at hw
      subst w
      have hout : w0 ∈ D.chOut c := List.mem_filter.2 ⟨hw0, by simpa using hb⟩
      have hl1 := h.out_lt w0 hout
      have hl2 : D.chLab0 c W w0.2 < W := by rw [h.outW2 w0 hout]; exact hl1
      rw [labK_low k _ (h.outIds w0 hout).1 hl1, labK_low k _ (h.outIds w0 hout).2 hl2,
        h.outW2 w0 hout]
  · obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hl
    exact (h.link_labels k t ht).2 w hw

theorem ChainFamOK.portsX (h : ChainFamOK D c W) (k : ℕ) : (D.expandChain c k).portsOK := by
  set X := D.expandChain c k
  have wIn : ∀ w ∈ D.chIn c, ∀ j < k, (D.cpPort c.box j w.1, D.cpPort c.box j w.2) ∈ X.wires :=
    fun w hw j hj => by
      rw [h.closed.wires_expand k]
      refine List.mem_append_left _ (List.mem_flatMap.2 ⟨w, chIn_base w hw, ?_⟩)
      simp only [chainWire, chIn_node w hw, if_true]
      exact List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
  have wOut : ∀ w ∈ D.chOut c, w ∈ X.wires := fun w hw => by
    rw [h.closed.wires_expand k]
    refine List.mem_append_left _ (List.mem_flatMap.2 ⟨w, chOut_base w hw, ?_⟩)
    simp [chainWire, chOut_node w hw]
  have wLk : ∀ t (ht : t < c.links.length), ∀ w ∈ D.linkWires c k c.links[t], w ∈ X.wires :=
    fun t ht w hw => by
      rw [h.closed.wires_expand k]
      exact List.mem_append_right _ (List.mem_flatMap.2 ⟨_, List.getElem_mem ht, hw⟩)
  have key : ∀ q : Port, D.chPortOK c q = true →
      (q.node ∈ c.box.nodes → ∀ j < k, X.okPort (D.cpPort c.box j q) = true) ∧
      (q.node ∉ c.box.nodes → X.okPort q = true) := by
    intro q hq
    unfold chPortOK at hq
    split_ifs at hq with hb
    · refine ⟨fun _ j hj => okPort_of_mem X _ (Or.inr ?_), fun hn => absurd hb hn⟩
      simp only [Bool.or_eq_true, List.any_eq_true, decide_eq_true_eq] at hq
      rcases hq with ⟨w, hw, e⟩ | ⟨l, hl, e⟩
      · refine ⟨_, wIn w hw j hj, ?_⟩
        rcases e with rfl | rfl
        · exact Or.inl rfl
        · exact Or.inr rfl
      · obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hl
        obtain ⟨P, Q, -, -, -, m0, mL, mM⟩ := h.lw_mem k t ht
        rcases e with e | e
        · by_cases hj' : j + 1 < k
          · exact ⟨_, wLk t ht _ (mM j hj'), Or.inl (by rw [e])⟩
          · have : j = k - 1 := by omega
            subst this
            exact ⟨_, wLk t ht _ (mL (by omega)), Or.inl (by rw [e])⟩
        · rcases j with _ | j
          · exact ⟨_, wLk t ht _ (m0 (by omega)), Or.inr (by rw [e])⟩
          · exact ⟨_, wLk t ht _ (mM j (by omega)), Or.inr (by rw [e])⟩
    · refine ⟨fun hn => absurd hn hb, fun _ => okPort_of_mem X _ ?_⟩
      simp only [Bool.or_eq_true, List.any_eq_true, decide_eq_true_eq] at hq
      rcases hq with (hq | ⟨w, hw, e⟩) | ⟨l, hl, e⟩
      · left; rw [h.closed.bnd_expand k]; exact hq
      · exact Or.inr ⟨w, wOut w hw, e⟩
      · right
        obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hl
        obtain ⟨P, Q, e1, e2, mz, m0, mL, -⟩ := h.lw_mem k t ht
        rcases e with e | e
        · rw [e1, Option.some.injEq] at e
          subst e
          rcases Nat.eq_zero_or_pos k with hk | hk
          · exact ⟨_, wLk t ht _ (mz hk), Or.inr rfl⟩
          · exact ⟨_, wLk t ht _ (mL hk), Or.inr rfl⟩
        · rw [e2, Option.some.injEq] at e
          subst e
          rcases Nat.eq_zero_or_pos k with hk | hk
          · exact ⟨_, wLk t ht _ (mz hk), Or.inl rfl⟩
          · exact ⟨_, wLk t ht _ (m0 hk), Or.inl rfl⟩
  intro n hn
  rw [h.closed.nodes_expand k, List.mem_flatMap] at hn
  obtain ⟨n0, hn0, hn⟩ := hn
  obtain ⟨p1, p2⟩ := h.ports n0 hn0
  split_ifs at hn with hb
  · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hn
    exact ⟨fun i hi => (key _ (p1 i hi)).1 hb j (List.mem_range.1 hj),
      fun i hi => (key _ (p2 i hi)).1 hb j (List.mem_range.1 hj)⟩
  · rw [List.mem_singleton] at hn
    subst hn
    exact ⟨fun i hi => (key _ (p1 i hi)).2 hb, fun i hi => (key _ (p2 i hi)).2 hb⟩

/-- The network of a closed chain box expanded at `k`, at any offset above the slots, is its
chain family at `k`, for every environment whose labels share one dimension. -/
theorem expand_chFam (h : ChainFamOK D c W) (E : Env) (hD : ∀ x y, labDim E x = labDim E y) (o : ℕ)
    (ho : D.bnd.length ≤ o) (k : ℕ) :
    Equal E 1 ((D.expandChain c k).toNet o) 1 (D.chFam c W k) := by
  set X := D.expandChain c k
  have hXb : X.bnd = D.bnd := h.closed.bnd_expand k
  have hperm := h.intX k
  have hnd : (X.wires.map fun w => D.chLabK c W k w.1).Nodup :=
    hperm.nodup_iff.2 (h.famQ_nodup k)
  have hb : ∀ i (hi : i < X.bnd.length), D.chLabK c W k X.bnd[i] = i := by
    intro i hi
    have hi' : i < D.bnd.length := hXb ▸ hi
    rw [getElem_eq_of hXb i hi]
    have hl := h.bndL i hi'
    have hW := h.W_gt
    rw [labK_low k _ (h.bndIds _ (List.getElem_mem _)) (by rw [hl]; omega), hl]
  have e1 := equal_netBy X E o (D.chLabK c W k) (by rw [hXb]; exact ho) hb (h.hwX k) (h.portsX k)
    hnd (fun w hw => by
      rw [hXb]
      exact h.famQ_ge k _ (hperm.mem_iff.1 (List.mem_map.2 ⟨w, hw, rfl⟩)))
    (fun j _ => hD _ _)
  refine e1.trans (Equal.reorder E (Equal.refl E 1 _) (h.nodesX k) (hperm.trans (h.famInt k)) ?_)
  show (List.range X.bnd.length).Perm (List.range D.bnd.length)
  exact List.Perm.of_eq (by rw [hXb])

end Diagram

namespace ChainFamExample

open Diagram

/-- One Z(1 -> 1) in a chain box, its output linked to its input, between the identities `0` (the
chain's input end) and `2` (its output end). -/
def D : Diagram :=
  { nodes := [⟨0, .I, 0, 0, [0], [0]⟩, ⟨1, .Z, 0, 0, [0], [0]⟩, ⟨2, .I, 0, 0, [0], [0]⟩],
    wires := [(⟨0, true, 0⟩, ⟨1, false, 0⟩), (⟨1, true, 0⟩, ⟨2, false, 0⟩)],
    outs := [⟨2, true, 0⟩],
    ins := [⟨0, false, 0⟩],
    boxes := [⟨0, 0, none, [1], []⟩] }

/-- The chain: output `0` of a copy feeds input `0` of the next. -/
def c : DChain := ⟨⟨0, 0, none, [1], []⟩, [(⟨1, true, 0⟩, ⟨1, false, 0⟩)]⟩

theorem ok : ChainFamOK D c 8 :=
  ⟨by decide, ⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide⟩

/-- The chain expanded at every `k` is its chain family. -/
theorem every (E : Env) (hD : ∀ x y, labDim E x = labDim E y) (k : ℕ) :
    Equal E 1 ((D.expandChain c k).toNet 2) 1 (D.chFam c 8 k) :=
  expand_chFam ok E hD 2 (by decide) k

end ChainFamExample

end ArchytasZX
