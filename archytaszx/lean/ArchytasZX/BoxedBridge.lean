/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.FamK
import ArchytasZX.DiagramBridge

/-!
# Boxed diagrams as bundled networks, checked in linear time

`bridgeB` checks `D.toNetB off = N` and `D.FamOK F off` through `bridgeD` on the plain network
`Np` (`D.toNet off`): box membership is read from bit masks `MS` (each box's node mask, keyed
by box id) and `MN` (the node ids), box chains are compared by box ids, the other end of a wire
leg is read from the packed wires `WB`, each leg's dimension index from the table `T`, and each
label's bundled dimension index in `F.labs` from the table `TB` of `b'`-bit entries.
`bridgeB_ok` reads both from `bridgeB`.
-/

namespace ArchytasZX
namespace BBridge

open DBridge

/-! ## Masks, walks and chains -/

/-- The bit mask of a list of numbers. -/
def maskOf : List ℕ → ℕ
  | [] => 0
  | x :: l => 2 ^ x ||| maskOf l

theorem testBit_maskOf (x : ℕ) : ∀ l : List ℕ, (maskOf l).testBit x = decide (x ∈ l)
  | [] => by simp [maskOf]
  | y :: l => by
    rw [maskOf, Nat.testBit_lor, Nat.testBit_two_pow, testBit_maskOf x l]
    by_cases h : x = y
    · subst h; simp
    · simp [h, Ne.symm h]

/-- Box `y`'s mask in the table `MS` of box ids and masks. -/
def bmask (MS : List (ℕ × ℕ)) (y : DBox) : ℕ := (MS.lookup y.id).getD 0

/-- Every box's mask in `MS` is the mask of its nodes. -/
def masksOK (MS : List (ℕ × ℕ)) (D : Diagram) : Bool :=
  D.boxes.all fun y => MS.lookup y.id == some (maskOf y.nodes)

theorem bmask_ok (MS : List (ℕ × ℕ)) (D : Diagram) (h : masksOK MS D = true) (y : DBox)
    (hy : y ∈ D.boxes) (x : ℕ) : (bmask MS y).testBit x = decide (x ∈ y.nodes) := by
  have := List.all_eq_true.1 h y hy
  simp only [beq_iff_eq] at this
  rw [bmask, this, Option.getD_some, testBit_maskOf]

theorem find?_congr' {α : Type*} (f g : α → Bool) :
    ∀ l : List α, (∀ y ∈ l, f y = g y) → l.find? f = l.find? g
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.find?_cons, h a List.mem_cons_self]
    rw [find?_congr' f g l fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem walk_congr (D : Diagram) (p q : DBox → Bool) (h : ∀ y ∈ D.boxes, p y = q y) :
    ∀ fuel start, D.walk p fuel start = D.walk q fuel start
  | 0, _ => rfl
  | f + 1, s => by
    simp only [Diagram.walk]
    rw [find?_congr' (fun b => p b && b.parent == s) (fun b => q b && b.parent == s) D.boxes
      fun y hy => by rw [h y hy]]
    cases D.boxes.find? (fun b => q b && b.parent == s) with
    | none => rfl
    | some b => exact congrArg (b :: ·) (walk_congr D p q h f (some b.id))

theorem walk_sub (D : Diagram) (p : DBox → Bool) :
    ∀ fuel start, ∀ y ∈ D.walk p fuel start, y ∈ D.boxes ∧ p y = true
  | 0, _, y, hy => by simp [Diagram.walk] at hy
  | f + 1, s, y, hy => by
    simp only [Diagram.walk] at hy
    split at hy
    · simp at hy
    · next b hb =>
      have hm := List.mem_of_find?_eq_some hb
      have hp := List.find?_some hb
      simp only [Bool.and_eq_true] at hp
      rcases List.mem_cons.1 hy with rfl | hy
      · exact ⟨hm, hp.1⟩
      · exact walk_sub D p f (some b.id) y hy

/-- Each box's parent is the previous box's id, the first's `start`. -/
def parOK : Option ℕ → List DBox → Prop
  | _, [] => True
  | s, b :: L => b.parent = s ∧ parOK (some b.id) L

theorem walk_parOK (D : Diagram) (p : DBox → Bool) :
    ∀ fuel start, parOK start (D.walk p fuel start)
  | 0, _ => trivial
  | f + 1, s => by
    simp only [Diagram.walk]
    split
    · trivial
    · next b hb =>
      have hp := List.find?_some hb
      simp only [Bool.and_eq_true, beq_iff_eq] at hp
      exact And.intro hp.2 (walk_parOK D p f (some b.id))

theorem parOK_get : ∀ (start : Option ℕ) (L : List DBox), parOK start L →
    ∀ (i : ℕ) (h : i < L.length),
      L[i].parent = (start :: L.map (fun b => some b.id))[i]'(by simp; omega)
  | _, [], _, i, h => absurd h (by simp)
  | s, b :: L, hL, 0, _ => (show b.parent = s ∧ parOK (some b.id) L from hL).1
  | s, b :: L, hL, i + 1, h => by
    simp only [List.getElem_cons_succ, List.map_cons]
    exact parOK_get (some b.id) L (show b.parent = s ∧ parOK (some b.id) L from hL).2 i
      (by simpa using h)

theorem walk_parent (D : Diagram) (p : DBox → Bool) (fuel : ℕ) (start : Option ℕ) :
    ∀ (i : ℕ) (h : i < (D.walk p fuel start).length),
      (D.walk p fuel start)[i].parent =
        (start :: (D.walk p fuel start).map (fun b => some b.id))[i]'(by simp; omega) :=
  parOK_get start _ (walk_parOK D p fuel start)

theorem eq_of_id {D : Diagram} (hids : (D.boxes.map (·.id)).Nodup) {y z : DBox}
    (hy : y ∈ D.boxes) (hz : z ∈ D.boxes) (h : y.id = z.id) : y = z :=
  List.inj_on_of_nodup_map hids hy hz h

theorem mem_of_id {D : Diagram} (hids : (D.boxes.map (·.id)).Nodup) {y : DBox} {L : List DBox}
    (hL : ∀ z ∈ L, z ∈ D.boxes) (hy : y ∈ D.boxes) (h : y.id ∈ L.map (·.id)) : y ∈ L := by
  obtain ⟨z, hz, e⟩ := List.mem_map.1 h
  rw [eq_of_id hids hy (hL z hz) e.symm]
  exact hz

/-- `ChainOK` of a walk's output, its boxes compared by id. -/
def chainB (D : Diagram) (pr : DBox → Bool) (start : Option ℕ) (L : List DBox) : Bool :=
  decide ((start :: L.map (fun b => some b.id)).Nodup) &&
    D.boxes.all fun y => !pr y || !decide (y.parent ∈ start :: L.map (fun b => some b.id)) ||
      decide (y.id ∈ L.map (·.id))

theorem chainB_ok {D : Diagram} (hids : (D.boxes.map (·.id)).Nodup) (pr : DBox → Bool)
    (fuel : ℕ) (start : Option ℕ) (h : chainB D pr start (D.walk pr fuel start) = true) :
    Diagram.ChainOK D pr start (D.walk pr fuel start) := by
  simp only [chainB, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, Bool.or_eq_true,
    Bool.not_eq_true', decide_eq_false_iff_not] at h
  have hs := walk_sub D pr fuel start
  refine ⟨hs, fun y hy hp hpar => ?_, walk_parent D pr fuel start, h.1⟩
  rcases h.2 y hy with (h1 | h1) | h1
  · simp [hp] at h1
  · exact absurd hpar h1
  · exact mem_of_id hids (fun z hz => (hs z hz).1) hy h1

theorem ChainOK_congr {D : Diagram} {p q : DBox → Bool} (h : ∀ y ∈ D.boxes, p y = q y)
    {start : Option ℕ} {L : List DBox} (hc : Diagram.ChainOK D q start L) :
    Diagram.ChainOK D p start L := by
  obtain ⟨h1, h2, h3, h4⟩ := hc
  refine ⟨fun y hy => ⟨(h1 y hy).1, by rw [h y (h1 y hy).1]; exact (h1 y hy).2⟩,
    fun y hy hp => h2 y hy (by rw [← h y hy]; exact hp), h3, h4⟩

theorem prefix_of_ids {D : Diagram} (hids : (D.boxes.map (·.id)).Nodup) :
    ∀ (L₁ L₂ : List DBox), (∀ z ∈ L₁, z ∈ D.boxes) → (∀ z ∈ L₂, z ∈ D.boxes) →
      L₁.map (·.id) <+: L₂.map (·.id) → L₁ <+: L₂
  | [], _, _, _, _ => List.nil_prefix
  | _ :: _, [], _, _, h => by simp at h
  | a :: L₁, b :: L₂, h₁, h₂, h => by
    simp only [List.map_cons, List.cons_prefix_cons] at h
    rw [eq_of_id hids (h₁ a List.mem_cons_self) (h₂ b List.mem_cons_self) h.1]
    exact List.cons_prefix_cons.2 ⟨rfl, prefix_of_ids hids L₁ L₂
      (fun z hz => h₁ z (List.mem_cons_of_mem _ hz))
      (fun z hz => h₂ z (List.mem_cons_of_mem _ hz)) h.2⟩

/-- `nodeBoxes` through the masks. -/
def nodeBoxesF (MS : List (ℕ × ℕ)) (D : Diagram) (x : ℕ) : List DBox :=
  D.walk (fun y => (bmask MS y).testBit x) D.boxes.length none

/-- `portBoxes` through the masks. -/
def portBoxesF (MS : List (ℕ × ℕ)) (D : Diagram) (q : Port) : List DBox :=
  D.walk (fun y => decide (q ∈ y.ports)) D.boxes.length
    ((nodeBoxesF MS D q.node).getLast?.map (·.id))

/-- `nodeChain` through the masks. -/
def nodeChainF (MS : List (ℕ × ℕ)) (D : Diagram) (x : ℕ) : List ℕ :=
  (nodeBoxesF MS D x).map (·.mult)

theorem nodeBoxes_eq {MS : List (ℕ × ℕ)} {D : Diagram} (h : masksOK MS D = true) (x : ℕ) :
    D.nodeBoxes x = nodeBoxesF MS D x :=
  walk_congr D _ _ (fun y hy => (bmask_ok MS D h y hy x).symm) _ _

theorem portBoxes_eq {MS : List (ℕ × ℕ)} {D : Diagram} (h : masksOK MS D = true) (q : Port) :
    D.portBoxes q = portBoxesF MS D q := by
  unfold Diagram.portBoxes portBoxesF
  rw [nodeBoxes_eq h]

theorem nodeChain_eq {MS : List (ℕ × ℕ)} {D : Diagram} (h : masksOK MS D = true) (x : ℕ) :
    D.nodeChain x = nodeChainF MS D x := by
  unfold Diagram.nodeChain nodeChainF
  rw [nodeBoxes_eq h]

/-! ## Partners and tails -/

theorem pkey_node (P : ℕ) (p : Port) (hp : p.pos < P) : pkey P p / P / 2 = p.node := by
  have hP : 0 < P := by omega
  unfold pkey
  rw [Nat.mul_comm _ P, Nat.mul_add_div hP, Nat.div_eq_of_lt hp, Nat.add_zero]
  split <;> omega

/-- The node of the other end of wire `j`, from the packed wires, for the leg with key `k`. -/
def otherNode (P K WB j k : ℕ) : ℕ :=
  let e := (WB >>> (2 * K * j)) % 2 ^ (2 * K)
  (if e % 2 ^ K == k then e / 2 ^ K else e % 2 ^ K) / P / 2

/-- `tail` of the leg `q` at the label `l`. -/
def tailF (P K off WB : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (q : Port) (l : ℕ) : List ℕ :=
  if l < off then (portBoxesF MS D q).map (·.mult)
  else
    let a := nodeChainF MS D q.node
    let c := nodeChainF MS D (otherNode P K WB (l - off) (pkey P q))
    if a.length < c.length then c.drop a.length else []

theorem partner_wire (D : Diagram) (hnd : (ends D).Nodup) (j : ℕ) (hj : j < D.wires.length)
    (q : Port) (hq : D.wires[j].1 = q ∨ D.wires[j].2 = q) :
    D.partner q = some (if D.wires[j].1 = q then D.wires[j].2 else D.wires[j].1) := by
  unfold ends at hnd
  have hpair := (List.nodup_flatMap.1 (List.nodup_append.1 hnd).2.1).2
  have hfind : D.wires.find? (fun w => decide (w.1 = q ∨ w.2 = q)) = some D.wires[j] := by
    refine find?_unique _ _ _ (List.getElem_mem hj) (by simpa using hq) fun y hy hy' => ?_
    obtain ⟨j', hj', rfl⟩ := List.mem_iff_getElem.1 hy
    by_contra hne
    have hjj : j' ≠ j := fun e => hne (by subst e; rfl)
    simp only [decide_eq_true_eq] at hy'
    have h1 : q ∈ [D.wires[j'].1, D.wires[j'].2] := by rcases hy' with e | e <;> simp [e]
    have h2 : q ∈ [D.wires[j].1, D.wires[j].2] := by rcases hq with e | e <;> simp [e]
    rcases Nat.lt_or_gt_of_ne hjj with hlt | hlt
    · have hd := List.pairwise_iff_getElem.1 hpair j' j hj' hj hlt
      simp only [Function.onFun] at hd
      exact hd h1 h2
    · have hd := List.pairwise_iff_getElem.1 hpair j j' hj hj' hlt
      simp only [Function.onFun] at hd
      exact hd h2 h1
  unfold Diagram.partner
  rw [hfind]
  rfl

theorem partner_bnd (D : Diagram) (hnd : (ends D).Nodup) (q : Port) (hq : q ∈ D.bnd) :
    D.partner q = none := by
  unfold ends at hnd
  have hdis := (List.nodup_append.1 hnd).2.2
  unfold Diagram.partner
  rw [find?_none _ _ fun w hw => ?_]
  · rfl
  · simp only [decide_eq_false_iff_not, not_or]
    refine ⟨fun e => ?_, fun e => ?_⟩
    · exact hdis q hq q (List.mem_flatMap.2 ⟨w, hw, by simp [e]⟩) rfl
    · exact hdis q hq q (List.mem_flatMap.2 ⟨w, hw, by simp [e]⟩) rfl

theorem tail_eq (P K off WB : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (hm : masksOK MS D = true)
    (hnd : (ends D).Nodup) (hpos : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K)
    (hW : packW P K D.wires = WB) (q : Port) (hq : q.pos < P) (l : ℕ)
    (h : legOK P K off WB D q l = true) : D.tail q = tailF P K off WB MS D q l := by
  unfold legOK at h
  unfold tailF
  split_ifs at h ⊢ with hl
  · simp only [beq_iff_eq] at h
    obtain ⟨hi, hbi⟩ := List.getElem?_eq_some_iff.1 h
    unfold Diagram.tail
    rw [partner_bnd D hnd q (hbi ▸ List.getElem_mem hi), portBoxes_eq hm]
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨hj, hh⟩ := h
    have hk : ∀ w ∈ D.wires, pkey P w.1 < 2 ^ K ∧ pkey P w.2 < 2 ^ K := fun w hw' =>
      ⟨(hpos w.1 (List.mem_append_right _ (List.mem_flatMap.2 ⟨w, hw', by simp⟩))).2,
        (hpos w.2 (List.mem_append_right _ (List.mem_flatMap.2 ⟨w, hw', by simp⟩))).2⟩
    have hmem := wire_ends_mem D (l - off) hj
    have hp1 := hpos _ hmem.1
    have hp2 := hpos _ hmem.2
    have hpw : D.wires[l - off].1 = q ∨ D.wires[l - off].2 = q := by
      rcases wireHas_ok P K WB D.wires hW (l - off) hj hk _ hh with e | e
      · exact .inl (pkey_inj P _ _ hp1.1 hq e)
      · exact .inr (pkey_inj P _ _ hp2.1 hq e)
    have hent : (WB >>> (2 * K * (l - off))) % 2 ^ (2 * K) =
        pkey P D.wires[l - off].1 + 2 ^ K * pkey P D.wires[l - off].2 := by
      rw [Nat.shiftRight_eq_div_pow, ← hW, packW_get P K D.wires (l - off) hj hk]
    have hK : 0 < 2 ^ K := by positivity
    have hlo : (pkey P D.wires[l - off].1 + 2 ^ K * pkey P D.wires[l - off].2) % 2 ^ K =
        pkey P D.wires[l - off].1 := by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp1.2]
    have hhi : (pkey P D.wires[l - off].1 + 2 ^ K * pkey P D.wires[l - off].2) / 2 ^ K =
        pkey P D.wires[l - off].2 := by
      rw [Nat.add_mul_div_left _ _ hK, Nat.div_eq_of_lt hp1.2, Nat.zero_add]
    have hother : otherNode P K WB (l - off) (pkey P q) =
        (if D.wires[l - off].1 = q then D.wires[l - off].2 else D.wires[l - off].1).node := by
      unfold otherNode
      simp only [hent, hlo, hhi]
      by_cases e : D.wires[l - off].1 = q
      · rw [if_pos e, if_pos (by simp [e])]
        exact pkey_node P _ hp2.1
      · rw [if_neg e, if_neg fun e' => e (pkey_inj P _ _ hp1.1 hq (beq_iff_eq.1 e'))]
        exact pkey_node P _ hp1.1
    unfold Diagram.tail
    rw [partner_wire D hnd (l - off) hj q hpw]
    simp only [nodeChain_eq hm, hother]

/-! ## The plain bridge's facts -/

theorem bridgeD_facts (P K b off WB T : ℕ) (D : Diagram) (N : Net) (L : List (ℕ × ℕ))
    (h : bridgeD P K b off WB T D N L = true) :
    D.bnd.length ≤ off ∧ (D.nodes.map (·.id)).Nodup ∧ (ends D).Nodup ∧
    (∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K) ∧ packW P K D.wires = WB ∧
    (∀ n ∈ D.nodes, n.outs.length ≤ P ∧ n.ins.length ≤ P) ∧ N.nodes.length = D.nodes.length ∧
    (∀ n ∈ D.nodes, ∀ p ∈ legs n, legOK P K off WB D p (D.label off p) = true) ∧
    (D.nodes.flatMap legs).Perm (ends D) ∧
    (∀ p ∈ ends D, D.portDim p = ent b T (D.label off p)) := by
  unfold bridgeD at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hoff, hinc⟩, hP⟩, hpos⟩, hnd⟩, hW⟩, hcnt⟩, -⟩, -⟩, -⟩, hlen⟩,
    hnodes⟩ := h
  have hid := incrL_ok _ hinc
  have hnd' : (ends D).Nodup := List.Nodup.of_map (pkey P) (nodupB_ok _ 0 hnd).1
  have hpos' : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K := fun p hp => hpos p hp
  have hnode : ∀ i (hi : i < D.nodes.length),
      nodeOKD P K b off WB T D D.nodes[i] (N.nodes[i]'(by omega)) = true := fun i hi =>
    hnodes (D.nodes[i], N.nodes[i]'(by omega))
      (List.mem_iff_getElem.2 ⟨i, by simp [hi, hlen], by simp⟩)
  have hleg : ∀ i (hi : i < D.nodes.length) (o : Bool) (k : ℕ),
      k < (if o then D.nodes[i].outs.length else D.nodes[i].ins.length) →
      (⟨D.nodes[i].id, o, k⟩ ∈ ends D ∧
      D.portDim ⟨D.nodes[i].id, o, k⟩ = ent b T (D.label off ⟨D.nodes[i].id, o, k⟩)) ∧
      legOK P K off WB D ⟨D.nodes[i].id, o, k⟩ (D.label off ⟨D.nodes[i].id, o, k⟩) = true := by
    intro i hi o k hk
    have hn := hnode i hi
    unfold nodeOKD at hn
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hn
    obtain ⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, ho⟩, hi'⟩, hao⟩, hai⟩ := hn
    obtain ⟨hPo, hPi⟩ := hP _ (List.getElem_mem hi)
    rw [portDim_leg D hid i hi]
    cases o with
    | true =>
      simp only [ite_true] at hk ⊢
      have := all_zip_get _ _ _ hao k (by simpa using hk) (by omega)
      simp only [List.getElem_range, Bool.and_eq_true, beq_iff_eq] at this
      have hl := locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this.1
      refine ⟨⟨hl.2, ?_⟩, ?_⟩
      · rw [hl.1.1]
        exact this.2
      · rw [hl.1.1]
        exact this.1
    | false =>
      simp only [Bool.false_eq_true, ite_false] at hk ⊢
      have := all_zip_get _ _ _ hai k (by simpa using hk) (by omega)
      simp only [List.getElem_range, Bool.and_eq_true, beq_iff_eq] at this
      have hl := locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this.1
      refine ⟨⟨hl.2, ?_⟩, ?_⟩
      · rw [hl.1.1]
        exact this.2
      · rw [hl.1.1]
        exact this.1
  have hsub : D.nodes.flatMap legs ⊆ ends D := by
    intro p hp
    obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 hp
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
    unfold legs at hpn
    simp only [List.mem_append, List.mem_map, List.mem_range] at hpn
    rcases hpn with ⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩
    · exact (hleg i hi true k (by simpa using hk)).1.1
    · exact (hleg i hi false k (by simpa using hk)).1.1
  have hperm := ((legs_nodup D hid).subperm hsub).perm_of_length_le (by
    rw [← hcnt, List.length_flatMap]; simp [legs])
  refine ⟨hoff, hid, hnd', hpos', hW, hP, hlen, fun n hn p hpn => ?_, hperm, fun p hp => ?_⟩
  · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
    unfold legs at hpn
    simp only [List.mem_append, List.mem_map, List.mem_range] at hpn
    rcases hpn with ⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩
    · exact (hleg i hi true k (by simpa using hk)).2
    · exact (hleg i hi false k (by simpa using hk)).2
  obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 (hperm.mem_iff.2 hp)
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
  unfold legs at hpn
  simp only [List.mem_append, List.mem_map, List.mem_range] at hpn
  rcases hpn with ⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩
  · exact (hleg i hi true k (by simpa using hk)).1.2
  · exact (hleg i hi false k (by simpa using hk)).1.2

/-! ## The bundled network -/

/-- The nonempty tails of the side `out` of node `x`, leg `j` at the label `ls.getD j 0`. -/
def tlF (P K off WB : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (x : ℕ) (out : Bool)
    (ls : List ℕ) : List (ℕ × List ℕ) :=
  ((List.range ls.length).map fun j =>
    (j, tailF P K off WB MS D ⟨x, out, j⟩ (ls.getD j 0))).filter fun p => p.2 ≠ []

/-- `bnode` of `n` from its plain labelled node `m`. -/
def bnodeF (P K off WB : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (n : DNode) (m : Node) : Node :=
  let b := bundleNode m (tlF P K off WB MS D n.id true m.outs)
    (tlF P K off WB MS D n.id false m.ins)
  { b with kind := (nodeChainF MS D n.id).foldr Kind.Bun b.kind }

/-- The side `out` of `n` labelled by `D.label off`. -/
theorem tl_eq (P K off WB : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (hm : masksOK MS D = true)
    (hnd : (ends D).Nodup) (hpos : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K)
    (hW : packW P K D.wires = WB) (n : DNode) (hP : n.outs.length ≤ P ∧ n.ins.length ≤ P)
    (hl : ∀ p ∈ legs n, legOK P K off WB D p (D.label off p) = true) (out : Bool) :
    ((List.range (if out then n.outs.length else n.ins.length)).map fun j =>
        (j, D.tail ⟨n.id, out, j⟩)).filter (fun p => p.2 ≠ []) =
      tlF P K off WB MS D n.id out
        (if out then (n.node (D.label off)).outs else (n.node (D.label off)).ins) := by
  obtain ⟨hP1, hP2⟩ := hP
  unfold tlF
  congr 1
  have hlen : (if out then (n.node (D.label off)).outs else (n.node (D.label off)).ins).length =
      (if out then n.outs.length else n.ins.length) := by
    cases out <;> simp [DNode.node]
  rw [hlen]
  refine List.map_congr_left fun j hj => ?_
  have hj' := List.mem_range.1 hj
  have hget : (if out then (n.node (D.label off)).outs else (n.node (D.label off)).ins).getD j 0 =
      D.label off ⟨n.id, out, j⟩ := by
    cases out <;> simp only [ite_true, Bool.false_eq_true, ite_false] at hj' ⊢ <;>
      simp [DNode.node, hj']
  rw [hget]
  have hmem : (⟨n.id, out, j⟩ : Port) ∈ legs n := by
    cases out <;> simp only [ite_true, Bool.false_eq_true, ite_false] at hj' <;> simp [legs, hj']
  have hjP : j < P := by
    cases out <;> simp only [ite_true, Bool.false_eq_true, ite_false] at hj' <;> omega
  rw [tail_eq P K off WB MS D hm hnd hpos hW _ hjP _ (hl _ hmem)]

theorem bnode_eq (P K off WB : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (hm : masksOK MS D = true)
    (hnd : (ends D).Nodup) (hpos : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K)
    (hW : packW P K D.wires = WB) (n : DNode) (hP : n.outs.length ≤ P ∧ n.ins.length ≤ P)
    (hl : ∀ p ∈ legs n, legOK P K off WB D p (D.label off p) = true) :
    D.bnode off n = bnodeF P K off WB MS D n (n.node (D.label off)) := by
  have ho := tl_eq P K off WB MS D hm hnd hpos hW n hP hl true
  have hi := tl_eq P K off WB MS D hm hnd hpos hW n hP hl false
  simp only [ite_true, Bool.false_eq_true, ite_false] at ho hi
  unfold Diagram.bnode bnodeF
  simp only [ho, hi, nodeChain_eq hm]

/-! ## The family checks -/

/-- `NodeSb` of `n` with its chain `c` and side tails `to`, `ti`. -/
def nodeSB (n : DNode) (c : List ℕ) (tO tI : List (ℕ × List ℕ)) : Bool :=
  decide (n.kind.plain = true ∧ (c ≠ [] → n.kind ≠ .B ∧ n.kind ≠ .S) ∧
    (¬(tO = [] ∧ tI = []) → n.kind = .Z ∨ n.kind = .X ∨ (n.kind = .W ∧ tI = [])) ∧
    ((c ≠ [] ∨ ¬(tO = [] ∧ tI = [])) →
      (∀ k < n.outs.length, n.outs.getD k 0 = n.dim) ∧ ∀ k < n.ins.length, n.ins.getD k 0 = n.dim))

/-- Label `l` is, in `F` and in the table `TB`, the bundle of `base` along `c`. -/
def labB (b' TB : ℕ) (F : Fam) (l base : ℕ) (c : List ℕ) : Bool :=
  ent b' TB l == dimFor F.bund base c && decide (Diagram.DimOK F base c)

/-- The tail of a wire's first end. -/
def wtail (MS : List (ℕ × ℕ)) (D : Diagram) (w : Port × Port) : List ℕ :=
  let a := nodeChainF MS D w.1.node
  let c := nodeChainF MS D w.2.node
  if a.length < c.length then c.drop a.length else []

/-- `FamOK` past the plain bridge, `Np` the plain network. -/
def famB (P K b b' off WB T TB MN : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (Np : Net)
    (F : Fam) : Bool :=
  masksOK MS D && decide ((D.boxes.map (·.id)).Nodup) &&
    D.boxes.all (fun y => decide (y.nodes = [] ∨ y.ports = [])) &&
    maskOf (D.nodes.map (·.id)) == MN &&
    D.boxes.all (fun y => y.nodes.all fun x => MN.testBit x) &&
    D.nodes.all (fun n =>
      chainB D (fun y => (bmask MS y).testBit n.id) none (nodeBoxesF MS D n.id)) &&
    D.boxes.all (fun y => y.nodes.all fun x => decide (y.id ∈ (nodeBoxesF MS D x).map (·.id))) &&
    D.bnd.all (fun p => chainB D (fun y => decide (p ∈ y.ports))
      ((nodeBoxesF MS D p.node).getLast?.map (·.id)) (portBoxesF MS D p)) &&
    D.bnd.all (fun p => D.boxes.all fun y =>
      !decide (p ∈ y.ports) || decide (y.id ∈ (portBoxesF MS D p).map (·.id))) &&
    D.boxes.all (fun y => y.ports.all fun p => decide (p ∈ D.bnd)) &&
    D.wires.all (fun w =>
      decide ((nodeBoxesF MS D w.1.node).map (·.id) <+: (nodeBoxesF MS D w.2.node).map (·.id)) ||
      decide ((nodeBoxesF MS D w.2.node).map (·.id) <+: (nodeBoxesF MS D w.1.node).map (·.id))) &&
    D.boxes.all (fun y => decide (y.nodes ≠ [] ∨ y.ports ≠ [])) &&
    (D.nodes.zip Np.nodes).all (fun q => nodeSB q.1 (nodeChainF MS D q.1.id)
      (tlF P K off WB MS D q.1.id true q.2.outs) (tlF P K off WB MS D q.1.id false q.2.ins)) &&
    labsOK b' TB D.bnd.length off D.wires.length F.labs &&
    ((List.range D.wires.length).zip D.wires).all (fun q =>
      labB b' TB F (off + q.1) (ent b T (off + q.1)) (nodeChainF MS D q.2.1.node ++ wtail MS D q.2)) &&
    ((List.range D.bnd.length).zip D.bnd).all (fun q =>
      labB b' TB F q.1 (ent b T q.1) (nodeChainF MS D q.2.node ++ (portBoxesF MS D q.2).map (·.mult)))

/-- The plain label table of `T`: the boundary slots, then the wire labels from `off`. -/
def plainL (b T nb off W : ℕ) : List (ℕ × ℕ) :=
  (List.range nb ++ (List.range W).map (off + ·)).map fun l => (l, ent b T l)

/-- `N` is `D.toNetB off` and `D.FamOK F off` holds, `Np` the plain network. -/
def bridgeB (P K b b' off WB T TB MN : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram) (N Np : Net)
    (F : Fam) : Bool :=
  bridgeD P K b off WB T D Np (plainL b T D.bnd.length off D.wires.length) &&
    famB P K b b' off WB T TB MN MS D Np F &&
    decide (N = ⟨(D.nodes.zip Np.nodes).map (fun q => bnodeF P K off WB MS D q.1 q.2),
      Np.int, Np.bnd⟩)

theorem zip_map_self {α β γ : Type*} (f : α → β) (g : α → β → γ) :
    ∀ l : List α, (l.zip (l.map f)).map (fun q => g q.1 q.2) = l.map fun x => g x (f x)
  | [] => rfl
  | a :: l => by simp only [List.map_cons, List.zip_cons_cons, zip_map_self f g l]

theorem flatMap_ends_perm (ws : List (Port × Port)) :
    (ws.flatMap fun w => [w.1, w.2]).Perm (ws.map (·.1) ++ ws.map (·.2)) := by
  induction ws with
  | nil => simp
  | cons w ws ih =>
    simp only [List.flatMap_cons, List.map_cons, List.cons_append, List.nil_append]
    exact ((ih.cons w.2).cons w.1).trans (List.perm_middle.symm.cons w.1)

theorem perm_ends (ws : List (Port × Port)) (B : List Port) :
    (ws.map (·.1) ++ (ws.map (·.2) ++ B)).Perm (B ++ ws.flatMap fun w => [w.1, w.2]) := by
  rw [← List.append_assoc]
  exact List.perm_append_comm.trans ((flatMap_ends_perm ws).symm.append_left B)

theorem bridgeB_ok (P K b b' off WB T TB MN : ℕ) (MS : List (ℕ × ℕ)) (D : Diagram)
    (N Np : Net) (F : Fam) (h : bridgeB P K b b' off WB T TB MN MS D N Np F = true) :
    D.toNetB off = N ∧ D.FamOK F off := by
  unfold bridgeB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hD, hF⟩, hN⟩ := h
  have hNp := (bridgeD_ok P K b off WB T D Np _ hD).2
  obtain ⟨hoff, hid, hnd, hpos, hW, hP, -, hlo, hperm, hdim⟩ :=
    bridgeD_facts P K b off WB T D Np _ hD
  unfold famB at hF
  simp only [Bool.and_eq_true] at hF
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hm, hbid⟩, h4⟩, hMN⟩, h5⟩, h6⟩, h7⟩, h8⟩, h9⟩, h10⟩, h11⟩, h12⟩, hS⟩,
    hlabs⟩, hwl⟩, hbl⟩ := hF
  have hbid' : (D.boxes.map (·.id)).Nodup := of_decide_eq_true hbid
  have hids : ∀ x, MN.testBit x = decide (x ∈ D.nodes.map (·.id)) := by
    intro x; rw [← beq_iff_eq.1 hMN, testBit_maskOf]
  have hbn : ∀ n ∈ D.nodes, D.bnode off n = bnodeF P K off WB MS D n (n.node (D.label off)) :=
    fun n hn => bnode_eq P K off WB MS D hm hnd hpos hW n (hP n hn) (hlo n hn)
  have hNn : Np.nodes = D.nodes.map (·.node (D.label off)) := by rw [← hNp]; rfl
  refine ⟨?_, ?_⟩
  · rw [hN, hNn, zip_map_self, ← hNp]
    unfold Diagram.toNetB Diagram.toNet
    congr 1
    exact List.map_congr_left hbn
  have hwlab : ∀ j (hj : j < D.wires.length) (p : Port), (D.wires[j].1 = p ∨ D.wires[j].2 = p) →
      D.label off p = off + j := fun j hj p hpw => by
    unfold Diagram.label; rw [locate_wire D hnd j hj p hpw]
  have hwdim : ∀ j (hj : j < D.wires.length),
      D.portDim D.wires[j].1 = ent b T (off + j) ∧ D.portDim D.wires[j].2 = ent b T (off + j) :=
    fun j hj => by
      have hme := wire_ends_mem D j hj
      rw [hdim _ hme.1, hdim _ hme.2, hwlab j hj _ (.inl rfl), hwlab j hj _ (.inr rfl)]
      exact ⟨rfl, rfl⟩
  have hbdim : ∀ i (hi : i < D.bnd.length), D.portDim D.bnd[i] = ent b T i := fun i hi => by
    rw [hdim _ (List.mem_append_left _ (List.getElem_mem hi))]
    unfold Diagram.label; rw [locate_bnd D hnd i hi]
  have hlk : ∀ l, inQ D.bnd.length off D.wires.length l = true →
      F.labs.lookup l = some (ent b' TB l) :=
    labs_lookup b' TB _ off _ F.labs hoff hlabs
  have hwchain : ∀ j (hj : j < D.wires.length),
      D.legChain D.wires[j].1 = nodeChainF MS D D.wires[j].1.node ++ wtail MS D D.wires[j] :=
    fun j hj => by
      unfold Diagram.legChain Diagram.tail
      simp only [partner_wire D hnd j hj _ (.inl rfl), ite_true, nodeChain_eq hm, wtail]
  have hbchain : ∀ i (hi : i < D.bnd.length), D.legChain D.bnd[i] =
      nodeChainF MS D D.bnd[i].node ++ (portBoxesF MS D D.bnd[i]).map (·.mult) := fun i hi => by
    unfold Diagram.legChain Diagram.tail
    simp only [partner_bnd D hnd _ (List.getElem_mem hi), nodeChain_eq hm, portBoxes_eq hm]
  have hwalk : ∀ (p : DBox → Bool) fuel start, ∀ z ∈ D.walk p fuel start, z ∈ D.boxes :=
    fun p fuel start z hz => (walk_sub D p fuel start z hz).1
  unfold Diagram.FamOK
  refine ⟨?_, hoff, ?_, ?_, ?_⟩
  · unfold Diagram.WFb
    refine ⟨hid, hbid', ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact (perm_ends D.wires D.bnd).trans hperm.symm
    · exact fun y hy => of_decide_eq_true (List.all_eq_true.1 h4 y hy)
    · intro y hy x hx
      have := List.all_eq_true.1 (List.all_eq_true.1 h5 y hy) x hx
      rw [hids] at this
      exact of_decide_eq_true this
    · intro x hx
      obtain ⟨n, hn, rfl⟩ := List.mem_map.1 hx
      rw [nodeBoxes_eq hm]
      exact ChainOK_congr (fun y hy => (bmask_ok MS D hm y hy n.id).symm)
        (chainB_ok hbid' _ D.boxes.length none (List.all_eq_true.1 h6 n hn))
    · intro y hy x hx
      rw [nodeBoxes_eq hm]
      exact mem_of_id hbid' (hwalk _ _ _) hy
        (of_decide_eq_true (List.all_eq_true.1 (List.all_eq_true.1 h7 y hy) x hx))
    · intro p hp
      rw [nodeBoxes_eq hm, portBoxes_eq hm]
      exact chainB_ok hbid' _ D.boxes.length _ (List.all_eq_true.1 h8 p hp)
    · intro p hp y hy hpy
      have := List.all_eq_true.1 (List.all_eq_true.1 h9 p hp) y hy
      simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not,
        decide_eq_true_eq] at this
      rw [portBoxes_eq hm]
      exact mem_of_id hbid' (hwalk _ _ _) hy (this.resolve_left (not_not.2 hpy))
    · intro y hy p hp
      exact of_decide_eq_true (List.all_eq_true.1 (List.all_eq_true.1 h10 y hy) p hp)
    · intro w hw
      have := List.all_eq_true.1 h11 w hw
      simp only [Bool.or_eq_true, decide_eq_true_eq] at this
      rw [nodeBoxes_eq hm, nodeBoxes_eq hm]
      rcases this with e | e
      · exact .inl (prefix_of_ids hbid' _ _ (hwalk _ _ _) (hwalk _ _ _) e)
      · exact .inr (prefix_of_ids hbid' _ _ (hwalk _ _ _) (hwalk _ _ _) e)
    · exact fun y hy => of_decide_eq_true (List.all_eq_true.1 h12 y hy)
    · intro w hw
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hw
      rw [(hwdim j hj).1, (hwdim j hj).2]
  · intro n hn
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
    have hlenN : i < Np.nodes.length := by rw [hNn]; simpa using hi
    have hs := all_zip_get _ _ _ hS i hi hlenN
    have hNi : Np.nodes[i] = D.nodes[i].node (D.label off) := by
      rw [List.getElem_of_eq hNn hlenN, List.getElem_map]
    simp only [hNi, nodeSB] at hs
    have ho := tl_eq P K off WB MS D hm hnd hpos hW _ (hP _ (List.getElem_mem hi))
      (hlo _ (List.getElem_mem hi)) true
    have hi' := tl_eq P K off WB MS D hm hnd hpos hW _ (hP _ (List.getElem_mem hi))
      (hlo _ (List.getElem_mem hi)) false
    simp only [ite_true, Bool.false_eq_true, ite_false] at ho hi'
    unfold Diagram.NodeSb
    simp only [tls]
    rw [nodeChain_eq hm, ho, hi']
    obtain ⟨a1, a2, a3, a4⟩ := of_decide_eq_true hs
    refine ⟨a1, a2, a3, fun hc p hp => ?_⟩
    obtain ⟨ho', hi''⟩ := a4 hc
    unfold DNode.legs at hp
    simp only [List.mem_append, List.mem_map, List.mem_range] at hp
    rcases hp with ⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩
    · rw [portDim_leg D hid i hi]; exact ho' k hk
    · rw [portDim_leg D hid i hi]; exact hi'' k hk
  · intro j hj
    have := all_zip_get _ _ _ hwl j (by simpa using hj) hj
    simp only [List.getElem_range, labB, Bool.and_eq_true, beq_iff_eq] at this
    rw [(hwdim j hj).1, hwchain j hj, hlk _ (by simp [inQ, hj]), this.1]
    exact ⟨rfl, of_decide_eq_true this.2⟩
  · intro i hi
    have := all_zip_get _ _ _ hbl i (by simpa using hi) hi
    simp only [List.getElem_range, labB, Bool.and_eq_true, beq_iff_eq] at this
    rw [hbdim i hi, hbchain i hi, hlk _ (by simp [inQ, hi]), this.1]
    exact ⟨rfl, of_decide_eq_true this.2⟩

end BBridge
end ArchytasZX
