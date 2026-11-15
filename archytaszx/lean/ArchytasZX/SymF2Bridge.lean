/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.SymF2
import ArchytasZX.Diagram

/-!
# Diagrams as networks, checked in linear time

`bridgeQ` checks that a network is `D.toNet off` and that `D.check` holds for the labels
`(List.range M).map (·, 0)`, for a diagram whose legs all have dimension index `0`. A port's
key is `(2 node + [input]) P + pos`; the wires are packed as one number `WB` holding each
wire's two keys in `2K` bits; every boundary port and wire end is a distinct key, so a leg
found at its stated label (a boundary slot or a wire of `WB`) is located there by
`Diagram.locate`. `bridge_ok` reads `D.check` and `D.toNet off = N` from `bridgeQ`.
-/

namespace ArchytasZX
namespace SymF2

/-- A port's key `(2 node + [input]) P + pos`. -/
def pkey (P : ℕ) (p : Port) : ℕ := (2 * p.node + (if p.out then 0 else 1)) * P + p.pos

theorem pkey_inj (P : ℕ) (p q : Port) (hp : p.pos < P) (hq : q.pos < P)
    (h : pkey P p = pkey P q) : p = q := by
  obtain ⟨a, b, c⟩ := p
  obtain ⟨a', b', c'⟩ := q
  simp only [pkey] at h hp hq
  have hP : 0 < P := by omega
  have hc : c = c' := by
    have e := congrArg (· % P) h
    rwa [Nat.mul_add_mod_of_lt hp, Nat.mul_add_mod_of_lt hq] at e
  subst hc
  have hX := Nat.eq_of_mul_eq_mul_right hP (Nat.add_right_cancel h)
  cases b <;> cases b' <;> simp at hX <;> first | (subst hX; rfl) | omega | (congr 1; omega)

/-- The wires packed: wire `j`'s keys `k₁ + 2^K k₂` at `2^(2K j)`. -/
def packW (P K : ℕ) : List (Port × Port) → ℕ
  | [] => 0
  | w :: ws => pkey P w.1 + 2 ^ K * pkey P w.2 + 2 ^ (2 * K) * packW P K ws

theorem pair_lt (K a b : ℕ) (ha : a < 2 ^ K) (hb : b < 2 ^ K) : a + 2 ^ K * b < 2 ^ (2 * K) := by
  have : a + 2 ^ K * b < 2 ^ K * (b + 1) := by nlinarith
  calc a + 2 ^ K * b < 2 ^ K * (b + 1) := this
    _ ≤ 2 ^ K * 2 ^ K := Nat.mul_le_mul_left _ hb
    _ = 2 ^ (2 * K) := by rw [← pow_add]; ring_nf

theorem packW_get (P K : ℕ) : ∀ (ws : List (Port × Port)) (j : ℕ) (hj : j < ws.length),
    (∀ w ∈ ws, pkey P w.1 < 2 ^ K ∧ pkey P w.2 < 2 ^ K) →
    packW P K ws / 2 ^ (2 * K * j) % 2 ^ (2 * K) = pkey P ws[j].1 + 2 ^ K * pkey P ws[j].2
  | [], j, hj, _ => absurd hj (by simp)
  | w :: ws, 0, _, hk => by
    have h := hk w List.mem_cons_self
    simp only [packW, Nat.mul_zero, pow_zero, Nat.div_one, List.getElem_cons_zero]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (pair_lt K _ _ h.1 h.2)]
  | w :: ws, j + 1, hj, hk => by
    have h := hk w List.mem_cons_self
    have hpos : 0 < 2 ^ (2 * K) := by positivity
    simp only [packW, List.getElem_cons_succ]
    rw [show 2 * K * (j + 1) = 2 * K + 2 * K * j by ring, pow_add, ← Nat.div_div_eq_div_mul,
      Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt (pair_lt K _ _ h.1 h.2), Nat.zero_add]
    exact packW_get P K ws j (by simpa using hj) fun w' hw' => hk w' (List.mem_cons_of_mem _ hw')

/-- Entry `j` of the packed wires holds the key `k`. -/
def wireHas (K WB j k : ℕ) : Bool :=
  let e := (WB >>> (2 * K * j)) % 2 ^ (2 * K)
  e % 2 ^ K == k || e / 2 ^ K == k

theorem wireHas_ok (P K WB : ℕ) (ws : List (Port × Port)) (hW : packW P K ws = WB) (j : ℕ)
    (hj : j < ws.length) (hk : ∀ w ∈ ws, pkey P w.1 < 2 ^ K ∧ pkey P w.2 < 2 ^ K) (k : ℕ)
    (h : wireHas K WB j k = true) : pkey P ws[j].1 = k ∨ pkey P ws[j].2 = k := by
  simp only [wireHas] at h
  rw [Nat.shiftRight_eq_div_pow, ← hW, packW_get P K ws j hj hk] at h
  have h1 := (hk ws[j] (List.getElem_mem hj)).1
  have hpos : 0 < 2 ^ K := by positivity
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h1, Nat.add_mul_div_left _ _ hpos,
    Nat.div_eq_of_lt h1, Nat.zero_add] at h
  simpa using h

/-- The boundary ports and wire ends. -/
def ends (D : Diagram) : List Port := D.bnd ++ D.wires.flatMap fun w => [w.1, w.2]

/-- Leg `p` is at label `l`: boundary slot `l`, else wire `l - off` of `WB`. -/
def legOK (P K off WB : ℕ) (D : Diagram) (p : Port) (l : ℕ) : Bool :=
  if l < off then D.bnd[l]? == some p
  else decide (l - off < D.wires.length) && wireHas K WB (l - off) (pkey P p)

/-- The legs of `n` at the labels of `m`. -/
def nodeOKQ (P K off WB : ℕ) (D : Diagram) (n : DNode) (m : Node) : Bool :=
  decide (m.kind = n.kind) && m.dim == n.dim && m.ph == n.ph &&
    m.outs.length == n.outs.length && m.ins.length == n.ins.length &&
    ((List.range n.outs.length).zip m.outs).all (fun q => legOK P K off WB D ⟨n.id, true, q.1⟩ q.2) &&
    ((List.range n.ins.length).zip m.ins).all (fun q => legOK P K off WB D ⟨n.id, false, q.1⟩ q.2)

/-- `N` is `D.toNet off` and `D.check` holds, the labels below `M` of index `0`. -/
def bridgeQ (P K off M WB : ℕ) (D : Diagram) (N : Net) : Bool :=
  decide (D.bnd.length ≤ off) && decide (off + D.wires.length ≤ M) &&
    D.nodes.all (fun n => (n.outs ++ n.ins).all (· == 0) && decide (n.outs.length ≤ P) &&
      decide (n.ins.length ≤ P)) &&
    (ends D).all (fun p => decide (p.pos < P) && decide (pkey P p < 2 ^ K)) &&
    nodupS ((ends D).map (pkey P)) 0 && packW P K D.wires == WB &&
    N.int == (List.range D.wires.length).map (off + ·) && N.bnd == List.range D.bnd.length &&
    N.nodes.length == D.nodes.length &&
    (D.nodes.zip N.nodes).all fun q => nodeOKQ P K off WB D q.1 q.2

theorem all_zip_get {α β : Type} (f : α × β → Bool) (xs : List α) (ys : List β)
    (h : (xs.zip ys).all f = true) (i : ℕ) (hi : i < xs.length) (hj : i < ys.length) :
    f (xs[i], ys[i]) = true :=
  List.all_eq_true.1 h _ (List.mem_iff_getElem.2 ⟨i, by simp [hi, hj], by simp⟩)

theorem lookup_range (M l : ℕ) :
    ((List.range M).map fun x => (x, 0)).lookup l = if l < M then some 0 else none := by
  have key : ∀ xs : List ℕ, (xs.map fun x => (x, 0)).lookup l = if l ∈ xs then some 0 else none := by
    intro xs
    induction xs with
    | nil => simp
    | cons x xs ih =>
      by_cases h : l = x
      · subst h; simp [List.lookup_cons]
      · have hb : (l == x) = false := by simp [h]
        rw [List.map_cons, List.lookup_cons, ih, hb]
        simp [h]
  rw [key]; simp

theorem portDim_zero (D : Diagram) (hz : ∀ n ∈ D.nodes, ∀ x ∈ n.outs ++ n.ins, x = 0) (p : Port) :
    D.portDim p = 0 := by
  unfold Diagram.portDim
  cases h : D.nodes.find? (fun n => n.id = p.node) with
  | none => rfl
  | some n =>
    have hn := List.mem_of_find?_eq_some h
    simp only
    split_ifs
    · rcases Nat.lt_or_ge p.pos n.outs.length with hl | hl
      · rw [List.getD_eq_getElem _ _ hl]; exact hz n hn _ (List.mem_append_left _ (List.getElem_mem hl))
      · rw [List.getD_eq_default _ _ hl]
    · rcases Nat.lt_or_ge p.pos n.ins.length with hl | hl
      · rw [List.getD_eq_getElem _ _ hl]; exact hz n hn _ (List.mem_append_right _ (List.getElem_mem hl))
      · rw [List.getD_eq_default _ _ hl]

/-- A leg found at its label is located there. -/
theorem locate_of_leg (P K off WB : ℕ) (D : Diagram) (hnd : (ends D).Nodup)
    (hpos : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K) (hW : packW P K D.wires = WB)
    (p : Port) (hp : p.pos < P) (l : ℕ) (h : legOK P K off WB D p l = true) :
    D.label off p = l ∧ D.okPort p = true := by
  unfold ends at hnd hpos
  rw [List.nodup_append] at hnd
  obtain ⟨hb, hw, hdis⟩ := hnd
  unfold legOK at h
  split_ifs at h with hl
  · simp only [beq_iff_eq] at h
    obtain ⟨hi, hbi⟩ := List.getElem?_eq_some_iff.1 h
    have hidx : D.bnd.idxOf? p = some l := by
      refine List.idxOf?_eq_some_iff.2 ⟨hi, hbi, fun j hj e => ?_⟩
      have := (hb.getElem_inj_iff (hi := by omega) (hj := hi)).1 (e.trans hbi.symm)
      omega
    unfold Diagram.label Diagram.okPort Diagram.locate
    rw [hidx]
    exact ⟨rfl, by simpa using hi⟩
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨hj, hh⟩ := h
    have hk : ∀ w ∈ D.wires, pkey P w.1 < 2 ^ K ∧ pkey P w.2 < 2 ^ K := fun w hw' =>
      ⟨(hpos w.1 (List.mem_append_right _ (List.mem_flatMap.2 ⟨w, hw', by simp⟩))).2,
        (hpos w.2 (List.mem_append_right _ (List.mem_flatMap.2 ⟨w, hw', by simp⟩))).2⟩
    have hmem : ∀ w ∈ D.wires, w.1 ∈ D.wires.flatMap (fun w => [w.1, w.2]) ∧
        w.2 ∈ D.wires.flatMap (fun w => [w.1, w.2]) := fun w hw' =>
      ⟨List.mem_flatMap.2 ⟨w, hw', by simp⟩, List.mem_flatMap.2 ⟨w, hw', by simp⟩⟩
    have hwj := List.getElem_mem hj
    have hpw : D.wires[l - off].1 = p ∨ D.wires[l - off].2 = p := by
      rcases wireHas_ok P K WB D.wires hW (l - off) hj hk _ hh with e | e
      · left
        exact pkey_inj P _ _ (hpos _ (List.mem_append_right _ (hmem _ hwj).1)).1 hp e
      · right
        exact pkey_inj P _ _ (hpos _ (List.mem_append_right _ (hmem _ hwj).2)).1 hp e
    have hpin : p ∈ D.wires.flatMap (fun w => [w.1, w.2]) := by
      rcases hpw with e | e
      · rw [← e]; exact (hmem _ hwj).1
      · rw [← e]; exact (hmem _ hwj).2
    have hnb : p ∉ D.bnd := fun hb' => hdis p hb' p hpin rfl
    have hpair := (List.nodup_flatMap.1 hw).2
    have hfind : D.wires.findIdx? (fun w => decide (w.1 = p ∨ w.2 = p)) = some (l - off) := by
      refine List.findIdx?_eq_some_iff_getElem.2 ⟨hj, by simpa using hpw, fun j' hj' e => ?_⟩
      have hd := List.pairwise_iff_getElem.1 hpair j' (l - off) (by omega) hj hj'
      simp only [Function.onFun] at hd
      simp only [decide_eq_true_eq] at e
      have h1 : p ∈ [D.wires[j'].1, D.wires[j'].2] := by rcases e with e | e <;> simp [e]
      have h2 : p ∈ [D.wires[l - off].1, D.wires[l - off].2] := by
        rcases hpw with e | e <;> simp [e]
      exact hd h1 h2
    unfold Diagram.label Diagram.okPort Diagram.locate
    rw [List.idxOf?_eq_none_iff.2 hnb]
    simp only [hfind, Option.map_some]
    exact ⟨by omega, by simpa using hj⟩

theorem bridge_ok (P K off M WB : ℕ) (D : Diagram) (N : Net)
    (h : bridgeQ P K off M WB D N = true) :
    D.check ((List.range M).map fun l => (l, 0)) off = true ∧ D.toNet off = N := by
  unfold bridgeQ at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hoff, hM⟩, hdims⟩, hpos⟩, hnd⟩, hW⟩, hint⟩, hbnd⟩, hlen⟩, hnodes⟩ := h
  rw [nodupS_eq] at hnd
  have hnd' : (ends D).Nodup := List.Nodup.of_map (pkey P) (nodupM_ok _ 0 hnd).1
  have hpos' : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K := fun p hp => hpos p hp
  have hz : ∀ n ∈ D.nodes, ∀ x ∈ n.outs ++ n.ins, x = 0 := fun n hn x hx => by
    have := (hdims n hn).1.1 x hx
    simpa using this
  have hnode : ∀ i (hi : i < D.nodes.length),
      nodeOKQ P K off WB D D.nodes[i] (N.nodes[i]'(by omega)) = true := fun i hi =>
    hnodes (D.nodes[i], N.nodes[i]'(by omega))
      (List.mem_iff_getElem.2 ⟨i, by simp [hi, hlen], by simp⟩)
  have hleg : ∀ i (hi : i < D.nodes.length) (o : Bool) (k : ℕ),
      k < (if o then D.nodes[i].outs.length else D.nodes[i].ins.length) →
      D.label off ⟨D.nodes[i].id, o, k⟩ =
        (if o then (N.nodes[i]'(by omega)).outs.getD k 0 else (N.nodes[i]'(by omega)).ins.getD k 0) ∧
      D.okPort ⟨D.nodes[i].id, o, k⟩ = true := by
    intro i hi o k hk
    have hn := hnode i hi
    unfold nodeOKQ at hn
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hn
    obtain ⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, ho⟩, hi'⟩, hao⟩, hai⟩ := hn
    obtain ⟨⟨-, hPo⟩, hPi⟩ := hdims _ (List.getElem_mem hi)
    cases o with
    | true =>
      simp only [if_true] at hk ⊢
      have := all_zip_get _ _ _ hao k (by simpa using hk) (by omega)
      simp only [List.getElem_range] at this
      rw [List.getD_eq_getElem _ _ (by omega)]
      exact locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this
    | false =>
      simp only [Bool.false_eq_true, if_false] at hk ⊢
      have := all_zip_get _ _ _ hai k (by simpa using hk) (by omega)
      simp only [List.getElem_range] at this
      rw [List.getD_eq_getElem _ _ (by omega)]
      exact locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this
  refine ⟨?_, ?_⟩
  · unfold Diagram.check
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range]
    refine ⟨⟨⟨⟨hoff, fun n hn => ?_⟩, fun w _ => by rw [portDim_zero D hz, portDim_zero D hz]⟩,
      fun j hj => ?_⟩, ?_⟩
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
      refine ⟨fun k hk => (hleg i hi true k (by simpa using hk)).2,
        fun k hk => (hleg i hi false k (by simpa using hk)).2⟩
    · rw [lookup_range, if_pos (by omega)]
      unfold Diagram.wireDim
      cases D.wires[j]? <;> simp [portDim_zero D hz]
    · apply List.ext_getElem
      · simp [Diagram.bdims, Diagram.bnd]
      · intro i h1 h2
        simp only [List.getElem_map, List.getElem_range, lookup_range]
        rw [if_pos (by simp at h1; omega)]
        simp [Diagram.bdims, portDim_zero D hz]
  · rw [show N = ⟨N.nodes, N.int, N.bnd⟩ from rfl]
    unfold Diagram.toNet
    rw [hint, hbnd]
    congr 1
    apply List.ext_getElem (by simp [hlen])
    intro i h1 h2
    rw [List.getElem_map]
    have hi : i < D.nodes.length := by simpa using h1
    have hn := hnode i hi
    unfold nodeOKQ at hn
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hn
    obtain ⟨⟨⟨⟨⟨⟨hk, hd⟩, hph⟩, ho⟩, hi'⟩, -⟩, -⟩ := hn
    rw [show N.nodes[i] = ⟨N.nodes[i].kind, N.nodes[i].outs, N.nodes[i].ins, N.nodes[i].dim,
      N.nodes[i].ph⟩ from rfl]
    unfold DNode.node
    rw [hk, hd, hph]
    congr 1
    · apply List.ext_getElem (by simp [ho])
      intro k hk1 hk2
      simp only [List.getElem_map, List.getElem_range]
      have := (hleg i hi true k (by simpa using hk1)).1
      simp only [if_true] at this
      rw [this, List.getD_eq_getElem _ _ hk2]
    · apply List.ext_getElem (by simp [hi'])
      intro k hk1 hk2
      simp only [List.getElem_map, List.getElem_range]
      have := (hleg i hi false k (by simpa using hk1)).1
      simp only [Bool.false_eq_true, if_false] at this
      rw [this, List.getD_eq_getElem _ _ hk2]

end SymF2
end ArchytasZX
