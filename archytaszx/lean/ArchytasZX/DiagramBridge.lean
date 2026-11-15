/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Diagram

/-!
# Diagrams as networks, checked in linear time

`bridgeD` checks that a network is `D.toNet off` and that `D.check L off` holds. A port's key
is `(2 node + [input]) P + pos`; the wires are packed as one number `WB` holding each wire's two
keys in `2K` bits; every boundary port and wire end is a distinct key, so a leg found at its
stated label (a boundary slot or a wire of `WB`) is located there by `Diagram.locate`. The
node ids increase, the legs are as many as the boundary ports and wire ends, so every wire end
and boundary port is a leg; each leg's dimension index is entry `l` of the table `T` of
`b`-bit entries at its label `l`, and `L` lists the labels in increasing order, giving every
boundary slot and wire label its entry of `T`. `bridgeD_ok` reads `D.check L off` and
`D.toNet off = N` from `bridgeD`.
-/

namespace ArchytasZX
namespace DBridge

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

theorem all_zip_get {α β : Type} (f : α × β → Bool) (xs : List α) (ys : List β)
    (h : (xs.zip ys).all f = true) (i : ℕ) (hi : i < xs.length) (hj : i < ys.length) :
    f (xs[i], ys[i]) = true :=
  List.all_eq_true.1 h _ (List.mem_iff_getElem.2 ⟨i, by simp [hi, hj], by simp⟩)

theorem wire_ends_mem (D : Diagram) (j : ℕ) (hj : j < D.wires.length) :
    D.wires[j].1 ∈ ends D ∧ D.wires[j].2 ∈ ends D :=
  ⟨List.mem_append_right _ (List.mem_flatMap.2 ⟨_, List.getElem_mem hj, by simp⟩),
    List.mem_append_right _ (List.mem_flatMap.2 ⟨_, List.getElem_mem hj, by simp⟩)⟩

/-- A boundary port is located at its slot. -/
theorem locate_bnd (D : Diagram) (hnd : (ends D).Nodup) (i : ℕ) (hi : i < D.bnd.length) :
    D.locate D.bnd[i] = some (.inl i) := by
  unfold ends at hnd
  have hb := (List.nodup_append.1 hnd).1
  have hidx : D.bnd.idxOf? D.bnd[i] = some i := by
    refine List.idxOf?_eq_some_iff.2 ⟨hi, rfl, fun j hj e => ?_⟩
    have := (hb.getElem_inj_iff (hi := by omega) (hj := hi)).1 e
    omega
  unfold Diagram.locate
  rw [hidx]

/-- A wire end is located at its wire. -/
theorem locate_wire (D : Diagram) (hnd : (ends D).Nodup) (j : ℕ) (hj : j < D.wires.length)
    (p : Port) (hpw : D.wires[j].1 = p ∨ D.wires[j].2 = p) : D.locate p = some (.inr j) := by
  have hmem := wire_ends_mem D j hj
  unfold ends at hnd hmem
  rw [List.nodup_append] at hnd
  obtain ⟨-, hw, hdis⟩ := hnd
  have hpin : p ∈ D.wires.flatMap (fun w => [w.1, w.2]) := by
    rcases hpw with e | e
    · rw [← e]; exact (List.mem_append.1 hmem.1).resolve_left fun h' =>
        hdis _ h' _ (List.mem_flatMap.2 ⟨_, List.getElem_mem hj, by simp⟩) rfl
    · rw [← e]; exact (List.mem_append.1 hmem.2).resolve_left fun h' =>
        hdis _ h' _ (List.mem_flatMap.2 ⟨_, List.getElem_mem hj, by simp⟩) rfl
  have hnb : p ∉ D.bnd := fun hb' => hdis p hb' p hpin rfl
  have hpair := (List.nodup_flatMap.1 hw).2
  have hfind : D.wires.findIdx? (fun w => decide (w.1 = p ∨ w.2 = p)) = some j := by
    refine List.findIdx?_eq_some_iff_getElem.2 ⟨hj, by simpa using hpw, fun j' hj' e => ?_⟩
    have hd := List.pairwise_iff_getElem.1 hpair j' j (by omega) hj hj'
    simp only [Function.onFun] at hd
    simp only [decide_eq_true_eq] at e
    have h1 : p ∈ [D.wires[j'].1, D.wires[j'].2] := by rcases e with e | e <;> simp [e]
    have h2 : p ∈ [D.wires[j].1, D.wires[j].2] := by rcases hpw with e | e <;> simp [e]
    exact hd h1 h2
  unfold Diagram.locate
  rw [List.idxOf?_eq_none_iff.2 hnb]
  simp only [hfind, Option.map_some]

/-- A leg found at its label is located there, and is a boundary port or wire end. -/
theorem locate_of_leg (P K off WB : ℕ) (D : Diagram) (hnd : (ends D).Nodup)
    (hpos : ∀ p ∈ ends D, p.pos < P ∧ pkey P p < 2 ^ K) (hW : packW P K D.wires = WB)
    (p : Port) (hp : p.pos < P) (l : ℕ) (h : legOK P K off WB D p l = true) :
    (D.label off p = l ∧ D.okPort p = true) ∧ p ∈ ends D := by
  unfold legOK at h
  split_ifs at h with hl
  · simp only [beq_iff_eq] at h
    obtain ⟨hi, hbi⟩ := List.getElem?_eq_some_iff.1 h
    have hloc := locate_bnd D hnd l hi
    rw [hbi] at hloc
    unfold Diagram.label Diagram.okPort
    rw [hloc]
    exact ⟨⟨rfl, by simpa using hi⟩, List.mem_append_left _ (hbi ▸ List.getElem_mem hi)⟩
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨hj, hh⟩ := h
    have hk : ∀ w ∈ D.wires, pkey P w.1 < 2 ^ K ∧ pkey P w.2 < 2 ^ K := fun w hw' =>
      ⟨(hpos w.1 (List.mem_append_right _ (List.mem_flatMap.2 ⟨w, hw', by simp⟩))).2,
        (hpos w.2 (List.mem_append_right _ (List.mem_flatMap.2 ⟨w, hw', by simp⟩))).2⟩
    have hmem := wire_ends_mem D (l - off) hj
    have hpw : D.wires[l - off].1 = p ∨ D.wires[l - off].2 = p := by
      rcases wireHas_ok P K WB D.wires hW (l - off) hj hk _ hh with e | e
      · exact .inl (pkey_inj P _ _ (hpos _ hmem.1).1 hp e)
      · exact .inr (pkey_inj P _ _ (hpos _ hmem.2).1 hp e)
    have hloc := locate_wire D hnd (l - off) hj p hpw
    unfold Diagram.label Diagram.okPort
    rw [hloc]
    dsimp only
    refine ⟨⟨by omega, by simpa using hj⟩, ?_⟩
    rcases hpw with e | e
    · exact e ▸ hmem.1
    · exact e ▸ hmem.2

/-! ## Increasing lists, masks and tables -/

/-- Every entry above `a` and above the one before it. -/
def incr : ℕ → List ℕ → Bool
  | _, [] => true
  | a, b :: l => decide (a < b) && incr b l

/-- The entries increase. -/
def incrL : List ℕ → Bool
  | [] => true
  | a :: l => incr a l

theorem incr_ok : ∀ (a : ℕ) (l : List ℕ), incr a l = true → (∀ x ∈ l, a < x) ∧ l.Nodup
  | _, [], _ => ⟨by simp, List.nodup_nil⟩
  | a, b :: l, h => by
    simp only [incr, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨h1, h2⟩ := incr_ok b l h.2
    refine ⟨fun x hx => ?_, List.nodup_cons.2 ⟨fun hb => absurd (h1 b hb) (lt_irrefl b), h2⟩⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact h.1
    · exact h.1.trans (h1 x hx)

theorem incrL_ok (l : List ℕ) (h : incrL l = true) : l.Nodup := by
  cases l with
  | nil => exact List.nodup_nil
  | cons a l =>
    obtain ⟨h1, h2⟩ := incr_ok a l h
    exact List.nodup_cons.2 ⟨fun ha => absurd (h1 a ha) (lt_irrefl a), h2⟩

/-- No number twice, none of the mask `m`; each mask reduced before its next use. -/
def nodupB : List ℕ → ℕ → Bool
  | [], _ => true
  | l :: ls, m => !m.testBit l && match m ||| 2 ^ l with
    | 0 => nodupB ls 0
    | m' + 1 => nodupB ls (m' + 1)

theorem nodupB_ok : ∀ (ls : List ℕ) (m : ℕ), nodupB ls m = true →
    ls.Nodup ∧ ∀ l ∈ ls, m.testBit l = false
  | [], _, _ => ⟨List.nodup_nil, by simp⟩
  | l :: ls, m, h => by
    simp only [nodupB, Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨h0, h⟩ := h
    have h' : nodupB ls (m ||| 2 ^ l) = true := by
      revert h; generalize m ||| 2 ^ l = m'; cases m' <;> exact id
    obtain ⟨h1, h2⟩ := nodupB_ok ls _ h'
    refine ⟨List.nodup_cons.2 ⟨fun hl => ?_, h1⟩, fun l' hl' => ?_⟩
    · have := h2 l hl
      rw [Nat.testBit_lor, Nat.testBit_two_pow] at this; simp at this
    · rcases List.mem_cons.1 hl' with rfl | hl'
      · exact h0
      · have := h2 l' hl'
        rw [Nat.testBit_lor] at this; simp only [Bool.or_eq_false_iff] at this; exact this.1

/-- Entry `l` of the table `T` of `b`-bit entries. -/
def ent (b T l : ℕ) : ℕ := (T >>> (b * l)) % 2 ^ b

/-- `l` is a boundary slot below `nb` or a wire label from `off` below `off + W`. -/
def inQ (nb off W l : ℕ) : Bool := decide (l < nb) || (decide (off ≤ l) && decide (l < off + W))

/-- `L`'s labels increase, the boundary slots and wire labels among them have their entries
of `T`, and all of them are there. -/
def labsOK (b T nb off W : ℕ) (L : List (ℕ × ℕ)) : Bool :=
  incrL (L.map Prod.fst) && L.all (fun q => !inQ nb off W q.1 || q.2 == ent b T q.1) &&
    ((L.map Prod.fst).filter (inQ nb off W)).length == nb + W

theorem lookup_of_mem : ∀ (L : List (ℕ × ℕ)), (L.map Prod.fst).Nodup → ∀ l d, (l, d) ∈ L →
    L.lookup l = some d
  | [], _, _, _, h => absurd h (by simp)
  | (a, c) :: L, hn, l, d, h => by
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 h with e | e
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj e
      simp
    · have hla : l ≠ a := fun h' => hn.1 (h' ▸ List.mem_map.2 ⟨_, e, rfl⟩)
      have hb : (l == a) = false := by simp [hla]
      rw [List.lookup_cons, hb]
      exact lookup_of_mem L hn.2 l d e

theorem labs_lookup (b T nb off W : ℕ) (L : List (ℕ × ℕ)) (hnb : nb ≤ off)
    (h : labsOK b T nb off W L = true) (l : ℕ) (hl : inQ nb off W l = true) :
    L.lookup l = some (ent b T l) := by
  simp only [labsOK, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨hinc, hall⟩, hcnt⟩ := h
  have hnd := incrL_ok _ hinc
  let Q := List.range nb ++ (List.range W).map (off + ·)
  have hQ : Q.Nodup := by
    refine List.nodup_append.2 ⟨List.nodup_range, (List.nodup_range).map (fun _ _ h => by omega),
      fun x hx y hy e => ?_⟩
    simp only [List.mem_range, List.mem_map] at hx hy
    obtain ⟨j, -, rfl⟩ := hy
    omega
  have hsub : (L.map Prod.fst).filter (inQ nb off W) ⊆ Q := by
    intro x hx
    have := (List.mem_filter.1 hx).2
    simp only [inQ, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at this
    rcases this with h1 | ⟨h1, h2⟩
    · exact List.mem_append_left _ (List.mem_range.2 h1)
    · exact List.mem_append_right _ (List.mem_map.2 ⟨x - off, List.mem_range.2 (by omega),
        by omega⟩)
  have hperm := ((hnd.filter _).subperm hsub).perm_of_length_le (by simp [Q, hcnt])
  have hlQ : l ∈ Q := by
    simp only [inQ, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hl
    rcases hl with h1 | ⟨h1, h2⟩
    · exact List.mem_append_left _ (List.mem_range.2 h1)
    · exact List.mem_append_right _ (List.mem_map.2 ⟨l - off, List.mem_range.2 (by omega),
        by omega⟩)
  have hlK := (List.mem_filter.1 (hperm.mem_iff.2 hlQ)).1
  obtain ⟨⟨l', d⟩, hmem, hl'⟩ := List.mem_map.1 hlK
  simp only at hl'
  subst hl'
  have hd := List.all_eq_true.1 hall _ hmem
  simp only [hl, Bool.not_true, Bool.false_or, beq_iff_eq] at hd
  rw [lookup_of_mem L hnd _ _ hmem, hd]

/-! ## The bridge -/

/-- The legs of `n`: outputs, then inputs. -/
def legs (n : DNode) : List Port :=
  (List.range n.outs.length).map (fun k => ⟨n.id, true, k⟩) ++
    (List.range n.ins.length).map (fun k => ⟨n.id, false, k⟩)

theorem legs_nodup (D : Diagram) (hid : (D.nodes.map (·.id)).Nodup) :
    (D.nodes.flatMap legs).Nodup := by
  refine List.nodup_flatMap.2 ⟨fun n _ => ?_, ?_⟩
  · refine List.nodup_append.2 ⟨List.nodup_range.map fun _ _ h => by simpa using h,
      List.nodup_range.map fun _ _ h => by simpa using h, fun x hx y hy e => ?_⟩
    simp only [List.mem_map] at hx hy
    obtain ⟨_, -, rfl⟩ := hx
    obtain ⟨_, -, rfl⟩ := hy
    simp at e
  · refine (List.pairwise_map.1 hid).imp fun {a b} hab => ?_
    show List.Disjoint (legs a) (legs b)
    intro x hx hy
    unfold legs at hx hy
    simp only [List.mem_append, List.mem_map] at hx hy
    have ha : x.node = a.id := by rcases hx with ⟨_, -, rfl⟩ | ⟨_, -, rfl⟩ <;> rfl
    have hb : x.node = b.id := by rcases hy with ⟨_, -, rfl⟩ | ⟨_, -, rfl⟩ <;> rfl
    exact hab (ha.symm.trans hb)

theorem portDim_leg (D : Diagram) (hid : (D.nodes.map (·.id)).Nodup) (i : ℕ)
    (hi : i < D.nodes.length) (o : Bool) (k : ℕ) :
    D.portDim ⟨D.nodes[i].id, o, k⟩ =
      if o then D.nodes[i].outs.getD k 0 else D.nodes[i].ins.getD k 0 := by
  have hf : D.nodes.find? (fun n => decide (n.id = D.nodes[i].id)) = some D.nodes[i] := by
    refine List.find?_eq_some_iff_getElem.2 ⟨by simp, i, hi, rfl, fun j hj => ?_⟩
    rw [Bool.not_eq_true', decide_eq_false_iff_not]
    intro e
    have := (hid.getElem_inj_iff (i := j) (j := i) (hi := by simp; omega)
      (hj := by simpa using hi)).1 (by simpa using e)
    omega
  unfold Diagram.portDim
  simp only [hf]

/-- The legs of `n` at the labels of `m`, each with its entry of `T`. -/
def nodeOKD (P K b off WB T : ℕ) (D : Diagram) (n : DNode) (m : Node) : Bool :=
  decide (m.kind = n.kind) && m.dim == n.dim && m.ph == n.ph &&
    m.outs.length == n.outs.length && m.ins.length == n.ins.length &&
    ((List.range n.outs.length).zip m.outs).all (fun q =>
      legOK P K off WB D ⟨n.id, true, q.1⟩ q.2 && n.outs.getD q.1 0 == ent b T q.2) &&
    ((List.range n.ins.length).zip m.ins).all (fun q =>
      legOK P K off WB D ⟨n.id, false, q.1⟩ q.2 && n.ins.getD q.1 0 == ent b T q.2)

/-- `N` is `D.toNet off` and `D.check L off` holds. -/
def bridgeD (P K b off WB T : ℕ) (D : Diagram) (N : Net) (L : List (ℕ × ℕ)) : Bool :=
  decide (D.bnd.length ≤ off) && incrL (D.nodes.map (·.id)) &&
    D.nodes.all (fun n => decide (n.outs.length ≤ P) && decide (n.ins.length ≤ P)) &&
    (ends D).all (fun p => decide (p.pos < P) && decide (pkey P p < 2 ^ K)) &&
    nodupB ((ends D).map (pkey P)) 0 && packW P K D.wires == WB &&
    (D.nodes.map fun n => n.outs.length + n.ins.length).sum == (ends D).length &&
    labsOK b T D.bnd.length off D.wires.length L &&
    N.int == (List.range D.wires.length).map (off + ·) && N.bnd == List.range D.bnd.length &&
    N.nodes.length == D.nodes.length &&
    (D.nodes.zip N.nodes).all fun q => nodeOKD P K b off WB T D q.1 q.2

theorem bridgeD_ok (P K b off WB T : ℕ) (D : Diagram) (N : Net) (L : List (ℕ × ℕ))
    (h : bridgeD P K b off WB T D N L = true) :
    D.check L off = true ∧ D.toNet off = N := by
  unfold bridgeD at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hoff, hinc⟩, hP⟩, hpos⟩, hnd⟩, hW⟩, hcnt⟩, hlabs⟩, hint⟩, hbnd⟩, hlen⟩,
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
      ((D.label off ⟨D.nodes[i].id, o, k⟩ =
        (if o then (N.nodes[i]'(by omega)).outs.getD k 0 else (N.nodes[i]'(by omega)).ins.getD k 0) ∧
      D.okPort ⟨D.nodes[i].id, o, k⟩ = true) ∧ ⟨D.nodes[i].id, o, k⟩ ∈ ends D) ∧
      D.portDim ⟨D.nodes[i].id, o, k⟩ = ent b T (D.label off ⟨D.nodes[i].id, o, k⟩) := by
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
      rw [List.getD_eq_getElem (N.nodes[i]'(by omega)).outs 0 (by omega)]
      refine ⟨hl, ?_⟩
      rw [hl.1.1]
      exact this.2
    | false =>
      simp only [Bool.false_eq_true, ite_false] at hk ⊢
      have := all_zip_get _ _ _ hai k (by simpa using hk) (by omega)
      simp only [List.getElem_range, Bool.and_eq_true, beq_iff_eq] at this
      have hl := locate_of_leg P K off WB D hnd' hpos' hW _ (by simp; omega) _ this.1
      rw [List.getD_eq_getElem (N.nodes[i]'(by omega)).ins 0 (by omega)]
      refine ⟨hl, ?_⟩
      rw [hl.1.1]
      exact this.2
  -- every boundary port and wire end is a leg
  have hsub : D.nodes.flatMap legs ⊆ ends D := by
    intro p hp
    obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 hp
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
    unfold legs at hpn
    simp only [List.mem_append, List.mem_map, List.mem_range] at hpn
    rcases hpn with ⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩
    · exact (hleg i hi true k (by simpa using hk)).1.2
    · exact (hleg i hi false k (by simpa using hk)).1.2
  have hperm := ((legs_nodup D hid).subperm hsub).perm_of_length_le (by
    rw [← hcnt, List.length_flatMap]; simp [legs])
  have hdim : ∀ p ∈ ends D, D.portDim p = ent b T (D.label off p) := by
    intro p hp
    obtain ⟨n, hn, hpn⟩ := List.mem_flatMap.1 (hperm.mem_iff.2 hp)
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
    unfold legs at hpn
    simp only [List.mem_append, List.mem_map, List.mem_range] at hpn
    rcases hpn with ⟨k, hk, rfl⟩ | ⟨k, hk, rfl⟩
    · exact (hleg i hi true k (by simpa using hk)).2
    · exact (hleg i hi false k (by simpa using hk)).2
  have hwlab : ∀ j (hj : j < D.wires.length) (p : Port), (D.wires[j].1 = p ∨ D.wires[j].2 = p) →
      D.label off p = off + j := fun j hj p hpw => by
    unfold Diagram.label; rw [locate_wire D hnd' j hj p hpw]
  have hwdim : ∀ j (hj : j < D.wires.length),
      D.portDim D.wires[j].1 = ent b T (off + j) ∧ D.portDim D.wires[j].2 = ent b T (off + j) :=
    fun j hj => by
      have hm := wire_ends_mem D j hj
      rw [hdim _ hm.1, hdim _ hm.2, hwlab j hj _ (.inl rfl), hwlab j hj _ (.inr rfl)]
      exact ⟨rfl, rfl⟩
  have hbdim : ∀ i (hi : i < D.bnd.length), D.portDim D.bnd[i] = ent b T i := fun i hi => by
    rw [hdim _ (List.mem_append_left _ (List.getElem_mem hi))]
    unfold Diagram.label; rw [locate_bnd D hnd' i hi]
  have hlk : ∀ l, inQ D.bnd.length off D.wires.length l = true → L.lookup l = some (ent b T l) :=
    labs_lookup b T _ off _ L hoff hlabs
  refine ⟨?_, ?_⟩
  · unfold Diagram.check
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range]
    refine ⟨⟨⟨⟨hoff, fun n hn => ?_⟩, fun w hw => ?_⟩, fun j hj => ?_⟩, ?_⟩
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hn
      refine ⟨fun k hk => (hleg i hi true k (by simpa using hk)).1.1.2,
        fun k hk => (hleg i hi false k (by simpa using hk)).1.1.2⟩
    · obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hw
      rw [(hwdim j hj).1, (hwdim j hj).2]
    · rw [hlk _ (by simp [inQ, hj]), Diagram.wireDim, List.getElem?_eq_getElem hj]
      simp [(hwdim j hj).1]
    · apply List.ext_getElem
      · simp [Diagram.bdims]
      · intro i h1 h2
        have hi : i < D.bnd.length := by simpa using h1
        simp only [List.getElem_map, List.getElem_range, Diagram.bdims]
        rw [hlk _ (by simp [inQ, hi]), hbdim i hi]
  · rw [show N = ⟨N.nodes, N.int, N.bnd⟩ from rfl]
    unfold Diagram.toNet
    rw [hint, hbnd]
    congr 1
    apply List.ext_getElem (by simp [hlen])
    intro i h1 h2
    rw [List.getElem_map]
    have hi : i < D.nodes.length := by simpa using h1
    have hn := hnode i hi
    unfold nodeOKD at hn
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
      have := (hleg i hi true k (by simpa using hk1)).1.1.1
      simp only [ite_true] at this
      rw [this, List.getD_eq_getElem _ _ hk2]
    · apply List.ext_getElem (by simp [hi'])
      intro k hk1 hk2
      simp only [List.getElem_map, List.getElem_range]
      have := (hleg i hi false k (by simpa using hk1)).1.1.1
      simp only [Bool.false_eq_true, ite_false] at this
      rw [this, List.getD_eq_getElem _ _ hk2]

end DBridge
end ArchytasZX
