/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.Bridge
import ArchytasZX.WZ

/-!
# Diagrams as constrained sums over legs

`Diagram.evalG` sums over a value on every leg, each over its dimension, the product of the node
tensors where every wire's two legs agree and every boundary leg reads its slot. Leg `p` is the
label `p.code`; slot `i` is the odd label `2 i + 1`. `evalG_eq` states it is the sum over wire
values that `Diagram.eval` takes.
-/

namespace ArchytasZX

open Finset

/-- A port as an even label. -/
def Port.code (p : Port) : ℕ := 2 * Nat.pair p.node (2 * p.pos + if p.out then 1 else 0)

/-- The port of an even label. -/
def Port.decode (c : ℕ) : Port :=
  let u := (c / 2).unpair
  ⟨u.1, u.2 % 2 = 1, u.2 / 2⟩

theorem Port.decode_code (p : Port) : Port.decode p.code = p := by
  obtain ⟨n, o, q⟩ := p
  cases o <;> simp [Port.code, Port.decode, Nat.unpair_pair]; omega

theorem Port.code_inj : Function.Injective Port.code := fun p q h => by
  rw [← Port.decode_code p, ← Port.decode_code q, h]

theorem Port.code_even (p : Port) : p.code % 2 = 0 := by simp [Port.code]

/-- Summing labels `f x` for `x ∈ ls` is summing `ls`, for integrands that agree when the value
at `f x` is the value at `x`. -/
theorem sumOver_map_on (D D' : ℕ → ℕ) (f : ℕ → ℕ) (ls : List ℕ) (hnd : ls.Nodup)
    (hf : ∀ x ∈ ls, ∀ y ∈ ls, f x = f y → x = y) (hd : ∀ x ∈ ls, D (f x) = D' x)
    (g h : (ℕ → ℕ) → ℂ) (σ σ' : ℕ → ℕ)
    (H : ∀ τ τ', (∀ x, x ∉ ls.map f → τ x = σ x) → (∀ y, y ∉ ls → τ' y = σ' y) →
      (∀ x ∈ ls, τ' x = τ (f x)) → g τ = h τ')
    (h₀ : ∀ x ∈ ls, σ' x = σ (f x)) :
    sumOver D (ls.map f) g σ = sumOver D' ls h σ' := by
  induction ls generalizing σ σ' with
  | nil => exact H σ σ' (fun _ _ => rfl) (fun _ _ => rfl) (by simp)
  | cons l ls ih =>
    simp only [List.nodup_cons] at hnd
    simp only [List.map_cons, sumOver]
    rw [hd l List.mem_cons_self]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hl : ∀ x ∈ ls, f x ≠ f l := fun x hx e => hnd.1
      (hf x (List.mem_cons_of_mem _ hx) l List.mem_cons_self e ▸ hx)
    have hfl : f l ∉ ls.map f := by
      simp only [List.mem_map, not_exists, not_and]; exact fun x hx e => hl x hx e
    refine ih hnd.2 (fun x hx y hy => hf x (List.mem_cons_of_mem _ hx) y
      (List.mem_cons_of_mem _ hy)) (fun x hx => hd x (List.mem_cons_of_mem _ hx)) _ _
      (fun τ τ' hτ hτ' hτf => H τ τ' (fun x hx => ?_) (fun y hy => ?_) (fun x hx => ?_))
      (fun x hx => ?_)
    · have h1 : x ≠ f l := fun e => hx (by simp [e])
      have h2 : x ∉ ls.map f := fun e => hx (by simp only [List.map_cons]; exact List.mem_cons_of_mem _ e)
      rw [hτ x h2, Function.update_of_ne h1]
    · have h1 : y ≠ l := fun e => hy (e ▸ List.mem_cons_self)
      rw [hτ' y (fun e => hy (List.mem_cons_of_mem _ e)), Function.update_of_ne h1]
    · rcases List.mem_cons.1 hx with rfl | hx
      · rw [hτ' x hnd.1, hτ (f x) hfl]; simp
      · exact hτf x hx
    · rw [Function.update_of_ne (fun e => hnd.1 (by subst e; exact hx)), Function.update_of_ne (hl x hx)]
      exact h₀ x (List.mem_cons_of_mem _ hx)

/-- With keys once each, every pair agreeing is every key agreeing with its value. -/
theorem pins_iff (ps : List (ℕ × ℕ)) (hnd : (ps.map Prod.fst).Nodup) (ν : ℕ → ℕ) :
    (∀ q ∈ ps, ν q.1 = ν q.2) ↔ ∀ l ∈ ps.map Prod.fst, ν l = ν ((ps.lookup l).getD l) := by
  induction ps with
  | nil => simp
  | cons q ps ih =>
    obtain ⟨a, b⟩ := q
    simp only [List.map_cons, List.nodup_cons, List.mem_map, not_exists, not_and] at hnd
    have hrest : ∀ l ∈ ps.map Prod.fst, (((a, b) :: ps).lookup l).getD l = (ps.lookup l).getD l := by
      intro l hl
      have : l ≠ a := fun e => by
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hl
        exact hnd.1 x hx e
      have hb : (l == a) = false := by simpa using this
      simp [List.lookup, hb]
    simp only [List.mem_cons, forall_eq_or_imp, List.map_cons]
    rw [ih hnd.2]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨by simp [List.lookup, h1], fun l hl => by rw [hrest l hl]; exact h2 l hl⟩
    · rintro ⟨h1, h2⟩
      refine ⟨by simpa [List.lookup] using h1, fun l hl => by rw [← hrest l hl]; exact h2 l hl⟩

/-- The slot values at the odd labels. -/
def slots (σ : ℕ → ℕ) (c : ℕ) : ℕ := if c % 2 = 1 then σ (c / 2) else 0

/-- A node's legs, outputs first. -/
def DNode.legs (n : DNode) : List Port :=
  (List.range n.outs.length).map (⟨n.id, true, ·⟩) ++ (List.range n.ins.length).map (⟨n.id, false, ·⟩)

namespace Diagram

variable (D : Diagram)

/-- Every node's legs. -/
def legs : List Port := D.nodes.flatMap DNode.legs

/-- Each wire's second leg pinned to its first, each boundary leg to its slot's odd label. -/
def pins : List (ℕ × ℕ) :=
  D.wires.map (fun w => (w.2.code, w.1.code)) ++ D.bnd.zipIdx.map fun (p, i) => (p.code, 2 * i + 1)

/-- The sum over every leg's value below `δ` of the node values `nv`, where every pin holds. -/
noncomputable def evalG (δ : Port → ℕ) (nv : DNode → (Port → ℕ) → ℂ) (σ : ℕ → ℕ) : ℂ :=
  sumOver (fun c => δ (Port.decode c)) (D.legs.map Port.code)
    (fun ν => if ∀ q ∈ D.pins, ν q.1 = ν q.2 then
      (D.nodes.map fun n => nv n (ν ∘ Port.code)).prod else 0) (slots σ)

/-- The legs are the wire ends and boundary ports, each once; wire ends have one dimension. -/
def checkC : Bool :=
  (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).isPerm D.legs && decide D.legs.Nodup &&
    D.wires.all fun w => D.portDim w.1 = D.portDim w.2

end Diagram

theorem lookup_of_mem (ps : List (ℕ × ℕ)) (hnd : (ps.map Prod.fst).Nodup) (q : ℕ × ℕ)
    (hq : q ∈ ps) : ps.lookup q.1 = some q.2 := by
  induction ps with
  | nil => simp at hq
  | cons r ps ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map, not_exists, not_and] at hnd
    rcases List.mem_cons.1 hq with rfl | hq
    · simp [List.lookup]
    · have : q.1 ≠ r.1 := fun e => hnd.1 q hq e
      have hb : (q.1 == r.1) = false := by simpa using this
      simp [List.lookup, hb, ih hnd.2 hq]

namespace Diagram

variable (D : Diagram)

theorem value_wire (hnd : (D.wires.map (·.1) ++ D.wires.map (·.2)).Nodup) {p : Port}
    (hb : p ∉ D.bnd) (σ τ : ℕ → ℕ) (j : ℕ) (hj : j < D.wires.length)
    (hp : D.wires[j].1 = p ∨ D.wires[j].2 = p) : D.value σ τ p = τ j := by
  have hf : D.wires.findIdx? (fun w => decide (w.1 = p ∨ w.2 = p)) = some j := by
    rw [List.findIdx?_eq_some_iff_getElem]
    refine ⟨hj, by simpa using hp, fun i hij h => ?_⟩
    have hi : i < D.wires.length := by omega
    have hne : i ≠ j := by omega
    rw [List.nodup_append] at hnd
    obtain ⟨n1, n2, dis⟩ := hnd
    have g1 : ∀ a b (ha : a < D.wires.length) (hb : b < D.wires.length),
        D.wires[a].1 = D.wires[b].1 → a = b := fun a b ha hb e => by
      have := (List.Nodup.getElem_inj_iff n1 (i := a) (j := b) (hi := by simpa using ha)
        (hj := by simpa using hb)).1 (by simpa using e)
      exact this
    have g2 : ∀ a b (ha : a < D.wires.length) (hb : b < D.wires.length),
        D.wires[a].2 = D.wires[b].2 → a = b := fun a b ha hb e => by
      exact (List.Nodup.getElem_inj_iff n2 (i := a) (j := b) (hi := by simpa using ha)
        (hj := by simpa using hb)).1 (by simpa using e)
    have g3 : ∀ a b (ha : a < D.wires.length) (hb : b < D.wires.length),
        D.wires[a].1 ≠ D.wires[b].2 := fun a b ha hb e =>
      dis _ (List.mem_map.2 ⟨_, List.getElem_mem ha, rfl⟩) _
        (List.mem_map.2 ⟨_, List.getElem_mem hb, rfl⟩) e
    simp only [decide_eq_true_eq] at h
    rcases h with h | h <;> rcases hp with hp | hp
    · exact hne (g1 i j hi hj (h.trans hp.symm))
    · exact g3 i j hi hj (h.trans hp.symm)
    · exact g3 j i hj hi (hp.trans h.symm)
    · exact hne (g2 i j hi hj (h.trans hp.symm))
  unfold value locate
  rw [List.idxOf?_eq_none_iff.2 hb, hf]
  rfl

theorem value_bnd (hnd : D.bnd.Nodup) (σ τ : ℕ → ℕ) (i : ℕ) (hi : i < D.bnd.length) :
    D.value σ τ D.bnd[i] = σ i := by
  have : D.bnd.idxOf? D.bnd[i] = some i := by
    rw [List.idxOf?_eq_some_iff]
    exact ⟨hi, rfl, fun j hj e => by
      have := (List.Nodup.getElem_inj_iff hnd (i := j) (j := i) (hi := by omega) (hj := hi)).1 e
      omega⟩
  unfold value locate
  rw [this]

end Diagram

namespace Diagram

variable (D : Diagram)

/-- The constrained sum is the sum over wire values with every leg read through `value`. -/
theorem evalG_eq (δ : Port → ℕ) (nv : DNode → (Port → ℕ) → ℂ)
    (hnv : ∀ n ∈ D.nodes, ∀ f g : Port → ℕ, (∀ p ∈ n.legs, f p = g p) → nv n f = nv n g)
    (hP : (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).Perm D.legs)
    (hN : D.legs.Nodup) (hδ : ∀ w ∈ D.wires, δ w.1 = δ w.2)
    (σ : ℕ → ℕ) (hσ : ∀ i (hi : i < D.bnd.length), σ i < δ D.bnd[i]) :
    D.evalG δ nv σ = sumOver (fun j => (D.wires[j]?.map fun w => δ w.1).getD 0)
      (List.range D.wires.length) (fun τ => (D.nodes.map fun n => nv n (D.value σ τ)).prod)
      (fun _ => 0) := by
  have hall := hP.nodup_iff.2 hN
  rw [List.nodup_append] at hall
  obtain ⟨n1, n23, d1⟩ := hall
  have n23' := n23
  rw [List.nodup_append] at n23'
  obtain ⟨n2, n3, d23⟩ := n23'
  have n12 : (D.wires.map (·.1) ++ D.wires.map (·.2)).Nodup :=
    List.nodup_append.2 ⟨n1, n2, fun a ha b hb => d1 a ha b (List.mem_append_left _ hb)⟩
  set ls := (D.wires.map (·.2) ++ D.bnd).map Port.code with hls_def
  have hls : ls.Nodup := n23.map Port.code_inj
  have hkeys : D.pins.map Prod.fst = ls := by
    simp only [pins, hls_def, List.map_append, List.map_map]
    congr 1
    refine List.ext_getElem (by simp) fun i h1 h2 => ?_
    simp
  set f := fun x => (D.pins.lookup x).getD x with hf_def
  have pinW : ∀ w ∈ D.wires, f w.2.code = w.1.code := fun w hw => by
    simp only [hf_def]
    rw [lookup_of_mem D.pins (hkeys ▸ hls) (w.2.code, w.1.code)
      (List.mem_append_left _ (List.mem_map.2 ⟨w, hw, rfl⟩))]
    rfl
  have pinB : ∀ i (hi : i < D.bnd.length), f D.bnd[i].code = 2 * i + 1 := fun i hi => by
    simp only [hf_def]
    rw [lookup_of_mem D.pins (hkeys ▸ hls) (D.bnd[i].code, 2 * i + 1)
      (List.mem_append_right _ (List.mem_map.2 ⟨(D.bnd[i], i),
        List.mem_zipIdx_iff_getElem?.2 (by simp), rfl⟩))]
    rfl
  have memls : ∀ l ∈ ls, (∃ w ∈ D.wires, l = w.2.code) ∨
      (∃ i, ∃ hi : i < D.bnd.length, l = D.bnd[i].code) := fun l hl => by
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hl
    rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hp
      exact Or.inl ⟨w, hw, rfl⟩
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
      exact Or.inr ⟨i, hi, rfl⟩
  have odd_not : ∀ x ∈ (D.wires.map (·.1) ++ (D.wires.map (·.2) ++ D.bnd)).map Port.code,
      x % 2 = 0 := fun x hx => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 hx
    exact Port.code_even p
  unfold evalG
  rw [sumOver_perm _ (hP.map Port.code).symm, List.map_append, sumOver_append]
  have hF : (fun ν : ℕ → ℕ => if ∀ q ∈ D.pins, ν q.1 = ν q.2 then
      (D.nodes.map fun n => nv n (ν ∘ Port.code)).prod else 0) =
      fun ν => if ∀ l ∈ ls, ν l = ν (f l) then
      (D.nodes.map fun n => nv n (ν ∘ Port.code)).prod else 0 := by
    funext ν
    refine if_congr ((pins_iff D.pins (hkeys ▸ hls) ν).trans ?_) rfl rfl
    rw [hkeys]
  rw [hF]
  set fmap : ℕ → ℕ := fun j => (D.wires[j]?.map fun w : Port × Port => w.1.code).getD 0 with hfmap
  have fmap_even : ∀ j, fmap j % 2 = 0 := fun j => by
    simp only [hfmap]; cases D.wires[j]? <;> simp [Port.code_even]
  have fmap_get : ∀ j (hj : j < D.wires.length), fmap j = D.wires[j].1.code := fun j hj => by
    simp [hfmap, List.getElem?_eq_getElem hj]
  have odd_out : ∀ i, 2 * i + 1 ∉ (List.range D.wires.length).map fmap := fun i hm => by
    obtain ⟨j, -, e⟩ := List.mem_map.1 hm
    have := fmap_even j; omega
  have hW1 : (D.wires.map (·.1)).map Port.code = (List.range D.wires.length).map fmap := by
    refine List.ext_getElem (by simp) fun i h1 h2 => ?_
    simp only [List.getElem_map, List.getElem_range]
    rw [fmap_get i (by simpa using h1)]
  rw [hW1]
  have hpin : ∀ τ : ℕ → ℕ, (∀ x ∉ (List.range D.wires.length).map fmap, τ x = slots σ x) →
      (∀ x ∈ (List.range D.wires.length).map fmap, τ x < δ (Port.decode x)) →
      sumOver (fun c => δ (Port.decode c)) ls (fun ν => if ∀ l ∈ ls, ν l = ν (f l) then
        (D.nodes.map fun n => nv n (ν ∘ Port.code)).prod else 0) τ =
      (D.nodes.map fun n => nv n ((fun x => if x ∈ ls then τ (f x) else τ x) ∘ Port.code)).prod := by
    intro τ hτ hr
    refine sumOver_pin _ ls hls f (fun l hl => ?_) _ τ (fun l hl => ?_)
    · rcases memls l hl with ⟨w, hw, rfl⟩ | ⟨i, hi, rfl⟩
      · rw [pinW w hw]
        intro hm
        obtain ⟨q, hq, hqc⟩ := List.mem_map.1 hm
        have := Port.code_inj hqc
        exact d1 _ (List.mem_map.2 ⟨w, hw, rfl⟩) _ hq this.symm
      · rw [pinB i hi]
        intro hm
        obtain ⟨q, -, hqc⟩ := List.mem_map.1 hm
        have := Port.code_even q
        omega
    · rcases memls l hl with ⟨w, hw, rfl⟩ | ⟨i, hi, rfl⟩
      · rw [pinW w hw]
        have h1 := hr w.1.code (by
          rw [← hW1]; exact List.mem_map.2 ⟨w.1, List.mem_map.2 ⟨w, hw, rfl⟩, rfl⟩)
        simpa [Port.decode_code, hδ w hw] using h1
      · rw [pinB i hi, hτ _ (odd_out i)]
        simp only [slots, Port.decode_code]
        rw [if_pos (by omega), show (2 * i + 1) / 2 = i by omega]
        exact hσ i hi
  rw [sumOver_congr _ _ _ _ _ fun τ hτ hr => hpin τ hτ hr]
  refine sumOver_map_on _ _ fmap _ List.nodup_range (fun x hx y hy e => ?_) (fun x hx => ?_)
    _ _ _ _ (fun τ τ' hτ hτ' hl => ?_) (fun x hx => ?_)
  · have hx' := List.mem_range.1 hx
    have hy' := List.mem_range.1 hy
    rw [fmap_get x hx', fmap_get y hy'] at e
    have e' := Port.code_inj e
    exact (List.Nodup.getElem_inj_iff n1 (i := x) (j := y) (hi := by simpa using hx')
      (hj := by simpa using hy')).1 (by simpa using e')
  · have hx' := List.mem_range.1 hx
    simp [fmap_get x hx', List.getElem?_eq_getElem hx', Port.decode_code]
  · congr 1
    refine List.map_congr_left fun n hn => hnv n hn _ _ fun p hp => ?_
    simp only [Function.comp]
    have hpL : p ∈ D.legs := List.mem_flatMap.2 ⟨n, hn, hp⟩
    rcases List.mem_append.1 (hP.mem_iff.2 hpL) with h | h
    · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h
      obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hw
      have hnl : D.wires[j].1.code ∉ ls := fun hm => by
        obtain ⟨q, hq, hqc⟩ := List.mem_map.1 hm
        have := Port.code_inj hqc
        exact d1 _ h _ hq this.symm
      rw [if_neg hnl, D.value_wire n12 (fun hb => d1 _ h _ (List.mem_append_right _ hb) rfl)
        σ τ' j hj (Or.inl rfl), hl j (List.mem_range.2 hj), fmap_get j hj]
    · have hin : p.code ∈ ls := List.mem_map.2 ⟨p, h, rfl⟩
      rw [if_pos hin]
      rcases List.mem_append.1 h with h | h
      · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h
        obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hw
        rw [pinW _ (List.getElem_mem hj), D.value_wire n12 (fun hb => d23 _ h _ hb rfl)
          σ τ' j hj (Or.inr rfl), hl j (List.mem_range.2 hj), fmap_get j hj]
      · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 h
        rw [pinB i hi, hτ _ (odd_out i), D.value_bnd n3 σ τ' i hi]
        simp only [slots]
        rw [if_pos (by omega), show (2 * i + 1) / 2 = i by omega]
  · have := fmap_even x
    simp only [slots]
    split_ifs <;> first | rfl | omega

end Diagram

end ArchytasZX
