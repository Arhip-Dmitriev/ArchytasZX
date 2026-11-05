/-
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0.
-/
import ArchytasZX.NodeBox

/-!
# Instantiating every box

`evalD_expand`: under `WF`, the expansion is box-free and its digit sum is the boxed diagram's,
the slots read through `bmap`, the composite of the per-box slot readings.
-/

namespace ArchytasZX

namespace Diagram

theorem Linked.take {D : Diagram} {s : Option ℕ} {L : List DBox} (h : Linked D s L) (n : ℕ) :
    Linked D s (L.take n) := by
  refine ⟨fun y hy => h.1 y (List.mem_of_mem_take hy), fun i hi => ?_⟩
  have hi' : i < L.length := by simp at hi; omega
  have := h.2 i hi'
  simp only [List.getElem_take]
  rw [this]
  cases i with
  | zero => simp
  | succ i => simp [List.getElem_take]

theorem Linked.nodup {D : Diagram} (hb : (D.boxes.map (·.id)).Nodup) :
    ∀ (n : ℕ) (L : List DBox), L.length = n → Linked D none L → L.Nodup := by
  intro n
  induction n with
  | zero => intro L hn _; rw [List.length_eq_zero_iff.1 hn]; exact List.nodup_nil
  | succ n ih =>
    intro L hn h
    obtain ⟨A, z, rfl⟩ := List.eq_nil_or_concat L |>.resolve_left (by rintro rfl; simp at hn)
    simp only [List.concat_eq_append] at *
    have hA : Linked D none A := by
      have := h.dropLast; simpa using this
    have hAn := ih A (by simp at hn; omega) hA
    refine List.nodup_append.2 ⟨hAn, List.nodup_singleton z, ?_⟩
    intro a ha c hc e
    rw [List.mem_singleton] at hc
    subst hc; subst e
    obtain ⟨j, hj, hja⟩ := List.mem_iff_getElem.1 ha
    have ht := h.take (j + 1)
    have hlast : ((A ++ [a]).take (j + 1)).getLast? = (A ++ [a]).getLast? := by
      rw [List.getLast?_concat, List.getLast?_eq_getElem?]
      simp only [List.length_take, List.length_append, List.length_singleton]
      rw [Nat.min_eq_left (by omega), List.getElem?_take, if_pos (by omega),
        show j + 1 - 1 = j by omega, List.getElem?_append_left hj, List.getElem?_eq_getElem hj, hja]
    have := Linked.eq_of_last hb _ _ _ rfl ht h (by simp) hlast
    have hl := congrArg List.length this
    simp at hl
    omega

theorem Linked.snoc {D : Diagram} {L : List DBox} (h : Linked D none L) {y z : DBox}
    (hl : L.getLast? = some y) (hz : z ∈ D.boxes) (hp : z.parent = some y.id) :
    Linked D none (L ++ [z]) := by
  refine ⟨fun x hx => ?_, fun i hi => ?_⟩
  · rcases List.mem_append.1 hx with h' | h'
    · exact h.1 x h'
    · rw [List.mem_singleton.1 h']; exact hz
  · simp only [List.length_append, List.length_singleton] at hi
    by_cases hiL : i < L.length
    · rw [List.getElem_append_left hiL, h.2 i hiL]
      apply Option.some.inj
      rw [← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem]
      cases i with
      | zero => simp
      | succ i =>
        simp only [List.getElem?_cons_succ, List.map_append, List.map_cons, List.map_nil]
        rw [List.getElem?_append_left (by simp; omega)]
    · have : i = L.length := by omega
      subst this
      rw [List.getElem_append_right (le_refl _)]
      simp only [Nat.sub_self, List.getElem_cons_zero, hp]
      obtain ⟨A, a, rfl⟩ := List.eq_nil_or_concat L |>.resolve_left (by rintro rfl; simp at hl)
      simp only [List.concat_eq_append, List.getLast?_concat, Option.some.injEq] at hl
      subst hl
      simp [List.getElem_append]

theorem Linked.of_chain_append {D : Diagram} {pr pr' : DBox → Bool} {N P : List DBox}
    (hN : ChainOK D pr none N) (hP : ChainOK D pr' (N.getLast?.map (·.id)) P) :
    Linked D none (N ++ P) := by
  refine ⟨fun y hy => ?_, fun i hi => ?_⟩
  · rcases List.mem_append.1 hy with h | h
    · exact (hN.1 y h).1
    · exact (hP.1 y h).1
  · simp only [List.length_append] at hi
    by_cases hiN : i < N.length
    · rw [List.getElem_append_left hiN, hN.2.2.1 i hiN]
      apply Option.some.inj
      rw [← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem]
      cases i with
      | zero => simp
      | succ i =>
        simp only [List.getElem?_cons_succ, List.map_append]
        rw [List.getElem?_append_left (by simp; omega)]
    · have hiP : i - N.length < P.length := by omega
      rw [List.getElem_append_right (by omega)]
      refine (hP.2.2.1 _ hiP).trans ?_
      apply Option.some.inj
      rw [← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem]
      by_cases h0 : i - N.length = 0
      · have hi0 : i = N.length := by omega
        subst hi0
        rw [h0]
        cases N using List.reverseRecOn with
        | nil => simp
        | append_singleton A a _ =>
          simp only [List.getElem?_cons_zero, List.getLast?_concat, Option.map_some,
            List.length_append, List.length_singleton, List.getElem?_cons_succ, List.map_append]
          rw [List.getElem?_append_left (by simp), List.getElem?_append_right (by simp)]
          simp
      · obtain ⟨j, hj⟩ : ∃ j, i - N.length = j + 1 := ⟨i - N.length - 1, by omega⟩
        rw [hj]
        obtain ⟨i', rfl⟩ : ∃ i', i = i' + 1 := ⟨i - 1, by omega⟩
        simp only [List.getElem?_cons_succ, List.map_append]
        rw [List.getElem?_append_right (by simp; omega)]
        simp only [List.length_map]
        congr 1
        omega

theorem exists_linked {D : Diagram} (hW : WF D) {y : DBox} (hy : y ∈ D.boxes) :
    ∃ L, Linked D none L ∧ L.getLast? = some y := by
  rcases hW.ne y hy with hn | hp
  · obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hn
    have hc := hW.nchain x (hW.nsub y hy x hx)
    have hyL := hW.nall x y hy hx
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hyL
    refine ⟨(D.nodeBoxes x).take (i + 1), (Linked.of_chain hc).take _, ?_⟩
    rw [List.getLast?_eq_getElem?]
    simp [Nat.min_eq_left (show i + 1 ≤ (D.nodeBoxes x).length by omega)]
  · obtain ⟨p, hp'⟩ := List.exists_mem_of_ne_nil _ hp
    have hpb := hW.pbnd y hy p hp'
    have hpl := bnd_legs hW hpb
    have hN := hW.nchain p.node (node_of_leg hpl)
    have hP := hW.pchain p hpb
    have hyL := hW.pall p hpb y hy hp'
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hyL
    have hL := Linked.of_chain_append hN hP
    refine ⟨(D.nodeBoxes p.node ++ D.portBoxes p).take ((D.nodeBoxes p.node).length + i + 1),
      hL.take _, ?_⟩
    rw [List.getLast?_eq_getElem?]
    simp only [List.length_take, List.length_append]
    rw [Nat.min_eq_left (by omega), List.getElem?_take]
    rw [if_pos (by omega)]
    rw [show (D.nodeBoxes p.node).length + i + 1 - 1 = (D.nodeBoxes p.node).length + i by omega,
      List.getElem?_append_right (by omega)]
    simp [List.getElem?_eq_getElem hi]

/-- A box no box has as parent exists in a diagram with boxes. -/
theorem exists_leaf {D : Diagram} (hW : WF D) (hne : D.boxes ≠ []) :
    ∃ b ∈ D.boxes, ∀ z ∈ D.boxes, z.parent ≠ some b.id := by
  by_contra hno
  push_neg at hno
  obtain ⟨y₀, hy₀⟩ := List.exists_mem_of_ne_nil _ hne
  obtain ⟨L₀, hL₀, hl₀⟩ := exists_linked hW hy₀
  have grow : ∀ m, ∃ L, Linked D none L ∧ L₀.length + m ≤ L.length ∧ ∃ y ∈ D.boxes, L.getLast? = some y := by
    intro m
    induction m with
    | zero => exact ⟨L₀, hL₀, le_refl _, y₀, hy₀, hl₀⟩
    | succ m ih =>
      obtain ⟨L, hL, hlen, y, hy, hl⟩ := ih
      obtain ⟨z, hz, hzp⟩ := hno y hy
      exact ⟨L ++ [z], hL.snoc hl hz hzp, by simp; omega, z, hz, by simp⟩
  obtain ⟨L, hL, hlen, -⟩ := grow (D.boxes.length + 1)
  have := (Linked.nodup hW.bids _ L rfl hL).length_le_of_subset fun y hy => hL.1 y hy
  omega

/-- The box `expandFuel` instantiates next: the least-id leaf. -/
def pick (D : Diagram) : Option DBox :=
  (D.boxes.filter fun b => !D.boxes.any (·.parent = some b.id)).foldr (fun b acc => match acc with
    | none => some b
    | some c => if b.id < c.id then some b else some c) none

/-- One box instantiated. -/
def step (D : Diagram) (ks : ℕ → ℕ) (b : DBox) : Diagram :=
  if b.ports = [] then D.expandNode b (ks b.mult) else D.expandPort b (ks b.mult)

theorem expandFuel_succ (ks : ℕ → ℕ) (f : ℕ) (D : Diagram) :
    expandFuel ks (f + 1) D = match D.pick with
      | none => D
      | some b => expandFuel ks f (D.step ks b) := rfl

theorem foldr_pick_mem (l : List DBox) {b : DBox} (h : l.foldr (fun b acc => match acc with
    | none => some b
    | some c => if b.id < c.id then some b else some c) none = some b) : b ∈ l := by
  induction l generalizing b with
  | nil => simp at h
  | cons a l ih =>
    simp only [List.foldr_cons] at h
    split at h
    · cases h; simp
    · next c hc =>
      split_ifs at h
      · cases h; simp
      · cases h; exact List.mem_cons_of_mem _ (ih hc)

theorem foldr_pick_none (l : List DBox) (h : l.foldr (fun b acc => match acc with
    | none => some b
    | some c => if b.id < c.id then some b else some c) none = none) : l = [] := by
  cases l with
  | nil => rfl
  | cons a l =>
    simp only [List.foldr_cons] at h
    split at h <;> [simp at h; (split_ifs at h <;> simp at h)]

theorem pick_some {D : Diagram} {b : DBox} (h : D.pick = some b) :
    b ∈ D.boxes ∧ ∀ y ∈ D.boxes, y.parent ≠ some b.id := by
  have := List.mem_filter.1 (foldr_pick_mem _ h)
  refine ⟨this.1, fun y hy e => ?_⟩
  have h2 := this.2
  simp only [Bool.not_eq_true', List.any_eq_false, decide_eq_true_eq] at h2
  exact h2 y hy e

theorem pick_none {D : Diagram} (hW : WF D) (h : D.pick = none) : D.boxes = [] := by
  by_contra hne
  obtain ⟨b, hb, hleaf⟩ := exists_leaf hW hne
  have := foldr_pick_none _ h
  have hm : b ∈ D.boxes.filter fun b => !D.boxes.any (·.parent = some b.id) := by
    refine List.mem_filter.2 ⟨hb, ?_⟩
    simp only [Bool.not_eq_true', List.any_eq_false, decide_eq_true_eq]
    exact fun y hy => hleaf y hy
  rw [this] at hm
  simp at hm

/-- The slot reading of one box's instantiation. -/
def stepMap (D : Diagram) (ks : ℕ → ℕ) (b : DBox) (σ' : ℕ → List ℕ → ℕ) : ℕ → List ℕ → ℕ :=
  if b.ports = [] then nodeMap D b (ks b.mult) σ' else portMap D b (ks b.mult) σ'

theorem pre_facts (D : Diagram) (b : DBox) (k : ℕ) (hW : WF D) :
    WF (if k = 0 then D.gather b else D) ∧ (if k = 0 then D.gather b else D).boxes = D.boxes := by
  obtain ⟨o, i, e, ho, hi⟩ := gather_eq D b
  split_ifs
  · rw [e]; exact ⟨WF_perm_bnd D o i ho hi hW, rfl⟩
  · exact ⟨hW, rfl⟩

theorem step_facts (D : Diagram) (ks : ℕ → ℕ) (hW : WF D) {b : DBox} (hb : b ∈ D.boxes)
    (hleaf : ∀ y ∈ D.boxes, y.parent ≠ some b.id) (E : Env) :
    (∀ σ', (D.step ks b).evalD E ks σ' = D.evalD E ks (D.stepMap ks b σ')) ∧
      WF (D.step ks b) ∧ (D.step ks b).boxes.length < D.boxes.length := by
  obtain ⟨hW₀, hb₀⟩ := pre_facts D b (ks b.mult) hW
  have hlt : ((D.boxes.filter (·.id ≠ b.id))).length < D.boxes.length :=
    List.length_filter_lt_length_iff_exists.2 ⟨b, hb, by simp⟩
  unfold step stepMap
  split_ifs with hbp
  · refine ⟨fun σ' => evalD_expandNode D b ks hW hb hleaf hbp E σ',
      cn_WF _ hW₀ (hb₀ ▸ hb) (hb₀ ▸ hleaf) hbp, ?_⟩
    show (copyNodes _ b _).boxes.length < _
    rw [cn_boxes, hb₀]
    exact lt_of_le_of_lt ((List.length_filter_le _ _).trans (List.length_map _).le) hlt
  · have hbn : b.nodes = [] := (hW.kinds b hb).resolve_right hbp
    refine ⟨fun σ' => (evalD_expandPort D b ks hW hb hbn hleaf E σ').1,
      (evalD_expandPort D b ks hW hb hbn hleaf E (fun _ _ => 0)).2, ?_⟩
    show (growPorts _ b _).boxes.length < _
    rw [ep_boxes, hb₀]
    exact lt_of_le_of_lt ((List.length_filter_le _ _).trans (List.length_map _).le) hlt

/-- The composite slot reading of `fuel` instantiation steps. -/
def bmapFuel (ks : ℕ → ℕ) : ℕ → Diagram → (ℕ → List ℕ → ℕ) → ℕ → List ℕ → ℕ
  | 0, _, σ' => σ'
  | f + 1, D, σ' => match D.pick with
    | none => σ'
    | some b => D.stepMap ks b (bmapFuel ks f (D.step ks b) σ')

theorem evalD_expandFuel (ks : ℕ → ℕ) (E : Env) :
    ∀ (f : ℕ) (D : Diagram), WF D → D.boxes.length < f → ∀ σ',
      (expandFuel ks f D).evalD E ks σ' = D.evalD E ks (bmapFuel ks f D σ') ∧
        WF (expandFuel ks f D) ∧ (expandFuel ks f D).boxes = [] := by
  intro f
  induction f with
  | zero => intro D _ h; omega
  | succ f ih =>
    intro D hW hf σ'
    rw [expandFuel_succ]
    simp only [bmapFuel]
    cases hp : D.pick with
    | none => exact ⟨rfl, hW, pick_none hW hp⟩
    | some b =>
      obtain ⟨hb, hleaf⟩ := pick_some hp
      obtain ⟨he, hW', hlt⟩ := step_facts D ks hW hb hleaf E
      obtain ⟨h1, h2, h3⟩ := ih (D.step ks b) hW' (by omega) σ'
      exact ⟨h1.trans (he _), h2, h3⟩

/-- The slot reading of the whole expansion. -/
def bmap (D : Diagram) (ks : ℕ → ℕ) (σ' : ℕ → List ℕ → ℕ) : ℕ → List ℕ → ℕ :=
  bmapFuel ks (D.boxes.length + 1) D σ'

/-- The expansion is box-free, keeps the box invariant, and keeps the digit sum. -/
theorem evalD_expand (D : Diagram) (ks : ℕ → ℕ) (E : Env) (hW : WF D) (σ' : ℕ → List ℕ → ℕ) :
    (D.expand ks).evalD E ks σ' = D.evalD E ks (D.bmap ks σ') ∧ WF (D.expand ks) ∧
      (D.expand ks).boxes = [] :=
  evalD_expandFuel ks E _ D hW (by omega) σ'

end Diagram

end ArchytasZX
